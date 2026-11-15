"""Keypoint detection, matching between consecutive frames, and track building."""

import cv2
import numpy as np


def make_detector(name, n_features):
    name = name.upper()
    if name == 'SIFT':
        return cv2.SIFT_create(nfeatures=n_features), cv2.NORM_L2
    if name == 'ORB':
        return cv2.ORB_create(nfeatures=n_features), cv2.NORM_HAMMING
    if name == 'AKAZE':
        return cv2.AKAZE_create(), cv2.NORM_HAMMING
    if name == 'SURF':
        try:
            return cv2.xfeatures2d.SURF_create(400), cv2.NORM_L2
        except (AttributeError, cv2.error):
            print('[features] SURF not available in this OpenCV build, falling back to SIFT')
            return cv2.SIFT_create(nfeatures=n_features), cv2.NORM_L2
    raise ValueError(f'unknown detector {name}')


def detect(frames, detector):
    """Returns a list of (pts (N,2) float32, descriptors) per frame."""
    out = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kps, desc = detector.detectAndCompute(gray, None)
        pts = np.float32([k.pt for k in kps]).reshape(-1, 2)
        out.append((pts, desc))
    return out


def match_pair(f1, f2, norm, ratio, min_flow, ransac_thresh):
    """Mutual ratio-test matches between two frames, pruned with a fundamental-matrix RANSAC.

    Returns an (M,2) array of index pairs (i in frame 1, j in frame 2).
    """
    (p1, d1), (p2, d2) = f1, f2
    if d1 is None or d2 is None or len(p1) < 8 or len(p2) < 8:
        return np.empty((0, 2), int)
    bf = cv2.BFMatcher(norm)

    def ratio_matches(da, db):
        res = {}
        for m in bf.knnMatch(da, db, k=2):
            if len(m) == 2 and m[0].distance < ratio * m[1].distance:
                res[m[0].queryIdx] = m[0].trainIdx
        return res

    fwd, bwd = ratio_matches(d1, d2), ratio_matches(d2, d1)
    pairs = np.array([(i, j) for i, j in fwd.items() if bwd.get(j) == i], int).reshape(-1, 2)
    if len(pairs) < 8:
        return np.empty((0, 2), int)

    if min_flow > 0:  # drop (near) static matches, e.g. background when the object spins
        flow = np.linalg.norm(p1[pairs[:, 0]] - p2[pairs[:, 1]], axis=1)
        pairs = pairs[flow >= min_flow]
        if len(pairs) < 8:
            return np.empty((0, 2), int)

    _, mask = cv2.findFundamentalMat(p1[pairs[:, 0]], p2[pairs[:, 1]], cv2.FM_RANSAC,
                                     ransac_thresh, 0.999)
    if mask is None:
        return np.empty((0, 2), int)
    return pairs[mask.ravel().astype(bool)]


class Tracks:
    """Keypoint tracks across frames.

    obs_frame, obs_tid, obs_xy: one row per observation (frame index, track id, pixel).
    frame_obs[f]: dict track id -> pixel (x, y) for frame f.
    """

    def __init__(self, obs_frame, obs_tid, obs_xy, n_frames):
        self.obs_frame = obs_frame
        self.obs_tid = obs_tid
        self.obs_xy = obs_xy
        self.frame_obs = [dict() for _ in range(n_frames)]
        for f, t, xy in zip(obs_frame, obs_tid, obs_xy):
            self.frame_obs[f][t] = xy

    def common(self, f1, f2, tids=None):
        """Track ids visible in both frames, with their pixels in each."""
        a, b = self.frame_obs[f1], self.frame_obs[f2]
        ids = [t for t in a if t in b and (tids is None or t in tids)]
        p1 = np.float64([a[t] for t in ids]).reshape(-1, 2)
        p2 = np.float64([b[t] for t in ids]).reshape(-1, 2)
        return np.array(ids, int), p1, p2


def build_tracks(frames, cfg):
    """Detect, match consecutive frames and chain matches into tracks."""
    detector, norm = make_detector(cfg['detector'], cfg['n_features'])
    feats = detect(frames, detector)
    n = len(frames)
    track_of = [np.full(len(p), -1, int) for p, _ in feats]
    next_id = 0
    n_matches = []
    for f in range(n - 1):
        pairs = match_pair(feats[f], feats[f + 1], norm, cfg['ratio'], cfg['min_flow'],
                           cfg['ransac_thresh'])
        n_matches.append(len(pairs))
        for i, j in pairs:
            tid = track_of[f][i]
            if tid < 0:
                tid = next_id
                next_id += 1
                track_of[f][i] = tid
            track_of[f + 1][j] = tid

    obs_f, obs_t, obs_xy = [], [], []
    for f in range(n):
        sel = track_of[f] >= 0
        obs_f.append(np.full(sel.sum(), f, int))
        obs_t.append(track_of[f][sel])
        obs_xy.append(feats[f][0][sel])
    obs_f, obs_t = np.concatenate(obs_f), np.concatenate(obs_t)
    obs_xy = np.concatenate(obs_xy).astype(np.float64)

    # Accuracy / density trade-off: keep only tracks seen in >= min_track_len frames
    lengths = np.bincount(obs_t, minlength=next_id)
    keep = lengths[obs_t] >= cfg['min_track_len']
    tracks = Tracks(obs_f[keep], obs_t[keep], obs_xy[keep], n)

    kept_len = lengths[lengths >= cfg['min_track_len']]
    print(f'  [tracks] {cfg["detector"]}: mean matches/pair {np.mean(n_matches):.0f}, '
          f'{len(kept_len)} tracks, mean length {kept_len.mean() if len(kept_len) else 0:.1f}')
    return tracks
