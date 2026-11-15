"""Input/output helpers: calibration files, per-video config, frames and PLY export."""

import os

import cv2
import numpy as np
import yaml

# Default algorithm parameters. Any key can be overridden in in/a.yaml or in/b.yaml.
DEFAULTS = {
    # --- frame sampling ---
    'frame_step': 5,          # keep every Nth frame of the video (more baseline, less runtime)
    'max_frames': 80,         # hard cap on the number of frames used
    'max_width': 640,         # frames wider than this are downscaled
    'start_time': 0.0,        # seconds to skip at the start (e.g. before the object starts moving)
    'keyframe_flow': 0.0,     # px; if > 0, only keep a frame once the fastest-moving 10% of
                              # tracked corners moved this much since the last kept frame
    'crop': None,             # [x0, y0, x1, y1] as fractions of the frame; K's principal point is
                              # shifted to match, so the intrinsics stay valid for the crop
    'calib_width': None,      # width of the images K.csv was calibrated on (None = same as video)
    # --- features / tracking ---
    'detector': 'SIFT',       # SIFT | ORB | AKAZE | SURF (SURF needs opencv-contrib nonfree)
    'n_features': 3000,
    'ratio': 0.75,            # Lowe ratio test
    'min_flow': 0.0,          # px; drop matches that barely move (static background in video b)
    'min_track_len': 2,       # minimum number of frames a keypoint must be seen in
    # --- geometry ---
    'ransac_thresh': 1.0,     # px, essential matrix / PnP RANSAC threshold
    'max_reproj': 3.0,        # px, points with larger reprojection error are discarded
    'min_tri_angle': 1.0,     # degrees, minimum triangulation angle
    'tri_window': 5,          # look this many frames back for a wider triangulation baseline
    'init_search': 4,         # try initial pairs (0,1) ... (0,init_search), keep the best
    # --- bundle adjustment ---
    'ba_every': 5,            # run a windowed BA every N registered frames
    'ba_window': 8,           # frames in each windowed BA (>= 5 as the lab asks)
    'global_ba': True,        # final BA over all frames
    'ba_max_nfev': 50,
    # --- dense reconstruction ---
    'dense': True,
    'dense_gap': 1,           # frame gap between the two views of each stereo pair
    'dense_every': 2,         # use every Nth frame as the left view of a pair
    'num_disparities': 64,    # SGBM search range (multiple of 16)
    'block_size': 5,
    'voxel_frac': 0.005,      # voxel size as a fraction of the sparse cloud extent
    'min_voxel_count': 2,     # voxel must be hit by >= N points (from different pairs) to be kept
}


def load_matrix(path):
    """Load a comma (or whitespace) separated numeric matrix, skipping a header row if present."""
    with open(path) as f:
        sample = f.read()
    delim = ',' if ',' in sample else None
    data = np.genfromtxt(path, delimiter=delim, dtype=np.float64)
    data = np.atleast_2d(data)
    data = data[~np.all(np.isnan(data), axis=1)]  # drop header / empty rows
    return data


def load_K(path):
    K = load_matrix(path)
    if K.size != 9:
        raise ValueError(f'{path}: expected a 3x3 matrix, got shape {K.shape}')
    return K.reshape(3, 3)


def load_D(path):
    D = load_matrix(path).ravel()
    return D[~np.isnan(D)]


def load_config(path):
    cfg = dict(DEFAULTS)
    if os.path.exists(path):
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        unknown = set(user) - set(DEFAULTS)
        if unknown:
            print(f'[config] warning: unknown keys in {path}: {sorted(unknown)}')
        cfg.update(user)
    return cfg


def read_frames(path, step, max_frames, max_width, start_time=0.0):
    """Read every `step`-th frame from `start_time` seconds, downscaling to `max_width`.
    Returns (frames, scale)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    start = int(round(start_time * (cap.get(cv2.CAP_PROP_FPS) or 30)))
    for _ in range(start):
        cap.grab()
    frames, scale, idx = [], 1.0, 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / w
                frame = cv2.resize(frame, (max_width, int(round(h * scale))),
                                   interpolation=cv2.INTER_AREA)
            frames.append(frame)
        idx += 1
    cap.release()
    return frames, scale


def crop_frames(frames, K, crop):
    """Crop all frames to the fractional box `crop`; returns (frames, K with shifted cx, cy)."""
    h, w = frames[0].shape[:2]
    x0, y0 = int(round(crop[0] * w)), int(round(crop[1] * h))
    x1, y1 = int(round(crop[2] * w)), int(round(crop[3] * h))
    K = K.copy()
    K[0, 2] -= x0
    K[1, 2] -= y0
    return [f[y0:y1, x0:x1].copy() for f in frames], K


def select_keyframes(frames, min_flow):
    """Drop frames with too little motion since the previous kept frame.

    Motion is the 90th percentile of Lucas-Kanade flow of corners, so a small moving object in
    front of a static background still counts. Returns (kept frames, kept indices).
    """
    kept, idx = [frames[0]], [0]
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for i in range(1, len(frames)):
        cur = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(prev, 1500, 0.01, 5)
        if p0 is None:
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(21, 21), maxLevel=3)
        ok = st.ravel() == 1
        if ok.sum() < 10:
            continue
        flow = np.linalg.norm((p1 - p0).reshape(-1, 2)[ok], axis=1)
        if np.percentile(flow, 90) >= min_flow:
            kept.append(frames[i])
            idx.append(i)
            prev = cur
    return kept, idx


def scale_intrinsics(K, s):
    """Intrinsics for an image resized by factor s (fx, fy, cx, cy all scale)."""
    K = K.copy()
    K[:2, :] *= s
    return K


def write_ply(path, pts, colors=None):
    """Write an ASCII PLY point cloud. `colors` are BGR uint8 (OpenCV order)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(pts)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        if colors is not None:
            f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        if colors is None:
            np.savetxt(f, pts, fmt='%.6f')
        else:
            rgb = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)[:, ::-1]
            for p, c in zip(pts, rgb):
                f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n')
