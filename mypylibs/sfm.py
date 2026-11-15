"""Incremental Structure from Motion over N frames.

1. Initialise from frames 0 and 1: essential matrix -> (R, t), frame 0 is the world origin.
2. Triangulate the shared tracks and measure their reprojection error in both frames.
3. For every following frame: estimate the essential matrix with the previous frame (relative
   motion), then register the frame against the existing 3D points with PnP, which keeps the
   scale consistent across the sequence. Triangulate tracks that do not have a 3D point yet.
4. Every `ba_every` frames run a bundle adjustment over the last `ba_window` (>= 5) frames to
   stop the error from accumulating, and a final global bundle adjustment at the end.
"""

import cv2
import numpy as np

from . import geometry as geo
from .bundle import bundle_adjust, HAVE_SCIPY


class Reconstruction:
    def __init__(self):
        self.poses = {}    # frame -> (R, t), world -> camera
        self.points = {}   # track id -> X (3,)
        self.log = []      # per-frame diagnostics

    def points_array(self):
        ids = np.array(sorted(self.points), int)
        return ids, np.array([self.points[t] for t in ids]).reshape(-1, 3)


def _triangulate_new(recon, tracks, K, f, cfg):
    """Triangulate tracks seen in frame f that have no 3D point yet, pairing f with the
    earliest registered frame (within tri_window) that also sees them, for a wider baseline."""
    added = 0
    pending = [t for t in tracks.frame_obs[f] if t not in recon.points]
    for back in range(cfg['tri_window'], 0, -1):
        g = f - back
        if g < 0 or g not in recon.poses or not pending:
            continue
        ids, pg, pf = tracks.common(g, f, set(pending))
        if len(ids) == 0:
            continue
        X = geo.triangulate(K, recon.poses[g], recon.poses[f], pg, pf)
        ok = geo.valid_points(K, recon.poses[g], recon.poses[f], X, pg, pf,
                              cfg['max_reproj'], cfg['min_tri_angle'])
        for t, x in zip(ids[ok], X[ok]):
            recon.points[t] = x
        added += int(ok.sum())
        done = set(ids[ok].tolist())
        pending = [t for t in pending if t not in done]
    return added


def _frame_error(recon, tracks, K, f):
    obs = tracks.frame_obs[f]
    ids = [t for t in obs if t in recon.points]
    if not ids:
        return float('nan'), 0
    X = np.array([recon.points[t] for t in ids])
    p = np.array([obs[t] for t in ids])
    e = geo.reprojection_error(K, *recon.poses[f], X, p)
    return float(np.mean(e)), len(ids)


def _filter_points(recon, tracks, K, max_err):
    """Remove points whose reprojection error is too large in any registered frame."""
    bad = set()
    for f, t, xy in zip(tracks.obs_frame, tracks.obs_tid, tracks.obs_xy):
        if t in recon.points and f in recon.poses and t not in bad:
            e = geo.reprojection_error(K, *recon.poses[f], recon.points[t][None], xy[None])[0]
            if not np.isfinite(e) or e > max_err:
                bad.add(t)
    for t in bad:
        del recon.points[t]
    return len(bad)


def _try_init(tracks, K, cfg, k):
    """Two-view initialisation from frames 0 and k. Returns (n_valid, R, t, ids, X, errors)."""
    ids, p0, pk = tracks.common(0, k)
    if len(ids) < 8:
        return None
    E, R, t, mask = geo.essential_pose(p0, pk, K, cfg['ransac_thresh'])
    if E is None:
        return None
    pose0, posek = (np.eye(3), np.zeros(3)), (R, t)
    X = geo.triangulate(K, pose0, posek, p0[mask], pk[mask])
    ok = geo.valid_points(K, pose0, posek, X, p0[mask], pk[mask],
                          cfg['max_reproj'], cfg['min_tri_angle'])
    e0 = geo.reprojection_error(K, *pose0, X[ok], p0[mask][ok])
    ek = geo.reprojection_error(K, *posek, X[ok], pk[mask][ok])
    return int(ok.sum()), R, t, ids[mask][ok], X[ok], (e0, ek, int(mask.sum()), len(ids))


def _init_pair(recon, tracks, K, cfg, n_frames):
    """Initialise from the pair (0, k), k = 1..init_search, that yields the most valid points.
    Each pair is tried a few times because RANSAC can land on a degenerate solution."""
    best, best_k = None, None
    for k in range(1, min(cfg['init_search'], n_frames - 1) + 1):
        for _ in range(3):
            r = _try_init(tracks, K, cfg, k)
            if r is not None and (best is None or r[0] > best[0]):
                best, best_k = r, k
    if best is None or best[0] < 10:
        raise RuntimeError('could not initialise: too few valid points (not enough parallax '
                           'or matches between the first frames)')
    n, R, t, ids, X, (e0, ek, n_inl, n_all) = best
    recon.poses[0] = (np.eye(3), np.zeros(3))
    recon.poses[best_k] = (R, t)
    for tid, x in zip(ids, X):
        recon.points[tid] = x
    print(f'  [init] frames 0-{best_k}: {n_inl}/{n_all} E-inliers, {n} points, '
          f'reproj err {e0.mean():.2f}px / {ek.mean():.2f}px')


def run_sfm(tracks, K, cfg, n_frames):
    if not HAVE_SCIPY:
        print('  [sfm] scipy not available: bundle adjustment disabled')
    recon = Reconstruction()
    _init_pair(recon, tracks, K, cfg, n_frames)
    since_ba = 0
    for f in range(1, n_frames):
        if f in recon.poses:
            continue
        prev = max(g for g in recon.poses if g < f)
        if f - prev > cfg['tri_window']:
            break  # tracking was lost; keep what we have
        Rp, tp = recon.poses[prev]

        # Relative motion from the essential matrix (step 3 of the lab)
        ids_e, pa, pb = tracks.common(prev, f)
        E_rel = None
        if len(ids_e) >= 8:
            E_rel, R_rel, t_rel, _ = geo.essential_pose(pa, pb, K, cfg['ransac_thresh'])

        # Absolute pose with PnP on already triangulated points (keeps a consistent scale)
        obs = tracks.frame_obs[f]
        ids = [t for t in obs if t in recon.points]
        pose = None
        if len(ids) >= 6:
            X = np.array([recon.points[t] for t in ids])
            p = np.array([obs[t] for t in ids])
            rvec0 = cv2.Rodrigues(Rp)[0]
            ok, rvec, tvec, inl = cv2.solvePnPRansac(
                X, p, K, None, rvec0.copy(), tp.reshape(3, 1).copy(), useExtrinsicGuess=True,
                iterationsCount=200, reprojectionError=cfg['ransac_thresh'] * 2, confidence=0.999)
            if ok and inl is not None and len(inl) >= 6:
                inl = inl.ravel()
                rvec, tvec = cv2.solvePnPRefineLM(X[inl], p[inl], K, None, rvec, tvec)
                pose = (cv2.Rodrigues(rvec)[0], tvec.ravel())
        if pose is None and E_rel is not None:
            # Fallback: chain the relative motion, reusing the previous baseline length
            if prev - 1 in recon.poses:
                scale = np.linalg.norm(geo.camera_center(*recon.poses[prev]) -
                                       geo.camera_center(*recon.poses[prev - 1]))
            else:
                scale = 1.0
            pose = (R_rel @ Rp, R_rel @ tp + scale * t_rel)
            print(f'  [sfm] frame {f}: PnP failed, chained essential-matrix pose')
        if pose is None:
            print(f'  [sfm] frame {f}: could not register, stopping at {len(recon.poses)} frames')
            break
        recon.poses[f] = pose

        added = _triangulate_new(recon, tracks, K, f, cfg)
        err, n_vis = _frame_error(recon, tracks, K, f)
        recon.log.append({'frame': f, 'pnp_points': len(ids), 'new_points': added,
                          'reproj_err': err})

        since_ba += 1
        if since_ba >= cfg['ba_every']:
            since_ba = 0
            window = list(range(max(0, f - cfg['ba_window'] + 1), f + 1))
            r = bundle_adjust(recon, tracks, K, window, {0, window[0]},
                              cfg['max_reproj'], cfg['ba_max_nfev'])
            if r:
                print(f'  [ba] frames {window[0]}-{f}: rmse {r[0]:.2f} -> {r[1]:.2f}px')

    if cfg['global_ba'] and len(recon.poses) > 2:
        r = bundle_adjust(recon, tracks, K, sorted(recon.poses), {0},
                          cfg['max_reproj'], cfg['ba_max_nfev'] * 2)
        if r:
            print(f'  [ba] global over {len(recon.poses)} frames: rmse {r[0]:.2f} -> {r[1]:.2f}px')
    removed = _filter_points(recon, tracks, K, cfg['max_reproj'] * 2)
    errs = [_frame_error(recon, tracks, K, f)[0] for f in sorted(recon.poses)]
    recon.mean_error = float(np.nanmean(errs)) if errs else float('nan')
    print(f'  [sfm] {len(recon.poses)}/{n_frames} frames registered, {len(recon.points)} points '
          f'({removed} outliers removed), mean reproj err {recon.mean_error:.2f}px')
    return recon
