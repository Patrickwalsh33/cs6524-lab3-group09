"""Bundle adjustment: jointly refine camera poses and 3D points by minimising reprojection error.

Uses scipy.optimize.least_squares with a sparse Jacobian. If scipy is not installed the
functions return without changing anything (the pipeline then relies on PnP alone).
"""

import cv2
import numpy as np

try:
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False


def _rotate(points, rvecs):
    """Rotate points by axis-angle vectors (Rodrigues formula, vectorised)."""
    theta = np.linalg.norm(rvecs, axis=1)[:, None]
    with np.errstate(invalid='ignore', divide='ignore'):
        v = np.where(theta > 0, rvecs / theta, 0.0)
    dot = np.sum(points * v, axis=1)[:, None]
    cos, sin = np.cos(theta), np.sin(theta)
    return cos * points + sin * np.cross(v, points) + dot * (1 - cos) * v


def bundle_adjust(recon, tracks, K, frame_ids, fixed_ids, loss_scale, max_nfev):
    """Refine the poses of `frame_ids` (except `fixed_ids`) and every point they observe.

    recon.poses: dict frame -> (R, t); recon.points: dict track id -> X (3,)
    Returns (rmse_before, rmse_after) in pixels, or None if nothing was done.
    """
    if not HAVE_SCIPY:
        return None
    frame_ids = [f for f in frame_ids if f in recon.poses]
    fset = set(frame_ids)
    sel = np.array([(f in fset) and (t in recon.points)
                    for f, t in zip(tracks.obs_frame, tracks.obs_tid)], bool)
    if sel.sum() < 20:
        return None
    o_f, o_t, o_xy = tracks.obs_frame[sel], tracks.obs_tid[sel], tracks.obs_xy[sel]

    free = [f for f in frame_ids if f not in fixed_ids]
    if not free:
        return None
    cam_idx = {f: i for i, f in enumerate(frame_ids)}
    free_idx = {f: i for i, f in enumerate(free)}
    pt_ids = np.unique(o_t)
    pt_idx = {t: i for i, t in enumerate(pt_ids)}

    rv = np.zeros((len(frame_ids), 3))
    tv = np.zeros((len(frame_ids), 3))
    for f, i in cam_idx.items():
        R, t = recon.poses[f]
        rv[i] = cv2.Rodrigues(R)[0].ravel()
        tv[i] = t
    X0 = np.array([recon.points[t] for t in pt_ids])

    oc = np.array([cam_idx[f] for f in o_f])
    op = np.array([pt_idx[t] for t in o_t])
    is_free = np.array([f in free_idx for f in o_f])
    of = np.array([free_idx.get(f, -1) for f in o_f])
    n_free = len(free)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def unpack(x):
        cams = x[:6 * n_free].reshape(n_free, 6)
        pts = x[6 * n_free:].reshape(-1, 3)
        r, t = rv.copy(), tv.copy()
        for f, i in free_idx.items():
            r[cam_idx[f]], t[cam_idx[f]] = cams[i, :3], cams[i, 3:]
        return r, t, pts

    def residuals(x):
        r, t, pts = unpack(x)
        Xc = _rotate(pts[op], r[oc]) + t[oc]
        z = Xc[:, 2]
        z = np.where(np.abs(z) < 1e-6, 1e-6, z)
        u = fx * Xc[:, 0] / z + cx
        v = fy * Xc[:, 1] / z + cy
        return np.concatenate([u - o_xy[:, 0], v - o_xy[:, 1]])

    x0 = np.concatenate([np.hstack([rv[[cam_idx[f] for f in free]],
                                    tv[[cam_idx[f] for f in free]]]).ravel(), X0.ravel()])

    m, n_obs = 2 * len(o_f), len(o_f)
    A = lil_matrix((m, len(x0)), dtype=int)
    rows = np.arange(n_obs)
    for k in range(6):
        A[rows[is_free], 6 * of[is_free] + k] = 1
        A[n_obs + rows[is_free], 6 * of[is_free] + k] = 1
    for k in range(3):
        A[rows, 6 * n_free + 3 * op + k] = 1
        A[n_obs + rows, 6 * n_free + 3 * op + k] = 1

    r0 = residuals(x0)
    res = least_squares(residuals, x0, jac_sparsity=A, x_scale='jac', loss='huber',
                        f_scale=loss_scale, method='trf', max_nfev=max_nfev)
    r, t, pts = unpack(res.x)
    for f, i in free_idx.items():
        recon.poses[f] = (cv2.Rodrigues(r[cam_idx[f]])[0], t[cam_idx[f]].copy())
    for tid, i in pt_idx.items():
        recon.points[tid] = pts[i]

    def rmse(rr):
        return float(np.sqrt(np.mean(rr[:n_obs] ** 2 + rr[n_obs:] ** 2)))
    return rmse(r0), rmse(res.fun)
