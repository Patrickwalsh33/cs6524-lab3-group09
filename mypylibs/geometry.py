"""Two-view geometry: essential matrix, relative pose, triangulation and reprojection error.

Convention: a pose (R, t) maps world to camera coordinates, x_cam = R @ X + t.
"""

import cv2
import numpy as np


def essential_pose(p1, p2, K, thresh):
    """Essential matrix between two views and the relative pose of view 2 w.r.t. view 1.

    Returns (E, R, t, inlier_mask). t has unit norm (scale is unobservable from two views).
    """
    method = getattr(cv2, 'USAC_MAGSAC', cv2.RANSAC)  # MAGSAC is more robust to near-planar scenes
    E, mask = cv2.findEssentialMat(p1, p2, K, method, 0.999, thresh)
    if E is None:
        return None, None, None, None
    if E.shape[0] > 3:  # several solutions returned, keep the first
        E = E[:3]
    _, R, t, mask = cv2.recoverPose(E, p1, p2, K, mask=mask)
    return E, R, t.ravel(), mask.ravel().astype(bool)


def projection_matrix(K, R, t):
    return K @ np.hstack([R, np.reshape(t, (3, 1))])


def triangulate(K, pose1, pose2, p1, p2):
    """Linear (DLT) triangulation of matching pixels. Returns (N,3) world points."""
    if len(p1) == 0:
        return np.empty((0, 3))
    P1 = projection_matrix(K, *pose1)
    P2 = projection_matrix(K, *pose2)
    Xh = cv2.triangulatePoints(P1, P2, p1.T.astype(np.float64), p2.T.astype(np.float64))
    return (Xh[:3] / Xh[3]).T


def project(K, R, t, X):
    Xc = X @ R.T + np.reshape(t, (1, 3))
    x = Xc @ K.T
    return x[:, :2] / x[:, 2:3], Xc[:, 2]


def reprojection_error(K, R, t, X, pts):
    proj, _ = project(K, R, t, X)
    return np.linalg.norm(proj - pts, axis=1)


def camera_center(R, t):
    return -R.T @ np.reshape(t, 3)


def valid_points(K, pose1, pose2, X, p1, p2, max_reproj, min_angle_deg):
    """Mask of triangulated points that are in front of both cameras, reproject within
    max_reproj px in both views and have a triangulation angle above min_angle_deg."""
    e1 = reprojection_error(K, *pose1, X, p1)
    e2 = reprojection_error(K, *pose2, X, p2)
    _, z1 = project(K, *pose1, X)
    _, z2 = project(K, *pose2, X)
    c1, c2 = camera_center(*pose1), camera_center(*pose2)
    r1, r2 = X - c1, X - c2
    cosang = np.sum(r1 * r2, axis=1) / (np.linalg.norm(r1, axis=1) * np.linalg.norm(r2, axis=1) + 1e-12)
    angle = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    return (z1 > 0) & (z2 > 0) & (e1 < max_reproj) & (e2 < max_reproj) & (angle > min_angle_deg) \
        & np.all(np.isfinite(X), axis=1)
