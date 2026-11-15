"""Dense reconstruction from the estimated camera poses.

For pairs of registered frames, rectify them with the known relative pose, compute a dense
disparity map with semi-global block matching, back-project every valid pixel to 3D and move it
to the world frame. Points from all pairs are then averaged on a voxel grid: each voxel's
position and colour is the mean of the points that fell into it, and voxels hit by fewer than
`min_voxel_count` points are dropped as unreliable.
"""

import cv2
import numpy as np

from . import geometry as geo


def _pair_points(img1, img2, K, pose1, pose2, cfg, depth_range):
    h, w = img1.shape[:2]
    R1, t1 = pose1
    R2, t2 = pose2
    R = R2 @ R1.T
    t = t2 - R @ t1
    D0 = np.zeros(5)
    Rr1, Rr2, P1, P2, Q, _, _ = cv2.stereoRectify(K, D0, K, D0, (w, h), R, t.reshape(3, 1),
                                                  flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    if abs(P2[1, 3]) > abs(P2[0, 3]):
        return None  # vertical stereo (mostly up/down motion): skip this pair
    m1 = cv2.initUndistortRectifyMap(K, D0, Rr1, P1, (w, h), cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(K, D0, Rr2, P2, (w, h), cv2.CV_32FC1)
    r1 = cv2.remap(img1, *m1, cv2.INTER_LINEAR)
    r2 = cv2.remap(img2, *m2, cv2.INTER_LINEAR)

    flip = P2[0, 3] > 0  # second camera is on the left: mirror so disparities are positive
    g1 = cv2.cvtColor(r1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(r2, cv2.COLOR_BGR2GRAY)
    if flip:
        g1, g2 = cv2.flip(g1, 1), cv2.flip(g2, 1)
    bs = cfg['block_size']
    sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=cfg['num_disparities'],
                                 blockSize=bs, P1=8 * bs * bs, P2=32 * bs * bs,
                                 uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                                 disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(g1, g2).astype(np.float32) / 16.0
    if flip:
        disp = cv2.flip(disp, 1)
        Q = Q.copy()
        Q[3, 2] = -Q[3, 2]  # disparity sign convention reversed
    valid = disp > 0.5
    pts = cv2.reprojectImageTo3D(disp, Q)[valid]          # rectified camera-1 frame
    cols = r1[valid]
    Xc1 = pts @ Rr1                                       # back to (unrectified) camera 1
    z = Xc1[:, 2]
    keep = np.isfinite(Xc1).all(1) & (z > depth_range[0]) & (z < depth_range[1])
    Xw = (Xc1[keep] - t1) @ R1                            # camera 1 -> world
    return Xw, cols[keep]


def voxel_average(X, C, voxel, min_count):
    keys = np.floor(X / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    n = len(counts)
    Xs = np.zeros((n, 3))
    Cs = np.zeros((n, 3))
    np.add.at(Xs, inv, X)
    np.add.at(Cs, inv, C.astype(np.float64))
    keep = counts >= min_count
    return Xs[keep] / counts[keep, None], (Cs[keep] / counts[keep, None]).astype(np.uint8)


def dense_reconstruction(frames, recon, K, cfg):
    _, sparse = recon.points_array()
    if len(sparse) < 10:
        return None, None
    # Depth range of the object seen by the cameras, from the sparse points (with a margin)
    depths = []
    for R, t in recon.poses.values():
        z = sparse @ R[2] + t[2]
        depths.append(z[z > 0])
    depths = np.concatenate(depths)
    depth_range = (np.percentile(depths, 2) * 0.7, np.percentile(depths, 98) * 1.4)

    all_X, all_C = [], []
    regs = sorted(recon.poses)
    used = 0
    for f in regs[::cfg['dense_every']]:
        g = f + cfg['dense_gap']
        if g not in recon.poses:
            continue
        res = _pair_points(frames[f], frames[g], K, recon.poses[f], recon.poses[g], cfg,
                           depth_range)
        if res is None or len(res[0]) == 0:
            continue
        all_X.append(res[0])
        all_C.append(res[1])
        used += 1
    if not all_X:
        print('  [dense] no usable stereo pairs')
        return None, None
    X, C = np.concatenate(all_X), np.concatenate(all_C)
    extent = np.linalg.norm(np.percentile(sparse, 95, 0) - np.percentile(sparse, 5, 0))
    voxel = max(extent * cfg['voxel_frac'], 1e-9)
    Xv, Cv = voxel_average(X, C, voxel, cfg['min_voxel_count'])
    print(f'  [dense] {used} pairs, {len(X)} raw points -> {len(Xv)} voxels')
    return Xv, Cv
