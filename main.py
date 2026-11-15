"""CS6524 Lab 3 - Structure from Motion.

Runs the pipeline on in/a.mp4 (camera moves around the object) and in/b.mp4 (object spins in
front of a static camera) for the sensitivity analysis: focal length x {0.8, 1.0, 1.2}, with and
without undistortion. Writes out/[a,b]_[08,10,12][u,d].ply (dense clouds), the sparse clouds in
out/sparse/ and a summary of every run in out/metrics.csv.
"""

import csv
import time
import traceback
from os import path as opath

import cv2
import numpy as np

from mypylibs.io_utils import (load_K, load_D, load_config, read_frames, crop_frames, select_keyframes,
                               scale_intrinsics, write_ply)
from mypylibs.features import build_tracks
from mypylibs.sfm import run_sfm
from mypylibs.dense import dense_reconstruction

ROOT_IN = 'in'
ROOT_OUT = 'out'
CASES = ['a', 'b']
FOCAL_FACTORS = [0.8, 1.0, 1.2]


def run_one(name, frames, K, D, undistort, cfg):
    if undistort:
        # cv2.undistort keeps the same camera matrix, so K stays valid for the undistorted frames
        frames = [cv2.undistort(fr, K, D) for fr in frames]
    tracks = build_tracks(frames, cfg)
    recon = run_sfm(tracks, K, cfg, len(frames))
    ids, sparse = recon.points_array()
    sparse_cols = np.array([_point_colour(frames, tracks, t) for t in ids]).reshape(-1, 3)
    write_ply(opath.join(ROOT_OUT, 'sparse', f'{name}.ply'), sparse, sparse_cols)

    dense, dense_cols = (None, None)
    if cfg['dense']:
        dense, dense_cols = dense_reconstruction(frames, recon, K, cfg)
    if dense is None:  # fall back to the sparse cloud so the output always exists
        dense, dense_cols = sparse, sparse_cols
    write_ply(opath.join(ROOT_OUT, f'{name}.ply'), dense, dense_cols)
    return {'frames_registered': len(recon.poses), 'sparse_points': len(sparse),
            'dense_points': len(dense), 'mean_reproj_err_px': round(recon.mean_error, 3)}


def _point_colour(frames, tracks, tid):
    """Colour of a track at its first observation."""
    i = np.flatnonzero(tracks.obs_tid == tid)[0]
    x, y = tracks.obs_xy[i]
    f = tracks.obs_frame[i]
    h, w = frames[f].shape[:2]
    return frames[f][min(int(round(y)), h - 1), min(int(round(x)), w - 1)]


def main():
    K_calib = load_K(opath.join(ROOT_IN, 'K.csv'))
    D = load_D(opath.join(ROOT_IN, 'D.csv'))
    rows = []
    for case in CASES:
        video = opath.join(ROOT_IN, f'{case}.mp4')
        if not opath.exists(video):
            print(f'[{case}] {video} not found, skipping')
            continue
        cfg = load_config(opath.join(ROOT_IN, f'{case}.yaml'))
        frames, s = read_frames(video, cfg['frame_step'], cfg['max_frames'], cfg['max_width'],
                                cfg['start_time'])
        h, w = frames[0].shape[:2]
        # K must match the resolution of the frames we process
        calib_w = cfg['calib_width'] or w / s
        K_case = scale_intrinsics(K_calib, w / calib_w)
        if cfg['crop']:
            frames, K_case = crop_frames(frames, K_case, cfg['crop'])
            h, w = frames[0].shape[:2]
        if cfg['keyframe_flow'] > 0:
            frames, _ = select_keyframes(frames, cfg['keyframe_flow'])
        print(f'[{case}] {len(frames)} frames at {w}x{h}, fx={K_case[0, 0]:.1f}')

        for undistort in (True, False):
            for factor in FOCAL_FACTORS:
                name = f'{case}_{int(round(factor * 10)):02d}{"u" if undistort else "d"}'
                K = K_case.copy()
                K[0, 0] *= factor
                K[1, 1] *= factor
                print(f'--- {name} (f x{factor}, {"undistorted" if undistort else "distorted"})')
                t0 = time.time()
                try:
                    stats = run_one(name, frames, K, D, undistort, cfg)
                except Exception as e:  # keep going so the other runs still produce output
                    traceback.print_exc()
                    write_ply(opath.join(ROOT_OUT, f'{name}.ply'), np.empty((0, 3)))
                    stats = {'error': str(e)}
                stats.update({'run': name, 'seconds': round(time.time() - t0, 1)})
                print(f'    {stats}')
                rows.append(stats)

    if rows:
        keys = ['run', 'frames_registered', 'sparse_points', 'dense_points',
                'mean_reproj_err_px', 'seconds', 'error']
        with open(opath.join(ROOT_OUT, 'metrics.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)


if __name__ == '__main__':
    main()
