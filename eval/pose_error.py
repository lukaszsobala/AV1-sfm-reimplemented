"""Camera pose accuracy of a reconstruction against KITTI odometry ground truth.

Reconstructions from images alone have an arbitrary scale, so the estimated
camera centres are first aligned to the ground truth with a similarity
transform (Umeyama's least-squares Sim(3)). Reported, as in the usual
odometry / SfM evaluations:

  ate_rmse_m         absolute trajectory error: RMSE of the aligned camera
                     centres against the ground truth (metres)
  rpe{1,10}_*        relative pose error over 1 and 10 frames: translation
                     error of the scaled relative motion (metres) and its
                     rotation error (degrees); independent of the alignment's
                     rotation, which a nearly straight drive leaves poorly
                     determined about the direction of travel

Only registered images enter the errors; `registered` counts them.
The ground truth is the pose of KITTI's camera 0 (`poses/<seq>.txt`); the
centre of the colour camera 2 is offset from it by the rectified baseline in
`calib.txt` (P2), which is applied here. Both files are fetched by
eval/fetch_kitti.py.

    uv run python eval/pose_error.py runs/kitti117/rec_mv data/kitti/00 --stats pose_mv.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pycolmap

from av1sfm.export import pick_model


def kitti_ground_truth(seq_dir: str | Path, camera: str = "P2") -> np.ndarray:
    """(N, 4, 4) world-from-camera poses of the given rectified camera."""
    seq_dir = Path(seq_dir)
    poses = np.loadtxt(next(seq_dir.glob("[0-9][0-9].txt"))).reshape(-1, 3, 4)
    calib = {
        k.strip(): np.array(v.split(), float)
        for k, v in (ln.split(":", 1) for ln in (seq_dir / "calib.txt").read_text().splitlines())
    }
    P = calib[camera].reshape(3, 4)
    offset = np.linalg.solve(P[:, :3], P[:, 3])  # x_cam = x_cam0 + offset (rectified)
    T = np.tile(np.eye(4), (len(poses), 1, 1))
    T[:, :3, :] = poses
    cam0_from_cam = np.eye(4)
    cam0_from_cam[:3, 3] = -offset
    return T @ cam0_from_cam


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity (s, R, t) minimising sum |dst - (s R src + t)|^2."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / a.var(0).sum())
    return s, R, mu_d - s * R @ mu_s


def rotation_angle_deg(R: np.ndarray) -> np.ndarray:
    """Rotation angle of each (..., 3, 3) matrix in degrees."""
    c = (np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def trajectory_errors(frames: np.ndarray, E: np.ndarray, G: np.ndarray) -> dict:
    """Errors of estimated world-from-camera poses `E` against ground truth `G` ((N, 4, 4)).

    `frames` are the frame indices of the N poses; relative errors use the
    pairs of frames 1 and 10 apart that are both present.
    """
    s, R, t = umeyama(E[:, :3, 3], G[:, :3, 3])
    aligned = np.tile(np.eye(4), (len(frames), 1, 1))
    aligned[:, :3, :3] = R @ E[:, :3, :3]
    aligned[:, :3, 3] = s * E[:, :3, 3] @ R.T + t
    ate = np.linalg.norm(aligned[:, :3, 3] - G[:, :3, 3], axis=1)
    out = {
        "registered": len(frames),
        "sim3_scale": s,
        "gt_path_length_m": float(np.linalg.norm(np.diff(G[:, :3, 3], axis=0), axis=1).sum()),
        "ate_rmse_m": float(np.sqrt(np.mean(ate**2))),
        "ate_median_m": float(np.median(ate)),
        "ate_max_m": float(ate.max()),
    }
    index = {int(f): i for i, f in enumerate(frames)}
    for d in (1, 10):
        pairs = [(index[f], index[f + d]) for f in index if f + d in index]
        if not pairs:
            continue
        i, j = np.array(pairs).T
        rel_e = np.linalg.inv(aligned[i]) @ aligned[j]
        rel_g = np.linalg.inv(G[i]) @ G[j]
        err = np.linalg.inv(rel_g) @ rel_e
        out[f"rpe{d}_trans_rmse_m"] = float(np.sqrt(np.mean(np.sum(err[:, :3, 3] ** 2, 1))))
        out[f"rpe{d}_rot_mean_deg"] = float(rotation_angle_deg(err[:, :3, :3]).mean())
    return out


def evaluate(model: str | Path, seq_dir: str | Path) -> dict:
    """Pose errors of the model's registered images (named by KITTI frame index)."""
    rec = pycolmap.Reconstruction(model)
    est = {}
    for im in rec.images.values():
        if im.has_pose:
            T = np.eye(4)
            T[:3, :3] = im.cam_from_world().rotation.matrix().T
            T[:3, 3] = im.projection_center()
            est[int(Path(im.name).stem)] = T  # world-from-camera
    frames = np.array(sorted(est))
    if len(frames) < 3:
        return {"model": str(model), "registered": len(frames)}
    gt = kitti_ground_truth(seq_dir)
    E = np.stack([est[f] for f in frames])
    return {"model": str(model), **trajectory_errors(frames, E, gt[frames])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("rec", type=Path, help="model folder or mapper output folder")
    ap.add_argument("sequence", type=Path, help="KITTI sequence folder with NN.txt and calib.txt")
    ap.add_argument("--stats", type=Path, default=None)
    a = ap.parse_args()
    out = evaluate(pick_model(a.rec), a.sequence)
    text = json.dumps(out, indent=2)
    if a.stats:
        a.stats.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
