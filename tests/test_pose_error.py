"""Pose errors against ground truth (eval/pose_error.py)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from pose_error import kitti_ground_truth, trajectory_errors, umeyama


def random_rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def drive(n=30):
    """World-from-camera poses of a camera driving forward along z and turning slowly."""
    T = np.tile(np.eye(4), (n, 1, 1))
    for k in range(n):
        a = 0.02 * k
        T[k, :3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
        T[k, :3, 3] = [3 * np.sin(a), 0.1 * np.sin(k), 1.5 * k]
    return T


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(40, 3))
    R = random_rotation(rng)
    s, R_est, t = umeyama(src, 2.5 * src @ R.T + [1.0, -2.0, 3.0])
    assert np.isclose(s, 2.5) and np.allclose(R_est, R) and np.allclose(t, [1.0, -2.0, 3.0])


def test_scaled_rotated_copy_of_the_truth_has_no_error():
    G = drive()
    # Estimate = truth in another similarity frame: x_est = s R x_gt + t.
    rng = np.random.default_rng(1)
    s, R, t = 0.13, random_rotation(rng), np.array([5.0, -1.0, 2.0])
    E = G.copy()
    E[:, :3, :3] = R @ G[:, :3, :3]
    E[:, :3, 3] = s * G[:, :3, 3] @ R.T + t
    out = trajectory_errors(np.arange(len(G)), E, G)
    assert np.isclose(out["sim3_scale"], 1 / s)
    for k in ("ate_rmse_m", "rpe1_trans_rmse_m", "rpe10_trans_rmse_m"):
        assert out[k] < 1e-9, (k, out[k])
    assert out["rpe1_rot_mean_deg"] < 1e-5 and out["rpe10_rot_mean_deg"] < 1e-5


def test_relative_errors_see_a_perturbed_step():
    G = drive()
    E = G.copy()
    E[15:, :3, 3] += [0.0, 0.0, 0.5]  # one step 0.5 m too long
    out = trajectory_errors(np.arange(len(G)), E, G)
    assert out["ate_rmse_m"] > 0.1
    assert 0.05 < out["rpe1_trans_rmse_m"] < 0.2  # 1 of 29 steps is wrong


def test_kitti_colour_camera_offset(tmp_path):
    # Camera 0 at the origin looking along z; P2's rectified baseline puts camera 2
    # 6 cm to the left of camera 0 (KITTI: P2[0, 3] = f * 0.06).
    f = 718.856
    (tmp_path / "00.txt").write_text(" ".join(map(str, np.eye(4)[:3].ravel())) + "\n")
    P2 = [f, 0, 607.19, f * 0.06, 0, f, 185.2, 0, 0, 0, 1, 0]
    (tmp_path / "calib.txt").write_text("P2: " + " ".join(map(str, P2)) + "\n")
    T = kitti_ground_truth(tmp_path)
    assert T.shape == (1, 4, 4) and np.allclose(T[0, :3, 3], [-0.06, 0, 0])
