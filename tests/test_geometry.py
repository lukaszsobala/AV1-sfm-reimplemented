import numpy as np
import pycolmap

from av1sfm.geometry import (
    RansacSettings,
    sampson_epipolar,
    sampson_homography,
    score_pair,
    summarize,
)


def skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def test_sampson_epipolar_closed_form():
    E = skew([1.0, 0.0, 0.0])  # pure x translation: epipolar lines are horizontal
    rng = np.random.default_rng(0)
    x1 = rng.uniform(-1, 1, (50, 2))
    d = rng.uniform(-0.01, 0.01, 50)
    x2 = x1 + np.stack([rng.uniform(-0.3, 0.3, 50), d], axis=1)
    np.testing.assert_allclose(sampson_epipolar(E, x1, x2), np.abs(d) / np.sqrt(2), rtol=1e-9)
    assert np.allclose(sampson_epipolar(E, x1, x1 + [[0.2, 0.0]]), 0.0)


def test_sampson_homography_closed_form():
    rng = np.random.default_rng(1)
    x1 = rng.uniform(-1, 1, (50, 2))
    d = rng.normal(scale=0.003, size=(50, 2))
    s = sampson_homography(np.eye(3), x1, x1 + d)
    np.testing.assert_allclose(s, np.linalg.norm(d, axis=1) / np.sqrt(2), rtol=1e-6)
    H = np.array([[1.1, 0.05, 0.1], [-0.02, 0.95, -0.05], [0.01, 0.02, 1.0]])
    X2 = np.hstack([x1, np.ones((50, 1))]) @ H.T
    assert np.allclose(sampson_homography(H, x1, X2[:, :2] / X2[:, 2:]), 0.0, atol=1e-12)


def scene(n=400, outlier_frac=0.2, planar=False, seed=0):
    rng = np.random.default_rng(seed)
    f, w, h = 600.0, 800, 600
    cam = pycolmap.Camera(model="SIMPLE_PINHOLE", width=w, height=h, params=[f, w / 2, h / 2])
    Z = np.full(n, 6.0) if planar else rng.uniform(4, 10, n)
    X = np.c_[rng.uniform(-2, 2, n), rng.uniform(-1.5, 1.5, n), Z]
    ang = 0.05
    R = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
    t = np.array([0.5, 0.05, 0.1])
    proj = lambda P: P[:, :2] / P[:, 2:] * f + [w / 2, h / 2]
    p1 = proj(X) + rng.normal(scale=0.3, size=(n, 2))
    p2 = proj(X @ R.T + t) + rng.normal(scale=0.3, size=(n, 2))
    k = int(outlier_frac * n)
    p2[:k] = rng.uniform([0, 0], [w, h], (k, 2))
    return cam, p1, p2


def test_score_pair_general_scene_prefers_essential():
    cam, p1, p2 = scene()
    s = score_pair(cam, cam, p1, p2, RansacSettings())
    assert s.model == "E"
    assert 0.78 <= s.inlier_ratio <= 0.82  # 20 % outliers (a few may fall on epipolar lines)
    assert 0.1 < s.median_sampson_px < 0.5  # 0.3 px noise per coordinate


def test_score_pair_planar_scene_both_fit():
    cam, p1, p2 = scene(planar=True, outlier_frac=0.1)
    s = score_pair(cam, cam, p1, p2, RansacSettings())
    assert s.model in ("E", "H") and s.inlier_ratio > 0.85


def test_score_pair_degenerate_inputs():
    cam, p1, p2 = scene(n=4)
    assert score_pair(cam, cam, p1, p2).model == "none"
    agg = summarize([score_pair(cam, cam, *scene()[1:])])
    assert agg["pairs"] == 1 and agg["frac_E"] == 1.0


def test_repeats_and_squared_sampson():
    cam, p1, p2 = scene(seed=3)
    one = score_pair(cam, cam, p1, p2, RansacSettings(repeats=1))
    rep = score_pair(cam, cam, p1, p2, RansacSettings(repeats=5))
    assert rep.model == "E" and abs(rep.inlier_ratio - one.inlier_ratio) < 0.02
    # the paper's SE is the squared Sampson distance (median of squares = square of median)
    assert np.isclose(rep.median_sampson_sq_norm, rep.median_sampson_norm**2, rtol=0.05)
