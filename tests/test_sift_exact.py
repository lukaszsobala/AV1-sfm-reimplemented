"""Exact SIFT matcher against a literal port of COLMAP's brute-force loop."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from av1sfm.sift_exact import match_descriptors


def colmap_one_way(dots: np.ndarray, max_ratio: float, max_distance: float) -> list[int]:
    """FindBestMatchesOneWayBruteForce (colmap/feature/sift.cc), line by line."""
    inv = np.float32(1.0 / (512 * 512))
    out = []
    for row in dots.astype(np.float32):
        best_idx, best, second = -1, np.float32(0), np.float32(0)
        for j, d in enumerate(row):
            if d > best:
                best_idx, second, best = j, best, d
            elif d > second:
                second = d
        if best_idx == -1:
            out.append(-1)
            continue
        bd = np.float32(math.acos(min(float(inv * best), 1.0)))
        sd = np.float32(math.acos(min(float(inv * second), 1.0)))
        if bd > np.float32(max_distance) or bd >= np.float32(max_ratio) * sd:
            out.append(-1)
        else:
            out.append(best_idx)
    return out


def colmap_match(d1, d2, max_ratio=0.8, max_distance=0.7, cross_check=True) -> np.ndarray:
    dots = d1.astype(np.int64) @ d2.astype(np.int64).T
    m12 = colmap_one_way(dots, max_ratio, max_distance)
    m21 = colmap_one_way(dots.T, max_ratio, max_distance) if cross_check else None
    pairs = [(i, j) for i, j in enumerate(m12) if j != -1 and (m21 is None or m21[j] == i)]
    return np.array(pairs, np.uint32).reshape(-1, 2)


def sift_like(rng, n: int) -> np.ndarray:
    """Non-negative descriptors with norm ~512, like COLMAP's RootSIFT uint8."""
    x = rng.gamma(0.5, size=(n, 128))
    x = 512 * x / np.linalg.norm(x, axis=1, keepdims=True)
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("cross_check", [True, False])
def test_matches_colmap_brute_force(cross_check):
    rng = np.random.default_rng(0)
    d1 = sift_like(rng, 150)
    noise = rng.integers(-6, 7, size=(150, 128))
    d2 = np.clip(d1[rng.permutation(150)].astype(int) + noise, 0, 255).astype(np.uint8)
    d2 = np.concatenate([d2, sift_like(rng, 60)])
    d2[200] = d2[3]  # exact tie for some query: must be rejected as in COLMAP
    d2[201] = d1[7]  # exact copy: distance 0 vs a non-zero second best
    got = match_descriptors(d1, d2, cross_check=cross_check)
    want = colmap_match(d1, d2, cross_check=cross_check)
    assert len(want) > 50
    np.testing.assert_array_equal(got, want)


def test_threshold_variants_and_edge_cases():
    rng = np.random.default_rng(1)
    d1, d2 = sift_like(rng, 40), sift_like(rng, 30)
    for ratio, dist in [(0.95, 1.2), (0.6, 0.4)]:
        np.testing.assert_array_equal(
            match_descriptors(d1, d2, max_ratio=ratio, max_distance=dist),
            colmap_match(d1, d2, ratio, dist),
        )
    # One descriptor in the other image: no second best (COLMAP keeps 0).
    np.testing.assert_array_equal(match_descriptors(d1, d1[5:6]), colmap_match(d1, d1[5:6]))
    assert match_descriptors(d1[:0], d2).shape == (0, 2)


def test_match_database_writes_exact_matches(tmp_path):
    import cv2
    import pycolmap

    from av1sfm.colmap_db import create_database
    from av1sfm.devices import pick_device
    from av1sfm.sift_exact import generate_pairs, match_database

    rng = np.random.default_rng(2)
    names = [f"{i:03d}.png" for i in range(4)]
    for n in names:
        cv2.imwrite(str(tmp_path / n), np.zeros((60, 80), np.uint8))
    ids = create_database(tmp_path / "db.db", tmp_path, names)
    base = sift_like(rng, 120)
    desc = {}
    with pycolmap.Database.open(tmp_path / "db.db") as db:
        for n in names:
            d = np.clip(base.astype(int) + rng.integers(-8, 9, base.shape), 0, 255)
            desc[ids[n]] = d.astype(np.uint8)
            db.write_keypoints(ids[n], rng.uniform(1, 59, (120, 2)).astype(np.float32))
            db.write_descriptors(
                ids[n],
                pycolmap.FeatureDescriptors(pycolmap.FeatureExtractorType.SIFT, desc[ids[n]]),
            )
        pairs = generate_pairs(db, "sequential", overlap=2)
    assert len(pairs) == 5
    stats = match_database(tmp_path / "db.db", pairs, pick_device("cpu"))
    with pycolmap.Database.open(tmp_path / "db.db") as db:
        for a, b in pairs:
            np.testing.assert_array_equal(db.read_matches(a, b), colmap_match(desc[a], desc[b]))
    assert stats["pairs"] == 5 and stats["raw_matches"] > 300


def test_pick_device():
    from av1sfm.devices import pick_device

    assert pick_device("cpu").type == "cpu"
    assert pick_device("auto").type in ("cpu", "xpu", "cuda", "mps")
    with pytest.raises(ValueError):
        pick_device("npu")
