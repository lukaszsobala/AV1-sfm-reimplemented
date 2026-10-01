import json

import cv2
import numpy as np
import pycolmap
import pytest

from av1sfm.colmap_db import CameraSpec, create_database, write_match_graph
from av1sfm.tracks import MatchGraph


@pytest.fixture
def image_dir(tmp_path):
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"{i:03d}.png"), np.zeros((120, 160), np.uint8))
    return tmp_path


def graph_with_homography(n_pts=200, outliers=20, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform([5, 5], [155, 115], size=(n_pts, 2))
    kps, matches = {}, {}
    for f in range(3):
        kps[f] = (base + [2.0 * f, 1.0 * f]).astype(np.float32)
    kps[2][:outliers] = rng.uniform([5, 5], [155, 115], size=(outliers, 2))
    idx = np.arange(n_pts, dtype=np.uint32)
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        matches[(a, b)] = np.stack([idx, idx], axis=1)
    return MatchGraph(kps, matches)


def test_create_database_single_shared_camera(image_dir, tmp_path):
    names = [f"{i:03d}.png" for i in range(3)]
    ids = create_database(
        tmp_path / "db.db", image_dir, names, CameraSpec("SIMPLE_RADIAL", (200.0, 80.0, 60.0, 0.0))
    )
    assert sorted(ids) == names
    with pycolmap.Database.open(tmp_path / "db.db") as db:
        cams = db.read_all_cameras()
        assert len(cams) == 1 and cams[0].model.name == "SIMPLE_RADIAL"
        np.testing.assert_allclose(cams[0].params, [200.0, 80.0, 60.0, 0.0])
        assert (cams[0].width, cams[0].height) == (160, 120)
        assert db.num_frames() == 3 and db.num_rigs() >= 1


@pytest.mark.parametrize("mode", ["verify", "trust"])
def test_write_match_graph_round_trip(image_dir, tmp_path, mode):
    names = [f"{i:03d}.png" for i in range(3)]
    ids = create_database(tmp_path / "db.db", image_dir, names)
    frame_to_id = {i: ids[n] for i, n in enumerate(names)}
    g = graph_with_homography()
    stats = write_match_graph(tmp_path / "db.db", frame_to_id, g, two_view=mode)
    assert stats["pairs"] == 3
    with pycolmap.Database.open(tmp_path / "db.db") as db:
        for f, iid in frame_to_id.items():
            np.testing.assert_allclose(db.read_keypoints(iid)[:, :2], g.keypoints[f])
        m = db.read_matches(frame_to_id[0], frame_to_id[2])
        np.testing.assert_array_equal(m, g.matches[(0, 2)])
        tvg = db.read_two_view_geometry(frame_to_id[0], frame_to_id[2])
        if mode == "trust":
            assert len(tvg.inlier_matches) == 200
        else:
            # the 20 corrupted keypoints in frame 2 are rejected by verification
            inl = set(tvg.inlier_matches[:, 0].tolist())
            assert 170 <= len(inl) <= 180 and not inl & set(range(20))
            assert tvg.config != pycolmap.TwoViewGeometryConfiguration.UNDEFINED


def test_pairs_below_min_matches_are_skipped(image_dir, tmp_path):
    names = [f"{i:03d}.png" for i in range(3)]
    ids = create_database(tmp_path / "db.db", image_dir, names)
    g = graph_with_homography(n_pts=10, outliers=0)
    stats = write_match_graph(tmp_path / "db.db", {i: ids[n] for i, n in enumerate(names)}, g)
    assert stats["pairs"] == 0


def test_score_database_checkpoint_resumes(image_dir, tmp_path):
    from av1sfm.geometry import RansacSettings
    from av1sfm.pipeline import score_database

    names = [f"{i:03d}.png" for i in range(3)]
    ids = create_database(tmp_path / "db.db", image_dir, names)
    write_match_graph(
        tmp_path / "db.db",
        {i: ids[n] for i, n in enumerate(names)},
        graph_with_homography(),
        two_view="trust",
    )
    ckpt = tmp_path / "score.partial.jsonl"
    full, _ = score_database(tmp_path / "db.db", checkpoint=ckpt)
    lines = ckpt.read_text().splitlines()
    assert len(lines) == 4  # settings header + 3 pairs

    # Simulate a killed run: one pair done, a truncated line, then resume.
    first = json.loads(lines[1])
    first["num_inliers"] = -1  # marker: must be reused, not recomputed
    ckpt.write_text(lines[0] + "\n" + json.dumps(first) + '\n{"image1": "00')
    resumed, _ = score_database(tmp_path / "db.db", checkpoint=ckpt)
    assert [(s.image1, s.image2) for s in resumed] == [(s.image1, s.image2) for s in full]
    assert sum(s.num_inliers == -1 for s in resumed) == 1
    assert len(ckpt.read_text().splitlines()) == 4

    # Different settings discard the checkpoint.
    again, _ = score_database(tmp_path / "db.db", RansacSettings(repeats=2), checkpoint=ckpt)
    assert all(s.num_inliers >= 0 for s in again)
