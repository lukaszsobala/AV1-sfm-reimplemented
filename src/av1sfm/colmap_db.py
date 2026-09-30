"""Write MV keypoints and matches into a COLMAP database.

Cameras, rigs, frames and images are created with `pycolmap.import_images`,
i.e. exactly as COLMAP's own feature extractor would create them, so the
resulting database is schema-compatible with the installed pycolmap/COLMAP
version. We then add keypoints, raw matches and two-view geometries, after
which `pycolmap.incremental_mapping` (or `colmap mapper` of the same version)
runs without SIFT extraction or matching.

Two-view geometries can be produced in two ways:
  * "verify" (default): COLMAP's own multi-threaded geometric verification
    (`pycolmap.geometric_verification`, the code path its matchers use) with
    the same options as the SIFT baselines, so every method goes through
    identical verification.
  * "trust": all raw matches are stored as inliers (config UNCALIBRATED),
    i.e. the MVs themselves are the verification.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pycolmap

from .tracks import MatchGraph


@dataclass
class CameraSpec:
    model: str = "SIMPLE_RADIAL"
    params: tuple[float, ...] | None = None  # None: COLMAP's default prior (f = 1.2 max(w, h))


def two_view_options(max_error: float = 4.0, min_inlier_ratio: float = 0.25,
                     max_num_trials: int = 10000) -> pycolmap.TwoViewGeometryOptions:  # fmt: skip
    """Verification options shared by all methods (SIFT baselines and MVs)."""
    opts = pycolmap.TwoViewGeometryOptions()
    opts.ransac.max_error = max_error
    opts.ransac.min_inlier_ratio = min_inlier_ratio
    opts.ransac.max_num_trials = max_num_trials
    return opts


def create_database(
    db_path: str | Path,
    image_dir: str | Path,
    image_names: list[str],
    camera: CameraSpec | None = None,
) -> dict[str, int]:
    """Create a database with one shared camera and the given images.

    Returns image name -> image_id.
    """
    camera = camera or CameraSpec()
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()
    opts = pycolmap.ImageReaderOptions()
    opts.camera_model = camera.model
    if camera.params is not None:
        opts.camera_params = ",".join(f"{v:.10g}" for v in camera.params)
    pycolmap.Database.open(db_path).close()  # creates the schema
    pycolmap.import_images(db_path, image_dir, pycolmap.CameraMode.SINGLE, image_names, opts)
    with pycolmap.Database.open(db_path) as db:
        return {im.name: im.image_id for im in db.read_all_images()}


def write_match_graph(
    db_path: str | Path,
    image_ids: Mapping[int, int],
    graph: MatchGraph,
    *,
    two_view: Literal["verify", "trust"] = "verify",
    verify_options: pycolmap.TwoViewGeometryOptions | None = None,
    min_matches: int = 15,
) -> dict:
    """Write keypoints/matches/two-view geometries. `image_ids` maps frame -> image_id.

    Pairs with fewer than `min_matches` raw matches are skipped (COLMAP's own
    `min_num_inliers` default is 15).
    """
    verify_options = verify_options or two_view_options()
    stats = {"pairs": 0, "raw_matches": 0, "inlier_pairs": 0, "inlier_matches": 0}
    with pycolmap.Database.open(db_path) as db, pycolmap.DatabaseTransaction(db):
        for frame, kps in graph.keypoints.items():
            if frame in image_ids:
                db.write_keypoints(image_ids[frame], np.ascontiguousarray(kps, np.float32))
        for (fa, fb), m in graph.matches.items():
            if fa not in image_ids or fb not in image_ids or len(m) < min_matches:
                continue
            ia, ib = image_ids[fa], image_ids[fb]
            m = np.ascontiguousarray(m, np.uint32)
            if ia > ib:  # COLMAP stores pairs with image_id1 < image_id2
                ia, ib, m = ib, ia, np.ascontiguousarray(m[:, ::-1])
            db.write_matches(ia, ib, m)
            stats["pairs"] += 1
            stats["raw_matches"] += len(m)
            if two_view == "trust":
                tvg = pycolmap.TwoViewGeometry()
                tvg.config = pycolmap.TwoViewGeometryConfiguration.UNCALIBRATED
                tvg.inlier_matches = m
                db.write_two_view_geometry(ia, ib, tvg)
    if two_view == "verify":
        pycolmap.geometric_verification(
            db_path, pycolmap.GeometricVerifierOptions(),
            pycolmap.ExistingMatchedPairingOptions(), verify_options,
        )  # fmt: skip
    with pycolmap.Database.open(db_path) as db:
        stats["inlier_pairs"] = db.num_verified_image_pairs()
        stats["inlier_matches"] = db.num_inlier_matches()
    return stats
