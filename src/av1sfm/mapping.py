"""Sparse reconstruction (camera poses and 3D points) from a prepared COLMAP database.

The same mapper settings are used for every matching method:

  incremental  COLMAP's incremental mapper with its default settings, except
               that global bundle adjustments skip redundant 3D points
               (`ba_global_ignore_redundant_points3D`). Dense MV tracks add
               more than 10 % new points with almost every image, which
               triggers a global bundle adjustment after nearly every
               registration; skipping redundant points makes the mapper
               14-36 % faster on KITTI with the same points, reprojection
               error and pose accuracy (docs/RESULTS.md). `prune=False`
               restores COLMAP's default.
  global       COLMAP's global mapper (GLOMAP): faster, but on KITTI it gives
               3-12 % fewer points and slightly worse poses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pycolmap


def incremental_options(
    *,
    num_threads: int = -1,
    init_max_forward_motion: float = 0.95,
    init_min_tri_angle: float = 16.0,
    fix_intrinsics: bool = False,
    prune: bool = True,
) -> pycolmap.IncrementalPipelineOptions:
    """COLMAP's incremental mapper with one shared model.

    `init_max_forward_motion` (COLMAP default 0.95) rejects initial pairs whose
    baseline is mostly along the optical axis; forward-driving sequences such
    as KITTI need 1.0 or no initial pair is ever accepted. For the same reason
    their short-baseline pairs rarely reach the default 16 degree initial
    triangulation angle (COLMAP relaxes it by halving).

    `fix_intrinsics` keeps the shared camera at its (calibrated) initial value.
    With COLMAP's default refinement, a short-baseline initial pair can drive
    the SIMPLE_RADIAL distortion to |k| >> 1 in the first bundle adjustment;
    COLMAP then treats the camera as bogus and registers nothing else.
    """
    opts = pycolmap.IncrementalPipelineOptions()
    opts.num_threads = num_threads
    opts.multiple_models = False
    opts.mapper.init_max_forward_motion = init_max_forward_motion
    opts.mapper.init_min_tri_angle = init_min_tri_angle
    opts.mapper.ba_global_ignore_redundant_points3D = prune
    if fix_intrinsics:
        opts.ba_refine_focal_length = False
        opts.ba_refine_principal_point = False
        opts.ba_refine_extra_params = False
    return opts


def global_options(
    *, num_threads: int = -1, fix_intrinsics: bool = False, tracks_per_view: int | None = None
) -> pycolmap.GlobalPipelineOptions:
    """COLMAP's global mapper (GLOMAP) with its defaults and one shared model.

    `tracks_per_view` positions the cameras with that many tracks per image
    instead of all of them; every track is triangulated afterwards.
    """
    opts = pycolmap.GlobalPipelineOptions()
    opts.num_threads = num_threads
    opts.multiple_models = False
    if tracks_per_view is not None:
        opts.mapper.track_required_tracks_per_view = tracks_per_view
    if fix_intrinsics:
        ba = opts.mapper.bundle_adjustment
        ba.refine_focal_length = ba.refine_principal_point = ba.refine_extra_params = False
    return opts


def run_mapper(
    database: str | Path,
    image_dir: str | Path,
    out_dir: str | Path,
    *,
    mapper: Literal["incremental", "global"] = "incremental",
    fix_intrinsics: bool = False,
    init_max_forward_motion: float = 0.95,
    init_min_tri_angle: float = 16.0,
    prune: bool = True,
    global_tracks_per_view: int | None = None,
) -> dict[int, pycolmap.Reconstruction]:
    """Reconstruct; models are written to `out_dir/<index>`."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if mapper == "global":
        gopts = global_options(
            fix_intrinsics=fix_intrinsics, tracks_per_view=global_tracks_per_view
        )
        return pycolmap.global_mapping(database, image_dir, out_dir, gopts)
    if mapper != "incremental":
        raise ValueError(f"mapper must be 'incremental' or 'global', got {mapper!r}")
    opts = incremental_options(
        init_max_forward_motion=init_max_forward_motion,
        init_min_tri_angle=init_min_tri_angle,
        fix_intrinsics=fix_intrinsics,
        prune=prune,
    )
    return pycolmap.incremental_mapping(database, image_dir, out_dir, opts)


def largest_model(recs: dict[int, pycolmap.Reconstruction]) -> tuple[int, pycolmap.Reconstruction]:
    """(index, model) with the most registered images."""
    k = max(recs, key=lambda i: recs[i].num_reg_images())
    return k, recs[k]


def model_stats(rec: pycolmap.Reconstruction) -> dict:
    return {
        "registered_images": rec.num_reg_images(),
        "points3D": rec.num_points3D(),
        "mean_reprojection_error_px": rec.compute_mean_reprojection_error(),
        "mean_track_length": rec.compute_mean_track_length(),
        "mean_observations_per_image": rec.compute_mean_observations_per_reg_image(),
        "camera": rec.cameras[next(iter(rec.cameras))].params.tolist(),
    }
