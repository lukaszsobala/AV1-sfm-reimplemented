"""Incremental SfM on a prepared database with fixed, shared mapper settings.

Reports registered images, 3D points, mean reprojection error, total mapper
wall time, and the wall time of one final global bundle adjustment of the
largest model (pycolmap does not expose the BA share of incremental mapping).

    uv run python eval/run_mapper.py DB IMAGES OUT_DIR --stats out.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pycolmap

from av1sfm.timing import Timer


def mapper_options(
    num_threads: int = -1,
    init_max_forward_motion: float = 0.95,
    init_min_tri_angle: float = 16.0,
    fix_intrinsics: bool = False,
) -> pycolmap.IncrementalPipelineOptions:
    """Identical for every method: COLMAP defaults, shared intrinsics across images.

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
    if fix_intrinsics:
        opts.ba_refine_focal_length = False
        opts.ba_refine_principal_point = False
        opts.ba_refine_extra_params = False
    return opts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("database", type=Path)
    ap.add_argument("image_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--stats", type=Path, default=None)
    ap.add_argument("--init-max-forward-motion", type=float, default=0.95)
    ap.add_argument("--init-min-tri-angle", type=float, default=16.0)
    ap.add_argument("--fix-intrinsics", action="store_true", help="do not refine the camera")
    a = ap.parse_args()

    if a.out_dir.exists():
        shutil.rmtree(a.out_dir)
    a.out_dir.mkdir(parents=True)
    timer = Timer()
    with timer.stage("mapper"):
        opts = mapper_options(
            init_max_forward_motion=a.init_max_forward_motion,
            init_min_tri_angle=a.init_min_tri_angle,
            fix_intrinsics=a.fix_intrinsics,
        )
        recs = pycolmap.incremental_mapping(a.database, a.image_dir, a.out_dir, opts)
    stats: dict = {
        "timings": timer.asdict(),
        "num_models": len(recs),
        "init_max_forward_motion": a.init_max_forward_motion,
        "init_min_tri_angle": a.init_min_tri_angle,
        "fix_intrinsics": a.fix_intrinsics,
    }
    if recs:
        rec = max(recs.values(), key=lambda r: r.num_reg_images())
        stats.update(
            {
                "registered_images": rec.num_reg_images(),
                "points3D": rec.num_points3D(),
                "mean_reprojection_error_px": rec.compute_mean_reprojection_error(),
                "mean_track_length": rec.compute_mean_track_length(),
                "mean_observations_per_image": rec.compute_mean_observations_per_reg_image(),
                "camera": rec.cameras[next(iter(rec.cameras))].params.tolist(),
            }
        )
        t0 = time.perf_counter()
        ba = pycolmap.BundleAdjustmentOptions()
        if a.fix_intrinsics:
            ba.refine_focal_length = ba.refine_principal_point = ba.refine_extra_params = False
        pycolmap.bundle_adjustment(rec, ba)
        stats["final_global_ba_s"] = round(time.perf_counter() - t0, 3)
        stats["mean_reprojection_error_after_ba_px"] = rec.compute_mean_reprojection_error()
    text = json.dumps(stats, indent=2)
    if a.stats:
        a.stats.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
