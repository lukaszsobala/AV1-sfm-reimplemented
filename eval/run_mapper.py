"""SfM on a prepared database with fixed, shared mapper settings.

Reports registered images, 3D points, mean reprojection error, total mapper
wall time, and the wall time of one final global bundle adjustment of the
largest model (pycolmap does not expose the BA share of incremental mapping).
With `--kitti-sequence`, camera poses are also compared with the KITTI ground
truth (eval/pose_error.py).

The default is COLMAP's incremental mapper with its default settings, as for
all published results. Two faster variants are opt-in (docs/RESULTS.md has
their accuracy on KITTI):

  --prune-redundant-points  incremental mapper; global bundle adjustments skip
                            3D points that add little image coverage (COLMAP's
                            ba_global_ignore_redundant_points3D). Dense MV
                            tracks trigger a global BA after almost every image.
  --mapper global           COLMAP's global mapper (GLOMAP): rotation averaging,
                            global positioning, then bundle adjustment. Faster,
                            but fewer points and slightly worse poses on KITTI.
                            `--global-tracks-per-view N` positions the cameras
                            with N tracks per image instead of all of them; all
                            tracks are triangulated afterwards.

    uv run python eval/run_mapper.py DB IMAGES OUT_DIR --stats out.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pycolmap
from pose_error import evaluate as pose_error

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


def global_mapper_options(
    num_threads: int = -1, fix_intrinsics: bool = False, tracks_per_view: int | None = None
) -> pycolmap.GlobalPipelineOptions:
    """COLMAP's global mapper (GLOMAP) with its defaults and one shared model."""
    opts = pycolmap.GlobalPipelineOptions()
    opts.num_threads = num_threads
    opts.multiple_models = False
    if tracks_per_view is not None:
        opts.mapper.track_required_tracks_per_view = tracks_per_view
    if fix_intrinsics:
        ba = opts.mapper.bundle_adjustment
        ba.refine_focal_length = ba.refine_principal_point = ba.refine_extra_params = False
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
    ap.add_argument("--mapper", choices=["incremental", "global"], default="incremental")
    ap.add_argument(
        "--prune-redundant-points",
        action="store_true",
        help="incremental: global BAs skip redundant 3D points (faster on dense MV tracks)",
    )
    ap.add_argument(
        "--global-tracks-per-view",
        type=int,
        default=None,
        help="global: tracks per image used for global positioning (default: all)",
    )
    ap.add_argument(
        "--kitti-sequence",
        type=Path,
        default=None,
        help="KITTI sequence folder (NN.txt, calib.txt): add camera pose errors to the stats",
    )
    a = ap.parse_args()

    if a.out_dir.exists():
        shutil.rmtree(a.out_dir)
    a.out_dir.mkdir(parents=True)
    timer = Timer()
    with timer.stage("mapper"):
        if a.mapper == "global":
            gopts = global_mapper_options(
                fix_intrinsics=a.fix_intrinsics, tracks_per_view=a.global_tracks_per_view
            )
            recs = pycolmap.global_mapping(a.database, a.image_dir, a.out_dir, gopts)
        else:
            opts = mapper_options(
                init_max_forward_motion=a.init_max_forward_motion,
                init_min_tri_angle=a.init_min_tri_angle,
                fix_intrinsics=a.fix_intrinsics,
            )
            opts.mapper.ba_global_ignore_redundant_points3D = a.prune_redundant_points
            recs = pycolmap.incremental_mapping(a.database, a.image_dir, a.out_dir, opts)
    stats: dict = {
        "timings": timer.asdict(),
        "num_models": len(recs),
        "mapper": a.mapper,
        "init_max_forward_motion": a.init_max_forward_motion,
        "init_min_tri_angle": a.init_min_tri_angle,
        "fix_intrinsics": a.fix_intrinsics,
        "prune_redundant_points": a.prune_redundant_points,
        "global_tracks_per_view": a.global_tracks_per_view,
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
        if a.kitti_sequence is not None:
            # Poses as written by the mapper, before the timing-only final BA.
            best = max(recs, key=lambda k: recs[k].num_reg_images())
            stats["pose_error"] = pose_error(a.out_dir / str(best), a.kitti_sequence)
    text = json.dumps(stats, indent=2)
    if a.stats:
        a.stats.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
