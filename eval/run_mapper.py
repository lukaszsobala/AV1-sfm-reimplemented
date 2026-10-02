"""SfM on a prepared database with fixed, shared mapper settings (av1sfm.mapping).

Reports registered images, 3D points, mean reprojection error, total mapper
wall time, and the wall time of one final global bundle adjustment of the
largest model (pycolmap does not expose the BA share of incremental mapping).
With `--kitti-sequence`, camera poses are also compared with the KITTI ground
truth (eval/pose_error.py).

The default is COLMAP's incremental mapper with two changes that save time
but not accuracy (av1sfm.mapping, docs/RESULTS.md): global bundle adjustments
skip redundant 3D points, and each registered image gets one local bundle
adjustment instead of up to two. The results published before these defaults
used COLMAP's unmodified settings: `--no-prune-redundant-points
--ba-local-refinements 2`.

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

from av1sfm.mapping import largest_model, model_stats, run_mapper
from av1sfm.timing import Timer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="incremental: global BAs skip redundant 3D points (default; --no-... for "
        "COLMAP's unmodified setting)",
    )
    ap.add_argument(
        "--ba-local-refinements",
        type=int,
        default=1,
        help="incremental: local bundle adjustments per registered image (COLMAP: 2)",
    )
    ap.add_argument(
        "--ba-global-ratio",
        type=float,
        default=1.1,
        help="incremental: global BA when the model grew by this factor (COLMAP: 1.1)",
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
        recs = run_mapper(
            a.database,
            a.image_dir,
            a.out_dir,
            mapper=a.mapper,
            fix_intrinsics=a.fix_intrinsics,
            init_max_forward_motion=a.init_max_forward_motion,
            init_min_tri_angle=a.init_min_tri_angle,
            prune=a.prune_redundant_points,
            local_refinements=a.ba_local_refinements,
            global_ratio=a.ba_global_ratio,
            global_tracks_per_view=a.global_tracks_per_view,
        )
    stats: dict = {
        "timings": timer.asdict(),
        "num_models": len(recs),
        "mapper": a.mapper,
        "init_max_forward_motion": a.init_max_forward_motion,
        "init_min_tri_angle": a.init_min_tri_angle,
        "fix_intrinsics": a.fix_intrinsics,
        "prune_redundant_points": a.prune_redundant_points if a.mapper == "incremental" else None,
        "ba_local_refinements": a.ba_local_refinements if a.mapper == "incremental" else None,
        "ba_global_ratio": a.ba_global_ratio if a.mapper == "incremental" else None,
        "global_tracks_per_view": a.global_tracks_per_view,
    }
    if recs:
        best, rec = largest_model(recs)
        stats.update(model_stats(rec))
        t0 = time.perf_counter()
        ba = pycolmap.BundleAdjustmentOptions()
        if a.fix_intrinsics:
            ba.refine_focal_length = ba.refine_principal_point = ba.refine_extra_params = False
        pycolmap.bundle_adjustment(rec, ba)
        stats["final_global_ba_s"] = round(time.perf_counter() - t0, 3)
        stats["mean_reprojection_error_after_ba_px"] = rec.compute_mean_reprojection_error()
        if a.kitti_sequence is not None:
            # Poses as written by the mapper, before the timing-only final BA.
            stats["pose_error"] = pose_error(a.out_dir / str(best), a.kitti_sequence)
    text = json.dumps(stats, indent=2)
    if a.stats:
        a.stats.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
