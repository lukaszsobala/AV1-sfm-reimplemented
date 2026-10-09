"""From images or a video to a sparse 3D reconstruction in one command.

    av1sfm reconstruct INPUT OUT_DIR

INPUT is a folder of frames (in display order when sorted by name) or a video
file. The outputs in OUT_DIR:

  images/           frames decoded from a video input (an image folder is used in place)
  clip.ivf          MV matcher: the AV1 stream, copied from an AV1 video without
                    re-encoding, otherwise encoded from the frames (low delay)
  database.db       COLMAP database: keypoints, matches, two-view geometries
  sparse/0/         the reconstruction (cameras, images, points3D)
  points.ply        coloured point cloud of the largest model
  dataset/          with `export_dataset`: undistorted workspace for OpenMVS / Brush
  reconstruct.json  settings, stage timings and statistics
  reconstruct.log   messages of COLMAP, FFmpeg and dav1d (`av1sfm reconstruct`
                    shows a progress display instead; `--verbose` prints them)

Matchers: `mv` (AV1 motion vectors, the default) or `sift` (COLMAP SIFT
features, by default with exact sequential matching on a PyTorch device, as
`eval/run_sift.py --matching sequential --matcher exact`). Both use the same
camera model, verification and mapper settings (av1sfm.mapping).

Copying an AV1 video's stream is the fastest route, but ordinary AV1 videos
are random access (hidden alt-refs, references to later frames): frames that
no other frame references can only start tracks, so tracks are shorter than
in the low-delay stream the encoders here produce. On KITTI frames 0-39, a
copied libaom stream (default good mode) gave 16 k points and 1.3 m trajectory
error, a low-delay re-encode 72 k points and 0.22 m. `reencode` re-encodes
AV1 videos as well.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from .encode import list_images
from .export import export_dataset, export_ply
from .mapping import largest_model, model_stats, run_mapper
from .pipeline import MVMatchConfig, _jsonable, matches_per_image, run_mv_matching
from .progress import Progress, global_mapper, incremental_mapper
from .timing import Timer
from .video import copy_av1_to_ivf, extract_frames, probe_video


@dataclass
class ReconstructConfig:
    matcher: Literal["mv", "sift"] = "mv"
    mv: MVMatchConfig = field(default_factory=MVMatchConfig)  # also holds the shared camera
    ivf: Path | None = None  # image-folder input: use this AV1 stream instead of encoding
    reencode: bool = False  # AV1 video input: encode the frames instead of copying the stream
    sift_matching: Literal["sequential", "exhaustive"] = "sequential"
    sift_overlap: int = 10
    sift_matcher: Literal["exact", "colmap"] = "exact"
    device: str = "auto"  # PyTorch device for exact SIFT matching
    max_features: int = 8192
    mapper: Literal["incremental", "global"] = "incremental"
    fix_intrinsics: bool = False
    init_max_forward_motion: float = 0.95
    init_min_tri_angle: float = 16.0
    prune: bool = True
    local_refinements: int = 1
    global_ratio: float = 1.1
    global_tracks_per_view: int | None = None
    export_dataset: bool = False
    frame_format: Literal["png", "jpg"] = "png"


def reconstruct(
    input_path: str | Path,
    out_dir: str | Path,
    cfg: ReconstructConfig,
    progress: Progress | None = None,
) -> dict:
    """Run every stage; returns the statistics also written to `reconstruct.json`."""
    inp, out = Path(input_path), Path(out_dir)
    if cfg.matcher not in ("mv", "sift"):
        raise ValueError(f"matcher must be 'mv' or 'sift', got {cfg.matcher!r}")
    out.mkdir(parents=True, exist_ok=True)
    progress = progress or Progress(enabled=False)
    stats: dict = {"input": str(inp), "config": _jsonable(asdict(cfg)), "cpu_count": os.cpu_count()}
    pre, post = Timer(), Timer()

    ivf = out / "clip.ivf"
    encode = True
    copy = False
    info = None
    if inp.is_dir():
        image_dir = inp
        if cfg.ivf is not None:
            ivf, encode = Path(cfg.ivf), False
    else:
        info = probe_video(inp)
        stats["video"] = asdict(info)
        image_dir = out / "images"
        copy = cfg.matcher == "mv" and info.codec == "av1" and not cfg.reencode
        encode = not copy
    progress.total_steps = planned_steps(cfg, video=info is not None, encode=encode, copy=copy)

    if info is not None:
        with (
            pre.stage("frames"),
            progress.step(
                f"Decoding the video into {image_dir.name}/",
                total=info.frames,
                unit="frames",
            ) as step,
        ):
            n = len(extract_frames(inp, image_dir, cfg.frame_format, progress=step.update))
            step.update(done=n, total=n)
        if copy:
            with (
                pre.stage("copy"),  # the MVs are in the stream already
                progress.step("Copying the AV1 stream (no re-encoding)"),
            ):
                copy_av1_to_ivf(inp, ivf)

    db = out / "database.db"
    if cfg.matcher == "mv":
        matching = run_mv_matching(image_dir, db, ivf, cfg.mv, encode=encode, progress=progress)
        matching["matches"] = matches_per_image(db)
    else:
        from .sift import run_sift_matching

        matching = run_sift_matching(
            image_dir,
            db,
            matching=cfg.sift_matching,
            overlap=cfg.sift_overlap,
            matcher=cfg.sift_matcher,
            device=cfg.device,
            camera=cfg.mv.camera,
            max_features=cfg.max_features,
            progress=progress,
        )
    stats["matching"] = matching

    sparse = out / "sparse"
    if sparse.exists():
        shutil.rmtree(sparse)
    num_images = len(list_images(image_dir))
    with (
        post.stage("mapper"),
        progress.step(
            "Reconstructing cameras and 3D points"
            + (" (global mapper)" if cfg.mapper == "global" else ""),
            total=num_images if cfg.mapper == "incremental" else None,
            unit="images",
            parse=incremental_mapper if cfg.mapper == "incremental" else global_mapper,
        ) as step,
    ):
        recs = run_mapper(
            db,
            image_dir,
            sparse,
            mapper=cfg.mapper,
            fix_intrinsics=cfg.fix_intrinsics,
            init_max_forward_motion=cfg.init_max_forward_motion,
            init_min_tri_angle=cfg.init_min_tri_angle,
            prune=cfg.prune,
            local_refinements=cfg.local_refinements,
            global_ratio=cfg.global_ratio,
            global_tracks_per_view=cfg.global_tracks_per_view,
        )
        if recs:
            rec = largest_model(recs)[1]
            step.update(done=rec.num_reg_images(), detail="")
            step.result = (
                f"{rec.num_reg_images()}/{num_images} images, {rec.num_points3D():,} points"
            )
        else:
            step.update(done=0, detail="")
            step.result = "no model"
    stats["num_images"] = num_images
    stats["num_models"] = len(recs)
    if recs:
        best, rec = largest_model(recs)
        stats["model"] = str(sparse / str(best))
        stats.update(model_stats(rec))
        with post.stage("export"):
            with progress.step("Writing points.ply"):
                export_ply(sparse / str(best), out / "points.ply")
            if cfg.export_dataset:
                with progress.step("Undistorting the images into dataset/"):
                    export_dataset(sparse / str(best), image_dir, out / "dataset")
    else:
        stats["registered_images"] = 0
    stats["timings"] = {**pre.asdict(), **matching["timings"], **post.asdict()}
    stats["total_wall_s"] = round(sum(t["wall_s"] for t in stats["timings"].values()), 3)
    (out / "reconstruct.json").write_text(json.dumps(stats, indent=2))
    return stats


def planned_steps(cfg: ReconstructConfig, *, video: bool, encode: bool, copy: bool) -> int:
    """Number of progress steps `reconstruct` shows (the "[k/N]" of the display)."""
    n = (1 + copy) if video else 0
    if cfg.matcher == "mv":
        n += encode + 3 + (cfg.mv.two_view == "verify")  # MVs, tracks, database (+ verify)
    else:
        n += 2 + (cfg.sift_matcher == "exact") * 2  # extract, match (+ warm-up, verify)
    return n + 2 + cfg.export_dataset  # mapper, points.ply (+ dataset); no model: one less
