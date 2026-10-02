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


def reconstruct(input_path: str | Path, out_dir: str | Path, cfg: ReconstructConfig) -> dict:
    """Run every stage; returns the statistics also written to `reconstruct.json`."""
    inp, out = Path(input_path), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats: dict = {"input": str(inp), "config": _jsonable(asdict(cfg)), "cpu_count": os.cpu_count()}
    pre, post = Timer(), Timer()

    ivf = out / "clip.ivf"
    encode = True
    if inp.is_dir():
        image_dir = inp
        if cfg.ivf is not None:
            ivf, encode = Path(cfg.ivf), False
    else:
        info = probe_video(inp)
        stats["video"] = asdict(info)
        image_dir = out / "images"
        with pre.stage("frames"):
            extract_frames(inp, image_dir, cfg.frame_format)
        if cfg.matcher == "mv" and info.codec == "av1" and not cfg.reencode:
            with pre.stage("copy"):  # the MVs are in the stream already
                copy_av1_to_ivf(inp, ivf)
            encode = False

    db = out / "database.db"
    if cfg.matcher == "mv":
        matching = run_mv_matching(image_dir, db, ivf, cfg.mv, encode=encode)
        matching["matches"] = matches_per_image(db)
    elif cfg.matcher == "sift":
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
        )
    else:
        raise ValueError(f"matcher must be 'mv' or 'sift', got {cfg.matcher!r}")
    stats["matching"] = matching

    sparse = out / "sparse"
    if sparse.exists():
        shutil.rmtree(sparse)
    with post.stage("mapper"):
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
    stats["num_images"] = len(list_images(image_dir))
    stats["num_models"] = len(recs)
    if recs:
        best, rec = largest_model(recs)
        stats["model"] = str(sparse / str(best))
        stats.update(model_stats(rec))
        with post.stage("export"):
            export_ply(sparse / str(best), out / "points.ply")
            if cfg.export_dataset:
                export_dataset(sparse / str(best), image_dir, out / "dataset")
    else:
        stats["registered_images"] = 0
    stats["timings"] = {**pre.asdict(), **matching["timings"], **post.asdict()}
    stats["total_wall_s"] = round(sum(t["wall_s"] for t in stats["timings"].values()), 3)
    (out / "reconstruct.json").write_text(json.dumps(stats, indent=2))
    return stats
