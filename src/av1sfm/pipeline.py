"""End-to-end MV matching: images -> IVF -> MVs -> tracks -> COLMAP database."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pycolmap

from .colmap_db import CameraSpec, create_database, two_view_options, write_match_graph
from .encode import EncodeParams, encode_images, list_images
from .extract import FrameMotion, load_frame_motion
from .geometry import PairScore, RansacSettings, score_pair, summarize
from .timing import Timer
from .tracks import TrackParams, build_tracks, tracks_to_matches


@dataclass
class MVMatchConfig:
    track: TrackParams = field(default_factory=TrackParams)
    encode: EncodeParams = field(default_factory=EncodeParams)
    camera: CameraSpec = field(default_factory=CameraSpec)
    max_pair_gap: int | None = None
    two_view: Literal["verify", "trust"] = "verify"
    decoder_threads: int = 0


def run_mv_matching(
    image_dir: str | Path,
    db_path: str | Path,
    ivf_path: str | Path,
    cfg: MVMatchConfig | None = None,
    *,
    encode: bool = True,
) -> dict:
    """Run the MV pipeline. The i-th decoded frame is the i-th image (sorted by name).

    With `encode=False` an existing IVF is used (the "MVs come for free with the
    video" scenario); the encode stage is then absent from the timings.
    """
    cfg = cfg or MVMatchConfig()
    images = list_images(image_dir)
    timer = Timer()
    ffmpeg_cmd: list[str] | None = None
    if encode:
        with timer.stage("encode"):
            ffmpeg_cmd = encode_images(images, ivf_path, cfg.encode)
    with timer.stage("extract"):
        frames = load_frame_motion(ivf_path, n_threads=cfg.decoder_threads)
    if len(frames) != len(images):
        raise ValueError(f"{len(frames)} decoded frames but {len(images)} images")
    clamp_to_image_size(frames, images[0])
    with timer.stage("tracks"):
        tracks = build_tracks(frames, cfg.track)
        graph = tracks_to_matches(tracks, cfg.max_pair_gap)
    with timer.stage("database"):
        ids = create_database(db_path, image_dir, [p.name for p in images], cfg.camera)
        frame_to_id = {i: ids[p.name] for i, p in enumerate(images)}
        db_stats = write_match_graph(
            db_path, frame_to_id, graph, two_view=cfg.two_view, verify_options=two_view_options()
        )
    kp = np.array([len(v) for v in graph.keypoints.values()]) if graph.keypoints else np.zeros(1)
    lengths = tracks.lengths()
    return {
        "method": "av1-mv",
        "config": _jsonable(asdict(cfg)),
        "timings": timer.asdict(),
        "tracks": {
            **tracks.stats,
            "mean_length": float(lengths.mean()) if len(lengths) else 0.0,
            "max_length": int(lengths.max()) if len(lengths) else 0,
        },
        "keypoints_per_image": float(kp.mean()),
        "database": db_stats,
        "num_images": len(images),
        "ffmpeg_command": ffmpeg_cmd,
        "cpu_count": os.cpu_count(),
    }


def clamp_to_image_size(frames: list[FrameMotion], image: Path) -> None:
    """Limit frames to the source image size.

    Hardware encoders may code a padded frame (e.g. width rounded up to 16 or
    64 px) and signal the true size only as the render size; the decoded frame
    is then larger than the images. Keypoints and MV targets must stay inside
    the real image, so the usable frame size is clamped to it.
    """
    h, w = cv2.imread(str(image), cv2.IMREAD_UNCHANGED).shape[:2]
    for fm in frames:
        if fm.width < w or fm.height < h:
            raise ValueError(f"decoded frame {fm.width}x{fm.height} smaller than image {w}x{h}")
        fm.width, fm.height = w, h


def score_database(
    db_path: str | Path,
    settings: RansacSettings | None = None,
    max_pairs: int | None = None,
    num_threads: int | None = None,
) -> tuple[list[PairScore], dict]:
    """Score the raw matches of every pair in a COLMAP database (any method).

    Pairs are scored in parallel threads (pycolmap releases the GIL); each pair
    uses a fixed RANSAC seed, so results do not depend on scheduling.
    """
    settings = settings or RansacSettings()
    with pycolmap.Database.open(db_path) as db:
        cams = {c.camera_id: c for c in db.read_all_cameras()}
        imgs = {im.image_id: im for im in db.read_all_images()}
        kps = {i: db.read_keypoints(i)[:, :2].astype(np.float64) for i in imgs}
        pair_ids, matches = db.read_all_matches()
    order = [k for k in np.argsort(pair_ids) if len(matches[k])]
    if max_pairs is not None:
        order = order[:max_pairs]

    def job(k: int) -> PairScore:
        i1, i2 = pycolmap.pair_id_to_image_pair(pair_ids[k])
        a, b, m = imgs[i1], imgs[i2], matches[k]
        return score_pair(
            cams[a.camera_id],
            cams[b.camera_id],
            kps[i1][m[:, 0]],
            kps[i2][m[:, 1]],
            settings,
            (a.name, b.name),
        )

    with ThreadPoolExecutor(num_threads or os.cpu_count()) as pool:
        scores = list(pool.map(job, order))
    return scores, summarize(scores)


def matches_per_image(db_path: str | Path) -> dict:
    """Raw and verified matches per image: sum over pairs touching the image, averaged."""
    with pycolmap.Database.open(db_path) as db:
        n_img = db.num_images()
        raw = db.num_matches()
        inl = db.num_inlier_matches()
        n_kp = db.num_keypoints()
        verified_pairs = db.num_verified_image_pairs()
    return {
        "num_images": n_img,
        "keypoints_per_image": n_kp / max(n_img, 1),
        "raw_matches_per_image": 2 * raw / max(n_img, 1),
        "verified_matches_per_image": 2 * inl / max(n_img, 1),
        "verified_pairs": verified_pairs,
    }


def _jsonable(x):
    return json.loads(json.dumps(x, default=str))
