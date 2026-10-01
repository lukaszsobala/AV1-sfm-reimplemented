"""Command-line interface: `av1sfm {encode,match,score,validate-warp,encoders}`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .colmap_db import CameraSpec
from .encode import (
    BACKENDS,
    EncodeParams,
    available_encoders,
    encode_images,
    find_ffmpeg,
    list_images,
    probe_encoder,
)
from .geometry import RansacSettings
from .pipeline import MVMatchConfig, matches_per_image, run_mv_matching, score_database
from .tracks import TrackParams


def _params(s: str | None) -> tuple[float, ...] | None:
    return tuple(float(v) for v in s.split(",")) if s else None


def _add_encode_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("encoding")
    g.add_argument(
        "--encoder",
        default="libaom",
        choices=[*BACKENDS, "auto"],
        help="auto: first working of vulkan, qsv, vaapi, svtav1",
    )
    g.add_argument("--crf", type=int, default=32, help="libaom / svtav1 CRF")
    g.add_argument("--qp", type=int, default=128, help="vulkan / qsv / vaapi AV1 qindex (0-255)")
    g.add_argument("--usage", default="realtime", choices=["realtime", "good"], help="libaom")
    g.add_argument("--cpu-used", type=int, default=6, help="libaom speed")
    g.add_argument("--svt-preset", type=int, default=10, help="SVT-AV1 preset")
    g.add_argument(
        "--svt-params", default="", help="extra SVT-AV1 params, e.g. hierarchical-levels=2"
    )
    g.add_argument(
        "--hw-device",
        default=None,
        help="vulkan: device index; qsv, vaapi: DRM render node (e.g. /dev/dri/renderD128)",
    )
    g.add_argument("--fps", type=int, default=10)
    g.add_argument("--threads", type=int, default=0)


def _encode_params(a: argparse.Namespace) -> EncodeParams:
    return EncodeParams(
        encoder=a.encoder,
        usage=a.usage,
        cpu_used=a.cpu_used,
        svt_preset=a.svt_preset,
        svt_params=a.svt_params,
        crf=a.crf,
        qp=a.qp,
        hw_device=a.hw_device,
        fps=a.fps,
        threads=a.threads,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="av1sfm", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("encode", help="image sequence -> streaming AV1 IVF")
    p.add_argument("image_dir", type=Path)
    p.add_argument("out_ivf", type=Path)
    _add_encode_args(p)

    p = sub.add_parser("match", help="IVF MVs -> tracks -> COLMAP database")
    p.add_argument("image_dir", type=Path, help="frames in display order (sorted by name)")
    p.add_argument("database", type=Path)
    p.add_argument("--ivf", type=Path, required=True)
    p.add_argument("--encode", action="store_true", help="(re)encode image_dir to --ivf first")
    p.add_argument("--eps", type=float, default=0.1, help="cosine tolerance; 1 disables")
    p.add_argument("--tau", type=float, default=2.0, help="min MV length (px) for the cosine test")
    p.add_argument("--min-length", type=int, default=3)
    p.add_argument(
        "--on-violation",
        choices=["cut", "split", "drop"],
        default="cut",
        help="cut: terminate the track (paper); split: continue as a new track",
    )
    p.add_argument(
        "--grid",
        choices=["block", "cell"],
        default="block",
        help="keypoints per coded block, or per 4x4 unit (zero-order hold)",
    )
    p.add_argument(
        "--seed",
        choices=["all", "uncovered"],
        default="all",
        help="all: every block of every frame starts a track (paper)",
    )
    p.add_argument("--prev-only", action="store_true", help="only MVs pointing to frame n-1")
    p.add_argument("--keep-zero", action="store_true", help="keep (0,0) MVs")
    p.add_argument("--max-pair-gap", type=int, default=None)
    p.add_argument("--two-view", choices=["verify", "trust"], default="verify")
    p.add_argument("--camera-model", default="SIMPLE_RADIAL")
    p.add_argument("--camera-params", default=None, help="comma-separated, COLMAP order")
    p.add_argument("--stats", type=Path, default=None, help="write run statistics (JSON)")
    _add_encode_args(p)

    p = sub.add_parser("score", help="pairwise E/H scoring of raw matches in a database")
    p.add_argument("database", type=Path)
    p.add_argument("--out", type=Path, default=None, help="JSON with per-pair scores + summary")
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--repeats", type=int, default=1, help="RANSAC runs per pair (median)")

    p = sub.add_parser("encoders", help="list AV1 encoders and test which work here")
    _add_encode_args(p)

    p = sub.add_parser("validate-warp", help="check MV conventions by warping references")
    p.add_argument("ivf", type=Path)
    p.add_argument("image_dir", type=Path)
    p.add_argument("--frames", type=int, default=10)

    a = ap.parse_args(argv)

    if a.cmd == "encode":
        cmd = encode_images(list_images(a.image_dir), a.out_ivf, _encode_params(a))
        print(" ".join(cmd))

    elif a.cmd == "match":
        cfg = MVMatchConfig(
            track=TrackParams(
                eps=a.eps,
                tau=a.tau,
                min_length=a.min_length,
                on_violation=a.on_violation,
                grid=a.grid,
                seed=a.seed,
                skip_zero=not a.keep_zero,
                prev_only=a.prev_only,
            ),
            encode=_encode_params(a),
            camera=CameraSpec(a.camera_model, _params(a.camera_params)),
            max_pair_gap=a.max_pair_gap,
            two_view=a.two_view,
        )
        stats = run_mv_matching(a.image_dir, a.database, a.ivf, cfg, encode=a.encode)
        stats["matches"] = matches_per_image(a.database)
        text = json.dumps(stats, indent=2)
        if a.stats:
            a.stats.write_text(text)
        print(text)

    elif a.cmd == "score":
        ckpt = a.out.with_suffix(".partial.jsonl") if a.out else None
        scores, summary = score_database(
            a.database, RansacSettings(repeats=a.repeats), a.max_pairs, checkpoint=ckpt
        )
        if a.out:
            a.out.write_text(
                json.dumps({"summary": summary, "pairs": [s.asdict() for s in scores]}, indent=1)
            )
            ckpt.unlink(missing_ok=True)
        print(json.dumps(summary, indent=2))

    elif a.cmd == "encoders":
        ff = find_ffmpeg()
        print(f"ffmpeg: {ff}")
        compiled = available_encoders(ff)
        for backend in BACKENDS:
            if backend not in compiled:
                print(f"  {backend:7s} not compiled into this ffmpeg")
                continue
            params = _encode_params(a)
            params.encoder = backend
            ok, err = probe_encoder(params, ff, verbose=True)
            print(f"  {backend:7s} {'works' if ok else 'FAILS: ' + err}")

    elif a.cmd == "validate-warp":
        from .extract import iter_frame_motion
        from .validate import score_frame

        paths = list_images(a.image_dir)
        imgs: dict[int, np.ndarray] = {}
        for fm in iter_frame_motion(a.ivf):
            if fm.index > a.frames:
                continue
            for k in [fm.index, *fm.ref_frame_index]:
                if k >= 0 and k not in imgs:
                    imgs[k] = cv2.imread(str(paths[k]), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            s = score_frame(fm, imgs)
            if s:
                print(
                    f"frame {s.frame:4d} coverage {s.coverage:.2f}  MAE warp {s.mae_warp:6.2f}"
                    f"  identity {s.mae_identity:6.2f}  flipped {s.mae_flipped:6.2f}"
                )


if __name__ == "__main__":
    main()
