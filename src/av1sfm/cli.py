"""Command-line interface: `av1sfm {encode,match,score,validate-warp}`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .colmap_db import CameraSpec
from .encode import EncodeParams, encode_images, list_images
from .geometry import RansacSettings
from .pipeline import MVMatchConfig, matches_per_image, run_mv_matching, score_database
from .tracks import TrackParams


def _params(s: str | None) -> tuple[float, ...] | None:
    return tuple(float(v) for v in s.split(",")) if s else None


def _add_encode_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("encoding")
    g.add_argument("--encoder", default="libaom-av1", choices=["libaom-av1", "av1_nvenc"])
    g.add_argument("--cpu-used", type=int, default=6)
    g.add_argument("--crf", type=int, default=30)
    g.add_argument("--fps", type=int, default=10)
    g.add_argument("--threads", type=int, default=0)


def _encode_params(a: argparse.Namespace) -> EncodeParams:
    return EncodeParams(encoder=a.encoder, cpu_used=a.cpu_used, crf=a.crf, fps=a.fps,
                        threads=a.threads)  # fmt: skip


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
    p.add_argument("--tau", type=float, default=1.0, help="min MV length (px) for the cosine test")
    p.add_argument("--min-length", type=int, default=3)
    p.add_argument("--on-violation", choices=["split", "drop"], default="split")
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
                skip_zero=not a.keep_zero,
                prev_only=a.prev_only,
            ),  # fmt: skip
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
        scores, summary = score_database(a.database, RansacSettings(), a.max_pairs)
        if a.out:
            a.out.write_text(json.dumps({"summary": summary,
                                         "pairs": [s.asdict() for s in scores]}, indent=1))  # fmt: skip
        print(json.dumps(summary, indent=2))

    elif a.cmd == "validate-warp":
        from .extract import iter_frame_motion
        from .validate import score_frame

        paths = list_images(a.image_dir)
        imgs: dict[int, np.ndarray] = {}
        for fm in iter_frame_motion(a.ivf):
            if fm.index > a.frames:
                break
            for k in [fm.index, *fm.ref_frame_index]:
                if k not in imgs:
                    imgs[k] = cv2.imread(str(paths[k]), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            s = score_frame(fm, imgs)
            if s:
                print(f"frame {s.frame:4d} coverage {s.coverage:.2f}  MAE warp {s.mae_warp:6.2f}"
                      f"  identity {s.mae_identity:6.2f}  flipped {s.mae_flipped:6.2f}")  # fmt: skip


if __name__ == "__main__":
    main()
