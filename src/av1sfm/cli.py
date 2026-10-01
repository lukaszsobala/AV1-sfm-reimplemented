"""Command-line interface: `av1sfm {reconstruct,encode,match,score,validate-warp,encoders}`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .colmap_db import CameraSpec
from .devices import DEVICES
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


def _add_track_args(p) -> None:
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


def _add_camera_args(p) -> None:
    p.add_argument("--camera-model", default="SIMPLE_RADIAL")
    p.add_argument("--camera-params", default=None, help="comma-separated, COLMAP order")


def _mv_config(a: argparse.Namespace) -> MVMatchConfig:
    return MVMatchConfig(
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
    _add_track_args(p)
    _add_camera_args(p)
    p.add_argument("--stats", type=Path, default=None, help="write run statistics (JSON)")
    _add_encode_args(p)

    p = sub.add_parser(
        "reconstruct",
        help="images or a video -> matches -> sparse reconstruction -> point cloud",
        description="Images or a video to a COLMAP reconstruction and a PLY point cloud. "
        "An AV1 video's stream is copied without re-encoding; other inputs are encoded "
        "with --encoder. Outputs: OUT_DIR/{images/, clip.ivf, database.db, sparse/0/, "
        "points.ply, reconstruct.json}.",
    )
    p.add_argument("input", type=Path, help="folder of frames (sorted by name) or a video file")
    p.add_argument("out_dir", type=Path)
    p.add_argument(
        "--matcher",
        choices=["mv", "sift"],
        default="mv",
        help="mv: AV1 motion vectors; sift: COLMAP SIFT features (see --sift-*)",
    )
    p.add_argument(
        "--ivf", type=Path, default=None, help="image folder input: its AV1 stream (no encoding)"
    )
    p.add_argument(
        "--reencode",
        action="store_true",
        help="AV1 video: encode the frames (low delay) instead of copying the stream; slower, "
        "but random-access videos give shorter tracks than a low-delay stream",
    )
    p.add_argument(
        "--frame-format", choices=["png", "jpg"], default="png", help="frames from a video"
    )
    p.add_argument("--export-dataset", action="store_true", help="also OUT_DIR/dataset/")
    g = p.add_argument_group("SIFT matcher")
    g.add_argument("--sift-matching", choices=["sequential", "exhaustive"], default="sequential")
    g.add_argument("--sift-overlap", type=int, default=10, help="sequential: neighbours per image")
    g.add_argument(
        "--sift-matcher",
        choices=["exact", "colmap"],
        default="exact",
        help="exact: exact matching on a PyTorch device (needs PyTorch); colmap: COLMAP's",
    )
    g.add_argument("--device", choices=DEVICES, default="auto", help="PyTorch device for exact")
    g.add_argument("--max-features", type=int, default=8192)
    g = p.add_argument_group("mapper")
    g.add_argument("--mapper", choices=["incremental", "global"], default="incremental")
    g.add_argument("--fix-intrinsics", action="store_true", help="do not refine the camera")
    g.add_argument("--init-max-forward-motion", type=float, default=0.95, help="1.0 for driving")
    g.add_argument("--init-min-tri-angle", type=float, default=16.0, help="4 for driving")
    g.add_argument(
        "--prune-redundant-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="global BAs skip redundant 3D points (incremental mapper)",
    )
    g.add_argument("--global-tracks-per-view", type=int, default=None)
    _add_track_args(p.add_argument_group("MV tracks"))
    _add_camera_args(p)
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
        stats = run_mv_matching(a.image_dir, a.database, a.ivf, _mv_config(a), encode=a.encode)
        stats["matches"] = matches_per_image(a.database)
        text = json.dumps(stats, indent=2)
        if a.stats:
            a.stats.write_text(text)
        print(text)

    elif a.cmd == "reconstruct":
        from .reconstruct import ReconstructConfig, reconstruct

        cfg = ReconstructConfig(
            matcher=a.matcher,
            mv=_mv_config(a),
            ivf=a.ivf,
            reencode=a.reencode,
            sift_matching=a.sift_matching,
            sift_overlap=a.sift_overlap,
            sift_matcher=a.sift_matcher,
            device=a.device,
            max_features=a.max_features,
            mapper=a.mapper,
            fix_intrinsics=a.fix_intrinsics,
            init_max_forward_motion=a.init_max_forward_motion,
            init_min_tri_angle=a.init_min_tri_angle,
            prune=a.prune_redundant_points,
            global_tracks_per_view=a.global_tracks_per_view,
            export_dataset=a.export_dataset,
            frame_format=a.frame_format,
        )
        st = reconstruct(a.input, a.out_dir, cfg)
        for name, t in st["timings"].items():
            print(f"  {name:10s} {t['wall_s']:9.1f} s")
        print(
            f"{st['registered_images']}/{st['num_images']} images registered, "
            f"{st.get('points3D', 0):,} points, "
            f"{st.get('mean_reprojection_error_px', float('nan')):.3f} px, "
            f"{st['total_wall_s']:.1f} s; see {a.out_dir / 'reconstruct.json'}"
        )

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
