"""Collect run statistics (runs/<set>/*.json) into markdown tables.

uv run python eval/make_table.py runs > runs/results.md
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

METHODS = [
    ("mv", "AV1 MV, libaom (COLMAP verification)"),
    ("mv_trust", "AV1 MV, libaom (MVs trusted, no verification)"),
    ("mv_svt", "AV1 MV, SVT-AV1 (COLMAP verification)"),
    ("mv_qsv", "AV1 MV, Intel QSV (COLMAP verification)"),
    ("mv_vaapi", "AV1 MV, VA-API (COLMAP verification)"),
    ("mv_vulkan", "AV1 MV, Vulkan Video (COLMAP verification)"),
    ("sift_seq", "SIFT sequential (overlap 10)"),
    ("sift_exh", "SIFT exhaustive"),
]


def load(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def stages(stats: dict) -> tuple[float, float, float, int]:
    """(pre-processing s, feature matching s, CPU % of one core, cores) as in the paper's
    Table II: pre-processing = video encoding (MV) or SIFT extraction; feature
    matching = everything else before mapping. CPU % covers both stages."""
    t = stats["timings"]
    pre_keys = {"encode"} if stats.get("method", "").startswith("av1") else {"extract"}
    pre = sum(v["wall_s"] for k, v in t.items() if k in pre_keys) if pre_keys & t.keys() else None
    match = sum(v["wall_s"] for k, v in t.items() if k not in pre_keys)
    wall = sum(v["wall_s"] for v in t.values())
    cpu = 100 * sum(v["cpu_s"] for v in t.values()) / wall if wall else 0.0
    return pre, match, cpu, stats.get("cpu_count") or os.cpu_count() or 1


def fmt(x, nd=2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return f"{x:,}"


def sci(x) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2e}"


def main(root: Path) -> None:
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "img").exists()):
        n_img = len(list((d / "img").iterdir()))
        print(f"## {d.name} ({n_img} frames)\n")
        print(
            "| Method | Pre-processing (s) | Feature matching (s) | CPU % (1 core = 100) "
            "| CPU % of machine | Keypoints / img | Raw matches / img | Verified matches / img "
            "| Scored pairs | Inlier ratio | Median Sampson (px) | Median SE (normalised²) |"
        )
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for key, label in METHODS:
            st = load(d / f"{key}.json")
            if not st:
                continue
            pre, match, cpu, cores = stages(st)
            m = st["matches"]
            sc = load(d / f"score_{key}.json")
            s = sc["summary"] if sc else {}
            print(
                f"| {label} | {fmt(pre, 1)} | {fmt(match, 1)} | {fmt(cpu, 0)} "
                f"| {fmt(cpu / cores, 1)} "
                f"| {fmt(m['keypoints_per_image'], 0)} | {fmt(m['raw_matches_per_image'], 0)} "
                f"| {fmt(m['verified_matches_per_image'], 0)} | {fmt(s.get('pairs'))} "
                f"| {fmt(s.get('inlier_ratio_pooled'), 3)} "
                f"| {fmt(s.get('sampson_px_median_of_pairs'), 3)} "
                f"| {sci(s.get('sampson_sq_norm_median_of_pairs'))} |"
            )
        for prefix, title in (
            ("map_", "intrinsics fixed at calibration"),
            ("map_refine_", "COLMAP default intrinsics refinement"),
        ):
            sfm_table(d, n_img, prefix, title)
        print()


def sfm_table(d: Path, n_img: int, prefix: str, title: str) -> None:
    maps = [(label, load(d / f"{prefix}{key}.json")) for key, label in METHODS]
    if not any(m for _, m in maps):
        return
    print(f"\nSfM, {title}:\n")
    print(
        "| Method | Registered | 3D points | Reproj. error (px) | Mean track length "
        "| Mapper wall (s) | Final global BA (s) |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for label, m in maps:
        if not m:
            continue
        wall = fmt(m["timings"]["mapper"]["wall_s"], 1)
        if not m.get("num_models"):
            print(f"| {label} | 0/{n_img} | – | – | – | {wall} | – |")
            continue
        print(
            f"| {label} | {m['registered_images']}/{n_img} | {fmt(m['points3D'])} "
            f"| {fmt(m['mean_reprojection_error_px'], 3)} | {fmt(m['mean_track_length'], 2)} "
            f"| {wall} | {fmt(m['final_global_ba_s'], 1)} |"
        )


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "runs"))
