"""Collect run statistics (runs/<set>/*.json) into markdown tables.

uv run python eval/make_table.py runs > runs/results.md
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

METHODS = [
    ("mv", "AV1 MV (COLMAP verification)"),
    ("mv_trust", "AV1 MV (MVs trusted, no verification)"),
    ("sift_seq", "SIFT sequential (overlap 10)"),
    ("sift_exh", "SIFT exhaustive"),
]


def load(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def prestage(stats: dict) -> tuple[float, float, float]:
    """(wall s, cpu %, encode wall s) of the matching pre-stage, excluding encoding."""
    t = {k: v for k, v in stats["timings"].items() if k != "encode"}
    wall = sum(v["wall_s"] for v in t.values())
    cpu = sum(v["cpu_s"] for v in t.values())
    enc = stats["timings"].get("encode", {}).get("wall_s", float("nan"))
    return wall, 100 * cpu / wall if wall else 0.0, enc


def fmt(x, nd=2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return f"{x:,}"


def main(root: Path) -> None:
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "img").exists()):
        n_img = len(list((d / "img").iterdir()))
        print(f"## {d.name} ({n_img} frames)\n")
        print(
            "| Method | Pre-stage wall (s) | Avg CPU % | Encode (s) | Keypoints / img "
            "| Raw matches / img | Verified matches / img | Scored pairs | Inlier ratio "
            "| Median Sampson (px) |"
        )
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for key, label in METHODS:
            st = load(d / f"{key}.json")
            if not st:
                continue
            wall, cpu, enc = prestage(st)
            m = st["matches"]
            sc = load(d / f"score_{key}.json")
            s = sc["summary"] if sc else {}
            print(
                f"| {label} | {fmt(wall, 1)} | {fmt(cpu, 0)} | {fmt(enc, 1)} "
                f"| {fmt(m['keypoints_per_image'], 0)} | {fmt(m['raw_matches_per_image'], 0)} "
                f"| {fmt(m['verified_matches_per_image'], 0)} | {fmt(s.get('pairs'))} "
                f"| {fmt(s.get('inlier_ratio_pooled'), 3)} "
                f"| {fmt(s.get('sampson_px_median_of_pairs'), 3)} |"
            )
        maps = [(label, load(d / f"map_{key}.json")) for key, label in METHODS]
        if any(m for _, m in maps):
            print(
                "\n| Method | Registered | 3D points | Reproj. error (px) | Mean track length "
                "| Mapper wall (s) | Final global BA (s) |"
            )
            print("|---|---:|---:|---:|---:|---:|---:|")
            for label, m in maps:
                if not m:
                    continue
                if not m.get("num_models"):
                    print(
                        f"| {label} | 0 | – | – | – | {fmt(m['timings']['mapper']['wall_s'], 1)} | – |"
                    )
                    continue
                print(
                    f"| {label} | {m['registered_images']}/{n_img} | {fmt(m['points3D'])} "
                    f"| {fmt(m['mean_reprojection_error_px'], 3)} "
                    f"| {fmt(m['mean_track_length'], 2)} "
                    f"| {fmt(m['timings']['mapper']['wall_s'], 1)} "
                    f"| {fmt(m['final_global_ba_s'], 1)} |"
                )
        print()


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "runs"))
