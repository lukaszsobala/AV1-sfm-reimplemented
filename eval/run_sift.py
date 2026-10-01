"""COLMAP SIFT baseline (exhaustive or sequential matching) with shared settings.

Uses the same single shared camera and the same two-view verification options
as the MV pipeline (av1sfm.colmap_db.two_view_options); see av1sfm.sift.
Extraction runs on the CPU unless pycolmap was built with CUDA. Matching uses
COLMAP's own matcher, or with `--matcher exact` the exact nearest-neighbour
search of av1sfm.sift_exact on a PyTorch device (Intel GPU, CUDA or CPU),
followed by the same COLMAP geometric verification.

    uv run python eval/run_sift.py IMAGES DB --matching exhaustive --stats out.json
    python eval/run_sift.py IMAGES DB --matching sequential --matcher exact --device xpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from av1sfm.colmap_db import CameraSpec
from av1sfm.devices import DEVICES
from av1sfm.sift import run_sift_matching


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image_dir", type=Path)
    ap.add_argument("database", type=Path)
    ap.add_argument("--matching", choices=["exhaustive", "sequential"], required=True)
    ap.add_argument("--overlap", type=int, default=10, help="sequential: neighbours per image")
    ap.add_argument("--max-features", type=int, default=8192)
    ap.add_argument(
        "--num-threads",
        type=int,
        default=-1,
        help="matching threads (-1: all cores). COLMAP's default approximate CPU matcher "
        "returns different matches for different thread counts (ASSUMPTIONS.md R4)",
    )
    ap.add_argument(
        "--brute-force",
        action="store_true",
        help="exact CPU matching (deterministic, ~40x slower than the default)",
    )
    ap.add_argument(
        "--matcher",
        choices=["colmap", "exact"],
        default="colmap",
        help="colmap: COLMAP's matcher; exact: exact matching on a PyTorch device, "
        "identical to COLMAP's --brute-force result (needs PyTorch)",
    )
    ap.add_argument("--device", choices=DEVICES, default="auto", help="--matcher exact only")
    ap.add_argument("--camera-model", default="SIMPLE_RADIAL")
    ap.add_argument("--camera-params", default="")
    ap.add_argument("--stats", type=Path, default=None)
    a = ap.parse_args()

    params = tuple(float(v) for v in a.camera_params.split(",")) if a.camera_params else None
    stats = run_sift_matching(
        a.image_dir,
        a.database,
        matching=a.matching,
        overlap=a.overlap,
        matcher=a.matcher,
        device=a.device,
        camera=CameraSpec(a.camera_model, params),
        max_features=a.max_features,
        num_threads=a.num_threads,
        brute_force=a.brute_force,
    )
    text = json.dumps(stats, indent=2)
    if a.stats:
        a.stats.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
