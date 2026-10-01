"""COLMAP SIFT baseline (exhaustive or sequential matching) with shared settings.

Uses the same single shared camera and the same two-view verification options
as the MV pipeline (av1sfm.colmap_db.two_view_options). Extraction runs on the
CPU unless pycolmap was built with CUDA. Matching uses COLMAP's own matcher, or
with `--matcher exact` the exact nearest-neighbour search of av1sfm.sift_exact
on a PyTorch device (Intel GPU, CUDA or CPU), followed by the same COLMAP
geometric verification.

    uv run python eval/run_sift.py IMAGES DB --matching exhaustive --stats out.json
    python eval/run_sift.py IMAGES DB --matching sequential --matcher exact --device xpu
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pycolmap

from av1sfm.colmap_db import two_view_options
from av1sfm.devices import DEVICES, device_name, pick_device, synchronize
from av1sfm.encode import list_images
from av1sfm.pipeline import matches_per_image
from av1sfm.sift_exact import generate_pairs, match_database, match_descriptors
from av1sfm.timing import Timer


def exact_match(database, matching, overlap, device, sift) -> None:
    """Exact matching of the stored descriptors, then COLMAP's geometric verification."""
    with pycolmap.Database.open(database) as db:
        pairs = generate_pairs(db, matching, overlap)
    match_database(database, pairs, device, sift)
    synchronize(device)
    pycolmap.geometric_verification(
        database,
        pycolmap.GeometricVerifierOptions(),
        pycolmap.ExistingMatchedPairingOptions(),
        two_view_options(),
    )


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

    if a.database.exists():
        a.database.unlink()
    names = [p.name for p in list_images(a.image_dir)]
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = a.camera_model
    reader.camera_params = a.camera_params
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.max_num_features = a.max_features
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = a.num_threads
    matching.sift.cpu_brute_force_matcher = a.brute_force
    device = pycolmap.Device.auto
    torch_device = pick_device(a.device) if a.matcher == "exact" else None
    if torch_device is not None:  # not timed: first-call kernel compilation on GPUs
        warm = np.random.default_rng(0).integers(0, 64, (512, 128), dtype=np.uint8)
        match_descriptors(warm, warm, device=torch_device)

    timer = Timer()
    with timer.stage("extract"):
        pycolmap.extract_features(
            a.database, a.image_dir, names, pycolmap.CameraMode.SINGLE, reader, extraction, device
        )
    with timer.stage("match"):
        if torch_device is not None:
            exact_match(a.database, a.matching, a.overlap, torch_device, matching.sift)
        elif a.matching == "exhaustive":
            pycolmap.match_exhaustive(
                a.database,
                matching,
                pycolmap.ExhaustivePairingOptions(),
                two_view_options(),
                device,
            )
        else:
            pairing = pycolmap.SequentialPairingOptions()
            pairing.overlap = a.overlap
            pairing.quadratic_overlap = False
            pycolmap.match_sequential(a.database, matching, pairing, two_view_options(), device)

    stats = {
        "method": f"sift-{a.matching}",
        "config": {
            "max_features": a.max_features,
            "overlap": a.overlap,
            "matcher": a.matcher,
            "device": device_name(torch_device) if torch_device is not None else None,
            "num_threads": a.num_threads,
            "brute_force": a.brute_force,
            "camera": [a.camera_model, a.camera_params],
            "cuda": pycolmap.has_cuda,
        },
        "timings": timer.asdict(),
        "cpu_count": os.cpu_count(),
        "matches": matches_per_image(a.database),
    }
    text = json.dumps(stats, indent=2)
    if a.stats:
        a.stats.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
