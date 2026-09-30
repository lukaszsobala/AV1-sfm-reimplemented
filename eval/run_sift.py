"""COLMAP SIFT baseline (exhaustive or sequential matching) with shared settings.

Uses the same single shared camera and the same two-view verification options
as the MV pipeline (av1sfm.colmap_db.two_view_options). Runs on the CPU unless
pycolmap was built with CUDA.

    uv run python eval/run_sift.py IMAGES DB --matching exhaustive --stats out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pycolmap

from av1sfm.colmap_db import two_view_options
from av1sfm.encode import list_images
from av1sfm.pipeline import matches_per_image
from av1sfm.timing import Timer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image_dir", type=Path)
    ap.add_argument("database", type=Path)
    ap.add_argument("--matching", choices=["exhaustive", "sequential"], required=True)
    ap.add_argument("--overlap", type=int, default=10, help="sequential: neighbours per image")
    ap.add_argument("--max-features", type=int, default=8192)
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
    device = pycolmap.Device.auto

    timer = Timer()
    with timer.stage("extract"):
        pycolmap.extract_features(a.database, a.image_dir, names, pycolmap.CameraMode.SINGLE,
                                  reader, extraction, device)  # fmt: skip
    with timer.stage("match"):
        if a.matching == "exhaustive":
            pycolmap.match_exhaustive(a.database, matching, pycolmap.ExhaustivePairingOptions(),
                                      two_view_options(), device)  # fmt: skip
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
            "camera": [a.camera_model, a.camera_params],
            "cuda": pycolmap.has_cuda,
        },  # fmt: skip
        "timings": timer.asdict(),
        "matches": matches_per_image(a.database),
    }
    text = json.dumps(stats, indent=2)
    if a.stats:
        a.stats.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
