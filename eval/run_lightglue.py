"""DISK + LightGlue baseline (exhaustive or sequential pairs) with shared settings.

Same database layout, single shared camera, pair generation and COLMAP
geometric verification as eval/run_sift.py; features and matches come from
av1sfm.learned on a PyTorch device (Intel GPU, CUDA, else CPU). Needs PyTorch
and kornia (README, "GPU matchers").

    python eval/run_lightglue.py IMAGES DB --matching sequential --device xpu --stats out.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pycolmap

from av1sfm.colmap_db import CameraSpec, create_database, two_view_options
from av1sfm.devices import DEVICES, device_name, pick_device, synchronize
from av1sfm.encode import list_images
from av1sfm.learned import MAX_KEYPOINTS, DiskLightGlue
from av1sfm.pipeline import matches_per_image
from av1sfm.sift_exact import generate_pairs
from av1sfm.timing import Timer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image_dir", type=Path)
    ap.add_argument("database", type=Path)
    ap.add_argument("--matching", choices=["exhaustive", "sequential"], required=True)
    ap.add_argument("--overlap", type=int, default=10, help="sequential: neighbours per image")
    ap.add_argument("--max-keypoints", type=int, default=MAX_KEYPOINTS)
    ap.add_argument("--device", choices=DEVICES, default="auto")
    ap.add_argument(
        "--fp32-attention", action="store_true", help="GPU: attention in float32, not float16"
    )
    ap.add_argument("--no-pruning", action="store_true", help="disable LightGlue's point pruning")
    ap.add_argument("--cpu-softmax", action="store_true", help="GPU: dual softmax on the CPU")
    ap.add_argument("--camera-model", default="SIMPLE_RADIAL")
    ap.add_argument("--camera-params", default="")
    ap.add_argument("--stats", type=Path, default=None)
    a = ap.parse_args()

    device = pick_device(a.device)
    params = tuple(float(v) for v in a.camera_params.split(",")) if a.camera_params else None
    images = list_images(a.image_dir)
    ids = create_database(
        a.database, a.image_dir, [p.name for p in images], CameraSpec(a.camera_model, params)
    )
    model = DiskLightGlue(  # not timed: one-off load / download
        device,
        a.max_keypoints,
        half_attention=False if a.fp32_attention else None,
        prune=not a.no_pruning,
        assignment_on_device=False if a.cpu_softmax else None,
    )
    if device.type != "cpu":  # not timed: first-call kernel compilation on GPUs
        f = model.extract(images[0])
        model.match(f, f)

    timer = Timer()
    features = {}
    with timer.stage("extract"):
        with pycolmap.Database.open(a.database) as db, pycolmap.DatabaseTransaction(db):
            for p in images:
                f = model.extract(p)
                features[ids[p.name]] = f
                db.write_keypoints(ids[p.name], f.colmap_keypoints())
        synchronize(device)
    with timer.stage("match"):
        with pycolmap.Database.open(a.database) as db:
            pairs = generate_pairs(db, a.matching, a.overlap)
            with pycolmap.DatabaseTransaction(db):
                for i, j in pairs:
                    db.write_matches(i, j, model.match(features[i], features[j]))
        synchronize(device)
        pycolmap.geometric_verification(
            a.database,
            pycolmap.GeometricVerifierOptions(),
            pycolmap.ExistingMatchedPairingOptions(),
            two_view_options(),
        )

    stats = {
        "method": f"disk-lightglue-{a.matching}",
        "config": {
            "max_keypoints": a.max_keypoints,
            "overlap": a.overlap,
            "device": device_name(device),
            "attention": "float16" if model.half_attention else "float32",
            "pruning": model.prune,
            "pruning_min_keypoints": model.prune_min_keypoints,
            "dual_softmax": "device" if model.assignment_on_device else "cpu",
            "camera": [a.camera_model, a.camera_params],
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
