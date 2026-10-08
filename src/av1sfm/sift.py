"""COLMAP SIFT extraction and matching with the settings shared by all methods.

Uses one shared camera and the same two-view verification options as the MV
pipeline (`colmap_db.two_view_options`). Extraction runs on the CPU unless
pycolmap was built with CUDA. Matching uses COLMAP's own matcher, or with
`matcher="exact"` the exact nearest-neighbour search of `av1sfm.sift_exact`
on a PyTorch device (Intel GPU, CUDA or CPU), identical to COLMAP's
brute-force matcher, followed by the same COLMAP geometric verification.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
import pycolmap

from .colmap_db import CameraSpec, two_view_options, verify_pairs
from .encode import list_images
from .pipeline import matches_per_image
from .progress import SEQUENTIAL_MATCHING, SIFT_EXTRACTION, Progress, exhaustive_matching
from .timing import Timer


def run_sift_matching(
    image_dir: str | Path,
    db_path: str | Path,
    *,
    matching: Literal["sequential", "exhaustive"] = "sequential",
    overlap: int = 10,
    matcher: Literal["colmap", "exact"] = "colmap",
    device: str = "auto",
    camera: CameraSpec | None = None,
    max_features: int = 8192,
    num_threads: int = -1,
    brute_force: bool = False,
    progress: Progress | None = None,
) -> dict:
    """Extract and match SIFT features of all images into a new database; returns run statistics.

    `matching`: COLMAP's sequential pairs (`overlap` neighbours per image, no
    quadratic overlap) or all pairs. `num_threads` and `brute_force` apply to
    COLMAP's matcher: its default approximate matcher returns different matches
    for different thread counts (ASSUMPTIONS.md R4).
    """
    camera = camera or CameraSpec()
    progress = progress or Progress(enabled=False)
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()
    names = [p.name for p in list_images(image_dir)]
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = camera.model
    if camera.params is not None:
        reader.camera_params = ",".join(f"{v:.10g}" for v in camera.params)
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.max_num_features = max_features
    options = pycolmap.FeatureMatchingOptions()
    options.num_threads = num_threads
    options.sift.cpu_brute_force_matcher = brute_force
    torch_device = None
    if matcher == "exact":
        from .devices import device_name, pick_device
        from .sift_exact import match_descriptors

        torch_device = pick_device(device)
        # Not timed: first-call kernel compilation on GPUs.
        with progress.step(f"Preparing the SIFT matcher ({device_name(torch_device)})"):
            warm = np.random.default_rng(0).integers(0, 64, (512, 128), dtype=np.uint8)
            match_descriptors(warm, warm, device=torch_device)
    elif matcher != "colmap":
        raise ValueError(f"matcher must be 'colmap' or 'exact', got {matcher!r}")

    timer = Timer()
    with (
        timer.stage("extract"),
        progress.step(
            "Extracting SIFT features",
            total=len(names),
            unit="images",
            parse=SIFT_EXTRACTION,
        ),
    ):
        pycolmap.extract_features(
            db_path,
            image_dir,
            names,
            pycolmap.CameraMode.SINGLE,
            reader,
            extraction,
            pycolmap.Device.auto,
        )
    with timer.stage("match"):
        if torch_device is not None:
            _exact_match(db_path, matching, overlap, torch_device, options.sift, progress)
        elif matching == "exhaustive":
            with progress.step(
                "Matching and verifying all image pairs",
                parse=exhaustive_matching,
                unit="blocks",
            ) as step:
                pycolmap.match_exhaustive(
                    db_path,
                    options,
                    pycolmap.ExhaustivePairingOptions(),
                    two_view_options(),
                    pycolmap.Device.auto,
                )
                step.update(done=step.total or 0)
        else:
            pairing = pycolmap.SequentialPairingOptions()
            pairing.overlap = overlap
            pairing.quadratic_overlap = False
            with progress.step(
                f"Matching and verifying, {overlap} neighbours per image",
                total=len(names),
                unit="images",
                parse=SEQUENTIAL_MATCHING,
            ) as step:
                pycolmap.match_sequential(
                    db_path, options, pairing, two_view_options(), pycolmap.Device.auto
                )
                step.update(done=len(names))

    device_label = None
    if torch_device is not None:
        from .devices import device_name

        device_label = device_name(torch_device)
    return {
        "method": f"sift-{matching}",
        "config": {
            "max_features": max_features,
            "overlap": overlap,
            "matcher": matcher,
            "device": device_label,
            "num_threads": num_threads,
            "brute_force": brute_force,
            "camera": [camera.model, reader.camera_params],
            "cuda": pycolmap.has_cuda,
        },
        "timings": timer.asdict(),
        "cpu_count": os.cpu_count(),
        "matches": matches_per_image(db_path),
    }


def _exact_match(
    db_path: Path, matching: str, overlap: int, device, sift, progress: Progress
) -> None:
    """Exact matching of the stored descriptors, then COLMAP's geometric verification."""
    from .devices import device_name, synchronize
    from .sift_exact import generate_pairs, match_database

    with pycolmap.Database.open(db_path) as db:
        pairs = generate_pairs(db, matching, overlap)
    with progress.step(
        f"Matching SIFT features ({device_name(device)})",
        total=len(pairs),
        unit="pairs",
    ) as step:
        match_database(db_path, pairs, device, sift, progress=lambda done, _: step.update(done))
        synchronize(device)
    verify_pairs(db_path, two_view_options(), progress)
