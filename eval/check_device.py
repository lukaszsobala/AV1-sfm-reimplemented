"""Compare the PyTorch matchers on a GPU against the CPU on two images.

    python eval/check_device.py runs/kitti117/img/000000.png runs/kitti117/img/000001.png

Reports: exact SIFT matching (must be identical), the DISK score map and
keypoints, and LightGlue matches on identical inputs (small differences are
expected from floating-point arithmetic; large ones mean a device problem).
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pycolmap

from av1sfm.devices import DEVICES, device_name, pick_device
from av1sfm.learned import DiskLightGlue, Features
from av1sfm.sift_exact import match_descriptors


def sift_descriptors(images: list[Path]) -> list[np.ndarray]:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db.db"
        opts = pycolmap.FeatureExtractionOptions()
        opts.sift.max_num_features = 8192
        pycolmap.extract_features(
            db, images[0].parent, [p.name for p in images], extraction_options=opts
        )
        with pycolmap.Database.open(db) as d:
            ids = {im.name: im.image_id for im in d.read_all_images()}
            return [np.array(d.read_descriptors(ids[p.name]).data) for p in images]


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = {tuple(r) for r in a.tolist()}, {tuple(r) for r in b.tolist()}
    return len(sa & sb) / max(len(sa | sb), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image0", type=Path)
    ap.add_argument("image1", type=Path)
    ap.add_argument("--device", choices=DEVICES, default="auto")
    a = ap.parse_args()
    dev, cpu = pick_device(a.device), pick_device("cpu")
    print(f"device: {device_name(dev)}")

    d0, d1 = sift_descriptors([a.image0, a.image1])
    m_cpu = match_descriptors(d0, d1, device=cpu)
    try:
        m_dev = match_descriptors(d0, d1, device=dev)
        same = np.array_equal(m_cpu, m_dev)
        print(f"exact SIFT: {len(m_cpu)} matches on cpu, {len(m_dev)} on device, identical: {same}")
    except RuntimeError as e:
        print(f"exact SIFT: FAILED: {e}")

    nets = {"cpu": DiskLightGlue(cpu), "device": DiskLightGlue(dev)}
    torch = nets["cpu"].torch
    with torch.inference_mode():
        x = torch.rand(1, 3, 384, 1248)
        hc = nets["cpu"].disk.heatmap_and_dense_descriptors(x)[0]
        hd = nets["device"].disk.heatmap_and_dense_descriptors(x.to(dev))[0].cpu()
    print(
        f"DISK score map: max |cpu - device| = {(hc - hd).abs().max().item():.2e} "
        f"(scores span {hc.min().item():.1f} .. {hc.max().item():.1f})"
    )
    f = {k: (n.extract(a.image0), n.extract(a.image1)) for k, n in nets.items()}
    kc, kd = (f[k][0].keypoints.cpu().numpy() for k in ("cpu", "device"))
    print(f"DISK keypoints: {len(kc)} cpu, {len(kd)} device, same positions: {overlap(kc, kd):.3f}")

    def on(dev_, feats: Features) -> Features:
        return Features(feats.keypoints.to(dev_), feats.descriptors.to(dev_), feats.size)

    # LightGlue on identical (CPU-extracted) features.
    fc0, fc1 = f["cpu"]
    lc = nets["cpu"].match(fc0, fc1)
    ld = nets["device"].match(on(dev, fc0), on(dev, fc1))
    print(
        f"LightGlue (same input): {len(lc)} matches cpu, {len(ld)} device, "
        f"same matches: {overlap(lc, ld):.3f}"
    )


if __name__ == "__main__":
    main()
