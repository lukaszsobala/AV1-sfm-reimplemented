"""DISK + LightGlue baseline on a PyTorch device (Intel GPU, CUDA or CPU).

The paper's Table I compares against DISK [Tyszkiewicz et al. 2020] features
matched with LightGlue [Lindenberger et al. 2023] on a GPU, without giving
settings. We use kornia's ports of both (Apache-2.0): DISK with the `depth`
weights, non-maximum suppression window 5, no score threshold, at most
`max_keypoints` per image (hloc's DISK default, 5000); LightGlue with the
`disk_lightglue` weights and its default adaptive depth / width and match
threshold (0.1). Weights are downloaded on first use into PyTorch's hub cache.

Keypoints are written to the COLMAP database with +0.5 px (DISK returns pixel
indices; COLMAP puts the centre of the top-left pixel at (0.5, 0.5)). Raw
matches then go through COLMAP's geometric verification like every other method.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .devices import import_torch

MAX_KEYPOINTS = 5000


@dataclass
class Features:
    keypoints: object  # (n, 2) float tensor on the device, DISK convention (pixel indices)
    descriptors: object  # (n, 128) float tensor on the device
    size: tuple[int, int]  # (width, height)

    def colmap_keypoints(self) -> np.ndarray:
        return np.ascontiguousarray(self.keypoints.cpu().numpy() + 0.5, np.float32)


class DiskLightGlue:
    def __init__(self, device, max_keypoints: int = MAX_KEYPOINTS):
        torch = import_torch()
        try:
            from kornia.feature import DISK, LightGlue
            from kornia.feature.disk.detector import heatmap_to_keypoints
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError("DISK + LightGlue needs kornia: uv pip install kornia") from e
        self.torch, self.device, self.max_keypoints = torch, device, max_keypoints
        self.heatmap_to_keypoints = heatmap_to_keypoints
        self.disk = DISK.from_pretrained("depth", device=device).eval()
        self.lightglue = LightGlue("disk").to(device).eval()
        # kornia has no point-pruning threshold for Intel GPUs: use its CUDA value.
        thresholds = dict(LightGlue.pruning_keypoint_thresholds)
        thresholds.setdefault("xpu", thresholds["cuda"])
        self.lightglue.pruning_keypoint_thresholds = thresholds

    def extract(self, image: Path) -> Features:
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image)
        torch = self.torch
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        h, w = rgb.shape[:2]
        t = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1)[None].float() / 255.0
        t = torch.nn.functional.pad(t, (0, -w % 16, 0, -h % 16))  # DISK needs multiples of 16
        with torch.inference_mode():
            heatmap, dense = self.disk.heatmap_and_dense_descriptors(t)
            # Keypoint selection (NMS, top-n) on the CPU: on PyTorch XPU 2.14 the
            # boolean indexing in kornia's selection gives inconsistent sizes.
            kp = self.heatmap_to_keypoints(
                heatmap[:, :, :h, :w].float().cpu(), n=self.max_keypoints, window_size=5
            )[0]
            xy = kp.xys.to(self.device)
            desc = dense[0][:, xy[:, 1], xy[:, 0]].T
            desc = torch.nn.functional.normalize(desc, dim=-1)
        return Features(xy.float(), desc.float(), (w, h))

    def match(self, f0: Features, f1: Features) -> np.ndarray:
        """Raw matches (k, 2) uint32, indices into f0 / f1."""
        if len(f0.keypoints) == 0 or len(f1.keypoints) == 0:
            return np.zeros((0, 2), np.uint32)
        torch = self.torch

        def item(f: Features) -> dict:
            return {
                "keypoints": f.keypoints[None],
                "descriptors": f.descriptors[None],
                "image_size": torch.tensor([f.size], device=self.device, dtype=torch.float32),
            }

        with torch.inference_mode():
            out = self.lightglue({"image0": item(f0), "image1": item(f1)})
        return np.ascontiguousarray(out["matches"][0].cpu().numpy(), np.uint32).reshape(-1, 2)
