"""DISK + LightGlue baseline on a PyTorch device (Intel GPU, CUDA or CPU).

The paper's Table I compares against DISK [Tyszkiewicz et al. 2020] features
matched with LightGlue [Lindenberger et al. 2023] on a GPU, without giving
settings. We use kornia's ports of both (Apache-2.0): DISK with the `depth`
weights, non-maximum suppression window 5, no score threshold, at most
`max_keypoints` per image (hloc's DISK default, 5000); LightGlue with the
`disk_lightglue` weights and its default adaptive depth / width and match
threshold (0.1). Weights are downloaded on first use into PyTorch's hub cache.

LightGlue runs through `DiskLightGlue.match`, which calls kornia's layers in
the order of kornia's forward pass with two changes: no point pruning (a speed
optimisation whose kept points depend on floating-point noise), and the final
mutual-match selection on the CPU. Both avoid boolean-mask indexing on the
device, which is broken on PyTorch XPU 2.14 (Intel GPUs): it returned
inconsistent sizes and out-of-bounds indices. Likewise, DISK's keypoint
selection runs on the CPU and descriptors are sampled with index_select.

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
            from kornia.feature.lightglue import normalize_keypoints
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError("DISK + LightGlue needs kornia: uv pip install kornia") from e
        self.torch, self.device, self.max_keypoints = torch, device, max_keypoints
        self.heatmap_to_keypoints = heatmap_to_keypoints
        self.normalize_keypoints = normalize_keypoints
        self.disk = DISK.from_pretrained("depth", device=device).eval()
        self.lightglue = LightGlue("disk").to(device).eval()

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
            xy = kp.xys
            flat = (xy[:, 1] * dense.shape[-1] + xy[:, 0]).to(self.device)
            desc = dense[0].flatten(1).index_select(1, flat).T
            desc = torch.nn.functional.normalize(desc, dim=-1)
        return Features(xy.float().to(self.device), desc.float(), (w, h))

    def match(self, f0: Features, f1: Features) -> np.ndarray:
        """Raw matches (k, 2) uint32, indices into f0 / f1."""
        if len(f0.keypoints) == 0 or len(f1.keypoints) == 0:
            return np.zeros((0, 2), np.uint32)
        torch, lg = self.torch, self.lightglue

        def prepare(f: Features):
            size = torch.tensor([f.size], device=self.device, dtype=torch.float32)
            kpts = self.normalize_keypoints(f.keypoints[None], size)
            return lg.input_proj(f.descriptors[None].contiguous()), lg.posenc(kpts)

        with torch.inference_mode():
            (d0, e0), (d1, e1) = prepare(f0), prepare(f1)
            m, n = d0.shape[1], d1.shape[1]
            for i in range(lg.conf.n_layers):
                d0, d1 = lg.transformers[i](d0, d1, e0, e1)
                if i == lg.conf.n_layers - 1:
                    continue  # no early stopping at the last layer
                if lg.conf.depth_confidence > 0:
                    t0, t1 = lg.token_confidence[i](d0, d1)
                    if lg.check_if_stop(t0, t1, i, m + n):
                        break
            scores, _ = lg.log_assignment[i](d0, d1)
            # Row / column maxima on the device; selection on the CPU.
            max0 = scores[0, :-1, :-1].max(1)
            max1 = scores[0, :-1, :-1].max(0)
            v0, i0 = max0.values.cpu().numpy(), max0.indices.cpu().numpy()
            i1 = max1.indices.cpu().numpy()
        return mutual_matches(v0, i0, i1, lg.conf.filter_threshold)


def mutual_matches(v0: np.ndarray, i0: np.ndarray, i1: np.ndarray, th: float) -> np.ndarray:
    """kornia's `filter_matches` for one pair: mutual best matches whose
    probability exp(log-assignment) exceeds `th`."""
    rows = np.arange(len(i0))
    keep = (i1[i0] == rows) & (np.exp(v0) > th)
    return np.stack([rows[keep], i0[keep]], axis=1).astype(np.uint32)
