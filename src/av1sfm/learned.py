"""DISK + LightGlue baseline on a PyTorch device (Intel GPU, CUDA or CPU).

The paper's Table I compares against DISK [Tyszkiewicz et al. 2020] features
matched with LightGlue [Lindenberger et al. 2023] on a GPU, without giving
settings. We use kornia's ports of both (Apache-2.0): DISK with the `depth`
weights, non-maximum suppression window 5, no score threshold, at most
`max_keypoints` per image (hloc's DISK default, 5000); LightGlue with the
`disk_lightglue` weights and its default adaptive depth / width and match
threshold (0.1). Weights are downloaded on first use into PyTorch's hub cache.

LightGlue runs through `DiskLightGlue.match`, which calls kornia's layers in
the order of kornia's forward pass, with its early stopping and point pruning
(kornia's defaults: depth confidence 0.95, width confidence 0.99; on a GPU,
pruning starts above 1536 keypoints, on the CPU always, as in kornia). On a
GPU, attention runs in float16 as kornia (and the original LightGlue) does on
CUDA with its default `flash=True`; on the CPU in float32. On the CPU, `match`
gives the same matches as kornia's own forward pass.

Everything runs on the device: DISK's keypoint selection, LightGlue's layers,
pruning, dual softmax and match selection (kornia's functions); only the
final matches are copied to the CPU. On Intel GPUs this needs GPU driver
(compute runtime) 26.31.39395 or newer (`devices.check_xpu_driver`): older
drivers return wrong results from `nonzero`, boolean-mask indexing and
softmax over more than 4096 elements (docs/DEPENDENCY_ISSUES.md,
eval/xpu_repro.py). Self- and cross-attention go through PyTorch's fused
attention kernel on every device (`cross_block_forward`); kornia uses it on
CUDA only.

Keypoints are written to the COLMAP database with +0.5 px (DISK returns pixel
indices; COLMAP puts the centre of the top-left pixel at (0.5, 0.5)). Raw
matches then go through COLMAP's geometric verification like every other method.

Licence of adapted code: `DiskLightGlue.match`, `DiskLightGlue.attention`,
`DiskLightGlue._prune` and `cross_block_forward` are adapted from kornia's
kornia/feature/lightglue.py (kornia 0.8.3, Copyright 2018 Kornia Team),
itself a port of LightGlue (github.com/cvg/LightGlue, Copyright 2023 ETH
Zurich), both under the Apache License 2.0 (LICENSES/Apache-2.0.txt).
Modified in 2026 for AV1-sfm-reimplemented: attention through
scaled_dot_product_attention on every device, and the forward pass split into
these functions. The rest of this file is original and, like the
modifications, licensed under the AGPL (see LICENSE).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

from .devices import check_xpu_driver, import_torch

if TYPE_CHECKING:  # PyTorch and kornia are optional
    import torch
    from kornia.feature.lightglue import MatchAssignment, TransformerLayer

MAX_KEYPOINTS = 5000


@dataclass
class Features:
    keypoints: torch.Tensor  # (n, 2) float on the device, DISK convention (pixel indices)
    descriptors: torch.Tensor  # (n, 128) float on the device
    size: tuple[int, int]  # (width, height)

    def colmap_keypoints(self) -> np.ndarray:
        return np.ascontiguousarray(self.keypoints.cpu().numpy() + 0.5, np.float32)


class DiskLightGlue:
    """DISK features and LightGlue matching on `device`.

    `half_attention` (default: on a GPU) computes attention in float16;
    `prune` enables LightGlue's point pruning.
    """

    def __init__(
        self,
        device,
        max_keypoints: int = MAX_KEYPOINTS,
        *,
        half_attention: bool | None = None,
        prune: bool = True,
    ):
        torch = import_torch()
        check_xpu_driver(device)
        try:
            from kornia.feature import DISK, LightGlue
            from kornia.feature.disk.detector import heatmap_to_keypoints
            from kornia.feature.lightglue import filter_matches, normalize_keypoints
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError("DISK + LightGlue needs kornia: uv pip install kornia") from e
        self.torch, self.device, self.max_keypoints = torch, device, max_keypoints
        self.heatmap_to_keypoints = heatmap_to_keypoints
        self.normalize_keypoints = normalize_keypoints
        self.filter_matches = filter_matches
        gpu = device.type != "cpu"
        self.half_attention = gpu if half_attention is None else half_attention
        self.prune = prune
        # kornia's pruning_min_kpts: -1 (always) on the CPU, 1536 with flash
        # (float16) attention on CUDA, else 1024.
        self.prune_min_keypoints = -1 if not gpu else 1536 if self.half_attention else 1024
        self.disk = DISK.from_pretrained("depth", device=device).eval()
        self.lightglue = LightGlue("disk").to(device).eval()
        for layer in cast("list[TransformerLayer]", self.lightglue.transformers):
            layer.self_attn.inner_attn.forward = self.attention
            layer.cross_attn.forward = functools.partial(
                cross_block_forward, layer.cross_attn, attention=self.attention
            )

    def attention(self, q, k, v, mask=None):
        """kornia's Attention.forward (scaled_dot_product_attention), in
        float16 if `half_attention` as kornia does on CUDA."""
        if mask is not None:
            raise NotImplementedError("masks are only used for padded (compiled) inputs")
        sdpa = self.torch.nn.functional.scaled_dot_product_attention
        if self.half_attention:
            return sdpa(*(x.half().contiguous() for x in (q, k, v))).to(q.dtype)
        return sdpa(*(x.contiguous() for x in (q, k, v)))

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
            kp = self.heatmap_to_keypoints(
                heatmap[:, :, :h, :w], n=self.max_keypoints, window_size=5
            )[0]
            xy = kp.xys
            flat = xy[:, 1] * dense.shape[-1] + xy[:, 0]
            desc = dense[0].flatten(1).index_select(1, flat).T
            desc = torch.nn.functional.normalize(desc, dim=-1)
        return Features(xy.float(), desc.float(), (w, h))

    def match(self, f0: Features, f1: Features) -> np.ndarray:
        """Raw matches (k, 2) uint32, indices into f0 / f1."""
        none = np.zeros((0, 2), np.uint32)
        if len(f0.keypoints) == 0 or len(f1.keypoints) == 0:
            return none
        torch, lg = self.torch, self.lightglue

        def prepare(f: Features):
            size = torch.tensor([f.size], device=self.device, dtype=torch.float32)
            kpts = self.normalize_keypoints(f.keypoints[None], size)
            return lg.input_proj(f.descriptors[None].contiguous()), lg.posenc(kpts)

        with torch.inference_mode():
            (d0, e0), (d1, e1) = prepare(f0), prepare(f1)
            m, n = d0.shape[1], d1.shape[1]
            ind0 = torch.arange(m, device=self.device)  # kept points (pruning)
            ind1 = torch.arange(n, device=self.device)
            i = 0  # the last layer run (LightGlue has at least one)
            for i in range(lg.conf.n_layers):
                d0, d1 = lg.transformers[i](d0, d1, e0, e1)
                if i == lg.conf.n_layers - 1:
                    continue  # no early stopping or pruning at the last layer
                t0 = t1 = None
                if lg.conf.depth_confidence > 0:
                    t0, t1 = lg.token_confidence[i](d0, d1)
                    if lg.check_if_stop(t0, t1, i, m + n):
                        break
                if self.prune and lg.conf.width_confidence > 0:
                    if d0.shape[-2] > self.prune_min_keypoints:
                        ind0, d0, e0 = self._prune(i, t0, ind0, d0, e0)
                    if d1.shape[-2] > self.prune_min_keypoints:
                        ind1, d1, e1 = self._prune(i, t1, ind1, d1, e1)
                    if len(ind0) == 0 or len(ind1) == 0:
                        return none
            # kornia's match assignment (dual softmax) and filter_matches.
            scores, _ = lg.log_assignment[i](d0, d1)
            m0 = self.filter_matches(scores, lg.conf.filter_threshold)[0][0]
            k0 = torch.where(m0 > -1)[0]
            matches = torch.stack([ind0[k0], ind1[m0[k0]]], -1)
        return matches.cpu().numpy().astype(np.uint32)

    def _prune(self, i: int, token, ind, desc, encoding):
        """kornia's point pruning (get_pruning_mask) for one image's points."""
        lg = self.lightglue
        head = cast("MatchAssignment", lg.log_assignment[i])
        mask = lg.get_pruning_mask(token, head.get_matchability(desc), i)
        keep = self.torch.where(mask[0])[0]
        return (
            ind.index_select(0, keep),
            desc.index_select(1, keep),
            encoding.index_select(-2, keep),
        )


def cross_block_forward(self, x0, x1, mask=None, *, attention):
    """kornia's CrossBlock.forward through `attention` (DiskLightGlue.attention).

    m0 = softmax(q0 q1^T / sqrt(d)) v1 and m1 = softmax(q1 q0^T / sqrt(d)) v0,
    as in kornia (which scales q0 and q1 by d^-1/4 each and, on CUDA, uses its
    attention module like this). On other devices kornia computes the attention
    matrix and its softmax explicitly instead of using the fused kernel.
    """
    import torch

    qk0, qk1, v0, v1 = (
        t.unflatten(-1, (self.heads, -1)).transpose(1, 2)
        for t in (self.to_qk(x0), self.to_qk(x1), self.to_v(x0), self.to_v(x1))
    )
    m0, m1 = attention(qk0, qk1, v1, mask), attention(qk1, qk0, v0, mask)  # scale d^-1/2
    m0, m1 = (t.transpose(1, 2).contiguous().flatten(start_dim=-2) for t in (m0, m1))
    m0, m1 = self.to_out(m0), self.to_out(m1)
    x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
    x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
    return x0, x1
