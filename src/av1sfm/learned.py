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

Changes for PyTorch XPU 2.14 (Intel GPUs), where `nonzero` and boolean-mask
indexing on large tensors return too few elements, and softmax and
log_softmax over more than 4096 elements are wrong (docs/DEPENDENCY_ISSUES.md,
eval/xpu_repro.py): point selection (DISK's keypoints, LightGlue's pruning and
final matches) uses index lists computed on the CPU; cross-attention goes
through PyTorch's fused attention kernel (`cross_block_forward`), which is
correct; and on a GPU, the dual softmax is written with logsumexp
(`double_softmax_maxima`).

Keypoints are written to the COLMAP database with +0.5 px (DISK returns pixel
indices; COLMAP puts the centre of the top-left pixel at (0.5, 0.5)). Raw
matches then go through COLMAP's geometric verification like every other method.

Licence of adapted code: `DiskLightGlue.match`, `DiskLightGlue.attention`,
`DiskLightGlue._keep`, `double_softmax_maxima`, `mutual_matches` and
`cross_block_forward` are adapted from kornia's kornia/feature/lightglue.py
(kornia 0.8.3, Copyright 2018 Kornia Team), itself a port of LightGlue
(github.com/cvg/LightGlue, Copyright 2023 ETH Zurich), both under the Apache
License 2.0 (LICENSES/Apache-2.0.txt). Modified in 2026 for AV1-sfm-reimplemented:
point selection on the CPU, attention through scaled_dot_product_attention,
the dual softmax written with logsumexp, and the forward pass split into
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

from .devices import import_torch

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
    `prune` enables LightGlue's point pruning; `assignment_on_device` (default:
    on a GPU) computes the dual softmax on the device instead of the CPU.
    """

    def __init__(
        self,
        device,
        max_keypoints: int = MAX_KEYPOINTS,
        *,
        half_attention: bool | None = None,
        prune: bool = True,
        assignment_on_device: bool | None = None,
    ):
        torch = import_torch()
        try:
            from kornia.feature import DISK, LightGlue
            from kornia.feature.disk.detector import heatmap_to_keypoints
            from kornia.feature.lightglue import normalize_keypoints, sigmoid_log_double_softmax
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError("DISK + LightGlue needs kornia: uv pip install kornia") from e
        self.torch, self.device, self.max_keypoints = torch, device, max_keypoints
        self.heatmap_to_keypoints = heatmap_to_keypoints
        self.normalize_keypoints = normalize_keypoints
        self.sigmoid_log_double_softmax = sigmoid_log_double_softmax
        gpu = device.type != "cpu"
        self.half_attention = gpu if half_attention is None else half_attention
        self.assignment_on_device = gpu if assignment_on_device is None else assignment_on_device
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
            ind0, ind1 = np.arange(m), np.arange(n)  # kept points (pruning)
            thresholds = lg.confidence_thresholds.cpu()
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
                        keep = self._keep(i, d0, t0, thresholds)
                        ind0, d0, e0 = ind0[keep], *self._select(keep, d0, e0)
                    if d1.shape[-2] > self.prune_min_keypoints:
                        keep = self._keep(i, d1, t1, thresholds)
                        ind1, d1, e1 = ind1[keep], *self._select(keep, d1, e1)
                    if len(ind0) == 0 or len(ind1) == 0:
                        return none
            # kornia's MatchAssignment.forward, split: the similarity matrix on
            # the device, the dual softmax on the device or the CPU.
            head = cast("MatchAssignment", lg.log_assignment[i])
            md0, md1 = head.final_proj(d0), head.final_proj(d1)
            scale = md0.shape[-1] ** 0.25
            sim = torch.einsum("bmd,bnd->bmn", md0 / scale, md1 / scale)
            z0, z1 = head.matchability(d0), head.matchability(d1)
            if self.assignment_on_device:
                v0, i0, i1 = double_softmax_maxima(sim, z0, z1)
            else:
                scores = self.sigmoid_log_double_softmax(sim.cpu(), z0.cpu(), z1.cpu())
                max0 = scores[0, :-1, :-1].max(1)
                max1 = scores[0, :-1, :-1].max(0)
                v0, i0 = max0.values.numpy(), max0.indices.numpy()
                i1 = max1.indices.numpy()
        mm = mutual_matches(v0, i0, i1, lg.conf.filter_threshold)
        return np.stack([ind0[mm[:, 0]], ind1[mm[:, 1]]], axis=1).astype(np.uint32)

    def _keep(self, i: int, desc, token, thresholds) -> np.ndarray:
        """kornia's get_pruning_mask, evaluated on the CPU: indices of the kept points."""
        lg = self.lightglue
        head = cast("MatchAssignment", lg.log_assignment[i])
        keep = head.get_matchability(desc).cpu() > (1 - lg.conf.width_confidence)
        if token is not None:  # low-confidence points are never pruned
            keep |= token.cpu() <= thresholds[i]
        return np.flatnonzero(keep[0].numpy())

    def _select(self, keep: np.ndarray, desc, encoding):
        k = self.torch.from_numpy(keep).to(self.device)
        return desc.index_select(1, k), encoding.index_select(-2, k)


def double_softmax_maxima(sim, z0, z1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row maxima and argmaxima, and column argmaxima, of kornia's
    `sigmoid_log_double_softmax(sim, z0, z1)[0, :-1, :-1]`, computed on sim's device.

    log_softmax(x) is written as x - logsumexp(x), and reductions run over the
    last dimension of contiguous tensors: on PyTorch XPU 2.14, log_softmax over
    more than 4096 entries is wrong, while logsumexp, max and transposed copies
    are correct (docs/DEPENDENCY_ISSUES.md).
    """
    import torch

    logsigmoid = torch.nn.functional.logsigmoid
    sim_t = sim.transpose(-1, -2).contiguous()
    scores0 = sim - torch.logsumexp(sim, -1, keepdim=True)
    scores1 = (sim_t - torch.logsumexp(sim_t, -1, keepdim=True)).transpose(-1, -2)
    scores = scores0 + scores1 + (logsigmoid(z0) + logsigmoid(z1).transpose(1, 2))
    max0 = scores[0].max(-1)
    i1 = scores[0].transpose(-1, -2).contiguous().max(-1).indices
    return max0.values.cpu().numpy(), max0.indices.cpu().numpy(), i1.cpu().numpy()


def mutual_matches(v0: np.ndarray, i0: np.ndarray, i1: np.ndarray, th: float) -> np.ndarray:
    """kornia's `filter_matches` for one pair: mutual best matches whose
    probability exp(log-assignment) exceeds `th`."""
    rows = np.arange(len(i0))
    keep = (i1[i0] == rows) & (np.exp(v0) > th)
    return np.stack([rows[keep], i0[keep]], axis=1).astype(np.uint32)


def cross_block_forward(self, x0, x1, mask=None, *, attention):
    """kornia's CrossBlock.forward through `attention` (DiskLightGlue.attention).

    m0 = softmax(q0 q1^T / sqrt(d)) v1 and m1 = softmax(q1 q0^T / sqrt(d)) v0,
    as in kornia (which scales q0 and q1 by d^-1/4 each and, on CUDA, uses its
    attention module like this). kornia's own version on other devices takes a
    softmax over all keypoints of the other image, which is wrong on PyTorch
    XPU 2.14 above 4096 keypoints; scaled_dot_product_attention is correct.
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
