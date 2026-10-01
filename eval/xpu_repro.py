"""Minimal reproductions of the PyTorch XPU problems worked around in av1sfm.

Self-contained (PyTorch only), for bug reports; docs/DEPENDENCY_ISSUES.md
describes each problem. Every line compares the device with the CPU on the
same random inputs:

    python eval/xpu_repro.py            # Intel GPU (xpu)
    python eval/xpu_repro.py cuda       # any other PyTorch device

On a correct device, every relative difference is below about 1e-5 (1e-3 for
float16) and every count matches.
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F


def rel(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref, got = ref.float(), got.float().cpu()
    return ((ref - got).abs().max() / ref.abs().max().clamp_min(1e-12)).item()


def main(dev: str) -> None:
    print(f"torch {torch.__version__}, device {dev}", end="")
    if dev == "xpu":
        print(f": {torch.xpu.get_device_name()}", end="")
    print()
    g = torch.Generator().manual_seed(0)
    s = torch.randn(5000, 5000, generator=g)  # e.g. attention logits, 5000 keypoints
    v = torch.randn(5000, 64, generator=g)  # e.g. attention values
    sd, vd = s.to(dev), v.to(dev)

    print("Reductions over a long dimension (max relative difference to the CPU):")
    for n in (1024, 2048, 4096, 5000):
        p = torch.softmax(s[:, :n], -1)
        got = torch.softmax(sd[:, :n], -1) @ vd[:n]
        print(f"  softmax(S[:, :{n}]) @ V[:{n}]        {rel(p @ v[:n], got):.2e}")
    p, pd = torch.softmax(s, -1), torch.softmax(sd, -1)
    b, bd = p.reshape(1, 1, 5000, 5000), pd.reshape(1, 1, 5000, 5000)
    w, wd = v.reshape(1, 1, 5000, 64), vd.reshape(1, 1, 5000, 64)
    print(f"  batched 4-D matmul, sum over 5000      {rel(b @ w, bd @ wd):.2e}")
    ein = "bhji,bhjd->bhid"
    print(
        f"  einsum {ein}, P^T operand  "
        f"{rel(torch.einsum(ein, b.transpose(-2, -1), w), torch.einsum(ein, bd.transpose(-2, -1), wd)):.2e}"
    )
    print(
        f"  log_softmax over 5000                  {rel(torch.log_softmax(s, -1), torch.log_softmax(sd, -1)):.2e}"
    )
    print(
        f"  logsumexp over 5000 (workaround)       {rel(torch.logsumexp(s, -1), torch.logsumexp(sd, -1)):.2e}"
    )
    q = torch.randn(1, 4, 5000, 64, generator=g)
    qd = q.to(dev)
    sdpa = F.scaled_dot_product_attention
    print(f"  scaled_dot_product_attention (workaround) {rel(sdpa(q, q, q), sdpa(qd, qd, qd)):.2e}")
    print(
        "  scaled_dot_product_attention float16   "
        f"{rel(sdpa(q, q, q), sdpa(qd.half(), qd.half(), qd.half())):.2e}"
    )

    # kornia's DISK keypoint selection: non-maximum suppression mask, then
    # mask.nonzero() and x[mask] with the same mask must agree in length.
    print("Boolean masks (counts must agree; a mismatch or a device assert is the bug):")
    for seed in range(3):
        h = torch.randn(1, 1, 384, 1248, generator=torch.Generator().manual_seed(seed)).to(dev)
        nms = (h == F.max_pool2d(h, 5, stride=1, padding=2)) & (h > 0)
        m = nms[0, 0]
        nonzero = m.nonzero().shape[0]
        indexed = h[0, 0][m].numel()
        print(f"  seed {seed}: mask.sum() {int(m.sum())}, nonzero {nonzero}, x[mask] {indexed}")
    print("done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "xpu")
