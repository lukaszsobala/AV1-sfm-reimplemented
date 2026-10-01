"""Minimal reproductions of the PyTorch XPU problems worked around in av1sfm.

Self-contained (PyTorch only), for bug reports; docs/DEPENDENCY_ISSUES.md
describes each problem. Every line compares the device with the CPU on the
same random inputs:

    python eval/xpu_repro.py            # Intel GPU (xpu)
    python eval/xpu_repro.py cuda       # any other PyTorch device

On a correct device, every relative difference is below about 1e-5 (1e-3 for
float16) and every count equals the CPU's.
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F


def rel(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref, got = ref.float(), got.float().cpu()
    return ((ref - got).abs().max() / ref.abs().max().clamp_min(1e-12)).item()


def environment(dev: str) -> None:
    print(f"torch {torch.__version__}, device {dev}")
    if dev == "xpu":
        props = torch.xpu.get_device_properties()
        for key in ("name", "platform_name", "driver_version", "version", "gpu_eu_count"):
            if hasattr(props, key):
                print(f"  {key}: {getattr(props, key)}")


def line(name: str, value: str) -> None:
    print(f"  {name:44s} {value}")


def main(dev: str) -> None:
    environment(dev)
    g = torch.Generator().manual_seed(0)
    s = torch.randn(5000, 5000, generator=g)  # e.g. attention logits, 5000 keypoints
    v = torch.randn(5000, 64, generator=g)  # e.g. attention values
    sd, vd = s.to(dev), v.to(dev)

    print("Matrix products (5000 x n) @ (n x 64), softmax on the CPU (max relative difference):")
    for n in (1024, 4096, 4097, 4352, 4608, 5000):
        p = torch.softmax(s[:, :n], -1)
        line(f"softmax(S[:, :{n}]) @ V[:{n}]", f"{rel(p @ v[:n], p.to(dev) @ vd[:n]):.2e}")
    p, pd = torch.softmax(s, -1), torch.softmax(sd, -1)
    b, bd = p.reshape(1, 1, 5000, 5000), pd.reshape(1, 1, 5000, 5000)
    w, wd = v.reshape(1, 1, 5000, 64), vd.reshape(1, 1, 5000, 64)
    print("The same with the softmax on the device:")
    line("softmax(S) @ V, 2-D, n = 5000", f"{rel(p @ v, pd @ vd):.2e}")
    line("batched 4-D matmul, n = 5000", f"{rel(b @ w, bd @ wd):.2e}")
    ein = "bhji,bhjd->bhid"
    ref = torch.einsum(ein, b.transpose(-2, -1), w)
    line(
        f"einsum {ein}, P^T operand", f"{rel(ref, torch.einsum(ein, bd.transpose(-2, -1), wd)):.2e}"
    )

    print("Reductions over the last dimension, 5000 rows of n:")
    for n in (4096, 4097, 5000):
        x, xd = s[:, :n], sd[:, :n]
        line(f"softmax, n = {n}", f"{rel(torch.softmax(x, -1), torch.softmax(xd, -1)):.2e}")
        line(
            f"log_softmax, n = {n}",
            f"{rel(torch.log_softmax(x, -1), torch.log_softmax(xd, -1)):.2e}",
        )
    line(
        "logsumexp, n = 5000 (workaround)",
        f"{rel(torch.logsumexp(s, -1), torch.logsumexp(sd, -1)):.2e}",
    )
    q = torch.randn(1, 4, 5000, 64, generator=g)
    qd = q.to(dev)
    sdpa = F.scaled_dot_product_attention
    line("scaled_dot_product_attention (workaround)", f"{rel(sdpa(q, q, q), sdpa(qd, qd, qd)):.2e}")
    line(
        "scaled_dot_product_attention float16",
        f"{rel(sdpa(q, q, q), sdpa(qd.half(), qd.half(), qd.half())):.2e}",
    )

    # kornia's DISK keypoint selection: a non-maximum-suppression mask, then
    # mask.nonzero() and x[mask]. Both must return mask.sum() elements.
    print("Boolean masks: mask.sum() / nonzero / x[mask] (all three must equal the CPU count):")
    for seed in range(3):
        h = torch.randn(1, 1, 384, 1248, generator=torch.Generator().manual_seed(seed))
        nms = (h == F.max_pool2d(h, 5, stride=1, padding=2)) & (h > 0)  # on the CPU
        hd = h.to(dev)
        md = ((hd == F.max_pool2d(hd, 5, stride=1, padding=2)) & (hd > 0))[0, 0]  # on the device
        got = f"{int(md.sum())} / {md.nonzero().shape[0]} / {hd[0, 0][md].numel()}"
        line(f"384x1248 NMS mask, seed {seed}, CPU {int(nms.sum())}", got)
    for n in (4096, 65536, 479232):
        m = torch.rand(n, generator=torch.Generator().manual_seed(n)) < 0.04
        md = m.to(dev)
        line(
            f"1-D random mask, {n} elements, CPU {int(m.sum())}",
            f"{int(md.sum())} / {md.nonzero().shape[0]}",
        )
    print("done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "xpu")
