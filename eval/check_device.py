"""Compare the PyTorch matchers on a GPU against the CPU on two images.

    python eval/check_device.py runs/kitti117/img/000000.png runs/kitti117/img/000001.png

Reports: exact SIFT matching (must be identical), the DISK score map and
keypoints, and LightGlue matches on identical inputs (small differences are
expected from floating-point arithmetic and float16 attention; large ones mean
a device problem), with the time per image pair of each LightGlue setting.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pycolmap

from av1sfm.devices import DEVICES, device_name, pick_device, synchronize
from av1sfm.learned import DiskLightGlue, Features, double_softmax_maxima
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
    sys.stdout.reconfigure(line_buffering=True)  # keep output if the device aborts
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
    if len(kc) == len(kd) and overlap(kc, kd) == 1.0:
        dc, dd = (f[k][0].descriptors.cpu() for k in ("cpu", "device"))
        print(
            f"DISK descriptors: max |cpu - device| = {(dc - dd).abs().max().item():.2e} (unit vectors)"
        )

    def on(dev_, feats: Features) -> Features:
        return Features(feats.keypoints.to(dev_), feats.descriptors.to(dev_), feats.size)

    # LightGlue on identical (CPU-extracted) features.
    fc0, fc1 = f["cpu"]
    isolate_modules(nets, fc0, fc1, dev)
    primitives(dev)
    lightglue_variants(nets, on(dev, fc0), on(dev, fc1), fc0, fc1)


def rel_diff(a, b) -> float:
    a, b = a.float().cpu(), b.float().cpu()
    return (a - b).abs().max().item() / max(a.abs().max().item(), 1e-12)


def isolate_modules(nets, fc0, fc1, dev) -> None:
    """Each module of the positional encoding and LightGlue layer 0, run on the
    device with exactly the inputs it received on the CPU (no error propagation)."""
    torch = nets["cpu"].torch
    cpu_lg, dev_lg = nets["cpu"].lightglue, nets["device"].lightglue
    for net in nets.values():
        configure(net, early_stop=False, prune=False, half=False, on_device=False)
    wanted = [
        (n, m)
        for n, m in cpu_lg.named_modules()
        if n == "posenc" or n.startswith(("posenc.", "transformers.0"))
    ]
    seen, rec = set(), []

    def hook(name):
        def f(_m, args, kwargs, out):
            if name not in seen:  # first call only (self-attention runs on both images)
                seen.add(name)
                rec.append((name, args, kwargs, out))

        return f

    hooks = [m.register_forward_hook(hook(n), with_kwargs=True) for n, m in wanted]
    nets["cpu"].match(fc0, fc1)
    for h in hooks:
        h.remove()

    def to_dev(x):
        if isinstance(x, torch.Tensor):
            return x.to(dev)
        if isinstance(x, (tuple, list)):
            return type(x)(to_dev(v) for v in x)
        return x

    def tensors(x):
        if isinstance(x, torch.Tensor):
            return [x]
        if isinstance(x, (tuple, list)):
            return [t for v in x for t in tensors(v)]
        return []

    dev_mods = dict(dev_lg.named_modules())
    print("LightGlue modules, same inputs on both devices (max rel. diff of the output):")
    for name, args, kwargs, out in rec:
        with torch.inference_mode():
            got = dev_mods[name](*to_dev(args), **{k: to_dev(v) for k, v in kwargs.items()})
        diffs = [rel_diff(a, b) for a, b in zip(tensors(out), tensors(got), strict=False)]
        kind = type(dev_mods[name]).__name__
        print(f"  {name or '(root)':40s} {kind:34s} {max(diffs, default=float('nan')):.2e}")


def primitives(dev) -> None:
    """Single operations used by LightGlue's layers, on random inputs."""
    import torch
    from kornia.feature.lightglue import apply_cached_rotary_emb, rotate_half

    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 4, 5000, 64, generator=g)
    freqs = torch.randn(2, 1, 1, 5000, 64, generator=g)
    w = torch.randn(256, 512, generator=g) / 16
    sim = torch.randn(1, 4, 5000, 5000, generator=g)
    ops = {
        "rotate_half (unflatten/unbind/stack)": lambda t: rotate_half(t[0]),
        "apply_cached_rotary_emb": lambda t: apply_cached_rotary_emb(t[1], t[0]),
        "x[..., ::2] (strided view)": lambda t: t[0][..., ::2].contiguous(),
        "transpose(1, 2).contiguous()": lambda t: t[0].transpose(1, 2).contiguous(),
        "softmax(-1)": lambda t: torch.softmax(t[0], -1),
        "einsum q k^T": lambda t: torch.einsum("...id,...jd->...ij", t[0], t[0]),
        "scaled_dot_product_attention": lambda t: torch.nn.functional.scaled_dot_product_attention(
            t[0], t[0], t[0]
        ),
        "scaled_dot_product_attention float16": lambda t: (
            torch.nn.functional.scaled_dot_product_attention(t[0].half(), t[0].half(), t[0].half())
        ),
        "layer_norm": lambda t: torch.nn.functional.layer_norm(t[0], (64,)),
        "gelu": lambda t: torch.nn.functional.gelu(t[0]),
        "matmul (5000x256 @ 256x512)": lambda t: t[0].reshape(-1, 256) @ t[2],
        "cos / sin": lambda t: torch.cos(t[0]) + torch.sin(t[0]),
        "einsum, transposed operand": lambda t: torch.einsum(
            "bhji, bhjd -> bhid", torch.softmax(t[3], -1).transpose(-2, -1), t[0]
        ),
        "transpose(-2, -1).contiguous() 5000²": lambda t: t[3].transpose(-2, -1).contiguous(),
        "matmul, sum over 5000 (batched 4-D)": lambda t: torch.matmul(
            torch.softmax(t[3], -1), t[0]
        ),
        "matmul, sum over 5000 (2-D)": lambda t: torch.softmax(t[3][0, 0], -1) @ t[0][0, 0],
        "log_softmax over 5000": lambda t: torch.log_softmax(t[3], -1),
        "logsumexp over 5000": lambda t: torch.logsumexp(t[3], -1),
        "double_softmax_maxima (values)": lambda t: torch.from_numpy(
            double_softmax_maxima(t[3][0, :1], t[3][0, 1, :, :1][None], t[3][0, 2, :, :1][None])[0]
        ),
        "max over 5000 (values)": lambda t: t[3].max(-1).values,
        "sum over 5000": lambda t: t[3].sum(-1),
        "add transposed (x + x^T)": lambda t: t[3] + t[3].transpose(-1, -2),
        "matmul, sum over 1024 (2-D)": lambda t: (
            torch.softmax(t[3][0, 0, :, :1024], -1) @ t[0][0, 0, :1024]
        ),
    }
    print("Primitive operations, random inputs (max rel. diff):")
    for name, op in ops.items():
        with torch.inference_mode():
            a = op((x, freqs, w, sim))
            try:
                b = op((x.to(dev), freqs.to(dev), w.to(dev), sim.to(dev)))
                print(f"  {name:38s} {rel_diff(a, b):.2e}")
            except RuntimeError as e:  # report and continue
                print(f"  {name:38s} FAILED: {type(e).__name__}: {e}")


def configure(net: DiskLightGlue, *, early_stop=True, prune=True, half=None, on_device=None):
    gpu = net.device.type != "cpu"
    net.lightglue.conf.depth_confidence = 0.95 if early_stop else -1
    net.prune = prune
    net.half_attention = gpu if half is None else half
    net.assignment_on_device = gpu if on_device is None else on_device
    net.prune_min_keypoints = 1536 if net.half_attention else 1024


def lightglue_variants(nets, fd0, fd1, fc0, fc1) -> None:
    """LightGlue on identical inputs: per-layer differences, then full matching."""
    # Layer by layer in float32, without early stopping or pruning (all layers run).
    for net in nets.values():
        configure(net, early_stop=False, prune=False, half=False, on_device=False)
    outs = {}
    for key, (f0, f1) in {"cpu": (fc0, fc1), "device": (fd0, fd1)}.items():
        lg, rec = nets[key].lightglue, []
        hooks = [
            m.register_forward_hook(lambda _m, _i, o, rec=rec: rec.append(o))
            for m in [lg.input_proj, *lg.transformers]
        ]
        nets[key].match(f0, f1)
        for h in hooks:
            h.remove()
        flat = []
        for o in rec:
            flat.extend(o if isinstance(o, tuple) else (o,))
        outs[key] = [t.float().cpu() for t in flat]
    n_layers = len(nets["cpu"].lightglue.transformers)
    names = ["input_proj 0", "input_proj 1"]
    names += [f"layer {i} desc{j}" for i in range(n_layers) for j in (0, 1)]
    for name, a, b in zip(names, outs["cpu"], outs["device"], strict=False):
        rel = (a - b).abs().max().item() / max(a.abs().max().item(), 1e-12)
        print(f"  {name:18s} max rel. diff {rel:.2e}")

    # Full matching. The CPU reference always runs in float32 with the CPU dual
    # softmax, pruning (if any) from the same number of keypoints as the device.
    variants = {
        "fp32, no pruning, CPU softmax": {"prune": False, "half": False, "on_device": False},
        "fp32, pruning, device softmax": {"half": False},
        "fp16 attention, no pruning": {"prune": False},
        "default (fp16, pruning)": {},
    }
    for name, cfg in variants.items():
        configure(nets["device"], **cfg)
        configure(nets["cpu"], prune=cfg.get("prune", True), half=False, on_device=False)
        nets["cpu"].prune_min_keypoints = nets["device"].prune_min_keypoints
        lc = nets["cpu"].match(fc0, fc1)
        ld = nets["device"].match(fd0, fd1)
        dev = nets["device"].device
        synchronize(dev)
        t = time.perf_counter()
        for _ in range(3):
            nets["device"].match(fd0, fd1)
        synchronize(dev)
        ms = (time.perf_counter() - t) / 3 * 1000
        print(
            f"LightGlue, {name:30s}: {len(lc):5d} cpu, {len(ld):5d} device, "
            f"same matches: {overlap(lc, ld):.3f}, device {ms:6.0f} ms/pair"
        )


if __name__ == "__main__":
    main()
