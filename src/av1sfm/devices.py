"""PyTorch device selection for the optional GPU matchers.

PyTorch is an optional dependency (see README, "GPU matchers"); this module
imports it lazily so the rest of the package works without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

DEVICES = ("auto", "cpu", "xpu", "cuda", "mps")


def import_torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "PyTorch is needed for this matcher. Intel GPU: uv pip install torch "
            "--index-url https://download.pytorch.org/whl/xpu (see README)"
        ) from e
    return torch


def pick_device(name: str = "auto") -> torch.device:
    """`name` as a torch.device; "auto" prefers an Intel GPU, then CUDA, then MPS, then CPU."""
    torch = import_torch()
    if name not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {name!r}")
    available = {
        "cpu": True,
        "xpu": hasattr(torch, "xpu") and torch.xpu.is_available(),
        "cuda": torch.cuda.is_available(),
        "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }
    if name == "auto":
        name = next(d for d in ("xpu", "cuda", "mps", "cpu") if available[d])
    elif not available[name]:
        raise RuntimeError(f"PyTorch device {name!r} is not available in this installation")
    return torch.device(name)


def device_name(device: torch.device) -> str:
    """Human-readable device description for run statistics."""
    torch = import_torch()
    if device.type == "xpu":
        return f"xpu: {torch.xpu.get_device_name(device)}"
    if device.type == "cuda":
        return f"cuda: {torch.cuda.get_device_name(device)}"
    return device.type


def synchronize(device: torch.device) -> None:
    """Wait for queued GPU work, so that stage timings include it."""
    torch = import_torch()
    if device.type == "xpu":
        torch.xpu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
