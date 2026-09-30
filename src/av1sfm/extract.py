"""Per-frame AV1 block metadata, in display order with absolute frame indices.

Thin layer over the vendored `dav1d_inspect.iter_frames` (sigmedia/AV1-Optical-Flow).
It resolves AV1's cyclic order hints and per-slot reference order hints into
absolute frame indices, so downstream code can say "this block's MV points into
frame m" without knowing about reference slots.

Conventions (verified against a synthetic pan, see tests/test_extract_integration.py):
  * `mv[..., 0:2]` is list 0 (x, y) and `mv[..., 2:4]` is list 1, in 1/8 pel.
  * A block at pixel position p in frame n with MV v and reference slot r is
    predicted from position p + v/8 in frame `ref_frame_index[r - 1]`.
  * `ref[..., k]`: 0 = intra, 1..7 = LAST..ALTREF slot, -1 (or <= 0) = none.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._vendor.dav1d_inspect import iter_frames

# AV1 streams produced by libaom use 7 order-hint bits by default; the upstream
# AV1-Optical-Flow pipeline makes the same assumption (see ASSUMPTIONS.md).
ORDER_HINT_MODULO = 128

KEY_FRAME = 0
INTRA_ONLY_FRAME = 2


@dataclass
class FrameMotion:
    """Block motion of one decoded frame on the 4x4 grid."""

    index: int  # absolute display-order frame index
    width: int
    height: int
    frame_type: int
    mv: np.ndarray  # (H/4, W/4, 4) int16, 1/8 pel: [mv0_x, mv0_y, mv1_x, mv1_y]
    ref: np.ndarray  # (H/4, W/4, 2) int16 reference slots
    block_map: np.ndarray  # (H/4, W/4) uint8 AOM BLOCK_* enum
    ref_frame_index: tuple[int, ...]  # absolute frame index per slot 1..7

    @property
    def is_intra(self) -> bool:
        return self.frame_type in (KEY_FRAME, INTRA_ONLY_FRAME)


def _unwrap(order_hint: int, anchor: int) -> int:
    """Absolute index congruent to `order_hint` (mod 128) closest to `anchor`."""
    base = anchor - ((anchor - order_hint) % ORDER_HINT_MODULO)
    return base + ORDER_HINT_MODULO if anchor - base > ORDER_HINT_MODULO // 2 else base


def iter_frame_motion(ivf_path: str | Path, n_threads: int = 0) -> Iterator[FrameMotion]:
    """Decode `ivf_path` and yield `FrameMotion` for every frame in decode order.

    For the streaming (low-delay, no hidden frames) configuration used by this
    project decode order equals display order; a `ValueError` is raised if a
    frame's order hint goes backwards, which would indicate a stream with
    future references or hidden frames.
    """
    prev_abs: int | None = None
    for fr in iter_frames(ivf_path, n_threads=n_threads):
        oh = int(fr["frame_offset"])
        cur = oh if prev_abs is None else _unwrap(oh, prev_abs + 1)
        if prev_abs is not None and cur <= prev_abs:
            raise ValueError(
                f"{ivf_path}: order hint went backwards (frame {prev_abs} -> {cur}); "
                "only low-delay streams with past references are supported"
            )
        refs = tuple(cur - ((oh - int(r)) % ORDER_HINT_MODULO) for r in fr["refpoc"])
        prev_abs = cur
        yield FrameMotion(
            index=cur,
            width=int(fr["width"]),
            height=int(fr["height"]),
            frame_type=int(fr["frame_type"]),
            mv=fr["motion_vectors"],
            ref=fr["reference_map"],
            block_map=fr["block_map"],
            ref_frame_index=refs,
        )


def load_frame_motion(ivf_path: str | Path, n_threads: int = 0) -> list[FrameMotion]:
    return list(iter_frame_motion(ivf_path, n_threads=n_threads))
