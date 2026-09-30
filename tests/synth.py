"""Helpers to build synthetic `FrameMotion` objects without an encoder."""

from __future__ import annotations

import numpy as np

from av1sfm.blocks import AOM_BLOCK_SIZES
from av1sfm.extract import KEY_FRAME, FrameMotion

INTER_FRAME = 1


def bsize(w: int, h: int) -> int:
    return AOM_BLOCK_SIZES.index((w, h))


def make_frame(
    index: int,
    width: int = 64,
    height: int = 64,
    *,
    block: tuple[int, int] = (16, 16),
    mv_px: tuple[float, float] = (0.0, 0.0),
    ref_slot: int = 1,
    ref_frames: tuple[int, ...] | None = None,
    intra: bool = False,
) -> FrameMotion:
    """Uniform block partition and uniform motion; slot 1 = previous frame by default."""
    gh, gw = -(-height // 8) * 2, -(-width // 8) * 2  # grid padded to 8 px like dav1d
    mv = np.zeros((gh, gw, 4), np.int16)
    ref = np.zeros((gh, gw, 2), np.int16)
    ref[..., 1] = -1
    bm = np.zeros((gh, gw), np.uint8)
    if not intra:
        mv[..., 0] = round(mv_px[0] * 8)
        mv[..., 1] = round(mv_px[1] * 8)
        ref[..., 0] = ref_slot
        bm[:] = bsize(*block)
    if ref_frames is None:
        ref_frames = tuple(max(index - 1 - k, 0) for k in range(7))
    return FrameMotion(
        index=index, width=width, height=height,
        frame_type=KEY_FRAME if intra else INTER_FRAME,
        mv=mv, ref=ref, block_map=bm, ref_frame_index=ref_frames,
    )  # fmt: skip


def set_block(fm: FrameMotion, x0: int, y0: int, w: int, h: int, **fields) -> None:
    """Overwrite one block's 4x4 cells (pixel coordinates)."""
    sl = np.s_[y0 // 4 : (y0 + h) // 4, x0 // 4 : (x0 + w) // 4]
    fm.block_map[sl] = bsize(w, h)
    if "mv_px" in fields:
        fm.mv[sl + (0,)] = round(fields["mv_px"][0] * 8)
        fm.mv[sl + (1,)] = round(fields["mv_px"][1] * 8)
    if "mv1_px" in fields:
        fm.mv[sl + (2,)] = round(fields["mv1_px"][0] * 8)
        fm.mv[sl + (3,)] = round(fields["mv1_px"][1] * 8)
    if "ref0" in fields:
        fm.ref[sl + (0,)] = fields["ref0"]
    if "ref1" in fields:
        fm.ref[sl + (1,)] = fields["ref1"]
