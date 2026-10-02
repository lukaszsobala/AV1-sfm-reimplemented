"""Block geometry: collapse the 4x4 metadata grid to coded blocks, and turn
block motion vectors into point correspondences.

The inspection grid stores one record per 4x4 luma unit, so a 32x32 block
appears 64 times. AV1 partitions are aligned to their own size (a WxH block
starts at a multiple of W horizontally and of H vertically), so the grid cell
at the block's top-left corner is the one whose 4x4 coordinates are multiples of
(W/4, H/4). We keep exactly those cells.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .extract import FrameMotion

# AOM `BLOCK_*` enum value -> (width, height) in pixels. The vendored extractor
# remaps dav1d's BlockSize to this ordering (AOM av1/common/enums.h).
AOM_BLOCK_SIZES: tuple[tuple[int, int], ...] = (
    (4, 4),
    (4, 8),
    (8, 4),
    (8, 8),
    (8, 16),
    (16, 8),
    (16, 16),
    (16, 32),
    (32, 16),
    (32, 32),
    (32, 64),
    (64, 32),
    (64, 64),
    (64, 128),
    (128, 64),
    (128, 128),
    (4, 16),
    (16, 4),
    (8, 32),
    (32, 8),
    (16, 64),
    (64, 16),
)
_BLOCK_W4 = np.array([w // 4 for w, _ in AOM_BLOCK_SIZES], dtype=np.int32)
_BLOCK_H4 = np.array([h // 4 for _, h in AOM_BLOCK_SIZES], dtype=np.int32)


@dataclass
class BlockMotion:
    """One row per (coded block, reference list) with a usable motion vector."""

    frame: int  # absolute index of the frame the blocks belong to
    x0: np.ndarray  # (N,) block top-left, pixels
    y0: np.ndarray
    w: np.ndarray  # (N,) block size, pixels
    h: np.ndarray
    center: np.ndarray  # (N, 2) float64 source keypoint (COLMAP pixel convention)
    mv: np.ndarray  # (N, 2) float64 displacement in pixels (x, y)
    ref_frame: np.ndarray  # (N,) int64 absolute index of the reference frame
    ref_list: np.ndarray  # (N,) int8, 0 or 1

    @property
    def target(self) -> np.ndarray:
        """Matching point in `ref_frame`: centre displaced by the MV."""
        return self.center + self.mv

    def __len__(self) -> int:
        return len(self.x0)


def block_origins(block_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(row, col) 4x4-grid indices of the top-left cell of every coded block."""
    gy, gx = np.indices(block_map.shape)
    w4 = _BLOCK_W4[block_map]
    h4 = _BLOCK_H4[block_map]
    mask = (gx % w4 == 0) & (gy % h4 == 0)
    return gy[mask], gx[mask]


def block_centers(
    x0: np.ndarray, y0: np.ndarray, w: np.ndarray, h: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Centre of the visible part of each block.

    Keypoints use COLMAP's convention: pixel (i, j) covers [i, i+1) x [j, j+1),
    so the image spans [0, width) x [0, height) and a block covering pixel
    columns x0 .. x0+w-1 has centre x0 + w/2. Blocks that overhang the frame
    edge (the coded frame is padded to a multiple of 8) are clipped first.
    """
    x1 = np.minimum(x0 + w, width)
    y1 = np.minimum(y0 + h, height)
    return np.stack([(x0 + x1) / 2.0, (y0 + y1) / 2.0], axis=-1).astype(np.float64)


def frame_block_motion(
    fm: FrameMotion,
    *,
    skip_zero: bool = True,
    use_compound: bool = True,
    prev_only: bool = False,
    keep_out_of_frame: bool = False,
) -> BlockMotion:
    """List the coded blocks of `fm` with their MVs as correspondences.

    Args:
        skip_zero: drop (0, 0) MVs; the paper treats them as ambiguous
            (static, skipped or intra).
        use_compound: also emit the list-1 MV of compound (two-reference) blocks.
        prev_only: keep only MVs whose reference is the immediately previous
            frame (the paper's "every MV points to the previous frame").
            Otherwise any earlier or later frame is accepted, but not another
            decoded frame shown at the same time (an overlay's hidden frame).
        keep_out_of_frame: keep correspondences whose target lies outside the
            reference image.
    """
    empty = BlockMotion(
        fm.index,
        np.zeros(0, np.int64),
        np.zeros(0, np.int64),
        np.zeros(0, np.int64),
        np.zeros(0, np.int64),
        np.zeros((0, 2)),
        np.zeros((0, 2)),
        np.zeros(0, np.int64),
        np.zeros(0, np.int8),
    )
    if fm.is_intra or fm.mv.size == 0:
        return empty

    gy, gx = block_origins(fm.block_map)
    bs = fm.block_map[gy, gx]
    x0 = gx.astype(np.int64) * 4
    y0 = gy.astype(np.int64) * 4
    w = _BLOCK_W4[bs].astype(np.int64) * 4
    h = _BLOCK_H4[bs].astype(np.int64) * 4
    inside = (x0 < fm.width) & (y0 < fm.height)
    gy, gx, x0, y0, w, h = gy[inside], gx[inside], x0[inside], y0[inside], w[inside], h[inside]
    center = block_centers(x0, y0, w, h, fm.width, fm.height)
    slot_to_frame = np.array((-1,) + fm.ref_frame_index, dtype=np.int64)

    rows: list[tuple] = []
    for lst in (0, 1) if use_compound else (0,):
        slot = fm.ref[gy, gx, lst].astype(np.int64)
        mv = fm.mv[gy, gx, 2 * lst : 2 * lst + 2].astype(np.float64) / 8.0
        ok = slot >= 1
        if skip_zero:
            ok &= np.any(mv != 0.0, axis=-1)
        ref_frame = slot_to_frame[np.clip(slot, 0, 7)]
        ok &= (ref_frame >= 0) & (ref_frame != fm.index)
        if prev_only:
            ok &= ref_frame == fm.index - 1
        if not keep_out_of_frame:
            t = center + mv
            ok &= (t[:, 0] >= 0) & (t[:, 0] < fm.width) & (t[:, 1] >= 0) & (t[:, 1] < fm.height)
        rows.append(
            (
                x0[ok],
                y0[ok],
                w[ok],
                h[ok],
                center[ok],
                mv[ok],
                ref_frame[ok],
                np.full(int(ok.sum()), lst, np.int8),
            )
        )

    cat = [np.concatenate(col) for col in zip(*rows)]
    return BlockMotion(fm.index, *cat)


@dataclass
class MotionLookup:
    """Per-4x4-cell motion used to propagate an arbitrary point to a reference.

    For compound blocks we follow the MV whose reference is temporally nearest
    (list 0 on ties), since tracks link consecutive frames.
    """

    frame: int
    width: int
    height: int
    mv: np.ndarray  # (H/4, W/4, 2) float64 pixels
    ref_frame: np.ndarray  # (H/4, W/4) int64 display index, -1 where unusable
    ref_node: np.ndarray  # (H/4, W/4) int64 decode index of the reference, -1 where unusable

    @classmethod
    def from_frame(
        cls, fm: FrameMotion, *, skip_zero: bool = True, prev_only: bool = False
    ) -> MotionLookup:
        gh, gw = fm.block_map.shape
        if fm.is_intra or fm.mv.size == 0:
            none = np.full((gh, gw), -1, np.int64)
            return cls(fm.index, fm.width, fm.height, np.zeros((gh, gw, 2)), none, none.copy())
        slot_to_frame = np.array((-1,) + fm.ref_frame_index, dtype=np.int64)
        slot_to_node = np.array((-1,) + fm.ref_decode_index, dtype=np.int64)
        cand_mv, cand_ref, cand_node = [], [], []
        for lst in (0, 1):
            slot = np.clip(fm.ref[..., lst].astype(np.int64), 0, 7)  # 0: intra / none
            mv = fm.mv[..., 2 * lst : 2 * lst + 2].astype(np.float64) / 8.0
            ref, node = slot_to_frame[slot], slot_to_node[slot]
            bad = (ref < 0) | (node < 0) | (ref == fm.index)
            if skip_zero:
                bad |= np.all(mv == 0.0, axis=-1)
            if prev_only:
                bad |= ref != fm.index - 1
            cand_mv.append(mv)
            cand_ref.append(np.where(bad, -1, ref))
            cand_node.append(np.where(bad, -1, node))
        # Nearer reference in time, list 0 on ties; an unusable one never wins.
        dist = [np.where(r >= 0, np.abs(fm.index - r), np.iinfo(np.int64).max) for r in cand_ref]
        use1 = dist[1] < dist[0]
        mv = np.where(use1[..., None], cand_mv[1], cand_mv[0])
        ref = np.where(use1, cand_ref[1], cand_ref[0])
        node = np.where(use1, cand_node[1], cand_node[0])
        return cls(fm.index, fm.width, fm.height, mv, ref, node)

    def query(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """MV (pixels) and reference frame (-1 if none) of the cell holding each point."""
        mv, ref, _ = self.query_nodes(pts)
        return mv, ref

    def query_nodes(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MV, reference display index and reference decode index (-1 if none) per point."""
        gh, gw = self.ref_frame.shape
        cx = np.floor(pts[:, 0] / 4.0).astype(np.int64)
        cy = np.floor(pts[:, 1] / 4.0).astype(np.int64)
        inb = (
            (pts[:, 0] >= 0)
            & (pts[:, 0] < self.width)
            & (pts[:, 1] >= 0)
            & (pts[:, 1] < self.height)
            & (cx < gw)
            & (cy < gh)
        )
        cx = np.clip(cx, 0, gw - 1)
        cy = np.clip(cy, 0, gh - 1)
        mv = self.mv[cy, cx]
        ref = np.where(inb, self.ref_frame[cy, cx], -1)
        node = np.where(inb, self.ref_node[cy, cx], -1)
        return mv, ref, node
