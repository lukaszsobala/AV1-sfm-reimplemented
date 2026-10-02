"""Per-frame AV1 block metadata with absolute display and decode indices.

Thin layer over the vendored `dav1d_inspect.iter_frames` (sigmedia/AV1-Optical-Flow).
It resolves AV1's cyclic order hints and per-slot reference order hints into
absolute frame indices, so downstream code can say "this block's MV points into
frame m" without knowing about reference slots.

Any AV1 stream is accepted, not only low-delay ones: with hidden frames
(alt-refs) and future references, frames are decoded out of display order and
a block may be predicted from a later frame. Every decoded frame gets

  * `index`: its display index, i.e. the index of the image it shows. Display
    times come from the order hints; they are ranked, so the shown frames are
    numbered 0 .. N-1 whatever the encoder's order-hint numbering. A hidden
    frame and the overlay frame that later shows it share one display index.
  * `decode_index`: its position in decode order, which identifies the frame
    buffer that later frames reference.

Conventions (verified against a synthetic pan, see tests/test_integration.py):
  * `mv[..., 0:2]` is list 0 (x, y) and `mv[..., 2:4]` is list 1, in 1/8 pel.
  * A block at pixel position p in frame n with MV v and reference slot r is
    predicted from position p + v/8 in frame `ref_frame_index[r - 1]`, which
    is the decoded frame `ref_decode_index[r - 1]`.
  * `ref[..., k]`: 0 = intra, 1..7 = LAST..ALTREF slot, -1 (or <= 0) = none.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._vendor.dav1d_inspect import _DAV1D_LIB, _find_lib, iter_frames

# AV1 streams produced by libaom use 7 order-hint bits by default; the upstream
# AV1-Optical-Flow pipeline makes the same assumption (see ASSUMPTIONS.md).
ORDER_HINT_MODULO = 128

KEY_FRAME = 0
INTRA_ONLY_FRAME = 2


@dataclass
class FrameMotion:
    """Block motion of one decoded frame on the 4x4 grid."""

    index: int  # display index: this frame shows the index-th image
    width: int
    height: int
    frame_type: int
    mv: np.ndarray  # (H/4, W/4, 4) int16, 1/8 pel: [mv0_x, mv0_y, mv1_x, mv1_y]
    ref: np.ndarray  # (H/4, W/4, 2) int16 reference slots
    block_map: np.ndarray  # (H/4, W/4) uint8 AOM BLOCK_* enum
    ref_frame_index: tuple[int, ...]  # display index per slot 1..7, -1 if unknown
    decode_index: int = -1  # position in decode order (default: `index`)
    ref_decode_index: tuple[int, ...] = ()  # decode index per slot (default: display)

    def __post_init__(self) -> None:
        if self.decode_index < 0:
            self.decode_index = self.index
        if not self.ref_decode_index:
            self.ref_decode_index = self.ref_frame_index

    @property
    def is_intra(self) -> bool:
        return self.frame_type in (KEY_FRAME, INTRA_ONLY_FRAME)


def _unwrap(order_hint: int, anchor: int) -> int:
    """Absolute index congruent to `order_hint` (mod 128) closest to `anchor`."""
    base = anchor - ((anchor - order_hint) % ORDER_HINT_MODULO)
    return base + ORDER_HINT_MODULO if anchor - base > ORDER_HINT_MODULO // 2 else base


def resolve_order(
    order_hints: list[int], frame_types: list[int], ref_order_hints: list[list[int]]
) -> tuple[list[int], list[tuple[int, ...]]]:
    """Display index of each decoded frame and decode index of its reference slots.

    Frames are given in decode order. Order hints are unwrapped against the
    previous decoded frame (jumps up to half the order-hint range); a key
    frame whose order hint does not lie after every earlier frame restarts the
    numbering after them. A reference slot holds the most recently decoded
    frame with that order hint since the last key frame (-1 if there is none).
    """
    times: list[int] = []
    refs: list[tuple[int, ...]] = []
    offset = 0  # display time = unwrapped order hint + offset
    unwrapped = 0
    latest: dict[int, int] = {}  # order hint -> decode index of the latest such frame
    for k, (oh, ft, slots) in enumerate(zip(order_hints, frame_types, ref_order_hints)):
        unwrapped = oh if k == 0 else _unwrap(oh, unwrapped)
        t = unwrapped + offset
        if ft == KEY_FRAME:
            if times and t <= max(times):
                offset += max(times) + 1 - t
                t = max(times) + 1
            latest = {}
            refs.append((-1,) * len(slots))
        else:
            refs.append(tuple(latest.get(r % ORDER_HINT_MODULO, -1) for r in slots))
        times.append(t)
        latest[oh % ORDER_HINT_MODULO] = k
    rank = {t: i for i, t in enumerate(sorted(set(times)))}
    return [rank[t] for t in times], refs


def load_frame_motion(ivf_path: str | Path, n_threads: int = 0) -> list[FrameMotion]:
    """Decode `ivf_path` and return a `FrameMotion` per decoded frame, in decode order."""
    if _find_lib(_DAV1D_LIB) is None:
        raise FileNotFoundError(
            f"patched dav1d not found at {_DAV1D_LIB}.*: run `bash setup.sh` in the "
            "repository root (needs meson, ninja, nasm and a C compiler), or set "
            "AV1SFM_BUILD_DIR to an existing build"
        )
    raw = list(iter_frames(ivf_path, n_threads=n_threads))
    display, ref_nodes = resolve_order(
        [int(fr["frame_offset"]) for fr in raw],
        [int(fr["frame_type"]) for fr in raw],
        [[int(r) for r in fr["refpoc"]] for fr in raw],
    )
    return [
        FrameMotion(
            index=display[k],
            width=int(fr["width"]),
            height=int(fr["height"]),
            frame_type=int(fr["frame_type"]),
            mv=fr["motion_vectors"],
            ref=fr["reference_map"],
            block_map=fr["block_map"],
            ref_frame_index=tuple(display[r] if r >= 0 else -1 for r in ref_nodes[k]),
            decode_index=k,
            ref_decode_index=ref_nodes[k],
        )
        for k, fr in enumerate(raw)
    ]


def iter_frame_motion(ivf_path: str | Path, n_threads: int = 0) -> Iterator[FrameMotion]:
    """`load_frame_motion` as an iterator (the whole stream is decoded first)."""
    yield from load_frame_motion(ivf_path, n_threads=n_threads)


def num_shown_frames(frames: list[FrameMotion]) -> int:
    """Number of displayed frames (images) among decoded frames."""
    return len({f.index for f in frames})
