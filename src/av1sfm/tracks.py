"""Track building from block motion vectors.

A track is a physical point followed backwards in time through the chain of
motion vectors:

  1. It starts at the centre of a coded block B in frame n (the source keypoint).
  2. B's MV v_nm moves it to x + v_nm in reference frame m (the target keypoint).
  3. In frame m the point lies inside some block B'; B''s MV v_ml moves it on
     to frame l, and so on until it reaches a block with no usable MV
     (intra, (0,0), out of frame) or the first frame.

Frames are visited from last to first, so by the time frame n is processed all
tracks arriving in it are known. Blocks of n that no arriving track lands in
seed new tracks at their centres; blocks that already hold an arriving point
do not, which avoids stacking near-duplicate keypoints on static content.

Every consecutive triple (n, m, l) of a track must satisfy
cos(v_nm, v_ml) >= 1 - eps, unless either vector is shorter than tau pixels. On
a violation the track is either split at m (default) or dropped entirely.
Tracks with fewer than `min_length` observations are discarded, and every
pair of frames on a surviving track becomes a match, which yields matches
between non-adjacent frames.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .blocks import _BLOCK_H4, _BLOCK_W4, MotionLookup, block_centers, block_origins
from .extract import FrameMotion


@dataclass
class TrackParams:
    eps: float = 0.1  # require cos >= 1 - eps; eps >= 1 disables the test (paper)
    tau: float = 2.0  # pixels; skip the cosine test when either MV is shorter (ASSUMPTIONS.md)
    min_length: int = 3
    on_violation: Literal["split", "drop"] = "split"
    skip_zero: bool = True
    prev_only: bool = False

    def __post_init__(self) -> None:
        if self.on_violation not in ("split", "drop"):
            raise ValueError(f"on_violation must be 'split' or 'drop', got {self.on_violation!r}")

    @property
    def cosine_enabled(self) -> bool:
        # Literally, eps = 1 would still reject cos < 0; the paper describes
        # eps = 1 as disabling the filter, so we treat eps >= 1 as "off".
        return self.eps < 1.0


@dataclass
class Tracks:
    """Observations of all kept tracks, sorted by (track, descending frame)."""

    track: np.ndarray  # (K,) int64 compact track ids 0..T-1
    frame: np.ndarray  # (K,) int64 absolute frame index
    xy: np.ndarray  # (K, 2) float64 keypoint position (COLMAP convention)
    stats: dict = field(default_factory=dict)

    @property
    def num_tracks(self) -> int:
        return int(self.track.max()) + 1 if len(self.track) else 0

    def lengths(self) -> np.ndarray:
        return np.bincount(self.track) if len(self.track) else np.zeros(0, np.int64)


def cosine_ok(v1: np.ndarray, v2: np.ndarray, eps: float, tau: float) -> np.ndarray:
    """Vectorised cosine consistency test for consecutive track steps.

    Returns True where cos(v1, v2) >= 1 - eps, or where either vector is
    shorter than `tau` (direction too uncertain to test), or where v1 is NaN
    (no previous step). `eps >= 1` disables the test (paper convention).
    """
    v1 = np.atleast_2d(np.asarray(v1, dtype=np.float64))
    v2 = np.atleast_2d(np.asarray(v2, dtype=np.float64))
    if eps >= 1.0:
        return np.ones(len(v1), dtype=bool)
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    testable = np.isfinite(n1) & (n1 >= tau) & (n2 >= tau)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(v1 * v2, axis=-1) / (n1 * n2)
    return ~testable | (cos >= 1.0 - eps)


def _block_id_map(block_map: np.ndarray) -> np.ndarray:
    """Per 4x4 cell, the linear grid index of its block's top-left cell."""
    gw = block_map.shape[1]
    gy, gx = np.indices(block_map.shape)
    oy = gy - gy % _BLOCK_H4[block_map]
    ox = gx - gx % _BLOCK_W4[block_map]
    return oy * gw + ox


def build_tracks(frames: list[FrameMotion], params: TrackParams | None = None) -> Tracks:
    """Propagate block MVs into tracks (see module docstring)."""
    p = params or TrackParams()
    by_index = {f.index: f for f in frames}
    order = sorted(by_index, reverse=True)

    arrivals: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
    obs_t: list[np.ndarray] = []
    obs_f: list[np.ndarray] = []
    obs_xy: list[np.ndarray] = []
    dropped: list[np.ndarray] = []
    next_id = 0
    n_seeds = n_violations = 0

    def record(ids: np.ndarray, frame: int, pts: np.ndarray) -> None:
        obs_t.append(ids)
        obs_f.append(np.full(len(ids), frame, np.int64))
        obs_xy.append(pts)

    for n in order:
        fm = by_index[n]
        lookup = MotionLookup.from_frame(fm, skip_zero=p.skip_zero, prev_only=p.prev_only)

        if arrivals[n]:
            a_ids, a_pts, a_prev = (np.concatenate(c) for c in zip(*arrivals.pop(n)))
        else:
            arrivals.pop(n, None)
            a_ids = np.zeros(0, np.int64)
            a_pts = np.zeros((0, 2))
            a_prev = np.zeros((0, 2))
        record(a_ids, n, a_pts)

        # Seed tracks at centres of blocks with a usable MV and no arrival.
        s_pts = np.zeros((0, 2))
        if not fm.is_intra and fm.mv.size:
            gy, gx = block_origins(fm.block_map)
            usable = lookup.ref_frame[gy, gx] >= 0
            gy, gx = gy[usable], gx[usable]
            gw = fm.block_map.shape[1]
            if len(a_pts):
                ids_map = _block_id_map(fm.block_map)
                cx = np.clip((a_pts[:, 0] // 4).astype(np.int64), 0, gw - 1)
                cy = np.clip((a_pts[:, 1] // 4).astype(np.int64), 0, fm.block_map.shape[0] - 1)
                covered = np.isin(gy * gw + gx, ids_map[cy, cx])
                gy, gx = gy[~covered], gx[~covered]
            bs = fm.block_map[gy, gx]
            x0, y0 = gx.astype(np.int64) * 4, gy.astype(np.int64) * 4
            inside = (x0 < fm.width) & (y0 < fm.height)
            s_pts = block_centers(
                x0[inside],
                y0[inside],
                _BLOCK_W4[bs][inside] * 4,
                _BLOCK_H4[bs][inside] * 4,
                fm.width,
                fm.height,
            )
        s_ids = np.arange(next_id, next_id + len(s_pts), dtype=np.int64)
        next_id += len(s_pts)
        n_seeds += len(s_pts)
        record(s_ids, n, s_pts)

        ids = np.concatenate([a_ids, s_ids])
        pts = np.concatenate([a_pts, s_pts])
        prev = np.concatenate([a_prev, np.full((len(s_pts), 2), np.nan)])
        if not len(ids):
            continue

        v, m = lookup.query(pts)
        nxt = pts + v
        valid = (m >= 0) & (nxt[:, 0] >= 0) & (nxt[:, 0] < fm.width)
        valid &= (nxt[:, 1] >= 0) & (nxt[:, 1] < fm.height)

        if p.cosine_enabled:
            viol = valid & ~cosine_ok(prev, v, p.eps, p.tau)
            n_violations += int(viol.sum())
            if viol.any():
                if p.on_violation == "drop":
                    dropped.append(ids[viol])
                    valid &= ~viol
                else:  # split: the old track ends here, a new one continues
                    new_ids = np.arange(next_id, next_id + int(viol.sum()), dtype=np.int64)
                    next_id += len(new_ids)
                    record(new_ids, n, pts[viol])
                    ids = ids.copy()
                    ids[viol] = new_ids

        for ref in np.unique(m[valid]):
            sel = valid & (m == ref)
            if int(ref) in by_index:
                arrivals[int(ref)].append((ids[sel], nxt[sel], v[sel]))

    track = np.concatenate(obs_t) if obs_t else np.zeros(0, np.int64)
    frame = np.concatenate(obs_f) if obs_f else np.zeros(0, np.int64)
    xy = np.concatenate(obs_xy) if obs_xy else np.zeros((0, 2))

    keep = np.ones(next_id, dtype=bool)
    if dropped:
        keep[np.concatenate(dropped)] = False
    counts = np.bincount(track, minlength=next_id)
    keep &= counts >= p.min_length
    sel = keep[track]
    track, frame, xy = track[sel], frame[sel], xy[sel]

    remap = np.cumsum(keep) - 1
    track = remap[track]
    o = np.lexsort((-frame, track))
    stats = {
        "seeds": n_seeds,
        "raw_tracks": next_id,
        "cosine_violations": n_violations,
        "kept_tracks": int(keep.sum()),
        "observations": len(track),
    }
    return Tracks(track[o], frame[o], xy[o], stats)


@dataclass
class MatchGraph:
    """Keypoints per frame and matches per frame pair, ready for COLMAP."""

    keypoints: dict[int, np.ndarray]  # frame -> (N, 2) float32
    matches: dict[tuple[int, int], np.ndarray]  # (fa, fb), fa < fb -> (M, 2) uint32 kp idx


def tracks_to_matches(tracks: Tracks, max_pair_gap: int | None = None) -> MatchGraph:
    """One keypoint per observation; every pair of observations of a track is a match."""
    kp_idx = np.zeros(len(tracks.track), np.int64)
    keypoints: dict[int, np.ndarray] = {}
    for f in np.unique(tracks.frame):
        sel = np.flatnonzero(tracks.frame == f)
        kp_idx[sel] = np.arange(len(sel))
        keypoints[int(f)] = tracks.xy[sel].astype(np.float32)

    lengths = tracks.lengths()
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]) if len(lengths) else lengths
    pa_list, pb_list = [], []
    for L in np.unique(lengths):
        if L < 2:
            continue
        s = starts[lengths == L]
        i, j = np.triu_indices(int(L), k=1)
        pa_list.append((s[:, None] + i[None, :]).ravel())
        pb_list.append((s[:, None] + j[None, :]).ravel())
    if not pa_list:
        return MatchGraph(keypoints, {})
    pa = np.concatenate(pa_list)
    pb = np.concatenate(pb_list)

    fa, fb = tracks.frame[pa], tracks.frame[pb]
    swap = fa > fb
    pa, pb = np.where(swap, pb, pa), np.where(swap, pa, pb)
    fa, fb = tracks.frame[pa], tracks.frame[pb]
    if max_pair_gap is not None:
        ok = (fb - fa) <= max_pair_gap
        pa, pb, fa, fb = pa[ok], pb[ok], fa[ok], fb[ok]

    o = np.lexsort((fb, fa))
    pa, pb, fa, fb = pa[o], pb[o], fa[o], fb[o]
    key = fa * (int(fb.max()) + 1 if len(fb) else 1) + fb
    cuts = np.flatnonzero(np.diff(key)) + 1
    matches: dict[tuple[int, int], np.ndarray] = {}
    for seg_a, seg_b in zip(np.split(pa, cuts), np.split(pb, cuts)):
        if not len(seg_a):
            continue
        pair = (int(tracks.frame[seg_a[0]]), int(tracks.frame[seg_b[0]]))
        matches[pair] = np.stack([kp_idx[seg_a], kp_idx[seg_b]], axis=1).astype(np.uint32)
    return MatchGraph(keypoints, matches)
