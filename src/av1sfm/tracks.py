"""Track building from block motion vectors.

A track is a physical point followed backwards in time through the chain of
motion vectors:

  1. It starts at the centre of a block B in frame n (the source keypoint).
  2. B's MV v_nm moves it to x + v_nm in reference frame m (the target keypoint).
  3. In frame m the point lies inside some block B'; B''s MV v_ml moves it on
     to frame l, and so on until it reaches a block with no usable MV
     (intra, (0,0), out of frame) or the first frame.

"Block" is either a coded block (`grid="block"`, the 4x4 metadata grid
collapsed with the block map) or every 4x4 unit of the zero-order-hold
upsampled motion field (`grid="cell"`, as the follow-up paper arXiv
2605.14629 describes the original pipeline).

Frames are visited from last to first, so by the time frame n is processed all
tracks arriving in it are known. With `seed="all"` (paper: "for each block
(p,q) in a frame n, we emit a source keypoint at the center of the block ...
the generated target point is added to the source keypoints of frame m")
every block of every frame starts a track, next to the points arriving from
later frames. `seed="uncovered"` only seeds blocks no arriving point lands in.

Every consecutive triple (n, m, l) of a track must satisfy
cos(v_nm, v_ml) >= 1 - eps, unless either vector is shorter than tau pixels. On
a violation the track is terminated at m (`cut`, paper: bad MVs "are deleted
and not considered for matches"; the follow-up paper: "they terminate a
trajectory"), split at m into two tracks (`split`) or dropped (`drop`).
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
    on_violation: Literal["cut", "split", "drop"] = "cut"
    grid: Literal["block", "cell"] = "block"
    seed: Literal["all", "uncovered"] = "all"
    skip_zero: bool = True
    prev_only: bool = False

    def __post_init__(self) -> None:
        for name, allowed in (
            ("on_violation", ("cut", "split", "drop")),
            ("grid", ("block", "cell")),
            ("seed", ("all", "uncovered")),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")

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


def _seed_points(
    fm: FrameMotion, lookup: MotionLookup, p: TrackParams, arrived: np.ndarray
) -> np.ndarray:
    """Source keypoints of frame `fm`: centres of its blocks/cells with a usable MV."""
    gh, gw = fm.block_map.shape
    if p.grid == "block":
        gy, gx = block_origins(fm.block_map)
        bs = fm.block_map[gy, gx]
        w4, h4 = _BLOCK_W4[bs], _BLOCK_H4[bs]
        id_map = _block_id_map(fm.block_map)
    else:
        gy, gx = (a.ravel() for a in np.indices((gh, gw)))
        w4 = h4 = np.ones(len(gy), np.int32)
        id_map = np.arange(gh * gw).reshape(gh, gw)
    keep = lookup.ref_frame[gy, gx] >= 0
    keep &= (gx * 4 < fm.width) & (gy * 4 < fm.height)
    if p.seed == "uncovered" and len(arrived):
        cx = np.clip((arrived[:, 0] // 4).astype(np.int64), 0, gw - 1)
        cy = np.clip((arrived[:, 1] // 4).astype(np.int64), 0, gh - 1)
        keep &= ~np.isin(gy * gw + gx, id_map[cy, cx])
    gy, gx, w4, h4 = gy[keep], gx[keep], w4[keep], h4[keep]
    return block_centers(
        gx.astype(np.int64) * 4,
        gy.astype(np.int64) * 4,
        w4.astype(np.int64) * 4,
        h4.astype(np.int64) * 4,
        fm.width,
        fm.height,
    )


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

        # Seed tracks at centres of blocks (or 4x4 cells) with a usable MV.
        s_pts = np.zeros((0, 2))
        if not fm.is_intra and fm.mv.size:
            s_pts = _seed_points(fm, lookup, p, a_pts)
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
                if p.on_violation == "cut":  # the track ends here
                    valid &= ~viol
                elif p.on_violation == "drop":
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


class TooManyMatches(RuntimeError):
    pass


def tracks_to_matches(
    tracks: Tracks, max_pair_gap: int | None = None, max_matches: int | None = 100_000_000
) -> MatchGraph:
    """One keypoint per observation; every pair of observations of a track is a match.

    A track of length L yields L(L-1)/2 matches. `max_matches` guards against
    configurations (e.g. `grid="cell"` on long clips) whose match count would
    exhaust memory; limit the pair gap or raise the cap to proceed.
    """
    if max_matches is not None and max_pair_gap is None:
        lengths = tracks.lengths()
        total = int((lengths * (lengths - 1) // 2).sum())
        if total > max_matches:
            raise TooManyMatches(
                f"{total:,} matches from {tracks.num_tracks:,} tracks exceed max_matches="
                f"{max_matches:,}; pass --max-pair-gap (e.g. 10) or a coarser --grid"
            )
    # Keypoints of a frame are its observations in track order. Frames are also
    # replaced by their rank, which fits in 16 bits, so that sorting tens of
    # millions of matches by frame pair below is a linear-time radix sort.
    n_obs = len(tracks.track)
    by_frame = np.argsort(tracks.frame, kind="stable")
    fs = tracks.frame[by_frame]
    first = np.flatnonzero(np.r_[True, fs[1:] != fs[:-1]]) if n_obs else np.zeros(0, np.int64)
    counts = np.diff(np.r_[first, n_obs])
    n_frames = len(first)
    rank = np.empty(n_obs, np.uint16 if n_frames <= 1 << 16 else np.int64)
    rank[by_frame] = np.repeat(np.arange(n_frames), counts)
    kp_idx = np.empty(n_obs, np.int64)
    kp_idx[by_frame] = np.arange(n_obs) - np.repeat(first, counts)
    keypoints: dict[int, np.ndarray] = {
        int(fs[s]): tracks.xy[by_frame[s : s + c]].astype(np.float32)
        for s, c in zip(first, counts)
    }

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

    # Orient every match from the earlier to the later frame. Tracks are stored
    # in descending frame order, so normally every pair is swapped.
    ra, rb = rank[pa], rank[pb]
    swap = ra > rb
    if swap.all():
        pa, pb, ra, rb = pb, pa, rb, ra
    elif swap.any():
        pa, pb = np.where(swap, pb, pa), np.where(swap, pa, pb)
        ra, rb = rank[pa], rank[pb]
    if max_pair_gap is not None:
        ok = (tracks.frame[pb] - tracks.frame[pa]) <= max_pair_gap
        pa, pb, ra, rb = pa[ok], pb[ok], ra[ok], rb[ok]

    # Group by frame pair, keeping the generation order within a pair: the
    # order of np.lexsort((rb, ra)), as two stable (radix) sorts of 16-bit ranks.
    if rank.dtype == np.uint16:
        o = np.argsort(rb, kind="stable")
        o = o[np.argsort(ra[o], kind="stable")]
    else:
        o = np.lexsort((rb, ra))
    pa, pb, ra, rb = pa[o], pb[o], ra[o], rb[o]
    cuts = np.flatnonzero((ra[1:] != ra[:-1]) | (rb[1:] != rb[:-1])) + 1
    bounds = np.r_[0, cuts, len(pa)] if len(pa) else np.zeros(1, np.int64)
    kp_pairs = np.empty((len(pa), 2), np.uint32)
    kp_pairs[:, 0] = kp_idx[pa]
    kp_pairs[:, 1] = kp_idx[pb]
    matches: dict[tuple[int, int], np.ndarray] = {}
    for s, e in zip(bounds[:-1], bounds[1:]):
        matches[(int(tracks.frame[pa[s]]), int(tracks.frame[pb[s]]))] = kp_pairs[s:e]
    return MatchGraph(keypoints, matches)
