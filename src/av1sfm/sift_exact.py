"""Exact SIFT descriptor matching on a PyTorch device (Intel GPU, CUDA, CPU).

Reproduces COLMAP's exact CPU matcher (`SiftMatchingOptions.cpu_brute_force_matcher`,
`FindBestMatchesBruteForce` in colmap/feature/sift.cc, version 4.2.1): for each
descriptor, the best and second-best dot products over all descriptors of the
other image; the match is kept if its angular distance acos(dot / 512²) is at
most `max_distance` and below `max_ratio` times the second best; with
`cross_check`, only mutual matches are kept.

COLMAP's default CPU matcher is an approximate (FAISS) search whose result
depends on the thread count (ASSUMPTIONS.md R4); the exact one is ~40x slower on
the CPU. On a GPU the exact search is one matrix product per image pair.

Exactness: descriptors are uint8 (RootSIFT scaled to norm 512), so every dot
product is an integer below 512² = 2^18 and is exact in float32 whatever the
summation order, provided the device really multiplies in float32 (no TF32 or
bfloat16 shortcuts). We request that, and `check=True` recomputes the selected
dot products in integer arithmetic on the CPU and raises on any difference.
Ties need no special handling: COLMAP rejects a best match tied with the second
best (ratio test with >=), whichever index it is.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pycolmap

from .devices import import_torch

SQ_DESCRIPTOR_NORM = 512 * 512


def one_way(
    best: np.ndarray, second: np.ndarray, idx: np.ndarray, max_ratio: float, max_distance: float
) -> np.ndarray:
    """COLMAP's one-way test on best / second-best dot products; -1 where rejected.

    Thresholds are compared in float32 as in COLMAP; acos is evaluated in float64
    and rounded, i.e. correctly rounded float32.
    """

    def dist(dot: np.ndarray) -> np.ndarray:
        return np.arccos(np.minimum(dot.astype(np.float64) / SQ_DESCRIPTOR_NORM, 1.0)).astype(
            np.float32
        )

    b, s = dist(best), dist(second)
    ok = (best > 0) & (b <= np.float32(max_distance)) & (b < np.float32(max_ratio) * s)
    return np.where(ok, idx, -1)


def _top2(scores, dim: int):
    """Top-2 values and indices along `dim`, as (n, 2) CPU arrays; second = 0 if absent."""
    torch = import_torch()
    k = min(2, scores.shape[dim])
    v, i = torch.topk(scores, k, dim=dim)
    v, i = v.cpu().numpy(), i.cpu().numpy()
    if dim == 0:
        v, i = v.T, i.T
    if k == 1:
        v = np.concatenate([v, np.zeros_like(v)], axis=1)
        i = np.concatenate([i, np.full_like(i, -1)], axis=1)
    return v, i


def _check_exact(d1: np.ndarray, d2: np.ndarray, v: np.ndarray, i: np.ndarray) -> None:
    """Recompute rows' top-2 dot products of d1 against d2 in int64 and compare."""
    for c in range(2):
        valid = i[:, c] >= 0
        exact = np.einsum(
            "ij,ij->i",
            d1[valid].astype(np.int64),
            d2[i[valid, c]].astype(np.int64),
        )
        if not np.array_equal(exact, v[valid, c].astype(np.int64)):
            bad = int(np.sum(exact != v[valid, c].astype(np.int64)))
            raise RuntimeError(
                f"device dot products are not exact ({bad} differ): the device uses reduced "
                "float32 matmul precision; run with --device cpu or report this"
            )


def match_descriptors(
    d1: np.ndarray,
    d2: np.ndarray,
    *,
    device=None,
    max_ratio: float = 0.8,
    max_distance: float = 0.7,
    cross_check: bool = True,
    check: bool = True,
    t1=None,
    t2=None,
) -> np.ndarray:
    """Matches (k, 2) uint32 between uint8 descriptor arrays d1 (n1, 128), d2 (n2, 128).

    `t1` / `t2` may hold the same descriptors already on `device` (any dtype).
    """
    torch = import_torch()
    if len(d1) == 0 or len(d2) == 0:
        return np.zeros((0, 2), np.uint32)
    device = device or torch.device("cpu")
    t1 = torch.tensor(d1, device=device) if t1 is None else t1
    t2 = torch.tensor(d2, device=device) if t2 is None else t2
    with torch.no_grad():
        scores = t1.float() @ t2.float().T
        rv, ri = _top2(scores, 1)
        cv, ci = _top2(scores, 0) if cross_check else (None, None)
        del scores
    if check:
        _check_exact(d1, d2, rv, ri)
        if cross_check:
            _check_exact(d2, d1, cv, ci)
    m12 = one_way(rv[:, 0], rv[:, 1], ri[:, 0], max_ratio, max_distance)
    i1 = np.flatnonzero(m12 >= 0)
    if cross_check:
        m21 = one_way(cv[:, 0], cv[:, 1], ci[:, 0], max_ratio, max_distance)
        i1 = i1[m21[m12[i1]] == i1]
    return np.stack([i1, m12[i1]], axis=1).astype(np.uint32)


def generate_pairs(
    db: pycolmap.Database, matching: str, overlap: int = 10
) -> list[tuple[int, int]]:
    """Image-id pairs exactly as COLMAP's sequential / exhaustive matchers generate them."""
    if matching == "sequential":
        opts = pycolmap.SequentialPairingOptions()
        opts.overlap = overlap
        opts.quadratic_overlap = False
        gen = pycolmap.SequentialPairGenerator(opts, db)
    elif matching == "exhaustive":
        gen = pycolmap.ExhaustivePairGenerator(pycolmap.ExhaustivePairingOptions(), db)
    else:
        raise ValueError(matching)
    return [tuple(p) for p in gen.all_pairs()]


def match_database(
    db_path: str | Path,
    pairs: Sequence[tuple[int, int]],
    device,
    options: pycolmap.SiftMatchingOptions | None = None,
    *,
    check: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Match the SIFT descriptors stored in a COLMAP database and write raw matches.

    Two-view geometries are not computed; run `pycolmap.geometric_verification`
    afterwards, as COLMAP's matchers do.
    """
    torch = import_torch()
    torch.set_float32_matmul_precision("highest")
    options = options or pycolmap.SiftMatchingOptions()
    desc: dict[int, np.ndarray] = {}
    on_device: dict[int, object] = {}

    def get(image_id: int):
        if image_id not in desc:
            d = db.read_descriptors(image_id)
            if d.type != pycolmap.FeatureExtractorType.SIFT:
                raise ValueError(f"image {image_id}: descriptors are {d.type}, not SIFT")
            desc[image_id] = np.array(d.data, np.uint8)
            on_device[image_id] = torch.as_tensor(desc[image_id], device=device)
        return desc[image_id], on_device[image_id]

    stats = {"pairs": 0, "raw_matches": 0}
    with pycolmap.Database.open(db_path) as db, pycolmap.DatabaseTransaction(db):
        for n, (a, b) in enumerate(pairs):
            (d1, t1), (d2, t2) = get(a), get(b)
            m = match_descriptors(
                d1,
                d2,
                device=device,
                max_ratio=options.max_ratio,
                max_distance=options.max_distance,
                cross_check=options.cross_check,
                check=check,
                t1=t1,
                t2=t2,
            )
            db.write_matches(a, b, m)
            stats["pairs"] += 1
            stats["raw_matches"] += len(m)
            if progress:
                progress(n + 1, len(pairs))
    return stats
