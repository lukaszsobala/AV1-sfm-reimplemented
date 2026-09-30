"""Pairwise geometric scoring of matches, identical for every method.

For each image pair an essential matrix (five-point) and a homography (DLT)
are fitted with COLMAP's LO-RANSAC on the raw matches. Both work in
normalised camera coordinates K^-1 x; the pixel threshold is converted with
the mean focal length (for E pycolmap does this conversion itself). The
model with more inliers wins; ties go to the lower median residual.

Reported per pair: inlier ratio (inliers / raw matches) and the median
Sampson error over the winning model's inliers, as a distance
(sqrt of the Sampson squared error) in normalised units and converted to
pixels (x f). For a homography we use its Sampson (first-order geometric)
error, the analogue of the epipolar Sampson error (Hartley & Zisserman, 4.2.6).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pycolmap


@dataclass
class RansacSettings:
    max_error: float = 4.0  # pixels
    min_inlier_ratio: float = 0.25
    max_num_trials: int = 10000
    random_seed: int = 0
    repeats: int = 1  # independent RANSAC runs (seeds random_seed ..); medians are reported

    def options(self, scale: float = 1.0, seed_offset: int = 0) -> pycolmap.RANSACOptions:
        o = pycolmap.RANSACOptions()
        o.max_error = self.max_error * scale
        o.min_inlier_ratio = self.min_inlier_ratio
        o.max_num_trials = self.max_num_trials
        o.random_seed = self.random_seed + seed_offset
        return o


@dataclass
class PairScore:
    image1: str
    image2: str
    num_matches: int
    model: str  # "E", "H" or "none" (most frequent over repeats)
    num_inliers: int  # median over repeats
    inlier_ratio: float  # median over repeats
    median_sampson_norm: float  # Sampson distance, normalised units
    median_sampson_px: float  # Sampson distance x mean focal length
    median_sampson_sq_norm: float = float("nan")  # paper's SE (squared), normalised units

    def asdict(self) -> dict:
        return asdict(self)


def normalize(cam: pycolmap.Camera, pts: np.ndarray) -> np.ndarray:
    """K^-1 x (undistorted), using COLMAP's pixel convention."""
    return np.asarray(cam.cam_from_img(np.asarray(pts, np.float64)), np.float64)


def _h(x: np.ndarray) -> np.ndarray:
    return np.hstack([x, np.ones((len(x), 1))])


def sampson_epipolar(E: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Sampson distance for x2^T E x1 = 0 (normalised coordinates)."""
    X1, X2 = _h(x1), _h(x2)
    Ex1 = X1 @ E.T
    Etx2 = X2 @ E
    num = np.sum(X2 * Ex1, axis=1)
    den = Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2 + Etx2[:, 0] ** 2 + Etx2[:, 1] ** 2
    return np.abs(num) / np.sqrt(np.maximum(den, 1e-300))


def sampson_homography(H: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Sampson (first-order geometric) distance for x2 ~ H x1.

    The algebraic residual is the first two rows of [x2]_x H x1; its Jacobian
    w.r.t. (x1, y1, x2, y2) gives the first-order correction.
    """
    X1 = _h(x1)
    Hx = X1 @ H.T  # (N, 3): a, b, c
    a, b, c = Hx[:, 0], Hx[:, 1], Hx[:, 2]
    u, v = x2[:, 0], x2[:, 1]
    # [x2]_x (Hx1), rows 0 and 1 with x2 = (u, v, 1)
    e1 = v * c - b
    e2 = a - u * c

    # d(Hx)/dx1 = H[:, 0], d(Hx)/dy1 = H[:, 1]
    def de1(col):
        return v * H[2, col] - H[1, col]

    def de2(col):
        return H[0, col] - u * H[2, col]

    J1 = np.stack([de1(0), de1(1), np.zeros_like(u), c], axis=1)
    J2 = np.stack([de2(0), de2(1), -c, np.zeros_like(u)], axis=1)
    e = np.stack([e1, e2], axis=1)
    JJt = np.empty((len(u), 2, 2))
    JJt[:, 0, 0] = np.sum(J1 * J1, axis=1)
    JJt[:, 0, 1] = JJt[:, 1, 0] = np.sum(J1 * J2, axis=1)
    JJt[:, 1, 1] = np.sum(J2 * J2, axis=1)
    sol = np.linalg.solve(JJt + np.eye(2) * 1e-300, e[..., None])[..., 0]
    return np.sqrt(np.maximum(np.sum(e * sol, axis=1), 0.0))


def score_pair(
    cam1: pycolmap.Camera,
    cam2: pycolmap.Camera,
    pts1: np.ndarray,
    pts2: np.ndarray,
    settings: RansacSettings | None = None,
    names: tuple[str, str] = ("", ""),
) -> PairScore:
    """Fit E and H to matched pixel coordinates and score the better model.

    With `settings.repeats > 1` the estimation is repeated with different
    RANSAC seeds and the per-run metrics are summarised by their median, as in
    the paper ("multiple independent runs ... median performance metrics").
    """
    s = settings or RansacSettings()
    n = len(pts1)
    nan = float("nan")
    none = PairScore(names[0], names[1], n, "none", 0, 0.0, nan, nan, nan)
    if n < 5:
        return none
    f = 0.5 * (cam1.mean_focal_length() + cam2.mean_focal_length())
    x1, x2 = normalize(cam1, pts1), normalize(cam2, pts2)

    runs = []
    for r in range(max(1, s.repeats)):
        cands = []
        e = pycolmap.estimate_essential_matrix(pts1, pts2, cam1, cam2, s.options(seed_offset=r))
        if e is not None and e["num_inliers"] > 0:
            mask = np.asarray(e["inlier_mask"], bool)
            cands.append(("E", int(mask.sum()), sampson_epipolar(e["E"], x1[mask], x2[mask])))
        h = pycolmap.estimate_homography_matrix(x1, x2, s.options(1.0 / f, seed_offset=r))
        if h is not None and h["num_inliers"] > 0:
            mask = np.asarray(h["inlier_mask"], bool)
            cands.append(("H", int(mask.sum()), sampson_homography(h["H"], x1[mask], x2[mask])))
        if cands:
            model, k, res = min(cands, key=lambda c: (-c[1], float(np.median(c[2]))))
            runs.append((model, k, float(np.median(res)), float(np.median(res**2))))
    if not runs:
        return none
    models = [r[0] for r in runs]
    model = max(set(models), key=models.count)
    k = int(np.median([r[1] for r in runs]))
    med = float(np.median([r[2] for r in runs]))
    med_sq = float(np.median([r[3] for r in runs]))
    return PairScore(names[0], names[1], n, model, k, k / n, med, med * f, med_sq)


def summarize(scores: list[PairScore]) -> dict:
    """Aggregate per-pair scores (pairs with >= 1 match)."""
    sc = [s for s in scores if s.num_matches > 0]
    if not sc:
        return {"pairs": 0}
    ratio = np.array([s.inlier_ratio for s in sc])
    med_px = np.array([s.median_sampson_px for s in sc if s.num_inliers > 0])
    tot_m = sum(s.num_matches for s in sc)
    tot_i = sum(s.num_inliers for s in sc)
    return {
        "pairs": len(sc),
        "matches_total": tot_m,
        "inliers_total": tot_i,
        "inlier_ratio_mean": float(ratio.mean()),
        "inlier_ratio_pooled": tot_i / tot_m if tot_m else float("nan"),
        "sampson_px_median_of_pairs": float(np.median(med_px)) if len(med_px) else float("nan"),
        "sampson_sq_norm_median_of_pairs": float(
            np.median([s.median_sampson_sq_norm for s in sc if s.num_inliers > 0])
        )
        if len(med_px)
        else float("nan"),
        "frac_E": float(np.mean([s.model == "E" for s in sc])),
    }
