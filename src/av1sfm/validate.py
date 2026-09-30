"""Check MV sign/direction conventions by warping reference frames.

If the convention "block at p in frame n is predicted from p + mv in frame m"
is right, warping frame m with the MVs reproduces frame n much better than no
warp at all, and much better than the same MVs with the sign flipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .blocks import MotionLookup
from .extract import FrameMotion


@dataclass
class WarpScore:
    frame: int
    coverage: float  # fraction of pixels with a usable MV
    mae_warp: float  # mean |I_n - warp(I_m, +mv)| over covered pixels
    mae_identity: float  # mean |I_n - I_m| (same pixels, same reference)
    mae_flipped: float  # mean |I_n - warp(I_m, -mv)|


def warp_from_references(
    fm: FrameMotion, images: dict[int, np.ndarray], sign: float = 1.0, *, skip_zero: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Predict frame `fm.index` by sampling each pixel's reference at p + sign * mv.

    Returns (prediction float32 HxW[xC], mask of predicted pixels). Pixel (i, j)
    has centre (j + 0.5, i + 0.5) in keypoint coordinates; cv2.remap samples at
    integer-centred coordinates, hence the -0.5 shift.
    """
    lk = MotionLookup.from_frame(fm, skip_zero=skip_zero)
    h, w = fm.height, fm.width
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    pts = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1)
    mv, ref = lk.query(pts)
    tgt = pts + sign * mv - 0.5
    shape = images[next(iter(images))].shape
    pred = np.zeros(shape, np.float32)
    mask = np.zeros((h, w), bool)
    ref = ref.reshape(h, w)
    for m in np.unique(ref):
        if m < 0 or int(m) not in images:
            continue
        sel = ref == m
        mx = tgt[:, 0].reshape(h, w).astype(np.float32)
        my = tgt[:, 1].reshape(h, w).astype(np.float32)
        warped = cv2.remap(images[int(m)].astype(np.float32), mx, my, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)  # fmt: skip
        pred[sel] = warped[sel]
        mask |= sel
    return pred, mask


def score_frame(fm: FrameMotion, images: dict[int, np.ndarray]) -> WarpScore | None:
    if fm.is_intra or fm.index not in images:
        return None
    cur = images[fm.index].astype(np.float32)
    pred, mask = warp_from_references(fm, images, +1.0)
    if not mask.any():
        return None
    flip, _ = warp_from_references(fm, images, -1.0)
    ident, _ = warp_from_references(fm, images, 0.0)
    err = lambda a: float(np.abs(cur - a)[mask].mean())
    return WarpScore(fm.index, float(mask.mean()), err(pred), err(ident), err(flip))
