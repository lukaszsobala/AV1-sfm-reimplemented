"""End-to-end checks on a real AV1 encode of a synthetic sequence with known motion.

Run once per software encoder (libaom, SVT-AV1) that the ffmpeg in use provides;
skipped when ffmpeg or the patched dav1d shim is unavailable.
"""

import cv2
import numpy as np
import pytest

from av1sfm._vendor import dav1d_inspect
from av1sfm.blocks import frame_block_motion
from av1sfm.encode import EncodeParams, available_encoders, encode_images, find_ffmpeg
from av1sfm.extract import load_frame_motion
from av1sfm.tracks import TrackParams, build_tracks, tracks_to_matches
from av1sfm.validate import score_frame


def _encoders() -> list[str]:
    try:
        return available_encoders(find_ffmpeg())
    except FileNotFoundError, OSError:
        return []


ENCODERS = _encoders()
pytestmark = pytest.mark.skipif(
    not ENCODERS or dav1d_inspect._load_shim() is None,
    reason="needs ffmpeg with an AV1 encoder and the patched dav1d (run setup.sh)",
)

W, H, N = 321, 181, 8  # odd sizes on purpose: the coded frame is padded


def frame_transform(n: int) -> np.ndarray:
    """2x3 map x = s_n * u + b_n from texture coords u to frame n (keypoint convention).

    Content pans by about (+5, +2) px per frame and zooms in by 1 % per frame, so
    the motion field is not a pure translation.
    """
    s = 1.0 + 0.01 * n
    b = np.array([-300.0 + 5.0 * n, -150.0 + 2.0 * n])
    return np.hstack([np.eye(2) * s, b[:, None]])


def to_frame(M: np.ndarray, u: np.ndarray) -> np.ndarray:
    return u @ M[:, :2].T + M[:, 2]


def to_texture(M: np.ndarray, x: np.ndarray) -> np.ndarray:
    return (x - M[:, 2]) @ np.linalg.inv(M[:, :2]).T


@pytest.fixture(scope="module", params=["libaom", "svtav1"])
def clip(request, tmp_path_factory):
    if request.param not in ENCODERS:
        pytest.skip(f"{request.param} not available in this ffmpeg")
    tmp = tmp_path_factory.mktemp(f"clip_{request.param}")
    rng = np.random.default_rng(1)
    tex = cv2.GaussianBlur(rng.random((1000, 1400)).astype(np.float32), (0, 0), 2.5)
    tex = cv2.normalize(tex, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    paths, images = [], {}
    for n in range(N):
        M = frame_transform(n)
        # keypoint coords x_kp = pixel index + 0.5; cv2 works on pixel indices.
        M_cv = M.copy()
        M_cv[:, 2] = M[:, 2] + M[:, :2] @ np.array([0.5, 0.5]) - 0.5
        img = cv2.warpAffine(tex, M_cv, (W, H), flags=cv2.INTER_CUBIC)
        p = tmp / f"{n:03d}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)
        images[n] = img
    ivf = tmp / "clip.ivf"
    encode_images(paths, ivf, EncodeParams(encoder=request.param, crf=20))
    return load_frame_motion(ivf), images


def test_frame_indices_and_references(clip):
    frames, _ = clip
    assert [f.index for f in frames] == list(range(N))
    assert frames[0].is_intra and not frames[1].is_intra
    for f in frames[1:]:  # every reference actually used is an earlier frame
        used = np.unique(f.ref[..., 0][f.ref[..., 0] >= 1])
        assert all(0 <= f.ref_frame_index[s - 1] < f.index for s in used)


def test_block_mvs_match_ground_truth_motion(clip):
    frames, _ = clip
    errs = []
    for f in frames[1:]:
        bm = frame_block_motion(f)
        Mn = frame_transform(f.index)
        u = to_texture(Mn, bm.center)
        for m in np.unique(bm.ref_frame):
            sel = bm.ref_frame == m
            truth = to_frame(frame_transform(int(m)), u[sel])
            errs.append(np.linalg.norm(bm.target[sel] - truth, axis=1))
    err = np.concatenate(errs)
    # Block MVs are a piecewise-translational fit of a zoom; within a 64 px block
    # the zoom term alone is up to ~0.3 px per frame.
    assert np.median(err) < 0.25, np.median(err)
    assert np.mean(err < 2.0) > 0.9


def test_warping_with_mvs_beats_identity_and_flipped_sign(clip):
    frames, images = clip
    imgs = {k: v.astype(np.float32) for k, v in images.items()}
    for f in frames[1:]:
        s = score_frame(f, imgs)
        assert s is not None and s.coverage > 0.8
        assert s.mae_warp < 0.2 * s.mae_identity, s
        assert s.mae_warp < 0.2 * s.mae_flipped, s


def test_tracks_on_real_encode_are_geometrically_consistent(clip):
    frames, _ = clip
    tr = build_tracks(frames, TrackParams())
    assert tr.num_tracks > 100
    # Tracks span the clip. With SVT-AV1's layered low-delay references they may
    # skip frames (e.g. 7 -> 6 -> 4 -> 2 -> 0), so test the span, not the length.
    span = max(np.ptp(tr.frame[tr.track == t]) for t in range(tr.num_tracks))
    assert span >= N - 2
    mg = tracks_to_matches(tr)
    assert (0, N - 1) in mg.matches  # non-adjacent matches from propagation
    errs = []
    for (a, b), m in mg.matches.items():
        u = to_texture(frame_transform(b), mg.keypoints[b][m[:, 1]].astype(np.float64))
        truth = to_frame(frame_transform(a), u)
        errs.append(np.linalg.norm(mg.keypoints[a][m[:, 0]] - truth, axis=1))
    err = np.concatenate(errs)
    assert np.median(err) < 1.5, np.median(err)
