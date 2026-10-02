"""`av1sfm reconstruct` end to end on a synthetic 3D scene with known camera motion.

Two textured fronto-parallel planes (depths 5 and 14) seen by a camera moving
sideways in equal steps: enough parallax for SfM, and the recovered camera
centres must be equally spaced on a line. Skipped without ffmpeg (libaom) or
the patched dav1d.
"""

import subprocess

import cv2
import numpy as np
import pycolmap
import pytest

from av1sfm._vendor import dav1d_inspect
from av1sfm.colmap_db import CameraSpec
from av1sfm.encode import available_encoders, find_ffmpeg
from av1sfm.pipeline import MVMatchConfig
from av1sfm.reconstruct import ReconstructConfig, reconstruct


def _encoders() -> list[str]:
    try:
        return available_encoders(find_ffmpeg())
    except FileNotFoundError, OSError:
        return []


pytestmark = pytest.mark.skipif(
    "libaom" not in _encoders() or dav1d_inspect._load_shim() is None,
    reason="needs ffmpeg with libaom and the patched dav1d (run setup.sh)",
)

W, H, F, N = 320, 240, 300.0, 14
CAMERA = CameraSpec(params=(F, W / 2, H / 2, 0.0))


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    rng = np.random.default_rng(0)

    def texture(n):
        t = cv2.GaussianBlur(rng.random((n, n)).astype(np.float32), (0, 0), 2.0)
        return cv2.normalize(t, t, 0, 255, cv2.NORM_MINMAX).astype(np.float32)

    near, far = texture(1024), texture(2048)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    out = tmp_path_factory.mktemp("scene") / "img"
    out.mkdir()
    for k in range(N):
        tx = 0.25 * k  # camera centre (tx, 0, 0), looking along +z

        def plane(tex, depth, px_per_unit, tx=tx):
            X = tx + (u - W / 2) * depth / F
            Y = (v - H / 2) * depth / F
            mx = (X * px_per_unit + tex.shape[1] / 2 - 0.5).astype(np.float32)
            my = (Y * px_per_unit + tex.shape[0] / 2 - 0.5).astype(np.float32)
            return cv2.remap(tex, mx, my, cv2.INTER_LINEAR)

        x_near = tx + (u - W / 2) * 5.0 / F
        img = np.where(np.abs(x_near) < 1.5, plane(near, 5.0, 60.0), plane(far, 14.0, 40.0))
        cv2.imwrite(str(out / f"{k:03d}.png"), np.clip(img, 0, 255).astype(np.uint8))
    return out


def video(frames, path, codec_args):
    cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-framerate", "10"]
    cmd += ["-i", str(frames / "%03d.png"), "-pix_fmt", "yuv420p", *codec_args, str(path)]
    subprocess.run(cmd, check=True)
    return path


def check_trajectory(st, min_registered=N):
    assert st["registered_images"] >= min_registered, st
    assert st["mean_reprojection_error_px"] < 0.5
    rec = pycolmap.Reconstruction(st["model"])
    order = sorted(rec.images.values(), key=lambda im: im.name)
    centres = np.array([im.projection_center() for im in order])
    steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    assert np.all(np.abs(steps / steps.mean() - 1) < 0.1), steps  # equal steps
    d = centres - centres.mean(0)
    s = np.linalg.svd(d, compute_uv=False)
    assert s[1] < 0.02 * s[0], s  # on a line


def test_image_folder_with_motion_vectors(frames, tmp_path):
    cfg = ReconstructConfig(mv=MVMatchConfig(camera=CAMERA), fix_intrinsics=True)
    st = reconstruct(frames, tmp_path, cfg)
    assert "encode" in st["timings"] and st["matching"]["stream"]["low_delay"]
    check_trajectory(st)
    assert (tmp_path / "points.ply").stat().st_size > 1000
    assert (tmp_path / "reconstruct.json").exists()


def test_av1_video_is_copied_not_reencoded(frames, tmp_path):
    # A random-access stream, as libaom writes by default (hidden alt-refs).
    mp4 = video(frames, tmp_path / "in.mp4", ["-c:v", "libaom-av1", "-cpu-used", "8", "-crf", "20"])
    cfg = ReconstructConfig(mv=MVMatchConfig(camera=CAMERA), fix_intrinsics=True)
    st = reconstruct(mp4, tmp_path / "out", cfg)
    assert "copy" in st["timings"] and "encode" not in st["timings"]
    assert st["video"]["codec"] == "av1" and st["num_images"] == N
    assert not st["matching"]["stream"]["low_delay"]
    check_trajectory(st, min_registered=N - 2)


def test_other_codecs_are_encoded_to_av1(frames, tmp_path):
    mkv = video(frames, tmp_path / "in.mkv", ["-c:v", "ffv1"])
    cfg = ReconstructConfig(mv=MVMatchConfig(camera=CAMERA), fix_intrinsics=True)
    st = reconstruct(mkv, tmp_path / "out", cfg)
    assert st["video"]["codec"] == "ffv1" and "encode" in st["timings"]
    check_trajectory(st)


@pytest.mark.parametrize("matcher", ["colmap", "exact"])
def test_sift_matcher(frames, tmp_path, matcher):
    if matcher == "exact":
        pytest.importorskip("torch")
    cfg = ReconstructConfig(
        matcher="sift",
        sift_matcher=matcher,
        device="cpu",
        mv=MVMatchConfig(camera=CAMERA),
        fix_intrinsics=True,
    )
    st = reconstruct(frames, tmp_path, cfg)
    assert st["matching"]["config"]["matcher"] == matcher
    check_trajectory(st)
