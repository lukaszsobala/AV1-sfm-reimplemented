"""Turn an ordered image sequence into a streaming-configuration AV1 IVF file.

Streaming (low-delay) configuration, as in the paper: a single intra frame at
the start and only past references (no B-frames / look-ahead / ALTREF). Five
FFmpeg backends are supported:

  libaom  libaom-av1, `-usage realtime -cpu-used 6 -lag-in-frames 0 -crf 32`
          (the paper's encoder; CRF 32 is FFmpeg's libaom default).
  svtav1  libsvtav1 (SVT-AV1 >= 2), `pred-struct=1:rtc=1` low-delay real-time
          mode, `keyint=-1` (one keyframe), CRF 32.
  vulkan  av1_vulkan (Vulkan Video AV1 encode, FFmpeg >= 8; Mesa RADV/ANV on
          AMD/Intel GPUs), constant QP, no B-frames.
  qsv     av1_qsv (Intel Quick Sync via oneVPL, Arc / Meteor Lake and newer),
          constant QP, no B-frames, low-delay BRC off look-ahead.
  vaapi   av1_vaapi (VA-API directly, Intel media driver on Arc / Meteor Lake /
          Lunar Lake and newer, or Mesa on AMD), constant qindex, no B-frames.

Measured on KITTI 00 with libaom (eval/mv_stats.py, see ASSUMPTIONS.md): in the
realtime profile ~98 % of MVs reference the previous frame and all are
quarter-pel, matching the paper's description. No encoder can be restricted to
LAST-only prediction, so `blocks.frame_block_motion` always resolves each MV's
actual reference frame.

The ffmpeg binary is `$AV1SFM_FFMPEG` if set, else the one built by
scripts/build_ffmpeg.sh (third_party/build/media/bin/ffmpeg), else `ffmpeg` on
PATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
BACKENDS = ("libaom", "svtav1", "vulkan", "qsv", "vaapi")
FFMPEG_CODEC = {
    "libaom": "libaom-av1",
    "svtav1": "libsvtav1",
    "vulkan": "av1_vulkan",
    "qsv": "av1_qsv",
    "vaapi": "av1_vaapi",
}
HARDWARE = ("vulkan", "qsv", "vaapi")
_LOCAL_FFMPEG = Path(__file__).resolve().parents[2] / "third_party/build/media/bin/ffmpeg"


@dataclass
class EncodeParams:
    encoder: str = "libaom"  # one of BACKENDS, or "auto" (vulkan > qsv > vaapi > svtav1)
    crf: int = 32  # libaom / svtav1 constant quality (0-63)
    qp: int = 128  # vulkan / qsv / vaapi constant AV1 quantizer index (0-255); ~CRF 32
    usage: str = "realtime"  # libaom usage profile: "realtime" or "good"
    cpu_used: int = 6  # libaom speed (paper: 6)
    svt_preset: int = 10  # SVT-AV1 preset (0-13, higher = faster)
    svt_params: str = ""  # extra SVT-AV1 key=value pairs, ':'-separated (appended last)
    hw_device: str | None = None  # vulkan: device index/name; qsv, vaapi: DRM render node
    threads: int = 0
    fps: int = 10


def find_ffmpeg() -> str:
    env = os.environ.get("AV1SFM_FFMPEG")
    if env:
        return env
    if _LOCAL_FFMPEG.exists():
        return str(_LOCAL_FFMPEG)
    found = shutil.which("ffmpeg")
    if found is None:
        raise FileNotFoundError(
            "ffmpeg not found (run scripts/build_ffmpeg.sh or set AV1SFM_FFMPEG)"
        )
    return found


def list_images(image_dir: str | Path) -> list[Path]:
    return sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _hw_args(params: EncodeParams) -> tuple[list[str], str]:
    """(global args initialising the device, filter chain uploading frames)."""
    dev = params.hw_device
    if params.encoder == "vulkan":
        init = f"vulkan=vk:{dev}" if dev else "vulkan=vk"
        return ["-init_hw_device", init, "-filter_hw_device", "vk"], "format=nv12,hwupload"
    if params.encoder == "vaapi":
        init = f"vaapi=va:{dev}" if dev else "vaapi=va"
        return ["-init_hw_device", init, "-filter_hw_device", "va"], "format=nv12,hwupload"
    # QSV on Linux sits on VA-API; an explicit render node selects the GPU.
    if dev:
        init = ["-init_hw_device", f"vaapi=va:{dev}", "-init_hw_device", "qsv=qs@va"]
    else:
        init = ["-init_hw_device", "qsv=qs"]
    return init + ["-filter_hw_device", "qs"], "format=nv12,hwupload=extra_hw_frames=64"


def codec_args(params: EncodeParams, gop: int) -> list[str]:
    """Encoder-specific arguments for a low-delay, single-keyframe stream."""
    e = params.encoder
    if e == "libaom":
        return [
            "-c:v",
            "libaom-av1",
            "-usage",
            params.usage,
            "-cpu-used",
            str(params.cpu_used),
            "-lag-in-frames",
            "0",
            "-crf",
            str(params.crf),
            "-b:v",
            "0",
            "-row-mt",
            "1",
            "-threads",
            str(params.threads),
            "-g",
            str(gop),
            "-keyint_min",
            str(gop),
        ]
    if e == "svtav1":
        svt = "pred-struct=1:rtc=1:keyint=-1"
        if params.threads:
            svt += f":lp={params.threads}"
        if params.svt_params:
            svt += ":" + params.svt_params
        return [
            "-c:v",
            "libsvtav1",
            "-preset",
            str(params.svt_preset),
            "-crf",
            str(params.crf),
            "-svtav1-params",
            svt,
        ]
    if e == "vulkan":
        return [
            "-c:v",
            "av1_vulkan",
            "-rc_mode",
            "cqp",
            "-qp",
            str(params.qp),
            "-bf",
            "0",
            "-tune",
            "ll",
            "-usage",
            "stream",
            "-content",
            "camera",
            "-g",
            str(gop),
        ]
    if e == "qsv":
        return [
            "-c:v",
            "av1_qsv",
            "-preset",
            "veryfast",
            "-q:v",
            str(params.qp),
            "-bf",
            "0",
            "-look_ahead_depth",
            "0",
            "-async_depth",
            "1",
            "-g",
            str(gop),
        ]
    if e == "vaapi":
        # CQP: without -q:v's QSCALE flag, global_quality is the qindex itself.
        return [
            "-c:v",
            "av1_vaapi",
            "-rc_mode",
            "CQP",
            "-global_quality",
            str(params.qp),
            "-bf",
            "0",
            "-async_depth",
            "1",
            "-g",
            str(gop),
        ]
    raise ValueError(f"unknown encoder {e!r}; choose from {BACKENDS}")


def ffmpeg_command(
    input_args: list[str],
    out_ivf: str | Path,
    params: EncodeParams,
    *,
    num_frames: int,
    scale: tuple[int, int] | None = None,
    size: tuple[int, int] | None = None,
    ffmpeg: str | None = None,
) -> list[str]:
    """`size` is the input frame size (width, height), used to pad odd sizes for SVT-AV1."""
    cmd = [ffmpeg or find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y"]
    filters = [f"scale={scale[0]}:{scale[1]}:flags=area"] if scale else []
    size = scale or size
    if params.encoder == "svtav1" and size and (size[0] % 2 or size[1] % 2):
        # SVT-AV1 2.x rejects odd sizes for 4:2:0. Repeat the last column / row,
        # as the encoder's own padding to its block grid would; the decoded
        # frame is then clamped back to the image size (clamp_to_image_size).
        dx, dy = size[0] % 2, size[1] % 2
        filters.append(
            f"pad={size[0] + dx}:{size[1] + dy},"
            f"fillborders=right={dx}:bottom={dy}:mode=smear"
        )
    if params.encoder in HARDWARE:
        init, upload = _hw_args(params)
        cmd += init
        filters.append(upload)
    else:
        filters.append("format=yuv420p")
    cmd += [*input_args, "-vf", ",".join(filters)]
    cmd += codec_args(params, gop=num_frames + 1)
    cmd += ["-f", "ivf", str(out_ivf)]
    return cmd


def available_encoders(ffmpeg: str | None = None) -> list[str]:
    """Backends whose FFmpeg encoder is compiled in (not that the hardware works)."""
    out = subprocess.run(
        [ffmpeg or find_ffmpeg(), "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    names = {line.split()[1] for line in out.splitlines() if len(line.split()) > 1}
    return [b for b in BACKENDS if FFMPEG_CODEC[b] in names]


def probe_encoder(
    params: EncodeParams, ffmpeg: str | None = None, *, verbose: bool = False
) -> tuple[bool, str]:
    """Try a 3-frame test encode; returns (works, error message)."""
    with tempfile.TemporaryDirectory(prefix="av1sfm_probe_") as tmp:
        out = Path(tmp) / "probe.ivf"
        src = ["-f", "lavfi", "-i", "testsrc2=size=320x240:rate=10", "-frames:v", "3"]
        cmd = ffmpeg_command(src, out, params, num_frames=3, ffmpeg=ffmpeg)
        if verbose:  # device and encoder-open errors are only logged at this level
            cmd[cmd.index("-loglevel") + 1] = "verbose"
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        ok = r.returncode == 0 and out.exists() and out.stat().st_size > 32
        return ok, "" if ok else _first_error(r.stderr)


def _first_error(stderr: str) -> str:
    """Most specific line of an ffmpeg failure (the last line is usually generic)."""
    lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
    for key in (
        "not support",
        "No device",
        "Failed to",
        "Error creating",
        "Error parsing",
        "Error initializing",
        "Error while opening",
    ):
        for ln in lines:
            if key.lower() in ln.lower():
                return ln
    return lines[-1] if lines else "failed"


def resolve_encoder(params: EncodeParams, ffmpeg: str | None = None) -> EncodeParams:
    """Replace encoder="auto" by the first working backend: vulkan, qsv, vaapi, svtav1."""
    if params.encoder != "auto":
        return params
    for backend in ("vulkan", "qsv", "vaapi", "svtav1"):
        cand = EncodeParams(**{**params.__dict__, "encoder": backend})
        if backend in available_encoders(ffmpeg) and probe_encoder(cand, ffmpeg)[0]:
            return cand
    raise RuntimeError("no working AV1 encoder among vulkan, qsv, vaapi, svtav1")


def encode_images(
    images: list[Path],
    out_ivf: str | Path,
    params: EncodeParams | None = None,
    *,
    scale: tuple[int, int] | None = None,
) -> list[str]:
    """Encode `images` (in order) to `out_ivf`. Returns the ffmpeg command used.

    Frames are linked into a temporary numbered sequence so arbitrary file
    names and subsets work. `scale=(w, h)` resizes before encoding.
    """
    params = resolve_encoder(params or EncodeParams())
    if not images:
        raise ValueError("no images to encode")
    ext = images[0].suffix.lower()
    Path(out_ivf).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="av1sfm_seq_") as tmp:
        for i, img in enumerate(images):
            (Path(tmp) / f"{i:06d}{ext}").symlink_to(Path(img).resolve())
        src = ["-framerate", str(params.fps), "-i", str(Path(tmp) / f"%06d{ext}")]
        size = None
        if params.encoder == "svtav1" and scale is None:
            h, w = cv2.imread(str(images[0]), cv2.IMREAD_UNCHANGED).shape[:2]
            size = (w, h)
        cmd = ffmpeg_command(
            src, out_ivf, params, num_frames=len(images), scale=scale, size=size
        )
        subprocess.run(cmd, check=True)
    return cmd
