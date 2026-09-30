"""Turn an ordered image sequence into a streaming-configuration AV1 IVF file.

Streaming (low-delay) configuration, as in the paper: a single intra frame at
the start and only past references. With libaom this is `lag-in-frames=0`
(no look-ahead, hence no ALTREF/BWDREF pointing to the future) plus a keyframe
interval longer than the clip. libaom cannot be restricted to LAST-only
prediction (`max-reference-frames` >= 3), so blocks may reference LAST2, LAST3
or GOLDEN; `blocks.frame_block_motion` resolves the actual reference frame.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class EncodeParams:
    encoder: str = "libaom-av1"  # or "av1_nvenc" when an NVIDIA GPU is available
    cpu_used: int = 6  # libaom speed preset (paper: 6)
    crf: int = 30  # constant quality; the paper does not state a rate target
    nvenc_preset: str = "p1"  # paper: NVENC preset 1
    threads: int = 0
    fps: int = 10


def list_images(image_dir: str | Path) -> list[Path]:
    return sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def ffmpeg_command(
    pattern: str, out_ivf: str | Path, params: EncodeParams, vf: str | None
) -> list[str]:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-framerate", str(params.fps), "-i", pattern]  # fmt: skip
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-pix_fmt", "yuv420p", "-c:v", params.encoder]
    if params.encoder == "libaom-av1":
        cmd += [
            "-cpu-used", str(params.cpu_used),
            "-lag-in-frames", "0",
            "-crf", str(params.crf), "-b:v", "0",
            "-row-mt", "1",
            "-threads", str(params.threads),
        ]  # fmt: skip
    elif params.encoder == "av1_nvenc":
        # Low-delay: no B-frames / look-ahead, so all references are in the past.
        cmd += ["-preset", params.nvenc_preset, "-tune", "ll", "-bf", "0",
                "-rc", "constqp", "-qp", str(params.crf)]  # fmt: skip
    else:
        raise ValueError(f"unsupported encoder {params.encoder!r}")
    cmd += ["-g", "1000000", "-keyint_min", "1000000", "-f", "ivf", str(out_ivf)]
    return cmd


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
    params = params or EncodeParams()
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg not found on PATH")
    if not images:
        raise ValueError("no images to encode")
    ext = images[0].suffix.lower()
    Path(out_ivf).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="av1sfm_seq_") as tmp:
        for i, img in enumerate(images):
            (Path(tmp) / f"{i:06d}{ext}").symlink_to(Path(img).resolve())
        vf = f"scale={scale[0]}:{scale[1]}:flags=area" if scale else None
        cmd = ffmpeg_command(str(Path(tmp) / f"%06d{ext}"), out_ivf, params, vf)
        subprocess.run(cmd, check=True)
    return cmd
