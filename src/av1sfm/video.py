"""Video files as input: probe, copy an AV1 stream into IVF, extract frames.

An AV1 video already contains the motion vectors, so its stream is copied into
an IVF file without re-encoding (`-c:v copy`), and the decoded frames become the
images for COLMAP. Other codecs have to be encoded to AV1 from the extracted
frames (`reconstruct` does this with the chosen encoder).

Frames are extracted exactly as coded, so that frame i is the i-th frame shown
by the AV1 stream: every frame is kept (`-fps_mode passthrough`), edit lists
are ignored on both paths (they may hide leading frames from the decoder but
not from a stream copy), and the rotation metadata of phone videos is not
applied, because the motion vectors refer to the coded orientation.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .encode import find_ffmpeg, run_ffmpeg

VIDEO_EXTS = {".mp4", ".m4v", ".mov", ".mkv", ".webm", ".ivf", ".obu", ".ts", ".mts", ".avi"}


def find_ffprobe(ffmpeg: str | None = None) -> str:
    """The ffprobe next to the ffmpeg in use, else the one on PATH."""
    sibling = Path(ffmpeg or find_ffmpeg()).with_name("ffprobe")
    if sibling.exists():
        return str(sibling)
    found = shutil.which("ffprobe")
    if found is None:
        raise FileNotFoundError("ffprobe not found (it comes with ffmpeg)")
    return found


@dataclass
class VideoInfo:
    codec: str  # FFmpeg codec name of the first video stream, e.g. "av1", "hevc"
    width: int
    height: int
    frames: int | None  # frame count from the container, if it records one
    rotation: float  # display rotation in degrees (not applied to the frames)
    container: str = ""  # FFmpeg demuxer names, e.g. "mov,mp4,m4a,3gp,3g2,mj2"


def probe_video(path: str | Path, ffmpeg: str | None = None) -> VideoInfo:
    out = subprocess.run(
        [
            find_ffprobe(ffmpeg),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,nb_frames:stream_side_data=rotation:format=format_name",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    probe = json.loads(out)
    streams = probe.get("streams", [])
    if not streams:
        raise ValueError(f"{path}: no video stream")
    s = streams[0]
    nb = s.get("nb_frames")
    rotation = next(
        (float(d["rotation"]) for d in s.get("side_data_list", []) if "rotation" in d), 0.0
    )
    return VideoInfo(
        codec=s["codec_name"],
        width=int(s["width"]),
        height=int(s["height"]),
        frames=int(nb) if nb not in (None, "N/A") else None,
        rotation=rotation,
        container=probe.get("format", {}).get("format_name", ""),
    )


def _input_args(video: str | Path, ffmpeg: str | None) -> list[str]:
    """Input options; MP4 / MOV edit lists are ignored, as they can hide frames."""
    mov = "mov" in probe_video(video, ffmpeg).container.split(",")
    return (["-ignore_editlist", "1"] if mov else []) + ["-i", str(video)]


def copy_av1_to_ivf(video: str | Path, out_ivf: str | Path, ffmpeg: str | None = None) -> list[str]:
    """Copy the first video stream (AV1) into an IVF file, bit for bit. Returns the command."""
    Path(out_ivf).parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg or find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y"]
    cmd += [*_input_args(video, ffmpeg), "-map", "0:v:0", "-c:v", "copy"]
    cmd += ["-f", "ivf", str(out_ivf)]
    subprocess.run(cmd, check=True)
    return cmd


def extract_frames(
    video: str | Path,
    out_dir: str | Path,
    ext: str = "png",
    ffmpeg: str | None = None,
    progress: Callable[[int], None] | None = None,
) -> list[Path]:
    """Decode every frame of the first video stream to `out_dir/000000.<ext>`, ...

    8-bit RGB; JPEG at FFmpeg's highest quality (-q:v 2) if `ext` is "jpg".
    `progress` gets the number of frames written so far.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    cmd = [ffmpeg or find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y"]
    cmd += ["-noautorotate", *_input_args(video, ffmpeg), "-map", "0:v:0"]
    cmd += ["-fps_mode", "passthrough", "-start_number", "0"]
    cmd += ["-pix_fmt", "yuvj420p", "-q:v", "2"] if ext == "jpg" else ["-pix_fmt", "rgb24"]
    cmd += [str(out_dir / f"%06d.{ext}")]
    run_ffmpeg(cmd, progress)
    return sorted(out_dir.glob(f"*.{ext}"))
