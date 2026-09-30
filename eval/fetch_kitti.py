"""Fetch the first N frames of a KITTI odometry sequence without the full archive.

The official zips support HTTP range requests, so only the needed members are
downloaded (about 0.8 MB per colour frame instead of the 65 GB archive).

    uv run python eval/fetch_kitti.py --sequence 00 --frames 230 --camera image_2 --out data/kitti
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from remotezip import RemoteZip

BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
ARCHIVES = {"image_2": "data_odometry_color.zip", "image_3": "data_odometry_color.zip",
            "image_0": "data_odometry_gray.zip", "image_1": "data_odometry_gray.zip"}  # fmt: skip


def extract(url: str, members: list[str], out: Path) -> None:
    with RemoteZip(url) as z:
        names = set(z.namelist())
        for m in members:
            if m not in names:
                raise KeyError(f"{m} not in {url}")
            dst = out / Path(m).name
            if dst.exists():
                continue
            with z.open(m) as src, open(dst, "wb") as f:
                shutil.copyfileobj(src, f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sequence", default="00")
    ap.add_argument("--frames", type=int, default=230)
    ap.add_argument("--camera", default="image_2", choices=sorted(ARCHIVES))
    ap.add_argument("--out", type=Path, default=Path("data/kitti"))
    a = ap.parse_args()
    seq_dir = a.out / a.sequence
    img_dir = seq_dir / a.camera
    img_dir.mkdir(parents=True, exist_ok=True)
    extract(
        BASE + "data_odometry_calib.zip", [f"dataset/sequences/{a.sequence}/calib.txt"], seq_dir
    )
    if int(a.sequence) <= 10:
        extract(BASE + "data_odometry_poses.zip", [f"dataset/poses/{a.sequence}.txt"], seq_dir)
    members = [f"dataset/sequences/{a.sequence}/{a.camera}/{i:06d}.png" for i in range(a.frames)]
    extract(BASE + ARCHIVES[a.camera], members, img_dir)
    print(f"{len(list(img_dir.glob('*.png')))} frames in {img_dir}")


if __name__ == "__main__":
    main()
