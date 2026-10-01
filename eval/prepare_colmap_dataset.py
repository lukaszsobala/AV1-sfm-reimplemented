"""Turn a COLMAP example dataset (Gerrard Hall, Person Hall, ...) into a frame sequence.

The paper (§III-B) uses "a subset of each dataset" because the method "requires
images to have the same dimensions and to be temporally adjacent", and converts
the image sets to videos. This script:

  1. orders the images by file name (capture order: IMG_xxxx),
  2. keeps the longest run of consecutive images with the most common size and
     orientation (Person Hall has two blocks of portrait shots),
  3. resizes them so the long side is `--long-side` px (default 1920),
  4. writes the shared SIMPLE_RADIAL camera (f, cx, cy, k), scaled from the
     dataset's reference reconstruction, to `camera.txt` for all methods.

Download (COLMAP release assets):
  https://github.com/colmap/colmap/releases/download/3.11.1/gerrard-hall.zip
  https://github.com/colmap/colmap/releases/download/3.11.1/person-hall.zip (+ person-hall.z01)

    uv run python eval/prepare_colmap_dataset.py data/colmap/gerrard-hall runs/gerrard-hall
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from av1sfm.encode import list_images


def longest_uniform_run(shapes: list[tuple[int, int]]) -> tuple[int, int]:
    """(start, length) of the longest run of consecutive images with the modal shape."""
    values, counts = np.unique(np.array(shapes), axis=0, return_counts=True)
    modal = tuple(values[np.argmax(counts)])
    best, start = (0, 0), None
    for i, s in enumerate([*shapes, None]):
        if s == modal and start is None:
            start = i
        elif s != modal and start is not None:
            if i - start > best[1]:
                best = (start, i - start)
            start = None
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset", type=Path, help="folder with images/ and sparse/")
    ap.add_argument("out", type=Path, help="output folder (img/ and camera.txt are written)")
    ap.add_argument("--long-side", type=int, default=1920)
    a = ap.parse_args()

    images = list_images(a.dataset / "images")
    shapes = [cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_8).shape[:2] for p in images]
    start, length = longest_uniform_run(shapes)
    subset = images[start : start + length]

    out_img = a.out / "img"
    out_img.mkdir(parents=True, exist_ok=True)
    scale = None
    for p in subset:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)  # applies EXIF orientation
        h, w = img.shape[:2]
        scale = a.long_side / max(h, w)
        small = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_img / (p.stem + ".jpg")), small, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Shared camera from the reference model (OPENCV fx, fy, cx, cy, k1, k2, p1, p2),
    # approximated by SIMPLE_RADIAL (f, cx, cy, k1) and scaled to the new size.
    rec = pycolmap.Reconstruction(a.dataset / "sparse")
    cam = next(iter(rec.cameras.values()))
    fx, fy, cx, cy, k1 = (float(v) for v in cam.params[:5])
    s = a.long_side / max(cam.width, cam.height)
    params = [0.5 * (fx + fy) * s, cx * s, cy * s, k1]
    (a.out / "camera.txt").write_text(",".join(f"{v:.6f}" for v in params) + "\n")
    info = {
        "source": str(a.dataset),
        "images_total": len(images),
        "subset": [subset[0].name, subset[-1].name, len(subset)],
        "long_side": a.long_side,
        "size": [round(cam.width * s), round(cam.height * s)],
        "reference_camera": {"model": cam.model.name, "params": cam.params.tolist()},
        "simple_radial": params,
    }
    (a.out / "prepare.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
