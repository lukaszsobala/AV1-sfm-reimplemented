"""Export a sparse reconstruction for viewing, dense reconstruction and splatting.

Writes, under OUT (see av1sfm.export):

  points.ply      coloured sparse point cloud (MeshLab, CloudCompare, Blender)
  dataset/        undistorted COLMAP workspace (PINHOLE camera) for OpenMVS or Brush

None of this needs an NVIDIA GPU. Next steps printed at the end:
OpenMVS (CPU) for a textured mesh, Brush (Vulkan / WebGPU) for a Gaussian splat.

    uv run python eval/export_model.py runs/kitti117/rec_mv runs/kitti117/img runs/kitti117/export_mv

REC may be a model folder (with cameras.bin) or the mapper's output folder,
in which case the model with the most registered images is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from av1sfm.export import NEXT_STEPS, export_dataset, export_ply, pick_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("rec", type=Path, help="model folder or mapper output folder")
    ap.add_argument("images", type=Path, help="folder with the images the model was built from")
    ap.add_argument("out", type=Path)
    ap.add_argument(
        "--max-image-size", type=int, default=-1, help="cap the undistorted long side (px)"
    )
    a = ap.parse_args()

    model = pick_model(a.rec)
    a.out.mkdir(parents=True, exist_ok=True)
    rec = export_ply(model, a.out / "points.ply")
    print(f"model {model}: {rec.num_reg_images()} images, {rec.num_points3D()} points")
    ds = export_dataset(model, a.images, a.out / "dataset", a.max_image_size)
    print(f"\nwrote {a.out / 'points.ply'} and {ds}/\n")
    print(NEXT_STEPS.format(out=a.out, dataset=ds))


if __name__ == "__main__":
    main()
