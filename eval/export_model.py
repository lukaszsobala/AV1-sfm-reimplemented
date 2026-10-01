"""Export a sparse reconstruction for viewing, dense reconstruction and splatting.

Writes, under OUT:

  points.ply      coloured sparse point cloud (MeshLab, CloudCompare, Blender)
  dataset/        undistorted COLMAP workspace (PINHOLE camera):
    images/         undistorted images
    sparse/         cameras / images / points3D, binary and text
    sparse/0/       the same files, in the layout splatting tools expect
    stereo/         COLMAP dense-stereo config (unused by OpenMVS / Brush)

None of this needs an NVIDIA GPU. Next steps printed at the end:
OpenMVS (CPU) for a textured mesh, Brush (Vulkan / WebGPU) for a Gaussian splat.

    uv run python eval/export_model.py runs/kitti117/rec_mv runs/kitti117/img runs/kitti117/export_mv

REC may be a model folder (with cameras.bin) or the mapper's output folder,
in which case the model with the most registered images is used.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pycolmap


def pick_model(rec: Path) -> Path:
    """`rec` itself if it holds a model, else its largest numbered sub-model."""
    if (rec / "cameras.bin").exists() or (rec / "cameras.txt").exists():
        return rec
    subs = [d for d in rec.iterdir() if d.is_dir() and d.name.isdigit()]
    if not subs:
        raise FileNotFoundError(f"no COLMAP model in {rec}")
    return max(subs, key=lambda d: pycolmap.Reconstruction(d).num_reg_images())


def as_pinhole(rec: pycolmap.Reconstruction) -> None:
    """Relabel cameras without distortion as PINHOLE, in place.

    COLMAP's undistorter copies images whose camera has zero distortion (e.g.
    SIMPLE_RADIAL with k = 0, our fixed-intrinsics runs) and keeps the model
    name, which importers such as OpenMVS may not accept. The relabelled
    camera is identical.
    """
    for cid, cam in list(rec.cameras.items()):
        if cam.model_name in ("PINHOLE", "SIMPLE_PINHOLE"):
            continue
        extra = [cam.params[i] for i in cam.extra_params_idxs()]
        if any(abs(v) > 1e-12 for v in extra):
            raise ValueError(f"camera {cid} still has distortion {extra} after undistortion")
        rec.cameras[cid] = pycolmap.Camera(
            model="PINHOLE",
            width=cam.width,
            height=cam.height,
            params=[
                cam.focal_length_x,
                cam.focal_length_y,
                cam.principal_point_x,
                cam.principal_point_y,
            ],
            camera_id=cid,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rec", type=Path, help="model folder or mapper output folder")
    ap.add_argument("images", type=Path, help="folder with the images the model was built from")
    ap.add_argument("out", type=Path)
    ap.add_argument(
        "--max-image-size", type=int, default=-1, help="cap the undistorted long side (px)"
    )
    a = ap.parse_args()

    model = pick_model(a.rec)
    rec = pycolmap.Reconstruction(model)
    print(f"model {model}: {rec.num_reg_images()} images, {rec.num_points3D()} points")
    a.out.mkdir(parents=True, exist_ok=True)
    rec.export_PLY(a.out / "points.ply")

    ds = a.out / "dataset"
    if ds.exists():
        shutil.rmtree(ds)
    opts = pycolmap.UndistortCameraOptions()
    opts.max_image_size = a.max_image_size
    pycolmap.undistort_images(ds, model, a.images, undistort_options=opts)

    # Text copies for older readers (OpenMVS InterfaceCOLMAP before binary
    # support), and sparse/0 for tools that expect the 3DGS layout.
    sparse = ds / "sparse"
    und = pycolmap.Reconstruction(sparse)
    as_pinhole(und)
    und.write_binary(sparse)
    und.write_text(sparse)
    (sparse / "0").mkdir()
    und.write_binary(sparse / "0")

    print(f"""
wrote {a.out / "points.ply"} and {ds}/

Textured mesh with OpenMVS (CPU):
  cd {a.out}
  InterfaceCOLMAP -i dataset -o scene.mvs --image-folder dataset/images
  DensifyPointCloud scene.mvs
  ReconstructMesh scene_dense.mvs
  TextureMesh scene_dense.mvs --mesh-file scene_dense_mesh.ply --export-type obj
  -> scene_dense_texture.obj (Blender, MeshLab, Godot, Unity, Unreal)

Gaussian splat with Brush (Vulkan GPU, e.g. Intel Xe2):
  open {ds} in Brush (images/ + sparse/0/)
  -> export .ply, view in Brush or SuperSplat""")


if __name__ == "__main__":
    main()
