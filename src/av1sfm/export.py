"""Export a sparse reconstruction for viewing, dense reconstruction and splatting.

  points.ply      coloured sparse point cloud (MeshLab, CloudCompare, Blender)
  dataset/        undistorted COLMAP workspace (PINHOLE camera):
    images/         undistorted images
    sparse/         cameras / images / points3D, binary and text
    sparse/0/       the same files, in the layout splatting tools expect
    stereo/         COLMAP dense-stereo config (unused by OpenMVS / Brush)

None of this needs an NVIDIA GPU: OpenMVS (CPU) makes a textured mesh from
`dataset/`, Brush (Vulkan / WebGPU) a Gaussian splat.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pycolmap


def pick_model(rec: str | Path) -> Path:
    """`rec` itself if it holds a model, else its numbered sub-model with the most images."""
    rec = Path(rec)
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


def export_ply(model: str | Path, out_ply: str | Path) -> pycolmap.Reconstruction:
    rec = pycolmap.Reconstruction(model)
    rec.export_PLY(out_ply)
    return rec


def export_dataset(
    model: str | Path, image_dir: str | Path, out_dir: str | Path, max_image_size: int = -1
) -> Path:
    """Undistorted COLMAP workspace with a PINHOLE camera; returns its path."""
    ds = Path(out_dir)
    if ds.exists():
        shutil.rmtree(ds)
    opts = pycolmap.UndistortCameraOptions()
    opts.max_image_size = max_image_size
    pycolmap.undistort_images(ds, model, image_dir, undistort_options=opts)
    # Text copies for older readers (OpenMVS InterfaceCOLMAP before binary
    # support), and sparse/0 for tools that expect the 3DGS layout.
    sparse = ds / "sparse"
    und = pycolmap.Reconstruction(sparse)
    as_pinhole(und)
    und.write_binary(sparse)
    und.write_text(sparse)
    (sparse / "0").mkdir()
    und.write_binary(sparse / "0")
    return ds


NEXT_STEPS = """Textured mesh with OpenMVS (CPU):
  cd {out}
  InterfaceCOLMAP -i dataset -o scene.mvs --image-folder dataset/images
  DensifyPointCloud scene.mvs
  ReconstructMesh scene_dense.mvs
  TextureMesh scene_dense.mvs --mesh-file scene_dense_mesh.ply --export-type obj
  -> scene_dense_texture.obj (Blender, MeshLab, Godot, Unity, Unreal)

Gaussian splat with Brush (Vulkan GPU, e.g. Intel Xe2):
  open {dataset} in Brush (images/ + sparse/0/)
  -> export .ply, view in Brush or SuperSplat"""
