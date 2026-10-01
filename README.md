# AV1-sfm-reimplemented

**av1sfm** builds 3D reconstructions from video by reusing the motion
information that a video encoder computes anyway, instead of searching every
image for features. It is an open implementation of the method in

> J. Zouein, H. Javidnia, F. Pitié, A. Kokaram, *Leveraging AV1 motion vectors
> for Fast and Dense Feature Matching*, arXiv:2510.17434 (2025),

whose authors did not release code. Its output plugs into
[COLMAP](https://colmap.github.io/), the standard open-source
structure-from-motion software.

## The idea

*Structure from motion* (SfM) recovers where each camera stood and a 3D point
cloud of the scene from a set of overlapping images. Its first step, and often
its most expensive one, is finding **correspondences**: the same physical point
seen in two or more images. Conventional pipelines detect distinctive points
(for example SIFT keypoints) in every image, describe each one by its
surrounding texture, and compare those descriptions between pairs of images.

Video compression solves a closely related problem. To store a frame cheaply,
an AV1 encoder divides it into blocks and, for each block, searches earlier
frames for the patch that predicts it best. The offset to that patch is the
block's **motion vector**. A compressed video therefore already contains a
dense set of frame-to-frame correspondences, computed by heavily optimised
encoder software or by the dedicated video hardware found in most recent GPUs.

av1sfm turns those motion vectors into input for COLMAP:

1. **Encode.** The image sequence is compressed as AV1 video with settings that
   keep the motion simple to interpret: a single keyframe, and every frame
   predicted only from earlier frames.
2. **Read the motion.** An instrumented AV1 decoder (dav1d, with the
   inspection layer of [sigmedia/AV1-Optical-Flow](https://github.com/sigmedia/AV1-Optical-Flow))
   reports every block's motion vector and the frame it refers to.
3. **Make keypoints.** Each block centre becomes a keypoint, and its motion
   vector gives the position of the same point in the referenced frame.
4. **Build tracks.** These frame-to-frame links are chained into tracks that
   follow a point through many frames. A track is cut where consecutive motion
   vectors disagree in direction, a cheap test for unreliable motion.
5. **Hand over to COLMAP.** Keypoints and every pair of observations along each
   track are written to a COLMAP database. COLMAP checks them against the
   geometry of two-view camera motion and reconstructs the scene.

No image features are detected or compared at any point. The result is a
COLMAP reconstruction (camera poses and a coloured point cloud) that can be
viewed in COLMAP, exported as a point cloud, or used as the starting point for
a textured mesh or a Gaussian splat.

## What is in the repository

- **The matcher** (`av1sfm match`): an image folder in, a COLMAP database out.
  It works with five AV1 encoders: libaom and SVT-AV1 in software; Intel Quick
  Sync (QSV) and VA-API on Intel GPUs; and Vulkan Video.
- **Baselines** run with the same pair selection, verification and mapper
  settings: COLMAP's SIFT with sequential or exhaustive matching; an exact
  SIFT matcher that runs on a GPU and reproduces COLMAP's slow exact CPU
  matcher; and DISK + LightGlue, a learned detector and matcher.
- **An evaluation harness** that downloads the datasets (KITTI odometry,
  and COLMAP's Gerrard Hall and Person Hall), runs every method, scores
  every method's raw matches with the same geometric test, reconstructs, and
  writes the result tables.
- **Export** of a reconstruction to a point cloud (PLY) and to an undistorted
  COLMAP workspace for meshing (OpenMVS) or Gaussian splatting (Brush), which
  run without an NVIDIA GPU.

## Results in brief

KITTI odometry sequence 00, frames 0–116 (a car driving through a
residential area, 1241×376 pixels), on an Intel Lunar Lake laptop. Every method registers all
117 images.

| Method | Images → matches | 3D points | Reprojection error |
|---|---:|---:|---:|
| AV1 motion vectors, Intel GPU encoder (VA-API) | 34 s | 114,000 | 0.60 px |
| AV1 motion vectors, libaom software encoder | 87 s | 202,000 | 0.89 px |
| SIFT, sequential matching | 22 s | 34,000 | 0.38 px |
| SIFT, exhaustive matching | 95 s | 37,000 | 0.39 px |
| SIFT, exact matching on the GPU | 27 s | 35,000 | 0.38 px |
| DISK + LightGlue on the GPU | 1,055 s* | 44,000 | 0.89 px |

\* Measured before the current LightGlue speed-ups, which were 5.6× faster
per image pair in a test; the full run is being repeated.

- **Density.** Motion vectors give 3–6× more 3D points than SIFT, the
  paper's main qualitative result. Each 3D point is also seen in more images
  (8.8–10.5 on average, against 7.4 with SIFT).
- **Precision.** Individual motion-vector matches are less precise than SIFT
  matches, and the reconstructions have a higher reprojection error. The
  Intel GPU encoder's motion vectors are the most precise of the encoders
  tested.
- **Speed.** Encoding is very cheap (0.5 s for 117 frames on the Intel GPU,
  against 7.5 s for SIFT feature extraction), but checking hundreds of
  thousands of matches geometrically, and later bundle adjustment over many
  more points, cost more than SIFT's whole pipeline. Trusting the motion
  vectors without the geometric check cuts matching from 189 s to 20 s with
  the same reconstruction (cloud measurement). Motion vectors buy density
  first; speed only where verification can be skipped.
- **Video versus photo collections.** On Gerrard Hall and Person Hall, photo
  collections with large jumps between shots, motion-vector matches are much
  less reliable (67–68 % inliers against SIFT's 97–98 %) unless libaom's slower
  `good` mode is used (91–94 %).

The paper's own 1080p test clips are not public, so the numbers are not
directly comparable to its tables. [docs/RESULTS.md](docs/RESULTS.md) has every
table and finding, and [ASSUMPTIONS.md](ASSUMPTIONS.md) lists each point where
the paper is silent, where it was followed, and where this implementation
deliberately differs.

## Platform support

| Component | Status |
|---|---|
| Linux on x86-64 | Tested: Ubuntu 24.04 (cloud, CPU only) and Ubuntu 26.04 (Intel Lunar Lake laptop). |
| libaom, SVT-AV1 (software encoders) | Tested. libaom is the paper's encoder and the default. |
| Intel QSV and VA-API (GPU encoders) | Tested on Intel Lunar Lake. |
| Vulkan Video | Implemented; the Lunar Lake driver tested does not offer AV1 encoding, and AMD GPUs are untested. |
| NVIDIA NVENC (the paper's second encoder) | Not supported. |
| GPU baselines on Intel GPUs (PyTorch XPU) | Tested, with workarounds for PyTorch bugs ([docs/DEPENDENCY_ISSUES.md](docs/DEPENDENCY_ISSUES.md)). |
| GPU baselines on NVIDIA GPUs (CUDA) | Expected to work; untested. |

## Getting started

Requires Linux, [uv](https://docs.astral.sh/uv/), a C toolchain (meson,
ninja, nasm) and FFmpeg with an AV1 encoder.

```bash
bash setup.sh                                  # builds the instrumented decoder, installs the Python package
uv run av1sfm encoders                         # which AV1 encoders work on this machine
uv run av1sfm match images/ out/database.db --ivf out/clip.ivf --encode
uv run python eval/run_mapper.py out/database.db images/ out/sparse
```

[docs/USAGE.md](docs/USAGE.md) covers installation (including Intel GPUs),
every command and option, the encoders, the GPU baselines and exporting a
reconstruction.

## Documentation

| Document | Contents |
|---|---|
| [docs/USAGE.md](docs/USAGE.md) | Installation, commands, options, encoders, GPU baselines, export, code layout, tests. |
| [docs/RESULTS.md](docs/RESULTS.md) | Reproducing the evaluation; all result tables and findings. |
| [ASSUMPTIONS.md](ASSUMPTIONS.md) | Every implementation decision, checked against the paper, with measurements. |
| [docs/DEPENDENCY_ISSUES.md](docs/DEPENDENCY_ISSUES.md) | Bugs and pitfalls found in PyTorch (Intel GPU), kornia, COLMAP and FFmpeg, and how av1sfm works around them. |

## Licence

AGPL-3.0-or-later (see [LICENSE](LICENSE)). The motion-vector extraction layer
is vendored from sigmedia/AV1-Optical-Flow (AGPL-3.0, © 2026 Sigmedia.tv /
Julien Zouein); see [third_party/av1of/NOTICE.md](third_party/av1of/NOTICE.md)
for the files, the upstream commit and the one modification.

## Citation

```bibtex
@article{zouein2025av1matching,
  title   = {Leveraging AV1 motion vectors for Fast and Dense Feature Matching},
  author  = {Zouein, Julien and Javidnia, Hossein and Piti{\'e}, Fran{\c{c}}ois and Kokaram, Anil},
  journal = {arXiv preprint arXiv:2510.17434},
  year    = {2025}
}
```
