# AV1-sfm-reimplemented

An open reimplementation of

> J. Zouein, H. Javidnia, F. Pitié, A. Kokaram, *Leveraging AV1 motion vectors
> for Fast and Dense Feature Matching*, arXiv:2510.17434.

The authors did not release code. This implementation follows the paper and
the description of its matcher in the group's follow-up, *Efficient dense
matching for enhanced Gaussian splatting using AV1 motion vectors*
(arXiv:2605.14629); both PDFs are in the repository root. The motion vectors
(MVs) that an AV1 encoder already computes are turned into dense keypoints,
multi-frame tracks and COLMAP-compatible matches, so COLMAP's mapper can run
with no SIFT extraction or matching at all.

**Read [ASSUMPTIONS.md](ASSUMPTIONS.md).** Each item there is marked as
confirmed by the paper, changed to match it, open (the paper is silent, so it
is our choice, with evidence), or a deliberate deviation.

## Pipeline

```
images ──ffmpeg: libaom | SVT-AV1 | Vulkan Video | QSV | VA-API (low delay, 1 keyframe)──▶ clip.ivf
clip.ivf ──patched dav1d (in-memory block metadata)──▶ per-frame 4×4 MV grids
   ──collapse to coded blocks──▶ block-centre keypoints + MV targets
   ──propagate through the MV chain, cosine filter, min length 3──▶ tracks
   ──all pairs along each track (triangular adjacency)──▶ matches
   ──pycolmap──▶ database.db (keypoints, matches, two_view_geometries)
```

| Module | Role |
|---|---|
| `src/av1sfm/encode.py` | Image sequence → streaming AV1 IVF with libaom, SVT-AV1, Vulkan Video, Intel QSV or VA-API (see [Encoders](#encoders)). |
| `src/av1sfm/extract.py` | Wraps the vendored extractor; resolves order hints and reference slots to absolute frame indices. |
| `src/av1sfm/blocks.py` | Collapses the 4×4 grid to coded blocks; block-centre keypoints; MV → target point; per-point MV lookup. |
| `src/av1sfm/tracks.py` | Track propagation, cosine consistency filter (ε, τ), minimum length, all-pairs matches. |
| `src/av1sfm/colmap_db.py` | Writes keypoints, matches and two-view geometries into a COLMAP database. |
| `src/av1sfm/geometry.py` | Pairwise scoring: E (five-point) vs H (DLT) with LO-RANSAC, inlier ratio, Sampson error. |
| `src/av1sfm/validate.py` | Checks the MV convention by warping reference frames. |
| `src/av1sfm/_vendor/`, `third_party/av1of/` | Extraction layer vendored from [sigmedia/AV1-Optical-Flow](https://github.com/sigmedia/AV1-Optical-Flow) (AGPL-3.0). |

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) (Python 3.14 is fetched
automatically), meson, ninja, cmake, a C compiler, nasm, pkg-config. On
Ubuntu 24.04:

```bash
sudo apt-get install -y build-essential meson ninja-build cmake nasm pkg-config \
     libaom-dev libdav1d-dev libva-dev libdrm-dev libvulkan-dev
bash setup.sh                 # patched dav1d + shim into third_party/build, then `uv sync`
                              # (AV1SFM_SKIP_SYNC=1: native part only, for an active venv)
bash scripts/build_ffmpeg.sh  # FFmpeg n9.0.2 + SVT-AV1 v4.2.0 + Vulkan + QSV + VA-API (~15 min)
uv run av1sfm encoders        # which AV1 encoders work on this machine
uv run pytest                 # 61 tests; integration tests need ffmpeg and the dav1d build
```

`scripts/build_ffmpeg.sh` installs into `third_party/build/media`, and av1sfm
uses that ffmpeg automatically. A distribution FFmpeg ≥ 8 with the needed
encoders also works: set `AV1SFM_FFMPEG=/usr/bin/ffmpeg`.

`setup.sh` fetches dav1d commit `14c73c7d` from code.videolan.org and falls
back to the GitHub mirror, then applies the vendored inspection patch.
[pycolmap](https://pypi.org/project/pycolmap/) (4.2.1, CPU build) provides
COLMAP; no separate COLMAP binary is needed.

## Usage

```bash
# 1. image folder (frames in display order, sorted by name) -> MV matches in a COLMAP database
uv run av1sfm match path/to/images out/database.db --ivf out/clip.ivf --encode \
    --camera-params "f,cx,cy,k"      # optional; SIMPLE_RADIAL shared by all images

# 2. reconstruct (identical mapper settings for every method)
uv run python eval/run_mapper.py out/database.db path/to/images out/sparse

# pairwise geometric scoring of the raw matches in any COLMAP database
uv run av1sfm score out/database.db --out out/score.json

# check the MV sign convention on your clip
uv run av1sfm validate-warp out/clip.ivf path/to/images
```

Main `match` options (defaults in brackets):

| Option | Meaning |
|---|---|
| `--eps` [0.1] | cosine tolerance: require cos(v_nm, v_ml) ≥ 1 − ε; `--eps 1` disables the filter (use for image collections with large frame gaps). |
| `--tau` [2.0] | skip the cosine test when either MV is shorter than τ px. |
| `--min-length` [3] | minimum track length in frames. |
| `--on-violation` [cut] | on a cosine violation, `cut` (terminate) the track as in the paper, `split` it into two tracks, or `drop` it. |
| `--seed` [all] | `all`: every block of every frame starts a track (paper); `uncovered`: only blocks no arriving track lands in. |
| `--grid` [block] | keypoints per coded block, or per 4×4 unit of the zero-order-hold motion field (`cell`; very dense, see ASSUMPTIONS.md B1). |
| `--prev-only` | use only MVs whose reference is the previous frame. |
| `--max-pair-gap` | only emit matches between frames at most this far apart. |
| `--two-view` [verify] | `verify`: COLMAP's geometric verification with the shared RANSAC settings; `trust`: all MV matches are stored as inliers. |
| `--encoder` [libaom] | `libaom`, `svtav1`, `vulkan`, `qsv`, `vaapi`, or `auto` (see below). |
| `--crf` [32], `--qp` [128] | CRF for libaom/SVT-AV1; constant AV1 qindex (0–255) for Vulkan/QSV/VA-API. |
| `--usage`, `--cpu-used` [realtime, 6] | libaom settings. |
| `--svt-preset` [10], `--svt-params` | SVT-AV1 preset and extra `key=value:...` parameters. |
| `--hw-device` | Vulkan device index, or QSV / VA-API DRM render node (e.g. `/dev/dri/renderD128`). |

## Encoders

All backends produce a low-delay stream: one keyframe, past references only,
no B-frames or look-ahead. Each MV is attached to its actual reference frame,
so every backend's reference structure is handled.

| `--encoder` | FFmpeg encoder | Configuration | Status |
|---|---|---|---|
| `libaom` (default) | `libaom-av1` | `-usage realtime -cpu-used 6 -lag-in-frames 0 -crf 32` | The paper's encoder. Tested. 97 % of MVs reference the previous frame; quarter-pel. |
| `svtav1` | `libsvtav1` (SVT-AV1 4.2) | `-preset 10 -crf 32 -svtav1-params pred-struct=1:rtc=1:keyint=-1` | Tested. About 4× faster to encode than libaom on KITTI, but layered references (51 % to n−1), 19 % compound blocks and noisier MVs (warp error 8.9 vs 5.1). |
| `vulkan` | `av1_vulkan` (Vulkan Video) | hwupload to a Vulkan device, `-rc_mode cqp -qp 128 -bf 0 -tune ll -usage stream` | Needs `VK_KHR_video_encode_av1`: Mesa RADV (AMD RDNA3+) or ANV (Intel Arc / Xe2+). **Not tested on hardware** (development container has only lavapipe). |
| `qsv` | `av1_qsv` (oneVPL) | VA-API device, hwupload, `-preset veryfast -q:v 128 -bf 0 -look_ahead_depth 0` | Needs Intel Arc / Meteor Lake or newer with the VPL GPU runtime. **Not tested on hardware.** |
| `vaapi` | `av1_vaapi` (VA-API) | hwupload to a VA-API device, `-rc_mode CQP -global_quality 128 -bf 0` | Needs a VA-API driver with AV1 encode (Intel media driver on Arc / Meteor Lake / Lunar Lake, Mesa on AMD RDNA3+). The most direct Intel path on Linux. **Not tested on hardware.** |
| `auto` | | first working of `vulkan`, `qsv`, `vaapi`, `svtav1` | |

`uv run av1sfm encoders` lists the backends compiled into the ffmpeg in use
and runs a 3-frame test encode with each, printing the specific failure (for
example, a driver without `VK_KHR_video_encode_queue`). Hardware MV search is
usually coarser than software, so check the MVs on a new encoder with
`uv run python eval/mv_stats.py clip.ivf` and `uv run av1sfm validate-warp clip.ivf images/`.

### Intel Lunar Lake / Arc on Ubuntu 26.04

Install the GPU user-space stack from Intel's graphics PPA (package names as
in that PPA; check `apt search` if they differ):

```bash
sudo add-apt-repository ppa:kobuk-team/intel-graphics
sudo apt-get install -y intel-media-va-driver-non-free libmfx-gen1 libvpl2 \
     vainfo mesa-vulkan-drivers vulkan-tools
vainfo | grep -i av1           # want VAProfileAV1Profile0 : VAEntrypointEncSliceLP
vulkaninfo | grep -i video_encode_av1   # Vulkan Video path (Mesa ANV)
bash scripts/build_ffmpeg.sh && uv run av1sfm encoders
```

Then check the hardware MVs before running the evaluation, for example on KITTI:

```bash
uv run av1sfm encode runs/kitti117/img runs/kitti117/clip_vaapi.ivf --encoder vaapi
uv run python eval/mv_stats.py runs/kitti117/clip_vaapi.ivf     # MV precision, refs, block sizes
uv run av1sfm validate-warp runs/kitti117/clip_vaapi.ivf runs/kitti117/img
```

Compare against the libaom numbers in ASSUMPTIONS.md (E4: 97.7 % of MVs to the
previous frame, quarter-pel). If most MVs reference older frames, try
`--prev-only`. The quality setting (`--qp`) may also need re-tuning (E5d).

## Reproducing the evaluation

```bash
bash eval/run_kitti.sh              # fetches KITTI 00 frames 0-229, runs everything, writes runs/results.md
bash eval/run_colmap_datasets.sh    # fetches Gerrard Hall and Person Hall, pairwise metrics (eps = 1)
SFM=1 bash eval/run_colmap_datasets.sh   # ... plus incremental mapping
```

Hardware encoders are added as extra methods (`mv_qsv`, `mv_vaapi`,
`mv_vulkan`); `ONLY` restricts the run to some methods and `RUN=` drops the
`uv run` prefix inside an active virtualenv:

```bash
RUN= HW="qsv vaapi" ONLY="mv mv_qsv mv_vaapi" SETS=117 bash eval/run_kitti.sh
```

Gerrard Hall and Person Hall come from the COLMAP release assets
(`https://github.com/colmap/colmap/releases/download/3.11.1/gerrard-hall.zip`,
`person-hall.zip` plus `person-hall.z01`). As in the paper, a subset of
same-size, temporally adjacent images is used (Gerrard Hall: all 100; Person
Hall: IMG_1015–IMG_1229, 215 images), resized to 1920×1280, with the cosine
filter disabled (ε = 1). See ASSUMPTIONS.md D2. On these datasets libaom
also runs in its default `good` usage (`mv_good`, ASSUMPTIONS.md E2b).

`eval/run_kitti.sh` holds every command behind the table below. Steps whose
output already exists are skipped. For each set (117 and 230 frames) it runs:
the MV pipeline on libaom (with COLMAP verification and with trusted MVs) and
on SVT-AV1, COLMAP SIFT
with sequential (overlap 10) and exhaustive matching, identical pairwise
scoring of every method's raw matches, and, for the 117-frame set, incremental
mapping with identical settings.

## Viewing and using a reconstruction

The mapper writes a sparse COLMAP model (camera poses and a coloured point
cloud) to `runs/<clip>/rec_<method>/0`. Open it directly in the COLMAP GUI
(File → Import model), or export it:

```bash
uv run python eval/export_model.py runs/kitti117/rec_mv runs/kitti117/img runs/kitti117/export_mv
```

This writes `points.ply` (MeshLab, CloudCompare, Blender) and `dataset/`, an
undistorted COLMAP workspace with a PINHOLE camera: `images/`, `sparse/`
(binary and text) and `sparse/0/`. Both follow-on paths below run without an
NVIDIA GPU; neither is tested here.

- **Textured mesh with [OpenMVS](https://github.com/cdcseacave/openMVS)** (CPU):
  `InterfaceCOLMAP -i dataset -o scene.mvs --image-folder dataset/images`, then
  `DensifyPointCloud`, `ReconstructMesh` and `TextureMesh` (the script prints the
  exact commands). The OBJ opens in Blender, MeshLab, Godot, Unity or Unreal.
  COLMAP's own dense stereo needs CUDA.
- **Gaussian splat with [Brush](https://github.com/ArthurBrussee/brush)**
  (Vulkan / WebGPU, so Intel Xe2 should work): open `dataset/`. View the
  result in Brush or SuperSplat.

A building filmed around (Gerrard Hall, Person Hall, with `SFM=1`) makes a
better model to move around in than KITTI's forward drive.

## Results

Produced by `bash eval/run_kitti.sh` and `bash eval/run_colmap_datasets.sh`
on a 4-core x86 cloud VM **without a GPU** (SIFT extraction and matching on
the CPU), with the paper-faithful track defaults (every block seeds a track,
tracks are cut at the first cosine violation, ε = 0.1 on KITTI and ε = 1 on
the COLMAP datasets). Software: FFmpeg n9.0.2, libaom 3.8.2, SVT-AV1 4.2.0,
pycolmap 4.2.1, Python 3.14.7. The same tables are in
[`results/results.md`](results/results.md); per-run JSON summaries are in
[`results/`](results/).

How to read the tables:

- **Pre-processing**: AV1 encoding for MVs, SIFT extraction for SIFT.
  **Feature matching**: for MVs, decoding, track building, database writing
  and COLMAP geometric verification; for SIFT, matching and verification.
  **CPU %**: 100 = one core; **CPU % of machine**: share of all cores
  (the paper's convention).
- **Inlier ratio** and **Sampson error** come from identical E/H LO-RANSAC
  scoring of each method's *raw* matches (max_error 4 px, min_inlier_ratio
  0.25, 10 000 trials, median of 3 runs per pair). *Pooled* is inliers over
  matches across all pairs; *median of pairs* is the statistic in the paper's
  Table IV. **Median SE** is the paper's squared Sampson error in normalised
  coordinates (Table III).
- **SfM** uses identical mapper settings for all methods, with KITTI's
  forward-motion initialisation settings (ASSUMPTIONS.md R5). The first table
  fixes the shared SIMPLE_RADIAL camera at the KITTI calibration; the second
  uses COLMAP's default intrinsics refinement, under which some MV databases
  collapse to two images (R5).
- The trusted-MV variant has no scoring row: its raw matches are the verified
  libaom row's.

### KITTI 00, frames 0–116

| Method | Pre-processing (s) | Feature matching (s) | CPU % (1 core = 100) | CPU % of machine | Keypoints / img | Raw matches / img | Verified matches / img | Scored pairs | Inlier ratio (pooled) | Inlier ratio (median of pairs) | Median Sampson (px) | Median SE (normalised²) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AV1 MV, libaom (COLMAP verification) | 10.5 | 189.3 | 362 | 90.5 | 20,757 | 351,240 | 343,851 | 5,171 | 0.970 | 0.978 | 0.611 | 7.23e-07 |
| AV1 MV, libaom (MVs trusted, no verification) | – | 19.9 | 102 | 25.6 | 20,757 | 351,240 | 351,240 | – | – | – | – | – |
| AV1 MV, SVT-AV1 (COLMAP verification) | 2.0 | 6.4 | 274 | 68.5 | 1,824 | 6,455 | 6,190 | 983 | 0.949 | 0.945 | 0.562 | 6.11e-07 |
| SIFT sequential (overlap 10) | 24.7 | 33.8 | 345 | 86.2 | 5,117 | 17,218 | 16,908 | 1,115 | 0.976 | 0.971 | 0.160 | 4.96e-08 |
| SIFT exhaustive | 24.5 | 193.6 | 395 | 98.6 | 5,117 | 23,932 | 21,664 | 6,713 | 0.905 | 0.540 | 0.247 | 1.19e-07 |

SfM, intrinsics fixed at calibration:

| Method | Registered | 3D points | Reproj. error (px) | Mean track length | Mapper wall (s) | Final global BA (s) |
|---|---:|---:|---:|---:|---:|---:|
| AV1 MV, libaom (COLMAP verification) | 117/117 | 201,697 | 0.893 | 10.49 | 1,508.5 | 49.9 |
| AV1 MV, libaom (MVs trusted, no verification) | 117/117 | 203,661 | 0.895 | 10.41 | 1,514.1 | 47.5 |
| AV1 MV, SVT-AV1 (COLMAP verification) | 116/117 | 35,737 | 0.792 | 3.71 | 62.6 | 1.4 |
| SIFT sequential (overlap 10) | 117/117 | 35,055 | 0.379 | 7.38 | 74.7 | 3.7 |
| SIFT exhaustive | 117/117 | 36,812 | 0.391 | 7.47 | 117.2 | 6.1 |

SfM, COLMAP default intrinsics refinement:

| Method | Registered | 3D points | Reproj. error (px) | Mean track length | Mapper wall (s) | Final global BA (s) |
|---|---:|---:|---:|---:|---:|---:|
| AV1 MV, libaom (COLMAP verification) | 2/117 | 29,414 | 0.349 | 2.00 | 19.5 | 6.3 |
| AV1 MV, libaom (MVs trusted, no verification) | 117/117 | 204,765 | 0.893 | 10.40 | 1,691.4 | 63.0 |
| AV1 MV, SVT-AV1 (COLMAP verification) | 116/117 | 36,179 | 0.789 | 3.72 | 80.5 | 4.4 |
| SIFT sequential (overlap 10) | 117/117 | 35,034 | 0.374 | 7.39 | 88.8 | 4.8 |
| SIFT exhaustive | 117/117 | 36,799 | 0.385 | 7.46 | 132.2 | 7.5 |

### KITTI 00, frames 0–229

| Method | Pre-processing (s) | Feature matching (s) | CPU % (1 core = 100) | CPU % of machine | Keypoints / img | Raw matches / img | Verified matches / img | Scored pairs | Inlier ratio (pooled) | Inlier ratio (median of pairs) | Median Sampson (px) | Median SE (normalised²) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AV1 MV, libaom (COLMAP verification) | 17.2 | 482.5 | 371 | 92.8 | 21,710 | 433,433 | 422,965 | 11,306 | 0.971 | 0.976 | 0.599 | 6.95e-07 |
| AV1 MV, libaom (MVs trusted, no verification) | – | 39.5 | 103 | 25.6 | 21,710 | 433,433 | 433,433 | – | – | – | – | – |
| AV1 MV, SVT-AV1 (COLMAP verification) | 3.8 | 13.4 | 278 | 69.4 | 1,783 | 6,923 | 6,654 | 2,057 | 0.952 | 0.949 | 0.543 | 5.72e-07 |
| SIFT sequential (overlap 10) | 46.8 | 62.6 | 347 | 86.6 | 4,831 | 15,380 | 15,085 | 2,245 | 0.977 | 0.972 | 0.154 | 4.60e-08 |
| SIFT exhaustive | 48.9 | 825.6 | 396 | 99.1 | 4,831 | 23,966 | 20,030 | 23,712 | 0.868 | 0.444 | 0.248 | 1.21e-07 |

### Gerrard Hall (100 images, 1920×1280, ε = 1)

| Method | Pre-processing (s) | Feature matching (s) | CPU % (1 core = 100) | CPU % of machine | Keypoints / img | Raw matches / img | Verified matches / img | Scored pairs | Inlier ratio (pooled) | Inlier ratio (median of pairs) | Median Sampson (px) | Median SE (normalised²) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AV1 MV, libaom (COLMAP verification) | 12.9 | 122.3 | 363 | 90.8 | 10,669 | 53,407 | 37,415 | 1,264 | 0.684 | 0.414 | 0.994 | 5.76e-07 |
| AV1 MV, libaom (MVs trusted, no verification) | – | 10.2 | 117 | 29.3 | 10,669 | 53,407 | 53,407 | – | – | – | – | – |
| AV1 MV, libaom good usage (COLMAP verification) | 83.0 | 40.3 | 306 | 76.4 | 11,078 | 62,083 | 56,782 | 1,053 | 0.910 | 0.728 | 0.486 | 1.37e-07 |
| AV1 MV, SVT-AV1 (COLMAP verification) | 6.4 | 27.2 | 302 | 75.4 | 1,155 | 2,689 | 1,367 | 512 | 0.480 | 0.442 | 0.846 | 4.15e-07 |
| SIFT sequential (overlap 10) | 108.3 | 78.6 | 350 | 87.5 | 12,862 | 23,498 | 23,015 | 902 | 0.970 | 0.952 | 0.314 | 5.71e-08 |
| SIFT exhaustive | 108.6 | 309.3 | 395 | 98.7 | 12,862 | 26,476 | 24,886 | 3,657 | 0.943 | 0.421 | 0.284 | 4.74e-08 |

### Person Hall (IMG_1015–1229, 215 images, 1920×1280, ε = 1)

| Method | Pre-processing (s) | Feature matching (s) | CPU % (1 core = 100) | CPU % of machine | Keypoints / img | Raw matches / img | Verified matches / img | Scored pairs | Inlier ratio (pooled) | Inlier ratio (median of pairs) | Median Sampson (px) | Median SE (normalised²) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AV1 MV, libaom (COLMAP verification) | 32.6 | 359.2 | 367 | 91.6 | 12,232 | 53,562 | 37,448 | 3,794 | 0.674 | 0.400 | 0.966 | 5.43e-07 |
| AV1 MV, libaom (MVs trusted, no verification) | – | 23.7 | 116 | 29.0 | 12,232 | 53,562 | 53,562 | – | – | – | – | – |
| AV1 MV, libaom good usage (COLMAP verification) | 193.5 | 150.0 | 312 | 78.1 | 15,957 | 117,766 | 111,640 | 3,193 | 0.941 | 0.722 | 0.457 | 1.21e-07 |
| AV1 MV, SVT-AV1 (COLMAP verification) | 14.9 | 114.1 | 338 | 84.5 | 1,989 | 5,526 | 2,391 | 1,409 | 0.407 | 0.367 | 0.910 | 4.82e-07 |
| SIFT sequential (overlap 10) | 278.5 | 205.9 | 355 | 88.7 | 14,076 | 49,278 | 48,763 | 2,014 | 0.981 | 0.977 | 0.344 | 6.87e-08 |
| SIFT exhaustive | 284.2 | 1,473.5 | 387 | 96.8 | 14,076 | 72,356 | 70,400 | 10,045 | 0.971 | 0.467 | 0.290 | 4.92e-08 |

### Intel Lunar Lake: hardware AV1 encoders (KITTI 00, frames 0–116)

Run by the repository owner on an Intel Lunar Lake laptop (Ubuntu 26.04,
Intel graphics PPA, system FFmpeg, power profile "Balanced", so not peak
performance) with
`RUN= HW="qsv vaapi" ONLY="mv mv_qsv mv_vaapi" SETS=117 bash eval/run_kitti.sh`, then
`RUN= ONLY="sift_seq sift_exh" SETS=117 bash eval/run_kitti.sh`. The MV rows were scored
before the median-of-pairs statistic was added, hence the dashes.
Vulkan Video AV1 encode was not exposed by the driver (ASSUMPTIONS.md E5b).
Source: [`results/lunar-lake/kitti117.md`](results/lunar-lake/kitti117.md).

| Method | Pre-processing (s) | Feature matching (s) | CPU % (1 core = 100) | CPU % of machine | Keypoints / img | Raw matches / img | Verified matches / img | Scored pairs | Inlier ratio (pooled) | Inlier ratio (median of pairs) | Median Sampson (px) | Median SE (normalised²) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AV1 MV, libaom (COLMAP verification) | 4.5 | 82.1 | 721 | 90.2 | 20,757 | 351,240 | 343,857 | 5,171 | 0.970 | – | 0.611 | 7.23e-07 |
| AV1 MV, Intel QSV (COLMAP verification) | 0.6 | 39.5 | 726 | 90.8 | 10,261 | 157,105 | 154,600 | 4,857 | 0.977 | – | 0.472 | 4.31e-07 |
| AV1 MV, VA-API (COLMAP verification) | 0.5 | 33.2 | 718 | 89.8 | 10,839 | 129,221 | 128,076 | 4,415 | 0.988 | – | 0.308 | 1.83e-07 |
| SIFT sequential (overlap 10) | 7.7 | 15.0 | 570 | 71.3 | 5,117 | 15,961 | 15,664 | 1,115 | 0.975 | 0.971 | 0.162 | 5.06e-08 |
| SIFT exhaustive | 7.8 | 90.0 | 755 | 94.3 | 5,117 | 23,901 | 21,653 | 6,675 | 0.906 | 0.544 | 0.245 | 1.18e-07 |

SfM, intrinsics fixed at calibration:

| Method | Registered | 3D points | Reproj. error (px) | Mean track length | Mapper wall (s) | Final global BA (s) |
|---|---:|---:|---:|---:|---:|---:|
| AV1 MV, libaom (COLMAP verification) | 117/117 | 201,749 | 0.893 | 10.49 | 759.6 | 22.0 |
| AV1 MV, Intel QSV (COLMAP verification) | 117/117 | 99,962 | 0.807 | 9.97 | 252.7 | 10.1 |
| AV1 MV, VA-API (COLMAP verification) | 117/117 | 114,328 | 0.598 | 8.75 | 188.5 | 7.4 |
| SIFT sequential (overlap 10) | 117/117 | 33,796 | 0.378 | 7.44 | 34.0 | 1.6 |
| SIFT exhaustive | 117/117 | 36,708 | 0.390 | 7.48 | 49.9 | 2.1 |

SfM, COLMAP default intrinsics refinement:

| Method | Registered | 3D points | Reproj. error (px) | Mean track length | Mapper wall (s) | Final global BA (s) |
|---|---:|---:|---:|---:|---:|---:|
| AV1 MV, libaom (COLMAP verification) | 2/117 | 29,395 | 0.352 | 2.00 | 12.1 | 5.6 |
| AV1 MV, Intel QSV (COLMAP verification) | 2/117 | 9,667 | 0.221 | 2.00 | 1.9 | 0.6 |
| AV1 MV, VA-API (COLMAP verification) | 117/117 | 115,251 | 0.594 | 8.72 | 257.2 | 21.7 |
| SIFT sequential (overlap 10) | 117/117 | 33,804 | 0.372 | 7.44 | 44.9 | 4.1 |
| SIFT exhaustive | 117/117 | 36,692 | 0.384 | 7.48 | 69.7 | 4.5 |

### Findings

- **Reproducibility.** libaom MV matching is deterministic across machines:
  the Lunar Lake run reproduces the cloud run's 20,757 keypoints,
  351,240 raw matches, 0.970 inlier ratio and 0.611 px Sampson error.
- **Density (KITTI).** libaom MVs give 4× more keypoints per image than SIFT
  (20.8 k vs 5.1 k) and 15–20× more raw matches. Tracks average 11 frames and
  reach 92. The 117-frame reconstruction has 202 k points against SIFT's
  35–37 k: the paper's qualitative result.
- **Precision (KITTI).** The inlier ratio matches SIFT sequential (0.97;
  median of pairs 0.98), but the median Sampson error is 0.61 px against
  0.16 px (sequential) and 0.25 px (exhaustive), and the reconstruction's
  reprojection error is 0.89 px against 0.38 px. The paper reports
  460–616 k points at 0.51–0.53 px on its own 1080×1920 clip (4.4× more
  pixels than KITTI).
- **Speed and CPU.** Encoding (10.5 s) is cheaper than SIFT extraction
  (24.7 s). With COLMAP verification, MV matching is the most expensive
  stage (189 s for 20.5 M matches), and CPU use is no lower than SIFT's. Without
  verification (trusted MVs), matching takes 20 s on one core (26 % of the
  machine) and the reconstruction is the same (117/117, 204 k points,
  0.90 px). That is the configuration in which MV matching is cheap. The
  mapper is much slower on MV databases (1,509 s vs 75–117 s), because bundle
  adjustment scales with the 2.1 M observations.
- **Hardware encoders (Lunar Lake).** QSV and VA-API encode 117 frames in
  0.5–0.6 s, 7–9× faster than libaom on the same machine. They give half the
  keypoints (larger blocks) but cleaner matches: VA-API reaches 0.988 inlier
  ratio, 0.31 px Sampson error and a 0.60 px reconstruction, against libaom's
  0.970, 0.61 px and 0.89 px. With COLMAP's default intrinsics refinement,
  the VA-API database still registers 117/117, while the libaom and QSV
  databases collapse to two images (as does libaom's in the cloud run; the
  trusted-MV and SVT-AV1 databases survive there).
- **End to end on Lunar Lake.** Up to the mapper, VA-API takes 34 s (0.5 s
  encode + 33 s matching with COLMAP verification), SIFT sequential 23 s
  (7.7 + 15 s) and SIFT exhaustive 98 s (7.8 + 90 s). VA-API's encode is
  15× cheaper than SIFT extraction, but verifying its 129 k raw matches per
  image dominates. With the mapper, VA-API takes 222 s for 114 k points at
  0.60 px; SIFT sequential 57 s for 34 k points at 0.38 px. MV matching buys
  density, not end-to-end speed, unless verification is skipped (trusted MVs).
- **SVT-AV1** encodes 5× faster than libaom, but it codes fewer, larger
  blocks (129 k track seeds vs 261 k) and its tracks are shorter (mean 4.0
  vs 11.3 frames; only about half its MVs point to the previous frame,
  ASSUMPTIONS.md E5a). That leaves 1.8 k keypoints per image and a 36 k-point
  reconstruction (116/117).
- **Gerrard Hall and Person Hall.** These are photo collections, not video:
  only 37–41 % of blocks are inter-coded. With libaom's realtime usage, MV
  matches are much worse than SIFT (pooled inlier ratio 0.68 and 0.67, median
  Sampson error about 1 px). libaom's default `good` usage, with a wider
  motion search, fixes most of this: 0.91 and 0.94 pooled (0.94 and 0.97
  between adjacent images, close to the paper's 0.96), and half the Sampson
  error (0.49 and 0.46 px), at 6× the encoding time. SIFT sequential remains
  more precise (0.97–0.98, 0.31–0.34 px). The paper's Table IV values for NVENC on
  these datasets (0.47, 0.43) are close to ours with realtime usage.
  ASSUMPTIONS.md E2b.
- **SIFT exhaustive** degrades with sequence length: its pooled inlier ratio
  drops to 0.87 at 230 frames, and the median over pairs is 0.42–0.54 on all
  datasets, because most of its pairs are far apart and share few true
  matches. MV matching scales linearly (482 s for 230 frames with verification,
  40 s without; SIFT exhaustive 826 s).


## Tests

`tests/` uses synthetic data for:

- the cosine filter (`test_cosine.py`): thresholds, τ, ε = 1;
- MV-to-correspondence geometry (`test_blocks.py`): block collapse, centres, references, compound blocks;
- track building (`test_tracks.py`): propagation, seeding modes, cut/split/drop, 4×4 grid, gaps, triangular matches, match-count guard;
- the COLMAP writer (`test_colmap_db.py`) and the Sampson formulas (`test_geometry.py`);
- an end-to-end check on a real AV1 encode of a synthetic pan + zoom with known ground truth (`test_integration.py`).

## Licence

AGPL-3.0-or-later (see [LICENSE](LICENSE)). The extraction layer is vendored
from sigmedia/AV1-Optical-Flow (AGPL-3.0, © 2026 Sigmedia.tv / Julien Zouein);
see [third_party/av1of/NOTICE.md](third_party/av1of/NOTICE.md) for the files,
the upstream commit and the one modification.

## Citation

```bibtex
@article{zouein2025av1matching,
  title   = {Leveraging AV1 motion vectors for Fast and Dense Feature Matching},
  author  = {Zouein, Julien and Javidnia, Hossein and Piti{\'e}, Fran{\c{c}}ois and Kokaram, Anil},
  journal = {arXiv preprint arXiv:2510.17434},
  year    = {2025}
}
```
