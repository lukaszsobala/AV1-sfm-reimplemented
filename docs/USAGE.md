# Using av1sfm

Installation, commands and options. For what the software does, see the
[README](../README.md); for the evaluation and all measured results, see
[RESULTS.md](RESULTS.md); for every design decision, see
[ASSUMPTIONS.md](../ASSUMPTIONS.md).

## How the pipeline works

```
images ──ffmpeg: libaom | SVT-AV1 | Vulkan Video | QSV | VA-API (low delay, 1 keyframe)──▶ clip.ivf
AV1 video ──ffmpeg -c:v copy (no re-encoding)──▶ clip.ivf, and its frames ──▶ images
clip.ivf ──patched dav1d (in-memory block metadata)──▶ per-frame 4×4 MV grids
   ──collapse to coded blocks──▶ block-centre keypoints + MV targets
   ──propagate through the MV chain, cosine filter, min length 3──▶ tracks
   ──all pairs along each track (triangular adjacency)──▶ matches
   ──pycolmap──▶ database.db (keypoints, matches, two_view_geometries)
   ──COLMAP incremental mapper──▶ sparse/0 ──▶ points.ply
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
| `src/av1sfm/pipeline.py`, `cli.py` | The `av1sfm` command (`reconstruct`, `match`, `encode`, `encoders`, `score`, `validate-warp`) and its run statistics. |
| `src/av1sfm/reconstruct.py`, `video.py` | `av1sfm reconstruct`: images or a video → matches → reconstruction → PLY; probing videos, copying AV1 streams to IVF, extracting frames. |
| `src/av1sfm/mapping.py`, `export.py` | Mapper settings shared by all methods; PLY and undistorted-workspace export. |
| `src/av1sfm/sift.py`, `sift_exact.py` | SIFT extraction and matching with the shared settings; exact matching on a PyTorch device, identical to COLMAP's brute-force matcher. |
| `src/av1sfm/learned.py`, `devices.py` | DISK + LightGlue baseline (kornia) and PyTorch device selection. |
| `eval/` | Evaluation: dataset download, SIFT / DISK baselines, mapper, pose accuracy against KITTI ground truth, tables, export, GPU checks. |
| `src/av1sfm/_vendor/`, `third_party/av1of/` | Extraction layer vendored from [sigmedia/AV1-Optical-Flow](https://github.com/sigmedia/AV1-Optical-Flow) (AGPL-3.0). |

## Setup

### Ubuntu 24.04 (the cloud machine of the main results)

Requirements: [uv](https://docs.astral.sh/uv/) (Python 3.14 is fetched
automatically), meson, ninja, cmake, a C compiler, nasm, pkg-config.

```bash
sudo apt-get install -y build-essential meson ninja-build cmake nasm pkg-config \
     libaom-dev libdav1d-dev libva-dev libdrm-dev libvulkan-dev
bash setup.sh                 # patched dav1d + shim into third_party/build, then `uv sync`
bash scripts/build_ffmpeg.sh  # FFmpeg n9.0.2 + SVT-AV1 v4.2.0 + Vulkan + QSV + VA-API (~15 min)
uv run av1sfm encoders        # which AV1 encoders work on this machine
uv run pytest
```

`scripts/build_ffmpeg.sh` installs into `third_party/build/media`. av1sfm
uses `$AV1SFM_FFMPEG` if set, else that build, else the `ffmpeg` on the
`PATH`; a distribution FFmpeg ≥ 8 with the needed encoders works. `setup.sh` fetches dav1d commit `14c73c7d` from
code.videolan.org (falling back to the GitHub mirror) and applies the vendored
inspection patch. [pycolmap](https://pypi.org/project/pycolmap/) (4.2.1, CPU
build) provides COLMAP; no separate COLMAP binary is needed.

### Intel Lunar Lake / Arc on Ubuntu 26.04

Tested on a Lunar Lake laptop in a virtualenv without `uv run`. For
encoding, the GPU drivers and FFmpeg from Ubuntu 26.04's own archive
(universe and multiverse) are enough. **The GPU matchers on an Intel GPU need the compute
runtime (`intel-opencl-icd`, `libze-intel-gpu1`) 26.31.39395 or newer**, which
the archive does not have (it has 26.05.37020, which computes wrong results;
[DEPENDENCY_ISSUES.md](DEPENDENCY_ISSUES.md) 1–2). Intel's graphics PPA
provides it. With the PPA's media driver too, QSV and VA-API give
byte-identical streams on KITTI (frames 0–116), so encoding results do not
depend on which of the two sources is used:

```bash
sudo apt-get install -y build-essential meson ninja-build nasm pkg-config ffmpeg vainfo \
     intel-media-va-driver-non-free libmfx-gen1.2 libvpl2 intel-opencl-icd libze-intel-gpu1
uv venv --python 3.14 && source .venv/bin/activate
uv pip install -e . pytest
AV1SFM_SKIP_SYNC=1 bash setup.sh   # patched dav1d + shim only
av1sfm encoders                    # libaom, svtav1, qsv and vaapi should work

# For the GPU matchers: compute runtime 26.31.39395 or newer
sudo add-apt-repository ppa:kobuk-team/intel-graphics
sudo apt-get update && sudo apt-get upgrade
apt-cache policy libze-intel-gpu1   # installed version: 26.31.39395 or newer
```

The packages provide:

- VA-API AV1 encoding: `intel-media-va-driver-non-free` 26.1.2 (archive) or
  26.3.2 (PPA).
- QSV: `libvpl2` and the VPL GPU runtime `libmfx-gen1.2` (26.3.2 in the PPA).
- The GPU compute runtime used by PyTorch: `intel-opencl-icd` and `libze-intel-gpu1`,
  at least 26.31.39395 (tested: 26.31.39395.14 from the PPA; PyTorch reports
  it as Level Zero driver 1.17.39395). With an older one, `DiskLightGlue`
  refuses to run on the GPU.
- FFmpeg 8.0.1 and Mesa 26.0.8.

The laptop also has Intel's oneAPI repository
(`https://apt.repos.intel.com/oneapi`) with the oneAPI Base Toolkit
installed. av1sfm does not need it. That repository holds compilers and
libraries (DPC++, MKL, oneDNN, the oneVPL SDK), not GPU drivers. PyTorch's
XPU wheels bring their own oneAPI runtime as pip packages (`intel-sycl-rt`,
`onemkl-sycl-*`, `tcmlib`, `umf`). With the toolkit's OpenCL CPU runtime
hidden (`OCL_ICD_VENDORS` listing only the GPU driver), PyTorch loads nothing
from `/opt/intel` and the GPU tests pass.

With a virtualenv active, prefix evaluation scripts with `RUN=` (see
[RESULTS.md](RESULTS.md#reproducing-the-evaluation)). The optional GPU
matchers need PyTorch's Intel-GPU build ([below](#gpu-matchers-pytorch-intel-gpu-nvidia-gpu-or-cpu)).

## Usage

Commands are shown with `uv run`; inside an active virtualenv, leave it out.

### `av1sfm reconstruct`: images or a video to a 3D model

`av1sfm reconstruct INPUT OUT_DIR` is the one command for most uses. It runs
the whole pipeline in one process:

1. frames;
2. AV1 stream;
3. matching (AV1 motion vectors or SIFT);
4. COLMAP's mapper with the shared settings;
5. export.

INPUT is a folder of frames (in display order when sorted by name) or a
video file.

```bash
# AV1 motion vectors (the default matcher)
uv run av1sfm reconstruct video.mp4 out/                  # an AV1 video: its stream is copied
uv run av1sfm reconstruct phone.mov out/                  # any other video: encoded with libaom
uv run av1sfm reconstruct frames/ out/ --encoder vaapi    # a folder of frames, Intel GPU encoder
uv run av1sfm reconstruct frames/ out/ --ivf clip.ivf     # frames plus their existing AV1 stream

# exact SIFT matching on the GPU (the most accurate method on KITTI)
uv run av1sfm reconstruct frames/ out/ --matcher sift

# a driving sequence with a calibrated camera (KITTI 00, colour camera)
uv run av1sfm reconstruct runs/kitti117/img out/ --matcher sift \
    --camera-params 718.856,607.1928,185.2157,0 --fix-intrinsics \
    --init-max-forward-motion 1.0 --init-min-tri-angle 4
```

The last command takes 41.5 s for 117 KITTI frames on a Core Ultra 7 258V
("Performance" power profile): 7.5 s SIFT extraction, 12.0 s matching and
verification, 22.0 s mapping. All 117 images register, with 35,285 points and
a trajectory error of 0.205 m against KITTI's ground truth.

**Outputs** in OUT_DIR:

| Path | Content |
|---|---|
| `images/` | video input: its frames, `000000.png` …, one per frame of the stream (replaced on every run). An image folder is used where it is. |
| `clip.ivf` | MV matcher: the AV1 stream, copied from an AV1 video or encoded from the frames. |
| `database.db` | COLMAP database: keypoints, raw matches, two-view geometries. |
| `sparse/0/` | the reconstruction (cameras, images, points3D) in COLMAP's format; open it in the COLMAP GUI (File → Import model). If the images do not connect into one model, the others are `sparse/1/` …, and `reconstruct.json` names the largest. |
| `points.ply` | coloured point cloud of the largest model (MeshLab, CloudCompare, Blender). |
| `dataset/` | with `--export-dataset`: the undistorted workspace for OpenMVS or Brush ([Viewing and using a reconstruction](#viewing-and-using-a-reconstruction)). |
| `reconstruct.json` | the settings, the video's codec, size and frame count, matching statistics, the wall and CPU time of every stage, and the largest model's registered images, 3D points, mean reprojection error, mean track length and refined camera. |

**How the input is handled:**

- **AV1 videos are not re-encoded.** Their stream is copied into `clip.ivf`
  (`-c:v copy`), so the encoder's motion vectors come for free. Frames are
  extracted with every frame kept, MP4 edit lists ignored and the rotation of
  phone videos not applied, so that frame i is the i-th frame of the stream.
- **Ordinary AV1 videos are random access**: they have hidden alt-ref frames
  and references to later frames. These streams are supported
  (ASSUMPTIONS.md M9), but their tracks are shorter than in a low-delay
  stream. On KITTI frames 0–39, a copied libaom stream gave 16 k points and a
  1.3 m trajectory error; a low-delay re-encode gave 72 k points and 0.22 m.
  `--reencode` encodes AV1 videos as well, which is slower but more accurate.
- **Other videos and image folders** are encoded with `--encoder` (default
  libaom; `vaapi` or `qsv` on Intel GPUs) in the low-delay configuration of
  [Encoders](#encoders). With `--ivf`, an image folder's existing AV1 stream
  is used instead; it must have one shown frame per image.
- **`--matcher sift`** detects COLMAP SIFT features on the frames and does
  not need an AV1 stream. It matches exactly, in sequential order, on a
  PyTorch device; install PyTorch first
  ([GPU matchers](#gpu-matchers-pytorch-intel-gpu-nvidia-gpu-or-cpu)). This
  is the same as `eval/run_sift.py --matching sequential --matcher exact`.
  `--sift-matcher colmap` uses COLMAP's own CPU matcher and needs no PyTorch.
- **Camera:** all images share one camera, `SIMPLE_RADIAL` by default.
  Without `--camera-params`, COLMAP's prior focal length (1.2 × the larger
  image side) is the starting value and is refined. `--fix-intrinsics` keeps
  the camera at its starting values.
- OUT_DIR may already exist: `database.db`, `sparse/` and the other outputs
  are replaced.

**Options of `reconstruct`** (defaults in brackets):

| Option | Meaning |
|---|---|
| `--matcher` [mv] | `mv`: AV1 motion-vector tracks; `sift`: COLMAP SIFT features. |
| `--ivf FILE` | image-folder input: use this AV1 stream instead of encoding the frames. |
| `--reencode` | AV1 video input: encode the frames in the low-delay configuration instead of copying the stream. |
| `--frame-format` [png] | frames of a video input: `png` (lossless RGB) or `jpg` (FFmpeg's highest quality, `-q:v 2`; smaller). |
| `--export-dataset` | also write `dataset/`. |
| `--camera-model` [SIMPLE_RADIAL], `--camera-params` | the shared camera; parameters comma-separated in COLMAP's order (`f,cx,cy,k` for SIMPLE_RADIAL). |
| `--sift-matching` [sequential] | `--matcher sift`: `sequential` (each image with its next `--sift-overlap` images) or `exhaustive` (all pairs). |
| `--sift-overlap` [10] | sequential matching: neighbours per image. |
| `--sift-matcher` [exact] | `exact`: on a PyTorch device, identical to COLMAP's brute-force matcher; `colmap`: COLMAP's default (approximate) CPU matcher. |
| `--device` [auto] | PyTorch device for `exact`: `auto` picks an Intel GPU, then CUDA, then the CPU. |
| `--max-features` [8192] | SIFT features per image. |

All MV track and encoder options of `match` and all mapper options of
`eval/run_mapper.py` also apply. Both are listed in the tables below, for
example `--eps`, `--encoder vaapi`, `--qp`, `--fix-intrinsics`,
`--init-max-forward-motion` and `--no-prune-redundant-points`. `uv run
av1sfm reconstruct --help` lists everything.

### Step by step

The stages of `reconstruct` as separate commands, for experiments with one
stage or for comparing methods on the same database.

```bash
# 1. image folder (frames in display order, sorted by name) -> MV matches in a COLMAP database
uv run av1sfm match path/to/images out/database.db --ivf out/clip.ivf --encode \
    --camera-params "f,cx,cy,k"      # optional; SIMPLE_RADIAL shared by all images
#    (SIFT instead: uv run python eval/run_sift.py path/to/images out/database.db \
#                       --matching sequential --matcher exact)

# 2. mapper (identical settings for every method), statistics as JSON
uv run python eval/run_mapper.py out/database.db path/to/images out/sparse --stats out/mapper.json

# 3. point cloud and undistorted workspace
uv run python eval/export_model.py out/sparse path/to/images out/export

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

`eval/run_mapper.py` options (also accepted by `av1sfm reconstruct`). The
default is COLMAP's incremental mapper with two settings changed for every
method, which make it 24–44 % faster on KITTI without changing the accuracy:
global bundle adjustments skip redundant 3D points, and each registered image
gets one local bundle adjustment instead of up to two
([RESULTS.md](RESULTS.md#speed-where-the-time-goes-and-the-mapper-defaults-lunar-lake-performance-profile)
has the measurements). `--no-prune-redundant-points --ba-local-refinements 2`
restores COLMAP's settings, as used for the published tables.

| Option | Meaning |
|---|---|
| `--fix-intrinsics` | keep the shared camera at its initial (calibrated) value. |
| `--init-max-forward-motion`, `--init-min-tri-angle` [0.95, 16] | initial-pair constraints; KITTI's forward drive needs 1.0 and 4. |
| `--[no-]prune-redundant-points` [on] | incremental mapper; global bundle adjustments skip 3D points that add little image coverage (COLMAP's `ba_global_ignore_redundant_points3D`, off in COLMAP). |
| `--ba-local-refinements` [1] | local bundle adjustments per registered image (COLMAP: 2). |
| `--ba-global-ratio` [1.1] | run a global bundle adjustment when the model has grown by this factor (COLMAP's `ba_global_frames_ratio` and `ba_global_points_ratio`). 1.2 is another 9–18 % faster; the trajectory error was 2 % higher on one of three KITTI tests. |
| `--mapper global` | COLMAP's global mapper (GLOMAP): rotation averaging and global positioning instead of image-by-image registration. About 3× faster on MV databases, with 3–12 % fewer points and slightly worse poses. |
| `--global-tracks-per-view N` | with `--mapper global`, position the cameras with N tracks per image instead of all; every track is still triangulated afterwards. Much faster, but less accurate on noisy matches (libaom on KITTI). |
| `--kitti-sequence DIR` | compare camera poses with the KITTI ground truth in `DIR` (`NN.txt`, `calib.txt`, as fetched by `eval/fetch_kitti.py`): absolute trajectory error after a similarity alignment and relative pose error over 1 and 10 frames (`eval/pose_error.py`). |

```bash
# pose accuracy of any reconstruction against KITTI ground truth
uv run python eval/pose_error.py runs/kitti117/rec_mv data/kitti/00

# reconstruction -> point cloud (PLY) and an undistorted COLMAP workspace
uv run python eval/export_model.py out/sparse path/to/images out/export
```

## Encoders

All backends produce a low-delay stream: one keyframe, past references only,
no B-frames or look-ahead. Each MV is attached to its actual reference frame,
so every backend's reference structure is handled.

| `--encoder` | FFmpeg encoder | Configuration | Status |
|---|---|---|---|
| `libaom` (default) | `libaom-av1` | `-usage realtime -cpu-used 6 -lag-in-frames 0 -crf 32` | The paper's encoder. Tested. 97 % of MVs reference the previous frame; quarter-pel. |
| `svtav1` | `libsvtav1` (SVT-AV1 4.2) | `-preset 10 -crf 32 -svtav1-params pred-struct=1:rtc=1:keyint=-1` | Tested. About 4× faster to encode than libaom on KITTI, but layered references (51 % to n−1), 19 % compound blocks and noisier MVs (warp error 8.9 vs 5.1). Odd frame sizes (KITTI: 1241 px) are padded by one repeated column or row, which SVT-AV1 2.x requires. |
| `vulkan` | `av1_vulkan` (Vulkan Video) | hwupload to a Vulkan device, `-rc_mode cqp -qp 128 -bf 0 -tune ll -usage stream` | Needs `VK_KHR_video_encode_av1`: Mesa RADV (AMD RDNA3+) or ANV (Intel Arc / Xe2+). **Does not work on Intel Lunar Lake** (Ubuntu 26.04, Mesa 26.0.8 ANV): the driver does not expose Vulkan video encode (ASSUMPTIONS.md E5b). Not tested on AMD. |
| `qsv` | `av1_qsv` (oneVPL) | VA-API device, hwupload, `-preset veryfast -q:v 128 -bf 0 -look_ahead_depth 0` | Needs Intel Arc / Meteor Lake or newer with the VPL GPU runtime. Tested on Intel Lunar Lake: 117 KITTI frames in 0.6 s, all MVs to the previous frame (ASSUMPTIONS.md E5c). |
| `vaapi` | `av1_vaapi` (VA-API) | hwupload to a VA-API device, `-rc_mode CQP -global_quality 128 -bf 0` | Needs a VA-API driver with AV1 encode (Intel media driver on Arc / Meteor Lake / Lunar Lake, Mesa on AMD RDNA3+). Tested on Intel Lunar Lake: 117 KITTI frames in 0.5 s, the most precise MVs of all encoders tested (ASSUMPTIONS.md E5f). Not tested on AMD. |
| `auto` | | first working of `vulkan`, `qsv`, `vaapi`, `svtav1` | |

`uv run av1sfm encoders` lists the backends compiled into the ffmpeg in use
and runs a 3-frame test encode with each, printing the specific failure (for
example, a driver without `VK_KHR_video_encode_queue`). Hardware MV search is
usually coarser than software, so check the MVs on a new encoder with
`uv run python eval/mv_stats.py clip.ivf` and `uv run av1sfm validate-warp clip.ivf images/`.

### Checking a hardware encoder

On an Intel GPU (see [Setup](#intel-lunar-lake--arc-on-ubuntu-2604)), check
that the driver offers AV1 encoding:

```bash
vainfo | grep -i av1                     # want VAProfileAV1Profile0 : VAEntrypointEncSliceLP
vulkaninfo | grep -i video_encode_av1    # Vulkan Video path (Mesa ANV)
av1sfm encoders
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

## GPU matchers (PyTorch: Intel GPU, NVIDIA GPU or CPU)

Two optional baselines run on a PyTorch device (`--device auto` picks an Intel
GPU, then CUDA, then the CPU):

- **Exact SIFT matching**, `eval/run_sift.py --matcher exact`
  ([`av1sfm/sift_exact.py`](../src/av1sfm/sift_exact.py)). It returns the same
  matches as COLMAP's exact CPU matcher (`--brute-force`), pair for pair, on
  any machine and thread count. COLMAP's default CPU matcher is approximate
  and its result depends on the thread count (ASSUMPTIONS.md R4); its exact
  matcher is about 40× slower. On a GPU each pair is one matrix product of
  the two descriptor sets.
- **DISK + LightGlue**, `eval/run_lightglue.py`
  ([`av1sfm/learned.py`](../src/av1sfm/learned.py)): the learned baseline of the
  paper's Table I, using kornia's ports of both networks with their defaults
  (early stopping, point pruning; float16 attention on a GPU, as kornia does
  on CUDA). Everything runs on the GPU. `--fp32-attention` and
  `--no-pruning` select slower variants. On Intel GPUs it needs compute
  runtime 26.31.39395 or newer
  ([Intel Lunar Lake / Arc](#intel-lunar-lake--arc-on-ubuntu-2604)) and
  raises an error with an older one.

On a new GPU, run `python eval/check_device.py IMG0 IMG1` first: it compares
both matchers on the GPU against the CPU on two images and times each
LightGlue variant. `python eval/xpu_repro.py` tests an Intel GPU driver for
the bugs of compute runtime 26.05.37020
([DEPENDENCY_ISSUES.md](DEPENDENCY_ISSUES.md)).

Both use COLMAP's own pair generation and geometric verification and write
the same database layout as the other methods, so scoring and mapping are
unchanged. Install into the active virtualenv:

```bash
# Intel GPU (Lunar Lake, Arc): PyTorch's XPU build. It uses the GPU compute
# runtime (intel-opencl-icd, libze-intel-gpu1 >= 26.31.39395, see above) and
# brings its own oneAPI runtime libraries; the oneAPI toolkit is not needed.
uv pip install torch --index-url https://download.pytorch.org/whl/xpu
uv pip install kornia
python -c "import torch; print(torch.xpu.is_available(), torch.xpu.get_device_name())"

RUN= TORCH=1 ONLY="sift_seq_exact disk_seq" SETS=117 bash eval/run_kitti.sh
```

Install torch before kornia; otherwise kornia pulls PyTorch's default (CUDA)
build from PyPI. On NVIDIA, `uv pip install torch kornia` is enough. Don't run
`uv sync` afterwards: it removes packages that are not in `uv.lock`. The DISK
(`depth`) and LightGlue (`disk_lightglue`) weights are downloaded from GitHub
on first use into `~/.cache/torch/hub`. `TORCH=1` adds `sift_seq_exact`,
`sift_exh_exact`, `disk_seq` and `disk_exh`; `DEVICE=cpu` forces the CPU.

## Viewing and using a reconstruction

The mapper writes a sparse COLMAP model (camera poses and a coloured point
cloud): `OUT_DIR/sparse/0` for `av1sfm reconstruct`, or
`runs/<clip>/rec_<method>/0` in the evaluation. Open it directly in the
COLMAP GUI (File → Import model), or export it. `av1sfm reconstruct` always
writes `points.ply`, and also writes `dataset/` with `--export-dataset`. For
any other model:

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

## Tests

`tests/` (92 tests; `pytest`) uses synthetic data for:

- the cosine filter (`test_cosine.py`): thresholds, τ, ε = 1;
- MV-to-correspondence geometry (`test_blocks.py`): block collapse, centres, references, compound blocks;
- track building (`test_tracks.py`): propagation, seeding modes, cut/split/drop, 4×4 grid, gaps, triangular matches, match-count guard, references to later frames, hidden and overlay frames;
- display order and reference resolution from AV1 order hints (`test_extract.py`): low-delay and random-access streams, overlays, key frames, wrap-around;
- the encoder command lines (`test_encode.py`), the COLMAP writer (`test_colmap_db.py`) and the Sampson formulas (`test_geometry.py`);
- an end-to-end check on real AV1 encodes (libaom and SVT-AV1 low delay, libaom random access) of a synthetic pan + zoom with known ground truth (`test_integration.py`);
- `av1sfm reconstruct` on a synthetic 3D scene with known camera poses (`test_reconstruct.py`): an image folder, a copied AV1 video, a video in another codec, and both SIFT matchers;
- the exact SIFT matcher against a line-by-line port of COLMAP's brute-force loop (`test_sift_exact.py`);
- the Intel GPU driver version check (`test_devices.py`);
- pose errors against ground truth: similarity alignment, relative errors, KITTI's colour-camera offset (`test_pose_error.py`).

The PyTorch tests are skipped when PyTorch or kornia is not installed.
