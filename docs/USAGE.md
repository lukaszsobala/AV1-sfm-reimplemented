# Using av1sfm

Installation, commands and options. For what the software does, see the
[README](../README.md); for the evaluation and all measured results, see
[RESULTS.md](RESULTS.md); for every design decision, see
[ASSUMPTIONS.md](../ASSUMPTIONS.md).

## How the pipeline works

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
| `src/av1sfm/pipeline.py`, `cli.py` | The `av1sfm` command (`match`, `encode`, `encoders`, `score`, `validate-warp`) and its run statistics. |
| `src/av1sfm/sift_exact.py` | Exact SIFT matching on a PyTorch device, identical to COLMAP's brute-force matcher. |
| `src/av1sfm/learned.py`, `devices.py` | DISK + LightGlue baseline (kornia) and PyTorch device selection. |
| `eval/` | Evaluation: dataset download, SIFT / DISK baselines, mapper, tables, export, GPU checks. |
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

Tested on a Lunar Lake laptop with the GPU user-space stack from Intel's
graphics PPA and the distribution's FFmpeg, in a virtualenv without `uv run`
(package names as in that PPA; check `apt search` if they differ):

```bash
sudo add-apt-repository ppa:kobuk-team/intel-graphics
sudo apt-get install -y build-essential meson ninja-build nasm pkg-config ffmpeg vainfo \
     intel-media-va-driver-non-free libmfx-gen1 libvpl2 intel-opencl-icd libze-intel-gpu1
uv venv --python 3.14 && source .venv/bin/activate
uv pip install -e . pytest
AV1SFM_SKIP_SYNC=1 bash setup.sh   # patched dav1d + shim only
av1sfm encoders                    # libaom, svtav1, qsv and vaapi should work
```

With a virtualenv active, prefix evaluation scripts with `RUN=` (see
[RESULTS.md](RESULTS.md#reproducing-the-evaluation)). The optional GPU
matchers need PyTorch's Intel-GPU build ([below](#gpu-matchers-pytorch-intel-gpu-nvidia-gpu-or-cpu)).

## Usage

Commands are shown with `uv run`; inside an active virtualenv, leave it out.

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

```bash
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
| `svtav1` | `libsvtav1` (SVT-AV1 4.2) | `-preset 10 -crf 32 -svtav1-params pred-struct=1:rtc=1:keyint=-1` | Tested. About 4× faster to encode than libaom on KITTI, but layered references (51 % to n−1), 19 % compound blocks and noisier MVs (warp error 8.9 vs 5.1). |
| `vulkan` | `av1_vulkan` (Vulkan Video) | hwupload to a Vulkan device, `-rc_mode cqp -qp 128 -bf 0 -tune ll -usage stream` | Needs `VK_KHR_video_encode_av1`: Mesa RADV (AMD RDNA3+) or ANV (Intel Arc / Xe2+). **Does not work on Intel Lunar Lake** (Ubuntu 26.04, Intel graphics PPA): the driver does not expose Vulkan video encode (ASSUMPTIONS.md E5b). Not tested on AMD. |
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
  on CUDA). `--fp32-attention`, `--no-pruning` and `--cpu-softmax` select
  slower variants.

On a new GPU, run `python eval/check_device.py IMG0 IMG1` first: it compares
both matchers on the GPU against the CPU on two images and times each
LightGlue variant. `python eval/xpu_repro.py` reproduces the PyTorch bugs that
av1sfm works around on Intel GPUs ([DEPENDENCY_ISSUES.md](DEPENDENCY_ISSUES.md)).

Both use COLMAP's own pair generation and geometric verification and write
the same database layout as the other methods, so scoring and mapping are
unchanged. Install into the active virtualenv:

```bash
# Intel GPU (Lunar Lake, Arc): PyTorch's XPU build. It uses the GPU compute
# runtime (intel-opencl-icd, libze-intel-gpu1); oneAPI is not needed.
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

## Tests

`tests/` (67 tests; `pytest`) uses synthetic data for:

- the cosine filter (`test_cosine.py`): thresholds, τ, ε = 1;
- MV-to-correspondence geometry (`test_blocks.py`): block collapse, centres, references, compound blocks;
- track building (`test_tracks.py`): propagation, seeding modes, cut/split/drop, 4×4 grid, gaps, triangular matches, match-count guard;
- the COLMAP writer (`test_colmap_db.py`) and the Sampson formulas (`test_geometry.py`);
- an end-to-end check on a real AV1 encode of a synthetic pan + zoom with known ground truth (`test_integration.py`);
- the exact SIFT matcher against a line-by-line port of COLMAP's brute-force loop (`test_sift_exact.py`);
- LightGlue's dual softmax as computed on a GPU against kornia's (`test_learned.py`).

The PyTorch tests are skipped when PyTorch or kornia is not installed.
