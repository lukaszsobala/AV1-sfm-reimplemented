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

Gerrard Hall and Person Hall come from the COLMAP release assets
(`https://github.com/colmap/colmap/releases/download/3.11.1/gerrard-hall.zip`,
`person-hall.zip` plus `person-hall.z01`). As in the paper, a subset of
same-size, temporally adjacent images is used (Gerrard Hall: all 100; Person
Hall: IMG_1015–IMG_1229, 215 images), resized to 1920×1280, with the cosine
filter disabled (ε = 1). See ASSUMPTIONS.md D2.

`eval/run_kitti.sh` holds every command behind the table below. Steps whose
output already exists are skipped. For each set (117 and 230 frames) it runs:
the MV pipeline on libaom (with COLMAP verification and with trusted MVs) and
on SVT-AV1, COLMAP SIFT
with sequential (overlap 10) and exhaustive matching, identical pairwise
scoring of every method's raw matches, and, for the 117-frame set, incremental
mapping with identical settings.

## Results

KITTI odometry sequence 00, left colour camera (1241×376), produced by
`bash eval/run_kitti.sh` on a 4-core x86 cloud VM **without a GPU**, so SIFT
extraction and matching ran on the CPU. Software: FFmpeg n9.0.2, libaom 3.8.2,
SVT-AV1 4.2.0, pycolmap 4.2.1, Python 3.14.7. Per-run JSON summaries are in
[`results/`](results/). Gerrard Hall and Person Hall are not included: their
download host was blocked in the development environment.

How to read the tables:

- **Pre-stage** is everything before mapping, excluding encoding, which is
  listed separately. For MVs that is decode + tracks + database (+ COLMAP
  verification); for SIFT it is extraction + matching + verification.
  **Avg CPU %**: 100 % = one core.
- **Inlier ratio** and **median Sampson error** come from identical E/H
  LO-RANSAC scoring of each method's *raw* matches (max_error 4 px,
  min_inlier_ratio 0.25, 10 000 trials).
- **SfM** uses identical mapper settings for all methods, with KITTI's
  forward-motion initialisation settings (ASSUMPTIONS.md R5). The primary
  table fixes the shared SIMPLE_RADIAL camera at the KITTI calibration. With
  COLMAP's default intrinsics refinement (second table) every MV database
  fails: the first two-view bundle adjustment on an adjacent-frame initial
  pair drives the distortion to |k| ≫ 1, and COLMAP then stops registering
  images.
- The trusted-MV variant has no scoring row, because its raw matches are the
  same as the verified libaom row's.

### KITTI 00, frames 0–116

| Method | Pre-stage wall (s) | Avg CPU % | Encode (s) | Keypoints / img | Raw matches / img | Verified matches / img | Scored pairs | Inlier ratio | Median Sampson (px) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AV1 MV, libaom (COLMAP verification) | 123.0 | 366 | 8.2 | 13,915 | 234,594 | 230,615 | 5,044 | 0.972 | 0.554 |
| AV1 MV, libaom (MVs trusted, no verification) | 10.8 | 104 | – | 13,915 | 234,594 | 234,594 | – | – | – |
| AV1 MV, SVT-AV1 (COLMAP verification) | 10.9 | 318 | 1.9 | 3,510 | 14,267 | 14,036 | 975 | 0.978 | 0.460 |
| SIFT sequential (overlap 10) | 58.5 | 345 | – | 5,117 | 17,218 | 16,908 | 1,115 | 0.976 | 0.162 |
| SIFT exhaustive | 218.1 | 395 | – | 5,117 | 23,932 | 21,664 | 6,713 | 0.905 | 0.245 |

SfM, intrinsics fixed at calibration:

| Method | Registered | 3D points | Reproj. error (px) | Mean track length | Mapper wall (s) | Final global BA (s) |
|---|---:|---:|---:|---:|---:|---:|
| AV1 MV, libaom (COLMAP verification) | 117/117 | 99,499 | 0.901 | 11.07 | 827.6 | 23.0 |
| AV1 MV, libaom (MVs trusted, no verification) | 117/117 | 101,001 | 0.903 | 10.93 | 923.8 | 18.6 |
| AV1 MV, SVT-AV1 (COLMAP verification) | 116/117 | 59,992 | 0.671 | 4.15 | 219.9 | 4.4 |
| SIFT sequential (overlap 10) | 117/117 | 35,055 | 0.379 | 7.38 | 74.7 | 3.7 |
| SIFT exhaustive | 117/117 | 36,812 | 0.391 | 7.47 | 117.2 | 6.1 |

SfM, COLMAP default intrinsics refinement:

| Method | Registered | 3D points | Reproj. error (px) | Mean track length | Mapper wall (s) | Final global BA (s) |
|---|---:|---:|---:|---:|---:|---:|
| AV1 MV, libaom (COLMAP verification) | 2/117 | 17,804 | 0.441 | 2.00 | 14.9 | 3.6 |
| AV1 MV, libaom (MVs trusted, no verification) | 2/117 | 18,873 | 0.323 | 2.00 | 16.0 | 3.9 |
| AV1 MV, SVT-AV1 (COLMAP verification) | 2/117 | 8,883 | 0.260 | 2.00 | 1.7 | 1.1 |
| SIFT sequential (overlap 10) | 117/117 | 35,034 | 0.374 | 7.39 | 88.8 | 4.8 |
| SIFT exhaustive | 117/117 | 36,799 | 0.385 | 7.46 | 132.2 | 7.5 |

### KITTI 00, frames 0–229

| Method | Pre-stage wall (s) | Avg CPU % | Encode (s) | Keypoints / img | Raw matches / img | Verified matches / img | Scored pairs | Inlier ratio | Median Sampson (px) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AV1 MV, libaom (COLMAP verification) | 249.7 | 371 | 16.8 | 11,864 | 234,516 | 229,065 | 10,823 | 0.970 | 0.560 |
| AV1 MV, libaom (MVs trusted, no verification) | 22.6 | 104 | – | 11,864 | 234,516 | 234,516 | – | – | – |
| AV1 MV, SVT-AV1 (COLMAP verification) | 20.2 | 311 | 3.6 | 2,821 | 12,796 | 12,545 | 2,025 | 0.975 | 0.495 |
| SIFT sequential (overlap 10) | 109.5 | 347 | – | 4,831 | 15,380 | 15,085 | 2,245 | 0.977 | 0.154 |
| SIFT exhaustive | 874.6 | 396 | – | 4,831 | 23,966 | 20,030 | 23,712 | 0.868 | 0.247 |


### Findings

- **Density.** libaom MVs give 2.7× more keypoints and 10–14× more raw matches
  per image than SIFT, and track propagation yields matches between frames up
  to 90 apart.
- **Speed.** Without COLMAP verification the MV pre-stage takes 10.8 s for 117
  frames on one core (+8.2 s libaom encode, or 1.9 s with SVT-AV1), against
  58.5 s (SIFT sequential) and 218 s (SIFT exhaustive) on four cores.
  Verifying 13.7 M MV matches with COLMAP costs 123 s, which dominates.
- **Precision.** Inlier ratios are the same as SIFT sequential (0.97–0.98),
  but the median Sampson error is 0.55 px against 0.16 px (sequential) and
  0.25 px (exhaustive). Block-level, quarter-pel MVs are coarser than SIFT
  keypoints.
- **SfM (117 frames).** MVs reconstruct all 117 frames with 99.5 k points at
  0.90 px, against about 35–37 k points at 0.38–0.39 px for SIFT. This matches
  the paper's qualitative result (many more points, higher reprojection
  error), but the paper reports 0.46–0.62 M points at 0.51–0.53 px, against
  SIFT-sequential's 55 k at 0.30 px. Possible reasons: a different scene (the
  paper's 117-frame clip is not KITTI), track splitting, fixed intrinsics, and
  details in ASSUMPTIONS.md that have not yet been checked against the paper.
  Incremental mapping is much slower on the MV databases (828 s vs 75 s),
  because bundle adjustment scales with the 1.6 M observations.
- **SVT-AV1** encodes 4.3× faster than libaom, but its low-delay RTC MVs are
  less consistent: 31 % of track steps fail the cosine test, against 7 % for
  libaom. It ends with a quarter of the keypoints and 60 k points at 0.67 px
  (116/117 registered).
- **230 frames.** MV matching scales linearly (250 s with verification, 23 s
  without). SIFT exhaustive grows quadratically (875 s) and its inlier ratio
  drops to 0.87.


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
