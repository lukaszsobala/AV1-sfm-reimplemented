# AV1-sfm-reimplemented

An open reimplementation of

> J. Zouein, H. Javidnia, F. Pitié, A. Kokaram, *Leveraging AV1 motion vectors
> for Fast and Dense Feature Matching*, arXiv:2510.17434.

The authors did not release code; this is written from the paper's method
section. The motion vectors (MVs) that an AV1 encoder already computes are
turned into dense keypoints, multi-frame tracks and COLMAP-compatible matches,
so COLMAP's mapper can run with no SIFT extraction or matching at all.

**Read [ASSUMPTIONS.md](ASSUMPTIONS.md).** It lists every convention that was
verified (MV sign, units, reference frames) and every choice the paper leaves
open (τ, how tracks link, compound blocks, metric definitions).

## Pipeline

```
images ──ffmpeg: libaom | SVT-AV1 | Vulkan Video | QSV (low delay, 1 keyframe)──▶ clip.ivf
clip.ivf ──patched dav1d (in-memory block metadata)──▶ per-frame 4×4 MV grids
   ──collapse to coded blocks──▶ block-centre keypoints + MV targets
   ──propagate through the MV chain, cosine filter, min length 3──▶ tracks
   ──all pairs along each track (triangular adjacency)──▶ matches
   ──pycolmap──▶ database.db (keypoints, matches, two_view_geometries)
```

| Module | Role |
|---|---|
| `src/av1sfm/encode.py` | Image sequence → streaming AV1 IVF with libaom, SVT-AV1, Vulkan Video or Intel QSV (see [Encoders](#encoders)). |
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
bash scripts/build_ffmpeg.sh  # FFmpeg n9.0.2 + SVT-AV1 v4.2.0 + Vulkan + QSV (~15 min)
uv run av1sfm encoders        # which AV1 encoders work on this machine
uv run pytest                 # 54 tests; integration tests need ffmpeg and the dav1d build
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
| `--on-violation` [split] | `split` the track at a violation, or `drop` the whole track. |
| `--prev-only` | use only MVs whose reference is the previous frame. |
| `--max-pair-gap` | only emit matches between frames at most this far apart. |
| `--two-view` [verify] | `verify`: COLMAP's geometric verification with the shared RANSAC settings; `trust`: all MV matches are stored as inliers. |
| `--encoder` [libaom] | `libaom`, `svtav1`, `vulkan`, `qsv`, or `auto` (see below). |
| `--crf` [32], `--qp` [128] | CRF for libaom/SVT-AV1; constant AV1 qindex (0–255) for Vulkan/QSV. |
| `--usage`, `--cpu-used` [realtime, 6] | libaom settings. |
| `--svt-preset` [10], `--svt-params` | SVT-AV1 preset and extra `key=value:...` parameters. |
| `--hw-device` | Vulkan device index, or QSV DRM render node (e.g. `/dev/dri/renderD128`). |

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
| `auto` | | first working of `vulkan`, `qsv`, `svtav1` | |

`uv run av1sfm encoders` lists the backends compiled into the ffmpeg in use
and runs a 3-frame test encode with each, printing the specific failure (for
example, a driver without `VK_KHR_video_encode_queue`). Hardware MV search is
usually coarser than software, so check the MVs on a new encoder with
`uv run python eval/mv_stats.py clip.ivf` and `uv run av1sfm validate-warp clip.ivf images/`.

## Reproducing the evaluation

```bash
bash eval/run_kitti.sh     # fetches KITTI 00 frames 0-229, runs everything, writes runs/results.md
```

`eval/run_kitti.sh` holds every command behind the table below. Steps whose
output already exists are skipped. For each set (117 and 230 frames) it runs:
the MV pipeline on libaom (with COLMAP verification and with trusted MVs) and
on SVT-AV1, COLMAP SIFT
with sequential (overlap 10) and exhaustive matching, identical pairwise
scoring of every method's raw matches, and, for the 117-frame set, incremental
mapping with identical settings.

## Results

RESULTS_PLACEHOLDER

## Tests

`tests/` uses synthetic data for:

- the cosine filter (`test_cosine.py`): thresholds, τ, ε = 1;
- MV-to-correspondence geometry (`test_blocks.py`): block collapse, centres, references, compound blocks;
- track building (`test_tracks.py`): propagation, seeding, splitting and dropping, gaps, triangular matches;
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
