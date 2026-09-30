# Assumptions and verified conventions

This repository reimplements Zouein, Javidnia, Pitié, Kokaram, *Leveraging AV1
motion vectors for Fast and Dense Feature Matching* (arXiv 2510.17434). The
authors did not release code. **The paper text was not accessible from the
development environment (arXiv was blocked by the network policy)**, so the
method was implemented from a written summary of its method section. Every
point below is either *verified* (with the check that verified it) or an
*assumption* (a choice the summary leaves open). Revisit the assumptions
against the paper.

Measured numbers refer to KITTI odometry sequence 00, left colour camera
(`image_2`), frames 0–116, encoded with the default settings below, unless
stated otherwise. They are produced by `eval/mv_stats.py`.

## Encoding

| # | Item | Status | Details |
|---|---|---|---|
| E1 | Encoder profile | assumption, evidence-backed | Default: libaom via FFmpeg, `-usage realtime -cpu-used 6 -lag-in-frames 0 -crf 32 -b:v 0`, one keyframe (`-g 1000000`). The summary says "streaming configuration, libaom cpu-used=6, quarter-pel". CRF 32 is FFmpeg's libaom default (FFmpeg prints "Neither bitrate nor constrained quality specified, using default CRF of 32"). |
| E2 | Why `realtime` | verified | In the realtime profile at CRF 32, **0 %** of MV components are odd in 1/8 pel (i.e. quarter-pel, as the paper states) and **≈98 %** of MVs reference the previous frame on a 12-frame KITTI probe (0.54 % and 97.6 % over 117 frames, E4). In the `good` profile at the same speed, ≈86 % reference the previous frame and ≈44–48 % of components are odd 1/8-pel at every CRF tried (30, 31, 32, 33, 35, 40, 45, 48, 50, 52, 55). The realtime profile at CRF 30 still used 1/8 pel; from CRF 32 on it is quarter-pel. `--usage good` remains available. |
| E3 | "Every MV points to the previous frame" | verified false as a hard guarantee | libaom always keeps ≥3 reference slots (`max-reference-frames` minimum is 3), so some blocks reference n−2 … n−7. We never assume the reference is n−1; each MV is attached to its actual reference frame via the decoder's reference order hints (`refpoc`). `--prev-only` restricts to MVs whose reference is n−1 (the paper's idealisation). |
| E4 | Measured reference statistics (117 frames) | measured | 317 k coded blocks, 82.6 % inter. Reference distance of the MVs: 1 frame 97.6 %, 4 frames 1.9 %, all other distances 0.5 % combined (a long tail of GOLDEN references up to 80 frames back, typically on distant or static content). **Compound (two-reference) blocks: 0 %.** Zero MVs: 0.44 % of inter blocks. Odd 1/8-pel components: 0.54 % (almost entirely quarter-pel). Most common block sizes: 8×8 (49 %), 16×16 (48 %). |
| E5 | Encoder backends | implemented | `--encoder libaom` (default, the paper's encoder), `svtav1`, `vulkan`, `qsv`, or `auto` (first working of vulkan, qsv, svtav1). NVENC was removed: no NVIDIA hardware is targeted. FFmpeg n9.0.2 with SVT-AV1 v4.2.0, libvpl v2.17.0 and Vulkan headers v1.4.364 is built by `scripts/build_ffmpeg.sh`; libaom is the system 3.8.2. |
| E5a | SVT-AV1 configuration | assumption, measured | `libsvtav1 -preset 10 -crf 32 -svtav1-params pred-struct=1:rtc=1:keyint=-1` (low delay, real-time mode, one keyframe). On KITTI 00 frames 0–29 (libaom realtime in brackets): encode 0.8 s (3.0 s); MVs to n−1 51 % (97 %), n−2 25 %, n−3 10 %, n−4 12 % (SVT's low-delay structure is layered into temporal layers, and `hierarchical-levels=2` did not change it in RTC mode); **compound blocks 19 % of inter blocks** (0 %); odd 1/8-pel components 0 % (0 %); inter coverage 71 % of pixels (83 %); warp MAE 8.9 (5.1) against 34 unwarped. Preset 8 has 1/8-pel MVs (36 %) and 57 % coverage; preset 12 has 11.2 warp MAE. `rtc=0` gave lower coverage and higher warp error. Because references skip frames, SVT tracks visit fewer frames for the same span (tests use span, not length). |
| E5b | Vulkan Video (`av1_vulkan`) | untested on hardware | `-init_hw_device vulkan=vk[:N] -filter_hw_device vk -vf format=nv12,hwupload -c:v av1_vulkan -rc_mode cqp -qp 128 -bf 0 -tune ll -usage stream -content camera -g <frames+1>`. Needs a driver exposing VK_KHR_video_encode_queue + VK_KHR_video_encode_av1 (Mesa RADV on AMD RDNA3+, ANV on Intel Arc / Xe2+). The development container only has lavapipe: device creation, upload and option parsing succeed, and the encoder stops at "Device does not support the VK_KHR_video_encode_queue extension". `av1sfm encoders` reports this per machine. |
| E5c | Intel QSV (`av1_qsv`) | untested on hardware | `-init_hw_device vaapi=va:/dev/dri/renderDxxx -init_hw_device qsv=qs@va` (or `qsv=qs`) `-filter_hw_device qs -vf format=nv12,hwupload=extra_hw_frames=64 -c:v av1_qsv -preset veryfast -q:v 128 -bf 0 -look_ahead_depth 0 -async_depth 1 -g <frames+1>`. Needs Intel Arc / Meteor Lake or newer with the VPL GPU runtime and iHD VA driver. Not testable here (no Intel GPU). |
| E5d | Hardware quality setting | assumption | Hardware encoders take a constant AV1 quantizer index (0–255) instead of a CRF. Default 128 ≈ libaom CRF 32 (libaom maps CRF q to qindex ≈ 4q). Hardware MV search is usually coarser than software, so MV statistics should be re-measured with `eval/mv_stats.py` and `av1sfm validate-warp` on the target machine. |
| E5e | Padded hardware frames | handled | Hardware encoders may code a frame padded to 16/64 px and signal the real size as render size. Decoded frames are clamped to the source image size (`pipeline.clamp_to_image_size`). |
| E6 | Odd frame sizes | verified | KITTI is 1241×376. libaom and dav1d handle odd sizes; the 4×4 metadata grid is padded to a multiple of 8 px, and blocks overhanging the frame are clipped (B3). Covered by `tests/test_integration.py` (321×181 clip). |

## Motion-vector extraction

| # | Item | Status | Details |
|---|---|---|---|
| M1 | Extraction layer | reused | Vendored from sigmedia/AV1-Optical-Flow @ `a9327960` (dav1d inspection patch + GIL-free shim + `dav1d_inspect.py`); see `third_party/av1of/NOTICE.md`. |
| M2 | Units | verified | Raw integers are 1/8 pel; we divide by 8. Channel order `[mv0_x, mv0_y, mv1_x, mv1_y]` (x = horizontal). |
| M3 | Sign / direction | verified | A block at position **p** in frame *n* with MV **v** is predicted from **p + v** in its reference frame *m* < *n* ("backward", current → reference). Verified three ways: (a) synthetic pan: content moving +6 px/frame gives MV −6 px per frame of reference distance, including a 6-frame GOLDEN reference (−36 px); (b) `tests/test_integration.py`: synthetic pan + zoom, block targets are within 0.07 px (median) of ground truth; (c) warping real KITTI references with the MVs (`av1sfm validate-warp`) gives MAE ≈ 5 grey levels vs ≈ 26–36 without warping and ≈ 34–45 with the sign flipped. This agrees with the upstream repo naming list-0 MVs "backward" (current to past reference). |
| M4 | Block-level vs dense flow | assumption | Block-level MVs. The upstream `--upscale_function` output interpolates the block field to per-pixel flow, which adds no information but would create one keypoint per pixel. The paper's keypoints are block centres, so the block grid is the right level. |
| M5 | Zero MVs | as specified | (0, 0) MVs are skipped (ambiguous: static, skip or global-motion blocks). They are 0.3–1 % of inter blocks on KITTI. |
| M6 | Intra blocks / frames | as specified | Blocks with reference ≤ 0 (intra) and the key frame carry no MV and yield no source keypoint. A track that lands in one ends there. |
| M7 | Bi-predicted (compound) blocks | assumption | In the streaming configuration both references are in the past. Each of the two MVs is a separate correspondence to its own reference (`frame_block_motion`). When propagating a track through a compound block we follow the MV whose reference is temporally nearest (list 0 on ties), because tracks link consecutive frames. Measured compound share on KITTI: 0 % with libaom (E4) but 19 % with SVT-AV1 preset 10 (E5a), so the SVT-AV1 runs and the SVT integration test exercise this path on real data. |
| M8 | Order hints | assumption | 7 order-hint bits (modulo 128), as in the upstream repo. Absolute frame indices are recovered by unwrapping relative to the previous frame; an order hint that goes backwards raises an error (would indicate hidden frames or future references). |

## Blocks and correspondences

| # | Item | Status | Details |
|---|---|---|---|
| B1 | Collapsing the 4×4 grid | as specified | The grid holds one record per 4×4 unit, so a 32×32 block appears 64 times. AV1 partitions are aligned to their own size, so a W×H block's top-left unit is the one whose grid coordinates are multiples of (W/4, H/4); we keep only those. `tests/test_blocks.py` covers mixed partitions. |
| B2 | Keypoint coordinates | assumption | COLMAP convention: pixel (i, j) covers [i, i+1)×[j, j+1). A block covering pixel columns x0 … x0+W−1 has centre x0 + W/2. |
| B3 | Blocks overhanging the frame | assumption | The centre of the *visible* part of the block is used. |
| B4 | Targets outside the frame | assumption | Dropped. |

## Tracks and filtering

| # | Item | Status | Details |
|---|---|---|---|
| T1 | Linking rule | assumption | A track is a point propagated backwards through the MV chain. It starts at a block centre **c** in frame *n*, moves to **c + v_nm** in *m*, then to **c + v_nm + v_ml** in *l*, where v_ml is the MV of the block of *m* that contains **c + v_nm**, and so on. Points are not snapped to block centres: keypoints in frame *m* are the propagated positions. |
| T2 | Seeding | assumption | Blocks that no arriving track lands in seed new tracks at their centres. Blocks already holding an arriving point do not, which avoids near-duplicate keypoints on static content. Frames are processed last to first so all arrivals are known before seeding. |
| T3 | Cosine test | as specified | For each consecutive triple (n, m, l): cos(v_nm, v_ml) ≥ 1 − ε, ε = 0.1 by default. MVs spanning several frames are compared by direction only, since the cosine is scale-invariant. |
| T4 | ε = 1 | interpretation | The summary says ε = 1 disables the filter. Taken literally, ε = 1 would still reject cos < 0, so ε ≥ 1 is treated as "off". |
| T5 | τ (not given in the paper) | assumption, evidence-backed | **τ = 2 px**: the test is skipped when either vector is shorter. Violation rate (cos < 0.9) by the shorter vector's length, over raw two-step chains on KITTI (30 frames): 1–2 px: 42 %, 2–4 px: 19 %, 4–8 px: 5.8 %, 8–16 px: 4.5 %, ≥16 px: 5.0 %. Above ≈4 px the rate is flat at the outlier floor. Below 2 px, block-level MVs are too noisy for their direction to be tested. `--tau` sets it. |
| T6 | On a violation | assumption | The track is **split** at the middle frame: the part before ends there and a new track continues from the same point. `--on-violation drop` discards the whole track instead. |
| T7 | Minimum track length | as specified | 3 observations (frames); shorter tracks are removed after splitting. |
| T8 | Matches from tracks | as specified | Every pair of frames on a track becomes a match (triangular adjacency). No frame-gap cap by default (`--max-pair-gap`). |

## COLMAP export

| # | Item | Status | Details |
|---|---|---|---|
| C1 | Database creation | assumption | Cameras, rigs, frames and images come from `pycolmap.import_images` (same code path as COLMAP's extractor), so the schema matches the installed version (pycolmap 4.2.1). No descriptors are written. |
| C2 | Shared intrinsics | as specified | One SIMPLE_RADIAL camera shared by all images and all methods. KITTI uses the rectified P2 intrinsics (f = 718.856, cx = 607.1928, cy = 185.2157, k = 0) as the initial value. |
| C3 | two_view_geometries | assumption | Default: COLMAP's own multi-threaded `geometric_verification` over the raw MV matches, with exactly the options used for the SIFT baselines (max_error 4 px, min_inlier_ratio 0.25, max_num_trials 10 000; other options COLMAP defaults). `--two-view trust` stores all MV matches as inliers (config UNCALIBRATED), i.e. the MVs are their own verification; both are reported. |
| C4 | Minimum matches per pair | assumption | Pairs with fewer than 15 raw matches are not written (COLMAP's `min_num_inliers` default). |

## Pairwise geometric scoring

| # | Item | Status | Details |
|---|---|---|---|
| G1 | What is scored | assumption | The **raw** (pre-verification) matches of each method, so verification does not pre-filter the numbers. |
| G2 | Estimators | as specified | Five-point essential matrix (`pycolmap.estimate_essential_matrix`) and DLT homography (`pycolmap.estimate_homography_matrix`), COLMAP LO-RANSAC, max_error 4 px, min_inlier_ratio 0.25, max_num_trials 10 000, confidence 0.9999 and min_num_trials 1000 (COLMAP defaults), fixed seed 0. |
| G3 | Normalised coordinates | as specified | Points are mapped with K⁻¹ (`Camera.cam_from_img`). The 4 px threshold is converted with the mean focal length: pycolmap does this itself for E, and we do it explicitly for H. |
| G4 | Model choice | as specified | More inliers wins; ties go to the lower median residual. |
| G5 | Sampson error | assumption | Reported as a distance (square root of the Sampson squared error) over the winning model's inliers, in normalised units and converted to pixels (× mean focal length). For a winning homography we use its Sampson (first-order geometric) error, the analogue of the epipolar Sampson error. Both formulas are checked against closed forms in `tests/test_geometry.py`. |
| G6 | Aggregation | assumption | Inlier ratio is reported pooled (Σ inliers / Σ matches); the per-pair mean is in the JSON. Sampson error is the median over pairs of per-pair medians. |

## Metrics and timing

| # | Item | Status | Details |
|---|---|---|---|
| R1 | Pre-stage wall time | assumption | MV: decode + MV extraction + tracks + database write + verification. Encoding is reported separately: the paper's premise is that MVs come for free with a compressed stream. SIFT: extraction + matching + verification. |
| R2 | Average CPU % | assumption | (user + system CPU of the process and its children) / wall time; 100 % = one fully busy core. The machine has 4 cores. |
| R3 | Matches per image | assumption | 2 × (total matches) / (number of images), i.e. the mean over images of the matches in all pairs involving that image; reported raw and verified. |
| R4 | SIFT on CPU | environment | No GPU was available, so SIFT extraction and matching ran on the CPU (pycolmap 4.2.1, 8192 features max). SIFT timings are therefore slower than a GPU setup. |
| R5 | Mapper settings | assumption | COLMAP incremental mapper defaults for every method, a single model, and the shared camera (C2). KITTI is forward motion: with the default `init_max_forward_motion = 0.95` no initial pair is ever accepted (all methods failed), and SIFT-sequential pairs (≤10 frames apart) have 1–2° median triangulation angle against a 16° requirement. For KITTI all methods therefore use `init_max_forward_motion = 1.0` and `init_min_tri_angle = 4` (COLMAP relaxes by halving). |
| R6 | BA time | assumption | pycolmap does not expose the bundle-adjustment share of incremental mapping. We report total mapper wall time plus the wall time of one final global BA of the reconstructed model. |
| R7 | 117-frame SfM demo | assumption | The summary gives the clip length (≈117 frames) but not its content; KITTI 00 frames 0–116 are used. |

## Datasets

| # | Item | Status | Details |
|---|---|---|---|
| D1 | KITTI 00 | as specified | First 230 frames of `image_2`, downloaded member-by-member from the official archive with HTTP range requests (`eval/fetch_kitti.py`). |
| D2 | Gerrard Hall, Person Hall | blocked | The COLMAP dataset host (demuc.de) was blocked by the network policy. The pipeline supports them unchanged (image folder → video, `--eps 1`). |
| D3 | AV1-3D-Reconstruction | unavailable | github.com/sigmedia/AV1-3D-Reconstruction was still not reachable (not found / private), so no comparison was made. |
