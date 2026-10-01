# Assumptions and verified conventions

This repository reimplements Zouein, Javidnia, Pitié, Kokaram, *Leveraging AV1
motion vectors for Fast and Dense Feature Matching* (arXiv 2510.17434v2,
ICIR 2025; "the paper"). The authors did not release code. The implementation
was first written from a summary of the method section, then checked against
the paper and against the group's follow-up paper, Zouein, Vibhoothi, Pitié,
Kokaram, *Efficient dense matching for enhanced Gaussian splatting using AV1
motion vectors* (arXiv 2605.14629, "the follow-up"). Section 3 of the follow-up
describes the paper's matcher. Both PDFs are in the repository root.

Each item has a **paper status**:

- **confirmed**: the paper states it, and the implementation follows it (quote or section given);
- **changed to match**: the first implementation differed and was changed;
- **open**: the paper is silent or ambiguous, so this is our choice, with the evidence for it;
- **deviation**: we knowingly differ (reason given).

Measured numbers refer to KITTI odometry sequence 00, left colour camera
(`image_2`), frames 0–116, with the default settings below, unless stated
otherwise. They are produced by `eval/mv_stats.py`.

## Encoding

| # | Item | Paper status | Details |
|---|---|---|---|
| E1 | Encoder and profile | confirmed / open | Paper §III-D: libaom-av1 v3.12.1 via FFmpeg n6.0, `cpu-used=6`; NVENC-AV1 preset 1; "Streaming Configuration (S3-SCC-03 [3GPP TR 26.955]) where all the reference frames are in the past, with only one intra-frame … and all motion vectors pointing to the previous frame"; §II-A: "we use 1/4th pixel precision". The libaom usage profile and rate setting are not stated (open). We use `-usage realtime -cpu-used 6 -lag-in-frames 0 -crf 32 -b:v 0` and one keyframe; CRF 32 is FFmpeg's libaom default. **Deviation:** libaom 3.8.2 (the system library; the 3.12.1 source host was unreachable) and FFmpeg n9.0.2. |
| E2 | Why `realtime` | open, evidence-backed | It is the only libaom configuration we found that reproduces the paper's two stated properties. At CRF 32, **0 %** of MV components are odd in 1/8 pel (quarter-pel, as stated) and **≈98 %** of MVs reference the previous frame (12-frame KITTI probe; 0.55 % and 97.7 % over 117 frames, E4). In the `good` profile, ≈86 % reference the previous frame and 44–48 % of components use 1/8 pel at every CRF tried (30–55). The realtime profile at CRF 30 still used 1/8 pel. |
| E2b | `good` usage on image collections | measured, deviation from E2 | On Gerrard Hall and Person Hall, which are photo collections with large jumps between images (only 37–41 % of blocks inter-coded), realtime MVs give a pooled inlier ratio of 0.68 / 0.67 (0.78 / 0.79 between adjacent images) and a median Sampson error of about 1 px, against the paper's 0.96 (Table IV, libaom). libaom's default `good` usage (cpu-used 6, CRF 32) has a wider motion search and gives 0.91 / 0.94 pooled, 0.94 / 0.97 between adjacent images, and 0.49 / 0.46 px, at 6× the encoding time (83 s / 194 s vs 13 s / 33 s). Reported as a separate method, `mv_good` (`--usage good`), in `eval/run_colmap_datasets.sh`; realtime stays the default because it matches the paper's stated MV properties (E2) and is better on KITTI. The paper's NVENC values on these datasets (0.47 / 0.43) resemble ours with realtime usage. |
| E3 | "All motion vectors pointing to the previous frame" | deviation (encoder limit) | libaom always keeps ≥3 reference slots, so some blocks reference n−2 … n−80. We never assume the reference is n−1: each MV is attached to its actual reference frame via the decoder's reference order hints. `--prev-only` keeps only MVs whose reference is n−1 (the paper's idealisation). |
| E4 | Measured reference statistics | measured | Encode used in the evaluation (FFmpeg n9.0.2, libaom 3.8.2, E1): 316 k coded blocks, 82.8 % inter. MV reference distance: 1 frame 97.7 %, 4 frames 1.9 %, all others 0.4 % (a long tail of GOLDEN references up to 80 frames back). **Compound (two-reference) blocks: 0 %.** Zero MVs 0.40 % of inter blocks. Odd 1/8-pel components 0.55 %. Block sizes: 8×8 49 %, 16×16 48 %. |
| E5 | Encoder backends | extension | `--encoder libaom` (default), `svtav1`, `vulkan`, `qsv`, `vaapi`, `auto`. The paper's second encoder, NVENC, is not supported: no NVIDIA hardware is targeted. For reference, the follow-up's NVENC command is `-c:v av1_nvenc -tune hq -preset p1 -rc constqp -qp <qp> -g 9999 -b_ref_mode 0 -bf 0`. FFmpeg n9.0.2 with SVT-AV1 v4.2.0, libvpl v2.17.0 and Vulkan headers v1.4.364 is built by `scripts/build_ffmpeg.sh`. |
| E5a | SVT-AV1 configuration | extension, measured | `libsvtav1 -preset 10 -crf 32 -svtav1-params pred-struct=1:rtc=1:keyint=-1`. On KITTI frames 0–29 (libaom in brackets): encode 0.8 s (3.0 s); MVs to n−1 51 % (97 %), n−2 25 %, n−3 10 %, n−4 12 % (SVT's low-delay structure has temporal layers; `hierarchical-levels=2` did not change it in RTC mode); **compound blocks 19 %** (0 %); odd 1/8-pel components 0 % (0 %); pixels with a usable MV 71 % (83 %); warp MAE 8.9 (5.1), against 34 unwarped. Preset 8 has 1/8-pel MVs (36 %); preset 12 has warp MAE 11.2; `rtc=0` was worse. |
| E5b | Vulkan Video (`av1_vulkan`) | extension, not working on the tested hardware | `-init_hw_device vulkan=vk[:N] -filter_hw_device vk -vf format=nv12,hwupload -c:v av1_vulkan -rc_mode cqp -qp 128 -bf 0 -tune ll -usage stream -content camera -g <frames+1>`. Needs VK_KHR_video_encode_queue + VK_KHR_video_encode_av1 (Mesa RADV on AMD RDNA3+, ANV on Intel Arc / Xe2+). In the development container (lavapipe only), device creation, upload and option parsing succeed, and the encoder stops at "Device does not support the VK_KHR_video_encode_queue extension". On Intel Lunar Lake (Ubuntu 26.04, Intel graphics PPA, system FFmpeg) it stops at the same message: the driver did not expose Vulkan video encode. |
| E5c | Intel QSV (`av1_qsv`) | extension, tested on Intel Lunar Lake | `-init_hw_device vaapi=va:/dev/dri/renderDxxx -init_hw_device qsv=qs@va` (or `qsv=qs`), `-filter_hw_device qs -vf format=nv12,hwupload=extra_hw_frames=64 -c:v av1_qsv -preset veryfast -q:v 128 -bf 0 -look_ahead_depth 0 -async_depth 1 -g <frames+1>`. Needs Intel Arc / Meteor Lake or newer with the VPL GPU runtime. On Lunar Lake (KITTI 00, 117 frames): encode 0.6 s, **100 %** of MVs reference the previous frame, quarter-pel, 75 % inter blocks, larger blocks than libaom (215 k vs 316 k coded blocks), warp MAE 4.6–6.1 (libaom 4.5–6.1). Results in `results/lunar-lake/`. |
| E5d | Hardware quality setting | open | Constant AV1 qindex 128 ≈ libaom CRF 32 (libaom maps CRF q to qindex ≈ 4q). The follow-up uses constant QP with NVENC but does not give the value. Re-measure MVs with `eval/mv_stats.py` and `av1sfm validate-warp` on the target GPU. |
| E5e | Padded hardware frames | handled | Decoded frames larger than the source images (hardware padding) are clamped to the image size. |
| E5f | VA-API (`av1_vaapi`) | extension, tested on Intel Lunar Lake | `-init_hw_device vaapi=va[:/dev/dri/renderDxxx] -filter_hw_device va -vf format=nv12,hwupload -c:v av1_vaapi -rc_mode CQP -global_quality 128 -bf 0 -async_depth 1 -g <frames+1>`. `-global_quality` is the qindex itself; `-q:v` would set the QSCALE flag and FFmpeg would divide it by `FF_QP2LAMBDA`. The I-frame qindex is scaled by FFmpeg's default `i_qfactor` (0.8). Intended for Intel Lunar Lake (Xe2) on Ubuntu 26.04 with Intel's graphics PPA; the direct VA-API path avoids the oneVPL layer. `auto` tries it after Vulkan and QSV. On Lunar Lake (KITTI 00, 117 frames): encode 0.5 s, 92.6 % of MVs to the previous frame and 7.4 % to the one before, quarter-pel, 67 % inter blocks, warp MAE 4.1–5.0, the lowest of all encoders. Results in `results/lunar-lake/`. |
| E6 | Odd frame sizes | verified | KITTI is 1241×376. The 4×4 metadata grid is padded to a multiple of 8 px; blocks overhanging the frame are clipped (B3). Covered by `tests/test_integration.py` (321×181 clip). |

## Motion-vector extraction

| # | Item | Paper status | Details |
|---|---|---|---|
| M1 | Extraction layer | deviation (planned by the authors) | The paper uses libaom's `inspect` tool and lists dav1d as future work (§IV). We vendor the dav1d-based extractor from sigmedia/AV1-Optical-Flow @ `a9327960`, which is validated bit-exact against `inspect` for MVs and reference maps (upstream `test/validate_dav1d_vs_json.py`). |
| M2 | Units | confirmed | §II-A: "a division by a factor of 8 is required". Channel order `[mv0_x, mv0_y, mv1_x, mv1_y]`. |
| M3 | Sign / direction | verified | The paper says the MV "points to a source location in a designated Reference Frame" but gives no sign convention. Verified: a block at **p** in frame *n* with MV **v** is predicted from **p + v** in reference frame *m* < *n*. Checks: (a) synthetic pan, content +6 px/frame gives MV −6 px per frame of reference distance, including a 6-frame GOLDEN reference (−36 px); (b) `tests/test_integration.py`, synthetic pan + zoom, block targets within 0.07 px (median) of ground truth; (c) warping real KITTI references gives MAE ≈ 5 grey levels vs ≈ 26–36 unwarped and ≈ 34–45 with the sign flipped. |
| M4 | Block level vs dense flow | confirmed / open | Not per-pixel dense flow: the paper's keypoints are block centres (§II-A, Fig. 2 "Use center of each block as keypoint"). Whether "block" means the coded block or the 4×4 unit is open, see B1. |
| M5 | Zero MVs | confirmed | §II-A: "We do not consider blocks with a (0,0) motion vector, as this value is ambiguous." 0.4–1 % of inter blocks on KITTI. |
| M6 | Intra blocks / frames | confirmed | §II-A: intra and skipped blocks "do not have any associated motion vectors". They yield no source keypoint, and a track landing in one ends there. |
| M7 | Bi-predicted (compound) blocks | open | Not discussed (the paper assumes a single previous-frame reference). Each of the two MVs is a separate correspondence to its own reference; a track passing through a compound block follows the MV to the temporally nearest reference. Compound share: 0 % with libaom (E4), 19 % with SVT-AV1 (E5a). |
| M8 | Order hints | open | 7 order-hint bits (modulo 128), as in the upstream extractor; an order hint that goes backwards raises an error. |

## Blocks and correspondences

| # | Item | Paper status | Details |
|---|---|---|---|
| B1 | Keypoint grid | open; default follows the project brief | The paper says "for each block (p,q) in a frame n, we emit a source keypoint at the center of the block", and Fig. 2 shows "Get Block Map". The follow-up describes the paper's pipeline as upsampling "the motion field (using zero order hold) from non-uniform blocks to yield a motion vector for every 4×4 block". Default `--grid block`: the 4×4 metadata grid collapsed to coded blocks (partitions are size-aligned, so a W×H block's top-left unit has grid coordinates divisible by (W/4, H/4)). `--grid cell` uses every 4×4 unit. Evidence for `block`: on KITTI, `cell` produces 215 k keypoints and 3.6 M raw matches per image (212 M in total for 117 frames). That is far beyond what the paper's 460–616 k reconstructed points on a 117-frame 1080p clip suggest, whereas `block` gave 99.5 k points on KITTI (4.4× fewer pixels than 1080p), which extrapolates to ≈440 k. |
| B2 | Keypoint coordinates | open | COLMAP convention: pixel (i, j) covers [i, i+1)×[j, j+1). A block covering columns x0 … x0+W−1 has centre x0 + W/2. |
| B3 | Blocks overhanging the frame | open | The centre of the visible part of the block is used. |
| B4 | Targets outside the frame | open | Dropped. |

## Tracks and filtering

| # | Item | Paper status | Details |
|---|---|---|---|
| T1 | Linking rule | confirmed | Paper §II-B: tracks link "consistent correspondences across consecutive frames", t_k = {(n, x_n), (m, x_m), (ℓ, x_ℓ), …}. Fig. 2: "Apply motion vector to keypoint → Get target block from reference frame → Save as match → Generate tracks". Follow-up: "they build motion trajectories by accumulating motion vectors themselves". Implementation: a point starts at a block centre **c** in frame *n*, moves to **c + v_nm** in *m*, then by the MV of the block of *m* containing that point, and so on. Positions are accumulated, not snapped to block centres. |
| T2 | Seeding | changed to match | Paper §II-A: every block of frame *n* emits a source keypoint, and "the generated target point is added to the source keypoints of frame m". **Default `--seed all`**: every block of every frame starts a track, and arriving points are additional keypoints. The first implementation seeded only blocks that no arriving track landed in (`--seed uncovered`, kept as an option); `all` gives ≈1.5× more keypoints and matches on KITTI. |
| T3 | Cosine test | confirmed | Eq. (1): ⟨v_nm, v_mℓ⟩ / (‖v_nm‖‖v_mℓ‖) ≥ 1 − ε for each consecutive triple, ε = 0.1 by default. The cosine is scale-invariant, so MVs spanning several frames are compared by direction only. |
| T4 | ε = 1 | confirmed | §II-B: "we disable the filter by setting ε=1 (threshold 0)". Literally, a threshold of 0 still rejects cos < 0; following the stated intent, ε ≥ 1 disables the test. §III-E: ε = 1 for Gerrard Hall and Person Hall, 0.1 elsewhere. |
| T5 | τ | open (paper: "a small τ") | §II-B: "If either vector has magnitude below a small τ, we skip the test to avoid numerical instability." No value given. **τ = 2 px.** Violation rate (cos < 0.9) by the shorter vector's length on KITTI: 1–2 px 42 %, 2–4 px 19 %, 4–8 px 5.8 %, ≥ 8 px ≈ 5 % (libaom good profile, 30 frames); realtime profile, 117 frames: 2–4 px 33 %, 4–8 px 19 %, 8–16 px 5.8 %, ≥ 16 px 1.8 %. Below ≈2 px, block-level quarter-pel directions are too noisy to test. |
| T5a | Violation rates in practice | measured | Per consecutive step with τ = 2 px: libaom (realtime) 6.9 %, SVT-AV1 (preset 10, RTC) 31 %. SVT-AV1's MVs are much less consistent. |
| T6 | On a violation | changed to match | Paper: "Motion vectors with bad cosine similarity are deleted and not considered for matches." Follow-up: "They terminate a trajectory when the cosine difference … is greater than some threshold, since that indicates loss of track." **Default `--on-violation cut`**: the track ends at the middle frame *m* and the offending MV is not followed for it (frame *m*'s own blocks still start tracks, T2). The first implementation split the track and continued from the same point (`split`); `drop` discards the whole track. |
| T7 | Minimum track length | confirmed | Paper: "discard short tracks (length < 3)". The follow-up says "persist across more than 3 frames"; we follow the paper (≥ 3 observations). |
| T8 | Matches from tracks | confirmed | §II-B: propagation "transforms the diagonal adjacency … into a triangular pattern" (Fig. 1). Every pair of frames on a track is a match. No frame-gap cap by default (`--max-pair-gap`); more than 10⁸ matches is refused unless a gap is set. |

## COLMAP export

| # | Item | Paper status | Details |
|---|---|---|---|
| C1 | Database creation | confirmed / open | Fig. 2: "import images in COLMAP → retrieve image's ID from COLMAP database → … save matches in COLMAP database → COLMAP incremental mapping". Cameras, rigs, frames and images come from `pycolmap.import_images`; no descriptors are written. |
| C2 | Shared intrinsics | confirmed / open | §III-C: "identical SIMPLE RADIAL intrinsics and mapper settings across all methods". Whether they were refined is not stated. KITTI uses the rectified P2 intrinsics (f = 718.856, cx = 607.1928, cy = 185.2157, k = 0), see R5. |
| C3 | two_view_geometries | open | Not described. Default: COLMAP's own multi-threaded `geometric_verification` over the raw MV matches, with the options used for the SIFT baselines (max_error 4 px, min_inlier_ratio 0.25, max_num_trials 10 000). `--two-view trust` stores all MV matches as inliers. Table I's "front-end = features + matching + geometric verification" suggests the MV matches were verified too. |
| C4 | Minimum matches per pair | open | Pairs with fewer than 15 raw matches are not written (COLMAP's `min_num_inliers` default). |

## Pairwise geometric scoring

| # | Item | Paper status | Details |
|---|---|---|---|
| G1 | What is scored | open | "All metrics use the same correspondences and identical RANSAC settings across methods." We score each method's raw (pre-verification) matches. |
| G2 | Estimators and settings | confirmed | §II-C, §III-E: E (five-point) and H (DLT) with (LO-)RANSAC, max error 4.0, min inlier ratio 0.25, max trials 10 000. COLMAP defaults otherwise (confidence 0.9999, min trials 1000). |
| G3 | Normalised coordinates | confirmed | §II-C: "Points are expressed in normalized camera coordinates x̃ = K⁻¹x"; §III-E: "Sampson error is computed on normalized coordinates." The 4 px threshold is converted with the mean focal length (pycolmap does this for E; we do it explicitly for H). |
| G4 | Model choice | confirmed / open | "Select the model with the larger inlier set, breaking ties by lower median residual (favoring E unless H explains substantially more matches)." We apply the first rule literally. "Substantially" has no value, so no extra margin in favour of E is applied. |
| G5 | Sampson error | confirmed, both forms reported | The paper's SE(x, x′) = (x′ᵀEx)² / ((Ex)₁² + (Ex)₂² + (Eᵀx′)₁² + (Eᵀx′)₂²) is the *squared* Sampson error in normalised units; it is reported as "Median SE (normalised²)". We also report the Sampson *distance* √SE in pixels (× mean focal length), which is easier to interpret. For a winning homography, its Sampson (first-order geometric) error is used. The paper's Table III values (e.g. KITTI: MV 0.002, SIFT sequential 0.111) cannot be reproduced under any unit convention we tried: COLMAP accepts an E inlier only if its squared normalised Sampson error is below (4 px / f)², which is ≈ 3×10⁻⁵ on KITTI, so a median of 0.002 or 0.111 over inliers is impossible under that threshold. |
| G6 | Repeated runs | confirmed | §III-E: "We perform multiple independent runs with RANSAC and report median performance metrics over these repeated runs." The count is not given; `--repeats 3` (seeds 0, 1, 2) in the evaluation. |
| G7 | Aggregation | open | Per pair: medians over repeats. Across pairs: inlier ratio pooled (Σ inliers / Σ matches; per-pair mean in the JSON) and median of per-pair median Sampson errors. The paper's Tables II–IV take the median over sequences. |

## Metrics and timing

| # | Item | Paper status | Details |
|---|---|---|---|
| R1 | Pre-processing vs feature matching | confirmed / open | Table II splits "Pre-Processing (s)" from "Feature Matching (s)" without defining the stages. We count video encoding (MV) or SIFT extraction as pre-processing, and everything else before mapping as feature matching: MV decode + tracks + database (+ verification), or SIFT matching + verification. |
| R2 | CPU % | confirmed / open | Table II: "CPU is average during the pre-stages" (AOM 4.27 %, SIFT sequential 46.65 %, SIFT exhaustive 95.15 % on a 20-thread i7-12700K, so it must be a share of the whole machine). We report both (process + child CPU time) / wall time with 100 % = one core, and that value divided by the core count (4 here). |
| R3 | Matches per image | open | 2 × (total matches) / (number of images); raw and verified. |
| R4 | SIFT on CPU | environment | No GPU was available, so SIFT extraction and matching ran on the CPU (pycolmap 4.2.1, 8192 features max). |
| R5 | Mapper settings | open | Identical for all methods, as in the paper; the paper gives no values. KITTI is forward motion: with the default `init_max_forward_motion = 0.95`, no initial pair is ever accepted. For KITTI all methods use `init_max_forward_motion = 1.0` and `init_min_tri_angle = 4`. **Intrinsics:** the primary SfM runs keep the shared camera fixed at the KITTI calibration (`--fix-intrinsics`). With COLMAP's default refinement, all MV databases chose an adjacent-frame initial pair whose first bundle adjustment drove the SIMPLE_RADIAL k to 63–5 500 (SIFT: 0.01), so COLMAP rejected the camera and registered only 2/117 images. Both variants are reported. With the paper-faithful track defaults (seed all, cut) the collapse happens for the verified libaom database but not for the trusted-MV one (117/117) or SVT-AV1 (116/117), so it depends on which initial pair the mapper picks. On Intel Lunar Lake, the QSV database collapses too, and the VA-API database registers 117/117 with refinement (0.59 px). When MV reconstructions survive refinement, the camera stays plausible but the focal length comes out 2–4 % below the calibration (718.9 px): VA-API 701.3 (k 0.003), trusted libaom 693.0 (k −0.004), SVT-AV1 697.4 (k −0.007), against 714.0 / 713.7 (k 0.010) for SIFT. The cause is not established. |
| R6 | BA time | open | Table I reports "Time (front-end / BA)". pycolmap does not expose the BA share of incremental mapping, so we report total mapper wall time plus one final global BA. |
| R7 | 117-frame SfM demo | deviation (data unavailable) | The paper's clip is Paris Seq. 1 (117 frames, 1080×1920, iPhone 15 Pro; Fig. 3, and the follow-up's Fig. 1: 621 k MV points vs 48 k for SIFT exhaustive). It is not public; we use KITTI 00 frames 0–116 (1241×376, 4.4× fewer pixels). |

## Datasets

| # | Item | Paper status | Details |
|---|---|---|---|
| D1 | KITTI 00 | confirmed | "the first 230 frames of the Sequence 0 from KITTI Odometry" (the follow-up uses 231). `image_2`, downloaded member-by-member from the official archive (`eval/fetch_kitti.py`). |
| D2 | Gerrard Hall, Person Hall | confirmed / open | Paper §III-B: "we use a subset of each dataset. Our technique requires images to have the same dimensions and to be temporally adjacent. These two image sets were converted into videos"; ε = 1 (§III-E). Source: COLMAP 3.11.1 release assets (`gerrard-hall.zip`; `person-hall.zip` + `person-hall.z01`, a split archive). The subset is not specified; we take the longest run of consecutive images (file-name order = capture order) with the most common size and orientation: Gerrard Hall all 100 images (5616×3744); Person Hall IMG_1015–IMG_1229, 215 of 330 (the rest includes two blocks of 28 and 20 portrait shots). Images are resized to 1920×1280 (resolution not stated; comparable to the paper's 1080p clips; 21 MP frames are impractical to encode) and re-saved as JPEG q95, identical for all methods. The shared SIMPLE_RADIAL camera (f, cx, cy, k1) is scaled from each dataset's reference reconstruction (OPENCV model, f ≈ 3838 px at full size → ≈ 1312 px). `eval/prepare_colmap_dataset.py`, `eval/run_colmap_datasets.sh`. The libaom usage matters here (E2b). |
| D3 | Dublin Seq. 1, Paris Seq. 1 and 2 | not available | Custom iPhone recordings, not public. |
| D4 | AV1-3D-Reconstruction | unavailable | github.com/sigmedia/AV1-3D-Reconstruction was still not reachable, so no comparison with the authors' code was possible. |
