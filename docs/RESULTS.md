# Evaluation and results

How the evaluation is run, and every measured result. The
[README](../README.md) summarises them; [ASSUMPTIONS.md](../ASSUMPTIONS.md)
explains each setting.

## Reproducing the evaluation

```bash
bash eval/run_kitti.sh              # fetches KITTI 00 frames 0-229, runs everything, writes runs/results.md
bash eval/run_colmap_datasets.sh    # fetches Gerrard Hall and Person Hall, pairwise metrics (eps = 1)
SFM=1 bash eval/run_colmap_datasets.sh   # ... plus incremental mapping
```

Hardware encoders are added as extra methods (`mv_qsv`, `mv_vaapi`,
`mv_vulkan`), and `TORCH=1` adds the PyTorch baselines (`sift_seq_exact`,
`sift_exh_exact`, `disk_seq`, `disk_exh`; [USAGE.md](USAGE.md#gpu-matchers-pytorch-intel-gpu-nvidia-gpu-or-cpu)).
`ONLY` restricts the run to some methods, `SETS` to some frame counts, and
`RUN=` drops the `uv run` prefix inside an active virtualenv:

```bash
RUN= HW="qsv vaapi" ONLY="mv mv_qsv mv_vaapi" SETS=117 bash eval/run_kitti.sh
RUN= TORCH=1 ONLY="sift_seq_exact disk_seq" SETS=117 bash eval/run_kitti.sh
```

Gerrard Hall and Person Hall come from the COLMAP release assets
(`https://github.com/colmap/colmap/releases/download/3.11.1/gerrard-hall.zip`,
`person-hall.zip` plus `person-hall.z01`). As in the paper, a subset of
same-size, temporally adjacent images is used (Gerrard Hall: all 100; Person
Hall: IMG_1015–IMG_1229, 215 images), resized to 1920×1280, with the cosine
filter disabled (ε = 1). See ASSUMPTIONS.md D2. On these datasets libaom
also runs in its default `good` usage (`mv_good`, ASSUMPTIONS.md E2b).

`eval/run_kitti.sh` holds every command behind the tables below. Steps whose
output already exists are skipped. For each set (117 and 230 frames) it runs:
the MV pipeline on libaom (with COLMAP verification and with trusted MVs) and
on SVT-AV1, COLMAP SIFT
with sequential (overlap 10) and exhaustive matching, identical pairwise
scoring of every method's raw matches, and, for the 117-frame set, incremental
mapping with identical settings. `eval/make_table.py runs` turns the run
summaries into the tables below.

## Results on a cloud VM (CPU only)

Produced by `bash eval/run_kitti.sh` and `bash eval/run_colmap_datasets.sh`
on a 4-core x86 cloud VM **without a GPU** (SIFT extraction and matching on
the CPU), with the paper-faithful track defaults (every block seeds a track,
tracks are cut at the first cosine violation, ε = 0.1 on KITTI and ε = 1 on
the COLMAP datasets). Software: FFmpeg n9.0.2, libaom 3.8.2, SVT-AV1 4.2.0,
pycolmap 4.2.1, Python 3.14.7. The same tables are in
[`results/results.md`](../results/results.md); per-run JSON summaries are in
[`results/`](../results/).

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

## Results on an Intel Lunar Lake laptop (KITTI 00, frames 0–116)

Run by the repository owner on an Intel Lunar Lake laptop (Ubuntu 26.04,
Intel graphics PPA, system FFmpeg, power profile "Balanced", so not peak
performance) with
`RUN= HW="qsv vaapi" ONLY="mv mv_qsv mv_vaapi" SETS=117 bash eval/run_kitti.sh`,
`RUN= ONLY="sift_seq sift_exh" SETS=117 bash eval/run_kitti.sh` and
`RUN= TORCH=1 ONLY="sift_seq_exact disk_seq" SETS=117 bash eval/run_kitti.sh`
(PyTorch 2.14 XPU on the integrated Arc GPU). The MV rows were scored
before the median-of-pairs statistic was added, hence the dashes.
Vulkan Video AV1 encode was not exposed by the driver (ASSUMPTIONS.md E5b).
Sources: [`results/lunar-lake/kitti117_torch.md`](../results/lunar-lake/kitti117_torch.md)
(these tables), [`kitti117_torch_fp32.md`](../results/lunar-lake/kitti117_torch_fp32.md)
(DISK + LightGlue before the speed-ups: float32 attention, no pruning) and
[`kitti117.md`](../results/lunar-lake/kitti117.md) (an earlier SIFT run).

| Method | Pre-processing (s) | Feature matching (s) | CPU % (1 core = 100) | CPU % of machine | Keypoints / img | Raw matches / img | Verified matches / img | Scored pairs | Inlier ratio (pooled) | Inlier ratio (median of pairs) | Median Sampson (px) | Median SE (normalised²) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AV1 MV, libaom (COLMAP verification) | 4.5 | 82.1 | 721 | 90.2 | 20,757 | 351,240 | 343,857 | 5,171 | 0.970 | – | 0.611 | 7.23e-07 |
| AV1 MV, Intel QSV (COLMAP verification) | 0.6 | 39.5 | 726 | 90.8 | 10,261 | 157,105 | 154,600 | 4,857 | 0.977 | – | 0.472 | 4.31e-07 |
| AV1 MV, VA-API (COLMAP verification) | 0.5 | 33.2 | 718 | 89.8 | 10,839 | 129,221 | 128,076 | 4,415 | 0.988 | – | 0.308 | 1.83e-07 |
| SIFT sequential (overlap 10) | 7.5 | 14.6 | 578 | 72.2 | 5,117 | 16,013 | 15,717 | 1,115 | 0.976 | 0.972 | 0.162 | 5.05e-08 |
| SIFT exhaustive | 7.7 | 87.7 | 763 | 95.3 | 5,117 | 23,891 | 21,633 | 6,701 | 0.905 | 0.542 | 0.247 | 1.19e-07 |
| SIFT sequential, exact matching (PyTorch) | 7.6 | 16.1 | 385 | 48.1 | 5,117 | 17,457 | 17,141 | 1,115 | 0.976 | 0.971 | 0.159 | 4.92e-08 |
| DISK + LightGlue sequential (overlap 10) | 18.7 | 172.7 | 134 | 16.8 | 4,623 | 47,778 | 47,724 | 1,115 | 0.993 | 0.993 | 0.345 | 2.30e-07 |

SfM, intrinsics fixed at calibration:

| Method | Registered | 3D points | Reproj. error (px) | Mean track length | Mapper wall (s) | Final global BA (s) |
|---|---:|---:|---:|---:|---:|---:|
| AV1 MV, libaom (COLMAP verification) | 117/117 | 201,749 | 0.893 | 10.49 | 759.6 | 22.0 |
| AV1 MV, Intel QSV (COLMAP verification) | 117/117 | 99,962 | 0.807 | 9.97 | 252.7 | 10.1 |
| AV1 MV, VA-API (COLMAP verification) | 117/117 | 114,328 | 0.598 | 8.75 | 188.5 | 7.4 |
| SIFT sequential (overlap 10) | 117/117 | 33,765 | 0.378 | 7.44 | 33.3 | 1.7 |
| SIFT exhaustive | 117/117 | 36,733 | 0.390 | 7.47 | 50.2 | 2.3 |
| SIFT sequential, exact matching (PyTorch) | 117/117 | 35,350 | 0.381 | 7.38 | 34.3 | 1.9 |
| DISK + LightGlue sequential (overlap 10) | 117/117 | 44,202 | 0.893 | 11.04 | 111.9 | 8.4 |

SfM, COLMAP default intrinsics refinement:

| Method | Registered | 3D points | Reproj. error (px) | Mean track length | Mapper wall (s) | Final global BA (s) |
|---|---:|---:|---:|---:|---:|---:|
| AV1 MV, libaom (COLMAP verification) | 2/117 | 29,395 | 0.352 | 2.00 | 12.1 | 5.6 |
| AV1 MV, Intel QSV (COLMAP verification) | 2/117 | 9,667 | 0.221 | 2.00 | 1.9 | 0.6 |
| AV1 MV, VA-API (COLMAP verification) | 117/117 | 115,251 | 0.594 | 8.72 | 257.2 | 21.7 |
| SIFT sequential (overlap 10) | 117/117 | 33,768 | 0.372 | 7.45 | 45.2 | 4.2 |
| SIFT exhaustive | 117/117 | 36,722 | 0.384 | 7.47 | 70.7 | 5.8 |
| SIFT sequential, exact matching (PyTorch) | 117/117 | 35,323 | 0.375 | 7.39 | 48.0 | 2.9 |
| DISK + LightGlue sequential (overlap 10) | 117/117 | 44,244 | 0.890 | 11.03 | 173.3 | 7.0 |

## Speed: where the time goes and the mapper defaults (Lunar Lake, Performance profile)

Measured on the Lunar Lake laptop in the "Performance" power profile, on the
databases of the table above (KITTI 00, frames 0–116, intrinsics fixed at
the calibration) and on exact SIFT for frames 0–229. Pose accuracy is
measured against the KITTI ground truth with `eval/pose_error.py`: absolute
trajectory error (ATE) of the camera centres after a similarity alignment,
and relative pose error (RPE) over 1 and 10 frames. The ground-truth path is
90.9 m long for 117 frames and 160.0 m for 230.

**Where the time goes.** Up to the mapper, COLMAP's geometric verification
dominates MV matching (VA-API: 21.9 s of about 24 s; 31 s in the Balanced
profile); encoding, MV extraction, tracks and database writing take about
2 s together. In the mapper, the log line "Retriangulation and Global bundle
adjustment" is almost entirely bundle adjustment: retriangulation itself
takes well under 1 % of the mapper's time on both kinds of database. With
COLMAP's settings plus pruning, the exact-SIFT mapper (31 s) spends 41 % in
global and 31 % in local bundle adjustment and 16 % completing and merging
tracks; the VA-API mapper (145 s) 40 % in local and 19 % in global bundle
adjustment, 25 % completing and merging tracks and 8 % triangulating new
images. Most local bundle adjustments of the SIFT database have fewer than
50,000 residuals and run single-threaded; letting Ceres use all threads for
them (`ba_min_num_residuals_for_cpu_multi_threading`) was slower.

**Mapper defaults.** `eval/run_mapper.py` and `av1sfm reconstruct` change two
COLMAP settings that cost time without improving the reconstruction
(av1sfm.mapping): global bundle adjustments skip redundant 3D points
(`--prune-redundant-points`, COLMAP's `ba_global_ignore_redundant_points3D`),
and each registered image gets one local bundle adjustment instead of up to
two (`--ba-local-refinements 1`). The tables above were made with COLMAP's
settings (`--no-prune-redundant-points --ba-local-refinements 2`).

| Database | Mapper settings | Mapper wall (s) | 3D points | Reproj. error (px) | ATE RMSE (m) | RPE 1 frame (m) | RPE 10 frames (m) | RPE 10 frames (°) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| exact SIFT, 117 frames | COLMAP's | 31.1 | 35,350 | 0.381 | 0.202 | 0.0384 | 0.240 | 0.287 |
| | pruning | 26.7 | 35,347 | 0.382 | 0.205 | 0.0384 | 0.241 | 0.288 |
| | pruning, 1 local refinement (**default**) | 22.3 | 35,303 | 0.381 | 0.205 | 0.0384 | 0.241 | 0.288 |
| | default, `--ba-global-ratio 1.2` | 20.3 | 35,303 | 0.381 | 0.209 | 0.0385 | 0.242 | 0.288 |
| exact SIFT, 230 frames | COLMAP's | 67.1 | 59,302 | 0.392 | 1.301 | 0.0493 | 0.412 | 0.260 |
| | pruning | 58.5 | 59,298 | 0.394 | 1.291 | 0.0491 | 0.410 | 0.262 |
| | pruning, 1 local refinement (**default**) | 51.0 | 59,185 | 0.393 | 1.282 | 0.0489 | 0.408 | 0.262 |
| | default, `--ba-global-ratio 1.2` | 41.9 | 59,182 | 0.392 | 1.272 | 0.0488 | 0.406 | 0.261 |
| VA-API, 117 frames | COLMAP's | 177.3 | 114,328 | 0.598 | 0.250 | 0.0400 | 0.259 | 0.278 |
| | pruning | 152.4 | 114,246 | 0.600 | 0.234 | 0.0391 | 0.251 | 0.282 |
| | pruning, 1 local refinement (**default**) | 111.4 | 113,842 | 0.597 | 0.237 | 0.0392 | 0.252 | 0.282 |
| | default, `--ba-global-ratio 1.2` | 98.9 | 113,878 | 0.597 | 0.237 | 0.0392 | 0.252 | 0.283 |
| libaom, 117 frames | COLMAP's | 731.5 | 201,751 | 0.893 | 0.225 | 0.0392 | 0.250 | 0.300 |
| | pruning | 471.6 | 201,499 | 0.893 | 0.180 | 0.0370 | 0.233 | 0.292 |
| | pruning, 1 local refinement (**default**) | 412.3 | 201,024 | 0.890 | 0.170 | 0.0365 | 0.229 | 0.290 |

- Every setting registers all images. The defaults make the mapper 24–44 %
  faster than COLMAP's settings (exact SIFT 117 frames: 31.1 → 22.3 s;
  libaom: 732 → 412 s) with 0.1–0.4 % fewer points, a reprojection error
  within 0.001 px, and pose errors at most 1.5 % higher (exact SIFT, 117
  frames) and lower on VA-API (trajectory error −5 %) and libaom (−24 %).
  Any small change to the
  optimisation moves the pose errors by about that much in either direction
  (pruning alone: −20 % to +1.5 %); repeating a run with the same settings
  moves them by about 0.1 %.
- `--ba-global-ratio 1.2` (COLMAP's `ba_global_frames_ratio` and
  `ba_global_points_ratio`, 1.1) runs global bundle adjustments less often:
  another 9–18 % faster. The trajectory error was 2 % higher on exact SIFT
  (117 frames), unchanged on VA-API and 1 % lower on exact SIFT (230 frames).
  It is not a default. 1.4 was 16 %
  faster than the default on exact SIFT, with a 2.5 % higher trajectory error.
- Each global refinement (bundle adjustment, track completion, filtering,
  repeated up to `ba_global_max_refinements = 5` times) stops after one or
  two rounds, so lowering that limit to 2 saved nothing (26.5 s).

**Global mapper.** COLMAP's global mapper (GLOMAP, `--mapper global`) is
faster, but not like for like (same databases, COLMAP's incremental settings
for comparison):

| Database | Mapper | Mapper wall (s) | 3D points | Reproj. error (px) | ATE RMSE (m) | RPE 1 frame (m) | RPE 10 frames (m) | RPE 10 frames (°) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| VA-API | global, all tracks | 161.6 | 110,864 | 0.608 | 0.266 | 0.0407 | 0.265 | 0.277 |
| VA-API | global, 1000 tracks / image | 58.5 | 110,389 | 0.601 | 0.268 | 0.0407 | 0.266 | 0.277 |
| libaom | global, all tracks | 246.6 | 176,709 | 0.918 | 0.247 | 0.0404 | 0.260 | 0.283 |
| libaom | global, 1000 tracks / image | 95.9 | 172,218 | 0.888 | 0.824 | 0.0654 | 0.534 | 0.329 |

It gives 3 % (VA-API) to 12 % (libaom) fewer points and slightly worse
poses; positioning the cameras with 1000 tracks per image
(`--global-tracks-per-view`) is harmless on the clean VA-API matches, but on
libaom's noisier matches the trajectory error grows 3.7×.

**Exact SIFT matching.** On the GPU, the exact matcher keeps several pairs in
flight, so that the GPU computes while the CPU checks exactness and applies
COLMAP's tests, and takes the column-wise top two by masking the maximum
instead of a strided `topk`: the 1,115 sequential pairs take 9.3 s instead of
12.3 s, with the same 1,021,246 matches pair for pair. With COLMAP's
verification (2.7 s) and SIFT extraction (7.4 s), the exact-SIFT front end
takes about 20 s, and with the default mapper about 42 s up to a
reconstruction (`av1sfm reconstruct --matcher sift`).

**Copied AV1 videos.** On KITTI frames 0–39 encoded by libaom in its default
good-quality mode (random access: hidden alt-refs, 41 % of MVs to later
frames), copying the stream (`av1sfm reconstruct video.mp4`) gives a model in
8.4 s with 16 k points and 1.34 m trajectory error over 31 m; re-encoding the
frames with low-delay libaom gives 72 k points and 0.22 m, in 124 s (ASSUMPTIONS.md M9).

## Findings

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
  encode + 33 s matching with COLMAP verification), SIFT sequential 22 s
  (7.5 + 14.6 s) and SIFT exhaustive 95 s (7.7 + 87.7 s). VA-API's encode is
  15× cheaper than SIFT extraction, but verifying its 129 k raw matches per
  image dominates. With the mapper, VA-API takes 222 s for 114 k points at
  0.60 px; SIFT sequential 55 s for 34 k points at 0.38 px. MV matching buys
  density, not end-to-end speed, unless verification is skipped (trusted MVs).
- **Exact SIFT matching on the GPU (Lunar Lake).** Matching plus verification
  takes 16.1 s, against 14.6 s for COLMAP's approximate CPU matcher, at
  two thirds of the CPU use (385 % vs 578 %). The matches are those of COLMAP's
  exact brute-force matcher (17,457 raw per image, as in the cloud run),
  which on the cloud VM's CPU took 1,348 s against 33 s for the approximate
  one (ASSUMPTIONS.md R4). Exact matching gives 9 % more raw matches than
  the approximate matcher, and 35.4 k points instead of 33.8 k at the same
  0.38 px reprojection error.
- **DISK + LightGlue (Lunar Lake GPU)** is the slowest front end: 191 s
  (19 s extraction, 173 s matching and verification for 1,115 pairs). That is
  6× VA-API's 34 s, 2.2× libaom's 87 s and 9× SIFT sequential's 22 s; the
  paper's Table I has it at 9× its libaom MV front end.
  Its raw matches have the highest inlier ratio (0.993), but they are less
  precise: median Sampson error 0.35 px (SIFT 0.16 px, VA-API 0.31 px) and
  0.89 px reprojection error (SIFT 0.38 px, VA-API 0.60 px). It gives 44 k
  points (SIFT 34–37 k, VA-API 114 k) and the longest tracks (11.0 images).
  This is the pattern of the paper's Table I, which has DISK + LightGlue on a
  T4 at 1,067 s front end and 85 k points at 1.07 px, against SIFT
  sequential's 55 k points at 0.30 px (on its own 1080×1920 clip).
  With float32 attention, no point pruning and the dual softmax on the CPU,
  matching took 1,037 s instead of 173 s, for practically the same result
  (47,805 against 47,778 raw matches per image; 44,233 against 44,202 points,
  both at 0.89 px; ASSUMPTIONS.md R8).
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
