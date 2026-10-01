## gerrard-hall (100 frames)

| Method | Pre-processing (s) | Feature matching (s) | CPU % (1 core = 100) | CPU % of machine | Keypoints / img | Raw matches / img | Verified matches / img | Scored pairs | Inlier ratio (pooled) | Inlier ratio (median of pairs) | Median Sampson (px) | Median SE (normalised²) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AV1 MV, libaom (COLMAP verification) | 12.9 | 122.3 | 363 | 90.8 | 10,669 | 53,407 | 37,415 | 1,264 | 0.684 | 0.414 | 0.994 | 5.76e-07 |
| AV1 MV, libaom (MVs trusted, no verification) | – | 10.2 | 117 | 29.3 | 10,669 | 53,407 | 53,407 | – | – | – | – | – |
| AV1 MV, libaom good usage (COLMAP verification) | 83.0 | 40.3 | 306 | 76.4 | 11,078 | 62,083 | 56,782 | 1,053 | 0.910 | 0.728 | 0.486 | 1.37e-07 |
| AV1 MV, SVT-AV1 (COLMAP verification) | 6.4 | 27.2 | 302 | 75.4 | 1,155 | 2,689 | 1,367 | 512 | 0.480 | 0.442 | 0.846 | 4.15e-07 |
| SIFT sequential (overlap 10) | 108.3 | 78.6 | 350 | 87.5 | 12,862 | 23,498 | 23,015 | 902 | 0.970 | 0.952 | 0.314 | 5.71e-08 |
| SIFT exhaustive | 108.6 | 309.3 | 395 | 98.7 | 12,862 | 26,476 | 24,886 | 3,657 | 0.943 | 0.421 | 0.284 | 4.74e-08 |

## kitti117 (117 frames)

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

## kitti230 (230 frames)

| Method | Pre-processing (s) | Feature matching (s) | CPU % (1 core = 100) | CPU % of machine | Keypoints / img | Raw matches / img | Verified matches / img | Scored pairs | Inlier ratio (pooled) | Inlier ratio (median of pairs) | Median Sampson (px) | Median SE (normalised²) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AV1 MV, libaom (COLMAP verification) | 17.2 | 482.5 | 371 | 92.8 | 21,710 | 433,433 | 422,965 | 11,306 | 0.971 | 0.976 | 0.599 | 6.95e-07 |
| AV1 MV, libaom (MVs trusted, no verification) | – | 39.5 | 103 | 25.6 | 21,710 | 433,433 | 433,433 | – | – | – | – | – |
| AV1 MV, SVT-AV1 (COLMAP verification) | 3.8 | 13.4 | 278 | 69.4 | 1,783 | 6,923 | 6,654 | 2,057 | 0.952 | 0.949 | 0.543 | 5.72e-07 |
| SIFT sequential (overlap 10) | 46.8 | 62.6 | 347 | 86.6 | 4,831 | 15,380 | 15,085 | 2,245 | 0.977 | 0.972 | 0.154 | 4.60e-08 |
| SIFT exhaustive | 48.9 | 825.6 | 396 | 99.1 | 4,831 | 23,966 | 20,030 | 23,712 | 0.868 | 0.444 | 0.248 | 1.21e-07 |

## person-hall (215 frames)

| Method | Pre-processing (s) | Feature matching (s) | CPU % (1 core = 100) | CPU % of machine | Keypoints / img | Raw matches / img | Verified matches / img | Scored pairs | Inlier ratio (pooled) | Inlier ratio (median of pairs) | Median Sampson (px) | Median SE (normalised²) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AV1 MV, libaom (COLMAP verification) | 32.6 | 359.2 | 367 | 91.6 | 12,232 | 53,562 | 37,448 | 3,794 | 0.674 | 0.400 | 0.966 | 5.43e-07 |
| AV1 MV, libaom (MVs trusted, no verification) | – | 23.7 | 116 | 29.0 | 12,232 | 53,562 | 53,562 | – | – | – | – | – |
| AV1 MV, libaom good usage (COLMAP verification) | 193.5 | 150.0 | 312 | 78.1 | 15,957 | 117,766 | 111,640 | 3,193 | 0.941 | 0.722 | 0.457 | 1.21e-07 |
| AV1 MV, SVT-AV1 (COLMAP verification) | 14.9 | 114.1 | 338 | 84.5 | 1,989 | 5,526 | 2,391 | 1,409 | 0.407 | 0.367 | 0.910 | 4.82e-07 |
| SIFT sequential (overlap 10) | 278.5 | 205.9 | 355 | 88.7 | 14,076 | 49,278 | 48,763 | 2,014 | 0.981 | 0.977 | 0.344 | 6.87e-08 |
| SIFT exhaustive | 284.2 | 1,473.5 | 387 | 96.8 | 14,076 | 72,356 | 70,400 | 10,045 | 0.971 | 0.467 | 0.290 | 4.92e-08 |

