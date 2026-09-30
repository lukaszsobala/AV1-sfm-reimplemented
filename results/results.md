## kitti117 (117 frames)

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

## kitti230 (230 frames)

| Method | Pre-stage wall (s) | Avg CPU % | Encode (s) | Keypoints / img | Raw matches / img | Verified matches / img | Scored pairs | Inlier ratio | Median Sampson (px) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AV1 MV, libaom (COLMAP verification) | 249.7 | 371 | 16.8 | 11,864 | 234,516 | 229,065 | 10,823 | 0.970 | 0.560 |
| AV1 MV, libaom (MVs trusted, no verification) | 22.6 | 104 | – | 11,864 | 234,516 | 234,516 | – | – | – |
| AV1 MV, SVT-AV1 (COLMAP verification) | 20.2 | 311 | 3.6 | 2,821 | 12,796 | 12,545 | 2,025 | 0.975 | 0.495 |
| SIFT sequential (overlap 10) | 109.5 | 347 | – | 4,831 | 15,380 | 15,085 | 2,245 | 0.977 | 0.154 |
| SIFT exhaustive | 874.6 | 396 | – | 4,831 | 23,966 | 20,030 | 23,712 | 0.868 | 0.247 |

