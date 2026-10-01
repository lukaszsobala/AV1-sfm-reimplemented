## kitti117 (117 frames)

| Method | Pre-processing (s) | Feature matching (s) | CPU % (1 core = 100) | CPU % of machine | Keypoints / img | Raw matches / img | Verified matches / img | Scored pairs | Inlier ratio (pooled) | Inlier ratio (median of pairs) | Median Sampson (px) | Median SE (normalised²) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AV1 MV, libaom (COLMAP verification) | 4.5 | 82.1 | 721 | 90.2 | 20,757 | 351,240 | 343,857 | 5,171 | 0.970 | – | 0.611 | 7.23e-07 |
| AV1 MV, Intel QSV (COLMAP verification) | 0.6 | 39.5 | 726 | 90.8 | 10,261 | 157,105 | 154,600 | 4,857 | 0.977 | – | 0.472 | 4.31e-07 |
| AV1 MV, VA-API (COLMAP verification) | 0.5 | 33.2 | 718 | 89.8 | 10,839 | 129,221 | 128,076 | 4,415 | 0.988 | – | 0.308 | 1.83e-07 |
| SIFT sequential (overlap 10) | 7.5 | 14.6 | 578 | 72.2 | 5,117 | 16,013 | 15,717 | 1,115 | 0.976 | 0.972 | 0.162 | 5.05e-08 |
| SIFT exhaustive | 7.7 | 87.7 | 763 | 95.3 | 5,117 | 23,891 | 21,633 | 6,701 | 0.905 | 0.542 | 0.247 | 1.19e-07 |
| SIFT sequential, exact matching (PyTorch) | 8.1 | 18.8 | 355 | 44.3 | 5,117 | 17,457 | 17,141 | 1,115 | 0.976 | 0.971 | 0.159 | 4.92e-08 |
| DISK + LightGlue sequential (overlap 10) | 18.2 | 1,036.5 | 180 | 22.5 | 4,623 | 47,805 | 47,752 | 1,115 | 0.993 | 0.993 | 0.345 | 2.30e-07 |

SfM, intrinsics fixed at calibration:

| Method | Registered | 3D points | Reproj. error (px) | Mean track length | Mapper wall (s) | Final global BA (s) |
|---|---:|---:|---:|---:|---:|---:|
| AV1 MV, libaom (COLMAP verification) | 117/117 | 201,749 | 0.893 | 10.49 | 759.6 | 22.0 |
| AV1 MV, Intel QSV (COLMAP verification) | 117/117 | 99,962 | 0.807 | 9.97 | 252.7 | 10.1 |
| AV1 MV, VA-API (COLMAP verification) | 117/117 | 114,328 | 0.598 | 8.75 | 188.5 | 7.4 |
| SIFT sequential (overlap 10) | 117/117 | 33,765 | 0.378 | 7.44 | 33.3 | 1.7 |
| SIFT exhaustive | 117/117 | 36,733 | 0.390 | 7.47 | 50.2 | 2.3 |
| SIFT sequential, exact matching (PyTorch) | 117/117 | 35,363 | 0.381 | 7.38 | 37.3 | 3.2 |
| DISK + LightGlue sequential (overlap 10) | 117/117 | 44,233 | 0.893 | 11.03 | 107.6 | 4.5 |

SfM, COLMAP default intrinsics refinement:

| Method | Registered | 3D points | Reproj. error (px) | Mean track length | Mapper wall (s) | Final global BA (s) |
|---|---:|---:|---:|---:|---:|---:|
| AV1 MV, libaom (COLMAP verification) | 2/117 | 29,395 | 0.352 | 2.00 | 12.1 | 5.6 |
| AV1 MV, Intel QSV (COLMAP verification) | 2/117 | 9,667 | 0.221 | 2.00 | 1.9 | 0.6 |
| AV1 MV, VA-API (COLMAP verification) | 117/117 | 115,251 | 0.594 | 8.72 | 257.2 | 21.7 |
| SIFT sequential (overlap 10) | 117/117 | 33,768 | 0.372 | 7.45 | 45.2 | 4.2 |
| SIFT exhaustive | 117/117 | 36,722 | 0.384 | 7.47 | 70.7 | 5.8 |
| SIFT sequential, exact matching (PyTorch) | 117/117 | 35,323 | 0.375 | 7.39 | 49.2 | 3.8 |
| DISK + LightGlue sequential (overlap 10) | 117/117 | 44,285 | 0.890 | 11.02 | 164.0 | 7.8 |
