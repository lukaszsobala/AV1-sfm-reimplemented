## kitti117 (117 frames)

| Method | Pre-processing (s) | Feature matching (s) | CPU % (1 core = 100) | CPU % of machine | Keypoints / img | Raw matches / img | Verified matches / img | Scored pairs | Inlier ratio | Median Sampson (px) | Median SE (normalised²) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AV1 MV, libaom (COLMAP verification) | 4.5 | 82.1 | 721 | 90.2 | 20,757 | 351,240 | 343,857 | 5,171 | 0.970 | 0.611 | 7.23e-07 |
| AV1 MV, Intel QSV (COLMAP verification) | 0.6 | 39.5 | 726 | 90.8 | 10,261 | 157,105 | 154,600 | 4,857 | 0.977 | 0.472 | 4.31e-07 |
| AV1 MV, VA-API (COLMAP verification) | 0.5 | 33.2 | 718 | 89.8 | 10,839 | 129,221 | 128,076 | 4,415 | 0.988 | 0.308 | 1.83e-07 |

SfM, intrinsics fixed at calibration:

| Method | Registered | 3D points | Reproj. error (px) | Mean track length | Mapper wall (s) | Final global BA (s) |
|---|---:|---:|---:|---:|---:|---:|
| AV1 MV, libaom (COLMAP verification) | 117/117 | 201,749 | 0.893 | 10.49 | 759.6 | 22.0 |
| AV1 MV, Intel QSV (COLMAP verification) | 117/117 | 99,962 | 0.807 | 9.97 | 252.7 | 10.1 |
| AV1 MV, VA-API (COLMAP verification) | 117/117 | 114,328 | 0.598 | 8.75 | 188.5 | 7.4 |

SfM, COLMAP default intrinsics refinement:

| Method | Registered | 3D points | Reproj. error (px) | Mean track length | Mapper wall (s) | Final global BA (s) |
|---|---:|---:|---:|---:|---:|---:|
| AV1 MV, libaom (COLMAP verification) | 2/117 | 29,395 | 0.352 | 2.00 | 12.1 | 5.6 |
| AV1 MV, Intel QSV (COLMAP verification) | 2/117 | 9,667 | 0.221 | 2.00 | 1.9 | 0.6 |
| AV1 MV, VA-API (COLMAP verification) | 117/117 | 115,251 | 0.594 | 8.72 | 257.2 | 21.7 |

