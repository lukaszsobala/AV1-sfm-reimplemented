# Third-party material and licences

av1sfm's own code is licensed under the GNU Affero General Public License,
version 3 or (at your option) any later version ([LICENSE](LICENSE)). The
repository also contains, or downloads at run time, the third-party material
below. Every licence listed allows use in an AGPL-3.0 work.

## In this repository

| Material | Where | Licence | Notes |
|---|---|---|---|
| Motion-vector extraction layer from [sigmedia/AV1-Optical-Flow](https://github.com/sigmedia/AV1-Optical-Flow) (© 2026 Sigmedia.tv / Julien Zouein) | `src/av1sfm/_vendor/`, `third_party/av1of/` | AGPL-3.0 | Upstream states version 3.0 without "or later", so these files are AGPL-3.0 only. One file is modified, with a dated notice. Details: [third_party/av1of/NOTICE.md](third_party/av1of/NOTICE.md). |
| Code adapted from kornia's LightGlue (© 2018 Kornia Team), itself a port of [cvg/LightGlue](https://github.com/cvg/LightGlue) (© 2023 ETH Zurich) | Marked functions in `src/av1sfm/learned.py` | Apache-2.0 ([LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)) | The file's docstring lists the adapted functions and the changes made. Apache-2.0 code may be included in an AGPL-3.0 work. |
| Python port of COLMAP's brute-force matching loop (© ETH Zurich and UNC Chapel Hill) | `colmap_one_way` in `tests/test_sift_exact.py` | BSD-3-Clause ([LICENSES/BSD-3-Clause-COLMAP.txt](LICENSES/BSD-3-Clause-COLMAP.txt)) | Used only as a test reference. |
| Patch to dav1d (from AV1-Optical-Flow) | `third_party/av1of/dav1d-inspection.patch` | AGPL-3.0; dav1d itself is BSD-2-Clause | `setup.sh` downloads dav1d and applies the patch locally. |

Taken together, the repository can be distributed under AGPL-3.0. The
"or later" option applies to av1sfm's own files only.

## Downloaded or installed at run time (not distributed here)

| Component | Licence |
|---|---|
| numpy, pycolmap / COLMAP, PyTorch | BSD-3-Clause (numpy also bundles 0BSD, MIT, Zlib and CC0 parts) |
| OpenCV (`opencv-python-headless`), kornia | Apache-2.0 |
| remotezip, libvpl | MIT |
| dav1d, libaom | BSD-2-Clause (libaom also carries the AOM patent licence) |
| SVT-AV1 | BSD-3-Clause-Clear, with the AOM patent licence |
| FFmpeg, as built by `scripts/build_ffmpeg.sh` | LGPL-2.1-or-later (built without `--enable-gpl` or `--enable-nonfree`) |
| LightGlue `disk_lightglue` weights | Apache-2.0 (cvg/LightGlue) |
| DISK `depth` weights | The DISK repository has no licence file. LightGlue's README states that DISK follows Apache-2.0. Check with the DISK authors before commercial use. |

## Datasets

The evaluation scripts download the datasets from their original sources;
none of their images are in this repository. The `results/` files contain
statistics, and `camera.txt` holds four calibration numbers derived from each
dataset's reference model.

- **KITTI odometry** is licensed under CC BY-NC-SA 3.0, which does not allow
  commercial use.
- **Gerrard Hall and Person Hall** come from the COLMAP project's release
  assets, which state no separate licence. Check with the COLMAP project
  before publishing images or models made from them.

## The paper

av1sfm is an independent implementation of Zouein et al. (arXiv:2510.17434),
written from the paper and its follow-up. No code from the authors'
AV1-3D-Reconstruction repository was used. ASSUMPTIONS.md quotes short
passages of both papers, with section references.
