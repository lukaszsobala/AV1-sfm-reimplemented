# Vendored from sigmedia/AV1-Optical-Flow

Source: https://github.com/sigmedia/AV1-Optical-Flow
Commit: a93279609639a308b598b94aff00d88eed4d6489 (2026-06-22)
License: GNU Affero General Public License v3.0 (see `/LICENSE`). Upstream states
version 3.0 without "or later", so these files are AGPL-3.0 only.
Copyright © 2026 Sigmedia.tv / Julien Zouein (zoueinj@tcd.ie)

Official code for "AV1 Motion Vector Fidelity and Application for Efficient
Optical Flow" (Zouein, Vibhoothi, Kokaram, PCS 2025).

| File here | Upstream path | Modified? |
|---|---|---|
| `third_party/av1of/dav1d-inspection.patch` | `patches/dav1d-inspection.patch` | no |
| `third_party/av1of/av1of_inspect_shim.c` | `src/av1of_inspect_shim.c` | no |
| `src/av1sfm/_vendor/dav1d_inspect.py` | `src/modules/dav1d_inspect.py` | yes: library search path (marked `MODIFIED` in the file) |

The patch applies to dav1d commit `14c73c7db38eebfd3202146b76a1ad4df90dd3a2`
(`1.5.3-58-g14c73c7d`); `setup.sh` fetches that commit (from code.videolan.org,
or its GitHub mirror) and applies the patch.
