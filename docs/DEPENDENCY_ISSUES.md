# Dependency issues and workarounds

Problems found in the libraries and drivers av1sfm depends on while building
and evaluating it, what they cause, and how av1sfm deals with them. Bugs come
first. Behaviours that are documented or intended upstream, but change
results in ways that are easy to miss, follow them.

| # | Component | Problem | Kind | av1sfm |
|---|---|---|---|---|
| 1 | PyTorch XPU 2.14.1 | `nonzero` and boolean-mask indexing on large tensors return too few elements | bug | Index lists computed on the CPU |
| 2 | PyTorch XPU 2.14.1 | `softmax` and `log_softmax` over more than 4096 elements are wrong | bug | Fused attention kernel; `x - logsumexp(x)` |
| 3 | kornia 0.8.3 | `LightGlue` raises `KeyError: 'xpu'` on any device but CPU, MPS and CUDA | bug | Own forward pass over kornia's layers |
| 4 | kornia 0.8.3 | LightGlue's fast attention path is CUDA-only | limitation | Same kernels on every device |
| 5 | kornia 0.8.3 | `LightGlue` prints to stdout | nuisance | Statistics written to a file |
| 6 | COLMAP 4.2.1 | Default SIFT matcher is approximate; result depends on thread count and run | behaviour | GPU exact matcher; documented |
| 7 | COLMAP 4.2.1 | Geometric verification is not deterministically seeded | behaviour | Documented |
| 8 | COLMAP 4.2.1 | Default mapper settings fail on forward motion; intrinsics refinement collapses MV reconstructions | behaviour | Settings in ASSUMPTIONS.md R5 |
| 9 | FFmpeg `av1_vaapi` | `-q:v` is not the AV1 qindex | pitfall | `-global_quality` |
| 10 | Mesa ANV (Lunar Lake) | Vulkan Video AV1 encode not exposed | driver limitation | VA-API or QSV |
| 11 | libaom, SVT-AV1 | Low-delay settings still use references other than the previous frame | behaviour | Each MV follows its real reference |

## PyTorch on Intel GPUs (XPU)

**Environment:** PyTorch 2.14.1 XPU wheel (`download.pytorch.org/whl/xpu`),
Python 3.14, Intel Core Ultra (Lunar Lake) with integrated Arc graphics
(`torch.xpu.get_device_name()`: "Intel(R) Arc(TM) Graphics"), Ubuntu 26.04,
GPU compute runtime from Intel's graphics PPA. `torch.xpu.get_device_properties()`:
platform "Intel(R) oneAPI Unified Runtime over Level-Zero V2", driver version
1.14.37020, device version 20.4.4, 64 EUs.

`eval/check_device.py` found these by running every LightGlue module and the
operations inside them on the GPU and on the CPU with identical inputs.
`eval/xpu_repro.py` reproduces both with PyTorch alone and random inputs,
and prints the driver version, for bug reports. They have not been reported
upstream yet; the place is
[intel/torch-xpu-ops](https://github.com/intel/torch-xpu-ops/issues). Basic
operations failing this plainly suggests a problem specific to this GPU or
driver version rather than to PyTorch in general, but that is not established.

Operations that were **correct** on the same machine (maximum relative
difference to the CPU): `scaled_dot_product_attention` (3·10⁻⁶ in float32,
4·10⁻⁴ in float16), `Linear`, `LayerNorm`, `GELU`, convolutions (DISK's score
map), `logsumexp`, `max` and `sum` over 5000 elements, transposed copies,
and matrix products summing over 5000 elements (≤ 4·10⁻⁶).

### 1. `nonzero` and boolean-mask indexing on large tensors return too few elements

On the GPU, `mask.nonzero()` and `x[mask]` on large tensors return far fewer
elements than the mask contains, while `mask.sum()` on the same mask is
correct. A random 1-D mask is handled correctly with 4,096 and 65,536 elements
(155 and 2,625 true), but with 479,232 elements `nonzero` returns 78 indices
for 19,051 true entries.
`eval/xpu_repro.py` builds the mask that kornia's DISK keypoint selection uses
(non-maximum suppression on a 384×1248 = 479,232-pixel score map) from random
data:

| Seed | `mask.sum()` (CPU and GPU) | `nonzero()` and `x[mask]` on the GPU |
|---|---:|---:|
| 0 | 19,195 | 148 |
| 1 | 19,211 | 136 |
| 2 | 19,280 | 112 |

The two wrong results need not even agree with each other. In kornia's DISK,
which uses both on the same mask, they did not:

```
kornia/feature/disk/detector.py, line 49, in heatmap_to_keypoints
    xy = xy[mask]
IndexError: The shape of the mask [1312] at index 0 does not match the shape of the indexed tensor [2166, 2] at index 0
```

In LightGlue's match selection, indices built this way triggered device-side
asserts, followed by a lost device and an abort at exit:

```
torch-xpu-ops/src/ATen/native/xpu/sycl/Indexing.h:622: operator(): global id: [185,0,0], local id: [185,0,0]
Assertion `index >= -sizes_[i] && index < sizes_[i] && "index out of bounds"` failed.
...
UR_RESULT_ERROR_DEVICE_LOST
```

**Workaround** (`av1sfm/learned.py`): no boolean indexing on the GPU. DISK's
keypoints are selected on the CPU from the score map. LightGlue's point
pruning and final match selection compute index lists on the CPU, and the GPU
gathers with `index_select`.

### 2. `softmax` and `log_softmax` over more than 4096 elements are wrong

Along a dimension of more than 4096 elements, `softmax` and `log_softmax`
return wrong values. `log_softmax` over rows of n elements, maximum relative
difference to the CPU on random inputs:

| n | Relative difference |
|---:|---:|
| 4096 | 6.8·10⁻⁸ (correct) |
| 4097 | 0.97 |
| 5000 | 1.00 |

`softmax` over 5000 elements fails as well, while matrix products do not:
`softmax(S) @ V` summing over 5000 elements is correct (3·10⁻⁶) when the
softmax is computed on the CPU, and wrong (0.06–0.8, varying between runs)
when it is computed on the GPU, whether written with `@`, a batched `matmul`
or `einsum`. (`eval/xpu_repro.py` now also prints `softmax` alone at 4096,
4097 and 5000.) `logsumexp`, `max` and `sum` over the same rows are correct, and so
is `scaled_dot_product_attention`, whose fused kernel computes the softmax
itself.

Attention between two images with 5000 keypoints each takes a softmax over
5000 elements. This broke kornia's LightGlue cross-attention, which writes the
softmax out explicitly on non-CUDA devices: matches were wrong from the first
layer (3,967 matches on the CPU against 44 on the GPU, none in common). It
also broke LightGlue's final assignment (its dual softmax). Our first
diagnosis, inside LightGlue, blamed the matrix products that follow the
softmax; the minimal reproduction separated the two.

**Workarounds:** attention runs through `scaled_dot_product_attention`
(`cross_block_forward`), which is mathematically identical, and on the CPU
gives the same matches as kornia. The dual softmax is written as
`x - logsumexp(x)` (`double_softmax_maxima`), with reductions over the last
dimension of contiguous tensors; `tests/test_learned.py` checks it against
kornia's version. Exact SIFT matching uses neither operation, and every
selected product is re-checked in integer arithmetic on each run.

With these workarounds and the one for bug 1, DISK + LightGlue on the Intel
GPU gives the same matches as on the CPU (3,969 of 3,969 on a KITTI pair in
float32; 2,666 of 2,666 with float16 attention and pruning).

## kornia 0.8.3

### 3. `LightGlue` fails on devices other than CPU, MPS and CUDA

`LightGlue._forward` always calls `pruning_min_kpts(device)`, which looks the
device type up in `pruning_keypoint_thresholds = {"cpu": -1, "mps": -1,
"cuda": 1024, "flash": 1536}`. Any other device fails before matching starts,
even with pruning disabled:

```python
LightGlue("disk").pruning_min_kpts(torch.device("xpu"))   # KeyError: 'xpu'
```

**Workaround:** `DiskLightGlue.match` runs kornia's layers in the order of its
forward pass, with the same early stopping and pruning. On a GPU it uses
CUDA's thresholds: 1536 keypoints with float16 attention, else 1024. On the
CPU it gives the same matches as kornia's own forward pass. A fix upstream
would be a default for unknown device types.

### 4. Fast attention only on CUDA

kornia (like the original LightGlue) runs attention in float16 through
`scaled_dot_product_attention` only when `device.type == "cuda"`. Elsewhere,
self-attention runs in float32, and cross-attention builds the full similarity
matrix with `einsum` and applies two softmaxes. On the CPU, kornia's forward
pass took 4.6–12 s per KITTI pair against 1.1–2.2 s for the same computation
through `scaled_dot_product_attention`. On Intel GPUs it also triggers bug 2.

**What av1sfm does:** the same attention kernels on every device, float16 on
any GPU. On the Lunar Lake GPU, float16 attention with pruning took 166 ms per
pair against 923 ms in float32 without pruning, and gave the same matches as
the CPU (2,666 of 2,666). On all 117 KITTI frames, matching took 173 s instead
of 1,037 s, with practically the same matches and reconstruction.

### 5. `LightGlue` prints to stdout

The constructor prints `Loaded LightGlue model`, so a script cannot write
machine-readable output to stdout. `eval/run_lightglue.py` writes its
statistics with `--stats FILE`.

## COLMAP / pycolmap 4.2.1

### 6. The default SIFT matcher is approximate and not reproducible

COLMAP's CPU matcher defaults to an approximate nearest-neighbour search
(`cpu_brute_force_matcher = false`). Its matches depend on the thread count:
KITTI 00, 117 frames, sequential matching, raw matches per image on a 4-core
machine: 17,218 with 4 threads, 17,179 with 8, 17,126 with 16; the exact
search gives 17,457. On an 8-thread laptop the result also varied from run to
run (15.6–16.0 k), and 10 % of the sequential pairs got different matches
from the exhaustive run for the same pair of images. The exact matcher is
about 40× slower (1,348 s against 33 s).

**What av1sfm does:** the baselines keep COLMAP's default, as the paper does,
and `eval/run_sift.py --num-threads N` fixes the thread count. `--matcher
exact` runs the exact search on a GPU with PyTorch: the same matches as
COLMAP's exact matcher, pair for pair (1,021,246 raw matches over 1,115 pairs),
in 16.1 s on the Lunar Lake GPU, including verification. ASSUMPTIONS.md R4.

### 7. Geometric verification is not deterministically seeded

COLMAP's verification of two identical match sets kept 1,002,714 and
1,002,709 inliers. Differences of this size between runs are noise. av1sfm's
own pairwise scoring is seeded and takes the median of three runs
(ASSUMPTIONS.md G6).

### 8. Mapper defaults and forward motion

With the default `init_max_forward_motion = 0.95`, the mapper never finds an
initial image pair on KITTI's forward drive. With COLMAP's default intrinsics
refinement, several motion-vector databases collapse to two registered images:
the first bundle adjustment drives the radial distortion to implausible values
and COLMAP rejects the camera. The evaluation uses `init_max_forward_motion =
1.0`, `init_min_tri_angle = 4`, and reports both fixed and refined
intrinsics (ASSUMPTIONS.md R5).

## Video encoding

### 9. FFmpeg `av1_vaapi`: `-q:v` is not the AV1 qindex

For `av1_vaapi`, `-q:v 128` sets FFmpeg's QSCALE flag, and the value is then
divided by `FF_QP2LAMBDA`, which encodes at a different quality than intended.
`-global_quality 128` sets the qindex itself. The keyframe's qindex is further
scaled by FFmpeg's default `i_qfactor` (0.8). av1sfm uses `-global_quality`
(ASSUMPTIONS.md E5f).

### 10. Vulkan Video AV1 encode not exposed on Lunar Lake

With Mesa ANV from Intel's graphics PPA on Ubuntu 26.04, FFmpeg's
`av1_vulkan` stops at "Device does not support the VK_KHR_video_encode_queue
extension". The same Lunar Lake GPU encodes AV1 through VA-API and QSV, so
this is a driver limitation. `av1sfm encoders` reports it, and `--encoder
auto` falls through to QSV or VA-API (ASSUMPTIONS.md E5b).

### 11. Low-delay encodes still reference older frames

The paper assumes all motion vectors point to the previous frame. libaom's
realtime mode keeps at least three reference slots, so 2.3 % of its motion
vectors on KITTI point further back (up to 80 frames). SVT-AV1's real-time
mode uses temporal layers even with a low-delay prediction structure (51 % to
the previous frame, 19 % two-reference blocks), and `hierarchical-levels=2`
did not change that. VA-API sends 7.4 % to the frame before the previous one.
av1sfm attaches every motion vector to its actual reference frame, through the
decoder's reference order hints; `--prev-only` keeps only those to the
previous frame (ASSUMPTIONS.md E3, E5a).
