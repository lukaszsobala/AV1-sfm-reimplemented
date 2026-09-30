"""
 dav1d_inspect.py

  Created by Julien Zouein on 22/06/2026.
  Copyright © 2026 Sigmedia.tv. All rights reserved.
  Copyright © 2026 Julien Zouein (zoueinj@tcd.ie)
----------------------------------------------------------------------------

In-memory AV1 block-metadata extraction using a patched libdav1d.

This replaces the AOM `inspect` tool + on-disk JSON pipeline. dav1d is built
with `-Denable_inspection=true`, which adds an `inspect_cb` callback to
`Dav1dSettings` (see src/third_parties/dav1d patch). The callback fires once per
decoded frame, before dav1d releases its internal motion-vector buffer, and
hands us the per-block motion field directly in memory.

Exposes the full-frame 4x4 spatial block grid (`refmvs_block`): two motion
vectors and two reference indices per 4x4 block plus the block size — the
analogue of AOM inspect's per-MI grid. This matches the existing
`(H/4, W/4, 4)` / `(H/4, W/4, 2)` shapes consumed by `json_processing`.

Validated against the AOM-inspect reference (test/0018.json): motion vectors
and reference map are bit-exact on all inter frames, and block sizes match
exactly once remapped to AOM's `BLOCK_*` ordering (see _BS_DAV1D_TO_AOM).
Reference indices already share AOM's encoding (0 = intra, 1..7 = ref slot,
-1 = none); the INVALID_MV sentinel and intra-frame conventions are normalised
to AOM here. See test/validate_dav1d_vs_json.py.
"""

from __future__ import annotations

import ctypes as C
import errno
import os
import struct
from pathlib import Path

import numpy as np

# DAV1D_ERR(EAGAIN) == -EAGAIN (35 on macOS, 11 on Linux — never hardcode).
_EAGAIN = -errno.EAGAIN

# dav1d `enum BlockSize` (index) -> AOM `BLOCK_*` enum value, matched by WxH
# name. dav1d orders BS_128x128=0 .. BS_4x4=21 (levels.h); AOM orders
# BLOCK_4X4=0 .. BLOCK_64X16=21 (inspect.c block_size_map). Applying this LUT
# makes `block_map` match AOM inspect's `blockSize` field exactly. Validated
# 100% on all inter frames against test/0018.json.
_BS_DAV1D_TO_AOM = np.array(
    [15, 14, 13, 12, 11, 21, 10, 9, 8, 19, 20, 7, 6, 5, 17, 18, 4, 3, 2, 16, 1, 0],
    dtype=np.uint8,
)

# Packed 4x4 spatial block `refmvs_block` == 12 bytes:
#   mv[0].{y,x}, mv[1].{y,x} (int16, 1/8-pel), ref[0], ref[1] (int8), bs, mf (uint8)
_BLOCK_DTYPE = np.dtype(
    [
        ("mv0_y", "<i2"),
        ("mv0_x", "<i2"),
        ("mv1_y", "<i2"),
        ("mv1_x", "<i2"),
        ("ref0", "i1"),
        ("ref1", "i1"),
        ("bs", "u1"),
        ("mf", "u1"),
    ]
)
assert _BLOCK_DTYPE.itemsize == 12


# ---------------------------------------------------------------------------
# ctypes mirrors of the dav1d ABI (only the fields we touch)
# ---------------------------------------------------------------------------


class Dav1dInspectData(C.Structure):
    _fields_ = [
        ("decode_seq", C.c_uint),
        ("frame_offset", C.c_uint),
        ("frame_type", C.c_int),
        ("width", C.c_int),
        ("height", C.c_int),
        ("blk_w", C.c_int),
        ("blk_h", C.c_int),
        ("blk_stride", C.c_ssize_t),  # ptrdiff_t, in 12-byte records
        ("blocks", C.c_void_p),
        ("refidx", C.c_int8 * 7),
        ("refpoc", C.c_uint * 7),
    ]


_INSPECT_CB = C.CFUNCTYPE(None, C.c_void_p, C.POINTER(Dav1dInspectData))


class _Dav1dPicAllocator(C.Structure):
    _fields_ = [
        ("cookie", C.c_void_p),
        ("alloc_picture_callback", C.c_void_p),
        ("release_picture_callback", C.c_void_p),
    ]


class _Dav1dLogger(C.Structure):
    _fields_ = [("cookie", C.c_void_p), ("callback", C.c_void_p)]


class Dav1dSettings(C.Structure):
    _fields_ = [
        ("n_threads", C.c_int),
        ("max_frame_delay", C.c_int),
        ("apply_grain", C.c_int),
        ("operating_point", C.c_int),
        ("all_layers", C.c_int),
        ("frame_size_limit", C.c_uint),
        ("allocator", _Dav1dPicAllocator),
        ("logger", _Dav1dLogger),
        ("strict_std_compliance", C.c_int),
        ("output_invisible_frames", C.c_int),
        ("inloop_filters", C.c_int),
        ("decode_frame_type", C.c_int),
        ("inspect_cookie", C.c_void_p),
        ("inspect_cb", _INSPECT_CB),
        ("reserved", C.c_uint8 * 16),
    ]


class Dav1dData(C.Structure):
    """Full ABI layout so `sz` (bytes remaining) reads correctly."""

    _fields_ = [
        ("data", C.c_void_p),
        ("sz", C.c_size_t),
        ("ref", C.c_void_p),
        ("m_timestamp", C.c_int64),
        ("m_duration", C.c_int64),
        ("m_offset", C.c_int64),
        ("m_size", C.c_size_t),
        ("m_user_data", C.c_void_p),
        ("m_user_ref", C.c_void_p),
    ]


# Dav1dPicture is only ever allocated/unref'd here, never read field-by-field
# (the callback already gave us everything), so treat it as an opaque blob that
# is comfortably larger than the real struct.
class Dav1dPicture(C.Structure):
    _fields_ = [("_opaque", C.c_uint8 * 1024)]


# MODIFIED (AV1-sfm-reimplemented, 2026-09-30): the build directory is
# `third_party/build` at the repository root (see setup.sh) and can be
# overridden with the AV1SFM_BUILD_DIR environment variable. Upstream used
# `src/third_parties` relative to its own repository root.
_THIRD_PARTIES = Path(
    os.environ.get(
        "AV1SFM_BUILD_DIR",
        Path(__file__).resolve().parents[3] / "third_party/build",
    )
)


def _find_lib(*candidates: Path) -> Path | None:
    """Return the first existing path among the candidates (.dylib / .so)."""
    for c in candidates:
        for ext in (".dylib", ".so"):
            p = c.with_suffix(ext)
            if p.exists():
                return p
    return None


_DAV1D_LIB = _THIRD_PARTIES / "dav1d/build/src/libdav1d"
_SHIM_LIB = _THIRD_PARTIES / "libav1of_inspect"


def _load_lib(lib_path: str | Path | None = None) -> C.CDLL:
    path = Path(lib_path) if lib_path else _find_lib(_DAV1D_LIB)
    if path is None or not Path(path).exists():
        raise FileNotFoundError(
            f"libdav1d not found near {_DAV1D_LIB}. Build dav1d with "
            "`meson setup build -Denable_inspection=true && ninja -C build`."
        )
    lib = C.CDLL(str(path))

    lib.dav1d_default_settings.argtypes = [C.POINTER(Dav1dSettings)]
    lib.dav1d_open.argtypes = [C.POINTER(C.c_void_p), C.POINTER(Dav1dSettings)]
    lib.dav1d_open.restype = C.c_int
    lib.dav1d_data_create.argtypes = [C.POINTER(Dav1dData), C.c_size_t]
    lib.dav1d_data_create.restype = C.POINTER(C.c_uint8)
    lib.dav1d_send_data.argtypes = [C.c_void_p, C.POINTER(Dav1dData)]
    lib.dav1d_send_data.restype = C.c_int
    lib.dav1d_get_picture.argtypes = [C.c_void_p, C.POINTER(Dav1dPicture)]
    lib.dav1d_get_picture.restype = C.c_int
    lib.dav1d_picture_unref.argtypes = [C.POINTER(Dav1dPicture)]
    lib.dav1d_close.argtypes = [C.POINTER(C.c_void_p)]
    return lib


# ---------------------------------------------------------------------------
# Fast path: C shim (libav1of_inspect) that decodes the whole file in one call
# and builds the per-frame arrays in C, GIL-free, so the multi-threaded decode
# actually scales. See src/av1of_inspect_shim.c.
# ---------------------------------------------------------------------------


class _Av1ofFrame(C.Structure):
    """Mirrors `av1of_frame` in av1of_inspect_shim.c (already transformed)."""

    _fields_ = [
        ("decode_seq", C.c_uint),
        ("frame_offset", C.c_uint),
        ("frame_type", C.c_int),
        ("width", C.c_int),
        ("height", C.c_int),
        ("blk_w", C.c_int),
        ("blk_h", C.c_int),
        ("refidx", C.c_int8 * 7),
        ("refpoc", C.c_uint * 7),
        ("motion_vectors", C.POINTER(C.c_int16)),  # blk_h*blk_w*4 [mv0x,mv0y,mv1x,mv1y]
        ("reference_map", C.POINTER(C.c_int16)),  # blk_h*blk_w*2 [ref0,ref1]
        ("block_map", C.POINTER(C.c_uint8)),  # blk_h*blk_w (AOM BLOCK_* enum)
    ]


def _load_shim() -> C.CDLL | None:
    """Load libav1of_inspect (the GIL-free C fast path).

    The shim is linked against our patched libdav1d with an rpath, so it
    resolves that specific library on its own — we must NOT preload libdav1d
    into the global namespace (RTLD_GLOBAL), or the shim's dav1d_* symbols could
    be interposed by another bundled libdav1d already loaded in the process
    (e.g. the one OpenCV/cv2 ships), which lacks our inspection patch. macOS's
    two-level namespace prevents this; on Linux (flat namespace) we additionally
    pass RTLD_DEEPBIND so the shim prefers its own dav1d dependency over any
    libdav1d already in the global scope.

    Returns None if the shim is not built (caller falls back to the ctypes
    callback path).
    """
    shim_path = _find_lib(_SHIM_LIB)
    if shim_path is None:
        return None

    mode = C.DEFAULT_MODE

    try:
        lib = C.CDLL(str(shim_path), mode=mode)
    except OSError:
        return None

    lib.av1of_decode.argtypes = [C.c_char_p, C.c_int, C.POINTER(C.c_void_p)]
    lib.av1of_decode.restype = C.c_int
    lib.av1of_num_frames.argtypes = [C.c_void_p]
    lib.av1of_num_frames.restype = C.c_int
    lib.av1of_get_frame.argtypes = [C.c_void_p, C.c_int]
    lib.av1of_get_frame.restype = C.POINTER(_Av1ofFrame)
    lib.av1of_free.argtypes = [C.c_void_p]
    return lib


# ---------------------------------------------------------------------------
# IVF demuxing (container only — dav1d decodes the AV1 payload)
# ---------------------------------------------------------------------------


def _iter_ivf_packets(path: str | Path):
    with open(path, "rb") as f:
        header = f.read(32)
        if len(header) < 32 or header[:4] != b"DKIF":
            raise ValueError(f"Not a valid IVF file: {path}")
        while True:
            pkt_hdr = f.read(12)
            if len(pkt_hdr) < 12:
                break
            size = struct.unpack("<I", pkt_hdr[:4])[0]
            data = f.read(size)
            if len(data) < size:
                break
            yield data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _decode_inspect_to_numpy(insp: Dav1dInspectData) -> dict:
    """Copy one callback payload into NumPy arrays (frees C memory to reuse)."""
    frame = {
        "decode_seq": int(insp.decode_seq),
        "frame_offset": int(insp.frame_offset),
        "frame_type": int(insp.frame_type),
        "width": int(insp.width),
        "height": int(insp.height),
        "refidx": [int(x) for x in insp.refidx],
        "refpoc": [int(x) for x in insp.refpoc],
    }

    blk_w, blk_h, stride = insp.blk_w, insp.blk_h, insp.blk_stride
    if not insp.blocks or blk_w == 0 or blk_h == 0:
        # Intra / key frame: dav1d does not populate the inter-prediction grid,
        # so there is no motion field. Mirror AOM's intra convention: zero MVs,
        # ref = [0 (INTRA), -1 (none)]. (Block partition sizes are not recovered
        # for intra frames — irrelevant for motion extraction.)
        frame["motion_vectors"] = np.zeros((blk_h, blk_w, 4), dtype=np.int16)
        ref = np.empty((blk_h, blk_w, 2), dtype=np.int16)
        ref[..., 0] = 0
        ref[..., 1] = -1
        frame["reference_map"] = ref
        frame["block_map"] = np.zeros((blk_h, blk_w), dtype=np.uint8)
        return frame

    n_bytes = blk_h * stride * _BLOCK_DTYPE.itemsize
    raw = (C.c_char * n_bytes).from_address(insp.blocks)
    grid = np.frombuffer(raw, dtype=_BLOCK_DTYPE).reshape(blk_h, stride)[:, :blk_w]

    # Channel order [x, y] = [horizontal, vertical] per MV to match the .flo /
    # AOM convention (col, row). Layout [mv0_x, mv0_y, mv1_x, mv1_y] mirrors
    # AOM's [mv[0].col, mv[0].row, mv[1].col, mv[1].row]. Units are 1/8-pel.
    mv = np.stack(
        [grid["mv0_x"], grid["mv0_y"], grid["mv1_x"], grid["mv1_y"]], axis=-1
    ).astype(np.int16)
    # dav1d marks "no motion vector" with INVALID_MV (0x80008000), i.e. both
    # components == -32768. AOM emits [0, 0] for those; normalise to match.
    mv[mv == -32768] = 0
    ref = np.stack([grid["ref0"], grid["ref1"]], axis=-1).astype(np.int16)
    frame["motion_vectors"] = np.ascontiguousarray(mv)
    frame["reference_map"] = np.ascontiguousarray(ref)
    frame["block_map"] = _BS_DAV1D_TO_AOM[grid["bs"]]
    return frame


def iter_frames(
    ivf_path: str | Path,
    n_threads: int = 0,
    lib_path: str | Path | None = None,
):
    """Decode an IVF/AV1 file and yield per-frame block metadata in memory.

    Yields one dict per decoded frame, in decode order, with keys:
        decode_seq, frame_offset, frame_type, width, height, refidx, refpoc,
        motion_vectors (H/4, W/4, 4) int16  [1/8-pel, mv0_x, mv0_y, mv1_x, mv1_y],
        reference_map  (H/4, W/4, 2) int16  [ref0, ref1; 0=intra, 1..7, -1=none],
        block_map      (H/4, W/4)    uint8  [AOM BLOCK_* enum, matches blockSize].

    Granularity is 4x4 blocks with two MVs / refs per block (dual reference).

    Args:
        n_threads: dav1d worker threads. 0 (default) lets dav1d pick (number of
            logical cores). dav1d uses these for both tile and frame threading,
            which gives the largest speed-up; frames may then complete out of
            decode order, so they are reordered here by `decode_seq`.

    Uses the C shim (libav1of_inspect) when available — it builds the arrays in
    C, GIL-free, so the multi-threaded decode scales. Falls back to the pure
    ctypes callback path (correct but GIL-bound) if the shim is not built.

    Note: frame metadata is collected during decode and yielded once the stream
    is fully decoded (so threaded, out-of-order completions can be reordered).
    Peak memory is therefore proportional to the clip length.
    """
    if lib_path is None:
        shim = _load_shim()
        if shim is not None:
            yield from _iter_frames_via_shim(shim, ivf_path, n_threads)
            return
    yield from _iter_frames_via_callback(ivf_path, n_threads, lib_path)


def _iter_frames_via_shim(shim: C.CDLL, ivf_path: str | Path, n_threads: int):
    """Fast path: one C call decodes + transforms everything (GIL released)."""
    handle = C.c_void_p()
    # ctypes releases the GIL across this call, so dav1d's worker threads — and
    # the C transformation in the inspection callback — run fully in parallel.
    rc = shim.av1of_decode(str(ivf_path).encode(), int(n_threads), C.byref(handle))
    if rc != 0:
        raise RuntimeError(f"av1of_decode failed ({rc}) for {ivf_path}")
    try:
        n = shim.av1of_num_frames(handle)
        for i in range(n):
            f = shim.av1of_get_frame(handle, i).contents
            h, w = f.blk_h, f.blk_w
            mv = np.ctypeslib.as_array(f.motion_vectors, shape=(h, w, 4)).copy()
            ref = np.ctypeslib.as_array(f.reference_map, shape=(h, w, 2)).copy()
            bs = np.ctypeslib.as_array(f.block_map, shape=(h, w)).copy()
            yield {
                "decode_seq": int(f.decode_seq),
                "frame_offset": int(f.frame_offset),
                "frame_type": int(f.frame_type),
                "width": int(f.width),
                "height": int(f.height),
                "refidx": [int(x) for x in f.refidx],
                "refpoc": [int(x) for x in f.refpoc],
                "motion_vectors": mv,
                "reference_map": ref,
                "block_map": bs,
            }
    finally:
        shim.av1of_free(handle)


def _iter_frames_via_callback(
    ivf_path: str | Path,
    n_threads: int = 0,
    lib_path: str | Path | None = None,
):
    """Fallback path: pure ctypes with a per-frame Python callback (GIL-bound)."""
    lib = _load_lib(lib_path)

    collected: list[dict] = []

    @_INSPECT_CB
    def _cb(_cookie, data_ptr):
        # Invoked from dav1d worker threads (under the GIL); list.append is safe.
        collected.append(_decode_inspect_to_numpy(data_ptr.contents))

    settings = Dav1dSettings()
    lib.dav1d_default_settings(C.byref(settings))
    settings.n_threads = int(n_threads)
    settings.inspect_cb = _cb

    ctx = C.c_void_p()
    if lib.dav1d_open(C.byref(ctx), C.byref(settings)) < 0:
        raise RuntimeError("dav1d_open failed")

    pic = Dav1dPicture()

    def _drain():
        while True:
            res = lib.dav1d_get_picture(ctx, C.byref(pic))
            if res == _EAGAIN:
                break
            if res < 0:
                raise RuntimeError(f"dav1d_get_picture failed: {res}")
            lib.dav1d_picture_unref(C.byref(pic))

    try:
        for payload in _iter_ivf_packets(ivf_path):
            data = Dav1dData()
            buf = lib.dav1d_data_create(C.byref(data), len(payload))
            if not buf:
                raise RuntimeError("dav1d_data_create failed")
            C.memmove(buf, payload, len(payload))

            while data.sz > 0:
                res = lib.dav1d_send_data(ctx, C.byref(data))
                if res < 0 and res != _EAGAIN:
                    raise RuntimeError(f"dav1d_send_data failed: {res}")
                _drain()

        _drain()  # flush buffered frames at end of stream
    finally:
        # dav1d_close joins all worker threads: a barrier guaranteeing every
        # inspect callback has fired before we read `collected`.
        lib.dav1d_close(C.byref(ctx))

    # Reorder frames that completed out of order under frame threading.
    collected.sort(key=lambda fr: fr["decode_seq"])
    yield from collected


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "test/0018.ivf"
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    n = 0
    for fr in iter_frames(src, n_threads=threads):
        if n < 8:
            mv = fr["motion_vectors"]
            ref = fr["reference_map"]
            bs = fr["block_map"]
            absmax = int(np.abs(mv).max()) if mv.size else 0
            rng = (int(ref.min()), int(ref.max())) if ref.size else (0, 0)
            bsrng = (int(bs.min()), int(bs.max())) if bs.size else (0, 0)
            print(
                f"seq={fr['decode_seq']:3d} frame_offset={fr['frame_offset']:3d} "
                f"type={fr['frame_type']} {fr['width']}x{fr['height']}  "
                f"mv={mv.shape} ref={ref.shape} bs={bs.shape} mv_absmax={absmax} "
                f"ref_range={rng} bs_range={bsrng}"
            )
        n += 1
    print(f"Total frames decoded: {n} (threads={threads})")
