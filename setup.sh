#!/usr/bin/env bash
# Build the patched dav1d (per-frame block-metadata inspection callback) and the
# GIL-free extraction shim, both vendored from sigmedia/AV1-Optical-Flow, then
# create the Python environment.
#
# Requirements: meson, ninja, a C compiler (nasm recommended for SIMD); uv for
# the Python environment (optional, see AV1SFM_SKIP_SYNC below).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD="${AV1SFM_BUILD_DIR:-$ROOT/third_party/build}"
DAV1D_COMMIT="14c73c7db38eebfd3202146b76a1ad4df90dd3a2"  # 1.5.3-58-g14c73c7d, as pinned upstream
# Upstream is code.videolan.org; the GitHub mirror is tried as a fallback.
DAV1D_REMOTES=(
  "https://code.videolan.org/videolan/dav1d.git"
  "https://github.com/videolan/dav1d.git"
)

mkdir -p "$BUILD"
cd "$BUILD"

if [ ! -d dav1d/.git ]; then
  rm -rf dav1d
  git init -q dav1d
  fetched=0
  for remote in "${DAV1D_REMOTES[@]}"; do
    if git -C dav1d fetch -q --depth 1 "$remote" "$DAV1D_COMMIT"; then
      fetched=1
      break
    fi
    echo "setup.sh: could not fetch dav1d from $remote, trying next remote" >&2
  done
  [ "$fetched" = 1 ] || { echo "setup.sh: failed to fetch dav1d" >&2; exit 1; }
  git -C dav1d checkout -q FETCH_HEAD
  git -C dav1d apply "$ROOT/third_party/av1of/dav1d-inspection.patch"
fi

cd dav1d
if [ ! -d build ]; then
  meson setup build --buildtype release -Denable_inspection=true \
    -Denable_tools=false -Denable_tests=false
fi
ninja -C build
cd "$BUILD"

# Link the shim against OUR patched libdav1d with an rpath so that it is not
# interposed by another libdav1d loaded in the process (e.g. OpenCV's copy).
SHIM_CC_FLAGS="-O3 -shared -fPIC -I dav1d/include"
if [ "$(uname)" = "Darwin" ]; then
  cc $SHIM_CC_FLAGS -o libav1of_inspect.dylib "$ROOT/third_party/av1of/av1of_inspect_shim.c" \
    dav1d/build/src/libdav1d.7.dylib -Wl,-rpath,@loader_path/dav1d/build/src
else
  cc $SHIM_CC_FLAGS -o libav1of_inspect.so "$ROOT/third_party/av1of/av1of_inspect_shim.c" \
    -L dav1d/build/src -ldav1d -Wl,-rpath,'$ORIGIN/dav1d/build/src'
fi

cd "$ROOT"
# AV1SFM_SKIP_SYNC=1 builds only the native libraries, e.g. into an already
# active virtualenv set up with `pip install -e .`.
if [ "${AV1SFM_SKIP_SYNC:-0}" = 1 ]; then
  echo "setup.sh: done (skipped uv sync). Try: pytest"
elif command -v uv > /dev/null; then
  uv sync
  echo "setup.sh: done. Try: uv run pytest"
else
  echo "setup.sh: native libraries built; uv not found, so install the package" \
       "into your environment with: pip install -e ." >&2
fi
