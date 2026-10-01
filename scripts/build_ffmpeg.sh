#!/usr/bin/env bash
# Build a recent FFmpeg with SVT-AV1, libaom, Vulkan (AV1 Vulkan Video encode),
# Intel QSV (oneVPL) and VA-API into third_party/build/media. av1sfm uses this ffmpeg
# automatically when it exists (override with AV1SFM_FFMPEG=/path/to/ffmpeg).
#
# Built from source (latest releases at the time of writing):
#   FFmpeg n9.0.2, SVT-AV1 v4.2.0, Vulkan-Headers v1.4.364, libvpl v2.17.0
# Taken from the system: libaom, libva/libdrm (QSV on Linux sits on VA-API),
# the Vulkan loader. On Ubuntu 24.04:
#   sudo apt-get install -y build-essential cmake ninja-build nasm pkg-config \
#        libaom-dev libva-dev libdrm-dev libvulkan-dev
# Runtime GPU drivers are needed only for hardware encoding: Mesa RADV/ANV
# (Vulkan Video AV1 encode), a VA-API driver with AV1 encode (VA-API), or
# Intel's VPL GPU runtime on top of it (QSV). For Intel Lunar Lake on Ubuntu
# 26.04 see docs/USAGE.md ("Intel Lunar Lake / Arc on Ubuntu 26.04").
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="${AV1SFM_MEDIA_PREFIX:-$ROOT/third_party/build/media}"
SRC="$PREFIX/src"
JOBS="${JOBS:-$(nproc)}"

FFMPEG_TAG="${FFMPEG_TAG:-n9.0.2}"
SVTAV1_TAG="${SVTAV1_TAG:-v4.2.0}"
VKHDR_TAG="${VKHDR_TAG:-v1.4.364}"
VPL_TAG="${VPL_TAG:-v2.17.0}"

mkdir -p "$SRC"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PREFIX/lib/x86_64-linux-gnu/pkgconfig:${PKG_CONFIG_PATH:-}"

# fetch <dir> <tag> <remote>...: shallow-fetch a tag from the first remote that works.
fetch() {
  local dir="$1" tag="$2"; shift 2
  [ -d "$SRC/$dir/.git" ] && return
  rm -rf "${SRC:?}/$dir"; git init -q "$SRC/$dir"
  for remote in "$@"; do
    if git -C "$SRC/$dir" fetch -q --depth 1 "$remote" "refs/tags/$tag:refs/tags/$tag"; then
      git -C "$SRC/$dir" checkout -q "$tag"; return
    fi
    echo "build_ffmpeg.sh: $remote unavailable, trying next" >&2
  done
  echo "build_ffmpeg.sh: cannot fetch $dir $tag" >&2; exit 1
}

cmake_install() {  # cmake_install <dir> [cmake args...]
  local dir="$1"; shift
  cmake -S "$SRC/$dir" -B "$SRC/$dir/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_INSTALL_RPATH="$PREFIX/lib" "$@" > /dev/null
  cmake --build "$SRC/$dir/build" -j "$JOBS"
  cmake --install "$SRC/$dir/build" > /dev/null
}

# Vulkan headers new enough for VK_KHR_video_encode_av1.
fetch vulkan-headers "$VKHDR_TAG" https://github.com/KhronosGroup/Vulkan-Headers.git
cmake_install vulkan-headers

fetch svt-av1 "$SVTAV1_TAG" https://gitlab.com/AOMediaCodec/SVT-AV1.git \
  https://github.com/AOMediaCodec/SVT-AV1.git
cmake_install svt-av1 -DBUILD_SHARED_LIBS=ON -DBUILD_APPS=ON -DBUILD_TESTING=OFF

fetch libvpl "$VPL_TAG" https://github.com/intel/libvpl.git
cmake_install libvpl -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DINSTALL_EXAMPLES=OFF -DBUILD_EXPERIMENTAL=OFF

fetch ffmpeg "$FFMPEG_TAG" https://git.ffmpeg.org/ffmpeg.git https://github.com/FFmpeg/FFmpeg.git
cd "$SRC/ffmpeg"
if [ ! -f ffbuild/config.mak ] || [ "${RECONFIGURE:-0}" = 1 ]; then
  ./configure --prefix="$PREFIX" \
    --extra-cflags="-I$PREFIX/include" \
    --extra-ldflags="-L$PREFIX/lib -Wl,-rpath,$PREFIX/lib" \
    --enable-shared --disable-static --disable-doc --disable-debug --disable-ffplay \
    --enable-libsvtav1 --enable-libaom --enable-libdav1d \
    --enable-vulkan --enable-libvpl --enable-vaapi
fi
make -j "$JOBS"
make install > /dev/null

"$PREFIX/bin/ffmpeg" -hide_banner -version | head -1
"$PREFIX/bin/ffmpeg" -hide_banner -encoders 2>/dev/null | grep -E "av1" || true
echo "build_ffmpeg.sh: installed to $PREFIX/bin/ffmpeg"
