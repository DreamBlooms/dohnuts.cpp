#!/usr/bin/env bash
# Cross-compile dohnuts.cpp for Windows x86_64 from Ubuntu / Debian.
#
# Usage: scripts/build_windows.sh [build_dir] [cmake_extra_args...]
#
# Produces a self-contained build-win/dohnuts-cli.exe (no extra DLLs).
# Requires the MinGW-w64 toolchain: sudo apt-get install mingw-w64
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD=${1:-$ROOT/build-win}
if [ $# -gt 0 ]; then shift; fi

if ! command -v x86_64-w64-mingw32-g++ >/dev/null; then
    echo "error: MinGW-w64 not found. Install it with:" >&2
    echo "  sudo apt-get install mingw-w64" >&2
    exit 1
fi

if ! command -v cmake >/dev/null; then
    echo "error: cmake not found; run scripts/setup.sh first" >&2
    exit 1
fi

if [ ! -f "$ROOT/third_party/llama.cpp/CMakeLists.txt" ]; then
    echo "==> initializing llama.cpp submodule"
    git -C "$ROOT" submodule update --init --depth 1
fi

echo "==> configuring ($BUILD)"
cmake -S "$ROOT" -B "$BUILD" \
    -DCMAKE_TOOLCHAIN_FILE="$ROOT/cmake/mingw-w64-x86_64.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DDOHNUTS_STATIC=ON \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    "$@"

echo "==> building"
cmake --build "$BUILD" -j"$(nproc)"

if [ -f "$BUILD/dohnuts-cli.exe" ]; then
    echo "==> stripping debug symbols"
    x86_64-w64-mingw32-strip --strip-all "$BUILD/dohnuts-cli.exe"
fi

echo "==> done: $BUILD/dohnuts-cli.exe"