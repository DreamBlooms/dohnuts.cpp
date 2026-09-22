#!/usr/bin/env bash
# Build dohnuts.cpp from source on Ubuntu / Debian.
#
# Usage: scripts/build.sh [build_dir] [cmake_extra_args...]
#
# Backends: pass any of -DDOHNUTS_CUDA=ON, -DDOHNUTS_VULKAN=ON,
# -DDOHNUTS_HIP=ON. The default is CPU only.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD=${1:-$ROOT/build}
if [ $# -gt 0 ]; then shift; fi

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
    -DCMAKE_BUILD_TYPE=Release \
    "$@"

echo "==> building"
cmake --build "$BUILD" -j"$(nproc)"

echo "==> done: $BUILD/dohnuts-cli"
echo "    try: $BUILD/dohnuts-cli --help"