#!/usr/bin/env bash
# Install build dependencies for dohnuts.cpp on Ubuntu / Debian.
#
# Usage: sudo scripts/setup.sh
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "error: run as root (e.g. sudo $0)" >&2
    exit 1
fi

if ! command -v apt-get >/dev/null; then
    echo "error: apt-get not found; this script targets Ubuntu / Debian" >&2
    exit 1
fi

echo "==> apt-get update"
apt-get update

echo "==> installing build dependencies"
apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    libcurl4-openssl-dev \
    nlohmann-json3-dev

# Optional: python3 and pip are only needed to export/convert models.
if ! command -v python3 >/dev/null; then
    echo "note: python3 not found; install it only if you plan to convert models"
fi

echo "==> versions"
cmake --version | head -n1
g++ --version | head -n1

echo "==> done"