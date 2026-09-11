#!/usr/bin/env bash
# CUDA build, for the pinned hardware. sm_120 is RTX 5090 and RTX PRO 6000 Blackwell;
# sm_121 is DGX Spark (GB10).
#
# This is the build that can MEASURE anything. Without it, `burnish probe`, `burnish calibrate`,
# `burnish gate` and `burnish bench` all refuse to run rather than estimating.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ARCH="${CMAKE_CUDA_ARCHITECTURES:-120}"
command -v nvcc >/dev/null || { echo "!! nvcc is not on PATH"; exit 2; }
cmake -S . -B build-cuda -DBURNISHER_BUILD_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="${ARCH}" -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j"$(nproc)"
ctest --test-dir build-cuda --output-on-failure
echo ">> built build-cuda/burnisher for sm_${ARCH}"
