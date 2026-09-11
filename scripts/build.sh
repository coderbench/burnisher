#!/usr/bin/env bash
# CPU build: the correctness oracle and the whole harness. Needs only a C++17 compiler.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
cmake -S . -B build -DCMAKE_BUILD_TYPE="${BUILD_TYPE:-Release}" -DBURNISHER_BUILD_TESTS=ON
cmake --build build -j"$(nproc)"
echo ">> built build/burnisher"
