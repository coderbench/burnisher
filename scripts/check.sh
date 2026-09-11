#!/usr/bin/env bash
# Everything that can be checked without a GPU. This is what CI runs and what a contributor
# should run before opening a PR.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
fail=0
step() { echo; echo "=== $* ==="; }

step "C++ build and tests"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBURNISHER_BUILD_TESTS=ON >/dev/null
cmake --build build -j"$(nproc)" >/dev/null
ctest --test-dir build --output-on-failure || fail=1

step "the whole graph, on synthetic weights"
./build/burnisher selftest || fail=1

step "harness tests"
python3 -m unittest discover -s eval -t eval -p 'test_*.py' || fail=1

step "the frozen generation still matches configs/"
python3 eval/make_generation.py --check || fail=1

step "the published roofline table is regenerable"
python3 eval/roofline_table.py --markdown /tmp/burnish-roofline.md >/dev/null
diff -q /tmp/burnish-roofline.md docs/ROOFLINE.md >/dev/null || {
  echo "!! docs/ROOFLINE.md is stale. Regenerate it:"
  echo "     python3 eval/roofline_table.py --markdown docs/ROOFLINE.md"
  echo "   Never edit that file by hand -- every figure in it comes from the generation."
  fail=1
}

step "--help on every entry point"
for e in tools/burnish eval/screen.py eval/roofline_table.py eval/make_generation.py \
         eval/bench.py eval/calibrate.py eval/gate.py; do
  python3 "$e" --help >/dev/null || { echo "!! $e --help failed"; fail=1; }
done
./build/burnisher --help >/dev/null || { echo "!! burnisher --help failed"; fail=1; }

step "manifest"
python3 scripts/manifest.py --check || fail=1

echo
if [ "$fail" -eq 0 ]; then echo "all no-GPU checks passed"; else echo "FAILURES above"; fi
exit "$fail"
