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

step "the generated documents are regenerable"
# docs/ROOFLINE.md, docs/SCREEN.md and issues/ are all GENERATED from configs/. A number typed
# into any of them by hand would contradict the generation, which is the file the scorer reads.
python3 eval/screen.py --markdown /tmp/burnish-screen.md >/dev/null
diff -q /tmp/burnish-screen.md docs/SCREEN.md >/dev/null || {
  echo "!! docs/SCREEN.md is stale: python3 eval/screen.py --markdown docs/SCREEN.md"; fail=1; }
tmpissues=$(mktemp -d)
python3 scripts/make_issues.py --write --out "$tmpissues" >/dev/null
diff -r -q "$tmpissues" issues >/dev/null || {
  echo "!! issues/ is stale: python3 scripts/make_issues.py --write"; fail=1; }
rm -rf "$tmpissues"

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

step "the runtime's tensor names match the pinned checkpoint"
# Offline, against the layout record in configs/. The record was produced by reading the real
# checkpoint's safetensors headers over HTTP range requests at the pinned revisions. This is the
# check that turned "these names are probably right" into evidence.
python3 scripts/verify_checkpoint_layout.py --against-saved || fail=1

step "manifest"
python3 scripts/manifest.py --check || fail=1

echo
if [ "$fail" -eq 0 ]; then echo "all no-GPU checks passed"; else echo "FAILURES above"; fi
exit "$fail"
