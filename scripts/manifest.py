#!/usr/bin/env python3
"""Regenerate or check repo-manifest.json.

A manifest exists so that "what is in this repository, and what state is it in" has one answer
that a machine produced. The file list is generated from git rather than typed, and the counts
are generated from the suites' own output rather than remembered, because both of those drift
the moment they are maintained by hand.

    scripts/manifest.py --write
    scripts/manifest.py --check      # CI
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def tracked_files():
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True,
                         check=True)
    return sorted(p for p in out.stdout.splitlines() if p)


def count_python_tests():
    out = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "eval",
                          "-t", "eval", "-p", "test_*.py", "-v"],
                         cwd=ROOT, capture_output=True, text=True)
    text = out.stdout + out.stderr
    for line in text.splitlines():
        if line.startswith("Ran ") and " test" in line:
            return int(line.split()[1]), out.returncode == 0
    return None, False


def count_cpp_checks():
    """Read the check counts out of the test binaries' own output rather than remembering them."""
    build = ROOT / "build"
    total, ok = 0, True
    names = []
    for t in ("test_tensor", "test_ops", "test_scheduler", "test_models"):
        binary = build / t
        if not binary.exists():
            return None, False, []
        r = subprocess.run([str(binary)], capture_output=True, text=True)
        ok = ok and r.returncode == 0
        for line in (r.stdout + r.stderr).splitlines():
            if line.startswith(f"{t}: "):
                total += int(line.split()[1])
                names.append(t)
    return total, ok, names


def build_manifest(with_checks: bool):
    generation = json.loads((ROOT / "eval" / "cells" / "BG-1" / "generation.json").read_text())
    reference = json.loads((ROOT / "eval" / "cells" / "BG-1" / "reference.json").read_text())
    uncalibrated = [c for c, v in reference["cells"].items() if v.get("achieved") is None]

    doc = {
        "name": "burnisher",
        "version": "0.1.0",
        "what": ("A native C++/CUDA image and video generation runtime for consumer Blackwell, "
                 "and the instrument that scores changes to it."),
        "status": (
            "v0. A correct, complete, SLOW pipeline plus the harness that scores changes to it. "
            "Speed at launch is not the deliverable; contributors optimize, they do not "
            "bootstrap. The harness is complete and tested. The runtime's CPU reference path is "
            "complete, runs the whole graph end to end and reproduces itself byte for byte. "
            "NOTHING HAS BEEN MEASURED: no Blackwell device was available when this was built, "
            "so every cell's achieved fraction and every cell's noise floor is null, the device "
            "peaks behind every roofline are VENDOR figures rather than probed ones, and the "
            "CUDA backend has never been compiled. docs/STATUS.md is the full list and says "
            "exactly what a first session on the pinned hardware would fill in."),
        "scoring": (
            "Score is the fraction of a cell's REMAINING arithmetic-roofline gap that a change "
            "closes, credited only when it clears that cell's own MEASURED noise floor under a "
            "paired bootstrap at a stated confidence. No letter grades. Latency, peak VRAM and "
            "output fidelity form a frontier, so a change that is faster and hungrier has moved "
            "along it rather than expanded it. Adding a cell nobody had measured is itself a "
            "scored contribution."),
        "generation": {
            "name": generation["name"],
            "model": generation["model"]["repo"],
            "revision": generation["model"]["revision"],
            "device": generation["device"],
            "cells": [c["id"] for c in generation["cells"]],
            "implemented_cells": [c["id"] for c in generation["cells"] if c["implemented"]],
            "uncalibrated_cells": uncalibrated,
            "calibrated": not uncalibrated,
        },
        "files": tracked_files(),
    }
    if with_checks:
        py_count, py_ok = count_python_tests()
        cpp_count, cpp_ok, cpp_names = count_cpp_checks()
        doc["checks"] = {
            "python_tests": py_count, "python_pass": py_ok,
            "cpp_assertions": cpp_count, "cpp_pass": cpp_ok, "cpp_suites": cpp_names,
            "_note": ("Counted from the suites' own output by scripts/manifest.py --write "
                      "--with-checks. These used to be typed, and they drifted."),
        }
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--with-checks", action="store_true",
                    help="also count the tests; requires a built tree")
    args = ap.parse_args()

    path = ROOT / "repo-manifest.json"
    doc = build_manifest(args.with_checks)

    if args.check:
        if not path.exists():
            print("!! repo-manifest.json is missing; run scripts/manifest.py --write",
                  file=sys.stderr)
            return 2
        current = json.loads(path.read_text())
        drift = []
        if current.get("files") != doc["files"]:
            added = sorted(set(doc["files"]) - set(current.get("files", [])))
            removed = sorted(set(current.get("files", [])) - set(doc["files"]))
            for p in added:
                drift.append(f"  + {p}")
            for p in removed:
                drift.append(f"  - {p}")
        if current.get("generation") != doc["generation"]:
            drift.append("  generation block has moved")
        if drift:
            print("!! repo-manifest.json is out of date:", file=sys.stderr)
            for d in drift[:40]:
                print(d, file=sys.stderr)
            print("   regenerate with: scripts/manifest.py --write", file=sys.stderr)
            return 1
        print("ok: repo-manifest.json matches the tree")
        return 0

    if args.write:
        keep = json.loads(path.read_text()).get("checks") if path.exists() else None
        if keep and not args.with_checks:
            doc["checks"] = keep
        path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print(f">> wrote {path} ({len(doc['files'])} tracked files)")
        return 0

    print(json.dumps(doc, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
