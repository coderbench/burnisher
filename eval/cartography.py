#!/usr/bin/env python3
"""Evaluate a submission that opens a new cell, rather than one that closes a gap.

    burnish cartography check --generation BG-2 --base origin/main
    burnish cartography check --generation BG-2 --base origin/main --measure  [GPU]

Why this is a different evaluation, not a special case of the other one
-----------------------------------------------------------------------

A speedup submission is asked "did this get faster, beyond the noise?". A cartography submission
adds a place where future work can be measured, and the question is "is this cell real, and can
anybody be credited on it?" -- which no amount of paired benching answers.

`docs/CARTOGRAPHY.md` names four things a cell must come with: the geometry, a reference
implementation that passes the gate, the roofline, and the calibration. This checks all four, and
the checking is deliberately asymmetric about who supplies what:

    the submission supplies    the geometry and the ORACLE -- the cell definition and the
                               reference latents a correct runtime must reproduce.

    the evaluator supplies     every MEASUREMENT. The submitted achieved fractions and noise
                               floors are read, reported, and then thrown away; the cell is
                               recalibrated here. A cartography submission that arrived with a
                               fabricated floor of 0.0001% would gain nothing by it.

That asymmetry is the whole security argument. The one thing it does not settle is whether the
submitted reference latents really came from the pinned reference implementation rather than from
the submitter's own runtime -- against which the gate would pass trivially. Nothing this
evaluator can compute settles that either; it is settled the same way every other measurement
claim here is settled, by being reproducible and challengeable. `docs/CORRECTNESS.md` has the
procedure, and the manifest records what produced them.

What this deliberately does not accept
--------------------------------------

A new MODEL needs a new op enumeration in `eval/burnscore/geometry.py`, which is instrument, and
the guard blocks it -- correctly, because a geometry that miscounts a stage moves every ceiling
computed from it. That path needs a maintainer and is not automatable. A new resolution, dtype or
stage of a model already enumerated needs no code at all, and is.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from burnscore import cells as C
from burnscore.floor import resolution_gate
from make_generation import local_ceilings
from paths import add_argument as add_cells_root_arg, cells_root

ROOT = Path(__file__).resolve().parent.parent

# How closely a submitted ceiling must match the one recomputed from the base configs.
#
# Exactly, to floating-point noise. A ceiling is a deterministic function of frozen geometry and
# a probed device peak, so the only reasons for it to differ are a different device profile or a
# hand-edited number -- and the second is a submission choosing its own denominator.
CEILING_TOLERANCE = 1e-9


class CartographyError(ValueError):
    """A proposed cell cannot be accepted as opened."""


def _base_generations(base, repo=None) -> set:
    out = subprocess.run(["git", "-C", str(repo or ROOT), "ls-tree", "-d", "--name-only",
                          f"{base}:eval/cells"], capture_output=True, text=True)
    return set(out.stdout.split()) if out.returncode == 0 else set()


def _existing_cell_ids(base, repo=None) -> set:
    """Every cell id any generation on the base already declares."""
    ids = set()
    for gen in _base_generations(base, repo):
        out = subprocess.run(["git", "-C", str(repo or ROOT), "show",
                              f"{base}:eval/cells/{gen}/generation.json"],
                             capture_output=True, text=True)
        if out.returncode != 0:
            continue
        try:
            ids |= {c["id"] for c in json.loads(out.stdout).get("cells", [])}
        except (ValueError, KeyError):
            continue
    return ids


def check(generation_name, *, base, repo=None, root=None, verbose=True) -> dict:
    checks = []

    def ok(name, passed, detail=""):
        checks.append({"check": name, "pass": bool(passed), "detail": detail})
        if verbose:
            print(f"   [{'ok' if passed else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
        return passed

    gdir = Path(root or cells_root(None)) / generation_name
    if not (gdir / "generation.json").exists():
        raise CartographyError(f"{gdir}/generation.json does not exist")
    gen = C.load(gdir / "generation.json")
    doc = gen.raw

    # 1. It is genuinely NEW. An added generation is safe precisely because it cannot change what
    #    an existing receipt meant; one that already exists on the base is an edit wearing a new
    #    name, and edits to frozen generations silently re-score history.
    base_gens = _base_generations(base, repo)
    ok("the generation does not already exist on the base", generation_name not in base_gens,
       f"base has {', '.join(sorted(base_gens)) or 'none'}")

    # 2. It opens something. A "new" generation whose cells all exist already adds no surface --
    #    it re-measures ground the map already covers, and pays for cartography without doing any.
    existing = _existing_cell_ids(base, repo)
    proposed = {c["id"] for c in doc.get("cells", [])}
    novel = sorted(proposed - existing)
    ok("it declares at least one cell that does not already exist", bool(novel),
       f"new: {', '.join(novel) or 'none'}")

    # 3. The ceiling is recomputed here, from the BASE configs, and must match what was submitted.
    #    A submission that could pick its own ceiling could pick its own denominator, and every
    #    gap-closed score in the cell forever after would be measured against it.
    recomputed = local_ceilings(doc)
    mismatched = []
    for c in doc.get("cells", []):
        want = (recomputed.get(c["id"]) or {}).get("ceiling_seconds")
        got = c.get("ceiling_seconds")
        if want is None or got is None:
            continue
        if abs(want - got) > max(CEILING_TOLERANCE, abs(want) * CEILING_TOLERANCE):
            mismatched.append(f"{c['id']}: submitted {got:.9g}, recomputed {want:.9g}")
    ok("every submitted ceiling recomputes from the base configs", not mismatched,
       "; ".join(mismatched))

    # 4. The oracle exists. Without reference latents the correctness gate has nothing to compare
    #    against, and a cell whose correctness cannot be gated is a cell where a wrong answer
    #    scores.
    refs = gdir / "reference-latents"
    manifest = refs / "manifest.json"
    ok("reference latents are present with a manifest", manifest.exists(),
       str(refs) if manifest.exists() else f"{refs} has no manifest.json")

    # 5. The submission's own calibration is read and then ignored. Reported so a reviewer can see
    #    what was claimed, and compared against the evaluator's own measurement when --measure
    #    ran. A cell arriving with a fabricated floor gains nothing by it.
    claimed = {}
    ref_json = gdir / "reference.json"
    if ref_json.exists():
        claimed = (json.loads(ref_json.read_text()).get("cells") or {})
    if claimed and verbose:
        print(f"   [--] the submission claims a calibration for {len(claimed)} cell(s). "
              f"It is not used.")

    return {"generation": generation_name, "novel_cells": novel,
            "claimed_calibration": {k: {"achieved": v.get("achieved"),
                                        "floor_pct": v.get("floor_pct")}
                                    for k, v in claimed.items()},
            "checks": checks, "pass": all(c["pass"] for c in checks),
            "_measurement_note": (
                "These checks are arithmetic and structural. Whether the cell can actually be "
                "RUN, reproduces its oracle, and resolves against its own measured floor is "
                "settled by --measure, which gates and recalibrates it on this box."),
            }


class MeasurementError(RuntimeError):
    """The proposed cell could not be run, gated, or calibrated on this box."""


def measure(generation_name, *, root, binary, weights, noise, device="cuda", repeats=9) -> dict:
    """Gate and calibrate the proposed cell HERE. The submission's own numbers are not used.

    This is the half of a cartography evaluation that cannot be faked by the submitter, and it is
    the reason the other half can afford to be permissive. They supply the cell definition and the
    oracle; every number that ends up frozen into the generation is measured on the evaluator's
    box, by the evaluator's calibrator, from the evaluator's probe.

    A cell that does not run, does not reproduce its oracle, or does not reproduce itself, fails
    here -- before anything is published and before anybody is paid for a map of somewhere that
    does not exist.
    """
    for need, what in ((binary, "--binary"), (weights, "--weights"), (noise, "--noise")):
        if not need:
            raise MeasurementError(f"{what} is required to measure a proposed cell")
    gdir = Path(root) / generation_name

    print(f"\n>> gating {generation_name} on this box")
    gate = subprocess.run(
        [sys.executable, str(ROOT / "eval" / "gate.py"), "--binary", str(binary),
         "--weights", str(weights), "--impl", "cuda", "--device", device,
         "--dtype", "fp32", "--noise", str(noise),
         "--reference", str(gdir / "reference-latents"),
         "--generation", generation_name, "--cells-root", str(root),
         "--repeats", "2", "--output", str(gdir / "_gate.json")],
        capture_output=True, text=True, timeout=7200)
    print(gate.stdout[-1200:])
    if gate.returncode != 0:
        raise MeasurementError(
            "the proposed cell does not pass the correctness gate on this box. A cell whose "
            "correctness cannot be gated is a cell where a wrong answer scores.\n"
            + (gate.stdout + gate.stderr)[-1500:])

    print(f">> calibrating {generation_name} on this box "
          f"(the submission's own calibration is discarded)")
    cal = subprocess.run(
        [sys.executable, str(ROOT / "eval" / "calibrate.py"), "--binary", str(binary),
         "--weights", str(weights), "--impl", "cuda", "--device", device,
         "--generation", generation_name, "--cells-root", str(root),
         "--repeats", str(repeats), "--output", str(gdir / "_calibration.json")],
        capture_output=True, text=True, timeout=14400)
    print(cal.stdout[-2000:])
    if cal.returncode != 0:
        raise MeasurementError("the proposed cell could not be calibrated on this box.\n"
                               + (cal.stdout + cal.stderr)[-1500:])

    doc = json.loads((gdir / "_calibration.json").read_text())
    out = {}
    for cid, c in (doc.get("cells") or {}).items():
        g = resolution_gate(c["floor_pct"], c["achieved"])
        out[cid] = {"achieved": c["achieved"], "floor_pct": c["floor_pct"],
                    "floors_of_room": g["floors_of_room"], "resolvable": g["resolvable"],
                    "measured_by": "the evaluator, not the submission"}
    return out


def verdict_for(result, measured=None) -> dict:
    """Whether the cell is opened, and on what terms.

    Three outcomes, and the middle one is the interesting one: a cell that runs correctly and
    CANNOT resolve a contribution is a successful cartography result, not a failed one. It is
    worth more than a cell that looks open and is not, because the alternative is somebody
    spending a week inside a noise floor. `docs/CARTOGRAPHY.md` says so and this agrees with it.
    """
    if not result["pass"]:
        return {"outcome": "REJECTED", "pays": False,
                "why": "the proposed cell did not pass the structural checks"}
    if measured is None:
        return {"outcome": "UNMEASURED", "pays": False,
                "why": ("structurally sound, and nothing has been run. A cell is opened by "
                        "measurement, not by declaration -- re-run with --measure on the "
                        "pinned hardware.")}
    unresolvable = [c for c, m in measured.items() if not m.get("resolvable")]
    return {
        "outcome": "OPENED", "pays": True,
        "unresolvable_cells": unresolvable,
        "why": ("the cell runs, reproduces its oracle, and was calibrated here"
                + (f". {len(unresolvable)} of {len(measured)} cell(s) cannot resolve a "
                   f"contribution at their measured floor, which is published as a result "
                   f"rather than as a failure" if unresolvable else "")),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["check"])
    ap.add_argument("--generation", required=True)
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--repo", help="the checkout to diff against the base")
    ap.add_argument("--measure", action="store_true",
                    help="gate and recalibrate the cell on this box [needs a GPU]")
    ap.add_argument("--binary")
    ap.add_argument("--weights")
    ap.add_argument("--noise")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repeats", type=int, default=9)
    ap.add_argument("--json")
    add_cells_root_arg(ap)
    a = ap.parse_args()

    print(f">> cartography check: {a.generation}")
    try:
        r = check(a.generation, base=a.base, repo=a.repo,
                  root=cells_root(a.cells_root))
    except CartographyError as exc:
        print(f"!! {exc}", file=sys.stderr)
        return 2
    measured = None
    if a.measure and r["pass"]:
        try:
            measured = measure(a.generation, root=cells_root(a.cells_root), binary=a.binary,
                               weights=a.weights, noise=a.noise, device=a.device,
                               repeats=a.repeats)
            r["measured"] = measured
        except MeasurementError as exc:
            print(f"\n!! {exc}", file=sys.stderr)
            r["measurement_error"] = str(exc)
            r["pass"] = False
    v = verdict_for(r, measured)
    r["verdict"] = v
    print()
    print(f"   {v['outcome']}: {v['why']}")
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1, sort_keys=True) + "\n")
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
