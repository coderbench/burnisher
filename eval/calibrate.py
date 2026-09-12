#!/usr/bin/env python3
"""Measure what a config file cannot: how full each cell is, and how noisy it is.

    burnish calibrate --generation BG-1 --repeats 9 --write

This is the command that turns `eval/cells/BG-N/reference.json` from a file full of nulls into
the thing every score depends on. It produces exactly two numbers per cell and both are
measurements:

    achieved   ceiling / measured. The DENOMINATOR of every gap-closed score in that cell.
    floor_pct  the run-to-run spread of the measurement procedure itself.

**The floor is measured by running the base against ITSELF.** Two arms, both the unmodified
base, interleaved exactly the way a scored comparison runs, with every guard a scored comparison
uses. The paired ratios should centre on 1.0; how far they wander is the floor. Measuring it any
other way -- a quieter loop, fewer guards, a different process layout -- measures the noise of a
procedure nobody will use.

**Nothing here may be estimated.** If the device is absent this command fails. A calibration is
the one thing in the system that cannot be derived, and a guessed one would make every receipt
downstream a fiction with a checksum on it.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from burnscore import cells as C
from burnscore.floor import measure_floor, resolution_gate
from bench import measure
from paths import add_argument as add_cells_root_arg, generation_path
from runner import GpuLock, RunnerError, device_fingerprint, interleave, require_idle_device

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--generation", default="BG-1")
    add_cells_root_arg(ap)
    ap.add_argument("--impl", default="stock", help="the base implementation, run as both arms")
    ap.add_argument("--repeats", type=int, default=9,
                    help="paired control-vs-control repeats. Nine, not three: a floor estimated "
                         "from three pairs is itself noisy, and it is the number every later "
                         "credit decision is compared against.")
    ap.add_argument("--timer-digits", type=int, default=6,
                    help="significant digits the runtime prints a duration to; sets the "
                         "instrument-resolution term of the floor")
    ap.add_argument("--cells", nargs="*")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--weights", help="checkpoint directory; omit for synthetic weights")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5,
                    help="timed iterations per invocation, whose MEDIAN is the reading. The "
                         "floor comes from the spread ACROSS invocations, not within one, so "
                         "this trades a little within-run stability for more paired repeats -- "
                         "and the repeats are what the floor is made of.")
    # --write updates the committed reference.json, which is the REFERENCE DEVICE's calibration
    # and part of the instrument. A validator calibrating their own box wants --output instead:
    # their calibration belongs beside their ledger, not in a repository everybody shares.
    ap.add_argument("--write", action="store_true",
                    help="update the committed reference.json in place. This is the reference "
                         "device's calibration and part of the instrument -- if you are a "
                         "validator calibrating your own box, use --output.")
    ap.add_argument("--merge", metavar="FILE",
                    help="fold this calibration into an earlier one, keeping the WORST floor "
                         "per cell. A floor is a spread statistic and a spread estimated from "
                         "nine repeats is not a stable number -- measured twice on one card, "
                         "hours apart, these floors moved by up to 24x while the achieved "
                         "fractions held to three figures. Taking the worst can only refuse a "
                         "real gain that was too small to see; taking the best would credit "
                         "noise, and only one of those two errors is recoverable.")
    ap.add_argument("--output",
                    help="write this box's calibration here, to pass to `burnish score "
                         "--calibration`. Keep it beside your ledger: it describes YOUR card, "
                         "and a calibration used on a different one biases every score it "
                         "produces in the same direction.")
    args = ap.parse_args()

    gpath = generation_path(args.generation, args.cells_root)
    generation = C.load(gpath)
    cells = [generation.cell(c) for c in args.cells] if args.cells \
        else [c for c in generation.cells.values() if c.implemented]

    out_cells = {}
    with GpuLock():
        require_idle_device()
        fp = device_fingerprint()
        print(f">> calibrating {generation.name} on {fp.get('name')} "
              f"driver {fp.get('driver_version')}")
        print(f">> {args.repeats} paired control-vs-control repeats per cell\n")
        # The ceilings for THIS box, recomputed from the local probe rather than taken from the
        # frozen generation. A ceiling is max(flops/peak, bytes/bandwidth): the geometry is
        # frozen, the peaks are whatever card is in this machine. Freezing the two together is
        # what made a score depend on which validator ran it -- `achieved` became one card's
        # ceiling over another card's measurement, a systematic ~6% bias between two RTX 5090s.
        from make_generation import local_ceilings
        local = local_ceilings(generation.raw)
        for cid, c in local.items():
            if cid in generation.cells and c["ceiling_seconds"]:
                generation.cells[cid].ceiling_seconds = c["ceiling_seconds"]
        for cell in cells:
            arm_a, arm_b, vram = [], [], []
            for repeat, which in interleave(("a", "b"), args.repeats):
                r = measure(args.binary, generation, cell, args.impl, repeat,
                            device=args.device, weights=args.weights,
                            warmup=args.warmup, iters=args.iters)
                (arm_a if which == "a" else arm_b).append(r["metrics"]["latency_s"])
                vram.append(r["metrics"]["peak_vram_bytes"])
            floor = measure_floor(cell.id, arm_a, arm_b, reported_digits=args.timer_digits)
            measured = statistics.median(arm_a + arm_b)
            achieved = cell.ceiling_seconds / measured
            if achieved > 1.0:
                print(f"!! {cell.id}: measured {measured * 1e3:.3f} ms BEATS the arithmetic "
                      f"ceiling {cell.ceiling_seconds * 1e3:.3f} ms.\n"
                      f"   The geometry undercounts the work, the device peak is overstated, "
                      f"or the run\n   did not do what it claimed. Not writing this cell.",
                      file=sys.stderr)
                continue
            gate = resolution_gate(floor.floor_pct, achieved)
            out_cells[cell.id] = {
                "achieved": achieved,
                "measured_seconds": measured,
                "ceiling_seconds": cell.ceiling_seconds,
                "ceiling_peak_basis": (local.get(cell.id) or {}).get("peak_basis"),
                "ceiling_bound_by": (local.get(cell.id) or {}).get("bound_by"),
                "floor_pct": floor.floor_pct,
                "floor_repeats": floor.repeats,
                "floor_decided_by": floor.decided_by,
                "floor_spread_pct": floor.spread_pct,
                "floor_resolution_pct": floor.resolution_pct,
                "floor_median_ratio": floor.median_ratio,
                "peak_vram_bytes": max(vram),
                "resolvable": gate["resolvable"],
                "floors_of_room": gate["floors_of_room"],
                "basis": "measured",
            }
            flag = "" if gate["resolvable"] else "   <-- UNRESOLVABLE"
            print(f"   {cell.id:26s} achieved {achieved:6.1%}  floor {floor.floor_pct:6.3f}%  "
                  f"room {gate['floors_of_room']:7.1f} floors{flag}")
            if not gate["resolvable"]:
                print(f"       {gate['verdict']}")

    sessions = 1
    if args.merge:
        prev = json.loads(Path(args.merge).read_text())
        prev_probe = (prev.get("device_probe") or {})
        if prev_probe.get("uuid") and fp.get("uuid") and prev_probe["uuid"] != fp["uuid"]:
            print(f"!! --merge names a calibration of {prev_probe['uuid']} and this box is "
                  f"{fp['uuid']}.\n   Pooling two cards' floors would describe neither. "
                  f"Calibrate each box separately.", file=sys.stderr)
            return 2
        sessions = int(prev.get("calibration_sessions", 1)) + 1
        for cid, prior in (prev.get("cells") or {}).items():
            now = out_cells.get(cid)
            if not now or prior.get("floor_pct") is None:
                continue
            if prior["floor_pct"] > now["floor_pct"]:
                # The WORST floor across sessions, deliberately. A floor is a spread statistic,
                # and one estimated from nine repeats carries large uncertainty -- measured
                # twice on one card, hours apart, these moved by up to 24x while the achieved
                # fractions held to three figures. Which session a validator happened to
                # calibrate in would otherwise decide what counts as resolved.
                #
                # Worst rather than mean because the two errors are not symmetric: too tight
                # credits noise as a contribution and the ledger compounds it permanently; too
                # loose refuses a gain too small to see, and the contributor can come back with
                # a bigger one. Only one of those is recoverable.
                now["floor_pct"] = prior["floor_pct"]
                now["floor_decided_by"] = "worst-of-sessions"
                now["floor_sessions_max_from"] = prior.get("_session") or "earlier session"

    doc = {
        "generation": generation.name,
        "calibration_sessions": sessions,
        "_floor_stability": (
            "A floor is a spread statistic and is not stable between sessions: measured twice "
            "on one RTX 5090, hours apart, the per-cell floors here moved by up to 24x while "
            "the achieved fractions held to three significant figures. That is expected -- a "
            "median is robust and a spread over nine repeats is not -- but it means whether a "
            "submission RESOLVES can depend on which session the box was calibrated in. "
            "`--merge` folds sessions together keeping the worst floor per cell, which can only "
            "refuse a gain too small to see rather than credit noise as a contribution."),
        "device": args.device,
        "weights": args.weights or "synthetic",
        "_what_this_is": ("Per-cell calibration: the fraction of the arithmetic ceiling "
                          "currently achieved, and the measured run-to-run noise floor. Both "
                          "are measurements."),
        "device_probe": fp,
        "calibrated_with": {"impl": args.impl, "repeats": args.repeats,
                            "timer_digits": args.timer_digits,
                            "warmup": args.warmup, "iters": args.iters},
        "_floor_method": ("Two arms, both the unmodified base, interleaved with every guard a "
                          "scored comparison uses. The floor is the larger of the paired "
                          "control-vs-control spread and the instrument's own resolution."),
        "cells": out_cells,
    }
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text)
        print(f"\n>> wrote {args.output}")
    if args.write:
        ref = gpath.parent / "reference.json"
        # A calibration REPLACES the previous one rather than merging into it: a reference.json
        # holding cells measured in two different sessions describes a box that never existed.
        ref.write_text(text)
        print(f"\n>> wrote {ref}")
        print("   Every gap-closed score in this generation now has a measured denominator.")
    if not (args.output or args.write):
        print(text)
    unresolvable = [c for c, v in out_cells.items() if not v["resolvable"]]
    if unresolvable:
        print(f"\n!! {len(unresolvable)} cell(s) cannot resolve a twentieth of their own room: "
              f"{', '.join(unresolvable)}.\n"
              f"   An axis whose room sits inside its own noise is OPEN, not solved. Publish "
              f"them as\n   unresolvable, quieten them, or drop them -- do not publish them as "
              f"places to work.")
    return 0


# A measurement command that cannot measure must say so in a sentence, not a stack trace.
#
# These runners are invoked three ways -- by hand, by `tools/burnish` (which catches this), and
# by another runner as a subprocess. The third is why the handler lives here: `cartography.py`
# shells out to this file, and a traceback from inside `require_idle_device` arrives as a wall
# of Python in a pull request comment instead of "this box has no GPU".
#
# Exit 3 rather than 1, so a caller can tell "there is no device" from "the thing I measured
# failed" -- those need different responses and conflating them makes a missing driver look
# like a rejected submission.
EXIT_NO_DEVICE = 3

if __name__ == "__main__":
    try:
        sys.exit(main())
    except RunnerError as exc:
        print(f"!! {exc}", file=sys.stderr)
        sys.exit(EXIT_NO_DEVICE)
