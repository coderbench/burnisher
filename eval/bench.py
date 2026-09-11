#!/usr/bin/env python3
"""Paired interleaved base-vs-candidate measurement. The only thing that produces a score.

    burnish bench --generation BG-1 --impl-base stock --impl-candidate fused-adaln \
                  --repeats 5 --output raw.json

Shape of the experiment, and why each part of it is the way it is:

  one process         Base and candidate are two registered implementations of the same op,
                      selected by name, in ONE binary with ONE model load. Two separately
                      linked binaries cannot separate the change from the link, and two
                      processes cannot share a thermal state. This is why the runtime carries a
                      kernel registry instead of letting a contributor replace a file.

  interleaved         base, candidate, base, candidate -- adjacent, never blocked. Clocks
                      cannot be pinned in a container, so absolute numbers drift over minutes.
                      Running one arm to completion and then the other puts the drift between
                      the arms and attributes it to whichever went second.

  held-out shape      Every cell is also run at a shape drawn at evaluation time from the
                      generation's held-out list. A kernel fast only on the benchmarked shape
                      scores nothing.

  gate first          `burnish gate` must have passed. Correctness precedes speed and is never
                      traded against it, and a build that does not reproduce itself cannot be
                      compared to anything.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from burnscore import cells as C
from runner import (GpuLock, RunnerError, device_fingerprint, interleave, parse_result,
                    require_idle_device, require_not_degenerate, require_ran_what_it_claimed,
                    run_once)

ROOT = Path(__file__).resolve().parent.parent


def measure(binary, generation, cell, impl, repeat, *, shape_override=None, timeout=1800):
    """One arm, one cell, one repeat."""
    label = f"{cell.id} {impl} r{repeat}"
    shape = dict(cell.shape)
    if shape_override:
        shape.update(shape_override)
    cmd = [str(binary), "bench",
           "--stage", cell.stage,
           "--dtype", cell.wdtype,
           "--impl", impl,
           "--resolution", str(shape.get("resolution", generation.model["resolution"])),
           "--caption-len", str(shape.get("caption_len", generation.model["caption_len"])),
           "--batch", str(shape.get("batch", generation.model["batch"])),
           "--seed", str(generation.raw.get("seed", 20260911)),
           "--warmup", "3", "--iters", "10"]
    code, out, wall = run_once(cmd, timeout=timeout)
    if code != 0:
        raise RunnerError(f"{label}: the runtime exited {code}\n{out[-4000:]}")
    result = parse_result(out, label)
    require_ran_what_it_claimed(
        result, {"stage": cell.stage, "dtype": cell.wdtype, "impl": impl}, label)
    require_not_degenerate(result, label)
    for key in ("latency_s", "peak_vram_bytes"):
        if key not in result.get("metrics", {}):
            raise RunnerError(f"{label}: the runtime reported no {key}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--generation", default="BG-1")
    ap.add_argument("--impl-base", default="stock",
                    help="the registered implementation the base commit uses")
    ap.add_argument("--impl-candidate", required=True,
                    help="the registered implementation this submission adds")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--cells", nargs="*", help="restrict to these cells (produces a PARTIAL "
                                               "receipt, which credits nothing)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--gate-result", help="the JSON written by `burnish gate`; required, "
                                          "because a build that has not passed the gate must "
                                          "not be timed")
    ap.add_argument("--held-out-seed", type=int,
                    help="fix the held-out shape choice; for reproducing a past run only")
    ap.add_argument("--skip-held-out", action="store_true",
                    help="skip the held-out shapes; the receipt cannot be credited without them")
    args = ap.parse_args()

    generation = C.load(ROOT / "eval" / "cells" / args.generation / "generation.json")

    # Correctness precedes speed, always, and the check is that the gate RAN -- not that
    # somebody remembered to run it.
    if not args.gate_result:
        print("!! --gate-result is required. A submission that has not passed "
              "`burnish gate`\n   must not be timed: correctness precedes speed and is never "
              "traded against it.", file=sys.stderr)
        return 2
    gate = json.loads(Path(args.gate_result).read_text())
    if gate.get("correctness") != "PASS" or gate.get("determinism") is not True:
        print(f"!! the gate did not pass (correctness={gate.get('correctness')}, "
              f"determinism={gate.get('determinism')}). Not timing it.", file=sys.stderr)
        return 2
    if gate.get("impl") != args.impl_candidate:
        print(f"!! the gate was run against impl {gate.get('impl')!r} and this bench is for "
              f"{args.impl_candidate!r}.\n   A gate result for a different implementation is "
              f"not a gate result.", file=sys.stderr)
        return 2

    cells = [generation.cell(c) for c in args.cells] if args.cells \
        else [c for c in generation.cells.values() if c.implemented]

    rng = random.Random(args.held_out_seed if args.held_out_seed is not None
                        else int.from_bytes(Path("/dev/urandom").read_bytes()[:4]
                                            if Path("/dev/urandom").exists() else b"\0\0\0\1",
                                            "big"))
    held_shapes = generation.held_out.get("resolutions") or []
    held_choice = rng.choice(held_shapes) if held_shapes and not args.skip_held_out else None

    records, held_records = [], []
    arms = {"base": args.impl_base, "candidate": args.impl_candidate}
    with GpuLock():
        require_idle_device()
        fp = device_fingerprint()
        print(f">> {fp.get('name')} driver {fp.get('driver_version')}")
        if held_choice:
            print(f">> held-out shape for this run: {held_choice}px "
                  f"(chosen now, after the candidate was frozen)")
        for cell in cells:
            for repeat, variant in interleave(("base", "candidate"), args.repeats):
                r = measure(args.binary, generation, cell, arms[variant], repeat)
                records.append({"cell": cell.id, "variant": variant, "repeat": repeat,
                                "config_id": "default", "status": "OK",
                                "impl": arms[variant], "metrics": r["metrics"],
                                "effective": r.get("effective"),
                                "output_stats": r.get("output_stats")})
                print(f"   {cell.id:26s} {variant:9s} r{repeat} "
                      f"{r['metrics']['latency_s'] * 1e3:8.3f} ms")
            if held_choice:
                for repeat, variant in interleave(("base", "candidate"), max(2, args.repeats // 2)):
                    r = measure(args.binary, generation, cell, arms[variant], repeat,
                                shape_override={"resolution": held_choice})
                    held_records.append({"cell": cell.id, "variant": variant, "repeat": repeat,
                                         "config_id": f"held-{held_choice}", "status": "OK",
                                         "metrics": r["metrics"]})

    doc = {
        "generation": args.generation,
        "records": records,
        "held_out": held_records or None,
        "held_out_shape": held_choice,
        "correctness": gate.get("correctness"),
        "determinism": gate.get("determinism"),
        "provenance": {
            "device": fp, "impl_base": args.impl_base, "impl_candidate": args.impl_candidate,
            "base_commit": gate.get("base_commit"),
            "candidate_commit": gate.get("candidate_commit"),
            "instrument_from": gate.get("instrument_from"),
            "repeats": args.repeats,
            "_pairing": "interleaved base/candidate, one process, one model load",
        },
    }
    Path(args.output).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f">> wrote {args.output} ({len(records)} records"
          + (f", {len(held_records)} held-out" if held_records else "") + ")")
    if not held_records:
        print("   NOTE: no held-out shapes were run, so the shape-overfit guard did not fire.\n"
              "         `burnish score` will report the guard as not run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
