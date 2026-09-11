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
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from burnscore import cells as C
from paths import add_argument as add_cells_root_arg, generation_path
from runner import (GpuLock, RunnerError, device_fingerprint, interleave, parse_result,
                    require_idle_device, require_not_degenerate, require_ran_what_it_claimed,
                    run_once)

ROOT = Path(__file__).resolve().parent.parent


def measure(binary, generation, cell, impl, repeat, *, shape_override=None, timeout=1800,
            fidelity=None):
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
    # Output fidelity is a property of the BUILD, not of a cell or a repeat, so it comes from
    # that arm's gate result rather than from the timing run. It has to be attached HERE, though:
    # the generation declares it as a frontier objective, and a record missing a declared
    # objective produces no operating point at all -- so the frontier would silently come out as
    # exactly zero for both arms and every result would read MOVED_ALONG_FRONTIER.
    if fidelity is not None:
        result["metrics"]["latent_l2_vs_reference"] = float(fidelity)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--generation", default="BG-1")
    add_cells_root_arg(ap)
    ap.add_argument("--impl-base", default="stock",
                    help="the registered implementation the base commit uses")
    ap.add_argument("--impl-candidate", required=True,
                    help="the registered implementation this submission adds")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--cells", nargs="*", help="restrict to these cells (produces a PARTIAL "
                                               "receipt, which credits nothing)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--gate-result", help="the JSON written by `burnish gate` for the CANDIDATE "
                                          "implementation; required, because a build that has "
                                          "not passed the gate must not be timed")
    ap.add_argument("--gate-base-result",
                    help="the gate JSON for the BASE implementation. Required for the same "
                         "reason and for one more: output fidelity is a scored frontier "
                         "objective and it is measured by the gate, so without the base's gate "
                         "there is nothing to compare the candidate's fidelity against.")
    ap.add_argument("--held-out-seed", type=int,
                    help="fix the held-out shape choice; for reproducing a past run only")
    ap.add_argument("--skip-held-out", action="store_true",
                    help="skip the held-out shapes; the receipt cannot be credited without them")
    args = ap.parse_args()

    generation = C.load(generation_path(args.generation, args.cells_root))

    # Correctness precedes speed, always, and the check is that the gate RAN -- not that
    # somebody remembered to run it.
    if not args.gate_result:
        print("!! --gate-result is required. A submission that has not passed "
              "`burnish gate`\n   must not be timed: correctness precedes speed and is never "
              "traded against it.", file=sys.stderr)
        return 2
    if not args.gate_base_result:
        print("!! --gate-base-result is required.\n"
              "   `latent_l2_vs_reference` is a scored frontier objective and the gate is what "
              "measures\n   it. Without the base arm's gate there is nothing to compare the "
              "candidate against, and\n   the frontier would come out as exactly zero for both "
              "arms -- which reads as a\n   quality-neutral result rather than as a missing "
              "measurement.", file=sys.stderr)
        return 2
    gate = json.loads(Path(args.gate_result).read_text())
    gate_base = json.loads(Path(args.gate_base_result).read_text())
    if gate_base.get("correctness") != "PASS" or gate_base.get("determinism") is not True:
        print(f"!! the BASE arm's gate did not pass "
              f"(correctness={gate_base.get('correctness')}, "
              f"determinism={gate_base.get('determinism')}).\n"
              f"   The base is the thing everything is measured against; if it does not "
              f"reproduce itself\n   or does not match the reference, nothing downstream means "
              f"anything.", file=sys.stderr)
        return 2
    if gate_base.get("impl") != args.impl_base:
        print(f"!! the base gate was run against impl {gate_base.get('impl')!r} and this bench "
              f"uses {args.impl_base!r}.", file=sys.stderr)
        return 2
    fidelity = {"base": gate_base.get("worst_relative_l2"),
                "candidate": gate.get("worst_relative_l2")}
    missing = [k for k, v in fidelity.items() if v is None]
    if missing:
        print(f"!! the gate result for {', '.join(missing)} carries no `worst_relative_l2`.\n"
              f"   That field is the fidelity objective. A gate run with --determinism-only "
              f"does not\n   produce it, and scoring without it would hand the frontier a "
              f"missing dimension.", file=sys.stderr)
        return 2
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

    # os.urandom, not a read of /dev/urandom: `Path("/dev/urandom").read_bytes()` reads the whole
    # "file", and that stream never ends. The held-out shape has to be unpredictable to the
    # candidate -- that is what makes the guard a guard -- so the seed is fixable only for
    # reproducing a past run.
    rng = random.Random(args.held_out_seed if args.held_out_seed is not None
                        else int.from_bytes(os.urandom(4), "big"))
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
                r = measure(args.binary, generation, cell, arms[variant], repeat,
                            fidelity=fidelity[variant])
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
                                shape_override={"resolution": held_choice},
                                fidelity=fidelity[variant])
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
            "fidelity": fidelity,
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
