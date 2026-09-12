#!/usr/bin/env python3
"""Paired interleaved base-vs-candidate measurement. The only thing that produces a score.

    burnish bench --generation BG-1 --impl-base stock --impl-candidate fused-adaln \
                  --repeats 3 --output raw.json

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


def instrument_settings(generation):
    """The warmup and iteration counts the cell's NOISE FLOOR was measured with.

    Not a default and not a tuning knob. A floor is the run-to-run spread of one particular
    measurement procedure: the median of `iters` timed invocations after `warmup` untimed ones.
    Average over more invocations and the measurement gets quieter than the floor describes;
    average over fewer and it gets noisier. Either way the effect and the floor it is judged
    against come from two different instruments, and the comparison means nothing.

    This was wrong in the cheap direction and the expensive one at once: the floors were
    calibrated at warmup 2 / iters 5 while `measure()` had 3 / 10 hardcoded with no flag. Every
    scored run therefore did 13 invocations per record where 7 would do -- roughly twice the GPU
    time -- to produce a number quieter than the floor it was compared with.
    """
    w, i = generation.calibrated_warmup, generation.calibrated_iters
    if w is None or i is None:
        raise RunnerError(
            f"{generation.name} has no record of the warmup and iteration counts its noise "
            f"floors were calibrated with, so a measurement cannot be made with the same "
            f"instrument that measured the noise. Re-run `burnish calibrate`, which writes "
            f"`calibrated_with` into reference.json.")
    return int(w), int(i)


def measure(binary, generation, cell, impl, repeat, *, shape_override=None, timeout=1800,
            fidelity=None, device="cuda", weights=None, warmup=None, iters=None):
    """One arm, one cell, one repeat.

    `warmup` and `iters` default to whatever the floor was calibrated with; passing something
    else is for calibration itself, which is the run that DEFINES them.
    """
    if warmup is None or iters is None:
        cw, ci = instrument_settings(generation)
        warmup = cw if warmup is None else warmup
        iters = ci if iters is None else iters
    label = f"{cell.id} {impl} r{repeat}"
    shape = dict(cell.shape)
    if shape_override:
        shape.update(shape_override)
    cmd = [str(binary), "bench",
           "--stage", cell.stage,
           "--dtype", cell.wdtype,
           "--impl", impl,
           "--device", device,
           "--resolution", str(shape.get("resolution", generation.model["resolution"])),
           "--caption-len", str(shape.get("caption_len", generation.model["caption_len"])),
           "--batch", str(shape.get("batch", generation.model["batch"])),
           "--seed", str(generation.raw.get("seed", 20260911)),
           "--warmup", str(warmup), "--iters", str(iters)] + (["--weights", str(weights)] if weights else [])
    code, out, wall = run_once(cmd, timeout=timeout)
    if code != 0:
        raise RunnerError(f"{label}: the runtime exited {code}\n{out[-4000:]}")
    result = parse_result(out, label)
    require_ran_what_it_claimed(
        result, {"stage": cell.stage, "dtype": cell.wdtype, "impl": impl, "device": device},
        label)
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
    # What the record COST, alongside what it measured. `run_once` already has it and was
    # throwing it away. It is not a metric -- nothing is scored on it -- but the gap between it
    # and `latency_s` is the evaluator's own overhead, and a subnet that cannot say what
    # scoring costs cannot tell whether it can afford the submissions it is asking for.
    result["wall_s"] = wall
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
    # Default None, resolved from the generation after it is loaded. A hardcoded default here
    # is a second opinion about a number the frozen generation already declares, and the two
    # drift the moment either moves.
    ap.add_argument("--repeats", type=int, default=None,
                    help="paired repeats per cell; defaults to what the generation declares, "
                         "and the scorer refuses fewer")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--weights", help="checkpoint directory; omit for synthetic weights, which "
                                      "time the same kernels on the same shapes")
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
    if args.repeats is None:
        args.repeats = generation.repeats

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
        warmup, iters = instrument_settings(generation)
        print(f">> {fp.get('name')} driver {fp.get('driver_version')}")
        print(f">> instrument: {warmup} warmup + {iters} timed invocations per record, "
              f"{args.repeats} paired repeats -- the settings the noise floors were "
              f"calibrated with")
        if held_choice:
            print(f">> held-out shape for this run: {held_choice}px "
                  f"(chosen now, after the candidate was frozen)")
        for cell in cells:
            for repeat, variant in interleave(("base", "candidate"), args.repeats):
                r = measure(args.binary, generation, cell, arms[variant], repeat,
                            fidelity=fidelity[variant], device=args.device,
                            weights=args.weights)
                records.append({"cell": cell.id, "variant": variant, "repeat": repeat,
                                "config_id": "default", "status": "OK",
                                "impl": arms[variant], "metrics": r["metrics"],
                                # What this record cost the evaluator, as opposed to what it
                                # measured. The difference is process startup and the
                                # checkpoint map, and it is most of why a scoring run takes
                                # longer than the arithmetic says. Recorded rather than
                                # estimated, because the cost of scoring is a property of the
                                # subnet worth knowing and "never type a benchmark number by
                                # hand" applies to the harness's own bill too.
                                "wall_s": r.get("wall_s"),
                                "effective": r.get("effective"),
                                "output_stats": r.get("output_stats")})
                print(f"   {cell.id:26s} {variant:9s} r{repeat} "
                      f"{r['metrics']['latency_s'] * 1e3:8.3f} ms")
            if held_choice:
                for repeat, variant in interleave(("base", "candidate"), max(2, args.repeats // 2)):
                    r = measure(args.binary, generation, cell, arms[variant], repeat,
                                shape_override={"resolution": held_choice},
                                fidelity=fidelity[variant], device=args.device,
                                weights=args.weights)
                    held_records.append({"cell": cell.id, "variant": variant, "repeat": repeat,
                                         "config_id": f"held-{held_choice}", "status": "OK",
                                         "wall_s": r.get("wall_s"),
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
            "warmup": warmup,
            "iters": iters,
            "_instrument_note": ("warmup and iters are the counts the cell noise floors were "
                                 "calibrated with. A measurement averaged over a different "
                                 "number of invocations than its floor was is a different "
                                 "instrument, and the comparison does not mean anything."),
            "wall_seconds_total": round(sum(r.get("wall_s") or 0 for r in records)
                                        + sum(r.get("wall_s") or 0 for r in held_records), 1),
            "_pairing": "interleaved base/candidate, one process per record",
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
