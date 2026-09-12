#!/usr/bin/env python3
"""What a narrower dtype actually buys on this box, measured rather than assumed.

    python3 scripts/dtype_latency.py --binary build-cuda/burnisher --weights /workspace/ckpt \
        --output eval/cells/BG-1/dtype-latency.json

The roofline says a DiT step at fp32 must move twice the weight bytes of one at bf16, so a
bandwidth-bound step should take about twice as long. Whether it DOES is the question, and it is
the question that decides what the fp8 and NVFP4 cells are worth: a narrower weight only pays
where the wide path is limited by the width. If halving the bytes buys nothing, the path is
limited by something else, and quantizing it moves the published ceiling down while moving the
measurement not at all.

Paired and interleaved for the same reason every other measurement here is -- clocks cannot be
pinned in a container, so fp32, bf16, fp32, bf16 puts any drift across BOTH arms instead of
between them. This is a ratio between two dtypes of the same implementation, not a score: it
never enters the ledger, and it is written to its own artifact so that scripts/make_issues.py
can quote it without anybody typing it.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from runner import (GpuLock, RunnerError, device_fingerprint, interleave, parse_result,
                    require_idle_device, require_not_degenerate, require_ran_what_it_claimed,
                    run_once)


def one(binary, weights, dtype, impl, *, warmup, iters, seed):
    cmd = [str(binary), "bench", "--stage", "dit-step", "--dtype", dtype, "--impl", impl,
           "--device", "cuda", "--resolution", "1024", "--caption-len", "300", "--batch", "2",
           "--seed", str(seed), "--warmup", str(warmup), "--iters", str(iters),
           "--weights", str(weights)]
    code, out, _ = run_once(cmd)
    if code != 0:
        raise RunnerError(f"dit-step {dtype}: the runtime exited {code}\n{out[-4000:]}")
    r = parse_result(out, f"dit-step {dtype}")
    require_ran_what_it_claimed(
        r, {"stage": "dit-step", "dtype": dtype, "impl": impl, "device": "cuda"},
        f"dit-step {dtype}")
    require_not_degenerate(r, f"dit-step {dtype}")
    return r["metrics"]["latency_s"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--binary", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--impl", default="cuda")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260911)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    with GpuLock():
        require_idle_device()
        dev = device_fingerprint()
        runs = {"fp32": [], "bf16": []}
        for k, dtype in interleave(("fp32", "bf16"), a.repeats):
            s = one(a.binary, a.weights, dtype, a.impl,
                    warmup=a.warmup, iters=a.iters, seed=a.seed)
            runs[dtype].append(s)
            print(f"   r{k}  {dtype:5s}  {s:.4f} s", flush=True)

    fp32 = statistics.median(runs["fp32"])
    bf16 = statistics.median(runs["bf16"])
    doc = {
        "_what_this_is": "One DiT step at fp32 vs bf16, paired and interleaved. A ratio between "
                         "two dtypes of the same implementation -- not a score, never scored.",
        "_why": "fp32 reads twice the weight bytes of bf16. A bandwidth-bound step would take "
                "about twice as long. What it actually takes decides whether a narrower weight "
                "format has anything to sell on this path yet.",
        "basis": "measured",
        "stage": "dit-step",
        "shape": {"resolution": 1024, "caption_len": 300, "batch": 2},
        "impl": a.impl,
        "repeats": a.repeats,
        "warmup": a.warmup,
        "iters": a.iters,
        "device": dev,
        "dit_step_fp32_s": fp32,
        "dit_step_bf16_s": bf16,
        "ratio_fp32_over_bf16": fp32 / bf16,
        "samples": runs,
    }
    Path(a.output).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"\n   fp32 {fp32:.4f} s   bf16 {bf16:.4f} s   ratio {fp32 / bf16:.3f}x")
    print(f"   (a bandwidth-bound step would read ~2.00x the bytes and show ~2.00x here)")
    print(f">> wrote {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
