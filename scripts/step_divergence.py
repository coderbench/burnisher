#!/usr/bin/env python3
"""Where does a bf16 disagreement take off with step count, and where can a gate still see?

    scripts/step_divergence.py --weights DIR --binary ./build-cuda/burnisher --steps 1 2 4 8 20

**The question this answers.** This runtime agrees with the reference implementation to 5e-4 in
fp32 over twenty steps, and to 1.20 -- essentially uncorrelated -- in bf16 against a bf16 oracle.
Both cannot be explained by one story. Either there is a bf16-specific defect, or the denoise
trajectory amplifies bf16's rounding until two legitimate implementations that round at different
points end up in different basins.

Those two look identical at twenty steps and completely different at one. So: sweep the step
count, run BOTH sides at each, and watch the curve. A defect is present at every step count. Chaos
starts small and saturates -- and the saturation value for two tensors of similar statistics is
about sqrt(2), which is what "no relationship at all" looks like in relative L2.

This is the same move that localised the DiT disagreement by truncating its block stack, and it
is the only way to tell an implementation error from an amplified rounding difference.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd, **kw):
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"!! exited {p.returncode}: {' '.join(str(c) for c in cmd[:4])}")
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--noise", required=True)
    ap.add_argument("--generation", default="BG-1")
    ap.add_argument("--prompt", default="short-caption")
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8, 20])
    ap.add_argument("--dtypes", nargs="+", default=["fp32", "bf16"])
    ap.add_argument("--output")
    args = ap.parse_args()

    import numpy as np

    gdir = ROOT / "eval" / "cells" / args.generation
    gen = json.loads((gdir / "generation.json").read_text())
    ids = gdir / f"token-ids-{args.prompt}.txt"
    work = Path(tempfile.mkdtemp())
    torch_dtype = {"fp32": "float32", "bf16": "bfloat16"}

    rows = []
    for dtype in args.dtypes:
        for steps in args.steps:
            ref_dir = work / f"ref-{dtype}-{steps}"
            run([sys.executable, str(ROOT / "scripts" / "make_reference_latents.py"),
                 "--weights", args.weights, "--noise", args.noise, "--device", "cuda",
                 "--dtype", torch_dtype[dtype], "--steps", str(steps),
                 "--prompts", args.prompt, "--write"],
                env=_env_with_outdir(ref_dir))
            ours = work / f"ours-{dtype}-{steps}.npy"
            run([args.binary, "generate", "--weights", args.weights, "--token-ids", str(ids),
                 "--noise", args.noise, "--seed", str(gen["model"].get("seed", 20260911)),
                 "--impl", "cuda", "--device", "cuda", "--dtype", dtype,
                 "--resolution", str(gen["model"]["resolution"]), "--steps", str(steps),
                 "--guidance-scale", str(gen["model"]["guidance_scale"]),
                 "--dump-latents", str(ours)])
            a = np.load(ref_dir / f"{args.prompt}.npy").astype(np.float64)
            b = np.load(ours).astype(np.float64)
            rel = float(np.linalg.norm(a - b) / np.linalg.norm(a))
            rows.append({"dtype": dtype, "steps": steps, "relative_l2": rel,
                         "max_abs": float(np.abs(a - b).max())})
            print(f"  {dtype:5s} {steps:3d} steps   rel L2 {rel:9.6f}   "
                  f"max abs {rows[-1]['max_abs']:8.4f}", flush=True)

    print("\n  A defect shows at EVERY step count. Amplified rounding starts small and climbs")
    print("  toward ~1.41, which is what two unrelated tensors of similar statistics measure.")
    if args.output:
        Path(args.output).write_text(json.dumps(
            {"_what": __doc__.strip().splitlines()[0], "prompt": args.prompt,
             "rows": rows}, indent=1) + "\n")
        print(f"\n>> wrote {args.output}")
    return 0


def _env_with_outdir(d):
    import os
    e = dict(os.environ)
    e["BURNISH_REF_OUTDIR"] = str(d)
    return e


if __name__ == "__main__":
    sys.exit(main())
