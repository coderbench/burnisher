#!/usr/bin/env python3
"""Set the correctness gate's thresholds from a measurement instead of an argument.

    burnish gate --calibrate-tolerance --dtype bf16 --reference <same-dtype oracle> ...
    scripts/apply_tolerance.py gate-calibration.json --write

**Why this is a script.** The thresholds decide what counts as a correct implementation, so they
are exactly the kind of number that must not be typed. BG-1's first pair were argued from a
single forward pass and both turned out wrong: the relative-L2 bound was unsatisfiable across
dtypes, and the max-abs bound rejected a correct fp32 implementation whose whole-tensor norm was
fifty times inside its own limit.

**What the margin means, and why it is a judgement.** The gate asks whether an implementation
computes something DIFFERENT, so the threshold must sit above the spread two correct
implementations show at the same dtype, and below what an actual algorithm change produces. The
first is measured here; the second is not a single number. The multiple is therefore a stated
judgement rather than a derivation, and both the measurement and the multiple are recorded so the
next person can disagree with the multiple without re-running anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Headroom over the measured same-dtype drift.
#
# Five, because the measurement is a handful of prompts on one box and the quantity it measures
# is amplified by a chaotic trajectory -- a threshold sitting just above the observed worst case
# would reject a correct implementation on the sixth prompt. It is not derived from anything and
# it is not meant to look as though it is.
MARGIN = 5.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("calibration", help="a gate report written with --calibrate-tolerance")
    ap.add_argument("--generation", default="BG-1")
    ap.add_argument("--margin", type=float, default=MARGIN)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    report = json.loads(Path(args.calibration).read_text())
    drift = report.get("measured_drift")
    if not drift:
        print("!! that report has no `measured_drift`. Run the gate with --calibrate-tolerance, "
              "which measures instead of asserting.", file=sys.stderr)
        return 2
    dtype = report.get("dtype")
    if not dtype:
        print("!! the report does not say which dtype it measured. A tolerance that does not "
              "know its own dtype is the mistake this file exists to fix.", file=sys.stderr)
        return 2

    l2 = drift["relative_l2"] * args.margin
    mx = drift["max_abs"] * args.margin

    path = ROOT / "configs" / "tolerance.json"
    doc = json.loads(path.read_text())
    gen = doc.setdefault(args.generation, {})
    old = {"latent_l2_relative": gen.get("latent_l2_relative"),
           "latent_max_abs": gen.get("latent_max_abs"), "basis": gen.get("basis")}

    gen["latent_l2_relative"] = round(l2, 6)
    gen["latent_max_abs"] = round(mx, 6)
    gen["basis"] = "measured"
    gen["scored_dtype"] = dtype
    gen["_basis_note"] = (
        f"MEASURED. {args.margin:g}x the worst same-dtype drift between this runtime and the "
        f"reference implementation at {dtype}, over the frozen prompt set. The multiple is a "
        f"stated judgement: the measurement is a handful of prompts on one box and the quantity "
        f"is amplified by a chaotic trajectory, so a threshold at the observed worst case would "
        f"reject a correct implementation on the next prompt. Both the measurement and the "
        f"multiple are recorded so the multiple can be disputed without re-running anything.")
    gen.setdefault("measured", {})[f"{dtype}_vs_{dtype}_oracle"] = {
        "worst_relative_l2": drift["relative_l2"],
        "worst_max_abs": drift["max_abs"],
        "margin": args.margin,
        "prompt_set_digest": report.get("prompt_set_digest"),
        "noise_sha256": report.get("noise_sha256"),
        "device": (report.get("device_fingerprint") or report.get("device")),
        "_what": "This runtime against the reference at the SAME dtype. The only comparison that "
                 "isolates implementation error from the cost of the dtype.",
    }
    gen["measured"].pop("_bf16_note", None)
    gen["previous"] = old

    print(f"  {'threshold':24s} {'was':>12s} {'now':>12s}")
    print(f"  {'latent_l2_relative':24s} {str(old['latent_l2_relative']):>12s} "
          f"{gen['latent_l2_relative']:>12.6f}")
    print(f"  {'latent_max_abs':24s} {str(old['latent_max_abs']):>12s} "
          f"{gen['latent_max_abs']:>12.6f}")
    print(f"  basis: {old['basis']} -> measured (at {dtype}, {args.margin:g}x margin)")
    print(f"\n  measured drift: rel L2 {drift['relative_l2']:.6f}, "
          f"max abs {drift['max_abs']:.6f}")

    if args.write:
        path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print(f"\n>> wrote {path}")
        print("   now: eval/make_generation.py --write && "
              "eval/roofline_table.py --markdown docs/ROOFLINE.md")
    else:
        print("\n(dry run; --write to apply)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
