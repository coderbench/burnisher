#!/usr/bin/env python3
"""Render what a pull request gets back, for every outcome the evaluator can reach.

    scripts/sample_outcomes.py                 # all of them
    scripts/sample_outcomes.py --only gap      # one

This is documentation that cannot go stale, because it is produced by the same two functions the
bot uses -- `burnscore.verdict.verdict` and `pr_bot.report`. If a comment here looks wrong, the
comment a real contributor receives is wrong in the same way.

Where the numbers come from, stated per sample rather than in aggregate:

  MEASURED    the committed example in examples/ -- a real paired run on the pinned RTX 5090.
  DERIVED     that same run's measurements, with the status and score set to what the scorer
              would have produced had the candidate behaved differently. The prose, the tables
              and the label are then real output; the figures are illustrative and are labelled
              as such on every sample. Nothing here is presented as a measurement that happened.

That distinction is the whole discipline of this repository, so it would be absurd to blur it in
the document that shows people what a score looks like.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

import pr_bot as B
from burnscore import verdict as V

REAL_RECEIPT = ROOT / "examples" / "BG-1-pr-000001-receipt.json"


def _shaped(base, *, status, gap=None, per_cell_gap=None, resolved=True):
    r = copy.deepcopy(base)
    r["status"] = status
    if gap is not None:
        r["score"] = dict(r["score"], gap_closed=gap, credited_gap_closed=(
            gap if status in ("FRONTIER_EXPANDED", "EXPANDED_OFF_LATENCY") else 0.0),
            resolved=resolved)
        lo, hi = gap * 0.97, gap * 1.03
        r["score"]["confidence_interval"] = dict(
            r["score"].get("confidence_interval") or {}, lower=min(lo, hi), upper=max(lo, hi))
    if per_cell_gap:
        for cid, g in per_cell_gap.items():
            if cid not in r["per_cell"]:
                continue
            cell = dict(r["per_cell"][cid], gap_closed=g, resolved=resolved)
            # The achieved fraction has to FOLLOW the gap, not sit beside it. Setting a gap and
            # leaving the original run's achieved column produced a sample showing a gain with
            # the achieved fraction going DOWN -- incoherent, and in the one document whose job
            # is to show people what a score looks like.
            #
            #   g = (a_cand - a_base) / (1 - a_base)   =>   a_cand = a_base + g (1 - a_base)
            a_base = cell.get("achieved_base")
            if a_base is not None:
                cell["achieved_candidate"] = a_base + g * (1.0 - a_base)
            r["per_cell"][cid] = cell
    return r


def samples():
    real = json.loads(REAL_RECEIPT.read_text())
    out = [("unresolved", "MEASURED",
            "The first run this instrument ever scored. A 1024-wide attention tile, which "
            "spilled the working set and made one DiT step 60% slower.", real)]

    out.append((
        "gap", "DERIVED",
        "What a paying submission looks like. The number on the label IS the payout basis.",
        _shaped(real, status="FRONTIER_EXPANDED", gap=0.0342,
                per_cell_gap={"dit-step/1024/bf16": 0.0361,
                              "t5-encode/1024/bf16": 0.0088,
                              "vae-decode/1024/bf16": 0.0002})))

    out.append((
        "no-gain", "DERIVED",
        "Resolved, and measurably not an improvement. A real result, not a failure -- the "
        "measurement was good enough to say so.",
        _shaped(real, status="NO_GAIN", gap=0.00002,
                per_cell_gap={"dit-step/1024/bf16": 0.00002})))

    out.append((
        "shape-overfit", "DERIVED",
        "The gain was real at the benchmarked shape and gone at one drawn after the candidate "
        "was frozen.",
        _shaped(real, status="SHAPE_OVERFIT", gap=0.0410,
                per_cell_gap={"dit-step/1024/bf16": 0.0432})))

    out.append((
        "correctness-fail", "DERIVED",
        "Rejected before timing was considered. Never traded against a speed win.",
        _shaped(real, status="CORRECTNESS_FAIL", gap=None)))

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="render one sample by its label suffix")
    a = ap.parse_args()

    for key, basis, why, receipt in samples():
        if a.only and a.only != key:
            continue
        v = V.verdict(receipt)
        print("=" * 78)
        print(f"  {v['label']}")
        print(f"  figures: {basis}   {why}")
        print("=" * 78)
        print()
        print(B.report(v, receipt, raw_name="pr-000042-raw.json",
                       receipt_name="pr-000042.json"))
        print()

    if not a.only:
        print("=" * 78)
        print("  burnish:skipped-instrument")
        print("  figures: none -- nothing was measured, and that is the point")
        print("=" * 78)
        print()
        print(B._skip_note({"blocked": [
            {"path": "eval/burnscore/floor.py",
             "why": "modifies the measuring instrument. This decides what is measured, so a "
                    "submission that could change it could win by editing the ruler."}]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
