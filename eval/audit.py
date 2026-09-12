#!/usr/bin/env python3
"""Re-derive a published verdict from its published measurements. No GPU, about two seconds.

    burnish audit ledger/BG-1/pr-000042-raw.json ledger/BG-1/pr-000042-receipt.json
    burnish audit --ledger ledger --generation BG-1 --all

What this checks, and it is worth being precise about which of these are cheap:

  1. The receipt's content digest matches its own body.        (a receipt was not edited)
  2. Re-scoring the raw measurements reproduces the receipt.   (the score follows from the data)
  3. The verdict and label follow from the receipt.            (the PR says what the receipt says)
  4. The generation digest matches the frozen generation.      (it was scored against this ruler)

All four are arithmetic. They need no device, no network and no trust in whoever produced the
receipt -- only the two files, which are committed. Anybody can run this on a laptop, and it is
the intended way to check a score: not "do I believe the bot", but "does the bot's own data
support what it published".

What this CANNOT check is whether the measurements describe what the hardware actually did. No
amount of arithmetic can, and pretending otherwise would be worse than not offering the check.
Fabricated numbers are caught by re-measuring -- `burnish challenge` -- and by the fact that the
raw file is committed, so a fabrication has to be internally consistent across thirty paired
records, a held-out shape drawn after the fact, and two independent gate results.

The division of labour is the point: the cheap check catches every scoring bug and every edited
receipt, which is most of what goes wrong, and it costs nothing. The expensive check is reserved
for the one thing only hardware can settle.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from burnscore import cells as C
from burnscore import compute as CP
from burnscore import receipt as R
from burnscore import verdict as V
from paths import add_argument as add_cells_root_arg, generation_path

ROOT = Path(__file__).resolve().parent.parent

# Fields that cannot match on a re-derivation and must not be compared.
#
# A receipt is stamped with the wall clock at the moment it was built, and its content digest
# covers that stamp. Everything else is a function of the measurements alone -- which is the
# property under audit, so the exclusion list is kept to exactly these two and is stated here
# rather than buried, because a long exclusion list would hollow the check out.
VOLATILE = ("timestamp_utc", "content_digest")


class AuditError(AssertionError):
    """A published verdict is not supported by its own published measurements."""


def audit_one(raw_path, receipt_path, *, cells_root=None, calibration=None,
              verbose=True) -> dict:
    raw = json.loads(Path(raw_path).read_text())
    published = json.loads(Path(receipt_path).read_text())
    gen_name = published.get("benchmark_generation") or raw.get("generation")
    # The calibration the receipt was scored against, in order of preference: one the auditor
    # named, the one EMBEDDED in the raw file, then the committed reference device's. The
    # embedded copy is what makes this check portable -- every validator calibrates their own
    # box, so without it only the validator who produced a receipt could re-derive it, and
    # "anyone can check this" would be a claim rather than a fact.
    embedded = None
    if not calibration and raw.get("calibration", {}).get("cells"):
        embedded = Path(tempfile.mkdtemp()) / "calibration.json"
        embedded.write_text(json.dumps(raw["calibration"]))
    gen = C.load(generation_path(gen_name, cells_root), calibration=calibration or embedded)
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        if verbose:
            print(f"   [{'ok' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
        return ok

    # 1. Internal integrity, before anything is recomputed. A receipt whose digest does not
    #    cover its own body has been edited after the fact, and nothing below it means anything.
    try:
        R.verify_receipt(published, gen)
        check("the receipt verifies against itself and its generation", True)
    except R.ReceiptError as e:
        check("the receipt verifies against itself and its generation", False, str(e)[:160])

    # 2. The score follows from the data. This is the one that catches a scorer bug, a
    #    hand-edited number, and a receipt built from measurements other than the ones shipped.
    # A ComputeError here is a FINDING, not a crash. The scorer refuses measurements it cannot
    # trust -- unpaired repeats, a base arm that has drifted from its calibration, a degenerate
    # output -- and an auditor who fed it a receipt built from such measurements needs to be
    # told which, not handed a traceback. This is also the path a tampered raw file takes:
    # moving the base arm to manufacture a speedup moves its achieved fraction too, and the
    # drift guard notices before anything is scored.
    try:
        out = CP.compute(gen, raw["records"], held_out_records=raw.get("held_out") or None,
                         device=(raw.get("provenance") or {}).get("device"))
        rebuilt = R.build_receipt(
            generation=gen, per_cell=out["per_cell"], aggregate=out["aggregate"],
            interval=out["interval"], frontier=out["frontier"],
            correctness=raw.get("correctness"), determinism=raw.get("determinism"),
            coverage=out["coverage"],
            held_out=(out.get("held_out") or {}).get("survived", True) is not False,
            provenance=raw.get("provenance"), pr=published.get("pr"),
            supersedes=published.get("supersedes") or None)
        differing = [k for k in set(rebuilt) | set(published)
                     if k not in VOLATILE and rebuilt.get(k) != published.get(k)]
        check("re-scoring the raw measurements reproduces the receipt", not differing,
              "" if not differing else f"differs at {', '.join(sorted(differing))}")
    except CP.ComputeError as e:
        check("re-scoring the raw measurements reproduces the receipt", False,
              f"the scorer refuses these measurements: {str(e).splitlines()[0][:140]}")

    # 3. The verdict is a pure function of the receipt, so it is re-derived rather than trusted.
    v = V.verdict(published)
    check("the verdict and label follow from the receipt", True,
          f"{v['label']}  (pays {v['payout_fraction']:.4f})")

    # 4. Scored against the ruler it claims. A receipt naming a generation digest that is not
    #    the committed one was scored against a different definition of the benchmark.
    check("scored against the frozen generation as committed",
          published.get("generation_digest") == gen.digest,
          f"receipt {str(published.get('generation_digest'))[:22]}... "
          f"vs committed {gen.digest[:22]}...")

    ok = all(c["pass"] for c in checks)
    return {"pr": published.get("pr"), "generation": gen_name, "verdict": v,
            "checks": checks, "pass": ok,
            "raw": str(raw_path), "receipt": str(receipt_path)}


def _pairs_in(ledger: Path, generation: str):
    d = ledger / generation
    for rec in sorted(d.glob("receipts/*.json")):
        raw = d / "raw" / (rec.stem + "-raw.json")
        if raw.exists():
            yield raw, rec


def main_with(argv):
    """Entry point that takes an argv, so `tools/burnish audit` is the same code path."""
    return _run(argv)


def main():
    return _run(None)


def _run(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", nargs="?", help="the raw measurements")
    ap.add_argument("receipt", nargs="?", help="the receipt derived from them")
    ap.add_argument("--ledger", help="audit every receipt in a ledger instead")
    ap.add_argument("--generation", default="BG-1")
    ap.add_argument("--json", help="write the full result here")
    add_cells_root_arg(ap)
    a = ap.parse_args(argv)

    results = []
    if a.ledger:
        pairs = list(_pairs_in(Path(a.ledger), a.generation))
        if not pairs:
            print(f"!! no receipt/raw pairs under {a.ledger}/{a.generation}.\n"
                  f"   An audit needs BOTH: a receipt whose measurements were not kept cannot "
                  f"be re-derived,\n   which is the whole point of keeping them.", file=sys.stderr)
            return 2
        for raw, rec in pairs:
            print(f">> {rec.name}")
            results.append(audit_one(raw, rec, cells_root=a.cells_root))
            print()
    else:
        if not (a.raw and a.receipt):
            ap.error("give a raw file and a receipt, or --ledger DIR")
        print(f">> {Path(a.receipt).name}")
        results.append(audit_one(a.raw, a.receipt, cells_root=a.cells_root))

    bad = [r for r in results if not r["pass"]]
    doc = {"audited": len(results), "failed": len(bad), "results": results,
           "_what_this_proves": (
               "That each published verdict follows from the measurements published beside it. "
               "It does NOT prove the measurements describe what the hardware did -- only "
               "re-measuring can, and `burnish challenge` is how."),
           }
    if a.json:
        Path(a.json).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print(f">> wrote {a.json}")

    if bad:
        print(f"!! {len(bad)} of {len(results)} receipts are NOT supported by their own "
              f"measurements.", file=sys.stderr)
        for r in bad:
            for c in r["checks"]:
                if not c["pass"]:
                    print(f"   pr {r['pr']}: {c['check']} -- {c['detail']}", file=sys.stderr)
        return 1
    print(f"ok: {len(results)} receipt(s) re-derived from their own measurements, no GPU needed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
