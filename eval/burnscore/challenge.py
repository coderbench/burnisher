#!/usr/bin/env python3
"""Independent re-measurement, and what the ledger does when two boxes disagree.

The problem this solves
-----------------------

`burnish audit` proves a published verdict follows from its published measurements, and it costs
nothing. What it cannot prove is that those measurements describe what the hardware did. No
arithmetic can. A bot that fabricated a plausible raw file -- thirty paired records, a held-out
shape, two gate results, all internally consistent and all clearing the drift guard -- would pass
every cheap check there is.

The only thing that settles it is somebody else measuring the same submission on their own box.
So that is made a first-class operation rather than a thing people are invited to do informally:

    burnish challenge --pr 42 ...        measure it yourself, append a counter-receipt
    burnish ledger audit                 disagreements beyond the floor put the PR on HOLD

Why a HOLD rather than a rejection
----------------------------------

When two receipts for one submission disagree by more than the cell's own measured noise floor,
at least one of them is wrong and nobody yet knows which. Rejecting would punish a contributor
for an evaluator's bad afternoon; paying would pay for a number nobody can reproduce. Neither is
honest, and there is a third option: say so, hold the credit, and let a third measurement break
the tie. A score no one else can reproduce does not pay, and does not have to be called fraud to
be refused.

Why the floor is the threshold
------------------------------

Not a tolerance, and not a percentage anybody chose. Two runs of the same code on two boxes
differ by hardware and thermal luck, and the cell's floor is exactly the measured size of that
kind of difference -- it is what the calibration measured, control against control. A
disagreement inside the floor is two measurements of the same thing. A disagreement outside it
is two measurements of different things, and the ledger should not guess which one was real.
"""
from __future__ import annotations

import json
from pathlib import Path

from .floor import floor_as_gap_closed


class ChallengeError(ValueError):
    """A counter-receipt cannot be attached to the submission it names."""


def challenge_dir(root, generation_name) -> Path:
    return Path(root) / generation_name / "challenges"


def challenge_path(root, generation_name, receipt_id, box) -> Path:
    return challenge_dir(root, generation_name) / f"{receipt_id}--{box}.json"


def box_id(receipt: dict) -> str:
    """A stable short name for the hardware a receipt was measured on.

    The GPU UUID rather than the model name, because "RTX 5090" is not an identity: two of them
    differ by 3% on the achievable GEMM rate, which is larger than most cells' floors. A
    challenge from the SAME physical card is not independent evidence and this is how the ledger
    can tell.
    """
    dev = (receipt.get("provenance") or {}).get("device") or {}
    uuid = dev.get("uuid") or dev.get("pci.bus_id") or "unknown"
    return uuid.replace("GPU-", "")[:12]


def _subject(receipt: dict) -> tuple:
    """What a receipt is a measurement OF. Two receipts agree only if these match."""
    prov = receipt.get("provenance") or {}
    return (receipt.get("benchmark_generation"),
            receipt.get("generation_digest"),
            prov.get("candidate_commit"),
            prov.get("impl_candidate"),
            prov.get("impl_base"))


def attach(root, canonical: dict, counter: dict, *, receipt_id) -> Path:
    """Record a counter-receipt beside the canonical one it disputes.

    Refused unless the two are measurements of the same thing. A counter-receipt for a different
    commit, a different implementation, or a different generation is not a disagreement -- it is
    a different experiment, and filing it as a challenge would hold a submission hostage to an
    unrelated result.
    """
    if _subject(canonical) != _subject(counter):
        raise ChallengeError(
            "this counter-receipt is not a measurement of the same thing.\n"
            f"  canonical: {_subject(canonical)}\n"
            f"  counter:   {_subject(counter)}\n"
            "A challenge has to re-measure the SAME candidate commit, the same implementations, "
            "against the same frozen generation. Anything else is a different experiment.")
    if box_id(counter) == box_id(canonical):
        raise ChallengeError(
            f"the counter-receipt was measured on the same physical device as the canonical one "
            f"({box_id(counter)}). Re-running on the same box re-measures the same thermal "
            f"regime and the same silicon lottery; it is a repeat, not independent evidence. "
            f"Two RTX 5090s differ by 3% on achievable GEMM, which is larger than most cells' "
            f"floors -- that spread is the thing a challenge exists to surface.")
    p = challenge_path(root, canonical["benchmark_generation"], receipt_id, box_id(counter))
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        existing = json.loads(p.read_text())
        if existing.get("content_digest") == counter.get("content_digest"):
            return p
        raise ChallengeError(
            f"{p} already holds a different counter-receipt from this box. A challenge is "
            f"append-only too; measure again and file under a new box, or supersede it.")
    p.write_text(json.dumps(counter, indent=1, sort_keys=True) + "\n")
    return p


def load_challenges(root, generation_name, receipt_id) -> list:
    d = challenge_dir(root, generation_name)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob(f"{receipt_id}--*.json")):
        out.append((p.stem.split("--", 1)[1], json.loads(p.read_text())))
    return out


def disagreement(canonical: dict, counter: dict, generation) -> dict:
    """How far apart two receipts are, measured in the only unit that means anything here.

    The comparison is per cell and in gap-closed, because that is the currency being paid. Two
    receipts can differ by tens of milliseconds and agree perfectly about what was earned, if the
    cell is far from its ceiling; and they can agree to the millisecond and disagree about the
    credit, if it is close.
    """
    rows, worst, worst_cell = [], 0.0, None
    a_cells = canonical.get("per_cell") or {}
    b_cells = counter.get("per_cell") or {}
    for cid in sorted(set(a_cells) | set(b_cells)):
        cell = generation.cells.get(cid)
        a = (a_cells.get(cid) or {}).get("gap_closed")
        b = (b_cells.get(cid) or {}).get("gap_closed")
        if a is None or b is None or cell is None or not cell.calibrated:
            rows.append({"cell": cid, "canonical": a, "counter": b, "floor_as_gap_closed": None,
                         "within_floor": None,
                         "note": "one side did not measure this cell, or it is not calibrated"})
            continue
        floor = floor_as_gap_closed(cell.floor_pct, cell.achieved)
        delta = abs(a - b)
        if delta > worst:
            worst, worst_cell = delta, cid
        rows.append({"cell": cid, "canonical": a, "counter": b, "delta": delta,
                     "floor_as_gap_closed": floor, "within_floor": delta <= floor,
                     "floors_apart": (delta / floor) if floor > 0 else None})
    agree = all(r.get("within_floor") is not False for r in rows)
    return {
        "agree": agree, "worst_delta": worst, "worst_cell": worst_cell,
        "per_cell": rows,
        "canonical_box": box_id(canonical), "counter_box": box_id(counter),
        "_threshold": (
            "Each cell's own MEASURED noise floor, converted into gap-closed -- the currency "
            "being paid. Not a tolerance anybody chose: the floor is what the calibration "
            "measured running the unmodified base against itself, so a disagreement inside it "
            "is two measurements of the same thing and a disagreement outside it is not."),
    }


def status_for(root, generation_name, receipt_id, canonical, generation) -> dict:
    """Whether this submission's credit stands, and why.

    Called by the ledger before compounding. A HELD submission credits nothing until the
    disagreement is resolved -- by further measurements on other cards, or by a superseding receipt
    that says what went wrong. The canonical receipt counts as one measurement: the credit stands
    only while the measurements that agree with it outnumber the ones that do not, so one
    disagreement holds it and a third measurement that agrees with the canonical one releases it.
    """
    challenges = load_challenges(root, generation_name, receipt_id)
    if not challenges:
        return {"held": False, "challenges": 0, "confirmations": 0, "disagreements": []}
    results = [(box, disagreement(canonical, c, generation)) for box, c in challenges]
    bad = [{"box": box, **d} for box, d in results if not d["agree"]]
    confirmations = sum(1 for _, d in results if d["agree"])
    held = len(bad) >= 1 + confirmations
    return {
        "held": held,
        "challenges": len(results),
        "confirmations": confirmations,
        "disagreements": bad,
        "_why_held": (
            "Independent re-measurements that disagree with the canonical receipt by more than "
            "the cell's own noise floor are at least as many as the measurements that agree with "
            "it. One side is wrong and the ledger does not know which, so the credit is held "
            "rather than paid or withdrawn. A further measurement on another card that agrees "
            "with the canonical receipt releases it; a superseding receipt corrects it."
        ) if held else None,
    }
