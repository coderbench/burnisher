#!/usr/bin/env python3
"""The payable outcome of a receipt, and the label that carries it.

The inversion this module exists for
------------------------------------

The usual shape of a scored subnet is: a bot measures, the bot decides a grade, and the grade is
the record. To check the grade you must reproduce the measurement, which means owning the
hardware. Everyone without a GPU is asked to trust the bot.

Burnisher can do better, and not by being clever -- by an accident of arithmetic that is worth
naming. Scoring here is a pure function of recorded measurements: no device, no clock, no
randomness. A receipt scored on a Blackwell part re-derives, figure for figure, on a laptop. So
the record is not the verdict; the record is the MEASUREMENTS, and the verdict is a derivation
anyone can run in about two seconds.

That splits verification into two tiers with very different costs:

    arithmetic   Did the published numbers produce the published verdict?
                 Anyone. No GPU. Seconds. `burnish audit`.

    measurement  Do those numbers describe what the hardware actually did?
                 Needs an RTX 5090 and half an hour. `burnish challenge`.

The cheap tier catches every scoring bug, every transcription error, every quietly-edited
receipt, and every disagreement between what a bot said and what its own data supports. It does
not catch a fabricated measurement -- nothing arithmetic can. That is what the expensive tier and
the append-only ledger are for.

What the verdict is NOT
-----------------------

It is not a grade. There are no tiers here, and the absence is deliberate: a bucket boundary
means two submissions that measured differently are paid identically, and one that measured
almost identically to a third is paid differently. The number this module publishes IS the
payout basis -- the fraction of the remaining roofline gap the change closed -- because that
quantity is already normalised to [0, 1], already comparable across cells and models, and
already self-terminating. Rounding it into a letter would throw away the only property that
makes it worth computing.
"""
from __future__ import annotations

from .receipt import CREDITING, STATUSES

# The label prefix. One namespace, so a repository that scores more than one thing can tell its
# own labels apart from everybody else's at a glance.
PREFIX = "burnish"

# How many decimals of the payout fraction the label carries.
#
# Four, because the smallest thing this instrument can resolve is the noise floor expressed as
# gap-closed, and on the tightest cell in BG-1 that is 0.00009 -- so four decimals can represent
# a contribution roughly at the floor, and a fifth would encode noise as though it were signal.
# Fixed width rather than significant figures: fixed-width decimals sort lexicographically in
# the same order as the numbers they represent, so a list of labels is already ranked.
GAP_DECIMALS = 4

# Statuses that pay, and what each of the others means to somebody reading it on a PR.
#
# This vocabulary is the runtime's own -- burnscore.receipt.STATUSES -- rather than a second set
# invented for display. A status that means one thing in the receipt and another on the PR is
# how a contributor comes to believe they were paid for something they were not.
OUTCOMES = {
    "FRONTIER_EXPANDED": (
        "resolved gain, and the frontier grew",
        "Paid. The number on the label is the fraction of this generation's REMAINING "
        "roofline gap that this change closed."),
    "EXPANDED_OFF_LATENCY": (
        "the frontier grew without a latency gain",
        "Paid. Latency did not move, but peak memory or output fidelity did, and both are "
        "scored objectives."),
    "MOVED_ALONG_FRONTIER": (
        "faster, and paid for in memory or fidelity",
        "Not paid. The runtime could already make this trade by turning a knob; a submission "
        "has to move the frontier, not slide along it."),
    "NO_GAIN": (
        "resolved, and measurably not an improvement",
        "Not paid, and this is a real result rather than a failure: the measurement was good "
        "enough to say so. The interval is on the receipt."),
    "UNRESOLVED": (
        "inside the cell's own measured noise",
        "Not paid, and NOT a judgement about the idea. The effect could not be told from a "
        "quiet afternoon on this box. An axis whose spread sits inside its own noise is open, "
        "not solved."),
    "SHAPE_OVERFIT": (
        "the gain did not survive the held-out shape",
        "Not paid. The shape was drawn after the candidate was frozen; a kernel fast only at "
        "the benchmarked shape is a tuned constant."),
    "PARTIAL": (
        "the matrix was incomplete",
        "Not paid. Dropping the cell a change hurts is the cheapest way to raise a score, so a "
        "partial matrix credits nothing rather than being policed."),
    "CORRECTNESS_FAIL": (
        "the latents moved outside the stated tolerance",
        "Rejected before timing was considered. This is not traded against a speed win, and "
        "widening the tolerance is not the fix."),
    "DETERMINISM_FAIL": (
        "the build did not reproduce itself",
        "Rejected. Nothing can be attributed to a change against a baseline that does not "
        "produce the same bytes twice."),
    "BUILD_FAIL": ("the submission did not build", "Not evaluated."),
    "EVAL_ERROR": ("the evaluator failed", "Not the submission's fault; it will be re-run."),
}

# Outcomes that are not a receipt status, because they are properties of the LEDGER rather than
# of any single run.
HELD = "HELD"
CELL_OPENED = "CELL_OPENED"

EXTRA_OUTCOMES = {
    HELD: (
        "an independent re-measurement disagrees beyond the noise floor",
        "Held, not rejected. Two receipts for this submission disagree by more than the cell's "
        "own floor, so at least one of them is wrong and nobody yet knows which. A score no "
        "one else can reproduce does not pay."),
    CELL_OPENED: (
        "a new cell was opened",
        "Paid as cartography. Landing a new cell -- its reference, its calibration, its "
        "roofline -- adds a place where future work can be measured. A benchmark that does not "
        "grow its own surface stops paying anyone."),
}

ALL_OUTCOMES = {**OUTCOMES, **EXTRA_OUTCOMES}


# Colour by what the outcome MEANS to the person who wrote the pull request, not by severity.
#
# A reader scanning a list of pull requests should be able to tell, without reading a word, which
# of four things happened: you were paid, you were measured and not paid, we could not tell, or
# it was rejected. Severity colouring gets this wrong -- it would paint `unresolved` and
# `correctness-fail` the same alarming red, when one of them is "we could not measure your idea"
# and the other is "your change is incorrect".
#
# Kept here rather than in the shell script that creates the labels, so the colour and the
# meaning sit in one place and a new outcome cannot be added without one.
GREEN = "0E8A16"       # paid
BLUE = "1D76DB"        # a real measurement that did not pay
PALE = "C5DEF5"        # we could not tell
RED = "B60205"         # rejected on correctness
AMBER = "FBCA04"       # rejected on a guard
PURPLE = "8250DF"      # disputed
GREY = "BFD4F2"        # not evaluated
ORANGE = "F9A825"      # the evaluator's own fault

COLORS = {
    "FRONTIER_EXPANDED": GREEN,
    "EXPANDED_OFF_LATENCY": GREEN,
    "CELL_OPENED": GREEN,
    "NO_GAIN": BLUE,
    "MOVED_ALONG_FRONTIER": BLUE,
    "UNRESOLVED": PALE,
    "PARTIAL": PALE,
    "SHAPE_OVERFIT": AMBER,
    "CORRECTNESS_FAIL": RED,
    "DETERMINISM_FAIL": RED,
    "HELD": PURPLE,
    "BUILD_FAIL": ORANGE,
    "EVAL_ERROR": ORANGE,
}

# The paying label carries a NUMBER, so it is created on demand rather than pre-registered --
# and a label GitHub auto-creates gets a RANDOM colour. The most important outcome in the system
# would have come out a different shade every time, occasionally red.
#
# So it is coloured here, and the shade carries the magnitude: a bigger contribution is a deeper
# green. Logarithmic, because real values span orders of magnitude -- the noise floor on the
# tightest cell is worth 0.00009 of the gap and a large win is 0.1, and a linear ramp would paint
# everything below a tenth the same pale colour.
GAP_RAMP_LO, GAP_RAMP_HI = 1e-4, 0.5
GAP_PALE = (0xC6, 0xE6, 0xC6)
GAP_DEEP = (0x04, 0x4D, 0x0C)


def color_for(receipt: dict) -> str:
    """The label colour for this outcome. Six hex digits, no leading '#', as GitHub wants."""
    import math
    status = receipt.get("status")
    if status in CREDITING:
        credited = float((receipt.get("score") or {}).get("credited_gap_closed") or 0.0)
        if credited <= 0:
            return COLORS.get(status, GREY)
        lo, hi = math.log10(GAP_RAMP_LO), math.log10(GAP_RAMP_HI)
        t = (math.log10(max(credited, GAP_RAMP_LO)) - lo) / (hi - lo)
        t = min(max(t, 0.0), 1.0)
        rgb = tuple(round(a + (b - a) * t) for a, b in zip(GAP_PALE, GAP_DEEP))
        return "%02X%02X%02X" % rgb
    return COLORS.get(status, GREY)


class VerdictError(ValueError):
    """A receipt cannot be turned into a verdict."""


def _slug(status: str) -> str:
    return status.lower().replace("_", "-")


def format_gap(value: float) -> str:
    """The payout fraction as it appears in a label: fixed width, signed, four decimals.

    Signed even when positive, so `gap+0.0342` and `gap-0.0058` are the same width and a reader
    never has to work out which way a bare number went.
    """
    return f"{value:+.{GAP_DECIMALS}f}"


def label_for(receipt: dict) -> str:
    """The single label that carries this receipt's outcome to whatever reads labels.

    A paying outcome carries the NUMBER. Every other outcome carries the reason, because a
    number is the wrong thing to publish about a submission that was not measured, was not
    correct, or could not be told apart from noise.
    """
    status = receipt.get("status")
    if status not in STATUSES:
        raise VerdictError(f"receipt carries unknown status {status!r}")
    if status in CREDITING:
        credited = float((receipt.get("score") or {}).get("credited_gap_closed") or 0.0)
        return f"{PREFIX}:gap{format_gap(credited)}"
    return f"{PREFIX}:{_slug(status)}"


def verdict(receipt: dict) -> dict:
    """Everything a reader, a payer, or an auditor needs, derived from the receipt alone.

    Pure. Same receipt in, same verdict out, on any machine, forever. That property is the whole
    reason this is a separate function rather than something the bot decides while it has the
    numbers in hand.
    """
    status = receipt.get("status")
    if status not in STATUSES:
        raise VerdictError(f"receipt carries unknown status {status!r}")
    score = receipt.get("score") or {}
    credited = float(score.get("credited_gap_closed") or 0.0)
    measured = score.get("gap_closed")
    pays = status in CREDITING and credited > 0.0
    headline, meaning = ALL_OUTCOMES[status]
    ci = score.get("confidence_interval") or {}
    return {
        "status": status,
        "pays": pays,
        # THE payout basis. Not an input to a tier table -- the quantity itself.
        "payout_fraction": credited if pays else 0.0,
        "_payout_basis": (
            "The fraction of this generation's REMAINING roofline gap that the change closed, "
            "in [0, 1]. It is already normalised, already comparable across cells and models, "
            "and already self-terminating -- the same kernel win is worth less once the gap it "
            "closes is smaller. Emission proportional to this number needs no tier table, and a "
            "tier table would discard the only property that makes it worth computing."),
        "measured_gap_closed": measured,
        "confidence_interval": (
            [ci.get("lower"), ci.get("upper")] if ci else None),
        "confidence_level": ci.get("level"),
        "resolved": score.get("resolved"),
        "label": label_for(receipt),
        "headline": headline,
        "meaning": meaning,
        "generation": receipt.get("benchmark_generation"),
        "generation_digest": receipt.get("generation_digest"),
        "content_digest": receipt.get("content_digest"),
        "pr": receipt.get("pr"),
        "provenance_complete": bool(
            (receipt.get("provenance") or {}).get("code_provenance_complete")),
        "_verifiable_without_a_gpu": (
            "This verdict is a pure function of the receipt, and the receipt is a pure function "
            "of the raw measurements committed beside it. `burnish audit` re-derives both and "
            "compares, on any machine, in about two seconds. What that CANNOT check is whether "
            "the measurements describe what the hardware did -- for that, re-measure and append "
            "a counter-receipt with `burnish challenge`."),
    }


def all_labels() -> list:
    """Every label this repository can apply, for the one-time label setup.

    The paying outcome is excluded: it is not a fixed string, it carries a number, and
    pre-creating one label per representable value would be 20001 labels.
    """
    return [f"{PREFIX}:{_slug(s)}" for s in sorted(set(STATUSES) - set(CREDITING))] + \
           [f"{PREFIX}:{_slug(s)}" for s in sorted(EXTRA_OUTCOMES)]
