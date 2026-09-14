"""The Burnisher Receipt: the permanent, machine-readable evidence for one submission.

A receipt is the only thing that survives. It has to answer, years later and without the box
that produced it: which generation, which base commit, which candidate commit, on what hardware,
against what model, with what correctness result, at what confidence, and how much of each
cell's remaining roofline gap was closed.

**No letter grades.** There is no XS/S/M/L/XL anywhere in this system and there will not be.
A status here describes evaluation STATE -- what happened -- and never magnitude. The magnitude
is a number with an interval beside it, and a number with an interval cannot be argued into a
higher bucket.

**Every figure is measured or it is marked.** `basis` is `"measured"` only for a value a run
produced. Ceilings are `"model"`. A receipt carrying a modelled figure in a measured field is
refused by `verify`, and `eval/tests/test_receipt.py` asserts it, because this is precisely the
mistake that a confident evaluator makes silently.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

SCHEMA_VERSION = 1

# Evaluation state. None of these categorizes impact magnitude.
STATUSES = (
    "FRONTIER_EXPANDED",      # resolved gain, and the frontier grew
    "MOVED_ALONG_FRONTIER",   # resolved latency gain paid for in memory or quality
    "EXPANDED_OFF_LATENCY",   # no latency gain, but the frontier grew (memory, or quality)
    "NO_GAIN",                # resolved, and not an improvement
    "UNRESOLVED",             # inside the cell's own noise; we do not know
    "CORRECTNESS_FAIL",       # rejected before timing was considered
    "DETERMINISM_FAIL",       # the build does not reproduce itself
    "SHAPE_OVERFIT",          # the gain did not survive the held-out shape
    "PARTIAL",                # the matrix was incomplete; credits nothing
    "BUILD_FAIL",
    "EVAL_ERROR",
)

# EXPANDED_OFF_LATENCY is not here. Payment is latency gap closed, and a run is resolved only when
# its latency result clears the floor, so a gain in memory or fidelity alone has no number to pay.
# It is recorded, and it keeps a speedup from being paid for in those objectives.
CREDITING = frozenset({"FRONTIER_EXPANDED"})


class ReceiptError(ValueError):
    """A receipt is malformed, or does not verify against its own contents."""


def decide_status(*, correctness, determinism, coverage, held_out, resolved, gap_closed,
                  frontier_delta) -> str:
    """The status, derived from the numbers rather than chosen.

    Order is the specification's and each step vetoes the ones after it: a build that does not
    build cannot be correct, a build that does not reproduce itself cannot serve as a reference
    for anything, correctness precedes any performance consideration, an overfitted shape is not
    a speedup, a partial matrix credits nothing, and confidence qualifies rather than scales.
    """
    if correctness == "BUILD_FAIL":
        return "BUILD_FAIL"
    if determinism is False:
        return "DETERMINISM_FAIL"
    if correctness != "PASS":
        return "CORRECTNESS_FAIL"
    if held_out is False:
        return "SHAPE_OVERFIT"
    if not coverage.get("complete", False):
        return "PARTIAL"
    if not resolved:
        return "UNRESOLVED"
    if gap_closed > 0 and frontier_delta > 0:
        return "FRONTIER_EXPANDED"
    if gap_closed > 0:
        return "MOVED_ALONG_FRONTIER"
    if frontier_delta > 0:
        return "EXPANDED_OFF_LATENCY"
    return "NO_GAIN"


def content_digest(receipt: dict) -> str:
    """SHA-256 over everything except the digest and signature fields themselves."""
    scored = {k: v for k, v in receipt.items() if k not in ("content_digest", "signature")}
    return "sha256:" + hashlib.sha256(
        json.dumps(scored, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_receipt(*, generation, per_cell, aggregate, interval, frontier, correctness,
                  determinism, coverage, held_out, provenance, pr=None, supersedes=None,
                  supersede_reason=None, timestamp=None) -> dict:
    """Assemble the canonical receipt.

    `per_cell` is one entry per scored cell carrying, at minimum: gap_closed, achieved before
    and after, the cell's floor, whether the cell's own result resolved, and the raw paired
    timings. The raw timings are in the receipt on purpose -- a receipt that cannot be
    recomputed from its own contents is a claim, not evidence.
    """
    resolved = bool(interval.get("resolved"))
    gap = float(aggregate["gap_closed"])
    dhv = float(frontier.get("delta", 0.0))
    status = decide_status(correctness=correctness, determinism=determinism,
                           coverage=coverage, held_out=held_out, resolved=resolved,
                           gap_closed=gap, frontier_delta=dhv)
    credited = gap if status in CREDITING else 0.0

    credit_withheld = None
    if gap > 0 and credited == 0.0:
        credit_withheld = {
            "measured_gap_closed": gap, "status": status,
            "reason": {
                "PARTIAL": "the matrix was incomplete; dropping the cell a change hurts is the "
                           "cheapest way to raise a score",
                "UNRESOLVED": "the observed difference is inside the noise floor of the cells "
                              "that produced it",
                "MOVED_ALONG_FRONTIER": "latency improved and the frontier did not: the gain "
                                        "was paid for in memory or in output quality, which is "
                                        "a trade the runtime could already make",
                "SHAPE_OVERFIT": "the gain did not survive the held-out shape",
                "CORRECTNESS_FAIL": "correctness precedes speed and is not traded against it",
                "DETERMINISM_FAIL": "a build that does not reproduce itself cannot be a "
                                    "reference for anything",
            }.get(status, "see status"),
            "_note": "The measured figure is reported. What is withheld is the credit.",
        }

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_generation": generation.name,
        "generation_digest": generation.digest,
        "timestamp_utc": timestamp or datetime.now(timezone.utc).isoformat(),
        "pr": pr,
        "status": status,
        "score": {
            "gap_closed": gap,
            "credited_gap_closed": credited,
            "confidence_interval": {"lower": interval.get("lower"),
                                    "upper": interval.get("upper"),
                                    "level": interval.get("level")},
            "resolved": resolved,
            "resolution_rule": interval.get("rule"),
            "aggregation": aggregate.get("method"),
            "basis": "measured",
            "_what_this_means": (
                "The fraction of each cell's REMAINING arithmetic-roofline gap that this change "
                "closed, weighted across the matrix. 0.25 means a quarter of what was left. It "
                "is scale-free, comparable across cells, and deliberately harder near the "
                "ceiling -- and it self-terminates, so grinding an exhausted cell stops paying "
                "and opening a new one starts paying more."),
        },
        "frontier": frontier,
        "per_cell": per_cell,
        "correctness": correctness,
        "determinism": determinism,
        "coverage": coverage,
        "held_out_survived": held_out,
        "credit_withheld": credit_withheld,
        "provenance": _with_completeness(with_calibration(provenance, generation)),
        "supersedes": supersedes or [],
        "supersede_reason": supersede_reason,
    }
    receipt["content_digest"] = content_digest(receipt)
    return receipt



# Provenance fields that identify the CODE a receipt scored, as opposed to the box it ran on.
# A receipt that cannot name them is still a valid measurement -- it just is not evidence about
# any particular commit, which is a different and much weaker thing.
_CODE_PROVENANCE = ("candidate_commit", "base_commit", "instrument_from")


def with_calibration(provenance: dict, generation) -> dict:
    """Stamp the receipt with the anchor it was scored against.

    Any card of the pinned class scores against the same anchor, so two receipts for one
    submission share it. Naming it is what makes a re-anchor visible: a challenge between a
    receipt scored before a re-anchor and one scored after would otherwise look like a
    disagreement about the measurement.
    """
    p = dict(provenance or {})
    p["calibration"] = {
        "device_uuid": generation.calibration_device,
        "device_name": generation.calibration_device_name,
        "driver_version": generation.calibration_driver,
        "_why": ("The card the anchor's achieved fractions and floors were measured on. The "
                 "run itself may come from any card of the pinned class: its score takes only "
                 "the paired base/candidate ratio from the run."),
    }
    return p


def _with_completeness(provenance: dict) -> dict:
    """State outright whether this receipt can say what code it scored.

    These fields go null whenever the runner has no git metadata -- a tarball deploy onto a
    benchmark box does it, and so does a checkout with the instrument overlay skipped. Null is
    an honest answer, but a reader skimming a receipt reads a null field as "not applicable"
    rather than as "unknown", and those are opposite meanings when the question is whether a
    submission graded its own homework. So the receipt answers the question in one flag instead
    of leaving it to be inferred from three absences.
    """
    p = dict(provenance or {})
    missing = [k for k in _CODE_PROVENANCE if not p.get(k)]
    p["code_provenance_complete"] = not missing
    p["code_provenance_missing"] = missing
    if missing:
        p["_code_provenance_note"] = (
            "This receipt records a measurement but cannot name the code it measured: "
            + ", ".join(missing) + " unknown. It is reproducible as an experiment and is NOT "
            "admissible as evidence that a particular commit earned a score. A scored "
            "submission runs through eval/run_from_base.sh in a git checkout, which fills all "
            "three.")
    return p


def verify_receipt(receipt: dict, generation=None) -> None:
    """Recompute what the receipt asserts about itself. Raises on the first disagreement."""
    for key in ("schema_version", "benchmark_generation", "status", "score", "per_cell",
                "provenance", "content_digest"):
        if key not in receipt:
            raise ReceiptError(f"receipt is missing {key!r}")
    if receipt["status"] not in STATUSES:
        raise ReceiptError(f"unknown status {receipt['status']!r}")
    if content_digest(receipt) != receipt["content_digest"]:
        raise ReceiptError(
            "content digest does not match the receipt body. A finalized receipt is not "
            "rewritten; if an evaluator bug requires a correction, the answer is a NEW receipt "
            "whose `supersedes` names this one.")

    score = receipt["score"]
    if score.get("basis") != "measured":
        raise ReceiptError(
            f"score basis is {score.get('basis')!r}. Only a measured run is evidence about a "
            f"speedup; a cost-model figure carries basis 'model' and is never a score.")
    credited = score.get("credited_gap_closed", 0.0)
    if receipt["status"] not in CREDITING and credited != 0.0:
        raise ReceiptError(
            f"status {receipt['status']} credits nothing, but credited_gap_closed is "
            f"{credited}. Withholding credit is the whole mechanism; a receipt that credits a "
            f"non-crediting status has disabled it.")
    if receipt["status"] in CREDITING and not score.get("resolved"):
        raise ReceiptError("a crediting status requires resolved=true")

    for cell_id, c in (receipt["per_cell"] or {}).items():
        for k in ("gap_closed", "achieved_base", "achieved_candidate", "floor_pct", "resolved"):
            if k not in c:
                raise ReceiptError(f"per_cell[{cell_id}] is missing {k!r}")
        if c.get("ceiling_basis") not in ("model", None):
            raise ReceiptError(
                f"per_cell[{cell_id}] declares ceiling_basis {c.get('ceiling_basis')!r}. A "
                f"roofline is arithmetic; calling it measured would make it evidence it is not.")
        a_b, a_c = c["achieved_base"], c["achieved_candidate"]
        for name, v in (("achieved_base", a_b), ("achieved_candidate", a_c)):
            if v is None or not (0.0 < v <= 1.0 + 1e-9):
                raise ReceiptError(
                    f"per_cell[{cell_id}].{name} = {v!r}. An achieved fraction above 1 means "
                    f"the run beat the arithmetic ceiling, which is a bug in the geometry, the "
                    f"device peak, or the run -- never a result.")

    if generation is not None:
        if receipt["benchmark_generation"] != generation.name:
            raise ReceiptError(f"receipt is for {receipt['benchmark_generation']}, "
                               f"generation is {generation.name}")
        if receipt.get("generation_digest") != generation.digest:
            raise ReceiptError(
                "the generation has changed since this receipt was written. A generation is "
                "frozen for its lifetime; if the meaning of the evaluation changed, the answer "
                "is a new generation, and this receipt stays attached to the old one.")
