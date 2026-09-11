"""The Burnisher Ledger: append-only receipt history, written outside the candidate's reach.

    eval/cells/
      BG-1/
        generation.json        the frozen definition
        reference.json         calibrated achieved fractions and per-cell noise floors
        current.json           the running totals and every receipt that moved them
        receipts/pr-000184.json

Two rules, enforced here rather than described:

* **A finalized receipt is never silently rewritten.** Writing over an existing id with
  different content is refused. A correction is a NEW receipt naming what it supersedes and why.
* **A receipt stays attached to the generation that produced it.** When BG-2 launches, BG-1
  receipts are not re-scored and not migrated.

The ledger path is deliberately an argument rather than a constant: the evaluator writes it
somewhere the candidate's branch cannot reach. A ledger inside the tree being scored is a ledger
a submission can edit, and editing one does not look like cheating in a diff.
"""
from __future__ import annotations

import json
from pathlib import Path

from .receipt import content_digest, verify_receipt, ReceiptError, CREDITING


class LedgerError(ValueError):
    """The ledger would have to lose or rewrite history to accept this."""


def receipt_path(root, generation_name, receipt_id) -> Path:
    return Path(root) / generation_name / "receipts" / f"{receipt_id}.json"


def default_receipt_id(receipt: dict) -> str:
    pr = receipt.get("pr")
    if pr:
        return f"pr-{int(pr):06d}"
    commit = ((receipt.get("provenance") or {}).get("candidate_commit") or "unknown")[:12]
    return f"commit-{commit}"


def append_receipt(root, receipt: dict, *, receipt_id=None, generation=None) -> Path:
    verify_receipt(receipt, generation)
    name = receipt["benchmark_generation"]
    receipt_id = receipt_id or default_receipt_id(receipt)
    path = receipt_path(root, name, receipt_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("content_digest") == receipt.get("content_digest"):
            return path                      # idempotent: the same receipt, written again
        raise LedgerError(
            f"{path} already holds a different receipt ({existing.get('content_digest')}). "
            f"A finalized receipt is not rewritten. If an evaluator bug requires a correction, "
            f"write a NEW receipt whose `supersedes` names this one and whose "
            f"`supersede_reason` says why.")
    path.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    update_current(root, name)
    return path


def load_receipts(root, generation_name) -> list:
    d = Path(root) / generation_name / "receipts"
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            out.append((p.stem, json.loads(p.read_text())))
        except json.JSONDecodeError as exc:
            raise LedgerError(f"{p}: {exc}")
    return out


def update_current(root, generation_name) -> dict:
    """Recompute the generation's running totals from every canonical receipt.

    Gap-closed COMPOUNDS toward the ceiling rather than summing, and this is the arithmetic that
    makes the whole scoring model coherent. Each receipt closes a fraction of what was left AT
    ITS OWN MOMENT, so two contributions of 0.25 leave `(1-0.25)*(1-0.25) = 0.5625` of the
    original gap, which is 0.4375 closed -- not 0.5. Summing them would let a generation report
    more than 100% of a gap closed, which is not a rounding problem, it is a claim that the
    runtime is faster than arithmetic allows.
    """
    receipts = load_receipts(root, generation_name)
    superseded = {old for _, r in receipts for old in (r.get("supersedes") or [])}
    canonical = [(i, r) for i, r in receipts if i not in superseded]

    remaining = 1.0
    history = []
    for rid, r in sorted(canonical, key=lambda x: x[1].get("timestamp_utc") or ""):
        credited = float(r.get("score", {}).get("credited_gap_closed", 0.0) or 0.0)
        if r.get("status") in CREDITING and credited > 0:
            remaining *= (1.0 - min(credited, 1.0))
        history.append({
            "receipt": rid, "pr": r.get("pr"), "timestamp_utc": r.get("timestamp_utc"),
            "candidate_commit": (r.get("provenance") or {}).get("candidate_commit"),
            "status": r.get("status"),
            "gap_closed": r.get("score", {}).get("gap_closed"),
            "credited_gap_closed": credited,
            "content_digest": r.get("content_digest"),
        })

    doc = {
        "generation": generation_name,
        "canonical_receipts": len(canonical),
        "superseded_receipts": sorted(superseded),
        "crediting_receipts": sum(1 for h in history if h["status"] in CREDITING),
        "gap_remaining_fraction": remaining,
        "gap_closed_cumulative": 1.0 - remaining,
        "_compounding_note": (
            "Compounded toward the ceiling, not summed: each receipt closes a fraction of what "
            "was left at its own moment, so two 0.25 contributions close 0.4375 together and "
            "not 0.5. Summing would let a generation claim more gap closed than existed."),
        "history": history,
    }
    p = Path(root) / generation_name / "current.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc


def show(root, generation_name) -> dict:
    p = Path(root) / generation_name / "current.json"
    return json.loads(p.read_text()) if p.exists() else update_current(root, generation_name)


def audit(root, generation_name, generation=None) -> dict:
    """Verify every receipt in a generation. Returns the failures rather than raising."""
    problems = []
    receipts = load_receipts(root, generation_name)
    for rid, r in receipts:
        try:
            verify_receipt(r, generation)
        except ReceiptError as exc:
            problems.append({"receipt": rid, "problem": str(exc)})
        else:
            if r.get("content_digest") != content_digest(r):
                problems.append({"receipt": rid, "problem": "digest mismatch"})
    return {"generation": generation_name, "checked": len(receipts),
            "problems": problems, "ok": not problems}
