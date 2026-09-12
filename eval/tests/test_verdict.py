#!/usr/bin/env python3
"""The verdict is a pure function of the receipt, and the audit catches a receipt that lies.

Two properties carry the whole eval design and both are asserted here rather than described:

  purity      The same receipt yields the same verdict on any machine, forever. That is what
              lets the verdict be a DERIVATION anybody can run instead of a decision a bot makes
              while it happens to be holding the numbers.

  sufficiency The audit must fail a receipt whose published score does not follow from its
              published measurements -- INCLUDING one whose content digest has been recomputed
              to cover the edit, which is what an attacker with a text editor would do.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "eval"))

import audit as A
from burnscore import cells as C
from burnscore import receipt as R
from burnscore import verdict as V

RAW = ROOT / "examples" / "BG-1-pr-000001-raw.json"
RECEIPT = ROOT / "examples" / "BG-1-pr-000001-receipt.json"


class TestTheVerdictIsAPureFunction(unittest.TestCase):
    def setUp(self):
        self.rec = json.loads(RECEIPT.read_text())

    def test_the_same_receipt_always_gives_the_same_verdict(self):
        a, b = V.verdict(self.rec), V.verdict(copy.deepcopy(self.rec))
        self.assertEqual(a, b)

    def test_a_paying_outcome_carries_the_number_and_nothing_else_does(self):
        """The label IS the payout basis when it pays, and a reason when it does not.

        A number published about a submission that was never resolved, never correct, or never
        measured would be read as a score. The taxonomy refuses to produce one.
        """
        rec = copy.deepcopy(self.rec)
        rec["status"] = "FRONTIER_EXPANDED"
        rec["score"]["credited_gap_closed"] = 0.0342
        self.assertEqual(V.label_for(rec), "burnish:gap+0.0342")
        self.assertEqual(V.verdict(rec)["payout_fraction"], 0.0342)

        for status in ("UNRESOLVED", "CORRECTNESS_FAIL", "SHAPE_OVERFIT", "PARTIAL"):
            rec["status"] = status
            lab = V.label_for(rec)
            self.assertNotIn("gap", lab, f"{status} publishes a number; it has not earned one")
            self.assertFalse(V.verdict(rec)["pays"])
            self.assertEqual(V.verdict(rec)["payout_fraction"], 0.0)

    def test_the_label_sorts_in_the_same_order_as_the_payout(self):
        """Fixed-width decimals, so a list of labels is already ranked.

        Significant figures would not do this: `gap+3.4e-2` and `gap+9.0e-3` sort the wrong way
        round, and whoever reads a sorted label list would silently get the ranking backwards.
        """
        values = [0.0001, 0.0009, 0.0342, 0.1000, 0.9999]
        labels = [V.format_gap(v) for v in values]
        self.assertEqual(labels, sorted(labels))
        self.assertEqual(len({len(x) for x in labels}), 1, "labels are not fixed width")

    def test_every_receipt_status_has_a_stated_meaning(self):
        """A contributor reading a label must be able to find out what it means.

        The vocabulary comes from the runtime's own STATUSES rather than a display-only set
        invented beside it: a status that means one thing in the receipt and another on the PR
        is how somebody comes to believe they were paid for something they were not.
        """
        for status in R.STATUSES:
            self.assertIn(status, V.ALL_OUTCOMES, f"{status} has no stated meaning")
            headline, meaning = V.ALL_OUTCOMES[status]
            self.assertTrue(headline and meaning)

    def test_no_letter_grades_anywhere_in_the_taxonomy(self):
        """The brief's requirement, asserted so it cannot creep back in.

        A tier boundary pays two differently-measured submissions the same, and two
        almost-identical ones differently. The number is already in [0, 1] and already
        comparable; bucketing it discards the only property worth having.
        """
        for lab in V.all_labels():
            tail = lab.split(":", 1)[1]
            self.assertNotIn(tail.upper(), ("XL", "L", "M", "S", "XS", "A", "B", "C", "D", "F"),
                             f"{lab} is a letter grade")


class TestTheAuditCatchesAReceiptThatLies(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.raw = Path(self.tmp.name) / "raw.json"
        self.rec = Path(self.tmp.name) / "receipt.json"
        self.raw.write_text(RAW.read_text())
        self.rec.write_text(RECEIPT.read_text())

    def tearDown(self):
        self.tmp.cleanup()

    def _audit(self):
        return A.audit_one(self.raw, self.rec, verbose=False)

    def test_the_committed_example_passes(self):
        r = self._audit()
        self.assertTrue(r["pass"], [c for c in r["checks"] if not c["pass"]])

    def test_a_fabricated_score_fails_even_with_the_digest_recomputed(self):
        """The attack an editor makes possible, and why the cheap check still wins.

        Anyone can change a number in a receipt and recompute its content digest -- the digest
        function is in this repository. That defeats `receipt verify`, which only asks whether a
        receipt covers its own body. It does NOT defeat the audit, because the score has to
        follow from thirty paired records that were committed beside it.
        """
        d = json.loads(self.rec.read_text())
        d["status"] = "FRONTIER_EXPANDED"
        d["score"]["gap_closed"] = 0.42
        d["score"]["credited_gap_closed"] = 0.42
        d["score"]["resolved"] = True
        d["content_digest"] = R.content_digest({k: v for k, v in d.items()
                                                if k != "content_digest"})
        self.rec.write_text(json.dumps(d, indent=1, sort_keys=True))

        gen = C.load(ROOT / "eval" / "cells" / "BG-1" / "generation.json")
        R.verify_receipt(d, gen)          # self-consistency alone is satisfied by the forgery

        r = self._audit()                 # the audit is not
        self.assertFalse(r["pass"])
        failed = [c["check"] for c in r["checks"] if not c["pass"]]
        self.assertIn("re-scoring the raw measurements reproduces the receipt", failed)

    def test_measurements_edited_to_match_a_fabricated_score_change_the_score_again(self):
        """Why faking the RAW file is not an easier attack than faking the receipt.

        The obvious next move is to edit the measurements instead. But the score is a function
        of all of them -- paired, interleaved, with a bootstrap over repeat indices and a drift
        guard against the calibration -- so moving the base arm to manufacture a speedup moves
        the achieved fraction too, and the drift guard notices.
        """
        raw = json.loads(self.raw.read_text())
        for rec in raw["records"]:
            if rec["variant"] == "base":
                rec["metrics"]["latency_s"] *= 1.5      # pretend the baseline was slower
        self.raw.write_text(json.dumps(raw, indent=1, sort_keys=True))
        r = self._audit()
        self.assertFalse(r["pass"], "a rewritten baseline was accepted")

    def test_a_receipt_scored_against_a_different_ruler_is_caught(self):
        d = json.loads(self.rec.read_text())
        d["generation_digest"] = "sha256:" + "0" * 64
        d["content_digest"] = R.content_digest({k: v for k, v in d.items()
                                                if k != "content_digest"})
        self.rec.write_text(json.dumps(d, indent=1, sort_keys=True))
        r = self._audit()
        self.assertFalse(r["pass"])


class TestTheAuditRunsFromTheCli(unittest.TestCase):
    def test_burnish_audit_exits_zero_on_the_committed_example(self):
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "burnish"), "audit",
                            str(RAW), str(RECEIPT)], capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("no GPU needed", r.stdout)

    def test_burnish_verdict_label_only_prints_one_line(self):
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "burnish"), "verdict",
                            str(RECEIPT), "--label-only"],
                           capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "burnish:unresolved")


if __name__ == "__main__":
    unittest.main(verbosity=2)
