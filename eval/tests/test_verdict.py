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


class TestLabelColour(unittest.TestCase):
    """Colour by MEANING, and the paying label's shade carries the magnitude.

    Two things go wrong if this is left alone. Colouring by severity paints `unresolved` and
    `correctness-fail` the same alarming red, when one is "we could not measure your idea" and
    the other is "your change is incorrect". And the paying label cannot be pre-registered --
    it carries the measured number, so there is one per value -- which means GitHub creates it
    on first use with a RANDOM colour. The most important outcome in the system came out a
    different shade every time, occasionally red.
    """

    def _paying(self, gap):
        return {"status": "FRONTIER_EXPANDED", "score": {"credited_gap_closed": gap}}

    def test_every_outcome_has_a_colour(self):
        for status in R.STATUSES:
            self.assertIn(status, V.COLORS, f"{status} has no colour")
            self.assertRegex(V.COLORS[status], r"^[0-9A-F]{6}$")

    def test_a_bigger_contribution_is_a_deeper_green(self):
        """Monotone, so a reader scanning a list sees relative size without reading numbers."""
        def luminance(hexcolor):
            r, g, b = (int(hexcolor[i:i + 2], 16) for i in (0, 2, 4))
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        gaps = [0.0002, 0.001, 0.005, 0.02, 0.08, 0.3]
        lums = [luminance(V.color_for(self._paying(g))) for g in gaps]
        self.assertEqual(lums, sorted(lums, reverse=True),
                         f"the ramp is not monotone: {list(zip(gaps, lums))}")

    def test_the_ramp_is_logarithmic_not_linear(self):
        """Real values span orders of magnitude -- the tightest cell's floor is worth 0.00009 of
        the gap and a large win is 0.1. A linear ramp paints everything below a tenth the same
        pale colour, which is most of what will ever be earned."""
        def lum(g):
            c = V.color_for(self._paying(g))
            return sum(int(c[i:i + 2], 16) for i in (0, 2, 4))
        # A decade near the bottom must move the colour comparably to a decade near the top.
        low = lum(0.0005) - lum(0.005)
        high = lum(0.005) - lum(0.05)
        self.assertGreater(low, 0)
        self.assertGreater(high, 0)
        self.assertLess(abs(low - high) / max(low, high), 0.35,
                        f"the ramp is not close to logarithmic: {low} vs {high} per decade")

    def test_a_paying_outcome_is_never_a_rejection_colour(self):
        for g in (0.0001, 0.01, 0.9):
            self.assertNotIn(V.color_for(self._paying(g)), (V.RED, V.AMBER, V.ORANGE))

    def test_meaning_not_severity(self):
        """The distinction the palette exists to make."""
        unresolved = V.color_for({"status": "UNRESOLVED", "score": {}})
        wrong = V.color_for({"status": "CORRECTNESS_FAIL", "score": {}})
        self.assertNotEqual(unresolved, wrong,
                            "'we could not measure this' and 'this is incorrect' share a colour")
        self.assertEqual(wrong, V.RED)

    def test_the_setup_script_reads_colours_from_the_code(self):
        """One place for colour and meaning. A second copy in the shell script would drift
        silently -- a label with the wrong colour still works, so nobody notices."""
        script = (ROOT / "eval" / "setup_labels.sh").read_text()
        self.assertIn("V.COLORS", script)
        self.assertIn("V.ALL_OUTCOMES", script)
        self.assertNotIn("declare -A COLOR", script)

    def test_the_bot_creates_the_paying_label_before_attaching_it(self):
        bot = (ROOT / "eval" / "pr_bot.py").read_text()
        self.assertIn("def ensure_label(", bot)
        self.assertIn("color=V.color_for(receipt)", bot)


class TestTheDeclaredModelMatchesTheCode(unittest.TestCase):
    """`.gittensor/weights.json` declares how this repository pays. The code decides.

    A declaration that drifts from the implementation is worse than none: it is the document a
    contributor reads before deciding whether to spend a week, and whoever reads it has no way
    to know it went stale.
    """

    def setUp(self):
        self.decl = json.loads((ROOT / ".gittensor" / "weights.json").read_text())

    def test_every_label_the_code_can_apply_is_declared(self):
        for lab in V.all_labels():
            self.assertIn(lab, self.decl["outcomes"],
                          f"the code can apply {lab} and the declaration does not explain it")

    def test_the_paying_label_is_declared_with_its_shape(self):
        pattern = "burnish:gap+N.NNNN"
        self.assertIn(pattern, self.decl["outcomes"])
        rec = json.loads(RECEIPT.read_text())
        rec["status"] = "FRONTIER_EXPANDED"
        rec["score"]["credited_gap_closed"] = 0.0342
        actual = V.label_for(rec)
        self.assertEqual(len(actual), len(pattern),
                         f"the declared shape {pattern} is not the width of {actual}")

    def test_the_declared_payout_basis_is_the_field_the_code_pays_on(self):
        rec = json.loads(RECEIPT.read_text())
        rec["status"] = "FRONTIER_EXPANDED"
        rec["score"]["credited_gap_closed"] = 0.25
        self.assertEqual(self.decl["payout_basis"], "credited_gap_closed")
        self.assertEqual(V.verdict(rec)["payout_fraction"],
                         rec["score"][self.decl["payout_basis"]])

    def test_the_declaration_names_no_tier(self):
        blob = json.dumps(self.decl)
        for tier in ('"XL"', '"XS"', '"tier"'):
            self.assertNotIn(tier, blob, f"{tier} appears in a declaration that says it has none")


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


class TestAnAuditWorksWithoutTheValidatorsMachine(unittest.TestCase):
    """The property that makes "anyone can check this" true rather than aspirational.

    A receipt from any card is scored against the generation's anchor, and a generation is
    re-anchored when its base code changes. If the raw file merely NAMED the anchor, a receipt
    scored before a re-anchor would stop re-deriving the moment the committed one moved on.

    So the anchor travels inside the raw file. This file plus the frozen generation is everything
    needed to reproduce the receipt, on any machine, forever.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.raw = Path(self.tmp.name) / "raw.json"
        self.rec = Path(self.tmp.name) / "receipt.json"
        self.rec.write_text(RECEIPT.read_text())

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_receipt_from_another_box_audits_from_its_own_embedded_calibration(self):
        raw = json.loads(RAW.read_text())
        cal = json.loads((ROOT / "eval" / "cells" / "BG-1" / "reference.json").read_text())
        other = "GPU-aaaaaaaa-0000-0000-0000-000000000000"

        # Another validator's run: their card, their calibration, travelling together.
        raw["provenance"] = dict(raw["provenance"])
        raw["provenance"]["device"] = dict(raw["provenance"]["device"], uuid=other)
        raw["calibration"] = {
            "device_probe": dict(cal["device_probe"], uuid=other),
            "cells": {k: {"achieved": v["achieved"], "floor_pct": v["floor_pct"],
                          "ceiling_seconds": v["ceiling_seconds"],
                          "measured_seconds": v["measured_seconds"],
                          "floor_repeats": v["floor_repeats"]}
                      for k, v in cal["cells"].items()},
        }
        self.raw.write_text(json.dumps(raw))

        # The receipt as that validator would have published it.
        rec = json.loads(RECEIPT.read_text())
        rec["provenance"] = dict(rec["provenance"])
        rec["provenance"]["device"] = dict(rec["provenance"]["device"], uuid=other)
        rec["provenance"]["calibration"] = dict(rec["provenance"]["calibration"],
                                                device_uuid=other)
        rec.pop("content_digest")
        rec["content_digest"] = R.content_digest(rec)
        self.rec.write_text(json.dumps(rec))

        r = A.audit_one(self.raw, self.rec, verbose=False)
        self.assertTrue(r["pass"],
                        [c for c in r["checks"] if not c["pass"]])

    def test_without_its_embedded_anchor_a_receipt_stops_auditing_after_a_re_anchor(self):
        """The failure the embedding prevents, kept so the reason is not forgotten."""
        raw = json.loads(RAW.read_text())
        raw.pop("calibration", None)
        self.raw.write_text(json.dumps(raw))
        # The generation re-anchored after this receipt was scored: the base code got faster, so
        # the committed achieved fractions moved on.
        anchor = json.loads((ROOT / "eval" / "cells" / "BG-1" / "reference.json").read_text())
        for c in anchor["cells"].values():
            c["achieved"] *= 1.05
        moved = Path(self.tmp.name) / "re-anchored.json"
        moved.write_text(json.dumps(anchor))
        r = A.audit_one(self.raw, self.rec, calibration=moved, verbose=False)
        self.assertFalse(r["pass"])

    def test_the_bench_embeds_it(self):
        src = (ROOT / "eval" / "bench.py").read_text()
        self.assertIn('"calibration": {', src)
        self.assertIn("_why_embedded", src)


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
