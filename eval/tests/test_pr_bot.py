#!/usr/bin/env python3
"""The bot's decisions, tested without GitHub and without a GPU.

Everything the bot decides is a pure function of things already on disk: a guard verdict, a
receipt, a label. Those are what is tested here. What is deliberately NOT tested is the network
-- `gh` calls are the one part that cannot be exercised offline, and mocking them would test the
mock.

The property worth protecting is that the bot adds no judgement of its own. If it ever starts
deciding something that is not derivable from a published artifact, the derivation stops being
checkable by anyone else, and the whole point of publishing the measurements goes with it.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "eval"))

import pr_bot as B
from burnscore import verdict as V

RECEIPT = ROOT / "examples" / "BG-1-pr-000001-receipt.json"


class TestTheBotDecidesNothingItself(unittest.TestCase):
    def setUp(self):
        self.receipt = json.loads(RECEIPT.read_text())

    def test_the_label_it_applies_is_the_one_burnish_verdict_derives(self):
        """Not 'the bot computes a label'. The bot READS one off a pure function.

        A bot that computed its own label would be the only thing that knew how the label was
        reached, and every contributor without a GPU would be asked to trust it.
        """
        self.assertEqual(V.verdict(self.receipt)["label"], V.label_for(self.receipt))

    def test_the_comment_tells_the_reader_how_to_check_it(self):
        v = V.verdict(self.receipt)
        body = B.report(v, self.receipt, raw_name="pr-000042-raw.json",
                        receipt_name="pr-000042.json")
        self.assertIn("burnish audit pr-000042-raw.json pr-000042.json", body)
        self.assertIn("no GPU", body)
        self.assertIn(v["label"], body)
        # The honest boundary is stated on every PR, not buried in a document nobody opens.
        self.assertIn("cannot prove the measurements describe", body)

    def test_the_comment_carries_the_number_and_the_interval(self):
        rec = json.loads(RECEIPT.read_text())
        rec["status"] = "FRONTIER_EXPANDED"
        rec["score"]["credited_gap_closed"] = 0.0342
        rec["score"]["resolved"] = True
        body = B.report(V.verdict(rec), rec, raw_name="r.json", receipt_name="x.json")
        self.assertIn("+0.0342", body)
        self.assertIn("interval", body)

    def test_an_unprovenanced_receipt_is_flagged_on_the_pr(self):
        rec = json.loads(RECEIPT.read_text())
        rec["provenance"] = dict(rec["provenance"], code_provenance_complete=False)
        body = B.report(V.verdict(rec), rec, raw_name="r.json", receipt_name="x.json")
        self.assertIn("not evidence about a particular commit", body)


def _pr(labels=(), head="a" * 40, body=""):
    return {"number": 5, "headRefOid": head, "body": body, "labels": [{"name": n} for n in labels]}


class TestWhenAPullRequestIsEvaluatedAgain(unittest.TestCase):
    def test_a_new_commit_is_evaluated_again_whatever_its_label(self):
        for label in ("burnish:needs-rebase", "burnish:build-fail", "burnish:unresolved",
                      "burnish:reregistered", "burnish:no-candidate"):
            self.assertTrue(B.needs_evaluation(_pr([label], head="b" * 40), {"head": "a" * 40}), label)
            self.assertFalse(B.needs_evaluation(_pr([label]),
                                                {"head": "a" * 40,
                                                 "body_sha256": B._body_sha(_pr())}), label)

    def test_an_evaluator_error_is_retried_a_bounded_number_of_times(self):
        rec = {"head": "a" * 40, "attempts": B.MAX_ATTEMPTS - 1}
        self.assertTrue(B.needs_evaluation(_pr([B.EVAL_ERROR]), rec))
        self.assertFalse(B.needs_evaluation(_pr([B.EVAL_ERROR]), dict(rec, attempts=B.MAX_ATTEMPTS)))

    def test_a_copy_waits_for_a_maintainer_not_a_push(self):
        self.assertFalse(B.needs_evaluation(_pr([B.COPYCAT], head="b" * 40), {"head": "a" * 40}))
        self.assertTrue(B.needs_evaluation(_pr([B.COPYCAT, B.CLEARED]), {"head": "a" * 40}))
        self.assertTrue(B.needs_evaluation(_pr([B.COPYCAT_REVIEW, B.CLEARED]), {"head": "a" * 40}))
        self.assertTrue(B.needs_evaluation(_pr([B.REREGISTERED, B.REREGISTRATION_CLEARED]),
                                           {"head": "a" * 40}))

    def test_naming_the_kernel_in_the_description_is_enough(self):
        rec = {"head": "a" * 40, "body_sha256": B._body_sha(_pr())}
        self.assertTrue(B.needs_evaluation(_pr([B.NO_CANDIDATE], body="**Implementation name:** `x`"),
                                           rec))

    def test_a_label_with_no_record_is_left_alone(self):
        self.assertFalse(B.needs_evaluation(_pr(["burnish:unresolved"]), None))
        self.assertTrue(B.needs_evaluation(_pr(), None))


class TestWhichKernelIsMeasured(unittest.TestCase):
    class Args:
        impl_candidate = ""

    def test_the_one_new_name_is_the_candidate(self):
        impl, _ = B.candidate_impl({"candidate_names": ["flash-sm120"]}, _pr(), self.Args())
        self.assertEqual(impl, "flash-sm120")

    def test_no_new_name_is_not_measured(self):
        impl, why = B.candidate_impl({"candidate_names": []}, _pr(), self.Args())
        self.assertIsNone(impl)
        self.assertIn("measured against itself", why)

    def test_several_names_need_the_description_and_only_among_those_names(self):
        rr = {"candidate_names": ["a", "b"]}
        self.assertIsNone(B.candidate_impl(rr, _pr(), self.Args())[0])
        self.assertEqual(B.candidate_impl(rr, _pr(body="**Implementation name:** `b`"),
                                          self.Args())[0], "b")
        self.assertIsNone(B.candidate_impl(rr, _pr(body="**Implementation name:** `cuda`"),
                                           self.Args())[0],
                          "the description picked a kernel the pull request does not register")

    def test_the_bot_measures_the_detected_kernel_not_cuda(self):
        src = (ROOT / "eval" / "pr_bot.py").read_text()
        self.assertIn('"--impl-candidate", impl,', src)
        self.assertNotIn('"--impl-candidate", required=False, default="cuda"', src)


class TestTheInstrumentSkipNote(unittest.TestCase):
    def test_it_says_skipped_not_rejected_and_points_at_cartography(self):
        """A contributor who reads 'rejected' stops. One who reads 'separated' resubmits.

        The distinction matters more here than in most benchmarks, because one of the two things
        this repository pays for -- opening a cell -- looks superficially like the thing it
        forbids.
        """
        note = B._skip_note({"blocked": [
            {"path": "eval/burnscore/floor.py", "why": "modifies the measuring instrument."}]})
        self.assertIn("not a rejection", note.lower())
        self.assertIn("no GPU time was spent", note)
        self.assertIn("cartography", note.lower())
        self.assertIn("eval/burnscore/floor.py", note)

    def test_the_skip_label_is_not_an_outcome_label(self):
        """Nothing was measured, so it must not sit in the same namespace as a measurement."""
        self.assertNotIn(B.SKIPPED, V.all_labels())
        self.assertTrue(B.SKIPPED.startswith(f"{V.PREFIX}:"))


class TestLabelSetupCoversWhatTheBotCanApply(unittest.TestCase):
    def test_every_label_in_the_code_is_created_by_the_setup_script(self):
        """The setup script reads the list from the code rather than repeating it.

        A hand-maintained second list is how a bot comes to apply a label that does not exist,
        which on GitHub is a silent no-op -- and a silently dropped label costs a contributor
        their credit.
        """
        script = (ROOT / "eval" / "setup_labels.sh").read_text()
        self.assertIn("from burnscore import verdict as V", script)
        self.assertIn("V.all_labels()", script)
        # The one label not in all_labels() must be added explicitly, and is.
        self.assertIn("skipped-instrument", script)


class TestEveryOutcomeRenders(unittest.TestCase):
    """The sample document is generated, so it cannot describe a comment nobody would receive.

    `scripts/sample_outcomes.py` renders what a pull request gets back for each outcome, using
    the same two functions the bot uses. A hand-written version of that document would drift the
    first time a label or a sentence changed, and the people it misleads are exactly the ones
    deciding whether to spend a week on a kernel.
    """

    def test_the_sample_renderer_produces_every_outcome_without_crashing(self):
        import subprocess
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "sample_outcomes.py")],
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        for expected in ("burnish:gap+0.0342", "burnish:unresolved", "burnish:no-gain",
                         "burnish:shape-overfit", "burnish:correctness-fail",
                         "burnish:skipped-instrument"):
            self.assertIn(expected, r.stdout, f"{expected} does not render")

    def test_a_derived_sample_is_labelled_as_derived(self):
        """Illustrative figures must never read as measurements.

        This repository's whole discipline is the model/measured distinction, so blurring it in
        the document that shows people what a score looks like would be absurd.
        """
        import subprocess
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "sample_outcomes.py")],
                           capture_output=True, text=True, timeout=120)
        self.assertIn("figures: MEASURED", r.stdout)
        self.assertIn("figures: DERIVED", r.stdout)

    def test_a_gain_shows_the_achieved_fraction_going_up(self):
        """The achieved column has to follow the gap, not sit beside it.

        The first version of the renderer set a gap and kept the original run's achieved column,
        so the paying sample showed a gain with the achieved fraction going DOWN.
        """
        import subprocess
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "sample_outcomes.py"),
                            "--only", "gap"], capture_output=True, text=True, timeout=120)
        row = next(l for l in r.stdout.splitlines() if "dit-step/1024/bf16" in l)
        # Backtick-delimited: ['| ', cell, ' | ', gap, ' | ', achieved, ...]
        achieved = next(f for f in row.split("`") if "->" in f)
        before, after = [float(x.strip(" %")) for x in achieved.split("->")]
        self.assertGreater(after, before,
                           f"a paying sample shows achieved going {before}% -> {after}%")


if __name__ == "__main__":
    unittest.main(verbosity=2)
