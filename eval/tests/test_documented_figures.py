#!/usr/bin/env python3
"""Every measured figure quoted in prose must still match the artifact it came from.

`issues/` is generated, so its numbers cannot go stale. README.md and docs/STATUS.md are written
by hand, and they are the two documents anybody reads first -- which makes them the two places a
stale number does the most damage. They had gone stale by exactly one round of measurement:
STATUS.md was still saying "every cell's achieved fraction is null" and "the runtime cannot
currently run on a device at all" after calibration had run, and README.md still said the CUDA op
backend did not exist.

Nobody noticed because nothing checked. So this checks.

It is deliberately crude: it renders each figure from its artifact exactly as the prose should
spell it, and asserts that string appears in the document. That means re-measuring forces the
prose to be edited, which is the point -- a figure that can drift silently is a figure that will.
The failure message says which artifact to read.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CELL = ROOT / "eval" / "cells" / "BG-1"


def _load(name):
    return json.loads((CELL / name).read_text())


class TestProseMatchesTheArtifacts(unittest.TestCase):
    def setUp(self):
        self.cal = _load("reference.json")["cells"]
        self.lat = _load("dtype-latency.json")
        self.readme = (ROOT / "README.md").read_text()
        self.status = (ROOT / "docs" / "STATUS.md").read_text()

    def _want(self, doc, doc_name, text, artifact):
        """Assert a figure appears, tolerating the two ways a document spells a ratio.

        Prose uses the multiplication sign (24x with U+00D7) where code and artifacts use ASCII.
        Demanding one spelling would make the check fail on correct prose, and the first version
        of it did -- so both are accepted and neither is preferred.

        The failure message quotes the missing figure, NOT the document: the first version
        interpolated the whole of STATUS.md into an assertion message, which buried the one line
        that mattered under thirty kilobytes of the file being checked.
        """
        variants = {text, text.replace("x", "\u00d7"), text.replace("\u00d7", "x")}
        self.assertTrue(any(v in doc for v in variants),
                        f"{doc_name} no longer quotes {text!r}. It is a MEASURED figure and its "
                        f"source of truth is {artifact}; if that artifact changed, the prose has "
                        f"to change with it rather than keep the old number.")

    def test_the_gap_to_pytorch_is_quoted_from_its_artifact(self):
        pt = _load("pytorch-baseline.json")
        src = "eval/cells/BG-1/pytorch-baseline.json"
        for c in pt["comparison"].values():
            ratio = f"{c['burnisher_over_pytorch']:.1f}\u00d7"
            self._want(self.readme, "README.md", ratio, src)
            self._want(self.status, "docs/STATUS.md", ratio, src)
        whole = f"{pt['whole_image']['median_s']:.2f} s"
        self._want(self.readme, "README.md", whole, src)
        self._want(self.status, "docs/STATUS.md", whole, src)
        dit = "dit-step/1024/bf16"
        reached = f"{100 * self.cal[dit]['ceiling_seconds'] / pt['stages'][dit]['median_s']:.1f}%"
        self._want(self.status, "docs/STATUS.md", reached, src)

    def test_the_readme_quotes_the_calibrated_achieved_fractions(self):
        for cid in ("dit-step/1024/bf16", "vae-decode/1024/bf16", "t5-encode/1024/bf16"):
            pct = f"{100 * self.cal[cid]['achieved']:.1f}%"
            self._want(self.readme, "README.md", pct, "eval/cells/BG-1/reference.json")

    def test_the_status_page_quotes_the_calibrated_floors(self):
        for cid, cal in self.cal.items():
            self._want(self.status, "docs/STATUS.md", f"{cal['floor_pct']:.3f}%",
                       "eval/cells/BG-1/reference.json")

    def test_both_documents_quote_the_dtype_measurement_consistently(self):
        fp32 = f"{self.lat['dit_step_fp32_s']:.3f} s"
        bf16 = f"{self.lat['dit_step_bf16_s']:.3f} s"
        for doc, name in ((self.readme, "README.md"), (self.status, "docs/STATUS.md")):
            self._want(doc, name, fp32, "eval/cells/BG-1/dtype-latency.json")
            self._want(doc, name, bf16, "eval/cells/BG-1/dtype-latency.json")

    def test_the_documented_scoring_cost_matches_what_the_screen_computes(self):
        """docs/STATUS.md quotes what a submission costs. The screen computes it.

        Two places carrying one number is how the number goes wrong, and this one matters more
        than most: it is the figure a validator uses to decide whether it can afford to run.
        """
        import subprocess, sys, tempfile
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            subprocess.run([sys.executable, str(ROOT / "eval" / "screen.py"),
                            "--json", f.name], check=True, capture_output=True)
            doc = json.loads(Path(f.name).read_text())
        sc = doc["results"]["pixart-sigma-xl2-1024"]["score_cost"]
        self.assertEqual(sc["basis"], "measured")
        minutes = sc["predicted_receipt_seconds"] / 60
        self._want(self.status, "docs/STATUS.md", f"{minutes:.0f} modelled-from-measurement",
                   "eval/screen.py SCORE_COST")
        # The verdict is the screen's, and the document states it rather than deciding it. It
        # was a hard-coded "fails", which stopped being true the day the runtime got fast.
        verdict = "SCORE_COST as PASS" if sc["pass"] else "SCORE_COST as FAIL"
        self._want(self.status, "docs/STATUS.md", verdict, "eval/screen.py SCORE_COST")

    def test_the_floor_instability_claim_has_its_measurement_in_the_tree(self):
        """docs/EVAL.md and docs/STATUS.md publish how far floors move between sessions.

        For a while the only place those numbers existed in this repository was a test
        docstring, which is prose. A published figure whose measurement is not in the tree
        cannot be checked by anybody -- the one thing this repository is built not to do. Both
        sessions of the current anchor are in the tree, and so are the first kernels' two, whose
        24x is what made the anchor keep the worst floor of its sessions.
        """
        p = CELL / "calibration-session-2.json"
        self.assertTrue(p.exists(),
                        "the second calibration session is not committed, so the floor "
                        "instability claim cites a measurement nobody can see")
        second = json.loads(p.read_text())
        self.assertTrue(second["same_card_as_reference"],
                        "the claim is about one card measured twice; this is a different card")
        eval_doc = (ROOT / "docs" / "EVAL.md").read_text()
        src = "eval/cells/BG-1/reference.json and calibration-session-2.json"
        worst = 0.0
        for cid, c in self.cal.items():
            # The anchor's floor is the merged one; its own session measured `floor_spread_pct`.
            a, b = c["floor_spread_pct"], second["cells"][cid]["floor_pct"]
            worst = max(worst, max(a, b) / min(a, b))
            # The other half of the claim: achieved held while the floor moved. Two places, not
            # three: at a hundred milliseconds a step, the fraction moves in the third.
            self.assertAlmostEqual(c["achieved"], second["cells"][cid]["achieved"], places=2,
                                   msg=f"{cid}: achieved moved between sessions too, so the "
                                       f"documented contrast is wrong")
            self._want(eval_doc, "docs/EVAL.md", f"{a:.3f}%", src)
            self._want(eval_doc, "docs/EVAL.md", f"{b:.3f}%", src)
        self._want(self.status, "docs/STATUS.md", f"{worst:.1f}\u00d7", src)
        v0 = json.loads((ROOT / "examples" / "BG-1-anchor-v0.json").read_text())["cells"]
        v0b = json.loads((ROOT / "examples" / "BG-1-anchor-v0-session-2.json").read_text())["cells"]
        v0_worst = max(max(v0[c]["floor_pct"], v0b[c]["floor_pct"]) /
                       min(v0[c]["floor_pct"], v0b[c]["floor_pct"]) for c in v0)
        for doc, name in ((eval_doc, "docs/EVAL.md"), (self.status, "docs/STATUS.md")):
            self._want(doc, name, f"{v0_worst:.1f}\u00d7", "examples/BG-1-anchor-v0-session-2.json")

    def test_the_round_budget_comes_from_an_artifact(self):
        """`eval/round.py` decides how many submissions fit an interval. That number has to be
        checkable, and it was two constants whose provenance was a file mtime."""
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "eval"))
        import round as RD
        cost = json.loads((CELL / "round-cost.json").read_text())
        self.assertEqual(cost["basis"], "measured")
        self.assertEqual(RD.MEASURED_MINUTES_PER_PR, cost["total_minutes_warm_cache"])
        self.assertEqual(RD.COLD_GATE_MINUTES, cost["stages_minutes"]["gate_base"])
        # The stage breakdown must add up to the published total.
        stages = cost["stages_minutes"]
        cold = stages["gate_base"] + stages["gate_candidate"] + stages["bench"] + stages["score"]
        self.assertAlmostEqual(cold, cost["total_minutes_cold_cache"], places=1)
        self.assertAlmostEqual(cold - stages["gate_base"],
                               cost["total_minutes_warm_cache"], places=1)
        # And the documented per-PR figure must be this one.
        self._want(self.status, "docs/STATUS.md",
                   f"{cost['total_minutes_warm_cache']:.0f} min",
                   "eval/cells/BG-1/round-cost.json")

    def test_the_gate_records_what_it_cost(self):
        """The round budget was inferred from artifact mtimes because nothing measured the gate.
        It measures itself now, so the next run produces this directly."""
        src = (ROOT / "eval" / "gate.py").read_text()
        self.assertIn('report["wall_seconds"]', src)
        self.assertIn("_gate_started", src)

    def test_neither_document_still_claims_the_repository_is_unmeasured(self):
        """The specific sentences that were true before the hardware arrived and false after.

        Kept as a list rather than a single check because each one was a separate claim in a
        separate place, and a reader who hits any one of them concludes the whole page is stale.
        """
        stale = [
            "Nothing has been measured on a GPU",
            "the CUDA op backend does not exist yet",
            "there is no CUDA implementation of any op",
            "It has never been near a compiler",
            "the runtime cannot currently run on a device at all",
            "Reference latents for the gate | **do not exist**",
            "Every cell's achieved fraction | **null**",
            "Every cell's noise floor | **null**",
        ]
        for doc, name in ((self.readme, "README.md"), (self.status, "docs/STATUS.md")):
            for claim in stale:
                self.assertNotIn(claim, doc,
                                 f"{name} still says {claim!r}, which stopped being true when "
                                 f"the runtime was calibrated on the pinned part.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
