#!/usr/bin/env python3
"""Opening a cell is a different evaluation, and this is the half a submitter cannot fake.

The repository declares cartography payable and, until this existed, could not pay it: the guard
allowed a submission to ADD a generation, and then `run_from_base.sh` overlaid `eval/` from the
base commit and stripped it again. The submission was scored as an ordinary speedup against BG-1,
changed nothing measurable, and came back `unresolved`.

The fix has two halves and the security argument lives in the split:

    the submission supplies    the cell definition and the ORACLE -- what a correct runtime must
                               reproduce.
    the evaluator supplies     every MEASUREMENT. Submitted achieved fractions and floors are
                               read, reported and discarded; the cell is recalibrated here.

So the structural checks below can afford to be permissive about what is proposed, because
nothing a submission claims about its own performance survives to the receipt.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "eval"))

import cartography as CG
from make_generation import local_ceilings

SRC = ROOT / "eval" / "cells" / "BG-1"


class TestAProposedCell(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def propose(self, name, *, resolution=512, inflate_ceiling=False, duplicate=False,
                with_refs=True, claim_calibration=True):
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        doc = json.loads((SRC / "generation.json").read_text())
        doc["name"] = name
        if not duplicate:
            doc["model"] = dict(doc["model"], resolution=resolution)
            for c in doc["cells"]:
                c["id"] = c["id"].replace("/1024/", f"/{resolution}/")
                c["shape"] = dict(c["shape"], resolution=resolution)
            real = local_ceilings(doc)
            for c in doc["cells"]:
                if c["id"] in real:
                    c["ceiling_seconds"] = real[c["id"]]["ceiling_seconds"]
        if inflate_ceiling:
            doc["cells"][0]["ceiling_seconds"] *= 4.0
        (d / "generation.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
        if with_refs:
            (d / "reference-latents").mkdir(exist_ok=True)
            (d / "reference-latents" / "manifest.json").write_text('{"produced_by": "ref"}')
        if claim_calibration:
            # A flattering calibration: everything is nearly at its ceiling and nothing is noisy.
            (d / "reference.json").write_text(json.dumps({"cells": {
                c["id"]: {"achieved": 0.99, "floor_pct": 0.00001} for c in doc["cells"]}}))
        return d

    def _check(self, name):
        return CG.check(name, base="HEAD", root=self.root, verbose=False)

    def _failed(self, r):
        return [c["check"] for c in r["checks"] if not c["pass"]]

    def test_an_honest_new_generation_passes_the_structural_checks(self):
        self.propose("BG-NEW")
        r = self._check("BG-NEW")
        self.assertTrue(r["pass"], self._failed(r))
        # Every cell BG-1 declares, re-declared at 512px -- including the two that are
        # `implemented: false` (fp8, nvfp4). A declared-but-unimplemented cell is still new
        # surface: it publishes a ceiling so the room is visible, and carries weight 0 so it
        # cannot drag an aggregate it is not part of.
        self.assertIn("dit-step/512/bf16", r["novel_cells"])
        self.assertIn("t5-encode/512/bf16", r["novel_cells"])
        self.assertIn("vae-decode/512/bf16", r["novel_cells"])
        self.assertTrue(all("/512/" in c for c in r["novel_cells"]))

    def test_it_is_not_opened_until_it_has_been_measured(self):
        """A cell is opened by measurement, never by declaration.

        This is the outcome that matters most: a structurally perfect submission with a
        flattering calibration attached still pays nothing until the evaluator has run it.
        """
        self.propose("BG-NEW")
        r = self._check("BG-NEW")
        v = CG.verdict_for(r, measured=None)
        self.assertEqual(v["outcome"], "UNMEASURED")
        self.assertFalse(v["pays"])

    def test_the_submissions_own_calibration_is_read_and_not_used(self):
        """The flattering numbers reach the report and go no further."""
        self.propose("BG-NEW")
        r = self._check("BG-NEW")
        claimed = r["claimed_calibration"]
        self.assertTrue(claimed, "the claim is not even reported, so nobody can see it was made")
        self.assertEqual({c["achieved"] for c in claimed.values()}, {0.99})
        # ... and it does not make the cell payable.
        self.assertFalse(CG.verdict_for(r, measured=None)["pays"])

    def test_an_inflated_ceiling_is_recomputed_and_refused(self):
        """A submission that picks its own ceiling picks its own denominator, forever."""
        self.propose("BG-FAT", inflate_ceiling=True)
        r = self._check("BG-FAT")
        self.assertFalse(r["pass"])
        self.assertIn("every submitted ceiling recomputes from the base configs", self._failed(r))

    def test_a_generation_that_opens_nothing_is_refused(self):
        """Re-measuring ground the map already covers is not cartography."""
        self.propose("BG-DUP", duplicate=True)
        r = self._check("BG-DUP")
        self.assertFalse(r["pass"])
        self.assertIn("it declares at least one cell that does not already exist",
                      self._failed(r))

    def test_a_cell_with_no_oracle_is_refused(self):
        """A cell whose correctness cannot be gated is a cell where a wrong answer scores."""
        self.propose("BG-NOREF", with_refs=False)
        r = self._check("BG-NOREF")
        self.assertFalse(r["pass"])
        self.assertIn("reference latents are present with a manifest", self._failed(r))

    def test_an_existing_generation_cannot_be_resubmitted_as_new(self):
        r = CG.check("BG-1", base="HEAD", root=ROOT / "eval" / "cells", verbose=False)
        self.assertFalse(r["pass"])
        self.assertIn("the generation does not already exist on the base", self._failed(r))

    def test_an_unresolvable_cell_is_a_successful_contribution(self):
        """docs/CARTOGRAPHY.md says so, and the verdict has to agree with the document.

        A cell that runs correctly and cannot resolve a contribution is worth more than one that
        looks open and is not: the alternative is somebody spending a week inside a noise floor.
        """
        self.propose("BG-NEW")
        r = self._check("BG-NEW")
        v = CG.verdict_for(r, measured={
            "dit-step/512/bf16": {"resolvable": False, "achieved": 0.1, "floor_pct": 9.0,
                                  "floors_of_room": 3.0}})
        self.assertEqual(v["outcome"], "OPENED")
        self.assertTrue(v["pays"], "an unresolvable cell is a result, not a failure")
        self.assertEqual(v["unresolvable_cells"], ["dit-step/512/bf16"])


class TestTheOverlayKeepsAnAddedGeneration(unittest.TestCase):
    def test_run_from_base_preserves_only_generations_absent_from_the_base(self):
        script = (ROOT / "eval" / "run_from_base.sh").read_text()
        self.assertIn("BURNISH_CARTOGRAPHY", script)
        self.assertIn("BASE_CELLS", script)
        # The restriction is what makes it safe: only eval/cells/<absent from base>.
        self.assertIn('grep -qx "$n"', script,
                      "the overlay does not restrict what it keeps to generations that are "
                      "genuinely new, so a submission could smuggle an edit through it")

    def test_the_bot_routes_cartography_away_from_the_speedup_path(self):
        bot = (ROOT / "eval" / "pr_bot.py").read_text()
        self.assertIn('if g["outcome"] == "CARTOGRAPHY":', bot)
        self.assertIn("_evaluate_cartography", bot)

    def test_the_measurement_half_discards_what_was_submitted(self):
        src = (ROOT / "eval" / "cartography.py").read_text()
        self.assertIn("the evaluator, not the submission", src)
        self.assertIn("the submission's own calibration is discarded", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
