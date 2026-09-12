"""The CLI paths, end to end, on synthetic measurements.

The modules are tested individually elsewhere. This file exists because the wiring between them
is its own failure surface: a scorer that works and a CLI that passes it the wrong generation
produces a confident receipt for the wrong thing, and no module test would notice.
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
sys.path.insert(0, str(HERE.parent))

from burnscore import cells as C
from tests import fixtures

BURNISH = [sys.executable, str(ROOT / "tools" / "burnish")]


def run(*args, **kw):
    return subprocess.run(BURNISH + list(args), capture_output=True, text=True,
                          cwd=str(ROOT), **kw)


class TestScoreCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.gen_name = "BG-TEST"
        self.cells_root, gpath = fixtures.scratch_generation(self.dir, self.gen_name)
        self.gen = C.load(gpath)

    def tearDown(self):
        self.tmp.cleanup()

    def _raw(self, **kw):
        recs = fixtures.records(self.gen, **kw)
        doc = {"generation": self.gen_name, "records": recs,
               "held_out": fixtures.records(self.gen, repeats=3, **kw),
               "correctness": "PASS", "determinism": True,
               "provenance": fixtures.provenance()}
        p = self.dir / "raw.json"
        p.write_text(json.dumps(doc))
        return p

    def test_a_clean_win_scores_and_the_receipt_verifies(self):
        raw = self._raw(speedups={"dit-step/1024/bf16": 1.15})
        out = self.dir / "receipt.json"
        r = run("score", str(raw), "--generation", self.gen_name, "--output", str(out),
                "--cells-root", str(self.cells_root))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("FRONTIER_EXPANDED", r.stdout)
        rec = json.loads(out.read_text())
        self.assertGreater(rec["score"]["credited_gap_closed"], 0)
        v = run("receipt", "verify", str(out), "--cells-root", str(self.cells_root))
        self.assertEqual(v.returncode, 0, v.stderr)

    def test_a_tampered_receipt_is_refused_by_the_cli(self):
        raw = self._raw(speedups={"dit-step/1024/bf16": 1.15})
        out = self.dir / "receipt.json"
        run("score", str(raw), "--generation", self.gen_name, "--output", str(out),
            "--cells-root", str(self.cells_root))
        rec = json.loads(out.read_text())
        rec["score"]["gap_closed"] = 0.99
        out.write_text(json.dumps(rec))
        v = run("receipt", "verify", str(out), "--cells-root", str(self.cells_root))
        self.assertEqual(v.returncode, 1)
        self.assertIn("digest does not match", v.stderr)

    def test_results_without_a_determinism_verdict_are_refused(self):
        """A build that has not passed the gate must not be scored: it would attribute noise to
        a candidate."""
        recs = fixtures.records(self.gen, speedups={"dit-step/1024/bf16": 1.15})
        p = self.dir / "nogate.json"
        p.write_text(json.dumps({"generation": self.gen_name, "records": recs,
                                 "correctness": "PASS",
                                 "provenance": fixtures.provenance()}))
        r = run("score", str(p), "--generation", self.gen_name,
                "--cells-root", str(self.cells_root))
        self.assertEqual(r.returncode, 2)
        self.assertIn("reproduces itself", r.stderr)

    def test_the_ledger_refuses_to_rewrite_history(self):
        raw = self._raw(speedups={"dit-step/1024/bf16": 1.15})
        ledger = self.dir / "ledger"
        r1 = run("score", str(raw), "--generation", self.gen_name,
                 "--cells-root", str(self.cells_root), "--ledger", str(ledger), "--pr", "7")
        self.assertEqual(r1.returncode, 0, r1.stderr)
        # A different result under the same PR id is a rewrite, and a rewrite is refused.
        raw2 = self._raw(speedups={"dit-step/1024/bf16": 1.30})
        r2 = run("score", str(raw2), "--generation", self.gen_name,
                 "--cells-root", str(self.cells_root), "--ledger", str(ledger), "--pr", "7")
        self.assertNotEqual(r2.returncode, 0)
        self.assertIn("supersedes", r2.stderr + r2.stdout)

    def test_the_ledger_compounds_rather_than_summing(self):
        ledger = self.dir / "ledger"
        for pr, speed in ((1, 1.15), (2, 1.10)):
            raw = self._raw(speedups={"dit-step/1024/bf16": speed})
            r = run("score", str(raw), "--generation", self.gen_name,
                    "--cells-root", str(self.cells_root), "--ledger", str(ledger),
                    "--pr", str(pr))
            self.assertEqual(r.returncode, 0, r.stderr)
        show = run("ledger", "show", "--root", str(ledger), "--generation", self.gen_name)
        doc = json.loads(show.stdout)
        summed = sum(h["credited_gap_closed"] for h in doc["history"])
        self.assertEqual(doc["crediting_receipts"], 2)
        self.assertLess(doc["gap_closed_cumulative"], summed)
        audit = run("ledger", "audit", "--root", str(ledger), "--generation", self.gen_name)
        self.assertTrue(json.loads(audit.stdout)["ok"])

    def test_a_partial_matrix_needs_the_flag_and_then_credits_nothing(self):
        recs = [r for r in fixtures.records(self.gen, speedups={"dit-step/1024/bf16": 1.15})
                if r["cell"] != "vae-decode/1024/bf16"]
        p = self.dir / "partial.json"
        p.write_text(json.dumps({"generation": self.gen_name, "records": recs,
                                 "correctness": "PASS", "determinism": True,
                                 "provenance": fixtures.provenance()}))
        r = run("score", str(p), "--generation", self.gen_name,
                "--cells-root", str(self.cells_root))
        self.assertNotEqual(r.returncode, 0)
        out = self.dir / "partial-receipt.json"
        r2 = run("score", str(p), "--generation", self.gen_name, "--allow-partial",
                 "--cells-root", str(self.cells_root), "--output", str(out))
        self.assertEqual(r2.returncode, 0, r2.stderr)
        rec = json.loads(out.read_text())
        self.assertEqual(rec["status"], "PARTIAL")
        self.assertEqual(rec["score"]["credited_gap_closed"], 0.0)
        self.assertGreater(rec["score"]["gap_closed"], 0.0)


class TestTestsDoNotPolluteTheTree(unittest.TestCase):
    """A test that leaves a calibrated generation in `eval/cells/` leaves the next
    `burnish generation show` reporting numbers nobody measured. This pins the invariant, because
    an interrupted run is exactly when it would break and exactly when nobody is watching."""

    def test_scratch_generations_are_written_outside_the_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, gpath = fixtures.scratch_generation(tmp, "BG-INVARIANT")
            self.assertTrue(gpath.exists())
            self.assertNotIn(str(ROOT / "eval" / "cells"), str(root.resolve()))
            self.assertFalse((ROOT / "eval" / "cells" / "BG-INVARIANT").exists())

    def test_the_committed_tree_holds_only_real_generations(self):
        present = sorted(p.name for p in (ROOT / "eval" / "cells").iterdir() if p.is_dir())
        self.assertEqual(present, ["BG-1"],
                         f"eval/cells holds {present}; a scratch generation leaked into the "
                         f"repository")


class TestGenerationCli(unittest.TestCase):
    def test_show_reports_every_bg1_cell_as_calibrated_and_scorable(self):
        r = run("generation", "show", "--name", "BG-1")
        self.assertEqual(r.returncode, 0, r.stderr)
        doc = json.loads(r.stdout)
        # This asserted the opposite until BG-1 was calibrated on the pinned box. The guard it
        # carried -- "a scorable cell means somebody filled in a measurement that was never
        # taken" -- is kept below, pointed at provenance instead of absence: a cell is scorable
        # only if a measurement with a basis and a box behind it says so.
        self.assertFalse(doc["uncalibrated"])
        ref = json.loads((ROOT / "eval" / "cells" / "BG-1" / "reference.json").read_text())
        self.assertEqual(sorted(doc["scorable_now"]), sorted(ref["cells"]),
                         "a cell is scorable that has no calibration behind it")
        for cid in doc["scorable_now"]:
            cal = ref["cells"][cid]
            self.assertEqual(cal["basis"], "measured", f"{cid} is scorable on a modelled figure")
            self.assertGreater(cal["floor_pct"], 0.0,
                               f"{cid} claims a zero noise floor; no instrument has one, and a "
                               f"zero floor resolves every difference including the noise")
            self.assertTrue(cal["resolvable"],
                            f"{cid} is offered for scoring but its ceiling does not clear its "
                            f"own floor")
        self.assertTrue(doc["digest"].startswith("sha256:"))

    def test_check_passes_against_the_committed_configs(self):
        r = run("generation", "check")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
