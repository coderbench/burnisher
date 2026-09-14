#!/usr/bin/env python3
"""A score must not depend on which card of the pinned class measured it, or on calibrating it.

Rented boxes change constantly. A design in which every box calibrates itself before it can score
spends half an hour per box, and it is still not invariant: dividing a card's time into that card's
probed ceiling moves the score by the card's peak difference whenever the code is limited by
something other than that peak -- which, at a few percent of its ceiling, it is.

So a generation is anchored once. The anchor holds each cell's achieved fraction, the base time
behind it, and the worst noise floor seen in any session. A run contributes only its paired
base/candidate ratio: the ceiling in the run's own seconds is `achieved x base time`. These tests
pin that on the committed real measurements, with no GPU.
"""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from burnscore import cells as C
from burnscore import compute as CP

GEN = ROOT / "eval" / "cells" / "BG-1" / "generation.json"
RAW = ROOT / "examples" / "BG-1-pr-000001-raw.json"
OTHER_CARD = {"uuid": "GPU-bbbbbbbb-0000-0000-0000-000000000000", "name": "NVIDIA GeForce RTX 5090"}


def _scaled(records, base=1.0, candidate=1.0):
    out = copy.deepcopy(records)
    for r in out:
        r["metrics"]["latency_s"] *= base if r["variant"] == "base" else candidate
    return out


class TestAnyCardScoresTheSame(unittest.TestCase):
    def setUp(self):
        self.gen = C.load(GEN)
        self.records = json.loads(RAW.read_text())["records"]
        self.reference = CP.compute(self.gen, self.records)["aggregate"]["gap_closed"]

    def test_a_uniformly_slower_or_faster_card_scores_identically(self):
        """No calibration on the other card, and the same number to twelve places."""
        for k in (0.97, 1.03, 1.08):
            got = CP.compute(self.gen, _scaled(self.records, k, k),
                             device=OTHER_CARD)["aggregate"]["gap_closed"]
            self.assertAlmostEqual(got, self.reference, places=12,
                                   msg=f"a card {k}x the reference's speed scores {got:+.10f} "
                                       f"where the reference scores {self.reference:+.10f}")

    def test_a_different_probed_ceiling_changes_nothing(self):
        """The failure per-box calibration had: a card with a different peak moved the score by
        that difference even when the code is not limited by that peak. The ceiling a run is
        scored against now comes from the anchor and the run, not from a probe."""
        gen = copy.deepcopy(self.gen)
        for c in gen.cells.values():
            if c.calibrated:
                c.ceiling_seconds *= 1.03
        got = CP.compute(gen, self.records, device=OTHER_CARD)["aggregate"]["gap_closed"]
        self.assertAlmostEqual(got, self.reference, places=12)

    def test_a_run_from_another_card_is_not_refused(self):
        CP.compute(self.gen, self.records, device=OTHER_CARD)

    def test_a_base_arm_outside_the_anchor_band_is_refused(self):
        """Further from the anchor than two cards of the class differ: the base code changed, or
        this is not the pinned hardware. Either way the anchored achieved fraction is wrong."""
        k = 1.0 + (CP.BASE_ANCHOR_BAND_PCT + 5.0) / 100.0
        with self.assertRaises(CP.ComputeError) as cm:
            CP.compute(self.gen, _scaled(self.records, k, k))
        self.assertIn("band", str(cm.exception))
        self.assertIn("burnish calibrate --write", str(cm.exception))

    def test_a_noisy_base_arm_is_refused_as_the_boxs_fault(self):
        recs = copy.deepcopy(self.records)
        cell = "vae-decode/1024/bf16"
        floor = self.gen.cell(cell).floor_pct
        bump = 1.0 + (floor * (CP.BASE_SPREAD_FLOORS + 1.0)) / 100.0
        for r in recs:
            if r["cell"] == cell and r["variant"] == "base" and r["repeat"] == 0:
                r["metrics"]["latency_s"] *= bump
        with self.assertRaises(CP.ComputeError) as cm:
            CP.compute(self.gen, recs)
        self.assertIn("too noisy", str(cm.exception))
        self.assertIn("not the submission's fault", str(cm.exception))

    def test_the_anchor_records_where_it_was_measured(self):
        for field in ("calibration_device", "calibration_device_name", "calibration_driver"):
            self.assertTrue(getattr(self.gen, field), f"the anchor does not record {field}")
        for c in self.gen.scorable_cells():
            self.assertTrue(c.measured_seconds, f"{c.id}: the anchor has no base time")


class TestFloorsArePooledWorstFirst(unittest.TestCase):
    """A floor is a spread statistic and is not stable between sessions: two calibrations of one
    RTX 5090, hours apart, moved `t5-encode`'s floor 24x while every achieved fraction held to
    three figures. The anchor's floor is used on every card, so it is the worst seen anywhere --
    too tight would credit noise permanently, too loose only refuses a gain too small to see."""

    def _merge(self, a, b):
        out = copy.deepcopy(b)
        for cid, prior in a["cells"].items():
            now = out["cells"].get(cid)
            if now and prior.get("floor_pct") is not None \
                    and prior["floor_pct"] > now["floor_pct"]:
                now["floor_pct"] = prior["floor_pct"]
                now["floor_decided_by"] = "worst-of-sessions"
        return out

    def test_merging_keeps_the_worst_floor_per_cell_in_either_direction(self):
        a = {"cells": {"x": {"floor_pct": 0.578}, "y": {"floor_pct": 0.259}}}
        b = {"cells": {"x": {"floor_pct": 0.205}, "y": {"floor_pct": 0.845}}}
        m = self._merge(a, b)
        self.assertEqual(m["cells"]["x"]["floor_pct"], 0.578)
        self.assertEqual(m["cells"]["y"]["floor_pct"], 0.845)

    def test_merging_never_tightens_a_floor(self):
        import random
        rng = random.Random(20260912)
        for _ in range(200):
            fa, fb = rng.uniform(0.01, 5.0), rng.uniform(0.01, 5.0)
            m = self._merge({"cells": {"c": {"floor_pct": fa}}},
                            {"cells": {"c": {"floor_pct": fb}}})
            self.assertEqual(m["cells"]["c"]["floor_pct"], max(fa, fb))

    def test_the_calibrator_pools_sessions_from_any_card(self):
        src = (ROOT / "eval" / "calibrate.py").read_text()
        self.assertNotIn("Pooling two cards' floors would describe neither", src)
        self.assertIn("worst-of-sessions", src)
        self.assertIn("calibration_sessions", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
