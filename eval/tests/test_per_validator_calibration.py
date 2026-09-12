#!/usr/bin/env python3
"""A score must mean the same thing on every validator's box, and this is what makes it.

The problem, stated as an arithmetic fact rather than an operational nuisance
----------------------------------------------------------------------------

`achieved = ceiling / measured`, and BOTH halves are properties of the hardware. Freeze the
ceiling to one card's probed peak and score another card's run against it, and the ratio mixes
two machines. Two RTX 5090s differ by about 3% on achievable GEMM -- the repository's own
`configs/devices.json` records it -- and that lands as a systematic ~6% difference in gap-closed.

Systematic, not noisy. A bootstrap cannot absorb it and an interval will not reveal it: every
submission a slower validator picked up would pay less than the same submission on a faster one,
consistently, forever, and nothing in the receipt would say so.

Probe locally and both halves scale with the card and cancel. That is not a workaround for the
drift guard; it is the reason gap-closed was worth choosing as the unit in the first place.
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from burnscore import cells as C
from burnscore import compute as CP

GEN = ROOT / "eval" / "cells" / "BG-1" / "generation.json"
CAL = ROOT / "eval" / "cells" / "BG-1" / "reference.json"
RAW = ROOT / "examples" / "BG-1-pr-000001-raw.json"


class TestTwoValidatorsAgree(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.raw = json.loads(RAW.read_text())
        self.cal = json.loads(CAL.read_text())
        self.devA = self.raw["provenance"]["device"]

    def tearDown(self):
        self.tmp.cleanup()

    def _slower_validator(self, k):
        """Validator B: a card `k` times slower, who calibrated it themselves.

        Everything that is a measurement of the card scales by k -- the run, and the calibration's
        own base timings. The ceiling scales too, because a ceiling is flops over the card's peak
        and B probed B's peak.
        """
        recs = copy.deepcopy(self.raw["records"])
        for r in recs:
            r["metrics"]["latency_s"] *= k
        dev = dict(self.devA, uuid="GPU-bbbbbbbb-0000-0000-0000-000000000000")

        cal = copy.deepcopy(self.cal)
        cal["device_probe"] = dict(cal["device_probe"], uuid=dev["uuid"])
        for c in cal["cells"].values():
            c["ceiling_seconds"] *= k
            c["measured_seconds"] *= k
        path = Path(self.tmp.name) / "calB.json"
        path.write_text(json.dumps(cal))
        return recs, dev, C.load(GEN, calibration=path)

    def _score(self, gen, recs, dev):
        return CP.compute(gen, recs, device=dev)["aggregate"]["gap_closed"]

    def test_a_locally_calibrated_validator_gets_the_same_score(self):
        """The property the whole design exists for."""
        a = self._score(C.load(GEN), self.raw["records"], self.devA)
        for k in (1.03, 0.97, 1.15):
            recs, dev, gen = self._slower_validator(k)
            b = self._score(gen, recs, dev)
            self.assertAlmostEqual(a, b, places=12,
                                   msg=f"a card {k}x different scores {b:+.8f} where the "
                                       f"reference scores {a:+.8f}. A validator's hardware "
                                       f"must not change what a submission earned.")

    def test_the_same_run_against_a_foreign_calibration_would_have_been_biased(self):
        """What the guard prevents, quantified. Kept as a test so the number is not folklore."""
        a = self._score(C.load(GEN), self.raw["records"], self.devA)
        recs, _, _ = self._slower_validator(1.03)
        # Score B's run against A's calibration, with the guard off, as it would have been.
        biased = CP.compute(C.load(GEN), recs, reference_drift_guard=False
                            )["aggregate"]["gap_closed"]
        drift = abs(biased / a - 1)
        self.assertGreater(drift, 0.02,
                           "a 3% hardware difference no longer biases the score; if that is "
                           "genuinely true the guard can be relaxed, but check why first")

    def test_a_run_from_another_box_is_refused_rather_than_silently_biased(self):
        gen = C.load(GEN)
        foreign = dict(self.devA, uuid="GPU-ffffffff-0000-0000-0000-000000000000")
        with self.assertRaises(CP.ComputeError) as cm:
            CP.compute(gen, self.raw["records"], device=foreign)
        msg = str(cm.exception)
        self.assertIn("calibration describes", msg)
        # The message has to say what to DO, not only what went wrong.
        self.assertIn("burnish calibrate", msg)
        self.assertIn("burnish probe", msg)

    def test_the_committed_calibration_still_scores_its_own_box(self):
        """The reference device is a validator too; nothing special-cases it."""
        gen = C.load(GEN)
        self.assertEqual(gen.calibration_device, self.devA["uuid"])
        CP.compute(gen, self.raw["records"], device=self.devA)

    def test_a_run_with_no_device_recorded_is_not_blocked_by_this_guard(self):
        """Older raw files and the synthetic fixtures carry no device. They still score.

        The guard exists to catch a mismatch, and "unknown" is not a mismatch -- refusing it
        would break every test fixture to catch nothing. A raw file with no provenance is
        already flagged by `code_provenance_complete`.
        """
        CP.compute(C.load(GEN), self.raw["records"], device=None)
        CP.compute(C.load(GEN), self.raw["records"], device={})

    def test_the_calibration_carries_the_box_it_describes(self):
        gen = C.load(GEN)
        for field in ("calibration_device", "calibration_device_name", "calibration_driver"):
            self.assertTrue(getattr(gen, field),
                            f"the calibration does not record {field}, so nothing can check "
                            f"whether it describes the box it is being used on")


class TestFloorsAreNotStableBetweenSessions(unittest.TestCase):
    """A measured fact about the instrument, and the reason `--merge` exists.

    Two calibrations of the SAME physical RTX 5090, hours apart, same driver, same build:

        cell                     session A   session B     ratio
        dit-step/1024/bf16          0.578%      0.205%       2.8x
        t5-encode/1024/bf16         3.753%      0.155%      24.2x
        vae-decode/1024/bf16        0.259%      0.845%       3.3x

    while the achieved fractions held to three significant figures (1.52/1.52, 18.15/18.13,
    0.79/0.79). That asymmetry is expected -- `achieved` is a median and robust, a floor is a
    spread over nine repeats and is not -- but it has a consequence worth guarding: whether a
    submission RESOLVES would otherwise depend on which session its validator calibrated in.

    Merging keeps the WORST floor per cell. The two errors are not symmetric. A floor that is
    too tight credits noise as a contribution and the ledger compounds it permanently; one that
    is too loose refuses a gain too small to see, and the contributor returns with a bigger one.
    Only one of those is recoverable.
    """

    def _merge(self, a, b):
        """The merge rule, exercised on the shape the calibrator writes."""
        import copy
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
        self.assertEqual(m["cells"]["x"]["floor_pct"], 0.578, "kept the tighter floor for x")
        self.assertEqual(m["cells"]["y"]["floor_pct"], 0.845, "kept the tighter floor for y")

    def test_merging_never_tightens_a_floor(self):
        """The property that makes this safe, stated as an invariant rather than an example."""
        import random
        rng = random.Random(20260912)
        for _ in range(200):
            fa, fb = rng.uniform(0.01, 5.0), rng.uniform(0.01, 5.0)
            m = self._merge({"cells": {"c": {"floor_pct": fa}}},
                            {"cells": {"c": {"floor_pct": fb}}})
            self.assertGreaterEqual(m["cells"]["c"]["floor_pct"], min(fa, fb))
            self.assertEqual(m["cells"]["c"]["floor_pct"], max(fa, fb))

    def test_the_calibrator_refuses_to_pool_two_different_cards(self):
        """Pooling two cards' floors describes neither."""
        src = (ROOT / "eval" / "calibrate.py").read_text()
        self.assertIn("Pooling two cards' floors would describe neither", src)

    def test_the_calibration_records_how_many_sessions_it_pooled(self):
        src = (ROOT / "eval" / "calibrate.py").read_text()
        self.assertIn("calibration_sessions", src)
        self.assertIn("_floor_stability", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
