"""The correctness gate, exercised against a fake device.

The gate is the thing every other number depends on: correctness precedes speed, and a build that
does not reproduce itself cannot be a reference for anything. Until this file existed none of
`eval/gate.py` had ever run, because it refuses to start without a GPU and there was no GPU.

As in `test_device_runners.py`, the fake replaces the DEVICE, not the guards.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FAKES = HERE / "fakes"
sys.path.insert(0, str(HERE.parent))

from burnscore import cells as C
from tests import fixtures
from tests.test_device_runners import fake_env


class TestGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.gen_name = "BG-GATE"
        self.cells_root, gpath = fixtures.scratch_generation(
            self.dir, self.gen_name, with_prompts=True)
        self.gen_dir = gpath.parent
        self.prompts = json.loads((self.gen_dir / "prompts.json").read_text())
        self.tolerance = json.loads(gpath.read_text())["tolerance"]
        # The frozen prompt set's token ids, and the pinned starting noise. Both are INPUTS to a
        # gate run -- the gate compares latents grown from them, so a difference in either is a
        # difference in the comparison rather than in the runtime.
        ids_src = ROOT / "eval" / "cells" / "BG-1"
        for p in self.prompts["prompts"]:
            src = ids_src / f"token-ids-{p['id']}.txt"
            if src.exists():
                (self.gen_dir / f"token-ids-{p['id']}.txt").write_text(src.read_text())
        (self.gen_dir / "token-ids.json").write_text(
            (ids_src / "token-ids.json").read_text())
        self.noise = self.dir / "noise.npy"
        np.save(self.noise, np.zeros((1, 4, 8, 8), dtype=np.float32))

    def tearDown(self):
        self.tmp.cleanup()

    def gate(self, *extra, env=None, out="gate.json"):
        cmd = [sys.executable, str(ROOT / "eval" / "gate.py"),
               "--binary", str(FAKES / "burnisher"),
               "--generation", self.gen_name,
               "--work-dir", str(self.dir / "work"),
               "--cells-root", str(self.cells_root),
               "--weights", str(self.dir / "fake-weights"),
               "--noise", str(self.noise),
               "--device", "cpu",
               "--output", str(self.dir / out), *extra]
        return subprocess.run(cmd, capture_output=True, text=True,
                              env=env or fake_env(), cwd=str(ROOT))

    def make_reference(self, impl="stock", env=None):
        """Produce reference latents the way the real procedure does: from a PINNED build, not
        from the candidate. Making them with the candidate would let it be its own oracle."""
        ref = self.dir / "reference-latents"
        ref.mkdir(exist_ok=True)
        for p in self.prompts["prompts"]:
            subprocess.run([str(FAKES / "burnisher"), "generate",
                            "--token-ids", str(self.gen_dir / f"token-ids-{p['id']}.txt"),
                            "--noise", str(self.noise),
                            "--seed", str(self.prompts["seed"]),
                            "--impl", impl, "--dump-latents", str(ref / f"{p['id']}.npy")],
                           check=True, capture_output=True, env=env or fake_env())
        return ref

    # --- determinism, checked first ---

    def test_a_reproducible_build_passes_determinism(self):
        r = self.gate("--determinism-only", "--repeats", "5")
        self.assertEqual(r.returncode, 0, r.stderr)
        doc = json.loads((self.dir / "gate.json").read_text())
        self.assertTrue(doc["determinism"])
        self.assertEqual(len(doc["determinism_digests"]), 1)
        self.assertEqual(doc["determinism_replays"], 5)

    def test_a_flaky_build_fails_determinism_and_stops_there(self):
        """If two replays of the same build disagree, no candidate can be attributed a
        difference, so the gate must not go on to the reference comparison."""
        r = self.gate("--repeats", "4", env=fake_env(BURNISH_FAKE_FLAKY=1))
        self.assertEqual(r.returncode, 1)
        self.assertIn("NOT DETERMINISTIC", r.stderr)
        doc = json.loads((self.dir / "gate.json").read_text())
        self.assertFalse(doc["determinism"])
        self.assertGreater(len(doc["determinism_digests"]), 1)
        self.assertEqual(doc["correctness"], "NOT_RUN")

    def test_the_failure_message_names_the_usual_causes(self):
        r = self.gate("--repeats", "3", env=fake_env(BURNISH_FAKE_FLAKY=1))
        for cause in ("autotuning", "atomic", "TF32", "launch order"):
            self.assertIn(cause, r.stderr)

    # --- against the pinned reference ---

    def test_a_missing_reference_is_refused_rather_than_inferred(self):
        r = self.gate("--repeats", "2", "--reference", str(self.dir / "nope"))
        self.assertEqual(r.returncode, 2)
        self.assertIn("no pinned reference latents", r.stderr)
        self.assertEqual(json.loads((self.dir / "gate.json").read_text())["correctness"],
                         "NO_REFERENCE")

    def test_an_identical_build_passes_the_tolerance(self):
        ref = self.make_reference()
        r = self.gate("--repeats", "2", "--reference", str(ref), "--impl", "stock")
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        doc = json.loads((self.dir / "gate.json").read_text())
        self.assertEqual(doc["correctness"], "PASS")
        self.assertEqual(doc["worst_relative_l2"], 0.0)
        self.assertEqual(len(doc["per_prompt"]), len(self.prompts["prompts"]))

    def test_a_small_drift_passes_and_is_reported_as_a_number(self):
        """Inside the tolerance is a PASS, and the distance is still recorded -- it is a scored
        frontier objective, so 'inside the gate but worse' must not be invisible."""
        ref = self.make_reference()
        r = self.gate("--repeats", "2", "--reference", str(ref), "--impl", "candidate",
                      env=fake_env(BURNISH_FAKE_DRIFT=0.005))
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        doc = json.loads((self.dir / "gate.json").read_text())
        self.assertEqual(doc["correctness"], "PASS")
        self.assertGreater(doc["worst_relative_l2"], 0.0)
        self.assertLessEqual(doc["worst_relative_l2"], self.tolerance["latent_l2_relative"])

    def test_a_large_drift_is_a_rejection(self):
        ref = self.make_reference()
        r = self.gate("--repeats", "2", "--reference", str(ref), "--impl", "candidate",
                      env=fake_env(BURNISH_FAKE_DRIFT=0.5))
        self.assertEqual(r.returncode, 1)
        self.assertIn("This is a rejection, not a trade-off", r.stderr)
        doc = json.loads((self.dir / "gate.json").read_text())
        self.assertEqual(doc["correctness"], "FAIL")
        # EITHER threshold rejects, and they catch different things: relative L2 is a whole-tensor
        # norm, max-abs is a single element. A change that moves one value a long way and the
        # norm barely at all is still a different model, and at this drift it is max-abs that
        # trips first -- which is the case a norm-only gate would have let through.
        self.assertTrue(
            doc["worst_relative_l2"] > self.tolerance["latent_l2_relative"]
            or doc["worst_max_abs"] > self.tolerance["latent_max_abs"],
            f"neither threshold was exceeded: L2 {doc['worst_relative_l2']}, "
            f"max-abs {doc['worst_max_abs']}")

    def test_the_two_thresholds_catch_different_things(self):
        """max-abs is not redundant with relative L2.

        One is a whole-tensor norm and the other is a single element, so a change that moves one
        value a long way and the norm barely at all is caught only by max-abs. The drift that
        demonstrates it is SEARCHED FOR rather than hardcoded, because the thresholds are now set
        from a measurement and a test that assumed particular values would break every time the
        gate was recalibrated -- which it should be, whenever the hardware or the reference moves.
        """
        ref = self.make_reference()
        straddled = None
        for drift in (0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2, 2.0):
            self.gate("--repeats", "2", "--reference", str(ref), "--impl", "candidate",
                      env=fake_env(BURNISH_FAKE_DRIFT=drift), out="probe.json")
            d = json.loads((self.dir / "probe.json").read_text())
            l2_ok = d["worst_relative_l2"] <= self.tolerance["latent_l2_relative"]
            abs_ok = d["worst_max_abs"] <= self.tolerance["latent_max_abs"]
            if l2_ok != abs_ok:
                straddled = (drift, d, l2_ok, abs_ok)
                break
        self.assertIsNotNone(
            straddled,
            "no drift made the two thresholds disagree, so one of them is doing no work that "
            "the other does not already do. Either they are badly calibrated against each other "
            "or the pair is redundant -- both are worth knowing.")
        drift, d, l2_ok, abs_ok = straddled
        self.assertEqual(d["correctness"], "FAIL")
        which = "max-abs" if l2_ok else "relative L2"
        self.assertIn(which, ("max-abs", "relative L2"))

    def test_a_very_large_drift_trips_the_norm_gate_too(self):
        ref = self.make_reference()
        r = self.gate("--repeats", "2", "--reference", str(ref), "--impl", "candidate",
                      env=fake_env(BURNISH_FAKE_DRIFT=5.0))
        self.assertEqual(r.returncode, 1)
        doc = json.loads((self.dir / "gate.json").read_text())
        self.assertGreater(doc["worst_relative_l2"], self.tolerance["latent_l2_relative"])

    def test_calibrate_tolerance_measures_instead_of_asserting(self):
        ref = self.make_reference()
        r = self.gate("--repeats", "2", "--reference", str(ref), "--impl", "candidate",
                      "--calibrate-tolerance", env=fake_env(BURNISH_FAKE_DRIFT=0.003))
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        doc = json.loads((self.dir / "gate.json").read_text())
        self.assertEqual(doc["correctness"], "NOT_RUN")
        self.assertIn("measured_drift", doc)
        self.assertGreater(doc["measured_drift"]["relative_l2"], 0.0)

    def test_a_prompt_set_the_reference_does_not_cover_is_refused(self):
        """A prompt set and a reference that disagree is a gate that checks nothing."""
        ref = self.make_reference()
        (ref / f"{self.prompts['prompts'][1]['id']}.npy").unlink()
        r = self.gate("--repeats", "2", "--reference", str(ref))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("reference directory has no", r.stdout + r.stderr)

    def test_the_noise_digest_is_recorded(self):
        """The starting noise is an input, and a gate report that did not say WHICH noise could
        not be checked against the reference latents' own manifest."""
        self.gate("--determinism-only", "--repeats", "2")
        doc = json.loads((self.dir / "gate.json").read_text())
        self.assertEqual(len(doc["noise_sha256"]), 64)
        self.assertEqual(doc["dtype"], "bf16")

    def test_the_prompt_set_digest_is_recorded(self):
        """The prompt set is part of the oracle; changing it would make every comparison against
        the existing reference meaningless, and would not look like anything in a diff."""
        r = self.gate("--determinism-only", "--repeats", "2")
        doc = json.loads((self.dir / "gate.json").read_text())
        self.assertEqual(len(doc["prompt_set_digest"]), 64)
        self.assertEqual(doc["seed"], self.prompts["seed"])

    def test_a_busy_device_stops_the_gate(self):
        r = self.gate("--determinism-only", "--repeats", "2", env=fake_env(BURNISH_FAKE_BUSY=1))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not idle", r.stdout + r.stderr)

    def test_a_shape_mismatch_is_not_a_tolerance_question(self):
        ref = self.make_reference()
        np.save(ref / f"{self.prompts['prompts'][0]['id']}.npy",
                np.zeros((1, 4, 8, 8), dtype=np.float32))
        r = self.gate("--repeats", "2", "--reference", str(ref))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not a tolerance question", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
