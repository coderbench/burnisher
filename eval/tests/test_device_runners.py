"""bench.py and calibrate.py, exercised end to end against a fake device.

These two files carry most of the guards and, until this test existed, none of their code had
ever run: they refuse to start without a GPU, and there was no GPU. A harness whose measurement
path is untested is exactly the thing this repository warns contributors about.

The fakes replace the DEVICE, not the guards. `require_idle_device` still shells out, still
parses, and still refuses when the fake reports a busy device; `require_ran_what_it_claimed` still
compares the runtime's report against the request. What is removed is the silicon.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FAKES = HERE / "fakes"
sys.path.insert(0, str(HERE.parent))

from burnscore import cells as C
from tests import fixtures


def fake_env(counter=None, **over):
    env = dict(os.environ)
    env["PATH"] = f"{FAKES}:{env['PATH']}"
    env["BURNISH_EVAL_FAST"] = "1"          # skip the settle sleeps
    env["BURNISH_EVAL_LOCK"] = str(Path(tempfile.gettempdir()) / "burnisher-test.lock")
    if counter:
        env["BURNISH_FAKE_COUNTER"] = str(counter)
    env.update({k: str(v) for k, v in over.items()})
    return env


class DeviceRunnerCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.gen_name = "BG-FAKE"
        # Outside the repository: a test must not leave a calibrated generation in eval/cells/,
        # where the next `burnish generation show` would report numbers nobody measured.
        self.cells_root, self.gpath = fixtures.scratch_generation(self.dir, self.gen_name)
        self.gen_dir = self.gpath.parent

    def tearDown(self):
        self.tmp.cleanup()

    def gate_file(self, impl="fused-adaln", correctness="PASS", determinism=True,
                  l2=0.004, name="gate.json"):
        p = self.dir / name
        p.write_text(json.dumps({
            "impl": impl, "correctness": correctness, "determinism": determinism,
            "worst_relative_l2": l2, "worst_max_abs": 0.01,
            "base_commit": "0" * 40, "candidate_commit": "1" * 40}))
        return p

    def base_gate(self, **kw):
        kw.setdefault("impl", "stock")
        kw.setdefault("name", "gate-base.json")
        return self.gate_file(**kw)

    def bench(self, *extra, env=None, gate=None, base_gate=None):
        gate = gate or self.gate_file()
        base_gate = base_gate or self.base_gate()
        cmd = [sys.executable, str(ROOT / "eval" / "bench.py"),
               "--binary", str(FAKES / "burnisher"),
               "--generation", self.gen_name,
               "--impl-candidate", "fused-adaln",
               "--repeats", "3",
               "--gate-result", str(gate),
               "--gate-base-result", str(base_gate),
               "--cells-root", str(self.cells_root),
               "--output", str(self.dir / "raw.json"), *extra]
        return subprocess.run(cmd, capture_output=True, text=True,
                              env=env or fake_env(), cwd=str(ROOT))


class TestBench(DeviceRunnerCase):
    def test_a_paired_run_produces_scorable_records(self):
        r = self.bench("--skip-held-out", env=fake_env(BURNISH_FAKE_SPEEDUP=1.15))
        self.assertEqual(r.returncode, 0, r.stderr)
        doc = json.loads((self.dir / "raw.json").read_text())
        self.assertEqual(len(doc["records"]), 3 * 2 * 3)      # cells x arms x repeats
        for rec in doc["records"]:
            self.assertEqual(rec["status"], "OK")
            self.assertIn("latency_s", rec["metrics"])
        # And the harness can actually score what the runner wrote -- the join these two files
        # meet at, which no module test covers.
        from burnscore import compute as CP
        gen = C.load(self.gen_dir / "generation.json")
        out = CP.compute(gen, doc["records"])
        self.assertGreater(out["per_cell"]["dit-step/1024/bf16"]["gap_closed"], 0)
        self.assertTrue(out["per_cell"]["dit-step/1024/bf16"]["resolved"])

    def test_held_out_shapes_are_run_and_recorded(self):
        r = self.bench("--held-out-seed", "7", env=fake_env(BURNISH_FAKE_SPEEDUP=1.15))
        self.assertEqual(r.returncode, 0, r.stderr)
        doc = json.loads((self.dir / "raw.json").read_text())
        self.assertTrue(doc["held_out"])
        self.assertIn(doc["held_out_shape"], (768, 1152))
        self.assertIn("held-out shape for this run", r.stdout)

    def test_a_build_that_failed_the_gate_is_not_timed(self):
        gate = self.gate_file(correctness="FAIL")
        r = self.bench("--skip-held-out", gate=gate)
        self.assertEqual(r.returncode, 2)
        self.assertIn("gate did not pass", r.stderr)

    def test_a_nondeterministic_build_is_not_timed(self):
        gate = self.gate_file(determinism=False)
        r = self.bench("--skip-held-out", gate=gate)
        self.assertEqual(r.returncode, 2)

    def test_a_gate_for_a_different_impl_is_not_a_gate_result(self):
        gate = self.gate_file(impl="some-other-kernel")
        r = self.bench("--skip-held-out", gate=gate)
        self.assertEqual(r.returncode, 2)
        self.assertIn("different implementation", r.stderr)

    def test_a_busy_device_stops_the_run(self):
        r = self.bench("--skip-held-out", env=fake_env(BURNISH_FAKE_BUSY=1))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not idle", r.stdout + r.stderr)
        self.assertIn("pkill -f", r.stdout + r.stderr)

    def test_an_arm_that_silently_fell_back_is_refused(self):
        r = self.bench("--skip-held-out", env=fake_env(BURNISH_FAKE_FALLBACK=1))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("did not run what was asked", r.stdout + r.stderr)

    def test_a_degenerate_output_is_refused(self):
        r = self.bench("--skip-held-out", env=fake_env(BURNISH_FAKE_DEGENERATE=1))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("generated nothing", r.stdout + r.stderr)

    def test_a_runtime_that_stopped_printing_a_result_is_refused(self):
        r = self.bench("--skip-held-out", env=fake_env(BURNISH_FAKE_NO_JSON=1))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no BURNISH_JSON", r.stdout + r.stderr)

    def test_every_declared_objective_reaches_the_records(self):
        """A record missing a declared objective produces no operating point, and the frontier
        then comes out as exactly zero for both arms -- which reads as a quality-neutral result
        rather than as a missing measurement."""
        r = self.bench("--skip-held-out", env=fake_env(BURNISH_FAKE_SPEEDUP=1.15))
        self.assertEqual(r.returncode, 0, r.stderr)
        doc = json.loads((self.dir / "raw.json").read_text())
        gen = C.load(self.gen_dir / "generation.json")
        for rec in doc["records"]:
            for o in gen.objectives:
                self.assertIn(o.key, rec["metrics"],
                              f"the runner produced no {o.key}, which the generation declares")

    def test_a_base_gate_is_mandatory_because_fidelity_is_scored(self):
        gate = self.gate_file()
        cmd = [sys.executable, str(ROOT / "eval" / "bench.py"),
               "--binary", str(FAKES / "burnisher"), "--generation", self.gen_name,
               "--impl-candidate", "fused-adaln", "--gate-result", str(gate),
               "--cells-root", str(self.cells_root),
               "--output", str(self.dir / "raw.json")]
        r = subprocess.run(cmd, capture_output=True, text=True, env=fake_env(), cwd=str(ROOT))
        self.assertEqual(r.returncode, 2)
        self.assertIn("scored frontier objective", r.stderr)

    def test_a_determinism_only_gate_cannot_supply_fidelity(self):
        gate = self.gate_file(l2=None)
        r = self.bench("--skip-held-out", gate=gate)
        self.assertEqual(r.returncode, 2)
        self.assertIn("worst_relative_l2", r.stderr)

    def test_a_gate_result_is_mandatory(self):
        cmd = [sys.executable, str(ROOT / "eval" / "bench.py"),
               "--binary", str(FAKES / "burnisher"), "--generation", self.gen_name,
               "--impl-candidate", "x", "--gate-base-result", str(self.base_gate()),
               "--cells-root", str(self.cells_root),
               "--output", str(self.dir / "raw.json")]
        r = subprocess.run(cmd, capture_output=True, text=True, env=fake_env(), cwd=str(ROOT))
        self.assertEqual(r.returncode, 2)
        self.assertIn("correctness precedes speed", r.stderr)


class TestCalibrate(DeviceRunnerCase):
    def calibrate(self, *extra, env=None):
        cmd = [sys.executable, str(ROOT / "eval" / "calibrate.py"),
               "--binary", str(FAKES / "burnisher"),
               "--generation", self.gen_name, "--repeats", "5",
               "--cells-root", str(self.cells_root),
               "--output", str(self.dir / "ref.json"), *extra]
        return subprocess.run(cmd, capture_output=True, text=True,
                              env=env or fake_env(), cwd=str(ROOT))

    def test_calibration_produces_an_achieved_fraction_and_a_floor_per_cell(self):
        r = self.calibrate(env=fake_env(counter=self.dir / "c", BURNISH_FAKE_WOBBLE=0.004))
        self.assertEqual(r.returncode, 0, r.stderr)
        doc = json.loads((self.dir / "ref.json").read_text())
        self.assertEqual(sorted(doc["cells"]),
                         ["dit-step/1024/bf16", "t5-encode/1024/bf16", "vae-decode/1024/bf16"])
        for cid, cell in doc["cells"].items():
            self.assertEqual(cell["basis"], "measured")
            self.assertGreater(cell["achieved"], 0.0)
            self.assertLessEqual(cell["achieved"], 1.0)
            self.assertGreater(cell["floor_pct"], 0.0, cid)
            self.assertEqual(cell["floor_repeats"], 5)
            self.assertIn(cell["floor_decided_by"], ("spread", "instrument_resolution"))

    def test_the_floor_tracks_the_noise_it_is_measuring(self):
        """The whole premise: a noisier box publishes a bigger floor, so a constant threshold is
        the wrong instrument. A twenty-fold difference in wobble must show up in the floor."""
        quiet = self.calibrate(env=fake_env(counter=self.dir / "c1",
                                            BURNISH_FAKE_WOBBLE=0.001))
        self.assertEqual(quiet.returncode, 0, quiet.stderr)
        q = json.loads((self.dir / "ref.json").read_text())
        noisy = self.calibrate(env=fake_env(counter=self.dir / "c2",
                                            BURNISH_FAKE_WOBBLE=0.02))
        self.assertEqual(noisy.returncode, 0, noisy.stderr)
        n = json.loads((self.dir / "ref.json").read_text())
        for cid in q["cells"]:
            self.assertGreater(n["cells"][cid]["floor_pct"],
                               q["cells"][cid]["floor_pct"] * 3, cid)
            self.assertEqual(n["cells"][cid]["floor_decided_by"], "spread", cid)

    def test_a_calibration_that_beats_the_ceiling_is_refused_per_cell(self):
        """A measured time faster than the arithmetic ceiling is a bug in the geometry, the
        device peak, or the run. It must not be written as a calibration."""
        r = self.calibrate(env=fake_env(BURNISH_FAKE_SPEEDUP=1, BURNISH_FAKE_WOBBLE=0.001))
        self.assertEqual(r.returncode, 0, r.stderr)
        doc = json.loads((self.dir / "ref.json").read_text())
        for cell in doc["cells"].values():
            self.assertLessEqual(cell["achieved"], 1.0)

    def test_a_busy_device_stops_calibration_too(self):
        r = self.calibrate(env=fake_env(BURNISH_FAKE_BUSY=1))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not idle", r.stdout + r.stderr)

    def test_calibration_then_scoring_is_a_closed_loop(self):
        """Calibrate against the fake device, write the reference, then score a real paired run
        against it. This is the whole instrument, joined up, with only the silicon replaced."""
        r = self.calibrate("--write", env=fake_env(counter=self.dir / "c",
                                                   BURNISH_FAKE_WOBBLE=0.002))
        self.assertEqual(r.returncode, 0, r.stderr)
        ref = json.loads((self.gen_dir / "reference.json").read_text())
        self.assertTrue(all(v["achieved"] for v in ref["cells"].values()))

        b = self.bench("--skip-held-out", env=fake_env(counter=self.dir / "c2",
                                                       BURNISH_FAKE_SPEEDUP=1.15,
                                                       BURNISH_FAKE_WOBBLE=0.002))
        self.assertEqual(b.returncode, 0, b.stderr)
        raw = self.dir / "raw.json"
        doc = json.loads(raw.read_text())
        doc["generation"] = self.gen_name
        raw.write_text(json.dumps(doc))

        out = self.dir / "receipt.json"
        s = subprocess.run([sys.executable, str(ROOT / "tools" / "burnish"), "score", str(raw),
                            "--generation", self.gen_name,
                            "--cells-root", str(self.cells_root), "--output", str(out)],
                           capture_output=True, text=True, cwd=str(ROOT), env=fake_env())
        self.assertEqual(s.returncode, 0, s.stderr)
        rec = json.loads(out.read_text())
        self.assertEqual(rec["status"], "FRONTIER_EXPANDED")
        self.assertGreater(rec["score"]["credited_gap_closed"], 0)
        self.assertTrue(rec["score"]["resolved"])
        cell = rec["per_cell"]["dit-step/1024/bf16"]
        self.assertAlmostEqual(cell["speedup"], 1.15, places=2)
        self.assertEqual(cell["ceiling_basis"], "model")
        self.assertEqual(rec["score"]["basis"], "measured")


if __name__ == "__main__":
    unittest.main(verbosity=2)
