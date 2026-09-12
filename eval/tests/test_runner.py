"""The device-side guards, tested without a device.

Every one of these encodes an incident. They are tested here because nothing about a
contaminated control, a silent fallback or a degenerate output looks wrong in the output -- that
is precisely what makes them dangerous, and it is why asserting them is the only way they stay
correct.
"""
from __future__ import annotations

import os
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import runner as RN


class TestEnvironmentScrub(unittest.TestCase):
    def test_every_runtime_variable_is_removed(self):
        """INCIDENT: an operator with a tuning variable exported in their shell runs a control
        that is not the control, and the harness reports ~0% for a comparison of the candidate
        against itself."""
        base = {"PATH": "/bin", "BURNISHER_IMPL": "fused", "BURNISH_RT_TILE": "128",
                "HOME": "/root"}
        env = RN.scrubbed_environment(base)
        self.assertNotIn("BURNISHER_IMPL", env)
        self.assertNotIn("BURNISH_RT_TILE", env)
        self.assertEqual(env["PATH"], "/bin")
        self.assertEqual(env["HOME"], "/root")

    def test_autotuning_is_pinned_for_both_arms(self):
        """An autotuner that picks a different algorithm per process makes a build
        non-deterministic in a way that looks exactly like a policy effect."""
        self.assertEqual(RN.scrubbed_environment({})["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")


class TestResultParsing(unittest.TestCase):
    def test_a_missing_result_line_is_an_error(self):
        with self.assertRaises(RN.RunnerError) as cm:
            RN.parse_result("some prose about how fast it was\nlatency: 12ms\n", "arm")
        self.assertIn("no BURNISH_JSON", str(cm.exception))

    def test_two_result_lines_mean_the_pairing_is_gone(self):
        text = 'BURNISH_JSON: {"a":1}\nBURNISH_JSON: {"a":2}\n'
        with self.assertRaises(RN.RunnerError) as cm:
            RN.parse_result(text, "arm")
        self.assertIn("ran twice", str(cm.exception))

    def test_a_single_line_parses(self):
        self.assertEqual(RN.parse_result('noise\nBURNISH_JSON: {"a":1}\nmore\n', "arm"),
                         {"a": 1})


class TestRanWhatItClaimed(unittest.TestCase):
    """INCIDENT CLASS: an arm that silently fell back. A requested implementation that is not
    registered, a tile clamped to fit, a dtype demoted because the kernel was missing -- each
    produces a perfectly good number for a configuration nobody asked for."""

    def test_a_fallback_is_refused(self):
        result = {"effective": {"impl": "stock", "dtype": "bf16", "stage": "dit-step"}}
        with self.assertRaises(RN.RunnerError) as cm:
            RN.require_ran_what_it_claimed(result, {"impl": "fused-adaln"}, "arm")
        self.assertIn("asked 'fused-adaln', ran 'stock'", str(cm.exception))

    def test_a_runtime_that_stopped_reporting_is_refused(self):
        with self.assertRaises(RN.RunnerError) as cm:
            RN.require_ran_what_it_claimed({"effective": {}}, {"impl": "x"}, "arm")
        self.assertIn("did not report", str(cm.exception))

    def test_a_matching_report_passes(self):
        RN.require_ran_what_it_claimed(
            {"effective": {"impl": "fused", "dtype": "bf16"}},
            {"impl": "fused", "dtype": "bf16"}, "arm")


class TestDegenerateOutput(unittest.TestCase):
    """A generation that produced a black image or a NaN latent runs fast and means nothing."""

    def test_a_constant_output_is_refused(self):
        with self.assertRaises(RN.RunnerError) as cm:
            RN.require_not_degenerate(
                {"output_stats": {"latent_mean": 0.0, "latent_std": 0.0,
                                  "latent_absmax": 0.0}}, "arm")
        self.assertIn("generated nothing", str(cm.exception))

    def test_a_diverged_output_is_refused(self):
        with self.assertRaises(RN.RunnerError):
            RN.require_not_degenerate(
                {"output_stats": {"latent_mean": 0.0, "latent_std": 1.0,
                                  "latent_absmax": float("inf")}}, "arm")

    def test_missing_statistics_are_refused(self):
        with self.assertRaises(RN.RunnerError) as cm:
            RN.require_not_degenerate({"output_stats": {"latent_mean": 0.0}}, "arm")
        self.assertIn("indistinguishable from a fast one", str(cm.exception))

    def test_a_healthy_output_passes(self):
        RN.require_not_degenerate(
            {"output_stats": {"latent_mean": -0.02, "latent_std": 1.1,
                              "latent_absmax": 4.3}}, "arm")


class TestInterleaving(unittest.TestCase):
    def test_arms_alternate_rather_than_blocking(self):
        """Running one arm to completion and then the other puts the thermal ramp between the
        arms and attributes it to whichever went second."""
        order = list(RN.interleave(("base", "candidate"), 3))
        self.assertEqual(order, [(0, "base"), (0, "candidate"),
                                 (1, "base"), (1, "candidate"),
                                 (2, "base"), (2, "candidate")])


class TestCliRefusesToEstimate(unittest.TestCase):
    def test_every_gpu_subcommand_fails_without_a_device(self):
        """A measurement command with no device must not degrade into an estimate."""
        import subprocess
        burnish = Path(__file__).resolve().parent.parent.parent / "tools" / "burnish"
        for cmd in ("probe", "calibrate", "gate", "bench"):
            p = subprocess.run([sys.executable, str(burnish), cmd],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 3, f"{cmd} did not refuse cleanly")
            self.assertIn("needs a GPU", p.stderr)
            self.assertIn("There is no fallback", p.stderr)

    def test_every_entry_point_answers_help(self):
        """The cheapest possible check that a binary is not simply broken.

        DISCOVERED, not listed. A hand-maintained list is how this test came to cover six entry
        points while the repository had ten -- and the four it missed included the ones most
        recently written, which are the ones most likely to be broken.

        It earns its keep: `calibrate.py` once referenced `RunnerError` without importing it, in
        a handler that only runs when there is no device. Every no-device failure would have
        been a NameError. Nothing else in the suite would have noticed, because nothing else
        runs that file on a machine without a GPU.
        """
        import subprocess
        root = Path(__file__).resolve().parent.parent.parent
        entries = [root / "tools" / "burnish"]
        for py in sorted((root / "eval").glob("*.py")) + sorted((root / "scripts").glob("*.py")):
            src = py.read_text()
            if 'if __name__ == "__main__"' in src and "argparse" in src:
                entries.append(py)
        self.assertGreaterEqual(len(entries), 10,
                                f"only found {len(entries)} entry points; the discovery is "
                                f"probably broken rather than the repository shrinking")
        for e in entries:
            p = subprocess.run([sys.executable, str(e), "--help"],
                               capture_output=True, text=True, timeout=120)
            self.assertEqual(p.returncode, 0,
                             f"{e.name} --help exited {p.returncode}\n{p.stderr[-600:]}")
            self.assertTrue(p.stdout.strip(), f"{e.name} --help printed nothing")

    def test_every_burnish_subcommand_answers_help(self):
        """Same discipline one level down: the subcommands are read off the parser itself."""
        import subprocess
        root = Path(__file__).resolve().parent.parent.parent
        burnish = root / "tools" / "burnish"
        out = subprocess.run([sys.executable, str(burnish), "--help"],
                             capture_output=True, text=True, timeout=120).stdout
        m = re.search(r"\{([a-z,\-]+)\}", out)
        self.assertIsNotNone(m, "cannot read the subcommand list off `burnish --help`")
        subs = m.group(1).split(",")
        self.assertGreaterEqual(len(subs), 10)
        for cmd in subs:
            p = subprocess.run([sys.executable, str(burnish), cmd, "--help"],
                               capture_output=True, text=True, timeout=120)
            self.assertEqual(p.returncode, 0,
                             f"burnish {cmd} --help exited {p.returncode}\n{p.stderr[-400:]}")
