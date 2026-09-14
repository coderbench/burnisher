#!/usr/bin/env python3
"""The re-registration guard on the runtime's real registry."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from burnscore import reregistration as RR

CUDA = "src/cuda/ops_cuda.cu"
BASE = {CUDA: (ROOT / CUDA).read_text(), "src/cpu/ops_cpu.cpp": (ROOT / "src/cpu/ops_cpu.cpp").read_text()}
ANCHOR = '    register_impl<ScaleArgs>("scale", "cuda", scale_cuda, "one multiply per element");'
EXTRA = """
    const float warm = 0.5f * (float)(n % 7);
    if (warm > 2.0f) { return; }
    float carry = 0.0f;
    for (int64_t z = 0; z < 16; ++z) { carry += 0.125f * (float)z; }
    for (int64_t z = 0; z < 8; ++z) { carry -= 0.25f * (float)(z * z); }
    const float fold = carry > 1.0f ? carry - 1.0f : carry + 1.0f;
    if (fold < 0.0f) { carry = -fold; } else { carry = fold * 0.5f; }
    float spare[4] = {carry, fold, warm, 0.0f};
"""


def candidate(registration="", appended=""):
    text = BASE[CUDA]
    if registration:
        assert text.count(ANCHOR) == 1
        text = text.replace(ANCHOR, ANCHOR + "\n    " + registration)
    return {**BASE, CUDA: text + ("\n" + appended if appended else "")}


def renamed_modulate(modify=False):
    defs = RR.definitions(BASE)
    kernel, wrapper = defs["k_modulate"][0]["text"], defs["modulate_cuda"][0]["text"]
    if modify:
        i = kernel.index("{") + 1
        kernel = kernel[:i] + EXTRA + kernel[i:]
    text = kernel + "\n\n" + wrapper
    for old, new in (("k_modulate", "k_fastmod"), ("modulate_cuda", "fastmod_cuda"), ("grid", "g2")):
        text = re.sub(rf"\b{old}\b", new, text)
    return re.sub(r"[ \t]+", " ", text)


class TestTheRegistryOnMain(unittest.TestCase):
    def test_the_registrations_on_main_are_found(self):
        regs = {(r["op"], r["name"]): r for r in RR.registrations(BASE)}
        self.assertEqual(regs[("attention", "cuda-tile64")]["callable"], "attention_cuda_tiled<64>")
        self.assertEqual(regs[("modulate", "cuda")]["function"], "modulate_cuda")
        self.assertIn(("gemm", "stock"), regs)

    def test_helpers_most_kernels_call_are_infrastructure(self):
        infra = RR.infrastructure(RR.registrations(BASE), RR.definitions(BASE))
        self.assertIn("check_launch", infra)
        self.assertIn("require_device", infra)

    def test_main_against_itself_is_clear(self):
        self.assertEqual(RR.judge(BASE, BASE)["outcome"], "CLEAR")


class TestReregistration(unittest.TestCase):
    def test_the_same_callable_under_a_new_name_is_a_reregistration(self):
        v = RR.judge(candidate('register_impl<AttentionArgs>("attention", "fast", '
                               'attention_cuda_tiled<256>, "x");'), BASE)
        self.assertEqual(v["outcome"], "REREGISTERED")
        self.assertEqual(v["findings"][0]["matches"]["name"], "cuda-tiled")

    def test_the_new_names_are_reported_as_what_the_candidate_arm_can_run(self):
        v = RR.judge(candidate('register_impl<ModulateArgs>("modulate", "fast", fastmod_cuda, "f");',
                               renamed_modulate(modify=True)), BASE)
        self.assertEqual(v["candidate_names"], ["fast"])
        self.assertEqual(RR.judge(BASE, BASE)["candidate_names"], [])

    def test_a_new_tile_width_is_a_variant_not_a_reregistration(self):
        """cuda-tile64 and cuda-tile1024 exist to measure exactly this kind of difference."""
        v = RR.judge(candidate('register_impl<AttentionArgs>("attention", "cuda-tile512", '
                               'attention_cuda_tiled<512>, "x");'), BASE)
        self.assertEqual(v["outcome"], "CLEAR")

    def test_a_renamed_reformatted_copy_of_a_merged_kernel_is_a_reregistration(self):
        v = RR.judge(candidate('register_impl<ModulateArgs>("modulate", "fast", fastmod_cuda, "f");',
                               renamed_modulate()), BASE)
        self.assertEqual(v["outcome"], "REREGISTERED", v)
        f = v["findings"][0]
        self.assertEqual((f["kind"], f["matches"]["op"], f["matches"]["name"]),
                         ("same-kernel", "modulate", "cuda"))

    def test_a_changed_copy_registered_beside_the_old_one_is_clear(self):
        """CONTRIBUTING.md: copy the old kernel, change it, register the new one beside it."""
        v = RR.judge(candidate('register_impl<ModulateArgs>("modulate", "fast", fastmod_cuda, "f");',
                               renamed_modulate(modify=True)), BASE)
        self.assertEqual(v["outcome"], "CLEAR", v)


class TestTheGuardEndToEnd(unittest.TestCase):
    def test_it_reports_a_reregistration_and_honours_a_clearance(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            (repo / "src" / "cuda").mkdir(parents=True)
            (repo / CUDA).write_text(BASE[CUDA])
            g = ["git", "-C", str(repo), "-c", "user.email=t@example.com", "-c", "user.name=t"]
            subprocess.run(g + ["init", "-q", "-b", "main"], check=True)
            subprocess.run(g + ["add", "-A"], check=True)
            subprocess.run(g + ["commit", "-q", "-m", "base"], check=True)
            subprocess.run(g + ["checkout", "-q", "-b", "sub"], check=True)
            (repo / CUDA).write_text(candidate('register_impl<AttentionArgs>("attention", "fast", '
                                               'attention_cuda_tiled<256>, "x");')[CUDA])
            subprocess.run(g + ["commit", "-qam", "sub"], check=True)
            out = Path(tmp) / "v.json"
            for extra, want in (([], "REREGISTERED"), (["--cleared"], "CLEARED")):
                r = subprocess.run([sys.executable, str(ROOT / "scripts" / "reregistration_guard.py"),
                                    "--repo", str(repo), "--base", "main", "--json", str(out), *extra],
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(json.loads(out.read_text())["outcome"], want)


if __name__ == "__main__":
    unittest.main(verbosity=2)
