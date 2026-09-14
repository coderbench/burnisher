#!/usr/bin/env python3
"""The reference implementation must run with TF32 off wherever it produces or checks an oracle.

Torch leaves cuDNN's TF32 on by default. The first BG-1 and BG-2 reference latents were made that
way, with nothing in the scripts saying so. An oracle whose convolutions may round differently on
another torch build or card is not pinned, so both scripts turn it off, and this keeps it off.

Turning it off changed nothing already committed: on 2026-09-14 all sixteen BG-1 and BG-2 reference
latents, fp32 and bf16, regenerated with TF32 off were byte-identical to the committed ones.
"""
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


class TestTheOracleRunsWithoutTf32(unittest.TestCase):
    def test_both_reference_scripts_turn_tf32_off(self):
        for name in ("make_reference_latents.py", "differential_test.py"):
            src = (ROOT / "scripts" / name).read_text()
            self.assertIn("torch.backends.cuda.matmul.allow_tf32 = False", src, name)
            self.assertIn("torch.backends.cudnn.allow_tf32 = False", src, name)

    def test_new_reference_manifests_record_it(self):
        src = (ROOT / "scripts" / "make_reference_latents.py").read_text()
        self.assertIn('"tf32": {"matmul": False, "cudnn": False}', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
