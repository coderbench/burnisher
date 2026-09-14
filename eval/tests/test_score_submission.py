#!/usr/bin/env python3
"""`eval/score_submission.sh` must score the generation it is given, at every stage.

It once passed `--generation` to the scoring stage only. The gates and the bench fell back to
their default, BG-1, so a BG-2 submission ran a 1024px gate against 512px noise and reference
latents, and the runtime refused before anything was measured. The first control run on BG-2
found it; this keeps it found.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "eval" / "score_submission.sh"


def stage_invocations(src: str) -> dict:
    """Each `BURNISH_ENTRY=<stage>` command, continued across trailing backslashes."""
    out = {}
    lines = src.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"\s*BURNISH_ENTRY=(\w+)\s", line)
        if not m:
            continue
        call = [line]
        j = i
        while lines[j].rstrip().endswith("\\") and j + 1 < len(lines):
            j += 1
            call.append(lines[j])
        out.setdefault(m.group(1), []).append("\n".join(call))
    return out


class TestEveryStageGetsTheGeneration(unittest.TestCase):
    def test_gate_bench_and_score_all_pass_the_generation(self):
        stages = stage_invocations(SCRIPT.read_text())
        self.assertEqual(sorted(stages), ["bench", "gate", "score"])
        self.assertEqual(len(stages["gate"]), 2, "expected a base gate and a candidate gate")
        for name, calls in stages.items():
            for call in calls:
                self.assertIn('--generation "$GEN"', call,
                              f"a {name} stage does not pass the generation, so it measures its "
                              f"default generation instead of the one being scored:\n{call}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
