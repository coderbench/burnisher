#!/usr/bin/env python3
"""Every measured figure quoted in prose must still match the artifact it came from.

`issues/` is generated, so its numbers cannot go stale. README.md and docs/STATUS.md are written
by hand, and they are the two documents anybody reads first -- which makes them the two places a
stale number does the most damage. They had gone stale by exactly one round of measurement:
STATUS.md was still saying "every cell's achieved fraction is null" and "the runtime cannot
currently run on a device at all" after calibration had run, and README.md still said the CUDA op
backend did not exist.

Nobody noticed because nothing checked. So this checks.

It is deliberately crude: it renders each figure from its artifact exactly as the prose should
spell it, and asserts that string appears in the document. That means re-measuring forces the
prose to be edited, which is the point -- a figure that can drift silently is a figure that will.
The failure message says which artifact to read.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CELL = ROOT / "eval" / "cells" / "BG-1"


def _load(name):
    return json.loads((CELL / name).read_text())


class TestProseMatchesTheArtifacts(unittest.TestCase):
    def setUp(self):
        self.cal = _load("reference.json")["cells"]
        self.lat = _load("dtype-latency.json")
        self.readme = (ROOT / "README.md").read_text()
        self.status = (ROOT / "docs" / "STATUS.md").read_text()

    def _want(self, doc, doc_name, text, artifact):
        self.assertIn(text, doc,
                      f"{doc_name} no longer quotes {text!r}. It is a MEASURED figure and its "
                      f"source of truth is {artifact}; if that artifact changed, the prose has "
                      f"to change with it rather than keep the old number.")

    def test_the_readme_quotes_the_calibrated_achieved_fractions(self):
        for cid in ("dit-step/1024/bf16", "vae-decode/1024/bf16", "t5-encode/1024/bf16"):
            pct = f"{100 * self.cal[cid]['achieved']:.1f}%"
            self._want(self.readme, "README.md", pct, "eval/cells/BG-1/reference.json")

    def test_the_status_page_quotes_the_calibrated_floors(self):
        for cid, cal in self.cal.items():
            self._want(self.status, "docs/STATUS.md", f"{cal['floor_pct']:.3f}%",
                       "eval/cells/BG-1/reference.json")

    def test_both_documents_quote_the_dtype_measurement_consistently(self):
        fp32 = f"{self.lat['dit_step_fp32_s']:.3f} s"
        bf16 = f"{self.lat['dit_step_bf16_s']:.3f} s"
        for doc, name in ((self.readme, "README.md"), (self.status, "docs/STATUS.md")):
            self._want(doc, name, fp32, "eval/cells/BG-1/dtype-latency.json")
            self._want(doc, name, bf16, "eval/cells/BG-1/dtype-latency.json")

    def test_neither_document_still_claims_the_repository_is_unmeasured(self):
        """The specific sentences that were true before the hardware arrived and false after.

        Kept as a list rather than a single check because each one was a separate claim in a
        separate place, and a reader who hits any one of them concludes the whole page is stale.
        """
        stale = [
            "Nothing has been measured on a GPU",
            "the CUDA op backend does not exist yet",
            "there is no CUDA implementation of any op",
            "It has never been near a compiler",
            "the runtime cannot currently run on a device at all",
            "Reference latents for the gate | **do not exist**",
            "Every cell's achieved fraction | **null**",
            "Every cell's noise floor | **null**",
        ]
        for doc, name in ((self.readme, "README.md"), (self.status, "docs/STATUS.md")):
            for claim in stale:
                self.assertNotIn(claim, doc,
                                 f"{name} still says {claim!r}, which stopped being true when "
                                 f"the runtime was calibrated on the pinned part.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
