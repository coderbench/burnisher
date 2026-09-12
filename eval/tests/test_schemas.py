"""The hard rules, enforced by a test rather than by a convention.

Two of these encode rules the brief states as non-negotiable, and both are the kind of rule that
decays the moment nothing checks it:

    "Never publish a predicted figure as a gain. Cost-model output carries `basis: model` and a
     schema test enforces it."

    "No letter grades."

The rest check that the artifacts this repository publishes actually match the schemas it
publishes them under, because a schema nobody validates against is documentation.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))

from burnscore import cells as C, compute as CP, receipt as R, roofline as RL
from burnscore import geometry as G
from tests import fixtures

BANNED_BANDS = ("XS", "XL")
BANNED_KEYS = ("impact_band", "tier", "grade", "score_band", "band")


def walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield f"{path}.{k}", k, v
            yield from walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f"{path}[{i}]")


class TestNoPredictedFigureIsEverAGain(unittest.TestCase):
    """`basis` is the whole mechanism. A modelled number must never sit in a measured field."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.gen = C.load(fixtures.calibrated_generation(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_roofline_bound_declares_basis_model(self):
        cand = json.loads((ROOT / "configs" / "candidates.json").read_text())
        dev = json.loads((ROOT / "configs" / "devices.json").read_text())["rtx5090"]
        p = G.pixart_dit(cand["candidates"]["pixart-sigma-xl2-1024"]["denoiser"],
                         resolution=1024, caption_len=300, batch=2)
        b = RL.bound_for(p, dev, cell="x")
        self.assertEqual(b.basis, "model")
        self.assertEqual(b.to_json()["basis"], "model")

    def test_the_frozen_generation_marks_every_ceiling_as_model(self):
        doc = json.loads((ROOT / "eval" / "cells" / "BG-1" / "generation.json").read_text())
        for cell in doc["cells"]:
            self.assertEqual(cell["ceiling_basis"], "model", cell["id"])

    def test_a_receipt_with_a_modelled_score_is_refused(self):
        out = CP.compute(self.gen, fixtures.records(
            self.gen, speedups={"dit-step/1024/bf16": 1.15}))
        rec = R.build_receipt(
            generation=self.gen, per_cell=out["per_cell"], aggregate=out["aggregate"],
            interval=out["interval"], frontier=out["frontier"], correctness="PASS",
            determinism=True, coverage=out["coverage"], held_out=True,
            provenance=fixtures.provenance())
        self.assertEqual(rec["score"]["basis"], "measured")
        for bad in ("model", "estimated", "predicted", None):
            tampered = json.loads(json.dumps(rec))
            tampered["score"]["basis"] = bad
            tampered["content_digest"] = R.content_digest(tampered)
            with self.assertRaises(R.ReceiptError, msg=f"basis {bad!r} was accepted"):
                R.verify_receipt(tampered, self.gen)

    def test_a_receipt_with_a_measured_ceiling_is_refused(self):
        out = CP.compute(self.gen, fixtures.records(
            self.gen, speedups={"dit-step/1024/bf16": 1.15}))
        rec = R.build_receipt(
            generation=self.gen, per_cell=out["per_cell"], aggregate=out["aggregate"],
            interval=out["interval"], frontier=out["frontier"], correctness="PASS",
            determinism=True, coverage=out["coverage"], held_out=True,
            provenance=fixtures.provenance())
        rec["per_cell"]["dit-step/1024/bf16"]["ceiling_basis"] = "measured"
        rec["content_digest"] = R.content_digest(rec)
        with self.assertRaises(R.ReceiptError) as cm:
            R.verify_receipt(rec, self.gen)
        self.assertIn("arithmetic", str(cm.exception))

    def test_the_screen_output_is_entirely_modelled(self):
        import subprocess
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            subprocess.run([sys.executable, str(ROOT / "eval" / "screen.py"),
                            "--json", f.name], check=True, capture_output=True)
            doc = json.loads(Path(f.name).read_text())
        self.assertEqual(doc["basis"], "model")
        for path, key, value in walk(doc):
            if key == "basis":
                self.assertEqual(value, "model", f"{path} claims basis {value!r}; the screen "
                                                 f"runs no model and measures nothing")


class TestNoLetterGrades(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.gen = C.load(fixtures.calibrated_generation(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_status_vocabulary_carries_no_magnitude(self):
        for s in R.STATUSES:
            self.assertNotIn(s, BANNED_BANDS)
            self.assertNotIn(s.lower(), ("small", "medium", "large", "xlarge"))

    def test_no_banned_key_appears_in_a_receipt_or_a_generation(self):
        out = CP.compute(self.gen, fixtures.records(
            self.gen, speedups={"dit-step/1024/bf16": 1.15}))
        rec = R.build_receipt(
            generation=self.gen, per_cell=out["per_cell"], aggregate=out["aggregate"],
            interval=out["interval"], frontier=out["frontier"], correctness="PASS",
            determinism=True, coverage=out["coverage"], held_out=True,
            provenance=fixtures.provenance())
        gen_doc = json.loads((ROOT / "eval" / "cells" / "BG-1" / "generation.json").read_text())
        for doc, name in ((rec, "receipt"), (gen_doc, "generation")):
            for path, key, value in walk(doc):
                self.assertNotIn(key, BANNED_KEYS, f"{name}{path}")
                if isinstance(value, str):
                    self.assertNotIn(value, BANNED_BANDS, f"{name}{path}")


class TestPublishedArtifactsMatchTheirSchemas(unittest.TestCase):
    """Hand-rolled validation of the fields that carry meaning. A full JSON Schema validator is
    a dependency this repository does not take; what matters is that the required fields exist,
    the enums are respected, and the two basis fields cannot be confused."""

    def _schema(self, name):
        return json.loads((ROOT / "schemas" / name).read_text())

    def _check_required(self, doc, schema, label):
        for key in schema.get("required", []):
            self.assertIn(key, doc, f"{label} is missing required field {key!r}")

    def test_receipt_schema_matches_the_code(self):
        schema = self._schema("receipt.schema.json")
        self.assertEqual(sorted(schema["properties"]["status"]["enum"]), sorted(R.STATUSES),
                         "the schema's status list and receipt.STATUSES have drifted; a status "
                         "the schema does not know is a receipt nothing validates")
        self.assertEqual(schema["properties"]["score"]["properties"]["basis"]["const"],
                         "measured")

    def test_a_real_receipt_validates(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            gen = C.load(fixtures.calibrated_generation(tmp.name))
            out = CP.compute(gen, fixtures.records(
                gen, speedups={"dit-step/1024/bf16": 1.15}))
            rec = R.build_receipt(
                generation=gen, per_cell=out["per_cell"], aggregate=out["aggregate"],
                interval=out["interval"], frontier=out["frontier"], correctness="PASS",
                determinism=True, coverage=out["coverage"], held_out=True,
                provenance=fixtures.provenance())
            schema = self._schema("receipt.schema.json")
            self._check_required(rec, schema, "receipt")
            self.assertIn(rec["status"], schema["properties"]["status"]["enum"])
            self.assertIn(rec["correctness"],
                          schema["properties"]["correctness"]["enum"])
            for cid, cell in rec["per_cell"].items():
                self._check_required(
                    cell, schema["properties"]["per_cell"]["additionalProperties"], cid)
                self.assertGreater(cell["achieved_base"], 0)
                self.assertLessEqual(cell["achieved_candidate"], 1.000000001)
        finally:
            tmp.cleanup()

    def test_the_published_roofline_rows_validate_and_admit_nulls(self):
        import subprocess
        schema = self._schema("roofline.schema.json")
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            subprocess.run([sys.executable, str(ROOT / "eval" / "roofline_table.py"),
                            "--json", f.name], check=True, capture_output=True)
            doc = json.loads(Path(f.name).read_text())
        self.assertTrue(doc["rows"])
        for row in doc["rows"]:
            self._check_required(row, schema, row["cell"])
            self.assertEqual(row["ceiling_basis"], "model")
            self.assertIn(row["peak_basis"], schema["properties"]["peak_basis"]["enum"])
            # The schema must ALLOW null here: a cell that has never been calibrated publishes
            # no achieved fraction, and that is the honest state for a newly added cell.
            #
            # What this once asserted was that every row IS null, because no hardware had run in
            # this tree. Hardware has now run, so asserting absence would assert a stale fact.
            # The guard underneath it does not retire, though: it was never really about null,
            # it was about a published number that nobody measured. So it now checks the
            # property that actually distinguishes the two -- every non-null figure in the
            # published table must be DERIVED from the calibration receipt, to the digit, rather
            # than typed next to it.
            if row["achieved"] is None:
                self.assertIsNone(row["floor_pct"], f"{row['cell']} publishes a noise floor "
                                                    f"without an achieved fraction; half a "
                                                    f"calibration is not a calibration")
                continue
            cal = self._calibration().get(row["cell"])
            self.assertIsNotNone(cal, f"{row['cell']} publishes an achieved fraction with no "
                                      f"entry in reference.json; nobody measured this")
            self.assertEqual(cal["basis"], "measured",
                             f"{row['cell']} publishes a figure whose basis is not a "
                             f"measurement. A modelled number is never a gain.")
            self.assertAlmostEqual(row["achieved"], cal["achieved"], places=12,
                                   msg=f"{row['cell']}: the published achieved fraction is not "
                                       f"the calibrated one")
            self.assertAlmostEqual(row["floor_pct"], cal["floor_pct"], places=12,
                                   msg=f"{row['cell']}: the published floor is not the "
                                       f"calibrated one")
            self.assertTrue(row["resolvable"])

    def _calibration(self):
        ref = json.loads((ROOT / "eval" / "cells" / "BG-1" / "reference.json").read_text())
        # A calibration with no box behind it is a guess with provenance-shaped fields. The
        # probe identifies the exact part, because two RTX 5090s differ by 3% on GEMM and an
        # achieved fraction carried over from another box is a number nobody measured HERE.
        probe = ref["device_probe"]
        for k in ("name", "uuid", "driver_version"):
            self.assertTrue(probe.get(k), f"reference.json calibration has no {k}")
        return ref["cells"]


if __name__ == "__main__":
    unittest.main(verbosity=2)
