"""Synthetic, clearly-labelled fixtures. Nothing here is a measurement of anything."""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def calibrated_generation(tmpdir, *, achieved=None, floor_pct=None):
    """BG-1's real frozen definition plus a SYNTHETIC calibration, so the scorer can run.

    The calibration is invented and is labelled as invented in the file it writes. It exists so
    the evaluator's own logic can be tested without hardware; it must never be confused with the
    real `eval/cells/BG-1/reference.json`, which is null on purpose until a 5090 fills it in.
    """
    achieved = achieved or {"t5-encode/1024/bf16": 0.42, "dit-step/1024/bf16": 0.55,
                            "vae-decode/1024/bf16": 0.18}
    floor_pct = floor_pct or {"t5-encode/1024/bf16": 0.60, "dit-step/1024/bf16": 0.35,
                              "vae-decode/1024/bf16": 0.90}
    src = json.loads((ROOT / "eval" / "cells" / "BG-1" / "generation.json").read_text())
    d = Path(tmpdir) / "BG-1"
    d.mkdir(parents=True, exist_ok=True)
    (d / "generation.json").write_text(json.dumps(src, indent=1, sort_keys=True) + "\n")
    (d / "reference.json").write_text(json.dumps({
        "generation": "BG-1",
        "_status": "SYNTHETIC TEST FIXTURE -- these numbers were invented to exercise the "
                   "scorer and are not measurements of anything.",
        "cells": {cid: {"achieved": achieved.get(cid), "floor_pct": floor_pct.get(cid),
                        "floor_repeats": 9, "measured_seconds": None}
                  for cid in [c["id"] for c in src["cells"]]},
    }, indent=1, sort_keys=True) + "\n")
    return d / "generation.json"


def records(generation, *, speedups=None, repeats=5, jitter=0.002, vram=11.0e9,
            base_fidelity=0.004, cand_fidelity=None, cand_vram=None, status="OK"):
    """Paired interleaved raw records whose candidate arm is faster by `speedups[cell]`.

    Both arms carry a NONZERO distance from the reference by default, because both are bf16 and
    the reference is fp32: the base is not the oracle. A fixture that gave the base a perfect
    fidelity would make every candidate look like a quality regression, which is a mistake worth
    not encoding into the tests that are supposed to catch it.
    """
    cand_fidelity = base_fidelity if cand_fidelity is None else cand_fidelity
    cand_vram = vram if cand_vram is None else cand_vram
    speedups = speedups or {}
    out = []
    for cell in generation.scorable_cells():
        base_t = cell.ceiling_seconds / cell.achieved
        s = speedups.get(cell.id, 1.0)
        for k in range(repeats):
            wobble = 1.0 + jitter * ((k % 3) - 1)
            for variant, t in (("base", base_t * wobble),
                               ("candidate", base_t / s * wobble)):
                out.append({
                    "cell": cell.id, "variant": variant, "repeat": k,
                    "config_id": "default", "status": status,
                    "metrics": {
                        "latency_s": t,
                        "peak_vram_bytes": vram if variant == "base" else cand_vram,
                        "latent_l2_vs_reference": (base_fidelity if variant == "base"
                                                   else cand_fidelity)},
                })
    return out


def scratch_generation(tmpdir, name, *, with_prompts=False, **calib):
    """A calibrated generation under `tmpdir/cells/<name>`, outside the repository.

    Tests must not write into the tree they are scoring. An interrupted run that left a
    calibrated generation in `eval/cells/` would leave the next `burnish generation show`
    reporting numbers for a cell nobody measured -- which is the exact confusion this whole
    repository is built to prevent.

    Returns (cells_root, generation_path).
    """
    src = calibrated_generation(Path(tmpdir) / "src", **calib)
    root = Path(tmpdir) / "cells"
    gdir = root / name
    gdir.mkdir(parents=True, exist_ok=True)
    doc = json.loads(src.read_text())
    doc["name"] = name
    (gdir / "generation.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (gdir / "reference.json").write_text((src.parent / "reference.json").read_text())
    if with_prompts:
        # The frozen prompt set is part of the oracle, so tests use the real one.
        (gdir / "prompts.json").write_text(
            (ROOT / "eval" / "cells" / "BG-1" / "prompts.json").read_text())
    return root, gdir / "generation.json"


def provenance(**over):
    p = {"base_commit": "0" * 40, "candidate_commit": "1" * 40,
         "host": "synthetic", "device": "rtx5090", "driver": "n/a",
         "cuda": "n/a", "instrument_from": "0" * 40,
         "_note": "synthetic fixture"}
    p.update(over)
    return p
