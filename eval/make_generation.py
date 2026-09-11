#!/usr/bin/env python3
"""Generate a frozen generation definition from the pinned configs. Never type a ceiling.

Every `ceiling_seconds` in `eval/cells/BG-N/generation.json` is computed here from
`configs/candidates.json` and `configs/devices.json` by the same code the scorer uses. Typing
one by hand is how a published table and the scorer come to disagree, and the disagreement is
invisible until a receipt is wrong.

    eval/make_generation.py --name BG-1 --write
    eval/make_generation.py --name BG-1 --check     # CI: regenerate and diff

`--check` is what CI runs. If a config moves and the generation is not regenerated, the check
fails and says which cell drifted. If a generation has already been published and receipts exist
against it, the answer to a drift is BG-2, never an edit -- the check reports that too.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from burnscore import geometry as G
from burnscore.pipeline import pixart_stages
from burnscore.roofline import bound_for

ROOT = Path(__file__).resolve().parent.parent


def build(name, candidate_key, device_key, *, resolution, steps, caption_len, cfg):
    cands = json.loads((ROOT / "configs" / "candidates.json").read_text())["candidates"]
    devices = json.loads((ROOT / "configs" / "devices.json").read_text())
    axes = json.loads((ROOT / "configs" / "axes.json").read_text())
    cand = cands[candidate_key]
    device = devices[device_key]

    cells = []
    # --- the three implemented bf16 cells at the pinned resolution ---
    stages = pixart_stages(cand, resolution=resolution, steps=steps,
                           caption_len=caption_len, cfg=cfg)
    batch = 2 if cfg else 1
    for s in stages:
        cell_id = f"{s.stage}/{resolution}/bf16"
        # Per INVOCATION, not per pipeline: a cell is a thing a kernel change moves, and
        # scoring twenty DiT steps as one cell would hide which step count the gain needs.
        one = s.__class__(stage=s.stage, shape=dict(s.shape), wdtype=s.wdtype,
                          adtype=s.adtype, ops=s.ops, input_bytes=s.input_bytes,
                          output_bytes=s.output_bytes, invocations=1, notes=list(s.notes))
        b = bound_for(one, device, cell=cell_id, device_name=device_key)
        cells.append({
            "id": cell_id, "stage": s.stage,
            "shape": dict(s.shape, invocations_per_generation=s.invocations),
            "wdtype": "bf16", "adtype": "bf16", "implemented": True,
            "weight": round(s.invocations * b.ceiling_seconds, 9),
            "ceiling_seconds": b.ceiling_seconds,
            "ceiling_basis": "model",
            "ceiling_compute_seconds": b.compute_seconds,
            "ceiling_memory_seconds": b.memory_seconds,
            "ceiling_decomposed_seconds": b.decomposed_seconds,
            "bound_by": b.bound_by,
            "flops": b.flops, "unavoidable_bytes": b.unavoidable_bytes,
            "traffic_bytes": b.traffic_bytes, "param_bytes": b.param_bytes,
            "arithmetic_intensity": b.arithmetic_intensity, "ridge_point": b.ridge_point,
            "peak_basis": b.peak_basis,
            "notes": "; ".join(one.notes),
        })

    # --- declared-but-unimplemented dtype cells: the ceiling is computable today ---
    for dt in ("fp8", "nvfp4"):
        d = G.pixart_dit(cand["denoiser"], resolution=resolution, caption_len=caption_len,
                         batch=batch, wdtype=dt, adtype="bf16")
        b = bound_for(d, device, cell=f"dit-step/{resolution}/{dt}", device_name=device_key)
        cells.append({
            "id": f"dit-step/{resolution}/{dt}", "stage": "dit-step",
            "shape": dict(d.shape, invocations_per_generation=steps),
            "wdtype": dt, "adtype": "bf16", "implemented": False, "weight": 0.0,
            "ceiling_seconds": b.ceiling_seconds, "ceiling_basis": "model",
            "ceiling_compute_seconds": b.compute_seconds,
            "ceiling_memory_seconds": b.memory_seconds,
            "ceiling_decomposed_seconds": b.decomposed_seconds,
            "bound_by": b.bound_by, "flops": b.flops,
            "unavoidable_bytes": b.unavoidable_bytes, "traffic_bytes": b.traffic_bytes,
            "param_bytes": b.param_bytes, "arithmetic_intensity": b.arithmetic_intensity,
            "ridge_point": b.ridge_point, "peak_basis": b.peak_basis,
            "notes": (f"DECLARED, NOT IMPLEMENTED. The ceiling is arithmetic and is published "
                      f"so the room is visible; no reference implementation exists on this "
                      f"silicon. Landing one is a cartography contribution. Weight 0 -- it "
                      f"cannot drag the aggregate of a matrix it is not part of."),
        })

    vram = devices[device_key]["vram_bytes"]["value"]
    doc = {
        "name": name,
        "description": (f"{cand['family']} at {resolution}px, {steps} steps, "
                        f"{'CFG' if cfg else 'no CFG'}, on one {device['name']}. Three "
                        f"implemented bf16 cells scored on latency, peak VRAM and output "
                        f"fidelity; two declared fp8/NVFP4 cells published for their ceilings."),
        "_what_a_generation_is": (
            "Everything here is FROZEN for the lifetime of " + name + ". A receipt stays "
            "attached to the generation that produced it, so if the meaning of the evaluation "
            "changes materially the answer is the next generation, never an edit here. The "
            "generation's SHA-256 is recorded in every receipt and `burnish receipt verify` "
            "refuses one whose generation has moved."),
        "_ceilings_are_arithmetic": (
            "Every `ceiling_seconds` is `max(flops/peak, unavoidable_bytes/bandwidth)` computed "
            "from the model config and the device peak. It is a LOWER BOUND ON TIME, it is not "
            "reachable, and `peak_basis` says whether the peak behind it was measured on the "
            "part or taken from a specification. `unavoidable_bytes` excludes every intermediate "
            "on purpose: an intermediate is removable by fusion, and a ceiling that moved when a "
            "contributor fused would not be a ceiling."),
        "model": {
            "key": candidate_key, "repo": cand["repo"], "revision": cand.get("revision"),
            "text_encoder_repo": cand.get("text_encoder_repo"),
            "text_encoder_revision": cand.get("text_encoder_revision"),
            "license": cand["license"], "gated": cand["gated"],
            "resolution": resolution, "steps": steps, "caption_len": caption_len,
            "classifier_free_guidance": cfg, "batch": batch,
            "scheduler": cand["scheduler"],
            "_pin_note": ("The revision is pinned because the reference drifts between versions "
                          "and it is the oracle for everything else. A moved revision is a "
                          "changed oracle and therefore a new generation."),
        },
        "device": device_key,
        "objectives": [
            {"key": "latency_s", "direction": "min", "lo": 60.0, "hi": 0.0, "unit": "s",
             "_note": "End-to-end wall time for one generation. `lo` is the timeout: a run at "
                      "or beyond it scores zero on this axis rather than being a slow point."},
            {"key": "peak_vram_bytes", "direction": "min", "lo": float(vram), "hi": 0.0,
             "unit": "B",
             "_note": "Peak device allocation. `lo` is the card, because a configuration that "
                      "does not fit is the ABSENCE of an operating point and must not normalize "
                      "into a small positive score that still contributes volume."},
            {"key": "latent_l2_vs_reference", "direction": "min", "lo": 0.02, "hi": 0.0,
             "unit": "relative L2",
             "_note": "Distance from the pinned reference latents. `lo` is the correctness "
                      "tolerance, so a change that stays inside the gate while measurably "
                      "degrading shows up as a smaller number here rather than as an invisible "
                      "pass. This is the axis that makes 'faster but worse' a move along the "
                      "frontier instead of a win."},
        ],
        "reference_point": [0.0, 0.0, 0.0],
        "aggregation": "weighted_mean",
        "_weights_note": ("Each implemented cell is weighted by its share of the pipeline's "
                          "predicted wall time (invocations x ceiling). Published rather than "
                          "implied. Equal weights would price a once-per-generation VAE decode "
                          "the same as a DiT step that runs twenty times, which is not what a "
                          "user of the runtime experiences."),
        "confidence_level": 0.99,
        "bootstrap_resamples": 20000,
        "bootstrap_seed": 20260911,
        "repeats": axes["receipt_shape"]["repeats"],
        "tolerance": {
            "latent_l2_relative": 0.02,
            "latent_max_abs": 0.05,
            "determinism_replays": 10,
            "determinism_rule": "byte-identical latents across replays of the same build",
            "_justification": (
                "The bf16 pipeline is compared against an fp32 CPU reference of the same graph, "
                "so the tolerance has to admit bf16 rounding accumulated over the whole denoise "
                "loop and admit nothing else. 2% relative L2 is roughly four times the "
                "step-to-step drift bf16 rounding alone produces over 20 DPM-Solver++ steps at "
                "this resolution, which leaves room for a legitimately different kernel order "
                "and not for a different algorithm. It is a STATED threshold and it is "
                "falsifiable: `burnish gate --calibrate-tolerance` measures the bf16-vs-fp32 "
                "drift directly and the number here is expected to move once when it first "
                "runs on hardware. DETERMINISM is separate and is not a tolerance -- the same "
                "build must reproduce ITSELF exactly, or it cannot be a reference for anything."),
        },
        "held_out": axes["held_out"],
        "cells": cells,
        "_cells_note": (
            "A cell is (stage, shape, dtype) and is scored PER INVOCATION. The DiT cell runs "
            f"{steps} times per generation and the other two run once; that is carried in the "
            "weight, not folded into the cell, so a receipt says which stage moved."),
        "_calibration": (
            "`achieved` and `floor_pct` are NOT here. They live in reference.json because they "
            "are measurements, they require the pinned hardware, and a generation that shipped "
            "them as part of its frozen definition could never be calibrated without being "
            "reissued. Until they are filled in, every cell is `calibrated: false` and the "
            "scorer refuses to produce a receipt."),
    }
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="BG-1")
    ap.add_argument("--candidate", default="pixart-sigma-xl2-1024")
    ap.add_argument("--device", default="rtx5090")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--caption-len", type=int, default=300)
    ap.add_argument("--no-cfg", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    doc = build(args.name, args.candidate, args.device, resolution=args.resolution,
                steps=args.steps, caption_len=args.caption_len, cfg=not args.no_cfg)
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    path = ROOT / "eval" / "cells" / args.name / "generation.json"

    if args.check:
        if not path.exists():
            print(f"!! {path} does not exist; run --write", file=sys.stderr)
            return 2
        current = path.read_text()
        if current != text:
            print(f"!! {path} is out of date with configs/. Regenerate with:\n"
                  f"     eval/make_generation.py --name {args.name} --write\n"
                  f"   If receipts already exist against {args.name}, a drifted config means a "
                  f"CHANGED ORACLE and the answer is a new generation, not an edit.",
                  file=sys.stderr)
            old = json.loads(current)
            oldc = {c["id"]: c.get("ceiling_seconds") for c in old.get("cells", [])}
            for c in doc["cells"]:
                was = oldc.get(c["id"])
                if was != c["ceiling_seconds"]:
                    print(f"   {c['id']}: {was} -> {c['ceiling_seconds']}", file=sys.stderr)
            return 1
        print(f"ok: {path} matches configs/")
        return 0

    if args.write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        ref = path.parent / "reference.json"
        if not ref.exists():
            ref.write_text(json.dumps({
                "generation": args.name,
                "_what_this_is": (
                    "Per-cell CALIBRATION: the fraction of the arithmetic ceiling currently "
                    "achieved, and the measured run-to-run noise floor. Both are measurements "
                    "and neither can be computed from a config file."),
                "_status": (
                    "UNCALIBRATED. No cell has been measured on the pinned hardware, so every "
                    "`achieved` and `floor_pct` below is null and the scorer will refuse to "
                    "produce a receipt. This is the honest state of the repository and not an "
                    "oversight: publishing a plausible achieved fraction would make every "
                    "gap-closed score a fiction with a checksum on it."),
                "_how_to_fill_this_in": "burnish calibrate --generation " + args.name +
                                        " --repeats 9 --write",
                "device_probe": None,
                "_device_probe_note": (
                    "Null until `burnish probe --device` runs. Until then the ceilings behind "
                    "these cells stand on VENDOR peaks, which no kernel reaches -- so every "
                    "achieved fraction computed against them is a LOWER bound on how done a "
                    "cell really is."),
                "cells": {c["id"]: {"achieved": None, "floor_pct": None,
                                    "floor_repeats": None,
                                    "measured_seconds": None,
                                    "peak_vram_bytes": None}
                          for c in doc["cells"]},
            }, indent=1, sort_keys=True) + "\n")
            print(f">> wrote {ref} (uncalibrated)")
        print(f">> wrote {path}")
        return 0

    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
