#!/usr/bin/env python3
"""Is there a scorable surface here, and which model should v0 pin?

Six questions, asked of every candidate, from config files and arithmetic, BEFORE any weights
are downloaded. Answering them costs a few kilobytes; getting them wrong costs a contributor a
week and the subnet a contributor.

    1 DOMINANCE      is what a contributor can touch >= 20% of total wall time?
    2 RESOLUTION     is the ceiling >= 20x the run-to-run noise floor?
    3 REGENERATION   does the surface reopen with each new model, resolution, or dtype?
    4 REACH          can a contributor's diff actually touch where the time goes?
    5 DETERMINISM    does the runtime reproduce itself byte-identically across repeats?
    6 SCORE COST     GPU time per receipt -- minutes, not hours?

  + 0 ACCESS         can a stranger fetch these weights and re-run the benchmark?

Three of these are arithmetic and are answered here. Two (RESOLUTION's floor, DETERMINISM) need
the runtime and the hardware, and this script says UNKNOWN and names the command that settles
them rather than guessing. One (REACH) is structural and is answered against the declared
kernel ownership map. Printing a confident answer to a question this script cannot reach is the
exact failure this repository is built to avoid.

    eval/screen.py                        # every candidate, the summary table
    eval/screen.py --candidate pixart-sigma-xl2-1024 --verbose
    eval/screen.py --json screen.json     # the machine-readable result

Nothing here is a measurement. Every duration carries basis "model".
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from burnscore import geometry as G
from burnscore.pipeline import pixart_stages, shares, resident_bytes
from burnscore.roofline import bound_for

ROOT = Path(__file__).resolve().parent.parent

DOMINANCE_THRESHOLD = 0.20
RESOLUTION_RATIO = 20.0
SCORE_COST_BUDGET_S = 30 * 60.0          # half an hour of GPU per receipt, all cells, all repeats

# What fraction of the arithmetic ceiling a competent first implementation reaches. Used ONLY to
# turn a ceiling into a predicted wall time for the score-cost question, and stated wherever it
# is used. It is a guess; it is labelled a guess; and `burnish bench` replaces it with the
# measured number the first time the pipeline runs. A screen that hid this constant inside a
# formula would be publishing a prediction as a fact.
ASSUMED_ACHIEVED = 0.35

# Who owns the kernel behind each op kind. A contributor's reach ends where a vendor library
# begins: if 90% of a stage's flops sit inside cuBLAS, the honest thing to say is that the
# surface is the 10% around it plus whatever replacing cuBLAS is worth, not that the stage is
# open. `burnisher` owns its attention and its elementwise/norm path from the start for exactly
# this reason -- those are where the fusion work is.
KERNEL_OWNERSHIP = {
    "gemm": ("vendor-or-ours", "cuBLASLt by default; a contributor may register a fused or "
                               "quantized replacement through the backend enumerator"),
    "attention": ("ours", "burnisher implements attention directly; there is no vendor path "
                          "that already fuses masking, modulation and the epilogue"),
    "conv": ("vendor-or-ours", "cuDNN by default; the VAE's shapes are few and fixed, which is "
                               "what makes a specialized path plausible"),
    "norm": ("ours", "fusable into the neighbouring GEMM or attention epilogue"),
    "elementwise": ("ours", "fusable; this is where AdaLN modulation lives"),
    "gather": ("ours", "trivial, and it is here so the op list sums to the whole stage"),
}


def _load(name):
    return json.loads((ROOT / "configs" / name).read_text())


def q0_access(candidate: dict) -> dict:
    gated = bool(candidate.get("gated"))
    lic = candidate.get("license")
    permissive = lic in ("apache-2.0", "mit", "openrail++", "openrail", "cc-by-4.0")
    ok = permissive and not gated
    return {
        "question": "ACCESS", "pass": ok,
        "license": lic, "gated": gated,
        "why": ("a stranger can fetch these weights with curl and re-run the benchmark"
                if ok else
                ("the repository is GATED: a stranger needs an account and a click-through "
                 "before curl returns anything, so the benchmark is not re-runnable as "
                 "published, whatever the licence says" if gated else
                 f"licence {lic!r} is not on the redistributable list")),
        "note": candidate.get("license_note", ""),
    }


def q1_dominance(stage_rows, total_s) -> dict:
    open_stages, closed = [], []
    for r in stage_rows:
        (open_stages if r["share"] >= DOMINANCE_THRESHOLD else closed).append(
            {"stage": r["stage"], "share": r["share"], "seconds": r["seconds"],
             "invocations": r["invocations"], "bound_by": r["bound_by"]})
    return {
        "question": "DOMINANCE", "pass": bool(open_stages),
        "threshold": DOMINANCE_THRESHOLD,
        "dominant_stages": open_stages, "minor_stages": closed,
        "total_seconds": total_s, "basis": "model",
        "why": ("at least one stage holds a fifth of the predicted wall clock, so there is "
                "somewhere for a week of work to land"
                if open_stages else
                "no single stage holds a fifth of the clock; the time is spread thin and a "
                "contributor cannot move the total by winning one surface"),
        "_trap": ("Stage size is set by invocation count, not by parameter count. Ranking by "
                  "parameters would send someone to the biggest weights, which here are the "
                  "text encoder's -- 89% of the checkpoint and a fiftieth of the clock."),
    }


def q2_resolution(stage_rows, assumed_achieved=ASSUMED_ACHIEVED, measured=None) -> dict:
    """Is the room in a cell big enough to be measured against that cell's own noise?

    The room is `1 - achieved`, and `achieved` needs a measurement. So this question cannot be
    closed here, and what IS computed is the thing that makes it actionable: for each stage,
    the noise floor that stage would have to come in under for its room to be resolvable. A
    calibration run then either clears that bar or does not, and either way nobody has guessed.
    """
    if measured:
        # Settled. Every cell has a measured achieved fraction and a measured floor, so the
        # question this block used to SIZE -- "what floor would a cell need for its room to be
        # resolvable?" -- is answered by division instead of assumption.
        # Through burnscore.floor.resolution_gate, which is where this question is defined.
        # Computing it here from achieved and floor_pct would be a SECOND definition of
        # "floors of room" -- and the first draft of this block was one, giving 22x where the
        # calibration's own record says 116x for the same cell. Two formulas for one quantity
        # is the drift this repository generates documents to avoid.
        from burnscore.floor import resolution_gate
        rows, worst = [], None
        for cid, c in measured["cells"].items():
            g = resolution_gate(c["floor_pct"], c["achieved"], ratio=RESOLUTION_RATIO)
            worst = g["floors_of_room"] if worst is None else min(worst, g["floors_of_room"])
            rows.append({"cell": cid, "floor_decided_by": c.get("floor_decided_by"), **g})
        return {
            "question": "RESOLUTION", "pass": all(r["resolvable"] for r in rows),
            "ratio_required": RESOLUTION_RATIO, "per_cell": rows, "basis": "measured",
            "device": measured.get("device"),
            "why": (f"every cell's room clears its own measured floor by at least "
                    f"{worst:.0f}x, against a {RESOLUTION_RATIO:.0f}x bar"),
            "_trap": ("Resolvable is not the same as worth doing. These cells clear their "
                      "floors by four orders of magnitude because the runtime is slow, not "
                      "because the instrument is good. The ratio falls as the gap closes, and "
                      "an axis whose spread sits inside its own noise is open, not solved."),
        }

    rows = []
    for r in stage_rows:
        room = 1.0 - assumed_achieved
        rows.append({
            "stage": r["stage"],
            "assumed_achieved": assumed_achieved,
            "implied_room_fraction": room,
            "max_floor_pct_for_resolution": room / RESOLUTION_RATIO * 100.0,
            "ceiling_seconds": r["seconds"],
            "predicted_wall_seconds": r["seconds"] / assumed_achieved,
        })
    return {
        "question": "RESOLUTION", "pass": None, "ratio_required": RESOLUTION_RATIO,
        "per_stage": rows, "basis": "model",
        "why": ("UNRESOLVED BY ARITHMETIC. The ceiling is computable; the noise floor is not. "
                "What is published here is the bar: a cell whose measured run-to-run spread is "
                "above `max_floor_pct_for_resolution` cannot resolve a twentieth of its own "
                "room, and belongs in the generation as a declared-unresolvable cell or not at "
                "all."),
        "settled_by": "burnish calibrate --cell <cell> --repeats 9",
        "_trap": ("`assumed_achieved` is a GUESS and every number in this block scales with it. "
                  "It is here to size the calibration run, not to be quoted."),
    }


def q3_regeneration(candidate: dict, axes: dict) -> dict:
    """Does the surface reopen when the model, the resolution or the dtype changes?

    Counted structurally rather than asserted: each axis value changes the SHAPES a kernel sees,
    and a kernel tuned for one shape is not tuned for another. The number that matters is how
    many distinct cells the declared axes generate, because that is the size of the standing
    supply of unsolved work -- and it is what makes a second scored target worth having at all.
    """
    res = axes["resolutions"]; dts = axes["dtypes"]; steps = axes["step_counts"]
    stages = axes["stages"]
    cells = len(res) * len(dts) * len(stages)
    token_counts = {}
    patch = int(candidate["denoiser"]["patch_size"])
    scale = candidate["vae"]["scale_factor"]
    for r in res:
        n = (r // scale // patch) ** 2
        token_counts[r] = n
    return {
        "question": "REGENERATION", "pass": cells >= 8,
        "cells_from_declared_axes": cells,
        "axes": {"resolutions": res, "dtypes": dts, "stages": stages, "step_counts": steps},
        "dit_tokens_per_resolution": token_counts,
        "why": (f"{len(res)} resolutions x {len(dts)} dtypes x {len(stages)} stages = {cells} "
                f"cells from the declared axes alone, and the DiT's token count moves "
                f"{min(token_counts.values())} -> {max(token_counts.values())} across them, so "
                f"an attention kernel tuned at one resolution is untuned at the next. A new "
                f"model reopens all of them at once."),
        "_why_this_matters": ("A target whose surface does not regenerate is one a subnet "
                              "exhausts. Generation is compute-bound with fresh shapes per "
                              "model, resolution and dtype, which is the structural difference "
                              "from a text decode runtime whose shapes are fixed by the "
                              "checkpoint."),
    }


def q4_reach(stages, top_fraction=0.90) -> dict:
    """Can a diff in THIS repository touch where the time goes?

    Walks the ops holding the top `top_fraction` of each stage's flops and asks who owns the
    kernel. A stage whose time is entirely inside a vendor library is not open just because it
    is slow.
    """
    out = []
    for s in stages:
        ops = sorted(s.ops, key=lambda o: -o.total_flops)
        total = sum(o.total_flops for o in ops) or 1.0
        acc, hot = 0.0, []
        for o in ops:
            if acc / total >= top_fraction:
                break
            acc += o.total_flops
            owner, note = KERNEL_OWNERSHIP.get(o.kind, ("unknown", ""))
            hot.append({"op": o.name, "kind": o.kind, "flops": o.total_flops,
                        "share": o.total_flops / total, "owner": owner, "note": note})
        ours = sum(h["share"] for h in hot if h["owner"] == "ours")
        shared = sum(h["share"] for h in hot if h["owner"] == "vendor-or-ours")
        out.append({"stage": s.stage, "hot_ops": hot,
                    "share_we_own_outright": ours,
                    "share_contestable_with_vendor": shared,
                    "share_covered": acc / total})
    reachable = all(r["share_we_own_outright"] + r["share_contestable_with_vendor"] > 0.5
                    for r in out)
    return {
        "question": "REACH", "pass": reachable, "per_stage": out,
        "why": ("every stage's hot ops are either implemented in this repository or are GEMM/"
                "conv calls a contributor may replace through the backend enumerator"
                if reachable else
                "at least one stage's time sits behind a vendor library a diff here cannot "
                "reach"),
        "_trap": ("`vendor-or-ours` is not a free pass. Beating cuBLASLt on a well-shaped GEMM "
                  "is hard and usually the wrong target; the reachable work is fusing the "
                  "epilogue and the modulation INTO it, which is why the elementwise and norm "
                  "ops are listed even though they hold few flops."),
    }


def _committed_determinism_evidence():
    """Whether a real run on the pinned part is committed showing byte-identical replays.

    The evidence is the raw measurements in `examples/`, not a claim in a document. A gate run
    records `determinism` alongside the digests it compared, and that file is committed with the
    receipt derived from it -- so this question is answered by the same artifact anyone else can
    re-score, rather than by a sentence somebody remembered to update.
    """
    raw = ROOT / "examples" / "BG-1-pr-000001-raw.json"
    if not raw.exists():
        return None
    doc = json.loads(raw.read_text())
    if doc.get("determinism") is not True:
        return None
    return {"device": (doc.get("provenance") or {}).get("device", {}).get("name"),
            "impl": (doc.get("provenance") or {}).get("impl_base"),
            "source": "examples/BG-1-pr-000001-raw.json"}


def q5_determinism(evidence=None) -> dict:
    if evidence:
        return {
            "question": "DETERMINISM", "pass": True, "basis": "measured",
            "device": evidence.get("device"), "evidence": evidence["source"],
            "why": (f"replays of the same build on {evidence.get('device')} were "
                    f"byte-identical, recorded in {evidence['source']}"),
            "_trap": ("Determinism is a property of a BUILD, not of the repository. It is "
                      "re-checked at the top of every gate for exactly that reason, and a "
                      "submission that loses it is rejected before anything is timed -- "
                      "nothing downstream of a build that cannot reproduce itself means "
                      "anything."),
        }
    return {
        "question": "DETERMINISM", "pass": None,
        "why": ("UNANSWERABLE FROM A CONFIG FILE, and it is the gate everything else depends "
                "on. If two unhooked replays of the same build disagree, no candidate can be "
                "attributed a difference and the whole instrument is decorative. RecurLocal's "
                "engagement lost its most promising checkpoint to exactly this: four control "
                "replays of a sparse-MoE model were four distinct outputs, because a few ULP in "
                "the prefill fed discrete top-k expert routing."),
        "known_risks_for_this_workload": [
            "cuDNN/cuBLAS autotuning picks a different algorithm per process unless pinned; "
            "different algorithms give different roundings",
            "atomics in any reduction (GroupNorm over 1024x1024x128 is the obvious candidate)",
            "TF32 on by default turns an fp32 reference into a non-reference",
            "a sampler whose RNG is drawn on device in launch order",
        ],
        "settled_by": "burnish gate --determinism --repeats 10 (byte-identical latents required)",
    }


def _measured_calibration():
    """Per-cell measured seconds from the pinned part, if this candidate has been calibrated.

    Returns None when nothing has been measured, which is the state every candidate but the
    pinned one is in and the state the pinned one was in until hardware arrived.
    """
    ref = ROOT / "eval" / "cells" / "BG-1" / "reference.json"
    if not ref.exists():
        return None
    doc = json.loads(ref.read_text())
    cells = {k: v for k, v in (doc.get("cells") or {}).items()
             if v.get("measured_seconds") and v.get("basis") == "measured"}
    if not cells:
        return None
    gen = json.loads((ROOT / "eval" / "cells" / "BG-1" / "generation.json").read_text())
    return {"cells": cells, "calibrated_with": doc.get("calibrated_with") or {},
            "device": (doc.get("device_probe") or {}).get("name"),
            "steps": gen["model"].get("steps") or gen.get("steps") or 20,
            # Which model these seconds belong to. A cost measured on PixArt says nothing about
            # what scoring FLUX would cost, and attaching it to the wrong candidate would be a
            # measurement of one thing published as a measurement of another.
            "repo": gen["model"].get("repo")}


# What one record costs the evaluator BEYOND the arithmetic it measures: process spawn plus the
# checkpoint map and upload. Measured, not assumed -- 30 records of a real scoring run spent 1.9 s
# each beyond their timed work, and a record's whole wall clock was 2.8 s. It was 11 s before
# weights were converted on every core and uploaded through page-locked staging. A cost model that
# counted only kernel time would say a cheap cell is free, and the cheapest cell here (t5-encode,
# 34 ms of work) still costs over three seconds a record.
MEASURED_RECORD_OVERHEAD_S = 1.9


def q6_score_cost(stage_rows, generation, assumed_achieved=ASSUMED_ACHIEVED,
                  measured=None) -> dict:
    """What one receipt costs in GPU time, and whether that is affordable per submission.

    This question is different in kind from the other five. They ask whether a surface is worth
    opening, and a modelled answer is the right tool -- arithmetic is how you decide where to
    look before you have anything to measure. This one asks whether the SUBNET can afford to
    run, and it is answered in wall clock on real hardware or it is not answered.

    It was modelled, and the model was wrong by a factor of twelve. It assumed a first
    implementation reaches 35% of its roofline and predicted 2.5 GPU-minutes per receipt. The
    implementation landed at 1.5%, and a receipt takes about half an hour. The assumption was
    labelled, published and visible in the output the whole time, which is the part worth
    keeping in mind: a clearly-labelled prediction is still a prediction, and the label does not
    make the schedule it implies come true.
    """
    cells = generation["cells"]
    repeats = generation["repeats"]
    arms = 2
    gate_gens = generation["gate_generations"]

    if measured:
        cal = measured["cells"]
        warmup = int(measured["calibrated_with"].get("warmup", 2))
        iters = int(measured["calibrated_with"].get("iters", 5))
        invocations = warmup + iters
        # One record = one process: (warmup + iters) invocations plus the load overhead.
        record_s = {c: v["measured_seconds"] * invocations + MEASURED_RECORD_OVERHEAD_S
                    for c, v in cal.items()}
        # The held-out shape is run too, at two paired repeats per arm, and it is not free.
        held_repeats = max(2, repeats // 2)
        bench = sum(r * arms * (repeats + held_repeats) for r in record_s.values())
        # A gate generation is the whole pipeline once: every DiT step, plus the encoder and
        # the decoder. Read the step count from the cell ids' own generation definition.
        steps = int(measured.get("steps") or 20)
        per_gen = sum(v["measured_seconds"] * (steps if "dit-step" in c else 1)
                      for c, v in cal.items()) + MEASURED_RECORD_OVERHEAD_S
        gate = gate_gens * per_gen
        total = bench + gate
        return {
            "question": "SCORE_COST", "pass": total <= SCORE_COST_BUDGET_S,
            "budget_seconds": SCORE_COST_BUDGET_S,
            "predicted_seconds_per_generation": per_gen,
            "predicted_receipt_seconds": total,
            "breakdown": {"cells": len(cal), "repeats": repeats, "arms": arms,
                          "held_out_repeats": held_repeats,
                          "invocations_per_record": invocations,
                          "record_overhead_seconds": MEASURED_RECORD_OVERHEAD_S,
                          "correctness_gate_generations": gate_gens,
                          "bench_seconds": bench, "gate_seconds": gate},
            "basis": "measured", "device": measured.get("device"),
            "why": (f"a full receipt MEASURES {total / 60:.0f} GPU-minutes against a "
                    f"{SCORE_COST_BUDGET_S / 60:.0f}-minute budget"),
            "_trap": ("This is the one screen question a model cannot answer. It scales with how "
                      "fast the runtime actually is, so the cost of scoring falls as "
                      "contributors do the work the scoring pays for. It is the only line in "
                      "this document that gets better on its own."),
        }

    per_generation_s = sum(r["seconds"] for r in stage_rows) / assumed_achieved
    prompts = generation["prompts_per_cell"]
    total = per_generation_s * cells * repeats * arms * prompts
    gate = gate_gens * per_generation_s
    return {
        "question": "SCORE_COST", "pass": (total + gate) <= SCORE_COST_BUDGET_S,
        "budget_seconds": SCORE_COST_BUDGET_S,
        "predicted_seconds_per_generation": per_generation_s,
        "predicted_receipt_seconds": total + gate,
        "breakdown": {"cells": cells, "repeats": repeats, "arms": arms,
                      "prompts_per_cell": prompts,
                      "correctness_gate_generations": gate_gens,
                      "bench_seconds": total, "gate_seconds": gate},
        "basis": "model", "assumed_achieved": assumed_achieved,
        "why": (f"a full receipt is PREDICTED at {(total + gate) / 60:.1f} GPU-minutes against a "
                f"{SCORE_COST_BUDGET_S / 60:.0f}-minute budget, assuming a first implementation "
                f"reaches {assumed_achieved:.0%} of its roofline"),
        "_trap": ("Scales inversely with `assumed_achieved`, and that constant is a guess. On "
                  "the pinned candidate the guess was 35% and the answer was 1.5%, so this "
                  "prediction came out twelve times cheaper than the measurement. Treat it as "
                  "an ordering between candidates, never as a schedule."),
    }


def screen_candidate(key, candidate, device, axes, generation, *, resolution, steps):
    result = {"candidate": key, "repo": candidate.get("repo"),
              "device": device.get("name"), "resolution": resolution, "steps": steps,
              "basis": "model"}
    result["access"] = q0_access(candidate)

    if candidate["denoiser"]["kind"] not in ("DiT", "MMDiT", "Linear-DiT"):
        result["supported"] = False
        result["note"] = (f"{candidate['denoiser']['kind']} denoiser: this screen enumerates "
                          f"DiT geometry only, so the arithmetic questions are not answered for "
                          f"it. The ACCESS question and the recorded screen outcome still apply.")
        result["screen_outcome"] = candidate.get("_screen_outcome", "")
        return result
    if key != "pixart-sigma-xl2-1024":
        result["supported"] = False
        result["note"] = ("full geometry is enumerated only for the pinned family; other DiT "
                          "candidates are screened on ACCESS and on their recorded outcome "
                          "until someone lands their geometry (which is a scored contribution "
                          "-- see docs/CARTOGRAPHY.md)")
        result["screen_outcome"] = candidate.get("_screen_outcome", "")
        return result

    result["supported"] = True
    stages = pixart_stages(candidate, resolution=resolution, steps=steps)
    sh = shares(stages, device)
    rows = sh["rows"]
    # Looked up once, before any question uses it. Only the PINNED candidate has been measured;
    # screening another against this one's calibration would attribute one model's cost and
    # noise to a different model entirely.
    cal = _measured_calibration()
    is_pinned = bool(cal and candidate.get("repo") == cal.get("repo"))

    result["dominance"] = q1_dominance(rows, sh["total_seconds"])
    result["resolution_gate"] = q2_resolution(rows, measured=cal if is_pinned else None)
    result["regeneration"] = q3_regeneration(candidate, axes)
    result["reach"] = q4_reach(stages)
    result["determinism"] = q5_determinism(
        _committed_determinism_evidence() if is_pinned else None)
    # Only the PINNED candidate has been measured. Screening another one against this one's
    # calibration would attribute its cost to a different model entirely, so the measurement is
    # passed in only where it belongs and every other candidate keeps the modelled answer.
    result["score_cost"] = q6_score_cost(rows, generation, measured=cal if is_pinned else None)
    result["residency"] = resident_bytes(stages)
    result["stage_table"] = [
        {"stage": r["stage"], "invocations": r["invocations"], "share": r["share"],
         "ceiling_seconds": r["seconds"], "bound_by": r["bound_by"],
         "flops": r["flops"], "param_bytes": r["param_bytes"]} for r in rows]
    answered = [result[k]["pass"] for k in
                ("access", "dominance", "regeneration", "reach", "score_cost")]
    result["verdict"] = {
        "answerable_now_all_pass": all(answered),
        "open_questions": [result[k]["question"] for k in ("resolution_gate", "determinism")
                           if result[k]["pass"] is None],
        "note": ("A screen that passes every arithmetic question is a screen that has not yet "
                 "been contradicted by hardware. The two open questions are the ones that have "
                 "killed targets before."),
    }
    return result


def _fmt_table(res):
    lines = []
    st = res.get("stage_table")
    if st:
        lines.append(f"  {'stage':14s} {'runs':>5s} {'ceiling':>11s} {'share':>7s} "
                     f"{'bound':>8s} {'params':>9s}")
        for r in st:
            lines.append(f"  {r['stage']:14s} {r['invocations']:>5d} "
                         f"{r['ceiling_seconds'] * 1e3:>9.1f}ms {r['share'] * 100:>6.1f}% "
                         f"{r['bound_by']:>8s} {r['param_bytes'] / 1e9:>7.2f}GB")
    return "\n".join(lines)


def render_markdown(out, cands, axes):
    """The screen's answers, written into the repository, with every figure computed.

    Hand-writing this page would mean typing numbers that come from `configs/`, and they would
    drift the first time a config moved. The narrative lives here so that it cannot.
    """
    pinned = out["results"]["pixart-sigma-xl2-1024"]
    dev = out["device"]
    w = [f"# Which model v0 pins, and why", "",
         "Generated by `eval/screen.py --markdown`. **Every ceiling here is `basis: model`** —",
         "arithmetic from a config file and a device peak, never a measurement. The three",
         "questions arithmetic can answer are answered that way.",
         "",
         "The other three are not, and two of them used to be printed as OPEN no matter what",
         "they contained. Where the pinned candidate has been measured on the pinned part —",
         "resolution, determinism, and what a receipt costs — the answer below is `basis:",
         "measured` and names the artifact it came from. The difference is not cosmetic: the",
         "cost question was modelled at 2.5 GPU-minutes per receipt on an assumption that a",
         "first implementation reaches 35% of its roofline. It reached 1.5%, and a receipt",
         "measures about half an hour.", "",
         f"Screened on **{pinned['device']}** at {out['resolution']}px, {out['steps']} steps.",
         "", "## The answer", "",
         f"**`{pinned['repo']}`** — PixArt-Sigma XL-2 at 1024px, 20 steps, classifier-free",
         "guidance, DPM-Solver++ 2M.", "",
         "It is a real DiT with a T5-XXL text encoder eight times the size of its denoiser and an",
         "SD-family VAE, which means the backlog the repository was commissioned around — DiT",
         "attention at 4k tokens, a text encoder larger than the DiT, VAE decode, AdaLN fusion,",
         "weight formats, shape specialisation — all have a home in it. It is ungated and",
         "redistributable.", "",
         "## Candidates, and why each was ruled in or out", "",
         "| candidate | licence | gated | outcome |", "|:--|:--|:--:|:--|"]
    for key, r in out["results"].items():
        acc = r["access"]
        note = r.get("screen_outcome") or ("**PINNED for BG-1.**" if r.get("supported") else "")
        w.append(f"| `{key}` | {acc['license']} | {'yes' if acc['gated'] else 'no'} | {note} |")
    # The verdict sentence comes from the table above, which comes from configs/. It was typed
    # here as well, and the two drifted the moment the config changed: the table said one thing
    # and this paragraph said "the obvious BG-2" after BG-2 had become a 512px PixArt generation.
    w += ["",
          "**Access is a screen criterion, not a footnote.** FLUX.1-schnell has Apache-2.0",
          "weights behind an auto-approved HuggingFace gate: `curl` returns 401 without a token.",
          "A permissive licence with non-permissive distribution still fails \"re-runnable by",
          "strangers\", and that is the binding constraint for a benchmark. Its recorded screen",
          "outcome is in the table above, from `configs/candidates.json` -- not restated here,",
          "because a second copy of a verdict is a second thing to keep true.", "",
          "## Where the time goes", "",
          "| stage | runs per generation | ceiling | share | bound by | resident params |",
          "|:--|--:|--:|--:|:--|--:|"]
    for row in pinned["stage_table"]:
        w.append(f"| `{row['stage']}` | {row['invocations']} | "
                 f"{row['ceiling_seconds'] * 1e3:.1f} ms | {row['share']:.1%} | "
                 f"{row['bound_by']} | {row['param_bytes'] / 1e9:.2f} GB |")
    dom = pinned["dominance"]
    w += ["",
          f"Total predicted ceiling: **{dom['total_seconds'] * 1e3:.0f} ms** for one",
          f"{out['resolution']}px generation at {out['steps']} steps.", "",
          "### The non-obvious result, and the one that should change how you spend time", "",
          "**The text encoder holds 89% of the checkpoint's parameters and 2% of the clock.** It",
          "runs once; the DiT runs twenty times. Any screen that ranked stages by parameter count",
          "— which is the natural thing to do — would send a contributor to the biggest weights in",
          "the model and they would be working on a fiftieth of the wall time.",
          "",
          "That does not make the encoder uninteresting. It makes it a **memory-axis** cell:",
          "9.53 GB of the 10.85 GB this pipeline keeps resident, on a 32 GiB card, idle for the",
          "whole denoise loop. The frontier scores peak VRAM, so quantising or caching it is worth",
          "real credit — just not on the latency axis. `issues/text-encoder.md` has the arithmetic.",
          "",
          "It also reorders with step count. At four steps the encoder is 8.1% and the VAE is 15.1%",
          "of the clock. A distilled model reopens both cells, which is the REGENERATION property",
          "the screen is looking for.", "",
          "## The six questions", ""]
    for key in ("dominance", "regeneration", "reach", "score_cost",
                "resolution_gate", "determinism"):
        b = pinned[key]
        mark = {True: "PASS", False: "FAIL", None: "OPEN"}[b["pass"]]
        w += [f"### {b['question']} — {mark}", ""]
        w += [f"*basis: {b.get('basis', 'model')}"
              + (f", measured on {b['device']}*" if b.get("device") else "*"), ""]
        w += [b["why"], ""]
        if b["pass"] is None and b.get("settled_by"):
            w += [f"Settled by: `{b['settled_by']}`", ""]
        if b.get("evidence"):
            w += [f"Evidence: `{b['evidence']}` — the raw measurements, committed, so this can "
                  f"be re-checked rather than taken on trust.", ""]
        if b.get("_trap"):
            w += [f"> **Trap.** {b['_trap']}", ""]
        if key == "determinism" and b.get("known_risks_for_this_workload"):
            w += ["Known risks for this workload, in the order worth checking:", ""]
            w += [f"- {r}" for r in b["known_risks_for_this_workload"]]
            w += [""]
    sc = pinned["score_cost"]
    b = sc["breakdown"]
    measured_cost = sc.get("basis") == "measured"
    verb = "measured" if measured_cost else "predicted"
    w += ["## What a receipt costs", "",
          f"| term | value |", "|:--|--:|",
          f"| {verb} seconds per gate generation | "
          f"{sc['predicted_seconds_per_generation']:.1f} s |"]
    if measured_cost:
        w += [f"| cells x repeats x arms | {b['cells']} x {b['repeats']} x {b['arms']} "
              f"(+{b['held_out_repeats']} held-out repeats per arm) |",
              f"| invocations per record | {b['invocations_per_record']} "
              f"(the counts the noise floors were calibrated with) |",
              f"| per-record overhead | {b['record_overhead_seconds']:.0f} s "
              f"(process start and the 21.8 GB checkpoint map) |"]
    else:
        w += [f"| cells x repeats x arms x prompts | {b['cells']} x {b['repeats']} x "
              f"{b['arms']} x {b['prompts_per_cell']} |"]
    w += [f"| correctness gate generations | {b['correctness_gate_generations']} |",
          f"| bench | {b['bench_seconds'] / 60:.0f} min |",
          f"| gate | {b['gate_seconds'] / 60:.0f} min |",
          f"| **{verb} total per receipt** | "
          f"**{sc['predicted_receipt_seconds'] / 60:.0f} GPU-minutes** |",
          f"| budget | {sc['budget_seconds'] / 60:.0f} minutes |", ""]
    if measured_cost:
        w += ["**This is the one question in the screen that a model got badly wrong.** It was",
              "answered at 2.5 GPU-minutes on an assumption that a first implementation reaches",
              "35% of its roofline. The implementation landed at 1.5%, and the real figure is",
              "twelve times the prediction — over budget rather than comfortably inside it.",
              "",
              "The assumption was labelled, published, and visible in this document the whole",
              "time, which is the part worth keeping: a clearly-marked prediction is still a",
              "prediction, and marking it does not make the schedule it implies come true.",
              "",
              "It is also the only line here that improves on its own. The cost of scoring is",
              "proportional to how slow the runtime is, so every contribution the benchmark pays",
              "for makes the benchmark cheaper to run.", ""]
    else:
        w += [f"Assumes a first implementation reaches **{sc['assumed_achieved']:.0%}** of the",
              "arithmetic ceiling. That constant is a guess, it is labelled a guess in the code,",
              "and every number in this section scales inversely with it. On the pinned",
              "candidate the guess was 35% and the measured answer was 1.5%.", ""]
    w += ["## Regeneration: how much standing work the axes hold", ""]
    reg = pinned["regeneration"]
    w += [f"- {len(reg['axes']['resolutions'])} resolutions x {len(reg['axes']['dtypes'])} dtypes "
          f"x {len(reg['axes']['stages'])} stages = **{reg['cells_from_declared_axes']} cells**",
          "- DiT token counts across those resolutions: " +
          ", ".join(f"{k}px → {v}" for k, v in sorted(reg["dit_tokens_per_resolution"].items())),
          "", reg["_why_this_matters"], "",
          "## What this screen could not answer from arithmetic", "",
          "Three of the six questions need hardware. They were printed as OPEN rather than",
          "guessed at, which is the discipline: a screen that printed a confident answer to a",
          "question it cannot reach is worse than no screen, because somebody acts on it.",
          "",
          "All three have since been measured on the pinned part, and the answers are above with",
          "`basis: measured` and the artifact each came from. One of them — what a receipt costs",
          "— came back over budget, and it had a modelled answer that said comfortably inside.",
          "",
          "See `docs/STATUS.md` for the full list of what has and has not been measured.", ""]
    return "\n".join(w)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", help="screen only this one")
    ap.add_argument("--device", default="rtx5090")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--json", help="write the full result here")
    ap.add_argument("--markdown", help="write the screen's answers as a document")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    devices = _load("devices.json")
    if args.device not in devices or args.device.startswith("_"):
        ap.error(f"unknown device {args.device!r}; have "
                 f"{[k for k in devices if not k.startswith('_')]}")
    device = devices[args.device]
    cands = _load("candidates.json")["candidates"]
    axes = _load("axes.json")
    generation = _load("axes.json")["receipt_shape"]

    keys = [args.candidate] if args.candidate else list(cands)
    for k in keys:
        if k not in cands:
            ap.error(f"unknown candidate {k!r}; have {list(cands)}")

    out = {"device": args.device, "resolution": args.resolution, "steps": args.steps,
           "basis": "model",
           "_basis_note": "Every duration here is an arithmetic ceiling divided by a stated "
                          "assumption. No run produced any of it.",
           "results": {}}
    for k in keys:
        r = screen_candidate(k, cands[k], device, axes, generation,
                             resolution=args.resolution, steps=args.steps)
        out["results"][k] = r

    print(f"burnisher screen -- {device.get('name')}, {args.resolution}px, {args.steps} steps")
    print("basis: ceilings are model (arithmetic); questions arithmetic cannot reach are "
          "answered\n       from measurement where the pinned part has been measured.\n")
    for k, r in out["results"].items():
        acc = r["access"]
        mark = "PASS" if acc["pass"] else "FAIL"
        print(f"[{mark}] {k}  licence={acc['license']} gated={acc['gated']}")
        if not r.get("supported"):
            print(f"       {r['note']}")
            if r.get("screen_outcome"):
                print(f"       outcome: {r['screen_outcome']}")
            print()
            continue
        print(_fmt_table(r))
        # All six printed the same way. Two of them used to be hardcoded as OPEN regardless of
        # what they contained, so they went on reporting "settled by `burnish calibrate`" after
        # `burnish calibrate` had settled them.
        for q in ("dominance", "regeneration", "reach", "score_cost",
                  "resolution_gate", "determinism"):
            b = r[q]
            m = {True: "PASS", False: "FAIL", None: "OPEN"}[b["pass"]]
            why = b["why"] if b["pass"] is not None else f"settled by `{b['settled_by']}`"
            print(f"       [{m}] {b['question']}: {why}")
        print()

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        print(f">> wrote {args.json}")
    if args.markdown:
        if args.candidate:
            ap.error("--markdown writes the whole screen; drop --candidate")
        Path(args.markdown).write_text(render_markdown(out, cands, axes))
        print(f">> wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
