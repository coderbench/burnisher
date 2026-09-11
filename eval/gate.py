#!/usr/bin/env python3
"""The correctness gate. It runs before any timing, and a failure is a rejection, not a trade.

    burnish gate --impl fused-adaln --repeats 10 --output gate.json

Two questions, in this order, and the second one is the one that surprises people:

**1. Does this build reproduce ITSELF?** Ten replays of the same build with the same seed must
produce byte-identical latents. Not close -- identical. If they do not, no candidate can ever be
attributed a difference, because the instrument cannot tell a change from a replay, and every
number this repository would go on to print would be decorated noise. This is checked FIRST
because a build that fails it cannot even be compared against the reference meaningfully.

The engagement this harness descends from lost its most promising checkpoint to exactly this:
four unhooked control replays produced four distinct outputs, because a few ULP in the prefill
fed discrete top-k expert routing. Diffusion has its own versions -- an autotuner picking a
different algorithm per process, an atomic in a GroupNorm reduction over a 1024x1024x128
activation, a sampler drawing its noise on device in launch order.

**2. Does it match the pinned reference?** Fixed seed, frozen prompt set, latents compared
against the pinned reference implementation within a stated tolerance. Latents rather than
pixels: the VAE decode is itself one of the things under optimization, so comparing images would
fold two questions into one and let a decoder change hide a denoiser change.

The tolerance is in the generation, is justified in writing there, and is falsifiable --
`--calibrate-tolerance` measures the bf16-vs-fp32 drift directly instead of arguing about it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from burnscore import cells as C
from paths import add_argument as add_cells_root_arg, cells_root, generation_path
from runner import (GpuLock, RunnerError, device_fingerprint, parse_result,
                    require_idle_device, require_not_degenerate, run_once)

ROOT = Path(__file__).resolve().parent.parent


def _git(*a):
    try:
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True,
                              text=True, timeout=30).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def generate(binary, generation, token_ids_file, seed, impl, *, out_dir, label, weights,
             device):
    """One generation, dumping the denoised LATENT.

    Token IDS, not a prompt string. The T5 tokenizer is a SentencePiece model and the runtime
    does not carry one -- vendoring it would put a second oracle in the repository. The ids come
    from `scripts/tokenize_prompts.py` and their digest is recorded in this report, so a changed
    tokenization is a changed comparison and says so.
    """
    out = Path(out_dir) / f"{label}.npy"
    cmd = [str(binary), "generate",
           "--weights", str(weights),
           "--token-ids", str(token_ids_file),
           "--seed", str(seed),
           "--impl", impl,
           "--device", device,
           "--resolution", str(generation.model["resolution"]),
           "--steps", str(generation.model["steps"]),
           "--guidance-scale", str(generation.raw["model"]["guidance_scale"]),
           "--dump-latents", str(out)]
    code, text, _ = run_once(cmd)
    if code != 0:
        raise RunnerError(f"{label}: the runtime exited {code}\n{text[-4000:]}")
    result = parse_result(text, label)
    require_not_degenerate(result, label)
    result["latent_path"] = str(out)
    result["latent_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    return result


def compare(a_path, b_path):
    """Relative L2 and max absolute difference between two latent dumps."""
    import numpy as np
    a = np.load(a_path).astype(np.float64)
    b = np.load(b_path).astype(np.float64)
    if a.shape != b.shape:
        raise RunnerError(f"latent shapes differ: {a.shape} vs {b.shape}. That is not a "
                          f"tolerance question.")
    denom = float(np.linalg.norm(a))
    return {"relative_l2": float(np.linalg.norm(a - b) / denom) if denom else float("inf"),
            "max_abs": float(np.max(np.abs(a - b))),
            "shape": list(a.shape)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--generation", default="BG-1")
    add_cells_root_arg(ap)
    ap.add_argument("--impl", default="stock")
    ap.add_argument("--weights", required=True, help="checkpoint directory")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--repeats", type=int, default=10,
                    help="determinism replays. Byte-identical is the bar.")
    ap.add_argument("--prompts", help="frozen prompt set (default: the generation's)")
    ap.add_argument("--reference", help="directory of pinned reference latents")
    ap.add_argument("--determinism-only", action="store_true")
    ap.add_argument("--calibrate-tolerance", action="store_true",
                    help="measure the bf16-vs-fp32 drift instead of asserting a threshold")
    ap.add_argument("--work-dir", default="/tmp/burnish-gate")
    ap.add_argument("--output")
    args = ap.parse_args()

    generation = C.load(generation_path(args.generation, args.cells_root))
    prompts_path = Path(args.prompts) if args.prompts else (
        cells_root(args.cells_root) / args.generation / "prompts.json")
    prompts = json.loads(prompts_path.read_text())
    seed = prompts["seed"]
    ids_dir = prompts_path.parent
    ids_doc_path = ids_dir / "token-ids.json"
    if not ids_doc_path.exists():
        print(f"!! {ids_doc_path} does not exist. The gate compares latents generated from "
              f"TOKEN IDS,\n   and the ids are part of the oracle. Produce them with "
              f"scripts/tokenize_prompts.py.", file=sys.stderr)
        return 2
    ids_doc = json.loads(ids_doc_path.read_text())
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    tol = generation.tolerance

    report = {
        "generation": generation.name, "impl": args.impl,
        "prompt_set": prompts["name"], "prompt_set_digest": hashlib.sha256(
            prompts_path.read_bytes()).hexdigest(), "seed": seed,
        "base_commit": _git("rev-parse", "HEAD~1"),
        "candidate_commit": _git("rev-parse", "HEAD"),
        "instrument_from": None,
        "tolerance": tol,
    }

    with GpuLock():
        require_idle_device()
        report["device"] = device_fingerprint()

        # --- 1. self-determinism, first ---
        print(f">> determinism: {args.repeats} replays of {args.impl}, byte-identical required")
        digests, first = [], None
        det_ids = ids_dir / f"token-ids-{prompts['prompts'][0]['id']}.txt"
        for i in range(args.repeats):
            r = generate(args.binary, generation, det_ids, seed, args.impl,
                         out_dir=work, label=f"det-{i}", weights=args.weights,
                         device=args.device)
            digests.append(r["latent_sha256"])
            first = first or r["latent_path"]
            print(f"   replay {i:2d}  {r['latent_sha256'][:16]}")
        unique = sorted(set(digests))
        report["determinism"] = len(unique) == 1
        report["determinism_digests"] = unique
        report["determinism_replays"] = args.repeats
        if not report["determinism"]:
            print(f"\n!! NOT DETERMINISTIC: {len(unique)} distinct outputs across "
                  f"{args.repeats} replays of the SAME build.\n"
                  f"   No candidate can be attributed a difference against this baseline, so "
                  f"nothing\n   downstream means anything. Usual causes, in the order worth "
                  f"checking:\n"
                  f"     - autotuning picking a different algorithm per process "
                  f"(CUBLAS_WORKSPACE_CONFIG,\n       and any cuDNN/cutlass heuristic cache)\n"
                  f"     - an atomic in a reduction; GroupNorm over a 1024x1024x128 activation "
                  f"is the\n       obvious candidate in this pipeline\n"
                  f"     - TF32 enabled somewhere an fp32 reference is expected\n"
                  f"     - RNG drawn on device in launch order rather than from a fixed "
                  f"sequence",
                  file=sys.stderr)
            report["correctness"] = "NOT_RUN"
            _write(args, report)
            return 1
        print(f"   all {args.repeats} replays identical\n")

        if args.determinism_only:
            report["correctness"] = "NOT_RUN"
            _write(args, report)
            return 0

        # --- 2. against the pinned reference ---
        ref_dir = Path(args.reference) if args.reference else (
            cells_root(args.cells_root) / generation.name / "reference-latents")
        if not ref_dir.is_dir():
            print(f"!! no pinned reference latents at {ref_dir}.\n"
                  f"   The reference is the oracle for everything else and it cannot be "
                  f"inferred from\n   the candidate. Produce it once, from the pinned "
                  f"reference implementation at the\n   pinned revision, and commit the "
                  f"digests: docs/CORRECTNESS.md has the procedure.",
                  file=sys.stderr)
            report["correctness"] = "NO_REFERENCE"
            _write(args, report)
            return 2

        print(f">> correctness: {len(prompts['prompts'])} frozen prompts against {ref_dir.name}")
        per_prompt, worst_l2, worst_abs = [], 0.0, 0.0
        for p in prompts["prompts"]:
            label = f"gate-{p['id']}"
            r = generate(args.binary, generation, ids_dir / f"token-ids-{p['id']}.txt", seed,
                         args.impl, out_dir=work, label=label, weights=args.weights,
                         device=args.device)
            ref = ref_dir / f"{p['id']}.npy"
            if not ref.exists():
                raise RunnerError(f"the frozen prompt set names {p['id']} and the reference "
                                  f"directory has no {ref.name}. A prompt set and a reference "
                                  f"that disagree is a gate that checks nothing.")
            d = compare(ref, r["latent_path"])
            per_prompt.append({"id": p["id"], **d})
            worst_l2 = max(worst_l2, d["relative_l2"])
            worst_abs = max(worst_abs, d["max_abs"])
            print(f"   {p['id']:14s} rel L2 {d['relative_l2']:.5f}  max abs {d['max_abs']:.5f}")

        report["per_prompt"] = per_prompt
        report["worst_relative_l2"] = worst_l2
        report["worst_max_abs"] = worst_abs

        if args.calibrate_tolerance:
            report["correctness"] = "NOT_RUN"
            report["measured_drift"] = {"relative_l2": worst_l2, "max_abs": worst_abs}
            print(f"\n>> tolerance calibration: the measured drift of this build against the "
                  f"reference is\n   rel L2 {worst_l2:.5f}, max abs {worst_abs:.5f}. A "
                  f"threshold should sit above this and\n   below anything an actual algorithm "
                  f"change produces; state the reasoning in\n   generation.json rather than "
                  f"just the number.")
            _write(args, report)
            return 0

        ok = (worst_l2 <= tol["latent_l2_relative"] and worst_abs <= tol["latent_max_abs"])
        report["correctness"] = "PASS" if ok else "FAIL"
        if ok:
            print(f"\n   PASS: worst rel L2 {worst_l2:.5f} <= {tol['latent_l2_relative']}, "
                  f"worst max abs {worst_abs:.5f} <= {tol['latent_max_abs']}")
        else:
            print(f"\n!! FAIL: worst rel L2 {worst_l2:.5f} against a tolerance of "
                  f"{tol['latent_l2_relative']},\n   worst max abs {worst_abs:.5f} against "
                  f"{tol['latent_max_abs']}.\n"
                  f"   This is a rejection, not a trade-off. Correctness precedes speed and is "
                  f"never\n   weighed against it.", file=sys.stderr)
    _write(args, report)
    return 0 if report.get("correctness") == "PASS" else 1


def _write(args, report):
    text = json.dumps(report, indent=1, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text)
        print(f">> wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    sys.exit(main())
