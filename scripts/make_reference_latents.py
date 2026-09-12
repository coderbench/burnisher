#!/usr/bin/env python3
"""Produce the pinned reference latents: the oracle everything else is checked against.

    scripts/make_reference_latents.py --weights /path/to/checkpoint --noise noise.npy --write

**This must never be run with this repository's own runtime.** The reference is the thing a
candidate is compared against; producing it with the candidate would make the candidate its own
oracle, and it would pass. This script uses the reference implementation -- diffusers -- at a
pinned version, and records that version beside the latents.

**The starting noise is an INPUT, not something either side generates.** `burnisher noise` writes
it and both sides read it. Two RNG implementations agreeing bit for bit is not something to
depend on, and if they disagree the latents diverge from step zero and the gate measures the
random number generator instead of the runtime.

**The token ids are an input too**, from `scripts/tokenize_prompts.py`, so the tokenizer is not a
second place the two can differ.

What is left inside the comparison is therefore exactly what should be: the encoder, the
denoiser, the scheduler and the arithmetic.

Memory: the T5-XXL encoder is 19 GB in fp32 and this is expected to run on a machine with less
RAM than that. It is loaded with `low_cpu_mem_usage`, used, and freed before the denoiser is
loaded -- which is also why the two stages are separate passes rather than a pipeline call.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def versions():
    import torch, diffusers, transformers
    return {"torch": torch.__version__, "diffusers": diffusers.__version__,
            "transformers": transformers.__version__}


def encode_prompts(weights, ids_doc, prompt_ids, dtype, device="cpu"):
    """T5 hidden states for each (negative, positive) pair, with the attention mask."""
    import torch
    from transformers import T5EncoderModel

    print(">> loading the text encoder (19 GB fp32, mapped)", flush=True)
    t0 = time.time()
    enc = T5EncoderModel.from_pretrained(str(Path(weights) / "text_encoder"),
                                         torch_dtype=dtype, low_cpu_mem_usage=True)
    enc.eval().to(device)
    print(f"   loaded in {time.time() - t0:.0f}s", flush=True)

    out = {}
    neg = torch.tensor([ids_doc["negative"]["ids"]], dtype=torch.long, device=device)
    for pid in prompt_ids:
        pos = torch.tensor([ids_doc["prompts"][pid]["ids"]], dtype=torch.long, device=device)
        ids = torch.cat([neg, pos], dim=0)
        # The mask is derived from the ids, never passed alongside them.
        mask = (ids != ids_doc["pad_id"]).long()
        t0 = time.time()
        with torch.no_grad():
            h = enc(input_ids=ids, attention_mask=mask).last_hidden_state
        out[pid] = (h.detach().cpu().clone(), mask.detach().cpu().clone())
        print(f"   {pid:16s} encoded in {time.time() - t0:.0f}s  "
              f"{tuple(h.shape)}  real tokens {int(mask[1].sum())}", flush=True)

    del enc
    gc.collect()
    if device != "cpu":
        # Freed before the denoiser is loaded. 19 GB of encoder and 2.4 GB of denoiser both
        # resident is avoidable and, at 32 GB, eventually not survivable at higher resolutions.
        torch.cuda.empty_cache()
    return out


def denoise(weights, embeds, noise, steps, guidance, dtype, out_dir=None, device="cpu"):
    """The reference denoise loop: the reference transformer and the reference scheduler."""
    import torch
    from diffusers import Transformer2DModel, DPMSolverMultistepScheduler

    print(">> loading the transformer", flush=True)
    model = Transformer2DModel.from_pretrained(str(Path(weights) / "transformer"),
                                               torch_dtype=dtype, low_cpu_mem_usage=True)
    model.eval().to(device)
    sched_cfg = json.loads((ROOT / "configs" / "candidates.json").read_text())
    sched_cfg = sched_cfg["candidates"]["pixart-sigma-xl2-1024"]["scheduler"]
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=sched_cfg["num_train_timesteps"],
        beta_start=0.0001, beta_end=0.02, beta_schedule="linear",
        solver_order=sched_cfg["solver_order"],
        algorithm_type=sched_cfg["algorithm_type"],
        solver_type=sched_cfg["solver_type"],
        prediction_type=sched_cfg["prediction_type"],
        timestep_spacing="linspace", lower_order_final=True)

    out = {}
    for pid, (h, mask) in embeds.items():
        scheduler.set_timesteps(steps)
        latent = torch.from_numpy(noise.copy()).to(dtype).to(device)
        t0 = time.time()
        for i, t in enumerate(scheduler.timesteps):
            batched = torch.cat([latent, latent], dim=0)
            with torch.no_grad():
                pred = model(batched,
                             encoder_hidden_states=h.to(dtype).to(device),
                             encoder_attention_mask=mask.to(device),
                             timestep=t.to(device).expand(2),
                             added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                             return_dict=False)[0]
            # The model predicts 2*in_channels: epsilon and a learned variance. The sampler is
            # epsilon-only, so the variance half is discarded -- computed and thrown away, which
            # is what the reference does and therefore what the runtime must do.
            eps = pred.chunk(2, dim=1)[0]
            uncond, cond = eps.chunk(2, dim=0)
            eps = uncond + guidance * (cond - uncond)
            latent = scheduler.step(eps, t.to(device), latent, return_dict=False)[0]
            if i == 0 or (i + 1) % 5 == 0:
                print(f"   {pid:16s} step {i + 1}/{steps}  "
                      f"{(time.time() - t0) / (i + 1):.1f}s/step", flush=True)
        out[pid] = latent.float().cpu().numpy()
        if out_dir is not None:
            # Written as each one finishes, not at the end. A five-hour job that loses everything
            # to an interruption is a five-hour job nobody runs twice.
            import numpy as np
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / f"{pid}.npy", out[pid])
            print(f"   {pid:16s} saved", flush=True)
        print(f"   {pid:16s} done in {time.time() - t0:.0f}s", flush=True)
    del model
    gc.collect()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--noise", required=True, help="from `burnisher noise`")
    ap.add_argument("--generation", default="BG-1")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--guidance", type=float,
                    help="defaults to the generation's PINNED guidance_scale; override only to "
                         "explore, never to produce an oracle")
    ap.add_argument("--prompts", nargs="*", help="subset, for a smoke run")
    # Accepts the repository's spelling as well as torch's. Everything else here says fp32 and
    # bf16; this one script said float32/bfloat16 and turned a muscle-memory flag into an
    # AttributeError from inside torch.
    ap.add_argument("--dtype", default="float32",
                    help="float32/fp32 or bfloat16/bf16")
    ap.add_argument("--device", default="cpu",
                    help="cuda makes this minutes instead of hours. The REFERENCE may run "
                         "wherever it likes -- it is the oracle, not the thing being timed.")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    try:
        import numpy as np
        import torch
    except ImportError:
        print("!! this script needs torch, diffusers and transformers -- the REFERENCE\n"
              "   implementation. It is deliberately not a dependency of the runtime or the\n"
              "   harness: the reference must not be produced by the thing being checked.",
              file=sys.stderr)
        return 2

    gdir = ROOT / "eval" / "cells" / args.generation
    gen = json.loads((gdir / "generation.json").read_text())
    ids_doc = json.loads((gdir / "token-ids.json").read_text())
    steps = args.steps or gen["model"]["steps"]
    if args.guidance is None:
        args.guidance = gen["model"]["guidance_scale"]
    noise = np.load(args.noise)
    prompt_ids = args.prompts or list(ids_doc["prompts"])
    # Normalised ONCE, here, because the dtype name decides the output directory as well as the
    # arithmetic -- `--dtype fp32` and `--dtype float32` must not write to two different places.
    # They did: the first wrote `reference-latents-fp32`, which the gate does not look for, so
    # the oracle was produced correctly and filed where nothing would find it.
    args.dtype = {"fp32": "float32", "bf16": "bfloat16",
                  "fp16": "float16"}.get(args.dtype, args.dtype)
    dtype = getattr(torch, args.dtype)

    print(f"reference: {json.dumps(versions())}")
    print(f"checkpoint: {gen['model']['repo']} @ {gen['model']['revision'][:12]}")
    print(f"noise: {noise.shape}, steps {steps}, guidance {args.guidance}, "
          f"dtype {args.dtype}, device {args.device}\n")

    # One directory per dtype. The gate compares at the dtype the cell is SCORED in, because a
    # bf16 run against an fp32 oracle measures the dtype rather than the implementation -- which
    # this repository found the expensive way: every stage agreed with the reference to 1e-5 and
    # the assembled bf16 pipeline came out at 1.19 relative L2, a completely different image.
    # BURNISH_REF_OUTDIR redirects a run that is EXPLORING rather than producing the oracle --
    # a step sweep, say. Without it a sweep would silently overwrite the pinned reference
    # latents, and the gate would then be comparing against whatever the last experiment left.
    import os
    override = os.environ.get("BURNISH_REF_OUTDIR")
    out_dir = (Path(override) if override else
               gdir / ("reference-latents" if args.dtype == "float32"
                       else f"reference-latents-{args.dtype}"))
    embeds = encode_prompts(args.weights, ids_doc, prompt_ids, dtype, args.device)
    latents = denoise(args.weights, embeds, noise, steps, args.guidance, dtype,
                      out_dir if args.write else None, args.device)
    manifest = {
        "_what": "The pinned reference latents: the oracle the correctness gate compares "
                 "against. Produced by the REFERENCE implementation, never by this runtime -- a "
                 "candidate that produced its own oracle would pass.",
        "generation": args.generation,
        "reference_versions": versions(),
        "checkpoint": {"repo": gen["model"]["repo"], "revision": gen["model"]["revision"],
                       "text_encoder_repo": gen["model"]["text_encoder_repo"],
                       "text_encoder_revision": gen["model"]["text_encoder_revision"]},
        "token_ids_digest": hashlib.sha256((gdir / "token-ids.json").read_bytes()).hexdigest(),
        "noise_sha256": hashlib.sha256(Path(args.noise).read_bytes()).hexdigest(),
        "steps": steps, "guidance_scale": args.guidance, "dtype": args.dtype,
        "device": args.device,
        "_device_note": "Where the REFERENCE ran. It is the oracle, not the thing being timed, "
                        "so it may run anywhere -- but fp32 on a GPU and fp32 on a CPU are not "
                        "bit-identical, and a receipt should be able to say which produced its "
                        "oracle.",
        "_pin_note": "A moved reference version, checkpoint revision, token-id set or noise "
                     "tensor is a CHANGED ORACLE and therefore a new generation, never an edit.",
        "latents": {},
    }
    for pid, arr in latents.items():
        manifest["latents"][pid] = {
            "shape": list(arr.shape),
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
            "mean": float(arr.mean()), "std": float(arr.std()),
            "absmax": float(abs(arr).max()),
        }
        print(f"  {pid:16s} mean {arr.mean():+.5f} std {arr.std():.5f} "
              f"absmax {abs(arr).max():.3f}")
    if args.write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        print(f"\n>> wrote {len(latents)} latents to {out_dir}")
    else:
        print("\n(dry run; --write to save)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
