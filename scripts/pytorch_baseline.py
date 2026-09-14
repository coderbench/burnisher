#!/usr/bin/env python3
"""What the same image costs in PyTorch on the same card: the number this runtime has to beat.

    scripts/pytorch_baseline.py --weights /workspace/ckpt            # print
    scripts/pytorch_baseline.py --weights /workspace/ckpt --write    # and commit it
    scripts/pytorch_baseline.py --recompare                          # no GPU: after a re-anchor

**Why this exists.** The roofline says how far each cell is from an arithmetic limit nobody can
reach. It does not say whether anybody would choose this runtime over what they already have. That
question has one answer, and it is this measurement: the reference implementation -- diffusers on
PyTorch, as a user installs it -- doing the same work, at the same shapes and dtype, on the same
card. A cell faster than its PyTorch time is a cell somebody would use.

**Same work, stage by stage.** The inputs are the generation's own: its token ids (negative and
positive, 300 tokens each), its batch of two under guidance, its resolution and dtype. Each stage
is timed around exactly what the runtime's cell of the same name does, with the device synchronised
on both sides of the timer:

    t5-encode    the text encoder over both prompts
    dit-step     one transformer forward over the guided batch
    vae-decode   the VAE decoder, latent to pixels

plus the whole image, encoder to pixels, with the pinned scheduler.

**Eager, as installed.** No `torch.compile`, no TensorRT: the default path, which already uses
PyTorch's fused scaled-dot-product attention. A faster PyTorch configuration is a higher bar and a
fair one to add; it is a second row, not a replacement for this one.

**Re-anchoring moves one side of the comparison.** The PyTorch times are a measurement of PyTorch and
stay valid; the runtime's side is read from the anchor. `--recompare` rebuilds only the comparison
from the committed PyTorch stages and the current `reference.json`, with no GPU.

**This is not scored.** Nothing reads it to pay anybody. It is published beside the anchor so the
roofline's "how far from the limit" has a "how far from being useful" next to it.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def median_and_spread(fn, warmup, repeats, sync):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    return {"median_s": med, "min_s": min(times), "max_s": max(times),
            "spread_pct": 100.0 * (max(times) - min(times)) / med, "repeats": repeats}


def comparison_against(anchor_cells, stages):
    """The runtime's anchored time over PyTorch's, per cell both of them measured."""
    out = {}
    for cid, st in stages.items():
        if cid in anchor_cells and anchor_cells[cid].get("measured_seconds"):
            burnisher = anchor_cells[cid]["measured_seconds"]
            out[cid] = {"burnisher_s": burnisher, "pytorch_s": st["median_s"],
                        "burnisher_over_pytorch": burnisher / st["median_s"]}
    return out


def print_comparison(comparison, whole_image_s):
    print(f"  {'cell':22s} {'burnisher':>11s} {'pytorch':>11s} {'burnisher/pytorch':>18s}")
    for cid, c in comparison.items():
        print(f"  {cid:22s} {c['burnisher_s'] * 1e3:9.1f}ms {c['pytorch_s'] * 1e3:9.1f}ms "
              f"{c['burnisher_over_pytorch']:17.2f}x")
    print(f"  whole image in PyTorch: {whole_image_s:.2f} s")


def device_identity():
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True)
    name, uuid, driver = [s.strip() for s in out.stdout.splitlines()[0].split(",")]
    return {"name": name, "uuid": uuid, "driver": driver}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", help="checkpoint directory (required unless --recompare)")
    ap.add_argument("--generation", default="BG-1")
    ap.add_argument("--prompt", default="long-caption", help="which frozen prompt to encode")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=9)
    ap.add_argument("--write", action="store_true",
                    help="save eval/cells/<generation>/pytorch-baseline.json")
    ap.add_argument("--recompare", action="store_true",
                    help="no GPU: rebuild the comparison from the committed PyTorch stages and "
                         "the current anchor, and write it back")
    args = ap.parse_args()

    cell_dir = ROOT / "eval" / "cells" / args.generation
    if args.recompare:
        dest = cell_dir / "pytorch-baseline.json"
        doc = json.loads(dest.read_text())
        anchor = json.loads((cell_dir / "reference.json").read_text())["cells"]
        doc["comparison"] = comparison_against(anchor, doc["stages"])
        print_comparison(doc["comparison"], doc["whole_image"]["median_s"])
        dest.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print(f"\n>> rewrote the comparison in {dest}")
        return 0
    if not args.weights:
        ap.error("--weights is required to measure")

    import torch
    import diffusers
    import transformers
    from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, PixArtTransformer2DModel
    from transformers import T5EncoderModel

    gen = json.loads((cell_dir / "generation.json").read_text())
    model_cfg = gen["model"]
    ids_doc = json.loads((cell_dir / "token-ids.json").read_text())
    dtype = torch.bfloat16
    dev = "cuda"
    res = model_cfg["resolution"]
    steps = model_cfg["steps"]
    guidance = model_cfg["guidance_scale"]
    sync = torch.cuda.synchronize

    w = Path(args.weights)
    print(">> loading text encoder, transformer and VAE in bf16", flush=True)
    enc = T5EncoderModel.from_pretrained(w / "text_encoder", torch_dtype=dtype).to(dev).eval()
    dit = PixArtTransformer2DModel.from_pretrained(w / "transformer", torch_dtype=dtype).to(dev).eval()
    vae = AutoencoderKL.from_pretrained(w / "vae", torch_dtype=dtype).to(dev).eval()

    ids = torch.tensor([ids_doc["negative"]["ids"], ids_doc["prompts"][args.prompt]["ids"]],
                       dtype=torch.long, device=dev)
    mask = (ids != ids_doc["pad_id"]).long()
    lat_side = res // 8
    latent = torch.randn((1, dit.config.in_channels, lat_side, lat_side), device=dev,
                         dtype=dtype, generator=torch.Generator(dev).manual_seed(0))
    t = torch.tensor([999], device=dev).expand(2)

    stages = {}
    with torch.inference_mode():
        hidden = enc(input_ids=ids, attention_mask=mask).last_hidden_state

        def t5():
            enc(input_ids=ids, attention_mask=mask)

        def dit_step():
            dit(torch.cat([latent, latent]), encoder_hidden_states=hidden,
                encoder_attention_mask=mask, timestep=t,
                added_cond_kwargs={"resolution": None, "aspect_ratio": None}, return_dict=False)

        def vae_decode():
            vae.decode(latent / vae.config.scaling_factor, return_dict=False)

        for name, fn in (("t5-encode", t5), ("dit-step", dit_step), ("vae-decode", vae_decode)):
            torch.cuda.reset_peak_memory_stats()
            stages[f"{name}/{res}/bf16"] = median_and_spread(fn, args.warmup, args.repeats, sync)
            stages[f"{name}/{res}/bf16"]["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
            print(f"   {name:10s} {stages[f'{name}/{res}/bf16']['median_s'] * 1e3:9.2f} ms",
                  flush=True)

        s = model_cfg["scheduler"]
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=s["num_train_timesteps"], beta_start=s["beta_start"],
            beta_end=s["beta_end"], beta_schedule=s["beta_schedule"],
            solver_order=s["solver_order"], algorithm_type=s["algorithm_type"],
            solver_type=s["solver_type"], prediction_type=s["prediction_type"],
            timestep_spacing=s["timestep_spacing"], lower_order_final=s["lower_order_final"])

        def whole_image():
            h = enc(input_ids=ids, attention_mask=mask).last_hidden_state
            x = latent.clone()
            scheduler.set_timesteps(steps)
            for step_t in scheduler.timesteps:
                pred = dit(torch.cat([x, x]), encoder_hidden_states=h,
                           encoder_attention_mask=mask, timestep=step_t.to(dev).expand(2),
                           added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                           return_dict=False)[0]
                eps = pred.chunk(2, dim=1)[0]
                uncond, cond = eps.chunk(2)
                x = scheduler.step(uncond + guidance * (cond - uncond), step_t, x,
                                   return_dict=False)[0]
            vae.decode(x / vae.config.scaling_factor, return_dict=False)

        torch.cuda.reset_peak_memory_stats()
        whole = median_and_spread(whole_image, 1, 3, sync)
        whole["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
        print(f"   {'image':10s} {whole['median_s']:9.2f} s", flush=True)

    anchor = json.loads((cell_dir / "reference.json").read_text())["cells"]
    comparison = comparison_against(anchor, stages)

    doc = {
        "_what_this_is": "The same stages, at the same shapes and dtype, run by the reference "
                         "implementation (diffusers on PyTorch, eager, as installed) on a card of "
                         "the pinned class. Not scored. It answers whether the runtime is worth "
                         "using, which the roofline cannot. `burnisher_over_pytorch` above 1 means "
                         "the runtime is that many times SLOWER than PyTorch.",
        "_comparison_basis": "burnisher_s is the anchor's measured time (reference.json), taken on "
                             "a card of the same class. Cards of this class have differed by up to "
                             "9.4% per cell (second-card-check.json), which is small beside the "
                             "ratios here but is not zero.",
        "generation": args.generation,
        "prompt": args.prompt,
        "mode": "eager",
        "versions": {"torch": torch.__version__, "diffusers": diffusers.__version__,
                     "transformers": transformers.__version__},
        "device": device_identity(),
        "stages": stages,
        "whole_image": whole,
        "comparison": comparison,
    }

    print()
    print_comparison(comparison, whole["median_s"])

    if args.write:
        dest = cell_dir / "pytorch-baseline.json"
        dest.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print(f"\n>> wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
