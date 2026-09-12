#!/usr/bin/env python3
"""Compare each stage against the REFERENCE implementation, at shapes a CPU can run.

    scripts/differential_test.py --weights /path/to/checkpoint --stage vae-decode

This is the correctness gate's question — does the runtime compute what the reference computes? —
asked at a scale that does not need a GPU. The full gate runs at 1024px and twenty steps and is
hours of CPU arithmetic; a 4x4 latent through the VAE decoder, or one DiT step at 64px, is
minutes, and it exercises exactly the same code.

It is not a substitute for the gate. It cannot see anything that only appears at scale, it runs
in fp32 where the scored path is bf16, and it uses shapes no receipt is scored on. What it CAN
see is the class of defect that this repository has already found twice: an arithmetic or layout
error that is deterministic, self-consistent, and invisible to every test that does not have a
second implementation to disagree with.

Both sides read the same weights and the same input tensor. Nothing is generated twice.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# What counts as agreement, per stage, MEASURED rather than assumed.
#
# fp32 against fp32 is not bit-exact between two implementations: the reduction orders differ,
# and a deep residual stack amplifies the difference. That amplification was measured here on the
# pinned DiT at 64px, one forward pass, truncating the block stack on both sides:
#
#     layers    1        2        4        8       16       28
#     rel L2  1.3e-6   1.4e-6   3.2e-6   3.1e-5   6.6e-5   7.3e-4
#
# Per layer the two agree to fp32 epsilon. Over twenty-eight blocks the difference grows by about
# three orders of magnitude, because each block's softmax and normalisation amplify what came in.
# So a single threshold across stages would either pass a broken VAE or fail a correct DiT, and
# these are per stage with the measurement above as their basis.
TOLERANCE = {
    "vae-decode": 1e-4,      # shallow; measured at 6.9e-6
    "dit-step": 2e-3,        # 28 residual blocks; measured at 7.3e-4
    "t5-encode (real tokens only)": 2e-3,   # 24 blocks, same reason
}


def report(name, ours, theirs, tolerance=None):
    import numpy as np
    if ours.shape != theirs.shape:
        print(f"  {name}: SHAPE MISMATCH {ours.shape} vs {theirs.shape}")
        print("    That is not a tolerance question.")
        return False
    diff = ours.astype(np.float64) - theirs.astype(np.float64)
    denom = np.linalg.norm(theirs.astype(np.float64))
    rel = float(np.linalg.norm(diff) / denom) if denom else float("inf")
    mx = float(np.abs(diff).max())
    print(f"  {name}:")
    print(f"    relative L2 : {rel:.3e}")
    print(f"    max abs     : {mx:.3e}")
    print(f"    ours        : mean {ours.mean():+.6f} std {ours.std():.6f}")
    print(f"    reference   : mean {theirs.mean():+.6f} std {theirs.std():.6f}")
    tol = tolerance if tolerance is not None else TOLERANCE.get(name, 1e-4)
    ok = rel < tol
    print(f"    tolerance   : {tol:.1e}")
    print(f"    -> {'AGREE' if ok else 'DISAGREE'}")
    if not ok:
        same_moments = (abs(ours.mean() - theirs.mean()) < 1e-4 * max(1.0, abs(theirs.mean()))
                        and abs(ours.std() - theirs.std()) < 1e-4 * max(1.0, theirs.std()))
        if same_moments:
            print("    The moments MATCH and the values do not: that is a permutation, not an")
            print("    arithmetic error. It is how the output patch ordering was found.")
        else:
            print("    The moments differ too, so this is arithmetic rather than layout. Bisect")
            print("    it: --layers truncates the DiT stack on both sides, and --stage scheduler")
            print("    isolates the sampler from everything that has weights.")
    return ok


def run_ours(binary, args, out, impl=None, device=None):
    cmd = ([str(binary)] + args + (["--impl", impl] if impl else []) +
           (["--device", device] if device else []))
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-3000:], p.stderr[-3000:])
        raise SystemExit(f"!! runtime exited {p.returncode}")
    print(f"    (ours: {time.time() - t0:.0f}s)")
    import numpy as np
    return np.load(out)


def stage_vae(args):
    """VAE decode: same latent, both decoders."""
    import numpy as np
    import torch
    from diffusers import AutoencoderKL

    latent = np.load(args.input) if args.input else None
    if latent is None:
        rng = np.random.default_rng(7)
        h = args.resolution // 8
        latent = rng.standard_normal((1, 4, h, h)).astype(np.float32)
        np.save("/tmp/diff_latent.npy", latent)

    print(">> reference")
    vae = AutoencoderKL.from_pretrained(str(Path(args.weights) / "vae"), torch_dtype=torch.float32)
    vae.eval()
    t0 = time.time()
    with torch.no_grad():
        ref = vae.decode(torch.from_numpy(latent), return_dict=False)[0].numpy()
    print(f"    ({time.time() - t0:.0f}s)")

    argv = ["decode", "--weights", args.weights, "--latent", "/tmp/diff_latent.npy",
            "--out", "/tmp/diff_pixels.npy", "--dtype", args.dtype]
    if args.against != "reference":
        print(f">> ours, impl={args.against}")
        ref = run_ours(args.binary, argv, "/tmp/diff_pixels.npy", args.against,
                       args.against_device)
    print(f">> ours, impl={args.impl}")
    ours = run_ours(args.binary, argv, "/tmp/diff_pixels.npy", args.impl, args.device)
    return report("vae-decode", ours, ref, tolerance=args.tolerance)


def stage_dit(args):
    """One denoiser forward pass: same latent, same caption embedding, same mask, same timestep.

    The caption embedding is fabricated rather than produced by the text encoder, on purpose. If
    it came from T5 then a disagreement here could be the encoder's and this test would not say
    which. One stage at a time is the whole point.
    """
    import numpy as np
    import torch
    from diffusers import Transformer2DModel

    rng = np.random.default_rng(11)
    h = args.resolution // 8
    cap_len = args.caption_len
    latent = rng.standard_normal((2, 4, h, h)).astype(np.float32)
    caption = (rng.standard_normal((2, cap_len, 4096)) * 0.2).astype(np.float32)
    # Rows of DIFFERENT lengths, which is the case a shared mask gets wrong and the case
    # classifier-free guidance always produces.
    mask = np.zeros((2, cap_len), dtype=np.float32)
    mask[0, : max(1, cap_len // 4)] = 1.0
    mask[1, : max(1, cap_len // 2)] = 1.0
    np.save("/tmp/diff_dit_latent.npy", latent)
    np.save("/tmp/diff_dit_caption.npy", caption)
    np.save("/tmp/diff_dit_mask.npy", mask)

    print(">> reference")
    model = Transformer2DModel.from_pretrained(str(Path(args.weights) / "transformer"),
                                               torch_dtype=torch.float32)
    model.eval()
    if args.layers:
        # Same truncation on both sides, so a disagreement can be bisected by depth.
        model.transformer_blocks = model.transformer_blocks[: args.layers]
    t0 = time.time()
    with torch.no_grad():
        ref = model(torch.from_numpy(latent),
                    encoder_hidden_states=torch.from_numpy(caption),
                    encoder_attention_mask=torch.from_numpy(mask).long(),
                    timestep=torch.tensor([args.timestep, args.timestep]),
                    added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                    return_dict=False)[0].numpy()
    print(f"    ({time.time() - t0:.0f}s)")

    argv = (["dit-step", "--weights", args.weights,
             "--latent", "/tmp/diff_dit_latent.npy",
             "--caption", "/tmp/diff_dit_caption.npy",
             "--mask", "/tmp/diff_dit_mask.npy",
             "--timestep", str(args.timestep),
             "--out", "/tmp/diff_dit_out.npy", "--dtype", args.dtype] +
            (["--layers", str(args.layers)] if args.layers else []))
    if args.against != "reference":
        print(f">> ours, impl={args.against}")
        ref = run_ours(args.binary, argv, "/tmp/diff_dit_out.npy", args.against,
                       args.against_device)
    print(f">> ours, impl={args.impl}")
    ours = run_ours(args.binary, argv, "/tmp/diff_dit_out.npy", args.impl, args.device)
    return report("dit-step", ours, ref, tolerance=args.tolerance)


def stage_t5(args):
    """The text encoder: same token ids, same derived mask."""
    import numpy as np
    import torch
    from transformers import T5EncoderModel

    gdir = ROOT / "eval" / "cells" / "BG-1"
    ids_doc = json.loads((gdir / "token-ids.json").read_text())
    n = args.caption_len
    # Truncated to keep a CPU run tractable; the graph is the same at any length, and the
    # padding is what the mask has to handle.
    neg = ids_doc["negative"]["ids"][:n]
    pos = ids_doc["prompts"]["short-caption"]["ids"][:n]
    Path("/tmp/diff_ids.txt").write_text(
        " ".join(map(str, neg)) + "\n" + " ".join(map(str, pos)) + "\n")
    ids = torch.tensor([neg, pos], dtype=torch.long)
    mask = (ids != ids_doc["pad_id"]).long()

    print(">> reference")
    enc = T5EncoderModel.from_pretrained(str(Path(args.weights) / "text_encoder"),
                                         torch_dtype=torch.float32, low_cpu_mem_usage=True)
    enc.eval()
    t0 = time.time()
    with torch.no_grad():
        ref = enc(input_ids=ids, attention_mask=mask).last_hidden_state.numpy()
    print(f"    ({time.time() - t0:.0f}s)")
    del enc

    argv = ["encode", "--weights", args.weights, "--token-ids", "/tmp/diff_ids.txt",
            "--out", "/tmp/diff_t5_out.npy", "--dtype", args.dtype]
    if args.against != "reference":
        print(f">> ours, impl={args.against}")
        ref = run_ours(args.binary, argv, "/tmp/diff_t5_out.npy", args.against,
                       args.against_device)
    print(f">> ours, impl={args.impl}")
    ours = run_ours(args.binary, argv, "/tmp/diff_t5_out.npy", args.impl, args.device)
    # The encoder's output at PADDED positions is meaningless on both sides -- it is masked out
    # downstream. Comparing it would compare two different pieces of garbage, so the comparison
    # is restricted to real tokens, which is also what the model's consumer sees.
    keep = mask.numpy().astype(bool)
    return report("t5-encode (real tokens only)", ours[keep], ref[keep],
                  tolerance=args.tolerance)


def stage_scheduler(args):
    """The sampler alone: no weights, no model, identical synthetic inputs on both sides.

    Cheap and worth doing first. A scheduler that differs by one index convention produces a
    plausible image from the same seed and fails the latent comparison with no clue as to why,
    and this isolates it from everything that has weights.
    """
    import numpy as np
    import torch
    from diffusers import DPMSolverMultistepScheduler

    steps, n = args.steps, 16
    ours_json = subprocess.run(
        [str(args.binary), "schedule", "--steps", str(steps), "--size", str(n),
         "--dtype", args.dtype, "--out", "/tmp/diff_sched.npy"],
        capture_output=True, text=True, check=True)
    ours = np.load("/tmp/diff_sched.npy")
    our_ts = json.loads(ours_json.stdout.split("BURNISH_JSON:", 1)[1])["timesteps"]

    cfg = json.loads((ROOT / "configs" / "candidates.json").read_text())
    cfg = cfg["candidates"]["pixart-sigma-xl2-1024"]["scheduler"]
    sched = DPMSolverMultistepScheduler(
        num_train_timesteps=cfg["num_train_timesteps"], beta_start=0.0001, beta_end=0.02,
        beta_schedule="linear", solver_order=cfg["solver_order"],
        algorithm_type=cfg["algorithm_type"], solver_type=cfg["solver_type"],
        prediction_type=cfg["prediction_type"], timestep_spacing="linspace",
        lower_order_final=True)
    sched.set_timesteps(steps)
    ref_ts = [int(t) for t in sched.timesteps]
    if ref_ts != our_ts:
        print(f"  timesteps DISAGREE\n    ours      {our_ts}\n    reference {ref_ts}")
        print("    Off by one here is the likeliest way to match the reference ALMOST, which is")
        print("    the least useful outcome available.")
        return False
    print(f"  timesteps agree ({steps} values, {ref_ts[0]} down to {ref_ts[-1]})")

    # The reference sampler at the SAME dtype. At high sigma the x0 prediction is two orders of
    # magnitude larger than the latent, so where a sampler rounds matters enormously -- which is
    # exactly what this comparison is for.
    td = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    sample = torch.tensor([np.sin(i * 0.7) for i in range(n)], dtype=td)
    traj = [sample.float().numpy().copy()]
    for i, t in enumerate(sched.timesteps):
        eps = torch.tensor([np.cos(j * 0.3 + i * 0.11) for j in range(n)], dtype=td)
        sample = sched.step(eps, t, sample, return_dict=False)[0]
        traj.append(sample.float().numpy().copy())
    ref = np.stack(traj)
    tol = args.tolerance if args.tolerance else (1e-5 if args.dtype == "fp32" else 5e-2)
    return report("scheduler", ours, ref, tolerance=tol)


STAGES = {"scheduler": stage_scheduler, "vae-decode": stage_vae, "dit-step": stage_dit,
          "t5-encode": stage_t5}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="", help="checkpoint directory (not needed for "
                                                  "--stage scheduler)")
    ap.add_argument("--binary", default=str(ROOT / "build" / "burnisher"))
    ap.add_argument("--stage", default="vae-decode", choices=sorted(STAGES))
    ap.add_argument("--resolution", type=int, default=32)
    ap.add_argument("--input", help="input tensor .npy; generated if omitted")
    ap.add_argument("--caption-len", type=int, default=16)
    ap.add_argument("--timestep", type=float, default=500.0)
    ap.add_argument("--layers", type=int, help="truncate the DiT block stack on BOTH sides")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--impl", default="stock", help="which registered implementation to test")
    ap.add_argument("--against", default="reference",
                    help="'reference' compares against diffusers; an IMPL NAME compares two of "
                         "this runtime's own implementations against each other, which is how a "
                         "CUDA kernel is checked against the CPU oracle")
    ap.add_argument("--device", default=None,
                    help="where the tested implementation runs (cpu|cuda)")
    ap.add_argument("--against-device", default=None,
                    help="where the comparison implementation runs")
    ap.add_argument("--tolerance", type=float,
                    help="override the per-stage default")
    args = ap.parse_args()

    if args.against == "reference":
        try:
            import torch  # noqa: F401
        except ImportError:
            pass
    try:
        import torch  # noqa: F401
    except ImportError:
        print("!! needs torch and diffusers -- the REFERENCE implementation, deliberately not a "
              "dependency of the runtime or the harness.", file=sys.stderr)
        return 2

    against = "diffusers (the reference)" if args.against == "reference" else \
              f"impl '{args.against}' (this runtime)"
    print(f"differential test: {args.stage} at {args.resolution}px, "
          f"impl '{args.impl}' vs {against}, dtype {args.dtype}\n")
    return 0 if STAGES[args.stage](args) else 1


if __name__ == "__main__":
    sys.exit(main())
