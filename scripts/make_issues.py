#!/usr/bin/env python3
"""Generate the backlog in issues/, with each item's arithmetic attached.

Every figure is computed from `configs/` by the scorer's own geometry, or read from the measured
artifact it names. None is typed, because an issue is a pitch for a week of somebody's time.

    scripts/make_issues.py --write
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from burnscore import geometry as G
from burnscore.pipeline import pixart_stages, shares, resident_bytes
from burnscore.roofline import bound_for


def load():
    cands = json.loads((ROOT / "configs" / "candidates.json").read_text())["candidates"]
    devices = json.loads((ROOT / "configs" / "devices.json").read_text())
    axes = json.loads((ROOT / "configs" / "axes.json").read_text())
    return cands["pixart-sigma-xl2-1024"], devices, axes


def facts(cand, devices, axes):
    """Every number the issues quote, computed once."""
    dev = devices["rtx5090"]
    f = {"resolutions": {}, "device": dev["name"]}

    for res in axes["resolutions"]:
        dit = G.pixart_dit(cand["denoiser"], resolution=res, caption_len=300, batch=2)
        b = bound_for(dit, dev, cell=f"dit/{res}")
        by_name = {}
        for op in dit.ops:
            by_name.setdefault(op.kind, 0.0)
            by_name[op.kind] += op.total_flops
        attn = sum(o.total_flops for o in dit.ops if o.kind == "attention")
        mod_bytes = sum(o.total_bytes for o in dit.ops
                        if o.kind in ("elementwise", "norm"))
        launches = sum(o.count for o in dit.ops)
        vae = G.vae_decoder(cand["vae"], resolution=res)
        vb = bound_for(vae, dev, cell=f"vae/{res}")
        vae_mat = G.vae_decoder(cand["vae"], resolution=res, attn_impl="materialized")
        vae_flash = G.vae_decoder(cand["vae"], resolution=res, attn_impl="flash")
        f["resolutions"][res] = {
            "tokens": dit.shape["tokens"],
            "dit_ceiling_ms": b.ceiling_seconds * 1e3,
            "dit_flops_t": dit.flops / 1e12,
            "dit_attention_share": attn / dit.flops,
            "dit_fusion_headroom": b.decomposed_seconds / b.ceiling_seconds,
            "dit_elementwise_bytes_gb": mod_bytes / 1e9,
            "dit_traffic_gb": dit.traffic_bytes / 1e9,
            "dit_launches_per_step": launches,
            "vae_ceiling_ms": vb.ceiling_seconds * 1e3,
            "vae_flops_t": vae.flops / 1e12,
            "vae_fusion_headroom": vb.decomposed_seconds / vb.ceiling_seconds,
            # What materializing the mid-block score matrix adds over flash attention: the
            # matrix is written once and read back once, so the traffic difference is two copies.
            # (`vae` above is the geometry's default, which is already the materialized decoder --
            # subtracting it from itself printed 0.00 GB.)
            "vae_attn_score_bytes_gb": (vae_mat.traffic_bytes - vae_flash.traffic_bytes) / 2 / 1e9,
            "vae_launches": sum(o.count for o in vae.ops),
            "vae_bound_by": vb.bound_by,
            "dit_bound_by": b.bound_by,
        }

    for steps in axes["step_counts"]:
        st = pixart_stages(cand, resolution=1024, steps=steps)
        sh = shares(st, dev)
        f[f"shares_{steps}steps"] = {r["stage"]: r["share"] for r in sh["rows"]}
        f[f"total_ms_{steps}steps"] = sh["total_seconds"] * 1e3

    st = pixart_stages(cand, resolution=1024, steps=20)
    res_all = resident_bytes(st, resident_all=True)
    res_stream = resident_bytes(st, resident_all=False)
    f["resident_all_gb"] = res_all["peak_bytes"] / 1e9
    f["resident_streamed_gb"] = res_stream["peak_bytes"] / 1e9
    f["resident_per_stage_gb"] = {k: v / 1e9 for k, v in res_all["per_stage"].items()}
    # GiB, not decimal GB. The card is sold as 32 GB and `vram_bytes` is 32 * 2^30; dividing by
    # 1e9 prints "34", which is correct in SI and reads as a mistake to every person who owns one.
    f["vram_gib"] = devices["rtx5090"]["vram_bytes"]["value"] / (1 << 30)

    dit = G.pixart_dit(cand["denoiser"], resolution=1024, caption_len=300, batch=2)
    for dt in ("bf16", "fp8", "nvfp4"):
        d = G.pixart_dit(cand["denoiser"], resolution=1024, caption_len=300, batch=2,
                         wdtype=dt, adtype="bf16")
        f[f"dit_ceiling_{dt}_ms"] = bound_for(d, dev, cell="x").ceiling_seconds * 1e3
        f[f"dit_params_{dt}_gb"] = d.param_bytes / 1e9
    # From the verification's own record rather than re-derived here, and certainly not typed.
    # The figure is how many tensors the RUNTIME requires -- not how many the checkpoint holds,
    # which includes a VAE encoder this pipeline never reads.
    layout = ROOT / "configs" / "checkpoint-layout.json"
    v = json.loads(layout.read_text()).get("verified", {}) if layout.exists() else {}
    f["checkpoint_tensors_verified"] = v.get("required_by_runtime", 0)
    f["checkpoint_missing"] = v.get("missing")
    f["checkpoint_wrong_shape"] = v.get("wrong_shape")
    # Measured, if anyone has measured it. Read from the artifact rather than restated, and
    # absent from the issue entirely when the artifact is absent: a claim about what the
    # hardware does is worth nothing without a run behind it, and the whole point of this
    # generator is that a figure in an issue cannot be typed.
    lat = ROOT / "eval" / "cells" / "BG-1" / "dtype-latency.json"
    f["dtype_latency"] = json.loads(lat.read_text()) if lat.exists() else None
    ref = ROOT / "eval" / "cells" / "BG-1" / "reference.json"
    cal = json.loads(ref.read_text())["cells"] if ref.exists() else {}
    f["dit_achieved"] = (cal.get("dit-step/1024/bf16") or {}).get("achieved")
    f["calibrated"] = {c: v["achieved"] for c, v in sorted(cal.items())
                       if v.get("achieved") is not None}
    t5 = G.t5_encoder(cand["text_encoder"], seq=300, batch=2)
    f["t5_params_gb"] = t5.param_bytes / 1e9
    f["t5_ceiling_ms"] = bound_for(t5, dev, cell="t5").ceiling_seconds * 1e3
    for dt in ("fp8", "nvfp4"):
        t = G.t5_encoder(cand["text_encoder"], seq=300, batch=2, wdtype=dt, adtype="bf16")
        f[f"t5_params_{dt}_gb"] = t.param_bytes / 1e9
    tol = json.loads((ROOT / "configs" / "tolerance.json").read_text())["BG-1"]["measured"]
    f["step_divergence"] = {k: v for k, v in tol["step_divergence_bf16"].items()
                            if not k.startswith("_")}
    f["reference_self_dtype_l2"] = tol["reference_fp32_vs_reference_bf16"]["worst_relative_l2"]
    f["dit_measured_ms"] = 1e3 * ((cal.get("dit-step/1024/bf16") or {}).get("measured_seconds")
                                  or 0.0)
    return f


# Marks a body that quotes a MEASURED artifact, so its Basis line says so. Most issues quote only
# arithmetic ceilings, and the header must not claim otherwise for the ones that don't.
MEASURED_MARK = "<!--cites-measurement-->"


def _narrowing_note(f):
    """What narrower weights are worth today, if measured. The wording follows the sign."""
    d = f.get("dtype_latency")
    if not d:
        return ""
    fp32, bf16 = d["dit_step_fp32_s"], d["dit_step_bf16_s"]
    ratio = fp32 / bf16
    measured = (f"- One DiT step costs {fp32:.3f} s at fp32 and {bf16:.3f} s at bf16\n"
                f"  ({d['repeats']} paired repeats, `eval/cells/BG-1/dtype-latency.json`).")
    if ratio > 2.05:
        return MEASURED_MARK + f"""
**Measured: narrower weights pay, and by more than the bytes.**

{measured}
- That is {ratio:.2f}x where the bytes are 2x. The bf16 path also runs fused attention and
  tensor-core GEMMs that fp32 cannot, so the ratio measures kernels as much as bandwidth.
- What fp8 and NVFP4 add on top of bf16 is unmeasured, and that is what these cells would show.
"""
    if ratio > 1.05:
        verdict = f"narrower weights pay, but {ratio:.2f}x rather than the 2x the bytes allow"
    elif ratio >= 0.98:
        verdict = "halving the bytes changed nothing"
    else:
        verdict = f"halving the bytes made the step {100 * (1 / ratio - 1):.0f}% slower"
    return MEASURED_MARK + f"""
**Measured: narrower weights do not pay yet.**

{measured}
- So {verdict}.
- At {100 * f['dit_achieved']:.1f}% of its ceiling, the step is limited by how the kernels are
  written, not by bytes.
- Do fused AdaLN, CUDA graphs and attention first.
"""


ISSUES = []


def issue(slug, title, labels, closed_by=None):
    """Register one issue. A closed issue stays, saying what settled it."""
    def wrap(fn):
        ISSUES.append((slug, title, labels, fn, closed_by))
        return fn
    return wrap


def _attention_calibration_note(f):
    """How full the attention cell is, from the calibration when it exists."""
    cal = f.get("calibrated") or {}
    if not cal:
        return "\nNo cell is calibrated yet."
    return MEASURED_MARK + (
        "\nCalibrated in BG-1 (`eval/cells/BG-1/reference.json`): "
        + ", ".join(f"`{c}` at {100 * a:.1f}%" for c, a in cal.items())
        + " of its ceiling.")


@issue("dit-attention", "DiT self-attention at 4k-16k tokens", ["kernel", "dit", "cuda"])
def _(f):
    r = f["resolutions"]
    rows = "\n".join(
        f"| {res} | {v['tokens']} | {v['dit_attention_share']:.1%} | {v['dit_ceiling_ms']:.1f} ms |"
        for res, v in sorted(r.items()))
    return f"""
Self-attention is the biggest block of arithmetic in a DiT step, and its share grows with the
square of the resolution.

| resolution | image tokens | attention share of the step | step ceiling |
|--:|--:|--:|--:|
{rows}

**Now:** `cuda` is cuDNN's fused attention for bf16, and cuBLAS scores with a float softmax for
fp32 and for T5's biased attention. It is the baseline you are measured against. `cuda-tiled`,
`cuda-tile64` and `cuda-tile1024` are the first kernel, a tiled online softmax, at three tile widths.

**What counts:** a faster CUDA attention kernel under a new name. fp8
({f['dit_ceiling_fp8_ms']:.1f} ms) and NVFP4 ({f['dit_ceiling_nvfp4_ms']:.1f} ms) are separate
cells with no implementation yet, against bf16's {f['dit_ceiling_bf16_ms']:.1f} ms. Landing one is
also cartography (`docs/CARTOGRAPHY.md`).

Ceilings are arithmetic: how big the box is, not how full.{_attention_calibration_note(f)}
"""


@issue("vae-decode", "VAE decode: fusion and the mid-block attention", ["kernel", "vae", "cuda"])
def _(f):
    r = f["resolutions"]
    rows = "\n".join(
        f"| {res} | {v['vae_ceiling_ms']:.1f} ms | {v['vae_bound_by']} | "
        f"{v['vae_fusion_headroom']:.2f}x |"
        for res, v in sorted(r.items()))
    return f"""
| resolution | ceiling | bound by | fusion headroom |
|--:|--:|:--|--:|
{rows}

VAE decode is compute-bound on this card, not memory-bound. `cuda` convolves through cuDNN in
float, so a bf16 decode reads every convolution's operands into float and rounds its output, and
the decode moves a lot of intermediate data. The first kernel, one thread per output element, is
still registered as `cuda-direct`.

- **Fusion.** The headroom column is what removing intermediate traffic is worth.
- **Mid-block attention** covers {r[1024]['tokens'] * 4} positions at 1024px. Materialized, its
  score matrix is {r[1024]['vae_attn_score_bytes_gb']:.2f} GB. The host kernel `materialized` is
  kept as the naive baseline to measure that against.
- **Tiling** changes the op list, so a tiled decode is a new cell, not a faster version of this one.
"""


@issue("text-encoder", "T5-XXL: most of the checkpoint, little of the clock",
       ["memory", "quantization", "text-encoder"])
def _(f):
    s20, s4 = f["shares_20steps"], f["shares_4steps"]
    return f"""
The text encoder is {f['t5_params_gb']:.2f} GB of the {f['resident_all_gb']:.2f} GB kept on the
card, but only {s20['t5-encode']:.1%} of the time at 20 steps. It runs once; the DiT runs twenty
times.

| steps | t5-encode | dit-step | vae-decode |
|--:|--:|--:|--:|
| 20 | {s20['t5-encode']:.1%} | {s20['dit-step']:.1%} | {s20['vae-decode']:.1%} |
| 4 | {s4['t5-encode']:.1%} | {s4['dit-step']:.1%} | {s4['vae-decode']:.1%} |

So the work here is **memory**, which the frontier scores:

- **Quantization.** fp8 takes it to {f['t5_params_fp8_gb']:.2f} GB and NVFP4 to
  {f['t5_params_nvfp4_gb']:.2f} GB, freeing up to {f['t5_params_gb'] - f['t5_params_nvfp4_gb']:.1f}
  GB of a {f['vram_gib']:.0f} GiB card.
- **Caching.** The output depends only on the prompt, so an unchanged prompt need not be encoded
  again. That matters most at few steps.
"""


@issue("fused-adaln", "Fuse AdaLN modulation into its neighbours", ["kernel", "fusion", "dit"])
def _(f):
    r = f["resolutions"][1024]
    return f"""
`modulate` computes `x * (1 + scale) + shift`: almost no arithmetic, but a full pass over the
activation.

At 1024px the elementwise and norm ops move {r['dit_elementwise_bytes_gb']:.2f} GB of the step's
{r['dit_traffic_gb']:.2f} GB of data traffic
({r['dit_elementwise_bytes_gb'] / r['dit_traffic_gb']:.0%}) for a negligible share of its
arithmetic. The whole step's fusion headroom is {r['dit_fusion_headroom']:.2f}x, mostly here.

- **Fusing raises your achieved fraction without moving the ceiling**, because the ceiling already
  ignores intermediate data.
- **Start with** `ModulateArgs`: it carries `residual` and `gate`, so the gated residual can be one
  op. The DiT calls it that way in `src/models/pixart_dit.cpp`.
"""


@issue("weight-formats", "NVFP4 and MXFP4 on Blackwell", ["quantization", "cuda", "cartography"])
def _(f):
    return f"""
Blackwell's FP4 path is new, and no open diffusion runtime has tuned it.

| dtype | DiT step ceiling @1024px | DiT weights | T5 weights |
|:--|--:|--:|--:|
| bf16 | {f['dit_ceiling_bf16_ms']:.2f} ms | {f['dit_params_bf16_gb']:.2f} GB | {f['t5_params_gb']:.2f} GB |
| fp8 | {f['dit_ceiling_fp8_ms']:.2f} ms | {f['dit_params_fp8_gb']:.2f} GB | {f['t5_params_fp8_gb']:.2f} GB |
| nvfp4 | {f['dit_ceiling_nvfp4_ms']:.2f} ms | {f['dit_params_nvfp4_gb']:.2f} GB | {f['t5_params_nvfp4_gb']:.2f} GB |

- The fp8 and NVFP4 cells are declared with no implementation. Landing one with its calibration is
  **cartography** (`docs/CARTOGRAPHY.md`).
- NVFP4 costs 0.5625 bytes per element (4 bits plus a scale per 16), not 0.5.
- A 4-bit path needs its own tolerance in its own generation.
{_narrowing_note(f)}"""


@issue("step-caching", "Step and feature caching", ["algorithm", "scheduler"])
def _(f):
    share = f["shares_20steps"]["dit-step"]
    return f"""
Reusing work across denoise steps can be worth a multiple. The DiT is {share:.0%} of the time at
20 steps, so skipping 4 steps saves about {4 / 20 * share:.0%} of a generation.

**But caching computes different numbers on purpose**, so the gate is the whole problem:

- It won't fit BG-1's tolerance, and the fix is **not** a wider tolerance. It is a new generation
  with its own measured tolerance (`docs/CORRECTNESS.md`).
- Faster but further from the reference is `MOVED_ALONG_FRONTIER` and credits nothing.

To be paid, show the cache is free within a stated tolerance, not just that it is fast.
"""


@issue("weight-upload", "Every generation re-uploads the weights", ["cuda", "startup"])
def _(f):
    return f"""
`burnisher generate` uploads {f['resident_all_gb']:.1f} GB of weights every run. The whole 20-step
denoise has a ceiling of {f['shares_20steps']['dit-step'] * f['total_ms_20steps']:.0f} ms. The upload
itself has never been timed.

It isn't a scored cell (the bench loads once), but it dominates the gate's run time and anyone
actually using the runtime.

- **Keep the process alive**, with an explicit reset between runs.
- **Map the checkpoint straight to the device** instead of copying it twice.
- **Keep less on the card:** the {f['t5_params_gb']:.2f} GB text encoder sits idle during
  denoising (`issues/offload.md`).

**Measure first:** time map, convert and upload separately in `DeviceWeights::get`.
"""


@issue("sampler-precision", "The sampler rounds a large intermediate to bf16",
       ["numerics", "measured", "scheduler"])
def _(f):
    sd = f["step_divergence"]
    return MEASURED_MARK + f"""
At the first step sigma is about 157, so the sampler's x0 estimate is about 300 while the latent is
about 1. Stored in bf16, values near 300 are spaced about 2 apart. That is a precision cliff.

It shows at one step and fades after (bf16 vs the bf16 reference, relative L2, from
`configs/tolerance.json`): {sd['1']:.3f} at 1 step, {sd['2']:.3f} at 2, {sd['20']:.3f} at 20.

- **The fix is cheap:** keep the sampler's state in fp32. It is one latent, 256 kB.
- **It is not done on purpose.** It would make the runtime more accurate than the reference, so the
  gate would reject it. It belongs in a new generation with regenerated reference latents.
- **Measure first:** the same sweep with an fp32 sampler and a bf16 model.
"""


@issue("cuda-graphs", "Capture the denoise loop as a CUDA graph", ["cuda", "launch-overhead"])
def _(f):
    r = f["resolutions"]
    rows = "\n".join(
        f"| {res} | {v['dit_launches_per_step']} | {v['dit_launches_per_step'] * 20} |"
        for res, v in sorted(r.items()))
    return f"""
The denoise loop is the same graph with the same shapes twenty times, the textbook case for a
CUDA graph.

| resolution | kernel launches per step | per 20-step generation |
|--:|--:|--:|
{rows}

The launch count is exact (from `eval/burnscore/geometry.py`). What a launch costs on this card
has not been measured.

- **A small, certain win**, bigger at low resolution where steps are short.
- **Do it early:** it removes launch noise from every later measurement.
"""


@issue("offload", "Offload and streaming: bigger models do not fit",
       ["memory", "streaming", "frontier"])
def _(f):
    per = f["resident_per_stage_gb"]
    rows = "\n".join(f"| `{k}` | {v:.2f} GB |" for k, v in sorted(per.items()))
    return f"""
| stage | weights on the card (bf16) |
|:--|--:|
{rows}
| **all at once** | **{f['resident_all_gb']:.2f} GB** |
| **one stage at a time** | **{f['resident_streamed_gb']:.2f} GB** |

On a {f['vram_gib']:.0f} GiB card PixArt-Sigma fits either way. Streaming matters for the next,
bigger models, which were ruled out of v0 because they don't fit (`configs/candidates.json`).

- Scored on peak VRAM. Streaming that costs latency is a move along the frontier; streaming that
  costs nothing is a gain.
- **Trap:** peak memory is the device allocator's high-water mark on a CUDA run and host peak RSS
  on a CPU run. They are different resources; never compare one with the other.
"""


@issue("shape-specialization", "Per-resolution kernels and the held-out guard",
       ["kernel", "anti-gaming"])
def _(f):
    r = f["resolutions"]
    rows = "\n".join(f"| {res} | {v['tokens']} | {v['dit_ceiling_ms']:.1f} ms |"
                     for res, v in sorted(r.items()))
    held = json.loads((ROOT / "configs" / "axes.json").read_text())["held_out"]
    return f"""
| resolution | DiT tokens | step ceiling |
|--:|--:|--:|
{rows}

A kernel tuned for one token count is untuned for the next.

Every cell is also scored at a **held-out shape** drawn after your code is frozen: resolutions
{held['resolutions']}, caption lengths {held['caption_lengths']} (`configs/axes.json`). Faster only
on the published shape is `SHAPE_OVERFIT` and credits nothing.

A well-tuned general kernel counts. A lookup table keyed on one token count does not.
"""


@issue("video-temporal", "Temporal and sparse attention across frames", ["future", "video"])
def _(f):
    r = f["resolutions"]
    return f"""
BG-1 is images only. Video multiplies tokens by frames: sixteen 1024px frames with full attention is
{r[1024]['tokens'] * 16} tokens and 256x the attention arithmetic. That is why video models
factorise or sparsify attention, and that is the cell worth building.

Not in BG-1 because a video receipt would take hours. `eval/screen.py` only screens the models in
`configs/candidates.json`, none of which is a video model, and the geometry has no frame axis. A
video generation starts with both, and both are instrument changes.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default="issues")
    args = ap.parse_args()

    cand, devices, axes = load()
    f = facts(cand, devices, axes)
    out_dir = ROOT / args.out
    index = ["# Backlog", "",
             "Generated by `scripts/make_issues.py` from `configs/` and the measurements under",
             "`eval/cells/`. Do not edit by hand.", "",
             "Ceilings are arithmetic: how big a box could be, never a measured gain. An issue",
             "that quotes a measurement says so in its Basis line.",
             "", "## Open", "",
             "| issue | title | labels |", "|:--|:--|:--|"]
    closed_rows = []

    for slug, title, labels, fn, closed_by in ISSUES:
        body = fn(f).strip()
        measured = MEASURED_MARK in body
        body = body.replace(MEASURED_MARK, "").strip()
        basis = ("arithmetic ceilings, plus measured figures from the artifacts named"
                 if measured else "arithmetic ceilings only")
        status = (f"**Status:** CLOSED -- {closed_by}  \n" if closed_by
                  else "**Status:** open  \n")
        text = (f"# {title}\n\n"
                f"{status}"
                f"**Labels:** {', '.join(labels)}  \n"
                f"**Basis:** {basis}  \n"
                f"**Device:** {f['device']}\n\n"
                f"{body}\n\n"
                f"---\n\n"
                f"*Generated by `scripts/make_issues.py`. Do not edit by hand.*\n")
        if args.write:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{slug}.md").write_text(text)
        row = f"| [`{slug}`]({slug}.md) | {title} | {', '.join(labels)} |"
        (closed_rows if closed_by else index).append(row)

    if closed_rows:
        index += ["", "## Closed", "",
                  "Kept as a record of what was settled and how.", "",
                  "| issue | title | labels |", "|:--|:--|:--|"] + closed_rows
    index_text = "\n".join(index) + "\n"
    if args.write:
        (out_dir / "README.md").write_text(index_text)
        print(f">> wrote {len(ISSUES)} issues to {out_dir}/")
    else:
        print(index_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
