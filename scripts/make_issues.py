#!/usr/bin/env python3
"""Generate the opportunity backlog as issue files, with each item's arithmetic attached.

Every figure in every issue is computed here from `configs/` and the same geometry the scorer
uses. None of them is typed. That matters more for the backlog than anywhere else: an issue is
a pitch for a week of somebody's time, and a pitch built on a remembered number is how a
contributor ends up chasing a surface that is not there.

Every figure is also `basis: model`. An issue says how big a box COULD be; it cannot say how
full it is, because nothing here has been measured on the pinned hardware.

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
            "vae_attn_score_bytes_gb": (vae_mat.traffic_bytes - vae.traffic_bytes) / 1e9,
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
    t5 = G.t5_encoder(cand["text_encoder"], seq=300, batch=2)
    f["t5_params_gb"] = t5.param_bytes / 1e9
    f["t5_ceiling_ms"] = bound_for(t5, dev, cell="t5").ceiling_seconds * 1e3
    for dt in ("fp8", "nvfp4"):
        t = G.t5_encoder(cand["text_encoder"], seq=300, batch=2, wdtype=dt, adtype="bf16")
        f[f"t5_params_{dt}_gb"] = t.param_bytes / 1e9
    return f


ISSUES = []


def issue(slug, title, labels):
    def wrap(fn):
        ISSUES.append((slug, title, labels, fn))
        return fn
    return wrap


@issue("dit-attention", "DiT self-attention at 4k-16k tokens", ["kernel", "dit", "cuda"])
def _(f):
    r = f["resolutions"]
    rows = "\n".join(
        f"| {res} | {v['tokens']} | {v['dit_attention_share']:.1%} | "
        f"{v['dit_ceiling_ms']:.1f} ms | {v['dit_flops_t']:.2f} T |"
        for res, v in sorted(r.items()))
    return f"""
Self-attention is the largest single block of arithmetic in the denoise step, and its share
grows quadratically with resolution while everything around it grows linearly.

| resolution | image tokens | share of DiT-step FLOPs | DiT-step ceiling | DiT-step FLOPs |
|--:|--:|--:|--:|--:|
{rows}

At 2048px the self-attention alone is {r[2048]['dit_attention_share']:.0%} of the step, because
{r[2048]['tokens']} tokens is a {r[2048]['tokens'] // r[512]['tokens']}x token count over 512px
and attention costs the square of it.

**What is here now.** `attention` has two registered implementations: `stock`, an online-softmax
streaming reference, and `materialized`, which writes the whole score matrix. Both are CPU only.
There is no CUDA implementation of either, so this cell currently cannot run on a device at all.

**What would count.** A CUDA attention kernel registered under a new name, A/B'd against `stock`
in one process. The fp8 and NVFP4 paths are separate cells with their own published ceilings --
`dit-step/1024/fp8` at {f['dit_ceiling_fp8_ms']:.1f} ms and `dit-step/1024/nvfp4` at
{f['dit_ceiling_nvfp4_ms']:.1f} ms against bf16's {f['dit_ceiling_bf16_ms']:.1f} ms -- and
neither has a reference implementation, so landing one is also a cartography contribution
(docs/CARTOGRAPHY.md).

**Read this before starting.** The ceilings above are ARITHMETIC and the achieved fraction of
every one of them is currently `null`, because no Blackwell device has run this code. A ceiling
tells you how big the box is and says nothing about how full it is. `burnish calibrate` fills
that in and it has not been run.
"""


@issue("vae-decode", "VAE decode: tiling, fusion, and the mid-block attention",
       ["kernel", "vae", "cuda"])
def _(f):
    r = f["resolutions"]
    rows = "\n".join(
        f"| {res} | {v['vae_ceiling_ms']:.1f} ms | {v['vae_flops_t']:.2f} T | "
        f"{v['vae_bound_by']} | {v['vae_fusion_headroom']:.2f}x | "
        f"{v['vae_attn_score_bytes_gb']:.2f} GB |"
        for res, v in sorted(r.items()))
    return f"""
| resolution | ceiling | FLOPs | bound by | fusion headroom | score matrix if materialized |
|--:|--:|--:|:--|--:|--:|
{rows}

**A correction worth making before anyone starts.** VAE decode is widely described as
memory-bound. On this part, at its arithmetic ceiling, it is not: the convolutions have
arithmetic intensities in the thousands against a ridge point near 117, and the stage is
compute-bound at every resolution in the table. What is true is narrower and more useful --
*as implemented*, it moves a great deal of intermediate traffic, and a direct convolution at
128 and 256 channels reaches a small fraction of tensor peak. Both of those are statements about
the ACHIEVED FRACTION, which is exactly the room this cell has, and neither is a statement about
the bound.

Two concrete surfaces the arithmetic does point at:

- **Fusion.** The `fusion headroom` column is the sum of per-op bounds over the whole-stage
  bound. At 2048px it is {r[2048]['vae_fusion_headroom']:.2f}x, and that entire gap is
  intermediate traffic a fused implementation would not move. One 1024x1024x128 activation is
  268 MB in bf16 and a ResNet block touches several.
- **The mid-block attention.** It is spatial self-attention over every latent position --
  {r[1024]['tokens'] * 4} positions at 1024px. Materialized, its score matrix alone is
  {r[1024]['vae_attn_score_bytes_gb']:.2f} GB of round trip at 1024px and
  {r[2048]['vae_attn_score_bytes_gb']:.2f} GB at 2048px. `attention=materialized` is registered
  precisely so a contributor can measure the naive path and show that removing it helped.

Tiling is the other half and it changes the op list rather than scaling it, so a tiled decode is
a new cell with its own ceiling rather than a faster version of this one.
"""


@issue("text-encoder", "T5-XXL: 89% of the checkpoint, 2% of the clock",
       ["memory", "quantization", "text-encoder"])
def _(f):
    s20 = f["shares_20steps"]
    s4 = f["shares_4steps"]
    return f"""
**The finding that should decide how you spend time here.** The text encoder holds
{f['t5_params_gb']:.2f} GB of the {f['resident_all_gb']:.2f} GB this pipeline keeps resident --
{f['t5_params_gb'] / f['resident_all_gb']:.0%} of it -- and at twenty steps it is
{s20['t5-encode']:.1%} of the predicted wall clock. It runs once; the DiT runs twenty times.

Ranking stages by parameter count would send somebody to the biggest weights, which are these,
and they would be working on a fiftieth of the clock.

| steps | t5-encode | dit-step | vae-decode | total ceiling |
|--:|--:|--:|--:|--:|
| 20 | {s20['t5-encode']:.1%} | {s20['dit-step']:.1%} | {s20['vae-decode']:.1%} | {f['total_ms_20steps']:.0f} ms |
| 4 | {s4['t5-encode']:.1%} | {s4['dit-step']:.1%} | {s4['vae-decode']:.1%} | {f['total_ms_4steps']:.0f} ms |

So this is a **memory-axis** cell, not a latency one, and the frontier scores memory. Two things
are worth real money here and neither is a faster kernel:

- **Quantization.** fp8 takes the encoder to {f['t5_params_fp8_gb']:.2f} GB and NVFP4 to
  {f['t5_params_nvfp4_gb']:.2f} GB, against {f['t5_params_gb']:.2f} GB at bf16. That is
  {f['t5_params_gb'] - f['t5_params_nvfp4_gb']:.1f} GB of a {f['vram_gib']:.0f} GiB card returned
  to the denoise loop, and it is scored on the `peak_vram_bytes` objective directly.
- **Caching.** The encoder output depends only on the prompt. A pipeline that re-encodes an
  unchanged prompt is doing {f['t5_ceiling_ms']:.0f} ms of arithmetic for nothing. This is worth
  the most exactly where the latency share is worst -- at four steps it is
  {s4['t5-encode']:.1%} of the clock.

Note also that the share rises as step counts fall, so a distilled model reopens this cell.
"""


@issue("fused-adaln", "Fuse AdaLN modulation into its neighbours",
       ["kernel", "fusion", "dit"])
def _(f):
    r = f["resolutions"][1024]
    return f"""
`modulate` is `x * (1 + scale) + shift`: two flops per element against a full activation round
trip. It is the canonical fusion target in this pipeline and it is already its own registered op
so that a fused version can be registered beside the unfused one and the two compared directly.

At 1024px, one DiT step's elementwise and norm ops move
**{r['dit_elementwise_bytes_gb']:.2f} GB** of the step's {r['dit_traffic_gb']:.2f} GB of total
traffic -- {r['dit_elementwise_bytes_gb'] / r['dit_traffic_gb']:.0%} of it -- to do a negligible
share of its {r['dit_flops_t']:.2f} TFLOPs. Every byte of that is removable in principle by
folding the modulation into the epilogue of the GEMM before it or the prologue of the one after.

The whole-step fusion headroom at 1024px is **{r['dit_fusion_headroom']:.2f}x** (the sum of
per-op bounds over the whole-stage bound), and this op family is most of it.

**Why the ceiling does not move when you win.** The published ceiling counts only *unavoidable*
bytes -- weights read once, stage input, stage output -- and excludes every intermediate
precisely so that fusing does not move the target you are scored against. Removing this traffic
raises the achieved fraction; it does not lower the ceiling.

**Where to start.** `ModulateArgs` already carries `residual` and `gate`, so the gated residual
`x + gate * modulated` is one op rather than two loops. The DiT calls it that way in
`src/models/pixart_dit.cpp`. A fused GEMM epilogue would subsume it entirely.
"""


@issue("weight-formats", "NVFP4 and MXFP4 on silicon with no reference tuning",
       ["quantization", "cuda", "cartography"])
def _(f):
    return f"""
Blackwell's FP4 path is new and there is no established tuning for it in any open diffusion
runtime. That is unusual and it is the reason this is worth more than it looks: the cells are
published, their ceilings are computable today, and nobody has a reference to beat.

| dtype | DiT-step ceiling @1024px | DiT resident | T5 resident |
|:--|--:|--:|--:|
| bf16 | {f['dit_ceiling_bf16_ms']:.2f} ms | {f['dit_params_bf16_gb']:.2f} GB | {f['t5_params_gb']:.2f} GB |
| fp8 | {f['dit_ceiling_fp8_ms']:.2f} ms | {f['dit_params_fp8_gb']:.2f} GB | {f['t5_params_fp8_gb']:.2f} GB |
| nvfp4 | {f['dit_ceiling_nvfp4_ms']:.2f} ms | {f['dit_params_nvfp4_gb']:.2f} GB | {f['t5_params_nvfp4_gb']:.2f} GB |

Both cells are declared in BG-1 with `implemented: false` and weight 0 -- the ceiling is
published so the room is visible, and the cell cannot drag an aggregate it is not part of.

**This is a cartography contribution as much as a kernel one.** Landing the reference
implementation and the calibration for one of these cells is scored in its own right
(docs/CARTOGRAPHY.md), because the subnet's health depends on axis supply and making that an
admin chore rather than a paid contribution is how a benchmark stops growing.

**The width is not 0.5 bytes.** `dtype_bytes(NVFP4)` is 0.5625 -- four bits of payload plus one
fp8 scale per sixteen elements. A roofline that priced it at half a byte would be wrong by 12%
in the optimistic direction, which is the direction that costs somebody a week.

**Correctness comes first and it will be the hard part.** The tolerance in BG-1 admits bf16
rounding accumulated over twenty DPM-Solver++ steps and nothing else. A 4-bit weight path will
need its own tolerance, argued in writing, in its own generation -- not a widened version of
this one.
"""


@issue("step-caching", "Step and feature caching: algorithmic, and it must pass the gate",
       ["algorithm", "scheduler"])
def _(f):
    return f"""
The denoise loop runs the same graph twenty times on inputs that change slowly. Caching a block's
output across adjacent steps, or skipping a step's computation entirely and reusing the previous
residual, is worth a multiple rather than a percentage -- reported elsewhere at 1.5-2x.

The arithmetic is trivial and that is the point: at twenty steps the DiT is
{f['shares_20steps']['dit-step']:.1%} of the predicted clock, so skipping k steps of twenty
removes k/20 of {f['shares_20steps']['dit-step']:.0%} of it. Skipping four is worth about
{4 / 20 * f['shares_20steps']['dit-step']:.0%} of the whole generation.

**And this is the one backlog item where the correctness gate is the whole problem.** Every other
item on this list is a faster way to compute the same numbers, and passes the gate by
construction. This one computes DIFFERENT numbers on purpose. That makes it:

- **a tolerance question first.** BG-1's tolerance is 2% relative L2 on the latents, justified as
  roughly four times the bf16 rounding drift over twenty steps. A caching scheme will not fit
  inside that, and the answer is NOT to widen it -- it is a new generation with its own
  tolerance, argued in writing, calibrated with `burnish gate --calibrate-tolerance`.
- **a frontier question second.** The generation scores `latent_l2_vs_reference` as an objective.
  A change that is faster and measurably further from the reference has moved along the frontier
  rather than expanded it, and the receipt will say `MOVED_ALONG_FRONTIER` and credit nothing.
  That is the correct answer for a quality/speed trade and it is not a bug to be worked around.

If you want this scored as a win, the work is to show the cache is *free* within a stated
tolerance -- not to show it is fast.
"""


@issue("weight-upload", "A single generation is dominated by uploading the weights",
       ["cuda", "measured", "startup"])
def _(f):
    return f"""
**Measured on the pinned RTX 5090, and it is the largest single cost in a one-shot generation.**

`burnisher generate` maps {f['resident_all_gb']:.1f} GB of checkpoint and uploads it to the
device on every invocation. At 1024px and 20 steps the whole denoise loop has an arithmetic
ceiling of {f['shares_20steps']['dit-step'] * f['total_ms_20steps']:.0f} ms, and the upload takes
tens of seconds. The correctness gate runs seven generations and spends the overwhelming majority
of its wall time moving weights it already moved six times.

This does NOT affect any scored cell. `burnish bench` loads once and times the stage afterwards,
and the generation's cells are per-invocation; the arithmetic in `docs/ROOFLINE.md` counts a
weight read per invocation, not per process. It affects the COST of producing a receipt, which is
screen question six, and it affects anybody actually using the runtime.

Three directions, in increasing order of effort:

- **Keep the process alive.** The gate and the bench both spawn one process per run so that no
  state leaks between arms — a deliberate choice — but a resident server with an explicit reset
  would keep the guarantee and pay the upload once.
- **Map the checkpoint to the device directly.** The weights are already mmapped on the host and
  then copied; `cudaHostRegister` on the mapping, or a direct read into device memory, removes
  one full copy.
- **Keep less resident.** The text encoder is {f['t5_params_gb']:.2f} GB of the
  {f['resident_all_gb']:.2f} GB and is idle for the entire denoise loop -- see
  `issues/offload.md` and `issues/text-encoder.md`.

**The measurement to take first** is the split between map, convert and upload. All three are in
`DeviceWeights::get`, none of them is separately timed, and guessing which dominates is exactly
the habit this repository is built against.
"""


@issue("cuda-graphs", "Capture the denoise loop as a CUDA graph", ["cuda", "launch-overhead"])
def _(f):
    r = f["resolutions"]
    rows = "\n".join(
        f"| {res} | {v['dit_launches_per_step']} | {v['dit_launches_per_step'] * 20} | "
        f"{v['dit_ceiling_ms']:.1f} ms |"
        for res, v in sorted(r.items()))
    return f"""
The denoise loop is perfectly static: the same graph, the same shapes, the same twenty times.
Nothing about it needs to be re-recorded per step, which makes it the textbook case for
`cudaGraphLaunch`.

| resolution | kernel launches per DiT step | per 20-step generation | step ceiling |
|--:|--:|--:|--:|
{rows}

Those counts come from the op enumeration in `eval/burnscore/geometry.py`, which is the same
enumeration the roofline is computed from, so they are the launches the runtime actually issues
rather than an estimate.

**What the arithmetic can and cannot tell you.** It can tell you the launch COUNT. It cannot tell
you what a launch costs on this part, because that is a measurement and nobody has taken it here.
At a few microseconds each, {r[1024]['dit_launches_per_step'] * 20} launches is single-digit
milliseconds against a {f['total_ms_20steps']:.0f} ms ceiling -- worth having and not
transformative. The honest framing is that this is a *small, certain* win rather than a large
speculative one, and it becomes more interesting at low resolution where the step is short and
the launch count is unchanged.

**It also interacts with everything else on this list**, which is the real argument for doing it
early: a captured graph makes every subsequent kernel change measurable without launch noise
underneath it, and the calibration in `burnish calibrate` measures a quieter cell as a result.
"""


@issue("offload", "Offload and streaming: video models do not fit",
       ["memory", "streaming", "frontier"])
def _(f):
    per = f["resident_per_stage_gb"]
    rows = "\n".join(f"| `{k}` | {v:.2f} GB |" for k, v in sorted(per.items()))
    return f"""
| stage | resident parameters (bf16) |
|:--|--:|
{rows}
| **all resident** | **{f['resident_all_gb']:.2f} GB** |
| **streamed (largest stage only)** | **{f['resident_streamed_gb']:.2f} GB** |

The card is {f['vram_gib']:.0f} GiB. The difference between those last two rows --
{f['resident_all_gb'] - f['resident_streamed_gb']:.2f} GB -- is what a streaming arrangement
returns, and almost all of it is the text encoder sitting idle through the entire denoise loop.

For PixArt-Sigma at 1024px this is comfortable either way, and that is worth saying plainly
rather than overselling it: {f['resident_all_gb']:.2f} GB of {f['vram_gib']:.0f} GiB is not a
crisis. **It stops being comfortable immediately outside this generation.** Qwen-Image's DiT
alone is 20B parameters -- 40 GB at bf16 against 32 -- and was ruled out of v0 on FIT, not on
arithmetic (`configs/candidates.json`). Video models are worse again.

So this item is best understood as the prerequisite for BG-2 rather than as a win in BG-1. It is
scored on the frontier's `peak_vram_bytes` objective, which means a streaming scheme that costs
latency is a move ALONG the frontier, and one that costs nothing is an expansion.

**The measurement trap here is specific.** On a CPU build `peak_vram_bytes` reports host peak RSS.
On a CUDA build it must report the device allocator's high-water mark. Scoring the wrong resource
would make every result in this cell meaningless, and it would look completely reasonable.
"""


@issue("shape-specialization", "Per-resolution shape specialization and the held-out guard",
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

A kernel tuned at one token count is untuned at the next, and the counts here span
{min(v['tokens'] for v in r.values())} to {max(v['tokens'] for v in r.values())}. This is the
REGENERATION property that makes generation worth having as a second scored target: the surface
reopens with every resolution, every dtype and every model, where a text decode runtime's shapes
are fixed by its checkpoint.

**And it is the item the anti-gaming guard is aimed at.** Specializing for exactly the benchmarked
shape is the cheapest possible way to produce a number, so every cell is scored on its published
shape AND on a held-out shape the evaluator picks at run time, from the base commit, after the
candidate is frozen. The held-out resolutions are {held['resolutions']} and the held-out caption
lengths are {held['caption_lengths']}; they are listed in the open because hiding them would not
help. What makes the guard work is that the candidate cannot know which one will be drawn.

A candidate faster on the published shape and slower on a held-out one is reported as
`SHAPE_OVERFIT` and credits nothing. A general kernel that happens to be tuned well is a
contribution; a lookup table keyed on 4096 tokens is not.
"""


@issue("cuda-op-backend", "The CUDA op backend does not exist yet",
       ["cuda", "blocking", "v0"])
def _(f):
    return f"""
**This is the largest missing piece in the repository and it blocks every measurement.**

What exists: the op registry, six ops with complete CPU reference implementations, the three
model graphs, the scheduler, the pipeline, and a `probe` that measures a device's real peaks.
`burnisher selftest` runs the entire graph end to end and reproduces itself byte for byte.

What does not exist: a CUDA implementation of any op. `src/cuda/` contains `device.cu` -- the
probe -- and nothing else. There was no CUDA toolkit and no Blackwell part available when this
was written, and shipping kernels that had never been compiled, let alone run, as though they
worked is the exact failure this repository is built to avoid. docs/STATUS.md says so in those
words.

**The consequence, stated plainly.** Every cell's `achieved` and `floor_pct` is null. Every
ceiling stands on a VENDOR device peak rather than a probed one. `burnish calibrate`,
`burnish gate` and `burnish bench` all refuse to run rather than estimating. The scorer is
complete and tested against synthetic measurements and has never scored a real one.

**The shape of the work**, in the order that unblocks the most:

1. Device storage for `Tensor` (a `Device::CUDA` allocation path with a high-water mark, which
   the memory objective needs anyway).
2. `gemm` through cuBLASLt, honouring `b_transposed` and the epilogue -- the epilogue is where
   the AdaLN fusion lands later.
3. `attention`, both registered names, so the naive path stays measurable.
4. `norm`, `modulate`, `activation` -- simple, and they are the fusion targets.
5. `conv2d` -- the VAE's shapes are few and fixed, which is what makes a specialized path
   plausible.

Each one registers a NAME beside the CPU reference rather than replacing it, so the CPU oracle
stays runnable and a device kernel can be diffed against it under the correctness gate.

The CI job `cuda-compile` compiles `device.cu` for sm_120 on a runner with a toolkit. It is
`continue-on-error` today because nothing in this repository has ever seen nvcc.
"""


@issue("checkpoint-load", "Load the pinned checkpoint and pin the reference latents",
       ["blocking", "correctness", "v0"])
def _(f):
    n_tensors = f["checkpoint_tensors_verified"]
    return f"""
`burnisher generate --weights DIR --token-ids FILE` is wired and loads a real checkpoint. Three
things stood between this repository and its first real number. One and a half are now done.

**1. The checkpoint layout mapping — DONE, and verified.** `SafeTensors` maps a file and resolves
tensors by name, and `declare_pixart_shapes()` enumerates every name the three models ask for.
All **{n_tensors}** of them have now been checked against the real checkpoint at the pinned
revisions: **{f['checkpoint_missing']} missing, {f['checkpoint_wrong_shape']} wrong shape**.

The check costs about 1.8 MB rather than 22 GB. A safetensors file begins with an 8-byte header
length and then that many bytes of JSON naming every tensor and its shape, so two HTTP range
requests per shard fetch the whole layout. `scripts/verify_checkpoint_layout.py` does it;
`configs/checkpoint-layout.json` is the committed record, and CI re-checks the runtime against it
offline on every push.

It found one real defect on its first run — `pos_embed.proj.weight` was declared flattened as
`[1152, 16]` where the checkpoint stores the conv layout `[1152, 4, 2, 2]`. Same bytes in the same
order, so the runtime would have worked; the declaration was still wrong, and a shape that file
gets wrong is a shape nothing else can catch.

**2. Pre-tokenized prompt ids.** The T5 tokenizer is a SentencePiece model. Vendoring one would
put a second oracle in the repository, so `burnisher generate --token-ids FILE` takes ids
directly — one prompt per line, negative first under classifier-free guidance. What is missing is
the ids themselves: the frozen prompt set is `eval/cells/BG-1/prompts.json` and its four prompts
need their ids produced with the pinned tokenizer and committed with a digest.
docs/CORRECTNESS.md has the procedure.

**3. The reference latents.** The gate compares against latents produced by the PINNED reference
implementation at the PINNED revision. They cannot be produced by this runtime -- that would make
the candidate its own oracle -- and they do not exist yet. Until they do, `burnish gate` reports
`NO_REFERENCE` and refuses, which is the correct behaviour and not a workaround to be removed.

**Pin the reference hard.** It drifts between versions and it is the oracle for everything else.
The revision in `configs/candidates.json` is pinned to a commit; the reference implementation's
own version must be pinned the same way, in docs/CORRECTNESS.md, and a moved pin is a new
generation rather than an edit.
"""


@issue("video-temporal", "Temporal and sparse attention across frames", ["future", "video"])
def _(f):
    r = f["resolutions"]
    return f"""
Burnisher is named for image *and video* generation and BG-1 is images only. This issue records
what the arithmetic already says about the video case so it is not rediscovered later.

Attention cost is quadratic in the token count, and a video model's token count is the image
count times the frames. At 1024px one frame is {r[1024]['tokens']} tokens and self-attention is
already {r[1024]['dit_attention_share']:.0%} of the step. Sixteen frames of full 3D attention is
{r[1024]['tokens'] * 16} tokens and {16 * 16}x the attention arithmetic -- which is why every
video model in practice factorises it, and why temporal and sparse attention patterns are the
cell that matters there rather than a faster dense kernel.

**This is deliberately not in BG-1** and the reason is the screen rather than ambition. A video
generation is minutes of GPU time, the SCORE_COST question asks for a receipt in minutes rather
than hours, and a matrix nobody can afford to calibrate is a matrix with guessed noise floors in
it. `eval/screen.py` is the tool that settles this: run it against a video candidate's config
before proposing BG-N, not after.

The Apache-2.0 ungated video checkpoints are the obvious place to start when that happens.
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
    index = ["# Opportunity backlog", "",
             "Generated by `scripts/make_issues.py`. Every figure below is computed from",
             "`configs/` by the same geometry the scorer uses; none is typed. Re-generate after",
             "any change to a config, or the backlog and the roofline table will disagree.", "",
             "**Every number here is `basis: model`** -- arithmetic, from a config file and a",
             "device peak. An issue can say how big a box could be. It cannot say how full it is,",
             "because nothing in this repository has been measured on the pinned hardware yet.",
             "", "| issue | title | labels |", "|:--|:--|:--|"]

    for slug, title, labels, fn in ISSUES:
        body = fn(f).strip()
        text = (f"# {title}\n\n"
                f"**Labels:** {', '.join(labels)}  \n"
                f"**Basis:** model (arithmetic). No measurement appears below.  \n"
                f"**Device:** {f['device']}\n\n"
                f"{body}\n\n"
                f"---\n\n"
                f"*Generated by `scripts/make_issues.py`. Do not edit by hand: every figure is\n"
                f"computed from `configs/`, and a number typed here would disagree with\n"
                f"`docs/ROOFLINE.md` and with the scorer.*\n")
        if args.write:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{slug}.md").write_text(text)
        index.append(f"| [`{slug}`]({slug}.md) | {title} | {', '.join(labels)} |")

    index_text = "\n".join(index) + "\n"
    if args.write:
        (out_dir / "README.md").write_text(index_text)
        print(f">> wrote {len(ISSUES)} issues to {out_dir}/")
    else:
        print(index_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
