# The correctness gate

**Correctness precedes speed, always. A submission that fails the gate is rejected, not traded
off.** The gate runs before any timing and its cost is paid whether or not the candidate turns
out to be fast.

Two questions, in this order. The second one is the one people expect and the first one is the
one that actually decides whether the instrument works at all.

## 1. Does this build reproduce itself?

Ten replays of the same build, the same seed, the same prompt. **Byte-identical latents.** Not
close — identical.

If a build does not reproduce itself, no candidate can ever be attributed a difference, because
the instrument cannot tell a change from a replay. Every number the repository would go on to
print would be decorated noise. This is checked first because a build that fails it cannot even
be compared against the reference meaningfully.

The engagement this harness descends from lost its most promising checkpoint to exactly this:
four unhooked control replays of a sparse-MoE model produced four distinct outputs, because a few
ULP in the prefill fed discrete top-k expert routing. The model was fine. The measurement was
impossible.

Diffusion has its own versions, in the order worth checking:

- **Autotuning that picks a different algorithm per process.** `CUBLAS_WORKSPACE_CONFIG` is
  pinned for every measured run, both arms, in `eval/runner.py`. Any cuDNN or CUTLASS heuristic
  cache is the same class of problem.
- **An atomic in a reduction.** GroupNorm over a 1024×1024×128 activation is the obvious
  candidate in this pipeline.
- **TF32 enabled where an fp32 reference is expected.**
- **RNG drawn on device in launch order.** The pipeline draws its initial latent on the host, in
  a fixed order, from a splitmix64 stream keyed by the seed, specifically to avoid this.

The CPU reference ops are written for this property: every reduction runs in a fixed order with
an fp32 accumulator, and `-ffast-math` is not used anywhere — it permits reassociation, which
makes a reduction's result depend on how the compiler felt about vectorising it.

```
burnish gate --determinism --repeats 10
```

## 2. Does it match the pinned reference?

Fixed seed, frozen prompt set, latents compared against the pinned reference implementation
within a stated tolerance.

**Latents, not pixels.** The VAE decode is itself one of the things under optimization, so
comparing images would fold two questions into one and let a decoder change hide a denoiser
change.

### The frozen prompt set

`eval/cells/BG-1/prompts.json`. Four prompts, chosen to exercise different regions of the model
rather than to look good:

| id | role |
|:--|:--|
| `dense-detail` | dense high-frequency detail; the denoiser has work at every token |
| `flat-graphic` | large uniform regions; a collapsed output is obvious here and can hide in a busy image |
| `long-caption` | fills the 300-token T5 window |
| `short-caption` | almost all padding; the opposite end of the same axis |

The four span **4 to 127 real tokens in a 300-token window**, so padding is the majority of every
sequence. That is not incidental to the choice: attention masking of padding was broken in this
runtime and nothing caught it until a test asked about a masked position
(`docs/STATUS.md`). A prompt set of uniform length would not have exposed it.

The last two matter because cross-attention K/V length is the axis a caching or sparsity change is
most likely to break, and a prompt set of uniform length would never show it.

The prompt set is part of the oracle. Changing it would make every comparison against the existing
reference meaningless, and the change would not look like anything in a diff — so its SHA-256 goes
into every gate report.

### The tolerance, and why it is that number

```json
"latent_l2_relative": 0.02,
"latent_max_abs": 0.05
```

The bf16 pipeline is compared against an fp32 reference of the same graph, so the tolerance has to
admit bf16 rounding accumulated over the whole denoise loop and admit nothing else. 2% relative L2
is roughly four times the step-to-step drift bf16 rounding alone produces over 20 DPM-Solver++
steps at this resolution — enough room for a legitimately different kernel order, not enough for a
different algorithm.

**That reasoning is an argument, not a measurement, and it is expected to move exactly once.**

```
burnish gate --calibrate-tolerance
```

measures the bf16-vs-fp32 drift directly instead of arguing about it. The threshold should sit
above the measured drift and below anything an actual algorithm change produces, and when it is
set from a measurement the reasoning in `generation.json` gets updated with it rather than
replaced by a bare number.

**Widening the tolerance to admit a submission is never the answer.** If a change computes
genuinely different numbers — step caching is the obvious case — that is a new generation with its
own tolerance, argued in writing. See `issues/step-caching.md`.

**Determinism is not a tolerance.** The same build must reproduce *itself* exactly. Only the
comparison against the reference has a tolerance.

## Producing the pinned reference — not yet done

The reference latents do not exist yet. `burnish gate` reports `NO_REFERENCE` and refuses, which
is correct behaviour and not a workaround to be removed. `issues/checkpoint-load.md` tracks it.

**Pin the reference hard. It drifts between versions and it is the oracle for everything else.**

The procedure, when it happens:

1. Pin the reference implementation's exact version — not a range, a commit or release tag — and
   record it here and in `generation.json`.
2. Pin the checkpoint revision. `configs/candidates.json` already carries
   `e102b3591cc82e97071b8b4cb90d834d0c487207` for the transformer/VAE and
   `2c17b4e85261cd549b4068d086b7c2ba9d468e9f` for the T5 encoder.
3. ~~Produce token ids for the four frozen prompts with the pinned T5 tokenizer.~~ **Done.**
   `scripts/tokenize_prompts.py` produces them with the pinned `spiece.model`;
   `eval/cells/BG-1/token-ids.json` holds them alongside the prompt-set digest and the
   tokenizer's own SHA-256, and per-prompt `.txt` files feed `burnisher generate --token-ids`.
   The runtime takes ids rather than text because vendoring a SentencePiece model into a C++
   binary would put a second oracle in the repository.

   Two pinned decisions live in that script rather than being inherited as defaults:

   * **`clean_caption` is OFF.** The reference pipeline defaults it ON and silently falls back to
     OFF when `ftfy` and `BeautifulSoup` are absent — so the reference's behaviour depends on
     what happens to be installed. A benchmark cannot. The frozen prompts are written
     already-clean so both branches agree on them, and the pin says which one is meant.
   * **Truncation happens before the EOS is appended**, at `max_length - 1`, which is what
     HuggingFace does. The other order drops the EOS on exactly the captions long enough to need
     truncating — a silent difference that shows up only on the longest prompt.
4. Generate latents at the fixed seed, in fp32, at the pinned revision, on any device. Commit them
   under `eval/cells/BG-1/reference-latents/` with their digests.
5. A moved pin is a changed oracle and therefore a **new generation**, never an edit.

The reference must never be produced by this runtime. That would make the candidate its own
oracle, and it would pass.

## What the gate does not check

- **Perceptual quality.** The gate is a numerical comparison. `latent_l2_vs_reference` is
  additionally a scored frontier objective, so a change that stays inside the gate while
  measurably degrading shows up as a smaller number rather than as an invisible pass — but neither
  is a judgement about whether the image looks good.
- **Anything at a shape the prompt set does not cover.** That is what the held-out shape guard in
  `burnish bench` is for, and it is a separate mechanism.
