# What is real in this repository, and what is not

Overselling the surface is the single failure mode that kills a subnet. A contributor who burns a
week of GPU time for a result inside the noise floor does not come back. So this page is the
first one to read and it is deliberately blunt.

## The one-sentence version

**Everything the scoring model needs is measured. Every implemented cell has an achieved
fraction and its own noise floor, the scorer produces receipts, and the correctness gate is
calibrated from measurement rather than argument.**

### Measured on the pinned RTX 5090

| | |
|:--|--:|
| sustained bandwidth | **1506.7 GB/s** (84% of the 1792 spec) |
| bf16 GEMM through cuBLAS | **236.9 TFLOPS** |
| determinism, 5 replays of the bf16 CUDA pipeline | **byte-identical** |
| CUDA scheduler vs the reference | 1.7e-07 |
| CUDA T5 encoder vs the reference | 7.6e-07 |
| CUDA VAE decoder vs the CPU oracle | 5.8e-06 |
| CUDA DiT vs the reference | 5.2e-05 |
| reference latents, 4 prompts at 1024px / 20 steps, fp32 AND bf16 | **committed** |

### The calibrated cells

| cell | achieved | still available | noise floor | resolvable |
|:--|--:|--:|--:|:--:|
| `t5-encode/1024/bf16` | 18.2% | 5.5x | 3.753% | yes |
| `dit-step/1024/bf16` | 1.5% | **65.6x** | 0.578% | yes |
| `vae-decode/1024/bf16` | 0.8% | **126.4x** | 0.259% | yes |

Nine paired control-vs-control repeats per cell, real weights, on the pinned part. The floors are
`spread`-decided in every cell, meaning the run-to-run variation is larger than the instrument's
own resolution -- which is the honest case; a floor decided by resolution would mean the bench
cannot see its own noise.

**That headroom is real and it is the point.** The kernels are deliberately naive: attention is
one block per query row with no tiling and no tensor cores, convolution is one thread per output
element, and every GEMM epilogue is a separate pass. v0 ships a correct, complete, SLOW pipeline
and contributors make it fast.

**And the floors say a contribution can be small and still count.** In `dit-step` the floor is
worth 0.00009 of the remaining gap, so a change that closes nine parts in a hundred thousand of
what is left is already outside the noise. That is the opposite of a 2% threshold, and it is what
measuring the floor instead of guessing it buys.

## What has been built and checked

| thing | state | how you can check it yourself |
|:--|:--|:--|
| The six-question screen | complete, runs | `burnish screen` |
| Per-cell arithmetic rooflines | complete, generated from configs | `burnish roofline` |
| Gap-closed scoring | complete, and it has scored a real run | `python3 -m unittest discover -s eval -t eval` |
| Measured noise floor, paired bootstrap | complete, tested | same |
| Frontier over latency / VRAM / fidelity | complete, tested | same |
| Receipts and the append-only ledger | complete, tested | same |
| Trusted-instrument overlay | complete | `eval/run_from_base.sh` |
| Op registry with named implementations | complete, 4303 assertions | `ctest --test-dir build` |
| CPU reference ops | complete | `burnisher info` |
| T5 encoder / PixArt DiT / VAE decoder graphs | complete | `burnisher selftest` |
| DPM-Solver++ scheduler | complete, pinned against the reference construction | `ctest` |
| End-to-end pipeline, byte-identical replays | complete | `burnisher selftest` |
| Correctness gate (determinism + reference) | complete, and it has rejected and passed real builds | `eval/gate.py` |
| Paired bench and calibration runners | complete, run on the pinned part | `examples/BG-1-pr-000001-raw.json` |
| CUDA device probe | compiled for sm_120 and run on the pinned part | `burnisher probe` |
| CUDA op backend | all 15 ops, gated against the pinned reference | `issues/cuda-op-backend.md` |
| Checkpoint tensor names and shapes | verified against the pinned revisions, 962/962 | `scripts/verify_checkpoint_layout.py` |
| Checkpoint load path (mmap, shard search, dtypes) | complete, run against real weights | `burnisher check-weights --weights DIR` |
| `burnisher generate` on a real checkpoint | run, gated, and calibrated on the pinned part | `issues/checkpoint-load.md` |
| Reference latents for the gate | committed, 4 prompts in fp32 and bf16 | `eval/cells/BG-1/reference-latents/` |
| Every implemented cell's achieved fraction | measured, 9 paired repeats | `burnish roofline` |
| Every implemented cell's noise floor | measured, `spread`-decided in all three | `burnish roofline` |
| Device peaks behind every implemented cell | measured on the pinned 5090 | `configs/devices.json` |
| Device peaks behind the fp8/NVFP4 cells | **vendor, not probed** | `configs/devices.json` |

### On "run against real weights"

All three stages have been loaded from the pinned checkpoint and run. All **962** required
tensors resolve through the real loader — `burnisher check-weights` maps 21.8 GB and finds
0 missing and 0 wrong shape.

| stage | tensors | mapped | wall (CPU reference) | output |
|:--|--:|--:|--:|:--|
| `t5-encode`, 16 tokens | 219 | 19049 MB | 443 s | mean +0.0002, std 0.167, \|max\| 7.64 |
| `dit-step` at 64px, batch 2 | 603 | 2443 MB | 119 s | mean +0.005, std 0.616, \|max\| 4.25 |
| `vae-decode` at 32px | 140 | 334 MB | 33 s | mean +0.164, std 0.111, \|max\| 0.465 |

Each is what a healthy result looks like: a T5 hidden state is small-magnitude with occasional
outliers, a DiT's epsilon prediction sits near zero at roughly unit scale, a VAE decoder's output
is a bounded image-like tensor. A wrong weight mapping generally does not look like any of them —
it gives NaNs, or magnitudes in the thousands, or a constant.

(Those wall times are the deliberately-slow CPU reference doing scalar arithmetic. They are not
benchmarks of anything and are recorded only to say the runs happened.)

That exercises the mmap, the header parse, the offset arithmetic, the dtype mapping, the shard
search and the two model graphs against real bytes. On its own it establishes only that the
output is *plausible*, which is weaker than "the output is right" — the difference is the
correctness gate, and the gate now runs against committed reference latents. Both tiers are kept
here because they fail differently: a plausible-looking output is what every defect below
produced.

### Five correctness defects the runtime had, and how they were found

Every one of them was invisible to every self-consistency check in the repository, which is the
point worth recording: a deterministic wrong answer passes a determinism test, and two
implementations that are wrong the same way agree with each other. Each produced a plausible
image.

**Attention was reading the wrong slices.** The op indexed `[batch, heads, seq, head_dim]` while
every projection GEMM produces `[batch * seq, heads * head_dim]` — which is `[batch, seq, heads,
head_dim]` contiguously. So attention attended over reinterpreted data, consistently and
deterministically, in the DiT, the T5 encoder and the VAE mid-block. The streaming and
materialised implementations agreed with each other because they were wrong identically; the
whole-model determinism test passed because the wrongness was deterministic. It surfaced only
when a cross-attention *mask* test asked a question the layout could not answer. The op is now
head-last and `tests/test_ops.cpp` pins the invariant: H-head attention must equal H independent
single-head attentions over the corresponding slices.

**The gather kernel read fp32 token ids as the weight table's dtype.** The CUDA gather was
templated on the table's type and cast the ids pointer to that same type. The ids are always
fp32. At fp32 the two coincide and everything works; at **bf16** it read fp32 data as bf16,
produced garbage token ids, and the text encoder returned embeddings for the wrong tokens. The
assembled pipeline then disagreed with the reference by a relative L2 of **1.20** — essentially
uncorrelated — while every fp32 check in the repository passed, because fp32 is exactly the case
where the bug cannot appear. Determinism passed too: it was deterministically wrong.

It is the only pair of operands in the backend with **different dtypes**, which is the whole
lesson. `GatherArgs` now carries an explicit contract and both implementations refuse a
non-fp32 id tensor rather than reinterpreting it.

Finding it took five halvings, and the first one is the transferable part:

| question | answer | what it ruled out |
|:--|:--|:--|
| end-to-end, 20 steps, bf16 | 1.20 | nothing — a defect and chaos look identical here |
| sweep the STEP COUNT | **0.993 at one step** | chaos: amplification starts small, this did not |
| stage by stage at bf16 | T5 1.028, DiT 0.137, VAE 0.020 | the DiT and the VAE |
| T5 by layer | **1.95 at one layer** | amplification within the encoder |
| my CUDA bf16 vs my CPU bf16 | CPU right, CUDA 1.69x too large | the graph — it was the kernel |

After the fix the T5 agrees at **0.0124**, and the DiT's 0.137 turns out to be honest
amplification: 0.005 at one block, 0.009 at four, 0.137 at twenty-eight — the same curve shape
as fp32, starting higher because bf16 starts higher.

**The move worth keeping:** when two hypotheses look identical at full scale, shrink the scale
until they separate. A defect is present at one step and one layer; amplified rounding is not.
The same bisection localised the attention layout bug by truncating the block stack.

**The sampler used the wrong sigma.** DPM-Solver++ carries two sigmas per endpoint: the
Karras-style `sqrt((1-acp)/acp)`, whose ratio is `exp(-h)`, and the variance-preserving
`sigma/sqrt(sigma^2+1)`, whose ratio is the coefficient on the sample in the update. This code
used the Karras ratio in both places. At t=999 those two numbers are **157 and 0.99998**, so it
is not a small error — and the update still ran, still stayed finite, and still produced a
plausible trajectory. Found by comparing the sampler alone against the reference scheduler, with
no weights and no model involved: `scripts/differential_test.py --stage scheduler`. It now agrees
to 1.7e-7 and `tests/test_scheduler.cpp` pins four steps of the reference's own trajectory,
covering both solver orders and the `lower_order_final` case.

**The output patch ordering was transposed.** The reference lays each token's output vector out
as (row, column, CHANNEL) — channel varying fastest — and then permutes. The input side is
(channel, row, column), because the patch embedding is a `Conv2d` and that is a conv weight's
layout while the output projection is a `Linear`. Two different orderings at the two ends of the
same model is not a design; it is what the reference does, and it is not optional.

The signature is worth remembering: the disagreement was a relative L2 of **1.37** with the
reference — completely different values — and an **identical mean and standard deviation to five
decimal places**. The same numbers in a different arrangement. A round-trip test of
`patchify`/`unpatchify` passes with both ends wrong, and did.

**Padding was not masked at all.** A prompt is padded to a fixed 300 tokens, so most of a short
caption is padding, and both the T5 self-attention and the DiT cross-attention were attending to
it on every layer. That is a different model — one that still produces a plausible image. The
attention op now takes a `[batch, kv_len]` key mask, per batch row, because under classifier-free
guidance the negative and positive prompts have different lengths and one shared mask either
attends to padding or drops real tokens.

None of them would have survived the correctness gate against a reference. All of them survived
everything this repository could check without one. That is the whole argument for the gate, and
it is why the gate is not negotiable for a submission: correctness before speed, always.

**And the strongest check now available: stage-by-stage against the reference.**

`scripts/differential_test.py` runs one stage in this runtime and the same stage in diffusers, on
the same weights and the same input tensor, and compares. That is the correctness gate's question
at a scale a CPU can answer, and it is what found the patch-ordering and sampler defects.

**All four stages agree with the reference.** That is the strongest correctness statement
available without a GPU, and it is weaker than the gate: one forward pass per stage, in fp32, at
shapes no receipt is scored on. It says the arithmetic is right. It does not say the twenty-step
bf16 loop at 1024px is — which is exactly what the gate now checks, against committed latents, on
the pinned part.

| stage | shape | relative L2 vs reference | verdict |
|:--|:--|--:|:--|
| `vae-decode` | 4x4 latent | **6.9e-06** | agrees to fp32 epsilon |
| `dit-step` | 64px, batch 2, 28 blocks | **7.3e-04** | agrees; see below |
| `scheduler` | 20 steps, no weights | **1.7e-07** | agrees to fp32 epsilon |
| `t5-encode` | 16 tokens, batch 2, 24 blocks | **2.4e-06** | agrees to fp32 epsilon |

The DiT figure needed explaining rather than accepting, so the block stack was truncated on both
sides and the divergence measured against depth:

| blocks | 1 | 2 | 4 | 8 | 16 | 28 |
|---|--:|--:|--:|--:|--:|--:|
| relative L2 | 1.3e-6 | 1.4e-6 | 3.2e-6 | 3.1e-5 | 6.6e-5 | 7.3e-4 |

Per block the two agree to fp32 epsilon. The growth is the residual stack amplifying different
reduction orders — the network, not the arithmetic. **This is a measured lower bound on any
workable tolerance**: two correct fp32 implementations already differ by 7.3e-4 after one forward
pass, so a gate set below about 1e-3 would reject a correct implementation. It is recorded in
`eval/cells/BG-1/generation.json` under `tolerance._measured_floor`.

It does *not* establish the gate threshold itself. That is one forward pass in fp32; the scored
path is twenty steps in bf16. The thresholds now in `configs/tolerance.json` are measured rather
than argued, and the measurement produced a result that changed the gate's design — see
**How the gate is actually set** below.

**What was done about the rest of that class.** The remaining intricate oracle details are now
differential-tested against independent implementations written from the reference's published
algorithms rather than from this code, with the expected values committed as golden fixtures in
`tests/test_models.cpp`:

| detail | why it is a trap | agreement |
|:--|:--|:--|
| 2D position embedding | sin-then-cos, meshgrid with x as the first axis, interpolation scale pinned to the checkpoint | to 1e-9 |
| timestep embedding | cos-then-sin — the *opposite* order, in the same model, because of `flip_sin_to_cos` | to 1e-6 |
| T5 relative-position bucketing | enters every layer's scores; off by one shifts every attention distribution | exact, across every boundary |

Two conventions in one model is not a design, it is history, and matching it is not optional.

### On "exercised against a fake device"

`eval/tests/fakes/` holds a stub `nvidia-smi` and a stub runtime that speaks the `BURNISH_JSON`
protocol. `eval/tests/test_device_runners.py` drives `bench.py` and `calibrate.py` through them
end to end, including the closed loop: calibrate, bench, score, receipt.

**The fakes replace the device, not the guards.** The idle check still shells out, still parses,
and still refuses when the stub reports a busy device. The fallback check still compares the
runtime's report against the request. What is removed is the silicon, and that is the only way
code that runs exclusively beside a GPU gets tested at all.

Writing those tests found two defects in the harness: `bench.py` read the whole of `/dev/urandom` (a stream
that never ends) when choosing a held-out shape, and it never produced the
`latent_l2_vs_reference` objective the generation declares — so the frontier would have come out
as exactly zero for both arms and every result would have read `MOVED_ALONG_FRONTIER`.

## How the gate is actually set, because the obvious way does not work

The obvious correctness gate is: run the pipeline, compare the final latent against the
reference, fail if the difference exceeds a threshold. That gate cannot be built at the scored
dtype, and finding out why changed the design.

`eval/cells/BG-1/dtype-cost.json` records the reference implementation compared against
**itself** across dtypes. Worst relative L2: **0.3713**.
The oracle disagrees with itself, at bf16, by far more than a real defect needs to produce. A
threshold above that admits everything; a threshold below it rejects the reference.

So the gate asks two narrower questions instead:

1. **Is the assembly right?** The whole pipeline in fp32 against the fp32 reference latents,
   where the same comparison lands at **0.0005**. The
   threshold is 0.0025 — five times this build's measured drift, not a round
   number.
2. **Is the reduced-precision path right?** Stage by stage at the scored dtype, against per-stage
   tolerances, because a stage's own error is measurable where the composed loop's is not.

Plus determinism — byte-identical replays — which gates both, because a build that does not
reproduce itself cannot be compared to anything.

**One defect was found and deliberately left in.** The sampler quantises an intermediate of
magnitude ~300 to bf16 at sigma=157. Fixing it would make this runtime *more* accurate than the
oracle it is gated against, so the gate would then reject the fix. It is recorded in
`issues/sampler-precision.md` and belongs to a future generation, not to a tolerance edit.

## What is still not known

**0. ~~Whether a score means the same thing on another validator's box.~~ SETTLED, and the fix
changed the design.** The calibration used to be effectively frozen to one card: `achieved` is
`ceiling / measured` and the ceiling was baked into the frozen generation, so scoring another
card's run against it mixed two machines. Measured, a 3% hardware difference produced a **6%**
difference in gap-closed — systematic, in one direction, invisible to the interval.

The ceiling now comes from the validator's own probe, which makes both halves of the ratio scale
with the card and cancel: the same 3% difference produces **0.0000%**. Every validator calibrates
their own box, the scorer refuses a run whose GPU UUID does not match its calibration, and every
receipt records which calibration produced it. `docs/EVAL.md` has the setup.

**0b. How stable a noise floor is between sessions — MEASURED, and it is not very.** Two
calibrations of the same card, hours apart: floors moved by up to **24×** (`t5-encode`, 3.753% →
0.155%), in both directions, while every achieved fraction held to three significant figures. A
median is robust and a spread over nine repeats is not. The consequence is that whether a
submission *resolves* could depend on which session its validator calibrated in, so
`burnish calibrate --merge` folds sessions together keeping the worst floor per cell — which can
only refuse a gain too small to see, never credit noise as a contribution.

**0c. BG-2 exists and is NOT finished.** `eval/cells/BG-2/` declares a 512px generation --
`dit-step/512/bf16` at a 10.58 ms ceiling against 1024px's 54.45 ms, while `t5-encode` is
unchanged at 23.08 ms because the text encoder never sees the image. That asymmetry is the
regeneration property the screen claims, made concrete.

What it does not have: **reference latents, a calibration, or a measured tolerance.**
`configs/tolerance.json` marks it `basis: provisional` and says in as many words that it must not
be used to reject anything. It was created to prove the cartography evaluation path end to end;
the structural half passed on hardware and the measurement half did not complete. Nothing is
scored against BG-2 and nothing should be until `burnish cartography check --measure` finishes on
it.

**1. Whether the fp8 and NVFP4 peaks are reachable.** Those two cells still read
`peak_basis: vendor`. Every implemented cell reads `measured`.

**A correction, from the measurement.** This repository previously stated that a vendor peak is
always optimistic, so every achieved fraction computed against one is a lower bound and the real
room is smaller. That is a guess about direction, and the probe contradicts half of it:

| term | assumed | measured on the part | effect on the published room |
|:--|--:|--:|:--|
| bf16 GEMM | 209.5 TFLOPS | **236.9** | assumed peak was LOW, so room was **understated** — conservative |
| bandwidth | 1792 GB/s | **1506.7** | assumed peak was HIGH, so room was **overstated** — the dangerous direction |

Every BG-1 cell is compute-bound, so the first term governed and the published table was
conservative. That was luck, not design. The honest general statement: **an unmeasured peak makes
the room wrong in an unknown direction, and only a probe settles it.**

**2. What the fp8/NVFP4 cells are actually worth.** Their ceilings are published, and they carry
`implemented: false` and weight 0. A measurement says those ceilings are not collectable yet: one
DiT step costs **3.266 s at fp32** and **3.576 s at
bf16** (0.913x, 5 paired repeats). Halving the weight
traffic made the step *slower*. A path at 1.5%
of its ceiling is bound by neither bytes nor flops, so narrowing its weights moves the published
ceiling down and the measurement not at all. `issues/weight-formats.md` carries this.

**3. How this behaves with many miners submitting at once.** Queueing and scheduling across
submissions are unmeasured. What one submission costs is now measured, and it is the screen
question that fails:

| stage | cost on the pinned part |
|:--|--:|
| gate the base arm | 6.3 min, **cached per base commit** — paid once, not once per PR |
| gate the candidate | 7.2 min |
| paired bench, 3 repeats | 17.1 min |
| **total, typical PR** | **~24 min** |

That was 44 minutes until four things were fixed: the bench was averaging over more invocations
than the floors were calibrated with (a defect, not a setting — see below), the gate regenerated
a prompt the determinism replays had already produced, the base arm's gate was recomputed per
submission although it depends only on the base commit, and the repeat count was five where
three resolves the same verdict.

**`burnish screen` now reports SCORE_COST as FAIL**, at 31 modelled-from-measurement minutes
against a 30-minute budget. It read PASS at 2.5 minutes for as long as the question was answered
from arithmetic — it assumed a first implementation reaches 35% of its roofline, and this one
reaches 1.5%. The assumption was labelled and published the whole time. A clearly-marked
prediction is still a prediction.

It is also the only measurement in this repository that improves without anyone working on it
directly: scoring cost is proportional to how slow the runtime is, so every contribution the
benchmark pays for makes the benchmark cheaper to run.

**4. Whether the roofline is the right ceiling for every cell.** It is an arithmetic bound:
`max(flops/peak, unavoidable_bytes/bandwidth)`. For a cell whose real limit is launch overhead —
which, per (2), `dit-step` currently is — the roofline is correct but distant, and gap-closed
scoring treats a launch-structure win and a tensor-core win as the same currency. That is
intentional, and it is worth knowing it is a choice.

## The first run the instrument ever scored

It was a regression, and it is committed in `examples/` with the raw measurements behind it.

The candidate was `cuda-tile1024` — the same attention kernel with a 1024-wide key tile instead
of 64 — chosen because nobody expected it to win. It didn't: one DiT step went from 3.58 s to
5.69 s. What the receipt did with that is the part worth reading:

```
  status            UNRESOLVED
  gap closed        -0.0056   (credited +0.0000)
  99% interval     [-0.0057, -0.0056]

    dit-step/1024/bf16          -0.0058     1.5% -> 1.0%     0.578%  yes
    t5-encode/1024/bf16         -0.0094    18.1% -> 17.3%    3.753%  yes
    vae-decode/1024/bf16        -0.0000     0.8% -> 0.8%     0.259%   NO
```

Two cells resolved the regression and named it. `vae-decode` differed by about 0.2% against a
0.259% floor and **did not resolve** — not a small effect, *no measurement* — so it contributed
zero without blocking the submission. And the regression credited **zero, not negative**: the
ledger compounds toward a ceiling, and a mechanism that could be pushed backwards would let
anyone move it.

**It also found a bug in the runtime, which is why it was worth running.** The first attempt died
in the gate: an implementation name registered by one op fell back to the *host* kernels for the
other fourteen, so any single-kernel submission — which is what almost every submission is — got
host kernels handed device pointers. That is fixed (`resolve_impl` takes the device, and
`tests/test_ops.cpp` pins that no op resolves to `stock` on a device run). It is exactly the class
of thing a first real run exists to catch, and it would have hit the first miner instead.

## Things that would be easy to get wrong later

- **`peak_vram_bytes` on a CPU build is host RSS.** On a CUDA build it must be the device
  allocator's high-water mark. Scoring the wrong resource would make the entire memory axis
  meaningless and would look completely reasonable.
- ~~The checkpoint tensor names were written from the reference implementation's module
  structure, not checked against the real file.~~ **Now checked**: all 962 names and shapes match
  the pinned revisions, verified by reading the real safetensors headers over HTTP range requests
  (about 1.8 MB, not 22 GB). `configs/checkpoint-layout.json` is the committed record and CI
  re-checks it offline. That check found one real defect on its first run.
- **The 2D position embedding is sin-then-cos and the timestep embedding is cos-then-sin.** Two
  conventions in one model is not a design, it is history. Both are marked `ORACLE` in the source.
- ~~The tolerance in BG-1 is a stated, falsifiable threshold and is expected to move once.~~
  **It moved.** Both thresholds were argued from a single fp32 forward pass and both were wrong;
  `configs/tolerance.json` now carries measured values, the reasoning, and the measurements
  behind them. A tolerance is now a frozen part of a generation: changing one is a new
  generation, not an edit, because receipts already scored against it would otherwise mean
  something different.
- **An implementation name that only one op registers still has to run on the device.** The
  fallback for the other fourteen ops is per-device, and a host baseline under a device run is a
  fault rather than a slow path. This cost the first real scoring run a restart, and
  `tests/test_ops.cpp` now pins it as a property: on a device run, no op resolves to `stock`.
- **A floor and the effect it judges must come from the same instrument.** A cell's noise floor
  is the spread of one procedure — the median of `iters` timed invocations after `warmup`
  untimed ones. The floors were calibrated at 2/5 and the bench ran 3/10 with no flag to change
  it, so every scored run cost twice the GPU time it needed to produce a number quieter than the
  floor it was compared against. `bench.py` now reads both from the calibration and refuses a
  generation that cannot say what measured its floors.
- **A receipt that cannot name the code it scored is not evidence.** `candidate_commit`,
  `base_commit` and `instrument_from` go null whenever the runner has no git metadata — a tarball
  deploy does it. Null reads as "not applicable" to anybody skimming, and the correct reading is
  "unknown", so the receipt now states `code_provenance_complete` outright instead of leaving it
  to be inferred from three absences.
