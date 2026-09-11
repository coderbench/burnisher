# What is real in this repository, and what is not

Overselling the surface is the single failure mode that kills a subnet. A contributor who burns a
week of GPU time for a result inside the noise floor does not come back. So this page is the
first one to read and it is deliberately blunt.

## The one-sentence version

**The instrument is complete and tested. The runtime runs end to end on the CPU and reproduces
itself exactly. Nothing has been measured on a GPU, because no Blackwell device and no CUDA
toolkit were available when this was built.**

## What has been built and checked

| thing | state | how you can check it yourself |
|:--|:--|:--|
| The six-question screen | complete, runs | `burnish screen` |
| Per-cell arithmetic rooflines | complete, generated from configs | `burnish roofline` |
| Gap-closed scoring | complete, 79 tests | `python3 -m unittest discover -s eval -t eval` |
| Measured noise floor, paired bootstrap | complete, tested | same |
| Frontier over latency / VRAM / fidelity | complete, tested | same |
| Receipts and the append-only ledger | complete, tested | same |
| Trusted-instrument overlay | complete | `eval/run_from_base.sh` |
| Op registry with named implementations | complete, 4303 assertions | `ctest --test-dir build` |
| CPU reference ops | complete | `burnisher info` |
| T5 encoder / PixArt DiT / VAE decoder graphs | complete | `burnisher selftest` |
| DPM-Solver++ scheduler | complete, pinned against the reference construction | `ctest` |
| End-to-end pipeline, byte-identical replays | complete | `burnisher selftest` |
| Correctness gate (determinism + reference) | complete, exercised against a fake device | `python3 -m unittest discover -s eval -t eval` |
| Paired bench and calibration runners | complete, exercised against a fake device | `python3 -m unittest discover -s eval -t eval` |
| CUDA device probe | **written, never compiled** | CI job `cuda-compile` |
| CUDA op backend | **does not exist** | `issues/cuda-op-backend.md` |
| Checkpoint tensor names and shapes | verified against the pinned revisions, 962/962 | `scripts/verify_checkpoint_layout.py` |
| Checkpoint load path (mmap, shard search, dtypes) | complete, run against real weights | `burnisher check-weights --weights DIR` |
| `burnisher generate` on a real checkpoint | wired; needs token ids and the 22 GB download | `issues/checkpoint-load.md` |
| Reference latents for the gate | **do not exist** | `issues/checkpoint-load.md` |
| Every cell's achieved fraction | **null** | `burnish roofline` |
| Every cell's noise floor | **null** | `burnish roofline` |
| Device peaks behind every ceiling | **vendor, not probed** | `configs/devices.json` |

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
search and the two model graphs against real bytes. It does **not** verify the numerics against
the reference implementation. That is the correctness gate, and it needs reference latents that do
not exist yet — so "the output is plausible" is the strongest claim available here, and it is
weaker than "the output is right".

### Four correctness defects the runtime had, and how they were found

Both were invisible to every self-consistency check in the repository, which is the point worth
recording: a deterministic wrong answer passes a determinism test, and two implementations that
are wrong the same way agree with each other.

**Attention was reading the wrong slices.** The op indexed `[batch, heads, seq, head_dim]` while
every projection GEMM produces `[batch * seq, heads * head_dim]` — which is `[batch, seq, heads,
head_dim]` contiguously. So attention attended over reinterpreted data, consistently and
deterministically, in the DiT, the T5 encoder and the VAE mid-block. The streaming and
materialised implementations agreed with each other because they were wrong identically; the
whole-model determinism test passed because the wrongness was deterministic. It surfaced only
when a cross-attention *mask* test asked a question the layout could not answer. The op is now
head-last and `tests/test_ops.cpp` pins the invariant: H-head attention must equal H independent
single-head attentions over the corresponding slices.

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

Neither would have survived the correctness gate against a reference. Both survived everything
this repository could check without one, which is the argument for getting the reference latents
made.

**And the strongest check now available: stage-by-stage against the reference.**

`scripts/differential_test.py` runs one stage in this runtime and the same stage in diffusers, on
the same weights and the same input tensor, and compares. That is the correctness gate's question
at a scale a CPU can answer, and it is what found the patch-ordering and sampler defects.

**All four stages now agree with the reference.** That is the strongest correctness statement this
repository can make without a GPU, and it is still weaker than the gate: it is one forward pass
per stage, in fp32, at shapes no receipt is scored on. It says the arithmetic is right. It does
not say the twenty-step bf16 loop at 1024px is.

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

It does *not* establish the 2% figure itself. That is one forward pass in fp32; the scored path
is twenty steps in bf16, where both the rounding and the accumulation are larger. 2% remains a
stated threshold expected to move once, when `burnish gate --calibrate-tolerance` runs.

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

## The four things that are not known, stated precisely

**1. How full any cell is.** Every cell publishes an arithmetic ceiling and `achieved: null`. A
ceiling says how big the box is. It says nothing about how full it is, and a cell at 95% of its
ceiling looks identical in the published table to one at 8%. Nothing in this repository should be
read as a claim that there is room in a cell — only that there could be, and at most how much.

**2. What any cell's noise floor is.** The scoring model credits a gain only when it clears that
cell's own measured run-to-run spread. Those spreads are null. Until `burnish calibrate` runs,
the scorer refuses to produce a receipt at all — it raises rather than substituting a constant,
because a guessed floor is precisely the mistake this scoring model exists to replace.

**3. Whether the peaks the ceilings stand on are reachable.** `configs/devices.json` carries
vendor figures with a `confidence` field. No kernel reaches a vendor peak. The direction of the
error matters: **every achieved fraction computed against a vendor peak is a LOWER bound on how
done a cell really is, so the real remaining room is smaller than the table implies, not larger.**
`burnish probe` measures sustained bandwidth and an achievable bf16 GEMM rate on the part and
rewrites the basis to `measured`.

**4. Whether the CUDA code compiles.** It has never been near a compiler. `src/cuda/device.cu` is
the probe and it is the only CUDA in the tree; there is no CUDA implementation of any op, so the
runtime cannot currently run on a device at all. The CI job `cuda-compile` exists to make that
stop being true and is `continue-on-error` today.

## What a first session on the pinned hardware would produce

In this order, because each step unblocks the next:

1. `scripts/build_cuda.sh` — does the probe compile for sm_120? This is the first time anyone
   will know.
2. `burnish probe` — sustained bandwidth and achievable bf16 GEMM rate. Copy into
   `configs/devices.json` with `source: measured`, regenerate the generation and the roofline
   table. **Every ceiling in the repository changes at this point**, and every one of them
   changes in the direction of less room.
3. The CUDA op backend (`issues/cuda-op-backend.md`), then the checkpoint load path
   (`issues/checkpoint-load.md`). Neither is a measurement; both are prerequisites for one.
4. `burnish gate --determinism` — ten replays, byte-identical. If this fails, stop: nothing
   downstream means anything until it passes.
5. `burnish gate` against the pinned reference latents.
6. `burnish calibrate --repeats 9 --write` — the achieved fractions and the noise floors. The
   roofline table becomes real here, and some cells may turn out to be unresolvable, which is a
   result and should be published as one rather than quietly dropped.
7. Only now can `burnish bench` produce a receipt.

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
- **The tolerance in BG-1 is a stated, falsifiable threshold and is expected to move once**, the
  first time `burnish gate --calibrate-tolerance` measures the real bf16-vs-fp32 drift.
