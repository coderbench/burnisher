# Burnisher

A native C++/CUDA **image and video generation** runtime for consumer Blackwell — and the
instrument that scores changes to it.

A burnisher is the tool a mezzotint printmaker uses on a plate roughened to solid black — pure
noise — pressing it smooth until the image emerges. It is the instrument that removes noise to
produce a picture. It also means to polish by repeated passes, which is what contributors do here.

```
burnish roofline          # how big every box is, and how full
burnish screen            # which model v0 pins, from arithmetic alone
burnish bench ... | burnish score   # a number with an interval on it
```

---

## Read this first

**The instrument is complete and tested. The runtime runs end to end on the CPU and reproduces
itself byte for byte. Nothing has been measured on a GPU.** No Blackwell device and no CUDA
toolkit were available when this was built, so every cell's achieved fraction and noise floor is
`null`, every ceiling stands on a vendor device peak rather than a probed one, and the CUDA op
backend does not exist yet.

`docs/STATUS.md` is the complete list, with what a first session on the pinned hardware would fill
in and in what order. Overselling the surface is the single failure mode that kills a subnet, so
that page comes before the pitch.

---

## What this is for

Gittensor SN74 pays merged PRs carrying a bot-verified marginal speedup, built from source on
pinned RTX 5090 hardware, correctness gated before speed counts. This repository is being built to
become a second scored target under that mechanism.

### Position relative to SparkInfer

[`gittensor-ai-lab/sparkinfer`](https://github.com/gittensor-ai-lab/sparkinfer) is the existing
SN74 target: a native C++/CUDA **LLM** runtime for consumer Blackwell. It does text decode and
prefill, scored on contexts 128/512/4k/16k/32k plus concurrency. It accepts images as *input* via
a vision tower; its own `docs/image_input.md` says "It does not generate images." Its "DFlash
block-diffusion speculative decode" is block diffusion as a *text drafting* strategy — same word,
unrelated mechanism, no pixels.

**SparkInfer is not running out of room**, and this repository does not claim otherwise. Recent
PRs there land 1.2×–1.9× on the concurrency axes. Burnisher is the **generation** counterpart, not
a competitor: a different workload class — compute-bound, iterative, fresh shapes per model and
resolution — that a text runtime structurally cannot reach.

The *shape* is matched deliberately, so a contributor transfers without relearning the workflow:
small native binary, no Python runtime dependency, sm_120/sm_121, RTX 5090 / DGX Spark /
RTX PRO 6000. The *scoring* is deliberately different, and that is the next section.

---

## Scoring: the fraction of the remaining roofline gap you close

Full detail in `docs/SCORING.md`. The short version:

**1. Score is `(a_candidate − a_base) / (1 − a_base)`,** where `a = ceiling / measured` is the
fraction of a cell's arithmetic roofline it achieves. Taking a cell from 40% to 55% closes 0.25 of
what was left. So does taking it from 90% to 92.5%.

This fixes an inverted reward. Under raw percent, a 20% gain on a cell at 10% of roofline outscores
a 2% gain on a cell at 95% — the first is ordinary, the second is extraordinary. Here:

| change | gap closed |
|:--|--:|
| 20% faster, cell at 10% of roofline | 0.022 |
| 2% faster, cell at 95% of roofline | **0.388** |

It also self-terminates. A cell's total closable gap is 1.0, the ledger compounds toward it, and
the physically available speedup shrinks to `1/achieved`. Grinding an exhausted cell stops paying
and opening a new one starts paying more — so axis supply becomes an incentive rather than an
admin chore.

**2. Credit against the measured noise floor, not a constant.** Each cell publishes its own
run-to-run spread, measured by repeated paired control runs, and a gain is credited only when it
clears that floor under a paired bootstrap at a stated confidence. A 2% threshold is a guess at the
noise: it throws away a real 0.5% gain on a quiet cell and pays for a meaningless 3% on a noisy one.

The floor is published **in the currency of the score**, because the same 1% noise means completely
different things in different cells:

| cell at | floor | as gap-closed | floors of room |
|--:|--:|--:|--:|
| 40% of roofline | 1.0% | 0.007 | 148 |
| 95% of roofline | 1.0% | 0.193 | 5.2 |

**3. No letter grades.** A receipt reports gap closed, the interval, the cell's floor, whether it
resolved, and the frontier position. A number with an interval cannot be argued into a higher
bucket. A schema test fails if `XS`, `XL`, `tier`, `grade` or `band` appears anywhere.

**4. Both objectives count.** Latency, peak VRAM and output fidelity form a frontier. Faster but
4 GB hungrier is `MOVED_ALONG_FRONTIER` and credits nothing — it is a trade the runtime could
already make. So is staying inside the correctness tolerance while measurably degrading.

**5. Cartography pays.** Adding a cell nobody had measured, with its reference implementation and
its calibration, is a scored contribution. `docs/CARTOGRAPHY.md`. Finding that a cell *cannot*
resolve a contribution is also a successful one.

**6. Anti-gaming, non-negotiable.** Held-out shapes drawn at evaluation time from the base commit
after the candidate is frozen; the instrument overlaid from the base commit so a submission cannot
edit the ruler; an append-only ledger written outside the candidate's reach; a partial matrix
credits nothing; a frozen generation cannot be edited.

---

## What v0 pins, and why

`burnish screen` answers six questions from config files and arithmetic, before downloading any
weights. The answers are written up in `docs/SCREEN.md`. It picks:

**PixArt-Sigma XL-2 at 1024px**, 20 steps, CFG, DPM-Solver++ 2M. A real DiT (4096 image tokens at
1024px), a T5-XXL text encoder eight times the size of the denoiser, an SD-family VAE. Ungated and
redistributable — which ruled out FLUX.1-schnell, whose Apache-2.0 weights sit behind a
HuggingFace gate that returns 401 to `curl`.

| stage | runs | ceiling | share | resident params |
|:--|--:|--:|--:|--:|
| `t5-encode` | 1 | 26.9 ms | 2.0% | 9.53 GB |
| `dit-step` | 20 | 63.4 ms | 94.3% | 1.22 GB |
| `vae-decode` | 1 | 50.1 ms | 3.7% | 0.10 GB |

**The text encoder is 89% of the checkpoint and 2% of the clock.** It runs once; the DiT runs
twenty times. A screen that ranked stages by parameter count — the natural thing to do — would send
a contributor to the biggest weights in the model and a fiftieth of the wall time. It is a
*memory-axis* cell instead: 9.53 GB idle on a 32 GiB card for the whole denoise loop, and the
frontier scores peak VRAM.

---

## The roofline table

`burnish roofline` publishes, per `(stage, shape, dtype)` cell:

```
  cell                       runs    ceiling   bound       ai  fuse  achieved    left   floor  res
  t5-encode/1024/bf16           1    26.87ms compute      607  1.03        --      --      --   --
  dit-step/1024/bf16           20    63.40ms compute    10864  1.12        --      --      --   --
  vae-decode/1024/bf16          1    50.06ms compute    99505  1.23        --      --      --   --
  dit-step/1024/fp8            20    31.70ms compute    21715  1.23        --      --      --   --
  dit-step/1024/nvfp4          20    15.85ms compute    38565  1.49        --      --      --   --
```

`ceiling` is arithmetic: `max(flops / peak, unavoidable_bytes / bandwidth)`. `unavoidable_bytes` is
weights-read-once plus stage input plus stage output, excluding every intermediate — because an
intermediate is removable by fusion, and **a ceiling that moved when you fused would not be a
ceiling**. `fuse` is the sum of per-op bounds over the whole-stage bound, which is what fusion is
worth before anybody writes a kernel.

`achieved`, `left` and `floor` are **measurements**, and they print `--` because nobody has taken
them. A cell at 95% of its ceiling looks identical here to one at 8%. That is stated on the table
itself, not buried.

---

## Quick start

```bash
scripts/build.sh          # CPU build: the correctness oracle and the whole harness
./build/burnisher info    # every registered op implementation in this build
./build/burnisher selftest# the whole graph, end to end, on synthetic weights
scripts/check.sh          # everything checkable without a GPU
burnish roofline          # the published ceiling table
burnish screen            # why v0 pins what it pins
```

Everything above needs no GPU. Everything that produces a **measurement** —
`burnish probe | calibrate | gate | bench` — refuses to run without a device rather than
estimating. There is no fallback and there is not supposed to be one.

---

## How the runtime is arranged

A contributor adds a kernel by **registering a new name beside the old one**, never by replacing a
file:

```cpp
register_impl<AttentionArgs>("attention", "flash-sm120", my_kernel, "…");
```

So base and candidate run in one process, one model load, one thermal state; the old
implementation stays runnable forever; and `--impl <name>` fails loudly if the name is not
registered rather than silently measuring something else. The harness compares the impl the
runtime *reports* against the one it asked for and refuses a run that fell back.

```
include/burnisher/     tensor, dtype, op registry, models, scheduler, pipeline
src/cpu/               reference implementations — the correctness ORACLE, not the product
src/cuda/              device probe. The op backend does not exist yet (issues/cuda-op-backend.md)
src/models/            T5 encoder, PixArt DiT, VAE decoder as explicit graphs over the registry
eval/burnscore/        the scorer: geometry, roofline, floor, bootstrap, frontier, receipt, ledger
eval/cells/BG-1/       the frozen generation: definition, calibration, prompts, receipts
tools/burnish          the harness CLI
issues/                the backlog, with every figure computed from configs/
```

The op sequence in `src/models/pixart_dit.cpp` is meant to be read side by side with the op
enumeration in `eval/burnscore/geometry.py`. If they drift, the published ceiling stops describing
the thing that runs and nothing else would notice.

---

## Where the work is

`issues/README.md` — twelve items, each carrying its own arithmetic. The two that block everything:

- **`cuda-op-backend`** — there is no CUDA implementation of any op, so nothing can be measured.
- **`checkpoint-load`** — the load path and the pinned reference latents do not exist yet. (The
  962 tensor names and shapes the runtime requires *are* verified against the pinned revisions.)

Then: DiT attention at 4k–16k tokens (fp8/NVFP4), VAE decode tiling and the 16384-token mid-block
attention, T5 quantization and caching, fused AdaLN, weight formats on silicon with no reference
tuning, step caching, CUDA-graph capture of the 571-launch denoise loop, offload and streaming,
per-resolution shape specialisation, temporal and sparse attention for video.

---

## The rules this repository was built around

Every one was learned by somebody getting it wrong.

- **Never publish a predicted figure as a gain.** Cost-model output carries `basis: "model"`, and
  `eval/tests/test_schemas.py` fails if a modelled figure reaches a measured field.
- **The evaluator is where the bugs are.** A broken evaluator prints a confident number. Never
  remove a guard without knowing which incident it encodes — they are all named where they live.
  (Writing the tests for this one found four real defects: unresolved cells blocking a
  submission, a pooled frontier that made improvements invisible, a read of the whole of
  `/dev/urandom`, and a bench runner that never produced one of the three objectives it is
  scored on.)
- **An axis whose spread sits inside its own noise is open, not solved.**
- **Correctness before speed, always.** A submission failing the gate is rejected, not traded off.
- **Never type a benchmark number by hand.** Every figure in `docs/ROOFLINE.md`, `docs/SCREEN.md`
  and `issues/` is generated; CI fails if any of them is stale.
- **Check the assumption rather than restating it.** The runtime's 962 required tensor names were
  written from the reference implementation's module structure — usually right, and not evidence.
  Reading the real checkpoint's safetensors headers by HTTP range request (1.8 MB, not 22 GB)
  turned that into evidence and found a wrong shape declaration on the first run.
- **Never run two benchmarks at once.** They race for VRAM and the harness turns the loser into a
  plausible-looking number.
- **Clocks cannot be pinned in a container**, so only paired interleaved same-box deltas mean
  anything.
- **Kill by PID.** `pkill -f` over ssh kills your own session.

---

## Licence

Apache-2.0. The pinned checkpoint is CreativeML Open RAIL++-M (ungated); its T5-XXL encoder is
Apache-2.0 (ungated). Both are fetchable without credentials, which is a screen criterion rather
than a footnote — the benchmark has to be re-runnable by strangers.
