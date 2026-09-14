# Burnisher

A native C++/CUDA **image and video generation** runtime for consumer Blackwell — and the
instrument that scores changes to it.

A burnisher is the tool a mezzotint printmaker uses on a plate roughened to solid black — pure
noise — pressing it smooth until the image emerges. It also means to polish by repeated passes,
which is what contributors do here.

```
burnish roofline          # how big every box is, and how full
burnish screen            # which model v0 pins, from arithmetic alone
burnish bench ... | burnish score   # a number with an interval on it
burnish audit             # re-derive somebody else's verdict. no GPU, two seconds
```

**Here to contribute?** Read `CONTRIBUTING.md`. It is the one page you need; everything else is
linked from it when you need it.

---

## Read this first

**The instrument is complete, and it has scored a real run on the pinned hardware.** The runtime
runs end to end on CPU and CUDA, reproduces itself byte for byte, loads the real pinned
checkpoint, and passes the correctness gate against committed reference latents. Every
implemented cell has a measured achieved fraction and its own measured noise floor.

**What that measurement says is that the runtime is very slow, which is the plan.** `dit-step` is
at **1.5%** of its arithmetic ceiling, `vae-decode` at **0.8%**, `t5-encode` at **18.2%**. v0
ships a correct, complete, *slow* pipeline; contributors make it fast and are paid for the
fraction of the remaining gap they close. The kernels are deliberately naive — no tensor cores,
convolution one thread per output element, every GEMM epilogue a separate pass.

**What is not known** is in `docs/STATUS.md`, along with every defect found so far and the first
receipt the instrument ever produced. Overselling the surface is the single failure mode that
kills a subnet, so that page comes before the pitch.

---

## What this is for

Gittensor SN74 pays merged PRs carrying a bot-verified marginal speedup, built from source on
pinned RTX 5090 hardware, correctness gated before speed counts. This repository is being built to
become a second scored target under that mechanism.

[`gittensor-ai-lab/sparkinfer`](https://github.com/gittensor-ai-lab/sparkinfer) is the existing
target: a native C++/CUDA **LLM** runtime for the same hardware. It does not generate images — its
own `docs/image_input.md` says so. Burnisher is the **generation** counterpart, not a competitor:
compute-bound, iterative, fresh shapes per model and resolution. The contributor workflow is
matched deliberately; the scoring is deliberately different.

---

## How a change is scored, in five lines

1. **You are paid the fraction of the remaining gap you close:**
   `(a_candidate − a_base) / (1 − a_base)`, where `a` is how close a cell is to its ceiling.
2. **A gain counts only if it clears that cell's own measured noise floor**, not a fixed 2%.
3. **Faster but hungrier or less faithful pays nothing** — latency, VRAM and fidelity form a
   frontier.
4. **Correctness is gated before anything is timed**, and a held-out shape drawn after your code is
   frozen catches kernels tuned to one shape.
5. **Opening a new cell pays too** — that is cartography.

Why each rule is shaped the way it is: `docs/SCORING.md`. What happens to a pull request, what
every label means, and how to check a verdict yourself: `docs/EVAL.md`. New cells:
`docs/CARTOGRAPHY.md`.

---

## Where the work is

`issues/README.md` — the open items, each carrying its own arithmetic. `burnish roofline` prints
every cell's ceiling and how full it is; `docs/ROOFLINE.md` is the same table, generated.

**Read `issues/weight-formats.md` before picking a quantization cell.** One DiT step costs 3.266 s
at fp32 and 3.576 s at bf16 — halving the weight traffic made it *slower*. At 1.5% of its ceiling
the step is bound by neither bytes nor flops, so an fp8 or NVFP4 cell is worth much less than its
published ceiling until the kernels in front of it improve.

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

Against a real checkpoint, and still without a GPU:

```bash
burnisher check-weights --weights DIR        # all 962 required tensors, through the real loader
scripts/verify_checkpoint_layout.py --against-saved   # names and shapes vs the pinned revisions
scripts/differential_test.py --weights DIR --stage vae-decode   # vs the reference, same weights
```

Everything that produces a **measurement** — `burnish probe | calibrate | gate | bench` — refuses to
run without a device rather than estimating. There is no fallback and there is not supposed to be
one.

---

## How the runtime is arranged

A kernel is added by **registering a new name beside the old one**, never by replacing a file —
`CONTRIBUTING.md` says how, `docs/ARCHITECTURE.md` says why.

```
include/burnisher/     tensor, dtype, op registry, models, scheduler, pipeline
src/cpu/               reference implementations — the correctness ORACLE, not the product
src/cuda/              device allocator and probe (device.cu), and the op backend (ops_cuda.cu)
src/models/            T5 encoder, PixArt DiT, VAE decoder as explicit graphs over the registry
eval/burnscore/        the scorer: geometry, roofline, floor, bootstrap, frontier, receipt, ledger
eval/cells/            the frozen generations: definition, calibration, prompts, receipts
tools/burnish          the harness CLI
issues/                the backlog, with every figure computed from configs/
```

The op sequence in `src/models/pixart_dit.cpp` must match the op enumeration in
`eval/burnscore/geometry.py`; if they drift, the published ceiling stops describing what runs.

---

## Which document answers what

| document | for | answers |
|:--|:--|:--|
| `CONTRIBUTING.md` | contributors | how to pick, write, check and submit a change, and what comes back |
| `issues/README.md` | contributors | what to work on, with the arithmetic for each item |
| `docs/SCORING.md` | contributors, skeptics | why the score is gap-closed, floor-credited and frontier-based |
| `docs/EVAL.md` | contributors, validators | the evaluation loop, labels, rounds, audits, challenges, the instrument guard |
| `docs/CARTOGRAPHY.md` | contributors | what a new cell must come with, and how it is paid |
| `docs/CORRECTNESS.md` | contributors | the correctness gate and its tolerance |
| `docs/ARCHITECTURE.md` | contributors | how the runtime and harness fit, and why |
| `docs/HARDWARE.md` | validators | the measurement box and the order to run things in |
| `docs/STATUS.md` | everyone | what is measured, what is not, and every defect found so far |
| `docs/SCREEN.md`, `docs/ROOFLINE.md` | skeptics | why v0 pins this model, and every ceiling (both generated) |

---

## The rules this repository was built around

Every one was learned by somebody getting it wrong; `docs/STATUS.md` and `docs/HARDWARE.md` say
how.

- **Never publish a predicted figure as a gain.** Modelled output carries `basis: "model"`, and
  `eval/tests/test_schemas.py` fails if one reaches a measured field.
- **The evaluator is where the bugs are.** Never remove a guard without knowing which incident it
  encodes — they are named where they live.
- **Correctness before speed, always.** A submission failing the gate is rejected, not traded off.
- **An axis whose spread sits inside its own noise is open, not solved.**
- **Never type a benchmark number by hand.** `docs/ROOFLINE.md`, `docs/SCREEN.md` and `issues/`
  are generated, and CI fails if any of them is stale.
- **Never run two benchmarks at once, and kill by PID.** `docs/HARDWARE.md`.

---

## Licence

Apache-2.0. The pinned checkpoint is CreativeML Open RAIL++-M (ungated); its T5-XXL encoder is
Apache-2.0 (ungated). Both are fetchable without credentials, which is a screen criterion rather
than a footnote — the benchmark has to be re-runnable by strangers.
