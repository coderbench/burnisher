# Burnisher

A native C++/CUDA **image and video generation** runtime for consumer Blackwell GPUs — and the
instrument that scores changes to it.

v0 runs PixArt-Sigma at 1024px correctly, and slowly on purpose. Contributors make it fast and are
paid for how much of the remaining gap to the hardware's limit they close.

```
burnish roofline      # every cell's ceiling, and how full it is
burnish screen        # which model v0 pins, and why
burnish audit         # re-check anyone's score. no GPU, two seconds
```

**Here to contribute? Read `CONTRIBUTING.md`.**

## Where it stands

- Runs end to end on CPU and CUDA, reproduces itself byte for byte, and passes the correctness
  gate against committed reference latents.
- Measured on an RTX 5090: `dit-step` is at **1.5%** of its arithmetic ceiling, `vae-decode` at
  **0.8%**, `t5-encode` at **18.2%**.
- The kernels are deliberately naive: no tensor cores, convolution one thread per output element,
  every GEMM epilogue a separate pass.

What is measured, what is not, and every defect found so far: `docs/STATUS.md`.

## What it is for

Gittensor SN74 pays merged pull requests that carry a bot-verified speedup. Burnisher is being built
to become a scored target there, for image generation.

## How a change is scored

1. You are paid the **fraction of the remaining gap** to a cell's ceiling that you close.
2. A gain counts only if it clears **that cell's own measured noise floor**.
3. **Faster but hungrier or less faithful pays nothing.**
4. **Correctness is gated first**, and a **held-out shape** catches kernels tuned to one shape.
5. **Opening a new cell pays too.**

Details: `docs/SCORING.md`.

## Quick start

```bash
scripts/build.sh           # CPU build: the correctness oracle and the whole harness
./build/burnisher info     # every registered op implementation
./build/burnisher selftest # the whole graph on synthetic weights
scripts/check.sh           # everything checkable without a GPU
```

With a checkpoint, still without a GPU:

```bash
./build/burnisher check-weights --weights DIR                    # all required tensors load
scripts/differential_test.py --weights DIR --stage vae-decode   # a stage vs the reference
```

Commands that produce a **measurement** (`burnish probe | calibrate | gate | bench`) refuse to run
without a GPU. They never estimate.

## Where the work is

`issues/README.md` lists open work, each item with its arithmetic.

Read `issues/weight-formats.md` before picking a quantization cell: one DiT step costs 3.266 s at
fp32 and 3.576 s at bf16, so narrower weights do not pay until the kernels improve.

## Layout

```
include/burnisher/   tensor, dtype, op registry, models, scheduler, pipeline
src/cpu/             reference ops: the correctness oracle, not the product
src/cuda/            device probe and the CUDA op backend
src/models/          T5 encoder, PixArt DiT, VAE decoder as graphs over the registry
eval/burnscore/      the scorer: geometry, roofline, floor, bootstrap, frontier, receipt, ledger
eval/cells/          frozen generations: definition, calibration, prompts, reference latents
tools/burnish        the harness CLI
issues/              the backlog, generated from configs/
```

## Documents

| document | answers |
|:--|:--|
| `CONTRIBUTING.md` | how to pick, write, check and submit a change |
| `docs/SCORING.md` | how the number is computed |
| `docs/EVAL.md` | what happens to a pull request, labels, audits, and running a validator |
| `docs/CORRECTNESS.md` | the correctness gate |
| `docs/CARTOGRAPHY.md` | how to open a new cell |
| `docs/STATUS.md` | what is measured, what is not, and what went wrong |
| `docs/ROOFLINE.md` | every cell's ceiling (generated) |

## Licence

Apache-2.0. The pinned PixArt-Sigma checkpoint (CreativeML Open RAIL++-M) and its T5-XXL encoder
(Apache-2.0) are both ungated, so anyone can re-run the benchmark.
