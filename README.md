# Burnisher

A native C++/CUDA **image and video generation** runtime for consumer Blackwell GPUs — and the
instrument that scores changes to it.

It runs PixArt-Sigma at 1024px correctly, on cuBLAS and cuDNN. Contributors make it faster and are
paid for how much of the remaining gap to the hardware's limit they close.

```
tools/burnish roofline      # every cell's ceiling, and how full it is
tools/burnish screen        # which model v0 pins, and why
tools/burnish audit         # re-check anyone's score. no GPU, two seconds
```

**Here to contribute? Read `CONTRIBUTING.md`.**

## Where it stands

- Runs end to end on CPU and CUDA, reproduces itself byte for byte, and passes the correctness
  gate against committed reference latents.
- Measured on an RTX 5090: `dit-step` is at **55.6%** of its arithmetic ceiling, `vae-decode` at
  **25.4%**, `t5-encode` at **68.7%**.
- **Against PyTorch on the same card**, which makes the whole image in **1.92 s**
  (`eval/cells/BG-1/pytorch-baseline.json`): `t5-encode` takes **0.9×** PyTorch's time,
  `dit-step` **1.1×** and `vae-decode` **1.4×**.
- `cuda` stands on cuBLAS and cuDNN. What is still unfused is the backlog: every GEMM epilogue is a
  separate pass, and AdaLN modulation is a full round trip.

What is measured, what is not, and every defect found so far: `docs/STATUS.md`.

## What it is for

**A runtime people choose over PyTorch** for image and video generation on consumer Blackwell cards.
Its text encoder already beats PyTorch on the same card; its denoiser and decoder do not yet.
`tools/burnish roofline` shows each cell's gap to PyTorch beside its gap to the ceiling, and a cell
that passes its PyTorch time is a reason to run it here. The ceiling is past PyTorch, so the room
exists.

**The work is paid.** Gittensor SN74 pays merged pull requests that carry a bot-verified speedup,
and Burnisher is being built to become a scored target there. Every contribution it pays makes the
runtime faster for anyone who uses it, whether or not they mine.

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

Make an image from a prompt (CUDA build; the checkpoint with its `<checkpoint>/tokenizer/`):

```bash
scripts/generate.py --weights DIR "a red fox asleep in fresh snow" --out fox.png
```

Commands that produce a **measurement** (`burnish probe | calibrate | gate | bench`) refuse to run
without a GPU. They never estimate.

## Where the work is

`issues/README.md` lists open work, each item with its arithmetic.

Read `issues/weight-formats.md` before picking a quantization cell: one DiT step costs 0.473 s at
fp32 and 0.098 s at bf16. That is more than the halved bytes alone would buy, because the bf16 path
also gets fused attention and the tensor cores.

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
