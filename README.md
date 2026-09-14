# Burnisher

A native C++/CUDA image generation runtime for consumer Blackwell GPUs, and the harness that scores
changes to it. It runs PixArt-Sigma at 1024px on cuBLAS and cuDNN. Contributors make it faster and
are paid for the share of the remaining gap to the hardware's limit that they close.

## Where it stands

On an RTX 5090, against PyTorch on the same card (`eval/cells/BG-1/pytorch-baseline.json`), which
makes the whole image in **1.92 s**:

| stage | share of its ceiling reached | time against PyTorch |
|:--|--:|--:|
| `t5-encode` | 68.7% | **0.9×** |
| `dit-step` | 55.6% | **1.1×** |
| `vae-decode` | 25.4% | **1.4×** |

It passes the fp32 correctness gate and reproduces itself byte for byte. More: `docs/STATUS.md`.

## Quick start

```bash
scripts/build.sh          # CPU build: the correctness oracle and the harness, no GPU
scripts/check.sh          # every check that needs no GPU
scripts/build_cuda.sh     # CUDA build: needs the CUDA toolkit and cuDNN 9
scripts/generate.py --weights DIR "a red fox asleep in fresh snow" --out fox.png
```

`DIR` is the pinned checkpoint, with its tokenizer in `<checkpoint>/tokenizer/`. Commands that
measure (`tools/burnish probe | calibrate | gate | bench`) refuse to run without a GPU.

## Contributing

- **How to submit a kernel:** `CONTRIBUTING.md`.
- **Open work:** `issues/README.md`. Before a quantization cell, read `issues/weight-formats.md`:
  one DiT step costs 0.473 s at fp32 and 0.098 s at bf16.
- **Payment:** Gittensor SN74 pays merged pull requests that carry a bot-verified speedup, and
  Burnisher is being built to be scored there.

## Documents

| document | answers |
|:--|:--|
| `CONTRIBUTING.md` | how to pick, write, check and submit a kernel |
| `docs/SCORING.md` | how a change is scored |
| `docs/EVAL.md` | what happens to a pull request, and how to run a validator |
| `docs/CORRECTNESS.md` | the correctness gate |
| `docs/CARTOGRAPHY.md` | how to add a new cell |
| `docs/STATUS.md` | what is measured and what is not |
| `docs/ROOFLINE.md` | every cell's ceiling (generated) |

## Layout

```
include/burnisher/   tensor, dtype, op registry, models, scheduler, pipeline
src/cpu/             reference ops: the correctness oracle, not the product
src/cuda/            device probe and the CUDA op backend
src/models/          T5 encoder, PixArt DiT, VAE decoder as graphs over the registry
eval/burnscore/      the scorer: geometry, roofline, floor, bootstrap, frontier, receipt, ledger
eval/cells/          frozen generations: definition, anchor, prompts, reference latents
tools/burnish        the harness CLI
issues/              the backlog, generated from configs/
```

## Licence

Apache-2.0. The pinned PixArt-Sigma checkpoint (CreativeML Open RAIL++-M) and its T5-XXL encoder
(Apache-2.0) are both ungated, so anyone can re-run the benchmark.
