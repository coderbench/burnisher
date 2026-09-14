# Status

What is real here and what is not. Read this before spending a week on anything.

## Measured on the pinned RTX 5090

| | |
|:--|--:|
| sustained memory bandwidth | 1510.0 GB/s |
| bf16 GEMM through cuBLAS | 243.9 TFLOPS |
| replays of the bf16 CUDA pipeline | byte-identical |
| CUDA stages vs the reference (VAE vs the CPU oracle), one run, not kept as data | agree to between 1.7e-07 and 5.2e-05 |
| reference latents, 4 prompts, fp32 and bf16 | committed |
| BG-1 fp32 gate drift on a second card | reproduces the anchor card's |

| cell | achieved | room left | noise floor |
|:--|--:|--:|--:|
| `t5-encode/1024/bf16` | 68.7% | 1.5x | 0.293% |
| `dit-step/1024/bf16` | 55.6% | 1.8x | 0.420% |
| `vae-decode/1024/bf16` | 25.4% | 3.9x | 0.064% |

Nine paired repeats per cell, in two sessions, with the vendor kernels as `cuda`. Floors are **not
stable between sessions** -- 1.6× apart here -- so each cell's
floor is the worst of the two (`docs/EVAL.md`).

## Against PyTorch on the same card

`eval/cells/BG-1/pytorch-baseline.json`: the same stages in diffusers on PyTorch, eager as
installed, at the same shapes and dtype, on an RTX 5090 (`scripts/pytorch_baseline.py`).

| cell | Burnisher | PyTorch | Burnisher's time over PyTorch's |
|:--|--:|--:|--:|
| `t5-encode/1024/bf16` | 33.6 ms | 38.4 ms | 0.9× |
| `dit-step/1024/bf16` | 97.9 ms | 87.7 ms | 1.1× |
| `vae-decode/1024/bf16` | 169.4 ms | 119.0 ms | 1.4× |

PyTorch makes the whole image in 1.92 s. **The text encoder is faster than PyTorch's; the denoiser
step and the decoder are not yet**, and a stage that passes its PyTorch time is a reason to run it
here. The ceiling leaves room past that: PyTorch's `dit-step` reaches 62.1% of its ceiling, where
this runtime is at 55.6%.

Not yet measured: PyTorch with `torch.compile`, which is a higher bar and the next row to add.

## Built and working

- CPU reference ops, CUDA backend for all 15 ops, T5 / PixArt DiT / VAE graphs, DPM-Solver++.
- `cuda` on cuBLAS and cuDNN: fused attention for bf16, cuDNN convolution computed in float, a
  chunked GroupNorm, and a caching allocator. `tests/test_cuda_ops.cpp` checks them against the CPU
  oracle on a device.
- Checkpoint loading, verified against the real checkpoint's tensor names and shapes.
- Correctness gate, paired bench, calibration, scorer, receipts, ledger, audit, challenge.
- Evaluation in rounds, with the instrument taken from the base commit.
- A copycat guard against open pull requests that blocks copying accounts.
- A re-registration guard for kernels already on main, tested on the real registry.

`scripts/check.sh` checks all of it without a GPU.

## Not known yet

- **fp8 and NVFP4 cells** have no implementation, and their device peaks are vendor figures, not
  probed. Probing already corrected two assumed peaks, in opposite directions.
- **What narrower weights are worth.** One DiT step costs 0.473 s at fp32 and 0.098 s at bf16,
  4.8× apart where the bytes are 2×: the bf16 path also gets fused attention and the tensor cores,
  so the ratio measures kernels as much as bandwidth. What fp8 and NVFP4 add on top is unmeasured.
- **Many submissions at once.** One submission costs ~4 min on a warm gate cache
  (gate candidate 1.6, bench 1.4, score 1.5; the base gate, 1.5, is cached per commit).
  `tools/burnish screen` reports SCORE_COST as PASS, at 2 modelled-from-measurement minutes against a 30-minute budget.
  Queueing is unmeasured, and the round's slot count still assumes the old cost.
- **BG-2 (512px) is anchored, but its gate is loose.** Its anchor (two sessions), reference
  latents and a measured tolerance are committed. Its fp32 drift against the reference grows
  with steps -- 0.019% at 1 step, 0.093% at 4, 1.36% at 20 -- so the
  trajectory amplifies rounding rather than hiding a defect. The 5x tolerance that follows
  (6.8% relative L2) catches gross defects, not subtle ones.
- **How the copycat guard does on real submissions.** It is tested on constructed cases only. A
  pull request force-pushed with copied code before the evaluator first observed it can still look
  original, and a block is automatic, so a wrong one waits for a maintainer.
- **How the re-registration guard does on real submissions.** It is tested on this repository's
  registry only. A copy that changes a kernel just enough to fall under 95% similarity is clear.
- **Whether the sandbox holds on a rented box.** Submitted code builds and runs as its own account
  (`eval/sandbox.py`). Its tests run without root, so the account switch itself has not run on a
  GPU box; the check before each round is where it first will.
- **Device memory under the cache.** `peak_vram_bytes` is the device allocator's high-water mark and
  the committed anchors record it. Blocks the cache holds are not counted, so it is what the runtime
  asked for; how much the driver holds on top of that is not measured.
- **Launch overhead vs raw compute.** The ceiling treats both kinds of win the same. That is a
  choice.

## Easy to get wrong later

- `peak_vram_bytes` is host RSS on a CPU run and the device allocator's high-water mark on a CUDA
  run. Never compare one with the other.
- The position embedding is sin-then-cos, the timestep embedding cos-then-sin. Both are required.
- A tolerance is part of a generation. Changing it means a new generation.
- A noise floor and the bench it judges must use the same warmup and iteration counts.
- A receipt without git metadata sets `code_provenance_complete: false`. It is not evidence.
