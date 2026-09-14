# Status

What is real here and what is not. Read this before spending a week on anything.

## Measured on the pinned RTX 5090

| | |
|:--|--:|
| sustained memory bandwidth | 1506.7 GB/s |
| bf16 GEMM through cuBLAS | 236.9 TFLOPS |
| replays of the bf16 CUDA pipeline | byte-identical |
| CUDA stages vs the reference (VAE vs the CPU oracle) | agree to between 1.7e-07 and 5.2e-05 |
| reference latents, 4 prompts, fp32 and bf16 | committed |
| BG-1 fp32 gate drift on a second card | reproduces the anchor card's |

| cell | achieved | room left | noise floor |
|:--|--:|--:|--:|
| `t5-encode/1024/bf16` | 18.2% | 5.5x | 3.753% |
| `dit-step/1024/bf16` | 1.5% | 65.6x | 0.578% |
| `vae-decode/1024/bf16` | 0.8% | 126.4x | 0.845% |

Nine paired repeats per cell, in two sessions. Floors are **not stable between sessions**: the
second moved one by 24×, so each cell's floor is the worst of the two (`docs/EVAL.md`).

## Built and working

- CPU reference ops, CUDA backend for all 15 ops, T5 / PixArt DiT / VAE graphs, DPM-Solver++.
- Checkpoint loading, verified against the real checkpoint's tensor names and shapes.
- Correctness gate, paired bench, calibration, scorer, receipts, ledger, audit, challenge.
- Evaluation in rounds, with the instrument taken from the base commit.

`scripts/check.sh` checks all of it without a GPU.

## Defects found so far

**Five correctness defects in the runtime.** Each gave a deterministic, plausible-looking image,
so self-consistency tests passed every one. Comparing against the reference found them.

1. **Attention read the wrong layout:** `[batch, heads, seq, dim]` over `[batch, seq, heads, dim]`.
2. **The CUDA gather read fp32 token ids as bf16.** Every fp32 test passed.
3. **The sampler used the wrong sigma ratio:** 157 where the reference uses 0.99998 at t=999.
4. **The output patch order was transposed.** Identical mean and std, different arrangement.
5. **Padding was never masked.**

**In the harness:**
- `bench.py` read all of `/dev/urandom` when picking a held-out shape.
- The fidelity objective was never produced, so every result would have read
  `MOVED_ALONG_FRONTIER`.
- Floors and benches used different iteration counts.
- A single-kernel submission fell back to CPU kernels on a GPU run. The first scored run found it.

**Left in on purpose:** the sampler rounds a ~300-magnitude value to bf16. Fixing it would make the
runtime more accurate than the reference, so the gate would reject it
(`issues/sampler-precision.md`).

**The lesson:** when two explanations look the same at full scale, shrink the scale (one step,
one layer) until they separate.

## Not known yet

- **fp8 and NVFP4 cells** have no implementation, and their device peaks are vendor figures, not
  probed. Probing already corrected two assumed peaks, in opposite directions.
- **What narrower weights are worth.** One DiT step costs 3.266 s at fp32 and 3.576 s at bf16.
  Halving the bytes made it slower, so the step isn't limited by memory speed yet.
- **Many submissions at once.** One submission costs ~24 min (gate candidate 7.2, bench 17.1; the
  base gate is cached per commit). `burnish screen` reports SCORE_COST as FAIL, at
  31 modelled-from-measurement minutes against a 30-minute budget. Queueing is unmeasured.
- **BG-2 (512px) is anchored, but its gate is loose.** Its anchor (two sessions), reference
  latents and a measured tolerance are committed. Its fp32 drift against the reference grows
  with steps -- 0.019% at 1 step, 0.093% at 4, 1.36% at 20 -- so the
  trajectory amplifies rounding rather than hiding a defect. The 5x tolerance that follows
  (6.8% relative L2) catches gross defects, not subtle ones.
- **Launch overhead vs raw compute.** The ceiling treats both kinds of win the same. That is a
  choice.

## The first scored run

`cuda-tile1024`, a change nobody expected to win. It was slower, credited zero, correctly left one
cell unresolved, and found the CPU-fallback bug above. `examples/README.md`.

## Easy to get wrong later

- `peak_vram_bytes` is host RSS on a CPU build and must be the device high-water mark on CUDA.
- The position embedding is sin-then-cos, the timestep embedding cos-then-sin. Both are required.
- A tolerance is part of a generation. Changing it means a new generation.
- A noise floor and the bench it judges must use the same warmup and iteration counts.
- A receipt without git metadata sets `code_provenance_complete: false`. It is not evidence.
