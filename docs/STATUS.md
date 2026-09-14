# Status

What is measured, what is built, and what is still unknown.

## Measured on the pinned RTX 5090

| | |
|:--|--:|
| sustained memory bandwidth | 1510.0 GB/s |
| bf16 GEMM through cuBLAS | 243.9 TFLOPS |
| replays of the bf16 CUDA pipeline | byte-identical |
| reference latents, 4 prompts, fp32 and bf16 | committed |

| cell | achieved | room left | noise floor |
|:--|--:|--:|--:|
| `t5-encode/1024/bf16` | 68.7% | 1.5x | 0.293% |
| `dit-step/1024/bf16` | 55.6% | 1.8x | 0.420% |
| `vae-decode/1024/bf16` | 25.4% | 3.9x | 0.064% |

Nine paired repeats per cell in each of two sessions. Floors moved up to 1.6× between the sessions,
so each cell keeps the worse of the two ([`docs/EVAL.md`](EVAL.md#anchoring-a-generation)).

## Against PyTorch on the same card

The same stages in diffusers on PyTorch, eager as installed, at the same shapes and dtype
([`eval/cells/BG-1/pytorch-baseline.json`](../eval/cells/BG-1/pytorch-baseline.json), from [`scripts/pytorch_baseline.py`](../scripts/pytorch_baseline.py)).

| cell | Burnisher | PyTorch | Burnisher's time over PyTorch's |
|:--|--:|--:|--:|
| `t5-encode/1024/bf16` | 33.6 ms | 38.4 ms | 0.9× |
| `dit-step/1024/bf16` | 97.9 ms | 87.7 ms | 1.1× |
| `vae-decode/1024/bf16` | 169.4 ms | 119.0 ms | 1.4× |

PyTorch makes the whole image in 1.92 s. The text encoder is faster than PyTorch's; the denoiser and
the decoder are not yet. The ceiling leaves room past PyTorch: its `dit-step` reaches 62.1% of the
ceiling, this runtime 55.6%. Not measured yet: PyTorch with `torch.compile`.

## Built and working

- CPU reference ops, a CUDA backend for all 15 ops, T5 / PixArt DiT / VAE graphs, DPM-Solver++.
- `cuda` on cuBLAS and cuDNN: fused attention for bf16, cuDNN convolution computed in float, a
  chunked GroupNorm and a caching allocator, checked against the CPU oracle by
  [`tests/test_cuda_ops.cpp`](../tests/test_cuda_ops.cpp).
- Correctness gate, paired bench, calibration, scorer, receipts, ledger, audit and challenge.
- Rounds that take the instrument from the base commit, a copycat guard and a re-registration guard.

[`scripts/check.sh`](../scripts/check.sh) checks all of it without a GPU.

## Cost of scoring

One submission costs about 4 min on a warm gate cache (gate candidate 1.6, bench 1.4, score 1.5; the
base gate, 1.5, is cached per commit), so a two-hour round fits twelve. `tools/burnish screen`
reports SCORE_COST as PASS, at 2 modelled-from-measurement minutes against a 30-minute budget.

## Not known yet

- **fp8 and NVFP4.** No implementation, and their device peaks are vendor figures, not probed.
- **What narrower weights are worth.** One DiT step costs 0.473 s at fp32 and 0.098 s at bf16, but
  the bf16 path also gets fused attention and the tensor cores, so that is not a bandwidth result.
- **BG-2 (512px) catches gross defects only.** Its fp32 drift against the reference grows with the
  step count, 1.36% at 20 steps, so its tolerance (6.8% relative L2) is loose.
- **The guards on real submissions.** The copycat and re-registration guards are tested on
  constructed cases only.
- **The sandbox under a real submission.** The per-round check passes on the evaluator box, but no
  submitted pull request has been built and run through it yet.
- **Many submissions queued at once.**
- **Driver memory above the cache.** `peak_vram_bytes` is what the runtime asked for, not what the
  driver holds on top of it.

## Easy to get wrong

- `peak_vram_bytes` is host RSS on a CPU run and the device allocator's high-water mark on a CUDA
  run. Never compare one with the other.
- The position embedding is sin-then-cos, the timestep embedding cos-then-sin. Both are required.
- A tolerance is part of a generation. Changing it means a new generation.
- A noise floor and the bench it judges must use the same warmup and iteration counts.
- A receipt without git metadata sets `code_provenance_complete: false`. It is not evidence.
