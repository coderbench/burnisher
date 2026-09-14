# The correctness gate

**Correctness comes before speed.** A submission that fails the gate is rejected before anything
is timed. It is never traded against a speedup.

## What it checks

1. **Determinism.** The same build, seed and prompts, replayed ten times, must give
   **byte-identical** latents. A build that can't reproduce itself can't be compared with anything.
2. **Assembly, in fp32.** The whole pipeline is compared with the fp32 reference latents. The
   threshold is five times this runtime's measured drift.
3. **The scored dtype, stage by stage.** Each stage runs at bf16 against the reference at bf16,
   with its own tolerance.

Why not compare the full bf16 run end to end? The reference disagrees **with itself** across
dtypes by more than a real bug would (`eval/cells/BG-1/dtype-cost.json`).

Thresholds and the measurements behind them: `configs/tolerance.json`.

## The oracle

- **Reference implementation:** diffusers, at pinned checkpoint revisions
  (`configs/candidates.json`). Produced by `scripts/make_reference_latents.py`, never by this
  runtime.
- **Prompts:** four frozen prompts (`eval/cells/BG-1/prompts.json`), from 4 to 127 real tokens in a
  300-token window, so padding and masking are always exercised.
- **Token ids** are committed with the tokenizer's digest (`eval/cells/BG-1/token-ids.json`).
- **Reference latents** are committed in fp32 and bf16 under `eval/cells/BG-1/`, with the starting
  noise passed in as an input.
- **Latents, not pixels**, because the VAE decoder is itself being optimized.

## Rules

- **Never widen a tolerance to let a submission through.** A change that computes different
  numbers on purpose (step caching, for example) needs a new generation with its own tolerance.
- **Determinism has no tolerance.** Only the comparison with the reference does.
- **Avoid hidden non-determinism:** per-process autotuning, atomic reductions, TF32 and on-device
  RNG. `-ffast-math` is not used anywhere.

## Run it

```bash
burnish gate --determinism-only --repeats 10 --weights <checkpoint> --noise <pinned noise .npy>
burnish gate --impl <your-impl> --weights <checkpoint> --noise <pinned noise .npy>
scripts/differential_test.py --weights DIR --stage dit-step --resolution 64   # no GPU
```

## What it does not check

- **How the image looks.** Fidelity is a number; drifting inside the tolerance still costs you on
  the frontier (`docs/SCORING.md`).
- **Other shapes.** The held-out shape in `burnish bench` covers those.
