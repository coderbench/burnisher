# The correctness gate

**Correctness comes before speed.** A submission that fails the gate is rejected before anything
is timed. It is never traded against a speedup.

## What it checks

1. **Determinism.** The same build, seed and prompt, replayed, must give **byte-identical**
   latents. `tools/burnish gate` replays 10 times by default; [`eval/score_submission.sh`](../eval/score_submission.sh) uses 2
   per arm.
   A build that can't reproduce itself can't be compared with anything.
2. **Correctness, in fp32.** The whole pipeline, for every frozen prompt, is compared with the fp32
   reference latents. The threshold is five times this runtime's measured drift. Submissions are
   gated at fp32, which is also the gate's default. At any other dtype the gate reports
   `INFORMATIONAL`, never PASS or FAIL: the fp32 threshold says nothing about a bf16 run.

**Not automated yet: the bf16 path.** Comparing a full bf16 run end to end can't gate, because the
reference disagrees **with itself** across dtypes by more than a real bug would
([`eval/cells/BG-1/dtype-cost.json`](../eval/cells/BG-1/dtype-cost.json)). Per-stage bf16 tolerances are recorded in
[`configs/tolerance.json`](../configs/tolerance.json), but nothing runs them yet. [`scripts/differential_test.py`](../scripts/differential_test.py) compares one
stage with the fp32 reference using its own fp32-sized defaults, so to check a stage at bf16 pass
`--dtype bf16` and that stage's recorded tolerance with `--tolerance`.

Thresholds and the measurements behind them: [`configs/tolerance.json`](../configs/tolerance.json).

## The oracle

- **Reference implementation:** diffusers, at pinned checkpoint revisions
  ([`configs/candidates.json`](../configs/candidates.json)). Produced by [`scripts/make_reference_latents.py`](../scripts/make_reference_latents.py), never by this
  runtime.
- **Prompts:** four frozen prompts ([`eval/cells/BG-1/prompts.json`](../eval/cells/BG-1/prompts.json)), from 4 to 127 real tokens in a
  300-token window, so padding and masking are always exercised.
- **Token ids** are committed with the tokenizer's digest ([`eval/cells/BG-1/token-ids.json`](../eval/cells/BG-1/token-ids.json)).
- **Reference latents** are committed in fp32 and bf16 under [`eval/cells/BG-1/`](../eval/cells/BG-1/), with the starting
  noise passed in as an input.
- **Latents, not pixels**, because the VAE decoder is itself being optimized.
- **TF32 off.** The reference runs with TF32 disabled. The committed BG-1 and BG-2 latents reproduce
  byte for byte with it off.

## Rules

- **Never widen a tolerance to let a submission through.** A change that computes different
  numbers on purpose (step caching, for example) needs a new generation with its own tolerance.
- **Determinism has no tolerance.** Only the comparison with the reference does.
- **Avoid hidden non-determinism:** per-process autotuning, atomic reductions, TF32 and on-device
  RNG. `-ffast-math` is not used anywhere.

## Run it

```bash
tools/burnish gate --determinism-only --repeats 10 --weights <checkpoint> --noise <pinned noise .npy>
tools/burnish gate --impl <your-impl> --dtype fp32 --weights <checkpoint> --noise <pinned noise .npy>
scripts/differential_test.py --weights DIR --stage dit-step --resolution 64 --impl <your-impl>
scripts/differential_test.py --weights DIR --stage dit-step --impl <your-impl> --device cuda  # GPU
```

## What it does not check

- **How the image looks.** Fidelity is a number; drifting inside the tolerance still costs you on
  the frontier ([`docs/SCORING.md`](SCORING.md#faster-is-not-enough)).
- **Other shapes.** The held-out shape in `tools/burnish bench` is timed, not compared with the
  reference, so a kernel wrong only at another shape is not caught.
