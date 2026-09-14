# Opening a new cell

A cell nobody has measured yet (a new resolution, dtype or model) is a **paid** contribution,
labelled `burnish:cell-opened`. Finished cells stop paying, so the benchmark has to keep growing.

## What a new cell needs

1. **Geometry.** An op enumeration in [`eval/burnscore/geometry.py`](../eval/burnscore/geometry.py) that reproduces the model's
   parameter count within 1%.
2. **A reference implementation.** A registered implementation that runs the cell and passes the
   [correctness gate](CORRECTNESS.md) against a pinned reference.
3. **A roofline.** Generated from (1) and the device config, never typed.
4. **A calibration.** Achieved fraction and noise floor, from `tools/burnish calibrate` on the
   pinned hardware.

A cell with only (1) and (3) is **declared**: its ceiling is published, marked
`implemented: false`, with weight 0. The fp8 and NVFP4 cells are declared today.

## How it is evaluated

A new cell arrives as a **new generation** under `eval/cells/<name>/`. Existing generations are
frozen.

```bash
tools/burnish cartography check --generation BG-3 --base origin/main --measure \
    --binary build-cuda/burnisher --weights <checkpoint> --noise <noise.npy>
```

- **You supply** the cell definition and the oracle, meaning what a correct runtime must produce.
- **The evaluator supplies every measurement.** Your `reference.json` is read and discarded; the
  cell is gated and calibrated on the validator's box.

Checked before anything runs:
- the generation doesn't already exist on the base;
- it declares at least one cell that doesn't exist yet, and none at a held-out resolution
  ([`configs/axes.json`](../configs/axes.json));
- its model is in [`configs/candidates.json`](../configs/candidates.json), and every bf16 pipeline cell's ceiling recomputes from
  the evaluator's configs (other cells are not recomputed yet);
- the oracle is complete: `prompts.json`, `token-ids.json`, and for every prompt a
  `token-ids-<id>.txt` and a reference latent, plus
  `eval/cells/<name>/reference-latents/manifest.json`.

Then, on the hardware, the cell is gated with `cuda` in fp32 and calibrated. It must run,
reproduce itself byte for byte, pass the gate, and calibrate.

## Worth knowing

- **A new resolution** of a model that is already enumerated needs no code: a new generation and
  its own entry in [`configs/tolerance.json`](../configs/tolerance.json). Adding that entry, named after the generation the pull
  request opens, is the one change to [`configs/`](../configs/) the [instrument guard](EVAL.md#the-instrument-guard) allows.
- **A new dtype or stage** also needs a runtime implementation, and today's checks recompute only
  bf16 pipeline ceilings and gate only in fp32. Ask a maintainer first.
- **A new model** needs a geometry change in [`eval/`](../eval/). That is the instrument, so it needs a
  maintainer.
- **A cell too noisy to resolve any gain** is still a successful result. Publish it as
  unresolvable and say what would make it measurable.
- **Doesn't count:** adding cells to an existing generation, a guessed calibration, or a cell at a
  held-out resolution (refused by the check).
