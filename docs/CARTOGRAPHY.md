# Opening a new cell

A cell nobody has measured yet (a new resolution, dtype or model) is a **paid** contribution,
labelled `burnish:cell-opened`. Finished cells stop paying, so the benchmark has to keep growing.

## What a new cell needs

1. **Geometry.** An op enumeration in `eval/burnscore/geometry.py` that reproduces the model's
   parameter count within 1%.
2. **A reference implementation.** A registered implementation that runs the cell and passes the
   correctness gate against a pinned reference.
3. **A roofline.** Generated from (1) and the device config, never typed.
4. **A calibration.** Achieved fraction and noise floor, from `burnish calibrate` on the pinned
   hardware.

A cell with only (1) and (3) is **declared**: its ceiling is published, marked
`implemented: false`, with weight 0. The fp8 and NVFP4 cells are declared today.

## How it is evaluated

A new cell arrives as a **new generation** under `eval/cells/<name>/`. Existing generations are
frozen.

```bash
burnish cartography check --generation BG-2 --base origin/main --measure \
    --binary build-cuda/burnisher --weights <checkpoint> --noise <noise.npy>
```

- **You supply** the cell definition and the oracle, meaning what a correct runtime must produce.
- **The evaluator supplies every measurement.** Your `reference.json` is read and discarded; the
  cell is gated and calibrated on the validator's box.

Checked before anything runs:
- the generation doesn't already exist on the base;
- it declares at least one cell that doesn't exist yet;
- every ceiling recomputes from the base configs;
- reference latents are present, with a manifest.

Then, on the hardware, the cell must run, reproduce itself byte for byte, pass the gate, and
calibrate.

## Worth knowing

- **A new resolution, dtype or stage** of a model that is already enumerated needs no code.
- **A new model** needs a geometry change in `eval/`. That is the instrument, so it needs a
  maintainer.
- **A cell too noisy to resolve any gain** is still a successful result. Publish it as
  unresolvable and say what would make it measurable.
- **Doesn't count:** adding cells to an existing generation, a guessed calibration, or duplicating a
  held-out shape.
