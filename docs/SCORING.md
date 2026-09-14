# How a change is scored

One number: **the fraction of a cell's remaining gap to its ceiling that your change closes**,
credited only if it clears that cell's measured noise floor.

A **cell** is one stage at one shape and dtype, like `dit-step/1024/bf16`.

## The number

```
a = ceiling_seconds / measured_seconds       how close the cell is to its ceiling, 0 to 1
g = (a_candidate - a_base) / (1 - a_base)    your score
```

The **ceiling** is arithmetic: the fastest the card's measured compute and memory speed allow.

**Why not raw percent?** A percentage overpays easy cells:

| change | gap closed |
|:--|--:|
| 20% faster on a cell at 10% of its ceiling | 0.022 |
| 2% faster on a cell at 95% of its ceiling | 0.388 |

- **Gains compound, they don't add.** Two 0.25 wins close 0.4375, not 0.5. A cell's total is 1.0,
  so a nearly finished cell stops paying and opening a new cell pays more.
- **A regression is recorded but credits zero.**

## When a gain counts

- Each cell's **noise floor** is measured by running the unchanged base against itself.
- A gain is credited when the lower end of its 99% bootstrap interval clears that floor.
- A cell that doesn't resolve contributes zero and doesn't block the others.
- `burnish roofline` marks cells with too little room over their floor (`res: no`).

## Faster is not enough

Latency, peak VRAM and fidelity to the reference form a **frontier**.

- Faster but using more memory, or drifting from the reference: `MOVED_ALONG_FRONTIER`, pays
  nothing. The runtime could already make that trade.
- Same speed but less memory or closer to the reference: `EXPANDED_OFF_LATENCY`, pays.
- A crash, OOM, timeout or failed gate is no result at all, never a slow one.

## What cannot be gamed

- **Held-out shape.** Every cell is also run at a shape the evaluator draws after your code is
  frozen (list in `configs/axes.json`). A gain only on the published shape: `SHAPE_OVERFIT`.
- **Partial runs.** Skipping a cell your change hurts credits nothing.
- **The ruler.** The measuring code comes from the base commit, and generations are frozen. See
  `docs/EVAL.md`.
- **Grades.** A schema test fails if a receipt or generation contains `XS`, `XL`, `tier`, `grade`
  or `band`. The number is the result.

## A receipt

```
  status            FRONTIER_EXPANDED
  gap closed        +0.1834   (credited +0.1834)
  99% interval      [+0.1828, +0.1842]
  resolved          True

    dit-step/1024/bf16          +0.1835    55.0% -> 63.3%   0.350%  yes
    t5-encode/1024/bf16         +0.0000    42.0% -> 42.0%   0.600%   NO
```

(Illustrative. A real one is in `examples/`.) Status-to-label mapping: `docs/EVAL.md`.
