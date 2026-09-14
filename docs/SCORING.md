# How a change is scored

One number: **the fraction of a cell's remaining gap to its ceiling that your change closes**,
credited only if it clears that cell's measured noise floor.

A **cell** is one stage at one shape and dtype, like `dit-step/1024/bf16`.

## The number

```
a = ceiling_seconds / measured_seconds       how close the cell is to its ceiling, 0 to 1
g = (a_candidate - a_base) / (1 - a_base)    your score
```

The **ceiling** is arithmetic: the fastest the pinned card's measured compute and memory speed
allow. A generation is anchored once, which fixes each cell's achieved fraction; a run contributes
only its paired base/candidate ratio, so any card of the pinned class gives the same score
(`docs/EVAL.md`).

**Why not raw percent?** A percentage overpays easy cells:

| change | gap closed |
|:--|--:|
| 20% less time on a cell at 10% of its ceiling | 0.028 |
| 2% less time on a cell at 95% of its ceiling | 0.388 |

- **Gains compound, they don't add.** Two 0.25 wins close 0.4375, not 0.5. A cell's total is 1.0,
  so a nearly finished cell stops paying and opening a new cell pays more.
- **The score is a weighted mean over the cells.** A cell that resolved counts at its measured
  value, **a regression included**, so a win in one cell has to pay for a real loss in another. A
  cell that did not resolve counts as zero.

## When a gain counts

- Each cell's **noise floor** is measured by running the unchanged base against itself. Floors move
  between sessions, so the one kept is the worst measured.
- A cell **resolves** when its 99% bootstrap interval clears that floor, in either direction.
- The submission resolves when, over the cells that resolved, the weighted interval clears their
  weighted floor. A cell that doesn't resolve contributes zero and doesn't block the others.
- Not resolved: `UNRESOLVED`. Resolved and not better: `NO_GAIN`.
- `tools/burnish roofline` marks cells whose remaining room is under 20 times their floor
  (`res: no`). A gain there has to be large to show.

## Faster is not enough

Latency, peak memory and fidelity to the reference form a **frontier**, computed per cell and
averaged by weight. Peak memory is the device allocator's high-water mark on a CUDA run.

- A resolved gain that also grows the frontier: `FRONTIER_EXPANDED`, **paid**.
- A resolved gain whose frontier shrank, because memory or fidelity got worse by more than the speed
  was worth: `MOVED_ALONG_FRONTIER`, not paid. Speed can outweigh a small memory cost: each cell's
  latency axis runs from 1.5 times its base time to its ceiling.
- Less memory or closer to the reference without a resolved speedup: not paid. Only latency
  resolves, so this usually reads `UNRESOLVED`. `EXPANDED_OFF_LATENCY` is the rare resolved
  slowdown whose frontier still grew.
- A crash, OOM, timeout or failed gate is no result at all, never a slow one.

## What cannot be gamed

- **Held-out shape.** Every cell is also run at a resolution the evaluator draws after your code is
  frozen (`held_out.resolutions` in `configs/axes.json`). Every cell whose gain resolved must still
  be faster there by more than its own noise floor, or the result is `SHAPE_OVERFIT`.
- **Partial runs.** Skipping a cell your change hurts credits nothing (`PARTIAL`).
- **Copies.** New code copied from someone else's open pull request is not paid (`docs/EVAL.md`).
- **The ruler.** The measuring code comes from the base commit, and generations are frozen. See
  `docs/EVAL.md`.
- **Grades.** A schema test fails if a receipt or generation contains `XS`, `XL`, `tier`, `grade`
  or `band`. The number is the result.

## A receipt

```
  status            FRONTIER_EXPANDED
  gap closed        +0.1730   (credited +0.1730)
  99% interval      [+0.1690, +0.1771]
  resolved          True

    dit-step/1024/bf16          +0.1835     1.5% -> 19.6%   0.578%  yes
    t5-encode/1024/bf16         +0.0011    18.2% -> 18.3%   3.753%   NO
    vae-decode/1024/bf16        -0.0002     0.8% -> 0.8%    0.845%   NO
```

(Illustrative, with BG-1's cell weights: the untouched cells count as zero. A real receipt is in
`examples/`.) Status-to-label mapping: `docs/EVAL.md`.
