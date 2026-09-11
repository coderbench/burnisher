# How a submission is scored

The short version: **score is the fraction of a cell's remaining roofline gap that a change
closes, credited only when it clears that cell's own measured noise floor.** No letter grades.

## Why not raw percent

SparkInfer, the existing SN74 target, scores a raw marginal speedup and rounds it into
XL/L/M/S/XS, discarding anything under 2%. That works, and it has three problems this repository
was built to fix. They are worth stating precisely because the fixes only make sense against them.

**Raw percent pays the easy cells.** A 20% gain on a cell sitting at 10% of its roofline is
ordinary work — there was a factor of ten lying around. A 2% gain on a cell already at 95% is
extraordinary, and under a percentage regime it gets a *smaller* label. The reward is inverted.

**A fixed threshold is a guess at the noise.** On a quiet cell a 0.5% gain is real and gets
thrown away. On a noisy one a 3% gain is nothing and gets paid. The threshold is a proxy for a
quantity you can simply measure, and measuring it costs one calibration run.

**Discrete labels round away information** and turn every borderline result into an argument
about which side of a boundary it landed on.

## 1. The score: fraction of the remaining gap

Every cell publishes an arithmetic ceiling and the fraction of it currently achieved:

```
a = ceiling_seconds / measured_seconds          the achieved fraction, in (0, 1]
g = (a_candidate - a_base) / (1 - a_base)       the score
```

Taking a cell from 40% to 55% of roofline closes `(0.55-0.40)/(1-0.40) = 0.25` of what was left.
So does taking it from 90% to 92.5%. That is the whole idea: **scale-free, comparable across
cells, and correctly harder near the ceiling.**

Run the comparison the brief's framing asks for and the inversion is gone:

| change | achieved | gap closed |
|:--|:--|--:|
| 20% faster, cell at 10% of roofline | 10% → 12% | 0.022 |
| 2% faster, cell at 95% of roofline | 95% → 96.9% | 0.388 |

A seventeen-fold difference, in the right direction. `eval/tests/test_scoring.py` asserts it.

### It self-terminates, and here is how it actually does that

The score does not decay per PR — the same *fraction* of remaining gap always pays the same,
which is the point. Three other things terminate a cell:

- **The total is bounded.** A cell's whole remaining gap is 1.0, and the ledger compounds toward
  it: two contributions of 0.25 leave `0.75 × 0.75 = 0.5625` remaining, so 0.4375 closed, not 0.5.
  Summing would let a generation claim more gap closed than existed.
- **The physics runs out.** At 95% of roofline no submission can ever be worth more than a
  1.053× speedup there, however clever.
- **The floor eats the gap.** This is the one that bites first, and it is the next section.

So grinding an exhausted cell stops paying and opening a new one starts paying more. The
axis-supply problem becomes an incentive rather than an admin chore.

## 2. Credit against the measured floor, not a constant

Each cell publishes its own run-to-run spread, measured by **repeated paired control runs** — two
arms, both the unmodified base, interleaved with every guard a scored comparison uses. The floor
is the larger of that spread and the instrument's own resolution, so a cell whose repeats happened
to agree cannot publish a floor of zero and then accept anything.

The floor is also published **in the currency of the score**, because a percentage of runtime and
a gap-closed number are not comparable and treating them as if they were is the original mistake
in a new form:

| cell at | floor | as gap-closed | floors of room |
|--:|--:|--:|--:|
| 40% of roofline | 1.0% | 0.0068 | 148 |
| 95% of roofline | 1.0% | 0.1925 | 5.2 |

The same 1% noise is trivial in one cell and eats a fifth of the remaining gap in the other. A
contributor reading only the percent would pick the wrong cell. `burnish roofline` prints the
`res` column — does this cell's remaining room clear 20× its own floor? — and a cell that fails it
is published as unresolvable rather than as a place to work.

**Two gates, not one.** A result is credited when the paired bootstrap's lower bound clears the
weighted floor. An interval alone does not say an effect is bigger than the noise — with enough
repeats a tiny thermal bias becomes statistically significant — and a floor alone does not say
the effect is real. Both, or `resolved: false`, said plainly rather than discarded silently.

A cell whose own result does not resolve contributes **zero** and does not block the submission.
"We cannot tell" and "nothing happened" are different, and an untouched cell can never resolve
because there is nothing there to resolve.

## 3. No letter grades

A receipt reports gap closed, the confidence interval, the cell's floor, whether the result
resolved, and the frontier position. A number with an interval beats a letter, and it cannot be
argued into a higher bucket. `eval/tests/test_schemas.py` fails if `XS`, `XL`, `impact_band`,
`tier`, `grade` or `band` appears anywhere in a receipt or a generation.

The `status` field describes evaluation **state**, never magnitude:

| status | meaning |
|:--|:--|
| `FRONTIER_EXPANDED` | resolved gain, and the frontier grew. Credits. |
| `MOVED_ALONG_FRONTIER` | resolved latency gain, paid for in memory or fidelity. Credits nothing. |
| `EXPANDED_OFF_LATENCY` | no latency gain, but the frontier grew. Credits. |
| `UNRESOLVED` | inside the noise. We do not know. |
| `NO_GAIN` | resolved, and not an improvement. |
| `SHAPE_OVERFIT` | the gain did not survive the held-out shape. |
| `PARTIAL` | incomplete matrix. Credits nothing. |
| `CORRECTNESS_FAIL` / `DETERMINISM_FAIL` / `BUILD_FAIL` | rejected before speed was considered. |

## 4. Both objectives count

Latency, peak VRAM and output fidelity form a frontier. Each arm's operating configurations become
points in a normalized higher-is-better space; the non-dominated subset is its frontier; the
volume that frontier dominates is compared.

A change that is 10% faster and needs 4 GB more has not made the runtime better — it has picked a
point on a trade-off that was already available. It scores `MOVED_ALONG_FRONTIER` and credits
nothing. Same for one that stays inside the correctness tolerance while measurably degrading:
`latent_l2_vs_reference` is a scored objective with the tolerance as its zero, so "inside the gate
but worse" is visible as a smaller number rather than invisible as a pass.

**A failure is the absence of a point, never a bad one.** An OOM, a timeout, a blown tolerance or
a degenerate output must never normalize into a small positive score that still contributes
volume. A candidate that drops one of the base's frontier points loses that volume.

Bounds are frozen per generation. Normalizing against today's best makes every historical receipt
mean something different the moment a new PR lands, and a ledger whose past entries silently
change is not a ledger.

Latency is normalized **per cell** — the good end is that cell's own ceiling, the zero end is 1.5×
its calibrated time. Pooling every cell into one bracket makes the latency axis nearly constant
and hands the whole frontier to memory and fidelity. That is not hypothetical; it is what this
code did until the test for a clean win caught it.

## 5. Cartography pays

A contributor who adds a cell nobody had measured — a new model, resolution or dtype — and lands
its reference implementation and its calibration earns credit for it. See `docs/CARTOGRAPHY.md`.
The subnet's health depends on axis supply, and the scoring model above makes exhausted cells stop
paying; if supplying new ones is an unpaid chore, the benchmark stops growing exactly when it
needs to.

## 6. Anti-gaming

Non-negotiable, and each is enforced by code rather than by policy:

- **Held-out shapes per cell family.** Every cell is scored on its published shape and on a shape
  the evaluator draws at run time, from the base commit, after the candidate is frozen. A kernel
  fast only on the benchmarked shape is reported `SHAPE_OVERFIT` and credits nothing. The held-out
  values are listed in the open in `configs/axes.json`; hiding them would not help, and what makes
  the guard work is that the candidate cannot know which is drawn.
- **The instrument comes from the base commit.** `eval/run_from_base.sh` overlays `eval/`,
  `configs/`, `schemas/` and `tools/burnish` from the base ref before scoring anything. A one-line
  change to a noise floor, a confidence level, a ceiling, a tolerance or the model revision does
  not look like cheating in a diff — several of them look like tidying.
- **The ledger is append-only and written outside the candidate's reach.** A finalized receipt is
  never rewritten; a correction is a new receipt naming what it supersedes and why.
- **A partial matrix credits nothing.** Dropping the cell a change hurts is the cheapest way to
  raise a score, and a rule that stops a drop from *paying* removes the incentive rather than
  policing it.
- **A frozen generation cannot be edited.** If the meaning of the evaluation should change, the
  answer is a new generation. Receipts stay attached to the generation that produced them.

## What a receipt looks like

```
  status            FRONTIER_EXPANDED
  gap closed        +0.1834   (credited +0.1834)
  99% interval      [+0.1828, +0.1842]
  resolved          True
  frontier          +0.03399 (expanded)

  per cell:
    cell                            gap           achieved   floor  res
    dit-step/1024/bf16          +0.1835    55.0% -> 63.3%   0.350%  yes
    t5-encode/1024/bf16         +0.0000    42.0% -> 42.0%   0.600%   NO
    vae-decode/1024/bf16        +0.0000    18.0% -> 18.0%   0.900%   NO
```

Gap closed, the interval, each cell's floor, whether it resolved, and where the frontier moved.
Every figure generated from the receipt, and the receipt from raw measurements that are themselves
in the receipt — so it can be recomputed by anyone, years later, without the box that produced it.
