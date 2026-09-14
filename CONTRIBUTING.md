# Contributing

The deal: **land a kernel, get a number with an interval on it.** No letter grades, no arguing
about which bucket a result fell into, and no maintainer discretion over the score.

## Before you pick something

1. Read `docs/STATUS.md`. It says what has been measured and what has not.
2. Run `burnish roofline`. It prints every cell's arithmetic ceiling and how full it is. Cells
   with `achieved: --` have not been measured -- today the fp8 and NVFP4 cells, which have no
   implementation yet -- so nobody knows how much room they have.
3. Read `issues/README.md`. Every item carries its own arithmetic, computed from `configs/`.
4. Check the `res` column. A cell whose remaining room does not clear 20× its own noise floor
   **cannot be shown to have improved**, however good your kernel is. Do not start there.

## The rule that shapes the whole runtime

**Register a new name beside the old one. Do not replace a file.**

```cpp
register_impl<AttentionArgs>("attention", "flash-sm120", my_kernel,
                             "tiled online-softmax for sm_120, 128x64 blocks");
```

Three things follow, and none of them is available if you replace a file:

- Base and candidate run in **one process, one model load, one thermal state**, so the paired
  delta is between two kernels rather than between two program startups.
- The old implementation stays runnable forever, so a regression bisects to a kernel rather than
  to a commit range.
- `burnisher bench --impl <name>` **fails loudly** if the name is not registered, instead of
  silently measuring whatever the build happened to contain. The harness also compares the impl
  the runtime *reports* against the one it asked for, and refuses a run that fell back.

`burnisher info` lists everything a build contains.

## Before you open a PR

```bash
scripts/check.sh
```

That runs the C++ tests, the harness tests, the whole graph on synthetic weights, the generation
consistency check, the roofline regeneration check, `--help` on every entry point, and the
manifest. It needs no GPU.

If you touched anything that computes — a kernel, a graph, a layout — also run the stage-by-stage
comparison against the reference implementation. It needs a checkpoint but no GPU, and it is the
only check here that has a second opinion:

```bash
scripts/differential_test.py --weights DIR --stage vae-decode
scripts/differential_test.py --weights DIR --stage dit-step --resolution 64
```

A disagreement far above the tolerance with an **identical mean and standard deviation** is a
permutation, not an arithmetic error. That is how the output patch ordering was found, and no
self-consistency test in this repository could see it.

Then, on the pinned hardware, one command:

```bash
eval/score_submission.sh --base <the commit you branched from> --worktree . \
    --impl-base cuda --impl-candidate <your-impl> \
    --pr <n> --ledger <a directory OUTSIDE this worktree> \
    --weights <checkpoint dir> --noise <the pinned noise .npy>
```

That runs the four stages in the order they have to happen in: gate the base arm, gate yours,
bench both paired and interleaved, score into the ledger. It is a script rather than a list of
instructions because two people following the same prose write two different drivers, and the
difference shows up as a difference in scores that no receipt can explain.

The stages underneath it are `burnish gate | bench | score` and you can run them by hand while
iterating. Two things about the order are not negotiable:

- **`burnish bench` refuses to time a build that has not passed the gate.** Correctness precedes
  speed and is never traded against it.
- **The base arm is gated too.** A baseline that does not reproduce itself makes every delta
  measured against it noise.

**One name, fifteen ops.** `--impl-candidate <your-impl>` applies to whichever ops register that
name; the rest fall back to the baseline for the device the run is placed on. Registering one
kernel is the normal case, not an edge case. The runtime reports the resolved implementation for
every op and the harness refuses a run whose report disagrees with what was asked for, so a
fallback you did not intend shows up as a rejected run rather than a wrong number.

## What your pull request gets back

One label, and a comment saying how it was reached. A paying outcome carries **the number**:

```
burnish:gap+0.0342
```

— the fraction of this generation's *remaining* arithmetic-roofline gap that your change closed.
That is the payout basis itself. There are no XS/S/M/L/XL tiers here: a bucket boundary pays two
differently-measured submissions the same and two almost-identical ones differently, and the
number is already comparable across cells, models and hardware.

**You can check the verdict yourself, with no GPU, in about two seconds.** The measurements are
published next to the receipt and the verdict is a pure function of them:

```bash
burnish audit pr-000042-raw.json pr-000042.json
```

If you think the measurement itself is wrong, re-run it on your own 5090 and file a
counter-receipt with `burnish challenge`. A disagreement beyond the cell's own noise floor puts
the credit on hold rather than paying it or withdrawing it. `docs/EVAL.md` has the whole loop.

| outcome | cause |
|:--|:--|
| `burnish:determinism-fail` | replays of your build were not byte-identical. Nothing can be attributed to a change against a baseline that does not reproduce itself. |
| `burnish:correctness-fail` | the latents moved outside the stated tolerance. A rejection, not a trade-off, and widening the tolerance is not the fix. |
| `burnish:shape-overfit` | the gain vanished on a held-out shape the evaluator drew after your code was frozen. A kernel fast only on the benchmarked shape is a tuned constant. |
| `burnish:partial` | you ran some cells and not others. Dropping the cell a change hurts is the cheapest way to raise a score. |
| `burnish:unresolved` | the effect is inside the cell's own measured noise floor. You get the number and the interval; you do not get credit — and this is not a judgement about the idea. |
| `burnish:moved-along-frontier` | faster, but it cost memory or fidelity. That is a trade the runtime could already make. |
| `burnish:skipped-instrument` | it changes the measuring instrument, so it was not evaluated and no GPU time was spent on it. Not a rejection — see below. |

## Commit messages

One line. No body. No trailers. Ever.

```
<type>: <short imperative description>
```

`type` is one of `feat`, `fix`, `perf`, `docs`, `test`, `build`, `refactor`, `chore`. Subject 72
characters or fewer. If the change needs explanation it goes in the code, the docs, or the PR
description.

```
perf: fuse AdaLN into the modulation kernel
fix: the VAE tiler dropped the last row at odd heights
docs: state the measured floor for every published cell
```

## Changing the instrument — and the one exception that is paid

`eval/`, `configs/`, `schemas/` and `tools/burnish` are the **instrument**. Two mechanisms guard
them, because one is not enough: `eval/run_from_base.sh` overlays them from the base commit
before scoring, so an edit cannot affect its own author's score; and a required CI check blocks
the merge, because a change that lands on main becomes the instrument for everybody after it.

**Opening a new cell is the exception, and it is paid.** Adding a new frozen generation under
`eval/cells/<name>/` — with its reference latents, its calibration and its roofline — is
cartography, and it is scored in its own right. It is allowed where editing an existing
generation is not, for a specific reason: an added generation cannot change what any existing
receipt meant, and an edited one silently re-scores history. `docs/CARTOGRAPHY.md` has what a
new cell has to come with.

That is not a prohibition. Improving the evaluator is a real contribution — the evaluator is where
the bugs are, and a broken one prints a confident number. Send it as its own PR, scored as a change
to what is measured.

**Never remove or relax a guard without knowing which incident it encodes.** They are all named in
comments where they live. If one is in your way, that comment is the first thing to read.

Anything under `eval/cells/` is a **frozen generation** and cannot be edited at all. If the meaning
of the evaluation should change, the answer is a new generation — receipts stay attached to the one
that produced them.
