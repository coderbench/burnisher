# Contributing

The deal: **land a kernel, get a number with an interval on it.** No letter grades, no arguing
about which bucket a result fell into, and no maintainer discretion over the score.

## Before you pick something

1. Read `docs/STATUS.md`. It says what has been measured (nothing, yet) and what has not.
2. Run `burnish roofline`. It prints every cell's arithmetic ceiling and how full it is. Cells
   with `achieved: --` are cells where nobody knows how much room there is.
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

Then, on the pinned hardware:

```bash
burnish gate --impl <your-impl> --repeats 10 --output gate.json   # correctness first, always
burnish bench --impl-candidate <your-impl> \
    --gate-result gate.json --gate-base-result gate-base.json --output raw.json
burnish score raw.json
```

`burnish bench` refuses to time a build that has not passed the gate. That is not a convenience
check — correctness precedes speed and is never traded against it.

## What will get your submission rejected, and why

| outcome | cause |
|:--|:--|
| `DETERMINISM_FAIL` | ten replays of your build were not byte-identical. Nothing can be attributed to a change against a baseline that does not reproduce itself. |
| `CORRECTNESS_FAIL` | the latents moved outside the stated tolerance. This is a rejection, not a trade-off, and widening the tolerance is not the fix. |
| `SHAPE_OVERFIT` | the gain vanished on a held-out shape the evaluator drew after your code was frozen. A kernel fast only on the benchmarked shape is a tuned constant. |
| `PARTIAL` | you ran some cells and not others. Dropping the cell a change hurts is the cheapest way to raise a score. |
| `UNRESOLVED` | the effect is inside the cell's own measured noise floor. You will get the number and the interval; you will not get credit. |
| `MOVED_ALONG_FRONTIER` | faster, but it cost memory or fidelity. That is a trade the runtime could already make. |

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

## Changing the instrument

`eval/`, `configs/`, `schemas/` and `tools/burnish` are the **instrument**. `eval/run_from_base.sh`
overlays them from the base commit before scoring anything, so changes you make there are reported
and then discarded for scoring purposes.

That is not a prohibition. Improving the evaluator is a real contribution — the evaluator is where
the bugs are, and a broken one prints a confident number. Send it as its own PR, scored as a change
to what is measured.

**Never remove or relax a guard without knowing which incident it encodes.** They are all named in
comments where they live. If one is in your way, that comment is the first thing to read.

Anything under `eval/cells/` is a **frozen generation** and cannot be edited at all. If the meaning
of the evaluation should change, the answer is a new generation — receipts stay attached to the one
that produced them.
