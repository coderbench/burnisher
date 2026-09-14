# Contributing

The deal: **land a kernel, get a number with an interval on it.** No letter grades, no arguing
about which bucket a result fell into, and no maintainer discretion over the score.

This page is everything you need to make and submit a change. Each section links to the document
that explains the *why*, for when you want it.

## What you need

- **To build and check:** no GPU. The CPU build (`scripts/build.sh`) is the correctness oracle and
  runs the whole harness.
- **To submit:** nothing more. The validator builds, gates and measures every submission on its
  own RTX 5090.
- **To know whether your kernel is faster before a round tells you:** an RTX 5090 and the
  checkpoint. Nothing that produces a measurement runs without a device.

## 1. Pick something

1. Run `burnish roofline`. It prints every cell's arithmetic ceiling and how full it is. Cells with
   `achieved: --` have not been measured — today the fp8 and NVFP4 cells, which have no
   implementation yet.
2. Read `issues/README.md`. Every item carries its own arithmetic, computed from `configs/`.
3. Check the `res` column. A cell whose remaining room does not clear 20× its own noise floor
   **cannot be shown to have improved**, however good your kernel is. Do not start there.

`docs/STATUS.md` says what has been measured and what has not.

## 2. Write it beside the old one

**Register a new name. Do not replace a file.**

```cpp
register_impl<AttentionArgs>("attention", "flash-sm120", my_kernel,
                             "tiled online-softmax for sm_120, 128x64 blocks");
```

That is what lets base and candidate run in one process, one model load and one thermal state,
keeps the old implementation runnable forever, and makes `--impl <name>` fail loudly if the name
is not registered. `burnisher info` lists everything a build contains.

**One name, fifteen ops.** Your name applies to whichever ops register it; every other op falls
back to the baseline for the device the run is on. Registering one kernel is the normal case. The
harness refuses a run whose reported implementations disagree with what was asked for, so an
unintended fallback is a rejected run rather than a wrong number.

Why it is built this way: `docs/ARCHITECTURE.md`.

## 3. Check it without a GPU

```bash
scripts/check.sh
```

C++ tests, harness tests, the whole graph on synthetic weights, generated-document staleness, and
the manifest. If you touched anything that computes — a kernel, a graph, a layout — also compare
against the reference implementation. It needs a checkpoint but no GPU:

```bash
scripts/differential_test.py --weights DIR --stage vae-decode
scripts/differential_test.py --weights DIR --stage dit-step --resolution 64
```

A disagreement far above tolerance with an **identical mean and standard deviation** is a
permutation, not an arithmetic error. The gate and its tolerance: `docs/CORRECTNESS.md`.

## 4. Measure it yourself, if you have a 5090

The same command the validator runs:

```bash
eval/score_submission.sh --base <the commit you branched from> --worktree . \
    --impl-base cuda --impl-candidate <your-impl> \
    --pr <n> --ledger <a directory OUTSIDE this worktree> \
    --weights <checkpoint dir> --noise <the pinned noise .npy>
```

It gates the base arm, gates yours, benches both paired and interleaved, and scores into the
ledger. Correctness is gated before anything is timed, and the base arm is gated too. Running the
box: `docs/HARDWARE.md`.

## 5. How it is scored

- You are paid the **fraction of the remaining gap to the arithmetic ceiling** that you close. The
  same speedup pays more on a cell that is already close to its ceiling.
- A gain counts only if it clears **that cell's own measured noise floor**.
- **Faster but hungrier or less faithful pays nothing**: latency, VRAM and fidelity form a frontier.
- Your gain must survive a **held-out shape** drawn after your code is frozen, on **every** cell.
- A regression credits zero, not negative.

The reasoning behind each rule: `docs/SCORING.md`.

## 6. What comes back

Submissions are evaluated **in rounds**: every two hours, three at a time, oldest first. At most one
is merged per round — the largest credited gain — because gains measured against the same `main`
do not compose. Every other scored submission is asked to rebase and is re-measured. That is not a
rejection.

You get one label and a comment naming the commit that was measured. A paying outcome carries the
number itself:

```
burnish:gap+0.0342
```

Every other label carries a reason instead. Not being paid is not the same as being rejected —
`burnish:unresolved` means the effect was inside the noise, not that the idea was wrong. The full
list, and what each colour means: `docs/EVAL.md`.

**You can check the verdict yourself with no GPU, in about two seconds:**

```bash
burnish audit pr-000042-raw.json pr-000042.json
```

If you think the measurement itself is wrong, re-run it on your own 5090 and file a
counter-receipt with `burnish challenge`. Both are in `docs/EVAL.md`.

## Other contributions

- **Opening a new cell is paid.** A new frozen generation under `eval/cells/<name>/`, with its
  reference latents, calibration and roofline, is cartography: `docs/CARTOGRAPHY.md`.
- **Changing the instrument is welcome, and evaluated separately.** `eval/`, `configs/`, `schemas/`
  and `tools/burnish` decide what is measured, so a submission touching them is labelled
  `burnish:skipped-instrument` and no GPU time is spent on it. Send it as its own PR. Never remove
  or relax a guard without reading the incident named in its comment. Anything under
  `eval/cells/` that already exists is frozen and cannot be edited. Details: `docs/EVAL.md`.
- **Docs, tooling and refactors** are welcome and score zero. That is not a judgement of their
  worth; the subnet pays for measured movement toward the ceiling.

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
