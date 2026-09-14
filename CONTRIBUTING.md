# Contributing

**Land a faster kernel, get a number with an interval on it.** No grades and no maintainer
discretion over the score.

## What you need

- **To build, run `scripts/check.sh` and submit:** no GPU. The validator builds, gates and measures
  every submission: it finds the kernel name your pull request newly registers and measures it
  against `cuda` (the stock CUDA kernels) in the same binary. `check.sh` builds the CPU runtime
  only, so it never compiles a CUDA kernel; compiling one yourself needs the CUDA toolkit, not a
  GPU.
- **To run your CUDA kernel, or to know if it is faster before a round tells you:** an RTX 5090,
  the checkpoint and the generation's pinned noise file. Any RTX 5090 will do.

## 1. Pick something

- `tools/burnish roofline` shows every cell's ceiling and how full it is. `--` means not measured
  yet.
- `issues/README.md` lists the open work.
- A cell whose `res` column says `no` has less than 20 times its noise floor left, so a gain there
  has to be large to show.

## 2. Add your kernel beside the old one

**Register a new name. Never replace a file.**

```cpp
register_impl<AttentionArgs>("attention", "flash-sm120", my_kernel, "what it does");
```

- Base and candidate then run in one process, so the comparison is fair.
- Every op that doesn't register your name falls back to the baseline for that device. Registering
  one kernel is normal.
- `burnisher info` lists what a build contains. A name that isn't registered fails loudly.
- **The validator measures the new name you register.** Use one name (for as many ops as you like).
  If you register several, put the one to measure in the PR description's `Implementation name`
  field. Changing an existing kernel in place would be measured against itself, so it is not
  evaluated (`burnish:no-candidate`).

## 3. Check it

```bash
scripts/check.sh          # no GPU
```

If you changed anything that computes, also compare your kernel against the reference
implementation (needs the checkpoint):

```bash
scripts/differential_test.py --weights DIR --stage vae-decode --impl <your-impl>
scripts/differential_test.py --weights DIR --stage dit-step --resolution 64 --impl <your-impl>
```

Add `--device cuda` for a CUDA kernel; that needs a GPU.

A big difference with **identical mean and standard deviation** means values are in the wrong
order, not wrong. The gate itself: `docs/CORRECTNESS.md`.

## 4. Measure it yourself (optional, needs a 5090)

Build the CUDA runtime first (`scripts/build_cuda.sh` writes `build-cuda/burnisher`). This is the
command the validator runs, which also passes the generation and a work directory:

```bash
eval/score_submission.sh --base <commit you branched from> --worktree . \
    --impl-base cuda --impl-candidate <your-impl> \
    --pr <n> --ledger <directory outside this repo> \
    --weights <checkpoint dir> --noise <pinned noise .npy>
```

It scores BG-1 by default. To score against BG-2, add `--generation BG-2` and use BG-2's noise.

## 5. How you are scored

- You are paid the **fraction of the remaining gap** to the ceiling that you close, averaged over
  the cells.
- A cell counts only if it clears **its own noise floor**. Cells you didn't touch count as zero
  and don't block you.
- A regression that clears the floor counts **against** you.
- Faster, but paid for in more memory or less fidelity than the speed is worth: **nothing**.
- Every cell you sped up must still be faster at a **held-out shape**.

Details: `docs/SCORING.md`.

## 6. What comes back

- Evaluation runs **every two hours, oldest first**, measuring up to three submissions a round.
- A paid result is labelled with its number, e.g. `burnish:gap+0.0342`.
- At most **one** submission per round is picked to merge: the biggest verified gain. Other gains
  in the round get `burnish:needs-rebase`. That is not a rejection: push a rebase and it is
  measured again.
- `burnish:unresolved` means the effect was inside the noise. It says nothing about your idea.
- Pushing a new commit gets your pull request measured again, whatever its label.

Check any verdict yourself, no GPU:

```bash
tools/burnish audit pr-000042-raw.json pr-000042.json
```

All labels, disputes and the evaluation loop: `docs/EVAL.md`.

## Other contributions

- **New cell (cartography): paid.** See `docs/CARTOGRAPHY.md`.
- **Changes to `eval/`, `configs/`, `schemas/`, `tools/burnish`, `scripts/` (except
  `scripts/build*`), `.github/` or `.gittensor/`:** these are the measuring instrument and its
  governance. They are not scored and get `burnish:skipped-instrument`. Send them as their own PR,
  and never remove a guard without reading the incident named in its comment.
- **Anything under `eval/cells/` that already exists is frozen.** Change means a new generation.
- **Docs and refactors:** welcome, score zero. Tooling under `scripts/` is instrument (above).
- **Someone else's work is not paid.** New code that copies another author's open pull request is
  labelled `burnish:copycat`, the pull request is closed and the account is blocked. Registering a
  kernel already on main under a new name is labelled `burnish:reregistered` and not evaluated.
  Starting from a kernel on main, changing it, and iterating on your own pull requests are fine.
  Details: `docs/EVAL.md`.

## Commit messages

One line. No body. No trailers.

```
<type>: <short imperative description>
```

`type` is `feat`, `fix`, `perf`, `docs`, `test`, `build`, `refactor` or `chore`. 72 characters or
fewer. Explanations go in the code, the docs or the PR description.

```
perf: fuse AdaLN into the modulation kernel
```
