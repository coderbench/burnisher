# Contributing

**Land a faster kernel, get a number with an interval on it.** No grades and no maintainer
discretion over the score.

## What you need

- **To build, test and submit:** no GPU. The validator builds, gates and measures every submission.
- **To know if your kernel is faster before a round tells you:** an RTX 5090 and the checkpoint.

## 1. Pick something

- `burnish roofline` shows every cell's ceiling and how full it is. `--` means not measured yet.
- `issues/README.md` lists the open work.
- Skip cells whose `res` column says `no`: their noise is too large for any gain to show.

## 2. Add your kernel beside the old one

**Register a new name. Never replace a file.**

```cpp
register_impl<AttentionArgs>("attention", "flash-sm120", my_kernel, "what it does");
```

- Base and candidate then run in one process, so the comparison is fair.
- Every op that doesn't register your name falls back to the baseline for that device. Registering
  one kernel is normal.
- `burnisher info` lists what a build contains. A name that isn't registered fails loudly.

## 3. Check it, no GPU needed

```bash
scripts/check.sh
```

If you changed anything that computes, also compare against the reference implementation
(needs the checkpoint):

```bash
scripts/differential_test.py --weights DIR --stage vae-decode
scripts/differential_test.py --weights DIR --stage dit-step --resolution 64
```

A big difference with **identical mean and standard deviation** means values are in the wrong
order, not wrong. The gate itself: `docs/CORRECTNESS.md`.

## 4. Measure it yourself (optional, needs a 5090)

This is the same command the validator runs:

```bash
eval/score_submission.sh --base <commit you branched from> --worktree . \
    --impl-base cuda --impl-candidate <your-impl> \
    --pr <n> --ledger <directory outside this repo> \
    --weights <checkpoint dir> --noise <pinned noise .npy>
```

It scores BG-1 by default. To score against BG-2, add `--generation BG-2` and use BG-2's noise.

## 5. How you are scored

- You are paid the **fraction of the remaining gap** to the ceiling that you close.
- It must clear **the cell's own noise floor**.
- Faster but using more memory, or less faithful to the reference: **nothing**.
- It must hold at a **held-out shape** and on **every** cell.
- A regression credits zero, not negative.

Details: `docs/SCORING.md`.

## 6. What comes back

- Evaluation runs **every two hours, three submissions at a time, oldest first**.
- At most **one** submission per round is picked to merge: the biggest verified gain. The others
  get `burnish:needs-rebase`. That is not a rejection: rebase and it is measured again.
- A paid result is labelled with its number, e.g. `burnish:gap+0.0342`.
- `burnish:unresolved` means the effect was inside the noise. It says nothing about your idea.

Check any verdict yourself, no GPU:

```bash
burnish audit pr-000042-raw.json pr-000042.json
```

All labels, disputes and the evaluation loop: `docs/EVAL.md`.

## Other contributions

- **New cell (cartography): paid.** See `docs/CARTOGRAPHY.md`.
- **Changes to `eval/`, `configs/`, `schemas/` or `tools/burnish`:** these are the measuring
  instrument. They are not scored and get `burnish:skipped-instrument`. Send them as their own PR,
  and never remove a guard without reading the incident named in its comment.
- **Anything under `eval/cells/` that already exists is frozen.** Change means a new generation.
- **Docs, tooling, refactors:** welcome, score zero.
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
