# Contributing

Land a faster kernel and get a measured number for it. No grades and no maintainer discretion over
the score.

## What you need

- **To write, check and submit a kernel:** no GPU. `scripts/check.sh` builds the CPU runtime only.
  Compiling a CUDA kernel needs the CUDA toolkit and cuDNN 9 (`libcudnn9-dev-cuda-<major>` from
  NVIDIA's apt repository).
- **To run it, or measure it before a round does:** an RTX 5090, the checkpoint and the generation's
  pinned noise file.

## 1. Pick something

- `tools/burnish roofline`: every cell's ceiling, how full it is, its noise floor and its time against
  PyTorch. A cell whose `res` column says `no` needs a large gain to show.
- `issues/README.md`: the open work.

## 2. Register your kernel under a new name

```cpp
register_impl<AttentionArgs>("attention", "flash-sm120", my_kernel, "what it does");
```

- **Never change an existing kernel in place.** The validator measures the new name you register
  against `cuda`, in the same binary. A kernel changed in place is measured against itself, so it is
  not evaluated (`burnish:no-candidate`).
- **To make an existing kernel faster,** copy it, change the copy and register the copy under a new
  name. A copy that is 95% or more the same code is `burnish:reregistered`.
- Ops that don't register your name run `cuda`. Registering one kernel is normal.
- Register one new name. If you register several, put the one to measure in the pull request's
  `Implementation name` field.
- `burnisher info` lists what a build contains.

## 3. Check it

```bash
scripts/check.sh
scripts/differential_test.py --weights DIR --stage vae-decode --impl <your-impl> --device cuda
```

The second compares one stage with the reference implementation; it needs the checkpoint, and
`--device cuda` needs a GPU. A large difference with identical mean and standard deviation means
values are in the wrong order. The gate: `docs/CORRECTNESS.md`.

## 4. Measure it (optional)

```bash
scripts/build_cuda.sh
eval/score_submission.sh --base <commit you branched from> --worktree . \
    --impl-base cuda --impl-candidate <your-impl> --pr <n> \
    --ledger <directory outside this repo> --weights <checkpoint dir> --noise <pinned noise .npy>
```

This is the command the validator runs. It scores BG-1; add `--generation BG-2`, with BG-2's noise,
for the 512px generation.

## 5. What comes back

- Rounds run **every two hours, oldest first**, up to twelve submissions each.
- **Paid:** `burnish:gap+N.NNNN`, the share of the remaining gap you closed (`docs/SCORING.md`).
- **One merge per round:** the biggest verified gain. Other gains get `burnish:needs-rebase`; push a
  rebase and it is measured again.
- `burnish:unresolved` means the effect was inside the noise, not that the idea is wrong.
- Every new commit is measured again, whatever its label.

Check any verdict yourself, no GPU: `tools/burnish audit pr-000042-raw.json pr-000042.json`. Every
label: `docs/EVAL.md`.

## Other contributions

- **A new cell is paid:** `docs/CARTOGRAPHY.md`.
- **Instrument changes** (`eval/`, `configs/`, `schemas/`, `tools/burnish`, `scripts/` except
  `scripts/build*`, `.github/`, `.gittensor/`) are not scored (`burnish:skipped-instrument`). Send
  them as their own pull request, and read the incident a guard's comment names before removing it.
- **Existing generations under `eval/cells/` are frozen.** A change means a new generation.
- **Docs and refactors** are welcome and score zero.
- **Copying is not paid.** Copying another author's open pull request gets `burnish:copycat`, the
  pull request closed and the account blocked. Registering a kernel already on main under a new name
  gets `burnish:reregistered`. Starting from a kernel on main and changing it, or iterating on your
  own pull requests, is fine.

## Commit messages

One line of 72 characters or fewer, no body, no trailers: `<type>: <short imperative description>`,
where `type` is `feat`, `fix`, `perf`, `docs`, `test`, `build`, `refactor` or `chore`.

```
perf: fuse AdaLN into the modulation kernel
```
