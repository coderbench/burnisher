<!--
Fill in each section. Text inside these comment markers stays hidden on the pull request.
Full guide: CONTRIBUTING.md. How the number is computed: docs/SCORING.md.
-->

## What this changes

<!-- One or two sentences: what you changed, and why it should be faster. Commit messages are one
line, so the explanation goes here. -->

## Kind of change

<!-- Tick exactly one. A kernel or cell pull request that also touches an instrument path is
labelled `burnish:skipped-instrument` and is not scored, so send each kind separately. -->

- [ ] **Kernel:** a faster kernel, including a faster version of an existing one, registered under a
      new name
- [ ] **New cell:** a new resolution, dtype or model, added as a new generation
      (`docs/CARTOGRAPHY.md`)
- [ ] **Instrument:** touches `eval/`, `configs/`, `schemas/`, `tools/burnish`, `scripts/` (except
      `scripts/build*`), `.github/` or `.gittensor/`. Reviewed by a maintainer, not scored.
- [ ] **Docs, build or other:** not scored

## For a kernel

<!--
How it is measured: the validator builds your branch and runs `cuda`, the current kernels, against
the kernel name your pull request newly registers, in the same binary. You are paid for the
difference.

Making an existing kernel faster? Do not edit it in place.
  1. Copy it (for example `attention_cuda` in `src/cuda/ops_cuda.cu`) and change the copy.
  2. Register the copy beside the original, under a new name:
       register_impl<AttentionArgs>("attention", "flash-sm120", my_attention, "what changed");
  3. Write that name on the Implementation name line below.

These are not evaluated and not paid:
  - an existing kernel edited in place, or no new name   -> burnish:no-candidate
  - an existing kernel registered again under a new
    name, or a copy that is 95% or more the same code    -> burnish:reregistered
  - code copied from another author's open pull request  -> burnish:copycat
                                                            (closed, account blocked)
-->

**Implementation name:** `<name>`

<!-- The new name you registered. Required if you register more than one new name. -->

- [ ] The new name shows in `burnisher info` from the CUDA build (`scripts/build_cuda.sh`; needs the
      CUDA toolkit and cuDNN 9, not a GPU)
- [ ] It touches no instrument path (see *Kind of change*)
- [ ] `scripts/check.sh` passes (no GPU)
- [ ] It matches the reference implementation:
      `scripts/differential_test.py --weights DIR --stage <stage> --impl <name> --device cuda`
- [ ] Any new source file is added to `CMakeLists.txt`

### Measured yourself (optional)

<!-- The validator measures every submission, so this is optional. If you ran
`eval/score_submission.sh` on an RTX 5090, paste the output of
`tools/burnish receipt show <receipt>` below, in a code block. Never retype a number. -->

## For a new cell

<!-- What a cell must come with, and how it is evaluated: docs/CARTOGRAPHY.md. -->

- [ ] A new generation under `eval/cells/<name>/`; no existing generation is edited
- [ ] Its oracle is complete: `prompts.json`, `token-ids.json`, a `token-ids-<id>.txt` and a
      reference latent for every prompt, and `eval/cells/<name>/reference-latents/manifest.json`
- [ ] `tools/burnish cartography check --generation <name> --base origin/main` passes
- [ ] A new resolution has its own entry in `configs/tolerance.json`

## For an instrument change

- [ ] No guard is removed or relaxed, or the description names the incident its comment encodes and
      why it no longer applies
- [ ] No existing generation under `eval/cells/` is edited
- [ ] It is not mixed with a kernel or a new cell

## Before you open it

- [ ] Every commit message is one line of 72 characters or fewer: `<type>: <description>`, where
      `type` is `feat`, `fix`, `perf`, `docs`, `test`, `build`, `refactor` or `chore`

<!--
What comes back: rounds run every two hours, oldest first.
  - burnish:gap+N.NNNN     paid: the share of the remaining gap you closed
  - burnish:unresolved     the effect was inside the noise; it says nothing about the idea
  - burnish:needs-rebase   a bigger gain in the same round was merged; rebase and push
Every new commit is measured again, whatever its label. All labels: docs/EVAL.md.
-->
