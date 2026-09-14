# How a submission is evaluated

## What happens to a pull request

1. **Copies.** If its author is blocked, or its new code copies another open pull request, it is
   closed and no GPU time is spent (see *Copies* below).
2. **Guard.** If it changes the measuring instrument it is labelled `burnish:skipped-instrument`,
   and no GPU time is spent.
3. **Re-registration.** If it registers a kernel already on main under a new name it is labelled
   `burnish:reregistered`, and no GPU time is spent (see *Re-registered kernels* below).
4. **Build** from source on the evaluation box.
5. **Gate** correctness (in fp32) and determinism for both base and candidate
   (`docs/CORRECTNESS.md`).
6. **Bench.** Base and candidate runs alternate, plus a held-out shape drawn after the code is
   frozen.
7. **Score** into an append-only ledger outside the repository.
8. **Publish** the raw measurements and the receipt, and **label** the pull request from the
   receipt.

Steps 4–7 are `eval/score_submission.sh`, the same command anyone can run.

## Rounds

- **Every two hours, three submissions, oldest first** (`eval/run_round_cron.sh`). One submission
  costs about 24 GPU-minutes, and two benchmarks cannot share a GPU.
- **A round scores one generation:** BG-1, unless `BURNISH_GENERATION` names another. Its
  `BURNISH_NOISE` must be that generation's pinned noise.
- **The order never depends on the submission's content**, so it can't be gamed.
- **The head commit is frozen when the round starts.** The comment names the commit it measured.
- **At most one merge per round:** the largest credited gain. Gains measured against the same
  `main` overlap, so the others get `burnish:needs-rebase` and are measured again after rebasing.
- **Nothing resolved means nothing merges.** Merging must be switched on (`--merge`); otherwise the
  winner is labelled `burnish:merge-first` for a person to merge.

## Labels

| label | meaning |
|:--|:--|
| `burnish:gap+N.NNNN` | **Paid.** The fraction of the remaining gap closed. |
| `burnish:cell-opened` | **Paid.** A new cell (`docs/CARTOGRAPHY.md`). |
| `burnish:merge-first` | The best verified gain of its round. |
| `burnish:needs-rebase` | Measured correctly, but another gain merged first. Rebase. |
| `burnish:unresolved` | Inside the cell's noise. Not a judgement of the idea. |
| `burnish:no-gain` | Measured, and not an improvement. |
| `burnish:moved-along-frontier` | Faster, but paid for in memory or fidelity. |
| `burnish:expanded-off-latency` | Less memory or closer to the reference, no faster. Not paid. |
| `burnish:shape-overfit` | The gain vanished at the held-out shape. |
| `burnish:partial` | Not every cell was run. |
| `burnish:correctness-fail` | Failed the gate. Rejected. |
| `burnish:determinism-fail` | Did not reproduce itself. Rejected. |
| `burnish:held` | An independent re-measurement disagrees. Waiting for a third. |
| `burnish:skipped-instrument` | Changes the instrument. Not evaluated. |
| `burnish:copycat` | Most of its new code copies another open pull request. Closed, author blocked. |
| `burnish:blocked` | The author was blocked for an earlier copy. Closed, not evaluated. |
| `burnish:copycat-review` | Part of it matches another pull request. Measured, not paid until cleared. |
| `burnish:reregistered` | Registers a kernel already on main under a new name. Not evaluated. |
| `burnish:build-fail` | Did not build. Not evaluated. |
| `burnish:eval-error` | The evaluator's fault, not yours. Re-run. |

Colours: green = paid or merge-first (darker is bigger), blue = measured but not paid, pale blue =
could not tell, red = correctness or determinism failure, amber = shape overfit, needs rebase,
copycat, blocked or re-registered, purple = disputed or copycat review, orange = build or evaluator failure, grey = not
evaluated. They are defined in `eval/burnscore/verdict.py`.

## Copies

Before anything is built, the new code is compared with the pull requests **open now** by other
authors.

- **Compared as structure, not text.** Identifiers, numbers, strings and comments are normalized
  away, so renaming and reformatting do not hide a copy.
- **Only new code counts.** Code already in the change's context, boilerplate that several
  submissions share, and the baseline on main are never evidence. Copying the old kernel and
  registering the new one beside it is the intended workflow.
- **The original is whoever the evaluator observed first,** from an append-only record under the
  ledger. A pull request force-pushed with copied code gets the time its copied head was first seen.
- **Iterating on your own pull request is never flagged.** Maintainers named in
  `.github/CODEOWNERS` are exempt.

| verdict | when | result |
|:--|:--|:--|
| `burnish:copycat` | 70% or more of a change's new code is in one open pull request | closed, not paid, author blocked |
| `burnish:blocked` | the author was blocked before | closed, not evaluated |
| `burnish:copycat-review` | part of it matches, it contains most of another pull request, or it is a small identical change | measured, not paid until cleared |

**A copy blocks the account automatically.** The block is appended to `<ledger>/copycat/blocked.jsonl`,
and every later pull request from that account is closed unevaluated. A branch stacked on somebody
else's unmerged work also contains most of it, so that is review, not a block. The comment quotes the
matching lines and names the original.

A maintainer who finds a block wrong lifts it with a recorded reason, then reopens the pull request
and adds `copycat-cleared`:

```bash
scripts/copycat_guard.py --corpus <ledger>/copycat --unblock <login> --reason "independent work"
```

A review is cleared by adding `copycat-cleared` and removing the label.

## Re-registered kernels

Submissions are measured against `cuda`, which never runs any other registered kernel. So a kernel
already on main -- a merged contributor's or a baseline one -- registered again under a new name
would be measured as a gain that already landed. Before anything is built,
`scripts/reregistration_guard.py` reads the registry on main and in the submission:

- **The same callable under a new name** (`attention_cuda_tiled<256>` again as `fast`) is
  re-registered.
- **A renamed, reformatted copy** of a registered kernel -- its wrapper and the device kernels it
  launches -- is re-registered at 95% or more token similarity. Identifiers and comments are
  normalized; helpers most kernels call are left out.
- **Different constants are a variant, not a copy.** Numbers are kept, so a new tile width like
  `cuda-tile64` is clear.
- **A changed copy is clear.** Copying a kernel, changing it and registering it beside the old one
  is the intended workflow.

It is labelled `burnish:reregistered`, not evaluated and not paid. The account is not blocked: the
kernel on main is public. A maintainer who finds it wrong adds `reregistration-cleared` and removes
the label.

## Checking a verdict

The score is pure arithmetic over recorded measurements, so anyone can re-derive it.

| | audit | challenge |
|:--|:--|:--|
| asks | does the verdict follow from the published measurements? | did the measurements really happen? |
| needs | nothing, about two seconds | an RTX 5090, about 25 minutes |
| catches | scoring bugs, edited receipts | measurements that never happened |

```bash
burnish audit pr-000042-raw.json pr-000042.json
burnish challenge <your receipt for the same submission> --ledger <the public ledger>
```

- The raw file carries the anchor it was scored against, so an audit works on any machine.
- A challenge must measure **the same commit, implementations and generation**, on a **different
  physical card** (by GPU UUID).
- If two receipts disagree by more than the cell's noise floor, the credit is **held**: neither paid
  nor rejected, until a third measurement settles it.

## The instrument guard

`eval/`, `configs/`, `schemas/` and `tools/burnish` decide what is measured.

- **`eval/run_from_base.sh` takes them from the base commit** before scoring, so an edit can't help
  its author.
- **A required CI check blocks merging such edits**, and `.github/CODEOWNERS` also covers
  `.github/`, `.gittensor/` and `scripts/`.
- **One exception:** adding a new generation under `eval/cells/<new>/` is allowed (cartography).
  Editing an existing one is not.
- **Limit:** the box builds and runs submitted code. Isolate it. The guard is not a sandbox.

## Running a validator

**No box needs a calibration of its own.** Each generation is anchored once, on any card of the
pinned class. `reference.json` holds each cell's achieved fraction, the base time behind it, and
the worst noise floor measured in any session. A run contributes only its paired base/candidate
ratio: the ceiling in that run's own seconds is `achieved × base time`. So a card that is uniformly
slower, or slower at a resource the code is not limited by, gives the same score
(`eval/tests/test_portable_scoring.py`).

Setting up a new box:

```bash
scripts/build_cuda.sh             # CMAKE_CUDA_ARCHITECTURES=121 for DGX Spark
build-cuda/burnisher check-weights --weights <checkpoint>
```

Two guards on every run replace per-box calibration:
- **The base arm must be within 25% of the anchor's time.** On a second card, BG-1's base times
  differed from the anchor's by up to 9.4% (`eval/cells/BG-1/second-card-check.json`). Further
  than 25% means the base code changed, or this is not the pinned hardware.
- **The base arm's repeats must spread less than 3× the floor.** Otherwise the box was too noisy,
  and the run is refused as the box's fault, not the submission's.

## Anchoring a generation

Once per generation, and again only when the base code changes:

```bash
burnish calibrate --generation BG-N --impl cuda --repeats 9 --weights <checkpoint> --write
# a second session, on any card, keeping the worst floor per cell
burnish calibrate --generation BG-N --impl cuda --repeats 9 --weights <checkpoint> \
    --merge eval/cells/BG-N/reference.json --write
```

**Anchor with two sessions, because floors move.** Two calibrations of the same RTX 5090, hours
apart:

| cell | session A | session B | ratio |
|:--|--:|--:|--:|
| `dit-step/1024/bf16` | 0.578% | 0.205% | 2.8× |
| `t5-encode/1024/bf16` | 3.753% | 0.155% | 24.2× |
| `vae-decode/1024/bf16` | 0.259% | 0.845% | 3.3× |

`--merge` keeps the worst floor per cell. A floor too tight would pay for noise permanently; one
too loose only refuses a gain too small to see.

**On the box:**
- **Never run two benchmarks at once.** They race for VRAM and produce plausible wrong numbers.
  `burnish` refuses a busy device and takes a lock.
- **Base and candidate always alternate.** The box drifts within a single run by more than some
  cells' floors, and `eval/tests/test_scoring.py` pins that. A base timing is never cached.
- **Kill by PID** from `nvidia-smi`. `pkill -f` over ssh kills your own session.

## What this does not claim

- **It is not a proof.** It is agreement between independent measurers.
- **An audit checks arithmetic, not whether a measurement happened.**
- **Many simultaneous submissions are unmeasured.**
