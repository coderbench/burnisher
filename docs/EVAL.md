# How a submission is evaluated

## What happens to a pull request

1. **Copies.** If its author is blocked, or its new code copies another open pull request, it is
   labelled and closed, and no GPU time is spent (see *Copies*).
2. **Guard.** If it changes the measuring instrument it is labelled `burnish:skipped-instrument`,
   and no GPU time is spent (see *The instrument guard*).
3. **Kernel.** If it registers a kernel already on main under a new name it is labelled
   `burnish:reregistered` (see *Re-registered kernels*). Otherwise the new kernel name it registers
   is the candidate arm. With no new name, or several and none named in the description's
   `Implementation name` field, it is labelled `burnish:no-candidate`. No GPU time is spent on
   either.
4. **Build** from source on the evaluation box (`scripts/build_cuda.sh`).
5. **Gate** correctness (in fp32) and determinism for the base arm (`cuda`) and the candidate
   (`docs/CORRECTNESS.md`).
6. **Bench.** Base and candidate runs alternate, plus a held-out shape drawn after the code is
   frozen.
7. **Score** into an append-only ledger outside the repository.
8. **Publish** the raw measurements and the receipt, and **label** the pull request from the
   receipt.

Steps 5–7 are `eval/score_submission.sh`, the same command anyone can run.

## Rounds

- **Every two hours, oldest first** (`eval/run_round_cron.sh`). A round measures up to three
  submissions. One stopped at steps 1–3 costs no GPU time and does not use a slot. A measured
  submission costs about 24 GPU-minutes, and two benchmarks cannot share a GPU.
- **A round scores one generation:** BG-1, unless `BURNISH_GENERATION` names another. Give it that
  generation's pinned noise in `BURNISH_NOISE`. The gate records the noise file's hash; nothing
  checks it against the generation.
- **The order never depends on the submission's content**, so it can't be gamed.
- **The head commit is frozen when the round starts.** The comment names the commit it measured.
- **A pull request is evaluated again** when a new commit is pushed, whatever its label. Also:
  after `no-candidate` when its description changes, after `eval-error` up to three times for the
  same commit, and when a maintainer clears a copy, a copycat review or a re-registration. The bot
  records which commit each outcome was for under `<ledger>/evaluations/`.
- **At most one merge per round:** the largest credited gain. The other gains in the round were
  measured against the same `main`, so they get `burnish:needs-rebase` and are measured again after
  a pushed rebase. Results that did not pay keep their own label.
- **Nothing resolved means nothing merges.** Merging must be switched on (`--merge`). Otherwise
  `burnish:merge-first` is added beside the winner's paid label, for a person to merge.
- **Holds reach the label.** Each round first puts `burnish:held` in place of the label of any pull
  request whose latest receipt the ledger holds, and gives the receipt's label back when the hold
  is released.

## Labels

| label | meaning |
|:--|:--|
| `burnish:gap+N.NNNN` | **Paid.** The fraction of the remaining gap closed. |
| `burnish:cell-opened` | **Paid.** A new cell (`docs/CARTOGRAPHY.md`). |
| `burnish:merge-first` | Beside the paid label: the best verified gain of its round. |
| `burnish:needs-rebase` | A resolved gain, but a larger one in its round was picked to merge. Push a rebase. |
| `burnish:unresolved` | Inside the cells' noise. Not a judgement of the idea. |
| `burnish:no-gain` | Resolved, and not an improvement. |
| `burnish:moved-along-frontier` | Faster, but the frontier shrank: paid for in memory or fidelity. |
| `burnish:expanded-off-latency` | Resolved without a speedup, and the frontier still grew. Not paid. |
| `burnish:shape-overfit` | A resolved gain did not stay above its floor at the held-out shape. |
| `burnish:partial` | Not every cell was run. |
| `burnish:correctness-fail` | Failed the gate. Rejected. |
| `burnish:determinism-fail` | Did not reproduce itself. Rejected. |
| `burnish:held` | Replaces the label while an independent re-measurement disagrees. |
| `burnish:skipped-instrument` | Changes the instrument. Not evaluated. |
| `burnish:copycat` | Most of its new code copies another open pull request. Closed, author blocked. |
| `burnish:blocked` | The author was blocked for an earlier copy. Closed, not evaluated. |
| `burnish:copycat-review` | A paying result that matches another open pull request. Not paid until cleared. |
| `burnish:reregistered` | Registers a kernel already on main under a new name. Not evaluated. |
| `burnish:no-candidate` | No single new kernel name to measure. Not evaluated. |
| `burnish:build-fail` | Did not build. Not evaluated. Push a fix. |
| `burnish:eval-error` | The evaluator's fault, not yours. Retried automatically. |

Colours: green = paid or merge-first (darker is bigger), blue = measured but not paid, pale blue =
could not tell, red = correctness or determinism failure, amber = shape overfit, needs rebase,
copycat, blocked or re-registered, purple = held or copycat review, orange-red = build or evaluator
failure, grey = not evaluated (skipped instrument, no candidate). They are defined in
`eval/burnscore/verdict.py`.

## Copies

Before anything is built, the new code is compared with the pull requests **open now** by other
authors that the evaluator observed earlier.

- **Compared as structure, not text.** Identifiers, numbers, strings and comments are normalized
  away, so renaming and reformatting do not hide a copy. Code is fingerprinted as hashed windows of
  16 tokens.
- **Only new code counts.** Code already in the change's context, fingerprints that more than three
  open pull requests share, and code already on main are never evidence. Copying the old kernel and
  registering the new one beside it is the intended workflow.
- **The original is whoever the evaluator observed first,** from an append-only record under the
  ledger. A pull request force-pushed with copied code gets the time its copied head was first seen.
- **Iterating on your own pull request is never flagged.** Maintainers named in
  `.github/CODEOWNERS` (or `BURNISH_MAINTAINERS`) are exempt.
- **Building on another open pull request is not copying.** If its head commit is in your branch's
  history, a match with it is a review, never a copy.

| verdict | when | result |
|:--|:--|:--|
| `burnish:copycat` | 20 or more new-code fingerprints, and 70% or more of them are in one open pull request | closed, not paid, author blocked |
| `burnish:blocked` | the author was blocked before | closed, not evaluated |
| `burnish:copycat-review` | it holds 70% of another pull request's new code (60 or more shared fingerprints); or 40–69% of its 20 or more fingerprints match one; or a change of 2–19 fingerprints is identical to one; or it is stacked on one and matches it | measured; a paying result is not paid until cleared |

Anything below those lines is clear. The comment quotes the matching lines and names the original.

**A copy blocks the account automatically.** The block is appended to
`<ledger>/copycat/blocked.jsonl`, and every later pull request from that account is closed
unevaluated.

**Clearing:**
- **A review:** a maintainer adds `copycat-cleared`. The pull request is measured again in a later
  round and paid like any other, and the label covers its later commits too.
- **A wrong copy verdict:** a maintainer reopens the pull request and adds `copycat-cleared`, which
  gets that pull request evaluated. The account's other pull requests stay blocked until the block
  is lifted, with a recorded reason:

```bash
scripts/copycat_guard.py --corpus <ledger>/copycat --unblock <login> --reason "independent work"
```

## Re-registered kernels

Submissions are measured against `cuda`, which never runs any other registered kernel. So a kernel
already on main -- a merged contributor's or a baseline one -- registered again under a new name
would be measured as a gain that already landed. Before anything is built,
`scripts/reregistration_guard.py` reads the registry on main and in the submission:

- **The same callable under a new name** (`attention_cuda_tiled<256>` again as `fast`) is
  re-registered.
- **A renamed, reformatted copy** of a registered kernel -- its wrapper and the device kernels it
  launches -- is re-registered at 95% or more token similarity. Identifiers and comments are
  normalized and numbers are kept; helpers that three or more registered kernels call are left out.
- **A new template argument to a registered function is a variant.** `attention_cuda_tiled<512>`
  beside `cuda-tile64` and `cuda-tile1024` is clear. A copy registered with template arguments no
  existing registration uses is not compared.
- **A changed copy is clear** once the change takes it below 95%. Changing one constant in a copied
  kernel is not enough.

It is labelled `burnish:reregistered`, not evaluated and not paid. The account is not blocked: the
kernel on main is public. A maintainer who finds it wrong adds `reregistration-cleared`, and the
pull request is evaluated again.

## Checking a verdict

The score is pure arithmetic over recorded measurements, so anyone can re-derive it.

| | audit | challenge |
|:--|:--|:--|
| asks | does the verdict follow from the published measurements? | did the measurements really happen? |
| needs | nothing, about two seconds | an RTX 5090, about 30 minutes plus a build |
| catches | scoring bugs, edited receipts | measurements that never happened |

```bash
tools/burnish audit pr-000042-raw.json pr-000042.json
tools/burnish challenge <your receipt for the same submission> --ledger <the public ledger>
```

- The raw file carries the anchor it was scored against, so an audit works on any machine.
- A challenge must measure **the same commit, implementations and generation**, on a **different
  physical card** (by GPU UUID).
- If a re-measurement disagrees by more than the cell's noise floor, the credit is **held**: not
  paid while the disagreeing measurements are at least as many as those that agree with the
  receipt. A further measurement on another card that agrees releases it. Rounds carry the hold to
  the pull request's label.

## The instrument guard

`eval/`, `configs/`, `schemas/` and `tools/burnish` decide what is measured. `.github/`,
`.gittensor/` and `scripts/` (except `scripts/build*`) govern how. A change to any of them is
labelled `burnish:skipped-instrument`.

- **`eval/run_from_base.sh` takes the instrument from the base commit** before scoring, so an edit
  can't help its author.
- **A required CI check blocks merging such edits**, and `.github/CODEOWNERS` requires review for
  them too.
- **Exceptions:** adding a new generation under `eval/cells/<new>/` (cartography), with its own new
  entry in `configs/tolerance.json`. Editing an existing generation or entry is not allowed.
- **The guard is not a sandbox.** Submitted code runs as its own account (*Isolating submitted
  code* below).

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
eval/setup_sandbox.sh             # as root: the account submissions run as
eval/pr_bot.py --repo <owner/name> --check-box
```

Two guards on every run replace per-box calibration:
- **The base arm must be within 25% of the anchor's time.** On a second card, BG-1's base times
  differed from the anchor's by up to 9.4% (`eval/cells/BG-1/second-card-check.json`). Further
  than 25% means the base code changed, or this is not the pinned hardware.
- **The base arm's repeats must spread less than 3× the floor.** Otherwise the box was too noisy,
  and the run is refused as the box's fault, not the submission's.

## Isolating submitted code

The evaluator runs as root, holds the GitHub token and writes the ledger. Submitted code -- the
build, its tests and every launch of the runtime -- runs as a separate account named by
`BURNISH_SANDBOX_USER` (`eval/sandbox.py`).

- **Its environment is an allowlist:** CUDA, locale and the runtime's own variables. The token
  never reaches it.
- **It builds its own copy of the head commit**, in its own home. The evaluator never runs git in a
  tree the submission could write.
- **Nothing it starts outlives the step.** Every process running as the account is killed when a
  step ends.
- **Checked every round, as the account.** Nothing is evaluated if the account can read
  `.env.eval`, `gh`'s config or ssh keys; can write the checkout, the ledger, the copycat record,
  the gate cache, the weights or the noise; cannot read the weights or see the GPU; if a git remote
  URL carries credentials; or if a lock is in a directory anyone can write.
- **`--no-sandbox` runs everything as the evaluator.** Only on a machine with nothing to protect.

Limits:
- **The network is not cut.** Rented boxes are containers and cannot nest one. The account can
  read nothing secret, which the check verifies, so there is nothing to send.
- **Merged code is trusted.** Each round rebuilds `main` as the evaluator. With
  `BURNISH_AUTOMERGE` on, merged means scored, not reviewed.

## Anchoring a generation

Once per generation, and again only when the base code changes:

```bash
tools/burnish calibrate --generation BG-N --impl cuda --repeats 9 --weights <checkpoint> --write
# a second session, on any card, keeping the worst floor per cell
tools/burnish calibrate --generation BG-N --impl cuda --repeats 9 --weights <checkpoint> \
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
  `tools/burnish` refuses a busy device and takes a lock.
- **Base and candidate always alternate.** The box drifts within a single run by more than some
  cells' floors, and `eval/tests/test_scoring.py` pins that. A base timing is never cached.
- **Kill by PID** from `nvidia-smi`. `pkill -f` over ssh kills your own session.

## What this does not claim

- **It is not a proof.** It is agreement between independent measurers.
- **An audit checks arithmetic, not whether a measurement happened.**
- **Many simultaneous submissions are unmeasured.**
