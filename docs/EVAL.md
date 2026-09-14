# How a submission is evaluated

## What happens to a pull request

1. **Copies.** A pull request from a blocked author, or whose new code copies another open pull
   request, is labelled and closed (see *Copies*).
2. **Instrument.** A change to the measuring instrument gets `burnish:skipped-instrument` (see *The
   instrument guard*).
3. **Candidate.** A kernel already on main registered under a new name gets `burnish:reregistered`
   (see *Re-registered kernels*). Otherwise the new kernel name is the candidate. No new name, or
   several with none named in the `Implementation name` field, gets `burnish:no-candidate`.
4. **Build** on the evaluation box.
5. **Gate** correctness in fp32 and determinism, for `cuda` and the candidate (`docs/CORRECTNESS.md`).
6. **Bench** base and candidate alternately, plus a held-out shape drawn after the code is frozen.
7. **Score** into an append-only ledger outside the repository.
8. **Publish** the measurements and the receipt, and **label** the pull request.

Steps 1–3 use no GPU time. Steps 5–7 are `eval/score_submission.sh`, which anyone can run.

## Rounds

- **Every two hours, oldest first,** up to twelve measured submissions (`eval/run_round_cron.sh`).
  One costs about 4.5 GPU-minutes (`eval/cells/BG-1/round-cost.json`). Two benchmarks never share a
  GPU.
- **One generation per round:** BG-1 unless `BURNISH_GENERATION` names another, with that
  generation's pinned noise in `BURNISH_NOISE`.
- **The order ignores content,** and the head commit is frozen when the round starts.
- **Measured again** after a new commit, whatever the label; after `no-candidate` when the
  description changes; after `eval-error`, up to three times per commit; and after a maintainer
  clears a copy, a copycat review or a re-registration.
- **At most one merge per round:** the largest credited gain. Other gains in the round get
  `burnish:needs-rebase`. Nothing resolved means nothing merges. Without `--merge`, the winner gets
  `burnish:merge-first` for a person to merge.
- **Holds reach the label:** a credit the ledger holds shows `burnish:held` until it is released.

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

## Copies

New code is compared with the pull requests **open now** by other authors, as structure:
identifiers, numbers, strings and comments are normalized, and code is fingerprinted in windows of
16 tokens.

- **Only new code counts.** Code already on main, already in the change's context, or shared by more
  than three open pull requests is never evidence.
- **The original is whoever the evaluator saw first,** from an append-only record under the ledger.
- **Never flagged:** iterating on your own pull request, maintainers in `.github/CODEOWNERS`, and
  building on a pull request whose head commit is in your branch's history.

| verdict | when | result |
|:--|:--|:--|
| `burnish:copycat` | 20 or more new-code fingerprints, and 70% or more of them are in one open pull request | closed, not paid, author blocked |
| `burnish:blocked` | the author was blocked before | closed, not evaluated |
| `burnish:copycat-review` | it holds 70% of another pull request's new code (60 or more shared fingerprints); or 40–69% of its 20 or more fingerprints match one; or a change of 2–19 fingerprints is identical to one; or it is stacked on one and matches it | measured; a paying result is not paid until cleared |

A copy blocks the account, and its later pull requests are closed unevaluated. To clear a review, a
maintainer adds `copycat-cleared`. To undo a wrong copy verdict, the maintainer reopens the pull
request, adds `copycat-cleared` and lifts the block:

```bash
scripts/copycat_guard.py --corpus <ledger>/copycat --unblock <login> --reason "independent work"
```

## Re-registered kernels

A kernel already on main registered under a new name would be measured as a gain that already
landed. `scripts/reregistration_guard.py` compares the registry on main with the submission's:

- **The same callable under a new name** (`attention_cuda` again as `fast`) is re-registered.
- **A renamed, reformatted copy** at 95% or more token similarity is re-registered. Identifiers and
  comments are normalized, numbers are kept, and helpers most kernels call are left out.
- **A new template argument** to a registered function is a variant, and clear.
- **A changed copy** below 95% is clear. Changing one constant is not enough.

It gets `burnish:reregistered` and is not evaluated; the account is not blocked. A maintainer who
finds it wrong adds `reregistration-cleared`.

## Checking a verdict

The score is arithmetic over recorded measurements, so anyone can re-derive it.

| | audit | challenge |
|:--|:--|:--|
| asks | does the verdict follow from the published measurements? | did the measurements really happen? |
| needs | nothing, about two seconds | an RTX 5090, about six minutes plus a build (`eval/cells/BG-1/round-cost.json`) |
| catches | scoring bugs, edited receipts | measurements that never happened |

```bash
tools/burnish audit pr-000042-raw.json pr-000042.json
tools/burnish challenge <your receipt for the same submission> --ledger <a clone of the published ledger>
```

- The raw file carries the anchor it was scored against, so an audit works on any machine.
- A challenge measures **the same commit, implementations and generation** on a **different
  physical card** (by GPU UUID).
- If re-measurements disagree by more than the cell's noise floor, the credit is **held** while the
  disagreeing measurements are at least as many as the agreeing ones.

## The instrument guard

`eval/`, `configs/`, `schemas/`, `tools/burnish`, `.github/`, `.gittensor/` and `scripts/` (except
`scripts/build*`) are the instrument. Changing any of them gets `burnish:skipped-instrument`.

- **The instrument always comes from the base commit** (`eval/run_from_base.sh`), so an edit can't
  help its author. A required CI check and `.github/CODEOWNERS` review also block merging it.
- **The one exception** is a new generation under `eval/cells/<new>/` with its own new entry in
  `configs/tolerance.json` (`docs/CARTOGRAPHY.md`).

## Running a validator

**No box needs its own calibration.** A generation is anchored once, on any card of the pinned
class, and a run contributes only its paired base/candidate ratio, so a uniformly slower card scores
the same (`eval/tests/test_portable_scoring.py`). Two checks on every run stand in for calibration:

- **The base arm must be within 25% of the anchor's time.** Otherwise the base code changed, or this
  is not the pinned hardware. Two cards of the class differed by up to 9.4% on the first kernels
  (`eval/cells/BG-1/second-card-check.json`).
- **The base arm's repeats must spread less than 3× the floor.** Otherwise the box was too noisy.

Setting up a box. The CUDA build needs the CUDA toolkit and cuDNN 9.

```bash
scripts/build_cuda.sh             # CMAKE_CUDA_ARCHITECTURES=121 for DGX Spark
build-cuda/burnisher check-weights --weights <checkpoint>
eval/setup_sandbox.sh             # as root: the account submissions run as
eval/pr_bot.py --repo <owner/name> --check-box
```

**The ledger is published after every round** (`eval/publish_ledger.py`) to `BURNISH_LEDGER_REMOTE`,
with a token in `BURNISH_LEDGER_TOKEN` that can write only that repository. Never force-push it. A
rented box is returned with its disk; the published ledger survives it.

### Isolating submitted code

Submitted code (its build, its tests and every launch of the runtime) runs as the account named by
`BURNISH_SANDBOX_USER` (`eval/sandbox.py`), never as the evaluator that holds the tokens.

- **Its environment is an allowlist.** It builds its own copy of the head commit, and every process
  it starts is killed when the step ends.
- **Checked every round.** Nothing is evaluated if the account can read `.env.eval`, `gh`'s config or
  ssh keys; can write the checkout, the ledger, the copycat record, the gate cache, the weights or the
  noise; cannot read the weights or see the GPU; can reach any service but ssh; if a git remote URL
  carries credentials; or if a lock sits in a directory anyone can write.
- **Stop the notebook server rented images start.** It runs as root, with its token on its command
  line.
- **Limits:** the network is not cut, because containers cannot nest one; and merged code is
  trusted, because each round rebuilds `main` as the evaluator. `--no-sandbox` runs everything as the
  evaluator, only on a machine with nothing to protect.

### Anchoring a generation

Once per generation, and again when the base code changes:

```bash
tools/burnish calibrate --generation BG-N --impl cuda --repeats 9 --weights <checkpoint> --write
# a second session, keeping the worst floor per cell
tools/burnish calibrate --generation BG-N --impl cuda --repeats 9 --weights <checkpoint> \
    --merge eval/cells/BG-N/reference.json --write
```

Floors move between sessions. Two back-to-back sessions of BG-1 on one RTX 5090
(`eval/cells/BG-1/reference.json` and `calibration-session-2.json`):

| cell | session A | session B | ratio |
|:--|--:|--:|--:|
| `dit-step/1024/bf16` | 0.261% | 0.420% | 1.6× |
| `t5-encode/1024/bf16` | 0.237% | 0.293% | 1.2× |
| `vae-decode/1024/bf16` | 0.039% | 0.064% | 1.6× |

A floor too tight would pay for noise permanently; one too loose only refuses a gain too small to
see.

**On the box:** never run two benchmarks at once (`tools/burnish` takes a lock and refuses a busy
device). Base and candidate always alternate, because the box drifts within a run. Kill by PID from
`nvidia-smi`; `pkill -f` over ssh kills your own session.

## What this does not claim

- **It is not a proof,** only agreement between independent measurers.
- **An audit checks arithmetic,** not whether a measurement happened.
- **Many simultaneous submissions are unmeasured.**
