# How a submission is evaluated

## What happens to a pull request

1. **Guard.** If it changes the measuring instrument it is labelled `burnish:skipped-instrument`,
   and no GPU time is spent.
2. **Build** from source on the evaluation box.
3. **Gate** correctness and determinism for both base and candidate (`docs/CORRECTNESS.md`).
4. **Bench.** Base and candidate runs alternate, plus a held-out shape drawn after the code is
   frozen.
5. **Score** into an append-only ledger outside the repository.
6. **Publish** the raw measurements and the receipt.
7. **Label** the pull request from the receipt.

Steps 2–5 are `eval/score_submission.sh`, the same command anyone can run.

## Rounds

- **Every two hours, three submissions, oldest first** (`eval/run_round_cron.sh`). One submission
  costs about 24 GPU-minutes, and two benchmarks cannot share a GPU.
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
| `burnish:shape-overfit` | The gain vanished at the held-out shape. |
| `burnish:partial` | Not every cell was run. |
| `burnish:correctness-fail` | Failed the gate. Rejected. |
| `burnish:determinism-fail` | Did not reproduce itself. Rejected. |
| `burnish:held` | An independent re-measurement disagrees. Waiting for a third. |
| `burnish:skipped-instrument` | Changes the instrument. Not evaluated. |
| `burnish:build-fail` | Did not build. Not evaluated. |
| `burnish:eval-error` | The evaluator's fault, not yours. Re-run. |

Colours: green = paid or merge-first (darker is bigger), blue = measured but not paid, pale blue =
could not tell, red = correctness or determinism failure, amber = shape overfit or needs rebase,
purple = disputed, orange = build or evaluator failure, grey = not evaluated. They are defined in
`eval/burnscore/verdict.py`.

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

- The raw file carries the validator's calibration, so an audit works on any machine.
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

**Every validator calibrates their own card.** The ceiling and the measurement then both come
from the same card and cancel out. Scoring a card 3% slower against someone else's calibration
shifts every score by 6%. Against its own, 0.0000%. The scorer refuses a calibration from a
different GPU UUID.

```bash
scripts/build_cuda.sh               # CMAKE_CUDA_ARCHITECTURES=121 for DGX Spark
build-cuda/burnisher probe > probe.txt                       # this card's real peaks
scripts/apply_probe.py probe.txt --device rtx5090 --write
burnish calibrate --repeats 9 --weights <checkpoint> --output cal.json      # about 30 minutes
burnish calibrate --repeats 9 --weights <checkpoint> --merge cal.json --output cal.json
burnish score <raw.json> --calibration cal.json
```

Keep `cal.json` beside your ledger, not in the repository.

**Calibrate more than once.** Noise floors move between sessions on the same card:

| cell | session A | session B | ratio |
|:--|--:|--:|--:|
| `dit-step/1024/bf16` | 0.578% | 0.205% | 2.8× |
| `t5-encode/1024/bf16` | 3.753% | 0.155% | 24.2× |
| `vae-decode/1024/bf16` | 0.259% | 0.845% | 3.3× |

`--merge` keeps the worst floor. A floor too tight would pay for noise permanently; one too loose
only refuses a gain too small to see.

**Recalibrate** after a driver or card change. If the base arm drifts more than three floors from
the calibration, scoring stops.

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
