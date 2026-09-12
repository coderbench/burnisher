# How a submission is evaluated, and how you check the answer without a GPU

Most scored benchmarks work like this: a bot measures your change, the bot decides a grade, and
the grade is the record. To check the grade you have to reproduce the measurement, which means
owning the hardware. Everyone else is asked to trust the bot.

Burnisher inverts that, and not by being clever — by an accident of arithmetic worth naming.

**Scoring here is a pure function of recorded measurements.** No device, no clock, no randomness.
A receipt scored on a Blackwell part re-derives, figure for figure, on a laptop. So the record is
not the verdict. The record is the **measurements**, and the verdict is a derivation anybody can
run in about two seconds.

```bash
burnish audit pr-000042-raw.json pr-000042.json
```

That re-scores the published measurements and compares the result to the published receipt, field
for field. It needs no GPU, no network, and no trust in whoever produced the receipt.

---

## The two tiers, and what each one actually proves

| | **arithmetic** | **measurement** |
|:--|:--|:--|
| question | did the published numbers produce the published verdict? | do those numbers describe what the hardware did? |
| who | anyone | anyone with an RTX 5090 |
| cost | seconds, no GPU | ~25 minutes |
| command | `burnish audit` | `burnish challenge` |
| catches | scoring bugs, edited receipts, a verdict that does not follow from its own data | a measurement that never happened |

**The cheap tier catches most of what goes wrong,** because most of what goes wrong is a bug in
the evaluator rather than a liar. It also catches a deliberately edited receipt — including one
whose content digest has been recomputed to cover the edit, which defeats `burnish receipt
verify` and does not defeat this, because the score still has to follow from the measurements
committed beside it.

**The expensive tier is the only thing that settles the rest.** No amount of arithmetic can prove
a measurement happened. Saying so is cheaper than being found out.

---

## What happens to a pull request

1. **Guard.** Does the submission change the measuring instrument? If so it is **skipped, not
   closed**, and no GPU time is spent on it — a number produced by a modified instrument cannot
   be accepted either way, so measuring it first would be waste rather than diligence.
2. **Build** from source on the eval box. No prebuilt artifacts.
3. **Gate** correctness and self-determinism, for **both** arms, before anything is timed. A
   baseline that does not reproduce itself makes every delta measured against it noise.
4. **Bench** paired and interleaved, with a held-out shape drawn *now* — after your code is
   frozen.
5. **Score** into an append-only ledger outside the submission's reach.
6. **Publish** the raw measurements *and* the receipt, so the verdict can be re-derived.
7. **Label** — a pure function of the receipt.

Steps 2–5 are `eval/score_submission.sh`, which is the same command you can run by hand. The bot
is not a privileged path; it is a scheduled one.

---

## What the label means

A paying outcome carries **the number**, not a grade:

```
burnish:gap+0.0342
```

That is the fraction of this generation's **remaining** arithmetic-roofline gap that your change
closed. It is the payout basis itself — not an input to a tier table.

There is no XS/S/M/L/XL here and the absence is deliberate. A tier boundary pays two
differently-measured submissions identically, and two almost-identical ones differently. The
number is already in `[0, 1]`, already comparable across cells and models, and already
self-terminating — the same kernel win is worth less once the gap it closes is smaller. Bucketing
it discards the only property that makes it worth computing.

Every other outcome carries a reason instead of a number, because a number published about a
submission that was never resolved, never correct, or never measured would be read as a score:

| label | meaning |
|:--|:--|
| `burnish:gap+N.NNNN` | **paid.** The number is the fraction of the remaining gap closed. |
| `burnish:cell-opened` | **paid as cartography.** You opened a new cell. |
| `burnish:unresolved` | Not paid, and **not a judgement about the idea** — the effect could not be told from this cell's own measured noise. |
| `burnish:no-gain` | Not paid. Resolved, and measurably not an improvement. A real result. |
| `burnish:moved-along-frontier` | Not paid. Faster, but paid for in memory or fidelity. |
| `burnish:shape-overfit` | Not paid. The gain did not survive a shape drawn after the freeze. |
| `burnish:partial` | Not paid. An incomplete matrix credits nothing. |
| `burnish:correctness-fail` | Rejected before timing was considered. |
| `burnish:determinism-fail` | Rejected. The build does not reproduce itself. |
| `burnish:held` | An independent re-measurement disagrees beyond the floor. |
| `burnish:skipped-instrument` | Changes the instrument. Not evaluated, no GPU spent. |

---

## Disputing a result

If you think a published score is wrong, measure it yourself:

```bash
eval/score_submission.sh --base <the base it was scored against> --worktree . \
    --impl-base cuda --impl-candidate <the same impl> \
    --pr <n> --ledger /tmp/mine --weights <ckpt> --noise <noise.npy>

burnish challenge /tmp/mine/BG-1/receipts/pr-000042.json --ledger <the public ledger>
```

Two rules make a challenge evidence rather than noise:

- **It must measure the same thing** — same candidate commit, same implementations, same frozen
  generation. Anything else is a different experiment, and filing it as a challenge would hold a
  submission hostage to an unrelated result.
- **It must be a different physical card.** Identified by GPU UUID, not model name: two RTX 5090s
  differ by about 3% on achievable GEMM, which is larger than most cells' floors, and that spread
  is exactly what a challenge exists to surface. Re-running on the same card re-measures the same
  silicon and the same thermal regime.

### When two receipts disagree, the credit is **held** — not rejected

The threshold is the cell's **own measured noise floor**, converted into gap-closed. Not a
tolerance anybody chose: the floor is what the calibration measured running the unmodified base
against itself, so a disagreement inside it is two measurements of the same thing, and one
outside it is not.

If they disagree, at least one receipt is wrong and nobody yet knows which. Rejecting would
punish a contributor for an evaluator's bad afternoon. Paying would pay for a number nobody can
reproduce. So the ledger does neither: it holds the credit, says why, and waits for a third
measurement to break the tie.

A score no one else can reproduce does not pay, and does not have to be called fraud to be
refused.

---

## The instrument, and the one exception

`eval/`, `configs/`, `schemas/` and `tools/burnish` decide **what is measured**. A submission that
could edit them could win by editing the ruler — a one-line change to a noise floor, a confidence
level, a ceiling, a tolerance, a held-out shape list, or the model revision. None of those look
like cheating in a diff. Several look like tidying.

Two independent mechanisms, because one is not enough:

- `eval/run_from_base.sh` **overlays the instrument from the base commit** before scoring, so such
  an edit cannot affect its own author's score.
- A required CI check **blocks the merge**, because "it didn't help you" is weaker than "it didn't
  happen": a change that lands on main becomes the instrument for everybody after it.

**The exception is the point.** Burnisher pays for cartography — opening a new cell is a scored
contribution, because a benchmark whose surface only a maintainer may extend stops growing. So
the guard distinguishes:

| | |
|:--|:--|
| **adding** a new generation under `eval/cells/<new>/` | cartography. Allowed, and paid. |
| **modifying** a generation that already exists | blocked. Editing one silently re-scores history. |
| **modifying** anything else in the instrument | blocked. |
| **adding** a file elsewhere in the instrument | blocked — a second scorer beside the first is a modification wearing a hat. |

An added generation cannot change what any existing receipt meant, because generations are frozen
and receipts stay attached to the one that produced them. That property is what makes "added"
safe and "modified" not, and it is why the rule can be this simple.

Improving the evaluator is a real contribution — the evaluator is where the bugs are, and a broken
one prints a confident number. It is separated, not refused: send it as its own pull request,
scored as a change to what is measured.

---

## What this does not claim

- **It is not a proof.** There is no cryptographic proof of a benchmark number, only agreement
  among independent measurers. Anyone claiming otherwise about wall-clock on a consumer card is
  selling something.
- **The cheap tier does not verify measurements.** It verifies that a verdict follows from data.
  Those are different claims and conflating them would be the exact failure this design exists to
  avoid.
- **One box is one box.** Every figure in this repository comes from a single RTX 5090 running one
  submission at a time. The calibration is pinned to that physical card; a validator on different
  silicon must run `burnish probe` and `burnish calibrate` first, or the drift guard will refuse
  to score rather than quietly measuring from the wrong denominator.
