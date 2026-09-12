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

**The raw file is self-contained on purpose.** It carries the validator's calibration inside it,
not a reference to one. Every validator calibrates their own card, so a file that merely *named*
a calibration could be re-derived by exactly one person — the validator who produced it — and the
free tier would be a tier of one. Raw file plus frozen generation is everything needed, on any
machine, forever.

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

## Rounds: every two hours, three at a time, oldest first

Submissions are not evaluated on arrival. Two benchmarks cannot run at once — they race for VRAM
and both results are worthless — so throughput is a hard constraint, and a submission costs about
**24 measured GPU-minutes**.

```
0 */2 * * *  eval/run_round_cron.sh
```

| | |
|:--|:--|
| **cadence** | every two hours |
| **slots** | **three**. Three fit the interval with 41 minutes of slack; five overrun it outright at the measured cost |
| **order** | first in, first out |
| **the rest** | wait for the next round — nothing is dropped |

Three rather than five because the slack matters more than the throughput: a round that runs past
its interval meets the next one holding the lock, and the next one skips. The slot count should
rise as the runtime gets faster — it is the one quantity here that improves without anyone
working on it directly.

**FIFO, not "most promising first".** Any ordering that reads the submission to decide whether to
measure it is an ordering somebody can game, and it makes the wait a function of the evaluator's
opinion rather than of the queue. A submission the instrument guard skips costs nothing and does
not consume a slot.

**A second round never queues behind the first.** It takes a lock and exits. Waiting an hour and
then running would measure against a `main` that moved while it waited; the next tick is two
hours away and the work is still there.

### At most one merge per round, and why

**Gains do not compose.** The ledger compounds toward the ceiling, so two submissions each closing
20% of the remaining gap close 36% together, not 40%. And two wins can overlap *entirely* — fused
AdaLN and CUDA-graph capture both attack launch overhead, so a gain measured against the old
`main` may be worth nothing once the other has landed.

So the largest credited gain in a round is merged, and every other scored submission gets
`burnish:needs-rebase`. That label is **not a rejection**: the result was correct, it is simply a
measurement of a baseline that no longer exists. Rebase and it is re-measured.

**A round where nothing resolved a gain merges nothing.** Giving away a merge for an unresolved
measurement is paying for a number nobody could distinguish from a quiet afternoon.

Merging is opted into (`--merge`), not assumed — a round that merges unattended is an
outward-facing action.

### The head commit is frozen when the round starts

Later pushes cannot reach the measurement: the worktree is built from the SHA resolved at round
start. But GitHub does not remove a label when you push, so **every comment names the commit it
measured**. A verdict overtaken by a push is visibly stale rather than quietly wrong.

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

### The colours mean something

Labels are coloured by what the outcome **means to you**, not by severity — severity colouring
would paint `unresolved` and `correctness-fail` the same alarming red, when one is *"we could not
measure your idea"* and the other is *"your change is incorrect"*.

| | |
|:--|:--|
| green | paid — a gain, or a cell opened |
| blue | a real measurement that did not pay |
| pale blue | we could not tell |
| red | rejected on correctness or determinism |
| amber | rejected by a guard (shape overfit) |
| purple | disputed — an independent re-measurement disagrees |
| orange | the evaluator's own fault; not yours |

**The paying label's shade carries the magnitude**: a bigger contribution is a deeper green, on a
log scale, because real values span orders of magnitude — the tightest cell's noise floor is worth
0.00009 of the gap and a large win is 0.1. So a list of pull requests is readable at a glance
before anybody opens one.

That label cannot be pre-registered — it carries the measured number, so there is one per value —
and a label GitHub creates on first use gets a **random** colour. The bot therefore creates it
with the right one before attaching it. Colours and meanings both live in
`eval/burnscore/verdict.py`; `eval/setup_labels.sh` reads them from there rather than keeping a
second copy, because a label with a stale colour still works and so nobody notices it has come to
mean something else.

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
| **adding** a new generation under `eval/cells/<new>/` | cartography. Allowed, and paid — `burnish cartography check` is the gate, and it measures the cell itself rather than believing what was submitted. |
| **modifying** a generation that already exists | blocked. Editing one silently re-scores history. |
| **modifying** anything else in the instrument | blocked. |
| **adding** a file elsewhere in the instrument | blocked — a second scorer beside the first is a modification wearing a hat. |

An added generation cannot change what any existing receipt meant, because generations are frozen
and receipts stay attached to the one that produced them. That property is what makes "added"
safe and "modified" not, and it is why the rule can be this simple.

Improving the evaluator is a real contribution — the evaluator is where the bugs are, and a broken
one prints a confident number. It is separated, not refused: send it as its own pull request,
scored as a change to what is measured.

### Can a submission raise its own score by editing something?

**No, and not because of the guard.** Two mechanisms on the validator's side settle it before the
guard is even consulted:

- the evaluator runs **its own copy** of the guard against the submission's tree, not the
  submission's copy of it;
- `run_from_base.sh` **overlays the instrument from the base commit** before anything is measured.

Nothing in a pull request reaches either. A submission that rewrote every rule in this repository
would be measured by the rules on `main` regardless.

**What a submission can do, if it merges unreviewed, is poison the instrument for whoever submits
next.** Disable the CI check that enforces the guard; rewrite the guard's own rules; neuter the
check suite; restate the declared scoring model. None of that helps its author — which is exactly
why it is worth blocking, because it is the slower and better-disguised version of the same
attack. So those paths are guarded too, and `.github/CODEOWNERS` is a second lock: a CI check can
be edited in the same commit that needs it edited, an ownership rule is enforced by the forge.

| | |
|:--|:--|
| `eval/` `configs/` `schemas/` `tools/burnish` | **what is measured** — guarded *and* overlaid from base |
| `.github/` `.gittensor/` `scripts/` | **how the evaluation is governed** — guarded; cannot help the author, can help the next one |
| `scripts/build*` `CMakeLists.txt` `src/` `include/` `tests/` | contributor surface |

### The honest limit

The evaluator **builds the submission from source**, so `CMakeLists.txt` and the build scripts run
code the submitter wrote, on the eval box, by design. Every benchmark that builds from source has
this exposure. It cannot manufacture a speedup — base and candidate are two registered
implementations of one binary, so a compiler flag moves both arms equally and a correctness change
is caught by the gate — but it is code execution. Isolate the eval box accordingly. This guard is
not a sandbox and should not be mistaken for one.

---

## What this does not claim

- **It is not a proof.** There is no cryptographic proof of a benchmark number, only agreement
  among independent measurers. Anyone claiming otherwise about wall-clock on a consumer card is
  selling something.
- **The cheap tier does not verify measurements.** It verifies that a verdict follows from data.
  Those are different claims and conflating them would be the exact failure this design exists to
  avoid.
- **Queueing is unmeasured.** Every figure here comes from one box running one submission at a
  time. What a single submission costs is measured (~25 minutes); what happens when many arrive
  at once is not.


---

## Running a validator: calibrate your own box first

**Every validator calibrates their own hardware, and the score comes out the same anyway.** That
is not a concession to operational reality — it is the arithmetic working.

`achieved = ceiling / measured`, and both halves are properties of the card. Probe your own box
and both scale with it and cancel. Use somebody else's ceiling with your own measurement and they
do not:

| | a card 3% slower than the reference |
|:--|:--|
| scored against the reference's calibration | **6% different score**, systematically, forever |
| scored against its own calibration | **0.0000% different** |

Two RTX 5090s really do differ by about 3% on achievable GEMM — `configs/devices.json` records
the measurement. A 6% bias is not noise a bootstrap absorbs and not something an interval
reveals; every submission a slower validator happened to pick up would pay less than the same
submission elsewhere, and nothing in the receipt would say so.

So the scorer **refuses** a run whose GPU UUID does not match the calibration it is being scored
against. Loud failure instead of a quiet 6%.

### Setup, once per box

```bash
# 1. measure this card's real peaks -- the ceilings are computed from them
burnish probe --write

# 2. measure each cell's achieved fraction and its own noise floor  (~30 min)
burnish calibrate --repeats 9 --output /var/burnish/calibration.json

# 3. point everything at it
export BURNISH_CALIBRATION=/var/burnish/calibration.json
```

Keep the calibration **beside your ledger, not in the repository**. It describes your card. The
committed `eval/cells/BG-1/reference.json` is the reference device's, kept so the repository's
own published tables have something to stand on — it is not a default that happens to work for
you.

### The floor is not a permanent property of your box

Two calibrations of the same RTX 5090, hours apart, same driver, same build:

| cell | session A | session B | ratio |
|:--|--:|--:|--:|
| `dit-step/1024/bf16` | 0.578% | 0.205% | 2.8× |
| `t5-encode/1024/bf16` | 3.753% | 0.155% | **24.2×** |
| `vae-decode/1024/bf16` | 0.259% | 0.845% | 3.3× |

Meanwhile the achieved fractions held to three significant figures — 1.52/1.52, 18.15/18.13,
0.79/0.79.

That asymmetry is expected rather than alarming: `achieved` is a median and robust, a floor is a
spread over nine repeats and is not. But it means **whether a submission resolves can depend on
which session you happened to calibrate in**, which is not a property a benchmark should have.

So calibrate more than once and fold the sessions together:

```bash
burnish calibrate --repeats 9 --output cal.json
# later, on the same box
burnish calibrate --repeats 9 --merge cal.json --output cal.json
```

`--merge` keeps the **worst** floor per cell. The two errors are not symmetric: a floor that is
too tight credits noise as a contribution and the ledger compounds it permanently, while one that
is too loose refuses a gain too small to see and the contributor comes back with a bigger one.
Only one of those is recoverable.

### Recalibrate when the box changes

A driver update, a different card, a new thermal regime. You do not have to guess when: the drift
guard measures it. If the base arm moves more than three noise floors from where the calibration
says it should be, scoring stops and tells you, rather than computing every score from a stale
denominator.

The two checks answer different questions and both are needed:

| check | question | settled by |
|:--|:--|:--|
| calibration identity | is this calibration even about this machine? | the GPU UUID |
| drift guard | has this machine changed since? | a measurement |

### What every receipt now says

Each receipt records the calibration it was scored against — device, name, driver. Two validators
scoring the same submission produce two receipts with two different calibrations and, if both
boxes are honest, the same number. That is what makes `burnish challenge` meaningful: a
disagreement is about the measurement, not about whose card it ran on.
