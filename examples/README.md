# A real receipt, and the raw measurements it came from

These two files are one scoring run on the pinned hardware, kept as a worked example. They are
not part of any ledger: a live ledger is written outside the worktree, where a submission cannot
reach it (docs/SCORING.md). What they are for is that everything else in this repository
describes the scoring model, and this is the model having actually happened.

| file | what it is |
|:--|:--|
| `BG-1-pr-000001-raw.json` | 18 timing records -- 3 cells x 2 arms x 3 repeats, paired and interleaved -- plus 12 more at a held-out shape drawn at run time, the gate results, and the device fingerprint |
| `BG-1-pr-000001-receipt.json` | what `burnish score` made of them |

The whole run cost **16 minutes** of GPU wall clock for the bench
stage, recorded in the receipt rather than estimated. Each record carries its own `wall_s`
alongside what it measured, and the gap between the two is the evaluator's own overhead --
process start plus a 21.8 GB checkpoint map, about 11 s a record. A subnet that cannot say what
scoring costs cannot tell whether it can afford the submissions it is asking for.

`raw.json` is the measurement and the receipt is derived from it, never the other way around.
You can check that here, on a machine with no GPU:

```
tools/burnish score examples/BG-1-pr-000001-raw.json --generation BG-1 \
    --output /tmp/r.json --ledger /tmp/ledger --pr 1
```

Every figure comes out the same. Scoring is deterministic and does not touch hardware -- the
hardware's only job was to produce the raw file. A test pins this
(`eval/tests/test_example_receipt.py`), so the example cannot drift away from the scorer that
claims to produce it.

## What this run actually was, and why it is a regression

The candidate is `cuda-tile1024`: the same tiled attention kernel as the base, with the key/value
tile widened from 64 to 1024. It was chosen as the first thing ever scored because it is a real
change to a real kernel with no idea attached -- nobody expected it to win.

It lost, and the receipt says so:

```
  status            UNRESOLVED
  gap closed        -0.0056   (credited +0.0000)
  99% interval     [-0.0057, -0.0056]

    dit-step/1024/bf16          -0.0058     1.5% -> 1.0%     0.578%  yes
    t5-encode/1024/bf16         -0.0089    18.1% -> 17.3%    3.753%  yes
    vae-decode/1024/bf16        -0.0000     0.8% -> 0.8%     0.259%   NO
```

A 1024-wide tile spills the working set, and one DiT step goes from 3.58 s to 5.69 s. That is the
uninteresting part. The interesting part is what the three cells do differently:

- **Two cells resolved a regression and are named.** The change is far outside their floors, so
  the instrument says so plainly rather than reporting a number near zero.
- **`vae-decode` did not resolve.** The arms differ by about 0.2% against a 0.259% measured
  floor. That is not a small effect, it is *no measurement*: an axis whose spread sits inside its
  own noise is open, not solved. The cell contributes zero and does not block the submission --
  which is a bug this harness had and which a test now pins.
- **The regression credits zero, not negative.** `gap_closed` is reported at -0.0056 because that
  is what was measured; `credited_gap_closed` is 0.0. The ledger compounds toward a ceiling, and
  a mechanism that could be pushed backwards by a bad submission would let anyone move it.

A worked *win* would demonstrate less. This one exercises the paths a benchmark actually spends
its life in: a change that does not help, a cell that cannot tell, and an honest zero.

## What makes this receipt evidence rather than just a measurement

`provenance.code_provenance_complete` is `true`. The run went through
`eval/score_submission.sh`, which drives `eval/run_from_base.sh` at every stage, so the receipt
names three commits: the candidate, the base, and -- the one that matters -- the commit the
INSTRUMENT came from.

That last field is what says the submission did not grade its own homework. `eval/`, `configs/`,
`schemas/` and `tools/burnish` decide *what is measured*; they are overlaid from the base ref
before anything runs, so a one-line edit to a noise floor, a ceiling, a tolerance or the model
revision cannot reach the scorer. None of those look like cheating in a diff. Several look like
tidying.

The first two receipts this harness ever produced had all three fields null, and still verified.
They were valid measurements and they were not evidence about any particular commit. The receipt
now says which of the two it is, in one flag, instead of leaving it to be inferred from three
absences.
