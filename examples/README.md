# Example receipt

One real scoring run on the pinned RTX 5090, kept as a worked example and as the fixture the scoring
tests read. It is not part of any ledger.

| file | what it is |
|:--|:--|
| `BG-1-pr-000001-raw.json` | the measurements: 3 cells × 2 arms × 3 repeats, a held-out shape, the gate results, the device and the anchor they were scored against |
| `BG-1-pr-000001-receipt.json` | what `tools/burnish score` made of them |

Re-score it on any machine, no GPU. Every figure comes out the same, and
`eval/tests/test_example_receipt.py` checks it:

```
tools/burnish score examples/BG-1-pr-000001-raw.json --generation BG-1 \
    --output /tmp/r.json --ledger /tmp/ledger --pr 1
```

## What happened

Scored at `5a71998`, with `cuda` as the base and the first GroupNorm kernel as the candidate: one
shared-memory block per group, accumulating in float. It has since been removed.

```
$ tools/burnish receipt show examples/BG-1-pr-000001-receipt.json

  status            NO_GAIN
  gap closed        -0.0042   (credited +0.0000)
  99% interval     [-0.0047, -0.0034]
  resolved          True
  frontier          -0.03738 (not expanded)

  per cell:
    cell                            gap           achieved   floor  res
    dit-step/1024/bf16          +0.0001    55.6% -> 55.6%    0.420%   NO
    t5-encode/1024/bf16         +0.0007    68.7% -> 68.7%    0.293%   NO
    vae-decode/1024/bf16        -0.1136    25.4% -> 16.9%    0.064%  yes
```

- **`vae-decode` resolved the regression.** The candidate's GroupNorm made the decode slower by far
  more than that cell's noise floor, so the submission resolved as measurably slower: `NO_GAIN`.
- **`dit-step` and `t5-encode` did not resolve.** Their norms are LayerNorm and RMSNorm, the same
  kernel in both arms, so the difference sat inside their floors. Each contributes zero without
  blocking the others.
- **Nothing is credited.** `NO_GAIN` does not pay. The resolved regression counts against the
  measured figure rather than being zeroed.
- **The receipt names the code it scored:** `code_provenance_complete` is true, with the
  candidate, base and instrument commits.
