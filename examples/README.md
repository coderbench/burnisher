# Example receipt

One real scoring run on the pinned RTX 5090, kept as a worked example. It is not part of any
ledger.

| file | what it is |
|:--|:--|
| `BG-1-pr-000001-raw.json` | the measurements: 3 cells × 2 arms × 3 repeats, a held-out shape, the gate results and the device |
| `BG-1-pr-000001-receipt.json` | what `burnish score` made of them |

Re-score it on any machine, no GPU. Every figure comes out the same, and
`eval/tests/test_example_receipt.py` checks it:

```
tools/burnish score examples/BG-1-pr-000001-raw.json --generation BG-1 \
    --output /tmp/r.json --ledger /tmp/ledger --pr 1
```

## What happened

The candidate was `cuda-tile1024`: the same attention kernel with a wider tile. It was slower.

```
$ tools/burnish receipt show examples/BG-1-pr-000001-receipt.json
  status            UNRESOLVED
  gap closed        -0.0056   (credited +0.0000)
  99% interval     [-0.0056, -0.0056]
  resolved          False
  frontier          -0.94392 (not expanded)

  per cell:
    cell                            gap           achieved   floor  res
    dit-step/1024/bf16          -0.0058     1.5% -> 1.0%     0.578%  yes
    t5-encode/1024/bf16         -0.0090    18.2% -> 17.4%    3.753%  yes
    vae-decode/1024/bf16        -0.0000     0.8% -> 0.8%     0.845%   NO
```

- **Two cells resolved the regression.** It was far outside their noise floors.
- **`vae-decode` did not resolve.** The difference was inside its floor, so the cell contributes
  zero without blocking the others.
- **The regression credits zero, not negative.**
- **The receipt names the code it scored:** `code_provenance_complete` is true, with the
  candidate, base and instrument commits.
