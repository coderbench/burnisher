## What this changes

<!-- One or two sentences. The commit message is one line with no body, so this is where the
     explanation goes. -->

## Kind of contribution

- [ ] **Kernel / optimization** — registered as a new named implementation beside the existing one
- [ ] **Cartography** — a new cell, with its geometry, reference implementation, roofline and calibration (`docs/CARTOGRAPHY.md`)
- [ ] **Instrument** — a change to `eval/`, `configs/`, `schemas/` or `tools/burnish`
- [ ] **Docs / build / other**

## For a kernel change

**Implementation name:** `<the name you registered>`

- [ ] Registered a new name; did **not** replace an existing implementation
- [ ] `burnisher info` lists it
- [ ] `burnish gate --impl <name> --repeats 10` passed — determinism **and** correctness
- [ ] `burnish bench --impl-candidate <name> --gate-result gate.json` run on the pinned hardware
- [ ] Held-out shapes were run (no `--skip-held-out`)

Paste the receipt summary from `burnish score`:

```
  status
  gap closed
  99% interval
  resolved
  frontier

  per cell:
```

**Do not type any number into this PR by hand.** Every figure comes from the receipt, and the
receipt from raw measurements that are themselves in the receipt.

## For an instrument change

`eval/run_from_base.sh` overlays the instrument from the base commit, so this change is reported
and then discarded for scoring purposes. That is not a rejection — improving the evaluator is a
real contribution, and the evaluator is where the bugs are.

- [ ] I did not remove or relax a guard
- [ ] …or, if I did: which incident does it encode, and why does it no longer apply?

<!-- Guards are named in comments where they live. That comment is the first thing to read. -->

- [ ] I did not edit anything under `eval/cells/` (a frozen generation cannot change; the answer
      is a new generation)

## Checks

- [ ] `scripts/check.sh` passes (no GPU needed)
- [ ] Commit messages are a single line, `<type>: <description>`, 72 chars or fewer, no body, no
      trailers
