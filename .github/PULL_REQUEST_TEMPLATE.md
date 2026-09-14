## What this changes

<!-- One or two sentences. Commit messages are one line, so the explanation goes here. -->

## Kind

- [ ] **Kernel** — registered under a new name beside the existing one
- [ ] **New cell** — see `docs/CARTOGRAPHY.md`
- [ ] **Instrument** — touches `eval/`, `configs/`, `schemas/`, `tools/burnish`, `scripts/`, `.github/`
      or `.gittensor/` (not scored)
- [ ] **Docs / build / other**

## For a kernel

**Implementation name:** `<name>`

<!-- The validator measures the new kernel name this PR registers. The field is read only when the
PR registers more than one new name. -->

- [ ] `burnisher info` lists it
- [ ] `scripts/check.sh` passes

Optional — the validator measures every submission. If you measured it on an RTX 5090, paste the
`tools/burnish score` summary. Never type a number by hand.

```
```

## For an instrument change

- [ ] I did not remove or relax a guard, or I explain which incident it encodes and why it no
      longer applies
- [ ] I did not edit an existing generation under `eval/cells/`

## Checks

- [ ] Commit messages are one line, `<type>: <description>`, 72 characters or fewer
