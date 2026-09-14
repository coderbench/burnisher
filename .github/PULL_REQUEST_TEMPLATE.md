## What this changes

<!-- One or two sentences. Commit messages are one line, so the explanation goes here. -->

## Kind

- [ ] **Kernel:** a new kernel name, registered beside the existing ones
- [ ] **New cell:** see `docs/CARTOGRAPHY.md`
- [ ] **Instrument:** touches `eval/`, `configs/`, `schemas/`, `tools/burnish`, `scripts/` (except
      `scripts/build*`), `.github/` or `.gittensor/`. Not scored.
- [ ] **Docs, build or other**

## For a kernel

**Implementation name:** `<name>`

<!-- Read only when this pull request registers more than one new kernel name. -->

- [ ] `burnisher info` lists it
- [ ] `scripts/check.sh` passes
- [ ] Optional: measured on an RTX 5090, with the `tools/burnish receipt show` output pasted below
      (never retype a number)

## For an instrument change

- [ ] No guard is removed or relaxed, or the description names the incident it encodes and why it
      no longer applies
- [ ] No existing generation under `eval/cells/` is edited

## Checks

- [ ] Every commit message is one line, `<type>: <description>`, 72 characters or fewer
