# Paying for maps

A contributor who adds a cell nobody had measured — a new model, a new resolution, a new dtype —
and lands its reference implementation and its roofline earns credit for it.

## Why this is a scored contribution rather than an admin chore

The scoring model is designed so that exhausted cells stop paying: as a cell approaches its
roofline the remaining gap shrinks, the physically available speedup shrinks toward `1/achieved`,
and the cell's own noise floor eventually eats whatever is left. That is the right behaviour and
it has a consequence — **the benchmark has to keep growing, or the incentive dies with the last
cell.**

SparkInfer's maintainers currently manage axis supply by hand. Making it a paid contribution turns
the thing that keeps the subnet alive from a maintainer's chore into something a contributor
competes to do.

## What counts as a cartography contribution

Landing all four of these for a cell that did not previously exist:

1. **The geometry.** An op enumeration in `eval/burnscore/geometry.py` that reproduces the model's
   published parameter count. `selfcheck()` compares the enumerated geometry against the
   checkpoint's file size and the test fails outside 1%. That check is not ceremony: a geometry
   that reproduces a 610,000,000-parameter checkpoint to four decimal places is one that read the
   config correctly, and one that does not is a table of confident fiction.
2. **The reference implementation.** A path through the runtime that actually runs the cell,
   registered as a named implementation, passing the correctness gate against a pinned reference.
3. **The roofline.** Which follows from (1) and a device spec, and is generated rather than typed.
4. **The calibration.** `burnish calibrate` on the pinned hardware: the achieved fraction and the
   measured noise floor, with enough repeats that the floor is itself stable.

A cell with (1) and (3) but not (2) and (4) is a **declared** cell. It is published with its
ceiling, `implemented: false`, and weight 0 — the room is visible and the cell cannot drag an
aggregate it is not part of. `dit-step/1024/fp8` and `dit-step/1024/nvfp4` are both in that state
today. Declaring a cell nobody can run is deliberate and is the opposite of overselling: it shows
the room and marks plainly that no reference exists.

## The honest outcome is sometimes "this cell is not worth having"

A calibration that comes back with a noise floor larger than a twentieth of the cell's remaining
room means the cell **cannot resolve a contribution**. `burnish calibrate` says so and
`burnish roofline` prints `res: no`.

**That is a successful cartography contribution, not a failed one.** It is worth more than a cell
that looks open and is not, because the alternative is a contributor spending a week inside the
noise floor. Publish it as unresolvable, with the numbers, and say what would have to change —
a longer run, a quieter box, a different shape — for it to become measurable.

The engagement this harness came from found seven of ten cells in that state, said so in its own
verdict document, and that was the most useful thing it produced.

## How it is scored

A cartography contribution does not close a gap in an existing cell, so it does not produce a
gap-closed number. It is scored against the generation it creates:

- A new cell in a **new generation** (BG-N+1) carries its calibration and its ceiling, and every
  later submission in that cell is measured against them.
- The contributor who landed it is recorded in the generation's provenance.
- Because a generation is frozen for its lifetime, a cell landed badly is expensive to fix — which
  is why (1) through (4) are all required before it counts, rather than being filled in later.

## How a cartography submission is actually evaluated

```bash
burnish cartography check --generation BG-2 --base origin/main --measure \
    --binary build-cuda/burnisher --weights <ckpt> --noise <noise.npy>
```

The evaluator asks a different question than it asks a speedup — *is this cell real, and can
anybody be credited on it?* — and the work is split deliberately:

| | |
|:--|:--|
| **you supply** | the cell definition and the **oracle** — what a correct runtime must reproduce |
| **the evaluator supplies** | every **measurement**. Your `reference.json` is read, reported, and discarded; the cell is gated and recalibrated here. |

That asymmetry is the security argument, and it is why the structural checks can afford to be
permissive about what you propose. **A submission arriving with a floor of 0.00001% gains nothing
by it** — the floor that ends up frozen into the generation is the one this box measured.

What is checked before anything is run:

- the generation does not already exist on the base (an existing one is an edit wearing a new name);
- it declares at least one cell that does not already exist (re-measuring covered ground is not cartography);
- **every submitted ceiling recomputes from the base configs**, to floating-point noise — a submission that could pick its own ceiling would pick its own denominator for every score in that cell, forever;
- reference latents are present with a manifest.

Then, on the hardware: the cell runs, reproduces itself byte for byte, passes the gate against
your oracle, and is calibrated. A cell that fails any of those is not opened.

**`eval/run_from_base.sh` keeps a generation the submission adds** — and only one that is absent
from the base. Everything else in the instrument still comes from the base commit. An added
generation is safe precisely because it cannot change what any existing receipt meant.

### A new model is not automatable, and is not pretended to be

Requirement (1) above puts a new op enumeration in `eval/burnscore/geometry.py`, which is
instrument — the guard blocks it, correctly, because a geometry that miscounts a stage moves every
ceiling computed from it. That path needs a maintainer.

A new **resolution, dtype or stage of a model already enumerated** needs no code at all: the
geometry is parameterised, so `pixart_stages(..., resolution=512)` already produces the right
ceilings. That is the case this evaluates end to end.

## What does not count

- Adding a cell to an existing frozen generation. Generations do not change; receipts stay
  attached to the one that produced them. `eval/run_from_base.sh` refuses a submission that edits
  anything under `eval/cells/`.
- A cell whose ceiling is computed but whose calibration is guessed. The scorer refuses to produce
  a receipt against an uncalibrated cell and raises rather than substituting a constant.
- A cell that duplicates an existing one at a shape the held-out guard already draws from.
