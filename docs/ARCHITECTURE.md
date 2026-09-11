# How the pieces fit

Two halves that meet at exactly one place: a JSON line.

```
  RUNTIME (C++/CUDA)                          INSTRUMENT (Python)
  ────────────────────                        ───────────────────
  tools/burnisher_main.cpp                    tools/burnish
    │ info | selftest | bench | generate        │ screen | roofline | generation
    │ probe                                     │ probe | calibrate | gate | bench
    ▼                                           │ score | receipt | ledger
  src/models/  T5 · PixArt DiT · VAE            ▼
    │  explicit graphs over named ops         eval/runner.py    guards, GPU lock, env scrub
    ▼                                           │
  include/burnisher/registry.h                  ├─ eval/gate.py        correctness + determinism
    │  op → {name: implementation}               ├─ eval/calibrate.py   achieved + noise floor
    ▼                                           ├─ eval/bench.py       paired interleaved runs
  src/cpu/  the correctness ORACLE              ▼
  src/cuda/ device probe (op backend: TODO)   eval/burnscore/
                                                geometry  → flops and bytes per op
        ── BURNISH_JSON: {...} ──────────▶      roofline  → the arithmetic ceiling
           one line, per invocation             floor     → the measured noise floor
                                                bootstrap → the paired interval
                                                frontier  → latency × VRAM × fidelity
                                                compute   → all of it → a score
                                                receipt   → the evidence artifact
                                                ledger    → append-only history
```

## Why the seam is a single JSON line

The runtime prints exactly one `BURNISH_JSON: {...}` per invocation and the harness parses that
and nothing else. Parsing a number out of prose is how a harness silently accepts a changed output
format, so a missing or duplicated line is an error rather than a fallback to a regex.

That line always carries an `effective` block saying what the run **actually did** — which
implementation of each op, which dtype, which shape. `eval/runner.py` compares it against what it
asked for and refuses the run if they differ. An arm that silently fell back produces a perfectly
good number for a configuration nobody requested, and that is invisible from outside the process.

## The two numbers that must never be confused

Everything in `burnscore` exists to keep these apart:

| | where it comes from | `basis` |
|:--|:--|:--|
| **ceiling** | `max(flops / peak, unavoidable_bytes / bandwidth)`, from a config file | `model` |
| **achieved** | `ceiling / measured`, from a run on the pinned hardware | `measured` |

A ceiling is arithmetic and is never evidence about a speedup. `eval/tests/test_schemas.py` fails
if a modelled figure reaches a measured field, and `verify_receipt` refuses a receipt that tries.

## Why the geometry lives in the harness and not the runtime

`eval/burnscore/geometry.py` enumerates the ops of each stage with their flops and bytes, and
`src/models/*.cpp` executes them. The two are separate on purpose — the harness must be able to
compute a ceiling for a cell the runtime cannot yet run, which is what makes a *declared*
unimplemented cell possible and what makes cartography a scored contribution.

The cost is that they can drift. Three things hold them together:

- `geometry.selfcheck()` reproduces each model's published parameter count from the enumerated
  ops, and the test fails outside 1%.
- The op sequence in `pixart_dit.cpp` is written to be read line by line against the enumeration.
- `eval/make_generation.py --check` fails in CI if a config moved without the generation being
  regenerated.

## Why the instrument is overlaid from the base commit

`eval/`, `configs/`, `schemas/` and `tools/burnish` decide **what is measured**. If the evaluator
ran the submission's copy of them, a submission could win by editing the ruler — a noise floor, a
confidence level, a ceiling, a tolerance, a held-out list, the model revision. None of those look
like cheating in a diff and several look like tidying.

`eval/run_from_base.sh` is a script rather than a policy document because a rule nothing enforces
is a rule that holds only for honest submissions.

## Why a kernel is a registered name, not a replaced file

`include/burnisher/registry.h`. Base and candidate coexist in one binary, so a paired measurement
is one process, one model load, one thermal state. The old implementation stays runnable forever.
`--impl X` fails loudly when X is absent instead of silently measuring something else.

Registration is an explicit call in `register_builtin_cpu_ops()` rather than a file-scope object,
because a translation unit whose symbols are otherwise unreferenced is dropped from a static
archive — and then `--impl X` reports that X does not exist in a build that plainly contains it.
One line of bookkeeping beats an hour of confusion.
