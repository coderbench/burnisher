#!/usr/bin/env python3
"""Which paths a submission may touch, and the one distinction that makes this repo different.

    scripts/instrument_guard.py --base origin/main
    scripts/instrument_guard.py --base origin/main --json verdict.json

The instrument
--------------

`eval/`, `configs/`, `schemas/` and `tools/burnish` decide WHAT IS MEASURED. A submission that
could edit them could win by editing the ruler: a one-line change to a noise floor, a confidence
level, a roofline ceiling, a tolerance, a held-out shape list or the model revision. None of
those look like cheating in a diff. Several look like tidying.

`eval/run_from_base.sh` already overlays all of it from the base ref before scoring, so editing
the instrument cannot affect the editor's own score. This guard exists because that is not
enough on its own: a change that lands on main becomes the instrument for everybody AFTER it,
and "it didn't help you" is a weaker property than "it didn't happen".

The distinction this repository has to make and most benchmarks do not
---------------------------------------------------------------------

Burnisher PAYS for cartography. Adding a cell -- a new resolution, dtype or stage, with its
reference latents, its calibration and its roofline -- is a scored contribution, because the
subnet's health depends on axis supply and making that an admin chore is how a benchmark stops
growing. So a blanket "contributors may not touch the instrument" would forbid one of the two
things this benchmark is trying to buy.

The line is ADD versus MODIFY, and git can see it exactly:

    adding a new generation under eval/cells/<NEW>/     cartography. Allowed, and paid.
    modifying anything that already exists              editing the ruler. Blocked.
    deleting anything in the instrument                 blocked, always.
    adding a new file elsewhere in the instrument       blocked -- a new scorer beside the old
                                                        one is a modification wearing a hat.

An added generation cannot change what any existing receipt meant, because generations are
frozen and receipts stay attached to the one that produced them. That is the property that makes
"added" safe and "modified" not, and it is why the rule can be this simple.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Everything that decides what is measured, as opposed to how well the runtime does it.
# One list, shared with eval/run_from_base.sh's overlay -- a path guarded here but not overlaid
# there is a hole, and one overlaid but not guarded is a silent discard.
INSTRUMENT = ("eval/", "configs/", "schemas/", "tools/burnish")

# The contributor surface: the runtime itself, which is what a submission is scored on.
CONTRIBUTOR = ("src/", "include/", "tests/", "CMakeLists.txt", "scripts/build")

# A new frozen generation: eval/cells/<NAME>/... where <NAME> did not exist on the base.
CELL_DIR = re.compile(r"^eval/cells/([^/]+)/")


def changed(base: str) -> list:
    """(status, path) for every file this branch changes against the base.

    `--diff-filter` is deliberately not used: the statuses are the whole point, and a rename is
    reported as its own letter rather than being silently split into an add and a delete.
    """
    out = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--name-status", f"{base}...HEAD"],
        capture_output=True, text=True)
    if out.returncode != 0:
        out = subprocess.run(["git", "-C", str(ROOT), "diff", "--name-status", base],
                             capture_output=True, text=True, check=True)
    rows = []
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            rows.append((parts[0][0], parts[-1]))
    return rows


def existing_generations(base: str) -> set:
    out = subprocess.run(["git", "-C", str(ROOT), "ls-tree", "-d", "--name-only",
                          f"{base}:eval/cells"], capture_output=True, text=True)
    return set(out.stdout.split()) if out.returncode == 0 else set()


def classify(rows, base_generations) -> dict:
    blocked, cartography, contributor, other = [], [], [], []
    for status, path in rows:
        in_instrument = any(path.startswith(p) for p in INSTRUMENT)
        if not in_instrument:
            (contributor if any(path.startswith(p) for p in CONTRIBUTOR) else other).append(path)
            continue
        m = CELL_DIR.match(path)
        new_generation = bool(m and m.group(1) not in base_generations)
        if status == "A" and new_generation:
            cartography.append(path)
        elif status == "A":
            blocked.append((path, "adds a file to the instrument. A new scorer, config or "
                                 "schema beside the existing one is a modification wearing a "
                                 "hat -- it changes what the next submission is measured with."))
        elif status == "D":
            blocked.append((path, "deletes part of the instrument."))
        elif m and m.group(1) in base_generations:
            blocked.append((path, f"edits the frozen generation {m.group(1)}. A generation is "
                                  f"frozen for its lifetime and its receipts stay attached to "
                                  f"it; editing one silently re-scores history. If the meaning "
                                  f"of the evaluation should change, the answer is a NEW "
                                  f"generation."))
        else:
            blocked.append((path, "modifies the measuring instrument. This decides what is "
                                  "measured, so a submission that could change it could win by "
                                  "editing the ruler."))
    return {"blocked": blocked, "cartography": cartography,
            "contributor": contributor, "other": other}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--json")
    a = ap.parse_args()

    rows = changed(a.base)
    if not rows:
        print(f"ok: nothing changed against {a.base}")
        return 0
    r = classify(rows, existing_generations(a.base))

    if r["cartography"]:
        gens = sorted({CELL_DIR.match(p).group(1) for p in r["cartography"]})
        print(f">> CARTOGRAPHY: this submission opens {', '.join(gens)}")
        print(f"   {len(r['cartography'])} new file(s) under a generation that does not exist "
              f"on {a.base}.")
        print(f"   Adding a generation cannot change what any existing receipt meant, which is "
              f"why\n   it is allowed where editing one is not. It is scored as cartography "
              f"rather than\n   as a speedup: `burnish cartography check` is the gate it has "
              f"to pass.\n")

    if r["blocked"]:
        print(f"!! this submission changes the MEASURING INSTRUMENT:\n", file=sys.stderr)
        for path, why in r["blocked"]:
            print(f"   {path}\n       {why}\n", file=sys.stderr)
        print("   Improving the instrument is a real contribution -- the evaluator is where the "
              "bugs are,\n   and a broken one prints a confident number. It is not refused, it "
              "is separated: send it\n   as its own pull request, scored as a change to what is "
              "measured rather than riding in\n   on a change it would score.\n", file=sys.stderr)

    doc = {"base": a.base, "blocked": [{"path": p, "why": w} for p, w in r["blocked"]],
           "cartography": r["cartography"], "contributor": r["contributor"],
           "other": r["other"], "ok": not r["blocked"],
           "outcome": ("BLOCKED" if r["blocked"] else
                       "CARTOGRAPHY" if r["cartography"] else "CONTRIBUTOR")}
    if a.json:
        Path(a.json).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")

    if r["blocked"]:
        return 1
    print(f"ok: {len(r['contributor'])} contributor file(s), "
          f"{len(r['cartography'])} cartography file(s), instrument untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
