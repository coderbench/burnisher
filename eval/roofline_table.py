#!/usr/bin/env python3
"""Publish the per-cell roofline table: how big each box is, and how full it is.

This is the table a contributor reads before deciding where to spend a week, and it is the one
artifact in this repository whose honesty matters most. Overselling a surface is the single
failure mode that kills a subnet: somebody burns a week of GPU time for a result inside the
noise floor and does not come back.

So it prints three things per cell and refuses to blur them together:

    ceiling          arithmetic, from the config and the device peak. Never a measurement.
    achieved         the fraction currently reached. NEEDS A RUN. Prints `--` when there is none.
    floor            the cell's measured run-to-run spread. NEEDS REPEATED RUNS. Same.

A cell with a ceiling and no achieved fraction is a cell where the size of the box is known and
how full it is is not. That is printed as `--`, not as a plausible number, and the summary line
says how many cells are in that state.

    eval/roofline_table.py                       # the pinned generation, as a table
    eval/roofline_table.py --markdown docs/ROOFLINE.md
    eval/roofline_table.py --device dgx_spark    # the same cells on a different part
    eval/roofline_table.py --json roofline.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from burnscore import cells as C
from burnscore.floor import floor_as_gap_closed, resolution_gate

ROOT = Path(__file__).resolve().parent.parent


def rows_for(generation, raw):
    out = []
    for spec in raw["cells"]:
        cell = generation.cells[spec["id"]]
        row = {
            "cell": cell.id, "stage": cell.stage, "dtype": cell.wdtype,
            "implemented": cell.implemented,
            "invocations": spec["shape"].get("invocations_per_generation"),
            "ceiling_s": spec["ceiling_seconds"],
            "ceiling_basis": spec["ceiling_basis"],
            "peak_basis": spec["peak_basis"],
            "bound_by": spec["bound_by"],
            "flops": spec["flops"],
            "unavoidable_bytes": spec["unavoidable_bytes"],
            "traffic_bytes": spec["traffic_bytes"],
            "param_bytes": spec["param_bytes"],
            "arithmetic_intensity": spec["arithmetic_intensity"],
            "ridge_point": spec["ridge_point"],
            "fusion_headroom": (spec["ceiling_decomposed_seconds"] / spec["ceiling_seconds"]
                                if spec["ceiling_seconds"] else None),
            "achieved": cell.achieved,
            "floor_pct": cell.floor_pct,
            "weight": cell.weight,
            "notes": cell.notes,
        }
        if cell.achieved is not None:
            row["measured_s"] = spec["ceiling_seconds"] / cell.achieved
            row["max_further_speedup"] = 1.0 / cell.achieved
            row["room_fraction"] = 1.0 - cell.achieved
            if cell.floor_pct is not None:
                row["floor_as_gap_closed"] = floor_as_gap_closed(cell.floor_pct, cell.achieved)
                row["resolvable"] = resolution_gate(cell.floor_pct, cell.achieved)["resolvable"]
        out.append(row)
    return out


def _fmt(v, spec, dash="--"):
    return dash if v is None else format(v, spec)


def render_text(generation, rows, device_name):
    w = ["", f"Burnisher roofline table -- {generation.name} on {device_name}", ""]
    w.append("  ceiling: ARITHMETIC (max of flops/peak and unavoidable-bytes/bandwidth).")
    w.append("           A lower bound on time. Not reachable. Never a measurement.")
    w.append("  achieved / floor: MEASURED. `--` means nobody has measured this cell yet.")
    w.append("")
    hdr = (f"  {'cell':26s} {'runs':>4s} {'ceiling':>10s} {'bound':>7s} {'ai':>8s} "
           f"{'fuse':>5s} {'achieved':>9s} {'left':>7s} {'floor':>7s} {'res':>4s}")
    w.append(hdr)
    w.append("  " + "-" * (len(hdr) - 2))
    for r in rows:
        res = r.get("resolvable")
        w.append(
            f"  {r['cell']:26s} {_fmt(r['invocations'], 'd', ' --'):>4s} "
            f"{r['ceiling_s'] * 1e3:>8.2f}ms {r['bound_by']:>7s} "
            f"{r['arithmetic_intensity']:>8.0f} "
            f"{_fmt(r['fusion_headroom'], '.2f'):>5s} "
            f"{_fmt(r.get('achieved'), '.1%'):>9s} "
            f"{_fmt(r.get('max_further_speedup'), '.2f') + ('x' if r.get('max_further_speedup') else ''):>7s} "
            f"{_fmt(r.get('floor_pct'), '.3f'):>7s} "
            f"{('yes' if res else 'NO') if res is not None else '--':>4s}")
        if not r["implemented"]:
            w.append(f"      ^ DECLARED, NOT IMPLEMENTED -- ceiling published, no reference exists")
    unmeasured = sum(1 for r in rows if r.get("achieved") is None)
    w.append("")
    w.append(f"  ai   = arithmetic intensity, flop per unavoidable byte. Ridge point for this "
             f"part is {rows[0]['ridge_point']:.0f};")
    w.append(f"         above it the cell is compute-bound, below it memory-bound.")
    w.append(f"  fuse = how much the CURRENT op decomposition costs over the ideal: the ratio "
             f"of the")
    w.append(f"         sum of per-op bounds to the whole-stage bound. That gap is what fusion "
             f"is worth.")
    w.append(f"  left = the most a perfect implementation could still gain in this cell, ever.")
    w.append(f"  res  = does this cell's remaining room clear 20x its own measured noise floor?")
    w.append("")
    if unmeasured:
        w.append(f"  {unmeasured} of {len(rows)} cells have NO measured achieved fraction and no "
                 f"measured floor.")
        w.append(f"  For those the size of the box is known and how full it is is not. Nothing "
                 f"here")
        w.append(f"  should be read as a claim that there is room in them -- only that there "
                 f"could be.")
        w.append(f"  Fill them in with: tools/burnish calibrate --generation {generation.name} "
                 f"--repeats 9 --write")
        w.append("")
    if rows and rows[0]["peak_basis"] != "measured":
        w.append(f"  PEAK BASIS: {rows[0]['peak_basis']}. These ceilings stand on a published "
                 f"device peak rather")
        w.append(f"  than one measured on the part, so the room they imply is wrong in an "
                 f"UNKNOWN direction:")
        w.append(f"  an overstated peak overstates the room, an understated one understates it, "
                 f"and on this")
        w.append(f"  hardware the probe found one of each. `burnisher probe` settles it.")
        w.append("")
    return "\n".join(w)


def render_markdown(generation, rows, device_name, raw):
    w = [f"# Roofline table -- {generation.name}", "",
         f"**Device:** {device_name}. **Model:** `{raw['model']['repo']}` at revision "
         f"`{(raw['model'].get('revision') or '?')[:12]}`, {raw['model']['resolution']}px, "
         f"{raw['model']['steps']} steps"
         f"{', CFG' if raw['model']['classifier_free_guidance'] else ''}.", "",
         "Generated by `eval/roofline_table.py`. Do not edit. Ceilings come from",
         "`eval/cells/" + generation.name + "/generation.json`, which is itself generated from",
         "`configs/`; achieved, left and floor come from `eval/cells/" + generation.name
         + "/reference.json`.",
         "A number typed into this file by hand would disagree with the scorer, and the",
         "disagreement would be invisible until a receipt was wrong.", "",
         "## What the columns mean", "",
         "| column | basis | meaning |",
         "|---|---|---|",
         "| ceiling | **arithmetic** | `max(flops / peak, unavoidable_bytes / bandwidth)`. A lower bound on time. Not reachable. |",
         "| bound | arithmetic | which of the two terms is larger. |",
         "| ai | arithmetic | flop per unavoidable byte. Above the part's ridge point the cell is compute-bound. |",
         "| fuse | arithmetic | the sum of per-op bounds over the whole-stage bound. What fusion is worth, before anybody writes a kernel. |",
         "| achieved | **measured** | `ceiling / measured`. Needs a run on the pinned hardware. |",
         "| left | measured | `1 / achieved` -- the most any implementation could still gain here, ever. |",
         "| floor | measured | this cell's run-to-run spread, from repeated paired control runs. |",
         "| res | measured | does the remaining room clear 20x this cell's own floor? |",
         "",
         "`unavoidable_bytes` is weights-read-once plus stage input plus stage output. Every",
         "intermediate is excluded on purpose: an intermediate is removable by fusion, and a",
         "ceiling that moved when a contributor fused would not be a ceiling.",
         "",
         "## Cells", "",
         "| cell | runs | ceiling | bound | ai | fuse | achieved | left | floor | res |",
         "|---|--:|--:|:--|--:|--:|--:|--:|--:|:--:|"]
    for r in rows:
        res = r.get("resolvable")
        w.append(
            f"| `{r['cell']}` | {_fmt(r['invocations'], 'd')} | "
            f"{r['ceiling_s'] * 1e3:.2f} ms | {r['bound_by']} | "
            f"{r['arithmetic_intensity']:.0f} | {_fmt(r['fusion_headroom'], '.2f')} | "
            f"{_fmt(r.get('achieved'), '.1%')} | "
            f"{_fmt(r.get('max_further_speedup'), '.2f')} | "
            f"{_fmt(r.get('floor_pct'), '.3f')} | "
            f"{('yes' if res else '**no**') if res is not None else '--'} |")
    w.append("")
    unmeasured = [r["cell"] for r in rows if r.get("achieved") is None]
    if unmeasured:
        w += ["## What is not known yet", "",
              f"**{len(unmeasured)} of {len(rows)} cells have no measured achieved fraction and "
              f"no measured floor.**", "",
              "For these, the size of the box is known and how full it is is not. Nothing in "
              "this table",
              "should be read as a claim that there is room in them -- only that there could be, "
              "and how",
              "much room there could be at most. A cell can be at 95% of its ceiling and look "
              "identical",
              "here to one at 8%.", "",
              "```", f"tools/burnish calibrate --generation {generation.name} --repeats 9 --write",
              "```", ""]
    unmeasured_peak = [r["cell"] for r in rows if r["peak_basis"] != "measured"]
    if unmeasured_peak:
        w += ["## The peaks these ceilings stand on", "",
              f"{len(rows) - len(unmeasured_peak)} of {len(rows)} cells stand on peaks MEASURED "
              f"on the pinned part by `burnisher probe`.",
              "", "These do not:", ""]
        w += [f"- `{c}`" for c in unmeasured_peak]
        w += ["",
              "For those, the room implied by the ceiling is wrong in an **unknown direction**. "
              "An overstated",
              "peak overstates the room; an understated one understates it. On this hardware the "
              "probe found",
              "one of each — bandwidth was assumed 19% too high, and the bf16 GEMM peak 14% too "
              "low — so",
              "there is no safe default to assume. Measure it.", ""]
    else:
        w += ["## The peaks these ceilings stand on", "",
              "All measured on the pinned part by `burnisher probe`: sustained bandwidth from a "
              "grid-stride",
              "read+write over a working set far larger than L2, and the GEMM rate from a large "
              "square bf16",
              "matmul with fp32 accumulate through cuBLAS — what a well-tuned kernel achieves "
              "rather than",
              "what the ALUs could issue, because a roofline is only useful if a contributor "
              "could in",
              "principle reach it.", ""]
    w += ["## Notes per cell", ""]
    for r in rows:
        if r["notes"]:
            w.append(f"- **`{r['cell']}`** -- {r['notes']}")
    w.append("")
    return "\n".join(w)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generation", default="BG-1")
    ap.add_argument("--device", help="recompute the ceilings for a different part")
    ap.add_argument("--json")
    ap.add_argument("--markdown")
    args = ap.parse_args()

    gpath = ROOT / "eval" / "cells" / args.generation / "generation.json"
    if not gpath.exists():
        print(f"!! no such generation: {gpath}", file=sys.stderr)
        return 2
    generation = C.load(gpath)
    raw = json.loads(gpath.read_text())
    devices = json.loads((ROOT / "configs" / "devices.json").read_text())
    device_key = args.device or raw["device"]

    if args.device and args.device != raw["device"]:
        # A different part is a different generation's worth of ceilings. Recompute them rather
        # than reusing the frozen ones, and say so: these numbers are NOT the scored ceilings.
        from make_generation import build
        raw = build(generation.name, raw["model"]["key"], args.device,
                    resolution=raw["model"]["resolution"], steps=raw["model"]["steps"],
                    caption_len=raw["model"]["caption_len"],
                    cfg=raw["model"]["classifier_free_guidance"])
        for spec in raw["cells"]:
            if spec["id"] in generation.cells:
                generation.cells[spec["id"]].ceiling_seconds = spec["ceiling_seconds"]

    rows = rows_for(generation, raw)
    name = devices[device_key]["name"]
    print(render_text(generation, rows, name))
    if args.device and args.device != json.loads(gpath.read_text())["device"]:
        print(f"  NOTE: recomputed for {device_key}. The SCORED ceilings are the ones frozen "
              f"in\n        {gpath.relative_to(ROOT)}, which are for "
              f"{json.loads(gpath.read_text())['device']}.\n")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"generation": generation.name, "device": device_key, "rows": rows,
             "_basis": "ceiling columns are arithmetic; achieved and floor are measured or null"},
            indent=1, sort_keys=True) + "\n")
        print(f">> wrote {args.json}")
    if args.markdown:
        Path(args.markdown).write_text(render_markdown(generation, rows, name, raw))
        print(f">> wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
