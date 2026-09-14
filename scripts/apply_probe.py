#!/usr/bin/env python3
"""Fold a `burnisher probe` result into configs/devices.json, marking the fields `measured`.

    burnisher probe > probe.txt          # on the pinned hardware
    scripts/apply_probe.py probe.txt --device rtx5090 --write

A script rather than an edit, for the rule that runs through this whole repository: never type a
benchmark number by hand. The probe's own JSON is the source, the config is generated from it,
and `eval/make_generation.py` regenerates every ceiling from the config.

**Every ceiling in the repository changes when this runs.** That is the point, and it is why the
previous values are kept alongside under `vendor_was`: a contributor who read the old table
deserves to be able to see what moved and in which direction.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probe", help="output of `burnisher probe` (the BURNISH_JSON line)")
    ap.add_argument("--device", default="rtx5090")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    text = Path(args.probe).read_text()
    if "BURNISH_JSON:" not in text:
        print("!! no BURNISH_JSON line in that file", file=sys.stderr)
        return 2
    probe = json.loads(text.split("BURNISH_JSON:", 1)[1])
    if probe.get("basis") != "measured":
        print(f"!! probe basis is {probe.get('basis')!r}, not 'measured'. Only a run on the "
              f"part may write these fields.", file=sys.stderr)
        return 2

    path = ROOT / "configs" / "devices.json"
    doc = json.loads(path.read_text())
    if args.device not in doc:
        print(f"!! no device {args.device!r} in {path}", file=sys.stderr)
        return 2
    dev = doc[args.device]
    m, d = probe["measured"], probe["device"]

    def put(key, value, note=None):
        old = dev.get(key)
        was = None
        if isinstance(old, dict):
            was = {"value": old.get("value"), "source": old.get("source")}
        entry = {"value": value, "source": "measured",
                 "probed_on": d.get("name"), "probe_note": probe.get("_note")}
        if note:
            entry["_note"] = note
        if was and was["source"] != "measured":
            entry["vendor_was"] = was
            entry["_direction"] = _direction(key, was["value"], value)
        dev[key] = entry
        return was, value

    rows = []
    rows.append(("memory_bandwidth_gbs", *put(
        "memory_bandwidth_gbs", m["memory_bandwidth_gbs"],
        "Sustained read+write over a working set far larger than L2.")))
    rows.append(("bf16_tensor_tflops", *put(
        "bf16_tensor_tflops", m["bf16_tensor_tflops"],
        "What a well-tuned square bf16 GEMM with fp32 accumulate achieves through cuBLAS -- "
        "not what the ALUs could issue. A roofline is only useful if a contributor could in "
        "principle reach it.")))
    rows.append(("sm_count", *put("sm_count", d["sm"])))
    rows.append(("l2_bytes", *put("l2_bytes", d["l2_bytes"])))
    rows.append(("vram_bytes", *put(
        "vram_bytes", d["vram_bytes"],
        "Usable device memory as the driver reports it, which is less than the marketed "
        "capacity.")))
    if d.get("clock_khz") is not None:           # not reported by CUDA 13 and later
        rows.append(("clock_boost_ghz", *put("clock_boost_ghz", d["clock_khz"] / 1e6)))
    rows.append(("persisting_l2_bytes", *put(
        "persisting_l2_bytes", d["persisting_l2_bytes"],
        "cudaDeviceProp::accessPolicyMaxWindowSize on this part. Not used by any ceiling here; "
        "recorded because it bounds any residency trick in the denoise loop.")))

    print(f"{'field':26s} {'was':>16s} {'measured':>16s}  direction")
    for key, was, value in rows:
        old = was["value"] if was else None
        print(f"{key:26s} {_fmt(old):>16s} {_fmt(value):>16s}  "
              f"{_direction(key, old, value) if old is not None else ''}")

    doc["_probe"] = {
        "device": args.device, "probed": d.get("name"),
        "_what_changed": ("Every ceiling in eval/cells/*/generation.json is computed from these "
                          "fields. Regenerate with `eval/make_generation.py --write` and "
                          "`eval/roofline_table.py --markdown docs/ROOFLINE.md` after applying "
                          "a probe, or the published table and the scorer disagree."),
    }
    if args.write:
        path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print(f"\n>> wrote {path}")
        print("   now: eval/make_generation.py --write && "
              "eval/roofline_table.py --markdown docs/ROOFLINE.md")
    else:
        print("\n(dry run; --write to apply)")
    return 0


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return f"{v:,}" if isinstance(v, int) and v > 9999 else str(v)


def _direction(key, old, new):
    """Which way an assumption was wrong, in the terms a contributor cares about.

    A ceiling is `max(flops/peak, bytes/bandwidth)` and `achieved = ceiling / measured`, so:
    understating a peak makes the ceiling too LARGE, achieved too LARGE, and the published room
    too SMALL -- conservative. Overstating it does the opposite, which is the dangerous one,
    because it tells a contributor there is room that is not there.
    """
    if old is None or not isinstance(old, (int, float)) or old == new:
        return ""
    if key not in ("memory_bandwidth_gbs", "bf16_tensor_tflops", "fp8_tensor_tflops",
                   "fp4_tensor_tflops", "fp32_tflops"):
        return "(not a ceiling term)"
    if new > old:
        return "assumed peak was LOW -> room had been understated (conservative)"
    return "assumed peak was HIGH -> room had been OVERSTATED (the dangerous direction)"


if __name__ == "__main__":
    sys.exit(main())
