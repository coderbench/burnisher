#!/usr/bin/env python3
"""Check what the runtime requires of a checkpoint against what the checkpoint actually holds.

    scripts/verify_checkpoint_layout.py            # fetch headers over HTTP and compare
    scripts/verify_checkpoint_layout.py --local /path/to/checkpoint
    scripts/verify_checkpoint_layout.py --save configs/checkpoint-layout.json

**Why this exists.** `declare_pixart_shapes()` names 962 tensors, and those names were written
from the reference implementation's module structure. That is usually right and it is not
evidence. A model that runs with a misread name does not crash -- it produces a plausible image
from the wrong weights, or it dies at load time against 22 GB of download, after the download.

**Why it is cheap.** A safetensors file begins with an 8-byte little-endian header length and
then that many bytes of JSON naming every tensor, its dtype and its shape. Two HTTP range
requests per shard fetch it: about 68 kB for the DiT against a 2.4 GB file, and about 1.7 MB for
the T5 encoder against 19 GB. Nothing else is downloaded.

This turns the largest unverified assumption in the repository into a check that runs in seconds.
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF = "https://huggingface.co"

# Which files hold which component, from configs/candidates.json's pinned revisions.
SHARDS = {
    "transformer": [("repo", "transformer/diffusion_pytorch_model.safetensors")],
    "vae": [("repo", "vae/diffusion_pytorch_model.safetensors")],
    "text_encoder": [("text_encoder_repo", "text_encoder/model-00001-of-00002.safetensors"),
                     ("text_encoder_repo", "text_encoder/model-00002-of-00002.safetensors")],
}


def candidate():
    doc = json.loads((ROOT / "configs" / "candidates.json").read_text())
    return doc["candidates"]["pixart-sigma-xl2-1024"]


def _get_range(url, start, length, timeout=60):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + length - 1}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def remote_header(repo, revision, path):
    """The safetensors header, by two range requests. Nothing else is transferred."""
    url = f"{HF}/{repo}/resolve/{revision or 'main'}/{path}"
    n = struct.unpack("<Q", _get_range(url, 0, 8))[0]
    if n == 0 or n > 64 << 20:
        raise RuntimeError(f"{path}: implausible header length {n}")
    return json.loads(_get_range(url, 8, n).decode()), url


def local_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode()), str(path)


def required():
    """What the runtime says it needs, from the runtime itself rather than from a copy of it."""
    binary = ROOT / "build" / "burnisher"
    if not binary.exists():
        raise SystemExit(f"!! {binary} is not built. scripts/build.sh first -- this check asks "
                         f"the RUNTIME what it requires rather than keeping a second list.")
    out = subprocess.run([str(binary), "weights-manifest"], capture_output=True, text=True,
                         check=True)
    return json.loads(out.stdout)["tensors"]


def compare(want, have):
    """Three kinds of disagreement, and they mean different things.

    MISSING   the runtime asks for a tensor the checkpoint does not have. Fatal: the model would
              fail at load, or worse, a caller would paper over it.
    SHAPE     the name matches and the geometry does not. Fatal and much more dangerous, because
              a transposed or mis-sized weight loads fine and produces a plausible image.
    EXTRA     the checkpoint holds a tensor the runtime never reads. Usually benign -- the
              reference computes things this pipeline does not need -- but it is reported,
              because an EXTRA that looks like a MISSING is a renamed tensor.
    """
    missing, wrong, extra = [], [], []
    for name, spec in sorted(want.items()):
        if name not in have:
            missing.append({"name": name, "component": spec["component"],
                            "wanted_shape": spec["shape"]})
            continue
        got = list(have[name].get("shape", []))
        if got != list(spec["shape"]):
            wrong.append({"name": name, "wanted_shape": spec["shape"], "checkpoint_shape": got})
    for name in sorted(have):
        if name != "__metadata__" and name not in want:
            extra.append(name)
    return missing, wrong, extra


def suggest(missing, have):
    """For each missing name, the closest checkpoint key. A rename is the commonest cause and
    'not found' among two thousand keys is useless without a candidate."""
    keys = [k for k in have if k != "__metadata__"]
    out = {}
    for m in missing[:40]:
        tail = m["name"].split(".")[-2:]
        near = [k for k in keys if all(t in k for t in tail)]
        if not near:
            near = [k for k in keys if tail[-1] in k]
        out[m["name"]] = near[:3]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", help="a local checkpoint directory instead of HuggingFace")
    ap.add_argument("--against-saved", action="store_true",
                    help="check the runtime against the pinned layout record in configs/ "
                         "instead of fetching. No network; this is what CI runs.")
    ap.add_argument("--save", help="write the checkpoint's real layout here")
    ap.add_argument("--components", nargs="*",
                    default=["transformer", "vae", "text_encoder"])
    args = ap.parse_args()

    cand = candidate()
    want = required()
    print(f"runtime requires {len(want)} tensors "
          f"({', '.join(sorted(set(v['component'] for v in want.values())))})\n")

    saved_path = ROOT / "configs" / "checkpoint-layout.json"
    if args.against_saved:
        # Offline. The record was produced by a live run against the pinned revisions and is
        # committed, so CI gets this check without depending on a network or on HuggingFace
        # being up. `--save` refreshes it, and a refresh that CHANGES anything means the
        # checkpoint moved under a pinned revision -- which is a changed oracle.
        if not saved_path.exists():
            print(f"!! {saved_path} does not exist. Run this without --against-saved and with "
                  f"--save first.", file=sys.stderr)
            return 2
        record = json.loads(saved_path.read_text())
        have = {k: v for k, v in record["tensors"].items()}
        wanted_here = {k: v for k, v in want.items() if v["component"] in args.components}
        missing, wrong, extra = compare(wanted_here, have)
        print(f"  checked against the pinned record in configs/checkpoint-layout.json")
        print(f"  ({record['repo']} @ {(record.get('revision') or '?')[:12]})\n")
        print(f"  required by the runtime : {len(wanted_here)}")
        print(f"  MISSING (fatal)          : {len(missing)}")
        print(f"  WRONG SHAPE (fatal)      : {len(wrong)}")
        for w in wrong[:10]:
            print(f"     {w['name']}: runtime wants {w['wanted_shape']}, "
                  f"record has {w['checkpoint_shape']}")
        for m in missing[:10]:
            print(f"     MISSING {m['name']}")
        ok = not missing and not wrong
        print("\n" + ("ok: the runtime matches the pinned checkpoint layout" if ok else
                      "FAIL: the runtime cannot load the pinned checkpoint as written"))
        return 0 if ok else 1

    have = {}
    sources = {}
    for component in args.components:
        for repo_key, rel in SHARDS[component]:
            try:
                if args.local:
                    header, src = local_header(Path(args.local) / rel)
                else:
                    repo = cand[repo_key] if repo_key != "repo" else cand["repo"]
                    rev = (cand.get("revision") if repo_key == "repo"
                           else cand.get("text_encoder_revision"))
                    header, src = remote_header(repo, rev, rel)
            except (urllib.error.URLError, OSError, RuntimeError) as exc:
                print(f"!! {component}: could not read {rel}: {exc}", file=sys.stderr)
                print(f"   The check could not run. That is not the same as passing.",
                      file=sys.stderr)
                return 2
            for k, v in header.items():
                if k != "__metadata__":
                    have[k] = v
            sources[rel] = src
            print(f"  {component:14s} {rel.split('/')[-1]:44s} "
                  f"{len(header) - ('__metadata__' in header)} tensors")

    wanted_here = {k: v for k, v in want.items() if v["component"] in args.components}
    missing, wrong, extra = compare(wanted_here, have)

    print(f"\n  required by the runtime : {len(wanted_here)}")
    print(f"  present in the checkpoint: {len(have)}")
    print(f"  MISSING (fatal)          : {len(missing)}")
    print(f"  WRONG SHAPE (fatal)      : {len(wrong)}")
    print(f"  extra, unread            : {len(extra)}")

    if wrong:
        print("\n!! SHAPE MISMATCHES. These are the dangerous ones: a mis-sized or transposed")
        print("   weight loads without complaint and produces a plausible image from wrong")
        print("   numbers.")
        for w in wrong[:20]:
            print(f"     {w['name']}: runtime wants {w['wanted_shape']}, "
                  f"checkpoint has {w['checkpoint_shape']}")
    if missing:
        print("\n!! MISSING TENSORS. The runtime asks for these and the checkpoint has no such")
        print("   key. The commonest cause is a rename; closest matches shown.")
        hints = suggest(missing, have)
        for m in missing[:20]:
            near = hints.get(m["name"]) or ["(nothing resembling it)"]
            print(f"     {m['name']}")
            print(f"       closest: {', '.join(near)}")
        if len(missing) > 20:
            print(f"     ... and {len(missing) - 20} more")
    if extra:
        print(f"\n  {len(extra)} tensor(s) in the checkpoint the runtime never reads, e.g.:")
        for e in extra[:8]:
            print(f"     {e}")
        print("  Usually benign -- the reference computes things this pipeline does not need.")

    if args.save:
        Path(args.save).write_text(json.dumps({
            "_what": "The pinned checkpoint's real safetensors layout, read by HTTP range "
                     "request from the pinned revisions. Compared against what the runtime "
                     "requires by scripts/verify_checkpoint_layout.py.",
            # The outcome, recorded here so nothing downstream has to re-derive it or, worse,
            # type it.
            "verified": {
                "required_by_runtime": len(wanted_here),
                "missing": len(missing),
                "wrong_shape": len(wrong),
                "extra_unread": len(extra),
                "components": sorted(args.components),
                "_note": "0 missing and 0 wrong_shape means the runtime can name every tensor "
                         "it needs in this checkpoint, at these revisions.",
            },
            "repo": cand["repo"], "revision": cand.get("revision"),
            "text_encoder_repo": cand.get("text_encoder_repo"),
            "text_encoder_revision": cand.get("text_encoder_revision"),
            "sources": sources,
            "tensors": {k: {"dtype": v.get("dtype"), "shape": v.get("shape")}
                        for k, v in sorted(have.items())},
        }, indent=1, sort_keys=True) + "\n")
        print(f"\n>> wrote {args.save}")

    ok = not missing and not wrong
    print("\n" + ("ok: the runtime's tensor names and shapes match the pinned checkpoint"
                  if ok else
                  "FAIL: the runtime cannot load this checkpoint as written"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
