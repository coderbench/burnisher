#!/usr/bin/env python3
"""Text in, PNG out: the runtime used the way a person uses an image generator.

    scripts/generate.py --weights /workspace/ckpt "a red fox asleep in fresh snow" --out fox.png
    scripts/generate.py --weights DIR "a lighthouse at dusk" --negative "blurry" --seed 7

The runtime takes token ids rather than text, so a tokenizer is not vendored into C++ as a second
oracle (`scripts/tokenize_prompts.py` says why). This wrapper does the three steps around it:
tokenize with the pinned SentencePiece model, run `burnisher generate`, and write the pixels as a
PNG. Guidance scale, steps and resolution default to the frozen generation's, so an image made here
is made the way the benchmark makes one.

Needs `sentencepiece` and `numpy`, and the tokenizer, which `eval/provision_box.sh` downloads beside
the checkpoint as `tokenizer/spiece.model`.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from tokenize_prompts import encode  # noqa: E402


def find_binary():
    """The same order `tools/burnish` uses: an explicit binary, then the CUDA build, then CPU."""
    for candidate in (os.environ.get("BURNISHER_BIN"), ROOT / "build-cuda" / "burnisher",
                      ROOT / "build" / "burnisher"):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def to_rgb8(pixels):
    """[1, 3, H, W] in the decoder's [-1, 1] range -> H x W x 3 bytes."""
    import numpy as np
    if pixels.ndim != 4 or pixels.shape[0] != 1 or pixels.shape[1] != 3:
        raise ValueError(f"expected decoded pixels [1, 3, H, W], got {list(pixels.shape)}")
    x = np.clip(pixels[0].astype(np.float32) / 2.0 + 0.5, 0.0, 1.0)
    return np.ascontiguousarray((x * 255.0).round().astype(np.uint8).transpose(1, 2, 0))


def png_bytes(rgb):
    """An 8-bit RGB PNG, with nothing but zlib: no imaging library to install for one image."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("--out", default="burnisher.png")
    ap.add_argument("--weights", required=True, help="checkpoint directory")
    ap.add_argument("--negative", default="", help="negative prompt (default: empty)")
    ap.add_argument("--spiece", help="SentencePiece model (default: WEIGHTS/tokenizer/spiece.model)")
    ap.add_argument("--generation", default="BG-1", help="whose defaults to use")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--resolution", type=int)
    ap.add_argument("--guidance-scale", type=float)
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--impl", default=None,
                    help="registered implementation (default: the device's own, e.g. cuda)")
    ap.add_argument("--device", choices=["cpu", "cuda"],
                    help="default: cuda with the CUDA build, cpu otherwise")
    args = ap.parse_args()

    try:
        import numpy as np
        import sentencepiece as spm
    except ImportError as e:
        print(f"!! {e.name} is not installed: pip install numpy sentencepiece", file=sys.stderr)
        return 2

    binary = find_binary()
    if binary is None:
        print("!! the runtime is not built: scripts/build_cuda.sh (or scripts/build.sh for CPU)",
              file=sys.stderr)
        return 2
    device = args.device or ("cuda" if binary.parent.name == "build-cuda" else "cpu")
    spiece = Path(args.spiece) if args.spiece else Path(args.weights) / "tokenizer" / "spiece.model"
    if not spiece.is_file():
        print(f"!! no tokenizer at {spiece}. It is `tokenizer/spiece.model` in the text encoder's "
              f"repository; pass --spiece", file=sys.stderr)
        return 2

    model = json.loads((ROOT / "eval" / "cells" / args.generation / "generation.json")
                       .read_text())["model"]
    sp = spm.SentencePieceProcessor(model_file=str(spiece))
    negative, _ = encode(sp, args.negative, model["caption_len"])
    positive, _ = encode(sp, args.prompt, model["caption_len"])

    with tempfile.TemporaryDirectory() as tmp:
        ids = Path(tmp) / "ids.txt"
        ids.write_text(" ".join(map(str, negative)) + "\n" + " ".join(map(str, positive)) + "\n")
        pixels = Path(tmp) / "pixels.npy"
        cmd = [str(binary), "generate", "--weights", args.weights, "--token-ids", str(ids),
               "--device", device, "--impl", args.impl or device,
               "--steps", str(args.steps or model["steps"]),
               "--resolution", str(args.resolution or model["resolution"]),
               "--guidance-scale", str(args.guidance_scale or model["guidance_scale"]),
               "--seed", str(args.seed), "--dump-pixels", str(pixels)]
        run = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        if run.returncode != 0:
            print(run.stdout, end="")
            return run.returncode
        rgb = to_rgb8(np.load(pixels))

    Path(args.out).write_bytes(png_bytes(rgb))
    report = next((json.loads(line.split("BURNISH_JSON:", 1)[1])
                   for line in run.stdout.splitlines() if line.startswith("BURNISH_JSON:")), None)
    if report:
        m = report["metrics"]
        print(f">> {args.out}  {rgb.shape[1]}x{rgb.shape[0]}  {m['latency_s']:.1f} s  "
              f"(text {m['text_encode_s']:.2f} s, denoise {m['denoise_s']:.1f} s, "
              f"decode {m['vae_decode_s']:.2f} s)")
    else:
        print(f">> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
