#!/usr/bin/env python3
"""Produce T5 token ids for a frozen prompt set, with the pinned tokenizer.

    scripts/tokenize_prompts.py --spiece /path/to/spiece.model --write

**Why this is a script and not a committed table.** The ids ARE part of the oracle: the gate
compares latents produced from them, so a different tokenization is a different comparison. They
have to be reproducible by a stranger from the pinned tokenizer, and typing them once by hand
would make them unauditable forever.

**Why the runtime takes ids rather than text.** The T5 tokenizer is a SentencePiece model.
Vendoring one into a C++ runtime would put a second oracle in the repository, and the two would
drift. `burnisher generate --token-ids FILE` takes the ids directly.

**Two choices here are oracle decisions and are pinned rather than defaulted:**

* `clean_caption` is OFF. The reference pipeline defaults it ON, but it degrades to OFF when ftfy
  and BeautifulSoup are absent — so the reference's behaviour depends on what happens to be
  installed. A benchmark cannot have that. The frozen prompt set is written already-clean so the
  two agree by construction, and this states which branch is pinned.
* `max_length` is 300, PixArt-Sigma's own value, with truncation to 299 tokens plus the EOS the
  reference appends, then right-padding with the pad id. A ragged batch would give the two
  classifier-free-guidance branches different caption lengths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MAX_LENGTH = 300
PAD_ID = 0
EOS_ID = 1


def encode(sp, text, max_length=MAX_LENGTH):
    """What `T5Tokenizer(padding="max_length", truncation=True, add_special_tokens=True)` does.

    Truncate to `max_length - 1` and append EOS -- HuggingFace reserves room for the special
    token before truncating, so a long caption keeps its EOS. Then right-pad. Getting the order
    wrong drops the EOS on exactly the captions that need truncating, which is a silent
    difference on the longest prompts only.
    """
    ids = sp.encode(text)[: max_length - 1] + [EOS_ID]
    mask = [1] * len(ids)
    ids += [PAD_ID] * (max_length - len(ids))
    mask += [0] * (max_length - len(mask))
    return ids, mask


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spiece", required=True, help="the pinned spiece.model")
    ap.add_argument("--generation", default="BG-1")
    ap.add_argument("--negative", default="", help="the negative prompt (empty, as the "
                                                   "reference pipeline uses)")
    ap.add_argument("--max-length", type=int, default=MAX_LENGTH)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    try:
        import sentencepiece as spm
    except ImportError:
        print("!! sentencepiece is not installed.\n"
              "   This script deliberately uses the REFERENCE tokenizer rather than a "
              "reimplementation:\n   a hand-rolled tokenizer would be a second oracle, and the "
              "two would drift.\n   pip install sentencepiece", file=sys.stderr)
        return 2

    sp = spm.SentencePieceProcessor(model_file=args.spiece)
    prompts_path = ROOT / "eval" / "cells" / args.generation / "prompts.json"
    prompts = json.loads(prompts_path.read_text())

    out = {
        "_what": "T5 token ids for the frozen prompt set, produced by the pinned SentencePiece "
                 "model. Part of the oracle: the gate compares latents generated from these.",
        "generation": args.generation,
        "prompt_set_digest": hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(Path(args.spiece).read_bytes()).hexdigest(),
        "tokenizer_vocab_size": sp.get_piece_size(),
        "max_length": args.max_length,
        "pad_id": PAD_ID, "eos_id": EOS_ID,
        "clean_caption": False,
        "_clean_caption_note": (
            "OFF, pinned. The reference pipeline defaults it ON and silently falls back to OFF "
            "when ftfy and BeautifulSoup are absent, so its behaviour depends on the "
            "environment. A benchmark cannot. The prompt set is written already-clean so both "
            "branches agree on these four prompts."),
        "negative_prompt": args.negative,
        "prompts": {},
    }
    neg_ids, neg_mask = encode(sp, args.negative, args.max_length)
    out["negative"] = {"ids": neg_ids, "real_tokens": sum(neg_mask)}

    print(f"tokenizer: {Path(args.spiece).name}, vocab {sp.get_piece_size()}")
    print(f"max_length {args.max_length}, pad {PAD_ID}, eos {EOS_ID}, clean_caption off\n")
    print(f"  {'prompt':16s} {'real tokens':>12s}  {'truncated':>10s}")
    for p in prompts["prompts"]:
        ids, mask = encode(sp, p["text"], args.max_length)
        raw = len(sp.encode(p["text"]))
        out["prompts"][p["id"]] = {"ids": ids, "real_tokens": sum(mask),
                                   "tokens_before_padding": raw,
                                   "truncated": raw > args.max_length - 1}
        print(f"  {p['id']:16s} {sum(mask):>12d}  "
              f"{'yes' if raw > args.max_length - 1 else 'no':>10s}")
    print(f"  {'(negative)':16s} {sum(neg_mask):>12d}")

    dest = ROOT / "eval" / "cells" / args.generation / "token-ids.json"
    text = json.dumps(out, indent=1, sort_keys=True) + "\n"
    if args.write:
        dest.write_text(text)
        print(f"\n>> wrote {dest}")
        print(f"   prompt set digest {out['prompt_set_digest'][:16]}")
        print(f"   tokenizer digest  {out['tokenizer_sha256'][:16]}")
        # The runtime takes a plain text file: one prompt per line, negative first.
        for p in prompts["prompts"]:
            f = dest.parent / f"token-ids-{p['id']}.txt"
            f.write_text(" ".join(map(str, neg_ids)) + "\n" +
                         " ".join(map(str, out["prompts"][p["id"]]["ids"])) + "\n")
        print(f"   and {len(prompts['prompts'])} per-prompt files for "
              f"`burnisher generate --token-ids`")
    else:
        print(f"\n(dry run; --write to save to {dest})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
