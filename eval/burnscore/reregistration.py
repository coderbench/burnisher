"""Does a new kernel registration register a kernel that is already on main?

The runtime has one rule for contributors: register a new name beside the old one, never replace
a file. Submissions are then measured against the `cuda` baseline, which never runs any other
registered kernel. So a submission that registers a kernel already on main -- a merged
contributor's kernel, or a baseline one -- under a new name is measured as a gain it did not make,
and a merged kernel would be paid a second time for a gain that already landed.

A copy of an open pull request is the copycat guard's question. This one asks only about main,
and answers it from the registry itself:

* **The same callable under a new name** -- `attention_cuda_tiled<256>` registered again as
  `fast` -- is a re-registration, whatever the name says.
* **The same kernel with different constants is a variant, not a re-registration.**
  `cuda-tile64` and `cuda-tile1024` are one function at different tile widths, and that
  difference is exactly what they exist to measure. So constants are KEPT when kernels are
  compared, and a new template argument to an existing function is never flagged.
* **A renamed, reformatted copy of a kernel** -- its wrapper and the device kernels it launches,
  with every identifier changed -- is compared as structure: identifiers and comments normalized,
  numbers kept. At or above SIMILAR it is a re-registration. A copy of the baseline that was
  actually changed, which is the recommended way to start a new kernel, falls below it.

Shared infrastructure -- `require_device`, `check_launch`, anything most registered kernels call --
is left out of the comparison, so two unrelated kernels do not look alike for calling it.
"""
from __future__ import annotations

import difflib
import re
from collections import Counter

from .copycat import KEYWORDS, _TOKEN, is_code, strip_comments

SIMILAR = 0.95          # token-sequence similarity at which a new kernel IS an existing one
MIN_TOKENS = 40         # a smaller kernel than this is not compared: too little to tell apart
INFRA_USERS = 3         # a helper that this many registered kernels call is infrastructure

_REG = re.compile(r'register_impl\s*<\s*(\w+)\s*>\s*\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*'
                  r'([A-Za-z_]\w*\s*(?:<[^<>()]*>)?)\s*,', re.S)
_DEF = re.compile(r'\b([A-Za-z_]\w*)\s*\(')
_IDENT = re.compile(r'\b([A-Za-z_]\w*)\b')
_NOT_FUNCTIONS = frozenset({"if", "for", "while", "switch", "return", "sizeof", "catch",
                            "DISPATCH", "defined", "alignof", "decltype", "static_cast",
                            "reinterpret_cast", "const_cast", "dynamic_cast"})


def _clean(text, path):
    return "\n".join(strip_comments(text.splitlines(), path))


def _match(src, i, opener, closer):
    depth, quote, k = 0, None, i
    while k < len(src):
        ch = src[k]
        if quote:
            if ch == "\\":
                k += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return k
        k += 1
    return -1


def registrations(files: dict) -> list:
    """Every `register_impl` call: op, name, the registered callable, its function and arguments."""
    out = []
    for path, text in sorted(files.items()):
        if not is_code(path) or path.endswith(".py"):
            continue
        for m in _REG.finditer(_clean(text, path)):
            _args_type, op, name, callable_ = m.groups()
            callable_ = "".join(callable_.split())
            base = re.match(r"[A-Za-z_]\w*", callable_).group(0)
            out.append({"op": op, "name": name, "callable": callable_, "function": base,
                        "template_args": callable_[len(base):], "path": path})
    return out


def definitions(files: dict) -> dict:
    """name -> [{"path", "body", "text"}] for every function defined with a body."""
    defs = {}
    for path, text in sorted(files.items()):
        if not is_code(path) or path.endswith(".py"):
            continue
        src = _clean(text, path)
        for m in _DEF.finditer(src):
            name = m.group(1)
            if name in _NOT_FUNCTIONS or name in KEYWORDS:
                continue
            close = _match(src, m.end() - 1, "(", ")")
            if close < 0:
                continue
            j = close + 1
            while True:
                while j < len(src) and src[j].isspace():
                    j += 1
                q = re.match(r"(const|noexcept|override|final)\b", src[j:])
                if not q:
                    break
                j += q.end()
            if j >= len(src) or src[j] != "{":
                continue
            end = _match(src, j, "{", "}")
            if end < 0:
                continue
            start = src.rfind("\n", 0, m.start()) + 1
            prev = src.rfind("\n", 0, max(0, start - 1)) + 1
            if src[prev:start].strip().startswith(("template", "__global__", "__device__")):
                start = prev
            defs.setdefault(name, []).append({"path": path, "body": src[j + 1:end],
                                              "text": src[start:end + 1]})
    return defs


def _tokens(text):
    out = []
    for m in _TOKEN.finditer(text):
        string, number, ident, op = m.groups()
        if string is not None:
            out.append("S")
        elif number is not None:
            out.append(number)                       # constants are kept: they are the variant
        elif ident is not None:
            out.append(ident if ident in KEYWORDS or ident.startswith("__") else "I")
        else:
            out.append(op)
    return out


def infrastructure(regs: list, defs: dict) -> set:
    users = Counter()
    for function in {r["function"] for r in regs}:
        body = " ".join(e["body"] for e in defs.get(function, []))
        for callee in set(_IDENT.findall(body)):
            if callee in defs and callee != function:
                users[callee] += 1
    return {c for c, n in users.items() if n >= INFRA_USERS}


def kernel_tokens(function: str, defs: dict, infra: set) -> list:
    """The registered function's body, then the body of each kernel or helper it calls."""
    entries = defs.get(function) or []
    if not entries:
        return []
    body = " ".join(e["body"] for e in entries)
    toks, seen = _tokens(body), {function}
    for callee in _IDENT.findall(body):
        if callee in defs and callee not in seen and callee not in infra:
            seen.add(callee)
            toks += _tokens(" ".join(e["body"] for e in defs[callee]))
    return toks


def judge(candidate_files: dict, base_files: dict) -> dict:
    """REREGISTERED or CLEAR, with one finding per new registration that is an existing kernel."""
    base_regs, cand_regs = registrations(base_files), registrations(candidate_files)
    existing = {(r["op"], r["name"]) for r in base_regs}
    new = [r for r in cand_regs if (r["op"], r["name"]) not in existing]
    result = {"outcome": "CLEAR", "findings": [],
              "new_registrations": [f"{r['op']}/{r['name']}" for r in new],
              # The names a candidate arm could run. The validator measures ONE name against
              # `cuda`, and a submission that registers a new kernel says which by registering it.
              "candidate_names": sorted({r["name"] for r in new})}
    if not new:
        return result
    base_defs, cand_defs = definitions(base_files), definitions(candidate_files)
    infra = infrastructure(base_regs, base_defs)
    signatures = {}
    for b in base_regs:
        signatures.setdefault(b["callable"], (b, kernel_tokens(b["function"], base_defs, infra)))
    base_functions = {b["function"] for b in base_regs}

    def describe(r):
        return {"op": r["op"], "name": r["name"], "callable": r["callable"], "path": r["path"]}

    for r in new:
        same = [b for b in base_regs if b["callable"] == r["callable"]]
        if same:
            result["findings"].append({"registration": describe(r), "kind": "same-callable",
                                       "matches": describe(same[0]), "similarity": 1.0})
            continue
        if r["function"] in base_functions:
            continue                                   # same function, new arguments: a variant
        toks = kernel_tokens(r["function"], cand_defs, infra)
        if len(toks) < MIN_TOKENS:
            continue
        best = None
        for b, btoks in signatures.values():
            if not btoks or (r["template_args"] and b["template_args"]
                             and r["template_args"] != b["template_args"]):
                continue
            ratio = difflib.SequenceMatcher(None, toks, btoks, autojunk=False).ratio()
            if best is None or ratio > best[0]:
                best = (ratio, b)
        if best and best[0] >= SIMILAR:
            result["findings"].append({"registration": describe(r), "kind": "same-kernel",
                                       "matches": describe(best[1]),
                                       "similarity": round(best[0], 4)})
    if result["findings"]:
        result["outcome"] = "REREGISTERED"
    return result
