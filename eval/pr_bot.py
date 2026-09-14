#!/usr/bin/env python3
"""Evaluate open pull requests and publish what was measured.

    eval/pr_bot.py --repo owner/name --once
    eval/pr_bot.py --repo owner/name --pr 42 --dry-run

What this bot does NOT do
-------------------------

It does not decide anything. Every outcome it publishes is a derivation somebody else can
repeat: the receipt comes from `burnish score` applied to measurements it commits alongside, and
the label comes from `burnish verdict` applied to the receipt. If this bot vanished, its verdicts
would still be checkable, and a replacement would produce the same ones.

That is the whole reason the loop is shaped this way. A bot that measured and then announced a
grade would be the only thing that knew how the grade was reached, and everybody without a GPU
would be asked to trust it. Here the expensive part -- the measurement -- is published as data,
and the cheap part -- turning data into a verdict -- is a function anyone can run in two seconds
with `burnish audit`.

The order of operations, and why
--------------------------------

  1. guard        Does the submission change the measuring instrument? A PR that does is SKIPPED,
                  not closed, and never spends GPU time -- a number produced by a modified
                  instrument cannot be accepted either way, so measuring it would be waste. The
                  exception is opening a NEW generation, which is cartography and is paid.
  2. build        From source, on the eval box. No prebuilt artifacts.
  3. gate         Correctness and self-determinism, before anything is timed, for BOTH arms.
  4. bench        Paired, interleaved, with a held-out shape drawn now -- after the candidate is
                  frozen.
  5. score        Into an append-only ledger outside the submission's reach.
  6. publish      Raw measurements AND receipt, so the verdict can be re-derived by anyone.
  7. label        A pure function of the receipt.

Steps 2-5 are `eval/score_submission.sh`, which is the same command a contributor runs by hand.
The bot is not a privileged path; it is a scheduled one.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from burnscore import verdict as V

ROOT = Path(__file__).resolve().parent.parent

# A label the bot applies to say "seen, and deliberately not measured". Distinct from every
# outcome label because it is not an outcome -- nothing was measured.
SKIPPED = f"{V.PREFIX}:skipped-instrument"


def gh(args, *, check=True, timeout=120):
    r = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed: {r.stderr.strip()[:400]}")
    return r


def open_prs(repo):
    r = gh(["pr", "list", "-R", repo, "--state", "open", "--limit", "50",
            "--json", "number,headRefOid,headRefName,labels,title,author"])
    return json.loads(r.stdout)


def already_labelled(pr) -> bool:
    """Has this exact commit already been given an outcome?

    Keyed on the label set rather than a local file, so the answer survives the bot being
    restarted, moved, or replaced -- the PR itself is the state.
    """
    names = {l["name"] for l in pr.get("labels", [])}
    return any(n.startswith(f"{V.PREFIX}:") for n in names)


def ensure_label(repo, label, color, description, *, dry_run=False):
    """Create the label with the colour we chose, before attaching it.

    Attaching a label that does not exist creates it -- with a colour GitHub picks at RANDOM.
    The fixed outcomes are pre-registered by eval/setup_labels.sh and so keep their meaning, but
    the PAYING label carries the measured number and therefore cannot be pre-registered: there is
    a different one for every value. Left alone, the most important outcome in the system came
    out a different shade every time, occasionally red.
    """
    if dry_run or not color:
        return
    owner, name = repo.split("/", 1)
    r = gh(["api", f"repos/{owner}/{name}/labels", "--method", "POST",
            "-f", f"name={label}", "-f", f"color={color}",
            "-f", f"description={description[:100]}"], check=False)
    if r.returncode != 0 and "already_exists" not in (r.stderr + r.stdout):
        print(f"   (could not set the colour for {label}: {r.stderr.strip()[:120]})")


def set_label(repo, num, label, *, color=None, description="", dry_run=False):
    """Replace any previous burnish:* label with this one, via the REST API.

    REST rather than `gh pr edit`: that path goes through a GraphQL query that fails on
    repositories with Projects-classic disabled, which is an outage the bot should not share.
    """
    owner, name = repo.split("/", 1)
    if dry_run:
        print(f"   [dry-run] would label #{num}: {label}"
              + (f"  (#{color})" if color else ""))
        return
    ensure_label(repo, label, color, description)
    cur = gh(["api", f"repos/{owner}/{name}/issues/{num}/labels", "--jq", "[.[].name]"])
    for old in json.loads(cur.stdout or "[]"):
        if old.startswith(f"{V.PREFIX}:") and old != label:
            gh(["api", f"repos/{owner}/{name}/issues/{num}/labels/{old}",
                "--method", "DELETE"], check=False)
    gh(["api", f"repos/{owner}/{name}/issues/{num}/labels",
        "--method", "POST", "-f", f"labels[]={label}"])


def comment(repo, num, body, *, dry_run=False):
    if dry_run:
        print(f"   [dry-run] would comment on #{num}:\n{body}")
        return
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(body)
        path = f.name
    try:
        gh(["pr", "comment", str(num), "-R", repo, "--body-file", path])
    finally:
        os.unlink(path)


def guard(worktree: Path, base: str) -> dict:
    out = Path(tempfile.mkdtemp()) / "guard.json"
    # --repo, not cwd. The guard script resolves its own repository from __file__, so running
    # it with a different working directory diffs the WRONG tree and reports clean. The
    # instrument copy is deliberately this repo's (the base one), not the submission's.
    subprocess.run([sys.executable, str(ROOT / "scripts" / "instrument_guard.py"),
                    "--base", base, "--repo", str(worktree), "--json", str(out)],
                   capture_output=True, text=True)
    return json.loads(out.read_text()) if out.exists() else {"ok": True, "outcome": "CONTRIBUTOR"}


def report(verdict: dict, receipt: dict, *, raw_name, receipt_name) -> str:
    """What the bot writes on the PR. Says what was measured and how to check it."""
    v = verdict
    lines = [f"### `{v['label']}`", "", f"**{v['headline']}**", "", v["meaning"], ""]
    if v["measured_gap_closed"] is not None:
        ci = v["confidence_interval"]
        lines += ["| | |", "|:--|--:|",
                  f"| measured gap closed | `{v['measured_gap_closed']:+.4f}` |"]
        if ci and ci[0] is not None:
            lines += [f"| {int(100 * (v['confidence_level'] or 0))}% interval | "
                      f"`[{ci[0]:+.4f}, {ci[1]:+.4f}]` |"]
        lines += [f"| resolved against the cell's own floor | "
                  f"`{'yes' if v['resolved'] else 'no'}` |",
                  f"| **credited** | **`{v['payout_fraction']:.4f}`** |", ""]
    per = receipt.get("per_cell") or {}
    if per:
        lines += ["| cell | gap closed | achieved | floor | resolved |",
                  "|:--|--:|--:|--:|:--:|"]
        for cid in sorted(per):
            c = per[cid]
            lines.append(
                f"| `{cid}` | `{c.get('gap_closed', 0):+.4f}` | "
                f"`{100 * (c.get('achieved_base') or 0):.1f}% -> "
                f"{100 * (c.get('achieved_candidate') or 0):.1f}%` | "
                f"`{c.get('floor_pct', 0):.3f}%` | "
                f"{'yes' if c.get('resolved') else 'no'} |")
        lines.append("")
    # The commit this measured, stated on the pull request rather than only in the receipt.
    #
    # A round freezes each submission's head when it starts, so a push during evaluation cannot
    # reach the measurement -- but GitHub does not remove a label when you push, so the author
    # would see a verdict that looks like it describes their new head. Naming the commit makes
    # an overtaken label visibly stale instead of quietly wrong.
    commit = (receipt.get("provenance") or {}).get("candidate_commit")
    if commit:
        lines += [f"Measured at `{commit[:12]}`. A round freezes the head commit when it "
                  f"starts, so anything pushed after that is not in this result — it will be "
                  f"measured in a later round.", ""]
    lines += [
        "---",
        "",
        "**Check this yourself — no GPU, about two seconds.** The measurements are published "
        "next to the receipt, and the verdict is a pure function of them:",
        "",
        "```bash",
        f"burnish audit {raw_name} {receipt_name}",
        "```",
        "",
        "That re-scores the raw measurements and compares the result to the published receipt, "
        "field for field. It catches a scoring bug or an edited receipt — including one whose "
        "digest was recomputed to cover the edit. It cannot prove the measurements describe "
        "what the hardware did; for that, re-measure on your own RTX 5090 and append a "
        "counter-receipt with `burnish challenge`. A disagreement beyond the cell's own noise "
        "floor puts the credit on hold.",
    ]
    if not v["provenance_complete"]:
        lines += ["", "> This receipt cannot name the code it scored, so it is a valid "
                      "measurement and not evidence about a particular commit."]
    return "\n".join(lines)


def evaluate(repo, pr, args) -> dict:
    num = pr["number"]
    print(f">> #{num}  {pr['title'][:70]}")
    work = Path(tempfile.mkdtemp(prefix=f"burnish-pr{num}-"))
    try:
        subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "-q", "--detach",
                        str(work / "src"), pr["headRefOid"]], check=True,
                       capture_output=True, text=True)
        wt = work / "src"

        g = guard(wt, args.base)
        if not g["ok"]:
            print(f"   SKIPPED: changes the instrument ({len(g['blocked'])} path(s))")
            set_label(repo, num, SKIPPED, dry_run=args.dry_run)
            comment(repo, num, _skip_note(g), dry_run=args.dry_run)
            return {"pr": num, "outcome": "SKIPPED", "guard": g}

        if args.dry_run:
            print(f"   [dry-run] would evaluate: {g['outcome']}")
            return {"pr": num, "outcome": "DRY_RUN", "guard": g}

        # A cartography submission is a different evaluation, not a harder speedup. Routing it
        # through the speedup path is what the repository did until now: the overlay stripped the
        # new generation, the submission changed nothing measurable in BG-1, and it came back
        # `unresolved`. Declared payable, no path to payment.
        if g["outcome"] == "CARTOGRAPHY":
            return _evaluate_cartography(repo, num, wt, g, args)

        # Checked HERE rather than at startup. The guard runs first and costs nothing, so a pass
        # over a queue of instrument-only pull requests needs no checkpoint, no noise file and
        # no ledger -- and demanding them up front would stop an operator from running the
        # cheap half of the loop on a machine that has no GPU attached to it.
        missing = [n for n in ("ledger", "weights", "noise") if not getattr(args, n)]
        if missing:
            raise RuntimeError(
                "this submission needs a measurement, and "
                + ", ".join(f"--{m}" for m in missing) + " "
                + ("is" if len(missing) == 1 else "are") + " not set. The guard pass before "
                "this point needs none of them, which is why they are not required at startup.")

        build = subprocess.run(["./scripts/build_cuda.sh"], cwd=str(wt),
                               capture_output=True, text=True, timeout=3600)
        if build.returncode != 0:
            set_label(repo, num, f"{V.PREFIX}:build-fail")
            comment(repo, num, "### `burnish:build-fail`\n\nThe submission did not build on the "
                               "eval box.\n\n```\n" + build.stdout[-2500:] + "\n```")
            return {"pr": num, "outcome": "BUILD_FAIL"}

        out_dir = work / "run"
        r = subprocess.run(
            [str(wt / "eval" / "score_submission.sh"),
             "--base", args.base, "--worktree", str(wt),
             "--impl-base", args.impl_base, "--impl-candidate", args.impl_candidate,
             "--pr", str(num), "--ledger", args.ledger,
             "--generation", args.generation,
             "--weights", args.weights, "--noise", args.noise,
             *(["--calibration", args.calibration] if args.calibration else []),
             "--work-dir", str(out_dir)],
            capture_output=True, text=True, timeout=args.timeout)
        print(r.stdout[-1500:])
        if r.returncode != 0 or not (out_dir / "receipt.json").exists():
            set_label(repo, num, f"{V.PREFIX}:eval-error")
            comment(repo, num, "### `burnish:eval-error`\n\nThe evaluator failed. This is not "
                               "the submission's fault and it will be re-run.\n\n```\n"
                    + (r.stdout + r.stderr)[-2500:] + "\n```")
            return {"pr": num, "outcome": "EVAL_ERROR"}

        receipt = json.loads((out_dir / "receipt.json").read_text())
        v = V.verdict(receipt)

        # Publish the MEASUREMENTS beside the receipt. A receipt whose raw file was thrown away
        # cannot be re-derived, which would make every claim in this bot's comment unverifiable.
        rid = f"pr-{num:06d}"
        pub = Path(args.ledger) / receipt["benchmark_generation"] / "raw"
        pub.mkdir(parents=True, exist_ok=True)
        shutil.copy(out_dir / "raw.json", pub / f"{rid}-raw.json")

        set_label(repo, num, v["label"], color=V.color_for(receipt),
                  description=v["headline"])
        comment(repo, num, report(v, receipt,
                                  raw_name=f"{rid}-raw.json", receipt_name=f"{rid}.json"))
        print(f"   {v['label']}   pays {v['payout_fraction']:.4f}")
        return {"pr": num, "outcome": v["status"], "label": v["label"],
                "payout_fraction": v["payout_fraction"]}
    finally:
        subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force",
                        str(work / "src")], capture_output=True)
        shutil.rmtree(work, ignore_errors=True)


def _evaluate_cartography(repo, num, wt, g, args) -> dict:
    """Ask whether the cell is real and measurable, not whether it got faster."""
    import re as _re
    names = sorted({m.group(1) for m in
                    (_re.match(r"eval/cells/([^/]+)/", p) for p in g["cartography"]) if m})
    out = Path(tempfile.mkdtemp()) / "cartography.json"
    cmd = [sys.executable, str(ROOT / "eval" / "cartography.py"), "check",
           "--generation", names[0], "--base", args.base, "--repo", str(wt),
           "--cells-root", str(wt / "eval" / "cells"), "--json", str(out)]
    if args.weights and args.noise:
        cmd += ["--measure", "--binary", str(wt / "build-cuda" / "burnisher"),
                "--weights", args.weights, "--noise", args.noise]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    print(r.stdout[-2000:])
    result = json.loads(out.read_text()) if out.exists() else {"pass": False, "checks": []}
    v = result.get("verdict") or {}

    if v.get("outcome") == "OPENED":
        label = f"{V.PREFIX}:cell-opened"
    elif v.get("outcome") == "UNMEASURED":
        label = f"{V.PREFIX}:eval-error"
    else:
        label = f"{V.PREFIX}:partial"
    set_label(repo, num, label, dry_run=args.dry_run)
    comment(repo, num, _cartography_note(names, result, v), dry_run=args.dry_run)
    print(f"   {label}   {v.get('outcome')}")
    return {"pr": num, "outcome": v.get("outcome"), "label": label, "generations": names}


def _cartography_note(names, result, v) -> str:
    rows = "\n".join(
        f"- {'PASS' if c['pass'] else '**FAIL**'} — {c['check']}"
        + (f"  \n  `{c['detail']}`" if c.get("detail") else "")
        for c in result.get("checks", []))
    measured = result.get("measured") or {}
    table = ""
    if measured:
        table = "\n\n| cell | achieved | floor | room | resolvable |\n|:--|--:|--:|--:|:--:|\n" + \
            "\n".join(f"| `{c}` | `{100 * m['achieved']:.1f}%` | `{m['floor_pct']:.3f}%` | "
                       f"`{m['floors_of_room']:.0f} floors` | "
                       f"{'yes' if m['resolvable'] else '**no**'} |"
                       for c, m in sorted(measured.items()))
    claimed = result.get("claimed_calibration") or {}
    claim_note = ""
    if claimed:
        claim_note = (
            "\n\nThis submission shipped a calibration of its own. It was read and **not "
            "used** — every number above was measured here, by this evaluator, on this box. "
            "That asymmetry is deliberate: you supply the cell definition and the oracle, the "
            "evaluator supplies the measurement, so a favourable floor cannot be submitted.")
    tail = ""
    if v.get("unresolvable_cells"):
        tail = ("\n\n**Some of these cells cannot resolve a contribution at their measured "
                "floor, and that is published as a result rather than as a failure.** It is "
                "worth more than a cell that looks open and is not — the alternative is "
                "somebody spending a week inside a noise floor.")
    return (f"### `{v.get('outcome', 'REJECTED')}` — cartography: {', '.join(names)}\n\n"
            f"This submission opens a new cell rather than closing a gap in an existing one, so "
            f"it is asked a different question: *is this cell real, and can anybody be credited "
            f"on it?*\n\n{rows}{table}{claim_note}{tail}\n\n"
            f"See `docs/CARTOGRAPHY.md` for what a cell has to come with.")


def _skip_note(g) -> str:
    rows = "\n".join(f"- `{b['path']}`\n  {b['why']}" for b in g["blocked"])
    return (
        f"### `{SKIPPED}`\n\n"
        f"This pull request changes the **measuring instrument**, so it was not evaluated — and "
        f"no GPU time was spent on it. A number produced by a modified instrument cannot be "
        f"accepted either way, so measuring it first would be waste rather than diligence.\n\n"
        f"{rows}\n\n"
        f"**This is not a rejection.** Improving the evaluator is a real contribution — the "
        f"evaluator is where the bugs are, and a broken one prints a confident number. It is "
        f"separated, not refused: send it as its own pull request, scored as a change to what "
        f"is measured rather than riding in on a change it would score.\n\n"
        f"**Opening a new cell is different and is paid.** Adding a new frozen generation under "
        f"`eval/cells/<name>/` is cartography: it cannot change what any existing receipt meant, "
        f"so it is allowed where editing one is not. See `docs/CARTOGRAPHY.md`.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pr", type=int, help="evaluate one PR instead of polling")
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--ledger", default=os.environ.get("BURNISH_LEDGER", ""))
    ap.add_argument("--weights", default=os.environ.get("BURNISH_WEIGHTS", ""))
    ap.add_argument("--noise", default=os.environ.get("BURNISH_NOISE", ""))
    ap.add_argument("--generation", default=os.environ.get("BURNISH_GENERATION", "BG-1"),
                    help="the frozen generation this round scores; --noise must be that "
                         "generation's pinned noise")
    ap.add_argument("--calibration", default=os.environ.get("BURNISH_CALIBRATION", ""),
                    help="an anchor other than the generation's committed one. Normally empty: "
                         "every card of the pinned class scores against the committed anchor.")
    ap.add_argument("--impl-base", default="cuda")
    ap.add_argument("--impl-candidate", required=False, default="cuda")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="classify and report, touch no GPU and write no labels")
    ap.add_argument("--json", help="write the pass result here")
    a = ap.parse_args()


    prs = ([p for p in open_prs(a.repo) if p["number"] == a.pr] if a.pr
           else [p for p in open_prs(a.repo) if not already_labelled(p)])
    if not prs:
        print("nothing to evaluate")
        return 0

    results = []
    for pr in prs:
        try:
            results.append(evaluate(a.repo, pr, a))
        except Exception as exc:                      # one bad PR must not stop the pass
            print(f"!! #{pr['number']}: {exc}", file=sys.stderr)
            results.append({"pr": pr["number"], "outcome": "EVAL_ERROR", "error": str(exc)})
        if not a.once and not a.pr:
            time.sleep(2)

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"repo": a.repo, "results": results}, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
