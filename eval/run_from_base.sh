#!/usr/bin/env bash
# Score a submission with the MEASURING INSTRUMENT taken from the base commit.
#
#   eval/run_from_base.sh <base-ref> <submission-worktree> [-- args...]
#   BURNISH_ENTRY=score eval/run_from_base.sh <base-ref> <worktree> -- raw.json --pr 184
#
# Why this exists. `eval/`, `configs/`, the frozen generations and the model pin define WHAT IS
# MEASURED. If the evaluator runs the submission's copy of them, a submission can win by editing
# the ruler: a one-line change to a noise floor, a repeat count, a confidence level, a roofline
# ceiling, a tolerance, a held-out shape list, or the model revision the whole comparison is
# against. None of those look like cheating in a diff. Several of them look like tidying.
#
# So the submission supplies src/, include/ and tools/ -- the parts it is being scored ON -- and
# this script overlays the instrument from <base-ref> before running anything. It is a script
# and not a policy document because a rule nothing enforces is a rule that holds only for honest
# submissions.
#
# This is methodology. It takes no view on who may submit and there is no tier or payout in it.
set -euo pipefail

BASE="${1:?usage: run_from_base.sh <base-ref> <submission-worktree> [-- args...]}"
SUB="${2:?usage: run_from_base.sh <base-ref> <submission-worktree> [-- args...]}"
shift 2
[ "${1:-}" = "--" ] && shift || true

[ -d "$SUB/.git" ] || [ -f "$SUB/.git" ] || { echo "!! $SUB is not a git worktree"; exit 2; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Everything that decides WHAT IS MEASURED rather than how well the candidate does it.
#
# One list, used for both the overlay and the report, so the two cannot drift: a path reported
# as instrument but not overlaid is a hole, and one overlaid but not reported is a silent
# discard. `configs/` is on it because the rooflines are computed from it -- moving a device
# peak changes every achieved fraction without touching a line of evaluator code, and moving a
# model revision changes the oracle.
INSTRUMENT=(
    eval
    configs
    schemas
    tools/burnish
)

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

present=()
for path in "${INSTRUMENT[@]}"; do
    if git -C "$REPO" cat-file -e "$BASE:$path" 2>/dev/null; then
        present+=("$path")
    else
        # Loud, not silent. A rename that was not reflected in INSTRUMENT is a hole in the
        # anti-gaming property, and a quiet skip is exactly how it would go unnoticed.
        echo ">> NOTE: $path is not in $BASE; nothing to overlay from there."
    fi
done
git -C "$REPO" archive "$BASE" "${present[@]}" | tar -x -C "$STAGE"
BASE_SHA="$(git -C "$REPO" rev-parse --short "$BASE")"
echo ">> instrument: ${present[*]} from $BASE_SHA"

# Report what the submission tried to change about the instrument rather than dropping it
# silently. A submission may legitimately want to improve the evaluator -- that is a
# contribution, and it is scored as a change to what is measured, in its own PR, against the
# generation it would create.
CHANGED="$(git -C "$SUB" diff --name-only "$BASE" -- "${INSTRUMENT[@]}" 2>/dev/null || true)"
if [ -n "$CHANGED" ]; then
    echo ">> NOTE: the submission modifies the instrument. These changes are NOT used for scoring:"
    echo "$CHANGED" | sed 's/^/     /'
    echo ">>       Score them separately, as a change to what is measured."
fi

# A frozen generation that already carries receipts must not move at all. Editing one silently
# re-scores history; the ledger refuses it at write time, and saying so here is cheaper than
# finding out after a benchmark run.
if echo "$CHANGED" | grep -q '^eval/cells/'; then
    echo "!! the submission edits a FROZEN GENERATION under eval/cells/."
    echo "   A generation is frozen for its lifetime and its receipts stay attached to it."
    echo "   If the meaning of the evaluation should change, the answer is a new generation."
    exit 2
fi

# Self-check: the staged instrument has to be runnable, or the run fails later with a stack
# trace that reads like a submission bug.
for required in eval/burnscore/__init__.py eval/screen.py tools/burnish; do
    [ -e "$STAGE/$required" ] || { echo "!! $BASE has no $required"; exit 2; }
done
chmod +x "$STAGE/tools/burnish" 2>/dev/null || true

# The runtime under test comes from the SUBMISSION; the instrument driving it comes from base.
export BURNISHER_BIN="${BURNISHER_BIN:-$SUB/build/burnisher}"
export BURNISH_INSTRUMENT_FROM="$BASE_SHA"
# A scoring run takes tens of minutes and its output is almost always redirected to a log, which
# means Python block-buffers it and the log stays empty until the stage ends. Somebody watching a
# forty-minute bench cannot tell a working run from a hung one, and the reasonable thing to do
# about a run that looks hung is kill it. Line-buffered costs nothing here.
export PYTHONUNBUFFERED=1
# The instrument runs from a staging directory that is not a git repository -- `git archive`
# extracts files, not history -- so the evaluator cannot find out what it is scoring by asking
# git about its own location. It would get nothing, and it did: the first receipts this harness
# produced named the instrument and left the candidate blank. Only this script knows where the
# submission is, so it passes both commits down.
export BURNISH_CANDIDATE_COMMIT="$(git -C "$SUB" rev-parse HEAD 2>/dev/null || true)"
export BURNISH_BASE_COMMIT="$(git -C "$REPO" rev-parse "$BASE" 2>/dev/null || true)"
# Printed, not just exported. A receipt that cannot name the code it scored is not evidence, and
# the run log is where somebody notices that before forty minutes of GPU time have been spent on
# a receipt that will come back unprovenanced.
if [ -n "$BURNISH_CANDIDATE_COMMIT" ]; then
    echo ">> scoring candidate $BURNISH_CANDIDATE_COMMIT against base $BURNISH_BASE_COMMIT"
else
    echo "!! NOTE: no git metadata for $SUB, so this receipt will not be able to name the code"
    echo "         it scored. It will still be a valid measurement; it will not be evidence"
    echo "         that a particular commit earned anything."
fi
echo ">> runtime under test: $BURNISHER_BIN"

case "${BURNISH_ENTRY:-bench}" in
    bench|calibrate|gate|score|receipt|ledger|screen|roofline|generation)
        exec python3 "$STAGE/tools/burnish" "${BURNISH_ENTRY:-bench}" "$@"
        ;;
    *)
        echo "!! BURNISH_ENTRY must be a burnish subcommand, not '${BURNISH_ENTRY}'"
        exit 2
        ;;
esac
