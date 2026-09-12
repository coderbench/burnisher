#!/usr/bin/env bash
# Take one submission from a built binary to a signed receipt in the ledger. One command.
#
#   eval/score_submission.sh --base <ref> --worktree <dir> \
#       --impl-base cuda --impl-candidate <name> --pr <n> --ledger <dir outside the worktree>
#
# Why this is a script in the repository rather than instructions in a document.
#
# The four stages below have to run in this order, with these guards, or the receipt means
# something other than it says. The first time this pipeline ran it was an ad-hoc shell script on
# a rented box, which is fine for one run and wrong for a scored subnet: two validators following
# the same prose would write two different drivers, and the difference between them would show up
# as a difference in scores that no receipt could explain. A submission's score must depend on the
# submission.
#
# The order, and what each step is protecting:
#
#   1. gate the BASE arm        The comparison's reference point has to pass the same correctness
#                               and determinism checks as the candidate. A base that does not
#                               reproduce itself makes every delta measured against it noise.
#   2. gate the CANDIDATE arm   Correctness before speed, always. A submission that fails is
#                               REJECTED, not traded off against a latency win, and the pipeline
#                               stops here rather than timing something that is wrong.
#   3. paired interleaved bench Both arms, one process, one model load, alternating. Clocks
#                               cannot be pinned in a container, so only paired same-box deltas
#                               mean anything. The held-out shape is drawn HERE, after the
#                               candidate is frozen.
#   4. score                    Into an append-only ledger, outside the submission's reach.
#
# Between every stage the device must be idle. Two benchmarks at once race for VRAM and both
# results are worthless -- the runners check, and this script checks before it starts.
set -euo pipefail

BASE=""; SUB=""; IMPL_BASE="cuda"; IMPL_CAND=""; PR=""; LEDGER=""
WEIGHTS="${BURNISH_WEIGHTS:-}"; NOISE="${BURNISH_NOISE:-}"
GEN="BG-1"; REPEATS=5; GATE_REPEATS=2; DTYPE="fp32"; DEVICE="cuda"; OUT=""

usage() {
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-2}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --base)            BASE="$2"; shift 2 ;;
        --worktree)        SUB="$2"; shift 2 ;;
        --impl-base)       IMPL_BASE="$2"; shift 2 ;;
        --impl-candidate)  IMPL_CAND="$2"; shift 2 ;;
        --pr)              PR="$2"; shift 2 ;;
        --ledger)          LEDGER="$2"; shift 2 ;;
        --weights)         WEIGHTS="$2"; shift 2 ;;
        --noise)           NOISE="$2"; shift 2 ;;
        --generation)      GEN="$2"; shift 2 ;;
        --repeats)         REPEATS="$2"; shift 2 ;;
        --gate-repeats)    GATE_REPEATS="$2"; shift 2 ;;
        --gate-dtype)      DTYPE="$2"; shift 2 ;;
        --device)          DEVICE="$2"; shift 2 ;;
        --work-dir)        OUT="$2"; shift 2 ;;
        -h|--help)         usage 0 ;;
        *) echo "!! unknown argument '$1'"; usage ;;
    esac
done

# Variable name -> the flag a caller actually types. Deriving the flag from the variable name
# printed "--sub is required" for a flag spelled --worktree, which sends the reader to the
# wrong place in the usage text.
require() { [ -n "$2" ] || { echo "!! $1 is required"; usage; }; }
require --base "$BASE"
require --worktree "$SUB"
require --impl-candidate "$IMPL_CAND"
require --pr "$PR"
require --ledger "$LEDGER"
require "--weights (or BURNISH_WEIGHTS)" "$WEIGHTS"
require "--noise (or BURNISH_NOISE)" "$NOISE"

# The ledger is append-only and must live where the submission cannot reach it. A ledger inside
# the worktree can be rewritten by the thing being scored, which is not a ledger.
case "$(readlink -f "$LEDGER")" in
    "$(readlink -f "$SUB")"/*) echo "!! the ledger is inside the submission worktree. A" \
        "submission that can rewrite its own history is not being scored."; exit 2 ;;
esac

OUT="${OUT:-$(mktemp -d)}"
mkdir -p "$OUT"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$REPO/eval/run_from_base.sh"
BIN="${BURNISHER_BIN:-$SUB/build-cuda/burnisher}"
[ -x "$BIN" ] || { echo "!! no runtime at $BIN -- build the submission first"; exit 2; }
export BURNISHER_BIN="$BIN"

REF="eval/cells/$GEN/reference-latents"
say() { echo; echo "### $*"; }

# Refuse to start beside another benchmark. The runners check this too, before each measurement;
# checking here as well turns a failure forty minutes in into one before anything is loaded.
if command -v nvidia-smi >/dev/null 2>&1; then
    busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader || true)"
    [ -z "$busy" ] || { echo "!! the device is busy (pids: $busy). Never run two benchmarks at" \
        "once; they race for VRAM and both results are wrong. Kill by PID."; exit 2; }
fi

say "1/4  gate the BASE arm ($IMPL_BASE)"
BURNISH_ENTRY=gate "$RUN" "$BASE" "$SUB" -- \
    --binary "$BIN" --weights "$WEIGHTS" --impl "$IMPL_BASE" --device "$DEVICE" \
    --dtype "$DTYPE" --noise "$NOISE" --reference "$REF" \
    --repeats "$GATE_REPEATS" --output "$OUT/gate-base.json"

say "2/4  gate the CANDIDATE arm ($IMPL_CAND)"
BURNISH_ENTRY=gate "$RUN" "$BASE" "$SUB" -- \
    --binary "$BIN" --weights "$WEIGHTS" --impl "$IMPL_CAND" --device "$DEVICE" \
    --dtype "$DTYPE" --noise "$NOISE" --reference "$REF" \
    --repeats "$GATE_REPEATS" --output "$OUT/gate-cand.json"

say "3/4  paired interleaved bench, $REPEATS repeats, both arms"
BURNISH_ENTRY=bench "$RUN" "$BASE" "$SUB" -- \
    --binary "$BIN" --weights "$WEIGHTS" --device "$DEVICE" \
    --impl-base "$IMPL_BASE" --impl-candidate "$IMPL_CAND" --repeats "$REPEATS" \
    --gate-result "$OUT/gate-cand.json" --gate-base-result "$OUT/gate-base.json" \
    --output "$OUT/raw.json"

say "4/4  score into the ledger"
BURNISH_ENTRY=score "$RUN" "$BASE" "$SUB" -- \
    "$OUT/raw.json" --generation "$GEN" --output "$OUT/receipt.json" \
    --ledger "$LEDGER" --pr "$PR"

say "done"
echo "   receipt      $OUT/receipt.json"
echo "   raw          $OUT/raw.json"
echo "   ledger       $LEDGER"
echo
echo "   The raw file is the measurement and the receipt is derived from it. Keep both: a"
echo "   receipt whose raw measurements were thrown away cannot be re-derived if the scorer"
echo "   is ever found to have a bug."
