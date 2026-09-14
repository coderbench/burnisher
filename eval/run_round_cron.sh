#!/usr/bin/env bash
# Cron wrapper for one evaluation round. Every two hours:
#
#   0 */2 * * * /path/to/burnisher/eval/run_round_cron.sh >> /var/log/burnish-round.log 2>&1
#
# Three slots, not five. A submission costs about 24 measured GPU-minutes, so three fit a
# two-hour interval with 41 minutes of slack -- enough to absorb a slow build, a checkpoint
# reload, or a retry -- and five overrun the window outright. The slack matters more than the
# throughput: a round that runs past its interval meets the next one holding the lock, and the
# next one skips.
#
# The slot count should rise as the runtime gets faster. It is the one quantity here that
# improves without anybody working on it directly: scoring cost is proportional to how slow the
# runtime is, so every contribution the benchmark pays for makes the benchmark cheaper to run.
# Re-read docs/STATUS.md's measured cost before raising it, and raise it in one step.
set -euo pipefail

export HOME="${HOME:-/root}"
export PATH="/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Secrets and box-local paths. Not committed: the ledger lives outside the worktree, and the
# token is a token. BURNISH_CALIBRATION is optional and normally unset.
#   GH_TOKEN, BURNISH_REPO, BURNISH_LEDGER, BURNISH_WEIGHTS, BURNISH_NOISE, BURNISH_CALIBRATION
[ -f "$REPO_DIR/.env.eval" ] && . "$REPO_DIR/.env.eval"

: "${BURNISH_REPO:?set BURNISH_REPO=owner/name in .env.eval}"

# Bring the instrument up to date BEFORE the round. The evaluator scores against `main`, and a
# stale checkout would measure submissions against a baseline that has already moved -- which is
# the very thing the round's merge-one-rebase-the-rest rule exists to prevent.
git fetch --quiet origin main
git checkout --quiet main
git merge --quiet --ff-only origin/main

# Rebuild, because the base arm is the runtime at `main` and it has to be the current one.
./scripts/build_cuda.sh >/dev/null

exec python3 -u eval/round.py \
    --repo "$BURNISH_REPO" \
    --slots "${BURNISH_SLOTS:-3}" \
    --interval 120 \
    --base origin/main \
    --json "${BURNISH_ROUND_JSON:-/tmp/burnish-round-$(date -u +%Y%m%dT%H%M%SZ).json}" \
    ${BURNISH_AUTOMERGE:+--merge} \
    "$@"
