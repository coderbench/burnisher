#!/usr/bin/env bash
# One-time, as root, on each evaluation box: the account submitted code is built and run as.
#
#   eval/setup_sandbox.sh [account]        # default: burnish-sandbox
#
# Idempotent. Creates the account with no extra groups and a private home, and makes the
# evaluator's own files unreachable from it. Whether that worked is not decided here: the bot
# checks AS the account before every round (eval/sandbox.py), and refuses to evaluate if anything
# secret is readable or anything it trusts is writable.
set -euo pipefail
NAME="${1:-burnish-sandbox}"
[ "$(id -u)" -eq 0 ] || { echo "!! run as root: the evaluator switches accounts, which needs root"; exit 2; }
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

id -u "$NAME" >/dev/null 2>&1 || \
    useradd --system --create-home --user-group --shell /usr/sbin/nologin "$NAME"
HOME_DIR="$(getent passwd "$NAME" | cut -d: -f6)"
install -d -m 700 -o "$NAME" -g "$NAME" "$HOME_DIR"

# Locks and round results. Their defaults are under /tmp, where the account could create them first.
install -d -m 700 "${BURNISH_STATE:-/var/lib/burnish}"

if [ -f "$REPO_DIR/.env.eval" ]; then
    chmod 600 "$REPO_DIR/.env.eval"
    set +u; . "$REPO_DIR/.env.eval"; set -u
fi
# Receipts, the evaluation records, the copycat record and the block list.
for d in "${BURNISH_LEDGER:-}" "${BURNISH_COPYCAT_CORPUS:-}"; do
    [ -z "$d" ] || { mkdir -p "$d"; chmod 700 "$d"; }
done

echo ">> $NAME is ready. Add to $REPO_DIR/.env.eval:"
echo "     export BURNISH_SANDBOX_USER=$NAME"
echo "   then check the box, which evaluates nothing:"
echo "     eval/pr_bot.py --repo <owner/name> --check-box"
