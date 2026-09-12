#!/usr/bin/env bash
# One-time: create the burnish:* labels the eval bot applies. Idempotent.
#
#   eval/setup_labels.sh [owner/repo]
#
# The paying outcome is NOT created here and cannot be: it carries the measured number
# (`burnish:gap+0.0342`), so there is no fixed string to pre-create -- the bot creates each one
# as it is earned. That is the intended shape. A fixed set of tier labels is a fixed set of
# answers, and this benchmark publishes the measurement instead.
set -euo pipefail
REPO="${1:-}"
[ -n "$REPO" ] || { echo "usage: eval/setup_labels.sh <owner/repo>"; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COUNT=0

# Colour and meaning both come from the CODE (burnscore/verdict.py), not from a second copy
# here. A hand-maintained colour map beside a hand-maintained label list is two things to forget
# to update, and the failure is silent: a label that exists with the wrong colour still works,
# so nobody notices it now means something else.
#
# The paying label is not created here and cannot be: it carries the measured number, so there
# is one per value. The bot creates each with a colour derived from the magnitude -- a bigger
# contribution is a deeper green -- because a label GitHub auto-creates gets a random colour.

while IFS=$'\t' read -r name color desc; do
    [ -n "$name" ] || continue
    gh label create "$name" -R "$REPO" --color "$color" --description "$desc" --force >/dev/null
    printf '   %-32s #%s  %s\n' "$name" "$color" "$desc"
    COUNT=$((COUNT + 1))
done < <(cd "$HERE/.." && python3 -c "
import sys; sys.path.insert(0, 'eval')
from burnscore import verdict as V
for status, (headline, _) in sorted(V.ALL_OUTCOMES.items()):
    label = f'{V.PREFIX}:' + status.lower().replace('_', '-')
    if label not in V.all_labels():
        continue                       # a paying status: created by the bot, with the number
    print('\t'.join((label, V.COLORS.get(status, 'EDEDED'), headline)))
print('\t'.join((f'{V.PREFIX}:skipped-instrument', V.GREY,
                  'changes the measuring instrument; not evaluated, no GPU spent')))
")

echo
echo ">> created $COUNT outcome labels on $REPO"
echo "   The paying label is not among them: it carries the measured number"
echo "   (burnish:gap+0.0342) and the bot creates each one as it is earned."
