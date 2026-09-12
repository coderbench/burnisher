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

# Colour by what the outcome MEANS to a contributor, not by severity. Green pays, blue is a
# real measurement that did not pay, grey is "we could not tell", red is a correctness
# rejection, amber is the evaluator's own fault.
declare -A COLOR=(
  [gap]=0E8A16              [cell-opened]=0E8A16
  [no-gain]=1D76DB          [moved-along-frontier]=1D76DB
  [unresolved]=C5DEF5       [partial]=C5DEF5       [shape-overfit]=FBCA04
  [correctness-fail]=B60205 [determinism-fail]=B60205
  [held]=8250DF             [skipped-instrument]=BFD4F2
  [build-fail]=F9A825       [eval-error]=F9A825
)

declare -A DESC=(
  [cell-opened]="cartography: opened a new cell, and is paid for it"
  [no-gain]="resolved, and measurably not an improvement"
  [moved-along-frontier]="faster, but paid for in memory or fidelity"
  [unresolved]="inside the cell's own measured noise -- open, not solved"
  [partial]="incomplete matrix; credits nothing"
  [shape-overfit]="the gain did not survive a shape drawn after the freeze"
  [correctness-fail]="rejected before timing was considered"
  [determinism-fail]="the build does not reproduce itself"
  [held]="an independent re-measurement disagrees beyond the noise floor"
  [skipped-instrument]="changes the measuring instrument; not evaluated, no GPU spent"
  [build-fail]="did not build on the eval box"
  [eval-error]="the evaluator failed; not the submission's fault"
)

# The list comes from the code, so a label the bot can apply is always a label that exists.
mapfile -t LABELS < <(cd "$HERE/.." && python3 -c "
import sys; sys.path.insert(0, 'eval')
from burnscore import verdict as V
for l in V.all_labels(): print(l.split(':', 1)[1])
")
LABELS+=("skipped-instrument")

for key in "${LABELS[@]}"; do
    gh label create "burnish:$key" -R "$REPO" \
       --color "${COLOR[$key]:-EDEDED}" \
       --description "${DESC[$key]:-burnish eval outcome}" --force >/dev/null
    echo "   burnish:$key"
done

echo
echo ">> created ${#LABELS[@]} outcome labels on $REPO"
echo "   The paying label is not among them: it carries the measured number"
echo "   (burnish:gap+0.0342) and the bot creates each one as it is earned."
