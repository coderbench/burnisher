#!/usr/bin/env bash
# Turn a freshly rented GPU box into the evaluator, from your own machine. One command per box:
#
#   eval/provision_box.sh root@91.224.44.226 -p 30035
#   eval/provision_box.sh --no-rounds root@91.224.44.226 -p 30035   # everything but the cron job
#
# Everything after the options is passed to ssh as it is.
#
# The tokens live in ONE file on your machine, written once and never committed:
#
#   ~/.config/burnisher/secrets.env        (chmod 600; BURNISH_SECRETS_FILE to put it elsewhere)
#     export GH_TOKEN=github_pat_...                 pull requests + issues on the scored repo
#     export BURNISH_LEDGER_REMOTE=https://github.com/coderbench/burnisher-ledger.git
#     export BURNISH_LEDGER_TOKEN=github_pat_...     contents on the ledger repo only
#
# It travels over ssh's stdin into a root-only file on the box, never on a command line, which every
# account on the box can read. Rented boxes are returned with their disks; this file is what makes
# the next one a single command.
#
# Safe to run again on the same box: each step skips what is already done.
set -euo pipefail

ROUNDS=1
[ "${1:-}" = "--no-rounds" ] && { ROUNDS=0; shift; }
[ $# -gt 0 ] || { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

SECRETS="${BURNISH_SECRETS_FILE:-$HOME/.config/burnisher/secrets.env}"
[ -f "$SECRETS" ] || { echo "!! no $SECRETS -- write it once, as described at the top of $0"; exit 2; }
case "$(stat -c %a "$SECRETS")" in
    600|400) ;;
    *) echo "!! $SECRETS is readable by other accounts here: chmod 600 $SECRETS"; exit 2 ;;
esac
for name in GH_TOKEN BURNISH_LEDGER_REMOTE BURNISH_LEDGER_TOKEN; do
    grep -q "^export $name=." "$SECRETS" || { echo "!! $SECRETS does not set $name"; exit 2; }
done

SSH=(ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "$@")

echo ">> sending the tokens"
"${SSH[@]}" 'umask 077 && mkdir -p /var/lib/burnish && chmod 700 /var/lib/burnish && cat > /var/lib/burnish/secrets.env' < "$SECRETS"

echo ">> setting up the box (a new one takes about ten minutes)"
"${SSH[@]}" "ROUNDS=$ROUNDS bash -s" <<'REMOTE'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
say() { echo; echo ">> $*"; }
REPO=https://github.com/coderbench/burnisher.git
R=/workspace/burnisher
STATE=/var/lib/burnish

. /etc/os-release
[ "$VERSION_ID" = "24.04" ] && [ "$(uname -m)" = x86_64 ] || { echo "!! expects Ubuntu 24.04 on x86_64"; exit 2; }
nvidia-smi -L

say "stopping every listening service but ssh (rented images start a root notebook server)"
for pid in $(ss -ltnpH | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u); do
    comm="$(cat /proc/$pid/comm 2>/dev/null || true)"
    [ "$comm" = sshd ] || { echo "   stopping $comm (pid $pid)"; kill "$pid" || true; }
done

say "packages"
# A toolkit the image already ships is kept: 12.8 and 13.x both build the runtime, and a second one
# beside it is gigabytes for nothing.
TOOLKIT=cuda-toolkit-12-8
[ -x /usr/local/cuda/bin/nvcc ] && { TOOLKIT=; /usr/local/cuda/bin/nvcc --version | grep release; }
if ! dpkg -s $TOOLKIT cmake build-essential cron python3-pip gh >/dev/null 2>&1; then
    apt-get update -q
    apt-get install -y -q --no-install-recommends ca-certificates curl
    if [ ! -f /usr/share/keyrings/cuda-archive-keyring.gpg ]; then
        curl -fsSLo /tmp/cuda-keyring.deb \
            https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
        dpkg -i /tmp/cuda-keyring.deb
    fi
    curl -fsSLo /usr/share/keyrings/githubcli-archive-keyring.gpg \
        https://cli.github.com/packages/githubcli-archive-keyring.gpg
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list
    apt-get update -q
    apt-get install -y -q --no-install-recommends $TOOLKIT cmake build-essential cron python3-pip gh
fi
# cuDNN for the CUDA major the toolkit is, because the build links the one its headers match.
CUDA_MAJOR="$(/usr/local/cuda/bin/nvcc --version | grep -oE 'release [0-9]+' | grep -oE '[0-9]+')"
dpkg -s "libcudnn9-dev-cuda-$CUDA_MAJOR" >/dev/null 2>&1 || \
    apt-get install -y -q --no-install-recommends "libcudnn9-dev-cuda-$CUDA_MAJOR"
python3 -c "import numpy, huggingface_hub, sentencepiece" 2>/dev/null || \
    python3 -m pip install -q --break-system-packages numpy huggingface_hub sentencepiece

say "code"
[ -d "$R/.git" ] || git clone -q "$REPO" "$R"
git -C "$R" fetch -q origin main
git -C "$R" checkout -q main
git -C "$R" merge -q --ff-only origin/main
git -C "$R" log --oneline -1

say "checkpoint, at the revisions the frozen generation pins"
python3 -u - <<'PY'
import json
from huggingface_hub import snapshot_download
m = json.load(open("/workspace/burnisher/eval/cells/BG-1/generation.json"))["model"]
snapshot_download(m["repo"], revision=m["revision"],
                  allow_patterns=["transformer/*", "vae/*"], local_dir="/workspace/ckpt")
snapshot_download(m["text_encoder_repo"], revision=m["text_encoder_revision"],
                  allow_patterns=["text_encoder/*", "tokenizer/*"], local_dir="/workspace/ckpt")
PY
chmod -R a+rX /workspace/ckpt

say "build"
cd "$R"
export PATH=/usr/local/cuda/bin:$PATH
./scripts/build_cuda.sh > "$STATE/build.log" 2>&1 || { tail -30 "$STATE/build.log"; exit 1; }
build-cuda/burnisher check-weights --weights /workspace/ckpt | grep required

say "the pinned starting noise"
NOISE=/workspace/noise1024.npy
want="$(python3 -c 'import json; print(json.load(open("eval/cells/BG-1/reference-latents/manifest.json"))["noise_sha256"])')"
[ -f "$NOISE" ] || build-cuda/burnisher noise --out "$NOISE" >/dev/null
chmod 644 "$NOISE"
[ "$(sha256sum "$NOISE" | cut -d' ' -f1)" = "$want" ] || { echo "!! $NOISE is not the pinned noise"; exit 1; }
echo "   sha256 matches the reference latents' manifest"

say "configuration"
cat > "$R/.env.eval" <<ENV
export BURNISH_REPO=coderbench/burnisher
export BURNISH_LEDGER=$STATE/ledger
export BURNISH_WEIGHTS=/workspace/ckpt
export BURNISH_NOISE=$NOISE
export BURNISH_SANDBOX_USER=burnish-sandbox
export BURNISH_STATE=$STATE
export BURNISH_ROUND_LOCK=$STATE/round.lock
export BURNISH_EVAL_LOCK=$STATE/eval.lock
export BURNISH_SECRETS=$STATE/secrets.env
. $STATE/secrets.env
ENV
chmod 600 "$R/.env.eval"
set -a; . "$R/.env.eval"; set +a

say "ledger: continue the published history rather than start a new one"
if [ ! -d "$BURNISH_LEDGER/.git" ]; then
    if [ -n "$(ls -A "$BURNISH_LEDGER" 2>/dev/null)" ]; then
        echo "!! $BURNISH_LEDGER has files but is not a clone of the ledger; not touching it"; exit 1
    fi
    rm -rf "$BURNISH_LEDGER"
    GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=http.extraHeader \
    GIT_CONFIG_VALUE_0="Authorization: Basic $(printf 'x-access-token:%s' "$BURNISH_LEDGER_TOKEN" | base64 -w0)" \
        git clone -q "$BURNISH_LEDGER_REMOTE" "$BURNISH_LEDGER"
fi

say "the sandbox account and the labels"
eval/setup_sandbox.sh > /dev/null
eval/setup_labels.sh "$BURNISH_REPO" > /dev/null

say "box check"
python3 -u eval/pr_bot.py --repo "$BURNISH_REPO" --check-box

if [ "$ROUNDS" = 1 ]; then
    say "rounds every two hours"
    service cron start > /dev/null 2>&1 || cron
    line="0 */2 * * * $R/eval/run_round_cron.sh >> /var/log/burnish-round.log 2>&1"
    ( crontab -l 2>/dev/null | grep -v run_round_cron.sh || true; echo "$line" ) | crontab -
    crontab -l | grep run_round_cron.sh
else
    say "rounds NOT started (--no-rounds). Practice first: eval/round.py --repo \$BURNISH_REPO --dry-run"
fi
echo; echo ">> the box is the evaluator"
REMOTE
