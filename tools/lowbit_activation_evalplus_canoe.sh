#!/usr/bin/env bash
set -euo pipefail

case "$(hostname)" in
  j-*-master-*) ;;
  *)
    echo "REFUSING: EvalPlus scoring may run only on a Canoe experiment node." >&2
    exit 78
    ;;
esac

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 SAMPLES_PATH RUN_ROOT [PARALLEL]" >&2
  exit 64
fi

REPO_ROOT=/minimax-avatar-new/zhangqian/realq/gptq_plus
SAMPLES_PATH=$1
RUN_ROOT=$2
PARALLEL=${3:-32}
PYTHON_BIN=${REALQ_PYTHON:-$REPO_ROOT/.venv/bin/python}
PYTHON_BIN_DIR=$(dirname "$PYTHON_BIN")
NO_INET="$REPO_ROOT/tools/bin/lowbit_activation_no_inet"
DATASET="$REPO_ROOT/datasets/reasoning_eval/humaneval_plus/HumanEvalPlus.jsonl.gz"
ENTRYPOINT="$REPO_ROOT/tools/lowbit_activation_evalplus_entrypoint.py"
STAGE_ROOT=$(mktemp -d /tmp/realq_evalplus.XXXXXX)

if [[ ! -f "$SAMPLES_PATH" ]]; then
  echo "Missing samples file: $SAMPLES_PATH" >&2
  exit 66
fi
if [[ ! -x "$NO_INET" ]]; then
  echo "Missing seccomp launcher: $NO_INET" >&2
  exit 69
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing audited Python interpreter: $PYTHON_BIN" >&2
  exit 69
fi
if [[ ! "$PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "PARALLEL must be a positive integer: $PARALLEL" >&2
  exit 64
fi

mkdir -p "$RUN_ROOT"
chown 65534:65534 "$STAGE_ROOT"
chmod 0700 "$STAGE_ROOT"
install -d -o 65534 -g 65534 -m 0700 "$STAGE_ROOT/home"
install -o 65534 -g 65534 -m 0600 \
  "$SAMPLES_PATH" "$STAGE_ROOT/evalplus_samples.jsonl"
install -o 65534 -g 65534 -m 0400 \
  "$DATASET" "$STAGE_ROOT/HumanEvalPlus.jsonl.gz"

set +e
/usr/bin/timeout --foreground --signal=TERM --kill-after=60s 7200s \
  /usr/bin/prlimit --as=68719476736 --cpu=21600 --nproc=4096 --nofile=1024 -- \
  /usr/bin/setpriv \
    --reuid=65534 \
    --regid=65534 \
    --clear-groups \
    --bounding-set=-all \
    --inh-caps=-all \
    --ambient-caps=-all \
    --no-new-privs \
  /usr/bin/env -i \
    PATH="$PYTHON_BIN_DIR:/usr/local/bin:/usr/bin:/bin" \
    HOME="$STAGE_ROOT/home" \
    PYTHONPATH="$REPO_ROOT:$REPO_ROOT/datasets/reasoning_eval/python_packages:$REPO_ROOT/.venv/lib/python3.12/site-packages:$REPO_ROOT/datasets/reasoning_eval/scoring_python_packages" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HUMANEVAL_OVERRIDE_PATH="$STAGE_ROOT/HumanEvalPlus.jsonl.gz" \
    REALQ_ALLOW_UNTRUSTED_CODE=1 \
    "$NO_INET" \
    "$PYTHON_BIN" -u "$ENTRYPOINT" \
      "$STAGE_ROOT/evalplus_samples.jsonl" "$PARALLEL" \
  >"$RUN_ROOT/official_eval_canoe.log" 2>&1
rc=$?
set -e

shopt -s nullglob
for result in "$STAGE_ROOT"/evalplus_samples*.json; do
  install -m 0644 "$result" "$RUN_ROOT/$(basename "$result")"
done
printf 'stage_root=%s\nexit_code=%s\n' "$STAGE_ROOT" "$rc" \
  >"$RUN_ROOT/canoe_run_status.txt"
exit "$rc"
