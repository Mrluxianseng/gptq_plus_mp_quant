#!/usr/bin/env bash
set -uo pipefail

if [[ $# != 1 || ! $1 =~ ^[0-9]+$ ]]; then
  echo "usage: $0 ACTIVE_Q4_PAIR_PID" >&2
  exit 64
fi
if [[ $(hostname) != j-zogxxxduju-master-0 ]]; then
  echo "this continuation is pinned to j-zogxxxduju-master-0" >&2
  exit 78
fi

active_pair_pid=$1
repo=/minimax-avatar-new/zhangqian/realq/gptq_plus
python=$repo/.venv.py312-broken-20260729/bin/python
venv=$repo/.venv.py312-broken-20260729
torch_lib=$venv/lib/python3.12/site-packages/torch/lib
gpu=1

run_python_module() {
  env \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV="$venv" \
    PATH="$venv/bin:$PATH" \
    LD_LIBRARY_PATH="$torch_lib:${LD_LIBRARY_PATH:-}" \
    PYTHONPATH="$repo" \
    "$python" -u -m "$@"
}

while kill -0 "$active_pair_pid" 2>/dev/null; do
  sleep 15
done
echo "$(date -Iseconds) Q4 allopts pair finished pid=$active_pair_pid"

echo "$(date -Iseconds) starting current-code deterministic-SDPA cache diagnostic"
run_python_module \
  experiments.realq_allopts_loss_ablation_20260821.cache_backend_diagnostic \
  --physical-gpu "$gpu"
cache_rc=$?
echo "$(date -Iseconds) SDPA cache diagnostic exited rc=$cache_rc"

echo "$(date -Iseconds) starting REALQ allopts pair config=qwen3-8b_w4a4kv4"
run_python_module experiments.realq_allopts_loss_ablation_20260821.runner \
  --config qwen3-8b_w4a4kv4 --physical-gpu "$gpu"
pair_rc=$?
echo "$(date -Iseconds) REALQ allopts pair config=qwen3-8b_w4a4kv4 exited rc=$pair_rc"

echo "$(date -Iseconds) REALQ diagnostic queue drained; restoring YAQA lane"
env \
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONHASHSEED=1 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  HF_HUB_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  VIRTUAL_ENV="$venv" \
  PATH="$venv/bin:$PATH" \
  LD_LIBRARY_PATH="$torch_lib:${LD_LIBRARY_PATH:-}" \
  PYTHONPATH="$repo:$repo/YAQA_wclip:$repo/YAQA_wclip/hessian_llama" \
  "$python" -u -m experiments.yaqa_wclip_fair20_20260821.worker \
    --plan-file "$repo/experiments/yaqa_wclip_fair20_20260821/plan.json" \
    --expected-plan-sha256 a7bbd0db640b6d87edbb286f640d744fa13c66c2adccc102d16be9253145bda1 \
    --physical-gpu "$gpu"
yaqa_rc=$?
echo "$(date -Iseconds) YAQA lane exited rc=$yaqa_rc; restoring evaluation worker"

exec env \
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  HF_DATASETS_OFFLINE=1 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  VIRTUAL_ENV="$venv" \
  PATH="$venv/bin:$PATH" \
  LD_LIBRARY_PATH="$torch_lib:${LD_LIBRARY_PATH:-}" \
  PYTHONPATH="$repo:$repo/YAQA_wclip:$repo/YAQA_wclip/hessian_llama" \
  "$python" -u -m experiments.additional_methods_fair20_eval_20260821.worker \
    --physical-gpu "$gpu" --poll-seconds 30
