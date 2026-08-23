#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 1 || $# -gt 2 || ! $1 =~ ^[0-7]$ ]]; then
  echo "usage: $0 PHYSICAL_GPU [RUN_TAG]" >&2
  exit 64
fi
if [[ $(hostname) != j-zogxxxduju-master-0 ]]; then
  echo "opportunistic quality supervisor must run on j-zogxxxduju-master-0" >&2
  exit 78
fi

gpu=$1
run_tag=${2:-$(date -u +%Y%m%dT%H%M%SZ)}
if [[ ! $run_tag =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_TAG contains unsafe characters: $run_tag" >&2
  exit 64
fi
repo=/minimax-avatar-new/zhangqian/realq/gptq_plus
data=/minimax-avatar-new/zhangqian/realq/experiment_data
python=$repo/.venv.py312-broken-20260729/bin/python
venv=$repo/.venv.py312-broken-20260729
torch_lib=$venv/lib/python3.12/site-packages/torch/lib
eval_root=$data/additional_methods_fair20_eval_20260821_v1
evalplus_only=$eval_root/_runtime_dependencies/evalplus_only_0.3.1
eval_pythonpath=$repo:$repo/YAQA_wclip:$repo/YAQA_wclip/hessian_llama:$evalplus_only

echo "$(date -Iseconds) opportunistic_quality_start gpu=$gpu run_tag=$run_tag"
env \
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONPATH="$eval_pythonpath" \
  PYTHONHASHSEED=1234 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  HF_DATASETS_OFFLINE=1 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  VIRTUAL_ENV="$venv" \
  PATH="$venv/bin:$PATH" \
  LD_LIBRARY_PATH="$torch_lib:${LD_LIBRARY_PATH:-}" \
  "$python" -u -m \
    experiments.additional_methods_fair20_eval_20260821.opportunistic_worker \
    --physical-gpu "$gpu" --quality-only
quality_rc=$?
echo "$(date -Iseconds) opportunistic_quality_end gpu=$gpu rc=$quality_rc"

yaqa_log=$data/yaqa_wclip_fair20_20260821_v1/queue_logs/$(hostname)/gpu${gpu}.after_opportunistic_quality.${run_tag}.log
if [[ -e $yaqa_log ]]; then
  echo "refusing existing YAQA relaunch log: $yaqa_log" >&2
  exit 73
fi
nohup setsid env \
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
    --physical-gpu "$gpu" \
  >"$yaqa_log" 2>&1 </dev/null &
echo "$(date -Iseconds) yaqa_relaunched gpu=$gpu pid=$! log=$yaqa_log"

exit "$quality_rc"
