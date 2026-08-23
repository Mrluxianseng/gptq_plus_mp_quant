#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 2 || ! $1 =~ ^[0-7]$ || ! $2 =~ ^(5|7)$ ]]; then
  echo "usage: $0 PHYSICAL_GPU {5|7}" >&2
  exit 64
fi
if [[ $(hostname) != j-zogxxxduju-master-0 ]]; then
  echo "clean-retime recovery must run on j-zogxxxduju-master-0" >&2
  exit 78
fi

physical_gpu=$1
run_group_gpu=$2
repo=/minimax-avatar-new/zhangqian/realq/gptq_plus
data=/minimax-avatar-new/zhangqian/realq/experiment_data
python=$repo/.venv.py312-broken-20260729/bin/python
venv=$repo/.venv.py312-broken-20260729
torch_lib=$venv/lib/python3.12/site-packages/torch/lib

echo "$(date -Iseconds) clean_retime_recovery_start physical_gpu=$physical_gpu run_group_gpu=$run_group_gpu"
env \
  CUDA_VISIBLE_DEVICES="$physical_gpu" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
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
  PYTHONPATH="$repo" \
  "$python" -u -m \
    experiments.turboboa_contended_clean_retime_20260821.worker \
    --physical-gpu "$physical_gpu" \
    --run-group-gpu "$run_group_gpu"
retime_rc=$?
echo "$(date -Iseconds) clean_retime_recovery_end physical_gpu=$physical_gpu run_group_gpu=$run_group_gpu rc=$retime_rc"

yaqa_log=$data/yaqa_wclip_fair20_20260821_v1/queue_logs/$(hostname)/gpu${physical_gpu}.after_turboboa_clean_retime_group${run_group_gpu}.log
if [[ -e $yaqa_log ]]; then
  echo "refusing existing YAQA relaunch log: $yaqa_log" >&2
  exit 73
fi
nohup setsid env \
  CUDA_VISIBLE_DEVICES="$physical_gpu" \
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
    --physical-gpu "$physical_gpu" \
  >"$yaqa_log" 2>&1 </dev/null &
echo "$(date -Iseconds) yaqa_relaunched gpu=$physical_gpu pid=$! log=$yaqa_log"

exit "$retime_rc"
