#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/gptq_plus_realq
PYTHON=/home/uky/.venvs/realq-experiment/bin/python
MODEL=/mnt/d/llamaModels/Qwen3-0.6B

for seed in 42 43 44; do
  tag="b6-medium-all28-64x256-e64x512-seed${seed}-20261007"
  campaign="outputs/realq-adam-history-${tag}"
  if [[ -e "$campaign" ]]; then
    echo "ERROR: campaign already exists: $campaign"
    exit 2
  fi
  echo "START seed=$seed tag=$tag time=$(date -Is)"
  "$PYTHON" -u tools/run_realq_adam_history_ablation.py \
    --model "$MODEL" \
    --tag "$tag" \
    --stop-layer 27 \
    --nsamples 64 \
    --seq-len 256 \
    --eval-seq-len 512 \
    --eval-samples 64 \
    --seed "$seed" \
    --refresh-seed 0 \
    --timeout-seconds 14400 \
    --candidate-policy reset_m_and_m_age_each_refresh \
    --candidate-layers all \
    --discard-checkpoints-after-hash
  echo "DONE seed=$seed time=$(date -Is)"
done

echo "ALL_SEEDS_DONE time=$(date -Is)"
