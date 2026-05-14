#!/usr/bin/env bash
# RealQ launch template. Copy/adapt for sweeps.
#
# Usage:
#   bash realq/scripts/run.sh <model_path> [w_bits] [cuda_ids]
# Example:
#   bash realq/scripts/run.sh models/Qwen/Qwen3-0.6B 4 0,1,2,3
#
# Match the old gptq_plus_lr_sweep.sh defaults so sub-task validations stay
# comparable. Any flag you omit falls back to realq.config.Config defaults.

set -euo pipefail

MODEL=${1:-models/Qwen/Qwen3-0.6B}
W_BITS=${2:-4}
CUDA_IDS=${3:-0}

# Count GPUs from the comma-separated CUDA_IDS.
NUM_GPUS=$(echo "${CUDA_IDS}" | awk -F',' '{print NF}')

# Output dir mirrors the experiment name so multiple sweeps don't stomp each
# other's logs.
EXP=${EXP:-realq_$(basename "${MODEL}")_w${W_BITS}}
OUTPUT_ROOT=${OUTPUT_ROOT:-./output}

CUDA_VISIBLE_DEVICES=${CUDA_IDS} \
torchrun --nproc_per_node="${NUM_GPUS}" --master_port=$((29500 + RANDOM % 1000)) \
  -m realq.ptq \
  --model "${MODEL}" \
  --w_bits "${W_BITS}" \
  --num_groups "${NUM_GROUPS:-4}" \
  --grad_lr "${GRAD_LR:-3e-4}" \
  --nsamples "${NSAMPLES:-2048}" \
  --seq_len "${SEQ_LEN:-2048}" \
  --bsz "${BSZ:-64}" \
  --global_loss_bsz "${GLOBAL_LOSS_BSZ:-16}" \
  --backward_samples "${BACKWARD_SAMPLES:-32}" \
  --backward_bsz "${BACKWARD_BSZ:-32}" \
  --blocksize "${BLOCKSIZE:-128}" \
  --rotate "${ROTATE:-1}" \
  --seed "${SEED:-0}" \
  --output_dir "${OUTPUT_ROOT}" \
  --exp "${EXP}" \
  ${EXTRA_ARGS:-}
