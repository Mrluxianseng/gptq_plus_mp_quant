#!/usr/bin/env bash
# RealQ launch template. Copy/adapt for sweeps.
#
# Usage:
#   bash realq/scripts/run.sh <model_path> [w_bits] [cuda_ids]
# Example:
#   bash realq/scripts/run.sh models/Qwen/Qwen3-0.6B 4 0,1,2,3
#
# This is a convenience template, not an exact paper-table launcher.  The
# paper does not currently disclose every numerical knob (notably gradient
# clipping), and its model/setting-specific learning rates cannot be expressed
# by one default.  See docs/REALQ_PAPER_PROTOCOL.md before reproducing tables.

set -euo pipefail

MODEL=${1:-models/Qwen/Qwen3-0.6B}
W_BITS=${2:-4}
CUDA_IDS=${3:-0}
A_BITS=${A_BITS:-16}
K_BITS=${K_BITS:-16}
V_BITS=${V_BITS:-16}

# Paper protocol: W4A16 is per-row; W2/W3 and every W*x*A4KV4 setting use
# group-128 weights. An explicit W_GROUPSIZE always wins.
if [[ -z ${W_GROUPSIZE+x} ]]; then
  if (( W_BITS < 4 || A_BITS < 16 || K_BITS < 16 || V_BITS < 16 )); then
    W_GROUPSIZE=128
  else
    W_GROUPSIZE=-1
  fi
fi

# Count GPUs from the comma-separated CUDA_IDS.
NUM_GPUS=$(echo "${CUDA_IDS}" | awk -F',' '{print NF}')

# Output dir mirrors the experiment name so multiple sweeps don't stomp each
# other's logs.
EXP=${EXP:-realq_$(basename "${MODEL}")_w${W_BITS}}
OUTPUT_ROOT=${OUTPUT_ROOT:-./output}

CUDA_VISIBLE_DEVICES=${CUDA_IDS} \
torchrun --nproc_per_node="${NUM_GPUS}" --master_port=$((29500 + RANDOM % 1000)) \
  -m realq_benchmark.ptq \
  --model "${MODEL}" \
  --w_bits "${W_BITS}" \
  --w_groupsize "${W_GROUPSIZE}" \
  --a_bits "${A_BITS}" \
  --k_bits "${K_BITS}" \
  --v_bits "${V_BITS}" \
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
  --seed "${SEED:-1}" \
  --output_dir "${OUTPUT_ROOT}" \
  --exp "${EXP}" \
  ${EXTRA_ARGS:-}
