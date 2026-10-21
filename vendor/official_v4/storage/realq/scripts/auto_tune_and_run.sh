#!/usr/bin/env bash
# Tune REAL-Q's grad_lr on partial quantization, then run the formal profile.
#
# Usage:
#   bash realq/scripts/auto_tune_and_run.sh MODEL [W_BITS] [CUDA_IDS] [-- EXTRA_REALQ_ARGS...]
#
# Example (8 concurrent one-GPU tuning arms, then an 8-GPU formal run):
#   bash realq/scripts/auto_tune_and_run.sh \
#     models/Qwen/Qwen3-4B 3 0,1,2,3,4,5,6,7
#
# Environment overrides:
#   PARALLELISM, MAIN_CUDA_IDS, OUTPUT_ROOT, CACHE_ROOT,
#   TUNING_W_GROUPSIZE, MAIN_W_GROUPSIZE,
#   A_BITS, K_BITS, V_BITS, PYTHON

set -euo pipefail

REPO_ROOT=${REALQ_WORKDIR:-${PWD}}
MODEL=${1:?usage: auto_tune_and_run.sh MODEL [W_BITS] [CUDA_IDS] [-- EXTRA_REALQ_ARGS...]}
W_BITS=${2:-4}
CUDA_IDS=${3:-0}
shift $(( $# >= 3 ? 3 : $# ))
if [[ ${1:-} == -- ]]; then
  shift
fi

PYTHON=${PYTHON:-$(command -v python)}
if [[ ! -x ${PYTHON} ]]; then
  echo "repository venv python is not executable: ${PYTHON}" >&2
  exit 2
fi

GPU_COUNT=$(awk -F',' '{print NF}' <<<"${CUDA_IDS}")
PARALLELISM=${PARALLELISM:-${GPU_COUNT}}
MAIN_CUDA_IDS=${MAIN_CUDA_IDS:-${CUDA_IDS}}
MODEL_NAME=$(basename "${MODEL%/}")
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/output/realq_auto_tune/${MODEL_NAME}_w${W_BITS}}
CACHE_ROOT=${CACHE_ROOT:-${OUTPUT_ROOT}/cache}

# These are the normal/full experiment values from 调参方法.md. Additional
# arguments supplied by the caller come last and therefore may specialize the
# formal experiment. The Python tuner independently replaces every tuning-only
# value with its frozen fast profile.
exec "${PYTHON}" -m realq.utils.auto_tune \
  --model "${MODEL}" \
  --cuda-ids "${CUDA_IDS}" \
  --parallelism "${PARALLELISM}" \
  --tuning-w-groupsize "${TUNING_W_GROUPSIZE:--1}" \
  --main-cuda-ids "${MAIN_CUDA_IDS}" \
  --output-root "${OUTPUT_ROOT}" \
  --cache-root "${CACHE_ROOT}" \
  --python "${PYTHON}" \
  --run-main \
  -- \
  --w_bits "${W_BITS}" \
  --w_groupsize "${MAIN_W_GROUPSIZE:-128}" \
  --a_bits "${A_BITS:-16}" \
  --k_bits "${K_BITS:-16}" \
  --v_bits "${V_BITS:-16}" \
  --nsamples 256 \
  --seq_len 2048 \
  --bsz 128 \
  --global_loss_bsz 32 \
  --backward_samples 32 \
  --backward_bsz 32 \
  --blocksize 128 \
  --loss_slide_window true \
  --lm_eval true \
  "$@"
