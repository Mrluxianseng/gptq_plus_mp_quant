#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <1b|3b> <W4A16KV16|W3A16KV16|W2A4KV4> <raw-run-tag> <hf-run-tag> <cuda-visible-gpu>"
  exit 2
fi

MODEL_SIZE=$1
SETTING=$2
RAW_RUN_TAG=$3
HF_RUN_TAG=$4
VISIBLE_GPU=$5

WORKSPACE=/minimax-avatar-new/zhangqian/realq/gptq_plus
OUTPUT_ROOT=${WORKSPACE}/output/yaqa_compare_20260725/j-8j1en3m0aq

case "${MODEL_SIZE}" in
  1b) MODEL_PATH=${WORKSPACE}/modelzoo/Llama/Llama-3.2-1B ;;
  3b) MODEL_PATH=${WORKSPACE}/modelzoo/Llama/Llama-3.2-3B ;;
  *) echo "unsupported model size: ${MODEL_SIZE}"; exit 2 ;;
esac

case "${SETTING}" in
  W4A16KV16) SETTING_SLUG=w4a16kv16 ;;
  W3A16KV16) SETTING_SLUG=w3a16kv16 ;;
  W2A4KV4) SETTING_SLUG=w2a4kv4_aware ;;
  *) echo "unsupported setting: ${SETTING}"; exit 2 ;;
esac

RAW_NAME=llama32_${MODEL_SIZE}_${SETTING_SLUG}_${RAW_RUN_TAG}
HF_NAME=llama32_${MODEL_SIZE}_${SETTING_SLUG}_${HF_RUN_TAG}

cd "${WORKSPACE}"
export CUDA_VISIBLE_DEVICES="${VISIBLE_GPU}"
export PYTHONPATH="${WORKSPACE}:${WORKSPACE}/YAQA:${WORKSPACE}/YAQA/qtip-kernels"

exec "${WORKSPACE}/.venv/bin/python" \
  -m experiments.yaqa_compare.validate_hfized \
  --model "${MODEL_PATH}" \
  --setting "${SETTING}" \
  --hf-dir "${OUTPUT_ROOT}/hfized/${HF_NAME}" \
  --raw-validation "${OUTPUT_ROOT}/validation/${RAW_NAME}.json" \
  --output "${OUTPUT_ROOT}/validation/${HF_NAME}_hf.json"
