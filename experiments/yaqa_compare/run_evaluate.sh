#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 <1b|3b> <W4A16KV16|W3A16KV16|W2A4KV4> <hf-run-tag> <eval-run-tag> <cuda-visible-gpu> <lm-eval-batch-size>"
  exit 2
fi

MODEL_SIZE=$1
SETTING=$2
HF_RUN_TAG=$3
EVAL_RUN_TAG=$4
VISIBLE_GPU=$5
LM_EVAL_BATCH_SIZE=$6

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

HF_NAME=llama32_${MODEL_SIZE}_${SETTING_SLUG}_${HF_RUN_TAG}

cd "${WORKSPACE}"
export CUDA_VISIBLE_DEVICES="${VISIBLE_GPU}"
export PYTHONPATH="${WORKSPACE}:${WORKSPACE}/YAQA:${WORKSPACE}/YAQA/qtip-kernels"
export HF_HOME="${WORKSPACE}/datasets/lm_eval_hf_cache"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# lm-eval 0.4.4 calls `git describe` once per task for metadata.  The Canoe
# process user differs from the shared checkout owner, so keep this permission
# process-local instead of mutating the job's global Git configuration.
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="${WORKSPACE}"

exec "${WORKSPACE}/.venv/bin/python" \
  -m experiments.yaqa_compare.evaluate \
  --model "${MODEL_PATH}" \
  --setting "${SETTING}" \
  --hf-dir "${OUTPUT_ROOT}/hfized/${HF_NAME}" \
  --hf-validation "${OUTPUT_ROOT}/validation/${HF_NAME}_hf.json" \
  --output "${OUTPUT_ROOT}/results/${HF_NAME}_${EVAL_RUN_TAG}.json" \
  --lm-eval-batch-size "${LM_EVAL_BATCH_SIZE}"
