#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <1b|3b> <fp|akv4-aware> <run-tag>"
  exit 2
fi

MODEL_SIZE=$1
AKV_MODE=$2
RUN_TAG=$3

if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid run tag: ${RUN_TAG}"
  exit 2
fi

WORKSPACE=/minimax-avatar-new/zhangqian/realq/gptq_plus
OUTPUT_ROOT=${WORKSPACE}/output/yaqa_compare_20260725/j-8j1en3m0aq
TOKEN_PATH=${WORKSPACE}/cache/tokens/Llama-3.2-1B_wikitext2_train_n256_sl2048_seed1.pt
TOKEN_SHA=3090c0cae6c16fd8cd04b23f08e82a7be7d2cb4f6a7ceb995a50bc4472a9ee41

case "${MODEL_SIZE}" in
  1b)
    MODEL_PATH=${WORKSPACE}/modelzoo/Llama/Llama-3.2-1B
    ;;
  3b)
    MODEL_PATH=${WORKSPACE}/modelzoo/Llama/Llama-3.2-3B
    ;;
  *)
    echo "unsupported model size: ${MODEL_SIZE}"
    exit 2
    ;;
esac

case "${AKV_MODE}" in
  fp)
    HESSIAN_NAME=llama32_${MODEL_SIZE}_a16kv16_n256_${RUN_TAG}
    EXPECTED_AKV_MODE=n/a
    ;;
  akv4-aware)
    HESSIAN_NAME=llama32_${MODEL_SIZE}_a4kv4_aware_n256_${RUN_TAG}
    EXPECTED_AKV_MODE=aware
    ;;
  *)
    echo "unsupported A/K/V mode: ${AKV_MODE}"
    exit 2
    ;;
esac

cd "${WORKSPACE}"
export PYTHONPATH="${WORKSPACE}"

exec "${WORKSPACE}/.venv/bin/python" \
  -m experiments.yaqa_compare.validate_hessian \
  --model "${MODEL_PATH}" \
  --hessian-dir "${OUTPUT_ROOT}/hessians/${HESSIAN_NAME}" \
  --expected-samples 256 \
  --expected-seq-len 2048 \
  --expected-calib-path "${TOKEN_PATH}" \
  --expected-calib-sha256 "${TOKEN_SHA}" \
  --expected-world-size 4 \
  --expected-batch-size-per-rank 2 \
  --expected-akv-mode "${EXPECTED_AKV_MODE}" \
  --output "${OUTPUT_ROOT}/validation/${HESSIAN_NAME}.json"
