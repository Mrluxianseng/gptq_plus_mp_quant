#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <1b|3b> <W4A16KV16|W3A16KV16|W2A4KV4> <hessian-tag> <raw-run-tag> <hf-run-tag>"
  exit 2
fi

MODEL_SIZE=$1
SETTING=$2
HESSIAN_TAG=$3
RAW_RUN_TAG=$4
HF_RUN_TAG=$5

for TAG in "${HESSIAN_TAG}" "${RAW_RUN_TAG}" "${HF_RUN_TAG}"; do
  if [[ ! "${TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "invalid tag: ${TAG}"
    exit 2
  fi
done

WORKSPACE=/minimax-avatar-new/zhangqian/realq/gptq_plus
OUTPUT_ROOT=${WORKSPACE}/output/yaqa_compare_20260725/j-8j1en3m0aq
LUT_PATH=${WORKSPACE}/cache/yaqa/qtip_kmeans_9_2.pt

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

case "${SETTING}" in
  W4A16KV16)
    SETTING_SLUG=w4a16kv16
    HESSIAN_NAME=llama32_${MODEL_SIZE}_a16kv16_n256_${HESSIAN_TAG}
    ;;
  W3A16KV16)
    SETTING_SLUG=w3a16kv16
    HESSIAN_NAME=llama32_${MODEL_SIZE}_a16kv16_n256_${HESSIAN_TAG}
    ;;
  W2A4KV4)
    SETTING_SLUG=w2a4kv4_aware
    HESSIAN_NAME=llama32_${MODEL_SIZE}_a4kv4_aware_n256_${HESSIAN_TAG}
    ;;
  *)
    echo "unsupported setting: ${SETTING}"
    exit 2
    ;;
esac

RAW_NAME=llama32_${MODEL_SIZE}_${SETTING_SLUG}_${RAW_RUN_TAG}
RAW_DIR=${OUTPUT_ROOT}/quantized/${RAW_NAME}
RAW_VALIDATION=${OUTPUT_ROOT}/validation/${RAW_NAME}.json
HESSIAN_VALIDATION=${OUTPUT_ROOT}/validation/${HESSIAN_NAME}.json
HF_DIR=${OUTPUT_ROOT}/hfized/llama32_${MODEL_SIZE}_${SETTING_SLUG}_${HF_RUN_TAG}

cd "${WORKSPACE}"
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH="${WORKSPACE}:${WORKSPACE}/YAQA:${WORKSPACE}/YAQA/qtip-kernels"

"${WORKSPACE}/.venv/bin/python" \
  -m experiments.yaqa_compare.validate_quantized \
  --model "${MODEL_PATH}" \
  --setting "${SETTING}" \
  --raw-dir "${RAW_DIR}" \
  --hessian-validation "${HESSIAN_VALIDATION}" \
  --lut "${LUT_PATH}" \
  --output "${RAW_VALIDATION}"

mkdir -p "${OUTPUT_ROOT}/hfized"

"${WORKSPACE}/.venv/bin/python" \
  "${WORKSPACE}/YAQA/quantize_llama/hfize_llama.py" \
  --quantized_path "${RAW_DIR}" \
  --hf_output_path "${HF_DIR}"

# The shared workspace is read from both the Canoe process identity and the
# host identity.  save_pretrained may inherit a restrictive job umask, so make
# every published HF artifact cross-identity readable after a successful save.
find "${HF_DIR}" -maxdepth 1 -type f -exec chmod 0644 {} +
