#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <1b|3b> <W4A16KV16|W3A16KV16|W2A4KV4> <cuda-visible-list> <hessian-tag> <run-tag>"
  exit 2
fi

MODEL_SIZE=$1
SETTING=$2
VISIBLE_GPUS=$3
HESSIAN_TAG=$4
RUN_TAG=$5

for TAG in "${HESSIAN_TAG}" "${RUN_TAG}"; do
  if [[ ! "${TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "invalid tag: ${TAG}"
    exit 2
  fi
done

WORKSPACE=/minimax-avatar-new/zhangqian/realq/gptq_plus
OUTPUT_ROOT=${WORKSPACE}/output/yaqa_compare_20260725/j-8j1en3m0aq
TOKEN_PATH=${WORKSPACE}/cache/tokens/Llama-3.2-1B_wikitext2_train_n256_sl2048_seed1.pt
TOKEN_SHA=3090c0cae6c16fd8cd04b23f08e82a7be7d2cb4f6a7ceb995a50bc4472a9ee41
LUT_PATH=${WORKSPACE}/cache/yaqa/qtip_kmeans_9_2.pt
LUT_MANIFEST=${WORKSPACE}/cache/yaqa/qtip_kmeans_9_2.manifest.json

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
    K_BITS=4
    SCALE_OVERRIDE=1.0
    SETTING_SLUG=w4a16kv16
    HESSIAN_NAME=llama32_${MODEL_SIZE}_a16kv16_n256_${HESSIAN_TAG}
    AKV_ARGS=(
      --a_bits 16
      --k_bits 16
      --v_bits 16
      --akv_groupsize -1
      --akv_clip_ratio 1.0
    )
    ;;
  W3A16KV16)
    K_BITS=3
    SCALE_OVERRIDE=1.0
    SETTING_SLUG=w3a16kv16
    HESSIAN_NAME=llama32_${MODEL_SIZE}_a16kv16_n256_${HESSIAN_TAG}
    AKV_ARGS=(
      --a_bits 16
      --k_bits 16
      --v_bits 16
      --akv_groupsize -1
      --akv_clip_ratio 1.0
    )
    ;;
  W2A4KV4)
    K_BITS=2
    SCALE_OVERRIDE=0.9
    SETTING_SLUG=w2a4kv4_aware
    HESSIAN_NAME=llama32_${MODEL_SIZE}_a4kv4_aware_n256_${HESSIAN_TAG}
    AKV_ARGS=(
      --a_bits 4
      --k_bits 4
      --v_bits 4
      --akv_groupsize -1
      --akv_clip_ratio 0.9
      --akv_aware_hessian
    )
    ;;
  *)
    echo "unsupported setting: ${SETTING}"
    exit 2
    ;;
esac

HESSIAN_DIR=${OUTPUT_ROOT}/hessians/${HESSIAN_NAME}
VALIDATION_REPORT=${OUTPUT_ROOT}/validation/${HESSIAN_NAME}.json
SAVE_PATH=${OUTPUT_ROOT}/quantized/llama32_${MODEL_SIZE}_${SETTING_SLUG}_${RUN_TAG}

cd "${WORKSPACE}"
export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}"
export PYTHONPATH="${WORKSPACE}:${WORKSPACE}/YAQA:${WORKSPACE}/YAQA/qtip-kernels"
export PYTORCH_ALLOC_CONF=max_split_size_mb:512

"${WORKSPACE}/.venv/bin/python" \
  -m experiments.yaqa_compare.prepare_qtip_lut \
  --output "${LUT_PATH}" \
  --manifest "${LUT_MANIFEST}" \
  --seed 0

"${WORKSPACE}/.venv/bin/python" \
  -m experiments.yaqa_compare.preflight_quantize \
  --model "${MODEL_PATH}" \
  --setting "${SETTING}" \
  --hessian-dir "${HESSIAN_DIR}" \
  --validation-report "${VALIDATION_REPORT}" \
  --output-dir "${SAVE_PATH}" \
  --lut "${LUT_PATH}" \
  --lut-manifest "${LUT_MANIFEST}"

mkdir -p "${OUTPUT_ROOT}/quantized"
mkdir "${SAVE_PATH}"

exec "${WORKSPACE}/.venv/bin/python" \
  "${WORKSPACE}/YAQA/quantize_llama/quantize_finetune_llama.py" \
  --save_path "${SAVE_PATH}" \
  --hess_path "${HESSIAN_DIR}" \
  --base_model "${MODEL_PATH}" \
  --seed 0 \
  --num_cpu_threads 8 \
  --ctx_size 2048 \
  --sigma_reg 1e-2 \
  --scale_override "${SCALE_OVERRIDE}" \
  --codebook bitshift \
  --ft_epochs 0 \
  --td_x 16 \
  --td_y 16 \
  --L 16 \
  --K "${K_BITS}" \
  --V 2 \
  --tlut_bits 9 \
  --decode_mode quantlut_sym \
  --calib_tokens_path "${TOKEN_PATH}" \
  --calib_tokens_sha256 "${TOKEN_SHA}" \
  --calib_n_seqs 256 \
  "${AKV_ARGS[@]}"
