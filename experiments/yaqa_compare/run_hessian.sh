#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 <1b|3b> <fp|akv4-aware> <cuda-visible-list> <nproc> [run-tag]"
  exit 2
fi

MODEL_SIZE=$1
AKV_MODE=$2
VISIBLE_GPUS=$3
NPROC=$4
RUN_TAG=${5:-formal}

if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid run tag: ${RUN_TAG}"
  exit 2
fi

WORKSPACE=/minimax-avatar-new/zhangqian/realq/gptq_plus
TOKEN_PATH=${WORKSPACE}/cache/tokens/Llama-3.2-1B_wikitext2_train_n256_sl2048_seed1.pt
TOKEN_SHA=3090c0cae6c16fd8cd04b23f08e82a7be7d2cb4f6a7ceb995a50bc4472a9ee41
OUTPUT_ROOT=${WORKSPACE}/output/yaqa_compare_20260725/j-8j1en3m0aq

case "${MODEL_SIZE}" in
  1b)
    MODEL_PATH=${WORKSPACE}/modelzoo/Llama/Llama-3.2-1B
    END_LAYER=16
    ;;
  3b)
    MODEL_PATH=${WORKSPACE}/modelzoo/Llama/Llama-3.2-3B
    END_LAYER=28
    ;;
  *)
    echo "unsupported model size: ${MODEL_SIZE}"
    exit 2
    ;;
esac

case "${AKV_MODE}" in
  fp)
    OUTPUT_DIR=${OUTPUT_ROOT}/hessians/llama32_${MODEL_SIZE}_a16kv16_n256_${RUN_TAG}
    AKV_ARGS=(
      --a_bits 16
      --k_bits 16
      --v_bits 16
      --akv_groupsize -1
      --akv_clip_ratio 1.0
    )
    ;;
  akv4-aware)
    OUTPUT_DIR=${OUTPUT_ROOT}/hessians/llama32_${MODEL_SIZE}_a4kv4_aware_n256_${RUN_TAG}
    AKV_ARGS=(
      --a_bits 4
      --k_bits 4
      --v_bits 4
      --akv_groupsize -1
      --akv_clip_ratio 0.9
      --akv_aware
    )
    ;;
  *)
    echo "unsupported A/K/V mode: ${AKV_MODE}"
    exit 2
    ;;
esac

mkdir -p "${OUTPUT_ROOT}/hessians"
mkdir "${OUTPUT_DIR}"

cd "${WORKSPACE}"
export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}"
export PYTHONPATH="${WORKSPACE}:${WORKSPACE}/YAQA:${WORKSPACE}/YAQA/hessian_llama"
export PYTORCH_ALLOC_CONF=max_split_size_mb:512

exec "${WORKSPACE}/.venv/bin/torchrun" \
  --standalone \
  --nproc-per-node="${NPROC}" \
  "${WORKSPACE}/YAQA/hessian_llama/get_hess_llama.py" \
  --orig_model "${MODEL_PATH}" \
  --save_path "${OUTPUT_DIR}" \
  --calib_tokens_path "${TOKEN_PATH}" \
  --calib_tokens_sha256 "${TOKEN_SHA}" \
  --seed 42 \
  --n_seqs 256 \
  --ctx_size 2048 \
  --batch_size 2 \
  --power_iters 1 \
  --hessian_sketch B \
  --start_layer 0 \
  --end_layer "${END_LAYER}" \
  "${AKV_ARGS[@]}"
