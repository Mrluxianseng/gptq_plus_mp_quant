#!/bin/bash

set -u

ROOT_DIR=${ROOT_DIR:-/root/lh/gptq_plus}
MODEL_ROOT=${MODEL_ROOT:-/root/lh/llmModels}
OUT_DIR=${OUT_DIR:-${ROOT_DIR}/outputs}
CACHE_DIR=${CACHE_DIR:-${ROOT_DIR}/cache}
DEVICE=${DEVICE:-0}
NUM_GROUPS=${NUM_GROUPS:-4}

if [[ ! -d "${ROOT_DIR}" ]]; then
    echo "ROOT_DIR does not exist: ${ROOT_DIR}"
    exit 1
fi

if [[ ! -d "${MODEL_ROOT}" ]]; then
    echo "MODEL_ROOT does not exist: ${MODEL_ROOT}"
    exit 1
fi

mkdir -p "${OUT_DIR}" "${CACHE_DIR}"

MODEL_PATH="${MODEL_ROOT}/Qwen3-0.6B"

cd "${ROOT_DIR}" || exit 1

CUDA_VISIBLE_DEVICES="${DEVICE}" python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=localhost:2940"${DEVICE}" \
    ./ptq.py \
    --model "${MODEL_PATH}" \
    --exp "qwen3_0p6b_guided_quarot_w4_g${NUM_GROUPS}_s512_t2048" \
    --output_dir "${OUT_DIR}" \
    --cache_dir "${CACHE_DIR}" \
    --dataset wikitext2 \
    --nsamples 512 \
    --seq_len 2048 \
    --eval_seq_len 2048 \
    --seed 42 \
    --w_method gptq_guided \
    --w_bits 4 \
    --w_clip \
    --num_groups "${NUM_GROUPS}" \
    --act_order \
    --eval_datasets wikitext2 ultrachat_2k numinamath \
    --lm_eval \
    --lm_eval_batch_size 32 \
    --rotate
