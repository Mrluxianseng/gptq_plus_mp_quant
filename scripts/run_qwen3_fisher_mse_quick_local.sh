#!/bin/bash

set -u

# Quick local sanity-run config for a single RTX 4060 8GB-class GPU.
# Goal: finish quickly and avoid OOM, not reproduce full server-grade metrics.

ROOT_DIR=${ROOT_DIR:-/mnt/d/gptq_plus}
MODEL_ROOT=${MODEL_ROOT:-/mnt/d/llamaModels}
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

MODEL_NAME=${MODEL_NAME:-Qwen3-0.6B}
MODEL_PATH="${MODEL_ROOT}/${MODEL_NAME}"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH does not exist: ${MODEL_PATH}"
    exit 1
fi

cd "${ROOT_DIR}" || exit 1

OUTPUT_ROOT="${ROOT_DIR}/outputs" \
BASE_EXP="qwen3_0p6b_fisher_mse_quick_local" \
DATASET="wikitext2" \
N_SAMPLES=32 \
SEQ_LEN=256 \
ENABLE_DYN_SAL="0" \
BSZ=1 \
FINAL_LAYER_STATS_BSZ=1 \
HESSIAN_ACCUM_BSZ=4 \
ENABLE_GPTQ_PLUS="0" \
BACKWARD_SAMPLES=32 \
BACKWARD_BSZ=2 \
FINAL_LAYER_BACKWARD_BSZ=2 \
BLOCKSIZE=128 \
GRAD_OPTIMIZER="adam" \
FINAL_LAYER_GRAD_OPTIMIZER="adam" \
GRAD_CLIP="5e-5" \
FINAL_LAYER_GRAD_CLIP="5e-4" \
GRAD_REFRESH_LOSS="fisher_diag_mse" \
FINAL_LAYER_GRAD_LR="0.000001" \
PRE_GD_STEPS=0 \
PRE_GRAD_LR="0.00003" \
PRE_FINAL_LAYER_GRAD_LR="0.3" \
PRE_GRAD_OPTIMIZER="adam" \
PRE_FINAL_LAYER_GRAD_OPTIMIZER="sgd" \
GRAD_REG_STRATEGY="none" \
GLOBAL_LOSS=1 \
GLOBAL_LOSS_BSZ=1 \
LOSS_SLIDE_WINDOW=0 \
DP_GLOBAL_SHUFFLE=0 \
GRAD_LR_LAYER_SCHEDULE="none" \
ENABLE_QA_EVAL=0 \
LM_EVAL_BATCH_SIZE=8 \
GRAD_LRS="0.0002" \
bash "${ROOT_DIR}/scripts/gptq_plus_lr_sweep.sh" "${MODEL_PATH}" "${NUM_GROUPS}" "${DEVICE}" \
    --seed 42 \
    --skip_eval \
    --eval_seq_len 256
