#!/bin/bash

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <MODEL_PATH> <NUM_GROUPS> <DEVICE> [extra ptq.py args ...]"
    echo "Example: $0 ./modelzoo/Qwen3/Qwen3-0.6B 4 0 --lm_eval_batch_size 16"
    exit 1
fi

MODEL_PATH=${1}
NUM_GROUPS=${2}
DEVICE=${3}
shift 3

# Sweep configuration. Override from the shell when needed.
ALPHAS_STR=${ALPHAS:-"0.03 0.05 0.08 0.10 0.15 0.20"}
N_SAMPLES=${N_SAMPLES:-512}
SEQ_LEN=${SEQ_LEN:-1024}
BSZ=${BSZ:-4}
BACKWARD_SAMPLES=${BACKWARD_SAMPLES:-32}
BACKWARD_BSZ=${BACKWARD_BSZ:-4}
FINAL_LAYER_FULL_BACKWARD=${FINAL_LAYER_FULL_BACKWARD:-0}
BLOCKSIZE=${BLOCKSIZE:-256}
GRAD_LR=${GRAD_LR:-0.2}
FINAL_LAYER_GRAD_LR=${FINAL_LAYER_GRAD_LR:-${GRAD_LR}}
GRAD_OPTIMIZER=${GRAD_OPTIMIZER:-sgd}
FINAL_LAYER_GRAD_OPTIMIZER=${FINAL_LAYER_GRAD_OPTIMIZER:-${GRAD_OPTIMIZER}}
GRAD_CLIP=${GRAD_CLIP:-1.0}
GRAD_REFRESH_LOSS=${GRAD_REFRESH_LOSS:-kl}
GRAD_REG_STRATEGY=${GRAD_REG_STRATEGY:-none}
GRAD_REG_LAMBDA=${GRAD_REG_LAMBDA:-0.0}
GRAD_GATE_FLOOR=${GRAD_GATE_FLOOR:-0.1}
GRAD_GATE_SHARPNESS=${GRAD_GATE_SHARPNESS:-1.0}
GRAD_GATE_SINE_AMP=${GRAD_GATE_SINE_AMP:-0.0}
SECOND_ORDER_SCALE=${SECOND_ORDER_SCALE:-1.0}
KL_TOPK=${KL_TOPK:-20}
LM_EVAL_BATCH_SIZE=${LM_EVAL_BATCH_SIZE:-32}
ENABLE_QA_EVAL=${ENABLE_QA_EVAL:-0}
BASE_EXP=${BASE_EXP:-gptq_plus_alpha_sweep}
OUTPUT_ROOT=${OUTPUT_ROOT:-./outputs}

IFS=' ' read -r -a ALPHAS <<< "${ALPHAS_STR}"

export CUDA_VISIBLE_DEVICES=${DEVICE}
MODEL_NAME=$(basename "${MODEL_PATH}")

sanitize_float() {
    local value="${1}"
    value="${value//./p}"
    value="${value//-/m}"
    echo "${value}"
}

QA_EVAL_ARGS=()
if [[ "${ENABLE_QA_EVAL}" == "1" ]]; then
    QA_EVAL_ARGS=(--lm_eval --lm_eval_batch_size "${LM_EVAL_BATCH_SIZE}")
fi

FINAL_LAYER_FULL_BACKWARD_ARGS=()
FINAL_LAYER_FULL_BACKWARD_TAG=""
if [[ "${FINAL_LAYER_FULL_BACKWARD}" == "1" ]]; then
    FINAL_LAYER_FULL_BACKWARD_ARGS=(--final_layer_full_backward)
    FINAL_LAYER_FULL_BACKWARD_TAG="_flfb"
fi

for alpha in "${ALPHAS[@]}"; do
    alpha_tag=$(sanitize_float "${alpha}")
    final_layer_grad_lr_tag=$(sanitize_float "${FINAL_LAYER_GRAD_LR}")
    second_order_tag=$(sanitize_float "${SECOND_ORDER_SCALE}")
    reg_suffix=""
    if [[ "${GRAD_REG_STRATEGY}" != "none" ]]; then
        reg_suffix="_${GRAD_REG_STRATEGY}"
        if [[ "${GRAD_REG_STRATEGY}" == "l2" || "${GRAD_REG_STRATEGY}" == "hessian" ]]; then
            reg_suffix="${reg_suffix}_l$(sanitize_float "${GRAD_REG_LAMBDA}")"
        else
            reg_suffix="${reg_suffix}_f$(sanitize_float "${GRAD_GATE_FLOOR}")_k$(sanitize_float "${GRAD_GATE_SHARPNESS}")"
        fi
    fi
    refresh_suffix=""
    if [[ "${GRAD_REFRESH_LOSS}" != "kl" ]]; then
        refresh_suffix="_${GRAD_REFRESH_LOSS}"
    fi
    exp_name="${BASE_EXP}_block_gd_${GRAD_OPTIMIZER}${refresh_suffix}${reg_suffix}_a${alpha_tag}_lr$(sanitize_float "${GRAD_LR}")_fllr${final_layer_grad_lr_tag}_s${second_order_tag}${FINAL_LAYER_FULL_BACKWARD_TAG}"

    echo "============================================================"
    echo "Running GPTQ+ Alpha sweep"
    echo "  model  : ${MODEL_PATH}"
    echo "  groups : ${NUM_GROUPS}"
    echo "  mode   : block_gd"
    echo "  flfb   : ${FINAL_LAYER_FULL_BACKWARD}"
    echo "  opt    : ${GRAD_OPTIMIZER}"
    echo "  flopt  : ${FINAL_LAYER_GRAD_OPTIMIZER}"
    echo "  gclip  : ${GRAD_CLIP}"
    echo "  rloss  : ${GRAD_REFRESH_LOSS}"
    echo "  reg    : ${GRAD_REG_STRATEGY}"
    echo "  reg_l  : ${GRAD_REG_LAMBDA}"
    echo "  gate_f : ${GRAD_GATE_FLOOR}"
    echo "  gate_k : ${GRAD_GATE_SHARPNESS}"
    echo "  gate_a : ${GRAD_GATE_SINE_AMP}"
    echo "  alpha  : ${alpha}"
    echo "  gradlr : ${GRAD_LR}"
    echo "  fllr   : ${FINAL_LAYER_GRAD_LR}"
    echo "  so_scl : ${SECOND_ORDER_SCALE}"
    echo "  block  : ${BLOCKSIZE}"
    echo "  bsz    : ${BSZ}"
    echo "  bw_smp : ${BACKWARD_SAMPLES}"
    echo "  bw_bsz : ${BACKWARD_BSZ}"
    echo "  exp    : ${exp_name}"
    echo "============================================================"

    RUN_LOG_DIR="${OUTPUT_ROOT}/${MODEL_NAME}/${exp_name}/sweep_logs"
    mkdir -p "${RUN_LOG_DIR}"
    RUN_LOG_PATH="${RUN_LOG_DIR}/stdout.log"

    python -m torch.distributed.run \
        --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=localhost:2940${DEVICE} ./ptq.py \
        --model "${MODEL_PATH}" \
        --exp "${exp_name}" \
        --dataset neuralmagic --nsamples "${N_SAMPLES}" --seq_len "${SEQ_LEN}" \
        --w_method gptq_plus --w_bits 4 --w_clip --num_groups "${NUM_GROUPS}" --act_order \
        --kl_topk "${KL_TOPK}" --bsz "${BSZ}" --alpha "${alpha}" --blocksize "${BLOCKSIZE}" \
        --backward_samples "${BACKWARD_SAMPLES}" --backward_bsz "${BACKWARD_BSZ}" \
        --g_update_mode block_gd --grad_lr "${GRAD_LR}" --grad_optimizer "${GRAD_OPTIMIZER}" --grad_refresh_loss "${GRAD_REFRESH_LOSS}" \
        --final_layer_grad_optimizer "${FINAL_LAYER_GRAD_OPTIMIZER}" \
        --grad_clip "${GRAD_CLIP}" \
        --final_layer_grad_lr "${FINAL_LAYER_GRAD_LR}" \
        "${FINAL_LAYER_FULL_BACKWARD_ARGS[@]}" \
        --grad_reg_strategy "${GRAD_REG_STRATEGY}" --grad_reg_lambda "${GRAD_REG_LAMBDA}" \
        --grad_gate_floor "${GRAD_GATE_FLOOR}" --grad_gate_sharpness "${GRAD_GATE_SHARPNESS}" --grad_gate_sine_amp "${GRAD_GATE_SINE_AMP}" \
        --second_order_scale "${SECOND_ORDER_SCALE}" \
        "${QA_EVAL_ARGS[@]}" \
        "$@" 2>&1 | tee "${RUN_LOG_PATH}"
done
