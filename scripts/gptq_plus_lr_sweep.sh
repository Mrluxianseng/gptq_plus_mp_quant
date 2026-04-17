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
GRAD_LRS_STR=${GRAD_LRS:-"0.0001"}
N_SAMPLES=${N_SAMPLES:-512}
SEQ_LEN=${SEQ_LEN:-1024}
BSZ=${BSZ:-128}
FINAL_LAYER_STATS_BSZ=${FINAL_LAYER_STATS_BSZ:-16}
BACKWARD_SAMPLES=${BACKWARD_SAMPLES:-32}
BACKWARD_BSZ=${BACKWARD_BSZ:-32}
FINAL_LAYER_BACKWARD_BSZ=${FINAL_LAYER_BACKWARD_BSZ:-16}
FINAL_LAYER_FULL_BACKWARD=${FINAL_LAYER_FULL_BACKWARD:-0}
BLOCKSIZE=${BLOCKSIZE:-256}
BLOCK_ATOMIC_QUANT=${BLOCK_ATOMIC_QUANT:-0}
GRAD_OPTIMIZER=${GRAD_OPTIMIZER:-adam}
FINAL_LAYER_GRAD_OPTIMIZER=${FINAL_LAYER_GRAD_OPTIMIZER:-sgd}
GRAD_CLIP=${GRAD_CLIP:-1.0}
# --grad_refresh_loss {kl,hidden_mse,fisher_diag_mse}
GRAD_REFRESH_LOSS=${GRAD_REFRESH_LOSS:-fisher_diag_mse}
FINAL_LAYER_GRAD_LR=${FINAL_LAYER_GRAD_LR:-0.01}
PRE_GD_STEPS=${PRE_GD_STEPS:-10}
PRE_GRAD_LR=${PRE_GRAD_LR:-0.00003}
PRE_FINAL_LAYER_GRAD_LR=${PRE_FINAL_LAYER_GRAD_LR:-0.3}
PRE_GRAD_OPTIMIZER=${PRE_GRAD_OPTIMIZER:-adam}
PRE_FINAL_LAYER_GRAD_OPTIMIZER=${PRE_FINAL_LAYER_GRAD_OPTIMIZER:-sgd}
#--grad_reg_strategy {none,l2,hessian,quant_error_gate,quant_error_gate_optimized}
GRAD_REG_STRATEGY=${GRAD_REG_STRATEGY:-none}
GRAD_REG_LAMBDA=${GRAD_REG_LAMBDA:-0.01}
GRAD_GATE_FLOOR=${GRAD_GATE_FLOOR:-0.01}
GRAD_GATE_SHARPNESS=${GRAD_GATE_SHARPNESS:-5.0}
GRAD_GATE_SINE_AMP=${GRAD_GATE_SINE_AMP:-0.0005}
GRAD_HESSIAN_TOPK=${GRAD_HESSIAN_TOPK:-20}
PROJ_LR_SCALE=${PROJ_LR_SCALE:-1.0}
DOWN_PROJ_LR_SCALE=${DOWN_PROJ_LR_SCALE:-1.0}
SECOND_ORDER_SCALE=${SECOND_ORDER_SCALE:-1.0}
FISHER_NUM_GROUPS=${FISHER_NUM_GROUPS:-512}
PRE_CLIP=${PRE_CLIP:-0}
GLOBAL_LOSS=${GLOBAL_LOSS:-1}
GLOBAL_LOSS_BSZ=${GLOBAL_LOSS_BSZ:-16}
LOSS_SLIDE_WINDOW=${LOSS_SLIDE_WINDOW:-0}
DP_GLOBAL_SHUFFLE=${DP_GLOBAL_SHUFFLE:-0}
ALPHA=${ALPHA:-0.05}
KL_TOPK=${KL_TOPK:-20}
LM_EVAL_BATCH_SIZE=${LM_EVAL_BATCH_SIZE:-32}
ENABLE_QA_EVAL=${ENABLE_QA_EVAL:-0}
BASE_EXP=${BASE_EXP:-gptq_plus_lr_sweep}
OUTPUT_ROOT=${OUTPUT_ROOT:-./outputs}

IFS=' ' read -r -a GRAD_LRS <<< "${GRAD_LRS_STR}"

export CUDA_VISIBLE_DEVICES=${DEVICE}
MODEL_NAME=$(basename "${MODEL_PATH}")

# Infer number of ranks from DEVICE ("0" → 1, "0,1" → 2, "0,1,2,3" → 4).
# RDZV port decouples from DEVICE so the commas don't end up in the endpoint.
IFS=',' read -r -a _DEVICE_LIST <<< "${DEVICE}"
N_GPUS=${N_GPUS:-${#_DEVICE_LIST[@]}}
RDZV_PORT=${RDZV_PORT:-29400}

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

BLOCK_ATOMIC_ARGS=()
BLOCK_ATOMIC_TAG=""
if [[ "${BLOCK_ATOMIC_QUANT}" == "1" ]]; then
    BLOCK_ATOMIC_ARGS=(--block_atomic_quant)
    BLOCK_ATOMIC_TAG="_batomic"
fi

FINAL_LAYER_FULL_BACKWARD_ARGS=()
FINAL_LAYER_FULL_BACKWARD_TAG=""
if [[ "${FINAL_LAYER_FULL_BACKWARD}" == "1" ]]; then
    FINAL_LAYER_FULL_BACKWARD_ARGS=(--final_layer_full_backward)
    FINAL_LAYER_FULL_BACKWARD_TAG="_flfb"
fi

PRE_CLIP_ARGS=()
PRE_CLIP_TAG=""
if [[ "${PRE_CLIP}" == "1" ]]; then
    PRE_CLIP_ARGS=(--pre_clip)
else
    PRE_CLIP_ARGS=(--no_pre_clip)
    PRE_CLIP_TAG="_nopreclip"
fi

PRE_FINAL_LAYER_GRAD_LR_ARGS=()
if [[ -n "${PRE_FINAL_LAYER_GRAD_LR}" && "${PRE_FINAL_LAYER_GRAD_LR}" != "none" ]]; then
    PRE_FINAL_LAYER_GRAD_LR_ARGS=(--pre_final_layer_grad_lr "${PRE_FINAL_LAYER_GRAD_LR}")
fi

PRE_FINAL_LAYER_GRAD_OPTIMIZER_ARGS=()
if [[ -n "${PRE_FINAL_LAYER_GRAD_OPTIMIZER}" && "${PRE_FINAL_LAYER_GRAD_OPTIMIZER}" != "none" ]]; then
    PRE_FINAL_LAYER_GRAD_OPTIMIZER_ARGS=(--pre_final_layer_grad_optimizer "${PRE_FINAL_LAYER_GRAD_OPTIMIZER}")
fi

GLOBAL_LOSS_ARGS=()
GLOBAL_LOSS_TAG=""
if [[ "${GLOBAL_LOSS}" == "1" ]]; then
    GLOBAL_LOSS_ARGS=(--global_loss --global_loss_bsz "${GLOBAL_LOSS_BSZ}")
    GLOBAL_LOSS_TAG="_globalloss"
    if [[ "${GLOBAL_LOSS_BSZ}" != "${BSZ}" ]]; then
        GLOBAL_LOSS_TAG="${GLOBAL_LOSS_TAG}_bsz$(sanitize_float "${GLOBAL_LOSS_BSZ}")"
    fi
fi

LOSS_SLIDE_WINDOW_ARGS=()
LOSS_SLIDE_WINDOW_TAG=""
if [[ "${LOSS_SLIDE_WINDOW}" == "1" ]]; then
    LOSS_SLIDE_WINDOW_ARGS=(--loss_slide_window)
    LOSS_SLIDE_WINDOW_TAG="_slidewin"
fi

DP_GLOBAL_SHUFFLE_ARGS=()
DP_GLOBAL_SHUFFLE_TAG=""
if [[ "${DP_GLOBAL_SHUFFLE}" == "1" ]]; then
    DP_GLOBAL_SHUFFLE_ARGS=(--dp_global_shuffle)
    DP_GLOBAL_SHUFFLE_TAG="_gshuf"
fi

for grad_lr in "${GRAD_LRS[@]}"; do
    grad_lr_tag=$(sanitize_float "${grad_lr}")
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
    grad_hessian_suffix=""
    if [[ "${GRAD_HESSIAN_TOPK}" != "-1" ]]; then
        grad_hessian_suffix="_ghtk${GRAD_HESSIAN_TOPK}"
    fi
    fisher_groups_suffix=""
    if [[ "${FISHER_NUM_GROUPS}" != "${NUM_GROUPS}" ]]; then
        fisher_groups_suffix="_fng${FISHER_NUM_GROUPS}"
    fi
    pre_gd_suffix=""
    if [[ "${PRE_GD_STEPS}" != "0" ]]; then
        pre_gd_suffix="_pregd${PRE_GD_STEPS}_lr$(sanitize_float "${PRE_GRAD_LR}")_opt${PRE_GRAD_OPTIMIZER}"
        if [[ -n "${PRE_FINAL_LAYER_GRAD_LR}" && "${PRE_FINAL_LAYER_GRAD_LR}" != "none" ]]; then
            pre_gd_suffix="${pre_gd_suffix}_fllr$(sanitize_float "${PRE_FINAL_LAYER_GRAD_LR}")"
        fi
        if [[ -n "${PRE_FINAL_LAYER_GRAD_OPTIMIZER}" && "${PRE_FINAL_LAYER_GRAD_OPTIMIZER}" != "none" ]]; then
            pre_gd_suffix="${pre_gd_suffix}_flopt${PRE_FINAL_LAYER_GRAD_OPTIMIZER}"
        fi
    fi
    exp_name="${BASE_EXP}_block_gd_${GRAD_OPTIMIZER}${refresh_suffix}${reg_suffix}${grad_hessian_suffix}${fisher_groups_suffix}_lr${grad_lr_tag}_fllr${final_layer_grad_lr_tag}_s${second_order_tag}${pre_gd_suffix}${PRE_CLIP_TAG}${BLOCK_ATOMIC_TAG}${FINAL_LAYER_FULL_BACKWARD_TAG}${GLOBAL_LOSS_TAG}${LOSS_SLIDE_WINDOW_TAG}${DP_GLOBAL_SHUFFLE_TAG}"

    echo "============================================================"
    echo "Running GPTQ+ LR sweep"
    echo "  model  : ${MODEL_PATH}"
    echo "  groups : ${NUM_GROUPS}"
    echo "  mode   : block_gd"
    echo "  atomic : ${BLOCK_ATOMIC_QUANT}"
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
    echo "  gh_topk: ${GRAD_HESSIAN_TOPK}"
    echo "  fng    : ${FISHER_NUM_GROUPS}"
    echo "  proj_s : ${PROJ_LR_SCALE}"
    echo "  down_s : ${DOWN_PROJ_LR_SCALE}"
    echo "  preclip: ${PRE_CLIP}"
    echo "  global : ${GLOBAL_LOSS}"
    echo "  gl_bsz : ${GLOBAL_LOSS_BSZ}"
    echo "  slidew : ${LOSS_SLIDE_WINDOW}"
    echo "  gshuf  : ${DP_GLOBAL_SHUFFLE}"
    echo "  pregd  : ${PRE_GD_STEPS}"
    echo "  prelr  : ${PRE_GRAD_LR}"
    echo "  preopt : ${PRE_GRAD_OPTIMIZER}"
    echo "  preflr : ${PRE_FINAL_LAYER_GRAD_LR:-<default>}"
    echo "  prefo  : ${PRE_FINAL_LAYER_GRAD_OPTIMIZER:-<default>}"
    echo "  gradlr : ${grad_lr}"
    echo "  fllr   : ${FINAL_LAYER_GRAD_LR}"
    echo "  so_scl : ${SECOND_ORDER_SCALE}"
    echo "  block  : ${BLOCKSIZE}"
    echo "  bsz    : ${BSZ}"
    echo "  fl_stb : ${FINAL_LAYER_STATS_BSZ}"
    echo "  bw_smp : ${BACKWARD_SAMPLES}"
    echo "  bw_bsz : ${BACKWARD_BSZ}"
    echo "  fl_bwb : ${FINAL_LAYER_BACKWARD_BSZ}"
    echo "  exp    : ${exp_name}"
    echo "============================================================"

    RUN_LOG_DIR="${OUTPUT_ROOT}/${MODEL_NAME}/${exp_name}/sweep_logs"
    mkdir -p "${RUN_LOG_DIR}"
    RUN_LOG_PATH="${RUN_LOG_DIR}/stdout.log"

    python -m torch.distributed.run \
        --nnodes=1 --nproc_per_node=${N_GPUS} --rdzv_endpoint=localhost:${RDZV_PORT} ./ptq.py \
        --model "${MODEL_PATH}" \
        --exp "${exp_name}" \
        --dataset neuralmagic --nsamples "${N_SAMPLES}" --seq_len "${SEQ_LEN}" \
        --w_method gptq_plus --w_bits 4 --w_clip --num_groups "${NUM_GROUPS}" --fisher_num_groups "${FISHER_NUM_GROUPS}" --act_order \
        --kl_topk "${KL_TOPK}" --bsz "${BSZ}" --final_layer_stats_bsz "${FINAL_LAYER_STATS_BSZ}" --alpha "${ALPHA}" --blocksize "${BLOCKSIZE}" \
        --backward_samples "${BACKWARD_SAMPLES}" --backward_bsz "${BACKWARD_BSZ}" --final_layer_backward_bsz "${FINAL_LAYER_BACKWARD_BSZ}" \
        --g_update_mode block_gd --grad_lr "${grad_lr}" --grad_optimizer "${GRAD_OPTIMIZER}" --grad_refresh_loss "${GRAD_REFRESH_LOSS}" \
        "${GLOBAL_LOSS_ARGS[@]}" \
        "${LOSS_SLIDE_WINDOW_ARGS[@]}" \
        "${DP_GLOBAL_SHUFFLE_ARGS[@]}" \
        --final_layer_grad_optimizer "${FINAL_LAYER_GRAD_OPTIMIZER}" \
        --grad_clip "${GRAD_CLIP}" \
        --final_layer_grad_lr "${FINAL_LAYER_GRAD_LR}" \
        --grad_hessian_topk "${GRAD_HESSIAN_TOPK}" \
        --pre_gd_steps "${PRE_GD_STEPS}" \
        --pre_grad_lr "${PRE_GRAD_LR}" \
        --pre_grad_optimizer "${PRE_GRAD_OPTIMIZER}" \
        "${PRE_FINAL_LAYER_GRAD_LR_ARGS[@]}" \
        "${PRE_FINAL_LAYER_GRAD_OPTIMIZER_ARGS[@]}" \
        "${PRE_CLIP_ARGS[@]}" \
        "${BLOCK_ATOMIC_ARGS[@]}" \
        "${FINAL_LAYER_FULL_BACKWARD_ARGS[@]}" \
        --proj_lr_scale "${PROJ_LR_SCALE}" --down_proj_lr_scale "${DOWN_PROJ_LR_SCALE}" \
        --grad_reg_strategy "${GRAD_REG_STRATEGY}" --grad_reg_lambda "${GRAD_REG_LAMBDA}" \
        --grad_gate_floor "${GRAD_GATE_FLOOR}" --grad_gate_sharpness "${GRAD_GATE_SHARPNESS}" --grad_gate_sine_amp "${GRAD_GATE_SINE_AMP}" \
        --second_order_scale "${SECOND_ORDER_SCALE}" \
        "${QA_EVAL_ARGS[@]}" \
        "$@" 2>&1 | tee "${RUN_LOG_PATH}"
done
