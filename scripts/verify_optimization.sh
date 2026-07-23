#!/bin/bash

# scripts/verify_optimization.sh
#
# Mirror of scripts/quant_profile_quick.sh for verify_gptq_plus.py.
# Use `save` to cache a golden baseline before optimization, and `check`
# after each optimization pass to make sure numerical output is unchanged.
#
# Usage:
#   bash scripts/verify_optimization.sh <MODEL_PATH> <save|check> [baseline_path]

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <MODEL_PATH> <save|check> [baseline_path]"
    exit 1
fi

MODEL_PATH=$1
MODE=$2
BASELINE=${3:-./outputs/verify/gptq_plus_baseline.pt}

# Keep defaults aligned with scripts/quant_profile_quick.sh.
N_SAMPLES=${N_SAMPLES:-128}
SEQ_LEN=${SEQ_LEN:-512}
NUM_GROUPS=${NUM_GROUPS:-4}
FISHER_NUM_GROUPS=${FISHER_NUM_GROUPS:-512}
BSZ=${BSZ:-32}
FINAL_LAYER_STATS_BSZ=${FINAL_LAYER_STATS_BSZ:-4}
ALPHA=${ALPHA:-0.03}
BLOCKSIZE=${BLOCKSIZE:-256}
BLOCK_ATOMIC_QUANT=${BLOCK_ATOMIC_QUANT:-0}
BACKWARD_SAMPLES=${BACKWARD_SAMPLES:-32}
BACKWARD_BSZ=${BACKWARD_BSZ:-32}
FINAL_LAYER_BACKWARD_BSZ=${FINAL_LAYER_BACKWARD_BSZ:-4}
FINAL_LAYER_FULL_BACKWARD=${FINAL_LAYER_FULL_BACKWARD:-0}
GRAD_LR=${GRAD_LR:-0.0001}
FINAL_LAYER_GRAD_LR=${FINAL_LAYER_GRAD_LR:-0.01}
SECOND_ORDER_SCALE=${SECOND_ORDER_SCALE:-1.0}
G_UPDATE_MODE=${G_UPDATE_MODE:-block_gd}
GRAD_OPTIMIZER=${GRAD_OPTIMIZER:-${GRAD_OPT:-adam}}
FINAL_LAYER_GRAD_OPTIMIZER=${FINAL_LAYER_GRAD_OPTIMIZER:-${FINAL_LAYER_GRAD_OPT:-sgd}}
GRAD_CLIP=${GRAD_CLIP:-1.0}
GRAD_REFRESH_LOSS=${GRAD_REFRESH_LOSS:-fisher_diag_mse}
PRE_GD_STEPS=${PRE_GD_STEPS:-10}
PRE_GRAD_LR=${PRE_GRAD_LR:-0.00003}
PRE_FINAL_LAYER_GRAD_LR=${PRE_FINAL_LAYER_GRAD_LR:-0.3}
PRE_GRAD_OPTIMIZER=${PRE_GRAD_OPTIMIZER:-adam}
PRE_FINAL_LAYER_GRAD_OPTIMIZER=${PRE_FINAL_LAYER_GRAD_OPTIMIZER:-sgd}
GRAD_REG_STRATEGY=${GRAD_REG_STRATEGY:-none}
GRAD_REG_LAMBDA=${GRAD_REG_LAMBDA:-0.01}
GRAD_GATE_FLOOR=${GRAD_GATE_FLOOR:-0.01}
GRAD_GATE_SHARPNESS=${GRAD_GATE_SHARPNESS:-5.0}
GRAD_GATE_SINE_AMP=${GRAD_GATE_SINE_AMP:-0.0005}
GRAD_HESSIAN_TOPK=${GRAD_HESSIAN_TOPK:--1}
PROJ_LR_SCALE=${PROJ_LR_SCALE:-1.0}
DOWN_PROJ_LR_SCALE=${DOWN_PROJ_LR_SCALE:-1.0}
PRE_CLIP=${PRE_CLIP:-0}
GLOBAL_LOSS=${GLOBAL_LOSS:-1}
GLOBAL_LOSS_BSZ=${GLOBAL_LOSS_BSZ:-4}
KL_TOPK=${KL_TOPK:--1}
CACHE_DIR=${CACHE_DIR:-./cache}
OUTPUT_ROOT=${OUTPUT_ROOT:-./outputs}
DEVICE=${DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}

QUANT_STOP_LAYER=${QUANT_STOP_LAYER:-1}
EXP_NAME=${EXP_NAME:-verify_optimization}
VERIFY_TOL_ABS=${VERIFY_TOL_ABS:-0.0}
VERIFY_TOL_REL=${VERIFY_TOL_REL:-0.0}

export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/tmp/hf_datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-/tmp/hf_hub}
export CUDA_VISIBLE_DEVICES=${DEVICE}

BLOCK_ATOMIC_ARGS=()
if [[ "${BLOCK_ATOMIC_QUANT}" == "1" ]]; then
    BLOCK_ATOMIC_ARGS=(--block_atomic_quant)
fi

FINAL_LAYER_FULL_BACKWARD_ARGS=()
if [[ "${FINAL_LAYER_FULL_BACKWARD}" == "1" ]]; then
    FINAL_LAYER_FULL_BACKWARD_ARGS=(--final_layer_full_backward)
fi

PRE_CLIP_ARGS=()
if [[ "${PRE_CLIP}" == "1" ]]; then
    PRE_CLIP_ARGS=(--pre_clip)
else
    PRE_CLIP_ARGS=(--no_pre_clip)
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
if [[ "${GLOBAL_LOSS}" == "1" ]]; then
    GLOBAL_LOSS_ARGS=(--global_loss --global_loss_bsz "${GLOBAL_LOSS_BSZ}")
else
    GLOBAL_LOSS_ARGS=(--no_global_loss)
fi

FINAL_LAYER_GRAD_LR_ARGS=()
if [[ -n "${FINAL_LAYER_GRAD_LR}" && "${FINAL_LAYER_GRAD_LR}" != "none" ]]; then
    FINAL_LAYER_GRAD_LR_ARGS=(--final_layer_grad_lr "${FINAL_LAYER_GRAD_LR}")
fi

FINAL_LAYER_GRAD_OPTIMIZER_ARGS=()
if [[ -n "${FINAL_LAYER_GRAD_OPTIMIZER}" && "${FINAL_LAYER_GRAD_OPTIMIZER}" != "none" ]]; then
    FINAL_LAYER_GRAD_OPTIMIZER_ARGS=(--final_layer_grad_optimizer "${FINAL_LAYER_GRAD_OPTIMIZER}")
fi

echo "============================================================"
echo "Running GPTQ+ verify (${MODE})"
echo "  model    : ${MODEL_PATH}"
echo "  baseline : ${BASELINE}"
echo "  device   : ${DEVICE}"
echo "  stop     : layer ${QUANT_STOP_LAYER}"
echo "============================================================"

python verify_gptq_plus.py \
    --verify_mode "${MODE}" \
    --verify_output "${BASELINE}" \
    --verify_tol_abs "${VERIFY_TOL_ABS}" \
    --verify_tol_rel "${VERIFY_TOL_REL}" \
    --model "${MODEL_PATH}" \
    --exp "${EXP_NAME}" \
    --output_dir "${OUTPUT_ROOT}" \
    --cache_dir "${CACHE_DIR}" \
    --dataset neuralmagic \
    --nsamples "${N_SAMPLES}" \
    --seq_len "${SEQ_LEN}" \
    --w_method gptq_plus \
    --w_bits 4 \
    --w_clip \
    --num_groups "${NUM_GROUPS}" \
    --fisher_num_groups "${FISHER_NUM_GROUPS}" \
    --kl_topk "${KL_TOPK}" \
    --bsz "${BSZ}" \
    --final_layer_stats_bsz "${FINAL_LAYER_STATS_BSZ}" \
    --alpha "${ALPHA}" \
    --blocksize "${BLOCKSIZE}" \
    --backward_samples "${BACKWARD_SAMPLES}" \
    --backward_bsz "${BACKWARD_BSZ}" \
    --final_layer_backward_bsz "${FINAL_LAYER_BACKWARD_BSZ}" \
    --g_update_mode "${G_UPDATE_MODE}" \
    --grad_lr "${GRAD_LR}" \
    --grad_optimizer "${GRAD_OPTIMIZER}" \
    --grad_refresh_loss "${GRAD_REFRESH_LOSS}" \
    "${GLOBAL_LOSS_ARGS[@]}" \
    "${FINAL_LAYER_GRAD_OPTIMIZER_ARGS[@]}" \
    --grad_clip "${GRAD_CLIP}" \
    "${FINAL_LAYER_GRAD_LR_ARGS[@]}" \
    --grad_hessian_topk "${GRAD_HESSIAN_TOPK}" \
    --pre_gd_steps "${PRE_GD_STEPS}" \
    --pre_grad_lr "${PRE_GRAD_LR}" \
    --pre_grad_optimizer "${PRE_GRAD_OPTIMIZER}" \
    "${PRE_FINAL_LAYER_GRAD_LR_ARGS[@]}" \
    "${PRE_FINAL_LAYER_GRAD_OPTIMIZER_ARGS[@]}" \
    "${PRE_CLIP_ARGS[@]}" \
    "${BLOCK_ATOMIC_ARGS[@]}" \
    "${FINAL_LAYER_FULL_BACKWARD_ARGS[@]}" \
    --proj_lr_scale "${PROJ_LR_SCALE}" \
    --down_proj_lr_scale "${DOWN_PROJ_LR_SCALE}" \
    --grad_reg_strategy "${GRAD_REG_STRATEGY}" \
    --grad_reg_lambda "${GRAD_REG_LAMBDA}" \
    --grad_gate_floor "${GRAD_GATE_FLOOR}" \
    --grad_gate_sharpness "${GRAD_GATE_SHARPNESS}" \
    --grad_gate_sine_amp "${GRAD_GATE_SINE_AMP}" \
    --second_order_scale "${SECOND_ORDER_SCALE}" \
    --quant_stop_layer "${QUANT_STOP_LAYER}" \
    --act_order
