#!/bin/bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <MODEL_PATH> [extra analyze_quant_profile.py args ...]"
    echo "Example: TARGET_LAYERS=0,14,27 $0 ./modelzoo/Qwen3/Qwen3-0.6B"
    exit 1
fi

MODEL_PATH=${1}
shift

N_SAMPLES=${N_SAMPLES:-512}
SEQ_LEN=${SEQ_LEN:-1024}
NUM_GROUPS=${NUM_GROUPS:-4}
BSZ=${BSZ:-4}
ALPHA=${ALPHA:-0.05}
BLOCKSIZE=${BLOCKSIZE:-256}
BLOCK_ATOMIC_QUANT=${BLOCK_ATOMIC_QUANT:-1}
BACKWARD_SAMPLES=${BACKWARD_SAMPLES:-32}
BACKWARD_BSZ=${BACKWARD_BSZ:-4}
FINAL_LAYER_BACKWARD_BSZ=${FINAL_LAYER_BACKWARD_BSZ:-${BACKWARD_BSZ}}
GRAD_LR=${GRAD_LR:-0.00005}
SECOND_ORDER_SCALE=${SECOND_ORDER_SCALE:-1.0}
G_UPDATE_MODE=${G_UPDATE_MODE:-block_gd}
TARGET_LAYERS=${TARGET_LAYERS:-0,1}
TARGET_MODULES=${TARGET_MODULES:-all}
QUANT_STOP_LAYER=${QUANT_STOP_LAYER:-1}
EXP_NAME=${EXP_NAME:-quant_profile_quick}
NSYS=${NSYS:-1}
NSYS_OUTPUT=${NSYS_OUTPUT:-./outputs/nsight/${EXP_NAME}}
NSYS_TRACE=${NSYS_TRACE:-cuda,nvtx}
NSYS_WAIT=${NSYS_WAIT:-primary}
if [[ -z "${NSYS_BIN:-}" ]]; then
    if [[ -x /usr/local/bin/nsys ]]; then
        NSYS_BIN=/usr/local/bin/nsys
    else
        NSYS_BIN=$(command -v nsys || true)
    fi
fi
GRAD_OPT=${GRAD_OPT:-adam}
FINAL_LAYER_GRAD_LR=${FINAL_LAYER_GRAD_LR:-0.3}
FINAL_LAYER_GRAD_OPT=${FINAL_LAYER_GRAD_OPT:-sgd}
GRAD_REFRESH_LOSS=${GRAD_REFRESH_LOSS:-fisher_diag_mse}
GRAD_REG_STRATEGY=${GRAD_REG_STRATEGY:-quant_error_gate_optimized}
GRAD_REG_LAMBDA=${GRAD_REG_LAMBDA:-0.0}
GRAD_GATE_FLOOR=${GRAD_GATE_FLOOR:-0.01}
GRAD_GATE_SHARPNESS=${GRAD_GATE_SHARPNESS:-5.0}
GRAD_GATE_SINE_AMP=${GRAD_GATE_SINE_AMP:-0.0005}
GRAD_HESSIAN_TOPK=${GRAD_HESSIAN_TOPK:-20}
PROJ_LR_SCALE=${PROJ_LR_SCALE:-1.0}
DOWN_PROJ_LR_SCALE=${DOWN_PROJ_LR_SCALE:-1.0}
GLOBAL_LOSS=${GLOBAL_LOSS:-0}
GLOBAL_LOSS_BSZ=${GLOBAL_LOSS_BSZ:-${BSZ}}

export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/tmp/hf_datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-/tmp/hf_hub}

CMD=(
python analyze_quant_profile.py \
    --model "${MODEL_PATH}" \
    --exp "${EXP_NAME}" \
    --dataset neuralmagic \
    --nsamples "${N_SAMPLES}" \
    --seq_len "${SEQ_LEN}" \
    --num_groups "${NUM_GROUPS}" \
    --bsz "${BSZ}" \
    --alpha "${ALPHA}" \
    --blocksize "${BLOCKSIZE}" \
    --backward_samples "${BACKWARD_SAMPLES}" \
    --backward_bsz "${BACKWARD_BSZ}" \
    --final_layer_backward_bsz "${FINAL_LAYER_BACKWARD_BSZ}" \
    --g_update_mode "${G_UPDATE_MODE}" \
    --grad_lr "${GRAD_LR}" \
    --grad_optimizer "${GRAD_OPT}" \
    --grad_refresh_loss "${GRAD_REFRESH_LOSS}" \
    --proj_lr_scale "${PROJ_LR_SCALE}" \
    --down_proj_lr_scale "${DOWN_PROJ_LR_SCALE}" \
    --grad_reg_strategy "${GRAD_REG_STRATEGY}" \
    --grad_reg_lambda "${GRAD_REG_LAMBDA}" \
    --grad_gate_floor "${GRAD_GATE_FLOOR}" \
    --grad_gate_sharpness "${GRAD_GATE_SHARPNESS}" \
    --grad_gate_sine_amp "${GRAD_GATE_SINE_AMP}" \
    --grad_hessian_topk "${GRAD_HESSIAN_TOPK}" \
    --second_order_scale "${SECOND_ORDER_SCALE}" \
    --quant_profile_target_layers "${TARGET_LAYERS}" \
    --quant_profile_target_modules "${TARGET_MODULES}" \
    --quant_stop_layer "${QUANT_STOP_LAYER}" \
    --act_order \
    --w_clip \
)

if [[ "${GLOBAL_LOSS}" == "1" ]]; then
    CMD+=(--global_loss --global_loss_bsz "${GLOBAL_LOSS_BSZ}")
fi

if [[ "${BLOCK_ATOMIC_QUANT}" == "1" ]]; then
    CMD+=(--block_atomic_quant)
fi

if [[ -n "${FINAL_LAYER_GRAD_LR}" ]]; then
    CMD+=(--final_layer_grad_lr "${FINAL_LAYER_GRAD_LR}")
fi

if [[ -n "${FINAL_LAYER_GRAD_OPT}" ]]; then
    CMD+=(--final_layer_grad_optimizer "${FINAL_LAYER_GRAD_OPT}")
fi

CMD+=("$@")

if [[ "${NSYS}" == "1" ]]; then
    mkdir -p "$(dirname "${NSYS_OUTPUT}")"
    if [[ -n "${NSYS_BIN}" && -x "${NSYS_BIN}" ]]; then
        "${NSYS_BIN}" profile \
            --force-overwrite=true \
            --trace="${NSYS_TRACE}" \
            --sample=none \
            --cpuctxsw=none \
            --backtrace=none \
            --python-sampling=false \
            --wait="${NSYS_WAIT}" \
            --capture-range=none \
            --output="${NSYS_OUTPUT}" \
            "${CMD[@]}"
    else
        echo "Warning: nsys not found; running with NVTX ranges enabled but without Nsight capture." >&2
        "${CMD[@]}"
    fi
else
    "${CMD[@]}"
fi
