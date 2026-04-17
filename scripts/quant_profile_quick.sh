#!/bin/bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <MODEL_PATH> [extra analyze_quant_profile.py args ...]"
    echo "Example: TARGET_LAYERS=0,14,27 NUM_GROUPS=4 DEVICE=0 $0 ./modelzoo/Qwen3/Qwen3-0.6B"
    exit 1
fi

MODEL_PATH=${1}
shift

# Quantization configuration. Keep these defaults aligned with gptq_plus_lr_sweep.sh
# unless the setting is profile-specific.
N_SAMPLES=${N_SAMPLES:-512}
SEQ_LEN=${SEQ_LEN:-1024}
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
GRAD_HESSIAN_TOPK=${GRAD_HESSIAN_TOPK:-20}
PROJ_LR_SCALE=${PROJ_LR_SCALE:-1.0}
DOWN_PROJ_LR_SCALE=${DOWN_PROJ_LR_SCALE:-1.0}
PRE_CLIP=${PRE_CLIP:-0}
GLOBAL_LOSS=${GLOBAL_LOSS:-1}
GLOBAL_LOSS_BSZ=${GLOBAL_LOSS_BSZ:-4}
KL_TOPK=${KL_TOPK:-20}
CACHE_DIR=${CACHE_DIR:-./cache}
OUTPUT_ROOT=${OUTPUT_ROOT:-./outputs}
DEVICE=${DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}

# Profile-only controls.
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
echo "Running GPTQ+ quant profile quick"
echo "  model  : ${MODEL_PATH}"
echo "  device : ${DEVICE}"
echo "  groups : ${NUM_GROUPS}"
echo "  fng    : ${FISHER_NUM_GROUPS}"
echo "  target : layers=${TARGET_LAYERS} modules=${TARGET_MODULES} stop=${QUANT_STOP_LAYER}"
echo "  nsys   : ${NSYS} output=${NSYS_OUTPUT}"
echo "  mode   : ${G_UPDATE_MODE}"
echo "  atomic : ${BLOCK_ATOMIC_QUANT}"
echo "  flfb   : ${FINAL_LAYER_FULL_BACKWARD}"
echo "  opt    : ${GRAD_OPTIMIZER}"
echo "  flopt  : ${FINAL_LAYER_GRAD_OPTIMIZER}"
echo "  gclip  : ${GRAD_CLIP}"
echo "  rloss  : ${GRAD_REFRESH_LOSS}"
echo "  global : ${GLOBAL_LOSS}"
echo "  gl_bsz : ${GLOBAL_LOSS_BSZ}"
echo "  preclip: ${PRE_CLIP}"
echo "  pregd  : ${PRE_GD_STEPS}"
echo "  prelr  : ${PRE_GRAD_LR}"
echo "  preopt : ${PRE_GRAD_OPTIMIZER}"
echo "  preflr : ${PRE_FINAL_LAYER_GRAD_LR:-<default>}"
echo "  prefo  : ${PRE_FINAL_LAYER_GRAD_OPTIMIZER:-<default>}"
echo "  gradlr : ${GRAD_LR}"
echo "  fllr   : ${FINAL_LAYER_GRAD_LR:-<default>}"
echo "  so_scl : ${SECOND_ORDER_SCALE}"
echo "  alpha  : ${ALPHA}"
echo "  block  : ${BLOCKSIZE}"
echo "  bsz    : ${BSZ}"
echo "  fl_stb : ${FINAL_LAYER_STATS_BSZ}"
echo "  bw_smp : ${BACKWARD_SAMPLES}"
echo "  bw_bsz : ${BACKWARD_BSZ}"
echo "  fl_bwb : ${FINAL_LAYER_BACKWARD_BSZ}"
echo "============================================================"

CMD=(
python analyze_quant_profile.py \
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
    --quant_profile_target_layers "${TARGET_LAYERS}" \
    --quant_profile_target_modules "${TARGET_MODULES}" \
    --quant_stop_layer "${QUANT_STOP_LAYER}" \
    --act_order \
)

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
