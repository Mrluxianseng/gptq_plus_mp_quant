#!/bin/bash

set -euo pipefail

# HF offline / timeout knobs. Datasets (piqa, winogrande, hellaswag, arc, lambada,
# ceval, wikitext) are pre-cached under ~/.cache/huggingface/datasets, so we go
# fully offline to avoid hf-mirror.com ReadTimeout during lm_eval.
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-180}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-600}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_TRUST_REMOTE_CODE=${HF_DATASETS_TRUST_REMOTE_CODE:-1}

# CUDA VMM expandable segments. Kills the cudaMalloc staircase we hit when
# down_proj's 485MB H matrix / refresh buffers can't reuse the smaller slabs
# cached from q/k/v/o/gate/up_proj. Supported on CUDA >= 11.3 + Ampere+; older
# setups just ignore it. Caller can override by exporting the env var first.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

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
GRAD_LRS_STR=${GRAD_LRS:-"0.0000001 0.000001 0.000005"}
DATASET=${DATASET:-wikitext2} # wikitext2 / neuralmagic / ultrachat_2k / numinamath
N_SAMPLES=${N_SAMPLES:-256}
SEQ_LEN=${SEQ_LEN:-2048}
BSZ=${BSZ:-32}
FINAL_LAYER_STATS_BSZ=${FINAL_LAYER_STATS_BSZ:-8}
HESSIAN_ACCUM_BSZ=${HESSIAN_ACCUM_BSZ:-32}
ENABLE_GPTQ_PLUS=${ENABLE_GPTQ_PLUS:-0}
BACKWARD_SAMPLES=${BACKWARD_SAMPLES:-8}
BACKWARD_BSZ=${BACKWARD_BSZ:-8}
FINAL_LAYER_BACKWARD_BSZ=${FINAL_LAYER_BACKWARD_BSZ:-8}
FINAL_LAYER_FULL_BACKWARD=${FINAL_LAYER_FULL_BACKWARD:-0}
BLOCKSIZE=${BLOCKSIZE:-512}
W_GROUPSIZE=${W_GROUPSIZE:--1}
ACT_ORDER=${ACT_ORDER:-1}
BLOCK_ATOMIC_QUANT=${BLOCK_ATOMIC_QUANT:-0}
# Activation / KV quantization knobs. A/V/K quantization is applied to the
# final quantized model when the bit-width is <16. The quant-aware flags also
# enable the corresponding fake-quant paths inside GPTQ+ student forwards.
A_BITS=${A_BITS:-16}
A_GROUPSIZE=${A_GROUPSIZE:--1}
A_ASYM=${A_ASYM:-0}
A_CLIP_RATIO=${A_CLIP_RATIO:-1.0}
A_LOSS_RATIO=${A_LOSS_RATIO:-1.0}
K_BITS=${K_BITS:-16}
K_GROUPSIZE=${K_GROUPSIZE:--1}
K_ASYM=${K_ASYM:-0}
K_CLIP_RATIO=${K_CLIP_RATIO:-1.0}
V_BITS=${V_BITS:-16}
V_GROUPSIZE=${V_GROUPSIZE:--1}
V_ASYM=${V_ASYM:-0}
V_CLIP_RATIO=${V_CLIP_RATIO:-1.0}
ACT_QUANT_AWARE_GPTQ=${ACT_QUANT_AWARE_GPTQ:-0}
K_CACHE_QUANT_AWARE_GPTQ=${K_CACHE_QUANT_AWARE_GPTQ:-0}
GRAD_OPTIMIZER=${GRAD_OPTIMIZER:-adam}
FINAL_LAYER_GRAD_OPTIMIZER=${FINAL_LAYER_GRAD_OPTIMIZER:-adam}
GRAD_CLIP=${GRAD_CLIP:-5e-5}
# Optional: clip threshold applied ONLY to the final transformer layer. The
# final layer's grads flow through lm_head + final norm and often blow up
# relative to earlier blocks. Empty / "none" → reuse GRAD_CLIP for every layer.
FINAL_LAYER_GRAD_CLIP=${FINAL_LAYER_GRAD_CLIP:-5e-4}
# --grad_refresh_loss {kl,hidden_mse,fisher_diag_mse,residual_kl,refined_residual_kl,refined_mse,refined_mix}
GRAD_REFRESH_LOSS=${GRAD_REFRESH_LOSS:-fisher_diag_mse}
# refined_residual_kl knobs (only used when GRAD_REFRESH_LOSS=refined_residual_kl)
REFINED_RKL_NUM_A=${REFINED_RKL_NUM_A:-1}
REFINED_RKL_DAMP=${REFINED_RKL_DAMP:-0.01}
# refined_mse knobs (only used when GRAD_REFRESH_LOSS=refined_mse). Per-layer
# random pool size (rank-local) used for end-to-end grad collection before
# that layer's quant loop opens. Must divide GLOBAL_LOSS_BSZ / world.
NUM_SAMPLES_FOR_REFINED_MSE=${NUM_SAMPLES_FOR_REFINED_MSE:-32}
# refined_mix knobs (only used when GRAD_REFRESH_LOSS=refined_mix). Front
# [0, SPLIT) layers use fisher_diag_mse, [SPLIT, N-1) use refined_residual_kl,
# and the final layer stays on kl. RKL_LR_RATIO multiplies GRAD_LR / PRE_GRAD_LR
# on the back half only (not swept). Empty SPLIT = default to N // 2 at runtime.
REFINED_MIX_SPLIT_LAYER=${REFINED_MIX_SPLIT_LAYER:-}
REFINED_MIX_RKL_LR_RATIO=${REFINED_MIX_RKL_LR_RATIO:-0.2}
# Dynamic saliency update knobs. When ENABLE_DYN_SAL=1, saliency is refreshed at
# each of the 4 module-group boundaries per layer using a rank-R low-rank
# decomposition of per-module end-to-end gradient matrices G = U·Σ·V^T captured
# in precompute. DYN_SAL_RANK is the low-rank dimension. See
# saliency_dynamic_update_design.md.
ENABLE_DYN_SAL=${ENABLE_DYN_SAL:-0}
DYN_SAL_RANK=${DYN_SAL_RANK:-16}
DYN_SAL_EVD_THRESH=${DYN_SAL_EVD_THRESH:-1e-6}
# Refresh cadence. `per_boundary` (default) = 4 refreshes per layer (qkv / o /
# up+gate / down entry). `per_layer` = 1 refresh per layer (at layer entry,
# weights still FP; captures only upstream drift, halves the current-state
# forwards per layer).
DYN_SAL_REFRESH_MODE=${DYN_SAL_REFRESH_MODE:-per_layer} # per_layer or per_boundary
FINAL_LAYER_GRAD_LR=${FINAL_LAYER_GRAD_LR:-0.000001}
PRE_GD_STEPS=${PRE_GD_STEPS:-10}
PRE_GRAD_LR=${PRE_GRAD_LR:-0.00003}
PRE_FINAL_LAYER_GRAD_LR=${PRE_FINAL_LAYER_GRAD_LR:-0.3}
PRE_GRAD_OPTIMIZER=${PRE_GRAD_OPTIMIZER:-adam}
PRE_FINAL_LAYER_GRAD_OPTIMIZER=${PRE_FINAL_LAYER_GRAD_OPTIMIZER:-sgd}
# H-Adam KFAC-curvature damping (only used when a pre-grad optimizer is h_adam).
# Empty -> fall back to the process_args default (0.1).
H_ADAM_CURVATURE_DAMPING=${H_ADAM_CURVATURE_DAMPING:-}
#--grad_reg_strategy {none,l2,hessian,quant_error_gate,quant_error_gate_optimized}
GRAD_REG_STRATEGY=${GRAD_REG_STRATEGY:-none}
GRAD_REG_LAMBDA=${GRAD_REG_LAMBDA:-50.0}
GRAD_GATE_FLOOR=${GRAD_GATE_FLOOR:-0.01}
GRAD_GATE_SHARPNESS=${GRAD_GATE_SHARPNESS:-5.0}
GRAD_GATE_SINE_AMP=${GRAD_GATE_SINE_AMP:-0.00005}
GRAD_HESSIAN_TOPK=${GRAD_HESSIAN_TOPK:-20}
SALIENCY_CLIP_PERCENTILE=${SALIENCY_CLIP_PERCENTILE:-0.99}
PROJ_LR_SCALE=${PROJ_LR_SCALE:-1.0}
DOWN_PROJ_LR_SCALE=${DOWN_PROJ_LR_SCALE:-1.0}
SECOND_ORDER_SCALE=${SECOND_ORDER_SCALE:-1.0}
PRE_CLIP=${PRE_CLIP:-0}
GLOBAL_LOSS=${GLOBAL_LOSS:-1}
GLOBAL_LOSS_BSZ=${GLOBAL_LOSS_BSZ:-8}
LOSS_SLIDE_WINDOW=${LOSS_SLIDE_WINDOW:-0}
DP_GLOBAL_SHUFFLE=${DP_GLOBAL_SHUFFLE:-1}
# Drop first ATTENTION_SINK_SIZE tokens from every loss (NLL/KL/MSE) when
# IGNORE_ATTENTION_SINK=1. Sink tokens still flow through forward / KV; only
# the loss values and the gradients flowing back through them are zeroed.
# Affects static saliency/Fisher precompute (cache key gets `_sink{N}` so it
# doesn't collide with non-sink runs).
IGNORE_ATTENTION_SINK=${IGNORE_ATTENTION_SINK:-0}
ATTENTION_SINK_SIZE=${ATTENTION_SINK_SIZE:-256}
# --grad_lr_layer_schedule {none, cosine, linear, sqrt}
GRAD_LR_LAYER_SCHEDULE=${GRAD_LR_LAYER_SCHEDULE:-cosine}
# Scheduled non-final layers start at this fraction of GRAD_LR / PRE_GRAD_LR.
GRAD_LR_LAYER_BASE_RATIO=${GRAD_LR_LAYER_BASE_RATIO:-0.01}
ALPHA=${ALPHA:-0.0}
KL_TOPK=${KL_TOPK:-20}
LM_EVAL_BATCH_SIZE=${LM_EVAL_BATCH_SIZE:-32}
ENABLE_QA_EVAL=${ENABLE_QA_EVAL:-0}
BASE_EXP=${BASE_EXP:-gptq_plus_lr_sweep}
OUTPUT_ROOT=${OUTPUT_ROOT:-./outputs}
# FSDP2 precompute: shard params + grads across ranks during
# `collect_static_end_to_end_saliency_and_fisher`. Needed for 70B where full
# model + grads won't fit on a single card. Two-stage workflow:
#   Stage 1: FSDP_PRECOMPUTE=1 EXIT_AFTER_PRECOMPUTE=1 STATIC_CACHE_PATH=...
#   Stage 2: STATIC_CACHE_PATH=<same>  (no FSDP, reads cache)
FSDP_PRECOMPUTE=${FSDP_PRECOMPUTE:-0}
FSDP_CPU_OFFLOAD=${FSDP_CPU_OFFLOAD:-0}
FSDP_META_INIT=${FSDP_META_INIT:-${FSDP_PRECOMPUTE}}
STATIC_CACHE_PATH=${STATIC_CACHE_PATH:-cache/qwen32}
EXIT_AFTER_PRECOMPUTE=${EXIT_AFTER_PRECOMPUTE:-0}

IFS=' ' read -r -a GRAD_LRS <<< "${GRAD_LRS_STR}"

export CUDA_VISIBLE_DEVICES=${DEVICE}
MODEL_NAME=$(basename "${MODEL_PATH}")

# HuggingFace datasets / hub configuration — lm_eval pulls benchmark datasets
# (arc, hellaswag, ...) from the Hub at eval time. On china-region hosts the
# default `huggingface.co` endpoint usually times out; fall back to the mirror.
# Override from the shell with your own endpoint / cache dirs if needed.
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HOME}/.cache/huggingface/hub}
# Prefer cached copies whenever available (avoid the online freshness check).
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}

# Infer number of ranks from DEVICE ("0" → 1, "0,1" → 2, "0,1,2,3" → 4).
# RDZV port decouples from DEVICE so the commas don't end up in the endpoint.
IFS=',' read -r -a _DEVICE_LIST <<< "${DEVICE}"
N_GPUS=${N_GPUS:-${#_DEVICE_LIST[@]}}
RDZV_PORT=${RDZV_PORT:-29500}

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

FINAL_LAYER_GRAD_CLIP_ARGS=()
if [[ -n "${FINAL_LAYER_GRAD_CLIP}" && "${FINAL_LAYER_GRAD_CLIP}" != "none" ]]; then
    FINAL_LAYER_GRAD_CLIP_ARGS=(--final_layer_grad_clip "${FINAL_LAYER_GRAD_CLIP}")
fi

H_ADAM_CURVATURE_DAMPING_ARGS=()
if [[ -n "${H_ADAM_CURVATURE_DAMPING}" ]]; then
    H_ADAM_CURVATURE_DAMPING_ARGS=(--h_adam_curvature_damping "${H_ADAM_CURVATURE_DAMPING}")
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

ATTENTION_SINK_ARGS=()
ATTENTION_SINK_TAG=""
if [[ "${IGNORE_ATTENTION_SINK}" == "1" ]]; then
    ATTENTION_SINK_ARGS=(--ignore_attention_sink --attention_sink_size "${ATTENTION_SINK_SIZE}")
    ATTENTION_SINK_TAG="_sink${ATTENTION_SINK_SIZE}"
fi

DP_GLOBAL_SHUFFLE_ARGS=()
DP_GLOBAL_SHUFFLE_TAG=""
if [[ "${DP_GLOBAL_SHUFFLE}" == "1" ]]; then
    DP_GLOBAL_SHUFFLE_ARGS=(--dp_global_shuffle)
    DP_GLOBAL_SHUFFLE_TAG="_gshuf"
fi

W_QUANT_ARGS=(--w_groupsize "${W_GROUPSIZE}")
W_QUANT_TAG=""
if [[ "${ACT_ORDER}" == "1" ]]; then
    W_QUANT_ARGS+=(--act_order)
else
    W_QUANT_TAG="${W_QUANT_TAG}_noactorder"
fi
if [[ "${W_GROUPSIZE}" != "-1" ]]; then
    W_QUANT_TAG="${W_QUANT_TAG}_wg$(sanitize_float "${W_GROUPSIZE}")"
fi

ACT_KV_QUANT_ARGS=(
    --a_bits "${A_BITS}" --a_groupsize "${A_GROUPSIZE}" --a_clip_ratio "${A_CLIP_RATIO}"
    --a_loss_ratio "${A_LOSS_RATIO}"
    --k_bits "${K_BITS}" --k_groupsize "${K_GROUPSIZE}" --k_clip_ratio "${K_CLIP_RATIO}"
    --v_bits "${V_BITS}" --v_groupsize "${V_GROUPSIZE}" --v_clip_ratio "${V_CLIP_RATIO}"
)
ACT_KV_QUANT_TAG=""
if [[ "${A_ASYM}" == "1" ]]; then
    ACT_KV_QUANT_ARGS+=(--a_asym)
fi
if [[ "${K_ASYM}" == "1" ]]; then
    ACT_KV_QUANT_ARGS+=(--k_asym)
fi
if [[ "${V_ASYM}" == "1" ]]; then
    ACT_KV_QUANT_ARGS+=(--v_asym)
fi
if [[ "${A_BITS}" != "16" ]]; then
    ACT_KV_QUANT_TAG="${ACT_KV_QUANT_TAG}_a${A_BITS}g$(sanitize_float "${A_GROUPSIZE}")"
    if [[ "${A_ASYM}" == "1" ]]; then
        ACT_KV_QUANT_TAG="${ACT_KV_QUANT_TAG}asym"
    fi
    if [[ "${A_CLIP_RATIO}" != "1.0" && "${A_CLIP_RATIO}" != "1" ]]; then
        ACT_KV_QUANT_TAG="${ACT_KV_QUANT_TAG}clip$(sanitize_float "${A_CLIP_RATIO}")"
    fi
fi
if [[ "${A_LOSS_RATIO}" != "1.0" && "${A_LOSS_RATIO}" != "1" ]]; then
    ACT_KV_QUANT_TAG="${ACT_KV_QUANT_TAG}_aloss$(sanitize_float "${A_LOSS_RATIO}")"
fi
if [[ "${K_BITS}" != "16" ]]; then
    ACT_KV_QUANT_TAG="${ACT_KV_QUANT_TAG}_k${K_BITS}g$(sanitize_float "${K_GROUPSIZE}")"
    if [[ "${K_ASYM}" == "1" ]]; then
        ACT_KV_QUANT_TAG="${ACT_KV_QUANT_TAG}asym"
    fi
    if [[ "${K_CLIP_RATIO}" != "1.0" && "${K_CLIP_RATIO}" != "1" ]]; then
        ACT_KV_QUANT_TAG="${ACT_KV_QUANT_TAG}clip$(sanitize_float "${K_CLIP_RATIO}")"
    fi
fi
if [[ "${V_BITS}" != "16" ]]; then
    ACT_KV_QUANT_TAG="${ACT_KV_QUANT_TAG}_v${V_BITS}g$(sanitize_float "${V_GROUPSIZE}")"
    if [[ "${V_ASYM}" == "1" ]]; then
        ACT_KV_QUANT_TAG="${ACT_KV_QUANT_TAG}asym"
    fi
    if [[ "${V_CLIP_RATIO}" != "1.0" && "${V_CLIP_RATIO}" != "1" ]]; then
        ACT_KV_QUANT_TAG="${ACT_KV_QUANT_TAG}clip$(sanitize_float "${V_CLIP_RATIO}")"
    fi
fi
if [[ "${ACT_QUANT_AWARE_GPTQ}" == "1" ]]; then
    ACT_KV_QUANT_ARGS+=(--act_quant_aware_gptq)
    ACT_KV_QUANT_TAG="${ACT_KV_QUANT_TAG}_aqaware"
fi
if [[ "${K_CACHE_QUANT_AWARE_GPTQ}" == "1" ]]; then
    ACT_KV_QUANT_ARGS+=(--k_cache_quant_aware_gptq)
    ACT_KV_QUANT_TAG="${ACT_KV_QUANT_TAG}_kqaware"
fi

GRAD_LR_LAYER_SCHEDULE_ARGS=()
GRAD_LR_LAYER_SCHEDULE_TAG=""
if [[ "${GRAD_LR_LAYER_SCHEDULE}" != "none" ]]; then
    GRAD_LR_LAYER_SCHEDULE_ARGS=(
        --grad_lr_layer_schedule "${GRAD_LR_LAYER_SCHEDULE}"
        --grad_lr_layer_base_ratio "${GRAD_LR_LAYER_BASE_RATIO}"
    )
    GRAD_LR_LAYER_SCHEDULE_TAG="_lrsched${GRAD_LR_LAYER_SCHEDULE}_base$(sanitize_float "${GRAD_LR_LAYER_BASE_RATIO}")"
fi

FSDP_ARGS=()
if [[ "${FSDP_PRECOMPUTE}" == "1" ]]; then
    FSDP_ARGS+=(--fsdp_precompute)
fi
if [[ "${FSDP_CPU_OFFLOAD}" == "1" ]]; then
    FSDP_ARGS+=(--fsdp_cpu_offload)
fi
if [[ "${EXIT_AFTER_PRECOMPUTE}" == "1" ]]; then
    FSDP_ARGS+=(--exit_after_precompute)
fi
if [[ "${FSDP_META_INIT}" == "1" ]]; then
    FSDP_ARGS+=(--fsdp_meta_init)
fi
if [[ -n "${STATIC_CACHE_PATH}" ]]; then
    FSDP_ARGS+=(--static_cache_path "${STATIC_CACHE_PATH}")
fi

MIX_ARGS=()
if [[ "${GRAD_REFRESH_LOSS}" == "refined_mix" ]]; then
    if [[ -n "${REFINED_MIX_SPLIT_LAYER}" ]]; then
        MIX_ARGS+=(--refined_mix_split_layer "${REFINED_MIX_SPLIT_LAYER}")
    fi
    MIX_ARGS+=(--refined_mix_rkl_lr_ratio "${REFINED_MIX_RKL_LR_RATIO}")
fi

# ---------------------------------------------------------------
# Auto two-stage when FSDP_PRECOMPUTE=1:
#   Stage 1 — FSDP precompute only (write saliency/fisher cache, exit).
#   Stage 2 — normal quantisation loop reads the cache; no FSDP.
# We split like this because in-process FSDP2 unwrap is fragile — a fresh
# process is the most reliable way to drop DTensor state cleanly.
# User can still manually control this by setting FSDP_PRECOMPUTE=0 and
# pointing STATIC_CACHE_PATH at a cache populated elsewhere.
# ---------------------------------------------------------------
if [[ "${FSDP_PRECOMPUTE}" == "1" && "${EXIT_AFTER_PRECOMPUTE}" != "1" ]]; then
    if [[ -z "${STATIC_CACHE_PATH}" ]]; then
        STATIC_CACHE_PATH="./cache/static_stats/${MODEL_NAME}_fsdp_auto"
        echo "[sweep] FSDP_PRECOMPUTE=1: auto-setting STATIC_CACHE_PATH=${STATIC_CACHE_PATH}"
    fi
    mkdir -p "${STATIC_CACHE_PATH}"

    PRECOMPUTE_CPU_OFFLOAD_ARG=()
    if [[ "${FSDP_CPU_OFFLOAD}" == "1" ]]; then
        PRECOMPUTE_CPU_OFFLOAD_ARG=(--fsdp_cpu_offload)
    fi
    PRECOMPUTE_META_INIT_ARG=()
    if [[ "${FSDP_META_INIT}" == "1" ]]; then
        PRECOMPUTE_META_INIT_ARG=(--fsdp_meta_init)
    fi

    echo "============================================================"
    echo "[sweep] Stage 1/2: FSDP precompute → ${STATIC_CACHE_PATH}"
    echo "  Passes that affect the cache key must match Stage 2:"
    echo "    model, nsamples=${N_SAMPLES}, seq_len=${SEQ_LEN},"
    echo "    grad_hessian_topk=${GRAD_HESSIAN_TOPK}, global_loss_bsz=${GLOBAL_LOSS_BSZ},"
    echo "    world_size=${N_GPUS}, rotate=1"
    echo "============================================================"
    python -m torch.distributed.run \
        --nnodes=1 --nproc_per_node=${N_GPUS} --rdzv_endpoint=localhost:${RDZV_PORT} ./ptq.py \
        --model "${MODEL_PATH}" \
        --exp "precompute_fsdp" \
        --dataset "${DATASET}" --nsamples "${N_SAMPLES}" --seq_len "${SEQ_LEN}" \
        --w_method gptq_plus --w_bits 4 --w_clip --num_groups "${NUM_GROUPS}" --blocksize "${BLOCKSIZE}" \
        "${W_QUANT_ARGS[@]}" \
        "${ACT_KV_QUANT_ARGS[@]}" \
        --kl_topk "${KL_TOPK}" --bsz "${BSZ}" --final_layer_stats_bsz "${FINAL_LAYER_STATS_BSZ}" --alpha "${ALPHA}" \
        --enable_gptq_plus "${ENABLE_GPTQ_PLUS}" \
        --g_update_mode block_gd --grad_refresh_loss "${GRAD_REFRESH_LOSS}" \
        --refined_rkl_num_A "${REFINED_RKL_NUM_A}" --refined_rkl_damp "${REFINED_RKL_DAMP}" \
        --num_samples_for_refined_mse "${NUM_SAMPLES_FOR_REFINED_MSE}" \
        --grad_hessian_topk "${GRAD_HESSIAN_TOPK}" \
        --saliency_clip_percentile "${SALIENCY_CLIP_PERCENTILE}" \
        --enable_dynamic_saliency "${ENABLE_DYN_SAL}" \
        --dyn_sal_rank "${DYN_SAL_RANK}" \
        --dyn_sal_evd_thresh "${DYN_SAL_EVD_THRESH}" \
        --dyn_sal_refresh_mode "${DYN_SAL_REFRESH_MODE}" \
        "${MIX_ARGS[@]}" \
        "${GLOBAL_LOSS_ARGS[@]}" \
        "${LOSS_SLIDE_WINDOW_ARGS[@]}" \
        "${DP_GLOBAL_SHUFFLE_ARGS[@]}" \
        "${ATTENTION_SINK_ARGS[@]}" \
        --rotate \
        --skip_eval \
        --fsdp_precompute --exit_after_precompute --static_cache_path "${STATIC_CACHE_PATH}" \
        "${PRECOMPUTE_CPU_OFFLOAD_ARG[@]}" \
        "${PRECOMPUTE_META_INIT_ARG[@]}" \
        "$@"

    # For the sweep loop below, drop FSDP flags (model is fresh each run) and
    # just point at the cache so each quantisation pass reads precomputed
    # saliency/fisher from disk.
    FSDP_ARGS=(--static_cache_path "${STATIC_CACHE_PATH}")
    echo "[sweep] Stage 2/2: sweeping quantisation LRs (FSDP off, reading cache)"
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
    refined_rkl_suffix=""
    if [[ "${GRAD_REFRESH_LOSS}" == "refined_residual_kl" ]]; then
        if [[ "${REFINED_RKL_NUM_A}" != "1" ]]; then
            refined_rkl_suffix="_numA${REFINED_RKL_NUM_A}"
        fi
        if [[ "${REFINED_RKL_DAMP}" != "0.01" ]]; then
            refined_rkl_suffix="${refined_rkl_suffix}_damp$(sanitize_float "${REFINED_RKL_DAMP}")"
        fi
    fi
    refined_mse_suffix=""
    if [[ "${GRAD_REFRESH_LOSS}" == "refined_mse" ]]; then
        if [[ "${NUM_SAMPLES_FOR_REFINED_MSE}" != "32" ]]; then
            refined_mse_suffix="_nRM${NUM_SAMPLES_FOR_REFINED_MSE}"
        fi
    fi
    refined_mix_suffix=""
    if [[ "${GRAD_REFRESH_LOSS}" == "refined_mix" ]]; then
        if [[ -n "${REFINED_MIX_SPLIT_LAYER}" ]]; then
            refined_mix_suffix="_split${REFINED_MIX_SPLIT_LAYER}"
        fi
        if [[ "${REFINED_MIX_RKL_LR_RATIO}" != "1.0" ]]; then
            refined_mix_suffix="${refined_mix_suffix}_rklr$(sanitize_float "${REFINED_MIX_RKL_LR_RATIO}")"
        fi
    fi
    grad_hessian_suffix=""
    if [[ "${GRAD_HESSIAN_TOPK}" != "-1" ]]; then
        grad_hessian_suffix="_ghtk${GRAD_HESSIAN_TOPK}"
    fi
    dyn_sal_suffix=""
    if [[ "${ENABLE_DYN_SAL}" == "1" ]]; then
        dyn_sal_suffix="_dynsalR${DYN_SAL_RANK}"
        if [[ "${DYN_SAL_EVD_THRESH}" != "1e-6" ]]; then
            dyn_sal_suffix="${dyn_sal_suffix}_evd$(sanitize_float "${DYN_SAL_EVD_THRESH}")"
        fi
        if [[ "${DYN_SAL_REFRESH_MODE}" == "per_layer" ]]; then
            dyn_sal_suffix="${dyn_sal_suffix}_reflayer"
        fi
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
    exp_name="${BASE_EXP}_block_gd_${GRAD_OPTIMIZER}${refresh_suffix}${refined_rkl_suffix}${refined_mse_suffix}${refined_mix_suffix}${reg_suffix}${grad_hessian_suffix}${dyn_sal_suffix}${W_QUANT_TAG}${ACT_KV_QUANT_TAG}_lr${grad_lr_tag}_fllr${final_layer_grad_lr_tag}_s${second_order_tag}${pre_gd_suffix}${PRE_CLIP_TAG}${BLOCK_ATOMIC_TAG}${FINAL_LAYER_FULL_BACKWARD_TAG}${GLOBAL_LOSS_TAG}${LOSS_SLIDE_WINDOW_TAG}${ATTENTION_SINK_TAG}${DP_GLOBAL_SHUFFLE_TAG}${GRAD_LR_LAYER_SCHEDULE_TAG}"

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
    echo "  flgclip: ${FINAL_LAYER_GRAD_CLIP:-<default>}"
    echo "  rloss  : ${GRAD_REFRESH_LOSS}"
    if [[ "${GRAD_REFRESH_LOSS}" == "refined_residual_kl" ]]; then
        echo "  rkl_NA : ${REFINED_RKL_NUM_A}"
        echo "  rkl_dmp: ${REFINED_RKL_DAMP}"
    fi
    if [[ "${GRAD_REFRESH_LOSS}" == "refined_mix" ]]; then
        echo "  mix_sp : ${REFINED_MIX_SPLIT_LAYER:-<N//2>}"
        echo "  mix_rlr: ${REFINED_MIX_RKL_LR_RATIO}"
    fi
    echo "  reg    : ${GRAD_REG_STRATEGY}"
    echo "  reg_l  : ${GRAD_REG_LAMBDA}"
    echo "  gate_f : ${GRAD_GATE_FLOOR}"
    echo "  gate_k : ${GRAD_GATE_SHARPNESS}"
    echo "  gate_a : ${GRAD_GATE_SINE_AMP}"
    echo "  gh_topk: ${GRAD_HESSIAN_TOPK}"
    echo "  w_quant: group=${W_GROUPSIZE} act_order=${ACT_ORDER}"
    echo "  a_quant: bits=${A_BITS} g=${A_GROUPSIZE} asym=${A_ASYM} clip=${A_CLIP_RATIO} loss_ratio=${A_LOSS_RATIO} aware=${ACT_QUANT_AWARE_GPTQ}"
    echo "  k_quant: bits=${K_BITS} g=${K_GROUPSIZE} asym=${K_ASYM} clip=${K_CLIP_RATIO} aware=${K_CACHE_QUANT_AWARE_GPTQ}"
    echo "  v_quant: bits=${V_BITS} g=${V_GROUPSIZE} asym=${V_ASYM} clip=${V_CLIP_RATIO}"
    echo "  proj_s : ${PROJ_LR_SCALE}"
    echo "  down_s : ${DOWN_PROJ_LR_SCALE}"
    echo "  preclip: ${PRE_CLIP}"
    echo "  global : ${GLOBAL_LOSS}"
    echo "  gl_bsz : ${GLOBAL_LOSS_BSZ}"
    echo "  slidew : ${LOSS_SLIDE_WINDOW}"
    echo "  gshuf  : ${DP_GLOBAL_SHUFFLE}"
    echo "  lrsch  : ${GRAD_LR_LAYER_SCHEDULE} base_ratio=${GRAD_LR_LAYER_BASE_RATIO}"
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
        --dataset "${DATASET}" --nsamples "${N_SAMPLES}" --seq_len "${SEQ_LEN}" \
        --w_method gptq_plus --w_bits 4 --w_clip --num_groups "${NUM_GROUPS}" \
        "${W_QUANT_ARGS[@]}" \
        "${ACT_KV_QUANT_ARGS[@]}" \
        --kl_topk "${KL_TOPK}" --bsz "${BSZ}" --final_layer_stats_bsz "${FINAL_LAYER_STATS_BSZ}" --alpha "${ALPHA}" --blocksize "${BLOCKSIZE}" \
        --enable_gptq_plus "${ENABLE_GPTQ_PLUS}" \
        ${HESSIAN_ACCUM_BSZ:+--hessian_accum_bsz "${HESSIAN_ACCUM_BSZ}"} \
        --backward_samples "${BACKWARD_SAMPLES}" --backward_bsz "${BACKWARD_BSZ}" --final_layer_backward_bsz "${FINAL_LAYER_BACKWARD_BSZ}" \
        --g_update_mode block_gd --grad_lr "${grad_lr}" --grad_optimizer "${GRAD_OPTIMIZER}" --grad_refresh_loss "${GRAD_REFRESH_LOSS}" \
        --refined_rkl_num_A "${REFINED_RKL_NUM_A}" --refined_rkl_damp "${REFINED_RKL_DAMP}" \
        --num_samples_for_refined_mse "${NUM_SAMPLES_FOR_REFINED_MSE}" \
        --enable_dynamic_saliency "${ENABLE_DYN_SAL}" \
        --dyn_sal_rank "${DYN_SAL_RANK}" \
        --dyn_sal_evd_thresh "${DYN_SAL_EVD_THRESH}" \
        --dyn_sal_refresh_mode "${DYN_SAL_REFRESH_MODE}" \
        "${MIX_ARGS[@]}" \
        --rotate \
        "${GLOBAL_LOSS_ARGS[@]}" \
        "${LOSS_SLIDE_WINDOW_ARGS[@]}" \
        "${ATTENTION_SINK_ARGS[@]}" \
        "${DP_GLOBAL_SHUFFLE_ARGS[@]}" \
        "${GRAD_LR_LAYER_SCHEDULE_ARGS[@]}" \
        "${FSDP_ARGS[@]}" \
        --final_layer_grad_optimizer "${FINAL_LAYER_GRAD_OPTIMIZER}" \
        --grad_clip "${GRAD_CLIP}" \
        "${FINAL_LAYER_GRAD_CLIP_ARGS[@]}" \
        --final_layer_grad_lr "${FINAL_LAYER_GRAD_LR}" \
        --grad_hessian_topk "${GRAD_HESSIAN_TOPK}" \
        --saliency_clip_percentile "${SALIENCY_CLIP_PERCENTILE}" \
        --pre_gd_steps "${PRE_GD_STEPS}" \
        --pre_grad_lr "${PRE_GRAD_LR}" \
        --pre_grad_optimizer "${PRE_GRAD_OPTIMIZER}" \
        "${H_ADAM_CURVATURE_DAMPING_ARGS[@]}" \
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
