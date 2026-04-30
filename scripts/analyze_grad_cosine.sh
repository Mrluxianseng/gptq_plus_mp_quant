#!/bin/bash

# Usage: bash scripts/analyze_grad_cosine.sh <MODEL_PATH> <DEVICE> [TARGET_LAYERS]
# Example: bash scripts/analyze_grad_cosine.sh ./modelzoo/Qwen3/Qwen3-0.6B 0 "1,5,10,15"

MODEL_PATH=${1}
DEVICE=${2}
TARGET_LAYERS=${3:-"5,10,15,20,25"}

MODEL_NAME=$(basename ${MODEL_PATH})
N_SAMPLES=256
SEQ_LEN=2048
MEASURE_SAMPLES=256
MEASURE_BSZ=4
BSZ=${BSZ:-32}
GLOBAL_LOSS_BSZ=${GLOBAL_LOSS_BSZ:-16}
BACKWARD_SAMPLES=${BACKWARD_SAMPLES:-32}
BACKWARD_BSZ=${BACKWARD_BSZ:-32}
BLOCKSIZE=${BLOCKSIZE:-128}
GRAD_LR=${GRAD_LR:-0.0002}
FINAL_LAYER_GRAD_LR=${FINAL_LAYER_GRAD_LR:-0.00001}
GRAD_OPTIMIZER=${GRAD_OPTIMIZER:-adam}
FINAL_LAYER_GRAD_OPTIMIZER=${FINAL_LAYER_GRAD_OPTIMIZER:-adam}
ENABLE_GPTQ_PLUS=${ENABLE_GPTQ_PLUS:-0}
ALPHA=${ALPHA:-0.0}
PRE_CLIP=${PRE_CLIP:-0}
# Subset of surrogate losses to measure against the true end-to-end KL gradient.
# Choices: fisher_diag_mse, residual_kl, refined_residual_kl,
# refined_diag_residual_kl, refined_mse, layer_mse, module_mse.
# Skipping refined_residual_kl avoids the H×H A fit (big CPU-RAM win on 70B).
MEASURE_LOSSES=${MEASURE_LOSSES:-"fisher_diag_mse,layer_mse,module_mse"}
# Reference quantization path used before measuring target-layer cosine.
# Choices: rtn (default) / gptaq / gptq_plus.
ANALYSIS_QUANT_METHOD=${ANALYSIS_QUANT_METHOD:-rtn}
# Grad clip applied element-wise to every captured gradient before cosine /
# L2-norm measurement. Mirrors the main pipeline so the diagnostic reflects
# what block_gd actually sees. Negative → disable. FINAL_LAYER_GRAD_CLIP is
# an optional override used only when the final transformer block is in the
# target list; leave empty / "none" to reuse GRAD_CLIP everywhere.
GRAD_CLIP=${GRAD_CLIP:--1}
FINAL_LAYER_GRAD_CLIP=${FINAL_LAYER_GRAD_CLIP:-}

FINAL_LAYER_GRAD_CLIP_ARGS=()
if [[ -n "${FINAL_LAYER_GRAD_CLIP}" && "${FINAL_LAYER_GRAD_CLIP}" != "none" ]]; then
    FINAL_LAYER_GRAD_CLIP_ARGS=(--final_layer_grad_clip "${FINAL_LAYER_GRAD_CLIP}")
fi

FINAL_LAYER_GRAD_OPTIMIZER_ARGS=()
if [[ -n "${FINAL_LAYER_GRAD_OPTIMIZER}" && "${FINAL_LAYER_GRAD_OPTIMIZER}" != "none" ]]; then
    FINAL_LAYER_GRAD_OPTIMIZER_ARGS=(--final_layer_grad_optimizer "${FINAL_LAYER_GRAD_OPTIMIZER}")
fi

FINAL_LAYER_GRAD_LR_ARGS=()
if [[ -n "${FINAL_LAYER_GRAD_LR}" && "${FINAL_LAYER_GRAD_LR}" != "none" ]]; then
    FINAL_LAYER_GRAD_LR_ARGS=(--final_layer_grad_lr "${FINAL_LAYER_GRAD_LR}")
fi

PRE_CLIP_ARGS=()
if [[ "${PRE_CLIP}" == "1" ]]; then
    PRE_CLIP_ARGS=(--pre_clip)
else
    PRE_CLIP_ARGS=(--no_pre_clip)
fi

# Regularizer added to the surrogate gradient before cosine measurement.
# Supported: none (default) / l2 / hessian. l2 gives reg_grad = λ·(W_q - W_fp);
# hessian uses reg_grad = λ·(W_q - W_fp)·H where H = inp.T@inp (num_groups=1).
# Gate variants (quant_error_gate*) are rejected — they're multiplicative on
# the optimizer update, not additive on the gradient.
GRAD_REG_STRATEGY=${GRAD_REG_STRATEGY:-none}
GRAD_REG_LAMBDA=${GRAD_REG_LAMBDA:-0.0}

export CUDA_VISIBLE_DEVICES=${DEVICE}
NPROC_PER_NODE=${NPROC_PER_NODE:-$(awk -F',' '{print NF}' <<< "${DEVICE}")}

python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" --rdzv_endpoint=localhost:29600 ./analyze_grad_cosine.py \
    --model ${MODEL_PATH} \
    --exp grad_cosine \
    --dataset wikitext2 --nsamples ${N_SAMPLES} --seq_len ${SEQ_LEN} \
    --w_method "${ANALYSIS_QUANT_METHOD}" --analysis_quant_method "${ANALYSIS_QUANT_METHOD}" \
    --w_bits 4 --w_clip --act_order \
    --rotate \
    --skip_eval \
    --kl_topk 20 --grad_hessian_topk 20 \
    --num_groups 4 --bsz "${BSZ}" --global_loss_bsz "${GLOBAL_LOSS_BSZ}" \
    --global_loss --enable_gptq_plus "${ENABLE_GPTQ_PLUS}" \
    --g_update_mode block_gd --grad_refresh_loss fisher_diag_mse \
    --backward_samples "${BACKWARD_SAMPLES}" --backward_bsz "${BACKWARD_BSZ}" \
    --blocksize "${BLOCKSIZE}" \
    --grad_lr "${GRAD_LR}" --grad_optimizer "${GRAD_OPTIMIZER}" \
    "${FINAL_LAYER_GRAD_LR_ARGS[@]}" \
    "${FINAL_LAYER_GRAD_OPTIMIZER_ARGS[@]}" \
    --pre_gd_steps 0 --alpha "${ALPHA}" \
    "${PRE_CLIP_ARGS[@]}" \
    --target_layers ${TARGET_LAYERS} \
    --measure_samples ${MEASURE_SAMPLES} --measure_batch_size ${MEASURE_BSZ} \
    --measure_losses "${MEASURE_LOSSES}" \
    --grad_clip "${GRAD_CLIP}" \
    --grad_reg_strategy "${GRAD_REG_STRATEGY}" \
    --grad_reg_lambda "${GRAD_REG_LAMBDA}" \
    --dp_global_shuffle \
    "${FINAL_LAYER_GRAD_CLIP_ARGS[@]}" \
    --fisher_rademacher_k 0 \
