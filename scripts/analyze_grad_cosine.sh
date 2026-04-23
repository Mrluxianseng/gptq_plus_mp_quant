#!/bin/bash

# Usage: bash scripts/analyze_grad_cosine.sh <MODEL_PATH> <DEVICE> [TARGET_LAYERS]
# Example: bash scripts/analyze_grad_cosine.sh ./modelzoo/Qwen3/Qwen3-0.6B 0 "1,5,10,15"

MODEL_PATH=${1}
DEVICE=${2}
TARGET_LAYERS=${3:-"5,10,15,20,25,30"}

MODEL_NAME=$(basename ${MODEL_PATH})
N_SAMPLES=128
SEQ_LEN=2048
MEASURE_SAMPLES=128
MEASURE_BSZ=4
# Subset of surrogate losses to measure against the true end-to-end KL gradient.
# Choices: fisher_diag_mse, residual_kl, refined_residual_kl, refined_diag_residual_kl.
# Skipping refined_residual_kl avoids the H×H A fit (big CPU-RAM win on 70B).
MEASURE_LOSSES=${MEASURE_LOSSES:-"refined_mse"}
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

# Regularizer added to the surrogate gradient before cosine measurement.
# Supported: none (default) / l2 / hessian. l2 gives reg_grad = λ·(W_q - W_fp);
# hessian uses reg_grad = λ·(W_q - W_fp)·H where H = inp.T@inp (num_groups=1).
# Gate variants (quant_error_gate*) are rejected — they're multiplicative on
# the optimizer update, not additive on the gradient.
GRAD_REG_STRATEGY=${GRAD_REG_STRATEGY:-none}
GRAD_REG_LAMBDA=${GRAD_REG_LAMBDA:-0.0}

export CUDA_VISIBLE_DEVICES=${DEVICE}

python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=localhost:2950${DEVICE} ./analyze_grad_cosine.py \
    --model ${MODEL_PATH} \
    --exp grad_cosine \
    --dataset wikitext2 --nsamples ${N_SAMPLES} --seq_len ${SEQ_LEN} \
    --w_method gptaq --w_bits 4 --w_clip --act_order \
    --rotate \
    --skip_eval \
    --kl_topk 20 --grad_hessian_topk 20 \
    --num_groups 4 --fisher_num_groups 512 --bsz 32 --global_loss_bsz 2 \
    --target_layers ${TARGET_LAYERS} \
    --measure_samples ${MEASURE_SAMPLES} --measure_batch_size ${MEASURE_BSZ} \
    --measure_losses "${MEASURE_LOSSES}" \
    --grad_clip "${GRAD_CLIP}" \
    --grad_reg_strategy "${GRAD_REG_STRATEGY}" \
    --grad_reg_lambda "${GRAD_REG_LAMBDA}" \
    "${FINAL_LAYER_GRAD_CLIP_ARGS[@]}"
