#!/bin/bash

set -u

if [[ -z "${MODEL:-}" || -z "${OUT:-}" || -z "${CACHE:-}" ]]; then
    echo "Please export MODEL, OUT, and CACHE before running."
    echo "Example:"
    echo "  export MODEL=/mnt/d/llamaModels/Llama-2-7b-hf"
    echo "  export OUT=/mnt/d/gptq_outputs"
    echo "  export CACHE=/mnt/d/gptq_cache"
    exit 1
fi

REST_SECONDS=300
FAILED_STEPS=()

run_step() {
    local name="$1"
    shift

    echo "============================================================"
    echo "[serial] Starting: ${name}"
    echo "[serial] Start time: $(date '+%F %T')"
    echo "============================================================"

    "$@"
    local status=$?

    if [[ ${status} -ne 0 ]]; then
        echo "[serial] FAILED: ${name} (exit=${status})"
        FAILED_STEPS+=("${name}")
    else
        echo "[serial] Finished: ${name}"
    fi

    echo "[serial] End time: $(date '+%F %T')"
    echo "[serial] Resting for ${REST_SECONDS} seconds..."
    sleep "${REST_SECONDS}"
}

# run_step "GPTAQ" \
#     python -m torch.distributed.run \
#     --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=localhost:29400 \
#     ./ptq.py \
#     --model "${MODEL}" \
#     --exp formal_gptaq_w4_ns512_seq256 \
#     --output_dir "${OUT}" \
#     --cache_dir "${CACHE}" \
#     --dataset wikitext2 --nsamples 512 --seq_len 256 --eval_seq_len 256 \
#     --w_method gptaq --w_bits 4 --w_clip \
#     --rotate \
#     --num_groups 4 --act_order \
#     --blocksize 256 \
#     --bsz 2 --final_layer_stats_bsz 1 --hessian_accum_bsz 4

run_step "GuidedQuant saliency precompute" \
    python ./save_grads.py \
    --model "${MODEL}" \
    --exp formal_guidedquant_saliency_ns512_seq256 \
    --output_dir "${OUT}" \
    --cache_dir "${CACHE}" \
    --dataset wikitext2 --nsamples 512 --seq_len 256 \
    --mode gradients \
    --num_groups 4

run_step "GuidedQuant" \
    python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=localhost:29400 \
    ./ptq.py \
    --model "${MODEL}" \
    --exp formal_guidedquant_w4_ns512_seq256 \
    --output_dir "${OUT}" \
    --cache_dir "${CACHE}" \
    --dataset wikitext2 --nsamples 512 --seq_len 256 --eval_seq_len 256 \
    --w_method gptq_guided --w_bits 4 --w_clip \
    --rotate \
    --num_groups 4 --act_order \
    --blocksize 256 \
    --bsz 2 --final_layer_stats_bsz 1 --hessian_accum_bsz 4

run_step "GPTQ+ fisher_diag_mse" \
    bash -lc '
        OUTPUT_ROOT="${OUT}" \
        BASE_EXP="formal_gptq_plus_fisher_diag_mse" \
        DATASET="wikitext2" \
        N_SAMPLES=64 \
        SEQ_LEN=64 \
        BSZ=2 \
        FINAL_LAYER_STATS_BSZ=1 \
        HESSIAN_ACCUM_BSZ=4 \
        BACKWARD_SAMPLES=32 \
        BACKWARD_BSZ=2 \
        FINAL_LAYER_BACKWARD_BSZ=1 \
        BLOCKSIZE=256 \
        GRAD_LRS="1e-5" \
        GRAD_OPTIMIZER="adam" \
        FINAL_LAYER_GRAD_OPTIMIZER="adam" \
        GRAD_CLIP=1.0 \
        GRAD_REFRESH_LOSS="fisher_diag_mse" \
        FINAL_LAYER_GRAD_LR="1e-5" \
        PRE_GD_STEPS=0 \
        PRE_GRAD_LR=0 \
        PRE_FINAL_LAYER_GRAD_LR="none" \
        PRE_GRAD_OPTIMIZER="sgd" \
        PRE_FINAL_LAYER_GRAD_OPTIMIZER="none" \
        GRAD_REG_STRATEGY="none" \
        GRAD_HESSIAN_TOPK=20 \
        SALIENCY_CLIP_PERCENTILE=0.99 \
        PROJ_LR_SCALE=1.0 \
        DOWN_PROJ_LR_SCALE=1.0 \
        SECOND_ORDER_SCALE=1.0 \
        FISHER_NUM_GROUPS=512 \
        PRE_CLIP=0 \
        GLOBAL_LOSS=1 \
        GLOBAL_LOSS_BSZ=1 \
        LOSS_SLIDE_WINDOW=1 \
        DP_GLOBAL_SHUFFLE=1 \
        GRAD_LR_LAYER_SCHEDULE="none" \
        ALPHA=0.0 \
        KL_TOPK=20 \
        ENABLE_QA_EVAL=1 \
        ENABLE_GPTQ_PLUS=1 \
        bash scripts/gptq_plus_lr_sweep.sh "${MODEL}" 4 0 --eval_seq_len 256
    '

run_step "RTN" \
    python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=localhost:29400 \
    ./ptq.py \
    --model "${MODEL}" \
    --exp formal_rtn_w4_ns512_seq256 \
    --output_dir "${OUT}" \
    --cache_dir "${CACHE}" \
    --dataset wikitext2 --nsamples 512 --seq_len 256 --eval_seq_len 256 \
    --w_method rtn --w_bits 4 --w_clip \
    --rotate

run_step "GPTQ" \
    python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=localhost:29400 \
    ./ptq.py \
    --model "${MODEL}" \
    --exp formal_gptq_w4_ns512_seq256 \
    --output_dir "${OUT}" \
    --cache_dir "${CACHE}" \
    --dataset wikitext2 --nsamples 512 --seq_len 256 --eval_seq_len 256 \
    --w_method gptq --w_bits 4 --w_clip \
    --rotate \
    --num_groups 4 --act_order \
    --blocksize 256 \
    --bsz 2 --final_layer_stats_bsz 1 --hessian_accum_bsz 4

echo "============================================================"
echo "[serial] All steps finished at $(date '+%F %T')"
if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
    echo "[serial] Failed steps:"
    printf '  - %s\n' "${FAILED_STEPS[@]}"
    exit 1
fi
echo "[serial] All steps completed successfully."
