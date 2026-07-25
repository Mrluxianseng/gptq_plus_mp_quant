#!/bin/bash

set -u

ROOT_DIR=${ROOT_DIR:-/root/lh/gptq_plus}
MODEL_ROOT=${MODEL_ROOT:-/root/lh/llmModels}
DEVICE=${DEVICE:-0}
REST_SECONDS=${REST_SECONDS:-120}
NUM_GROUPS=${NUM_GROUPS:-4}
W_BITS=${W_BITS:-4}

if [[ ! -d "${ROOT_DIR}" ]]; then
    echo "ROOT_DIR does not exist: ${ROOT_DIR}"
    exit 1
fi

if [[ ! -d "${MODEL_ROOT}" ]]; then
    echo "MODEL_ROOT does not exist: ${MODEL_ROOT}"
    exit 1
fi

FAILED_STEPS=()

run_step() {
    local name="$1"
    shift

    echo "============================================================"
    echo "[serial] Starting: ${name}"
    echo "[serial] Start time: $(date '+%F %T')"
    echo "============================================================"

    (
        cd "${ROOT_DIR}" || exit 1
        "$@"
    )
    local status=$?

    if [[ ${status} -ne 0 ]]; then
        echo "[serial] FAILED: ${name} (exit=${status})"
        FAILED_STEPS+=("${name}")
    else
        echo "[serial] Finished: ${name}"
    fi

    echo "[serial] End time: $(date '+%F %T')"
    if [[ "${REST_SECONDS}" -gt 0 ]]; then
        echo "[serial] Resting for ${REST_SECONDS} seconds..."
        sleep "${REST_SECONDS}"
    fi
}

run_refined_sweep() {
    local model_name="$1"
    local grad_refresh_loss="$2"
    local grad_lrs="$3"
    local loss_slide_window="$4"
    local bsz="$5"
    local final_layer_stats_bsz="$6"
    local hessian_accum_bsz="$7"
    local backward_bsz="$8"
    local final_layer_backward_bsz="$9"
    local lm_eval_batch_size="${10}"
    local enable_gptq_plus="${11}"
    local base_exp="${12}"
    local global_loss_bsz="${13}"
    local num_samples_for_refined_mse="${14}"
    local final_layer_grad_lr="${15}"

    local model_path="${MODEL_ROOT}/${model_name}"

    OUTPUT_ROOT="${ROOT_DIR}/outputs" \
    BASE_EXP="${base_exp}" \
    DATASET="wikitext2" \
    N_SAMPLES=512 \
    SEQ_LEN=2048 \
    ENABLE_DYN_SAL="0" \
    BSZ="${bsz}" \
    FINAL_LAYER_STATS_BSZ="${final_layer_stats_bsz}" \
    HESSIAN_ACCUM_BSZ="${hessian_accum_bsz}" \
    ENABLE_GPTQ_PLUS="${enable_gptq_plus}" \
    BACKWARD_SAMPLES=512 \
    BACKWARD_BSZ="${backward_bsz}" \
    FINAL_LAYER_BACKWARD_BSZ="${final_layer_backward_bsz}" \
    BLOCKSIZE=128 \
    GRAD_OPTIMIZER="adam" \
    FINAL_LAYER_GRAD_OPTIMIZER="adam" \
    GRAD_CLIP="5e-5" \
    FINAL_LAYER_GRAD_CLIP="5e-4" \
    GRAD_REFRESH_LOSS="${grad_refresh_loss}" \
    NUM_SAMPLES_FOR_REFINED_MSE="${num_samples_for_refined_mse}" \
    FINAL_LAYER_GRAD_LR="${final_layer_grad_lr}" \
    PRE_GD_STEPS=10 \
    PRE_GRAD_LR="0.00003" \
    PRE_FINAL_LAYER_GRAD_LR="0.3" \
    PRE_GRAD_OPTIMIZER="adam" \
    PRE_FINAL_LAYER_GRAD_OPTIMIZER="sgd" \
    GRAD_REG_STRATEGY="none" \
    GLOBAL_LOSS=1 \
    GLOBAL_LOSS_BSZ="${global_loss_bsz}" \
    LOSS_SLIDE_WINDOW="${loss_slide_window}" \
    DP_GLOBAL_SHUFFLE=1 \
    GRAD_LR_LAYER_SCHEDULE="cosine" \
    ENABLE_QA_EVAL=1 \
    LM_EVAL_BATCH_SIZE="${lm_eval_batch_size}" \
    GRAD_LRS="${grad_lrs}" \
    bash "${ROOT_DIR}/scripts/gptq_plus_lr_sweep.sh" "${model_path}" "${NUM_GROUPS}" "${DEVICE}" \
        --seed 42
}

run_step "Qwen3-0.6B refined_mse lr=2e-4 final_layer_lr=5e-7" \
    run_refined_sweep \
    "Qwen3-0.6B" \
    "refined_mse" \
    "0.0002" \
    "0" \
    "16" \
    "8" \
    "32" \
    "16" \
    "16" \
    "32" \
    "0" \
    "qwen3_0p6b_refined_mse_fllr5e7" \
    "8" \
    "32" \
    "0.0000005"

run_step "Qwen3-0.6B refined_mse lr=2e-4 final_layer_lr=1e-6" \
    run_refined_sweep \
    "Qwen3-0.6B" \
    "refined_mse" \
    "0.0002" \
    "0" \
    "16" \
    "8" \
    "32" \
    "16" \
    "16" \
    "32" \
    "0" \
    "qwen3_0p6b_refined_mse_fllr1e6" \
    "8" \
    "32" \
    "0.000001"

run_step "Qwen3-0.6B refined_mse lr=2e-4 final_layer_lr=2e-6" \
    run_refined_sweep \
    "Qwen3-0.6B" \
    "refined_mse" \
    "0.0002" \
    "0" \
    "16" \
    "8" \
    "32" \
    "16" \
    "16" \
    "32" \
    "0" \
    "qwen3_0p6b_refined_mse_fllr2e6" \
    "8" \
    "32" \
    "0.000002"

echo "============================================================"
echo "[serial] All steps finished at $(date '+%F %T')"
if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
    echo "[serial] Failed steps:"
    printf '  - %s\n' "${FAILED_STEPS[@]}"
    exit 1
fi
echo "[serial] All steps completed successfully."
