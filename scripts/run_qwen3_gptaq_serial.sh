#!/bin/bash

set -u

ROOT_DIR=${ROOT_DIR:-/root/lh/gptq_plus}
MODEL_ROOT=${MODEL_ROOT:-/root/lh/llmModels}
OUT_DIR=${OUT_DIR:-${ROOT_DIR}/outputs}
CACHE_DIR=${CACHE_DIR:-${ROOT_DIR}/cache}
DEVICE=${DEVICE:-0}
REST_SECONDS=${REST_SECONDS:-120}

if [[ ! -d "${ROOT_DIR}" ]]; then
    echo "ROOT_DIR does not exist: ${ROOT_DIR}"
    exit 1
fi

if [[ ! -d "${MODEL_ROOT}" ]]; then
    echo "MODEL_ROOT does not exist: ${MODEL_ROOT}"
    exit 1
fi

mkdir -p "${OUT_DIR}" "${CACHE_DIR}"

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

run_gptaq() {
    local model_path="$1"
    local exp_name="$2"
    local bsz="$3"
    local final_layer_stats_bsz="$4"
    local hessian_accum_bsz="$5"
    local lm_eval_batch_size="$6"
    local use_rotate="$7"

    local cmd=(
        python -m torch.distributed.run
        --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=localhost:2940"${DEVICE}"
        ./ptq.py
        --model "${model_path}"
        --exp "${exp_name}"
        --output_dir "${OUT_DIR}"
        --cache_dir "${CACHE_DIR}"
        --dataset wikitext2
        --nsamples 512
        --seq_len 2048
        --eval_seq_len 2048
        --seed 42
        --w_method gptaq
        --w_bits 4
        --w_clip
        --act_order
        --blocksize 128
        --bsz "${bsz}"
        --final_layer_stats_bsz "${final_layer_stats_bsz}"
        --hessian_accum_bsz "${hessian_accum_bsz}"
        --lm_eval
        --lm_eval_batch_size "${lm_eval_batch_size}"
    )

    if [[ "${use_rotate}" == "1" ]]; then
        cmd+=(--rotate)
    fi

    CUDA_VISIBLE_DEVICES="${DEVICE}" "${cmd[@]}"
}

# Priority 1: GPTAQ+Quarot Qwen3-4B
run_step "GPTAQ+Quarot Qwen3-4B" \
    run_gptaq \
    "${MODEL_ROOT}/Qwen3-4B" \
    "qwen3_4b_gptaq_quarot_w4_s512_t2048" \
    "4" \
    "2" \
    "8" \
    "16" \
    "1"

# Priority 2: GPTAQ Qwen3-0.6B
run_step "GPTAQ Qwen3-0.6B" \
    run_gptaq \
    "${MODEL_ROOT}/Qwen3-0.6B" \
    "qwen3_0p6b_gptaq_w4_s512_t2048" \
    "16" \
    "8" \
    "32" \
    "32" \
    "0"

echo "============================================================"
echo "[serial] All steps finished at $(date '+%F %T')"
if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
    echo "[serial] Failed steps:"
    printf '  - %s\n' "${FAILED_STEPS[@]}"
    exit 1
fi
echo "[serial] All steps completed successfully."
