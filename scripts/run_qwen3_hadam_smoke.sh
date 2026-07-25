#!/bin/bash
#
# H-Adam KFAC smoke test (single run).
#
# Purpose: confirm the new pre-GD `h_adam` code path runs end-to-end and yields
# sane KL/PPL/acc numbers. This is NOT a sweep — it mirrors the known-good
# refined_mse pipeline from run_qwen3_refined_mse_fllr_sweep.sh and flips ONLY
# `PRE_GRAD_OPTIMIZER=adam -> h_adam` (everything else identical), so the result
# is directly comparable against the adam baseline table
# (qwen3_0p6b_refined_lr_results.md, refined_mse row).
#
# After it finishes cleanly, reuse the SAME adam LR grid with PRE_GRAD_OPTIMIZER
# =h_adam to produce the full comparison arm.

set -u

ROOT_DIR=${ROOT_DIR:-/root/lh/gptq_plus}
MODEL_ROOT=${MODEL_ROOT:-/root/lh/llmModels}
DEVICE=${DEVICE:-0}
NUM_GROUPS=${NUM_GROUPS:-4}
W_BITS=${W_BITS:-4}
# Curvature damping: empty -> process_args default (0.1). Override to ablate.
H_ADAM_CURVATURE_DAMPING=${H_ADAM_CURVATURE_DAMPING:-}

if [[ ! -d "${ROOT_DIR}" ]]; then
    echo "ROOT_DIR does not exist: ${ROOT_DIR}"
    exit 1
fi
if [[ ! -d "${MODEL_ROOT}" ]]; then
    echo "MODEL_ROOT does not exist: ${MODEL_ROOT}"
    exit 1
fi

model_name="Qwen3-0.6B"
model_path="${MODEL_ROOT}/${model_name}"

echo "============================================================"
echo "[smoke] H-Adam KFAC single run"
echo "[smoke] model=${model_name} pre_grad_optimizer=h_adam damping=${H_ADAM_CURVATURE_DAMPING:-<default 0.1>}"
echo "[smoke] Start time: $(date '+%F %T')"
echo "============================================================"

cd "${ROOT_DIR}" || exit 1

OUTPUT_ROOT="${ROOT_DIR}/outputs" \
BASE_EXP="qwen3_0p6b_hadam_smoke" \
DATASET="wikitext2" \
N_SAMPLES=512 \
SEQ_LEN=2048 \
ENABLE_DYN_SAL="0" \
BSZ="16" \
FINAL_LAYER_STATS_BSZ="8" \
HESSIAN_ACCUM_BSZ="32" \
ENABLE_GPTQ_PLUS="16" \
BACKWARD_SAMPLES=512 \
BACKWARD_BSZ="16" \
FINAL_LAYER_BACKWARD_BSZ="32" \
BLOCKSIZE=128 \
GRAD_OPTIMIZER="adam" \
FINAL_LAYER_GRAD_OPTIMIZER="adam" \
GRAD_CLIP="5e-5" \
FINAL_LAYER_GRAD_CLIP="5e-4" \
GRAD_REFRESH_LOSS="refined_mse" \
NUM_SAMPLES_FOR_REFINED_MSE="8" \
FINAL_LAYER_GRAD_LR="0.000001" \
PRE_GD_STEPS=10 \
PRE_GRAD_LR="0.00003" \
PRE_FINAL_LAYER_GRAD_LR="0.3" \
PRE_GRAD_OPTIMIZER="h_adam" \
PRE_FINAL_LAYER_GRAD_OPTIMIZER="sgd" \
H_ADAM_CURVATURE_DAMPING="${H_ADAM_CURVATURE_DAMPING}" \
GRAD_REG_STRATEGY="none" \
GLOBAL_LOSS=1 \
GLOBAL_LOSS_BSZ="32" \
LOSS_SLIDE_WINDOW="0" \
DP_GLOBAL_SHUFFLE=1 \
GRAD_LR_LAYER_SCHEDULE="cosine" \
ENABLE_QA_EVAL=1 \
LM_EVAL_BATCH_SIZE="16" \
GRAD_LRS="0.0002" \
bash "${ROOT_DIR}/scripts/gptq_plus_lr_sweep.sh" "${model_path}" "${NUM_GROUPS}" "${DEVICE}" \
    --seed 42
status=$?

echo "============================================================"
echo "[smoke] End time: $(date '+%F %T')"
if [[ ${status} -ne 0 ]]; then
    echo "[smoke] FAILED (exit=${status})"
    exit ${status}
fi
echo "[smoke] Finished OK. Compare against refined_mse lr=2e-4 row in qwen3_0p6b_refined_lr_results.md"
