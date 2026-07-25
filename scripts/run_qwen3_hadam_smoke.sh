#!/bin/bash
#
# H-Adam KFAC smoke test (single run) — BLOCK_GD variant.
#
# Purpose: confirm the new block_gd `h_adam` code path runs end-to-end and yields
# sane wikitext2 KL/PPL. This mirrors the known-good refined_mse pipeline from
# run_qwen3_refined_mse_fllr_sweep.sh with the SAME baseline config (preclip off,
# same sample counts / seq_len), flipping ONLY `GRAD_OPTIMIZER=adam -> h_adam`
# (the block_gd optimizer, which is where refined_mse actually optimises). So the
# result is directly comparable against the adam baseline once the full arm runs.
#
# Notes:
#   - refined_mse forces --enable_gptq_plus 0 and pre_gd is disabled (preclip off),
#     so the pre-GD h_adam path is inactive here by design; the block_gd path is
#     what we are validating.
#   - QA eval is disabled and eval is limited to wikitext2 to keep the smoke fast
#     and fully offline (only wikitext2 parquet is cached locally). Neither changes
#     the quantization result.
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True reduces fragmentation so the
#     static-Fisher precompute fits on a 32 GB GPU at the baseline seq_len/bsz. If
#     it still OOMs, lower GLOBAL_LOSS_BSZ (8 -> 4): that only re-tiles the Fisher
#     accumulation and is numerically identical (the Fisher is a sum over samples).

set -u

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
echo "[smoke] H-Adam KFAC single run (block_gd)"
echo "[smoke] model=${model_name} grad_optimizer=h_adam damping=${H_ADAM_CURVATURE_DAMPING:-<default 0.1>}"
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
ENABLE_GPTQ_PLUS="0" \
BACKWARD_SAMPLES=512 \
BACKWARD_BSZ="16" \
FINAL_LAYER_BACKWARD_BSZ="16" \
BLOCKSIZE=128 \
GRAD_OPTIMIZER="h_adam" \
FINAL_LAYER_GRAD_OPTIMIZER="adam" \
GRAD_CLIP="5e-5" \
FINAL_LAYER_GRAD_CLIP="5e-4" \
GRAD_REFRESH_LOSS="refined_mse" \
NUM_SAMPLES_FOR_REFINED_MSE="32" \
FINAL_LAYER_GRAD_LR="0.000001" \
PRE_GD_STEPS=10 \
PRE_GRAD_LR="0.00003" \
PRE_FINAL_LAYER_GRAD_LR="0.3" \
PRE_GRAD_OPTIMIZER="adam" \
PRE_FINAL_LAYER_GRAD_OPTIMIZER="sgd" \
H_ADAM_CURVATURE_DAMPING="${H_ADAM_CURVATURE_DAMPING}" \
GRAD_REG_STRATEGY="none" \
GLOBAL_LOSS=1 \
GLOBAL_LOSS_BSZ="8" \
LOSS_SLIDE_WINDOW="0" \
DP_GLOBAL_SHUFFLE=1 \
GRAD_LR_LAYER_SCHEDULE="cosine" \
ENABLE_QA_EVAL=0 \
LM_EVAL_BATCH_SIZE="32" \
GRAD_LRS="0.0002" \
bash "${ROOT_DIR}/scripts/gptq_plus_lr_sweep.sh" "${model_path}" "${NUM_GROUPS}" "${DEVICE}" \
    --seed 42 --eval_datasets wikitext2
status=$?

echo "============================================================"
echo "[smoke] End time: $(date '+%F %T')"
if [[ ${status} -ne 0 ]]; then
    echo "[smoke] FAILED (exit=${status})"
    exit ${status}
fi
echo "[smoke] Finished OK. Compare wikitext2 KL/PPL against the refined_mse lr=2e-4 row."
