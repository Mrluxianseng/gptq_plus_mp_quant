#!/bin/bash
#
# TEMPORARY. Answers the one question the null test cannot: does the warm_adam
# prior land on the weights it was measured for?
#
# t0=1 so the prior actually enters the computation (at t0=0 it is multiplied by
# zero and any misalignment is invisible). Only layer 0 is quantized and eval is
# skipped -- this is an alignment probe, not a quality run.
#
# Read the `warm-align-dbg` lines:
#   log-corr ~0.5-0.9 with shuffled-control ~0  -> aligned
#   log-corr ~= shuffled-control ~0             -> the permutation is wrong
set -u
ROOT=/mnt/d/gptq_plus_realq
VENV=/mnt/d/gptq_plus/.venv_wsl
MODEL=${MODEL:-/mnt/d/llamaModels/Qwen3-0.6B}
cd "$ROOT" || exit 1
source "$VENV/bin/activate"
mkdir -p "$ROOT/outputs"

MODE=${MODE:-rank}          # rank -> batched path; none -> plain path
OUT="$ROOT/outputs/alignprobe_${MODE}.log"

OUTPUT_ROOT="$ROOT/outputs" \
DATASET=wikitext2 N_SAMPLES=32 SEQ_LEN=256 \
BSZ=1 FINAL_LAYER_STATS_BSZ=1 HESSIAN_ACCUM_BSZ=4 \
BACKWARD_SAMPLES=32 BACKWARD_BSZ=2 FINAL_LAYER_BACKWARD_BSZ=2 \
BLOCKSIZE=128 W_GROUPSIZE=-1 ACT_ORDER=1 \
GRAD_LRS=0.0002 FINAL_LAYER_GRAD_LR=0.000001 \
GRAD_CLIP=5e-5 FINAL_LAYER_GRAD_CLIP=5e-4 \
GRAD_REFRESH_LOSS=fisher_diag_mse LOSS_SLIDE_WINDOW=0 \
GLOBAL_LOSS=1 GLOBAL_LOSS_BSZ=1 DP_GLOBAL_SHUFFLE=0 \
ALPHA=0.0 ENABLE_GPTQ_PLUS=0 \
PRE_GD_STEPS=0 PRE_GRAD_LR=0 PRE_FINAL_LAYER_GRAD_LR=none \
PRE_GRAD_OPTIMIZER=sgd PRE_FINAL_LAYER_GRAD_OPTIMIZER=none \
GRAD_REG_STRATEGY=none KL_TOPK=20 GRAD_HESSIAN_TOPK=20 \
SALIENCY_CLIP_PERCENTILE=0.99 A_LOSS_RATIO=1.0 \
ENABLE_QA_EVAL=0 STAGE2_CPU_MASTER=0 \
GROUP_PARALLEL_QUANT="$MODE" \
GRAD_OPTIMIZER=warm_adam FINAL_LAYER_GRAD_OPTIMIZER=warm_adam \
BASE_EXP="alignprobe_${MODE}" \
STATIC_CACHE_PATH="$ROOT/cache/nulltest" \
RDZV_PORT=29419 \
WARM_ADAM_DEBUG=1 \
env bash scripts/gptq_plus_lr_sweep.sh "$MODEL" 4 0 \
    --eval_seq_len 256 --eval_datasets wikitext2 \
    --warm_start_steps 1 --quant_stop_layer 0 --skip_eval \
    > "$OUT" 2>&1
echo "exit=$? log=$OUT"
grep -a "warm-align-dbg" "$OUT" | head -20
