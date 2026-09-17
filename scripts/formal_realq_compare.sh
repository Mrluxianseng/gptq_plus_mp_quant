#!/bin/bash
#
# adam vs warm_adam on Qwen3-0.6B W4A16, at the REAL-Q paper's own parameters.
# For the server -- 2048x2048 calibration will not fit a desktop card.
#
# Paper target (Table 5 / App. D.2), which the adam arm should land near:
#     REAL-Q       KL 6.79e-2   PPL 21.57      (bf16 reference PPL 20.95)
#     GuidedQuant  KL 8.76e-2   PPL 22.31
#     GPTAQ        KL 20.8e-2   PPL 25.88
#     GPTQ         KL 30.2e-2   PPL 28.37
#     RTN          KL 55.5e-2   PPL 34.37
#
# Paper -> flag mapping, with the source for each value:
#
#   2048 WikiText-2 calibration samples        (S6.1)      N_SAMPLES=2048
#   sequence length 2048 (2048x2048 ~ 4M tok)  (S6.1)      SEQ_LEN=2048
#   W4A16, per-row weights, no grouping        (S6.1)      w_bits 4, W_GROUPSIZE=-1
#   symmetric quantisation throughout          (S6.1)      no *_asym flags
#   QuaRot rotation shared by all methods      (S6.1)      --rotate (always on)
#   Adam mini-batch 32 per gradient step       (S6.1)      BACKWARD_SAMPLES=32
#   Adam b1=.9 b2=.999 eps=1e-8, bias corr.    (D.1)       code defaults
#   column block size B=128                    (Table 4)   BLOCKSIZE=128  (upstream default)
#   slide window on (base ablation config)     (Table 4)   LOSS_SLIDE_WINDOW=1 (upstream default)
#   aggregated Fisher MSE surrogate            (S6.6)      GRAD_REFRESH_LOSS=fisher_diag_mse
#     (upstream README: the "diag" in that name is a legacy misnomer -- it is the
#      full Fisher now; `legacy_fisher_diag_mse` is the genuinely diagonal one.
#      `--loss_slide_window` also hard-requires this loss.)
#   a_loss_clip=0.95 for small Qwen3 models    (D.1)       A_LOSS_RATIO=0.95
#   Hessian saliency clipped at 99th pct       (D.1)       SALIENCY_CLIP_PERCENTILE=0.99
#   scheduled final LR  3.0e-4  (Qwen3-0.6B)   (Table 6)   LR=3e-4
#   final-layer LR      1.0e-5  (Qwen3-0.6B)   (Table 6)   FINAL_LAYER_GRAD_LR=1e-5
#   reverse-cosine layer LR, base = 0.01*final (D.1/D.3)   GRAD_LR_LAYER_SCHEDULE=cosine
#     (upstream's `cosine` IS the paper's sin(pi*x/2) ramp, and its default)
#   final block optimises true KL vs lm_head   (D.1)       in-code for the last layer
#   4 GPUs, 0.18 h/GPU  (~0.72 GPU-hours)      (Table 7)   DEVICE=0,1,2,3
#   KL/PPL on WikiText-2 + 10 zero-shot tasks  (S6.1)      ENABLE_QA_EVAL=1
#
# Not stated in the paper; all env-overridable:
#   PRE_GD_STEPS=0   -- no pre-quantisation GD stage is described anywhere
#                       (S6.1 / D.1 / Algorithm 1). Upstream's sweep default of
#                       10 looks like a sweep convenience.
#   KL_TOPK / GRAD_HESSIAN_TOPK = -1 (upstream defaults, no truncation). The
#                       paper mentions no top-k anywhere.
#   GRAD_CLIP / FINAL_LAYER_GRAD_CLIP -- upstream defaults, left alone.
#   ACT_ORDER=1      -- upstream default.
#
# ON t0
# -----
# warm_adam's t0 defaults to nsamples/backward_samples, which is 1 at the local
# probe scale but 2048/32 = 64 HERE. That is a different regime, not a bigger
# version of the same one: at t0=64 the prior still supplies ~86% of v-hat at a
# module's 23rd step, so warm_adam sits close to a frozen preconditioner rather
# than a warm-started EMA. T0_LIST sweeps that on purpose -- do not assume the
# local t0=1 behaviour carries over.
#
# Usage:
#   export MODEL=/path/to/Qwen3-0.6B
#   DRY_RUN=1 bash scripts/formal_realq_compare.sh     # print plan, launch nothing
#   bash scripts/formal_realq_compare.sh
#   LR="1e-4 3e-4 1e-3" bash scripts/formal_realq_compare.sh   # LR sweep
#
# The sweep script defaults HF_DATASETS_OFFLINE=1. If a dataset is missing
# locally, the run aborts up front naming that flag; pass HF_DATASETS_OFFLINE=0
# once to let it download and snapshot, then it is local from then on.
#
# LR may be a space-separated list: the sweep script loops over it internally,
# and the Stage-0 cache key does not depend on the learning rate, so a whole
# sweep still shares the single Stage 1 precompute below.
#
set -u

ROOT_DIR=${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "${ROOT_DIR}"

MODEL=${MODEL:?export MODEL=/path/to/Qwen3-0.6B}
DEVICE=${DEVICE:-0,1,2,3}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT_DIR}/outputs}
LOG=${LOG:-${OUTPUT_ROOT}/formal_compare_queue.log}

# --- calibration (paper-pinned) ------------------------------------------
# 2048 WikiText-2 sequences of 2048 tokens (S6.1). Seed 1 is the calibration
# sample behind the reported Qwen3-0.6B W4A16 row; the sweep script never
# passes --seed, so without pinning it argparse's 42 applies silently. The
# paper's own five-seed sweep (Table 9) spans only KL 6.79-6.92e-2, so the
# seed is not a lever for closing a large gap -- it is pinned to match the
# reported row, not to tune.
DATASET=${DATASET:-wikitext2}
N_SAMPLES=${N_SAMPLES:-2048}
SEED=${SEED:-1}

# --- paper-pinned ---------------------------------------------------------
SEQ_LEN=${SEQ_LEN:-2048}
EVAL_SEQ_LEN=${EVAL_SEQ_LEN:-2048}
BLOCKSIZE=${BLOCKSIZE:-128}
BACKWARD_SAMPLES=${BACKWARD_SAMPLES:-32}
LR=${LR:-3e-4}
FINAL_LAYER_GRAD_LR=${FINAL_LAYER_GRAD_LR:-1e-5}
A_LOSS_RATIO=${A_LOSS_RATIO:-0.95}
SALIENCY_CLIP_PERCENTILE=${SALIENCY_CLIP_PERCENTILE:-0.99}
LOSS_SLIDE_WINDOW=${LOSS_SLIDE_WINDOW:-1}
GRAD_REFRESH_LOSS=${GRAD_REFRESH_LOSS:-fisher_diag_mse}
GRAD_LR_LAYER_SCHEDULE=${GRAD_LR_LAYER_SCHEDULE:-cosine}
GRAD_LR_LAYER_BASE_RATIO=${GRAD_LR_LAYER_BASE_RATIO:-0.01}
NUM_GROUPS=${NUM_GROUPS:-4}
W_GROUPSIZE=${W_GROUPSIZE:--1}

# The Stage-0 cache key embeds nsamples / seq_len / seed / num_groups / world
# size, so a cache built for one calibration set is useless for another. Keying
# the directory on the same values keeps the "reuse if present" check below from
# skipping Stage 1 on a directory whose contents belong to a different set --
# which would then fail in Stage 2, since STAGE2_CPU_MASTER refuses to compute
# it inline.
STATIC_CACHE_PATH=${STATIC_CACHE_PATH:-${ROOT_DIR}/cache/formal_realq_qwen3_0p6b_s${N_SAMPLES}_l${SEQ_LEN}_seed${SEED}}

# Paper-pinned: the final block optimises "true full-vocabulary KL against
# the LM head" (D.1), and full-vocabulary means no top-k truncation.
KL_TOPK=${KL_TOPK:--1}

# --- not stated in the paper ---------------------------------------------
# docs/REALQ_PAPER_PROTOCOL.md lists what main.tex leaves undisclosed:
# gradient-clipping operator and thresholds, GPTQ damping, activation order,
# weight-clipping search, Hessian accumulation batch, backward chunk size,
# evaluation chunk length, the populations the P95/P99 percentiles are taken
# over, the Fisher label seed, and refresh-sample ordering. These are upstream
# defaults, and any residual gap to the paper table lives in this list.
PRE_GD_STEPS=${PRE_GD_STEPS:-0}
GRAD_HESSIAN_TOPK=${GRAD_HESSIAN_TOPK:--1}
ACT_ORDER=${ACT_ORDER:-1}

# --- REAL-Q structure -----------------------------------------------------
ALPHA=${ALPHA:-0.0}
ENABLE_GPTQ_PLUS=${ENABLE_GPTQ_PLUS:-0}
GLOBAL_LOSS=${GLOBAL_LOSS:-1}
DP_GLOBAL_SHUFFLE=${DP_GLOBAL_SHUFFLE:-1}

# --- memory / speed only --------------------------------------------------
BSZ=${BSZ:-8}
GLOBAL_LOSS_BSZ=${GLOBAL_LOSS_BSZ:-8}
HESSIAN_ACCUM_BSZ=${HESSIAN_ACCUM_BSZ:-8}
BACKWARD_BSZ=${BACKWARD_BSZ:-4}
FINAL_LAYER_BACKWARD_BSZ=${FINAL_LAYER_BACKWARD_BSZ:-4}
FINAL_LAYER_STATS_BSZ=${FINAL_LAYER_STATS_BSZ:-4}
ENABLE_QA_EVAL=${ENABLE_QA_EVAL:-1}
LM_EVAL_BATCH_SIZE=${LM_EVAL_BATCH_SIZE:-16}
EVAL_DATASETS=${EVAL_DATASETS:-wikitext2}
RDZV_PORT=${RDZV_PORT:-29500}

# warm_adam arms. "-1" means derive t0 = nsamples / backward_samples (= 64 here).
T0_LIST=${T0_LIST:--1}
# Control arm: scalar prior (magnitude kept, per-coordinate shape removed).
WARM_PRIOR_SCALAR=${WARM_PRIOR_SCALAR:-none}   # none | mean | geomean
WARM_EXTRA=()
if [[ "${WARM_PRIOR_SCALAR}" != "none" ]]; then
    WARM_EXTRA=(--warm_prior_scalar "${WARM_PRIOR_SCALAR}")
fi

IFS=',' read -r -a _DEVS <<< "${DEVICE}"
N_GPUS=${#_DEVS[@]}
if (( BACKWARD_SAMPLES % N_GPUS != 0 )); then
    echo "BACKWARD_SAMPLES (${BACKWARD_SAMPLES}) must divide by the GPU count (${N_GPUS})." >&2
    exit 1
fi

# Everything except the optimizer and t0. Stage 0 (saliency / Fisher) depends on
# none of those, so every arm below reads ONE precomputed cache and the optimizer
# is the only moving part.
common() {
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    DATASET="${DATASET}" \
    N_SAMPLES="${N_SAMPLES}" SEQ_LEN="${SEQ_LEN}" \
    BSZ="${BSZ}" FINAL_LAYER_STATS_BSZ="${FINAL_LAYER_STATS_BSZ}" \
    HESSIAN_ACCUM_BSZ="${HESSIAN_ACCUM_BSZ}" \
    BACKWARD_SAMPLES="${BACKWARD_SAMPLES}" BACKWARD_BSZ="${BACKWARD_BSZ}" \
    FINAL_LAYER_BACKWARD_BSZ="${FINAL_LAYER_BACKWARD_BSZ}" \
    BLOCKSIZE="${BLOCKSIZE}" W_GROUPSIZE="${W_GROUPSIZE}" ACT_ORDER="${ACT_ORDER}" \
    A_LOSS_RATIO="${A_LOSS_RATIO}" \
    GRAD_LRS="${LR}" FINAL_LAYER_GRAD_LR="${FINAL_LAYER_GRAD_LR}" \
    GRAD_REFRESH_LOSS="${GRAD_REFRESH_LOSS}" \
    LOSS_SLIDE_WINDOW="${LOSS_SLIDE_WINDOW}" \
    GLOBAL_LOSS="${GLOBAL_LOSS}" GLOBAL_LOSS_BSZ="${GLOBAL_LOSS_BSZ}" \
    DP_GLOBAL_SHUFFLE="${DP_GLOBAL_SHUFFLE}" \
    GRAD_LR_LAYER_SCHEDULE="${GRAD_LR_LAYER_SCHEDULE}" \
    GRAD_LR_LAYER_BASE_RATIO="${GRAD_LR_LAYER_BASE_RATIO}" \
    ALPHA="${ALPHA}" ENABLE_GPTQ_PLUS="${ENABLE_GPTQ_PLUS}" \
    PRE_GD_STEPS="${PRE_GD_STEPS}" PRE_GRAD_LR=0 PRE_FINAL_LAYER_GRAD_LR=none \
    PRE_GRAD_OPTIMIZER=sgd PRE_FINAL_LAYER_GRAD_OPTIMIZER=none \
    GRAD_REG_STRATEGY=none \
    KL_TOPK="${KL_TOPK}" GRAD_HESSIAN_TOPK="${GRAD_HESSIAN_TOPK}" \
    SALIENCY_CLIP_PERCENTILE="${SALIENCY_CLIP_PERCENTILE}" \
    PROJ_LR_SCALE=1.0 DOWN_PROJ_LR_SCALE=1.0 SECOND_ORDER_SCALE=1.0 PRE_CLIP=0 \
    ENABLE_QA_EVAL="${ENABLE_QA_EVAL}" LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE}" \
    STATIC_CACHE_PATH="${STATIC_CACHE_PATH}" RDZV_PORT="${RDZV_PORT}" \
    env "$@"
}

cat <<BANNER
============================================================
REAL-Q formal comparison - Qwen3-0.6B W4A16
  tree         : $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?') @ $(git rev-parse --short HEAD 2>/dev/null || echo '?')
  model        : ${MODEL}
  gpus         : ${DEVICE}  (${N_GPUS} ranks)
  calibration  : ${DATASET} ${N_SAMPLES} x ${SEQ_LEN} tokens  seed=${SEED}
  rotation     : QuaRot (--rotate, always on in the sweep)
  block size B : ${BLOCKSIZE}    backward smp : ${BACKWARD_SAMPLES}
  refresh loss : ${GRAD_REFRESH_LOSS}  slide_window=${LOSS_SLIDE_WINDOW}
  lr           : ${LR}  (final block ${FINAL_LAYER_GRAD_LR})
  lr schedule  : ${GRAD_LR_LAYER_SCHEDULE}  base_ratio=${GRAD_LR_LAYER_BASE_RATIO}
  topk         : kl=${KL_TOPK}  grad_hessian=${GRAD_HESSIAN_TOPK}
  arms         : adam, warm_adam t0 in { ${T0_LIST} }   (-1 => ${N_SAMPLES}/${BACKWARD_SAMPLES})
  static cache : ${STATIC_CACHE_PATH}
  paper target : KL 6.79e-2 / PPL 21.57  (Table 5, seed 1; Table 9 gives the
                 five-seed spread 6.79-6.92)
============================================================
BANNER

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] set DRY_RUN=0 (or unset it) to launch."
    exit 0
fi

mkdir -p "${STATIC_CACHE_PATH}" "${OUTPUT_ROOT}"
exec >> "${LOG}" 2>&1
say() { echo "=== $(date '+%F %T') | $* ==="; }
# Every result line, not just the last: LR may be a space-separated list, in
# which case the sweep script loops internally and one log holds one result per
# learning rate. Requiring a digit after the colon keeps the bare "KL&PPL on
# <dataset>" progress line from printing as if it were a result.
result() { grep -aoE 'KL&PPL on [a-z0-9_]+: [0-9].*' "${OUTPUT_ROOT}/$1.log"; }

FAILED_ARMS=()
# $? has to be read by the caller right after the run: capturing it inside the
# function would report the function's own status instead.
note_exit() {
    local arm="$1" status="$2"
    if [[ "${status}" -ne 0 ]]; then
        say "FAILED ${arm} exit=${status} -- see ${OUTPUT_ROOT}/${arm}.log"
        FAILED_ARMS+=("${arm}")
    else
        say "done ${arm}"
    fi
}

# Stage 1 -- build the static saliency/Fisher cache once, then exit. Every arm
# below runs with STAGE2_CPU_MASTER=1 (upstream's default), which REFUSES to
# compute this inline and requires the cache to exist. Doing it once also means
# all arms share bit-identical Stage-0 inputs.
# The cache key covers everything Stage 1 depends on and none of it varies
# across the arms below, so an existing cache is reusable and rebuilding it
# costs minutes for nothing. FORCE_PRECOMPUTE=1 rebuilds anyway.
#
# Match the real filename rather than mirroring the key in the directory name:
# mirroring means silently reusing a mismatched cache the moment a field is
# forgotten. That is exactly what happened with GLOBAL_LOSS_BSZ -- a glbsz=16
# run reused a glbsz=8 directory and both arms died in Stage 2, where
# STAGE2_CPU_MASTER refuses to compute the cache inline. The pattern below
# pins every field this script can vary; anything it cannot vary (model hash,
# rotation id, saliency clip) is constant for a given model.
CACHE_GLOB="${STATIC_CACHE_PATH}/*_s${N_SAMPLES}_blk${SEQ_LEN}_*_g${NUM_GROUPS}_*_ghtk${GRAD_HESSIAN_TOPK}_glbsz${GLOBAL_LOSS_BSZ}_cseed${SEED}_*_world${N_GPUS}_rank*.pt"
if [[ "${FORCE_PRECOMPUTE:-0}" != "1" ]] && compgen -G "${CACHE_GLOB}" >/dev/null; then
    say "Stage 1: reusing the cache matching ${CACHE_GLOB} (FORCE_PRECOMPUTE=1 to rebuild)"
else
    say "Stage 1: static precompute -> ${STATIC_CACHE_PATH}"
    common STAGE2_CPU_MASTER=0 EXIT_AFTER_PRECOMPUTE=1 \
        GRAD_OPTIMIZER=adam FINAL_LAYER_GRAD_OPTIMIZER=adam \
        BASE_EXP=formal_precompute \
        bash scripts/gptq_plus_lr_sweep.sh "${MODEL}" "${NUM_GROUPS}" "${DEVICE}" \
        --eval_seq_len "${EVAL_SEQ_LEN}" --seed "${SEED}" --skip_eval \
        > "${OUTPUT_ROOT}/formal_precompute.log" 2>&1
    say "Stage 1 exit=$?"
    if ! compgen -G "${CACHE_GLOB}" >/dev/null; then
        say "ABORT: Stage 1 wrote no cache matching ${CACHE_GLOB}; see formal_precompute.log"
        exit 1
    fi
fi

say "Stage 2a: adam @ lr=${LR}  (the paper's own configuration)"
common STAGE2_CPU_MASTER=1 \
    GRAD_OPTIMIZER=adam FINAL_LAYER_GRAD_OPTIMIZER=adam \
    BASE_EXP=formal_adam \
    bash scripts/gptq_plus_lr_sweep.sh "${MODEL}" "${NUM_GROUPS}" "${DEVICE}" \
    --eval_seq_len "${EVAL_SEQ_LEN}" --seed "${SEED}" --eval_datasets ${EVAL_DATASETS} \
    > "${OUTPUT_ROOT}/formal_adam.log" 2>&1
note_exit formal_adam "$?"
result formal_adam

for t0 in ${T0_LIST}; do
    tag="formal_warm_t${t0//-/m}"
    say "Stage 2b: warm_adam t0=${t0} @ lr=${LR}"
    common STAGE2_CPU_MASTER=1 \
        GRAD_OPTIMIZER=warm_adam FINAL_LAYER_GRAD_OPTIMIZER=warm_adam \
        BASE_EXP="${tag}" \
        bash scripts/gptq_plus_lr_sweep.sh "${MODEL}" "${NUM_GROUPS}" "${DEVICE}" \
        --eval_seq_len "${EVAL_SEQ_LEN}" --seed "${SEED}" --eval_datasets ${EVAL_DATASETS} \
        --warm_start_steps "${t0}" \
        > "${OUTPUT_ROOT}/${tag}.log" 2>&1
    note_exit "${tag}" "$?"
    result "${tag}"
done

# "complete" on its own hid a non-zero exit once already: the adam arm printed
# its KL/PPL and then died in the downstream QA eval, and the run read as a
# success for a whole round of analysis.
if [[ ${#FAILED_ARMS[@]} -gt 0 ]]; then
    say "formal comparison finished WITH FAILURES: ${FAILED_ARMS[*]}"
    exit 1
fi
say "formal comparison complete (all arms exit=0)"
