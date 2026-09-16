#!/bin/bash
#
# Null test for the warm_adam port on upstream (origin/zq).
#
# `--grad_optimizer warm_adam --warm_start_steps 0` seeds v0 = P*(1-b2^0) = 0
# and v_offset = 0, so it is mathematically identical to plain adam. It must
# reproduce the adam run bit for bit -- while still executing the whole new code
# path (pre-pass, prior build, aligner, warm_adam state branch). Anything else
# means the port perturbs the run somewhere it should not.
#
# This is the check that caught the real bug on the old branch: the pre-pass was
# consuming a chunk of the sample scheduler, shifting every later refresh, and
# moving final KL from 0.211 to 0.222 at lr=2e-4.
#
# Small config on purpose -- this tests plumbing, not quality. The numbers are
# not comparable to the paper.
set -u
ROOT=/mnt/d/gptq_plus_realq
VENV=/mnt/d/gptq_plus/.venv_wsl          # shared with the main worktree
MODEL=${MODEL:-/mnt/d/llamaModels/Qwen3-0.6B}
cd "$ROOT" || exit 1
source "$VENV/bin/activate"
mkdir -p "$ROOT/outputs"
LOG="$ROOT/outputs/nulltest_queue.log"
exec >> "$LOG" 2>&1
say() { echo "=== $(date '+%F %T') | $* ==="; }
result() { grep -aoE 'KL&PPL on wikitext2: .*' "$ROOT/outputs/run$1.log" | tail -1; }

# Held identical across both runs. backward_samples == nsamples is the case
# where the scheduler wrap reshuffles on every call, i.e. the case the pre-pass
# bug was visible in.
common() {
    OUTPUT_ROOT="$ROOT/outputs" \
    DATASET=wikitext2 N_SAMPLES=32 SEQ_LEN=256 \
    BSZ=1 FINAL_LAYER_STATS_BSZ=1 HESSIAN_ACCUM_BSZ=4 \
    BACKWARD_SAMPLES=32 BACKWARD_BSZ=2 FINAL_LAYER_BACKWARD_BSZ=2 \
    BLOCKSIZE=128 W_GROUPSIZE=-1 ACT_ORDER=1 \
    GRAD_LRS=0.0002 FINAL_LAYER_GRAD_LR=0.000001 \
    GRAD_CLIP=5e-5 FINAL_LAYER_GRAD_CLIP=5e-4 \
    GRAD_REFRESH_LOSS=fisher_diag_mse LOSS_SLIDE_WINDOW=0 \
    GLOBAL_LOSS=1 GLOBAL_LOSS_BSZ=1 DP_GLOBAL_SHUFFLE=0 \
    GRAD_LR_LAYER_SCHEDULE=none ALPHA=0.0 ENABLE_GPTQ_PLUS=0 \
    PRE_GD_STEPS=0 PRE_GRAD_LR=0 PRE_FINAL_LAYER_GRAD_LR=none \
    PRE_GRAD_OPTIMIZER=sgd PRE_FINAL_LAYER_GRAD_OPTIMIZER=none \
    GRAD_REG_STRATEGY=none KL_TOPK=20 GRAD_HESSIAN_TOPK=20 \
    SALIENCY_CLIP_PERCENTILE=0.99 A_LOSS_RATIO=1.0 \
    ENABLE_QA_EVAL=0 \
    STAGE2_CPU_MASTER=0 \
    STATIC_CACHE_PATH="$ROOT/cache/nulltest" \
    RDZV_PORT="${RDZV_PORT:-29417}" \
    env "$@"
}

say "null-test queue start (port: warm_adam on origin/zq)"

say "start Adam"
common GRAD_OPTIMIZER=adam FINAL_LAYER_GRAD_OPTIMIZER=adam \
    BASE_EXP=nulltest_adam \
    bash scripts/gptq_plus_lr_sweep.sh "$MODEL" 4 0 \
    --eval_seq_len 256 --eval_datasets wikitext2 \
    > "$ROOT/outputs/runNT_adam.log" 2>&1
say "done Adam exit=$?"
result NT_adam

say "start warm_adam t0=0"
common GRAD_OPTIMIZER=warm_adam FINAL_LAYER_GRAD_OPTIMIZER=warm_adam \
    BASE_EXP=nulltest_warm_t0 \
    bash scripts/gptq_plus_lr_sweep.sh "$MODEL" 4 0 \
    --eval_seq_len 256 --eval_datasets wikitext2 --warm_start_steps 0 \
    > "$ROOT/outputs/runNT_warm_t0.log" 2>&1
say "done warm_adam exit=$?"
result NT_warm_t0

say "null-test queue complete -- the two KL/PPL lines above must match exactly"
