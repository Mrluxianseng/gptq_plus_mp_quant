#!/bin/bash
#
# adam vs warm_adam on upstream (origin/zq), at upstream's own defaults.
#
# Baseline is the current upstream adam, not the old fork's. In particular the
# reverse-cosine layer schedule stays ON (upstream's argparse default) and the
# group-parallel batched path stays on, because that is what upstream actually
# runs. No attempt is made to reproduce the old branch's flat-LR collapse.
#
# warm_adam takes its default t0 = nsamples / backward_samples = 1 here. The
# port is verified: `--warm_start_steps 0` reproduced adam exactly
# (KL 1.14e-01 / PPL 39.67 both, outputs/nulltest_queue.log).
#
# Everything except grad_lr and grad_optimizer is held fixed. Small config --
# this is for the shape of the two curves, not for paper-comparable numbers.
#
# Known-good reference point already measured at lr=2e-4: adam = 1.14e-01 / 39.67.
set -u
ROOT=/mnt/d/gptq_plus_realq
VENV=/mnt/d/gptq_plus/.venv_wsl
MODEL=${MODEL:-/mnt/d/llamaModels/Qwen3-0.6B}
cd "$ROOT" || exit 1
source "$VENV/bin/activate"
mkdir -p "$ROOT/outputs"
LOG="$ROOT/outputs/compare_queue.log"
exec >> "$LOG" 2>&1
say() { echo "=== $(date '+%F %T') | $* ==="; }

# Same STATIC_CACHE_PATH as the null test: the Stage-0 saliency/Fisher cache key
# does not depend on grad_lr or the optimizer, so every run below reuses one
# precompute. That also means all runs share bit-identical Stage-0 inputs, which
# is what makes the optimizer the only moving part.
common() {
    OUTPUT_ROOT="$ROOT/outputs" \
    DATASET=wikitext2 N_SAMPLES=32 SEQ_LEN=256 \
    BSZ=1 FINAL_LAYER_STATS_BSZ=1 HESSIAN_ACCUM_BSZ=4 \
    BACKWARD_SAMPLES=32 BACKWARD_BSZ=2 FINAL_LAYER_BACKWARD_BSZ=2 \
    BLOCKSIZE=128 W_GROUPSIZE=-1 ACT_ORDER=1 \
    FINAL_LAYER_GRAD_LR=0.000001 \
    GRAD_CLIP=5e-5 FINAL_LAYER_GRAD_CLIP=5e-4 \
    GRAD_REFRESH_LOSS=fisher_diag_mse LOSS_SLIDE_WINDOW=0 \
    GLOBAL_LOSS=1 GLOBAL_LOSS_BSZ=1 DP_GLOBAL_SHUFFLE=0 \
    ALPHA=0.0 ENABLE_GPTQ_PLUS=0 \
    PRE_GD_STEPS=0 PRE_GRAD_LR=0 PRE_FINAL_LAYER_GRAD_LR=none \
    PRE_GRAD_OPTIMIZER=sgd PRE_FINAL_LAYER_GRAD_OPTIMIZER=none \
    GRAD_REG_STRATEGY=none KL_TOPK=20 GRAD_HESSIAN_TOPK=20 \
    SALIENCY_CLIP_PERCENTILE=0.99 A_LOSS_RATIO=1.0 \
    ENABLE_QA_EVAL=0 STAGE2_CPU_MASTER=0 \
    STATIC_CACHE_PATH="$ROOT/cache/nulltest" \
    RDZV_PORT="${RDZV_PORT:-29417}" \
    env "$@"
}

run() {
    local opt="$1" lr="$2" tag="$3"
    shift 3
    say "start $tag (opt=$opt lr=$lr)"
    common GRAD_OPTIMIZER="$opt" FINAL_LAYER_GRAD_OPTIMIZER="$opt" \
        GRAD_LRS="$lr" BASE_EXP="cmp_${tag}" \
        bash scripts/gptq_plus_lr_sweep.sh "$MODEL" 4 0 \
        --eval_seq_len 256 --eval_datasets wikitext2 "$@" \
        > "$ROOT/outputs/runCMP_${tag}.log" 2>&1
    say "done $tag exit=$?"
    grep -aoE 'KL&PPL on wikitext2: .*' "$ROOT/outputs/runCMP_${tag}.log" | tail -1
}

say "adam vs warm_adam comparison start (upstream defaults, schedule=cosine)"

for lr in 0.00005 0.001 0.005; do
    run adam "$lr" "adam_${lr}"
done
for lr in 0.00005 0.0002 0.001 0.005; do
    run warm_adam "$lr" "warm_${lr}"
done

say "comparison complete"
