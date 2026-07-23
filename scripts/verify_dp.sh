#!/bin/bash
#
# scripts/verify_dp.sh
#
# Compares single-GPU vs multi-GPU DP runs of the GPTQ+ pipeline.
#
# Methodology:
#   1. Run verify_gptq_plus.py with 1 rank and save the resulting quantized
#      weights as `gptq_plus_1gpu.pt` (treated as the reference).
#   2. Run verify_gptq_plus.py with N ranks (via torchrun) and save as
#      `gptq_plus_${N}gpu.pt`.
#   3. Use compare_snapshots.py to tolerant-diff the two snapshots.
#
# To avoid shuffle-induced divergence between the two runs, this harness
# sets `backward_samples == nsamples` (everyone uses every sample every time)
# and drops both to a small value (32 by default) so the run is quick. With
# backward_samples == nsamples the BackwardSampleScheduler never produces a
# random subset — each refresh uses the full local shard — so the only
# remaining source of DP-vs-serial drift is FP-summation order across the
# allreduce boundary, which sits well under the default atol.
#
# Usage on a dual-GPU host:
#   bash scripts/verify_dp.sh ./modelzoo/Qwen3/Qwen3-0.6B
#
# Environment overrides (all optional, defaults are tuned for quick runs):
#   N_GPUS=2                  number of ranks for the DP leg
#   NSAMPLES=32               total calibration samples (must be divisible by N_GPUS)
#   BACKWARD_SAMPLES=32       refresh sample budget (must equal NSAMPLES)
#   SEQ_LEN=512               tokens per sample
#   BSZ=4                     stats/forward batch size (global; per-rank = BSZ/N)
#   BACKWARD_BSZ=4            refresh batch size (global; per-rank = /N)
#   BLOCKSIZE=256             fasterquant block size
#   QUANT_STOP_LAYER=1        stop after this layer (keep small for quick turnaround)
#   DEVICE=0,1                CUDA_VISIBLE_DEVICES for the DP leg; first id also
#                             used for the 1-GPU leg
#   ATOL=1e-3 RTOL=1e-4       tolerance for compare_snapshots.py
#   SNAPSHOT_DIR=./outputs/verify   where to park snapshot .pt files
#   EXP_NAME=verify_dp        experiment dir name (under outputs/<model>/)
#   ENABLE_DEBUG=0            set to 1 to enable the in-run bit-exact assertion
#                             between ranks inside fasterquant (debug only)

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <MODEL_PATH>"
    exit 1
fi
MODEL_PATH=$1

N_GPUS=${N_GPUS:-2}
NSAMPLES=${NSAMPLES:-32}
SEQ_LEN=${SEQ_LEN:-512}
BACKWARD_SAMPLES=${BACKWARD_SAMPLES:-${NSAMPLES}}
BSZ=${BSZ:-4}
BACKWARD_BSZ=${BACKWARD_BSZ:-4}
FINAL_LAYER_STATS_BSZ=${FINAL_LAYER_STATS_BSZ:-${BSZ}}
FINAL_LAYER_BACKWARD_BSZ=${FINAL_LAYER_BACKWARD_BSZ:-${BACKWARD_BSZ}}
BLOCKSIZE=${BLOCKSIZE:-256}
ALPHA=${ALPHA:-0.03}
GRAD_LR=${GRAD_LR:-0.0001}
FINAL_LAYER_GRAD_LR=${FINAL_LAYER_GRAD_LR:-0.01}
GRAD_OPTIMIZER=${GRAD_OPTIMIZER:-adam}
FINAL_LAYER_GRAD_OPTIMIZER=${FINAL_LAYER_GRAD_OPTIMIZER:-sgd}
GRAD_CLIP=${GRAD_CLIP:-1.0}
GRAD_REFRESH_LOSS=${GRAD_REFRESH_LOSS:-fisher_diag_mse}
G_UPDATE_MODE=${G_UPDATE_MODE:-block_gd}
PRE_GD_STEPS=${PRE_GD_STEPS:-0}
PRE_GRAD_LR=${PRE_GRAD_LR:-0.00003}
PRE_GRAD_OPTIMIZER=${PRE_GRAD_OPTIMIZER:-adam}
PRE_FINAL_LAYER_GRAD_LR=${PRE_FINAL_LAYER_GRAD_LR:-0.3}
PRE_FINAL_LAYER_GRAD_OPTIMIZER=${PRE_FINAL_LAYER_GRAD_OPTIMIZER:-sgd}
GRAD_REG_STRATEGY=${GRAD_REG_STRATEGY:-none}
GRAD_REG_LAMBDA=${GRAD_REG_LAMBDA:-0.01}
GRAD_GATE_FLOOR=${GRAD_GATE_FLOOR:-0.01}
GRAD_GATE_SHARPNESS=${GRAD_GATE_SHARPNESS:-5.0}
GRAD_GATE_SINE_AMP=${GRAD_GATE_SINE_AMP:-0.0005}
GRAD_HESSIAN_TOPK=${GRAD_HESSIAN_TOPK:--1}
KL_TOPK=${KL_TOPK:--1}
NUM_GROUPS=${NUM_GROUPS:-4}
FISHER_NUM_GROUPS=${FISHER_NUM_GROUPS:-512}
SECOND_ORDER_SCALE=${SECOND_ORDER_SCALE:-1.0}
GLOBAL_LOSS=${GLOBAL_LOSS:-1}
GLOBAL_LOSS_BSZ=${GLOBAL_LOSS_BSZ:-${BSZ}}
QUANT_STOP_LAYER=${QUANT_STOP_LAYER:-1}
EXP_NAME=${EXP_NAME:-verify_dp}
CACHE_DIR=${CACHE_DIR:-./cache}
OUTPUT_ROOT=${OUTPUT_ROOT:-./outputs}
SNAPSHOT_DIR=${SNAPSHOT_DIR:-./outputs/verify}
ATOL=${ATOL:-1e-3}
RTOL=${RTOL:-1e-4}
DEVICE=${DEVICE:-0,1}
ENABLE_DEBUG=${ENABLE_DEBUG:-0}
RDZV_PORT=${RDZV_PORT:-29501}

mkdir -p "${SNAPSHOT_DIR}"
SNAP_1GPU="${SNAPSHOT_DIR}/gptq_plus_1gpu_n${NSAMPLES}_b${BACKWARD_SAMPLES}.pt"
SNAP_NGPU="${SNAPSHOT_DIR}/gptq_plus_${N_GPUS}gpu_n${NSAMPLES}_b${BACKWARD_SAMPLES}.pt"

# Pre-flight: divisibility constraints that make DP well-defined.
if (( NSAMPLES % N_GPUS != 0 )); then
    echo "ERROR: NSAMPLES ($NSAMPLES) must be divisible by N_GPUS ($N_GPUS)"; exit 2
fi
if (( BACKWARD_SAMPLES != NSAMPLES )); then
    echo "WARNING: BACKWARD_SAMPLES ($BACKWARD_SAMPLES) != NSAMPLES ($NSAMPLES); shuffle subsampling will"
    echo "         introduce divergence between the 1-GPU and ${N_GPUS}-GPU runs. Expect larger diffs."
fi
if (( BACKWARD_SAMPLES % N_GPUS != 0 )); then
    echo "ERROR: BACKWARD_SAMPLES ($BACKWARD_SAMPLES) must be divisible by N_GPUS ($N_GPUS)"; exit 2
fi
if (( BSZ % N_GPUS != 0 )); then
    echo "ERROR: BSZ ($BSZ) must be divisible by N_GPUS ($N_GPUS)"; exit 2
fi
if (( BACKWARD_BSZ % N_GPUS != 0 )); then
    echo "ERROR: BACKWARD_BSZ ($BACKWARD_BSZ) must be divisible by N_GPUS ($N_GPUS)"; exit 2
fi

# Share env vars with verify_gptq_plus.py.
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/tmp/hf_datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-/tmp/hf_hub}

DEBUG_FLAG=()
if [[ "${ENABLE_DEBUG}" == "1" ]]; then
    DEBUG_FLAG=(--enable_debug)
fi

GLOBAL_LOSS_ARGS=()
if [[ "${GLOBAL_LOSS}" == "1" ]]; then
    GLOBAL_LOSS_ARGS=(--global_loss --global_loss_bsz "${GLOBAL_LOSS_BSZ}")
else
    GLOBAL_LOSS_ARGS=(--no_global_loss)
fi

COMMON_ARGS=(
    --model "${MODEL_PATH}"
    --exp "${EXP_NAME}"
    --output_dir "${OUTPUT_ROOT}"
    --cache_dir "${CACHE_DIR}"
    --dataset neuralmagic
    --nsamples "${NSAMPLES}"
    --seq_len "${SEQ_LEN}"
    --w_method gptq_plus
    --w_bits 4
    --w_clip
    --num_groups "${NUM_GROUPS}"
    --fisher_num_groups "${FISHER_NUM_GROUPS}"
    --kl_topk "${KL_TOPK}"
    --bsz "${BSZ}"
    --final_layer_stats_bsz "${FINAL_LAYER_STATS_BSZ}"
    --alpha "${ALPHA}"
    --blocksize "${BLOCKSIZE}"
    --backward_samples "${BACKWARD_SAMPLES}"
    --backward_bsz "${BACKWARD_BSZ}"
    --final_layer_backward_bsz "${FINAL_LAYER_BACKWARD_BSZ}"
    --g_update_mode "${G_UPDATE_MODE}"
    --grad_lr "${GRAD_LR}"
    --grad_optimizer "${GRAD_OPTIMIZER}"
    --grad_refresh_loss "${GRAD_REFRESH_LOSS}"
    "${GLOBAL_LOSS_ARGS[@]}"
    --final_layer_grad_optimizer "${FINAL_LAYER_GRAD_OPTIMIZER}"
    --grad_clip "${GRAD_CLIP}"
    --final_layer_grad_lr "${FINAL_LAYER_GRAD_LR}"
    --grad_hessian_topk "${GRAD_HESSIAN_TOPK}"
    --pre_gd_steps "${PRE_GD_STEPS}"
    --pre_grad_lr "${PRE_GRAD_LR}"
    --pre_grad_optimizer "${PRE_GRAD_OPTIMIZER}"
    --pre_final_layer_grad_lr "${PRE_FINAL_LAYER_GRAD_LR}"
    --pre_final_layer_grad_optimizer "${PRE_FINAL_LAYER_GRAD_OPTIMIZER}"
    --no_pre_clip
    --grad_reg_strategy "${GRAD_REG_STRATEGY}"
    --grad_reg_lambda "${GRAD_REG_LAMBDA}"
    --grad_gate_floor "${GRAD_GATE_FLOOR}"
    --grad_gate_sharpness "${GRAD_GATE_SHARPNESS}"
    --grad_gate_sine_amp "${GRAD_GATE_SINE_AMP}"
    --second_order_scale "${SECOND_ORDER_SCALE}"
    --quant_stop_layer "${QUANT_STOP_LAYER}"
    --act_order
    --verify_tol_abs "${ATOL}"
    --verify_tol_rel "${RTOL}"
)

FIRST_DEVICE=${DEVICE%%,*}

echo "============================================================"
echo "[verify_dp] model=${MODEL_PATH}"
echo "[verify_dp] N_GPUS=${N_GPUS}  devices=${DEVICE}  rdzv_port=${RDZV_PORT}"
echo "[verify_dp] NSAMPLES=${NSAMPLES}  BACKWARD_SAMPLES=${BACKWARD_SAMPLES}"
echo "[verify_dp] BSZ=${BSZ}  BACKWARD_BSZ=${BACKWARD_BSZ}"
echo "[verify_dp] stop layer=${QUANT_STOP_LAYER}"
echo "[verify_dp] snapshots: ${SNAP_1GPU}  ${SNAP_NGPU}"
echo "[verify_dp] tolerance: atol=${ATOL} rtol=${RTOL}"
echo "============================================================"

echo ""
echo "[verify_dp] (1/3) running 1-GPU reference run on device ${FIRST_DEVICE}..."
CUDA_VISIBLE_DEVICES=${FIRST_DEVICE} python verify_gptq_plus.py \
    --verify_mode save \
    --verify_output "${SNAP_1GPU}" \
    "${DEBUG_FLAG[@]}" \
    "${COMMON_ARGS[@]}"

echo ""
echo "[verify_dp] (2/3) running ${N_GPUS}-GPU DP run on devices ${DEVICE}..."
CUDA_VISIBLE_DEVICES=${DEVICE} torchrun \
    --nnodes=1 \
    --nproc_per_node=${N_GPUS} \
    --rdzv_endpoint=localhost:${RDZV_PORT} \
    verify_gptq_plus.py \
    --verify_mode save \
    --verify_output "${SNAP_NGPU}" \
    "${DEBUG_FLAG[@]}" \
    "${COMMON_ARGS[@]}"

echo ""
echo "[verify_dp] (3/3) comparing snapshots..."
python compare_snapshots.py "${SNAP_1GPU}" "${SNAP_NGPU}" --atol "${ATOL}" --rtol "${RTOL}"
echo "[verify_dp] done"
