#!/bin/bash
# Ablation: bit-allocation granularity comparison for mixed-precision quantization
# Compares module-level vs layer-level vs type-level granularity at 3.5-bit target.
# All experiments share the same static saliency cache (precomputed once).
#
# Usage: ROOT_DIR=/path/to/gptq_plus bash scripts/run_mp_granularity_ablation.sh

set -e

ROOT_DIR=${ROOT_DIR:-/root/lh/gptq_plus}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-0.6B}
DEVICE=${DEVICE:-0}
NSAMPLES=${NSAMPLES:-512}
SEQ_LEN=${SEQ_LEN:-2048}
GLOBAL_LOSS_BSZ=${GLOBAL_LOSS_BSZ:-4}
BACKWARD_SAMPLES=${BACKWARD_SAMPLES:-512}

export CUDA_VISIBLE_DEVICES=${DEVICE}

cd "${ROOT_DIR}" || exit 1

if [ -f "${ROOT_DIR}/.venv_wsl/bin/torchrun" ]; then
    TORCHRUN="${ROOT_DIR}/.venv_wsl/bin/torchrun"
else
    TORCHRUN=$(which torchrun)
fi

# Shared static cache so all granularity runs reuse the same precomputed saliency/fisher
STATIC_CACHE="${ROOT_DIR}/cache/saliency_mp_granularity_ablation"

COMMON_ARGS="
    --model ${MODEL_PATH}
    --w_method gptq_plus --w_clip
    --dataset wikitext2 --nsamples ${NSAMPLES} --seq_len ${SEQ_LEN}
    --global_loss --global_loss_bsz ${GLOBAL_LOSS_BSZ}
    --grad_refresh_loss fisher_diag_mse
    --g_update_mode block_gd
    --alpha 0
    --grad_optimizer adam --grad_lr 0.0002
    --final_layer_grad_lr 0.000002
    --grad_clip 5e-5 --final_layer_grad_clip 5e-4
    --num_groups 4 --bsz 1
    --backward_samples ${BACKWARD_SAMPLES} --backward_bsz 1
    --eval_seq_len ${SEQ_LEN}
    --mixed_precision
    --mp_target_avg_bits 3.5 --mp_high_bits 4 --mp_low_bits 3
    --mp_saliency_metric fisher_mean
    --static_cache_path ${STATIC_CACHE}
    --eval_datasets wikitext2 ultrachat_2k numinamath
"

echo "================================================"
echo "[1/3] Granularity: module (finest, default)"
echo "================================================"
${TORCHRUN} --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=localhost:29410 ptq.py \
    ${COMMON_ARGS} \
    --exp mp_gran_module \
    --mp_granularity module \
    --w_bits 4

echo "================================================"
echo "[2/3] Granularity: layer (coarser)"
echo "================================================"
${TORCHRUN} --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=localhost:29411 ptq.py \
    ${COMMON_ARGS} \
    --exp mp_gran_layer \
    --mp_granularity layer \
    --w_bits 4

echo "================================================"
echo "[3/3] Granularity: type (coarsest: by module role)"
echo "================================================"
${TORCHRUN} --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=localhost:29412 ptq.py \
    ${COMMON_ARGS} \
    --exp mp_gran_type \
    --mp_granularity type \
    --w_bits 4

echo "================================================"
echo "All granularity ablations done."
echo "Results in: ${ROOT_DIR}/outputs/$(basename ${MODEL_PATH})/"
echo "================================================"
