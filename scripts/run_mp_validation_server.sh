#!/bin/bash
# Server-side mixed precision validation: W4, W3, mixed W4/3 avg 3.5bit
# Full scale: 512 calibration samples × 2048 seq_len, three eval datasets
# Run order: mixed first, then W4, then W3

set -e

# export HF_ENDPOINT=https://hf-mirror.com  # uncomment for China servers
unset TRANSFORMERS_OFFLINE
unset HF_DATASETS_OFFLINE

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
"

echo "========================================"
echo "[1/3] Mixed precision W4/3 avg 3.5bit"
echo "========================================"
${TORCHRUN} --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=localhost:29400 ptq.py \
    ${COMMON_ARGS} \
    --exp mp_mixed_3p5bit \
    --w_bits 4 \
    --mixed_precision \
    --mp_target_avg_bits 3.5 \
    --mp_high_bits 4 \
    --mp_low_bits 3 \
    --eval_datasets wikitext2 ultrachat_2k numinamath

echo "========================================"
echo "[2/3] Uniform W4"
echo "========================================"
${TORCHRUN} --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=localhost:29401 ptq.py \
    ${COMMON_ARGS} \
    --exp mp_uniform_w4 \
    --w_bits 4 \
    --eval_datasets wikitext2 ultrachat_2k numinamath

echo "========================================"
echo "[3/3] Uniform W3"
echo "========================================"
${TORCHRUN} --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=localhost:29402 ptq.py \
    ${COMMON_ARGS} \
    --exp mp_uniform_w3 \
    --w_bits 3 \
    --eval_datasets wikitext2 ultrachat_2k numinamath

echo "========================================"
echo "All experiments done."
echo "Results in: ${ROOT_DIR}/outputs/Qwen3-0.6B/"
echo "========================================"
