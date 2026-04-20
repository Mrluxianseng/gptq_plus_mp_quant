#!/bin/bash

# Usage: bash scripts/analyze_grad_cosine.sh <MODEL_PATH> <DEVICE> [TARGET_LAYERS]
# Example: bash scripts/analyze_grad_cosine.sh ./modelzoo/Qwen3/Qwen3-0.6B 0 "1,5,10,15"

MODEL_PATH=${1}
DEVICE=${2}
TARGET_LAYERS=${3:-"7,14,20,26"}

MODEL_NAME=$(basename ${MODEL_PATH})
N_SAMPLES=128
SEQ_LEN=2048
MEASURE_SAMPLES=128
MEASURE_BSZ=4

export CUDA_VISIBLE_DEVICES=${DEVICE}

python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=localhost:2950${DEVICE} ./analyze_grad_cosine.py \
    --model ${MODEL_PATH} \
    --exp grad_cosine \
    --dataset wikitext2 --nsamples ${N_SAMPLES} --seq_len ${SEQ_LEN} \
    --w_method gptaq --w_bits 4 --w_clip --act_order \
    --rotate \
    --skip_eval \
    --kl_topk 20 --grad_hessian_topk 20 \
    --num_groups 4 --fisher_num_groups 512 --bsz 64 --global_loss_bsz 8 \
    --target_layers ${TARGET_LAYERS} \
    --measure_samples ${MEASURE_SAMPLES} --measure_batch_size ${MEASURE_BSZ}
