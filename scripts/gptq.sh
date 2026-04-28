#!/bin/bash

# Input arguments
MODEL_PATH=${1}     # ./modelzoo/Qwen3/Qwen3-0.6B
DEVICE=${2}         # 0

MODEL_NAME=$(basename ${MODEL_PATH})
N_SAMPLES=2048
SEQ_LEN=2048

# Set environment variables
export CUDA_VISIBLE_DEVICES=${DEVICE}

# Execute the distributed run
python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=localhost:2940${DEVICE} ./ptq.py \
    --model ${MODEL_PATH} \
    --exp gptq \
    --dataset wikitext2 --nsamples ${N_SAMPLES} --seq_len ${SEQ_LEN} \
    --w_method gptq --w_bits 4 --w_clip --act_order \
    --lm_eval --lm_eval_batch_size 32 \
    --rotate \
    --a_clip_ratio 0.9 --k_clip_ratio 0.9 --k_clip_ratio 0.9 \
    --w_groupsize 128 \
    --a_bits 4 --k_bits 4 --v_bits 4 \
