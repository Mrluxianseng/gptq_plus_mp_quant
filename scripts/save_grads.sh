#!/bin/bash

# Input arguments
MODEL_PATH=${1}     # ./modelzoo/Qwen3/Qwen3-0.6B
NUM_GROUPS=${2}     # 4
DEVICE=${3}         # 0

MODEL_NAME=$(basename ${MODEL_PATH})
N_SAMPLES=2048
SEQ_LEN=2048

# Set environment variables
export CUDA_VISIBLE_DEVICES=${DEVICE}

# Execute the script
python ./save_grads.py \
    --model ${MODEL_PATH} \
    --exp save_grads \
    --mode gradients \
    --dataset wikitext2 --nsamples ${N_SAMPLES} --seq_len ${SEQ_LEN} \
    --num_groups ${NUM_GROUPS} \
    --rotate \
