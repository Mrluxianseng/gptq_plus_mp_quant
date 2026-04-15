#!/bin/bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <MODEL_PATH> [extra analyze_quant_profile.py args ...]"
    echo "Example: $0 ./modelzoo/Qwen3/Qwen3-0.6B --target_layers 0,14,27"
    exit 1
fi

MODEL_PATH=${1}
shift

N_SAMPLES=${N_SAMPLES:-128}
SEQ_LEN=${SEQ_LEN:-512}
NUM_GROUPS=${NUM_GROUPS:-4}
BSZ=${BSZ:-4}
ALPHA=${ALPHA:-0.05}
BLOCKSIZE=${BLOCKSIZE:-128}
BACKWARD_SAMPLES=${BACKWARD_SAMPLES:-4}
BACKWARD_BSZ=${BACKWARD_BSZ:-4}
GRAD_LR=${GRAD_LR:-1.0}
SECOND_ORDER_SCALE=${SECOND_ORDER_SCALE:-1.0}
G_UPDATE_MODE=${G_UPDATE_MODE:-block_gd}
TARGET_LAYERS=${TARGET_LAYERS:-all}
TARGET_MODULES=${TARGET_MODULES:-all}
EXP_NAME=${EXP_NAME:-quant_profile_quick}

export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/tmp/hf_datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-/tmp/hf_hub}

python analyze_quant_profile.py \
    --model "${MODEL_PATH}" \
    --exp "${EXP_NAME}" \
    --dataset neuralmagic \
    --nsamples "${N_SAMPLES}" \
    --seq_len "${SEQ_LEN}" \
    --num_groups "${NUM_GROUPS}" \
    --bsz "${BSZ}" \
    --alpha "${ALPHA}" \
    --blocksize "${BLOCKSIZE}" \
    --backward_samples "${BACKWARD_SAMPLES}" \
    --backward_bsz "${BACKWARD_BSZ}" \
    --g_update_mode "${G_UPDATE_MODE}" \
    --grad_lr "${GRAD_LR}" \
    --second_order_scale "${SECOND_ORDER_SCALE}" \
    --target_layers "${TARGET_LAYERS}" \
    --target_modules "${TARGET_MODULES}" \
    --act_order \
    --w_clip \
    "$@"
