#!/bin/bash
# Smoke comparison for the joint block-GD (transform block-GD so each boundary
# also updates the not-yet-quantised modules with persistent Adam state).
# Uses fisher_diag_mse (needs --global_loss, shrunk config to avoid 8GB OOM).
#
# Usage:
#   QUANT_STOP_LAYER=4 bash scripts/smoke_situation_b.sh baseline
#   QUANT_STOP_LAYER=4 bash scripts/smoke_situation_b.sh joint
set -euo pipefail

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-180}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-600}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_TRUST_REMOTE_CODE=${HF_DATASETS_TRUST_REMOTE_CODE:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODE=${1:-baseline}
QUANT_STOP_LAYER=${QUANT_STOP_LAYER:-4}

MODEL=${MODEL:-/mnt/d/llamaModels/Qwen3-0.6B}
PY=${PY:-$HOME/gptq_plus/.venv/bin/python}

EXTRA=()
if [[ "${MODE}" == "joint" ]]; then
    EXTRA=(--joint_block_gd)
elif [[ "${MODE}" == "cross" ]]; then
    EXTRA=(--cross_module_compensate)
elif [[ "${MODE}" == "all" ]]; then
    EXTRA=(--cross_module_compensate_all)
fi

mkdir -p outputs

echo "=== smoke_situation_b: mode=${MODE} stop_layer=${QUANT_STOP_LAYER} ==="

${PY} -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=localhost:29400 ./ptq.py \
    --model "${MODEL}" \
    --exp "smoke_${MODE}" \
    --dataset wikitext2 --nsamples 4 --seq_len 256 \
    --w_method gptq_plus --w_bits 4 --w_clip --num_groups 4 --blocksize 128 --act_order \
    --kl_topk -1 --bsz 1 --final_layer_stats_bsz 4 --alpha 0.0 \
    --enable_gptq_plus 0 \
    --backward_samples 4 --backward_bsz 4 --final_layer_backward_bsz 4 \
    --g_update_mode block_gd --grad_lr 5e-5 --grad_optimizer adam \
    --grad_refresh_loss fisher_diag_mse \
    --global_loss --global_loss_bsz 1 \
    --grad_clip 5e-5 --final_layer_grad_clip 5e-4 \
    --final_layer_grad_lr 1e-6 \
    --group_parallel_quant none \
    --quant_stop_layer "${QUANT_STOP_LAYER}" \
    --eval_seq_len 1024 \
    --eval_datasets wikitext2 \
    "${EXTRA[@]}" \
    2>&1 | tee "outputs/smoke_${MODE}.log"
