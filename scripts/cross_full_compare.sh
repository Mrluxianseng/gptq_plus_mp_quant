#!/bin/bash
# Full-config cross vs baseline: no QUANT_STOP_LAYER (all 28 layers quantised),
# nsamples raised from the smoke's 4 to 32. Everything else matches the smoke
# that gave cross = -0.4% (grad_lr 5e-5, fisher_diag_mse, global_loss, blocksize
# 128, act_order, group_parallel none).
#
# Usage (WSL, from the worktree root):
#   nohup bash scripts/cross_full_compare.sh > outputs/cross_full_queue.log 2>&1 &
#   tail -F outputs/cross_full_queue.log
set -euo pipefail

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-180}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-600}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_TRUST_REMOTE_CODE=${HF_DATASETS_TRUST_REMOTE_CODE:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

N_SAMPLES=${N_SAMPLES:-32}
SEQ_LEN=${SEQ_LEN:-256}
BACKWARD_SAMPLES=${BACKWARD_SAMPLES:-32}

MODEL=${MODEL:-/mnt/d/llamaModels/Qwen3-0.6B}
PY=${PY:-$HOME/gptq_plus/.venv/bin/python}

mkdir -p outputs

for MODE in baseline cross; do
    EXTRA=()
    if [[ "${MODE}" == "cross" ]]; then
        EXTRA=(--cross_module_compensate)
    fi
    echo "===== full ${MODE} start $(date +%H:%M:%S) nsamples=${N_SAMPLES} seq_len=${SEQ_LEN} ====="

    ${PY} -m torch.distributed.run \
        --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=localhost:29400 ./ptq.py \
        --model "${MODEL}" \
        --exp "full_${MODE}" \
        --dataset wikitext2 --nsamples "${N_SAMPLES}" --seq_len "${SEQ_LEN}" \
        --w_method gptq_plus --w_bits 4 --w_clip --num_groups 4 --blocksize 128 --act_order \
        --kl_topk -1 --bsz 1 --final_layer_stats_bsz 4 --alpha 0.0 \
        --enable_gptq_plus 0 \
        --backward_samples "${BACKWARD_SAMPLES}" --backward_bsz 4 --final_layer_backward_bsz 4 \
        --g_update_mode block_gd --grad_lr 5e-5 --grad_optimizer adam \
        --grad_refresh_loss fisher_diag_mse \
        --global_loss --global_loss_bsz 1 \
        --grad_clip 5e-5 --final_layer_grad_clip 5e-4 \
        --final_layer_grad_lr 1e-6 \
        --group_parallel_quant none \
        --eval_seq_len 1024 \
        --eval_datasets wikitext2 \
        "${EXTRA[@]}" \
        2>&1 | tee "outputs/full_${MODE}.log"

    echo "===== full ${MODE} done  $(date +%H:%M:%S) ====="
done

echo "===== full baseline+cross done ====="
