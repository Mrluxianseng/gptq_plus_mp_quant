#!/usr/bin/env bash
# Single-GPU, 256-sample formal-numerics profile for Qwen3-4B W4A4.
# The default captures zero-based layers 10..15 and stops after layer 16;
# callers may override the three layer-boundary environment variables.

set -euo pipefail

REPO_ROOT=/minimax-avatar-new/zhangqian/realq/gptq_plus
CANONICAL_VENV=${REPO_ROOT}/.venv
CONTAINER_FALLBACK_VENV=${REPO_ROOT}/.venv.py312-broken-20260729
VENV_ROOT=${CANONICAL_VENV}
USING_CONTAINER_FALLBACK=0
MODEL_PATH=${REPO_ROOT}/modelzoo/Qwen3/Qwen3-4B
TOKENS_FILE=${REPO_ROOT}/cache/tokens/Qwen3-4B_wikitext2_train_n256_sl2048_seed1.pt
RUN_TAG=${RUN_TAG:-20260807_run1}
QUANT_STOP_LAYER=${QUANT_STOP_LAYER:-16}
# ``-`` rather than ``:-`` lets numerical-control runs explicitly pass an
# empty value to disable layer-scoped profiler calls when NSYS=NVTX=0.
NSYS_CAPTURE_START_LAYER=${NSYS_CAPTURE_START_LAYER-10}
NSYS_CAPTURE_END_LAYER=${NSYS_CAPTURE_END_LAYER-15}
A_LOSS_RATIO=${A_LOSS_RATIO:-0.95}
FISHER_LOSS_TF32=${FISHER_LOSS_TF32:-0}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-sdpa}
HESSIAN_TF32=${HESSIAN_TF32:-1}
FULL_BLOCK_REFRESH=${FULL_BLOCK_REFRESH:-true}
ALIGNMENT_TRACE_PATH=${ALIGNMENT_TRACE_PATH:-}
NSYS=${NSYS:-1}
NVTX=${NVTX:-1}
EXP_NAME=realq_qwen3_4b_w4a4_n256_nsys_layer${NSYS_CAPTURE_START_LAYER}_${NSYS_CAPTURE_END_LAYER}_${RUN_TAG}
OUTPUT_ROOT=${REPO_ROOT}/output/realq_nsys_qwen3_4b_w4a4_20260807

if [[ ${HOSTNAME:-$(hostname)} != j-*-master-0 ]]; then
    echo "ERROR: this profile must run inside a Canoe debug pod; host=${HOSTNAME:-$(hostname)}" >&2
    exit 78
fi
if [[ ! -x ${CANONICAL_VENV}/bin/python ]]; then
    # The current Canoe image has no /usr/local/bin/python3, which is the
    # canonical venv's recorded base interpreter.  Reuse the already-installed
    # repository-local Python 3.12 environment rather than system Python.
    if [[ ! -x ${CONTAINER_FALLBACK_VENV}/bin/python ]]; then
        echo "ERROR: neither repository Python environment is executable." >&2
        exit 78
    fi
    VENV_ROOT=${CONTAINER_FALLBACK_VENV}
    USING_CONTAINER_FALLBACK=1
    echo "WARNING: canonical .venv base interpreter is absent in this Canoe image; using ${VENV_ROOT}." >&2
fi
if [[ ! -f ${TOKENS_FILE} ]]; then
    echo "ERROR: frozen calibration artifact is missing: ${TOKENS_FILE}" >&2
    exit 78
fi

if [[ ${USING_CONTAINER_FALLBACK} == 1 ]]; then
    # This fallback was preserved by renaming an older venv, so its generated
    # activate file still names the canonical .venv.  Activate it explicitly.
    export VIRTUAL_ENV=${VENV_ROOT}
    export PATH=${VENV_ROOT}/bin:${PATH}
    hash -r
else
    source "${VENV_ROOT}/bin/activate"
fi
# The Canoe image exports its system Torch 2.6 library directory globally.
# CUDA extensions compiled against the repository's frozen Torch 2.9.1 must
# resolve libc10/libtorch from the active venv first (not merely import the
# Python package from it).
for torch_lib_dir in "${VENV_ROOT}"/lib/python*/site-packages/torch/lib; do
    if [[ -d ${torch_lib_dir} ]]; then
        export LD_LIBRARY_PATH=${torch_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
        break
    fi
done
export REALQ_ACTIVE_VENV=${VENV_ROOT}
python - <<'PY'
import os
import sys
import torch
import transformers

assert sys.prefix == os.environ["REALQ_ACTIVE_VENV"], (sys.prefix, os.environ["REALQ_ACTIVE_VENV"])
assert torch.__version__ == "2.9.1+cu128", torch.__version__
assert transformers.__version__ == "4.56.2", transformers.__version__
print(f"python={sys.version.split()[0]}")
print(f"venv={sys.prefix}")
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
PY

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/nsight" "${OUTPUT_ROOT}/static_cache"

export HF_HOME=${REPO_ROOT}/datasets/lm_eval_hf_cache
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
# This debug pod intentionally requests no RDMA, but Canoe still injects
# bond1/RDMA settings.  REAL-Q initializes a world-one process group, so force
# bootstrap to loopback and keep NCCL away from the absent IB devices.
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1

export DEVICE=${DEVICE:-0}
export N_GPUS=1
export NSAMPLES=256
export SEQ_LEN=2048
export BSZ=128
export W_BITS=4
export NUM_GROUPS=4
export GRAD_LR=8.5e-6
export CPU_MASTER=0
export FSDP=0
export QUANT_STOP_LAYER
export NSYS_CAPTURE_START_LAYER
export NSYS_CAPTURE_END_LAYER
export NSYS
export NVTX
export NSYS_MODE=rank0_cuda
export NSYS_TRACE_RANK0=cuda,nvtx
export NSYS_WAIT=primary
export EXP_NAME
export OUTPUT_ROOT
export NSYS_OUTPUT=${OUTPUT_ROOT}/nsight/${EXP_NAME}

if [[ ${FISHER_LOSS_TF32} == 1 ]]; then
    TF32_PROFILE_SITE_DIR=${REPO_ROOT}/tools/profile_overrides/fisher_loss_tf32
    if [[ ! -f ${TF32_PROFILE_SITE_DIR}/sitecustomize.py ]]; then
        echo "ERROR: Fisher-loss TF32 profile override is missing: ${TF32_PROFILE_SITE_DIR}/sitecustomize.py" >&2
        exit 78
    fi
    export REALQ_PROFILE_FISHER_LOSS_TF32=1
    export PYTHONPATH=${TF32_PROFILE_SITE_DIR}${PYTHONPATH:+:${PYTHONPATH}}
fi

echo "run_tag=${RUN_TAG}"
echo "host=${HOSTNAME:-$(hostname)} device=${DEVICE}"
echo "tokens_sha256=e87b6d8faf62ab6d481fc36ebb5fd091c0455aaa1ca727d52ba1d41563d84a42"
echo "nsys_output=${NSYS_OUTPUT}.nsys-rep"
echo "a_loss_ratio=${A_LOSS_RATIO} fisher_loss_tf32=${FISHER_LOSS_TF32} attention_backend=${ATTENTION_BACKEND} hessian_tf32=${HESSIAN_TF32} full_block_refresh=${FULL_BLOCK_REFRESH} fused_block_adam=true"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

ALIGNMENT_TRACE_ARG=()
if [[ -n ${ALIGNMENT_TRACE_PATH} ]]; then
    ALIGNMENT_TRACE_ARG=(--alignment_trace_path "${ALIGNMENT_TRACE_PATH}")
fi

exec bash realq/scripts/profile.sh "${MODEL_PATH}" "${DEVICE}" \
    --w_groupsize 128 \
    --w_asym false \
    --w_clip true \
    --a_bits 4 \
    --k_bits 4 \
    --v_bits 4 \
    --a_clip_ratio 0.9 \
    --k_clip_ratio 0.9 \
    --v_clip_ratio 0.9 \
    --act_quant_aware_gptq true \
    --k_cache_quant_aware_gptq true \
    --a_loss_ratio "${A_LOSS_RATIO}" \
    --a_loss_clip_scope local_backward_chunk \
    --attention_backend "${ATTENTION_BACKEND}" \
    --hessian_tf32 "${HESSIAN_TF32}" \
    --full_block_refresh "${FULL_BLOCK_REFRESH}" \
    --fused_block_adam true \
    --percdamp 0.01 \
    --act_order true \
    --blocksize 128 \
    --group_parallel_quant rank \
    --global_loss_bsz 8 \
    --hessian_accum_bsz 64 \
    --backward_samples 32 \
    --backward_bsz 32 \
    --final_layer_backward_bsz 8 \
    --grad_clip 1.0 \
    --final_layer_grad_lr 1e-5 \
    --grad_lr_layer_schedule none \
    --saliency_clip_percentile 0.99 \
    --grad_hessian_topk -1 \
    --kl_topk -1 \
    --loss_slide_window true \
    --rotate true \
    --seed 1 \
    --rotation_seed 0 \
    --refresh_seed 0 \
    --tokens_cache_file "${TOKENS_FILE}" \
    --static_cache_path "${OUTPUT_ROOT}/static_cache" \
    --log_column_block_loss true \
    "${ALIGNMENT_TRACE_ARG[@]}"
