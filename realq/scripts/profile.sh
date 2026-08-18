#!/usr/bin/env bash
# Quick Nsight / NVTX profile of the RealQ pipeline.
#
# Mirrors the structure of scripts/quant_profile_quick.sh (which targets the
# legacy gptq_plus path) but trimmed to the realq config surface — only the
# knobs realq's `Config` actually consumes are exposed. Everything else falls
# through to realq.config defaults via the `"$@"` passthrough at the end.
#
# Crucially keeps the multi-rank `nsys profile` wrapper pattern: nsys MUST be
# attached per-rank (torchrun --no-python → bash wrapper → exec nsys python),
# not around the launcher. Wrapping `nsys profile torchrun ...` injects CUPTI
# into 4 worker processes that then race on NCCL teardown and SIGSEGV
# non-deterministically.
#
# Usage:
#   bash realq/scripts/profile.sh [MODEL_PATH] [DEVICE] [extra realq.ptq args ...]
# Examples:
#   bash realq/scripts/profile.sh                      # 0.6B single-GPU defaults
#   DEVICE=0,1,2,3 bash realq/scripts/profile.sh modelzoo/Qwen/Qwen3-8B
#   QUANT_STOP_LAYER=1 NSAMPLES=128 bash realq/scripts/profile.sh
#   NSYS_MODE=all_nvtx DEVICE=0,1,2,3 bash realq/scripts/profile.sh MODEL
#
# Multi-rank Nsight modes:
#   rank0_cuda  rank 0 traces CUDA+NVTX; other ranks run without Nsight
#   all_nvtx    every rank traces NVTX only (low-overhead straggler view)

set -euo pipefail

# ---- HF / runtime env (parity with scripts/quant_profile_quick.sh) ---------
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-180}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-600}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_TRUST_REMOTE_CODE=${HF_DATASETS_TRUST_REMOTE_CODE:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# nsys streams .qdstrm via $TMPDIR before converting to .nsys-rep. autodl /tmp
# is small (~32 GB); 4-rank traces fill it in minutes and the worker crashes
# mid-trace. Redirect to the autodl data disk if present.
if [[ -z "${TMPDIR:-}" && -d /root/autodl-tmp ]]; then
    export TMPDIR=/root/autodl-tmp/tmp
    mkdir -p "${TMPDIR}"
fi

# ---- positional args -------------------------------------------------------
MODEL_PATH=${1:-models/Qwen/Qwen3-0.6B}
DEVICE=${2:-${CUDA_VISIBLE_DEVICES:-0}}
if [[ $# -ge 2 ]]; then shift 2
elif [[ $# -ge 1 ]]; then shift 1
fi

# ---- realq Config knobs (minimal set; realq.config.Config holds the rest) --
DATASET=${DATASET:-wikitext2}
NSAMPLES=${NSAMPLES:-128}
SEQ_LEN=${SEQ_LEN:-2048}
BSZ=${BSZ:-64}
W_BITS=${W_BITS:-4}
NUM_GROUPS=${NUM_GROUPS:-4}
GRAD_LR=${GRAD_LR:-3e-4}
CPU_MASTER=${CPU_MASTER:-0}
FSDP=${FSDP:-0}
QUANT_STOP_LAYER=${QUANT_STOP_LAYER:-}   # empty = full quant (Config default = None)
NSYS_CAPTURE_START_LAYER=${NSYS_CAPTURE_START_LAYER:-}
NSYS_CAPTURE_END_LAYER=${NSYS_CAPTURE_END_LAYER:-}

# ---- profile-only ----------------------------------------------------------
EXP_NAME=${EXP_NAME:-realq_profile}
OUTPUT_ROOT=${OUTPUT_ROOT:-./outputs}
NSYS=${NSYS:-1}
NVTX=${NVTX:-${NSYS}}
NSYS_OUTPUT=${NSYS_OUTPUT:-${OUTPUT_ROOT}/nsight/${EXP_NAME}}
# Nsight 2024.6.2 in the experiment image does not expose `nccl` as a trace
# category. NCCL CUDA kernels remain visible in a CUDA trace, but requesting
# `nccl` makes nsys reject the command before launch.
NSYS_MODE=${NSYS_MODE:-rank0_cuda}
NSYS_TRACE_RANK0=${NSYS_TRACE_RANK0:-${NSYS_TRACE:-cuda,nvtx}}
NSYS_TRACE_ALL=${NSYS_TRACE_ALL:-nvtx}
NSYS_WAIT=${NSYS_WAIT:-primary}
NSYS_GPU_METRICS_DEVICES=${NSYS_GPU_METRICS_DEVICES:-}
NSYS_GPU_METRICS_FREQUENCY=${NSYS_GPU_METRICS_FREQUENCY:-}
if [[ -z "${NSYS_BIN:-}" ]]; then
    if [[ -x /usr/local/bin/nsys ]]; then
        NSYS_BIN=/usr/local/bin/nsys
    else
        NSYS_BIN=$(command -v nsys || true)
    fi
fi

# cpu_master requires fsdp (realq/config.py:179). Auto-coerce so the user
# doesn't trip the runtime error.
if [[ "${CPU_MASTER}" == "1" && "${FSDP}" != "1" ]]; then
    echo "[realq.profile] CPU_MASTER=1 implies FSDP=1; auto-enabling FSDP." >&2
    FSDP=1
fi
if [[ "${NSYS}" != "0" && "${NSYS}" != "1" ]]; then
    echo "ERROR: NSYS must be 0 or 1; got ${NSYS}." >&2
    exit 2
fi
if [[ "${NVTX}" != "0" && "${NVTX}" != "1" ]]; then
    echo "ERROR: NVTX must be 0 or 1; got ${NVTX}." >&2
    exit 2
fi
if [[ "${NSYS_MODE}" != "rank0_cuda" && "${NSYS_MODE}" != "all_nvtx" ]]; then
    echo "ERROR: NSYS_MODE must be rank0_cuda or all_nvtx; got ${NSYS_MODE}." >&2
    exit 2
fi
if [[ -n "${NSYS_GPU_METRICS_FREQUENCY}" && -z "${NSYS_GPU_METRICS_DEVICES}" ]]; then
    echo "ERROR: NSYS_GPU_METRICS_FREQUENCY requires NSYS_GPU_METRICS_DEVICES." >&2
    exit 2
fi
if [[ -n "${NSYS_GPU_METRICS_FREQUENCY}" && ! "${NSYS_GPU_METRICS_FREQUENCY}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: NSYS_GPU_METRICS_FREQUENCY must be an integer in Hz; got ${NSYS_GPU_METRICS_FREQUENCY}." >&2
    exit 2
fi
if [[ -n "${NSYS_CAPTURE_START_LAYER}" || -n "${NSYS_CAPTURE_END_LAYER}" ]]; then
    if [[ -z "${NSYS_CAPTURE_START_LAYER}" || -z "${NSYS_CAPTURE_END_LAYER}" ]]; then
        echo "ERROR: NSYS_CAPTURE_START_LAYER and NSYS_CAPTURE_END_LAYER must be set together." >&2
        exit 2
    fi
    if [[ "${NSYS}" != "1" || "${NVTX}" != "1" ]]; then
        echo "ERROR: layer-scoped capture requires NSYS=1 and NVTX=1." >&2
        exit 2
    fi
    NSYS_CAPTURE_RANGE=cudaProfilerApi
else
    NSYS_CAPTURE_RANGE=none
fi

export CUDA_VISIBLE_DEVICES=${DEVICE}
IFS=',' read -r -a _DEVICE_LIST <<< "${DEVICE}"
N_GPUS=${N_GPUS:-${#_DEVICE_LIST[@]}}
RDZV_PORT=${RDZV_PORT:-29400}

echo "============================================================"
echo "RealQ profile quick"
echo "  model    : ${MODEL_PATH}"
echo "  device   : ${DEVICE} (N_GPUS=${N_GPUS})"
echo "  nsys     : ${NSYS} mode=${NSYS_MODE} output=${NSYS_OUTPUT}"
echo "  traces   : rank0=${NSYS_TRACE_RANK0} all=${NSYS_TRACE_ALL} nvtx=${NVTX}"
echo "  dataset  : ${DATASET}  n=${NSAMPLES}  seq=${SEQ_LEN}  bsz=${BSZ}"
echo "  quant    : w_bits=${W_BITS}  groups=${NUM_GROUPS}  grad_lr=${GRAD_LR}"
echo "  shard    : cpu_master=${CPU_MASTER}  fsdp=${FSDP}"
echo "  stop@    : ${QUANT_STOP_LAYER:-<none, full quant>}"
echo "  capture  : ${NSYS_CAPTURE_START_LAYER:-<process start>}-${NSYS_CAPTURE_END_LAYER:-<process end>} (${NSYS_CAPTURE_RANGE})"
echo "  metrics  : devices=${NSYS_GPU_METRICS_DEVICES:-<off>} frequency=${NSYS_GPU_METRICS_FREQUENCY:-<default>}"
echo "============================================================"

# ---- realq.ptq arg list ----------------------------------------------------
# NVTX emission is independently controllable. By default it follows NSYS,
# while `NSYS=0 NVTX=1` supports an externally attached profiler without
# launching nsys here.
NSYS_PROFILE_ARG=()
if [[ "${NVTX}" == "1" ]]; then
    NSYS_PROFILE_ARG=(--nsys_profile true)
fi

QUANT_STOP_ARG=()
if [[ -n "${QUANT_STOP_LAYER}" ]]; then
    QUANT_STOP_ARG=(--quant_stop_layer "${QUANT_STOP_LAYER}")
fi

NSYS_CAPTURE_ARGS=()
if [[ -n "${NSYS_CAPTURE_START_LAYER}" ]]; then
    NSYS_CAPTURE_ARGS=(
        --nsys_capture_start_layer "${NSYS_CAPTURE_START_LAYER}"
        --nsys_capture_end_layer "${NSYS_CAPTURE_END_LAYER}"
    )
fi

NSYS_GPU_METRICS_ARGS=()
if [[ -n "${NSYS_GPU_METRICS_DEVICES}" ]]; then
    NSYS_GPU_METRICS_ARGS+=(--gpu-metrics-devices="${NSYS_GPU_METRICS_DEVICES}")
fi
if [[ -n "${NSYS_GPU_METRICS_FREQUENCY}" ]]; then
    NSYS_GPU_METRICS_ARGS+=(--gpu-metrics-frequency="${NSYS_GPU_METRICS_FREQUENCY}")
fi

CMD=(
    python -m torch.distributed.run
    --nnodes=1 --nproc_per_node=${N_GPUS} --rdzv_endpoint=localhost:${RDZV_PORT}
    -m realq.ptq
    --model "${MODEL_PATH}"
    --exp "${EXP_NAME}"
    --output_dir "${OUTPUT_ROOT}"
    --dataset "${DATASET}"
    --nsamples "${NSAMPLES}"
    --seq_len "${SEQ_LEN}"
    --bsz "${BSZ}"
    --w_bits "${W_BITS}"
    --num_groups "${NUM_GROUPS}"
    --grad_lr "${GRAD_LR}"
    --cpu_master "${CPU_MASTER}"
    --fsdp "${FSDP}"
    --skip_eval true
    "${QUANT_STOP_ARG[@]}"
    "${NSYS_CAPTURE_ARGS[@]}"
    "${NSYS_PROFILE_ARG[@]}"
    "$@"
)

if [[ "${NSYS}" != "1" ]]; then
    "${CMD[@]}"
    exit $?
fi

if [[ -z "${NSYS_BIN:-}" || ! -x "${NSYS_BIN}" ]]; then
    echo "================================================================" >&2
    echo "ERROR: NSYS=1 but nsys binary not found." >&2
    echo "  Searched: \$NSYS_BIN=${NSYS_BIN:-<unset>}, /usr/local/bin/nsys, PATH." >&2
    echo "  Install Nsight Systems CLI, or export NSYS_BIN=/path/to/nsys." >&2
    echo "  Or pass NSYS=0 to run without nsys." >&2
    echo "  Add NVTX=1 only if an external profiler should receive ranges." >&2
    echo "================================================================" >&2
    exit 1
fi

mkdir -p "$(dirname "${NSYS_OUTPUT}")"

if [[ "${N_GPUS}" -le 1 ]]; then
    # Single-GPU: wrap nsys around the whole launcher. With nproc_per_node=1
    # there are no NCCL collectives and CUPTI has nothing to race against, so
    # the simple outer wrap is safe.
    if [[ "${NSYS_MODE}" == "all_nvtx" ]]; then
        SINGLE_TRACE="${NSYS_TRACE_ALL}"
    else
        SINGLE_TRACE="${NSYS_TRACE_RANK0}"
    fi
    "${NSYS_BIN}" profile \
        --force-overwrite=true \
        --trace="${SINGLE_TRACE}" \
        --sample=none \
        --cpuctxsw=none \
        --backtrace=none \
        --python-sampling=false \
        --wait="${NSYS_WAIT}" \
        --capture-range="${NSYS_CAPTURE_RANGE}" \
        --capture-range-end=stop \
        "${NSYS_GPU_METRICS_ARGS[@]}" \
        --output="${NSYS_OUTPUT}" \
        "${CMD[@]}"
    exit $?
fi

# Multi-rank: wrapping `nsys profile` around `torch.distributed.run` only
# traces the launcher — workers spawn as children and either get skipped or
# (with default --children=true on recent nsys) all share one trace ring
# buffer, which races with NCCL kernel callbacks and SIGSEGVs randomly. The
# fix is to let torchrun launch a per-rank bash wrapper. In `rank0_cuda` only
# rank 0 execs nsys and all other ranks exec Python directly; in `all_nvtx`
# every rank gets an independent NVTX-only .nsys-rep.
export NSYS_BIN NSYS_MODE NSYS_TRACE_RANK0 NSYS_TRACE_ALL NSYS_WAIT NSYS_CAPTURE_RANGE
export NSYS_GPU_METRICS_DEVICES NSYS_GPU_METRICS_FREQUENCY
NSYS_OUTPUT_BASE="${NSYS_OUTPUT}"
export NSYS_OUTPUT_BASE

RANK_WRAPPER=$(mktemp /tmp/realq_nsys_rank_wrap.XXXXXX)
trap 'rm -f "${RANK_WRAPPER}"' EXIT
cat >"${RANK_WRAPPER}" <<'EOF'
#!/bin/bash
set -e
if [[ "${NSYS_MODE}" == "rank0_cuda" && "${LOCAL_RANK:-0}" != "0" ]]; then
    exec "$@"
fi
if [[ "${NSYS_MODE}" == "all_nvtx" ]]; then
    NSYS_RANK_TRACE="${NSYS_TRACE_ALL}"
else
    NSYS_RANK_TRACE="${NSYS_TRACE_RANK0}"
fi
NSYS_RANK_GPU_METRICS_ARGS=()
if [[ -n "${NSYS_GPU_METRICS_DEVICES}" ]]; then
    NSYS_RANK_GPU_METRICS_ARGS+=(--gpu-metrics-devices="${NSYS_GPU_METRICS_DEVICES}")
fi
if [[ -n "${NSYS_GPU_METRICS_FREQUENCY}" ]]; then
    NSYS_RANK_GPU_METRICS_ARGS+=(--gpu-metrics-frequency="${NSYS_GPU_METRICS_FREQUENCY}")
fi
exec "${NSYS_BIN}" profile \
    --force-overwrite=true \
    --trace="${NSYS_RANK_TRACE}" \
    --sample=none \
    --cpuctxsw=none \
    --backtrace=none \
    --python-sampling=false \
    --wait="${NSYS_WAIT}" \
    --capture-range="${NSYS_CAPTURE_RANGE}" \
    --capture-range-end=stop \
    "${NSYS_RANK_GPU_METRICS_ARGS[@]}" \
    --output="${NSYS_OUTPUT_BASE}_rank${LOCAL_RANK:-0}" \
    "$@"
EOF
chmod +x "${RANK_WRAPPER}"

# CMD index map (must match the CMD=( ... ) block above):
#   0: python   1: -m   2: torch.distributed.run
#   3..5: torchrun flags (--nnodes / --nproc_per_node / --rdzv_endpoint)
#   6: -m       7: realq.ptq    8+: realq args
# Re-launch torchrun with --no-python; each worker enters RANK_WRAPPER, which
# either execs Python directly or execs `nsys profile python` according to
# NSYS_MODE and LOCAL_RANK.
python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=${N_GPUS} --rdzv_endpoint=localhost:${RDZV_PORT} \
    --no-python \
    "${RANK_WRAPPER}" python "${CMD[@]:6}"
