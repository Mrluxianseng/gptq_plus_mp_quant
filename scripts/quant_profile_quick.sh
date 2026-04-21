#!/bin/bash

# Quick Nsight / NVTX profile of the GPTQ+ pipeline. Mirrors the arg set of
# scripts/gptq_plus_lr_sweep.sh so what you profile matches what you run in
# production — only the profile-specific knobs (TARGET_LAYERS, QUANT_STOP_LAYER,
# NSYS*) and a shorter default sample budget differ. Defaults assume a 1-GPU
# run; pass DEVICE=0,1 to go multi-rank.

set -euo pipefail

# HF offline / timeout knobs — same as sweep.
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-180}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-600}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_TRUST_REMOTE_CODE=${HF_DATASETS_TRUST_REMOTE_CODE:-1}

# CUDA VMM expandable segments. Same rationale as sweep.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# nsys streams intermediate traces (.qdstrm) via $TMPDIR before converting to
# .nsys-rep. On autodl / docker containers /tmp is often only ~32 GB total,
# and 4-rank runs fill it in minutes → rank crashes mid-trace. Redirect to the
# autodl data disk when it's mounted and the user hasn't overridden TMPDIR.
if [[ -z "${TMPDIR:-}" && -d /root/autodl-tmp ]]; then
    export TMPDIR=/root/autodl-tmp/tmp
    mkdir -p "${TMPDIR}"
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <MODEL_PATH> [NUM_GROUPS=4] [DEVICE=0] [extra ptq.py args ...]"
    echo "Example: TARGET_LAYERS=0,1 QUANT_STOP_LAYER=1 $0 ./modelzoo/Qwen3/Qwen3-0.6B"
    exit 1
fi

MODEL_PATH=${1}
NUM_GROUPS=${2:-4}
DEVICE=${3:-${CUDA_VISIBLE_DEVICES:-0}}
if [[ $# -ge 3 ]]; then
    shift 3
elif [[ $# -ge 2 ]]; then
    shift 2
else
    shift 1
fi

# ============================================================
# All defaults below are aligned with scripts/gptq_plus_lr_sweep.sh unless
# annotated PROFILE-ONLY. When you tweak the sweep, mirror the change here.
# ============================================================

# Quantization configuration (SWEEP-ALIGNED)
GRAD_LR=${GRAD_LR:-0.00002}
DATASET=${DATASET:-wikitext2}
N_SAMPLES=${N_SAMPLES:-512}      # PROFILE-ONLY: smaller pool so nsys capture stays snappy
SEQ_LEN=${SEQ_LEN:-2048}
BSZ=${BSZ:-128}
FINAL_LAYER_STATS_BSZ=${FINAL_LAYER_STATS_BSZ:-8}
HESSIAN_ACCUM_BSZ=${HESSIAN_ACCUM_BSZ:-64}
ENABLE_GPTQ_PLUS=${ENABLE_GPTQ_PLUS:-0}
BACKWARD_SAMPLES=${BACKWARD_SAMPLES:-32}
BACKWARD_BSZ=${BACKWARD_BSZ:-32}
FINAL_LAYER_BACKWARD_BSZ=${FINAL_LAYER_BACKWARD_BSZ:-8}
FINAL_LAYER_FULL_BACKWARD=${FINAL_LAYER_FULL_BACKWARD:-0}
BLOCKSIZE=${BLOCKSIZE:-256}
BLOCK_ATOMIC_QUANT=${BLOCK_ATOMIC_QUANT:-0}
GRAD_OPTIMIZER=${GRAD_OPTIMIZER:-adam}
FINAL_LAYER_GRAD_OPTIMIZER=${FINAL_LAYER_GRAD_OPTIMIZER:-adam}
GRAD_CLIP=${GRAD_CLIP:-5e-5}
FINAL_LAYER_GRAD_CLIP=${FINAL_LAYER_GRAD_CLIP:-5e-4}
GRAD_REFRESH_LOSS=${GRAD_REFRESH_LOSS:-refined_mse}
REFINED_RKL_NUM_A=${REFINED_RKL_NUM_A:-1}
REFINED_RKL_DAMP=${REFINED_RKL_DAMP:-0.01}
NUM_SAMPLES_FOR_REFINED_MSE=${NUM_SAMPLES_FOR_REFINED_MSE:-32}
FINAL_LAYER_GRAD_LR=${FINAL_LAYER_GRAD_LR:-0.000001}
PRE_GD_STEPS=${PRE_GD_STEPS:-10}
PRE_GRAD_LR=${PRE_GRAD_LR:-0.00003}
PRE_FINAL_LAYER_GRAD_LR=${PRE_FINAL_LAYER_GRAD_LR:-0.3}
PRE_GRAD_OPTIMIZER=${PRE_GRAD_OPTIMIZER:-adam}
PRE_FINAL_LAYER_GRAD_OPTIMIZER=${PRE_FINAL_LAYER_GRAD_OPTIMIZER:-sgd}
GRAD_REG_STRATEGY=${GRAD_REG_STRATEGY:-none}
GRAD_REG_LAMBDA=${GRAD_REG_LAMBDA:-0.01}
GRAD_GATE_FLOOR=${GRAD_GATE_FLOOR:-0.01}
GRAD_GATE_SHARPNESS=${GRAD_GATE_SHARPNESS:-5.0}
GRAD_GATE_SINE_AMP=${GRAD_GATE_SINE_AMP:-0.00005}
GRAD_HESSIAN_TOPK=${GRAD_HESSIAN_TOPK:-20}
SALIENCY_CLIP_PERCENTILE=${SALIENCY_CLIP_PERCENTILE:-1.0}
PROJ_LR_SCALE=${PROJ_LR_SCALE:-1.0}
DOWN_PROJ_LR_SCALE=${DOWN_PROJ_LR_SCALE:-1.0}
SECOND_ORDER_SCALE=${SECOND_ORDER_SCALE:-1.0}
FISHER_NUM_GROUPS=${FISHER_NUM_GROUPS:-512}
PRE_CLIP=${PRE_CLIP:-0}
GLOBAL_LOSS=${GLOBAL_LOSS:-1}
GLOBAL_LOSS_BSZ=${GLOBAL_LOSS_BSZ:-8}
LOSS_SLIDE_WINDOW=${LOSS_SLIDE_WINDOW:-0}
DP_GLOBAL_SHUFFLE=${DP_GLOBAL_SHUFFLE:-1}
GRAD_LR_LAYER_SCHEDULE=${GRAD_LR_LAYER_SCHEDULE:-none}
ALPHA=${ALPHA:-0.0}
KL_TOPK=${KL_TOPK:-20}
G_UPDATE_MODE=${G_UPDATE_MODE:-block_gd}
CACHE_DIR=${CACHE_DIR:-./cache}
OUTPUT_ROOT=${OUTPUT_ROOT:-./outputs}

# FSDP (rarely needed for a profile run; expose for parity with sweep)
FSDP_PRECOMPUTE=${FSDP_PRECOMPUTE:-0}
FSDP_CPU_OFFLOAD=${FSDP_CPU_OFFLOAD:-0}
STATIC_CACHE_PATH=${STATIC_CACHE_PATH:-}
EXIT_AFTER_PRECOMPUTE=${EXIT_AFTER_PRECOMPUTE:-0}

# ============================================================
# Profile-only controls
# ============================================================
TARGET_LAYERS=${TARGET_LAYERS:-0,1}
TARGET_MODULES=${TARGET_MODULES:-all}
QUANT_STOP_LAYER=${QUANT_STOP_LAYER:-1}
EXP_NAME=${EXP_NAME:-quant_profile_quick}
NSYS=${NSYS:-1}
NSYS_OUTPUT=${NSYS_OUTPUT:-./outputs/nsight/${EXP_NAME}}
NSYS_TRACE=${NSYS_TRACE:-cuda,nvtx}
NSYS_WAIT=${NSYS_WAIT:-primary}
# When 1, flip the wall-clock cuda.Event summary at end-of-run. Independent of
# nsys; useful when Nsight isn't available.
WALL_PROFILE=${WALL_PROFILE:-0}
if [[ "${WALL_PROFILE}" == "1" ]]; then
    export GPTQ_PLUS_WALL_PROFILE=1
fi
if [[ -z "${NSYS_BIN:-}" ]]; then
    if [[ -x /usr/local/bin/nsys ]]; then
        NSYS_BIN=/usr/local/bin/nsys
    else
        NSYS_BIN=$(command -v nsys || true)
    fi
fi

export CUDA_VISIBLE_DEVICES=${DEVICE}
MODEL_NAME=$(basename "${MODEL_PATH}")

# HF cache layout — same as sweep so they share downloaded datasets.
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HOME}/.cache/huggingface/hub}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}

# DP: infer rank count from DEVICE.
IFS=',' read -r -a _DEVICE_LIST <<< "${DEVICE}"
N_GPUS=${N_GPUS:-${#_DEVICE_LIST[@]}}
RDZV_PORT=${RDZV_PORT:-29400}

# ============================================================
# Arg assembly — mirror sweep's shape exactly so the plumbing stays in sync.
# ============================================================

BLOCK_ATOMIC_ARGS=()
if [[ "${BLOCK_ATOMIC_QUANT}" == "1" ]]; then
    BLOCK_ATOMIC_ARGS=(--block_atomic_quant)
fi

FINAL_LAYER_FULL_BACKWARD_ARGS=()
if [[ "${FINAL_LAYER_FULL_BACKWARD}" == "1" ]]; then
    FINAL_LAYER_FULL_BACKWARD_ARGS=(--final_layer_full_backward)
fi

PRE_CLIP_ARGS=()
if [[ "${PRE_CLIP}" == "1" ]]; then
    PRE_CLIP_ARGS=(--pre_clip)
else
    PRE_CLIP_ARGS=(--no_pre_clip)
fi

PRE_FINAL_LAYER_GRAD_LR_ARGS=()
if [[ -n "${PRE_FINAL_LAYER_GRAD_LR}" && "${PRE_FINAL_LAYER_GRAD_LR}" != "none" ]]; then
    PRE_FINAL_LAYER_GRAD_LR_ARGS=(--pre_final_layer_grad_lr "${PRE_FINAL_LAYER_GRAD_LR}")
fi

FINAL_LAYER_GRAD_CLIP_ARGS=()
if [[ -n "${FINAL_LAYER_GRAD_CLIP}" && "${FINAL_LAYER_GRAD_CLIP}" != "none" ]]; then
    FINAL_LAYER_GRAD_CLIP_ARGS=(--final_layer_grad_clip "${FINAL_LAYER_GRAD_CLIP}")
fi

PRE_FINAL_LAYER_GRAD_OPTIMIZER_ARGS=()
if [[ -n "${PRE_FINAL_LAYER_GRAD_OPTIMIZER}" && "${PRE_FINAL_LAYER_GRAD_OPTIMIZER}" != "none" ]]; then
    PRE_FINAL_LAYER_GRAD_OPTIMIZER_ARGS=(--pre_final_layer_grad_optimizer "${PRE_FINAL_LAYER_GRAD_OPTIMIZER}")
fi

GLOBAL_LOSS_ARGS=()
if [[ "${GLOBAL_LOSS}" == "1" ]]; then
    GLOBAL_LOSS_ARGS=(--global_loss --global_loss_bsz "${GLOBAL_LOSS_BSZ}")
else
    GLOBAL_LOSS_ARGS=(--no_global_loss)
fi

LOSS_SLIDE_WINDOW_ARGS=()
if [[ "${LOSS_SLIDE_WINDOW}" == "1" ]]; then
    LOSS_SLIDE_WINDOW_ARGS=(--loss_slide_window)
fi

DP_GLOBAL_SHUFFLE_ARGS=()
if [[ "${DP_GLOBAL_SHUFFLE}" == "1" ]]; then
    DP_GLOBAL_SHUFFLE_ARGS=(--dp_global_shuffle)
fi

GRAD_LR_LAYER_SCHEDULE_ARGS=()
if [[ "${GRAD_LR_LAYER_SCHEDULE}" != "none" ]]; then
    GRAD_LR_LAYER_SCHEDULE_ARGS=(--grad_lr_layer_schedule "${GRAD_LR_LAYER_SCHEDULE}")
fi

FSDP_ARGS=()
if [[ "${FSDP_PRECOMPUTE}" == "1" ]]; then
    FSDP_ARGS+=(--fsdp_precompute)
fi
if [[ "${FSDP_CPU_OFFLOAD}" == "1" ]]; then
    FSDP_ARGS+=(--fsdp_cpu_offload)
fi
if [[ "${EXIT_AFTER_PRECOMPUTE}" == "1" ]]; then
    FSDP_ARGS+=(--exit_after_precompute)
fi
if [[ -n "${STATIC_CACHE_PATH}" ]]; then
    FSDP_ARGS+=(--static_cache_path "${STATIC_CACHE_PATH}")
fi

echo "============================================================"
echo "GPTQ+ quant profile quick"
echo "  model  : ${MODEL_PATH}"
echo "  device : ${DEVICE} (N_GPUS=${N_GPUS})"
echo "  groups : ${NUM_GROUPS} (fng=${FISHER_NUM_GROUPS})"
echo "  target : layers=${TARGET_LAYERS} modules=${TARGET_MODULES} stop=${QUANT_STOP_LAYER}"
echo "  nsys   : ${NSYS} output=${NSYS_OUTPUT}  wall=${WALL_PROFILE}"
echo "  gpplus : ${ENABLE_GPTQ_PLUS}"
echo "  mode   : ${G_UPDATE_MODE}"
echo "  atomic : ${BLOCK_ATOMIC_QUANT}"
echo "  flfb   : ${FINAL_LAYER_FULL_BACKWARD}"
echo "  opt    : ${GRAD_OPTIMIZER}  flopt=${FINAL_LAYER_GRAD_OPTIMIZER}"
echo "  gclip  : ${GRAD_CLIP}  flgclip=${FINAL_LAYER_GRAD_CLIP:-<default>}"
echo "  rloss  : ${GRAD_REFRESH_LOSS}"
if [[ "${GRAD_REFRESH_LOSS}" == "refined_residual_kl" ]]; then
    echo "  rkl_NA : ${REFINED_RKL_NUM_A}  damp=${REFINED_RKL_DAMP}"
fi
if [[ "${GRAD_REFRESH_LOSS}" == "refined_mse" ]]; then
    echo "  nRM    : ${NUM_SAMPLES_FOR_REFINED_MSE}"
fi
echo "  reg    : ${GRAD_REG_STRATEGY}  lambda=${GRAD_REG_LAMBDA}"
echo "  gate   : floor=${GRAD_GATE_FLOOR} sharp=${GRAD_GATE_SHARPNESS} sineA=${GRAD_GATE_SINE_AMP}"
echo "  gh_tk  : ${GRAD_HESSIAN_TOPK}  salclip=${SALIENCY_CLIP_PERCENTILE}"
echo "  proj_s : ${PROJ_LR_SCALE}  down_s=${DOWN_PROJ_LR_SCALE}"
echo "  preclip: ${PRE_CLIP}  pregd=${PRE_GD_STEPS}  prelr=${PRE_GRAD_LR}  preopt=${PRE_GRAD_OPTIMIZER}"
echo "  preflr : ${PRE_FINAL_LAYER_GRAD_LR:-<default>}  prefo=${PRE_FINAL_LAYER_GRAD_OPTIMIZER:-<default>}"
echo "  global : ${GLOBAL_LOSS}  gl_bsz=${GLOBAL_LOSS_BSZ}"
echo "  slide  : ${LOSS_SLIDE_WINDOW}  gshuf=${DP_GLOBAL_SHUFFLE}  lrsched=${GRAD_LR_LAYER_SCHEDULE}"
echo "  gradlr : ${GRAD_LR}  fllr=${FINAL_LAYER_GRAD_LR}  so_scl=${SECOND_ORDER_SCALE}  alpha=${ALPHA}"
echo "  block  : ${BLOCKSIZE}  bsz=${BSZ}  fl_stb=${FINAL_LAYER_STATS_BSZ}  ha_bsz=${HESSIAN_ACCUM_BSZ}"
echo "  bw_smp : ${BACKWARD_SAMPLES}  bw_bsz=${BACKWARD_BSZ}  fl_bwb=${FINAL_LAYER_BACKWARD_BSZ}"
echo "  n_samp : ${N_SAMPLES}  seq=${SEQ_LEN}  dataset=${DATASET}"
echo "============================================================"

CMD=(
    python -m torch.distributed.run
    --nnodes=1 --nproc_per_node=${N_GPUS} --rdzv_endpoint=localhost:${RDZV_PORT} ./ptq.py
    --model "${MODEL_PATH}"
    --exp "${EXP_NAME}"
    --output_dir "${OUTPUT_ROOT}"
    --cache_dir "${CACHE_DIR}"
    --dataset "${DATASET}" --nsamples "${N_SAMPLES}" --seq_len "${SEQ_LEN}"
    --w_method gptq_plus --w_bits 4 --w_clip --num_groups "${NUM_GROUPS}" --fisher_num_groups "${FISHER_NUM_GROUPS}" --act_order
    --kl_topk "${KL_TOPK}" --bsz "${BSZ}" --final_layer_stats_bsz "${FINAL_LAYER_STATS_BSZ}" --alpha "${ALPHA}" --blocksize "${BLOCKSIZE}"
    --enable_gptq_plus "${ENABLE_GPTQ_PLUS}"
    ${HESSIAN_ACCUM_BSZ:+--hessian_accum_bsz "${HESSIAN_ACCUM_BSZ}"}
    --backward_samples "${BACKWARD_SAMPLES}" --backward_bsz "${BACKWARD_BSZ}" --final_layer_backward_bsz "${FINAL_LAYER_BACKWARD_BSZ}"
    --g_update_mode "${G_UPDATE_MODE}" --grad_lr "${GRAD_LR}" --grad_optimizer "${GRAD_OPTIMIZER}" --grad_refresh_loss "${GRAD_REFRESH_LOSS}"
    --refined_rkl_num_A "${REFINED_RKL_NUM_A}" --refined_rkl_damp "${REFINED_RKL_DAMP}"
    --num_samples_for_refined_mse "${NUM_SAMPLES_FOR_REFINED_MSE}"
    --rotate
    "${GLOBAL_LOSS_ARGS[@]}"
    "${LOSS_SLIDE_WINDOW_ARGS[@]}"
    "${DP_GLOBAL_SHUFFLE_ARGS[@]}"
    "${GRAD_LR_LAYER_SCHEDULE_ARGS[@]}"
    "${FSDP_ARGS[@]}"
    --final_layer_grad_optimizer "${FINAL_LAYER_GRAD_OPTIMIZER}"
    --grad_clip "${GRAD_CLIP}"
    "${FINAL_LAYER_GRAD_CLIP_ARGS[@]}"
    --final_layer_grad_lr "${FINAL_LAYER_GRAD_LR}"
    --grad_hessian_topk "${GRAD_HESSIAN_TOPK}"
    --saliency_clip_percentile "${SALIENCY_CLIP_PERCENTILE}"
    --pre_gd_steps "${PRE_GD_STEPS}"
    --pre_grad_lr "${PRE_GRAD_LR}"
    --pre_grad_optimizer "${PRE_GRAD_OPTIMIZER}"
    "${PRE_FINAL_LAYER_GRAD_LR_ARGS[@]}"
    "${PRE_FINAL_LAYER_GRAD_OPTIMIZER_ARGS[@]}"
    "${PRE_CLIP_ARGS[@]}"
    "${BLOCK_ATOMIC_ARGS[@]}"
    "${FINAL_LAYER_FULL_BACKWARD_ARGS[@]}"
    --proj_lr_scale "${PROJ_LR_SCALE}" --down_proj_lr_scale "${DOWN_PROJ_LR_SCALE}"
    --grad_reg_strategy "${GRAD_REG_STRATEGY}" --grad_reg_lambda "${GRAD_REG_LAMBDA}"
    --grad_gate_floor "${GRAD_GATE_FLOOR}" --grad_gate_sharpness "${GRAD_GATE_SHARPNESS}" --grad_gate_sine_amp "${GRAD_GATE_SINE_AMP}"
    --second_order_scale "${SECOND_ORDER_SCALE}"
    # Profile-specific flags: emit NVTX, skip eval, stop early.
    --enable_quant_profile
    --quant_profile_target_layers "${TARGET_LAYERS}"
    --quant_profile_target_modules "${TARGET_MODULES}"
    --skip_eval
)

if [[ -n "${QUANT_STOP_LAYER}" && "${QUANT_STOP_LAYER}" != "none" ]]; then
    CMD+=(--quant_stop_layer "${QUANT_STOP_LAYER}")
fi

CMD+=("$@")

if [[ "${NSYS}" == "1" && -n "${NSYS_BIN:-}" && -x "${NSYS_BIN}" ]]; then
    mkdir -p "$(dirname "${NSYS_OUTPUT}")"

    if [[ "${N_GPUS}" -gt 1 ]]; then
        # Multi-rank: wrapping `nsys profile` around `torch.distributed.run`
        # only traces the launcher process — the worker ranks (where all the
        # CUDA kernels + NVTX ranges actually live) are spawned as children
        # and get skipped. Instead, let torchrun launch a small bash wrapper
        # per rank, which execs `nsys profile → python`. Each rank drops its
        # own .nsys-rep suffixed by LOCAL_RANK so they open cleanly in Nsight
        # without one giant merged file.
        export NSYS_BIN NSYS_TRACE NSYS_WAIT
        NSYS_OUTPUT_BASE="${NSYS_OUTPUT}"
        export NSYS_OUTPUT_BASE

        RANK_WRAPPER=$(mktemp /tmp/nsys_rank_wrap.XXXXXX)
        trap 'rm -f "${RANK_WRAPPER}"' EXIT
        cat >"${RANK_WRAPPER}" <<'EOF'
#!/bin/bash
set -e
exec "${NSYS_BIN}" profile \
    --force-overwrite=true \
    --trace="${NSYS_TRACE}" \
    --sample=none \
    --cpuctxsw=none \
    --backtrace=none \
    --python-sampling=false \
    --wait="${NSYS_WAIT}" \
    --capture-range=none \
    --output="${NSYS_OUTPUT_BASE}_rank${LOCAL_RANK:-0}" \
    "$@"
EOF
        chmod +x "${RANK_WRAPPER}"

        # CMD[0..5] = launcher (python -m torch.distributed.run + its flags);
        # CMD[6]    = ./ptq.py; CMD[7..] = ptq.py args. We re-assemble so
        # torchrun (with --no-python) execs `${RANK_WRAPPER} python ptq.py ...`
        # in each rank, and the wrapper in turn execs `nsys profile python ...`.
        python -m torch.distributed.run \
            --nnodes=1 --nproc_per_node=${N_GPUS} --rdzv_endpoint=localhost:${RDZV_PORT} \
            --no-python \
            "${RANK_WRAPPER}" python "${CMD[@]:6}"
    else
        # Single-GPU: wrap nsys around the whole thing. With nproc_per_node=1
        # the launcher overhead is negligible, and having nsys at the outer
        # layer keeps the one-file report simple.
        "${NSYS_BIN}" profile \
            --force-overwrite=true \
            --trace="${NSYS_TRACE}" \
            --sample=none \
            --cpuctxsw=none \
            --backtrace=none \
            --python-sampling=false \
            --wait="${NSYS_WAIT}" \
            --capture-range=none \
            --output="${NSYS_OUTPUT}" \
            "${CMD[@]}"
    fi
elif [[ "${NSYS}" == "1" ]]; then
    echo "================================================================" >&2
    echo "ERROR: NSYS=1 but nsys binary not found." >&2
    echo "  Searched: \$NSYS_BIN=${NSYS_BIN:-<unset>}, /usr/local/bin/nsys, PATH." >&2
    echo "  Install Nsight Systems CLI, or export NSYS_BIN=/path/to/nsys." >&2
    echo "  Or pass NSYS=0 to run with NVTX ranges only (no capture)." >&2
    echo "================================================================" >&2
    exit 1
else
    "${CMD[@]}"
fi
