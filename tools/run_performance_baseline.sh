#!/usr/bin/env bash
# Build and run the frozen Qwen3-4B REAL-Q Stage-1 correctness fixture.
#
# This is intentionally a correctness/provenance harness, not a timing
# benchmark.  The two cases in each wave run concurrently on disjoint GPU
# quartets, so their wall times must not be used as primary speedup evidence.
# Isolated warm-up/repetition/Nsight runs are a separate step documented in
# docs/REALQ_PERFORMANCE_WORKLOG.md.
#
# The harness refuses to reuse an existing run directory, requires the
# committed source tree to have no tracked changes, copies the already
# validated smoke caches to the exact current cache names, and verifies after
# every run that Stage 0/tokenization were cache hits.  It also compares the
# group-128 A/A checkpoints by complete canonical tensor state.

set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true
umask 022

readonly SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly DEFAULT_MODEL="/minimax-avatar-new/zhangqian/realq/gptq_plus/modelzoo/Qwen3/Qwen3-4B"
readonly DEFAULT_SMOKE_ROOT="/minimax-avatar-new/zhangqian/realq/experiment_data/schedule_ablation_smoke_20260724"
readonly DEFAULT_OUTPUT_PARENT="/minimax-avatar-new/zhangqian/realq/experiment_data/perf_stage1_debugging_zhangqian"
readonly EXPECTED_STATIC_KEY="Qwen3-4B_wikitext2_n4_sl128_b792a0815fdc"
readonly EXPECTED_TOKEN_SHA256="a0b5aa6f444868a7810a87774a96da3eb4dd4a0ad19bd6525e9f774f7f1c4cae"
readonly WORLD_SIZE=4
readonly MIN_FREE_KIB=$((40 * 1024 * 1024))

MODE=""
MODEL="$DEFAULT_MODEL"
SMOKE_ROOT="$DEFAULT_SMOKE_ROOT"
OUTPUT_PARENT="$DEFAULT_OUTPUT_PARENT"
PYTHON_BIN="${PYTHON:-python}"
ALLOW_DIRTY_SOURCE=0
ALLOW_BUSY_GPUS=0
ALLOW_LOW_DISK=0

usage() {
    cat <<'EOF'
Usage:
  bash tools/run_performance_baseline.sh --run [options]
  bash tools/run_performance_baseline.sh --prepare-only [options]

Modes:
  --run                 Prepare a fresh root and execute both GPU waves.
  --prepare-only        Prepare/provenance-check a fresh root without GPU work.

Options:
  --model PATH          Local Qwen3-4B model directory.
  --smoke-root PATH     Existing validated smoke cache root.
  --output-parent PATH  Parent for the new immutable run root.
  --python PATH         Python executable (default: $PYTHON or python).
  --allow-dirty-source  Permit tracked source changes (recorded, not advised).
  --allow-busy-gpus     Permit pre-existing compute processes on GPUs 0--7.
  --allow-low-disk      Permit less than 40 GiB free at the output parent.
  -h, --help            Show this help.

The --run mode uses:
  wave 1: group-128 GPUs 0--3 / port 29601; per-row GPUs 4--7 / port 29603
  wave 2: group-128 A/A GPUs 0--3 / port 29602; aware GPUs 4--7 / port 29604
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

note() {
    printf '[realq-perf-baseline] %s\n' "$*"
}

while (($#)); do
    case "$1" in
        --run)
            [[ -z "$MODE" ]] || die "choose exactly one of --run/--prepare-only"
            MODE="run"
            shift
            ;;
        --prepare-only)
            [[ -z "$MODE" ]] || die "choose exactly one of --run/--prepare-only"
            MODE="prepare"
            shift
            ;;
        --model)
            (($# >= 2)) || die "--model requires a path"
            MODEL="$2"
            shift 2
            ;;
        --smoke-root)
            (($# >= 2)) || die "--smoke-root requires a path"
            SMOKE_ROOT="$2"
            shift 2
            ;;
        --output-parent)
            (($# >= 2)) || die "--output-parent requires a path"
            OUTPUT_PARENT="$2"
            shift 2
            ;;
        --python)
            (($# >= 2)) || die "--python requires an executable"
            PYTHON_BIN="$2"
            shift 2
            ;;
        --allow-dirty-source)
            ALLOW_DIRTY_SOURCE=1
            shift
            ;;
        --allow-busy-gpus)
            ALLOW_BUSY_GPUS=1
            shift
            ;;
        --allow-low-disk)
            ALLOW_LOW_DISK=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ -n "$MODE" ]] || {
    usage >&2
    die "choose --run or --prepare-only"
}

MODEL="$(readlink -f "$MODEL")"
SMOKE_ROOT="$(readlink -f "$SMOKE_ROOT")"
mkdir -p "$OUTPUT_PARENT"
OUTPUT_PARENT="$(readlink -f "$OUTPUT_PARENT")"
PYTHON_BIN="$(command -v "$PYTHON_BIN")"

[[ -d "$MODEL" ]] || die "model directory does not exist: $MODEL"
[[ -d "$SMOKE_ROOT/static" ]] || die "smoke static directory is missing"
[[ -d "$SMOKE_ROOT/tokens" ]] || die "smoke token directory is missing"
[[ -x "$PYTHON_BIN" ]] || die "python is not executable: $PYTHON_BIN"
command -v git >/dev/null || die "git is required"
command -v sha256sum >/dev/null || die "sha256sum is required"
command -v nvidia-smi >/dev/null || die "nvidia-smi is required"

cd "$REPO_ROOT"

if ((ALLOW_DIRTY_SOURCE == 0)); then
    git diff --quiet --ignore-submodules -- ||
        die "tracked working-tree changes exist; commit/stash them first"
    git diff --cached --quiet --ignore-submodules -- ||
        die "staged source changes exist; commit/stash them first"
fi

free_kib="$(df -Pk "$OUTPUT_PARENT" | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ ]] || die "could not determine free disk space"
if ((free_kib < MIN_FREE_KIB && ALLOW_LOW_DISK == 0)); then
    die "less than 40 GiB free under $OUTPUT_PARENT; use --allow-low-disk only after checking checkpoint capacity"
fi

commit_sha="$(git rev-parse HEAD)"
short_sha="${commit_sha:0:12}"
raw_job_id="${CANOE_JOB_ID:-${JOB_ID:-$(hostname -s)}}"
job_slug="$(printf '%s' "$raw_job_id" | tr -c 'A-Za-z0-9._-' '_')"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="${job_slug}_${timestamp}_${short_sha}"
RUN_ROOT="$OUTPUT_PARENT/$run_id"
[[ ! -e "$RUN_ROOT" ]] || die "refusing to reuse existing run root: $RUN_ROOT"
mkdir -p "$RUN_ROOT"/{runs,static,tokens,provenance}

readonly RUN_ROOT
readonly STATIC_DIR="$RUN_ROOT/static"
readonly TOKENS_DIR="$RUN_ROOT/tokens"
readonly TOKEN_DEST="$TOKENS_DIR/Qwen3-4B_wikitext2_train_n4_sl128_seed1.pt"

declare -Ar RUN_DIRS=(
    [pre_group128]="$RUN_ROOT/runs/pre_group128"
    [pre_row]="$RUN_ROOT/runs/pre_row"
    [pre_group128_repeat]="$RUN_ROOT/runs/pre_group128_repeat"
    [pre_aware_a4k4v4]="$RUN_ROOT/runs/pre_aware_a4k4v4"
)

declare -a ACTIVE_CASE_PIDS=()

stop_active_cases() {
    local pid
    for pid in "${ACTIVE_CASE_PIDS[@]:-}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in "${ACTIVE_CASE_PIDS[@]:-}"; do
        wait "$pid" 2>/dev/null || true
    done
}

on_interrupt() {
    note "interrupted; terminating active torchrun wrappers"
    stop_active_cases
    printf '%s\n' "interrupted $(date -u +%FT%TZ)" >"$RUN_ROOT/INTERRUPTED"
    exit 130
}

trap on_interrupt INT TERM

source_snapshot() {
    local destination="$1"
    "$PYTHON_BIN" - "$REPO_ROOT" "$SCRIPT_PATH" "$MODEL" "$destination" <<'PY'
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone

repo = Path(sys.argv[1])
script = Path(sys.argv[2])
model = Path(sys.argv[3])
destination = Path(sys.argv[4])
sys.path.insert(0, str(repo))
from utils.cache_identity import artifact_identity


def git_bytes(*args: str) -> bytes:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={repo}", *args],
        cwd=repo,
    )


def git_text(*args: str) -> str:
    return git_bytes(*args).decode("utf-8", errors="replace").rstrip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


tracked = git_bytes("diff", "--no-ext-diff", "--binary", "HEAD", "--")
staged = git_bytes("diff", "--cached", "--no-ext-diff", "--binary", "HEAD", "--")
packages = {}
for name in ("torch", "transformers", "datasets", "accelerate", "numpy"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None

payload = {
    "schema_version": 1,
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "repo_root": str(repo.resolve()),
    "git_commit": git_text("rev-parse", "HEAD"),
    "git_branch": git_text("rev-parse", "--abbrev-ref", "HEAD"),
    "tracked_diff_sha256": hashlib.sha256(tracked).hexdigest(),
    "tracked_diff_bytes": len(tracked),
    "staged_diff_sha256": hashlib.sha256(staged).hexdigest(),
    "staged_diff_bytes": len(staged),
    "status_porcelain_v1": git_text("status", "--porcelain=v1", "-uall"),
    "script_path": str(script.resolve()),
    "script_sha256": file_sha256(script),
    "model_path": str(model.resolve()),
    "model_artifact_identity": artifact_identity(model),
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "packages": packages,
    "hostname": socket.gethostname(),
}
destination.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

note "capturing frozen source and model identity"
source_snapshot "$RUN_ROOT/provenance/source_before.json"

computed_static_key="$(
    "$PYTHON_BIN" - "$MODEL" <<'PY'
import sys
from realq.config import Config
from realq.precompute.cache import build_cache_key

cfg = Config(
    model=sys.argv[1],
    dataset="wikitext2",
    eval_datasets=["wikitext2"],
    seed=1,
    rotation_seed=0,
    refresh_seed=0,
    nsamples=4,
    seq_len=128,
    eval_seq_len=2048,
    num_groups=4,
    saliency_clip_percentile=0.99,
    grad_hessian_topk=-1,
    global_loss_bsz=4,
    rotate=True,
    optimized_rotation_path=None,
)
print(build_cache_key(cfg, 4))
PY
)"
[[ "$computed_static_key" == "$EXPECTED_STATIC_KEY" ]] ||
    die "current production static key is $computed_static_key, expected $EXPECTED_STATIC_KEY; do not relabel the smoke cache"

TOKEN_SOURCE="$SMOKE_ROOT/tokens/Qwen3-4B_wikitext2_train_n4_sl128_seed1.dataset-cc6d8a8764f5.tokenizer-64b5baa184e0.pt"
[[ -f "$TOKEN_SOURCE" ]] || die "known smoke token cache is missing: $TOKEN_SOURCE"
token_source_sha="$(sha256sum "$TOKEN_SOURCE" | awk '{print $1}')"
[[ "$token_source_sha" == "$EXPECTED_TOKEN_SHA256" ]] ||
    die "known smoke token cache SHA256 mismatch: $token_source_sha"

note "reflink/copying validated caches to exact current cache names"
for rank in 0 1 2 3; do
    source_static="$SMOKE_ROOT/static/${EXPECTED_STATIC_KEY}_datasetcc6d8a8764f5_world4_rank${rank}.pt"
    dest_static="$STATIC_DIR/${EXPECTED_STATIC_KEY}_world4_rank${rank}.pt"
    [[ -f "$source_static" ]] ||
        die "known smoke static cache is missing for rank $rank: $source_static"
    cp --reflink=auto --preserve=timestamps -- "$source_static" "$dest_static"
done
cp --reflink=auto --preserve=timestamps -- "$TOKEN_SOURCE" "$TOKEN_DEST"

cache_snapshot() {
    local destination="$1"
    "$PYTHON_BIN" - \
        "$SMOKE_ROOT" "$STATIC_DIR" "$TOKEN_SOURCE" "$TOKEN_DEST" \
        "$EXPECTED_STATIC_KEY" "$destination" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

smoke_root = Path(sys.argv[1])
static_dir = Path(sys.argv[2])
token_source = Path(sys.argv[3])
token_dest = Path(sys.argv[4])
key = sys.argv[5]
destination = Path(sys.argv[6])


def identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "allocated_512b_blocks": stat.st_blocks,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


pairs = []
for rank in range(4):
    source = (
        smoke_root
        / "static"
        / f"{key}_datasetcc6d8a8764f5_world4_rank{rank}.pt"
    )
    dest = static_dir / f"{key}_world4_rank{rank}.pt"
    source_identity = identity(source)
    dest_identity = identity(dest)
    pairs.append(
        {
            "kind": "static",
            "rank": rank,
            "source": source_identity,
            "destination": dest_identity,
            "byte_identical": (
                source_identity["size_bytes"] == dest_identity["size_bytes"]
                and source_identity["sha256"] == dest_identity["sha256"]
            ),
        }
    )
source_identity = identity(token_source)
dest_identity = identity(token_dest)
pairs.append(
    {
        "kind": "tokens",
        "rank": None,
        "source": source_identity,
        "destination": dest_identity,
        "byte_identical": (
            source_identity["size_bytes"] == dest_identity["size_bytes"]
            and source_identity["sha256"] == dest_identity["sha256"]
        ),
    }
)
payload = {
    "schema_version": 1,
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "copy_command": "cp --reflink=auto --preserve=timestamps",
    "production_static_key": key,
    "pairs": pairs,
    "passed": all(pair["byte_identical"] for pair in pairs),
}
destination.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
if not payload["passed"]:
    raise SystemExit("cache copy identity validation failed")
PY
}

cache_snapshot "$RUN_ROOT/provenance/cache_before.json"

note "recording hardware and software identity"
{
    printf 'captured_at_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'hostname=%s\n' "$(hostname -f 2>/dev/null || hostname)"
    uname -a
    "$PYTHON_BIN" --version
    nvidia-smi
} >"$RUN_ROOT/provenance/hardware.txt" 2>&1
nvidia-smi -L >"$RUN_ROOT/provenance/gpu_inventory.txt"
nvidia-smi topo -m >"$RUN_ROOT/provenance/gpu_topology.txt"
nvidia-smi \
    --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total \
    --format=csv,noheader,nounits \
    >"$RUN_ROOT/provenance/gpu_mapping.csv"
{
    printf 'CANOE_JOB_ID=%s\n' "${CANOE_JOB_ID:-}"
    printf 'CANOE_JOB_NAME=%s\n' "${CANOE_JOB_NAME:-}"
    printf 'JOB_ID=%s\n' "${JOB_ID:-}"
    printf 'HOSTNAME=%s\n' "${HOSTNAME:-}"
    printf 'CUDA_VISIBLE_DEVICES_overridden_per_case=true\n'
    printf 'CUBLAS_WORKSPACE_CONFIG=:4096:8\n'
    printf 'HF_DATASETS_OFFLINE=1\n'
    printf 'HF_HUB_OFFLINE=1\n'
    printf 'OMP_NUM_THREADS=1\n'
    printf 'TOKENIZERS_PARALLELISM=false\n'
    printf 'TRANSFORMERS_OFFLINE=1\n'
    printf 'NCCL_DEBUG=%s\n' "${NCCL_DEBUG:-}"
    printf 'NCCL_IB_DISABLE=%s\n' "${NCCL_IB_DISABLE:-}"
    printf 'NCCL_P2P_DISABLE=%s\n' "${NCCL_P2P_DISABLE:-}"
} >"$RUN_ROOT/provenance/environment.txt"

declare -a COMMON_CONFIG_ARGS=(
    --model "$MODEL"
    --dataset wikitext2
    --eval_datasets wikitext2
    --nsamples 4
    --seq_len 128
    --eval_seq_len 2048
    --seed 1
    --rotation_seed 0
    --refresh_seed 0
    --w_bits 2
    --w_asym false
    --w_clip true
    --num_groups 4
    --percdamp 0.01
    --blocksize 128
    --act_order true
    --group_parallel_quant rank
    --global_loss_bsz 4
    --saliency_clip_percentile 0.99
    --grad_hessian_topk -1
    --static_cache_path "$STATIC_DIR"
    --exit_after_precompute false
    --grad_lr 0
    --grad_clip 1
    --grad_lr_layer_schedule cosine
    --grad_lr_layer_base_ratio 0.01
    --backward_samples 4
    --backward_bsz 4
    --final_layer_backward_bsz 4
    --a_loss_ratio 1
    --a_loss_clip_scope local_backward_chunk
    --bsz 4
    --hessian_accum_bsz 4
    --fsdp false
    --cpu_master false
    --a_groupsize -1
    --a_asym false
    --v_groupsize -1
    --v_asym false
    --k_groupsize -1
    --k_asym false
    --loss_slide_window false
    --final_layer_grad_lr 0
    --kl_topk -1
    --rotate true
    --skip_eval true
    --lm_eval false
    --lm_eval_batch_size 4
    --tokens_cache_path "$TOKENS_DIR"
    --quant_stop_layer 0
    --nsys_profile false
)

declare -a CASE_CONFIG_ARGS=()
declare -a TORCH_COMMAND=()

build_case_command() {
    local case_name="$1"
    local run_dir="$2"
    local master_port="$3"
    local -a quant_args=()

    case "$case_name" in
        pre_group128|pre_group128_repeat)
            quant_args=(
                --w_groupsize 128
                --a_bits 16
                --a_clip_ratio 1
                --v_bits 16
                --v_clip_ratio 1
                --k_bits 16
                --k_clip_ratio 1
                --act_quant_aware_gptq false
                --k_cache_quant_aware_gptq false
            )
            ;;
        pre_row)
            quant_args=(
                --w_groupsize -1
                --a_bits 16
                --a_clip_ratio 1
                --v_bits 16
                --v_clip_ratio 1
                --k_bits 16
                --k_clip_ratio 1
                --act_quant_aware_gptq false
                --k_cache_quant_aware_gptq false
            )
            ;;
        pre_aware_a4k4v4)
            quant_args=(
                --w_groupsize 128
                --a_bits 4
                --a_clip_ratio 0.9
                --v_bits 4
                --v_clip_ratio 0.9
                --k_bits 4
                --k_clip_ratio 0.9
                --act_quant_aware_gptq true
                --k_cache_quant_aware_gptq true
            )
            ;;
        *)
            die "unknown case: $case_name"
            ;;
    esac

    CASE_CONFIG_ARGS=(
        "${COMMON_CONFIG_ARGS[@]}"
        "${quant_args[@]}"
        --cache_dir "$run_dir/runtime_cache"
        --output_dir "$run_dir/program_output"
        --exp "$case_name"
        --save_qmodel_path "$run_dir/checkpoint/model.pt"
    )
    TORCH_COMMAND=(
        "$PYTHON_BIN"
        -m torch.distributed.run
        --nproc_per_node "$WORLD_SIZE"
        --master_port "$master_port"
        --module realq.ptq
        "${CASE_CONFIG_ARGS[@]}"
    )
}

quote_command() {
    local item
    for item in "$@"; do
        printf '%q ' "$item"
    done
    printf '\n'
}

prepare_case() {
    local case_name="$1"
    local gpu_ids="$2"
    local master_port="$3"
    local wave="$4"
    local run_dir="${RUN_DIRS[$case_name]}"

    mkdir -p \
        "$run_dir/checkpoint" \
        "$run_dir/program_output" \
        "$run_dir/runtime_cache"
    build_case_command "$case_name" "$run_dir" "$master_port"

    "$PYTHON_BIN" - "$run_dir/resolved_config.json" "${CASE_CONFIG_ARGS[@]}" <<'PY'
from dataclasses import asdict
import json
from pathlib import Path
import sys

from realq.config import parse_cli

destination = Path(sys.argv[1])
cfg = parse_cli(sys.argv[2:])
destination.write_text(
    json.dumps(asdict(cfg), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

    {
        printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu_ids"
        printf 'CUBLAS_WORKSPACE_CONFIG=%q ' ":4096:8"
        printf 'HF_DATASETS_OFFLINE=%q ' "1"
        printf 'HF_HUB_OFFLINE=%q ' "1"
        printf 'OMP_NUM_THREADS=%q ' "1"
        printf 'TOKENIZERS_PARALLELISM=%q ' "false"
        printf 'TRANSFORMERS_OFFLINE=%q ' "1"
        printf 'PYTHONPATH=%q ' "$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
        quote_command "${TORCH_COMMAND[@]}"
    } >"$run_dir/command.txt"

    "$PYTHON_BIN" - \
        "$run_dir/manifest.json" "$case_name" "$gpu_ids" "$master_port" \
        "$wave" "$run_id" "$run_dir/resolved_config.json" \
        "$run_dir/command.txt" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

(
    destination_raw,
    case_name,
    gpu_ids,
    master_port,
    wave,
    root_run_id,
    config_raw,
    command_raw,
) = sys.argv[1:]
destination = Path(destination_raw)
config_path = Path(config_raw)
command_path = Path(command_raw)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


payload = {
    "schema_version": 1,
    "root_run_id": root_run_id,
    "case": case_name,
    "wave": int(wave),
    "status": "prepared",
    "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
    "gpu_ids": [int(item) for item in gpu_ids.split(",")],
    "world_size": 4,
    "master_port": int(master_port),
    "concurrent_correctness_run": True,
    "eligible_for_primary_speedup_claim": False,
    "resolved_config_path": str(config_path),
    "resolved_config_sha256": sha256(config_path),
    "command_path": str(command_path),
    "command_sha256": sha256(command_path),
    "run_log_path": str(destination.parent / "run.log"),
    "gpu_telemetry_path": str(destination.parent / "gpu_memory.csv"),
    "checkpoint_path": str(destination.parent / "checkpoint" / "model.pt"),
}
destination.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
    chmod a-w "$run_dir/resolved_config.json" "$run_dir/command.txt"
}

prepare_case pre_group128 "0,1,2,3" 29601 1
prepare_case pre_row "4,5,6,7" 29603 1
prepare_case pre_group128_repeat "0,1,2,3" 29602 2
prepare_case pre_aware_a4k4v4 "4,5,6,7" 29604 2

"$PYTHON_BIN" - "$RUN_ROOT" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
names = (
    "pre_group128",
    "pre_row",
    "pre_group128_repeat",
    "pre_aware_a4k4v4",
)
configs = {
    name: json.loads(
        (root / "runs" / name / "resolved_config.json").read_text(
            encoding="utf-8"
        )
    )
    for name in names
}
path_fields = {"cache_dir", "exp", "output_dir", "save_qmodel_path"}


def normalized(config: dict) -> dict:
    return {key: value for key, value in config.items() if key not in path_fields}


def differences(left: dict, right: dict) -> dict:
    return {
        key: {"left": left.get(key), "right": right.get(key)}
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    }


group = normalized(configs["pre_group128"])
repeat = normalized(configs["pre_group128_repeat"])
row = normalized(configs["pre_row"])
aware = normalized(configs["pre_aware_a4k4v4"])
group_repeat_diff = differences(group, repeat)
row_diff = differences(group, row)
aware_diff = differences(group, aware)
expected_row = {"w_groupsize"}
expected_aware = {
    "a_bits",
    "a_clip_ratio",
    "act_quant_aware_gptq",
    "k_bits",
    "k_cache_quant_aware_gptq",
    "k_clip_ratio",
    "v_bits",
    "v_clip_ratio",
}
passed = (
    not group_repeat_diff
    and set(row_diff) == expected_row
    and set(aware_diff) == expected_aware
)
report = {
    "schema_version": 1,
    "ignored_artifact_path_fields": sorted(path_fields),
    "group_repeat_differences": group_repeat_diff,
    "group_vs_row_differences": row_diff,
    "group_vs_aware_differences": aware_diff,
    "expected_group_vs_row_fields": sorted(expected_row),
    "expected_group_vs_aware_fields": sorted(expected_aware),
    "passed": passed,
}
(root / "provenance" / "config_matrix.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
if not passed:
    raise SystemExit("resolved config matrix has unexpected differences")
PY

"$PYTHON_BIN" - \
    "$RUN_ROOT/manifest.json" "$run_id" "$commit_sha" "$MODE" \
    "$MODEL" "$SMOKE_ROOT" <<'PY'
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

destination, run_id, commit, mode, model, smoke_root = sys.argv[1:]
payload = {
    "schema_version": 1,
    "run_id": run_id,
    "status": "prepared",
    "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
    "mode": mode,
    "git_commit": commit,
    "model": model,
    "smoke_cache_root": smoke_root,
    "concurrent_correctness_run": True,
    "eligible_for_primary_speedup_claim": False,
    "waves": [
        {
            "wave": 1,
            "cases": ["pre_group128", "pre_row"],
            "concurrent": True,
        },
        {
            "wave": 2,
            "cases": [
                "pre_group128_repeat",
                "pre_aware_a4k4v4",
            ],
            "concurrent": True,
        },
    ],
}
Path(destination).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

note "fresh run root prepared: $RUN_ROOT"
if [[ "$MODE" == "prepare" ]]; then
    note "prepare-only requested; no GPU process was launched"
    printf '%s\n' "$RUN_ROOT" >"$RUN_ROOT/RUN_ROOT"
    exit 0
fi

gpu_count="$(nvidia-smi -L | wc -l)"
((gpu_count >= 8)) || die "the fixture requires at least 8 visible GPUs; found $gpu_count"

if ((ALLOW_BUSY_GPUS == 0)); then
    busy_processes="$(
        nvidia-smi \
            --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
            --format=csv,noheader,nounits 2>/dev/null |
            sed '/^[[:space:]]*$/d'
    )"
    [[ -z "$busy_processes" ]] ||
        die "pre-existing GPU compute processes detected; use --allow-busy-gpus only after checking ownership:\n$busy_processes"
fi

ports_available() {
    "$PYTHON_BIN" - "$@" <<'PY'
import socket
import sys

sockets = []
try:
    for raw in sys.argv[1:]:
        port = int(raw)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", port))
        sockets.append(sock)
finally:
    for sock in sockets:
        sock.close()
PY
}

validate_case() {
    local case_name="$1"
    local command_rc="$2"
    local run_dir="${RUN_DIRS[$case_name]}"
    "$PYTHON_BIN" - \
        "$run_dir/run.log" \
        "$run_dir/checkpoint/model.pt" \
        "$TOKEN_DEST" \
        "$STATIC_DIR" \
        "$EXPECTED_STATIC_KEY" \
        "$command_rc" \
        "$run_dir/validation.json" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

log_path = Path(sys.argv[1])
checkpoint_path = Path(sys.argv[2])
token_path = Path(sys.argv[3])
static_dir = Path(sys.argv[4])
static_key = sys.argv[5]
command_rc = int(sys.argv[6])
destination = Path(sys.argv[7])
text = log_path.read_text(encoding="utf-8", errors="replace")

expected_rank0_path = static_dir / f"{static_key}_world4_rank0.pt"
rank0_marker = (
    f"[realq.precompute] cache hit (rank 0): {expected_rank0_path}"
)
rank0_hit_count = text.count(rank0_marker)

token_marker = f"Loading tokens from {token_path}"
quant_stop_marker = (
    "[realq] quant_stop_layer=0 reached; remaining layers stay FP."
)
save_marker = (
    "[realq] reproducible quantized checkpoint saved "
    f"\u2192 {checkpoint_path}"
)
forbidden_markers = {
    "static_cache_miss": "[realq.precompute] cache miss",
    "static_cache_recompute_consensus": "all ranks will recompute Stage 0",
    "static_cache_write": "[realq.precompute] wrote rank-",
    "dataset_fetch": "Fetching dataset:",
    "token_save": "Saving tokens to",
    "unreadable_cache": "ignoring unreadable cache file",
    "invalid_cache": "ignoring invalid cache payload",
    "python_traceback": "Traceback (most recent call last)",
    "torchrun_child_failure": "ChildFailedError",
    "cuda_oom": "CUDA out of memory",
}
forbidden_counts = {
    name: text.count(marker)
    for name, marker in forbidden_markers.items()
}
checks = {
    "command_rc_zero": command_rc == 0,
    # Only rank zero emits to the merged console log. static_e2e.run returns
    # through this branch only after its distributed MIN hit-consensus says
    # every rank loaded its shard; before/after hashes separately cover all
    # four copied files.
    "collective_static_cache_hit": rank0_hit_count >= 1,
    "token_cache_hit": text.count(token_marker) >= 1,
    "no_recompute_or_failure_markers": all(
        count == 0 for count in forbidden_counts.values()
    ),
    "quant_stop_marker": text.count(quant_stop_marker) >= 1,
    "checkpoint_save_marker": text.count(save_marker) >= 1,
    "checkpoint_exists_nonempty": (
        checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0
    ),
}
report = {
    "schema_version": 1,
    "command_return_code": command_rc,
    "checks": checks,
    "rank0_static_cache_hit_count": rank0_hit_count,
    "token_cache_hit_count": text.count(token_marker),
    "quant_stop_marker_count": text.count(quant_stop_marker),
    "checkpoint_save_marker_count": text.count(save_marker),
    "forbidden_marker_counts": forbidden_counts,
    "checkpoint_size_bytes": (
        checkpoint_path.stat().st_size if checkpoint_path.is_file() else None
    ),
    "passed": all(checks.values()),
}
destination.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
raise SystemExit(0 if report["passed"] else 1)
PY
}

finalize_case_manifest() {
    local case_name="$1"
    local run_dir="${RUN_DIRS[$case_name]}"
    "$PYTHON_BIN" - \
        "$run_dir/manifest.json" \
        "$run_dir/timing.json" \
        "$run_dir/validation.json" \
        "$run_dir/gpu_memory.csv" <<'PY'
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

manifest_path, timing_path, validation_path, telemetry_path = map(
    Path, sys.argv[1:]
)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
timing = json.loads(timing_path.read_text(encoding="utf-8"))
validation = json.loads(validation_path.read_text(encoding="utf-8"))
peaks = {}
sample_count = 0
if telemetry_path.is_file():
    with telemetry_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample_count += 1
            try:
                index = str(int(row["index"]))
                used = int(float(row["memory_used_mib"]))
            except (KeyError, TypeError, ValueError):
                continue
            peaks[index] = max(peaks.get(index, 0), used)
manifest.update(
    {
        "status": "passed" if validation["passed"] else "failed",
        "timing": timing,
        "validation": validation,
        "sampled_gpu_memory_peak_mib": peaks,
        "gpu_telemetry_rows": sample_count,
    }
)
manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

run_case() {
    local case_name="$1"
    local gpu_ids="$2"
    local master_port="$3"
    local run_dir="${RUN_DIRS[$case_name]}"
    local telemetry_pid=""
    local command_pid=""
    local command_rc=1
    local validation_rc=1
    local start_epoch_ns end_epoch_ns
    local start_utc end_utc

    build_case_command "$case_name" "$run_dir" "$master_port"
    start_epoch_ns="$(date +%s%N)"
    start_utc="$(date -u +%FT%T.%NZ)"

    {
        printf '%s\n' \
            "timestamp,index,uuid,memory_used_mib,memory_total_mib,utilization_gpu_percent,utilization_memory_percent"
        nvidia-smi \
            --id="$gpu_ids" \
            --query-gpu=timestamp,index,uuid,memory.used,memory.total,utilization.gpu,utilization.memory \
            --format=csv,noheader,nounits \
            --loop=5
    } >"$run_dir/gpu_memory.csv" 2>"$run_dir/gpu_memory.stderr" &
    telemetry_pid=$!

    env \
        "CUDA_VISIBLE_DEVICES=$gpu_ids" \
        "CUBLAS_WORKSPACE_CONFIG=:4096:8" \
        "HF_DATASETS_OFFLINE=1" \
        "HF_HUB_OFFLINE=1" \
        "OMP_NUM_THREADS=1" \
        "TOKENIZERS_PARALLELISM=false" \
        "TRANSFORMERS_OFFLINE=1" \
        "PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        "${TORCH_COMMAND[@]}" \
        >"$run_dir/run.log" 2>&1 &
    command_pid=$!

    terminate_case_children() {
        kill -TERM "$command_pid" 2>/dev/null || true
        kill -TERM "$telemetry_pid" 2>/dev/null || true
        wait "$command_pid" 2>/dev/null || true
        wait "$telemetry_pid" 2>/dev/null || true
        exit 143
    }
    trap terminate_case_children INT TERM

    set +e
    wait "$command_pid"
    command_rc=$?
    set -e
    kill -TERM "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
    trap - INT TERM

    end_epoch_ns="$(date +%s%N)"
    end_utc="$(date -u +%FT%T.%NZ)"
    "$PYTHON_BIN" - \
        "$run_dir/timing.json" "$start_epoch_ns" "$end_epoch_ns" \
        "$start_utc" "$end_utc" "$command_rc" <<'PY'
import json
from pathlib import Path
import sys

path, start_ns, end_ns, start_utc, end_utc, command_rc = sys.argv[1:]
start_ns = int(start_ns)
end_ns = int(end_ns)
payload = {
    "scope": "process_wall",
    "start_utc": start_utc,
    "end_utc": end_utc,
    "start_epoch_ns": start_ns,
    "end_epoch_ns": end_ns,
    "elapsed_seconds": (end_ns - start_ns) / 1e9,
    "command_return_code": int(command_rc),
    "concurrent_correctness_run": True,
    "eligible_for_primary_speedup_claim": False,
}
Path(path).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

    set +e
    validate_case "$case_name" "$command_rc"
    validation_rc=$?
    set -e
    finalize_case_manifest "$case_name"
    if ((command_rc != 0 || validation_rc != 0)); then
        printf '%s\n' 1 >"$run_dir/return_code"
        return 1
    fi
    printf '%s\n' 0 >"$run_dir/return_code"
    return 0
}

wait_wave() {
    local left_pid="$1"
    local right_pid="$2"
    local left_name="$3"
    local right_name="$4"
    local left_rc right_rc
    set +e
    wait "$left_pid"
    left_rc=$?
    wait "$right_pid"
    right_rc=$?
    set -e
    ACTIVE_CASE_PIDS=()
    note "$left_name rc=$left_rc; $right_name rc=$right_rc"
    ((left_rc == 0 && right_rc == 0))
}

finalize_root() {
    local requested_status="$1"
    local requested_rc="$2"
    source_snapshot "$RUN_ROOT/provenance/source_after.json"
    cache_snapshot "$RUN_ROOT/provenance/cache_after.json"
    "$PYTHON_BIN" - \
        "$RUN_ROOT" "$requested_status" "$requested_rc" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys
from datetime import datetime, timezone

root = Path(sys.argv[1])
requested_status = sys.argv[2]
requested_rc = int(sys.argv[3])
manifest_path = root / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
before = json.loads(
    (root / "provenance" / "source_before.json").read_text(encoding="utf-8")
)
after = json.loads(
    (root / "provenance" / "source_after.json").read_text(encoding="utf-8")
)
source_fields = (
    "git_commit",
    "tracked_diff_sha256",
    "tracked_diff_bytes",
    "staged_diff_sha256",
    "staged_diff_bytes",
    "status_porcelain_v1",
    "script_sha256",
    "model_artifact_identity",
)
source_differences = {
    key: {"before": before.get(key), "after": after.get(key)}
    for key in source_fields
    if before.get(key) != after.get(key)
}
cache_before = json.loads(
    (root / "provenance" / "cache_before.json").read_text(encoding="utf-8")
)
cache_after = json.loads(
    (root / "provenance" / "cache_after.json").read_text(encoding="utf-8")
)


def cache_identity(report: dict) -> list:
    return [
        {
            "kind": pair["kind"],
            "rank": pair["rank"],
            "source_size": pair["source"]["size_bytes"],
            "source_sha256": pair["source"]["sha256"],
            "destination_size": pair["destination"]["size_bytes"],
            "destination_sha256": pair["destination"]["sha256"],
        }
        for pair in report["pairs"]
    ]


cache_unchanged = (
    cache_before["passed"]
    and cache_after["passed"]
    and cache_identity(cache_before) == cache_identity(cache_after)
)
case_reports = {}
for case_dir in sorted((root / "runs").iterdir()):
    case_manifest = case_dir / "manifest.json"
    if case_manifest.is_file():
        case_reports[case_dir.name] = json.loads(
            case_manifest.read_text(encoding="utf-8")
        )
all_cases_passed = (
    len(case_reports) == 4
    and all(report.get("status") == "passed" for report in case_reports.values())
)
aa_path = root / "checkpoint_compare.json"
aa_report = (
    json.loads(aa_path.read_text(encoding="utf-8"))
    if aa_path.is_file()
    else None
)
passed = (
    requested_rc == 0
    and not source_differences
    and cache_unchanged
    and all_cases_passed
    and aa_report is not None
    and aa_report.get("passed") is True
)
manifest.update(
    {
        "status": "passed" if passed else requested_status,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "return_code": 0 if passed else (requested_rc or 1),
        "source_unchanged": not source_differences,
        "source_differences": source_differences,
        "cache_unchanged": cache_unchanged,
        "all_cases_passed": all_cases_passed,
        "aa_checkpoint_compare_passed": (
            aa_report.get("passed") if aa_report is not None else None
        ),
        "cases": case_reports,
    }
)
manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
raise SystemExit(0 if passed else 1)
PY
}

note "wave 1 starting: group-128 GPUs 0--3; per-row GPUs 4--7"
ports_available 29601 29603
run_case pre_group128 "0,1,2,3" 29601 &
wave1_left=$!
run_case pre_row "4,5,6,7" 29603 &
wave1_right=$!
ACTIVE_CASE_PIDS=("$wave1_left" "$wave1_right")
if ! wait_wave "$wave1_left" "$wave1_right" pre_group128 pre_row; then
    note "wave 1 failed; wave 2 will not start"
    finalize_root failed_wave1 1 || true
    exit 1
fi

note "wave 2 starting: group-128 A/A GPUs 0--3; aware A4K4V4 GPUs 4--7"
ports_available 29602 29604
run_case pre_group128_repeat "0,1,2,3" 29602 &
wave2_left=$!
run_case pre_aware_a4k4v4 "4,5,6,7" 29604 &
wave2_right=$!
ACTIVE_CASE_PIDS=("$wave2_left" "$wave2_right")
if ! wait_wave \
    "$wave2_left" "$wave2_right" \
    pre_group128_repeat pre_aware_a4k4v4; then
    note "wave 2 failed"
    finalize_root failed_wave2 1 || true
    exit 1
fi

note "building canonical raw-byte checkpoint state manifests"
set +e
"$PYTHON_BIN" - "$REPO_ROOT" "$RUN_ROOT" <<'PY'
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import struct
import sys
import torch

repo = Path(sys.argv[1])
root = Path(sys.argv[2])
sys.path.insert(0, str(repo))
from tools.run_alignment_matrix import (  # noqa: E402
    _canonical_state_dict,
    _checkpoint_payload,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_field(digest, raw: bytes) -> None:
    digest.update(struct.pack("<Q", len(raw)))
    digest.update(raw)


def state_manifest(path: Path) -> dict:
    payload = _checkpoint_payload(path)
    state = _canonical_state_dict(payload["model"])
    state_digest = hashlib.sha256(b"REALQ_CANONICAL_STATE_V1\0")
    tensors = {}
    for canonical_key in sorted(state):
        raw_key, tensor = state[canonical_key]
        cpu = tensor.detach().cpu().contiguous()
        raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
        dtype = str(cpu.dtype)
        shape = list(cpu.shape)
        tensor_digest = hashlib.sha256(raw).hexdigest()
        update_field(state_digest, canonical_key.encode("utf-8"))
        update_field(state_digest, dtype.encode("ascii"))
        update_field(
            state_digest,
            json.dumps(shape, separators=(",", ":")).encode("ascii"),
        )
        update_field(state_digest, raw)
        tensors[canonical_key] = {
            "serialized_key": raw_key,
            "dtype": dtype,
            "shape": shape,
            "numel": cpu.numel(),
            "nbytes": len(raw),
            "sha256": tensor_digest,
        }
    return {
        "schema_version": 1,
        "digest_algorithm": "REALQ_CANONICAL_STATE_V1",
        "checkpoint_path": str(path.resolve()),
        "checkpoint_size_bytes": path.stat().st_size,
        "checkpoint_archive_sha256": file_sha256(path),
        "canonical_state_sha256": state_digest.hexdigest(),
        "canonical_tensor_count": len(tensors),
        "tensors": tensors,
    }


for name in (
    "pre_group128",
    "pre_row",
    "pre_group128_repeat",
    "pre_aware_a4k4v4",
):
    checkpoint = root / "runs" / name / "checkpoint" / "model.pt"
    report = state_manifest(checkpoint)
    destination = checkpoint.parent / "state_manifest.json"
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    del report
    gc.collect()
PY
state_manifest_rc=$?
set -e
if ((state_manifest_rc != 0)); then
    note "canonical checkpoint state-manifest generation failed"
    finalize_root failed_state_manifest 1 || true
    exit 1
fi

note "running the independent group-128 A/A checkpoint comparator"
set +e
"$PYTHON_BIN" - "$REPO_ROOT" "$RUN_ROOT" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

repo = Path(sys.argv[1])
root = Path(sys.argv[2])
sys.path.insert(0, str(repo))
from tools.run_alignment_matrix import (  # noqa: E402
    _compare_checkpoint_manifests,
    _compare_model_weights,
)

left = root / "runs" / "pre_group128" / "checkpoint" / "model.pt"
right = (
    root
    / "runs"
    / "pre_group128_repeat"
    / "checkpoint"
    / "model.pt"
)
left_state = json.loads(
    (left.parent / "state_manifest.json").read_text(encoding="utf-8")
)
right_state = json.loads(
    (right.parent / "state_manifest.json").read_text(encoding="utf-8")
)
weights = _compare_model_weights(left, right)
manifests = _compare_checkpoint_manifests(left, right)
state_sha_equal = (
    left_state["canonical_state_sha256"]
    == right_state["canonical_state_sha256"]
)
report = {
    "schema_version": 1,
    "comparison": "pre_group128_vs_pre_group128_repeat",
    "physical_gpu_quartet": [0, 1, 2, 3],
    "archive_sha_equal_not_required": True,
    "left_archive_sha256": left_state["checkpoint_archive_sha256"],
    "right_archive_sha256": right_state["checkpoint_archive_sha256"],
    "left_canonical_state_sha256": left_state["canonical_state_sha256"],
    "right_canonical_state_sha256": right_state["canonical_state_sha256"],
    "canonical_state_sha256_equal": state_sha_equal,
    "weights": weights,
    "checkpoint_manifest": manifests,
    "passed": (
        state_sha_equal
        and weights["passed"]
        and weights["max_abs_difference"] == 0
        and manifests["passed"]
    ),
}
destination = root / "checkpoint_compare.json"
destination.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
for checkpoint_parent in (left.parent, right.parent):
    shutil.copy2(destination, checkpoint_parent / "checkpoint_compare.json")
raise SystemExit(0 if report["passed"] else 1)
PY
aa_rc=$?
set -e

if ((aa_rc != 0)); then
    note "group-128 A/A checkpoint equality failed"
    finalize_root failed_aa_compare 1 || true
    exit 1
fi

if ! finalize_root passed 0; then
    note "final E0/cache/source validation failed"
    exit 1
fi

printf '%s\n' "$RUN_ROOT" >"$RUN_ROOT/RUN_ROOT"
printf '%s\n' "passed $(date -u +%FT%TZ)" >"$RUN_ROOT/COMPLETE"
note "all correctness/provenance gates passed"
note "run root: $RUN_ROOT"
