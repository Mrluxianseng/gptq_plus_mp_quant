#!/usr/bin/env bash
# Single-quartet REAL-Q Stage-1 correctness/checkpoint harness.
#
# One invocation creates one baseline or candidate arm.  It never refers to a
# second GPU quartet: every GPU operation is scoped to the explicit --gpus
# list, and CUDA_VISIBLE_DEVICES uses that same list.  The default mode is a
# dry-run that prepares immutable commands and provenance; GPU execution
# requires an explicit --run.

set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true
umask 022

readonly SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly COMPARE_TOOL="$REPO_ROOT/tools/compare_performance_checkpoints.py"
readonly DEFAULT_MODEL="/minimax-avatar-new/zhangqian/realq/gptq_plus/modelzoo/Qwen3/Qwen3-4B"
readonly DEFAULT_SMOKE_ROOT="/minimax-avatar-new/zhangqian/realq/experiment_data/schedule_ablation_smoke_20260724"
readonly DEFAULT_OUTPUT_PARENT="/minimax-avatar-new/zhangqian/realq/experiment_data/perf_stage1_correctness_zhangqian"
readonly EXPECTED_STATIC_KEY="Qwen3-4B_wikitext2_n4_sl128_b792a0815fdc"
readonly EXPECTED_TOKEN_SHA256="a0b5aa6f444868a7810a87774a96da3eb4dd4a0ad19bd6525e9f774f7f1c4cae"
readonly WORLD_SIZE=4
readonly MIN_FREE_KIB=$((20 * 1024 * 1024))

MODE="prepare"
MODE_EXPLICIT=0
CASE_NAME=""
GPU_CSV=""
LABEL=""
BASELINE_ID="qwen3-4b-stage1-correctness"
MASTER_PORT=29820
MODEL="$DEFAULT_MODEL"
SMOKE_ROOT="$DEFAULT_SMOKE_ROOT"
OUTPUT_PARENT="$DEFAULT_OUTPUT_PARENT"
PYTHON_BIN="${PYTHON:-python}"
ALLOW_LOW_DISK=0
RUN_ROOT=""
declare -a GPU_IDS=()
declare -a CANDIDATE_SPECS=()
declare -a CANDIDATE_NAMES=()
declare -a CANDIDATE_CLI_ARGS=()
declare -a ACTIVE_PIDS=()

usage() {
    cat <<'EOF'
Usage:
  bash tools/run_performance_correctness.sh \
    --case CASE --gpus I,J,K,L --label LABEL [options]

The default is a dry-run (prepare-only). GPU execution is impossible unless
--run is supplied explicitly.

Cases:
  group_stress       W2 group-128, A16/K16/V16, unaware
  per_row            W2 per-row, A16/K16/V16, unaware
  group_akv_aware    W2 group-128, A4/K4/V4 clip=0.9, aware
  block_gd_stress    W2 group-128 with Block-GD + loss sliding enabled

Required:
  --case NAME                 One case listed above.
  --gpus I,J,K,L              Exactly four distinct physical GPU indices.
  --label LABEL               Human-readable arm label.

Modes:
  --run                       Explicitly launch one four-rank GPU run.
  --prepare-only              Explicit spelling of the default dry-run.

Options:
  --baseline-id ID            A/B family identifier stored verbatim.
  --candidate-arg NAME=VALUE  Append one optimization Config override.
                              Repeatable; mathematical/protocol fields are
                              protected and duplicate fields are rejected.
  --master-port PORT          torchrun rendezvous port (default: 29820).
  --model PATH                Local Qwen3-4B model directory.
  --smoke-root PATH           Validated Stage-0/token smoke-cache root.
  --output-parent PATH        Parent of the fresh immutable run root.
  --python PATH               Python executable (default: $PYTHON or python).
  --allow-low-disk            Permit less than 20 GiB free after inspection.
  -h, --help                  Show this help.

Examples:
  # Baseline dry-run; does not launch a GPU process.
  bash tools/run_performance_correctness.sh \
    --case group_stress --gpus 4,5,6,7 --label legacy

  # Candidate execution; --run is mandatory.
  bash tools/run_performance_correctness.sh --run \
    --case group_stress --gpus 4,5,6,7 --label p01 \
    --candidate-arg quantizer_inner_fastpath=true

  # Strict A/B comparison (pass run roots, not only checkpoint files, to also
  # enforce same commit/case/quartet/baseline-id and exact declared config diff).
  python tools/compare_performance_checkpoints.py compare \
    BASELINE_RUN_ROOT CANDIDATE_RUN_ROOT -o checkpoint_compare.json
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

note() {
    printf '[realq-perf-correctness] %s\n' "$*"
}

while (($#)); do
    case "$1" in
        --run)
            ((MODE_EXPLICIT == 0)) || die "choose at most one mode"
            MODE="run"
            MODE_EXPLICIT=1
            shift
            ;;
        --prepare-only)
            ((MODE_EXPLICIT == 0)) || die "choose at most one mode"
            MODE="prepare"
            MODE_EXPLICIT=1
            shift
            ;;
        --case)
            (($# >= 2)) || die "--case requires a value"
            CASE_NAME="$2"
            shift 2
            ;;
        --gpus)
            (($# >= 2)) || die "--gpus requires I,J,K,L"
            GPU_CSV="$2"
            shift 2
            ;;
        --label)
            (($# >= 2)) || die "--label requires a value"
            LABEL="$2"
            shift 2
            ;;
        --baseline-id)
            (($# >= 2)) || die "--baseline-id requires a value"
            BASELINE_ID="$2"
            shift 2
            ;;
        --candidate-arg)
            (($# >= 2)) || die "--candidate-arg requires NAME=VALUE"
            CANDIDATE_SPECS+=("$2")
            shift 2
            ;;
        --master-port)
            (($# >= 2)) || die "--master-port requires an integer"
            MASTER_PORT="$2"
            shift 2
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

case "$CASE_NAME" in
    group_stress|per_row|group_akv_aware|block_gd_stress) ;;
    "") die "--case is required" ;;
    *) die "unknown case: $CASE_NAME" ;;
esac
[[ -n "$GPU_CSV" ]] || die "--gpus is required; no quartet is implicit"
[[ -n "$LABEL" ]] || die "--label is required"
[[ -n "$BASELINE_ID" ]] || die "--baseline-id may not be empty"
[[ "$MASTER_PORT" =~ ^[0-9]+$ ]] ||
    die "--master-port must be an integer"
((MASTER_PORT >= 1024 && MASTER_PORT <= 65535)) ||
    die "--master-port must be in 1024..65535"

IFS=',' read -r -a GPU_IDS <<<"$GPU_CSV"
((${#GPU_IDS[@]} == WORLD_SIZE)) ||
    die "--gpus must contain exactly four physical indices"
declare -A SEEN_GPUS=()
for gpu_id in "${GPU_IDS[@]}"; do
    [[ "$gpu_id" =~ ^[0-9]+$ ]] ||
        die "invalid physical GPU index: $gpu_id"
    [[ -z "${SEEN_GPUS[$gpu_id]:-}" ]] ||
        die "duplicate physical GPU index: $gpu_id"
    SEEN_GPUS["$gpu_id"]=1
done
GPU_CSV="$(IFS=,; printf '%s' "${GPU_IDS[*]}")"

MODEL="$(readlink -f "$MODEL")"
SMOKE_ROOT="$(readlink -f "$SMOKE_ROOT")"
mkdir -p "$OUTPUT_PARENT"
OUTPUT_PARENT="$(readlink -f "$OUTPUT_PARENT")"
PYTHON_BIN="$(command -v "$PYTHON_BIN")"

[[ -d "$MODEL" ]] || die "model directory does not exist: $MODEL"
[[ -d "$SMOKE_ROOT/static" ]] || die "smoke static directory is missing"
[[ -d "$SMOKE_ROOT/tokens" ]] || die "smoke token directory is missing"
[[ -f "$COMPARE_TOOL" ]] || die "checkpoint comparison tool is missing"
[[ -x "$PYTHON_BIN" ]] || die "python is not executable: $PYTHON_BIN"
command -v cp >/dev/null || die "cp is required"
command -v git >/dev/null || die "git is required"
command -v nvidia-smi >/dev/null || die "nvidia-smi is required"
command -v setsid >/dev/null || die "setsid is required"
command -v sha256sum >/dev/null || die "sha256sum is required"

cd "$REPO_ROOT"

repo_git() {
    git -c "safe.directory=$REPO_ROOT" "$@"
}

repo_git diff --quiet --ignore-submodules -- ||
    die "tracked working-tree changes exist; commit/stash them first"
repo_git diff --cached --quiet --ignore-submodules -- ||
    die "staged source changes exist; commit/stash them first"

free_kib="$(df -Pk "$OUTPUT_PARENT" | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ ]] || die "could not determine free disk space"
if ((free_kib < MIN_FREE_KIB && ALLOW_LOW_DISK == 0)); then
    die "less than 20 GiB free; inspect capacity or use --allow-low-disk"
fi

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
    --quant_stop_layer 0
    --nsys_profile false
)

declare -a CASE_CONFIG_ARGS=()
case "$CASE_NAME" in
    group_stress)
        CASE_CONFIG_ARGS=(
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
    per_row)
        CASE_CONFIG_ARGS=(
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
    group_akv_aware)
        CASE_CONFIG_ARGS=(
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
    block_gd_stress)
        CASE_CONFIG_ARGS=(
            --w_groupsize 128
            --a_bits 16
            --a_clip_ratio 1
            --v_bits 16
            --v_clip_ratio 1
            --k_bits 16
            --k_clip_ratio 1
            --act_quant_aware_gptq false
            --k_cache_quant_aware_gptq false
            --grad_lr 0.0003
            --loss_slide_window true
        )
        ;;
esac

# Only independently switchable fields added after the frozen protocol may be
# candidate args.  The allow-list names the current reviewed candidates;
# existing mathematical, data, path, logging, and execution fields are
# protected even if they are not explicitly present in this fixture.
declare -A ALLOWED_CANDIDATE_KEYS=(
    [quantizer_inner_fastpath]=1
    [w_clip_search_impl]=1
    [w_clip_update_impl]=1
    [w_group_param_layout]=1
    [fisher_fp32_cache]=1
    [act_order_stitch_impl]=1
)
declare -A PROTECTED_KEYS=()
declare -a FROZEN_CONFIG_FIELDS=(
    model dataset eval_datasets seed rotation_seed refresh_seed nsamples
    seq_len eval_seq_len w_bits w_groupsize w_asym w_clip num_groups
    percdamp blocksize act_order group_parallel_quant global_loss_bsz
    saliency_clip_percentile grad_hessian_topk static_cache_path
    exit_after_precompute alignment_trace_path alignment_run_id grad_lr
    grad_clip final_layer_grad_clip grad_lr_layer_schedule
    grad_lr_layer_base_ratio backward_samples backward_bsz
    final_layer_backward_bsz a_loss_ratio a_loss_clip_scope bsz
    hessian_accum_bsz fsdp fsdp_cpu_offload fsdp_max_shard_size
    fsdp_prepared_dir cpu_master a_bits a_groupsize a_asym a_clip_ratio
    v_bits v_groupsize v_asym v_clip_ratio k_bits k_groupsize k_asym
    k_clip_ratio act_quant_aware_gptq k_cache_quant_aware_gptq
    loss_slide_window final_layer_grad_lr kl_topk rotate
    optimized_rotation_path skip_eval lm_eval lm_eval_batch_size
    tokens_cache_path cache_dir quant_stop_layer perf_measure_layer
    nsys_profile load_qmodel_path allow_unsafe_legacy_checkpoint
    save_qmodel_path output_dir exp model_name
)
for protected in "${FROZEN_CONFIG_FIELDS[@]}"; do
    PROTECTED_KEYS["$protected"]=1
done
for ((i = 0; i < ${#COMMON_CONFIG_ARGS[@]}; i += 2)); do
    PROTECTED_KEYS["${COMMON_CONFIG_ARGS[$i]#--}"]=1
done
for ((i = 0; i < ${#CASE_CONFIG_ARGS[@]}; i += 2)); do
    PROTECTED_KEYS["${CASE_CONFIG_ARGS[$i]#--}"]=1
done

declare -A SEEN_CANDIDATE_KEYS=()
for spec in "${CANDIDATE_SPECS[@]}"; do
    [[ "$spec" == *=* ]] ||
        die "candidate arg must be NAME=VALUE, got: $spec"
    name="${spec%%=*}"
    value="${spec#*=}"
    [[ "$name" =~ ^[A-Za-z][A-Za-z0-9_]*$ ]] ||
        die "invalid candidate Config field name: $name"
    [[ -n "$value" ]] || die "candidate value may not be empty: $name"
    [[ -n "${ALLOWED_CANDIDATE_KEYS[$name]:-}" ]] ||
        die "candidate Config field is not a reviewed optimization: $name"
    [[ -z "${PROTECTED_KEYS[$name]:-}" ]] ||
        die "candidate arg attempts to override frozen field: $name"
    [[ -z "${SEEN_CANDIDATE_KEYS[$name]:-}" ]] ||
        die "duplicate candidate Config field: $name"
    SEEN_CANDIDATE_KEYS["$name"]=1
    CANDIDATE_NAMES+=("$name")
    CANDIDATE_CLI_ARGS+=("--$name" "$value")
done

commit_sha="$(repo_git rev-parse HEAD)"
short_sha="${commit_sha:0:12}"
raw_job_id="${CANOE_JOB_ID:-${JOB_ID:-$(hostname -s)}}"
job_slug="$(printf '%s' "$raw_job_id" | tr -c 'A-Za-z0-9._-' '_')"
label_slug="$(printf '%s' "$LABEL" | tr -c 'A-Za-z0-9._-' '_' | sed 's/^_*//;s/_*$//')"
[[ -n "$label_slug" ]] || label_slug="arm"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="${job_slug}_${timestamp}_${short_sha}_${CASE_NAME}_${label_slug}"
RUN_ROOT="$OUTPUT_PARENT/$run_id"
[[ ! -e "$RUN_ROOT" ]] || die "refusing to reuse run root: $RUN_ROOT"
mkdir -p "$RUN_ROOT"/{checkpoint,program_output,provenance,runtime_cache,static,tokens}
readonly RUN_ROOT
readonly STATIC_DIR="$RUN_ROOT/static"
readonly TOKENS_DIR="$RUN_ROOT/tokens"
readonly CHECKPOINT_PATH="$RUN_ROOT/checkpoint/model.pt"
readonly TOKEN_DEST="$TOKENS_DIR/Qwen3-4B_wikitext2_train_n4_sl128_seed1.pt"

source_snapshot() {
    local destination="$1"
    "$PYTHON_BIN" - \
        "$REPO_ROOT" "$SCRIPT_PATH" "$COMPARE_TOOL" "$MODEL" \
        "$destination" <<'PY'
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import socket
import subprocess
import sys
from datetime import datetime, timezone

repo = Path(sys.argv[1])
script = Path(sys.argv[2])
compare_tool = Path(sys.argv[3])
model = Path(sys.argv[4])
destination = Path(sys.argv[5])
sys.path.insert(0, str(repo))
from utils.cache_identity import artifact_identity


def git_bytes(*args: str) -> bytes:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={repo}", *args], cwd=repo
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
staged = git_bytes(
    "diff", "--cached", "--no-ext-diff", "--binary", "HEAD", "--"
)
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
    "status_porcelain_v1_audit_only": git_text(
        "status", "--porcelain=v1", "-uall"
    ),
    "script_path": str(script.resolve()),
    "script_sha256": file_sha256(script),
    "compare_tool_path": str(compare_tool.resolve()),
    "compare_tool_sha256": file_sha256(compare_tool),
    "model_path": str(model.resolve()),
    "model_artifact_identity": artifact_identity(model),
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "packages": packages,
    "hostname": socket.gethostname(),
}
destination.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
}

cache_snapshot() {
    local destination="$1"
    "$PYTHON_BIN" - \
        "$SMOKE_ROOT" "$STATIC_DIR" "$TOKEN_SOURCE" "$TOKEN_DEST" \
        "$EXPECTED_STATIC_KEY" "$destination" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
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
    item = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": item.st_size,
        "mtime_ns": item.st_mtime_ns,
        "mode": stat.S_IMODE(item.st_mode),
        "sha256": digest.hexdigest(),
    }


pairs = []
for rank in range(4):
    source = (
        smoke_root
        / "static"
        / f"{key}_datasetcc6d8a8764f5_world4_rank{rank}.pt"
    )
    copied = static_dir / f"{key}_world4_rank{rank}.pt"
    source_identity = identity(source)
    copied_identity = identity(copied)
    pairs.append(
        {
            "kind": "static",
            "rank": rank,
            "source": source_identity,
            "destination": copied_identity,
            "byte_identical": (
                source_identity["size_bytes"]
                == copied_identity["size_bytes"]
                and source_identity["sha256"]
                == copied_identity["sha256"]
            ),
        }
    )
source_identity = identity(token_source)
copied_identity = identity(token_dest)
pairs.append(
    {
        "kind": "tokens",
        "rank": None,
        "source": source_identity,
        "destination": copied_identity,
        "byte_identical": (
            source_identity["size_bytes"] == copied_identity["size_bytes"]
            and source_identity["sha256"] == copied_identity["sha256"]
        ),
    }
)
payload = {
    "schema_version": 1,
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "copy_command": "cp --reflink=auto --preserve=timestamps",
    "copied_cache_permissions_read_only": all(
        pair["destination"]["mode"] & 0o222 == 0 for pair in pairs
    ),
    "production_static_key": key,
    "pairs": pairs,
    "passed": all(pair["byte_identical"] for pair in pairs),
}
destination.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
if not payload["passed"]:
    raise SystemExit("cache copy identity validation failed")
PY
}

note "capturing frozen source/model identity"
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
    die "production static key is $computed_static_key, expected $EXPECTED_STATIC_KEY"

TOKEN_SOURCE="$SMOKE_ROOT/tokens/Qwen3-4B_wikitext2_train_n4_sl128_seed1.dataset-cc6d8a8764f5.tokenizer-64b5baa184e0.pt"
readonly TOKEN_SOURCE
[[ -f "$TOKEN_SOURCE" ]] || die "known smoke token cache is missing"
token_source_sha="$(sha256sum "$TOKEN_SOURCE" | awk '{print $1}')"
[[ "$token_source_sha" == "$EXPECTED_TOKEN_SHA256" ]] ||
    die "known smoke token SHA256 mismatch: $token_source_sha"

note "copying validated caches into a read-only run-local view"
declare -a COPIED_CACHE_FILES=()
for rank in 0 1 2 3; do
    source_static="$SMOKE_ROOT/static/${EXPECTED_STATIC_KEY}_datasetcc6d8a8764f5_world4_rank${rank}.pt"
    dest_static="$STATIC_DIR/${EXPECTED_STATIC_KEY}_world4_rank${rank}.pt"
    [[ -f "$source_static" ]] ||
        die "known smoke static cache missing for rank $rank"
    cp --reflink=auto --preserve=timestamps -- "$source_static" "$dest_static"
    COPIED_CACHE_FILES+=("$dest_static")
done
cp --reflink=auto --preserve=timestamps -- "$TOKEN_SOURCE" "$TOKEN_DEST"
COPIED_CACHE_FILES+=("$TOKEN_DEST")
chmod a-w "${COPIED_CACHE_FILES[@]}" "$STATIC_DIR" "$TOKENS_DIR"
cache_snapshot "$RUN_ROOT/provenance/cache_before.json"

note "recording only the explicitly selected GPU mapping"
nvidia-smi --id="$GPU_CSV" \
    --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total \
    --format=csv,noheader,nounits \
    >"$RUN_ROOT/provenance/selected_gpu_mapping.csv"
"$PYTHON_BIN" - \
    "$RUN_ROOT/provenance/selected_gpu_mapping.csv" "$GPU_CSV" \
    "$RUN_ROOT/provenance/selected_gpu_mapping.json" <<'PY'
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

mapping_path = Path(sys.argv[1])
selected_ids = [int(item) for item in sys.argv[2].split(",")]
destination = Path(sys.argv[3])
inventory = {}
with mapping_path.open(newline="", encoding="utf-8") as handle:
    for row in csv.reader(handle):
        if len(row) < 6:
            raise SystemExit(f"malformed nvidia-smi GPU mapping row: {row!r}")
        index = int(row[0].strip())
        inventory[index] = {
            "physical_gpu_id": index,
            "uuid": row[1].strip(),
            "name": row[2].strip(),
            "pci_bus_id": row[3].strip(),
            "driver_version": row[4].strip(),
            "memory_total_mib": int(float(row[5].strip())),
        }
missing = [index for index in selected_ids if index not in inventory]
unexpected = sorted(set(inventory) - set(selected_ids))
if missing or unexpected:
    raise SystemExit(
        f"selected GPU mapping mismatch: missing={missing}, "
        f"unexpected={unexpected}"
    )
rank_mapping = []
for rank, physical_id in enumerate(selected_ids):
    item = dict(inventory[physical_id])
    item.update(
        {
            "global_rank": rank,
            "local_rank": rank,
            "visible_cuda_device_index": rank,
        }
    )
    rank_mapping.append(item)
uuids = [item["uuid"] for item in rank_mapping]
if len(set(uuids)) != len(uuids):
    raise SystemExit(f"selected physical GPU UUIDs are not unique: {uuids}")
payload = {
    "schema_version": 1,
    "cuda_visible_devices": ",".join(str(item) for item in selected_ids),
    "physical_gpu_ids": selected_ids,
    "physical_gpu_uuids_in_rank_order": uuids,
    "rank_mapping": rank_mapping,
}
destination.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
nvidia-smi --id="$GPU_CSV" >"$RUN_ROOT/provenance/selected_hardware.txt" 2>&1
{
    printf 'CANOE_JOB_ID=%s\n' "${CANOE_JOB_ID:-}"
    printf 'CANOE_JOB_NAME=%s\n' "${CANOE_JOB_NAME:-}"
    printf 'JOB_ID=%s\n' "${JOB_ID:-}"
    printf 'HOSTNAME=%s\n' "${HOSTNAME:-}"
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "$GPU_CSV"
    printf 'CUBLAS_WORKSPACE_CONFIG=:4096:8\n'
    printf 'HF_DATASETS_OFFLINE=1\n'
    printf 'HF_HUB_OFFLINE=1\n'
    printf 'OMP_NUM_THREADS=1\n'
    printf 'TOKENIZERS_PARALLELISM=false\n'
    printf 'TRANSFORMERS_OFFLINE=1\n'
} >"$RUN_ROOT/provenance/environment.txt"

declare -a BASE_ARGS=(
    "${COMMON_CONFIG_ARGS[@]}"
    "${CASE_CONFIG_ARGS[@]}"
    --static_cache_path "$STATIC_DIR"
    --tokens_cache_path "$TOKENS_DIR"
    --cache_dir "$RUN_ROOT/runtime_cache"
    --output_dir "$RUN_ROOT/program_output"
    --exp "${CASE_NAME}_${label_slug}"
    --save_qmodel_path "$CHECKPOINT_PATH"
)
declare -a RESOLVED_ARGS=("${BASE_ARGS[@]}" "${CANDIDATE_CLI_ARGS[@]}")
declare -a TORCH_COMMAND=(
    "$PYTHON_BIN"
    -m torch.distributed.run
    --nproc_per_node "$WORLD_SIZE"
    --master_port "$MASTER_PORT"
    --module realq.ptq
    "${RESOLVED_ARGS[@]}"
)

"$PYTHON_BIN" - \
    "$RUN_ROOT/baseline_config.json" \
    "$RUN_ROOT/resolved_config.json" \
    "$RUN_ROOT/config_diff.json" \
    "$CHECKPOINT_PATH" \
    "${#CANDIDATE_NAMES[@]}" \
    "${CANDIDATE_NAMES[@]}" \
    -- "${BASE_ARGS[@]}" --candidate "${RESOLVED_ARGS[@]}" <<'PY'
from dataclasses import asdict
import json
from pathlib import Path
import sys

baseline_path = Path(sys.argv[1])
resolved_path = Path(sys.argv[2])
diff_path = Path(sys.argv[3])
checkpoint_path = str(Path(sys.argv[4]).resolve())
count = int(sys.argv[5])
names = sys.argv[6 : 6 + count]
rest = sys.argv[6 + count :]
separator = rest.index("--")
candidate_separator = rest.index("--candidate")
baseline_args = rest[separator + 1 : candidate_separator]
candidate_args = rest[candidate_separator + 1 :]
from realq.config import parse_cli

baseline = asdict(parse_cli(baseline_args))
resolved = asdict(parse_cli(candidate_args))
actual = {
    key: {"baseline": baseline.get(key), "candidate": resolved.get(key)}
    for key in sorted(set(baseline) | set(resolved))
    if baseline.get(key) != resolved.get(key)
}
checks = {
    "diff_exactly_matches_declared_fields": set(actual) == set(names),
    "checkpoint_enabled_at_exact_path": (
        str(Path(resolved["save_qmodel_path"]).resolve()) == checkpoint_path
    ),
    "evaluation_disabled": (
        resolved["skip_eval"] is True and resolved["lm_eval"] is False
    ),
    "one_layer_only": resolved["quant_stop_layer"] == 0,
    "nsys_disabled": resolved["nsys_profile"] is False,
}
baseline_path.write_text(
    json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
resolved_path.write_text(
    json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
diff_path.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "declared_candidate_fields": names,
            "actual_resolved_differences": actual,
            "checks": checks,
            "passed": all(checks.values()),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
if not all(checks.values()):
    raise SystemExit("unsafe or inconsistent resolved candidate config")
PY

quote_command() {
    local item
    for item in "$@"; do
        printf '%q ' "$item"
    done
    printf '\n'
}

{
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU_CSV"
    printf 'CUBLAS_WORKSPACE_CONFIG=%q ' ":4096:8"
    printf 'HF_DATASETS_OFFLINE=%q ' "1"
    printf 'HF_HUB_OFFLINE=%q ' "1"
    printf 'OMP_NUM_THREADS=%q ' "1"
    printf 'TOKENIZERS_PARALLELISM=%q ' "false"
    printf 'TRANSFORMERS_OFFLINE=%q ' "1"
    printf 'PYTHONPATH=%q ' "$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    quote_command "${TORCH_COMMAND[@]}"
} >"$RUN_ROOT/command.txt"

"$PYTHON_BIN" - \
    "$RUN_ROOT/manifest.json" "$run_id" "$MODE" "$commit_sha" \
    "$BASELINE_ID" "$LABEL" "$CASE_NAME" "$GPU_CSV" "$MASTER_PORT" \
    "$MODEL" "$SMOKE_ROOT" "$RUN_ROOT/provenance/selected_gpu_mapping.json" \
    "$RUN_ROOT/provenance/source_before.json" \
    "$RUN_ROOT/provenance/cache_before.json" \
    "$RUN_ROOT/resolved_config.json" "$RUN_ROOT/config_diff.json" \
    "$RUN_ROOT/command.txt" "${CANDIDATE_SPECS[@]}" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

(
    destination,
    run_id,
    mode,
    commit,
    baseline_id,
    label,
    case_name,
    gpu_csv,
    master_port,
    model,
    smoke_root,
    selected_mapping_raw,
    source_raw,
    cache_raw,
    config_raw,
    diff_raw,
    command_raw,
    *candidate_specs,
) = sys.argv[1:]


def sha256(raw: str) -> str:
    digest = hashlib.sha256()
    with Path(raw).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


selected = json.loads(Path(selected_mapping_raw).read_text(encoding="utf-8"))
source = json.loads(Path(source_raw).read_text(encoding="utf-8"))
cache = json.loads(Path(cache_raw).read_text(encoding="utf-8"))
source_cache_identity = [
    {
        "kind": pair["kind"],
        "rank": pair["rank"],
        "size_bytes": pair["source"]["size_bytes"],
        "mtime_ns": pair["source"]["mtime_ns"],
        "mode": pair["source"]["mode"],
        "sha256": pair["source"]["sha256"],
    }
    for pair in cache["pairs"]
]
payload = {
    "schema_version": 1,
    "run_id": run_id,
    "status": "prepared",
    "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
    "mode": mode,
    "gpu_execution_requires_explicit_run": True,
    "gpu_process_launched": False,
    "git_commit": commit,
    "baseline_id": baseline_id,
    "label": label,
    "case": case_name,
    "candidate_arg_specs": candidate_specs,
    "physical_gpu_ids": [int(item) for item in gpu_csv.split(",")],
    "physical_gpu_uuids_in_rank_order": selected[
        "physical_gpu_uuids_in_rank_order"
    ],
    "rank_to_physical_gpu_mapping": selected["rank_mapping"],
    "cuda_visible_devices": gpu_csv,
    "world_size": 4,
    "master_port": int(master_port),
    "model": model,
    "model_artifact_identity": source["model_artifact_identity"],
    "smoke_cache_root": smoke_root,
    "source_cache_identity": source_cache_identity,
    "source_provenance_path": source_raw,
    "source_provenance_sha256": sha256(source_raw),
    "cache_provenance_path": cache_raw,
    "cache_provenance_sha256": sha256(cache_raw),
    "selected_gpu_mapping_path": selected_mapping_raw,
    "selected_gpu_mapping_sha256": sha256(selected_mapping_raw),
    "resolved_config_path": config_raw,
    "resolved_config_sha256": sha256(config_raw),
    "config_diff_path": diff_raw,
    "config_diff_sha256": sha256(diff_raw),
    "command_path": command_raw,
    "command_sha256": sha256(command_raw),
    "harness_sha256": source["script_sha256"],
    "checkpoint_compare_tool_sha256": source["compare_tool_sha256"],
    "checkpoint_path": str(Path(destination).parent / "checkpoint" / "model.pt"),
    "checkpoint_state_manifest_path": str(
        Path(destination).parent / "checkpoint" / "state_manifest.json"
    ),
    "cache_files_are_post_run_hash_gated": True,
    "untracked_source_status_is_audit_only": True,
}
Path(destination).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
chmod a-w \
    "$RUN_ROOT/baseline_config.json" \
    "$RUN_ROOT/resolved_config.json" \
    "$RUN_ROOT/config_diff.json" \
    "$RUN_ROOT/command.txt"
printf '%s\n' "$RUN_ROOT" >"$RUN_ROOT/RUN_ROOT"
note "fresh correctness root prepared: $RUN_ROOT"

source_and_cache_gate() {
    local status="$1"
    source_snapshot "$RUN_ROOT/provenance/source_after.json"
    cache_snapshot "$RUN_ROOT/provenance/cache_after.json"
    "$PYTHON_BIN" - \
        "$RUN_ROOT" "$status" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys
from datetime import datetime, timezone

root = Path(sys.argv[1])
requested_status = sys.argv[2]
before = json.loads(
    (root / "provenance" / "source_before.json").read_text(encoding="utf-8")
)
after = json.loads(
    (root / "provenance" / "source_after.json").read_text(encoding="utf-8")
)
cache_before = json.loads(
    (root / "provenance" / "cache_before.json").read_text(encoding="utf-8")
)
cache_after = json.loads(
    (root / "provenance" / "cache_after.json").read_text(encoding="utf-8")
)
source_fields = (
    "git_commit",
    "tracked_diff_sha256",
    "tracked_diff_bytes",
    "staged_diff_sha256",
    "staged_diff_bytes",
    "script_sha256",
    "compare_tool_sha256",
    "model_artifact_identity",
)
source_unchanged = all(before.get(key) == after.get(key) for key in source_fields)
source_clean = (
    before["tracked_diff_bytes"] == 0
    and before["staged_diff_bytes"] == 0
    and after["tracked_diff_bytes"] == 0
    and after["staged_diff_bytes"] == 0
)
untracked_unchanged = (
    before["status_porcelain_v1_audit_only"]
    == after["status_porcelain_v1_audit_only"]
)


def cache_identity(payload: dict) -> list:
    fields = ("size_bytes", "mtime_ns", "mode", "sha256")
    return [
        {
            "kind": pair["kind"],
            "rank": pair["rank"],
            "source": {key: pair["source"][key] for key in fields},
            "destination": {
                key: pair["destination"][key] for key in fields
            },
        }
        for pair in payload["pairs"]
    ]


cache_unchanged = (
    cache_before["passed"]
    and cache_after["passed"]
    and cache_before["copied_cache_permissions_read_only"]
    and cache_after["copied_cache_permissions_read_only"]
    and cache_identity(cache_before) == cache_identity(cache_after)
)
passed = source_clean and source_unchanged and cache_unchanged
result = {
    "source_clean": source_clean,
    "source_unchanged": source_unchanged,
    "untracked_status_unchanged_audit_only": untracked_unchanged,
    "cache_unchanged": cache_unchanged,
}
if requested_status == "prepared":
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "prepared" if passed else "failed_prepare_gate",
            "dry_run_completed_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "gpu_process_launched": False,
            **result,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
(root / "provenance" / "source_cache_gate.json").write_text(
    json.dumps(
        {"schema_version": 1, "passed": passed, **result},
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
raise SystemExit(0 if passed else 1)
PY
}

if [[ "$MODE" == "prepare" ]]; then
    source_and_cache_gate prepared
    printf 'prepared %s\n' "$(date -u +%FT%TZ)" >"$RUN_ROOT/PREPARED"
    note "dry-run complete; no GPU process was launched"
    note "add --run to a new invocation to execute"
    exit 0
fi

assert_selected_gpus_idle() {
    local gpu_id candidate_busy busy=""
    for gpu_id in "${GPU_IDS[@]}"; do
        candidate_busy="$(
            nvidia-smi --id="$gpu_id" \
                --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
                --format=csv,noheader,nounits 2>/dev/null |
                sed '/^[[:space:]]*$/d;/No running processes found/d'
        )"
        if [[ -n "$candidate_busy" ]]; then
            busy+="${candidate_busy}"$'\n'
        fi
    done
    [[ -z "$busy" ]] ||
        die "selected GPU quartet is not idle:"$'\n'"$busy"
}

port_available() {
    "$PYTHON_BIN" - "$MASTER_PORT" <<'PY'
import socket
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
finally:
    sock.close()
PY
}

stop_active_processes() {
    local pid
    for pid in "${ACTIVE_PIDS[@]:-}"; do
        # Every registered child is launched through setsid, so its PID is
        # also its harness-owned process-group ID. A just-forked setsid child
        # may not have created that group yet; positive-PID fallback is safe
        # only while /proc still proves it is our shell's direct child.
        if ! kill -TERM -- "-$pid" 2>/dev/null; then
            if registered_pid_is_direct_child "$pid"; then
                kill -TERM "$pid" 2>/dev/null || true
            fi
        fi
    done
    for pid in "${ACTIVE_PIDS[@]:-}"; do
        wait "$pid" 2>/dev/null || true
    done
    ACTIVE_PIDS=()
}

registered_pid_is_direct_child() {
    local pid="$1"
    local key value ignored
    [[ -r "/proc/$pid/status" ]] || return 1
    while read -r key value ignored; do
        if [[ "$key" == "PPid:" ]]; then
            if [[ "$value" == "$$" ]]; then
                return 0
            fi
            return 1
        fi
    done <"/proc/$pid/status"
    return 1
}

cleanup_on_exit() {
    local exit_rc=$?
    if ((${#ACTIVE_PIDS[@]})); then
        note "unexpected exit; terminating only harness-owned process groups"
        stop_active_processes
        if [[ -n "${RUN_ROOT:-}" && -d "$RUN_ROOT" ]]; then
            printf 'aborted rc=%s %s\n' \
                "$exit_rc" "$(date -u +%FT%TZ)" \
                >"$RUN_ROOT/ABORTED" || true
        fi
    fi
    return "$exit_rc"
}

on_interrupt() {
    note "interrupted; terminating selected-quartet processes"
    stop_active_processes
    printf 'interrupted %s\n' "$(date -u +%FT%TZ)" >"$RUN_ROOT/INTERRUPTED"
    exit 130
}
trap on_interrupt INT TERM
trap cleanup_on_exit EXIT

assert_selected_gpus_idle
port_available
note "launching explicit selected quartet only: $GPU_CSV"
printf '%s\n' \
    "timestamp,index,uuid,memory_used_mib,memory_total_mib,utilization_gpu_percent,utilization_memory_percent" \
    >"$RUN_ROOT/gpu_telemetry.csv"
setsid nvidia-smi --id="$GPU_CSV" \
    --query-gpu=timestamp,index,uuid,memory.used,memory.total,utilization.gpu,utilization.memory \
    --format=csv,noheader,nounits --loop=5 \
    >>"$RUN_ROOT/gpu_telemetry.csv" \
    2>"$RUN_ROOT/gpu_telemetry.stderr" &
telemetry_pid=$!
ACTIVE_PIDS=("$telemetry_pid")

start_ns="$(date +%s%N)"
start_utc="$(date -u +%FT%T.%NZ)"
setsid env \
    "CUDA_VISIBLE_DEVICES=$GPU_CSV" \
    "CUBLAS_WORKSPACE_CONFIG=:4096:8" \
    "HF_DATASETS_OFFLINE=1" \
    "HF_HUB_OFFLINE=1" \
    "OMP_NUM_THREADS=1" \
    "TOKENIZERS_PARALLELISM=false" \
    "TRANSFORMERS_OFFLINE=1" \
    "PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "${TORCH_COMMAND[@]}" >"$RUN_ROOT/run.log" 2>&1 &
command_pid=$!
ACTIVE_PIDS=("$command_pid" "$telemetry_pid")

set +e
wait "$command_pid"
command_rc=$?
# The leader has been reaped. Retain only the still-live telemetry group so
# the EXIT trap never targets a stale/reusable command PID.
ACTIVE_PIDS=("$telemetry_pid")
set -e
end_ns="$(date +%s%N)"
end_utc="$(date -u +%FT%T.%NZ)"
kill -TERM -- "-$telemetry_pid" 2>/dev/null ||
    kill -TERM "$telemetry_pid" 2>/dev/null || true
wait "$telemetry_pid" 2>/dev/null || true
ACTIVE_PIDS=()

"$PYTHON_BIN" - \
    "$RUN_ROOT/process_wall.json" "$start_ns" "$end_ns" \
    "$start_utc" "$end_utc" "$command_rc" <<'PY'
import json
from pathlib import Path
import sys

path, start_ns, end_ns, start_utc, end_utc, command_rc = sys.argv[1:]
start_ns = int(start_ns)
end_ns = int(end_ns)
Path(path).write_text(
    json.dumps(
        {
            "schema_version": 1,
            "scope": "full_torchrun_process_wall",
            "start_utc": start_utc,
            "end_utc": end_utc,
            "start_epoch_ns": start_ns,
            "end_epoch_ns": end_ns,
            "elapsed_seconds": (end_ns - start_ns) / 1e9,
            "command_return_code": int(command_rc),
            "eligible_for_primary_speedup_claim": False,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

set +e
"$PYTHON_BIN" - \
    "$RUN_ROOT/run.log" "$CHECKPOINT_PATH" "$TOKEN_DEST" "$STATIC_DIR" \
    "$EXPECTED_STATIC_KEY" "$command_rc" "$GPU_CSV" \
    "$RUN_ROOT/gpu_telemetry.csv" "$RUN_ROOT/validation.json" <<'PY'
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

log_path = Path(sys.argv[1])
checkpoint_path = Path(sys.argv[2])
token_path = Path(sys.argv[3])
static_dir = Path(sys.argv[4])
static_key = sys.argv[5]
command_rc = int(sys.argv[6])
selected_ids = {int(item) for item in sys.argv[7].split(",")}
telemetry_path = Path(sys.argv[8])
destination = Path(sys.argv[9])
text = log_path.read_text(encoding="utf-8", errors="replace")
rank0_marker = (
    "[realq.precompute] cache hit (rank 0): "
    f"{static_dir / f'{static_key}_world4_rank0.pt'}"
)
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
    "static_cache_recompute": "all ranks will recompute Stage 0",
    "static_cache_write": "[realq.precompute] wrote rank-",
    "dataset_fetch": "Fetching dataset:",
    "token_save": "Saving tokens to",
    "unreadable_cache": "ignoring unreadable cache file",
    "invalid_cache": "ignoring invalid cache payload",
    "python_traceback": "Traceback (most recent call last)",
    "torchrun_child_failure": "ChildFailedError",
    "cuda_oom": "CUDA out of memory",
    "fatal_signal": "SignalException",
}
forbidden_counts = {
    name: text.count(marker) for name, marker in forbidden_markers.items()
}
observed_gpu_ids = set()
telemetry_rows = 0
if telemetry_path.is_file():
    with telemetry_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            telemetry_rows += 1
            try:
                observed_gpu_ids.add(int(row["index"]))
            except (KeyError, TypeError, ValueError):
                pass
checks = {
    "command_rc_zero": command_rc == 0,
    "collective_static_cache_hit": text.count(rank0_marker) >= 1,
    "token_cache_hit": text.count(token_marker) >= 1,
    "no_recompute_or_failure_markers": all(
        count == 0 for count in forbidden_counts.values()
    ),
    "quant_stop_marker": text.count(quant_stop_marker) >= 1,
    "checkpoint_save_marker": text.count(save_marker) >= 1,
    "checkpoint_exists_nonempty": (
        checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0
    ),
    "telemetry_contains_exact_selected_quartet": (
        observed_gpu_ids == selected_ids
    ),
}
report = {
    "schema_version": 1,
    "command_return_code": command_rc,
    "checks": checks,
    "rank0_static_cache_hit_count": text.count(rank0_marker),
    "token_cache_hit_count": text.count(token_marker),
    "quant_stop_marker_count": text.count(quant_stop_marker),
    "checkpoint_save_marker_count": text.count(save_marker),
    "forbidden_marker_counts": forbidden_counts,
    "selected_gpu_ids": sorted(selected_ids),
    "telemetry_gpu_ids": sorted(observed_gpu_ids),
    "telemetry_rows": telemetry_rows,
    "checkpoint_size_bytes": (
        checkpoint_path.stat().st_size if checkpoint_path.is_file() else None
    ),
    "passed": all(checks.values()),
}
destination.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
raise SystemExit(0 if report["passed"] else 1)
PY
validation_rc=$?
state_rc=1
if ((command_rc == 0 && validation_rc == 0)); then
    "$PYTHON_BIN" "$COMPARE_TOOL" state "$CHECKPOINT_PATH" \
        -o "$RUN_ROOT/checkpoint/state_manifest.json"
    state_rc=$?
fi
set -e

set +e
source_and_cache_gate run
source_cache_rc=$?
set -e

set +e
"$PYTHON_BIN" - \
    "$RUN_ROOT" "$command_rc" "$validation_rc" "$state_rc" \
    "$source_cache_rc" <<'PY'
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

root = Path(sys.argv[1])
command_rc, validation_rc, state_rc, source_cache_rc = map(
    int, sys.argv[2:]
)
manifest_path = root / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))


def load_or(path: Path, fallback: dict) -> dict:
    if not path.is_file():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


validation = load_or(
    root / "validation.json",
    {"passed": False, "error": "validation report is missing"},
)
wall = load_or(
    root / "process_wall.json",
    {"error": "process wall report is missing"},
)
gate = load_or(
    root / "provenance" / "source_cache_gate.json",
    {
        "passed": False,
        "source_clean": False,
        "source_unchanged": False,
        "untracked_status_unchanged_audit_only": False,
        "cache_unchanged": False,
        "error": "source/cache gate report is missing",
    },
)
state_path = root / "checkpoint" / "state_manifest.json"
state = (
    json.loads(state_path.read_text(encoding="utf-8"))
    if state_path.is_file()
    else None
)
peaks = {}
with (root / "gpu_telemetry.csv").open(
    newline="", encoding="utf-8"
) as handle:
    for row in csv.DictReader(handle):
        try:
            index = str(int(row["index"]))
            used = int(float(row["memory_used_mib"]))
        except (KeyError, TypeError, ValueError):
            continue
        peaks[index] = max(peaks.get(index, 0), used)
passed = (
    command_rc == 0
    and validation_rc == 0
    and state_rc == 0
    and source_cache_rc == 0
    and validation["passed"]
    and gate["passed"]
    and state is not None
    and state["canonical_tensor_count"] > 0
)
summary = {
    "schema_version": 1,
    "status": "passed" if passed else "failed",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "case": manifest["case"],
    "label": manifest["label"],
    "baseline_id": manifest["baseline_id"],
    "git_commit": manifest["git_commit"],
    "candidate_arg_specs": manifest["candidate_arg_specs"],
    "physical_gpu_ids": manifest["physical_gpu_ids"],
    "physical_gpu_uuids_in_rank_order": manifest[
        "physical_gpu_uuids_in_rank_order"
    ],
    "command_return_code": command_rc,
    "validation_return_code": validation_rc,
    "state_manifest_return_code": state_rc,
    "source_cache_gate_return_code": source_cache_rc,
    "validation": validation,
    "source_cache_gate": gate,
    "process_wall": wall,
    "sampled_gpu_memory_peak_mib": peaks,
    "checkpoint_archive_sha256": (
        state["checkpoint_archive_sha256"] if state is not None else None
    ),
    "canonical_state_sha256": (
        state["canonical_state_sha256"] if state is not None else None
    ),
    "canonical_tensor_count": (
        state["canonical_tensor_count"] if state is not None else None
    ),
    "eligible_for_correctness_comparison": passed,
    "eligible_for_primary_speedup_claim": False,
}
(root / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
manifest.update(
    {
        "status": summary["status"],
        "completed_at_utc": summary["completed_at_utc"],
        "gpu_process_launched": True,
        "source_clean": gate["source_clean"],
        "source_unchanged": gate["source_unchanged"],
        "untracked_status_unchanged_audit_only": gate[
            "untracked_status_unchanged_audit_only"
        ],
        "cache_unchanged": gate["cache_unchanged"],
        "canonical_state_sha256": summary["canonical_state_sha256"],
        "summary_path": str(root / "summary.json"),
        "eligible_for_correctness_comparison": passed,
        "eligible_for_primary_speedup_claim": False,
    }
)
manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
raise SystemExit(0 if passed else 1)
PY
final_rc=$?
set -e

if ((final_rc == 0)); then
    printf 'passed %s\n' "$(date -u +%FT%TZ)" >"$RUN_ROOT/COMPLETE"
    note "correctness/checkpoint gates passed"
    note "canonical state: $RUN_ROOT/checkpoint/state_manifest.json"
else
    note "correctness/checkpoint suite failed; inspect $RUN_ROOT"
fi
exit "$final_rc"
