#!/usr/bin/env bash
# Isolated one-layer timing harness for the frozen Qwen3-4B REAL-Q fixture.
#
# Each invocation selects exactly one case and one physical four-GPU quartet.
# It prepares the validated Stage-0/token caches, runs one excluded warm-up,
# then at least three serial repetitions.  The primary metric is the maximum
# synchronized quantize-one-layer elapsed time across ranks.  Full process
# wall time and GPU telemetry are retained as supporting diagnostics.

set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true
umask 022

readonly SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly DEFAULT_MODEL="/minimax-avatar-new/zhangqian/realq/gptq_plus/modelzoo/Qwen3/Qwen3-4B"
readonly DEFAULT_SMOKE_ROOT="/minimax-avatar-new/zhangqian/realq/experiment_data/schedule_ablation_smoke_20260724"
readonly DEFAULT_OUTPUT_PARENT="/minimax-avatar-new/zhangqian/realq/experiment_data/perf_stage1_timing_zhangqian"
readonly EXPECTED_STATIC_KEY="Qwen3-4B_wikitext2_n4_sl128_b792a0815fdc"
readonly EXPECTED_TOKEN_SHA256="a0b5aa6f444868a7810a87774a96da3eb4dd4a0ad19bd6525e9f774f7f1c4cae"
readonly INSTRUMENTATION_SCHEMA_BASE_SHA="f8e5cca2c857e8e813fe6ec13e31b29a46345ab2"
readonly WORLD_SIZE=4
readonly MIN_FREE_KIB=$((8 * 1024 * 1024))

MODE=""
CASE_NAME=""
MODEL="$DEFAULT_MODEL"
SMOKE_ROOT="$DEFAULT_SMOKE_ROOT"
OUTPUT_PARENT="$DEFAULT_OUTPUT_PARENT"
PYTHON_BIN="${PYTHON:-python}"
GPU_CSV="0,1,2,3"
IDLE_SCOPE="all"
REPETITIONS=3
MASTER_PORT_BASE=29720
BASELINE_ID="qwen3-4b-layer0-f8e5cca"
LABEL="baseline"
declare -a CANDIDATE_SPECS=()
declare -a CANDIDATE_NAMES=()
declare -a CANDIDATE_CLI_ARGS=()
declare -a GPU_IDS=()
declare -a RUN_NAMES=()
declare -a ACTIVE_PIDS=()
RUN_ROOT=""

usage() {
    cat <<'EOF'
Usage:
  bash tools/run_performance_timing.sh --run --case CASE [options]
  bash tools/run_performance_timing.sh --prepare-only --case CASE [options]

Cases:
  group_stress       W2 group-128, A16/K16/V16, unaware
  per_row            W2 per-row, A16/K16/V16, unaware
  group_akv_aware    W2 group-128, A4/K4/V4 clip=0.9, aware

Modes:
  --run               Prepare, run one warm-up, then serial repetitions.
  --prepare-only      Prepare all immutable commands/configs; launch no GPU job.

Options:
  --case NAME                 One case listed above (required).
  --gpus I,J,K,L              Four physical GPU indices (default: 0,1,2,3).
  --idle-scope all|selected   Require all visible GPUs, or only the selected
                              quartet, to have no compute process (default: all).
  --repetitions N             Measured repetitions after warm-up (minimum 3).
  --master-port-base PORT     First of N+1 serial torchrun ports.
  --baseline-id ID            Comparison-family identifier stored verbatim.
  --label LABEL               Human-readable arm label stored verbatim.
  --candidate-arg NAME=VALUE  Append a known Config field override. Repeatable.
                              Fields present at the frozen instrumentation base
                              are rejected; only new optimization switches pass.
                              Every accepted override and resolved diff is
                              recorded in the manifest.
  --model PATH                Local Qwen3-4B model directory.
  --smoke-root PATH           Existing validated smoke cache root.
  --output-parent PATH        Parent of the new immutable run root.
  --python PATH               Python executable (default: $PYTHON or python).
  -h, --help                  Show this help.

Examples:
  bash tools/run_performance_timing.sh --run --case group_stress \
    --baseline-id p01 --label legacy

  bash tools/run_performance_timing.sh --run --case group_stress \
    --baseline-id p01 --label inner-fastpath \
    --candidate-arg quantizer_inner_fastpath=true

The harness never saves a checkpoint and always uses --skip_eval true,
--lm_eval false, --quant_stop_layer 0, and --perf_measure_layer 0.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

note() {
    printf '[realq-perf-timing] %s\n' "$*"
}

while (($#)); do
    case "$1" in
        --run)
            [[ -z "$MODE" ]] || die "choose exactly one mode"
            MODE="run"
            shift
            ;;
        --prepare-only)
            [[ -z "$MODE" ]] || die "choose exactly one mode"
            MODE="prepare"
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
        --idle-scope)
            (($# >= 2)) || die "--idle-scope requires all or selected"
            IDLE_SCOPE="$2"
            shift 2
            ;;
        --repetitions)
            (($# >= 2)) || die "--repetitions requires an integer"
            REPETITIONS="$2"
            shift 2
            ;;
        --master-port-base)
            (($# >= 2)) || die "--master-port-base requires an integer"
            MASTER_PORT_BASE="$2"
            shift 2
            ;;
        --baseline-id)
            (($# >= 2)) || die "--baseline-id requires a value"
            BASELINE_ID="$2"
            shift 2
            ;;
        --label)
            (($# >= 2)) || die "--label requires a value"
            LABEL="$2"
            shift 2
            ;;
        --candidate-arg)
            (($# >= 2)) || die "--candidate-arg requires NAME=VALUE"
            CANDIDATE_SPECS+=("$2")
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
case "$CASE_NAME" in
    group_stress|per_row|group_akv_aware) ;;
    "") die "--case is required" ;;
    *) die "unknown case: $CASE_NAME" ;;
esac
case "$IDLE_SCOPE" in
    all|selected) ;;
    *) die "--idle-scope must be all or selected" ;;
esac
[[ "$REPETITIONS" =~ ^[0-9]+$ ]] || die "--repetitions must be an integer"
((REPETITIONS >= 3)) || die "--repetitions must be at least 3"
[[ "$MASTER_PORT_BASE" =~ ^[0-9]+$ ]] ||
    die "--master-port-base must be an integer"
((MASTER_PORT_BASE >= 1024 && MASTER_PORT_BASE + REPETITIONS <= 65535)) ||
    die "requested torchrun port range is outside 1024..65535"
[[ -n "$BASELINE_ID" ]] || die "--baseline-id may not be empty"
[[ -n "$LABEL" ]] || die "--label may not be empty"

IFS=',' read -r -a GPU_IDS <<<"$GPU_CSV"
((${#GPU_IDS[@]} == WORLD_SIZE)) ||
    die "--gpus must contain exactly four physical indices"
declare -A SEEN_GPUS=()
for gpu_id in "${GPU_IDS[@]}"; do
    [[ "$gpu_id" =~ ^[0-9]+$ ]] || die "invalid physical GPU index: $gpu_id"
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
[[ -x "$PYTHON_BIN" ]] || die "python is not executable: $PYTHON_BIN"
command -v git >/dev/null || die "git is required"
command -v sha256sum >/dev/null || die "sha256sum is required"
command -v nvidia-smi >/dev/null || die "nvidia-smi is required"
command -v setsid >/dev/null || die "setsid is required"

cd "$REPO_ROOT"

repo_git() {
    # Workers may run as root against a user-owned shared checkout. Keep the
    # ownership exception command-local rather than mutating global Git config.
    git -c "safe.directory=$REPO_ROOT" "$@"
}

repo_git diff --quiet --ignore-submodules -- ||
    die "tracked working-tree changes exist; commit/stash them first"
repo_git diff --cached --quiet --ignore-submodules -- ||
    die "staged source changes exist; commit/stash them first"

gpu_count="$(nvidia-smi -L | wc -l)"
[[ "$gpu_count" =~ ^[0-9]+$ ]] || die "could not count visible GPUs"
for gpu_id in "${GPU_IDS[@]}"; do
    ((gpu_id < gpu_count)) ||
        die "physical GPU $gpu_id is not visible (count=$gpu_count)"
done
if [[ "$IDLE_SCOPE" == "all" ]]; then
    ((gpu_count >= 8)) ||
        die "--idle-scope all requires at least 8 visible GPUs; found $gpu_count"
fi

free_kib="$(df -Pk "$OUTPUT_PARENT" | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ ]] || die "could not determine free disk space"
((free_kib >= MIN_FREE_KIB)) ||
    die "less than 8 GiB free under $OUTPUT_PARENT"

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
    --perf_measure_layer 0
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
esac

# Candidate args are deliberately key/value Config fields, never raw shell
# fragments. Every Config field that existed at the frozen instrumentation
# base is mathematical/protocol state and is protected, including fields not
# explicitly present in COMMON_CONFIG_ARGS. Thus only newly introduced,
# independently switchable optimization fields can be varied by this harness.
declare -A PROTECTED_KEYS=()
declare -a INSTRUMENTATION_SCHEMA_BASE_CONFIG_FIELDS=(
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
for protected in "${INSTRUMENTATION_SCHEMA_BASE_CONFIG_FIELDS[@]}"; do
    PROTECTED_KEYS["$protected"]=1
done
for ((i = 0; i < ${#COMMON_CONFIG_ARGS[@]}; i += 2)); do
    PROTECTED_KEYS["${COMMON_CONFIG_ARGS[$i]#--}"]=1
done
for ((i = 0; i < ${#CASE_CONFIG_ARGS[@]}; i += 2)); do
    PROTECTED_KEYS["${CASE_CONFIG_ARGS[$i]#--}"]=1
done
for protected in \
    cache_dir static_cache_path tokens_cache_path output_dir exp \
    save_qmodel_path load_qmodel_path model_name optimized_rotation_path
do
    PROTECTED_KEYS["$protected"]=1
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
mkdir -p "$RUN_ROOT"/{provenance,runs,static,tokens}
readonly RUN_ROOT
readonly STATIC_DIR="$RUN_ROOT/static"
readonly TOKENS_DIR="$RUN_ROOT/tokens"
readonly TOKEN_DEST="$TOKENS_DIR/Qwen3-4B_wikitext2_train_n4_sl128_seed1.pt"

stop_active_processes() {
    local pid
    for pid in "${ACTIVE_PIDS[@]:-}"; do
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in "${ACTIVE_PIDS[@]:-}"; do
        wait "$pid" 2>/dev/null || true
    done
    ACTIVE_PIDS=()
}

on_interrupt() {
    note "interrupted; terminating active measurement processes"
    stop_active_processes
    printf 'interrupted %s\n' "$(date -u +%FT%TZ)" >"$RUN_ROOT/INTERRUPTED"
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
from pathlib import Path
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone

repo, script, model, destination = map(Path, sys.argv[1:])
sys.path.insert(0, str(repo))
from utils.cache_identity import artifact_identity


def git_bytes(*args: str) -> bytes:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={repo}", *args], cwd=repo
    )


def git_text(*args: str) -> str:
    return git_bytes(*args).decode("utf-8", errors="replace").rstrip()


def sha256(path: Path) -> str:
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
    "status_porcelain_v1": git_text("status", "--porcelain=v1", "-uall"),
    "script_path": str(script.resolve()),
    "script_sha256": sha256(script),
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
import sys
from datetime import datetime, timezone

smoke_root, static_dir, token_source, token_dest = map(Path, sys.argv[1:5])
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
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
if not payload["passed"]:
    raise SystemExit("cache identity validation failed")
PY
}

note "capturing source, model, cache, and hardware provenance"
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
[[ -f "$TOKEN_SOURCE" ]] || die "known smoke token cache is missing"
token_source_sha="$(sha256sum "$TOKEN_SOURCE" | awk '{print $1}')"
[[ "$token_source_sha" == "$EXPECTED_TOKEN_SHA256" ]] ||
    die "known smoke token SHA256 mismatch: $token_source_sha"

for rank in 0 1 2 3; do
    source_static="$SMOKE_ROOT/static/${EXPECTED_STATIC_KEY}_datasetcc6d8a8764f5_world4_rank${rank}.pt"
    dest_static="$STATIC_DIR/${EXPECTED_STATIC_KEY}_world4_rank${rank}.pt"
    [[ -f "$source_static" ]] ||
        die "known smoke static cache missing for rank $rank"
    cp --reflink=auto --preserve=timestamps -- "$source_static" "$dest_static"
done
cp --reflink=auto --preserve=timestamps -- "$TOKEN_SOURCE" "$TOKEN_DEST"
cache_snapshot "$RUN_ROOT/provenance/cache_before.json"

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
"$PYTHON_BIN" - \
    "$RUN_ROOT/provenance/gpu_mapping.csv" \
    "$GPU_CSV" \
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
if missing:
    raise SystemExit(f"selected physical GPUs missing from inventory: {missing}")
rank_mapping = []
for local_rank, physical_id in enumerate(selected_ids):
    item = dict(inventory[physical_id])
    item["global_rank"] = local_rank
    item["local_rank"] = local_rank
    item["visible_cuda_device_index"] = local_rank
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
    printf 'idle_scope=%s\n' "$IDLE_SCOPE"
} >"$RUN_ROOT/provenance/environment.txt"

quote_command() {
    local item
    for item in "$@"; do
        printf '%q ' "$item"
    done
    printf '\n'
}

declare -a RESOLVED_ARGS=()
declare -a BASE_ARGS=()
declare -a TORCH_COMMAND=()

build_command() {
    local run_dir="$1"
    local exp_name="$2"
    local master_port="$3"
    BASE_ARGS=(
        "${COMMON_CONFIG_ARGS[@]}"
        "${CASE_CONFIG_ARGS[@]}"
        --static_cache_path "$STATIC_DIR"
        --tokens_cache_path "$TOKENS_DIR"
        --cache_dir "$run_dir/runtime_cache"
        --output_dir "$run_dir/program_output"
        --exp "$exp_name"
    )
    RESOLVED_ARGS=("${BASE_ARGS[@]}" "${CANDIDATE_CLI_ARGS[@]}")
    TORCH_COMMAND=(
        "$PYTHON_BIN"
        -m torch.distributed.run
        --nproc_per_node "$WORLD_SIZE"
        --master_port "$master_port"
        --module realq.ptq
        "${RESOLVED_ARGS[@]}"
    )
}

prepare_run() {
    local run_name="$1"
    local phase="$2"
    local measured_index="$3"
    local ordinal="$4"
    local master_port=$((MASTER_PORT_BASE + ordinal))
    local run_dir="$RUN_ROOT/runs/$run_name"
    local exp_name="${CASE_NAME}_${label_slug}_${run_name}"
    mkdir -p "$run_dir"/{program_output,runtime_cache}
    build_command "$run_dir" "$exp_name" "$master_port"

    "$PYTHON_BIN" - \
        "$run_dir/baseline_config.json" \
        "$run_dir/resolved_config.json" \
        "$run_dir/config_diff.json" \
        "${#CANDIDATE_NAMES[@]}" \
        "${CANDIDATE_NAMES[@]}" \
        -- "${BASE_ARGS[@]}" \
        --candidate "${RESOLVED_ARGS[@]}" <<'PY'
from dataclasses import asdict
import json
from pathlib import Path
import sys

baseline_path = Path(sys.argv[1])
resolved_path = Path(sys.argv[2])
diff_path = Path(sys.argv[3])
count = int(sys.argv[4])
names = sys.argv[5 : 5 + count]
rest = sys.argv[5 + count :]
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
    # A declared candidate switch must actually change exactly that resolved
    # Config field. This rejects misspelled/no-op arms such as explicitly
    # selecting the current default while labeling the run as a candidate.
    "diff_exactly_matches_declared_fields": set(actual) == set(names),
    "no_checkpoint": resolved["save_qmodel_path"] is None,
    "no_eval": resolved["skip_eval"] is True and resolved["lm_eval"] is False,
    "one_layer_only": resolved["quant_stop_layer"] == 0,
    "layer_zero_instrumented": resolved["perf_measure_layer"] == 0,
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
    } >"$run_dir/command.txt"

    "$PYTHON_BIN" - \
        "$run_dir/manifest.json" "$run_id" "$BASELINE_ID" "$LABEL" \
        "$CASE_NAME" "$run_name" "$phase" "$measured_index" "$ordinal" \
        "$GPU_CSV" "$IDLE_SCOPE" "$master_port" \
        "$run_dir/resolved_config.json" "$run_dir/config_diff.json" \
        "$run_dir/command.txt" \
        "$RUN_ROOT/provenance/selected_gpu_mapping.json" \
        "${CANDIDATE_SPECS[@]}" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

(
    destination_raw,
    root_run_id,
    baseline_id,
    label,
    case_name,
    run_name,
    phase,
    measured_index,
    ordinal,
    gpu_csv,
    idle_scope,
    master_port,
    config_raw,
    diff_raw,
    command_raw,
    selected_mapping_raw,
    *candidate_specs,
) = sys.argv[1:]
destination = Path(destination_raw)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


config_path = Path(config_raw)
diff_path = Path(diff_raw)
command_path = Path(command_raw)
selected_mapping_path = Path(selected_mapping_raw)
selected_mapping = json.loads(
    selected_mapping_path.read_text(encoding="utf-8")
)
payload = {
    "schema_version": 1,
    "root_run_id": root_run_id,
    "baseline_id": baseline_id,
    "label": label,
    "case": case_name,
    "run_name": run_name,
    "phase": phase,
    "measured_index": (
        None if measured_index == "none" else int(measured_index)
    ),
    "ordinal": int(ordinal),
    "status": "prepared",
    "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
    "physical_gpu_ids": [int(item) for item in gpu_csv.split(",")],
    "physical_gpu_uuids_in_rank_order": selected_mapping[
        "physical_gpu_uuids_in_rank_order"
    ],
    "rank_to_physical_gpu_mapping": selected_mapping["rank_mapping"],
    "selected_gpu_mapping_path": str(selected_mapping_path),
    "selected_gpu_mapping_sha256": sha256(selected_mapping_path),
    "world_size": 4,
    "idle_scope": idle_scope,
    "master_port": int(master_port),
    "serial_isolated_run": True,
    "concurrent_with_harness_run": False,
    "warmup_excluded_from_summary": phase == "warmup",
    "candidate_arg_specs": candidate_specs,
    "resolved_config_path": str(config_path),
    "resolved_config_sha256": sha256(config_path),
    "config_diff_path": str(diff_path),
    "config_diff_sha256": sha256(diff_path),
    "command_path": str(command_path),
    "command_sha256": sha256(command_path),
    "run_log_path": str(destination.parent / "run.log"),
    "process_wall_path": str(destination.parent / "process_wall.json"),
    "gpu_telemetry_path": str(destination.parent / "gpu_telemetry.csv"),
    "timing_artifact_dir": str(
        destination.parent / "program_output"
    ),
}
destination.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
    chmod a-w \
        "$run_dir/baseline_config.json" \
        "$run_dir/resolved_config.json" \
        "$run_dir/config_diff.json" \
        "$run_dir/command.txt"
}

RUN_NAMES+=("warmup")
prepare_run "warmup" "warmup" "none" 0
for ((rep = 1; rep <= REPETITIONS; rep++)); do
    printf -v run_name 'rep_%03d' "$rep"
    RUN_NAMES+=("$run_name")
    prepare_run "$run_name" "measured" "$rep" "$rep"
done

"$PYTHON_BIN" - \
    "$RUN_ROOT/manifest.json" "$run_id" "$commit_sha" \
    "$INSTRUMENTATION_SCHEMA_BASE_SHA" "$MODE" \
    "$BASELINE_ID" "$LABEL" "$CASE_NAME" "$GPU_CSV" "$IDLE_SCOPE" \
    "$REPETITIONS" "$MODEL" "$SMOKE_ROOT" \
    "$RUN_ROOT/provenance/selected_gpu_mapping.json" \
    "${CANDIDATE_SPECS[@]}" <<'PY'
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

(
    destination,
    run_id,
    commit,
    instrumentation_schema_base_commit,
    mode,
    baseline_id,
    label,
    case_name,
    gpu_csv,
    idle_scope,
    repetitions,
    model,
    smoke_root,
    selected_mapping_raw,
    *candidate_specs,
) = sys.argv[1:]
selected_mapping_path = Path(selected_mapping_raw)
selected_mapping = json.loads(
    selected_mapping_path.read_text(encoding="utf-8")
)
payload = {
    "schema_version": 1,
    "run_id": run_id,
    "status": "prepared",
    "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
    "mode": mode,
    "git_commit": commit,
    "instrumentation_schema_base_commit": instrumentation_schema_base_commit,
    "baseline_id": baseline_id,
    "label": label,
    "case": case_name,
    "physical_gpu_ids": [int(item) for item in gpu_csv.split(",")],
    "physical_gpu_uuids_in_rank_order": selected_mapping[
        "physical_gpu_uuids_in_rank_order"
    ],
    "rank_to_physical_gpu_mapping": selected_mapping["rank_mapping"],
    "selected_gpu_mapping_path": str(selected_mapping_path),
    "world_size": 4,
    "idle_scope": idle_scope,
    "warmup_count": 1,
    "measured_repetitions": int(repetitions),
    "serial_isolated_run": True,
    "model": model,
    "smoke_cache_root": smoke_root,
    "candidate_arg_specs": candidate_specs,
    "checkpoint_saved": False,
    "evaluation_run": False,
    "perf_measure_layer": 0,
}
Path(destination).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

printf '%s\n' "$RUN_ROOT" >"$RUN_ROOT/RUN_ROOT"
note "fresh timing root prepared: $RUN_ROOT"
if [[ "$MODE" == "prepare" ]]; then
    note "prepare-only requested; no GPU process was launched"
    exit 0
fi

assert_gpus_idle() {
    local busy=""
    local gpu_id
    if [[ "$IDLE_SCOPE" == "all" ]]; then
        busy="$(
            nvidia-smi \
                --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
                --format=csv,noheader,nounits 2>/dev/null |
                sed '/^[[:space:]]*$/d;/No running processes found/d'
        )"
    else
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
    fi
    [[ -z "$busy" ]] ||
        die "GPU idle preflight failed (scope=$IDLE_SCOPE):"$'\n'"$busy"
}

port_available() {
    "$PYTHON_BIN" - "$1" <<'PY'
import socket
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
finally:
    sock.close()
PY
}

validate_run() {
    local run_name="$1"
    local command_rc="$2"
    local run_dir="$RUN_ROOT/runs/$run_name"
    "$PYTHON_BIN" - \
        "$run_dir" "$TOKEN_DEST" "$STATIC_DIR" "$EXPECTED_STATIC_KEY" \
        "$command_rc" "$run_dir/validation.json" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
token_path = Path(sys.argv[2])
static_dir = Path(sys.argv[3])
static_key = sys.argv[4]
command_rc = int(sys.argv[5])
destination = Path(sys.argv[6])
log_path = run_dir / "run.log"
config = json.loads(
    (run_dir / "resolved_config.json").read_text(encoding="utf-8")
)
run_manifest = json.loads(
    (run_dir / "manifest.json").read_text(encoding="utf-8")
)
expected_rank_mapping = run_manifest["rank_to_physical_gpu_mapping"]
if len(expected_rank_mapping) != 4:
    raise SystemExit(
        f"expected four rank-to-physical-GPU mappings, got "
        f"{len(expected_rank_mapping)}"
    )
text = log_path.read_text(encoding="utf-8", errors="replace")
output_dir = Path(config["output_dir"]).resolve()
exp = config["exp"]
artifact_dir = output_dir / exp


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


rank0_marker = (
    "[realq.precompute] cache hit (rank 0): "
    f"{static_dir / f'{static_key}_world4_rank0.pt'}"
)
token_marker = f"Loading tokens from {token_path}"
quant_stop_marker = (
    "[realq] quant_stop_layer=0 reached; remaining layers stay FP."
)
forbidden_markers = {
    "static_cache_miss": "[realq.precompute] cache miss",
    "static_cache_recompute_consensus": "all ranks will recompute Stage 0",
    "static_cache_write": "[realq.precompute] wrote rank-",
    "dataset_fetch": "Fetching dataset:",
    "token_save": "Saving tokens to",
    "unreadable_cache": "ignoring unreadable cache file",
    "invalid_cache": "ignoring invalid cache payload",
    "checkpoint_save": "[realq] reproducible quantized checkpoint saved",
    "python_traceback": "Traceback (most recent call last)",
    "torchrun_child_failure": "ChildFailedError",
    "cuda_oom": "CUDA out of memory",
    "fatal_signal": "SignalException",
}
forbidden_counts = {
    name: text.count(marker) for name, marker in forbidden_markers.items()
}
expected_paths = [
    artifact_dir / f"perf_measure_layer_0_rank{rank}.json"
    for rank in range(4)
]
discovered_paths = sorted(
    artifact_dir.glob("perf_measure_layer_*_rank*.json")
) if artifact_dir.is_dir() else []
rank_payloads = []
rank_errors = []
for expected_rank, path in enumerate(expected_paths):
    expected_gpu = expected_rank_mapping[expected_rank]
    if not path.is_file():
        rank_errors.append(f"missing rank timing file: {path}")
        continue
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        rank_errors.append(f"unreadable rank timing file {path}: {exc}")
        continue
    rank_checks = {
        "schema_version": payload.get("schema_version") == 1,
        "metric_name": payload.get("metric_name")
        == "quant_layer_critical_wall",
        "global_rank": payload.get("global_rank") == expected_rank,
        "local_rank": payload.get("local_rank") == expected_rank,
        "visible_cuda_device_index": (
            payload.get("cuda_device_index") == expected_rank
        ),
        "physical_gpu_uuid": (
            payload.get("cuda_device_uuid") == expected_gpu["uuid"]
        ),
        "world_size": payload.get("world_size") == 4,
        "layer_idx": payload.get("layer_idx") == 0,
        "output_dir": payload.get("output_dir") == str(output_dir),
        "exp": payload.get("exp") == exp,
        "positive_elapsed": isinstance(payload.get("elapsed_ns"), int)
        and payload["elapsed_ns"] > 0,
        "elapsed_consistent": (
            payload.get("end_perf_counter_ns", 0)
            - payload.get("start_perf_counter_ns", 0)
            == payload.get("elapsed_ns")
        ),
        "nonnegative_memory": all(
            isinstance(payload.get(key), int) and payload[key] >= 0
            for key in (
                "cuda_start_allocated_bytes",
                "cuda_end_allocated_bytes",
                "cuda_peak_allocated_bytes",
                "cuda_start_reserved_bytes",
                "cuda_end_reserved_bytes",
                "cuda_peak_reserved_bytes",
            )
        ),
    }
    if not all(rank_checks.values()):
        rank_errors.append(
            f"rank {expected_rank} schema mismatch: "
            + json.dumps(rank_checks, sort_keys=True)
        )
    rank_payloads.append(
        {
            "path": str(path),
            "sha256": sha256(path),
            "checks": rank_checks,
            "expected_physical_gpu": expected_gpu,
            "payload": payload,
        }
    )

elapsed = [
    item["payload"]["elapsed_ns"]
    for item in rank_payloads
    if isinstance(item["payload"].get("elapsed_ns"), int)
]
observed_uuids = [
    item["payload"].get("cuda_device_uuid")
    for item in rank_payloads
]
expected_uuids = [item["uuid"] for item in expected_rank_mapping]
checks = {
    "command_rc_zero": command_rc == 0,
    "collective_static_cache_hit": text.count(rank0_marker) >= 1,
    "token_cache_hit": text.count(token_marker) >= 1,
    "no_recompute_checkpoint_or_failure_markers": all(
        count == 0 for count in forbidden_counts.values()
    ),
    "quant_stop_marker": text.count(quant_stop_marker) >= 1,
    "resolved_no_checkpoint": config["save_qmodel_path"] is None,
    "resolved_no_eval": (
        config["skip_eval"] is True and config["lm_eval"] is False
    ),
    "resolved_layer_zero_only": (
        config["quant_stop_layer"] == 0
        and config["perf_measure_layer"] == 0
    ),
    "exactly_four_rank_files": (
        len(expected_paths) == len(discovered_paths) == len(rank_payloads) == 4
        and set(expected_paths) == set(discovered_paths)
    ),
    "rank_payloads_valid": not rank_errors,
    "rank_ordered_physical_gpu_uuid_mapping": (
        observed_uuids == expected_uuids
    ),
}
report = {
    "schema_version": 1,
    "command_return_code": command_rc,
    "checks": checks,
    "rank_errors": rank_errors,
    "rank_timing_artifacts": rank_payloads,
    "expected_rank_to_physical_gpu_mapping": expected_rank_mapping,
    "expected_physical_gpu_uuids_in_rank_order": expected_uuids,
    "observed_cuda_device_uuids_in_rank_order": observed_uuids,
    "rank_elapsed_ns": elapsed,
    "critical_layer_elapsed_ns": max(elapsed) if len(elapsed) == 4 else None,
    "critical_layer_elapsed_seconds": (
        max(elapsed) / 1e9 if len(elapsed) == 4 else None
    ),
    "fastest_rank_elapsed_ns": min(elapsed) if len(elapsed) == 4 else None,
    "rank_elapsed_spread_ns": (
        max(elapsed) - min(elapsed) if len(elapsed) == 4 else None
    ),
    "rank0_static_cache_hit_count": text.count(rank0_marker),
    "token_cache_hit_count": text.count(token_marker),
    "quant_stop_marker_count": text.count(quant_stop_marker),
    "forbidden_marker_counts": forbidden_counts,
    "passed": all(checks.values()),
}
destination.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
raise SystemExit(0 if report["passed"] else 1)
PY
}

finalize_run_manifest() {
    local run_name="$1"
    local run_dir="$RUN_ROOT/runs/$run_name"
    "$PYTHON_BIN" - \
        "$run_dir/manifest.json" "$run_dir/process_wall.json" \
        "$run_dir/validation.json" "$run_dir/gpu_telemetry.csv" <<'PY'
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

manifest_path, wall_path, validation_path, telemetry_path = map(
    Path, sys.argv[1:]
)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
wall = json.loads(wall_path.read_text(encoding="utf-8"))
validation = json.loads(validation_path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


peaks = {}
telemetry_uuids = {}
telemetry_uuid_conflicts = {}
sample_count = 0
if telemetry_path.is_file():
    with telemetry_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample_count += 1
            try:
                index = str(int(row["index"]))
                uuid = row["uuid"].strip()
                used = int(float(row["memory_used_mib"]))
            except (KeyError, TypeError, ValueError):
                continue
            previous_uuid = telemetry_uuids.setdefault(index, uuid)
            if previous_uuid != uuid:
                telemetry_uuid_conflicts.setdefault(index, set()).update(
                    (previous_uuid, uuid)
                )
            peaks[index] = max(peaks.get(index, 0), used)
expected_gpu_ids = {str(item) for item in manifest["physical_gpu_ids"]}
expected_uuid_by_id = {
    str(item["physical_gpu_id"]): item["uuid"]
    for item in manifest["rank_to_physical_gpu_mapping"]
}
telemetry_checks = {
    "file_exists_nonempty": (
        telemetry_path.is_file() and telemetry_path.stat().st_size > 0
    ),
    "at_least_one_sample_per_selected_gpu": (
        sample_count >= len(expected_gpu_ids)
        and expected_gpu_ids == set(peaks)
    ),
    "physical_gpu_uuid_mapping_exact": (
        not telemetry_uuid_conflicts
        and telemetry_uuids == expected_uuid_by_id
    ),
}
artifact_hashes = {
    "process_wall_sha256": sha256(wall_path),
    "validation_sha256": sha256(validation_path),
    "gpu_telemetry_sha256": (
        sha256(telemetry_path) if telemetry_path.is_file() else None
    ),
}
passed = validation["passed"] and all(telemetry_checks.values())
manifest.update(
    {
        "status": "passed" if passed else "failed",
        "process_wall": wall,
        "validation": validation,
        "sampled_gpu_memory_peak_mib": peaks,
        "gpu_telemetry_rows": sample_count,
        "gpu_telemetry_uuid_by_physical_id": telemetry_uuids,
        "expected_gpu_uuid_by_physical_id": expected_uuid_by_id,
        "gpu_telemetry_uuid_conflicts": {
            key: sorted(values)
            for key, values in telemetry_uuid_conflicts.items()
        },
        "gpu_telemetry_checks": telemetry_checks,
        "artifact_hashes": artifact_hashes,
    }
)
manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
raise SystemExit(0 if passed else 1)
PY
}

run_one() {
    local run_name="$1"
    local run_dir="$RUN_ROOT/runs/$run_name"
    local manifest="$run_dir/manifest.json"
    local master_port
    local telemetry_pid command_pid
    local command_rc=1 validation_rc=1 manifest_rc=1
    local start_ns end_ns start_utc end_utc

    master_port="$(
        "$PYTHON_BIN" - "$manifest" <<'PY'
import json
from pathlib import Path
import sys
print(json.loads(Path(sys.argv[1]).read_text())["master_port"])
PY
    )"
    assert_gpus_idle
    port_available "$master_port" ||
        die "torchrun port is unavailable: $master_port"
    build_command \
        "$run_dir" \
        "${CASE_NAME}_${label_slug}_${run_name}" \
        "$master_port"
    setsid bash -c '
        printf "%s\n" \
            "timestamp,index,uuid,memory_used_mib,memory_total_mib,utilization_gpu_percent,utilization_memory_percent"
        exec nvidia-smi \
            --id="$1" \
            --query-gpu=timestamp,index,uuid,memory.used,memory.total,utilization.gpu,utilization.memory \
            --format=csv,noheader,nounits \
            --loop=1
    ' _ "$GPU_CSV" >"$run_dir/gpu_telemetry.csv" \
        2>"$run_dir/gpu_telemetry.stderr" &
    telemetry_pid=$!

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
        "${TORCH_COMMAND[@]}" >"$run_dir/run.log" 2>&1 &
    command_pid=$!
    ACTIVE_PIDS=("$command_pid" "$telemetry_pid")

    set +e
    wait "$command_pid"
    command_rc=$?
    set -e
    end_ns="$(date +%s%N)"
    end_utc="$(date -u +%FT%T.%NZ)"
    kill -TERM -- "-$telemetry_pid" 2>/dev/null ||
        kill -TERM "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
    ACTIVE_PIDS=()

    "$PYTHON_BIN" - \
        "$run_dir/process_wall.json" "$start_ns" "$end_ns" \
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
            "serial_isolated_run": True,
            "primary_metric": False,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
    set +e
    validate_run "$run_name" "$command_rc"
    validation_rc=$?
    finalize_run_manifest "$run_name"
    manifest_rc=$?
    set -e
    if ((command_rc != 0 || validation_rc != 0 || manifest_rc != 0)); then
        return 1
    fi
    return 0
}

finalize_root() {
    local requested_status="$1"
    local requested_rc="$2"
    source_snapshot "$RUN_ROOT/provenance/source_after.json"
    cache_snapshot "$RUN_ROOT/provenance/cache_after.json"
    "$PYTHON_BIN" - \
        "$RUN_ROOT" "$requested_status" "$requested_rc" "$REPETITIONS" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys
from datetime import datetime, timezone

root = Path(sys.argv[1])
requested_status = sys.argv[2]
requested_rc = int(sys.argv[3])
repetitions = int(sys.argv[4])
manifest_path = root / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
    "status_porcelain_v1",
    "script_sha256",
)
source_unchanged = all(before[field] == after[field] for field in source_fields)
source_clean = (
    before["tracked_diff_bytes"] == 0
    and before["staged_diff_bytes"] == 0
    and after["tracked_diff_bytes"] == 0
    and after["staged_diff_bytes"] == 0
)


def cache_map(payload: dict) -> dict:
    return {
        (item["kind"], item["rank"]): (
            item["destination"]["size_bytes"],
            item["destination"]["sha256"],
        )
        for item in payload["pairs"]
    }


cache_unchanged = (
    cache_before["passed"]
    and cache_after["passed"]
    and cache_map(cache_before) == cache_map(cache_after)
)
runs = []
for manifest_file in sorted((root / "runs").glob("*/manifest.json")):
    runs.append(json.loads(manifest_file.read_text(encoding="utf-8")))
warmups = [item for item in runs if item["phase"] == "warmup"]
measured = [item for item in runs if item["phase"] == "measured"]
measured.sort(key=lambda item: item["measured_index"])
passed_runs = all(item.get("status") == "passed" for item in runs)
completed_measured = [
    item
    for item in measured
    if isinstance(item.get("validation"), dict)
    and isinstance(item.get("process_wall"), dict)
]


def describe(values: list[float]) -> dict | None:
    if not values:
        return None
    median = statistics.median(values)
    minimum = min(values)
    maximum = max(values)
    spread = maximum - minimum
    return {
        "count": len(values),
        "median": median,
        "min": minimum,
        "max": maximum,
        "range": spread,
        "range_over_median": spread / median if median else None,
        "range_over_median_percent": (
            100.0 * spread / median if median else None
        ),
        "population_stdev": statistics.pstdev(values),
        "coefficient_of_variation": (
            statistics.pstdev(values) / statistics.fmean(values)
            if statistics.fmean(values)
            else None
        ),
        "median_absolute_deviation": statistics.median(
            [abs(value - median) for value in values]
        ),
    }


critical_seconds = [
    item["validation"]["critical_layer_elapsed_seconds"]
    for item in completed_measured
]
process_seconds = [
    item["process_wall"]["elapsed_seconds"] for item in completed_measured
]
warmup_critical = (
    warmups[0]["validation"]["critical_layer_elapsed_seconds"]
    if len(warmups) == 1 and "validation" in warmups[0]
    else None
)
complete_shape = len(warmups) == 1 and len(measured) == repetitions
passed = (
    requested_status == "passed"
    and requested_rc == 0
    and complete_shape
    and passed_runs
    and source_unchanged
    and source_clean
    and cache_unchanged
)
summary = {
    "schema_version": 1,
    "status": "passed" if passed else "failed",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "baseline_id": manifest["baseline_id"],
    "label": manifest["label"],
    "case": manifest["case"],
    "git_commit": manifest["git_commit"],
    "instrumentation_schema_base_commit": manifest[
        "instrumentation_schema_base_commit"
    ],
    "candidate_arg_specs": manifest["candidate_arg_specs"],
    "physical_gpu_ids": manifest["physical_gpu_ids"],
    "physical_gpu_uuids_in_rank_order": manifest[
        "physical_gpu_uuids_in_rank_order"
    ],
    "rank_to_physical_gpu_mapping": manifest[
        "rank_to_physical_gpu_mapping"
    ],
    "primary_metric": (
        "max rank elapsed_ns for synchronized quantize_one_layer(layer=0)"
    ),
    "warmup_excluded": True,
    "warmup_critical_layer_seconds": warmup_critical,
    "measured_critical_layer_seconds": critical_seconds,
    "critical_layer_seconds": describe(critical_seconds),
    "full_process_wall_seconds": describe(process_seconds),
    "aa_repeat_dispersion": {
        "definition": "measured max-min divided by measured median",
        "absolute_seconds": (
            max(critical_seconds) - min(critical_seconds)
            if critical_seconds
            else None
        ),
        "relative_percent": (
            100.0
            * (max(critical_seconds) - min(critical_seconds))
            / statistics.median(critical_seconds)
            if critical_seconds and statistics.median(critical_seconds)
            else None
        ),
    },
    "per_repetition": [
        {
            "run_name": item["run_name"],
            "measured_index": item["measured_index"],
            "critical_layer_elapsed_seconds": item["validation"][
                "critical_layer_elapsed_seconds"
            ],
            "rank_elapsed_ns": item["validation"]["rank_elapsed_ns"],
            "rank_elapsed_spread_ns": item["validation"][
                "rank_elapsed_spread_ns"
            ],
            "full_process_wall_seconds": item["process_wall"][
                "elapsed_seconds"
            ],
            "sampled_gpu_memory_peak_mib": item[
                "sampled_gpu_memory_peak_mib"
            ],
        }
        for item in completed_measured
    ],
    "gates": {
        "requested_status_passed": requested_status == "passed",
        "requested_rc_zero": requested_rc == 0,
        "one_warmup_and_expected_repetitions": complete_shape,
        "all_run_validations_passed": passed_runs,
        "source_clean": source_clean,
        "source_unchanged": source_unchanged,
        "cache_unchanged": cache_unchanged,
    },
    "eligible_for_isolated_timing_comparison": passed,
}
(root / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
manifest.update(
    {
        "status": summary["status"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_status": requested_status,
        "requested_return_code": requested_rc,
        "source_unchanged": source_unchanged,
        "source_clean": source_clean,
        "cache_unchanged": cache_unchanged,
        "summary_path": str(root / "summary.json"),
        "eligible_for_isolated_timing_comparison": passed,
    }
)
manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
raise SystemExit(0 if passed else 1)
PY
}

overall_rc=0
for run_name in "${RUN_NAMES[@]}"; do
    note "starting serial isolated run: $run_name (${GPU_CSV})"
    if ! run_one "$run_name"; then
        overall_rc=1
        note "run failed validation: $run_name"
        break
    fi
done

if ((overall_rc == 0)); then
    requested_status="passed"
else
    requested_status="failed"
fi
set +e
finalize_root "$requested_status" "$overall_rc"
finalize_rc=$?
set -e
if ((finalize_rc != 0)); then
    overall_rc=1
fi

if ((overall_rc == 0)); then
    note "isolated timing suite passed: $RUN_ROOT/summary.json"
else
    note "isolated timing suite failed; inspect: $RUN_ROOT"
fi
exit "$overall_rc"
