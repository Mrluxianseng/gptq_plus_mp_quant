#!/usr/bin/env python3
"""Fail-closed 20-group REAL-Q single-linear-refresh campaign.

Two Canoe nodes each supervise eight single-GPU lanes.  Every LR search,
formal quantization, and reasoning invocation is world-size one.  Shared
precompute artifacts are produced once per model before consumers are
released.  OOM retries may only lower ``global_loss_bsz``; the sole explicit
exception is Qwen3-32B's user-authorized Hessian accumulation microbatch of
32 instead of the otherwise frozen value 64.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT = Path(__file__).resolve()
WORKSPACE = SCRIPT.parents[2]
MODULE = "experiments.realq_linear_refresh_20group_20260811.campaign"
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
CANONICAL_PYTHON = WORKSPACE / ".venv" / "bin" / "python"
CONTAINER_FALLBACK_PYTHON = (
    WORKSPACE / ".venv.py312-broken-20260729" / "bin" / "python"
)


def configured_python() -> Path:
    """Return one of the two audited repository-local environments.

    The canonical venv was rebuilt with the host's Python 3.10 and its base
    interpreter does not exist in the frozen Canoe image.  The preserved
    Python 3.12 venv is the previously validated environment for that image.
    Requiring an exact path keeps the compatibility exception fail-closed.
    """

    raw = os.environ.get("REALQ_PYTHON", str(CANONICAL_PYTHON))
    value = Path(os.path.abspath(os.fspath(Path(raw).expanduser())))
    allowed = {
        Path(os.path.abspath(os.fspath(CANONICAL_PYTHON))),
        Path(os.path.abspath(os.fspath(CONTAINER_FALLBACK_PYTHON))),
    }
    if value not in allowed:
        raise RuntimeError(
            f"REALQ_PYTHON must name an audited repository venv, got {value}"
        )
    return value


PYTHON = configured_python()
PREVIOUS_CAMPAIGN_ID = "realq-linear-refresh-unused-v0"
CAMPAIGN_ID = "realq-linear-refresh-llama31-qwen3-20group-20260811-v1"
DEFAULT_ROOT = (
    WORKSPACE.parent
    / "experiment_data"
    / "realq_linear_refresh_20group_20260811"
)
GPU_IDS = tuple(range(8))
TUNING_GLOBAL_LOSS_LADDER = (8, 4, 2, 1)
FORMAL_GLOBAL_LOSS_LADDER = (32, 16, 8, 4, 2, 1)
DEFAULT_HESSIAN_ACCUM_BSZ = 64
QWEN32_HESSIAN_ACCUM_BSZ = 32
CHECKPOINT_STABILITY_SECONDS = 60
EVALPLUS_SAMPLE_STABILITY_SECONDS = 30
EVALPLUS_PARALLEL = 32
EVALPLUS_CANDIDATE_STATUSES = frozenset({"pass", "fail", "timeout"})
SCORE_WORKER_SLOTS = 2
GENERATION_CLAIM_POLL_SECONDS = 2
GENERATION_CLAIM_MAX_WAIT_SECONDS = 6 * 60 * 60
OOM_RE = re.compile(
    r"CUDA out of memory|OutOfMemoryError|CUDA error: out of memory",
    re.IGNORECASE,
)
OVERLAY = WORKSPACE / "datasets" / "reasoning_eval" / "python_packages"
CUDA_ALLOCATOR_CONF = "expandable_segments:True"
CUDA_ALLOCATOR_ENV = {
    # PyTorch 2.9 prefers PYTORCH_ALLOC_CONF; keep the legacy spelling too
    # because the frozen CUDA image still prints it in OOM diagnostics.
    "PYTORCH_ALLOC_CONF": CUDA_ALLOCATOR_CONF,
    "PYTORCH_CUDA_ALLOC_CONF": CUDA_ALLOCATOR_CONF,
}


class CampaignError(RuntimeError):
    pass


def python_runtime_snapshot() -> dict[str, Any]:
    """Validate and record the exact repository environment inside Canoe."""

    if not PYTHON.is_file() or not os.access(PYTHON, os.X_OK):
        raise CampaignError(f"configured repository Python is not executable: {PYTHON}")
    venv = PYTHON.parent.parent
    os.environ["VIRTUAL_ENV"] = str(venv)
    os.environ["PATH"] = f"{venv / 'bin'}:{os.environ.get('PATH', '')}"
    torch_libs = sorted(venv.glob("lib/python*/site-packages/torch/lib"))
    if torch_libs:
        current = os.environ.get("LD_LIBRARY_PATH")
        os.environ["LD_LIBRARY_PATH"] = (
            f"{torch_libs[0]}:{current}" if current else str(torch_libs[0])
        )
    probe = (
        "import importlib.metadata as m,json,sys,torch,transformers;"
        "print(json.dumps({'python':sys.version.split()[0],"
        "'executable':sys.executable,'prefix':sys.prefix,"
        "'torch':torch.__version__,'transformers':transformers.__version__,"
        "'lm_eval':m.version('lm_eval')}))"
    )
    completed = subprocess.run(
        [str(PYTHON), "-c", probe],
        cwd=WORKSPACE,
        text=True,
        capture_output=True,
    )
    if completed.returncode:
        raise CampaignError(
            "repository Python preflight failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    try:
        snapshot = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise CampaignError(
            f"invalid repository Python preflight output: {completed.stdout!r}"
        ) from exc
    expected = {
        "prefix": str(venv),
        "torch": "2.9.1+cu128",
        "transformers": "4.56.2",
        "lm_eval": "0.4.4",
    }
    mismatches = {
        key: (snapshot.get(key), value)
        for key, value in expected.items()
        if snapshot.get(key) != value
    }
    if mismatches:
        raise CampaignError(f"repository Python preflight mismatch: {mismatches}")
    snapshot["configured_path"] = str(PYTHON)
    return snapshot


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    path: str
    final_layer_lr: float


@dataclass(frozen=True)
class QuantSpec:
    slug: str
    w_bits: int
    a_bits: int
    k_bits: int
    v_bits: int
    aware: bool
    lr_schedule: str


@dataclass(frozen=True)
class RunSpec:
    index: int
    model_index: int
    quant_index: int
    node: int

    @property
    def model(self) -> ModelSpec:
        return MODELS[self.model_index]

    @property
    def quant(self) -> QuantSpec:
        return QUANTS[self.quant_index]

    @property
    def run_id(self) -> str:
        return f"{self.model.slug}_{self.quant.slug}"


MODELS = (
    ModelSpec("qwen3-0.6b", "modelzoo/Qwen3/Qwen3-0.6B", 1e-5),
    ModelSpec(
        "llama31-8b-instruct",
        "modelzoo/Llama/Llama-3.1-8B-Instruct",
        1e-6,
    ),
    ModelSpec("qwen3-4b", "modelzoo/Qwen3/Qwen3-4B", 1e-5),
    ModelSpec("qwen3-8b", "modelzoo/Qwen3/Qwen3-8B", 1e-6),
    ModelSpec("qwen3-32b", "modelzoo/Qwen3/Qwen3-32B", 1e-6),
)

QUANTS = (
    QuantSpec("w4a16", 4, 16, 16, 16, False, "cosine"),
    QuantSpec("w4a4kv4", 4, 4, 4, 4, True, "none"),
    QuantSpec("w3a16", 3, 16, 16, 16, False, "cosine"),
    QuantSpec("w2a16", 2, 16, 16, 16, False, "cosine"),
)

RUNS = tuple(
    RunSpec(
        index=model_index * len(QUANTS) + quant_index,
        model_index=model_index,
        quant_index=quant_index,
        node=(model_index + quant_index) % 2,
    )
    for model_index in range(len(MODELS))
    for quant_index in range(len(QUANTS))
)
RUN_BY_ID = {run.run_id: run for run in RUNS}

TASKS = {
    "gsm8k": (1024, 32, 1319),
    "math_500": (2048, 16, 500),
    "humaneval_plus": (2048, 16, 164),
}

DATASET_PATHS = {
    "gsm8k": "datasets/reasoning_eval/gsm8k/test.jsonl",
    "math_500": "datasets/reasoning_eval/math_500/test.jsonl",
    "humaneval_plus": "datasets/reasoning_eval/humaneval_plus/HumanEvalPlus.jsonl.gz",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_inventories() -> dict[str, dict[str, Any]]:
    inventories = {}
    for task, relative in DATASET_PATHS.items():
        path = (WORKSPACE / relative).resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise CampaignError(f"reasoning dataset missing or empty: {path}")
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            count = sum(1 for line in handle if line.strip())
        expected = TASKS[task][2]
        if count != expected:
            raise CampaignError(
                f"reasoning dataset {task} expected {expected} rows, got {count}"
            )
        inventories[task] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "rows": count,
        }
    return inventories


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        # Canoe workers run as root while the host-side agent runs as the
        # workspace owner.  Keep manifests read-only but host-auditable.
        os.chmod(path, 0o644)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CampaignError(f"JSON root must be an object: {path}")
    return value


def output_root(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    forbidden = {Path("/").resolve(), Path.home().resolve(), WORKSPACE.resolve()}
    if path in forbidden or len(path.parts) < 4:
        raise CampaignError(f"unsafe campaign root: {path}")
    return path


def model_inventory(model: ModelSpec) -> dict[str, Any]:
    path = (WORKSPACE / model.path).resolve()
    if not (path / "config.json").is_file():
        raise CampaignError(f"model is incomplete: {path}")
    incomplete = sorted(path.glob("model*.safetensors.incomplete"))
    if incomplete:
        raise CampaignError(
            f"model still has incomplete shards: {', '.join(item.name for item in incomplete)}"
        )
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        index = read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise CampaignError(f"invalid safetensors index: {index_path}")
        raw_shard_names = list(weight_map.values())
        if not all(
            isinstance(name, str) and Path(name).name == name
            for name in raw_shard_names
        ):
            raise CampaignError(f"unsafe shard name in index: {index_path}")
        shard_names = sorted(set(raw_shard_names))
        shards = [path / name for name in shard_names]
    else:
        shards = [path / "model.safetensors"]
    missing = [item.name for item in shards if not item.is_file() or item.stat().st_size <= 0]
    if missing:
        raise CampaignError(f"model shards missing or empty in {path}: {missing}")
    actual_size = sum(item.stat().st_size for item in shards)
    if index_path.is_file():
        expected_size = index.get("metadata", {}).get("total_size")
        if not isinstance(expected_size, int) or expected_size <= 0:
            raise CampaignError(f"safetensors index has no valid total_size: {index_path}")
        header_overhead = actual_size - expected_size
        if header_overhead < 0 or header_overhead > 64 * 1024 * 1024:
            raise CampaignError(
                f"indexed safetensors size mismatch in {path}: "
                f"tensor_bytes={expected_size}, file_bytes={actual_size}"
            )
    if not (path / "tokenizer_config.json").is_file() or not any(
        (path / name).is_file()
        for name in ("tokenizer.json", "tokenizer.model", "vocab.json")
    ):
        raise CampaignError(f"tokenizer files are incomplete: {path}")
    return {
        "path": str(path),
        "shards": [
            {"name": item.name, "size_bytes": item.stat().st_size} for item in shards
        ],
        "index": str(index_path) if index_path.is_file() else None,
        "config_sha256": file_sha256(path / "config.json"),
        "index_sha256": file_sha256(index_path) if index_path.is_file() else None,
        "total_file_bytes": actual_size,
    }


def model_path(model: ModelSpec) -> str:
    return str(model_inventory(model)["path"])


def cache_root(root: Path, model: ModelSpec) -> Path:
    return root / "shared_cache" / model.slug


def run_root(root: Path, run: RunSpec) -> Path:
    return root / "runs" / run.run_id


def recover_interrupted_trial(
    root: Path, run: RunSpec, lr: float, reason: str
) -> dict[str, Any]:
    """Archive one proven-stale no-result trial so it can be relaunched.

    This is deliberately separate from normal scheduling.  The supervising
    agent must first prove that the owning Canoe job has terminated and must
    supply that evidence in ``reason``.  The interrupted launch remains
    charged against the tuner's 20-launch hard limit.
    """

    if not reason.strip():
        raise CampaignError("interrupted-trial recovery requires an audit reason")
    if (
        not isinstance(lr, (int, float))
        or not math.isfinite(float(lr))
        or float(lr) < 0
    ):
        raise CampaignError(f"invalid recovery LR: {lr}")
    normalized_lr = 0.0 if float(lr) == 0 else float(format(float(lr), ".15g"))
    key = format(normalized_lr, ".15g")
    slug = "lr_" + re.sub(r"[^0-9A-Za-z]+", "_", key).strip("_")
    tuning_root = run_root(root, run) / "tuning"
    state_path = tuning_root / "state.json"
    trial_dir = tuning_root / "trials" / slug
    result_path = trial_dir / "result.json"
    spec_path = trial_dir / "spec.json"
    lock_path = tuning_root / ".controller.lock"
    if not state_path.is_file() or not spec_path.is_file():
        raise CampaignError(f"interrupted trial state/spec is missing: {trial_dir}")
    if result_path.exists():
        raise CampaignError(f"refusing recovery because an atomic result exists: {result_path}")

    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignError(f"tuner controller is still active: {lock_path}") from exc
        state = read_json(state_path)
        running = state.get("running")
        if not isinstance(running, dict) or key not in running:
            raise CampaignError(f"state has no interrupted running record for lr={key}")
        record = running[key]
        if not isinstance(record, dict):
            raise CampaignError(f"invalid interrupted running record for lr={key}")
        spec = read_json(spec_path)
        if format(float(spec.get("lr", -1)), ".15g") != key:
            raise CampaignError(f"interrupted trial spec LR mismatch: {spec_path}")
        if spec.get("protocol_fingerprint") != state.get("protocol_fingerprint"):
            raise CampaignError(f"interrupted trial protocol mismatch: {spec_path}")

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive = (
            root
            / "diagnostics"
            / "interrupted_trials"
            / run.run_id
            / f"{slug}_{stamp}"
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            raise CampaignError(f"interrupted trial archive collision: {archive}")
        os.replace(trial_dir, archive)
        recovery = {
            "run_id": run.run_id,
            "lr": normalized_lr,
            "old_worker_pid": record.get("worker_pid"),
            "old_gpu": record.get("gpu"),
            "old_started_at": record.get("started_at"),
            "reason": reason.strip(),
            "archive_path": str(archive),
            "protocol_fingerprint": state.get("protocol_fingerprint"),
            "recovered_at": now(),
            "launch_count_charged": True,
        }
        atomic_json(archive / "recovery.json", recovery)
        running.pop(key)
        state.setdefault("interrupted_trial_recoveries", []).append(recovery)
        state["updated_at"] = now()
        atomic_json(state_path, state)
        return recovery


def common_args(
    root: Path,
    model: ModelSpec,
    quant: QuantSpec,
    *,
    global_loss_bsz: int,
) -> list[str]:
    cache = cache_root(root, model)
    quantized = quant.a_bits < 16
    clip = "0.9" if quantized else "1.0"
    return [
        "--model", model_path(model),
        "--dataset", "wikitext2",
        "--eval_datasets", "wikitext2",
        "--seed", "1",
        "--rotation_seed", "0",
        "--refresh_seed", "0",
        "--nsamples", "256",
        "--seq_len", "2048",
        "--eval_seq_len", "2048",
        "--w_bits", str(quant.w_bits),
        "--w_groupsize", "128",
        "--w_asym", "false",
        "--w_clip", "true",
        "--num_groups", "4",
        "--percdamp", "0.01",
        "--blocksize", "128",
        "--act_order", "true",
        # This campaign is the controlled ablation requested by the user:
        # retain Block-GD on the current linear's unquantized suffix, but do
        # not update any other linear in the Transformer block.
        "--full_block_refresh", "false",
        "--group_parallel_quant", "rank",
        "--global_loss_bsz", str(global_loss_bsz),
        "--saliency_clip_percentile", "0.99",
        "--grad_hessian_topk", "-1",
        "--grad_clip", "1.0",
        "--grad_lr_layer_schedule", quant.lr_schedule,
        "--grad_lr_layer_base_ratio", "0.01",
        "--backward_samples", "32",
        "--backward_bsz", "32",
        "--final_layer_backward_bsz", "32",
        "--a_loss_ratio", "1.0",
        "--a_loss_clip_scope", "local_backward_chunk",
        "--bsz", "128",
        "--hessian_accum_bsz", str(hessian_accum_bsz(model)),
        "--fsdp", "false",
        "--fsdp_cpu_offload", "false",
        "--cpu_master", "false",
        "--a_bits", str(quant.a_bits),
        "--a_groupsize", "-1",
        "--a_asym", "false",
        "--a_clip_ratio", clip,
        "--k_bits", str(quant.k_bits),
        "--k_groupsize", "-1",
        "--k_asym", "false",
        "--k_clip_ratio", clip,
        "--v_bits", str(quant.v_bits),
        "--v_groupsize", "-1",
        "--v_asym", "false",
        "--v_clip_ratio", clip,
        "--act_quant_aware_gptq", str(quant.aware).lower(),
        "--k_cache_quant_aware_gptq", str(quant.aware).lower(),
        "--loss_slide_window", "true",
        "--final_layer_grad_lr", str(model.final_layer_lr),
        "--kl_topk", "-1",
        "--rotate", "true",
        "--static_cache_path", str(cache / "static"),
        "--cache_dir", str(cache / "runtime"),
        "--tokens_cache_path", str(cache / "tokens"),
        "--quantizer_inner_fastpath", "true",
        "--w_clip_search_impl", "symmetric_union_exact",
        "--fisher_fp32_cache", "true",
        "--act_order_stitch_impl", "prefix_q_trailing_w_exact",
        "--w_clip_update_impl", "where_out",
        "--w_group_param_layout", "compact",
        "--prepared_clamp_bound_cache", "true",
        "--triton_column_block", "true",
        "--fused_block_adam", "true",
        "--attention_backend", "sdpa",
    ]


def hessian_accum_bsz(model: ModelSpec) -> int:
    """Return the one user-authorized model-specific Hessian microbatch."""

    return (
        QWEN32_HESSIAN_ACCUM_BSZ
        if model.slug == "qwen3-32b"
        else DEFAULT_HESSIAN_ACCUM_BSZ
    )


def validate_args(argv: Sequence[str]):
    from realq.config import parse_cli

    cfg = parse_cli(list(argv))
    if cfg.full_block_refresh:
        raise CampaignError(
            "linear-refresh campaign requires full_block_refresh=false"
        )
    return cfg


def tuning_precompute_args(
    root: Path, model: ModelSpec, global_loss_bsz: int
) -> list[str]:
    from tools.realq_auto_tune import tuning_realq_args

    base = common_args(root, model, QUANTS[0], global_loss_bsz=32)
    cache = cache_root(root, model)
    argv = tuning_realq_args(
        base,
        model=model_path(model),
        grad_lr=0,
        quant_stop_layer=0,
        static_cache_path=cache / "static",
        cache_dir=cache / "runtime",
        tokens_cache_path=cache / "tokens",
        output_dir=(
            root
            / "precompute"
            / model.slug
            / f"tuning_glbsz{global_loss_bsz}_output"
        ),
        exp=f"tuning_producer_glbsz{global_loss_bsz}",
        w_groupsize=128,
        producer=True,
    )
    offset = argv.index("--global_loss_bsz")
    argv[offset + 1] = str(global_loss_bsz)
    validate_args(argv)
    return argv


def formal_precompute_args(
    root: Path, model: ModelSpec, global_loss_bsz: int
) -> list[str]:
    argv = common_args(root, model, QUANTS[0], global_loss_bsz=global_loss_bsz)
    argv.extend(
        [
            "--grad_lr", "0",
            "--skip_eval", "true",
            "--skip_kl_ppl_eval", "true",
            "--lm_eval", "false",
            "--reasoning_eval", "false",
            "--exit_after_precompute", "true",
            "--require_static_cache_hit", "false",
            "--require_reference_cache_hit", "false",
            "--output_dir",
            str(root / "precompute" / model.slug / f"formal_glbsz{global_loss_bsz}"),
            "--exp", "formal_producer",
        ]
    )
    validate_args(argv)
    return argv


def tuner_command(
    root: Path,
    run: RunSpec,
    gpu: int,
    candidates: Sequence[float],
    tuning_global_loss_bsz: int = 8,
) -> list[str]:
    from tools.realq_auto_tune import lr_key

    directory = run_root(root, run)
    base = common_args(root, run.model, run.quant, global_loss_bsz=32)
    checkpoint = directory / "checkpoint" / "quantized.pt"
    base.extend(
        [
            "--skip_eval", "true",
            "--skip_kl_ppl_eval", "true",
            "--lm_eval", "false",
            "--reasoning_eval", "false",
            "--save_qmodel_path", str(checkpoint),
            "--output_dir", str(directory / "formal" / "realq_output"),
            "--exp", "formal",
        ]
    )
    validate_args(base)
    return [
        str(PYTHON),
        str(WORKSPACE / "tools" / "realq_auto_tune.py"),
        "--model", model_path(run.model),
        "--cuda-ids", str(gpu),
        "--parallelism", "1",
        "--tuning-w-groupsize", "128",
        "--tuning-global-loss-bsz", str(tuning_global_loss_bsz),
        "--output-root", str(directory / "tuning"),
        "--cache-root", str(cache_root(root, run.model)),
        "--controlled-candidates", ",".join(lr_key(value) for value in candidates),
        "--python", str(PYTHON),
        "--",
        *base,
    ]


def tuner_select_command(
    root: Path,
    run: RunSpec,
    selected_lr: float,
    reason: str,
    *,
    physical_zero_boundary: bool = False,
) -> list[str]:
    directory = run_root(root, run)
    base = common_args(root, run.model, run.quant, global_loss_bsz=32)
    checkpoint = directory / "checkpoint" / "quantized.pt"
    base.extend(
        [
            "--skip_eval", "true",
            "--skip_kl_ppl_eval", "true",
            "--lm_eval", "false",
            "--reasoning_eval", "false",
            "--save_qmodel_path", str(checkpoint),
            "--output_dir", str(directory / "formal" / "realq_output"),
            "--exp", "formal",
        ]
    )
    validate_args(base)
    marker = read_json(
        root / "precompute" / run.model.slug / "tuning_success.json"
    )
    tuning_global_loss_bsz = int(marker["tuning_global_loss_bsz"])
    command = [
        str(PYTHON),
        str(WORKSPACE / "tools" / "realq_auto_tune.py"),
        "--model", model_path(run.model),
        "--cuda-ids", "0",
        "--parallelism", "1",
        "--tuning-w-groupsize", "128",
        "--tuning-global-loss-bsz", str(tuning_global_loss_bsz),
        "--output-root", str(directory / "tuning"),
        "--cache-root", str(cache_root(root, run.model)),
        "--controlled-select-lr", format(selected_lr, ".15g"),
        "--decision-reason", reason,
        "--skip-gpu-check",
        "--python", str(PYTHON),
        "--",
        *base,
    ]
    if physical_zero_boundary:
        command.insert(command.index("--python"), "--allow-physical-zero-boundary")
    return command


def formal_args(
    root: Path,
    run: RunSpec,
    *,
    grad_lr: float,
    global_loss_bsz: int,
    attempt_dir: Path,
) -> list[str]:
    argv = common_args(
        root, run.model, run.quant, global_loss_bsz=global_loss_bsz
    )
    argv.extend(
        [
            "--grad_lr", format(grad_lr, ".17g"),
            "--skip_eval", "true",
            "--skip_kl_ppl_eval", "true",
            "--lm_eval", "false",
            "--reasoning_eval", "false",
            "--exit_after_precompute", "false",
            "--require_static_cache_hit", "true",
            "--require_reference_cache_hit", "false",
            "--save_qmodel_path",
            str(run_root(root, run) / "checkpoint" / "quantized.pt"),
            "--output_dir", str(attempt_dir / "realq_output"),
            "--exp", "formal",
        ]
    )
    validate_args(argv)
    return argv


def eval_args(root: Path, run: RunSpec, task: str, output: Path) -> list[str]:
    max_tokens, batch_size, _ = TASKS[task]
    checkpoint = run_root(root, run) / "checkpoint" / "quantized.pt"
    argv = common_args(root, run.model, run.quant, global_loss_bsz=32)
    argv.extend(
        [
            "--load_qmodel_path", str(checkpoint),
            "--skip_eval", "false",
            "--skip_kl_ppl_eval", "true",
            "--lm_eval", "false",
            "--reasoning_eval", "true",
            "--reasoning_tasks", task,
            "--reasoning_data_dir", str(WORKSPACE / "datasets/reasoning_eval"),
            "--reasoning_output_dir", str(output),
            "--reasoning_batch_size", str(batch_size),
            "--reasoning_limit", "-1",
            "--reasoning_max_new_tokens", str(max_tokens),
            "--reasoning_num_samples", "1",
            "--reasoning_apply_chat_template", "true",
            "--reasoning_enable_thinking", "true",
            "--reasoning_do_sample", "false",
            "--reasoning_temperature", "0",
            "--reasoning_top_p", "1",
            "--reasoning_top_k", "0",
            "--reasoning_seed", "1234",
            "--reasoning_resume", "true",
            "--reasoning_protocol", "realq_zero_shot_v1",
            "--reasoning_system_prompt",
            "You are a careful reasoning assistant. Follow the requested output format exactly.",
            "--require_static_cache_hit", "false",
            "--require_reference_cache_hit", "false",
            "--output_dir", str(output / "pipeline_output"),
            "--exp", task,
        ]
    )
    validate_args(argv)
    return argv


def worker_env(gpu: int, *, reasoning: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
        "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR",
        "MASTER_PORT", "TORCHELASTIC_RUN_ID",
    ):
        env.pop(key, None)
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_DATASETS_OFFLINE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        **CUDA_ALLOCATOR_ENV,
    )
    if reasoning:
        prior = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(OVERLAY), str(WORKSPACE), prior) if value
        )
    return env


def run_logged(
    command: Sequence[str],
    log_path: Path,
    gpu: int,
    *,
    reasoning: bool = False,
    deterministic_sdpa: bool = False,
) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("ab", buffering=0) as handle:
        header = (
            f"\n[{now()}] command={json.dumps(list(command), ensure_ascii=False)}\n"
            f"[{now()}] cuda_allocator_conf={CUDA_ALLOCATOR_CONF}\n"
        ).encode()
        handle.write(header)
        process = subprocess.Popen(
            list(command),
            cwd=WORKSPACE,
            env={
                **worker_env(gpu, reasoning=reasoning),
                **(
                    {"REALQ_DETERMINISTIC_SDPA": "1"}
                    if deterministic_sdpa
                    else {}
                ),
            },
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        returncode = process.wait()
        elapsed = time.monotonic() - started
        handle.write(
            f"[{now()}] returncode={returncode} elapsed_seconds={elapsed:.6f}\n".encode()
        )
    return returncode, elapsed


def log_is_oom(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as handle:
        with contextlib.suppress(OSError):
            handle.seek(-min(path.stat().st_size, 4 * 1024 * 1024), os.SEEK_END)
        tail = handle.read().decode("utf-8", errors="replace")
    return OOM_RE.search(tail) is not None


LOGGED_RUN_FOOTER_RE = re.compile(
    r"\[[^\]\n]+\] returncode=(-?\d+) "
    r"elapsed_seconds=([0-9]+(?:\.[0-9]+)?)\s*\Z"
)


def completed_logged_run(path: Path) -> tuple[int, float] | None:
    """Return the final run_logged footer, or None for an interrupted log.

    The footer must be the final non-whitespace content.  This deliberately
    rejects a historical footer followed by a newer command header so an
    interrupted retry can never be mistaken for a completed attempt.
    """

    if not path.is_file():
        return None
    with path.open("rb") as handle:
        with contextlib.suppress(OSError):
            handle.seek(-min(path.stat().st_size, 4 * 1024 * 1024), os.SEEK_END)
        tail = handle.read().decode("utf-8", errors="replace")
    match = LOGGED_RUN_FOOTER_RE.search(tail)
    if match is None:
        return None
    return int(match.group(1)), float(match.group(2))


def adopt_compatible_v4_tuning_marker(
    root: Path,
    model: ModelSpec,
    success: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Adopt a v4 producer marker whose cached mathematics is v5-invariant.

    The only numerical-configuration delta from v4 to v5 is the Qwen3-32B
    Hessian accumulation microbatch used after static/reference precompute.
    ``hessian_accum_bsz`` is intentionally absent from the static cache key,
    so all five already-completed producers remain exact inputs to v5.  This
    path validates the marker and concrete cache artifact, archives the old
    payload, and publishes an explicit adoption record instead of silently
    accepting a stale campaign identity.
    """

    if payload.get("campaign_id") != PREVIOUS_CAMPAIGN_ID:
        raise CampaignError(f"precompute identity mismatch: {success}")
    selected = payload.get("tuning_global_loss_bsz")
    attempts = payload.get("tuning_attempts")
    if (
        payload.get("profile") != "tuning"
        or payload.get("model") != dataclasses.asdict(model)
        or selected not in TUNING_GLOBAL_LOSS_LADDER
        or not isinstance(attempts, list)
        or not any(
            attempt.get("global_loss_bsz") == selected
            and attempt.get("returncode") == 0
            for attempt in attempts
            if isinstance(attempt, dict)
        )
    ):
        raise CampaignError(f"incompatible v4 precompute marker: {success}")

    from realq.precompute import cache as cache_mod

    cfg = validate_args(tuning_precompute_args(root, model, int(selected)))
    key = cache_mod.build_cache_key(cfg, 1)
    cache_path = Path(
        cache_mod.cache_path(str(cache_root(root, model) / "static"), key, 1, 0)
    )
    if not cache_path.is_file() or cache_path.stat().st_size <= 0:
        raise CampaignError(f"v4 precompute cache is missing or empty: {cache_path}")

    archive = (
        root
        / "diagnostics"
        / "campaign_v5_precompute_marker_adoption"
        / f"{model.slug}_tuning_success_v4.json"
    )
    if archive.is_file():
        if canonical_sha256(read_json(archive)) != canonical_sha256(payload):
            raise CampaignError(f"precompute marker archive collision: {archive}")
    else:
        atomic_json(archive, dict(payload))
    adopted = {
        **dict(payload),
        "campaign_id": CAMPAIGN_ID,
        "reused_from_campaign_id": PREVIOUS_CAMPAIGN_ID,
        "reused_marker_sha256": canonical_sha256(payload),
        "reused_cache_path": str(cache_path),
        "reuse_reason": (
            "v5 only lowers Qwen3-32B hessian_accum_bsz during the "
            "post-precompute quantization phase; the producer cache key and "
            "payload mathematics are unchanged"
        ),
        "adopted_at": now(),
    }
    atomic_json(success, adopted)
    return adopted


def ensure_tuning_precompute(
    root: Path, model: ModelSpec, gpu: int
) -> dict[str, Any]:
    directory = root / "precompute" / model.slug
    success = directory / "tuning_success.json"
    lock_path = directory / ".tuning.lock"
    directory.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if success.is_file():
            payload = read_json(success)
            if payload.get("campaign_id") != CAMPAIGN_ID:
                payload = adopt_compatible_v4_tuning_marker(
                    root, model, success, payload
                )
            return payload
        attempts = []
        selected = None
        for global_loss_bsz in TUNING_GLOBAL_LOSS_LADDER:
            tuning_log = directory / f"tuning_glbsz{global_loss_bsz}.log"
            code, tuning_seconds = run_logged(
                [
                    str(PYTHON),
                    "-m",
                    "realq.ptq",
                    *tuning_precompute_args(root, model, global_loss_bsz),
                ],
                tuning_log,
                gpu,
                deterministic_sdpa=True,
            )
            attempt = {
                "global_loss_bsz": global_loss_bsz,
                "elapsed_seconds": tuning_seconds,
                "returncode": code,
                "oom": bool(code and log_is_oom(tuning_log)),
                "cuda_allocator_conf": CUDA_ALLOCATOR_CONF,
                "log": str(tuning_log),
            }
            attempts.append(attempt)
            if code == 0:
                selected = global_loss_bsz
                break
            if not attempt["oom"]:
                raise CampaignError(
                    f"non-OOM tuning precompute failure: {tuning_log}"
                )
        if selected is None:
            raise CampaignError(
                f"tuning precompute exhausted OOM ladder: {model.slug}"
            )
        payload = {
            "campaign_id": CAMPAIGN_ID,
            "profile": "tuning",
            "model": dataclasses.asdict(model),
            "tuning_attempts": attempts,
            "tuning_global_loss_bsz": selected,
            "completed_at": now(),
        }
        atomic_json(success, payload)
        return payload


def ensure_formal_precompute(
    root: Path, model: ModelSpec, gpu: int
) -> dict[str, Any]:
    directory = root / "precompute" / model.slug
    success = directory / "formal_success.json"
    lock_path = directory / ".formal.lock"
    directory.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if success.is_file():
            payload = read_json(success)
            if payload.get("campaign_id") != CAMPAIGN_ID:
                raise CampaignError(f"precompute identity mismatch: {success}")
            return payload
        formal_attempts = []
        selected = None
        for global_loss_bsz in FORMAL_GLOBAL_LOSS_LADDER:
            log = directory / f"formal_glbsz{global_loss_bsz}.log"
            code, seconds = run_logged(
                [
                    str(PYTHON), "-m", "realq.ptq",
                    *formal_precompute_args(root, model, global_loss_bsz),
                ],
                log,
                gpu,
            )
            attempt = {
                "global_loss_bsz": global_loss_bsz,
                "elapsed_seconds": seconds,
                "returncode": code,
                "oom": bool(code and log_is_oom(log)),
                "cuda_allocator_conf": CUDA_ALLOCATOR_CONF,
                "log": str(log),
            }
            formal_attempts.append(attempt)
            if code == 0:
                selected = global_loss_bsz
                break
            if not attempt["oom"]:
                raise CampaignError(f"non-OOM formal precompute failure: {log}")
        if selected is None:
            raise CampaignError(f"formal precompute exhausted OOM ladder: {model.slug}")
        payload = {
            "campaign_id": CAMPAIGN_ID,
            "profile": "formal",
            "model": dataclasses.asdict(model),
            "formal_attempts": formal_attempts,
            "formal_global_loss_bsz": selected,
            "completed_at": now(),
        }
        atomic_json(success, payload)
        return payload


def checkpoint_stable(path: Path) -> dict[str, int]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise CampaignError(f"checkpoint missing or empty: {path}")
    before = path.stat()
    time.sleep(CHECKPOINT_STABILITY_SECONDS)
    after = path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise CampaignError(f"checkpoint did not stabilize: {path}")
    return {"size_bytes": after.st_size, "mtime_ns": after.st_mtime_ns}


def tuning_candidate_success(
    root: Path, run: RunSpec, batch_id: str, candidate_index: int
) -> Path:
    return (
        run_root(root, run)
        / "tuning"
        / "batches"
        / batch_id
        / f"candidate_{candidate_index:02d}_success.json"
    )


def run_tuning_candidate(
    root: Path,
    run: RunSpec,
    gpu: int,
    *,
    batch_id: str,
    candidates: Sequence[float],
    candidate_index: int,
) -> None:
    directory = run_root(root, run)
    if candidate_index < 0 or candidate_index >= len(candidates):
        raise CampaignError(f"invalid tuning candidate index: {candidate_index}")
    success = tuning_candidate_success(root, run, batch_id, candidate_index)
    if success.is_file():
        return
    if candidate_index > 0:
        prior = tuning_candidate_success(root, run, batch_id, candidate_index - 1)
        if not prior.is_file():
            raise CampaignError(
                f"controlled candidates must run low-to-high; missing {prior}"
            )
    marker = ensure_tuning_precompute(root, run.model, gpu)
    candidate = candidates[candidate_index]
    log = (
        directory
        / "tuning"
        / "batches"
        / batch_id
        / f"candidate_{candidate_index:02d}.log"
    )
    code, seconds = run_logged(
        tuner_command(
            root,
            run,
            gpu,
            [candidate],
            int(marker["tuning_global_loss_bsz"]),
        ),
        log,
        gpu,
    )
    if code:
        raise CampaignError(f"controlled tuner batch failed: {log}")
    state_path = directory / "tuning" / "state.json"
    state = read_json(state_path)
    if state.get("status") != "awaiting_agent_decision":
        raise CampaignError(f"unexpected controlled tuner state: {state.get('status')}")
    from tools.realq_auto_tune import lr_key

    result = state.get("trials", {}).get(lr_key(candidate))
    if not isinstance(result, dict) or result.get("status") != "succeeded":
        raise CampaignError(f"controlled tuner did not publish candidate: {candidate}")
    atomic_json(
        success,
        {
            "campaign_id": CAMPAIGN_ID,
            "run_id": run.run_id,
            "batch_id": batch_id,
            "candidate_index": candidate_index,
            "candidate": candidate,
            "tuning_global_loss_bsz": int(marker["tuning_global_loss_bsz"]),
            "result": result,
            "elapsed_seconds": seconds,
            "trial_launch_count": len(state.get("trials", {})),
            "completed_at": now(),
        },
    )


def run_experiment(root: Path, run: RunSpec, gpu: int) -> None:
    directory = run_root(root, run)
    success = directory / "formal_success.json"
    if success.is_file():
        return
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".formal.lock"
    with lock.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignError(
                f"formal run already has an active controller: {run.run_id}"
            ) from exc
        _run_experiment_locked(root, run, gpu)


def _run_experiment_locked(root: Path, run: RunSpec, gpu: int) -> None:
    directory = run_root(root, run)
    success = directory / "formal_success.json"
    if success.is_file():
        return
    marker = ensure_formal_precompute(root, run.model, gpu)
    directory.mkdir(parents=True, exist_ok=True)
    tuning_root = directory / "tuning"
    best_path = tuning_root / "best_lr.txt"
    state_path = tuning_root / "state.json"
    if not best_path.is_file() or not state_path.is_file():
        raise CampaignError(f"tuner did not publish best LR: {tuning_root}")
    state = read_json(state_path)
    if state.get("status") != "tuning_complete":
        raise CampaignError(
            f"agent-supervised LR decision is missing: {run.run_id} "
            f"({state.get('status')})"
        )
    decision_path = tuning_root / "controlled_decision.json"
    if not decision_path.is_file():
        raise CampaignError(f"controlled LR decision is missing: {decision_path}")
    decision = read_json(decision_path)
    best_lr = float(best_path.read_text().strip())
    selected_index = FORMAL_GLOBAL_LOSS_LADDER.index(
        int(marker["formal_global_loss_bsz"])
    )
    attempts = []

    def publish_success(global_loss_bsz: int) -> None:
        checkpoint = directory / "checkpoint" / "quantized.pt"
        stable = checkpoint_stable(checkpoint)
        payload = {
            "campaign_id": CAMPAIGN_ID,
            "run": dataclasses.asdict(run),
            "run_id": run.run_id,
            "selected_lr": best_lr,
            "selected_kl": state.get("selected_kl"),
            "controlled_decision": decision,
            "formal_attempts": attempts,
            "formal_global_loss_bsz": global_loss_bsz,
            "checkpoint": str(checkpoint),
            "checkpoint_stat": stable,
            "completed_at": now(),
        }
        atomic_json(success, payload)

    for global_loss_bsz in FORMAL_GLOBAL_LOSS_LADDER[selected_index:]:
        # A lower OOM retry needs its matching cache before a fail-closed consumer.
        if global_loss_bsz != int(marker["formal_global_loss_bsz"]):
            log = root / "precompute" / run.model.slug / f"retry_glbsz{global_loss_bsz}.log"
            lock = root / "precompute" / run.model.slug / f".glbsz{global_loss_bsz}.lock"
            with lock.open("a+") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                ready = log.with_suffix(".success.json")
                if not ready.is_file():
                    rc, seconds = run_logged(
                        [
                            str(PYTHON), "-m", "realq.ptq",
                            *formal_precompute_args(root, run.model, global_loss_bsz),
                        ],
                        log,
                        gpu,
                    )
                    if rc:
                        if log_is_oom(log):
                            continue
                        raise CampaignError(f"retry precompute failed: {log}")
                    atomic_json(ready, {"elapsed_seconds": seconds, "completed_at": now()})
        attempt_dir = directory / "formal" / f"glbsz{global_loss_bsz}"
        log = attempt_dir / "execution.log"
        argv = formal_args(
            root,
            run,
            grad_lr=best_lr,
            global_loss_bsz=global_loss_bsz,
            attempt_dir=attempt_dir,
        )
        config = attempt_dir / "config.json"
        expected_config = dataclasses.asdict(validate_args(argv))
        if attempt_dir.is_dir() and any(attempt_dir.iterdir()):
            if not config.is_file() or read_json(config) != expected_config:
                raise CampaignError(
                    f"formal retry config mismatch: {attempt_dir}"
                )
            completed = completed_logged_run(log)
            if completed is not None:
                rc, seconds = completed
                attempt = {
                    "global_loss_bsz": global_loss_bsz,
                    "elapsed_seconds": seconds,
                    "returncode": rc,
                    "oom": bool(rc and log_is_oom(log)),
                    "log": str(log),
                    "resumed_from_completed_log": True,
                }
                attempts.append(attempt)
                if rc == 0:
                    publish_success(global_loss_bsz)
                    return
                if attempt["oom"]:
                    continue
                raise CampaignError(f"non-OOM formal failure: {log}")

            archive = (
                root
                / "diagnostics"
                / "interrupted_formal_attempts"
                / f"{run.run_id}_glbsz{global_loss_bsz}_{time.time_ns()}"
            )
            archive.parent.mkdir(parents=True, exist_ok=True)
            attempt_dir.replace(archive)
            checkpoint = directory / "checkpoint" / "quantized.pt"
            archived_checkpoint = None
            if checkpoint.is_file():
                archived_checkpoint = archive / "interrupted_checkpoint_quantized.pt"
                checkpoint.replace(archived_checkpoint)
            atomic_json(
                archive / "recovery.json",
                {
                    "campaign_id": CAMPAIGN_ID,
                    "run_id": run.run_id,
                    "global_loss_bsz": global_loss_bsz,
                    "reason": "missing terminal run_logged footer",
                    "archived_checkpoint": (
                        str(archived_checkpoint) if archived_checkpoint else None
                    ),
                    "recovered_at": now(),
                },
            )
        atomic_json(config, expected_config)
        rc, seconds = run_logged(
            [str(PYTHON), "-m", "realq.ptq", *argv], log, gpu
        )
        attempt = {
            "global_loss_bsz": global_loss_bsz,
            "elapsed_seconds": seconds,
            "returncode": rc,
            "oom": bool(rc and log_is_oom(log)),
            "log": str(log),
        }
        attempts.append(attempt)
        if rc == 0:
            publish_success(global_loss_bsz)
            return
        if not attempt["oom"]:
            raise CampaignError(f"non-OOM formal failure: {log}")
    raise CampaignError(f"formal OOM ladder exhausted: {run.run_id}")


def audit_eval_output(output: Path, task: str) -> None:
    manifest = read_json(output / "manifest.json")
    if manifest.get("status") != "completed" or manifest.get("tasks") != [task]:
        raise CampaignError(f"incomplete reasoning manifest: {output}")
    generations = output / task / "generations.jsonl"
    count = sum(1 for line in generations.open(encoding="utf-8") if line.strip())
    expected = TASKS[task][2]
    if count != expected:
        raise CampaignError(f"{task}: expected {expected} generations, got {count}")


@contextlib.contextmanager
def generation_claim(output: Path, success: Path):
    """Claim one reasoning output with an atomic shared-filesystem mkdir.

    CPFS does not provide reliable ``flock`` exclusion for this campaign, even
    between two processes in the same Canoe pod.  Directory creation is the
    filesystem-level atomic primitive used here instead.  A dead owner is not
    reclaimed automatically: duplicating a generation writer is more harmful
    than failing closed and asking the supervisor to audit a stale claim.
    """

    claim = output / ".generation.claim"
    owner = claim / "owner.json"
    token = canonical_sha256(
        {
            "host": os.uname().nodename,
            "pid": os.getpid(),
            "started_ns": time.time_ns(),
            "output": str(output),
        }
    )
    started = time.monotonic()
    while True:
        if success.is_file():
            yield False
            return
        try:
            claim.mkdir()
        except FileExistsError:
            if time.monotonic() - started >= GENERATION_CLAIM_MAX_WAIT_SECONDS:
                owner_value: Any = "missing or unreadable"
                with contextlib.suppress(Exception):
                    owner_value = read_json(owner)
                raise CampaignError(
                    f"generation claim wait exceeded for {output}; owner={owner_value}"
                )
            time.sleep(GENERATION_CLAIM_POLL_SECONDS)
            continue
        try:
            atomic_json(
                owner,
                {
                    "campaign_id": CAMPAIGN_ID,
                    "host": os.uname().nodename,
                    "pid": os.getpid(),
                    "started_at": now(),
                    "token": token,
                },
            )
        except Exception:
            with contextlib.suppress(OSError):
                claim.rmdir()
            raise
        break
    try:
        if success.is_file():
            yield False
        else:
            yield True
    finally:
        owns_claim = False
        with contextlib.suppress(Exception):
            owns_claim = read_json(owner).get("token") == token
        if owns_claim:
            with contextlib.suppress(FileNotFoundError):
                owner.unlink()
            with contextlib.suppress(FileNotFoundError):
                claim.rmdir()


def run_evaluation(root: Path, run: RunSpec, gpu: int) -> None:
    if not (run_root(root, run) / "formal_success.json").is_file():
        raise CampaignError(f"formal result missing: {run.run_id}")
    reasoning = run_root(root, run) / "reasoning"
    reasoning.mkdir(parents=True, exist_ok=True)
    with (reasoning / ".eval.lock").open("a+") as handle:
        # Block instead of failing: manually backfilled idle GPU lanes may
        # overlap the normal node evaluator, whose second entrant should
        # simply resume from the atomic per-task success markers.
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        _run_evaluation_locked(root, run, gpu)


def _run_evaluation_locked(root: Path, run: RunSpec, gpu: int) -> None:
    for task in TASKS:
        run_evaluation_task(root, run, gpu, task, require_formal=False)


def run_evaluation_task(
    root: Path,
    run: RunSpec,
    gpu: int,
    task: str,
    *,
    require_formal: bool = True,
) -> None:
    """Run one reasoning task with an output-scoped atomic claim.

    The normal per-run evaluator remains serialized by ``.eval.lock``.  This
    narrower entry point is only for tail scheduling, where different tasks
    of the same immutable checkpoint can safely occupy otherwise idle GPUs.
    Each task has a disjoint output directory and an atomic success marker.
    """

    if task not in TASKS:
        raise CampaignError(f"unknown reasoning task: {task}")
    if require_formal and not (run_root(root, run) / "formal_success.json").is_file():
        raise CampaignError(f"formal result missing: {run.run_id}")
    output = run_root(root, run) / "reasoning" / task
    output.mkdir(parents=True, exist_ok=True)
    success = output / "generation_success.json"
    if success.is_file():
        return
    with generation_claim(output, success) as acquired:
        # Blocking is intentional: a duplicate entrant waits for the atomic
        # marker and then skips instead of launching a second writer.  Do not
        # replace this atomic-directory claim with flock on the shared CPFS.
        if not acquired:
            return
        if success.is_file():
            return
        log = output / "execution.log"
        argv = eval_args(root, run, task, output)
        atomic_json(output / "config.json", dataclasses.asdict(validate_args(argv)))
        rc, seconds = run_logged(
            [str(PYTHON), "-m", "realq.ptq", *argv],
            log,
            gpu,
            reasoning=True,
        )
        if rc:
            raise CampaignError(f"reasoning evaluation failed: {log}")
        audit_eval_output(output, task)
        atomic_json(
            success,
            {
                "campaign_id": CAMPAIGN_ID,
                "run_id": run.run_id,
                "task": task,
                "elapsed_seconds": seconds,
                "completed_at": now(),
                "official_code_score_pending": task == "humaneval_plus",
            },
        )


def run_evalplus_score(root: Path, run: RunSpec) -> None:
    """Run and audit the official HumanEval+ evaluator on a Canoe node."""

    reasoning = run_root(root, run) / "reasoning" / "humaneval_plus"
    generation_success = reasoning / "generation_success.json"
    if not generation_success.is_file():
        raise CampaignError(
            f"HumanEval+ generation result missing: {generation_success}"
        )
    official = reasoning / "official_eval"
    success = official / "official_success.json"
    if success.is_file():
        return
    official.mkdir(parents=True, exist_ok=True)
    lock = official / ".official.lock"
    with lock.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignError(
                f"official EvalPlus scorer already active: {run.run_id}"
            ) from exc
        if success.is_file():
            return

        samples = reasoning / "humaneval_plus" / "evalplus_samples.jsonl"
        dataset = WORKSPACE / DATASET_PATHS["humaneval_plus"]
        if not samples.is_file() or not dataset.is_file():
            raise CampaignError(
                f"HumanEval+ samples or dataset missing: {samples}, {dataset}"
            )
        sample_rows = [
            json.loads(line)
            for line in samples.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        with gzip.open(dataset, "rt", encoding="utf-8") as source:
            expected_ids = {
                json.loads(line)["task_id"] for line in source if line.strip()
            }
        sample_ids = [row.get("task_id") for row in sample_rows]
        if (
            len(sample_rows) != TASKS["humaneval_plus"][2]
            or len(expected_ids) != TASKS["humaneval_plus"][2]
            or len(set(sample_ids)) != len(sample_ids)
            or set(sample_ids) != expected_ids
            or any(not isinstance(row.get("solution"), str) for row in sample_rows)
        ):
            raise CampaignError(
                f"HumanEval+ sample identity/count gate failed: {samples}"
            )
        before = samples.stat()
        time.sleep(EVALPLUS_SAMPLE_STABILITY_SECONDS)
        after = samples.stat()
        if (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise CampaignError(f"HumanEval+ samples did not stabilize: {samples}")

        scorer = WORKSPACE / "tools" / "lowbit_activation_evalplus_canoe.sh"
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "REALQ_PYTHON": str(PYTHON),
        }
        started = time.monotonic()
        completed = subprocess.run(
            [
                "bash",
                str(scorer),
                str(samples),
                str(official),
                str(EVALPLUS_PARALLEL),
            ],
            cwd=WORKSPACE,
            env=environment,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        elapsed = time.monotonic() - started
        if completed.returncode:
            raise CampaignError(
                f"official EvalPlus scorer exited {completed.returncode}: {official}"
            )
        result = official / "evalplus_samples_eval_results.json"
        if not result.is_file():
            raise CampaignError(f"official EvalPlus result missing: {result}")
        payload = read_json(result)
        evaluations = payload.get("eval")
        if not isinstance(evaluations, dict) or set(evaluations) != expected_ids:
            raise CampaignError(
                f"official EvalPlus result does not cover 164 tasks: {result}"
            )
        base_pass = 0
        plus_pass = 0
        for task_id, candidates in evaluations.items():
            if not isinstance(candidates, list) or len(candidates) != 1:
                raise CampaignError(
                    f"EvalPlus candidate count mismatch: {task_id}"
                )
            candidate = candidates[0]
            # EvalPlus 0.3.1 defines pass/fail/timeout as the complete
            # untrusted-check status vocabulary.  A timeout is a valid
            # evaluated non-pass result, not a malformed or missing result.
            if candidate.get("base_status") not in EVALPLUS_CANDIDATE_STATUSES:
                raise CampaignError(f"invalid EvalPlus base status: {task_id}")
            if candidate.get("plus_status") not in EVALPLUS_CANDIDATE_STATUSES:
                raise CampaignError(f"invalid EvalPlus plus status: {task_id}")
            base_ok = candidate["base_status"] == "pass"
            base_pass += int(base_ok)
            plus_pass += int(base_ok and candidate["plus_status"] == "pass")
        total = TASKS["humaneval_plus"][2]
        atomic_json(
            success,
            {
                "campaign_id": CAMPAIGN_ID,
                "run_id": run.run_id,
                "status": "official_scored",
                "samples": str(samples),
                "samples_sha256": file_sha256(samples),
                "samples_bytes": after.st_size,
                "generation_success": str(generation_success),
                "generation_success_sha256": file_sha256(generation_success),
                "official_result": str(result),
                "official_result_sha256": file_sha256(result),
                "official_result_bytes": result.stat().st_size,
                "task_count": total,
                "base_pass": base_pass,
                "base_total": total,
                "base_pass_at_1": base_pass / total,
                "plus_pass": plus_pass,
                "plus_total": total,
                "plus_pass_at_1": plus_pass / total,
                "parallel": EVALPLUS_PARALLEL,
                "elapsed_seconds": elapsed,
                "completed_at": now(),
            },
        )


def gpu_snapshot() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    rows = []
    for line in completed.stdout.splitlines():
        index, name, memory, uuid = [part.strip() for part in line.split(",", 3)]
        rows.append(
            {"index": int(index), "name": name, "memory_total_mib": int(memory), "uuid": uuid}
        )
    if [row["index"] for row in rows] != list(GPU_IDS):
        raise CampaignError(f"expected exactly GPUs 0..7, got {rows}")
    return rows


def load_batch(path: Path) -> dict[str, Any]:
    from tools.realq_auto_tune import parse_controlled_candidates

    payload = read_json(path)
    if payload.get("campaign_id") != CAMPAIGN_ID:
        raise CampaignError(f"batch campaign mismatch: {path}")
    batch_id = payload.get("batch_id")
    candidates = payload.get("candidates")
    if not isinstance(batch_id, str) or not batch_id or not isinstance(candidates, dict):
        raise CampaignError(f"invalid batch manifest: {path}")
    plan_path = path.parent.parent / "plan.json"
    plan_fingerprint = read_json(plan_path).get("fingerprint")
    if (
        not isinstance(plan_fingerprint, str)
        or payload.get("plan_fingerprint") != plan_fingerprint
    ):
        raise CampaignError(f"batch plan fingerprint mismatch: {path}")
    unknown = set(candidates) - set(RUN_BY_ID)
    if unknown:
        raise CampaignError(f"batch has unknown runs: {sorted(unknown)}")
    normalized = {}
    for run_id, values in candidates.items():
        if not isinstance(values, list):
            raise CampaignError(f"batch candidates must be a list: {run_id}")
        normalized[run_id] = parse_controlled_candidates(
            ",".join(format(float(value), ".15g") for value in values)
        )
    identity = {
        "campaign_id": payload["campaign_id"],
        "plan_fingerprint": plan_fingerprint,
        "batch_id": batch_id,
        "candidates": normalized,
        "rationale": payload.get("rationale"),
    }
    expected = canonical_sha256(identity)
    if payload.get("fingerprint") != expected:
        raise CampaignError(f"batch fingerprint mismatch: {path}")
    return {**identity, "fingerprint": expected}


def write_batch(
    root: Path,
    *,
    batch_id: str,
    candidates: Mapping[str, Sequence[float]],
    rationale: str,
) -> Path:
    from tools.realq_auto_tune import parse_controlled_candidates

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", batch_id):
        raise CampaignError(f"unsafe batch id: {batch_id}")
    if not rationale.strip() or not candidates:
        raise CampaignError("batch rationale and candidates are required")
    unknown = set(candidates) - set(RUN_BY_ID)
    if unknown:
        raise CampaignError(f"batch has unknown runs: {sorted(unknown)}")
    normalized = {
        run_id: parse_controlled_candidates(
            ",".join(format(float(value), ".15g") for value in values)
        )
        for run_id, values in candidates.items()
    }
    plan_path = root / "plan.json"
    plan_fingerprint = read_json(plan_path).get("fingerprint")
    if not isinstance(plan_fingerprint, str):
        raise CampaignError(f"frozen plan fingerprint is missing: {plan_path}")
    identity = {
        "campaign_id": CAMPAIGN_ID,
        "plan_fingerprint": plan_fingerprint,
        "batch_id": batch_id,
        "candidates": normalized,
        "rationale": rationale.strip(),
    }
    payload = {
        **identity,
        "fingerprint": canonical_sha256(identity),
        "created_at": now(),
    }
    path = root / "batches" / f"{batch_id}.json"
    if path.is_file():
        existing = load_batch(path)
        if existing["fingerprint"] != payload["fingerprint"]:
            raise CampaignError(f"batch id already exists with another identity: {path}")
        return path
    atomic_json(path, payload)
    return path


def run_scheduler(
    root: Path,
    node: int,
    stage: str,
    *,
    batch: Mapping[str, Any] | None = None,
) -> None:
    if node not in (0, 1):
        raise CampaignError("node index must be 0 or 1")
    runs = [run for run in RUNS if run.node == node]
    pending: list[tuple[str, str, int]] = []
    if stage == "tune":
        if batch is None:
            raise CampaignError("tune stage requires a controlled batch manifest")
        selected_runs = [run for run in runs if run.run_id in batch["candidates"]]
        selected_models = {run.model.slug for run in selected_runs}
        for index, model in enumerate(MODELS):
            if index % 2 == node and model.slug in selected_models:
                pending.append(("tuning_precompute", model.slug, -1))
        for run in selected_runs:
            pending.extend(
                ("tune", run.run_id, index)
                for index in range(len(batch["candidates"][run.run_id]))
            )
    elif stage == "formal":
        missing = [
            run.run_id
            for run in runs
            if not (run_root(root, run) / "tuning" / "controlled_decision.json").is_file()
        ]
        if missing:
            raise CampaignError(f"formal stage is gated by LR decisions: {missing}")
        selected_models = {run.model.slug for run in runs}
        for index, model in enumerate(MODELS):
            if index % 2 == node and model.slug in selected_models:
                pending.append(("formal_precompute", model.slug, -1))
        pending.extend(("experiment", run.run_id, -1) for run in runs)
    elif stage == "eval":
        pending.extend(("eval", run.run_id, -1) for run in runs)
    elif stage == "score":
        missing = [
            f"{run.run_id}/{task}"
            for run in runs
            for task in TASKS
            if not (
                run_root(root, run)
                / "reasoning"
                / task
                / "generation_success.json"
            ).is_file()
        ]
        if missing:
            raise CampaignError(
                f"official scoring is gated by complete generation: {missing}"
            )
        pending.extend(("score", run.run_id, -1) for run in runs)
    else:
        raise CampaignError(stage)
    worker_slots = (
        tuple(range(SCORE_WORKER_SLOTS)) if stage == "score" else GPU_IDS
    )
    active: dict[int, tuple[tuple[str, str, int], subprocess.Popen[Any]]] = {}
    failures = []
    while pending or active:
        for gpu, (item, process) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            del active[gpu]
            if code:
                failures.append((item, code))
        if failures:
            # A fail-closed worker error must not SIGTERM unrelated GPU lanes:
            # doing so leaves no atomic trial result and turns a single failure
            # into a cascade of stale controller PIDs.  Stop dispatching new
            # work, drain already-launched workers to their atomic boundary,
            # then fail the node so the original error remains visible.
            if active:
                time.sleep(2)
                continue
            raise CampaignError(f"node {node} worker failures: {failures}")
        # Producers stay pinned to deterministic model-index parity. Cross-Pod
        # flock on the shared CPFS mount is not a reliable exclusion primitive,
        # so the secondary node waits for the primary node's success marker
        # instead of launching a concurrent failover producer.
        free = [slot for slot in worker_slots if slot not in active]
        for gpu in free:
            selected = None
            for offset, item in enumerate(pending):
                kind, identity, ordinal = item
                if kind == "tune":
                    run = RUN_BY_ID[identity]
                    if not (
                        root / "precompute" / run.model.slug / "tuning_success.json"
                    ).is_file():
                        continue
                if kind == "experiment":
                    run = RUN_BY_ID[identity]
                    if not (
                        root / "precompute" / run.model.slug / "formal_success.json"
                    ).is_file():
                        continue
                if kind == "tune" and ordinal > 0:
                    run = RUN_BY_ID[identity]
                    if not tuning_candidate_success(
                        root, run, str(batch["batch_id"]), ordinal - 1
                    ).is_file():
                        continue
                if kind == "eval":
                    run = RUN_BY_ID[identity]
                    if not (run_root(root, run) / "formal_success.json").is_file():
                        continue
                selected = pending.pop(offset)
                break
            if selected is None:
                break
            kind, identity, ordinal = selected
            # Module execution preserves the repository root on sys.path for
            # late imports such as ``tools.realq_auto_tune`` in worker roles.
            command = [str(PYTHON), "-m", MODULE]
            if kind in {"tuning_precompute", "formal_precompute"}:
                profile = "tuning" if kind == "tuning_precompute" else "formal"
                command.extend(
                    ["_precompute", "--model", identity, "--profile", profile]
                )
            elif kind == "tune":
                command.extend(
                    ["_tune", "--run-id", identity, "--batch-file", str(batch["path"])]
                )
                command.extend(["--candidate-index", str(ordinal)])
            elif kind == "experiment":
                command.extend(["_experiment", "--run-id", identity])
            elif kind == "score":
                command.extend(["_score", "--run-id", identity])
            else:
                command.extend(["_eval", "--run-id", identity])
            if kind == "score":
                command.extend(["--root", str(root)])
            else:
                command.extend(["--gpu", str(gpu), "--root", str(root)])
            process = subprocess.Popen(
                command,
                cwd=WORKSPACE,
                env={**os.environ, **CUDA_ALLOCATOR_ENV},
            )
            active[gpu] = (selected, process)
        time.sleep(2)


def write_plan(root: Path) -> dict[str, Any]:
    inventories = {model.slug: model_inventory(model) for model in MODELS}
    datasets = dataset_inventories()
    rows = []
    for run in RUNS:
        args = common_args(root, run.model, run.quant, global_loss_bsz=32)
        cfg = validate_args(args)
        rows.append(
            {
                "run_id": run.run_id,
                "node": run.node,
                "model": dataclasses.asdict(run.model),
                "quant": dataclasses.asdict(run.quant),
                "formal_config": dataclasses.asdict(cfg),
            }
        )
    payload = {
        "campaign_id": CAMPAIGN_ID,
        "created_at": now(),
        "workspace": str(WORKSPACE),
        "python": str(PYTHON),
        "tuning_global_loss_ladder": TUNING_GLOBAL_LOSS_LADDER,
        "formal_global_loss_ladder": FORMAL_GLOBAL_LOSS_LADDER,
        "hessian_accum_bsz_by_model": {
            model.slug: hessian_accum_bsz(model) for model in MODELS
        },
        "checkpoint_stability_seconds": CHECKPOINT_STABILITY_SECONDS,
        "weight_update_scope": "current_linear_trailing_columns",
        "full_block_refresh": False,
        "tasks": TASKS,
        "search": {
            "mode": "agent_supervised_controlled_batches",
            "max_trial_launches_per_run": 20,
            "selection_gate": (
                "global best observed point with strict lower/higher Exact-KL "
                "neighbours, or audited unique optimum at physical lr=0 with "
                "at least three worse positive-LR points"
            ),
            "tuning_deterministic_sdpa": True,
            "formal_deterministic_sdpa": False,
        },
        "model_inventories": inventories,
        "dataset_inventories": datasets,
        "runs": rows,
    }
    payload["fingerprint"] = canonical_sha256(payload)
    atomic_json(root / "plan.json", payload)
    return payload


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser()
    sub = top.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--root", default=str(DEFAULT_ROOT))
    node = sub.add_parser("run-node")
    node.add_argument("--root", default=str(DEFAULT_ROOT))
    node.add_argument("--node", type=int, required=True)
    node.add_argument(
        "--stage", choices=("tune", "formal", "eval", "score"), required=True
    )
    node.add_argument("--batch-file", default=None)
    select = sub.add_parser("select-lr")
    select.add_argument("--root", default=str(DEFAULT_ROOT))
    select.add_argument("--run-id", required=True)
    select.add_argument("--lr", type=float, required=True)
    select.add_argument("--reason", required=True)
    select.add_argument("--physical-zero-boundary", action="store_true")
    batch = sub.add_parser("make-batch")
    batch.add_argument("--root", default=str(DEFAULT_ROOT))
    batch.add_argument("--batch-id", required=True)
    batch.add_argument("--candidate-lrs", default=None)
    batch.add_argument("--run-ids", nargs="*", default=None)
    batch.add_argument(
        "--entry",
        action="append",
        default=[],
        help="per-run assignment RUN_ID=LR,LR; repeat as needed",
    )
    batch.add_argument("--rationale", required=True)
    recover = sub.add_parser("recover-interrupted-trial")
    recover.add_argument("--root", default=str(DEFAULT_ROOT))
    recover.add_argument("--run-id", required=True)
    recover.add_argument("--lr", type=float, required=True)
    recover.add_argument("--reason", required=True)
    for name in ("_precompute", "_tune", "_experiment", "_eval", "_eval-task"):
        child = sub.add_parser(name)
        child.add_argument("--root", required=True)
        child.add_argument("--gpu", type=int, required=True)
        if name == "_precompute":
            child.add_argument("--model", required=True)
            child.add_argument("--profile", choices=("tuning", "formal"), required=True)
        else:
            child.add_argument("--run-id", required=True)
        if name == "_eval-task":
            child.add_argument("--task", choices=tuple(TASKS), required=True)
        if name == "_tune":
            child.add_argument("--batch-file", required=True)
            child.add_argument("--candidate-index", type=int, required=True)
    score = sub.add_parser("_score")
    score.add_argument("--root", required=True)
    score.add_argument("--run-id", required=True)
    return top


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = output_root(args.root)
    if args.command == "plan":
        payload = write_plan(root)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if not (root / "plan.json").is_file():
        raise CampaignError(f"plan is missing: {root / 'plan.json'}")
    if args.command == "select-lr":
        run = RUN_BY_ID.get(args.run_id)
        if run is None:
            raise CampaignError(f"unknown run: {args.run_id}")
        completed = subprocess.run(
            tuner_select_command(
                root,
                run,
                args.lr,
                args.reason,
                physical_zero_boundary=args.physical_zero_boundary,
            ),
            cwd=WORKSPACE,
            text=True,
            capture_output=True,
        )
        if completed.returncode:
            raise CampaignError(
                f"controlled LR selection failed: {completed.stderr.strip()}"
            )
        print(completed.stdout, end="")
        return 0
    if args.command == "recover-interrupted-trial":
        run = RUN_BY_ID.get(args.run_id)
        if run is None:
            raise CampaignError(f"unknown run: {args.run_id}")
        recovery = recover_interrupted_trial(root, run, args.lr, args.reason)
        print(json.dumps(recovery, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "make-batch":
        if bool(args.candidate_lrs) == bool(args.entry):
            raise CampaignError(
                "provide exactly one of --candidate-lrs or repeated --entry"
            )
        assignments: dict[str, list[float]] = {}
        if args.candidate_lrs:
            run_ids = args.run_ids or list(RUN_BY_ID)
            values = [float(value) for value in args.candidate_lrs.split(",")]
            assignments = {run_id: values for run_id in run_ids}
        else:
            if args.run_ids:
                raise CampaignError("--run-ids is only valid with --candidate-lrs")
            for entry in args.entry:
                run_id, separator, raw = entry.partition("=")
                if not separator or run_id in assignments:
                    raise CampaignError(f"invalid or duplicate --entry: {entry}")
                assignments[run_id] = [float(value) for value in raw.split(",")]
        path = write_batch(
            root,
            batch_id=args.batch_id,
            candidates=assignments,
            rationale=args.rationale,
        )
        print(path)
        return 0
    if args.command == "run-node":
        batch = None
        if args.stage == "tune":
            if args.batch_file is None:
                raise CampaignError("tune stage requires --batch-file")
            batch_path = Path(args.batch_file).expanduser().resolve()
            batch = load_batch(batch_path)
            batch["path"] = str(batch_path)
        runtime = python_runtime_snapshot()
        snapshot = gpu_snapshot()
        snapshot_suffix = (
            f"_{batch['batch_id']}" if batch is not None else ""
        )
        atomic_json(
            root / "nodes" / f"node{args.node}_{args.stage}{snapshot_suffix}.json",
            {
                "campaign_id": CAMPAIGN_ID,
                "python_runtime": runtime,
                "gpu_snapshot": snapshot,
                "started_at": now(),
            },
        )
        run_scheduler(root, args.node, args.stage, batch=batch)
        return 0
    if args.command == "_score":
        run = RUN_BY_ID.get(args.run_id)
        if run is None:
            raise CampaignError(f"unknown run: {args.run_id}")
        run_evalplus_score(root, run)
        return 0
    if args.gpu not in GPU_IDS:
        raise CampaignError(f"invalid GPU: {args.gpu}")
    if args.command == "_precompute":
        model = next((model for model in MODELS if model.slug == args.model), None)
        if model is None:
            raise CampaignError(f"unknown model: {args.model}")
        if args.profile == "tuning":
            ensure_tuning_precompute(root, model, args.gpu)
        else:
            ensure_formal_precompute(root, model, args.gpu)
    elif args.command == "_tune":
        batch_path = Path(args.batch_file).expanduser().resolve()
        batch = load_batch(batch_path)
        run = RUN_BY_ID[args.run_id]
        candidates = batch["candidates"].get(run.run_id)
        if candidates is None:
            raise CampaignError(f"run is absent from batch: {run.run_id}")
        run_tuning_candidate(
            root,
            run,
            args.gpu,
            batch_id=batch["batch_id"],
            candidates=candidates,
            candidate_index=args.candidate_index,
        )
    elif args.command == "_experiment":
        run_experiment(root, RUN_BY_ID[args.run_id], args.gpu)
    elif args.command == "_eval-task":
        run_evaluation_task(
            root,
            RUN_BY_ID[args.run_id],
            args.gpu,
            args.task,
        )
    else:
        run_evaluation(root, RUN_BY_ID[args.run_id], args.gpu)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CampaignError as exc:
        print(f"campaign error: {exc}", file=sys.stderr)
        raise SystemExit(2)
