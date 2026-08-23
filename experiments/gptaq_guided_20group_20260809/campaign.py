#!/usr/bin/env python3
"""Run GPTAQ/GuidedQuant and the frozen evaluations on two 8-GPU nodes.

The matrix is the exact five-model/four-setting matrix from
``experiments.realq_20group_20260808``.  Baseline quantization follows the
reviewed low-bit activation protocol: one GPU, WikiText-2 256x2048,
group/block 128, symmetric MSE clipping, act-order, QuaRot, GPTAQ alpha 0.25,
and one validated Guided saliency producer per model.  Checkpoints are saved
before evaluation and all work is restartable from atomic success markers.
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
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


SCRIPT = Path(__file__).resolve()
WORKSPACE = SCRIPT.parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
TIMING_DIR = WORKSPACE / "experiments" / "lowbit_activation"
if str(TIMING_DIR) not in sys.path:
    sys.path.insert(0, str(TIMING_DIR))

from experiments.realq_20group_20260808 import campaign as base  # noqa: E402
from experiments.realq_20group_20260808 import quality as quality  # noqa: E402
from tools import lowbit_activation_runner as legacy_runner  # noqa: E402
import formal_timing_adapter as timing_adapter  # noqa: E402


CAMPAIGN_ID = "gptaq-guided-llama31-qwen3-20group-20260809-v1"
SCHEMA_VERSION = 1
DEFAULT_ROOT = WORKSPACE.parent / "experiment_data" / "gptaq_guided_20group_20260809"
BASE_ROOT = WORKSPACE.parent / "experiment_data" / "realq_20group_20260808"
METHODS = ("gptaq", "guided_gptq")
METHOD_LABELS = {"gptaq": "GPTAQ", "guided_gptq": "GuidedQuant"}
SETTING_NAMES = {
    "w4a16": "W4A16",
    "w4a4kv4": "W4A4KV4",
    "w3a16": "W3A16",
    "w2a16": "W2A16",
}
GPU_IDS = tuple(range(8))
MAX_ATTEMPTS = 3
FAILURE_COOLDOWN_SECONDS = 60
CHECKPOINT_STABILITY_SECONDS = 10
SALIENCY_STABILITY_SECONDS = 5
CONTROLLER_POLL_SECONDS = 2
HEARTBEAT_SECONDS = 30
EVALPLUS_PARALLEL = 32
EVALPLUS_STATUSES = frozenset({"pass", "fail", "timeout"})
SOURCE_PATHS = (
    "ptq.py",
    "process_args.py",
    "save_grads.py",
    "gptq_utils/main.py",
    "gptq_utils/gptaq_utils.py",
    "gptq_utils/gptq_guided_utils.py",
    "gptq_utils/quant_aware_utils.py",
    "utils/checkpoint_utils.py",
    "utils/eval_utils.py",
    "utils/gradients.py",
    "utils/quant_utils.py",
    "utils/rotation_utils.py",
    "utils/saliency_utils.py",
    "tools/lowbit_activation_runner.py",
    "tools/validate_guided_saliency.py",
    "experiments/lowbit_activation/formal_timing_adapter.py",
    "experiments/lowbit_activation/formal_timing_sitecustomize/sitecustomize.py",
    "experiments/gptaq_guided_20group_20260809/campaign.py",
)


class CampaignError(RuntimeError):
    pass


@dataclass(frozen=True)
class Run:
    method_index: int
    base_index: int

    @property
    def method(self) -> str:
        return METHODS[self.method_index]

    @property
    def base_run(self) -> base.RunSpec:
        return base.RUNS[self.base_index]

    @property
    def model(self) -> base.ModelSpec:
        return self.base_run.model

    @property
    def quant(self) -> base.QuantSpec:
        return self.base_run.quant

    @property
    def run_id(self) -> str:
        return f"{self.method}_{self.base_run.run_id}"

    @property
    def node(self) -> int:
        return (self.base_run.model_index + self.base_run.quant_index + self.method_index) % 2


RUNS = tuple(
    Run(method_index, base_run.index)
    for method_index in range(len(METHODS))
    for base_run in base.RUNS
)
RUN_BY_ID = {run.run_id: run for run in RUNS}


@dataclass(frozen=True)
class Producer:
    model_index: int

    @property
    def model(self) -> base.ModelSpec:
        return base.MODELS[self.model_index]

    @property
    def work_id(self) -> str:
        return f"guided_saliency_{self.model.slug}"

    @property
    def node(self) -> int:
        return self.model_index % 2


PRODUCERS = tuple(Producer(index) for index in range(len(base.MODELS)))
PRODUCER_BY_MODEL = {producer.model.slug: producer for producer in PRODUCERS}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o644)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CampaignError(f"JSON root must be an object: {path}")
    return value


def safe_root(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    forbidden = {Path("/").resolve(), Path.home().resolve(), WORKSPACE.resolve(), BASE_ROOT.resolve()}
    if path in forbidden or len(path.parts) < 4:
        raise CampaignError(f"unsafe campaign root: {path}")
    return path


def run_dir(root: Path, run: Run) -> Path:
    return root / "runs" / run.method / run.base_run.run_id


def producer_dir(root: Path, producer: Producer) -> Path:
    return root / "producers" / producer.model.slug


def checkpoint_path(root: Path, run: Run) -> Path:
    return run_dir(root, run) / "checkpoint" / "model.pt"


def marker(root: Path, run: Run, name: str) -> Path:
    return run_dir(root, run) / f"{name}_success.json"


def source_identity() -> dict[str, str]:
    result = {}
    for relative in SOURCE_PATHS:
        path = WORKSPACE / relative
        if not path.is_file():
            raise CampaignError(f"required source missing: {path}")
        result[relative] = base.file_sha256(path)
    return result


def _model_source_cache(model: base.ModelSpec) -> tuple[Path, Path]:
    shared = base.cache_root(BASE_ROOT, model)
    token_files = sorted((shared / "tokens").glob("*.pt"))
    if len(token_files) != 1:
        raise CampaignError(f"expected one frozen token cache for {model.slug}: {token_files}")
    ref_files = sorted(
        (shared / "runtime" / "ref_logits").glob("*.cache"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    if not ref_files:
        raise CampaignError(f"no frozen reference cache for {model.slug}")
    # The earliest entry is the quantization-protocol cache.  Later entries
    # were produced by the deduplicated BF16 campaign with a different
    # rotation/runtime identity.  Qwen3-4B and Qwen3-32B filenames also match
    # the independently completed legacy baseline cache exactly.
    return token_files[0], ref_files[0]


def _ensure_symlink(target: Path, source: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve(strict=True) != source.resolve(strict=True):
            raise CampaignError(f"cache symlink changed: {target}")
        return
    if target.exists():
        raise CampaignError(f"refusing to replace existing cache path: {target}")
    relative = os.path.relpath(source, target.parent)
    target.symlink_to(relative)


def prepare_cache_links(root: Path) -> dict[str, Any]:
    cache = root / "cache" / "legacy"
    rows: dict[str, Any] = {}
    for model in base.MODELS:
        token_source, ref_source = _model_source_cache(model)
        model_name = Path(base.model_path(model)).name
        token_target = cache / "tokens" / f"{model_name}-wikitext2_s256_blk2048_seed1.pt"
        ref_target = cache / "ref_logits" / ref_source.name
        _ensure_symlink(token_target, token_source)
        _ensure_symlink(ref_target, ref_source)
        rows[model.slug] = {
            "token": {
                "source": str(token_source),
                "target": str(token_target),
                "size_bytes": token_source.stat().st_size,
                "source_mtime_ns": token_source.stat().st_mtime_ns,
            },
            "reference_logits": {
                "source": str(ref_source),
                "target": str(ref_target),
                "size_bytes": ref_source.stat().st_size,
                "source_mtime_ns": ref_source.stat().st_mtime_ns,
            },
        }
    return rows


def plan_payload(root: Path, caches: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for run in RUNS:
        rows.append(
            {
                "run_id": run.run_id,
                "node": run.node,
                "method": run.method,
                "method_label": METHOD_LABELS[run.method],
                "base_run_id": run.base_run.run_id,
                "model": dataclasses.asdict(run.model),
                "quant": dataclasses.asdict(run.quant),
                "setting_name": SETTING_NAMES[run.quant.slug],
            }
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "created_at": now(),
        "root": str(root),
        "base_campaign": {
            "root": str(BASE_ROOT),
            "plan_fingerprint": base.read_json(BASE_ROOT / "plan.json")["fingerprint"],
            "final_audit_fingerprint": base.read_json(BASE_ROOT / "final_audit.json")["audit_fingerprint"],
            "quality_plan_fingerprint": base.read_json(BASE_ROOT / "quality_plan.json")["fingerprint"],
        },
        "protocol": {
            "models": [model.slug for model in base.MODELS],
            "settings": [quant.slug for quant in base.QUANTS],
            "methods": list(METHODS),
            "quantized_runs": len(RUNS),
            "single_gpu_world_size": 1,
            "dataset": "wikitext2",
            "calibration": {"nsamples": 256, "seq_len": 2048, "seed": 1},
            "weight": {
                "groupsize": 128,
                "blocksize": 128,
                "symmetric": True,
                "mse_clip": True,
                "mse_norm": 2.4,
                "mse_grid": 50,
                "mse_maxshrink": 0.5,
                "mse_search": "cartesian_legacy",
                "act_order": True,
            },
            "rotation": {"enabled": True, "rotation_seed": 0},
            "gptaq_alpha": 0.25,
            "guided": {
                "num_groups": 4,
                "saliency_gradient_scale": 1000.0,
                "saliency_group_reduction": "squared_euclidean_sum",
                "act_order_score": "shared_unweighted_input_energy",
                "one_producer_per_model": True,
            },
            "activation_aware": {
                "a16": False,
                "a4kv4": True,
                "a4kv4_clip_ratio": 0.9,
            },
            "offload_inps": False,
            "quality": {
                "wikitext2_exact_ppl_kl": True,
                "paper_qa_tasks": list(quality.PAPER_QA_TASKS),
                "lm_eval_batch_size": quality.LM_EVAL_BATCH_SIZE,
            },
            "reasoning": {
                "tasks": list(base.TASKS),
                "protocol": "realq_zero_shot_v1",
                "official_humaneval_plus": True,
            },
            "checkpoint_before_evaluation": True,
            "quantization_gpu_hour_scope": "algorithm_core_v1",
        },
        "audited_python_paths": [
            str(base.CANONICAL_PYTHON),
            str(base.CONTAINER_FALLBACK_PYTHON),
        ],
        "model_inventories": {
            model.slug: base.model_inventory(model) for model in base.MODELS
        },
        "cache_links": dict(caches),
        "sources": source_identity(),
        "producers": [
            {"work_id": producer.work_id, "node": producer.node, "model": producer.model.slug}
            for producer in PRODUCERS
        ],
        "runs": rows,
    }
    payload["fingerprint"] = canonical_sha256(payload)
    return payload


def write_plan(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "plan.json"
    if path.exists():
        return load_plan(root)
    caches = prepare_cache_links(root)
    payload = plan_payload(root, caches)
    atomic_json(path, payload)
    return payload


def load_plan(root: Path) -> dict[str, Any]:
    path = root / "plan.json"
    plan = read_json(path)
    fingerprint = plan.get("fingerprint")
    identity = dict(plan)
    identity.pop("fingerprint", None)
    if fingerprint != canonical_sha256(identity):
        raise CampaignError(f"plan fingerprint mismatch: {path}")
    if plan.get("campaign_id") != CAMPAIGN_ID or Path(plan.get("root", "")) != root:
        raise CampaignError(f"plan identity mismatch: {path}")
    if plan.get("sources") != source_identity():
        raise CampaignError("numerical source changed after plan freeze")
    for model in base.MODELS:
        if plan["model_inventories"].get(model.slug) != base.model_inventory(model):
            raise CampaignError(f"model inventory changed: {model.slug}")
        cache_row = plan["cache_links"][model.slug]
        for kind in ("token", "reference_logits"):
            source = Path(cache_row[kind]["source"])
            target = Path(cache_row[kind]["target"])
            if not target.is_symlink() or target.resolve(strict=True) != source.resolve(strict=True):
                raise CampaignError(f"cache link changed: {target}")
            stat = source.stat()
            if stat.st_size != cache_row[kind]["size_bytes"] or stat.st_mtime_ns != cache_row[kind]["source_mtime_ns"]:
                raise CampaignError(f"frozen cache source changed: {source}")
    return plan


def legacy_plan(root: Path, *, cache_root: Path | None = None) -> dict[str, Any]:
    settings = {
        SETTING_NAMES[quant.slug]: {
            "w_bits": quant.w_bits,
            "a_bits": quant.a_bits,
            "k_bits": quant.k_bits,
            "v_bits": quant.v_bits,
        }
        for quant in base.QUANTS
    }
    return {
        "models": {model.slug: base.model_path(model) for model in base.MODELS},
        "settings": settings,
        "resolutions": {
            "final_nsamples_policy": "all_quantized_runs_256",
            "q4_protocol": "paper_a4k4v4_no_independent_q",
            "a16_aware_semantics": "weight_only_aware_not_applicable",
        },
        "fixed_numerics": {
            "dataset": "wikitext2",
            "eval_seq_len": 2048,
            "seed": 1,
            "rotation_seed": 0,
            "refresh_seed": 0,
            "activation_clip_ratio": 0.9,
            "num_groups": 4,
            "percdamp": 0.01,
            "kl_topk": -1,
        },
        "final": {
            "nsamples": 256,
            "seq_len": 2048,
            "blocksize": 128,
            "lm_eval_batch_size": quality.LM_EVAL_BATCH_SIZE,
        },
        "baseline_numerics": {"offload_inps": False, "gptaq_alpha": 0.25},
        "runtime_environment": {
            "HF_HOME": str(WORKSPACE / "datasets" / "lm_eval_hf_cache"),
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_TRUST_REMOTE_CODE": "1",
            "PYTORCH_ALLOC_CONF": base.CUDA_ALLOCATOR_CONF,
            "PYTORCH_CUDA_ALLOC_CONF": base.CUDA_ALLOCATOR_CONF,
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        },
        "python_environment": {"venv": str(base.PYTHON.parent.parent)},
        "output_root": str(root / "legacy_output"),
        "legacy_cache_root": str(cache_root or (root / "cache" / "legacy")),
    }


def _set_option(argv: list[str], flag: str, value: str) -> None:
    indices = [index for index, item in enumerate(argv) if item == flag]
    if len(indices) != 1 or indices[0] + 1 >= len(argv):
        raise CampaignError(f"expected one {flag} option")
    argv[indices[0] + 1] = value


def render_quant(root: Path, run: Run, gpu: int, attempt: Path):
    rendered = legacy_runner.render_baseline(
        legacy_plan(root),
        method=run.method,
        model=run.model.slug,
        setting_name=SETTING_NAMES[run.quant.slug],
        cuda_devices=str(gpu),
    )
    attempt_id = f"{rendered.run_id}_attempt{attempt.name.removeprefix('attempt')}"
    argv = list(rendered.argv)
    _set_option(argv, "--output_dir", str(attempt / "pipeline"))
    _set_option(argv, "--exp", attempt_id)
    rendered = dataclasses.replace(
        rendered,
        argv=argv,
        run_id=attempt_id,
        output_dir=attempt,
    )
    args = SimpleNamespace(
        method=run.method,
        phase="final",
        model=run.model.slug,
        setting=SETTING_NAMES[run.quant.slug],
    )
    rendered = timing_adapter.prepare_formal_rendered(args, rendered)
    spec = timing_adapter.spec_from_executor_args(args, rendered)
    if spec is None:
        raise CampaignError("quantization unexpectedly has no timing spec")
    rendered, provenance = timing_adapter.instrument_rendered(rendered, spec, repo_root=WORKSPACE)
    return rendered, provenance, spec


def render_producer(root: Path, producer: Producer, gpu: int, attempt: Path):
    attempt_cache = attempt / "cache"
    source_token, _ = _model_source_cache(producer.model)
    model_name = Path(base.model_path(producer.model)).name
    _ensure_symlink(
        attempt_cache / "tokens" / f"{model_name}-wikitext2_s256_blk2048_seed1.pt",
        source_token,
    )
    rendered = legacy_runner.render_guided_saliency(
        legacy_plan(root, cache_root=attempt_cache),
        model=producer.model.slug,
        cuda_devices=str(gpu),
    )
    attempt_id = f"{rendered.run_id}_attempt{attempt.name.removeprefix('attempt')}"
    argv = list(rendered.argv)
    _set_option(argv, "--output_dir", str(attempt / "pipeline"))
    _set_option(argv, "--exp", attempt_id)
    rendered = dataclasses.replace(rendered, argv=argv, run_id=attempt_id, output_dir=attempt)
    args = SimpleNamespace(
        method="guided_saliency",
        phase="precompute",
        target_phase=None,
        model=producer.model.slug,
        setting=None,
    )
    spec = timing_adapter.spec_from_executor_args(args, rendered)
    if spec is None:
        raise CampaignError("producer unexpectedly has no timing spec")
    rendered, provenance = timing_adapter.instrument_rendered(rendered, spec, repo_root=WORKSPACE)
    return rendered, provenance, spec, attempt_cache


def worker_env(gpu: int, *, reasoning: bool = False, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env = base.worker_env(gpu, reasoning=reasoning)
    env.update(
        REALQ_PYTHON=str(base.PYTHON),
        HF_HOME=str(WORKSPACE / "datasets" / "lm_eval_hf_cache"),
        HF_DATASETS_TRUST_REMOTE_CODE="1",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_DISABLE_TELEMETRY="1",
    )
    if extra:
        env.update({str(key): str(value) for key, value in extra.items()})
    return env


def run_logged_env(
    command: Sequence[str],
    log: Path,
    gpu: int,
    *,
    extra_env: Mapping[str, str] | None = None,
    reasoning: bool = False,
) -> tuple[int, float]:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open("ab", buffering=0) as handle:
        handle.write(
            (
                f"\n[{now()}] command={json.dumps(list(command), ensure_ascii=False)}\n"
                f"[{now()}] gpu={gpu} hostname={socket.gethostname()}\n"
            ).encode()
        )
        process = subprocess.Popen(
            list(command),
            cwd=WORKSPACE,
            env=worker_env(gpu, reasoning=reasoning, extra=extra_env),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        returncode = process.wait()
        elapsed = time.monotonic() - started
        handle.write(f"[{now()}] returncode={returncode} elapsed_seconds={elapsed:.6f}\n".encode())
    return returncode, elapsed


def next_attempt(directory: Path) -> tuple[int, Path]:
    indices = []
    for item in directory.glob("attempt[0-9][0-9][0-9]"):
        if item.is_dir():
            with contextlib.suppress(ValueError):
                indices.append(int(item.name.removeprefix("attempt")))
    index = max(indices, default=0) + 1
    return index, directory / f"attempt{index:03d}"


def stable_file(path: Path, seconds: int) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise CampaignError(f"missing or empty file: {path}")
    before = path.stat()
    time.sleep(seconds)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise CampaignError(f"file did not stabilize: {path}")
    return {"path": str(path), "size_bytes": after.st_size, "mtime_ns": after.st_mtime_ns}


def timing_evidence(attempt: Path) -> dict[str, Any]:
    path = attempt / "phase_timing_rank0.json"
    evidence = read_json(path)
    if (
        evidence.get("scope_id") != "algorithm_core_v1"
        or evidence.get("timing_status") != "complete"
        or evidence.get("complete") is not True
        or evidence.get("rank") != 0
    ):
        raise CampaignError(f"incomplete algorithm timing: {path}")
    elapsed = float(evidence.get("elapsed_seconds", -1))
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise CampaignError(f"invalid algorithm elapsed time: {path}")
    return {
        "path": str(path),
        "sha256": base.file_sha256(path),
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0,
        "spec_sha256": evidence.get("spec_sha256"),
        "source_set_sha256": evidence.get("source_set_sha256"),
    }


def publish_checkpoint(source: Path, target: Path) -> dict[str, Any]:
    stat = stable_file(source, CHECKPOINT_STABILITY_SECONDS)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.resolve(strict=True) != source.resolve(strict=True):
            raise CampaignError(f"canonical checkpoint already points elsewhere: {target}")
    else:
        target.symlink_to(os.path.relpath(source, target.parent))
    stat["canonical_path"] = str(target)
    stat["canonical_resolved_path"] = str(target.resolve(strict=True))
    return stat


def parse_quality(log: Path) -> dict[str, Any]:
    return quality.validate_metrics(log)


def publish_quality(root: Path, run: Run, log: Path, elapsed: float, source: str) -> None:
    metrics = parse_quality(log)
    atomic_json(
        marker(root, run, "quality"),
        {
            "campaign_id": CAMPAIGN_ID,
            "run_id": run.run_id,
            "status": "complete",
            "source": source,
            "log": str(log),
            "log_sha256": base.file_sha256(log),
            "elapsed_seconds": elapsed,
            "metrics": metrics,
            "completed_at": now(),
        },
    )


def recover_completed_quant(root: Path, run: Run) -> bool:
    success = marker(root, run, "quant")
    if success.is_file():
        return True
    attempts_root = run_dir(root, run) / "quant_attempts"
    attempts = sorted(
        (item for item in attempts_root.glob("attempt[0-9][0-9][0-9]") if item.is_dir()),
        reverse=True,
    )
    for attempt in attempts:
        source = attempt / "checkpoint" / "model.pt"
        try:
            timing = timing_evidence(attempt)
            checkpoint = publish_checkpoint(source, checkpoint_path(root, run))
        except (CampaignError, OSError, json.JSONDecodeError):
            continue
        log = attempt / "execution.log"
        payload = {
            "campaign_id": CAMPAIGN_ID,
            "run_id": run.run_id,
            "status": "complete",
            "method": run.method,
            "model": run.model.slug,
            "setting": run.quant.slug,
            "attempt": str(attempt),
            "checkpoint": checkpoint,
            "algorithm_timing": timing,
            "recovered_from_attempt": True,
            "completed_at": now(),
        }
        atomic_json(success, payload)
        if log.is_file() and not marker(root, run, "quality").is_file():
            with contextlib.suppress(Exception):
                publish_quality(root, run, log, 0.0, "quantization_process")
        return True
    return False


def run_quant(root: Path, run: Run, gpu: int) -> None:
    load_plan(root)
    if recover_completed_quant(root, run):
        return
    attempts_root = run_dir(root, run) / "quant_attempts"
    index, attempt = next_attempt(attempts_root)
    if index > MAX_ATTEMPTS:
        raise CampaignError(f"quant attempts exhausted: {run.run_id}")
    attempt.mkdir(parents=True)
    rendered, provenance, spec = render_quant(root, run, gpu, attempt)
    atomic_json(
        attempt / "config.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "run_id": run.run_id,
            "attempt_index": index,
            "gpu": gpu,
            "hostname": socket.gethostname(),
            "method": run.method,
            "model": dataclasses.asdict(run.model),
            "quant": dataclasses.asdict(run.quant),
            "setting_name": SETTING_NAMES[run.quant.slug],
            "argv": rendered.argv,
            "environment": rendered.env,
            "timing_spec": spec.payload(),
            "timing_spec_sha256": spec.sha256,
            "plan_fingerprint": read_json(root / "plan.json")["fingerprint"],
            "started_at": now(),
        },
    )
    rc, elapsed = run_logged_env(
        rendered.argv,
        attempt / "execution.log",
        gpu,
        extra_env={**rendered.env, **provenance.environment},
    )
    atomic_json(
        attempt / "process_result.json",
        {"returncode": rc, "elapsed_seconds": elapsed, "completed_at": now()},
    )
    # The algorithm boundary ends immediately before checkpoint I/O, so a QA
    # failure after a valid checkpoint must not trigger repeated quantization.
    if not recover_completed_quant(root, run):
        raise CampaignError(f"quantization did not produce a valid checkpoint/timing pair: {run.run_id}")
    if rc == 0 and not marker(root, run, "quality").is_file():
        publish_quality(root, run, attempt / "execution.log", elapsed, "quantization_process")


def _producer_saliency_dir(cache_root: Path, model: base.ModelSpec) -> Path:
    from tools.validate_guided_saliency import expected_saliency_dir

    return expected_saliency_dir(
        cache_root=cache_root,
        model=Path(base.model_path(model)),
        dataset="wikitext2",
        nsamples=256,
        seq_len=2048,
        seed=1,
        rotation_seed=0,
        num_groups=4,
    )


def producer_success_path(root: Path, producer: Producer) -> Path:
    return producer_dir(root, producer) / "producer_success.json"


def _producer_process_elapsed(attempt: Path) -> float:
    log = attempt / "execution.log"
    matches = re.findall(
        r"returncode=(\d+) elapsed_seconds=([0-9]+(?:\.[0-9]+)?)",
        log.read_text(encoding="utf-8", errors="replace"),
    )
    if not matches or int(matches[-1][0]) != 0:
        raise CampaignError(f"producer process did not complete successfully: {attempt}")
    elapsed = float(matches[-1][1])
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise CampaignError(f"invalid producer process elapsed time: {attempt}")
    return elapsed


def _finalize_producer_attempt(
    root: Path,
    producer: Producer,
    attempt: Path,
    elapsed: float,
    *,
    recovered: bool,
) -> None:
    timing = timing_evidence(attempt)
    saliency = _producer_saliency_dir(attempt / "cache", producer.model)
    validation_path = attempt / "validation.json"
    completed = subprocess.run(
        [
            str(base.PYTHON),
            "-m", "tools.validate_guided_saliency",
            "--model", base.model_path(producer.model),
            "--saliency-dir", str(saliency),
            "--nsamples", "256",
            "--seq-len", "2048",
            "--seed", "1",
            "--rotation-seed", "0",
            "--num-groups", "4",
            "--write-manifest", str(validation_path),
        ],
        cwd=WORKSPACE,
        env=worker_env(0),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode:
        raise CampaignError(f"Guided saliency validation failed: {completed.stderr[-4000:]}")
    validation = read_json(validation_path)
    if validation.get("valid") is not True:
        raise CampaignError(f"Guided saliency is invalid: {validation_path}")
    before = [
        (path.name, path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(saliency.glob("l*.pt"))
    ]
    time.sleep(SALIENCY_STABILITY_SECONDS)
    after = [
        (path.name, path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(saliency.glob("l*.pt"))
    ]
    if before != after:
        raise CampaignError(f"Guided saliency files did not stabilize: {saliency}")
    canonical = _producer_saliency_dir(root / "cache" / "legacy", producer.model)
    _ensure_symlink(canonical, saliency)
    atomic_json(
        producer_success_path(root, producer),
        {
            "campaign_id": CAMPAIGN_ID,
            "work_id": producer.work_id,
            "status": "complete",
            "model": producer.model.slug,
            "attempt": str(attempt),
            "canonical_saliency_dir": str(canonical),
            "resolved_saliency_dir": str(canonical.resolve(strict=True)),
            "validation": str(validation_path),
            "validation_sha256": base.file_sha256(validation_path),
            "algorithm_timing": timing,
            "process_elapsed_seconds": elapsed,
            "recovered_from_completed_numerical_attempt": recovered,
            "completed_at": now(),
        },
    )


def recover_completed_producer(root: Path, producer: Producer) -> bool:
    if producer_success_path(root, producer).is_file():
        return True
    attempts_root = producer_dir(root, producer) / "attempts"
    attempts = sorted(
        (item for item in attempts_root.glob("attempt[0-9][0-9][0-9]") if item.is_dir()),
        reverse=True,
    )
    for attempt in attempts:
        try:
            elapsed = _producer_process_elapsed(attempt)
            _finalize_producer_attempt(
                root, producer, attempt, elapsed, recovered=True
            )
        except (CampaignError, OSError, json.JSONDecodeError):
            continue
        return True
    return False


def run_producer(root: Path, producer: Producer, gpu: int) -> None:
    load_plan(root)
    if recover_completed_producer(root, producer):
        return
    attempts_root = producer_dir(root, producer) / "attempts"
    index, attempt = next_attempt(attempts_root)
    if index > MAX_ATTEMPTS:
        raise CampaignError(f"producer attempts exhausted: {producer.work_id}")
    attempt.mkdir(parents=True)
    rendered, provenance, spec, attempt_cache = render_producer(root, producer, gpu, attempt)
    atomic_json(
        attempt / "config.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "work_id": producer.work_id,
            "attempt_index": index,
            "gpu": gpu,
            "hostname": socket.gethostname(),
            "model": dataclasses.asdict(producer.model),
            "argv": rendered.argv,
            "timing_spec": spec.payload(),
            "timing_spec_sha256": spec.sha256,
            "started_at": now(),
        },
    )
    rc, elapsed = run_logged_env(
        rendered.argv,
        attempt / "execution.log",
        gpu,
        extra_env={**rendered.env, **provenance.environment},
    )
    if rc:
        raise CampaignError(f"Guided saliency producer exited {rc}: {attempt}")
    _finalize_producer_attempt(
        root, producer, attempt, elapsed, recovered=False
    )


def quality_args(root: Path, run: Run, output: Path, batch_size: int) -> list[str]:
    argv = base.common_args(BASE_ROOT, run.model, run.quant, global_loss_bsz=32)
    argv.extend(
        [
            "--load_qmodel_path", str(checkpoint_path(root, run)),
            "--skip_eval", "false",
            "--skip_kl_ppl_eval", "false",
            "--lm_eval", "true",
            "--lm_eval_batch_size", str(batch_size),
            "--reasoning_eval", "false",
            "--require_static_cache_hit", "false",
            "--require_reference_cache_hit", "true",
            "--output_dir", str(output / "pipeline"),
            "--exp", "wikitext2_paperqa",
        ]
    )
    base.validate_args(argv)
    return argv


def run_quality(root: Path, run: Run, gpu: int) -> None:
    load_plan(root)
    success = marker(root, run, "quality")
    if success.is_file():
        return
    if not marker(root, run, "quant").is_file():
        raise CampaignError(f"quant marker missing: {run.run_id}")
    attempts_root = run_dir(root, run) / "quality_attempts"
    index, attempt = next_attempt(attempts_root)
    if index > len((32, 16, 8, 4, 2, 1)):
        raise CampaignError(f"quality batch ladder exhausted: {run.run_id}")
    batch_size = (32, 16, 8, 4, 2, 1)[index - 1]
    attempt.mkdir(parents=True)
    argv = quality_args(root, run, attempt, batch_size)
    atomic_json(
        attempt / "config.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "run_id": run.run_id,
            "attempt_index": index,
            "lm_eval_batch_size": batch_size,
            "argv": argv,
            "started_at": now(),
        },
    )
    rc, elapsed = base.run_logged(
        [str(base.PYTHON), "-m", "realq.ptq", *argv],
        attempt / "execution.log",
        gpu,
    )
    if rc:
        if base.log_is_oom(attempt / "execution.log"):
            raise CampaignError(f"quality OOM at batch={batch_size}: {run.run_id}")
        raise CampaignError(f"quality failed rc={rc}: {run.run_id}")
    publish_quality(root, run, attempt / "execution.log", elapsed, "checkpoint_reload")


def reasoning_dir(root: Path, run: Run, task: str) -> Path:
    return run_dir(root, run) / "reasoning" / task


def reasoning_success(root: Path, run: Run, task: str) -> Path:
    return reasoning_dir(root, run, task) / "generation_success.json"


def reasoning_args(root: Path, run: Run, task: str, output: Path) -> list[str]:
    argv = base.eval_args(BASE_ROOT, run.base_run, task, output)
    index = argv.index("--load_qmodel_path")
    argv[index + 1] = str(checkpoint_path(root, run))
    base.validate_args(argv)
    return argv


def run_reasoning(root: Path, run: Run, task: str, gpu: int) -> None:
    load_plan(root)
    success = reasoning_success(root, run, task)
    if success.is_file():
        return
    if not marker(root, run, "quant").is_file():
        raise CampaignError(f"quant marker missing: {run.run_id}")
    output = reasoning_dir(root, run, task)
    output.mkdir(parents=True, exist_ok=True)
    attempts_root = output / "attempts"
    index, attempt = next_attempt(attempts_root)
    if index > MAX_ATTEMPTS:
        raise CampaignError(f"reasoning attempts exhausted: {run.run_id}/{task}")
    attempt.mkdir(parents=True)
    argv = reasoning_args(root, run, task, output)
    atomic_json(
        attempt / "config.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "run_id": run.run_id,
            "task": task,
            "attempt_index": index,
            "argv": argv,
            "started_at": now(),
        },
    )
    rc, elapsed = base.run_logged(
        [str(base.PYTHON), "-m", "realq.ptq", *argv],
        attempt / "execution.log",
        gpu,
        reasoning=True,
    )
    if rc:
        raise CampaignError(f"reasoning failed rc={rc}: {run.run_id}/{task}")
    base.audit_eval_output(output, task)
    atomic_json(
        success,
        {
            "campaign_id": CAMPAIGN_ID,
            "run_id": run.run_id,
            "task": task,
            "status": "generation_complete",
            "attempt": str(attempt),
            "elapsed_seconds": elapsed,
            "completed_at": now(),
            "official_code_score_pending": task == "humaneval_plus",
        },
    )


def official_success(root: Path, run: Run) -> Path:
    return reasoning_dir(root, run, "humaneval_plus") / "official_eval" / "official_success.json"


def run_score(root: Path, run: Run) -> None:
    load_plan(root)
    generation = reasoning_success(root, run, "humaneval_plus")
    if not generation.is_file():
        raise CampaignError(f"HumanEval+ generation missing: {run.run_id}")
    success = official_success(root, run)
    if success.is_file():
        return
    output = success.parent
    output.mkdir(parents=True, exist_ok=True)
    samples = reasoning_dir(root, run, "humaneval_plus") / "humaneval_plus" / "evalplus_samples.jsonl"
    dataset = WORKSPACE / base.DATASET_PATHS["humaneval_plus"]
    sample_rows = [json.loads(line) for line in samples.read_text(encoding="utf-8").splitlines() if line.strip()]
    with gzip.open(dataset, "rt", encoding="utf-8") as handle:
        expected_ids = {json.loads(line)["task_id"] for line in handle if line.strip()}
    sample_ids = [row.get("task_id") for row in sample_rows]
    if (
        len(sample_rows) != base.TASKS["humaneval_plus"][2]
        or len(set(sample_ids)) != len(sample_ids)
        or set(sample_ids) != expected_ids
        or any(not isinstance(row.get("solution"), str) for row in sample_rows)
    ):
        raise CampaignError(f"HumanEval+ sample gate failed: {samples}")
    stable_file(samples, 5)
    started = time.monotonic()
    completed = subprocess.run(
        [
            "bash",
            str(WORKSPACE / "tools" / "lowbit_activation_evalplus_canoe.sh"),
            str(samples),
            str(output),
            str(EVALPLUS_PARALLEL),
        ],
        cwd=WORKSPACE,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "REALQ_PYTHON": str(base.PYTHON)},
        stdin=subprocess.DEVNULL,
        check=False,
    )
    elapsed = time.monotonic() - started
    if completed.returncode:
        raise CampaignError(f"official EvalPlus failed rc={completed.returncode}: {run.run_id}")
    result = output / "evalplus_samples_eval_results.json"
    payload = read_json(result)
    evaluations = payload.get("eval")
    if not isinstance(evaluations, dict) or set(evaluations) != expected_ids:
        raise CampaignError(f"official EvalPlus coverage mismatch: {result}")
    base_pass = plus_pass = 0
    for task_id, candidates in evaluations.items():
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise CampaignError(f"EvalPlus candidate count mismatch: {task_id}")
        candidate = candidates[0]
        if candidate.get("base_status") not in EVALPLUS_STATUSES or candidate.get("plus_status") not in EVALPLUS_STATUSES:
            raise CampaignError(f"invalid EvalPlus status: {task_id}")
        base_ok = candidate["base_status"] == "pass"
        base_pass += int(base_ok)
        plus_pass += int(base_ok and candidate["plus_status"] == "pass")
    total = base.TASKS["humaneval_plus"][2]
    atomic_json(
        success,
        {
            "campaign_id": CAMPAIGN_ID,
            "run_id": run.run_id,
            "status": "official_scored",
            "samples": str(samples),
            "samples_sha256": base.file_sha256(samples),
            "official_result": str(result),
            "official_result_sha256": base.file_sha256(result),
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


def failure_path(root: Path, work_id: str, stage: str) -> Path:
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", work_id)
    return root / "failures" / stage / f"{safe}.json"


def record_worker_failure(root: Path, work_id: str, stage: str, error: BaseException) -> None:
    path = failure_path(root, work_id, stage)
    previous = read_json(path) if path.is_file() else {"failures": []}
    failures = previous.get("failures")
    if not isinstance(failures, list):
        failures = []
    failures.append(
        {
            "at": now(),
            "hostname": socket.gethostname(),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    )
    atomic_json(path, {"campaign_id": CAMPAIGN_ID, "work_id": work_id, "stage": stage, "failures": failures})


def failure_ready(root: Path, work_id: str, stage: str) -> bool:
    path = failure_path(root, work_id, stage)
    if not path.is_file():
        return True
    payload = read_json(path)
    failures = payload.get("failures", [])
    if not isinstance(failures, list) or len(failures) >= MAX_ATTEMPTS:
        return False
    return time.time() - path.stat().st_mtime >= FAILURE_COOLDOWN_SECONDS


def work_priority(kind: str, run: Run | None = None) -> tuple[int, int, int, str]:
    if kind == "producer":
        return (0, 0, 0, "")
    if run is None:
        return (99, 0, 0, "")
    model_index = run.base_run.model_index
    size_rank = -model_index
    if kind == "quant":
        method_rank = 0 if run.method == "guided_gptq" else 1
        return (1 + method_rank, size_rank, run.base_run.quant_index, run.run_id)
    if kind == "quality":
        return (3, size_rank, run.base_run.quant_index, run.run_id)
    return (4, size_rank, run.base_run.quant_index, run.run_id)


def ready_gpu_works(root: Path, node: int) -> list[tuple[str, str, str | None]]:
    works: list[tuple[tuple[int, int, int, str], tuple[str, str, str | None]]] = []
    for producer in PRODUCERS:
        if producer.node != node or producer_success_path(root, producer).is_file():
            continue
        if failure_ready(root, producer.work_id, "producer"):
            works.append((work_priority("producer"), ("producer", producer.work_id, None)))
    for run in RUNS:
        if run.node != node:
            continue
        quant_done = marker(root, run, "quant").is_file()
        if not quant_done:
            dependency = run.method != "guided_gptq" or producer_success_path(root, PRODUCER_BY_MODEL[run.model.slug]).is_file()
            if dependency and failure_ready(root, run.run_id, "quant"):
                works.append((work_priority("quant", run), ("quant", run.run_id, None)))
            continue
        if not marker(root, run, "quality").is_file() and failure_ready(root, run.run_id, "quality"):
            works.append((work_priority("quality", run), ("quality", run.run_id, None)))
        for task in base.TASKS:
            work_id = f"{run.run_id}/{task}"
            if not reasoning_success(root, run, task).is_file() and failure_ready(root, work_id, "reasoning"):
                works.append((work_priority("reasoning", run), ("reasoning", run.run_id, task)))
    return [work for _, work in sorted(works, key=lambda item: item[0])]


def ready_scores(root: Path, node: int) -> list[Run]:
    return [
        run
        for run in RUNS
        if run.node == node
        and reasoning_success(root, run, "humaneval_plus").is_file()
        and not official_success(root, run).is_file()
        and failure_ready(root, run.run_id, "score")
    ]


def internal_command(root: Path, gpu: int, work: tuple[str, str, str | None]) -> list[str]:
    kind, work_id, task = work
    command = [str(base.PYTHON), "-m", "experiments.gptaq_guided_20group_20260809.campaign", f"_{kind}", "--root", str(root), "--gpu", str(gpu)]
    if kind == "producer":
        command.extend(["--producer-id", work_id])
    else:
        command.extend(["--run-id", work_id])
    if task:
        command.extend(["--task", task])
    return command


def gpu_snapshot() -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,uuid", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=True,
    )
    rows = []
    for line in completed.stdout.splitlines():
        index, name, memory, uuid = [part.strip() for part in line.split(",", 3)]
        rows.append({"index": int(index), "name": name, "memory_total_mib": int(memory), "uuid": uuid})
    if [row["index"] for row in rows] != list(GPU_IDS):
        raise CampaignError(f"expected GPUs 0..7: {rows}")
    return rows


def campaign_complete(root: Path) -> bool:
    return (
        sum(producer_success_path(root, producer).is_file() for producer in PRODUCERS) == len(PRODUCERS)
        and sum(marker(root, run, "quant").is_file() for run in RUNS) == len(RUNS)
        and sum(marker(root, run, "quality").is_file() for run in RUNS) == len(RUNS)
        and sum(reasoning_success(root, run, task).is_file() for run in RUNS for task in base.TASKS) == len(RUNS) * len(base.TASKS)
        and sum(official_success(root, run).is_file() for run in RUNS) == len(RUNS)
    )


def run_node(root: Path, node: int) -> None:
    load_plan(root)
    if node not in (0, 1):
        raise CampaignError(f"node must be 0 or 1: {node}")
    snapshot = gpu_snapshot()
    node_root = root / "nodes" / f"node{node}"
    node_root.mkdir(parents=True, exist_ok=True)
    lock_path = node_root / ".controller.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignError(f"node controller already active: {node}") from exc
        active: dict[int, tuple[tuple[str, str, str | None], subprocess.Popen[Any]]] = {}
        scores: dict[str, subprocess.Popen[Any]] = {}
        last_heartbeat = 0.0
        while True:
            for gpu, (work, process) in list(active.items()):
                rc = process.poll()
                if rc is None:
                    continue
                active.pop(gpu)
                if rc:
                    kind, work_id, _ = work
                    # The child already wrote the detailed failure marker.
                    print(f"worker failed rc={rc}: {kind}/{work_id}", flush=True)
            for run_id, process in list(scores.items()):
                rc = process.poll()
                if rc is not None:
                    scores.pop(run_id)
                    if rc:
                        print(f"score worker failed rc={rc}: {run_id}", flush=True)

            active_ids = {(work[0], work[1], work[2]) for work, _ in active.values()}
            ready = [work for work in ready_gpu_works(root, node) if work not in active_ids]
            for gpu in GPU_IDS:
                if gpu in active or not ready:
                    continue
                work = ready.pop(0)
                process = subprocess.Popen(
                    internal_command(root, gpu, work),
                    cwd=WORKSPACE,
                    env=worker_env(gpu),
                    stdin=subprocess.DEVNULL,
                    stdout=(node_root / "controller_children.log").open("ab", buffering=0),
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active[gpu] = (work, process)

            for run in ready_scores(root, node):
                if len(scores) >= 2 or run.run_id in scores:
                    break
                process = subprocess.Popen(
                    [str(base.PYTHON), "-m", "experiments.gptaq_guided_20group_20260809.campaign", "_score", "--root", str(root), "--run-id", run.run_id],
                    cwd=WORKSPACE,
                    env={**worker_env(0), "CUDA_VISIBLE_DEVICES": ""},
                    stdin=subprocess.DEVNULL,
                    stdout=(node_root / "score_children.log").open("ab", buffering=0),
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                scores[run.run_id] = process

            if time.monotonic() - last_heartbeat >= HEARTBEAT_SECONDS:
                atomic_json(
                    node_root / "heartbeat.json",
                    {
                        "campaign_id": CAMPAIGN_ID,
                        "node": node,
                        "hostname": socket.gethostname(),
                        "pid": os.getpid(),
                        "gpu_snapshot": snapshot,
                        "active_gpu": {
                            str(gpu): {"kind": work[0], "work_id": work[1], "task": work[2], "pid": process.pid}
                            for gpu, (work, process) in active.items()
                        },
                        "active_scores": {run_id: process.pid for run_id, process in scores.items()},
                        "updated_at": now(),
                    },
                )
                last_heartbeat = time.monotonic()

            if campaign_complete(root) and not active and not scores:
                atomic_json(node_root / "complete.json", {"campaign_id": CAMPAIGN_ID, "node": node, "completed_at": now()})
                return
            time.sleep(CONTROLLER_POLL_SECONDS)


def status_payload(root: Path) -> dict[str, Any]:
    nodes = {}
    for node in (0, 1):
        heartbeat = root / "nodes" / f"node{node}" / "heartbeat.json"
        nodes[str(node)] = read_json(heartbeat) if heartbeat.is_file() else None
    failures = []
    for path in sorted((root / "failures").glob("*/*.json")) if (root / "failures").is_dir() else []:
        payload = read_json(path)
        failures.append({"path": str(path), "stage": payload.get("stage"), "work_id": payload.get("work_id"), "count": len(payload.get("failures", []))})
    return {
        "campaign_id": CAMPAIGN_ID,
        "root": str(root),
        "counts": {
            "producer_success": sum(producer_success_path(root, producer).is_file() for producer in PRODUCERS),
            "producer_total": len(PRODUCERS),
            "quant_success": sum(marker(root, run, "quant").is_file() for run in RUNS),
            "quant_total": len(RUNS),
            "quality_success": sum(marker(root, run, "quality").is_file() for run in RUNS),
            "quality_total": len(RUNS),
            "reasoning_generation_success": sum(reasoning_success(root, run, task).is_file() for run in RUNS for task in base.TASKS),
            "reasoning_generation_total": len(RUNS) * len(base.TASKS),
            "official_evalplus_success": sum(official_success(root, run).is_file() for run in RUNS),
            "official_evalplus_total": len(RUNS),
        },
        "nodes": nodes,
        "failures": failures,
        "complete": campaign_complete(root),
        "reported_at": now(),
    }


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__)
    sub = top.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--root", default=str(DEFAULT_ROOT))
    status = sub.add_parser("status")
    status.add_argument("--root", default=str(DEFAULT_ROOT))
    node = sub.add_parser("run-node")
    node.add_argument("--root", default=str(DEFAULT_ROOT))
    node.add_argument("--node", type=int, required=True)
    for name in ("_producer", "_quant", "_quality", "_reasoning"):
        child = sub.add_parser(name)
        child.add_argument("--root", required=True)
        child.add_argument("--gpu", type=int, required=True)
        child.add_argument("--producer-id")
        child.add_argument("--run-id")
        child.add_argument("--task", choices=tuple(base.TASKS))
    score = sub.add_parser("_score")
    score.add_argument("--root", required=True)
    score.add_argument("--run-id", required=True)
    return top


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = safe_root(args.root)
    try:
        if args.command == "plan":
            print(json.dumps(write_plan(root), indent=2, ensure_ascii=False))
        elif args.command == "status":
            load_plan(root)
            print(json.dumps(status_payload(root), indent=2, ensure_ascii=False))
        elif args.command == "run-node":
            run_node(root, args.node)
        elif args.command == "_producer":
            producer = next((item for item in PRODUCERS if item.work_id == args.producer_id), None)
            if producer is None:
                raise CampaignError(f"unknown producer: {args.producer_id}")
            try:
                run_producer(root, producer, args.gpu)
            except BaseException as exc:
                record_worker_failure(root, producer.work_id, "producer", exc)
                raise
        elif args.command in {"_quant", "_quality", "_reasoning", "_score"}:
            run = RUN_BY_ID.get(args.run_id)
            if run is None:
                raise CampaignError(f"unknown run: {args.run_id}")
            if args.command == "_score":
                try:
                    run_score(root, run)
                except BaseException as exc:
                    record_worker_failure(root, run.run_id, "score", exc)
                    raise
            elif args.command == "_quant":
                try:
                    run_quant(root, run, args.gpu)
                except BaseException as exc:
                    record_worker_failure(root, run.run_id, "quant", exc)
                    raise
            elif args.command == "_quality":
                try:
                    run_quality(root, run, args.gpu)
                except BaseException as exc:
                    record_worker_failure(root, run.run_id, "quality", exc)
                    raise
            else:
                if args.task is None:
                    raise CampaignError("_reasoning requires --task")
                work_id = f"{run.run_id}/{args.task}"
                try:
                    run_reasoning(root, run, args.task, args.gpu)
                except BaseException as exc:
                    record_worker_failure(root, work_id, "reasoning", exc)
                    raise
        return 0
    except (CampaignError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
