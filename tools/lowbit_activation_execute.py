#!/usr/bin/env python3
"""Execute a reviewed low-bit command with fail-closed provenance capture.

The default mode is a read-only dry run.  Passing ``--execute`` is the only
way this wrapper creates an output directory or launches the rendered command.
It deliberately invokes argv without a shell and merges child stdout/stderr
into an execution log in the command's final output directory.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import getpass
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import lowbit_activation_runner as runner  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = runner.DEFAULT_PLAN
RenderedCommand = runner.RenderedCommand

MANIFEST_FILENAME = "execution_manifest.json"
LOG_FILENAME = "execution.log"
LOCK_FILENAME = ".execution.lock"
MODEL_SAMPLE_BYTES = 1024 * 1024
MAX_ATTEMPT_INDEX = 20
GIB = 1024**3

PACKAGE_DISTRIBUTIONS = {
    "accelerate": "accelerate",
    "datasets": "datasets",
    "huggingface_hub": "huggingface-hub",
    "lm_eval": "lm-eval",
    "numpy": "numpy",
    "safetensors": "safetensors",
    "scipy": "scipy",
    "tokenizers": "tokenizers",
    "torch": "torch",
    "transformers": "transformers",
}

_SENSITIVE_ENV_FRAGMENTS = (
    "ACCESS_KEY",
    "API_KEY",
    "AUTH",
    "CREDENTIAL",
    "PASSWORD",
    "PASSWD",
    "PRIVATE_KEY",
    "PROXY",
    "SECRET",
)


class ExecutionError(RuntimeError):
    """The execution request is unsafe or its provenance is incomplete."""


class ConcurrentRunError(ExecutionError):
    """Another wrapper process currently owns this run's output directory."""


class ExistingRunError(ExecutionError):
    """The output directory already contains artifacts from another attempt."""


def _require_python_environment(plan: Mapping[str, Any]) -> None:
    """Refuse writes unless the plan's explicitly activated venv is in use."""

    configured = plan.get("python_environment")
    if configured is None:
        return
    if not isinstance(configured, Mapping):
        raise ExecutionError("plan python_environment must be an object")
    expected_value = configured.get("venv")
    if not isinstance(expected_value, str) or not expected_value:
        raise ExecutionError(
            "plan python_environment.venv must be a non-empty path"
        )
    expected = Path(expected_value).resolve(strict=False)
    virtual_env_value = os.environ.get("VIRTUAL_ENV")
    virtual_env = (
        Path(virtual_env_value).resolve(strict=False)
        if virtual_env_value
        else None
    )
    prefix = Path(sys.prefix).resolve(strict=False)
    if configured.get("activation_required") is not True:
        raise ExecutionError(
            "plan python_environment.activation_required must be true"
        )
    if virtual_env != expected or prefix != expected:
        raise ExecutionError(
            "required experiment venv is not active; source "
            f"{expected_value}/bin/activate before --execute "
            f"(VIRTUAL_ENV={virtual_env_value!r}, sys.prefix={sys.prefix!r})"
        )


def _expected_venv_python(plan: Mapping[str, Any]) -> Path:
    configured = plan.get("python_environment")
    if not isinstance(configured, Mapping):
        raise ExecutionError("plan python_environment must be configured")
    value = configured.get("venv")
    if not isinstance(value, str) or not value:
        raise ExecutionError("plan python_environment.venv must be a non-empty path")
    venv = Path(value)
    if not venv.is_absolute():
        raise ExecutionError("plan python_environment.venv must be absolute")
    return venv / "bin" / "python"


def _require_rendered_executable(
    rendered: RenderedCommand,
    plan: Mapping[str, Any],
) -> Path:
    """Bind every child to the exact reviewed venv interpreter.

    Do not resolve the final symlink: ordinary venv interpreters are symlinks
    to the base Python, while invoking the path inside ``.venv/bin`` is what
    gives the child the intended ``sys.prefix`` and site-packages.
    """

    if not rendered.argv:
        raise ExecutionError("rendered command has no executable")
    expected = Path(os.path.abspath(_expected_venv_python(plan)))
    actual_raw = rendered.argv[0]
    actual = Path(actual_raw)
    if not actual.is_absolute():
        raise ExecutionError(
            "rendered executable must be the absolute experiment venv Python, "
            f"got {actual_raw!r}"
        )
    actual = Path(os.path.abspath(actual))
    if actual != expected:
        raise ExecutionError(
            "rendered executable is outside the exact experiment venv: "
            f"expected {expected}, got {actual}"
        )
    if not actual.is_file() or not os.access(actual, os.X_OK):
        raise ExecutionError(
            f"experiment venv Python is missing or not executable: {actual}"
        )
    return actual


def _current_distribution_versions(
    plan: Mapping[str, Any],
) -> dict[str, str]:
    expected = plan.get("runtime_versions")
    if not isinstance(expected, Mapping):
        raise ExecutionError("plan runtime_versions must be configured")
    actual: dict[str, str] = {}
    for distribution in ("torch", "transformers", "lm-eval"):
        expected_value = expected.get(distribution)
        if not isinstance(expected_value, str) or not expected_value:
            raise ExecutionError(
                f"plan runtime_versions.{distribution} must be configured"
            )
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ExecutionError(
                f"required distribution is missing: {distribution}"
            ) from exc
        actual[distribution] = version
        if version != expected_value:
            raise ExecutionError(
                f"{distribution} version changed after preflight: "
                f"expected {expected_value}, got {version}"
            )
    return actual


def _current_model_signature(
    rendered: RenderedCommand,
    *,
    plan: Mapping[str, Any],
    cwd: Path,
) -> dict[str, Any]:
    model_argument = _extract_option(rendered.argv, "--model")
    if model_argument is None:
        raise ExecutionError("rendered argv has no --model value")
    model_path = Path(model_argument)
    if not model_path.is_absolute():
        model_path = cwd / model_path
    model_path = model_path.resolve(strict=False)

    matched_name: str | None = None
    models = plan.get("models")
    if not isinstance(models, Mapping):
        raise ExecutionError("plan models must be configured")
    for name, value in models.items():
        candidate = Path(str(value))
        if not candidate.is_absolute():
            candidate = cwd / candidate
        if candidate.resolve(strict=False) == model_path:
            matched_name = str(name)
            break
    if matched_name is None:
        raise ExecutionError(
            f"rendered model path is not one of the exact planned artifacts: "
            f"{model_path}"
        )

    config_path = model_path / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExecutionError(
            f"cannot validate current model config {config_path}: {exc}"
        ) from exc
    actual = {
        key: config.get(key)
        for key in (
            "model_type",
            "hidden_size",
            "num_hidden_layers",
            "architectures",
        )
    }
    signatures = plan.get("model_signatures")
    expected = (
        signatures.get(matched_name)
        if isinstance(signatures, Mapping)
        else None
    )
    if actual != expected:
        raise ExecutionError(
            f"model config signature changed for {matched_name}: "
            f"expected {expected!r}, got {actual!r}"
        )
    return {
        "model": matched_name,
        "path": str(model_path),
        "signature": actual,
        "config_sha256": _sha256_file(config_path),
    }


def _current_gpu_gate(
    rendered: RenderedCommand,
    *,
    plan: Mapping[str, Any],
    cwd: Path,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ExecutionError(
            f"current CUDA gate probe failed: {type(exc).__name__}: {exc}"
        ) from exc
    if completed.returncode != 0:
        raise ExecutionError(
            "current CUDA gate probe failed: "
            f"returncode={completed.returncode}, stderr={completed.stderr[-2000:]!r}"
        )
    devices: list[dict[str, int]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise ExecutionError(
                f"current CUDA gate returned an invalid row: {line!r}"
            )
        try:
            index, total_mib, free_mib = (int(field) for field in fields)
        except ValueError as exc:
            raise ExecutionError(
                f"current CUDA gate returned a non-integer row: {line!r}"
            ) from exc
        devices.append(
            {
                "index": index,
                "total_bytes": total_mib * 1024**2,
                "free_bytes": free_mib * 1024**2,
            }
        )

    requirements = plan.get("runtime_requirements")
    if not isinstance(requirements, Mapping):
        raise ExecutionError("plan runtime_requirements must be configured")
    required_count = requirements.get("minimum_gpu_count")
    required_memory_gib = requirements.get("minimum_gpu_memory_gib")
    if (
        type(required_count) is not int
        or required_count <= 0
        or isinstance(required_memory_gib, bool)
        or not isinstance(required_memory_gib, (int, float))
        or not math.isfinite(float(required_memory_gib))
        or required_memory_gib <= 0
    ):
        raise ExecutionError("plan runtime GPU requirements are invalid")
    minimum_bytes = math.ceil(float(required_memory_gib) * GIB)
    if len(devices) < required_count:
        raise ExecutionError(
            f"at least {required_count} current CUDA GPUs are required, "
            f"detected {len(devices)}"
        )
    eligible = [
        device
        for device in devices
        if isinstance(device, Mapping)
        and type(device.get("total_bytes")) is int
        and device["total_bytes"] >= minimum_bytes
    ]
    if len(eligible) < required_count:
        raise ExecutionError(
            f"at least {required_count} GPUs with >= {required_memory_gib} GiB "
            f"are required at launch, detected {len(eligible)}"
        )

    requested_raw = rendered.env.get("CUDA_VISIBLE_DEVICES")
    if not isinstance(requested_raw, str):
        raise ExecutionError("rendered command is missing CUDA_VISIBLE_DEVICES")
    try:
        requested = runner._cuda_indices(requested_raw)
    except runner.PlanError as exc:
        raise ExecutionError(str(exc)) from exc
    eligible_indices = {
        int(device["index"])
        for device in eligible
        if type(device.get("index")) is int
    }
    invalid = [index for index in requested if index not in eligible_indices]
    if invalid:
        raise ExecutionError(
            f"requested CUDA indices are not currently eligible: {invalid!r}; "
            f"eligible={sorted(eligible_indices)!r}"
        )
    return {
        "torch_runtime": plan.get("runtime_versions", {}).get("torch_runtime"),
        "detected_gpu_count": len(devices),
        "eligible_gpu_indices": sorted(eligible_indices),
        "requested_gpu_indices": requested,
        "devices": devices,
    }


def _require_current_runtime(
    rendered: RenderedCommand,
    *,
    plan: Mapping[str, Any],
    cwd: Path,
) -> dict[str, Any]:
    python = _require_rendered_executable(rendered, plan)
    return {
        "python": str(python),
        "distributions": _current_distribution_versions(plan),
        "model": _current_model_signature(rendered, plan=plan, cwd=cwd),
        "gpu": _current_gpu_gate(
            rendered,
            plan=plan,
            cwd=cwd,
        ),
    }


def _runtime_preflight_path(
    plan: Mapping[str, Any],
    *,
    cwd: Path,
) -> Path | None:
    resolutions = plan.get("resolutions")
    if not isinstance(resolutions, Mapping):
        return None
    value = resolutions.get("runtime_preflight_manifest")
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ExecutionError(
            "plan resolutions.runtime_preflight_manifest must be a path"
        )
    path = Path(value)
    if not path.is_absolute():
        path = cwd / path
    return path.resolve(strict=False)


def _runtime_preflight_identity(
    plan: Mapping[str, Any],
    *,
    cwd: Path,
) -> dict[str, Any]:
    path = _runtime_preflight_path(plan, cwd=cwd)
    if path is None:
        return {"configured": False, "exists": False}
    if not path.is_file():
        return {
            "configured": True,
            "exists": False,
            "path": str(path),
        }
    result = _full_file_identity(path)
    result.update({"configured": True, "exists": True})
    return result


def _require_runtime_preflight(
    plan: Mapping[str, Any],
    *,
    plan_path: Path,
    cwd: Path,
) -> dict[str, Any]:
    """Validate the exact successful preflight before any experiment write."""

    path = _runtime_preflight_path(plan, cwd=cwd)
    if path is None:
        raise ExecutionError(
            "runtime preflight manifest is not configured; execution is "
            "fail-closed until the exact campaign preflight is recorded"
        )
    if not path.is_file():
        raise ExecutionError(
            f"runtime preflight manifest is missing: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExecutionError(
            f"runtime preflight manifest is unreadable: {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ExecutionError("runtime preflight manifest must contain an object")
    if payload.get("schema_version") != 1 or payload.get("valid") is not True:
        raise ExecutionError("runtime preflight manifest is not valid")
    if payload.get("load_task_datasets") is not True:
        raise ExecutionError(
            "runtime preflight must actually load all task datasets"
        )

    preflight_plan = payload.get("plan")
    if not isinstance(preflight_plan, Mapping):
        raise ExecutionError("runtime preflight is missing plan identity")
    current_plan_hash = _sha256_file(plan_path.resolve(strict=True))
    if preflight_plan.get("sha256") != current_plan_hash:
        raise ExecutionError(
            "runtime preflight was generated for a different plan snapshot"
        )

    resolutions = plan.get("resolutions")
    expected_job_id = (
        resolutions.get("canoe_job_id")
        if isinstance(resolutions, Mapping)
        else None
    )
    canoe = payload.get("canoe")
    if (
        not isinstance(canoe, Mapping)
        or canoe.get("valid") is not True
        or canoe.get("expected_job_id") != expected_job_id
        or canoe.get("supplied_job_id") != expected_job_id
        or canoe.get("hostname") != socket.gethostname()
    ):
        raise ExecutionError(
            "runtime preflight does not prove execution is on the planned "
            f"Canoe job {expected_job_id!r}"
        )

    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ExecutionError("runtime preflight is missing runtime checks")
    python_environment = runtime.get("python_environment")
    expected_venv = plan.get("python_environment", {}).get("venv")
    if (
        not isinstance(python_environment, Mapping)
        or python_environment.get("valid") is not True
        or python_environment.get("expected_venv") != expected_venv
    ):
        raise ExecutionError(
            "runtime preflight did not validate the planned Python venv"
        )
    distributions = runtime.get("distributions")
    expected_versions = plan.get("runtime_versions")
    if not isinstance(expected_versions, Mapping):
        raise ExecutionError("plan runtime_versions must be configured")
    if not isinstance(distributions, Mapping) or any(
        distributions.get(name) != expected_versions.get(name)
        for name in ("torch", "transformers", "lm-eval")
    ):
        raise ExecutionError(
            "runtime preflight did not validate the exact planned torch, "
            "transformers, and lm-eval versions"
        )
    cuda = runtime.get("cuda")
    requirement_check = (
        cuda.get("requirement_check")
        if isinstance(cuda, Mapping)
        else None
    )
    if (
        not isinstance(requirement_check, Mapping)
        or requirement_check.get("passed") is not True
        or not isinstance(cuda, Mapping)
        or cuda.get("torch_version") != expected_versions.get("torch_runtime")
    ):
        raise ExecutionError(
            "runtime preflight did not pass the exact torch-build and GPU "
            "count/memory gate"
        )

    expected_tasks = tuple(plan.get("paper_zero_shot_tasks", ()))
    lm_eval = payload.get("lm_eval")
    resolved = lm_eval.get("resolved") if isinstance(lm_eval, Mapping) else None
    loaded = (
        lm_eval.get("datasets_loaded")
        if isinstance(lm_eval, Mapping)
        else None
    )
    if (
        not isinstance(resolved, Mapping)
        or set(resolved) != set(expected_tasks)
        or any(resolved.get(task) != [task] for task in expected_tasks)
        or not isinstance(loaded, Mapping)
        or set(loaded) != set(expected_tasks)
        or any(loaded.get(task) is not True for task in expected_tasks)
    ):
        raise ExecutionError(
            "runtime preflight did not resolve and load the exact paper tasks"
        )

    models = payload.get("models")
    expected_models = tuple(plan.get("models", {}).keys())
    if not isinstance(models, Mapping) or set(models) != set(expected_models):
        raise ExecutionError(
            "runtime preflight model set differs from the experiment plan"
        )
    for model_name in expected_models:
        model = models.get(model_name)
        expected_signature = plan.get("model_signatures", {}).get(model_name)
        expected_path = Path(plan["models"][model_name])
        if not expected_path.is_absolute():
            expected_path = cwd / expected_path
        expected_path = expected_path.resolve(strict=False)
        if (
            not isinstance(model, Mapping)
            or model.get("exists") is not True
            or model.get("errors") != []
            or model.get("path") != str(expected_path)
            or model.get("actual_signature") != expected_signature
        ):
            raise ExecutionError(
                f"runtime preflight model identity/signature check failed "
                f"for {model_name}"
            )
    return {
        "configured": True,
        "path": str(path),
        "sha256": _sha256_file(path),
    }


class _SignalForwarded(BaseException):
    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = signum


@dataclass(frozen=True)
class ExecutionResult:
    manifest: dict[str, Any]
    exit_code: int | None
    manifest_path: Path | None
    log_path: Path | None


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _local_now() -> str:
    return dt.datetime.now().astimezone().isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sampled_file_sha256(
    path: Path, *, sample_bytes: int = MODEL_SAMPLE_BYTES
) -> tuple[str, str]:
    """Return a cheap content identity and the exact hashing mode used."""
    size = path.stat().st_size
    if size <= sample_bytes * 2:
        return _sha256_file(path), "full_sha256"

    digest = hashlib.sha256()
    digest.update(b"lowbit-activation-shard-sample-v1\0")
    digest.update(str(size).encode("ascii"))
    digest.update(b"\0")
    with path.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
        handle.seek(-sample_bytes, os.SEEK_END)
        digest.update(handle.read(sample_bytes))
    return digest.hexdigest(), f"sha256(first_{sample_bytes}+last_{sample_bytes}+size)"


def _file_stat(path: Path) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }
    if path.is_symlink():
        result["symlink_target"] = os.readlink(path)
    return result


def _full_file_identity(path: Path) -> dict[str, Any]:
    result = _file_stat(path)
    result["sha256"] = _sha256_file(path)
    after = _file_stat(path)
    result["stable_during_hash"] = all(
        result[key] == after[key]
        for key in ("size_bytes", "mtime_ns", "device", "inode")
    )
    return result


def _extract_option(argv: Sequence[str], option: str) -> str | None:
    for index, item in enumerate(argv):
        if item == option:
            if index + 1 >= len(argv):
                return None
            return argv[index + 1]
        prefix = f"{option}="
        if item.startswith(prefix):
            return item[len(prefix) :]
    return None


def collect_model_identity(
    rendered: RenderedCommand,
    *,
    cwd: Path,
) -> dict[str, Any]:
    model_argument = _extract_option(rendered.argv, "--model")
    result: dict[str, Any] = {
        "argument": model_argument,
        "complete": False,
        "identity_algorithm": (
            "config/index full SHA256; shard size+mtime+inode+"
            "first/last 1MiB sampled SHA256"
        ),
    }
    if model_argument is None:
        result["error"] = "rendered argv has no --model value"
        return result

    model_path = Path(model_argument)
    if not model_path.is_absolute():
        model_path = cwd / model_path
    model_path = model_path.resolve(strict=False)
    result["resolved_path"] = str(model_path)
    if not model_path.is_dir():
        result["error"] = "local model directory does not exist"
        return result

    config_path = model_path / "config.json"
    if not config_path.is_file():
        result["error"] = "model config.json does not exist"
        return result
    try:
        config_content = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        result["error"] = f"cannot parse model config.json: {exc}"
        return result
    config_identity = _full_file_identity(config_path)
    if not config_identity["stable_during_hash"]:
        result["error"] = "model config.json changed while it was hashed"
        return result
    config_identity["content"] = config_content
    result["config"] = config_identity

    index_path: Path | None = None
    index_content: dict[str, Any] | None = None
    for candidate_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        candidate = model_path / candidate_name
        if candidate.is_file():
            index_path = candidate
            try:
                parsed = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                result["error"] = f"cannot parse {candidate_name}: {exc}"
                return result
            if not isinstance(parsed, dict):
                result["error"] = f"{candidate_name} must contain a JSON object"
                return result
            index_content = parsed
            break

    shard_names: list[str]
    if index_path is not None and index_content is not None:
        weight_map = index_content.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            result["error"] = f"{index_path.name} has no non-empty weight_map"
            return result
        shard_names = sorted({str(value) for value in weight_map.values()})
        result["weight_index"] = _full_file_identity(index_path)
        if not result["weight_index"]["stable_during_hash"]:
            result["error"] = f"{index_path.name} changed while it was hashed"
            return result
        result["weight_index"]["tensor_count"] = len(weight_map)
        result["weight_index"]["referenced_shard_count"] = len(shard_names)
    else:
        shard_paths = sorted(model_path.glob("*.safetensors"))
        if not shard_paths:
            shard_paths = sorted(model_path.glob("pytorch_model*.bin"))
        shard_names = [path.name for path in shard_paths]
        result["weight_index"] = None

    if not shard_names:
        result["error"] = "no model weight shards were found"
        return result

    shard_records: list[dict[str, Any]] = []
    missing_shards: list[str] = []
    for shard_name in shard_names:
        shard_path = model_path / shard_name
        if not shard_path.is_file():
            missing_shards.append(shard_name)
            continue
        record = _file_stat(shard_path)
        record["relative_path"] = shard_name
        sampled_sha256, hash_mode = _sampled_file_sha256(shard_path)
        after = _file_stat(shard_path)
        record["stable_during_hash"] = all(
            record[key] == after[key]
            for key in ("size_bytes", "mtime_ns", "device", "inode")
        )
        if not record["stable_during_hash"]:
            result["error"] = f"weight shard changed while hashed: {shard_name}"
            result["shards"] = shard_records + [record]
            return result
        record["sampled_sha256"] = sampled_sha256
        record["hash_mode"] = hash_mode
        shard_records.append(record)

    result["shards"] = shard_records
    result["missing_shards"] = missing_shards
    if missing_shards:
        result["error"] = "one or more weight-index shards are missing"
        return result

    identity_payload = {
        "config_sha256": config_identity["sha256"],
        "weight_index_sha256": (
            result["weight_index"]["sha256"]
            if result["weight_index"] is not None
            else None
        ),
        "shards": [
            {
                "relative_path": record["relative_path"],
                "size_bytes": record["size_bytes"],
                "mtime_ns": record["mtime_ns"],
                "device": record["device"],
                "inode": record["inode"],
                "sampled_sha256": record["sampled_sha256"],
                "hash_mode": record["hash_mode"],
            }
            for record in shard_records
        ],
    }
    result["combined_identity_sha256"] = _sha256_bytes(
        _canonical_json_bytes(identity_payload)
    )
    result["total_weight_bytes"] = sum(
        int(record["size_bytes"]) for record in shard_records
    )
    result["complete"] = True
    return result


def _capture_command(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout: float = 20.0,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(argv),
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return {
            "argv": list(argv),
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "argv": list(argv),
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def collect_git_snapshot(*, cwd: Path) -> dict[str, Any]:
    return {
        "head": _capture_command(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
        ),
        "status": _capture_command(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            cwd=cwd,
        ),
    }


def collect_gpu_snapshot(*, cwd: Path) -> dict[str, Any]:
    query_fields = (
        "index,uuid,name,driver_version,memory.total,memory.used,memory.free,"
        "utilization.gpu,temperature.gpu,pstate"
    )
    return {
        "query": _capture_command(
            [
                "nvidia-smi",
                f"--query-gpu={query_fields}",
                "--format=csv,noheader,nounits",
            ],
            cwd=cwd,
        ),
        "list": _capture_command(["nvidia-smi", "-L"], cwd=cwd),
    }


def collect_package_versions() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, distribution in PACKAGE_DISTRIBUTIONS.items():
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
        result[label] = {
            "distribution": distribution,
            "version": version,
        }
    return result


def collect_host_snapshot() -> dict[str, Any]:
    uname = platform.uname()
    return {
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
        "username": getpass.getuser(),
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "platform": platform.platform(),
        "uname": {
            "system": uname.system,
            "node": uname.node,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
            "processor": uname.processor,
        },
        "python": {
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
    }


def _is_sensitive_env_key(key: str) -> bool:
    upper = key.upper()
    if any(fragment in upper for fragment in _SENSITIVE_ENV_FRAGMENTS):
        return True
    return (
        upper == "TOKEN"
        or upper.endswith("_TOKEN")
        or "_TOKEN_" in upper
    )


def snapshot_environment(
    environment: Mapping[str, str],
) -> tuple[dict[str, Any], list[str], str]:
    visible: dict[str, Any] = {}
    redacted: list[str] = []
    for key, value in sorted(environment.items()):
        if _is_sensitive_env_key(key):
            visible[key] = {
                "redacted": True,
                "value_sha256": _sha256_bytes(value.encode("utf-8")),
                "value_length": len(value),
            }
            redacted.append(key)
        else:
            visible[key] = value
    full_hash = _sha256_bytes(
        _canonical_json_bytes(dict(sorted(environment.items())))
    )
    return visible, redacted, full_hash


def _resolve_output_paths(
    rendered: RenderedCommand,
    *,
    plan: Mapping[str, Any],
    cwd: Path,
) -> tuple[Path, Path]:
    output_root_value = plan.get("output_root")
    if not isinstance(output_root_value, str) or not output_root_value:
        raise ExecutionError("plan output_root must be a non-empty string")
    output_root = Path(output_root_value)
    if not output_root.is_absolute():
        output_root = cwd / output_root
    output_root = output_root.resolve(strict=False)

    output_dir = rendered.output_dir
    if not output_dir.is_absolute():
        output_dir = cwd / output_dir
    output_dir = output_dir.resolve(strict=False)
    try:
        output_dir.relative_to(output_root)
    except ValueError as exc:
        raise ExecutionError(
            f"rendered output_dir escapes plan output_root: "
            f"{output_dir} not under {output_root}"
        ) from exc
    return output_root, output_dir


def _load_plan_snapshot(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
) -> dict[str, Any]:
    resolved_path = plan_path.resolve(strict=True)
    raw = resolved_path.read_bytes()
    try:
        disk_content = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExecutionError(f"cannot parse plan snapshot: {exc}") from exc
    if disk_content != plan:
        raise ExecutionError(
            "in-memory plan differs from the plan file; refusing ambiguous run"
        )
    return {
        "path": str(resolved_path),
        "sha256": _sha256_bytes(raw),
        "size_bytes": len(raw),
        "content": disk_content,
    }


def _source_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    result = _full_file_identity(path)
    result["exists"] = True
    return result


def collect_numerical_source_tree(*, cwd: Path) -> dict[str, Any]:
    """Hash every in-repo Python source that can affect experiment numerics."""

    candidates: set[Path] = set()
    for directory_name in ("realq", "gptq_utils", "utils"):
        directory = cwd / directory_name
        if directory.is_dir():
            candidates.update(
                path for path in directory.rglob("*.py") if path.is_file()
            )
    for filename in ("process_args.py", "ptq.py", "save_grads.py"):
        path = cwd / filename
        if path.is_file():
            candidates.add(path)

    files: list[dict[str, Any]] = []
    combined_records: list[dict[str, str]] = []
    for path in sorted(candidates):
        identity = _full_file_identity(path)
        relative = str(path.relative_to(cwd))
        files.append(
            {
                "relative_path": relative,
                **identity,
            }
        )
        combined_records.append(
            {
                "relative_path": relative,
                "sha256": identity["sha256"],
            }
        )
    return {
        "algorithm": "sha256(canonical_json(relative_path,sha256)[])",
        "file_count": len(files),
        "combined_sha256": _sha256_bytes(
            _canonical_json_bytes(combined_records)
        ),
        "files": files,
    }


def prepare_manifest(
    rendered: RenderedCommand,
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    cwd: Path = REPO_ROOT,
) -> tuple[dict[str, Any], Path]:
    cwd = cwd.resolve(strict=True)
    output_root, output_dir = _resolve_output_paths(
        rendered,
        plan=plan,
        cwd=cwd,
    )
    plan_snapshot = _load_plan_snapshot(plan=plan, plan_path=plan_path)
    model = collect_model_identity(rendered, cwd=cwd)

    effective_environment = dict(os.environ)
    effective_environment.update(rendered.env)
    environment_snapshot, redacted_keys, environment_hash = (
        snapshot_environment(effective_environment)
    )
    (
        explicit_environment,
        explicit_redacted_keys,
        explicit_environment_hash,
    ) = snapshot_environment(rendered.env)
    safe_shell_preview = (
        None if explicit_redacted_keys else rendered.shell()
    )

    prepared_utc = _utc_now()
    manifest_path = output_dir / MANIFEST_FILENAME
    log_path = output_dir / LOG_FILENAME
    lock_path = output_dir / LOCK_FILENAME
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "execution_id": (
            f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-"
            f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
        ),
        "run_id": rendered.run_id,
        "status": "dry_run",
        "execute_requested": False,
        "exit_code": None,
        "timestamps": {
            "prepared_utc": prepared_utc,
            "prepared_local": _local_now(),
            "started_utc": None,
            "ended_utc": None,
            "duration_seconds": None,
        },
        "paths": {
            "cwd": str(cwd),
            "output_root": str(output_root),
            "output_dir": str(output_dir),
            "manifest": str(manifest_path),
            "log": str(log_path),
            "lock": str(lock_path),
        },
        "command": {
            "argv": list(rendered.argv),
            "env": explicit_environment,
            "env_redacted_keys": explicit_redacted_keys,
            "env_sha256": explicit_environment_hash,
            "shell_preview": safe_shell_preview,
            "shell_preview_omitted_for_sensitive_env": bool(
                explicit_redacted_keys
            ),
            "cwd": str(cwd),
            "uses_shell": False,
            "stdout": str(log_path),
            "stderr": f"STDOUT -> {log_path}",
        },
        "effective_environment": environment_snapshot,
        "effective_environment_redacted_keys": redacted_keys,
        "effective_environment_sha256": environment_hash,
        "plan": plan_snapshot,
        "model": model,
        "git": {
            "prepared": collect_git_snapshot(cwd=cwd),
            "finished": None,
        },
        "host": collect_host_snapshot(),
        "python_packages": collect_package_versions(),
        "gpu": {
            "before": collect_gpu_snapshot(cwd=cwd),
            "after": None,
        },
        "source_files": {
            "runner": _source_identity(Path(runner.__file__).resolve()),
            "executor": _source_identity(Path(__file__).resolve()),
        },
        "numerical_source_tree": collect_numerical_source_tree(cwd=cwd),
        "runtime_preflight": _runtime_preflight_identity(
            plan,
            cwd=cwd,
        ),
        "process": {
            "pid": None,
            "returncode": None,
            "termination_signal": None,
            "launch_error": None,
            "wrapper_error": None,
        },
        "safety": {
            "model_identity_complete": bool(model.get("complete")),
            "output_writes_performed": False,
            "lock_acquired": False,
            "preexisting_artifacts": [],
        },
    }
    return manifest, output_dir


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                manifest,
                handle,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary_path.unlink()
        raise


@contextlib.contextmanager
def execution_lock(output_dir: Path) -> Iterator[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / LOCK_FILENAME
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ConcurrentRunError(
                f"another execution owns lock {lock_path}"
            ) from exc
        lock_payload = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "acquired_utc": _utc_now(),
        }
        encoded = (
            json.dumps(lock_payload, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, encoded)
        os.fsync(fd)
        yield lock_path
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _preexisting_artifacts(output_dir: Path) -> list[str]:
    if not output_dir.exists():
        return []
    return sorted(
        entry.name
        for entry in output_dir.iterdir()
        if entry.name != LOCK_FILENAME
    )


def _current_plan_sha256(plan_path: str) -> str | None:
    try:
        return _sha256_file(Path(plan_path))
    except OSError:
        return None


def _finish_manifest(
    manifest: dict[str, Any],
    *,
    cwd: Path,
    started_monotonic: float,
) -> None:
    manifest["timestamps"]["ended_utc"] = _utc_now()
    manifest["timestamps"]["duration_seconds"] = round(
        time.monotonic() - started_monotonic,
        6,
    )
    manifest["gpu"]["after"] = collect_gpu_snapshot(cwd=cwd)
    manifest["git"]["finished"] = collect_git_snapshot(cwd=cwd)
    plan_end_hash = _current_plan_sha256(manifest["plan"]["path"])
    manifest["plan"]["sha256_at_end"] = plan_end_hash
    manifest["plan"]["changed_during_execution"] = (
        plan_end_hash != manifest["plan"]["sha256"]
    )
    preflight = manifest.get("runtime_preflight")
    if (
        isinstance(preflight, dict)
        and preflight.get("exists") is True
        and isinstance(preflight.get("path"), str)
    ):
        preflight_end_hash = _current_plan_sha256(preflight["path"])
        preflight["sha256_at_end"] = preflight_end_hash
        preflight["changed_during_execution"] = (
            preflight_end_hash != preflight.get("sha256")
        )
    source_tree = manifest.get("numerical_source_tree")
    if isinstance(source_tree, dict):
        source_tree_end = collect_numerical_source_tree(cwd=cwd)
        source_tree["combined_sha256_at_end"] = source_tree_end[
            "combined_sha256"
        ]
        source_tree["changed_during_execution"] = (
            source_tree["combined_sha256_at_end"]
            != source_tree.get("combined_sha256")
        )


def _terminate_process_group(
    process: subprocess.Popen[Any],
    signum: int,
) -> None:
    if process.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signum)
    try:
        process.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        pass
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=10)


@contextlib.contextmanager
def _forward_termination_signals() -> Iterator[None]:
    watched = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous: dict[int, Any] = {}

    def handler(signum: int, _frame: Any) -> None:
        raise _SignalForwarded(signum)

    for signum in watched:
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, handler)
    try:
        yield
    finally:
        for signum, old_handler in previous.items():
            signal.signal(signum, old_handler)


def run_rendered(
    rendered: RenderedCommand,
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    cwd: Path = REPO_ROOT,
    execute: bool = False,
) -> ExecutionResult:
    """Preview or execute one RenderedCommand with immutable provenance."""
    cwd = cwd.resolve(strict=True)
    validated_preflight: dict[str, Any] | None = None
    if execute:
        _require_python_environment(plan)
        validated_preflight = _require_runtime_preflight(
            plan,
            plan_path=plan_path,
            cwd=cwd,
        )
    manifest, output_dir = prepare_manifest(
        rendered,
        plan=plan,
        plan_path=plan_path,
        cwd=cwd,
    )
    manifest_path = output_dir / MANIFEST_FILENAME
    log_path = output_dir / LOG_FILENAME
    if not execute:
        return ExecutionResult(
            manifest=manifest,
            exit_code=None,
            manifest_path=None,
            log_path=None,
        )

    if not manifest["model"]["complete"]:
        raise ExecutionError(
            "model provenance is incomplete; refusing execution: "
            f"{manifest['model'].get('error', 'unknown model identity error')}"
        )

    current_plan_hash = _current_plan_sha256(manifest["plan"]["path"])
    if current_plan_hash != manifest["plan"]["sha256"]:
        raise ExecutionError(
            "plan changed after command rendering; refusing execution"
        )
    current_preflight = _require_runtime_preflight(
        plan,
        plan_path=plan_path,
        cwd=cwd,
    )
    if validated_preflight is None or current_preflight != validated_preflight:
        raise ExecutionError(
            "runtime preflight changed after the initial execution gate"
        )
    runtime_gate = _require_current_runtime(
        rendered,
        plan=plan,
        cwd=cwd,
    )
    manifest["launch_gate"] = runtime_gate
    manifest["runtime_preflight"]["validated_sha256"] = validated_preflight[
        "sha256"
    ]

    with execution_lock(output_dir):
        existing = _preexisting_artifacts(output_dir)
        manifest["safety"]["preexisting_artifacts"] = existing
        if existing:
            raise ExistingRunError(
                f"output directory already contains artifacts: {existing!r}"
            )

        # Recheck the two mutable control files immediately before the first
        # experiment artifact and child launch. The lock protects the output
        # identity; these hashes protect the reviewed control identity.
        current_plan_hash = _current_plan_sha256(manifest["plan"]["path"])
        if current_plan_hash != manifest["plan"]["sha256"]:
            raise ExecutionError(
                "plan changed immediately before launch; refusing execution"
            )
        current_preflight_hash = _current_plan_sha256(
            validated_preflight["path"]
        )
        if current_preflight_hash != validated_preflight["sha256"]:
            raise ExecutionError(
                "runtime preflight changed immediately before launch"
            )
        _require_python_environment(plan)
        _require_rendered_executable(rendered, plan)
        effective_environment = dict(os.environ)
        effective_environment.update(rendered.env)
        started_monotonic = time.monotonic()
        manifest["execute_requested"] = True
        manifest["status"] = "launching"
        manifest["timestamps"]["started_utc"] = _utc_now()
        manifest["safety"]["output_writes_performed"] = True
        manifest["safety"]["lock_acquired"] = True

        with log_path.open("xb") as log_handle:
            atomic_write_manifest(manifest_path, manifest)
            try:
                process = subprocess.Popen(
                    list(rendered.argv),
                    cwd=cwd,
                    env=effective_environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except (OSError, ValueError) as exc:
                manifest["status"] = "launch_failed"
                manifest["process"]["launch_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                _finish_manifest(
                    manifest,
                    cwd=cwd,
                    started_monotonic=started_monotonic,
                )
                atomic_write_manifest(manifest_path, manifest)
                raise ExecutionError(
                    f"failed to launch command; see {manifest_path}: {exc}"
                ) from exc

            manifest["status"] = "running"
            manifest["process"]["pid"] = process.pid
            try:
                atomic_write_manifest(manifest_path, manifest)
                try:
                    with _forward_termination_signals():
                        returncode = process.wait()
                except _SignalForwarded as exc:
                    _terminate_process_group(process, exc.signum)
                    manifest["status"] = "interrupted"
                    manifest["process"]["termination_signal"] = exc.signum
                    returncode = (
                        process.returncode
                        if process.returncode is not None
                        else -(exc.signum)
                    )
                else:
                    if returncode == 0:
                        manifest["status"] = "succeeded"
                    elif returncode < 0:
                        manifest["status"] = "terminated"
                        manifest["process"]["termination_signal"] = -returncode
                    else:
                        manifest["status"] = "failed"

                log_handle.flush()
                os.fsync(log_handle.fileno())
                manifest["exit_code"] = returncode
                manifest["process"]["returncode"] = returncode
                _finish_manifest(
                    manifest,
                    cwd=cwd,
                    started_monotonic=started_monotonic,
                )
                atomic_write_manifest(manifest_path, manifest)
            except BaseException as exc:
                # If provenance persistence or the wrapper itself fails after
                # launch, never leave a detached torchrun process consuming
                # GPUs without an owning wrapper.
                _terminate_process_group(process, signal.SIGTERM)
                manifest["status"] = "wrapper_failed"
                manifest["exit_code"] = process.returncode
                manifest["process"]["returncode"] = process.returncode
                manifest["process"]["wrapper_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                with contextlib.suppress(BaseException):
                    _finish_manifest(
                        manifest,
                        cwd=cwd,
                        started_monotonic=started_monotonic,
                    )
                    atomic_write_manifest(manifest_path, manifest)
                raise

    normalized_exit_code = (
        128 + abs(returncode) if returncode < 0 else returncode
    )
    return ExecutionResult(
        manifest=manifest,
        exit_code=normalized_exit_code,
        manifest_path=manifest_path,
        log_path=log_path,
    )


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="launch the command; omission is a read-only dry run",
    )
    parser.add_argument(
        "--phase",
        choices=("precompute", "tune", "final"),
        required=True,
    )
    parser.add_argument(
        "--method",
        choices=(
            "realq",
            "realq_static",
            "gptaq",
            "guided_gptq",
            "guided_saliency",
            "bf16",
        ),
        required=True,
    )
    parser.add_argument("--target-phase", choices=("tune", "final"))
    parser.add_argument("--model", choices=runner.MODEL_ORDER, required=True)
    parser.add_argument("--setting", choices=runner.SETTING_ORDER)
    parser.add_argument("--grad-lr", type=float)
    parser.add_argument("--cuda-devices", required=True)
    parser.add_argument(
        "--attempt-index",
        type=int,
        default=1,
        help=(
            "1-based launch attempt for this numerical candidate; retries "
            "above one get a unique immutable output identity. Values above "
            "20 are forbidden; all tuning launches must go through the "
            "campaign controller, which enforces the 20-attempt total across "
            "all LR candidates for one model/setting."
        ),
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=INTEGER",
        help="batch/memory-only override; may be repeated",
    )
    return parser


def _with_attempt_identity(
    rendered: RenderedCommand,
    attempt_index: int,
) -> RenderedCommand:
    if isinstance(attempt_index, bool) or attempt_index <= 0:
        raise runner.PlanError("--attempt-index must be a positive integer")
    if attempt_index > MAX_ATTEMPT_INDEX:
        raise runner.PlanError(
            f"--attempt-index cannot exceed {MAX_ATTEMPT_INDEX}; do not bypass "
            "the campaign controller's per-model/setting total-attempt ledger"
        )
    if attempt_index == 1:
        return RenderedCommand(
            env={
                **rendered.env,
                "LOWBIT_ACTIVATION_ATTEMPT_INDEX": "1",
            },
            argv=rendered.argv,
            run_id=rendered.run_id,
            output_dir=rendered.output_dir,
        )
    suffix = f"_attempt{attempt_index}"
    argv = list(rendered.argv)
    runner._set_option(argv, "--exp", f"{rendered.run_id}{suffix}")
    return RenderedCommand(
        env={
            **rendered.env,
            "LOWBIT_ACTIVATION_ATTEMPT_INDEX": str(attempt_index),
        },
        argv=argv,
        run_id=f"{rendered.run_id}{suffix}",
        output_dir=rendered.output_dir.parent
        / f"{rendered.output_dir.name}{suffix}",
    )


def _render_from_args(
    args: argparse.Namespace,
    plan: dict[str, Any],
) -> RenderedCommand:
    overrides = runner._parse_overrides(args.override)
    if args.method != "realq_static" and args.target_phase is not None:
        raise runner.PlanError("--target-phase is only for realq_static")

    if args.method == "realq_static":
        if args.phase != "precompute":
            raise runner.PlanError(
                "realq_static is rendered only in the precompute phase"
            )
        if args.target_phase is None:
            raise runner.PlanError("realq_static requires --target-phase")
        if args.setting is not None or args.grad_lr is not None:
            raise runner.PlanError(
                "realq_static does not accept --setting or --grad-lr"
            )
        return runner.render_realq_static_precompute(
            plan,
            target_phase=args.target_phase,
            model=args.model,
            cuda_devices=args.cuda_devices,
            overrides=overrides,
        )

    if args.method == "guided_saliency":
        if args.phase != "precompute":
            raise runner.PlanError(
                "guided_saliency is rendered only in the precompute phase"
            )
        if args.setting is not None or args.grad_lr is not None:
            raise runner.PlanError(
                "guided_saliency does not accept --setting or --grad-lr"
            )
        if overrides:
            raise runner.PlanError("guided_saliency does not accept overrides")
        return runner.render_guided_saliency(
            plan,
            model=args.model,
            cuda_devices=args.cuda_devices,
        )

    if args.method == "realq":
        if args.setting is None:
            raise runner.PlanError("REAL-Q requires --setting")
        if args.grad_lr is None:
            raise runner.PlanError("REAL-Q requires --grad-lr")
        return runner.render_realq(
            plan,
            phase=args.phase,
            model=args.model,
            setting_name=args.setting,
            grad_lr=args.grad_lr,
            cuda_devices=args.cuda_devices,
            overrides=overrides,
        )

    if args.phase != "final":
        raise runner.PlanError("baselines are rendered only in the final phase")
    return runner.render_baseline(
        plan,
        method=args.method,
        model=args.model,
        setting_name=args.setting,
        cuda_devices=args.cuda_devices,
        overrides=overrides,
    )


def _load_validated_plan_once(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    plan = json.loads(raw)
    runner.validate_structure(plan)
    return plan


def main(argv: list[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    try:
        plan = _load_validated_plan_once(args.plan)
        rendered = _render_from_args(args, plan)
        rendered = _with_attempt_identity(rendered, args.attempt_index)
        result = run_rendered(
            rendered,
            plan=plan,
            plan_path=args.plan,
            cwd=REPO_ROOT,
            execute=args.execute,
        )
    except (
        ExecutionError,
        OSError,
        json.JSONDecodeError,
        runner.PlanError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not args.execute:
        print(json.dumps(result.manifest, indent=2, ensure_ascii=False))
        return 0

    print(
        json.dumps(
            {
                "run_id": result.manifest["run_id"],
                "status": result.manifest["status"],
                "exit_code": result.exit_code,
                "manifest": str(result.manifest_path),
                "log": str(result.log_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return int(result.exit_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
