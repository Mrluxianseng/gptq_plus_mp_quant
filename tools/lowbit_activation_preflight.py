#!/usr/bin/env python3
"""Runtime preflight for the low-bit activation experiment campaign.

Run this inside the target Canoe container before any long experiment.  It
checks the exact model artifacts, Python/CUDA stack, paper task resolution,
and (optionally) instantiates every lm-eval task so missing datasets fail
before quantization starts.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import lowbit_activation_runner as runner  # noqa: E402


DEFAULT_PLAN = ROOT / "experiments" / "lowbit_activation" / "plan.json"
EXPECTED_LM_EVAL_VERSION = "0.4.4"
MINIMUM_TARGET_GPU_COUNT = 8
GIB = 1024**3
AUTHORIZED_AUXILIARY_JOBS = {
    "j-8j1en3m0aq",
    "j-ryis586loy",
    "j-nxfty1rcqx",
    "j-28b993voy4",
    "j-hzunwre5gq",
}


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_plan_structure(
    plan: dict[str, Any],
    *,
    plan_path: Path,
) -> list[str]:
    """Accept only the canonical plan or an exact authorized job derivative."""

    try:
        runner.validate_structure(plan)
        return []
    except runner.PlanError as original_error:
        marker = plan.get("formal_auxiliary_job")
        if not isinstance(marker, dict):
            return [f"experiment plan validation failed: {original_error}"]

        job_id = marker.get("job_id")
        preflight = marker.get("runtime_preflight_manifest")
        if (
            marker.get("schema_version") != 1
            or job_id not in AUTHORIZED_AUXILIARY_JOBS
            or not isinstance(preflight, str)
            or preflight
            != f"output/lowbit_activation/runtime_preflight_{job_id}.json"
        ):
            return ["formal auxiliary plan authorization is invalid"]

        base_path = DEFAULT_PLAN.resolve(strict=True)
        try:
            base_plan = json.loads(base_path.read_text(encoding="utf-8"))
            runner.validate_structure(base_plan)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, runner.PlanError) as exc:
            return [f"canonical base plan validation failed: {exc}"]

        base_sha = _sha256_file(base_path)
        if marker.get("base_plan_sha256") != base_sha:
            return ["formal auxiliary plan base-plan identity drifted"]

        expected = copy.deepcopy(base_plan)
        expected["resolutions"]["canoe_job_id"] = job_id
        expected["resolutions"]["runtime_preflight_manifest"] = preflight
        expected["formal_auxiliary_job"] = dict(marker)
        if plan != expected:
            return [
                "formal auxiliary plan differs from the canonical plan "
                "outside the authorized job/preflight fields"
            ]
        if plan_path.resolve(strict=False) == base_path:
            return ["canonical plan path must not contain an auxiliary derivative"]
        return []


def _canoe_context(
    plan: dict[str, Any],
    *,
    canoe_job_id: str | None,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    resolutions = plan.get("resolutions")
    expected = (
        resolutions.get("canoe_job_id")
        if isinstance(resolutions, dict)
        else None
    )
    hostname = socket.gethostname()
    if not isinstance(expected, str) or not expected:
        errors.append("plan resolutions.canoe_job_id must be a non-empty string")
    if canoe_job_id != expected:
        errors.append(
            f"preflight job id {canoe_job_id!r} does not match plan {expected!r}"
        )
    if isinstance(expected, str) and expected and not hostname.startswith(
        f"{expected}-"
    ):
        errors.append(
            f"hostname {hostname!r} does not belong to planned job {expected!r}"
        )
    return {
        "expected_job_id": expected,
        "supplied_job_id": canoe_job_id,
        "hostname": hostname,
        "valid": not errors,
    }, errors


def _python_environment(
    plan: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Verify that AGENT.md's isolated experiment environment is active."""

    errors: list[str] = []
    configured = plan.get("python_environment")
    if not isinstance(configured, dict):
        return {}, ["plan python_environment must be an object"]

    expected_value = configured.get("venv")
    activation_required = configured.get("activation_required")
    if not isinstance(expected_value, str) or not expected_value:
        return {}, ["plan python_environment.venv must be a non-empty path"]
    if activation_required is not True:
        errors.append("plan python_environment.activation_required must be true")

    expected = Path(expected_value).absolute()
    expected_resolved = expected.resolve(strict=False)
    virtual_env_value = os.environ.get("VIRTUAL_ENV")
    virtual_env_resolved = (
        Path(virtual_env_value).resolve(strict=False)
        if virtual_env_value
        else None
    )
    prefix_resolved = Path(sys.prefix).resolve(strict=False)
    executable = Path(sys.executable).absolute()

    tools = {
        name: shutil.which(name)
        for name in ("python", "python3", "torchrun")
    }

    if not expected.is_dir():
        errors.append(f"required experiment venv is missing: {expected}")
    if virtual_env_resolved != expected_resolved:
        errors.append(
            "required experiment venv is not activated: "
            f"VIRTUAL_ENV={virtual_env_value!r}, expected {str(expected)!r}"
        )
    if prefix_resolved != expected_resolved:
        errors.append(
            "current Python does not use the required experiment venv: "
            f"sys.prefix={sys.prefix!r}, expected {str(expected)!r}"
        )

    for name, value in tools.items():
        if value is None:
            errors.append(f"required venv executable is missing from PATH: {name}")
            continue
        candidate = Path(value).absolute()
        try:
            candidate.relative_to(expected)
        except ValueError:
            errors.append(
                f"{name} resolves outside the activated experiment venv: {value}"
            )

    return {
        "expected_venv": str(expected),
        "expected_venv_resolved": str(expected_resolved),
        "exists": expected.is_dir(),
        "activation_required": activation_required,
        "virtual_env": virtual_env_value,
        "sys_prefix": sys.prefix,
        "sys_base_prefix": sys.base_prefix,
        "sys_executable": str(executable),
        "path_executables": tools,
        "valid": not errors,
    }, errors


def _git(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *command],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _check_model(
    path: Path,
    expected_signature: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_dir(),
        "errors": [],
    }
    if not path.is_dir():
        result["errors"].append("model directory is missing")
        return result
    config_path = path / "config.json"
    if not config_path.is_file():
        result["errors"].append("config.json is missing")
        return result
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception as exc:  # noqa: BLE001 - report malformed artifact
        result["errors"].append(
            f"config.json is unreadable: {type(exc).__name__}: {exc}"
        )
        return result
    result["model_type"] = config.get("model_type")
    result["num_hidden_layers"] = config.get("num_hidden_layers")
    result["hidden_size"] = config.get("hidden_size")
    result["architectures"] = config.get("architectures")
    if expected_signature is not None:
        actual_signature = {
            key: config.get(key)
            for key in (
                "model_type",
                "hidden_size",
                "num_hidden_layers",
                "architectures",
            )
        }
        result["expected_signature"] = expected_signature
        result["actual_signature"] = actual_signature
        if actual_signature != expected_signature:
            result["errors"].append(
                "model config signature mismatch: "
                f"expected {expected_signature!r}, got {actual_signature!r}"
            )

    index_paths = sorted(path.glob("*.safetensors.index.json"))
    if index_paths:
        try:
            with index_paths[0].open("r", encoding="utf-8") as handle:
                index = json.load(handle)
            shards = sorted(set(index["weight_map"].values()))
        except Exception as exc:  # noqa: BLE001
            result["errors"].append(
                f"weight index is unreadable: {type(exc).__name__}: {exc}"
            )
            shards = []
        missing = [name for name in shards if not (path / name).is_file()]
        empty = [
            name
            for name in shards
            if (path / name).is_file() and (path / name).stat().st_size == 0
        ]
        result["weight_index"] = str(index_paths[0])
        result["shard_count"] = len(shards)
        result["missing_shards"] = missing
        result["empty_shards"] = empty
        if missing:
            result["errors"].append(f"missing weight shards: {missing}")
        if empty:
            result["errors"].append(f"empty weight shards: {empty}")
    else:
        single = path / "model.safetensors"
        result["shard_count"] = 1 if single.is_file() else 0
        if not single.is_file() or single.stat().st_size == 0:
            result["errors"].append(
                "no safetensors index and no non-empty model.safetensors"
            )
    return result


def _runtime_requirements(
    plan: dict[str, Any],
    *,
    min_gpu_memory_gib: float | None,
) -> tuple[dict[str, Any], list[str]]:
    """Resolve infrastructure gates without changing experiment numerics."""

    errors: list[str] = []
    planned_world_sizes: dict[str, int] = {}
    for phase in ("tuning", "final"):
        value = plan.get(phase, {}).get("world_size")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(
                f"plan {phase}.world_size must be a positive integer, got {value!r}"
            )
            continue
        planned_world_sizes[phase] = value

    configured = plan.get("runtime_requirements", {})
    if not isinstance(configured, dict):
        errors.append("plan runtime_requirements must be an object when present")
        configured = {}

    configured_gpu_count = configured.get("minimum_gpu_count")
    if configured_gpu_count is not None and (
        isinstance(configured_gpu_count, bool)
        or not isinstance(configured_gpu_count, int)
        or configured_gpu_count <= 0
    ):
        errors.append(
            "plan runtime_requirements.minimum_gpu_count must be a positive "
            f"integer, got {configured_gpu_count!r}"
        )
        configured_gpu_count = None

    gpu_count_candidates = [
        MINIMUM_TARGET_GPU_COUNT,
        *planned_world_sizes.values(),
    ]
    if isinstance(configured_gpu_count, int):
        gpu_count_candidates.append(configured_gpu_count)
    required_gpu_count = max(gpu_count_candidates)

    memory_candidates: list[tuple[str, float]] = []
    for source, value in (
        ("plan", configured.get("minimum_gpu_memory_gib")),
        ("cli", min_gpu_memory_gib),
    ):
        if value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            errors.append(
                f"{source} minimum GPU memory must be a positive finite GiB "
                f"value, got {value!r}"
            )
            continue
        memory_candidates.append((source, float(value)))

    if memory_candidates:
        resolved_memory_gib = max(value for _, value in memory_candidates)
        memory_source = "+".join(source for source, _ in memory_candidates)
    else:
        resolved_memory_gib = None
        memory_source = None

    return {
        "minimum_gpu_count": required_gpu_count,
        "minimum_gpu_memory_gib": (
            float(resolved_memory_gib)
            if resolved_memory_gib is not None
            else None
        ),
        "minimum_gpu_memory_bytes": (
            math.ceil(float(resolved_memory_gib) * GIB)
            if resolved_memory_gib is not None
            else None
        ),
        "memory_requirement_source": memory_source,
        "planned_world_sizes": planned_world_sizes,
    }, errors


def _validate_cuda_requirements(
    cuda: dict[str, Any],
    requirements: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    devices = cuda.get("devices", [])
    required_count = requirements["minimum_gpu_count"]
    minimum_memory_bytes = requirements["minimum_gpu_memory_bytes"]

    if minimum_memory_bytes is None:
        eligible = list(devices)
    else:
        eligible = [
            device
            for device in devices
            if isinstance(device.get("total_bytes"), int)
            and device["total_bytes"] >= minimum_memory_bytes
        ]

    result = {
        "detected_gpu_count": len(devices),
        "eligible_gpu_count": len(eligible),
        "eligible_gpu_indices": [device.get("index") for device in eligible],
        "passed": len(eligible) >= required_count,
    }
    if len(devices) < required_count:
        errors.append(
            f"at least {required_count} visible CUDA GPUs are required, "
            f"detected {len(devices)}"
        )
    elif len(eligible) < required_count:
        errors.append(
            f"at least {required_count} visible CUDA GPUs with "
            f">={requirements['minimum_gpu_memory_gib']} GiB total memory "
            f"are required, detected {len(eligible)} eligible out of "
            f"{len(devices)}"
        )
    return result, errors


def _runtime(
    plan: dict[str, Any],
    *,
    min_gpu_memory_gib: float | None,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    requirements, requirement_errors = _runtime_requirements(
        plan,
        min_gpu_memory_gib=min_gpu_memory_gib,
    )
    errors.extend(requirement_errors)
    python_environment, python_environment_errors = _python_environment(plan)
    errors.extend(python_environment_errors)
    distributions = {
        name: _version(name)
        for name in (
            "torch",
            "transformers",
            "accelerate",
            "datasets",
            "lm-eval",
        )
    }
    for name, version in distributions.items():
        if version is None:
            errors.append(f"missing Python distribution: {name}")
    expected_versions = plan.get("runtime_versions")
    if not isinstance(expected_versions, dict):
        errors.append("plan runtime_versions must be an object")
        expected_versions = {}
    for distribution in ("torch", "transformers", "lm-eval"):
        expected = expected_versions.get(distribution)
        actual = distributions.get(distribution)
        if not isinstance(expected, str) or not expected:
            errors.append(
                f"plan runtime_versions.{distribution} must be a non-empty string"
            )
        elif actual != expected:
            errors.append(
                f"{distribution} version must be {expected}, got {actual}"
            )

    cuda: dict[str, Any] = {"available": False, "devices": []}
    try:
        torch = importlib.import_module("torch")
        cuda["torch_version"] = str(torch.__version__)
        cuda["cuda_runtime"] = torch.version.cuda
        expected_torch_runtime = expected_versions.get("torch_runtime")
        if (
            not isinstance(expected_torch_runtime, str)
            or not expected_torch_runtime
        ):
            errors.append(
                "plan runtime_versions.torch_runtime must be a non-empty string"
            )
        elif cuda["torch_version"] != expected_torch_runtime:
            errors.append(
                "torch runtime build must be "
                f"{expected_torch_runtime}, got {cuda['torch_version']}"
            )
        cuda["available"] = bool(torch.cuda.is_available())
        if not cuda["available"]:
            errors.append("torch.cuda.is_available() is false")
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info(index)
            device_uuid = getattr(props, "uuid", None)
            cuda["devices"].append(
                {
                    "index": index,
                    "name": props.name,
                    # torch 2.9 exposes ``uuid`` as a private ``_CUuuid``
                    # object rather than a JSON-native string.
                    "uuid": (
                        str(device_uuid)
                        if device_uuid is not None
                        else None
                    ),
                    "total_bytes": int(total),
                    "free_bytes": int(free),
                    "capability": [
                        int(props.major),
                        int(props.minor),
                    ],
                }
            )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"torch/CUDA probe failed: {type(exc).__name__}: {exc}")
    requirement_check, cuda_requirement_errors = _validate_cuda_requirements(
        cuda,
        requirements,
    )
    errors.extend(cuda_requirement_errors)
    cuda["requirement_check"] = requirement_check
    return {
        "python_environment": python_environment,
        "distributions": distributions,
        "cuda": cuda,
        "requirements": requirements,
    }, errors


def _lm_eval_tasks(
    expected_tasks: list[str],
    *,
    load_datasets: bool,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    result: dict[str, Any] = {
        "resolved": {},
        "datasets_loaded": {},
    }
    try:
        lm_eval = importlib.import_module("lm_eval")
        lm_utils = importlib.import_module("lm_eval.utils")
        custom_tasks = ROOT / "datasets" / "lm_eval_configs" / "tasks"
        manager_kwargs: dict[str, Any] = {"include_defaults": True}
        if custom_tasks.is_dir():
            manager_kwargs["include_path"] = str(custom_tasks)
        manager = lm_eval.tasks.TaskManager(**manager_kwargs)
        for task in expected_tasks:
            matches = lm_utils.pattern_match([task], manager.all_tasks)
            result["resolved"][task] = matches
            if len(matches) != 1:
                errors.append(
                    f"task {task!r} resolved to {matches!r}, expected one match"
                )
                continue
            if load_datasets:
                try:
                    lm_eval.tasks.get_task_dict(
                        [matches[0]],
                        task_manager=manager,
                    )
                    result["datasets_loaded"][task] = True
                except Exception as exc:  # noqa: BLE001
                    result["datasets_loaded"][task] = False
                    errors.append(
                        f"task {task!r} dataset load failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
    except Exception as exc:  # noqa: BLE001
        errors.append(
            f"lm-eval task discovery failed: {type(exc).__name__}: {exc}"
        )
    return result, errors


def _path_probe(plan: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    output_root = ROOT / plan["output_root"]
    cache_root = ROOT / plan["static_cache_root"]
    checks: dict[str, Any] = {}
    for name, path in (("output_root", output_root), ("cache_root", cache_root)):
        parent = path
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        checks[name] = {
            "path": str(path),
            "probe_path": str(parent),
            "parent_exists": parent.exists(),
            "parent_writable": os.access(parent, os.W_OK) if parent.exists() else False,
        }
        if not checks[name]["parent_exists"] or not checks[name]["parent_writable"]:
            errors.append(f"{name} parent is missing or not writable: {parent}")
    disk = shutil.disk_usage(ROOT)
    checks["workspace_disk"] = {
        "total_bytes": disk.total,
        "used_bytes": disk.used,
        "free_bytes": disk.free,
    }
    return checks, errors


def run(
    plan_path: Path,
    *,
    load_task_datasets: bool,
    min_gpu_memory_gib: float | None = None,
    canoe_job_id: str | None = None,
) -> dict[str, Any]:
    with plan_path.open("r", encoding="utf-8") as handle:
        plan = json.load(handle)
    errors: list[str] = []
    errors.extend(_validate_plan_structure(plan, plan_path=plan_path))
    canoe, canoe_errors = _canoe_context(
        plan,
        canoe_job_id=canoe_job_id,
    )
    errors.extend(canoe_errors)

    models: dict[str, Any] = {}
    model_signatures = plan.get("model_signatures", {})
    for name, relative in plan["models"].items():
        expected_signature = (
            model_signatures.get(name)
            if isinstance(model_signatures, dict)
            else None
        )
        model_result = _check_model(
            ROOT / relative,
            expected_signature=expected_signature,
        )
        models[name] = model_result
        errors.extend(f"{name}: {item}" for item in model_result["errors"])

    runtime, runtime_errors = _runtime(
        plan,
        min_gpu_memory_gib=min_gpu_memory_gib,
    )
    errors.extend(runtime_errors)
    task_result, task_errors = _lm_eval_tasks(
        plan["paper_zero_shot_tasks"],
        load_datasets=load_task_datasets,
    )
    errors.extend(task_errors)
    paths, path_errors = _path_probe(plan)
    errors.extend(path_errors)

    return {
        "schema_version": 1,
        "valid": not errors,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "load_task_datasets": load_task_datasets,
        "repo_root": str(ROOT),
        "plan": {
            "path": str(plan_path.resolve(strict=True)),
            "sha256": _sha256_file(plan_path.resolve(strict=True)),
        },
        "canoe": canoe,
        "git_head": _git(["rev-parse", "HEAD"]),
        "git_status_short": _git(["status", "--short"]),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "HF_HOME",
                "HF_DATASETS_OFFLINE",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "CUDA_VISIBLE_DEVICES",
                "PYTORCH_CUDA_ALLOC_CONF",
            )
        },
        "models": models,
        "runtime": runtime,
        "lm_eval": task_result,
        "paths": paths,
        "errors": errors,
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--load-task-datasets",
        action="store_true",
        help="instantiate all ten tasks to prove their datasets are available",
    )
    parser.add_argument(
        "--min-gpu-memory-gib",
        type=float,
        help=(
            "optionally require enough visible GPUs (at least eight or the "
            "larger planned world size) with this much total memory"
        ),
    )
    parser.add_argument(
        "--canoe-job-id",
        required=True,
        help="must exactly match plan.resolutions.canoe_job_id and hostname",
    )
    parser.add_argument("--write-manifest", type=Path)
    args = parser.parse_args()
    payload = run(
        args.plan,
        load_task_datasets=args.load_task_datasets,
        min_gpu_memory_gib=args.min_gpu_memory_gib,
        canoe_job_id=args.canoe_job_id,
    )
    if args.write_manifest:
        _atomic_write(args.write_manifest, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
