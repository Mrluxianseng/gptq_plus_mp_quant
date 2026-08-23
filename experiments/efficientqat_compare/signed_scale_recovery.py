#!/usr/bin/env python3
"""Versioned post-process recovery for finite signed EfficientQAT scales.

The frozen formal runner intentionally fails materialisation when E2E-QP has
learned a zero or negative scale.  EfficientQAT's packed decode formula,
however, remains mathematically defined for every finite scale:

    weight = (integer_code - integer_zero_point) * learned_scale

This recovery preserves those learned values exactly.  It does not take an
absolute value, clamp, retrain, or modify the packed checkpoint.  Instead it
temporarily replaces only ``materialize._validate_packed_module`` with the
same structural validator minus the strict-positivity condition, then calls
the frozen ``run_materialize`` and ``run_evaluation`` entry points.

The default CLI action is a read-only CPU validation.  Actual recovery
requires ``--execute``, a self-SHA pin, a new recovery root, and exactly the
single physical evaluation GPU assigned to the frozen run.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time
import traceback
from typing import Any, Iterator, Mapping

import torch
import torch.nn as nn

from . import launcher
from . import materialize
from . import run_one


RECOVERY_FORMAT = "efficientqat.signed_scale_postprocess_recovery"
RECOVERY_VERSION = 1
RECOVERY_VARIANT = "signed_scale_v1"
SCALE_POLICY = (
    "Preserve every finite learned scale exactly, including negative and "
    "zero values; decode (code - zero_point) * scale with no abs, clamp, "
    "retraining, or checkpoint mutation."
)
_FAILURE_PATTERN = re.compile(
    r"^model\.layers\.\d+\."
    r"(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)"
    r"\.scales contains a non-positive learned quantization step\.$"
)


class RecoveryError(RuntimeError):
    """Raised when a recovery intake or execution invariant is violated."""


@dataclass(frozen=True)
class RecoveryPreflight:
    """Validated inputs plus JSON-safe evidence for one recovery."""

    plan_path: Path
    plan: dict[str, Any]
    run: dict[str, Any]
    original_run_dir: Path
    packed_checkpoint: Path
    recovery_root: Path
    recovery_dir: Path
    physical_eval_gpu: int
    script_sha256: str
    checkpoint_manifest: dict[str, Any]
    evidence: dict[str, Any]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json_atomic(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
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
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise RecoveryError(f"{label} is missing: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"{label} is not valid JSON: {source}") from exc
    if not isinstance(value, dict):
        raise RecoveryError(f"{label} root must be a JSON object: {source}")
    return value


def _parse_time(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise RecoveryError(f"{label} must be an ISO-8601 string.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RecoveryError(f"{label} is not valid ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        raise RecoveryError(f"{label} must include a timezone: {value!r}")
    return parsed.astimezone(timezone.utc)


def _finite_nonnegative(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecoveryError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise RecoveryError(f"{label} must be finite and non-negative.")
    return result


def apply_frozen_offline_environment(
    plan: Mapping[str, Any],
) -> dict[str, str]:
    """Install the exact offline cache environment before any HF loading."""

    raw = plan.get("runtime", {}).get("offline_environment")
    if not isinstance(raw, Mapping) or not raw:
        raise RecoveryError("plan runtime.offline_environment is missing")
    expected: dict[str, str] = {}
    for key, value in raw.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
        ):
            raise RecoveryError(
                "offline_environment must contain non-empty string pairs"
            )
        expected[key] = value
    os.environ.update(expected)
    actual = {key: os.environ.get(key) for key in expected}
    if actual != expected:
        raise RecoveryError(
            f"unable to install frozen offline environment: {actual!r}"
        )
    return expected


def _get_module(root: nn.Module, name: str) -> nn.Module:
    module: nn.Module = root
    for part in name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    if not isinstance(module, nn.Module):
        raise TypeError(f"{name!r} did not resolve to an nn.Module.")
    return module


def validate_finite_signed_packed_module(
    module: nn.Module,
    *,
    name: str,
    expected_bits: int,
    expected_group_size: int,
) -> tuple[int, int, int, int]:
    """Run the frozen packed validator while allowing finite signed scales.

    This function mirrors the frozen validator's shape, device, packed-word,
    natural-group, and bias gates.  The sole policy difference is that scales
    need only be finite: negative and zero values are accepted unchanged.
    """

    if not materialize.is_efficientqat_packed_linear(module):
        raise TypeError(f"{name!r} is not an EfficientQAT packed QuantLinear.")

    bits = int(module.bits)
    group_size = int(module.group_size)
    in_features = int(module.infeatures)
    out_features = int(module.outfeatures)
    if bits != expected_bits:
        raise ValueError(
            f"{name}: packed bit-width {bits} != expected {expected_bits}."
        )
    if group_size != expected_group_size:
        raise ValueError(
            f"{name}: packed group size {group_size} != "
            f"expected {expected_group_size}."
        )
    if in_features <= 0 or out_features <= 0:
        raise ValueError(
            f"{name}: invalid feature shape ({in_features}, {out_features})."
        )
    if in_features % group_size != 0:
        raise ValueError(
            f"{name}: in_features={in_features} is not divisible by "
            f"group_size={group_size}; upstream E2E-QP cannot represent a "
            "partial final group."
        )

    groups = in_features // group_size
    scales = module.scales
    if not isinstance(scales, torch.Tensor):
        raise TypeError(f"{name}.scales must be a tensor.")
    if scales.device.type != "cpu":
        raise ValueError(
            f"{name}.scales is on {scales.device}; CPU materialisation "
            "requires a CPU-resident checkpoint."
        )
    if tuple(scales.shape) != (groups, out_features):
        raise ValueError(
            f"{name}.scales has shape {tuple(scales.shape)}; "
            f"expected {(groups, out_features)}."
        )
    if not torch.isfinite(scales).all():
        raise ValueError(f"{name}.scales contains non-finite values.")

    lanes = 32 // bits
    expected_qweight = (math.ceil(in_features / lanes), out_features)
    expected_qzeros = (groups, math.ceil(out_features / lanes))
    if tuple(module.qweight.shape) != expected_qweight:
        raise ValueError(
            f"{name}.qweight has shape {tuple(module.qweight.shape)}; "
            f"expected {expected_qweight}."
        )
    if tuple(module.qzeros.shape) != expected_qzeros:
        raise ValueError(
            f"{name}.qzeros has shape {tuple(module.qzeros.shape)}; "
            f"expected {expected_qzeros}."
        )
    materialize._unsigned_words(module.qweight, name=f"{name}.qweight")
    materialize._unsigned_words(module.qzeros, name=f"{name}.qzeros")

    g_idx = getattr(module, "g_idx", None)
    if g_idx is not None:
        if not isinstance(g_idx, torch.Tensor) or g_idx.device.type != "cpu":
            raise ValueError(f"{name}.g_idx must be a CPU tensor.")
        expected_g_idx = (
            torch.arange(in_features, dtype=torch.int32) // group_size
        )
        if g_idx.dtype != torch.int32 or not torch.equal(
            g_idx.detach().contiguous(), expected_g_idx
        ):
            raise ValueError(
                f"{name}.g_idx does not describe contiguous natural groups."
            )

    bias = getattr(module, "bias", None)
    if bias is not None:
        if not isinstance(bias, torch.Tensor):
            raise TypeError(f"{name}.bias must be a tensor or None.")
        if bias.device.type != "cpu":
            raise ValueError(
                f"{name}.bias is on {bias.device}; CPU materialisation "
                "requires a CPU-resident checkpoint."
            )
        if tuple(bias.shape) != (out_features,):
            raise ValueError(
                f"{name}.bias has shape {tuple(bias.shape)}; "
                f"expected {(out_features,)}."
            )
        if not torch.isfinite(bias).all():
            raise ValueError(f"{name}.bias contains non-finite values.")

    return bits, group_size, in_features, out_features


@contextmanager
def signed_scale_validator_override() -> Iterator[None]:
    """Temporarily replace only the frozen scale-positivity validator."""

    original = materialize._validate_packed_module
    if original is validate_finite_signed_packed_module:
        raise RecoveryError("signed-scale validator override is already active")
    materialize._validate_packed_module = validate_finite_signed_packed_module
    try:
        yield
    finally:
        materialize._validate_packed_module = original


def build_checkpoint_manifest(root: str | Path) -> dict[str, Any]:
    """Hash every source-checkpoint file without following symbolic links."""

    checkpoint = Path(root).resolve()
    if not checkpoint.is_dir():
        raise RecoveryError(f"packed checkpoint directory is missing: {checkpoint}")

    records: list[dict[str, Any]] = []
    for path in sorted(checkpoint.rglob("*")):
        if path.is_symlink():
            raise RecoveryError(
                f"packed checkpoint may not contain symbolic links: {path}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise RecoveryError(
                f"packed checkpoint contains a non-regular entry: {path}"
            )
        records.append(
            {
                "path": path.relative_to(checkpoint).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    names = {record["path"] for record in records}
    if "config.json" not in names:
        raise RecoveryError("packed checkpoint is missing config.json")
    if not any(
        name.endswith(".safetensors")
        and (name == "model.safetensors" or name.startswith("model-"))
        for name in names
    ):
        raise RecoveryError(
            "packed checkpoint is missing model safetensor payloads"
        )
    if "tokenizer.json" not in names or "tokenizer_config.json" not in names:
        raise RecoveryError(
            "packed checkpoint is missing the frozen fast-tokenizer files"
        )
    if not records:
        raise RecoveryError("packed checkpoint contains no regular files")

    manifest = {
        "schema_version": 1,
        "root": str(checkpoint),
        "file_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "files": records,
    }
    manifest["files_sha256"] = _canonical_sha256(records)
    return manifest


def assert_checkpoint_unchanged(
    expected: Mapping[str, Any],
    checkpoint: str | Path,
) -> dict[str, Any]:
    actual = build_checkpoint_manifest(checkpoint)
    if actual != dict(expected):
        raise RecoveryError(
            "source packed checkpoint changed after recovery intake: "
            f"{actual.get('files_sha256')} != "
            f"{expected.get('files_sha256')}"
        )
    return actual


def _validate_stage(
    path: Path,
    *,
    stage_name: str,
    expected_output: Path,
) -> dict[str, Any]:
    stage = _read_json_object(path, label=f"original {stage_name} stage")
    if stage.get("schema_version") != 1:
        raise RecoveryError(f"original {stage_name} schema is not version 1")
    if stage.get("stage") != stage_name or stage.get("status") != "succeeded":
        raise RecoveryError(
            f"original {stage_name} stage did not succeed exactly"
        )
    output = stage.get("output")
    if not isinstance(output, str) or Path(output).resolve() != expected_output:
        raise RecoveryError(
            f"original {stage_name} output differs from {expected_output}"
        )
    if not expected_output.is_dir():
        raise RecoveryError(
            f"original {stage_name} output directory is missing: "
            f"{expected_output}"
        )
    _finite_nonnegative(
        stage.get("wall_seconds"), label=f"{stage_name}.wall_seconds"
    )
    _finite_nonnegative(
        stage.get("gpu_seconds"), label=f"{stage_name}.gpu_seconds"
    )
    _parse_time(stage.get("started_at"), label=f"{stage_name}.started_at")
    _parse_time(stage.get("ended_at"), label=f"{stage_name}.ended_at")
    return stage


def validate_original_run_contract(
    *,
    plan: Mapping[str, Any],
    run: Mapping[str, Any],
    plan_sha256: str,
    original_run_dir: str | Path,
) -> dict[str, Any]:
    """Fail closed unless the frozen run is eligible for post-processing."""

    run_dir = Path(original_run_dir).resolve()
    expected_run_dir = (
        Path(plan["runtime"]["output_root"]) / run["output_subdir"]
    ).resolve()
    if run_dir != expected_run_dir:
        raise RecoveryError(
            f"original run directory mismatch: {run_dir} != {expected_run_dir}"
        )
    if (run_dir / "result.json").exists():
        raise RecoveryError("original run already has a successful result")
    if (run_dir / "materialize" / "stage.json").exists():
        raise RecoveryError(
            "original run has a materialize success stage and is not eligible"
        )

    run_manifest = _read_json_object(
        run_dir / "run_manifest.json", label="original run manifest"
    )
    launcher_manifest = _read_json_object(
        run_dir / "launcher.json", label="original launcher manifest"
    )
    failure = _read_json_object(
        run_dir / "failure.json", label="original failure"
    )
    for label, value in (
        ("run manifest", run_manifest),
        ("launcher manifest", launcher_manifest),
        ("failure", failure),
    ):
        if value.get("schema_version") != 1:
            raise RecoveryError(f"original {label} schema is not version 1")

    if (
        run_manifest.get("run_id") != run["run_id"]
        or run_manifest.get("run") != dict(run)
        or run_manifest.get("plan_sha256") != plan_sha256
        or run_manifest.get("status") != "running"
    ):
        raise RecoveryError("original run manifest identity/status mismatch")
    if (
        launcher_manifest.get("run_id") != run["run_id"]
        or launcher_manifest.get("plan_sha256") != plan_sha256
    ):
        raise RecoveryError("original launcher identity/plan SHA mismatch")
    if (
        failure.get("status") != "failed"
        or failure.get("error_type") != "ValueError"
        or not isinstance(failure.get("error"), str)
        or _FAILURE_PATTERN.fullmatch(failure["error"]) is None
    ):
        raise RecoveryError(
            "original failure is not the exact learned-scale positivity gate"
        )
    failure_trace = failure.get("traceback")
    if (
        not isinstance(failure_trace, str)
        or "_validate_packed_module" not in failure_trace
        or failure["error"] not in failure_trace
    ):
        raise RecoveryError(
            "original failure traceback does not prove the positivity gate"
        )

    block_output = (run_dir / "block_ap" / "packed_model").resolve()
    e2e_output = (run_dir / "e2e_qp" / "packed_model").resolve()
    block_stage = _validate_stage(
        run_dir / "block_ap" / "stage.json",
        stage_name="block_ap",
        expected_output=block_output,
    )
    e2e_stage = _validate_stage(
        run_dir / "e2e_qp" / "stage.json",
        stage_name="e2e_qp",
        expected_output=e2e_output,
    )

    cfg = plan["method_contract"]["e2e_qp"]
    expected_steps = (
        int(plan["calibration_contract"]["num_samples"])
        // (
            int(cfg["micro_batch_size"])
            * int(cfg["gradient_accumulation_steps"])
        )
        * int(cfg["epochs"])
    )
    if expected_steps != 64 or e2e_stage.get("optimizer_steps") != 64:
        raise RecoveryError(
            "original E2E stage does not prove the frozen 64 optimizer steps"
        )

    run_started = _parse_time(
        run_manifest.get("started_at"), label="run_manifest.started_at"
    )
    block_started = _parse_time(
        block_stage.get("started_at"), label="block_ap.started_at"
    )
    block_ended = _parse_time(
        block_stage.get("ended_at"), label="block_ap.ended_at"
    )
    e2e_started = _parse_time(
        e2e_stage.get("started_at"), label="e2e_qp.started_at"
    )
    e2e_ended = _parse_time(
        e2e_stage.get("ended_at"), label="e2e_qp.ended_at"
    )
    failed_at = _parse_time(failure.get("ended_at"), label="failure.ended_at")
    if not (
        run_started <= block_started <= block_ended
        and block_ended <= e2e_started <= e2e_ended <= failed_at
    ):
        raise RecoveryError("original stage timestamps are not monotonic")

    block_gpu = _finite_nonnegative(
        block_stage["gpu_seconds"], label="block_ap.gpu_seconds"
    )
    e2e_gpu = _finite_nonnegative(
        e2e_stage["gpu_seconds"], label="e2e_qp.gpu_seconds"
    )
    first_attempt_runner_wall = (failed_at - run_started).total_seconds()
    first_attempt_wall = (failed_at - block_started).total_seconds()
    if first_attempt_runner_wall < 0 or first_attempt_wall < 0:
        raise RecoveryError("original first-attempt wall time is negative")
    original_gpu_seconds = block_gpu + e2e_gpu

    return {
        "run_manifest": run_manifest,
        "launcher_manifest": launcher_manifest,
        "failure": failure,
        "stages": {
            "block_ap": block_stage,
            "e2e_qp": e2e_stage,
        },
        "packed_checkpoint": str(e2e_output),
        "timing": {
            "first_attempt_runner_started_at": run_manifest["started_at"],
            "first_attempt_runner_wall_seconds": first_attempt_runner_wall,
            "first_attempt_started_at": block_stage["started_at"],
            "first_attempt_failed_at": failure["ended_at"],
            "first_attempt_wall_seconds": first_attempt_wall,
            "first_attempt_wall_definition": (
                "block_ap.started_at through failure.ended_at, including "
                "inter-stage transitions and the failed positivity gate"
            ),
            "block_ap_wall_seconds": float(block_stage["wall_seconds"]),
            "block_ap_gpu_seconds": block_gpu,
            "e2e_qp_wall_seconds": float(e2e_stage["wall_seconds"]),
            "e2e_qp_gpu_seconds": e2e_gpu,
            "original_quantization_gpu_seconds": original_gpu_seconds,
            "original_quantization_gpu_hours": original_gpu_seconds / 3600.0,
        },
    }


@torch.no_grad()
def collect_scale_statistics(
    packed_checkpoint: str | Path,
    *,
    wbits: int,
    group_size: int,
) -> dict[str, Any]:
    """Load the frozen checkpoint on CPU and audit every packed scale."""

    run_one._bootstrap_efficientqat()
    model, tokenizer = materialize._load_upstream_packed_checkpoint(
        Path(packed_checkpoint),
        wbits=wbits,
        group_size=group_size,
    )
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RecoveryError("scale-audit model is not entirely CPU-resident")
    names = materialize._expected_llama_projection_names(model)
    records: list[dict[str, Any]] = []
    total = negative = zero = positive = nonfinite = 0
    global_min: float | None = None
    global_max: float | None = None
    try:
        for name in names:
            module = _get_module(model, name)
            validate_finite_signed_packed_module(
                module,
                name=name,
                expected_bits=wbits,
                expected_group_size=group_size,
            )
            scales = module.scales.detach().cpu().contiguous()
            finite = torch.isfinite(scales)
            count = scales.numel()
            negative_count = int((scales < 0).sum().item())
            zero_count = int((scales == 0).sum().item())
            positive_count = int((scales > 0).sum().item())
            nonfinite_count = int((~finite).sum().item())
            if nonfinite_count:
                raise RecoveryError(f"{name}.scales contains non-finite values")
            minimum = float(scales.min().item())
            maximum = float(scales.max().item())
            total += count
            negative += negative_count
            zero += zero_count
            positive += positive_count
            nonfinite += nonfinite_count
            global_min = minimum if global_min is None else min(global_min, minimum)
            global_max = maximum if global_max is None else max(global_max, maximum)
            records.append(
                {
                    "name": name,
                    "shape": list(scales.shape),
                    "dtype": str(scales.dtype),
                    "numel": count,
                    "negative_count": negative_count,
                    "zero_count": zero_count,
                    "positive_count": positive_count,
                    "nonfinite_count": nonfinite_count,
                    "minimum": minimum,
                    "maximum": maximum,
                    "scale_sha256": materialize._tensor_sha256(scales),
                }
            )
    finally:
        del model, tokenizer
        gc.collect()

    if total == 0 or positive + zero + negative != total or nonfinite != 0:
        raise RecoveryError("scale statistics are internally inconsistent")
    if negative + zero == 0:
        raise RecoveryError(
            "checkpoint has no signed/zero scale and does not need this recovery"
        )
    summary = {
        "schema_version": 1,
        "policy": SCALE_POLICY,
        "module_count": len(records),
        "numel": total,
        "negative_count": negative,
        "zero_count": zero,
        "positive_count": positive,
        "nonfinite_count": nonfinite,
        "minimum": global_min,
        "maximum": global_max,
        "modules": records,
    }
    summary["modules_sha256"] = _canonical_sha256(records)
    return summary


def _resolve_recovery_dir(
    *,
    recovery_root: str | Path,
    original_output_root: str | Path,
    original_run_dir: str | Path,
    output_subdir: str,
) -> tuple[Path, Path]:
    root = Path(recovery_root).resolve()
    output_root = Path(original_output_root).resolve()
    original = Path(original_run_dir).resolve()

    def _is_within(candidate: Path, parent: Path) -> bool:
        try:
            candidate.relative_to(parent)
        except ValueError:
            return False
        return True

    if _is_within(root, output_root) or _is_within(output_root, root):
        raise RecoveryError(
            "recovery root and formal output root must be disjoint"
        )
    destination = (root / output_subdir / RECOVERY_VARIANT).resolve()
    if not _is_within(destination, root):
        raise RecoveryError("recovery destination escapes the recovery root")
    try:
        destination.relative_to(original)
    except ValueError:
        pass
    else:
        raise RecoveryError(
            "recovery destination may not be inside the original run directory"
        )
    if destination.exists():
        raise RecoveryError(
            f"refusing to overwrite an existing recovery run: {destination}"
        )
    return root, destination


def build_recovery_preflight(args: argparse.Namespace) -> RecoveryPreflight:
    """Validate plan, failed run, packed bytes, scales, and destination."""

    preflight_started_at = _now()
    preflight_started = time.monotonic()
    plan_path = Path(args.plan_file).resolve()
    actual_plan_sha = launcher.plan_sha256(plan_path)
    if actual_plan_sha != args.expected_plan_sha256:
        raise RecoveryError(
            "plan SHA differs from the caller pin: "
            f"{actual_plan_sha} != {args.expected_plan_sha256}"
        )
    plan = launcher.load_plan(plan_path)
    offline_environment = apply_frozen_offline_environment(plan)
    launcher.verify_artifacts(plan)
    launcher.verify_reference_cache_payloads(plan)
    matching = [run for run in plan["runs"] if run["run_id"] == args.run_id]
    if len(matching) != 1:
        raise RecoveryError(f"run_id must resolve exactly once: {args.run_id}")
    run = matching[0]

    physical_eval_gpu = int(args.physical_eval_gpu)
    if physical_eval_gpu != int(run["evaluation_gpu"]):
        raise RecoveryError(
            "physical evaluation GPU differs from the frozen run assignment"
        )
    script_path = Path(__file__).resolve()
    script_sha = _sha256_file(script_path)
    if script_sha != args.expected_recovery_script_sha256:
        raise RecoveryError(
            "recovery script SHA differs from the caller pin: "
            f"{script_sha} != {args.expected_recovery_script_sha256}"
        )

    original_run_dir = (
        Path(plan["runtime"]["output_root"]) / run["output_subdir"]
    ).resolve()
    original = validate_original_run_contract(
        plan=plan,
        run=run,
        plan_sha256=actual_plan_sha,
        original_run_dir=original_run_dir,
    )
    packed_checkpoint = Path(original["packed_checkpoint"]).resolve()
    checkpoint_manifest = build_checkpoint_manifest(packed_checkpoint)
    group_size = int(plan["method_contract"]["weight_group_size"])
    scale_stats = collect_scale_statistics(
        packed_checkpoint,
        wbits=int(run["w_bits"]),
        group_size=group_size,
    )
    assert_checkpoint_unchanged(checkpoint_manifest, packed_checkpoint)

    recovery_root, recovery_dir = _resolve_recovery_dir(
        recovery_root=args.recovery_root,
        original_output_root=plan["runtime"]["output_root"],
        original_run_dir=original_run_dir,
        output_subdir=run["output_subdir"],
    )
    code_contract = plan["code_identity"]["controlled_code_manifest"]
    code_manifest_path = Path(code_contract["path"]).resolve()
    actual_code_manifest_sha = _sha256_file(code_manifest_path)
    if actual_code_manifest_sha != code_contract["sha256"]:
        raise RecoveryError("controlled code-manifest SHA changed")
    code_manifest = _read_json_object(
        code_manifest_path, label="controlled code manifest"
    )
    controlled = code_manifest.get("files_sha256", {})
    if not isinstance(controlled, dict):
        raise RecoveryError("controlled code manifest files must be an object")
    required_controlled = (
        "experiments/efficientqat_compare/materialize.py",
        "experiments/efficientqat_compare/run_one.py",
    )
    if any(name not in controlled for name in required_controlled):
        raise RecoveryError(
            "controlled code manifest lacks materialize/run_one identities"
        )
    workspace = Path(plan["runtime"]["workspace"]).resolve()
    for relative_path in required_controlled:
        actual = _sha256_file(workspace / relative_path)
        if actual != controlled[relative_path]:
            raise RecoveryError(
                f"controlled source changed at recovery intake: {relative_path}"
            )

    preflight_ended_at = _now()
    preflight_wall_seconds = time.monotonic() - preflight_started
    evidence = {
        "schema_version": 1,
        "format": RECOVERY_FORMAT,
        "format_version": RECOVERY_VERSION,
        "variant": RECOVERY_VARIANT,
        "status": "validated",
        "validated_at": _now(),
        "run_id": run["run_id"],
        "run": run,
        "original_run_dir": str(original_run_dir),
        "recovery_root": str(recovery_root),
        "recovery_dir": str(recovery_dir),
        "physical_eval_gpu": physical_eval_gpu,
        "cuda_visible_devices_required_for_execute": str(physical_eval_gpu),
        "scale_policy": SCALE_POLICY,
        "offline_environment": offline_environment,
        "reference_cache_payloads_verified_at": _now(),
        "preflight_timing": {
            "started_at": preflight_started_at,
            "ended_at": preflight_ended_at,
            "wall_seconds": preflight_wall_seconds,
            "gpu_count": 0,
            "gpu_seconds": 0.0,
        },
        "identity": {
            "original_plan_file": str(plan_path),
            "original_plan_sha256": actual_plan_sha,
            "original_controlled_code_manifest": str(code_manifest_path),
            "original_controlled_code_manifest_sha256": actual_code_manifest_sha,
            "original_materialize_sha256": controlled[
                "experiments/efficientqat_compare/materialize.py"
            ],
            "original_run_one_sha256": controlled[
                "experiments/efficientqat_compare/run_one.py"
            ],
            "recovery_script": str(script_path),
            "recovery_script_sha256": script_sha,
        },
        "original": original,
        "packed_checkpoint": checkpoint_manifest,
        "scale_statistics": scale_stats,
    }
    return RecoveryPreflight(
        plan_path=plan_path,
        plan=plan,
        run=run,
        original_run_dir=original_run_dir,
        packed_checkpoint=packed_checkpoint,
        recovery_root=recovery_root,
        recovery_dir=recovery_dir,
        physical_eval_gpu=physical_eval_gpu,
        script_sha256=script_sha,
        checkpoint_manifest=checkpoint_manifest,
        evidence=evidence,
    )


def assert_preflight_inputs_unchanged(
    preflight: RecoveryPreflight,
    *,
    verify_reference_payloads: bool,
    phase: str,
) -> dict[str, Any]:
    """Recheck every frozen input at execution boundaries."""

    actual_plan_sha = launcher.plan_sha256(preflight.plan_path)
    expected_plan_sha = preflight.evidence["identity"][
        "original_plan_sha256"
    ]
    if actual_plan_sha != expected_plan_sha:
        raise RecoveryError(
            f"plan changed before {phase}: "
            f"{actual_plan_sha} != {expected_plan_sha}"
        )
    current_offline = apply_frozen_offline_environment(preflight.plan)
    if current_offline != preflight.evidence["offline_environment"]:
        raise RecoveryError(f"offline environment changed before {phase}")

    # This rehashes the token tensor, model configs, evaluator, and every
    # controlled source against the original formal code manifest.
    launcher.verify_artifacts(preflight.plan)
    if verify_reference_payloads:
        launcher.verify_reference_cache_payloads(preflight.plan)

    script_sha = _sha256_file(Path(__file__).resolve())
    if script_sha != preflight.script_sha256:
        raise RecoveryError(
            f"recovery script changed before {phase}: "
            f"{script_sha} != {preflight.script_sha256}"
        )
    original = validate_original_run_contract(
        plan=preflight.plan,
        run=preflight.run,
        plan_sha256=expected_plan_sha,
        original_run_dir=preflight.original_run_dir,
    )
    if original != preflight.evidence["original"]:
        raise RecoveryError(f"original failed run changed before {phase}")
    checkpoint = assert_checkpoint_unchanged(
        preflight.checkpoint_manifest,
        preflight.packed_checkpoint,
    )

    identity = preflight.evidence["identity"]
    materialize_sha = _sha256_file(Path(materialize.__file__).resolve())
    run_one_sha = _sha256_file(Path(run_one.__file__).resolve())
    if materialize_sha != identity["original_materialize_sha256"]:
        raise RecoveryError(
            f"frozen materialize source changed before {phase}"
        )
    if run_one_sha != identity["original_run_one_sha256"]:
        raise RecoveryError(f"frozen run_one source changed before {phase}")
    return {
        "phase": phase,
        "verified_at": _now(),
        "plan_sha256": actual_plan_sha,
        "recovery_script_sha256": script_sha,
        "materialize_sha256": materialize_sha,
        "run_one_sha256": run_one_sha,
        "checkpoint_files_sha256": checkpoint["files_sha256"],
        "reference_cache_payloads_rehashed": verify_reference_payloads,
        "offline_environment": current_offline,
    }


def dry_validate(preflight: RecoveryPreflight) -> dict[str, Any]:
    """Return read-only validation evidence without creating output files."""

    if preflight.recovery_dir.exists():
        raise RecoveryError(
            f"dry-validation destination unexpectedly exists: "
            f"{preflight.recovery_dir}"
        )
    return {
        **preflight.evidence,
        "status": "succeeded",
        "mode": "dry_validate",
        "executed": False,
        "note": (
            "No recovery directory was created and no materialization, "
            "evaluation, or CUDA operation was started."
        ),
    }


def _assert_execute_runtime(preflight: RecoveryPreflight) -> None:
    expected_host = preflight.plan["runtime"]["canoe_pod"]
    if socket.gethostname() != expected_host:
        raise RecoveryError(
            f"recovery execute is on {socket.gethostname()}, not {expected_host}"
        )
    expected_visible = str(preflight.physical_eval_gpu)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_visible:
        raise RecoveryError(
            "recovery execute requires exactly one visible evaluation GPU: "
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r} != "
            f"{expected_visible!r}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RecoveryError(
            "recovery execute requires exactly one CUDA device at runtime"
        )


def _timing_summary(
    preflight: RecoveryPreflight,
    *,
    recovery_execution_started_at: str,
    recovery_ended_at: str,
    recovery_execution_wall_seconds: float,
    materialize_stage: Mapping[str, Any],
    evaluation_stage: Mapping[str, Any],
) -> dict[str, Any]:
    original = preflight.evidence["original"]["timing"]
    first_wall = float(original["first_attempt_wall_seconds"])
    materialize_wall = float(materialize_stage["wall_seconds"])
    evaluation_wall = float(evaluation_stage["wall_seconds"])
    preflight_timing = preflight.evidence["preflight_timing"]
    preflight_wall = float(preflight_timing["wall_seconds"])
    recovery_wall = preflight_wall + float(recovery_execution_wall_seconds)
    original_gpu_seconds = float(
        original["original_quantization_gpu_seconds"]
    )
    original_start = _parse_time(
        original["first_attempt_started_at"],
        label="first_attempt_started_at",
    )
    recovery_end = _parse_time(
        recovery_ended_at, label="recovery_ended_at"
    )
    return {
        **original,
        "recovery_started_at": preflight_timing["started_at"],
        "recovery_preflight_ended_at": preflight_timing["ended_at"],
        "recovery_preflight_wall_seconds": preflight_wall,
        "recovery_execution_started_at": recovery_execution_started_at,
        "recovery_ended_at": recovery_ended_at,
        "recovery_execution_wall_seconds": float(
            recovery_execution_wall_seconds
        ),
        "recovery_wall_seconds": recovery_wall,
        "recovery_materialize_wall_seconds": materialize_wall,
        "recovery_materialize_gpu_seconds": 0.0,
        "recovery_evaluation_wall_seconds": evaluation_wall,
        "recovery_evaluation_gpu_hours_excluded": True,
        "recovery_quantization_gpu_seconds": 0.0,
        "recovery_quantization_gpu_hours": 0.0,
        # Effective wall is the sum of active attempts.  The idle interval
        # between the first failure and an explicitly launched recovery is
        # disclosed separately through calendar wall time.
        "effective_wall_seconds": first_wall + recovery_wall,
        "effective_wall_definition": (
            "first_attempt_wall_seconds + recovery_wall_seconds; excludes "
            "the idle gap before recovery launch"
        ),
        "effective_stage_wall_seconds": (
            float(original["block_ap_wall_seconds"])
            + float(original["e2e_qp_wall_seconds"])
            + materialize_wall
            + evaluation_wall
        ),
        "calendar_wall_seconds_from_original_start": (
            recovery_end - original_start
        ).total_seconds(),
        "effective_quantization_gpu_seconds": original_gpu_seconds,
        "effective_quantization_gpu_hours": original_gpu_seconds / 3600.0,
        "gpu_hours_definition": (
            "Block-AP plus E2E-QP GPU-seconds from the original succeeded "
            "training stages; CPU materialization and quality evaluation are "
            "excluded by the frozen timing contract."
        ),
    }


def execute_recovery(preflight: RecoveryPreflight) -> dict[str, Any]:
    """Materialize and evaluate one eligible run without retraining it."""

    _assert_execute_runtime(preflight)
    recovery_execution_started_at = _now()
    recovery_execution_started = time.monotonic()
    start_reverification = assert_preflight_inputs_unchanged(
        preflight,
        verify_reference_payloads=True,
        phase="execute_start",
    )
    running_manifest = {
        **preflight.evidence,
        "status": "running",
        "mode": "execute",
        "executed": True,
        "recovery_started_at": preflight.evidence["preflight_timing"][
            "started_at"
        ],
        "recovery_execution_started_at": recovery_execution_started_at,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "eval_device": "cuda:0",
        "execute_reverification": {
            "execute_start": start_reverification,
        },
    }
    manifest_path = preflight.recovery_dir / "recovery_manifest.json"
    directory_created = False
    try:
        preflight.recovery_dir.mkdir(parents=True, exist_ok=False)
        directory_created = True
        _write_json_atomic(manifest_path, running_manifest)
        logger = run_one._configure_logging(preflight.recovery_dir)
        logger.info(
            "SIGNED_SCALE_RECOVERY_V1 START run_id=%s pid=%s eval_gpu=%s "
            "negative_scales=%s zero_scales=%s",
            preflight.run["run_id"],
            os.getpid(),
            preflight.physical_eval_gpu,
            preflight.evidence["scale_statistics"]["negative_count"],
            preflight.evidence["scale_statistics"]["zero_count"],
        )
        before_materialize_source = assert_checkpoint_unchanged(
            preflight.checkpoint_manifest, preflight.packed_checkpoint
        )
        original_validator = materialize._validate_packed_module
        materialize_source = Path(materialize.__file__).resolve()
        materialize_sha_before = _sha256_file(materialize_source)
        logger.info(
            "SIGNED_SCALE_RECOVERY_V1 MATERIALIZE_START source=%s",
            preflight.packed_checkpoint,
        )
        with signed_scale_validator_override():
            materialized_dir, materialize_stage = run_one.run_materialize(
                preflight.plan,
                preflight.run,
                preflight.recovery_dir,
                preflight.packed_checkpoint,
            )
        if materialize._validate_packed_module is not original_validator:
            raise RecoveryError(
                "frozen materialize validator was not restored after recovery"
            )
        materialize_sha_after = _sha256_file(materialize_source)
        if materialize_sha_after != materialize_sha_before:
            raise RecoveryError(
                "frozen materialize source changed during recovery"
            )
        logger.info(
            "SIGNED_SCALE_RECOVERY_V1 MATERIALIZE_SUCCEEDED wall_seconds=%.6f",
            float(materialize_stage["wall_seconds"]),
        )
        after_materialize_source = assert_checkpoint_unchanged(
            preflight.checkpoint_manifest, preflight.packed_checkpoint
        )
        if before_materialize_source != after_materialize_source:
            raise RecoveryError("packed checkpoint changed during materialization")

        before_evaluation_reverification = assert_preflight_inputs_unchanged(
            preflight,
            verify_reference_payloads=True,
            phase="before_evaluation",
        )
        running_manifest["execute_reverification"][
            "before_evaluation"
        ] = before_evaluation_reverification
        _write_json_atomic(manifest_path, running_manifest)
        logger.info(
            "SIGNED_SCALE_RECOVERY_V1 EVALUATION_START physical_eval_gpu=%s",
            preflight.physical_eval_gpu,
        )
        metrics, evaluation_stage = run_one.run_evaluation(
            preflight.plan,
            preflight.run,
            preflight.recovery_dir,
            materialized_dir,
            "cuda:0",
            preflight.physical_eval_gpu,
            logger,
        )
        assert_checkpoint_unchanged(
            preflight.checkpoint_manifest, preflight.packed_checkpoint
        )
        after_evaluation_reverification = assert_preflight_inputs_unchanged(
            preflight,
            verify_reference_payloads=False,
            phase="after_evaluation",
        )
        running_manifest["execute_reverification"][
            "after_evaluation"
        ] = after_evaluation_reverification

        recovery_execution_wall = (
            time.monotonic() - recovery_execution_started
        )
        recovery_ended_at = _now()
        timing = _timing_summary(
            preflight,
            recovery_execution_started_at=recovery_execution_started_at,
            recovery_ended_at=recovery_ended_at,
            recovery_execution_wall_seconds=recovery_execution_wall,
            materialize_stage=materialize_stage,
            evaluation_stage=evaluation_stage,
        )
        result = {
            "schema_version": 1,
            "format": RECOVERY_FORMAT,
            "format_version": RECOVERY_VERSION,
            "variant": RECOVERY_VARIANT,
            "status": "succeeded",
            "run_id": preflight.run["run_id"],
            "run": preflight.run,
            "identity": preflight.evidence["identity"],
            "execute_reverification": running_manifest[
                "execute_reverification"
            ],
            "scale_policy": SCALE_POLICY,
            "scale_statistics": preflight.evidence["scale_statistics"],
            "packed_checkpoint": preflight.checkpoint_manifest,
            "original_training_stages": preflight.evidence["original"]["stages"],
            "original_failure": preflight.evidence["original"]["failure"],
            "recovery_stages": {
                "materialize": materialize_stage,
                "evaluation": evaluation_stage,
            },
            "timing": timing,
            "metrics": metrics,
            "materialized_checkpoint": str(materialized_dir),
            "recovery_started_at": preflight.evidence["preflight_timing"][
                "started_at"
            ],
            "recovery_execution_started_at": recovery_execution_started_at,
            "recovery_ended_at": recovery_ended_at,
        }
        result_path = preflight.recovery_dir / "result.json"
        _write_json_atomic(result_path, result)
        result_sha = _sha256_file(result_path)
        completed_manifest = {
            **running_manifest,
            "status": "succeeded",
            "recovery_ended_at": recovery_ended_at,
            "result": str(result_path),
            "result_sha256": result_sha,
            "materialized_checkpoint": str(materialized_dir),
            "timing": timing,
        }
        _write_json_atomic(manifest_path, completed_manifest)
        logger.info(
            "Signed-scale recovery succeeded: %s",
            json.dumps(metrics, sort_keys=True),
        )
        return result
    except Exception as exc:
        failed_at = _now()
        failure = {
            "schema_version": 1,
            "format": RECOVERY_FORMAT,
            "format_version": RECOVERY_VERSION,
            "status": "failed",
            "run_id": preflight.run["run_id"],
            "failed_at": failed_at,
            "error_type": type(exc).__qualname__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        if directory_created:
            try:
                _write_json_atomic(
                    preflight.recovery_dir / "failure.json", failure
                )
                _write_json_atomic(
                    manifest_path,
                    {
                        **running_manifest,
                        "status": "failed",
                        "recovery_ended_at": failed_at,
                        "failure": str(
                            preflight.recovery_dir / "failure.json"
                        ),
                    },
                )
            except Exception:
                pass
        raise


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    preflight = build_recovery_preflight(args)
    if not args.execute:
        return dry_validate(preflight)
    return execute_recovery(preflight)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-file", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--recovery-root", required=True)
    parser.add_argument("--physical-eval-gpu", type=int, required=True)
    parser.add_argument(
        "--expected-recovery-script-sha256",
        required=True,
        help="SHA256 of this exact recovery script, pinned by the caller.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-validate",
        action="store_true",
        help="Read-only CPU validation (also the default).",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Run materialization and locked single-GPU evaluation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        # EfficientQAT's model loader prints progress messages to stdout.
        # Keep the CLI's stdout machine-readable by routing all dependency
        # chatter to stderr and emitting exactly one JSON document below.
        with redirect_stdout(sys.stderr):
            result = run_from_args(args)
    except Exception:
        traceback.print_exc()
        return 1
    json.dump(
        result,
        sys.stdout,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
