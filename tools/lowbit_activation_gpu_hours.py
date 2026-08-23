#!/usr/bin/env python3
"""Aggregate formal low-bit activation GPU-hours from phase timing sidecars.

This tool is deliberately separate from the source-pinned campaign/result
modules.  It never estimates algorithm time from an execution manifest's
process-wide ``timestamps.duration_seconds``.  A number is admitted only when
the execution manifest pins a sibling phase-timing sidecar by SHA256 and the
sidecar passes all scope, segment, identity, rank, GPU, and producer-link
checks.

``algorithm_core_v1`` is a fail-closed contract.  Quantization timings include
rotation and the quantization core.  Formal shared producers include rotation
and their method-specific precompute core.  Model loading, FP-reference
generation, KL/PPL evaluation, lm-eval, and checkpoint saving are outside the
timed interval.  Multi-rank GPU-hours use synchronized critical-path wall time
(``max(rank elapsed) * allocated GPU count / 3600``).  If a failed attempt
lacks any allocated-rank evidence, that exact formula is forbidden: only the
sum of observed rank intervals is exposed as a lower bound, and the exact
campaign total remains unavailable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCOPE_ID = "algorithm_core_v1"
SIDECAR_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "execution_manifest.json"
SIDECAR_FILENAME = "phase_timing.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_QUANT_STAGES = {
    "realq": "realq_quantization",
    "gptaq": "gptaq_quantization",
    "guided_gptq": "guided_gptq_quantization",
}
_PRECOMPUTE_STAGES = {
    "realq": "realq_static",
    "guided_gptq": "guided_saliency",
}
_INCLUDED_SEGMENTS = {
    ("quantization", "realq"): ["rotation", "quantization_core"],
    ("quantization", "gptaq"): ["rotation", "quantization_core"],
    ("quantization", "guided_gptq"): ["rotation", "quantization_core"],
    (
        "shared_precompute",
        "realq",
    ): ["rotation", "realq_static_precompute_core"],
    (
        "shared_precompute",
        "guided_gptq",
    ): ["rotation", "guided_saliency_precompute_core"],
}
_EXCLUDED_SEGMENTS = [
    "model_load",
    "fp_reference_generation",
    "kl_ppl_evaluation",
    "lm_eval",
    "checkpoint_save",
]
_PRIMARY_SUCCESS_STATUSES = {"succeeded"}
_TERMINAL_ATTEMPT_STATUSES = {
    "failed",
    "terminated",
    "interrupted",
    "launch_failed",
    "wrapper_failed",
}
_FORMAL_TIMING_SOURCE_NAMES = (
    "formal_timing_adapter",
    "formal_timing_sitecustomize",
    "formal_timed_execute",
    "lowbit_activation_campaign",
    "formal_timed_campaign",
    "lowbit_activation_gpu_hours",
    "validate_guided_saliency",
)
_RANK_EVIDENCE_RE = re.compile(r"^phase_timing_rank(?P<rank>\d+)\.json$")


class GPUHourError(ValueError):
    """An accounting input is malformed or cannot be trusted."""


def _json_no_duplicates(path: Path) -> Mapping[str, Any]:
    def object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise GPUHourError(f"{path} contains duplicate key {key!r}")
            value[key] = item
        return value

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=object_hook,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GPUHourError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise GPUHourError(f"{path} must contain a JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        before = path.stat()
        if not path.is_file():
            raise GPUHourError(f"cannot hash non-file {path}")
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        after = path.stat()
    except OSError as exc:
        raise GPUHourError(f"cannot hash {path}: {exc}") from exc
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise GPUHourError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GPUHourError(f"{name} must be an object")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise GPUHourError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: Any, name: str) -> str:
    result = _require_string(value, name)
    if not _SHA256_RE.fullmatch(result):
        raise GPUHourError(f"{name} must be a lowercase SHA256")
    return result


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GPUHourError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise GPUHourError(f"{name} must be a finite non-negative number")
    return result


def _manifest_identity(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Return the immutable execution identity used by the timing sidecar."""

    execution_id = _require_string(
        manifest.get("execution_id"), "manifest.execution_id"
    )
    run_id = _require_string(manifest.get("run_id"), "manifest.run_id")

    plan = _require_mapping(manifest.get("plan"), "manifest.plan")
    plan_sha = _require_sha256(plan.get("sha256"), "manifest.plan.sha256")
    if (
        plan.get("changed_during_execution") is not False
        or plan.get("sha256_at_end") != plan_sha
    ):
        raise GPUHourError("manifest plan identity is not stable")

    model = _require_mapping(manifest.get("model"), "manifest.model")
    if model.get("complete") is not True:
        raise GPUHourError("manifest.model.complete must be true")
    model_sha = _require_sha256(
        model.get("combined_identity_sha256"),
        "manifest.model.combined_identity_sha256",
    )

    numerical = _require_mapping(
        manifest.get("numerical_source_tree"),
        "manifest.numerical_source_tree",
    )
    numerical_sha = _require_sha256(
        numerical.get("combined_sha256"),
        "manifest.numerical_source_tree.combined_sha256",
    )
    if (
        numerical.get("changed_during_execution") is not False
        or numerical.get("combined_sha256_at_end") != numerical_sha
    ):
        raise GPUHourError("manifest numerical source identity is not stable")

    return {
        "execution_id": execution_id,
        "run_id": run_id,
        "plan_sha256": plan_sha,
        "model_sha256": model_sha,
        "numerical_source_sha256": numerical_sha,
    }


def _option(argv: Sequence[str], flag: str) -> str | None:
    values: list[str] = []
    prefix = f"{flag}="
    for index, item in enumerate(argv):
        if item == flag and index + 1 < len(argv):
            values.append(argv[index + 1])
        elif item.startswith(prefix):
            values.append(item[len(prefix) :])
    if len(values) > 1:
        raise GPUHourError(f"argv duplicates {flag}")
    return values[0] if values else None


def _gpu_indices(manifest: Mapping[str, Any]) -> list[int]:
    command = _require_mapping(manifest.get("command"), "manifest.command")
    env = _require_mapping(command.get("env"), "manifest.command.env")
    visible = _require_string(
        env.get("CUDA_VISIBLE_DEVICES"),
        "manifest.command.env.CUDA_VISIBLE_DEVICES",
    )
    try:
        values = [int(item.strip()) for item in visible.split(",")]
    except ValueError as exc:
        raise GPUHourError("CUDA_VISIBLE_DEVICES must contain integer IDs") from exc
    if not values or any(value < 0 for value in values):
        raise GPUHourError("CUDA_VISIBLE_DEVICES must be non-empty and non-negative")
    if len(set(values)) != len(values):
        raise GPUHourError("CUDA_VISIBLE_DEVICES contains duplicate GPU IDs")

    launch_gate = _require_mapping(
        manifest.get("launch_gate"), "manifest.launch_gate"
    )
    gpu_gate = _require_mapping(launch_gate.get("gpu"), "manifest.launch_gate.gpu")
    requested = gpu_gate.get("requested_gpu_indices")
    if (
        not isinstance(requested, list)
        or any(type(value) is not int for value in requested)
        or requested != values
    ):
        raise GPUHourError(
            "launch_gate.requested_gpu_indices disagrees with CUDA_VISIBLE_DEVICES"
        )
    return values


def _world_size(manifest: Mapping[str, Any]) -> int:
    command = _require_mapping(manifest.get("command"), "manifest.command")
    argv_value = command.get("argv")
    if (
        not isinstance(argv_value, list)
        or not argv_value
        or any(not isinstance(item, str) for item in argv_value)
    ):
        raise GPUHourError("manifest.command.argv must be a non-empty string list")
    argv = list(argv_value)
    raw = _option(argv, "--nproc-per-node")
    if raw is None:
        raw = _option(argv, "--nproc_per_node")
    if raw is None:
        return 1
    try:
        result = int(raw)
    except ValueError as exc:
        raise GPUHourError("torchrun world size must be an integer") from exc
    if result <= 0:
        raise GPUHourError("torchrun world size must be positive")
    return result


def _phase_timing_sidecar(
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> tuple[Mapping[str, Any], Path, dict[str, Any]]:
    reference = _require_mapping(
        manifest.get("phase_timing"), "manifest.phase_timing"
    )
    if reference.get("schema_version") != SIDECAR_SCHEMA_VERSION:
        raise GPUHourError(
            "manifest.phase_timing.schema_version must equal "
            f"{SIDECAR_SCHEMA_VERSION}"
        )
    if reference.get("scope_id") != SCOPE_ID:
        raise GPUHourError(
            f"manifest.phase_timing.scope_id must equal {SCOPE_ID!r}"
        )
    relative = _require_string(reference.get("path"), "manifest.phase_timing.path")
    if relative != SIDECAR_FILENAME:
        raise GPUHourError(
            f"phase timing must be the sibling {SIDECAR_FILENAME!r}"
        )
    expected_sha = _require_sha256(
        reference.get("sha256"), "manifest.phase_timing.sha256"
    )
    if reference.get("stable_during_hash") is not True:
        raise GPUHourError("manifest.phase_timing.stable_during_hash must be true")
    sidecar_path = (manifest_path.parent / relative).resolve(strict=False)
    if sidecar_path.parent != manifest_path.parent.resolve(strict=False):
        raise GPUHourError("phase timing sidecar escapes the execution directory")
    actual_sha = _sha256_file(sidecar_path)
    if actual_sha != expected_sha:
        raise GPUHourError(
            "phase timing sidecar SHA256 disagrees with its manifest pin"
        )
    sidecar = _json_no_duplicates(sidecar_path)
    provenance = _validate_timing_provenance(
        reference=reference,
        sidecar=sidecar,
        sidecar_path=sidecar_path,
    )
    return sidecar, sidecar_path, provenance


def _validated_sibling(
    root: Path,
    relative: str,
    *,
    required_parent: str,
) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise GPUHourError("timing provenance path must be relative")
    if (
        len(relative_path.parts) != 2
        or relative_path.parts[0] != required_parent
    ):
        raise GPUHourError(
            f"timing provenance path must be under {required_parent!r}"
        )
    path = (root / relative_path).resolve(strict=False)
    expected_parent = (root / required_parent).resolve(strict=False)
    if path.parent != expected_parent:
        raise GPUHourError("timing provenance path escapes its execution")
    return path


def _validate_timing_provenance(
    *,
    reference: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    sidecar_path: Path,
) -> dict[str, Any]:
    """Rehash every frozen timing source and every rank evidence file."""

    source_set = _require_sha256(
        reference.get("source_set_sha256"),
        "manifest.phase_timing.source_set_sha256",
    )
    provenance = _require_mapping(
        sidecar.get("timing_provenance"),
        "phase_timing.timing_provenance",
    )
    if provenance.get("schema_version") != SIDECAR_SCHEMA_VERSION:
        raise GPUHourError(
            "timing_provenance.schema_version must equal 1"
        )
    if (
        _require_sha256(
            provenance.get("source_set_sha256"),
            "timing_provenance.source_set_sha256",
        )
        != source_set
    ):
        raise GPUHourError(
            "timing provenance source set disagrees with manifest pin"
        )

    sources_value = provenance.get("sources")
    if not isinstance(sources_value, list):
        raise GPUHourError("timing_provenance.sources must be a list")
    normalized_sources: list[dict[str, str]] = []
    original_by_name: dict[str, str] = {}
    for index, value in enumerate(sources_value):
        item = _require_mapping(
            value, f"timing_provenance.sources[{index}]"
        )
        name = _require_string(
            item.get("name"),
            f"timing_provenance.sources[{index}].name",
        )
        original = _require_string(
            item.get("original_path"),
            f"timing_provenance.sources[{index}].original_path",
        )
        if not Path(original).is_absolute():
            raise GPUHourError(
                "timing provenance original_path must be absolute"
            )
        expected_relative = f"phase_timing_sources/{name}.py"
        relative = _require_string(
            item.get("snapshot_path"),
            f"timing_provenance.sources[{index}].snapshot_path",
        )
        if relative != expected_relative:
            raise GPUHourError(
                "timing source snapshot path disagrees with source name"
            )
        digest = _require_sha256(
            item.get("sha256"),
            f"timing_provenance.sources[{index}].sha256",
        )
        snapshot = _validated_sibling(
            sidecar_path.parent,
            relative,
            required_parent="phase_timing_sources",
        )
        if _sha256_file(snapshot) != digest:
            raise GPUHourError(
                f"timing source snapshot SHA256 drifted: {snapshot}"
            )
        normalized_sources.append(
            {"name": name, "sha256": digest}
        )
        if name in original_by_name:
            raise GPUHourError("timing source names must be unique")
        original_by_name[name] = original
    if tuple(item["name"] for item in normalized_sources) != (
        _FORMAL_TIMING_SOURCE_NAMES
    ):
        raise GPUHourError(
            "formal timing source set names/order are incomplete or drifted"
        )
    if _canonical_sha256(normalized_sources) != source_set:
        raise GPUHourError(
            "timing source_set_sha256 disagrees with frozen snapshots"
        )
    instrumentation = _require_mapping(
        sidecar.get("instrumentation"),
        "phase_timing.instrumentation",
    )
    if (
        instrumentation.get("path")
        != original_by_name["formal_timing_sitecustomize"]
        or _require_sha256(
            instrumentation.get("sha256"),
            "phase_timing.instrumentation.sha256",
        )
        != {
            item["name"]: item["sha256"]
            for item in normalized_sources
        }["formal_timing_sitecustomize"]
    ):
        raise GPUHourError(
            "sidecar instrumentation identity disagrees with source snapshot"
        )

    evidence_value = provenance.get("rank_evidence")
    if not isinstance(evidence_value, list) or not evidence_value:
        raise GPUHourError(
            "timing_provenance.rank_evidence must be a non-empty list"
        )
    evidence: list[dict[str, Any]] = []
    seen_ranks: set[int] = set()
    for index, value in enumerate(evidence_value):
        item = _require_mapping(
            value, f"timing_provenance.rank_evidence[{index}]"
        )
        relative = _require_string(
            item.get("path"),
            f"timing_provenance.rank_evidence[{index}].path",
        )
        match = _RANK_EVIDENCE_RE.fullmatch(relative)
        if match is None or Path(relative).name != relative:
            raise GPUHourError("rank evidence path is invalid")
        rank = int(match.group("rank"))
        if rank in seen_ranks:
            raise GPUHourError("rank evidence contains duplicate ranks")
        seen_ranks.add(rank)
        digest = _require_sha256(
            item.get("sha256"),
            f"timing_provenance.rank_evidence[{index}].sha256",
        )
        path = (sidecar_path.parent / relative).resolve(strict=False)
        if path.parent != sidecar_path.parent.resolve(strict=False):
            raise GPUHourError("rank evidence escapes execution directory")
        if _sha256_file(path) != digest:
            raise GPUHourError(
                f"rank evidence SHA256 drifted: {path}"
            )
        evidence.append(
            {
                "rank": rank,
                "path": str(path),
                "sha256": digest,
                "payload": _json_no_duplicates(path),
            }
        )
    if [item["rank"] for item in evidence] != sorted(seen_ranks):
        raise GPUHourError("rank evidence rows must be sorted by rank")
    return {
        "strict": True,
        "source_set_sha256": source_set,
        "sources": normalized_sources,
        "rank_evidence": evidence,
    }


def _artifact_reference(value: Any, name: str) -> dict[str, str]:
    item = _require_mapping(value, name)
    return {
        "path": _require_string(item.get("path"), f"{name}.path"),
        "sha256": _require_sha256(item.get("sha256"), f"{name}.sha256"),
    }


def _producer_reference(value: Any, name: str) -> dict[str, str]:
    item = _require_mapping(value, name)
    return {
        "component_id": _require_sha256(
            item.get("component_id"), f"{name}.component_id"
        ),
        "producer_execution_id": _require_string(
            item.get("producer_execution_id"),
            f"{name}.producer_execution_id",
        ),
        "cache_identity_sha256": _require_sha256(
            item.get("cache_identity_sha256"),
            f"{name}.cache_identity_sha256",
        ),
        "artifact_set_sha256": _require_sha256(
            item.get("artifact_set_sha256"),
            f"{name}.artifact_set_sha256",
        ),
    }


def component_identity_sha256(
    *,
    scope_id: str,
    identity: Mapping[str, str],
    component: Mapping[str, Any],
) -> str:
    """Compute the frozen v1 component identity.

    Instrumentation should call the equivalent canonical-JSON hash after all
    producer references/artifact identities are known.  Timing values are
    deliberately excluded so the identifier names *what* ran; the manifest
    SHA256 pin protects the recorded elapsed values.
    """

    payload = {
        "scope_id": scope_id,
        "execution_id": identity["execution_id"],
        "run_id": identity["run_id"],
        "plan_sha256": identity["plan_sha256"],
        "model_sha256": identity["model_sha256"],
        "numerical_source_sha256": identity["numerical_source_sha256"],
        "kind": component.get("kind"),
        "stage": component.get("stage"),
        "method": component.get("method"),
        "model": component.get("model"),
        "setting": component.get("setting"),
        "phase": component.get("phase"),
        "target_phase": component.get("target_phase"),
        "included_segments": component.get("included_segments"),
        "excluded_segments": component.get("excluded_segments"),
        "shared_precompute_refs": component.get("shared_precompute_refs", []),
        "cache_identity_sha256": component.get("cache_identity_sha256"),
        "artifact_set_sha256": component.get("artifact_set_sha256"),
    }
    return _canonical_sha256(payload)


def _parse_component(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    *,
    primary: bool = True,
) -> dict[str, Any]:
    manifest_status = _require_string(
        manifest.get("status"), "manifest.status"
    )
    exit_code = manifest.get("exit_code")
    if primary and (
        manifest_status not in _PRIMARY_SUCCESS_STATUSES
        or type(exit_code) is not int
        or exit_code != 0
    ):
        raise GPUHourError("primary timed manifest must succeed with exit_code=0")
    if not primary and (
        (
            manifest_status == "succeeded"
            and (type(exit_code) is not int or exit_code != 0)
        )
        or (
            manifest_status != "succeeded"
            and (type(exit_code) is not int or exit_code == 0)
        )
    ):
        raise GPUHourError("attempt manifest status/exit_code are inconsistent")

    identity = _manifest_identity(manifest)
    execution_id = identity["execution_id"]
    sidecar, sidecar_path, provenance = _phase_timing_sidecar(
        manifest, manifest_path
    )
    if sidecar.get("schema_version") != SIDECAR_SCHEMA_VERSION:
        raise GPUHourError(
            f"phase timing schema_version must equal {SIDECAR_SCHEMA_VERSION}"
        )
    if sidecar.get("scope_id") != SCOPE_ID:
        raise GPUHourError(f"phase timing scope_id must equal {SCOPE_ID!r}")
    sidecar_identity = _require_mapping(
        sidecar.get("identity"), "phase_timing.identity"
    )
    if dict(sidecar_identity) != identity:
        raise GPUHourError("phase timing identity/hashes disagree with manifest")
    if sidecar.get("manifest_status") != manifest_status:
        raise GPUHourError("phase timing manifest_status disagrees with manifest")
    component = _require_mapping(sidecar.get("component"), "phase_timing.component")

    component_id = _require_sha256(
        component.get("component_id"), "component.component_id"
    )
    kind = _require_string(component.get("kind"), "component.kind")
    if kind not in {"quantization", "shared_precompute"}:
        raise GPUHourError("component.kind is unsupported")
    stage = _require_string(component.get("stage"), "component.stage")
    method = _require_string(component.get("method"), "component.method")
    model = _require_string(component.get("model"), "component.model")
    setting_value = component.get("setting")
    if setting_value is not None and (
        not isinstance(setting_value, str) or not setting_value
    ):
        raise GPUHourError("component.setting must be null or non-empty string")

    included_segments = component.get("included_segments")
    excluded_segments = component.get("excluded_segments")
    expected_included = _INCLUDED_SEGMENTS.get((kind, method))
    if included_segments != expected_included:
        raise GPUHourError(
            "component.included_segments does not match algorithm_core_v1"
        )
    if excluded_segments != _EXCLUDED_SEGMENTS:
        raise GPUHourError(
            "component.excluded_segments does not match algorithm_core_v1"
        )

    if component.get("clock") != "time.perf_counter_ns":
        raise GPUHourError("component.clock must equal 'time.perf_counter_ns'")
    timing_status = component.get("timing_status", "complete")
    if timing_status not in {"complete", "partial", "not_started"}:
        raise GPUHourError("component.timing_status is invalid")
    recorded_complete = component.get("complete")
    if recorded_complete is not None and recorded_complete is not (
        timing_status == "complete"
    ):
        raise GPUHourError(
            "component.complete disagrees with timing_status"
        )
    if primary and timing_status != "complete":
        raise GPUHourError(
            "primary timing component must be complete"
        )
    if timing_status == "complete":
        if (
            component.get("synchronized_start") is not True
            or component.get("synchronized_end") is not True
        ):
            raise GPUHourError(
                "complete component timing boundaries must be synchronized"
            )
    elif (
        type(component.get("synchronized_start")) is not bool
        or component.get("synchronized_end") is not False
    ):
        raise GPUHourError(
            "partial component must use boolean synchronized_start and "
            "synchronized_end=false"
        )

    gpu_indices = component.get("gpu_indices")
    if (
        not isinstance(gpu_indices, list)
        or any(type(value) is not int for value in gpu_indices)
        or len(set(gpu_indices)) != len(gpu_indices)
        or any(value < 0 for value in gpu_indices)
    ):
        raise GPUHourError("component.gpu_indices must be unique non-negative IDs")
    manifest_gpus = _gpu_indices(manifest)
    if gpu_indices != manifest_gpus:
        raise GPUHourError("component.gpu_indices disagrees with manifest GPUs")
    gpu_count = component.get("gpu_count")
    if type(gpu_count) is not int or gpu_count != len(gpu_indices):
        raise GPUHourError("component.gpu_count disagrees with gpu_indices")
    if _world_size(manifest) != gpu_count:
        raise GPUHourError("component GPU count disagrees with torchrun world size")

    timings = component.get("rank_timings")
    if (
        not isinstance(timings, list)
        or not timings
        or len(timings) > gpu_count
        or (
            timing_status == "complete"
            and len(timings) != gpu_count
        )
    ):
        raise GPUHourError(
            "component.rank_timings has invalid rank coverage"
        )
    observed_rank_count = component.get("observed_rank_count")
    if observed_rank_count is not None and (
        type(observed_rank_count) is not int
        or observed_rank_count != len(timings)
    ):
        raise GPUHourError(
            "component.observed_rank_count disagrees with rank_timings"
        )
    elapsed_by_rank: dict[int, float] = {}
    normalized_timings: list[dict[str, Any]] = []
    for index, value in enumerate(timings):
        item = _require_mapping(value, f"component.rank_timings[{index}]")
        rank = item.get("rank")
        start = item.get("started_monotonic_ns")
        end = item.get("ended_monotonic_ns")
        if (
            type(rank) is not int
            or rank < 0
            or rank >= gpu_count
            or rank in elapsed_by_rank
        ):
            raise GPUHourError("rank_timings ranks must be unique non-negative ints")
        row_status = item.get(
            "timing_status",
            "complete" if timing_status == "complete" else None,
        )
        if row_status not in {"complete", "partial", "not_started"}:
            raise GPUHourError("rank timing_status is invalid")
        elapsed = _finite_nonnegative(
            item.get("elapsed_seconds"),
            f"component.rank_timings[{index}].elapsed_seconds",
        )
        if row_status == "not_started":
            if start is not None or end is not None or elapsed != 0.0:
                raise GPUHourError(
                    "not_started rank must have null bounds and zero elapsed"
                )
        else:
            if type(start) is not int or type(end) is not int or end < start:
                raise GPUHourError("rank timing monotonic bounds are invalid")
            derived = (end - start) / 1_000_000_000
            if not math.isclose(
                elapsed, derived, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise GPUHourError(
                    "rank elapsed_seconds disagrees with monotonic bounds"
                )
        if timing_status == "complete" and row_status != "complete":
            raise GPUHourError(
                "complete component contains an incomplete rank"
            )
        if timing_status == "not_started" and row_status != "not_started":
            raise GPUHourError(
                "not_started component contains a started rank"
            )
        elapsed_by_rank[rank] = elapsed
        normalized_timings.append(
            {
                "rank": rank,
                "timing_status": row_status,
                "started_monotonic_ns": start,
                "ended_monotonic_ns": end,
                "elapsed_seconds": elapsed,
            }
        )
    if normalized_timings != sorted(
        normalized_timings, key=lambda item: item["rank"]
    ):
        raise GPUHourError("rank_timings must be sorted by rank")
    expected_ranks = set(range(gpu_count))
    observed_ranks = set(elapsed_by_rank)
    if timing_status == "complete" and observed_ranks != expected_ranks:
        raise GPUHourError("rank_timings must cover ranks 0..gpu_count-1")
    rank_coverage_complete = observed_ranks == expected_ranks
    if component.get("rank_coverage_complete") is not (
        rank_coverage_complete
    ):
        raise GPUHourError(
            "component.rank_coverage_complete disagrees with rank timings"
        )
    missing_ranks = component.get("missing_ranks")
    expected_missing = sorted(expected_ranks - observed_ranks)
    if missing_ranks != expected_missing:
        raise GPUHourError(
            "component.missing_ranks disagrees with rank timings"
        )

    wall_seconds = _finite_nonnegative(
        component.get("wall_seconds"), "component.wall_seconds"
    )
    if not math.isclose(
        wall_seconds,
        max(elapsed_by_rank.values()),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise GPUHourError("wall_seconds must equal max rank elapsed_seconds")
    statuses = {
        item["timing_status"] for item in normalized_timings
    }
    gpu_hour_exact = (
        timing_status == "complete"
        or (
            rank_coverage_complete
            and (
                statuses <= {"partial", "complete"}
                or statuses == {"not_started"}
            )
        )
    )
    expected_gpu_hour_status = (
        "exact" if gpu_hour_exact else "lower_bound"
    )
    if component.get("gpu_hour_status") != expected_gpu_hour_status:
        raise GPUHourError(
            "component.gpu_hour_status disagrees with rank coverage"
        )
    exact_gpu_hours = wall_seconds * gpu_count / 3600.0
    recorded_gpu_hours_value = component.get("allocated_gpu_hours")
    if gpu_hour_exact:
        recorded_gpu_hours = _finite_nonnegative(
            recorded_gpu_hours_value,
            "component.allocated_gpu_hours",
        )
        if not math.isclose(
            exact_gpu_hours,
            recorded_gpu_hours,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise GPUHourError(
                "allocated_gpu_hours must equal "
                "wall_seconds*gpu_count/3600"
            )
    else:
        if recorded_gpu_hours_value is not None:
            raise GPUHourError(
                "incomplete rank coverage cannot claim exact "
                "allocated_gpu_hours"
            )
        recorded_gpu_hours = None
    lower_bound = _finite_nonnegative(
        component.get("gpu_hours_lower_bound"),
        "component.gpu_hours_lower_bound",
    )
    expected_lower_bound = (
        exact_gpu_hours
        if gpu_hour_exact
        else sum(elapsed_by_rank.values()) / 3600.0
    )
    if not math.isclose(
        lower_bound,
        expected_lower_bound,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise GPUHourError(
            "gpu_hours_lower_bound disagrees with strict rank evidence"
        )

    references_value = component.get("shared_precompute_refs", [])
    if not isinstance(references_value, list):
        raise GPUHourError("component.shared_precompute_refs must be a list")
    references = [
        _producer_reference(value, f"shared_precompute_refs[{index}]")
        for index, value in enumerate(references_value)
    ]
    artifacts_value = component.get("artifacts", [])
    if not isinstance(artifacts_value, list):
        raise GPUHourError("component.artifacts must be a list")
    artifacts = [
        _artifact_reference(value, f"artifacts[{index}]")
        for index, value in enumerate(artifacts_value)
    ]
    if len({item["path"] for item in artifacts}) != len(artifacts):
        raise GPUHourError("component.artifacts contains duplicate paths")
    if artifacts != sorted(artifacts, key=lambda item: item["path"]):
        raise GPUHourError("component.artifacts must be sorted by path")

    result = {
        "component_id": component_id,
        "scope_id": SCOPE_ID,
        "kind": kind,
        "stage": stage,
        "method": method,
        "model": model,
        "setting": setting_value,
        "phase": component.get("phase"),
        "target_phase": component.get("target_phase"),
        "execution_id": execution_id,
        "run_id": identity["run_id"],
        "manifest_status": manifest_status,
        "identity": dict(identity),
        "manifest": str(manifest_path.resolve()),
        "sidecar": str(sidecar_path),
        "sidecar_sha256": _sha256_file(sidecar_path),
        "included_segments": list(included_segments),
        "excluded_segments": list(excluded_segments),
        "timing_status": timing_status,
        "complete": timing_status == "complete",
        "synchronized_start": component.get("synchronized_start"),
        "synchronized_end": component.get("synchronized_end"),
        "rank_timings": normalized_timings,
        "observed_rank_count": len(normalized_timings),
        "rank_coverage_complete": rank_coverage_complete,
        "missing_ranks": expected_missing,
        "wall_seconds": wall_seconds,
        "wall_hours": wall_seconds / 3600.0,
        "gpu_indices": list(gpu_indices),
        "gpu_count": gpu_count,
        "gpu_hour_status": expected_gpu_hour_status,
        "allocated_gpu_hours": recorded_gpu_hours,
        "gpu_hours_lower_bound": lower_bound,
        "shared_precompute_refs": references,
        "cache_outcome": component.get("cache_outcome"),
        "cache_identity_sha256": component.get("cache_identity_sha256"),
        "artifact_set_sha256": component.get("artifact_set_sha256"),
        "artifacts": artifacts,
        "timing_provenance_strict": True,
        "timing_source_set_sha256": provenance[
            "source_set_sha256"
        ],
    }
    _validate_rank_evidence_against_component(
        provenance=provenance,
        identity=identity,
        component=result,
    )
    _validate_component_semantics(result, manifest, primary=primary)
    expected_component_id = component_identity_sha256(
        scope_id=SCOPE_ID,
        identity=identity,
        component=result,
    )
    if component_id != expected_component_id:
        raise GPUHourError(
            "component.component_id disagrees with its canonical identity"
        )
    return result


def _validate_rank_evidence_against_component(
    *,
    provenance: Mapping[str, Any],
    identity: Mapping[str, str],
    component: Mapping[str, Any],
) -> None:
    """Bind strict rank files to the exact normalized component rows."""

    evidence = provenance["rank_evidence"]
    timings = component["rank_timings"]
    if [item["rank"] for item in evidence] != [
        item["rank"] for item in timings
    ]:
        raise GPUHourError(
            "rank evidence coverage disagrees with component.rank_timings"
        )
    source_sha = {
        item["name"]: item["sha256"]
        for item in provenance["sources"]
    }
    mode = {
        ("shared_precompute", "realq"): "realq_shared_precompute",
        ("shared_precompute", "guided_gptq"): "guided_precompute",
        ("quantization", "realq"): "realq_quantization",
        ("quantization", "gptaq"): "legacy_quantization",
        ("quantization", "guided_gptq"): "legacy_quantization",
    }.get((component["kind"], component["method"]))
    if mode is None:
        raise GPUHourError("cannot derive timing mode for rank evidence")
    spec_payload = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "scope_id": SCOPE_ID,
        "mode": mode,
        "kind": component["kind"],
        "method": component["method"],
        "stage": component["stage"],
        "model": component["model"],
        "setting": component["setting"],
        "phase": component["phase"],
        "target_phase": component["target_phase"],
        "run_id": component["run_id"],
        "gpu_indices": component["gpu_indices"],
    }
    expected_common = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "scope_id": SCOPE_ID,
        "source_set_sha256": provenance["source_set_sha256"],
        "sitecustomize_sha256": source_sha[
            "formal_timing_sitecustomize"
        ],
        "spec_sha256": _canonical_sha256(spec_payload),
        "run_id": component["run_id"],
        "mode": mode,
        "method": component["method"],
        "stage": component["stage"],
        "model": component["model"],
        "setting": component["setting"],
        "phase": component["phase"],
        "target_phase": component["target_phase"],
        "world_size": component["gpu_count"],
        "clock": "time.perf_counter_ns",
    }
    timing_by_rank = {item["rank"]: item for item in timings}
    for evidence_item in evidence:
        rank = evidence_item["rank"]
        payload = evidence_item["payload"]
        expected = {
            **expected_common,
            "rank": rank,
            "local_rank": rank,
            "physical_gpu_index": component["gpu_indices"][rank],
        }
        drift = {
            key: {"expected": value, "actual": payload.get(key)}
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if drift:
            raise GPUHourError(
                f"rank {rank} evidence identity drifted: {drift}"
            )
        row = timing_by_rank[rank]
        row_status = row["timing_status"]
        if (
            payload.get("timing_status") != row_status
            or payload.get("complete") is not (
                row_status == "complete"
            )
            or payload.get("started_monotonic_ns")
            != row["started_monotonic_ns"]
            or payload.get("ended_monotonic_ns")
            != row["ended_monotonic_ns"]
        ):
            raise GPUHourError(
                f"rank {rank} evidence interval disagrees with component"
            )
        if row_status == "not_started":
            if (
                payload.get("elapsed_seconds") is not None
                or payload.get("synchronized_start") is not False
                or payload.get("synchronized_end") is not False
                or not isinstance(payload.get("error"), str)
                or not payload["error"]
            ):
                raise GPUHourError(
                    f"rank {rank} not_started evidence is malformed"
                )
        else:
            elapsed = _finite_nonnegative(
                payload.get("elapsed_seconds"),
                f"rank evidence {rank}.elapsed_seconds",
            )
            if not math.isclose(
                elapsed,
                row["elapsed_seconds"],
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise GPUHourError(
                    f"rank {rank} evidence elapsed disagrees with component"
                )
            if payload.get("synchronized_start") is not True:
                raise GPUHourError(
                    f"rank {rank} started evidence is unsynchronized"
                )
            if row_status == "complete":
                if (
                    payload.get("synchronized_end") is not True
                    or payload.get("error") is not None
                ):
                    raise GPUHourError(
                        f"rank {rank} complete evidence is malformed"
                    )
            elif (
                payload.get("synchronized_end") is not False
                or not isinstance(payload.get("error"), str)
                or not payload["error"]
            ):
                raise GPUHourError(
                    f"rank {rank} partial evidence is malformed"
                )
        for field in ("cache_lookups", "cache_writes", "artifacts"):
            if not isinstance(payload.get(field), list):
                raise GPUHourError(
                    f"rank {rank} evidence {field} must be a list"
                )


def _validate_component_semantics(
    component: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    primary: bool,
) -> None:
    kind = component["kind"]
    method = component["method"]
    stage = component["stage"]
    setting = component["setting"]
    if kind == "quantization":
        if method not in _QUANT_STAGES or stage != _QUANT_STAGES[method]:
            raise GPUHourError("quantization component method/stage mismatch")
        allowed_phases = {"final"} if primary else {"tune", "final"}
        if (
            component.get("phase") not in allowed_phases
            or component.get("target_phase") is not None
        ):
            raise GPUHourError(
                "quantization component has an invalid accounting phase"
            )
        if not isinstance(setting, str):
            raise GPUHourError("quantization component requires a setting")
        if component.get("cache_outcome") is not None or component["artifacts"]:
            raise GPUHourError("quantization component cannot claim producer artifacts")
        command = _require_mapping(manifest.get("command"), "manifest.command")
        argv = command.get("argv")
        if not isinstance(argv, list) or any(
            not isinstance(item, str) for item in argv
        ):
            raise GPUHourError("manifest.command.argv must be a string list")
        run_id = _require_string(manifest.get("run_id"), "manifest.run_id")
        expected_prefix = (
            f"{component['phase']}_{method}_{component['model']}_"
            f"{setting.lower()}"
        )
        if not (
            run_id == expected_prefix
            or run_id.startswith(f"{expected_prefix}_")
        ):
            raise GPUHourError(
                "quantization run_id disagrees with component identity"
            )
        if _option(argv, "--exp") != run_id:
            raise GPUHourError("manifest --exp disagrees with run_id")
        if method == "realq":
            if "realq.ptq" not in argv:
                raise GPUHourError("REAL-Q timing must execute realq.ptq")
        else:
            expected_w_method = (
                "gptaq" if method == "gptaq" else "gptq_guided"
            )
            if _option(argv, "--w_method") != expected_w_method:
                raise GPUHourError(
                    "baseline --w_method disagrees with timed method"
                )
            if component["phase"] != "final":
                raise GPUHourError("baseline attempt timing must be final")
        return

    if method not in _PRECOMPUTE_STAGES or stage != _PRECOMPUTE_STAGES[method]:
        raise GPUHourError("shared precompute component method/stage mismatch")
    if setting is not None or component.get("phase") is not None:
        raise GPUHourError("shared precompute must not claim a setting/phase")
    allowed_targets = {"final"} if primary else {"tune", "final"}
    if component.get("target_phase") not in allowed_targets:
        raise GPUHourError("shared precompute target_phase is invalid")
    cache_outcome = component.get("cache_outcome")
    artifacts = component["artifacts"]
    if primary:
        if cache_outcome != "computed":
            raise GPUHourError(
                "shared precompute must be a cold computed producer"
            )
        _require_sha256(
            component.get("cache_identity_sha256"),
            "component.cache_identity_sha256",
        )
        artifact_set_sha = _require_sha256(
            component.get("artifact_set_sha256"),
            "component.artifact_set_sha256",
        )
        if not artifacts:
            raise GPUHourError(
                "shared precompute must pin at least one artifact"
            )
        if artifact_set_sha != _canonical_sha256(artifacts):
            raise GPUHourError(
                "component.artifact_set_sha256 disagrees with artifacts"
            )
    else:
        if cache_outcome not in {"computed", "partial", "failed"}:
            raise GPUHourError(
                "attempt precompute cache_outcome must be computed/partial/failed"
            )
        cache_identity = component.get("cache_identity_sha256")
        artifact_set_sha = component.get("artifact_set_sha256")
        if cache_identity is not None:
            _require_sha256(
                cache_identity, "component.cache_identity_sha256"
            )
        if artifacts:
            artifact_set_sha = _require_sha256(
                artifact_set_sha, "component.artifact_set_sha256"
            )
            if artifact_set_sha != _canonical_sha256(artifacts):
                raise GPUHourError(
                    "component.artifact_set_sha256 disagrees with artifacts"
                )
        elif artifact_set_sha is not None:
            raise GPUHourError(
                "attempt without artifacts must use null artifact_set_sha256"
            )
    if component["shared_precompute_refs"]:
        raise GPUHourError("shared precompute cannot reference another producer")

    run_id = _require_string(manifest.get("run_id"), "manifest.run_id")
    command = _require_mapping(manifest.get("command"), "manifest.command")
    argv = command.get("argv")
    if not isinstance(argv, list) or any(not isinstance(item, str) for item in argv):
        raise GPUHourError("manifest.command.argv must be a string list")
    model = component["model"]
    if method == "realq":
        target_phase = component["target_phase"]
        if not run_id.startswith(
            f"precompute_realq_static_{target_phase}_{model}"
        ):
            raise GPUHourError("REAL-Q producer run_id/target_phase is invalid")
        if _option(argv, "--exit_after_precompute") != "true":
            raise GPUHourError("REAL-Q producer must exit after precompute")
    else:
        guided_base = f"precompute_guided_saliency_{model}"
        if not (
            run_id == guided_base
            or re.fullmatch(
                rf"{re.escape(guided_base)}_attempt(?:[2-9]|[1-9][0-9]+)",
                run_id,
            )
        ):
            raise GPUHourError("GuidedGPTQ producer run_id is invalid")
        if not any(Path(item).name == "save_grads.py" for item in argv):
            raise GPUHourError("GuidedGPTQ producer must execute save_grads.py")
    if _option(argv, "--exp") != run_id:
        raise GPUHourError("producer --exp disagrees with run_id")


def _record_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        _require_string(record.get("model"), "record.model"),
        _require_string(record.get("setting"), "record.setting"),
        _require_string(record.get("method"), "record.method"),
    )


def _validate_record_identity(
    record: Mapping[str, Any],
    component: Mapping[str, Any],
) -> None:
    if record.get("execution_id") != component["execution_id"]:
        raise GPUHourError(
            "phase timing execution_id disagrees with accepted result row"
        )
    if record.get("run_id") != component["run_id"]:
        raise GPUHourError(
            "phase timing run_id disagrees with accepted result row"
        )
    record_identities = _require_mapping(
        record.get("identities"), "accepted record.identities"
    )
    timed_identity = component["identity"]
    expected = {
        "plan_sha256": timed_identity["plan_sha256"],
        "model_sha256": timed_identity["model_sha256"],
        "numerical_source_sha256": timed_identity[
            "numerical_source_sha256"
        ],
    }
    for name, expected_value in expected.items():
        if record_identities.get(name) != expected_value:
            raise GPUHourError(
                f"accepted result {name} disagrees with phase timing"
            )


def _incomplete_row(
    record: Mapping[str, Any],
    issue: str,
) -> dict[str, Any]:
    model, setting, method = _record_key(record)
    return {
        "row_type": "quantization",
        "model": model,
        "setting": setting,
        "method": method,
        "execution_id": record.get("execution_id"),
        "manifest": record.get("manifest"),
        "applicable": method != "bf16",
        "status": "timing_incomplete",
        "issues": [issue],
        "exclusive_wall_seconds": None,
        "gpu_indices": None,
        "gpu_count": None,
        "exclusive_gpu_hours": None,
        "primary_gpu_hours": None,
        "shared_precompute_gpu_hours": None,
        "standalone_inclusive_gpu_hours": None,
        "shared_precompute_ids": [],
    }


def _attempt_inputs(
    result_report: Mapping[str, Any],
    *,
    primary_manifests: set[str],
) -> list[dict[str, Any]]:
    """Collect terminal non-primary attempts with deterministic precedence."""

    candidates: dict[str, tuple[int, dict[str, Any]]] = {}

    def add(
        value: Mapping[str, Any],
        *,
        source: str,
        classification: str,
        priority: int,
    ) -> None:
        run_id = value.get("run_id")
        if (
            value.get("counts_toward_tune_budget") is True
            or value.get("phase") == "tune"
            or (
                isinstance(run_id, str)
                and (
                    run_id.startswith("tune_")
                    or run_id.startswith(
                        "precompute_realq_static_tune_"
                    )
                )
            )
        ):
            # Formal GPU-hour accounting excludes tuning spend.  A separate
            # campaign-cost report may aggregate tune sidecars, but old tune
            # attempts must never make the formal exact total incomplete.
            return
        manifest = value.get("manifest")
        if not isinstance(manifest, str) or not manifest:
            return
        canonical = str(Path(manifest).resolve(strict=False))
        if canonical in primary_manifests:
            return
        row = {
            "manifest": canonical,
            "source": source,
            "classification": classification,
            "reported_status": value.get("status"),
            "reported_execution_id": value.get("execution_id"),
            "reported_model": value.get("model") or value.get("tune_model"),
            "reported_setting": (
                value.get("setting") or value.get("tune_setting")
            ),
            "reported_phase": value.get("phase"),
        }
        previous = candidates.get(canonical)
        if previous is None or priority > previous[0]:
            candidates[canonical] = (priority, row)

    retired = result_report.get("retired_attempts", [])
    if not isinstance(retired, list):
        raise GPUHourError("result report retired_attempts must be a list")
    for index, value in enumerate(retired):
        item = _require_mapping(value, f"retired_attempts[{index}]")
        status = _require_string(
            item.get("status"), f"retired_attempts[{index}].status"
        )
        add(
            item,
            source="retirement_ledger",
            classification=f"retired_{status}",
            priority=3,
        )

    informational = result_report.get("informational", [])
    if not isinstance(informational, list):
        raise GPUHourError("result report informational must be a list")
    for index, value in enumerate(informational):
        item = _require_mapping(value, f"informational[{index}]")
        status = item.get("status")
        if status not in _TERMINAL_ATTEMPT_STATUSES:
            continue
        add(
            item,
            source="informational",
            classification=str(status),
            priority=2,
        )

    rejected = result_report.get("rejected", [])
    if not isinstance(rejected, list):
        raise GPUHourError("result report rejected must be a list")
    for index, value in enumerate(rejected):
        item = _require_mapping(value, f"rejected[{index}]")
        manifest_value = item.get("manifest")
        status = "unknown"
        execution_id: Any = None
        run_id: Any = None
        if isinstance(manifest_value, str):
            try:
                manifest = _json_no_duplicates(
                    Path(manifest_value).resolve(strict=False)
                )
                status_value = manifest.get("status")
                if isinstance(status_value, str):
                    status = status_value
                execution_id = manifest.get("execution_id")
                run_id = manifest.get("run_id")
            except GPUHourError:
                pass
        enriched = dict(item)
        enriched["status"] = status
        enriched["execution_id"] = execution_id
        enriched["run_id"] = run_id
        add(
            enriched,
            source="rejected",
            classification=f"rejected_{status}",
            priority=1,
        )

    return [
        value
        for _, value in sorted(
            candidates.values(),
            key=lambda item: (
                item[1]["manifest"],
                item[1]["classification"],
            ),
        )
    ]


def _attempt_row(
    candidate: Mapping[str, Any],
) -> dict[str, Any] | None:
    base = {
        "row_type": "attempt_spend",
        "primary": False,
        "source": candidate["source"],
        "classification": candidate["classification"],
        "manifest": candidate["manifest"],
        "status": "timing_incomplete",
        "issues": [],
        "execution_id": candidate.get("reported_execution_id"),
        "model": candidate.get("reported_model"),
        "setting": candidate.get("reported_setting"),
        "method": None,
        "phase": candidate.get("reported_phase"),
        "kind": None,
        "wall_seconds": None,
        "gpu_indices": None,
        "gpu_count": None,
        "gpu_hour_status": None,
        "gpu_hours": None,
        "gpu_hours_lower_bound": None,
        "missing_ranks": [],
    }
    manifest_path = Path(str(candidate["manifest"])).resolve(strict=False)
    try:
        manifest = _json_no_duplicates(manifest_path)
        component = _parse_component(
            manifest,
            manifest_path,
            primary=False,
        )
        resolved_phase = (
            component["phase"]
            if component["kind"] == "quantization"
            else component["target_phase"]
        )
        reported_execution_id = candidate.get("reported_execution_id")
        if (
            isinstance(reported_execution_id, str)
            and reported_execution_id != component["execution_id"]
        ):
            raise GPUHourError(
                "attempt execution_id disagrees with result/retirement report"
            )
        for field in ("model", "setting", "phase"):
            reported = candidate.get(f"reported_{field}")
            actual = component.get(field)
            if reported is not None and reported != actual:
                raise GPUHourError(
                    f"attempt {field} disagrees with result/retirement report"
                )
        if resolved_phase != "final":
            # Result rows can be sparse, so the preliminary metadata gate in
            # _attempt_inputs is not sufficient.  Apply the authoritative
            # final-only gate only after strict sidecar/provenance parsing.
            return None
        exact = component["gpu_hour_status"] == "exact"
        base.update(
            {
                "status": "recorded" if exact else "timing_incomplete",
                "execution_id": component["execution_id"],
                "model": component["model"],
                "setting": component["setting"],
                "method": component["method"],
                "phase": resolved_phase,
                "kind": component["kind"],
                "wall_seconds": component["wall_seconds"],
                "gpu_indices": component["gpu_indices"],
                "gpu_count": component["gpu_count"],
                "gpu_hour_status": component["gpu_hour_status"],
                "gpu_hours": component["allocated_gpu_hours"],
                "gpu_hours_lower_bound": component[
                    "gpu_hours_lower_bound"
                ],
                "missing_ranks": component["missing_ranks"],
                "component_id": component["component_id"],
                "sidecar": component["sidecar"],
                "sidecar_sha256": component["sidecar_sha256"],
            }
        )
        if not exact:
            base["issues"] = [
                "allocated rank timing evidence is incomplete; only a "
                "strict observed GPU-hour lower bound is available"
            ]
    except GPUHourError as exc:
        base["issues"] = [str(exc)]
    return base


def _unique_attempt_rows(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Parse final attempts and count each execution exactly once.

    A copied output directory can expose the same immutable timing evidence at
    more than one manifest path.  Byte-identical evidence is one execution,
    not additional spend.  Conflicting evidence for one execution fails
    closed and contributes only a conservative single-execution lower bound.
    """

    rows: list[dict[str, Any]] = []
    by_execution_id: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        row = _attempt_row(candidate)
        if row is None:
            continue
        execution_id = row.get("execution_id")
        if not isinstance(execution_id, str) or not execution_id:
            rows.append(row)
            continue
        previous = by_execution_id.get(execution_id)
        if previous is None:
            by_execution_id[execution_id] = row
            rows.append(row)
            continue
        if (
            isinstance(row.get("component_id"), str)
            and row.get("component_id") == previous.get("component_id")
            and isinstance(row.get("sidecar_sha256"), str)
            and row.get("sidecar_sha256") == previous.get("sidecar_sha256")
        ):
            # Exact copies of one strict sidecar are aliases of the same
            # execution.  Keep the first canonical input and count it once.
            continue

        observed_bounds = []
        for value in (previous, row):
            exact_value = value.get("gpu_hours")
            lower_value = value.get("gpu_hours_lower_bound")
            if isinstance(exact_value, (int, float)) and not isinstance(
                exact_value, bool
            ):
                observed_bounds.append(float(exact_value))
            elif isinstance(lower_value, (int, float)) and not isinstance(
                lower_value, bool
            ):
                observed_bounds.append(float(lower_value))
        previous["status"] = "timing_incomplete"
        previous["gpu_hour_status"] = "lower_bound"
        previous["gpu_hours"] = None
        previous["gpu_hours_lower_bound"] = (
            max(observed_bounds) if observed_bounds else None
        )
        previous.setdefault("issues", []).append(
            "conflicting timing evidence shares one execution_id; "
            "the execution cannot be counted exactly"
        )
    return rows


def build_gpu_hour_report(
    result_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one fail-closed formal accounting report.

    ``result_report`` is the JSON payload produced by
    ``lowbit_activation_results.py``.  Manifest paths in accepted records and
    successful informational precompute rows locate the pinned sidecars.
    """

    if result_report.get("schema_version") != 1:
        raise GPUHourError("result report schema_version must equal 1")
    records_value = result_report.get("records")
    informational_value = result_report.get("informational")
    if not isinstance(records_value, list) or not isinstance(
        informational_value, list
    ):
        raise GPUHourError("result report records/informational must be lists")
    formal = [
        _require_mapping(value, f"records[{index}]")
        for index, value in enumerate(records_value)
        if isinstance(value, Mapping) and value.get("phase") == "final"
    ]

    issues: list[str] = []
    if not formal:
        issues.append("no accepted formal records were supplied")
    matrix = result_report.get("matrix_completeness")
    if not isinstance(matrix, Mapping) or matrix.get("complete") is not True:
        issues.append("source result report does not contain a complete matrix")

    by_group: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for record in formal:
        key = _record_key(record)
        if key in by_group:
            issues.append(f"duplicate accepted formal group: {key}")
        by_group[key] = record

    components: dict[str, dict[str, Any]] = {}
    component_errors: dict[str, str] = {}
    record_components: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, record in sorted(by_group.items()):
        model, setting, method = key
        if method == "bf16":
            continue
        manifest_value = record.get("manifest")
        if not isinstance(manifest_value, str):
            component_errors[str(key)] = "accepted record lacks manifest path"
            continue
        manifest_path = Path(manifest_value).resolve(strict=False)
        try:
            manifest = _json_no_duplicates(manifest_path)
            component = _parse_component(manifest, manifest_path)
            if (
                component["kind"] != "quantization"
                or (component["model"], component["setting"], component["method"])
                != key
            ):
                raise GPUHourError(
                    "phase timing identity disagrees with accepted result row"
                )
            _validate_record_identity(record, component)
            world_size = record.get("world_size")
            if type(world_size) is not int or world_size != component["gpu_count"]:
                raise GPUHourError(
                    "accepted result world_size disagrees with timed GPU count"
                )
            if component["component_id"] in components:
                raise GPUHourError("duplicate component_id")
            components[component["component_id"]] = component
            record_components[key] = component
        except GPUHourError as exc:
            component_errors[str(key)] = str(exc)

    producer_candidates: list[dict[str, Any]] = []
    for index, value in enumerate(informational_value):
        if not isinstance(value, Mapping):
            continue
        if value.get("kind") != "precompute" or value.get("status") != "succeeded":
            continue
        manifest_value = value.get("manifest")
        if not isinstance(manifest_value, str):
            continue
        manifest_path = Path(manifest_value).resolve(strict=False)
        try:
            manifest = _json_no_duplicates(manifest_path)
            component = _parse_component(manifest, manifest_path)
            if component["kind"] != "shared_precompute":
                raise GPUHourError(
                    "successful precompute timing is not shared_precompute"
                )
            if component["component_id"] in components:
                raise GPUHourError("duplicate component_id")
            components[component["component_id"]] = component
            producer_candidates.append(component)
        except GPUHourError as exc:
            # Tune producers intentionally lack algorithm_core_v1 timing.
            run_id = ""
            with_context = value.get("run_id")
            if isinstance(with_context, str):
                run_id = with_context
            if "_final_" in run_id or run_id.startswith(
                "precompute_guided_saliency_"
            ):
                component_errors[f"precompute[{index}]"] = str(exc)

    producers_by_identity: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for component in producer_candidates:
        producers_by_identity.setdefault(
            (component["model"], component["method"]), []
        ).append(component)

    rows: list[dict[str, Any]] = []
    referenced_producers: set[str] = set()
    for key, record in sorted(by_group.items()):
        model, setting, method = key
        if method == "bf16":
            rows.append(
                {
                    "row_type": "quantization",
                    "model": model,
                    "setting": setting,
                    "method": method,
                    "execution_id": record.get("execution_id"),
                    "manifest": record.get("manifest"),
                    "applicable": False,
                    "status": "not_applicable",
                    "issues": [],
                    "exclusive_wall_seconds": None,
                    "gpu_indices": None,
                    "gpu_count": None,
                    "exclusive_gpu_hours": None,
                    "primary_gpu_hours": None,
                    "shared_precompute_gpu_hours": None,
                    "standalone_inclusive_gpu_hours": None,
                    "shared_precompute_ids": [],
                }
            )
            continue
        component = record_components.get(key)
        if component is None:
            rows.append(
                _incomplete_row(
                    record,
                    component_errors.get(str(key), "phase timing is missing"),
                )
            )
            continue

        expected_producer_method = (
            method if method in _PRECOMPUTE_STAGES else None
        )
        references = component["shared_precompute_refs"]
        producer: dict[str, Any] | None = None
        row_issues: list[str] = []
        if expected_producer_method is None:
            if references:
                row_issues.append("method must not reference shared precompute")
        else:
            candidates = producers_by_identity.get(
                (model, expected_producer_method), []
            )
            if len(candidates) != 1:
                row_issues.append(
                    "expected exactly one formal shared precompute producer"
                )
            elif len(references) != 1:
                row_issues.append(
                    "quantization must reference exactly one shared producer"
                )
            else:
                candidate = candidates[0]
                reference = references[0]
                expected_reference = {
                    "component_id": candidate["component_id"],
                    "producer_execution_id": candidate["execution_id"],
                    "cache_identity_sha256": candidate[
                        "cache_identity_sha256"
                    ],
                    "artifact_set_sha256": candidate["artifact_set_sha256"],
                }
                if reference != expected_reference:
                    row_issues.append(
                        "shared precompute reference disagrees with producer"
                    )
                elif any(
                    candidate["identity"][name]
                    != component["identity"][name]
                    for name in (
                        "plan_sha256",
                        "model_sha256",
                        "numerical_source_sha256",
                    )
                ):
                    row_issues.append(
                        "shared producer identity hashes disagree with consumer"
                    )
                else:
                    producer = candidate
                    referenced_producers.add(candidate["component_id"])

        if row_issues:
            row = _incomplete_row(record, "; ".join(row_issues))
            row.update(
                {
                    "exclusive_wall_seconds": component["wall_seconds"],
                    "gpu_indices": component["gpu_indices"],
                    "gpu_count": component["gpu_count"],
                    "exclusive_gpu_hours": component["allocated_gpu_hours"],
                    "primary_gpu_hours": None,
                    "shared_precompute_gpu_hours": None,
                    "standalone_inclusive_gpu_hours": None,
                }
            )
            rows.append(row)
            continue

        shared_ids = [producer["component_id"]] if producer else []
        inclusive_gpu_hours = component["allocated_gpu_hours"] + (
            producer["allocated_gpu_hours"] if producer else 0.0
        )
        rows.append(
            {
                "row_type": "quantization",
                "model": model,
                "setting": setting,
                "method": method,
                "execution_id": record.get("execution_id"),
                "manifest": record.get("manifest"),
                "applicable": True,
                "status": "complete",
                "issues": [],
                "exclusive_wall_seconds": component["wall_seconds"],
                "gpu_indices": component["gpu_indices"],
                "gpu_count": component["gpu_count"],
                "exclusive_gpu_hours": component["allocated_gpu_hours"],
                "primary_gpu_hours": inclusive_gpu_hours,
                "standalone_inclusive_gpu_hours": inclusive_gpu_hours,
                "shared_precompute_gpu_hours": (
                    producer["allocated_gpu_hours"] if producer else 0.0
                ),
                "shared_precompute_ids": shared_ids,
            }
        )

    expected_producer_keys = {
        (model, method)
        for model, _, method in by_group
        if method in _PRECOMPUTE_STAGES
    }
    for producer_key in sorted(expected_producer_keys):
        candidates = producers_by_identity.get(producer_key, [])
        if len(candidates) != 1:
            issues.append(
                "expected exactly one shared producer for "
                f"{producer_key}, found {len(candidates)}"
            )
    unreferenced = sorted(
        component["component_id"]
        for component in producer_candidates
        if component["component_id"] not in referenced_producers
    )
    if unreferenced:
        issues.append(
            f"{len(unreferenced)} formal shared producer(s) are not exactly linked"
        )

    for name, error in sorted(component_errors.items()):
        issues.append(f"{name}: {error}")
    incomplete_rows = [row for row in rows if row["status"] == "timing_incomplete"]
    if incomplete_rows:
        issues.append(f"{len(incomplete_rows)} formal group timing row(s) incomplete")

    shared_rows = sorted(
        producer_candidates,
        key=lambda item: (item["model"], item["method"], item["component_id"]),
    )
    exclusive_total = sum(
        float(row["exclusive_gpu_hours"])
        for row in rows
        if row["status"] == "complete"
    )
    shared_total = sum(
        float(row["allocated_gpu_hours"]) for row in shared_rows
    )
    primary_ok = not issues
    primary_manifests = {
        str(Path(str(row["manifest"])).resolve(strict=False))
        for row in rows
        if isinstance(row.get("manifest"), str)
    }
    attempt_rows = _unique_attempt_rows(
        _attempt_inputs(
            result_report,
            primary_manifests=primary_manifests,
        )
    )
    incomplete_attempt_rows = [
        row for row in attempt_rows if row["status"] == "timing_incomplete"
    ]
    attempt_recorded_total = sum(
        float(row["gpu_hours"])
        for row in attempt_rows
        if row["status"] == "recorded"
    )
    attempt_lower_bound_total = sum(
        (
            float(row["gpu_hours"])
            if row["status"] == "recorded"
            else float(row["gpu_hours_lower_bound"])
            if isinstance(row.get("gpu_hours_lower_bound"), (int, float))
            and not isinstance(row.get("gpu_hours_lower_bound"), bool)
            else 0.0
        )
        for row in attempt_rows
    )
    attempt_complete = not incomplete_attempt_rows
    successful_total = exclusive_total + shared_total
    ok = primary_ok and attempt_complete
    return {
        "schema_version": 1,
        "scope_id": SCOPE_ID,
        "status": "complete" if ok else "timing_incomplete",
        "ok": ok,
        "primary_status": (
            "complete" if primary_ok else "timing_incomplete"
        ),
        "attempt_spend_status": (
            "complete" if attempt_complete else "timing_incomplete"
        ),
        "accounting_policy": {
            "included": [
                "rotation inside every timed producer/quantization execution",
                "REAL-Q cold formal static precompute core once per model",
                "GuidedGPTQ cold formal saliency precompute core once per model",
                "successful formal quantization algorithm core per group",
            ],
            "excluded": [
                "model loading",
                "FP reference-logit generation",
                "checkpoint saving",
                "WikiText-2 KL/PPL evaluation",
                "lm-eval downstream evaluation",
            ],
            "non_primary": [
                "failed/OOM/retired attempt spend is included in "
                "unique_total_gpu_hours when its timing is exact",
            ],
            "gpu_hour_definition": (
                "max synchronized rank wall_seconds * validated "
                "allocated_gpu_count / 3600"
            ),
            "multi_rank_wall": "max synchronized per-rank elapsed_seconds",
            "shared_precompute": (
                "standalone rows are additive exactly once globally; each "
                "per-group primary_gpu_hours is exclusive quantization plus "
                "that group's complete shared producer and is non-additive "
                "across groups"
            ),
            "incomplete_attempts": (
                "missing allocated-rank evidence contributes only the sum of "
                "strictly observed rank intervals as a lower bound and makes "
                "the exact unique total unavailable"
            ),
            "process_duration_fallback_allowed": False,
        },
        "issues": issues,
        "quantization_rows": rows,
        "shared_precompute_rows": shared_rows,
        "attempt_spend_rows": attempt_rows,
        "attempt_spend_issues": [
            {
                "manifest": row["manifest"],
                "issues": row["issues"],
            }
            for row in incomplete_attempt_rows
        ],
        "totals": {
            "exclusive_quantization_gpu_hours": (
                exclusive_total if primary_ok else None
            ),
            "shared_precompute_gpu_hours": (
                shared_total if primary_ok else None
            ),
            "unique_successful_gpu_hours": (
                successful_total if primary_ok else None
            ),
            "unique_total_gpu_hours": (
                successful_total + attempt_recorded_total
                if primary_ok and attempt_complete
                else None
            ),
            "failed_attempt_spend_gpu_hours": (
                attempt_recorded_total if attempt_complete else None
            ),
            "failed_attempt_spend_gpu_hours_lower_bound": (
                attempt_lower_bound_total
            ),
            "unique_total_gpu_hours_lower_bound": (
                successful_total + attempt_lower_bound_total
                if primary_ok
                else None
            ),
            "bf16_contribution_gpu_hours": 0.0,
        },
    }


_GROUP_CSV_FIELDS = (
    "model",
    "setting",
    "method",
    "applicable",
    "status",
    "exclusive_wall_seconds",
    "gpu_count",
    "exclusive_gpu_hours",
    "standalone_inclusive_gpu_hours",
    "primary_gpu_hours",
    "shared_precompute_gpu_hours",
    "shared_precompute_ids",
    "execution_id",
    "manifest",
    "issues",
)
_PRECOMPUTE_CSV_FIELDS = (
    "component_id",
    "model",
    "method",
    "stage",
    "wall_seconds",
    "gpu_count",
    "allocated_gpu_hours",
    "execution_id",
    "manifest",
    "cache_identity_sha256",
    "artifact_set_sha256",
)
_ATTEMPT_CSV_FIELDS = (
    "classification",
    "source",
    "status",
    "kind",
    "phase",
    "model",
    "setting",
    "method",
    "wall_seconds",
    "gpu_count",
    "gpu_hour_status",
    "gpu_hours",
    "gpu_hours_lower_bound",
    "missing_ranks",
    "execution_id",
    "manifest",
    "issues",
)


def _csv_text(
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for source in rows:
        row = {field: source.get(field) for field in fields}
        for field in ("shared_precompute_ids", "missing_ranks", "issues"):
            if field in row:
                row[field] = json.dumps(
                    row[field], ensure_ascii=False, sort_keys=True
                )
        writer.writerow(row)
    return output.getvalue()


def group_csv(report: Mapping[str, Any]) -> str:
    rows: list[dict[str, Any]] = []
    for source in report["quantization_rows"]:
        row = dict(source)
        if row.get("method") == "bf16" and row.get("applicable") is False:
            for field in (
                "exclusive_wall_seconds",
                "gpu_count",
                "exclusive_gpu_hours",
                "standalone_inclusive_gpu_hours",
                "primary_gpu_hours",
                "shared_precompute_gpu_hours",
            ):
                row[field] = "N/A"
        rows.append(row)
    return _csv_text(rows, _GROUP_CSV_FIELDS)


def precompute_csv(report: Mapping[str, Any]) -> str:
    return _csv_text(report["shared_precompute_rows"], _PRECOMPUTE_CSV_FIELDS)


def attempts_csv(report: Mapping[str, Any]) -> str:
    return _csv_text(report["attempt_spend_rows"], _ATTEMPT_CSV_FIELDS)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-json",
        type=Path,
        required=True,
        help="JSON report emitted by lowbit_activation_results.py",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--groups-csv-out", type=Path)
    parser.add_argument("--precompute-csv-out", type=Path)
    parser.add_argument("--attempts-csv-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    try:
        result_report = _json_no_duplicates(args.results_json.resolve(strict=True))
        report = build_gpu_hour_report(result_report)
        rendered = json.dumps(
            report, indent=2, ensure_ascii=False, sort_keys=True
        ) + "\n"
        if args.json_out:
            _atomic_write(args.json_out, rendered)
        if args.groups_csv_out:
            _atomic_write(args.groups_csv_out, group_csv(report))
        if args.precompute_csv_out:
            _atomic_write(args.precompute_csv_out, precompute_csv(report))
        if args.attempts_csv_out:
            _atomic_write(args.attempts_csv_out, attempts_csv(report))
    except (GPUHourError, OSError) as exc:
        print(
            json.dumps(
                {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    print(rendered, end="")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
