#!/usr/bin/env python3
"""Parse and select low-bit activation experiment results, fail closed.

The executor deliberately records more provenance than a metric scraper
usually consumes.  This tool treats that provenance as part of the result:
only successful executions with complete plan/model/source identities and one
unambiguous WikiText-2 KL/PPL table are admitted.

JSON contains accepted records, informational attempts/precomputes, rejected
manifests, and REAL-Q tuning diagnostics.  CSV contains one row per accepted
result.  Every tune launch counts toward the hard attempt budget, while a
report is not considered OK when a result is rejected, a tuning sweep remains
unresolved, or a formal REAL-Q LR disagrees with its exact winner or a
cryptographically frozen, explicitly reviewed interpolation decision.
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence


MANIFEST_FILENAME = "execution_manifest.json"
LOG_FILENAME = "execution.log"
RETIREMENT_LEDGER_RELATIVE_PATH = Path("_campaign") / "retirements.jsonl"
CAMPAIGN_EVENTS_RELATIVE_PATH = Path("_campaign") / "events.jsonl"
RETIREMENT_SCHEMA_VERSION = 1
MANUAL_LR_FREEZE_SCOPE_ID = "lowbit_activation_manual_lr_selection_v1"
MANUAL_LR_FREEZE_GATE_ID = "results_parser_manual_lr_selection_v1"
MANUAL_LR_FREEZE_ARTIFACT_FILENAME = (
    "manual_lr_selection_freeze_v1.json"
)
MANUAL_LR_FREEZE_ARTIFACT_KIND = (
    "manual_lr_selection_freeze_attestation_v1"
)
MANUAL_INVALID_LEDGER_RELATIVE_PATH = (
    Path("_campaign") / "manual_invalid_results.jsonl"
)
MANUAL_LR_RECONCILIATION_POLICY_ID = (
    "retirement_filtered_results_authoritative_v1"
)
MANUAL_LR_RESULTS_ADOPTION_KIND = (
    "results_parser_gate_source_adoption_v1"
)
REVIEWED_INTERPOLATED_OVERRIDE = {
    "model": "qwen3-4b",
    "setting": "2W16A",
    "selected_lr": 1.2e-4,
    "evidence_lrs": (1.18e-4, 1.25e-4),
}
REVIEWED_MANUAL_INVALID_RESULT = {
    "model": "qwen3-4b",
    "setting": "2W16A",
    "grad_lr": 1.5e-4,
    "reason_code": "numerical_source_changed_during_execution",
    "rejection_errors": (
        "manifest.numerical_source_tree.changed_during_execution must be false",
    ),
}
REVIEWED_MANUAL_INVALID_RESULTS = {
    ("qwen3-4b", "2W16A", 1.5e-4, 1): {
        "reason_code": "numerical_source_changed_during_execution",
        "numerical_source_sha256_at_start": (
            "53c916a9ababb36c4e9fc16a78d04a51d360ebe46794495f3b18c992249cb64a"
        ),
        "numerical_source_sha256_at_end": (
            "817c218d7d969a754323d5d77dc1cb6f40d833b6be2ba818831223ad67d5a63e"
        ),
        "rejection_errors": REVIEWED_MANUAL_INVALID_RESULT[
            "rejection_errors"
        ],
    },
    ("qwen3-32b", "2W4A", 5e-6, 1): {
        "reason_code": "numerical_source_changed_during_execution",
        "numerical_source_sha256_at_start": (
            "53c916a9ababb36c4e9fc16a78d04a51d360ebe46794495f3b18c992249cb64a"
        ),
        "numerical_source_sha256_at_end": (
            "2c4f87b78eb862526cd00b534cbac65df89f116ed730b635485998be064018a2"
        ),
        "rejection_errors": REVIEWED_MANUAL_INVALID_RESULT[
            "rejection_errors"
        ],
    },
}
EXPECTED_TASKS = (
    "piqa",
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "lambada_openai",
    "ceval-valid",
    "boolq",
    "openbookqa",
    "social_iqa",
)
EXPECTED_TASK_METRICS = {
    "piqa": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "winogrande": "acc,none",
    "lambada_openai": "acc,none",
    "ceval-valid": "acc_norm,none",
    "boolq": "acc,none",
    "openbookqa": "acc_norm,none",
    "social_iqa": "acc,none",
}
EXPECTED_LM_EVAL_VERSION = "0.4.4"
EXPECTED_METHODS = ("realq", "gptaq", "guided_gptq", "bf16")
EXPECTED_SEARCH_POLICY = {
    "strategy": "adaptive_bracket_then_canonical_refine",
    "canonical_mantissas": [1, 2, 3, 5, 7],
    "boundary_expansion_batch_size": 2,
    "reserved_refinement_attempts": 3,
    "zero_boundary_probe_ratios": [0.1, 0.3, 0.7],
    "fixed_upper_lr_ceiling": None,
    "require_complete_coarse_round": True,
    "require_local_refinement": True,
}
_INFORMATIONAL_STATUSES = {
    "dry_run",
    "launching",
    "launch_failed",
    "running",
    "interrupted",
    "terminated",
    "failed",
    "wrapper_failed",
}
_CAMPAIGN_TERMINAL_TASK_STATES = {
    "SUCCEEDED",
    "FAILED",
    "OOM",
    "INVALID_RESULT",
    "ORPHANED",
    "SUPERSEDED",
    "CANCELLED",
    "LAUNCH_FAILED",
}
_RETIREMENT_ENTRY_KEYS = {
    "schema_version",
    "seq",
    "timestamp_utc",
    "campaign_id",
    "prev_record_sha256",
    "plan_sha256",
    "phase",
    "model",
    "setting",
    "reason",
    "trigger",
    "old_profile",
    "replacement_profile",
    "retired_manifests",
    "record_sha256",
}
_RETIREMENT_PROFILE_KEYS = {
    "level",
    "generation",
    "effective_overrides",
}
_RETIREMENT_FILE_KEYS = {
    "absolute_path",
    "relative_path",
    "sha256",
    "size_bytes",
}
_RETIREMENT_IDENTITY_KEYS = {
    "task_id",
    "launch_event_seq",
    "phase",
    "model",
    "setting",
    "grad_lr",
    "profile_level",
    "generation",
    "effective_overrides",
    "status",
    "manifest_status",
    "manifest",
    "log",
    "execution_id",
    "run_id",
    "plan_sha256",
    "model_sha256",
}
_RETIREMENT_STATUSES = {
    "succeeded",
    "oom",
    "failed",
    "invalid_result",
    "orphaned",
    "launch_failed",
}
_FAILED_MANIFEST_STATUSES = {
    "failed",
    "terminated",
    "interrupted",
    "launch_failed",
    "wrapper_failed",
}
_MANUAL_LR_FREEZE_WRAPPER_KEYS = {
    "schema_version",
    "scope_id",
    "ledger_sha256",
    "artifact",
    "ledger",
}
_MANUAL_LR_FREEZE_ARTIFACT_REF_KEYS = {
    "filename",
    "sha256",
    "size_bytes",
}
_MANUAL_LR_FREEZE_LEDGER_KEYS = {
    "schema_version",
    "scope_id",
    "campaign_id",
    "selection_sha256",
    "expected_state_sha256",
    "expected_plan_sha256",
    "source_identity_before",
    "timing_gate_sha256",
    "timing_source_set_sha256",
    "downstream_required_gate",
    "results_source_adoption",
    "selections",
}
_MANUAL_LR_FREEZE_ARTIFACT_KEYS = {
    "schema_version",
    "scope_id",
    "artifact_kind",
    "transaction_id",
    "created_utc",
    "ledger_sha256",
    "ledger",
    "pre_freeze_state",
    "timing_gate",
    "state_reconciliation",
}
_MANUAL_LR_FREEZE_PRE_STATE_KEYS = {
    "sha256",
    "size_bytes",
    "content_base64",
}
_MANUAL_LR_RECONCILIATION_KEYS = {
    "schema_version",
    "policy_id",
    "accepted_tune_results",
    "accepted_active_set_sha256",
    "group_attempt_counts",
    "group_attempt_counts_sha256",
    "tune_launches",
    "tune_launches_sha256",
    "invalid_result_resolutions",
    "invalid_result_resolutions_sha256",
    "raw_input_prefixes",
    "raw_input_prefixes_sha256",
    "report_snapshot_sha256",
    "terminalized_tasks",
    "terminalized_tasks_sha256",
    "imported_tasks",
    "imported_tasks_sha256",
}
_MANUAL_LR_RECONCILIATION_POINT_KEYS = {
    "model",
    "setting",
    "manifest",
    "manifest_sha256",
    "log",
    "log_sha256",
    "execution_id",
    "run_id",
    "attempt_index",
    "world_size",
    "grad_lr",
    "kl_wikitext2",
    "ppl_wikitext2",
    "plan_sha256",
    "model_sha256",
    "runner_sha256",
    "executor_sha256",
    "numerical_source_sha256",
    "config_sha256",
}
_MANUAL_LR_RECONCILIATION_COUNT_KEYS = {
    "model",
    "setting",
    "attempt_count",
}
_MANUAL_LR_RECONCILIATION_TERMINALIZED_KEYS = {
    "task_id",
    "previous_status",
    "reconciled_status",
    "reason",
}
_MANUAL_LR_RAW_PREFIX_REF_KEYS = {
    "relative_path",
    "present",
    "prefix_sha256",
    "prefix_size_bytes",
}
_MANUAL_LR_RAW_INPUT_PREFIX_KEYS = {
    "retirement_ledger",
    "campaign_events",
    "manual_invalid_results",
}
_MANUAL_LR_FILE_REF_KEYS = {
    "absolute_path",
    "relative_path",
    "present",
    "sha256",
    "size_bytes",
}
_MANUAL_LR_TUNE_LAUNCH_KEYS = {
    "manifest",
    "log",
    "classification",
    "rejection_errors",
    "execution_id",
    "run_id",
    "status",
    "exit_code",
    "model",
    "setting",
    "grad_lr",
    "attempt_index",
    "context_valid",
    "plan_sha256",
    "model_sha256",
    "runner_sha256",
    "executor_sha256",
    "numerical_source_sha256_at_start",
    "numerical_source_sha256_at_end",
    "numerical_source_changed_during_execution",
}
_MANUAL_INVALID_ENTRY_KEYS = {
    "schema_version",
    "seq",
    "timestamp_utc",
    "campaign_id",
    "reason_code",
    "authorization",
    "launch_event_present",
    "launch_event",
    "manifest",
    "log",
    "execution_id",
    "run_id",
    "phase",
    "method",
    "model",
    "setting",
    "grad_lr",
    "attempt_index",
    "manifest_status",
    "exit_code",
    "plan_sha256",
    "model_sha256",
    "runner_sha256",
    "executor_sha256",
    "numerical_source_sha256_at_start",
    "numerical_source_sha256_at_end",
    "numerical_source_changed_during_execution",
    "expected_rejection_errors",
    "prev_record_sha256",
    "record_sha256",
}
_MANUAL_INVALID_AUTHORIZATION_KEYS = {
    "kind",
    "authorized_by",
    "rationale",
}
_MANUAL_INVALID_LAUNCH_EVENT_KEYS = {
    "seq",
    "line_sha256",
    "task_id",
}
_MANUAL_LR_IMPORTED_TASK_KEYS = {
    "task_id",
    "spec",
    "status",
    "gpu_count",
    "priority",
    "gpus",
    "attempt_index",
    "executor_pid",
    "created_utc",
    "launched_utc",
    "output_dir",
    "manifest",
    "log",
    "exit_code",
    "failure_class",
    "cache_validation",
    "imported",
    "plan_compatible",
    "metric",
    "imported_by_manual_lr_freeze",
    "reconciliation_reason",
    "finished_utc",
}
_MANUAL_LR_IMPORTED_TASK_SPEC_KEYS = {
    "kind",
    "method",
    "phase",
    "target_phase",
    "model",
    "setting",
    "grad_lr",
    "overrides",
    "profile_level",
    "static_profile_level",
    "generation",
    "purpose",
}
_MANUAL_LR_IMPORTED_TASK_METRIC_KEYS = {
    "kl",
    "ppl",
    "grad_lr",
    "identities",
}
_MANUAL_LR_IMPORTED_TASK_IDENTITY_KEYS = {
    "plan_sha256",
    "model_sha256",
    "runner_sha256",
    "executor_sha256",
    "numerical_source_sha256",
}
_MANUAL_LR_STATE_RECONCILIATION_KEYS = {
    "schema_version",
    "policy_id",
    "phase",
    "artifact_sha256",
    "accepted_active_set_sha256",
    "group_attempt_counts_sha256",
    "tune_launches_sha256",
    "invalid_result_resolutions_sha256",
    "raw_input_prefixes_sha256",
    "report_snapshot_sha256",
    "terminalized_tasks_sha256",
    "imported_tasks_sha256",
}
_MANUAL_LR_FREEZE_ADOPTION_KEYS = {
    "kind",
    "expected_old_results_sha256",
    "accepted_current_results_sha256",
}
_MANUAL_LR_HANDOFF_KEYS = {
    "schema_version",
    "scope_id",
    "transaction_id",
    "phase",
    "artifact_sha256",
    "ledger_sha256",
    "selection_sha256",
    "expected_state_sha256",
    "expected_plan_sha256",
    "formal_plan_sha256",
}
_MANUAL_LR_FREEZE_GATE_KEYS = {
    "gate_id",
    "required",
    "status",
}
_MANUAL_LR_FREEZE_SELECTION_KEYS = {
    "model",
    "setting",
    "selected_lr",
    "decision_kind",
    "rationale",
    "selected_manifest",
    "evidence_set_sha256",
    "evidence",
}
_MANUAL_LR_FREEZE_EVIDENCE_KEYS = {
    "manifest",
    "manifest_sha256",
    "log",
    "log_sha256",
    "execution_id",
    "run_id",
    "attempt_index",
    "grad_lr",
    "kl_wikitext2",
    "plan_sha256",
    "model_sha256",
    "runner_sha256",
    "executor_sha256",
    "numerical_source_sha256",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_OOM_PATTERNS = (
    re.compile(r"\bcuda out of memory\b", re.IGNORECASE),
    re.compile(r"\btorch\.cuda\.OutOfMemoryError\b"),
    re.compile(r"\bCUBLAS_STATUS_ALLOC_FAILED\b"),
    re.compile(r"\bHIP out of memory\b", re.IGNORECASE),
)
_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")
_KL_DETAIL_RE = re.compile(
    r"(?:(Exact)\s+)?KL&PPL\s+on\s+([^:\s]+)\s*:\s*"
    r"([^,\s]+)\s*,\s*([^\s|]+)",
    re.IGNORECASE,
)


class ResultError(ValueError):
    """A result is incomplete, ambiguous, or inconsistent."""


@dataclass(frozen=True)
class MarkdownTable:
    headers: tuple[str, ...]
    values: tuple[str, ...]


@dataclass(frozen=True)
class ParsedResult:
    manifest_path: Path
    log_path: Path
    execution_id: str
    run_id: str
    model: str
    setting: str
    method: str
    phase: str
    world_size: int
    attempt_index: int
    grad_lr: float | None
    final_layer_grad_lr: float | None
    kl_wikitext2: float | None
    ppl_wikitext2: float
    tasks: dict[str, float]
    acc_avg: float | None
    lm_eval_version: str
    params: dict[str, Any]
    identities: dict[str, str]
    expected_lr_candidates: tuple[float, ...]
    max_attempts: int
    search_policy: dict[str, Any]
    plan_content: dict[str, Any]

    def public_dict(self) -> dict[str, Any]:
        return {
            "manifest": str(self.manifest_path),
            "log": str(self.log_path),
            "execution_id": self.execution_id,
            "run_id": self.run_id,
            "model": self.model,
            "setting": self.setting,
            "method": self.method,
            "phase": self.phase,
            "world_size": self.world_size,
            "attempt_index": self.attempt_index,
            "grad_lr": self.grad_lr,
            "final_layer_grad_lr": self.final_layer_grad_lr,
            "kl_wikitext2": self.kl_wikitext2,
            "ppl_wikitext2": self.ppl_wikitext2,
            "tasks": dict(self.tasks),
            "task_metric_keys": {
                task: EXPECTED_TASK_METRICS[task] for task in self.tasks
            },
            "acc_avg": self.acc_avg,
            "lm_eval_version": self.lm_eval_version,
            "params": dict(self.params),
            "identities": dict(self.identities),
        }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResultError(f"{name} must be an object")
    return value


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ResultError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: Any, name: str) -> str:
    value = _require_nonempty_string(value, name)
    if _SHA256_RE.fullmatch(value) is None:
        raise ResultError(f"{name} must be a lowercase SHA256 hex digest")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], name: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing {missing!r}")
        if unexpected:
            details.append(f"unexpected {unexpected!r}")
        raise ResultError(f"{name} fields mismatch: " + ", ".join(details))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ResultError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _finite_float(value: str, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ResultError(f"{name} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ResultError(f"{name} must be finite, got {value!r}")
    return parsed


def _manifest_identities(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, str], Mapping[str, Any]]:
    if manifest.get("schema_version") != 1:
        raise ResultError("manifest.schema_version must be 1")
    if manifest.get("status") != "succeeded":
        raise ResultError("manifest.status must equal 'succeeded'")
    exit_code = manifest.get("exit_code")
    if type(exit_code) is not int or exit_code != 0:
        raise ResultError("manifest.exit_code must be the integer 0")

    plan = _require_mapping(manifest.get("plan"), "manifest.plan")
    plan_sha = _require_sha256(plan.get("sha256"), "manifest.plan.sha256")
    _require_nonempty_string(plan.get("path"), "manifest.plan.path")
    if not _is_number(plan.get("size_bytes")) or plan["size_bytes"] <= 0:
        raise ResultError("manifest.plan.size_bytes must be positive")
    content = _require_mapping(plan.get("content"), "manifest.plan.content")
    if plan.get("changed_during_execution") is not False:
        raise ResultError(
            "manifest.plan.changed_during_execution must be false"
        )
    if plan.get("sha256_at_end") != plan_sha:
        raise ResultError(
            "manifest.plan.sha256_at_end must match manifest.plan.sha256"
        )
    if content.get("schema_version") != 1:
        raise ResultError("embedded plan.schema_version must be 1")
    if tuple(content.get("paper_zero_shot_tasks", ())) != EXPECTED_TASKS:
        raise ResultError(
            "embedded plan does not contain the exact paper ten-task protocol"
        )
    if tuple(content.get("methods", ())) != EXPECTED_METHODS:
        raise ResultError(
            "embedded plan methods must match the reviewed comparison matrix"
        )
    plan_models = _require_mapping(
        content.get("models"), "embedded plan.models"
    )
    plan_settings = _require_mapping(
        content.get("settings"), "embedded plan.settings"
    )
    if not plan_models or not plan_settings:
        raise ResultError("embedded plan models/settings must be non-empty")
    selected_lrs = _require_mapping(
        content.get("selected_grad_lr_by_model_setting"),
        "embedded plan.selected_grad_lr_by_model_setting",
    )
    if set(selected_lrs) != set(plan_models):
        raise ResultError(
            "embedded selected LR ledger must contain every model exactly"
        )
    for model_name in plan_models:
        model_lrs = _require_mapping(
            selected_lrs.get(model_name),
            f"embedded selected LR ledger for {model_name}",
        )
        if set(model_lrs) != set(plan_settings):
            raise ResultError(
                f"embedded selected LR ledger for {model_name} must contain "
                "every setting exactly"
            )
        for setting_name, selected_lr in model_lrs.items():
            if selected_lr is not None and (
                not _is_number(selected_lr)
                or not math.isfinite(float(selected_lr))
                or float(selected_lr) < 0
            ):
                raise ResultError(
                    "embedded selected LR must be null or a non-negative "
                    f"finite number: {model_name}/{setting_name}"
                )
    fixed = _require_mapping(
        content.get("fixed_numerics"), "embedded plan.fixed_numerics"
    )
    if fixed.get("eval_datasets") != ["wikitext2"]:
        raise ResultError(
            "embedded plan eval_datasets must be exactly ['wikitext2']"
        )
    tuning = _require_mapping(content.get("tuning"), "embedded plan.tuning")
    if tuning.get("max_attempts_per_model_setting") != 20:
        raise ResultError(
            "embedded plan tuning.max_attempts_per_model_setting must be 20"
        )
    if tuning.get("search_policy") != EXPECTED_SEARCH_POLICY:
        raise ResultError(
            "embedded plan tuning.search_policy does not match the reviewed "
            "adaptive bracket/refinement policy"
        )

    model = _require_mapping(manifest.get("model"), "manifest.model")
    if model.get("complete") is not True:
        raise ResultError("manifest.model.complete must be true")
    _require_nonempty_string(model.get("argument"), "manifest.model.argument")
    _require_nonempty_string(
        model.get("resolved_path"), "manifest.model.resolved_path"
    )
    model_sha = _require_sha256(
        model.get("combined_identity_sha256"),
        "manifest.model.combined_identity_sha256",
    )
    config = _require_mapping(model.get("config"), "manifest.model.config")
    _require_sha256(config.get("sha256"), "manifest.model.config.sha256")
    if config.get("stable_during_hash") is not True:
        raise ResultError(
            "manifest.model.config.stable_during_hash must be true"
        )
    shards = model.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ResultError("manifest.model.shards must be a non-empty list")
    if model.get("missing_shards") != []:
        raise ResultError("manifest.model.missing_shards must be empty")
    for index, shard_value in enumerate(shards):
        shard = _require_mapping(
            shard_value, f"manifest.model.shards[{index}]"
        )
        _require_nonempty_string(
            shard.get("relative_path"),
            f"manifest.model.shards[{index}].relative_path",
        )
        _require_sha256(
            shard.get("sampled_sha256"),
            f"manifest.model.shards[{index}].sampled_sha256",
        )
        if shard.get("stable_during_hash") is not True:
            raise ResultError(
                f"manifest.model.shards[{index}].stable_during_hash "
                "must be true"
            )

    sources = _require_mapping(
        manifest.get("source_files"), "manifest.source_files"
    )
    source_hashes: dict[str, str] = {}
    for label in ("runner", "executor"):
        source = _require_mapping(
            sources.get(label), f"manifest.source_files.{label}"
        )
        if source.get("exists") is not True:
            raise ResultError(
                f"manifest.source_files.{label}.exists must be true"
            )
        _require_nonempty_string(
            source.get("path"), f"manifest.source_files.{label}.path"
        )
        source_hashes[label] = _require_sha256(
            source.get("sha256"),
            f"manifest.source_files.{label}.sha256",
        )
        if source.get("stable_during_hash") is not True:
            raise ResultError(
                f"manifest.source_files.{label}.stable_during_hash "
                "must be true"
            )
    numerical_tree = _require_mapping(
        manifest.get("numerical_source_tree"),
        "manifest.numerical_source_tree",
    )
    file_count = numerical_tree.get("file_count")
    if type(file_count) is not int or file_count <= 0:
        raise ResultError(
            "manifest.numerical_source_tree.file_count must be positive"
        )
    numerical_sha = _require_sha256(
        numerical_tree.get("combined_sha256"),
        "manifest.numerical_source_tree.combined_sha256",
    )
    if numerical_tree.get("changed_during_execution") is not False:
        raise ResultError(
            "manifest.numerical_source_tree.changed_during_execution "
            "must be false"
        )
    if numerical_tree.get("combined_sha256_at_end") != numerical_sha:
        raise ResultError(
            "manifest numerical source tree end hash must match its start hash"
        )

    return (
        {
            "plan_sha256": plan_sha,
            "model_sha256": model_sha,
            "runner_sha256": source_hashes["runner"],
            "executor_sha256": source_hashes["executor"],
            "numerical_source_sha256": numerical_sha,
        },
        content,
    )


def _validate_python_packages(manifest: Mapping[str, Any]) -> str:
    packages = _require_mapping(
        manifest.get("python_packages"), "manifest.python_packages"
    )
    lm_eval = _require_mapping(
        packages.get("lm_eval"), "manifest.python_packages.lm_eval"
    )
    if lm_eval.get("distribution") != "lm-eval":
        raise ResultError(
            "manifest.python_packages.lm_eval.distribution must be 'lm-eval'"
        )
    if lm_eval.get("version") != EXPECTED_LM_EVAL_VERSION:
        raise ResultError(
            "manifest lm-eval version must be "
            f"{EXPECTED_LM_EVAL_VERSION}, got {lm_eval.get('version')!r}"
        )
    return EXPECTED_LM_EVAL_VERSION


def _option_values(argv: Sequence[str], option: str) -> list[str | None]:
    found: list[str | None] = []
    prefix = f"{option}="
    for index, item in enumerate(argv):
        if item == option:
            if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                found.append(argv[index + 1])
            else:
                found.append(None)
        elif item.startswith(prefix):
            found.append(item[len(prefix) :])
    return found


def _multi_option(argv: Sequence[str], option: str) -> list[str]:
    occurrences: list[list[str]] = []
    prefix = f"{option}="
    for index, item in enumerate(argv):
        if item == option:
            values: list[str] = []
            cursor = index + 1
            while cursor < len(argv) and not argv[cursor].startswith("--"):
                values.append(argv[cursor])
                cursor += 1
            occurrences.append(values)
        elif item.startswith(prefix):
            occurrences.append([item[len(prefix) :]])
    if len(occurrences) != 1:
        if not occurrences:
            raise ResultError(f"argv is missing required option {option}")
        raise ResultError(f"argv contains duplicate option {option}")
    if not occurrences[0] or any(not value for value in occurrences[0]):
        raise ResultError(f"argv option {option} has no value")
    return occurrences[0]


def _option(
    argv: Sequence[str],
    option: str,
    *,
    required: bool = False,
    flag_ok: bool = False,
) -> str | None:
    values = _option_values(argv, option)
    if len(values) > 1:
        raise ResultError(f"argv contains duplicate option {option}")
    if not values:
        if required:
            raise ResultError(f"argv is missing required option {option}")
        return None
    value = values[0]
    if value is None and not flag_ok:
        raise ResultError(f"argv option {option} has no value")
    return value


def _typed_option(
    argv: Sequence[str],
    option: str,
    kind: type,
    *,
    required: bool = False,
) -> Any:
    value = _option(argv, option, required=required)
    if value is None:
        return None
    if kind is str:
        return value
    if kind is int:
        try:
            return int(value)
        except ValueError as exc:
            raise ResultError(f"argv {option} must be an integer") from exc
    if kind is float:
        return _finite_float(value, f"argv {option}")
    raise AssertionError(f"unsupported option type {kind!r}")


def _bool_option(
    argv: Sequence[str],
    option: str,
    *,
    required: bool = False,
    legacy_flag: bool = False,
) -> bool | None:
    value = _option(
        argv, option, required=required, flag_ok=legacy_flag
    )
    if value is None:
        return True if _option_values(argv, option) else None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ResultError(f"argv {option} must be true or false")


def _resolved_path(value: str, cwd: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(cwd) / path
    return path.resolve(strict=False)


def _command_world_size(argv: Sequence[str]) -> int:
    torchrun_positions = [
        index for index, item in enumerate(argv) if Path(item).name == "torchrun"
    ]
    module_positions = [
        index
        for index in range(max(0, len(argv) - 2))
        if Path(argv[index]).name in {"python", "python3"}
        and argv[index + 1] == "-m"
        and argv[index + 2] == "torch.distributed.run"
    ]
    launcher_count = len(torchrun_positions) + len(module_positions)
    if launcher_count == 0:
        return 1
    if launcher_count != 1:
        raise ResultError(
            "argv must contain at most one torchrun or "
            "python -m torch.distributed.run launcher"
        )
    standalone = _option_values(argv, "--standalone")
    if standalone != [None]:
        raise ResultError(
            "torchrun argv must contain exactly one bare --standalone flag"
        )
    node_values = _option_values(argv, "--nnodes")
    if len(node_values) != 1 or node_values[0] is None:
        raise ResultError(
            "torchrun argv must contain exactly one valued --nnodes option"
        )
    try:
        node_count = int(node_values[0])
    except ValueError as exc:
        raise ResultError("torchrun nnodes must be an integer") from exc
    if node_count != 1:
        raise ResultError("accepted commands must use torchrun --nnodes=1")
    values = [
        *_option_values(argv, "--nproc-per-node"),
        *_option_values(argv, "--nproc_per_node"),
    ]
    if len(values) != 1 or values[0] is None:
        raise ResultError(
            "torchrun argv must contain exactly one valued "
            "--nproc-per-node/--nproc_per_node option"
        )
    try:
        world_size = int(values[0])
    except ValueError as exc:
        raise ResultError("torchrun nproc-per-node must be an integer") from exc
    if world_size <= 0:
        raise ResultError("torchrun nproc-per-node must be positive")
    return world_size


def _model_name(
    model_argument: str,
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> str:
    models = _require_mapping(plan.get("models"), "embedded plan.models")
    cwd = _require_nonempty_string(
        _require_mapping(manifest.get("command"), "manifest.command").get(
            "cwd"
        ),
        "manifest.command.cwd",
    )
    argument_path = _resolved_path(model_argument, cwd)
    matches = [
        str(name)
        for name, value in models.items()
        if isinstance(value, str)
        and (
            value == model_argument
            or _resolved_path(value, cwd) == argument_path
        )
    ]
    if len(matches) != 1:
        raise ResultError(
            "argv --model must resolve to exactly one embedded-plan model, "
            f"got matches={matches!r}"
        )
    model_identity = _require_mapping(
        manifest.get("model"), "manifest.model"
    )
    if model_identity.get("argument") != model_argument:
        raise ResultError(
            "argv --model disagrees with manifest.model.argument"
        )
    if (
        _resolved_path(str(model_identity["resolved_path"]), cwd)
        != argument_path
    ):
        raise ResultError(
            "argv --model disagrees with manifest.model.resolved_path"
        )
    return matches[0]


def _setting_name(
    argv: Sequence[str], plan: Mapping[str, Any], method: str
) -> str:
    bit_names = ("w_bits", "a_bits", "k_bits", "v_bits")
    bits = tuple(
        _typed_option(argv, f"--{name}", int, required=True)
        for name in bit_names
    )
    if method == "bf16":
        if bits != (16, 16, 16, 16):
            raise ResultError("BF16 argv must use W/A/K/V=16")
        return "BF16"
    settings = _require_mapping(
        plan.get("settings"), "embedded plan.settings"
    )
    matches = [
        str(name)
        for name, config_value in settings.items()
        if isinstance(config_value, Mapping)
        and tuple(config_value.get(name) for name in bit_names) == bits
    ]
    if len(matches) != 1:
        raise ResultError(
            "argv W/A/K/V bits must match exactly one embedded-plan setting, "
            f"got bits={bits!r}, matches={matches!r}"
        )
    return matches[0]


_COMMON_VALUE_OPTIONS: tuple[tuple[str, type, bool], ...] = (
    ("dataset", str, True),
    ("eval_seq_len", int, True),
    ("w_bits", int, True),
    ("w_groupsize", int, True),
    ("a_bits", int, True),
    ("k_bits", int, True),
    ("v_bits", int, True),
    ("a_groupsize", int, True),
    ("k_groupsize", int, True),
    ("v_groupsize", int, True),
    ("a_clip_ratio", float, True),
    ("k_clip_ratio", float, True),
    ("v_clip_ratio", float, True),
    ("num_groups", int, True),
    ("percdamp", float, True),
    ("blocksize", int, True),
    ("kl_topk", int, True),
    ("nsamples", int, True),
    ("seq_len", int, True),
    ("seed", int, True),
    ("rotation_seed", int, True),
    ("refresh_seed", int, True),
    ("lm_eval_batch_size", int, False),
    ("alpha", float, False),
    ("w_method", str, False),
)
_REALQ_VALUE_OPTIONS: tuple[tuple[str, type, bool], ...] = (
    ("bsz", int, True),
    ("global_loss_bsz", int, True),
    ("hessian_accum_bsz", int, True),
    ("backward_samples", int, True),
    ("backward_bsz", int, True),
    ("final_layer_backward_bsz", int, True),
    ("grad_lr", float, True),
    ("final_layer_grad_lr", float, True),
    ("grad_clip", float, True),
    ("grad_lr_layer_schedule", str, True),
    ("grad_lr_layer_base_ratio", float, True),
    ("a_loss_ratio", float, True),
    ("a_loss_clip_scope", str, True),
    ("saliency_clip_percentile", float, True),
    ("grad_hessian_topk", int, True),
    ("group_parallel_quant", str, True),
    ("static_cache_path", str, True),
    ("w_clip_search_impl", str, True),
    ("act_order_stitch_impl", str, True),
    ("w_clip_update_impl", str, True),
    ("w_group_param_layout", str, True),
)
_REALQ_BOOL_OPTIONS = (
    "w_asym",
    "a_asym",
    "k_asym",
    "v_asym",
    "w_clip",
    "act_order",
    "rotate",
    "act_quant_aware_gptq",
    "k_cache_quant_aware_gptq",
    "loss_slide_window",
    "fsdp",
    "cpu_master",
    "require_static_cache_hit",
    "require_reference_cache_hit",
    "skip_eval",
    "lm_eval",
    "log_column_block_loss",
    "quantizer_inner_fastpath",
    "fisher_fp32_cache",
)
_LEGACY_BOOL_OPTIONS = (
    "w_clip",
    "act_order",
    "rotate",
    "offload_inps",
    "act_quant_aware_gptq",
    "k_cache_quant_aware_gptq",
    "lm_eval",
)


def _parse_argv(
    manifest: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    command = _require_mapping(manifest.get("command"), "manifest.command")
    argv_value = command.get("argv")
    if (
        not isinstance(argv_value, list)
        or not argv_value
        or not all(isinstance(item, str) for item in argv_value)
    ):
        raise ResultError("manifest.command.argv must be a non-empty string list")
    argv: list[str] = argv_value
    world_size = _command_world_size(argv)

    is_realq = "realq.ptq" in argv
    is_legacy = any(Path(item).name == "ptq.py" for item in argv)
    if is_realq == is_legacy:
        raise ResultError(
            "argv must identify exactly one supported REAL-Q or legacy entrypoint"
        )

    exp = _typed_option(argv, "--exp", str, required=True)
    run_id = _require_nonempty_string(manifest.get("run_id"), "manifest.run_id")
    if exp != run_id:
        raise ResultError("argv --exp must equal manifest.run_id")
    command_env = _require_mapping(
        command.get("env"), "manifest.command.env"
    )
    attempt_text = command_env.get("LOWBIT_ACTIVATION_ATTEMPT_INDEX")
    if not isinstance(attempt_text, str):
        raise ResultError(
            "manifest.command.env must record "
            "LOWBIT_ACTIVATION_ATTEMPT_INDEX as a string"
        )
    try:
        attempt_index = int(attempt_text)
    except ValueError as exc:
        raise ResultError(
            "LOWBIT_ACTIVATION_ATTEMPT_INDEX must be an integer"
        ) from exc
    if attempt_index <= 0 or str(attempt_index) != attempt_text:
        raise ResultError(
            "LOWBIT_ACTIVATION_ATTEMPT_INDEX must be a canonical positive integer"
        )
    attempt_match = re.search(r"_attempt([1-9][0-9]*)$", exp)
    if attempt_index == 1:
        if attempt_match is not None:
            raise ResultError("attempt 1 run_id must not have an _attemptN suffix")
    elif attempt_match is None or int(attempt_match.group(1)) != attempt_index:
        raise ResultError(
            "retry run_id suffix must equal LOWBIT_ACTIVATION_ATTEMPT_INDEX"
        )
    if exp.startswith("tune_"):
        phase = "tune"
    elif exp.startswith("final_"):
        phase = "final"
    else:
        raise ResultError("argv --exp must identify tune or final phase")

    w_method = _typed_option(argv, "--w_method", str)
    w_bits = _typed_option(argv, "--w_bits", int, required=True)
    if is_realq:
        if w_method is not None:
            raise ResultError("REAL-Q argv must not contain --w_method")
        method = "realq"
    elif w_method == "gptaq":
        method = "gptaq"
    elif w_method == "gptq_guided":
        method = "guided_gptq"
    elif w_method is None and w_bits == 16:
        method = "bf16"
    else:
        raise ResultError(
            f"unsupported legacy method identity w_method={w_method!r}, "
            f"w_bits={w_bits!r}"
        )
    if method != "realq" and phase != "final":
        raise ResultError("legacy baseline results must be in final phase")

    model_argument = _typed_option(argv, "--model", str, required=True)
    model = _model_name(model_argument, manifest, plan)
    setting = _setting_name(argv, plan, method)
    expected_prefix = f"{phase}_{method}_{model}_{setting.lower()}"
    if not exp.startswith(expected_prefix):
        raise ResultError(
            "argv --exp disagrees with parsed phase/method/model/setting: "
            f"expected prefix {expected_prefix!r}, got {exp!r}"
        )

    params: dict[str, Any] = {}
    for name, kind, required in _COMMON_VALUE_OPTIONS:
        params[name] = _typed_option(
            argv, f"--{name}", kind, required=required
        )
    params["eval_datasets"] = _multi_option(argv, "--eval_datasets")
    if params["dataset"] != "wikitext2":
        raise ResultError("argv --dataset must be wikitext2")
    if params["eval_datasets"] != ["wikitext2"]:
        raise ResultError("argv --eval_datasets must be exactly wikitext2")

    if is_realq:
        for name, kind, required in _REALQ_VALUE_OPTIONS:
            params[name] = _typed_option(
                argv, f"--{name}", kind, required=required
            )
        for name in _REALQ_BOOL_OPTIONS:
            params[name] = _bool_option(
                argv, f"--{name}", required=True
            )
        if params["skip_eval"] is not False:
            raise ResultError("result argv must use --skip_eval false")
        expected_lm_eval = phase == "final"
        if params["lm_eval"] is not expected_lm_eval:
            raise ResultError(
                f"{phase} REAL-Q argv has inconsistent --lm_eval"
            )
        if params["fsdp"] is not False or params["cpu_master"] is not False:
            raise ResultError("accepted REAL-Q results must not use FSDP/cpu_master")
        if phase == "final" and params["lm_eval_batch_size"] is None:
            raise ResultError("final REAL-Q argv needs --lm_eval_batch_size")
        grad_lr = params["grad_lr"]
        final_layer_grad_lr = params["final_layer_grad_lr"]
        if grad_lr < 0 or final_layer_grad_lr < 0:
            raise ResultError("learning rates must be non-negative")
    else:
        for name in _LEGACY_BOOL_OPTIONS:
            params[name] = _bool_option(
                argv,
                f"--{name}",
                required=(name == "lm_eval"),
                legacy_flag=True,
            )
        if params["lm_eval"] is not True:
            raise ResultError("formal legacy argv must enable --lm_eval")
        if params["lm_eval_batch_size"] is None:
            raise ResultError("formal legacy argv needs --lm_eval_batch_size")
        grad_lr = None
        final_layer_grad_lr = None

    return {
        "argv": argv,
        "run_id": run_id,
        "phase": phase,
        "method": method,
        "model": model,
        "setting": setting,
        "world_size": world_size,
        "attempt_index": attempt_index,
        "grad_lr": grad_lr,
        "final_layer_grad_lr": final_layer_grad_lr,
        "params": params,
    }


def _expect_param(
    params: Mapping[str, Any], name: str, expected: Any
) -> None:
    actual = params.get(name)
    if actual != expected:
        raise ResultError(
            f"argv --{name} disagrees with embedded plan/protocol: "
            f"expected {expected!r}, got {actual!r}"
        )


def _allowed_memory_values(
    plan: Mapping[str, Any],
    *,
    model: str,
    phase: str,
    name: str,
    nominal: Any,
) -> set[Any]:
    allowed = {nominal}
    policy_value = plan.get("qwen3_32b_memory_policy")
    if policy_value is None:
        return allowed
    policy = _require_mapping(
        policy_value,
        "embedded plan.qwen3_32b_memory_policy",
    )
    ladder_key = "tuning_ladders" if phase == "tune" else "final_ladders"
    ladders = _require_mapping(
        policy.get(ladder_key),
        f"embedded plan.qwen3_32b_memory_policy.{ladder_key}",
    )
    ladder = ladders.get(name)
    if ladder is None:
        return allowed
    if (
        not isinstance(ladder, list)
        or not ladder
        or not all(type(value) is int and value > 0 for value in ladder)
    ):
        raise ResultError(
            f"embedded plan {phase} fallback ladder {name} is invalid"
        )
    return set(ladder)


def _expect_memory_param(
    params: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    model: str,
    phase: str,
    name: str,
    nominal: Any,
) -> None:
    actual = params.get(name)
    allowed = _allowed_memory_values(
        plan,
        model=model,
        phase=phase,
        name=name,
        nominal=nominal,
    )
    if actual not in allowed:
        raise ResultError(
            f"argv --{name}={actual!r} is outside embedded-plan "
            f"{phase} values {sorted(allowed)!r}"
        )


def _validate_command_protocol(
    parsed: Mapping[str, Any], plan: Mapping[str, Any]
) -> None:
    params = _require_mapping(parsed.get("params"), "parsed argv params")
    method = str(parsed["method"])
    phase = str(parsed["phase"])
    model = str(parsed["model"])
    setting_name = str(parsed["setting"])
    fixed = _require_mapping(
        plan.get("fixed_numerics"), "embedded plan.fixed_numerics"
    )
    final = _require_mapping(plan.get("final"), "embedded plan.final")
    tuning = _require_mapping(plan.get("tuning"), "embedded plan.tuning")
    phase_config = tuning if phase == "tune" else final

    if method == "bf16":
        setting: Mapping[str, Any] = {
            "w_bits": 16,
            "a_bits": 16,
            "k_bits": 16,
            "v_bits": 16,
        }
    else:
        settings = _require_mapping(
            plan.get("settings"), "embedded plan.settings"
        )
        setting = _require_mapping(
            settings.get(setting_name),
            f"embedded plan.settings.{setting_name}",
        )
    aware = method != "bf16" and any(
        setting.get(name) < 16 for name in ("a_bits", "k_bits", "v_bits")
    )

    baseline_numerics = _require_mapping(
        plan.get("baseline_numerics"),
        "embedded plan.baseline_numerics",
    )
    for name, expected in (
        ("dataset", fixed.get("dataset")),
        ("eval_seq_len", fixed.get("eval_seq_len")),
        ("w_bits", setting.get("w_bits")),
        ("a_bits", setting.get("a_bits")),
        ("k_bits", setting.get("k_bits")),
        ("v_bits", setting.get("v_bits")),
        ("a_groupsize", -1),
        ("k_groupsize", -1),
        ("v_groupsize", -1),
        (
            "a_clip_ratio",
            fixed.get("activation_clip_ratio") if aware else 1.0,
        ),
        (
            "k_clip_ratio",
            fixed.get("activation_clip_ratio") if aware else 1.0,
        ),
        (
            "v_clip_ratio",
            fixed.get("activation_clip_ratio") if aware else 1.0,
        ),
        ("num_groups", fixed.get("num_groups")),
        ("percdamp", fixed.get("percdamp")),
        ("kl_topk", fixed.get("kl_topk")),
        ("nsamples", phase_config.get("nsamples")),
        ("seq_len", phase_config.get("seq_len")),
        ("seed", fixed.get("seed")),
        ("rotation_seed", fixed.get("rotation_seed")),
        ("refresh_seed", fixed.get("refresh_seed")),
    ):
        _expect_param(params, name, expected)

    if parsed["world_size"] != (
        phase_config.get("world_size") if method == "realq" else 1
    ):
        raise ResultError(
            "argv torchrun world size disagrees with embedded plan/protocol: "
            f"got {parsed['world_size']!r}"
        )

    if method == "realq":
        expected_group = -1 if phase == "tune" else 128
        _expect_param(params, "w_groupsize", expected_group)
        _expect_param(params, "blocksize", phase_config.get("blocksize"))
        for name in (
            "bsz",
            "backward_samples",
        ):
            _expect_param(params, name, phase_config.get(name))
        for name in (
            "global_loss_bsz",
            "hessian_accum_bsz",
            "backward_bsz",
            "final_layer_backward_bsz",
        ):
            _expect_memory_param(
                params,
                plan,
                model=model,
                phase=phase,
                name=name,
                nominal=phase_config.get(name),
            )
        expected_final_lrs = _require_mapping(
            plan.get("final_layer_grad_lr_by_model"),
            "embedded plan.final_layer_grad_lr_by_model",
        )
        _expect_param(
            params, "final_layer_grad_lr", expected_final_lrs.get(model)
        )
        for name, expected in (
            ("grad_clip", fixed.get("grad_clip")),
            ("grad_lr_layer_schedule", "none" if aware else "cosine"),
            ("grad_lr_layer_base_ratio", 0.01),
            (
                "a_loss_ratio",
                fixed.get("qwen3_4b_a_loss_ratio")
                if model == "qwen3-4b"
                else fixed.get("other_model_a_loss_ratio"),
            ),
            ("a_loss_clip_scope", fixed.get("a_loss_clip_scope")),
            (
                "saliency_clip_percentile",
                fixed.get("saliency_clip_percentile"),
            ),
            ("grad_hessian_topk", fixed.get("grad_hessian_topk")),
            ("group_parallel_quant", fixed.get("group_parallel_quant")),
            ("w_clip_search_impl", fixed.get("w_clip_search_impl")),
            ("act_order_stitch_impl", fixed.get("act_order_stitch_impl")),
            ("w_clip_update_impl", fixed.get("w_clip_update_impl")),
            ("w_group_param_layout", fixed.get("w_group_param_layout")),
            ("w_asym", False),
            ("a_asym", False),
            ("k_asym", False),
            ("v_asym", False),
            ("w_clip", fixed.get("w_clip")),
            ("act_order", fixed.get("act_order")),
            ("rotate", fixed.get("rotate")),
            ("act_quant_aware_gptq", aware),
            ("k_cache_quant_aware_gptq", aware),
            ("loss_slide_window", fixed.get("loss_slide_window")),
            ("fsdp", fixed.get("fsdp")),
            ("cpu_master", fixed.get("cpu_master")),
            ("require_static_cache_hit", True),
            ("require_reference_cache_hit", True),
            ("skip_eval", False),
            ("lm_eval", phase == "final"),
            ("log_column_block_loss", True),
            (
                "quantizer_inner_fastpath",
                fixed.get("quantizer_inner_fastpath"),
            ),
            ("fisher_fp32_cache", fixed.get("fisher_fp32_cache")),
        ):
            _expect_param(params, name, expected)
        expected_static_cache = (
            Path(str(plan.get("static_cache_root")))
            / model
            / (
                f"{phase}_world{parsed['world_size']}_"
                f"glbsz{params['global_loss_bsz']}"
            )
        )
        _expect_param(params, "static_cache_path", str(expected_static_cache))
        if phase == "tune":
            _expect_param(params, "lm_eval_batch_size", None)
        else:
            selected_ledger = _require_mapping(
                plan.get("selected_grad_lr_by_model_setting"),
                "embedded plan.selected_grad_lr_by_model_setting",
            )
            selected_by_setting = _require_mapping(
                selected_ledger.get(model),
                (
                    "embedded plan.selected_grad_lr_by_model_setting."
                    f"{model}"
                ),
            )
            selected_lr = selected_by_setting.get(setting_name)
            if (
                not _is_number(selected_lr)
                or not math.isfinite(float(selected_lr))
                or float(selected_lr) < 0
            ):
                raise ResultError(
                    "formal REAL-Q embedded selected LR must be a "
                    "non-negative finite number"
                )
            _expect_param(params, "grad_lr", float(selected_lr))
            _expect_memory_param(
                params,
                plan,
                model=model,
                phase="final",
                name="lm_eval_batch_size",
                nominal=final.get("lm_eval_batch_size"),
            )
        return

    _expect_param(params, "blocksize", final.get("blocksize"))
    _expect_param(params, "w_groupsize", -1 if method == "bf16" else 128)
    _expect_memory_param(
        params,
        plan,
        model=model,
        phase="final",
        name="lm_eval_batch_size",
        nominal=final.get("lm_eval_batch_size"),
    )
    expected_w_method = {
        "gptaq": "gptaq",
        "guided_gptq": "gptq_guided",
        "bf16": None,
    }[method]
    _expect_param(params, "w_method", expected_w_method)
    _expect_param(
        params,
        "alpha",
        (
            _require_mapping(
                plan.get("baseline_numerics"),
                "embedded plan.baseline_numerics",
            ).get("gptaq_alpha")
            if method == "gptaq"
            else None
        ),
    )
    for name, expected in (
        ("lm_eval", True),
        ("w_clip", None if method == "bf16" else fixed.get("w_clip")),
        ("act_order", None if method == "bf16" else fixed.get("act_order")),
        ("rotate", None if method == "bf16" else fixed.get("rotate")),
        (
            "offload_inps",
            None if method == "bf16" else baseline_numerics.get("offload_inps"),
        ),
        ("act_quant_aware_gptq", True if aware else None),
        ("k_cache_quant_aware_gptq", True if aware else None),
    ):
        _expect_param(params, name, expected)


def _pipe_cells(line: str) -> tuple[str, ...] | None:
    first = line.find("|")
    last = line.rfind("|")
    if first < 0 or last <= first:
        return None
    body = line[first + 1 : last]
    cells = tuple(cell.strip() for cell in body.split("|"))
    if not cells or any(not cell for cell in cells):
        return None
    return cells


def _markdown_tables(text: str) -> list[MarkdownTable]:
    rows = [_pipe_cells(line) for line in text.splitlines()]
    tables: list[MarkdownTable] = []
    for index in range(len(rows) - 2):
        header, separator, values = rows[index : index + 3]
        if header is None or separator is None or values is None:
            continue
        if not (
            len(header) == len(separator) == len(values)
            and all(_SEPARATOR_RE.fullmatch(cell) for cell in separator)
        ):
            continue
        tables.append(MarkdownTable(header, values))
    return tables


def _parse_metrics_log(
    text: str, *, phase: str, method: str
) -> tuple[float | None, float, dict[str, float], float | None]:
    tables = _markdown_tables(text)
    metric_tables = [
        table
        for table in tables
        if any(
            header.startswith("KL-") or header.startswith("PPL-")
            for header in table.headers
        )
    ]
    if len(metric_tables) != 1:
        raise ResultError(
            "execution.log must contain exactly one KL/PPL Markdown table; "
            f"found {len(metric_tables)}"
        )
    metric_table = metric_tables[0]
    dual_table = metric_table.headers == (
        "KL-wikitext2",
        "PPL-wikitext2",
    )
    ppl_only_table = metric_table.headers == ("PPL-wikitext2",)
    if not dual_table and not (method == "bf16" and ppl_only_table):
        raise ResultError(
            "metric table must contain canonical WikiText-2 KL/PPL, or "
            "PPL-only for BF16"
        )
    table_kl_text = metric_table.values[0] if dual_table else None
    table_ppl_text = metric_table.values[1] if dual_table else metric_table.values[0]
    table_kl = (
        _finite_float(str(table_kl_text), "display KL-wikitext2")
        if table_kl_text is not None
        else None
    )
    table_ppl = _finite_float(table_ppl_text, "display PPL-wikitext2")
    if table_kl is not None and table_kl < 0:
        raise ResultError("display KL-wikitext2 must be non-negative")
    if table_ppl <= 0:
        raise ResultError("display PPL-wikitext2 must be positive")

    details = list(_KL_DETAIL_RE.finditer(text))
    exact_details = [match for match in details if match.group(1)]
    if len(exact_details) != 1:
        raise ResultError(
            "execution.log must contain exactly one Exact KL&PPL line; "
            f"found {len(exact_details)}"
        )
    for match in details:
        _, dataset, detail_kl_text, detail_ppl_text = match.groups()
        if dataset.lower() != "wikitext2":
            raise ResultError(
                f"log contains KL/PPL for forbidden dataset {dataset!r}"
            )
        if match.group(1):
            continue
        detail_ppl = _finite_float(detail_ppl_text, "detail PPL-wikitext2")
        detail_kl = (
            None
            if method == "bf16"
            else _finite_float(detail_kl_text, "detail KL-wikitext2")
        )
        if (
            detail_kl is not None
            and table_kl is not None
            and format(detail_kl, ".2e").lower()
            != format(table_kl, ".2e").lower()
        ) or format(detail_ppl, ".2f") != format(table_ppl, ".2f"):
            raise ResultError(
                "rounded KL/PPL detail line disagrees with Markdown table"
            )
    exact = exact_details[0]
    exact_kl = (
        None
        if method == "bf16"
        else _finite_float(exact.group(3), "exact KL-wikitext2")
    )
    exact_ppl = _finite_float(exact.group(4), "exact PPL-wikitext2")
    if exact_kl is not None and exact_kl < 0:
        raise ResultError("exact KL-wikitext2 must be non-negative")
    if exact_ppl <= 0:
        raise ResultError("exact PPL-wikitext2 must be positive")
    if (
        exact_kl is not None
        and table_kl_text is not None
        and str(table_kl_text).lower() != format(exact_kl, ".2e").lower()
    ):
        raise ResultError(
            "exact KL-wikitext2 does not round to the Markdown table value"
        )
    if table_ppl_text != format(exact_ppl, ".2f"):
        raise ResultError(
            "exact PPL-wikitext2 does not round to the Markdown table value"
        )
    kl = exact_kl
    ppl = exact_ppl

    qa_headers = EXPECTED_TASKS + ("acc_avg",)
    qa_like = [
        table
        for table in tables
        if "acc_avg" in table.headers
        or any(task in table.headers for task in EXPECTED_TASKS)
    ]
    if len(qa_like) > 1:
        raise ResultError(
            f"execution.log contains {len(qa_like)} QA result tables"
        )
    if qa_like and qa_like[0].headers != qa_headers:
        raise ResultError(
            "QA table must contain the exact paper ten tasks plus acc_avg "
            "in canonical order"
        )
    if phase == "final" and len(qa_like) != 1:
        raise ResultError(
            "formal/final execution.log must contain the complete paper QA table"
        )
    if not qa_like:
        return kl, ppl, {}, None

    table = qa_like[0]
    task_values = {
        task: _finite_float(value, f"QA {task}")
        for task, value in zip(EXPECTED_TASKS, table.values[:-1])
    }
    if any(value < 0 or value > 100 for value in task_values.values()):
        raise ResultError("QA accuracies must be between 0 and 100")
    acc_avg = _finite_float(table.values[-1], "QA acc_avg")
    if acc_avg < 0 or acc_avg > 100:
        raise ResultError("QA acc_avg must be between 0 and 100")
    recomputed = sum(task_values.values()) / len(EXPECTED_TASKS)
    if abs(recomputed - acc_avg) > 0.011:
        raise ResultError(
            "QA acc_avg disagrees with the mean of the ten task values: "
            f"reported={acc_avg}, recomputed={recomputed}"
        )
    return kl, ppl, task_values, acc_avg


def parse_result(manifest_path: Path) -> ParsedResult:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResultError(f"cannot read manifest JSON: {exc}") from exc
    manifest = _require_mapping(manifest, "manifest")
    identities, plan = _manifest_identities(manifest)
    lm_eval_version = _validate_python_packages(manifest)
    parsed_argv = _parse_argv(manifest, plan)
    _validate_command_protocol(parsed_argv, plan)

    log_path = manifest_path.parent / LOG_FILENAME
    if not log_path.is_file():
        raise ResultError(f"missing sibling {LOG_FILENAME}")
    try:
        log_text = log_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ResultError(f"cannot read execution.log: {exc}") from exc
    kl, ppl, tasks, acc_avg = _parse_metrics_log(
        log_text,
        phase=parsed_argv["phase"],
        method=parsed_argv["method"],
    )

    tuning = _require_mapping(plan.get("tuning"), "embedded plan.tuning")
    candidates_value = tuning.get("lr_candidates")
    if (
        not isinstance(candidates_value, list)
        or not candidates_value
        or not all(_is_number(value) for value in candidates_value)
    ):
        raise ResultError(
            "embedded plan.tuning.lr_candidates must be a non-empty number list"
        )
    expected_candidates = tuple(float(value) for value in candidates_value)
    if any(not math.isfinite(value) or value < 0 for value in expected_candidates):
        raise ResultError(
            "embedded plan tuning LR candidates must be finite and non-negative"
        )
    if len(set(expected_candidates)) != len(expected_candidates):
        raise ResultError(
            "embedded plan tuning LR candidates must not contain duplicates"
        )

    execution_id = _require_nonempty_string(
        manifest.get("execution_id"), "manifest.execution_id"
    )
    return ParsedResult(
        manifest_path=manifest_path.resolve(),
        log_path=log_path.resolve(),
        execution_id=execution_id,
        run_id=parsed_argv["run_id"],
        model=parsed_argv["model"],
        setting=parsed_argv["setting"],
        method=parsed_argv["method"],
        phase=parsed_argv["phase"],
        world_size=parsed_argv["world_size"],
        attempt_index=parsed_argv["attempt_index"],
        grad_lr=parsed_argv["grad_lr"],
        final_layer_grad_lr=parsed_argv["final_layer_grad_lr"],
        kl_wikitext2=kl,
        ppl_wikitext2=ppl,
        tasks=tasks,
        acc_avg=acc_avg,
        lm_eval_version=lm_eval_version,
        params=parsed_argv["params"],
        identities=identities,
        expected_lr_candidates=expected_candidates,
        max_attempts=int(tuning["max_attempts_per_model_setting"]),
        search_policy=dict(tuning["search_policy"]),
        plan_content=dict(plan),
    )


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _strict_canonical_sha256(value: Any, name: str) -> str:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ResultError(f"{name} is not canonical JSON: {exc}") from exc
    return hashlib.sha256(raw).hexdigest()


def _strict_canonical_json_bytes(value: Any, name: str) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ResultError(f"{name} is not canonical JSON: {exc}") from exc


def _current_numerical_source_sha256() -> str:
    """Recompute the executor's trusted in-repository numerical source hash."""

    repo_root = Path(__file__).resolve().parents[1]
    candidates: set[Path] = set()
    for directory_name in ("realq", "gptq_utils", "utils"):
        directory = repo_root / directory_name
        if directory.is_dir():
            candidates.update(
                path.resolve(strict=True)
                for path in directory.rglob("*.py")
                if path.is_file()
            )
    for filename in ("process_args.py", "ptq.py", "save_grads.py"):
        path = repo_root / filename
        if path.is_file():
            candidates.add(path.resolve(strict=True))
    if not candidates:
        raise ResultError("current numerical source tree is empty")

    records: list[dict[str, str]] = []
    for path in sorted(candidates):
        try:
            before = path.stat()
            sha256 = _sha256_file(path)
            after = path.stat()
        except OSError as exc:
            raise ResultError(
                f"cannot establish current numerical source identity: {exc}"
            ) from exc
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise ResultError(
                f"current numerical source changed while hashing: {path}"
            )
        records.append(
            {
                "relative_path": path.relative_to(repo_root).as_posix(),
                "sha256": sha256,
            }
        )
    return _strict_canonical_sha256(
        records, "current numerical source tree"
    )


def _current_trusted_base_source_identity() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[1]
    return {
        "numerical_source_sha256": _current_numerical_source_sha256(),
        "runner_sha256": _sha256_file(
            repo_root / "tools" / "lowbit_activation_runner.py"
        ),
        "executor_sha256": _sha256_file(
            repo_root / "tools" / "lowbit_activation_execute.py"
        ),
        "campaign_sha256": _sha256_file(
            repo_root / "tools" / "lowbit_activation_campaign.py"
        ),
        "results_sha256": _sha256_file(Path(__file__).resolve()),
    }


def _manual_lr_file_ref(
    path: Path, *, root: Path, name: str
) -> dict[str, Any]:
    resolved = path.resolve(strict=False)
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ResultError(f"{name} escapes the result root: {resolved}") from exc
    if not resolved.exists():
        result = {
            "absolute_path": str(resolved),
            "relative_path": relative,
            "present": False,
            "sha256": None,
            "size_bytes": None,
        }
    else:
        if not resolved.is_file():
            raise ResultError(f"{name} must be a regular file: {resolved}")
        try:
            before = resolved.stat()
            raw = resolved.read_bytes()
            after = resolved.stat()
        except OSError as exc:
            raise ResultError(f"cannot read {name} {resolved}: {exc}") from exc
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
            or len(raw) != after.st_size
        ):
            raise ResultError(f"{name} changed while being read: {resolved}")
        result = {
            "absolute_path": str(resolved),
            "relative_path": relative,
            "present": True,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
    if set(result) != _MANUAL_LR_FILE_REF_KEYS:
        raise AssertionError("internal manual LR file reference drift")
    return result


def _manual_lr_raw_input_prefixes(root: Path) -> dict[str, dict[str, Any]]:
    fixed_paths = {
        "retirement_ledger": RETIREMENT_LEDGER_RELATIVE_PATH,
        "campaign_events": CAMPAIGN_EVENTS_RELATIVE_PATH,
        "manual_invalid_results": MANUAL_INVALID_LEDGER_RELATIVE_PATH,
    }
    result: dict[str, dict[str, Any]] = {}
    for key, relative in fixed_paths.items():
        path = (root / relative).resolve(strict=False)
        try:
            canonical_relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ResultError(
                f"manual LR raw input {key} escapes the result root"
            ) from exc
        if canonical_relative != relative.as_posix():
            raise ResultError(
                f"manual LR raw input {key} path is not canonical"
            )
        if not path.exists():
            reference = {
                "relative_path": relative.as_posix(),
                "present": False,
                "prefix_sha256": None,
                "prefix_size_bytes": None,
            }
        else:
            if not path.is_file():
                raise ResultError(
                    f"manual LR raw input {key} must be a regular file"
                )
            raw = path.read_bytes()
            reference = {
                "relative_path": relative.as_posix(),
                "present": True,
                "prefix_sha256": hashlib.sha256(raw).hexdigest(),
                "prefix_size_bytes": len(raw),
            }
        result[key] = reference
    return result


def _validate_manual_lr_raw_input_prefixes(
    root: Path, value: Any
) -> dict[str, dict[str, Any]]:
    prefixes = _require_mapping(
        value, "manual LR freeze raw_input_prefixes"
    )
    _require_exact_keys(
        prefixes,
        _MANUAL_LR_RAW_INPUT_PREFIX_KEYS,
        "manual LR freeze raw_input_prefixes",
    )
    fixed_paths = {
        "retirement_ledger": RETIREMENT_LEDGER_RELATIVE_PATH,
        "campaign_events": CAMPAIGN_EVENTS_RELATIVE_PATH,
        "manual_invalid_results": MANUAL_INVALID_LEDGER_RELATIVE_PATH,
    }
    normalized: dict[str, dict[str, Any]] = {}
    for key, relative in fixed_paths.items():
        name = f"manual LR freeze raw_input_prefixes.{key}"
        reference = _require_mapping(prefixes.get(key), name)
        _require_exact_keys(
            reference, _MANUAL_LR_RAW_PREFIX_REF_KEYS, name
        )
        if reference.get("relative_path") != relative.as_posix():
            raise ResultError(f"{name}.relative_path is invalid")
        present = reference.get("present")
        if type(present) is not bool:
            raise ResultError(f"{name}.present must be boolean")
        if present:
            prefix_sha = _require_sha256(
                reference.get("prefix_sha256"),
                f"{name}.prefix_sha256",
            )
            size = reference.get("prefix_size_bytes")
            if type(size) is not int or size < 0:
                raise ResultError(
                    f"{name}.prefix_size_bytes must be non-negative"
                )
        else:
            if (
                reference.get("prefix_sha256") is not None
                or reference.get("prefix_size_bytes") is not None
            ):
                raise ResultError(
                    f"{name} absent prefix must use null hash and size"
                )
            prefix_sha = None
            size = 0
        path = (root / relative).resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ResultError(f"{name} escapes the result root") from exc
        if present:
            if not path.is_file():
                raise ResultError(
                    f"{name} freeze-time file is now unavailable"
                )
            try:
                raw = path.read_bytes()
            except OSError as exc:
                raise ResultError(f"cannot read {name}: {exc}") from exc
            if (
                len(raw) < size
                or hashlib.sha256(raw[:size]).hexdigest() != prefix_sha
            ):
                raise ResultError(
                    f"{name} historical prefix was rewritten or truncated"
                )
        normalized[key] = dict(reference)
    return normalized


def _current_formal_timing_gate() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    experiment_dir = repo_root / "experiments" / "lowbit_activation"
    source_paths = (
        (
            "formal_timing_adapter",
            experiment_dir / "formal_timing_adapter.py",
        ),
        (
            "formal_timing_sitecustomize",
            experiment_dir
            / "formal_timing_sitecustomize"
            / "sitecustomize.py",
        ),
        (
            "formal_timed_execute",
            experiment_dir / "formal_timed_execute.py",
        ),
        (
            "lowbit_activation_campaign",
            repo_root / "tools" / "lowbit_activation_campaign.py",
        ),
        (
            "formal_timed_campaign",
            experiment_dir / "formal_timed_campaign.py",
        ),
        (
            "lowbit_activation_gpu_hours",
            repo_root / "tools" / "lowbit_activation_gpu_hours.py",
        ),
        (
            "validate_guided_saliency",
            repo_root / "tools" / "validate_guided_saliency.py",
        ),
    )
    sources = [
        {
            "name": name,
            "original_path": str(path.resolve(strict=True)),
            "sha256": _sha256_file(path.resolve(strict=True)),
        }
        for name, path in source_paths
    ]
    by_name = {item["name"]: item for item in sources}
    source_set_sha = _strict_canonical_sha256(
        [
            {"name": item["name"], "sha256": item["sha256"]}
            for item in sources
        ],
        "current formal timing source set",
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "controller_path": by_name["formal_timed_campaign"][
            "original_path"
        ],
        "controller_sha256": by_name["formal_timed_campaign"]["sha256"],
        "base_campaign_path": by_name["lowbit_activation_campaign"][
            "original_path"
        ],
        "base_campaign_sha256": by_name["lowbit_activation_campaign"][
            "sha256"
        ],
        "timed_executor_path": by_name["formal_timed_execute"][
            "original_path"
        ],
        "timed_executor_sha256": by_name["formal_timed_execute"]["sha256"],
        "gpu_hour_aggregator_sha256": by_name[
            "lowbit_activation_gpu_hours"
        ]["sha256"],
        "guided_validator_sha256": by_name[
            "validate_guided_saliency"
        ]["sha256"],
        "source_set_sha256": source_set_sha,
        "sources": sources,
    }
    payload["gate_sha256"] = _strict_canonical_sha256(
        payload, "current formal timing gate"
    )
    return payload


def _manual_lr_freeze_selection(
    plan: Mapping[str, Any],
    *,
    model: str,
    setting: str,
) -> dict[str, Any] | None:
    """Validate and return this row's immutable manual-LR decision.

    The wrapper is embedded in the formal plan, so result validation does not
    depend on mutable campaign state or a sidecar remaining available.
    """

    wrapper_value = plan.get("manual_lr_selection_freeze")
    if wrapper_value is None:
        return None
    wrapper = _require_mapping(
        wrapper_value, "embedded plan.manual_lr_selection_freeze"
    )
    _require_exact_keys(
        wrapper,
        _MANUAL_LR_FREEZE_WRAPPER_KEYS,
        "embedded plan.manual_lr_selection_freeze",
    )
    if wrapper.get("schema_version") != 1:
        raise ResultError("manual LR freeze wrapper schema_version must be 1")
    if wrapper.get("scope_id") != MANUAL_LR_FREEZE_SCOPE_ID:
        raise ResultError("manual LR freeze wrapper scope_id is invalid")
    wrapper_sha = _require_sha256(
        wrapper.get("ledger_sha256"),
        "manual LR freeze wrapper ledger_sha256",
    )
    artifact_ref = _require_mapping(
        wrapper.get("artifact"), "manual LR freeze artifact reference"
    )
    _require_exact_keys(
        artifact_ref,
        _MANUAL_LR_FREEZE_ARTIFACT_REF_KEYS,
        "manual LR freeze artifact reference",
    )
    if (
        artifact_ref.get("filename")
        != MANUAL_LR_FREEZE_ARTIFACT_FILENAME
    ):
        raise ResultError("manual LR freeze artifact filename is invalid")
    _require_sha256(
        artifact_ref.get("sha256"),
        "manual LR freeze artifact sha256",
    )
    if (
        type(artifact_ref.get("size_bytes")) is not int
        or artifact_ref["size_bytes"] <= 0
    ):
        raise ResultError(
            "manual LR freeze artifact size_bytes must be positive"
        )
    ledger = _require_mapping(
        wrapper.get("ledger"), "manual LR freeze ledger"
    )
    _require_exact_keys(
        ledger,
        _MANUAL_LR_FREEZE_LEDGER_KEYS,
        "manual LR freeze ledger",
    )
    if _strict_canonical_sha256(ledger, "manual LR freeze ledger") != wrapper_sha:
        raise ResultError("manual LR freeze ledger_sha256 mismatch")
    if ledger.get("schema_version") != 1:
        raise ResultError("manual LR freeze ledger schema_version must be 1")
    if ledger.get("scope_id") != MANUAL_LR_FREEZE_SCOPE_ID:
        raise ResultError("manual LR freeze ledger scope_id is invalid")
    _require_nonempty_string(
        ledger.get("campaign_id"), "manual LR freeze campaign_id"
    )
    for field in (
        "selection_sha256",
        "expected_state_sha256",
        "expected_plan_sha256",
        "timing_gate_sha256",
        "timing_source_set_sha256",
    ):
        _require_sha256(
            ledger.get(field), f"manual LR freeze ledger {field}"
        )
    adoption_value = ledger.get("results_source_adoption")
    if adoption_value is not None:
        adoption = _require_mapping(
            adoption_value,
            "manual LR freeze results_source_adoption",
        )
        _require_exact_keys(
            adoption,
            _MANUAL_LR_FREEZE_ADOPTION_KEYS,
            "manual LR freeze results_source_adoption",
        )
        if adoption.get("kind") != MANUAL_LR_RESULTS_ADOPTION_KIND:
            raise ResultError(
                "manual LR freeze results source adoption kind is invalid"
            )
        for field in (
            "expected_old_results_sha256",
            "accepted_current_results_sha256",
        ):
            _require_sha256(
                adoption.get(field),
                f"manual LR freeze results_source_adoption.{field}",
            )

    source_identity = _require_mapping(
        ledger.get("source_identity_before"),
        "manual LR freeze source_identity_before",
    )
    expected_source_keys = {
        "numerical_source_sha256",
        "runner_sha256",
        "executor_sha256",
        "campaign_sha256",
        "results_sha256",
    }
    _require_exact_keys(
        source_identity,
        expected_source_keys,
        "manual LR freeze source_identity_before",
    )
    for key in expected_source_keys:
        _require_sha256(
            source_identity.get(key),
            f"manual LR freeze source_identity_before.{key}",
        )
    current_source_identity = _current_trusted_base_source_identity()
    if dict(source_identity) != current_source_identity:
        drifted = sorted(
            key
            for key in expected_source_keys
            if source_identity[key] != current_source_identity[key]
        )
        raise ResultError(
            "manual LR freeze source_identity_before disagrees with current "
            f"trusted source(s): {drifted!r}"
        )

    gate = _require_mapping(
        ledger.get("downstream_required_gate"),
        "manual LR freeze downstream_required_gate",
    )
    _require_exact_keys(
        gate,
        _MANUAL_LR_FREEZE_GATE_KEYS,
        "manual LR freeze downstream_required_gate",
    )
    if gate.get("gate_id") != MANUAL_LR_FREEZE_GATE_ID:
        raise ResultError("manual LR freeze downstream gate_id is invalid")
    if type(gate.get("required")) is not bool:
        raise ResultError("manual LR freeze downstream required must be boolean")

    selections_value = ledger.get("selections")
    if not isinstance(selections_value, list) or not selections_value:
        raise ResultError("manual LR freeze selections must be a non-empty list")
    normalized: dict[tuple[str, str], dict[str, Any]] = {}
    selection_projection: list[dict[str, Any]] = []
    any_override = False
    for index, selection_value in enumerate(selections_value):
        name = f"manual LR freeze selections[{index}]"
        selection = _require_mapping(selection_value, name)
        _require_exact_keys(
            selection, _MANUAL_LR_FREEZE_SELECTION_KEYS, name
        )
        selection_model = _require_nonempty_string(
            selection.get("model"), f"{name}.model"
        )
        selection_setting = _require_nonempty_string(
            selection.get("setting"), f"{name}.setting"
        )
        selected_lr_value = selection.get("selected_lr")
        if (
            not _is_number(selected_lr_value)
            or not math.isfinite(float(selected_lr_value))
            or float(selected_lr_value) < 0
        ):
            raise ResultError(
                f"{name}.selected_lr must be finite and non-negative"
            )
        decision_kind = selection.get("decision_kind")
        if decision_kind not in {"exact", "user_override_interpolated"}:
            raise ResultError(f"{name}.decision_kind is invalid")
        any_override = (
            any_override or decision_kind == "user_override_interpolated"
        )
        _require_nonempty_string(
            selection.get("rationale"), f"{name}.rationale"
        )
        selected_manifest = selection.get("selected_manifest")
        if decision_kind == "exact":
            _require_nonempty_string(
                selected_manifest, f"{name}.selected_manifest"
            )
        elif selected_manifest is not None:
            raise ResultError(
                f"{name}.selected_manifest must be null for an "
                "interpolated override"
            )
        evidence = selection.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ResultError(f"{name}.evidence must be a non-empty list")
        expected_evidence_count = (
            1 if decision_kind == "exact" else 2
        )
        if len(evidence) != expected_evidence_count:
            raise ResultError(
                f"{name}.evidence must contain exactly "
                f"{expected_evidence_count} item(s) for {decision_kind}"
            )
        evidence_sha = _require_sha256(
            selection.get("evidence_set_sha256"),
            f"{name}.evidence_set_sha256",
        )
        if (
            _strict_canonical_sha256(evidence, f"{name}.evidence")
            != evidence_sha
        ):
            raise ResultError(f"{name}.evidence_set_sha256 mismatch")
        evidence_manifests: set[str] = set()
        for evidence_index, evidence_value in enumerate(evidence):
            evidence_name = f"{name}.evidence[{evidence_index}]"
            item = _require_mapping(evidence_value, evidence_name)
            _require_exact_keys(
                item, _MANUAL_LR_FREEZE_EVIDENCE_KEYS, evidence_name
            )
            manifest_path = _require_nonempty_string(
                item.get("manifest"), f"{evidence_name}.manifest"
            )
            log_path = _require_nonempty_string(
                item.get("log"), f"{evidence_name}.log"
            )
            if not Path(manifest_path).is_absolute():
                raise ResultError(
                    f"{evidence_name}.manifest must be an absolute path"
                )
            if not Path(log_path).is_absolute():
                raise ResultError(
                    f"{evidence_name}.log must be an absolute path"
                )
            if manifest_path in evidence_manifests:
                raise ResultError(
                    f"{name}.evidence contains a duplicate manifest"
                )
            evidence_manifests.add(manifest_path)
            _require_sha256(
                item.get("manifest_sha256"),
                f"{evidence_name}.manifest_sha256",
            )
            _require_sha256(
                item.get("log_sha256"), f"{evidence_name}.log_sha256"
            )
            _require_nonempty_string(
                item.get("execution_id"), f"{evidence_name}.execution_id"
            )
            _require_nonempty_string(
                item.get("run_id"), f"{evidence_name}.run_id"
            )
            if (
                type(item.get("attempt_index")) is not int
                or item["attempt_index"] <= 0
            ):
                raise ResultError(
                    f"{evidence_name}.attempt_index must be positive"
                )
            for field in ("grad_lr", "kl_wikitext2"):
                value = item.get(field)
                if not _is_number(value) or not math.isfinite(float(value)):
                    raise ResultError(
                        f"{evidence_name}.{field} must be finite"
                    )
            if float(item["grad_lr"]) < 0:
                raise ResultError(
                    f"{evidence_name}.grad_lr must be non-negative"
                )
            for field in (
                "plan_sha256",
                "model_sha256",
                "runner_sha256",
                "executor_sha256",
                "numerical_source_sha256",
            ):
                _require_sha256(
                    item.get(field), f"{evidence_name}.{field}"
                )
        if decision_kind == "user_override_interpolated":
            reviewed = REVIEWED_INTERPOLATED_OVERRIDE
            if (
                selection_model != reviewed["model"]
                or selection_setting != reviewed["setting"]
                or float(selection["selected_lr"])
                != reviewed["selected_lr"]
                or tuple(float(item["grad_lr"]) for item in evidence)
                != reviewed["evidence_lrs"]
            ):
                raise ResultError(
                    f"{name} is not the uniquely reviewed interpolated "
                    "override"
                )
        if (
            decision_kind == "exact"
            and selected_manifest not in evidence_manifests
        ):
            raise ResultError(
                f"{name}.selected_manifest is absent from its evidence"
            )
        key = (selection_model, selection_setting)
        if key in normalized:
            raise ResultError(
                "manual LR freeze contains duplicate model/setting selection "
                f"{selection_model}/{selection_setting}"
            )
        normalized[key] = dict(selection)
        selection_projection.append(
            {
                "model": selection_model,
                "setting": selection_setting,
                "selected_lr": selection["selected_lr"],
                "decision_kind": decision_kind,
                "evidence": [
                    {
                        "manifest": item["manifest"],
                        "grad_lr": item["grad_lr"],
                        "kl_wikitext2": item["kl_wikitext2"],
                    }
                    for item in evidence
                ],
                "rationale": selection["rationale"],
            }
        )

    models = _require_mapping(plan.get("models"), "embedded plan.models")
    settings = _require_mapping(plan.get("settings"), "embedded plan.settings")
    expected_pair_order = [
        (str(model_name), str(setting_name))
        for model_name in models
        for setting_name in settings
    ]
    if list(normalized) != expected_pair_order:
        raise ResultError(
            "manual LR freeze selections do not match the ordered plan matrix"
        )
    reviewed_override_key = (
        REVIEWED_INTERPOLATED_OVERRIDE["model"],
        REVIEWED_INTERPOLATED_OVERRIDE["setting"],
    )
    if (
        reviewed_override_key in normalized
        and normalized[reviewed_override_key]["decision_kind"]
        != "user_override_interpolated"
    ):
        raise ResultError(
            "the reviewed qwen3-4b/2W16A row must use its frozen "
            "interpolated override"
        )
    selection_root = {
        "schema_version": 1,
        "selections": selection_projection,
    }
    if (
        _strict_canonical_sha256(
            selection_root, "manual LR freeze selection projection"
        )
        != ledger["selection_sha256"]
    ):
        raise ResultError("manual LR freeze selection_sha256 mismatch")
    required = bool(gate["required"])
    expected_gate_status = (
        "required_before_formal_results_acceptance"
        if any_override
        else "not_required"
    )
    if required != any_override or gate.get("status") != expected_gate_status:
        raise ResultError(
            "manual LR freeze downstream gate disagrees with decision kinds"
        )
    return {
        "wrapper_ledger_sha256": wrapper_sha,
        "artifact": dict(artifact_ref),
        "ledger": dict(ledger),
        "source_identity_before": dict(source_identity),
        "selection": normalized[(model, setting)],
    }


def _frozen_plan_protocol_sha256(plan: Mapping[str, Any]) -> str:
    """Hash protocol fields, excluding reviewed formal LR freeze metadata."""

    normalized = dict(plan)
    normalized.pop("selected_grad_lr_by_model_setting", None)
    normalized.pop("manual_lr_selection_freeze", None)
    return _canonical_sha256(normalized)


def _load_json_mapping_bytes(raw: bytes, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResultError(f"{name} is not valid UTF-8 JSON: {exc}") from exc
    return _require_mapping(value, name)


def _manual_lr_reconciliation_config_sha256(
    record: ParsedResult,
) -> str:
    return _strict_canonical_sha256(
        {
            "world_size": record.world_size,
            "final_layer_grad_lr": record.final_layer_grad_lr,
            "params": record.params,
            "max_attempts": record.max_attempts,
            "search_policy": record.search_policy,
        },
        "manual LR reconciliation tune configuration",
    )


def _manual_lr_manifest_launch_identity(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> dict[str, Any]:
    context = _tune_attempt_context(manifest, manifest_path)
    if context is None:
        raise ResultError(
            f"manifest is not an identifiable REAL-Q tune launch: "
            f"{manifest_path}"
        )
    model = _require_nonempty_string(
        context.get("model"), "tune launch model"
    )
    setting = _require_nonempty_string(
        context.get("setting"), "tune launch setting"
    )
    grad_lr = context.get("grad_lr")
    if (
        not _is_number(grad_lr)
        or not math.isfinite(float(grad_lr))
        or float(grad_lr) < 0
    ):
        raise ResultError("tune launch grad_lr must be finite and non-negative")
    attempt_index = context.get("attempt_index")
    if (
        type(attempt_index) is not int
        or attempt_index <= 0
        or attempt_index > 20
    ):
        raise ResultError("tune launch attempt_index is outside [1,20]")
    plan_wrapper = _require_mapping(
        manifest.get("plan"), "tune launch manifest.plan"
    )
    model_wrapper = _require_mapping(
        manifest.get("model"), "tune launch manifest.model"
    )
    source_files = _require_mapping(
        manifest.get("source_files"),
        "tune launch manifest.source_files",
    )
    runner_source = _require_mapping(
        source_files.get("runner"),
        "tune launch manifest.source_files.runner",
    )
    executor_source = _require_mapping(
        source_files.get("executor"),
        "tune launch manifest.source_files.executor",
    )
    numerical = _require_mapping(
        manifest.get("numerical_source_tree"),
        "tune launch manifest.numerical_source_tree",
    )
    status = _require_nonempty_string(
        manifest.get("status"), "tune launch status"
    )
    exit_code = manifest.get("exit_code")
    if exit_code is not None and type(exit_code) is not int:
        raise ResultError("tune launch exit_code must be integer or null")
    context_valid = context.get("context_valid")
    if type(context_valid) is not bool:
        raise ResultError("tune launch context_valid must be boolean")
    end_sha = numerical.get("combined_sha256_at_end")
    if end_sha is not None:
        end_sha = _require_sha256(
            end_sha, "tune launch numerical source end SHA256"
        )
    changed = numerical.get("changed_during_execution")
    if type(changed) is not bool:
        raise ResultError(
            "tune launch numerical source changed flag must be boolean"
        )
    return {
        "execution_id": _require_nonempty_string(
            manifest.get("execution_id"), "tune launch execution_id"
        ),
        "run_id": _require_nonempty_string(
            manifest.get("run_id"), "tune launch run_id"
        ),
        "status": status,
        "exit_code": exit_code,
        "model": model,
        "setting": setting,
        "grad_lr": float(grad_lr),
        "attempt_index": attempt_index,
        "context_valid": context_valid,
        "plan_sha256": _require_sha256(
            plan_wrapper.get("sha256"), "tune launch plan_sha256"
        ),
        "model_sha256": _require_sha256(
            model_wrapper.get("combined_identity_sha256"),
            "tune launch model_sha256",
        ),
        "runner_sha256": _require_sha256(
            runner_source.get("sha256"), "tune launch runner_sha256"
        ),
        "executor_sha256": _require_sha256(
            executor_source.get("sha256"), "tune launch executor_sha256"
        ),
        "numerical_source_sha256_at_start": _require_sha256(
            numerical.get("combined_sha256"),
            "tune launch numerical source start SHA256",
        ),
        "numerical_source_sha256_at_end": end_sha,
        "numerical_source_changed_during_execution": changed,
    }


def _manual_lr_campaign_events_by_seq(
    root: Path,
) -> dict[int, tuple[Mapping[str, Any], str]]:
    path = root / CAMPAIGN_EVENTS_RELATIVE_PATH
    if not path.is_file():
        return {}
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ResultError(f"cannot read campaign events: {exc}") from exc
    if raw and not raw.endswith(b"\n"):
        raise ResultError("campaign events must end with LF")
    events: dict[int, tuple[Mapping[str, Any], str]] = {}
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line:
            raise ResultError("campaign events must not contain blank lines")
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResultError(
                f"campaign events line {line_number} is invalid JSON"
            ) from exc
        event = _require_mapping(
            value, f"campaign events line {line_number}"
        )
        sequence = event.get("seq")
        if (
            type(sequence) is not int
            or sequence <= 0
            or sequence in events
        ):
            raise ResultError(
                f"campaign events line {line_number} has invalid/duplicate seq"
            )
        events[sequence] = (
            event,
            hashlib.sha256(line).hexdigest(),
        )
    return events


def _load_manual_invalid_result_resolutions(
    root: Path,
    *,
    campaign_id: str,
    plan_sha256: str,
    source_identity: Mapping[str, str],
) -> list[dict[str, Any]]:
    path = root / MANUAL_INVALID_LEDGER_RELATIVE_PATH
    if not path.is_file():
        return []
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ResultError(
            f"cannot read manual invalid-result ledger: {exc}"
        ) from exc
    if raw and not raw.endswith(b"\n"):
        raise ResultError("manual invalid-result ledger must end with LF")
    lines = raw.splitlines()
    if any(not line for line in lines):
        raise ResultError(
            "manual invalid-result ledger must not contain blank lines"
        )
    if len(lines) > len(REVIEWED_MANUAL_INVALID_RESULTS):
        raise ResultError(
            "manual invalid-result ledger exceeds the reviewed target set"
        )
    events = _manual_lr_campaign_events_by_seq(root)
    previous_hash: str | None = None
    resolutions: list[dict[str, Any]] = []
    for index, line in enumerate(lines, 1):
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResultError(
                f"manual invalid-result ledger line {index} is invalid JSON"
            ) from exc
        entry = _require_mapping(
            value, f"manual invalid-result ledger entry {index}"
        )
        if line != _strict_canonical_json_bytes(
            entry, f"manual invalid-result ledger entry {index}"
        ):
            raise ResultError(
                f"manual invalid-result ledger line {index} is not canonical"
            )
        name = f"manual invalid-result ledger entry {index}"
        _require_exact_keys(entry, _MANUAL_INVALID_ENTRY_KEYS, name)
        reviewed = REVIEWED_MANUAL_INVALID_RESULTS.get(
            (
                entry.get("model"),
                entry.get("setting"),
                entry.get("grad_lr"),
                entry.get("attempt_index"),
            )
        )
        if (
            entry.get("schema_version") != 1
            or entry.get("seq") != index
            or entry.get("campaign_id") != campaign_id
            or reviewed is None
            or entry.get("reason_code")
            != (reviewed or {}).get("reason_code")
            or entry.get("prev_record_sha256") != previous_hash
        ):
            raise ResultError(
                f"{name} schema/campaign/reason/hash-chain identity is invalid"
            )
        timestamp = _require_nonempty_string(
            entry.get("timestamp_utc"), f"{name}.timestamp_utc"
        )
        try:
            parsed_timestamp = dt.datetime.fromisoformat(
                timestamp.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ResultError(f"{name}.timestamp_utc is invalid") from exc
        if parsed_timestamp.tzinfo is None:
            raise ResultError(f"{name}.timestamp_utc must include timezone")
        record_sha = _require_sha256(
            entry.get("record_sha256"), f"{name}.record_sha256"
        )
        unsigned = dict(entry)
        unsigned.pop("record_sha256")
        if (
            _strict_canonical_sha256(unsigned, f"{name} unsigned")
            != record_sha
        ):
            raise ResultError(f"{name}.record_sha256 is invalid")

        authorization = _require_mapping(
            entry.get("authorization"), f"{name}.authorization"
        )
        _require_exact_keys(
            authorization,
            _MANUAL_INVALID_AUTHORIZATION_KEYS,
            f"{name}.authorization",
        )
        if (
            authorization.get("kind")
            != "user_authorized_manual_invalid_result_v1"
        ):
            raise ResultError(f"{name}.authorization.kind is invalid")
        _require_nonempty_string(
            authorization.get("authorized_by"),
            f"{name}.authorization.authorized_by",
        )
        _require_nonempty_string(
            authorization.get("rationale"),
            f"{name}.authorization.rationale",
        )

        manifest_ref = _require_mapping(
            entry.get("manifest"), f"{name}.manifest"
        )
        log_ref = _require_mapping(entry.get("log"), f"{name}.log")
        _require_exact_keys(
            manifest_ref, _MANUAL_LR_FILE_REF_KEYS, f"{name}.manifest"
        )
        _require_exact_keys(
            log_ref, _MANUAL_LR_FILE_REF_KEYS, f"{name}.log"
        )
        manifest_relative = _require_nonempty_string(
            manifest_ref.get("relative_path"),
            f"{name}.manifest.relative_path",
        )
        log_relative = _require_nonempty_string(
            log_ref.get("relative_path"),
            f"{name}.log.relative_path",
        )
        manifest_path = (root / manifest_relative).resolve(strict=False)
        log_path = (root / log_relative).resolve(strict=False)
        if (
            dict(manifest_ref)
            != _manual_lr_file_ref(
                manifest_path, root=root, name=f"{name}.manifest"
            )
            or dict(log_ref)
            != _manual_lr_file_ref(
                log_path, root=root, name=f"{name}.log"
            )
            or manifest_ref.get("present") is not True
            or log_ref.get("present") is not True
            or manifest_path.name != MANIFEST_FILENAME
            or log_path.name != LOG_FILENAME
            or manifest_path.parent != log_path.parent
        ):
            raise ResultError(f"{name} manifest/log references are invalid")
        manifest = _read_manifest(manifest_path)
        identity = _manual_lr_manifest_launch_identity(
            manifest, manifest_path=manifest_path
        )
        entry_identity = {
            "execution_id": entry.get("execution_id"),
            "run_id": entry.get("run_id"),
            "status": entry.get("manifest_status"),
            "exit_code": entry.get("exit_code"),
            "model": entry.get("model"),
            "setting": entry.get("setting"),
            "grad_lr": entry.get("grad_lr"),
            "attempt_index": entry.get("attempt_index"),
            "plan_sha256": entry.get("plan_sha256"),
            "model_sha256": entry.get("model_sha256"),
            "runner_sha256": entry.get("runner_sha256"),
            "executor_sha256": entry.get("executor_sha256"),
            "numerical_source_sha256_at_start": entry.get(
                "numerical_source_sha256_at_start"
            ),
            "numerical_source_sha256_at_end": entry.get(
                "numerical_source_sha256_at_end"
            ),
            "numerical_source_changed_during_execution": entry.get(
                "numerical_source_changed_during_execution"
            ),
        }
        if any(
            identity.get(field) != expected
            for field, expected in entry_identity.items()
        ):
            raise ResultError(
                f"{name} identity disagrees with current manifest"
            )
        if (
            entry.get("phase") != "tune"
            or entry.get("method") != "realq"
            or entry.get("manifest_status") != "succeeded"
            or entry.get("exit_code") != 0
            or entry.get(
                "numerical_source_changed_during_execution"
            )
            is not True
            or entry.get("numerical_source_sha256_at_start")
            != (
                reviewed or {}
            ).get("numerical_source_sha256_at_start")
            or entry.get("numerical_source_sha256_at_end")
            != (
                reviewed or {}
            ).get("numerical_source_sha256_at_end")
            or entry.get("numerical_source_sha256_at_start")
            != source_identity["numerical_source_sha256"]
            or entry.get("plan_sha256") != plan_sha256
            or entry.get("runner_sha256")
            != source_identity["runner_sha256"]
            or entry.get("executor_sha256")
            != source_identity["executor_sha256"]
            or entry.get("expected_rejection_errors")
            != list((reviewed or {}).get("rejection_errors", ()))
        ):
            raise ResultError(
                f"{name} is not the narrowly reviewed invalid launch"
            )

        event_present = entry.get("launch_event_present")
        event_ref = entry.get("launch_event")
        if type(event_present) is not bool:
            raise ResultError(
                f"{name}.launch_event_present must be boolean"
            )
        if not event_present:
            if event_ref is not None:
                raise ResultError(
                    f"{name}.launch_event must be null when absent"
                )
        else:
            event_ref = _require_mapping(
                event_ref, f"{name}.launch_event"
            )
            _require_exact_keys(
                event_ref,
                _MANUAL_INVALID_LAUNCH_EVENT_KEYS,
                f"{name}.launch_event",
            )
            sequence = event_ref.get("seq")
            if type(sequence) is not int or sequence <= 0:
                raise ResultError(f"{name}.launch_event.seq is invalid")
            event_pair = events.get(sequence)
            if event_pair is None:
                raise ResultError(
                    f"{name}.launch_event references a missing event"
                )
            event, event_sha = event_pair
            details = event.get("details")
            if (
                event_ref.get("line_sha256") != event_sha
                or not isinstance(details, Mapping)
                or event.get("schema_version") != 1
                or event.get("campaign_id") != campaign_id
                or event.get("category") != "task"
                or event.get("code") != "task_launching"
                or details.get("task_id") != event_ref.get("task_id")
                or details.get("model") != entry.get("model")
                or details.get("setting") != entry.get("setting")
                or details.get("grad_lr") != entry.get("grad_lr")
                or details.get("attempt_index")
                != entry.get("attempt_index")
                or details.get("plan_sha256")
                != entry.get("plan_sha256")
            ):
                raise ResultError(
                    f"{name}.launch_event identity is invalid"
                )
        resolutions.append(dict(entry))
        previous_hash = record_sha
    reviewed_keys = [
        (
            entry["model"],
            entry["setting"],
            entry["grad_lr"],
            entry["attempt_index"],
        )
        for entry in resolutions
    ]
    if len(reviewed_keys) != len(set(reviewed_keys)):
        raise ResultError(
            "manual invalid-result ledger resolves a reviewed target twice"
        )
    return resolutions


def _manual_lr_report_path_map(
    values: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(values):
        manifest = _require_nonempty_string(
            value.get("manifest"), f"{label}[{index}].manifest"
        )
        canonical = str(Path(manifest).resolve(strict=False))
        if manifest != canonical or canonical in result:
            raise ResultError(
                f"{label} has a non-canonical or duplicate manifest"
            )
        result[canonical] = value
    return result


def _manual_lr_tune_launch_projection(
    root: Path,
    *,
    matrix_order: Sequence[tuple[str, str]],
    records: Sequence[ParsedResult],
    rejected: Sequence[Mapping[str, Any]],
    informational: Sequence[Mapping[str, Any]],
    retired_attempts: Sequence[Mapping[str, Any]],
    invalid_result_resolutions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    accepted = {
        str(item.manifest_path): {
            "manifest": str(item.manifest_path),
            "method": item.method,
            "phase": item.phase,
        }
        for item in records
    }
    rejected_by_path = _manual_lr_report_path_map(
        rejected, label="rejected"
    )
    informational_by_path = _manual_lr_report_path_map(
        informational, label="informational"
    )
    retired_by_path = _manual_lr_report_path_map(
        retired_attempts, label="retired_attempts"
    )
    resolution_by_path: dict[str, Mapping[str, Any]] = {}
    for entry in invalid_result_resolutions:
        manifest_ref = _require_mapping(
            entry.get("manifest"),
            "manual invalid result resolution manifest",
        )
        path = _require_nonempty_string(
            manifest_ref.get("absolute_path"),
            "manual invalid result resolution manifest.absolute_path",
        )
        if path in resolution_by_path:
            raise ResultError(
                "manual invalid-result ledger resolves a manifest twice"
            )
        resolution_by_path[path] = entry
    if set(resolution_by_path) - set(rejected_by_path):
        raise ResultError(
            "manual invalid-result ledger references a manifest not rejected "
            "by the parser"
        )
    unresolved = sorted(set(rejected_by_path) - set(resolution_by_path))
    if unresolved:
        raise ResultError(
            "manual LR reconciliation contains unresolved rejected manifests: "
            f"{unresolved!r}"
        )
    for path, entry in resolution_by_path.items():
        if (
            rejected_by_path[path].get("errors")
            != entry.get("expected_rejection_errors")
        ):
            raise ResultError(
                "manual invalid-result rejection errors disagree with parser: "
                f"{path}"
            )

    launches: list[dict[str, Any]] = []
    classified_paths: set[str] = set()
    for manifest_path in sorted(root.rglob(MANIFEST_FILENAME)):
        try:
            manifest = _read_manifest(manifest_path)
            context = _tune_attempt_context(
                manifest, manifest_path.resolve()
            )
        except ResultError as exc:
            raise ResultError(
                f"cannot classify result manifest {manifest_path}: {exc}"
            ) from exc
        if context is None:
            continue
        manifest_path = manifest_path.resolve()
        manifest_text = str(manifest_path)
        identity = _manual_lr_manifest_launch_identity(
            manifest, manifest_path=manifest_path
        )
        if manifest_text in resolution_by_path:
            classification = "manual_invalid_result"
            rejection_errors = list(
                rejected_by_path[manifest_text]["errors"]
            )
        elif manifest_text in retired_by_path:
            classification = "retired"
            rejection_errors = []
        elif manifest_text in accepted:
            item = accepted[manifest_text]
            if item["method"] != "realq" or item["phase"] != "tune":
                raise ResultError(
                    "a tune launch maps to a non-tune accepted record"
                )
            classification = "accepted"
            rejection_errors = []
        elif manifest_text in rejected_by_path:
            raise ResultError(
                f"unresolved rejected tune launch: {manifest_text}"
            )
        elif manifest_text in informational_by_path:
            classification = "informational"
            rejection_errors = []
        else:
            raise ResultError(
                "identifiable tune launch is absent from every report "
                f"classification: {manifest_text}"
            )
        launch = {
            "manifest": _manual_lr_file_ref(
                manifest_path,
                root=root,
                name="tune launch manifest",
            ),
            "log": _manual_lr_file_ref(
                manifest_path.parent / LOG_FILENAME,
                root=root,
                name="tune launch log",
            ),
            "classification": classification,
            "rejection_errors": rejection_errors,
            **identity,
        }
        if set(launch) != _MANUAL_LR_TUNE_LAUNCH_KEYS:
            raise AssertionError("internal manual LR tune launch drift")
        launches.append(launch)
        classified_paths.add(manifest_text)

    for label, mapping in (
        ("accepted tune record", accepted),
        ("retired tune attempt", retired_by_path),
        ("manual invalid result", resolution_by_path),
    ):
        for path, value in mapping.items():
            if label == "accepted tune record" and (
                value.get("method") != "realq"
                or value.get("phase") != "tune"
            ):
                continue
            if label == "retired tune attempt" and (
                value.get("phase") != "tune"
            ):
                continue
            if path not in classified_paths:
                raise ResultError(
                    f"{label} lacks an identifiable raw tune launch: {path}"
                )
    matrix_index = {
        key: index for index, key in enumerate(matrix_order)
    }
    if len(matrix_index) != len(matrix_order):
        raise ResultError("manual LR reconciliation matrix order is invalid")
    if any(
        (item["model"], item["setting"]) not in matrix_index
        for item in launches
    ):
        raise ResultError(
            "tune launch projection contains a group outside the plan matrix"
        )
    launches.sort(
        key=lambda item: (
            matrix_index[(item["model"], item["setting"])],
            float(item["grad_lr"]),
            str(item["manifest"]["absolute_path"]),
        )
    )
    manifests = [
        item["manifest"]["absolute_path"] for item in launches
    ]
    execution_ids = [item["execution_id"] for item in launches]
    run_ids = [item["run_id"] for item in launches]
    if (
        len(manifests) != len(set(manifests))
        or len(execution_ids) != len(set(execution_ids))
        or len(run_ids) != len(set(run_ids))
    ):
        raise ResultError(
            "complete tune launch projection has duplicate manifest/run "
            "identity"
        )
    return launches


def _manual_lr_reconciliation_point(
    record: ParsedResult,
) -> dict[str, Any]:
    if (
        record.method != "realq"
        or record.phase != "tune"
        or record.grad_lr is None
        or record.kl_wikitext2 is None
    ):
        raise ResultError(
            "manual LR reconciliation accepts only complete REAL-Q tune "
            "results"
        )
    point = {
        "model": record.model,
        "setting": record.setting,
        "manifest": str(record.manifest_path),
        "manifest_sha256": _sha256_file(record.manifest_path),
        "log": str(record.log_path),
        "log_sha256": _sha256_file(record.log_path),
        "execution_id": record.execution_id,
        "run_id": record.run_id,
        "attempt_index": record.attempt_index,
        "world_size": record.world_size,
        "grad_lr": float(record.grad_lr),
        "kl_wikitext2": float(record.kl_wikitext2),
        "ppl_wikitext2": float(record.ppl_wikitext2),
        "plan_sha256": record.identities["plan_sha256"],
        "model_sha256": record.identities["model_sha256"],
        "runner_sha256": record.identities["runner_sha256"],
        "executor_sha256": record.identities["executor_sha256"],
        "numerical_source_sha256": record.identities[
            "numerical_source_sha256"
        ],
        "config_sha256": _manual_lr_reconciliation_config_sha256(record),
    }
    if set(point) != _MANUAL_LR_RECONCILIATION_POINT_KEYS:
        raise AssertionError("internal manual LR reconciliation point drift")
    return point


def _manual_lr_reconciliation_snapshot(
    *,
    root: Path,
    plan: Mapping[str, Any],
    records: Sequence[ParsedResult],
    tune_selections: Sequence[Mapping[str, Any]],
    retirement_ledger: Mapping[str, Any],
    rejected: Sequence[Mapping[str, Any]],
    informational: Sequence[Mapping[str, Any]],
    retired_attempts: Sequence[Mapping[str, Any]],
    campaign_id: str,
    expected_plan_sha256: str,
    source_identity: Mapping[str, str],
) -> dict[str, Any]:
    """Rebuild the freeze-time authoritative result snapshot without state."""

    models = _require_mapping(plan.get("models"), "embedded plan.models")
    settings = _require_mapping(plan.get("settings"), "embedded plan.settings")
    matrix = [
        (str(model), str(setting))
        for model in models
        for setting in settings
    ]
    matrix_set = set(matrix)

    tune_records = [
        item
        for item in records
        if item.method == "realq" and item.phase == "tune"
    ]
    grouped: dict[tuple[str, str], list[ParsedResult]] = {
        key: [] for key in matrix
    }
    seen_manifests: set[str] = set()
    seen_execution_ids: set[str] = set()
    seen_run_ids: set[str] = set()
    for item in tune_records:
        key = (item.model, item.setting)
        if key not in matrix_set:
            raise ResultError(
                "manual LR reconciliation found an accepted tune result "
                f"outside the plan matrix: {key!r}"
            )
        manifest = str(item.manifest_path)
        if manifest in seen_manifests:
            raise ResultError(
                "manual LR reconciliation found a duplicate manifest"
            )
        if item.execution_id in seen_execution_ids:
            raise ResultError(
                "manual LR reconciliation found a duplicate execution_id"
            )
        if item.run_id in seen_run_ids:
            raise ResultError(
                "manual LR reconciliation found a duplicate run_id"
            )
        seen_manifests.add(manifest)
        seen_execution_ids.add(item.execution_id)
        seen_run_ids.add(item.run_id)
        grouped[key].append(item)

    missing_groups = [
        f"{model}/{setting}"
        for (model, setting), group in grouped.items()
        if not group
    ]
    if missing_groups:
        raise ResultError(
            "manual LR reconciliation lacks an accepted active tune result "
            f"for {missing_groups!r}"
        )

    accepted: list[dict[str, Any]] = []
    for key in matrix:
        ordered = sorted(
            grouped[key],
            key=lambda item: (float(item.grad_lr), str(item.manifest_path)),
        )
        if len(
            {
                _strict_canonical_sha256(
                    _selection_config(item),
                    "manual LR reconciliation non-LR tune configuration",
                )
                for item in ordered
            }
        ) != 1:
            raise ResultError(
                "manual LR reconciliation accepted tune points have "
                f"non-LR configuration or provenance drift for {key!r}"
            )
        accepted.extend(
            _manual_lr_reconciliation_point(item) for item in ordered
        )

    selections_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, selection_value in enumerate(tune_selections):
        selection = _require_mapping(
            selection_value,
            f"REAL-Q tune selection[{index}]",
        )
        key = (
            _require_nonempty_string(
                selection.get("model"),
                f"REAL-Q tune selection[{index}].model",
            ),
            _require_nonempty_string(
                selection.get("setting"),
                f"REAL-Q tune selection[{index}].setting",
            ),
        )
        if key in selections_by_key:
            raise ResultError(
                "manual LR reconciliation found a duplicate tune-selection "
                f"group: {key!r}"
            )
        selections_by_key[key] = selection
    if set(selections_by_key) != matrix_set:
        missing = sorted(matrix_set - set(selections_by_key))
        unexpected = sorted(set(selections_by_key) - matrix_set)
        raise ResultError(
            "manual LR reconciliation tune-selection groups disagree with "
            f"the plan matrix: missing={missing!r}, unexpected={unexpected!r}"
        )

    group_attempt_counts: list[dict[str, Any]] = []
    for model, setting in matrix:
        selection = selections_by_key[(model, setting)]
        attempt_count = selection.get("attempt_count")
        if (
            type(attempt_count) is not int
            or attempt_count <= 0
            or attempt_count > 20
        ):
            raise ResultError(
                "manual LR reconciliation attempt_count for "
                f"{model}/{setting} must be in [1,20]"
            )
        if attempt_count < len(grouped[(model, setting)]):
            raise ResultError(
                "manual LR reconciliation attempt_count undercounts "
                f"accepted results for {model}/{setting}"
            )
        group_attempt_counts.append(
            {
                "model": model,
                "setting": setting,
                "attempt_count": attempt_count,
            }
        )

    if retirement_ledger.get("valid") is not True:
        raise ResultError(
            "manual LR reconciliation requires a valid retirement ledger"
        )
    accepted_sha = _strict_canonical_sha256(
        accepted, "manual LR reconciliation accepted active set"
    )
    counts_sha = _strict_canonical_sha256(
        group_attempt_counts,
        "manual LR reconciliation group attempt counts",
    )
    invalid_resolutions = _load_manual_invalid_result_resolutions(
        root,
        campaign_id=campaign_id,
        plan_sha256=expected_plan_sha256,
        source_identity=source_identity,
    )
    tune_launches = _manual_lr_tune_launch_projection(
        root,
        matrix_order=matrix,
        records=records,
        rejected=rejected,
        informational=informational,
        retired_attempts=retired_attempts,
        invalid_result_resolutions=invalid_resolutions,
    )
    derived_counts = {
        key: sum(
            1
            for launch in tune_launches
            if (launch["model"], launch["setting"]) == key
        )
        for key in matrix
    }
    expected_counts = {
        (item["model"], item["setting"]): item["attempt_count"]
        for item in group_attempt_counts
    }
    if derived_counts != expected_counts:
        raise ResultError(
            "manual LR reconciliation launch counts disagree with the "
            "complete raw tune launch projection"
        )
    launches_sha = _strict_canonical_sha256(
        tune_launches, "manual LR reconciliation tune launches"
    )
    resolutions_sha = _strict_canonical_sha256(
        invalid_resolutions,
        "manual LR reconciliation invalid-result resolutions",
    )
    return {
        "accepted_tune_results": accepted,
        "accepted_active_set_sha256": accepted_sha,
        "group_attempt_counts": group_attempt_counts,
        "group_attempt_counts_sha256": counts_sha,
        "tune_launches": tune_launches,
        "tune_launches_sha256": launches_sha,
        "invalid_result_resolutions": invalid_resolutions,
        "invalid_result_resolutions_sha256": resolutions_sha,
    }


def _validate_manual_lr_freeze_artifacts(
    root: Path,
    record: ParsedResult,
    freeze: Mapping[str, Any],
    *,
    reconciliation_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    plan_output_value = record.plan_content.get("output_root")
    plan_output_text = _require_nonempty_string(
        plan_output_value, "embedded plan.output_root"
    )
    plan_output = Path(plan_output_text)
    if not plan_output.is_absolute():
        plan_output = Path(__file__).resolve().parents[1] / plan_output
    if plan_output.resolve(strict=False) != root:
        raise ResultError(
            "embedded plan.output_root does not equal the parsed result root"
        )

    artifact_ref = _require_mapping(
        freeze.get("artifact"), "manual LR freeze artifact reference"
    )
    artifact_path = (
        root
        / "_campaign"
        / MANUAL_LR_FREEZE_ARTIFACT_FILENAME
    )
    try:
        artifact_raw = artifact_path.read_bytes()
    except OSError as exc:
        raise ResultError(
            f"cannot read manual LR freeze artifact {artifact_path}: {exc}"
        ) from exc
    if len(artifact_raw) != artifact_ref["size_bytes"]:
        raise ResultError("manual LR freeze artifact size_bytes mismatch")
    artifact_sha = hashlib.sha256(artifact_raw).hexdigest()
    if artifact_sha != artifact_ref["sha256"]:
        raise ResultError("manual LR freeze artifact sha256 mismatch")
    artifact = _load_json_mapping_bytes(
        artifact_raw, "manual LR freeze artifact"
    )
    _require_exact_keys(
        artifact,
        _MANUAL_LR_FREEZE_ARTIFACT_KEYS,
        "manual LR freeze artifact",
    )
    expected_artifact_raw = (
        _strict_canonical_json_bytes(
            artifact, "manual LR freeze artifact"
        )
        + b"\n"
    )
    if artifact_raw != expected_artifact_raw:
        raise ResultError(
            "manual LR freeze artifact is not exact canonical JSON plus LF"
        )
    if artifact.get("schema_version") != 1:
        raise ResultError(
            "manual LR freeze artifact schema_version must be 1"
        )
    if artifact.get("scope_id") != MANUAL_LR_FREEZE_SCOPE_ID:
        raise ResultError("manual LR freeze artifact scope_id is invalid")
    if artifact.get("artifact_kind") != MANUAL_LR_FREEZE_ARTIFACT_KIND:
        raise ResultError("manual LR freeze artifact_kind is invalid")
    transaction_id = _require_nonempty_string(
        artifact.get("transaction_id"),
        "manual LR freeze artifact transaction_id",
    )
    _require_nonempty_string(
        artifact.get("created_utc"),
        "manual LR freeze artifact created_utc",
    )
    ledger = _require_mapping(
        freeze.get("ledger"), "manual LR freeze ledger"
    )
    if (
        artifact.get("ledger_sha256")
        != freeze["wrapper_ledger_sha256"]
        or artifact.get("ledger") != ledger
    ):
        raise ResultError(
            "manual LR freeze artifact ledger differs from embedded plan"
        )

    pre_state_ref = _require_mapping(
        artifact.get("pre_freeze_state"),
        "manual LR freeze artifact pre_freeze_state",
    )
    _require_exact_keys(
        pre_state_ref,
        _MANUAL_LR_FREEZE_PRE_STATE_KEYS,
        "manual LR freeze artifact pre_freeze_state",
    )
    pre_state_sha = _require_sha256(
        pre_state_ref.get("sha256"),
        "manual LR freeze pre-state sha256",
    )
    if pre_state_sha != ledger["expected_state_sha256"]:
        raise ResultError(
            "manual LR freeze pre-state sha256 disagrees with ledger"
        )
    if (
        type(pre_state_ref.get("size_bytes")) is not int
        or pre_state_ref["size_bytes"] <= 0
    ):
        raise ResultError(
            "manual LR freeze pre-state size_bytes must be positive"
        )
    content_base64 = _require_nonempty_string(
        pre_state_ref.get("content_base64"),
        "manual LR freeze pre-state content_base64",
    )
    try:
        pre_state_raw = base64.b64decode(
            content_base64.encode("ascii"), validate=True
        )
    except (UnicodeEncodeError, ValueError) as exc:
        raise ResultError(
            "manual LR freeze pre-state content_base64 is invalid"
        ) from exc
    if base64.b64encode(pre_state_raw).decode("ascii") != content_base64:
        raise ResultError(
            "manual LR freeze pre-state content_base64 is not canonical"
        )
    if len(pre_state_raw) != pre_state_ref["size_bytes"]:
        raise ResultError("manual LR freeze pre-state size_bytes mismatch")
    if hashlib.sha256(pre_state_raw).hexdigest() != pre_state_sha:
        raise ResultError("manual LR freeze pre-state sha256 mismatch")
    pre_state = _load_json_mapping_bytes(
        pre_state_raw, "manual LR freeze pre-state"
    )
    if pre_state.get("campaign_id") != ledger["campaign_id"]:
        raise ResultError(
            "manual LR freeze pre-state campaign_id mismatch"
        )
    pre_plan = _require_mapping(
        pre_state.get("plan"), "manual LR freeze pre-state plan"
    )
    expected_plan_sha = ledger["expected_plan_sha256"]
    tune_protocol_sha = _require_sha256(
        pre_plan.get("tune_protocol_sha256"),
        "manual LR freeze pre-state plan.tune_protocol_sha256",
    )
    if (
        pre_plan.get("tune_sha256") != expected_plan_sha
        or pre_plan.get("active_sha256") != expected_plan_sha
        or pre_plan.get("formal_sha256") is not None
        or pre_plan.get("formal_protocol_sha256") is not None
        or pre_plan.get("pending_transition_sha256") is not None
    ):
        raise ResultError(
            "manual LR freeze pre-state plan transition is invalid"
        )
    for field in (
        "manual_lr_selection_freeze",
        "manual_lr_handoff",
        "manual_lr_state_reconciliation",
        "downstream_required_gate",
        "selected_lr_patch",
    ):
        if pre_state.get(field) is not None:
            raise ResultError(
                f"manual LR freeze pre-state unexpectedly contains {field}"
            )

    source_identity = _require_mapping(
        freeze.get("source_identity_before"),
        "manual LR freeze source_identity_before",
    )
    pre_source = _require_mapping(
        pre_state.get("source_identity"),
        "manual LR freeze pre-state source_identity",
    )
    expected_source_keys = {
        "numerical_source_sha256",
        "runner_sha256",
        "executor_sha256",
        "campaign_sha256",
        "results_sha256",
    }
    _require_exact_keys(
        pre_source,
        expected_source_keys,
        "manual LR freeze pre-state source_identity",
    )
    for key in expected_source_keys:
        _require_sha256(
            pre_source.get(key),
            f"manual LR freeze pre-state source_identity.{key}",
        )
    for key in expected_source_keys - {"results_sha256"}:
        if pre_source[key] != source_identity[key]:
            raise ResultError(
                "manual LR freeze pre-state source_identity drifted in "
                f"{key}"
            )
    current_results_sha = _sha256_file(Path(__file__).resolve())
    adoption_value = ledger.get("results_source_adoption")
    if pre_source["results_sha256"] == current_results_sha:
        if adoption_value is not None:
            raise ResultError(
                "manual LR freeze has an unnecessary results source adoption"
            )
    else:
        adoption = _require_mapping(
            adoption_value,
            "manual LR freeze results_source_adoption",
        )
        if (
            adoption.get("kind") != MANUAL_LR_RESULTS_ADOPTION_KIND
            or adoption.get("expected_old_results_sha256")
            != pre_source["results_sha256"]
            or adoption.get("accepted_current_results_sha256")
            != current_results_sha
            or source_identity["results_sha256"] != current_results_sha
        ):
            raise ResultError(
                "manual LR freeze results source adoption is invalid"
            )

    timing_gate = _require_mapping(
        artifact.get("timing_gate"),
        "manual LR freeze artifact timing_gate",
    )
    current_timing_gate = _current_formal_timing_gate()
    if timing_gate != current_timing_gate:
        raise ResultError(
            "manual LR freeze timing gate differs from current trusted sources"
        )
    if (
        ledger["timing_gate_sha256"]
        != current_timing_gate["gate_sha256"]
        or ledger["timing_source_set_sha256"]
        != current_timing_gate["source_set_sha256"]
    ):
        raise ResultError(
            "manual LR freeze ledger timing hashes disagree with timing gate"
        )

    reconciliation = _require_mapping(
        artifact.get("state_reconciliation"),
        "manual LR freeze artifact state_reconciliation",
    )
    _require_exact_keys(
        reconciliation,
        _MANUAL_LR_RECONCILIATION_KEYS,
        "manual LR freeze artifact state_reconciliation",
    )
    if (
        reconciliation.get("schema_version") != 1
        or reconciliation.get("policy_id")
        != MANUAL_LR_RECONCILIATION_POLICY_ID
    ):
        raise ResultError(
            "manual LR freeze state reconciliation policy is invalid"
        )
    accepted_value = reconciliation.get("accepted_tune_results")
    if not isinstance(accepted_value, list):
        raise ResultError(
            "manual LR freeze accepted_tune_results must be a list"
        )
    for index, point_value in enumerate(accepted_value):
        point = _require_mapping(
            point_value,
            f"manual LR freeze accepted_tune_results[{index}]",
        )
        _require_exact_keys(
            point,
            _MANUAL_LR_RECONCILIATION_POINT_KEYS,
            f"manual LR freeze accepted_tune_results[{index}]",
        )
    counts_value = reconciliation.get("group_attempt_counts")
    if not isinstance(counts_value, list):
        raise ResultError(
            "manual LR freeze group_attempt_counts must be a list"
        )
    for index, count_value in enumerate(counts_value):
        count = _require_mapping(
            count_value,
            f"manual LR freeze group_attempt_counts[{index}]",
        )
        _require_exact_keys(
            count,
            _MANUAL_LR_RECONCILIATION_COUNT_KEYS,
            f"manual LR freeze group_attempt_counts[{index}]",
        )
    tune_launches_value = reconciliation.get("tune_launches")
    if not isinstance(tune_launches_value, list):
        raise ResultError("manual LR freeze tune_launches must be a list")
    for index, launch_value in enumerate(tune_launches_value):
        launch = _require_mapping(
            launch_value,
            f"manual LR freeze tune_launches[{index}]",
        )
        _require_exact_keys(
            launch,
            _MANUAL_LR_TUNE_LAUNCH_KEYS,
            f"manual LR freeze tune_launches[{index}]",
        )
        for ref_name in ("manifest", "log"):
            reference = _require_mapping(
                launch.get(ref_name),
                f"manual LR freeze tune_launches[{index}].{ref_name}",
            )
            _require_exact_keys(
                reference,
                _MANUAL_LR_FILE_REF_KEYS,
                f"manual LR freeze tune_launches[{index}].{ref_name}",
            )
    invalid_resolutions_value = reconciliation.get(
        "invalid_result_resolutions"
    )
    if not isinstance(invalid_resolutions_value, list):
        raise ResultError(
            "manual LR freeze invalid_result_resolutions must be a list"
        )
    for index, entry_value in enumerate(invalid_resolutions_value):
        entry = _require_mapping(
            entry_value,
            f"manual LR freeze invalid_result_resolutions[{index}]",
        )
        _require_exact_keys(
            entry,
            _MANUAL_INVALID_ENTRY_KEYS,
            f"manual LR freeze invalid_result_resolutions[{index}]",
        )
    raw_prefixes = _validate_manual_lr_raw_input_prefixes(
        root, reconciliation.get("raw_input_prefixes")
    )
    terminalized_value = reconciliation.get("terminalized_tasks")
    if not isinstance(terminalized_value, list):
        raise ResultError(
            "manual LR freeze terminalized_tasks must be a list"
        )
    terminalized: list[Mapping[str, Any]] = []
    for index, item_value in enumerate(terminalized_value):
        item = _require_mapping(
            item_value,
            f"manual LR freeze terminalized_tasks[{index}]",
        )
        _require_exact_keys(
            item,
            _MANUAL_LR_RECONCILIATION_TERMINALIZED_KEYS,
            f"manual LR freeze terminalized_tasks[{index}]",
        )
        for field in (
            "task_id",
            "previous_status",
            "reconciled_status",
            "reason",
        ):
            _require_nonempty_string(
                item.get(field),
                f"manual LR freeze terminalized_tasks[{index}].{field}",
            )
        terminalized.append(item)
    terminalized_ids = [str(item["task_id"]) for item in terminalized]
    if (
        terminalized_ids != sorted(terminalized_ids)
        or len(terminalized_ids) != len(set(terminalized_ids))
    ):
        raise ResultError(
            "manual LR freeze terminalized_tasks must have unique task_id "
            "values sorted lexically"
        )
    imported_value = reconciliation.get("imported_tasks")
    if not isinstance(imported_value, list):
        raise ResultError("manual LR freeze imported_tasks must be a list")
    imported: list[Mapping[str, Any]] = []
    accepted_by_manifest = {
        point["manifest"]: point for point in accepted_value
    }
    for index, task_value in enumerate(imported_value):
        task = _require_mapping(
            task_value,
            f"manual LR freeze imported_tasks[{index}]",
        )
        _require_exact_keys(
            task,
            _MANUAL_LR_IMPORTED_TASK_KEYS,
            f"manual LR freeze imported_tasks[{index}]",
        )
        _require_nonempty_string(
            task.get("task_id"),
            f"manual LR freeze imported_tasks[{index}].task_id",
        )
        manifest = task.get("manifest")
        point = accepted_by_manifest.get(manifest)
        if point is None:
            raise ResultError(
                "manual LR freeze imported task does not map to an accepted "
                f"point: {manifest!r}"
            )
        spec = _require_mapping(
            task.get("spec"),
            f"manual LR freeze imported_tasks[{index}].spec",
        )
        _require_exact_keys(
            spec,
            _MANUAL_LR_IMPORTED_TASK_SPEC_KEYS,
            f"manual LR freeze imported_tasks[{index}].spec",
        )
        metric = _require_mapping(
            task.get("metric"),
            f"manual LR freeze imported_tasks[{index}].metric",
        )
        _require_exact_keys(
            metric,
            _MANUAL_LR_IMPORTED_TASK_METRIC_KEYS,
            f"manual LR freeze imported_tasks[{index}].metric",
        )
        metric_identities = _require_mapping(
            metric.get("identities"),
            f"manual LR freeze imported_tasks[{index}].metric.identities",
        )
        _require_exact_keys(
            metric_identities,
            _MANUAL_LR_IMPORTED_TASK_IDENTITY_KEYS,
            (
                f"manual LR freeze imported_tasks[{index}]"
                ".metric.identities"
            ),
        )
        expected_spec = {
            "kind": "reconciled_realq_tune_result",
            "method": "realq",
            "phase": "tune",
            "target_phase": None,
            "model": point["model"],
            "setting": point["setting"],
            "grad_lr": point["grad_lr"],
            "overrides": {},
            "profile_level": None,
            "static_profile_level": None,
            "generation": None,
            "purpose": "manual_lr_freeze_authoritative_import",
        }
        expected_metric = {
            "kl": point["kl_wikitext2"],
            "ppl": point["ppl_wikitext2"],
            "grad_lr": point["grad_lr"],
            "identities": {
                key: point[key]
                for key in _MANUAL_LR_IMPORTED_TASK_IDENTITY_KEYS
            },
        }
        expected_values = {
            "task_id": "reconciled_tune:" + point["manifest_sha256"],
            "spec": expected_spec,
            "status": "SUCCEEDED",
            "gpu_count": point["world_size"],
            "priority": 0,
            "gpus": [],
            "attempt_index": point["attempt_index"],
            "executor_pid": None,
            "created_utc": artifact["created_utc"],
            "launched_utc": None,
            "finished_utc": artifact["created_utc"],
            "output_dir": str(Path(point["manifest"]).parent),
            "manifest": point["manifest"],
            "log": point["log"],
            "exit_code": 0,
            "failure_class": None,
            "cache_validation": None,
            # The dedicated freeze marker is authoritative.  Keeping the
            # legacy flag false prevents the base controller from treating
            # this terminal evidence task as a runnable imported profile.
            "imported": False,
            "plan_compatible": True,
            "metric": expected_metric,
            "imported_by_manual_lr_freeze": True,
            "reconciliation_reason": (
                "retirement_filtered_results_authoritative_import"
            ),
        }
        if dict(task) != expected_values:
            raise ResultError(
                f"manual LR freeze imported_tasks[{index}] is not the "
                "deterministic accepted-result import"
            )
        imported.append(task)
    imported_ids = [str(item["task_id"]) for item in imported]
    if len(imported_ids) != len(set(imported_ids)):
        raise ResultError(
            "manual LR freeze imported_tasks must have unique task_id values"
        )
    if [task["manifest"] for task in imported] != [
        point["manifest"] for point in accepted_value
    ]:
        raise ResultError(
            "manual LR freeze imported_tasks must follow accepted point order"
        )

    for field in (
        "accepted_active_set_sha256",
        "group_attempt_counts_sha256",
        "tune_launches_sha256",
        "invalid_result_resolutions_sha256",
        "raw_input_prefixes_sha256",
        "report_snapshot_sha256",
        "terminalized_tasks_sha256",
        "imported_tasks_sha256",
    ):
        _require_sha256(
            reconciliation.get(field),
            f"manual LR freeze state_reconciliation.{field}",
        )
    if reconciliation["accepted_active_set_sha256"] != (
        _strict_canonical_sha256(
            accepted_value,
            "manual LR freeze accepted active set",
        )
    ):
        raise ResultError(
            "manual LR freeze accepted_active_set_sha256 mismatch"
        )
    if reconciliation["group_attempt_counts_sha256"] != (
        _strict_canonical_sha256(
            counts_value,
            "manual LR freeze group attempt counts",
        )
    ):
        raise ResultError(
            "manual LR freeze group_attempt_counts_sha256 mismatch"
        )
    if reconciliation["tune_launches_sha256"] != (
        _strict_canonical_sha256(
            tune_launches_value,
            "manual LR freeze tune launches",
        )
    ):
        raise ResultError("manual LR freeze tune_launches_sha256 mismatch")
    if reconciliation["invalid_result_resolutions_sha256"] != (
        _strict_canonical_sha256(
            invalid_resolutions_value,
            "manual LR freeze invalid-result resolutions",
        )
    ):
        raise ResultError(
            "manual LR freeze invalid_result_resolutions_sha256 mismatch"
        )
    if reconciliation["raw_input_prefixes_sha256"] != (
        _strict_canonical_sha256(
            raw_prefixes,
            "manual LR freeze raw input prefixes",
        )
    ):
        raise ResultError(
            "manual LR freeze raw_input_prefixes_sha256 mismatch"
        )
    if reconciliation["terminalized_tasks_sha256"] != (
        _strict_canonical_sha256(
            terminalized_value,
            "manual LR freeze terminalized tasks",
        )
    ):
        raise ResultError(
            "manual LR freeze terminalized_tasks_sha256 mismatch"
        )
    if reconciliation["imported_tasks_sha256"] != (
        _strict_canonical_sha256(
            imported_value,
            "manual LR freeze imported tasks",
        )
    ):
        raise ResultError(
            "manual LR freeze imported_tasks_sha256 mismatch"
        )
    expected_report_snapshot = {
        "accepted_active_set_sha256": reconciliation[
            "accepted_active_set_sha256"
        ],
        "group_attempt_counts_sha256": reconciliation[
            "group_attempt_counts_sha256"
        ],
        "tune_launches_sha256": reconciliation[
            "tune_launches_sha256"
        ],
        "invalid_result_resolutions_sha256": reconciliation[
            "invalid_result_resolutions_sha256"
        ],
        "raw_input_prefixes_sha256": reconciliation[
            "raw_input_prefixes_sha256"
        ],
    }
    if reconciliation["report_snapshot_sha256"] != (
        _strict_canonical_sha256(
            expected_report_snapshot,
            "manual LR freeze report snapshot",
        )
    ):
        raise ResultError(
            "manual LR freeze report_snapshot_sha256 mismatch"
        )
    for field in (
        "accepted_tune_results",
        "accepted_active_set_sha256",
        "group_attempt_counts",
        "group_attempt_counts_sha256",
        "tune_launches",
        "tune_launches_sha256",
        "invalid_result_resolutions",
        "invalid_result_resolutions_sha256",
    ):
        if reconciliation[field] != reconciliation_snapshot[field]:
            raise ResultError(
                "manual LR freeze state reconciliation disagrees with "
                f"retirement-filtered results in {field}"
            )
    for index, point in enumerate(accepted_value):
        if point["plan_sha256"] != expected_plan_sha:
            raise ResultError(
                "manual LR freeze reconciled tune result "
                f"{index} belongs to a different tune plan"
            )
        for source_key in (
            "runner_sha256",
            "executor_sha256",
            "numerical_source_sha256",
        ):
            if point[source_key] != source_identity[source_key]:
                raise ResultError(
                    "manual LR freeze reconciled tune result "
                    f"{index} disagrees with current trusted {source_key}"
                )
    for source_key in (
        "runner_sha256",
        "executor_sha256",
        "numerical_source_sha256",
    ):
        if record.identities[source_key] != source_identity[source_key]:
            raise ResultError(
                "formal REAL-Q result disagrees with current trusted "
                f"{source_key}"
            )

    state_path = root / "_campaign" / "state.json"
    try:
        state_raw = state_path.read_bytes()
    except OSError as exc:
        raise ResultError(
            f"cannot read current campaign state {state_path}: {exc}"
        ) from exc
    state = _load_json_mapping_bytes(
        state_raw, "current campaign state"
    )
    plan_wrapper = _require_mapping(
        record.plan_content.get("manual_lr_selection_freeze"),
        "embedded plan manual LR freeze wrapper",
    )
    if state.get("manual_lr_selection_freeze") != plan_wrapper:
        raise ResultError(
            "current campaign state freeze wrapper differs from formal plan"
        )
    handoff = _require_mapping(
        state.get("manual_lr_handoff"),
        "current campaign state manual_lr_handoff",
    )
    _require_exact_keys(
        handoff,
        _MANUAL_LR_HANDOFF_KEYS,
        "current campaign state manual_lr_handoff",
    )
    expected_handoff = {
        "schema_version": 1,
        "scope_id": MANUAL_LR_FREEZE_SCOPE_ID,
        "transaction_id": transaction_id,
        "phase": "committed",
        "artifact_sha256": artifact_sha,
        "ledger_sha256": freeze["wrapper_ledger_sha256"],
        "selection_sha256": ledger["selection_sha256"],
        "expected_state_sha256": ledger["expected_state_sha256"],
        "expected_plan_sha256": ledger["expected_plan_sha256"],
        "formal_plan_sha256": record.identities["plan_sha256"],
    }
    if dict(handoff) != expected_handoff:
        raise ResultError(
            "current campaign state manual_lr_handoff is invalid"
        )
    state_reconciliation = _require_mapping(
        state.get("manual_lr_state_reconciliation"),
        "current campaign state manual_lr_state_reconciliation",
    )
    _require_exact_keys(
        state_reconciliation,
        _MANUAL_LR_STATE_RECONCILIATION_KEYS,
        "current campaign state manual_lr_state_reconciliation",
    )
    expected_state_reconciliation = {
        "schema_version": 1,
        "policy_id": MANUAL_LR_RECONCILIATION_POLICY_ID,
        "phase": "committed",
        "artifact_sha256": artifact_sha,
        "accepted_active_set_sha256": reconciliation[
            "accepted_active_set_sha256"
        ],
        "group_attempt_counts_sha256": reconciliation[
            "group_attempt_counts_sha256"
        ],
        "tune_launches_sha256": reconciliation[
            "tune_launches_sha256"
        ],
        "invalid_result_resolutions_sha256": reconciliation[
            "invalid_result_resolutions_sha256"
        ],
        "raw_input_prefixes_sha256": reconciliation[
            "raw_input_prefixes_sha256"
        ],
        "report_snapshot_sha256": reconciliation[
            "report_snapshot_sha256"
        ],
        "terminalized_tasks_sha256": reconciliation[
            "terminalized_tasks_sha256"
        ],
        "imported_tasks_sha256": reconciliation[
            "imported_tasks_sha256"
        ],
    }
    if dict(state_reconciliation) != expected_state_reconciliation:
        raise ResultError(
            "current campaign state manual LR reconciliation is invalid"
        )

    ledger_selections = ledger.get("selections")
    if not isinstance(ledger_selections, list):
        raise ResultError("manual LR freeze ledger selections must be a list")
    plan_models = _require_mapping(
        record.plan_content.get("models"), "embedded formal plan.models"
    )
    plan_settings = _require_mapping(
        record.plan_content.get("settings"), "embedded formal plan.settings"
    )
    expected_pairs = [
        (str(model), str(setting))
        for model in plan_models
        for setting in plan_settings
    ]
    if not expected_pairs or len(ledger_selections) != len(expected_pairs):
        raise ResultError(
            "manual LR freeze current-state validation requires the exact "
            "plan model/setting matrix"
        )
    selections_by_pair: dict[
        tuple[str, str], Mapping[str, Any]
    ] = {}
    for index, selection_value in enumerate(ledger_selections):
        selection = _require_mapping(
            selection_value,
            f"manual LR freeze ledger selections[{index}]",
        )
        pair = (
            str(selection.get("model")),
            str(selection.get("setting")),
        )
        if pair != expected_pairs[index] or pair in selections_by_pair:
            raise ResultError(
                "manual LR freeze ledger selections do not follow the exact "
                "plan model/setting order"
            )
        selections_by_pair[pair] = selection
    counts_by_pair: dict[tuple[str, str], int] = {}
    for index, count_value in enumerate(counts_value):
        count = _require_mapping(
            count_value,
            f"manual LR freeze group_attempt_counts[{index}]",
        )
        pair = (str(count.get("model")), str(count.get("setting")))
        if pair != expected_pairs[index] or pair in counts_by_pair:
            raise ResultError(
                "manual LR freeze group_attempt_counts do not follow the "
                "exact plan model/setting order"
            )
        attempt_count = count.get("attempt_count")
        if (
            type(attempt_count) is not int
            or attempt_count <= 0
            or attempt_count > 20
        ):
            raise ResultError(
                f"manual LR freeze attempt_count is invalid for {pair!r}"
            )
        counts_by_pair[pair] = attempt_count

    expected_selected_map = {
        model: {
            setting: selections_by_pair[(model, setting)]["selected_lr"]
            for setting in plan_settings
        }
        for model in plan_models
    }
    expected_selected_patch = {
        "selected_grad_lr_by_model_setting": expected_selected_map
    }
    if state.get("selected_lr_patch") != expected_selected_patch:
        raise ResultError(
            "current campaign state selected_lr_patch differs from the "
            "immutable manual LR ledger"
        )
    if (
        record.plan_content.get("selected_grad_lr_by_model_setting")
        != expected_selected_map
    ):
        raise ResultError(
            "embedded formal plan selected LR matrix differs from the "
            "immutable manual LR ledger"
        )
    state_tuning = _require_mapping(
        state.get("tuning"), "current campaign state tuning"
    )
    expected_group_keys = {
        f"{model}/{setting}" for model, setting in expected_pairs
    }
    if set(state_tuning) != expected_group_keys:
        missing = sorted(expected_group_keys - set(state_tuning))
        unexpected = sorted(set(state_tuning) - expected_group_keys)
        raise ResultError(
            "current campaign state tuning groups differ from the exact "
            f"plan matrix: missing={missing!r}, unexpected={unexpected!r}"
        )
    for model, setting in expected_pairs:
        group_key = f"{model}/{setting}"
        group = _require_mapping(
            state_tuning.get(group_key),
            f"current campaign state tuning[{group_key!r}]",
        )
        selection = selections_by_pair[(model, setting)]
        expected_parameters = {
            "grad_lr": selection["selected_lr"],
            "decision_kind": selection["decision_kind"],
            "rationale": selection["rationale"],
            "selected_manifest": selection["selected_manifest"],
            "evidence_set_sha256": selection["evidence_set_sha256"],
            "evidence": selection["evidence"],
            "manual_lr_selection_scope_id": MANUAL_LR_FREEZE_SCOPE_ID,
            "selection_sha256": ledger["selection_sha256"],
            "ledger_sha256": freeze["wrapper_ledger_sha256"],
        }
        if (
            group.get("model") != model
            or group.get("setting") != setting
            or group.get("status") != "SELECTED"
            or group.get("selected_lr") != selection["selected_lr"]
            or group.get("selected_manifest")
            != selection["selected_manifest"]
            or group.get("selected_parameters") != expected_parameters
            or group.get("needs_user_action_reason") is not None
            or group.get("manual_selection_decision_kind")
            != selection["decision_kind"]
            or group.get("manual_selection_evidence_set_sha256")
            != selection["evidence_set_sha256"]
            or group.get("attempt_count")
            != counts_by_pair[(model, setting)]
        ):
            raise ResultError(
                "current campaign state tuning selection/attempt_count "
                f"differs from the freeze artifact for {group_key}"
            )

    pre_tasks = _require_mapping(
        pre_state.get("tasks"), "manual LR freeze pre-state tasks"
    )
    current_tasks = _require_mapping(
        state.get("tasks"), "current campaign state tasks"
    )
    expected_terminalized: list[dict[str, Any]] = []
    for task_id_value, task_value in sorted(
        pre_tasks.items(), key=lambda item: str(item[0])
    ):
        task_id = str(task_id_value)
        task = _require_mapping(
            task_value, f"manual LR freeze pre-state task {task_id}"
        )
        previous = _require_nonempty_string(
            task.get("status"),
            f"manual LR freeze pre-state task {task_id}.status",
        )
        spec = task.get("spec")
        tune_realq = (
            isinstance(spec, Mapping)
            and spec.get("kind") == "realq"
            and spec.get("phase") == "tune"
        )
        if not (
            (tune_realq and previous != "SUPERSEDED")
            or previous not in _CAMPAIGN_TERMINAL_TASK_STATES
        ):
            continue
        expected_terminalized.append(
            {
                "task_id": task_id,
                "previous_status": previous,
                "reconciled_status": "SUPERSEDED",
                "reason": (
                    "old_tune_task_superseded_by_authoritative_import"
                    if tune_realq
                    else "stale_runnable_task_closed_before_formal_handoff"
                ),
            }
        )
    if [dict(item) for item in terminalized] != expected_terminalized:
        raise ResultError(
            "manual LR freeze terminalized_tasks do not exactly reconcile "
            "the pre-freeze state"
        )
    for item in terminalized:
        task_id = item["task_id"]
        pre_task = _require_mapping(
            pre_tasks.get(task_id),
            f"manual LR freeze pre-state task {task_id}",
        )
        current_task = _require_mapping(
            current_tasks.get(task_id),
            f"current reconciled campaign task {task_id}",
        )
        if pre_task.get("status") != item["previous_status"]:
            raise ResultError(
                f"manual LR freeze terminalized task {task_id} previous "
                "status mismatch"
            )
        if (
            current_task.get("status") != item["reconciled_status"]
            or current_task.get("reconciled_previous_status")
            != item["previous_status"]
            or current_task.get("reconciliation_reason") != item["reason"]
            or current_task.get("executor_pid") is not None
            or current_task.get("gpus") != []
            or any(
                field in current_task
                for field in (
                    "executor_start_ticks",
                    "executor_session_id",
                    "executor_process_group_id",
                    "executor_cli",
                    "rendezvous",
                )
            )
        ):
            raise ResultError(
                f"current campaign task {task_id} does not implement its "
                "manual LR reconciliation"
            )
    for item in imported:
        task_id = item["task_id"]
        if task_id in pre_tasks:
            raise ResultError(
                f"manual LR freeze imported task already existed: {task_id}"
            )
        if current_tasks.get(task_id) != item:
            raise ResultError(
                f"current campaign imported task differs from artifact: "
                f"{task_id}"
            )
    imported_ids_set = {str(item["task_id"]) for item in imported}
    unexpected_imports = [
        str(task_id)
        for task_id, task_value in current_tasks.items()
        if isinstance(task_value, Mapping)
        and (
            task_value.get("imported_by_manual_lr_freeze") is True
            or (
                isinstance(task_value.get("spec"), Mapping)
                and task_value["spec"].get("kind")
                == "reconciled_realq_tune_result"
            )
        )
        and str(task_id) not in imported_ids_set
    ]
    if unexpected_imports:
        raise ResultError(
            "current campaign state contains unexpected manual LR imported "
            f"tasks: {sorted(unexpected_imports)!r}"
        )
    for point in accepted_value:
        matches = [
            task
            for task in current_tasks.values()
            if isinstance(task, Mapping)
            and task.get("manifest") == point["manifest"]
            and task.get("status") == "SUCCEEDED"
        ]
        if len(matches) != 1:
            raise ResultError(
                "current campaign state must contain exactly one reconciled "
                f"SUCCEEDED task for {point['manifest']}"
            )
        task = matches[0]
        metric = _require_mapping(
            task.get("metric"),
            "current campaign reconciled tune metric",
        )
        if (
            task.get("log") != point["log"]
            or task.get("exit_code") != 0
            or task.get("attempt_index") != point["attempt_index"]
            or task.get("plan_compatible") is not True
            or metric.get("kl") != point["kl_wikitext2"]
            or metric.get("ppl") != point["ppl_wikitext2"]
            or metric.get("grad_lr") != point["grad_lr"]
        ):
            raise ResultError(
                "current campaign reconciled tune task disagrees with its "
                f"accepted point: {point['manifest']}"
            )
    if state.get("downstream_required_gate") != ledger[
        "downstream_required_gate"
    ]:
        raise ResultError(
            "current campaign state downstream gate differs from ledger"
        )
    if state.get("campaign_id") != ledger["campaign_id"]:
        raise ResultError(
            "current campaign state campaign_id differs from ledger"
        )
    state_plan = _require_mapping(
        state.get("plan"), "current campaign state plan"
    )
    if (
        state_plan.get("tune_sha256") != expected_plan_sha
        or state_plan.get("tune_protocol_sha256")
        != tune_protocol_sha
        or state_plan.get("active_sha256")
        != record.identities["plan_sha256"]
        or state_plan.get("formal_sha256")
        != record.identities["plan_sha256"]
        or state_plan.get("formal_protocol_sha256")
        != _canonical_sha256(record.plan_content)
        or state_plan.get("pending_transition_sha256") is not None
    ):
        raise ResultError(
            "current campaign state plan hashes disagree with formal result"
        )
    state_plan_path = Path(
        _require_nonempty_string(
            state_plan.get("path"), "current campaign state plan.path"
        )
    )
    try:
        resolved_plan_path = state_plan_path.resolve(strict=True)
        before = resolved_plan_path.stat()
        current_plan_raw = resolved_plan_path.read_bytes()
        after = resolved_plan_path.stat()
    except OSError as exc:
        raise ResultError(
            f"cannot read current formal plan {state_plan_path}: {exc}"
        ) from exc
    if (
        not resolved_plan_path.is_file()
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(current_plan_raw) != after.st_size
    ):
        raise ResultError(
            "current formal plan changed while it was being validated"
        )
    if (
        hashlib.sha256(current_plan_raw).hexdigest()
        != record.identities["plan_sha256"]
    ):
        raise ResultError(
            "current formal plan raw SHA disagrees with formal manifest"
        )
    current_plan = _load_json_mapping_bytes(
        current_plan_raw, "current formal plan"
    )
    if current_plan != record.plan_content:
        raise ResultError(
            "current formal plan content differs from formal manifest"
        )
    formal_state = _require_mapping(
        state.get("formal"), "current campaign state formal"
    )
    if formal_state.get("timing_gate") != timing_gate:
        raise ResultError(
            "current campaign state timing gate differs from artifact"
        )
    expected_current_source = {
        **dict(source_identity),
        "formal_timed_campaign_sha256": current_timing_gate[
            "controller_sha256"
        ],
        "formal_timing_gate_sha256": current_timing_gate[
            "gate_sha256"
        ],
        "formal_timing_source_set_sha256": current_timing_gate[
            "source_set_sha256"
        ],
        "formal_timed_executor_sha256": current_timing_gate[
            "timed_executor_sha256"
        ],
    }
    current_state_source = _require_mapping(
        state.get("source_identity"),
        "current campaign state source_identity",
    )
    if dict(current_state_source) != expected_current_source:
        raise ResultError(
            "current campaign state source_identity is not the exact "
            "adopted formal source identity"
        )
    return {
        "artifact": str(artifact_path),
        "artifact_sha256": artifact_sha,
        "transaction_id": transaction_id,
        "expected_state_sha256": pre_state_sha,
        "tune_protocol_sha256": tune_protocol_sha,
        "timing_gate_sha256": current_timing_gate["gate_sha256"],
        "timing_source_set_sha256": current_timing_gate[
            "source_set_sha256"
        ],
        "state_reconciliation": dict(reconciliation),
    }


def _selection_config(record: ParsedResult) -> dict[str, Any]:
    params = dict(record.params)
    params.pop("grad_lr", None)
    return {
        "model": record.model,
        "setting": record.setting,
        "method": record.method,
        "phase": record.phase,
        "final_layer_grad_lr": record.final_layer_grad_lr,
        "params_except_grad_lr": params,
        "identities": record.identities,
        "lm_eval_version": record.lm_eval_version,
        "expected_lr_candidates": record.expected_lr_candidates,
        "max_attempts": record.max_attempts,
        "search_policy": record.search_policy,
    }


def _lr_text(value: float) -> str:
    return format(value, ".12g")


def _scaled_decimal_lr(mantissa: int, exponent: int) -> float:
    """Build a canonical decimal LR without an intermediate float product."""

    return float(Decimal(mantissa).scaleb(exponent))


def _multiplied_decimal_lr(value: float, ratio: float) -> float:
    return float(Decimal(str(value)) * Decimal(str(ratio)))


def _midpoint_decimal_lr(lower: float, upper: float) -> float:
    return float((Decimal(str(lower)) + Decimal(str(upper))) / 2)


def _canonical_between(
    lower: float, upper: float, mantissas: Sequence[int]
) -> list[float]:
    if not (0 < lower < upper):
        return []
    first_exp = math.floor(math.log10(lower)) - 1
    last_exp = math.ceil(math.log10(upper)) + 1
    values = {
        _scaled_decimal_lr(mantissa, exponent)
        for exponent in range(first_exp, last_exp + 1)
        for mantissa in mantissas
        if lower < _scaled_decimal_lr(mantissa, exponent) < upper
    }
    return sorted(values)


def _canonical_above(
    value: float, mantissas: Sequence[int], count: int
) -> list[float]:
    if value < 0 or count <= 0:
        return []
    exponent = math.floor(math.log10(value)) - 1 if value > 0 else -12
    result: list[float] = []
    while len(result) < count:
        for mantissa in mantissas:
            candidate = _scaled_decimal_lr(mantissa, exponent)
            if candidate > value and candidate not in result:
                result.append(candidate)
                if len(result) == count:
                    break
        exponent += 1
    return sorted(result)


def _canonical_below(
    value: float, mantissas: Sequence[int], count: int
) -> list[float]:
    if value <= 0 or count <= 0:
        return []
    exponent = math.floor(math.log10(value)) + 1
    candidates: list[float] = []
    while len(candidates) < count:
        for mantissa in reversed(mantissas):
            candidate = _scaled_decimal_lr(mantissa, exponent)
            if 0 <= candidate < value and candidate not in candidates:
                candidates.append(candidate)
                if len(candidates) == count:
                    break
        exponent -= 1
    return sorted(candidates)


def _dedupe_recommendations(
    values: Sequence[float], observed: set[float]
) -> list[float]:
    return sorted(
        {
            float(value)
            for value in values
            if math.isfinite(value) and value >= 0 and float(value) not in observed
        }
    )


def _tune_selection(
    records: Sequence[ParsedResult],
    attempts_by_group: Mapping[
        tuple[str, str], Sequence[Mapping[str, Any]]
    ] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    attempts_by_group = attempts_by_group or {}
    groups: dict[tuple[str, str], list[ParsedResult]] = {}
    for record in records:
        if record.method == "realq" and record.phase == "tune":
            groups.setdefault((record.model, record.setting), []).append(record)

    selections: list[dict[str, Any]] = []
    by_manifest: dict[str, dict[str, Any]] = {}
    all_keys = set(groups) | set(attempts_by_group)
    for model, setting in sorted(all_keys):
        group = groups.get((model, setting), [])
        attempts = list(attempts_by_group.get((model, setting), ()))
        group = sorted(
            group,
            key=lambda item: (
                float(item.grad_lr) if item.grad_lr is not None else math.inf,
                str(item.manifest_path),
            ),
        )
        issues: list[dict[str, str]] = []
        recommendations: list[float] = []
        required_refinement: list[float] = []
        by_lr: dict[float, list[ParsedResult]] = {}
        for record in group:
            assert record.grad_lr is not None
            by_lr.setdefault(record.grad_lr, []).append(record)
        observed_lrs = set(by_lr)
        duplicates = {
            lr: entries for lr, entries in by_lr.items() if len(entries) > 1
        }
        if duplicates:
            issues.append(
                {
                    "code": "duplicate_candidate",
                    "message": (
                        "multiple successful runs exist for candidate LR(s) "
                        + ", ".join(_lr_text(lr) for lr in sorted(duplicates))
                        + "; choose/retire attempts explicitly before selection"
                    ),
                }
            )

        config_hashes = {
            _canonical_sha256(_selection_config(record)) for record in group
        }
        if len(config_hashes) > 1:
            issues.append(
                {
                    "code": "config_inconsistency",
                    "message": (
                        "tuning arms differ in non-candidate parameters or "
                        "plan/model/source identity; rerun a homogeneous sweep"
                    ),
                }
            )

        expected_sets = {
            tuple(record.expected_lr_candidates) for record in group
        }
        if len(expected_sets) > 1:
            issues.append(
                {
                    "code": "coarse_grid_drift",
                    "message": "successful attempts embed different coarse LR grids",
                }
            )
        if expected_sets:
            expected_ordered = tuple(next(iter(expected_sets)))
        elif attempts:
            expected_ordered = next(
                (
                    tuple(
                        float(value)
                        for value in attempt.get(
                            "expected_lr_candidates", ()
                        )
                    )
                    for attempt in attempts
                    if attempt.get("expected_lr_candidates")
                ),
                (),
            )
        else:
            expected_ordered = ()
        expected = set(expected_ordered)
        missing = sorted(expected - observed_lrs)
        if missing:
            issues.append(
                {
                    "code": "incomplete_grid",
                    "message": (
                        "missing embedded-plan coarse LR candidate(s): "
                        + ", ".join(_lr_text(lr) for lr in missing)
                    ),
                }
            )
            recommendations.extend(missing)

        max_attempts = 20
        policy = dict(EXPECTED_SEARCH_POLICY)
        if group:
            max_attempt_values = {record.max_attempts for record in group}
            policy_values = {
                _canonical_sha256(record.search_policy) for record in group
            }
            if len(max_attempt_values) != 1 or len(policy_values) != 1:
                issues.append(
                    {
                        "code": "search_policy_drift",
                        "message": "attempts embed different tuning budgets/policies",
                    }
                )
            elif (
                next(iter(max_attempt_values)) != 20
                or group[0].search_policy != EXPECTED_SEARCH_POLICY
            ):
                issues.append(
                    {
                        "code": "search_policy_drift",
                        "message": (
                            "tuning policy is not the reviewed "
                            "20-attempt policy"
                        ),
                    }
                )
        elif attempts and any(
            attempt.get("max_attempts") != 20
            or attempt.get("search_policy") != EXPECTED_SEARCH_POLICY
            for attempt in attempts
        ):
            issues.append(
                {
                    "code": "search_policy_drift",
                    "message": (
                        "attempt manifests do not consistently embed the "
                        "reviewed 20-attempt policy"
                    ),
                }
            )

        unique_records = [entries[0] for _, entries in sorted(by_lr.items())]
        observed: ParsedResult | None = None
        tied: list[ParsedResult] = []
        boundary = False
        if not unique_records:
            issues.append(
                {
                    "code": "no_successful_candidates",
                    "message": "no successful proxy candidate has an exact KL",
                }
            )
        else:
            minimum_kl = min(
                float(record.kl_wikitext2) for record in unique_records
            )
            tied = [
                record
                for record in unique_records
                if record.kl_wikitext2 == minimum_kl
            ]
            observed = min(
                tied,
                key=lambda item: (
                    float(item.grad_lr),
                    str(item.manifest_path),
                ),
            )
            if len(tied) > 1:
                tie_lrs = sorted(float(record.grad_lr) for record in tied)
                issues.append(
                    {
                        "code": "tied_minimum",
                        "message": (
                            "minimum exact WikiText-2 KL is tied at LR(s) "
                            + ", ".join(_lr_text(lr) for lr in tie_lrs)
                        ),
                    }
                )
                if not missing and len(tie_lrs) > 1:
                    tie_refine = _canonical_between(
                        tie_lrs[0],
                        tie_lrs[-1],
                        policy["canonical_mantissas"],
                    )
                    if not tie_refine:
                        tie_refine = [
                            _midpoint_decimal_lr(
                                tie_lrs[0], tie_lrs[-1]
                            )
                        ]
                    recommendations.extend(tie_refine)

            if not missing:
                lrs = sorted(by_lr)
                boundary = (
                    float(observed.grad_lr) == lrs[0]
                    or float(observed.grad_lr) == lrs[-1]
                )
            if not missing and boundary:
                winner_lr = float(observed.grad_lr)
                batch = int(policy["boundary_expansion_batch_size"])
                if winner_lr == 0:
                    next_lr = next((lr for lr in lrs if lr > 0), None)
                    if next_lr is not None:
                        recommendations.extend(
                            _multiplied_decimal_lr(next_lr, ratio)
                            for ratio in policy["zero_boundary_probe_ratios"]
                        )
                    message = (
                        "minimum is at LR=0; probe strictly between zero and "
                        "the smallest positive candidate"
                    )
                elif winner_lr == lrs[0]:
                    recommendations.extend(
                        _canonical_below(
                            winner_lr,
                            policy["canonical_mantissas"],
                            batch,
                        )
                    )
                    message = "expand the canonical LR grid below the minimum"
                else:
                    recommendations.extend(
                        _canonical_above(
                            winner_lr,
                            policy["canonical_mantissas"],
                            batch,
                        )
                    )
                    message = "expand the canonical LR grid above the minimum"
                issues.append(
                    {"code": "boundary_minimum", "message": message}
                )

        # Refinement is defined around the currently bracketed winner, not
        # merely around the winner of the original coarse grid.  This matters
        # when a coarse boundary winner becomes interior only after outward
        # expansion: selecting at that point would otherwise skip the required
        # canonical local probes.
        if (
            not missing
            and observed is not None
            and len(tied) == 1
            and not boundary
        ):
            ordered_lrs = sorted(observed_lrs)
            winner_index = ordered_lrs.index(float(observed.grad_lr))
            lower_lr = ordered_lrs[winner_index - 1]
            upper_lr = ordered_lrs[winner_index + 1]
            local_candidates = (
                [
                    _multiplied_decimal_lr(upper_lr, ratio)
                    for ratio in policy["zero_boundary_probe_ratios"]
                ]
                if lower_lr == 0
                else _canonical_between(
                    lower_lr,
                    upper_lr,
                    policy["canonical_mantissas"],
                )
            )
            required_refinement = [
                lr
                for lr in local_candidates
                if lr not in observed_lrs
            ]
            if required_refinement:
                issues.append(
                    {
                        "code": "missing_local_refinement",
                        "message": (
                            "canonical local refinement is mandatory; missing "
                            + ", ".join(
                                _lr_text(lr) for lr in required_refinement
                            )
                        ),
                    }
                )
                recommendations.extend(required_refinement)

        attempt_count = max(len(attempts), len(group))
        budget_remaining = max(0, max_attempts - attempt_count)
        user_action = False
        if attempt_count > max_attempts:
            issues.append(
                {
                    "code": "attempt_budget_exceeded",
                    "message": (
                        f"{attempt_count} attempts exceed the hard cap "
                        f"{max_attempts}"
                    ),
                }
            )
            user_action = True
        elif issues and budget_remaining == 0:
            issues.append(
                {
                    "code": "attempt_budget_exhausted",
                    "message": (
                        "the proxy search is unresolved after the maximum "
                        f"{max_attempts} attempts"
                    ),
                }
            )
            user_action = True
        elif missing and budget_remaining < (
            len(missing) + int(policy["reserved_refinement_attempts"])
        ):
            issues.append(
                {
                    "code": "insufficient_search_budget",
                    "message": (
                        f"{len(missing)} coarse candidate(s) remain and "
                        f"{policy['reserved_refinement_attempts']} attempt(s) "
                        "must remain available for local refinement"
                    ),
                }
            )
            user_action = True
        elif boundary and issues:
            reserve = int(policy["reserved_refinement_attempts"])
            expansion_batch = int(policy["boundary_expansion_batch_size"])
            if budget_remaining < reserve + expansion_batch:
                issues.append(
                    {
                        "code": "insufficient_refinement_reserve",
                        "message": (
                            f"only {budget_remaining} attempt(s) remain; "
                            f"{expansion_batch} are needed for boundary "
                            f"expansion and {reserve} must remain reserved "
                            "for local refinement"
                        ),
                    }
                )
                user_action = True
        elif (
            required_refinement
            and budget_remaining < len(required_refinement)
        ):
            issues.append(
                {
                    "code": "insufficient_refinement_budget",
                    "message": (
                        f"{len(required_refinement)} mandatory local "
                        f"candidate(s) remain but only {budget_remaining} "
                        "attempt(s) are available"
                    ),
                }
            )
            user_action = True

        recommendations = _dedupe_recommendations(
            recommendations, observed_lrs
        )
        if user_action:
            recommendations = []
        else:
            recommendations = recommendations[:budget_remaining]

        status = (
            "selected"
            if not issues
            else ("needs_user_action" if user_action else "needs_action")
        )
        selected = (
            {
                "grad_lr": observed.grad_lr,
                "kl_wikitext2": observed.kl_wikitext2,
                "final_layer_grad_lr": observed.final_layer_grad_lr,
                "manifest": str(observed.manifest_path),
            }
            if status == "selected" and observed is not None
            else None
        )
        if status == "selected":
            recommendation = "Canonical coarse and local refinement are complete."
        elif recommendations:
            recommendation = (
                "Run recommended LR candidate(s): "
                + ", ".join(_lr_text(lr) for lr in recommendations)
            )
        else:
            recommendation = "User action is required before more proxy launches."
        selection = {
            "model": model,
            "setting": setting,
            "status": status,
            "selected": selected,
            "observed_minimum": (
                {
                    "grad_lr": observed.grad_lr,
                    "kl_wikitext2": observed.kl_wikitext2,
                    "manifest": str(observed.manifest_path),
                }
                if observed is not None
                else None
            ),
            "attempt_count": attempt_count,
            "attempt_budget": max_attempts,
            "budget_remaining": budget_remaining,
            "candidate_count": len(group),
            "candidates": [
                {
                    "grad_lr": record.grad_lr,
                    "kl_wikitext2": record.kl_wikitext2,
                    "manifest": str(record.manifest_path),
                }
                for record in group
            ],
            "issues": issues,
            "recommended_lrs": recommendations,
            "recommendation": recommendation,
        }
        selections.append(selection)
        for record in group:
            by_manifest[str(record.manifest_path)] = {
                "status": status,
                "selected": (
                    selected is not None
                    and str(record.manifest_path) == selected["manifest"]
                ),
                "issues": [issue["code"] for issue in issues],
            }
    return selections, by_manifest


def _read_manifest(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResultError(f"cannot read manifest JSON: {exc}") from exc
    return _require_mapping(value, "manifest")


def _is_precompute_manifest(manifest: Mapping[str, Any]) -> bool:
    run_id = manifest.get("run_id")
    if isinstance(run_id, str) and run_id.startswith("precompute_"):
        return True
    command = manifest.get("command")
    argv = command.get("argv") if isinstance(command, Mapping) else None
    return isinstance(argv, list) and (
        "--exit_after_precompute" in argv
        or any(Path(str(item)).name == "save_grads.py" for item in argv)
    )


def _tune_attempt_context(
    manifest: Mapping[str, Any], manifest_path: Path
) -> dict[str, Any] | None:
    if _is_precompute_manifest(manifest):
        return None
    plan_wrapper = manifest.get("plan")
    if not isinstance(plan_wrapper, Mapping):
        return None
    plan = plan_wrapper.get("content")
    if not isinstance(plan, Mapping):
        return None
    run_id = manifest.get("run_id")
    command = manifest.get("command")
    argv = command.get("argv") if isinstance(command, Mapping) else None
    if (
        not isinstance(run_id, str)
        or not run_id.startswith("tune_realq_")
        or not isinstance(argv, list)
        or not all(isinstance(item, str) for item in argv)
        or "realq.ptq" not in argv
    ):
        return None
    context_valid = True
    try:
        parsed = _parse_argv(manifest, plan)
        _validate_command_protocol(parsed, plan)
    except ResultError:
        # A malformed launch still consumed experiment budget.  Recover only a
        # unique model/setting identity from the immutable run-id convention;
        # the manifest remains informational/rejected through the normal path.
        context_valid = False
        models = plan.get("models")
        settings = plan.get("settings")
        if not isinstance(models, Mapping) or not isinstance(
            settings, Mapping
        ):
            return None
        matches = [
            (str(model), str(setting))
            for model in models
            for setting in settings
            if run_id.startswith(
                f"tune_realq_{model}_{str(setting).lower()}_"
            )
        ]
        if len(matches) != 1:
            return None
        model, setting = matches[0]
        try:
            grad_lr_value = _option(argv, "--grad_lr")
            grad_lr = (
                _finite_float(grad_lr_value, "argv --grad_lr")
                if grad_lr_value is not None
                else None
            )
        except ResultError:
            grad_lr = None
        attempt_index = None
        env = command.get("env")
        if isinstance(env, Mapping):
            attempt_value = env.get("LOWBIT_ACTIVATION_ATTEMPT_INDEX")
            try:
                parsed_attempt = int(attempt_value)
            except (TypeError, ValueError):
                parsed_attempt = 0
            if parsed_attempt > 0:
                attempt_index = parsed_attempt
        if attempt_index is None:
            suffix = re.search(r"_attempt([1-9][0-9]*)$", run_id)
            attempt_index = int(suffix.group(1)) if suffix else 1
        parsed = {
            "method": "realq",
            "phase": "tune",
            "model": model,
            "setting": setting,
            "grad_lr": grad_lr,
            "attempt_index": attempt_index,
        }
    if parsed["method"] != "realq" or parsed["phase"] != "tune":
        return None
    tuning = plan.get("tuning")
    if not isinstance(tuning, Mapping):
        return None
    candidates = tuning.get("lr_candidates")
    expected_candidates = (
        tuple(float(value) for value in candidates)
        if isinstance(candidates, list)
        and all(_is_number(value) for value in candidates)
        else ()
    )
    return {
        "manifest": str(manifest_path.resolve()),
        "status": manifest.get("status"),
        "model": parsed["model"],
        "setting": parsed["setting"],
        "grad_lr": parsed["grad_lr"],
        "attempt_index": parsed["attempt_index"],
        "expected_lr_candidates": expected_candidates,
        "max_attempts": tuning.get("max_attempts_per_model_setting"),
        "search_policy": tuning.get("search_policy"),
        "context_valid": context_valid,
    }


def _json_without_duplicate_keys(raw: str, name: str) -> Any:
    def object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ResultError(f"{name} contains duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        return json.loads(raw, object_pairs_hook=object_hook)
    except ResultError:
        raise
    except json.JSONDecodeError as exc:
        raise ResultError(f"{name} is not valid JSON: {exc}") from exc


def _reviewed_memory_profiles(
    plan: Mapping[str, Any], *, phase: str
) -> list[dict[str, Any]]:
    """Return the scheduler's deduplicated, non-static OOM profile ladder."""

    phase_config = _require_mapping(
        plan.get("tuning" if phase == "tune" else "final"),
        f"embedded plan.{phase}",
    )
    policy_value = plan.get("qwen3_32b_memory_policy")
    if policy_value is None:
        return [{"values": {}, "effective_overrides": {}}]
    policy = _require_mapping(
        policy_value, "embedded plan.qwen3_32b_memory_policy"
    )
    if policy.get("adjust_only_batch_knobs_on_oom") is not True:
        raise ResultError(
            "embedded memory policy must adjust only reviewed batch knobs on OOM"
        )
    ladder_name = "tuning_ladders" if phase == "tune" else "final_ladders"
    ladders_value = _require_mapping(
        policy.get(ladder_name),
        f"embedded plan.qwen3_32b_memory_policy.{ladder_name}",
    )
    ladders: dict[str, list[int]] = {}
    for name, raw_values in ladders_value.items():
        if name == "global_loss_bsz":
            continue
        if (
            not isinstance(name, str)
            or not isinstance(raw_values, list)
            or not raw_values
            or not all(type(value) is int and value > 0 for value in raw_values)
        ):
            raise ResultError(
                f"embedded plan {phase} OOM ladder {name!r} is invalid"
            )
        ladders[name] = list(raw_values)
    if not ladders:
        raise ResultError(
            f"embedded plan {phase} OOM ladder has no non-static knobs"
        )
    profiles: list[dict[str, Any]] = []
    for level in range(max(len(values) for values in ladders.values())):
        values = {
            name: raw_values[min(level, len(raw_values) - 1)]
            for name, raw_values in ladders.items()
        }
        if profiles and profiles[-1]["values"] == values:
            continue
        effective = {
            name: value
            for name, value in values.items()
            if phase_config.get(name) != value
        }
        profiles.append(
            {
                "values": values,
                "effective_overrides": effective,
            }
        )
    return profiles


def _manifest_profile(
    parsed: Mapping[str, Any], plan: Mapping[str, Any]
) -> tuple[int, dict[str, int]]:
    profiles = _reviewed_memory_profiles(plan, phase=str(parsed["phase"]))
    params = _require_mapping(parsed.get("params"), "parsed argv params")
    matches = [
        index
        for index, profile in enumerate(profiles)
        if all(
            params.get(name) == expected
            for name, expected in profile["values"].items()
        )
    ]
    if len(matches) != 1:
        raise ResultError(
            "REAL-Q argv does not identify exactly one reviewed OOM profile"
        )
    level = matches[0]
    policy_value = plan.get("qwen3_32b_memory_policy")
    if policy_value is None:
        return level, {}
    policy = _require_mapping(
        policy_value, "embedded plan.qwen3_32b_memory_policy"
    )
    ladder_name = (
        "tuning_ladders" if parsed["phase"] == "tune" else "final_ladders"
    )
    ladders = _require_mapping(
        policy.get(ladder_name),
        f"embedded plan.qwen3_32b_memory_policy.{ladder_name}",
    )
    phase_config = _require_mapping(
        plan.get("tuning" if parsed["phase"] == "tune" else "final"),
        f"embedded plan.{parsed['phase']}",
    )
    overrides = {
        name: int(params[name])
        for name in ladders
        if params.get(name) != phase_config.get(name)
    }
    return level, overrides


def _retirement_manifest_context(
    manifest_path: Path, log_path: Path
) -> dict[str, Any]:
    manifest = _read_manifest(manifest_path)
    plan_wrapper = _require_mapping(manifest.get("plan"), "manifest.plan")
    plan_sha256 = _require_sha256(
        plan_wrapper.get("sha256"), "manifest.plan.sha256"
    )
    plan = _require_mapping(plan_wrapper.get("content"), "manifest.plan.content")
    parsed = _parse_argv(manifest, plan)
    _validate_command_protocol(parsed, plan)
    if parsed["method"] != "realq":
        raise ResultError("retirement ledger may reference only REAL-Q manifests")
    model_wrapper = _require_mapping(manifest.get("model"), "manifest.model")
    model_sha256 = _require_sha256(
        model_wrapper.get("combined_identity_sha256"),
        "manifest.model.combined_identity_sha256",
    )
    execution_id = _require_nonempty_string(
        manifest.get("execution_id"), "manifest.execution_id"
    )
    run_id = _require_nonempty_string(manifest.get("run_id"), "manifest.run_id")
    level, overrides = _manifest_profile(parsed, plan)
    try:
        log_text = log_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ResultError(f"cannot read retirement execution log: {exc}") from exc
    return {
        "manifest": manifest,
        "plan": plan,
        "plan_sha256": plan_sha256,
        "model_sha256": model_sha256,
        "execution_id": execution_id,
        "run_id": run_id,
        "phase": parsed["phase"],
        "model": parsed["model"],
        "setting": parsed["setting"],
        "grad_lr": parsed["grad_lr"],
        "profile_level": level,
        "effective_overrides": overrides,
        "manifest_status": manifest.get("status"),
        "exit_code": manifest.get("exit_code"),
        "oom": any(pattern.search(log_text) for pattern in _OOM_PATTERNS),
    }


def _validate_retirement_file_reference(
    root: Path,
    value: Any,
    *,
    expected_name: str,
    name: str,
) -> Path:
    reference = _require_mapping(value, name)
    _require_exact_keys(reference, _RETIREMENT_FILE_KEYS, name)
    absolute_text = _require_nonempty_string(
        reference.get("absolute_path"), f"{name}.absolute_path"
    )
    relative_text = _require_nonempty_string(
        reference.get("relative_path"), f"{name}.relative_path"
    )
    relative = Path(relative_text)
    if relative.is_absolute():
        raise ResultError(f"{name}.relative_path must be relative")
    try:
        actual = (root / relative).resolve(strict=True)
        actual.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ResultError(f"{name} must resolve inside the result root") from exc
    if absolute_text != str(actual):
        raise ResultError(f"{name}.absolute_path does not match relative_path")
    if relative_text != actual.relative_to(root).as_posix():
        raise ResultError(f"{name}.relative_path is not canonical")
    if actual.name != expected_name or not actual.is_file():
        raise ResultError(f"{name} must reference an existing {expected_name}")
    size = reference.get("size_bytes")
    if type(size) is not int or size < 0 or size != actual.stat().st_size:
        raise ResultError(f"{name}.size_bytes does not match the current file")
    expected_sha = _require_sha256(reference.get("sha256"), f"{name}.sha256")
    if _sha256_file(actual) != expected_sha:
        raise ResultError(f"{name}.sha256 does not match the current file")
    return actual


def _load_campaign_launch_events(
    root: Path, *, campaign_id: str
) -> dict[int, Mapping[str, Any]]:
    path = root / CAMPAIGN_EVENTS_RELATIVE_PATH
    if not path.is_file():
        raise ResultError(
            f"retirement ledger requires campaign events at {path}"
        )
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ResultError(f"cannot read campaign events: {exc}") from exc
    events: dict[int, Mapping[str, Any]] = {}
    for line_number, raw in enumerate(lines, 1):
        if not raw:
            raise ResultError(
                f"campaign events line {line_number} must not be blank"
            )
        value = _json_without_duplicate_keys(
            raw, f"campaign events line {line_number}"
        )
        event = _require_mapping(value, f"campaign events line {line_number}")
        sequence = event.get("seq")
        if type(sequence) is not int or sequence <= 0:
            raise ResultError(
                f"campaign events line {line_number} has invalid seq"
            )
        if sequence in events:
            raise ResultError(f"campaign events duplicate seq {sequence}")
        if event.get("campaign_id") != campaign_id:
            raise ResultError(
                "campaign events and retirement ledger campaign_id mismatch"
            )
        events[sequence] = event
    return events


def _validate_retirement_status(
    *,
    ledger_status: Any,
    manifest_status: Any,
    context: Mapping[str, Any],
    parsed_records: Mapping[Path, ParsedResult],
    manifest_path: Path,
    name: str,
) -> str:
    if ledger_status not in _RETIREMENT_STATUSES:
        raise ResultError(f"{name}.status is not a canonical scheduler status")
    if manifest_status != context["manifest_status"]:
        raise ResultError(f"{name}.manifest_status disagrees with manifest")
    exit_code = context["exit_code"]
    is_oom = bool(context["oom"])
    if ledger_status == "succeeded":
        if manifest_status != "succeeded" or exit_code != 0:
            raise ResultError(f"{name} succeeded status disagrees with manifest")
        if manifest_path not in parsed_records:
            raise ResultError(
                f"{name} succeeded manifest is not a strictly parsed result"
            )
    elif ledger_status == "oom":
        if manifest_status not in _FAILED_MANIFEST_STATUSES or not is_oom:
            raise ResultError(f"{name} OOM status lacks a matching OOM failure")
    elif ledger_status == "failed":
        if manifest_status not in _FAILED_MANIFEST_STATUSES or is_oom:
            raise ResultError(f"{name} failed status disagrees with manifest/log")
    elif ledger_status == "invalid_result":
        if (
            manifest_status != "succeeded"
            or exit_code != 0
            or manifest_path in parsed_records
        ):
            raise ResultError(
                f"{name} invalid_result status disagrees with parsed manifest"
            )
    elif ledger_status == "orphaned":
        if manifest_status not in {"launching", "running", "interrupted"}:
            raise ResultError(f"{name} orphaned status disagrees with manifest")
    elif ledger_status == "launch_failed":
        if manifest_status not in {"launch_failed", "wrapper_failed"} or is_oom:
            raise ResultError(
                f"{name} launch_failed status disagrees with manifest/log"
            )
    return str(ledger_status)


def _validate_retirement_identity(
    root: Path,
    value: Any,
    *,
    name: str,
    campaign_id: str,
    events: Mapping[int, Mapping[str, Any]],
    parsed_records: Mapping[Path, ParsedResult],
) -> dict[str, Any]:
    identity = _require_mapping(value, name)
    _require_exact_keys(identity, _RETIREMENT_IDENTITY_KEYS, name)
    manifest_path = _validate_retirement_file_reference(
        root,
        identity.get("manifest"),
        expected_name=MANIFEST_FILENAME,
        name=f"{name}.manifest",
    )
    log_path = _validate_retirement_file_reference(
        root,
        identity.get("log"),
        expected_name=LOG_FILENAME,
        name=f"{name}.log",
    )
    if log_path.parent != manifest_path.parent:
        raise ResultError(f"{name} manifest/log must be siblings")
    context = _retirement_manifest_context(manifest_path, log_path)

    task_id = _require_nonempty_string(identity.get("task_id"), f"{name}.task_id")
    launch_seq = identity.get("launch_event_seq")
    if type(launch_seq) is not int or launch_seq <= 0:
        raise ResultError(f"{name}.launch_event_seq must be positive")
    event = events.get(launch_seq)
    if event is None:
        raise ResultError(f"{name} references missing launch event {launch_seq}")
    if (
        event.get("schema_version") != 1
        or event.get("campaign_id") != campaign_id
        or event.get("category") != "task"
        or event.get("code") != "task_launching"
    ):
        raise ResultError(f"{name} launch event identity is invalid")

    for field in (
        "phase",
        "model",
        "setting",
        "grad_lr",
        "profile_level",
        "effective_overrides",
        "execution_id",
        "run_id",
        "plan_sha256",
        "model_sha256",
    ):
        if identity.get(field) != context[field]:
            raise ResultError(f"{name}.{field} disagrees with current manifest")
    generation = identity.get("generation")
    if type(generation) is not int or generation < 0:
        raise ResultError(f"{name}.generation must be non-negative")
    overrides = _require_mapping(
        identity.get("effective_overrides"),
        f"{name}.effective_overrides",
    )
    if any(
        not isinstance(key, str) or type(item) is not int
        for key, item in overrides.items()
    ):
        raise ResultError(f"{name}.effective_overrides must be integer-valued")
    status = _validate_retirement_status(
        ledger_status=identity.get("status"),
        manifest_status=identity.get("manifest_status"),
        context=context,
        parsed_records=parsed_records,
        manifest_path=manifest_path,
        name=name,
    )

    details = _require_mapping(
        event.get("details"), f"campaign event {launch_seq}.details"
    )
    expected_event = {
        "task_id": task_id,
        "model": context["model"],
        "setting": context["setting"],
        "grad_lr": context["grad_lr"],
        "profile_level": context["profile_level"],
        "generation": generation,
        "output_dir": str(manifest_path.parent),
        "plan_sha256": context["plan_sha256"],
        "phase": context["phase"],
        "effective_overrides": context["effective_overrides"],
    }
    for field, expected in expected_event.items():
        if details.get(field) != expected:
            raise ResultError(
                f"{name} disagrees with launch event field {field!r}"
            )
    return {
        "identity": dict(identity),
        "manifest_path": manifest_path,
        "log_path": log_path,
        "context": context,
        "status": status,
        "task_id": task_id,
        "launch_event_seq": launch_seq,
    }


def _validate_retirement_profile(
    value: Any,
    *,
    name: str,
    profiles: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    phase: str,
) -> dict[str, Any]:
    profile = _require_mapping(value, name)
    _require_exact_keys(profile, _RETIREMENT_PROFILE_KEYS, name)
    level = profile.get("level")
    generation = profile.get("generation")
    if type(level) is not int or level < 0 or level >= len(profiles):
        raise ResultError(f"{name}.level is outside the reviewed OOM ladder")
    if type(generation) is not int or generation < 0:
        raise ResultError(f"{name}.generation must be non-negative")
    overrides = _require_mapping(
        profile.get("effective_overrides"), f"{name}.effective_overrides"
    )
    if any(
        not isinstance(key, str) or type(item) is not int
        for key, item in overrides.items()
    ):
        raise ResultError(f"{name}.effective_overrides must be integer-valued")
    policy = _require_mapping(
        plan.get("qwen3_32b_memory_policy"),
        "embedded plan.qwen3_32b_memory_policy",
    )
    ladder_name = "tuning_ladders" if phase == "tune" else "final_ladders"
    ladders = _require_mapping(
        policy.get(ladder_name),
        f"embedded plan.qwen3_32b_memory_policy.{ladder_name}",
    )
    phase_config = _require_mapping(
        plan.get("tuning" if phase == "tune" else "final"),
        f"embedded plan.{phase}",
    )
    if set(overrides) - set(ladders):
        raise ResultError(f"{name}.effective_overrides contains non-ladder knobs")
    effective_values = {
        ladder_name_item: overrides.get(
            ladder_name_item, phase_config.get(ladder_name_item)
        )
        for ladder_name_item in ladders
    }
    expected_dynamic = profiles[level]["values"]
    if any(
        effective_values.get(knob) != expected
        for knob, expected in expected_dynamic.items()
    ):
        raise ResultError(
            f"{name}.effective_overrides is not reviewed ladder level {level}"
        )
    for knob, raw_values in ladders.items():
        if effective_values[knob] not in raw_values:
            raise ResultError(
                f"{name}.effective_overrides uses an unreviewed {knob} value"
            )
        if (
            knob in overrides
            and overrides[knob] == phase_config.get(knob)
        ):
            raise ResultError(
                f"{name}.effective_overrides redundantly records nominal {knob}"
            )
    return {
        "level": level,
        "generation": generation,
        "effective_overrides": dict(overrides),
    }


def _realq_manifest_contexts(
    manifests: Sequence[Path],
) -> dict[Path, dict[str, Any]]:
    contexts: dict[Path, dict[str, Any]] = {}
    for manifest_path in manifests:
        try:
            manifest = _read_manifest(manifest_path)
            if _is_precompute_manifest(manifest):
                continue
            command = manifest.get("command")
            argv = command.get("argv") if isinstance(command, Mapping) else None
            if not isinstance(argv, list) or "realq.ptq" not in argv:
                continue
            log_path = manifest_path.parent / LOG_FILENAME
            if not log_path.is_file():
                continue
            contexts[manifest_path.resolve()] = _retirement_manifest_context(
                manifest_path.resolve(), log_path.resolve()
            )
        except ResultError:
            # Normal result/informational validation reports malformed manifests.
            # Retirement completeness is checked only for identifiable contexts.
            continue
    return contexts


def _consume_retirement_ledger(
    root: Path,
    *,
    manifests: Sequence[Path],
    parsed_records: Mapping[Path, ParsedResult],
) -> dict[str, Any]:
    ledger_path = root / RETIREMENT_LEDGER_RELATIVE_PATH
    contexts = _realq_manifest_contexts(manifests)
    base_result = {
        "path": str(ledger_path),
        "present": ledger_path.is_file(),
        "entry_count": 0,
        "valid": True,
        "errors": [],
        "retired_paths": set(),
        "retired_attempts": [],
    }
    if not ledger_path.is_file():
        noninitial = [
            str(path)
            for path, context in contexts.items()
            if context["profile_level"] != 0
        ]
        if noninitial:
            base_result["valid"] = False
            base_result["errors"] = [
                "missing retirement ledger for non-initial OOM profile "
                + ", ".join(sorted(noninitial))
            ]
        return base_result

    try:
        raw = ledger_path.read_bytes()
        text = raw.decode("utf-8")
        if raw and not raw.endswith(b"\n"):
            raise ResultError("retirement ledger must end with a newline")
        raw_lines = text.splitlines()
        if any(not line for line in raw_lines):
            raise ResultError("retirement ledger must not contain blank lines")
        entries = [
            _require_mapping(
                _json_without_duplicate_keys(
                    line, f"retirement ledger line {index}"
                ),
                f"retirement ledger line {index}",
            )
            for index, line in enumerate(raw_lines, 1)
        ]
        if not entries:
            noninitial = [
                str(path)
                for path, context in contexts.items()
                if context["profile_level"] != 0
            ]
            if noninitial:
                raise ResultError(
                    "empty retirement ledger cannot authorize non-initial "
                    "OOM profiles"
                )
            return base_result

        first_campaign_id = _require_nonempty_string(
            entries[0].get("campaign_id"),
            "retirement ledger campaign_id",
        )
        events = _load_campaign_launch_events(
            root, campaign_id=first_campaign_id
        )
        previous_hash: str | None = None
        active_profiles: dict[
            tuple[str, str, str, str | None], dict[str, Any]
        ] = {}
        retired_paths: set[Path] = set()
        retired_launches: set[tuple[str, int]] = set()
        public_retired: list[dict[str, Any]] = []

        for index, entry in enumerate(entries, 1):
            name = f"retirement ledger entry {index}"
            _require_exact_keys(entry, _RETIREMENT_ENTRY_KEYS, name)
            if entry.get("schema_version") != RETIREMENT_SCHEMA_VERSION:
                raise ResultError(f"{name}.schema_version must be 1")
            if entry.get("seq") != index:
                raise ResultError(f"{name}.seq must be the strict 1-based index")
            timestamp = _require_nonempty_string(
                entry.get("timestamp_utc"), f"{name}.timestamp_utc"
            )
            try:
                parsed_timestamp = dt.datetime.fromisoformat(
                    timestamp.replace("Z", "+00:00")
                )
            except ValueError as exc:
                raise ResultError(f"{name}.timestamp_utc is invalid") from exc
            if parsed_timestamp.tzinfo is None:
                raise ResultError(f"{name}.timestamp_utc must include timezone")
            if entry.get("campaign_id") != first_campaign_id:
                raise ResultError(f"{name}.campaign_id changed within the ledger")
            if entry.get("prev_record_sha256") != previous_hash:
                raise ResultError(f"{name}.prev_record_sha256 breaks the hash chain")
            record_sha = _require_sha256(
                entry.get("record_sha256"), f"{name}.record_sha256"
            )
            unsigned = dict(entry)
            unsigned.pop("record_sha256")
            if _canonical_sha256(unsigned) != record_sha:
                raise ResultError(f"{name}.record_sha256 is invalid")
            canonical_line = json.dumps(
                entry,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if raw_lines[index - 1] != canonical_line:
                raise ResultError(f"{name} is not canonical JSON")
            previous_hash = record_sha

            phase = entry.get("phase")
            if phase not in {"tune", "final"}:
                raise ResultError(f"{name}.phase must be tune or final")
            model = _require_nonempty_string(entry.get("model"), f"{name}.model")
            setting = entry.get("setting")
            if phase == "tune":
                setting = _require_nonempty_string(setting, f"{name}.setting")
            elif setting is not None:
                raise ResultError(f"{name}.setting must be null for final")
            if entry.get("reason") != "oom_profile_transition":
                raise ResultError(
                    f"{name}.reason must equal 'oom_profile_transition'"
                )
            plan_sha = _require_sha256(
                entry.get("plan_sha256"), f"{name}.plan_sha256"
            )

            trigger = _validate_retirement_identity(
                root,
                entry.get("trigger"),
                name=f"{name}.trigger",
                campaign_id=first_campaign_id,
                events=events,
                parsed_records=parsed_records,
            )
            trigger_context = trigger["context"]
            if trigger["status"] != "oom":
                raise ResultError(f"{name}.trigger must have status 'oom'")
            if (
                trigger_context["plan_sha256"] != plan_sha
                or trigger_context["phase"] != phase
                or trigger_context["model"] != model
                or (phase == "tune" and trigger_context["setting"] != setting)
            ):
                raise ResultError(f"{name}.trigger disagrees with entry identity")
            plan = trigger_context["plan"]
            plan_models = _require_mapping(
                plan.get("models"), "embedded plan.models"
            )
            plan_settings = _require_mapping(
                plan.get("settings"), "embedded plan.settings"
            )
            if model not in plan_models or (
                phase == "tune" and setting not in plan_settings
            ):
                raise ResultError(f"{name} model/setting is absent from the plan")
            profiles = _reviewed_memory_profiles(plan, phase=phase)
            old_profile = _validate_retirement_profile(
                entry.get("old_profile"),
                name=f"{name}.old_profile",
                profiles=profiles,
                plan=plan,
                phase=phase,
            )
            replacement = _validate_retirement_profile(
                entry.get("replacement_profile"),
                name=f"{name}.replacement_profile",
                profiles=profiles,
                plan=plan,
                phase=phase,
            )
            if (
                replacement["level"] != old_profile["level"] + 1
                or replacement["generation"] != old_profile["generation"] + 1
            ):
                raise ResultError(
                    f"{name} replacement is not the strict next ladder generation"
                )
            static_keys = (
                set(old_profile["effective_overrides"])
                | set(replacement["effective_overrides"])
            ) - set(profiles[0]["values"])
            phase_config = _require_mapping(
                plan.get("tuning" if phase == "tune" else "final"),
                f"embedded plan.{phase}",
            )
            if any(
                old_profile["effective_overrides"].get(
                    key, phase_config.get(key)
                )
                != replacement["effective_overrides"].get(
                    key, phase_config.get(key)
                )
                for key in static_keys
            ):
                raise ResultError(
                    f"{name} replacement changed a static-cache profile knob"
                )
            chain_key = (plan_sha, str(phase), model, setting)
            expected_old = active_profiles.get(chain_key)
            if expected_old is None:
                continuous = (
                    old_profile["level"] == 0
                    and old_profile["generation"] == 0
                )
            else:
                continuous = old_profile == expected_old
            if not continuous:
                raise ResultError(f"{name}.old_profile is not chain-continuous")

            retired_value = entry.get("retired_manifests")
            if not isinstance(retired_value, list) or not retired_value:
                raise ResultError(f"{name}.retired_manifests must be non-empty")
            validated: list[dict[str, Any]] = []
            for retired_index, retired_value_item in enumerate(retired_value):
                item = _validate_retirement_identity(
                    root,
                    retired_value_item,
                    name=f"{name}.retired_manifests[{retired_index}]",
                    campaign_id=first_campaign_id,
                    events=events,
                    parsed_records=parsed_records,
                )
                context = item["context"]
                if (
                    context["plan_sha256"] != plan_sha
                    or context["phase"] != phase
                    or context["model"] != model
                    or (
                        phase == "tune"
                        and context["setting"] != setting
                    )
                    or (
                        phase == "final"
                        and context["setting"] not in plan_settings
                    )
                    or context["profile_level"] != old_profile["level"]
                    or item["identity"]["generation"]
                    != old_profile["generation"]
                    or item["identity"]["effective_overrides"]
                    != old_profile["effective_overrides"]
                    or context["model_sha256"]
                    != trigger_context["model_sha256"]
                ):
                    raise ResultError(
                        f"{name}.retired_manifests[{retired_index}] "
                        "does not belong to the retired wave"
                    )
                path = item["manifest_path"]
                launch_key = (item["task_id"], item["launch_event_seq"])
                if path in retired_paths or launch_key in retired_launches:
                    raise ResultError(
                        f"{name} contains a duplicate/conflicting retirement"
                    )
                retired_paths.add(path)
                retired_launches.add(launch_key)
                validated.append(item)

            absolute_order = [
                item["identity"]["manifest"]["absolute_path"]
                for item in validated
            ]
            if absolute_order != sorted(absolute_order):
                raise ResultError(
                    f"{name}.retired_manifests must be sorted by absolute_path"
                )
            trigger_matches = [
                item
                for item in validated
                if item["identity"] == trigger["identity"]
            ]
            if len(trigger_matches) != 1:
                raise ResultError(
                    f"{name}.trigger must occur exactly once in retired_manifests"
                )

            active_profiles[chain_key] = replacement
            for item in validated:
                record = parsed_records.get(item["manifest_path"])
                public = {
                    "kind": "retired_attempt",
                    "ledger_seq": index,
                    "ledger_record_sha256": record_sha,
                    "campaign_id": first_campaign_id,
                    "reason": entry["reason"],
                    "task_id": item["task_id"],
                    "launch_event_seq": item["launch_event_seq"],
                    "status": item["status"],
                    "manifest_status": item["identity"]["manifest_status"],
                    "phase": item["context"]["phase"],
                    "model": item["context"]["model"],
                    "setting": item["context"]["setting"],
                    "grad_lr": item["context"]["grad_lr"],
                    "profile_level": item["context"]["profile_level"],
                    "generation": item["identity"]["generation"],
                    "effective_overrides": dict(
                        item["context"]["effective_overrides"]
                    ),
                    "manifest": str(item["manifest_path"]),
                    "log": str(item["log_path"]),
                    "manifest_sha256": item["identity"]["manifest"]["sha256"],
                    "log_sha256": item["identity"]["log"]["sha256"],
                    "counts_toward_tune_budget": phase == "tune",
                    "replacement_profile": dict(replacement),
                }
                if record is not None:
                    public["result"] = record.public_dict()
                public_retired.append(public)

        for manifest_path, context in contexts.items():
            setting_key = context["setting"] if context["phase"] == "tune" else None
            chain_key = (
                context["plan_sha256"],
                context["phase"],
                context["model"],
                setting_key,
            )
            plan_profiles = _reviewed_memory_profiles(
                context["plan"], phase=context["phase"]
            )
            active = active_profiles.get(
                chain_key,
                {
                    "level": 0,
                    "generation": 0,
                    "effective_overrides": dict(
                        plan_profiles[0]["effective_overrides"]
                    ),
                },
            )
            if context["profile_level"] > active["level"]:
                raise ResultError(
                    "missing retirement transition before non-current profile "
                    f"manifest {manifest_path}"
                )
            if (
                context["profile_level"] < active["level"]
                and manifest_path not in retired_paths
            ):
                raise ResultError(
                    "old-profile manifest is missing from retirement ledger: "
                    f"{manifest_path}"
                )
            if (
                context["profile_level"] == active["level"]
                and manifest_path in retired_paths
            ):
                raise ResultError(
                    "retirement ledger illegally retires a current-profile "
                    f"manifest: {manifest_path}"
                )

        base_result.update(
            {
                "entry_count": len(entries),
                "retired_paths": retired_paths,
                "retired_attempts": public_retired,
            }
        )
        return base_result
    except (OSError, UnicodeDecodeError, ResultError) as exc:
        base_result["valid"] = False
        base_result["errors"] = [str(exc)]
        # A malformed ledger has no authority to hide any result.
        base_result["retired_paths"] = set()
        base_result["retired_attempts"] = []
        return base_result


def _validate_manual_lr_freeze_evidence(
    record: ParsedResult,
    freeze: Mapping[str, Any],
    records_by_manifest: Mapping[str, ParsedResult],
    *,
    expected_tune_protocol_sha256: str,
    reconciled_attempt_count: int | None = None,
) -> list[str]:
    ledger = _require_mapping(
        freeze.get("ledger"), "manual LR freeze ledger"
    )
    selection = _require_mapping(
        freeze.get("selection"), "manual LR freeze selection"
    )
    source_identity = _require_mapping(
        freeze.get("source_identity_before"),
        "manual LR freeze source_identity_before",
    )
    selected_lr = float(selection["selected_lr"])
    decision_kind = str(selection["decision_kind"])
    expected_plan_sha = str(ledger["expected_plan_sha256"])
    frozen_identity_keys = (
        "model_sha256",
        "runner_sha256",
        "executor_sha256",
        "numerical_source_sha256",
    )
    evidence_records: list[ParsedResult] = []
    evidence_lrs: list[float] = []
    for index, evidence_value in enumerate(selection["evidence"]):
        name = f"manual LR freeze evidence[{index}]"
        evidence = _require_mapping(evidence_value, name)
        try:
            manifest_path = Path(str(evidence["manifest"])).resolve(strict=True)
            log_path = Path(str(evidence["log"])).resolve(strict=True)
        except OSError as exc:
            raise ResultError(f"{name} artifact is unavailable: {exc}") from exc
        if str(manifest_path) != evidence["manifest"]:
            raise ResultError(f"{name}.manifest is not a canonical path")
        if str(log_path) != evidence["log"]:
            raise ResultError(f"{name}.log is not a canonical path")
        if _sha256_file(manifest_path) != evidence["manifest_sha256"]:
            raise ResultError(f"{name}.manifest_sha256 mismatch")
        if _sha256_file(log_path) != evidence["log_sha256"]:
            raise ResultError(f"{name}.log_sha256 mismatch")
        tune_record = records_by_manifest.get(str(manifest_path))
        if tune_record is None:
            raise ResultError(
                f"{name} does not reference an accepted, active result"
            )
        if (
            tune_record.method != "realq"
            or tune_record.phase != "tune"
            or tune_record.model != record.model
            or tune_record.setting != record.setting
        ):
            raise ResultError(
                f"{name} is not a REAL-Q tune result for "
                f"{record.model}/{record.setting}"
            )
        if tune_record.log_path != log_path:
            raise ResultError(f"{name}.log does not match the tune result")
        expected_fields = {
            "execution_id": tune_record.execution_id,
            "run_id": tune_record.run_id,
            "attempt_index": tune_record.attempt_index,
            "grad_lr": tune_record.grad_lr,
            "kl_wikitext2": tune_record.kl_wikitext2,
            "plan_sha256": tune_record.identities["plan_sha256"],
            "model_sha256": tune_record.identities["model_sha256"],
            "runner_sha256": tune_record.identities["runner_sha256"],
            "executor_sha256": tune_record.identities["executor_sha256"],
            "numerical_source_sha256": tune_record.identities[
                "numerical_source_sha256"
            ],
        }
        for field, expected in expected_fields.items():
            if evidence.get(field) != expected:
                raise ResultError(
                    f"{name}.{field} disagrees with its accepted tune result"
                )
        if evidence["plan_sha256"] != expected_plan_sha:
            raise ResultError(
                f"{name}.plan_sha256 disagrees with expected_plan_sha256"
            )
        if (
            _canonical_sha256(tune_record.plan_content)
            != expected_tune_protocol_sha256
        ):
            raise ResultError(
                f"{name} disagrees with the pre-freeze tune protocol hash"
            )
        if (
            _frozen_plan_protocol_sha256(record.plan_content)
            != _frozen_plan_protocol_sha256(tune_record.plan_content)
        ):
            raise ResultError(
                f"{name} uses a different frozen plan protocol"
            )
        for identity_key in frozen_identity_keys:
            if (
                record.identities[identity_key]
                != tune_record.identities[identity_key]
            ):
                raise ResultError(
                    f"{name} differs from the formal result in "
                    f"{identity_key}"
                )
        for identity_key in (
            "runner_sha256",
            "executor_sha256",
            "numerical_source_sha256",
        ):
            if (
                source_identity[identity_key]
                != tune_record.identities[identity_key]
            ):
                raise ResultError(
                    f"{name} differs from source_identity_before in "
                    f"{identity_key}"
                )
        evidence_records.append(tune_record)
        assert tune_record.grad_lr is not None
        evidence_lrs.append(float(tune_record.grad_lr))

    if decision_kind == "user_override_interpolated":
        if not any(value < selected_lr for value in evidence_lrs):
            raise ResultError(
                "interpolated manual LR override lacks lower-side evidence"
            )
        if not any(value > selected_lr for value in evidence_lrs):
            raise ResultError(
                "interpolated manual LR override lacks upper-side evidence"
            )
    else:
        tune_group = [
            item
            for item in records_by_manifest.values()
            if item.method == "realq"
            and item.phase == "tune"
            and item.model == record.model
            and item.setting == record.setting
        ]
        if not tune_group:
            raise ResultError(
                "exact manual LR decision has no accepted active tune points"
            )
        config_hashes = {
            _canonical_sha256(_selection_config(item))
            for item in tune_group
        }
        if len(config_hashes) != 1:
            raise ResultError(
                "exact manual LR tune points have non-LR configuration or "
                "source drift"
            )
        for item in tune_group:
            if (
                _canonical_sha256(item.plan_content)
                != expected_tune_protocol_sha256
                or _frozen_plan_protocol_sha256(record.plan_content)
                != _frozen_plan_protocol_sha256(item.plan_content)
            ):
                raise ResultError(
                    "exact manual LR tune points do not share the frozen "
                    "plan protocol"
                )
            for identity_key in frozen_identity_keys:
                if record.identities[identity_key] != item.identities[
                    identity_key
                ]:
                    raise ResultError(
                        "exact manual LR tune point differs from formal "
                        f"result in {identity_key}"
                    )
            for identity_key in (
                "runner_sha256",
                "executor_sha256",
                "numerical_source_sha256",
            ):
                if source_identity[identity_key] != item.identities[
                    identity_key
                ]:
                    raise ResultError(
                        "exact manual LR tune point differs from frozen "
                        f"source identity in {identity_key}"
                    )
        minimum_kl = min(
            float(item.kl_wikitext2) for item in tune_group
        )
        minima = [
            item
            for item in tune_group
            if item.kl_wikitext2 == minimum_kl
        ]
        if len(minima) != 1:
            raise ResultError(
                "exact manual LR decision has a tied exact KL minimum"
            )
        winner = minima[0]
        if (
            selection["selected_manifest"] != str(winner.manifest_path)
            or selected_lr != winner.grad_lr
        ):
            raise ResultError(
                "exact manual LR decision is not the unique exact KL "
                "minimum across all accepted active tune points"
            )
        if record.model == "qwen3-32b":
            ordered = sorted(
                tune_group,
                key=lambda item: (
                    float(item.grad_lr),
                    str(item.manifest_path),
                ),
            )
            ordered_lrs = [float(item.grad_lr) for item in ordered]
            if len(ordered_lrs) != len(set(ordered_lrs)):
                raise ResultError(
                    "qwen3-32b exact manual LR decision has duplicate "
                    "accepted LR points"
                )
            winner_index = ordered.index(winner)
            if winner_index == 0 or winner_index == len(ordered) - 1:
                raise ResultError(
                    "qwen3-32b exact manual LR decision must be bracketed "
                    "by the nearest accepted active LR point on both sides"
                )
            nearest_lower = ordered[winner_index - 1]
            nearest_upper = ordered[winner_index + 1]
            side_kls = (
                float(nearest_lower.kl_wikitext2),
                float(nearest_upper.kl_wikitext2),
            )
            if any(side_kl <= minimum_kl for side_kl in side_kls):
                raise ResultError(
                    "qwen3-32b exact manual LR decision requires its nearest "
                    "accepted side points to have strictly worse Exact KL"
                )
            if (
                type(reconciled_attempt_count) is not int
                or reconciled_attempt_count <= 0
                or reconciled_attempt_count > 20
            ):
                raise ResultError(
                    "qwen3-32b exact manual LR decision lacks a valid "
                    "reconciled launch count"
                )
            if reconciled_attempt_count < 8:
                if minimum_kl == 0.0:
                    gaps_too_large = True
                else:
                    best_decimal = Decimal(str(minimum_kl))
                    gaps_too_large = any(
                        (
                            Decimal(str(side_kl)) - best_decimal
                        )
                        / best_decimal
                        > Decimal("0.02")
                        for side_kl in side_kls
                    )
                if gaps_too_large:
                    raise ResultError(
                        "qwen3-32b exact manual LR decision is premature: "
                        "fewer than 8 reconciled launches require both "
                        "nearest-side Exact KL gaps to be at most 2%"
                    )
    return [str(item.manifest_path) for item in evidence_records]


def _validate_final_lrs(
    records: Sequence[ParsedResult],
    selections: Sequence[Mapping[str, Any]],
    *,
    root: Path,
    retirement_ledger: Mapping[str, Any],
    rejected: Sequence[Mapping[str, Any]],
    informational: Sequence[Mapping[str, Any]],
    retired_attempts: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    by_key = {
        (selection["model"], selection["setting"]): selection
        for selection in selections
    }
    records_by_manifest = {
        str(record.manifest_path): record for record in records
    }
    validations: list[dict[str, Any]] = []
    by_manifest: dict[str, dict[str, Any]] = {}
    reconciliation_cache: dict[
        str, tuple[dict[str, Any] | None, str | None]
    ] = {}
    for record in records:
        if record.method != "realq" or record.phase != "final":
            continue
        selection = by_key.get((record.model, record.setting))
        issues: list[str] = []
        freeze_errors: list[str] = []
        decision_kind = "exact"
        freeze_selected_lr = None
        freeze_ledger_sha = None
        freeze_artifact_sha = None
        freeze_evidence_manifests: list[str] = []
        freeze_attestation: dict[str, Any] | None = None
        freeze_valid = False
        freeze: dict[str, Any] | None = None
        selected_lr = None
        ledger_lr = None
        ledger = record.plan_content.get(
            "selected_grad_lr_by_model_setting"
        )
        if not isinstance(ledger, Mapping):
            issues.append("missing_embedded_selected_lr_ledger")
        else:
            model_ledger = ledger.get(record.model)
            if not isinstance(model_ledger, Mapping):
                issues.append("missing_embedded_model_selected_lrs")
            else:
                ledger_value = model_ledger.get(record.setting)
                if (
                    not _is_number(ledger_value)
                    or not math.isfinite(float(ledger_value))
                    or float(ledger_value) < 0
                ):
                    issues.append("invalid_embedded_selected_lr")
                else:
                    ledger_lr = float(ledger_value)
                    if record.grad_lr != ledger_lr:
                        issues.append("final_lr_ledger_mismatch")
        try:
            freeze = _manual_lr_freeze_selection(
                record.plan_content,
                model=record.model,
                setting=record.setting,
            )
        except ResultError as exc:
            issues.append("manual_lr_selection_freeze_invalid")
            freeze_errors.append(str(exc))
        if freeze is not None:
            freeze_ledger_sha = freeze["wrapper_ledger_sha256"]
            freeze_artifact_sha = freeze["artifact"]["sha256"]
            freeze_selection = _require_mapping(
                freeze["selection"], "manual LR freeze selection"
            )
            decision_kind = str(freeze_selection["decision_kind"])
            freeze_selected_lr = float(freeze_selection["selected_lr"])
            if (
                record.grad_lr != freeze_selected_lr
                or ledger_lr != freeze_selected_lr
            ):
                issues.append("manual_lr_selection_freeze_lr_mismatch")
            try:
                cache_key = _frozen_plan_protocol_sha256(
                    record.plan_content
                )
                if cache_key not in reconciliation_cache:
                    try:
                        snapshot = _manual_lr_reconciliation_snapshot(
                            root=root,
                            plan=record.plan_content,
                            records=records,
                            tune_selections=selections,
                            retirement_ledger=retirement_ledger,
                            rejected=rejected,
                            informational=informational,
                            retired_attempts=retired_attempts,
                            campaign_id=freeze["ledger"]["campaign_id"],
                            expected_plan_sha256=freeze["ledger"][
                                "expected_plan_sha256"
                            ],
                            source_identity=freeze[
                                "source_identity_before"
                            ],
                        )
                        reconciliation_cache[cache_key] = (snapshot, None)
                    except ResultError as exc:
                        reconciliation_cache[cache_key] = (None, str(exc))
                reconciliation_snapshot, reconciliation_error = (
                    reconciliation_cache[cache_key]
                )
                if reconciliation_error is not None:
                    raise ResultError(reconciliation_error)
                assert reconciliation_snapshot is not None
                freeze_attestation = (
                    _validate_manual_lr_freeze_artifacts(
                        root,
                        record,
                        freeze,
                        reconciliation_snapshot=reconciliation_snapshot,
                    )
                )
                reconciled_counts = freeze_attestation[
                    "state_reconciliation"
                ]["group_attempt_counts"]
                matching_counts = [
                    item["attempt_count"]
                    for item in reconciled_counts
                    if item["model"] == record.model
                    and item["setting"] == record.setting
                ]
                if len(matching_counts) != 1:
                    raise ResultError(
                        "manual LR freeze reconciliation has no unique "
                        "attempt count for the formal row"
                    )
                freeze_evidence_manifests = (
                    _validate_manual_lr_freeze_evidence(
                        record,
                        freeze,
                        records_by_manifest,
                        expected_tune_protocol_sha256=freeze_attestation[
                            "tune_protocol_sha256"
                        ],
                        reconciled_attempt_count=matching_counts[0],
                    )
                )
                freeze_valid = True
            except ResultError as exc:
                issues.append("manual_lr_selection_freeze_invalid")
                freeze_errors.append(str(exc))

        if (
            selection is not None
            and selection.get("status") == "selected"
            and isinstance(selection.get("selected"), Mapping)
        ):
            selected_lr = selection["selected"]["grad_lr"]
        if (
            decision_kind == "user_override_interpolated"
            and freeze is not None
        ):
            pass
        elif decision_kind == "exact" and freeze is not None and freeze_valid:
            selected_lr = freeze_selected_lr
        elif selection is None:
            issues.append("missing_tune_selection")
        elif selection["status"] != "selected":
            issues.append("tune_selection_unresolved")
        else:
            if record.grad_lr != selected_lr:
                issues.append("final_lr_mismatch")
            selected_manifest = selection["selected"]["manifest"]
            tune_record = records_by_manifest.get(selected_manifest)
            frozen_identity_keys = (
                "model_sha256",
                "runner_sha256",
                "executor_sha256",
                "numerical_source_sha256",
            )
            if tune_record is None or (
                _frozen_plan_protocol_sha256(record.plan_content)
                != _frozen_plan_protocol_sha256(tune_record.plan_content)
            ) or any(
                record.identities[key] != tune_record.identities[key]
                for key in frozen_identity_keys
            ):
                issues.append("final_tune_provenance_mismatch")
            if freeze is not None:
                freeze_selection = _require_mapping(
                    freeze["selection"], "manual LR freeze selection"
                )
                if (
                    freeze_selected_lr != selected_lr
                    or freeze_selection["selected_manifest"]
                    != selected_manifest
                ):
                    issues.append(
                        "manual_lr_selection_freeze_exact_mismatch"
                    )
        issues = list(dict.fromkeys(issues))
        status = "valid" if not issues else "invalid"
        validation = {
            "manifest": str(record.manifest_path),
            "model": record.model,
            "setting": record.setting,
            "status": status,
            "final_grad_lr": record.grad_lr,
            "embedded_selected_grad_lr": ledger_lr,
            "selected_tune_grad_lr": selected_lr,
            "decision_kind": decision_kind,
            "freeze_selected_grad_lr": freeze_selected_lr,
            "freeze_ledger_sha256": freeze_ledger_sha,
            "freeze_artifact_sha256": freeze_artifact_sha,
            "freeze_evidence_manifests": freeze_evidence_manifests,
            "freeze_attestation": freeze_attestation,
            "freeze_resolves_tune_selection": freeze_valid,
            "freeze_errors": freeze_errors,
            "issues": issues,
        }
        validations.append(validation)
        by_manifest[str(record.manifest_path)] = validation
    return validations, by_manifest


def _matrix_completeness(
    records: Sequence[ParsedResult],
    selections: Sequence[Mapping[str, Any]],
    *,
    required: bool,
    freeze_resolved_tunes: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    if not records:
        return {
            "required": required,
            "complete": False,
            "issues": ["no accepted records define an experiment matrix"],
            "missing_final": [],
            "duplicate_final": [],
            "missing_tune_selections": [],
        }
    # The selected-LR ledger is intentionally all-null in proxy manifests and
    # populated before formal rendering.  Normalize that one reviewed mutable
    # field while requiring every other embedded-plan value to stay frozen.
    specs = {
        _frozen_plan_protocol_sha256(record.plan_content)
        for record in records
    }
    plan = records[0].plan_content
    models = tuple(
        _require_mapping(plan.get("models"), "embedded plan.models").keys()
    )
    settings = tuple(
        _require_mapping(plan.get("settings"), "embedded plan.settings").keys()
    )
    methods_value = plan.get("methods")
    if not isinstance(methods_value, list):
        raise ResultError("embedded plan.methods must be a list")
    methods = tuple(str(value) for value in methods_value)
    expected_final = {
        (model, "BF16", "bf16")
        for model in models
    } | {
        (model, setting, method)
        for model in models
        for setting in settings
        for method in methods
        if method != "bf16"
    }
    final_counts: dict[tuple[str, str, str], int] = {}
    for record in records:
        if record.phase != "final":
            continue
        key = (record.model, record.setting, record.method)
        final_counts[key] = final_counts.get(key, 0) + 1
    missing_final = sorted(expected_final - set(final_counts))
    duplicate_final = sorted(
        key for key, count in final_counts.items() if count > 1
    )
    selected_tunes = {
        (selection["model"], selection["setting"])
        for selection in selections
        if selection["status"] == "selected"
    }
    selected_tunes.update(freeze_resolved_tunes or set())
    expected_tunes = {
        (model, setting) for model in models for setting in settings
    }
    missing_tunes = sorted(expected_tunes - selected_tunes)
    issues: list[str] = []
    if len(specs) != 1:
        issues.append("accepted records do not share one frozen matrix/plan")
    if missing_final:
        issues.append(f"{len(missing_final)} formal matrix row(s) are missing")
    if duplicate_final:
        issues.append(
            f"{len(duplicate_final)} formal matrix row(s) have duplicates"
        )
    if missing_tunes:
        issues.append(
            f"{len(missing_tunes)} REAL-Q tune selection(s) are missing"
        )
    return {
        "required": required,
        "complete": not issues,
        "issues": issues,
        "missing_final": [
            {"model": model, "setting": setting, "method": method}
            for model, setting, method in missing_final
        ],
        "duplicate_final": [
            {"model": model, "setting": setting, "method": method}
            for model, setting, method in duplicate_final
        ],
        "missing_tune_selections": [
            {"model": model, "setting": setting}
            for model, setting in missing_tunes
        ],
    }


def _rank_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return 0.0
    x_order = {value: index for index, value in enumerate(sorted(set(xs)))}
    y_order = {value: index for index, value in enumerate(sorted(set(ys)))}
    xr = [float(x_order[value]) for value in xs]
    yr = [float(y_order[value]) for value in ys]
    x_mean = sum(xr) / len(xr)
    y_mean = sum(yr) / len(yr)
    numerator = sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xr, yr)
    )
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xr)
        * sum((y - y_mean) ** 2 for y in yr)
    )
    return numerator / denominator if denominator else 0.0


def _detect_anomalies(
    records: Sequence[ParsedResult],
    selections: Sequence[Mapping[str, Any]],
    final_validations: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    informational: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    anomalies: list[dict[str, Any]] = []
    tune_groups: dict[tuple[str, str], list[ParsedResult]] = {}
    for record in records:
        if record.method == "realq" and record.phase == "tune":
            tune_groups.setdefault((record.model, record.setting), []).append(record)
    for (model, setting), group in sorted(tune_groups.items()):
        ordered = sorted(group, key=lambda record: float(record.grad_lr))
        kls = [float(record.kl_wikitext2) for record in ordered]
        ppls = [record.ppl_wikitext2 for record in ordered]
        decreases_after_increase = any(
            kls[index] < kls[index - 1]
            and any(
                kls[earlier] > kls[earlier - 1]
                for earlier in range(1, index)
            )
            for index in range(2, len(kls))
        )
        if decreases_after_increase:
            anomalies.append(
                {
                    "code": "non_unimodal_kl_curve",
                    "severity": "warning",
                    "model": model,
                    "setting": setting,
                    "message": "exact KL rises and later falls across sorted LR",
                }
            )
        for index in range(1, len(kls) - 1):
            if kls[index] > max(kls[index - 1], kls[index + 1]):
                anomalies.append(
                    {
                        "code": "kl_curve_spike",
                        "severity": "warning",
                        "model": model,
                        "setting": setting,
                        "grad_lr": ordered[index].grad_lr,
                        "message": "exact KL is a strict local spike",
                    }
                )
        correlation = _rank_correlation(kls, ppls)
        if correlation < -0.5:
            anomalies.append(
                {
                    "code": "ppl_kl_rank_inversion",
                    "severity": "warning",
                    "model": model,
                    "setting": setting,
                    "correlation": correlation,
                    "message": "PPL and exact KL ranks are strongly inverted",
                }
            )
        cache_paths = {
            record.params.get("static_cache_path") for record in group
        }
        if len(cache_paths) > 1:
            anomalies.append(
                {
                    "code": "static_cache_drift",
                    "severity": "error",
                    "model": model,
                    "setting": setting,
                    "values": sorted(str(value) for value in cache_paths),
                    "message": "proxy attempts used different static cache paths",
                }
            )
    for selection in selections:
        codes = {issue["code"] for issue in selection["issues"]}
        if "config_inconsistency" in codes:
            anomalies.append(
                {
                    "code": "config_drift",
                    "severity": "error",
                    "model": selection["model"],
                    "setting": selection["setting"],
                    "message": "non-LR proxy configuration drift was detected",
                }
            )
    for validation in final_validations:
        validation_issues = set(validation.get("issues", ()))
        if validation_issues & {
            "final_tune_provenance_mismatch",
            "manual_lr_selection_freeze_invalid",
        }:
            anomalies.append(
                {
                    "code": "config_drift",
                    "severity": "error",
                    "model": validation["model"],
                    "setting": validation["setting"],
                    "manifest": validation["manifest"],
                    "message": (
                        "formal REAL-Q provenance or its frozen manual-LR "
                        "evidence is invalid"
                    ),
                }
            )
        if validation_issues & {
            "final_lr_ledger_mismatch",
            "final_lr_mismatch",
            "manual_lr_selection_freeze_lr_mismatch",
            "manual_lr_selection_freeze_exact_mismatch",
        }:
            anomalies.append(
                {
                    "code": "final_lr_drift",
                    "severity": "error",
                    "model": validation["model"],
                    "setting": validation["setting"],
                    "manifest": validation["manifest"],
                    "message": (
                        "formal REAL-Q LR differs from its embedded ledger "
                        "or parsed proxy winner"
                    ),
                }
            )
    for item in rejected:
        error = " ".join(str(value) for value in item.get("errors", ()))
        if "exact KL" in error:
            code = "invalid_exact_kl"
        elif "exact PPL" in error:
            code = "invalid_exact_ppl"
        elif (
            "lm-eval version" in error
            or "python_packages.lm_eval" in error
            or "QA table" in error
        ):
            code = "metric_drift"
        else:
            continue
        anomalies.append(
            {
                "code": code,
                "severity": "error",
                "manifest": item.get("manifest"),
                "message": error,
            }
        )
    for item in informational:
        if (
            item.get("counts_toward_tune_budget") is True
            and item.get("attempt_context_valid") is False
        ):
            anomalies.append(
                {
                    "code": "attempt_metadata_drift",
                    "severity": "warning",
                    "manifest": item.get("manifest"),
                    "message": (
                        "tune launch counted toward budget using fail-soft "
                        "run-id classification because strict attempt metadata "
                        "was invalid"
                    ),
                }
            )
        if (
            item.get("kind") == "precompute_attempt"
            and item.get("status") in {"launch_failed", "failed", "wrapper_failed"}
        ):
            anomalies.append(
                {
                    "code": "precompute_failure",
                    "severity": "warning",
                    "manifest": item.get("manifest"),
                    "message": (
                        "a cache precompute attempt failed; it is excluded from "
                        "the numerical tuning budget"
                    ),
                }
            )
    return anomalies


def build_report(
    root: Path, *, require_complete_matrix: bool = False
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ResultError(f"result root is not a directory: {root}")
    manifests = sorted(root.rglob(MANIFEST_FILENAME))
    records: list[ParsedResult] = []
    rejected: list[dict[str, Any]] = []
    informational: list[dict[str, Any]] = []
    attempts_by_group: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = {}
    for manifest_path in manifests:
        try:
            manifest = _read_manifest(manifest_path)
        except ResultError as exc:
            rejected.append(
                {
                    "manifest": str(manifest_path.resolve()),
                    "errors": [str(exc)],
                }
            )
            continue
        attempt = _tune_attempt_context(manifest, manifest_path)
        if attempt is not None:
            attempts_by_group.setdefault(
                (attempt["model"], attempt["setting"]), []
            ).append(attempt)
        status = manifest.get("status")
        precompute = _is_precompute_manifest(manifest)
        if status in _INFORMATIONAL_STATUSES:
            informational.append(
                {
                    "manifest": str(manifest_path.resolve()),
                    "kind": (
                        "precompute_attempt"
                        if precompute
                        else "execution_attempt"
                    ),
                    "status": status,
                    "run_id": manifest.get("run_id"),
                    "execution_id": manifest.get("execution_id"),
                    "counts_toward_tune_budget": attempt is not None,
                    "tune_model": attempt.get("model") if attempt else None,
                    "tune_setting": attempt.get("setting") if attempt else None,
                    "grad_lr": attempt.get("grad_lr") if attempt else None,
                    "attempt_index": (
                        attempt.get("attempt_index") if attempt else None
                    ),
                    "attempt_context_valid": (
                        attempt.get("context_valid") if attempt else None
                    ),
                }
            )
            continue
        if status != "succeeded":
            rejected.append(
                {
                    "manifest": str(manifest_path.resolve()),
                    "errors": [f"unknown manifest.status {status!r}"],
                }
            )
            continue
        if precompute:
            informational.append(
                {
                    "manifest": str(manifest_path.resolve()),
                    "kind": "precompute",
                    "status": status,
                    "run_id": manifest.get("run_id"),
                    "execution_id": manifest.get("execution_id"),
                    "counts_toward_tune_budget": False,
                }
            )
            continue
        try:
            records.append(parse_result(manifest_path))
        except ResultError as exc:
            rejected.append(
                {
                    "manifest": str(manifest_path.resolve()),
                    "errors": [str(exc)],
                }
            )
    records.sort(
        key=lambda item: (
            item.model,
            item.setting,
            item.method,
            item.phase,
            item.grad_lr if item.grad_lr is not None else -1.0,
            str(item.manifest_path),
        )
    )
    parsed_records = {record.manifest_path: record for record in records}
    retirement = _consume_retirement_ledger(
        root,
        manifests=manifests,
        parsed_records=parsed_records,
    )
    retired_paths = retirement["retired_paths"]
    records = [
        record for record in records if record.manifest_path not in retired_paths
    ]
    selections, selection_by_manifest = _tune_selection(
        records, attempts_by_group
    )
    retirement_public = {
        key: value
        for key, value in retirement.items()
        if key not in {"retired_paths", "retired_attempts"}
    }
    final_validations, final_validation_by_manifest = _validate_final_lrs(
        records,
        selections,
        root=root,
        retirement_ledger=retirement_public,
        rejected=rejected,
        informational=informational,
        retired_attempts=retirement["retired_attempts"],
    )
    freeze_resolved_tunes = {
        (validation["model"], validation["setting"])
        for validation in final_validations
        if validation["status"] == "valid"
        and validation.get("freeze_resolves_tune_selection") is True
    }
    committed_invalid_resolutions: dict[str, dict[str, Any]] = {}
    for validation in final_validations:
        if validation.get("status") != "valid":
            continue
        attestation = validation.get("freeze_attestation")
        if not isinstance(attestation, Mapping):
            continue
        reconciliation = attestation.get("state_reconciliation")
        if not isinstance(reconciliation, Mapping):
            continue
        resolutions = reconciliation.get("invalid_result_resolutions")
        if not isinstance(resolutions, list):
            continue
        for entry in resolutions:
            if not isinstance(entry, Mapping):
                continue
            manifest_ref = entry.get("manifest")
            manifest = (
                manifest_ref.get("absolute_path")
                if isinstance(manifest_ref, Mapping)
                else None
            )
            if not isinstance(manifest, str):
                continue
            public = {
                "manifest": manifest,
                "errors": list(entry["expected_rejection_errors"]),
                "ledger_entry": dict(entry),
                "freeze_artifact_sha256": attestation[
                    "artifact_sha256"
                ],
            }
            previous = committed_invalid_resolutions.get(manifest)
            if previous is not None and previous != public:
                raise ResultError(
                    "valid formal rows disagree on an invalid-result "
                    f"resolution: {manifest}"
                )
            committed_invalid_resolutions[manifest] = public
    resolved_invalid_results: list[dict[str, Any]] = []
    unresolved_rejected: list[dict[str, Any]] = []
    for item in rejected:
        manifest = item.get("manifest")
        resolution = (
            committed_invalid_resolutions.get(manifest)
            if isinstance(manifest, str)
            else None
        )
        if resolution is not None and item.get("errors") == resolution[
            "errors"
        ]:
            resolved_invalid_results.append(resolution)
        else:
            unresolved_rejected.append(item)
    resolved_invalid_results.sort(key=lambda item: item["manifest"])
    for selection in selections:
        key = (selection["model"], selection["setting"])
        resolved_by_freeze = key in freeze_resolved_tunes
        selection["resolved_by_manual_freeze"] = resolved_by_freeze
        selection["effective_status"] = (
            "selected"
            if resolved_by_freeze
            else selection["status"]
        )
    matrix = _matrix_completeness(
        records,
        selections,
        required=require_complete_matrix,
        freeze_resolved_tunes=freeze_resolved_tunes,
    )
    public_records: list[dict[str, Any]] = []
    for record in records:
        public = record.public_dict()
        public["tune_selection"] = selection_by_manifest.get(
            str(record.manifest_path)
        )
        public["final_lr_validation"] = final_validation_by_manifest.get(
            str(record.manifest_path)
        )
        public_records.append(public)
    selection_ok = all(
        selection["status"] == "selected"
        or (
            selection["model"],
            selection["setting"],
        )
        in freeze_resolved_tunes
        for selection in selections
    )
    final_lr_ok = all(
        validation["status"] == "valid"
        for validation in final_validations
    )
    matrix_ok = matrix["complete"] if require_complete_matrix else True
    report_issues: list[str] = []
    if not manifests:
        report_issues.append("no execution_manifest.json files found")
    elif not records:
        report_issues.append("no result manifest passed validation")
    if unresolved_rejected:
        report_issues.append(
            "one or more rejected manifests lack a committed exact "
            "manual-invalid resolution"
        )
    if not final_lr_ok:
        report_issues.append(
            "one or more formal REAL-Q runs do not satisfy an exact or "
            "frozen reviewed LR decision"
        )
    if not selection_ok:
        report_issues.append(
            "one or more REAL-Q proxy searches remain unresolved"
        )
    if require_complete_matrix and not matrix["complete"]:
        report_issues.extend(
            f"complete-matrix gate: {issue}" for issue in matrix["issues"]
        )
    report_issues.extend(
        f"retirement ledger: {error}" for error in retirement["errors"]
    )
    anomalies = _detect_anomalies(
        records,
        selections,
        final_validations,
        rejected,
        informational,
    )
    return {
        "schema_version": 1,
        "root": str(root),
        "scope": (
            "complete_matrix" if require_complete_matrix else "incremental"
        ),
        "ok": (
            bool(records)
            and not unresolved_rejected
            and retirement["valid"]
            and selection_ok
            and final_lr_ok
            and matrix_ok
        ),
        "manifest_count": len(manifests),
        "accepted_count": len(records),
        "rejected_count": len(rejected),
        "unresolved_rejected_count": len(unresolved_rejected),
        "resolved_invalid_result_count": len(resolved_invalid_results),
        "informational_count": len(informational),
        "retired_attempt_count": len(retirement["retired_attempts"]),
        "issues": report_issues,
        "records": public_records,
        "rejected": rejected,
        "unresolved_rejected": unresolved_rejected,
        "resolved_invalid_results": resolved_invalid_results,
        "informational": informational,
        "retired_attempts": retirement["retired_attempts"],
        "retirement_ledger": retirement_public,
        "anomalies": anomalies,
        "realq_tune_selection": selections,
        "realq_final_lr_validation": final_validations,
        "matrix_completeness": matrix,
    }


_CSV_FIELDS = (
    "manifest",
    "execution_id",
    "run_id",
    "model",
    "setting",
    "method",
    "phase",
    "world_size",
    "attempt_index",
    "grad_lr",
    "final_layer_grad_lr",
    "kl_wikitext2",
    "ppl_wikitext2",
    "lm_eval_version",
    *EXPECTED_TASKS,
    "acc_avg",
    "tune_selection_status",
    "tune_selected",
    "tune_selection_issues",
    "final_lr_decision_kind",
    "freeze_selected_grad_lr",
    "freeze_ledger_sha256",
    "freeze_artifact_sha256",
    "freeze_evidence_manifests_json",
    "plan_sha256",
    "model_sha256",
    "runner_sha256",
    "executor_sha256",
    "numerical_source_sha256",
    "task_metric_keys_json",
    "params_json",
)


def report_csv(report: Mapping[str, Any]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_CSV_FIELDS)
    writer.writeheader()
    for record in report["records"]:
        selection = record.get("tune_selection") or {}
        final_validation = record.get("final_lr_validation") or {}
        row: dict[str, Any] = {
            key: record.get(key)
            for key in (
                "manifest",
                "execution_id",
                "run_id",
                "model",
                "setting",
                "method",
                "phase",
                "world_size",
                "attempt_index",
                "grad_lr",
                "final_layer_grad_lr",
                "kl_wikitext2",
                "ppl_wikitext2",
                "lm_eval_version",
                "acc_avg",
            )
        }
        row.update(record["tasks"])
        row.update(record["identities"])
        row["tune_selection_status"] = selection.get("status")
        row["tune_selected"] = selection.get("selected")
        row["tune_selection_issues"] = ";".join(selection.get("issues", []))
        row["final_lr_decision_kind"] = final_validation.get(
            "decision_kind"
        )
        row["freeze_selected_grad_lr"] = final_validation.get(
            "freeze_selected_grad_lr"
        )
        row["freeze_ledger_sha256"] = final_validation.get(
            "freeze_ledger_sha256"
        )
        row["freeze_artifact_sha256"] = final_validation.get(
            "freeze_artifact_sha256"
        )
        row["freeze_evidence_manifests_json"] = json.dumps(
            final_validation.get("freeze_evidence_manifests", []),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        row["task_metric_keys_json"] = json.dumps(
            record.get("task_metric_keys", {}),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        row["params_json"] = json.dumps(
            record["params"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        writer.writerow(row)
    return output.getvalue()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--json-out",
        default="-",
        help="JSON output path, or '-' for stdout (default)",
    )
    parser.add_argument(
        "--csv-out",
        help="optional CSV output path, or '-' for stdout",
    )
    parser.add_argument(
        "--require-complete-matrix",
        action="store_true",
        help=(
            "formal close-out gate: require every planned final row and every "
            "REAL-Q tune selection; incremental reports leave this disabled"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    if args.json_out == "-" and args.csv_out == "-":
        print("JSON and CSV cannot both target stdout", file=sys.stderr)
        return 2
    try:
        report = build_report(
            args.root,
            require_complete_matrix=args.require_complete_matrix,
        )
        json_text = (
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n"
        )
        csv_text = report_csv(report)
        if args.json_out == "-":
            sys.stdout.write(json_text)
        else:
            _atomic_write_text(Path(args.json_out), json_text)
        if args.csv_out == "-":
            sys.stdout.write(csv_text)
        elif args.csv_out:
            _atomic_write_text(Path(args.csv_out), csv_text)
    except (OSError, ResultError) as exc:
        print(f"lowbit_activation_results: {exc}", file=sys.stderr)
        return 2
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
