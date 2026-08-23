#!/usr/bin/env python3
"""Run the reviewed campaign with fail-closed formal timing.

Tune work and BF16 keep using the reviewed base executor.  Applicable formal
work uses the external timed executor and cannot become terminal-successful
until the *outer* executor has exited and its pinned sidecar passes the strict
GPU-hour parser.  The formal transition also freezes this controller and the
complete timing source set into campaign state, making a later base-controller
restart fail closed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
EXPERIMENT_DIR = SCRIPT_PATH.parent
for directory in (str(TOOLS_DIR), str(EXPERIMENT_DIR)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

import formal_timing_adapter as timing_adapter  # noqa: E402
import lowbit_activation_campaign as base_campaign  # noqa: E402
import lowbit_activation_gpu_hours as gpu_hours  # noqa: E402


TIMED_EXECUTOR = SCRIPT_PATH.with_name("formal_timed_execute.py")
BASE_EXECUTOR = REPO_ROOT / "tools" / "lowbit_activation_execute.py"
TIMING_GATE_SCHEMA_VERSION = 1
_MANIFEST_TERMINAL_STATUSES = {
    "succeeded",
    "failed",
    "terminated",
    "interrupted",
    "launch_failed",
    "wrapper_failed",
}


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _requires_formal_timing(spec: Mapping[str, Any]) -> bool:
    kind = spec.get("kind")
    method = spec.get("method")
    phase = spec.get("phase")
    target_phase = spec.get("target_phase")
    if kind == "realq_static":
        return target_phase == "final"
    if kind == "guided_saliency":
        return True
    if kind == "realq":
        return phase == "final"
    if kind == "baseline":
        return phase == "final" and method in {
            "gptaq",
            "guided_gptq",
        }
    return False


def _timing_gate_now() -> dict[str, Any]:
    source_set = timing_adapter.timing_source_set_identity()
    sources = {
        item["name"]: item
        for item in source_set["sources"]
    }
    required = {
        "formal_timed_campaign",
        "formal_timed_execute",
        "formal_timing_adapter",
        "formal_timing_sitecustomize",
        "lowbit_activation_campaign",
        "lowbit_activation_gpu_hours",
        "validate_guided_saliency",
    }
    if set(sources) != required:
        raise base_campaign.CampaignBlocked(
            "formal timing source set is incomplete or contains extras: "
            f"{sorted(sources)!r}"
        )
    payload: dict[str, Any] = {
        "schema_version": TIMING_GATE_SCHEMA_VERSION,
        "controller_path": str(SCRIPT_PATH),
        "controller_sha256": sources[
            "formal_timed_campaign"
        ]["sha256"],
        "base_campaign_path": str(
            timing_adapter.BASE_CAMPAIGN_PATH
        ),
        "base_campaign_sha256": sources[
            "lowbit_activation_campaign"
        ]["sha256"],
        "timed_executor_path": str(TIMED_EXECUTOR),
        "timed_executor_sha256": sources[
            "formal_timed_execute"
        ]["sha256"],
        "gpu_hour_aggregator_sha256": sources[
            "lowbit_activation_gpu_hours"
        ]["sha256"],
        "guided_validator_sha256": sources[
            "validate_guided_saliency"
        ]["sha256"],
        "source_set_sha256": source_set["source_set_sha256"],
        "sources": [
            {
                "name": item["name"],
                "original_path": item["original_path"],
                "sha256": item["sha256"],
            }
            for item in source_set["sources"]
        ],
    }
    payload["gate_sha256"] = _canonical_sha256(payload)
    return payload


def _formal_source_identity(
    base: Mapping[str, str],
    gate: Mapping[str, Any],
) -> dict[str, str]:
    if base.get("campaign_sha256") != gate.get(
        "base_campaign_sha256"
    ):
        raise base_campaign.CampaignBlocked(
            "base campaign source disagrees with the frozen formal timing "
            "source set"
        )
    result = dict(base)
    result["formal_timed_campaign_sha256"] = str(
        gate["controller_sha256"]
    )
    result["formal_timing_gate_sha256"] = str(gate["gate_sha256"])
    result["formal_timing_source_set_sha256"] = str(
        gate["source_set_sha256"]
    )
    result["formal_timed_executor_sha256"] = str(
        gate["timed_executor_sha256"]
    )
    return result


class FormalTimedCampaign(base_campaign.Campaign):
    """Base scheduler plus a frozen, provenance-safe formal timing gate."""

    def _formal_timing_gate(self) -> Mapping[str, Any] | None:
        formal = self.state.get("formal")
        if not isinstance(formal, Mapping):
            return None
        gate = formal.get("timing_gate")
        return gate if isinstance(gate, Mapping) else None

    def _state_has_formal_work(self) -> bool:
        formal = self.state.get("formal")
        if (
            isinstance(formal, Mapping)
            and formal.get("initialized") is True
        ):
            return True
        tasks = self.state.get("tasks")
        return isinstance(tasks, Mapping) and any(
            isinstance(task, Mapping)
            and isinstance(task.get("spec"), Mapping)
            and _requires_formal_timing(task["spec"])
            for task in tasks.values()
        )

    def _assert_timing_gate(self) -> Mapping[str, Any]:
        gate = self._formal_timing_gate()
        if gate is None:
            raise base_campaign.CampaignBlocked(
                "formal work exists without the frozen timing gate; "
                "refusing untimed formal execution"
            )
        current = _timing_gate_now()
        if dict(gate) != current:
            raise base_campaign.CampaignBlocked(
                "formal timing controller/source set drifted after the "
                "formal transition"
            )
        return gate

    def _current_source_identity(self) -> dict[str, str]:
        base = super()._current_source_identity()
        gate = self._formal_timing_gate()
        if gate is None:
            if self._state_has_formal_work():
                raise base_campaign.CampaignBlocked(
                    "formal campaign state lacks its timing gate"
                )
            return base
        current = self._assert_timing_gate()
        return _formal_source_identity(base, current)

    def _initialize_formal(self) -> None:
        formal = self.state.get("formal")
        if (
            isinstance(formal, Mapping)
            and formal.get("initialized") is True
        ):
            self._assert_timing_gate()
            return
        if self._state_has_formal_work():
            raise base_campaign.CampaignBlocked(
                "pre-existing formal tasks lack a frozen timing gate"
            )

        base_identity = super()._current_source_identity()
        expected = self.state.get("source_identity")
        if expected != base_identity:
            raise base_campaign.CampaignBlocked(
                "cannot transition to formal timing from a drifted base "
                "controller/source identity"
            )
        super()._initialize_formal()
        gate = _timing_gate_now()
        self.state["formal"]["timing_gate"] = gate
        self.state["source_identity"] = _formal_source_identity(
            base_identity, gate
        )
        self.emit(
            "state",
            "formal_timing_gate_pinned",
            details={
                "gate_sha256": gate["gate_sha256"],
                "source_set_sha256": gate["source_set_sha256"],
                "controller_sha256": gate["controller_sha256"],
                "base_campaign_sha256": gate[
                    "base_campaign_sha256"
                ],
                "timed_executor_sha256": gate[
                    "timed_executor_sha256"
                ],
            },
        )

    def _executor_cli(
        self,
        task: Mapping[str, Any],
        gpu_ids: Sequence[int],
    ) -> list[str]:
        command = super()._executor_cli(task, gpu_ids)
        if not _requires_formal_timing(task["spec"]):
            return command
        gate = self._assert_timing_gate()
        if len(command) < 2:
            raise base_campaign.CampaignBlocked(
                "base executor CLI is unexpectedly short"
            )
        observed = Path(command[1]).resolve(strict=False)
        if observed != BASE_EXECUTOR.resolve(strict=True):
            raise base_campaign.CampaignBlocked(
                "base campaign executor path drifted; refusing ambiguous "
                f"timing substitution: {observed}"
            )
        if not TIMED_EXECUTOR.is_file():
            raise base_campaign.CampaignBlocked(
                f"formal timed executor is missing: {TIMED_EXECUTOR}"
            )
        if isinstance(task, dict):
            task["formal_timing_gate_sha256"] = gate["gate_sha256"]
            task["formal_timing_source_set_sha256"] = gate[
                "source_set_sha256"
            ]
            task["formal_timed_executor_sha256"] = gate[
                "timed_executor_sha256"
            ]
        result = list(command)
        result[1] = str(TIMED_EXECUTOR)
        return result

    def _revoke_timing_dependency(self, task: Mapping[str, Any]) -> None:
        spec = task["spec"]
        if spec.get("kind") == "realq_static":
            key = f"{spec.get('target_phase')}/{spec.get('model')}"
            entry = self.state.get("static_caches", {}).get(key)
            if (
                isinstance(entry, dict)
                and entry.get("producer_task") == task.get("task_id")
            ):
                entry["status"] = "NEEDS_USER_ACTION"
                entry["validation"] = None
                entry["reason"] = "formal timing evidence became invalid"
        elif spec.get("kind") == "guided_saliency":
            model_state = (
                self.state.get("formal", {})
                .get("models", {})
                .get(spec.get("model"))
            )
            if isinstance(model_state, dict):
                model_state["guided_status"] = "NEEDS_USER_ACTION"

    def _invalidate_timing_task(
        self,
        task: dict[str, Any],
        error: BaseException | str,
    ) -> None:
        message = (
            error
            if isinstance(error, str)
            else f"{type(error).__name__}: {error}"
        )
        task["status"] = "INVALID_RESULT"
        task["failure_class"] = "formal_timing_invalid"
        task["formal_timing_error"] = message
        task["gpus"] = []
        self._revoke_timing_dependency(task)
        self.anomaly(
            "formal_timing_invalid",
            details={
                "task_id": task.get("task_id"),
                "manifest": task.get("manifest"),
                "error": message,
            },
            key=f"{task.get('task_id')}:{_canonical_sha256(message)}",
        )

    def _validate_task_timing(
        self,
        task: Mapping[str, Any],
        manifest: Mapping[str, Any],
        manifest_path: Path,
        *,
        primary: bool,
    ) -> dict[str, Any]:
        gate = self._assert_timing_gate()
        component = gpu_hours._parse_component(
            manifest,
            manifest_path,
            primary=primary,
        )
        phase_timing = manifest.get("phase_timing")
        if (
            not isinstance(phase_timing, Mapping)
            or phase_timing.get("source_set_sha256")
            != gate["source_set_sha256"]
        ):
            raise gpu_hours.GPUHourError(
                "manifest timing source set disagrees with frozen formal gate"
            )
        spec = task["spec"]
        if spec["kind"] == "realq_static":
            expected = (
                "shared_precompute",
                "realq",
                spec["model"],
                None,
                None,
                spec["target_phase"],
            )
        elif spec["kind"] == "guided_saliency":
            expected = (
                "shared_precompute",
                "guided_gptq",
                spec["model"],
                None,
                None,
                "final",
            )
        else:
            method = (
                "realq"
                if spec["kind"] == "realq"
                else spec["method"]
            )
            expected = (
                "quantization",
                method,
                spec["model"],
                spec.get("setting"),
                spec.get("phase"),
                None,
            )
        observed = (
            component["kind"],
            component["method"],
            component["model"],
            component["setting"],
            component["phase"],
            component["target_phase"],
        )
        if observed != expected:
            raise gpu_hours.GPUHourError(
                "timing component disagrees with scheduled formal task: "
                f"expected={expected!r}, observed={observed!r}"
            )
        if component["gpu_count"] != int(task["gpu_count"]):
            raise gpu_hours.GPUHourError(
                "timing component GPU count disagrees with scheduled task"
            )
        if primary and component.get("timing_status") != "complete":
            raise gpu_hours.GPUHourError(
                "successful formal task lacks a complete timing interval"
            )
        return component

    def _reconcile_one_task(self, task: dict[str, Any]) -> None:
        spec = task.get("spec")
        if not isinstance(spec, Mapping) or not _requires_formal_timing(spec):
            super()._reconcile_one_task(task)
            return

        manifest_path = self._task_manifest_path(task)
        if manifest_path is None or not manifest_path.is_file():
            if task["status"] in base_campaign.TERMINAL_TASK_STATES:
                self._invalidate_timing_task(
                    task, "terminal formal task has no execution manifest"
                )
                return
            super()._reconcile_one_task(task)
            return
        try:
            manifest = gpu_hours._json_no_duplicates(manifest_path)
        except (gpu_hours.GPUHourError, OSError) as exc:
            if self._task_process_alive(task):
                return
            self._invalidate_timing_task(task, exc)
            return

        task["manifest"] = str(manifest_path)
        task["log"] = str(
            manifest_path.parent / base_campaign.executor.LOG_FILENAME
        )
        task["output_dir"] = str(manifest_path.parent)
        manifest_status = manifest.get("status")
        if manifest_status in _MANIFEST_TERMINAL_STATUSES:
            # The base executor writes its terminal manifest before the outer
            # timing adapter snapshots/pins evidence.  Never release a
            # dependency while that outer process is still alive.
            if self._task_process_alive(task):
                if task["status"] in base_campaign.TERMINAL_TASK_STATES:
                    self._revoke_timing_dependency(task)
                task["status"] = "RUNNING"
                task.pop("exit_code", None)
                task.pop("finished_utc", None)
                return

            primary = (
                manifest_status == "succeeded"
                and manifest.get("exit_code") == 0
            )
            try:
                component = self._validate_task_timing(
                    task,
                    manifest,
                    manifest_path,
                    primary=primary,
                )
            except (
                base_campaign.CampaignBlocked,
                gpu_hours.GPUHourError,
                OSError,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                self._invalidate_timing_task(task, exc)
                return
            task["formal_timing"] = {
                "component_id": component["component_id"],
                "timing_status": component["timing_status"],
                "wall_seconds": component["wall_seconds"],
                "gpu_hour_status": component["gpu_hour_status"],
                "allocated_gpu_hours": component[
                    "allocated_gpu_hours"
                ],
                "gpu_hours_lower_bound": component[
                    "gpu_hours_lower_bound"
                ],
                "missing_ranks": component["missing_ranks"],
                "sidecar": component["sidecar"],
                "sidecar_sha256": component["sidecar_sha256"],
            }

            if task["status"] in base_campaign.TERMINAL_TASK_STATES:
                # Terminal tasks are deliberately reparsed every tick.  This
                # branch preserves their base outcome while still detecting
                # later sidecar/rank/source tampering.
                if primary and task["status"] not in {
                    "SUCCEEDED",
                    "SUPERSEDED",
                }:
                    self._invalidate_timing_task(
                        task,
                        "successful manifest disagrees with terminal task "
                        f"state {task['status']!r}",
                    )
                return
            super()._reconcile_one_task(task)
            return

        if task["status"] in base_campaign.TERMINAL_TASK_STATES:
            self._invalidate_timing_task(
                task,
                "terminal formal task manifest regressed to a non-terminal "
                f"status {manifest_status!r}",
            )
            return
        super()._reconcile_one_task(task)


def main(argv: list[str] | None = None) -> int:
    original = base_campaign.Campaign
    base_campaign.Campaign = FormalTimedCampaign
    try:
        return base_campaign.main(argv)
    finally:
        base_campaign.Campaign = original


if __name__ == "__main__":
    raise SystemExit(main())
