#!/usr/bin/env python3
"""Fan out each YAQA Qwen3-32B suite across four physical GPUs.

The canonical evaluator runs quality, GSM8K, MATH-500, and HumanEval+
serially after loading one checkpoint.  These four outputs are independent
and deterministic.  This scheduling helper reserves only the four unfinished
YAQA-32B suites, calls the same frozen loader/evaluation functions on one GPU
per component, validates every terminal, and publishes the original
``suite_success.json`` schema only after all four components are complete.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback
from types import SimpleNamespace
from typing import Any, Mapping

from experiments.additional_methods_fair20_eval_20260821 import (
    common,
    loaders,
    run_suite,
    wait_recovery_then_eval,
)


HOST0 = "j-4mj21jb084-master-0"
HOST1 = "j-zogxxxduju-master-0"
AUDIT_ROOT = common.OUTPUT_ROOT / "yaqa_q32_evaluation_fanout_20260822"
PLAN_PATH = AUDIT_ROOT / "plan.json"
LOCK_ROOT = common.DATA_ROOT / "_fair20_physical_gpu_locks_20260821"
EXPECTED_REFERENCE_FINGERPRINT = (
    "3ecffa5d4cb97efb24aac4a668b6570d8e8193f4f4fa21f98825c2d9d238be05"
)
COMPONENTS = ("quality", "gsm8k", "math_500", "humaneval_plus")
MAX_COMPONENT_ATTEMPTS = 3

# W4A4/W4/W3 can finish within one layer-hour of each other and therefore use
# disjoint 4-GPU groups.  W2 is ~6 hours later, keeps quality on its quantizer
# lane, and safely reuses W4A4's three node1 reasoning lanes.
SCHEDULE: dict[str, dict[str, dict[str, Any]]] = {
    "yaqa_wclip__YQ-Q32-W4A4": {
        "quality": {"hostname": HOST0, "physical_gpu": 0, "delay_seconds": 0},
        "gsm8k": {"hostname": HOST1, "physical_gpu": 0, "delay_seconds": 30},
        "math_500": {"hostname": HOST1, "physical_gpu": 2, "delay_seconds": 60},
        "humaneval_plus": {
            "hostname": HOST1,
            "physical_gpu": 4,
            "delay_seconds": 90,
        },
    },
    "yaqa_wclip__YQ-Q32-W4": {
        "quality": {"hostname": HOST0, "physical_gpu": 1, "delay_seconds": 0},
        "gsm8k": {"hostname": HOST1, "physical_gpu": 3, "delay_seconds": 30},
        "math_500": {"hostname": HOST1, "physical_gpu": 5, "delay_seconds": 60},
        "humaneval_plus": {
            "hostname": HOST1,
            "physical_gpu": 6,
            "delay_seconds": 90,
        },
    },
    "yaqa_wclip__YQ-Q32-W3": {
        "quality": {"hostname": HOST1, "physical_gpu": 1, "delay_seconds": 0},
        "gsm8k": {"hostname": HOST0, "physical_gpu": 4, "delay_seconds": 30},
        "math_500": {"hostname": HOST0, "physical_gpu": 7, "delay_seconds": 60},
        "humaneval_plus": {
            "hostname": HOST1,
            "physical_gpu": 7,
            "delay_seconds": 90,
        },
    },
    "yaqa_wclip__YQ-Q32-W2": {
        "quality": {"hostname": HOST0, "physical_gpu": 2, "delay_seconds": 0},
        "gsm8k": {"hostname": HOST1, "physical_gpu": 0, "delay_seconds": 30},
        "math_500": {"hostname": HOST1, "physical_gpu": 2, "delay_seconds": 60},
        "humaneval_plus": {
            "hostname": HOST1,
            "physical_gpu": 4,
            "delay_seconds": 90,
        },
    },
}
SOURCE_FILES = (
    "experiments/yaqa_q32_eval_fanout_20260822/fanout.py",
    "experiments/additional_methods_fair20_eval_20260821/common.py",
    "experiments/additional_methods_fair20_eval_20260821/loaders.py",
    "experiments/additional_methods_fair20_eval_20260821/run_suite.py",
    "experiments/additional_methods_fair20_eval_20260821/wait_recovery_then_eval.py",
    "realq_benchmark/benchmarks/data.py",
    "realq_benchmark/benchmarks/generation.py",
    "realq_benchmark/benchmarks/runner.py",
    "realq_benchmark/benchmarks/schema.py",
    "realq_benchmark/benchmarks/scoring.py",
)


class FanoutError(RuntimeError):
    """A fan-out reservation, component, or assembly gate failed."""


def _read(path: str | Path) -> dict[str, Any]:
    value = common.read_object(path)
    if not isinstance(value, dict):
        raise FanoutError(f"JSON root is not an object: {path}")
    return value


def _spec(eval_id: str) -> common.EvalSpec:
    if eval_id not in SCHEDULE:
        raise FanoutError(f"fan-out is not allowed for {eval_id}")
    spec = run_suite._find_spec(eval_id)
    if (
        spec.method != "yaqa_wclip"
        or spec.model != "qwen3-32b"
        or spec.run_id != eval_id.split("__", 1)[1]
    ):
        raise FanoutError(f"unexpected fan-out spec: {spec}")
    return spec


def _source_snapshot() -> dict[str, Any]:
    files = [
        {
            "path": relative,
            "sha256": common.sha256_file(common.REPO_ROOT / relative),
            "size_bytes": (common.REPO_ROOT / relative).stat().st_size,
        }
        for relative in SOURCE_FILES
    ]
    return {"files": files, "sha256": common.canonical_sha256(files)}


def _build_plan() -> dict[str, Any]:
    reference = common.load_reference_manifest()
    if reference.get("fingerprint") != EXPECTED_REFERENCE_FINGERPRINT:
        raise FanoutError("reference manifest fingerprint changed")
    specs = {eval_id: _spec(eval_id) for eval_id in SCHEDULE}
    if {spec.setting for spec in specs.values()} != {
        "W4A4KV4",
        "W4A16KV16",
        "W3A16KV16",
        "W2A16KV16",
    }:
        raise FanoutError("fan-out does not cover the four Q32 settings")
    body = {
        "schema_version": 1,
        "experiment_id": "yaqa-q32-evaluation-fanout-20260822-v1",
        "reference_manifest": {
            "path": str(common.REFERENCE_MANIFEST_PATH),
            "sha256": common.sha256_file(common.REFERENCE_MANIFEST_PATH),
            "fingerprint": EXPECTED_REFERENCE_FINGERPRINT,
        },
        "yaqa_plan": {
            "path": str(common.YAQA_PLAN),
            "sha256": common.EXPECTED_PLAN_SHA256["yaqa_wclip"],
        },
        "source_snapshot": _source_snapshot(),
        "schedule": SCHEDULE,
        "specs": {
            eval_id: {
                "method": spec.method,
                "run_id": spec.run_id,
                "model": spec.model,
                "setting": spec.setting,
                "w_bits": spec.w_bits,
                "a_bits": spec.a_bits,
                "k_bits": spec.k_bits,
                "v_bits": spec.v_bits,
            }
            for eval_id, spec in specs.items()
        },
        "protocol": {
            "numerical_contract_changed": False,
            "quality_implementation": "frozen run_suite._run_quality",
            "reasoning_implementation": "frozen run_reasoning_eval",
            "reasoning_seed": 1234,
            "reasoning_protocol": "realq_zero_shot_v1",
            "components": list(COMPONENTS),
            "publication_schema": "frozen suite_success schema_version=1",
            "official_humaneval_scorer": "unchanged shared score_worker",
        },
    }
    body["plan_fingerprint"] = common.canonical_sha256(body)
    return body


def init_plan() -> dict[str, Any]:
    plan = _build_plan()
    if PLAN_PATH.is_file():
        if _read(PLAN_PATH) != plan:
            raise FanoutError("existing fan-out plan drifted")
        return plan
    if AUDIT_ROOT.exists() or AUDIT_ROOT.is_symlink():
        raise FanoutError(f"fan-out audit root is not fresh: {AUDIT_ROOT}")
    AUDIT_ROOT.mkdir(parents=True)
    common.atomic_json(PLAN_PATH, plan)
    return plan


def _verify_plan() -> dict[str, Any]:
    plan = _read(PLAN_PATH)
    stable = dict(plan)
    fingerprint = stable.pop("plan_fingerprint", None)
    if common.canonical_sha256(stable) != fingerprint:
        raise FanoutError("fan-out plan fingerprint mismatch")
    if plan != _build_plan():
        raise FanoutError("fan-out plan source or input drifted")
    return plan


def _suite_dir(eval_id: str) -> Path:
    return common.OUTPUT_ROOT / "evals" / eval_id


def _reservation_owner(eval_id: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "yaqa_q32_evaluation_fanout_reservation",
        "eval_id": eval_id,
        "plan": str(PLAN_PATH),
        "plan_fingerprint": plan["plan_fingerprint"],
        "reserved_at": common.now(),
    }


def _validate_reservation(eval_id: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    owner_path = _suite_dir(eval_id) / ".claim" / "owner.json"
    owner = _read(owner_path)
    if (
        owner.get("kind") != "yaqa_q32_evaluation_fanout_reservation"
        or owner.get("eval_id") != eval_id
        or owner.get("plan_fingerprint") != plan["plan_fingerprint"]
    ):
        raise FanoutError(f"suite reservation changed: {eval_id}")
    return owner


def reserve() -> dict[str, Any]:
    plan = _verify_plan()
    preflight = []
    for eval_id in SCHEDULE:
        directory = _suite_dir(eval_id)
        forbidden = (
            directory / "suite_success.json",
            directory / "quality_result.json",
            directory / ".claim",
            directory / "attempts",
            directory / "reasoning",
        )
        if any(path.exists() or path.is_symlink() for path in forbidden):
            raise FanoutError(f"suite is not fresh for reservation: {eval_id}")
        preflight.append(eval_id)

    created: list[tuple[Path, Path]] = []
    try:
        for eval_id in preflight:
            directory = _suite_dir(eval_id)
            directory.mkdir(parents=True, exist_ok=True)
            claim = directory / ".claim"
            claim.mkdir()
            common.atomic_json(
                claim / "owner.json", _reservation_owner(eval_id, plan)
            )
            created.append((directory, claim))
    except BaseException:
        for directory, claim in reversed(created):
            (claim / "owner.json").unlink(missing_ok=True)
            claim.rmdir()
            try:
                directory.rmdir()
            except OSError:
                pass
        raise
    receipt = {
        "schema_version": 1,
        "status": "reserved",
        "kind": "yaqa_q32_evaluation_fanout_reservation",
        "plan_fingerprint": plan["plan_fingerprint"],
        "eval_ids": preflight,
        "reserved_at": common.now(),
    }
    common.atomic_json(AUDIT_ROOT / "reservation_receipt.json", receipt)
    return receipt


def _component_dir(eval_id: str, component: str) -> Path:
    return AUDIT_ROOT / "components" / eval_id / component


def _component_receipt_path(eval_id: str, component: str) -> Path:
    return _component_dir(eval_id, component) / "receipt.json"


def _quant_failure(spec: common.EvalSpec) -> Path:
    return spec.quant_dir / "failure.json"


def _wait_artifact(spec: common.EvalSpec, poll_seconds: int) -> dict[str, Any]:
    last_report = 0.0
    while True:
        if _quant_failure(spec).is_file():
            raise FanoutError(f"quantization failed: {_quant_failure(spec)}")
        try:
            return common.resolve_completed_artifact(spec)
        except common.EvaluationError:
            current = time.monotonic()
            if current - last_report >= 300:
                print(
                    f"{common.now()} waiting_for_quantized_artifact={spec.eval_id}",
                    flush=True,
                )
                last_report = current
            time.sleep(poll_seconds)


def _gpu_compute_pids(gpu: int) -> list[int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise FanoutError(completed.stderr.strip())
    return [
        int(line.strip())
        for line in completed.stdout.splitlines()
        if line.strip().isdigit()
    ]


def _component_runtime_inputs(
    eval_id: str,
) -> tuple[
    common.EvalSpec,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    spec = _spec(eval_id)
    plans = common.load_quant_plans()
    reference_manifest = common.load_reference_manifest()
    if reference_manifest["fingerprint"] != EXPECTED_REFERENCE_FINGERPRINT:
        raise FanoutError("reference manifest changed at component boundary")
    reference_record = dict(reference_manifest["references"][spec.model])
    model_info = common.model_contract(spec.model, plans)
    artifact = common.resolve_completed_artifact(spec)
    return spec, plans, reference_manifest, reference_record, model_info, artifact


def _assert_runtime_assignment(
    *,
    runtime: Mapping[str, Any],
    eval_id: str,
    component: str,
    physical_gpu: int,
    plan: Mapping[str, Any],
) -> None:
    expected = plan["schedule"][eval_id][component]
    actual = {
        "hostname": runtime.get("hostname"),
        "physical_gpu": physical_gpu,
        "cuda_visible_devices": runtime.get("cuda_visible_devices"),
    }
    wanted = {
        "hostname": expected["hostname"],
        "physical_gpu": int(expected["physical_gpu"]),
        "cuda_visible_devices": str(expected["physical_gpu"]),
    }
    if actual != wanted:
        raise FanoutError(
            f"component runtime assignment changed: {actual!r} != {wanted!r}"
        )


def _run_component(
    eval_id: str,
    component: str,
    *,
    physical_gpu: int,
    attempt_index: int,
) -> dict[str, Any]:
    import torch

    plan = _verify_plan()
    _validate_reservation(eval_id, plan)
    if component not in COMPONENTS:
        raise FanoutError(f"unknown component: {component}")
    directory = _component_dir(eval_id, component)
    if directory.is_symlink():
        raise FanoutError(f"component audit directory is a symlink: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "receipt.json").exists():
        raise FanoutError(f"component receipt already exists: {directory}")
    if attempt_index < 1 or attempt_index > MAX_COMPONENT_ATTEMPTS:
        raise FanoutError(f"invalid component attempt index: {attempt_index}")
    attempt_manifest_path = (
        directory / "attempts" / f"attempt{attempt_index:03d}" / "manifest.json"
    )
    attempt_manifest = _read(attempt_manifest_path)
    expected_attempt = {
        "status": "running",
        "kind": "yaqa_q32_evaluation_component_attempt",
        "eval_id": eval_id,
        "component": component,
        "attempt": attempt_index,
        "hostname": socket.gethostname(),
        "physical_gpu": physical_gpu,
        "plan_fingerprint": plan["plan_fingerprint"],
    }
    if any(
        attempt_manifest.get(key) != value
        for key, value in expected_attempt.items()
    ):
        raise FanoutError("component attempt manifest changed")
    started_at = common.now()
    started = time.monotonic()
    runtime = run_suite._configure_runtime(seed=1234)
    _assert_runtime_assignment(
        runtime=runtime,
        eval_id=eval_id,
        component=component,
        physical_gpu=physical_gpu,
        plan=plan,
    )
    (
        spec,
        _plans,
        reference_manifest,
        reference_record,
        model_info,
        artifact,
    ) = _component_runtime_inputs(eval_id)
    output_dir = _suite_dir(eval_id)
    if component == "quality":
        analyzer, quality = run_suite._run_quality(
            spec,
            artifact,
            model_info,
            output_dir,
            reference_record=reference_record,
        )
        output_path = output_dir / "quality_result.json"
        output = run_suite._validate_quality_result(
            output_path,
            spec,
            reference_record["sha256"],
        )
        del analyzer
        output_summary = {"metrics": quality["metrics"]}
    else:
        task = component
        task_dir = output_dir / "reasoning" / task
        manifest_path = task_dir / "manifest.json"
        completed = False
        if manifest_path.is_file():
            partial = run_suite._validate_reasoning_contract(
                manifest_path,
                task,
                require_completed=False,
            )
            completed = partial.get("status") == "completed"
        elif manifest_path.exists() or manifest_path.is_symlink():
            raise FanoutError(f"invalid reasoning manifest path: {manifest_path}")
        if completed:
            output = run_suite._validate_reasoning_manifest(manifest_path, task)
        else:
            task_dir.mkdir(parents=True, exist_ok=True)
            analyzer = loaders.load_quantized(spec, artifact, model_info)
            analyzer.model.to(torch.device("cuda"))
            from realq_benchmark.benchmarks.runner import run_reasoning_eval

            cfg = run_suite._reasoning_config(spec, artifact, task, task_dir)
            manifest = run_reasoning_eval(analyzer.model, analyzer.tokenizer, cfg)
            if not manifest_path.is_file():
                raise FanoutError(f"reasoning manifest was not published: {task}")
            output = run_suite._validate_reasoning_manifest(manifest_path, task)
            del analyzer, manifest
        output_path = manifest_path
        output_summary = {"result": output["results"][0]}
    gc.collect()
    torch.cuda.empty_cache()
    receipt = {
        "schema_version": 1,
        "status": "succeeded",
        "kind": "yaqa_q32_evaluation_component_fanout",
        "numerical_contract_changed": False,
        "eval_id": eval_id,
        "component": component,
        "attempt": attempt_index,
        "hostname": socket.gethostname(),
        "physical_gpu": physical_gpu,
        "started_at": started_at,
        "finished_at": common.now(),
        "wall_seconds": time.monotonic() - started,
        "plan": str(PLAN_PATH),
        "plan_fingerprint": plan["plan_fingerprint"],
        "runtime": runtime,
        "reference_manifest_fingerprint": reference_manifest["fingerprint"],
        "artifact": artifact,
        "output": str(output_path),
        "output_sha256": common.sha256_file(output_path),
        "output_summary": output_summary,
    }
    common.atomic_json(directory / "receipt.json", receipt)
    return receipt


def _load_component_receipt(
    eval_id: str, component: str, plan: Mapping[str, Any]
) -> dict[str, Any]:
    path = _component_receipt_path(eval_id, component)
    receipt = _read(path)
    output = Path(str(receipt.get("output", "")))
    expected = plan["schedule"][eval_id][component]
    if (
        receipt.get("status") != "succeeded"
        or receipt.get("kind") != "yaqa_q32_evaluation_component_fanout"
        or receipt.get("numerical_contract_changed") is not False
        or receipt.get("eval_id") != eval_id
        or receipt.get("component") != component
        or receipt.get("attempt") not in range(1, MAX_COMPONENT_ATTEMPTS + 1)
        or receipt.get("hostname") != expected["hostname"]
        or receipt.get("physical_gpu") != int(expected["physical_gpu"])
        or receipt.get("plan_fingerprint") != plan["plan_fingerprint"]
        or receipt.get("reference_manifest_fingerprint")
        != EXPECTED_REFERENCE_FINGERPRINT
        or not output.is_file()
        or common.sha256_file(output) != receipt.get("output_sha256")
    ):
        raise FanoutError(f"component receipt gate failed: {eval_id}/{component}")
    _assert_runtime_assignment(
        runtime=receipt.get("runtime", {}),
        eval_id=eval_id,
        component=component,
        physical_gpu=int(expected["physical_gpu"]),
        plan=plan,
    )
    return receipt


def _assemble_payload(
    *,
    spec: common.EvalSpec,
    artifact: dict[str, Any],
    reference_manifest: Mapping[str, Any],
    quality_receipt: Mapping[str, Any],
    quality: Mapping[str, Any],
    reasoning: Mapping[str, Any],
    fanout_record: Mapping[str, Any],
) -> dict[str, Any]:
    quality_path = _suite_dir(spec.eval_id) / "quality_result.json"
    return {
        "schema_version": 1,
        "status": "generation_succeeded_official_humaneval_pending",
        "eval_id": spec.eval_id,
        "method": spec.method,
        "run_id": spec.run_id,
        "model": spec.model,
        "setting": spec.setting,
        "runtime": quality_receipt["runtime"],
        "reference_manifest_fingerprint": reference_manifest["fingerprint"],
        "artifact": artifact,
        "quality": {
            "path": str(quality_path),
            "sha256": common.sha256_file(quality_path),
            "metrics": quality["metrics"],
        },
        "reasoning": dict(reasoning),
        "official_humaneval_plus": "pending",
        "evaluation_fanout": dict(fanout_record),
        "finished_at": common.now(),
    }


def _release_reservation(eval_id: str, plan: Mapping[str, Any]) -> None:
    claim = _suite_dir(eval_id) / ".claim"
    if not claim.exists():
        return
    _validate_reservation(eval_id, plan)
    (claim / "owner.json").unlink()
    claim.rmdir()


def _publish_assembly_receipt(eval_id: str, success_path: Path) -> dict[str, Any]:
    assembly = {
        "schema_version": 1,
        "status": "succeeded",
        "kind": "yaqa_q32_four_component_evaluation_fanout",
        "eval_id": eval_id,
        "suite_success": str(success_path),
        "suite_success_sha256": common.sha256_file(success_path),
        "finished_at": common.now(),
    }
    path = AUDIT_ROOT / "assemblies" / f"{eval_id}.json"
    if path.is_file():
        existing = _read(path)
        stable_keys = (
            "schema_version",
            "status",
            "kind",
            "eval_id",
            "suite_success",
            "suite_success_sha256",
        )
        if any(existing.get(key) != assembly[key] for key in stable_keys):
            raise FanoutError(f"assembly receipt changed: {eval_id}")
        return existing
    common.atomic_json(path, assembly)
    return assembly


def try_assemble(eval_id: str) -> dict[str, Any]:
    plan = _verify_plan()
    spec = _spec(eval_id)
    success_path = _suite_dir(eval_id) / "suite_success.json"
    lock_path = AUDIT_ROOT / "assembly_locks" / f"{eval_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if success_path.is_file():
            success = _read(success_path)
            if success.get("evaluation_fanout", {}).get("plan_fingerprint") != plan[
                "plan_fingerprint"
            ]:
                raise FanoutError("suite exists without this fan-out contract")
            _release_reservation(eval_id, plan)
            _publish_assembly_receipt(eval_id, success_path)
            return success
        missing = [
            component
            for component in COMPONENTS
            if not _component_receipt_path(eval_id, component).is_file()
        ]
        if missing:
            return {"status": "waiting", "eval_id": eval_id, "missing": missing}
        reservation = _validate_reservation(eval_id, plan)
        receipts = {
            component: _load_component_receipt(eval_id, component, plan)
            for component in COMPONENTS
        }
        reference_manifest = common.load_reference_manifest()
        artifact = common.resolve_completed_artifact(spec)
        for component, receipt in receipts.items():
            declared = receipt["artifact"]
            if declared != artifact:
                raise FanoutError(f"artifact changed across component: {component}")
        reference_sha = reference_manifest["references"][spec.model]["sha256"]
        quality_path = _suite_dir(eval_id) / "quality_result.json"
        quality = run_suite._validate_quality_result(
            quality_path, spec, reference_sha
        )
        reasoning = {}
        for task in common.REASONING_TASKS:
            manifest_path = _suite_dir(eval_id) / "reasoning" / task / "manifest.json"
            manifest = run_suite._validate_reasoning_manifest(manifest_path, task)
            reasoning[task] = {
                "manifest": str(manifest_path),
                "manifest_sha256": common.sha256_file(manifest_path),
                "result": manifest["results"][0],
            }
        fanout_record = {
            "kind": "yaqa_q32_four_component_evaluation_fanout",
            "scheduling_only": True,
            "numerical_contract_changed": False,
            "plan": str(PLAN_PATH),
            "plan_sha256": common.sha256_file(PLAN_PATH),
            "plan_fingerprint": plan["plan_fingerprint"],
            "reservation": reservation,
            "components": {
                component: {
                    "receipt": str(_component_receipt_path(eval_id, component)),
                    "receipt_sha256": common.sha256_file(
                        _component_receipt_path(eval_id, component)
                    ),
                    "hostname": receipt["runtime"]["hostname"],
                    "physical_gpu": plan["schedule"][eval_id][component][
                        "physical_gpu"
                    ],
                }
                for component, receipt in receipts.items()
            },
        }
        payload = _assemble_payload(
            spec=spec,
            artifact=artifact,
            reference_manifest=reference_manifest,
            quality_receipt=receipts["quality"],
            quality=quality,
            reasoning=reasoning,
            fanout_record=fanout_record,
        )
        common.atomic_json(success_path, payload)
        if _read(success_path) != payload:
            raise FanoutError("suite terminal did not round-trip")
        _release_reservation(eval_id, plan)
        _publish_assembly_receipt(eval_id, success_path)
        return payload


def _component_attempt_count(eval_id: str, component: str) -> int:
    attempts = _component_dir(eval_id, component) / "attempts"
    return sum(
        path.is_dir()
        for path in attempts.glob("attempt[0-9][0-9][0-9]")
    )


def _run_component_attempt(
    *,
    eval_id: str,
    component: str,
    physical_gpu: int,
    python: str,
    plan: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    index = _component_attempt_count(eval_id, component) + 1
    if index > MAX_COMPONENT_ATTEMPTS:
        raise FanoutError(
            f"component retry budget exhausted: {eval_id}/{component}"
        )
    attempt = (
        _component_dir(eval_id, component)
        / "attempts"
        / f"attempt{index:03d}"
    )
    attempt.mkdir(parents=True, exist_ok=False)
    command = [
        python,
        "-u",
        "-m",
        "experiments.yaqa_q32_eval_fanout_20260822.fanout",
        "--component-attempt-child",
        "--eval-id",
        eval_id,
        "--component",
        component,
        "--physical-gpu",
        str(physical_gpu),
        "--attempt-index",
        str(index),
    ]
    manifest = {
        "schema_version": 1,
        "status": "running",
        "kind": "yaqa_q32_evaluation_component_attempt",
        "eval_id": eval_id,
        "component": component,
        "attempt": index,
        "hostname": socket.gethostname(),
        "physical_gpu": physical_gpu,
        "plan_fingerprint": plan["plan_fingerprint"],
        "command": command,
        "started_at": common.now(),
    }
    common.atomic_json(attempt / "manifest.json", manifest)
    started = time.monotonic()
    with (attempt / "execution.log").open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=common.REPO_ROOT,
            env=_worker_environment(physical_gpu),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    receipt_valid = False
    receipt_error = None
    if _component_receipt_path(eval_id, component).is_file():
        try:
            _load_component_receipt(eval_id, component, plan)
            receipt_valid = True
        except BaseException as exc:  # recorded before failing closed below
            receipt_error = f"{type(exc).__name__}: {exc}"
    # The receipt is the durable terminal.  A nonzero return after its atomic
    # publication (for example, a broken stdout pipe while printing it) must
    # not trigger a second expensive evaluation over already-final outputs.
    succeeded = receipt_valid
    result = {
        **manifest,
        "status": "succeeded" if succeeded else "failed",
        "returncode": completed.returncode,
        "receipt_valid": receipt_valid,
        "receipt_error": receipt_error,
        "wall_seconds": time.monotonic() - started,
        "finished_at": common.now(),
    }
    common.atomic_json(attempt / "result.json", result)
    return succeeded, result


def execute_component(
    *, eval_id: str, component: str, physical_gpu: int, poll_seconds: int
) -> dict[str, Any]:
    if socket.gethostname() not in {HOST0, HOST1}:
        raise FanoutError("component supervisor must run on a Canoe pod")
    plan = _verify_plan()
    expected = plan["schedule"][eval_id][component]
    if (
        expected["hostname"] != socket.gethostname()
        or int(expected["physical_gpu"]) != physical_gpu
        or poll_seconds < 10
    ):
        raise FanoutError("component host/GPU/poll contract changed")
    _validate_reservation(eval_id, plan)
    component_dir = _component_dir(eval_id, component)
    component_dir.mkdir(parents=True, exist_ok=True)
    supervisor_lock = component_dir / ".supervisor.lock"
    with supervisor_lock.open("a+", encoding="utf-8") as supervisor:
        try:
            fcntl.flock(
                supervisor.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise FanoutError("component supervisor is already live") from exc
        spec = _spec(eval_id)
        _wait_artifact(spec, poll_seconds)
        delay = int(expected["delay_seconds"])
        if delay:
            time.sleep(delay)
        lock_path = LOCK_ROOT / socket.gethostname() / f"gpu{physical_gpu}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"{common.now()} waiting_for_gpu_lock={lock_path}", flush=True)
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            pids = _gpu_compute_pids(physical_gpu)
            if pids:
                raise FanoutError(f"untracked GPU processes: {pids}")
            print(
                f"{common.now()} component_start={eval_id}/{component} "
                f"gpu={physical_gpu}",
                flush=True,
            )
            python = str(common.load_quant_plans()["efficientqat"]["venv_python"])
            attempt_results = []
            succeeded = False
            while _component_attempt_count(eval_id, component) < MAX_COMPONENT_ATTEMPTS:
                succeeded, attempt_result = _run_component_attempt(
                    eval_id=eval_id,
                    component=component,
                    physical_gpu=physical_gpu,
                    python=python,
                    plan=plan,
                )
                attempt_results.append(attempt_result)
                if succeeded:
                    break
                print(
                    f"{common.now()} component_attempt_failed={eval_id}/{component} "
                    f"attempt={attempt_result['attempt']}",
                    flush=True,
                )
            if not succeeded:
                failure = {
                    "schema_version": 1,
                    "status": "failed",
                    "kind": "yaqa_q32_evaluation_component_failure",
                    "eval_id": eval_id,
                    "component": component,
                    "hostname": socket.gethostname(),
                    "physical_gpu": physical_gpu,
                    "plan_fingerprint": plan["plan_fingerprint"],
                    "attempts": attempt_results,
                    "failed_at": common.now(),
                }
                common.atomic_json(component_dir / "failure.json", failure)
                raise FanoutError(
                    f"component failed after {MAX_COMPONENT_ATTEMPTS} attempts: "
                    f"{eval_id}/{component}"
                )
            receipt = _load_component_receipt(eval_id, component, plan)
    assembled = try_assemble(eval_id)
    return {"component": receipt, "assembly": assembled}


def _worker_environment(physical_gpu: int) -> dict[str, str]:
    environment = wait_recovery_then_eval._worker_environment()
    for key in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        environment.pop(key, None)
    environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    return environment


def launch_supervisors() -> dict[str, Any]:
    hostname = socket.gethostname()
    if hostname not in {HOST0, HOST1}:
        raise FanoutError("fan-out launch must run on a Canoe pod")
    plan = _verify_plan()
    reservation = _read(AUDIT_ROOT / "reservation_receipt.json")
    if (
        reservation.get("status") != "reserved"
        or reservation.get("plan_fingerprint") != plan["plan_fingerprint"]
    ):
        raise FanoutError("fan-out reservation receipt is invalid")
    python = str(common.load_quant_plans()["efficientqat"]["venv_python"])
    host_gpus = sorted(
        {
            int(assignment["physical_gpu"])
            for components in plan["schedule"].values()
            for assignment in components.values()
            if assignment["hostname"] == hostname
        }
    )
    if not host_gpus:
        raise FanoutError(f"fan-out schedule has no work on {hostname}")
    wait_recovery_then_eval._preflight(python, _worker_environment(host_gpus[0]))
    launches = []
    for eval_id, components in plan["schedule"].items():
        for component, assignment in components.items():
            if assignment["hostname"] != hostname:
                continue
            directory = AUDIT_ROOT / "launches" / hostname
            directory.mkdir(parents=True, exist_ok=True)
            stem = f"{eval_id}__{component}"
            log_path = directory / f"{stem}.log"
            receipt_path = directory / f"{stem}.json"
            if log_path.exists() or receipt_path.exists():
                raise FanoutError(f"component was already launched: {stem}")
            gpu = int(assignment["physical_gpu"])
            command = [
                python,
                "-u",
                "-m",
                "experiments.yaqa_q32_eval_fanout_20260822.fanout",
                "--component-child",
                "--eval-id",
                eval_id,
                "--component",
                component,
                "--physical-gpu",
                str(gpu),
                "--poll-seconds",
                "30",
            ]
            with log_path.open("x", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=common.REPO_ROOT,
                    env=_worker_environment(gpu),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            launch = {
                "schema_version": 1,
                "status": "launched_waiting_for_artifact",
                "eval_id": eval_id,
                "component": component,
                "hostname": hostname,
                "physical_gpu": gpu,
                "pid": process.pid,
                "plan_fingerprint": plan["plan_fingerprint"],
                "command": command,
                "log": str(log_path),
                "launched_at": common.now(),
            }
            common.atomic_json(receipt_path, launch)
            launches.append(launch)
    return {"status": "launched", "hostname": hostname, "launches": launches}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-plan", action="store_true")
    parser.add_argument("--reserve", action="store_true")
    parser.add_argument("--launch-supervisors", action="store_true")
    parser.add_argument("--component-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--component-attempt-child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--eval-id", choices=tuple(SCHEDULE))
    parser.add_argument("--component", choices=COMPONENTS)
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--attempt-index", type=int)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    try:
        modes = sum(
            (
                args.init_plan,
                args.reserve,
                args.launch_supervisors,
                args.component_child,
                args.component_attempt_child,
            )
        )
        if modes != 1:
            parser.error("select exactly one fan-out mode")
        if args.init_plan:
            result = init_plan()
        elif args.reserve:
            result = reserve()
        elif args.launch_supervisors:
            result = launch_supervisors()
        elif args.component_child:
            if (
                args.eval_id is None
                or args.component is None
                or args.physical_gpu is None
            ):
                parser.error("component child arguments are incomplete")
            result = execute_component(
                eval_id=args.eval_id,
                component=args.component,
                physical_gpu=args.physical_gpu,
                poll_seconds=args.poll_seconds,
            )
        else:
            if (
                args.eval_id is None
                or args.component is None
                or args.physical_gpu is None
                or args.attempt_index is None
            ):
                parser.error("component attempt child arguments are incomplete")
            result = _run_component(
                args.eval_id,
                args.component,
                physical_gpu=args.physical_gpu,
                attempt_index=args.attempt_index,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
