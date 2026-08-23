#!/usr/bin/env python3
"""Run and compare paired fixed-label signed REAL-Q backend traces."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.realq_allopts_loss_ablation_20260821 import (
    cache_backend_diagnostic as source,
)
from experiments.realq_fixed_label_backend_diagnostic_20260822 import (
    runner as fixed_runner,
)
from experiments.realq_fullmodel_retune_20260817 import campaign as base


DATA_ROOT = REPO_ROOT.parent / "experiment_data"
OUTPUT_ROOT = DATA_ROOT / "realq_fixed_label_gradient_trace_20260822_v1"
PLAN_PATH = OUTPUT_ROOT / "plan.json"
LOCK_ROOT = DATA_ROOT / "_fair20_physical_gpu_locks_20260821"
HOST = fixed_runner.HOST
ARMS = {
    "trace_sdpa": {"backend": "sdpa", "physical_gpu": 0},
    "trace_fa4": {"backend": "flash_attention_4", "physical_gpu": 2},
}
SOURCE_RECEIPT_SHA256 = fixed_runner.SOURCE_RECEIPT_SHA256
SOURCE_FILES = (
    "experiments/realq_fixed_label_gradient_trace_20260822/trace_driver.py",
    "experiments/realq_fixed_label_gradient_trace_20260822/runner.py",
    "experiments/realq_fixed_label_backend_diagnostic_20260822/fixed_label_driver.py",
    "experiments/realq_allopts_loss_ablation_20260821/cache_backend_diagnostic.py",
    "realq/ptq.py",
    "realq/pipeline.py",
    "realq/attention.py",
    "realq/precompute/__init__.py",
    "realq/precompute/static_e2e.py",
    "realq/precompute/labels.py",
    "realq/precompute/hooks.py",
    "utils/reproducibility.py",
)


class TraceRunnerError(RuntimeError):
    """A signed-trace scheduling, provenance, or comparison gate failed."""


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TraceRunnerError(f"JSON root is not an object: {path}")
    return value


def _flags(command: Sequence[str]) -> dict[str, str]:
    return source._flags(command)


def _set(command: list[str], flag: str, value: str) -> None:
    source._set(command, flag, value)


def _command(arm: str) -> list[str]:
    if arm not in ARMS:
        raise TraceRunnerError(f"unknown signed-trace arm: {arm}")
    _receipt, command = source._source_command()
    if command[1:3] != ["-m", "realq.ptq"]:
        raise TraceRunnerError("source command entry point changed")
    command[2] = (
        "experiments.realq_fixed_label_gradient_trace_20260822.trace_driver"
    )
    stage = OUTPUT_ROOT / arm
    overrides = {
        "--attention_backend": str(ARMS[arm]["backend"]),
        "--static_cache_path": "",
        "--cache_dir": str(stage / "runtime"),
        "--output_dir": str(stage / "realq_output"),
        "--exp": f"realq_fixed_label_signed_trace_{arm}",
        "--skip_eval": "true",
        "--skip_kl_ppl_eval": "true",
        "--lm_eval": "false",
        "--reasoning_eval": "false",
        "--require_static_cache_hit": "false",
    }
    for flag, value in overrides.items():
        _set(command, flag, value)
    values = _flags(command)
    required = {
        "--global_loss_bsz": "4",
        "--hessian_accum_bsz": "64",
        "--grad_hessian_topk": "-1",
        "--nsamples": "256",
        "--seq_len": "2048",
        "--rotate": "true",
        "--rotation_seed": "0",
        "--seed": "1",
        "--exit_after_precompute": "true",
        "--attention_backend": str(ARMS[arm]["backend"]),
        "--static_cache_path": "",
    }
    mismatches = {
        flag: (values.get(flag), expected)
        for flag, expected in required.items()
        if values.get(flag) != expected
    }
    if mismatches:
        raise TraceRunnerError(f"signed-trace command drifted: {mismatches!r}")
    return command


def _source_snapshot() -> dict[str, Any]:
    files = [
        {
            "path": relative,
            "sha256": base._file_sha256(REPO_ROOT / relative),
            "size_bytes": (REPO_ROOT / relative).stat().st_size,
        }
        for relative in SOURCE_FILES
    ]
    return {"files": files, "sha256": base._canonical_sha256(files)}


def _build_plan() -> dict[str, Any]:
    if base._file_sha256(source.SOURCE_RECEIPT) != SOURCE_RECEIPT_SHA256:
        raise TraceRunnerError("source producer receipt changed")
    fixed_plan = fixed_runner._verify_plan()
    fixed_labels = dict(fixed_plan["fixed_labels"])
    if base._file_sha256(Path(fixed_labels["path"])) != fixed_labels["sha256"]:
        raise TraceRunnerError("fixed-label artifact changed")
    body = {
        "schema_version": 1,
        "experiment_id": "realq-fixed-label-gradient-trace-20260822-v1",
        "hostname": HOST,
        "source_producer_receipt": {
            "path": str(source.SOURCE_RECEIPT),
            "sha256": SOURCE_RECEIPT_SHA256,
        },
        "fixed_label_plan": {
            "path": str(fixed_runner.PLAN_PATH),
            "sha256": base._file_sha256(fixed_runner.PLAN_PATH),
            "fingerprint": fixed_plan["plan_fingerprint"],
        },
        "fixed_labels": fixed_labels,
        "source_snapshot": _source_snapshot(),
        "arms": {
            arm: {**definition, "command": _command(arm)}
            for arm, definition in ARMS.items()
        },
        "paired_contract": {
            "only_numerical_difference_between_arms": "attention_backend",
            "model": "Qwen3-4B",
            "calibration": "same frozen WikiText2 256x2048 token artifact",
            "seeds": {"calibration": 1, "rotation": 0},
            "global_loss_bsz": 4,
            "fixed_targets": "native deterministic-SDPA categorical labels",
            "production_label_sampler_called": False,
            "trace_batch": "first global batch, sample indices 0..3",
            "trace_sites": [
                "block output",
                "q/k/v projection output",
                "o projection input/output",
            ],
            "trace_phases": ["signed forward", "signed backward gradient"],
            "numerical_path_changed": False,
        },
    }
    body["plan_fingerprint"] = base._canonical_sha256(body)
    return body


def init_plan() -> dict[str, Any]:
    plan = _build_plan()
    if PLAN_PATH.is_file():
        if _read(PLAN_PATH) != plan:
            raise TraceRunnerError("existing signed-trace plan drifted")
        return plan
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise TraceRunnerError(f"signed-trace output root is not fresh: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True)
    base._atomic_json(PLAN_PATH, plan)
    return plan


def _verify_plan() -> dict[str, Any]:
    plan = _read(PLAN_PATH)
    stable = dict(plan)
    fingerprint = stable.pop("plan_fingerprint", None)
    if base._canonical_sha256(stable) != fingerprint:
        raise TraceRunnerError("signed-trace plan fingerprint mismatch")
    if plan != _build_plan():
        raise TraceRunnerError("signed-trace plan source or input drifted")
    return plan


def _environment(gpu: int, arm: str, trace_root: Path) -> dict[str, str]:
    environment = source._environment(gpu)
    fixed_labels = _read(PLAN_PATH)["fixed_labels"]
    environment.update(
        REALQ_FIXED_LABELS_PATH=fixed_labels["path"],
        REALQ_FIXED_LABELS_SHA256=fixed_labels["sha256"],
        REALQ_FIXED_LABELS_SUMMARY=fixed_labels["source_summary"],
        REALQ_SIGNED_TRACE_ROOT=str(trace_root),
        REALQ_SIGNED_TRACE_BACKEND=str(ARMS[arm]["backend"]),
    )
    return environment


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
        raise TraceRunnerError(completed.stderr.strip())
    return [
        int(line.strip())
        for line in completed.stdout.splitlines()
        if line.strip().isdigit()
    ]


def execute(arm: str, gpu: int) -> dict[str, Any]:
    if socket.gethostname() != HOST:
        raise TraceRunnerError("signed-trace child is on the wrong host")
    if arm not in ARMS or gpu != int(ARMS[arm]["physical_gpu"]):
        raise TraceRunnerError("signed-trace arm/GPU mapping changed")
    plan = _verify_plan()
    stage = OUTPUT_ROOT / arm
    if stage.exists() or stage.is_symlink():
        raise TraceRunnerError(f"signed-trace stage is not fresh: {stage}")
    stage.mkdir()
    command = list(map(str, plan["arms"][arm]["command"]))
    manifest = {
        "schema_version": 1,
        "status": "running",
        "kind": "realq_fixed_label_signed_trace_stage",
        "arm": arm,
        "backend": ARMS[arm]["backend"],
        "hostname": HOST,
        "physical_gpu": gpu,
        "plan_fingerprint": plan["plan_fingerprint"],
        "command": command,
        "command_sha256": base._canonical_sha256(command),
        "started_at": base._utc_now(),
    }
    base._atomic_json(stage / "manifest.json", manifest)
    trace_root = stage / "trace"
    log_path = stage / "execution.log"
    lock_path = LOCK_ROOT / HOST / f"gpu{gpu}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        pids = _gpu_compute_pids(gpu)
        if pids:
            raise TraceRunnerError(f"GPU {gpu} has untracked PIDs: {pids}")
        with log_path.open("xb") as log:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=_environment(gpu, arm, trace_root),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
    result = {
        **manifest,
        "status": "failed",
        "returncode": completed.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "finished_at": base._utc_now(),
        "log": str(log_path),
        "log_sha256": base._file_sha256(log_path),
    }
    receipt_path = trace_root / "trace_receipt.json"
    if completed.returncode == 0 and receipt_path.is_file():
        receipt = _read(receipt_path)
        trace_path = Path(str(receipt.get("trace", "")))
        if (
            receipt.get("status") != "completed"
            or receipt.get("backend") != ARMS[arm]["backend"]
            or receipt.get("labels_sha256") != plan["fixed_labels"]["sha256"]
            or receipt.get("completed_label_calls") != 64
            or receipt.get("precompute_calls") != 1
            or receipt.get("production_label_sampler_called") is not False
            or receipt.get("trace_tensors") != 36 * 6 * 2
            or not trace_path.is_file()
            or base._file_sha256(trace_path) != receipt.get("trace_sha256")
        ):
            raise TraceRunnerError("signed-trace receipt gate failed")
        result.update(
            status="succeeded",
            trace_receipt=str(receipt_path),
            trace_receipt_sha256=base._file_sha256(receipt_path),
            trace={
                "path": str(trace_path),
                "sha256": receipt["trace_sha256"],
                "size_bytes": trace_path.stat().st_size,
                "tensor_count": receipt["trace_tensors"],
            },
        )
    base._atomic_json(stage / "result.json", result)
    if result["status"] != "succeeded":
        raise TraceRunnerError(f"signed-trace stage failed: {arm}")
    return result


def launch_arm(arm: str) -> dict[str, Any]:
    if socket.gethostname() != HOST:
        raise TraceRunnerError("signed-trace launch must run on node1")
    if arm not in ARMS:
        raise TraceRunnerError(f"unknown signed-trace arm: {arm}")
    plan = _verify_plan()
    definition = ARMS[arm]
    if (OUTPUT_ROOT / arm).exists():
        raise TraceRunnerError(f"signed-trace arm was already claimed: {arm}")
    command = [
        str(plan["arms"][arm]["command"][0]),
        "-u",
        "-m",
        "experiments.realq_fixed_label_gradient_trace_20260822.runner",
        "--child",
        "--arm",
        arm,
        "--physical-gpu",
        str(definition["physical_gpu"]),
    ]
    log_path = OUTPUT_ROOT / f"launcher_{arm}.log"
    launch_path = OUTPUT_ROOT / f"launch_{arm}.json"
    if log_path.exists() or launch_path.exists():
        raise TraceRunnerError(f"signed-trace launch exists: {arm}")
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=source._environment(int(definition["physical_gpu"])),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    receipt = {
        "schema_version": 1,
        "status": "launched",
        "kind": "realq_fixed_label_signed_trace_stage",
        "arm": arm,
        "hostname": HOST,
        "physical_gpu": definition["physical_gpu"],
        "pid": process.pid,
        "plan_fingerprint": plan["plan_fingerprint"],
        "command": command,
        "log": str(log_path),
        "launched_at": base._utc_now(),
    }
    base._atomic_json(launch_path, receipt)
    return receipt


def launch_all() -> dict[str, Any]:
    return {"status": "launched", "arms": {arm: launch_arm(arm) for arm in ARMS}}


def _tensor_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise TraceRunnerError(
            f"signed trace tensor contract changed: {left.shape}/{right.shape}"
        )
    lhs = left.to(dtype=torch.float64).reshape(-1)
    rhs = right.to(dtype=torch.float64).reshape(-1)
    difference = rhs - lhs
    count = int(lhs.numel())
    left_square = float(torch.dot(lhs, lhs).item())
    right_square = float(torch.dot(rhs, rhs).item())
    diff_square = float(torch.dot(difference, difference).item())
    cross = float(torch.dot(lhs, rhs).item())
    left_rms = math.sqrt(left_square / count)
    right_rms = math.sqrt(right_square / count)
    diff_rms = math.sqrt(diff_square / count)
    denominator = math.sqrt(left_square * right_square)
    return {
        "sampled_values": count,
        "unequal_values": int(torch.count_nonzero(lhs != rhs).item()),
        "unequal_fraction": float(torch.count_nonzero(lhs != rhs).item()) / count,
        "left_rms": left_rms,
        "right_rms": right_rms,
        "diff_rms": diff_rms,
        "relative_rms_to_left": diff_rms / left_rms if left_rms else None,
        "relative_rms_to_right": diff_rms / right_rms if right_rms else None,
        "cosine": cross / denominator if denominator else None,
        "max_abs": float(difference.abs().max().item()),
    }


def _load_trace(path: Path) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or not value:
        raise TraceRunnerError(f"signed trace is not a tensor dictionary: {path}")
    if not all(isinstance(key, str) and torch.is_tensor(tensor) for key, tensor in value.items()):
        raise TraceRunnerError(f"signed trace contains an invalid entry: {path}")
    return value


def compare() -> dict[str, Any]:
    plan = _verify_plan()
    results = {arm: _read(OUTPUT_ROOT / arm / "result.json") for arm in ARMS}
    traces = {}
    receipts = {}
    for arm, result in results.items():
        trace_path = Path(str(result.get("trace", {}).get("path", "")))
        receipt_path = Path(str(result.get("trace_receipt", "")))
        if (
            result.get("status") != "succeeded"
            or result.get("plan_fingerprint") != plan["plan_fingerprint"]
            or not trace_path.is_file()
            or base._file_sha256(trace_path) != result["trace"].get("sha256")
            or not receipt_path.is_file()
            or base._file_sha256(receipt_path)
            != result.get("trace_receipt_sha256")
        ):
            raise TraceRunnerError(f"unbound signed-trace result: {arm}")
        traces[arm] = _load_trace(trace_path)
        receipts[arm] = _read(receipt_path)
    left = traces["trace_sdpa"]
    right = traces["trace_fa4"]
    if set(left) != set(right) or len(left) != 36 * 6 * 2:
        raise TraceRunnerError("paired signed-trace key set changed")
    left_metadata = receipts["trace_sdpa"]["trace_metadata"]
    right_metadata = receipts["trace_fa4"]["trace_metadata"]
    if left_metadata != right_metadata or set(left_metadata) != set(left):
        raise TraceRunnerError("paired signed-trace metadata changed")

    entries = {
        key: _tensor_metrics(left[key], right[key]) for key in sorted(left)
    }
    block_by_layer = [
        {
            "layer": layer,
            "forward": entries[f"layer{layer:02d}/block_output/forward"],
            "gradient": entries[f"layer{layer:02d}/block_output/gradient"],
        }
        for layer in range(36)
    ]
    first_forward_difference = {}
    for site in (
        "block_output",
        "q_proj_output",
        "k_proj_output",
        "v_proj_output",
        "o_proj_input",
        "o_proj_output",
    ):
        changed = [
            layer
            for layer in range(36)
            if entries[f"layer{layer:02d}/{site}/forward"]["unequal_values"]
        ]
        first_forward_difference[site] = min(changed) if changed else None
    result = {
        "schema_version": 1,
        "status": "succeeded",
        "kind": "realq_fixed_label_signed_trace_comparison",
        "plan_fingerprint": plan["plan_fingerprint"],
        "scientific_difference": "attention_backend only",
        "arms": {
            arm: {
                "result": str(OUTPUT_ROOT / arm / "result.json"),
                "result_sha256": base._file_sha256(
                    OUTPUT_ROOT / arm / "result.json"
                ),
                "trace_sha256": results[arm]["trace"]["sha256"],
                "elapsed_seconds": results[arm]["elapsed_seconds"],
            }
            for arm in ARMS
        },
        "first_forward_difference_layer": first_forward_difference,
        "layer0": {
            site: {
                phase: entries[f"layer00/{site}/{phase}"]
                for phase in ("forward", "gradient")
            }
            for site in (
                "block_output",
                "q_proj_output",
                "k_proj_output",
                "v_proj_output",
                "o_proj_input",
                "o_proj_output",
            )
        },
        "block_by_layer": block_by_layer,
        "entries": entries,
        "finished_at": base._utc_now(),
    }
    base._atomic_json(OUTPUT_ROOT / "comparison.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-plan", action="store_true")
    parser.add_argument("--launch-all", action="store_true")
    parser.add_argument("--launch-arm", choices=tuple(ARMS))
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args()
    try:
        modes = sum(
            (
                args.init_plan,
                args.launch_all,
                args.launch_arm is not None,
                args.compare,
                args.child,
            )
        )
        if modes != 1:
            parser.error("select exactly one signed-trace mode")
        if args.init_plan:
            result = init_plan()
        elif args.launch_all:
            result = launch_all()
        elif args.launch_arm is not None:
            result = launch_arm(args.launch_arm)
        elif args.compare:
            result = compare()
        else:
            if args.arm is None or args.physical_gpu is None:
                parser.error("signed-trace child arguments are incomplete")
            result = execute(args.arm, args.physical_gpu)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
