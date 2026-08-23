#!/usr/bin/env python3
"""Compare full REAL-Q Stage-0 Fisher caches with labels held fixed."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.realq_allopts_loss_ablation_20260821 import (
    cache_backend_diagnostic as source,
    compare_static_cache,
)
from experiments.realq_fullmodel_retune_20260817 import campaign as base


DATA_ROOT = REPO_ROOT.parent / "experiment_data"
OUTPUT_ROOT = DATA_ROOT / "realq_fixed_label_backend_diagnostic_20260822_v1"
PLAN_PATH = OUTPUT_ROOT / "plan.json"
LOCK_ROOT = DATA_ROOT / "_fair20_physical_gpu_locks_20260821"
HOST = "j-zogxxxduju-master-0"
LABEL_CAPTURE_ROOT = DATA_ROOT / "realq_backend_label_diagnostic_20260822_v1"
LABEL_SUMMARY = LABEL_CAPTURE_ROOT / "sdpa/capture/capture_summary.json"
LABELS_PATH = LABEL_CAPTURE_ROOT / "sdpa/capture/labels.i64"
NATIVE_SDPA_CACHE = (
    DATA_ROOT
    / "realq_static_backend_diagnostic_20260821_v1/sdpa_current_code/static"
    / "Qwen3-4B_wikitext2_n256_sl2048_e306d5b89c27_world1_rank0.pt"
)
NATIVE_FA4_CACHE = source.RUN15_FA4_CACHE
SOURCE_RECEIPT_SHA256 = (
    "2405c6c6fa339e27e3b5d02f26564481c96c8f36e58d3231210ed533d047d2aa"
)
ARMS = {
    "fixed_sdpa": {"backend": "sdpa", "physical_gpu": 0},
    "fixed_fa4": {"backend": "flash_attention_4", "physical_gpu": 2},
}
SOURCE_FILES = (
    "experiments/realq_fixed_label_backend_diagnostic_20260822/fixed_label_driver.py",
    "experiments/realq_fixed_label_backend_diagnostic_20260822/runner.py",
    "experiments/realq_allopts_loss_ablation_20260821/cache_backend_diagnostic.py",
    "experiments/realq_allopts_loss_ablation_20260821/compare_static_cache.py",
    "realq/ptq.py",
    "realq/attention.py",
    "realq/precompute/cache.py",
    "realq/precompute/static_e2e.py",
    "realq/precompute/labels.py",
    "realq/precompute/hooks.py",
    "utils/reproducibility.py",
)


class FixedLabelDiagnosticError(RuntimeError):
    """A fixed-label diagnostic safety or provenance gate failed."""


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FixedLabelDiagnosticError(f"JSON root is not an object: {path}")
    return value


def _flags(command: Sequence[str]) -> dict[str, str]:
    return source._flags(command)


def _set(command: list[str], flag: str, value: str) -> None:
    source._set(command, flag, value)


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "inode": stat.st_ino,
    }


def _command(arm: str) -> list[str]:
    if arm not in ARMS:
        raise FixedLabelDiagnosticError(f"unknown arm: {arm}")
    _receipt, command = source._source_command()
    if command[1:3] != ["-m", "realq.ptq"]:
        raise FixedLabelDiagnosticError("source command entry point changed")
    command[2] = (
        "experiments.realq_fixed_label_backend_diagnostic_20260822."
        "fixed_label_driver"
    )
    stage = OUTPUT_ROOT / arm
    overrides = {
        "--attention_backend": str(ARMS[arm]["backend"]),
        "--static_cache_path": str(stage / "static"),
        "--cache_dir": str(stage / "runtime"),
        "--output_dir": str(stage / "realq_output"),
        "--exp": f"realq_fixed_label_backend_{arm}",
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
    }
    mismatches = {
        flag: (values.get(flag), expected)
        for flag, expected in required.items()
        if values.get(flag) != expected
    }
    if mismatches:
        raise FixedLabelDiagnosticError(
            f"fixed-label scientific contract drifted: {mismatches!r}"
        )
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
        raise FixedLabelDiagnosticError("source producer receipt changed")
    label_summary = _read(LABEL_SUMMARY)
    labels_sha256 = base._file_sha256(LABELS_PATH)
    if (
        label_summary.get("status") != "completed"
        or label_summary.get("backend") != "sdpa"
        or label_summary.get("labels_sha256") != labels_sha256
        or label_summary.get("total_labels") != 256 * 2048
        or label_summary.get("invocations") != 64
    ):
        raise FixedLabelDiagnosticError("native SDPA label capture changed")
    for cache in (NATIVE_SDPA_CACHE, NATIVE_FA4_CACHE):
        if not cache.is_file() or cache.stat().st_size <= 0:
            raise FixedLabelDiagnosticError(f"native cache is missing: {cache}")
    body = {
        "schema_version": 1,
        "experiment_id": "realq-fixed-label-backend-diagnostic-20260822-v1",
        "hostname": HOST,
        "source_producer_receipt": {
            "path": str(source.SOURCE_RECEIPT),
            "sha256": SOURCE_RECEIPT_SHA256,
        },
        "source_snapshot": _source_snapshot(),
        "fixed_labels": {
            "path": str(LABELS_PATH.resolve()),
            "sha256": labels_sha256,
            "size_bytes": LABELS_PATH.stat().st_size,
            "source_summary": str(LABEL_SUMMARY.resolve()),
            "source_summary_sha256": base._file_sha256(LABEL_SUMMARY),
            "source_backend": "sdpa",
        },
        "native_caches": {
            "sdpa": _file_identity(NATIVE_SDPA_CACHE),
            "flash_attention_4": _file_identity(NATIVE_FA4_CACHE),
        },
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
            "output": "full block-boundary Fisher and linear saliency cache",
        },
    }
    body["plan_fingerprint"] = base._canonical_sha256(body)
    return body


def init_plan() -> dict[str, Any]:
    plan = _build_plan()
    if PLAN_PATH.is_file():
        if _read(PLAN_PATH) != plan:
            raise FixedLabelDiagnosticError("existing fixed-label plan drifted")
        return plan
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise FixedLabelDiagnosticError(f"output root is not fresh: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True)
    base._atomic_json(PLAN_PATH, plan)
    return plan


def _verify_plan() -> dict[str, Any]:
    plan = _read(PLAN_PATH)
    stable = dict(plan)
    fingerprint = stable.pop("plan_fingerprint", None)
    if base._canonical_sha256(stable) != fingerprint:
        raise FixedLabelDiagnosticError("fixed-label plan fingerprint mismatch")
    if plan != _build_plan():
        raise FixedLabelDiagnosticError("fixed-label plan source or input drifted")
    return plan


def _environment(gpu: int, arm: str, audit_root: Path) -> dict[str, str]:
    environment = source._environment(gpu)
    fixed = _read(PLAN_PATH)["fixed_labels"]
    environment.update(
        REALQ_FIXED_LABELS_PATH=fixed["path"],
        REALQ_FIXED_LABELS_SHA256=fixed["sha256"],
        REALQ_FIXED_LABELS_SUMMARY=fixed["source_summary"],
        REALQ_FIXED_LABEL_AUDIT_ROOT=str(audit_root),
        REALQ_FIXED_LABEL_BACKEND=str(ARMS[arm]["backend"]),
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
        raise FixedLabelDiagnosticError(completed.stderr.strip())
    return [
        int(line.strip())
        for line in completed.stdout.splitlines()
        if line.strip().isdigit()
    ]


def execute(arm: str, gpu: int) -> dict[str, Any]:
    if socket.gethostname() != HOST:
        raise FixedLabelDiagnosticError("fixed-label child is on the wrong host")
    if arm not in ARMS or gpu != int(ARMS[arm]["physical_gpu"]):
        raise FixedLabelDiagnosticError("fixed-label arm/GPU mapping changed")
    plan = _verify_plan()
    stage = OUTPUT_ROOT / arm
    if stage.exists() or stage.is_symlink():
        raise FixedLabelDiagnosticError(f"fixed-label stage is not fresh: {stage}")
    stage.mkdir()
    command = list(map(str, plan["arms"][arm]["command"]))
    manifest = {
        "schema_version": 1,
        "status": "running",
        "kind": "realq_fixed_label_backend_stage0",
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
    audit_root = stage / "fixed_label_injection"
    log_path = stage / "execution.log"
    lock_path = LOCK_ROOT / HOST / f"gpu{gpu}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        pids = _gpu_compute_pids(gpu)
        if pids:
            raise FixedLabelDiagnosticError(f"GPU {gpu} has untracked PIDs: {pids}")
        with log_path.open("xb") as log:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=_environment(gpu, arm, audit_root),
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
    injection_path = audit_root / "injection_receipt.json"
    cache_files = tuple((stage / "static").glob("*.pt"))
    if completed.returncode == 0 and injection_path.is_file() and len(cache_files) == 1:
        injection = _read(injection_path)
        cache = cache_files[0]
        if (
            injection.get("status") != "completed"
            or injection.get("backend") != ARMS[arm]["backend"]
            or injection.get("labels_sha256") != plan["fixed_labels"]["sha256"]
            or injection.get("completed_calls") != 64
            or injection.get("production_label_sampler_called") is not False
        ):
            raise FixedLabelDiagnosticError("fixed-label injection receipt failed")
        result.update(
            status="succeeded",
            injection_receipt=str(injection_path),
            injection_receipt_sha256=base._file_sha256(injection_path),
            cache={
                "path": str(cache),
                "size_bytes": cache.stat().st_size,
                "sha256": base._file_sha256(cache),
            },
        )
    base._atomic_json(stage / "result.json", result)
    if result["status"] != "succeeded":
        raise FixedLabelDiagnosticError(f"fixed-label stage failed: {arm}")
    return result


def launch_arm(arm: str) -> dict[str, Any]:
    if socket.gethostname() != HOST:
        raise FixedLabelDiagnosticError("fixed-label launch must run on node1")
    plan = _verify_plan()
    definition = ARMS[arm]
    if (OUTPUT_ROOT / arm).exists():
        raise FixedLabelDiagnosticError(f"fixed-label arm was already claimed: {arm}")
    command = [
        str(plan["arms"][arm]["command"][0]),
        "-u",
        "-m",
        "experiments.realq_fixed_label_backend_diagnostic_20260822.runner",
        "--child",
        "--arm",
        arm,
        "--physical-gpu",
        str(definition["physical_gpu"]),
    ]
    log_path = OUTPUT_ROOT / f"launcher_{arm}.log"
    launch_path = OUTPUT_ROOT / f"launch_{arm}.json"
    if log_path.exists() or launch_path.exists():
        raise FixedLabelDiagnosticError(f"fixed-label launch exists: {arm}")
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
        "kind": "realq_fixed_label_backend_stage0",
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


def _compact(comparison: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "sampled_values",
        "unequal_fraction",
        "rms",
        "left_rms",
        "right_rms",
        "relative_rms_to_left",
        "relative_rms_to_right",
        "cosine",
        "max_abs",
        "max_abs_location",
    )
    return {
        kind: {key: comparison[kind][key] for key in keys}
        for kind in ("fisher", "saliency")
    }


def compare() -> dict[str, Any]:
    plan = _verify_plan()
    results = {arm: _read(OUTPUT_ROOT / arm / "result.json") for arm in ARMS}
    for arm, result in results.items():
        cache = Path(result.get("cache", {}).get("path", ""))
        if (
            result.get("status") != "succeeded"
            or result.get("plan_fingerprint") != plan["plan_fingerprint"]
            or not cache.is_file()
            or base._file_sha256(cache) != result["cache"].get("sha256")
        ):
            raise FixedLabelDiagnosticError(f"unbound fixed-label result: {arm}")
    fixed_sdpa = Path(results["fixed_sdpa"]["cache"]["path"])
    fixed_fa4 = Path(results["fixed_fa4"]["cache"]["path"])
    comparisons = {
        "fixed_sdpa_vs_fixed_fa4": compare_static_cache.compare(
            fixed_sdpa, fixed_fa4
        ),
        "native_sdpa_vs_native_fa4": compare_static_cache.compare(
            NATIVE_SDPA_CACHE, NATIVE_FA4_CACHE
        ),
        "native_sdpa_vs_fixed_sdpa": compare_static_cache.compare(
            NATIVE_SDPA_CACHE, fixed_sdpa
        ),
        "native_fa4_vs_fixed_fa4": compare_static_cache.compare(
            NATIVE_FA4_CACHE, fixed_fa4
        ),
    }
    for name, comparison in comparisons.items():
        base._atomic_json(OUTPUT_ROOT / f"{name}.json", comparison)
    result = {
        "schema_version": 1,
        "status": "succeeded",
        "kind": "realq_fixed_label_backend_fisher_comparison",
        "plan_fingerprint": plan["plan_fingerprint"],
        "scientific_question": (
            "attention backend difference after categorical targets are held fixed"
        ),
        "arms": {
            arm: {
                "result": str(OUTPUT_ROOT / arm / "result.json"),
                "result_sha256": base._file_sha256(
                    OUTPUT_ROOT / arm / "result.json"
                ),
                "cache_sha256": results[arm]["cache"]["sha256"],
                "elapsed_seconds": results[arm]["elapsed_seconds"],
            }
            for arm in ARMS
        },
        "comparisons": {
            name: _compact(comparison)
            for name, comparison in comparisons.items()
        },
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
            parser.error("select exactly one fixed-label diagnostic mode")
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
                parser.error("fixed-label child arguments are incomplete")
            result = execute(args.arm, args.physical_gpu)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
