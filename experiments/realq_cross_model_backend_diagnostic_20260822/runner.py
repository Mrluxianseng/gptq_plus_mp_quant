#!/usr/bin/env python3
"""Run fixed-label SDPA/FA4 REAL-Q traces across Qwen3 model sizes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
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
from experiments.realq_fullmodel_retune_20260817 import campaign as base


DATA_ROOT = REPO_ROOT.parent / "experiment_data"
VARIANT = os.environ.get("REALQ_CROSS_MODEL_BACKEND_VARIANT", "v1")
if VARIANT not in {"v1", "q8_bsz4_v2", "q4_bsz1_v3"}:
    raise RuntimeError(f"unknown REALQ_CROSS_MODEL_BACKEND_VARIANT: {VARIANT}")
OUTPUT_ROOT = DATA_ROOT / {
    "v1": "realq_cross_model_backend_diagnostic_20260822_v1",
    "q8_bsz4_v2": "realq_cross_model_backend_diagnostic_q8_bsz4_20260822_v2",
    "q4_bsz1_v3": "realq_cross_model_backend_diagnostic_q4_bsz1_20260822_v3",
}[VARIANT]
PLAN_PATH = OUTPUT_ROOT / "plan.json"
LOCK_ROOT = DATA_ROOT / "_fair20_physical_gpu_locks_20260821"
HOST = "j-zogxxxduju-master-0"
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "qwen3-0.6b": {
        "layers": 28,
        "global_loss_bsz": 1,
        "capture_gpu": 0,
        "trace_gpus": {"sdpa": 0, "flash_attention_4": 2},
        "producer_sha256": "729531f166fba063e621e28590e8f105ea10d78210a520c4c89a5aec17a37e88",
        "tokens_filename": "Qwen3-0.6B_wikitext2_train_n256_sl2048_seed1.pt",
        "tokens_sha256": "aed4e972d31207526d57a4bb62414c24b6c92ff51eda1fa9d70c36c1dfea5d34",
        "tokens_semantic_sha256": "5b7bd51b6f896a4b79e70f8ffa867974440dc5c63b6c8af7668c013d76998535",
    },
    "qwen3-8b": {
        "layers": 36,
        "global_loss_bsz": 8,
        "capture_gpu": 3,
        "trace_gpus": {"sdpa": 3, "flash_attention_4": 4},
        "producer_sha256": "3b478f1dad1985ffe5e95a9fe47e4ba6e98990108a72fbbca0481ffee7402969",
        "tokens_filename": "Qwen3-8B_wikitext2_train_n256_sl2048_seed1.pt",
        "tokens_sha256": "a7acbd907d640eb8ab33a52153da1bf36e4eb550cbc107fe081c9bdc047d3b6f",
        "tokens_semantic_sha256": "5b7bd51b6f896a4b79e70f8ffa867974440dc5c63b6c8af7668c013d76998535",
    },
}
if VARIANT == "q8_bsz4_v2":
    MODEL_SPECS = {
        "qwen3-8b": {
            **MODEL_SPECS["qwen3-8b"],
            "producer_global_loss_bsz": 8,
            "global_loss_bsz": 4,
        }
    }
elif VARIANT == "q4_bsz1_v3":
    MODEL_SPECS = {
        "qwen3-4b": {
            "layers": 36,
            "producer_global_loss_bsz": 4,
            "global_loss_bsz": 1,
            "capture_gpu": 0,
            "trace_gpus": {"sdpa": 0, "flash_attention_4": 2},
            "producer_sha256": "2405c6c6fa339e27e3b5d02f26564481c96c8f36e58d3231210ed533d047d2aa",
            "tokens_filename": "Qwen3-4B_wikitext2_train_n256_sl2048_seed1.pt",
            "tokens_sha256": "21210e1929aa90ea23572e7904f8ebda7b3af75ebf8d9037e8196b3d2c52a399",
            "tokens_semantic_sha256": "5b7bd51b6f896a4b79e70f8ffa867974440dc5c63b6c8af7668c013d76998535",
        }
    }
SOURCE_FILES = (
    "experiments/realq_cross_model_backend_diagnostic_20260822/runner.py",
    "experiments/realq_cross_model_backend_diagnostic_20260822/trace_driver.py",
    "experiments/realq_backend_label_diagnostic_20260822/capture_driver.py",
    "experiments/realq_fixed_label_backend_diagnostic_20260822/fixed_label_driver.py",
    "realq/ptq.py",
    "realq/pipeline.py",
    "realq/attention.py",
    "realq/precompute/static_e2e.py",
    "realq/precompute/labels.py",
    "realq/precompute/hooks.py",
    "utils/reproducibility.py",
)
BOOKKEEPING_FLAGS = {
    "--attention_backend",
    "--static_cache_path",
    "--cache_dir",
    "--output_dir",
    "--exp",
}


class CrossModelRunnerError(RuntimeError):
    """A scheduling, provenance, or numerical comparison gate failed."""


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CrossModelRunnerError(f"JSON root is not an object: {path}")
    return value


def _flags(command: Sequence[str]) -> dict[str, str]:
    return source._flags(command)


def _set(command: list[str], flag: str, value: str) -> None:
    source._set(command, flag, value)


def _producer_path(model: str) -> Path:
    return (
        DATA_ROOT
        / "realq_fullmodel_retune_20260818_v6_run15/shared_cache"
        / model
        / "producer_success.json"
    )


def _token_semantic_sha256(path: Path) -> str:
    values = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(values, list) or len(values) != 256:
        raise CrossModelRunnerError(f"unexpected calibration token archive: {path}")
    digest = hashlib.sha256()
    for value in values:
        if not torch.is_tensor(value):
            raise CrossModelRunnerError(f"non-tensor calibration entry: {path}")
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"|")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"|")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _source_command(model: str) -> tuple[dict[str, Any], list[str]]:
    if model not in MODEL_SPECS:
        raise CrossModelRunnerError(f"unknown model: {model}")
    path = _producer_path(model)
    if base._file_sha256(path) != MODEL_SPECS[model]["producer_sha256"]:
        raise CrossModelRunnerError(f"run15 producer receipt changed: {model}")
    receipt = _read(path)
    command = list(map(str, receipt.get("command", [])))
    if (
        receipt.get("status") != "succeeded"
        or receipt.get("model") != model
        or base._canonical_sha256(command) != receipt.get("command_sha256")
    ):
        raise CrossModelRunnerError(f"invalid run15 producer receipt: {model}")
    values = _flags(command)
    expected = {
        "--dataset": "wikitext2",
        "--seed": "1",
        "--rotation_seed": "0",
        "--nsamples": "256",
        "--seq_len": "2048",
        "--global_loss_bsz": str(
            MODEL_SPECS[model].get(
                "producer_global_loss_bsz", MODEL_SPECS[model]["global_loss_bsz"]
            )
        ),
        "--hessian_accum_bsz": "64",
        "--grad_hessian_topk": "-1",
        "--rotate": "true",
        "--attention_backend": "flash_attention_4",
        "--exit_after_precompute": "true",
        "--grad_lr": "0",
    }
    mismatch = {
        key: (values.get(key), value)
        for key, value in expected.items()
        if values.get(key) != value
    }
    if mismatch:
        raise CrossModelRunnerError(f"run15 command drifted: {model}: {mismatch}")
    tokens_dir = Path(values.get("--tokens_cache_path", ""))
    tokens_path = tokens_dir / str(MODEL_SPECS[model]["tokens_filename"])
    if (
        not tokens_path.is_file()
        or base._file_sha256(tokens_path) != MODEL_SPECS[model]["tokens_sha256"]
        or _token_semantic_sha256(tokens_path)
        != MODEL_SPECS[model]["tokens_semantic_sha256"]
    ):
        raise CrossModelRunnerError(f"run15 calibration tokens changed: {model}")
    _set(command, "--global_loss_bsz", str(MODEL_SPECS[model]["global_loss_bsz"]))
    return receipt, command


def _stage_command(model: str, kind: str, backend: str) -> list[str]:
    if kind not in {"capture", "trace"}:
        raise CrossModelRunnerError(f"invalid stage kind: {kind}")
    if backend not in {"sdpa", "flash_attention_4"}:
        raise CrossModelRunnerError(f"invalid backend: {backend}")
    if kind == "capture" and backend != "sdpa":
        raise CrossModelRunnerError("fixed labels must be captured from SDPA")
    _receipt, command = _source_command(model)
    command[2] = (
        "experiments.realq_backend_label_diagnostic_20260822.capture_driver"
        if kind == "capture"
        else "experiments.realq_cross_model_backend_diagnostic_20260822.trace_driver"
    )
    stage_name = "capture_sdpa" if kind == "capture" else f"trace_{backend}"
    stage = OUTPUT_ROOT / model / stage_name
    overrides = {
        "--attention_backend": backend,
        "--static_cache_path": "",
        "--cache_dir": str(stage / "runtime"),
        "--output_dir": str(stage / "realq_output"),
        "--exp": f"cross_model_backend_{model}_{stage_name}",
        "--skip_eval": "true",
        "--skip_kl_ppl_eval": "true",
        "--lm_eval": "false",
        "--reasoning_eval": "false",
        "--require_static_cache_hit": "false",
    }
    for flag, value in overrides.items():
        _set(command, flag, value)
    return command


def _scientific_projection(command: Sequence[str]) -> dict[str, str]:
    return {
        key: value
        for key, value in _flags(command).items()
        if key not in BOOKKEEPING_FLAGS
    }


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
    models: dict[str, Any] = {}
    for model, spec in MODEL_SPECS.items():
        capture = _stage_command(model, "capture", "sdpa")
        trace_sdpa = _stage_command(model, "trace", "sdpa")
        trace_fa4 = _stage_command(model, "trace", "flash_attention_4")
        if _scientific_projection(trace_sdpa) != _scientific_projection(trace_fa4):
            raise CrossModelRunnerError(f"trace arms differ beyond backend: {model}")
        models[model] = {
            **spec,
            "expected_label_calls": 256 // int(spec["global_loss_bsz"]),
            "producer_receipt": {
                "path": str(_producer_path(model)),
                "sha256": spec["producer_sha256"],
            },
            "calibration_tokens": {
                "path": str(
                    Path(_flags(capture)["--tokens_cache_path"])
                    / str(spec["tokens_filename"])
                ),
                "sha256": spec["tokens_sha256"],
                "semantic_sha256": spec["tokens_semantic_sha256"],
            },
            "commands": {
                "capture_sdpa": capture,
                "trace_sdpa": trace_sdpa,
                "trace_flash_attention_4": trace_fa4,
            },
        }
    body = {
        "schema_version": 1,
        "experiment_id": f"realq-cross-model-backend-diagnostic-20260822-{VARIANT}",
        "hostname": HOST,
        "source_snapshot": _source_snapshot(),
        "models": models,
        "contract": {
            "calibration": "each model's frozen run15 WikiText2 256x2048 token cache",
            "seeds": {"calibration": 1, "rotation": 0},
            "fixed_targets": "same model's deterministic-SDPA categorical labels",
            "trace_batch": "first production global-loss batch",
            "only_trace_arm_numerical_difference": "attention_backend",
            "deterministic_environment": True,
            "formal_global_loss_bsz_preserved_per_model": all(
                int(spec.get("producer_global_loss_bsz", spec["global_loss_bsz"]))
                == int(spec["global_loss_bsz"])
                for spec in MODEL_SPECS.values()
            ),
            "global_loss_bsz_override_reason": {
                "v1": None,
                "q8_bsz4_v2": (
                    "Qwen3-8B math-SDPA capture OOMs at run15 bsz=8; "
                    "both arms use bsz=4"
                ),
                "q4_bsz1_v3": (
                    "Qwen3-4B per-sample isolation; both arms use bsz=1 "
                    "instead of run15 bsz=4"
                ),
            }[VARIANT],
        },
    }
    body["plan_fingerprint"] = base._canonical_sha256(body)
    return body


def init_plan() -> dict[str, Any]:
    plan = _build_plan()
    if PLAN_PATH.is_file():
        if _read(PLAN_PATH) != plan:
            raise CrossModelRunnerError("existing cross-model plan drifted")
        return plan
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise CrossModelRunnerError(f"output root is not fresh: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True)
    base._atomic_json(PLAN_PATH, plan)
    return plan


def _verify_plan() -> dict[str, Any]:
    plan = _read(PLAN_PATH)
    stable = dict(plan)
    fingerprint = stable.pop("plan_fingerprint", None)
    if base._canonical_sha256(stable) != fingerprint:
        raise CrossModelRunnerError("plan fingerprint mismatch")
    if plan != _build_plan():
        raise CrossModelRunnerError("plan source or inputs drifted")
    return plan


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
        raise CrossModelRunnerError(completed.stderr.strip())
    return [
        int(line.strip())
        for line in completed.stdout.splitlines()
        if line.strip().isdigit()
    ]


def _environment(gpu: int, extra: Mapping[str, str]) -> dict[str, str]:
    environment = source._environment(gpu)
    environment.update({str(key): str(value) for key, value in extra.items()})
    return environment


def _run_command(
    *,
    model: str,
    stage_name: str,
    gpu: int,
    command: Sequence[str],
    extra_environment: Mapping[str, str],
) -> dict[str, Any]:
    if socket.gethostname() != HOST:
        raise CrossModelRunnerError("cross-model diagnostic is on the wrong host")
    stage = OUTPUT_ROOT / model / stage_name
    if stage.exists() or stage.is_symlink():
        raise CrossModelRunnerError(f"stage is not fresh: {stage}")
    stage.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "kind": "realq_cross_model_backend_stage",
        "model": model,
        "stage": stage_name,
        "hostname": HOST,
        "physical_gpu": gpu,
        "plan_fingerprint": _read(PLAN_PATH)["plan_fingerprint"],
        "command": list(map(str, command)),
        "command_sha256": base._canonical_sha256(command),
        "started_at": base._utc_now(),
    }
    base._atomic_json(stage / "manifest.json", manifest)
    lock_path = LOCK_ROOT / HOST / f"gpu{gpu}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = stage / "execution.log"
    started = time.monotonic()
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        pids = _gpu_compute_pids(gpu)
        if pids:
            raise CrossModelRunnerError(f"GPU {gpu} has untracked PIDs: {pids}")
        with log_path.open("xb") as log:
            completed = subprocess.run(
                list(map(str, command)),
                cwd=REPO_ROOT,
                env=_environment(gpu, extra_environment),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
    result = {
        **manifest,
        "status": "succeeded" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "finished_at": base._utc_now(),
        "log": str(log_path),
        "log_sha256": base._file_sha256(log_path),
    }
    base._atomic_json(stage / "result.json", result)
    if result["status"] != "succeeded":
        raise CrossModelRunnerError(f"stage failed: {model}/{stage_name}")
    return result


def execute_capture(model: str) -> dict[str, Any]:
    plan = _verify_plan()
    spec = plan["models"][model]
    stage = OUTPUT_ROOT / model / "capture_sdpa"
    capture_root = stage / "capture"
    result = _run_command(
        model=model,
        stage_name="capture_sdpa",
        gpu=int(spec["capture_gpu"]),
        command=spec["commands"]["capture_sdpa"],
        extra_environment={
            "REALQ_LABEL_CAPTURE_ROOT": str(capture_root),
            "REALQ_LABEL_CAPTURE_BACKEND": "sdpa",
        },
    )
    summary_path = capture_root / "capture_summary.json"
    summary = _read(summary_path)
    labels_path = Path(str(summary.get("labels", "")))
    flattened = [
        int(index)
        for record in summary.get("records", [])
        for index in record.get("global_sample_indices", [])
    ]
    if (
        summary.get("status") != "completed"
        or summary.get("backend") != "sdpa"
        or summary.get("total_labels") != 256 * 2048
        or summary.get("invocations") != spec["expected_label_calls"]
        or flattened != list(range(256))
        or not labels_path.is_file()
        or labels_path.stat().st_size != 256 * 2048 * 8
        or base._file_sha256(labels_path) != summary.get("labels_sha256")
    ):
        raise CrossModelRunnerError(f"capture terminal gate failed: {model}")
    result.update(
        capture_summary=str(summary_path),
        capture_summary_sha256=base._file_sha256(summary_path),
        labels=str(labels_path),
        labels_sha256=summary["labels_sha256"],
    )
    base._atomic_json(stage / "result.json", result)
    return result


def _capture_result(model: str) -> dict[str, Any]:
    result = _read(OUTPUT_ROOT / model / "capture_sdpa/result.json")
    labels = Path(str(result.get("labels", "")))
    summary = Path(str(result.get("capture_summary", "")))
    if (
        result.get("status") != "succeeded"
        or not labels.is_file()
        or base._file_sha256(labels) != result.get("labels_sha256")
        or not summary.is_file()
        or base._file_sha256(summary) != result.get("capture_summary_sha256")
    ):
        raise CrossModelRunnerError(f"unbound label capture: {model}")
    return result


def execute_trace(model: str, backend: str) -> dict[str, Any]:
    plan = _verify_plan()
    spec = plan["models"][model]
    capture = _capture_result(model)
    stage_name = f"trace_{backend}"
    trace_root = OUTPUT_ROOT / model / stage_name / "trace"
    result = _run_command(
        model=model,
        stage_name=stage_name,
        gpu=int(spec["trace_gpus"][backend]),
        command=spec["commands"][stage_name],
        extra_environment={
            "REALQ_FIXED_LABELS_PATH": capture["labels"],
            "REALQ_FIXED_LABELS_SHA256": capture["labels_sha256"],
            "REALQ_FIXED_LABELS_SUMMARY": capture["capture_summary"],
            "REALQ_CROSS_MODEL_TRACE_ROOT": str(trace_root),
            "REALQ_CROSS_MODEL_TRACE_BACKEND": backend,
            "REALQ_CROSS_MODEL_SLUG": model,
            "REALQ_EXPECTED_LAYERS": str(spec["layers"]),
            "REALQ_EXPECTED_LABEL_CALLS": str(spec["expected_label_calls"]),
        },
    )
    receipt_path = trace_root / "trace_receipt.json"
    receipt = _read(receipt_path)
    trace_path = Path(str(receipt.get("trace", "")))
    if (
        receipt.get("status") != "completed"
        or receipt.get("model") != model
        or receipt.get("backend") != backend
        or receipt.get("labels_sha256") != capture["labels_sha256"]
        or receipt.get("completed_label_calls") != spec["expected_label_calls"]
        or receipt.get("precompute_calls") != 1
        or receipt.get("trace_tensors") != int(spec["layers"]) * 6 * 2
        or receipt.get("production_label_sampler_called") is not False
        or not trace_path.is_file()
        or base._file_sha256(trace_path) != receipt.get("trace_sha256")
    ):
        raise CrossModelRunnerError(f"trace terminal gate failed: {model}/{backend}")
    result.update(
        trace_receipt=str(receipt_path),
        trace_receipt_sha256=base._file_sha256(receipt_path),
        trace={
            "path": str(trace_path),
            "sha256": receipt["trace_sha256"],
            "size_bytes": trace_path.stat().st_size,
            "tensor_count": receipt["trace_tensors"],
        },
    )
    base._atomic_json(OUTPUT_ROOT / model / stage_name / "result.json", result)
    return result


def _tensor_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise CrossModelRunnerError("paired trace tensor contract changed")
    lhs = left.to(dtype=torch.float64).reshape(-1)
    rhs = right.to(dtype=torch.float64).reshape(-1)
    difference = rhs - lhs
    count = int(lhs.numel())
    left_square = float(torch.dot(lhs, lhs).item())
    right_square = float(torch.dot(rhs, rhs).item())
    diff_square = float(torch.dot(difference, difference).item())
    cross = float(torch.dot(lhs, rhs).item())
    denominator = math.sqrt(left_square * right_square)
    left_rms = math.sqrt(left_square / count)
    right_rms = math.sqrt(right_square / count)
    diff_rms = math.sqrt(diff_square / count)
    unequal = int(torch.count_nonzero(lhs != rhs).item())
    return {
        "sampled_values": count,
        "unequal_values": unequal,
        "unequal_fraction": unequal / count,
        "left_rms": left_rms,
        "right_rms": right_rms,
        "right_to_left_rms_ratio": right_rms / left_rms if left_rms else None,
        "diff_rms": diff_rms,
        "relative_rms_to_left": diff_rms / left_rms if left_rms else None,
        "relative_rms_to_right": diff_rms / right_rms if right_rms else None,
        "cosine": cross / denominator if denominator else None,
        "max_abs": float(difference.abs().max().item()),
    }


def compare_model(model: str) -> dict[str, Any]:
    plan = _verify_plan()
    spec = plan["models"][model]
    results = {
        backend: _read(OUTPUT_ROOT / model / f"trace_{backend}/result.json")
        for backend in ("sdpa", "flash_attention_4")
    }
    traces: dict[str, dict[str, torch.Tensor]] = {}
    metadata = None
    for backend, result in results.items():
        path = Path(str(result.get("trace", {}).get("path", "")))
        receipt_path = Path(str(result.get("trace_receipt", "")))
        if (
            result.get("status") != "succeeded"
            or result.get("plan_fingerprint") != plan["plan_fingerprint"]
            or not path.is_file()
            or base._file_sha256(path) != result.get("trace", {}).get("sha256")
            or not receipt_path.is_file()
            or base._file_sha256(receipt_path) != result.get("trace_receipt_sha256")
        ):
            raise CrossModelRunnerError(f"unbound trace result: {model}/{backend}")
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(loaded, dict):
            raise CrossModelRunnerError(f"invalid trace archive: {model}/{backend}")
        traces[backend] = loaded
        current_metadata = _read(receipt_path)["trace_metadata"]
        if metadata is None:
            metadata = current_metadata
        elif current_metadata != metadata:
            raise CrossModelRunnerError(f"paired trace metadata changed: {model}")
    left = traces["sdpa"]
    right = traces["flash_attention_4"]
    expected_count = int(spec["layers"]) * 6 * 2
    if set(left) != set(right) or len(left) != expected_count:
        raise CrossModelRunnerError(f"paired trace key set changed: {model}")
    entries = {key: _tensor_metrics(left[key], right[key]) for key in sorted(left)}
    block_by_layer = [
        {
            "layer": layer,
            "forward": entries[f"layer{layer:02d}/block_output/forward"],
            "gradient": entries[f"layer{layer:02d}/block_output/gradient"],
        }
        for layer in range(int(spec["layers"]))
    ]
    result = {
        "schema_version": 1,
        "status": "succeeded",
        "kind": "realq_cross_model_fixed_label_trace_comparison",
        "model": model,
        "layers": spec["layers"],
        "global_loss_bsz": spec["global_loss_bsz"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "scientific_difference": "attention_backend only",
        "arms": {
            backend: {
                "result": str(OUTPUT_ROOT / model / f"trace_{backend}/result.json"),
                "result_sha256": base._file_sha256(
                    OUTPUT_ROOT / model / f"trace_{backend}/result.json"
                ),
                "elapsed_seconds": results[backend]["elapsed_seconds"],
            }
            for backend in results
        },
        "block_by_layer": block_by_layer,
        "entries": entries,
        "finished_at": base._utc_now(),
    }
    base._atomic_json(OUTPUT_ROOT / model / "comparison.json", result)
    return result


def pipeline(model: str) -> dict[str, Any]:
    execute_capture(model)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            backend: executor.submit(execute_trace, model, backend)
            for backend in ("sdpa", "flash_attention_4")
        }
        for future in futures.values():
            future.result()
    return compare_model(model)


def launch_model(model: str) -> dict[str, Any]:
    if socket.gethostname() != HOST:
        raise CrossModelRunnerError("cross-model launch must run on node1")
    plan = _verify_plan()
    model_root = OUTPUT_ROOT / model
    if model_root.exists() or model_root.is_symlink():
        raise CrossModelRunnerError(f"model was already claimed: {model}")
    command = [
        str(plan["models"][model]["commands"]["capture_sdpa"][0]),
        "-u",
        "-m",
        "experiments.realq_cross_model_backend_diagnostic_20260822.runner",
        "--pipeline",
        model,
    ]
    log_path = OUTPUT_ROOT / f"launcher_{model}.log"
    launch_path = OUTPUT_ROOT / f"launch_{model}.json"
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=source._environment(int(plan["models"][model]["capture_gpu"])),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    launch = {
        "schema_version": 1,
        "status": "launched",
        "model": model,
        "pid": process.pid,
        "plan_fingerprint": plan["plan_fingerprint"],
        "command": command,
        "log": str(log_path),
        "launched_at": base._utc_now(),
    }
    base._atomic_json(launch_path, launch)
    return launch


def launch_all() -> dict[str, Any]:
    return {"status": "launched", "models": {model: launch_model(model) for model in MODEL_SPECS}}


def status() -> dict[str, Any]:
    models = {}
    for model in MODEL_SPECS:
        model_root = OUTPUT_ROOT / model
        stages = {}
        for stage in ("capture_sdpa", "trace_sdpa", "trace_flash_attention_4"):
            result = model_root / stage / "result.json"
            stages[stage] = _read(result) if result.is_file() else None
        comparison = model_root / "comparison.json"
        models[model] = {
            "stages": stages,
            "comparison": _read(comparison) if comparison.is_file() else None,
        }
    return {"status": "observed", "models": models}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-plan", action="store_true")
    parser.add_argument("--launch-all", action="store_true")
    parser.add_argument("--launch-model", choices=tuple(MODEL_SPECS))
    parser.add_argument("--pipeline", choices=tuple(MODEL_SPECS), help=argparse.SUPPRESS)
    parser.add_argument("--compare-model", choices=tuple(MODEL_SPECS))
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    try:
        modes = sum(
            (
                args.init_plan,
                args.launch_all,
                args.launch_model is not None,
                args.pipeline is not None,
                args.compare_model is not None,
                args.status,
            )
        )
        if modes != 1:
            parser.error("select exactly one mode")
        if args.init_plan:
            result = init_plan()
        elif args.launch_all:
            result = launch_all()
        elif args.launch_model is not None:
            result = launch_model(args.launch_model)
        elif args.pipeline is not None:
            result = pipeline(args.pipeline)
        elif args.compare_model is not None:
            result = compare_model(args.compare_model)
        else:
            result = status()
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
