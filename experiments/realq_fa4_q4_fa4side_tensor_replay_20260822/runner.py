#!/usr/bin/env python3
"""Immutable single-GPU campaign for Qwen3-4B FA4-side attention replay."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
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
)
from experiments.realq_fa4_q4_fa4side_tensor_replay_20260822 import common


SOURCE_FILES = (
    "experiments/realq_fa4_q4_fa4side_tensor_replay_20260822/__init__.py",
    "experiments/realq_fa4_q4_fa4side_tensor_replay_20260822/common.py",
    "experiments/realq_fa4_q4_fa4side_tensor_replay_20260822/capture_driver.py",
    "experiments/realq_fa4_q4_fa4side_tensor_replay_20260822/replay_worker.py",
    "experiments/realq_fa4_q4_fa4side_tensor_replay_20260822/compare_worker.py",
    "experiments/realq_fa4_q4_fa4side_tensor_replay_20260822/runner.py",
    "experiments/realq_allopts_loss_ablation_20260821/cache_backend_diagnostic.py",
    "realq/ptq.py",
    "realq/pipeline.py",
    "realq/attention.py",
    "realq/precompute/static_e2e.py",
    "realq/precompute/hooks.py",
    "utils/reproducibility.py",
    "utils/rotation_utils.py",
)
THIRD_PARTY_RELATIVE_FILES = (
    "transformers/models/qwen3/modeling_qwen3.py",
    "transformers/integrations/sdpa_attention.py",
    "flash_attn/cute/interface.py",
    "flash_attn/cute/utils.py",
    "flash_attn/cute/flash_fwd_sm100.py",
    "flash_attn/cute/flash_bwd_sm100.py",
)
MINIMUM_FREE_DISK_BYTES = 20 * 1024**3
EXPECTED_OUTCOME_FILES_PER_ARM = len(common.SELECTED_LAYERS) * len(
    common.case_definitions()
)


def _flags(command: Sequence[str]) -> dict[str, str]:
    return source._flags(command)


def _set(command: list[str], flag: str, value: str) -> None:
    source._set(command, flag, value)


def _source_command() -> tuple[dict[str, Any], list[str]]:
    if (
        not common.SOURCE_PRODUCER_RECEIPT.is_file()
        or common.file_sha256(common.SOURCE_PRODUCER_RECEIPT)
        != common.SOURCE_PRODUCER_RECEIPT_SHA256
    ):
        raise common.ReplayDiagnosticError("run15 Qwen3-4B producer receipt changed")
    receipt = common.read_json(common.SOURCE_PRODUCER_RECEIPT)
    command = list(map(str, receipt.get("command", [])))
    if (
        receipt.get("status") != "succeeded"
        or receipt.get("model") != common.MODEL_SLUG
        or common.canonical_sha256(command) != receipt.get("command_sha256")
        or command[1:3] != ["-m", "realq.ptq"]
    ):
        raise common.ReplayDiagnosticError("run15 source command changed")
    values = _flags(command)
    expected = {
        "--model": str(common.MODEL_PATH),
        "--dataset": "wikitext2",
        "--seed": str(common.CALIBRATION_SEED),
        "--rotation_seed": str(common.ROTATION_SEED),
        "--refresh_seed": str(common.REFRESH_SEED),
        "--nsamples": "256",
        "--seq_len": str(common.SEQUENCE_LENGTH),
        "--global_loss_bsz": str(common.BATCH_SIZE),
        "--grad_hessian_topk": "-1",
        "--rotate": "true",
        "--attention_backend": "flash_attention_4",
        "--grad_lr": "0",
        "--exit_after_precompute": "true",
        "--tokens_cache_path": str(common.TOKEN_PATH.parent),
    }
    mismatch = {
        flag: (values.get(flag), value)
        for flag, value in expected.items()
        if values.get(flag) != value
    }
    if mismatch:
        raise common.ReplayDiagnosticError(f"run15 command drifted: {mismatch}")
    return receipt, command


def _capture_command() -> list[str]:
    _, command = _source_command()
    command[2] = (
        "experiments.realq_fa4_q4_fa4side_tensor_replay_20260822.capture_driver"
    )
    stage = common.OUTPUT_ROOT / "capture"
    overrides = {
        "--attention_backend": "flash_attention_4",
        "--static_cache_path": "",
        "--cache_dir": str(stage / "runtime"),
        "--output_dir": str(stage / "realq_output"),
        "--exp": "qwen3_4b_true_tensor_capture_fa4side",
        "--skip_eval": "true",
        "--skip_kl_ppl_eval": "true",
        "--lm_eval": "false",
        "--reasoning_eval": "false",
        "--require_static_cache_hit": "false",
    }
    for flag, value in overrides.items():
        _set(command, flag, value)
    return command


def _worker_command(arm: str) -> list[str]:
    _, source_command = _source_command()
    return [
        source_command[0],
        "-m",
        "experiments.realq_fa4_q4_fa4side_tensor_replay_20260822.replay_worker",
        "--arm",
        arm,
    ]


def _comparison_command() -> list[str]:
    _, source_command = _source_command()
    return [
        source_command[0],
        "-m",
        "experiments.realq_fa4_q4_fa4side_tensor_replay_20260822.compare_worker",
    ]


def _source_snapshot() -> dict[str, Any]:
    files = [
        {
            "path": relative,
            "sha256": common.file_sha256(REPO_ROOT / relative),
            "size_bytes": (REPO_ROOT / relative).stat().st_size,
        }
        for relative in SOURCE_FILES
    ]
    return {"files": files, "sha256": common.canonical_sha256(files)}


def _third_party_snapshot(python: str) -> dict[str, Any]:
    # Do not resolve the interpreter symlink: its absolute target is a
    # node-local binary, while the environment's site-packages live beside the
    # symlink on shared storage.
    environment_root = Path(python).absolute().parents[1]
    # The GPU node resolves this environment to Python 3.12; the paths are on
    # shared storage and can be hashed without importing CUDA packages here.
    site_packages = environment_root / "lib/python3.12/site-packages"
    files = []
    for relative in THIRD_PARTY_RELATIVE_FILES:
        path = site_packages / relative
        if not path.is_file():
            raise common.ReplayDiagnosticError(f"third-party source is missing: {path}")
        files.append(
            {
                "path": str(path),
                "relative": relative,
                "sha256": common.file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {"files": files, "sha256": common.canonical_sha256(files)}


def _validate_small_frozen_inputs() -> None:
    expected = {
        common.SOURCE_PRODUCER_RECEIPT: common.SOURCE_PRODUCER_RECEIPT_SHA256,
        common.TOKEN_PATH: common.TOKEN_SERIALIZATION_SHA256,
        common.FIXED_LABEL_RESULT: common.FIXED_LABEL_RESULT_SHA256,
        common.FIXED_LABEL_SUMMARY: common.FIXED_LABEL_SUMMARY_SHA256,
        common.FIXED_LABELS: common.FIXED_LABELS_SHA256,
        common.SDPA_TRAJECTORY_CAPTURE_RECEIPT: (
            common.SDPA_TRAJECTORY_CAPTURE_RECEIPT_SHA256
        ),
        common.MODEL_PATH / "config.json": common.MODEL_CONFIG_SHA256,
    }
    for path, expected_sha in expected.items():
        if not path.is_file() or common.file_sha256(path) != expected_sha:
            raise common.ReplayDiagnosticError(f"frozen input changed: {path}")
    tokens = __import__("torch").load(
        common.TOKEN_PATH, map_location="cpu", weights_only=True
    )
    if (
        not isinstance(tokens, list)
        or len(tokens) != 256
        or common.tensors_semantic_sha256(tokens) != common.TOKEN_SEMANTIC_SHA256
    ):
        raise common.ReplayDiagnosticError("calibration token semantics changed")


def _model_snapshot(*, verify_hashes: bool) -> dict[str, Any]:
    files = []
    for filename, expected_sha in common.MODEL_WEIGHT_SHA256.items():
        path = common.MODEL_PATH / filename
        if not path.is_file():
            raise common.ReplayDiagnosticError(f"model shard is missing: {path}")
        if verify_hashes and common.file_sha256(path) != expected_sha:
            raise common.ReplayDiagnosticError(f"model shard changed: {path}")
        files.append(
            {
                "path": str(path),
                "filename": filename,
                "sha256": expected_sha,
                "size_bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
        )
    return {"files": files, "sha256": common.canonical_sha256(files)}


def build_plan(physical_gpu: int, *, verify_model_hashes: bool) -> dict[str, Any]:
    if physical_gpu < 0:
        raise common.ReplayDiagnosticError("physical GPU index must be non-negative")
    _validate_small_frozen_inputs()
    source_receipt, source_command = _source_command()
    capture_command = _capture_command()
    body = {
        "schema_version": 1,
        "experiment_id": "realq-fa4-q4-fa4side-tensor-replay-20260822-v1",
        "hostname": common.HOST,
        "physical_gpu": int(physical_gpu),
        "source_snapshot": _source_snapshot(),
        "third_party_snapshot": _third_party_snapshot(capture_command[0]),
        "source_producer": {
            "path": str(common.SOURCE_PRODUCER_RECEIPT),
            "sha256": common.SOURCE_PRODUCER_RECEIPT_SHA256,
            "command_sha256": source_receipt["command_sha256"],
            "command": source_command,
        },
        "model": {
            "slug": common.MODEL_SLUG,
            "path": str(common.MODEL_PATH),
            "config_sha256": common.MODEL_CONFIG_SHA256,
            "weights": _model_snapshot(verify_hashes=verify_model_hashes),
            "layers": 36,
            "selected_layers": list(common.SELECTED_LAYERS),
            "followup_layers_not_in_v1": [
                layer
                for layer in common.ALL_REQUESTED_LAYERS
                if layer not in common.SELECTED_LAYERS
            ],
            "hidden_size": 2560,
            "query_heads": common.QUERY_HEADS,
            "key_value_heads": common.KEY_VALUE_HEADS,
            "head_dim": common.HEAD_DIM,
            "sliding_window": None,
        },
        "calibration": {
            "dataset": "wikitext2",
            "samples": 256,
            "sequence_length": common.SEQUENCE_LENGTH,
            "first_batch_indices": list(range(common.BATCH_SIZE)),
            "global_loss_bsz": common.BATCH_SIZE,
            "token_path": str(common.TOKEN_PATH),
            "token_serialization_sha256": common.TOKEN_SERIALIZATION_SHA256,
            "token_semantic_sha256": common.TOKEN_SEMANTIC_SHA256,
            "seeds": {
                "calibration": common.CALIBRATION_SEED,
                "rotation": common.ROTATION_SEED,
                "refresh": common.REFRESH_SEED,
                "random_control": common.RANDOM_CONTROL_SEED,
                "replay": common.REPLAY_SEED,
            },
        },
        "fixed_labels": {
            "path": str(common.FIXED_LABELS),
            "sha256": common.FIXED_LABELS_SHA256,
            "summary": str(common.FIXED_LABEL_SUMMARY),
            "summary_sha256": common.FIXED_LABEL_SUMMARY_SHA256,
            "source_result": str(common.FIXED_LABEL_RESULT),
            "source_result_sha256": common.FIXED_LABEL_RESULT_SHA256,
            "producer_backend": "math_sdpa",
            "used_rows": [0, 1, 2, 3],
        },
        "source_trajectory": {
            "backend": "math_sdpa",
            "capture_receipt": str(common.SDPA_TRAJECTORY_CAPTURE_RECEIPT),
            "capture_receipt_sha256": (
                common.SDPA_TRAJECTORY_CAPTURE_RECEIPT_SHA256
            ),
            "plan_fingerprint": common.SDPA_TRAJECTORY_PLAN_FINGERPRINT,
            "comparison": "all Q/K/V/dO elements, aggregate and per sample",
        },
        "capture": {
            "command": capture_command,
            "qkv_boundary": "post_qk_norm_post_rope_pre_attention_backend",
            "dout_boundary": "fa4 output pre_reshape_pre_o_proj",
            "loss": "fixed-label summed NLL times production LOSS_GRAD_SCALE=1000",
            "trajectory_backend": "flash_attention_4",
            "FA_DISABLE_2CTA": "0",
            "attention_mask": {
                "fa4_capture": "attention_mask=None, causal=true",
                "semantics": "dense unpadded causal attention",
            },
            "expected_q_shape": list(common.QUERY_SHAPE),
            "expected_kv_shape": list(common.KEY_VALUE_SHAPE),
            "expected_dout_shape": list(common.OUTPUT_SHAPE),
            "layout_contract": {
                "production_interface": (
                    "noncontiguous BHSD Q/K/V whose transpose is contiguous BSHD; "
                    "contiguous BSHD dO"
                ),
                "archive_container": "contiguous BHSD values plus contiguous BSHD dO",
                "reconstruction": (
                    "archive.transpose(1,2).contiguous().transpose(1,2) before kernel"
                ),
                "hash": (
                    "layout-aware SHA binds shape,dtype,stride,storage_offset,"
                    "is_contiguous and logical bytes"
                ),
            },
        },
        "replay": {
            "arms": {
                arm: {
                    "command": _worker_command(arm),
                    "FA_DISABLE_2CTA": "1" if arm == "fa4_no2cta" else "0",
                    "expected_forward_2cta_on_cuda12": (
                        False if arm.startswith("fa4_") else None
                    ),
                    "expected_backward_2cta_on_sm100_hd128": (
                        arm == "fa4_default" if arm.startswith("fa4_") else None
                    ),
                }
                for arm in common.ARMS
            },
            "comparison_command": _comparison_command(),
            "cases": [dict(item) for item in common.case_definitions()],
            "pairwise_comparisons": [list(pair) for pair in common.PAIRWISE_COMPARISONS],
            "reference": "explicit FP32 causal GQA matmul-softmax-matmul",
            "metrics": "all elements of output/dQ/dK/dV; no sampling",
            "causal_metrics": {
                "permutation": (
                    "per-arm real_b4[3,0,1,2] vs independent permuted B=4 call"
                ),
                "duplicate": (
                    "each slot of per-arm [s3,s3,s3,s3] B=4 vs independent B=1 s3"
                ),
                "sample_localization": (
                    "each real_b4 backend sample vs the corresponding FP32 sample"
                ),
            },
            "required_compute_capability": [10, 0],
            "fa4_import_isolation": "one fresh subprocess per FA4 arm",
            "fa4_cuda12_contract": (
                "CUDA12 automatically disables 2CTA forward; default FA4 "
                "uses 2CTA only in backward, FA_DISABLE_2CTA=1 disables it there"
            ),
        },
        "resource_contract": {
            "minimum_free_disk_bytes_before_launch": MINIMUM_FREE_DISK_BYTES,
            "raw_capture_bytes": (
                common.expected_capture_bytes_per_kind()
                * len(common.SELECTED_LAYERS)
                * 2
            ),
            "raw_outcome_bytes_per_arm": (
                common.expected_capture_bytes_per_kind()
                * len(common.SELECTED_LAYERS)
                * int(common.b4_equivalent_case_count())
            ),
            "expected_persistent_total_bytes_approx": 5 * 1024**3,
            "capture_peak_gpu_memory_note": (
                "full rotated Qwen3-4B FA4 first-batch graph; run only on "
                "an otherwise empty 180-GiB debug GPU"
            ),
            "reference_peak_gpu_memory_upper_estimate_bytes": 24 * 1024**3,
        },
        "environment_contract": {
            "PYTHONHASHSEED": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "NVIDIA_TF32_OVERRIDE": "0",
            "REALQ_DETERMINISTIC_SDPA": "1",
            "torch_deterministic_algorithms": True,
            "torch_matmul_tf32": False,
        },
    }
    body["plan_fingerprint"] = common.canonical_sha256(body)
    return body


def init_plan(physical_gpu: int) -> dict[str, Any]:
    candidate = build_plan(physical_gpu, verify_model_hashes=True)
    if common.PLAN_PATH.is_file():
        existing = common.read_json(common.PLAN_PATH)
        if existing != candidate:
            raise common.ReplayDiagnosticError("existing immutable plan drifted")
        return existing
    if common.OUTPUT_ROOT.exists() or common.OUTPUT_ROOT.is_symlink():
        raise common.ReplayDiagnosticError(
            f"output root is not fresh: {common.OUTPUT_ROOT}"
        )
    common.OUTPUT_ROOT.mkdir(parents=True)
    common.atomic_json(common.PLAN_PATH, candidate)
    return candidate


def verify_plan(*, verify_model_hashes: bool) -> dict[str, Any]:
    plan = common.read_json(common.PLAN_PATH)
    stable = dict(plan)
    fingerprint = stable.pop("plan_fingerprint", None)
    if common.canonical_sha256(stable) != fingerprint:
        raise common.ReplayDiagnosticError("plan fingerprint mismatch")
    expected = build_plan(
        int(plan["physical_gpu"]), verify_model_hashes=verify_model_hashes
    )
    if expected != plan:
        raise common.ReplayDiagnosticError("plan source or frozen inputs drifted")
    return plan


def _environment(gpu: int, extra: Mapping[str, str]) -> dict[str, str]:
    environment = source._environment(gpu)
    environment.update({str(key): str(value) for key, value in extra.items()})
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
        raise common.ReplayDiagnosticError(completed.stderr.strip())
    return [
        int(line.strip())
        for line in completed.stdout.splitlines()
        if line.strip().isdigit()
    ]


def _run_stage(
    plan: Mapping[str, Any],
    stage_name: str,
    command: Sequence[str],
    extra_environment: Mapping[str, str],
) -> dict[str, Any]:
    if socket.gethostname() != common.HOST:
        raise common.ReplayDiagnosticError("true-tensor replay is on the wrong host")
    current_plan = verify_plan(verify_model_hashes=False)
    if current_plan.get("plan_fingerprint") != plan.get("plan_fingerprint"):
        raise common.ReplayDiagnosticError("plan changed before stage launch")
    gpu = int(plan["physical_gpu"])
    stage = common.OUTPUT_ROOT / stage_name
    if stage.exists() or stage.is_symlink():
        raise common.ReplayDiagnosticError(f"stage is not fresh: {stage}")
    stage.mkdir()
    manifest = {
        "schema_version": 1,
        "status": "running",
        "kind": "qwen3_4b_true_tensor_replay_stage",
        "stage": stage_name,
        "hostname": common.HOST,
        "physical_gpu": gpu,
        "plan_fingerprint": plan["plan_fingerprint"],
        "command": list(map(str, command)),
        "command_sha256": common.canonical_sha256(list(map(str, command))),
        "environment_overrides": dict(sorted(extra_environment.items())),
        "started_at_epoch": time.time(),
    }
    common.atomic_json(stage / "manifest.json", manifest)
    lock_path = common.LOCK_ROOT / common.HOST / f"gpu{gpu}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = stage / "execution.log"
    started = time.monotonic()
    completed: subprocess.CompletedProcess[Any] | None = None
    try:
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            pids = _gpu_compute_pids(gpu)
            if pids:
                raise common.ReplayDiagnosticError(
                    f"physical GPU {gpu} has untracked PIDs: {pids}"
                )
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
        status = "succeeded" if completed.returncode == 0 else "failed"
        returncode = completed.returncode
    except BaseException:
        status = "failed"
        returncode = None if completed is None else completed.returncode
        raise
    finally:
        result = {
            **manifest,
            "status": status if "status" in locals() else "failed",
            "returncode": returncode if "returncode" in locals() else None,
            "elapsed_seconds": time.monotonic() - started,
            "finished_at_epoch": time.time(),
            "log": str(log_path),
            "log_sha256": common.file_sha256(log_path) if log_path.is_file() else None,
        }
        common.atomic_json(stage / "result.json", result)
    if result["status"] != "succeeded":
        raise common.ReplayDiagnosticError(f"stage failed: {stage_name}")
    return result


def _validate_capture(plan: Mapping[str, Any]) -> dict[str, Any]:
    root = common.OUTPUT_ROOT / "capture/capture"
    receipt_path = root / "capture_receipt.json"
    receipt = common.read_json(receipt_path)
    if (
        receipt.get("status") != "completed"
        or receipt.get("plan_fingerprint") != plan["plan_fingerprint"]
        or receipt.get("selected_layers") != list(common.SELECTED_LAYERS)
        or receipt.get("attention_backend") != "flash_attention_4"
        or receipt.get("fa_disable_2cta") != "0"
        or receipt.get("deterministic_algorithms") is not True
        or set(receipt.get("production_layouts", {}))
        != {str(layer) for layer in common.SELECTED_LAYERS}
        or receipt.get("production_layouts_sha256")
        != common.canonical_sha256(receipt.get("production_layouts", {}))
        or set(receipt.get("artifacts", {}))
        != {str(layer) for layer in common.SELECTED_LAYERS}
    ):
        raise common.ReplayDiagnosticError("capture terminal receipt changed")
    mask = receipt.get("attention_mask", {})
    mask_contract = mask.get("contract", {})
    if (
        mask.get("calls") != 36
        or mask.get("passed_to_fa4_unchanged") is not True
        or mask_contract.get("representation")
        != "none_dense_unpadded"
        or mask_contract.get("attention_mask") is not None
        or mask_contract.get("effective_is_causal") is not True
        or mask_contract.get("module_is_causal") is not True
    ):
        raise common.ReplayDiagnosticError("capture attention-mask receipt changed")
    for layer in common.SELECTED_LAYERS:
        production = receipt["production_layouts"][str(layer)]
        for name, heads in (
            ("q", common.QUERY_HEADS),
            ("k", common.KEY_VALUE_HEADS),
            ("v", common.KEY_VALUE_HEADS),
        ):
            original = production.get(name, {}).get("interface_bhsd", {})
            adapter = production.get(name, {}).get("adapter_bshd", {})
            expected_original = list(common.production_bhsd_stride(heads))
            expected_adapter = [
                common.SEQUENCE_LENGTH * heads * common.HEAD_DIM,
                heads * common.HEAD_DIM,
                common.HEAD_DIM,
                1,
            ]
            if (
                original.get("stride") != expected_original
                or original.get("storage_offset") != 0
                or original.get("is_contiguous") is not False
                or adapter.get("stride") != expected_adapter
                or adapter.get("storage_offset") != 0
                or adapter.get("is_contiguous") is not True
            ):
                raise common.ReplayDiagnosticError(
                    f"production attention layout receipt changed: layer={layer} {name}"
                )
        dout = production.get("dout", {}).get("interface_bshd", {})
        expected_dout = [
            common.SEQUENCE_LENGTH * common.QUERY_HEADS * common.HEAD_DIM,
            common.QUERY_HEADS * common.HEAD_DIM,
            common.HEAD_DIM,
            1,
        ]
        if (
            dout.get("stride") != expected_dout
            or dout.get("storage_offset") != 0
            or dout.get("is_contiguous") is not True
        ):
            raise common.ReplayDiagnosticError(
                f"production dO layout receipt changed: layer={layer}"
            )
        for kind in ("real", "random"):
            record = receipt["artifacts"][str(layer)][kind]
            path = Path(str(record["path"]))
            if (
                not path.is_file()
                or common.file_sha256(path) != record.get("serialization_sha256")
                or not record.get("layout_semantic_sha256")
                or record.get("representation")
                != "contiguous_value_archive_not_kernel_input"
            ):
                raise common.ReplayDiagnosticError(f"capture artifact failed: {path}")
    return receipt


def _validate_arm(plan: Mapping[str, Any], arm: str) -> dict[str, Any]:
    path = common.OUTPUT_ROOT / arm / "replay/replay_receipt.json"
    receipt = common.read_json(path)
    expected_disable = "1" if arm == "fa4_no2cta" else "0"
    expected_forward = False if arm.startswith("fa4_") else None
    expected_backward = arm == "fa4_default" if arm.startswith("fa4_") else None
    if (
        receipt.get("status") != "completed"
        or receipt.get("plan_fingerprint") != plan["plan_fingerprint"]
        or receipt.get("arm") != arm
        or receipt.get("fa_disable_2cta") != expected_disable
        or receipt.get("expected_forward_2cta") is not expected_forward
        or receipt.get("expected_backward_2cta") is not expected_backward
        or len(receipt.get("records", {})) != EXPECTED_OUTCOME_FILES_PER_ARM
    ):
        raise common.ReplayDiagnosticError(f"replay terminal receipt changed: {arm}")
    runtime = receipt.get("fa4_runtime")
    if arm.startswith("fa4_") and (
        not isinstance(runtime, dict)
        or not str(runtime.get("torch_cuda_version", "")).startswith("12.")
        or runtime.get("fa_disable_2cta_cuda12_forward_flag") is not True
        or runtime.get("expected_forward_2cta") is not False
        or runtime.get("expected_backward_2cta") is not expected_backward
    ):
        raise common.ReplayDiagnosticError(f"FA4 2CTA receipt changed: {arm}")
    for record in receipt["records"].values():
        outcome = Path(str(record["path"]))
        if (
            not outcome.is_file()
            or common.file_sha256(outcome) != record.get("serialization_sha256")
            or not record.get("layout_semantic_sha256")
            or not record.get("input_archive_layout_semantic_sha256")
            or not record.get("input_layouts", {}).get("kernel")
            or not isinstance(record.get("source_indices"), list)
        ):
            raise common.ReplayDiagnosticError(f"unbound replay outcome: {outcome}")
    return receipt


def _validate_comparison(plan: Mapping[str, Any]) -> dict[str, Any]:
    path = common.OUTPUT_ROOT / "comparison/comparison/comparison_receipt.json"
    receipt = common.read_json(path)
    if (
        receipt.get("status") != "completed"
        or receipt.get("plan_fingerprint") != plan["plan_fingerprint"]
        or receipt.get("metrics_scope")
        != "all tensor elements; no token/feature sampling"
        or set(receipt.get("layers", {}))
        != {str(layer) for layer in common.SELECTED_LAYERS}
    ):
        raise common.ReplayDiagnosticError("comparison terminal receipt changed")
    expected_all_arms = {"fp32_reference", *common.ARMS}
    for layer in common.SELECTED_LAYERS:
        layer_record = receipt["layers"][str(layer)]
        if set(layer_record.get("cases", {})) != {
            str(definition["name"]) for definition in common.case_definitions()
        }:
            raise common.ReplayDiagnosticError(
                f"comparison case coverage changed: layer={layer}"
            )
        causal = layer_record.get("causal_case_metrics", {})
        trajectory = layer_record.get("source_trajectory_comparison", {})
        permutation = causal.get("permutation_equivariance", {})
        repeated = causal.get("repeated_s3_vs_independent_b1_s3", {})
        per_sample = causal.get("real_b4_per_sample_vs_fp32", {})
        if (
            trajectory.get("left") != "math_sdpa_side"
            or trajectory.get("right") != "flash_attention_4_side"
            or set(trajectory.get("fields", {})) != {"q", "k", "v", "dout"}
            or set(trajectory.get("samples", {}))
            != {str(index) for index in range(common.BATCH_SIZE)}
            or any(
                metric.get("all_values") is not True
                for metric in trajectory.get("fields", {}).values()
            )
            or any(
                set(sample.get("fields", {})) != {"q", "k", "v", "dout"}
                or any(
                    metric.get("all_values") is not True
                    for metric in sample.get("fields", {}).values()
                )
                for sample in trajectory.get("samples", {}).values()
            )
            or
            set(permutation) != expected_all_arms
            or set(repeated) != expected_all_arms
            or set(per_sample) != set(common.ARMS)
        ):
            raise common.ReplayDiagnosticError(
                f"comparison causal metric coverage changed: layer={layer}"
            )
        metric_groups = []
        for arm in expected_all_arms:
            metric_groups.append(permutation[arm].get("fields", {}))
            slots = repeated[arm].get("slots", {})
            if set(slots) != {str(index) for index in range(common.BATCH_SIZE)}:
                raise common.ReplayDiagnosticError(
                    f"duplicate-slot metric coverage changed: layer={layer} arm={arm}"
                )
            metric_groups.extend(slot.get("fields", {}) for slot in slots.values())
        for arm in common.ARMS:
            samples = per_sample[arm].get("samples", {})
            if set(samples) != {str(index) for index in range(common.BATCH_SIZE)}:
                raise common.ReplayDiagnosticError(
                    f"per-sample metric coverage changed: layer={layer} arm={arm}"
                )
            metric_groups.extend(
                sample.get("fields", {}) for sample in samples.values()
            )
        for fields in metric_groups:
            if set(fields) != set(common.OUTPUT_FIELDS) or any(
                metric.get("all_values") is not True
                for metric in fields.values()
            ):
                raise common.ReplayDiagnosticError(
                    f"comparison metric stopped covering full tensors: layer={layer}"
                )
    return receipt


def pipeline() -> dict[str, Any]:
    plan = verify_plan(verify_model_hashes=False)
    if socket.gethostname() != common.HOST:
        raise common.ReplayDiagnosticError("pipeline must run on node1")
    if shutil.disk_usage(common.DATA_ROOT).free < MINIMUM_FREE_DISK_BYTES:
        raise common.ReplayDiagnosticError("less than 20 GiB free for immutable outcomes")
    fingerprint = str(plan["plan_fingerprint"])
    capture_root = common.OUTPUT_ROOT / "capture/capture"
    capture_result = _run_stage(
        plan,
        "capture",
        plan["capture"]["command"],
        {
            "FA_DISABLE_2CTA": "0",
            "REALQ_TRUE_TENSOR_CAPTURE_ROOT": str(capture_root),
            "REALQ_TRUE_TENSOR_PLAN_FINGERPRINT": fingerprint,
        },
    )
    capture_receipt = _validate_capture(plan)

    arm_results: dict[str, Any] = {}
    arm_receipts: dict[str, Any] = {}
    for arm in common.ARMS:
        replay_root = common.OUTPUT_ROOT / arm / "replay"
        arm_results[arm] = _run_stage(
            plan,
            arm,
            plan["replay"]["arms"][arm]["command"],
            {
                "FA_DISABLE_2CTA": plan["replay"]["arms"][arm][
                    "FA_DISABLE_2CTA"
                ],
                "REALQ_TRUE_TENSOR_REPLAY_ROOT": str(replay_root),
                "REALQ_TRUE_TENSOR_CAPTURE_ROOT": str(capture_root),
                "REALQ_TRUE_TENSOR_PLAN_FINGERPRINT": fingerprint,
            },
        )
        arm_receipts[arm] = _validate_arm(plan, arm)

    comparison_root = common.OUTPUT_ROOT / "comparison/comparison"
    comparison_result = _run_stage(
        plan,
        "comparison",
        plan["replay"]["comparison_command"],
        {
            "FA_DISABLE_2CTA": "0",
            "REALQ_TRUE_TENSOR_OUTPUT_ROOT": str(common.OUTPUT_ROOT),
            "REALQ_TRUE_TENSOR_COMPARISON_ROOT": str(comparison_root),
            "REALQ_TRUE_TENSOR_CAPTURE_ROOT": str(capture_root),
            "REALQ_TRUE_TENSOR_PLAN_FINGERPRINT": fingerprint,
        },
    )
    comparison_receipt = _validate_comparison(plan)
    final = {
        "schema_version": 1,
        "status": "completed",
        "kind": "qwen3_4b_fa4side_true_attention_replay_campaign",
        "plan_fingerprint": fingerprint,
        "physical_gpu": plan["physical_gpu"],
        "capture": {
            "stage_result_sha256": common.file_sha256(
                common.OUTPUT_ROOT / "capture/result.json"
            ),
            "receipt_sha256": common.file_sha256(
                capture_root / "capture_receipt.json"
            ),
            "elapsed_seconds": capture_result["elapsed_seconds"],
        },
        "arms": {
            arm: {
                "stage_result_sha256": common.file_sha256(
                    common.OUTPUT_ROOT / arm / "result.json"
                ),
                "receipt_sha256": common.file_sha256(
                    common.OUTPUT_ROOT / arm / "replay/replay_receipt.json"
                ),
                "elapsed_seconds": arm_results[arm]["elapsed_seconds"],
                "fa_disable_2cta": arm_receipts[arm]["fa_disable_2cta"],
            }
            for arm in common.ARMS
        },
        "comparison": {
            "stage_result_sha256": common.file_sha256(
                common.OUTPUT_ROOT / "comparison/result.json"
            ),
            "receipt": str(comparison_root / "comparison_receipt.json"),
            "receipt_sha256": common.file_sha256(
                comparison_root / "comparison_receipt.json"
            ),
            "elapsed_seconds": comparison_result["elapsed_seconds"],
            "metrics_scope": comparison_receipt["metrics_scope"],
        },
    }
    common.atomic_json(common.OUTPUT_ROOT / "campaign_result.json", final)
    return final


def launch() -> dict[str, Any]:
    if socket.gethostname() != common.HOST:
        raise common.ReplayDiagnosticError("launch must run on node1")
    plan = verify_plan(verify_model_hashes=True)
    if (common.OUTPUT_ROOT / "launch.json").exists():
        raise common.ReplayDiagnosticError("campaign was already launched")
    if any((common.OUTPUT_ROOT / stage).exists() for stage in ("capture", *common.ARMS, "comparison")):
        raise common.ReplayDiagnosticError("campaign stage already exists")
    command = [
        str(plan["capture"]["command"][0]),
        "-u",
        "-m",
        "experiments.realq_fa4_q4_fa4side_tensor_replay_20260822.runner",
        "--pipeline",
    ]
    log_path = common.OUTPUT_ROOT / "launcher.log"
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=_environment(int(plan["physical_gpu"]), {"FA_DISABLE_2CTA": "0"}),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    result = {
        "schema_version": 1,
        "status": "launched",
        "pid": process.pid,
        "physical_gpu": plan["physical_gpu"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "command": command,
        "log": str(log_path),
        "launched_at_epoch": time.time(),
    }
    common.atomic_json(common.OUTPUT_ROOT / "launch.json", result)
    return result


def status() -> dict[str, Any]:
    plan = common.read_json(common.PLAN_PATH)
    stages = {}
    for stage in ("capture", *common.ARMS, "comparison"):
        result = common.OUTPUT_ROOT / stage / "result.json"
        stages[stage] = common.read_json(result) if result.is_file() else None
    final = common.OUTPUT_ROOT / "campaign_result.json"
    return {
        "status": "observed",
        "plan_fingerprint": plan.get("plan_fingerprint"),
        "stages": stages,
        "campaign_result": common.read_json(final) if final.is_file() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-plan", action="store_true")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--pipeline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    try:
        modes = sum((args.init_plan, args.launch, args.pipeline, args.status))
        if modes != 1:
            parser.error("select exactly one execution mode")
        if args.init_plan:
            if args.physical_gpu is None:
                parser.error("--init-plan requires --physical-gpu")
            result = init_plan(args.physical_gpu)
        elif args.launch:
            if args.physical_gpu is not None:
                parser.error("GPU is already frozen in plan.json")
            result = launch()
        elif args.pipeline:
            result = pipeline()
        else:
            result = status()
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
