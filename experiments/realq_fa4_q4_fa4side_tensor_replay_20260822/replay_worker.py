#!/usr/bin/env python3
"""Replay one local attention backend on frozen Q/K/V+dO tensors."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
import traceback
from typing import Any, Mapping, Sequence

import torch

from experiments.realq_fa4_q4_fa4side_tensor_replay_20260822 import common
from utils.reproducibility import configure_reproducibility


class _AttentionModule(torch.nn.Module):
    def __init__(self, layer: int):
        super().__init__()
        self.layer_idx = int(layer)
        self.num_key_value_groups = common.QUERY_HEADS // common.KEY_VALUE_HEADS
        self.is_causal = True


def _run_math_sdpa(
    layer: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    module = _AttentionModule(layer)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        output, _ = sdpa_attention_forward(
            module,
            q,
            k,
            v,
            None,
            dropout=0.0,
            scaling=common.SCALING,
            is_causal=True,
            sliding_window=None,
        )
    return output


def _run_fa4(
    layer: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    # Import only after the runner has installed the arm-specific environment.
    from flash_attn.cute import utils as fa4_utils
    from realq.attention import flash_attention_4_forward

    requested_disable = os.environ.get("FA_DISABLE_2CTA") == "1"
    if bool(fa4_utils._fa_disable_2cta_enabled) != requested_disable:
        raise common.ReplayDiagnosticError(
            "FA_DISABLE_2CTA was not frozen before flash_attn import"
        )
    common.validate_kernel_qkv({"q": q, "k": k, "v": v}, batch=int(q.shape[0]))
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        native = tensor.transpose(1, 2)
        if not native.is_contiguous() or native.storage_offset() != 0:
            raise common.ReplayDiagnosticError(
                f"FA4 native {name} is not contiguous BSHD: "
                f"{common.tensor_layout(native)}"
            )
    module = _AttentionModule(layer)
    output, _ = flash_attention_4_forward(
        module,
        q,
        k,
        v,
        None,
        dropout=0.0,
        scaling=common.SCALING,
        sliding_window=None,
        is_causal=True,
    )
    return output


def run_attention_backward(
    arm: str,
    layer: int,
    tensors: Mapping[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    batch = int(tensors["q"].shape[0])
    common.validate_archive_qkvd(tensors, batch=batch)
    archive_gpu = {
        name: value.to(device=device, memory_format=torch.preserve_format)
        for name, value in tensors.items()
    }
    for name, value in archive_gpu.items():
        if common.tensor_layout(value) != common.tensor_layout(tensors[name]):
            raise common.ReplayDiagnosticError(
                f"CPU->GPU archive layout changed for {name}: "
                f"cpu={common.tensor_layout(tensors[name])}, gpu={common.tensor_layout(value)}"
            )
    kernel = common.reconstruct_kernel_qkvd(archive_gpu, batch=batch)
    common.validate_kernel_qkvd(kernel, batch=batch)
    q = kernel["q"].detach().requires_grad_(True)
    k = kernel["k"].detach().requires_grad_(True)
    v = kernel["v"].detach().requires_grad_(True)
    dout = kernel["dout"]
    kernel_layout = {
        name: {
            "interface": common.tensor_layout(value),
            **(
                {"adapter_native": common.tensor_layout(value.transpose(1, 2))}
                if name in {"q", "k", "v"}
                else {}
            ),
        }
        for name, value in kernel.items()
    }
    kernel_layout_sha256 = common.named_tensors_layout_sha256(kernel)
    if arm == "math_sdpa":
        output = _run_math_sdpa(layer, q, k, v)
    elif arm in {"fa4_default", "fa4_no2cta"}:
        output = _run_fa4(layer, q, k, v)
    else:
        raise common.ReplayDiagnosticError(f"unknown replay arm: {arm}")
    expected_output = (batch, common.SEQUENCE_LENGTH, common.QUERY_HEADS, common.HEAD_DIM)
    if tuple(output.shape) != expected_output or output.dtype != common.TENSOR_DTYPE:
        raise common.ReplayDiagnosticError(
            f"{arm} output contract changed: {tuple(output.shape)}/{output.dtype}"
        )
    dq, dk, dv = torch.autograd.grad(
        output,
        (q, k, v),
        grad_outputs=dout,
        retain_graph=False,
        create_graph=False,
    )
    torch.cuda.synchronize(device)
    result = {
        "output": output.detach().to(device="cpu", memory_format=torch.preserve_format),
        "dq": dq.detach().to(device="cpu", memory_format=torch.preserve_format),
        "dk": dk.detach().to(device="cpu", memory_format=torch.preserve_format),
        "dv": dv.detach().to(device="cpu", memory_format=torch.preserve_format),
    }
    common.validate_outcome(result, batch=batch)
    return result, {
        "archive": {
            name: common.tensor_layout(value) for name, value in tensors.items()
        },
        "kernel": kernel_layout,
        "archive_layout_semantic_sha256": common.named_tensors_layout_sha256(tensors),
        "kernel_layout_semantic_sha256": kernel_layout_sha256,
    }


def _capture_receipt(capture_root: Path, plan_fingerprint: str) -> dict[str, Any]:
    path = capture_root / "capture_receipt.json"
    receipt = common.read_json(path)
    if (
        receipt.get("status") != "completed"
        or receipt.get("plan_fingerprint") != plan_fingerprint
        or receipt.get("selected_layers") != list(common.SELECTED_LAYERS)
        or receipt.get("batch_size") != common.BATCH_SIZE
        or receipt.get("sequence_length") != common.SEQUENCE_LENGTH
        or receipt.get("attention_backend") != "flash_attention_4"
        or receipt.get("fixed_label_sha256") != common.FIXED_LABELS_SHA256
        or receipt.get("production_layouts_sha256")
        != common.canonical_sha256(receipt.get("production_layouts", {}))
    ):
        raise common.ReplayDiagnosticError("capture receipt contract changed")
    return receipt


def _load_capture_kind(
    receipt: Mapping[str, Any], layer: int, kind: str
) -> dict[str, torch.Tensor]:
    record = receipt["artifacts"][str(layer)][kind]
    path = Path(str(record["path"]))
    if (
        not path.is_file()
        or common.file_sha256(path) != record.get("serialization_sha256")
    ):
        raise common.ReplayDiagnosticError(f"unbound capture artifact: {path}")
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise common.ReplayDiagnosticError("capture archive root is not a mapping")
    common.validate_archive_qkvd(loaded)
    if common.named_tensors_semantic_sha256(loaded) != record.get("semantic_sha256"):
        raise common.ReplayDiagnosticError("capture artifact semantic SHA256 changed")
    if (
        common.named_tensors_layout_sha256(loaded)
        != record.get("layout_semantic_sha256")
    ):
        raise common.ReplayDiagnosticError("capture artifact layout SHA256 changed")
    actual_contracts = {
        name: common.tensor_contract(value) for name, value in sorted(loaded.items())
    }
    if actual_contracts != record.get("tensors"):
        raise common.ReplayDiagnosticError("capture artifact tensor contracts changed")
    return loaded


def _write_outcome(
    root: Path,
    arm: str,
    layer: int,
    case: str,
    input_tensors: Mapping[str, torch.Tensor],
    outcome: Mapping[str, torch.Tensor],
    input_layouts: Mapping[str, Any],
    source_indices: Sequence[int],
    *,
    elapsed_seconds: float,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
) -> dict[str, Any]:
    path = root / "outcomes" / f"layer{layer:02d}" / f"{case}.pt"
    common.atomic_torch_save(path, dict(outcome))
    return {
        "arm": arm,
        "layer": layer,
        "case": case,
        "path": str(path),
        "serialization_sha256": common.file_sha256(path),
        "semantic_sha256": common.named_tensors_semantic_sha256(outcome),
        "layout_semantic_sha256": common.named_tensors_layout_sha256(outcome),
        "input_semantic_sha256": common.named_tensors_semantic_sha256(input_tensors),
        "input_archive_layout_semantic_sha256": common.named_tensors_layout_sha256(
            input_tensors
        ),
        "input_layouts": dict(input_layouts),
        "source_indices": list(map(int, source_indices)),
        "size_bytes": path.stat().st_size,
        "elapsed_seconds": elapsed_seconds,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "tensors": {
            name: common.tensor_contract(value)
            for name, value in sorted(outcome.items())
        },
    }


def execute(arm: str, root: Path, capture_root: Path, plan_fingerprint: str) -> dict[str, Any]:
    if arm not in common.ARMS:
        raise common.ReplayDiagnosticError(f"unknown arm: {arm}")
    if root.exists() or root.is_symlink():
        raise common.ReplayDiagnosticError(f"replay root is not fresh: {root}")
    expected_disable = "1" if arm == "fa4_no2cta" else "0"
    if os.environ.get("FA_DISABLE_2CTA") != expected_disable:
        raise common.ReplayDiagnosticError(
            f"{arm} requires FA_DISABLE_2CTA={expected_disable} before import"
        )
    root.mkdir(parents=True)
    capture = _capture_receipt(capture_root, plan_fingerprint)
    configure_reproducibility(common.REPLAY_SEED, deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise common.ReplayDiagnosticError("replay worker requires exactly one visible GPU")
    device = torch.device("cuda:0")
    capability = tuple(map(int, torch.cuda.get_device_capability(device)))
    if capability != (10, 0):
        raise common.ReplayDiagnosticError(
            f"2CTA causal diagnostic requires SM100, got {capability}"
        )

    records: dict[str, Any] = {}
    for layer in common.SELECTED_LAYERS:
        sources = {
            kind: _load_capture_kind(capture, layer, kind)
            for kind in ("real", "random")
        }
        for definition in common.case_definitions():
            case = str(definition["name"])
            source = sources[str(definition["source"])]
            inputs = common.slice_case(source, definition)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            started = time.monotonic()
            outcome, input_layouts = run_attention_backward(arm, layer, inputs, device)
            elapsed = time.monotonic() - started
            records[f"layer{layer:02d}/{case}"] = _write_outcome(
                root,
                arm,
                layer,
                case,
                inputs,
                outcome,
                input_layouts,
                definition["source_indices"],
                elapsed_seconds=elapsed,
                peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
            )
            del inputs, outcome
        del sources

    expected_keys = {
        f"layer{layer:02d}/{definition['name']}"
        for layer in common.SELECTED_LAYERS
        for definition in common.case_definitions()
    }
    if set(records) != expected_keys:
        raise common.ReplayDiagnosticError("replay outcome coverage changed")
    fa4_runtime: dict[str, Any] | None = None
    if arm in {"fa4_default", "fa4_no2cta"}:
        from flash_attn.cute import utils as fa4_utils

        if not str(torch.version.cuda).startswith("12."):
            raise common.ReplayDiagnosticError(
                f"expected CUDA 12.x FA4 runtime, got {torch.version.cuda}"
            )
        fa4_runtime = {
            "torch_cuda_version": str(torch.version.cuda),
            "fa_disable_2cta_import_flag": bool(fa4_utils._fa_disable_2cta_enabled),
            "fa_disable_2cta_cuda12_forward_flag": bool(
                fa4_utils._fa_disable_2cta_cuda12
            ),
            "expected_forward_2cta": False,
            "expected_backward_2cta": arm == "fa4_default",
            "reason": (
                "CUDA12 auto-disables 2CTA forward; FA_DISABLE_2CTA controls backward"
            ),
        }
        if (
            fa4_runtime["fa_disable_2cta_import_flag"]
            != (arm == "fa4_no2cta")
            or fa4_runtime["fa_disable_2cta_cuda12_forward_flag"] is not True
        ):
            raise common.ReplayDiagnosticError("FA4 2CTA runtime flags changed")
    receipt = {
        "schema_version": 1,
        "status": "completed",
        "kind": "qwen3_4b_true_attention_local_replay_arm",
        "plan_fingerprint": plan_fingerprint,
        "arm": arm,
        "fa_disable_2cta": expected_disable,
        "fa4_runtime": fa4_runtime,
        "expected_forward_2cta": False if fa4_runtime is not None else None,
        "expected_backward_2cta": (
            arm == "fa4_default" if fa4_runtime is not None else None
        ),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "device_name": torch.cuda.get_device_name(device),
        "compute_capability": list(capability),
        "selected_layers": list(common.SELECTED_LAYERS),
        "cases": [dict(item) for item in common.case_definitions()],
        "capture_receipt": str(capture_root / "capture_receipt.json"),
        "capture_receipt_sha256": common.file_sha256(
            capture_root / "capture_receipt.json"
        ),
        "records": records,
    }
    common.atomic_json(root / "replay_receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=common.ARMS)
    args = parser.parse_args()
    root_value = os.environ.get("REALQ_TRUE_TENSOR_REPLAY_ROOT")
    capture_value = os.environ.get("REALQ_TRUE_TENSOR_CAPTURE_ROOT")
    fingerprint = os.environ.get("REALQ_TRUE_TENSOR_PLAN_FINGERPRINT")
    if not root_value or not capture_value or not fingerprint:
        raise common.ReplayDiagnosticError("replay environment is incomplete")
    root = Path(root_value).resolve()
    failure: dict[str, Any] | None = None
    try:
        execute(args.arm, root, Path(capture_value).resolve(), fingerprint)
        return 0
    except BaseException as exc:
        failure = {"kind": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
        if root.is_dir() and not (root / "replay_receipt.json").exists():
            common.atomic_json(
                root / "replay_receipt.json",
                {
                    "schema_version": 1,
                    "status": "failed",
                    "kind": "qwen3_4b_true_attention_local_replay_arm",
                    "plan_fingerprint": fingerprint,
                    "arm": args.arm,
                    "failure": failure,
                },
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
