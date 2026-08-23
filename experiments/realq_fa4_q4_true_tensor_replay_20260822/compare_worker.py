#!/usr/bin/env python3
"""Compute explicit FP32 references and exact full-tensor replay comparisons."""

from __future__ import annotations

import os
from pathlib import Path
import time
import traceback
from typing import Any, Mapping

import torch

from experiments.realq_fa4_q4_true_tensor_replay_20260822 import common
from utils.reproducibility import configure_reproducibility


def _load_capture_receipt(
    capture_root: Path, plan_fingerprint: str
) -> dict[str, Any]:
    path = capture_root / "capture_receipt.json"
    receipt = common.read_json(path)
    if (
        receipt.get("status") != "completed"
        or receipt.get("plan_fingerprint") != plan_fingerprint
        or receipt.get("selected_layers") != list(common.SELECTED_LAYERS)
        or receipt.get("production_layouts_sha256")
        != common.canonical_sha256(receipt.get("production_layouts", {}))
    ):
        raise common.ReplayDiagnosticError("comparison capture receipt changed")
    return receipt


def _load_capture_kind(
    receipt: Mapping[str, Any], layer: int, kind: str
) -> dict[str, torch.Tensor]:
    record = receipt["artifacts"][str(layer)][kind]
    path = Path(str(record["path"]))
    if not path.is_file() or common.file_sha256(path) != record["serialization_sha256"]:
        raise common.ReplayDiagnosticError(f"unbound capture tensor: {path}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise common.ReplayDiagnosticError("capture tensor archive changed")
    common.validate_archive_qkvd(value)
    if common.named_tensors_semantic_sha256(value) != record["semantic_sha256"]:
        raise common.ReplayDiagnosticError("capture tensor semantic hash changed")
    if (
        common.named_tensors_layout_sha256(value)
        != record["layout_semantic_sha256"]
    ):
        raise common.ReplayDiagnosticError("capture tensor layout hash changed")
    if {
        name: common.tensor_contract(tensor) for name, tensor in sorted(value.items())
    } != record.get("tensors"):
        raise common.ReplayDiagnosticError("capture tensor contracts changed")
    return value


def _load_arm_receipts(
    output_root: Path, plan_fingerprint: str
) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for arm in common.ARMS:
        path = output_root / arm / "replay" / "replay_receipt.json"
        receipt = common.read_json(path)
        expected_disable = "1" if arm == "fa4_no2cta" else "0"
        expected_forward = False if arm.startswith("fa4_") else None
        expected_backward = (
            arm == "fa4_default" if arm.startswith("fa4_") else None
        )
        if (
            receipt.get("status") != "completed"
            or receipt.get("plan_fingerprint") != plan_fingerprint
            or receipt.get("arm") != arm
            or receipt.get("fa_disable_2cta") != expected_disable
            or receipt.get("compute_capability") != [10, 0]
            or receipt.get("deterministic_algorithms") is not True
            or receipt.get("matmul_tf32") is not False
            or receipt.get("expected_forward_2cta") is not expected_forward
            or receipt.get("expected_backward_2cta") is not expected_backward
        ):
            raise common.ReplayDiagnosticError(f"unbound replay receipt: {arm}")
        runtime = receipt.get("fa4_runtime")
        if arm.startswith("fa4_") and (
            not isinstance(runtime, dict)
            or not str(runtime.get("torch_cuda_version", "")).startswith("12.")
            or runtime.get("fa_disable_2cta_cuda12_forward_flag") is not True
            or runtime.get("expected_forward_2cta") is not False
            or runtime.get("expected_backward_2cta") is not expected_backward
        ):
            raise common.ReplayDiagnosticError(f"FA4 runtime contract changed: {arm}")
        receipts[arm] = receipt
    return receipts


def _load_arm_outcome(
    receipt: Mapping[str, Any],
    layer: int,
    case: str,
    expected_input_sha: str,
    expected_input_layout_sha: str,
    expected_source_indices: list[int],
    expected_kernel_layout_sha: str,
    expected_kernel_layout: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    record = receipt["records"][f"layer{layer:02d}/{case}"]
    path = Path(str(record["path"]))
    if (
        record.get("input_semantic_sha256") != expected_input_sha
        or record.get("input_archive_layout_semantic_sha256")
        != expected_input_layout_sha
        or record.get("source_indices") != expected_source_indices
        or record.get("input_layouts", {}).get("kernel_layout_semantic_sha256")
        != expected_kernel_layout_sha
        or record.get("input_layouts", {}).get("kernel") != expected_kernel_layout
        or not path.is_file()
        or common.file_sha256(path) != record.get("serialization_sha256")
    ):
        raise common.ReplayDiagnosticError(
            f"unbound replay outcome: {receipt['arm']}/{layer}/{case}"
        )
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise common.ReplayDiagnosticError("replay outcome archive changed")
    batch = 1 if case.startswith("real_b1_") else common.BATCH_SIZE
    common.validate_outcome(value, batch=batch)
    if common.named_tensors_semantic_sha256(value) != record["semantic_sha256"]:
        raise common.ReplayDiagnosticError("replay outcome semantic hash changed")
    if (
        common.named_tensors_layout_sha256(value)
        != record["layout_semantic_sha256"]
    ):
        raise common.ReplayDiagnosticError("replay outcome layout hash changed")
    return value


def run_fp32_reference(
    tensors: Mapping[str, torch.Tensor], device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    batch = int(tensors["q"].shape[0])
    common.validate_archive_qkvd(tensors, batch=batch)
    archive_gpu = {
        name: value.to(device=device, memory_format=torch.preserve_format)
        for name, value in tensors.items()
    }
    kernel_bf16 = common.reconstruct_kernel_qkvd(archive_gpu, batch=batch)
    kernel_fp32 = {
        name: value.to(dtype=torch.float32, memory_format=torch.preserve_format)
        for name, value in kernel_bf16.items()
    }
    for name, value in kernel_fp32.items():
        source_layout = common.tensor_layout(kernel_bf16[name])
        target_layout = common.tensor_layout(value)
        for key in ("shape", "stride", "storage_offset", "is_contiguous"):
            if source_layout[key] != target_layout[key]:
                raise common.ReplayDiagnosticError(
                    f"FP32 reference cast changed {name} {key}: "
                    f"{source_layout} -> {target_layout}"
                )
    q = kernel_fp32["q"].detach().requires_grad_(True)
    k = kernel_fp32["k"].detach().requires_grad_(True)
    v = kernel_fp32["v"].detach().requires_grad_(True)
    dout = kernel_fp32["dout"]
    output = common.explicit_fp32_attention(q, k, v, scaling=common.SCALING)
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
        "kernel_bf16": {
            name: {
                "interface": common.tensor_layout(value),
                **(
                    {"adapter_native": common.tensor_layout(value.transpose(1, 2))}
                    if name in {"q", "k", "v"}
                    else {}
                ),
            }
            for name, value in kernel_bf16.items()
        },
        "kernel_fp32": {
            name: {
                "interface": common.tensor_layout(value),
                **(
                    {"adapter_native": common.tensor_layout(value.transpose(1, 2))}
                    if name in {"q", "k", "v"}
                    else {}
                ),
            }
            for name, value in kernel_fp32.items()
        },
        "archive_layout_semantic_sha256": common.named_tensors_layout_sha256(tensors),
        "kernel_bf16_layout_semantic_sha256": common.named_tensors_layout_sha256(
            kernel_bf16
        ),
        "kernel_fp32_layout_semantic_sha256": common.named_tensors_layout_sha256(
            kernel_fp32
        ),
    }


def _compare_outcomes(
    outcomes: Mapping[str, Mapping[str, torch.Tensor]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for left, right in common.PAIRWISE_COMPARISONS:
        pair = f"{left}__vs__{right}"
        result[pair] = {
            "left": left,
            "right": right,
            "fields": {
                field: common.full_tensor_metrics(
                    outcomes[left][field], outcomes[right][field]
                )
                for field in common.OUTPUT_FIELDS
            },
        }
    return result


def _batch_stitch_metrics(
    b4: Mapping[str, torch.Tensor], pieces: list[Mapping[str, torch.Tensor]]
) -> dict[str, Any]:
    if len(pieces) != common.BATCH_SIZE:
        raise common.ReplayDiagnosticError("B=1 stitch requires all four samples")
    return {
        field: common.full_tensor_metrics(
            b4[field], torch.cat([piece[field] for piece in pieces], dim=0)
        )
        for field in common.OUTPUT_FIELDS
    }


def _causal_case_metrics(
    retained: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
) -> dict[str, Any]:
    """Directly test batch-slot, permutation, and companion-sample hypotheses."""

    permutation = torch.tensor([3, 0, 1, 2], dtype=torch.int64)
    permutation_metrics: dict[str, Any] = {}
    repeated_metrics: dict[str, Any] = {}
    for name, values in retained.items():
        required = {
            "real_b4",
            "real_b1_s3",
            "real_b4_perm_s3_to_s0",
            "real_b4_s3_repeated",
        }
        missing = sorted(required - set(values))
        if missing:
            raise common.ReplayDiagnosticError(
                f"causal cases missing for {name}: {missing}"
            )
        permutation_metrics[name] = {
            "comparison": (
                "real_b4 outcomes reordered by [3,0,1,2] vs an independent "
                "real_b4_perm_s3_to_s0 kernel call"
            ),
            "source_indices": [3, 0, 1, 2],
            "fields": {
                field: common.full_tensor_metrics(
                    values["real_b4"][field].index_select(0, permutation),
                    values["real_b4_perm_s3_to_s0"][field],
                )
                for field in common.OUTPUT_FIELDS
            },
        }
        repeated_metrics[name] = {
            "comparison": (
                "each slot of one [s3,s3,s3,s3] B=4 call vs an independent "
                "real_b1_s3 kernel call"
            ),
            "source_indices": [3, 3, 3, 3],
            "slots": {
                str(slot): {
                    "fields": {
                        field: common.full_tensor_metrics(
                            values["real_b4_s3_repeated"][field][slot : slot + 1],
                            values["real_b1_s3"][field],
                        )
                        for field in common.OUTPUT_FIELDS
                    }
                }
                for slot in range(common.BATCH_SIZE)
            },
        }

    reference = retained["fp32_reference"]["real_b4"]
    per_sample: dict[str, Any] = {}
    for arm in common.ARMS:
        candidate = retained[arm]["real_b4"]
        per_sample[arm] = {
            "comparison": "one B=4 backend call vs FP32 reference, split by sample",
            "samples": {
                str(sample): {
                    "fields": {
                        field: common.full_tensor_metrics(
                            reference[field][sample : sample + 1],
                            candidate[field][sample : sample + 1],
                        )
                        for field in common.OUTPUT_FIELDS
                    }
                }
                for sample in range(common.BATCH_SIZE)
            },
        }

    return {
        "permutation_equivariance": permutation_metrics,
        "repeated_s3_vs_independent_b1_s3": repeated_metrics,
        "real_b4_per_sample_vs_fp32": per_sample,
    }


def execute(
    output_root: Path,
    root: Path,
    capture_root: Path,
    plan_fingerprint: str,
) -> dict[str, Any]:
    if root.exists() or root.is_symlink():
        raise common.ReplayDiagnosticError(f"comparison root is not fresh: {root}")
    if os.environ.get("FA_DISABLE_2CTA") != "0":
        raise common.ReplayDiagnosticError("FP32 comparator requires FA_DISABLE_2CTA=0")
    root.mkdir(parents=True)
    capture = _load_capture_receipt(capture_root, plan_fingerprint)
    arm_receipts = _load_arm_receipts(output_root, plan_fingerprint)
    configure_reproducibility(common.REPLAY_SEED, deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise common.ReplayDiagnosticError("comparison worker requires one visible GPU")
    device = torch.device("cuda:0")
    capability = list(map(int, torch.cuda.get_device_capability(device)))
    if capability != [10, 0]:
        raise common.ReplayDiagnosticError(f"comparison requires SM100, got {capability}")

    layers: dict[str, Any] = {}
    for layer in common.SELECTED_LAYERS:
        sources = {
            kind: _load_capture_kind(capture, layer, kind)
            for kind in ("real", "random")
        }
        case_results: dict[str, Any] = {}
        retained: dict[str, dict[str, dict[str, torch.Tensor]]] = {
            name: {} for name in ("fp32_reference", *common.ARMS)
        }
        for definition in common.case_definitions():
            case = str(definition["name"])
            inputs = common.slice_case(sources[str(definition["source"])], definition)
            input_sha = common.named_tensors_semantic_sha256(inputs)
            input_layout_sha = common.named_tensors_layout_sha256(inputs)
            expected_kernel = common.reconstruct_kernel_qkvd(
                inputs, batch=int(definition["batch"])
            )
            expected_kernel_layout_sha = common.named_tensors_layout_sha256(
                expected_kernel
            )
            expected_kernel_layout = {
                name: {
                    "interface": common.tensor_layout(value),
                    **(
                        {"adapter_native": common.tensor_layout(value.transpose(1, 2))}
                        if name in {"q", "k", "v"}
                        else {}
                    ),
                }
                for name, value in expected_kernel.items()
            }
            del expected_kernel
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            started = time.monotonic()
            reference, reference_input_layouts = run_fp32_reference(inputs, device)
            if (
                reference_input_layouts.get("kernel_bf16")
                != expected_kernel_layout
                or reference_input_layouts.get("kernel_bf16_layout_semantic_sha256")
                != expected_kernel_layout_sha
            ):
                raise common.ReplayDiagnosticError(
                    f"FP32 reference kernel layout changed: layer={layer} case={case}"
                )
            reference_elapsed = time.monotonic() - started
            outcomes: dict[str, Mapping[str, torch.Tensor]] = {
                "fp32_reference": reference
            }
            for arm in common.ARMS:
                outcomes[arm] = _load_arm_outcome(
                    arm_receipts[arm],
                    layer,
                    case,
                    input_sha,
                    input_layout_sha,
                    list(map(int, definition["source_indices"])),
                    expected_kernel_layout_sha,
                    expected_kernel_layout,
                )
            case_results[case] = {
                "definition": dict(definition),
                "input_semantic_sha256": input_sha,
                "input_archive_layout_semantic_sha256": input_layout_sha,
                "source_indices": list(map(int, definition["source_indices"])),
                "fp32_reference": {
                    "implementation": "explicit_fp32_causal_gqa_softmax",
                    "elapsed_seconds": reference_elapsed,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                    "input_layouts": reference_input_layouts,
                },
                "comparisons": _compare_outcomes(outcomes),
            }
            if case.startswith("real_"):
                for name, outcome in outcomes.items():
                    retained[name][case] = dict(outcome)
            del outcomes, reference, inputs

        batching_effects: dict[str, Any] = {}
        for name, values in retained.items():
            batching_effects[name] = {
                "comparison": "one B=4 call vs concatenated four independent B=1 calls",
                "fields": _batch_stitch_metrics(
                    values["real_b4"],
                    [values[f"real_b1_s{sample}"] for sample in range(common.BATCH_SIZE)],
                ),
            }
        layers[str(layer)] = {
            "cases": case_results,
            "batching_effects": batching_effects,
            "causal_case_metrics": _causal_case_metrics(retained),
        }
        del retained, sources

    receipt = {
        "schema_version": 1,
        "status": "completed",
        "kind": "qwen3_4b_true_attention_full_tensor_comparison",
        "plan_fingerprint": plan_fingerprint,
        "selected_layers": list(common.SELECTED_LAYERS),
        "arms": ["fp32_reference", *common.ARMS],
        "pairwise_comparisons": [list(pair) for pair in common.PAIRWISE_COMPARISONS],
        "fa4_2cta_contract": {
            arm: arm_receipts[arm]["fa4_runtime"]
            for arm in ("fa4_default", "fa4_no2cta")
        },
        "metrics_scope": "all tensor elements; no token/feature sampling",
        "fp32_reference": {
            "qkv_input": "captured BF16 values cast exactly to FP32",
            "dout_input": "captured BF16 dO cast exactly to FP32",
            "gqa": "K/V repeat_interleave to four Q heads per KV head",
            "mask": "explicit upper-triangular causal -inf mask",
            "softmax": "torch.float32",
            "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        },
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "device_name": torch.cuda.get_device_name(device),
        "compute_capability": capability,
        "capture_receipt": str(capture_root / "capture_receipt.json"),
        "capture_receipt_sha256": common.file_sha256(
            capture_root / "capture_receipt.json"
        ),
        "replay_receipts": {
            arm: {
                "path": str(output_root / arm / "replay" / "replay_receipt.json"),
                "sha256": common.file_sha256(
                    output_root / arm / "replay" / "replay_receipt.json"
                ),
            }
            for arm in common.ARMS
        },
        "layers": layers,
    }
    common.atomic_json(root / "comparison_receipt.json", receipt)
    return receipt


def main() -> int:
    output_value = os.environ.get("REALQ_TRUE_TENSOR_OUTPUT_ROOT")
    root_value = os.environ.get("REALQ_TRUE_TENSOR_COMPARISON_ROOT")
    capture_value = os.environ.get("REALQ_TRUE_TENSOR_CAPTURE_ROOT")
    fingerprint = os.environ.get("REALQ_TRUE_TENSOR_PLAN_FINGERPRINT")
    if not output_value or not root_value or not capture_value or not fingerprint:
        raise common.ReplayDiagnosticError("comparison environment is incomplete")
    root = Path(root_value).resolve()
    try:
        execute(
            Path(output_value).resolve(),
            root,
            Path(capture_value).resolve(),
            fingerprint,
        )
        return 0
    except BaseException as exc:
        traceback.print_exc()
        if root.is_dir() and not (root / "comparison_receipt.json").exists():
            common.atomic_json(
                root / "comparison_receipt.json",
                {
                    "schema_version": 1,
                    "status": "failed",
                    "kind": "qwen3_4b_true_attention_full_tensor_comparison",
                    "plan_fingerprint": fingerprint,
                    "failure": {"kind": type(exc).__name__, "message": str(exc)},
                },
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
