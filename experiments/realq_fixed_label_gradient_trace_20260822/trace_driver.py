#!/usr/bin/env python3
"""Trace signed first-batch activations and gradients during REAL-Q Stage 0.

The wrapper replays the audited native-SDPA categorical labels, attaches
read-only hooks after rotation at the Stage-0 boundary, samples a small fixed
set of token/feature positions from each transformer block and q/k/v/o site,
then delegates all numerical work to the production precompute function.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import traceback
from typing import Any, Mapping, Sequence

import torch

from realq import precompute
from realq.precompute import static_e2e
from experiments.realq_fixed_label_backend_diagnostic_20260822 import (
    fixed_label_driver as fixed,
)


TOKEN_INDICES = (0, 1, 2, 3, 7, 31, 127, 511, 1023, 1535, 2046, 2047)
SITE_NAMES = (
    "block_output",
    "q_proj_output",
    "k_proj_output",
    "v_proj_output",
    "o_proj_input",
    "o_proj_output",
)


class GradientTraceError(RuntimeError):
    """The signed trace or fixed-label contract was violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GradientTraceError(f"JSON root is not an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(
            dict(value),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _tensor_from_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise GradientTraceError(f"unsupported traced output type: {type(output)!r}")


def _feature_indices(width: int) -> tuple[int, ...]:
    if width < 1:
        raise GradientTraceError(f"invalid traced feature width: {width}")
    count = min(64, width)
    if count == 1:
        return (0,)
    return tuple((index * (width - 1)) // (count - 1) for index in range(count))


def _sample_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    if tensor.ndim != 3:
        raise GradientTraceError(f"expected [B,T,D] tensor, got {tuple(tensor.shape)}")
    if tensor.shape[1] != 2048:
        raise GradientTraceError(f"sequence length changed: {tuple(tensor.shape)}")
    tokens = torch.tensor(TOKEN_INDICES, device=tensor.device, dtype=torch.long)
    feature_values = _feature_indices(int(tensor.shape[2]))
    features = torch.tensor(feature_values, device=tensor.device, dtype=torch.long)
    sample = (
        tensor.detach()
        .index_select(1, tokens)
        .index_select(2, features)
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
    )
    metadata = {
        "tensor_shape": list(map(int, tensor.shape)),
        "tensor_dtype": str(tensor.dtype),
        "sample_shape": list(map(int, sample.shape)),
        "token_indices": list(TOKEN_INDICES),
        "feature_indices": list(feature_values),
    }
    return sample, metadata


class TraceManager:
    """Capture one signed forward sample and its gradient at every site."""

    def __init__(self, analyzer: Any):
        self.analyzer = analyzer
        self.samples: dict[str, torch.Tensor] = {}
        self.metadata: dict[str, dict[str, Any]] = {}
        self.handles: list[Any] = []

    def _capture_forward(self, key: str, tensor: torch.Tensor) -> None:
        forward_key = f"{key}/forward"
        gradient_key = f"{key}/gradient"
        if forward_key in self.samples:
            return
        sample, metadata = _sample_tensor(tensor)
        self.samples[forward_key] = sample
        self.metadata[forward_key] = metadata
        if not tensor.requires_grad:
            raise GradientTraceError(f"traced tensor has no gradient: {key}")

        def capture_gradient(gradient: torch.Tensor) -> torch.Tensor:
            if gradient_key in self.samples:
                raise GradientTraceError(f"gradient hook fired twice: {key}")
            grad_sample, grad_metadata = _sample_tensor(gradient)
            self.samples[gradient_key] = grad_sample
            self.metadata[gradient_key] = grad_metadata
            return gradient

        tensor.register_hook(capture_gradient)

    def _forward_hook(self, key: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            self._capture_forward(key, _tensor_from_output(output))

        return hook

    def _forward_pre_hook(self, key: str):
        def hook(_module: Any, inputs: Any) -> None:
            if not isinstance(inputs, tuple) or not inputs:
                raise GradientTraceError(f"traced module has no input: {key}")
            self._capture_forward(key, _tensor_from_output(inputs[0]))

        return hook

    def attach(self) -> None:
        layers = self.analyzer.get_layers()
        if len(layers) != 36:
            raise GradientTraceError(f"expected 36 layers, got {len(layers)}")
        for layer_index, layer in enumerate(layers):
            modules = static_e2e._find_layer_modules(layer)
            prefix = f"layer{layer_index:02d}"
            self.handles.append(
                layer.register_forward_hook(
                    self._forward_hook(f"{prefix}/block_output")
                )
            )
            for short_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                module = modules[f"self_attn.{short_name}"]
                self.handles.append(
                    module.register_forward_hook(
                        self._forward_hook(f"{prefix}/{short_name}_output")
                    )
                )
            o_proj = modules["self_attn.o_proj"]
            self.handles.append(
                o_proj.register_forward_pre_hook(
                    self._forward_pre_hook(f"{prefix}/o_proj_input")
                )
            )

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def validate(self) -> None:
        expected = {
            f"layer{layer:02d}/{site}/{phase}"
            for layer in range(36)
            for site in SITE_NAMES
            for phase in ("forward", "gradient")
        }
        if set(self.samples) != expected or set(self.metadata) != expected:
            missing = sorted(expected - set(self.samples))
            extra = sorted(set(self.samples) - expected)
            raise GradientTraceError(
                f"signed trace coverage changed: missing={missing[:8]}, extra={extra[:8]}"
            )
        for key in expected:
            if not torch.isfinite(self.samples[key]).all().item():
                raise GradientTraceError(f"non-finite signed trace: {key}")


def _atomic_trace(path: Path, samples: Mapping[str, torch.Tensor]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        torch.save(dict(samples), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    labels_value = os.environ.get("REALQ_FIXED_LABELS_PATH")
    labels_sha256 = os.environ.get("REALQ_FIXED_LABELS_SHA256")
    summary_value = os.environ.get("REALQ_FIXED_LABELS_SUMMARY")
    audit_value = os.environ.get("REALQ_SIGNED_TRACE_ROOT")
    backend = os.environ.get("REALQ_SIGNED_TRACE_BACKEND")
    if (
        not labels_value
        or not labels_sha256
        or not summary_value
        or not audit_value
        or backend not in {"sdpa", "flash_attention_4"}
    ):
        raise GradientTraceError("signed-trace environment is incomplete")

    labels_path = Path(labels_value).resolve()
    source_summary_path = Path(summary_value).resolve()
    audit_root = Path(audit_value).resolve()
    if audit_root.exists() or audit_root.is_symlink():
        raise GradientTraceError(f"signed-trace root is not fresh: {audit_root}")
    if _sha256(labels_path) != labels_sha256:
        raise GradientTraceError("fixed labels changed")
    source_summary = _read(source_summary_path)
    records = source_summary.get("records")
    if (
        source_summary.get("status") != "completed"
        or source_summary.get("backend") != "sdpa"
        or source_summary.get("labels_sha256") != labels_sha256
        or not isinstance(records, list)
        or len(records) != 64
    ):
        raise GradientTraceError("fixed-label source summary changed")
    fixed_labels = fixed._load_labels(labels_path)
    audit_root.mkdir(parents=True)

    original_labels = static_e2e.deterministic_categorical_labels
    original_run = precompute.run
    label_call_index = 0
    trace_manager: TraceManager | None = None
    precompute_calls = 0

    def injected_labels(
        logits: torch.Tensor,
        global_sample_indices: Sequence[int],
        *,
        base_seed: int = 0,
    ) -> torch.Tensor:
        nonlocal label_call_index
        if label_call_index >= len(records):
            raise GradientTraceError("production made too many label calls")
        labels = fixed._select_labels(
            fixed_labels,
            records[label_call_index],
            logits,
            global_sample_indices,
            base_seed=base_seed,
        )
        label_call_index += 1
        return labels

    def traced_run(cfg: Any, analyzer: Any):
        nonlocal trace_manager, precompute_calls
        precompute_calls += 1
        if precompute_calls != 1:
            raise GradientTraceError("production precompute ran more than once")
        trace_manager = TraceManager(analyzer)
        trace_manager.attach()
        try:
            result = original_run(cfg, analyzer)
        finally:
            trace_manager.remove()
        trace_manager.validate()
        return result

    static_e2e.deterministic_categorical_labels = injected_labels
    precompute.run = traced_run
    exit_code = 1
    failure: dict[str, Any] | None = None
    try:
        runpy.run_module("realq.ptq", run_name="__main__", alter_sys=True)
        exit_code = 0
    except SystemExit as exc:
        exit_code = 0 if exc.code is None else int(exc.code)
        if exit_code:
            failure = {"kind": "SystemExit", "code": exit_code}
    except BaseException as exc:
        failure = {"kind": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
    finally:
        static_e2e.deterministic_categorical_labels = original_labels
        precompute.run = original_run

    completed = (
        exit_code == 0
        and label_call_index == len(records)
        and precompute_calls == 1
        and trace_manager is not None
    )
    trace_path = audit_root / "signed_trace.pt"
    if completed and trace_manager is not None:
        trace_manager.validate()
        _atomic_trace(trace_path, trace_manager.samples)
    receipt = {
        "schema_version": 1,
        "status": "completed" if completed else "failed",
        "kind": "realq_fixed_label_signed_forward_backward_trace",
        "backend": backend,
        "labels": str(labels_path),
        "labels_sha256": labels_sha256,
        "source_summary": str(source_summary_path),
        "source_summary_sha256": _sha256(source_summary_path),
        "expected_label_calls": len(records),
        "completed_label_calls": label_call_index,
        "precompute_calls": precompute_calls,
        "production_label_sampler_called": False,
        "trace": str(trace_path) if completed else None,
        "trace_sha256": _sha256(trace_path) if completed else None,
        "trace_tensors": len(trace_manager.samples) if trace_manager else 0,
        "trace_metadata": trace_manager.metadata if trace_manager else {},
        "exit_code": exit_code,
        "failure": failure,
    }
    _atomic_json(audit_root / "trace_receipt.json", receipt)
    if not completed:
        raise GradientTraceError("signed Stage-0 trace did not complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
