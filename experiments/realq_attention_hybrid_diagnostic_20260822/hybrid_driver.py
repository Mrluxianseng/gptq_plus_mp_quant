#!/usr/bin/env python3
"""Run REAL-Q Stage 0 with independently selected attention value/Jacobian.

This is a diagnostic wrapper.  It leaves the production attention adapter and
Stage-0 implementation untouched, freezes the categorical labels captured by
the native deterministic-SDPA producer, and replaces only the function that
Transformers registers for ``flash_attention_4`` inside this child process.
"""

from __future__ import annotations

from array import array
import hashlib
import json
import os
from pathlib import Path
import runpy
import traceback
from typing import Any, Mapping, Sequence

import torch

from realq import attention as realq_attention
from realq.precompute import static_e2e


EXPECTED_LABELS = 256 * 2048
EXPECTED_LABEL_CALLS = 64
EXPECTED_ATTENTION_CALLS = EXPECTED_LABEL_CALLS * 36
ARMS = {
    "fa4_forward_sdpa_backward": {
        "forward_backend": "flash_attention_4",
        "backward_backend": "sdpa_math",
    },
    "sdpa_forward_fa4_backward": {
        "forward_backend": "sdpa_math",
        "backward_backend": "flash_attention_4",
    },
}
_FA4_FORWARD = realq_attention.flash_attention_4_forward


class HybridDiagnosticError(RuntimeError):
    """A hybrid attention or fixed-label contract was violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HybridDiagnosticError(f"JSON root is not an object: {path}")
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


def _load_labels(path: Path) -> torch.Tensor:
    if path.stat().st_size != EXPECTED_LABELS * 8:
        raise HybridDiagnosticError("fixed-label byte length changed")
    values = array("q")
    with path.open("rb") as handle:
        values.fromfile(handle, EXPECTED_LABELS)
    if len(values) != EXPECTED_LABELS:
        raise HybridDiagnosticError("fixed-label element count changed")
    return torch.tensor(values, dtype=torch.long).reshape(256, 2048)


class _ForwardValueBackwardSource(torch.autograd.Function):
    """Return the first tensor bit-exactly and route gradients to the second."""

    @staticmethod
    def forward(
        ctx: Any,
        value_source: torch.Tensor,
        gradient_source: torch.Tensor,
    ) -> torch.Tensor:
        if (
            value_source.shape != gradient_source.shape
            or value_source.dtype != gradient_source.dtype
            or value_source.device != gradient_source.device
        ):
            raise HybridDiagnosticError("hybrid attention branch metadata differs")
        if value_source.requires_grad:
            raise HybridDiagnosticError("hybrid value-only branch retained a graph")
        if not gradient_source.requires_grad:
            raise HybridDiagnosticError("hybrid gradient branch has no graph")
        return value_source

    @staticmethod
    def backward(
        ctx: Any,
        gradient: torch.Tensor,
    ) -> tuple[None, torch.Tensor]:
        return None, gradient


def _sdpa_math_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )


def _hybrid_forward(arm: str, counter: dict[str, int]):
    if arm not in ARMS:
        raise HybridDiagnosticError(f"unknown hybrid arm: {arm}")

    def forward(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        counter["attention_calls"] += 1
        common = dict(
            module=module,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )
        if arm == "fa4_forward_sdpa_backward":
            with torch.no_grad():
                value_output, _ = _FA4_FORWARD(**common)
            gradient_output, _ = _sdpa_math_forward(**common)
        else:
            with torch.no_grad():
                value_output, _ = _sdpa_math_forward(**common)
            gradient_output, _ = _FA4_FORWARD(**common)
        output = _ForwardValueBackwardSource.apply(
            value_output,
            gradient_output,
        )
        return output, None

    return forward


def _selected_labels(
    fixed: torch.Tensor,
    record: Mapping[str, Any],
    logits: torch.Tensor,
    global_sample_indices: Sequence[int],
    *,
    base_seed: int,
) -> torch.Tensor:
    indices = list(map(int, global_sample_indices))
    expected_shape = [len(indices), 2048]
    if (
        indices != record.get("global_sample_indices")
        or int(base_seed) != record.get("base_seed")
        or list(map(int, record.get("shape", []))) != expected_shape
        or list(logits.shape[:2]) != expected_shape
        or int(record.get("offset_labels", -1)) != indices[0] * 2048
    ):
        raise HybridDiagnosticError("production label invocation changed")
    return fixed[indices].to(device=logits.device, dtype=torch.long)


def main() -> int:
    labels_value = os.environ.get("REALQ_FIXED_LABELS_PATH")
    labels_sha256 = os.environ.get("REALQ_FIXED_LABELS_SHA256")
    summary_value = os.environ.get("REALQ_FIXED_LABELS_SUMMARY")
    audit_value = os.environ.get("REALQ_HYBRID_AUDIT_ROOT")
    arm = os.environ.get("REALQ_HYBRID_ARM")
    if (
        not labels_value
        or not labels_sha256
        or not summary_value
        or not audit_value
        or arm not in ARMS
    ):
        raise HybridDiagnosticError("hybrid diagnostic environment is incomplete")

    labels_path = Path(labels_value).resolve()
    summary_path = Path(summary_value).resolve()
    audit_root = Path(audit_value).resolve()
    if audit_root.exists() or audit_root.is_symlink():
        raise HybridDiagnosticError(f"hybrid audit root is not fresh: {audit_root}")
    if _sha256(labels_path) != labels_sha256:
        raise HybridDiagnosticError("fixed labels changed")
    summary = _read(summary_path)
    records = summary.get("records")
    if (
        summary.get("status") != "completed"
        or summary.get("backend") != "sdpa"
        or summary.get("labels_sha256") != labels_sha256
        or summary.get("total_labels") != EXPECTED_LABELS
        or not isinstance(records, list)
        or len(records) != EXPECTED_LABEL_CALLS
    ):
        raise HybridDiagnosticError("fixed-label source summary changed")
    fixed = _load_labels(labels_path)
    audit_root.mkdir(parents=True)

    original_labels = static_e2e.deterministic_categorical_labels
    original_fa4 = realq_attention.flash_attention_4_forward
    label_calls = 0
    counter = {"attention_calls": 0}

    def injected(
        logits: torch.Tensor,
        global_sample_indices: Sequence[int],
        *,
        base_seed: int = 0,
    ) -> torch.Tensor:
        nonlocal label_calls
        if label_calls >= len(records):
            raise HybridDiagnosticError("production made too many label calls")
        labels = _selected_labels(
            fixed,
            records[label_calls],
            logits,
            global_sample_indices,
            base_seed=base_seed,
        )
        label_calls += 1
        return labels

    static_e2e.deterministic_categorical_labels = injected
    realq_attention.flash_attention_4_forward = _hybrid_forward(arm, counter)
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
        realq_attention.flash_attention_4_forward = original_fa4

    completed = (
        exit_code == 0
        and label_calls == EXPECTED_LABEL_CALLS
        and counter["attention_calls"] == EXPECTED_ATTENTION_CALLS
    )
    definition = ARMS[arm]
    result = {
        "schema_version": 1,
        "status": "completed" if completed else "failed",
        "kind": "realq_attention_forward_backward_hybrid_stage0",
        "arm": arm,
        **definition,
        "labels": str(labels_path),
        "labels_sha256": labels_sha256,
        "source_summary": str(summary_path),
        "source_summary_sha256": _sha256(summary_path),
        "expected_label_calls": EXPECTED_LABEL_CALLS,
        "completed_label_calls": label_calls,
        "expected_attention_calls": EXPECTED_ATTENTION_CALLS,
        "completed_attention_calls": counter["attention_calls"],
        "value_branch_no_grad": True,
        "production_label_sampler_called": False,
        "production_source_modified": False,
        "exit_code": exit_code,
        "failure": failure,
    }
    _atomic_json(audit_root / "hybrid_receipt.json", result)
    if not completed:
        raise HybridDiagnosticError("hybrid REAL-Q Stage 0 did not complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
