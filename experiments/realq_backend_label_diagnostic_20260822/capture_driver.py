#!/usr/bin/env python3
"""Capture the exact categorical labels used by REAL-Q Stage-0.

This wrapper changes no model or loss input.  It delegates label generation to
the production implementation, copies the resulting int64 labels to a small
raw audit artifact, and then runs ``realq.ptq`` unchanged in the same process.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import traceback
from typing import Any, Sequence

import torch

from realq.precompute import static_e2e


class LabelCaptureError(RuntimeError):
    """The label-capture contract was violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    root_value = os.environ.get("REALQ_LABEL_CAPTURE_ROOT")
    backend = os.environ.get("REALQ_LABEL_CAPTURE_BACKEND")
    if not root_value or backend not in {"sdpa", "flash_attention_4"}:
        raise LabelCaptureError("label-capture environment is incomplete")
    root = Path(root_value).resolve()
    if root.exists() or root.is_symlink():
        raise LabelCaptureError(f"capture root is not fresh: {root}")
    root.mkdir(parents=True)
    raw_path = root / "labels.i64"
    records: list[dict[str, Any]] = []
    total_labels = 0
    original = static_e2e.deterministic_categorical_labels

    with raw_path.open("xb", buffering=0) as raw:

        def captured(
            logits: torch.Tensor,
            global_sample_indices: Sequence[int],
            *,
            base_seed: int = 0,
        ) -> torch.Tensor:
            nonlocal total_labels
            labels = original(
                logits,
                global_sample_indices,
                base_seed=base_seed,
            )
            if labels.dtype != torch.long or labels.ndim != 2:
                raise LabelCaptureError(
                    f"unexpected labels: dtype={labels.dtype}, shape={labels.shape}"
                )
            cpu = labels.detach().to(device="cpu", dtype=torch.int64).contiguous()
            payload = cpu.numpy().tobytes(order="C")
            written = raw.write(payload)
            if written != len(payload):
                raise LabelCaptureError("short write while capturing labels")
            record = {
                "global_sample_indices": list(map(int, global_sample_indices)),
                "shape": list(map(int, cpu.shape)),
                "base_seed": int(base_seed),
                "offset_labels": total_labels,
            }
            records.append(record)
            total_labels += int(cpu.numel())
            return labels

        static_e2e.deterministic_categorical_labels = captured
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
            static_e2e.deterministic_categorical_labels = original
            raw.flush()
            os.fsync(raw.fileno())

    flattened_indices = [
        index
        for record in records
        for index in record["global_sample_indices"]
    ]
    expected_indices = list(range(256))
    expected_labels = 256 * 2048
    completed = (
        exit_code == 0
        and flattened_indices == expected_indices
        and total_labels == expected_labels
        and raw_path.stat().st_size == expected_labels * 8
    )
    summary = {
        "schema_version": 1,
        "status": "completed" if completed else "failed",
        "backend": backend,
        "invocations": len(records),
        "records": records,
        "total_labels": total_labels,
        "expected_labels": expected_labels,
        "labels": str(raw_path),
        "labels_size_bytes": raw_path.stat().st_size,
        "labels_sha256": _sha256(raw_path),
        "exit_code": exit_code,
        "failure": failure,
    }
    _atomic_json(root / "capture_summary.json", summary)
    if not completed:
        raise LabelCaptureError("REAL-Q label capture did not complete exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
