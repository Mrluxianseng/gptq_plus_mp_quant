#!/usr/bin/env python3
"""Run REAL-Q Stage 0 after replacing sampled labels with one frozen tensor.

This is a diagnostic wrapper, not a production algorithm.  It validates the
exact labels captured from the native deterministic-SDPA run, returns the
corresponding rows at each production label call, and otherwise delegates to
``realq.ptq`` unchanged.  Both backend arms therefore receive byte-identical
categorical targets.
"""

from __future__ import annotations

from array import array
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import traceback
from typing import Any, Mapping, Sequence

import torch

from realq.precompute import static_e2e


EXPECTED_LABELS = 256 * 2048


class FixedLabelError(RuntimeError):
    """The fixed-label injection contract was violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FixedLabelError(f"JSON root is not an object: {path}")
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
        raise FixedLabelError("fixed-label byte length changed")
    values = array("q")
    with path.open("rb") as handle:
        values.fromfile(handle, EXPECTED_LABELS)
    if len(values) != EXPECTED_LABELS:
        raise FixedLabelError("fixed-label element count changed")
    return torch.tensor(values, dtype=torch.long).reshape(256, 2048)


def _select_labels(
    fixed: torch.Tensor,
    record: Mapping[str, Any],
    logits: torch.Tensor,
    global_sample_indices: Sequence[int],
    *,
    base_seed: int,
) -> torch.Tensor:
    indices = list(map(int, global_sample_indices))
    expected_shape = list(map(int, record.get("shape", [])))
    if (
        indices != record.get("global_sample_indices")
        or int(base_seed) != record.get("base_seed")
        or expected_shape != [len(indices), 2048]
        or list(logits.shape[:2]) != expected_shape
    ):
        raise FixedLabelError(
            "production label invocation changed: "
            f"indices={indices}, shape={list(logits.shape)}, seed={base_seed}"
        )
    selected = fixed[indices]
    offset = int(record.get("offset_labels", -1))
    if offset != indices[0] * 2048:
        raise FixedLabelError("fixed-label record offset changed")
    return selected.to(device=logits.device, dtype=torch.long)


def main() -> int:
    labels_value = os.environ.get("REALQ_FIXED_LABELS_PATH")
    labels_sha256 = os.environ.get("REALQ_FIXED_LABELS_SHA256")
    summary_value = os.environ.get("REALQ_FIXED_LABELS_SUMMARY")
    audit_value = os.environ.get("REALQ_FIXED_LABEL_AUDIT_ROOT")
    backend = os.environ.get("REALQ_FIXED_LABEL_BACKEND")
    if (
        not labels_value
        or not labels_sha256
        or not summary_value
        or not audit_value
        or backend not in {"sdpa", "flash_attention_4"}
    ):
        raise FixedLabelError("fixed-label environment is incomplete")

    labels_path = Path(labels_value).resolve()
    summary_path = Path(summary_value).resolve()
    audit_root = Path(audit_value).resolve()
    if audit_root.exists() or audit_root.is_symlink():
        raise FixedLabelError(f"fixed-label audit root is not fresh: {audit_root}")
    if _sha256(labels_path) != labels_sha256:
        raise FixedLabelError("fixed-label SHA256 changed")
    source = _read(summary_path)
    records = source.get("records")
    if (
        source.get("status") != "completed"
        or source.get("backend") != "sdpa"
        or source.get("labels_sha256") != labels_sha256
        or source.get("total_labels") != EXPECTED_LABELS
        or not isinstance(records, list)
        or len(records) != 64
    ):
        raise FixedLabelError("source label-capture receipt changed")
    fixed = _load_labels(labels_path)
    audit_root.mkdir(parents=True)
    original = static_e2e.deterministic_categorical_labels
    call_index = 0

    def injected(
        logits: torch.Tensor,
        global_sample_indices: Sequence[int],
        *,
        base_seed: int = 0,
    ) -> torch.Tensor:
        nonlocal call_index
        if call_index >= len(records):
            raise FixedLabelError("production made too many label calls")
        labels = _select_labels(
            fixed,
            records[call_index],
            logits,
            global_sample_indices,
            base_seed=base_seed,
        )
        call_index += 1
        return labels

    static_e2e.deterministic_categorical_labels = injected
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

    completed = exit_code == 0 and call_index == len(records)
    result = {
        "schema_version": 1,
        "status": "completed" if completed else "failed",
        "kind": "realq_stage0_fixed_categorical_labels",
        "backend": backend,
        "labels": str(labels_path),
        "labels_sha256": labels_sha256,
        "source_summary": str(summary_path),
        "source_summary_sha256": _sha256(summary_path),
        "expected_calls": len(records),
        "completed_calls": call_index,
        "exit_code": exit_code,
        "failure": failure,
        "production_label_sampler_called": False,
    }
    _atomic_json(audit_root / "injection_receipt.json", result)
    if not completed:
        raise FixedLabelError("fixed-label REAL-Q Stage 0 did not complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
