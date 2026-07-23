"""Machine-readable per-step traces used for legacy/refactor alignment."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from utils import dist_utils


TRACE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RefreshStep:
    """One block-GD loss observation.

    The identity tuple intentionally excludes the implementation and run ID so
    two traces can be joined without relying on record order.
    """

    layer: int
    module: str
    block: int
    col_start: int
    col_end: int
    adam_step: int
    loss: float
    loss_current: float | None = None
    loss_next: float | None = None
    slide_alpha: float | None = None
    sample_indices: tuple[int, ...] = ()

    @property
    def identity(self) -> tuple[int, str, int, int, int, int]:
        return (
            self.layer,
            self.module,
            self.block,
            self.col_start,
            self.col_end,
            self.adam_step,
        )


class RefreshTraceWriter:
    """Append-only JSONL writer; only distributed rank zero writes."""

    def __init__(
        self,
        path: str | None,
        *,
        implementation: str,
        run_id: str,
        config: dict[str, Any],
    ) -> None:
        self.path = path
        self.implementation = implementation
        self.run_id = run_id
        self._handle = None
        if path is None or not dist_utils.is_main():
            return
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self._handle = open(path, "w", encoding="utf-8")
        self._write(
            {
                "record_type": "metadata",
                "schema_version": TRACE_SCHEMA_VERSION,
                "implementation": implementation,
                "run_id": run_id,
                "config": config,
            }
        )

    def _write(self, payload: dict[str, Any]) -> None:
        if self._handle is None:
            return
        self._handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._handle.flush()

    def record(self, step: RefreshStep) -> None:
        payload = asdict(step)
        payload["sample_indices"] = list(step.sample_indices)
        payload.update(
            {
                "record_type": "refresh_step",
                "schema_version": TRACE_SCHEMA_VERSION,
                "implementation": self.implementation,
                "run_id": self.run_id,
            }
        )
        self._write(payload)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "RefreshTraceWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def symmetric_relative_difference(a: float, b: float, floor: float = 1e-12) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.inf
    return 2.0 * abs(a - b) / max(abs(a) + abs(b), floor)


def load_refresh_trace(path: str) -> tuple[dict[str, Any], dict[tuple, RefreshStep]]:
    metadata = None
    steps: dict[tuple, RefreshStep] = {}
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("schema_version") != TRACE_SCHEMA_VERSION:
                raise ValueError(
                    f"{path}:{line_number}: unsupported trace schema "
                    f"{payload.get('schema_version')!r}"
                )
            record_type = payload.get("record_type")
            if record_type == "metadata":
                if metadata is not None:
                    raise ValueError(f"{path}: duplicate metadata record")
                metadata = payload
                continue
            if record_type != "refresh_step":
                raise ValueError(
                    f"{path}:{line_number}: unknown record_type={record_type!r}"
                )
            step = RefreshStep(
                layer=int(payload["layer"]),
                module=str(payload["module"]),
                block=int(payload["block"]),
                col_start=int(payload["col_start"]),
                col_end=int(payload["col_end"]),
                adam_step=int(payload["adam_step"]),
                loss=float(payload["loss"]),
                loss_current=_optional_float(payload.get("loss_current")),
                loss_next=_optional_float(payload.get("loss_next")),
                slide_alpha=_optional_float(payload.get("slide_alpha")),
                sample_indices=tuple(int(x) for x in payload.get("sample_indices", ())),
            )
            if step.identity in steps:
                raise ValueError(f"{path}: duplicate step identity {step.identity!r}")
            steps[step.identity] = step
    if metadata is None:
        raise ValueError(f"{path}: missing metadata record")
    return metadata, steps


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def compare_refresh_traces(
    reference_path: str,
    candidate_path: str,
    *,
    max_relative_difference: float = 0.01,
) -> dict[str, Any]:
    """Strictly join two traces and return a serializable comparison report."""

    ref_meta, ref = load_refresh_trace(reference_path)
    cand_meta, cand = load_refresh_trace(candidate_path)
    missing = sorted(set(ref) - set(cand))
    extra = sorted(set(cand) - set(ref))
    rows = []
    for identity in sorted(set(ref) & set(cand)):
        a = ref[identity]
        b = cand[identity]
        loss_diff = symmetric_relative_difference(a.loss, b.loss)
        sample_match = a.sample_indices == b.sample_indices
        alpha_diff = _optional_difference(a.slide_alpha, b.slide_alpha)
        passed = (
            loss_diff < max_relative_difference
            and sample_match
            and alpha_diff <= 1e-12
        )
        rows.append(
            {
                "identity": list(identity),
                "reference_loss": a.loss,
                "candidate_loss": b.loss,
                "relative_difference": loss_diff,
                "sample_indices_match": sample_match,
                "slide_alpha_abs_difference": alpha_diff,
                "passed": passed,
            }
        )
    worst = sorted(rows, key=lambda row: row["relative_difference"], reverse=True)
    passed = not missing and not extra and all(row["passed"] for row in rows)
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "reference": ref_meta,
        "candidate": cand_meta,
        "threshold": max_relative_difference,
        "matched_steps": len(rows),
        "missing_steps": [list(x) for x in missing],
        "extra_steps": [list(x) for x in extra],
        "max_relative_difference": (
            max((row["relative_difference"] for row in rows), default=0.0)
        ),
        "failed_steps": [row for row in worst if not row["passed"]],
        "worst_steps": worst[:20],
        "passed": passed,
    }


def _optional_difference(a: float | None, b: float | None) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return math.inf
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.inf
    return abs(a - b)


def trace_config_subset(config: Any, keys: Iterable[str]) -> dict[str, Any]:
    """Extract a stable primitive-only config subset from args/dataclass objects."""

    out: dict[str, Any] = {}
    for key in keys:
        value = getattr(config, key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, (list, tuple)):
            out[key] = list(value)
        else:
            out[key] = str(value)
    return out
