"""Machine-readable per-step traces used for legacy/refactor alignment."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from utils import dist_utils


TRACE_SCHEMA_VERSION = 4
SUPPORTED_TRACE_SCHEMA_VERSIONS = (3, TRACE_SCHEMA_VERSION)

# Keep trace metadata intentionally limited to knobs that exist with the same
# meaning in both the legacy and refactored implementations.  Implementation-
# specific performance/debug flags do not belong here: the comparison is
# intended to reject numerically different runs, not harmless orchestration
# differences.
REFRESH_TRACE_CONFIG_KEYS = (
    "model",
    "dataset",
    "seed",
    "rotation_seed",
    "refresh_seed",
    "nsamples",
    "seq_len",
    "w_bits",
    "w_groupsize",
    "w_asym",
    "w_clip",
    "num_groups",
    "percdamp",
    "blocksize",
    "act_order",
    "group_parallel_quant",
    "global_loss_bsz",
    "hessian_accum_bsz",
    "hessian_tf32",
    "saliency_clip_percentile",
    "grad_hessian_topk",
    "grad_lr",
    "grad_clip",
    "final_layer_grad_lr",
    "final_layer_grad_clip",
    "grad_lr_layer_schedule",
    "grad_lr_layer_base_ratio",
    "backward_samples",
    "backward_bsz",
    "final_layer_backward_bsz",
    "bsz",
    "a_loss_ratio",
    "a_loss_clip_scope",
    "loss_slide_window",
    "a_bits",
    "a_groupsize",
    "a_asym",
    "a_clip_ratio",
    "k_bits",
    "k_groupsize",
    "k_asym",
    "k_clip_ratio",
    "v_bits",
    "v_groupsize",
    "v_asym",
    "v_clip_ratio",
    "act_quant_aware_gptq",
    "k_cache_quant_aware_gptq",
    "rotate",
    "kl_topk",
    "quant_stop_layer",
)


@dataclass(frozen=True)
class ActiveWeightAudit:
    """Runtime evidence for one leaf participating in a Block-GD backward."""

    scope: str
    name: str
    parameter_name: str
    is_current: bool
    used: bool
    active_column_count: int
    source_storage_id: str
    storage_id_after: str | None
    optimizer_step_before: int
    optimizer_step_after: int
    update_applied: bool
    update_l2: float
    source_l2_before: float
    source_l2_after: float | None


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
    objective: str | None = None
    backward_invocation_id: int | None = None
    backward_chunk_sizes: tuple[int, ...] = ()
    backward_bsz: int | None = None
    global_count: int | None = None
    active_weights: tuple[ActiveWeightAudit, ...] = ()

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
        self._backward_invocation_count = 0
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

    @property
    def enabled(self) -> bool:
        """Whether all ranks must collect trace statistics.

        Only rank zero owns a file handle, but every rank must take the same
        instrumentation branches so loss sums can participate in collective
        aggregation.  Consequently this property is based on ``path`` rather
        than ``_handle``.
        """

        return self.path is not None

    def _write(self, payload: dict[str, Any]) -> None:
        if self._handle is None:
            return
        self._handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._handle.flush()

    def record(self, step: RefreshStep) -> None:
        payload = asdict(step)
        payload["sample_indices"] = list(step.sample_indices)
        payload["backward_chunk_sizes"] = list(
            step.backward_chunk_sizes
        )
        payload["active_weights"] = [
            asdict(item) for item in step.active_weights
        ]
        payload.update(
            {
                "record_type": "refresh_step",
                "schema_version": TRACE_SCHEMA_VERSION,
                "implementation": self.implementation,
                "run_id": self.run_id,
            }
        )
        self._write(payload)

    def allocate_backward_invocation_id(self) -> int:
        """Return a unique run-local ID at the actual backward call site."""

        self._backward_invocation_count += 1
        return self._backward_invocation_count

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
            if payload.get("schema_version") not in (
                SUPPORTED_TRACE_SCHEMA_VERSIONS
            ):
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
                objective=_optional_str(payload.get("objective")),
                backward_invocation_id=_optional_int(
                    payload.get("backward_invocation_id")
                ),
                backward_chunk_sizes=tuple(
                    int(x)
                    for x in payload.get("backward_chunk_sizes", ())
                ),
                backward_bsz=_optional_int(
                    payload.get("backward_bsz")
                ),
                global_count=_optional_int(payload.get("global_count")),
                active_weights=tuple(
                    ActiveWeightAudit(
                        scope=str(item["scope"]),
                        name=str(item["name"]),
                        parameter_name=str(item["parameter_name"]),
                        is_current=bool(item["is_current"]),
                        used=bool(item["used"]),
                        active_column_count=int(
                            item["active_column_count"]
                        ),
                        source_storage_id=str(
                            item["source_storage_id"]
                        ),
                        storage_id_after=_optional_str(
                            item.get("storage_id_after")
                        ),
                        optimizer_step_before=int(
                            item["optimizer_step_before"]
                        ),
                        optimizer_step_after=int(
                            item["optimizer_step_after"]
                        ),
                        update_applied=bool(item["update_applied"]),
                        update_l2=float(item["update_l2"]),
                        source_l2_before=float(
                            item["source_l2_before"]
                        ),
                        source_l2_after=_optional_float(
                            item.get("source_l2_after")
                        ),
                    )
                    for item in payload.get("active_weights", ())
                ),
            )
            if step.identity in steps:
                raise ValueError(f"{path}: duplicate step identity {step.identity!r}")
            steps[step.identity] = step
    if metadata is None:
        raise ValueError(f"{path}: missing metadata record")
    return metadata, steps


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def compare_refresh_traces(
    reference_path: str,
    candidate_path: str,
    *,
    max_relative_difference: float = 0.01,
) -> dict[str, Any]:
    """Strictly join two traces and return a serializable comparison report."""

    ref_meta, ref = load_refresh_trace(reference_path)
    cand_meta, cand = load_refresh_trace(candidate_path)
    reference_config = ref_meta.get("config", {})
    candidate_config = cand_meta.get("config", {})
    config_differences = _mapping_differences(reference_config, candidate_config)
    missing = sorted(set(ref) - set(cand))
    extra = sorted(set(cand) - set(ref))
    rows = []
    for identity in sorted(set(ref) & set(cand)):
        a = ref[identity]
        b = cand[identity]
        loss_diff = symmetric_relative_difference(a.loss, b.loss)
        current_loss_diff = _optional_relative_difference(
            a.loss_current, b.loss_current,
        )
        next_loss_diff = _optional_relative_difference(a.loss_next, b.loss_next)
        sample_match = a.sample_indices == b.sample_indices
        alpha_diff = _optional_difference(a.slide_alpha, b.slide_alpha)
        passed = (
            loss_diff < max_relative_difference
            and current_loss_diff < max_relative_difference
            and next_loss_diff < max_relative_difference
            and sample_match
            and alpha_diff <= 1e-12
        )
        rows.append(
            {
                "identity": list(identity),
                "reference_loss": a.loss,
                "candidate_loss": b.loss,
                "relative_difference": loss_diff,
                "current_loss_relative_difference": current_loss_diff,
                "next_loss_relative_difference": next_loss_diff,
                "sample_indices_match": sample_match,
                "slide_alpha_abs_difference": alpha_diff,
                "passed": passed,
            }
        )
    worst = sorted(rows, key=lambda row: row["relative_difference"], reverse=True)
    passed = (
        not config_differences
        and not missing
        and not extra
        and bool(rows)
        and all(row["passed"] for row in rows)
    )
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "reference": ref_meta,
        "candidate": cand_meta,
        "threshold": max_relative_difference,
        "config_differences": config_differences,
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


def _optional_relative_difference(a: float | None, b: float | None) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return math.inf
    return symmetric_relative_difference(a, b)


def _mapping_differences(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    differences: dict[str, dict[str, Any]] = {}
    for key in sorted(set(reference) | set(candidate)):
        a = reference.get(key)
        b = candidate.get(key)
        if a != b:
            differences[key] = {"reference": a, "candidate": b}
    return differences


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


def default_refresh_trace_config(config: Any) -> dict[str, Any]:
    """Return the shared, comparison-safe metadata subset for a run."""

    from utils import model_utils

    out = trace_config_subset(config, REFRESH_TRACE_CONFIG_KEYS)
    if out.get("quant_stop_layer") is not None:
        out["quant_stop_layer"] = int(out["quant_stop_layer"])
    out.update(
        {
            "model_artifact_identity": (
                model_utils.source_model_cache_identity(config)
            ),
            "rotation_artifact_identity": (
                model_utils.rotation_cache_identity(config)
            ),
            "world_size": dist_utils.get_world_size(),
            "saliency_clip_scope": "global_calibration",
        }
    )
    return out


def refresh_step_from_metrics(
    *,
    layer: int,
    module: str,
    metrics: dict[str, Any],
) -> RefreshStep:
    """Convert a legacy block observer payload to the shared trace schema."""

    loss = metrics.get("mean_refresh_loss")
    if loss is None:
        raise ValueError(
            "cannot trace a refresh step without globally aggregated "
            "`mean_refresh_loss`"
        )
    block = int(metrics["block_idx"])
    return RefreshStep(
        layer=int(layer),
        module=str(module),
        block=block,
        col_start=int(metrics["col_start"]),
        col_end=int(metrics["col_end"]),
        adam_step=int(metrics.get("adam_step", block + 1)),
        loss=float(loss),
        loss_current=_optional_float(
            metrics.get("mean_refresh_loss_current"),
        ),
        loss_next=_optional_float(metrics.get("mean_refresh_loss_next")),
        slide_alpha=_optional_float(metrics.get("slide_alpha")),
        sample_indices=tuple(
            int(index) for index in metrics.get("sample_indices", ())
        ),
    )
