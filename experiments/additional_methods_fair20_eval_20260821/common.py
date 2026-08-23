"""Frozen identities and artifact gates for the 55-checkpoint evaluation.

The quantization campaigns deliberately remain immutable while they run.  This
module therefore reads, but never edits, their three plans and resolves an
evaluation artifact only after the campaign-specific terminal receipt passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT.parent / "experiment_data"
OUTPUT_ROOT = DATA_ROOT / "additional_methods_fair20_eval_20260821_v1"
REFERENCE_MANIFEST_PATH = OUTPUT_ROOT / "reference_manifest.json"
EFFICIENTQAT_RECOVERY_ROOT = (
    DATA_ROOT / "efficientqat_weightonly_15group_20260821_v4_recovery"
)
EXPECTED_EFFICIENTQAT_PARENT_WORKER_SHA256 = (
    "7dd53f109bfadbef9345c0b9e59667eda26c7249d266ec445fe9132cb2504094"
)

# Every repository file that can change the numerical evaluation, artifact
# reconstruction, prompt construction, or official HumanEval+ score is frozen
# in the reference manifest.  Keeping the exact expected set here prevents a
# newly-added entry point from being silently omitted from the release record.
EVALUATION_SOURCE_RELATIVE = (
    "experiments/additional_methods_fair20_eval_20260821/common.py",
    "experiments/additional_methods_fair20_eval_20260821/loaders.py",
    "experiments/additional_methods_fair20_eval_20260821/prepare_references.py",
    "experiments/additional_methods_fair20_eval_20260821/run_suite.py",
    "experiments/additional_methods_fair20_eval_20260821/worker.py",
    "experiments/additional_methods_fair20_eval_20260821/launcher.py",
    "experiments/additional_methods_fair20_eval_20260821/score_humaneval.py",
    "experiments/additional_methods_fair20_eval_20260821/score_worker.py",
    "experiments/efficientqat_weightonly_15group_recovery_20260821/worker.py",
    "experiments/efficientqat_weightonly_15group_recovery_20260821/launcher.py",
    "experiments/efficientqat_compare/run_one.py",
    "experiments/efficientqat_compare/materialize.py",
    "experiments/turboboa_yaqa_qwen3_rerun/eval_yaqa_kl_ppl.py",
    "experiments/yaqa_compare/akv_aware.py",
    "YAQA_wclip/lib/utils/unsafe_import.py",
    "YAQA_wclip/lib/linear/quantized_linear.py",
    "YAQA_wclip/model/cache_utils.py",
    "YAQA_wclip/model/llama.py",
    "gptq_utils/quant_aware_utils.py",
    "realq/attention.py",
    "realq/akv.py",
    "realq/pipeline.py",
    "realq_benchmark/benchmarks/data.py",
    "realq_benchmark/benchmarks/generation.py",
    "realq_benchmark/benchmarks/runner.py",
    "realq_benchmark/benchmarks/schema.py",
    "realq_benchmark/benchmarks/scoring.py",
    "tools/bin/lowbit_activation_no_inet",
    "tools/lowbit_activation_evalplus_canoe.sh",
    "tools/lowbit_activation_evalplus_entrypoint.py",
    "utils/checkpoint_utils.py",
    "utils/cache_identity.py",
    "utils/data_utils.py",
    "utils/dist_utils.py",
    "utils/eval_utils.py",
    "utils/log_utils.py",
    "utils/loss_utils.py",
    "utils/memory_utils.py",
    "utils/model_utils.py",
    "utils/monkeypatch.py",
    "utils/quant_utils.py",
    "utils/rotation_utils.py",
    "utils/triton_qwen3_fusions.py",
)

EFFICIENTQAT_PLAN = (
    REPO_ROOT
    / "experiments/efficientqat_weightonly_15group_20260821/plan.json"
)
TURBOBOA_PLAN = REPO_ROOT / "experiments/turboboa_fair20_20260821/plan.json"
YAQA_PLAN = REPO_ROOT / "experiments/yaqa_wclip_fair20_20260821/plan.json"
BASELINE_PLAN = DATA_ROOT / "gptaq_guided_20group_20260809/plan.json"
EXPECTED_BASELINE_PLAN_SHA256 = (
    "833653387df1053f8329086d515bc5f19a0ffd40ed9c96adbdc4ee7a8be28d07"
)

EXPECTED_PLAN_SHA256 = {
    "efficientqat": "ceffe64189f07a5753a4feffa21553d8c802fa4c798af645ca6199b0fa271032",
    "turboboa": "48ad6e6ca3df8c3af2194e09632e5b4db7380e9b2fabf414f666551db3d1e0cc",
    "yaqa_wclip": "a7bbd0db640b6d87edbb286f640d744fa13c66c2adccc102d16be9253145bda1",
}

MODEL_ORDER = {
    "qwen3-32b": 0,
    "qwen3-8b": 1,
    "llama31-8b-instruct": 2,
    "qwen3-4b": 3,
    "qwen3-0.6b": 4,
}
SETTING_ORDER = {
    "W4A4KV4": 0,
    "W4A16KV16": 1,
    "W3A16KV16": 2,
    "W2A16KV16": 3,
}
PAPER_QA_TASKS = (
    "piqa",
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "lambada_openai",
    "ceval-valid",
    "boolq",
    "openbookqa",
    "social_iqa",
)
REASONING_TASKS = {
    "gsm8k": {"max_new_tokens": 1024, "batch_size": 32, "count": 1319},
    "math_500": {"max_new_tokens": 2048, "batch_size": 16, "count": 500},
    "humaneval_plus": {
        "max_new_tokens": 2048,
        "batch_size": 16,
        "count": 164,
    },
}
REFERENCE_STAT_KEYS = (
    "size_bytes",
    "mtime_ns",
    "ctime_ns",
    "device",
    "inode",
)
# st_dev is a mount-namespace identifier rather than a portable identity: the
# same shared file is device 133 on one Canoe pod and 22020131 on the other.
# inode/ctime/mtime/size are identical across both views and remain the
# mutation-sensitive per-consumer gate.  The publisher's st_dev is retained in
# the release solely as an audit observation.
REFERENCE_PORTABLE_STAT_KEYS = (
    "size_bytes",
    "mtime_ns",
    "ctime_ns",
    "inode",
)


class EvaluationError(RuntimeError):
    """A frozen input, terminal receipt, or evaluation gate failed."""


@dataclass(frozen=True)
class EvalSpec:
    method: str
    run_id: str
    model: str
    setting: str
    w_bits: int
    a_bits: int
    k_bits: int
    v_bits: int
    quant_dir: Path

    @property
    def eval_id(self) -> str:
        return f"{self.method}__{self.run_id}"

    @property
    def priority(self) -> tuple[int, int, str, str]:
        return (
            MODEL_ORDER[self.model],
            SETTING_ORDER[self.setting],
            self.method,
            self.run_id,
        )


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read JSON object {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"JSON root must be an object: {source}")
    return value


def reference_stat_identity(path: str | Path) -> dict[str, int]:
    """Return mutation-sensitive, read-free identity fields for a cache file."""

    stat = Path(path).stat()
    return {
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
    }


def reference_stat_matches(
    path: str | Path,
    released: Mapping[str, Any],
) -> bool:
    """Match mutation-sensitive fields that are stable across pod mounts."""

    try:
        expected = {key: int(released[key]) for key in REFERENCE_STAT_KEYS}
    except (KeyError, TypeError, ValueError):
        return False
    actual = reference_stat_identity(path)
    return (
        expected["device"] >= 0
        and all(
            actual[key] == expected[key]
            for key in REFERENCE_PORTABLE_STAT_KEYS
        )
    )


def evaluation_source_files() -> tuple[Path, ...]:
    paths = tuple(REPO_ROOT / relative for relative in EVALUATION_SOURCE_RELATIVE)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise EvaluationError(f"evaluation source files are missing: {missing}")
    return paths


def load_reference_manifest(*, verify_sources: bool = True) -> dict[str, Any]:
    """Validate the one-time teacher-cache hash release without rehashing it.

    The multi-gigabyte cache contents were hashed once by prepare_references.
    Every consumer checks the manifest fingerprint plus inode/ctime/mtime/size,
    so 55 suites do not repeatedly stream the same files from shared storage.
    """

    manifest = read_object(REFERENCE_MANIFEST_PATH)
    identity = dict(manifest)
    fingerprint = identity.pop("fingerprint", None)
    expected_source_names = set(EVALUATION_SOURCE_RELATIVE)
    sources = manifest.get("sources")
    references = manifest.get("references")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("status") != "prepared"
        or fingerprint != canonical_sha256(identity)
        or not isinstance(sources, dict)
        or set(sources) != expected_source_names
        or not isinstance(references, dict)
        or set(references) != set(MODEL_ORDER)
        or manifest.get("matrix_counts")
        != {"efficientqat": 15, "turboboa": 20, "yaqa_wclip": 20}
    ):
        raise EvaluationError("reference manifest identity/coverage failed")
    if manifest.get("baseline_plan") != {
        "path": str(BASELINE_PLAN),
        "sha256": EXPECTED_BASELINE_PLAN_SHA256,
    }:
        raise EvaluationError("reference manifest baseline binding failed")
    expected_plans = {
        "efficientqat": {
            "path": str(EFFICIENTQAT_PLAN),
            "sha256": EXPECTED_PLAN_SHA256["efficientqat"],
        },
        "turboboa": {
            "path": str(TURBOBOA_PLAN),
            "sha256": EXPECTED_PLAN_SHA256["turboboa"],
        },
        "yaqa_wclip": {
            "path": str(YAQA_PLAN),
            "sha256": EXPECTED_PLAN_SHA256["yaqa_wclip"],
        },
    }
    if manifest.get("quantization_plans") != expected_plans:
        raise EvaluationError("reference manifest quantization-plan binding failed")
    for model, record in references.items():
        path = Path(record.get("path", ""))
        digest = record.get("sha256")
        if (
            not path.is_file()
            or path.is_symlink()
            or not isinstance(digest, str)
            or len(digest) != 64
            or not reference_stat_matches(path, record)
        ):
            raise EvaluationError(f"reference cache identity changed for {model}")
    if verify_sources:
        for relative, expected in sources.items():
            if sha256_file(REPO_ROOT / relative) != expected:
                raise EvaluationError(
                    f"evaluation source changed after release: {relative}"
                )
    return manifest


def atomic_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
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
        os.replace(temporary, destination)
        os.chmod(destination, 0o644)
    finally:
        temporary.unlink(missing_ok=True)


def _load_frozen_plan(method: str, path: Path) -> dict[str, Any]:
    expected = EXPECTED_PLAN_SHA256[method]
    actual = sha256_file(path)
    if actual != expected:
        raise EvaluationError(
            f"{method} plan SHA256 changed: {actual} != {expected}"
        )
    return read_object(path)


def load_quant_plans() -> dict[str, dict[str, Any]]:
    return {
        "efficientqat": _load_frozen_plan("efficientqat", EFFICIENTQAT_PLAN),
        "turboboa": _load_frozen_plan("turboboa", TURBOBOA_PLAN),
        "yaqa_wclip": _load_frozen_plan("yaqa_wclip", YAQA_PLAN),
    }


def _bits_from_setting(setting: str) -> tuple[int, int, int, int]:
    expected = {
        "W4A16KV16": (4, 16, 16, 16),
        "W3A16KV16": (3, 16, 16, 16),
        "W2A16KV16": (2, 16, 16, 16),
        "W4A4KV4": (4, 4, 4, 4),
    }
    try:
        return expected[setting]
    except KeyError as exc:
        raise EvaluationError(f"unsupported setting: {setting!r}") from exc


def iter_specs(
    plans: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[EvalSpec, ...]:
    frozen = dict(load_quant_plans() if plans is None else plans)
    specs: list[EvalSpec] = []

    eq = frozen["efficientqat"]
    eq_root = Path(eq["output_root"])
    for run in eq["runs"]:
        bits = _bits_from_setting(run["setting"])
        if bits[1:] != (16, 16, 16):
            raise EvaluationError("EfficientQAT evaluation must be weight-only")
        specs.append(
            EvalSpec(
                method="efficientqat",
                run_id=run["run_id"],
                model=run["model"],
                setting=run["setting"],
                w_bits=bits[0],
                a_bits=bits[1],
                k_bits=bits[2],
                v_bits=bits[3],
                quant_dir=eq_root / run["output_subdir"],
            )
        )

    tb = frozen["turboboa"]
    tb_root = Path(tb["output_root"])
    for run in tb["runs"]:
        bits = _bits_from_setting(run["setting"])
        specs.append(
            EvalSpec(
                method="turboboa",
                run_id=run["run_id"],
                model=run["model"],
                setting=run["setting"],
                w_bits=bits[0],
                a_bits=bits[1],
                k_bits=bits[2],
                v_bits=bits[3],
                quant_dir=(
                    tb_root
                    / "runs"
                    / run["model"]
                    / run["setting"].lower()
                    / run["run_id"]
                ),
            )
        )

    yaqa = frozen["yaqa_wclip"]
    yaqa_root = Path(yaqa["output_root"])
    for stage in yaqa["stages"]:
        if stage["kind"] != "quantize":
            continue
        bits = _bits_from_setting(stage["setting"])
        specs.append(
            EvalSpec(
                method="yaqa_wclip",
                run_id=stage["stage_id"],
                model=stage["model"],
                setting=stage["setting"],
                w_bits=bits[0],
                a_bits=bits[1],
                k_bits=bits[2],
                v_bits=bits[3],
                quant_dir=yaqa_root / "stages" / stage["stage_id"],
            )
        )

    if len(specs) != 55:
        raise EvaluationError(f"expected 55 evaluation specs, got {len(specs)}")
    keys = [(spec.method, spec.model, spec.setting) for spec in specs]
    if len(keys) != len(set(keys)):
        raise EvaluationError("method/model/setting keys are not unique")
    return tuple(sorted(specs, key=lambda spec: spec.priority))


def model_contract(
    model: str,
    plans: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    frozen = dict(load_quant_plans() if plans is None else plans)
    try:
        return dict(frozen["efficientqat"]["models"][model])
    except (KeyError, TypeError) as exc:
        raise EvaluationError(f"unknown model contract: {model}") from exc


def reference_cache_record(model: str) -> dict[str, Any]:
    actual_plan_sha = sha256_file(BASELINE_PLAN)
    if actual_plan_sha != EXPECTED_BASELINE_PLAN_SHA256:
        raise EvaluationError(
            "GPTAQ/GuidedQuant baseline plan changed: "
            f"{actual_plan_sha} != {EXPECTED_BASELINE_PLAN_SHA256}"
        )
    baseline = read_object(BASELINE_PLAN)
    try:
        record = dict(baseline["cache_links"][model]["reference_logits"])
    except (KeyError, TypeError) as exc:
        raise EvaluationError(
            f"baseline plan has no reference cache for {model}"
        ) from exc
    source = Path(record["source"])
    if not source.is_file() or source.is_symlink():
        raise EvaluationError(f"reference cache is not a regular source file: {source}")
    stat = source.stat()
    if (
        stat.st_size != int(record["size_bytes"])
        or stat.st_mtime_ns != int(record["source_mtime_ns"])
    ):
        raise EvaluationError(f"reference cache stat changed: {source}")
    return {**record, "source": str(source.resolve())}


def resolve_completed_artifact(spec: EvalSpec) -> dict[str, Any]:
    if spec.method == "efficientqat":
        terminal = spec.quant_dir / "quantization_result.json"
        recovery_used = False
        if not terminal.is_file():
            parent_failure = spec.quant_dir / "failure.json"
            recovery_terminal = (
                EFFICIENTQAT_RECOVERY_ROOT
                / spec.quant_dir.relative_to(
                    Path(load_quant_plans()["efficientqat"]["output_root"])
                )
                / "quantization_result.json"
            )
            if parent_failure.is_file() and recovery_terminal.is_file():
                terminal = recovery_terminal
                recovery_used = True
        payload = read_object(terminal)
        if payload.get("status") != "quantization_succeeded":
            raise EvaluationError(f"EfficientQAT run is not complete: {spec.run_id}")
        if recovery_used:
            policy = payload.get("recovery_policy", {})
            recovery_sources = payload.get("source_sha256", {})
            parent_failure_path = (spec.quant_dir / "failure.json").resolve()
            parent_block_stage_path = (
                spec.quant_dir / "block_ap/stage.json"
            ).resolve()
            required_recovery_sources = (
                "experiments/efficientqat_weightonly_15group_recovery_20260821/worker.py",
                "experiments/efficientqat_weightonly_15group_recovery_20260821/launcher.py",
                "experiments/efficientqat_compare/run_one.py",
                "experiments/efficientqat_compare/materialize.py",
            )
            if (
                payload.get("parent_plan_sha256")
                != EXPECTED_PLAN_SHA256["efficientqat"]
                or Path(payload.get("parent_run_dir", "")).resolve()
                != spec.quant_dir.resolve()
                or Path(payload.get("parent_failure", "")).resolve()
                != parent_failure_path
                or Path(payload.get("parent_block_stage", "")).resolve()
                != parent_block_stage_path
                or policy.get("training_configuration_changed") is not False
                or policy.get("rerun_stages") != ["E2E-QP", "materialize"]
                or policy.get("reused_stage") != "successful parent Block-AP"
                or policy.get("superseded_failed_e2e_excluded") is not True
                or policy.get("algorithm_gpu_hour_accounting")
                != "parent Block-AP + recovered E2E-QP"
                or any(
                    recovery_sources.get(relative)
                    != sha256_file(REPO_ROOT / relative)
                    for relative in required_recovery_sources
                )
                or recovery_sources.get(
                    "experiments/efficientqat_weightonly_15group_20260821/worker.py"
                )
                != EXPECTED_EFFICIENTQAT_PARENT_WORKER_SHA256
                or sha256_file(parent_failure_path)
                != payload.get("parent_failure_sha256")
                or sha256_file(parent_block_stage_path)
                != payload.get("parent_block_stage_sha256")
            ):
                raise EvaluationError(
                    f"EfficientQAT recovery contract mismatch: {spec.run_id}"
                )
            stages = payload.get("stages", {})
            block_stage = stages.get("block_ap", {})
            e2e_stage = stages.get("e2e_qp", {})
            materialize_stage = stages.get("materialize", {})
            try:
                block_gpu_seconds = float(block_stage.get("gpu_seconds", -1))
                e2e_gpu_seconds = float(e2e_stage.get("gpu_seconds", -1))
                materialize_gpu_seconds = float(
                    materialize_stage.get("gpu_seconds", -1)
                )
                recorded_gpu_seconds = float(
                    payload.get("quantization_gpu_seconds", -1)
                )
                recorded_gpu_hours = float(
                    payload.get("quantization_gpu_hours", -1)
                )
            except (TypeError, ValueError) as exc:
                raise EvaluationError(
                    f"EfficientQAT recovery timing is invalid: {spec.run_id}"
                ) from exc
            expected_gpu_seconds = block_gpu_seconds + e2e_gpu_seconds
            if (
                block_stage.get("status") != "succeeded"
                or block_stage.get("reused_from_parent") is not True
                or block_stage.get("source_stage_sha256")
                != payload.get("parent_block_stage_sha256")
                or e2e_stage.get("status") != "succeeded"
                or materialize_stage.get("status") != "succeeded"
                or not all(
                    math.isfinite(value) and value >= 0
                    for value in (
                        block_gpu_seconds,
                        e2e_gpu_seconds,
                        recorded_gpu_seconds,
                        recorded_gpu_hours,
                    )
                )
                or materialize_gpu_seconds != 0.0
                or not math.isclose(
                    recorded_gpu_seconds,
                    expected_gpu_seconds,
                    rel_tol=0,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    recorded_gpu_hours,
                    recorded_gpu_seconds / 3600.0,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
            ):
                raise EvaluationError(
                    f"EfficientQAT recovery stage/accounting mismatch: {spec.run_id}"
                )
        artifact = Path(payload["materialized_checkpoint"]).resolve()
        validation = artifact / "efficientqat_manifest.json"
        if not artifact.is_dir() or not validation.is_file():
            raise EvaluationError(
                f"EfficientQAT materialized artifact is incomplete: {artifact}"
            )
        gpu_hours = float(payload["quantization_gpu_hours"])
        kind = "hf_dense"
        materialize = payload.get("stages", {}).get("materialize", {})
        if (
            Path(materialize.get("manifest", "")).resolve() != validation
            or Path(materialize.get("output", "")).resolve() != artifact
            or materialize.get("status") != "succeeded"
        ):
            raise EvaluationError(
                f"EfficientQAT materialization receipt mismatch: {spec.run_id}"
            )
    elif spec.method == "turboboa":
        terminal = spec.quant_dir / "run_receipt.json"
        payload = read_object(terminal)
        if payload.get("status") != "quantization_succeeded":
            raise EvaluationError(f"TurboBOA run is not complete: {spec.run_id}")
        artifact = Path(payload["checkpoint"]).resolve()
        validation = Path(payload["result"]).resolve()
        if not artifact.is_file() or not validation.is_file():
            raise EvaluationError(
                f"TurboBOA artifact is incomplete: {artifact}"
            )
        gpu_hours = float(payload["quantization_gpu_hours"])
        kind = "realq_checkpoint"
        if (
            payload.get("result_sha256") != sha256_file(validation)
            or int(payload.get("checkpoint_bytes", -1))
            != artifact.stat().st_size
        ):
            raise EvaluationError(
                f"TurboBOA receipt artifact mismatch: {spec.run_id}"
            )
    elif spec.method == "yaqa_wclip":
        terminal = spec.quant_dir / "stage_receipt.json"
        payload = read_object(terminal)
        if payload.get("status") != "succeeded":
            raise EvaluationError(f"YAQA run is not complete: {spec.run_id}")
        artifact = Path(payload["hf_dir"]).resolve()
        validation = Path(payload["validation"]).resolve()
        if not artifact.is_dir() or not validation.is_file():
            raise EvaluationError(f"YAQA HF artifact is incomplete: {artifact}")
        gpu_hours = float(payload["campaign_amortized_gpu_hours"])
        kind = "yaqa_hf_dense"
        if payload.get("validation_sha256") != sha256_file(validation):
            raise EvaluationError(
                f"YAQA validation receipt mismatch: {spec.run_id}"
            )
    else:
        raise EvaluationError(f"unknown method: {spec.method}")

    if kind == "realq_checkpoint":
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            raise EvaluationError(f"checkpoint is missing/empty: {artifact}")
    elif not artifact.is_dir():
        raise EvaluationError(f"HF artifact directory is missing: {artifact}")
    if not validation.is_file():
        raise EvaluationError(f"artifact validation is missing: {validation}")
    return {
        "kind": kind,
        "path": str(artifact),
        "validation": str(validation),
        "terminal": str(terminal.resolve()),
        "terminal_sha256": sha256_file(terminal),
        "validation_sha256": sha256_file(validation),
        "quantization_gpu_hours": gpu_hours,
    }


def completed_specs(specs: Iterable[EvalSpec] | None = None) -> list[EvalSpec]:
    result = []
    for spec in iter_specs() if specs is None else specs:
        try:
            resolve_completed_artifact(spec)
        except EvaluationError:
            continue
        result.append(spec)
    return result
