#!/usr/bin/env python3
"""Provenance-safe formal phase timing for the low-bit activation campaign.

This module is intentionally outside the numerical source tree and does not
replace any numerical function.  The timed executor imports the reviewed base
renderer/executor, installs an opt-in ``sitecustomize`` directory only in the
child environment, and calls the original executor unchanged.

The timing contract is ``algorithm_core_v1``:

* REAL-Q shared precompute: rotation through the completed cold static cache
  write;
* GuidedGPTQ shared precompute: calibration-token loading, rotation, gradient
  and saliency generation through completed artifact writes;
* REAL-Q/GPTAQ/GuidedGPTQ quantization: rotation through completed
  quantization/runtime A/K/V setup;
* model loading, FP-reference generation, checkpoint saving, WikiText-2
  KL/PPL evaluation, and lm-eval are excluded.

Every accepted sidecar is bound to the immutable execution identity, exact
rank evidence, exact instrumentation sources, and (where applicable) one
validated shared producer.  There is no process-duration fallback.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import lowbit_activation_execute as base_executor  # noqa: E402
import lowbit_activation_gpu_hours as base_gpu_hours  # noqa: E402
import lowbit_activation_runner as base_runner  # noqa: E402


class TimingAdapterError(RuntimeError):
    """Formal timing is incomplete, ambiguous, or provenance-unsafe."""


SCOPE_ID = "algorithm_core_v1"
SCHEMA_VERSION = 1
RANK_EVIDENCE_TEMPLATE = "phase_timing_rank{rank}.json"
SIDECAR_FILENAME = "phase_timing.json"
SOURCE_SNAPSHOT_DIRNAME = "phase_timing_sources"
SITE_DIR = SCRIPT_PATH.with_name("formal_timing_sitecustomize")
SITE_PATH = SITE_DIR / "sitecustomize.py"
TIMED_EXECUTOR_PATH = SCRIPT_PATH.with_name("formal_timed_execute.py")
TIMED_CAMPAIGN_PATH = SCRIPT_PATH.with_name("formal_timed_campaign.py")
GPU_HOURS_PATH = TOOLS_DIR / "lowbit_activation_gpu_hours.py"
GUIDED_VALIDATOR_PATH = TOOLS_DIR / "validate_guided_saliency.py"
BASE_CAMPAIGN_PATH = TOOLS_DIR / "lowbit_activation_campaign.py"

SOURCE_PATHS = (
    ("formal_timing_adapter", SCRIPT_PATH),
    ("formal_timing_sitecustomize", SITE_PATH),
    ("formal_timed_execute", TIMED_EXECUTOR_PATH),
    ("lowbit_activation_campaign", BASE_CAMPAIGN_PATH),
    ("formal_timed_campaign", TIMED_CAMPAIGN_PATH),
    ("lowbit_activation_gpu_hours", GPU_HOURS_PATH),
    ("validate_guided_saliency", GUIDED_VALIDATOR_PATH),
)

ENV_PREFIX = "LOWBIT_FORMAL_TIMING_"
ENV_KEYS = {
    "dir": f"{ENV_PREFIX}DIR",
    "expected_world": f"{ENV_PREFIX}EXPECT_WORLD_SIZE",
    "source_set_sha": f"{ENV_PREFIX}SOURCE_SET_SHA256",
    "site_sha": f"{ENV_PREFIX}SITECUSTOMIZE_SHA256",
    "adapter_sha": f"{ENV_PREFIX}ADAPTER_SHA256",
    "executor_sha": f"{ENV_PREFIX}TIMED_EXECUTOR_SHA256",
    "spec_sha": f"{ENV_PREFIX}SPEC_SHA256",
    "mode": f"{ENV_PREFIX}MODE",
    "method": f"{ENV_PREFIX}METHOD",
    "stage": f"{ENV_PREFIX}STAGE",
    "model": f"{ENV_PREFIX}MODEL",
    "setting": f"{ENV_PREFIX}SETTING",
    "phase": f"{ENV_PREFIX}PHASE",
    "target_phase": f"{ENV_PREFIX}TARGET_PHASE",
    "run_id": f"{ENV_PREFIX}RUN_ID",
}

INCLUDED_SEGMENTS = {
    ("shared_precompute", "realq"): [
        "rotation",
        "realq_static_precompute_core",
    ],
    ("shared_precompute", "guided_gptq"): [
        "rotation",
        "guided_saliency_precompute_core",
    ],
    ("quantization", "realq"): ["rotation", "quantization_core"],
    ("quantization", "gptaq"): ["rotation", "quantization_core"],
    ("quantization", "guided_gptq"): [
        "rotation",
        "quantization_core",
    ],
}
EXCLUDED_SEGMENTS = [
    "model_load",
    "fp_reference_generation",
    "kl_ppl_evaluation",
    "lm_eval",
    "checkpoint_save",
]
FORMAL_CALIBRATION = {
    "dataset": "wikitext2",
    "split": "train",
    "nsamples": 256,
    "seq_len": 2048,
    "seed": 1,
}


@dataclasses.dataclass(frozen=True)
class TimingSpec:
    """Exact formal timing identity derived from reviewed executor arguments."""

    mode: str
    kind: str
    method: str
    stage: str
    model: str
    setting: str | None
    phase: str | None
    target_phase: str | None
    run_id: str
    gpu_indices: tuple[int, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "scope_id": SCOPE_ID,
            "mode": self.mode,
            "kind": self.kind,
            "method": self.method,
            "stage": self.stage,
            "model": self.model,
            "setting": self.setting,
            "phase": self.phase,
            "target_phase": self.target_phase,
            "run_id": self.run_id,
            "gpu_indices": list(self.gpu_indices),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.payload())


@dataclasses.dataclass(frozen=True)
class TimingProvenance:
    """Source identities and child-only environment fixed before execution."""

    sources: tuple[dict[str, str], ...]
    source_set_sha256: str
    environment: Mapping[str, str]
    entrypoint_source_tree: Mapping[str, Any] | None


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _stable_file_bytes(path: Path) -> bytes:
    """Read a regular file while proving it did not change during the read."""

    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise TimingAdapterError(f"timing source is not a file: {resolved}")
    before = resolved.stat()
    raw = resolved.read_bytes()
    after = resolved.stat()
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(raw) != after.st_size
    ):
        raise TimingAdapterError(
            f"timing source changed while being read: {resolved}"
        )
    return raw


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(_stable_file_bytes(path)).hexdigest()


def _load_json_no_duplicates(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TimingAdapterError(
                    f"duplicate JSON key {key!r} in {path}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except TimingAdapterError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TimingAdapterError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TimingAdapterError(f"JSON root must be an object: {path}")
    return value


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    raw = (
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(path, raw)


def _option(argv: Sequence[str], flag: str) -> str | None:
    values: list[str] = []
    prefix = f"{flag}="
    for index, item in enumerate(argv):
        if item == flag:
            if index + 1 >= len(argv):
                raise TimingAdapterError(
                    f"rendered option {flag} has no value"
                )
            values.append(argv[index + 1])
        elif item.startswith(prefix):
            values.append(item[len(prefix) :])
    if len(values) > 1:
        raise TimingAdapterError(f"rendered command duplicates {flag}")
    return values[0] if values else None


def _has_bare_flag(argv: Sequence[str], flag: str) -> bool:
    return sum(item == flag for item in argv) == 1


def _require_option(
    argv: Sequence[str],
    flag: str,
    expected: str,
) -> None:
    actual = _option(argv, flag)
    if actual != expected:
        raise TimingAdapterError(
            f"formal timing requires {flag}={expected!r}, got {actual!r}"
        )


def _reject_option(argv: Sequence[str], flag: str) -> None:
    prefix = f"{flag}="
    if flag in argv or any(item.startswith(prefix) for item in argv):
        raise TimingAdapterError(
            f"formal timing boundary is invalid with {flag}"
        )


def _checkpoint_path(
    rendered: base_runner.RenderedCommand,
) -> Path:
    return rendered.output_dir / "checkpoint" / "model.pt"


def prepare_formal_rendered(
    args: Any,
    rendered: base_runner.RenderedCommand,
) -> base_runner.RenderedCommand:
    """Apply formal-only entrypoint/checkpoint requirements.

    The reviewed base renderer stays byte-identical to the tuning renderer, so
    REAL-Q tune/final provenance can still be reconciled.  This outer formal
    layer only switches formal REAL-Q to ``realq_benchmark`` and adds the
    explicitly requested, attempt-local quantized checkpoint path.
    """

    argv = list(rendered.argv)
    method = str(args.method)
    if method in {"realq", "realq_static"}:
        matches = [
            index for index, value in enumerate(argv)
            if value == "realq.ptq"
        ]
        if len(matches) != 1:
            raise TimingAdapterError(
                "formal REAL-Q command must contain exactly one realq.ptq "
                "module before benchmark substitution"
            )
        argv[matches[0]] = "realq_benchmark.ptq"

    quantization = (
        (method == "realq" and args.phase == "final")
        or (
            method in {"gptaq", "guided_gptq"}
            and args.phase == "final"
        )
    )
    if quantization:
        _reject_option(argv, "--load_qmodel_path")
        _reject_option(argv, "--save_qmodel_path")
        argv.extend(
            ["--save_qmodel_path", str(_checkpoint_path(rendered))]
        )

    return dataclasses.replace(rendered, argv=argv)


def _require_checkpoint_option(
    rendered: base_runner.RenderedCommand,
) -> None:
    actual = _option(rendered.argv, "--save_qmodel_path")
    expected = str(_checkpoint_path(rendered))
    if actual != expected:
        raise TimingAdapterError(
            "formal quantization checkpoint path must be the immutable "
            f"attempt-local path {expected!r}, got {actual!r}"
        )


def _torchrun_world_size(argv: Sequence[str]) -> int:
    value = _option(argv, "--nproc-per-node")
    if value is None:
        value = _option(argv, "--nproc_per_node")
    if value is None:
        return 1
    try:
        world = int(value)
    except ValueError as exc:
        raise TimingAdapterError(
            "torchrun world size is not an integer"
        ) from exc
    if world <= 0:
        raise TimingAdapterError("torchrun world size must be positive")
    return world


def _calibration_protocol(
    rendered: base_runner.RenderedCommand,
) -> dict[str, Any]:
    argv = rendered.argv
    observed = {
        "dataset": _option(argv, "--dataset"),
        "split": "train",
        "nsamples": _option(argv, "--nsamples"),
        "seq_len": _option(argv, "--seq_len"),
        "seed": _option(argv, "--seed"),
    }
    expected = {
        "dataset": FORMAL_CALIBRATION["dataset"],
        "split": FORMAL_CALIBRATION["split"],
        "nsamples": str(FORMAL_CALIBRATION["nsamples"]),
        "seq_len": str(FORMAL_CALIBRATION["seq_len"]),
        "seed": str(FORMAL_CALIBRATION["seed"]),
    }
    drift = {
        key: {"expected": expected[key], "actual": observed[key]}
        for key in expected
        if observed[key] != expected[key]
    }
    if drift:
        raise TimingAdapterError(
            f"formal calibration protocol drifted: {drift}"
        )
    model = _option(argv, "--model")
    if not model:
        raise TimingAdapterError(
            "formal calibration protocol lacks --model/tokenizer identity"
        )
    payload = {
        "schema_version": 1,
        "dataset": expected["dataset"],
        "split": expected["split"],
        "nsamples": FORMAL_CALIBRATION["nsamples"],
        "seq_len": FORMAL_CALIBRATION["seq_len"],
        "seed": FORMAL_CALIBRATION["seed"],
        "model_and_tokenizer": model,
        "sampler": (
            "utils.data_utils.get_tokens/"
            "_sample_concat_and_tokenize"
        ),
    }
    return {
        **payload,
        "identity_sha256": _canonical_sha256(payload),
        "cache_file_may_differ_by_entrypoint": True,
        "semantic_identity_basis": (
            "same model/tokenizer, dataset split, sampler, sample count, "
            "sequence length, and seed"
        ),
    }


def _validate_command(
    rendered: base_runner.RenderedCommand,
    spec: TimingSpec,
) -> None:
    argv = rendered.argv
    if _option(argv, "--exp") != spec.run_id:
        raise TimingAdapterError(
            "rendered --exp does not match the formal timing run_id"
        )
    if _torchrun_world_size(argv) != len(spec.gpu_indices):
        raise TimingAdapterError(
            "rendered torchrun world size disagrees with allocated GPUs"
        )
    if tuple(
        base_runner._cuda_indices(rendered.env["CUDA_VISIBLE_DEVICES"])
    ) != spec.gpu_indices:
        raise TimingAdapterError(
            "formal timing GPU identity drifted after spec creation"
        )
    for forbidden in (
        "--load_qmodel_path",
        "--fsdp_meta_init",
        "--fsdp_precompute",
        "--stage2_cpu_master",
    ):
        _reject_option(argv, forbidden)
    _calibration_protocol(rendered)

    if spec.mode in {"realq_shared_precompute", "realq_quantization"}:
        if "realq_benchmark.ptq" not in argv:
            raise TimingAdapterError(
                "formal REAL-Q must execute realq_benchmark.ptq"
            )
        _require_option(argv, "--rotate", "true")
        _require_option(argv, "--skip_eval", "false")
        if spec.mode == "realq_shared_precompute":
            _reject_option(argv, "--save_qmodel_path")
            _require_option(argv, "--exit_after_precompute", "true")
            _require_option(argv, "--require_static_cache_hit", "false")
        else:
            _require_checkpoint_option(rendered)
            exit_after_precompute = _option(
                argv, "--exit_after_precompute"
            )
            if exit_after_precompute not in {None, "false"}:
                raise TimingAdapterError(
                    "formal REAL-Q quantization must not exit after "
                    "precompute"
                )
            _require_option(argv, "--require_static_cache_hit", "true")
            _require_option(argv, "--lm_eval", "true")
        return

    if spec.mode == "legacy_quantization":
        _require_checkpoint_option(rendered)
        if not any(Path(item).name == "ptq.py" for item in argv):
            raise TimingAdapterError(
                "legacy quantization timing must execute ptq.py"
            )
        if not _has_bare_flag(argv, "--rotate"):
            raise TimingAdapterError(
                "legacy quantization timing requires exactly one --rotate"
            )
        if not _has_bare_flag(argv, "--lm_eval"):
            raise TimingAdapterError(
                "legacy quantization timing requires formal lm-eval"
            )
        _reject_option(argv, "--skip_eval")
        expected_method = (
            "gptaq" if spec.method == "gptaq" else "gptq_guided"
        )
        _require_option(argv, "--w_method", expected_method)
        return

    if spec.mode == "guided_precompute":
        _reject_option(argv, "--save_qmodel_path")
        if not any(Path(item).name == "save_grads.py" for item in argv):
            raise TimingAdapterError(
                "GuidedGPTQ timing must execute save_grads.py"
            )
        if not _has_bare_flag(argv, "--rotate"):
            raise TimingAdapterError(
                "GuidedGPTQ precompute requires exactly one --rotate"
            )
        _require_option(argv, "--mode", "gradients")
        return

    raise TimingAdapterError(f"unsupported timing mode {spec.mode!r}")


def spec_from_executor_args(
    args: Any,
    rendered: base_runner.RenderedCommand,
) -> TimingSpec | None:
    """Map only formal, GPU-hour-applicable commands to a timing spec.

    BF16 is intentionally not applicable.  Tune commands are rejected rather
    than silently timed under the formal accounting scope.
    """

    method = str(args.method)
    if method == "bf16":
        if args.phase != "final":
            raise TimingAdapterError("BF16 is only valid in the final phase")
        return None

    gpu_indices = tuple(
        base_runner._cuda_indices(rendered.env["CUDA_VISIBLE_DEVICES"])
    )
    if method == "realq_static":
        if args.phase != "precompute" or args.target_phase != "final":
            raise TimingAdapterError(
                "timed REAL-Q static precompute must target formal/final"
            )
        spec = TimingSpec(
            mode="realq_shared_precompute",
            kind="shared_precompute",
            method="realq",
            stage="realq_static",
            model=str(args.model),
            setting=None,
            phase=None,
            target_phase="final",
            run_id=rendered.run_id,
            gpu_indices=gpu_indices,
        )
    elif method == "guided_saliency":
        if args.phase != "precompute":
            raise TimingAdapterError(
                "timed GuidedGPTQ saliency must be a precompute command"
            )
        spec = TimingSpec(
            mode="guided_precompute",
            kind="shared_precompute",
            method="guided_gptq",
            stage="guided_saliency",
            model=str(args.model),
            setting=None,
            phase=None,
            target_phase="final",
            run_id=rendered.run_id,
            gpu_indices=gpu_indices,
        )
    elif method == "realq":
        if args.phase != "final":
            raise TimingAdapterError(
                "the formal timed executor refuses REAL-Q tune commands"
            )
        spec = TimingSpec(
            mode="realq_quantization",
            kind="quantization",
            method="realq",
            stage="realq_quantization",
            model=str(args.model),
            setting=str(args.setting),
            phase="final",
            target_phase=None,
            run_id=rendered.run_id,
            gpu_indices=gpu_indices,
        )
    elif method in {"gptaq", "guided_gptq"}:
        if args.phase != "final":
            raise TimingAdapterError(
                "baseline timing is only valid in the final phase"
            )
        spec = TimingSpec(
            mode="legacy_quantization",
            kind="quantization",
            method=method,
            stage=f"{method}_quantization",
            model=str(args.model),
            setting=str(args.setting),
            phase="final",
            target_phase=None,
            run_id=rendered.run_id,
            gpu_indices=gpu_indices,
        )
    else:
        raise TimingAdapterError(
            f"unsupported formal timing method {method!r}"
        )

    if spec.setting == "None":
        raise TimingAdapterError(
            f"{method} formal quantization requires a setting"
        )
    _validate_command(rendered, spec)
    return spec


def _collect_source_identities() -> tuple[dict[str, str], ...]:
    identities: list[dict[str, str]] = []
    for name, path in SOURCE_PATHS:
        raw = _stable_file_bytes(path)
        identities.append(
            {
                "name": name,
                "original_path": str(path.resolve(strict=True)),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return tuple(identities)


def _source_set_sha256(
    sources: Sequence[Mapping[str, str]],
) -> str:
    return _canonical_sha256(
        [
            {"name": item["name"], "sha256": item["sha256"]}
            for item in sources
        ]
    )


def timing_source_set_identity() -> dict[str, Any]:
    """Return the exact instrumentation source set used by formal runs."""

    sources = _collect_source_identities()
    return {
        "schema_version": SCHEMA_VERSION,
        "source_set_sha256": _source_set_sha256(sources),
        "sources": [dict(item) for item in sources],
    }


def _formal_entrypoint_source_tree(
    spec: TimingSpec,
    *,
    repo_root: Path,
) -> dict[str, Any] | None:
    if spec.mode not in {
        "realq_shared_precompute",
        "realq_quantization",
    }:
        return None
    package_root = (repo_root / "realq_benchmark").resolve(strict=True)
    records: list[dict[str, str]] = []
    for path in sorted(package_root.rglob("*.py")):
        if not path.is_file():
            continue
        raw = _stable_file_bytes(path)
        records.append(
            {
                "relative_path": str(path.relative_to(repo_root)),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    if not records:
        raise TimingAdapterError(
            "realq_benchmark source tree contains no Python files"
        )
    return {
        "algorithm": "sha256(canonical_json(relative_path,sha256)[])",
        "root": str(package_root),
        "file_count": len(records),
        "combined_sha256": _canonical_sha256(records),
        "files": records,
    }


def instrument_rendered(
    rendered: base_runner.RenderedCommand,
    spec: TimingSpec,
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[base_runner.RenderedCommand, TimingProvenance]:
    """Prepare child-only timing environment without changing command.env.

    The base campaign validates ``manifest.command.env`` against the reviewed
    renderer byte-for-byte.  Therefore the overlay is applied to the outer
    executor process just while the unchanged base executor runs.  The base
    manifest still captures the overlay in ``effective_environment``.
    """

    _validate_command(rendered, spec)
    sources = _collect_source_identities()
    by_name = {item["name"]: item for item in sources}
    source_set_sha = _source_set_sha256(sources)
    output_dir = rendered.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir = output_dir.resolve(strict=False)

    collisions = sorted(
        key
        for key in (
            "PYTHONPATH",
            *ENV_KEYS.values(),
        )
        if key in rendered.env
    )
    if collisions:
        raise TimingAdapterError(
            "reviewed rendered env unexpectedly owns timing overlay keys: "
            f"{collisions!r}"
        )
    rendered_cublas = rendered.env.get("CUBLAS_WORKSPACE_CONFIG")
    if rendered_cublas not in {None, ":4096:8"}:
        raise TimingAdapterError(
            "rendered CUBLAS_WORKSPACE_CONFIG conflicts with deterministic "
            "timing bootstrap"
        )

    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    entries = [str(SITE_DIR), str(repo_root.resolve())]
    for entry in existing_pythonpath.split(os.pathsep):
        if entry and entry not in entries:
            entries.append(entry)
    environment = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONPATH": os.pathsep.join(entries),
        ENV_KEYS["dir"]: str(output_dir),
        ENV_KEYS["expected_world"]: str(len(spec.gpu_indices)),
        ENV_KEYS["source_set_sha"]: source_set_sha,
        ENV_KEYS["site_sha"]: by_name[
            "formal_timing_sitecustomize"
        ]["sha256"],
        ENV_KEYS["adapter_sha"]: by_name[
            "formal_timing_adapter"
        ]["sha256"],
        ENV_KEYS["executor_sha"]: by_name[
            "formal_timed_execute"
        ]["sha256"],
        ENV_KEYS["spec_sha"]: spec.sha256,
        ENV_KEYS["mode"]: spec.mode,
        ENV_KEYS["method"]: spec.method,
        ENV_KEYS["stage"]: spec.stage,
        ENV_KEYS["model"]: spec.model,
        ENV_KEYS["setting"]: spec.setting or "",
        ENV_KEYS["phase"]: spec.phase or "",
        ENV_KEYS["target_phase"]: spec.target_phase or "",
        ENV_KEYS["run_id"]: spec.run_id,
    }
    entrypoint_source_tree = _formal_entrypoint_source_tree(
        spec, repo_root=repo_root
    )
    return rendered, TimingProvenance(
        sources=sources,
        source_set_sha256=source_set_sha,
        environment=environment,
        entrypoint_source_tree=entrypoint_source_tree,
    )


@contextlib.contextmanager
def child_environment(
    provenance: TimingProvenance,
) -> Iterator[None]:
    """Temporarily install the isolated child overlay in this executor."""

    previous = {
        key: os.environ.get(key) for key in provenance.environment
    }
    try:
        os.environ.update(provenance.environment)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _manifest_identity(manifest: Mapping[str, Any]) -> dict[str, str]:
    try:
        plan = manifest["plan"]
        model = manifest["model"]
        numerical = manifest["numerical_source_tree"]
        identity = {
            "execution_id": str(manifest["execution_id"]),
            "run_id": str(manifest["run_id"]),
            "plan_sha256": str(plan["sha256"]),
            "model_sha256": str(model["combined_identity_sha256"]),
            "numerical_source_sha256": str(
                numerical["combined_sha256"]
            ),
        }
    except (KeyError, TypeError) as exc:
        raise TimingAdapterError(
            "execution manifest lacks immutable timing identity"
        ) from exc
    if (
        plan.get("changed_during_execution") is not False
        or plan.get("sha256_at_end") != plan.get("sha256")
    ):
        raise TimingAdapterError(
            "plan identity changed during timed execution"
        )
    if model.get("complete") is not True:
        raise TimingAdapterError("model identity is incomplete")
    if (
        numerical.get("changed_during_execution") is not False
        or numerical.get("combined_sha256_at_end")
        != numerical.get("combined_sha256")
    ):
        raise TimingAdapterError(
            "numerical source tree changed during timed execution"
        )
    if any(not value for value in identity.values()):
        raise TimingAdapterError("timing identity contains an empty value")
    return identity


def _validate_effective_environment(
    manifest: Mapping[str, Any],
    provenance: TimingProvenance,
) -> None:
    actual = manifest.get("effective_environment")
    if not isinstance(actual, Mapping):
        raise TimingAdapterError(
            "manifest effective_environment is missing"
        )
    drift = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in provenance.environment.items()
        if actual.get(key) != value
    }
    if drift:
        raise TimingAdapterError(
            f"timing child environment drifted: {drift}"
        )


def _verify_sources_unchanged(provenance: TimingProvenance) -> None:
    current = _collect_source_identities()
    if current != provenance.sources:
        raise TimingAdapterError(
            "timing instrumentation sources changed during execution"
        )
    if _source_set_sha256(current) != provenance.source_set_sha256:
        raise TimingAdapterError(
            "timing source-set identity changed during execution"
        )
    expected_entrypoint = provenance.entrypoint_source_tree
    if expected_entrypoint is not None:
        mode = str(provenance.environment[ENV_KEYS["mode"]])
        method = str(provenance.environment[ENV_KEYS["method"]])
        spec = TimingSpec(
            mode=mode,
            kind=(
                "shared_precompute"
                if mode == "realq_shared_precompute"
                else "quantization"
            ),
            method=method,
            stage=str(provenance.environment[ENV_KEYS["stage"]]),
            model=str(provenance.environment[ENV_KEYS["model"]]),
            setting=(
                str(provenance.environment[ENV_KEYS["setting"]]) or None
            ),
            phase=(
                str(provenance.environment[ENV_KEYS["phase"]]) or None
            ),
            target_phase=(
                str(provenance.environment[ENV_KEYS["target_phase"]])
                or None
            ),
            run_id=str(provenance.environment[ENV_KEYS["run_id"]]),
            gpu_indices=tuple(
                range(
                    int(
                        provenance.environment[
                            ENV_KEYS["expected_world"]
                        ]
                    )
                )
            ),
        )
        current_entrypoint = _formal_entrypoint_source_tree(
            spec, repo_root=REPO_ROOT
        )
        if current_entrypoint != expected_entrypoint:
            raise TimingAdapterError(
                "formal realq_benchmark source tree changed during execution"
            )


def _rank_evidence(
    *,
    output_dir: Path,
    spec: TimingSpec,
    provenance: TimingProvenance,
    require_complete: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    expected_by_path = {
        output_dir / RANK_EVIDENCE_TEMPLATE.format(rank=rank): rank
        for rank in range(len(spec.gpu_indices))
    }
    actual_paths = sorted(output_dir.glob("phase_timing_rank*.json"))
    extras = [
        path for path in actual_paths if path not in expected_by_path
    ]
    missing = [
        path for path in expected_by_path if path not in actual_paths
    ]
    if extras or (require_complete and missing) or not actual_paths:
        raise TimingAdapterError(
            "rank timing evidence set is incomplete or contains extras: "
            f"missing={missing!r}, extras={extras!r}, "
            f"actual={actual_paths!r}"
        )

    rows: list[dict[str, Any]] = []
    evidence_files: list[dict[str, str]] = []
    expected_common = {
        "schema_version": SCHEMA_VERSION,
        "scope_id": SCOPE_ID,
        "source_set_sha256": provenance.source_set_sha256,
        "sitecustomize_sha256": next(
            item["sha256"]
            for item in provenance.sources
            if item["name"] == "formal_timing_sitecustomize"
        ),
        "spec_sha256": spec.sha256,
        "run_id": spec.run_id,
        "mode": spec.mode,
        "method": spec.method,
        "stage": spec.stage,
        "model": spec.model,
        "setting": spec.setting,
        "phase": spec.phase,
        "target_phase": spec.target_phase,
        "world_size": len(spec.gpu_indices),
        "clock": "time.perf_counter_ns",
    }
    for path in actual_paths:
        rank = expected_by_path[path]
        row = _load_json_no_duplicates(path)
        expected = {
            **expected_common,
            "rank": rank,
            "local_rank": rank,
            "physical_gpu_index": spec.gpu_indices[rank],
        }
        drift = {
            key: {"expected": value, "actual": row.get(key)}
            for key, value in expected.items()
            if row.get(key) != value
        }
        if drift:
            raise TimingAdapterError(
                f"rank {rank} timing identity drifted: {drift}"
            )
        complete = row.get("complete")
        timing_status = row.get("timing_status")
        if require_complete and complete is not True:
            raise TimingAdapterError(
                f"rank {rank} timing interval is incomplete: "
                f"{row.get('error')!r}"
            )
        if complete is True:
            if (
                timing_status != "complete"
                or row.get("synchronized_start") is not True
                or row.get("synchronized_end") is not True
                or row.get("error") is not None
            ):
                raise TimingAdapterError(
                    f"rank {rank} completed timing lacks synchronized bounds"
                )
            start = row.get("started_monotonic_ns")
            end = row.get("ended_monotonic_ns")
            elapsed = row.get("elapsed_seconds")
            if (
                type(start) is not int
                or type(end) is not int
                or end < start
                or isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(float(elapsed))
                or not math.isclose(
                    float(elapsed),
                    (end - start) / 1_000_000_000,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            ):
                raise TimingAdapterError(
                    f"rank {rank} timing bounds/elapsed are invalid"
                )
        else:
            if complete is not False:
                raise TimingAdapterError(
                    f"rank {rank} complete must be a boolean"
                )
            if (
                row.get("synchronized_end") is not False
                or not isinstance(row.get("error"), str)
                or not row["error"]
            ):
                raise TimingAdapterError(
                    f"rank {rank} incomplete timing lacks an error/status"
                )
            start = row.get("started_monotonic_ns")
            end = row.get("ended_monotonic_ns")
            elapsed = row.get("elapsed_seconds")
            if timing_status == "not_started":
                if (
                    start is not None
                    or end is not None
                    or elapsed is not None
                    or row.get("synchronized_start") is not False
                ):
                    raise TimingAdapterError(
                        f"rank {rank} not_started timing has interval bounds"
                    )
            elif timing_status == "partial":
                if (
                    type(start) is not int
                    or type(end) is not int
                    or end < start
                    or row.get("synchronized_start") is not True
                    or isinstance(elapsed, bool)
                    or not isinstance(elapsed, (int, float))
                    or not math.isfinite(float(elapsed))
                    or not math.isclose(
                        float(elapsed),
                        (end - start) / 1_000_000_000,
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    )
                ):
                    raise TimingAdapterError(
                        f"rank {rank} partial timing bounds are invalid"
                    )
            else:
                raise TimingAdapterError(
                    f"rank {rank} has invalid timing_status "
                    f"{timing_status!r}"
                )
        for field in ("cache_lookups", "cache_writes", "artifacts"):
            if not isinstance(row.get(field), list):
                raise TimingAdapterError(
                    f"rank {rank} evidence field {field} must be a list"
                )
        rows.append(row)
        evidence_files.append(
            {
                "path": path.name,
                "sha256": _sha256_file(path),
            }
        )
    return rows, evidence_files


def _resolve_path(value: str | os.PathLike[str], *, repo_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve(strict=False)


def _file_artifact(
    value: str | os.PathLike[str],
    *,
    repo_root: Path,
    allowed_root: Path | None = None,
) -> dict[str, str]:
    path = _resolve_path(value, repo_root=repo_root).resolve(strict=True)
    if not path.is_file():
        raise TimingAdapterError(f"timed artifact is not a file: {path}")
    if allowed_root is not None:
        root = allowed_root.resolve(strict=True)
        if path != root and root not in path.parents:
            raise TimingAdapterError(
                f"timed artifact escapes expected root {root}: {path}"
            )
    return {"path": str(path), "sha256": _sha256_file(path)}


def _realq_cache_evidence(
    *,
    rows: Sequence[Mapping[str, Any]],
    spec: TimingSpec,
    rendered: base_runner.RenderedCommand,
    repo_root: Path,
) -> tuple[str, list[dict[str, str]]]:
    cache_root_value = _option(rendered.argv, "--static_cache_path")
    if not cache_root_value:
        raise TimingAdapterError(
            "REAL-Q timed command lacks --static_cache_path"
        )
    cache_root = _resolve_path(cache_root_value, repo_root=repo_root)
    keys: set[str] = set()
    artifacts: list[dict[str, str]] = []
    world = len(spec.gpu_indices)
    for rank, row in enumerate(rows):
        lookups = row["cache_lookups"]
        writes = row["cache_writes"]
        if len(lookups) != 1 or not isinstance(lookups[0], Mapping):
            raise TimingAdapterError(
                "each REAL-Q rank must record one cache lookup"
            )
        lookup = lookups[0]
        key = lookup.get("key")
        if not isinstance(key, str) or not key:
            raise TimingAdapterError("REAL-Q cache lookup has no key")
        keys.add(key)
        expected_path = (
            cache_root / f"{key}_world{world}_rank{rank}.pt"
        ).resolve(strict=False)
        observed_path = _resolve_path(
            str(lookup.get("path")), repo_root=repo_root
        )
        if observed_path != expected_path:
            raise TimingAdapterError(
                f"rank {rank} REAL-Q cache path drifted"
            )
        if spec.kind == "shared_precompute":
            if lookup.get("hit") is not False:
                raise TimingAdapterError(
                    "formal REAL-Q producer must be a cold cache miss"
                )
            if len(writes) != 1 or not isinstance(writes[0], Mapping):
                raise TimingAdapterError(
                    "each REAL-Q producer rank must record one cache write"
                )
            write = writes[0]
            if (
                write.get("key") != key
                or _resolve_path(
                    str(write.get("path")), repo_root=repo_root
                )
                != expected_path
            ):
                raise TimingAdapterError(
                    f"rank {rank} REAL-Q cache write drifted"
                )
            artifacts.append(
                _file_artifact(
                    expected_path,
                    repo_root=repo_root,
                    allowed_root=cache_root,
                )
            )
        else:
            if lookup.get("hit") is not True or writes != []:
                raise TimingAdapterError(
                    "formal REAL-Q quantization must consume, not write, "
                    "the static cache"
                )
    if len(keys) != 1:
        raise TimingAdapterError(
            f"REAL-Q ranks disagree on cache key: {keys!r}"
        )
    artifacts.sort(key=lambda item: item["path"])
    if len({item["path"] for item in artifacts}) != len(artifacts):
        raise TimingAdapterError("REAL-Q cache artifacts are not unique")
    return next(iter(keys)), artifacts


def _guided_artifacts(
    *,
    rows: Sequence[Mapping[str, Any]],
    rendered: base_runner.RenderedCommand,
    plan: Mapping[str, Any],
    identity: Mapping[str, str],
    repo_root: Path,
) -> tuple[str, list[dict[str, str]]]:
    if len(rows) != 1:
        raise TimingAdapterError(
            "GuidedGPTQ saliency producer must use exactly one rank"
        )
    evidence = rows[0]["artifacts"]
    if len(evidence) != 1 or not isinstance(evidence[0], Mapping):
        raise TimingAdapterError(
            "GuidedGPTQ rank evidence must record one artifact set"
        )
    artifact_evidence = evidence[0]
    gradients_value = artifact_evidence.get("gradients_path")
    saliency_value = artifact_evidence.get("saliency_path")
    num_groups_value = artifact_evidence.get("num_groups")
    if (
        not isinstance(gradients_value, str)
        or not isinstance(saliency_value, str)
        or type(num_groups_value) is not int
    ):
        raise TimingAdapterError(
            "GuidedGPTQ artifact evidence is malformed"
        )

    model_value = _option(rendered.argv, "--model")
    dataset = _option(rendered.argv, "--dataset")
    nsamples_value = _option(rendered.argv, "--nsamples")
    seq_len_value = _option(rendered.argv, "--seq_len")
    seed_value = _option(rendered.argv, "--seed")
    rotation_seed_value = _option(rendered.argv, "--rotation_seed")
    groups_value = _option(rendered.argv, "--num_groups")
    if None in {
        model_value,
        dataset,
        nsamples_value,
        seq_len_value,
        seed_value,
        rotation_seed_value,
        groups_value,
    }:
        raise TimingAdapterError(
            "GuidedGPTQ producer command lacks cache-identity options"
        )
    try:
        nsamples = int(str(nsamples_value))
        seq_len = int(str(seq_len_value))
        seed = int(str(seed_value))
        rotation_seed = int(str(rotation_seed_value))
        num_groups = int(str(groups_value))
    except ValueError as exc:
        raise TimingAdapterError(
            "GuidedGPTQ cache identity options are not integers"
        ) from exc
    if num_groups != num_groups_value:
        raise TimingAdapterError(
            "GuidedGPTQ runtime num_groups disagrees with command"
        )

    model_path = _resolve_path(str(model_value), repo_root=repo_root)
    cache_root = _resolve_path(
        str(plan["legacy_cache_root"]), repo_root=repo_root
    )
    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))
    import validate_guided_saliency as validator

    expected_saliency = validator.expected_saliency_dir(
        cache_root=cache_root,
        model=model_path,
        dataset=str(dataset),
        nsamples=nsamples,
        seq_len=seq_len,
        seed=seed,
        rotation_seed=rotation_seed,
        num_groups=num_groups,
    ).resolve(strict=False)
    actual_saliency = _resolve_path(
        saliency_value, repo_root=repo_root
    )
    if actual_saliency != expected_saliency:
        raise TimingAdapterError(
            "GuidedGPTQ saliency output path disagrees with reviewed cache "
            "identity"
        )
    validation = validator.validate(
        model=model_path,
        saliency_dir=actual_saliency,
        nsamples=nsamples,
        seq_len=seq_len,
        num_groups=num_groups,
        include_sha256=True,
    )
    if validation.get("valid") is not True:
        raise TimingAdapterError(
            "GuidedGPTQ saliency artifact validation failed: "
            f"{validation.get('errors')!r}"
        )
    artifacts = [
        _file_artifact(
            str(item["path"]),
            repo_root=repo_root,
            allowed_root=actual_saliency,
        )
        for item in validation["files"]
    ]
    artifacts.append(
        _file_artifact(
            gradients_value,
            repo_root=repo_root,
            allowed_root=cache_root / "gradients",
        )
    )
    artifacts.sort(key=lambda item: item["path"])
    if len({item["path"] for item in artifacts}) != len(artifacts):
        raise TimingAdapterError(
            "GuidedGPTQ producer artifact paths are not unique"
        )
    cache_identity = _canonical_sha256(
        {
            "schema_version": 1,
            "method": "guided_gptq",
            "model_sha256": identity["model_sha256"],
            "dataset": dataset,
            "nsamples": nsamples,
            "seq_len": seq_len,
            "seed": seed,
            "rotation_seed": rotation_seed,
            "num_groups": num_groups,
            "saliency_path": str(actual_saliency),
        }
    )
    return cache_identity, artifacts


def _producer_reference(
    *,
    plan: Mapping[str, Any],
    consumer_identity: Mapping[str, str],
    model: str,
    method: str,
    repo_root: Path,
    expected_cache_identity: str | None = None,
) -> dict[str, str]:
    output_root = _resolve_path(
        str(plan["output_root"]), repo_root=repo_root
    )
    candidates: list[dict[str, Any]] = []
    if output_root.is_dir():
        for manifest_path in sorted(
            output_root.rglob(base_executor.MANIFEST_FILENAME)
        ):
            try:
                manifest = _load_json_no_duplicates(manifest_path)
                if manifest.get("status") != "succeeded":
                    continue
                component = base_gpu_hours._parse_component(
                    manifest, manifest_path
                )
            except (
                TimingAdapterError,
                base_gpu_hours.GPUHourError,
                OSError,
                KeyError,
            ):
                continue
            component_identity = component["identity"]
            if (
                component["kind"] == "shared_precompute"
                and component["method"] == method
                and component["model"] == model
                and component["target_phase"] == "final"
                and component_identity["plan_sha256"]
                == consumer_identity["plan_sha256"]
                and component_identity["model_sha256"]
                == consumer_identity["model_sha256"]
                and component_identity["numerical_source_sha256"]
                == consumer_identity["numerical_source_sha256"]
                and (
                    expected_cache_identity is None
                    or component["cache_identity_sha256"]
                    == expected_cache_identity
                )
            ):
                for artifact in component["artifacts"]:
                    if _sha256_file(Path(artifact["path"])) != artifact["sha256"]:
                        raise TimingAdapterError(
                            "shared producer artifact changed after timing: "
                            f"{artifact['path']}"
                        )
                candidates.append(component)
    if len(candidates) != 1:
        raise TimingAdapterError(
            "formal quantization requires exactly one validated same-plan/"
            f"model/source {method} producer for {model}; "
            f"found {len(candidates)}"
        )
    producer = candidates[0]
    return {
        "component_id": producer["component_id"],
        "producer_execution_id": producer["execution_id"],
        "cache_identity_sha256": producer["cache_identity_sha256"],
        "artifact_set_sha256": producer["artifact_set_sha256"],
    }


def _build_component(
    *,
    manifest: Mapping[str, Any],
    rendered: base_runner.RenderedCommand,
    spec: TimingSpec,
    plan: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    repo_root: Path,
) -> dict[str, Any]:
    if any(row.get("complete") is not True for row in rows):
        raise TimingAdapterError(
            "a component cannot be built from incomplete rank timing"
        )
    identity = _manifest_identity(manifest)
    rank_timings = [
        {
            "rank": int(row["rank"]),
            "timing_status": "complete",
            "started_monotonic_ns": int(row["started_monotonic_ns"]),
            "ended_monotonic_ns": int(row["ended_monotonic_ns"]),
            "elapsed_seconds": float(row["elapsed_seconds"]),
        }
        for row in rows
    ]
    wall_seconds = max(
        item["elapsed_seconds"] for item in rank_timings
    )
    shared_refs: list[dict[str, str]] = []
    artifacts: list[dict[str, str]] = []
    cache_outcome: str | None = None
    cache_identity: str | None = None
    artifact_set_sha: str | None = None

    if spec.method == "realq":
        cache_key, artifacts = _realq_cache_evidence(
            rows=rows,
            spec=spec,
            rendered=rendered,
            repo_root=repo_root,
        )
        cache_identity = _canonical_sha256(
            {
                "schema_version": 1,
                "method": "realq",
                "model": spec.model,
                "target_phase": "final",
                "cache_key": cache_key,
            }
        )
        if spec.kind == "shared_precompute":
            cache_outcome = "computed"
            artifact_set_sha = _canonical_sha256(artifacts)
        else:
            cache_identity_for_consumer = cache_identity
            cache_identity = None
            shared_refs = [
                _producer_reference(
                    plan=plan,
                    consumer_identity=identity,
                    model=spec.model,
                    method="realq",
                    repo_root=repo_root,
                    expected_cache_identity=cache_identity_for_consumer,
                )
            ]
    elif spec.kind == "shared_precompute":
        cache_identity, artifacts = _guided_artifacts(
            rows=rows,
            rendered=rendered,
            plan=plan,
            identity=identity,
            repo_root=repo_root,
        )
        cache_outcome = "computed"
        artifact_set_sha = _canonical_sha256(artifacts)
    elif spec.method == "guided_gptq":
        if any(
            row["cache_lookups"] or row["cache_writes"] or row["artifacts"]
            for row in rows
        ):
            raise TimingAdapterError(
                "GuidedGPTQ consumer evidence unexpectedly claims producer "
                "activity"
            )
        shared_refs = [
            _producer_reference(
                plan=plan,
                consumer_identity=identity,
                model=spec.model,
                method="guided_gptq",
                repo_root=repo_root,
            )
        ]
    elif spec.method == "gptaq":
        if any(
            row["cache_lookups"] or row["cache_writes"] or row["artifacts"]
            for row in rows
        ):
            raise TimingAdapterError(
                "GPTAQ evidence unexpectedly claims shared producer activity"
            )
    else:
        raise TimingAdapterError(
            f"cannot build timing component for {spec.method!r}"
        )

    component: dict[str, Any] = {
        "kind": spec.kind,
        "stage": spec.stage,
        "method": spec.method,
        "model": spec.model,
        "setting": spec.setting,
        "phase": spec.phase,
        "target_phase": spec.target_phase,
        "included_segments": INCLUDED_SEGMENTS[(spec.kind, spec.method)],
        "excluded_segments": list(EXCLUDED_SEGMENTS),
        "clock": "time.perf_counter_ns",
        "timing_status": "complete",
        "complete": True,
        "synchronized_start": True,
        "synchronized_end": True,
        "rank_timings": rank_timings,
        "observed_rank_count": len(rank_timings),
        "rank_coverage_complete": True,
        "missing_ranks": [],
        "wall_seconds": wall_seconds,
        "gpu_indices": list(spec.gpu_indices),
        "gpu_count": len(spec.gpu_indices),
        "gpu_hour_status": "exact",
        "allocated_gpu_hours": (
            wall_seconds * len(spec.gpu_indices) / 3600.0
        ),
        "gpu_hours_lower_bound": (
            wall_seconds * len(spec.gpu_indices) / 3600.0
        ),
        "shared_precompute_refs": shared_refs,
        "cache_outcome": cache_outcome,
        "cache_identity_sha256": cache_identity,
        "artifact_set_sha256": artifact_set_sha,
        "artifacts": artifacts,
    }
    component["component_id"] = (
        base_gpu_hours.component_identity_sha256(
            scope_id=SCOPE_ID,
            identity=identity,
            component=component,
        )
    )
    return component


def _build_partial_component(
    *,
    manifest: Mapping[str, Any],
    spec: TimingSpec,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build failed-attempt spend from observed algorithm intervals only.

    Missing ranks and ranks that never reached the algorithm boundary add no
    fabricated process time.  Exact allocated GPU-hours are admitted only
    when every allocated rank supplied evidence and either every rank crossed
    the synchronized start boundary or every rank proves ``not_started``.
    Otherwise the component carries only the sum of observed per-rank
    intervals as a lower bound.
    """

    if not rows:
        raise TimingAdapterError(
            "partial timing requires at least one strict rank evidence file"
        )
    identity = _manifest_identity(manifest)
    rank_timings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for row in rows:
        timing_status = str(row["timing_status"])
        elapsed = (
            float(row["elapsed_seconds"])
            if timing_status in {"partial", "complete"}
            else 0.0
        )
        rank_timings.append(
            {
                "rank": int(row["rank"]),
                "timing_status": timing_status,
                "started_monotonic_ns": row["started_monotonic_ns"],
                "ended_monotonic_ns": row["ended_monotonic_ns"],
                "elapsed_seconds": elapsed,
            }
        )
        if timing_status != "complete":
            errors.append(
                {
                    "rank": int(row["rank"]),
                    "error": str(row["error"]),
                }
            )
    rank_timings.sort(key=lambda item: item["rank"])
    errors.sort(key=lambda item: item["rank"])
    observed_ranks = {int(item["rank"]) for item in rank_timings}
    expected_ranks = set(range(len(spec.gpu_indices)))
    missing_ranks = sorted(expected_ranks - observed_ranks)
    rank_coverage_complete = not missing_ranks
    all_started = (
        rank_coverage_complete
        and all(
            item["timing_status"] in {"partial", "complete"}
            for item in rank_timings
        )
    )
    all_not_started = (
        rank_coverage_complete
        and all(
            item["timing_status"] == "not_started"
            for item in rank_timings
        )
    )
    gpu_hour_exact = all_started or all_not_started
    wall_seconds = max(
        float(item["elapsed_seconds"]) for item in rank_timings
    )
    lower_bound_gpu_hours = sum(
        float(item["elapsed_seconds"]) for item in rank_timings
    ) / 3600.0
    allocated_gpu_hours = (
        wall_seconds * len(spec.gpu_indices) / 3600.0
        if gpu_hour_exact
        else None
    )
    gpu_hours_lower_bound = (
        allocated_gpu_hours
        if allocated_gpu_hours is not None
        else lower_bound_gpu_hours
    )
    timing_status = (
        "partial"
        if any(
            item["timing_status"] in {"partial", "complete"}
            for item in rank_timings
        )
        else "not_started"
    )
    component: dict[str, Any] = {
        "kind": spec.kind,
        "stage": spec.stage,
        "method": spec.method,
        "model": spec.model,
        "setting": spec.setting,
        "phase": spec.phase,
        "target_phase": spec.target_phase,
        "included_segments": INCLUDED_SEGMENTS[(spec.kind, spec.method)],
        "excluded_segments": list(EXCLUDED_SEGMENTS),
        "clock": "time.perf_counter_ns",
        "timing_status": timing_status,
        "complete": False,
        "synchronized_start": (
            len(rows) == len(spec.gpu_indices)
            and all(
                row.get("synchronized_start") is True for row in rows
            )
        ),
        "synchronized_end": False,
        "rank_timings": rank_timings,
        "observed_rank_count": len(rank_timings),
        "rank_coverage_complete": rank_coverage_complete,
        "missing_ranks": missing_ranks,
        "wall_seconds": wall_seconds,
        "gpu_indices": list(spec.gpu_indices),
        "gpu_count": len(spec.gpu_indices),
        "gpu_hour_status": (
            "exact" if gpu_hour_exact else "lower_bound"
        ),
        "allocated_gpu_hours": allocated_gpu_hours,
        "gpu_hours_lower_bound": gpu_hours_lower_bound,
        "shared_precompute_refs": [],
        "cache_outcome": (
            "failed" if spec.kind == "shared_precompute" else None
        ),
        "cache_identity_sha256": None,
        "artifact_set_sha256": None,
        "artifacts": [],
        "partial_errors": errors,
    }
    component["component_id"] = (
        base_gpu_hours.component_identity_sha256(
            scope_id=SCOPE_ID,
            identity=identity,
            component=component,
        )
    )
    return component


def _snapshot_sources(
    *,
    output_dir: Path,
    provenance: TimingProvenance,
) -> list[dict[str, str]]:
    snapshot_dir = output_dir / SOURCE_SNAPSHOT_DIRNAME
    snapshots: list[dict[str, str]] = []
    for source in provenance.sources:
        original = Path(source["original_path"])
        raw = _stable_file_bytes(original)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != source["sha256"]:
            raise TimingAdapterError(
                f"timing source drifted before snapshot: {original}"
            )
        destination = snapshot_dir / f"{source['name']}.py"
        if destination.exists():
            existing = _stable_file_bytes(destination)
            if existing != raw:
                raise TimingAdapterError(
                    f"existing timing source snapshot drifted: {destination}"
                )
        else:
            _atomic_write_bytes(destination, raw)
        if _sha256_file(destination) != digest:
            raise TimingAdapterError(
                f"timing source snapshot hash failed: {destination}"
            )
        snapshots.append(
            {
                "name": source["name"],
                "original_path": source["original_path"],
                "snapshot_path": str(destination.relative_to(output_dir)),
                "sha256": digest,
            }
        )
    if _source_set_sha256(snapshots) != provenance.source_set_sha256:
        raise TimingAdapterError(
            "snapshotted timing source-set identity drifted"
        )
    return snapshots


def _snapshot_entrypoint_sources(
    *,
    output_dir: Path,
    provenance: TimingProvenance,
) -> dict[str, Any] | None:
    tree = provenance.entrypoint_source_tree
    if tree is None:
        return None
    snapshot_root = (
        output_dir / SOURCE_SNAPSHOT_DIRNAME / "realq_benchmark"
    )
    snapshots: list[dict[str, str]] = []
    for record in tree["files"]:
        relative = Path(str(record["relative_path"]))
        original = REPO_ROOT / relative
        raw = _stable_file_bytes(original)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != record["sha256"]:
            raise TimingAdapterError(
                f"realq_benchmark source drifted before snapshot: {original}"
            )
        package_relative = relative.relative_to("realq_benchmark")
        destination = snapshot_root / package_relative
        if destination.exists():
            if _stable_file_bytes(destination) != raw:
                raise TimingAdapterError(
                    "existing realq_benchmark source snapshot drifted: "
                    f"{destination}"
                )
        else:
            _atomic_write_bytes(destination, raw)
        snapshots.append(
            {
                "relative_path": str(relative),
                "snapshot_path": str(destination.relative_to(output_dir)),
                "sha256": digest,
            }
        )
    normalized = [
        {
            "relative_path": item["relative_path"],
            "sha256": item["sha256"],
        }
        for item in snapshots
    ]
    combined = _canonical_sha256(normalized)
    if combined != tree["combined_sha256"]:
        raise TimingAdapterError(
            "snapshotted realq_benchmark source tree identity drifted"
        )
    return {
        "algorithm": tree["algorithm"],
        "file_count": tree["file_count"],
        "combined_sha256": combined,
        "files": snapshots,
    }


def _build_sidecar(
    *,
    manifest: Mapping[str, Any],
    component: Mapping[str, Any],
    provenance: TimingProvenance,
    snapshots: Sequence[Mapping[str, str]],
    entrypoint_snapshot: Mapping[str, Any] | None,
    evidence_files: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "scope_id": SCOPE_ID,
        "identity": _manifest_identity(manifest),
        "manifest_status": manifest["status"],
        "timing_provenance": {
            "schema_version": SCHEMA_VERSION,
            "source_set_sha256": provenance.source_set_sha256,
            "sources": [dict(item) for item in snapshots],
            "formal_entrypoint_source_tree": (
                dict(entrypoint_snapshot)
                if entrypoint_snapshot is not None
                else None
            ),
            "rank_evidence": [dict(item) for item in evidence_files],
        },
        "instrumentation": {
            "path": str(SITE_PATH),
            "sha256": next(
                item["sha256"]
                for item in provenance.sources
                if item["name"] == "formal_timing_sitecustomize"
            ),
        },
        "component": dict(component),
    }


def _pin_sidecar(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    sidecar: Mapping[str, Any],
    provenance: TimingProvenance,
) -> dict[str, Any]:
    path = manifest_path.parent / SIDECAR_FILENAME
    if path.exists():
        existing = _load_json_no_duplicates(path)
        if existing != sidecar:
            raise TimingAdapterError(
                f"existing timing sidecar drifted: {path}"
            )
    else:
        _atomic_write_json(path, sidecar)
    digest = _sha256_file(path)
    manifest["phase_timing"] = {
        "schema_version": SCHEMA_VERSION,
        "scope_id": SCOPE_ID,
        "path": SIDECAR_FILENAME,
        "sha256": digest,
        "stable_during_hash": _sha256_file(path) == digest,
        "source_set_sha256": provenance.source_set_sha256,
    }
    manifest.pop("phase_timing_error", None)
    base_executor.atomic_write_manifest(manifest_path, manifest)
    return manifest


def _quantized_checkpoint_artifact(
    *,
    rendered: base_runner.RenderedCommand,
    spec: TimingSpec,
    repo_root: Path,
) -> dict[str, Any] | None:
    if spec.kind != "quantization":
        return None
    raw_path = _option(rendered.argv, "--save_qmodel_path")
    if raw_path is None:
        raise TimingAdapterError(
            "formal quantization completed without --save_qmodel_path"
        )
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve(strict=False)
    expected = _checkpoint_path(rendered)
    if not expected.is_absolute():
        expected = repo_root / expected
    if path != expected.resolve(strict=False):
        raise TimingAdapterError(
            "saved checkpoint path disagrees with formal attempt identity"
        )
    if not path.is_file():
        raise TimingAdapterError(
            f"formal quantized checkpoint is missing: {path}"
        )
    before = path.stat()
    if before.st_size <= 0:
        raise TimingAdapterError(
            f"formal quantized checkpoint is empty: {path}"
        )
    sampled_sha, hash_mode = base_executor._sampled_file_sha256(path)
    after = path.stat()
    stable = (
        before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
        and before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
    )
    if not stable:
        raise TimingAdapterError(
            f"formal quantized checkpoint changed while sampled: {path}"
        )
    temporary_files = sorted(
        str(item)
        for item in path.parent.glob(f".{path.name}.*.tmp")
        if item.is_file()
    )
    if temporary_files:
        raise TimingAdapterError(
            "formal checkpoint directory retains temporary artifacts: "
            f"{temporary_files!r}"
        )
    return {
        "schema_version": 1,
        "path": str(path),
        "size_bytes": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "device": before.st_dev,
        "inode": before.st_ino,
        "sampled_sha256": sampled_sha,
        "hash_mode": hash_mode,
        "stable_during_hash": True,
        "save_excluded_from_gpu_hour": True,
        "safe_reload_status": "pending",
    }


def finalize_execution(
    result: base_executor.ExecutionResult,
    *,
    rendered: base_runner.RenderedCommand,
    spec: TimingSpec,
    provenance: TimingProvenance,
    plan: Mapping[str, Any],
    repo_root: Path = REPO_ROOT,
) -> base_executor.ExecutionResult:
    """Validate/pin timing, failing closed if a successful child lacks it."""

    if result.manifest_path is None:
        return result
    manifest_path = Path(result.manifest_path)
    manifest = _load_json_no_duplicates(manifest_path)
    original_succeeded = (
        manifest.get("status") == "succeeded"
        and manifest.get("exit_code") == 0
    )
    try:
        if manifest.get("run_id") != spec.run_id:
            raise TimingAdapterError(
                "execution manifest run_id disagrees with timing spec"
            )
        _verify_sources_unchanged(provenance)
        _validate_effective_environment(manifest, provenance)
        rows, evidence_files = _rank_evidence(
            output_dir=manifest_path.parent,
            spec=spec,
            provenance=provenance,
            # A complete core interval can remain valid when a later,
            # explicitly excluded evaluation step fails.
            require_complete=original_succeeded,
        )
        if (
            len(rows) == len(spec.gpu_indices)
            and all(row.get("complete") is True for row in rows)
        ):
            component = _build_component(
                manifest=manifest,
                rendered=rendered,
                spec=spec,
                plan=plan,
                rows=rows,
                repo_root=repo_root,
            )
        else:
            component = _build_partial_component(
                manifest=manifest,
                spec=spec,
                rows=rows,
            )
        snapshots = _snapshot_sources(
            output_dir=manifest_path.parent,
            provenance=provenance,
        )
        entrypoint_snapshot = _snapshot_entrypoint_sources(
            output_dir=manifest_path.parent,
            provenance=provenance,
        )
        checkpoint_artifact = None
        if original_succeeded:
            checkpoint_artifact = _quantized_checkpoint_artifact(
                rendered=rendered,
                spec=spec,
                repo_root=repo_root,
            )
        if checkpoint_artifact is not None:
            manifest["quantized_checkpoint"] = checkpoint_artifact
        manifest["calibration_protocol"] = _calibration_protocol(rendered)
        sidecar = _build_sidecar(
            manifest=manifest,
            component=component,
            provenance=provenance,
            snapshots=snapshots,
            entrypoint_snapshot=entrypoint_snapshot,
            evidence_files=evidence_files,
        )
        manifest = _pin_sidecar(
            manifest=manifest,
            manifest_path=manifest_path,
            sidecar=sidecar,
            provenance=provenance,
        )
    except (
        TimingAdapterError,
        base_gpu_hours.GPUHourError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        error = f"{type(exc).__name__}: {exc}"
        if original_succeeded:
            manifest["status"] = "wrapper_failed"
            manifest["exit_code"] = 2
            process = manifest.get("process")
            if isinstance(process, dict):
                process["wrapper_error"] = (
                    "formal phase timing finalization failed: " + error
                )
            manifest["phase_timing_error"] = error
            base_executor.atomic_write_manifest(manifest_path, manifest)
            return base_executor.ExecutionResult(
                manifest=manifest,
                exit_code=2,
                manifest_path=manifest_path,
                log_path=result.log_path,
            )
        manifest["phase_timing_error"] = error
        base_executor.atomic_write_manifest(manifest_path, manifest)

    return base_executor.ExecutionResult(
        manifest=manifest,
        exit_code=result.exit_code,
        manifest_path=manifest_path,
        log_path=result.log_path,
    )


__all__ = [
    "BASE_CAMPAIGN_PATH",
    "ENV_KEYS",
    "EXCLUDED_SEGMENTS",
    "INCLUDED_SEGMENTS",
    "REPO_ROOT",
    "SCHEMA_VERSION",
    "SCOPE_ID",
    "SITE_DIR",
    "SITE_PATH",
    "SOURCE_PATHS",
    "TIMED_CAMPAIGN_PATH",
    "TIMED_EXECUTOR_PATH",
    "TimingAdapterError",
    "TimingProvenance",
    "TimingSpec",
    "child_environment",
    "finalize_execution",
    "instrument_rendered",
    "prepare_formal_rendered",
    "spec_from_executor_args",
    "timing_source_set_identity",
]
