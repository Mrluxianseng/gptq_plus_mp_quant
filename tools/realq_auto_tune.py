#!/usr/bin/env python3
"""Parallel, resume-safe REAL-Q learning-rate tuner.

The search has two phases:

1. Evaluate ``0, base_lr * 2**k`` in batches of ``parallelism`` until an
   evaluated point is strictly better than its immediate neighbours.
2. Split that bracket into ``parallelism + 1`` equal intervals, evaluate the
   interior points in parallel, and repeat until both neighbour KL gaps are at
   most the configured tolerance (2% by default).

A dedicated producer creates the static Stage-0 and FP-reference caches.
Every LR consumer requires cache hits, so no concurrent trial is allowed to
silently repeat precompute.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
AUDITED_REPOSITORY_VENVS = (
    REPO_ROOT / ".venv",
    REPO_ROOT / ".venv.py312-broken-20260729",
)
STATE_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
PROTOCOL_VERSION = "realq_auto_lr_v5"
TUNING_DATASET = "wikitext2"
CONTROLLED_MAX_TRIAL_LAUNCHES = 20
ZERO_BOUNDARY_MIN_HIGHER_EVIDENCE = 3


def validate_repository_python(python: Path) -> Path:
    """Allow only the canonical venv or the audited Canoe compatibility venv."""

    value = Path(os.path.abspath(os.fspath(python.expanduser())))
    roots = [Path(os.path.abspath(os.fspath(root))) for root in AUDITED_REPOSITORY_VENVS]
    if not any(value == root / "bin" / "python" for root in roots):
        raise TunerError(
            f"execution must use an audited repository virtual environment, got {value}"
        )
    if not value.is_file():
        raise TunerError(f"python executable does not exist: {value}")
    return value

# Canoe and other schedulers commonly inject torchrun variables into every
# process in a pod.  Our producer and LR workers are deliberately independent
# bare-Python, world-size-1 processes; inheriting one shared MASTER_PORT makes
# them race for the same TCPStore.  Strip the complete torchrun/elastic context
# before starting RealQ.  A multi-GPU formal run invokes torchrun itself, which
# establishes a fresh context for its children.
TORCH_DISTRIBUTED_ENV = frozenset(
    {
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "GROUP_WORLD_SIZE",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_ERROR_FILE",
    }
)

EXACT_KL_RE = re.compile(
    r"Exact\s+KL&PPL\s+on\s+(?P<dataset>[^:\s]+)\s*:\s*"
    r"(?P<kl>[-+0-9.eE]+)\s*,\s*(?P<ppl>[-+0-9.eE]+)",
    re.IGNORECASE,
)
STATIC_CACHE_RE = re.compile(
    r"(?P<path>\S+_world1_rank0\.pt)"
)
REFERENCE_CACHE_RE = re.compile(
    r"(?:at|from)\s+(?P<path>\S+\.cache)", re.IGNORECASE
)


class TunerError(RuntimeError):
    """The tuner cannot continue without weakening the requested protocol."""


@dataclass(frozen=True)
class Point:
    lr: float
    kl: float
    ppl: float | None = None


@dataclass(frozen=True)
class Bracket:
    left: Point
    best: Point
    right: Point

    def as_dict(self) -> dict[str, dict[str, float | None]]:
        return {
            name: {
                "lr": point.lr,
                "kl": point.kl,
                "ppl": point.ppl,
            }
            for name, point in (
                ("left", self.left),
                ("best", self.best),
                ("right", self.right),
            )
        }


@dataclass(frozen=True)
class ZeroBoundaryEvidence:
    """Auditable evidence for an optimum at the physical LR lower bound."""

    best: Point
    higher: tuple[Point, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "physical_lower_bound": 0.0,
            "best": dataclasses.asdict(self.best),
            "higher_evidence": [dataclasses.asdict(point) for point in self.higher],
            "minimum_higher_evidence": ZERO_BOUNDARY_MIN_HIGHER_EVIDENCE,
        }


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TunerError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TunerError(f"expected a JSON object at {path}")
    return payload


def normalize_lr(value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"learning rate must be finite and non-negative: {value}")
    if value == 0:
        return 0.0
    return float(format(value, ".15g"))


def lr_key(value: float) -> str:
    return format(normalize_lr(value), ".15g")


def lr_slug(value: float) -> str:
    return "lr_" + re.sub(r"[^0-9A-Za-z]+", "_", lr_key(value)).strip("_")


def quarter_layer_count(total_layers: int) -> int:
    if total_layers <= 0:
        raise ValueError("total_layers must be positive")
    return (total_layers + 3) // 4


def _layer_count_from_mapping(config: Mapping[str, Any]) -> int | None:
    for name in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        value = config.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    text_config = config.get("text_config")
    if isinstance(text_config, Mapping):
        return _layer_count_from_mapping(text_config)
    return None


def load_model_layer_count(model: str) -> int:
    model_path = Path(model).expanduser()
    config_path = model_path / "config.json"
    if config_path.is_file():
        payload = read_json(config_path)
        count = _layer_count_from_mapping(payload)
        if count is not None:
            return count
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    except Exception as exc:  # transformers exposes several source-specific types
        raise TunerError(f"cannot load model config for {model!r}: {exc}") from exc
    for name in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        for name in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
            value = getattr(text_config, name, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    raise TunerError(f"model config for {model!r} does not expose a layer count")


def expansion_candidates(
    cursor: int,
    count: int,
    *,
    base_lr: float = 1e-6,
) -> list[float]:
    """Return a slice of ``[0, base, 2*base, 4*base, ...]``."""
    if cursor < 0 or count <= 0 or base_lr <= 0:
        raise ValueError("invalid expansion cursor/count/base_lr")
    values: list[float] = []
    for ordinal in range(cursor, cursor + count):
        value = 0.0 if ordinal == 0 else math.ldexp(base_lr, ordinal - 1)
        if not math.isfinite(value):
            raise TunerError("exponential LR expansion overflowed")
        values.append(normalize_lr(value))
    return values


def _sorted_unique_points(points: Sequence[Point]) -> list[Point]:
    by_lr: dict[str, Point] = {}
    for point in points:
        key = lr_key(point.lr)
        previous = by_lr.get(key)
        if previous is not None and previous.kl != point.kl:
            raise TunerError(f"conflicting results for lr={key}")
        if not math.isfinite(point.kl) or point.kl < 0:
            raise TunerError(f"invalid KL for lr={key}: {point.kl}")
        by_lr[key] = point
    return sorted(by_lr.values(), key=lambda point: point.lr)


def strict_bracket(
    points: Sequence[Point],
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> Bracket | None:
    """Pick the lowest-KL strictly bracketed point in the requested range."""
    ordered = _sorted_unique_points(points)
    if lower is not None:
        ordered = [point for point in ordered if point.lr >= lower]
    if upper is not None:
        ordered = [point for point in ordered if point.lr <= upper]
    candidates: list[Bracket] = []
    for index in range(1, len(ordered) - 1):
        left, middle, right = ordered[index - 1 : index + 2]
        if middle.kl < left.kl and middle.kl < right.kl:
            candidates.append(Bracket(left, middle, right))
    if not candidates:
        return None
    return min(candidates, key=lambda value: (value.best.kl, value.best.lr))


def neighbour_gaps(bracket: Bracket) -> tuple[float, float]:
    if bracket.best.kl <= 0:
        if bracket.left.kl == bracket.best.kl or bracket.right.kl == bracket.best.kl:
            return math.inf, math.inf
        return math.inf, math.inf
    return (
        (bracket.left.kl - bracket.best.kl) / bracket.best.kl,
        (bracket.right.kl - bracket.best.kl) / bracket.best.kl,
    )


def converged(bracket: Bracket, tolerance: float = 0.02) -> bool:
    if not 0 < tolerance < 1:
        raise ValueError("tolerance must be in (0, 1)")
    left_gap, right_gap = neighbour_gaps(bracket)
    def within(value: float) -> bool:
        # Decimal text such as 1.02 is not exactly representable as binary
        # float. Treat only machine-rounding at the inclusive boundary as
        # equal; this is many orders below any reported Exact-KL precision.
        return value < tolerance or math.isclose(
            value,
            tolerance,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )

    return (
        bracket.left.kl > bracket.best.kl
        and bracket.right.kl > bracket.best.kl
        and within(left_gap)
        and within(right_gap)
    )


def refinement_candidates(
    left: float,
    right: float,
    parallelism: int,
    *,
    existing: Sequence[float] = (),
) -> list[float]:
    if parallelism <= 0:
        raise ValueError("parallelism must be positive")
    if not 0 <= left < right:
        raise ValueError("refinement interval must satisfy 0 <= left < right")
    known = {lr_key(value) for value in existing}
    width = right - left
    values = [
        normalize_lr(left + width * index / (parallelism + 1))
        for index in range(1, parallelism + 1)
    ]
    return [value for value in values if lr_key(value) not in known]


def parse_controlled_candidates(raw: str) -> list[float]:
    values = [normalize_lr(float(item.strip())) for item in raw.split(",") if item.strip()]
    if not values:
        raise TunerError("controlled candidate list must not be empty")
    if len({lr_key(value) for value in values}) != len(values):
        raise TunerError("controlled candidate list contains duplicates")
    if values != sorted(values):
        raise TunerError("controlled candidates must be ordered from low to high")
    return values


def controlled_selection(points: Sequence[Point], selected_lr: float) -> Bracket:
    """Validate a human/agent decision against preserved Exact-KL evidence."""

    ordered = _sorted_unique_points(points)
    key = lr_key(selected_lr)
    index = next(
        (position for position, point in enumerate(ordered) if lr_key(point.lr) == key),
        None,
    )
    if index is None:
        raise TunerError(f"selected lr={key} has no successful controlled trial")
    if index == 0 or index == len(ordered) - 1:
        raise TunerError("selected LR needs both lower- and higher-LR evidence")
    selected = ordered[index]
    global_best = min(ordered, key=lambda point: (point.kl, point.lr))
    if selected != global_best:
        raise TunerError(
            f"selected lr={key} is not the lowest observed Exact KL; "
            f"best is lr={lr_key(global_best.lr)}"
        )
    bracket = Bracket(ordered[index - 1], selected, ordered[index + 1])
    if not (
        bracket.best.kl < bracket.left.kl
        and bracket.best.kl < bracket.right.kl
    ):
        raise TunerError("selected LR is not a strict local Exact-KL minimum")
    return bracket


def controlled_zero_boundary_selection(
    points: Sequence[Point], selected_lr: float
) -> ZeroBoundaryEvidence:
    """Validate a unique optimum at the physical non-negative LR boundary."""

    key = lr_key(selected_lr)
    if key != lr_key(0.0):
        raise TunerError("physical zero-boundary selection requires selected lr=0")
    ordered = _sorted_unique_points(points)
    if not ordered or lr_key(ordered[0].lr) != key:
        raise TunerError("selected lr=0 has no successful controlled trial")
    best = ordered[0]
    higher = tuple(ordered[1:])
    if len(higher) < ZERO_BOUNDARY_MIN_HIGHER_EVIDENCE:
        raise TunerError(
            "physical zero-boundary selection requires at least "
            f"{ZERO_BOUNDARY_MIN_HIGHER_EVIDENCE} distinct higher-LR trials"
        )
    non_worse = [point for point in higher if point.kl <= best.kl]
    if non_worse:
        point = min(non_worse, key=lambda value: (value.kl, value.lr))
        raise TunerError(
            "selected lr=0 is not a unique global Exact-KL minimum; "
            f"lr={lr_key(point.lr)} has KL={point.kl}"
        )
    return ZeroBoundaryEvidence(best=best, higher=higher)


def parse_exact_metric(text: str, dataset: str = TUNING_DATASET) -> tuple[float, float]:
    matches = [
        match
        for match in EXACT_KL_RE.finditer(text)
        if match.group("dataset").lower() == dataset.lower()
    ]
    if not matches:
        raise TunerError(f"no Exact KL&PPL metric found for {dataset}")
    match = matches[-1]
    kl, ppl = float(match.group("kl")), float(match.group("ppl"))
    if not math.isfinite(kl) or kl < 0 or not math.isfinite(ppl) or ppl <= 0:
        raise TunerError(f"invalid Exact KL/PPL for {dataset}: {kl}, {ppl}")
    return kl, ppl


def _strip_options(argv: Sequence[str], names: set[str]) -> list[str]:
    """Remove Config options owned by the orchestrator.

    Config options consume one value, except the two list options. The sole
    action flag (allow_unsafe_legacy_checkpoint) is never stripped here.
    """
    output: list[str] = []
    index = 0
    list_options = {"eval_datasets", "reasoning_tasks"}
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            output.append(token)
            index += 1
            continue
        name = token[2:].split("=", 1)[0]
        if name not in names:
            output.append(token)
            index += 1
            continue
        if "=" in token:
            index += 1
            continue
        index += 1
        if name in list_options:
            while index < len(argv) and not argv[index].startswith("--"):
                index += 1
        elif index < len(argv):
            index += 1
    return output


_TUNING_OWNED = {
    "model",
    "grad_lr",
    "w_groupsize",
    "nsamples",
    "seq_len",
    "bsz",
    "global_loss_bsz",
    "backward_samples",
    "backward_bsz",
    "blocksize",
    "loss_slide_window",
    "lm_eval",
    "eval_datasets",
    "static_cache_path",
    "cache_dir",
    "tokens_cache_path",
    "quant_stop_layer",
    "output_dir",
    "exp",
    "exit_after_precompute",
    "require_static_cache_hit",
    "require_reference_cache_hit",
    "skip_eval",
    "skip_kl_ppl_eval",
    "reasoning_eval",
    "save_qmodel_path",
    "load_qmodel_path",
}

_MAIN_OWNED = {
    "model",
    "grad_lr",
    "static_cache_path",
    "cache_dir",
    "tokens_cache_path",
    "quant_stop_layer",
    "output_dir",
    "exp",
    "exit_after_precompute",
    "require_static_cache_hit",
    "require_reference_cache_hit",
}


def tuning_realq_args(
    base_args: Sequence[str],
    *,
    model: str,
    grad_lr: float,
    quant_stop_layer: int,
    static_cache_path: Path,
    cache_dir: Path,
    tokens_cache_path: Path,
    output_dir: Path,
    exp: str,
    w_groupsize: int = -1,
    global_loss_bsz: int = 8,
    producer: bool = False,
) -> list[str]:
    argv = _strip_options(base_args, _TUNING_OWNED)
    argv.extend(
        [
            "--model", model,
            "--grad_lr", lr_key(grad_lr),
            "--w_groupsize", str(w_groupsize),
            "--nsamples", "256",
            "--seq_len", "2048",
            "--bsz", "32",
            "--global_loss_bsz", str(global_loss_bsz),
            "--backward_samples", "16",
            "--backward_bsz", "16",
            "--blocksize", "256",
            "--loss_slide_window", "false",
            "--lm_eval", "false",
            "--eval_datasets", TUNING_DATASET,
            "--skip_eval", "false",
            "--skip_kl_ppl_eval", "false",
            "--reasoning_eval", "false",
            "--quant_stop_layer", str(quant_stop_layer),
            "--static_cache_path", str(static_cache_path),
            "--cache_dir", str(cache_dir),
            "--tokens_cache_path", str(tokens_cache_path),
            "--output_dir", str(output_dir),
            "--exp", exp,
            "--exit_after_precompute", "true" if producer else "false",
            "--require_static_cache_hit", "false" if producer else "true",
            "--require_reference_cache_hit", "false" if producer else "true",
        ]
    )
    return argv


def main_realq_args(
    base_args: Sequence[str],
    *,
    model: str,
    grad_lr: float,
    static_cache_path: Path,
    cache_dir: Path,
    tokens_cache_path: Path,
    output_dir: Path,
) -> list[str]:
    argv = _strip_options(base_args, _MAIN_OWNED)
    argv.extend(
        [
            "--model", model,
            "--grad_lr", lr_key(grad_lr),
            "--static_cache_path", str(static_cache_path),
            "--cache_dir", str(cache_dir),
            "--tokens_cache_path", str(tokens_cache_path),
            "--output_dir", str(output_dir),
            "--exp", "main",
            "--exit_after_precompute", "false",
            "--require_static_cache_hit", "false",
            "--require_reference_cache_hit", "false",
        ]
    )
    return argv


def _validate_realq_args(argv: Sequence[str]) -> None:
    from realq.config import parse_cli

    parse_cli(list(argv))


def python_realq_command(python: Path, realq_args: Sequence[str]) -> list[str]:
    return [str(python), "-m", "realq.ptq", *realq_args]


def distributed_realq_command(
    python: Path,
    realq_args: Sequence[str],
    gpu_count: int,
    *,
    master_port: int,
) -> list[str]:
    if gpu_count <= 1:
        return python_realq_command(python, realq_args)
    torchrun = python.parent / "torchrun"
    return [
        str(torchrun),
        "--nproc_per_node", str(gpu_count),
        "--master_port", str(master_port),
        "-m", "realq.ptq",
        *realq_args,
    ]


def _worker_environment(spec: Mapping[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    unset = TORCH_DISTRIBUTED_ENV.union(
        str(key) for key in spec.get("unset_env", ())
    )
    for key in unset:
        env.pop(str(key), None)
    env.update({str(key): str(value) for key, value in spec.get("env", {}).items()})
    return env


def realq_worker_env(
    cuda_visible_devices: str,
    *,
    deterministic_sdpa: bool,
) -> dict[str, str]:
    """Build the explicit environment overrides for one RealQ worker."""
    environment = {
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
        "PYTHONUNBUFFERED": "1",
    }
    if deterministic_sdpa:
        environment["REALQ_DETERMINISTIC_SDPA"] = "1"
    return environment


def _worker(spec_path: Path) -> int:
    spec = read_json(spec_path)
    result_path = Path(spec["result_path"])
    log_path = Path(spec["log_path"])
    if result_path.exists():
        existing = read_json(result_path)
        return 0 if existing.get("status") == "succeeded" else 1
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(item) for item in spec["command"]]
    env = _worker_environment(spec)
    started = utc_now()
    child: subprocess.Popen[bytes] | None = None

    def forward(signum: int, _frame: Any) -> None:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    with log_path.open("wb") as log_handle:
        child = subprocess.Popen(
            command,
            cwd=spec["cwd"],
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        returncode = child.wait()
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "kind": spec["kind"],
        "identity": spec["identity"],
        "spec_sha256": sha256_value(
            {key: value for key, value in spec.items() if key != "created_at"}
        ),
        "command": command,
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "started_at": started,
        "finished_at": utc_now(),
        "returncode": returncode,
        "log_path": str(log_path),
    }
    if returncode == 0:
        if spec["kind"] == "trial":
            try:
                kl, ppl = parse_exact_metric(
                    log_path.read_text(encoding="utf-8", errors="replace"),
                    spec.get("metric_dataset", TUNING_DATASET),
                )
            except TunerError as exc:
                result.update(status="failed", error=str(exc))
            else:
                result.update(
                    status="succeeded",
                    lr=float(spec["lr"]),
                    kl=kl,
                    ppl=ppl,
                )
        else:
            result["status"] = "succeeded"
    else:
        result.update(
            status="failed",
            error=f"child exited with status {returncode}",
        )
    atomic_json(result_path, result)
    return 0 if result["status"] == "succeeded" else 1


def _parse_cuda_ids(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values or len(set(values)) != len(values):
        raise TunerError("CUDA ids must be a non-empty unique comma-separated list")
    if any(not value.isdigit() for value in values):
        raise TunerError(f"CUDA ids must be numeric physical indices: {raw!r}")
    return values


def _gpu_processes() -> dict[str, list[dict[str, str]]]:
    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    uuid_to_index: dict[str, str] = {}
    for line in gpu_query.stdout.splitlines():
        if not line.strip():
            continue
        index, uuid = [part.strip() for part in line.split(",", 1)]
        uuid_to_index[uuid] = index
    process_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    output: dict[str, list[dict[str, str]]] = {index: [] for index in uuid_to_index.values()}
    for line in process_query.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            continue
        uuid, pid, name, memory = parts
        index = uuid_to_index.get(uuid)
        if index is not None:
            output.setdefault(index, []).append(
                {"pid": pid, "process_name": name, "used_memory_mib": memory}
            )
    return output


def require_idle_gpus(cuda_ids: Sequence[str]) -> None:
    try:
        processes = _gpu_processes()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TunerError(f"cannot inspect GPUs with nvidia-smi: {exc}") from exc
    missing = [gpu for gpu in cuda_ids if gpu not in processes]
    if missing:
        raise TunerError(f"requested CUDA ids are not visible: {missing}")
    busy = {gpu: processes[gpu] for gpu in cuda_ids if processes[gpu]}
    if busy:
        raise TunerError(f"selected GPUs have existing compute processes: {busy}")


def _artifact_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = (REPO_ROOT / resolved).resolve()
    if not resolved.is_file():
        raise TunerError(f"expected cache artifact does not exist: {resolved}")
    stat = resolved.stat()
    if stat.st_size <= 0:
        raise TunerError(f"cache artifact is empty: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _validate_artifact_records(records: Sequence[Mapping[str, Any]]) -> None:
    if len(records) < 2:
        raise TunerError("precompute manifest must contain static and reference caches")
    for record in records:
        path = Path(str(record["path"]))
        if not path.is_file():
            raise TunerError(f"completed precompute artifact disappeared: {path}")
        stat = path.stat()
        if (
            stat.st_size != int(record["size_bytes"])
            or stat.st_mtime_ns != int(record["mtime_ns"])
        ):
            raise TunerError(f"completed precompute artifact drifted: {path}")


def _collect_precompute_artifacts(
    log_path: Path,
    static_dir: Path,
    cache_dir: Path,
) -> list[dict[str, Any]]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    static_paths = [Path(match.group("path")) for match in STATIC_CACHE_RE.finditer(text)]
    reference_paths = [
        Path(match.group("path")) for match in REFERENCE_CACHE_RE.finditer(text)
    ]
    if not static_paths:
        static_paths = sorted(static_dir.glob("*_world1_rank0.pt"))
    if not reference_paths:
        reference_paths = sorted((cache_dir / "ref_logits").glob("*.cache"))
    if not static_paths:
        raise TunerError("producer succeeded but no static cache artifact was found")
    if not reference_paths:
        raise TunerError("producer succeeded but no FP reference cache was found")
    # The output root is campaign-specific, so the final file in each class is
    # the producer's artifact even when a failed attempt left an older entry.
    return [_artifact_record(static_paths[-1]), _artifact_record(reference_paths[-1])]


def _run_direct_worker(spec: dict[str, Any], python: Path) -> dict[str, Any]:
    result_path = Path(spec["result_path"])
    spec_path = result_path.with_name("spec.json")
    launch_path = result_path.with_name("launch.json")
    if result_path.exists():
        result = read_json(result_path)
        if result.get("status") != "succeeded":
            raise TunerError(
                f"{spec['kind']} {spec['identity']} has a preserved failed result; "
                f"see {spec['log_path']}"
            )
        return result
    if launch_path.exists():
        launch = read_json(launch_path)
        pid = int(launch.get("worker_pid") or -1)
        alive = False
        if pid > 0:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            except PermissionError:
                alive = True
            else:
                alive = True
        if alive:
            raise TunerError(
                f"{spec['kind']} {spec['identity']} still has worker pid={pid}; "
                "wait for its atomic result before restarting"
            )
        raise TunerError(
            f"{spec['kind']} {spec['identity']} was launched but has no atomic "
            "result; refusing a duplicate launch"
        )
    atomic_json(spec_path, spec)
    launch = {
        "kind": spec["kind"],
        "identity": spec["identity"],
        "status": "launching",
        "worker_pid": None,
        "created_at": utc_now(),
    }
    atomic_json(launch_path, launch)
    process = subprocess.Popen(
        [str(python), str(Path(__file__).resolve()), "_worker", "--spec", str(spec_path)],
        cwd=REPO_ROOT,
    )
    launch.update(status="running", worker_pid=process.pid, started_at=utc_now())
    atomic_json(launch_path, launch)
    returncode = process.wait()
    launch.update(status="finished", returncode=returncode, finished_at=utc_now())
    atomic_json(launch_path, launch)
    if not result_path.is_file():
        raise TunerError(
            f"{spec['kind']} {spec['identity']} worker={returncode} wrote no result"
        )
    result = read_json(result_path)
    if returncode != 0 or result.get("status") != "succeeded":
        raise TunerError(
            f"{spec['kind']} {spec['identity']} failed; see {spec['log_path']}: "
            f"{result.get('error', 'unknown error')}"
        )
    return result


def _trial_spec(
    *,
    lr: float,
    gpu: str,
    command: Sequence[str],
    directory: Path,
    fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "kind": "trial",
        "identity": lr_key(lr),
        "lr": lr,
        "metric_dataset": TUNING_DATASET,
        "protocol_fingerprint": fingerprint,
        "command": list(command),
        "env": realq_worker_env(gpu, deterministic_sdpa=True),
        "unset_env": sorted(TORCH_DISTRIBUTED_ENV),
        "cwd": str(REPO_ROOT),
        "log_path": str(directory / "execution.log"),
        "result_path": str(directory / "result.json"),
        "created_at": utc_now(),
    }


def _load_points(state: Mapping[str, Any]) -> list[Point]:
    points: list[Point] = []
    trials = state.get("trials", {})
    if not isinstance(trials, Mapping):
        raise TunerError("state.trials must be a mapping")
    for result in trials.values():
        if not isinstance(result, Mapping) or result.get("status") != "succeeded":
            continue
        points.append(
            Point(
                lr=float(result["lr"]),
                kl=float(result["kl"]),
                ppl=float(result["ppl"]),
            )
        )
    return points


def _state_payload(
    *,
    fingerprint: str,
    model: str,
    total_layers: int,
    parallelism: int,
) -> dict[str, Any]:
    tuning_layers = quarter_layer_count(total_layers)
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_fingerprint": fingerprint,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "precompute_pending",
        "model": model,
        "total_layers": total_layers,
        "tuning_layers": tuning_layers,
        "quant_stop_layer": tuning_layers - 1,
        "parallelism": parallelism,
        "expansion_cursor": 0,
        "refinement_round": 0,
        "trials": {},
    }


def _save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    atomic_json(path, state)


def _trial_launch_count(state: Mapping[str, Any], trials_dir: Path) -> int:
    """Count successful/current specs plus agent-audited interrupted launches."""
    recoveries = state.get("interrupted_trial_recoveries", [])
    if not isinstance(recoveries, list):
        raise TunerError("state.interrupted_trial_recoveries must be a list")
    return len(list(trials_dir.glob("*/spec.json"))) + len(recoveries)


def _reconcile_trial_results(
    state: dict[str, Any],
    trials_dir: Path,
    fingerprint: str,
) -> None:
    for result_path in sorted(trials_dir.glob("*/result.json")):
        result = read_json(result_path)
        spec = read_json(result_path.with_name("spec.json"))
        if spec.get("protocol_fingerprint") != fingerprint:
            raise TunerError(f"trial protocol fingerprint mismatch: {result_path}")
        identity = str(result.get("identity"))
        if result.get("status") != "succeeded":
            raise TunerError(
                f"previous trial {identity} failed; preserve its evidence and "
                f"inspect {result.get('log_path', result_path)} before retrying"
            )
        key = lr_key(float(result["lr"]))
        state["trials"][key] = result
        state.setdefault("running", {}).pop(key, None)

    unfinished = state.get("running", {})
    if not isinstance(unfinished, Mapping):
        raise TunerError("state.running must be a mapping")
    for key, record in unfinished.items():
        if not isinstance(record, Mapping):
            raise TunerError(f"invalid running record for lr={key}")
        pid = int(record.get("worker_pid") or -1)
        alive = False
        if pid > 0:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            except PermissionError:
                alive = True
            else:
                alive = True
        if alive:
            raise TunerError(
                f"trial lr={key} still has worker pid={pid}; wait for its "
                "atomic result before restarting the controller"
            )
        raise TunerError(
            f"trial lr={key} worker pid={pid} ended without an atomic result; "
            "refusing to relaunch and risk duplicate computation"
        )


def _run_trial_batch(
    *,
    candidates: Sequence[float],
    cuda_ids: Sequence[str],
    python: Path,
    base_args: Sequence[str],
    model: str,
    quant_stop_layer: int,
    static_dir: Path,
    cache_dir: Path,
    tokens_dir: Path,
    trials_dir: Path,
    state: dict[str, Any],
    state_path: Path,
    fingerprint: str,
    skip_gpu_check: bool,
    tuning_w_groupsize: int,
    tuning_global_loss_bsz: int,
) -> None:
    missing = [lr for lr in candidates if lr_key(lr) not in state["trials"]]
    if not missing:
        return
    if len(missing) > len(cuda_ids):
        raise TunerError("candidate batch exceeds available CUDA ids")
    if not skip_gpu_check:
        require_idle_gpus(cuda_ids[: len(missing)])
    workers: list[tuple[float, Path, subprocess.Popen[bytes]]] = []
    for lr, gpu in zip(missing, cuda_ids):
        directory = trials_dir / lr_slug(lr)
        realq_args = tuning_realq_args(
            base_args,
            model=model,
            grad_lr=lr,
            quant_stop_layer=quant_stop_layer,
            static_cache_path=static_dir,
            cache_dir=cache_dir,
            tokens_cache_path=tokens_dir,
            output_dir=directory / "realq_output",
            exp="partial_quant",
            w_groupsize=tuning_w_groupsize,
            global_loss_bsz=tuning_global_loss_bsz,
        )
        _validate_realq_args(realq_args)
        spec = _trial_spec(
            lr=lr,
            gpu=gpu,
            command=python_realq_command(python, realq_args),
            directory=directory,
            fingerprint=fingerprint,
        )
        spec_path = directory / "spec.json"
        atomic_json(spec_path, spec)
        key = lr_key(lr)
        state.setdefault("running", {})[key] = {
            "worker_pid": None,
            "gpu": gpu,
            "status": "launching",
            "started_at": utc_now(),
        }
        _save_state(state_path, state)
        process = subprocess.Popen(
            [str(python), str(Path(__file__).resolve()), "_worker", "--spec", str(spec_path)],
            cwd=REPO_ROOT,
        )
        workers.append((lr, directory, process))
        state["running"][key].update(
            worker_pid=process.pid,
            status="running",
        )
        _save_state(state_path, state)
    failures: list[str] = []
    for lr, directory, process in workers:
        returncode = process.wait()
        result_path = directory / "result.json"
        if not result_path.is_file():
            failures.append(f"lr={lr_key(lr)} worker={returncode} wrote no result")
            continue
        result = read_json(result_path)
        state.setdefault("running", {}).pop(lr_key(lr), None)
        if returncode != 0 or result.get("status") != "succeeded":
            failures.append(
                f"lr={lr_key(lr)}: {result.get('error', f'worker={returncode}')}"
            )
        else:
            state["trials"][lr_key(lr)] = result
        _save_state(state_path, state)
    if failures:
        raise TunerError(
            "one or more LR trials failed (no result was used): " + "; ".join(failures)
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cuda-ids", required=True)
    parser.add_argument("--parallelism", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help=(
            "optional model-scoped cache root shared by compatible campaigns; "
            "defaults to OUTPUT_ROOT/cache"
        ),
    )
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--base-lr", type=float, default=1e-6)
    parser.add_argument(
        "--tuning-w-groupsize",
        type=int,
        default=-1,
        help=(
            "weight group size used by producer and LR trials; defaults to "
            "the historical per-row tuning profile (-1)"
        ),
    )
    parser.add_argument(
        "--tuning-global-loss-bsz",
        type=int,
        default=8,
        help="tuning producer/trial global-loss microbatch; backward batch remains 16",
    )
    parser.add_argument("--kl-tolerance", type=float, default=0.02)
    parser.add_argument("--max-expansion-rounds", type=int, default=20)
    parser.add_argument("--max-refinement-rounds", type=int, default=30)
    parser.add_argument("--run-main", action="store_true")
    parser.add_argument("--main-cuda-ids", default=None)
    parser.add_argument("--skip-gpu-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--controlled-candidates",
        default=None,
        help=(
            "comma-separated, low-to-high LR batch chosen by the supervising "
            "agent; runs only this batch and then returns"
        ),
    )
    parser.add_argument("--controlled-select-lr", type=float, default=None)
    parser.add_argument(
        "--allow-physical-zero-boundary",
        action="store_true",
        help=(
            "allow lr=0 only when it is the unique global Exact-KL minimum "
            "and at least three distinct positive-LR trials are all worse"
        ),
    )
    parser.add_argument("--decision-reason", default=None)
    parser.add_argument(
        "realq_args",
        nargs=argparse.REMAINDER,
        help="formal/main REAL-Q arguments after --; tuning-only values are overridden",
    )
    return parser


def _main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    controlled_candidates = (
        parse_controlled_candidates(args.controlled_candidates)
        if args.controlled_candidates is not None
        else None
    )
    controlled = controlled_candidates is not None or args.controlled_select_lr is not None
    if controlled_candidates is not None and args.controlled_select_lr is not None:
        raise TunerError("controlled candidate and selection modes are mutually exclusive")
    if controlled and args.run_main:
        raise TunerError("controlled tuning cannot use --run-main")
    if args.controlled_select_lr is not None:
        normalize_lr(args.controlled_select_lr)
        if not args.decision_reason or not args.decision_reason.strip():
            raise TunerError("controlled LR selection requires --decision-reason")
    elif args.decision_reason is not None:
        raise TunerError("--decision-reason is only valid with --controlled-select-lr")
    if args.allow_physical_zero_boundary:
        if args.controlled_select_lr is None:
            raise TunerError(
                "--allow-physical-zero-boundary requires --controlled-select-lr"
            )
        if lr_key(args.controlled_select_lr) != lr_key(0.0):
            raise TunerError(
                "--allow-physical-zero-boundary requires --controlled-select-lr 0"
            )
    if args.parallelism <= 0:
        raise TunerError("parallelism must be positive")
    if not math.isfinite(args.base_lr) or args.base_lr <= 0:
        raise TunerError("base-lr must be finite and positive")
    if not math.isfinite(args.kl_tolerance) or not 0 < args.kl_tolerance < 1:
        raise TunerError("kl-tolerance must be in (0, 1)")
    if args.max_expansion_rounds <= 0 or args.max_refinement_rounds <= 0:
        raise TunerError("search round limits must be positive")
    if args.tuning_w_groupsize != -1 and args.tuning_w_groupsize <= 0:
        raise TunerError("tuning-w-groupsize must be -1 or positive")
    if args.tuning_global_loss_bsz <= 0:
        raise TunerError("tuning-global-loss-bsz must be positive")
    if (
        args.tuning_w_groupsize != -1
        and 256 % args.tuning_w_groupsize != 0
    ):
        raise TunerError(
            "tuning-w-groupsize must divide the fixed tuning blocksize 256"
        )
    cuda_ids = _parse_cuda_ids(args.cuda_ids)
    if args.parallelism > len(cuda_ids):
        raise TunerError(
            f"parallelism={args.parallelism} exceeds {len(cuda_ids)} CUDA ids"
        )
    cuda_ids = cuda_ids[: args.parallelism]
    main_cuda_ids = _parse_cuda_ids(args.main_cuda_ids or cuda_ids[0])
    python = Path(os.path.abspath(os.fspath(args.python.expanduser())))
    if not args.dry_run:
        python = validate_repository_python(python)
    total_layers = load_model_layer_count(args.model)
    tuning_layers = quarter_layer_count(total_layers)
    quant_stop_layer = tuning_layers - 1
    model_name = Path(args.model.rstrip("/")).name or "model"
    output_root = (
        args.output_root
        if args.output_root is not None
        else REPO_ROOT / "output" / "realq_auto_tune" / model_name
    ).expanduser().resolve()
    cache_root = (
        args.cache_root.expanduser().resolve()
        if args.cache_root is not None
        else output_root / "cache"
    )
    base_args = list(args.realq_args)
    if base_args and base_args[0] == "--":
        base_args = base_args[1:]
    protocol = {
        "version": PROTOCOL_VERSION,
        "model": args.model,
        "base_args": base_args,
        "parallelism": args.parallelism,
        "base_lr": args.base_lr,
        "kl_tolerance": args.kl_tolerance,
        "total_layers": total_layers,
        "tuning_layers": tuning_layers,
        "quant_stop_layer": quant_stop_layer,
        "tuning_fixed": {
            "w_groupsize": args.tuning_w_groupsize,
            "nsamples": 256,
            "seq_len": 2048,
            "bsz": 32,
            "global_loss_bsz": args.tuning_global_loss_bsz,
            "backward_samples": 16,
            "backward_bsz": 16,
            "blocksize": 256,
            "loss_slide_window": False,
            "lm_eval": False,
            "deterministic_sdpa": True,
        },
    }
    if controlled:
        protocol["controller_mode"] = "manual"
    fingerprint = sha256_value(protocol)
    static_dir = cache_root / "static"
    cache_dir = cache_root / "runtime"
    tokens_dir = cache_root / "tokens"
    state_path = output_root / "state.json"
    trials_dir = output_root / "trials"
    producer_args = tuning_realq_args(
        base_args,
        model=args.model,
        grad_lr=0.0,
        quant_stop_layer=quant_stop_layer,
        static_cache_path=static_dir,
        cache_dir=cache_dir,
        tokens_cache_path=tokens_dir,
        output_dir=output_root / "precompute" / "realq_output",
        exp="producer",
        w_groupsize=args.tuning_w_groupsize,
        global_loss_bsz=args.tuning_global_loss_bsz,
        producer=True,
    )
    _validate_realq_args(producer_args)
    preview = {
        "protocol": protocol,
        "protocol_fingerprint": fingerprint,
        "output_root": str(output_root),
        "cache_root": str(cache_root),
        "cuda_ids": cuda_ids,
        "producer_command": python_realq_command(python, producer_args),
        "first_candidates": expansion_candidates(0, args.parallelism, base_lr=args.base_lr),
        "controlled_candidates": controlled_candidates,
        "controlled_select_lr": args.controlled_select_lr,
        "allow_physical_zero_boundary": args.allow_physical_zero_boundary,
    }
    if args.dry_run:
        print(json.dumps(preview, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    # Retain this handle for the full controller lifetime. A second process
    # must never race the single producer or launch duplicate LR workers.
    controller_lock = (output_root / ".controller.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(controller_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        controller_lock.close()
        raise TunerError(
            f"another auto-tune controller holds {output_root / '.controller.lock'}"
        ) from exc
    protocol_path = output_root / "protocol.json"
    if protocol_path.exists():
        existing_protocol = read_json(protocol_path)
        if sha256_value(existing_protocol) != fingerprint:
            raise TunerError(
                f"output root belongs to a different protocol: {output_root}"
            )
    else:
        atomic_json(protocol_path, protocol)
    if state_path.exists():
        state = read_json(state_path)
        if state.get("protocol_fingerprint") != fingerprint:
            raise TunerError("state protocol fingerprint mismatch")
    else:
        state = _state_payload(
            fingerprint=fingerprint,
            model=args.model,
            total_layers=total_layers,
            parallelism=args.parallelism,
        )
        _save_state(state_path, state)

    _reconcile_trial_results(state, trials_dir, fingerprint)
    _save_state(state_path, state)

    if args.controlled_select_lr is not None:
        if state.get("status") == "tuning_complete":
            existing = float(state.get("selected_lr", -1))
            if lr_key(existing) != lr_key(args.controlled_select_lr):
                raise TunerError("controlled selection conflicts with completed decision")
            return 0
        if args.allow_physical_zero_boundary:
            boundary = controlled_zero_boundary_selection(
                _load_points(state), normalize_lr(args.controlled_select_lr)
            )
            selected = boundary.best
            bracket_payload = None
            left_gap = None
            right_gap = (
                (boundary.higher[0].kl - selected.kl) / selected.kl
                if selected.kl > 0
                else math.inf
            )
            selection_kind = "physical_zero_boundary"
            boundary_payload = boundary.as_dict()
        else:
            bracket = controlled_selection(
                _load_points(state), normalize_lr(args.controlled_select_lr)
            )
            selected = bracket.best
            bracket_payload = bracket.as_dict()
            left_gap, right_gap = neighbour_gaps(bracket)
            selection_kind = "strict_bracket"
            boundary_payload = None
        decision = {
            "schema_version": STATE_SCHEMA_VERSION,
            "protocol_fingerprint": fingerprint,
            "selection_kind": selection_kind,
            "selected_lr": selected.lr,
            "selected_kl": selected.kl,
            "reason": args.decision_reason.strip(),
            "bracket": bracket_payload,
            "physical_zero_boundary": boundary_payload,
            "left_relative_gap": left_gap,
            "right_relative_gap": right_gap,
            "trial_launch_count": _trial_launch_count(state, trials_dir),
            "decided_at": utc_now(),
        }
        atomic_json(output_root / "controlled_decision.json", decision)
        state.update(
            status="tuning_complete",
            selected_lr=selected.lr,
            selected_kl=selected.kl,
            left_relative_gap=left_gap,
            right_relative_gap=right_gap,
            bracket=bracket_payload,
            physical_zero_boundary=boundary_payload,
            controlled_decision=decision,
        )
        _save_state(state_path, state)
        atomic_text(output_root / "best_lr.txt", lr_key(selected.lr) + "\n")
        print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    producer_status_path = output_root / "precompute" / "status.json"
    if producer_status_path.exists():
        producer_status = read_json(producer_status_path)
        if producer_status.get("protocol_fingerprint") != fingerprint:
            raise TunerError("precompute status protocol fingerprint mismatch")
        if producer_status.get("status") != "succeeded":
            raise TunerError("previous precompute producer did not succeed")
        _validate_artifact_records(producer_status.get("artifacts", []))
    else:
        if not args.skip_gpu_check:
            require_idle_gpus([cuda_ids[0]])
        producer_dir = output_root / "precompute"
        spec = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "kind": "precompute",
            "identity": "single_producer",
            "protocol_fingerprint": fingerprint,
            "command": python_realq_command(python, producer_args),
            "env": realq_worker_env(cuda_ids[0], deterministic_sdpa=True),
            "unset_env": sorted(TORCH_DISTRIBUTED_ENV),
            "cwd": str(REPO_ROOT),
            "log_path": str(producer_dir / "execution.log"),
            "result_path": str(producer_dir / "worker_result.json"),
            "created_at": utc_now(),
        }
        worker_result = _run_direct_worker(spec, python)
        artifacts = _collect_precompute_artifacts(
            Path(spec["log_path"]), static_dir, cache_dir
        )
        producer_status = {
            "schema_version": STATE_SCHEMA_VERSION,
            "status": "succeeded",
            "protocol_fingerprint": fingerprint,
            "worker_result": worker_result,
            "artifacts": artifacts,
            "finished_at": utc_now(),
        }
        atomic_json(producer_status_path, producer_status)
    state["precompute"] = producer_status
    if controlled_candidates is not None:
        if state.get("status") == "tuning_complete":
            raise TunerError("controlled tuning is already complete")
        missing = [
            lr for lr in controlled_candidates if lr_key(lr) not in state["trials"]
        ]
        launched = _trial_launch_count(state, trials_dir)
        if launched + len(missing) > CONTROLLED_MAX_TRIAL_LAUNCHES:
            raise TunerError(
                f"controlled batch would exceed the {CONTROLLED_MAX_TRIAL_LAUNCHES}-launch hard limit"
            )
        batch = {
            "requested": controlled_candidates,
            "missing_at_start": missing,
            "started_at": utc_now(),
        }
        state["status"] = "controlled_batch_running"
        state.setdefault("controlled_batches", []).append(batch)
        _save_state(state_path, state)
        for offset in range(0, len(controlled_candidates), len(cuda_ids)):
            _run_trial_batch(
                candidates=controlled_candidates[offset : offset + len(cuda_ids)],
                cuda_ids=cuda_ids,
                python=python,
                base_args=base_args,
                model=args.model,
                quant_stop_layer=quant_stop_layer,
                static_dir=static_dir,
                cache_dir=cache_dir,
                tokens_dir=tokens_dir,
                trials_dir=trials_dir,
                state=state,
                state_path=state_path,
                fingerprint=fingerprint,
                skip_gpu_check=args.skip_gpu_check,
                tuning_w_groupsize=args.tuning_w_groupsize,
                tuning_global_loss_bsz=args.tuning_global_loss_bsz,
            )
        batch["finished_at"] = utc_now()
        batch["trial_launch_count"] = _trial_launch_count(state, trials_dir)
        state["status"] = "awaiting_agent_decision"
        _save_state(state_path, state)
        print(
            json.dumps(
                {
                    "status": state["status"],
                    "points": [dataclasses.asdict(point) for point in _load_points(state)],
                    "trial_launch_count": batch["trial_launch_count"],
                    "remaining_launch_budget": CONTROLLED_MAX_TRIAL_LAUNCHES
                    - batch["trial_launch_count"],
                    "output_root": str(output_root),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    state["status"] = "expanding"
    _save_state(state_path, state)

    bracket: Bracket | None = None
    for _round in range(args.max_expansion_rounds):
        points = _load_points(state)
        bracket = strict_bracket(points)
        if bracket is not None:
            break
        cursor = int(state.get("expansion_cursor", 0))
        candidates = expansion_candidates(
            cursor, args.parallelism, base_lr=args.base_lr
        )
        _run_trial_batch(
            candidates=candidates,
            cuda_ids=cuda_ids,
            python=python,
            base_args=base_args,
            model=args.model,
            quant_stop_layer=quant_stop_layer,
            static_dir=static_dir,
            cache_dir=cache_dir,
            tokens_dir=tokens_dir,
            trials_dir=trials_dir,
            state=state,
            state_path=state_path,
            fingerprint=fingerprint,
            skip_gpu_check=args.skip_gpu_check,
            tuning_w_groupsize=args.tuning_w_groupsize,
            tuning_global_loss_bsz=args.tuning_global_loss_bsz,
        )
        state["expansion_cursor"] = cursor + args.parallelism
        _save_state(state_path, state)
    if bracket is None:
        bracket = strict_bracket(_load_points(state))
    if bracket is None:
        raise TunerError(
            "no strict three-point KL bracket found before max expansion rounds"
        )

    state["status"] = "refining"
    state["bracket"] = bracket.as_dict()
    _save_state(state_path, state)
    while not converged(bracket, args.kl_tolerance):
        round_index = int(state.get("refinement_round", 0))
        if round_index >= args.max_refinement_rounds:
            raise TunerError("refinement did not meet the KL tolerance before its limit")
        points = _load_points(state)
        candidates = refinement_candidates(
            bracket.left.lr,
            bracket.right.lr,
            args.parallelism,
            existing=[point.lr for point in points],
        )
        if not candidates:
            raise TunerError(
                "refinement grid contains no new LR; increase parallelism or precision"
            )
        _run_trial_batch(
            candidates=candidates,
            cuda_ids=cuda_ids,
            python=python,
            base_args=base_args,
            model=args.model,
            quant_stop_layer=quant_stop_layer,
            static_dir=static_dir,
            cache_dir=cache_dir,
            tokens_dir=tokens_dir,
            trials_dir=trials_dir,
            state=state,
            state_path=state_path,
            fingerprint=fingerprint,
            skip_gpu_check=args.skip_gpu_check,
            tuning_w_groupsize=args.tuning_w_groupsize,
            tuning_global_loss_bsz=args.tuning_global_loss_bsz,
        )
        refined = strict_bracket(
            _load_points(state),
            lower=bracket.left.lr,
            upper=bracket.right.lr,
        )
        if refined is None:
            raise TunerError("refinement lost the strict KL bracket")
        if not (
            refined.left.lr >= bracket.left.lr
            and refined.right.lr <= bracket.right.lr
            and (refined.left.lr > bracket.left.lr or refined.right.lr < bracket.right.lr)
        ):
            raise TunerError("refinement did not shrink the LR bracket")
        bracket = refined
        state["refinement_round"] = round_index + 1
        state["bracket"] = bracket.as_dict()
        _save_state(state_path, state)

    left_gap, right_gap = neighbour_gaps(bracket)
    state.update(
        status="tuning_complete",
        selected_lr=bracket.best.lr,
        selected_kl=bracket.best.kl,
        left_relative_gap=left_gap,
        right_relative_gap=right_gap,
        bracket=bracket.as_dict(),
    )
    _save_state(state_path, state)
    atomic_text(output_root / "best_lr.txt", lr_key(bracket.best.lr) + "\n")

    if args.run_main:
        main_dir = output_root / "main"
        main_status_path = main_dir / "status.json"
        if main_status_path.exists():
            main_status = read_json(main_status_path)
            if (
                main_status.get("status") != "succeeded"
                or float(main_status.get("selected_lr", -1)) != bracket.best.lr
            ):
                raise TunerError("existing main status is not a matching success")
        else:
            if not args.skip_gpu_check:
                require_idle_gpus(main_cuda_ids)
            formal_args = main_realq_args(
                base_args,
                model=args.model,
                grad_lr=bracket.best.lr,
                static_cache_path=static_dir,
                cache_dir=cache_dir,
                tokens_cache_path=tokens_dir,
                output_dir=main_dir / "realq_output",
            )
            _validate_realq_args(formal_args)
            main_spec = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "kind": "main",
                "identity": f"main_lr_{lr_key(bracket.best.lr)}",
                "protocol_fingerprint": fingerprint,
                "command": distributed_realq_command(
                    python,
                    formal_args,
                    len(main_cuda_ids),
                    master_port=29500 + os.getpid() % 1000,
                ),
                # Math-SDPA is necessary for stable cross-LR ranking, but its
                # O(seq_len^2) attention matrix can OOM the formal bsz=128
                # profile. The formal run does no candidate comparison, so it
                # retains RealQ's normal attention backend.
                "env": realq_worker_env(
                    ",".join(main_cuda_ids), deterministic_sdpa=False
                ),
                "unset_env": sorted(TORCH_DISTRIBUTED_ENV),
                "cwd": str(REPO_ROOT),
                "log_path": str(main_dir / "execution.log"),
                "result_path": str(main_dir / "worker_result.json"),
                "created_at": utc_now(),
            }
            worker_result = _run_direct_worker(main_spec, python)
            main_status = {
                "schema_version": STATE_SCHEMA_VERSION,
                "status": "succeeded",
                "protocol_fingerprint": fingerprint,
                "selected_lr": bracket.best.lr,
                "worker_result": worker_result,
                "finished_at": utc_now(),
            }
            atomic_json(main_status_path, main_status)
        state["main"] = main_status
        state["status"] = "complete"
        _save_state(state_path, state)

    print(
        json.dumps(
            {
                "status": state["status"],
                "selected_lr": bracket.best.lr,
                "selected_kl": bracket.best.kl,
                "left_relative_gap": left_gap,
                "right_relative_gap": right_gap,
                "output_root": str(output_root),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "_worker":
        parser = argparse.ArgumentParser()
        parser.add_argument("--spec", type=Path, required=True)
        worker_args = parser.parse_args(sys.argv[2:])
        return _worker(worker_args.spec)
    try:
        return _main()
    except (TunerError, ValueError) as exc:
        print(f"realq-auto-tune: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
