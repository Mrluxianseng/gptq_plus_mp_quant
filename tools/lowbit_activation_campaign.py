#!/usr/bin/env python3
"""Resume-safe scheduler for the low-bit activation experiment campaign.

The scheduler is deliberately fail closed:

* ``run`` is a read-only preview unless ``--execute`` is supplied.
* execution waits for the literal ready marker in ``AGENT.md``, the exact
  repository venv, and a successful preflight for the current plan snapshot.
* every experiment is launched through ``lowbit_activation_execute.py`` as a
  separate process; the scheduler never bypasses the executor's provenance,
  output-lock, preflight, or immutable-attempt checks.
* REAL-Q static caches have a single producer.  Consumers are released only
  after the producer manifest succeeds and every rank cache file is validated.
* FSDP and cpu_master are rejected unconditionally.  OOM recovery uses only
  the reviewed batch-size ladders.

The tuning search consumes ``tuning.search_policy`` from the plan.  It runs the
reviewed coarse grid, expands a boundary in small canonical 1/2/3/5/7 batches
without a fixed upper ceiling, then performs the required local refinement.
Every launched candidate counts against the per-model/setting cap, including
OOMs and retries.  Exhausting the cap produces ``NEEDS_USER_ACTION`` and never
manufactures a best learning rate.

Once all 15 proxy searches converge, the default behavior is to emit a
machine-readable plan patch.  ``commit-selected-lrs --execute`` is the only
operation that writes those values into the plan.  It does so once, atomically,
with no campaign process running.  Because that changes the plan SHA256, formal
runs remain in ``WAIT_PREFLIGHT`` until the same preflight path is regenerated
for the new plan snapshot.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import lowbit_activation_execute as executor  # noqa: E402
import lowbit_activation_eta as eta  # noqa: E402
import lowbit_activation_results as results  # noqa: E402
import lowbit_activation_runner as runner  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = runner.DEFAULT_PLAN
DEFAULT_AGENT = REPO_ROOT / "AGENT.md"
DEFAULT_STATE = (
    REPO_ROOT / "output" / "lowbit_activation" / "_campaign" / "state.json"
)
DEFAULT_READY_MARKER = "当前仓库已安装并验证可用的 `.venv`"

STATE_SCHEMA_VERSION = 1
EVENT_SCHEMA_VERSION = 1
RETIREMENT_SCHEMA_VERSION = 1
RETIREMENT_LEDGER_FILENAME = "retirements.jsonl"
MAX_POLL_SECONDS = 60.0
DEFAULT_POLL_SECONDS = 30.0
DEAD_PROCESS_GRACE_POLLS = 3
STALE_PROGRESS_SECONDS = 30 * 60
GUIDED_VALIDATION_TIMEOUT_SECONDS = 30 * 60

ACTIVE_TASK_STATES = {"LAUNCHING", "RUNNING", "STALLED", "VALIDATING"}
TERMINAL_TASK_STATES = {
    "SUCCEEDED",
    "FAILED",
    "OOM",
    "INVALID_RESULT",
    "ORPHANED",
    "SUPERSEDED",
    "CANCELLED",
    "LAUNCH_FAILED",
}

OOM_PATTERNS = (
    re.compile(r"\bcuda out of memory\b", re.IGNORECASE),
    re.compile(r"\btorch\.cuda\.OutOfMemoryError\b"),
    re.compile(r"\bCUBLAS_STATUS_ALLOC_FAILED\b"),
    re.compile(r"\bHIP out of memory\b", re.IGNORECASE),
)
CACHE_MISS_PATTERNS = (
    re.compile(r"\bcache miss\b", re.IGNORECASE),
    re.compile(r"required .*cache.*(?:missing|miss)", re.IGNORECASE),
)
RENDEZVOUS_COLLISION_PATTERNS = (
    re.compile(r"\bEADDRINUSE\b", re.IGNORECASE),
    re.compile(r"\baddress already in use\b", re.IGNORECASE),
)
STATIC_CACHE_PATH_RE = re.compile(
    r"(?P<path>\S+_world(?P<world>\d+)_rank(?P<rank>\d+)\.pt)"
)
EXACT_KL_RE = re.compile(
    r"(?:Exact\s+)?KL&PPL\s+on\s+[^:\s]+\s*:\s*"
    r"(?P<kl>[-+0-9.eEinfnaINFNA]+)\s*,\s*"
    r"(?P<ppl>[-+0-9.eEinfnaINFNA]+)",
    re.IGNORECASE,
)


class CampaignError(RuntimeError):
    """The campaign cannot proceed safely."""


class CampaignBlocked(CampaignError):
    """A fail-closed gate currently prevents launches."""


@dataclass(frozen=True)
class GPUSnapshot:
    devices: tuple[dict[str, Any], ...]
    compute_processes: tuple[dict[str, Any], ...]
    error: str | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "devices": [dict(item) for item in self.devices],
            "compute_processes": [
                dict(item) for item in self.compute_processes
            ],
            "error": self.error,
        }


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _parse_utc(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_sha256(plan: Mapping[str, Any]) -> str:
    """Canonical hash of the frozen plan content embedded in manifests."""
    return _sha256_bytes(_canonical_json_bytes(plan))


def _float_key(value: float) -> str:
    if value == 0:
        return "0"
    return format(float(value), ".12g")


def _normalize_lr(value: float) -> float:
    """Canonicalize arithmetic-generated LRs to the renderer's precision."""
    return float(format(float(value), ".12g"))


def _task_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _option(argv: Sequence[str], flag: str) -> str | None:
    values: list[str] = []
    prefix = f"{flag}="
    for index, item in enumerate(argv):
        if item == flag and index + 1 < len(argv):
            values.append(argv[index + 1])
        elif item.startswith(prefix):
            values.append(item[len(prefix) :])
    if len(values) > 1:
        raise CampaignError(f"rendered argv duplicates {flag}")
    return values[0] if values else None


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise


@contextlib.contextmanager
def _campaign_lock(state_path: Path) -> Iterator[None]:
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignBlocked(
                f"another campaign controller owns {lock_path}"
            ) from exc
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def agent_ready(agent_path: Path, marker: str) -> tuple[bool, str]:
    """Return true only when the literal user-controlled marker is present."""
    try:
        content = agent_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return False, f"cannot read {agent_path}: {exc}"
    if not marker:
        return False, "ready marker must be non-empty"
    if marker not in content:
        return False, f"AGENT.md does not contain ready marker {marker!r}"
    return True, "ready marker present"


def venv_ready(plan: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    configured = plan.get("python_environment")
    if not isinstance(configured, Mapping):
        return False, {"error": "plan python_environment is not an object"}
    expected_value = configured.get("venv")
    if not isinstance(expected_value, str) or not expected_value:
        return False, {"error": "plan venv path is missing"}
    expected_lexical = Path(os.path.abspath(expected_value))
    expected = expected_lexical.resolve(strict=False)
    virtual_value = os.environ.get("VIRTUAL_ENV")
    virtual = (
        Path(virtual_value).resolve(strict=False) if virtual_value else None
    )
    prefix = Path(sys.prefix).resolve(strict=False)
    tools = {
        name: shutil.which(name)
        for name in ("python", "python3", "torchrun")
    }
    tools_inside = {}
    expected_bin = expected_lexical / "bin"
    for name, value in tools.items():
        if value is None:
            tools_inside[name] = False
            continue
        try:
            # A standard venv's ``python`` entry is commonly a symlink to the
            # base interpreter. Its lexical PATH entry must live in venv/bin;
            # sys.prefix and VIRTUAL_ENV establish interpreter ownership.
            Path(os.path.abspath(value)).relative_to(expected_bin)
            tools_inside[name] = True
        except ValueError:
            tools_inside[name] = False
    valid = (
        configured.get("activation_required") is True
        and expected_lexical.is_dir()
        and virtual == expected
        and prefix == expected
        and all(tools_inside.values())
    )
    return valid, {
        "expected": str(expected_lexical),
        "expected_identity": str(expected),
        "exists": expected_lexical.is_dir(),
        "VIRTUAL_ENV": virtual_value,
        "sys_prefix": sys.prefix,
        "sys_executable": sys.executable,
        "tools": tools,
        "tools_inside_venv": tools_inside,
        "valid": valid,
    }


def _resolve_preflight_path(
    plan: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
) -> Path | None:
    resolutions = plan.get("resolutions")
    if not isinstance(resolutions, Mapping):
        return None
    value = resolutions.get("runtime_preflight_manifest")
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve(strict=False)


def validate_preflight_payload(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
    hostname: str,
) -> list[str]:
    """Validate the campaign-critical subset of the runtime preflight."""
    errors: list[str] = []
    if payload.get("schema_version") != 1 or payload.get("valid") is not True:
        errors.append("preflight schema/valid flag is invalid")
    if payload.get("load_task_datasets") is not True:
        errors.append("preflight did not load task datasets")
    plan_identity = payload.get("plan")
    if (
        not isinstance(plan_identity, Mapping)
        or plan_identity.get("sha256") != plan_sha256
    ):
        errors.append("preflight plan SHA256 does not match current plan")

    expected_job = plan.get("resolutions", {}).get("canoe_job_id")
    canoe = payload.get("canoe")
    if (
        not isinstance(canoe, Mapping)
        or canoe.get("valid") is not True
        or canoe.get("expected_job_id") != expected_job
        or canoe.get("supplied_job_id") != expected_job
        or canoe.get("hostname") != hostname
    ):
        errors.append("preflight does not prove the planned Canoe job/hostname")

    runtime = payload.get("runtime")
    python_env = (
        runtime.get("python_environment")
        if isinstance(runtime, Mapping)
        else None
    )
    if (
        not isinstance(python_env, Mapping)
        or python_env.get("valid") is not True
    ):
        errors.append("preflight Python environment is invalid")
    distributions = (
        runtime.get("distributions")
        if isinstance(runtime, Mapping)
        else None
    )
    if (
        not isinstance(distributions, Mapping)
        or distributions.get("lm-eval") != "0.4.4"
    ):
        errors.append("preflight did not validate lm-eval==0.4.4")
    cuda = runtime.get("cuda") if isinstance(runtime, Mapping) else None
    requirement = (
        cuda.get("requirement_check") if isinstance(cuda, Mapping) else None
    )
    if (
        not isinstance(requirement, Mapping)
        or requirement.get("passed") is not True
    ):
        errors.append("preflight GPU count/memory requirement did not pass")

    expected_tasks = tuple(plan.get("paper_zero_shot_tasks", ()))
    lm_eval = payload.get("lm_eval")
    resolved = (
        lm_eval.get("resolved") if isinstance(lm_eval, Mapping) else None
    )
    loaded = (
        lm_eval.get("datasets_loaded")
        if isinstance(lm_eval, Mapping)
        else None
    )
    if (
        not isinstance(resolved, Mapping)
        or set(resolved) != set(expected_tasks)
        or any(resolved.get(task) != [task] for task in expected_tasks)
        or not isinstance(loaded, Mapping)
        or set(loaded) != set(expected_tasks)
        or any(loaded.get(task) is not True for task in expected_tasks)
    ):
        errors.append("preflight did not resolve/load the exact ten tasks")
    return errors


def load_valid_preflight(
    plan: Mapping[str, Any],
    *,
    plan_sha256: str,
    repo_root: Path = REPO_ROOT,
    hostname: str | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    path = _resolve_preflight_path(plan, repo_root=repo_root)
    if path is None:
        return None, ["runtime preflight path is not configured"]
    if not path.is_file():
        return None, [f"runtime preflight is missing: {path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, [f"runtime preflight is unreadable: {exc}"]
    if not isinstance(payload, dict):
        return None, ["runtime preflight must be a JSON object"]
    errors = validate_preflight_payload(
        payload,
        plan=plan,
        plan_sha256=plan_sha256,
        hostname=hostname or socket.gethostname(),
    )
    return payload, errors


def query_gpus() -> GPUSnapshot:
    """Query physical GPUs and compute processes without importing torch."""
    try:
        device_text = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=20,
        )
        devices: list[dict[str, Any]] = []
        for line in device_text.splitlines():
            cells = [cell.strip() for cell in line.split(",")]
            if len(cells) != 6:
                raise ValueError(f"unexpected nvidia-smi device row: {line!r}")
            devices.append(
                {
                    "index": int(cells[0]),
                    "uuid": cells[1],
                    "name": cells[2],
                    "memory_total_mib": int(cells[3]),
                    "memory_free_mib": int(cells[4]),
                    "utilization_gpu_percent": int(cells[5]),
                }
            )
        try:
            process_text = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,gpu_uuid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=20,
            )
        except subprocess.CalledProcessError as exc:
            # Some driver builds return non-zero when no compute process exists.
            observed = (exc.output or "").strip()
            if observed.lower() != "no running processes found":
                raise
            process_text = ""
        processes: list[dict[str, Any]] = []
        for line in process_text.splitlines():
            cells = [cell.strip() for cell in line.split(",")]
            if len(cells) != 3 or cells[0] in {"", "No running processes found"}:
                continue
            if not cells[0].isdigit():
                continue
            used = None if cells[2] in {"N/A", "[N/A]"} else int(cells[2])
            processes.append(
                {
                    "pid": int(cells[0]),
                    "gpu_uuid": cells[1],
                    "used_memory_mib": used,
                }
            )
        return GPUSnapshot(tuple(devices), tuple(processes))
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return GPUSnapshot((), (), f"{type(exc).__name__}: {exc}")


def _proc_identity(pid: int) -> dict[str, int] | None:
    """Return PID-reuse-safe Linux process identity fields."""
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
        close = raw.rfind(")")
        if close < 0:
            return None
        fields = raw[close + 2 :].split()
        # Suffix starts at proc(5) field 3 (state).  Session is field 6 and
        # starttime is field 22.
        if len(fields) < 20:
            return None
        return {
            "pid": int(pid),
            "ppid": int(fields[1]),
            "process_group_id": int(fields[2]),
            "session_id": int(fields[3]),
            "start_ticks": int(fields[19]),
        }
    except (OSError, UnicodeDecodeError, ValueError):
        return None


def _proc_cmdline(pid: int) -> list[str] | None:
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except OSError:
        return None
    values = [
        value.decode("utf-8", errors="replace")
        for value in raw.split(b"\0")
        if value
    ]
    return values or None


def _cmdline_matches_argv(
    actual: Sequence[str] | None,
    expected: Sequence[str],
) -> bool:
    if actual is None:
        return False
    expected_values = [str(value) for value in expected]
    return list(actual) == expected_values or (
        len(expected_values) > 1
        and len(actual) >= len(expected_values) - 1
        and list(actual[-(len(expected_values) - 1) :])
        == expected_values[1:]
    )


def _proc_namespace_ids(pid: int) -> tuple[int, ...] | None:
    try:
        lines = Path(f"/proc/{int(pid)}/status").read_text(
            encoding="utf-8"
        ).splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    for line in lines:
        if not line.startswith("NSpid:"):
            continue
        try:
            values = tuple(int(value) for value in line.split()[1:])
        except ValueError:
            return None
        return values or None
    return None


def _resolve_gpu_process_pids(
    observed_pids: Iterable[int],
) -> tuple[dict[int, int], dict[int, list[int]]]:
    """Map host/NVML PIDs to PIDs visible in this namespace.

    In a PID namespace, ``nvidia-smi`` commonly reports the host PID while
    ``/proc`` and the controller state use the innermost PID.  Linux exposes
    the complete outer-to-inner chain in ``/proc/<local>/status:NSpid``.
    Ambiguous mappings are returned separately and are never treated as
    campaign-owned.
    """
    requested = {
        int(value)
        for value in observed_pids
        if type(value) is int and value > 0
    }
    candidates: dict[int, list[int]] = {value: [] for value in requested}
    if not requested:
        return {}, {}
    try:
        proc_entries = list(Path("/proc").iterdir())
    except OSError:
        proc_entries = []
    for entry in proc_entries:
        if not entry.name.isdigit():
            continue
        local_pid = int(entry.name)
        namespace_ids = _proc_namespace_ids(local_pid)
        if not namespace_ids:
            continue
        outer_pid = namespace_ids[0]
        if outer_pid in candidates:
            candidates[outer_pid].append(local_pid)
    resolved: dict[int, int] = {}
    ambiguous: dict[int, list[int]] = {}
    for observed_pid, local_pids in candidates.items():
        unique = sorted(set(local_pids))
        if len(unique) == 1:
            resolved[observed_pid] = unique[0]
            continue
        if len(unique) > 1:
            ambiguous[observed_pid] = unique
            continue
        # Non-namespaced deployments normally expose a one-element NSpid
        # chain equal to the /proc directory name.  Retain that direct case.
        namespace_ids = _proc_namespace_ids(observed_pid)
        if (
            namespace_ids
            and namespace_ids[0] == observed_pid
            and _proc_identity(observed_pid) is not None
        ):
            resolved[observed_pid] = observed_pid
    return resolved, ambiguous


def _pid_is_descendant(pid: int, root_pid: int) -> bool:
    current = int(pid)
    seen: set[int] = set()
    for _ in range(128):
        if current == root_pid:
            return True
        if current <= 1 or current in seen:
            return False
        seen.add(current)
        identity = _proc_identity(current)
        if identity is None:
            return False
        current = identity["ppid"]
    return False


def eligible_gpu_ids(
    snapshot: GPUSnapshot,
    *,
    minimum_memory_gib: float,
) -> list[int]:
    threshold_mib = minimum_memory_gib * 1024
    return sorted(
        int(device["index"])
        for device in snapshot.devices
        if float(device["memory_total_mib"]) >= threshold_mib
    )


def assign_gpus(
    free_gpu_ids: Sequence[int],
    *,
    count: int,
    contiguous: bool,
) -> tuple[int, ...] | None:
    """Choose a deterministic physical GPU set."""
    values = sorted(set(int(value) for value in free_gpu_ids))
    if count <= 0:
        raise ValueError("GPU count must be positive")
    if len(values) < count:
        return None
    if contiguous:
        for start in range(0, len(values) - count + 1):
            window = values[start : start + count]
            if window[-1] - window[0] == count - 1:
                return tuple(window)
        return None
    return tuple(values[:count])


def canonical_sequence_values(
    *,
    start_exclusive: float,
    direction: str,
    count: int,
    mantissas: Sequence[float],
) -> list[float]:
    """Return adjacent positive values on a canonical mantissa lattice."""
    if start_exclusive <= 0 or not math.isfinite(start_exclusive):
        raise ValueError("canonical expansion starts from a finite positive LR")
    if direction not in {"up", "down"}:
        raise ValueError("direction must be up or down")
    if count <= 0:
        return []
    canonical = sorted({float(value) for value in mantissas})
    if not canonical or canonical[0] <= 0:
        raise ValueError("canonical mantissas must be positive")

    exponent = int(math.floor(math.log10(start_exclusive)))
    candidates: list[float] = []
    # Generate a deliberately broad, but finite, local window.  The loop is
    # extended geometrically if an unusual mantissa list needs more points.
    radius = 4
    while len(candidates) < count:
        values = sorted(
            {
                mantissa * (10.0**power)
                for power in range(exponent - radius, exponent + radius + 1)
                for mantissa in canonical
                if math.isfinite(mantissa * (10.0**power))
                and mantissa * (10.0**power) > 0
            }
        )
        if direction == "up":
            candidates = [
                value
                for value in values
                if value > start_exclusive
                and not math.isclose(
                    value, start_exclusive, rel_tol=1e-12, abs_tol=0.0
                )
            ][:count]
        else:
            candidates = [
                value
                for value in reversed(values)
                if value < start_exclusive
                and not math.isclose(
                    value, start_exclusive, rel_tol=1e-12, abs_tol=0.0
                )
            ][:count]
        radius *= 2
        if radius > 64:
            break
    return [_normalize_lr(value) for value in candidates]


def canonical_between(
    low: float,
    high: float,
    *,
    mantissas: Sequence[float],
) -> list[float]:
    if not (0 <= low < high) or not math.isfinite(high):
        raise ValueError("canonical interval must satisfy 0 <= low < high")
    canonical = sorted({float(value) for value in mantissas})
    low_positive = low if low > 0 else high * 1e-12
    lo_exp = int(math.floor(math.log10(low_positive))) - 1
    hi_exp = int(math.floor(math.log10(high))) + 1
    return sorted(
        {
            _normalize_lr(mantissa * (10.0**power))
            for power in range(lo_exp, hi_exp + 1)
            for mantissa in canonical
            if low < mantissa * (10.0**power) < high
            and math.isfinite(mantissa * (10.0**power))
        }
    )


def local_refinement_candidates(
    observed_lrs: Sequence[float],
    winner_lr: float,
    *,
    mantissas: Sequence[float],
    zero_boundary_probe_ratios: Sequence[float] = (0.1, 0.3, 0.7),
) -> list[float]:
    values = sorted(set(float(value) for value in observed_lrs))
    if winner_lr not in values:
        raise ValueError("winner must be an observed LR")
    index = values.index(winner_lr)
    if index == 0 or index == len(values) - 1:
        return []
    left, right = values[index - 1], values[index + 1]
    if left == 0:
        # Zero has no logarithmic neighborhood. Reuse the reviewed zero-edge
        # ratios across the whole [0, upper-neighbor] bracket instead of
        # enumerating an arbitrary number of decades toward zero.
        candidates = sorted(
            {
                _normalize_lr(right * float(ratio))
                for ratio in zero_boundary_probe_ratios
                if 0 < float(ratio) < 1
            }
        )
    else:
        candidates = canonical_between(
            left, right, mantissas=mantissas
        )
    return [
        value
        for value in candidates
        if value not in values
    ]


def classify_failure(log_text: str) -> str:
    if any(pattern.search(log_text) for pattern in OOM_PATTERNS):
        return "oom"
    if any(pattern.search(log_text) for pattern in CACHE_MISS_PATTERNS):
        return "cache_miss"
    if any(
        pattern.search(log_text)
        for pattern in RENDEZVOUS_COLLISION_PATTERNS
    ):
        return "master_port_collision"
    return "other"


def metric_anomalies(
    points: Sequence[tuple[float, float, float]],
) -> list[dict[str, Any]]:
    """Detect notable LR-curve anomalies without altering selection."""
    anomalies: list[dict[str, Any]] = []
    ordered = sorted(points)
    for lr, kl, ppl in ordered:
        if not math.isfinite(kl) or not math.isfinite(ppl):
            anomalies.append(
                {
                    "code": "nonfinite_exact_metric",
                    "lr": lr,
                    "kl": kl,
                    "ppl": ppl,
                }
            )
        if math.isfinite(kl) and kl < 0:
            anomalies.append(
                {"code": "negative_exact_kl", "lr": lr, "kl": kl}
            )
    for index in range(1, len(ordered) - 1):
        lr, kl, _ = ordered[index]
        neighbors = max(ordered[index - 1][1], ordered[index + 1][1])
        if (
            math.isfinite(kl)
            and math.isfinite(neighbors)
            and kl > max(neighbors * 3.0, neighbors + 1e-6)
        ):
            anomalies.append(
                {
                    "code": "curve_spike",
                    "lr": lr,
                    "kl": kl,
                    "neighbor_max_kl": neighbors,
                }
            )
    for left, right in zip(ordered, ordered[1:]):
        delta_kl = right[1] - left[1]
        delta_ppl = right[2] - left[2]
        if (
            math.isfinite(delta_kl)
            and math.isfinite(delta_ppl)
            and abs(delta_kl) > 1e-12
            and abs(delta_ppl) > 1e-9
            and delta_kl * delta_ppl < 0
        ):
            anomalies.append(
                {
                    "code": "ppl_kl_divergence",
                    "left_lr": left[0],
                    "right_lr": right[0],
                    "delta_kl": delta_kl,
                    "delta_ppl": delta_ppl,
                }
            )
            break
    return anomalies


def _unique_minimum(
    metrics: Mapping[float, Mapping[str, float]],
) -> tuple[float | None, str | None]:
    if not metrics:
        return None, "no successful metrics"
    finite = {
        float(lr): float(value["kl"])
        for lr, value in metrics.items()
        if math.isfinite(float(value["kl"]))
        and float(value["kl"]) >= 0
    }
    if len(finite) != len(metrics):
        return None, "non-finite or negative exact KL"
    minimum = min(finite.values())
    winners = [lr for lr, value in finite.items() if value == minimum]
    if len(winners) != 1:
        return None, "minimum exact KL is tied"
    return winners[0], None


def search_decision(
    *,
    metrics: Mapping[float, Mapping[str, float]],
    coarse_lrs: Sequence[float],
    attempted_count: int,
    pending_count: int,
    refinement_done: bool,
    zero_probes_done: Sequence[float],
    policy: Mapping[str, Any],
    max_attempts: int,
) -> dict[str, Any]:
    """Plan the next bounded adaptive-search action.

    The function is pure and therefore used directly by the unit tests.  It
    never selects a boundary point and never schedules work beyond the cap.
    """
    observed = sorted({_normalize_lr(value) for value in metrics})
    missing_coarse = sorted(
        {_normalize_lr(value) for value in coarse_lrs} - set(observed)
    )
    if policy.get("require_complete_coarse_round") is True and missing_coarse:
        return {
            "action": "wait_coarse",
            "candidates": missing_coarse,
            "reason": "coarse round is incomplete",
        }
    winner, error = _unique_minimum(metrics)
    if error is not None:
        return {
            "action": "needs_user_action",
            "candidates": [],
            "reason": error,
        }
    assert winner is not None

    remaining = max_attempts - attempted_count - pending_count
    if remaining < 0:
        return {
            "action": "needs_user_action",
            "candidates": [],
            "reason": "attempt accounting already exceeds the hard cap",
        }
    reserve = (
        0
        if refinement_done
        else int(policy["reserved_refinement_attempts"])
    )
    batch_size = int(policy["boundary_expansion_batch_size"])
    mantissas = tuple(float(value) for value in policy["canonical_mantissas"])

    if winner == observed[0]:
        if winner != 0:
            return {
                "action": "needs_user_action",
                "candidates": [],
                "reason": "positive lower boundary is not bracketed",
            }
        positives = sorted(
            {
                _normalize_lr(value)
                for value in coarse_lrs
                if float(value) > 0
            }
        )
        if not positives:
            positives = [value for value in observed if value > 0]
        if not positives:
            return {
                "action": "needs_user_action",
                "candidates": [],
                "reason": "zero is the only observed learning rate",
            }
        nearest = positives[0]
        all_zero_probes = [
            _normalize_lr(nearest * float(ratio))
            for ratio in policy["zero_boundary_probe_ratios"]
        ]
        completed_zero_probes = {
            _normalize_lr(value) for value in zero_probes_done
        }
        todo = [
            value
            for value in all_zero_probes
            if value not in observed and value not in completed_zero_probes
        ]
        slots = max(0, remaining - reserve)
        candidates = todo[: min(batch_size, slots)]
        if candidates:
            return {
                "action": "expand_zero_boundary",
                "candidates": candidates,
                "reason": "zero boundary needs positive probes",
            }
        return {
            "action": "needs_user_action",
            "candidates": [],
            "reason": (
                "zero remains the minimum after the configured probes, or the "
                "20-attempt budget cannot preserve local refinement"
            ),
        }

    if winner == observed[-1]:
        slots = max(0, remaining - reserve)
        count = min(batch_size, slots)
        if count <= 0:
            return {
                "action": "needs_user_action",
                "candidates": [],
                "reason": (
                    "upper boundary remains unbracketed and the hard cap must "
                    "reserve local-refinement attempts"
                ),
            }
        candidates = canonical_sequence_values(
            start_exclusive=winner,
            direction="up",
            count=count,
            mantissas=mantissas,
        )
        if not candidates:
            return {
                "action": "needs_user_action",
                "candidates": [],
                "reason": "cannot generate a finite upper-boundary expansion",
            }
        return {
            "action": "expand_upper_boundary",
            "candidates": candidates,
            "reason": "upper KL minimum is not bracketed",
        }

    if policy.get("require_local_refinement") is True and not refinement_done:
        candidates = local_refinement_candidates(
            observed,
            winner,
            mantissas=mantissas,
            zero_boundary_probe_ratios=policy[
                "zero_boundary_probe_ratios"
            ],
        )
        if len(candidates) > reserve:
            # The policy reserves an exact bounded refinement budget. Keep the
            # logarithmically closest canonical points; this also handles the
            # zero-adjacent interval without allowing an unbounded tail.
            candidates = sorted(
                sorted(
                    candidates,
                    key=lambda value: (
                        abs(math.log10(value / winner)),
                        value,
                    ),
                )[:reserve]
            )
        if not candidates:
            return {
                "action": "needs_user_action",
                "candidates": [],
                "reason": "canonical local refinement produced no new point",
            }
        if len(candidates) > remaining:
            return {
                "action": "needs_user_action",
                "candidates": [],
                "reason": (
                    "insufficient attempts remain for the required canonical "
                    "local refinement"
                ),
            }
        return {
            "action": "local_refine",
            "candidates": candidates,
            "reason": "interior coarse minimum needs canonical refinement",
        }

    return {
        "action": "select",
        "candidates": [],
        "selected_lr": winner,
        "reason": "unique interior exact-KL minimum after local refinement",
    }


def _profile_values(
    ladders: Mapping[str, Sequence[int]],
    level: int,
) -> dict[str, int]:
    result: dict[str, int] = {}
    for name, raw_values in ladders.items():
        values = [int(value) for value in raw_values]
        if not values:
            raise CampaignError(f"empty OOM ladder for {name}")
        result[name] = values[min(level, len(values) - 1)]
    return result


def _deduplicated_profiles(
    ladders: Mapping[str, Sequence[int]],
) -> list[dict[str, int]]:
    count = max(len(values) for values in ladders.values())
    profiles: list[dict[str, int]] = []
    for level in range(count):
        value = _profile_values(ladders, level)
        if not profiles or value != profiles[-1]:
            profiles.append(value)
    return profiles


def _effective_overrides(
    nominal: Mapping[str, Any],
    profile: Mapping[str, int],
) -> dict[str, int]:
    return {
        name: int(value)
        for name, value in profile.items()
        if int(nominal.get(name, value)) != int(value)
    }


def validate_no_fsdp(rendered: runner.RenderedCommand) -> None:
    if "realq.ptq" not in rendered.argv:
        return
    if _option(rendered.argv, "--fsdp") != "false":
        raise CampaignError("REAL-Q command must explicitly use --fsdp false")
    if _option(rendered.argv, "--cpu_master") != "false":
        raise CampaignError(
            "REAL-Q command must explicitly use --cpu_master false"
        )


def extract_static_cache_paths(
    log_text: str,
    *,
    repo_root: Path = REPO_ROOT,
) -> list[Path]:
    paths: list[Path] = []
    for match in STATIC_CACHE_PATH_RE.finditer(log_text):
        path = Path(match.group("path"))
        if not path.is_absolute():
            path = repo_root / path
        resolved = path.resolve(strict=False)
        if resolved not in paths:
            paths.append(resolved)
    return paths


def validate_static_cache_paths(
    paths: Sequence[Path],
    *,
    expected_world_size: int,
) -> tuple[bool, dict[str, Any]]:
    """Validate one complete same-key set of atomic per-rank cache files."""
    groups: dict[str, dict[int, Path]] = {}
    pattern = re.compile(
        rf"^(?P<key>.+)_world{expected_world_size}_rank(?P<rank>\d+)\.pt$"
    )
    errors: list[str] = []
    for path in paths:
        match = pattern.match(path.name)
        if match is None:
            continue
        groups.setdefault(match.group("key"), {})[int(match.group("rank"))] = path
    expected_ranks = set(range(expected_world_size))
    complete = [
        (key, rank_paths)
        for key, rank_paths in groups.items()
        if set(rank_paths) == expected_ranks
    ]
    if len(complete) != 1:
        errors.append(
            "expected exactly one complete static-cache key, got "
            f"{[(key, sorted(value)) for key, value in groups.items()]!r}"
        )
        return False, {"valid": False, "errors": errors, "files": []}
    key, rank_paths = complete[0]
    files: list[dict[str, Any]] = []
    for rank in sorted(rank_paths):
        path = rank_paths[rank]
        if not path.is_file():
            errors.append(f"rank {rank} cache file is missing: {path}")
            continue
        stat = path.stat()
        if stat.st_size <= 0:
            errors.append(f"rank {rank} cache file is empty: {path}")
        files.append(
            {
                "rank": rank,
                "path": str(path),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    temporary_files = sorted(
        str(path)
        for path in rank_paths[0].parent.glob("*.tmp")
        if path.is_file()
    )
    if temporary_files:
        errors.append(f"static cache contains temporary files: {temporary_files}")
    return not errors, {
        "valid": not errors,
        "cache_key": key,
        "world_size": expected_world_size,
        "files": files,
        "errors": errors,
    }


def selected_lr_patch(state: Mapping[str, Any]) -> dict[str, Any] | None:
    tuning = state.get("tuning")
    if not isinstance(tuning, Mapping):
        return None
    selected: dict[str, dict[str, float]] = {
        model: {} for model in runner.MODEL_ORDER
    }
    for model in runner.MODEL_ORDER:
        for setting in runner.SETTING_ORDER:
            group = tuning.get(f"{model}/{setting}")
            if (
                not isinstance(group, Mapping)
                or group.get("status") != "SELECTED"
                or isinstance(group.get("selected_lr"), bool)
                or not isinstance(group.get("selected_lr"), (int, float))
                or not math.isfinite(float(group["selected_lr"]))
                or float(group["selected_lr"]) < 0
            ):
                return None
            selected[model][setting] = float(group["selected_lr"])
    return {"selected_grad_lr_by_model_setting": selected}


class Campaign:
    """One resume-safe campaign controller."""

    def __init__(
        self,
        *,
        plan_path: Path,
        state_path: Path,
        events_path: Path,
        agent_path: Path,
        ready_marker: str,
        execute: bool,
        poll_seconds: float,
        repo_root: Path = REPO_ROOT,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.plan_path = plan_path.resolve(strict=True)
        self.state_path = state_path.resolve(strict=False)
        self.events_path = events_path.resolve(strict=False)
        self.agent_path = agent_path.resolve(strict=False)
        self.ready_marker = ready_marker
        self.execute = execute
        self.poll_seconds = min(max(float(poll_seconds), 1.0), MAX_POLL_SECONDS)
        self.processes: dict[str, subprocess.Popen[Any]] = {}
        self.controller_logs: dict[str, Any] = {}
        self.preview_events: list[dict[str, Any]] = []

        self.plan_raw = self.plan_path.read_bytes()
        self.plan = json.loads(self.plan_raw)
        runner.validate_structure(self.plan)
        self.plan_sha256 = _sha256_bytes(self.plan_raw)
        output_root = Path(self.plan["output_root"])
        if not output_root.is_absolute():
            output_root = self.repo_root / output_root
        self.output_root = output_root.resolve(strict=False)
        self.retirements_path = (
            self.output_root / "_campaign" / RETIREMENT_LEDGER_FILENAME
        )
        self.state = self._load_or_create_state()
        self._ensure_plan_protocol_identities()
        loaded_retirements, retirement_errors = self._load_retirement_ledger()
        # A retirement ledger is an authorization to ignore otherwise
        # immutable successful manifests.  Never apply even the valid-looking
        # prefix of a ledger that fails any chain, artifact, or semantic
        # check: doing so would persist poisoned SUPERSEDED/generation state
        # before the launch gates have a chance to fail closed.
        self.retirement_records = (
            loaded_retirements if not retirement_errors else []
        )
        self.state["retirement_ledger"] = {
            "path": str(self.retirements_path),
            "records": len(loaded_retirements),
            "last_record_sha256": (
                loaded_retirements[-1]["record_sha256"]
                if loaded_retirements
                else None
            ),
            "valid": not retirement_errors,
            "errors": retirement_errors,
        }
        self._import_existing_tune_manifests()
        if not retirement_errors:
            self._apply_retirements_to_state()
        self._audit_state_task_plan_identities()

    def _load_or_create_state(self) -> dict[str, Any]:
        if self.state_path.is_file():
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if (
                not isinstance(payload, dict)
                or payload.get("schema_version") != STATE_SCHEMA_VERSION
            ):
                raise CampaignError("campaign state schema is invalid")
            return payload
        now = _utc_now()
        tuning: dict[str, dict[str, Any]] = {}
        for model in runner.MODEL_ORDER:
            for setting in runner.SETTING_ORDER:
                tuning[f"{model}/{setting}"] = {
                    "model": model,
                    "setting": setting,
                    "status": "WAIT_STATIC",
                    "profile_level": 0,
                    "generation": 0,
                    "canary_lr": self._canary_lr(),
                    "canary_succeeded": False,
                    "attempt_count": 0,
                    "refinement_done": False,
                    "zero_probes_done": [],
                    "selected_lr": None,
                    "selected_manifest": None,
                    "needs_user_action_reason": None,
                }
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "campaign_id": (
                f"lowbit-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
                f"{uuid.uuid4().hex[:10]}"
            ),
            "created_utc": now,
            "updated_utc": now,
            "status": "WAIT_ENV_READY",
            "execute_requested": False,
            "plan": {
                "path": str(self.plan_path),
                "tune_sha256": self.plan_sha256,
                "tune_protocol_sha256": _protocol_sha256(self.plan),
                "active_sha256": self.plan_sha256,
                "formal_sha256": None,
                "formal_protocol_sha256": None,
                "pending_transition_sha256": None,
            },
            "agent": {
                "path": str(self.agent_path),
                "ready_marker": self.ready_marker,
            },
            "preflight": None,
            "source_identity": None,
            "heartbeat": None,
            "eta": None,
            "eta_event_tracker": None,
            "tasks": {},
            "static_caches": {
                f"tune/{model}": {
                    "phase": "tune",
                    "model": model,
                    "status": "PENDING",
                    "profile_level": 0,
                    "producer_task": None,
                    "validation": None,
                }
                for model in runner.MODEL_ORDER
            },
            "tuning": tuning,
            "formal": {
                "initialized": False,
                "models": {},
            },
            "selected_lr_patch": None,
            "blockers": [],
            "anomaly_keys": [],
            "event_seq": 0,
            "last_event": None,
        }

    def _ensure_plan_protocol_identities(self) -> None:
        identity = self.state["plan"]
        if "tune_protocol_sha256" not in identity:
            identity["tune_protocol_sha256"] = (
                _protocol_sha256(self.plan)
                if identity.get("tune_sha256") == self.plan_sha256
                else None
            )
        if "formal_protocol_sha256" not in identity:
            identity["formal_protocol_sha256"] = (
                _protocol_sha256(self.plan)
                if identity.get("formal_sha256") == self.plan_sha256
                else None
            )

    def _expected_plan_identity(
        self,
        spec: Mapping[str, Any],
    ) -> tuple[str | None, str | None]:
        tune = (
            spec.get("phase") == "tune"
            or spec.get("target_phase") == "tune"
        )
        if tune:
            return (
                self.state["plan"].get("tune_sha256"),
                self.state["plan"].get("tune_protocol_sha256"),
            )
        return (
            self.state["plan"].get("formal_sha256")
            or self.state["plan"].get("active_sha256"),
            self.state["plan"].get("formal_protocol_sha256")
            or (
                _protocol_sha256(self.plan)
                if self.state["plan"].get("active_sha256")
                == self.plan_sha256
                else None
            ),
        )

    def _manifest_plan_matches(
        self,
        manifest: Mapping[str, Any],
        spec: Mapping[str, Any],
        *,
        require_finished: bool = False,
    ) -> tuple[bool, dict[str, Any]]:
        expected_sha, expected_protocol = self._expected_plan_identity(spec)
        plan_identity = manifest.get("plan")
        observed_sha = (
            plan_identity.get("sha256")
            if isinstance(plan_identity, Mapping)
            else None
        )
        content = (
            plan_identity.get("content")
            if isinstance(plan_identity, Mapping)
            else None
        )
        observed_protocol = (
            _protocol_sha256(content)
            if isinstance(content, Mapping)
            else None
        )
        valid = (
            isinstance(expected_sha, str)
            and isinstance(expected_protocol, str)
            and observed_sha == expected_sha
            and observed_protocol == expected_protocol
            and (
                not require_finished
                or (
                    isinstance(plan_identity, Mapping)
                    and plan_identity.get("changed_during_execution") is False
                    and plan_identity.get("sha256_at_end") == expected_sha
                )
            )
        )
        return valid, {
            "expected_plan_sha256": expected_sha,
            "observed_plan_sha256": observed_sha,
            "expected_protocol_sha256": expected_protocol,
            "observed_protocol_sha256": observed_protocol,
        }

    def _canary_lr(self) -> float:
        candidates = [float(value) for value in self.plan["tuning"]["lr_candidates"]]
        if 1e-5 in candidates:
            return 1e-5
        positives = [value for value in candidates if value > 0]
        if not positives:
            raise CampaignError("tuning coarse grid has no positive canary LR")
        return positives[len(positives) // 2]

    def _audit_state_task_plan_identities(self) -> None:
        """Make terminal state incapable of bypassing manifest provenance."""
        for task in self.state["tasks"].values():
            manifest_value = task.get("manifest")
            if not isinstance(manifest_value, str):
                continue
            manifest_path = Path(manifest_value)
            if not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ):
                manifest = {}
            compatible, details = self._manifest_plan_matches(
                manifest,
                task["spec"],
                require_finished=task.get("status") == "SUCCEEDED",
            )
            task["plan_compatible"] = compatible
            if compatible:
                continue
            spec = task["spec"]
            tune = (
                spec.get("phase") == "tune"
                or spec.get("target_phase") == "tune"
            )
            task["failure_class"] = "plan_identity_mismatch"
            if tune:
                task["status"] = "SUPERSEDED"
                spec["generation"] = -1
                if spec.get("kind") == "realq":
                    group = self.state["tuning"].get(
                        f"{spec.get('model')}/{spec.get('setting')}"
                    )
                    if isinstance(group, dict) and group.get("status") == "SELECTED":
                        group["status"] = "NEEDS_USER_ACTION"
                        group["selected_lr"] = None
                        group["selected_manifest"] = None
                        group["needs_user_action_reason"] = (
                            "selected tune manifest belongs to another "
                            "plan/protocol"
                        )
            else:
                task["status"] = "INVALID_RESULT"
            for entry in self.state["static_caches"].values():
                if entry.get("producer_task") == task.get("task_id"):
                    entry["status"] = "NEEDS_USER_ACTION"
                    entry["reason"] = "producer plan/protocol identity mismatch"
            self.anomaly(
                "config_drift",
                details={
                    "task_id": task.get("task_id"),
                    "manifest": str(manifest_path),
                    "reason": "state task plan/protocol identity mismatch",
                    **details,
                },
                key=f"state-task-plan:{task.get('task_id')}",
            )

    def _launch_events(self) -> tuple[dict[int, dict[str, Any]], list[str]]:
        events: dict[int, dict[str, Any]] = {}
        errors: list[str] = []
        if not self.events_path.is_file():
            return events, errors
        try:
            lines = self.events_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            return {}, [f"cannot read launch-event ledger: {exc}"]
        for line_number, line in enumerate(lines, 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                not isinstance(event, dict)
                or event.get("category") != "task"
                or event.get("code") != "task_launching"
            ):
                continue
            sequence = event.get("seq")
            if type(sequence) is not int or sequence <= 0:
                errors.append(
                    f"task_launching event line {line_number} has invalid seq"
                )
                continue
            if sequence in events:
                errors.append(f"duplicate task_launching event seq {sequence}")
                continue
            events[sequence] = event
        return events, errors

    def _artifact_reference(self, path: Path) -> dict[str, Any]:
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
        return {
            "absolute_path": str(resolved),
            "relative_path": os.path.relpath(resolved, self.output_root),
            "sha256": _sha256_file(resolved),
            "size_bytes": stat.st_size,
        }

    def _validate_artifact_reference(
        self,
        value: Any,
        *,
        label: str,
    ) -> list[str]:
        if not isinstance(value, Mapping):
            return [f"{label} must be an object"]
        absolute = value.get("absolute_path")
        relative = value.get("relative_path")
        expected_sha = value.get("sha256")
        size = value.get("size_bytes")
        if (
            not isinstance(absolute, str)
            or not isinstance(relative, str)
            or not isinstance(expected_sha, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
            or type(size) is not int
            or size < 0
        ):
            return [f"{label} has invalid path/hash/size fields"]
        path = Path(absolute)
        errors: list[str] = []
        if not path.is_absolute():
            errors.append(f"{label}.absolute_path is not absolute")
        if os.path.relpath(path, self.output_root) != relative:
            errors.append(f"{label}.relative_path disagrees with absolute_path")
        try:
            stat = path.stat()
            if stat.st_size != size:
                errors.append(f"{label} size changed")
            if _sha256_file(path) != expected_sha:
                errors.append(f"{label} SHA256 changed")
        except OSError as exc:
            errors.append(f"{label} is unavailable: {exc}")
        return errors

    def _load_retirement_ledger(
        self,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        if not self.retirements_path.is_file():
            return [], []
        try:
            lines = self.retirements_path.read_text(
                encoding="utf-8"
            ).splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            return [], [f"cannot read retirement ledger: {exc}"]
        launch_events, event_errors = self._launch_events()
        errors = list(event_errors)
        records: list[dict[str, Any]] = []
        previous: str | None = None
        ledger_campaign_id: str | None = None
        transition_heads: dict[
            tuple[str, str, str, str | None], Mapping[str, Any]
        ] = {}
        retired_targets: set[str] = set()
        for line_number, line in enumerate(lines, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(
                    f"retirement line {line_number} is invalid JSON: {exc}"
                )
                continue
            if not isinstance(record, dict):
                errors.append(
                    f"retirement line {line_number} must be an object"
                )
                continue
            supplied_hash = record.get("record_sha256")
            without_hash = dict(record)
            without_hash.pop("record_sha256", None)
            calculated_hash = _sha256_bytes(
                _canonical_json_bytes(without_hash)
            )
            if supplied_hash != calculated_hash:
                errors.append(
                    f"retirement line {line_number} record SHA256 mismatch"
                )
            if record.get("schema_version") != RETIREMENT_SCHEMA_VERSION:
                errors.append(
                    f"retirement line {line_number} schema_version mismatch"
                )
            if record.get("seq") != line_number:
                errors.append(
                    f"retirement line {line_number} has non-contiguous seq"
                )
            if record.get("prev_record_sha256") != previous:
                errors.append(
                    f"retirement line {line_number} hash chain is broken"
                )
            previous = supplied_hash if isinstance(supplied_hash, str) else None
            campaign_id = record.get("campaign_id")
            if not isinstance(campaign_id, str) or not campaign_id:
                errors.append(
                    f"retirement line {line_number} has invalid campaign_id"
                )
            elif ledger_campaign_id is None:
                ledger_campaign_id = campaign_id
            elif campaign_id != ledger_campaign_id:
                errors.append(
                    f"retirement line {line_number} mixes campaign_id values"
                )
            if record.get("reason") != "oom_profile_transition":
                errors.append(
                    f"retirement line {line_number} has invalid reason"
                )
            phase = record.get("phase")
            model = record.get("model")
            setting = record.get("setting")
            if (
                phase not in {"tune", "final"}
                or model not in runner.MODEL_ORDER
                or (
                    phase == "tune"
                    and setting not in runner.SETTING_ORDER
                )
                or (phase == "final" and setting is not None)
            ):
                errors.append(
                    f"retirement line {line_number} has invalid scope"
                )
            old = record.get("old_profile")
            replacement = record.get("replacement_profile")
            if not isinstance(old, Mapping) or not isinstance(
                replacement, Mapping
            ):
                errors.append(
                    f"retirement line {line_number} profile fields invalid"
                )
            else:
                old_level = old.get("level")
                old_generation = old.get("generation")
                replacement_level = replacement.get("level")
                replacement_generation = replacement.get("generation")
                if (
                    type(old_level) is not int
                    or type(old_generation) is not int
                    or type(replacement_level) is not int
                    or type(replacement_generation) is not int
                    or replacement_level != old_level + 1
                    or replacement_generation != old_generation + 1
                    or not isinstance(old.get("effective_overrides"), Mapping)
                    or not isinstance(
                        replacement.get("effective_overrides"), Mapping
                    )
                ):
                    errors.append(
                        f"retirement line {line_number} replacement is not "
                        "the strict next profile/generation"
                    )
                key = (str(record.get("plan_sha256")), phase, str(model), setting)
                prior = transition_heads.get(key)
                if prior is not None and (
                    old.get("level") != prior.get("level")
                    or old.get("generation") != prior.get("generation")
                    or old.get("effective_overrides")
                    != prior.get("effective_overrides")
                ):
                    errors.append(
                        f"retirement line {line_number} profile chain is "
                        "not contiguous"
                    )
                transition_heads[key] = replacement
            retired = record.get("retired_manifests")
            trigger = record.get("trigger")
            if not isinstance(retired, list) or not retired:
                errors.append(
                    f"retirement line {line_number} has no retired manifests"
                )
                retired = []
            trigger_matches = 0
            for index, item in enumerate(retired):
                label = f"retirement[{line_number}].retired_manifests[{index}]"
                if not isinstance(item, Mapping):
                    errors.append(f"{label} must be an object")
                    continue
                for artifact_name in ("manifest", "log"):
                    errors.extend(
                        self._validate_artifact_reference(
                            item.get(artifact_name),
                            label=f"{label}.{artifact_name}",
                        )
                    )
                manifest_ref = item.get("manifest")
                target = (
                    manifest_ref.get("absolute_path")
                    if isinstance(manifest_ref, Mapping)
                    else None
                )
                if not isinstance(target, str) or target in retired_targets:
                    errors.append(f"{label} is duplicated or lacks a path")
                else:
                    retired_targets.add(target)
                sequence = item.get("launch_event_seq")
                event = launch_events.get(sequence)
                details = (
                    event.get("details")
                    if isinstance(event, Mapping)
                    else None
                )
                expected_event = {
                    "task_id": item.get("task_id"),
                    "plan_sha256": item.get("plan_sha256"),
                    "phase": item.get("phase"),
                    "model": item.get("model"),
                    "setting": item.get("setting"),
                    "grad_lr": item.get("grad_lr"),
                    "profile_level": item.get("profile_level"),
                    "generation": item.get("generation"),
                    "effective_overrides": item.get("effective_overrides"),
                }
                if not isinstance(details, Mapping) or any(
                    details.get(name) != expected
                    for name, expected in expected_event.items()
                ):
                    errors.append(f"{label} launch-event identity mismatch")
                elif event.get("campaign_id") != campaign_id:
                    errors.append(f"{label} launch-event campaign_id mismatch")
                elif (
                    isinstance(manifest_ref, Mapping)
                    and str(
                        Path(
                            str(details.get("output_dir", ""))
                        ).resolve(strict=False)
                    )
                    != str(
                        Path(
                            str(manifest_ref.get("absolute_path", ""))
                        ).resolve(strict=False).parent
                    )
                ):
                    errors.append(f"{label} launch-event output_dir mismatch")
                if (
                    item.get("plan_sha256") != record.get("plan_sha256")
                    or item.get("phase") != phase
                    or item.get("model") != model
                    or (
                        phase == "tune"
                        and item.get("setting") != setting
                    )
                    or (
                        isinstance(old, Mapping)
                        and (
                            item.get("profile_level") != old.get("level")
                            or item.get("generation") != old.get("generation")
                            or item.get("effective_overrides")
                            != old.get("effective_overrides")
                        )
                    )
                ):
                    errors.append(f"{label} scope/profile mismatch")
                if item == trigger:
                    trigger_matches += 1
            if (
                not isinstance(trigger, Mapping)
                or trigger.get("status") != "oom"
                or trigger_matches != 1
            ):
                errors.append(
                    f"retirement line {line_number} OOM trigger is invalid"
                )
            records.append(record)
        return records, errors

    def _apply_retirements_to_state(
        self,
        *,
        phases: set[str] | None = None,
    ) -> None:
        by_manifest = {
            str(Path(task["manifest"]).resolve(strict=False)): task
            for task in self.state["tasks"].values()
            if task.get("manifest")
        }
        for record in self.retirement_records:
            if phases is not None and record.get("phase") not in phases:
                continue
            expected_plan = (
                self.state["plan"].get("tune_sha256")
                if record.get("phase") == "tune"
                else (
                    self.state["plan"].get("formal_sha256")
                    or self.state["plan"].get("active_sha256")
                )
            )
            if record.get("plan_sha256") != expected_plan:
                continue
            retired_paths = {
                str(
                    Path(item["manifest"]["absolute_path"]).resolve(
                        strict=False
                    )
                )
                for item in record.get("retired_manifests", [])
                if isinstance(item, Mapping)
                and isinstance(item.get("manifest"), Mapping)
            }
            for path in retired_paths:
                task = by_manifest.get(path)
                if task is not None:
                    task["status"] = "SUPERSEDED"
                    task["retirement_record_sha256"] = record[
                        "record_sha256"
                    ]
            replacement = record["replacement_profile"]
            if record["phase"] == "tune":
                group = self.state["tuning"][
                    f"{record['model']}/{record['setting']}"
                ]
                replacement_position = (
                    int(replacement["generation"]),
                    int(replacement["level"]),
                )
                current_position = (
                    int(group.get("generation", 0)),
                    int(group.get("profile_level", 0)),
                )
                # Applying an append-only transition must be idempotent.  A
                # restart (or later formal initialization) may revisit every
                # ledger record after this generation has already completed
                # tuning; never reset a SELECTED/refined group back to CANARY.
                if current_position < replacement_position:
                    group["profile_level"] = replacement_position[1]
                    group["generation"] = replacement_position[0]
                    group["canary_succeeded"] = False
                    group["refinement_done"] = False
                    group["zero_probes_done"] = []
                    if group.get("status") not in {
                        "NEEDS_USER_ACTION",
                        "SELECTED",
                    }:
                        group["status"] = "CANARY"
                for task in self.state["tasks"].values():
                    spec = task["spec"]
                    if (
                        spec.get("kind") == "realq"
                        and spec.get("phase") == "tune"
                        and spec.get("model") == record["model"]
                        and spec.get("setting") == record["setting"]
                        and int(spec.get("profile_level", -1))
                        == int(replacement["level"])
                        and str(
                            Path(task.get("manifest", "")).resolve(
                                strict=False
                            )
                        )
                        not in retired_paths
                    ):
                        spec["generation"] = int(replacement["generation"])
            elif self.state["formal"].get("initialized"):
                model_state = self.state["formal"]["models"][record["model"]]
                replacement_position = (
                    int(replacement["generation"]),
                    int(replacement["level"]),
                )
                current_position = (
                    int(model_state.get("realq_generation", 0)),
                    int(model_state.get("realq_profile_level", 0)),
                )
                if current_position < replacement_position:
                    model_state["realq_profile_level"] = (
                        replacement_position[1]
                    )
                    model_state["realq_generation"] = replacement_position[0]

    def _find_launch_event_for_task(
        self,
        task: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        events, errors = self._launch_events()
        if errors:
            return None
        output = str(Path(str(task["output_dir"])).resolve(strict=False))
        matches = [
            event
            for event in events.values()
            if isinstance(event.get("details"), Mapping)
            and str(
                Path(str(event["details"].get("output_dir", ""))).resolve(
                    strict=False
                )
            )
            == output
            and event["details"].get("plan_sha256")
            == self._expected_plan_identity(task["spec"])[0]
        ]
        return matches[0] if len(matches) == 1 else None

    def _retirement_item(self, task: Mapping[str, Any]) -> dict[str, Any]:
        manifest_path = Path(str(task["manifest"])).resolve(strict=True)
        log_path = Path(str(task["log"])).resolve(strict=True)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        compatible, details = self._manifest_plan_matches(
            manifest,
            task["spec"],
            require_finished=True,
        )
        if not compatible:
            raise CampaignBlocked(
                "cannot retire manifest with mismatched plan identity: "
                f"{details}"
            )
        event = self._find_launch_event_for_task(task)
        if event is None:
            raise CampaignBlocked(
                f"cannot retire unregistered launch {task['task_id']}"
            )
        event_details = event["details"]
        status = str(task["status"]).lower()
        if status not in {
            "succeeded",
            "oom",
            "failed",
            "invalid_result",
            "orphaned",
            "launch_failed",
        }:
            raise CampaignBlocked(
                f"cannot retire non-terminal task status {task['status']}"
            )
        model_identity = manifest.get("model")
        return {
            "task_id": event_details["task_id"],
            "launch_event_seq": event["seq"],
            "phase": task["spec"]["phase"],
            "model": task["spec"]["model"],
            "setting": task["spec"].get("setting"),
            "grad_lr": task["spec"].get("grad_lr"),
            "profile_level": int(task["spec"]["profile_level"]),
            "generation": int(task["spec"]["generation"]),
            "effective_overrides": dict(task["spec"].get("overrides", {})),
            "status": status,
            "manifest_status": manifest.get("status"),
            "manifest": self._artifact_reference(manifest_path),
            "log": self._artifact_reference(log_path),
            "execution_id": manifest.get("execution_id"),
            "run_id": manifest.get("run_id"),
            "plan_sha256": manifest["plan"]["sha256"],
            "model_sha256": (
                model_identity.get("combined_identity_sha256")
                if isinstance(model_identity, Mapping)
                else None
            ),
        }

    def _append_profile_retirement(
        self,
        *,
        phase: str,
        model: str,
        setting: str | None,
        tasks: Sequence[Mapping[str, Any]],
        old_profile: Mapping[str, Any],
        replacement_profile: Mapping[str, Any],
    ) -> dict[str, Any]:
        if any(task["status"] in ACTIVE_TASK_STATES for task in tasks):
            raise CampaignBlocked("profile retirement requires a drained wave")
        items = [
            self._retirement_item(task)
            for task in tasks
            if task.get("manifest") and Path(str(task["manifest"])).is_file()
        ]
        items.sort(key=lambda item: item["manifest"]["absolute_path"])
        triggers = [item for item in items if item["status"] == "oom"]
        if not items or not triggers:
            raise CampaignBlocked(
                "profile retirement requires registered manifests and an OOM "
                "trigger"
            )
        base = {
            "schema_version": RETIREMENT_SCHEMA_VERSION,
            "seq": len(self.retirement_records) + 1,
            "timestamp_utc": _utc_now(),
            "campaign_id": self.state["campaign_id"],
            "prev_record_sha256": (
                self.retirement_records[-1]["record_sha256"]
                if self.retirement_records
                else None
            ),
            "plan_sha256": self._expected_plan_identity(
                {
                    "phase": phase,
                    "target_phase": None,
                }
            )[0],
            "phase": phase,
            "model": model,
            "setting": setting,
            "reason": "oom_profile_transition",
            "trigger": triggers[0],
            "old_profile": dict(old_profile),
            "replacement_profile": dict(replacement_profile),
            "retired_manifests": items,
        }
        record = {
            **base,
            "record_sha256": _sha256_bytes(_canonical_json_bytes(base)),
        }
        if self.execute:
            self.retirements_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                self.retirements_path,
                os.O_WRONLY | os.O_APPEND | os.O_CREAT,
                0o644,
            )
            try:
                os.write(descriptor, _canonical_json_bytes(record) + b"\n")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            directory_fd = os.open(
                self.retirements_path.parent, os.O_RDONLY
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        self.retirement_records.append(record)
        self.state["retirement_ledger"] = {
            "path": str(self.retirements_path),
            "records": len(self.retirement_records),
            "last_record_sha256": record["record_sha256"],
            "valid": True,
            "errors": [],
        }
        self.emit(
            "state",
            "profile_retired",
            details={
                "retirement_ledger": str(self.retirements_path),
                "retirement_seq": record["seq"],
                "retirement_record_sha256": record["record_sha256"],
                "plan_sha256": record["plan_sha256"],
                "phase": phase,
                "model": model,
                "setting": setting,
                "old_profile": dict(old_profile),
                "replacement_profile": dict(replacement_profile),
                "reason": record["reason"],
            },
        )
        return record

    def emit(
        self,
        category: str,
        code: str,
        *,
        details: Mapping[str, Any] | None = None,
        anomaly_key: str | None = None,
    ) -> dict[str, Any] | None:
        if anomaly_key is not None:
            known = self.state.setdefault("anomaly_keys", [])
            if anomaly_key in known:
                return None
            known.append(anomaly_key)
        sequence = int(self.state.get("event_seq", 0)) + 1
        event = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "seq": sequence,
            "timestamp_utc": _utc_now(),
            "campaign_id": self.state["campaign_id"],
            "category": category,
            "code": code,
            "details": dict(details or {}),
        }
        self.state["event_seq"] = sequence
        self.state["last_event"] = event
        if not self.execute:
            self.preview_events.append(event)
            return event
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self.events_path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o644,
        )
        try:
            os.write(
                descriptor,
                (_canonical_json_bytes(event) + b"\n"),
            )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return event

    def anomaly(
        self,
        code: str,
        *,
        details: Mapping[str, Any],
        key: str,
    ) -> None:
        self.emit(
            "anomaly",
            code,
            details=details,
            anomaly_key=f"{code}:{key}",
        )

    def persist(self) -> None:
        self.state["updated_utc"] = _utc_now()
        if self.execute:
            _atomic_write_json(self.state_path, self.state)

    def _refresh_plan_snapshot(self) -> bool:
        """Reload the plan so a long-lived controller cannot miss file drift."""
        try:
            raw = self.plan_path.read_bytes()
        except OSError as exc:
            self.anomaly(
                "config_drift",
                details={"plan": str(self.plan_path), "error": str(exc)},
                key=f"plan-read:{type(exc).__name__}:{exc}",
            )
            self.state["status"] = "NEEDS_USER_ACTION"
            self.state["blockers"] = [f"cannot reread campaign plan: {exc}"]
            return False
        current_sha = _sha256_bytes(raw)
        if current_sha == self.plan_sha256:
            return True
        try:
            current = json.loads(raw)
            runner.validate_structure(current)
        except (json.JSONDecodeError, runner.PlanError) as exc:
            self.anomaly(
                "config_drift",
                details={
                    "plan": str(self.plan_path),
                    "observed_sha256": current_sha,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                key=f"plan-invalid:{current_sha}",
            )
            self.state["status"] = "NEEDS_USER_ACTION"
            self.state["blockers"] = [
                "campaign plan changed to an invalid snapshot"
            ]
            return False
        self.plan_raw = raw
        self.plan = current
        self.plan_sha256 = current_sha
        return True

    def _check_plan_transition(self) -> bool:
        active = self.state["plan"]["active_sha256"]
        if active == self.plan_sha256:
            return True
        pending = self.state["plan"].get("pending_transition_sha256")
        if pending == self.plan_sha256:
            self.state["plan"]["active_sha256"] = self.plan_sha256
            self.state["plan"]["formal_sha256"] = self.plan_sha256
            self.state["plan"]["formal_protocol_sha256"] = _protocol_sha256(
                self.plan
            )
            self.state["plan"]["pending_transition_sha256"] = None
            self.emit(
                "state",
                "accepted_selected_lr_plan_transition",
                details={"plan_sha256": self.plan_sha256},
            )
            return True
        self.anomaly(
            "config_drift",
            details={
                "expected_plan_sha256": active,
                "actual_plan_sha256": self.plan_sha256,
            },
            key=f"plan:{active}:{self.plan_sha256}",
        )
        self.state["status"] = "NEEDS_USER_ACTION"
        self.state["blockers"] = ["plan changed outside an approved LR commit"]
        return False

    def _current_source_identity(self) -> dict[str, str]:
        tree = executor.collect_numerical_source_tree(cwd=self.repo_root)
        files = tree.get("files")
        if (
            not isinstance(files, list)
            or not files
            or any(
                not isinstance(item, Mapping)
                or item.get("stable_during_hash") is not True
                for item in files
            )
        ):
            raise CampaignBlocked(
                "numerical source tree was empty or changed while hashing"
            )
        return {
            "numerical_source_sha256": str(tree["combined_sha256"]),
            "runner_sha256": _sha256_file(Path(runner.__file__).resolve()),
            "executor_sha256": _sha256_file(Path(executor.__file__).resolve()),
            "campaign_sha256": _sha256_file(Path(__file__).resolve()),
            "results_sha256": _sha256_file(Path(results.__file__).resolve()),
        }

    def _check_source_identity(self) -> bool:
        try:
            current = self._current_source_identity()
        except (OSError, KeyError, CampaignBlocked) as exc:
            self.anomaly(
                "config_drift",
                details={"source_identity_error": str(exc)},
                key=f"source-read:{type(exc).__name__}:{exc}",
            )
            self.state["status"] = "NEEDS_USER_ACTION"
            self.state["blockers"] = [
                f"cannot establish stable numerical source identity: {exc}"
            ]
            return False
        expected = self.state.get("source_identity")
        if expected is None:
            self.state["source_identity"] = current
            self.emit(
                "state",
                "source_identity_pinned",
                details=current,
            )
            return True
        if expected != current:
            self.anomaly(
                "config_drift",
                details={
                    "expected_source_identity": expected,
                    "actual_source_identity": current,
                },
                key=_sha256_bytes(_canonical_json_bytes(current)),
            )
            self.state["status"] = "NEEDS_USER_ACTION"
            self.state["blockers"] = [
                "runner/executor/numerical source changed during the campaign"
            ]
            return False
        return True

    def _gates(self) -> tuple[bool, GPUSnapshot | None]:
        self.state["blockers"] = []
        retirement = self.state.get("retirement_ledger")
        if (
            isinstance(retirement, Mapping)
            and retirement.get("valid") is False
        ):
            self.state["status"] = "NEEDS_USER_ACTION"
            self.state["blockers"] = [
                "retirement ledger is invalid: "
                + "; ".join(str(value) for value in retirement.get("errors", []))
            ]
            return False, None
        stalled = [
            task["task_id"]
            for task in self.state["tasks"].values()
            if task.get("status") == "STALLED"
        ]
        if stalled:
            self.state["status"] = "NEEDS_USER_ACTION"
            self.state["blockers"] = [
                "executor progress watchdog expired; no new launches while "
                f"monitoring stalled task(s): {stalled}"
            ]
            return False, None
        if not self._refresh_plan_snapshot():
            return False, None
        ready, ready_reason = agent_ready(self.agent_path, self.ready_marker)
        self.state["agent"].update(
            {
                "ready": ready,
                "reason": ready_reason,
                "sha256": (
                    _sha256_file(self.agent_path)
                    if self.agent_path.is_file()
                    else None
                ),
            }
        )
        if not ready:
            self.state["status"] = "WAIT_ENV_READY"
            self.state["blockers"] = [ready_reason]
            return False, None

        environment_ok, environment = venv_ready(self.plan)
        self.state["agent"]["venv"] = environment
        if self.execute and not environment_ok:
            self.state["status"] = "WAIT_VENV_ACTIVATION"
            self.state["blockers"] = [
                "source the exact repository .venv before --execute"
            ]
            return False, None

        if not self._check_plan_transition():
            return False, None

        preflight, preflight_errors = load_valid_preflight(
            self.plan,
            plan_sha256=self.plan_sha256,
            repo_root=self.repo_root,
        )
        self.state["preflight"] = {
            "path": (
                str(_resolve_preflight_path(self.plan, repo_root=self.repo_root))
                if _resolve_preflight_path(
                    self.plan, repo_root=self.repo_root
                )
                else None
            ),
            "valid": not preflight_errors,
            "errors": preflight_errors,
            "sha256": (
                _sha256_file(
                    _resolve_preflight_path(
                        self.plan, repo_root=self.repo_root
                    )
                )
                if not preflight_errors
                and _resolve_preflight_path(
                    self.plan, repo_root=self.repo_root
                )
                is not None
                else None
            ),
        }
        if preflight_errors:
            if any(
                "Canoe" in error or "hostname" in error
                for error in preflight_errors
            ):
                self.anomaly(
                    "heartbeat_pod_context_invalid",
                    details={"errors": preflight_errors},
                    key=_sha256_bytes(
                        _canonical_json_bytes(preflight_errors)
                    ),
                )
            self.state["status"] = "WAIT_PREFLIGHT"
            self.state["blockers"] = preflight_errors
            return False, None

        if not self._check_source_identity():
            return False, None

        snapshot = query_gpus()
        self.state["heartbeat"] = {
            "timestamp_utc": _utc_now(),
            "hostname": socket.gethostname(),
            "gpu": snapshot.public_dict(),
        }
        if snapshot.error:
            self.anomaly(
                "heartbeat_gpu_probe_failed",
                details={"error": snapshot.error},
                key=snapshot.error,
            )
            if self.execute:
                self.state["status"] = "WAIT_GPUS"
                self.state["blockers"] = [snapshot.error]
                return False, snapshot
        requirements = self.plan["runtime_requirements"]
        eligible = eligible_gpu_ids(
            snapshot,
            minimum_memory_gib=float(
                requirements["minimum_gpu_memory_gib"]
            ),
        )
        if self.execute and len(eligible) < int(
            requirements["minimum_gpu_count"]
        ):
            message = (
                f"need {requirements['minimum_gpu_count']} eligible GPUs, "
                f"detected {eligible}"
            )
            self.anomaly(
                "heartbeat_gpu_requirement_failed",
                details={"message": message, "snapshot": snapshot.public_dict()},
                key=message,
            )
            self.state["status"] = "WAIT_GPUS"
            self.state["blockers"] = [message]
            return False, snapshot
        return True, snapshot

    def _memory_profiles(
        self,
        *,
        phase: str,
        static_only: bool,
    ) -> list[dict[str, int]]:
        policy = self.plan["qwen3_32b_memory_policy"]
        ladders = policy[
            "tuning_ladders" if phase == "tune" else "final_ladders"
        ]
        if static_only:
            ladders = {"global_loss_bsz": ladders["global_loss_bsz"]}
        else:
            # global_loss_bsz is part of the static-cache identity and is
            # negotiated by the single producer, not independently by an arm.
            ladders = {
                name: values
                for name, values in ladders.items()
                if name != "global_loss_bsz"
            }
        return _deduplicated_profiles(ladders)

    def _manifest_model_setting(
        self,
        argv: Sequence[str],
    ) -> tuple[str, str] | None:
        model_argument = _option(argv, "--model")
        if model_argument is None:
            return None
        cwd = self.repo_root
        argument_path = Path(model_argument)
        if not argument_path.is_absolute():
            argument_path = cwd / argument_path
        argument_path = argument_path.resolve(strict=False)
        models = [
            model
            for model, configured in self.plan["models"].items()
            if (
                configured == model_argument
                or (cwd / configured).resolve(strict=False) == argument_path
            )
        ]
        if len(models) != 1:
            return None
        bit_names = ("w_bits", "a_bits", "k_bits", "v_bits")
        try:
            bits = tuple(
                int(_option(argv, f"--{name}") or "") for name in bit_names
            )
        except ValueError:
            return None
        settings = [
            setting
            for setting, values in self.plan["settings"].items()
            if tuple(int(values[name]) for name in bit_names) == bits
        ]
        if len(settings) != 1:
            return None
        return models[0], settings[0]

    def _profile_level_from_argv(
        self,
        argv: Sequence[str],
        *,
        phase: str,
    ) -> tuple[int, int, dict[str, int]] | None:
        profiles = self._memory_profiles(phase=phase, static_only=False)
        nominal = self.plan["tuning" if phase == "tune" else "final"]
        names = set().union(*(profile.keys() for profile in profiles))
        names.add("global_loss_bsz")
        observed: dict[str, int] = {}
        for name in names:
            raw = _option(argv, f"--{name}")
            if raw is None:
                return None
            try:
                observed[name] = int(raw)
            except ValueError:
                return None
        matches = [
            index
            for index, profile in enumerate(profiles)
            if all(observed.get(name) == value for name, value in profile.items())
        ]
        if len(matches) != 1:
            return None
        static_profiles = self._memory_profiles(phase=phase, static_only=True)
        static_matches = [
            index
            for index, profile in enumerate(static_profiles)
            if profile["global_loss_bsz"] == observed["global_loss_bsz"]
        ]
        if len(static_matches) != 1:
            return None
        return (
            matches[0],
            static_matches[0],
            _effective_overrides(nominal, observed),
        )

    def _import_existing_tune_manifests(self) -> None:
        """Recover all launched candidates so state deletion cannot reset 20."""
        output_root = Path(self.plan["output_root"])
        if not output_root.is_absolute():
            output_root = self.repo_root / output_root
        imported_by_group: dict[str, list[dict[str, Any]]] = {}
        launched_outputs_by_group: dict[str, dict[str, int]] = {}
        if self.events_path.is_file():
            with self.events_path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    try:
                        event = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    if (
                        not isinstance(event, Mapping)
                        or event.get("category") != "task"
                        or event.get("code") != "task_launching"
                    ):
                        continue
                    details = event.get("details")
                    if not isinstance(details, Mapping):
                        continue
                    model = details.get("model")
                    setting = details.get("setting")
                    output_dir = details.get("output_dir")
                    group_key = f"{model}/{setting}"
                    if (
                        details.get("phase") != "tune"
                        or details.get("kind") != "realq"
                        or details.get("method") != "realq"
                        or details.get("plan_sha256")
                        != self.state["plan"].get("tune_sha256")
                        or
                        model not in runner.MODEL_ORDER
                        or setting not in runner.SETTING_ORDER
                        or not isinstance(output_dir, str)
                        or not output_dir
                    ):
                        continue
                    resolved_output = str(
                        Path(output_dir).resolve(strict=False)
                    )
                    counts = launched_outputs_by_group.setdefault(group_key, {})
                    counts[resolved_output] = counts.get(resolved_output, 0) + 1
        known_manifest_paths = {
            str(Path(task["manifest"]).resolve(strict=False))
            for task in self.state["tasks"].values()
            if task.get("manifest")
        }
        manifest_paths = (
            sorted(output_root.rglob(executor.MANIFEST_FILENAME))
            if output_root.is_dir()
            else []
        )
        for manifest_path in manifest_paths:
            resolved_manifest = str(manifest_path.resolve(strict=False))
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            command = manifest.get("command")
            argv = command.get("argv") if isinstance(command, Mapping) else None
            if (
                not isinstance(argv, list)
                or "realq.ptq" not in argv
                or not str(manifest.get("run_id", "")).startswith(
                    "tune_realq_"
                )
            ):
                continue
            identity = self._manifest_model_setting(argv)
            if identity is None:
                self.anomaly(
                    "config_drift",
                    details={
                        "manifest": resolved_manifest,
                        "reason": "cannot map tune manifest to model/setting",
                    },
                    key=f"import:{resolved_manifest}",
                )
                continue
            model, setting = identity
            group_key = f"{model}/{setting}"
            plan_compatible, plan_details = self._manifest_plan_matches(
                manifest,
                {"phase": "tune", "target_phase": None},
                require_finished=manifest.get("status") == "succeeded",
            )
            imported_by_group.setdefault(group_key, []).append(
                {
                    "manifest": resolved_manifest,
                    "payload": manifest,
                    "argv": argv,
                    "plan_compatible": plan_compatible,
                }
            )
            if resolved_manifest in known_manifest_paths:
                continue
            profile = self._profile_level_from_argv(argv, phase="tune")
            if profile is None:
                group = self.state["tuning"][group_key]
                group["status"] = "NEEDS_USER_ACTION"
                group["needs_user_action_reason"] = (
                    "existing tune manifest uses an unreviewed memory profile"
                )
                self.anomaly(
                    "config_drift",
                    details={
                        "manifest": resolved_manifest,
                        "reason": "unreviewed tune memory profile",
                    },
                    key=f"profile:{resolved_manifest}",
                )
                continue
            level, static_level, overrides = profile
            raw_lr = _option(argv, "--grad_lr")
            try:
                grad_lr = float(raw_lr) if raw_lr is not None else None
            except ValueError:
                grad_lr = None
            if grad_lr is None or not math.isfinite(grad_lr) or grad_lr < 0:
                group = self.state["tuning"][group_key]
                group["status"] = "NEEDS_USER_ACTION"
                group["needs_user_action_reason"] = (
                    "existing tune manifest has an invalid grad_lr"
                )
                continue
            task_id = (
                "imported_tune:"
                + _sha256_bytes(resolved_manifest.encode("utf-8"))[:16]
            )
            environment = (
                command.get("env") if isinstance(command, Mapping) else None
            )
            raw_attempt = (
                environment.get("LOWBIT_ACTIVATION_ATTEMPT_INDEX")
                if isinstance(environment, Mapping)
                else None
            )
            try:
                attempt_index = int(raw_attempt)
            except (TypeError, ValueError):
                attempt_index = 1
            self.state["tasks"][task_id] = {
                "task_id": task_id,
                "spec": {
                    "kind": "realq",
                    "method": "realq",
                    "phase": "tune",
                    "target_phase": None,
                    "model": model,
                    "setting": setting,
                    "grad_lr": grad_lr,
                    "overrides": overrides,
                    "profile_level": level,
                    "static_profile_level": static_level,
                    "generation": 0 if plan_compatible else -1,
                    "purpose": "imported_resume_attempt",
                },
                "status": "RUNNING" if plan_compatible else "SUPERSEDED",
                "gpu_count": 1,
                "priority": 0,
                "gpus": [],
                "attempt_index": max(attempt_index, 1),
                "executor_pid": manifest.get("process", {}).get("pid"),
                "created_utc": manifest.get("timestamps", {}).get(
                    "prepared_utc"
                ),
                "launched_utc": manifest.get("timestamps", {}).get(
                    "started_utc"
                )
                or manifest.get("timestamps", {}).get("prepared_utc"),
                "finished_utc": manifest.get("timestamps", {}).get(
                    "ended_utc"
                ),
                "output_dir": str(manifest_path.parent),
                "manifest": resolved_manifest,
                "log": str(manifest_path.parent / executor.LOG_FILENAME),
                "exit_code": manifest.get("exit_code"),
                "failure_class": None,
                "metric": None,
                "cache_validation": None,
                "imported": True,
                "plan_compatible": plan_compatible,
            }
            imported_task = self.state["tasks"][task_id]
            if (
                plan_compatible
                and manifest.get("status") in {"launching", "running"}
            ):
                self._adopt_manifest_process_identity(
                    imported_task, manifest
                )
            if not plan_compatible:
                self.anomaly(
                    "config_drift",
                    details={
                        "manifest": resolved_manifest,
                        "reason": "tune manifest plan/protocol identity mismatch",
                        **plan_details,
                    },
                    key=f"manifest-plan:{resolved_manifest}",
                )

        # Recompute, never increment, the immutable total from the union of
        # manifests plus launched state entries that have not published one yet.
        for group_key, group in self.state["tuning"].items():
            manifest_outputs = {
                str(Path(entry["manifest"]).resolve(strict=False).parent)
                for entry in imported_by_group.get(group_key, [])
            }
            state_outputs = {
                (
                    str(Path(task["output_dir"]).resolve(strict=False))
                    if task.get("output_dir")
                    else f"task:{task['task_id']}"
                )
                for task in self.state["tasks"].values()
                if task["spec"].get("kind") == "realq"
                and task["spec"].get("phase") == "tune"
                and f"{task['spec'].get('model')}/{task['spec'].get('setting')}"
                == group_key
                and task.get("launched_utc")
            }
            event_counts = launched_outputs_by_group.get(group_key, {})
            event_outputs = set(event_counts)
            # Every task_launching event is an attempt, even if the same output
            # identity was retried after external deletion. Manifests/state add
            # only launches not already represented by an event.
            group["attempt_count"] = (
                sum(event_counts.values())
                + len((manifest_outputs | state_outputs) - event_outputs)
            )
            if group["attempt_count"] > int(
                self.plan["tuning"]["max_attempts_per_model_setting"]
            ):
                group["status"] = "NEEDS_USER_ACTION"
                group["needs_user_action_reason"] = (
                    "existing manifests exceed the 20-attempt hard cap"
                )

        # If state was reconstructed, the highest reviewed profile is the only
        # homogeneous active generation; older profiles remain counted but are
        # explicitly superseded.
        for group_key, group in self.state["tuning"].items():
            imported = [
                task
                for task in self.state["tasks"].values()
                if task.get("imported")
                and task.get("plan_compatible", True)
                and f"{task['spec']['model']}/{task['spec']['setting']}"
                == group_key
            ]
            if not imported:
                continue
            highest = max(int(task["spec"]["profile_level"]) for task in imported)
            highest_static = max(
                int(task["spec"].get("static_profile_level", 0))
                for task in imported
            )
            model = str(group["model"])
            cache = self.state["static_caches"][f"tune/{model}"]
            if cache.get("producer_task") is None and cache.get("status") != "READY":
                cache["profile_level"] = highest_static
            if not any(
                not task.get("imported")
                for task in self._active_group_tasks(group)
            ):
                group["profile_level"] = highest
                group["generation"] = 0
            for task in imported:
                if (
                    int(task["spec"]["profile_level"])
                    != int(group["profile_level"])
                    or int(task["spec"].get("static_profile_level", 0))
                    != int(cache["profile_level"])
                ):
                    task["status"] = "SUPERSEDED"

    def _task_base_key(
        self,
        spec: Mapping[str, Any],
    ) -> str:
        identity = {
            key: spec.get(key)
            for key in (
                "kind",
                "phase",
                "target_phase",
                "method",
                "model",
                "setting",
                "grad_lr",
                "profile_level",
                "generation",
                "purpose",
                "attempt_index",
            )
        }
        identity["overrides"] = spec.get("overrides", {})
        return _sha256_bytes(_canonical_json_bytes(identity))[:16]

    def _create_task(
        self,
        spec: Mapping[str, Any],
        *,
        gpu_count: int,
        priority: int,
    ) -> str:
        attempt_index = spec.get("attempt_index", 1)
        if (
            type(attempt_index) is not int
            or attempt_index <= 0
            or attempt_index > executor.MAX_ATTEMPT_INDEX
        ):
            raise CampaignError("task attempt_index is outside the safe range")
        base = self._task_base_key(spec)
        task_id = f"{spec['kind']}:{base}"
        tasks = self.state["tasks"]
        if task_id not in tasks:
            task: dict[str, Any] = {
                "task_id": task_id,
                "spec": dict(spec),
                "status": "PENDING",
                "gpu_count": gpu_count,
                "priority": priority,
                "gpus": [],
                "attempt_index": attempt_index,
                "executor_pid": None,
                "created_utc": _utc_now(),
                "launched_utc": None,
                "finished_utc": None,
                "output_dir": None,
                "manifest": None,
                "log": None,
                "exit_code": None,
                "failure_class": None,
                "metric": None,
                "cache_validation": None,
            }
            # Adopt an immutable executor output left by a previous controller
            # before considering a launch. This covers static/final/baseline
            # resume; tune manifests are additionally imported globally for
            # hard-cap accounting.
            dummy_gpus = tuple(range(gpu_count))
            with contextlib.suppress(
                CampaignError, runner.PlanError, KeyError, ValueError
            ):
                rendered = self._render_task(task, dummy_gpus)
                output_dir = (
                    self.repo_root / rendered.output_dir
                    if not rendered.output_dir.is_absolute()
                    else rendered.output_dir
                ).resolve(strict=False)
                manifest_path = output_dir / executor.MANIFEST_FILENAME
                existing_task = next(
                    (
                        existing
                        for existing in tasks.values()
                        if existing.get("manifest")
                        and Path(existing["manifest"]).resolve(strict=False)
                        == manifest_path
                    ),
                    None,
                )
                if (
                    existing_task is not None
                    and existing_task.get("plan_compatible", True)
                ):
                    return str(existing_task["task_id"])
                task["output_dir"] = str(output_dir)
                task["manifest"] = str(manifest_path)
                task["log"] = str(output_dir / executor.LOG_FILENAME)
                if manifest_path.is_file():
                    try:
                        manifest = json.loads(
                            manifest_path.read_text(encoding="utf-8")
                        )
                    except (
                        OSError,
                        UnicodeDecodeError,
                        json.JSONDecodeError,
                    ):
                        manifest = {}
                    compatible, details = self._manifest_plan_matches(
                        manifest,
                        task["spec"],
                        require_finished=manifest.get("status") == "succeeded",
                    )
                    task["plan_compatible"] = compatible
                    task["imported"] = True
                    if compatible:
                        task["status"] = "RUNNING"
                        timestamps = manifest.get("timestamps")
                        if isinstance(timestamps, Mapping):
                            task["created_utc"] = (
                                timestamps.get("prepared_utc")
                                or task["created_utc"]
                            )
                            task["launched_utc"] = (
                                timestamps.get("started_utc")
                                or timestamps.get("prepared_utc")
                            )
                        if manifest.get("status") in {"launching", "running"}:
                            self._adopt_manifest_process_identity(
                                task, manifest
                            )
                    else:
                        task["status"] = "INVALID_RESULT"
                        task["failure_class"] = "plan_identity_mismatch"
                        self.anomaly(
                            "config_drift",
                            details={
                                "task_id": task_id,
                                "manifest": str(manifest_path),
                                "reason": (
                                    "preexisting output belongs to a different "
                                    "plan/protocol"
                                ),
                                **details,
                            },
                            key=f"adopt-plan:{manifest_path}",
                        )
                elif output_dir.exists() and any(output_dir.iterdir()):
                    task["status"] = "ORPHANED"
                    task["failure_class"] = "preexisting_without_manifest"
                    self.anomaly(
                        "orphaned_output",
                        details={
                            "task_id": task_id,
                            "output_dir": str(output_dir),
                        },
                        key=str(output_dir),
                    )
            tasks[task_id] = task
            self.emit(
                "task",
                "task_created",
                details={"task_id": task_id, "spec": dict(spec)},
            )
        return task_id

    def _render_task(
        self,
        task: Mapping[str, Any],
        gpu_ids: Sequence[int],
    ) -> runner.RenderedCommand:
        spec = task["spec"]
        cuda = ",".join(str(value) for value in gpu_ids)
        overrides = {
            str(key): int(value)
            for key, value in spec.get("overrides", {}).items()
        }
        if spec["kind"] == "realq_static":
            rendered = runner.render_realq_static_precompute(
                self.plan,
                target_phase=spec["target_phase"],
                model=spec["model"],
                cuda_devices=cuda,
                overrides=overrides or None,
            )
        elif spec["kind"] == "realq":
            rendered = runner.render_realq(
                self.plan,
                phase=spec["phase"],
                model=spec["model"],
                setting_name=spec["setting"],
                grad_lr=float(spec["grad_lr"]),
                cuda_devices=cuda,
                overrides=overrides or None,
            )
        elif spec["kind"] == "guided_saliency":
            rendered = runner.render_guided_saliency(
                self.plan,
                model=spec["model"],
                cuda_devices=cuda,
            )
        else:
            rendered = runner.render_baseline(
                self.plan,
                method=spec["method"],
                model=spec["model"],
                setting_name=spec.get("setting"),
                cuda_devices=cuda,
                overrides=overrides or None,
            )
        rendered = executor._with_attempt_identity(
            rendered,
            int(task["attempt_index"]),
        )
        validate_no_fsdp(rendered)
        return rendered

    def _executor_cli(
        self,
        task: Mapping[str, Any],
        gpu_ids: Sequence[int],
    ) -> list[str]:
        spec = task["spec"]
        command = [
            sys.executable,
            str(self.repo_root / "tools" / "lowbit_activation_execute.py"),
            "--plan",
            str(self.plan_path),
            "--execute",
            "--phase",
            (
                "precompute"
                if spec["kind"] in {"realq_static", "guided_saliency"}
                else spec["phase"]
            ),
            "--method",
            (
                spec["kind"]
                if spec["kind"] in {"realq_static", "guided_saliency"}
                else spec["method"]
            ),
            "--model",
            spec["model"],
            "--cuda-devices",
            ",".join(str(value) for value in gpu_ids),
            "--attempt-index",
            str(task["attempt_index"]),
        ]
        if spec.get("target_phase"):
            command.extend(["--target-phase", spec["target_phase"]])
        if spec.get("setting"):
            command.extend(["--setting", spec["setting"]])
        if spec.get("grad_lr") is not None:
            command.extend(["--grad-lr", str(spec["grad_lr"])])
        for key, value in sorted(spec.get("overrides", {}).items()):
            command.extend(["--override", f"{key}={value}"])
        return command

    def _static_ready(self, model: str, phase: str) -> bool:
        entry = self.state["static_caches"].get(f"{phase}/{model}")
        structurally_ready = (
            isinstance(entry, Mapping)
            and entry.get("status") == "READY"
            and isinstance(entry.get("validation"), Mapping)
            and entry["validation"].get("valid") is True
        )
        if not structurally_ready:
            return False
        assert isinstance(entry, dict)
        for file_meta in entry["validation"].get("files", []):
            path = Path(file_meta["path"])
            try:
                stat = path.stat()
            except OSError as exc:
                entry["status"] = "NEEDS_USER_ACTION"
                self.anomaly(
                    "cache_drift",
                    details={
                        "model": model,
                        "phase": phase,
                        "path": str(path),
                        "error": str(exc),
                    },
                    key=f"{phase}:{model}:{path}",
                )
                return False
            if (
                stat.st_size != int(file_meta["size_bytes"])
                or stat.st_mtime_ns != int(file_meta["mtime_ns"])
            ):
                entry["status"] = "NEEDS_USER_ACTION"
                self.anomaly(
                    "cache_drift",
                    details={
                        "model": model,
                        "phase": phase,
                        "path": str(path),
                        "expected_size": file_meta["size_bytes"],
                        "actual_size": stat.st_size,
                        "expected_mtime_ns": file_meta["mtime_ns"],
                        "actual_mtime_ns": stat.st_mtime_ns,
                    },
                    key=f"{phase}:{model}:{path}:stat",
                )
                return False
        return True

    def _ensure_tune_static_tasks(self) -> None:
        profiles = self._memory_profiles(phase="tune", static_only=True)
        for model in runner.MODEL_ORDER:
            key = f"tune/{model}"
            entry = self.state["static_caches"][key]
            if entry["status"] in {
                "READY",
                "RUNNING",
                "QUEUED",
                "NEEDS_USER_ACTION",
            }:
                continue
            level = int(entry["profile_level"])
            if level >= len(profiles):
                entry["status"] = "NEEDS_USER_ACTION"
                entry["reason"] = "static precompute exhausted OOM ladder"
                continue
            nominal = self.plan["tuning"]
            overrides = _effective_overrides(nominal, profiles[level])
            task_id = self._create_task(
                {
                    "kind": "realq_static",
                    "method": "realq_static",
                    "phase": "precompute",
                    "target_phase": "tune",
                    "model": model,
                    "setting": None,
                    "grad_lr": None,
                    "overrides": overrides,
                    "profile_level": level,
                    "generation": 0,
                    "purpose": "tune_static_single_producer",
                    "attempt_index": int(
                        entry.get("next_attempt_index", 1)
                    ),
                },
                gpu_count=1,
                priority=100,
            )
            entry["producer_task"] = task_id
            entry["status"] = "QUEUED"

    def _active_group_tasks(
        self,
        group: Mapping[str, Any],
        *,
        include_terminal: bool = True,
    ) -> list[dict[str, Any]]:
        result_tasks = []
        for task in self.state["tasks"].values():
            spec = task["spec"]
            if (
                spec.get("kind") == "realq"
                and spec.get("phase") == "tune"
                and spec.get("model") == group["model"]
                and spec.get("setting") == group["setting"]
                and int(spec.get("generation", -1))
                == int(group["generation"])
                and int(spec.get("profile_level", -1))
                == int(group["profile_level"])
            ):
                if include_terminal or task["status"] not in TERMINAL_TASK_STATES:
                    result_tasks.append(task)
        return result_tasks

    def _group_metrics(
        self,
        group: Mapping[str, Any],
    ) -> dict[float, dict[str, float]]:
        metrics: dict[float, dict[str, float]] = {}
        for task in self._active_group_tasks(group):
            if task["status"] == "SUCCEEDED" and task.get("metric"):
                metrics[float(task["spec"]["grad_lr"])] = {
                    "kl": float(task["metric"]["kl"]),
                    "ppl": float(task["metric"]["ppl"]),
                }
        return metrics

    def _group_pending_lrs(self, group: Mapping[str, Any]) -> set[float]:
        return {
            float(task["spec"]["grad_lr"])
            for task in self._active_group_tasks(group)
            if task["status"] in {"PENDING", "LAUNCHING", "RUNNING"}
        }

    def _tune_profile(self, group: Mapping[str, Any]) -> dict[str, int]:
        profiles = self._memory_profiles(phase="tune", static_only=False)
        level = int(group["profile_level"])
        if level >= len(profiles):
            raise CampaignBlocked("tuning OOM profile ladder exhausted")
        result = dict(profiles[level])
        cache = self.state["static_caches"][f"tune/{group['model']}"]
        static_profiles = self._memory_profiles(
            phase="tune", static_only=True
        )
        static_level = int(cache["profile_level"])
        result.update(static_profiles[static_level])
        return result

    def _add_tune_candidates(
        self,
        group: dict[str, Any],
        lrs: Iterable[float],
        *,
        purpose: str,
        priority: int,
    ) -> None:
        profile = self._tune_profile(group)
        overrides = _effective_overrides(self.plan["tuning"], profile)
        existing = {
            float(task["spec"]["grad_lr"])
            for task in self._active_group_tasks(group)
        }
        for lr in lrs:
            value = float(lr)
            if value in existing:
                continue
            self._create_task(
                {
                    "kind": "realq",
                    "method": "realq",
                    "phase": "tune",
                    "target_phase": None,
                    "model": group["model"],
                    "setting": group["setting"],
                    "grad_lr": value,
                    "overrides": overrides,
                    "profile_level": int(group["profile_level"]),
                    "static_profile_level": int(
                        self.state["static_caches"][
                            f"tune/{group['model']}"
                        ]["profile_level"]
                    ),
                    "generation": int(group["generation"]),
                    "purpose": purpose,
                },
                gpu_count=1,
                priority=priority,
            )

    def _handle_group_oom(self, group: dict[str, Any]) -> None:
        active_tasks = self._active_group_tasks(group)
        if not any(task["status"] == "OOM" for task in active_tasks):
            return
        # Stop releasing the rest of this homogeneous wave as soon as one
        # candidate OOMs. Pending tasks have not consumed an attempt and will
        # be recreated under the next reviewed profile after running siblings
        # drain.
        for task in active_tasks:
            if task["status"] == "PENDING":
                task["status"] = "SUPERSEDED"
        if any(task["status"] in ACTIVE_TASK_STATES for task in active_tasks):
            group["status"] = "OOM_DRAINING"
            return
        non_oom_failures = [
            task
            for task in active_tasks
            if task["status"]
            in {"FAILED", "INVALID_RESULT", "ORPHANED", "LAUNCH_FAILED"}
        ]
        if non_oom_failures:
            group["status"] = "NEEDS_USER_ACTION"
            group["needs_user_action_reason"] = (
                "OOM wave also contains a non-OOM failure: "
                + ", ".join(task["task_id"] for task in non_oom_failures)
            )
            return
        profiles = self._memory_profiles(phase="tune", static_only=False)
        next_level = int(group["profile_level"]) + 1
        max_attempts = int(self.plan["tuning"]["max_attempts_per_model_setting"])
        coarse_count = len(self.plan["tuning"]["lr_candidates"])
        reserve = int(
            self.plan["tuning"]["search_policy"][
                "reserved_refinement_attempts"
            ]
        )
        if (
            next_level >= len(profiles)
            or int(group["attempt_count"]) + coarse_count + reserve
            > max_attempts
        ):
            group["status"] = "NEEDS_USER_ACTION"
            group["needs_user_action_reason"] = (
                "OOM requires a homogeneous profile restart, but the reviewed "
                "ladder or 20-attempt budget is exhausted"
            )
            return
        old_override_values = {
            _canonical_json_bytes(task["spec"].get("overrides", {}))
            for task in active_tasks
            if task.get("manifest")
        }
        if len(old_override_values) != 1:
            group["status"] = "NEEDS_USER_ACTION"
            group["needs_user_action_reason"] = (
                "OOM wave is not homogeneous in effective overrides"
            )
            return
        old_overrides = json.loads(next(iter(old_override_values)))
        replacement_effective = dict(profiles[next_level])
        cache = self.state["static_caches"][f"tune/{group['model']}"]
        static_profiles = self._memory_profiles(
            phase="tune", static_only=True
        )
        replacement_effective.update(
            static_profiles[int(cache["profile_level"])]
        )
        replacement_overrides = _effective_overrides(
            self.plan["tuning"], replacement_effective
        )
        try:
            self._append_profile_retirement(
                phase="tune",
                model=str(group["model"]),
                setting=str(group["setting"]),
                tasks=active_tasks,
                old_profile={
                    "level": int(group["profile_level"]),
                    "generation": int(group["generation"]),
                    "effective_overrides": old_overrides,
                },
                replacement_profile={
                    "level": next_level,
                    "generation": int(group["generation"]) + 1,
                    "effective_overrides": replacement_overrides,
                },
            )
        except CampaignBlocked as exc:
            group["status"] = "NEEDS_USER_ACTION"
            group["needs_user_action_reason"] = str(exc)
            self.anomaly(
                "retirement_ledger_invalid",
                details={
                    "model": group["model"],
                    "setting": group["setting"],
                    "error": str(exc),
                },
                key=f"tune:{group['model']}:{group['setting']}:{group['generation']}",
            )
            return
        for task in active_tasks:
            if task["status"] != "OOM":
                task["status"] = "SUPERSEDED"
        group["profile_level"] = next_level
        group["generation"] = int(group["generation"]) + 1
        group["canary_succeeded"] = False
        group["refinement_done"] = False
        group["zero_probes_done"] = []
        group["selected_lr"] = None
        group["selected_manifest"] = None
        group["status"] = "CANARY"
        self.emit(
            "state",
            "tune_profile_downgraded",
            details={
                "model": group["model"],
                "setting": group["setting"],
                "profile_level": next_level,
                "attempt_count": group["attempt_count"],
            },
        )

    def _advance_tuning_groups(self) -> None:
        coarse = [float(value) for value in self.plan["tuning"]["lr_candidates"]]
        policy = self.plan["tuning"]["search_policy"]
        max_attempts = int(self.plan["tuning"]["max_attempts_per_model_setting"])
        for key, group in self.state["tuning"].items():
            if group["status"] in {"SELECTED", "NEEDS_USER_ACTION"}:
                continue
            if not self._static_ready(group["model"], "tune"):
                group["status"] = "WAIT_STATIC"
                continue
            self._handle_group_oom(group)
            if group["status"] in {"OOM_DRAINING", "NEEDS_USER_ACTION"}:
                continue
            active_tasks = self._active_group_tasks(group)
            invalid = [
                task
                for task in active_tasks
                if task["status"]
                in {"FAILED", "INVALID_RESULT", "ORPHANED", "LAUNCH_FAILED"}
            ]
            if invalid:
                for task in active_tasks:
                    if task["status"] == "PENDING":
                        task["status"] = "SUPERSEDED"
                group["status"] = "NEEDS_USER_ACTION"
                group["needs_user_action_reason"] = (
                    "non-OOM candidate failure requires inspection: "
                    + ", ".join(task["task_id"] for task in invalid)
                )
                continue

            canary_lr = float(group["canary_lr"])
            canary_tasks = [
                task
                for task in active_tasks
                if float(task["spec"]["grad_lr"]) == canary_lr
            ]
            if not canary_tasks:
                metrics = self._group_metrics(group)
                pending = self._group_pending_lrs(group)
                missing_coarse = set(coarse) - set(metrics) - pending
                reserve = int(policy["reserved_refinement_attempts"])
                if (
                    int(group["attempt_count"])
                    + len(missing_coarse)
                    + reserve
                    > max_attempts
                ):
                    group["status"] = "NEEDS_USER_ACTION"
                    group["needs_user_action_reason"] = (
                        f"not enough of the {max_attempts}-attempt budget "
                        "remains to launch the memory canary, complete the "
                        "coarse round, and reserve local refinement"
                    )
                    continue
                group["status"] = "CANARY"
                self._add_tune_candidates(
                    group,
                    [canary_lr],
                    purpose="memory_canary",
                    priority=95,
                )
                continue
            if not any(task["status"] == "SUCCEEDED" for task in canary_tasks):
                group["status"] = "CANARY"
                continue
            group["canary_succeeded"] = True

            metrics = self._group_metrics(group)
            pending = self._group_pending_lrs(group)
            missing_coarse = set(coarse) - set(metrics) - pending
            if missing_coarse:
                reserve = int(
                    policy["reserved_refinement_attempts"]
                )
                if (
                    int(group["attempt_count"])
                    + len(missing_coarse)
                    + reserve
                    > max_attempts
                ):
                    group["status"] = "NEEDS_USER_ACTION"
                    group["needs_user_action_reason"] = (
                        f"not enough of the {max_attempts}-attempt budget "
                        "remains to "
                        "complete the coarse round and reserve local refinement"
                    )
                    continue
                group["status"] = "COARSE"
                self._add_tune_candidates(
                    group,
                    sorted(missing_coarse),
                    purpose="coarse",
                    priority=75,
                )
                continue
            if pending:
                group["status"] = (
                    "REFINE"
                    if group.get("search_stage") == "local_refine"
                    else "COARSE_OR_EXPAND"
                )
                continue

            decision = search_decision(
                metrics=metrics,
                coarse_lrs=coarse,
                attempted_count=int(group["attempt_count"]),
                pending_count=0,
                refinement_done=bool(group["refinement_done"]),
                zero_probes_done=group["zero_probes_done"],
                policy=policy,
                max_attempts=max_attempts,
            )
            group["last_search_decision"] = decision
            anomaly_points = [
                (lr, metric["kl"], metric["ppl"])
                for lr, metric in metrics.items()
            ]
            for anomaly in metric_anomalies(anomaly_points):
                self.anomaly(
                    anomaly["code"],
                    details={"group": key, **anomaly},
                    key=f"{key}:{_sha256_bytes(_canonical_json_bytes(anomaly))}",
                )

            action = decision["action"]
            if action == "needs_user_action":
                group["status"] = "NEEDS_USER_ACTION"
                group["needs_user_action_reason"] = decision["reason"]
            elif action == "select":
                selected_lr = float(decision["selected_lr"])
                selected_task = next(
                    task
                    for task in active_tasks
                    if task["status"] == "SUCCEEDED"
                    and float(task["spec"]["grad_lr"]) == selected_lr
                )
                group["status"] = "SELECTED"
                group["selected_lr"] = selected_lr
                group["selected_manifest"] = selected_task["manifest"]
                group["selected_parameters"] = {
                    "grad_lr": selected_lr,
                    "profile_level": group["profile_level"],
                    "overrides": selected_task["spec"]["overrides"],
                    "final_layer_grad_lr": selected_task["metric"].get(
                        "final_layer_grad_lr"
                    ),
                    "effective_command_params": selected_task["metric"].get(
                        "params"
                    ),
                    "result_identities": selected_task["metric"].get(
                        "identities"
                    ),
                    "controller_source_identity": dict(
                        self.state.get("source_identity") or {}
                    ),
                    "attempt_count_at_selection": group["attempt_count"],
                    "selected_task_attempt_index": selected_task.get(
                        "attempt_index"
                    ),
                    "manifest": selected_task["manifest"],
                }
                self.emit(
                    "selection",
                    "best_lr_selected",
                    details={
                        "model": group["model"],
                        "setting": group["setting"],
                        "grad_lr": selected_lr,
                        "attempt_count": group["attempt_count"],
                        "parameters": group["selected_parameters"],
                    },
                )
            else:
                candidates = [float(value) for value in decision["candidates"]]
                if (
                    int(group["attempt_count"])
                    + len(candidates)
                    > max_attempts
                ):
                    group["status"] = "NEEDS_USER_ACTION"
                    group["needs_user_action_reason"] = (
                        "candidate batch would exceed the 20-attempt hard cap"
                    )
                    continue
                if action == "local_refine":
                    group["search_stage"] = "local_refine"
                    group["refinement_done"] = True
                    priority = 90
                elif action == "expand_zero_boundary":
                    group["search_stage"] = "zero_boundary"
                    group["zero_probes_done"] = sorted(
                        set(group["zero_probes_done"]) | set(candidates)
                    )
                    priority = 85
                else:
                    group["search_stage"] = "upper_boundary"
                    priority = 85
                group["status"] = "REFINE_OR_EXPAND"
                self._add_tune_candidates(
                    group,
                    candidates,
                    purpose=action,
                    priority=priority,
                )

    def _formal_plan_matches_selection(self) -> bool:
        patch = selected_lr_patch(self.state)
        if patch is None:
            return False
        return (
            self.plan.get("selected_grad_lr_by_model_setting")
            == patch["selected_grad_lr_by_model_setting"]
        )

    def _initialize_formal(self) -> None:
        if self.state["formal"]["initialized"]:
            return
        final_models: dict[str, Any] = {}
        for model in runner.MODEL_ORDER:
            final_models[model] = {
                "static_status": "PENDING",
                "static_profile_level": 0,
                "static_task": None,
                "realq_profile_level": 0,
                "realq_generation": 0,
                "realq_canary_succeeded": False,
                "status": "PENDING",
                "guided_status": "PENDING",
                "guided_task": None,
            }
            self.state["static_caches"][f"final/{model}"] = {
                "phase": "final",
                "model": model,
                "status": "PENDING",
                "profile_level": 0,
                "producer_task": None,
                "validation": None,
            }
        self.state["formal"] = {
            "initialized": True,
            "models": final_models,
        }
        self._apply_retirements_to_state(phases={"final"})
        self.emit("state", "formal_dag_initialized")

    def _formal_realq_tasks(
        self,
        model: str,
        model_state: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        return [
            task
            for task in self.state["tasks"].values()
            if task["spec"].get("kind") == "realq"
            and task["spec"].get("phase") == "final"
            and task["spec"].get("model") == model
            and int(task["spec"].get("profile_level", -1))
            == int(model_state["realq_profile_level"])
            and int(task["spec"].get("generation", -1))
            == int(model_state["realq_generation"])
        ]

    def _advance_formal_realq_profile(
        self,
        model: str,
        model_state: dict[str, Any],
        profiles: Sequence[Mapping[str, int]],
    ) -> bool:
        """Return true when task creation may continue at the active profile."""
        tasks = self._formal_realq_tasks(model, model_state)
        failures = [
            task
            for task in tasks
            if task["status"]
            in {"FAILED", "INVALID_RESULT", "ORPHANED", "LAUNCH_FAILED"}
        ]
        if failures:
            for task in tasks:
                if task["status"] == "PENDING":
                    task["status"] = "SUPERSEDED"
            model_state["status"] = "NEEDS_USER_ACTION"
            model_state["reason"] = (
                "formal REAL-Q non-OOM failure: "
                + ", ".join(task["task_id"] for task in failures)
            )
            return False
        if not any(task["status"] == "OOM" for task in tasks):
            return True
        # Do not launch more settings with a profile that has already OOMed.
        # Running siblings drain naturally; never kill executor processes.
        for task in tasks:
            if task["status"] == "PENDING":
                task["status"] = "SUPERSEDED"
        if any(task["status"] in ACTIVE_TASK_STATES for task in tasks):
            model_state["status"] = "OOM_DRAINING"
            return False
        next_level = int(model_state["realq_profile_level"]) + 1
        if next_level >= len(profiles):
            model_state["status"] = "NEEDS_USER_ACTION"
            model_state["reason"] = (
                "formal REAL-Q exhausted every reviewed batch-size profile; "
                "FSDP remains forbidden"
            )
            return False
        old_override_values = {
            _canonical_json_bytes(task["spec"].get("overrides", {}))
            for task in tasks
            if task.get("manifest")
        }
        if len(old_override_values) != 1:
            model_state["status"] = "NEEDS_USER_ACTION"
            model_state["reason"] = (
                "formal OOM wave is not homogeneous in effective overrides"
            )
            return False
        old_overrides = json.loads(next(iter(old_override_values)))
        replacement_effective = dict(profiles[next_level])
        static_profiles = self._memory_profiles(
            phase="final", static_only=True
        )
        static_level = int(
            self.state["static_caches"][f"final/{model}"]["profile_level"]
        )
        replacement_effective.update(static_profiles[static_level])
        replacement_overrides = _effective_overrides(
            self.plan["final"], replacement_effective
        )
        try:
            self._append_profile_retirement(
                phase="final",
                model=model,
                setting=None,
                tasks=tasks,
                old_profile={
                    "level": int(model_state["realq_profile_level"]),
                    "generation": int(model_state["realq_generation"]),
                    "effective_overrides": old_overrides,
                },
                replacement_profile={
                    "level": next_level,
                    "generation": int(model_state["realq_generation"]) + 1,
                    "effective_overrides": replacement_overrides,
                },
            )
        except CampaignBlocked as exc:
            model_state["status"] = "NEEDS_USER_ACTION"
            model_state["reason"] = str(exc)
            self.anomaly(
                "retirement_ledger_invalid",
                details={"model": model, "phase": "final", "error": str(exc)},
                key=f"final:{model}:{model_state['realq_generation']}",
            )
            return False
        for task in tasks:
            if task["status"] != "OOM":
                task["status"] = "SUPERSEDED"
        model_state["realq_profile_level"] = next_level
        model_state["realq_generation"] = (
            int(model_state["realq_generation"]) + 1
        )
        model_state["realq_canary_succeeded"] = False
        model_state["status"] = "PENDING"
        self.emit(
            "state",
            "formal_realq_profile_downgraded",
            details={
                "model": model,
                "profile_level": next_level,
                "generation": model_state["realq_generation"],
                "fsdp": False,
            },
        )
        return True

    def _advance_baseline_ooms(self) -> None:
        ladder = [
            int(value)
            for value in self.plan["qwen3_32b_memory_policy"][
                "final_ladders"
            ]["lm_eval_batch_size"]
        ]
        for task in list(self.state["tasks"].values()):
            spec = task["spec"]
            if (
                spec.get("kind") != "baseline"
                or task["status"] != "OOM"
                or task.get("retry_scheduled")
            ):
                continue
            current = int(
                spec.get("overrides", {}).get(
                    "lm_eval_batch_size",
                    self.plan["final"]["lm_eval_batch_size"],
                )
            )
            try:
                index = ladder.index(current)
            except ValueError:
                task["retry_scheduled"] = False
                task["needs_user_action_reason"] = (
                    "baseline OOM used an unreviewed eval batch size"
                )
                continue
            if index + 1 >= len(ladder):
                task["retry_scheduled"] = False
                task["needs_user_action_reason"] = (
                    "baseline OOM exhausted lm_eval_batch_size ladder"
                )
                continue
            next_value = ladder[index + 1]
            retry_spec = dict(spec)
            retry_spec["overrides"] = {"lm_eval_batch_size": next_value}
            retry_spec["profile_level"] = int(spec.get("profile_level", 0)) + 1
            retry_spec["purpose"] = "formal_baseline_oom_retry"
            retry_id = self._create_task(
                retry_spec,
                gpu_count=1,
                priority=55,
            )
            task["retry_scheduled"] = True
            task["retry_task"] = retry_id
            self.emit(
                "state",
                "baseline_eval_batch_downgraded",
                details={
                    "task_id": task["task_id"],
                    "retry_task": retry_id,
                    "lm_eval_batch_size": next_value,
                    "fsdp": False,
                },
            )

    def _ensure_formal_tasks(self) -> None:
        self._initialize_formal()
        static_profiles = self._memory_profiles(
            phase="final", static_only=True
        )
        realq_profiles = self._memory_profiles(
            phase="final", static_only=False
        )
        for model, model_state in self.state["formal"]["models"].items():
            cache = self.state["static_caches"][f"final/{model}"]
            if cache["status"] not in {
                "READY",
                "RUNNING",
                "QUEUED",
                "NEEDS_USER_ACTION",
            }:
                level = int(cache["profile_level"])
                if level >= len(static_profiles):
                    cache["status"] = "NEEDS_USER_ACTION"
                else:
                    overrides = _effective_overrides(
                        self.plan["final"], static_profiles[level]
                    )
                    task_id = self._create_task(
                        {
                            "kind": "realq_static",
                            "method": "realq_static",
                            "phase": "precompute",
                            "target_phase": "final",
                            "model": model,
                            "setting": None,
                            "grad_lr": None,
                            "overrides": overrides,
                            "profile_level": level,
                            "generation": 0,
                            "purpose": "final_static_single_producer",
                            "attempt_index": int(
                                cache.get("next_attempt_index", 1)
                            ),
                        },
                        gpu_count=4,
                        priority=100,
                    )
                    cache["producer_task"] = task_id
                    cache["status"] = "QUEUED"

            if model_state["guided_status"] == "PENDING":
                task_id = self._create_task(
                    {
                        "kind": "guided_saliency",
                        "method": "guided_saliency",
                        "phase": "precompute",
                        "target_phase": None,
                        "model": model,
                        "setting": None,
                        "grad_lr": None,
                        "overrides": {},
                        "profile_level": 0,
                        "generation": 0,
                        "purpose": "guided_single_producer",
                    },
                    gpu_count=1,
                    priority=60,
                )
                model_state["guided_task"] = task_id
                model_state["guided_status"] = "QUEUED"

            # BF16 and GPTAQ are independent formal work. Guided arms are
            # released only after the one-per-model saliency producer succeeds.
            self._create_task(
                {
                    "kind": "baseline",
                    "method": "bf16",
                    "phase": "final",
                    "target_phase": None,
                    "model": model,
                    "setting": None,
                    "grad_lr": None,
                    "overrides": {},
                    "profile_level": 0,
                    "generation": 0,
                    "purpose": "formal_baseline",
                },
                gpu_count=1,
                priority=45,
            )
            for setting in runner.SETTING_ORDER:
                self._create_task(
                    {
                        "kind": "baseline",
                        "method": "gptaq",
                        "phase": "final",
                        "target_phase": None,
                        "model": model,
                        "setting": setting,
                        "grad_lr": None,
                        "overrides": {},
                        "profile_level": 0,
                        "generation": 0,
                        "purpose": "formal_baseline",
                    },
                    gpu_count=1,
                    priority=45,
                )
                if model_state["guided_status"] == "READY":
                    self._create_task(
                        {
                            "kind": "baseline",
                            "method": "guided_gptq",
                            "phase": "final",
                            "target_phase": None,
                            "model": model,
                            "setting": setting,
                            "grad_lr": None,
                            "overrides": {},
                            "profile_level": 0,
                            "generation": 0,
                            "purpose": "formal_baseline",
                        },
                        gpu_count=1,
                        priority=45,
                    )

            if not self._static_ready(model, "final"):
                continue
            profile_level = int(model_state["realq_profile_level"])
            if profile_level >= len(realq_profiles):
                model_state["status"] = "NEEDS_USER_ACTION"
                continue
            if not self._advance_formal_realq_profile(
                model, model_state, realq_profiles
            ):
                continue
            profile_level = int(model_state["realq_profile_level"])
            profile = dict(realq_profiles[profile_level])
            static_level = int(
                self.state["static_caches"][f"final/{model}"][
                    "profile_level"
                ]
            )
            profile.update(static_profiles[static_level])
            overrides = _effective_overrides(self.plan["final"], profile)
            canary_setting = "2W4A"
            existing = self._formal_realq_tasks(model, model_state)
            if not existing:
                selected = float(
                    self.plan["selected_grad_lr_by_model_setting"][model][
                        canary_setting
                    ]
                )
                self._create_task(
                    {
                        "kind": "realq",
                        "method": "realq",
                        "phase": "final",
                        "target_phase": None,
                        "model": model,
                        "setting": canary_setting,
                        "grad_lr": selected,
                        "overrides": overrides,
                        "profile_level": profile_level,
                        "generation": int(model_state["realq_generation"]),
                        "purpose": "formal_memory_canary",
                    },
                    gpu_count=4,
                    priority=95,
                )
                continue
            canary_success = any(
                task["spec"].get("setting") == canary_setting
                and task["status"] == "SUCCEEDED"
                for task in existing
            )
            if not canary_success:
                continue
            model_state["realq_canary_succeeded"] = True
            existing_settings = {
                task["spec"]["setting"] for task in existing
            }
            for setting in runner.SETTING_ORDER:
                if setting in existing_settings:
                    continue
                selected = float(
                    self.plan["selected_grad_lr_by_model_setting"][model][
                        setting
                    ]
                )
                self._create_task(
                    {
                        "kind": "realq",
                        "method": "realq",
                        "phase": "final",
                        "target_phase": None,
                        "model": model,
                        "setting": setting,
                        "grad_lr": selected,
                        "overrides": overrides,
                        "profile_level": profile_level,
                        "generation": int(model_state["realq_generation"]),
                        "purpose": "formal_result",
                    },
                    gpu_count=4,
                    priority=90,
                )
        self._advance_baseline_ooms()

    def _formal_identity_succeeded(
        self,
        *,
        model: str,
        method: str,
        setting: str | None,
    ) -> bool:
        return any(
            task["status"] == "SUCCEEDED"
            and task["spec"].get("kind") == "baseline"
            and task["spec"].get("model") == model
            and task["spec"].get("method") == method
            and task["spec"].get("setting") == setting
            for task in self.state["tasks"].values()
        )

    def _formal_campaign_status(self) -> str:
        if not self.state["formal"]["initialized"]:
            return "FORMAL"
        for model, model_state in self.state["formal"]["models"].items():
            if model_state.get("status") == "NEEDS_USER_ACTION":
                return "NEEDS_USER_ACTION"
            cache = self.state["static_caches"][f"final/{model}"]
            if cache.get("status") == "NEEDS_USER_ACTION":
                return "NEEDS_USER_ACTION"
            guided_task = self.state["tasks"].get(model_state.get("guided_task"))
            if guided_task and guided_task["status"] in {
                "FAILED",
                "OOM",
                "INVALID_RESULT",
                "ORPHANED",
                "LAUNCH_FAILED",
            }:
                return "NEEDS_USER_ACTION"
        terminal_baseline_failures = [
            task
            for task in self.state["tasks"].values()
            if task["spec"].get("kind") == "baseline"
            and task["status"]
            in {"FAILED", "INVALID_RESULT", "ORPHANED", "LAUNCH_FAILED"}
        ]
        if terminal_baseline_failures:
            return "NEEDS_USER_ACTION"
        exhausted_baseline_ooms = [
            task
            for task in self.state["tasks"].values()
            if task["spec"].get("kind") == "baseline"
            and task["status"] == "OOM"
            and task.get("retry_scheduled") is False
        ]
        if exhausted_baseline_ooms:
            return "NEEDS_USER_ACTION"

        for model, model_state in self.state["formal"]["models"].items():
            if not self._static_ready(model, "final"):
                return "FORMAL"
            if model_state.get("guided_status") != "READY":
                return "FORMAL"
            current_realq = self._formal_realq_tasks(model, model_state)
            succeeded_settings = {
                task["spec"]["setting"]
                for task in current_realq
                if task["status"] == "SUCCEEDED"
            }
            if succeeded_settings != set(runner.SETTING_ORDER):
                return "FORMAL"
            if not self._formal_identity_succeeded(
                model=model, method="bf16", setting=None
            ):
                return "FORMAL"
            for setting in runner.SETTING_ORDER:
                if not self._formal_identity_succeeded(
                    model=model, method="gptaq", setting=setting
                ):
                    return "FORMAL"
                if not self._formal_identity_succeeded(
                    model=model, method="guided_gptq", setting=setting
                ):
                    return "FORMAL"
        return "COMPLETE"

    def _task_manifest_path(
        self,
        task: dict[str, Any],
    ) -> Path | None:
        if task.get("manifest"):
            return Path(task["manifest"])
        if task.get("output_dir"):
            return Path(task["output_dir"]) / executor.MANIFEST_FILENAME
        return None

    def _task_process_alive(self, task: dict[str, Any]) -> bool:
        """Check both liveness and the persisted PID start identity."""
        process = self.processes.get(str(task.get("task_id")))
        if process is not None and process.poll() is not None:
            return False
        pid = task.get("executor_pid")
        if type(pid) is not int or pid <= 0:
            return False
        identity = _proc_identity(pid)
        if identity is None:
            return False
        expected_start = task.get("executor_start_ticks")
        expected_session = task.get("executor_session_id")
        expected_process_group = task.get("executor_process_group_id")
        if type(expected_start) is int and (
            identity["start_ticks"] != expected_start
        ):
            return False
        if type(expected_session) is int and (
            identity["session_id"] != expected_session
        ):
            return False
        if type(expected_process_group) is int and (
            identity.get("process_group_id", identity["session_id"])
            != expected_process_group
        ):
            return False
        if expected_start is None and process is None:
            # A state reconstructed without a PID start time cannot
            # distinguish a live executor from PID reuse.  Fail closed after
            # the normal dead-process grace rather than adopting by number.
            return False
        task["executor_start_ticks"] = identity["start_ticks"]
        task["executor_session_id"] = identity["session_id"]
        task["executor_process_group_id"] = identity.get(
            "process_group_id", identity["session_id"]
        )
        return True

    def _adopt_manifest_process_identity(
        self,
        task: dict[str, Any],
        manifest: Mapping[str, Any],
    ) -> bool:
        command_matches, command_details = (
            self._manifest_task_command_identity(task, manifest)
        )
        if not command_matches:
            task["active_manifest_identity_error"] = command_details
            return False
        task["gpus"] = list(command_details["gpu_ids"])
        process = manifest.get("process")
        command = manifest.get("command")
        pid = process.get("pid") if isinstance(process, Mapping) else None
        argv = command.get("argv") if isinstance(command, Mapping) else None
        if type(pid) is not int or not isinstance(argv, list) or not argv:
            return False
        actual = _proc_cmdline(pid)
        identity = _proc_identity(pid)
        if actual is None or identity is None:
            return False
        if not _cmdline_matches_argv(actual, [str(value) for value in argv]):
            return False
        task["executor_pid"] = pid
        task["executor_start_ticks"] = identity["start_ticks"]
        task["executor_session_id"] = identity["session_id"]
        task["executor_process_group_id"] = identity.get(
            "process_group_id", identity["session_id"]
        )
        task["process_identity_origin"] = "manifest_target"
        return True

    def _manifest_task_command_identity(
        self,
        task: Mapping[str, Any],
        manifest: Mapping[str, Any],
        manifest_path: Path | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        try:
            rendered = self._render_task(
                task,
                tuple(range(int(task.get("gpu_count", 1)))),
            )
            command = manifest.get("command")
            observed_argv = (
                command.get("argv") if isinstance(command, Mapping) else None
            )
            observed_env = (
                command.get("env") if isinstance(command, Mapping) else None
            )
            expected_env = dict(rendered.env)
            expected_cuda = expected_env.pop("CUDA_VISIBLE_DEVICES")
            if not isinstance(observed_env, Mapping):
                raise ValueError("manifest command.env is missing")
            observed_without_cuda = dict(observed_env)
            observed_cuda = observed_without_cuda.pop(
                "CUDA_VISIBLE_DEVICES", None
            )
            if not isinstance(observed_cuda, str):
                raise ValueError("manifest CUDA_VISIBLE_DEVICES is missing")
            gpu_ids = runner._cuda_indices(observed_cuda)
            expected_gpu_count = len(runner._cuda_indices(expected_cuda))
            expected_output = (
                self.repo_root / rendered.output_dir
                if not rendered.output_dir.is_absolute()
                else rendered.output_dir
            ).resolve(strict=False)
            valid = (
                observed_argv == rendered.argv
                and observed_without_cuda == expected_env
                and len(gpu_ids) == expected_gpu_count
                and len(set(gpu_ids)) == len(gpu_ids)
                and manifest.get("run_id") == rendered.run_id
                and (
                    manifest_path is None
                    or manifest_path.parent.resolve(strict=False)
                    == expected_output
                )
            )
            return valid, {
                "gpu_ids": list(gpu_ids),
                "expected_gpu_count": expected_gpu_count,
                "expected_run_id": rendered.run_id,
                "observed_run_id": manifest.get("run_id"),
                "expected_output_dir": str(expected_output),
                "observed_manifest": (
                    str(manifest_path) if manifest_path is not None else None
                ),
            }
        except (
            CampaignError,
            runner.PlanError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            return False, {"error": f"{type(exc).__name__}: {exc}"}

    def _task_progress_mtime_ns(self, task: Mapping[str, Any]) -> int | None:
        mtimes: list[int] = []
        for field in ("manifest", "log", "controller_log"):
            value = task.get(field)
            if not isinstance(value, str):
                continue
            with contextlib.suppress(OSError):
                mtimes.append(Path(value).stat().st_mtime_ns)
        return max(mtimes) if mtimes else None

    def _observe_active_progress(self, task: dict[str, Any]) -> None:
        observed = self._task_progress_mtime_ns(task)
        if observed is None:
            return
        previous = task.get("last_progress_mtime_ns")
        now = dt.datetime.now(dt.timezone.utc)
        if type(previous) is not int or observed > previous:
            task["last_progress_mtime_ns"] = observed
            task["last_progress_seen_utc"] = now.isoformat()
            if task["status"] == "STALLED":
                task["status"] = "RUNNING"
                self.emit(
                    "task",
                    "executor_progress_resumed",
                    details={"task_id": task["task_id"]},
                )
            return
        last_seen = _parse_utc(task.get("last_progress_seen_utc"))
        if last_seen is None:
            last_seen = dt.datetime.fromtimestamp(
                observed / 1_000_000_000,
                tz=dt.timezone.utc,
            )
            task["last_progress_seen_utc"] = last_seen.isoformat()
        stalled_seconds = max(0.0, (now - last_seen).total_seconds())
        if stalled_seconds >= STALE_PROGRESS_SECONDS:
            task["status"] = "STALLED"
            task["stalled_seconds"] = int(stalled_seconds)
            self.anomaly(
                "heartbeat_executor_stalled",
                details={
                    "task_id": task["task_id"],
                    "pid": task.get("executor_pid"),
                    "stalled_seconds": int(stalled_seconds),
                    "threshold_seconds": STALE_PROGRESS_SECONDS,
                },
                key=str(task["task_id"]),
            )

    def _record_dead_active_poll(self, task: dict[str, Any]) -> None:
        polls = int(task.get("dead_process_polls", 0)) + 1
        task["dead_process_polls"] = polls
        if polls < DEAD_PROCESS_GRACE_POLLS:
            return
        task["status"] = "ORPHANED"
        task["failure_class"] = "executor_disappeared"
        task["finished_utc"] = _utc_now()
        task["gpus"] = []
        self.anomaly(
            "heartbeat_orphaned_executor",
            details={
                "task_id": task["task_id"],
                "pid": task.get("executor_pid"),
                "reason": "executor disappeared while manifest was active",
                "grace_polls": polls,
            },
            key=str(task["task_id"]),
        )

    def _reconcile_one_task(self, task: dict[str, Any]) -> None:
        if task["status"] in TERMINAL_TASK_STATES:
            return
        manifest_path = self._task_manifest_path(task)
        if manifest_path is None or not manifest_path.is_file():
            if task["status"] in ACTIVE_TASK_STATES:
                if self._task_process_alive(task):
                    task["dead_process_polls"] = 0
                    self._observe_active_progress(task)
                    return
                self._record_dead_active_poll(task)
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return
        status = manifest.get("status")
        task["manifest"] = str(manifest_path)
        task["log"] = str(manifest_path.parent / executor.LOG_FILENAME)
        task["output_dir"] = str(manifest_path.parent)
        if status in {"launching", "running"}:
            if self._task_process_alive(task):
                task["dead_process_polls"] = 0
                if task["status"] != "STALLED":
                    task["status"] = "RUNNING"
                self._observe_active_progress(task)
            else:
                self._record_dead_active_poll(task)
            return
        if status == "succeeded" and manifest.get("exit_code") == 0:
            task["status"] = "SUCCEEDED"
            task["exit_code"] = 0
            task["finished_utc"] = manifest.get("timestamps", {}).get(
                "ended_utc"
            )
            self._validate_successful_task(task, manifest_path)
            return
        if status in {
            "failed",
            "terminated",
            "interrupted",
            "launch_failed",
            "wrapper_failed",
        }:
            task["exit_code"] = manifest.get("exit_code")
            task["finished_utc"] = manifest.get("timestamps", {}).get(
                "ended_utc"
            )
            log_text = ""
            log_path = manifest_path.parent / executor.LOG_FILENAME
            with contextlib.suppress(OSError, UnicodeDecodeError):
                log_text = log_path.read_text(encoding="utf-8")
            failure_class = classify_failure(log_text)
            task["failure_class"] = failure_class
            task["status"] = "OOM" if failure_class == "oom" else "FAILED"
            if failure_class == "oom":
                self.anomaly(
                    "oom",
                    details={
                        "task_id": task["task_id"],
                        "model": task["spec"].get("model"),
                        "setting": task["spec"].get("setting"),
                        "profile_level": task["spec"].get("profile_level"),
                        "attempt_index": task.get("attempt_index"),
                    },
                    key=task["task_id"],
                )
            elif failure_class == "cache_miss":
                self.anomaly(
                    "cache_miss",
                    details={"task_id": task["task_id"]},
                    key=task["task_id"],
                )

    def _validate_successful_task(
        self,
        task: dict[str, Any],
        manifest_path: Path,
    ) -> None:
        spec = task["spec"]
        log_path = manifest_path.parent / executor.LOG_FILENAME
        log_text = ""
        with contextlib.suppress(OSError, UnicodeDecodeError):
            log_text = log_path.read_text(encoding="utf-8")
        try:
            manifest_payload = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            task["status"] = "INVALID_RESULT"
            task["failure_class"] = "invalid_manifest"
            self.anomaly(
                "invalid_result",
                details={
                    "task_id": task["task_id"],
                    "error": f"cannot parse successful manifest: {exc}",
                },
                key=task["task_id"],
            )
            return
        plan_matches, plan_details = self._manifest_plan_matches(
            manifest_payload,
            spec,
            require_finished=True,
        )
        if not plan_matches:
            task["status"] = "INVALID_RESULT"
            task["failure_class"] = "plan_identity_mismatch"
            self.anomaly(
                "config_drift",
                details={
                    "task_id": task["task_id"],
                    "reason": "successful manifest plan/protocol mismatch",
                    **plan_details,
                },
                key=f"task-plan:{task['task_id']}",
            )
            return
        command_matches, command_details = (
            self._manifest_task_command_identity(
                task, manifest_payload, manifest_path
            )
        )
        if not command_matches:
            task["status"] = "INVALID_RESULT"
            task["failure_class"] = "task_command_identity_mismatch"
            self.anomaly(
                "config_drift",
                details={
                    "task_id": task["task_id"],
                    "manifest": str(manifest_path),
                    "reason": (
                        "successful manifest command/run/output does not "
                        "exactly match the scheduled task"
                    ),
                    "identity_details": command_details,
                },
                key=f"task-command:{task['task_id']}",
            )
            return
        expected_source = self.state.get("source_identity")
        if isinstance(expected_source, Mapping):
            manifest_expected_source = {
                key: expected_source.get(key)
                for key in (
                    "numerical_source_sha256",
                    "runner_sha256",
                    "executor_sha256",
                )
            }
            try:
                sources = manifest_payload["source_files"]
                numerical = manifest_payload["numerical_source_tree"]
                observed_source = {
                    "numerical_source_sha256": numerical["combined_sha256"],
                    "runner_sha256": sources["runner"]["sha256"],
                    "executor_sha256": sources["executor"]["sha256"],
                }
                numerical_files = numerical.get("files")
                source_stable = (
                    numerical.get("changed_during_execution") is False
                    and numerical.get("combined_sha256_at_end")
                    == numerical.get("combined_sha256")
                    and isinstance(numerical_files, list)
                    and bool(numerical_files)
                    and all(
                        isinstance(item, Mapping)
                        and item.get("stable_during_hash") is True
                        for item in numerical_files
                    )
                    and sources["runner"].get("stable_during_hash") is True
                    and sources["executor"].get("stable_during_hash") is True
                )
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
            ) as exc:
                observed_source = {"error": f"{type(exc).__name__}: {exc}"}
                source_stable = False
            if observed_source != manifest_expected_source or not source_stable:
                task["status"] = "INVALID_RESULT"
                task["failure_class"] = "source_identity_mismatch"
                self.anomaly(
                    "config_drift",
                    details={
                        "task_id": task["task_id"],
                        "expected_source_identity": expected_source,
                        "observed_source_identity": observed_source,
                        "source_stable_during_execution": source_stable,
                    },
                    key=f"task-source:{task['task_id']}",
                )
                return
        if spec["kind"] == "realq_static":
            world = (
                int(self.plan["tuning"]["world_size"])
                if spec["target_phase"] == "tune"
                else int(self.plan["final"]["world_size"])
            )
            paths = extract_static_cache_paths(
                log_text, repo_root=self.repo_root
            )
            valid, validation = validate_static_cache_paths(
                paths,
                expected_world_size=world,
            )
            task["cache_validation"] = validation
            entry = self.state["static_caches"][
                f"{spec['target_phase']}/{spec['model']}"
            ]
            if valid:
                entry["status"] = "READY"
                entry["validation"] = validation
                self.emit(
                    "cache",
                    "static_cache_validated",
                    details={
                        "model": spec["model"],
                        "phase": spec["target_phase"],
                        "validation": validation,
                    },
                )
            else:
                task["status"] = "INVALID_RESULT"
                entry["status"] = "NEEDS_USER_ACTION"
                entry["validation"] = validation
                self.anomaly(
                    "cache_validation_failed",
                    details={
                        "task_id": task["task_id"],
                        "validation": validation,
                    },
                    key=task["task_id"],
                )
            return
        if spec["kind"] == "guided_saliency":
            model_state = self.state.get("formal", {}).get("models", {}).get(
                spec["model"]
            )
            validation_command = [
                sys.executable,
                str(self.repo_root / "tools" / "validate_guided_saliency.py"),
                "--model",
                str(self.repo_root / self.plan["models"][spec["model"]]),
                "--cache-root",
                str(self.repo_root / self.plan["legacy_cache_root"]),
                "--dataset",
                str(self.plan["fixed_numerics"]["dataset"]),
                "--nsamples",
                str(self.plan["final"]["nsamples"]),
                "--seq-len",
                str(self.plan["final"]["seq_len"]),
                "--seed",
                str(self.plan["fixed_numerics"]["seed"]),
                "--rotation-seed",
                str(self.plan["fixed_numerics"]["rotation_seed"]),
                "--num-groups",
                str(self.plan["fixed_numerics"]["num_groups"]),
                "--sha256",
            ]
            try:
                completed = subprocess.run(
                    validation_command,
                    cwd=self.repo_root,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=1800,
                    check=False,
                )
                validation = json.loads(completed.stdout)
            except (
                OSError,
                subprocess.SubprocessError,
                json.JSONDecodeError,
            ) as exc:
                validation = {
                    "valid": False,
                    "errors": [f"{type(exc).__name__}: {exc}"],
                }
                completed = None
            task["cache_validation"] = validation
            if (
                completed is not None
                and completed.returncode == 0
                and isinstance(validation, Mapping)
                and validation.get("valid") is True
            ):
                if isinstance(model_state, dict):
                    model_state["guided_status"] = "READY"
                self.emit(
                    "cache",
                    "guided_saliency_validated",
                    details={
                        "model": spec["model"],
                        "validated_files": validation.get("validated_files"),
                        "saliency_dir": validation.get("saliency_dir"),
                    },
                )
            else:
                task["status"] = "INVALID_RESULT"
                if isinstance(model_state, dict):
                    model_state["guided_status"] = "NEEDS_USER_ACTION"
                self.anomaly(
                    "cache_validation_failed",
                    details={
                        "task_id": task["task_id"],
                        "cache": "guided_saliency",
                        "validation": validation,
                    },
                    key=task["task_id"],
                )
            return
        try:
            parsed = results.parse_result(manifest_path)
        except results.ResultError as exc:
            task["status"] = "INVALID_RESULT"
            task["failure_class"] = "invalid_result"
            self.anomaly(
                "invalid_result",
                details={"task_id": task["task_id"], "error": str(exc)},
                key=task["task_id"],
            )
            return
        expected_method = (
            "realq" if spec["kind"] == "realq" else spec.get("method")
        )
        if (
            parsed.phase != spec.get("phase")
            or parsed.method != expected_method
            or parsed.model != spec.get("model")
            or parsed.setting != spec.get("setting")
            or parsed.attempt_index != int(task.get("attempt_index", 1))
            or (
                spec.get("grad_lr") is None
                and parsed.grad_lr is not None
            )
            or (
                spec.get("grad_lr") is not None
                and (
                    parsed.grad_lr is None
                    or float(parsed.grad_lr)
                    != float(spec["grad_lr"])
                )
            )
        ):
            task["status"] = "INVALID_RESULT"
            task["failure_class"] = "parsed_task_identity_mismatch"
            self.anomaly(
                "config_drift",
                details={
                    "task_id": task["task_id"],
                    "scheduled": {
                        "phase": spec.get("phase"),
                        "method": expected_method,
                        "model": spec.get("model"),
                        "setting": spec.get("setting"),
                        "grad_lr": spec.get("grad_lr"),
                        "attempt_index": task.get("attempt_index", 1),
                    },
                    "parsed": {
                        "phase": parsed.phase,
                        "method": parsed.method,
                        "model": parsed.model,
                        "setting": parsed.setting,
                        "grad_lr": parsed.grad_lr,
                        "attempt_index": parsed.attempt_index,
                    },
                },
                key=f"parsed-task:{task['task_id']}",
            )
            return
        expected_plan_sha, expected_protocol = self._expected_plan_identity(spec)
        if (
            parsed.identities.get("plan_sha256") != expected_plan_sha
            or _protocol_sha256(parsed.plan_content) != expected_protocol
        ):
            task["status"] = "INVALID_RESULT"
            task["failure_class"] = "parsed_plan_identity_mismatch"
            self.anomaly(
                "config_drift",
                details={
                    "task_id": task["task_id"],
                    "expected_plan_sha256": expected_plan_sha,
                    "parsed_plan_sha256": parsed.identities.get(
                        "plan_sha256"
                    ),
                    "expected_protocol_sha256": expected_protocol,
                    "parsed_protocol_sha256": _protocol_sha256(
                        parsed.plan_content
                    ),
                },
                key=f"parsed-plan:{task['task_id']}",
            )
            return
        task["metric"] = {
            "kl": parsed.kl_wikitext2,
            "ppl": parsed.ppl_wikitext2,
            "acc_avg": parsed.acc_avg,
            "tasks": parsed.tasks,
            "grad_lr": parsed.grad_lr,
            "final_layer_grad_lr": parsed.final_layer_grad_lr,
            "params": parsed.params,
            "identities": parsed.identities,
        }
        if (
            parsed.kl_wikitext2 is not None
            and not math.isfinite(parsed.kl_wikitext2)
        ):
            self.anomaly(
                "nonfinite_exact_kl",
                details={
                    "task_id": task["task_id"],
                    "kl": parsed.kl_wikitext2,
                },
                key=task["task_id"],
            )
            task["status"] = "INVALID_RESULT"
        elif parsed.kl_wikitext2 is not None and parsed.kl_wikitext2 < 0:
            self.anomaly(
                "negative_exact_kl",
                details={
                    "task_id": task["task_id"],
                    "kl": parsed.kl_wikitext2,
                },
                key=task["task_id"],
            )
            task["status"] = "INVALID_RESULT"
        if (
            spec["kind"] == "realq"
            and CACHE_MISS_PATTERNS[0].search(log_text)
        ):
            self.anomaly(
                "cache_miss",
                details={
                    "task_id": task["task_id"],
                    "message": "consumer log reported a cache miss",
                },
                key=task["task_id"],
            )
            task["status"] = "INVALID_RESULT"

    def _reconcile_tasks(self) -> None:
        for task in self.state["tasks"].values():
            self._reconcile_one_task(task)
        for task_id, process in list(self.processes.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            handle = self.controller_logs.pop(task_id, None)
            if handle is not None:
                handle.close()
            self.processes.pop(task_id, None)
            task = self.state["tasks"].get(task_id)
            if task and task["status"] in ACTIVE_TASK_STATES:
                self._reconcile_one_task(task)

    def _candidate_launch_allowed(self, task: Mapping[str, Any]) -> bool:
        spec = task["spec"]
        if spec.get("kind") != "realq" or spec.get("phase") != "tune":
            return True
        group = self.state["tuning"][
            f"{spec['model']}/{spec['setting']}"
        ]
        max_attempts = int(self.plan["tuning"]["max_attempts_per_model_setting"])
        if int(group["attempt_count"]) >= max_attempts:
            group["status"] = "NEEDS_USER_ACTION"
            group["needs_user_action_reason"] = (
                "20 candidate attempts have already been launched"
            )
            return False
        return True

    def _task_dependencies_ready(self, task: Mapping[str, Any]) -> bool:
        spec = task["spec"]
        if spec["kind"] == "realq" and spec["phase"] in {"tune", "final"}:
            return self._static_ready(spec["model"], spec["phase"])
        if spec["kind"] == "baseline" and spec["method"] == "guided_gptq":
            model_state = self.state["formal"]["models"][spec["model"]]
            return model_state["guided_status"] == "READY"
        return True

    def _reserved_gpus(self) -> set[int]:
        reserved: set[int] = set()
        for task in self.state["tasks"].values():
            if task["status"] in ACTIVE_TASK_STATES:
                reserved.update(int(value) for value in task.get("gpus", []))
        return reserved

    def _gpu_process_owner(
        self,
        pid: Any,
        *,
        local_pid: int | None = None,
    ) -> dict[str, Any] | None:
        if type(pid) is not int or pid <= 0:
            return None
        resolved_pid = int(local_pid) if local_pid is not None else pid
        observed = _proc_identity(resolved_pid)
        if observed is None:
            return None
        for task in self.state["tasks"].values():
            if task.get("status") not in ACTIVE_TASK_STATES:
                continue
            if not self._task_process_alive(task):
                continue
            root_pid = task.get("executor_pid")
            if (
                type(root_pid) is int
                and _pid_is_descendant(resolved_pid, root_pid)
            ):
                # The executor intentionally starts the rendered experiment
                # in a second session.  Its GPU worker therefore has a
                # different SID/PGID from the controller-launched wrapper,
                # while the verified parent/descendant relation remains
                # stable and unambiguous.
                return task
            if task.get("process_identity_origin") == "manifest_target":
                continue
            session = task.get("executor_session_id")
            if (
                type(session) is int
                and observed["session_id"] == session
            ):
                return task
        return None

    def _gpu_process_is_owned(self, pid: Any) -> bool:
        return self._gpu_process_owner(pid) is not None

    def _ready_tasks(self) -> list[dict[str, Any]]:
        tasks = [
            task
            for task in self.state["tasks"].values()
            if task["status"] == "PENDING"
            and self._task_dependencies_ready(task)
            and self._candidate_launch_allowed(task)
        ]
        # A ready four-GPU REAL-Q job has priority over one-GPU backfill, so
        # small work cannot continually fragment both contiguous groups.
        return sorted(
            tasks,
            key=lambda task: (
                -int(task["priority"]),
                -int(task["gpu_count"]),
                task["created_utc"],
                task["task_id"],
            ),
        )

    def _launch_task(
        self,
        task: dict[str, Any],
        gpu_ids: Sequence[int],
    ) -> None:
        master_port = self._assign_master_port(task)
        rendered = self._render_task(task, gpu_ids)
        task["output_dir"] = str(
            (
                self.repo_root / rendered.output_dir
                if not rendered.output_dir.is_absolute()
                else rendered.output_dir
            ).resolve(strict=False)
        )
        task["manifest"] = str(
            Path(task["output_dir"]) / executor.MANIFEST_FILENAME
        )
        task["log"] = str(Path(task["output_dir"]) / executor.LOG_FILENAME)
        task["gpus"] = list(gpu_ids)
        task["launched_utc"] = _utc_now()
        task["status"] = "LAUNCHING"
        task["executor_cli"] = self._executor_cli(task, gpu_ids)
        if (
            task["spec"].get("kind") == "realq"
            and task["spec"].get("phase") == "tune"
        ):
            group = self.state["tuning"][
                f"{task['spec']['model']}/{task['spec']['setting']}"
            ]
            group["attempt_count"] = int(group["attempt_count"]) + 1
            task["group_attempt_number"] = int(group["attempt_count"])
        launch_event = self.emit(
            "task",
            "task_launching",
            details={
                "task_id": task["task_id"],
                "plan_sha256": self._expected_plan_identity(task["spec"])[0],
                "kind": task["spec"].get("kind"),
                "method": task["spec"].get("method"),
                "phase": task["spec"].get("phase"),
                "model": task["spec"].get("model"),
                "setting": task["spec"].get("setting"),
                "grad_lr": task["spec"].get("grad_lr"),
                "profile_level": task["spec"].get("profile_level"),
                "generation": task["spec"].get("generation"),
                "effective_overrides": task["spec"].get("overrides", {}),
                "rendezvous": {
                    "master_addr": "127.0.0.1",
                    "master_port": master_port,
                },
                "gpus": list(gpu_ids),
                "attempt_index": task["attempt_index"],
                "output_dir": task["output_dir"],
            },
        )
        if launch_event is not None:
            task["launch_event_seq"] = launch_event["seq"]
        self.persist()

        controller_dir = self.state_path.parent / "controller_logs"
        controller_dir.mkdir(parents=True, exist_ok=True)
        controller_path = controller_dir / f"{_task_slug(task['task_id'])}.log"
        log_handle = controller_path.open("ab", buffering=0)
        launch_environment = os.environ.copy()
        launch_environment.update(
            {
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(master_port),
            }
        )
        try:
            process = subprocess.Popen(
                task["executor_cli"],
                cwd=self.repo_root,
                env=launch_environment,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except (OSError, ValueError) as exc:
            log_handle.close()
            task["status"] = "LAUNCH_FAILED"
            task["failure_class"] = "controller_launch_failed"
            task["finished_utc"] = _utc_now()
            self.anomaly(
                "executor_launch_failed",
                details={
                    "task_id": task["task_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                },
                key=task["task_id"],
            )
            return
        task["executor_pid"] = process.pid
        identity = _proc_identity(process.pid)
        if identity is not None:
            task["executor_start_ticks"] = identity["start_ticks"]
            task["executor_session_id"] = identity["session_id"]
            task["executor_process_group_id"] = identity.get(
                "process_group_id", identity["session_id"]
            )
            task["process_identity_origin"] = "controller_session"
        task["controller_log"] = str(controller_path)
        task["status"] = "RUNNING"
        self.processes[task["task_id"]] = process
        self.controller_logs[task["task_id"]] = log_handle

    @staticmethod
    def _master_port_available(port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
                handle.bind(("127.0.0.1", int(port)))
            return True
        except OSError:
            return False

    def _assign_master_port(self, task: dict[str, Any]) -> int:
        occupied = {
            int(rendezvous["master_port"])
            for other in self.state["tasks"].values()
            if other is not task
            and other.get("status") in ACTIVE_TASK_STATES
            and isinstance(
                rendezvous := other.get("rendezvous"), Mapping
            )
            and type(rendezvous.get("master_port")) is int
        }
        existing = task.get("rendezvous")
        if isinstance(existing, Mapping):
            saved = existing.get("master_port")
            if (
                type(saved) is int
                and 15000 <= saved <= 59999
                and saved not in occupied
                and self._master_port_available(saved)
            ):
                return saved
        seed = int(
            _sha256_bytes(
                (
                    f"{self.repo_root}:{task['task_id']}:"
                    f"{task.get('attempt_index', 1)}"
                ).encode("utf-8")
            )[:12],
            16,
        )
        span = 45_000
        start = 15_000 + seed % span
        for offset in range(span):
            candidate = 15_000 + ((start - 15_000 + offset) % span)
            if (
                candidate not in occupied
                and self._master_port_available(candidate)
            ):
                task["rendezvous"] = {
                    "master_addr": "127.0.0.1",
                    "master_port": candidate,
                    "selection": "stable_task_hash_with_availability_probe",
                }
                return candidate
        raise CampaignBlocked("no available per-task MASTER_PORT was found")

    def _schedule(self, snapshot: GPUSnapshot) -> int:
        requirements = self.plan["runtime_requirements"]
        eligible = eligible_gpu_ids(
            snapshot,
            minimum_memory_gib=float(
                requirements["minimum_gpu_memory_gib"]
            ),
        )
        reserved = self._reserved_gpus()
        uuid_to_index = {
            str(device["uuid"]): int(device["index"])
            for device in snapshot.devices
        }
        unknown_uuid_processes = [
            process
            for process in snapshot.compute_processes
            if process.get("gpu_uuid") not in uuid_to_index
        ]
        observed_pids = [
            int(process["pid"])
            for process in snapshot.compute_processes
            if type(process.get("pid")) is int
        ]
        resolved_gpu_pids, ambiguous_gpu_pids = (
            _resolve_gpu_process_pids(observed_pids)
        )
        foreign_processes: list[dict[str, Any]] = []
        ownership_conflicts: list[dict[str, Any]] = []
        for process in snapshot.compute_processes:
            gpu_uuid = process.get("gpu_uuid")
            if gpu_uuid not in uuid_to_index:
                continue
            observed_pid = process.get("pid")
            owner = (
                None
                if observed_pid in ambiguous_gpu_pids
                else self._gpu_process_owner(
                    observed_pid,
                    local_pid=resolved_gpu_pids.get(observed_pid),
                )
            )
            if owner is None:
                foreign_processes.append(process)
                continue
            observed_gpu = uuid_to_index[str(gpu_uuid)]
            if observed_gpu not in {
                int(value) for value in owner.get("gpus", [])
            }:
                ownership_conflicts.append(
                    {
                        "process": process,
                        "owner_task_id": owner["task_id"],
                        "reserved_gpus": list(owner.get("gpus", [])),
                        "observed_gpu": observed_gpu,
                    }
                )
        foreign = {
            uuid_to_index[str(process["gpu_uuid"])]
            for process in foreign_processes
        }
        if foreign or unknown_uuid_processes or ownership_conflicts:
            self.anomaly(
                "gpu_foreign_process",
                details={
                    "gpu_ids": sorted(foreign),
                    "processes": foreign_processes,
                    "unknown_uuid_processes": unknown_uuid_processes,
                    "ownership_conflicts": ownership_conflicts,
                    "resolved_gpu_pids": resolved_gpu_pids,
                    "ambiguous_gpu_pids": ambiguous_gpu_pids,
                },
                key=_sha256_bytes(
                    _canonical_json_bytes(
                        {
                            "gpu_ids": sorted(foreign),
                            "processes": foreign_processes,
                            "unknown_uuid_processes": unknown_uuid_processes,
                            "ownership_conflicts": ownership_conflicts,
                            "resolved_gpu_pids": resolved_gpu_pids,
                            "ambiguous_gpu_pids": ambiguous_gpu_pids,
                        }
                    )
                ),
            )
            self.state["status"] = "WAIT_GPUS"
            self.state["blockers"] = [
                "unexpected GPU compute processes are present on "
                f"GPU(s) {sorted(foreign)}; no new task was launched"
            ]
            return 0
        free = sorted(set(eligible) - reserved - foreign)
        launched = 0
        for task in self._ready_tasks():
            count = int(task["gpu_count"])
            allocation = assign_gpus(
                free,
                count=count,
                contiguous=(count > 1),
            )
            if allocation is None:
                continue
            self._launch_task(task, allocation)
            free = [value for value in free if value not in allocation]
            launched += 1
        return launched

    def _update_static_entries_from_tasks(self) -> None:
        for key, entry in self.state["static_caches"].items():
            task_id = entry.get("producer_task")
            task = self.state["tasks"].get(task_id) if task_id else None
            if not task:
                continue
            if task["status"] in ACTIVE_TASK_STATES:
                entry["status"] = "RUNNING"
            elif task["status"] == "OOM":
                profiles = self._memory_profiles(
                    phase=entry["phase"], static_only=True
                )
                next_level = int(entry["profile_level"]) + 1
                if next_level < len(profiles):
                    entry["profile_level"] = next_level
                    entry["next_attempt_index"] = (
                        int(task.get("attempt_index", 1)) + 1
                    )
                    entry["status"] = "PENDING"
                    entry["producer_task"] = None
                    self.emit(
                        "state",
                        "static_profile_downgraded",
                        details={
                            "cache": key,
                            "profile_level": next_level,
                        },
                    )
                else:
                    entry["status"] = "NEEDS_USER_ACTION"
                    entry["reason"] = "static OOM ladder exhausted"
            elif task["status"] in {
                "FAILED",
                "INVALID_RESULT",
                "ORPHANED",
                "LAUNCH_FAILED",
            }:
                if (
                    task.get("failure_class") == "master_port_collision"
                    and not task.get("retry_scheduled")
                    and int(task.get("attempt_index", 1))
                    < executor.MAX_ATTEMPT_INDEX
                ):
                    retry_spec = dict(task["spec"])
                    retry_spec["attempt_index"] = (
                        int(task.get("attempt_index", 1)) + 1
                    )
                    retry_spec["purpose"] = (
                        f"{task['spec'].get('purpose', 'static')}"
                        "_rendezvous_retry"
                    )
                    retry_id = self._create_task(
                        retry_spec,
                        gpu_count=int(task["gpu_count"]),
                        priority=int(task["priority"]),
                    )
                    task["retry_scheduled"] = True
                    task["retry_task"] = retry_id
                    entry["producer_task"] = retry_id
                    entry["status"] = "QUEUED"
                    self.emit(
                        "state",
                        "static_rendezvous_retry_scheduled",
                        details={
                            "cache": key,
                            "failed_task": task["task_id"],
                            "retry_task": retry_id,
                            "attempt_index": retry_spec["attempt_index"],
                        },
                    )
                    continue
                entry["status"] = "NEEDS_USER_ACTION"
                entry["reason"] = f"producer ended as {task['status']}"

    def _campaign_status(self) -> str:
        group_statuses = {
            group["status"] for group in self.state["tuning"].values()
        }
        if "NEEDS_USER_ACTION" in group_statuses:
            return "NEEDS_USER_ACTION"
        if all(status == "SELECTED" for status in group_statuses):
            patch = selected_lr_patch(self.state)
            self.state["selected_lr_patch"] = patch
            if not self._formal_plan_matches_selection():
                return "WAIT_LR_COMMIT"
            if self.state["preflight"] and self.state["preflight"]["valid"]:
                return self._formal_campaign_status()
            return "WAIT_PREFLIGHT"
        return "TUNING"

    def _refresh_eta(self) -> dict[str, Any]:
        now = _utc_now()
        payload = eta.estimate_campaign_eta(
            self.state,
            self.plan,
            now_utc=now,
        )
        self.state["eta"] = payload
        signature_payload = {
            "status": payload["campaign_status"],
            "blocked": payload["blocked"],
            "counts": payload["counts"],
            "tuning_candidates": payload[
                "tuning_candidate_forecast"
            ]["future_candidates"],
            "stage_work": {
                stage: {
                    bound: int(
                        value["eta"][bound]["work_items"]
                    )
                    for bound in eta.BOUNDS
                }
                for stage, value in payload["stages"].items()
            },
            # Fifteen-minute buckets avoid a noisy event stream while every
            # exact ETA remains persisted and attached to its heartbeat.
            "whole_eta_buckets": {
                bound: int(
                    payload["whole"]["eta"][bound]["remaining_seconds"]
                    // (15 * 60)
                )
                for bound in eta.BOUNDS
            },
        }
        signature = _sha256_bytes(_canonical_json_bytes(signature_payload))
        tracker = self.state.get("eta_event_tracker")
        previous_time = (
            _parse_utc(tracker.get("timestamp_utc"))
            if isinstance(tracker, Mapping)
            else None
        )
        current_time = _parse_utc(now)
        due = (
            not isinstance(tracker, Mapping)
            or tracker.get("signature") != signature
            or previous_time is None
            or current_time is None
            or (current_time - previous_time).total_seconds() >= 5 * 60
        )
        if due:
            self.emit(
                "eta",
                "eta_updated",
                details={
                    "eta": payload,
                    "signature": signature,
                    "throttle_seconds": 5 * 60,
                },
            )
            self.state["eta_event_tracker"] = {
                "timestamp_utc": now,
                "signature": signature,
            }
        return payload

    def tick(self) -> dict[str, Any]:
        self.state["execute_requested"] = bool(self.execute)
        # Child reconciliation must continue even while a launch gate is
        # blocked. Otherwise a transient GPU/preflight/config blocker could
        # make the controller exit while detached executors are still running.
        self._reconcile_tasks()
        self._update_static_entries_from_tasks()
        gates_ok, snapshot = self._gates()
        if not gates_ok:
            eta_payload = self._refresh_eta()
            self.emit(
                "heartbeat",
                "scheduler_blocked_tick",
                details={
                    "status": self.state["status"],
                    "blockers": self.state["blockers"],
                    "running_tasks": [
                        task["task_id"]
                        for task in self.state["tasks"].values()
                        if task["status"] in ACTIVE_TASK_STATES
                    ],
                    "task_counts": self._task_counts(),
                    "eta": eta_payload,
                },
            )
            self.persist()
            return self.public_status()

        campaign_status = self._campaign_status()
        if campaign_status == "TUNING":
            self._ensure_tune_static_tasks()
            self._advance_tuning_groups()
            campaign_status = self._campaign_status()
        elif campaign_status == "FORMAL":
            self._ensure_formal_tasks()
            campaign_status = self._formal_campaign_status()

        self.state["status"] = campaign_status
        if self.execute and snapshot is not None and campaign_status in {
            "TUNING",
            "FORMAL",
        }:
            self._schedule(snapshot)
        eta_payload = self._refresh_eta()
        self.emit(
            "heartbeat",
            "scheduler_tick",
            details={
                "status": self.state["status"],
                "running_tasks": [
                    task["task_id"]
                    for task in self.state["tasks"].values()
                    if task["status"] in ACTIVE_TASK_STATES
                ],
                "task_counts": self._task_counts(),
                "gpu": snapshot.public_dict() if snapshot else None,
                "eta": eta_payload,
            },
        )
        self.persist()
        return self.public_status()

    def _task_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for task in self.state["tasks"].values():
            counts[task["status"]] = counts.get(task["status"], 0) + 1
        return counts

    def public_status(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "campaign_id": self.state["campaign_id"],
            "status": self.state["status"],
            "execute_requested": self.state["execute_requested"],
            "plan": self.state["plan"],
            "agent": self.state["agent"],
            "preflight": self.state["preflight"],
            "source_identity": self.state.get("source_identity"),
            "retirement_ledger": self.state.get("retirement_ledger"),
            "heartbeat": self.state["heartbeat"],
            "eta": self.state.get("eta"),
            "blockers": self.state["blockers"],
            "task_counts": self._task_counts(),
            "running_tasks": [
                {
                    "task_id": task["task_id"],
                    "spec": task["spec"],
                    "gpus": task["gpus"],
                    "executor_pid": task["executor_pid"],
                    "manifest": task["manifest"],
                }
                for task in self.state["tasks"].values()
                if task["status"] in ACTIVE_TASK_STATES
            ],
            "tuning": {
                key: {
                    field: group.get(field)
                    for field in (
                        "status",
                        "profile_level",
                        "generation",
                        "attempt_count",
                        "selected_lr",
                        "selected_parameters",
                        "needs_user_action_reason",
                        "last_search_decision",
                    )
                }
                for key, group in self.state["tuning"].items()
            },
            "selected_lr_patch": selected_lr_patch(self.state),
            "preview_events": self.preview_events,
            "state_path": str(self.state_path),
            "events_path": str(self.events_path),
        }

    def run(
        self,
        *,
        once: bool,
        stream_heartbeats: bool = False,
    ) -> dict[str, Any]:
        while True:
            payload = self.tick()
            if stream_heartbeats:
                print(
                    json.dumps(
                        {
                            "record_type": "campaign_heartbeat",
                            **payload,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
            status = payload["status"]
            if once or not self.execute:
                return payload
            if status in {
                "WAIT_ENV_READY",
                "WAIT_VENV_ACTIVATION",
                "WAIT_PREFLIGHT",
                "WAIT_GPUS",
                "WAIT_LR_COMMIT",
                "NEEDS_USER_ACTION",
                "COMPLETE",
            }:
                active_children = any(
                    task["status"] in ACTIVE_TASK_STATES
                    for task in self.state["tasks"].values()
                )
                if not active_children:
                    return payload
            time.sleep(self.poll_seconds)

    def _active_ownership_evidence(
        self,
        snapshot: GPUSnapshot,
    ) -> dict[str, Any]:
        if snapshot.error:
            raise CampaignBlocked(
                f"cannot verify active GPU ownership: {snapshot.error}"
            )
        active = [
            task
            for task in self.state["tasks"].values()
            if task.get("status") in ACTIVE_TASK_STATES
        ]
        task_evidence: list[dict[str, Any]] = []
        for task in active:
            if not self._task_process_alive(task):
                raise CampaignBlocked(
                    "active executor root PID/start identity is not live: "
                    f"{task.get('task_id')}"
                )
            manifest_path = self._task_manifest_path(task)
            if manifest_path is None or not manifest_path.is_file():
                raise CampaignBlocked(
                    f"active task lacks a manifest: {task.get('task_id')}"
                )
            try:
                manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as exc:
                raise CampaignBlocked(
                    f"cannot verify active manifest {manifest_path}: {exc}"
                ) from exc
            if manifest.get("status") not in {"launching", "running"}:
                raise CampaignBlocked(
                    "active state disagrees with manifest status for "
                    f"{task.get('task_id')}: {manifest.get('status')!r}"
                )
            manifest_process = manifest.get("process")
            target_pid = (
                manifest_process.get("pid")
                if isinstance(manifest_process, Mapping)
                else None
            )
            command = manifest.get("command")
            manifest_argv = (
                command.get("argv")
                if isinstance(command, Mapping)
                else None
            )
            root_pid = task.get("executor_pid")
            if (
                type(target_pid) is not int
                or target_pid <= 0
                or type(root_pid) is not int
                or _proc_identity(target_pid) is None
                or not isinstance(manifest_argv, list)
                or not _cmdline_matches_argv(
                    _proc_cmdline(target_pid), manifest_argv
                )
                or (
                    target_pid != root_pid
                    and not _pid_is_descendant(target_pid, root_pid)
                )
            ):
                raise CampaignBlocked(
                    "active manifest target PID/cmdline is not a live "
                    "descendant of its verified executor wrapper: "
                    f"task={task.get('task_id')}, root={root_pid}, "
                    f"target={target_pid}"
                )
            plan_matches, plan_details = self._manifest_plan_matches(
                manifest,
                task["spec"],
                require_finished=False,
            )
            command_matches, command_details = (
                self._manifest_task_command_identity(
                    task, manifest, manifest_path
                )
            )
            recorded_gpus = [int(value) for value in task.get("gpus", [])]
            if (
                not plan_matches
                or not command_matches
                or command_details.get("gpu_ids") != recorded_gpus
            ):
                raise CampaignBlocked(
                    "active task provenance/GPU identity mismatch for "
                    f"{task.get('task_id')}: "
                    f"plan={plan_details}, command={command_details}, "
                    f"state_gpus={recorded_gpus}"
                )
            task_evidence.append(
                {
                    "task_id": task["task_id"],
                    "executor_pid": task.get("executor_pid"),
                    "executor_start_ticks": task.get(
                        "executor_start_ticks"
                    ),
                    "executor_session_id": task.get("executor_session_id"),
                    "manifest_target_pid": target_pid,
                    "gpus": recorded_gpus,
                    "manifest": str(manifest_path),
                    "manifest_sha256": _sha256_file(manifest_path),
                }
            )

        uuid_to_index = {
            str(device["uuid"]): int(device["index"])
            for device in snapshot.devices
        }
        observed_pids = [
            int(process["pid"])
            for process in snapshot.compute_processes
            if type(process.get("pid")) is int
        ]
        resolved, ambiguous = _resolve_gpu_process_pids(observed_pids)
        process_evidence: list[dict[str, Any]] = []
        for process in snapshot.compute_processes:
            observed_pid = process.get("pid")
            gpu_uuid = process.get("gpu_uuid")
            if (
                gpu_uuid not in uuid_to_index
                or observed_pid in ambiguous
            ):
                raise CampaignBlocked(
                    "GPU process has unknown/ambiguous identity during "
                    f"controller migration: {process}"
                )
            owner = self._gpu_process_owner(
                observed_pid,
                local_pid=resolved.get(observed_pid),
            )
            observed_gpu = uuid_to_index[str(gpu_uuid)]
            if owner is None or observed_gpu not in {
                int(value) for value in owner.get("gpus", [])
            }:
                raise CampaignBlocked(
                    "foreign or wrong-GPU process is present during "
                    f"controller migration: process={process}, "
                    f"owner={owner and owner.get('task_id')}"
                )
            process_evidence.append(
                {
                    "observed_pid": observed_pid,
                    "local_pid": resolved.get(observed_pid, observed_pid),
                    "gpu": observed_gpu,
                    "owner_task_id": owner["task_id"],
                }
            )
        return {
            "active_tasks": task_evidence,
            "gpu_processes": process_evidence,
            "resolved_gpu_pids": resolved,
            "ambiguous_gpu_pids": ambiguous,
        }

    def migrate_controller_source(
        self,
        *,
        expected_old_campaign_sha256: str,
        accept_campaign_sha256: str,
    ) -> dict[str, Any]:
        """Explicitly accept a scheduler-only hotfix without numeric drift."""
        for label, value in (
            (
                "expected_old_campaign_sha256",
                expected_old_campaign_sha256,
            ),
            ("accept_campaign_sha256", accept_campaign_sha256),
        ):
            if (
                not isinstance(value, str)
                or not re.fullmatch(r"[0-9a-f]{64}", value)
            ):
                raise CampaignBlocked(
                    f"{label} must be an exact lowercase SHA256"
                )
        pending = self.state.get("pending_controller_source_transition")
        if pending is not None:
            transition_id = (
                pending.get("transition_id")
                if isinstance(pending, Mapping)
                else None
            )
            raise CampaignBlocked(
                "an incomplete controller-source migration is already "
                "recorded; inspect and recover it before retrying"
                + (
                    f": transition_id={transition_id}"
                    if transition_id
                    else ""
                )
            )
        current = self._current_source_identity()
        previous = self.state.get("source_identity")
        if not isinstance(previous, Mapping):
            raise CampaignBlocked(
                "campaign has no pinned source identity to migrate"
            )
        immutable_keys = (
            "numerical_source_sha256",
            "runner_sha256",
            "executor_sha256",
            "results_sha256",
        )
        drift = {
            key: {
                "expected": previous.get(key),
                "actual": current.get(key),
            }
            for key in immutable_keys
            if previous.get(key) != current.get(key)
        }
        if drift:
            raise CampaignBlocked(
                "controller-only migration refuses numerical/runner/"
                f"executor/results drift: {drift}"
            )
        old_campaign = previous.get("campaign_sha256")
        new_campaign = current.get("campaign_sha256")
        if (
            not isinstance(old_campaign, str)
            or not isinstance(new_campaign, str)
        ):
            raise CampaignBlocked("campaign source hashes are incomplete")
        if old_campaign != expected_old_campaign_sha256:
            raise CampaignBlocked(
                "pinned campaign SHA256 does not match explicit migration "
                "intent: "
                f"expected={expected_old_campaign_sha256}, "
                f"pinned={old_campaign}"
            )
        if new_campaign != accept_campaign_sha256:
            raise CampaignBlocked(
                "current campaign SHA256 does not match explicit migration "
                "intent: "
                f"accepted={accept_campaign_sha256}, current={new_campaign}"
            )
        if old_campaign == new_campaign:
            return {
                "status": "NO_CHANGE",
                "source_identity": current,
            }
        if not self.execute:
            return {
                "status": "DRY_RUN",
                "old_campaign_sha256": old_campaign,
                "new_campaign_sha256": new_campaign,
                "unchanged_source_identities": {
                    key: current[key] for key in immutable_keys
                },
            }
        if not self._refresh_plan_snapshot() or not self._check_plan_transition():
            raise CampaignBlocked(
                "plan drift prevents controller-source migration"
            )
        ready, ready_reason = agent_ready(self.agent_path, self.ready_marker)
        if not ready:
            raise CampaignBlocked(ready_reason)
        environment_ok, environment = venv_ready(self.plan)
        if not environment_ok:
            raise CampaignBlocked(
                "activate the exact repository .venv before controller "
                f"migration: {environment}"
            )
        snapshot = query_gpus()
        evidence = self._active_ownership_evidence(snapshot)
        current_after_evidence = self._current_source_identity()
        if current_after_evidence != current:
            raise CampaignBlocked(
                "source identity changed while controller migration evidence "
                "was being collected; retry with freshly reviewed hashes"
            )
        timestamp = _utc_now()
        transition_id = _sha256_bytes(
            _canonical_json_bytes(
                {
                    "reason": "controller_only_gpu_ownership_hotfix",
                    "old_source_identity": dict(previous),
                    "new_source_identity": current,
                    "plan_sha256": self.plan_sha256,
                }
            )
        )
        transition = {
            "schema_version": 1,
            "transition_id": transition_id,
            "timestamp_utc": timestamp,
            "reason": "controller_only_gpu_ownership_hotfix",
            "old_source_identity": dict(previous),
            "new_source_identity": current,
            "plan_sha256": self.plan_sha256,
            "ownership_evidence": evidence,
            "status": "PENDING_EVENT",
        }
        transitions = self.state.setdefault(
            "controller_source_transitions", []
        )
        transitions.append(transition)
        self.state["pending_controller_source_transition"] = transition
        self.persist()
        event = self.emit(
            "source",
            "controller_source_hotfix_accepted",
            details=transition,
        )
        if event is None:
            raise CampaignBlocked(
                "failed to append controller-source migration event"
            )
        transition["status"] = "COMMITTED"
        transition["event_seq"] = event["seq"]
        self.state["source_identity"] = current
        self.state["pending_controller_source_transition"] = None
        self.state["status"] = "WAIT_GPUS"
        self.state["blockers"] = [
            "controller source migrated; restart the campaign controller "
            "with the accepted hotfix"
        ]
        self.persist()
        return {
            "status": "MIGRATED",
            "old_campaign_sha256": old_campaign,
            "new_campaign_sha256": new_campaign,
            "event_seq": event["seq"],
            "active_task_count": len(evidence["active_tasks"]),
            "gpu_process_count": len(evidence["gpu_processes"]),
        }

    def commit_selected_lrs(self) -> dict[str, Any]:
        if not self.execute:
            patch = selected_lr_patch(self.state)
            return {
                "status": "DRY_RUN",
                "would_write": str(self.plan_path),
                "patch": patch,
            }
        if not self._refresh_plan_snapshot() or not self._check_plan_transition():
            raise CampaignBlocked(
                "plan changed outside the approved selected-LR transition"
            )
        ready, ready_reason = agent_ready(self.agent_path, self.ready_marker)
        if not ready:
            raise CampaignBlocked(ready_reason)
        environment_ok, environment = venv_ready(self.plan)
        if not environment_ok:
            raise CampaignBlocked(
                "source the exact repository .venv before committing selected "
                f"learning rates: {environment}"
            )
        patch = selected_lr_patch(self.state)
        if patch is None:
            raise CampaignBlocked(
                "all 15 model/setting searches must be SELECTED before commit"
            )
        if any(
            task["status"] in ACTIVE_TASK_STATES
            for task in self.state["tasks"].values()
        ):
            raise CampaignBlocked(
                "cannot change plan while campaign tasks are running"
            )
        snapshot = query_gpus()
        if snapshot.error:
            raise CampaignBlocked(
                "cannot prove that every GPU is idle before plan commit: "
                f"{snapshot.error}"
            )
        reserved = self._reserved_gpus()
        if reserved:
            raise CampaignBlocked(f"campaign still reserves GPUs {sorted(reserved)}")
        if snapshot.compute_processes:
            raise CampaignBlocked(
                "cannot change plan while GPU compute processes are visible"
            )
        current = json.loads(self.plan_path.read_text(encoding="utf-8"))
        current["selected_grad_lr_by_model_setting"] = patch[
            "selected_grad_lr_by_model_setting"
        ]
        new_bytes = (
            json.dumps(
                current,
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
        new_sha = _sha256_bytes(new_bytes)
        self.state["plan"]["pending_transition_sha256"] = new_sha
        self.state["selected_lr_patch"] = patch
        self.emit(
            "plan",
            "selected_lr_commit_intent",
            details={"new_plan_sha256": new_sha, "patch": patch},
        )
        self.persist()

        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.plan_path.parent,
            prefix=f".{self.plan_path.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(new_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.plan_path)
            directory_fd = os.open(self.plan_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()
            raise

        self.plan = current
        self.plan_raw = new_bytes
        self.plan_sha256 = new_sha
        self.state["plan"]["active_sha256"] = new_sha
        self.state["plan"]["formal_sha256"] = new_sha
        self.state["plan"]["pending_transition_sha256"] = None
        self.state["status"] = "WAIT_PREFLIGHT"
        preflight_path = _resolve_preflight_path(
            current, repo_root=self.repo_root
        )
        self.state["preflight"] = {
            "path": str(preflight_path) if preflight_path else None,
            "valid": False,
            "errors": [
                "plan SHA changed after selected-LR commit; rerun preflight at "
                "the configured path before formal experiments"
            ],
            "sha256": None,
        }
        self.emit(
            "plan",
            "selected_lr_committed",
            details={
                "plan_sha256": new_sha,
                "preflight_must_be_regenerated": (
                    str(preflight_path) if preflight_path else None
                ),
            },
        )
        self.persist()
        return {
            "status": "WAIT_PREFLIGHT",
            "plan": str(self.plan_path),
            "plan_sha256": new_sha,
            "patch": patch,
            "preflight_path": str(preflight_path) if preflight_path else None,
        }

    def close(self) -> None:
        for handle in self.controller_logs.values():
            with contextlib.suppress(Exception):
                handle.close()
        self.controller_logs.clear()


def _default_events_path(state_path: Path) -> Path:
    return state_path.with_name("events.jsonl")


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--agent", type=Path, default=DEFAULT_AGENT)
    parser.add_argument(
        "--agent-ready-marker",
        default=DEFAULT_READY_MARKER,
        help="literal AGENT.md substring required before any execution",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="allow state writes and executor launches; omission is read-only",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=DEFAULT_POLL_SECONDS,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument(
        "--once",
        action="store_true",
        help="perform one reconciliation/scheduling tick and return",
    )
    subparsers.add_parser("status")
    subparsers.add_parser("commit-selected-lrs")
    migration = subparsers.add_parser(
        "migrate-controller-source",
        help=(
            "explicitly accept a campaign.py-only hotfix after verifying "
            "all numerical identities and active GPU process ownership"
        ),
    )
    migration.add_argument(
        "--expected-old-campaign-sha256",
        required=True,
        help="exact pinned campaign.py SHA256 being replaced",
    )
    migration.add_argument(
        "--accept-campaign-sha256",
        required=True,
        help="exact reviewed campaign.py SHA256 to accept",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    events_path = args.events or _default_events_path(args.state)
    try:
        if args.command == "status":
            # Status never creates a lock/state file.
            campaign = Campaign(
                plan_path=args.plan,
                state_path=args.state,
                events_path=events_path,
                agent_path=args.agent,
                ready_marker=args.agent_ready_marker,
                execute=False,
                poll_seconds=args.poll_seconds,
            )
            try:
                payload = campaign.tick()
            finally:
                campaign.close()
        elif args.execute:
            with _campaign_lock(args.state.resolve(strict=False)):
                campaign = Campaign(
                    plan_path=args.plan,
                    state_path=args.state,
                    events_path=events_path,
                    agent_path=args.agent,
                    ready_marker=args.agent_ready_marker,
                    execute=True,
                    poll_seconds=args.poll_seconds,
                )
                try:
                    if args.command == "run":
                        payload = campaign.run(
                            once=args.once,
                            stream_heartbeats=not args.once,
                        )
                    elif args.command == "migrate-controller-source":
                        payload = campaign.migrate_controller_source(
                            expected_old_campaign_sha256=(
                                args.expected_old_campaign_sha256
                            ),
                            accept_campaign_sha256=(
                                args.accept_campaign_sha256
                            ),
                        )
                    else:
                        payload = campaign.commit_selected_lrs()
                finally:
                    campaign.close()
        else:
            campaign = Campaign(
                plan_path=args.plan,
                state_path=args.state,
                events_path=events_path,
                agent_path=args.agent,
                ready_marker=args.agent_ready_marker,
                execute=False,
                poll_seconds=args.poll_seconds,
            )
            try:
                if args.command == "run":
                    payload = campaign.run(once=True)
                elif args.command == "migrate-controller-source":
                    payload = campaign.migrate_controller_source(
                        expected_old_campaign_sha256=(
                            args.expected_old_campaign_sha256
                        ),
                        accept_campaign_sha256=(
                            args.accept_campaign_sha256
                        ),
                    )
                else:
                    payload = campaign.commit_selected_lrs()
            finally:
                campaign.close()
    except (
        CampaignError,
        OSError,
        json.JSONDecodeError,
        runner.PlanError,
    ) as exc:
        print(
            json.dumps(
                {
                    "status": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
