"""Run a CUDA exactness probe with immutable, fail-closed evidence.

The runner binds the raw log to a clean Git commit, exact probe bytes, command,
environment, return code, and the physical NVIDIA GPU ID/UUID snapshot.  It is
intentionally stricter than invoking ``torchrun`` directly.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


_PROBES = {
    "p03-p04": (
        "tools/p03_p04_cuda_probe.py",
        "REALQ_P03_P04_PROBE_DEVICE",
    ),
    "p06": (
        "tools/p06_distributed_cpu_probe.py",
        "REALQ_P06_PROBE_DEVICE",
    ),
}
_ALLOWED_PHYSICAL_GPUS = frozenset({4, 5, 6, 7})


class EvidenceFailure(RuntimeError):
    """The probe or one of its provenance contracts failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceFailure(message)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _command(
    argv: list[str],
    *,
    cwd: Path,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and completed.returncode != 0:
        raise EvidenceFailure(
            f"command failed ({completed.returncode}): "
            f"{shlex.join(argv)}\n"
            f"stdout={completed.stdout.decode(errors='replace')}\n"
            f"stderr={completed.stderr.decode(errors='replace')}"
        )
    return completed


def _git(repo: Path, *args: str) -> bytes:
    return _command(
        ["git", "-c", f"safe.directory={repo}", *args],
        cwd=repo,
    ).stdout


def _source_snapshot(repo: Path) -> dict[str, Any]:
    tracked = _git(repo, "diff", "--no-ext-diff", "--binary")
    staged = _git(
        repo,
        "diff",
        "--cached",
        "--no-ext-diff",
        "--binary",
    )
    status = _git(repo, "status", "--porcelain=v1", "-z")
    commit = _git(repo, "rev-parse", "HEAD").decode().strip()
    _require(
        len(commit) == 40,
        f"git rev-parse returned invalid commit {commit!r}",
    )
    return {
        "git_commit": commit,
        "tracked_diff_bytes": len(tracked),
        "tracked_diff_sha256": _sha256_bytes(tracked),
        "staged_diff_bytes": len(staged),
        "staged_diff_sha256": _sha256_bytes(staged),
        "status_porcelain_v1_z_bytes": len(status),
        "status_porcelain_v1_z_sha256": _sha256_bytes(status),
        "status_porcelain_v1": status.decode(errors="surrogateescape")
        .replace("\0", "\n")
        .splitlines(),
    }


def _parse_gpu_snapshot(text: str) -> list[dict[str, Any]]:
    rows = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = [part.strip() for part in line.split(",")]
        _require(
            len(fields) == 7,
            f"unexpected nvidia-smi GPU row: {raw_line!r}",
        )
        rows.append(
            {
                "physical_gpu_id": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "memory_total_mib": int(fields[3]),
                "memory_used_mib": int(fields[4]),
                "utilization_gpu_percent": int(fields[5]),
                "driver_version": fields[6],
            }
        )
    _require(rows, "nvidia-smi returned no GPUs")
    return rows


def _gpu_snapshot(repo: Path) -> list[dict[str, Any]]:
    query = _command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,"
            "utilization.gpu,driver_version",
            "--format=csv,noheader,nounits",
        ],
        cwd=repo,
    )
    return _parse_gpu_snapshot(query.stdout.decode())


def _compute_processes(repo: Path) -> list[dict[str, Any]]:
    completed = _command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        cwd=repo,
        check=False,
    )
    # NVIDIA returns a non-zero code on some driver versions when there are no
    # compute applications. An empty stdout is the only accepted such result.
    if completed.returncode != 0 and completed.stdout.strip():
        raise EvidenceFailure(
            "nvidia-smi compute-process query failed with output: "
            f"{completed.stdout.decode(errors='replace')}"
        )
    rows = []
    for line in completed.stdout.decode().splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            pid = int(fields[1])
            memory = int(fields[2])
        except ValueError:
            continue
        rows.append(
            {
                "uuid": fields[0],
                "pid": pid,
                "used_memory_mib": memory,
            }
        )
    return rows


def _normalize_uuid(uuid: str) -> str:
    return uuid.removeprefix("GPU-").lower()


def _validate_selected_idle(
    snapshot: list[dict[str, Any]],
    processes: list[dict[str, Any]],
    gpu_ids: list[int],
) -> list[dict[str, Any]]:
    by_id = {item["physical_gpu_id"]: item for item in snapshot}
    selected = []
    for physical_id in gpu_ids:
        _require(
            physical_id in by_id,
            f"physical GPU {physical_id} is absent from nvidia-smi",
        )
        item = by_id[physical_id]
        _require(
            item["memory_used_mib"] <= 16,
            f"physical GPU {physical_id} is not idle: "
            f"memory_used={item['memory_used_mib']} MiB",
        )
        _require(
            item["utilization_gpu_percent"] == 0,
            f"physical GPU {physical_id} is not idle: "
            f"utilization={item['utilization_gpu_percent']}%",
        )
        selected.append(item)
    selected_uuids = {
        _normalize_uuid(item["uuid"]) for item in selected
    }
    selected_processes = [
        process
        for process in processes
        if _normalize_uuid(process["uuid"]) in selected_uuids
    ]
    _require(
        not selected_processes,
        f"selected GPUs have compute processes: {selected_processes}",
    )
    return selected


def _parse_result_json(raw_log: Path) -> dict[str, Any]:
    payloads = []
    for raw_line in raw_log.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    _require(
        len(payloads) == 1,
        f"raw log must contain exactly one JSON result, found "
        f"{len(payloads)}",
    )
    return payloads[0]


def _validate_result(
    result: dict[str, Any],
    *,
    probe_sha256: str,
    commit: str,
    world_size: int,
    gpu_csv: str,
    selected_gpus: list[dict[str, Any]],
) -> None:
    _require(
        result.get("world_size") == world_size,
        f"probe world_size={result.get('world_size')} != {world_size}",
    )
    _require(
        result.get("backend") == "nccl",
        f"probe backend={result.get('backend')!r} is not nccl",
    )
    _require(
        result.get("cuda_visible_devices") == gpu_csv,
        "probe CUDA_VISIBLE_DEVICES does not match the command",
    )
    _require(
        result.get("probe_sha256") == probe_sha256,
        "raw probe result is not bound to the exact probe bytes",
    )
    _require(
        result.get("expected_git_commit") == commit,
        "raw probe result is not bound to the exact Git commit",
    )
    _require(
        result.get("python_optimize") == 0
        and result.get("python_debug") is True,
        "probe ran with optimized Python or disabled __debug__",
    )
    devices = result.get("devices")
    if isinstance(devices, list):
        _require(
            len(devices) == world_size,
            "probe device metadata count differs from world size",
        )
        for rank, (device, physical) in enumerate(
            zip(devices, selected_gpus)
        ):
            _require(
                device.get("rank") == rank
                and device.get("local_rank") == rank,
                f"probe rank mapping is invalid at rank {rank}: {device}",
            )
            _require(
                _normalize_uuid(str(device.get("uuid")))
                == _normalize_uuid(str(physical["uuid"])),
                f"rank {rank} UUID does not map to physical GPU "
                f"{physical['physical_gpu_id']}",
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", choices=sorted(_PROBES), required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    if sys.flags.optimize != 0 or not __debug__:
        raise EvidenceFailure(
            "evidence runner refuses optimized Python: "
            f"sys.flags.optimize={sys.flags.optimize}, "
            f"__debug__={__debug__}"
        )
    args = _parse_args()
    _require(
        args.world_size in (1, 2, 4),
        "world size must be 1, 2, or 4",
    )
    _require(
        1024 <= args.master_port <= 65535,
        "master port is outside 1024..65535",
    )
    try:
        gpu_ids = [int(item) for item in args.gpus.split(",")]
    except ValueError as error:
        raise EvidenceFailure("--gpus must be comma-separated integers") from error
    _require(
        len(gpu_ids) == args.world_size and len(set(gpu_ids)) == len(gpu_ids),
        "GPU count must equal world size and contain no duplicates",
    )
    _require(
        set(gpu_ids).issubset(_ALLOWED_PHYSICAL_GPUS),
        "this evidence gate is restricted to physical GPUs 4–7",
    )
    gpu_csv = ",".join(str(item) for item in gpu_ids)

    repo = Path(__file__).resolve().parents[1]
    probe_relative, device_env_name = _PROBES[args.probe]
    probe_path = repo / probe_relative
    _require(probe_path.is_file(), f"probe is missing: {probe_path}")
    torchrun = shutil.which("torchrun")
    _require(torchrun is not None, "torchrun is unavailable")
    _require(shutil.which("nvidia-smi") is not None, "nvidia-smi missing")
    source_before = _source_snapshot(repo)
    _require(
        source_before["tracked_diff_bytes"] == 0
        and source_before["staged_diff_bytes"] == 0,
        "tracked/staged source changes exist; commit them before evidence",
    )
    commit = str(source_before["git_commit"])
    probe_sha256 = _sha256_file(probe_path)
    harness_sha256 = _sha256_file(Path(__file__).resolve())

    gpu_snapshot = _gpu_snapshot(repo)
    compute_processes = _compute_processes(repo)
    selected_gpus = _validate_selected_idle(
        gpu_snapshot, compute_processes, gpu_ids
    )

    started = dt.datetime.now(dt.timezone.utc)
    run_id = (
        started.strftime("%Y%m%dT%H%M%SZ")
        + f"_{commit[:12]}_{args.probe}_w{args.world_size}"
    )
    output_parent = args.output_parent.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    run_root = output_parent / run_id
    run_root.mkdir()
    raw_log = run_root / "raw.log"
    result_path = run_root / "result.json"
    manifest_path = run_root / "manifest.json"

    command = [
        str(torchrun),
        "--standalone",
        f"--master-port={args.master_port}",
        f"--nproc_per_node={args.world_size}",
        str(probe_path),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": gpu_csv,
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(repo),
            "REALQ_PROBE_EXPECTED_GIT_COMMIT": commit,
            "REALQ_PROBE_EXPECTED_SHA256": probe_sha256,
            device_env_name: "cuda",
        }
    )
    recorded_environment = {
        key: environment.get(key)
        for key in sorted(environment)
        if key.startswith(("CUDA", "NCCL", "REALQ", "PYTHON"))
    }

    command_return_code = -1
    validation_errors: list[str] = []
    result: dict[str, Any] | None = None
    with raw_log.open("wb") as log_file:
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        _require(process.stdout is not None, "failed to capture probe output")
        for line in iter(process.stdout.readline, b""):
            log_file.write(line)
            log_file.flush()
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
        command_return_code = process.wait()

    source_after = _source_snapshot(repo)
    gpu_snapshot_after = _gpu_snapshot(repo)
    compute_processes_after = _compute_processes(repo)
    try:
        _require(
            command_return_code == 0,
            f"probe command returned {command_return_code}",
        )
        _require(
            source_after["git_commit"] == commit,
            "Git commit changed during the probe",
        )
        _require(
            source_after["tracked_diff_bytes"] == 0
            and source_after["staged_diff_bytes"] == 0,
            "tracked/staged source changed during the probe",
        )
        _require(
            source_after["tracked_diff_sha256"]
            == source_before["tracked_diff_sha256"]
            and source_after["staged_diff_sha256"]
            == source_before["staged_diff_sha256"],
            "source diff hashes changed during the probe",
        )
        _require(
            _sha256_file(probe_path) == probe_sha256,
            "probe bytes changed during execution",
        )
        selected_gpus_after = _validate_selected_idle(
            gpu_snapshot_after, compute_processes_after, gpu_ids
        )
        _require(
            [
                (item["physical_gpu_id"], item["uuid"])
                for item in selected_gpus_after
            ]
            == [
                (item["physical_gpu_id"], item["uuid"])
                for item in selected_gpus
            ],
            "physical GPU ID/UUID mapping changed during the probe",
        )
        result = _parse_result_json(raw_log)
        _validate_result(
            result,
            probe_sha256=probe_sha256,
            commit=commit,
            world_size=args.world_size,
            gpu_csv=gpu_csv,
            selected_gpus=selected_gpus,
        )
    except Exception as error:  # noqa: BLE001 - retained in evidence
        validation_errors.append(f"{type(error).__name__}: {error}")

    if result is not None:
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    ended = dt.datetime.now(dt.timezone.utc)
    passed = not validation_errors
    manifest = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "run_id": run_id,
        "probe": args.probe,
        "started_utc": started.isoformat(),
        "ended_utc": ended.isoformat(),
        "duration_seconds": (ended - started).total_seconds(),
        "repo_root": str(repo),
        "probe_path": str(probe_path),
        "probe_sha256": probe_sha256,
        "harness_path": str(Path(__file__).resolve()),
        "harness_sha256": harness_sha256,
        "command_argv": command,
        "command_shell": shlex.join(command),
        "command_return_code": command_return_code,
        "environment": recorded_environment,
        "runner_python": sys.executable,
        "runner_python_optimize": sys.flags.optimize,
        "runner_python_debug": __debug__,
        "source_before": source_before,
        "source_after": source_after,
        "physical_gpu_ids_in_rank_order": gpu_ids,
        "physical_gpu_uuids_in_rank_order": [
            item["uuid"] for item in selected_gpus
        ],
        "rank_to_physical_gpu_mapping": [
            {
                "rank": rank,
                "physical_gpu_id": item["physical_gpu_id"],
                "uuid": item["uuid"],
            }
            for rank, item in enumerate(selected_gpus)
        ],
        "nvidia_smi_global_gpu_snapshot_before": gpu_snapshot,
        "nvidia_smi_global_gpu_snapshot_after": gpu_snapshot_after,
        "nvidia_smi_compute_processes_before": compute_processes,
        "nvidia_smi_compute_processes_after": compute_processes_after,
        "raw_log_path": str(raw_log),
        "raw_log_sha256": _sha256_file(raw_log),
        "raw_log_bytes": raw_log.stat().st_size,
        "result_json_path": str(result_path) if result is not None else None,
        "result_json_sha256": (
            _sha256_file(result_path) if result is not None else None
        ),
        "validation_errors": validation_errors,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"evidence_root={run_root}")
    print(f"manifest={manifest_path}")
    print(f"status={manifest['status']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
