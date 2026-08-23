#!/usr/bin/env python3
"""Fail-closed static and artifact preflight for the formal TurboBOA matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = Path(__file__).with_name("plan.json")


class PreflightError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)


def validate_plan(plan: dict[str, Any], plan_sha: str) -> dict[str, Any]:
    import torch

    source_path = Path(plan["source_plan"]).resolve()
    if sha256_file(source_path) != plan["source_plan_sha256"]:
        raise PreflightError("comparison source plan SHA256 changed")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if Path(plan["workspace"]).resolve() != REPO_ROOT:
        raise PreflightError("workspace mismatch")
    runs = plan["runs"]
    if len(runs) != 20 or len({run["run_id"] for run in runs}) != 20:
        raise PreflightError("formal TurboBOA plan must contain 20 unique runs")
    expected_matrix = {
        (model, setting)
        for model in source["models"]
        for setting in ("W4A4KV4", "W4A16KV16", "W3A16KV16", "W2A16KV16")
    }
    actual_matrix = {(run["model"], run["setting"]) for run in runs}
    if actual_matrix != expected_matrix:
        raise PreflightError("model/setting matrix is not the exact 5x4 contract")

    queues: dict[tuple[str, int], list[int]] = {}
    artifact_records: dict[str, Any] = {}
    for run in runs:
        expected_bits = 4 if run["setting"].startswith("W4") else int(run["setting"][1])
        expected_akv = 4 if run["setting"] == "W4A4KV4" else 16
        if (
            run["w_bits"] != expected_bits
            or (run["a_bits"], run["k_bits"], run["v_bits"])
            != (expected_akv, expected_akv, expected_akv)
        ):
            raise PreflightError(f"bit contract mismatch: {run['run_id']}")
        job = plan["jobs"].get(run["job_id"])
        if job is None or job["canoe_pod"] != run["canoe_pod"]:
            raise PreflightError(f"job/pod mismatch: {run['run_id']}")
        gpu = int(run["physical_gpu"])
        if not 0 <= gpu < int(job["gpu_count"]):
            raise PreflightError(f"invalid GPU: {run['run_id']}")
        queues.setdefault((run["canoe_pod"], gpu), []).append(int(run["queue_order"]))
        model = source["models"][run["model"]]
        calibration = source["calibrations"][run["calibration"]]
        token = calibration["token_artifact"]
        if sha256_file(Path(model["path"]) / "config.json") != model["config_sha256"]:
            raise PreflightError(f"model config changed: {run['model']}")
        if (
            not Path(token["path"]).is_file()
            or Path(token["path"]).stat().st_size != token["size_bytes"]
            or sha256_file(token["path"]) != token["sha256"]
        ):
            raise PreflightError(f"token artifact changed: {run['calibration']}")
        artifact_records[run["model"]] = {
            "model_path": model["path"],
            "config_sha256": model["config_sha256"],
            "calibration_path": token["path"],
            "calibration_sha256": token["sha256"],
        }
    if any(sorted(orders) != list(range(len(orders))) for orders in queues.values()):
        raise PreflightError("every per-GPU queue_order must be contiguous from zero")

    for record in artifact_records.values():
        tokens = torch.load(record["calibration_path"], map_location="cpu", weights_only=True)
        if (
            type(tokens) is not list
            or len(tokens) != 256
            or any(
                not isinstance(item, torch.Tensor)
                or item.dtype != torch.int64
                or item.device.type != "cpu"
                or tuple(item.shape) != (2048,)
                for item in tokens
            )
        ):
            raise PreflightError("calibration tensor contract mismatch")

    versions = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "datasets": importlib.metadata.version("datasets"),
        "lm_eval": importlib.metadata.version("lm_eval"),
    }
    expected_versions = {
        "python": "3.12.3",
        "torch": "2.9.1",
        "transformers": "4.56.2",
        "datasets": "3.6.0",
        "lm_eval": "0.4.4",
    }
    if versions != expected_versions:
        raise PreflightError(f"formal environment mismatch: {versions}")

    source_files = [
        REPO_ROOT / "turboBOA/main.py",
        REPO_ROOT / "turboBOA/quantize.py",
        REPO_ROOT / "turboBOA/utils/process_args.py",
        REPO_ROOT / "turboBOA/utils/experiment_utils.py",
        REPO_ROOT / "turboBOA/quantizers/realq_mse.py",
        REPO_ROOT / "turboBOA/quantizers/turboboa.py",
        REPO_ROOT / "utils/quant_utils.py",
        REPO_ROOT / "utils/checkpoint_utils.py",
    ]
    return {
        "schema_version": 1,
        "status": "succeeded",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": plan_sha,
        "source_plan_sha256": plan["source_plan_sha256"],
        "run_count": len(runs),
        "queue_count": len(queues),
        "versions": versions,
        "artifacts": artifact_records,
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in source_files
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-file", default=str(DEFAULT_PLAN))
    parser.add_argument("--expected-plan-sha256", required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan_file).resolve()
    actual_sha = sha256_file(plan_path)
    if actual_sha != args.expected_plan_sha256:
        raise PreflightError(f"plan SHA256 mismatch: {actual_sha}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    output = Path(plan["preflight_path"])
    if output.exists() or output.is_symlink():
        raise PreflightError(f"preflight output must be fresh: {output}")
    write_json_atomic(output, validate_plan(plan, actual_sha))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
