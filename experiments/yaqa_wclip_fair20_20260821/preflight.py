#!/usr/bin/env python3
"""Fail-closed preflight for the formal five-model YAQA_wclip campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
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


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)


def validate(plan: dict[str, Any], plan_sha: str) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig

    yaqa_root = REPO_ROOT / "YAQA_wclip"
    sys.path.insert(0, str(yaqa_root))
    from lib.codebook.wclip import WClipQuantizer
    from quantize_llama.preflight_wclip import calibration_contract

    source_path = Path(plan["source_plan"]).resolve()
    if sha256_file(source_path) != plan["source_plan_sha256"]:
        raise PreflightError("source comparison plan SHA256 changed")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    stages = plan["stages"]
    if len(stages) != 30 or len({stage["stage_id"] for stage in stages}) != 30:
        raise PreflightError("YAQA plan must contain 30 unique stages")
    hessians = [stage for stage in stages if stage["kind"] == "hessian"]
    quantizations = [stage for stage in stages if stage["kind"] == "quantize"]
    if len(hessians) != 10 or len(quantizations) != 20:
        raise PreflightError("YAQA plan must contain 10 Hessians and 20 quantizations")
    expected_quant = {
        (model, setting)
        for model in source["models"]
        for setting in ("W4A4KV4", "W4A16KV16", "W3A16KV16", "W2A16KV16")
    }
    if {(stage["model"], stage["setting"]) for stage in quantizations} != expected_quant:
        raise PreflightError("YAQA quantization matrix is not exact 5x4")
    if {(stage["model"], stage["hessian_mode"]) for stage in hessians} != {
        (model, mode) for model in source["models"] for mode in ("a16", "a4")
    }:
        raise PreflightError("YAQA Hessian matrix is not exact 5x2")
    hessian_by_id = {stage["stage_id"]: stage for stage in hessians}
    queues: dict[tuple[str, int], list[int]] = {}
    for stage in stages:
        job = plan["jobs"].get(stage["job_id"])
        gpu = int(stage["physical_gpu"])
        if (
            job is None
            or job["canoe_pod"] != stage["canoe_pod"]
            or not 0 <= gpu < int(job["gpu_count"])
        ):
            raise PreflightError(f"host/GPU binding mismatch: {stage['stage_id']}")
        queues.setdefault((stage["canoe_pod"], gpu), []).append(int(stage["queue_order"]))
        if stage["kind"] == "quantize":
            hessian = hessian_by_id.get(stage["hessian_id"])
            expected_mode = "a4" if stage["setting"] == "W4A4KV4" else "a16"
            expected_bits = 4 if stage["setting"].startswith("W4") else int(stage["setting"][1])
            if (
                hessian is None
                or hessian["model"] != stage["model"]
                or hessian["calibration"] != stage["calibration"]
                or hessian["hessian_mode"] != expected_mode
                or stage["w_bits"] != expected_bits
            ):
                raise PreflightError(f"Hessian dependency mismatch: {stage['stage_id']}")
    if any(sorted(values) != list(range(len(values))) for values in queues.values()):
        raise PreflightError("per-GPU queue_order is not contiguous")

    records = {}
    for model_key, model in source["models"].items():
        path = Path(model["path"]).resolve()
        config = AutoConfig.from_pretrained(path)
        head_dim = int(
            getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        )
        q_width = int(config.num_attention_heads) * head_dim
        dimensions = (
            int(config.hidden_size),
            int(config.intermediate_size),
            q_width,
        )
        if any(dimension % 128 for dimension in dimensions):
            raise PreflightError(f"G128/td_y dimensions invalid: {model_key}")
        if sha256_file(path / "config.json") != model["config_sha256"]:
            raise PreflightError(f"model config changed: {model_key}")
        calibration_key = next(
            stage["calibration"] for stage in stages if stage["model"] == model_key
        )
        token = source["calibrations"][calibration_key]["token_artifact"]
        token_path = Path(token["path"]).resolve()
        if (
            not token_path.is_file()
            or token_path.stat().st_size != token["size_bytes"]
            or sha256_file(token_path) != token["sha256"]
            or calibration_contract(path) != (token_path, token["sha256"])
        ):
            raise PreflightError(f"token/preflight map mismatch: {model_key}")
        values = torch.load(token_path, map_location="cpu", weights_only=True)
        if (
            type(values) is not list
            or len(values) != 256
            or any(
                not isinstance(item, torch.Tensor)
                or item.dtype != torch.int64
                or item.device.type != "cpu"
                or tuple(item.shape) != (2048,)
                for item in values
            )
        ):
            raise PreflightError(f"token tensor contract mismatch: {model_key}")
        records[model_key] = {
            "model_path": str(path),
            "layer_count": int(config.num_hidden_layers),
            "projection_input_dimensions": dimensions,
            "token_path": str(token_path),
            "token_sha256": token["sha256"],
        }

    for bits in (2, 3, 4):
        quantizer = WClipQuantizer(bits=bits, groupsize=128)
        if (
            quantizer.quantizer_name != "realq_wclip"
            or quantizer.version != 1
            or quantizer.vector_dim != 1
            or quantizer.sym is not True
            or quantizer.mse is not True
            or quantizer.search_impl != "cartesian_legacy"
            or quantizer.tie_rule != "strict_lt_first_winner"
        ):
            raise PreflightError(f"WClipQuantizer contract mismatch at W{bits}")

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
        REPO_ROOT / "YAQA_wclip/hessian_llama/get_hess_llama.py",
        REPO_ROOT / "YAQA_wclip/quantize_llama/quantize_finetune_llama.py",
        REPO_ROOT / "YAQA_wclip/quantize_llama/preflight_wclip.py",
        REPO_ROOT / "YAQA_wclip/quantize_llama/hfize_llama.py",
        REPO_ROOT / "YAQA_wclip/eval/validate_hfized_wclip.py",
        REPO_ROOT / "YAQA_wclip/lib/algo/finetune.py",
        REPO_ROOT / "YAQA_wclip/lib/algo/ldlq.py",
        REPO_ROOT / "YAQA_wclip/lib/codebook/wclip.py",
        REPO_ROOT / "YAQA_wclip/lib/codebook/__init__.py",
        REPO_ROOT / "experiments/yaqa_compare/validate_hessian.py",
    ]
    return {
        "schema_version": 1,
        "status": "succeeded",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": plan_sha,
        "source_plan_sha256": plan["source_plan_sha256"],
        "stage_count": len(stages),
        "hessian_count": len(hessians),
        "quantization_count": len(quantizations),
        "queue_count": len(queues),
        "versions": versions,
        "artifacts": records,
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
    write_json_atomic(output, validate(plan, actual_sha))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
