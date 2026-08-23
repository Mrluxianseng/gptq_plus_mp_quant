#!/usr/bin/env python3
"""Freeze and prepare caches for the REALQ run15-LR SDPA replay.

The scientific delta from run15 is deliberately narrow: FA4 is replaced by
deterministic math-SDPA.  Every branch/config keeps its already-selected V6
learning rate and all other run15 flags.  Calibration tensors and WikiText-2
teacher logits are the exact artifacts consumed by the GPTAQ/GuidedQuant
fair-20 campaign; only the backend-dependent static Fisher cache is fresh.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import functools
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as base
from experiments.realq_fullmodel_retune_20260817 import campaign_v2 as v2
from experiments.realq_fullmodel_retune_20260817 import campaign_v6_run15 as v6


CAMPAIGN_ID = "realq-sdpa-frozen-run15-lr20-20260822-v1"
OUTPUT_ROOT = (
    base.REPO_ROOT.parent
    / "experiment_data"
    / "realq_sdpa_frozen_lr20_20260822_v1"
)
PLAN_PATH = OUTPUT_ROOT / "plan.json"
CACHE_ROOT = OUTPUT_ROOT / "shared_cache"
SOURCE_PLAN_PATH = v6.PLAN_PATH
SOURCE_SELECTION_PATH = v6.OUTPUT_ROOT / "selections.json"
BASELINE_ROOT = (
    base.REPO_ROOT.parent / "experiment_data" / "gptaq_guided_20group_20260809"
)
BASELINE_PLAN_PATH = BASELINE_ROOT / "plan.json"
LOCK_ROOT = (
    base.REPO_ROOT.parent / "experiment_data" / "_fair20_physical_gpu_locks_20260821"
)
MODULE_PATH = "experiments/realq_sdpa_frozen_lr20_20260822/campaign.py"
SELECTION_MODULE_PATH = "experiments/realq_sdpa_frozen_lr20_20260822/selection.py"

RUN_FLAGS = {
    **v6.RUN15_FLAGS,
    "--attention_backend": "sdpa",
}
WORKER_ENV_OVERRIDES = {
    **v6.RUN15_ENV_OVERRIDES,
    "REALQ_DETERMINISTIC_SDPA": "1",
}
WORKER_ENV_UNSET = ("NVIDIA_TF32_OVERRIDE",)
CODE_INPUTS = tuple(
    dict.fromkeys(
        (
            *base.CODE_INPUTS,
            *v6.RUN15_CODE_INPUTS,
            "utils/reproducibility.py",
            MODULE_PATH,
            SELECTION_MODULE_PATH,
        )
    )
)

TOKEN_ARCHIVE_SHA256 = {
    "qwen3-0.6b": "aed4e972d31207526d57a4bb62414c24b6c92ff51eda1fa9d70c36c1dfea5d34",
    "llama31-8b-instruct": "8030125e31c8738d6cea839d9afd5252617acf9517a226b56ebd1c69804be95e",
    "qwen3-4b": "21210e1929aa90ea23572e7904f8ebda7b3af75ebf8d9037e8196b3d2c52a399",
    "qwen3-8b": "a7acbd907d640eb8ab33a52153da1bf36e4eb550cbc107fe081c9bdc047d3b6f",
    "qwen3-32b": "749360f8fb36a6ac91936954f61a7ec688fa8eb0b0d17a61fe4349ff6ec06364",
}
TOKEN_SEMANTIC_SHA256 = {
    "qwen3-0.6b": "5b7bd51b6f896a4b79e70f8ffa867974440dc5c63b6c8af7668c013d76998535",
    "llama31-8b-instruct": "47afb29790e86ca5f6d4e297d5adf2c75e10e52997803f1e1b21bb1575b21993",
    "qwen3-4b": "5b7bd51b6f896a4b79e70f8ffa867974440dc5c63b6c8af7668c013d76998535",
    "qwen3-8b": "5b7bd51b6f896a4b79e70f8ffa867974440dc5c63b6c8af7668c013d76998535",
    "qwen3-32b": "5b7bd51b6f896a4b79e70f8ffa867974440dc5c63b6c8af7668c013d76998535",
}


def _load_fingerprinted(path: Path, fingerprint_key: str) -> dict[str, Any]:
    value = base._read_json(path)
    stable = dict(value)
    fingerprint = stable.pop(fingerprint_key, None)
    stable.pop("created_at", None)
    if base._canonical_sha256(stable) != fingerprint:
        raise base.CampaignError(f"frozen artifact fingerprint mismatch: {path}")
    return value


def _source_plan() -> dict[str, Any]:
    return _load_fingerprinted(SOURCE_PLAN_PATH, "protocol_fingerprint")


def _source_selection() -> dict[str, Any]:
    value = base._read_json(SOURCE_SELECTION_PATH)
    stable = dict(value)
    fingerprint = stable.pop("selection_fingerprint", None)
    stable.pop("created_at", None)
    if base._canonical_sha256(stable) != fingerprint:
        raise base.CampaignError("source V6 selection fingerprint mismatch")
    if value.get("campaign_id") != v6.CAMPAIGN_ID or len(value.get("rows", [])) != 40:
        raise base.CampaignError("source V6 selection identity/matrix mismatch")
    return value


def _baseline_plan() -> dict[str, Any]:
    value = base._read_json(BASELINE_PLAN_PATH)
    if value.get("campaign_id") != "gptaq-guided-llama31-qwen3-20group-20260809-v1":
        raise base.CampaignError("wrong GPTAQ/GuidedQuant baseline plan")
    stable = dict(value)
    fingerprint = stable.pop("fingerprint", None)
    if base._canonical_sha256(stable) != fingerprint:
        raise base.CampaignError("baseline plan fingerprint mismatch")
    return value


def _model_for_config(config: str) -> str:
    for model in base.MODEL_SLUGS:
        if config.startswith(f"{model}_"):
            return model
    raise base.CampaignError(f"unknown model for config: {config}")


def _flags(command: Sequence[str]) -> dict[str, str]:
    return v2._flag_map(command)


def _static_root(model: str) -> Path:
    return CACHE_ROOT / model / "static"


def _token_semantic_sha256(path: Path) -> tuple[str, int, tuple[int, ...], str]:
    import torch

    tensors = torch.load(path, map_location="cpu", weights_only=True)
    digest = hashlib.sha256()
    for tensor in tensors:
        tensor = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"|")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"|")
        digest.update(tensor.numpy().tobytes(order="C"))
    first = tensors[0]
    return digest.hexdigest(), len(tensors), tuple(first.shape), str(first.dtype)


@functools.lru_cache(maxsize=1)
def _baseline_cache_contracts() -> dict[str, Any]:
    source = _source_plan()
    baseline = _baseline_plan()
    rows: dict[str, Any] = {}
    for model in base.MODEL_SLUGS:
        link = baseline["cache_links"][model]
        token_path = Path(str(link["token"]["source"]))
        reference_path = Path(str(link["reference_logits"]["source"]))
        command = source["configurations"][
            f"single_linear/{model}_w4a16"
        ]["source_command"]
        token_root = Path(base._arg_value(command, "--tokens_cache_path"))
        token_files = sorted(token_root.glob("*.pt"))
        if token_files != [token_path]:
            raise base.CampaignError(f"token path differs from baseline: {model}")
        if base._file_sha256(token_path) != TOKEN_ARCHIVE_SHA256[model]:
            raise base.CampaignError(f"token archive changed: {model}")
        semantic_sha256, tensor_count, tensor_shape, tensor_dtype = (
            _token_semantic_sha256(token_path)
        )
        if semantic_sha256 != TOKEN_SEMANTIC_SHA256[model]:
            raise base.CampaignError(f"token semantic content changed: {model}")
        if (tensor_count, tensor_shape, tensor_dtype) != (256, (2048,), "torch.int64"):
            raise base.CampaignError(f"token tensor contract changed: {model}")
        token_stat = token_path.stat()
        reference_stat = reference_path.stat()
        if (
            token_stat.st_size != int(link["token"]["size_bytes"])
            or token_stat.st_mtime_ns != int(link["token"]["source_mtime_ns"])
            or reference_stat.st_size
            != int(link["reference_logits"]["size_bytes"])
            or reference_stat.st_mtime_ns
            != int(link["reference_logits"]["source_mtime_ns"])
        ):
            raise base.CampaignError(f"baseline cache metadata changed: {model}")
        if reference_path.parent.name != "ref_logits":
            raise base.CampaignError(f"invalid baseline reference path: {reference_path}")
        rows[model] = {
            "tokens": {
                "path": str(token_path),
                "size_bytes": token_stat.st_size,
                "mtime_ns": token_stat.st_mtime_ns,
                "archive_sha256": TOKEN_ARCHIVE_SHA256[model],
                "semantic_sha256": TOKEN_SEMANTIC_SHA256[model],
                "tensor_contract": "256 x [2048] torch.int64 in list order",
                "same_physical_source_as_gptaq_guidedquant": True,
            },
            "reference_logits": {
                "path": str(reference_path),
                "runtime_root": str(reference_path.parents[1]),
                "size_bytes": reference_stat.st_size,
                "mtime_ns": reference_stat.st_mtime_ns,
                "same_physical_source_as_gptaq_guidedquant": True,
                "regeneration_forbidden": True,
            },
        }
    return rows


def _configure_command(command: list[str], *, branch: str, config: str) -> list[str]:
    model = _model_for_config(config)
    contracts = _baseline_cache_contracts()
    base._set_or_append_arg(command, "--full_block_refresh", base.BRANCH_VALUES[branch])
    for flag, value in RUN_FLAGS.items():
        base._set_or_append_arg(command, flag, value)
    base._set_or_append_arg(command, "--static_cache_path", str(_static_root(model)))
    base._set_or_append_arg(
        command,
        "--cache_dir",
        str(contracts[model]["reference_logits"]["runtime_root"]),
    )
    base._set_or_append_arg(command, "--require_static_cache_hit", "true")
    base._set_or_append_arg(command, "--require_reference_cache_hit", "true")
    base._remove_arg(command, "--quant_stop_layer")
    return command


def _validate_full_profile(
    command: Sequence[str], *, branch: str, config: str
) -> None:
    model = _model_for_config(config)
    expected = {
        "--dataset": "wikitext2",
        "--eval_datasets": "wikitext2",
        "--seed": "1",
        "--rotation_seed": "0",
        "--refresh_seed": "0",
        "--nsamples": "256",
        "--seq_len": "2048",
        "--eval_seq_len": "2048",
        "--w_groupsize": "128",
        "--w_asym": "false",
        "--w_clip": "true",
        "--blocksize": "128",
        "--act_order": "true",
        "--backward_samples": "32",
        "--backward_bsz": "32",
        "--loss_slide_window": "true",
        "--full_block_refresh": base.BRANCH_VALUES[branch],
        "--grad_lr_layer_schedule": base._schedule_for(config),
        "--a_loss_ratio": "1",
        "--rotate": "true",
        "--require_static_cache_hit": "true",
        "--require_reference_cache_hit": "true",
        **RUN_FLAGS,
    }
    values = _flags(command)
    for flag, wanted in expected.items():
        if values.get(flag) != wanted:
            raise base.CampaignError(
                f"SDPA {branch}/{config} {flag}: expected {wanted}, got {values.get(flag)}"
            )
    wanted_hessian = "32" if model == "qwen3-32b" else "64"
    if values.get("--hessian_accum_bsz") != wanted_hessian:
        raise base.CampaignError(f"SDPA hessian_accum_bsz changed: {branch}/{config}")
    if values.get("--static_cache_path") != str(_static_root(model)):
        raise base.CampaignError("SDPA run points at wrong static cache")
    contract = _baseline_cache_contracts()[model]
    if values.get("--cache_dir") != contract["reference_logits"]["runtime_root"]:
        raise base.CampaignError("SDPA run is not using baseline reference cache")
    token_root = Path(values["--tokens_cache_path"])
    if sorted(token_root.glob("*.pt")) != [Path(contract["tokens"]["path"])]:
        raise base.CampaignError("SDPA run is not using baseline calibration tokens")
    if base._arg_indices(command, "--quant_stop_layer"):
        raise base.CampaignError("formal SDPA run must cover the full model")


def _build_plan() -> dict[str, Any]:
    source = _source_plan()
    selection = _source_selection()
    contracts = _baseline_cache_contracts()
    configurations: dict[str, Any] = {}
    parity: dict[str, Any] = {}
    allowed_deltas = {"--attention_backend", "--static_cache_path", "--cache_dir"}
    for config in base.CONFIG_IDS:
        normalized: dict[str, dict[str, str]] = {}
        for branch in base.BRANCH_VALUES:
            key = f"{branch}/{config}"
            source_row = source["configurations"][key]
            source_command = list(map(str, source_row["source_command"]))
            command = _configure_command(source_command.copy(), branch=branch, config=config)
            _validate_full_profile(command, branch=branch, config=config)
            before, after = _flags(source_command), _flags(command)
            changed = {
                flag: {"run15": before.get(flag), "sdpa": after.get(flag)}
                for flag in sorted(set(before) | set(after))
                if before.get(flag) != after.get(flag)
            }
            if set(changed) != allowed_deltas:
                raise base.CampaignError(f"unexpected run15 delta {key}: {changed}")
            row = copy.deepcopy(source_row)
            row.update(
                source_command=command,
                source_run15_command_sha256=base._canonical_sha256(source_command),
                sdpa_command_sha256=base._canonical_sha256(command),
                numerical_delta=changed,
                frozen_run15_lr_only=True,
            )
            configurations[key] = row
            branch_flags = dict(after)
            branch_flags.pop("--full_block_refresh")
            normalized[branch] = branch_flags
        if normalized["full_block"] != normalized["single_linear"]:
            raise base.CampaignError(f"branch parity failed: {config}")
        parity[config] = {
            "only_branch_difference": "--full_block_refresh=true/false",
            "normalized_command_sha256": base._canonical_sha256(normalized["full_block"]),
        }
    selected = {
        f"{row['branch']}/{row['config']}": float(row["selected_lr"])
        for row in selection["rows"]
    }
    if set(selected) != set(configurations):
        raise base.CampaignError("source selection matrix differs from 40 commands")
    body: dict[str, Any] = {
        "campaign_id": CAMPAIGN_ID,
        "output_root": str(OUTPUT_ROOT),
        "source_run15": {
            "plan": {"path": str(SOURCE_PLAN_PATH), "sha256": base._file_sha256(SOURCE_PLAN_PATH)},
            "protocol_fingerprint": source["protocol_fingerprint"],
            "selection": {
                "path": str(SOURCE_SELECTION_PATH),
                "sha256": base._file_sha256(SOURCE_SELECTION_PATH),
                "selection_fingerprint": selection["selection_fingerprint"],
            },
        },
        "baseline_gptaq_guidedquant": {
            "plan": {"path": str(BASELINE_PLAN_PATH), "sha256": base._file_sha256(BASELINE_PLAN_PATH)},
            "fingerprint": _baseline_plan()["fingerprint"],
        },
        "branches": dict(base.BRANCH_VALUES),
        "config_ids": list(base.CONFIG_IDS),
        "selected_lrs": selected,
        "calibration_and_reference_contracts": contracts,
        "determinism": {
            "seed": 1,
            "rotation_seed": 0,
            "refresh_seed": 0,
            "torch_deterministic_algorithms": True,
            "attention_backend": "math-SDPA",
            "realq_deterministic_sdpa": True,
            "cublas_workspace_config": ":4096:8",
            "pythonhashseed": "0",
            "nvidia_tf32_override": "unset; scoped Hessian/Fisher code controls TF32",
        },
        "optimization_profile": {
            "source": "run15",
            "only_numerical_codepath_delta": "attention_backend: flash_attention_4 -> sdpa",
            "fresh_backend_isolated_static_cache": True,
            "baseline_sdpa_reference_cache_is_read_only": True,
            "retained_flags": dict(RUN_FLAGS),
        },
        "paired_profile_audit": parity,
        "code_snapshot": base._code_snapshot(),
        "configurations": configurations,
    }
    body["protocol_fingerprint"] = base._canonical_sha256(body)
    body["created_at"] = base._utc_now()
    return body


def _selection_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    source = _source_selection()
    rows = []
    for row in source["rows"]:
        rows.append(
            {
                "branch": str(row["branch"]),
                "config": str(row["config"]),
                "selected_lr": float(row["selected_lr"]),
                "reason": "verbatim adoption of frozen deterministic-FA4 run15 LR; no SDPA retune",
                "source_selection_row": row,
            }
        )
    module = base.REPO_ROOT / SELECTION_MODULE_PATH
    value: dict[str, Any] = {
        "selection_id": "realq-sdpa-frozen-run15-lr-selection-20260822-v1",
        "campaign_id": CAMPAIGN_ID,
        "protocol_fingerprint": plan["protocol_fingerprint"],
        "plan": {"path": str(PLAN_PATH), "sha256": base._file_sha256(PLAN_PATH)},
        "source_selection": plan["source_run15"]["selection"],
        "selection_code": [{"path": str(module), "sha256": base._file_sha256(module)}],
        "rows": rows,
    }
    value["selection_fingerprint"] = base._canonical_sha256(value)
    value["created_at"] = base._utc_now()
    return value


def _write_plan(_: argparse.Namespace) -> int:
    candidate = _build_plan()
    if PLAN_PATH.is_file():
        existing = base._read_json(PLAN_PATH)
        _verify_plan(existing)
        if existing["protocol_fingerprint"] != candidate["protocol_fingerprint"]:
            raise base.CampaignError("existing SDPA plan differs from current inputs")
    else:
        base._atomic_json(PLAN_PATH, candidate)
    plan = base._read_json(PLAN_PATH)
    selection_path = OUTPUT_ROOT / "selections.json"
    selection = _selection_payload(plan)
    if selection_path.is_file():
        existing_selection = base._read_json(selection_path)
        existing_stable = dict(existing_selection)
        existing_stable.pop("created_at", None)
        candidate_stable = dict(selection)
        candidate_stable.pop("created_at", None)
        if existing_stable != candidate_stable:
            raise base.CampaignError("existing frozen-LR selection differs")
    else:
        base._atomic_json(selection_path, selection)
    print(PLAN_PATH)
    print(selection_path)
    return 0


def _verify_plan(plan: Mapping[str, Any]) -> None:
    base._verify_plan(plan)
    if plan.get("source_run15", {}).get("plan", {}).get("sha256") != base._file_sha256(
        SOURCE_PLAN_PATH
    ):
        raise base.CampaignError("source run15 plan changed")
    if plan.get("source_run15", {}).get("selection", {}).get("sha256") != base._file_sha256(
        SOURCE_SELECTION_PATH
    ):
        raise base.CampaignError("source run15 selection changed")
    _baseline_cache_contracts()


def _cache_command(plan: Mapping[str, Any], model: str, attempt: Path) -> list[str]:
    config = f"{model}_w4a16"
    command = list(plan["configurations"][f"single_linear/{config}"]["source_command"])
    base._set_arg(command, "--grad_lr", "0")
    base._set_arg(command, "--skip_eval", "false")
    base._set_arg(command, "--skip_kl_ppl_eval", "false")
    base._set_arg(command, "--lm_eval", "false")
    base._set_arg(command, "--reasoning_eval", "false")
    base._set_arg(command, "--exit_after_precompute", "true")
    base._set_arg(command, "--require_static_cache_hit", "false")
    base._set_arg(command, "--require_reference_cache_hit", "true")
    base._set_arg(command, "--output_dir", str(attempt / "realq_output"))
    base._set_arg(command, "--exp", "sdpa_frozen_lr20_cache_producer")
    base._remove_arg(command, "--save_qmodel_path")
    base._remove_arg(command, "--load_qmodel_path")
    return command


def _worker_environment(cuda_id: str) -> dict[str, str]:
    environment = os.environ.copy()
    for key in base.TORCH_DISTRIBUTED_ENV:
        environment.pop(key, None)
    for key in WORKER_ENV_UNSET:
        environment.pop(key, None)
    environment.update(WORKER_ENV_OVERRIDES)
    environment["CUDA_VISIBLE_DEVICES"] = cuda_id
    return environment


def _cache_marker(model: str) -> Path:
    return CACHE_ROOT / model / "producer_success.json"


def _cache_attempt_count(model: str) -> int:
    root = CACHE_ROOT / model / "attempts"
    return len(list(root.glob("attempt[0-9][0-9][0-9]"))) if root.is_dir() else 0


def _prepare_cache(args: argparse.Namespace) -> int:
    hostname = socket.gethostname()
    if not hostname.startswith("j-") or not hostname.endswith("-master-0"):
        raise base.CampaignError("cache producer must run inside a Canoe debug pod")
    if args.model not in base.MODEL_SLUGS or not 0 <= args.physical_gpu <= 7:
        raise base.CampaignError("invalid model/GPU")
    plan = base._read_json(PLAN_PATH)
    _verify_plan(plan)
    marker = _cache_marker(args.model)
    if marker.is_file():
        current = base._read_json(marker)
        if current.get("status") != "succeeded" or current.get("model") != args.model:
            raise base.CampaignError(f"invalid cache marker: {marker}")
        print(marker)
        return 0
    model_root = CACHE_ROOT / args.model
    model_root.mkdir(parents=True, exist_ok=True)
    model_lock = model_root / ".producer.lock"
    gpu_lock = LOCK_ROOT / hostname / f"gpu{args.physical_gpu}.lock"
    gpu_lock.parent.mkdir(parents=True, exist_ok=True)
    with model_lock.open("a+", encoding="utf-8") as model_handle:
        fcntl.flock(model_handle.fileno(), fcntl.LOCK_EX)
        if marker.is_file():
            print(marker)
            return 0
        index = _cache_attempt_count(args.model) + 1
        if index > 3:
            raise base.CampaignError(f"cache retry budget exhausted: {args.model}")
        attempt = model_root / "attempts" / f"attempt{index:03d}"
        attempt.mkdir(parents=True, exist_ok=False)
        command = _cache_command(plan, args.model, attempt)
        reference = plan["calibration_and_reference_contracts"][args.model][
            "reference_logits"
        ]
        reference_path = Path(reference["path"])
        before = reference_path.stat()
        manifest = {
            "campaign_id": CAMPAIGN_ID,
            "protocol_fingerprint": plan["protocol_fingerprint"],
            "stage": "deterministic-sdpa-static-cache-producer",
            "model": args.model,
            "hostname": hostname,
            "physical_gpu": args.physical_gpu,
            "command": command,
            "command_sha256": base._canonical_sha256(command),
            "environment": {"set": dict(sorted(WORKER_ENV_OVERRIDES.items())), "unset": list(WORKER_ENV_UNSET)},
            "gpu": base._gpu_inventory(str(args.physical_gpu)),
            "started_at": base._utc_now(),
        }
        base._atomic_json(attempt / "manifest.json", manifest)
        log_path = attempt / "execution.log"
        started = time.monotonic()
        with gpu_lock.open("a+", encoding="utf-8") as gpu_handle:
            fcntl.flock(gpu_handle.fileno(), fcntl.LOCK_EX)
            with log_path.open("xb") as handle:
                completed = subprocess.run(
                    command,
                    cwd=base.REPO_ROOT,
                    env=_worker_environment(str(args.physical_gpu)),
                    stdin=subprocess.DEVNULL,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
        after = reference_path.stat()
        result: dict[str, Any] = {
            **manifest,
            "status": "failed",
            "returncode": completed.returncode,
            "elapsed_seconds": time.monotonic() - started,
            "finished_at": base._utc_now(),
            "log": {"path": str(log_path), "sha256": base._file_sha256(log_path)},
        }
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise base.CampaignError(f"baseline reference cache changed: {args.model}")
        if completed.returncode == 0:
            files = sorted(path for path in _static_root(args.model).glob("*.pt") if path.is_file())
            if len(files) != 1 or files[0].stat().st_size <= 0:
                raise base.CampaignError(f"SDPA producer returned zero without one static cache: {args.model}")
            result.update(
                status="succeeded",
                static_cache={
                    "path": str(files[0]),
                    "size_bytes": files[0].stat().st_size,
                    "mtime_ns": files[0].stat().st_mtime_ns,
                },
                baseline_reference_unchanged=True,
                token_contract=plan["calibration_and_reference_contracts"][args.model]["tokens"],
            )
        base._atomic_json(attempt / "result.json", result)
        if completed.returncode != 0:
            return 1
        base._atomic_json(marker, result)
        print(marker)
        return 0


def _status(_: argparse.Namespace) -> int:
    rows = []
    for model in base.MODEL_SLUGS:
        marker = _cache_marker(model)
        rows.append(
            {
                "model": model,
                "status": "succeeded" if marker.is_file() else "pending",
                "attempts": _cache_attempt_count(model),
            }
        )
    print(json.dumps({"campaign_id": CAMPAIGN_ID, "cache_producers": rows}, indent=2, sort_keys=True))
    return 0


def _bootstrap() -> None:
    base.CAMPAIGN_ID = CAMPAIGN_ID
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.PLAN_PATH = PLAN_PATH
    base.WORKER_ENV_OVERRIDES = dict(WORKER_ENV_OVERRIDES)
    base.WORKER_ENV_UNSET = WORKER_ENV_UNSET
    base.CODE_INPUTS = CODE_INPUTS
    base._validate_full_profile = _validate_full_profile
    base._build_plan = _build_plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init")
    init.set_defaults(handler=_write_plan)
    producer = subparsers.add_parser("prepare-cache")
    producer.add_argument("--model", required=True, choices=base.MODEL_SLUGS)
    producer.add_argument("--physical-gpu", required=True, type=int)
    producer.set_defaults(handler=_prepare_cache)
    status = subparsers.add_parser("status")
    status.set_defaults(handler=_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _bootstrap()
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception:
        traceback.print_exc()
        return 1


_bootstrap()


if __name__ == "__main__":
    raise SystemExit(main())
