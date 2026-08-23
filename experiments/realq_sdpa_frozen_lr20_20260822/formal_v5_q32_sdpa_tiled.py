#!/usr/bin/env python3
"""Qwen3-32B formal runs using the existing exact SDPA memory path."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import re
import shutil
import socket
import time
import traceback
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as base
from experiments.realq_fullmodel_retune_20260817 import deterministic_sdpa_memory
from experiments.realq_fullmodel_retune_20260817 import formal_v3 as core
from experiments.realq_fullmodel_retune_20260817 import q32_memory_entry
from experiments.realq_sdpa_frozen_lr20_20260822 import campaign_v2_memory as campaign
from experiments.realq_sdpa_frozen_lr20_20260822 import formal_v2 as source_formal
from experiments.realq_sdpa_frozen_lr20_20260822 import formal_v3_q32_haccum16 as h16
from experiments.realq_sdpa_frozen_lr20_20260822 import formal_v4_q32_haccum8 as h8


CAMPAIGN_ID = "realq-sdpa-frozen-run15-lr20-20260822-v5-q32-sdpa-tiled"
FORMAL_ID = "realq-sdpa-frozen-run15-lr20-formal-20260822-v5-q32-sdpa-tiled"
OUTPUT_ROOT = base.REPO_ROOT.parent / "experiment_data" / (
    "realq_sdpa_frozen_lr20_20260822_v5_q32_sdpa_tiled"
)
FORMAL_PLAN_PATH = OUTPUT_ROOT / "formal_plan.json"
FORMAL_AUDIT_PATH = OUTPUT_ROOT / "formal_final_audit.json"
ENTRY_MODULE = "experiments.realq_fullmodel_retune_20260817.q32_memory_entry"
TERMINAL_FAILURE = "controller_terminal_failure.json"
OUTER_CLAIM = ".q32_sdpa_tiled_controller_claim"
OOM_RE = re.compile(r"CUDA out of memory|OutOfMemoryError", re.IGNORECASE)


def _pairs() -> list[tuple[str, str]]:
    return h16._pairs()


_V5_BINDING = SimpleNamespace(
    OUTPUT_ROOT=OUTPUT_ROOT,
    PLAN_PATH=FORMAL_PLAN_PATH,
    v2=SimpleNamespace(_balanced_pairs=_pairs),
)


def _flags(command: Sequence[str]) -> dict[str, str]:
    return source_formal._flags(command)


def _formal_dir(branch: str, config: str) -> Path:
    return OUTPUT_ROOT / "formal" / branch / config


def _checkpoint_path(branch: str, config: str) -> Path:
    return _formal_dir(branch, config) / "checkpoint" / "quantized.pt"


def _source_v2b_plan() -> dict[str, Any]:
    return h16._source_formal_plan()


def _failed_haccum_evidence() -> list[dict[str, Any]]:
    specs = (
        (h16.OUTPUT_ROOT, 16),
        (h8.OUTPUT_ROOT, 8),
    )
    rows = []
    for root, haccum in specs:
        path = root / "formal/full_block/qwen3-32b_w4a16/attempt001/result.json"
        result = base._read_json(path)
        log = Path(result["log"]["path"])
        tail = log.read_bytes()[-4 * 1024 * 1024 :].decode(errors="replace")
        if (
            result.get("status") != "failed"
            or result.get("failure_class") != "oom"
            or _flags(result["command"]).get("--hessian_accum_bsz") != str(haccum)
            or "Tried to allocate 32.00 GiB" not in tail
            or not OOM_RE.search(tail)
        ):
            raise base.CampaignError(f"invalid haccum={haccum} OOM evidence")
        rows.append(
            {
                "result": {"path": str(path), "sha256": base._file_sha256(path)},
                "log": {"path": str(log), "sha256": base._file_sha256(log)},
                "hessian_accum_bsz": haccum,
                "failed_allocation": "32.00 GiB math-SDPA attention",
            }
        )
    return rows


def _configure_command(
    source: Sequence[str], *, branch: str, config: str, output: Path
) -> list[str]:
    command = list(source)
    if len(command) < 3 or command[1:3] != ["-m", "realq.ptq"]:
        raise base.CampaignError(f"unexpected source entry point: {command[:3]}")
    command[2] = ENTRY_MODULE
    base._set_arg(command, "--save_qmodel_path", str(_checkpoint_path(branch, config)))
    base._set_arg(command, "--output_dir", str(output))
    base._set_arg(command, "--exp", "formal_sdpa_frozen_run15_lr20_q32_tiled")
    return command


def _validate_command(
    source: Sequence[str], command: Sequence[str], *, branch: str, config: str
) -> None:
    if (
        list(source[:2]) != list(command[:2])
        or source[2] != "realq.ptq"
        or command[2] != ENTRY_MODULE
    ):
        raise base.CampaignError(f"Q32 tiled entry mismatch: {branch}/{config}")
    before, after = _flags(source), _flags(command)
    changed = {
        flag for flag in set(before) | set(after) if before.get(flag) != after.get(flag)
    }
    if changed != {"--save_qmodel_path", "--output_dir", "--exp"}:
        raise base.CampaignError(f"unexpected Q32 tiled delta {branch}/{config}: {changed}")
    expected = {
        "--hessian_accum_bsz": "32",
        "--backward_samples": "32",
        "--backward_bsz": "32",
        "--final_layer_backward_bsz": "32",
        "--global_loss_bsz": "1",
        "--attention_backend": "sdpa",
        "--seed": "1",
        "--rotation_seed": "0",
        "--refresh_seed": "0",
        "--nsamples": "256",
        "--seq_len": "2048",
        "--w_groupsize": "128",
        "--w_asym": "false",
        "--a_asym": "false",
        "--k_asym": "false",
        "--v_asym": "false",
        "--act_order": "true",
        "--rotate": "true",
        "--require_static_cache_hit": "true",
        "--require_reference_cache_hit": "true",
    }
    for flag, wanted in expected.items():
        if after.get(flag) != wanted:
            raise base.CampaignError(
                f"Q32 tiled protocol drift {branch}/{config} {flag}: {after.get(flag)}"
            )
    contract = campaign.v1._baseline_cache_contracts()["qwen3-32b"]
    if sorted(Path(after["--tokens_cache_path"]).glob("*.pt")) != [
        Path(contract["tokens"]["path"])
    ]:
        raise base.CampaignError("Q32 tiled token cache differs from baseline")
    if after["--cache_dir"] != contract["reference_logits"]["runtime_root"]:
        raise base.CampaignError("Q32 tiled reference cache differs from baseline")
    marker = base._read_json(campaign._cache_marker("qwen3-32b"))
    if Path(after["--static_cache_path"]) != Path(marker["static_cache"]["path"]).parent:
        raise base.CampaignError("Q32 tiled static cache differs from V2 producer")


def _code_snapshot() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        Path(core.__file__).resolve(),
        Path(source_formal.__file__).resolve(),
        Path(deterministic_sdpa_memory.__file__).resolve(),
        Path(q32_memory_entry.__file__).resolve(),
    )
    files = [{"path": str(path), "sha256": base._file_sha256(path)} for path in paths]
    return {"files": files, "sha256": base._canonical_sha256(files)}


def _build_plan() -> dict[str, Any]:
    source = _source_v2b_plan()
    source_rows = {
        f"{row['branch']}/{row['config']}": row for row in source["rows"]
    }
    rows = []
    for branch, config in _pairs():
        key = f"{branch}/{config}"
        source_row = source_rows[key]
        output = _formal_dir(branch, config) / "attempt001" / "realq_output"
        command = _configure_command(
            source_row["command"], branch=branch, config=config, output=output
        )
        _validate_command(source_row["command"], command, branch=branch, config=config)
        rows.append(
            {
                "branch": branch,
                "config": config,
                "selected_lr": source_row["selected_lr"],
                "selection_evidence": {
                    "branch": branch,
                    "config": config,
                    "selected_lr": source_row["selected_lr"],
                    "source_v2b_command_sha256": source_row["command_sha256"],
                },
                "command": command,
                "command_sha256": base._canonical_sha256(command),
                "checkpoint": str(_checkpoint_path(branch, config)),
                "source_v2b_row": source_row,
            }
        )
    body: dict[str, Any] = {
        "formal_id": FORMAL_ID,
        "campaign_id": CAMPAIGN_ID,
        "source_v2b_formal_plan": {
            "path": str(source_formal.FORMAL_PLAN_PATH),
            "sha256": base._file_sha256(source_formal.FORMAL_PLAN_PATH),
            "fingerprint": source["formal_plan_fingerprint"],
        },
        "failed_hessian_accum_canaries": _failed_haccum_evidence(),
        "code": _code_snapshot(),
        "protocol": {
            "world_size": 1,
            "checkpoint_producers": 8,
            "models": ["qwen3-32b"],
            "only_command_delta_from_v2b": "dedicated exact SDPA memory entry",
            "logical_backward_samples": 32,
            "logical_backward_bsz": 32,
            "final_layer_backward_bsz": 32,
            "hessian_accum_bsz": 32,
            "math_sdpa_batch_tile": deterministic_sdpa_memory.DEFAULT_SDPA_BATCH_TILE,
            "checkpoint": "non-reentrant; preserve_rng_state=true",
            "loss_and_adam_cadence_changed": False,
            "learning_rates_retuned": False,
            "exact_gptaq_guidedquant_token_files": True,
            "exact_gptaq_guidedquant_reference_cache_files": True,
            "exact_v2_deterministic_sdpa_static_cache": True,
            "all_numerical_flags_unchanged": True,
        },
        "rows": rows,
    }
    body["formal_plan_fingerprint"] = base._canonical_sha256(body)
    return body


def _formal_command(
    plan: Mapping[str, Any], selection_row: Mapping[str, Any], output: Path
) -> list[str]:
    branch, config = str(selection_row["branch"]), str(selection_row["config"])
    row = {f"{item['branch']}/{item['config']}": item for item in plan["rows"]}[
        f"{branch}/{config}"
    ]
    command = _configure_command(
        row["source_v2b_row"]["command"],
        branch=branch,
        config=config,
        output=output,
    )
    _validate_command(
        row["source_v2b_row"]["command"], command, branch=branch, config=config
    )
    return command


def _worker_env(gpu: str) -> dict[str, str]:
    return source_formal._worker_env(gpu)


def _activate() -> None:
    campaign._bootstrap()
    source_formal._restore_campaign_bindings()
    core.v3 = _V5_BINDING
    core.FORMAL_ID = FORMAL_ID
    core.FORMAL_PLAN_PATH = FORMAL_PLAN_PATH
    core.FORMAL_AUDIT_PATH = FORMAL_AUDIT_PATH
    core._build_plan = _build_plan
    core._formal_command = _formal_command
    core._worker_env = _worker_env
    core._balanced_pairs = _pairs
    core._flags = _flags


def _claim_one(hostname: str, gpu: int) -> tuple[str, str, Path] | None:
    for branch, config in _pairs():
        root = _formal_dir(branch, config)
        if (root / "formal_success.json").is_file() or (root / TERMINAL_FAILURE).is_file():
            continue
        root.mkdir(parents=True, exist_ok=True)
        claim = root / OUTER_CLAIM
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        base._atomic_json(
            claim / "owner.json",
            {
                "formal_id": FORMAL_ID,
                "branch": branch,
                "config": config,
                "hostname": hostname,
                "physical_gpu": gpu,
                "claimed_at": base._utc_now(),
            },
        )
        return branch, config, claim
    return None


def _state() -> dict[str, int]:
    counts = {"succeeded": 0, "claimed": 0, "failed": 0, "pending": 0}
    for branch, config in _pairs():
        root = _formal_dir(branch, config)
        if (root / "formal_success.json").is_file():
            counts["succeeded"] += 1
        elif (root / OUTER_CLAIM).is_dir():
            counts["claimed"] += 1
        elif (root / TERMINAL_FAILURE).is_file():
            counts["failed"] += 1
        else:
            counts["pending"] += 1
    return counts


def _run_worker(args: argparse.Namespace) -> int:
    _activate()
    hostname = socket.gethostname()
    if not hostname.startswith("j-") or not hostname.endswith("-master-0"):
        raise base.CampaignError("worker must run inside a Canoe debug pod")
    if not 0 <= args.physical_gpu <= 7:
        raise base.CampaignError("physical GPU must be in [0, 7]")
    plan = core._load_plan()
    lock = campaign.LOCK_ROOT / hostname / f"gpu{args.physical_gpu}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    while True:
        state = _state()
        if state["succeeded"] == 8:
            return 0
        if state["failed"]:
            return 1
        claimed = _claim_one(hostname, args.physical_gpu)
        if claimed is None:
            time.sleep(20)
            continue
        branch, config, claim = claimed
        try:
            with lock.open("a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                status = core._run_one(plan, branch, config, str(args.physical_gpu))
            if status:
                paths = sorted(_formal_dir(branch, config).glob("attempt*/result.json"))
                result = base._read_json(paths[-1]) if paths else {}
                base._atomic_json(
                    _formal_dir(branch, config) / TERMINAL_FAILURE,
                    {
                        "formal_id": FORMAL_ID,
                        "branch": branch,
                        "config": config,
                        "status": "failed",
                        "failure_class": result.get("failure_class", "unknown"),
                        "result": str(paths[-1]) if paths else None,
                        "hostname": hostname,
                        "physical_gpu": args.physical_gpu,
                        "failed_at": base._utc_now(),
                    },
                )
        finally:
            shutil.rmtree(claim, ignore_errors=True)


def _status(_: argparse.Namespace) -> int:
    _activate()
    print(json.dumps({"formal_id": FORMAL_ID, "formal": _state()}, indent=2))
    return 0


def _write_plan(_: argparse.Namespace) -> int:
    _activate()
    value = _build_plan()
    if FORMAL_PLAN_PATH.is_file():
        if base._read_json(FORMAL_PLAN_PATH) != value:
            raise base.CampaignError("existing Q32 tiled plan differs")
    else:
        base._atomic_json(FORMAL_PLAN_PATH, value)
    print(FORMAL_PLAN_PATH)
    print(value["formal_plan_fingerprint"])
    return 0


def _audit(_: argparse.Namespace) -> int:
    _activate()
    return int(core._audit_all(SimpleNamespace()))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    write = subparsers.add_parser("write-plan")
    write.set_defaults(handler=_write_plan)
    worker = subparsers.add_parser("run-worker")
    worker.add_argument("--physical-gpu", required=True, type=int)
    worker.set_defaults(handler=_run_worker)
    status = subparsers.add_parser("status")
    status.set_defaults(handler=_status)
    audit = subparsers.add_parser("audit")
    audit.set_defaults(handler=_audit)
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
