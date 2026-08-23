#!/usr/bin/env python3
"""Formal checkpoints for the frozen-run15-LR deterministic-SDPA replay."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as c
from experiments.realq_sdpa_frozen_lr20_20260822 import campaign
from experiments.realq_sdpa_frozen_lr20_20260822 import selection

campaign._bootstrap()
_CAMPAIGN_BINDING_NAMES = (
    "CAMPAIGN_ID",
    "OUTPUT_ROOT",
    "PLAN_PATH",
    "WORKER_ENV_OVERRIDES",
    "WORKER_ENV_UNSET",
    "_validate_full_profile",
    "CODE_INPUTS",
    "_build_plan",
)
_CAMPAIGN_BINDINGS = {
    name: getattr(c, name) for name in _CAMPAIGN_BINDING_NAMES
}

from experiments.realq_fullmodel_retune_20260817 import formal_v3 as core  # noqa: E402


FORMAL_ID = "realq-sdpa-frozen-run15-lr20-formal-20260822-v1"
FORMAL_PLAN_PATH = campaign.OUTPUT_ROOT / "formal_plan.json"
FORMAL_AUDIT_PATH = campaign.OUTPUT_ROOT / "formal_final_audit.json"
_IMPLEMENTATION_PATH = Path(core.__file__).resolve()
_BASE_BUILD_PLAN = core._build_plan


def _balanced_pairs() -> list[tuple[str, str]]:
    return campaign.v2._balanced_pairs()


def _flags(command: Sequence[str]) -> dict[str, str]:
    return campaign._flags(command)


def _worker_env(gpu: str) -> dict[str, str]:
    _restore_campaign_bindings()
    return campaign._worker_environment(gpu)


def _formal_command(
    plan: Mapping[str, Any],
    selection_row: Mapping[str, Any],
    output: Path,
) -> list[str]:
    branch = str(selection_row["branch"])
    config = str(selection_row["config"])
    command = list(plan["configurations"][f"{branch}/{config}"]["source_command"])
    c._set_arg(
        command,
        "--grad_lr",
        c._stable_float(float(selection_row["selected_lr"])),
    )
    c._set_arg(command, "--skip_eval", "true")
    c._set_arg(command, "--skip_kl_ppl_eval", "true")
    c._set_arg(command, "--lm_eval", "false")
    c._set_arg(command, "--reasoning_eval", "false")
    c._set_arg(command, "--require_static_cache_hit", "true")
    c._set_arg(command, "--require_reference_cache_hit", "true")
    c._set_arg(command, "--output_dir", str(output))
    c._set_arg(command, "--exp", "formal_sdpa_frozen_run15_lr20")
    c._set_or_append_arg(
        command,
        "--save_qmodel_path",
        str(core._checkpoint_path(branch, config)),
    )
    c._validate_full_profile(command, branch=branch, config=config)
    return command


def _code_snapshot() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        _IMPLEMENTATION_PATH,
        Path(campaign.__file__).resolve(),
        Path(selection.__file__).resolve(),
    )
    files = [{"path": str(path), "sha256": c._file_sha256(path)} for path in paths]
    return {"files": files, "sha256": c._canonical_sha256(files)}


def _build_plan() -> dict[str, Any]:
    _activate()
    body = _BASE_BUILD_PLAN()
    body.pop("formal_plan_fingerprint", None)
    body["code"] = _code_snapshot()
    body["protocol"] = {
        **body["protocol"],
        "attention_backend": "deterministic math-SDPA",
        "frozen_learning_rate_source": "deterministic-FA4 run15 selection",
        "learning_rate_retuned_for_sdpa": False,
        "exact_gptaq_guidedquant_token_files": True,
        "exact_gptaq_guidedquant_reference_cache_files": True,
        "fresh_backend_isolated_static_cache": True,
        "fisher_hessian_tf32_is_scoped": True,
        "qwen3_32b_only_memory_exception": "hessian_accum_bsz=32",
        "execution_environment": {
            "set": dict(sorted(campaign.WORKER_ENV_OVERRIDES.items())),
            "unset": sorted(campaign.WORKER_ENV_UNSET),
        },
    }
    body["formal_plan_fingerprint"] = c._canonical_sha256(body)
    return body


def _restore_campaign_bindings() -> None:
    for name, value in _CAMPAIGN_BINDINGS.items():
        setattr(c, name, value)


def _activate() -> None:
    _restore_campaign_bindings()
    core.v3 = campaign
    core.select = selection
    core.FORMAL_ID = FORMAL_ID
    core.FORMAL_PLAN_PATH = FORMAL_PLAN_PATH
    core.FORMAL_AUDIT_PATH = FORMAL_AUDIT_PATH
    core._build_plan = _build_plan
    core._formal_command = _formal_command
    core._worker_env = _worker_env
    core._balanced_pairs = _balanced_pairs
    core._flags = _flags


_activate()


def __getattr__(name: str) -> Any:
    return getattr(core, name)


def main(argv: Sequence[str] | None = None) -> int:
    _activate()
    return int(core.main(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
