#!/usr/bin/env python3
"""Formal checkpoint stage for the frozen deterministic-FA4 V6 selection.

This adapter reuses the audited V3 checkpoint lifecycle while rebinding every
campaign, selection, output, environment, and code-fingerprint dependency to
the optimized run15 campaign.  It remains dormant until all 40 V6 learning
rates have passed the selector gates and ``selection_v6_run15 freeze`` has
published ``selections.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as c
from experiments.realq_fullmodel_retune_20260817 import campaign_v6_run15 as v6
from experiments.realq_fullmodel_retune_20260817 import selection_v6_run15 as select

v6._bootstrap()
_V6_BINDING_NAMES = (
    "CAMPAIGN_ID",
    "OUTPUT_ROOT",
    "PLAN_PATH",
    "INITIAL_LRS",
    "_a_loss_ratio_for",
    "_a_loss_ratio_text",
    "WORKER_ENV_OVERRIDES",
    "WORKER_ENV_UNSET",
    "_validate_full_profile",
    "CODE_INPUTS",
    "_build_plan",
)
_V6_BINDINGS = {name: getattr(c, name) for name in _V6_BINDING_NAMES}

from experiments.realq_fullmodel_retune_20260817 import formal_v3 as core


FORMAL_ID = "realq-fullmodel-two-branch-formal-20260820-v6-run15"
FORMAL_PLAN_PATH = v6.OUTPUT_ROOT / "formal_plan.json"
FORMAL_AUDIT_PATH = v6.OUTPUT_ROOT / "formal_final_audit.json"
_IMPLEMENTATION_PATH = Path(core.__file__).resolve()
_BASE_BUILD_PLAN = core._build_plan


def _balanced_pairs() -> list[tuple[str, str]]:
    return v6.v2._balanced_pairs()


def _flags(command: Sequence[str]) -> dict[str, str]:
    if len(command) < 3 or command[1] != "-m":
        raise c.CampaignError("formal command must use python -m")
    if (len(command) - 3) % 2:
        raise c.CampaignError("formal command is not strict flag/value form")
    output: dict[str, str] = {}
    for index in range(3, len(command), 2):
        flag, value = str(command[index]), str(command[index + 1])
        if not flag.startswith("--") or flag in output:
            raise c.CampaignError(f"invalid or duplicate formal flag: {flag}")
        output[flag] = value
    return output


def _worker_env(gpu: str) -> dict[str, str]:
    # V6 deliberately unsets the legacy math-SDPA/TF32 overrides and keeps
    # deterministic FA4 plus scoped Fisher/Hessian TF32.
    _restore_v6_bindings()
    return c._worker_environment(gpu)


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
    # Formal checkpoint production must retain the same reference-cache gate
    # as tuning; changing it to false would no longer be the frozen run15
    # numerical profile even though evaluation itself is skipped.
    c._set_arg(command, "--require_reference_cache_hit", "true")
    c._set_arg(command, "--output_dir", str(output))
    c._set_arg(command, "--exp", "formal_v6_run15")
    c._set_or_append_arg(
        command,
        "--save_qmodel_path",
        str(core._checkpoint_path(branch, config)),
    )
    c._validate_full_profile(command, branch=branch, config=config)
    return command


def _code_snapshot() -> dict[str, Any]:
    adapter = Path(__file__).resolve()
    dependencies = (
        adapter,
        _IMPLEMENTATION_PATH,
        Path(v6.__file__).resolve(),
        Path(select.__file__).resolve(),
    )
    files = [
        {"path": str(path), "sha256": c._file_sha256(path)}
        for path in dependencies
    ]
    return {"files": files, "sha256": c._canonical_sha256(files)}


def _build_plan() -> dict[str, Any]:
    _activate()
    body = _BASE_BUILD_PLAN()
    body.pop("formal_plan_fingerprint", None)
    body["code"] = _code_snapshot()
    protocol = dict(body["protocol"])
    protocol.update(
        {
            "attention_backend": "flash_attention_4==4.0.0b25",
            "fa4_deterministic_argument": True,
            "fisher_hessian_tf32_is_scoped": True,
            "qwen3_32b_only_memory_exception": "hessian_accum_bsz=32",
            "execution_environment": {
                "set": dict(sorted(v6.RUN15_ENV_OVERRIDES.items())),
                "unset": sorted(v6.RUN15_ENV_UNSET),
            },
        }
    )
    body["protocol"] = protocol
    body["formal_plan_fingerprint"] = c._canonical_sha256(body)
    return body


def _restore_v6_bindings() -> None:
    for name, value in _V6_BINDINGS.items():
        setattr(c, name, value)


def _activate() -> None:
    _restore_v6_bindings()
    # Importing the reusable V3 lifecycle imports its selector as a side
    # effect; restore the V6 selector's shared-core bindings explicitly.
    select.core.SELECTION_ID = select.SELECTION_ID
    select.core.SELECTION_PATH = select.SELECTION_PATH
    core.v3 = v6
    core.select = select
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
