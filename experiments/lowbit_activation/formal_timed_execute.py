#!/usr/bin/env python3
"""Run one formal low-bit command through the unchanged reviewed executor.

This entrypoint accepts exactly the base ``lowbit_activation_execute.py`` CLI.
It reuses the base parser, renderer, attempt identity, execution wrapper, and
manifest implementation.  Applicable formal commands receive child-only
algorithm-core timing instrumentation; BF16 passes through unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import json
import socket
import sys
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
EXPERIMENT_DIR = SCRIPT_PATH.parent
for directory in (str(TOOLS_DIR), str(EXPERIMENT_DIR)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

import lowbit_activation_execute as base_executor  # noqa: E402
import lowbit_activation_runner as base_runner  # noqa: E402

from formal_timing_adapter import (  # noqa: E402
    TimingAdapterError,
    child_environment,
    finalize_execution,
    instrument_rendered,
    prepare_formal_rendered,
    spec_from_executor_args,
)

AUTHORIZED_AUXILIARY_JOBS = {
    "j-8j1en3m0aq",
    "j-ryis586loy",
    "j-nxfty1rcqx",
    "j-28b993voy4",
    "j-hzunwre5gq",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_execution_plan(plan_path: Path) -> dict[str, Any]:
    """Load either the canonical plan or an exact job-only formal derivative."""

    try:
        return base_executor._load_validated_plan_once(plan_path)
    except base_runner.PlanError as original_error:
        try:
            derived = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise original_error
        marker = derived.get("formal_auxiliary_job")
        if not isinstance(marker, dict):
            raise original_error

        job_id = marker.get("job_id")
        preflight = marker.get("runtime_preflight_manifest")
        if (
            marker.get("schema_version") != 1
            or job_id not in AUTHORIZED_AUXILIARY_JOBS
            or not isinstance(preflight, str)
            or preflight
            != f"output/lowbit_activation/runtime_preflight_{job_id}.json"
        ):
            raise TimingAdapterError(
                "formal auxiliary plan authorization is invalid"
            )
        hostname = socket.gethostname()
        if not hostname.startswith(f"{job_id}-"):
            raise TimingAdapterError(
                f"formal auxiliary plan for {job_id!r} cannot run on "
                f"hostname {hostname!r}"
            )

        base_path = EXPERIMENT_DIR / "plan.json"
        base_plan = base_executor._load_validated_plan_once(base_path)
        base_sha = _sha256_file(base_path.resolve(strict=True))
        if marker.get("base_plan_sha256") != base_sha:
            raise TimingAdapterError(
                "formal auxiliary plan base-plan identity drifted"
            )
        expected = copy.deepcopy(base_plan)
        expected["resolutions"]["canoe_job_id"] = job_id
        expected["resolutions"]["runtime_preflight_manifest"] = preflight
        expected["formal_auxiliary_job"] = dict(marker)
        if derived != expected:
            raise TimingAdapterError(
                "formal auxiliary plan differs from the canonical plan "
                "outside the authorized job/preflight fields"
            )
        return derived


def main(argv: list[str] | None = None) -> int:
    args = base_executor._build_cli().parse_args(argv)
    try:
        plan = _load_execution_plan(args.plan)
        rendered = base_executor._render_from_args(args, plan)
        rendered = base_executor._with_attempt_identity(
            rendered, args.attempt_index
        )
        rendered = prepare_formal_rendered(args, rendered)
        spec = spec_from_executor_args(args, rendered)
        if spec is None:
            result = base_executor.run_rendered(
                rendered,
                plan=plan,
                plan_path=args.plan,
                cwd=REPO_ROOT,
                execute=args.execute,
            )
        else:
            rendered, provenance = instrument_rendered(
                rendered,
                spec,
                repo_root=REPO_ROOT,
            )
            with child_environment(provenance):
                result = base_executor.run_rendered(
                    rendered,
                    plan=plan,
                    plan_path=args.plan,
                    cwd=REPO_ROOT,
                    execute=args.execute,
                )
            if args.execute:
                result = finalize_execution(
                    result,
                    rendered=rendered,
                    spec=spec,
                    provenance=provenance,
                    plan=plan,
                    repo_root=REPO_ROOT,
                )
    except (
        TimingAdapterError,
        base_executor.ExecutionError,
        OSError,
        json.JSONDecodeError,
        base_runner.PlanError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not args.execute:
        print(json.dumps(result.manifest, indent=2, ensure_ascii=False))
        return 0

    print(
        json.dumps(
            {
                "run_id": result.manifest["run_id"],
                "status": result.manifest["status"],
                "exit_code": result.exit_code,
                "manifest": str(result.manifest_path),
                "log": str(result.log_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return int(result.exit_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
