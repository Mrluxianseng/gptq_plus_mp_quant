#!/usr/bin/env python3
"""Archive the GPU4 post-REALQ attempts that lacked EvalPlus metadata.

The REALQ lane supervisor successfully returned GPU4 to the unified worker,
but its inherited ``PYTHONPATH`` omitted the evaluator's audited
``evalplus==0.3.1`` metadata directory.  Ten Qwen3-0.6B suites consequently
used all three retry slots before model loading.  This recovery is deliberately
narrow: every attempt must have failed at the identical runtime metadata gate,
must have produced no quality/suite artifact, and must belong to node0/GPU4.
The immutable attempts are moved to a dedicated superseded audit tree so the
normal worker can retry them with the correct runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

from experiments.additional_methods_fair20_eval_20260821 import common


EVAL_IDS = (
    "turboboa__TB20-Q06-W4A4",
    "yaqa_wclip__YQ-Q06-W4A4",
    "efficientqat__EQ15-Q06-W4",
    "turboboa__TB20-Q06-W4",
    "yaqa_wclip__YQ-Q06-W4",
    "efficientqat__EQ15-Q06-W3",
    "turboboa__TB20-Q06-W3",
    "yaqa_wclip__YQ-Q06-W3",
    "turboboa__TB20-Q06-W2",
    "yaqa_wclip__YQ-Q06-W2",
)
HOSTNAME = "j-4mj21jb084-master-0"
PHYSICAL_GPU = 4
ERROR_MARKER = "PackageNotFoundError: No package metadata was found for evalplus"
DESTINATION = (
    common.OUTPUT_ROOT
    / "_superseded_missing_evalplus_pythonpath_20260822_post_realq_handoff"
)


class RecoveryError(RuntimeError):
    pass


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit() -> list[dict[str, Any]]:
    if DESTINATION.exists() or DESTINATION.is_symlink():
        raise RecoveryError(f"destination already exists: {DESTINATION}")
    rows = []
    for eval_id in EVAL_IDS:
        suite = common.OUTPUT_ROOT / "evals" / eval_id
        if any((suite / name).exists() for name in ("quality_result.json", "suite_success.json")):
            raise RecoveryError(f"scientific result already exists: {eval_id}")
        if (suite / ".claim").exists() or (suite / ".claim").is_symlink():
            raise RecoveryError(f"claim still exists: {eval_id}")
        attempts = suite / "attempts"
        children = sorted(attempts.glob("attempt[0-9][0-9][0-9]"))
        if [path.name for path in children] != [
            "attempt001",
            "attempt002",
            "attempt003",
        ]:
            raise RecoveryError(f"unexpected attempts for {eval_id}: {children}")
        attempt_rows = []
        for index, directory in enumerate(children, start=1):
            expected_names = {"manifest.json", "execution.log", "result.json"}
            actual_names = {path.name for path in directory.iterdir()}
            if actual_names != expected_names:
                raise RecoveryError(
                    f"unexpected attempt contents for {eval_id}: {actual_names}"
                )
            manifest = _read(directory / "manifest.json")
            result = _read(directory / "result.json")
            log = (directory / "execution.log").read_text(
                encoding="utf-8", errors="replace"
            )
            expected = {
                "attempt": index,
                "eval_id": eval_id,
                "hostname": HOSTNAME,
                "physical_gpu": PHYSICAL_GPU,
            }
            for key, value in expected.items():
                if manifest.get(key) != value or result.get(key) != value:
                    raise RecoveryError(
                        f"attempt identity mismatch for {eval_id} {directory.name} {key}"
                    )
            if result.get("status") != "failed" or result.get("returncode") != 1:
                raise RecoveryError(f"attempt is not the expected failure: {directory}")
            if ERROR_MARKER not in log:
                raise RecoveryError(f"unexpected failure cause: {directory}")
            attempt_rows.append(
                {
                    "attempt": index,
                    "manifest_sha256": common.sha256_file(directory / "manifest.json"),
                    "execution_log_sha256": common.sha256_file(
                        directory / "execution.log"
                    ),
                    "result_sha256": common.sha256_file(directory / "result.json"),
                    "started_at": result["started_at"],
                    "finished_at": result["finished_at"],
                }
            )
        rows.append(
            {
                "eval_id": eval_id,
                "source": str(attempts),
                "destination": str(DESTINATION / "evals" / eval_id / "attempts"),
                "attempts": attempt_rows,
            }
        )
    return rows


def execute() -> dict[str, Any]:
    rows = audit()
    for row in rows:
        source = Path(row["source"])
        destination = Path(row["destination"])
        destination.parent.mkdir(parents=True, exist_ok=False)
        shutil.move(str(source), str(destination))
    receipt = {
        "schema_version": 1,
        "status": "succeeded",
        "reason": "post-REALQ evaluator inherited PYTHONPATH without audited evalplus metadata",
        "error_marker": ERROR_MARKER,
        "hostname": HOSTNAME,
        "physical_gpu": PHYSICAL_GPU,
        "moved_eval_count": len(rows),
        "moved_attempt_count": sum(len(row["attempts"]) for row in rows),
        "rows": rows,
        "finished_at": common.now(),
    }
    common.atomic_json(DESTINATION / "recovery_receipt.json", receipt)
    for eval_id in EVAL_IDS:
        if (common.OUTPUT_ROOT / "evals" / eval_id / "attempts").exists():
            raise RecoveryError(f"attempt reset did not complete: {eval_id}")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    result = execute() if args.apply else {"status": "audited", "rows": audit()}
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
