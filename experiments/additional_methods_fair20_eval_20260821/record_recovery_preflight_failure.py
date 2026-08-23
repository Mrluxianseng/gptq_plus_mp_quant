#!/usr/bin/env python3
"""Close the one known pre-loader TurboBOA recovery attempt transparently."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from experiments.additional_methods_fair20_eval_20260821 import common
from experiments.additional_methods_fair20_eval_20260821 import (
    recover_turboboa_llama as recovery,
)


EVAL_ID = "turboboa__TB20-L8-W4A4"
ERROR = "No package metadata was found for evalplus"


def main() -> int:
    suite = common.OUTPUT_ROOT / "evals" / EVAL_ID
    attempt = suite / "contract_recovery" / "attempt001"
    manifest = common.read_object(attempt / "manifest.json")
    result_path = attempt / "result.json"
    if result_path.is_file():
        print(json.dumps(common.read_object(result_path), indent=2, sort_keys=True))
        return 0
    queue_log = (
        common.OUTPUT_ROOT
        / "turboboa_llama_contract_recovery"
        / "node0_gpu3_queue.log"
    )
    log = queue_log.read_text(encoding="utf-8", errors="replace")
    if ERROR not in log or manifest.get("eval_id") != EVAL_ID:
        raise RuntimeError("attempt is not the known EvalPlus metadata preflight failure")
    spec = recovery._find_spec(EVAL_ID)
    checkpoint = Path(common.resolve_completed_artifact(spec)["path"]).resolve()
    before = manifest["checkpoint_identity_before"]
    after = recovery._file_identity(checkpoint)
    if after != before:
        raise RuntimeError("checkpoint identity changed after the preflight failure")
    finished = dt.datetime.fromtimestamp(
        queue_log.stat().st_mtime, tz=dt.timezone.utc
    ).isoformat()
    result = {
        "schema_version": 1,
        "status": "failed_runtime_preflight",
        "eval_id": EVAL_ID,
        "returncode": 1,
        "failure_stage": "run_suite_runtime_metadata_before_model_loader",
        "error_type": "importlib.metadata.PackageNotFoundError",
        "error": ERROR,
        "checkpoint_loaded": False,
        "cuda_context_created": False,
        "checkpoint_modified": False,
        "checkpoint_identity_before": before,
        "checkpoint_identity_after": after,
        "started_at": manifest["started_at"],
        "finished_at": finished,
        "queue_log": str(queue_log.resolve()),
        "queue_log_sha256": common.sha256_file(queue_log),
        "recovery_action": (
            "explicit_evalplus_runtime_preflight_and_dead_claim_resume"
        ),
    }
    common.atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
