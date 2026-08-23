#!/usr/bin/env python3
"""Run the repository's restricted official EvalPlus scorer for one suite."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from . import common


VALID_STATUSES = frozenset({"pass", "fail", "timeout"})
PARALLEL = 32


def _spec(eval_id: str) -> common.EvalSpec:
    matches = [spec for spec in common.iter_specs() if spec.eval_id == eval_id]
    if len(matches) != 1:
        raise common.EvaluationError(f"unknown eval_id: {eval_id}")
    return matches[0]


def execute(eval_id: str) -> dict:
    spec = _spec(eval_id)
    reference_manifest = common.load_reference_manifest()
    suite_dir = common.OUTPUT_ROOT / "evals" / spec.eval_id
    suite_path = suite_dir / "suite_success.json"
    suite = common.read_object(suite_path)
    if (
        suite.get("eval_id") != spec.eval_id
        or suite.get("status")
        != "generation_succeeded_official_humaneval_pending"
    ):
        raise common.EvaluationError("suite is not ready for official scoring")
    declared = suite.get("reasoning", {}).get("humaneval_plus", {})
    manifest_path = Path(declared.get("manifest", "")).resolve()
    if common.sha256_file(manifest_path) != declared.get("manifest_sha256"):
        raise common.EvaluationError("HumanEval+ generation manifest changed")

    samples = (
        suite_dir
        / "reasoning/humaneval_plus/humaneval_plus/evalplus_samples.jsonl"
    )
    if not samples.is_file():
        raise common.EvaluationError(f"HumanEval+ samples missing: {samples}")
    rows = [
        json.loads(line)
        for line in samples.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dataset = (
        common.REPO_ROOT
        / "datasets/reasoning_eval/humaneval_plus/HumanEvalPlus.jsonl.gz"
    )
    with gzip.open(dataset, "rt", encoding="utf-8") as handle:
        expected_ids = {
            json.loads(line)["task_id"] for line in handle if line.strip()
        }
    sample_ids = [row.get("task_id") for row in rows]
    if (
        len(rows) != 164
        or len(set(sample_ids)) != 164
        or set(sample_ids) != expected_ids
        or any(not isinstance(row.get("solution"), str) for row in rows)
    ):
        raise common.EvaluationError("HumanEval+ generated sample gate failed")

    output = suite_dir / "reasoning/humaneval_plus/official_eval"
    success = output / "official_success.json"
    if success.is_file():
        payload = common.read_object(success)
        if (
            payload.get("eval_id") != spec.eval_id
            or payload.get("status") != "official_scored"
            or payload.get("samples_sha256") != common.sha256_file(samples)
            or payload.get("reference_manifest_fingerprint")
            != reference_manifest["fingerprint"]
        ):
            raise common.EvaluationError("official score identity mismatch")
        return payload
    if success.exists() or success.is_symlink():
        raise common.EvaluationError(f"invalid official success path: {success}")
    output.mkdir(parents=True, exist_ok=True)
    attempts_root = output / "attempts"
    attempt_index = (
        sum(
            path.is_dir()
            for path in attempts_root.glob("attempt[0-9][0-9][0-9]")
        )
        + 1
    )
    attempt = attempts_root / f"attempt{attempt_index:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    before = samples.stat()
    time.sleep(5)
    after = samples.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise common.EvaluationError("HumanEval+ samples are not stable")

    plans = common.load_quant_plans()
    python = plans["efficientqat"]["venv_python"]
    started = time.monotonic()
    log_path = attempt / "execution.log"
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            [
                "bash",
                str(common.REPO_ROOT / "tools/lowbit_activation_evalplus_canoe.sh"),
                str(samples),
                str(attempt),
                str(PARALLEL),
            ],
            cwd=common.REPO_ROOT,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "REALQ_PYTHON": python,
            },
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise common.EvaluationError(
            f"official EvalPlus exited {completed.returncode}: {log_path}"
        )
    result_path = attempt / "evalplus_samples_eval_results.json"
    result = common.read_object(result_path)
    evaluations = result.get("eval")
    if not isinstance(evaluations, dict) or set(evaluations) != expected_ids:
        raise common.EvaluationError("official EvalPlus coverage mismatch")
    base_pass = 0
    plus_pass = 0
    for task_id, candidates in evaluations.items():
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise common.EvaluationError(
                f"EvalPlus candidate count mismatch: {task_id}"
            )
        candidate = candidates[0]
        if (
            candidate.get("base_status") not in VALID_STATUSES
            or candidate.get("plus_status") not in VALID_STATUSES
        ):
            raise common.EvaluationError(
                f"invalid EvalPlus status for {task_id}"
            )
        base_ok = candidate["base_status"] == "pass"
        base_pass += int(base_ok)
        plus_pass += int(base_ok and candidate["plus_status"] == "pass")
    payload = {
        "schema_version": 1,
        "status": "official_scored",
        "eval_id": spec.eval_id,
        "method": spec.method,
        "model": spec.model,
        "setting": spec.setting,
        "samples": str(samples),
        "samples_sha256": common.sha256_file(samples),
        "official_result": str(result_path),
        "official_result_sha256": common.sha256_file(result_path),
        "attempt": attempt_index,
        "base_pass": base_pass,
        "base_total": 164,
        "base_pass_at_1": base_pass / 164,
        "plus_pass": plus_pass,
        "plus_total": 164,
        "plus_pass_at_1": plus_pass / 164,
        "parallel": PARALLEL,
        "wall_seconds": time.monotonic() - started,
        "reference_manifest_fingerprint": reference_manifest["fingerprint"],
        "finished_at": common.now(),
    }
    common.atomic_json(success, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-id", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(execute(args.eval_id), indent=2, sort_keys=True))
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
