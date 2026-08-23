#!/usr/bin/env python3
"""Run one frozen evaluation suite on an otherwise dependency-idle GPU lane.

The persistent evaluator deliberately waits for every quantization dependency
assigned to its lane.  Some YAQA stages have cross-lane Hessian dependencies,
which can leave a physical GPU idle even though completed checkpoints are ready
for evaluation.  This orchestration-only worker holds the same physical lock,
runs exactly one suite through the frozen worker implementation, and exits.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
from pathlib import Path
import socket
import traceback

from . import common
from . import worker


def _next_quality_ready(
    specs: tuple[common.EvalSpec, ...],
) -> tuple[common.EvalSpec, Path] | None:
    for spec in specs:
        directory = worker._suite_dir(spec)
        if (
            (directory / "suite_success.json").is_file()
            or (directory / "quality_result.json").is_file()
            or worker._attempt_count(directory) >= worker.MAX_ATTEMPTS
        ):
            continue
        try:
            common.resolve_completed_artifact(spec)
        except common.EvaluationError:
            continue
        claim = worker._claim(spec)
        if claim is not None:
            return spec, claim
    return None


def _run_quality_only(
    spec: common.EvalSpec,
    reference_manifest: dict[str, object],
) -> dict[str, object]:
    import torch

    from . import run_suite

    runtime = run_suite._configure_runtime(seed=1234)
    plans = common.load_quant_plans()
    model_info = common.model_contract(spec.model, plans)
    artifact = common.resolve_completed_artifact(spec)
    directory = worker._suite_dir(spec)
    directory.mkdir(parents=True, exist_ok=True)
    analyzer, quality = run_suite._run_quality(
        spec,
        artifact,
        model_info,
        directory,
        reference_record=dict(reference_manifest["references"][spec.model]),
    )
    del analyzer
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "schema_version": 1,
        "status": "quality_succeeded",
        "eval_id": spec.eval_id,
        "runtime": runtime,
        "quality": str(directory / "quality_result.json"),
        "quality_sha256": common.sha256_file(directory / "quality_result.json"),
        "metrics": quality["metrics"],
    }


def execute(
    physical_gpu: int,
    *,
    check_only: bool,
    quality_only: bool,
) -> dict[str, object]:
    hostname = socket.gethostname()
    if not hostname.startswith("j-") or not hostname.endswith("-master-0"):
        raise common.EvaluationError("opportunistic worker must run on a Canoe pod")

    plans = common.load_quant_plans()
    reference_manifest = common.load_reference_manifest()
    specs = common.iter_specs(plans)
    dependencies = worker._lane_quant_dependencies(hostname, physical_gpu, plans)
    if not dependencies:
        raise common.EvaluationError(
            f"no quantization lane exists for {hostname} GPU{physical_gpu}"
        )

    ready_ids: list[str] = []
    for spec in specs:
        directory = worker._suite_dir(spec)
        if (directory / "suite_success.json").is_file():
            continue
        if worker._attempt_count(directory) >= worker.MAX_ATTEMPTS:
            continue
        try:
            common.resolve_completed_artifact(spec)
        except common.EvaluationError:
            continue
        ready_ids.append(spec.eval_id)

    if check_only:
        return {
            "schema_version": 1,
            "status": "check_succeeded",
            "hostname": hostname,
            "physical_gpu": physical_gpu,
            "lane_dependencies_complete": all(path.is_file() for path in dependencies),
            "ready_eval_ids": ready_ids,
            "ready_quality_ids": [
                eval_id
                for eval_id in ready_ids
                if not (
                    common.OUTPUT_ROOT / "evals" / eval_id / "quality_result.json"
                ).is_file()
            ],
            "reference_fingerprint": reference_manifest["fingerprint"],
        }

    lock = worker.LOCK_ROOT / hostname / f"gpu{physical_gpu}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        failure = worker._lane_failure(hostname, physical_gpu, plans)
        if failure is not None:
            raise common.EvaluationError(
                f"quantization lane failed; refusing evaluation: {failure}"
            )
        if all(path.is_file() for path in dependencies):
            return {
                "schema_version": 1,
                "status": "skipped_lane_ready_for_persistent_worker",
                "hostname": hostname,
                "physical_gpu": physical_gpu,
            }

        ready = (
            _next_quality_ready(specs)
            if quality_only
            else worker._next_ready(specs)
        )
        if ready is None:
            return {
                "schema_version": 1,
                "status": "skipped_no_ready_suite",
                "hostname": hostname,
                "physical_gpu": physical_gpu,
            }

        spec, claim = ready
        print(
            f"{common.now()} opportunistic_eval_start eval_id={spec.eval_id} "
            f"gpu={physical_gpu}",
            flush=True,
        )
        try:
            if quality_only:
                quality = _run_quality_only(spec, reference_manifest)
                succeeded = True
            else:
                quality = None
                succeeded = worker._run_one(
                    spec,
                    physical_gpu,
                    reference_manifest,
                )
        finally:
            worker._release_claim(claim)
        result = {
            "schema_version": 1,
            "status": "succeeded" if succeeded else "failed",
            "hostname": hostname,
            "physical_gpu": physical_gpu,
            "eval_id": spec.eval_id,
            "mode": "quality_only" if quality_only else "full_suite",
        }
        if quality is not None:
            result["quality"] = quality
        print(json.dumps(result, sort_keys=True), flush=True)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--quality-only", action="store_true")
    args = parser.parse_args()
    try:
        result = execute(
            args.physical_gpu,
            check_only=args.check_only,
            quality_only=args.quality_only,
        )
        if args.check_only:
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0 if result["status"] != "failed" else 1
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
