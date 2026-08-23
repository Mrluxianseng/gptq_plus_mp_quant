#!/usr/bin/env python3
"""Fail-closed, CPU-only recovery for a completed MATH-500 generation set."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any

from . import common
from realq_benchmark.benchmarks.data import load_examples
from realq_benchmark.benchmarks.generation import (
    generation_config_sha256,
    generation_kwargs,
)
from realq_benchmark.benchmarks.scoring import score_or_export


KNOWN_ERROR = (
    "RuntimeError: MATH-500 scoring requires "
    "`math-verify[antlr4-13-2]==0.9.0`."
)
EXPECTED_VERSIONS = {
    "evalplus": "0.3.1",
    "math-verify": "0.9.0",
    "latex2sympy2_extended": "1.11.0",
    "antlr4-python3-runtime": "4.13.2",
    "transformers": "4.56.2",
    "huggingface_hub": "0.36.2",
    "torch": "2.9.1",
}


class RecoveryError(RuntimeError):
    """A scorer-only recovery precondition failed."""


def _sha256(path: Path) -> str:
    return common.sha256_file(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise RecoveryError(f"blank generation row at line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RecoveryError(
                    f"generation row {line_number} is not a JSON object"
                )
            rows.append(value)
    return rows


def _chunk_seed(base_seed: int, task: str, chunk_index: int) -> int:
    digest = hashlib.sha256(
        f"{base_seed}:{task}:{chunk_index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _generation_cfg(manifest: dict[str, Any]) -> SimpleNamespace:
    generation = manifest["generation"]
    quantization = manifest["quantization"]
    return SimpleNamespace(
        model=manifest["model"],
        load_qmodel_path=manifest["load_qmodel_path"],
        reasoning_max_new_tokens=generation["max_new_tokens"],
        reasoning_do_sample=generation["do_sample"],
        reasoning_temperature=generation["temperature"],
        reasoning_top_p=generation["top_p"],
        reasoning_top_k=generation["top_k"],
        reasoning_seed=generation["seed"],
        reasoning_batch_size=generation["batch_size"],
        reasoning_num_samples=generation["num_samples"],
        reasoning_protocol=generation["protocol"],
        **quantization,
    )


def _validate_manifest(manifest: dict[str, Any]) -> None:
    generation = manifest.get("generation", {})
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "failed"
        or manifest.get("error") != KNOWN_ERROR
        or manifest.get("tasks") != ["math_500"]
        or manifest.get("results") != []
        or manifest.get("versions", {}).get("math-verify") is not None
        or generation.get("batch_size") != 16
        or generation.get("max_new_tokens") != 2048
        or generation.get("num_samples") != 1
        or generation.get("limit") != -1
        or generation.get("protocol") != "realq_zero_shot_v1"
        or generation.get("seed") != 1234
        or generation.get("resume") is not True
        or generation.get("do_sample") is not False
        or generation.get("temperature") != 0.0
        or generation.get("top_p") != 1.0
        or generation.get("top_k") != 0
        or generation.get("apply_chat_template") is not True
        or generation.get("enable_thinking") is not True
    ):
        raise RecoveryError("failed MATH-500 manifest does not match the campaign")


def _validate_packages(isolate: Path) -> dict[str, dict[str, str]]:
    lexical_isolate = Path(os.path.abspath(isolate))
    records: dict[str, dict[str, str]] = {}
    for distribution, expected in EXPECTED_VERSIONS.items():
        actual = importlib.metadata.version(distribution)
        if actual != expected:
            raise RecoveryError(
                f"{distribution} version changed: {actual!r} != {expected!r}"
            )
        records[distribution] = {"version": actual}

    import antlr4
    import evalplus
    import huggingface_hub
    import latex2sympy2_extended
    import math_verify

    modules = {
        "evalplus": evalplus,
        "math-verify": math_verify,
        "latex2sympy2_extended": latex2sympy2_extended,
        "antlr4-python3-runtime": antlr4,
    }
    for distribution, module in modules.items():
        lexical_path = Path(os.path.abspath(str(module.__file__)))
        if not lexical_path.is_relative_to(lexical_isolate):
            raise RecoveryError(
                f"{distribution} was not imported from the isolated overlay: "
                f"{lexical_path}"
            )
        records[distribution]["module_path"] = str(lexical_path)
        records[distribution]["module_target"] = str(lexical_path.resolve())
    hub_path = Path(os.path.abspath(str(huggingface_hub.__file__)))
    if hub_path.is_relative_to(lexical_isolate):
        raise RecoveryError("isolated overlay unexpectedly shadows huggingface_hub")
    records["huggingface_hub"]["module_path"] = str(hub_path)
    return records


def _validate_inputs(
    *,
    suite_dir: Path,
    expected_eval_id: str,
    expected_manifest_sha256: str,
    expected_generations_sha256: str,
) -> tuple[dict[str, Any], list[Any], list[dict[str, Any]], dict[str, Any]]:
    if suite_dir.name != expected_eval_id:
        raise RecoveryError("suite directory and expected eval ID differ")
    if (suite_dir / "suite_success.json").exists():
        raise RecoveryError("suite is already complete")
    manifest_path = suite_dir / "reasoning/math_500/manifest.json"
    generations_path = (
        suite_dir / "reasoning/math_500/math_500/generations.jsonl"
    )
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or not generations_path.is_file()
        or generations_path.is_symlink()
    ):
        raise RecoveryError("MATH-500 manifest or generations are not regular files")
    if _sha256(manifest_path) != expected_manifest_sha256:
        raise RecoveryError("failed manifest SHA256 changed")
    if _sha256(generations_path) != expected_generations_sha256:
        raise RecoveryError("generation SHA256 changed")

    manifest = common.read_object(manifest_path)
    _validate_manifest(manifest)
    dataset_record = manifest.get("datasets", {}).get("math_500")
    if not isinstance(dataset_record, dict):
        raise RecoveryError("MATH-500 dataset record is missing")
    dataset_path = Path(str(dataset_record.get("path", ""))).resolve()
    if (
        not dataset_path.is_file()
        or dataset_record.get("sha256") != _sha256(dataset_path)
        or dataset_record.get("selected_examples") != 500
    ):
        raise RecoveryError("MATH-500 dataset identity or coverage changed")
    examples, reloaded_source = load_examples(
        "math_500",
        data_dir=str(dataset_path.parent.parent),
        lcb_release="release_v6",
        limit=-1,
    )
    if (
        len(examples) != 500
        or Path(reloaded_source["path"]).resolve() != dataset_path
        or reloaded_source["sha256"] != dataset_record["sha256"]
    ):
        raise RecoveryError("reloaded MATH-500 dataset differs from the manifest")

    rows = _read_jsonl(generations_path)
    cfg = _generation_cfg(manifest)
    fingerprint = generation_config_sha256(
        "math_500", cfg, generation_kwargs(cfg)
    )
    if len(rows) != 500:
        raise RecoveryError(f"expected 500 generations, found {len(rows)}")
    seen: set[tuple[str, str, int]] = set()
    for index, (row, example) in enumerate(zip(rows, examples, strict=True)):
        key = (
            str(row.get("task")),
            str(row.get("sample_id")),
            int(row.get("sample_index", -1)),
        )
        expected_chunk = index // 16
        if (
            row.get("schema_version") != 1
            or key != ("math_500", example.sample_id, 0)
            or key in seen
            or row.get("generation_config_sha256") != fingerprint
            or row.get("chunk_index") != expected_chunk
            or row.get("chunk_seed")
            != _chunk_seed(1234, "math_500", expected_chunk)
            or not isinstance(row.get("prompt_sha256"), str)
            or len(row["prompt_sha256"]) != 64
            or not isinstance(row.get("output"), str)
        ):
            raise RecoveryError(f"generation row {index} failed identity checks")
        seen.add(key)
    return manifest, examples, rows, dataset_record


def recover(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RecoveryError("CUDA_VISIBLE_DEVICES must be exactly -1")
    import torch

    if torch.cuda.is_initialized():
        raise RecoveryError("CUDA was initialized before scorer-only recovery")
    suite_dir = Path(args.suite_dir).resolve()
    isolate = Path(args.isolated_package_dir).resolve()
    package_records = _validate_packages(isolate)

    claim = suite_dir / ".claim"
    try:
        claim.mkdir()
    except FileExistsError as exc:
        raise RecoveryError("suite is currently claimed by another worker") from exc
    common.atomic_json(
        claim / "owner.json",
        {
            "eval_id": args.expected_eval_id,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "kind": "math500_scorer_only_recovery",
            "claimed_at": _now(),
        },
    )
    try:
        manifest, examples, rows, dataset_record = _validate_inputs(
            suite_dir=suite_dir,
            expected_eval_id=args.expected_eval_id,
            expected_manifest_sha256=args.expected_failed_manifest_sha256,
            expected_generations_sha256=args.expected_generations_sha256,
        )
        task_root = suite_dir / "reasoning/math_500"
        task_dir = task_root / "math_500"
        scores_path = task_dir / "scores.json"
        recovery_dir = task_dir / "score_recovery_20260821"
        if scores_path.exists() or recovery_dir.exists():
            raise RecoveryError("recovery output already exists")

        started_at = _now()
        started = time.monotonic()
        temporary = Path(
            tempfile.mkdtemp(prefix=".score-recovery.", dir=task_dir)
        )
        renamed = False
        try:
            os.link(task_root / "manifest.json", temporary / "failed_manifest.json")
            scoring_dir = temporary / "scoring"
            summary = score_or_export("math_500", examples, rows, scoring_dir)
            scoring_seconds = time.monotonic() - started
            if (
                summary.get("status") != "scored"
                or summary.get("num_examples") != 500
                or summary.get("num_generations") != 500
            ):
                raise RecoveryError("canonical scorer returned incomplete coverage")
            if _sha256(task_dir / "generations.jsonl") != args.expected_generations_sha256:
                raise RecoveryError("generations changed while scoring")
            if torch.cuda.is_initialized():
                raise RecoveryError("CUDA was initialized during scorer-only recovery")

            finished_at = _now()
            recovery = {
                "schema_version": 1,
                "kind": "math500_scorer_only_recovery",
                "mode": "reuse_exact_complete_generations_no_model_no_cuda",
                "previous_status": "failed",
                "previous_error": KNOWN_ERROR,
                "failed_manifest_sha256": args.expected_failed_manifest_sha256,
                "generations_sha256": args.expected_generations_sha256,
                "generation_reused_without_modification": True,
                "dataset": dataset_record,
                "packages": package_records,
                "isolated_package_dir": str(isolate),
                "recovery_source": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": _sha256(Path(__file__).resolve()),
                },
                "model_loaded": False,
                "cuda_used": False,
                "started_at": started_at,
                "finished_at": finished_at,
                "scoring_seconds": scoring_seconds,
                "receipt": str(recovery_dir / "receipt.json"),
            }
            completed = dict(manifest)
            completed.pop("error", None)
            completed["status"] = "completed"
            completed["versions"] = dict(manifest["versions"])
            completed["versions"]["math-verify"] = "0.9.0"
            completed_summary = dict(summary)
            completed_summary["elapsed_seconds"] = scoring_seconds
            completed["results"] = [completed_summary]
            completed["elapsed_seconds"] = (
                float(manifest["elapsed_seconds"]) + scoring_seconds
            )
            completed["recovery"] = recovery
            common.atomic_json(temporary / "completed_manifest.json", completed)
            completed_sha = _sha256(temporary / "completed_manifest.json")
            receipt = {
                "schema_version": 1,
                "status": "prepared",
                "kind": "math500_scorer_only_recovery_receipt",
                "eval_id": args.expected_eval_id,
                "recovery": recovery,
                "summary": summary,
                "scores_sha256": _sha256(scoring_dir / "scores.json"),
                "completed_manifest_sha256": completed_sha,
                "created_at": finished_at,
            }
            common.atomic_json(temporary / "receipt.json", receipt)
            os.replace(temporary, recovery_dir)
            renamed = True

            os.link(recovery_dir / "scoring/scores.json", scores_path)
            manifest_temporary = task_root / ".manifest.score-recovery.tmp"
            os.link(recovery_dir / "completed_manifest.json", manifest_temporary)
            os.replace(manifest_temporary, task_root / "manifest.json")

            if (
                _sha256(scores_path) != receipt["scores_sha256"]
                or _sha256(task_root / "manifest.json") != completed_sha
                or _sha256(task_dir / "generations.jsonl")
                != args.expected_generations_sha256
            ):
                raise RecoveryError("published recovery identities differ")
            receipt["status"] = "committed"
            receipt["committed_at"] = _now()
            common.atomic_json(recovery_dir / "receipt.json", receipt)
            return receipt
        finally:
            if not renamed:
                shutil.rmtree(temporary, ignore_errors=True)
    finally:
        owner = claim / "owner.json"
        owner.unlink(missing_ok=True)
        claim.rmdir()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--expected-eval-id", required=True)
    parser.add_argument("--expected-failed-manifest-sha256", required=True)
    parser.add_argument("--expected-generations-sha256", required=True)
    parser.add_argument("--isolated-package-dir", required=True)
    args = parser.parse_args(argv)
    try:
        result = recover(args)
    except BaseException as exc:
        print(f"MATH-500 scorer-only recovery failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
