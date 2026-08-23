#!/usr/bin/env python3
"""Fail-closed completion audit for the frozen 20-run REAL-Q campaign."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.realq_20group_20260808 import campaign as c


FROZEN_PLAN_FINGERPRINT = (
    "e187159ad237ac40d06b45aa3271b91ed104112d495d944d7374fdec2732f538"
)
HEX_SHA256_RE = re.compile(r"[0-9a-f]{64}")
GENERATION_MANIFEST_FIELDS = (
    "reasoning_batch_size",
    "reasoning_limit",
    "reasoning_max_new_tokens",
    "reasoning_num_samples",
    "reasoning_apply_chat_template",
    "reasoning_enable_thinking",
    "reasoning_do_sample",
    "reasoning_temperature",
    "reasoning_top_p",
    "reasoning_top_k",
    "reasoning_seed",
    "reasoning_resume",
    "reasoning_protocol",
    "reasoning_system_prompt",
    "reasoning_lcb_release",
    "reasoning_lcb_source_dir",
)
QUANTIZATION_MANIFEST_FIELDS = (
    "w_bits",
    "w_groupsize",
    "a_bits",
    "a_groupsize",
    "k_bits",
    "k_groupsize",
    "v_bits",
    "v_groupsize",
    "rotate",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise c.CampaignError(message)


def same_json(left: Any, right: Any) -> bool:
    """Compare JSON-compatible values while treating tuples as arrays."""

    return c.canonical_sha256(left) == c.canonical_sha256(right)


def same_number(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    try:
        first = float(left)
        second = float(right)
    except (TypeError, ValueError):
        return False
    return math.isfinite(first) and math.isfinite(second) and math.isclose(
        first, second, rel_tol=0.0, abs_tol=tolerance
    )


def read_object(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"required JSON file is missing: {path}")
    try:
        return c.read_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise c.CampaignError(f"invalid required JSON file {path}: {exc}") from exc


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(path.is_file(), f"JSONL file is missing: {path}")
    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise c.CampaignError(
                    f"invalid JSONL record {path}:{line_number}: {exc}"
                ) from exc
            require(
                isinstance(row, dict),
                f"JSONL record must be an object: {path}:{line_number}",
            )
            rows.append(row)
    stat = path.stat()
    return rows, {
        "path": str(path),
        "rows": len(rows),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def valid_evalplus_candidate(candidates: Any) -> bool:
    """Accept every terminal status emitted by the frozen EvalPlus 0.3.1."""

    return (
        isinstance(candidates, list)
        and len(candidates) == 1
        and isinstance(candidates[0], dict)
        and candidates[0].get("base_status") in c.EVALPLUS_CANDIDATE_STATUSES
        and candidates[0].get("plus_status") in c.EVALPLUS_CANDIDATE_STATUSES
    )


def tail_contains(path: Path, needle: bytes) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as handle:
        with contextlib.suppress(OSError):
            handle.seek(-min(path.stat().st_size, 4 * 1024 * 1024), os.SEEK_END)
        return needle in handle.read()


def audit_plan(root: Path) -> dict[str, Any]:
    plan_path = root / "plan.json"
    plan = read_object(plan_path)
    fingerprint = plan.get("fingerprint")
    identity = dict(plan)
    identity.pop("fingerprint", None)
    require(
        fingerprint == c.canonical_sha256(identity),
        f"plan fingerprint does not match its payload: {plan_path}",
    )
    require(
        fingerprint == FROZEN_PLAN_FINGERPRINT,
        f"plan is not the frozen v5 identity: {fingerprint}",
    )
    require(
        plan.get("campaign_id") == c.CAMPAIGN_ID,
        f"plan campaign mismatch: {plan_path}",
    )
    require(
        same_json(plan.get("tasks"), c.TASKS),
        f"plan task protocol mismatch: {plan_path}",
    )
    require(
        same_json(
            plan.get("formal_global_loss_ladder"), c.FORMAL_GLOBAL_LOSS_LADDER
        ),
        f"formal OOM ladder mismatch: {plan_path}",
    )
    require(
        plan.get("hessian_accum_bsz_by_model")
        == {model.slug: c.hessian_accum_bsz(model) for model in c.MODELS},
        f"Hessian accumulation contract mismatch: {plan_path}",
    )

    current_datasets = c.dataset_inventories()
    require(
        same_json(plan.get("dataset_inventories"), current_datasets),
        "reasoning datasets changed after the frozen plan was written",
    )
    current_models = {
        model.slug: c.model_inventory(model) for model in c.MODELS
    }
    require(
        same_json(plan.get("model_inventories"), current_models),
        "model files changed after the frozen plan was written",
    )

    rows = plan.get("runs")
    require(isinstance(rows, list) and len(rows) == len(c.RUNS), "plan must contain 20 runs")
    by_id = {
        row.get("run_id"): row for row in rows if isinstance(row, dict)
    }
    require(set(by_id) == set(c.RUN_BY_ID), "plan run identity set is not exact")
    for run in c.RUNS:
        row = by_id[run.run_id]
        require(row.get("node") == run.node, f"plan node mismatch: {run.run_id}")
        require(
            row.get("model") == dataclasses.asdict(run.model),
            f"plan model mismatch: {run.run_id}",
        )
        require(
            row.get("quant") == dataclasses.asdict(run.quant),
            f"plan quantization mismatch: {run.run_id}",
        )
        expected_cfg = dataclasses.asdict(
            c.validate_args(c.common_args(root, run.model, run.quant, global_loss_bsz=32))
        )
        require(
            same_json(row.get("formal_config"), expected_cfg),
            f"plan resolved config mismatch: {run.run_id}",
        )
    return {
        "path": str(plan_path),
        "fingerprint": fingerprint,
        "datasets": current_datasets,
        "models": current_models,
        "run_count": len(rows),
    }


def audit_tuning(root: Path, run: c.RunSpec) -> dict[str, Any]:
    tuning = c.run_root(root, run) / "tuning"
    state = read_object(tuning / "state.json")
    decision = read_object(tuning / "controlled_decision.json")
    require(state.get("status") == "tuning_complete", f"tuning incomplete: {run.run_id}")
    selected_lr = state.get("selected_lr")
    selected_kl = state.get("selected_kl")
    require(
        same_number(selected_lr, decision.get("selected_lr"))
        and same_number(selected_kl, decision.get("selected_kl")),
        f"tuning decision/state mismatch: {run.run_id}",
    )
    require(
        state.get("protocol_fingerprint") == decision.get("protocol_fingerprint")
        and HEX_SHA256_RE.fullmatch(str(state.get("protocol_fingerprint", "")))
        is not None,
        f"tuning protocol identity mismatch: {run.run_id}",
    )
    best_path = tuning / "best_lr.txt"
    require(best_path.is_file(), f"best LR file is missing: {run.run_id}")
    require(
        same_number(best_path.read_text(encoding="utf-8").strip(), selected_lr),
        f"best LR text disagrees with state: {run.run_id}",
    )

    trials = state.get("trials")
    require(isinstance(trials, dict) and trials, f"tuning trials missing: {run.run_id}")
    succeeded = [
        value
        for value in trials.values()
        if isinstance(value, dict)
        and value.get("status") == "succeeded"
        and value.get("returncode") == 0
        and same_number(value.get("kl"), value.get("kl"))
    ]
    require(succeeded, f"no successful tuning trial: {run.run_id}")
    selected_trials = [
        value
        for value in succeeded
        if same_number(value.get("lr"), selected_lr)
        and same_number(value.get("kl"), selected_kl)
    ]
    require(len(selected_trials) == 1, f"selected trial is not unique: {run.run_id}")
    require(
        same_number(min(float(value["kl"]) for value in succeeded), selected_kl),
        f"selected LR is not the observed global Exact-KL minimum: {run.run_id}",
    )
    return {
        "selected_lr": float(selected_lr),
        "selected_kl": float(selected_kl),
        "protocol_fingerprint": state["protocol_fingerprint"],
        "successful_trials": len(succeeded),
        "trial_launch_count": decision.get("trial_launch_count"),
    }


def audit_formal(
    root: Path, run: c.RunSpec, tuning: Mapping[str, Any]
) -> dict[str, Any]:
    directory = c.run_root(root, run)
    marker_path = directory / "formal_success.json"
    marker = read_object(marker_path)
    require(marker.get("campaign_id") == c.CAMPAIGN_ID, f"formal campaign mismatch: {run.run_id}")
    require(marker.get("run_id") == run.run_id, f"formal run mismatch: {run.run_id}")
    require(marker.get("run") == dataclasses.asdict(run), f"formal run spec mismatch: {run.run_id}")
    require(
        same_number(marker.get("selected_lr"), tuning["selected_lr"])
        and same_number(marker.get("selected_kl"), tuning["selected_kl"]),
        f"formal/tuning selection mismatch: {run.run_id}",
    )
    require(
        marker.get("controlled_decision")
        == read_object(directory / "tuning" / "controlled_decision.json"),
        f"formal marker embeds another tuning decision: {run.run_id}",
    )

    global_loss_bsz = marker.get("formal_global_loss_bsz")
    require(
        global_loss_bsz in c.FORMAL_GLOBAL_LOSS_LADDER,
        f"invalid formal global-loss batch: {run.run_id}",
    )
    precompute = read_object(root / "precompute" / run.model.slug / "formal_success.json")
    producer_bsz = precompute.get("formal_global_loss_bsz")
    require(
        precompute.get("campaign_id") == c.CAMPAIGN_ID
        and precompute.get("profile") == "formal"
        and precompute.get("model") == dataclasses.asdict(run.model)
        and producer_bsz in c.FORMAL_GLOBAL_LOSS_LADDER,
        f"formal precompute marker mismatch: {run.run_id}",
    )
    require(
        c.FORMAL_GLOBAL_LOSS_LADDER.index(global_loss_bsz)
        >= c.FORMAL_GLOBAL_LOSS_LADDER.index(producer_bsz),
        f"formal run increased global-loss batch after precompute: {run.run_id}",
    )

    attempts = marker.get("formal_attempts")
    require(isinstance(attempts, list) and attempts, f"formal attempts missing: {run.run_id}")
    attempt_batches = [attempt.get("global_loss_bsz") for attempt in attempts]
    require(
        all(value in c.FORMAL_GLOBAL_LOSS_LADDER for value in attempt_batches)
        and len(set(attempt_batches)) == len(attempt_batches)
        and [c.FORMAL_GLOBAL_LOSS_LADDER.index(value) for value in attempt_batches]
        == sorted(c.FORMAL_GLOBAL_LOSS_LADDER.index(value) for value in attempt_batches),
        f"formal OOM retry order is invalid: {run.run_id}",
    )
    require(
        attempts[-1].get("global_loss_bsz") == global_loss_bsz
        and attempts[-1].get("returncode") == 0
        and attempts[-1].get("oom") is False,
        f"formal marker has no matching terminal success: {run.run_id}",
    )
    for attempt in attempts[:-1]:
        require(
            attempt.get("returncode") != 0 and attempt.get("oom") is True,
            f"formal retry was not caused by OOM: {run.run_id}",
        )
    for attempt in attempts:
        log = Path(str(attempt.get("log", "")))
        completed = c.completed_logged_run(log)
        require(completed is not None, f"formal log has no terminal footer: {log}")
        require(
            completed[0] == attempt.get("returncode")
            and same_number(completed[1], attempt.get("elapsed_seconds"), tolerance=1e-6),
            f"formal marker/log footer mismatch: {log}",
        )
        require(
            bool(attempt.get("oom")) == bool(completed[0] and c.log_is_oom(log)),
            f"formal OOM classification mismatch: {log}",
        )

    attempt_dir = directory / "formal" / f"glbsz{global_loss_bsz}"
    expected_cfg = dataclasses.asdict(
        c.validate_args(
            c.formal_args(
                root,
                run,
                grad_lr=float(tuning["selected_lr"]),
                global_loss_bsz=int(global_loss_bsz),
                attempt_dir=attempt_dir,
            )
        )
    )
    require(
        same_json(read_object(attempt_dir / "config.json"), expected_cfg),
        f"successful formal config is not frozen: {run.run_id}",
    )
    require(
        expected_cfg["backward_samples"] == 32
        and expected_cfg["backward_bsz"] == 32
        and expected_cfg["final_layer_backward_bsz"] == 32
        and expected_cfg["hessian_accum_bsz"] == c.hessian_accum_bsz(run.model),
        f"formal backward/Hessian contract changed: {run.run_id}",
    )
    require(
        tail_contains(attempt_dir / "execution.log", b"bounded final-KL projection enabled"),
        f"successful formal log lacks the bounded final-KL gate: {run.run_id}",
    )

    checkpoint = directory / "checkpoint" / "quantized.pt"
    require(marker.get("checkpoint") == str(checkpoint), f"checkpoint path mismatch: {run.run_id}")
    require(checkpoint.is_file() and checkpoint.stat().st_size > 0, f"checkpoint missing: {run.run_id}")
    stat = checkpoint.stat()
    require(
        marker.get("checkpoint_stat")
        == {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        f"checkpoint changed after formal success: {run.run_id}",
    )
    return {
        "formal_global_loss_bsz": int(global_loss_bsz),
        "hessian_accum_bsz": int(expected_cfg["hessian_accum_bsz"]),
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "formal_attempts": len(attempts),
    }


def expected_examples(task: str, cfg: Any, cache: dict[str, Any]):
    if task not in cache:
        from realq.benchmarks.data import load_examples

        cache[task] = load_examples(
            task,
            data_dir=cfg.reasoning_data_dir,
            lcb_release=cfg.reasoning_lcb_release,
            limit=cfg.reasoning_limit,
        )
    return cache[task]


def audit_generation_records(
    path: Path,
    task: str,
    cfg: Any,
    examples: Sequence[Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    from realq.benchmarks.generation import (
        generation_config_sha256,
        generation_kwargs,
    )

    records, inventory = read_jsonl(path)
    expected_count = c.TASKS[task][2]
    require(len(records) == expected_count, f"{task} generation count mismatch: {path}")
    expected_ids = [example.sample_id for example in examples]
    require(
        len(expected_ids) == expected_count and len(set(expected_ids)) == expected_count,
        f"{task} source sample identities are not exact",
    )
    config_hash = generation_config_sha256(task, cfg, generation_kwargs(cfg))
    by_id: dict[str, dict[str, Any]] = {}
    keys = set()
    for record in records:
        sample_id = str(record.get("sample_id"))
        key = (record.get("task"), sample_id, record.get("sample_index"))
        require(key not in keys, f"duplicate generation identity: {path}: {key}")
        keys.add(key)
        require(
            record.get("schema_version") == 1
            and record.get("task") == task
            and record.get("sample_index") == 0
            and isinstance(record.get("output"), str)
            and isinstance(record.get("chunk_index"), int)
            and isinstance(record.get("chunk_seed"), int),
            f"invalid generation record: {path}: {key}",
        )
        require(
            HEX_SHA256_RE.fullmatch(str(record.get("prompt_sha256", ""))) is not None
            and record.get("generation_config_sha256") == config_hash,
            f"generation fingerprint mismatch: {path}: {key}",
        )
        by_id[sample_id] = record
    require(set(by_id) == set(expected_ids), f"{task} generation sample set mismatch: {path}")
    inventory["generation_config_sha256"] = config_hash
    return inventory, by_id


def audit_math_scores(
    path: Path,
    task: str,
    examples: Sequence[Any],
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    payload = read_object(path)
    summary = payload.get("summary")
    details = payload.get("details")
    expected = c.TASKS[task][2]
    require(isinstance(summary, dict), f"math score summary missing: {path}")
    require(
        summary.get("task") == task
        and summary.get("status") == "scored"
        and summary.get("num_examples") == expected
        and summary.get("num_generations") == expected,
        f"math score summary mismatch: {path}",
    )
    require(isinstance(details, list) and len(details) == expected, f"score detail count mismatch: {path}")
    detail_by_id: dict[str, dict[str, Any]] = {}
    correct = 0
    for detail in details:
        require(isinstance(detail, dict), f"invalid score detail: {path}")
        sample_id = str(detail.get("sample_id"))
        candidates = detail.get("candidates")
        require(
            sample_id not in detail_by_id
            and sample_id in records
            and detail.get("sample_count") == 1
            and isinstance(candidates, list)
            and len(candidates) == 1
            and candidates[0].get("sample_index") == 0
            and isinstance(candidates[0].get("correct"), bool),
            f"invalid score identity/candidate: {path}: {sample_id}",
        )
        passed = candidates[0]["correct"]
        require(
            detail.get("pass_at_1") is passed
            and detail.get("pass_at_n_oracle") is passed,
            f"score detail is internally inconsistent: {path}: {sample_id}",
        )
        detail_by_id[sample_id] = detail
        correct += int(passed)
    require(
        set(detail_by_id) == {example.sample_id for example in examples},
        f"score detail sample set mismatch: {path}",
    )
    rate = correct / expected
    require(
        same_number(summary.get("pass_at_1"), rate)
        and same_number(summary.get("pass_at_n_oracle"), rate),
        f"score pass rate does not match details: {path}",
    )
    return {
        "status": "scored",
        "pass": correct,
        "total": expected,
        "pass_at_1": rate,
        "scores": str(path),
        "scores_sha256": c.file_sha256(path),
    }


def audit_humaneval_scores(
    root: Path,
    run: c.RunSpec,
    output: Path,
    examples: Sequence[Any],
    generation_success: Path,
) -> dict[str, Any]:
    scores_path = output / "humaneval_plus" / "scores.json"
    scores = read_object(scores_path)
    summary = scores.get("summary")
    expected = c.TASKS["humaneval_plus"][2]
    samples = output / "humaneval_plus" / "evalplus_samples.jsonl"
    sample_rows, sample_inventory = read_jsonl(samples)
    expected_ids = {example.sample_id for example in examples}
    sample_ids = [str(row.get("task_id")) for row in sample_rows]
    require(
        isinstance(summary, dict)
        and summary.get("task") == "humaneval_plus"
        and summary.get("status") == "generated_unscored"
        and summary.get("num_examples") == expected
        and summary.get("num_generations") == expected
        and summary.get("official_input") == str(samples.resolve())
        and len(sample_rows) == expected
        and len(set(sample_ids)) == expected
        and set(sample_ids) == expected_ids
        and all(isinstance(row.get("solution"), str) for row in sample_rows),
        f"HumanEval+ exported samples are incomplete: {run.run_id}",
    )

    official_dir = output / "official_eval"
    success_path = official_dir / "official_success.json"
    success = read_object(success_path)
    result_path = official_dir / "evalplus_samples_eval_results.json"
    require(result_path.is_file(), f"official EvalPlus result is missing: {result_path}")
    require(
        success.get("campaign_id") == c.CAMPAIGN_ID
        and success.get("run_id") == run.run_id
        and success.get("status") == "official_scored"
        and success.get("samples") == str(samples)
        and success.get("samples_sha256") == sample_inventory["sha256"]
        and success.get("samples_bytes") == sample_inventory["size_bytes"]
        and success.get("generation_success") == str(generation_success)
        and success.get("generation_success_sha256") == c.file_sha256(generation_success)
        and success.get("official_result") == str(result_path)
        and success.get("official_result_sha256") == c.file_sha256(result_path)
        and success.get("official_result_bytes") == result_path.stat().st_size
        and success.get("parallel") == c.EVALPLUS_PARALLEL,
        f"official EvalPlus success identity mismatch: {run.run_id}",
    )
    result = read_object(result_path)
    evaluations = result.get("eval")
    require(isinstance(evaluations, dict) and set(evaluations) == expected_ids, f"official EvalPlus coverage mismatch: {run.run_id}")
    base_pass = 0
    plus_pass = 0
    for task_id, candidates in evaluations.items():
        require(
            valid_evalplus_candidate(candidates),
            f"invalid EvalPlus candidate result: {run.run_id}/{task_id}",
        )
        base_ok = candidates[0]["base_status"] == "pass"
        base_pass += int(base_ok)
        plus_pass += int(base_ok and candidates[0]["plus_status"] == "pass")
    require(
        success.get("task_count") == expected
        and success.get("base_pass") == base_pass
        and success.get("base_total") == expected
        and same_number(success.get("base_pass_at_1"), base_pass / expected)
        and success.get("plus_pass") == plus_pass
        and success.get("plus_total") == expected
        and same_number(success.get("plus_pass_at_1"), plus_pass / expected),
        f"official EvalPlus aggregate mismatch: {run.run_id}",
    )
    return {
        "status": "official_scored",
        "base_pass": base_pass,
        "base_total": expected,
        "base_pass_at_1": base_pass / expected,
        "plus_pass": plus_pass,
        "plus_total": expected,
        "plus_pass_at_1": plus_pass / expected,
        "samples": sample_inventory,
        "official_result": str(result_path),
        "official_result_sha256": success["official_result_sha256"],
        "official_success": str(success_path),
    }


def audit_evaluation(
    root: Path,
    run: c.RunSpec,
    task: str,
    example_cache: dict[str, Any],
) -> dict[str, Any]:
    output = c.run_root(root, run) / "reasoning" / task
    success_path = output / "generation_success.json"
    success = read_object(success_path)
    require(
        success.get("campaign_id") == c.CAMPAIGN_ID
        and success.get("run_id") == run.run_id
        and success.get("task") == task
        and success.get("official_code_score_pending")
        is (task == "humaneval_plus"),
        f"generation success marker mismatch: {run.run_id}/{task}",
    )
    expected_cfg = dataclasses.asdict(
        c.validate_args(c.eval_args(root, run, task, output))
    )
    actual_cfg = read_object(output / "config.json")
    require(same_json(actual_cfg, expected_cfg), f"evaluation config changed: {run.run_id}/{task}")
    cfg = c.validate_args(c.eval_args(root, run, task, output))
    examples, source = expected_examples(task, cfg, example_cache)
    require(len(examples) == c.TASKS[task][2], f"evaluation source count mismatch: {task}")

    manifest_path = output / "manifest.json"
    manifest = read_object(manifest_path)
    expected_generation = {
        name.removeprefix("reasoning_"): getattr(cfg, name)
        for name in GENERATION_MANIFEST_FIELDS
    }
    expected_quantization = {
        name: getattr(cfg, name) for name in QUANTIZATION_MANIFEST_FIELDS
    }
    require(
        manifest.get("schema_version") == 1
        and manifest.get("status") == "completed"
        and manifest.get("tasks") == [task]
        and manifest.get("model") == str(cfg.model)
        and manifest.get("load_qmodel_path") == cfg.load_qmodel_path
        and same_json(manifest.get("quantization"), expected_quantization)
        and same_json(manifest.get("generation"), expected_generation)
        and manifest.get("datasets", {}).get(task) == source,
        f"reasoning manifest identity mismatch: {run.run_id}/{task}",
    )
    tokenizer = manifest.get("tokenizer")
    versions = manifest.get("versions")
    require(
        isinstance(tokenizer, dict)
        and isinstance(tokenizer.get("class"), str)
        and HEX_SHA256_RE.fullmatch(str(tokenizer.get("chat_template_sha256", "")))
        is not None,
        f"tokenizer manifest is invalid: {run.run_id}/{task}",
    )
    require(
        isinstance(versions, dict)
        and set(versions)
        == {"torch", "transformers", "lm-eval", "datasets", "math-verify", "evalplus"}
        and all(isinstance(value, str) and value for value in versions.values()),
        f"dependency version manifest is invalid: {run.run_id}/{task}",
    )
    results = manifest.get("results")
    require(isinstance(results, list) and len(results) == 1, f"reasoning manifest result count mismatch: {run.run_id}/{task}")

    generation_path = output / task / "generations.jsonl"
    generation, records = audit_generation_records(
        generation_path, task, cfg, examples
    )
    if task in {"gsm8k", "math_500"}:
        result = audit_math_scores(
            output / task / "scores.json", task, examples, records
        )
        score_summary = read_object(output / task / "scores.json")["summary"]
    else:
        result = audit_humaneval_scores(
            root, run, output, examples, success_path
        )
        score_summary = read_object(output / task / "scores.json")["summary"]
    manifest_summary = dict(results[0])
    elapsed = manifest_summary.pop("elapsed_seconds", None)
    require(
        isinstance(elapsed, (int, float))
        and elapsed >= 0
        and same_json(manifest_summary, score_summary),
        f"manifest/score summary mismatch: {run.run_id}/{task}",
    )
    require(
        c.completed_logged_run(output / "execution.log") is not None
        and c.completed_logged_run(output / "execution.log")[0] == 0,
        f"reasoning log has no successful footer: {run.run_id}/{task}",
    )
    return {
        **result,
        "generation": generation,
        "generation_success": str(success_path),
        "generation_success_sha256": c.file_sha256(success_path),
        "manifest": str(manifest_path),
        "manifest_sha256": c.file_sha256(manifest_path),
        "elapsed_seconds": elapsed,
    }


def audit_campaign(root: Path, *, destination: Path | None = None) -> dict[str, Any]:
    root = c.output_root(root)
    plan = audit_plan(root)
    example_cache: dict[str, Any] = {}
    rows = []
    for run in c.RUNS:
        tuning = audit_tuning(root, run)
        formal = audit_formal(root, run, tuning)
        evaluations = {
            task: audit_evaluation(root, run, task, example_cache)
            for task in c.TASKS
        }
        rows.append(
            {
                "index": run.index + 1,
                "run_id": run.run_id,
                "node": run.node,
                "model": dataclasses.asdict(run.model),
                "quant": dataclasses.asdict(run.quant),
                "tuning": tuning,
                "formal": formal,
                "evaluations": evaluations,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "campaign_id": c.CAMPAIGN_ID,
        "plan_fingerprint": plan["fingerprint"],
        "audited_at": c.now(),
        "counts": {
            "runs": len(rows),
            "formal_success": len(rows),
            "generation_success": len(rows) * len(c.TASKS),
            "official_evalplus_success": len(rows),
        },
        "datasets": plan["datasets"],
        "rows": rows,
    }
    payload["audit_fingerprint"] = c.canonical_sha256(payload)
    output = destination or (root / "final_audit.json")
    c.atomic_json(output, payload)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--root", default=str(c.DEFAULT_ROOT))
    result.add_argument("--output", default=None)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = c.output_root(args.root)
    destination = Path(args.output).expanduser().resolve() if args.output else None
    payload = audit_campaign(root, destination=destination)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "campaign_id": payload["campaign_id"],
                "counts": payload["counts"],
                "audit_fingerprint": payload["audit_fingerprint"],
                "output": str(destination or (root / "final_audit.json")),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except c.CampaignError as exc:
        print(f"campaign audit error: {exc}", file=sys.stderr)
        raise SystemExit(2)
