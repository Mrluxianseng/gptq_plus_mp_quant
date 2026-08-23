#!/usr/bin/env python3
"""Deduplicated BF16 baselines for the 20-row REAL-Q comparison matrix.

There are four quantization rows per source model, but the corresponding BF16
model is identical.  This module evaluates each of the five unique source
models once on WikiText2, the paper ten QA tasks, GSM8K, MATH-500, and
HumanEval+, then records an explicit five-to-twenty-row mapping.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gzip
import json
import logging
import math
import os
import shutil
import socket
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.realq_20group_20260808 import campaign as c
from experiments.realq_20group_20260808 import quality as q


MODULE = "experiments.realq_20group_20260808.bf16"
BF16_ID = "realq-20group-deduplicated-bf16-baseline-20260809-v1"
SCHEMA_VERSION = 1
KINDS = ("quality", "gsm8k", "math_500", "humaneval_plus")
CLAIM_NAME = ".bf16.claim"


@dataclass(frozen=True)
class Work:
    model_index: int
    kind_index: int
    node: int

    @property
    def model(self) -> c.ModelSpec:
        return c.MODELS[self.model_index]

    @property
    def kind(self) -> str:
        return KINDS[self.kind_index]

    @property
    def work_id(self) -> str:
        return f"{self.model.slug}_{self.kind}"


WORKS = tuple(
    Work(model_index, kind_index, (model_index + kind_index) % 2)
    for model_index in range(len(c.MODELS))
    for kind_index in range(len(KINDS))
)
WORK_BY_ID = {work.work_id: work for work in WORKS}
MODEL_BY_SLUG = {model.slug: model for model in c.MODELS}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise c.CampaignError(message)


def bf16_root(root: Path) -> Path:
    return root / "bf16_baseline"


def work_dir(root: Path, work: Work) -> Path:
    return bf16_root(root) / "models" / work.model.slug / work.kind


def plan_path(root: Path) -> Path:
    return root / "bf16_plan.json"


def base_args(root: Path, model: c.ModelSpec) -> list[str]:
    cache = c.cache_root(root, model)
    return [
        "--model", c.model_path(model),
        "--dataset", "wikitext2",
        "--eval_datasets", "wikitext2",
        "--seed", "1",
        "--rotation_seed", "0",
        "--refresh_seed", "0",
        "--nsamples", "256",
        "--seq_len", "2048",
        "--eval_seq_len", "2048",
        "--w_bits", "16",
        "--w_groupsize", "-1",
        "--w_asym", "false",
        "--w_clip", "false",
        "--a_bits", "16",
        "--a_groupsize", "-1",
        "--a_asym", "false",
        "--a_clip_ratio", "1.0",
        "--k_bits", "16",
        "--k_groupsize", "-1",
        "--k_asym", "false",
        "--k_clip_ratio", "1.0",
        "--v_bits", "16",
        "--v_groupsize", "-1",
        "--v_asym", "false",
        "--v_clip_ratio", "1.0",
        "--rotate", "false",
        "--kl_topk", "-1",
        "--attention_backend", "sdpa",
        "--cache_dir", str(cache / "runtime"),
        "--tokens_cache_path", str(cache / "tokens"),
        "--static_cache_path", str(cache / "static"),
    ]


def work_args(root: Path, work: Work, output: Path) -> list[str]:
    argv = base_args(root, work.model)
    if work.kind == "quality":
        argv.extend(
            [
                "--skip_eval", "false",
                "--skip_kl_ppl_eval", "false",
                "--lm_eval", "true",
                "--lm_eval_batch_size", str(q.LM_EVAL_BATCH_SIZE),
                "--reasoning_eval", "false",
                "--require_reference_cache_hit", "true",
                "--output_dir", str(output / "pipeline_output"),
                "--exp", "bf16_wikitext2_paperqa",
            ]
        )
    else:
        max_tokens, batch_size, _ = c.TASKS[work.kind]
        argv.extend(
            [
                "--skip_eval", "true",
                "--skip_kl_ppl_eval", "true",
                "--lm_eval", "false",
                "--reasoning_eval", "true",
                "--reasoning_tasks", work.kind,
                "--reasoning_data_dir", str(c.WORKSPACE / "datasets/reasoning_eval"),
                "--reasoning_output_dir", str(output),
                "--reasoning_batch_size", str(batch_size),
                "--reasoning_limit", "-1",
                "--reasoning_max_new_tokens", str(max_tokens),
                "--reasoning_num_samples", "1",
                "--reasoning_apply_chat_template", "true",
                "--reasoning_enable_thinking", "true",
                "--reasoning_do_sample", "false",
                "--reasoning_temperature", "0",
                "--reasoning_top_p", "1",
                "--reasoning_top_k", "0",
                "--reasoning_seed", "1234",
                "--reasoning_resume", "true",
                "--reasoning_protocol", "realq_zero_shot_v1",
                "--reasoning_system_prompt", "You are a careful reasoning assistant. Follow the requested output format exactly.",
                "--require_reference_cache_hit", "false",
                "--output_dir", str(output / "pipeline_output"),
                "--exp", f"bf16_{work.kind}",
            ]
        )
    c.validate_args(argv)
    return argv


def semantic_config(root: Path, work: Work) -> dict[str, Any]:
    cfg = dataclasses.asdict(c.validate_args(work_args(root, work, Path("/bf16/output"))))
    cfg.pop("output_dir", None)
    cfg.pop("reasoning_output_dir", None)
    return cfg


def expected_works_for_node(node: int) -> tuple[Work, ...]:
    require(node in (0, 1), f"node must be 0 or 1, got {node}")
    return tuple(
        sorted(
            (work for work in WORKS if work.node == node),
            key=lambda work: (work.model_index, work.kind_index),
            reverse=True,
        )
    )


def build_plan(root: Path) -> dict[str, Any]:
    quality_plan = q.load_and_validate_plan(root)
    rows = []
    for work in WORKS:
        rows.append(
            {
                "work_id": work.work_id,
                "node": work.node,
                "model_index": work.model_index,
                "kind": work.kind,
                "model": dataclasses.asdict(work.model),
                "model_inventory": c.model_inventory(work.model),
                "semantic_config": semantic_config(root, work),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "bf16_id": BF16_ID,
        "root": str(root),
        "quality_plan_fingerprint": quality_plan["fingerprint"],
        "protocol": {
            "unique_source_models": len(c.MODELS),
            "comparison_rows": len(c.RUNS),
            "deduplication": "one BF16 measurement per source model, mapped to its four quantization rows",
            "wikitext2_ppl": True,
            "paper_qa_tasks": list(q.PAPER_QA_TASKS),
            "reasoning_tasks": list(c.TASKS),
            "reasoning_protocol": "realq_zero_shot_v1",
            "official_humaneval_plus": True,
        },
        "works": rows,
        "comparison_mapping": {run.run_id: run.model.slug for run in c.RUNS},
    }
    payload["fingerprint"] = c.canonical_sha256(payload)
    return payload


def write_plan(root: Path) -> dict[str, Any]:
    expected = build_plan(root)
    path = plan_path(root)
    if path.exists():
        require(c.read_json(path) == expected, f"existing BF16 plan differs: {path}")
    else:
        c.atomic_json(path, expected)
    return expected


def load_plan(root: Path) -> dict[str, Any]:
    plan = c.read_json(plan_path(root))
    fingerprint = plan.get("fingerprint")
    identity = dict(plan)
    identity.pop("fingerprint", None)
    require(fingerprint == c.canonical_sha256(identity), "BF16 plan fingerprint mismatch")
    require(plan == build_plan(root), "BF16 plan no longer matches frozen inputs")
    return plan


def plan_row(plan: Mapping[str, Any], work: Work) -> dict[str, Any]:
    rows = plan.get("works")
    matches = [row for row in rows if isinstance(row, dict) and row.get("work_id") == work.work_id] if isinstance(rows, list) else []
    require(len(matches) == 1, f"BF16 plan row not unique: {work.work_id}")
    return matches[0]


def verify_work(root: Path, plan: Mapping[str, Any], work: Work) -> dict[str, Any]:
    row = plan_row(plan, work)
    require(row.get("model_inventory") == c.model_inventory(work.model), f"BF16 model changed: {work.model.slug}")
    require(row.get("semantic_config") == semantic_config(root, work), f"BF16 config changed: {work.work_id}")
    return row


def _exec_quality(root: Path, work: Work, output: Path) -> None:
    from utils import data_utils, dist_utils, eval_utils, model_utils, memory_utils

    cfg = c.validate_args(work_args(root, work, output))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    analyzer = model_utils.ModelAnalyzer(cfg.model, cfg.seq_len)
    loader = data_utils.get_loaders(
        "wikitext2",
        split="test",
        tokenizer=analyzer.tokenizer,
        seq_len=cfg.eval_seq_len,
        num_samples=cfg.nsamples,
    )
    ref_logits, orig_lm_head = eval_utils.get_ref_logits(cfg, analyzer, "wikitext2", loader)
    eval_utils.kl_ppl_eval(
        cfg,
        analyzer,
        orig_lm_head,
        {"wikitext2": loader},
        {"wikitext2": ref_logits},
    )
    del loader, ref_logits, orig_lm_head
    memory_utils.cleanup_memory()
    dist_utils.distribute_model(analyzer.model)
    eval_utils.qa_eval(analyzer.model, analyzer.tokenizer, cfg.lm_eval_batch_size)


def build_reference_cache(root: Path, model_slug: str) -> None:
    """Produce the BF16 reference-logit cache required by quality workers."""
    from utils import data_utils, eval_utils, model_utils, memory_utils

    model = MODEL_BY_SLUG[model_slug]
    work = WORK_BY_ID[f"{model.slug}_quality"]
    cfg = c.validate_args(work_args(root, work, Path("/bf16/reference-producer")))
    cfg.require_reference_cache_hit = False
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    analyzer = model_utils.ModelAnalyzer(cfg.model, cfg.seq_len)
    loader = data_utils.get_loaders(
        "wikitext2",
        split="test",
        tokenizer=analyzer.tokenizer,
        seq_len=cfg.eval_seq_len,
        num_samples=cfg.nsamples,
    )
    ref_logits, orig_lm_head = eval_utils.get_ref_logits(
        cfg, analyzer, "wikitext2", loader
    )
    logging.info(
        "BF16 reference cache ready: model=%s shape=%s dtype=%s",
        model.slug,
        tuple(ref_logits.shape),
        ref_logits.dtype,
    )
    del loader, ref_logits, orig_lm_head, analyzer
    memory_utils.cleanup_memory()


def _exec_reasoning(root: Path, work: Work, output: Path) -> None:
    from realq.benchmarks import run_reasoning_eval
    from utils import dist_utils, model_utils

    cfg = c.validate_args(work_args(root, work, output))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    analyzer = model_utils.ModelAnalyzer(cfg.model, cfg.seq_len)
    dist_utils.distribute_model(analyzer.model)
    run_reasoning_eval(analyzer.model, analyzer.tokenizer, cfg)


def exec_work(root: Path, work: Work, output: Path) -> None:
    if work.kind == "quality":
        _exec_quality(root, work, output)
    else:
        _exec_reasoning(root, work, output)


def audit_reasoning_output(output: Path, work: Work) -> dict[str, Any]:
    manifest = c.read_json(output / "manifest.json")
    require(manifest.get("status") == "completed" and manifest.get("tasks") == [work.kind], f"incomplete BF16 reasoning manifest: {work.work_id}")
    generations = output / work.kind / "generations.jsonl"
    expected = c.TASKS[work.kind][2]
    count = sum(1 for line in generations.open(encoding="utf-8") if line.strip())
    require(count == expected, f"BF16 generation count mismatch: {work.work_id}")
    scores = c.read_json(output / work.kind / "scores.json")
    summary = scores.get("summary")
    require(isinstance(summary, dict), f"BF16 score summary missing: {work.work_id}")
    if work.kind in {"gsm8k", "math_500"}:
        require(summary.get("status") == "scored" and summary.get("num_examples") == expected and summary.get("num_generations") == expected, f"BF16 math score incomplete: {work.work_id}")
        rate = summary.get("pass_at_1")
        require(isinstance(rate, (int, float)) and math.isfinite(rate) and 0 <= rate <= 1, f"BF16 reasoning accuracy invalid: {work.work_id}")
        return {"pass_at_1": float(rate), "pass": round(float(rate) * expected), "total": expected}
    require(summary.get("status") == "generated_unscored" and summary.get("num_examples") == expected and summary.get("num_generations") == expected, f"BF16 HumanEval+ export incomplete: {work.work_id}")
    return {"generated": expected, "official_pending": True}


def run_evalplus(output: Path, work: Work) -> dict[str, Any]:
    samples = output / "humaneval_plus" / "evalplus_samples.jsonl"
    dataset = c.WORKSPACE / c.DATASET_PATHS["humaneval_plus"]
    official = output / "official_eval"
    official.mkdir(parents=True, exist_ok=True)
    result_path = official / "evalplus_samples_eval_results.json"
    expected = c.TASKS["humaneval_plus"][2]
    with gzip.open(dataset, "rt", encoding="utf-8") as source:
        expected_ids = {json.loads(line)["task_id"] for line in source if line.strip()}
    rows = [json.loads(line) for line in samples.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row.get("task_id") for row in rows]
    require(len(rows) == expected and len(set(ids)) == expected and set(ids) == expected_ids, f"BF16 HumanEval+ sample gate failed: {work.work_id}")
    before = samples.stat()
    time.sleep(c.EVALPLUS_SAMPLE_STABILITY_SECONDS)
    after = samples.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), f"BF16 HumanEval+ samples unstable: {work.work_id}")
    if not result_path.is_file():
        completed = subprocess.run(
            ["bash", str(c.WORKSPACE / "tools/lowbit_activation_evalplus_canoe.sh"), str(samples), str(official), str(c.EVALPLUS_PARALLEL)],
            cwd=c.WORKSPACE,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "REALQ_PYTHON": str(c.PYTHON)},
            stdin=subprocess.DEVNULL,
            check=False,
        )
        require(completed.returncode == 0, f"BF16 EvalPlus failed: {work.work_id}")
    payload = c.read_json(result_path)
    evaluations = payload.get("eval")
    require(isinstance(evaluations, dict) and set(evaluations) == expected_ids, f"BF16 EvalPlus coverage mismatch: {work.work_id}")
    base_pass = plus_pass = 0
    for task_id, candidates in evaluations.items():
        require(isinstance(candidates, list) and len(candidates) == 1, f"BF16 EvalPlus candidate mismatch: {task_id}")
        candidate = candidates[0]
        require(candidate.get("base_status") in c.EVALPLUS_CANDIDATE_STATUSES and candidate.get("plus_status") in c.EVALPLUS_CANDIDATE_STATUSES, f"BF16 EvalPlus status invalid: {task_id}")
        base_ok = candidate["base_status"] == "pass"
        base_pass += int(base_ok)
        plus_pass += int(base_ok and candidate["plus_status"] == "pass")
    return {
        "base_pass": base_pass,
        "base_total": expected,
        "base_pass_at_1": base_pass / expected,
        "plus_pass": plus_pass,
        "plus_total": expected,
        "plus_pass_at_1": plus_pass / expected,
        "official_result": str(result_path),
        "official_result_sha256": c.file_sha256(result_path),
    }


def validate_result_metrics(work: Work, output: Path, log_path: Path) -> dict[str, Any]:
    if work.kind == "quality":
        return q.validate_metrics(log_path)
    metrics = audit_reasoning_output(output, work)
    if work.kind == "humaneval_plus":
        metrics = run_evalplus(output, work)
    return metrics


def acquire_claim(directory: Path, work: Work, gpu: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    claim = directory / CLAIM_NAME
    try:
        claim.mkdir()
    except FileExistsError as exc:
        raise c.CampaignError(f"BF16 writer already claimed: {work.work_id}") from exc
    c.atomic_json(claim / "owner.json", {"work_id": work.work_id, "host": socket.gethostname(), "pid": os.getpid(), "gpu": gpu, "claimed_at": c.now()})
    return claim


def run_worker(root: Path, work: Work, gpu: int) -> None:
    plan = load_plan(root)
    row = verify_work(root, plan, work)
    directory = work_dir(root, work)
    success_path = directory / "bf16_success.json"
    if success_path.is_file():
        audit_one(root, plan, work)
        return
    claim = acquire_claim(directory, work, gpu)
    attempt_index, attempt = q.next_attempt(directory)
    attempt.mkdir()
    output = attempt / "output"
    log_path = attempt / "execution.log"
    result_path = attempt / "bf16_result.json"
    command = [str(c.PYTHON), "-m", MODULE, "_exec", "--root", str(root), "--work-id", work.work_id, "--output", str(output)]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "bf16_id": BF16_ID,
        "work_id": work.work_id,
        "attempt_index": attempt_index,
        "model": dataclasses.asdict(work.model),
        "kind": work.kind,
        "node": work.node,
        "started_at": c.now(),
        "hostname": socket.gethostname(),
        "gpu": q.gpu_snapshot(gpu),
        "plan": {"path": str(plan_path(root)), "fingerprint": plan["fingerprint"], "sha256": c.file_sha256(plan_path(root))},
        "model_inventory": row["model_inventory"],
        "command": command,
    }
    c.atomic_json(attempt / "manifest.json", manifest)
    started = time.monotonic()
    try:
        rc, elapsed = c.run_logged(command, log_path, gpu, reasoning=work.kind != "quality")
        require(rc == 0, f"BF16 subprocess failed with returncode={rc}")
        metrics = validate_result_metrics(work, output, log_path)
        verify_work(root, plan, work)
        result = {
            **manifest,
            "status": "succeeded",
            "finished_at": c.now(),
            "exit_code": 0,
            "evaluation_wall_seconds": elapsed,
            "metrics": metrics,
            "log": {"path": str(log_path), "sha256": c.file_sha256(log_path), "size_bytes": log_path.stat().st_size},
        }
        c.atomic_json(result_path, result)
        c.atomic_json(success_path, {"schema_version": SCHEMA_VERSION, "bf16_id": BF16_ID, "work_id": work.work_id, "completed_at": c.now(), "result": {"path": str(result_path), "sha256": c.file_sha256(result_path)}})
        audit_one(root, plan, work)
    except BaseException as exc:
        c.atomic_json(attempt / "failure.json", {**manifest, "status": "failed", "finished_at": c.now(), "evaluation_wall_seconds": time.monotonic() - started, "error_type": type(exc).__qualname__, "error": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        shutil.rmtree(claim, ignore_errors=True)


def audit_one(root: Path, plan: Mapping[str, Any], work: Work) -> dict[str, Any]:
    row = verify_work(root, plan, work)
    marker = c.read_json(work_dir(root, work) / "bf16_success.json")
    require(marker.get("bf16_id") == BF16_ID and marker.get("work_id") == work.work_id, f"BF16 success identity mismatch: {work.work_id}")
    ref = marker.get("result")
    require(isinstance(ref, dict), f"BF16 result ref invalid: {work.work_id}")
    path = Path(ref.get("path", ""))
    require(path.is_file() and c.file_sha256(path) == ref.get("sha256"), f"BF16 result hash mismatch: {work.work_id}")
    result = c.read_json(path)
    require(result.get("status") == "succeeded" and result.get("exit_code") == 0 and result.get("work_id") == work.work_id, f"BF16 result incomplete: {work.work_id}")
    require(result.get("model_inventory") == row["model_inventory"], f"BF16 result model mismatch: {work.work_id}")
    log = result.get("log")
    log_path = Path(log.get("path", "")) if isinstance(log, dict) else Path("")
    require(log_path.is_file() and c.file_sha256(log_path) == log.get("sha256"), f"BF16 log mismatch: {work.work_id}")
    require(c.completed_logged_run(log_path) is not None and c.completed_logged_run(log_path)[0] == 0, f"BF16 log footer invalid: {work.work_id}")
    metrics = q.validate_metrics(log_path) if work.kind == "quality" else result.get("metrics")
    require(metrics == result.get("metrics"), f"BF16 metrics mismatch: {work.work_id}")
    if work.kind != "quality":
        current = audit_reasoning_output(path.parent / "output", work)
        if work.kind in {"gsm8k", "math_500"}:
            require(current == metrics, f"BF16 reasoning parser mismatch: {work.work_id}")
        else:
            official_path = Path(metrics.get("official_result", "")) if isinstance(metrics, dict) else Path("")
            require(official_path.is_file() and c.file_sha256(official_path) == metrics.get("official_result_sha256"), f"BF16 official result mismatch: {work.work_id}")
    return {"work_id": work.work_id, "node": work.node, "model": work.model.slug, "kind": work.kind, "result": ref, "metrics": metrics}


def audit_all(root: Path) -> dict[str, Any]:
    plan = load_plan(root)
    work_rows = [audit_one(root, plan, work) for work in WORKS]
    by_model: dict[str, dict[str, Any]] = {model.slug: {} for model in c.MODELS}
    for row in work_rows:
        by_model[row["model"]][row["kind"]] = row["metrics"]
    comparison_rows = [
        {"run_id": run.run_id, "model": run.model.slug, "quant": run.quant.slug, "bf16_metrics": by_model[run.model.slug]}
        for run in c.RUNS
    ]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "bf16_id": BF16_ID,
        "audited_at": c.now(),
        "plan_fingerprint": plan["fingerprint"],
        "counts": {"unique_models": 5, "comparison_rows": 20, "wikitext2_ppl": 5, "paper_qa_tasks": 50, "reasoning_tasks": 15, "official_evalplus": 5},
        "models": by_model,
        "works": work_rows,
        "comparison_rows": comparison_rows,
    }
    identity = dict(payload)
    identity.pop("audited_at", None)
    payload["audit_fingerprint"] = c.canonical_sha256(identity)
    c.atomic_json(root / "bf16_final_audit.json", payload)
    return payload


def success_count(root: Path) -> int:
    return sum((work_dir(root, work) / "bf16_success.json").is_file() for work in WORKS)


def status(root: Path) -> dict[str, Any]:
    rows = []
    for work in WORKS:
        directory = work_dir(root, work)
        rows.append({"work_id": work.work_id, "node": work.node, "status": "succeeded" if (directory / "bf16_success.json").is_file() else "running" if (directory / CLAIM_NAME).is_dir() else "pending"})
    return {"bf16_id": BF16_ID, "success": sum(row["status"] == "succeeded" for row in rows), "rows": rows}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "audit", "status"):
        child = sub.add_parser(name)
        child.add_argument("--root", default=str(c.DEFAULT_ROOT))
    child = sub.add_parser("_worker")
    child.add_argument("--root", default=str(c.DEFAULT_ROOT))
    child.add_argument("--work-id", choices=tuple(WORK_BY_ID), required=True)
    child.add_argument("--gpu", type=int, choices=c.GPU_IDS, required=True)
    child = sub.add_parser("_exec")
    child.add_argument("--root", default=str(c.DEFAULT_ROOT))
    child.add_argument("--work-id", choices=tuple(WORK_BY_ID), required=True)
    child.add_argument("--output", required=True)
    child = sub.add_parser("_reference")
    child.add_argument("--root", default=str(c.DEFAULT_ROOT))
    child.add_argument("--model", choices=tuple(MODEL_BY_SLUG), required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = c.output_root(args.root)
    if args.command == "plan":
        print(json.dumps(write_plan(root), ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "audit":
        print(json.dumps(audit_all(root), ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "status":
        print(json.dumps(status(root), ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "_worker":
        run_worker(root, WORK_BY_ID[args.work_id], args.gpu)
    elif args.command == "_exec":
        exec_work(root, WORK_BY_ID[args.work_id], Path(args.output).resolve())
    elif args.command == "_reference":
        build_reference_cache(root, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
