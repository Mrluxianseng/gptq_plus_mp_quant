"""Top-level orchestration for the four reasoning benchmarks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from realq.benchmarks.data import (
    canonical_task_name,
    load_examples,
)
from realq.benchmarks.generation import (
    HFLMGenerator,
    generate_resumable,
    make_requests,
)
from realq.benchmarks.scoring import score_or_export


SUPPORTED_TASKS = (
    "gsm8k",
    "math_500",
    "humaneval_plus",
    "livecodebench_lite",
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _tokenizer_template_sha256(tokenizer: Any) -> str | None:
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return None
    return hashlib.sha256(str(template).encode("utf-8")).hexdigest()


def _git_commit(path: Path) -> str | None:
    git_dir = path / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref_path = git_dir / head.removeprefix("ref: ")
    return ref_path.read_text(encoding="utf-8").strip() if ref_path.is_file() else None


def _output_root(cfg: Any) -> Path:
    if cfg.reasoning_output_dir:
        return Path(cfg.reasoning_output_dir).expanduser().resolve()
    return (
        Path(cfg.output_dir).expanduser().resolve()
        / cfg.exp
        / "reasoning_eval"
    )


def _canonical_tasks(tasks: list[str]) -> list[str]:
    result = []
    for task in tasks:
        canonical = canonical_task_name(task)
        if canonical not in result:
            result.append(canonical)
    if not result:
        raise ValueError("reasoning_tasks must contain at least one task.")
    return result


def _write_code_eval_instructions(root: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Official code scoring commands",
        "",
        "Generated model code is untrusted. Run these only inside an isolated",
        "sandbox/container without writable model, dataset, or workspace mounts.",
        "",
    ]
    for summary in summaries:
        if summary["task"] == "humaneval_plus":
            lines.extend(
                [
                    "HumanEval+:",
                    "  python -m realq.benchmarks.official_eval \\",
                    "    --task humaneval_plus \\",
                    f"    --samples {summary['official_input']} \\",
                    f"    --humaneval-data {summary['official_dataset']} \\",
                    "    --i-understand-generated-code-will-run",
                    "",
                ]
            )
        elif summary["task"] == "livecodebench_lite":
            lines.extend(
                [
                    "LiveCodeBench-lite:",
                    "  python -m realq.benchmarks.official_eval \\",
                    "    --task livecodebench_lite \\",
                    f"    --samples {summary['official_input']} \\",
                    f"    --lcb-data {summary['official_dataset']} \\",
                    f"    --lcb-release {summary['release_version']} \\",
                    f"    --lcb-source {summary['official_source']} \\",
                    "    --i-understand-generated-code-will-run",
                    "",
                ]
            )
    if len(lines) > 5:
        (root / "CODE_EVALUATION.txt").write_text(
            "\n".join(lines),
            encoding="utf-8",
        )


def run_reasoning_eval(model: Any, tokenizer: Any, cfg: Any) -> dict[str, Any]:
    """Generate, score math tasks, and export code tasks for official scoring."""

    started = time.time()
    root = _output_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    tasks = _canonical_tasks(list(cfg.reasoning_tasks))
    manifest_path = root / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_unix": started,
        "tasks": tasks,
        "model": str(cfg.model),
        "load_qmodel_path": cfg.load_qmodel_path,
        "quantization": {
            name: getattr(cfg, name)
            for name in (
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
        },
        "generation": {
            name.removeprefix("reasoning_"): getattr(cfg, name)
            for name in (
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
        },
        "tokenizer": {
            "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "chat_template_sha256": _tokenizer_template_sha256(tokenizer),
        },
        "versions": {
            package: _version(package)
            for package in (
                "torch",
                "transformers",
                "lm-eval",
                "datasets",
                "math-verify",
                "evalplus",
            )
        },
        "datasets": {},
        "livecodebench": {
            "source_dir": str(
                Path(cfg.reasoning_lcb_source_dir).expanduser().resolve()
            ),
            "source_commit": _git_commit(
                Path(cfg.reasoning_lcb_source_dir).expanduser().resolve()
            ),
        },
        "results": [],
    }
    _atomic_json(manifest_path, manifest)

    # Load every requested dataset before allocating lm-eval's wrapper. Offline
    # cache errors therefore fail early, before any expensive generation.
    loaded = {}
    for task in tasks:
        examples, source = load_examples(
            task,
            data_dir=cfg.reasoning_data_dir,
            lcb_release=cfg.reasoning_lcb_release,
            limit=cfg.reasoning_limit,
        )
        if not examples:
            raise RuntimeError(f"{task} selected zero examples.")
        loaded[task] = examples
        manifest["datasets"][task] = source
    _atomic_json(manifest_path, manifest)

    generator = HFLMGenerator(model, tokenizer, cfg.reasoning_batch_size)
    code_summaries = []
    try:
        for task in tasks:
            task_started = time.time()
            examples = loaded[task]
            task_dir = root / task
            requests = make_requests(
                examples,
                tokenizer,
                num_samples=cfg.reasoning_num_samples,
                apply_chat_template=cfg.reasoning_apply_chat_template,
                enable_thinking=cfg.reasoning_enable_thinking,
                system_prompt=cfg.reasoning_system_prompt,
            )
            logging.info(
                "[realq] %s: %d examples, %d generations",
                task,
                len(examples),
                len(requests),
            )
            records = generate_resumable(
                task=task,
                requests=requests,
                destination=task_dir / "generations.jsonl",
                cfg=cfg,
                generate=generator,
            )
            if len(records) != len(requests):
                raise RuntimeError(
                    f"{task}: expected {len(requests)} complete generations, "
                    f"found {len(records)}."
                )
            if task == "livecodebench_lite":
                lcb_source = manifest["datasets"][task]
                summary = score_or_export(
                    task,
                    examples,
                    records,
                    task_dir,
                    lcb_release=cfg.reasoning_lcb_release,
                    lcb_data_path=str(
                        lcb_source.get("official_path", lcb_source["path"])
                    ),
                    lcb_source_path=str(
                        Path(cfg.reasoning_lcb_source_dir).expanduser().resolve()
                    ),
                )
            elif task == "humaneval_plus":
                summary = score_or_export(
                    task,
                    examples,
                    records,
                    task_dir,
                    humaneval_data_path=str(
                        manifest["datasets"][task]["path"]
                    ),
                )
            else:
                summary = score_or_export(task, examples, records, task_dir)
            summary["elapsed_seconds"] = time.time() - task_started
            manifest["results"].append(summary)
            if task in {"humaneval_plus", "livecodebench_lite"}:
                code_summaries.append(summary)
            _atomic_json(manifest_path, manifest)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["elapsed_seconds"] = time.time() - started
        _atomic_json(manifest_path, manifest)
        raise

    _write_code_eval_instructions(root, code_summaries)
    manifest["status"] = "completed"
    manifest["elapsed_seconds"] = time.time() - started
    _atomic_json(manifest_path, manifest)
    logging.info("[realq] reasoning results saved to %s", root)
    return manifest
