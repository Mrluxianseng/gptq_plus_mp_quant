"""Dataset loading and prompt construction for reasoning benchmarks.

Formal runs should use materialized files under ``reasoning_data_dir``.  A
Hugging Face fallback is provided for interactive use, but offline Canoe runs
must prepare the files on the networked host first.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

from realq.benchmarks.schema import BenchmarkExample


TASK_ALIASES = {
    "gsm8k": "gsm8k",
    "math500": "math_500",
    "math-500": "math_500",
    "math_500": "math_500",
    "humaneval+": "humaneval_plus",
    "humanevalplus": "humaneval_plus",
    "humaneval_plus": "humaneval_plus",
    "livecodebench-lite": "livecodebench_lite",
    "livecodebench_lite": "livecodebench_lite",
    "lcb_lite": "livecodebench_lite",
}


def _human_eval_plus_rows(
    problems: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Preserve EvalPlus mapping keys when a release omits value task IDs."""

    rows = []
    for task_id, problem in problems.items():
        row = dict(problem)
        row.setdefault("task_id", task_id)
        rows.append(row)
    return rows


def canonical_task_name(task: str) -> str:
    normalized = task.strip().lower()
    try:
        return TASK_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported reasoning task {task!r}; expected one of "
            f"{sorted(set(TASK_ALIASES.values()))}."
        ) from exc


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number} must contain a JSON object."
                )
            yield value


def _first_existing(root: Path, candidates: Iterable[str]) -> Path | None:
    for relative in candidates:
        path = root / relative
        if path.is_file():
            return path
    return None


def _materialized_path(task: str, root: Path, lcb_release: str) -> Path | None:
    candidates = {
        "gsm8k": ("gsm8k/test.jsonl", "gsm8k.jsonl"),
        "math_500": ("math_500/test.jsonl", "math500/test.jsonl", "math500.jsonl"),
        "humaneval_plus": (
            "humaneval_plus/HumanEvalPlus.jsonl.gz",
            "HumanEvalPlus.jsonl.gz",
            "humanevalplus.jsonl.gz",
        ),
        "livecodebench_lite": (
            f"livecodebench_lite/{lcb_release}.jsonl",
            f"livecodebench_lite/{lcb_release}.jsonl.gz",
            "livecodebench_lite.jsonl",
            "livecodebench_lite.jsonl.gz",
        ),
    }
    return _first_existing(root, candidates[task])


def _hf_rows(task: str, lcb_release: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "No materialized benchmark file was found and the optional "
            "`datasets` dependency is unavailable."
        ) from exc

    if task == "gsm8k":
        try:
            dataset = load_dataset("openai/gsm8k", "main", split="test")
        except Exception:
            dataset = load_dataset("gsm8k", "main", split="test")
    elif task == "math_500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    elif task == "humaneval_plus":
        try:
            from evalplus.data import get_human_eval_plus
        except ImportError as exc:
            raise RuntimeError(
                "HumanEval+ needs a materialized HumanEvalPlus.jsonl.gz or "
                "`evalplus==0.3.1`."
            ) from exc
        return _human_eval_plus_rows(get_human_eval_plus())
    elif task == "livecodebench_lite":
        dataset = load_dataset(
            "livecodebench/code_generation_lite",
            split="test",
            version_tag=lcb_release,
        )
    else:  # guarded by canonical_task_name
        raise AssertionError(task)
    return [dict(row) for row in dataset]


def _load_rows(
    task: str,
    data_dir: str,
    lcb_release: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(data_dir).expanduser().resolve()
    path = _materialized_path(task, root, lcb_release)
    if path is not None:
        rows = list(_iter_jsonl(path))
        stat = path.stat()
        source = {
            "source": "materialized_jsonl",
            "path": str(path),
            "size_bytes": stat.st_size,
            "sha256": _sha256(path),
        }
        if task == "livecodebench_lite":
            metadata_path = path.with_name(f"{lcb_release}.metadata.json")
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("release_version") != lcb_release:
                    raise ValueError(
                        f"{metadata_path} release_version does not match "
                        f"{lcb_release!r}."
                    )
                official_path = path.with_name(str(metadata["official_file"]))
                if not official_path.is_file():
                    raise FileNotFoundError(official_path)
                if metadata.get("prompt_sha256") != source["sha256"]:
                    raise ValueError(
                        f"{metadata_path} prompt hash does not match {path}."
                    )
                if metadata.get("num_examples") != len(rows):
                    raise ValueError(
                        f"{metadata_path} row count does not match {path}."
                    )
                if (
                    metadata.get("official_size_bytes")
                    != official_path.stat().st_size
                ):
                    raise ValueError(
                        f"{metadata_path} official size does not match "
                        f"{official_path}."
                    )
                source.update(
                    {
                        "metadata_path": str(metadata_path),
                        "official_path": str(official_path),
                        "official_size_bytes": metadata["official_size_bytes"],
                        "official_sha256": metadata["official_sha256"],
                    }
                )
        return rows, source
    if os.environ.get("HF_DATASETS_OFFLINE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        raise FileNotFoundError(
            f"No materialized file for {task!r} under {root}; Hugging Face "
            "fallback is disabled by HF_DATASETS_OFFLINE."
        )
    rows = _hf_rows(task, lcb_release)
    return rows, {
        "source": "huggingface_datasets",
        "path": None,
        "size_bytes": None,
        "sha256": None,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _gsm8k_example(row: dict[str, Any], index: int) -> BenchmarkExample:
    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()
    return BenchmarkExample(
        task="gsm8k",
        sample_id=str(row.get("id", index)),
        prompt=(
            "Solve the following grade-school math problem. Show your reasoning, "
            "then put only the final numeric answer after `Final answer:`.\n\n"
            f"{question}"
        ),
        target=answer,
        metadata={"question": question},
    )


def _math500_example(row: dict[str, Any], index: int) -> BenchmarkExample:
    problem = str(row["problem"]).strip()
    answer = str(row.get("answer", row.get("solution", ""))).strip()
    return BenchmarkExample(
        task="math_500",
        sample_id=str(row.get("unique_id", row.get("id", index))),
        prompt=(
            "Solve the following math problem efficiently and clearly. Think "
            "step by step. The last line must be exactly of the form "
            "`Final answer: \\\\boxed{ANSWER}`, where ANSWER is the final number "
            "or expression.\n\n"
            f"{problem}"
        ),
        target=answer,
        metadata={
            key: row[key]
            for key in ("subject", "level")
            if key in row
        },
    )


def _humaneval_example(row: dict[str, Any], index: int) -> BenchmarkExample:
    task_id = str(row.get("task_id", f"HumanEval/{index}"))
    source_prompt = str(row["prompt"])
    return BenchmarkExample(
        task="humaneval_plus",
        sample_id=task_id,
        prompt=(
            "Complete the following Python function. Return one self-contained "
            "Python solution and no explanation or Markdown fences.\n\n"
            f"{source_prompt}"
        ),
        metadata={
            "source_prompt": source_prompt,
            "entry_point": row.get("entry_point"),
        },
    )


def _lcb_example(row: dict[str, Any], index: int) -> BenchmarkExample:
    question_id = str(row.get("question_id", index))
    starter = str(row.get("starter_code", "") or "").strip()
    content = str(row["question_content"]).strip()
    starter_section = (
        f"\n\nUse this starter code:\n```python\n{starter}\n```"
        if starter
        else ""
    )
    return BenchmarkExample(
        task="livecodebench_lite",
        sample_id=question_id,
        prompt=(
            "Solve the following competitive-programming problem in Python. "
            "Return only the complete executable program, without explanation "
            "or Markdown fences.\n\n"
            f"{content}{starter_section}"
        ),
        metadata={
            key: row.get(key)
            for key in (
                "question_title",
                "contest_date",
                "difficulty",
                "platform",
                "starter_code",
            )
        },
    )


_ADAPTERS = {
    "gsm8k": _gsm8k_example,
    "math_500": _math500_example,
    "humaneval_plus": _humaneval_example,
    "livecodebench_lite": _lcb_example,
}


def load_examples(
    task: str,
    *,
    data_dir: str,
    lcb_release: str,
    limit: int = -1,
) -> tuple[list[BenchmarkExample], dict[str, Any]]:
    canonical = canonical_task_name(task)
    rows, source = _load_rows(canonical, data_dir, lcb_release)
    examples = [
        _ADAPTERS[canonical](row, index)
        for index, row in enumerate(rows)
    ]
    if limit > 0:
        examples = examples[:limit]
    source.update(
        {
            "task": canonical,
            "available_examples": len(rows),
            "selected_examples": len(examples),
            "lcb_release": lcb_release if canonical == "livecodebench_lite" else None,
        }
    )
    return examples, source
