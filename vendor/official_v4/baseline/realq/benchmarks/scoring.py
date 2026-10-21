"""Deterministic math scoring and official code-evaluator interchange files."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

from realq.benchmarks.schema import BenchmarkExample


_NUMBER = re.compile(r"-?(?:\d[\d,]*)(?:\.\d+)?(?:/[+-]?\d+)?")
_FINAL_ANSWER = re.compile(
    r"(?:final\s+answer|answer\s+is)\s*:?\s*(.+)",
    flags=re.IGNORECASE,
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _last_numeric(text: str) -> str | None:
    matches = _NUMBER.findall(text.replace("$", ""))
    return matches[-1] if matches else None


def extract_gsm8k_answer(text: str) -> str | None:
    matches = _FINAL_ANSWER.findall(text)
    if matches:
        candidate = _last_numeric(matches[-1])
        if candidate is not None:
            return candidate
    return _last_numeric(text)


def _numeric_value(value: str | None) -> Fraction | None:
    if value is None:
        return None
    normalized = value.replace(",", "").strip().rstrip(".")
    try:
        return Fraction(normalized)
    except (ValueError, ZeroDivisionError):
        return None


def gsm8k_correct(gold: str, prediction: str) -> tuple[bool, str | None, str | None]:
    gold_tail = gold.rsplit("####", 1)[-1]
    gold_answer = _last_numeric(gold_tail)
    predicted_answer = extract_gsm8k_answer(prediction)
    gold_value = _numeric_value(gold_answer)
    predicted_value = _numeric_value(predicted_answer)
    return (
        gold_value is not None
        and predicted_value is not None
        and gold_value == predicted_value,
        gold_answer,
        predicted_answer,
    )


def math500_correct(gold: str, prediction: str) -> bool:
    try:
        from math_verify import parse, verify
    except ImportError as exc:
        raise RuntimeError(
            "MATH-500 scoring requires `math-verify[antlr4-13-2]==0.9.0`."
        ) from exc
    try:
        return bool(verify(parse(gold), parse(prediction)))
    except Exception:
        return False


def extract_python_code(text: str) -> str:
    # Qwen3 thinking mode may emit reasoning outside a Markdown code block.
    # Preserve it in generations.jsonl for audit, but never feed it to a code
    # evaluator as Python source.
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if "</think>" in text.lower():
        text = re.split(
            r"</think>",
            text,
            flags=re.IGNORECASE,
        )[-1]
    fenced = re.findall(
        r"```(?:python|py)?\s*\n?(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        return max(fenced, key=len).strip()
    return text.strip()


def _records_by_example(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["sample_id"])].append(record)
    for values in grouped.values():
        values.sort(key=lambda value: int(value["sample_index"]))
    return grouped


def _score_math_task(
    task: str,
    examples: list[BenchmarkExample],
    records: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    grouped = _records_by_example(records)
    details = []
    first_correct = 0
    any_correct = 0
    for example in examples:
        candidates = grouped.get(example.sample_id, [])
        candidate_scores = []
        for record in candidates:
            output = str(record["output"])
            if task == "gsm8k":
                correct, gold_answer, predicted_answer = gsm8k_correct(
                    example.target or "",
                    output,
                )
            else:
                correct = math500_correct(example.target or "", output)
                gold_answer = example.target
                predicted_answer = None
            candidate_scores.append(
                {
                    "sample_index": int(record["sample_index"]),
                    "correct": correct,
                    "gold_answer": gold_answer,
                    "predicted_answer": predicted_answer,
                }
            )
        first = bool(candidate_scores and candidate_scores[0]["correct"])
        oracle = any(value["correct"] for value in candidate_scores)
        first_correct += int(first)
        any_correct += int(oracle)
        details.append(
            {
                "sample_id": example.sample_id,
                "sample_count": len(candidate_scores),
                "pass_at_1": first,
                "pass_at_n_oracle": oracle,
                "candidates": candidate_scores,
            }
        )
    denominator = len(examples)
    summary = {
        "task": task,
        "status": "scored",
        "num_examples": denominator,
        "num_generations": len(records),
        "pass_at_1": first_correct / denominator if denominator else 0.0,
        "pass_at_n_oracle": any_correct / denominator if denominator else 0.0,
        "note": (
            "pass_at_n_oracle is the fraction with any correct candidate; it "
            "is not the unbiased pass@k estimator."
        ),
    }
    _atomic_json(output_dir / "scores.json", {"summary": summary, "details": details})
    return summary


def _export_humaneval_plus(
    examples: list[BenchmarkExample],
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    dataset_path: str,
) -> dict[str, Any]:
    by_id = {example.sample_id: example for example in examples}
    destination = output_dir / "evalplus_samples.jsonl"
    with destination.open("w", encoding="utf-8") as handle:
        for record in sorted(
            records,
            key=lambda value: (
                str(value["sample_id"]),
                int(value["sample_index"]),
            ),
        ):
            example = by_id[str(record["sample_id"])]
            code = extract_python_code(str(record["output"]))
            entry_point = example.metadata.get("entry_point")
            if entry_point and f"def {entry_point}" not in code:
                code = str(example.metadata["source_prompt"]) + code
            handle.write(
                json.dumps(
                    {
                        "task_id": example.sample_id,
                        "solution": code,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    summary = {
        "task": "humaneval_plus",
        "status": "generated_unscored",
        "num_examples": len(examples),
        "num_generations": len(records),
        "official_input": str(destination.resolve()),
        "official_evaluator": "evalplus.evaluate --dataset humaneval",
        "official_dataset": dataset_path,
    }
    _atomic_json(output_dir / "scores.json", {"summary": summary})
    return summary


def _export_livecodebench(
    examples: list[BenchmarkExample],
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    lcb_release: str,
    lcb_data_path: str,
    lcb_source_path: str,
) -> dict[str, Any]:
    grouped = _records_by_example(records)
    payload = [
        {
            "question_id": example.sample_id,
            "code_list": [
                extract_python_code(str(record["output"]))
                for record in grouped.get(example.sample_id, [])
            ],
        }
        for example in examples
    ]
    destination = output_dir / "livecodebench_custom_outputs.json"
    _atomic_json(destination, payload)
    summary = {
        "task": "livecodebench_lite",
        "status": "generated_unscored",
        "num_examples": len(examples),
        "num_generations": len(records),
        "official_input": str(destination.resolve()),
        "official_evaluator": (
            "python -m lcb_runner.runner.custom_evaluator "
            f"--release_version {lcb_release}"
        ),
        "release_version": lcb_release,
        "official_dataset": lcb_data_path,
        "official_source": lcb_source_path,
    }
    _atomic_json(output_dir / "scores.json", {"summary": summary})
    return summary


def score_or_export(
    task: str,
    examples: list[BenchmarkExample],
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    humaneval_data_path: str = "",
    lcb_release: str = "release_v6",
    lcb_data_path: str = "",
    lcb_source_path: str = "",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if task in {"gsm8k", "math_500"}:
        return _score_math_task(task, examples, records, output_dir)
    if task == "humaneval_plus":
        return _export_humaneval_plus(
            examples,
            records,
            output_dir,
            dataset_path=humaneval_data_path,
        )
    if task == "livecodebench_lite":
        return _export_livecodebench(
            examples,
            records,
            output_dir,
            lcb_release=lcb_release,
            lcb_data_path=lcb_data_path,
            lcb_source_path=lcb_source_path,
        )
    raise ValueError(task)
