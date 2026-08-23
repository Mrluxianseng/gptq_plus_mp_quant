"""Materialize benchmark data on a networked host for offline Canoe runs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from realq_benchmark.benchmarks.data import (
    _human_eval_plus_rows,
    canonical_task_name,
)

_LCB_PROMPT_FIELDS = (
    "question_id",
    "question_title",
    "question_content",
    "contest_date",
    "difficulty",
    "platform",
    "starter_code",
)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _copy_gzip_jsonl(destination: Path, rows: Iterable[dict[str, Any]]) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    count = 0
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
                count += 1
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return count


def materialize(task: str, root: Path, lcb_release: str) -> tuple[Path, int]:
    canonical = canonical_task_name(task)
    if canonical == "humaneval_plus":
        try:
            from evalplus.data import get_human_eval_plus
        except ImportError as exc:
            raise RuntimeError("Install evalplus==0.3.1 to prepare HumanEval+.") from exc
        destination = root / "humaneval_plus" / "HumanEvalPlus.jsonl.gz"
        rows = _human_eval_plus_rows(get_human_eval_plus())
        return destination, _copy_gzip_jsonl(destination, rows)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets to prepare benchmark data.") from exc

    if canonical == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        destination = root / "gsm8k" / "test.jsonl"
    elif canonical == "math_500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        destination = root / "math_500" / "test.jsonl"
    elif canonical == "livecodebench_lite":
        dataset = load_dataset(
            "livecodebench/code_generation_lite",
            split="test",
            version_tag=lcb_release,
            trust_remote_code=True,
        )
        destination = root / "livecodebench_lite" / f"{lcb_release}.jsonl"
        official_destination = destination.with_name(
            f"{lcb_release}.official.jsonl"
        )
        official_count = _write_jsonl(official_destination, dataset)
        prompt_count = _write_jsonl(
            destination,
            (
                {
                    key: row.get(key)
                    for key in _LCB_PROMPT_FIELDS
                    if key in row
                }
                for row in dataset
            ),
        )
        if prompt_count != official_count:
            raise RuntimeError(
                "LiveCodeBench prompt and official files have different counts: "
                f"{prompt_count} != {official_count}."
            )
        metadata_path = destination.with_name(f"{lcb_release}.metadata.json")
        _write_json(
            metadata_path,
            {
                "schema_version": 1,
                "release_version": lcb_release,
                "num_examples": prompt_count,
                "prompt_file": destination.name,
                "prompt_size_bytes": destination.stat().st_size,
                "prompt_sha256": _sha256(destination),
                "official_file": official_destination.name,
                "official_size_bytes": official_destination.stat().st_size,
                "official_sha256": _sha256(official_destination),
            },
        )
        return destination, prompt_count
    else:
        raise AssertionError(canonical)
    return destination, _write_jsonl(destination, dataset)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "gsm8k",
            "math_500",
            "humaneval_plus",
            "livecodebench_lite",
        ],
    )
    parser.add_argument("--data-dir", default="./datasets/reasoning_eval")
    parser.add_argument("--lcb-release", default="release_v6")
    args = parser.parse_args(argv)
    root = Path(args.data_dir).expanduser().resolve()
    for task in args.tasks:
        destination, count = materialize(task, root, args.lcb_release)
        print(f"{canonical_task_name(task)}: {count} rows -> {destination}")


if __name__ == "__main__":
    main()
