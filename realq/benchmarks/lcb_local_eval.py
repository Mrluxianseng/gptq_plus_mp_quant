"""Feed a materialized local release to LiveCodeBench's official evaluator."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from realq.benchmarks.data import _iter_jsonl


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-file", required=True)
    parser.add_argument("--custom-output-file", required=True)
    parser.add_argument("--release-version", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> None:
    if os.environ.get("REALQ_ALLOW_UNTRUSTED_CODE") != "1":
        raise RuntimeError(
            "Refusing to run LiveCodeBench directly. Use official_eval from "
            "inside an isolated sandbox."
        )
    args = _parser().parse_args(argv)
    dataset_file = Path(args.dataset_file).expanduser().resolve()
    outputs_file = Path(args.custom_output_file).expanduser().resolve()
    source_dir = Path(args.source_dir).expanduser().resolve()
    if not dataset_file.is_file():
        raise FileNotFoundError(dataset_file)
    if not outputs_file.is_file():
        raise FileNotFoundError(outputs_file)
    if not (source_dir / "lcb_runner").is_dir():
        raise FileNotFoundError(source_dir)
    if args.release_version == "release_latest":
        raise ValueError("LiveCodeBench requires a pinned release.")

    sys.path.insert(0, str(source_dir))
    try:
        from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
        from lcb_runner.runner import scenario_router
    except ImportError as exc:
        raise RuntimeError(
            "The pinned LiveCodeBench source or one of its evaluator "
            "dependencies is unavailable."
        ) from exc

    rows = list(_iter_jsonl(dataset_file))

    def load_local(
        release_version: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[Any]:
        if release_version != args.release_version:
            raise ValueError(
                f"Requested {release_version!r}, expected "
                f"{args.release_version!r}."
            )
        problems = [CodeGenerationProblem(**row) for row in rows]
        if start_date is not None:
            lower = datetime.strptime(start_date, "%Y-%m-%d")
            problems = [
                problem for problem in problems if lower <= problem.contest_date
            ]
        if end_date is not None:
            upper = datetime.strptime(end_date, "%Y-%m-%d")
            problems = [
                problem for problem in problems if problem.contest_date <= upper
            ]
        return problems

    scenario_router.load_code_generation_dataset = load_local
    from lcb_runner.runner import custom_evaluator

    previous_argv = sys.argv
    try:
        sys.argv = [
            "lcb_runner.runner.custom_evaluator",
            "--custom_output_file",
            str(outputs_file),
            "--release_version",
            args.release_version,
            *args.extra,
        ]
        custom_evaluator.main()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
