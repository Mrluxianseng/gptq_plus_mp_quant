"""Explicit entry point for official execution-based code scorers.

This module intentionally refuses to execute generated code unless the caller
confirms that the current process is already inside an appropriate sandbox.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        required=True,
        choices=("humaneval_plus", "livecodebench_lite"),
    )
    parser.add_argument("--samples", required=True)
    parser.add_argument("--humaneval-data", default=None)
    parser.add_argument("--lcb-data", default=None)
    parser.add_argument("--lcb-release", default="release_v6")
    parser.add_argument(
        "--lcb-source",
        default="./datasets/reasoning_eval/vendor/LiveCodeBench",
    )
    parser.add_argument(
        "--i-understand-generated-code-will-run",
        action="store_true",
        help="Required confirmation that this process is inside a sandbox.",
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to the official evaluator.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    samples = Path(args.samples).expanduser().resolve()
    if not samples.is_file():
        raise FileNotFoundError(samples)
    if not args.i_understand_generated_code_will_run:
        raise RuntimeError(
            "Refusing to execute model-generated code. Enter an isolated "
            "sandbox first, then pass --i-understand-generated-code-will-run."
        )
    if args.lcb_release == "release_latest":
        raise ValueError("LiveCodeBench scoring requires a pinned release.")
    if args.task == "humaneval_plus":
        if args.humaneval_data is None:
            raise ValueError("--humaneval-data is required for HumanEval+.")
        humaneval_data = Path(args.humaneval_data).expanduser().resolve()
        if not humaneval_data.is_file():
            raise FileNotFoundError(humaneval_data)
        command = [
            sys.executable,
            "-m",
            "evalplus.evaluate",
            "--dataset",
            "humaneval",
            "--samples",
            str(samples),
        ]
    else:
        if args.lcb_data is None:
            raise ValueError("--lcb-data is required for LiveCodeBench.")
        lcb_data = Path(args.lcb_data).expanduser().resolve()
        if not lcb_data.is_file():
            raise FileNotFoundError(lcb_data)
        lcb_source = Path(args.lcb_source).expanduser().resolve()
        if not (lcb_source / "lcb_runner").is_dir():
            raise FileNotFoundError(
                f"LiveCodeBench source checkout not found at {lcb_source}."
            )
        command = [
            sys.executable,
            "-m",
            "realq_benchmark.benchmarks.lcb_local_eval",
            "--dataset-file",
            str(lcb_data),
            "--custom_output_file",
            str(samples),
            "--release-version",
            args.lcb_release,
            "--source-dir",
            str(lcb_source),
        ]
    environment = os.environ.copy()
    environment["REALQ_ALLOW_UNTRUSTED_CODE"] = "1"
    if args.task == "humaneval_plus":
        environment["HUMANEVAL_OVERRIDE_PATH"] = str(humaneval_data)
    subprocess.run(command + args.extra, check=True, env=environment)


if __name__ == "__main__":
    main()
