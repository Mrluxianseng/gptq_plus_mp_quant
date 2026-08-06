"""One-example-per-task GPU smoke test for ``realq.benchmarks``.

The command loads a real local checkpoint, generates one response for each
supported reasoning task, scores the math tasks, and exports (but never
executes) generated code for the two code tasks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from realq.benchmarks.runner import SUPPORTED_TASKS, run_reasoning_eval
from realq.config import Config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", default="datasets/reasoning_eval")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("reasoning smoke test requires CUDA")
    model_path = str(Path(args.model).expanduser().resolve())
    output_path = Path(args.output_dir).expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
    ).eval()
    cfg = Config(
        model=model_path,
        reasoning_eval=True,
        reasoning_tasks=list(SUPPORTED_TASKS),
        reasoning_data_dir=args.data_dir,
        reasoning_output_dir=str(output_path),
        reasoning_batch_size=1,
        reasoning_limit=1,
        reasoning_max_new_tokens=args.max_new_tokens,
        reasoning_num_samples=1,
        reasoning_enable_thinking=False,
        reasoning_do_sample=False,
    )
    manifest = run_reasoning_eval(model, tokenizer, cfg)
    completed_tasks = tuple(item["task"] for item in manifest["results"])
    if manifest["status"] != "completed":
        raise AssertionError(f"reasoning manifest did not complete: {manifest}")
    if completed_tasks != SUPPORTED_TASKS:
        raise AssertionError(
            f"reasoning task order drift: {completed_tasks} != {SUPPORTED_TASKS}"
        )
    required_outputs = (
        output_path / "manifest.json",
        output_path / "gsm8k" / "scores.json",
        output_path / "math_500" / "scores.json",
        output_path / "humaneval_plus" / "evalplus_samples.jsonl",
        output_path
        / "livecodebench_lite"
        / "livecodebench_custom_outputs.json",
        output_path / "CODE_EVALUATION.txt",
    )
    missing = [str(path) for path in required_outputs if not path.is_file()]
    if missing:
        raise AssertionError(f"reasoning smoke outputs are missing: {missing}")
    print(
        json.dumps(
            {
                "status": "passed",
                "manifest_status": manifest["status"],
                "tasks": list(completed_tasks),
                "output_dir": str(output_path),
                "cuda_device": torch.cuda.get_device_name(),
                "elapsed_seconds": manifest["elapsed_seconds"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
