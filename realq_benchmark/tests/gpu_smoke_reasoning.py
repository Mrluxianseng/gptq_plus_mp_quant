"""One-example-per-task CUDA smoke test for the reasoning-eval plumbing.

This is deliberately not named ``test_*.py``: it loads a real checkpoint and
must only be run explicitly on a GPU worker.  It verifies model generation,
chat rendering, resumable persistence, math score file creation, and code-task
official-input export.  It never executes generated code.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import tempfile
import types
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from realq_benchmark.benchmarks.runner import run_reasoning_eval
from realq_benchmark.config import Config


def _write_jsonl(path: Path, rows: list[dict], *, compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if compressed else open
    with opener(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _make_fixture(root: Path) -> None:
    _write_jsonl(
        root / "gsm8k" / "test.jsonl",
        [{"question": "What is 40 + 2?", "answer": "work #### 42"}],
    )
    _write_jsonl(
        root / "math_500" / "test.jsonl",
        [{"unique_id": "m0", "problem": "Compute 1+1.", "answer": "2"}],
    )
    _write_jsonl(
        root / "humaneval_plus" / "HumanEvalPlus.jsonl.gz",
        [
            {
                "task_id": "HumanEval/0",
                "prompt": "def answer():\n    \"\"\"Return 42.\"\"\"\n",
                "entry_point": "answer",
            }
        ],
        compressed=True,
    )
    _write_jsonl(
        root / "livecodebench_lite" / "release_v6.jsonl",
        [
            {
                "question_id": "lcb0",
                "question_content": "Read no input and print the integer 42.",
                "starter_code": "",
            }
        ],
    )


def _ensure_smoke_math_backend() -> bool:
    """Let wiring tests finish when optional Math-Verify is not installed."""

    try:
        import math_verify  # noqa: F401

        return True
    except ImportError:
        module = types.ModuleType("math_verify")
        module.parse = lambda value: str(value)
        module.verify = lambda _gold, _prediction: False
        sys.modules["math_verify"] = module
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires CUDA.")

    temporary = tempfile.TemporaryDirectory(prefix="realq_reasoning_gpu_smoke_")
    root = Path(temporary.name)
    data_dir = root / "data"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else root / "output"
    _make_fixture(data_dir)
    has_math_verify = _ensure_smoke_math_backend()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    ).eval()
    cfg = Config(
        model=args.model,
        reasoning_eval=True,
        reasoning_data_dir=str(data_dir),
        reasoning_output_dir=str(output_dir),
        reasoning_batch_size=1,
        reasoning_limit=1,
        reasoning_max_new_tokens=args.max_new_tokens,
        reasoning_num_samples=1,
        reasoning_enable_thinking=False,
        reasoning_do_sample=False,
    )
    manifest = run_reasoning_eval(model, tokenizer, cfg)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "tasks": [result["task"] for result in manifest["results"]],
                "output_dir": str(output_dir),
                "math_verify_installed": has_math_verify,
                "cuda_device": torch.cuda.get_device_name(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.output_dir:
        temporary.cleanup()


if __name__ == "__main__":
    main()
