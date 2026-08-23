"""Generation-based reasoning benchmarks for ``realq_benchmark``.

Keep this module lightweight: ``prepare_data`` is intentionally runnable on a
networked host that may not have the repository's PyTorch environment.
"""

from __future__ import annotations

from typing import Any


SUPPORTED_TASKS = (
    "gsm8k",
    "math_500",
    "humaneval_plus",
    "livecodebench_lite",
)


def run_reasoning_eval(model: Any, tokenizer: Any, cfg: Any) -> dict[str, Any]:
    from realq_benchmark.benchmarks.runner import run_reasoning_eval as _run

    return _run(model, tokenizer, cfg)


__all__ = ["SUPPORTED_TASKS", "run_reasoning_eval"]
