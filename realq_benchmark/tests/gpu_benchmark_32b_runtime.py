"""Measure 32B RealQ A/K/V runtime throughput on real LCB prompts.

This is an explicit GPU benchmark, not a pytest test and not an accuracy run.
The model weights remain BF16; the actual RealQ activation/K/V wrappers are
installed to measure their online cost and memory scaling.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from realq_benchmark import akv
from realq_benchmark.benchmarks.runner import run_reasoning_eval
from realq_benchmark.config import Config
from utils import dist_utils
from utils.model_utils import ModelAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="modelzoo/Qwen3/Qwen3-32B",
    )
    parser.add_argument(
        "--data-dir",
        default="datasets/reasoning_eval",
    )
    parser.add_argument(
        "--task",
        choices=(
            "gsm8k",
            "math_500",
            "humaneval_plus",
            "livecodebench_lite",
        ),
        default="livecodebench_lite",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[4, 8, 16],
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise ValueError("--batch-sizes must contain positive integers.")

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    load_started = time.perf_counter()
    analyzer = ModelAnalyzer(
        args.model,
        args.seq_len,
        skip_state_dict=True,
    )
    load_finished = time.perf_counter()
    runtime_cfg = Config(
        model=args.model,
        rotate=False,
        w_bits=16,
        a_bits=4,
        a_groupsize=-1,
        k_bits=4,
        k_groupsize=-1,
        v_bits=4,
        v_groupsize=-1,
    )
    akv.setup_unaware_post_quant(analyzer, runtime_cfg)
    wrapper_finished = time.perf_counter()
    dist_utils.distribute_model(analyzer.model)
    torch.cuda.synchronize()
    dispatch_finished = time.perf_counter()

    reports = []
    for batch_size in args.batch_sizes:
        destination = output_root / f"batch_{batch_size}"
        cfg = Config(
            model=args.model,
            rotate=False,
            w_bits=16,
            a_bits=4,
            a_groupsize=-1,
            k_bits=4,
            k_groupsize=-1,
            v_bits=4,
            v_groupsize=-1,
            reasoning_eval=True,
            reasoning_tasks=[args.task],
            reasoning_data_dir=args.data_dir,
            reasoning_output_dir=str(destination),
            reasoning_limit=batch_size,
            reasoning_batch_size=batch_size,
            reasoning_max_new_tokens=args.max_new_tokens,
            reasoning_num_samples=1,
            reasoning_enable_thinking=args.enable_thinking,
            reasoning_do_sample=args.do_sample,
            reasoning_seed=1234,
            reasoning_resume=False,
            reasoning_lcb_release="release_v6",
            reasoning_lcb_source_dir=(
                "datasets/reasoning_eval/vendor/LiveCodeBench"
            ),
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        manifest = run_reasoning_eval(
            analyzer.model,
            analyzer.tokenizer,
            cfg,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        rows = [
            json.loads(line)
            for line in (
                destination
                / args.task
                / "generations.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        token_counts = [
            len(
                analyzer.tokenizer.encode(
                    str(row["output"]),
                    add_special_tokens=False,
                )
            )
            for row in rows
        ]
        report = {
            "status": manifest["status"],
            "batch_size": batch_size,
            "generated_tokens": sum(token_counts),
            "tokens_per_sample": token_counts,
            "eval_seconds": elapsed,
            "aggregate_output_tok_s": sum(token_counts) / elapsed,
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated() / 2**30
            ),
            "peak_reserved_gib": (
                torch.cuda.max_memory_reserved() / 2**30
            ),
        }
        reports.append(report)
        print(json.dumps(report, indent=2), flush=True)

    summary = {
        "gpu": torch.cuda.get_device_name(),
        "weight_quantized": False,
        "runtime_proxy": "BF16 weights + RealQ A4K4V4 wrappers",
        "cpu_model_load_seconds": load_finished - load_started,
        "wrapper_setup_seconds": wrapper_finished - load_finished,
        "gpu_dispatch_seconds": dispatch_finished - wrapper_finished,
        "max_new_tokens": args.max_new_tokens,
        "task": args.task,
        "enable_thinking": args.enable_thinking,
        "do_sample": args.do_sample,
        "reports": reports,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
