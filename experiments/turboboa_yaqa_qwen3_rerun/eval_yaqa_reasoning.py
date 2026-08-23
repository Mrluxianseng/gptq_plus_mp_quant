#!/usr/bin/env python3
"""Run one frozen reasoning task from a validated YAQA_wclip HF artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import socket
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_PROTOCOL = {
    "qwen3-4b": {
        "gsm8k": (64, 1024),
        "math_500": (32, 2048),
        "humaneval_plus": (32, 2048),
    },
    "qwen3-32b": {
        "gsm8k": (32, 1024),
        "math_500": (16, 2048),
        "humaneval_plus": (16, 2048),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_key(model: Path) -> str:
    name = model.name.lower()
    if name == "qwen3-4b":
        return "qwen3-4b"
    if name == "qwen3-32b":
        return "qwen3-32b"
    raise ValueError(f"unsupported formal model: {model}")


def _bits(setting: str) -> tuple[int, int, int, int]:
    if setting == "W3A16KV16":
        return 3, 16, 16, 16
    if setting == "W4A4KV4":
        return 4, 4, 4, 4
    raise ValueError(f"unsupported formal setting: {setting}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--setting",
        required=True,
        choices=("W3A16KV16", "W4A4KV4"),
    )
    parser.add_argument("--hf-dir", required=True)
    parser.add_argument("--hf-validation", required=True)
    parser.add_argument("--expected-validation-sha256", required=True)
    parser.add_argument(
        "--task",
        required=True,
        choices=("gsm8k", "math_500", "humaneval_plus"),
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    hostname = socket.gethostname()
    if not re.fullmatch(r"j-[a-z0-9]+-master-0", hostname):
        raise RuntimeError(
            "reasoning generation may run only on a Canoe experiment pod"
        )

    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError("a reasoning worker must see exactly one GPU")
    model_path = Path(args.model).resolve()
    hf_dir = Path(args.hf_dir).resolve()
    validation_path = Path(args.hf_validation).resolve()
    output_dir = Path(args.output_dir).resolve()
    validation_sha = _sha256(validation_path)
    if validation_sha != args.expected_validation_sha256:
        raise RuntimeError(
            "YAQA HF validation receipt changed before reasoning launch"
        )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if (
        validation.get("schema_version") != 2
        or validation.get("status") != "validated"
        or validation.get("model") != str(model_path)
        or validation.get("setting") != args.setting
        or Path(validation.get("hf_dir", "")).resolve() != hf_dir
        or validation.get("runtime_weight_representation")
        != "dense_fake_quant"
        or validation.get("packed") is not False
        or validation.get("lowbit_kernel") is not False
    ):
        raise RuntimeError("YAQA HF validation receipt does not match this run")

    from lib.utils.unsafe_import import model_from_hf_path
    from realq_benchmark.benchmarks.runner import run_reasoning_eval
    from utils import model_utils

    model, _ = model_from_hf_path(
        str(hf_dir),
        device_map={"": torch.cuda.current_device()},
    )
    model.eval()
    analyzer = model_utils.ModelAnalyzer(
        model,
        2048,
        tokenizer_source=str(model_path),
        skip_state_dict=True,
    )
    w_bits, a_bits, k_bits, v_bits = _bits(args.setting)
    batch_size, max_new_tokens = TASK_PROTOCOL[_model_key(model_path)][
        args.task
    ]
    cfg = SimpleNamespace(
        model=str(hf_dir),
        load_qmodel_path=None,
        w_bits=w_bits,
        w_groupsize=128,
        a_bits=a_bits,
        a_groupsize=-1,
        k_bits=k_bits,
        k_groupsize=-1,
        v_bits=v_bits,
        v_groupsize=-1,
        rotate=False,
        reasoning_tasks=[args.task],
        reasoning_data_dir=str(REPO_ROOT / "datasets/reasoning_eval"),
        reasoning_output_dir=str(output_dir / "results"),
        reasoning_batch_size=batch_size,
        reasoning_limit=-1,
        reasoning_max_new_tokens=max_new_tokens,
        reasoning_num_samples=1,
        reasoning_apply_chat_template=True,
        reasoning_enable_thinking=True,
        reasoning_do_sample=False,
        reasoning_temperature=0.6,
        reasoning_top_p=0.95,
        reasoning_top_k=20,
        reasoning_seed=1234,
        reasoning_resume=True,
        reasoning_protocol="realq_zero_shot_v1",
        reasoning_system_prompt=(
            "You are a careful reasoning assistant. Follow the requested "
            "output format exactly."
        ),
        reasoning_lcb_release="release_v6",
        reasoning_lcb_source_dir=str(
            REPO_ROOT
            / "datasets/reasoning_eval/vendor/LiveCodeBench"
        ),
        output_dir=str(output_dir),
        exp=(
            f"reasoning_yaqa_wclip_{_model_key(model_path)}_"
            f"{args.setting}_{args.task}"
        ),
    )
    manifest = run_reasoning_eval(
        analyzer.model,
        analyzer.tokenizer,
        cfg,
    )
    if (
        manifest.get("status") != "completed"
        or manifest.get("tasks") != [args.task]
        or len(manifest.get("results", [])) != 1
    ):
        raise RuntimeError("YAQA reasoning manifest is incomplete")
    print(json.dumps(manifest["results"][0], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
