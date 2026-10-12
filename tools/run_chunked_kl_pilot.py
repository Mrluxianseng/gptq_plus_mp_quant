"""Run a low-resource full-model REAL-Q control or chunked-KL pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_arg(command: list[str], name: str, value: str) -> None:
    if name in command:
        command[command.index(name) + 1] = value
    else:
        command.extend((name, value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--chunk-tokens", type=int, default=0)
    parser.add_argument("--label-chunk-tokens", type=int, default=0)
    parser.add_argument(
        "--full-config",
        action="store_true",
        help="use the manifest's calibration and evaluation sizes instead of the small pilot sizes",
    )
    parser.add_argument(
        "--stop-after-layer",
        type=int,
        help="stop quantization after this inclusive transformer layer (diagnostic runs)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="skip final KL/PPL evaluation (diagnostic runs)",
    )
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    args = parser.parse_args()
    if not args.tag.replace("-", "").isalnum():
        parser.error("--tag may contain only letters, digits, and hyphens")
    if args.chunk_tokens < 0:
        parser.error("--chunk-tokens must be nonnegative")
    if args.label_chunk_tokens < 0:
        parser.error("--label-chunk-tokens must be nonnegative")

    source_manifest_path = Path(args.manifest).resolve()
    source_manifest = json.loads(source_manifest_path.read_text())
    command = list(source_manifest["command"])
    run_name = f"chunked_kl_pilot_{args.tag}"
    replace_arg(command, "--exp", run_name)
    if not args.full_config:
        # Scale sample/sequence work for a short full 28-layer code-path check.
        for name, value in (
            ("--nsamples", "8"),
            ("--seq_len", "128"),
            ("--backward_samples", "8"),
            ("--backward_bsz", "4"),
            ("--final_layer_backward_bsz", "4"),
            ("--eval_seq_len", "256"),
            ("--eval_samples", "2"),
        ):
            replace_arg(command, name, value)
    command.append("--enable_memory_profile")
    if args.chunk_tokens:
        replace_arg(command, "--kl_logit_chunk_tokens", str(args.chunk_tokens))
    if args.label_chunk_tokens:
        replace_arg(
            command,
            "--static_fisher_label_chunk_tokens",
            str(args.label_chunk_tokens),
        )
    if args.stop_after_layer is not None:
        replace_arg(command, "--quant_stop_layer", str(args.stop_after_layer))
    if args.skip_eval:
        command.append("--skip_eval")

    run_dir = ROOT / "outputs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    entry_index = next(
        i for i, value in enumerate(command)
        if str(value).endswith("/entry.py") or str(value).endswith("\\entry.py")
    )
    source_entry = Path(command[entry_index])
    entry_copy = run_dir / "entry.py"
    entry_copy.write_bytes(source_entry.read_bytes())
    command[entry_index] = str(entry_copy)

    tracked = [
        ROOT / "process_args.py",
        ROOT / "gptq_utils" / "gptq_plus_utils.py",
        ROOT / "utils" / "loss_utils.py",
        ROOT / "utils" / "memory_utils.py",
        ROOT / "ptq.py",
        entry_copy,
    ]
    run_manifest = {
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256(source_manifest_path),
        "source_purpose": source_manifest.get("purpose"),
        "command": command,
        "source_sha256_at_launch": {str(path): sha256(path) for path in tracked},
        "chunk_tokens": args.chunk_tokens,
        "label_chunk_tokens": args.label_chunk_tokens,
        "pilot_config": {
            "nsamples": command[command.index("--nsamples") + 1],
            "seq_len": command[command.index("--seq_len") + 1],
            "backward_samples": command[command.index("--backward_samples") + 1],
            "backward_bsz": command[command.index("--backward_bsz") + 1],
            "final_layer_backward_bsz": command[command.index("--final_layer_backward_bsz") + 1],
            "eval_seq_len": command[command.index("--eval_seq_len") + 1],
            "eval_samples": command[command.index("--eval_samples") + 1],
            "seed": 44,
            "full_model": True,
            "full_config": args.full_config,
            "stop_after_layer": args.stop_after_layer,
            "skip_eval": args.skip_eval,
        },
        "timeout_seconds": args.timeout_seconds,
    }
    (run_dir / "manifest.json").write_text(json.dumps(run_manifest, indent=2))

    env = os.environ.copy()
    env.update(
        RESEARCH_RUN_DIR=str(run_dir),
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    started = time.monotonic()
    with (run_dir / "run.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            ["timeout", "--signal=TERM", "--kill-after=30s", str(args.timeout_seconds), *command],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    status = {
        "returncode": proc.returncode,
        "wall_seconds": time.monotonic() - started,
        "finished": time.strftime("%F %T"),
    }
    (run_dir / "status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))
    if proc.returncode:
        raise RuntimeError(f"pilot failed; inspect {run_dir / 'run.log'}")


if __name__ == "__main__":
    main()
