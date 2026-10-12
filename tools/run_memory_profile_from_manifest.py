"""Run a saved GPTQ+ configuration with the opt-in CUDA phase memory profile."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Saved full-run command manifest")
    parser.add_argument("--tag", required=True, help="Unique run tag using letters, numbers, and hyphens")
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    if not args.tag.replace("-", "").isalnum():
        raise ValueError("tag may contain only letters, digits, and hyphens")
    if args.timeout_seconds <= 0:
        raise ValueError("timeout must be positive")

    manifest_path = Path(args.manifest).resolve()
    source_manifest = json.loads(manifest_path.read_text())
    command = list(source_manifest["command"])
    exp_index = command.index("--exp") + 1
    command[exp_index] = f"memory_profile_{args.tag}"
    command.append("--enable_memory_profile")
    if args.skip_eval and "--skip_eval" not in command:
        command.append("--skip_eval")

    run_dir = ROOT / "outputs" / f"memory_profile_{args.tag}"
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
        ROOT / "utils" / "memory_utils.py",
        ROOT / "ptq.py",
        entry_copy,
    ]
    run_manifest = {
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "source_run_purpose": source_manifest.get("purpose"),
        "command": command,
        "source_sha256_at_launch": {str(path): sha256(path) for path in tracked},
        "memory_profile_enabled": True,
        "skip_eval": bool(args.skip_eval),
        "timeout_seconds": args.timeout_seconds,
        "note": "Reuses saved quantization arguments. Source code is the current workspace version, not the historical source revision recorded by the source manifest.",
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
            [
                "timeout", "--signal=TERM", "--kill-after=30s",
                str(args.timeout_seconds), *command,
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    status = {
        "returncode": proc.returncode,
        "wall_seconds": time.monotonic() - started,
        "finished": time.strftime("%F %T"),
        "memory_profile_files": sorted(
            str(path.relative_to(ROOT))
            for path in ROOT.glob(f"outputs/Qwen3-0.6B/{command[command.index('--exp') + 1]}/memory_profile/*.jsonl")
        ),
    }
    (run_dir / "status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))
    if proc.returncode:
        raise RuntimeError("memory-profile run failed; inspect run.log and partial JSONL records")


if __name__ == "__main__":
    main()
