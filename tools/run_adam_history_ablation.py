"""Paired REAL-Q probe: retain Adam moments vs restart them per refresh.

Uses one saved baseline command for both arms. The candidate changes only the
research-only REALQ_ADAM_HISTORY_POLICY environment variable; it takes no
additional backward pass. Designed for short local prefix experiments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--stop-layer", type=int, default=None,
                        help="Override the manifest's short prefix length (inclusive).")
    args = parser.parse_args()
    if not args.tag.replace("-", "").isalnum():
        raise ValueError("tag must contain only letters, digits, and hyphens")

    source = json.loads(Path(args.manifest).read_text())
    base_command = list(source["command"])
    if base_command[base_command.index("--grad_optimizer") + 1] != "adam":
        raise ValueError("source manifest must use the baseline Adam optimizer")
    exp_pos = base_command.index("--exp") + 1
    if "--quant_stop_layer" not in base_command:
        raise ValueError("source command must define a short quantization prefix")
    if args.stop_layer is not None:
        base_command[base_command.index("--quant_stop_layer") + 1] = str(args.stop_layer)
        if "--quant_profile_target_layers" in base_command:
            base_command[base_command.index("--quant_profile_target_layers") + 1] = (
                ",".join(str(i) for i in range(args.stop_layer + 1))
            )

    campaign = ROOT / "outputs" / f"adam-history-{args.tag}"
    campaign.mkdir(parents=True, exist_ok=False)
    source_entry_index = next(
        i for i, value in enumerate(base_command)
        if Path(str(value)).name == "entry.py"
    )
    source_entry = Path(base_command[source_entry_index]).resolve()
    tracked = [
        ROOT / "realq" / "refresh" / "block_gd.py",
        ROOT / "realq" / "refresh" / "triton_block_adam.py",
        source_entry,
    ]
    initial_hashes = {str(path): sha256(path) for path in tracked}
    runs: dict[str, dict[str, object]] = {}
    for arm, policy in (("keep", None), ("reset", "reset_each_refresh")):
        run_dir = campaign / arm
        run_dir.mkdir()
        command = list(base_command)
        command[exp_pos] = f"adam_history_{args.tag}_{arm}"
        env = os.environ.copy()
        env.update(
            RESEARCH_RUN_DIR=str(run_dir),
            HF_HUB_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
        )
        if policy is None:
            env.pop("REALQ_ADAM_HISTORY_POLICY", None)
        else:
            env["REALQ_ADAM_HISTORY_POLICY"] = policy
        started = time.monotonic()
        with (run_dir / "run.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [
                    "timeout", "--signal=TERM", "--kill-after=30s",
                    str(args.timeout_seconds), *command,
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        log_text = (run_dir / "run.log").read_text(errors="replace")
        metric_files = list(run_dir.glob("*metrics.json"))
        metrics = json.loads(metric_files[0].read_text()) if metric_files else None
        match = re.search(
            r"Exact KL&PPL on wikitext2: ([0-9.eE+-]+), ([0-9.eE+-]+)",
            log_text,
        )
        policy_active = (
            policy is None
            or "[adam_history_ablation] policy=reset_each_refresh" in log_text
        )
        runs[arm] = {
            "returncode": result.returncode,
            "wall_seconds": time.monotonic() - started,
            "policy": policy or "keep",
            "metrics": metrics,
            "kl": float(match.group(1)) if match else None,
            "ppl": float(match.group(2)) if match else None,
            "policy_active": policy_active,
            "run_dir": str(run_dir),
        }
        (run_dir / "status.json").write_text(
            json.dumps(runs[arm], indent=2), encoding="utf-8"
        )
        if result.returncode:
            break

    unchanged = all(sha256(Path(path)) == digest for path, digest in initial_hashes.items())
    complete = all(runs.get(arm, {}).get("returncode") == 0 for arm in ("keep", "reset"))
    equal_state_hash = (
        complete
        and runs["keep"].get("metrics") is not None
        and runs["reset"].get("metrics") is not None
        and runs["keep"]["metrics"].get("state_sha256")
        == runs["reset"]["metrics"].get("state_sha256")
    )
    summary = {
        "tag": args.tag,
        "source_manifest": str(Path(args.manifest).resolve()),
        "source_manifest_sha256": sha256(Path(args.manifest).resolve()),
        "source_hashes_unchanged": unchanged,
        "quantized_state_hashes_equal": equal_state_hash,
        "runs": runs,
        "note": (
            "A short prefix screening only. Different states are expected if the "
            "moment policy changes the trajectory; evaluate KL/PPL as directional "
            "evidence, then repeat full-model/multiple-seed only if promising."
        ),
    }
    (campaign / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not complete or not unchanged or not runs.get("reset", {}).get("policy_active", False):
        raise RuntimeError(f"paired Adam-history probe failed; inspect {campaign}")


if __name__ == "__main__":
    main()
