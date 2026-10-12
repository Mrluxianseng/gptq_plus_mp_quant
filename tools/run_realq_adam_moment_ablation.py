"""Compare isolated Adam moment and bias-age policies across REAL-Q refreshes.

Both arms execute ``realq.ptq`` with identical settings and share the same
calibration/evaluation caches. The reset arm changes only
REALQ_ADAM_HISTORY_POLICY and performs no additional backward pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_tree(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.is_file():
        return sha256(path)
    h = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(file.relative_to(path)).encode())
        h.update(file.read_bytes())
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--stop-layer", type=int, default=4)
    parser.add_argument("--nsamples", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--eval-seq-len", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh-seed", type=int, default=0)
    parser.add_argument(
        "--policies",
        default="keep,reset_m_each_refresh,reset_v_each_refresh,reset_bias_age_each_refresh",
        help="comma-separated optimizer history policies to compare",
    )
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args()
    if not args.tag.replace("-", "").isalnum():
        raise ValueError("tag must contain only letters, digits, and hyphens")
    if args.stop_layer < 0 or args.nsamples <= 0 or args.seq_len <= 0:
        raise ValueError("prefix, sample count, and sequence length must be positive")
    if not Path(args.model, "config.json").is_file():
        raise FileNotFoundError(f"model config not found: {args.model}")

    campaign = ROOT / "outputs" / f"realq-adam-history-{args.tag}"
    campaign.mkdir(parents=True, exist_ok=False)
    tracked = [
        ROOT / "realq" / "refresh" / "block_gd.py",
        ROOT / "realq" / "runner" / "layer_loop.py",
        ROOT / "realq" / "config.py",
        ROOT / "realq" / "pipeline.py",
        ROOT / "realq" / "ptq.py",
    ]
    source_hashes = {str(p): sha256(p) for p in tracked}
    runs: dict[str, dict[str, object]] = {}
    policies = [value.strip() for value in args.policies.split(",") if value.strip()]
    supported = {
        "keep",
        "reset_m_each_refresh",
        "reset_v_each_refresh",
        "reset_bias_age_each_refresh",
        "reset_each_refresh",
    }
    if len(policies) < 2 or len(set(policies)) != len(policies):
        raise ValueError("provide at least two unique policies")
    if any(policy not in supported for policy in policies):
        raise ValueError(f"unsupported policy; choose from {sorted(supported)}")

    # Keep REAL-Q's default algorithm/config; reduce only calibration tokens,
    # sequence length, and quantized prefix for a local smoke experiment.
    common = [
        sys.executable, "-u", "-m", "realq.ptq",
        "--model", str(Path(args.model).resolve()),
        "--dataset", "wikitext2",
        "--eval_datasets", "wikitext2",
        "--seed", str(args.seed),
        "--refresh_seed", str(args.refresh_seed),
        "--nsamples", str(args.nsamples),
        "--seq_len", str(args.seq_len),
        "--eval_seq_len", str(args.eval_seq_len),
        "--eval_samples", str(args.eval_samples),
        "--w_bits", "4",
        "--w_clip", "true",
        "--num_groups", "4",
        "--blocksize", "128",
        "--act_order", "true",
        "--rotate", "false",
        "--group_parallel_quant", "none",
        "--global_loss_bsz", "1",
        "--bsz", "1",
        "--hessian_accum_bsz", "8",
        "--backward_samples", str(args.nsamples),
        "--backward_bsz", "1",
        "--final_layer_backward_bsz", "1",
        "--quant_stop_layer", str(args.stop_layer),
        "--log_column_block_loss", "true",
        "--skip_eval", "false",
        "--skip_kl_ppl_eval", "false",
    ]

    for arm in policies:
        policy = None if arm == "keep" else arm
        run_dir = campaign / arm
        output_dir = run_dir / "logs"
        checkpoint_dir = run_dir / "checkpoint"
        run_dir.mkdir()
        output_dir.mkdir()
        command = common + [
            "--exp", f"adam_history_{args.tag}_{arm}",
            "--output_dir", str(output_dir),
            "--save_qmodel_path", str(checkpoint_dir),
        ]
        env = os.environ.copy()
        env.update(
            PYTHONPATH=str(ROOT),
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
        klppl = re.findall(
            r"(?:Exact )?KL&PPL on wikitext2: ([0-9.eE+-]+), ([0-9.eE+-]+)",
            log_text,
        )
        marker = (
            f"[adam_history_ablation] policy={policy}"
            if policy is not None
            else None
        )
        runs[arm] = {
            "returncode": result.returncode,
            "wall_seconds": time.monotonic() - started,
            "policy": policy or "keep",
            "policy_active": policy is None or marker in log_text,
            "kl_ppl_records": [
                {"kl": float(k), "ppl": float(p)} for k, p in klppl
            ],
            "checkpoint_sha256": hash_tree(checkpoint_dir),
            "run_dir": str(run_dir),
        }
        (run_dir / "status.json").write_text(
            json.dumps(runs[arm], indent=2), encoding="utf-8"
        )
        if result.returncode:
            break

    unchanged = all(sha256(Path(p)) == digest for p, digest in source_hashes.items())
    complete = all(runs.get(arm, {}).get("returncode") == 0 for arm in policies)
    policy_active = all(
        runs.get(arm, {}).get("policy_active", False) for arm in policies
    )
    summary = {
        "tag": args.tag,
        "runner": "realq.ptq -> realq.pipeline -> realq.runner",
        "model": str(Path(args.model).resolve()),
        "smoke_config": {
            "w_bits": 4,
            "nsamples": args.nsamples,
            "seq_len": args.seq_len,
            "eval_seq_len": args.eval_seq_len,
            "eval_samples": args.eval_samples,
            "seed": args.seed,
            "refresh_seed": args.refresh_seed,
            "quant_stop_layer_inclusive": args.stop_layer,
            "other_algorithm_settings": "REAL-Q Config defaults, explicitly pinning the listed common flags in run.log",
        },
        "source_hashes_unchanged": unchanged,
        "policies": policies,
        "all_policies_active": policy_active,
        "runs": runs,
        "note": (
            "This is a small prefix screening, not a paper-scale result. "
            "Every arm has the same backward calls and configuration; each "
            "candidate resets only the named active Adam state component(s) "
            "before each discrete refresh."
        ),
    }
    (campaign / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not complete or not unchanged or not policy_active:
        raise RuntimeError(f"REAL-Q Adam-history probe failed its execution gate; inspect {campaign}")


if __name__ == "__main__":
    main()

