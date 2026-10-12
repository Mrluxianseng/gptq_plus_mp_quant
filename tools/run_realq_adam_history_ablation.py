"""Compare REAL-Q Adam history retention with per-refresh moment restarts.

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
import shutil
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


def parse_layer_scope(scope: str, num_layers: int) -> set[int]:
    """Parse `all`, inclusive ranges, or comma-separated layer indices."""
    value = scope.strip().lower()
    if value == "all":
        return set(range(num_layers))
    selected: set[int] = set()
    for part in value.split(","):
        item = part.strip()
        if not item:
            raise ValueError(f"empty item in layer scope {scope!r}")
        if "-" in item:
            bounds = item.split("-")
            if len(bounds) != 2 or not all(x.strip().isdigit() for x in bounds):
                raise ValueError(f"invalid layer range {item!r}")
            low, high = (int(x.strip()) for x in bounds)
            if high < low:
                raise ValueError(f"layer range end precedes start: {item!r}")
        else:
            if not item.isdigit():
                raise ValueError(f"invalid layer index {item!r}")
            low = high = int(item)
        if low < 0 or high >= num_layers:
            raise ValueError(
                f"layer scope {item!r} is outside [0, {num_layers - 1}]"
            )
        selected.update(range(low, high + 1))
    if not selected:
        raise ValueError("layer scope selects no layers")
    return selected


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
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "outputs"),
        help="Directory for campaign artifacts (use WSL ext4 when the repo drive is low on space).",
    )
    parser.add_argument(
        "--candidate-policy",
        choices=(
            "reset_each_refresh",
            "reset_m_each_refresh",
            "reset_m_and_m_age_each_refresh",
            "reset_v_each_refresh",
            "reset_v_and_v_age_each_refresh",
            "reset_bias_age_each_refresh",
            "radam",
            "adabelief",
            "adopt",
        ),
        default="reset_each_refresh",
        help="Adam history policy used by the treatment arm.",
    )
    parser.add_argument(
        "--candidate-layers",
        default="all",
        help=(
            "Layers receiving the candidate policy: all, an inclusive range "
            "such as 0-26, or comma-separated indices such as 27."
        ),
    )
    parser.add_argument(
        "--discard-checkpoints-after-hash",
        action="store_true",
        help="Hash generated checkpoints, then remove them to conserve local disk space.",
    )
    args = parser.parse_args()
    if not args.tag.replace("-", "").isalnum():
        raise ValueError("tag must contain only letters, digits, and hyphens")
    if args.stop_layer < 0 or args.nsamples <= 0 or args.seq_len <= 0:
        raise ValueError("prefix, sample count, and sequence length must be positive")
    if not Path(args.model, "config.json").is_file():
        raise FileNotFoundError(f"model config not found: {args.model}")
    model_config = json.loads(
        Path(args.model, "config.json").read_text(encoding="utf-8")
    )
    num_layers = int(
        model_config.get(
            "num_hidden_layers",
            model_config.get("n_layer", model_config.get("num_layers", 0)),
        )
    )
    if num_layers <= 0:
        raise ValueError("could not read model layer count from config.json")
    requested_layers = parse_layer_scope(args.candidate_layers, num_layers)
    expected_marker_layers = {
        layer for layer in requested_layers if layer <= args.stop_layer
    }

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    campaign = output_root / f"realq-adam-history-{args.tag}"
    campaign.mkdir(parents=True, exist_ok=False)
    tracked = [
        ROOT / "realq" / "refresh" / "block_gd.py",
        ROOT / "realq" / "refresh" / "kl_loss.py",
        ROOT / "realq" / "runner" / "layer_loop.py",
        ROOT / "realq" / "config.py",
        ROOT / "realq" / "pipeline.py",
        ROOT / "realq" / "ptq.py",
    ]
    source_hashes = {str(p): sha256(p) for p in tracked}
    runs: dict[str, dict[str, object]] = {}

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

    candidate_arm = args.candidate_policy
    for arm, policy in (("keep", None), (candidate_arm, args.candidate_policy)):
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
            env.pop("REALQ_ADAM_HISTORY_LAYERS", None)
        else:
            env["REALQ_ADAM_HISTORY_POLICY"] = policy
            env["REALQ_ADAM_HISTORY_LAYERS"] = args.candidate_layers
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
        marker = f"[adam_history_ablation] policy={args.candidate_policy}"
        final_kl_marker = (
            f"[adam_history_ablation] policy={args.candidate_policy} "
            "path=final_layer_kl"
        )
        is_candidate = policy is not None
        observed_marker_layers = {
            int(value)
            for value in re.findall(
                re.escape(marker) + r"[^\n]*?layer=(\d+)", log_text
            )
        }
        block_policy_active = (
            not is_candidate
            or observed_marker_layers == expected_marker_layers
        )
        final_layer_idx = num_layers - 1
        final_kl_policy_active = (
            not is_candidate
            or final_layer_idx not in expected_marker_layers
            or (
                final_kl_marker in log_text
                and f"layer={final_layer_idx}" in log_text
            )
        )
        checkpoint_digest = hash_tree(checkpoint_dir)
        if args.discard_checkpoints_after_hash and checkpoint_dir.exists():
            if checkpoint_dir.is_dir():
                shutil.rmtree(checkpoint_dir)
            else:
                checkpoint_dir.unlink()
        runs[arm] = {
            "returncode": result.returncode,
            "wall_seconds": time.monotonic() - started,
            "policy": policy or "keep",
            "block_policy_active": block_policy_active,
            "expected_policy_layers": sorted(expected_marker_layers),
            "observed_policy_layers": sorted(observed_marker_layers),
            "final_kl_policy_active": final_kl_policy_active,
            "policy_active": block_policy_active and final_kl_policy_active,
            "kl_ppl_records": [
                {"kl": float(k), "ppl": float(p)} for k, p in klppl
            ],
            "checkpoint_sha256": checkpoint_digest,
            "checkpoint_retained": checkpoint_dir.exists(),
            "run_dir": str(run_dir),
        }
        (run_dir / "status.json").write_text(
            json.dumps(runs[arm], indent=2), encoding="utf-8"
        )
        if result.returncode:
            break

    source_hashes_after = {
        path: sha256(Path(path)) for path in source_hashes
    }
    unchanged = all(
        source_hashes_after[path] == digest
        for path, digest in source_hashes.items()
    )
    complete = all(
        runs.get(arm, {}).get("returncode") == 0
        for arm in ("keep", candidate_arm)
    )
    policy_active = bool(
        runs.get(candidate_arm, {}).get("policy_active", False)
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
        "source_hashes_before": source_hashes,
        "source_hashes_after": source_hashes_after,
        "reset_policy_active": policy_active,
        "discard_checkpoints_after_hash": args.discard_checkpoints_after_hash,
        "policy_scope": (
            f"Candidate Adam policy applies to model layers {sorted(requested_layers)} "
            "in the default REAL-Q path, including the final-layer KL path when "
            "that layer is selected."
        ),
        "candidate_layers": args.candidate_layers,
        "candidate_policy": args.candidate_policy,
        "runs": runs,
        "note": (
            (
                "This is a small prefix screening, not a paper-scale result. "
                if args.stop_layer < 27
                else "This is a full 28-layer local comparison with reduced calibration/evaluation context, not a paper-scale result. "
            )
            + "The candidate arm has the same backward calls and all settings as keep; "
            "the selected Adam history policy changes first-moment history on both "
            "Fisher/Block-GD and final-layer KL refreshes."
        ),
    }
    (campaign / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not complete or not unchanged or not policy_active:
        raise RuntimeError(f"REAL-Q Adam-history probe failed its execution gate; inspect {campaign}")


if __name__ == "__main__":
    main()
