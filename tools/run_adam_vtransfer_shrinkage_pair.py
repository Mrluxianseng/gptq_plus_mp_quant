"""Matched REAL-Q prefix probe for a conservative cross-layer Adam-v prior.

The candidate differs from ordinary Adam only by a gated, log-space blend of
the previous same-role layer's row profile into the target's current g^2 prior.
No extra backward pass is performed. This is a mechanism/quality screen, not a
paper-scale result.
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
TRACKED = (
    ROOT / "realq" / "refresh" / "adam_transfer_diagnostic.py",
    ROOT / "realq" / "refresh" / "block_gd.py",
    ROOT / "realq" / "runner" / "layer_loop.py",
    ROOT / "realq" / "config.py",
    ROOT / "realq" / "pipeline.py",
    ROOT / "realq" / "ptq.py",
    ROOT / "tools" / "run_adam_vtransfer_shrinkage_pair.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        return None
    for item in files:
        digest.update(str(item.relative_to(path)).encode())
        digest.update(item.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--target-layer", type=int, default=5)
    parser.add_argument(
        "--stop-layer", type=int, default=None,
        help="Inclusive quantization stop layer; defaults to target layer.",
    )
    parser.add_argument("--target-module", default="mlp.down_proj")
    parser.add_argument("--shrinkage", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--nsamples", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--eval-seq-len", type=int, default=128)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args()
    if not args.tag.replace("-", "").isalnum():
        raise ValueError("tag must contain only letters, digits, and hyphens")
    if not 0.0 < args.shrinkage <= 1.0:
        raise ValueError("candidate shrinkage must be in (0, 1]")
    model = Path(args.model).resolve()
    if not (model / "config.json").is_file():
        raise FileNotFoundError(f"model config not found: {model}")

    campaign = ROOT / "outputs" / f"adam-vtransfer-shrinkage-{args.tag}"
    campaign.mkdir(parents=True, exist_ok=False)
    stop_layer = args.target_layer if args.stop_layer is None else args.stop_layer
    if stop_layer < args.target_layer:
        raise ValueError("stop layer must include the target transfer layer")
    initial_hashes = {str(p): sha256(p) for p in TRACKED}
    common = [
        sys.executable, "-u", "-m", "realq.ptq",
        "--model", str(model),
        "--dataset", "wikitext2", "--eval_datasets", "wikitext2",
        "--seed", str(args.seed), "--refresh_seed", "0",
        "--nsamples", str(args.nsamples), "--seq_len", str(args.seq_len),
        "--eval_seq_len", str(args.eval_seq_len),
        "--eval_samples", str(args.eval_samples),
        "--w_bits", "4", "--w_clip", "true", "--num_groups", "4",
        "--blocksize", "128", "--act_order", "true", "--rotate", "false",
        "--group_parallel_quant", "none", "--global_loss_bsz", "1",
        "--bsz", "1", "--hessian_accum_bsz", "8",
        "--backward_samples", str(args.nsamples), "--backward_bsz", "1",
        "--final_layer_backward_bsz", "1",
        "--quant_stop_layer", str(stop_layer),
        "--log_column_block_loss", "true", "--skip_eval", "false",
        "--skip_kl_ppl_eval", "false",
    ]
    arms: dict[str, dict[str, object]] = {}
    for arm in ("control", "shrinkage"):
        run_dir = campaign / arm
        output_dir = run_dir / "logs"
        checkpoint_dir = run_dir / "checkpoint"
        output_dir.mkdir(parents=True)
        command = common + [
            "--exp", f"adam_vtransfer_{args.tag}_{arm}",
            "--output_dir", str(output_dir),
            "--save_qmodel_path", str(checkpoint_dir),
        ]
        env = os.environ.copy()
        env.update(
            PYTHONPATH=str(ROOT),
            HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
            REALQ_ADAM_TRANSFER_DIAG_DIR=str(run_dir),
            REALQ_ADAM_TRANSFER_DIAG_MAX_LAYER=str(args.target_layer),
            REALQ_ADAM_TRANSFER_DIAG_MODULES=args.target_module,
        )
        for name in (
            "REALQ_ADAM_V_TRANSFER_MODE",
            "REALQ_ADAM_V_TRANSFER_TARGET_LAYER",
            "REALQ_ADAM_V_TRANSFER_TARGET_MODULE",
            "REALQ_ADAM_V_TRANSFER_MIN_CORR",
            "REALQ_ADAM_V_TRANSFER_SHRINKAGE",
        ):
            env.pop(name, None)
        if arm == "shrinkage":
            env.update(
                REALQ_ADAM_V_TRANSFER_MODE="normalized_row_prior",
                REALQ_ADAM_V_TRANSFER_TARGET_LAYER=str(args.target_layer),
                REALQ_ADAM_V_TRANSFER_TARGET_MODULE=args.target_module,
                REALQ_ADAM_V_TRANSFER_MIN_CORR="0.4",
                REALQ_ADAM_V_TRANSFER_SHRINKAGE=str(args.shrinkage),
            )
        started = time.monotonic()
        with (run_dir / "run.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                ["timeout", "--signal=TERM", "--kill-after=30s",
                 str(args.timeout_seconds), *command],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            )
        log_text = (run_dir / "run.log").read_text(errors="replace")
        metrics = list(run_dir.glob("*metrics.json"))
        klppl = re.findall(
            r"(?:Exact )?KL&PPL on wikitext2: ([0-9.eE+-]+), ([0-9.eE+-]+)",
            log_text,
        )
        transfer_lines = [
            line for line in log_text.splitlines()
            if "[adam_v_transfer]" in line
        ]
        decision_lines = [
            line for line in log_text.splitlines()
            if "[adam_v_transfer_decision]" in line
        ]
        arms[arm] = {
            "returncode": result.returncode,
            "wall_seconds": time.monotonic() - started,
            "command": command,
            "kl_ppl_records": [
                {"kl": float(k), "ppl": float(p)} for k, p in klppl
            ],
            "metrics": json.loads(metrics[0].read_text()) if metrics else None,
            "transfer_lines": transfer_lines,
            "decision_lines": decision_lines,
            "checkpoint_sha256": tree_sha256(checkpoint_dir),
            "run_dir": str(run_dir),
        }
        (run_dir / "status.json").write_text(
            json.dumps(arms[arm], indent=2), encoding="utf-8"
        )
        if result.returncode:
            break

    source_unchanged = all(sha256(Path(p)) == digest for p, digest in initial_hashes.items())
    control_done = arms.get("control", {}).get("returncode") == 0
    candidate_done = arms.get("shrinkage", {}).get("returncode") == 0
    candidate_events = arms.get("shrinkage", {}).get("transfer_lines", [])
    candidate_decisions = arms.get("shrinkage", {}).get("decision_lines", [])
    summary = {
        "tag": args.tag,
        "runner": "realq.ptq -> realq.pipeline -> realq.runner",
        "model": str(model),
        "matched_config": {
            "seed": args.seed, "refresh_seed": 0,
            "nsamples": args.nsamples, "seq_len": args.seq_len,
            "eval_samples": args.eval_samples, "eval_seq_len": args.eval_seq_len,
            "quant_stop_layer_inclusive": stop_layer,
            "candidate_target": [args.target_layer, args.target_module],
            "candidate_correlation_gate": 0.4,
            "candidate_log_vhat_shrinkage": args.shrinkage,
            "only_treatment_difference": "gated Adam-v initialization",
        },
        "source_hashes_unchanged": source_unchanged,
        "candidate_gate_decision_observed": bool(candidate_decisions),
        "candidate_transfer_applied_once": len(candidate_events) == 1,
        "candidate_transfer_skipped_by_gate": bool(candidate_decisions) and not candidate_events,
        "runs": arms,
        "note": "Small REAL-Q prefix mechanism screen. One seed and short calibration/evaluation; not thesis-scale evidence.",
    }
    (campaign / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if (
        not control_done or not candidate_done or not source_unchanged
        or len(candidate_events) > 1 or not candidate_decisions
    ):
        raise RuntimeError(f"paired Adam-v shrinkage probe failed an execution gate; inspect {campaign}")


if __name__ == "__main__":
    main()
