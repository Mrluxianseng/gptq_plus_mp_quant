"""Replay identical REAL-Q prefixes and branch at selected Adam refreshes."""
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
DEFAULT_POLICIES = (
    "adam",
    "reset_m",
    "grams_direction",
    "norm_matched_gradient",
    "cautious",
    "cautious_mask",
    "cautious_scale",
    "no_parameter_update",
)
POLICIES = DEFAULT_POLICIES + (
    "reset_m_direction_norm_matched_adam",
    "adam_direction_norm_matched_reset_m",
)
SOURCE_FILES = (
    "realq/refresh/block_gd.py",
    "realq/refresh/kl_loss.py",
    "realq/runner/layer_loop.py",
    "realq/config.py",
    "realq/pipeline.py",
    "realq/ptq.py",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(file.relative_to(path)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def parse_targets(value: str) -> list[dict[str, object]]:
    targets = []
    for part in value.split(";"):
        fields = [field.strip() for field in part.split(",")]
        if len(fields) != 3:
            raise ValueError("targets must be layer,module,block entries separated by semicolons")
        layer, module, block = fields
        targets.append({"layer": int(layer), "module": module, "block": int(block)})
    if not targets:
        raise ValueError("at least one target is required")
    if len({(item["layer"], item["module"], item["block"]) for item in targets}) != len(targets):
        raise ValueError("duplicate target refresh")
    return targets


def run_one(
    *,
    common: list[str],
    model: Path,
    campaign: Path,
    seed: int,
    target: dict[str, object],
    policy: str,
    timeout_seconds: int,
    discard_checkpoint: bool,
) -> dict[str, object]:
    label = f"s{seed}-l{target['layer']}-{target['module'].replace('.', '_')}-b{target['block']}-{policy}"
    run_dir = campaign / label
    logs_dir = run_dir / "logs"
    checkpoint_dir = run_dir / "checkpoint"
    logs_dir.mkdir(parents=True)
    command = common + [
        "--seed", str(seed),
        "--refresh_seed", "0",
        "--output_dir", str(logs_dir),
        "--save_qmodel_path", str(checkpoint_dir),
        "--exp", label,
    ]
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(ROOT),
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        REALQ_ADAM_PROBE_RULE=policy,
        REALQ_ADAM_PROBE_LAYER=str(target["layer"]),
        REALQ_ADAM_PROBE_MODULE=str(target["module"]),
        REALQ_ADAM_PROBE_BLOCK=str(target["block"]),
    )
    env.pop("REALQ_ADAM_HISTORY_POLICY", None)
    started = time.monotonic()
    with (run_dir / "run.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            ["timeout", "--signal=TERM", "--kill-after=30s", str(timeout_seconds), *command],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    log_text = (run_dir / "run.log").read_text(errors="replace")
    probes = re.findall(r"\[adam_probe_snapshot\] ([^\r\n]+)", log_text)
    direction_lines = re.findall(r"\[adam_probe_direction\] ([^\r\n]+)", log_text)
    applied_lines = re.findall(r"\[adam_probe_applied\] ([^\r\n]+)", log_text)
    metrics = re.findall(
        r"(?:Exact )?KL&PPL on wikitext2: ([0-9.eE+-]+), ([0-9.eE+-]+)",
        log_text,
    )
    probe = probes[0] if len(probes) == 1 else None
    fields = dict(re.findall(r"([a-z0-9_]+)=([^ ]+)", probe or ""))
    direction = (
        dict(re.findall(r"([a-z0-9_]+)=([^ ]+)", direction_lines[0]))
        if len(direction_lines) == 1
        else None
    )
    applied = (
        dict(re.findall(r"([a-z0-9_]+)=([^ ]+)", applied_lines[0]))
        if len(applied_lines) == 1
        else None
    )
    active = probe is not None and f"rule={policy}" in probe
    checkpoint_hash = tree_sha256(checkpoint_dir)
    record: dict[str, object] = {
        "returncode": result.returncode,
        "wall_seconds": time.monotonic() - started,
        "policy": policy,
        "policy_active_at_target": active,
        "target_probe_count": len(probes),
        "target_snapshot": fields,
        "target_adam_direction": direction,
        "target_direction_record_count": len(direction_lines),
        "target_applied_update": applied,
        "target_applied_update_record_count": len(applied_lines),
        "kl_ppl_records": [
            {"kl": float(kl), "ppl": float(ppl)} for kl, ppl in metrics
        ],
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_retained": not discard_checkpoint,
        "run_dir": str(run_dir),
    }
    (run_dir / "status.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    if discard_checkpoint and checkpoint_dir.exists():
        if checkpoint_dir.is_file():
            checkpoint_dir.unlink()
        else:
            shutil.rmtree(checkpoint_dir)
    if result.returncode != 0:
        raise RuntimeError(f"REAL-Q run failed: {label}; inspect {run_dir / 'run.log'}")
    if not active:
        raise RuntimeError(f"target intervention was not observed exactly once: {label}")
    if direction is None or applied is None:
        raise RuntimeError(f"missing unique target update diagnostics: {label}")
    if not metrics:
        raise RuntimeError(f"missing final KL/PPL record: {label}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--stop-layer", type=int, default=4)
    parser.add_argument("--nsamples", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--eval-seq-len", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument(
        "--policies",
        default=",".join(DEFAULT_POLICIES),
        help="comma-separated policy arms; adam is required as the same-snapshot control",
    )
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "outputs"),
        help="Directory for campaign artifacts.",
    )
    parser.add_argument(
        "--discard-checkpoints",
        action="store_true",
        help="Keep checkpoint SHA-256 in status, then remove the generated checkpoint to save disk.",
    )
    args = parser.parse_args()
    model = Path(args.model).resolve()
    if not (model / "config.json").is_file():
        raise FileNotFoundError(f"model config not found: {model}")
    targets = parse_targets(args.targets)
    seeds = [int(item) for item in args.seeds.split(",")]
    policies = tuple(item.strip() for item in args.policies.split(",") if item.strip())
    invalid_policies = sorted(set(policies) - set(POLICIES))
    if invalid_policies:
        raise ValueError(f"unknown policies: {invalid_policies}; allowed: {POLICIES}")
    if "adam" not in policies or len(set(policies)) != len(policies):
        raise ValueError("policies must contain adam exactly once and no duplicates")
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    campaign = output_root / f"realq-adam-direction-{args.tag}"
    campaign.mkdir(parents=True, exist_ok=False)
    sources = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    common = [
        sys.executable, "-u", "-m", "realq.ptq",
        "--model", str(model),
        "--dataset", "wikitext2", "--eval_datasets", "wikitext2",
        "--nsamples", str(args.nsamples), "--seq_len", str(args.seq_len),
        "--eval_seq_len", str(args.eval_seq_len), "--eval_samples", str(args.eval_samples),
        "--w_bits", "4", "--w_clip", "true", "--num_groups", "4",
        "--blocksize", "128", "--act_order", "true", "--rotate", "false",
        "--group_parallel_quant", "none", "--global_loss_bsz", "1", "--bsz", "1",
        "--hessian_accum_bsz", "8", "--backward_samples", str(args.nsamples),
        "--backward_bsz", "1", "--final_layer_backward_bsz", "1",
        "--quant_stop_layer", str(args.stop_layer),
        "--log_column_block_loss", "true", "--skip_eval", "false",
        "--skip_kl_ppl_eval", "false",
    ]
    results: dict[str, object] = {}
    failure: str | None = None
    try:
        for seed in seeds:
            for target in targets:
                key = f"seed{seed}-layer{target['layer']}-{target['module']}-block{target['block']}"
                arms: dict[str, dict[str, object]] = {}
                for policy in policies:
                    arms[policy] = run_one(
                        common=common,
                        model=model,
                        campaign=campaign,
                        seed=seed,
                        target=target,
                        policy=policy,
                        timeout_seconds=args.timeout_seconds,
                        discard_checkpoint=args.discard_checkpoints,
                    )
                    if policy != "adam":
                        expected = arms["adam"]["target_snapshot"]
                        got = arms[policy]["target_snapshot"]
                        keys = ("w_sha256", "g_sha256", "m_sha256", "v_sha256", "adam_step")
                        mismatch = {
                            name: (expected.get(name), got.get(name))
                            for name in keys
                            if expected.get(name) != got.get(name)
                        }
                        if mismatch:
                            raise RuntimeError(
                                f"pre-update snapshot mismatch for {key}/{policy}: {mismatch}"
                            )
                results[key] = {
                    "target": target,
                    "seed": seed,
                    "pre_update_snapshot_equal": True,
                    "arms": arms,
                }
                (campaign / "partial_summary.json").write_text(
                    json.dumps(results, indent=2), encoding="utf-8"
                )
    except Exception as exc:
        failure = repr(exc)
        raise
    finally:
        unchanged = all(
            sha256_file(ROOT / name) == digest for name, digest in sources.items()
        )
        summary = {
            "tag": args.tag,
            "runner": "realq.ptq -> realq.pipeline -> realq.runner",
            "model": str(model),
            "config": {
                "w_bits": 4,
                "nsamples": args.nsamples,
                "seq_len": args.seq_len,
                "eval_seq_len": args.eval_seq_len,
                "eval_samples": args.eval_samples,
                "stop_layer_inclusive": args.stop_layer,
                "refresh_seed": 0,
                "other_algorithm_settings": "REAL-Q defaults with command-line flags recorded in each status/log",
            },
            "targets": targets,
            "seeds": seeds,
            "arms": list(policies),
            "source_hashes_unchanged": unchanged,
            "discard_checkpoints_after_hash": args.discard_checkpoints,
            "failure": failure,
            "results": results,
            "notes": [
                "Every arm deterministically replays the prefix and must match the target pre-update weight, gradient, m, v, and Adam step hashes.",
                "Only the selected target refresh changes its weight update; later refreshes use ordinary Adam and actual GPTQ quantization/compensation.",
                "no_parameter_update still advances Adam moments at the target but leaves that step's floating weights unchanged; it is a diagnostic anchor.",
                "Small-prefix and 16-window evaluation results are screening evidence only.",
            ],
        }
        (campaign / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        if not unchanged:
            raise RuntimeError("tracked sources changed during campaign")


if __name__ == "__main__":
    main()
