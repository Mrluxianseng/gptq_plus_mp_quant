"""Paired full-model audit for exact symmetric packed Fisher accumulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from run_triton_gptq_reproducibility import refresh_manifest


ROOT = Path(__file__).resolve().parents[1]
PROFILE_RUNNER = ROOT / "tools" / "run_quant_profile_from_manifest.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(manifest: Path, tag: str, timeout: int) -> dict:
    command = [
        sys.executable, str(PROFILE_RUNNER),
        "--manifest", str(manifest),
        "--tag", tag,
        "--stop-layer", "27",
        "--preclip-search", "legacy",
        "--quantizer-search", "legacy",
        "--timeout-seconds", str(timeout),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    run_dir = ROOT / "outputs" / f"phase_profile_{tag}"
    status = json.loads((run_dir / "status.json").read_text())
    metrics = status.get("metrics") or {}
    if status.get("returncode") != 0 or not status.get("sources_unchanged"):
        raise RuntimeError(f"profile failed or source changed: {run_dir}")
    if not metrics.get("state_sha256") or not metrics.get("peak_allocated"):
        raise RuntimeError(f"missing full run metrics: {run_dir}")
    log = (run_dir / "run.log").read_text(errors="replace")
    matches = re.findall(
        r"Exact KL&PPL on wikitext2:\s*([0-9.eE+-]+),\s*([0-9.eE+-]+)", log
    )
    if not matches:
        raise RuntimeError(f"missing final KL/PPL in {run_dir / 'run.log'}")
    return {
        "tag": tag,
        "wall_seconds": status["wall_seconds"],
        "quantization_seconds": metrics.get("quantization_seconds"),
        "state_sha256": metrics["state_sha256"],
        "state_tensors": metrics.get("state_tensors"),
        "peak_allocated_bytes": metrics["peak_allocated"],
        "peak_reserved_bytes": metrics.get("peak_reserved"),
        "kl": float(matches[-1][0]),
        "ppl": float(matches[-1][1]),
        "run_dir": str(run_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Matched local Qwen3-0.6B baseline manifest")
    parser.add_argument("--tag-prefix", default="fisher-packed-qwen3-06b")
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument("--trace-layer-outputs", action="store_true")
    args = parser.parse_args()
    if not args.tag_prefix.replace("-", "").isalnum():
        parser.error("tag-prefix may contain only letters, numbers, and hyphens")

    campaign = ROOT / "outputs" / f"{args.tag_prefix}-campaign"
    campaign.mkdir(parents=True, exist_ok=False)
    source = Path(args.manifest).resolve()
    control_manifest = campaign / "control.json"
    candidate_manifest = campaign / "packed-candidate.json"
    refresh_manifest(
        source, control_manifest, trace_layer_outputs=args.trace_layer_outputs
    )
    refresh_manifest(
        source, candidate_manifest, trace_layer_outputs=args.trace_layer_outputs
    )
    candidate = json.loads(candidate_manifest.read_text())
    candidate["command"].append("--static_fisher_packed_accumulation")
    candidate["source_sha256"] = {
        rel: sha256(ROOT / rel)
        for rel in (
            "process_args.py",
            "gptq_utils/gptq_plus_utils.py",
            "gptq_utils/fisher_storage.py",
            "utils/loss_utils.py",
            "utils/memory_utils.py",
            "ptq.py",
        )
        if (ROOT / rel).is_file()
    } | {
        key: value for key, value in candidate.get("source_sha256", {}).items()
        if Path(key).is_file() and campaign in Path(key).parents
    }
    candidate_manifest.write_text(json.dumps(candidate, indent=2))

    control = run(control_manifest, f"{args.tag_prefix}-control", args.timeout_seconds)
    packed = run(candidate_manifest, f"{args.tag_prefix}-packed", args.timeout_seconds)
    for key in ("state_sha256", "state_tensors", "kl", "ppl"):
        if control[key] != packed[key]:
            raise RuntimeError(
                f"packed Fisher changed {key}: control={control[key]} candidate={packed[key]}"
            )
    if args.trace_layer_outputs:
        traces = []
        for result in (control, packed):
            path = Path(result["run_dir"]) / "layer_output_fingerprints.json"
            traces.append(json.loads(path.read_text()))
        if traces[0] != traces[1]:
            mismatch = next(
                (i for i, (a, b) in enumerate(zip(*traces)) if a != b), None
            )
            raise RuntimeError(f"layer output traces differ at layer {mismatch}")

    report = {
        "campaign": str(campaign),
        "source_manifest": str(source),
        "trace_layer_outputs": args.trace_layer_outputs,
        "exact_state_kl_ppl": True,
        "control": control,
        "packed": packed,
        "allocated_peak_reduction_percent": 100.0 * (
            1.0 - packed["peak_allocated_bytes"] / control["peak_allocated_bytes"]
        ),
        "reserved_peak_reduction_percent": 100.0 * (
            1.0 - packed["peak_reserved_bytes"] / control["peak_reserved_bytes"]
        ) if control["peak_reserved_bytes"] else None,
        "finished": time.strftime("%F %T"),
    }
    (campaign / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
