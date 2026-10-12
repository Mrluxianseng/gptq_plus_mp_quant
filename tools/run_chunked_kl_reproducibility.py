"""Run matched full-config REAL-Q repetitions and stop on any output mismatch."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_chunked_kl_pilot.py"


def load_result(tag: str) -> dict:
    run_dir = ROOT / "outputs" / f"chunked_kl_pilot_{tag}"
    status_path = run_dir / "status.json"
    metrics_path = run_dir / "baseline-profile_metrics.json"
    log_path = run_dir / "run.log"
    status = json.loads(status_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    log = log_path.read_text(errors="replace")
    match = re.findall(
        r"Exact KL&PPL on wikitext2:\s*([0-9.eE+-]+),\s*([0-9.eE+-]+)", log
    )
    if not match:
        raise RuntimeError(f"missing exact KL/PPL result in {log_path}")
    kl, ppl = map(float, match[-1])
    return {
        "tag": tag,
        "returncode": status["returncode"],
        "wall_seconds": status["wall_seconds"],
        "quantization_seconds": metrics["quantization_seconds"],
        "state_sha256": metrics["state_sha256"],
        "state_tensors": metrics["state_tensors"],
        "peak_allocated_bytes": metrics["peak_allocated"],
        "peak_reserved_bytes": metrics["peak_reserved"],
        "kl": kl,
        "ppl": ppl,
    }


def run(manifest: str, tag: str, chunk_tokens: int, timeout: int) -> dict:
    command = [
        sys.executable,
        str(RUNNER),
        "--manifest",
        manifest,
        "--tag",
        tag,
        "--chunk-tokens",
        str(chunk_tokens),
        "--full-config",
        "--timeout-seconds",
        str(timeout),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return load_result(tag)


def assert_equivalent(control: dict, candidate: dict) -> None:
    keys = ("state_sha256", "state_tensors", "kl", "ppl")
    mismatches = {
        key: (control[key], candidate[key])
        for key in keys
        if control[key] != candidate[key]
    }
    if mismatches:
        raise RuntimeError(f"control/candidate mismatch; stopped campaign: {mismatches}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--tag-prefix", default="kl16-final")
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    args = parser.parse_args()
    if args.pairs <= 0:
        parser.error("--pairs must be positive")
    results = []
    for i in range(2, 2 + args.pairs):
        control = run(args.manifest, f"{args.tag_prefix}-r{i}-control", 0, args.timeout_seconds)
        candidate = run(args.manifest, f"{args.tag_prefix}-r{i}-chunk16", 16, args.timeout_seconds)
        assert_equivalent(control, candidate)
        pair = {"control": control, "candidate": candidate}
        results.append(pair)
        report = ROOT / "outputs" / f"{args.tag_prefix}_paired_repetitions.json"
        report.write_text(json.dumps(results, indent=2))
        print(json.dumps({"completed_pair": i, **pair}, indent=2), flush=True)


if __name__ == "__main__":
    main()
