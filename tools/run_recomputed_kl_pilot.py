"""Paired full-pipeline pilot for recomputed, token-chunked lm_head KL."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from run_triton_gptq_reproducibility import refresh_manifest


ROOT = Path(__file__).resolve().parents[1]
PROFILE_RUNNER = ROOT / "tools" / "run_quant_profile_from_manifest.py"
TRACKED = (
    "process_args.py",
    "gptq_utils/gptq_plus_utils.py",
    "utils/loss_utils.py",
    "utils/memory_utils.py",
    "ptq.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(manifest: Path, tag: str, stop_layer: int, timeout: int) -> dict:
    command = [
        sys.executable,
        str(PROFILE_RUNNER),
        "--manifest", str(manifest),
        "--tag", tag,
        "--stop-layer", str(stop_layer),
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
        raise RuntimeError(f"missing quantization metrics: {run_dir}")
    log = (run_dir / "run.log").read_text(errors="replace")
    values = re.findall(
        r"Exact KL&PPL on wikitext2:\s*([0-9.eE+-]+),\s*([0-9.eE+-]+)", log
    )
    if not values:
        raise RuntimeError(f"final KL/PPL missing from {run_dir / 'run.log'}")
    return {
        "tag": tag,
        "run_dir": str(run_dir),
        "quantization_seconds": metrics.get("quantization_seconds"),
        "peak_allocated_bytes": metrics["peak_allocated"],
        "peak_reserved_bytes": metrics.get("peak_reserved"),
        "state_sha256": metrics["state_sha256"],
        "kl": float(values[-1][0]),
        "ppl": float(values[-1][1]),
    }


def _reuse_run(run_dir: Path) -> tuple[dict, list]:
    status = json.loads((run_dir / "status.json").read_text())
    metrics = status.get("metrics") or {}
    if status.get("returncode") != 0 or not status.get("sources_unchanged"):
        raise RuntimeError(f"cannot reuse invalid control run: {run_dir}")
    if not metrics.get("state_sha256") or not metrics.get("peak_allocated"):
        raise RuntimeError(f"reused control metrics are incomplete: {run_dir}")
    values = re.findall(
        r"Exact KL&PPL on wikitext2:\s*([0-9.eE+-]+),\s*([0-9.eE+-]+)",
        (run_dir / "run.log").read_text(errors="replace"),
    )
    if not values:
        raise RuntimeError(f"reused control has no final KL/PPL: {run_dir}")
    trace = json.loads((run_dir / "layer_output_fingerprints.json").read_text())
    return ({
        "tag": run_dir.name.removeprefix("phase_profile_"),
        "run_dir": str(run_dir),
        "quantization_seconds": metrics.get("quantization_seconds"),
        "peak_allocated_bytes": metrics["peak_allocated"],
        "peak_reserved_bytes": metrics.get("peak_reserved"),
        "state_sha256": metrics["state_sha256"],
        "kl": float(values[-1][0]),
        "ppl": float(values[-1][1]),
    }, trace)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tag-prefix", default="recomputed-kl-qwen3-06b")
    parser.add_argument("--stop-layer", type=int, default=27)
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument("--chunk-tokens", type=int, default=16)
    parser.add_argument(
        "--static-saved-tensor-offload-mib",
        type=int,
        default=0,
        help=(
            "Optionally offload Stage-0 saved activations of at least this many MiB "
            "to CPU in the candidate. This is an exact activation-memory tradeoff."
        ),
    )
    parser.add_argument(
        "--packed-fisher",
        action="store_true",
        help="Also use the previously validated exact packed Stage-0 Fisher accumulator.",
    )
    parser.add_argument(
        "--offload-only",
        action="store_true",
        help=(
            "Isolate saved-tensor offload: preserve the source manifest's KL and "
            "Fisher settings, adding only the offload option to the candidate. "
            "Use with a packed-Fisher source manifest and a matching reused control."
        ),
    )
    parser.add_argument(
        "--reuse-control-run",
        help="Existing valid phase-profile run with matching settings and layer fingerprints.",
    )
    args = parser.parse_args()
    if args.stop_layer < 0 or args.chunk_tokens <= 0 or args.static_saved_tensor_offload_mib < 0:
        parser.error("stop-layer and offload MiB must be nonnegative; chunk-tokens positive")
    campaign = ROOT / "outputs" / f"{args.tag_prefix}-campaign"
    campaign.mkdir(parents=True, exist_ok=False)
    source = Path(args.manifest).resolve()
    manifests = []
    for name, candidate in (("control", False), ("recomputed", True)):
        path = campaign / f"{name}.json"
        refresh_manifest(
            source,
            path,
            kl_chunk_tokens=args.chunk_tokens,
            trace_layer_outputs=True,
        )
        manifest = json.loads(path.read_text())
        command = manifest["command"]
        if candidate:
            if not args.offload_only:
                command.append("--kl_recompute_head")
                if args.packed_fisher:
                    command.append("--static_fisher_packed_accumulation")
            if args.static_saved_tensor_offload_mib:
                command.extend((
                    "--static_saved_tensor_offload_mib",
                    str(args.static_saved_tensor_offload_mib),
                ))
        # The source run's provenance predates this pilot. Refresh hashes for
        # the exact implementation under test and the copied entry wrapper.
        tracked = {str((ROOT / rel).resolve()): _sha256(ROOT / rel) for rel in TRACKED}
        entry = next(
            Path(str(value))
            for value in command
            if Path(str(value)).name.endswith("entry.py")
        )
        tracked[str(entry.resolve())] = _sha256(entry)
        manifest["source_sha256"] = tracked
        manifest["pilot_mode"] = (
            "saved-tensor-offload-only" if candidate and args.offload_only
            else "recomputed-lm-head-kl" if candidate
            else "materialized-kl-control"
        )
        if candidate and args.offload_only:
            manifest["pilot_mode"] += "+preserve-source-fisher-and-kl"
        if candidate and args.static_saved_tensor_offload_mib:
            manifest["pilot_mode"] += f"+saved-tensor-offload-{args.static_saved_tensor_offload_mib}mib"
        path.write_text(json.dumps(manifest, indent=2))
        manifests.append(path)

    reused_control_trace = None
    if args.reuse_control_run:
        control, reused_control_trace = _reuse_run(Path(args.reuse_control_run).resolve())
    else:
        control = _run(
            manifests[0], f"{args.tag_prefix}-control", args.stop_layer,
            args.timeout_seconds,
        )
    candidate = _run(
        manifests[1], f"{args.tag_prefix}-recomputed", args.stop_layer,
        args.timeout_seconds,
    )
    traces = [reused_control_trace] if reused_control_trace is not None else []
    results_to_trace = (candidate,) if reused_control_trace is not None else (control, candidate)
    for result in results_to_trace:
        trace_path = Path(result["run_dir"]) / "layer_output_fingerprints.json"
        traces.append(json.loads(trace_path.read_text()))
    trace_exact = traces[0] == traces[1]
    metrics_exact = all(
        control[key] == candidate[key] for key in ("state_sha256", "kl", "ppl")
    )
    summary = {
        "campaign": str(campaign),
        "source_manifest": str(source),
        "stop_layer_inclusive": args.stop_layer,
        "kl_chunk_tokens": args.chunk_tokens,
        "packed_fisher_candidate": args.packed_fisher,
        "static_saved_tensor_offload_mib": args.static_saved_tensor_offload_mib,
        "control": control,
        "candidate": candidate,
        "layer_output_fingerprints_exact": trace_exact,
        "state_kl_ppl_exact": metrics_exact,
        "allocated_peak_reduction_percent": 100.0 * (
            1.0 - candidate["peak_allocated_bytes"] / control["peak_allocated_bytes"]
        ),
        "reserved_peak_reduction_percent": 100.0 * (
            1.0 - candidate["peak_reserved_bytes"] / control["peak_reserved_bytes"]
        ) if control["peak_reserved_bytes"] else None,
    }
    (campaign / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if not trace_exact or not metrics_exact:
        raise RuntimeError("recomputed-KL pilot changed layer outputs or final quantization metrics")


if __name__ == "__main__":
    main()
