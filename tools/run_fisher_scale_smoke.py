"""Small paired Stage-0/Fisher scaling audit on a selected local model."""

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
    "gptq_utils/fisher_storage.py",
    "utils/loss_utils.py",
    "utils/memory_utils.py",
    "ptq.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _set_arg(command: list[str], key: str, value: str) -> None:
    if key in command:
        command[command.index(key) + 1] = value
    else:
        command.extend((key, value))


def _run(manifest: Path, tag: str, stop_layer: int, timeout: int) -> dict:
    subprocess.run(
        [
            sys.executable,
            str(PROFILE_RUNNER),
            "--manifest", str(manifest),
            "--tag", tag,
            "--stop-layer", str(stop_layer),
            "--timeout-seconds", str(timeout),
        ],
        cwd=ROOT,
        check=True,
    )
    run_dir = ROOT / "outputs" / f"phase_profile_{tag}"
    status = json.loads((run_dir / "status.json").read_text())
    metrics = status.get("metrics") or {}
    if status.get("returncode") or not status.get("sources_unchanged"):
        raise RuntimeError(f"scale smoke failed: {run_dir}")
    if not metrics.get("state_sha256"):
        raise RuntimeError(f"state hash missing: {run_dir}")
    phase_files = sorted((ROOT / "outputs").glob(
        f"**/phase_profile_{tag}/memory_profile/cuda_memory_quantization_rank*.jsonl"
    ))
    phase_rows = []
    for phase_file in phase_files:
        with phase_file.open() as stream:
            phase_rows.extend(json.loads(line) for line in stream if line.strip())
    phase_peaks = {}
    for row in phase_rows:
        phase = row.get("phase", "unknown")
        peak = phase_peaks.setdefault(phase, {
            "sampled_peak_allocated_bytes": 0,
            "sampled_peak_reserved_bytes": 0,
            "cumulative_peak_allocated_bytes": 0,
            "cumulative_peak_reserved_bytes": 0,
            "max_mark_device_used_bytes": 0,
            "records": 0,
        })
        peak["records"] += 1
        peak["max_mark_device_used_bytes"] = max(
            peak["max_mark_device_used_bytes"],
            int(row.get("start_device_used_bytes", 0) or 0),
            int(row.get("end_device_used_bytes", 0) or 0),
        )
        for key in tuple(peak):
            if key != "records":
                peak[key] = max(peak[key], int(row.get(key, 0) or 0))
    return {
        "tag": tag,
        "run_dir": str(run_dir),
        "quantization_seconds": metrics.get("quantization_seconds"),
        "peak_allocated_bytes": metrics.get("peak_allocated"),
        "peak_reserved_bytes": metrics.get("peak_reserved"),
        "partial_prefix_kl": status.get("partial_prefix_kl"),
        "partial_prefix_ppl": status.get("partial_prefix_ppl"),
        "state_sha256": metrics["state_sha256"],
        "state_tensors": metrics.get("state_tensors"),
        "phase_records": len(phase_rows),
        "phase_peaks": phase_peaks,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--tag-prefix", required=True)
    p.add_argument("--stop-layer", type=int, default=0)
    p.add_argument("--nsamples", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--backward-samples", type=int, default=8)
    p.add_argument(
        "--same-packed-fisher-control",
        action="store_true",
        help="Use packed Fisher in both arms to isolate another candidate-only option.",
    )
    p.add_argument(
        "--static-saved-tensor-offload-mib",
        type=int,
        default=0,
        help="Offload qualifying Stage-0 saved activations in the packed arm only.",
    )
    p.add_argument("--timeout-seconds", type=int, default=7200)
    args = p.parse_args()
    if min(args.stop_layer, args.nsamples, args.seq_len, args.backward_samples,
           args.static_saved_tensor_offload_mib) < 0:
        p.error("stop-layer and sample/sequence sizes must be nonnegative")
    if not Path(args.model_path, "config.json").is_file():
        p.error(f"model config not found under {args.model_path}")

    campaign = ROOT / "outputs" / f"{args.tag_prefix}-campaign"
    campaign.mkdir(parents=True, exist_ok=False)
    source = Path(args.manifest).resolve()
    manifests = []
    for arm, packed in (("control", False), ("packed", True)):
        use_packed = packed or args.same_packed_fisher_control
        manifest_path = campaign / f"{arm}.json"
        refresh_manifest(source, manifest_path, trace_layer_outputs=True)
        data = json.loads(manifest_path.read_text())
        command = data["command"]
        _set_arg(command, "--model", str(Path(args.model_path).resolve()))
        _set_arg(command, "--nsamples", str(args.nsamples))
        _set_arg(command, "--seq_len", str(args.seq_len))
        _set_arg(command, "--backward_samples", str(args.backward_samples))
        _set_arg(command, "--backward_bsz", "1")
        _set_arg(command, "--final_layer_backward_bsz", "1")
        _set_arg(command, "--bsz", "1")
        _set_arg(command, "--final_layer_stats_bsz", "1")
        _set_arg(command, "--eval_seq_len", "128")
        _set_arg(command, "--eval_samples", "2")
        if "--enable_memory_profile" not in command:
            command.append("--enable_memory_profile")
        command[:] = [x for x in command if x != "--static_fisher_packed_accumulation"]
        if use_packed:
            command.append("--static_fisher_packed_accumulation")
        if packed and args.static_saved_tensor_offload_mib:
            command.extend((
                "--static_saved_tensor_offload_mib",
                str(args.static_saved_tensor_offload_mib),
            ))
        entry = next(
            Path(str(x)) for x in command
            if Path(str(x)).name.endswith("entry.py")
        )
        data["source_sha256"] = {
            str((ROOT / rel).resolve()): _sha(ROOT / rel) for rel in TRACKED
        } | {str(entry.resolve()): _sha(entry)}
        data["model_scale_smoke"] = {
            "packed_fisher": use_packed,
            "static_saved_tensor_offload_mib": (
                args.static_saved_tensor_offload_mib if packed else 0
            ),
            "nsamples": args.nsamples,
            "seq_len": args.seq_len,
            "stop_layer": args.stop_layer,
        }
        manifest_path.write_text(json.dumps(data, indent=2))
        manifests.append(manifest_path)

    control = _run(
        manifests[0], f"{args.tag_prefix}-control", args.stop_layer,
        args.timeout_seconds,
    )
    candidate = _run(
        manifests[1], f"{args.tag_prefix}-packed", args.stop_layer,
        args.timeout_seconds,
    )
    traces = [
        json.loads((Path(x["run_dir"]) / "layer_output_fingerprints.json").read_text())
        for x in (control, candidate)
    ]
    exact = (
        control["state_sha256"] == candidate["state_sha256"]
        and control["partial_prefix_kl"] == candidate["partial_prefix_kl"]
        and control["partial_prefix_ppl"] == candidate["partial_prefix_ppl"]
        and traces[0] == traces[1]
    )
    summary = {
        "campaign": str(campaign),
        "source_manifest": str(source),
        "model_path": str(Path(args.model_path).resolve()),
        "nsamples": args.nsamples,
        "seq_len": args.seq_len,
        "stop_layer_inclusive": args.stop_layer,
        "same_packed_fisher_control": args.same_packed_fisher_control,
        "static_saved_tensor_offload_mib": args.static_saved_tensor_offload_mib,
        "control": control,
        "packed": candidate,
        "state_layer_outputs_and_eval_exact": exact,
        "allocated_peak_reduction_percent": 100.0 * (
            1.0 - candidate["peak_allocated_bytes"] / control["peak_allocated_bytes"]
        ),
        "reserved_peak_reduction_percent": 100.0 * (
            1.0 - candidate["peak_reserved_bytes"] / control["peak_reserved_bytes"]
        ) if control["peak_reserved_bytes"] else None,
    }
    (campaign / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if not exact:
        raise RuntimeError("packed Fisher scale smoke changed the model state or layer outputs")


if __name__ == "__main__":
    main()
