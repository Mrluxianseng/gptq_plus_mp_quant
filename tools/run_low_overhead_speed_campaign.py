"""Repeat a numerically accepted memory pair with reduced timing overhead.

Requires accepted complete-profile references and unchanged model sources.
No layer-copy/memory/event profiler runs in the measured quantization call.
State SHA and KL/PPL remain mandatory. Whole process wall includes audits;
only synchronized quantization_seconds is used for speed comparisons.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_full_pipeline_memory_campaign import CombinedLog, run_logged
from tools.run_full_pipeline_memory_profile import set_option
from tools.compare_full_pipeline_profiles import cache_hashes

PROFILE_FLAGS = {"--enable_memory_profile", "--enable_quant_profile"}
PROFILE_VALUES = {"--quant_profile_target_layers", "--quant_profile_target_modules"}


def without_profile(command):
    result, index = [], 0
    while index < len(command):
        item = command[index]
        if item in PROFILE_VALUES:
            if index+1 >= len(command) or command[index+1].startswith("--"):
                raise ValueError(f"missing diagnostic value for {item}")
            index += 2
            continue
        if item not in PROFILE_FLAGS:
            result.append(item)
        index += 1
    return result


def core_sources(reference):
    result = {}
    for name, digest in reference["source_sha256_at_launch"].items():
        relative = name.replace("\\", "/").split("gptq_plus_realq/")[-1]
        if relative in {"ptq.py", "process_args.py"} or relative.startswith(("gptq_utils/", "utils/")):
            result[relative] = digest
    if not result:
        raise ValueError("reference has no model-source provenance")
    return result


def check_sources(expected):
    changes = [name for name, digest in expected.items()
               if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest]
    if changes:
        raise ValueError(f"model implementation changed since exactness validation: {changes}")


def reference(directory):
    status = json.loads((directory / "status.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    if not status.get("valid") or status.get("scope") != "full_pipeline":
        raise ValueError("speed reference must be a successful complete-model profile")
    if not (status.get("state_hashes_match_across_ranks") and status.get("layer_fingerprints_match_across_ranks")):
        raise ValueError("reference did not establish all-rank agreement")
    if len(status.get("state_metrics_by_rank", [])) != status["expected_ranks"]["stage2"]:
        raise ValueError("reference rank metrics are incomplete")
    return status, manifest


def bind_reference_audit(audit, control, candidate):
    if (audit.get("scope") != "full_pipeline" or not audit.get("quantization_and_quality_validated")
            or not audit.get("full_model_quantization_validated")):
        raise ValueError("audit must establish complete-model quality")
    for side, directory in (("control", control), ("candidate", candidate)):
        if Path(audit[f"{side}_directory"]).resolve() != directory.resolve():
            raise ValueError(f"{side} reference does not belong to the supplied audit")


def validate_cache_identity(actual, expected):
    if not expected or actual != expected:
        raise RuntimeError("fresh static statistics or prepared model differ from the validated reference")


def run_arm(out, tag, status, manifest, expected_sources, timeout, log, expected_cache):
    destination = out / tag
    destination.mkdir()
    entry = destination / "entry.py"
    entry.write_bytes((ROOT / "tools/realq_low_overhead_speed_entry.py").read_bytes())
    # Pass only task-owned overrides. Never expand or log the inherited
    # environment: it may contain unrelated credentials.
    env_command = ["env", "-u", "REALQ_PHASE_PEAK_MODE", "-u", "GPTQ_PLUS_WALL_PROFILE_JSON",
                   f"RESEARCH_RUN_DIR={destination}", "GPTQ_PLUS_WALL_PROFILE=0",
                   "HF_HUB_OFFLINE=1", "HF_DATASETS_OFFLINE=1", "TRANSFORMERS_OFFLINE=1"]
    commands = {}
    stage_times = {}
    for stage in ("stage0", "stage2"):
        check_sources(expected_sources)
        command = without_profile(manifest[f"{stage}_command"])
        command[command.index("baseline-profile")-1] = str(entry)
        set_option(command, "--exp", f"speed_{tag}_{stage}")
        set_option(command, "--static_cache_path", str(destination / "static_cache"))
        commands[stage] = command
        (out / "active_run.json").write_text(json.dumps({"tag": tag, "stage": stage, "directory": str(destination)}, indent=2))
        started = time.monotonic()
        # run_logged forwards controller output; the worker writes directly to
        # its stdout stream here, so there is only one authoritative live log.
        code = run_logged([*env_command, "timeout", "--signal=TERM", "--kill-after=30s", str(timeout), *command], destination, log)
        stage_times[stage] = time.monotonic()-started
        if code:
            raise RuntimeError(f"{tag} {stage} failed: {code}")
        check_sources(expected_sources)
        if stage == "stage0":
            # Outside both CUDA-synchronized speed timers. Stage-0 hooks differ
            # from the diagnostic entry, so its complete cache must be checked
            # independently rather than inferred from the final state alone.
            log.write(f"[speed-audit] checking fresh Stage-0 cache for {tag}\n")
            observed = cache_hashes(destination, progress=lambda text: log.write(f"[speed-audit] {text}\n"))
            validate_cache_identity(observed, expected_cache)
            (destination / "static_cache_identity.json").write_text(json.dumps(
                {"exact": True, "files": observed, "outside_speed_timers": True}, indent=2))
    ranks = status["expected_ranks"]["stage2"]
    metrics = [json.loads((destination / f"speed_metrics_rank{rank}.json").read_text()) for rank in range(ranks)]
    quality = [json.loads((destination / f"quality_rank{rank}.json").read_text()) for rank in range(ranks)]
    static_times = [json.loads((destination / f"static_collection_timing_rank{rank}.json").read_text())
                    for rank in range(status["expected_ranks"]["stage0"])]
    for rank, row in enumerate(static_times):
        seconds = row["static_collection_seconds"]
        if row["rank"] != rank or not math.isfinite(seconds) or seconds <= 0:
            raise RuntimeError("missing or invalid all-rank static collector timing")
    for expected_rank, (metric, evaluated) in enumerate(zip(metrics, quality)):
        if metric["rank"] != expected_rank:
            raise RuntimeError(f"{tag} rank file identity mismatch")
        if metric["state_sha256"] != status["metrics"]["state_sha256"] or metric["state_tensors"] != status["metrics"]["state_tensors"]:
            raise RuntimeError(f"{tag} end-state differs from its validated memory reference")
        if not math.isfinite(metric["quantization_seconds"]) or metric["quantization_seconds"] <= 0:
            raise RuntimeError("invalid speed timer")
        if len(evaluated) != 1 or evaluated[0] != {"kl": status["kl"], "ppl": status["ppl"]}:
            raise RuntimeError(f"{tag} KL/PPL differs from its validated reference; only one eval dataset is supported")
    record = {"tag": tag, "commands": commands, "metrics_by_rank": metrics, "quality_by_rank": quality,
              "static_collection_timing_by_rank": static_times,
              "static_collection_seconds_slowest_rank": max(row["static_collection_seconds"] for row in static_times),
              "fresh_static_cache_exact_to_reference": True,
              "quantization_seconds_slowest_rank": max(row["quantization_seconds"] for row in metrics),
              "audit_including_stage_wall_seconds": stage_times,
              "whole_process_wall_is_not_uninstrumented_latency": True,
              "source_sha256": expected_sources, "numerical_reference_validated": True}
    (destination / "status.json").write_text(json.dumps(record, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-reference", type=Path, required=True)
    parser.add_argument("--candidate-reference", type=Path, required=True)
    parser.add_argument("--pair-audit", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument("--gpu-indices", default=None, help="Physical GPU list for idle-device preflight; defaults to all GPUs.")
    parser.add_argument("--allow-gpu-contention", action="store_true", help="Diagnostic override; resulting timing must be treated as contended.")
    args = parser.parse_args()
    if args.pairs <= 0 or args.timeout_seconds <= 0 or not args.tag.replace("-", "").isalnum():
        parser.error("positive counts/timeouts and an alphanumeric/hyphen tag are required")
    audit = json.loads(args.pair_audit.read_text())
    if not audit["exactness_passed"] or not audit["strict_measured_memory_gate_passed"]:
        parser.error("only a numerically exact, memory-accepted pair can enter speed reproduction")
    if audit.get("scope") == "stage0_only":
        parser.error("Stage-0-only does not validate full-model speed reproduction")
    bind_reference_audit(audit, args.control_reference, args.candidate_reference)
    if not args.allow_gpu_contention:
        query = ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
        if args.gpu_indices:
            query += [f"--id={args.gpu_indices}"]
        for _ in range(3):
            values = subprocess.check_output(query, text=True).splitlines()
            if not values or any(int(value.strip()) > 3 for value in values):
                raise RuntimeError("GPU is active; stop unrelated workloads before a speed campaign")
            time.sleep(.25)
    references = {"control": reference(args.control_reference.resolve()), "candidate": reference(args.candidate_reference.resolve())}
    states = [value[0] for value in references.values()]
    for field in ("kl", "ppl"):
        if states[0][field] != states[1][field] or not math.isfinite(states[0][field]):
            raise ValueError("reference pair quality is not exact")
    if any(states[0]["metrics"][key] != states[1]["metrics"][key] for key in ("state_sha256", "state_tensors")):
        raise ValueError("reference pair end-state is not exact")
    sources = [core_sources(value[1]) for value in references.values()]
    if sources[0] != sources[1]:
        raise ValueError("reference source versions differ")
    expected = sources[0]
    check_sources(expected)
    reference_caches = {side: cache_hashes(directory) for side, directory in
                        (("control", args.control_reference), ("candidate", args.candidate_reference))}
    validate_cache_identity(reference_caches["control"], reference_caches["candidate"])
    out = ROOT / "outputs" / f"low_overhead_speed_{args.tag}"
    out.mkdir(parents=True, exist_ok=False)
    manifest = {**vars(args), "source_sha256": expected, "speed_entry_sha256": hashlib.sha256((ROOT/"tools/realq_low_overhead_speed_entry.py").read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    log = CombinedLog(out / "combined.log")
    records = []
    valid = False
    try:
        for pair in range(1, args.pairs+1):
            arms = {}
            order = ("control", "candidate") if pair % 2 else ("candidate", "control")
            for side in order:
                state, launch = references[side]
                arms[side] = run_arm(out, f"{args.tag}-r{pair}-{side}", state, launch, expected, args.timeout_seconds, log,
                                     reference_caches[side])
            c, n = (arms[side]["quantization_seconds_slowest_rank"] for side in ("control", "candidate"))
            sc, sn = (arms[side]["static_collection_seconds_slowest_rank"] for side in ("control", "candidate"))
            records.append({"pair": pair, "order": order, "control_seconds": c, "candidate_seconds": n, "paired_speedup": c/n,
                            "control_static_collection_seconds": sc, "candidate_static_collection_seconds": sn,
                            "paired_static_collection_speedup": sc/sn,
                            "control_collector_plus_quantization_seconds": sc+c,
                            "candidate_collector_plus_quantization_seconds": sn+n})
            (out / "paired_repetitions.json").write_text(json.dumps(records, indent=2))
            log.write("[speed] audited pair: " + json.dumps(records[-1]) + "\n")
        valid = True
        summary = {"pairs": len(records), "all_exact_to_memory_references": True,
                   "control_mean_seconds": statistics.mean(row["control_seconds"] for row in records),
                   "candidate_mean_seconds": statistics.mean(row["candidate_seconds"] for row in records),
                   "mean_paired_speedup": statistics.mean(row["paired_speedup"] for row in records),
                   "min_paired_speedup": min(row["paired_speedup"] for row in records),
                   "max_paired_speedup": max(row["paired_speedup"] for row in records),
                   "fresh_static_caches_exact_all_arms": True,
                   "control_mean_static_collection_seconds": statistics.mean(row["control_static_collection_seconds"] for row in records),
                   "candidate_mean_static_collection_seconds": statistics.mean(row["candidate_static_collection_seconds"] for row in records),
                   "control_mean_collector_plus_quantization_seconds": statistics.mean(row["control_collector_plus_quantization_seconds"] for row in records),
                   "candidate_mean_collector_plus_quantization_seconds": statistics.mean(row["candidate_collector_plus_quantization_seconds"] for row in records),
                   "scope": "synchronized full quantization call without diagnostic layer/phase/event profiling; state audit after timer"}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        log.write("[speed] summary: " + json.dumps(summary) + "\n")
    finally:
        (out / "status.json").write_text(json.dumps({"valid": valid, "requested_pairs": args.pairs, "completed_pairs": len(records)}, indent=2))
        log.close()


if __name__ == "__main__":
    main()
