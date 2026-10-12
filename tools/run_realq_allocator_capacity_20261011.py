"""Serial full REAL-Q capacity checks at fixed allocator budgets.

Uses already accepted full-pipeline references, fresh calibration statistics,
and full 28-layer numerical audits. A PyTorch allocator budget is not a
physical GPU capacity limit and does not limit NCCL/driver allocations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_full_pipeline_profiles import algorithm_command, cache_hashes
from tools.run_full_pipeline_memory_campaign import stop_process_tree
from tools.run_low_overhead_speed_campaign import core_sources, check_sources

MODELS = (
    ("qwen17b", "qwen17b-checkpoint-ce-matched-full-threepairs-20261010-v6-2125", 6400),
    ("qwen06b", "qwen06b-checkpoint-ce-original-input-full-threepairs-20261010-1457", 4096),
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prefix(budget):
    return f'''# Research-only allocator budget, equal for control/candidate.
import atexit as _capacity_atexit
import json as _capacity_json
import os as _capacity_os
from pathlib import Path as _CapacityPath
import torch as _capacity_torch
_capacity_rank = int(_capacity_os.environ.get("LOCAL_RANK", "0"))
_capacity_torch.cuda.set_device(_capacity_rank)
_capacity_total = _capacity_torch.cuda.get_device_properties(_capacity_rank).total_memory
_capacity_bytes = {budget} * 1024**2
if not 0 < _capacity_bytes < _capacity_total:
    raise RuntimeError("allocator budget must be below visible device memory")
_capacity_fraction = _capacity_bytes / _capacity_total
_capacity_torch.cuda.set_per_process_memory_fraction(_capacity_fraction, _capacity_rank)
_capacity_record = dict(budget_mib={budget}, budget_bytes=_capacity_bytes,
                       total_visible_bytes=_capacity_total, fraction=_capacity_fraction,
                       observed_fraction=_capacity_torch.cuda.get_per_process_memory_fraction(_capacity_rank),
                       torch=str(_capacity_torch.__version__), rank=_capacity_rank,
                       scope="PyTorch caching allocator only; excludes driver/NCCL; not a different GPU")
_capacity_record["observed_budget_bytes"] = round(_capacity_record["observed_fraction"] * _capacity_total)
# Native API converts fractions through integer byte limits; one-byte rounding
# is possible. Compare actual byte limits, not exact floating-point equality.
_capacity_destination = _CapacityPath(_capacity_os.environ["RESEARCH_RUN_DIR"])
_capacity_stage = "stage0" if "--exit_after_precompute" in __import__("sys").argv else "stage2"
_capacity_record_path = _capacity_destination / f"allocator_budget_{{_capacity_stage}}_rank{{_capacity_rank}}.json"
_capacity_record_path.write_text(_capacity_json.dumps(_capacity_record, indent=2))
if abs(_capacity_record["observed_budget_bytes"] - _capacity_bytes) > 1:
    raise RuntimeError("allocator budget was not installed within one byte")
print("[allocator-capacity] " + _capacity_json.dumps(_capacity_record), flush=True)
def _capacity_exit_record():
    try:
        _capacity_record.update(peak_allocated_bytes=_capacity_torch.cuda.max_memory_allocated(_capacity_rank),
                                peak_reserved_bytes=_capacity_torch.cuda.max_memory_reserved(_capacity_rank))
        _capacity_record_path.write_text(_capacity_json.dumps(_capacity_record, indent=2))
    except BaseException as _capacity_error:
        print("[allocator-capacity] exit counters unavailable: " + repr(_capacity_error), flush=True)
_capacity_atexit.register(_capacity_exit_record)
'''


def numerical_audit(reference, actual):
    c, n = read(reference / "status.json"), read(actual / "status.json")
    cm, nm = read(reference / "manifest.json"), read(actual / "manifest.json")
    errors = []
    if not c.get("valid") or not n.get("valid"):
        errors.append("reference or constrained candidate did not complete")
    for field in ("kl", "ppl"):
        if c.get(field) != n.get(field):
            errors.append(field)
    for field in ("state_sha256", "state_tensors"):
        if c.get("metrics", {}).get(field) != n.get("metrics", {}).get(field):
            errors.append(field)
    if core_sources(cm) != core_sources(nm):
        errors.append("model source snapshots")
    for stage in ("stage0", "stage2"):
        if algorithm_command(cm[f"{stage}_command"]) != algorithm_command(nm[f"{stage}_command"]):
            errors.append(stage + " algorithm/data/parallelism command")
    caches = [cache_hashes(folder) for folder in (reference, actual)]
    if not caches[0] or caches[0] != caches[1]:
        errors.append("fresh static cache")
    traces = [read(folder / "layer_output_fingerprints.json") for folder in (reference, actual)]
    if traces[0] != traces[1] or {row["layer"] for row in traces[1]} != set(range(28)):
        errors.append("all 28 held-out layer fingerprints")
    return dict(valid=not errors, errors=errors, reference=str(reference), candidate=str(actual),
                fresh_cache_files=len(caches[1]), cache_sha256=caches[1],
                state_sha256=n.get("metrics", {}).get("state_sha256"), kl=n.get("kl"), ppl=n.get("ppl"),
                numerical_scope="Constrained candidate versus previously accepted unrestricted candidate; not failed control")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repeatability-audit", type=Path, required=True)
    args = parser.parse_args()
    if not args.tag.replace("-", "").isalnum():
        raise ValueError("invalid tag")
    # Finished cross-repeat validation is a dependency, never inferred.
    while not args.repeatability_audit.exists():
        time.sleep(5)
    if not read(args.repeatability_audit).get("valid"):
        raise RuntimeError("larger-model cross-repeat audit failed")
    out = ROOT / "outputs" / args.tag
    out.mkdir(exist_ok=False)
    records, proc = [], None
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for model, accepted, budget in MODELS:
            accepted_dir = ROOT / "outputs" / f"memory_campaign_{accepted}"
            accepted_status = read(accepted_dir / "status.json")
            if not accepted_status.get("valid") or accepted_status.get("completed_pairs") != 3:
                raise RuntimeError("capacity check requires all three complete reference pairs")
            audit = read(accepted_dir / "pair1_audit/comparison.json")
            if not audit.get("exactness_passed") or not audit.get("strict_measured_memory_gate_passed"):
                raise RuntimeError("reference numerical/resource gates failed")
            if shutil.disk_usage(ROOT).free < 10 * 1024**3:
                raise RuntimeError("requires 10 GiB free for independent complete capacity arms")
            folder = out / model
            folder.mkdir()
            references = {side: ROOT / "outputs" / f"full_pipeline_profile_{accepted}-r1-{side}"
                          for side in ("control", "candidate")}
            launch = {side: read(references[side] / "manifest.json") for side in references}
            expected_sources = core_sources(launch["control"])
            if expected_sources != core_sources(launch["candidate"]):
                raise RuntimeError("reference production sources differ")
            check_sources(expected_sources)
            source_manifest = read(launch["candidate"]["stage2_source_manifest"])
            command = source_manifest["command"]
            source = Path(command[command.index("baseline-profile") - 1])
            entry = folder / "entry.py"
            entry.write_text(prefix(budget) + "\n" + source.read_text())
            compile(entry.read_text(), str(entry), "exec")
            result = dict(model=model, budget_mib=budget, reference_campaign=accepted,
                          entry_sha256=sha(entry), model_sources=expected_sources, arms={})
            for side in ("control", "candidate"):
                for stage in ("stage0", "stage2"):
                    data = read(launch[side][f"{stage}_source_manifest"])
                    cmd = data["command"]
                    cmd[cmd.index("baseline-profile") - 1] = str(entry)
                    data["scope_note"] = "Reference algorithm/data unchanged; shared fixed allocator budget; fresh full calibration and quantization"
                    (folder / f"{side}-{stage}.json").write_text(json.dumps(data, indent=2))
                tag = f"{args.tag}-{model}-{side}"
                run = ROOT / "outputs" / f"full_pipeline_profile_{tag}"
                cmd = [sys.executable, "-u", str(ROOT / "tools/run_full_pipeline_memory_profile.py"),
                       "--tag", tag, "--stage0-manifest", str(folder / f"{side}-stage0.json"),
                       "--stage2-manifest", str(folder / f"{side}-stage2.json"),
                       "--gpu-indices", "0", "--timeout-seconds", "7200", "--exact-phase-peaks"]
                (out / "active.json").write_text(json.dumps(dict(label=tag, command=cmd), indent=2))
                print(f"[capacity] {model} {side} starting, budget={budget} MiB", flush=True)
                with (out / f"{tag}.log").open("x") as log:
                    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    code = proc.wait()
                check_sources(expected_sources)
                status_path = run / "status.json"
                state = read(status_path) if status_path.exists() else {}
                budgets = [read(path) for path in run.glob("allocator_budget_*_rank*.json")]
                if not budgets or any(row["budget_mib"] != budget for row in budgets):
                    raise RuntimeError("missing or incorrect worker allocator-budget evidence")
                logs = {stage: (run / f"{stage}.log").read_text(errors="replace")
                        for stage in ("stage0", "stage2") if (run / f"{stage}.log").exists()}
                oom_stages = [stage for stage, text in logs.items() if "CUDA out of memory" in text]
                result["arms"][side] = dict(returncode=code, valid=state.get("valid", False),
                                            cuda_oom_stages=oom_stages, budgets=budgets, directory=str(run))
                (folder / "result.json").write_text(json.dumps(result, indent=2))
                if side == "control" and code and not oom_stages:
                    raise RuntimeError("control failed for an unexpected reason; inspect original logs")
                if side == "candidate":
                    if code or not state.get("valid"):
                        raise RuntimeError("constrained candidate did not complete; inspect original logs")
                    result["candidate_exactness"] = numerical_audit(references[side], run)
                    if not result["candidate_exactness"]["valid"]:
                        raise RuntimeError("constrained candidate differs from validated numerical reference")
                print(f"[capacity] {model} {side} exited code={code}, OOM stages={oom_stages}", flush=True)
            result["capacity_contrast_established"] = bool(result["arms"]["control"]["cuda_oom_stages"])
            result["scope"] = "Fixed allocator-budget contrast; not physical smaller-GPU/fewer-GPU or formal calibration-scale evidence"
            result["finished"] = time.strftime("%F %T")
            (folder / "result.json").write_text(json.dumps(result, indent=2))
            records.append(result)
        status = dict(valid=all(row["capacity_contrast_established"] for row in records), models=records,
                      finished=time.strftime("%F %T"), scope="Two independent full-model allocator-capacity contrasts")
        (out / "status.json").write_text(json.dumps(status, indent=2))
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            stop_process_tree(proc.pid)
            proc.wait()
        (out / "status.json").write_text(json.dumps(dict(valid=False, error=repr(exc), models=records,
                                                        finished=time.strftime("%F %T")), indent=2))
        raise


if __name__ == "__main__":
    main()
