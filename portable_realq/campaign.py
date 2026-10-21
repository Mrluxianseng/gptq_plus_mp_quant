"""Explicit prepare/run commands for a serial two-GPU ablation campaign.

No rental, automatic polling, experiment retry, parameter tuning, or deletion.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import shutil

from . import resources, sources
from .observe import sample
from .protocol import cache_identity, jobs, make_config, validate_scope
from .storage import atomic_json, digest, exclusive_lock, file_manifest, identity, read_json, verify_files

REPO = Path(__file__).resolve().parents[1]
RUN_ENV = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "REALQ_DETERMINISTIC_SDPA": "1",
    "NCCL_P2P_LEVEL": "LOC", "NCCL_P2P_DISABLE": "0", "NCCL_SHM_DISABLE": "0", "NCCL_DEBUG": "INFO",
    "PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4"}


def runtime_info(devices: list[str]) -> dict:
    result = subprocess.run(["nvidia-smi", "--id=" + ",".join(devices),
        "--query-gpu=uuid,name,memory.total,driver_version", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=15)
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if len(rows) != 2 or {row.split(",")[0].strip() for row in rows} != set(devices):
        raise RuntimeError("Actual two-device UUID inventory differs")
    packages = {name: importlib.metadata.version(name) for name in
        ("torch", "transformers", "accelerate", "datasets", "huggingface-hub", "safetensors", "triton", "lm-eval", "fast-hadamard-transform")}
    expected = {"torch": "2.9.1", "transformers": "4.56.2", "accelerate": "1.12.0", "datasets": "3.6.0", "lm-eval": "0.4.4"}
    if any(packages[name].split("+")[0] != version for name, version in expected.items()):
        raise RuntimeError("Pinned runtime required; see portable_requirements.txt")
    if "+cu128" not in packages["torch"]:
        raise RuntimeError("Install the pinned CUDA 12.8 torch wheel, not CPU/CUDA13 wheels")
    return {"gpus": rows, "packages": packages, "python": sys.version, "environment": RUN_ENV}


def require_idle(devices: list[str]):
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=15)
    active = [row for row in result.stdout.splitlines() if row.split(",")[0].strip() in devices]
    if active:
        raise RuntimeError("Selected GPUs have active compute processes; no process will be signalled")


def cpu_import_preflight(source: Path) -> dict:
    """Check the actual selected native import closure with CUDA hidden.

    Missing/incompatible optional packages must fail BEFORE model downloads
    or GPU execution, instead of surfacing after paid quantization work.
    """
    code = ("import sys; sys.path.insert(0, " + repr(str(source)) + "); "
            "import torch; from realq import config, ptq, pipeline, precompute, runner; "
            "from realq.utils import data_utils, eval_utils, checkpoint_utils, dist_utils; "
            "from realq.runner import layer_loop; "
            "assert not torch.cuda.is_initialized(); print('cpu-import-preflight-ok')")
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO,
        env={**os.environ, **RUN_ENV, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True, text=True, timeout=120)
    if result.returncode != 0 or "cpu-import-preflight-ok" not in result.stdout:
        raise RuntimeError("Native CPU import preflight failed. Use a clean pinned environment.\n" + result.stderr[-6000:])
    return {"native_imports_completed": True, "CUDA_initialized": False}


def code_bindings() -> dict:
    result = {}
    for path in sorted((REPO / "portable_realq").glob("*")):
        if path.suffix in (".py", ".json"):
            result[path.relative_to(REPO).as_posix()] = digest(path)
    for name in ("manifest.json", "ablations.json"):
        result["vendor/official_v4/" + name] = digest(sources.VENDOR / name)
    return result


def prepare(args) -> Path:
    validate = {"world_size": 2, "devices": args.devices, "models": args.models, "variants": args.variants,
                "jobs": jobs(args.models, args.variants), "qa": args.qa, "qa_batch_size": 16, "qa_tasks": list(resources.TASKS)}
    validate_scope(validate)
    if os.name != "posix":
        raise RuntimeError("Prepare the real GPU campaign on its target Linux host")
    root = args.root.resolve()
    if root.is_relative_to(REPO):
        raise RuntimeError("Use a persistent data root outside the Git checkout")
    root.mkdir(parents=True, exist_ok=True)
    # A floor, not a predicted upper bound: actual model/cache sizes are checked
    # again before launch. No files are removed to satisfy this check.
    if shutil.disk_usage(root).free < 20 * 2**30:
        raise RuntimeError("At least 20 GiB free persistent disk required before downloading")
    shared = root / "shared"
    os.environ.update(resources.environment(shared))
    os.environ.update(RUN_ENV)
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "0"
    runtime = runtime_info(args.devices)
    source_rows = {}
    for variant in args.variants:
        path, sha = sources.materialize(shared, variant)
        source_rows[variant] = {"path": str(path), "sha256": sha}
    import_admission = cpu_import_preflight(Path(source_rows["full_v4"]["path"]))
    # Every failed download records the pinned resource and leaves partial chunks.
    # Users rerun prepare to continue downloads; run never retries an experiment.
    assets = {}
    for model in args.models:
        assets[model] = resources.resolve(shared, model, local=getattr(args, model))
    assets["wikitext2"] = resources.resolve(shared, "wikitext2", local=args.wikitext)
    if args.qa == "full":
        qa = resources.prepare_qa(shared)
    else:
        qa = {"tasks": list(resources.TASKS), "not_executed": True, "scope": "core ablation only; no new QA acceptance"}
    campaign_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = root / "runs" / campaign_id
    output.mkdir(parents=True, exist_ok=False)
    caches = {}
    for model in args.models:
        definition = cache_identity(model, assets, 2, runtime)
        caches[model] = {"key": identity(definition), "definition": definition}
    plan = {"schema": "portable-two-gpu-plan-v1", **validate,
        "root": str(root), "shared": str(shared), "output": str(output), "assets": assets,
        "sources": source_rows, "caches": caches, "runtime": runtime, "qa_preparation": qa,
        "code_bindings": code_bindings(), "timeout_seconds": args.timeout, "CPU_import_admission": import_admission,
        "cache_policy": "producer-once; immutable hit consumers; static-off/baseline fresh without duplicate saves",
        "checkpoint_policy": "content-addressed shared checkpoint, independent per-arm references",
        "topology_scope": "two-GPU results only; local-percentile populations and collective arithmetic can differ from four GPUs",
        "GPU_executed": False, "historical_formal_acceptance_inherited": False}
    atomic_json(output / "plan.json", plan)
    return output / "plan.json"


def wait_owned(command: list[str], output: Path, env: dict, timeout: float, devices: list[str]) -> dict:
    """Always reap our Popen. Never scan/kill unrelated or preexisting processes."""
    import psutil
    if os.name != "posix":
        raise RuntimeError("GPU execution requires Linux process-group isolation")
    stop = threading.Event()
    sample_errors = []
    def observe():
        try:
            sample(os.getpid(), devices, output / "host_nvml.jsonl", stop)
        except BaseException as exc:
            sample_errors.append(type(exc).__name__)
    with (output / "console.log").open("x", encoding="utf-8") as log:
        start = time.perf_counter()
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {"command": command, "pid": process.pid,
                   "start_utc": datetime.now(timezone.utc).isoformat(), "start_monotonic": start}
        sampler = None
        interrupted = timed_out = False
        try:
            receipt["create_time"] = psutil.Process(process.pid).create_time()
            atomic_json(output / "launch.json", receipt)
            sampler = threading.Thread(target=observe, daemon=True)
            sampler.start()
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        except BaseException:
            interrupted = True
            raise
        finally:
            if process.poll() is None:
                try: os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                try: process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try: os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    process.wait()
            stop.set()
            if sampler is not None and sampler.ident is not None: sampler.join(timeout=15)
            sampled = False
            sample_path = output / "host_nvml.jsonl"
            if sample_path.exists():
                with sample_path.open(encoding="utf-8") as samples:
                    for line in samples:
                        try:
                            row = json.loads(line)
                            sampled = {v.split(",")[0].strip() for v in row.get("nvml", [])} == set(devices)
                            if sampled: break
                        except (ValueError, KeyError): pass
            receipt.update(exit_code=process.returncode, timed_out=timed_out, interrupted=interrupted,
                wall_seconds=time.perf_counter() - start, end_utc=datetime.now(timezone.utc).isoformat(),
                sampler_stopped=sampler is None or not sampler.is_alive(),
                sampler_errors=sample_errors, telemetry_available=sampled and not sample_errors)
            atomic_json(output / "wait.json", receipt)
    return receipt


def quality(control: dict, candidate: dict) -> dict:
    checks = {"state_exact": control["state_sha256"] == candidate["state_sha256"]}
    for name in ("kl", "ppl"):
        left, right = control[name], candidate[name]
        checks[name + "_tolerance"] = math.isfinite(left) and math.isfinite(right) and abs(right - left) <= 1e-7 + 1e-6 * abs(left)
    checks["qa_exact"] = control.get("qa") == candidate.get("qa")
    return {"checks": checks, "valid": all(checks.values())}


def runtime_budget(shared: Path, assets: dict, devices: list[str]):
    """Read-only resource gate. Capacity admission is NOT an OOM guarantee."""
    import psutil
    memory = psutil.virtual_memory()
    cgroup_limit = Path("/sys/fs/cgroup/memory.max")
    if cgroup_limit.is_file():
        text = cgroup_limit.read_text().strip()
        used = Path("/sys/fs/cgroup/memory.current")
        if text.isdigit() and used.is_file():
            available = min(memory.available, max(0, int(text) - int(used.read_text())))
        else:
            available = memory.available
    else:
        available = memory.available
        v1_limit = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        v1_used = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        if v1_limit.is_file() and v1_used.is_file():
            available = min(available, max(0, int(v1_limit.read_text()) - int(v1_used.read_text())))
    if available < 128 * 2**30:
        raise RuntimeError("Need at least 128 GiB actually available host RAM for this two-rank candidate; no swap fallback")
    free = shutil.disk_usage(shared).free
    # Conservative space for single shared statistics/reference/rotation caches,
    # unique quantized states, and retained telemetry. Not a capacity proof.
    source_bytes = sum(row["bytes"] for name, asset in assets.items() if name in resources.MODELS for row in asset["files"].values())
    required = max(40 * 2**30, 4 * source_bytes)
    if free < required:
        raise RuntimeError("Insufficient persistent disk headroom; no cache or result will be deleted")
    return {"host_available_bytes": available, "disk_free_bytes": free,
            "disk_floor_bytes": required, "GPU_capacity_guaranteed": False}


def run(plan_path: Path):
    import psutil
    plan = read_json(plan_path)
    if plan.get("schema") != "portable-two-gpu-plan-v1":
        raise RuntimeError("Versioned prepared plan required")
    validate_scope(plan)
    if plan["code_bindings"] != code_bindings():
        raise RuntimeError("Execution code changed after prepare")
    if runtime_info(plan["devices"]) != plan["runtime"]:
        raise RuntimeError("Runtime/GPU identity changed after prepare")
    output, shared = Path(plan["output"]), Path(plan["shared"])
    if plan_path.resolve() != output / "plan.json":
        raise RuntimeError("Plan output identity differs")
    if (output / "started.json").exists():
        raise RuntimeError("This experiment was already attempted. No automatic retry; prepare a new campaign")
    sources.verify_vendor()
    for name, row in plan["assets"].items():
        verify_files(Path(row["path"]), row["files"])
    completed = False
    controls = {}
    with ExitStack() as leases:
        leases.enter_context(exclusive_lock(shared / "locks/campaign.lock"))
        for device in sorted(plan["devices"]):
            leases.enter_context(exclusive_lock(Path(tempfile.gettempdir()) / "realq-device-leases" / (device + ".lock")))
        require_idle(plan["devices"])
        budget = runtime_budget(shared, plan["assets"], plan["devices"])
        atomic_json(output / "resource_admission.json", budget)
        atomic_json(output / "started.json", {"actual_pid": os.getpid(), "plan_sha256": digest(plan_path), "GPU_launched": False})
        try:
            for index, job in enumerate(plan["jobs"]):
                model, variant = job["model"], job["variant"]
                require_idle(plan["devices"])
                path, source_hash = sources.materialize(shared, variant)
                if plan["sources"][variant] != {"path": str(path), "sha256": source_hash}:
                    raise RuntimeError("Prepared source differs")
                cache = shared / "stage0" / plan["caches"][model]["key"]
                with exclusive_lock(shared / "locks" / ("stage0-" + plan["caches"][model]["key"] + ".lock")):
                    hit = (cache / "complete.json").exists()
                    if hit:
                        cache_receipt = read_json(cache / "complete.json")
                        if cache_receipt["definition"] != plan["caches"][model]["definition"]:
                            raise RuntimeError("Shared cache semantic scope differs")
                        verify_files(cache, cache_receipt["files"])
                    elif cache.exists() and any(cache.iterdir()):
                        raise RuntimeError("Unsealed shared cache from an interrupted producer; inspect it, do not overwrite")
                    cache.mkdir(parents=True, exist_ok=True)
                    arm = output / f"{index:02d}-{model}-{variant}"
                    arm.mkdir(exist_ok=False)
                    cfg = make_config(model, plan["assets"], cache, arm, qa=plan["qa"], cache_hit=hit, save_checkpoint=True)
                    spec = {"schema": "portable-two-gpu-worker-v1", **job, "world_size": 2,
                        "devices": plan["devices"], "source": str(path), "source_sha256": source_hash,
                        "assets": plan["assets"], "shared": str(shared), "cache": str(cache), "cache_hit": hit,
                        "config": cfg, "qa": plan["qa"], "output": str(arm), "parent_pid": os.getpid(),
                        "parent_create_time": psutil.Process().create_time(), "plan_sha256": digest(plan_path)}
                    atomic_json(arm / "spec.json", spec)
                    env = {**os.environ, **resources.environment(shared), **RUN_ENV,
                        "CUDA_VISIBLE_DEVICES": ",".join(plan["devices"]), "PYTHONPATH": str(REPO),
                        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
                    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                        "--max_restarts=0", "-m", "portable_realq.runtime", "--spec", str(arm / "spec.json")]
                    wait = wait_owned(command, arm, env, plan["timeout_seconds"], plan["devices"])
                    if wait["exit_code"] != 0 or wait["timed_out"] or not wait["sampler_stopped"] or not wait["telemetry_available"]:
                        raise RuntimeError("Arm failed; serial campaign stopped: " + arm.name)
                    if hit:
                        verify_files(cache, cache_receipt["files"])
                    ranks = [read_json(arm / f"rank{i}/result.json") for i in range(2)]
                    if any(not r["execution_completed"] or r["world_size"] != 2 or r["layers"] != list(range(28)) for r in ranks):
                        raise RuntimeError("Full rank coverage missing")
                    if len({r["stages"]["calibration_sha256"] for r in ranks}) != 1:
                        raise RuntimeError("Ranks consumed different calibration tokens")
                    metrics = ranks[0]["metrics"]
                    state = read_json(arm / "state.json")
                    if identity(state) != metrics["state_sha256"] or len(state["state"]) != 535 or len(state["online_hadamard"]) != 28:
                        raise RuntimeError("Complete state/Hadamard evidence missing or inconsistent")
                    if variant == "full_v4": controls[model] = metrics
                    review = quality(controls[model], metrics)
                    atomic_json(arm / "quality.json", review)
                    if not review["valid"]:
                        raise RuntimeError("Same-topology numerical comparison failed; no next arm")
                    if not hit:
                        if variant != "full_v4":
                            raise RuntimeError("Only the full_v4 control may publish a shared cache")
                        files = file_manifest(cache, exclude=("tokens.lock",))
                        if not any(name.startswith("static/") for name in files) or "calibration.pt" not in files:
                            raise RuntimeError("Actual calibration and full static cache files required")
                        atomic_json(cache / "complete.json", {"definition": plan["caches"][model]["definition"],
                            "files": files, "rank_statistics": {str(r["rank"]): r["stages"]["statistics_sha256"] for r in ranks},
                            "producer": str(arm)})
                    atomic_json(arm / "complete.json", {"execution_completed": True, "quality_valid": True,
                        "cache_reused": hit, "core_scope": "28 layers + KL/PPL", "qa_scope": plan["qa"],
                        "actual_parent_wall_seconds": wait["wall_seconds"],
                        "two_gpu_exclusive_assumption_gpu_hours": 2 * wait["wall_seconds"] / 3600,
                        "historical_formal_acceptance": False})
            completed = True
        finally:
            atomic_json(output / "status.json", {"execution_completed": completed,
                "automatic_retry": False, "historical_four_gpu_acceptance": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", allow_abbrev=False)
    prep.add_argument("--root", type=Path, required=True, help="Persistent writable disk outside Git")
    prep.add_argument("--devices", nargs=2, required=True, help="Two complete GPU UUIDs from nvidia-smi -L")
    prep.add_argument("--models", nargs="+", default=["q06", "q17"], choices=resources.MODELS)
    prep.add_argument("--variants", nargs="+", default=list(sources.DEFAULT_VARIANTS), choices=sources.VARIANTS)
    prep.add_argument("--qa", choices=("full", "none"), default="full", help="none explicitly limits acceptance to core ablation")
    prep.add_argument("--q06", type=Path)
    prep.add_argument("--q17", type=Path)
    prep.add_argument("--wikitext", type=Path)
    prep.add_argument("--timeout", type=float, default=24 * 3600)
    launch = commands.add_parser("run", allow_abbrev=False)
    launch.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if not math.isfinite(args.timeout) or args.timeout <= 0: parser.error("timeout must be positive and finite")
        print(prepare(args))
    else:
        run(args.plan)


if __name__ == "__main__":
    main()
