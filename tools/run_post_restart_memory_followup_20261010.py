"""Serial, gated continuation after the restarted memory and capacity jobs.

Uses fresh statistics, scopes negative results, and expands only mechanisms
whose exactness and measured-memory audits pass. No concurrent GPU workers.
"""
import argparse
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_full_pipeline_memory_campaign import stop_process_tree
from tools.run_full_pipeline_memory_profile import set_option

MAIN_TAG = "qwen06b-checkpoint-ce-full-2x1024-threepairs-20261010-restart1408"
OUT = ROOT / "outputs/post-restart-memory-followup-20261010"


def read(path):
    return json.loads(path.read_text())


def accepted(tag):
    path = ROOT / "outputs" / f"memory_campaign_{tag}/pair1_audit/comparison.json"
    if not path.exists():
        return False
    audit = read(path)
    return audit.get("exactness_passed") and audit.get("strict_measured_memory_gate_passed")


def prepare_composition(folder, method, prefix):
    from tools.prepare_layer_residency_compaction_pilot_20261010 import INJECTION as compact
    from tools.prepare_quantizer_residency_pilot_20261010 import INJECTION as metadata
    folder.mkdir()
    base = (ROOT / "tools/realq_checkpoint_research_entry.py").read_text()
    needle = "original_fwrd = impl.gptq_fwrd"
    if base.count(needle) != 1:
        raise RuntimeError("entry injection boundary changed")
    injection = metadata if method == "metadata" else compact
    entry = folder / "entry.py"
    base = base.replace(needle, injection + "\n" + needle)
    if method == "compact":
        inner = "    result = original_fwrd(args, analyzer, dataloader, dev)"
        if base.count(inner) != 1:
            raise RuntimeError("quantization boundary changed")
        base = base.replace(inner, '''    if getattr(args, "_analysis_hook", None) is not None:
        raise RuntimeError("cannot replace an existing analysis hook")
    args._analysis_hook = _residency_boundary_hook
    try:
        result = original_fwrd(args, analyzer, dataloader, dev)
    finally:
        args._analysis_hook = None''')
    entry.write_text(base)
    compile(base, str(entry), "exec")
    source = ROOT / "outputs/memory_mechanism_followup_20261010/hessian-cpu-layer0-manifests"
    for side in ("control", "candidate"):
        for stage in ("stage0", "stage2"):
            data = read(source / f"control-{stage}.json")
            cmd = data["command"]
            cmd[cmd.index("baseline-profile") - 1] = str(entry)
            set_option(cmd, "--quant_stop_layer", "0" if prefix else "27")
            if not prefix:
                set_option(cmd, "--static_saved_tensor_offload_mib", "0")
                cmd[:] = [value for value in cmd if value != "--static_saved_tensor_offload_deduplicate"]
                if side == "candidate" and stage == "stage0":
                    cmd.append("--research_stage0_checkpoint")
                if side == "candidate" and stage == "stage2":
                    set_option(cmd, "--eval_ce_chunk_tokens", "256")
            if side == "candidate" and stage == "stage2":
                cmd += ["--stage2_hessian_cpu_offload",
                        "--research_stage2_quantizer_metadata_cpu" if method == "metadata"
                        else "--research_stage2_rehome_completed_layer"]
            data["scope_note"] = ("Fresh layer0 original-control composition diagnostic" if prefix
                                  else "Fresh complete Qwen3-1.7B 2x512, microbatch1, all28 quantized; repeated system composition, not paper/4-card scale")
            (folder / f"{side}-{stage}.json").write_text(json.dumps(data, indent=2))


def main():
    global OUT, MAIN_TAG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--main-tag", default=MAIN_TAG)
    parser.add_argument("--wait-for", type=Path, default=ROOT / "outputs/input-capacity-restart-20261010-1408/status.json")
    parser.add_argument("--main-manifests", type=Path, default=ROOT / "outputs/cpu-input-checkpoint-followup-20261010/qwen06b-checkpoint-ce-full-2x1024-threepairs-20261010-manifests")
    parser.add_argument("--metadata-tag-suffix", default="-restart1408")
    parser.add_argument("--tag-suffix", default="")
    parser.add_argument("--skip-hessian-repair", action="store_true",
                        help="Continue independent checkpoint/CE and speed work without repeating a completed failed Hessian composition")
    options = parser.parse_args()
    OUT, MAIN_TAG = options.output_dir.resolve(), options.main_tag
    OUT.mkdir(exist_ok=False)
    jobs = []
    proc = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)

    def execute(label, cmd):
        nonlocal proc
        (OUT / "active.json").write_text(json.dumps({"label": label, "command": cmd}, indent=2))
        print(f"[post-restart] starting {label}", flush=True)
        with (OUT / f"{label}.log").open("x") as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                proc.wait()
            except BaseException:
                stop_process_tree(proc.pid)
                proc.wait()
                raise
        jobs.append({"label": label, "returncode": proc.returncode, "finished": time.strftime("%F %T")})
        (OUT / "jobs.json").write_text(json.dumps(jobs, indent=2))
        print(f"[post-restart] {jobs[-1]}", flush=True)
        return proc.returncode == 0

    def campaign(tag, folder, pairs=1):
        required = (27 if pairs == 3 and "qwen17b" in tag else 9) * 1024**3
        if shutil.disk_usage(ROOT).free < required:
            raise RuntimeError(f"insufficient disk for fresh campaign: need {required/1024**3:.0f} GiB")
        cmd = [sys.executable, "-u", str(ROOT / "tools/run_full_pipeline_memory_campaign.py"),
               "--tag", tag, "--pairs", str(pairs), "--gpu-indices", "0", "--timeout-seconds", "7200"]
        for side in ("control", "candidate"):
            for stage in ("stage0", "stage2"):
                cmd += [f"--{side}-{stage}-manifest", str(folder / f"{side}-{stage}.json")]
        return execute(tag, cmd)

    def release_completed_disk_copies():
        # Run only between GPU jobs. Preserve independently collected statistics.
        # Reuse the established *completed* byte-identical model as a read-only
        # hardlink anchor, never as an input cache for a newly collected run.
        win_python = "/mnt/c/Users/65377/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
        anchors = {"Qwen3-0.6B": ROOT / "outputs/full_pipeline_profile_qwen06b-cpu-input-init-layer0-2x1024-20261010-r1-control",
                   "Qwen3-1.7B": ROOT / "outputs/full_pipeline_profile_qwen17b-cpu-input-init-layer0-2x512-20261010-r1-control"}
        prefixes = ("full_pipeline_profile_" + MAIN_TAG,
                    "full_pipeline_profile_qwen17b-quantizer-metadata-cpu-layer0-2x512-20261010",
                    "full_pipeline_profile_qwen17b-hessian-metadata-cpu-original-control-layer0-20261010",
                    "full_pipeline_profile_qwen17b-hessian-layer-residency-",
                    "full_pipeline_profile_qwen17b-checkpoint-hessian-",
                    "full_pipeline_profile_qwen17b-checkpoint-ce-matched-",
                    "full_pipeline_profile_qwen06b-checkpoint-ce-triton-")
        for directory in sorted((ROOT / "outputs").iterdir()):
            if not directory.name.startswith(prefixes) or not (directory / "status.json").exists():
                continue
            state = read(directory / "status.json")
            if not state.get("valid"):
                continue
            config = read(directory / "manifest.json")
            cmd = config["stage2_command"]
            model = Path(cmd[cmd.index("--model") + 1]).name
            if model not in anchors:
                continue
            result = OUT / f"disk-{directory.name}.json"
            if result.exists():
                continue
            def windows(path):
                value = str(path)
                if not value.startswith("/mnt/d/"):
                    raise RuntimeError("housekeeping path is outside the task's D drive")
                return "D:/" + value[len("/mnt/d/"):]
            command = [win_python, windows(ROOT / "tools/deduplicate_completed_prepared_checkpoint.py"),
                       "--control-run", windows(anchors[model]), "--candidate-run", windows(directory),
                       "--output", windows(result), "--apply"]
            subprocess.run(command, cwd=ROOT, check=True, timeout=180)

    try:
        waited = time.monotonic()
        dependency = options.wait_for
        print("[post-restart] waiting for earlier serial controllers; no CUDA work", flush=True)
        while not dependency.exists():
            if time.monotonic() - waited > 28800:
                raise RuntimeError("prior queue did not finish within eight hours")
            time.sleep(5)
        # A completed capacity failure is not a quality/memory pass. It merely
        # means its workers have exited; its own status remains independently reported.
        release_completed_disk_copies()
        method = None
        meta_tag = "qwen17b-hessian-metadata-cpu-original-control-layer0-20261010" + options.metadata_tag_suffix
        if options.skip_hessian_repair:
            print("[post-restart] skipping completed Hessian-repair experiments; preserving their negative audits", flush=True)
        elif accepted(meta_tag):
            method = "metadata"
        else:
            tag = "qwen17b-hessian-layer-residency-independent-layer0-20261010" + options.tag_suffix
            folder = ROOT / "outputs/layer-residency-compaction-prefix-manifests-20261010"
            if campaign(tag, folder):
                release_completed_disk_copies()
                original = OUT / "layer-residency-original-control-manifests"
                prepare_composition(original, "compact", prefix=True)
                tag = "qwen17b-hessian-layer-residency-original-control-layer0-20261010" + options.tag_suffix
                if campaign(tag, original):
                    method = "compact"
        release_completed_disk_copies()
        if method:
            folder = OUT / "qwen17b-full-composition-manifests"
            prepare_composition(folder, method, prefix=False)
            campaign(f"qwen17b-checkpoint-hessian-{method}-ce-full-threepairs-20261010" + options.tag_suffix, folder, pairs=3)
            release_completed_disk_copies()
        # The Hessian repair is an optional larger-model component. Its failure
        # must not erase the independent portable checkpoint/CE hypothesis.
        # Use the SAME sample count, sequence length and refresh microbatch as
        # the accepted 0.6B main protocol; change only the model in both arms.
        main_status = ROOT / "outputs" / f"memory_campaign_{MAIN_TAG}/status.json"
        if main_status.exists() and read(main_status).get("valid"):
            matched = OUT / "qwen17b-matched-checkpoint-ce-manifests"
            matched.mkdir()
            for side in ("control", "candidate"):
                for stage in ("stage0", "stage2"):
                    data = read(options.main_manifests / f"{side}-{stage}.json")
                    set_option(data["command"], "--model", "/mnt/d/llamaModels/Qwen3-1.7B")
                    data["scope_note"] = "Both models share 2x1024/microbatch2/eval2x1024/full28; candidate only checkpoint+CE. Independent from Hessian repair; not paper/4-card scale."
                    (matched / f"{side}-{stage}.json").write_text(json.dumps(data, indent=2))
            campaign("qwen17b-checkpoint-ce-matched-full-threepairs-20261010" + options.tag_suffix, matched, pairs=3)
            release_completed_disk_copies()
        # An independent speed component is expanded only if the earlier
        # complete memory composition passed all requested repetitions.
        main_status = ROOT / "outputs" / f"memory_campaign_{MAIN_TAG}/status.json"
        if main_status.exists() and read(main_status).get("valid"):
            source = options.main_manifests
            folder = OUT / "qwen06b-triton-composition-manifests"
            folder.mkdir()
            for side in ("control", "candidate"):
                for stage in ("stage0", "stage2"):
                    data = read(source / f"{side}-{stage}.json")
                    if side == "candidate" and stage == "stage2":
                        set_option(data["command"], "--gptq_inner_kernel", "triton_fused")
                    (folder / f"{side}-{stage}.json").write_text(json.dumps(data, indent=2))
            triton_tag = "qwen06b-checkpoint-ce-triton-full-fresh-20261010" + options.tag_suffix
            triton_ok = campaign(triton_tag, folder)
            release_completed_disk_copies()
            reference_tag = triton_tag if triton_ok else MAIN_TAG
            command = [sys.executable, "-u", str(ROOT / "tools/run_low_overhead_speed_campaign.py"),
                       "--tag", "qwen06b-accepted-system-threepairs-20261010" + options.tag_suffix, "--pairs", "3", "--gpu-indices", "0",
                       "--control-reference", str(ROOT / "outputs" / f"full_pipeline_profile_{reference_tag}-r1-control"),
                       "--candidate-reference", str(ROOT / "outputs" / f"full_pipeline_profile_{reference_tag}-r1-candidate"),
                       "--pair-audit", str(ROOT / "outputs" / f"memory_campaign_{reference_tag}/pair1_audit/comparison.json")]
            if shutil.disk_usage(ROOT).free < 11 * 1024**3:
                raise RuntimeError("requires eleven GiB for six fresh 0.6B speed arms")
            execute("qwen06b-accepted-system-threepairs-speed", command)
        result = {"finished": time.strftime("%F %T"), "jobs": jobs, "accepted_hessian_repair": method,
                  "scope": "local workload pilots; each campaign has independent validity, no formal/4-card assertion"}
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            stop_process_tree(proc.pid)
            proc.wait()
        (OUT / "status.json").write_text(json.dumps({"error": repr(exc), "jobs": jobs, "finished": time.strftime("%F %T")}, indent=2))
        raise
    (OUT / "status.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
