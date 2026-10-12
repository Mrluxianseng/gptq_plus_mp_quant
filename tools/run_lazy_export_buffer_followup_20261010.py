"""Serial API/prefix/full checks of non-consumed integer export copies.

Waits for preceding GPU work, keeps the production implementation untouched,
and requires fresh numerical and measured-memory audits for each expansion.
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
from tools.run_full_pipeline_memory_campaign import stop_process_tree
from tools.run_full_pipeline_memory_profile import set_option


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wait-for", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--main-tag", required=True)
    p.add_argument("--main-manifests", type=Path, required=True)
    args = p.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(exist_ok=False)
    jobs = []
    proc = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    def execute(label, command):
        nonlocal proc
        (out / "active.json").write_text(json.dumps(dict(label=label, command=command), indent=2))
        print(f"[lazy-export] starting {label}", flush=True)
        with (out/f"{label}.log").open("x") as log:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                proc.wait(timeout=8*3600)
            except BaseException:
                stop_process_tree(proc.pid)
                proc.wait()
                raise
        jobs.append(dict(label=label, returncode=proc.returncode, finished=time.strftime("%F %T")))
        (out / "jobs.json").write_text(json.dumps(jobs, indent=2))
        return proc.returncode == 0
    def make_entry(folder, checkpoint):
        folder.mkdir()
        source = ROOT / "tools" / ("realq_checkpoint_research_entry.py" if checkpoint else "realq_memory_profile_entry.py")
        base = source.read_text()
        helper = ROOT / "tools/export_buffer_lifetime_prototype.py"
        helper_sha = hashlib.sha256(helper.read_bytes()).hexdigest()
        injection = f'''
_lazy_export = "--research_stage2_lazy_export_buffers" in sys.argv
if _lazy_export:
    sys.argv.remove("--research_stage2_lazy_export_buffers")
_export_helper = Path.cwd() / "tools/export_buffer_lifetime_prototype.py"
_export_helper_sha = "{helper_sha}"
if hashlib.sha256(_export_helper.read_bytes()).hexdigest() != _export_helper_sha:
    raise RuntimeError("export helper changed since entry snapshot")
if _lazy_export:
    from tools.export_buffer_lifetime_prototype import build_method
    _patched_method, _patched_source = build_method(impl)
    (root / "lazy_export_fasterquant.py").write_text(_patched_source)
    impl.GPTQPlus.fasterquant = _patched_method
'''
        needle = "original_fwrd = impl.gptq_fwrd"
        inner = "    result = original_fwrd(args, analyzer, dataloader, dev)"
        if base.count(needle) != 1 or base.count(inner) != 1:
            raise RuntimeError("entry source boundary changed")
        base = base.replace(needle, injection+"\n"+needle).replace(inner, inner+'''
    if hashlib.sha256(_export_helper.read_bytes()).hexdigest() != _export_helper_sha:
        raise RuntimeError("export helper changed during quantization")''')
        entry = folder / "entry.py"
        entry.write_text(base)
        compile(base, str(entry), "exec")
        return entry
    def campaign(tag, folder, pairs=1):
        required = (27 if pairs == 3 and "qwen17b" in tag else 9)*1024**3
        if shutil.disk_usage(ROOT).free < required:
            raise RuntimeError(f"need {required/1024**3:.0f} GiB before fresh campaign")
        command = [sys.executable, "-u", str(ROOT / "tools/run_full_pipeline_memory_campaign.py"),
                   "--tag", tag, "--pairs", str(pairs), "--gpu-indices", "0", "--timeout-seconds", "7200"]
        for side in ("control", "candidate"):
            for stage in ("stage0", "stage2"):
                command += [f"--{side}-{stage}-manifest", str(folder/f"{side}-{stage}.json")]
        return execute(tag, command)
    def release_completed_model_copies():
        # Keep every path, model byte and independently collected statistic.
        # Only verified completed prepared FP checkpoint copies are linked.
        anchors = {
            "Qwen3-0.6B": ROOT / "outputs/full_pipeline_profile_qwen06b-cpu-input-init-layer0-2x1024-20261010-r1-control",
            "Qwen3-1.7B": ROOT / "outputs/full_pipeline_profile_qwen17b-cpu-input-init-layer0-2x512-20261010-r1-control",
        }
        prefixes = ("full_pipeline_profile_qwen06b-cpu-saved-kl-logits-", "full_pipeline_profile_qwen06b-lazy-export-",
                    "full_pipeline_profile_qwen17b-lazy-export-", "full_pipeline_profile_qwen17b-hessian-lazy-export-")
        def windows(path):
            value = str(path)
            if not value.startswith("/mnt/d/"):
                raise RuntimeError("housekeeping target outside intended D workspace")
            return "D:/"+value[len("/mnt/d/"):]
        for directory in sorted((ROOT / "outputs").iterdir()):
            if not directory.name.startswith(prefixes) or not (directory / "status.json").exists():
                continue
            state = json.loads((directory / "status.json").read_text())
            if not state.get("valid"):
                continue
            cmd = json.loads((directory / "manifest.json").read_text())["stage2_command"]
            model = Path(cmd[cmd.index("--model")+1]).name
            if model not in anchors:
                continue
            record = out / f"disk-{directory.name}.json"
            if record.exists():
                continue
            subprocess.run(["/mnt/c/Users/65377/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe",
                            windows(ROOT / "tools/deduplicate_completed_prepared_checkpoint.py"),
                            "--control-run", windows(anchors[model]), "--candidate-run", windows(directory),
                            "--output", windows(record), "--apply"], cwd=ROOT, check=True, timeout=180)
    try:
        print("[lazy-export] waiting for prior complete GPU queue; no CUDA work", flush=True)
        started = time.monotonic()
        while not args.wait_for.exists():
            if time.monotonic()-started > 24*3600:
                raise RuntimeError("prior queue did not exit within one day")
            time.sleep(5)
        release_completed_model_copies()
        if not execute("tiny-cuda-api-export-consumer-check", [sys.executable, "-u", str(ROOT/"tools/check_lazy_export_buffers_cuda.py"),
                                                             "--output-dir", str(out/"api-preflight")]):
            raise RuntimeError("actual CUDA export consumer preflight failed")
        template = ROOT / "outputs/memory_mechanism_followup_20261010/hessian-cpu-layer0-manifests"
        independent = out / "qwen17b-independent-prefix-manifests"
        entry = make_entry(independent, False)
        for side in ("control", "candidate"):
            for stage in ("stage0", "stage2"):
                data = json.loads((template/f"candidate-{stage}.json").read_text())
                command = data["command"]
                command[command.index("baseline-profile")-1] = str(entry)
                if side == "candidate" and stage == "stage2":
                    command.append("--research_stage2_lazy_export_buffers")
                data["scope_note"] = "Both Hessian CPU; candidate only avoids optional full/subgroup code/scale copies. Column/128-block arithmetic and buffers unchanged. Layer0 prefix, fresh stats."
                (independent/f"{side}-{stage}.json").write_text(json.dumps(data, indent=2))
        prefix_tag = "qwen17b-lazy-export-copies-independent-layer0-20261010-1625"
        if campaign(prefix_tag, independent):
            release_completed_model_copies()
            repaired = out / "qwen17b-original-control-prefix-manifests"
            entry = make_entry(repaired, False)
            for side in ("control", "candidate"):
                for stage in ("stage0", "stage2"):
                    data = json.loads((template/f"{side}-{stage}.json").read_text())
                    command = data["command"]
                    command[command.index("baseline-profile")-1] = str(entry)
                    if side == "candidate" and stage == "stage2":
                        command.append("--research_stage2_lazy_export_buffers")
                    data["scope_note"] = "Original-control vs Hessian CPU+unused export-copy elision; prefix only. No arithmetic/GEMM changes."
                    (repaired/f"{side}-{stage}.json").write_text(json.dumps(data, indent=2))
            campaign("qwen17b-hessian-lazy-export-original-control-layer0-20261010-1625", repaired)
        release_completed_model_copies()
        accepted = ROOT / "outputs" / f"memory_campaign_{args.main_tag}/status.json"
        if accepted.exists() and json.loads(accepted.read_text()).get("valid"):
            full = out / "qwen06b-full-independent-manifests"
            entry = make_entry(full, True)
            for side in ("control", "candidate"):
                for stage in ("stage0", "stage2"):
                    data = json.loads((args.main_manifests/f"candidate-{stage}.json").read_text())
                    command = data["command"]
                    command[command.index("baseline-profile")-1] = str(entry)
                    if side == "candidate" and stage == "stage2":
                        command.append("--research_stage2_lazy_export_buffers")
                    data["scope_note"] = "Shared accepted checkpoint/CE, all28 0.6B, 2x1024/microbatch2. Candidate only unused export-copy elision."
                    (full/f"{side}-{stage}.json").write_text(json.dumps(data, indent=2))
            campaign("qwen06b-lazy-export-copies-checkpoint-ce-full-20261010-1625", full)
        release_completed_model_copies()
        # A no-op or failing component is not expanded into three copies.
        (out/"gpu_work_finished.json").write_text(json.dumps(dict(jobs=jobs)))
        tags = [row["label"] for row in jobs if row["label"].startswith("qwen")]
        if tags:
            execute("record-completed-pairs", [sys.executable, str(ROOT/"tools/record_completed_memory_followup_20261010.py"),
                    "--wait-for", str(out/"gpu_work_finished.json"), "--output-dir", str(out/"completed-records"), "--campaign-tags", *tags])
        result = dict(completed=True, jobs=jobs, scope="Independent local component pilots, not formal/multi-card or repeat proof")
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            stop_process_tree(proc.pid)
            proc.wait()
        (out/"status.json").write_text(json.dumps(dict(error=repr(exc),jobs=jobs,finished=time.strftime("%F %T")),indent=2))
        raise
    result["finished"] = time.strftime("%F %T")
    (out/"status.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
