"""After the active serial queue, verify a new logits-residency-only policy.

Uses exact real-head shape preflights before a fresh full REAL-Q pair; stops
on fidelity or measured-memory failure. No timing/algorithm innovation claim.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for", type=Path, required=True)
    parser.add_argument("--main-campaign-tag", required=True)
    parser.add_argument("--main-manifests", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(exist_ok=False)
    proc = None
    jobs = []
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    def execute(label, command):
        nonlocal proc
        (out / "active.json").write_text(json.dumps(dict(label=label, command=command), indent=2))
        print(f"[saved-logits] starting {label}", flush=True)
        with (out / f"{label}.log").open("x") as log:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                proc.wait(timeout=7200)
            except BaseException:
                stop_process_tree(proc.pid)
                proc.wait()
                raise
        jobs.append(dict(label=label, returncode=proc.returncode, finished=time.strftime("%F %T")))
        (out / "jobs.json").write_text(json.dumps(jobs, indent=2))
        return proc.returncode == 0
    try:
        print("[saved-logits] waiting for earlier complete GPU/timing queue; no CUDA work", flush=True)
        started = time.monotonic()
        while not args.wait_for.exists():
            if time.monotonic()-started > 24*3600:
                raise RuntimeError("earlier queue not completed within one day")
            time.sleep(5)
        accepted = ROOT / "outputs" / f"memory_campaign_{args.main_campaign_tag}"
        if not json.loads((accepted / "status.json").read_text()).get("valid"):
            raise RuntimeError("source checkpoint/CE full repetitions did not pass; do not expand")
        # No silently changed core is allowed between the reference and this
        # addition. This check has no CUDA/tensor initialization.
        from tools.run_low_overhead_speed_campaign import core_sources, check_sources
        reference = ROOT / "outputs" / f"full_pipeline_profile_{args.main_campaign_tag}-r1-candidate"
        check_sources(core_sources(json.loads((reference / "manifest.json").read_text())))
        for model in ("Qwen3-0.6B", "Qwen3-1.7B"):
            result = out / f"{model}-cuda-fidelity.json"
            if not execute(f"{model}-real-head-fidelity", [sys.executable, "-u",
                           str(ROOT / "tools/check_cpu_saved_logits_kl.py"), "--device", "cuda",
                           "--model-path", f"/mnt/d/llamaModels/{model}", "--rows", "2048", "--output", str(result)]):
                raise RuntimeError(f"{model}: CUDA preflight worker failed; inspect {out / (model + '-real-head-fidelity.log')}")
            if not json.loads(result.read_text()).get("valid"):
                raise RuntimeError("fidelity worker did not establish exactness")
        if shutil.disk_usage(ROOT).free < 9*1024**3:
            raise RuntimeError("need nine GiB for a fresh pair; preserve previous artifacts")
        folder = out / "manifests"
        folder.mkdir()
        base = (ROOT / "tools/realq_checkpoint_research_entry.py").read_text()
        helper = ROOT / "tools/cpu_saved_logits_kl_prototype.py"
        helper_sha = hashlib.sha256(helper.read_bytes()).hexdigest()
        injection = f'''
_cpu_saved_logits = "--research_stage2_cpu_saved_logits" in sys.argv
if _cpu_saved_logits:
    sys.argv.remove("--research_stage2_cpu_saved_logits")
_logit_helper = Path.cwd() / "tools/cpu_saved_logits_kl_prototype.py"
_logit_helper_sha = "{helper_sha}"
if hashlib.sha256(_logit_helper.read_bytes()).hexdigest() != _logit_helper_sha:
    raise RuntimeError("logits helper changed since entry snapshot")
if _cpu_saved_logits:
    from tools.cpu_saved_logits_kl_prototype import cpu_saved_logits_kl
    impl.chunked_tokenwise_kl_from_logits = cpu_saved_logits_kl
'''
        needle = "original_fwrd = impl.gptq_fwrd"
        boundary = "    result = original_fwrd(args, analyzer, dataloader, dev)"
        if base.count(needle) != 1 or base.count(boundary) != 1:
            raise RuntimeError("entry injection boundaries changed")
        base = base.replace(needle, injection + "\n" + needle).replace(boundary, boundary + '''
    if hashlib.sha256(_logit_helper.read_bytes()).hexdigest() != _logit_helper_sha:
        raise RuntimeError("logits helper changed during quantization")''')
        entry = folder / "entry.py"
        entry.write_text(base)
        compile(base, str(entry), "exec")
        for side in ("control", "candidate"):
            for stage in ("stage0", "stage2"):
                # Independent additional-component ablation: both arms have
                # the now-accepted checkpoint/CE system, only N changes storage.
                data = json.loads((args.main_manifests / f"candidate-{stage}.json").read_text())
                data["command"][data["command"].index("baseline-profile")-1] = str(entry)
                if side == "candidate" and stage == "stage2":
                    data["command"].append("--research_stage2_cpu_saved_logits")
                data["scope_note"] = "Fresh all28 0.6B 2x1024/microbatch2; common checkpoint+CE. Candidate only CPU-saved KL logits/token-slice backward. Original head/gradient GEMM shapes unchanged."
                (folder / f"{side}-{stage}.json").write_text(json.dumps(data, indent=2))
        tag = "qwen06b-cpu-saved-kl-logits-checkpoint-ce-full-20261010-1545"
        command = [sys.executable, "-u", str(ROOT/"tools/run_full_pipeline_memory_campaign.py"),
                   "--tag", tag, "--pairs", "1", "--gpu-indices", "0", "--timeout-seconds", "7200"]
        for side in ("control", "candidate"):
            for stage in ("stage0", "stage2"):
                command += [f"--{side}-{stage}-manifest", str(folder/f"{side}-{stage}.json")]
        passed = execute(tag, command)
        # Record after this GPU pair has exited. Preserve failures rather than
        # running three copies of an unchanged failed implementation.
        (out / "gpu_work_finished.json").write_text(json.dumps(dict(pair_passed=passed)))
        execute("record-completed-pair", [sys.executable, str(ROOT/"tools/record_completed_memory_followup_20261010.py"),
                "--wait-for", str(out/"gpu_work_finished.json"), "--output-dir", str(out/"completed-records"),
                "--campaign-tags", tag])
        result = dict(valid=passed, jobs=jobs, scope="New additional-component single pair; not repeated/four-card/formal-scale proof")
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            stop_process_tree(proc.pid)
            proc.wait()
        (out/"status.json").write_text(json.dumps(dict(valid=False,error=repr(exc),jobs=jobs,finished=time.strftime("%F %T")),indent=2))
        raise
    result["finished"] = time.strftime("%F %T")
    (out/"status.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
