"""Conditional next candidate: cached real prefix0..10 after resource failure.

Keeps allocator history from layer0, catches layers3/10 replay regressions,
but is not arbitrary checkpoint resume or a full-process memory proof.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_full_pipeline_profiles import cache_hashes, digest, phase_peaks
from tools.run_full_pipeline_memory_profile import set_option
from tools.run_full_pipeline_memory_campaign import stop_process_tree


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wait-for", type=Path, required=True)
    p.add_argument("--workspace-queue", type=Path, required=True)
    p.add_argument("--reference-run", type=Path, required=True)
    p.add_argument("--workspace-prefix-run", type=Path, required=True)
    p.add_argument("--output-directory", type=Path, required=True)
    args = p.parse_args()
    out = args.output_directory.resolve()
    out.mkdir(exist_ok=False)
    proc = reader = None
    report = dict(valid=False, scope="Cached Stage-2 prefix0..10; no complete-process or multi-GPU claim")
    def progress(text):
        print("[all-weight-screen] " + text, flush=True)
    def interrupt(sig, frame):
        raise KeyboardInterrupt(f"signal {sig}")
    signal.signal(signal.SIGTERM, interrupt)
    try:
        progress("waiting for workspace/capture/replay completion; no GPU process")
        while not args.wait_for.exists():
            time.sleep(5)
        if not read(args.wait_for).get("valid"):
            raise RuntimeError("actual fixture capture/replay prerequisite failed; inspect first")
        preceding = read(args.workspace_queue / "status.json")
        if preceding.get("valid"):
            report.update(valid=True, skipped=True, reason="Workspace full queue passed; no blind additional candidate")
            return
        failed = []
        for tag in preceding.get("fallback_status", {}).get("campaigns", []):
            for filename in (ROOT / "outputs" / ("memory_campaign_" + tag)).glob("pair*_audit/comparison.json"):
                data = read(filename)
                if not data.get("strict_measured_memory_gate_passed"):
                    failed.append(dict(path=str(filename), exact=data.get("exactness_passed"),
                                       phases=data.get("phase_regressions"), boundaries=data.get("boundary_regressions")))
        if not failed or not all(row["exact"] for row in failed):
            raise RuntimeError("No complete exact resource failure; do not hide numerical/infrastructure errors")
        report["trigger_audits"] = failed
        time.sleep(5)
        gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                             text=True, capture_output=True, check=True)
        if any(x.strip().isdigit() for x in gpu.stdout.splitlines()):
            raise RuntimeError("another compute process is active; no concurrent CUDA screen")
        reader = subprocess.Popen([sys.executable, "-u", str(ROOT / "tools/stream_memory_research_progress.py"),
            "--queues", str(out), "--output", str(ROOT / "logs/realq-system-restart-20261010-1408.log"), "--append"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        reference = args.reference_run.resolve()
        ref_status, ref_manifest = read(reference / "status.json"), read(reference / "manifest.json")
        source_run = args.workspace_prefix_run.resolve()
        source_status, source_manifest = read(source_run / "status.json"), read(source_run / "manifest.json")
        if not ref_status.get("valid") or not source_status.get("valid") or source_manifest["expected_ranks"]["stage2"] != 1:
            raise RuntimeError("requires passed single-rank references")
        bound = source_manifest["source_sha256_at_launch"]
        helper = ROOT / "tools/hessian_all_weight_storage_prototype.py"
        bound = dict(bound, **{str(helper): digest(helper), str(Path(__file__).resolve()): digest(Path(__file__).resolve())})
        for file, expected in bound.items():
            if digest(file) != expected:
                raise RuntimeError(f"bound source changed: {file}")
        base = list(source_manifest["stage2_command"])
        ref_cmd = ref_manifest["stage2_command"]
        for flag in ("--model", "--dataset", "--nsamples", "--seq_len", "--seed", "--num_groups",
                     "--static_fisher_label_chunk_tokens"):
            if base[base.index(flag)+1] != ref_cmd[ref_cmd.index(flag)+1]:
                raise RuntimeError("cache-generating conditions mismatch: " + flag)
        cache = cache_hashes(reference, progress)
        entry_index = base.index("baseline-profile") - 1
        original_entry = Path(base[entry_index]).read_text()
        anchor = "try:\n    runpy.run_path('ptq.py', run_name='__main__')"
        if original_entry.count(anchor) != 1 or "_hinv_reuse_source=" not in original_entry:
            raise RuntimeError("bound workspace entry boundary changed")
        hook = '''from tools.hessian_all_weight_storage_prototype import install as _all_install
_all_enabled="--research_all_weight_stable" in sys.argv
if _all_enabled:
    sys.argv.remove("--research_all_weight_stable")
_all_records=[]
_all_source_sha=_all_install(impl, _hinv_reuse_source, enabled=_all_enabled, records=_all_records)
def _all_save():
    (Path(os.environ["RESEARCH_RUN_DIR"])/"all_weight_writes.json").write_text(json.dumps(dict(enabled=_all_enabled, source_sha256=_all_source_sha, records=_all_records),indent=2))
_hinv_atexit.register(_all_save)
def _forbid_precompute(*args, **kwargs):
    raise RuntimeError("cached screen must not silently rerun precompute")
impl.collect_static_end_to_end_saliency_and_fisher=_forbid_precompute

'''
        entry = out / "entry.py"
        entry.write_text(original_entry.replace(anchor, hook + anchor))
        compile(entry.read_text(), str(entry), "exec")
        base[entry_index] = str(entry)
        set_option(base, "--static_cache_path", ref_manifest["fresh_static_cache"])
        set_option(base, "--quant_stop_layer", "10")
        set_option(base, "--quant_profile_target_layers", ",".join(map(str, range(11))))
        commands = {}
        for side in ("control", "candidate"):
            cmd = list(base)
            set_option(cmd, "--exp", out.name + "-" + side)
            if side == "candidate":
                cmd.extend(["--research_hinv_solver_workspace", "--research_hinv_reuse_consumed_h",
                            "--research_hinv_stable_weight", "--research_all_weight_stable"])
            commands[side] = cmd
        bound[str(entry)] = digest(entry)
        (out / "manifest.json").write_text(json.dumps(dict(commands=commands, source_sha256=bound,
            reference_manifest_sha256=digest(reference / "manifest.json"), static_cache_sha256=cache,
            stage0_run=False, quantized_through_layer=10, hypotheses="All linear weight writes keep original storage; other weights may pin transient segments"), indent=2))
        statuses = {}
        for side, command in commands.items():
            run = out / side
            run.mkdir()
            env = os.environ.copy()
            env.update(RESEARCH_RUN_DIR=str(run), REALQ_PHASE_PEAK_MODE="exact", OMP_NUM_THREADS="1",
                       HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            (out / "active.json").write_text(json.dumps(dict(label=side, command=command)))
            with (out / (side + ".log")).open("x") as log, (run / "nvml.csv").open("x") as gpu_log:
                monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--id=0",
                                            "--format=csv,noheader,nounits", "--loop-ms=100"], stdout=gpu_log, stderr=subprocess.STDOUT)
                try:
                    proc = subprocess.Popen(["timeout", "--signal=TERM", "--kill-after=30s", "7200", *command],
                                            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    for line in proc.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    code = proc.wait()
                finally:
                    monitor.terminate()
                    monitor.wait(timeout=5)
            if code:
                raise RuntimeError(side + " trajectory failed; no further candidate arm")
            stats = read(run / "baseline-profile_metrics.json")
            quality = re.search(r"Exact KL&PPL on wikitext2: ([0-9.eE+-]+), ([0-9.eE+-]+)", (out / (side + ".log")).read_text(errors="replace"))
            if not quality:
                raise RuntimeError("missing held-out KL/PPL")
            peak = read(run / "process_allocator_peak_stage2_rank0.json")
            memory_files = sorted({str(path) for path in (ROOT / "outputs").glob("*/" + out.name + "-" + side + "/memory_profile/*.jsonl")})
            values = [int(line.strip()) for line in (run / "nvml.csv").read_text().splitlines() if line.strip().isdigit()]
            if not values:
                raise RuntimeError("NVML telemetry missing")
            statuses[side] = dict(metrics=stats, kl=float(quality.group(1)), ppl=float(quality.group(2)),
                layers=read(run / "layer_output_fingerprints.json"), memory_profile_files=memory_files,
                process_peak=peak, nvml_mib=max(values), writes=read(run / "all_weight_writes.json"))
            (run / "status.json").write_text(json.dumps(statuses[side], indent=2))
            if not all(digest(file) == expected for file, expected in bound.items()):
                raise RuntimeError("source changed during cached prefix")
        c, n = statuses["control"], statuses["candidate"]
        exact = all(c["metrics"][key] == n["metrics"][key] for key in ("state_sha256", "state_tensors")) and all(c[key] == n[key] for key in ("kl", "ppl", "layers"))
        phases = [phase_peaks(s, ROOT) for s in (c, n)]
        missing = sorted(set(phases[0]) ^ set(phases[1]))
        regressions = []
        for key in sorted(set(phases[0]) & set(phases[1])):
            delta = {metric: phases[1][key][metric] - phases[0][key][metric]
                     for metric in ("allocated", "reserved", "boundary_allocated", "boundary_reserved")}
            if any(v > 0 for v in delta.values()):
                regressions.append(dict(key=key, delta=delta))
        exact_peaks = bool(phases[0]) and all(p["measurement_mode"] == "exact" for arm in phases for p in arm.values())
        global_ok = n["nvml_mib"] <= c["nvml_mib"] and all(n["process_peak"][k] <= c["process_peak"][k]
                           for k in ("peak_allocated_bytes", "peak_reserved_bytes"))
        writes = n["writes"]["records"]
        write_coverage = len(writes) == 77 and all(row["pointer_preserved"] and row["enabled"] for row in writes)
        report.update(exactness_passed=exact, cache_unchanged=cache == cache_hashes(reference, progress),
            exact_interval_peaks=exact_peaks, missing_phases=missing, regressions=regressions,
            global_memory_gate=global_ok, all_weight_write_coverage=write_coverage,
            control_nvml_mib=c["nvml_mib"], candidate_nvml_mib=n["nvml_mib"])
        report["valid"] = exact and report["cache_unchanged"] and exact_peaks and not missing and not regressions and global_ok and write_coverage
        if not report["valid"]:
            raise RuntimeError("cached prefix numerical or memory gate failed; recorded diagnosis requires review")
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            stop_process_tree(proc.pid)
            proc.wait()
        report["error"] = repr(exc)
        raise
    finally:
        report["finished"] = time.strftime("%F %T")
        (out / "status.json").write_text(json.dumps(report, indent=2))
        with (ROOT / "docs/memory_speed_optimization_journal.md").open("a", encoding="utf-8") as journal:
            journal.write("\n\n### 条件续接：所有线性模块权重存储位置\n\n"
                "**理论/参考。** OLLA最后消费者与位置（https://arxiv.org/html/2210.12924v2），GMLake碎片化分析"
                "（https://arxiv.org/html/2401.08156v1）。\n\n"
                "**具体方法。** 仅在两缓冲区全流程出现数值正确的资源失败、真实捕获/重放通过后，"
                "将稳定最终权重存储从down_proj推广到所有顺序量化线性模块，保持转换/补偿/Adam数学不变。"
                "两臂复用同一核验缓存，跑真实0–10层前缀保留Stage-2分配历史，覆盖先前层3/10问题。\n\n"
                f"**实际效果。** `{out / 'status.json'}`；valid={report['valid']}，skipped={report.get('skipped',False)}。"
                "层27及Stage-0未验证，不能写成新全程收益。\n\n"
                "**评价与反思。** 其他权重钉住大段是待验证假设；若失败先审查逐区间差值，"
                "不用校准/学习率调整掩盖资源失败，重复完整实验留给稳定候选。\n")
        if reader:
            reader.wait(timeout=15)
        progress(json.dumps(report))


if __name__ == "__main__":
    main()
