"""One cached Stage-2 capture, then fast solver replays, after an owned queue.

Does not run Stage-0. Requires a completed, source-bound full reference and
verifies cache contents before/after. The capture's own memory/time are not
optimization evidence. Only single-rank diagnostic operation is implemented.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_full_pipeline_profiles import cache_hashes, digest
from tools.run_full_pipeline_memory_profile import set_option
from tools.run_full_pipeline_memory_campaign import stop_process_tree


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-run", type=Path, required=True)
    p.add_argument("--output-directory", type=Path, required=True)
    p.add_argument("--wait-for", type=Path)
    p.add_argument("--layers", type=int, nargs="+", default=[3, 10, 27])
    p.add_argument("--timeout-seconds", type=int, default=7200)
    args = p.parse_args()
    if not args.layers or min(args.layers) < 0 or max(args.layers) > 27:
        p.error("local Qwen capture requires layer indices 0..27")
    if args.timeout_seconds < 1:
        p.error("positive timeout required")
    out = args.output_directory.resolve()
    out.mkdir(exist_ok=False)
    proc = reader = None
    report = dict(valid=False, scope="Cached Stage-2 solver fixture capture; no global memory/time gain claimed")
    def progress(message):
        print("[cached-capture] " + message, flush=True)
    def interrupt(sig, frame):
        raise KeyboardInterrupt(f"signal {sig}")
    signal.signal(signal.SIGTERM, interrupt)
    def execute(label, command, env):
        nonlocal proc
        (out / "active.json").write_text(json.dumps(dict(label=label, command=command)))
        progress("starting " + label)
        # Mirror live child lines into this driver's log as well as its own log.
        with (out / (label + ".log")).open("x") as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return proc.wait()
    try:
        if args.wait_for:
            progress("waiting for existing queue; no GPU job launched")
            while not args.wait_for.exists():
                time.sleep(5)
            report["preceding_status"] = read(args.wait_for)
            time.sleep(5)  # Existing single progress reader drains its terminal status.
        # Never run CUDA concurrently with an experiment or another compute job.
        while True:
            check = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                   text=True, capture_output=True, check=True)
            if not any(line.strip().isdigit() for line in check.stdout.splitlines()):
                break
            progress("waiting for active CUDA compute processes to exit")
            time.sleep(10)
        reader = subprocess.Popen([sys.executable, "-u", str(ROOT / "tools/stream_memory_research_progress.py"),
            "--queues", str(out), "--output", str(ROOT / "logs/realq-system-restart-20261010-1408.log"),
            "--append"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        reference = args.reference_run.resolve()
        status, manifest = read(reference / "status.json"), read(reference / "manifest.json")
        if not status.get("valid") or status.get("returncode") != 0 or not status.get("sources_unchanged"):
            raise RuntimeError("reference must be a completed valid full run")
        if manifest["expected_ranks"]["stage2"] != 1:
            raise RuntimeError("capture tool is single-rank only; no multi-GPU claim")
        if shutil.disk_usage(ROOT).free < 3 * 1024**3:
            raise RuntimeError("capture requires at least 3 GiB free disk")
        bound = manifest["source_sha256_at_launch"]
        for filename, expected in bound.items():
            if digest(filename) != expected:
                raise RuntimeError(f"reference source changed: {filename}")
        helper = ROOT / "tools/hessian_bottleneck_capture.py"
        runner = Path(__file__).resolve()
        new_sources = {str(path): digest(path) for path in (helper, runner,
            ROOT / "tools/replay_hessian_bottleneck.py",
            ROOT / "tools/hessian_solver_workspace_prototype.py",
            ROOT / "tools/hessian_inverse_lifetime_prototype.py")}
        cache_before = cache_hashes(reference, progress)
        if not cache_before or not any(key.endswith(".pt") for key in cache_before):
            raise RuntimeError("reference static Fisher cache is empty")
        command = list(manifest["stage2_command"])
        entry_index = command.index("baseline-profile") - 1
        entry_source = Path(command[entry_index]).read_text()
        anchor = "try:\n    runpy.run_path('ptq.py', run_name='__main__')"
        if entry_source.count(anchor) != 1:
            raise RuntimeError("capture installation boundary changed; review entry")
        capture = out / "solver_fixtures"
        provenance = dict(reference_run=str(reference), reference_manifest_sha256=digest(reference / "manifest.json"),
            source_sha256=bound, capture_sources=new_sources, static_cache_sha256=cache_before,
            reference_command=manifest["stage2_command"], layers=args.layers,
            note="Post-dead-column/act_order actual solver H. Captured run is intrusive; no Adam or allocator checkpoint.")
        (out / "capture_provenance.json").write_text(json.dumps(provenance, indent=2))
        insertion = ("from tools.hessian_bottleneck_capture import install_capture as _capture_install\n"
                     f"_capture_provenance=json.loads(Path({str(out / 'capture_provenance.json')!r}).read_text())\n"
                     f"_capture_install(impl, {str(capture)!r}, {args.layers!r}, _capture_provenance)\n"
                     "def _capture_forbid_precompute(*args, **kwargs):\n"
                     "    raise RuntimeError('static cache miss; capture must not silently rerun Stage-0')\n"
                     "impl.collect_static_end_to_end_saliency_and_fisher=_capture_forbid_precompute\n\n")
        entry = out / "entry.py"
        entry.write_text(entry_source.replace(anchor, insertion + anchor))
        compile(entry.read_text(), str(entry), "exec")
        command[entry_index] = str(entry)
        set_option(command, "--exp", "bottleneck_capture_" + out.name)
        set_option(command, "--quant_stop_layer", str(max(args.layers)))
        # Retain held-out evaluation for full trajectory identity checks.
        (out / "manifest.json").write_text(json.dumps(dict(command=command, provenance=provenance,
            entry_sha256=digest(entry), cache_reused=True, stage0_run=False), indent=2))
        env = os.environ.copy()
        env.update(RESEARCH_RUN_DIR=str(out), HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", REALQ_PHASE_PEAK_MODE="exact")
        code = execute("capture-stage2", ["timeout", "--signal=TERM", "--kill-after=30s",
                                         str(args.timeout_seconds), *command], env)
        report["capture_returncode"] = code
        report["cache_unchanged"] = cache_before == cache_hashes(reference, progress)
        report["sources_unchanged"] = all(digest(path) == expected for path, expected in {**bound, **new_sources}.items())
        if code or not report["cache_unchanged"] or not report["sources_unchanged"]:
            raise RuntimeError("capture failed, cache changed, or source changed")
        actual_layers = {x["layer"] for x in read(capture / "index.json")["records"]}
        if actual_layers != set(args.layers):
            raise RuntimeError("capture missed requested actual layer boundaries")
        if max(args.layers) == 27:
            actual_metrics = read(out / "baseline-profile_metrics.json")
            report["state_matches_reference"] = actual_metrics["state_sha256"] == status["metrics"]["state_sha256"]
            report["layer_outputs_match_reference"] = read(out / "layer_output_fingerprints.json") == read(reference / "layer_output_fingerprints.json")
            quality = re.search(r"Exact KL&PPL on wikitext2: ([0-9.eE+-]+), ([0-9.eE+-]+)",
                                (out / "capture-stage2.log").read_text(errors="replace"))
            report["kl_ppl_matches_reference"] = bool(quality) and (float(quality.group(1)), float(quality.group(2))) == (status["kl"], status["ppl"])
            if not all(report[key] for key in ("state_matches_reference", "layer_outputs_match_reference", "kl_ppl_matches_reference")):
                raise RuntimeError("capture trajectory deviated from accepted state/layer outputs")
        replay = out / "replay"
        code = execute("solver-replay", [sys.executable, "-u", str(ROOT / "tools/replay_hessian_bottleneck.py"),
                                        "--capture", str(capture), "--output", str(replay)], env)
        report["replay_returncode"] = code
        report["replay_status"] = read(replay / "status.json") if (replay / "status.json").exists() else None
        report["valid"] = code == 0 and bool((report["replay_status"] or {}).get("valid"))
        if not report["valid"]:
            raise RuntimeError("replay failed; inspect before any full candidate run")
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
            journal.write("\n\n### 真实Hessian瓶颈捕获及组件重放\n\n"
                          "**理论支持与文献。** OLLA最后消费者与存储位置原则（https://arxiv.org/html/2210.12924v2）；"
                          "PyTorch cholesky/cholesky_inverse 的out=接口。\n\n"
                          "**具体方法。** 复用已完成、来源绑定且内容核验的Stage-0缓存，只运行一次实际Stage-2，"
                          "保存指定层首个down_proj子组的真实求逆输入/输出，保持dtype/stride。后续每方法独立进程重放。\n\n"
                          f"**实际效果。** 见 `{out / 'status.json'}`，valid={report['valid']}。"
                          "捕获/重放不是新增全程优化收益；失败不从记录删除。\n\n"
                          "**评价与反思。** 解决开发阶段重复预计算与求逆筛查成本，未实现完整block/Adam状态恢复，"
                          "不恢复CUDA allocator历史，不能据此认定层回放reserved回升已解决。"
                          "只有全流程候选通过原资源与质量门槛，才可作为终稿收益。\n")
        if reader is not None:
            reader.wait(timeout=15)
        progress(json.dumps({k: v for k, v in report.items() if k not in ("preceding_status", "replay_status")}))


if __name__ == "__main__":
    main()
