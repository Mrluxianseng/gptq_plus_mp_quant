"""Evidence-gated next ablation after a correct but resource-failed pool trial.

No concurrent GPU experiment, no unchanged retry, no parameter sweep. The next
candidate removes H-buffer reuse/pool isolation and retains early lifetime
release, separating the two causes of the observed fragmentation regression.
"""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_full_pipeline_memory_campaign import stop_process_tree


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary-queue", type=Path, required=True)
    p.add_argument("--output-directory", type=Path, required=True)
    p.add_argument("--fallback-directory", type=Path, required=True)
    p.add_argument("--tag-suffix", required=True)
    args = p.parse_args()
    out = args.output_directory
    out.mkdir(exist_ok=False)
    child = None
    reader = None
    report = dict(valid=False, next_method_started=False,
                  method="early-release only; actual measured fragmentation ablation, no original-method correctness study")

    def interrupt(sig, frame):
        raise KeyboardInterrupt(f"signal {sig}")

    signal.signal(signal.SIGTERM, interrupt)
    try:
        (out/"active.json").write_text(json.dumps(dict(label="await-pool-result", command=[])))
        print("[inverse-followup] waiting for pool queue; no GPU process", flush=True)
        while not (args.primary_queue/"status.json").exists():
            time.sleep(5)
        parent = read(args.primary_queue/"status.json")
        report["primary_status"] = parent
        if parent.get("valid"):
            report.update(valid=True, note="Primary method completed requested measured gates; no fallback needed")
            return
        audits = []
        for tag in parent.get("campaigns", []):
            campaign = ROOT/"outputs"/("memory_campaign_"+tag)
            for file in sorted(campaign.glob("pair*_audit/comparison.json")):
                audit = read(file)
                if not audit.get("strict_measured_memory_gate_passed"):
                    audits.append((file, audit))
        if not audits or not all(a.get("exactness_passed") for _, a in audits):
            report["note"] = "No correct paired resource failure: inspect infrastructure/numerical bug before further GPU work"
            return
        report["trigger_audits"] = [str(f) for f, _ in audits]
        report["reflection"] = (
            "Private-pool iteration stayed numerically correct but fails memory gates. "
            "Separate early tensor release from H-buffer placement/reuse instead of repeating the same failed combination."
        )
        (out/"decision.json").write_text(json.dumps(report,indent=2))
        with (ROOT/"docs/memory_speed_optimization_journal.md").open("a",encoding="utf-8") as f:
            f.write("\n\n### AS进一步消融：私有池资源失败后的预设下一实验\n\n"
                    "**理论与参考。** OLLA将lifetime和location分开建模（https://arxiv.org/html/2210.12924v2）；"
                    "GMLake指出重算/卸载引入不规则分配会加重碎片（https://arxiv.org/html/2401.08156v1）。"
                    "不是复现原论文正确性，也不是扫参数。\n\n"
                    "**触发证据及方法。** 本次私有池版本数值正确但资源门槛失败，路径"+str(out/"decision.json")+"。"
                    "下一候选仅提前释放H_work/chol，移除H缓冲复用及独立池；保持量化/Adam/校准参数。"
                    "用已有真实CUDA两模型预检作为源码绑定前提，再fresh 1.7B layer0–2检查真实Adam轨迹；"
                    "通过才运行完整28层三对。\n\n"
                    "**评价及反思。** 正向峰值不能覆盖局部资源回升，失败保留；本段只是实际触发后的下一方案，"
                    "尚无下一实验效果。结果由完成记录器按方法/效果/评价/反思格式追加。\n")
        command = [sys.executable,"-u",str(ROOT/"tools/run_hessian_inverse_lifetime_followup.py"),
                   "--preflight",str(ROOT/"outputs/hessian-inverse-lifetime-cuda-20261011-0945/status.json"),
                   "--output-directory",str(args.fallback_directory),"--tag-suffix",args.tag_suffix,
                   "--variant","early-release","--models","qwen17b","--prefix-stop-layer","2","--full-pairs","3"]
        report["next_method_started"] = True
        log = out/"early-release-controller.log"
        with log.open("x") as stream:
            child = subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            # Primary's reader exits when its status arrives. Leave it ample
            # time before launching the next single writer for fallback logs.
            time.sleep(5)
            reader = subprocess.Popen([sys.executable,"-u",str(ROOT/"tools/stream_memory_research_progress.py"),
                                       "--queues",str(args.fallback_directory),"--output",
                                       str(ROOT/"logs/realq-system-restart-20261010-1408.log"),"--append"],
                                      cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            while child.poll() is None:
                active = args.fallback_directory/"active.json"
                if active.exists():
                    try:
                        # Mirrors the active child's --tag to the single total
                        # log reader; starts no additional CUDA workers.
                        (out/"active.json").write_text(active.read_text())
                    except (OSError,json.JSONDecodeError):
                        pass
                time.sleep(5)
        report["returncode"] = child.returncode
        if reader is not None:
            reader.wait(timeout=10)
        if (args.fallback_directory/"status.json").exists():
            report["fallback_status"] = read(args.fallback_directory/"status.json")
            report["valid"] = bool(report["fallback_status"].get("valid"))
        # Full cache/reference audit happens only after this GPU queue exits.
        if report["valid"]:
            command = [sys.executable,str(ROOT/"tools/audit_hessian_inverse_lifetime_followup.py"),
                       "--queue",str(args.fallback_directory),"--output-directory",str(out/"independent-audit"),
                       "--tag-suffix",args.tag_suffix,"--variant","early-release","--models","qwen17b"]
            audit_code = subprocess.run(command,cwd=ROOT,check=False).returncode
            independent = read(out/"independent-audit/status.json")
            report["valid"] = not audit_code and independent.get("valid",False)
            report["independent_audit"] = independent
    except BaseException as exc:
        if child is not None and child.poll() is None:
            stop_process_tree(child.pid)
            child.wait()
        if reader is not None and reader.poll() is None:
            reader.terminate()
            reader.wait(timeout=10)
        report["error"] = repr(exc)
        raise
    finally:
        report["finished"] = time.strftime("%F %T")
        (out/"status.json").write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
