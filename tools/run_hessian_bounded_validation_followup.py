"""Wait for owned serial queues, then screen real last-layer solver fixtures.

No production-source changes, concurrent CUDA runs, or full-process gain
claims. Preserves every worker result and stops on numerical mismatches.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-run", type=Path, required=True)
    parser.add_argument("--wait-for", type=Path, nargs="+", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_directory.resolve()
    out.mkdir(exist_ok=False)
    bound = {str(ROOT / name): digest(ROOT / name) for name in (
        "tools/hessian_chunked_finite_prototype.py", "tools/replay_hessian_bounded_validation.py",
        "tools/run_hessian_bounded_validation_followup.py")}
    report = dict(valid=False, sources=bound, scope="Real layer27 solver component; not full memory acceptance")
    try:
        print("[bounded-hessian] Waiting for existing queues; no CUDA launched.", flush=True)
        for status in args.wait_for:
            while not status.exists():
                time.sleep(5)
        capture = read(args.capture_run / "status.json")
        if not capture.get("valid"):
            raise RuntimeError("capture correctness gate failed; inspect original failure")
        fixtures = args.capture_run.resolve() / "solver_fixtures"
        if 27 not in {row["layer"] for row in read(fixtures / "index.json")["records"]}:
            raise RuntimeError("real last-layer fixture is missing")
        while True:
            check = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                   text=True, capture_output=True, check=True)
            if not any(line.strip().isdigit() for line in check.stdout.splitlines()):
                break
            time.sleep(10)
        if any(digest(path) != expected for path, expected in bound.items()):
            raise RuntimeError("queued bounded-validation sources changed")
        command = [sys.executable, "-u", str(ROOT / "tools/replay_hessian_bounded_validation.py"),
                   "--capture", str(fixtures), "--output", str(out / "replay"),
                   "--layers", "27", "--methods", "reference", "workspace", "finite", "workspace_finite",
                   "--repeats", "3"]
        (out / "active.json").write_text(json.dumps(dict(label="last-layer-solver", command=command)))
        print("[bounded-hessian] Starting actual layer27 four-arm component screening.", flush=True)
        env = dict(os.environ, OMP_NUM_THREADS="1")
        with (out / "last-layer-solver.log").open("x") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        report["replay"] = read(out / "replay/status.json")
        report["sources_unchanged"] = all(digest(path) == expected for path, expected in bound.items())
        report["valid"] = bool(report["replay"].get("valid")) and report["sources_unchanged"]
        if not report["valid"]:
            raise RuntimeError("component or source gate failed")
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        report["finished"] = time.strftime("%F %T")
        (out / "status.json").write_text(json.dumps(report, indent=2))
        with (ROOT / "docs/memory_speed_optimization_journal.md").open("a", encoding="utf-8") as journal:
            journal.write("\n\n### 末层真实求逆：分块有限值检查局部对照\n\n"
                "**理论与文献。** OLLA张量寿命/临时存储调度（https://arxiv.org/html/2210.12924v2），"
                "PyTorch isfinite逐元素布尔判定；AND归约的精确等价，不替换数值分解。\n\n"
                "**具体方法。** 捕获的layer27 down_proj首组Hessian，reference/workspace/finite/workspace_finite，"
                "各独立进程三次；保留三步分解、stride、阻尼、重试和fallback。分块检查在CUDA归约标量，"
                "每矩阵仅原有一次主机判定，不逐chunk同步。\n\n"
                f"**实际效果。** valid={report['valid']}，证据 `{out / 'status.json'}` 及replay各臂报告。"
                "局部结果不计全程显存/速度成果，不能修复末层后续KL刷新或层3/10碎片的证明缺口。\n\n"
                "**评价与反思。** 6144平方的布尔输出理论36MiB，分块输出预算1MiB；"
                "isfinite内部临时量及kernel启动时间仍须实测。必须在完整真实轨迹各已测区间无回升后验收。\n")
        print("[bounded-hessian] " + json.dumps({k:v for k,v in report.items() if k != "replay"}), flush=True)


if __name__ == "__main__":
    main()
