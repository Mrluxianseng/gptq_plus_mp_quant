"""Append measured capacity and recomputation-cost evidence when available.

Capacity reporting only reads small completed JSON files. Cost reporting waits
until the complete timing queue exits, so it cannot overlap measured timers.
"""
import argparse
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def wait(path):
    while not path.exists():
        time.sleep(5)


def append_once(marker, text):
    journal = ROOT / "docs/memory_speed_optimization_journal.md"
    if marker not in journal.read_text(encoding="utf-8-sig"):
        with journal.open("a", encoding="utf-8") as stream:
            stream.write("\n\n" + marker + "\n" + text + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capacity-directory", type=Path, required=True)
    p.add_argument("--followup-directory", type=Path, required=True)
    p.add_argument("--tag-suffix", required=True)
    p.add_argument("--output-directory", type=Path, required=True)
    args = p.parse_args()
    out = args.output_directory.resolve()
    out.mkdir(exist_ok=False)
    capacity = args.capacity_directory.resolve()
    followup = args.followup_directory.resolve()
    wait(capacity / "status.json")
    rows = []
    for model in ("qwen17b", "qwen06b"):
        path = capacity / model / "result.json"
        if not path.exists():
            rows.append(dict(model=model, completed=False))
            continue
        value = read(path)
        reference = value.get("candidate_exactness", {})
        candidate = value.get("arms", {}).get("candidate", {})
        candidate_path = Path(candidate.get("directory", "")) / "status.json"
        state = read(candidate_path) if candidate_path.exists() else {}
        rows.append(dict(model=model, budget_mib=value["budget_mib"],
            control_cuda_oom_stages=value.get("arms", {}).get("control", {}).get("cuda_oom_stages", []),
            candidate_completed=bool(candidate.get("valid")),
            candidate_exact_to_reference=bool(reference.get("valid")),
            capacity_contrast_established=value.get("capacity_contrast_established", False),
            candidate_nvml_peak_mib=state.get("max_nvml_memory_used_mib"),
            candidate_allocated_peak_bytes=state.get("full_process_peak_allocated_bytes"),
            candidate_reserved_peak_bytes=state.get("full_process_peak_reserved_bytes"),
            kl=reference.get("kl"), ppl=reference.get("ppl"), source=str(path)))
    result = dict(capacity_status=read(capacity / "status.json"), rows=rows,
                  scope="Allocator budget only; no physical small-GPU or GPU-count claim")
    (out / "capacity_summary.json").write_text(json.dumps(result, indent=2))
    lines = ["### AQ续记：两模型固定预算的实测结果", "",
             "**理论、方法与配置。** 延续AQ的PyTorch allocator限额验证：1.7B6400MiB、0.6B4096MiB，校准/量化/数据与AP已接受参考一致。参考文献及实现见AQ，不是新算法或物理小显存卡实验。", "",
             "**实际效果。**", "",
             "| 模型 | 分配器预算MiB | baseline OOM阶段 | candidate完整完成 | 与参考cache/state/28层/KL/PPL精确一致 | 整卡NVML peak MiB |",
             "| --- | ---: | --- | --- | --- | ---: |"]
    for row in rows:
        lines.append(f"| {row['model']} | {row.get('budget_mib', '未执行')} | {row.get('control_cuda_oom_stages', [])} | {row.get('candidate_completed', False)} | {row.get('candidate_exact_to_reference', False)} | {row.get('candidate_nvml_peak_mib', '未知')} |")
    lines += ["", "**评价与反思。** 只有明确CUDA OOM且候选完整完成/精确一致时才算此固定预算可运行对比成立；非OOM错误、未完成或数值不一致均不算成功。预算是分配器限额，driver/NCCL不受该API限制，因此不把它写为物理4GB/6GB卡已经可运行或GPU数已经减少。OOM异常里的WDDM异常in-use数值不用于计量；native CUDA max计数可能被诊断阶段重置，全程峰值以profile status为准。原始失败日志全部保留。", "",
              f"**证据及后续。** {capacity}；完整摘要{out / 'capacity_summary.json'}。下一步按已排队方案扩大校准与评估、换数据种子，并测重算时间成本。"]
    append_once("<!-- two-model-allocator-capacity:" + capacity.name + " -->", "\n".join(lines))
    print("[record] capacity results preserved", flush=True)
    wait(followup / "status.json")
    costs = []
    for model in ("qwen17b", "qwen06b"):
        tag = model + "-checkpoint-ce-recompute-cost-threepairs-" + args.tag_suffix
        folder = ROOT / "outputs" / ("low_overhead_speed_" + tag)
        if not (folder / "status.json").exists():
            costs.append(dict(model=model, completed=False, note="Not launched/completed; not a measured negative"))
            continue
        value = dict(model=model, status=read(folder / "status.json"), source=str(folder))
        if (folder / "summary.json").exists():
            value["summary"] = read(folder / "summary.json")
        costs.append(value)
    result = dict(followup_status=read(followup / "status.json"), costs=costs,
                  scope="Time cost of accepted memory system, not a new speed-kernel contribution")
    (out / "cost_summary.json").write_text(json.dumps(result, indent=2))
    lines = ["### AR：省显存方案的重算成本复测", "",
             "**理论支持与具体方法。** Checkmate与Korthikanti的重算是用计算换显存，应同时报告成本。这里沿用AP已接受完整方案，不探索Triton。每模型三组交替顺序、fresh-stat校验，collector和正式量化用CUDA同步wall timer，hash在计时外。未删除极端值。", "",
             "**实际效果。**", "",
             "| 模型 | 已完成配对 | collector C/N秒 | 量化 C/N秒 | collector＋量化 C/N秒 |",
             "| --- | ---: | --- | --- | --- |"]
    for row in costs:
        s = row.get("summary", {})
        lines.append(f"| {row['model']} | {row.get('status', {}).get('completed_pairs', 0)} | {s.get('control_mean_static_collection_seconds', '未知')} / {s.get('candidate_mean_static_collection_seconds', '未知')} | {s.get('control_mean_seconds', '未知')} / {s.get('candidate_mean_seconds', '未知')} | {s.get('control_mean_collector_plus_quantization_seconds', '未知')} / {s.get('candidate_mean_collector_plus_quantization_seconds', '未知')} |")
    lines += ["", "**评价与反思。** 只对完整通过参考cache/state/KL/PPL的计时配对作成本解释；未完成/失败不作加速或成本均值。collector＋量化不包含模型准备、离线评估或事后hash，不能称完整端到端时间。若重算更慢，照实写GPU显存与时间权衡，不把省显存等同加速。扩展cal8/eval8/seed45配对的数值和资源门槛独立于这些cal2/eval2成本结果，失败配置不被合并或剔除。", "",
              f"**证据及后续。** {out / 'cost_summary.json'}及各计时目录原始paired_repetitions.json；正式配置/实际多卡仍待后续验证。"]
    append_once("<!-- two-model-recomputation-cost:" + followup.name + " -->", "\n".join(lines))
    (out / "status.json").write_text(json.dumps(dict(completed=True, finished=time.strftime("%F %T")), indent=2))
    print("[record] completed cost evidence preserved", flush=True)


if __name__ == "__main__":
    main()
