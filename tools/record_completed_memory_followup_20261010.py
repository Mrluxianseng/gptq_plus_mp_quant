"""Record completed serial experiments after the GPU/timing queue has exited.

Consumes authoritative campaign audits, never promotes a launch or a partial
pair to a result. Does no CPU reporting during low-overhead GPU timing.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for", type=Path, required=True)
    parser.add_argument("--campaign-tags", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--journal", type=Path, default=ROOT / "docs/memory_speed_optimization_journal.md")
    parser.add_argument("--only-unrecorded", action="store_true",
                        help="Record failed/partial follow-up campaigns without re-summarizing successful entries already in the journal")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    print("[record] waiting for GPU/timing queue exit before CPU summaries", flush=True)
    while not args.wait_for.exists():
        if time.monotonic() - started > 24 * 3600:
            raise RuntimeError("completion dependency did not arrive within 24 hours")
        time.sleep(5)
    records = []
    journal = args.journal.resolve()
    if not journal.exists():
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.write_text("# Completed memory records\n", encoding="utf-8")
    for tag in args.campaign_tags:
        marker = f"<!-- completed-memory-followup:{tag} -->"
        if args.only_unrecorded and marker in journal.read_text(encoding="utf-8-sig"):
            records.append(dict(tag=tag, already_recorded=True))
            continue
        campaign = ROOT / "outputs" / f"memory_campaign_{tag}"
        if not (campaign / "status.json").exists():
            records.append(dict(tag=tag, launched_or_completed=False, note="Skipped or not completed; not a negative method result"))
            continue
        status = json.loads((campaign / "status.json").read_text())
        rows_path = campaign / "paired_repetitions.json"
        if not rows_path.exists() or not json.loads(rows_path.read_text()):
            record = dict(tag=tag, audited_pairs=0, status=status,
                          note="No complete paired audit; inspect worker errors, not a measured method result")
            records.append(record)
            continue
        report = out / tag
        subprocess.run([sys.executable, str(ROOT / "tools/summarize_memory_campaign.py"),
                        "--campaign", str(campaign), "--output-dir", str(report)], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL, timeout=180)
        summary = json.loads((report / "summary.json").read_text())
        audits = [json.loads((campaign / f"pair{row['pair']}_audit/comparison.json").read_text())
                  for row in json.loads(rows_path.read_text())]
        passed = status.get("valid") and all(a["exactness_passed"] and a["strict_measured_memory_gate_passed"] for a in audits)
        model_full = all(a.get("full_model_quantization_validated", False) for a in audits)
        first = audits[0]
        source = json.loads((Path(first["control_directory"]) / "manifest.json").read_text())["stage2_command"]
        def option(flag):
            return source[source.index(flag) + 1] if flag in source else "default/unspecified"
        stats = summary["paired_statistics"]
        marker = f"<!-- completed-memory-followup:{tag} -->"
        if "hinv-workspace-stable" in tag:
            method_note = (
                "本轮是数值正确资源失败后的solver工作区分配调度：保留原三步Cholesky/inverse/upper调用，"
                "用最后消费者后的H_work、chol作为out=目标，每步输入/输出不共享存储，保持原因子stride。"
                "仅非inplace damping使用工作区复用；原inplace分支回到原helper。"
                "之后仍复用消费H组存储、向既有down_proj权重copy_，不启用私有池。"
                "参考[OLLA](https://arxiv.org/html/2210.12924v2)的lifetime/location原则和PyTorch2.11 out=接口；"
                "不是算法替换，不改变REAL-Q的Adam、校准或量化参数；局部组件证明不能替代全程严格资源审计。"
            )
        elif "hinv-stable-weight" in tag:
            method_note = (
                "本轮为AS后续的存储位置修正：两侧共享已接受checkpoint＋CE；候选保留已消费H组存储复用，"
                "不启用私有池，将down_proj最终量化权重copy_至既有权重存储，避免长期权重重新分配。"
                "保留原solver/因子布局/量化顺序/Adam/校准数据，附两侧相同的CPU allocator段归属观察。"
                "参考[OLLA](https://arxiv.org/html/2210.12924v2)的生命周期和存储位置原则；未实现其ILP。"
                "必须核对实际转换、缓存/每层输出/最终state/KL/PPL和全阶段资源；诊断计时不计加速。"
            )
        elif "hinv-isolated-pool" in tag:
            method_note = (
                "本轮为AS续记的down_proj求逆私有池修正；两侧共享已接受checkpoint＋CE，候选保持原求逆运算、"
                "精度和因子stride，提前释放临时矩阵并将结果复制到已消费H组；求逆临时量在独立CUDA native池分配，"
                "确认无活跃池张量后同步并释放池，避免临时大矩阵与后续长期存活张量混用存储段。"
                "参考[OLLA](https://arxiv.org/html/2210.12924v2)的存储位置/碎片分析及"
                "[PyTorch2.11局部MemPool文档](https://docs.pytorch.org/docs/2.11/notes/cuda.html)。"
                "不实现OLLA的ILP；不改变Adam/量化算法；同步代价属于待测项，诊断计时不作为加速证据。"
            )
        elif "hinv-reuse" in tag or "hinv-early-release" in tag:
            method_note = (
                "本轮是AS节down_proj Hessian求逆存储生命周期的独立消融。两侧共享已接受的checkpoint＋CE系统；"
                "候选提前释放最后一次使用后的H_work和chol；hinv-reuse标签还将每个已消费H分组的GPU存储"
                "复用于该组逆Cholesky因子，保留参考因子的shape/dtype/stride和全部solver调用。"
                "仅验证alpha0、顺序分组、无Hessian正则和无H/Hinv CPU卸载路径；不改变Adam或量化算法。"
                "参考[OLLA](https://arxiv.org/html/2210.12924v2#S2.SS2)的张量生命周期/内存位置思想，"
                "不实现其ILP调度器，也不将已有checkpoint收益再次计入本组件。"
            )
        elif "lazy-export" in tag:
            method_note = (
                "本轮是AJ节未消费导出副本的按需分配：当export_to_et=False且无diagnostic_recorder时，"
                "省去完整/子组W_int和Scale副本；保留Q、逐列量化和block整数码/scale临时量。"
                "导出/诊断消费者存在时仍分配原缓冲。参考[Capuchin](https://www.microsoft.com/en-us/research/?p=690597)"
                "的tensor生命周期管理，不实现其自动调度器。共享HCPU或checkpoint/CE以manifest为准，组件估算不能代替峰值实测。"
            )
        elif "cpu-saved-kl-logits" in tag:
            method_note = (
                "本轮是AH节logits保存位置的独立消融：两臂共同checkpoint＋CE，候选仅把KL节点保存的student/teacher logits转到CPU，"
                "反向逐token块恢复，并保留原始完整LM-head前向/反向GEMM形状。"
                "参考[Capuchin](https://alchem.usc.edu/portal/static/download/capuchin.pdf)的tensor生命周期管理与"
                "[CCE](https://arxiv.org/abs/2411.09009v2)对词表中间量开销的分析；不实现其自动调度器或融合CE算法。"
                "CPU搬运代价及调用者额外持有logits的影响必须另外核对，不预设能够加速。"
            )
        else:
            method_note = (
                "本轮配置与改动以保存manifest为准；checkpoint/CE组合延续Z/AD/AF节的重算和分块方法，"
                "元数据或完成block迁移延续AB节生命周期假设。"
                "[Korthikanti等](https://arxiv.org/abs/2205.05198)、[Checkmate](https://arxiv.org/abs/1910.02653)、"
                "[GMLake](https://arxiv.org/abs/2401.08156)支持资源调度/碎片化分析，不表示实现其所有算法。"
            )
        lines = ["", marker, f"### 自动续记：{tag}（{time.strftime('%F %T')}，已完成审计）", "",
                 "**理论支持、文献与具体方法。** " + method_note, "",
                 f"**实际配置与效果。** 模型 `{option('--model')}`，校准 `{option('--nsamples')}×{option('--seq_len')}`，refresh microbatch `{option('--backward_bsz')}`，量化截止层 `{option('--quant_stop_layer')}`。已审计 `{summary['audited_pairs']}/{summary['requested_pairs']}` 对，完整模型量化验证 `{model_full}`，campaign valid `{status.get('valid')}`。",
                 "", "| 指标 | control均值 | candidate均值 |", "| --- | ---: | ---: |"]
        for key, label in (("nvml_peak_mib", "整卡采样peak MiB"), ("allocated_peak_mib", "完整进程allocated peak MiB"),
                           ("reserved_peak_mib", "完整进程reserved peak MiB"), ("host_rss_peak_mib", "CPU RSS peak MiB")):
            lines.append(f"| {label} | {stats[key]['control_mean']:.4f} | {stats[key]['candidate_mean']:.4f} |")
        lines += ["", f"首对KL/PPL C `{first.get('control_kl')} / {first.get('control_ppl')}`，N `{first.get('candidate_kl')} / {first.get('candidate_ppl')}`。每对数值门槛 `{[a['exactness_passed'] for a in audits]}`，资源门槛 `{[a['strict_measured_memory_gate_passed'] for a in audits]}`；区间回升数量 `{[a['sampled_phase_regression_count'] for a in audits]}`、边界回升数量 `{[a['synchronized_boundary_regression_count'] for a in audits]}`。", "",
                  "**评价与反思。** " + ("所有请求配对通过已测门槛；仍只覆盖保存的本地pilot规模，不是正式论文规模/4卡证明。" if passed else "本轮未通过完整请求/严格门槛；即使局部peak下降也不能作为合格最终方案，失败配对保留，不剔除。") + " prefix不能称为完整量化；NVML为采样，不能证明连续每一时刻。这里的均值是各次peak的均值，不是时间平均显存。诊断计时不能支持加速。原因分析需结合原始profile和代码，不以这个自动摘要替代人工审查。", "",
                  f"**证据及后续。** `{campaign.relative_to(ROOT)}/`、`{report.relative_to(ROOT)}/summary.json`及原始每对comparison。完整量化与系统资源通过后才进入相同源码的低开销计时；尚未通过则先定位，不继续重复同一失败配置。", ""]
        if marker not in journal.read_text(encoding="utf-8"):
            with journal.open("a", encoding="utf-8") as stream:
                stream.write("\n".join(lines))
        records.append(dict(tag=tag, audited_pairs=summary["audited_pairs"], accepted=bool(passed),
                            full_model_quantization_validated=model_full, summary=str(report / "summary.json")))
    (out / "status.json").write_text(json.dumps(dict(completed=True, records=records,
                       finished=time.strftime("%F %T"), manual_interpretation_still_required=True), indent=2))
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
