# REAL-Q 全流程显存验证与终稿证据边界

## 当前研究对象与贡献范围

项目根目录 `14103_REAL_Q_E2E_LLM_Quantizat.pdf` 第 9 页第 7 节及第 22 页附录 E(3) 提出的系统问题包含 Stage-0 全精度模型驻留、每层 O(d²) Fisher、保存激活和动态 Block-GD 的资源开销。本文研究工程实现，保留量化目标、更新公式与补偿顺序。最终评估的低显存实现作为完整工作流的辅助能力，单独压低评估峰值不能代替校准/量化阶段的主贡献。

外部参考包括 [Cut Cross-Entropy](https://arxiv.org/abs/2411.09009) 的大词表中间量生命周期设计、[ZeRO-Offload](https://www.usenix.org/conference/atc21/presentation/ren-jie) 的 CPU/GPU 分层存放先例，以及 [PyTorch CUDA memory documentation](https://docs.pytorch.org/docs/stable/notes/cuda) 对 allocated、reserved 和缓存释放的定义。本项目未直接实现这些论文的全部算法。方法细节、负向结果和迭代依据持续记录在 `docs/memory_speed_optimization_journal.md`。

## 验证层次

| 层次 | 配置/对象 | 可支持的结论 |
| --- | --- | --- |
| CUDA/函数级验证 | 同一真实模型维度、相同输入和残差 | 单个算子的数值行为、临时量与边界；不能证明全流程效果 |
| 本地全流程 pilot | 两模型、fresh Stage-0、全 28 层、固定 held-out 输入 | 该缩减配置下的全流程显存与精确性 |
| 校准工作集扩展 | 固定方法，增加校准长度/样本量 | 峰值迁移、是否仍可运行；这是资源规模测试，不是调参寻找量化质量 |
| 正式配置服务器验证 | W4A16 per-row、2048 WikiText-2 sequences、2048 calibration tokens、block size 128、32 gradient samples/step；其他条件明确固定 | 论文规模下的工程 A/B，结合多 GPU 的每 rank 指标 |
| 非插桩性能复测 | 同配置、空闲设备、至少三组、交替执行顺序 | 时间分布、GPU-hours 和加速；诊断 profile 耗时不能代替它 |

截至本文件建立时，本轮匹配 pilot 为 16×128 校准、2×1024 评估、seed 44、W4A16 per-row、4 saliency groups、block size 128、单 RTX 4060 Laptop GPU。Stage-0 使用项目已有 FSDP CPU-offload/meta-init 路径，双方共享 KL chunk16 和 static-label chunk16。以上共享优化是该轮对照条件的一部分，因此下游增量比较不能宣称包含这些共享优化相对未经优化 REAL-Q 的全部收益。

`static_fisher_label_chunk_tokens` 改变 categorical sampling 的随机抽样轨迹，即使分布相同也不能在 A/B 的一侧单独修改并宣称严格等价。每项优化必须保持两侧该设置相同。

论文未公开若干 clipping、microbatch、评估长度、采样调度细节，见 `docs/REALQ_PAPER_PROTOCOL.md`。对齐公开的样本数量和位宽不代表已精确复现论文表格；正式结果需列出补充实现条件及基线来源。

## 显存审计与强标准

`tools/compare_full_pipeline_profiles.py` 比较两个已完成 `tools/run_full_pipeline_memory_profile.py` 的运行目录；不初始化 CUDA，不影响量化路径。它核对：

1. 完整 Stage-0/Stage-2 成功与源码快照一致。
2. 校准、量化和并行条件一致；实验名/cache 路径以及列出的系统存放/实现开关允许不同。
3. 逐层指纹覆盖直到 `quant_stop_layer` 的所有层；权重 state SHA、tensor count、KL、PPL 严格相同。
4. NVML raw CSV 的最大值与汇总相符；记录采样缺口和诊断性样本均值。
5. Stage-0、Stage-2 和全程 allocated/reserved/NVML 高水位分别不回升。
6. 匹配 stage/rank/phase 的 sampled maxima 和同步 start/end 边界；阶段回升不会被较低全局峰值掩盖。

`--compare-cache` 还会逐文件比较新建 static cache。CLI 若未通过正确性或严格采样显存门槛返回 2，同时保留完整报告，供诊断负向实验。输出为 `comparison.json`、`summary.md` 和逐阶段 CSV。示例：

```bash
python tools/compare_full_pipeline_profiles.py \
  --control outputs/full_pipeline_profile_<control-tag> \
  --candidate outputs/full_pipeline_profile_<candidate-tag> \
  --output-dir outputs/audit-<unique-tag> \
  --compare-cache
```

阶段采样存在 5 ms 分辨率限制，短暂分配可能被其中一臂漏采；同步边界可以辅助区分真实驻留与漏采。此工具仍不能证明两个不同时间轨迹在每个瞬间的大小关系。终稿应采用可审计表述“全局峰值下降、已测对应阶段无回升”，并明确测量分辨率，不写“数学上保证任何时刻都更低”。reserved 包含缓存；NVML 是整卡占用，包含运行之外的 GPU 资源，不能把它称为纯 tensor 内存。

## 2026-10-10 严格复核带来的结论修正

1.7B Hinv 暂存+原位阻尼的第二对仍保持 28 层指纹、state、KL/PPL 相同，全程 NVML 从 4459 降到 4325 MiB。但匹配阶段审计发现缓存 reserved 回升：例如 layer 17 replay 开始的 allocated 两边都是 1,475,357,696 B，reserved 为 1,660,944,384 / 2,168,455,168 B，candidate 高 484 MiB。563 个阶段的同步边界 reserved 最大值高于对应 control；同期同步 allocated 边界没有回升。sampled allocated 的额外差异不能未经检查当作真实增耗，因为短暂峰值可被 5 ms sampler 漏采。

因此现阶段证据支持“1.7B 总峰值优化”，尚未满足“每个对应阶段都不回升”的更严格目标。原记录中的越界表述已修正。下一步需诊断分配生命周期与 allocator fragmentation；代码已在模块 `free()` 中执行 `empty_cache()`，单纯追加相同调用缺乏依据。应先检查 inactive split/segment 占用，形成明确工程改动后再跑对照。改变 allocator 环境作为诊断时须记录到 manifest，且不将一个环境开关单独包装成主要代码贡献。

原始审计：`outputs/audit-qwen17b-hinv-inplace-boundaries-r2-20261010/`。候选 telemetry 有超过 500 秒的时间戳采样缺口，进一步说明该对计时和时间加权均值不能作为速度/平均显存结论。

## 当前进行中的实验与后续条件

0.6B CE256 首对已完成：16×128 校准 / 2×1024 评估，完整 NVML 峰值 3499→2173 MiB，缓存、权重、28 层指纹、KL/PPL 精确相同。旧 allocator 高水位只覆盖量化，不能称为含最终评估；故该对尚未满足完整 allocator 范围的强门槛，需用新测量器独立配对。方法仅作用于评估，不把它归为 Stage-0 memory-efficient backward 的主贡献。

长序列 0.6B 的 Stage-0 2×1024 首对已完成：只开启 saved-tensor CPU offload，NVML 6469→2763 MiB，缓存相同；精确区间每次出现的 phase（包括每个 batch 的重复 backward）分别核对，0 个区间和同步边界回升。对应审计 `outputs/audit-qwen06b-stage0-2x1024-offload4-occurrences-r1-20261010/`。Stage-0-only 模式必须明确声明 `quantization_and_quality_validated=false`，不能被引用为完整量化精确性。

随后在同条件下执行 fresh Stage-0→28 层量化→2×1024 评估，采用 candidate→control 顺序；只有 offload 控制变量不同，Stage-2 不叠加 Hinv 暂存、原位 damping 或 CE chunk。2026-10-10 02:51 首对完成：NVML 6469→3499 MiB（−45.91%），13 个缓存文件、28 层输出、507-tensor state SHA 和 KL/PPL 完全相同；精确区间及同步边界均无回升，完整 allocator 门槛通过。主机 RSS 约 5.92→12.18 GiB，诊断计时不支持加速结论。证据在 `outputs/full_pipeline_profile_qwen06b-full-2x1024-*-exact-r1-20261010/` 和 `outputs/audit-qwen06b-full-2x1024-offload4-exact-r1-20261010/`，完整数值已写回主记录 O 节。1.7B 的 2×512 Stage-0 配对已排入串行队列，避免并发 CUDA 进程影响测量；它不是和 0.6B 2×1024 相同序列长度的尺度对照，后续需补相同长度才可做规模归因。

新 runner 使用完整进程 `full_process_peak_*` 字段，另给最终 KL/PPL 评估建立 `cuda_memory_evaluation_rank*.jsonl`，明确旧 `stage2_allocator_peak_*` 仍是量化工作区范围。正常预计算的 `SystemExit(0)` 被捕获以执行监控 footer，异常退出不被吞掉。`status.json` 新增 `valid`，防止 worker 返回 0、但缺少完整审计数据时误认为成功；主日志每 30 秒报告阶段存活与工作日志路径。

比对工具现在保留同名 phase 的 occurrence，避免对多个 batch 取最大值后掩盖单 batch 的回升；区分 exact allocator interval 与 5 ms sampled 模式。两种计数都不证明两个不同执行轨迹在任意连续时刻的支配关系，NVML 仍为 100 ms 外部采样。仅在 `strict_measured_memory_gate_passed=true` 且完成正式验证层次后，才能引用相应层次的强结论。

所有新结果均应写回实验记录。终稿所需的正式 2048×2048、多卡每 rank 验证和非插桩三对速度实验尚未完成，不能由现有 pilot 替代。

## 2026-10-10 04:28 后续进展

1.7B 2×512 Stage-0 已完成并修复了退出 reserved 回升：新鲜配对缓存精确相同，NVML 4867→3301 MiB，两侧退出 reserved 均为2 MiB，严格区间/边界检查通过。初次失败仍保留于主记录 P 节。完整量化配对随后串行执行，阶段结果不作为完整1.7B质量验证。

0.6B 卸载+去重的三组完整 campaign 已通过前两组，两组 NVML 均为6469→3499 MiB，权重/28层输出/KL/PPL逐位一致，所有已测区间/边界无回升。主记录 R 节保存每组范围和主机代价。第三组仍在运行，不提前宣布三组平均结果。
