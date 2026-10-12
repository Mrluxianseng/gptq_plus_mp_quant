# REAL-Q系统显存研究报告（草稿v1）

记录范围：2026-10-12 17:23之前已完成并独立验收的研究。两模型容量验证和0.6B时间成本验证已完成；1.7B时间成本验证仍在运行，本文件不是研究链完成声明。原始日志、绑定源码、失败status和中断工件全部保留。

## 结论和比较对象

在本机单GPU、cal2×1024 pilot工作量下，workspace＋消费H复用＋down-only稳定写回＋末组私有H死亡点释放＋回放前统计释放，保持两模型所有507个state、28层非空heldout输出指纹以及KL/PPL与各自control精确一致，两模型各三对fresh全流程的全局allocated/reserved/NVML采样峰均下降。

control已经包含前期接受的checkpoint和CE分块等系统改进。下表是新增组合相对这一control的增量收益；不能把既有通用技术的收益全部归为本次新增方法，不能线性相加不同基线下的百分比。

| 模型 | 完整fresh配对 | 全流程allocated峰（MiB） | 全流程reserved峰（MiB） | 全设备NVML采样峰（MiB） |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | 3 | 5605.190918→4883.690918（−12.872%） | 5872→5332（−9.196%） | 6629→6089（−8.146%） |
| Qwen3-0.6B | 3 | 3151.033691→3010.087402（−4.473%） | 3490→3362（−3.668%） | 4247→4119（−3.014%） |

容量证据入口：[双模型六对独立汇总](../outputs/two-model-hardware-capacity-accepted-20261012-v1/summary.json)。每模型跨三次cache和全局峰值相同。设备是RTX4060 Laptop 8GB，环境为WSL Ubuntu-24.04、PyTorch2.11.0+cu128。实际权重BF16，Hessian求逆FP32，W4A16配置；没有降低求逆或权重计算精度来取得收益。

## 问题、理论与具体方法

求逆、量化和梯度刷新之间存在大矩阵的同时存活；量化最后消费者结束后，对象仍可能持有回放不再需要的统计。改进关注真实消费者、存储身份与死亡点，不改变原数学更新、校准数据、seed或Adam参数。

| 部分 | 工程处理 | 正确性依据和范围 |
| --- | --- | --- |
| 求逆工作区及H复用 | 复用已消费的H存储和工作区，减少同时存活的FP32矩阵副本 | 真实求逆覆盖、存储身份及原FP32计算一致；完整state和输出精确审查 |
| down-only稳定写回 | 在现有权重存储中完成目标dtype写回，减少转换时的额外分配 | 保留Parameter身份和目标dtype，不扩展到所有linear；每层实际调用记录必验 |
| 末组私有H释放 | 在该子组最后消费者结束后释放私有H_sub | 只释放死数据，保留仍被后续计算使用的因子；每层死亡点覆盖必验 |
| 回放统计生命周期 | AST核对后，将gptq/saliency/gradients对象删除提前到原after_layer_quantized hook之后、回放mark之前 | 保留hook、no_grad和实时module globals；无后续消费者/重绑定，CPU观察器不持有GPU Tensor |

[OLLA](https://arxiv.org/abs/2210.12924)提供生命周期与存储位置联合优化的研究依据；[Checkmate](https://arxiv.org/abs/1910.02653)说明内存容量与计算时间应作为可权衡的目标。本工作没有实现这些论文的ILP调度器，不能据引用声称复现其算法。[PyTorch2.11 CUDA文档](https://docs.pytorch.org/docs/2.11/notes/cuda.html)及已归档allocator源码用于解释allocated/reserved和段复用，不能仅凭allocator一般规则确认某次回升的唯一根因。

## 归因与独立验收

纯求逆归因证据记录4018→3298MiB，见[峰值归因](../outputs/hessian-peak-attribution-20261011-v1.json)。after_hessian_inverse是后续区间的起点；末组之后还有量化和梯度刷新，因此约5027MiB的后续峰不能全部归因于纯求逆。某层统计持有48.0234375MiB（1.7B）或12.01171875MiB（0.6B）也不等于完整流程峰值下降量。

每对独立验收覆盖全部507state、28层非空输出、KL/PPL、完整source/cache SHA、真实方法覆盖、两stage命令、runtime/GPU身份和完整遥测。完整方法包括112次down inverse/reuse、28次写回、28次dead观察和28次stats观察。每臂实际cache完整13文件，Windows长路径须使用extended路径，不能遗漏Fisher。

六对累计17766个对应区间和71064个独立start/end allocated/reserved point值，保留stage/rank/logkind/phase/occurrence身份。1.7B没有局部回升。0.6B每对layer25 replay reserved区间及两个对应边界point有+96MiB，累计3个区间/6个point；原strictFalse保留。

2026-10-12 11:15用户将主目标明确为设备全局容量：数值、工作量、来源及方法不变，full process allocated/reserved与全设备NVML采样峰各不高于control。局部回升仍完整报告，但不单独否决容量结果。这是用户验收目标变更，原失败status和比较器没有被回写成成功。NVML是采样结果，不证明连续每刻峰值；也不证明某个更小GPU一定能完成同任务。

## 已完成的时间成本证据

低扰动入口关闭可选dense memory/quant/event profiling及NVML/phase peak resets，保留共享CPU方法/存储观察。CUDA同步的collector和quant timer不含准备、评估或hash时间，因此其加和不是端到端延迟。实际source、fresh13cache、507state、all28输出、KL/PPL和方法metadata在计时后逐臂、逐对验收。

0.6B预定三个顺序是C→N、N→C、C→N，完整保留所有值：

| 配对 | control quant秒 | candidate quant秒 | control/candidate |
| --- | ---: | ---: | ---: |
| 1 | 1096.531017 | 988.822695 | 1.108926 |
| 2 | 811.464644 | 1249.974340 | 0.649185 |
| 3 | 1002.827937 | 1047.339430 | 0.957500 |

control量化均值970.274533秒、sampleSD145.294537秒；candidate均值1095.378822秒、sampleSD137.043361秒。candidate均值增加12.8937%，collector＋quant均值增加12.8720%。一快两慢、CV约15.0%/12.5%，本组三对没有稳定加速证据；观测平均成本增加，尚不能确认其唯一原因或方法固有惩罚。不能删除慢值、只展示首对或为了改善均值重跑。

证据：[0.6B三对fresh完整成本汇总](../outputs/replay-stats-low-overhead-cost-20261012-v3/independent_three_pair_qwen06_summary-20261012-1655.json)。1.7B时间成本仍待完整三对，不能用0.6B结果替代它，也不能将历史dense插桩秒数用作低扰动速度结论。

## 负向分支、工具错误和选择理由

| 分支/事件 | 完成证据与处理 |
| --- | --- |
| finite或all-linear写回 | 组件/前缀有局部收益，完整前缀出现reserved回升；原严格门槛拒绝，未进入full，不重复相同失败寻优 |
| dead-only缓存前缀 | 数值精确但layer3/10 replay reserved各+36MiB；earlystats组合消除此现象，唯一分配根因仍未确认 |
| 0.6B大小写回policy | 原layer25回升消失但转移到其他层；全局收益与简方案相同，未推广复杂度和重复 |
| 整层compact native池 | 完整BF16结果精确且局部零回升，但NVML/allocated没有比简方案进一步下降，candidate reserved3362→3394多32MiB；不推广池方案 |
| compact规格检查错误 | FP16组件被误作实际BF16规格，原失败保留，独立BF16架构审查纠正工具判定；没有修改权重精度 |
| 主机重启中断 | 13:24 Windows重启中断旧0.6B第三对，原不完整candidate不计入成功；新目录fresh恢复该对并完整审查，不补造旧status |
| 计时记录包装错误 | v2后eval写工件使用未定义root，all28工件缺失，0完整pairs；版本化仅修destination表达式并实际CPU执行回归，v3真实两臂完整验收；旧源/失败保留 |
| Windows路径inventory遗漏 | 普通路径漏MAX_PATH Fisher导致12文件清单；extended完整13文件重审，原错误证据保留，不误判真实cache改变 |

prepared checkpoint在GPU工作结束且完整数值/source/cache审查后按逐字SHA只读hardlink保所有路径和字节，Fisher仍独立。该操作只管理磁盘空间，不是显存贡献，失败和中断工件未据此删除。

## 限制与报告待完成项

本报告可支持当前pilot的系统存储优化结论；尚不支持正式2048样本、多卡、更小GPU容量边界、跨架构通用效果或稳定加速声明。论文贡献应围绕消费者/死亡点约束下的存储复用方法、正确性说明、可解释的消融和容量边界形成，不能把大量重复验收次数或常规del操作单独当作创新。

当前研究链仍须完成1.7B预定三对低扰动成本、逐对fresh独立审查、两模型成本汇总与最终报告。最后报告应保留速度代价和负向分支，显存可作为主目标；用户关于硕士论文是否需要速度的讨论不构成停止当前实验或自动启动额外正式工作量实验的指令。
