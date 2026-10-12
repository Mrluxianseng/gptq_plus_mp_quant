# REAL-Q系统显存研究报告：当前pilot研究链完成版v1

记录范围：截至2026-10-12 21:31，预定两模型各三对容量实验及各三对低扰动成本实验完成。最终CPU复审重新核对全部六对内存记录和十二个计时臂。本报告完成当前pilot研究链，不表示正式规模或论文全部验证完成。所有原始源码、日志、失败status、中断工件、草稿和方法整理v1保留。

## 结论与比较对象

在RTX4060 Laptop 8GB、WSL Ubuntu-24.04、PyTorch2.11.0+cu128、本机cal2×1024 pilot工作量下，workspace＋消费H复用＋down-only稳定写回＋末组私有H死亡点释放＋回放前统计释放，保持两模型全部507个state、28层非空heldout输出指纹及KL/PPL与各自control精确一致。每模型三对fresh Stage0＋28层量化＋评估的全局allocated/reserved及NVML采样峰均下降。

control已包含前期接受的checkpoint、CE分块及输入卸载等系统改进。下表是新增组合相对这一control的增量收益；不能将既有通用技术收益全部归为本方法，不能相加不同基线的百分比。实际权重BF16、Hessian求逆FP32，量化配置W4A16；没有通过改变求逆精度、校准、seed或Adam取得收益。

| 模型 | fresh完整配对 | allocated峰（MiB） | reserved峰（MiB） | 全设备NVML采样峰（MiB） |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | 3 | 5605.190918→4883.690918（−12.872%） | 5872→5332（−9.196%） | 6629→6089（−8.146%） |
| Qwen3-0.6B | 3 | 3151.033691→3010.087402（−4.473%） | 3490→3362（−3.668%） | 4247→4119（−3.014%） |

每模型跨三次完整缓存和全局峰值相同。[原容量六对汇总](../outputs/two-model-hardware-capacity-accepted-20261012-v1/summary.json)和[最终重新执行的六对验收](../outputs/realq-final-complete-acceptance-20261012-v1/fresh_memory_six_pair_summary.json)分别保留原验收与最终复审证据。

低扰动计时完整保留全部三次/模型：候选量化均值在1.7B增加2.3086%，在0.6B增加12.8937%，存在明显波动，没有稳定加速证据。显存优化作为主结果，时间分布作为成本报告；不将均值增加认定为已确认的方法固有惩罚或唯一热/功耗原因。

## 问题、理论与具体方法

REAL-Q顺序分组路径的求逆、块量化、梯度刷新和激活回放之间存在大矩阵同时存活，以及最后消费者结束后仍被对象持有的死统计。方法依据最后消费者、存储别名、dtype/shape/stride及原执行顺序安排复用与释放，保持原数学更新。理想活跃存储按不同storage计数，不能重复计算共享view；这一理想量不等于实测allocated，更不能直接推导缓存分配器reserved或设备NVML。

| 机制 | 具体处理 | 正确性约束及实现 |
| --- | --- | --- |
| 两工作区求逆 | 首个Cholesky完成后以已消费H_work承接逆矩阵out；求逆完成后以已消费chol承接最终因子out | 求解算子输入/out不重叠，保持原FP32运算和布局；阻尼重试保留原私有H。[workspace](../tools/hessian_solver_workspace_prototype.py) |
| 已消费H接管因子 | 当前子组原H无后续消费者后，将最终因子复制到对应self.H子组存储，保留原因子布局 | 未消费子组H和仍需保留因子不得覆盖；受限于已验证down_proj/alpha=0等顺序路径。[H复用](../tools/hessian_inverse_lifetime_prototype.py) |
| down-only稳定写回 | 在现有目标权重存储copy_同一FP32结果，执行原目标dtype转换，避免另建长期权重分配 | 原更新图已消费、目标连续且device/shape兼容，验证Parameter及实际存储身份。[写回](../tools/hessian_stable_weight_prototype.py) |
| 末组私有H死亡点 | 求解与诊断最后消费者之后，释放局部变量持有的末组私有H_sub | 不与保留因子共享存储，不提前释放仍需重试/诊断内容。[死亡点](../tools/hessian_dead_subgroup_input_prototype.py) |
| 回放统计死亡点 | 将gptq、saliency_dict、gradients_dict删除移至原after_layer_quantized hook之后、回放mark之前 | AST确认无后续消费者/重绑定；保留no_grad、实时globals、原hook及单wrapper。[统计生命周期](../tools/activation_replay_stats_lifetime_prototype.py)、[接入](../tools/activation_replay_stats_install.py) |

CUDA执行仍须满足所在流和分配器顺序，不能将Python返回解释为任意跨流工作已完成。CPU观察器只存整数，不能延长GPU Tensor寿命。这些条件说明特定实现中每次变换为何允许，不是所有REAL-Q配置逐位一致的普遍定理；当前真实路径的数值一致性由独立实验检验。

[REAL-Q原论文](https://arxiv.org/abs/2609.00049v2)提供被优化量化算法；块梯度修正数学与精度收益归于REAL-Q。[OLLA](https://arxiv.org/abs/2210.12924)提供生命周期与存储位置联合优化依据，[Checkmate](https://arxiv.org/abs/1910.02653)提供内存与计算成本权衡的依据。本实现未实现这些工作的ILP调度器。[PyTorch2.11 CUDA文档](https://docs.pytorch.org/docs/2.11/notes/cuda.html)及[归档allocator源码](literature/PyTorch_v2.11.0_CUDACachingAllocator.cpp)用于解释allocated/reserved、段复用和池生命周期；一般规则不能确认某次局部回升的唯一根因。参考文献记录在[references_memory_systems.bib](references_memory_systems.bib)。

## 归因、完整数值与容量验收

[纯求逆归因](../outputs/hessian-peak-attribution-20261011-v1.json)记录约4018→3298MiB。after_hessian_inverse是后续区间起点，末组之后还含量化及梯度刷新，约5027MiB后续峰不能全归纯求逆。1.7B统计持有48.0234375MiB/层、0.6B持有12.01171875MiB/层，均不是独立全流程峰下降量。组件、前缀、区间峰和全流程峰不能互换。

最终每对重新核对全部507state、28层非空输出、KL/PPL、完整source/13文件cache流式SHA、真实方法覆盖、两stage命令、runtime/GPU身份、完整遥测和全局峰。方法记录包括112次down inverse/reuse、28次稳定写回、28次dead和28次stats观察；控制组的启用状态也独立核对。缓存Windows extended路径包含Fisher，普通路径12文件清单不完整。

| 模型 | 全部507state SHA | C/N精确KL | C/N精确PPL |
| --- | --- | ---: | ---: |
| 1.7B | 237fd17bf88c35a522dc91f4a5c3c0bf519fd3d707967ce0a115350232053187 | .625495195388794 | 25.562191009521484 |
| 0.6B | 7ad9b1a7a890faf5175b136b0c12ea30228bea96060a84f9ee734cd2f0c57ff3 | .30534985661506653 | 25.013607025146484 |

每对2961个对应精确区间峰及11844个独立start/end allocated/reserved值，六对累计17766区间/71064point，保留stage/rank/logkind/phase/occurrence身份，无缺失。区间峰和[audit_memory_boundary_points](../tools/audit_memory_boundary_points.py)逐point审查分别执行，未以max(start,end)替代。

1.7B三对没有局部回升。0.6B每对layer25 activation_replay_and_cleanup reserved区间及两个对应边界值各+96MiB，累计3区间/6point；allocated未在这些点回升，原strictFalse保留。2026-10-12 11:15用户将主目标明确为全局硬件容量：数值、工作量、来源及方法不变，full-process allocated/reserved与全设备NVML采样峰各≤control。局部变化完整报告，不单独否决容量结果。没有改写旧比较器、status或失败证据，也没有将局部≤自身global这一必然关系当证据。

NVML是采样峰，不是连续每刻证明；峰下降不证明某个更小GPU一定能运行同任务。

## 完整低扰动时间成本

每模型预定C→N、N→C、C→N三对。关闭可选dense memory/quant/event profiling、NVML和phase peak resets，保留共享CPU方法/存储观察，故不是零开销。collector与quant使用CUDA同步的time.monotonic差值，计时不含准备、评估、hash；加和不是端到端延迟。每臂在计时后核对507state、all28非空输出、KL/PPL、61绑定source、实际entry、13完整缓存、两stage命令及完整方法metadata。全部原值保留，不剔慢值，不为改善均值重跑。

| 模型 | 对 / 顺序 | C quant秒 | N quant秒 | C collector秒 | N collector秒 | C合计秒 | N合计秒 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6B | 1 / C→N | 1096.531017 | 988.822695 | 2.773641 | 2.648891 | 1099.304658 | 991.471586 |
| 0.6B | 2 / N→C | 811.464644 | 1249.974340 | 2.554307 | 2.800949 | 814.018951 | 1252.775289 |
| 0.6B | 3 / C→N | 1002.827937 | 1047.339430 | 2.942660 | 3.253914 | 1005.770597 | 1050.593345 |
| 1.7B | 1 / C→N | 2498.588323 | 2355.394039 | 4.139627 | 4.640292 | 2502.727950 | 2360.034331 |
| 1.7B | 2 / N→C | 1759.998356 | 2052.422812 | 4.102323 | 4.552469 | 1764.100679 | 2056.975281 |
| 1.7B | 3 / C→N | 1756.192298 | 1745.818439 | 4.641346 | 4.205735 | 1760.833644 | 1750.024174 |

| 模型 | C quant均值 ± sampleSD（秒） | N quant均值 ± sampleSD（秒） | N量化均值变化 | C/N变异系数 | collector+quant均值变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.6B | 970.274533 ± 145.294537 | 1095.378822 ± 137.043361 | +12.8937% | 14.97% / 12.51% | +12.8720% |
| 1.7B | 2004.926326 ± 427.528066 | 2051.211763 ± 304.789604 | +2.3086% | 21.32% / 14.86% | +2.3122% |

0.6B一快两慢；1.7B两次候选稍快、一次较慢，但两模型候选量化均值都增加。这组三对/模型没有稳定加速证据，也不建立唯一波动原因或方法固有惩罚。不能混算模型，不能只展示快值，不能用历史dense插桩秒数替代低扰动计时。

19:07[只读时钟观察](../outputs/replay-stats-low-overhead-cost-20261012-v3/independent_clock_runtime_observation-20261012-1907.json)显示tqdm墙钟elapsed与ps进程elapsed不一致；主机电源/时间筛选无返回，原因未确证。原CUDA同步monotonic计时与日志保留，不用墙钟修timer，不确认暂停、热、功耗或分配器为唯一原因。该环境不确定性限制时间效果解释，后续更严格性能研究须额外控制并记录运行环境。

证据：[0.6B最终六臂fresh成本复审](../outputs/replay-stats-low-overhead-cost-20261012-v3/independent_three_pair_qwen06_summary-final-20261012-2129.json)、[1.7B三对六臂fresh成本复审](../outputs/replay-stats-low-overhead-cost-20261012-v3/independent_three_pair_qwen17_summary-20261012-2128.json)。计时入口没有新的dense峰遥测，不能从计时结果补造区间或point内存证据。

## 负向分支、工具错误与反思

| 分支或事件 | 证据及处理 |
| --- | --- |
| finite或all-linear写回 | 组件/前缀存在局部收益，但真实前缀reserved回升；原严格门槛拒绝，full为0pairs，没有重跑相同失败配置寻优；最终组合不含此分支 |
| dead-only前缀 | 数值精确，layer3/10 replay reserved各+36MiB；earlystats组合消除此现象，但唯一分配根因及单项独立全局贡献未确认 |
| 0.6B大小写回policy | 原layer25回升消失，转移到其他层，全局收益与简方案相同；没有推广或混入同配置三次重复 |
| 整层compact native池 | 完整BF16结果精确，局部零回升，但allocated/NVML无额外下降，候选reserved比简方案3362→3394多32MiB；不推广额外池、搬运及同步复杂度 |
| compact规格检查错误 | FP16组件规格误作实际BF16规范，原失败保留，独立实际BF16架构/覆盖审查纠正工具判定；FP16组件不证明BF16全程，没有降低精度 |
| 主机重启中断 | Windows13:24重启中断旧0.6B第三对；不完整候选未计成功，不补旧status。新目录fresh恢复该中断对并完整审查；底层重启动机未确证，不归实验OOM |
| 计时包装记录错误 | v2 eval之后写指纹使用未定义root，all28工件缺失，0完整pairs；新版本仅修destination表达式，CPU真实执行包装回归及v3完整GPU工件验收。旧源和失败保存，control-only时间不算配对 |
| 算法命令检查接入错误 | 旧比较器漏登记已审查存储flag，主动停止0完整pairs；版本化adapter登记该flag，其他算法/数据参数比较保持，不改旧冻结源 |
| Windows长路径inventory遗漏 | 普通路径漏Fisher，仅12文件；extended路径实际13文件SHA重审，保留原错误，不误判真实缓存变更 |

prepared checkpoint仅在GPU臂结束且完整数值/source/cache验收后按字节SHA只读hardlink，保留全部路径与字节、独立Fisher；失败及中断工件未据此删除。此操作只管理磁盘容量，不是显存贡献。

组合容量下降不等于各项独立贡献，组件和前缀不能拼成full消融。不断追局部零回升会增加复杂度而无全局收益；选择符合用户硬件目标的最简已验证组合。工具错误须与方法失败分开，修复记录/规格接入不能改数学或降低门槛遮掩错误。

## 最终验收与贡献边界

[最终容量与成本绑定汇总](../outputs/realq-final-complete-acceptance-20261012-v1/final_capacity_and_cost_summary.json)确认六对容量、六对计时及61计时source未变。内存六对在21:28重新执行全部数值/source/cache/方法、精确区间峰及独立每point/global审查，与原status/cache/global证据相同；0.6B局部诊断完整保留。计时两模型各六臂再次重新验收，原六对controller身份、顺序、SHA及timer与独立记录精确一致。此次最终复审仅CPU读取工件，无新增GPU实验。

统一方法定位见[方法贡献整理v1](realq_storage_method_contributions_20261012_v1.md)：依据消费者/死亡点与存储别名约束，协同管理求逆工作区、已消费H因子接管、稳定写回和边界释放。REAL-Q量化数学、既有checkpoint/CE通用技术、OLLA/Checkmate生命周期思路不归为本方法原创。五处实现属于一个协同方法，审查工具及重复次数主要支撑证据完整性。不宣称通用最优、首创、已完成全面新颖性排查或跨所有REAL-Q设置成立。

本报告支持当前pilot全流程显存下降及数值一致性，不支持正式2048样本、多卡、更小GPU实际可运行边界、跨架构通用效果或稳定加速。论文后续需要同工作量且保留依赖的规范消融、对象/消费者表、正式规模及实际容量边界实验；它们是待扩展研究，没有自动启动额外GPU队列。

当前授权研究链的预定实验、必要完整验收、文献记录、负向及工具错误反思、最终报告已完成。原草稿及方法整理v1保留其当时“1.7B成本待完成”的时间快照，以本报告和最终汇总更新结论。原始结果未覆盖，未推送。
