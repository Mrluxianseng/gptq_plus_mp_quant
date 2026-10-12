# REAL-Q Adam 优化研究记录

> 本文是 Adam 适配方向的单一持续记录。每种方法均按“理论支持—参考文献—实现—实验—评价—反思”记载。做新实验前先检查此前证据与局限；每轮实验完成后更新本文件。REAL-Q 是研究基线和本项目目标来源，不计作外部相关工作。

最后更新：2026-10-07。当前代码路径：`python -m realq.ptq -> realq.pipeline -> realq.runner -> realq.refresh.block_gd`。除非另行注明，模型为 Qwen3-0.6B，GPU 为本地 RTX 4060 Laptop 8 GB，配置为 W4、8 条 WikiText-2 校准序列×64 tokens、16 条评估序列×128 tokens、Adam refresh seed 0。小样本实验只作筛选，不是论文规模结论。KL 是主要质量指标，PPL 是辅助参考指标。

## 执行原则与目前判断

**B6 后续工作优先级（2026-10-07 更新）。** 全层 layer0–27、64×256 校准、64×512 评估的 control/B6 配对复验已按预先固定 seed42/43/44 完成：KL 平均变化 −0.535%，仅2/3种子改善；PPL 平均变化 +0.113%，仅1/3改善。固定 refresh 快照的方向/幅度机制诊断也已完成，结果见 B6 小节：三个对照分支里，单纯匹配 L2 步长没有恢复重置 m 后的 KL，方向与幅度反事实均未出现跨点稳定的正 KL 信号。因此 B6 不算可用方法，不能按“m 重置、v 保留有效”推进跨层 v 累积；本轮也不再扫层范围、学习率或重置阈值。可继续调研曲率度量的信赖域思想，但只有在数学映射到 REAL-Q 已有 Adam refresh、无需额外预热反向传播且实现预算可控时，才做一个小型、预注册机制试验；否则应停止这个重置家族。方法筛选用验证集与论文最终测试集分开。

**B6 的算法归属与文献支持。** 在相同梯度、裁剪、二阶矩及其age下，每次清m并正确重置m-age令hat(m)=g，更新为 lr*g/(sqrt(hat(v))+eps)；在实数算术上等价于Adam取beta1=0，不能将清m开关本身宣称为新的优化器（[Kingma & Ba, Adam](https://arxiv.org/abs/1412.6980)）。浮点实现的乘除顺序仍需单独验证，不宣称逐位等价。后续可检验的REAL-Q机制是：量化/闭式补偿造成的状态变化，是否使一阶方向历史跟踪滞后，而保留二阶尺度历史仍有必要。Asadi et al.的[目标变化后重置优化器实验](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e4bf5c3245fd92a4554a16af9803b757-Abstract-Conference.html)提供间接实证动机，但其重置范围和RL问题与B6不同；Sahu et al.的[非平稳优化中的动量跟踪理论](https://proceedings.mlr.press/v306/sahu26a.html)区分噪声与漂移误差，并在强凸光滑、SGD动量设定下给出历史动量的漂移代价，不能当作Adam或GPTQ轨迹的收益定理；[Adam-Rel](https://papers.nips.cc/paper_files/paper/2024/hash/f2733d3b0dde1d74995f35a9cf442d38-Abstract-Conference.html)则提示步幅/bias age变化本身也能影响效果，所以B6的KL改善不足以证明旧m方向有害。这些论文支持上述机制问题及幅度对照，不保证B6在更大配置有效。

**B6 的贡献门槛。** 全层扩大配置和固定快照方向/幅度诊断均已完成，暂不扩展 Qwen3-1.7B，也不把 B6 包装成论文贡献。若以后出现清晰的新机制证据，再设计不增额外 backward 的全模型配对验证。工程化可研究在已证明质量策略之后真正不分配/不维护 m，以减少状态存储与写入；在质量策略尚未站稳前，不把状态缓冲节省包装为 Adam 质量贡献。收益上限是一份所覆盖活跃优化器的 m 缓冲，不是整个 REAL-Q 显存的一半；该工程化目前没有实现或性能数据。

**2026-10-06 后续决策更正（优先于下文历史计划）。** “保留 m/v、每次仅重置 bias age”已经在 B 节试验过，两种子短前缀 KL 均恶化；后文文献复查把它列为尚未实现/实验的新候选不准确。按检测到的阶段变化才重置 age 是另一个尚未定义的方法，不能据此重新运行旧的每-refresh重置消融。当前也暂停推广 layer3–26 的 m+age 重启，因为64×256校准、64×512评估的seed42结果已反转；这不等价于否定所有范围、模型及配置，但不能用此前小配置均值称已 work。

**已完成的 B6 限定机制诊断。** 按相同原则固定真实 refresh 快照的权重、活动后缀、梯度、二阶矩与 Adam step，在预先选定的 layer3/layer12 `mlp.down_proj`、block1、seed42/43 分支一次更新，并继续执行真实量化到 layer12 后评估。除 Adam 与 B6 reset_m 原更新外，比较了“B6 方向匹配 Adam 全局 L2”与“Adam 方向匹配 B6 全局 L2”；这只拆分更新向量方向和总范数，不是新算法。所有 16 个运行 returncode=0，四组目标快照的权重、梯度、m、v、step 哈希逐位一致，故分支起点公平；summary 在 `outputs/realq-adam-direction-b6-direction-magnitude-l3-l12-downproj-b1-seed42-43-logged-20261007/summary.json`，可复跑入口为 `tools/run_realq_adam_direction_snapshot.py`。

| 目标点 | 分支 | KL 相对变化 | PPL 相对变化 | 更新 L2/Adam L2 | 与 Adam 方向 cosine |
|---|---|---:|---:|---:|---:|
| layer3, seed42 | B6 reset_m | +5.70% | −1.43% | 1.0865 | .9466 |
| | B6 方向、匹配 Adam L2 | +2.32% | −1.66% | 1.0000 | .9466 |
| | Adam 方向、匹配 B6 L2 | +5.54% | +0.46% | 1.0865 | 1.0000 |
| layer12, seed42 | B6 reset_m | +0.56% | +0.24% | 1.1724 | .8595 |
| | B6 方向、匹配 Adam L2 | +0.30% | +0.32% | 1.0000 | .8595 |
| | Adam 方向、匹配 B6 L2 | +0.12% | +0.14% | 1.1724 | 1.0000 |
| layer3, seed43 | B6 reset_m | +7.09% | +0.82% | 1.0317 | .9629 |
| | B6 方向、匹配 Adam L2 | +2.92% | −0.24% | 1.0000 | .9629 |
| | Adam 方向、匹配 B6 L2 | +8.13% | +2.58% | 1.0317 | 1.0000 |
| layer12, seed43 | B6 reset_m | −0.03% | −0.36% | 1.1457 | .8692 |
| | B6 方向、匹配 Adam L2 | +0.46% | −0.12% | 1.0000 | .8692 |
| | Adam 方向、匹配 B6 L2 | +0.14% | +0.18% | 1.1457 | 1.0000 |

变化按同目标点 Adam 分支配对；此处是 16×128 评估的局部诊断，不是完整量化轨迹或论文规模结论。B6 方向匹配 Adam 范数后，四个点 KL 仍全部更差（+2.32%、+0.30%、+2.92%、+0.46%），均值 +1.50%；可见仅把 B6 步长缩到 Adam 的普通 L2 大小，不能修复该局部 KL 退化。保留 Adam 方向、把范数放大到 B6 水平后，KL 也四点全恶化（均值 +3.48%），说明这几个点增大总步长本身有风险；但不能据此推出所有 Adam 更新都应缩小。两个 raw B6 点 KL 降低不一致（其余点恶化），PPL 又混合，故无跨点方向收益。诊断只支持“全局 L2 不是足够的步长等价度量”；没有证明是 Fisher 方向误差或具体 token 导致。

**文献与适配边界。** Liu et al., *G²PTQ: Improving LLM Post-Training Quantization with Generalized Gradient Compensation* (arXiv:2609.31009, 2026) 在 PTQ 闭式一阶梯度补偿中加入信赖域缩放，依据量化误差与二次曲率估计限制补偿项，避免过大的权重更新；作者在 Qwen3-0.6B W4A16 报告较低 KL。该工作的问题背景与工程场景高度相关，但被缩放的是其 GPTQ/FOEM 型补偿项，公式所需 H^{-1}、量化误差 ΔL_GPTQ 也不是 REAL-Q 的 Adam 状态/更新，因此不能照抄其 β 公式，也不能宣称文献已经验证了 Adam 信赖域。论文还通过每层阈值搜索调 α，不能把此搜索照搬成“工程创新”。来源：[arXiv HTML](https://arxiv.org/html/2609.31009)，信赖域推导在 §4.1 / Theorem 3，调参流程在 Appendix C.2。

**下一步评价。** B6 重置家族当前停止，跨层 v 前提不成立。G²PTQ 的缩放若迁移到 Adam，需要控制局部曲率项 (u^	op J_y^	op FJ_y u)，而不是匹配普通权重 L2；REAL-Q 存有输出 Fisher (F)，但当前没有每个 refresh 的 (J_yu)。精确 JVP 或有限差分候选至少需要一次额外的块前向，且量化发生在每列块 refresh 后，直接照搬会反复付出全 block 成本。故它目前只是需要先做成本估算的研究候选，并非今晚已实现/验证的方法；没有证明可以廉价复用时，不把它放大到全模型。若后续只在能接受该前向成本的筛选配置中验证，则需把这一成本单列，并检查 trust-region 预测与真实下一 GPTQ 后 KL 是否相关。

**诊断工具默认值修正。** 为本轮临时增加的两个 L2 反事实 arm 仍保留在 `--policies` 的允许集合中，但已从默认 arm 集合移出；本轮结果显式传入完整 arm，不受该默认项修正影响。`py_compile`、REAL-Q Adam 刷新状态算术检查及 CLI 导入/默认策略检查均通过。该修改仅防止后续诊断运行意外增加无关 arm，不改 REAL-Q 默认量化/Adam 更新路径。

REAL-Q 的 Adam 更新发生在序贯量化中：当前列块量化/闭式补偿后，对尚未量化的权重后缀做一次 refresh；随后继续量化下一列块。模块内状态会跨 refresh 延续，模块之间通常重新初始化。要研究的不是“Adam 不适合离散更新”这一未经论证的说法，而是特定的状态、预条件器或更新方向能否在同样的 REAL-Q 轨迹中改善**量化后 KL**，以及局部浮点收益能否经由下一次 GPTQ 舍入和补偿保留。

约束：主方法不得为每个模块另加预热反向传播；在完全相同校准、评估、量化配置和种子下配对；检查 treatment 是否确实应用、源码是否一致、完整命令和原始日志是否保存；先做有机制根据的低成本检查，再推进完整模型；KL 为主，PPL 分开报告；单种子、短前缀或极小评估窗不得称为已 work。不得把扫参数包装成贡献。

**当前结论**：既有 Cautious 掩码、Grams 式方向替换、RAdam、AdaBelief、ADOPT、影子量化预览和跨层二阶矩迁移均未形成可用改进；其中多个方法完整模型负向。“仅重启活跃后缀一阶矩及其 bias age、保留二阶矩及其 age”先前在小配置和 32×128/32×256 配置出现正向筛选信号，但最新全28层 64×256/64×512、seed42/43/44 配对复验 KL 平均仅 −0.535%（2/3改善），PPL 平均 +0.113%（1/3改善），因此 B6 当前不算稳健 work，也不能据此触发跨层 v 累积研究。旧配置结果仍作为历史筛选保留，不与新配置合并。只筛 Fisher-MSE objective 后重算的逐 refresh loss（每 seed 1,971 个对齐块事件）显示，candidate 的等事件权重平均 loss 变化从 seed42 −0.120% 到 seed43 +0.629%、44 +0.342%、45 +0.462%、46 −0.404%、47 −0.133%；之前按 2,044 个混合 objective 事件得到的均值把末层真实 KL 也计入，不应称为纯 Fisher surrogate。只读矩漂移诊断确认 \(\hat v\) 与当前 \(g^2\) 会有差异，但不代表差异有害。v-only 重启灾难性恶化；跨层状态迁移短前缀曾正向但 full28 不复现。重要经验：局部优化机制的训练理论不能直接保证序贯量化质量；同一完整量化路径、终层真实 KL 更新和相同种子必须覆盖并审计。

### 当前 Adam 矩漂移观测（诊断，不是优化方法）

**理论/动机。** Adam 的 \(v_t\) 是过去梯度平方的 EMA，当前 \(g_t^2\) 与 bias-corrected \(\hat v_t\) 不一致时，坐标预条件器会受历史轨迹影响。Kim (2026) 在特定有限方差两点随机 oracle 中分析了长二阶矩记忆可能抑制进展；Adam-Rel 讨论非平稳阶段的 bias age。它们提供测量动机，不证明 REAL-Q 的量化 refresh 应重置 \(v\)：REAL-Q 的确定性/有限采样梯度及逐列 GPTQ 轨迹都不同。

**方法/实现。** `realq/refresh/adam_transfer_diagnostic.py::record_moment_drift` 在原 refresh 的正常 EMA 后，只读统计活跃后缀的 \(\hat v\) 与本次裁剪梯度平方的尺度比、行/列 tile 的 log-profile 相关和误差。无新 backward、无状态写入；计算聚合后再写 JSONL。

**实际效果。** Qwen3-0.6B，seed42，REAL-Q 默认 full-precision optimizer state，W4、act_order，量化到 layer8；校准 8×64、评估 16×128。共 468 条记录，覆盖 layer0–8 的 attention/FFN 四类投影。refresh0 因 \(\hat v=g^2\) 而是恒等点。之后差异高度不均匀，例如 layer1 `mlp.down_proj` 最后一次记录的均值尺度比为 14.41、行/tile profile 相关约 .64/.64；layer2 `mlp.down_proj` 尺度比 1.34、相关约 .87/.87；layer8 `mlp.down_proj` 约 .99、相关约 .94/.95。观测以活跃后缀为准，尾部列数在逐步缩小，不能将“最终比值”单独解释为纯历史效应。开启 observer 的量化输出与既有同配置 seed42 control 完全相同：KL .1270953715，PPL 60.08094025。

**评价/反思。** 这排除了 observer 改变结果的明显实现风险，并证明某些 refresh 的矩统计会与当前梯度平方显著错位；它没有证明错位导致 KL 变差，更没有证明清零会改善。必须做成对真实 REAL-Q 量化结果检验，并把偏置修正年龄正确拆开；以往只清 \(v\) 但沿用旧 bias age 的结果仅能说明错误尺度很危险，不能用来拒绝一个一致定义的 v-only 重启。

## 方法记录

### A. warm_adam：数据估计二阶矩初值

**理论支持。** Adam 的初始二阶矩 (v_0=0) 使短轨迹初期的自适应步长敏感。Abuduweili 与 Liu 研究了用数据驱动或随机非零初始二阶矩缓解初期不稳定，因而有文献依据调查 REAL-Q 的模块冷启动。但 Adam 标准 bias correction 在平稳矩假设下抵消零初始化导致的期望缩放；不能简单声称零初始化必然造成系统低估。REAL-Q 中更可信的问题是梯度非平稳、样本有限和少数离散 refresh。

**参考文献。** Kingma & Ba, *Adam: A Method for Stochastic Optimization* (2014), [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)；Abuduweili & Liu, *Revisiting the Initial Steps in Adaptive Gradient Descent Optimization* (CPAL 2025), [PMLR](https://proceedings.mlr.press/v280/abuduweili25a.html)。

**方法/实现。** `ModuleGradSecondMoment` 从若干批已有 refresh backward 中估计逐坐标 (E[g_{batch}^2]=\bar g^2+Var_s(g_s)/B)，初始化 (v)，并以虚拟初始步校正二阶矩；一阶矩从零开始。它不需要额外类型的 forward，但需要额外的每模块 backward 采样。

**实际效果。** `warm_start_steps=0` 的 null test 只核对管线退化到基线，不是质量证据。layer 0 的 prior 与实际逐坐标梯度平方 log-correlation 约 0.50–0.81，但诊断只量化 layer 0，未评估完整质量。未找到可用的 warm_adam 全模型匹配 KL/PPL 对照。

**评价。** prior 可测，尚未证明有益；且额外 per-module backward 违反当前主约束，不能作为主要路线或贡献。

**反思/后续。** 不再为 warm_adam 做更多预热反传。若未来能够从正常 refresh 中免费复用统计，才重新评估；需与等预算基线比较总时间与最终 KL。

### B. 动量/二阶矩和 bias age 重置

**理论支持。** Adam 的 EMA 可在梯度变化时产生跟踪滞后；Adam-Rel 研究非平稳强化学习中目标变化后的局部时间与 bias correction；Kim (2026 preprint) 则在一个特定坐标级随机 oracle 下证明长二阶矩记忆可能抑制更新进展。这些说明状态 age、矩值和记忆长度需分别考虑，但 REAL-Q 的有限、确定性、随量化/补偿改变的梯度轨迹与上述设定不同，不能直接移植“每 refresh 清零”或调低 \(\beta_2\)。

**参考文献。** Kingma & Ba (2014)，同上；Ellis et al., *Adam on Local Time: Addressing Nonstationarity in RL with Relative Adam Timesteps* (NeurIPS 2024), [论文页](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f2733d3b0dde1d74995f35a9cf442d38-Abstract-Conference.html)；Kim, *Second-Moment Memory in Coordinatewise Adam* (2026 preprint), [arXiv:2608.15824](https://arxiv.org/abs/2608.15824)。

**方法/实现。** 在 layer 0–4 小前缀中，对照保留 Adam 状态、每次只清零 (m)、只清零 (v)、仅重置 bias age，以及先前的全部重置试验。每种策略与控制使用相同的反向次数和配置。该实验在实现审计后明确：(m) 与 (v) 有独立 age；对 (v) 重置但继续沿用较大 age 会改变有效更新尺度，不能把它解释成干净的状态冷启动。

**实际效果。** seed 42 保持状态 KL/PPL=.098157/58.044；重置 (m) 为 .097187/57.659（KL 改善、PPL改善）。seed 43 基线 .098095/59.182；重置 (m) 为 .103951/59.028（KL恶化、PPL略改善）。只重置 (v) 灾难性恶化：seed42 KL/PPL=3.8458/1810.56，seed43=2.6527/564.41。只重置 bias age 的 KL 分别 .109712、.106193，均差于各自控制；PPL 分别 57.925、59.022。更早的全部重置试验 seed 间结果不一致，详见 `outputs/adam-history-isolated-moments-20261005.md`。

**评价。** 不稳健的短前缀证据否定了这些具体重置政策；它没有证明保留状态是最优，也没有验证按真实梯度漂移触发的选择性重置。(v) 重置灾难说明它的尺度对下游量化特别敏感。

**反思。** 不能把 bias-correction 代数误当成历史矩本身问题，也不能用低精度 EMA 的“更新停滞”解释本项目：REAL-Q 当前 Adam 的 (m,v) 缓冲区为 FP32。Kim 的理论只给出后续值得直接测“平方梯度记忆与当前信号失配”的动机，不为减小本项目 \(\beta_2\) 提供现成参数。以后若重做，必须分别维护 (m/v) age、保持更新量尺度可比，并在相同快照上先诊断梯度漂移，再做真实后续 GPTQ 验证。当前不优先重跑固定周期重置。

### B2. 保留 m、每次 refresh 重启 v 与 v-age（单变量机制对照）

**理论支持。** 假设是逐 refresh 的量化/闭式补偿使当前平方梯度统计迅速改变，旧 \(v\) 会给当前活动坐标保留不合适的分母尺度/轮廓。Adam-Rel 的局部时间处理和 Kim (2026) 的二阶矩记忆分析提供相关机制背景，但都不直接支持每列块清空 v。这个实验只回答 REAL-Q 里 \(v\) 的短期历史是否有净作用，不宣称它是新优化器。

**参考文献。** Kingma & Ba, *Adam: A Method for Stochastic Optimization* (2014), [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)；Ellis et al., *Adam on Local Time* (NeurIPS 2024), [论文页](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f2733d3b0dde1d74995f35a9cf442d38-Abstract-Conference.html)；Kim, *Second-Moment Memory in Coordinatewise Adam* (2026 preprint), [arXiv:2608.15824](https://arxiv.org/abs/2608.15824)。上述来源与 REAL-Q 的优化轨迹并不相同，属于间接理论动机。

**方法/实现。** treatment 在每次已有 backward 后，仅对当前活跃权重后缀清零 \(v\)，并把独立 \(v\)-age 置为当前更新步 1；\(m\) 与 \(m\)-age 均沿用原 REAL-Q。此时本步 bias-corrected \(\hat v\) 恰为当前经相同梯度裁剪后的 \(g^2\)。不新增 backward，不改变采样/闭式 GPTQ 步骤；control 与 candidate 完整 REAL-Q 前缀量化配对。修复了之前候选只清 \(v\) 却沿用旧 age 的混杂。代码由 `REALQ_ADAM_HISTORY_POLICY=reset_v_and_v_age_each_refresh` 触发；单元测试检查新 age、保留 m 轨迹及旧错误政策的差别。

**实际效果。** Qwen3-0.6B，seed42，W4、act_order，8×64 校准、16×128 评估，layer0–4 前缀；相同 `realq.ptq` runner 配对。control KL/PPL=.0979209/58.0521；候选=1.234226/150.0302（KL约 12.60 倍，PPL约 2.58 倍）。两组运行成功、候选 policy marker 已出现、源码哈希一致，反向次数及其他配置相同。逐块日志第一处明显偏离出现在 layer1 `self_attn.v_proj` block3：refresh objective .00373→.10808；block4 .00262→.11187。之后多处漂移，layer1 `self_attn.o_proj` block2 已到 3.50，layer1 `mlp.down_proj` 后续块超过 2,500。量化后 KL/PPL 也明显变坏。这是筛选即停止的负结果，无需追加 seeds 来确认一个灾难性结果。

**评价。** 虽然年龄和 \(\hat v=g^2\) 的算术单测正确、treatment 确实生效，这个真实量化前缀结果仍强烈否定“保留历史 m、每步把 v 换成当前 g²”作为默认策略。它没有证明原始 Adam 最优；但证明 m 与 v 的时间统计在本项目中不可随意解耦。额外 seed 对同一个灾难性筛选没有决策价值，因此停止，不扫重置频率/学习率。

**反思。** 逐 refresh 的 \(\hat v\) 轮廓与当前 \(g^2\) 不同，并不意味着旧信息应丢弃：它可能正好为历史 \(m\) 提供所需的稳定分母；layer1 `v_proj` 的早期发散支持“矩耦合/分母突变敏感”这一解释，但尚未直接比较更新范数，故应视为机制假设。下一个方法应保留原始矩累积，只针对 Adam 的 early-step adaptive-denominator 方差做有文献根据的比较，先检查所需假设是否适用于当前小样本、短确定轨迹。

### B3. RAdam early-step 方差校正

**理论支持。** RAdam 从 Adam 自适应学习率的方差出发，使用 \(\rho_t\) 估计二阶矩样本数是否足以使用自适应分母；当 \(\rho_t\) 不足时退回一阶动量更新，满足条件后再启用方差校正的自适应分母。当前每个线性模块的 early refresh 很少，且确实出现过早期分母干预后严重发散，因此这是比直接丢弃 \(v\) 更保守的可检验候选。适用性仍存疑：RAdam主要分析随机优化 early-time adaptive-rate variance，而 REAL-Q 是短、有限校准轨迹，并且原 \(lr\) 是为 Adam 设计；照搬其 unrectified 阶段可能使有效步长过小。

**参考文献。** Liu et al., *On the Variance of the Adaptive Learning Rate and Beyond* (ICLR 2020), [arXiv:1908.03265](https://arxiv.org/abs/1908.03265)；PyTorch RAdam 公式与阈值的[官方文档](https://docs.pytorch.org/docs/stable/generated/torch.optim.radam.RAdam_class.html)。

**方法/实现。** 按标准 RAdam 计算 \(\rho_\infty=2/(1-\beta_2)-1\)、\(\rho_t=\rho_\infty-2t\beta_2^t/(1-\beta_2^t)\)，在 \(\rho_t>5\) 时使用方差 rectification，否则使用 bias-corrected 一阶动量；保留原 REAL-Q 的 backward、样本、GPTQ 舍入/补偿和 \(m,v\) 历史，不增加预热反传。以相同 Adam learning rate 做一次 layer0–4、seed42 小前缀对照，目的是先看是否出现明显不适配；不扫描阈值或学习率。

**实际效果。** 三个 seed 的 layer0–4 配对筛选均正常结束；candidate marker 生效、源码哈希不变、其他配置和 backward 次数匹配：

| seed | Control KL → RAdam KL | KL 相对变化 | Control PPL → RAdam PPL | PPL 相对变化 |
|---:|---:|---:|---:|---:|
| 42 | .097921 → .091888 | −6.16% | 58.052 → 59.288 | +2.13% |
| 43 | .108895 → .113525 | +4.25% | 59.485 → 59.355 | −0.22% |
| 44 | .105533 → .106865 | +1.26% | 58.317 → 58.212 | −0.18% |
| 均值 | .104116 → .104093 | −0.02% | 58.618 → 58.952 | +0.57% |

短前缀 KL 仅 seed42改善，其余两 seed恶化，均值基本持平；PPL均值恶化。单次 wall time 约 67–77 秒，配对噪声范围内，不作速度结论。

**评价与反思。** RAdam 是已发表方法的工程移植测试，不是新的 optimizer 创新。其 KL 平均收益为噪声量级且 seed 不稳，PPL均值更差，故在此筛选协议下停止，不推进 full-model、不做 lr sweep。由于候选在每模块前 5 步按 RAdam 设计退回无坐标自适应的一阶动量，原 Adam 的 `grad_lr` 可能不适配 fallback；这说明该适配存在尺度问题，但不能成为继续调参的理由。三 seed只足以淘汰当前直接移植实现，不足以否定 RAdam 在一般随机训练中的文献结论。

### B4. AdaBelief：以一阶矩预测残差更新二阶状态

**理论支持。** AdaBelief 用 \((g_t-m_t)^2\) 代替 Adam 分母中的 \(g_t^2\)，把二阶状态解释为对梯度预测的“belief”：梯度稳定时降低分母，梯度与动量预测不符时提高分母。当前观测说明 REAL-Q refresh 的 \(g^2\) 与历史 \(\hat v\) 可能不同，v-only 清零又导致灾难性恶化；因此值得检验更温和的“按预测残差调节”是否比完全舍弃历史好。局限是 AdaBelief 原本面向随机训练泛化，REAL-Q 的梯度变化来自量化轨迹，且数据/权重的误差不是独立同分布。

**参考文献。** Zhuang et al., *AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients* (NeurIPS 2020), [NeurIPS论文页](https://proceedings.neurips.cc/paper/2020/hash/d9d4f495e875a2e075a1a4a6e1b9770f-Abstract.html)。

**方法/实现。** 保留现有 \(m_t=\beta_1m_{t-1}+(1-\beta_1)g_t\)，将 \(v\) 输入替换为 \((g_t-m_t)^2\)，其余 bias correction、epsilon 放置、学习率、梯度裁剪、样本、列块 GPTQ 与 REAL-Q 控制完全一致。不引入可调温度或额外反向。先做 layer0–4 matched prefix 三种子筛选；若 KL 不呈可重复改善或平均恶化则停止；只有 KL 正向重复后才进一步考察完整模型与 PPL。

**实际效果。** Qwen3-0.6B，W4、act_order，8×64 校准、16×128 评估，layer0–4 前缀：

| seed | Control KL → AdaBelief KL | KL 相对变化 | Control PPL → AdaBelief PPL | PPL 相对变化 |
|---:|---:|---:|---:|---:|
| 42 | .097921 → .095314 | −2.66% | 58.052 → 58.838 | +1.35% |
| 43 | .108895 → .111418 | +2.32% | 59.485 → 60.121 | +1.07% |
| 44 | .105533 → .099897 | −5.34% | 58.317 → 58.912 | +1.02% |
| 均值 | .104116 → .102210 | −1.83% | 58.618 → 59.290 | +1.15% |

三组均成功、treatment marker 生效、源码哈希未变。KL 两好一坏且均值改善约 1.83%，PPL 三组均恶化。根据 KL 为主的预设，进入 full28 seed42 决策实验。

完整 28 层 REAL-Q seed42（其余校准和评估窗口仍为上述小配置）：control KL/PPL=.403959/71.3257，AdaBelief=.452411/74.1695，KL恶化11.99%，PPL恶化3.99%。两组量化成功，候选 marker 与源码检查均通过，量化时间约227.34s/231.67s；全模型质量显著变差，故停止本方案，不再加 full seeds。

**评价与反思。** 此实验针对与项目机制匹配的一个已发表方法，不是新算法。短前缀正向 KL 不能预测全模型收益；本例完整轨迹显著更差，直接否定这项 AdaBelief 移植。预测残差的标度和量化序列中的梯度变化并未匹配常规训练假设；不能通过调整 beta2、epsilon 来救结果。

### B5. ADOPT：用上一步二阶矩归一化当前梯度，再累积一阶动量

**理论支持。** ADOPT 的分析指出，Adam 当前梯度参与当前二阶矩估计会使 (g_t) 与归一化分母相关；仅把分母滞后一拍仍不足以处理与原始梯度累积的一阶动量之间的相关性，因此 ADOPT 改为先按 (v_{t-1}) 归一化当前梯度、再更新一阶动量，并在随后更新 (v_t)。这提供了“更新次序/当前梯度与预条件器耦合”这一可检验假设。与本项目的联系有限：REAL-Q 的梯度来自有限校准样本上的 Fisher surrogate，并随 GPTQ 量化/闭式补偿而变化，不满足论文的一般随机平滑优化设定；此前的矩漂移测量只显示尺度/轮廓有差异，并没有证明梯度—分母相关性造成 KL 损失。

**参考文献。** Taniguchi et al., *ADOPT: Modified Adam Can Converge with Any \(\beta_2\) with the Optimal Rate* (NeurIPS 2024), [arXiv:2411.02853](https://arxiv.org/abs/2411.02853)；[NeurIPS 2024 proceedings PDF](https://papers.neurips.cc/paper_files/paper/2024/file/84d286e32bbee8fa3a86ee9c50e00081-Paper-Conference.pdf)。Reddi et al., *On the Convergence of Adam and Beyond* (ICLR 2019), [OpenReview PDF](https://openreview.net/pdf?id=H1x-x309tm)，是其 Adam 收敛问题背景。以上是外部优化文献；REAL-Q 仍仅作为项目基线和动机。

**方法/实现。** treatment 保留 REAL-Q 的梯度裁剪、反向次数、校准样本、列块顺序和 GPTQ 闭式补偿；从第二次 refresh 起，按旧 (v) 归一化本次裁剪梯度后进入 (m)，随后才用本次 (g^2) 更新 (v)。为了不额外给每个模块做 warm-up backward，第一次已有 refresh 的梯度同时用于 (v_0=g_1^2) 初始化和首个归一化动量，使首步保持 Adam 的 sign-like 更新；这是一项符合算力约束的 REAL-Q 适配，不是论文原算法初始化的逐字复现。两条 block-GD/兼容路径均实现，并由算术单测核对首步与第二步的状态次序。源码 marker、源码哈希和 backward 配置均在配对 run 中检查。

**实际效果。** Qwen3-0.6B，W4、act_order，8×64 校准、16×128 评估，量化 layer0–4；seed42/43/44 使用完全相同的 REAL-Q 默认参数作 paired screening。每次两臂均成功，ADOPT marker 生效、源文件哈希未变；每臂量化耗时约 65–81 秒。结果为：

| seed | Control KL → ADOPT KL | KL 相对变化 | Control PPL → ADOPT PPL | PPL 相对变化 |
|---:|---:|---:|---:|---:|
| 42 | .097921 → .103352 | +5.55% | 58.0521 → 58.0311 | −0.04% |
| 43 | .108895 → .132807 | +21.96% | 59.4849 → 59.7815 | +0.50% |
| 44 | .105533 → .104483 | −1.00% | 58.3172 → 58.2331 | −0.14% |
| 均值 | .104116 → .113547 | +9.06% | 58.6180 → 58.6819 | +0.11% |

首个保存的明显 surrogate-loss 爆发出现在 seed43 的 layer1 `self_attn.o_proj` block14（control 约 .00575，ADOPT 约 3.382），同一层 `mlp.gate_proj` block3 也由约 .00365 升至 2.903；seed42 的大差异主要出现在 layer2 `mlp.down_proj` 多个块，seed44 在该处则改善。不同种子的损失轨迹方向相反，不能从单个模块推断全局因果。

**评价。** ADOPT 的 KL 主指标两 seed 变差、一 seed 略好，平均恶化约 9.1%；PPL 均值近乎持平。该结果不支持推广至完整模型，故停止本次直接移植，不扫描学习率、clip 或 β。此实验是筛选，不是全模型结论，也没有达到可提交的收益证据。

**反思。** “移除同一步 (g_t^2) 对分母的直接影响”并不足以让 REAL-Q 的有限样本梯度稳定。首个 refresh 的 (g/|g|) 初始化只保证不增加反向，并不能解决后续量化与补偿改变坐标梯度分布的问题；ADOPT 在普通训练中使用的裁剪设定和学习率也不能直接假定与 REAL-Q 的 `grad_lr` 配套。seed43 的局部 surrogate 爆发值得作为诊断证据，但不是单独再调一个局部参数的理由。下一步应先对同一快照重算候选/Adam 的归一化梯度、更新范数和下一个 GPTQ 块后的验证 KL；若首个分母切换已造成数量级漂移，就先停止家族级全模型移植。完整三种子日志及 summary 位于 `outputs/realq-adam-history-adopt-seed{42,43}-screen-20261006/` 和 `outputs/realq-adam-history-adopt-seed44-retry-screen-20261006/`；seed44 第一次运行因 D 盘满中断，重跑结果目录明确标为 retry。

### B6. 只重置活跃后缀的一阶矩与其 bias age（修正后的机制对照）

**提出原因与理论支持。** 既有 `reset_m_each_refresh` 实验只将 (m) 清零，却继续使用旧 (m)-age；这样既移除了动量方向，也把新梯度的一阶矩除以旧时间步的 bias correction，混入了更新幅值缩小。该结果不能干净判断量化/补偿改变目标后旧动量是否有害，因此按此前失败教训重新审查，而不是把旧负结果重复当作结论。Asadi et al. 在深度 RL 目标变化时实测完整重置优化器状态；Adam-Rel 则报告只重置时间年龄、保留动量也能改善多个 RL 任务，并指出旧动量未必应全部丢弃。两者结论说明状态污染与有用历史之间存在任务依赖取舍。SPAM 提供周期重置动量的实验证据，但研究的是训练中的梯度尖峰，且同时涉及二阶矩/裁剪，不能直接推导 REAL-Q 应重置。这里的窄假设是：在已发生 GPTQ 量化/补偿的离散 refresh 边界，仅清除尚未量化后缀的一阶方向历史，并用正确的一步 (m)-bias age；保留已表现出尺度敏感性的 (v) 与它的真实 age。文献只支持提出该机制问题，不足以预测量化质量方向。

**参考文献。** Asadi, K., Fakoor, R. & Sabach, S., *Resetting the Optimizer in Deep RL: An Empirical Study*, NeurIPS 2023, [论文页](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e4bf5c3245fd92a4554a16af9803b757-Abstract-Conference.html)；Ellis et al., *Adam on Local Time: Addressing Nonstationarity in RL with Relative Adam Timesteps*, NeurIPS 2024, [论文页](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f2733d3b0dde1d74995f35a9cf442d38-Abstract-Conference.html)；Huang et al., *SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training*, ICLR 2025, [ICLR论文页](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7a70ad3d9c704fb9b81b5c69eda722dc-Abstract-Conference.html)。这些工作均未研究 GPTQ 补偿后的 REAL-Q refresh；其发现相互补充但不等价，只为状态隔离/年龄处理提供问题动机，不构成收益保证。

**方法/实现。** 实验通过 `REALQ_ADAM_HISTORY_POLICY=reset_m_and_m_age_each_refresh` 在 Fisher/Block-GD refresh 中对仍浮点的活跃后缀清零 (m)，并将 (m)-age 设为 0 后正常递增到 1；保留 (v)、(v)-age、当前梯度裁剪、所有反传调用、量化列顺序和 GPTQ 闭式补偿。原始 `state.step` 在该诊断路径中承担 (m)-age；独立 `v_step` 保留 (v)-age。单线性和 full-block Fisher/Block-GD 路径接入了策略；之后也将默认最后一个 transformer block 的 legacy KL refresh 接入同一策略：它重启 (m) 与 (m)-age，但保留并继续累计 (v)-age。新增算术测试核对 KL control 的旧算式等价和 candidate 的独立年龄；全层 candidate 必须在日志中命中 final-layer KL marker。没有增加预热反传。运行器记录运行前后源文件 SHA-256、对 treatment 检查 Fisher 与 KL 路径是否生效，并在记录 checkpoint SHA-256 后清理模型文件。

**方法的精确解释。** 因为每个既有 refresh 都先清零活跃区域的 \(m\) 并将 \(m\)-age 归一到一步，本次 bias-corrected \(\hat m\) 就等于当前经过相同裁剪的梯度 \(g_t\)；候选更新等价于 \(g_t/(\sqrt{\hat v_t}+\epsilon)\)，其中 \(\hat v_t\) 仍保留历史平方梯度 EMA。它不是“让 Adam 动量更准确”，而是一次受控的优化器状态寿命改动：取消一阶方向的跨-refresh EMA，同时保留二阶坐标尺度记忆。这样才是与 REAL-Q 每个模块内持续复用 \(m,v\) 的基线作对照。Cautious 只屏蔽少数 \(g\cdot d<0\) 坐标并再缩放剩余坐标，与这里重启整个活跃一阶矩并不等价；Cautious 负结果不能直接反驳或证明 B6。

**实际效果（layer0–4 三种子筛选）。** Qwen3-0.6B，W4、act_order、8×64校准、16×128评估，REAL-Q默认其余设置。每个种子 candidate/control 都成功；reset marker/source hashes/configuration 检查通过，backward次数一致；为节省磁盘只记录 checkpoint SHA-256 后清除模型文件。

**最后层 KL 路径审计与扩展。** 第一轮 full28 结果只在 layer0–26 应用了策略，最后一层 KL refresh 保持基线，故与下表严格分开。现已将策略接入 `kl_loss.py` 的默认单线性 KL 更新和可选 full-block KL 更新，并保持 `keep` 分支原算式不变。seed42 的第一次扩展运行日志缺少专用 KL marker，虽然数值有效但不作为“已审计全路径”证据；该 seed 用当前代码重跑。下面的新表只纳入通过全层 KL marker、运行成功及源文件前后散列一致检查的 seed42-retry、43、44。

**实际效果（最后层 KL 也启用策略的全28层三种子复验）。** 仍为小规模筛选配置：Qwen3-0.6B W4、8×64 校准、16×128 评估，默认 REAL-Q 其余设置。三种子均覆盖 layer0–27；candidate 的 Fisher/Block-GD 及最后层 KL 更新均出现 marker，运行前后跟踪源码 SHA-256 完全一致，control 与 candidate 都正常退出。每臂约 248–263 秒，反向调用数和配置相同；checkpoint 只保留 SHA-256 后清理。

| seed | Control KL → candidate KL | KL 相对变化 | Control PPL → candidate PPL | PPL 相对变化 |
|---:|---:|---:|---:|---:|
| 42 | .403959 → .371787 | −7.96% | 71.3257 → 70.4988 | −1.16% |
| 43 | .476089 → .426266 | −10.47% | 79.0929 → 75.5474 | −4.48% |
| 44 | .381644 → .361421 | −5.30% | 71.4813 → 71.6811 | +0.28% |
| 算术均值 | .420564 → .386491 | −8.10% | 73.9666 → 72.5758 | −1.88% |

相对变化按每个 seed 单独计算后方向一致：KL 三种子均下降，PPL 两种子下降、一种子上升。旧 seed42 运行（KL .371787/PPL 70.4988）与重跑数值相同，但因缺少终层 marker 只作数值复现，不重复计入本表。

**实际效果（扩大校准/评估规模，六种子筛选）。** 为避免只在极小校准集报告改善，使用相同 Qwen3-0.6B W4 全28层配对，将校准改为 32×128 tokens、评估改为 32×256，refresh seed 仍为0；该配置在 RTX 4060 Laptop 8GB 正常完成，两个 arm 各量化约 765–800 秒，candidate 未增加 backward，末层 KL marker/源码哈希门控通过。summary 中每臂 wall_seconds 均值为 control 777.6 秒、candidate 785.6 秒，配对相对变化均值 +1.03%；该单机小样本计时不支持加速主张。六种子结果：

| seed | Control KL → candidate KL | KL 相对变化 | Control PPL → candidate PPL | PPL 相对变化 |
|---:|---:|---:|---:|---:|
| 42 | .239430 → .225144 | −5.97% | 49.8703 → 49.2497 | −1.24% |
| 43 | .241984 → .242016 | +0.01% | 50.1947 → 49.7427 | −0.90% |
| 44 | .246840 → .236678 | −4.12% | 50.4452 → 50.1711 | −0.54% |
| 45 | .240513 → .229288 | −4.67% | 49.6026 → 49.8724 | +0.54% |
| 46 | .248819 → .236230 | −5.06% | 50.9039 → 50.5324 | −0.73% |
| 47 | .232586 → .228072 | −1.94% | 49.5897 → 49.7374 | +0.30% |
| 算术均值 | .241695 → .232904 | −3.62% | 50.1011 → 49.8843 | −0.43% |

五/六种子 KL 改善，seed43 近乎持平，seed47 有较小改善；PPL 四/六种子改善、seed45 与 seed47 恶化。所有运行覆盖 layer0–27、末层 KL policy marker、candidate policy marker 与 source hashes unchanged；seed42–47 均 returncode 0。每臂量化约 765–800 秒。artifact 分别为 `outputs/realq-adam-history-allpath-m-restart-full28-n32-s128-e32x256-seed{42,43,44,45,46,47}-20261006/summary.json`。相对变化按每 seed 配对计算；均值是描述性 screening，不是显著性检验。

**中等配置全层复验（64×256校准、64×512评估，seed42–44）。** 针对 B6 六种子筛选可能受校准/评估样本规模影响的问题，预先固定三个新 seed，在相同 Qwen3-0.6B、W4、完整 layer0–27、refresh seed 0 上配对运行原始 Adam 与 B6。模型配置、量化路径、反向预算和评估窗口对齐；三组均 returncode 0，候选策略在 layer0–27 的 Fisher/KL 路径 marker 全部出现，source hashes unchanged，候选 backward 次数与 control 相同。checkpoint 记录 SHA-256 后删除。完整命令封装在 `scripts/run_b6_medium_replication_20261007.sh`；每组 summary 与日志位于 `outputs/realq-adam-history-b6-medium-all28-64x256-e64x512-seed{42,43,44}-20261007/`。

| seed | Control KL → B6 KL | KL 相对变化 | Control PPL → B6 PPL | PPL 相对变化 |
|---:|---:|---:|---:|---:|
| 42 | .1424823 → .1402158 | −1.59% | 31.45310 → 31.42367 | −0.09% |
| 43 | .1386254 → .1393988 | +0.56% | 31.25348 → 31.26209 | +0.03% |
| 44 | .1389646 → .1381698 | −0.57% | 31.38741 → 31.51410 | +0.40% |
| 算术均值 | .1400241 → .1392614 | −0.535% | 31.36466 → 31.39995 | +0.113% |

相对变化先逐 seed 配对再取算术均值；KL 2/3 seed 改善，PPL 仅1/3改善，且 KL 最好的一组 seed42 的 PPL 收益只有 −0.09%。量化时间不是本轮主要假设，不据此报告速度结论。

**评价与后续决策。** 这次更大校准/评估设置没有复现稳定的双指标改善：KL 均值收益小，PPL 均值略差，且不同 seed 方向混合。它不推翻旧配置中观察到的信号，但明确说明该信号对配置或种子敏感；目前不能称 B6 已 work、不能作为论文贡献，也不能把跨层 v 累积建立在它之上。依照预先约定，暂停“跨层同模块累积 v、每层重置 m”的实验。现有 cross-module screen 也显示 layerwise profile 相似度不稳定、旧的跨层初始化 v 全28层未获收益；文献只能提供非零二阶矩初始化/局部时间等邻近动机，未直接证明 GPTQ 层间 v 可迁移。固定 refresh 快照的方向/幅度拆分已在 B6 小节完成，但没有得到跨点正 KL 信号；后续不再扫 B6 范围/强度。若继续研究 Adam，先审查下文 G²PTQ 信赖域机制能否在已有 REAL-Q Adam 梯度/Fisher 下构造一个不增加 backward 的有界更新，再决定是否做一个小规模机制筛查。

**机制复核：逐 refresh surrogate 对照。** 第一轮离线解析按 layer/module/column-block/objective 对齐了每 seed 的 2,044 个 refresh 事件，其中同时包含 Fisher-MSE 与末层真实 KL；因此旧汇总文件 `outputs/realq-adam-history-allpath-m-restart-full28-n32-s128-e32x256-seed44-20261006/refresh_loss_pair_diagnostic.json` 中 seed42 +0.29%、43 +0.51%、44 +0.33% 的“主 loss”混合了不同 objective，不能称纯 Fisher surrogate。现按 `objective=fisher_mse` 重新筛选，并以每个对齐事件的算术均值比较（每种子 1,971 个事件）：

| seed | Control mean Fisher-MSE | Candidate mean | 相对变化 | Candidate loss 较低事件 |
|---:|---:|---:|---:|---:|
| 42 | 4.941165 | 4.935236 | −0.120% | 55.3% |
| 43 | 4.257350 | 4.284148 | +0.629% | 8.2% |
| 44 | 3.888520 | 3.901815 | +0.342% | 8.2% |
| 45 | 4.472909 | 4.493573 | +0.462% | 7.8% |
| 46 | 3.943386 | 3.927452 | −0.404% | 61.7% |
| 47 | 3.965403 | 3.960119 | −0.133% | 53.3% |

这是各 refresh 块等权的描述统计，不是校准集总能量；候选只有三种子平均 loss 下降，且 seed44/45 的最终 KL 改善并未对应 mean Fisher loss 下降，seed43 的 KL 持平也与 Fisher loss 明显上升一致。故它不足以解释整体结果，更不能据此选择下一算法。旧 all-objective paired percentages 已不作 Fisher-only 证据保留；所有 loss 均不替代末端 KL/PPL 质量评估。

**机制复核：按损失类型与深度定位（全路径六种子）。** 重新按每个种子内的对齐事件先求 loss 均值，再比较 candidate/control，避免把不同尺度事件的相对百分比直接平均。layer0–26 的 Fisher-MSE 按深度分组后，六种子平均相对变化依次为 layer0–6 +0.10%、layer7–13 +0.13%、layer14–20 +0.36%、layer21–26 +3.88%；也就是说 Fisher surrogate 没有随整体最终 KL 改善而一致下降，且靠近末端时平均更高。layer27 的 73 个真实 KL refresh 事件则在每个种子中都低于对照：

| seed | Control mean KL-refresh loss | Candidate mean | 相对变化 |
|---:|---:|---:|---:|
| 42 | 0.0226202 | 0.0191677 | −15.26% |
| 43 | 0.0231934 | 0.0201223 | −13.24% |
| 44 | 0.0211901 | 0.0181445 | −14.37% |
| 45 | 0.0233169 | 0.0204413 | −12.33% |
| 46 | 0.0220613 | 0.0190016 | −13.87% |
| 47 | 0.0242573 | 0.0204887 | −15.54% |
| 合并事件均值 | 0.0227732 | 0.0195610 | −14.11% |

这是一个定位线索，不是因果证明：candidate 在进入 layer27 前已重启过 layer0–26 的状态，故两臂 layer27 的输入和量化轨迹已不同；较低的 KL-refresh loss 既可能来自 layer27 的更新策略，也可能来自上游状态变化使该层更易拟合。现有 run.log 记录逐 refresh loss 与策略 marker，没有保存可配对复用的原始 (g,m,v,\Delta W) 张量，无法只靠离线分析确定更新方向/尺度机制。下一项完整模型验证应保持同一配置，采用 early/final scope 的 2×2 配对设计：不重置、仅 layer0–26 重置、仅 layer27 重置、全层重置；比较末层 KL loss 与最终 held-out KL/PPL。这样才能判断正向信号来自前段、末层，还是两者交互。当时该 scope 实验因本机磁盘/WSL I/O 问题暂缓；环境恢复后已启动，结果见下文。不扫学习率或刷新频率。

**范围隔离试验（seed42，单种子机制筛选）。** 为区分末层更新与上游状态变化，在候选重置策略上新增显式 layer scope（`all`、闭区间如 `0-26`、或索引列表）；control 与 candidate 均全28层量化，使用同一32×128校准、32×256评测、refresh seed 0。测试了仅 layer27 与仅 layer0–26 两个 treatment，并让 runner 核对实际 marker 层集合与预期完全相同。两个 paired control 均复现原 seed42 的 KL/PPL 0.2394298/49.87028，scope marker 检查通过，未增加 backward。结果：

| Policy scope | KL | KL 相对 control | PPL | PPL 相对 control |
|---|---:|---:|---:|---:|
| control（不重置） | 0.2394298 | — | 49.87028 | — |
| 仅 layer27 重置 | 0.2389911 | −0.18% | 49.84679 | −0.05% |
| 仅 layer0–26 重置 | 0.2253922 | −5.87% | 49.34835 | −1.05% |
| 全层重置（此前 B6 seed42） | 0.2251440 | −5.97% | 49.24970 | −1.24% |

逐 refresh 核对进一步区分了两个范围：末层单独重置时，layer0–26 的 1,971 个 Fisher-MSE 事件全部与 control 精确相同；layer27 的 73 个 KL 事件平均 loss 反而 +0.261%，只有14/73低于 control。仅重置 layer0–26 时，layer27 自身使用原始 Adam，但其73个 KL refresh loss 平均 −15.88%，73/73低于 control。由此，seed42 中全层 B6 的大部分最终 KL 改善可以在不重置末层 m 的情况下复现；末层 KL loss 的下降更像是上游量化/更新改变了该层的输入状态，不是末层重置本身的直接效果。seed42 单种子尚未识别 layer0–26 内哪个阶段造成收益；不能据此宣称上游重置机制已被解释或普遍有效。随后按相同配置在 seed43 做了 early-only 配对复验，结果见下表；seed44 仍待复验。

| seed / 处理范围 | Control KL → candidate KL | KL 相对变化 | Control PPL → candidate PPL | PPL 相对变化 | layer27 的 73 个 KL refresh loss 均值变化 |
|---|---:|---:|---:|---:|---:|
| 42 / 仅 layer0–26 重置 | 0.2394298 → 0.2253922 | −5.87% | 49.87028 → 49.34835 | −1.05% | −15.88%，73/73 更低 |
| 43 / 仅 layer0–26 重置 | 0.2419838 → 0.2426991 | +0.30% | 50.19474 → 49.92166 | −0.54% | −12.55%，73/73 更低 |
| 44 / 仅 layer0–26 重置 | 0.2468402 → 0.2376181 | −3.74% | 50.44523 → 50.00538 | −0.87% | −14.07%，73/73 更低 |
| 45 / 仅 layer0–26 重置 | 0.2405133 → 0.2287878 | −4.88% | 49.60265 → 49.82853 | +0.46% | −13.05%，73/73 更低 |

seed43–45 两臂均正常完成，return code 为0，源码哈希运行前后不变；candidate 实际 reset marker 恰为 layer0–26，layer27 没有 reset marker、KL 路径保留。每个 seed 两臂均记录 2,044 个逐 refresh 事件且顺序对齐。seed43、44、45 的 layer0/1/2 Fisher-MSE 层均值均下降，layer3–26 层均值均上升；其变化随深度放大，反映重置让整条量化/补偿轨迹偏离 control，而不能解释成各层 surrogate 同时变好。layer27 的局部 KL refresh loss 在 seed42–45 都逐事件下降（各73/73），但 seed43 的全模型 KL 却微升0.30%。因此局部末层 loss 下降不是最终 KL 改善的充分条件。四个种子中最终 KL 为3胜1负，平均相对变化 −3.54%；PPL 为3胜1负，平均 −0.50%。这个 reduced-config 筛选的平均 KL 变化接近此前 all-layer 六种子均值（−3.62%），说明不重置末层也可能保留大部分信号；但 early-only 仍有 KL/PPL 反例，不能把它当作已验证方法。

**更窄范围的否证筛选（仅 layer0–2，seed42）。** 为检验早期三层是否足以产生前述 layer0–26 的全局变化，复用相同 W4、32×128 校准、32×256 评测、seed42、refresh seed0 和完整28层量化，只在 layer0–2 每次 refresh 重置活跃后缀的 (m) 与 (m)-age，layer3–27 保持原 Adam。两臂 return code 均为0、被跟踪源码 SHA-256 前后相同；候选实际 policy marker 恰为 `[0,1,2]`，末层 KL policy 路径仍启用且不重置。两臂各有2,044个对齐事件，所有层各73次。

| 范围 | Control KL → candidate KL | KL 相对变化 | Control PPL → candidate PPL | PPL 相对变化 |
|---|---:|---:|---:|---:|
| 仅 layer0–2 重置，seed42 | 0.2394298 → 0.2410455 | +0.675% | 49.87028 → 49.82552 | −0.090% |

逐事件 Fisher-MSE 均值在 layer0 变化 −0.21%，layer1 +7.62%，layer2 +0.10%；这些层不足以复现 layer0–26 的 KL 改善。候选进入后续层后，其 Fisher-MSE 相对 control 在 layer3–20 多数略低，layer21–25 略高，layer26 低0.39%；末层73个真实 KL refresh loss 均值反而高0.83%。这再度说明局部 surrogate 改变不能直接推出最终 KL 改善，也不能把 layer0–2 的局部变化视为上游收益来源。由于 layer0–26 方案包含 layer3–26 而 layer0–2 单独处理无 KL 收益，当前 seed42 的可检验解释更偏向中后段（layer3–26）状态重启或其与早期层的交互；尚不能排除两者交互。

**互补范围筛选（仅 layer3–26，seed42）。** 采用相同的完整28层 W4 配置，只在 layer3–26 每次 refresh 重置活跃后缀的 (m) 与 (m)-age，layer0–2 和 layer27 保持 baseline。control 完全复现既有 seed42 的 KL/PPL 0.2394298/49.87028；两臂均正常退出、跟踪源码 SHA-256 前后相同、每臂2,044个事件。候选 policy marker 实际层恰为 `[3,…,26]`，末层 KL 路径 active。候选 KL=0.2306694（−3.66%），PPL=49.56869（−0.60%）。相比仅重置 layer0–26 的 seed42 结果（KL −5.87%、PPL −1.05%），中段范围单独保留了相当一部分改善，但此差异不能量化成独立贡献，因为两个处理来自不同候选轨迹且没有做交互设计。

逐事件对齐显示 layer0–2 的全部219个 Fisher-MSE 事件与 control 数值精确相同，证明进入 treatment 前缀一致。自 layer3 起轨迹开始改变；候选相对 control 的各层 Fisher-MSE 均值差从 layer3 的 −0.001% 逐渐扩大，到 layer26 为 +10.69%，而 layer27 的73个真实 KL refresh loss 均值低16.47%。因此同 seed42 的最终 KL 改善并非因为 Fisher surrogate 普遍变小；轨迹差异经过后续层传播后，末层更容易降低局部 KL，但局部量仍不是全模型结果的替代指标。和之前“仅重置 layer0–2”KL +0.68%、PPL −0.09% 的结果合看，当前证据支持“layer3–26 范围足以在 seed42 改变并改善终端结果”的机制假设；还没有证明一般性，也没有排除 seed 与层范围交互。

**范围复验（layer3–26，seed43–44）。** seed43 的 control/candidate KL/PPL=0.2419838/50.19474 → 0.2408733/50.55592，KL −0.46%、PPL +0.72%。seed44 为 0.2468402/50.44523 → 0.2386119/49.95589，KL −3.33%、PPL −0.97%。三种子配对相对变化如下：

| seed | KL 相对变化 | PPL 相对变化 |
|---:|---:|---:|
| 42 | −3.66% | −0.60% |
| 43 | −0.46% | +0.72% |
| 44 | −3.33% | −0.97% |
| 均值 | −2.48% | −0.29% |

三组均运行成功、源哈希不变；candidate marker 都恰为 layer3–26，末层 KL policy active；每臂2,044个事件。各 seed 的 layer0–2 前缀219个 Fisher-MSE 事件与 control 逐值完全相同。seed43/44 的 layer26 Fisher-MSE 均值分别 +10.97%/+8.14%，layer27 KL refresh loss 均值分别 −14.09%/−13.40%。因此最终 KL 三种子都改善，PPL 两胜一负；这给出值得继续检查的 KL 正向信号，但仍是32×128校准、32×256评估的筛选结果，不能视作稳健有效结论或论文贡献。末层局部 KL loss 一致下降仍不等同于全模型 KL 的变化幅度。

**中等配置扩展验证（layer3–26，seed42）。** 校准扩大到64×256、评测扩大到64×512，完整28层 W4 配对；candidate 只在layer3–26重置，其他参数不变。两臂正常退出，source hashes unchanged，candidate marker 精确为 layer3–26，末层 KL policy active；每臂2,044个事件，layer0–2的219个事件逐值与control相同。control KL/PPL=0.1424823/31.45310，candidate=0.1432755/31.72144，KL +0.56%、PPL +0.85%，两项均变差。对应日志和状态位于 `outputs/realq-adam-history-scope-mid0-26-medium64-seed42-20261006/`。

这项扩大样本量的完整模型复验没有保留小配置上的改善，满足了预先设定的暂停条件。因此不能把 layer3–26 的 (m)+age reset 作为有效方案继续扩展，也不应将32×128结果描述为可泛化收益。当前最重要的经验是：相同seed下，改变校准与评测规模的组合后，终端KL方向发生反转；因为校准和评测规模同时改变，本实验无法区分是哪一项导致变化。末层KL refresh loss改善也不能保证更大评测集上的终端KL改善。下一步先停止对该范围/边界/强度扫参，回到已记录的 Adam 状态轨迹与文献适配审查，寻找能由真实机制诊断支持的新假设，再决定是否启动另一种方法。所有结论以真实终端 KL 为主、PPL 为辅助；逐层 surrogate 和末层 KL refresh 仅作机制诊断。

| seed | Control KL → corrected m-restart KL | KL 相对变化 | Control PPL → corrected m-restart PPL | PPL 相对变化 |
|---:|---:|---:|---:|---:|
| 42 | .097921 → .097083 | −0.86% | 58.0521 → 59.0751 | +1.76% |
| 43 | .108895 → .105035 | −3.55% | 59.4849 → 59.4003 | −0.14% |
| 44 | .105533 → .101900 | −3.44% | 58.3172 → 58.0189 | −0.51% |
| 均值 | .104116 → .101339 | −2.67% | 58.6180 → 58.8314 | +0.36% |

三种子前缀 KL 同向改善，PPL 两组改善、一组恶化，平均略差。首轮 seed42 的 baseline 跑完后，新增 checkpoint 清理代码把单文件误当目录而退出；该不完整 campaign 被保留为 harness failure，checkpoint SHA-256 单独记在 `checkpoint_cleanup_audit.json`，修复后以 retry tag 重跑，正式结果只取通过脚本 gate 的配对结果。

**评价（历史覆盖范围）。** 三种子 layer0–4 筛选中 KL 均改善约 0.86%–3.55%；PPL 两好一坏、均值 +0.36%。此前 full28 历史筛选只在前27层应用重置、最后 KL 层保持基线：KL 变化 −8.12%、−9.75%、−5.29%，PPL 变化 −0.95%、−4.30%、+0.26%。这些历史数据不能代表最后层也应用策略；严格全路径结果见上一表，不与它合并。两类结果均为小配置筛选（8×64校准、16×128评估），未构成论文级确认。它们与先前错误 bias-age 的 m-only 实验不同；旧的正负结果都应保留为研究记录。

**评价、反思与下一步。** 大配置六种子 KL 五组改善、一组近乎持平，平均配对下降 3.62%；PPL 四组改善、两组恶化，平均下降 0.43%。按相对配对变化计算的 exploratory paired-t 95% interval：KL 为 [−5.97%, −1.28%]，PPL 为 [−1.17%, +0.31%]；只有六个种子、一个模型和一个 reduced config，这不是论文级显著性结论。seed47 的 KL/PPL 指标分歧说明目前不能只按单一指标宣称普遍收益；不过 KL 是论文主指标，六种子中五组向好、没有单一种子独占均值，因此保留为值得跨模型复验的候选。局部 surrogate 并未一致改善，警示不能以训练期 loss 代替最终 KL。Asadi 的整体状态重置、Adam-Rel 的保留矩重置 age、AdaR 的双状态周期交替彼此设计不同；不能把它们混成“momentum 应该重置”的共识。下一步不扫学习率或 β，也不重跑已筛过的 m/v 矩消融：先完成全路径六种子逐 refresh 损失类型/深度定位（已完成，见上文），随后在环境恢复后做 early/final scope 2×2 配对实验；若 scope 结果仍不能定位收益来源，再对固定 snapshot 增加 g/m/v/ΔW 记录并比较下一真实 GPTQ 块后的 KL。当前跨模型筛选已尝试启动，因磁盘 I/O 中断而没有可用质量数据；磁盘空间恢复后再重跑，不改变配置。现有 Cautious 样本中全局 Adam-update/gradient cosine 为正（约 .46–.76），坐标冲突虽有数个百分点但所占更新质量低于约 0.2%；因此“遇到冲突就清 m”目前缺少广泛触发证据，不立即实现。所有严格全路径 seed42–47 都通过对照和 candidate 完整执行、源哈希稳定、末层 KL policy 正确的审计。旧的“前27层策略、最后层基线”数据仍保留作历史筛选，不与严格全路径结果合并。当前严格全路径 artifacts：`outputs/realq-adam-history-allpath-m-restart-full28-seed42-retry-20261006/summary.json`、`...seed43-20261006/summary.json`、`...seed44-20261006/summary.json`、`...seed45-20261006/summary.json`、`...seed46-20261006/summary.json`、`...seed47-20261006/summary.json`；大配置 seed42–47 的完整路径见前述 artifact 命名。早期 seed42 全路径运行因缺专用 marker 被排除为有效验证集，但单独保留其日志与结果。

**跨模型尝试（Qwen3-1.7B；无效实验，未用于方法评价）。** 使用相同 runner 做 full28 配对初筛，W4、8×64 校准、16×128 评估、seed42、其余 REAL-Q 默认设置。第一次 control 完成全28层量化，但 D 盘剩余空间只有约 11 MB，保存约 2.9 GB checkpoint 时写入失败（`PytorchStreamWriter ... file write failed`），没有质量指标，candidate 未运行。第二次将 output root 移到 WSL ext4 后，control checkpoint 虽写出，但 WSL 随后出现 `Errno 5: Input/output error`，runner 读取日志失败，candidate 仍未运行。两次 run 都不能提供 Qwen3-1.7B KL/PPL 比较；第二次的量化过程也不得单独作有效 control。诊断时 D 盘可用空间约 11 MB、C 盘一度为 0，WSL distro VHDX 位于 `D:\ubuntu\ext4.vhdx`。试图清理项目下可再生成的 reference-logit 缓存以恢复空间时，被本机自动审查策略拒绝；未尝试绕过，也未删除任何缓存或旧检查点。后续本地 GPU 实验须待磁盘空间由用户释放或允许安全归档后再启动。失败日志的预期位置为 `outputs/realq-adam-history-allpath-m-restart-qwen17b-n8-s64-e16x128-seed42-20261006/` 和 `/home/uky/realq-adam-experiments/realq-adam-history-allpath-m-restart-qwen17b-n8-s64-e16x128-seed42-ext4-20261006/`；后者当前可能受 WSL I/O 故障影响。

### C. Grams 式方向替换（保留 Adam 更新的坐标幅值）

**理论支持。** Grams 将方向与幅值分解：当前梯度提供方向，动量负责自适应幅度。它适合作为判断 REAL-Q 中历史动量方向是否带偏的机制对照；并不保证离散量化后的 KL 更低。

**参考文献。** Cao, Li & Song, *Grams: Gradient Descent with Adaptive Momentum Scaling* (2024 preprint), [arXiv:2412.17107](https://arxiv.org/abs/2412.17107)。

**方法/实现。** 在保存的同一真实 REAL-Q refresh 快照上，计算原 Adam 实际减法更新 (u)，替换方向为 `sign(g) * abs(u)`，保留每坐标原更新幅值、学习率、矩状态、age、活动后缀和 loss；与原 Adam 及 no-parameter-update 锚点比较。无需新增 backward。

**实际效果。** 4 个同快照小前缀点中，Grams 式 KL：seed42 layer0 o_proj .097825（Adam .098157，KL稍好）但 PPL 58.579（Adam 58.044，变差）；seed42 layer3 down_proj .099848/58.284（KL、PPL均差）；seed43 layer0 o_proj .104489/59.207（均差）；seed43 layer3 down_proj .100598/59.061（KL差、PPL稍好）。因此 KL 仅 1/4 较好，PPL 也仅部分变化且方向不一致。此为短前缀、单次局部更新后的序列指标，不是完整模型质量。

**评价。** 没有可重复的 KL 收益。方向拆分的想法仍是有用诊断，但这项直接替换不应推进为全模型方法。

**反思。** “Adam 与当前梯度方向不一致”不等于“当前梯度方向更好”。真实下游 KL 的梯度/目标与 refresh 的 surrogate 不同，后续 GPTQ 闭式补偿也可能反转局部收益。未来的方向类方法必须在真实下一块 GPTQ 之后评分。

### C2. Adam 全局步长匹配的原始梯度方向（方向机制对照）

**理论支持。** 在光滑目标的一阶近似中，沿当前梯度方向的足够小步长有 \(\Delta L\approx-\eta\|g\|^2\)；自适应坐标预条件会改变方向与各坐标步幅。REAL-Q 对 surrogate 更新的论证采用 SGD 型方向，因此值得直接检验：在固定 Adam 更新的整体 L2 范数时，纯当前梯度方向能否比 Adam 的坐标预条件方向更好。这个一阶结论只作用于当下 surrogate，不能保证经过权重舍入、闭式补偿和下游层后 KL 变好。标准非凸 SGD 文献给出渐近优化理论，不是对 GPTQ/REAL-Q 离散轨迹的保证。

**参考文献。** Ghadimi & Lan, *Stochastic First- and Zeroth-Order Methods for Nonconvex Stochastic Programming* (SIAM J. Optim. 2013), [arXiv:1309.5549](https://arxiv.org/abs/1309.5549)。Adam 与随机梯度方向问题的相关讨论见 Reddi et al., *On the Convergence of Adam and Beyond* (ICLR 2019), [OpenReview PDF](https://openreview.net/pdf?id=H1x-x309tm)。REAL-Q 内部关于 SGD surrogate 更新的推导只作为本项目假设，不列为外部参考文献。

**方法/实现。** 在目标 refresh 使用 \(d_{cand}=g_{clip}\|d_{Adam}\|_2/\|g_{clip}\|_2\)，保持当前裁剪梯度方向和 Adam 的总更新 L2 范数，其他状态更新、列块量化/闭式补偿和其余 Adam refresh 不变。比较 layer0 attention early block1、layer3 FFN down_proj early block1，seed42/43各一组。介入前逐项要求权重、梯度、\(m,v\)、Adam step 哈希完全相同；两臂均走真实 `realq.ptq` 并继续量化 layer0–4。检查点仅计算并记录 SHA-256 后删除以控制 D 盘占用，指标、日志、状态和散列仍保留。

**实际效果。** 四个单点 refresh 分支都成功，treatment marker 生效、介入前快照相同、源码哈希不变：

| seed | refresh | Control KL → 梯度方向 KL | KL 相对变化 | Control PPL → 梯度方向 PPL | PPL 相对变化 | Adam-update / gradient cosine |
|---:|---|---:|---:|---:|---:|---:|
| 42 | layer0 `self_attn.o_proj`, block1 | .097921 → .109767 | +12.10% | 58.0521 → 58.6934 | +1.11% | .7489 |
| 43 | layer0 `self_attn.o_proj`, block1 | .108895 → .115532 | +6.09% | 59.4849 → 59.9873 | +0.85% | .7657 |
| 42 | layer3 `mlp.down_proj`, block1 | .097921 → .098503 | +0.59% | 58.0521 → 58.7896 | +1.27% | .4893 |
| 43 | layer3 `mlp.down_proj`, block1 | .108895 → .109938 | +0.96% | 59.4849 → 60.0296 | +0.92% | .4577 |

此处 PPL 是 16 个长度128评估窗口的前缀指标；同 seed 的 control 因只对一个目标 refresh 分叉而相同。四个 candidate KL 均更差，FFN 的差异较小但两种子一致，attention 差异较大且两种子一致。每臂约 80–86 秒；该单点诊断不代表端到端速度或显存收益。

**评价。** 在已测的两个模块位置，替换为梯度方向都没有改善 KL；对 attention 明显有害，对 down_proj 轻微有害。结果说明这些点上的 Adam 坐标缩放/动量方向对于量化后的质量可能有用，不能把方向与梯度不一致直接当成缺陷。停止当前纯梯度方向替换，不做全模型和参数 sweep。

**反思。** Cautious 冲突坐标所占更新质量很小，Adam 的方向内积虽为正但 cosine 中等；这说明“存在方向差异”不等于预条件器有害。保持整体步长不变后 KL 仍变差，支持差异可能来自 coordinate-wise scaling，而非单纯步长。仍不能认定预条件器是因果原因，因为换方向也改变了各坐标更新范数；下一步先比较相同快照上 Adam、当前梯度和跨 refresh 动量方向各自的校准 surrogate 变化、更新范数分布以及真实下一个 GPTQ 块后的 KL，只有观察到可解释且重复的局部优势才扩大评测。

### D. Cautious Adam：屏蔽与当前梯度冲突的坐标

**理论支持。** Cautious 优化器按当前梯度与优化器更新一致性屏蔽冲突坐标，并对保留坐标做归一化；论文提供的是其优化分析和训练任务经验，不直接保证 held-out KL 或 GPTQ 离散舍入后质量。

**参考文献。** Liang et al., *Cautious Optimizers: Improving Training with One Line of Code* (ICLR 2026), [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ae03bdef276132fae089692445725635-Abstract-Conference.html)；早期预印本 [arXiv:2411.16085](https://arxiv.org/abs/2411.16085)。

**方法/实现。** 同状态快照比较 Adam、mask-only、scale-only、完整 Cautious。完整式 (M=1[g\cdot d>0])，以 (N/(\sum M+1)) 归一化；保留 Adam 的 (m,v) 和 age。独立捕获 update 后和真实下一块 GPTQ 后的 KL。实验记录明确区分冲突坐标数与实际 update mass。

**实际效果。** 完整配对 6 个 checkpoint（2 seeds、早/晚 refresh、attention/FFN）：完整 Cautious 的 Adam 写入后 KL 有时好、有时坏；下一块 GPTQ 后只 3/6 改善。layer3/down_proj block16 两个种子均在 Adam 写入后略好、下一 GPTQ 后变差。冲突坐标为 2.32–5.26%，但对应更新平方 L2 仅 0.42–1.16%；mask 后的缩放系数影响所有保留坐标。所有 FP 更新后更低的 surrogate loss 都没有稳定转成验证 KL 收益。具体逐项结果见 [`cautious_transition_diagnostic_20261005.md`](cautious_transition_diagnostic_20261005.md)。

**评价。** 本项目被屏蔽的坐标占比和更新质量占比不是一回事；后者很小，故冲突掩码不是当前已证实的主要瓶颈。结果不支持启用 Cautious 默认路径。

**反思。** 局部连续 loss 的下降不能替代量化后 KL；还要区分掩码效果、全局缩放和后续离散轨迹。没有必要把该小筛选扩成全模型 sweep。若后续有证据，只对状态级明确的单点再测。

### E. Adam-aware 影子量化预览

**理论支持。** 当前 Adam 更新可能影响下一列块的量化答案；若预览真实下一块再更新当前后缀，理论上可让优化器面对更接近后续离散状态的目标。这是本项目机制假设，不是现有论文已证明的 Adam 算法。

**参考文献。** Lee et al., *Scheduling Weight Transitions for Quantization-Aware Training* (ICCV 2025), [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2025/html/Lee_Scheduling_Weight_Transitions_for_Quantization-Aware_Training_ICCV_2025_paper.html) 提醒 latent 连续权重变化不能等同量化权重转移；它研究 QAT，不是 GPTQ/REAL-Q 的影子预览。

**方法/实现。** Adam-aware shadow candidate 临时量化下一列块，依据该预览预测的状态调整当前 Adam refresh；对照执行原 REAL-Q。额外预览使路径更慢，显存峰值没有下降。

**实际效果。** 全模型配对 KL 0.225465→0.228842（恶化约 1.50%），PPL 32.7383→32.9167（恶化约 0.54%），量化时间约增加 16%，peak memory 持平。结果在 [`shadow_quant_mechanism_audit_20260929.md`](shadow_quant_mechanism_audit_20260929.md)。

**评价。** 候选质量和工程成本均不支持保留；停止该具体预览方法。

**反思。** 预测下一个量化结果并不会自动让当前梯度更有用；状态预测误差与当前优化目标的偏差可能增加。不要继续做同一路线的影子块调参。

### F. 跨层 Adam (v) 二阶矩迁移

**理论支持。** 相关文献提供的是相邻机制：Adam 的 (v) 是坐标级梯度平方 EMA；非零二阶矩初始化工作说明先验可影响短更新轨迹；Adam-Rel 说明非平稳变化下 moment age/scale 需要小心处理。REAL-Q Future Work 提出 optimizer-state/module transfer 是本项目的动机，但不能算独立相关工作，也不能替代实际效果证据。跨模块状态能否迁移必须通过相同角色、相同活跃权重区域的实测 profile 和后量化 KL 验证。

**参考文献。** Adam (Kingma & Ba 2014)；Abuduweili & Liu (CPAL 2025)；Ellis et al. (NeurIPS 2024)，链接见上。它们均未直接研究 REAL-Q 中 GPTQ 后缀之间的跨层二阶矩迁移。

**方法/实现。** `realq/refresh/adam_transfer_diagnostic.py` 记录 donor 行/列分块 log-profile，与目标已有梯度 (g^2) 的相似度，不额外反传。候选将同角色前一层 donor row profile 做尺度归一，并以已有目标梯度确定尺度，只初始化目标 (v)，保留 (m=0)。online Pearson 门槛 .4；当前代码在 log-space 混合 donor row prior 和目标坐标级 (g^2)，独立维护 (m) 与 (v) bias age（(m) 首步、(v) 虚拟第二步）。每次实验都通过真实 `python -m realq.ptq`，目标 layer8 `mlp.down_proj`。

**此前失败/纠错。** 早期历史 transfer 结果与当前 REAL-Q 配置/入口的 online corr 不一致（旧日志 layer8 seed44 .529，当前 fresh 同 named case 得 .173），旧 campaign 的 runner/source provenance 不足，不能混为同一实验结论。第一次 forced mechanism probe 还错误地让未初始化的 (m) 也提前 bias-correct；已修正为独立 age。layer5 seed44 的 log-space donor weight .5 matched prefix 使 KL .110368→.111058（+0.63%），PPL 58.523→58.813（+0.50%），两项皆差；update cosine .915、相对 update L2 变化 .445，说明 row profile correlation 不能约束逐坐标 preconditioner 足够好。故停止在 layer5 用 .5 混合。

**最新实际效果（有效口径）。** 以下均为纠正独立 (m/v) age 后，由当前 REAL-Q 入口新跑的 matched pairs。8-calibration-sample、16-evaluation-window、layer0–8 短前缀：

| seed | corr / donor:target scale | Adam KL → prior KL | Adam PPL → prior PPL | KL/PPL 方向 |
|---:|---:|---:|---:|---|
| 42 | .5768 / 2.340 | .127095 → .125784 | 60.081 → 59.962 | 均改善 |
| 43 | .5922 / 3.435 | .145597 → .144088 | 62.093 → 61.812 | 均改善 |
| 44 | .5294 / 2.543 | .129471 → .128759 | 59.938 → 59.835 | 均改善 |

三组短前缀各只有一个配对。完整 28 层 matched 结果如下：

| seed | Control KL → transfer KL | KL 相对变化 | Control PPL → transfer PPL | PPL 相对变化 |
|---:|---:|---:|---:|---:|
| 42 | .403959 → .415293 | +2.81% | 71.326 → 74.587 | +4.57% |
| 43 | .476089 → .468469 | −1.60% | 79.093 → 79.846 | +0.95% |
| 44 | .381644 → .379149 | −0.65% | 71.481 → 72.371 | +1.25% |
| 均值 | .420564 → .420970 | +0.10% | 73.967 → 75.601 | +2.21% |

seed42 的实际首更新相对普通 Adam cosine=.8516、relative update L2 change=.5264；候选/base 修正 (\hat v) 坐标比 p05/median/p95=.546/2.599/357.535。seed43 对应 cosine=.8505、relative L2=.5280、ratio=.545/2.623/376.910。row prior 大幅改变了坐标级预条件器，并非小扰动。短前缀三种子两项都改善，但 full28 不复现：KL 仅 seed43、44 改善且均值略差，PPL 三种子都变差。三组 layer8 transfer 都实际应用一次、相关系数过 .4、source hashes unchanged。结论是停止当前“跨层 row-profile 初始化 v”的方案；不得用短前缀结果宣称有效。时间差不作速度结论。

**评价。** 尽管三种子 layer0–8 短前缀的 KL、PPL 均同向改善，三种子 full28 的 PPL 均恶化、均值 KL 也略恶化；因此当前跨层 row-profile 初始化 v 方法为失败。中等 donor shrinkage 的 layer5 对照同样明确失败，说明“迁移 + 小系数”不会自然安全。该方法没有增加 backward，但会改变大量坐标级 Adam 分母，且增加 profile 数据/状态操作。

**反思与下一步。** 短前缀 KL/PPL 正向结果没有代表 downstream 层对扰动的累积反馈；三种子完整结果足以停止当前方案，不再扫迁移强度。seed42/43 的更新 cosine 约 .85、p95 (\hat v) 比约 360–377，说明粗 row prior 大幅改写坐标级预条件；这是待验证机制解释，不是已证明的因果结论。当前新证据显示，ADOPT 的更新范数/局部 surrogate 异常与 pure-gradient direction 的 KL 恶化都不足以支持“缩短记忆”或“撤销对角预条件”。下一步先用已保存的成对 refresh 轨迹，对比 Adam 和候选的更新 L2、坐标级更新分布、surrogate 前后变化，以及真实后续 GPTQ 后的 KL；先解释早期 block 的分歧，再决定是否有依据再启动一种方法。Kim 的理论只适用于特定随机 oracle；如果 REAL-Q 没有稳定失配，不启动动态 (\beta_2) 算法实验。

## 尚未验证的文献候选与实验门槛

### 2026-10-06 文献复查（历史记录，2026-10-07 后续结果见 B6 小节）

**问题与当前证据。** layer3–26 每次 refresh 重置活跃后缀的一阶矩 (m) 及其 bias-correction age，在 32×128 校准、32×256 评测的三个配对 seed 上平均 KL 下降 2.48%，但 PPL 平均只下降 0.29%；同一 seed42 换成 64×256 校准、64×512 评测后，KL/PPL 分别恶化 0.56%/0.85%。这不是“样本数增加”的单因素试验：校准/评测序列数和序列长度均改变，且只有一个中配置 seed。因此当前能下的结论是固定层范围的重启策略不具规模稳健性；还不能断言反转由校准数据量、校准序列长度、评测集变化或 Adam 状态本身造成。该固定范围策略暂停推广，不继续扫层边界、重置强度或学习率。

**文献证据与适配判断。** 目前没找到直接研究 REAL-Q 这种“逐列块舍入、GPTQ 闭式补偿、每次对活动浮点后缀做一次 Adam 更新”的工作。最有用的是下面几种相邻机制：

| 文献 | 文献实际支持什么 | 对 REAL-Q 的判断 |
|---|---|---|
| Asadi et al., *Resetting the Optimizer in Deep RL* (NeurIPS 2023) | 当连续优化阶段的目标函数变化时，旧的一、二阶矩可能污染新阶段；在 Rainbow/Atari 中重置优化器状态有实证收益。[论文](https://papers.nips.cc/paper_files/paper/2023/hash/e4bf5c3245fd92a4554a16af9803b757-Abstract-Conference.html) | 支持“目标变了时值得检验重置”，不支持每个 REAL-Q refresh 必须重置。其阶段更替是 RL 迭代，不是 GPTQ 格点与补偿。我们的固定层重置在中配置反转，故不能以该论文为正向结果依据。 |
| Ellis et al., *Adam on Local Time* (NeurIPS 2024) | 梯度尺度发生阶段变化时，全局 bias-correction 时间可能造成不合适的 Adam 步长；Adam-Rel 用阶段内局部时间，而不是只按全程步数修正。[论文](https://papers.nips.cc/paper_files/paper/2024/hash/f2733d3b0dde1d74995f35a9cf442d38-Abstract-Conference.html) | 与 REAL-Q 的离散 refresh 最接近。它提示要把“保留已有 (m,v)，调整局部 bias age”与“清空 (m)”分开研究；这不同于当前 (m+age) 重启。每个 refresh 是否真是新阶段仍未知，不能在所有 refresh 无条件归零 age。 |
| Huang et al., *SPAM* (ICLR 2025) | 对实测梯度尖峰使用 spike-aware clipping 与 momentum reset，并配套 reset 后 warmup。[论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7a70ad3d9c704fb9b81b5c69eda722dc-Abstract-Conference.html) | 只有轨迹先显示异常梯度尖峰时才相关。当前证据没有建立 REAL-Q refresh 存在这类 spike；照搬周期重置或 clipping 没有依据，而且多步 warmup 与每块仅一次 Adam 更新不匹配。 |
| Wang et al., *AdaR* (IJCAI 2025) | 用两套矩状态交替，在固定区间中提前积累下一阶段的状态，以丢弃长尾历史。[论文](https://www.ijcai.org/proceedings/2025/721) | REAL-Q 的模块、活动尾部和 refresh 次数不一致；双状态会增加状态显存，固定周期还引入新的模块级超参。目前不适合作为首个移植方案。 |
| Guo, *Delayed Optimizer-State Transport Shapes Short-Horizon Training Decisions* (arXiv:2608.24593, 2026) | 在固定未来 minibatch 路径的短程 AdamW 轨迹上，显式把参数、m、v 和时钟作为状态；报告即时导数与完整有限时域传递会选出不同干预，说明 optimizer state 的作用依赖后续路径。[论文](https://arxiv.org/abs/2608.24593) | 支持用相同快照、实际后续 GPTQ 路径评估状态改动，不能支持“跨层迁移 v”：论文研究的是训练轨迹中的切线传递和未来数据顺序，没有跨层矩转移或 PTQ 量化实验。它进一步说明单步/局部 KL 不能替代完整量化轨迹。 |
| Liu et al., *G²PTQ* (arXiv:2609.31009, 2026) | PTQ 中将 trust-region 缩放用于精确一阶 GPTQ 补偿项，限制局部二阶近似下的更新幅度；在 Qwen3-0.6B 的 W4A16 报告 KL 改善。[论文](https://arxiv.org/abs/2609.31009) | PTQ 场景直接相关，但对象是 GPTQ 解析补偿，不是 Adam 的 m/v 更新；论文用块级曲率、GPTQ 误差和阈值 α，其公式不可直接移植到 REAL-Q Adam。可作为“曲率约束步长”的设计参照；REAL-Q 若要用，必须从已有梯度/Fisher 推导自己的步长因子并先验证额外前向成本。 |

**已测与未测边界（更正）。** 先前将“保留 m/v、仅重置 bias age”笼统列为未实现/未实验，是记录审查遗漏：B 节已经完成每-refresh重置 age 的两种子前缀试验，KL 均恶化。按观测到的阶段变化才调整局部时间尚未定义，也没有触发证据，不能以 Adam-Rel 名义重新推出旧方案。C2 已完成原始梯度方向与 Adam 总 L2 范数匹配的四点实验，全部 KL 恶化；D 节也已完成 Cautious 的 mask-only/scale-only 与真实下一 GPTQ 对照。页首所提 B6 m+age 重启方向/幅度拆分现已完成，结果见 B6 小节：匹配全局 L2 后 KL 仍四点全恶化；这排除了“只需匹配整体步长就能修复 B6”的简单解释，但不单独识别坐标 Fisher 几何、模块非线性或后续层累积的贡献。不得把“匹配范数”或“量化后评分”本身描述为新的实验方法。

**评价与反思。** 文献提供了“阶段变化可让旧矩/全局时间失配”的动机，但没有一个已验证的方法能直接解决本项目问题。当前小配置到中配置的结果反转，更像是提醒固定范围 heuristic 对数据/上下文规模敏感；在校准和评测的序列数、长度拆开前，既不能把它归因于 Adam，也不应宣称局部时间方法会修复。最严谨的下一步是复用/捕获同一层同一 refresh 的原始 (g,m,v) 快照，并在相同真实 GPTQ 后续下比较更新机制；若没有可复用快照，则先增加轻量轨迹日志，不启动全模型候选实验。KL 为主指标，PPL 为辅助，surrogate loss 不单独作为成功标准。

以下属于有根据的候选或机制参照，不可描述成已在 REAL-Q 验证：CPAL 非零 (v_0) 已在 A 节审计，但额外 backward 成本不符合主约束；SPAM 的 spike-aware clipping/reset/warmup 与本项目离散列更新机制并不相同，只有出现梯度 spike 证据后才值得做局部检验。我们测试了 Grams 式梯度方向替换，但没有单独测试恒定幅值 SignSGD，不能混称两者。RAdam、AdaBelief 与 ADOPT 的直接移植已有本项目负结果。未来的动态记忆候选必须先由保存的真实轨迹给出机制预测，再做同预算、逐 seed 配对；不因方法来自文献就默认迁移有效。

### 梯度裁剪审查（代码事实；暂不作为新算法）

**理论支持与风险。** 裁剪可能限制少数异常大的梯度坐标，但逐坐标裁剪一般会改变梯度方向；在增量/分批梯度方法中，梯度裁剪引入的方向偏差取决于裁剪阈值、各样本梯度范数及其方向关系（Qian et al., AISTATS 2021，[论文](https://proceedings.mlr.press/v130/qian21a.html)）。Adam 的逐坐标归一化使“梯度值大”不自动意味着参数步长同样大；裁剪对首次近似无历史的 Adam 步，可能大部分被二阶矩归一化抵消，但跨 refresh 的 (m,v) 会记住裁剪后的统计，因此仍可能改变后续更新。必须量测裁剪比例和真实下一块 GPTQ 后的 KL，不能只凭数值稳定性推断量化质量。

**原论文与项目代码的区别。** REAL-Q 论文描述了小型 Qwen 模型 Fisher-MSE 中对输出扰动 \(\Delta y\) 的 P95 `a_loss_clip` 缩放，以及构造 Hessian 时对 saliency 的 P99 裁剪；它们作用在 Fisher 损失输入/显著性上，不是对 Adam 收到的权重梯度做逐坐标 clamp。论文给出 Adam 及超参数，但没有规定 Adam 权重梯度的裁剪算子和阈值（[REAL-Q 论文](https://arxiv.org/abs/2609.00049)；也见本地复现协议中的未披露项）。当前仓库实现则确实在 Adam 更新前逐元素执行 `clamp(-grad_clip, grad_clip)`；参数默认为 `grad_clip=1.0`，负值关闭，见 `process_args.py` 与 `realq/refresh/block_gd.py`。按本项目保存的代表性梯度统计（中位数绝对值约 (2.8\times10^{-6})，最大约 (7.5\times10^{-5})），阈值1.0在该运行中不会触发；这只说明该次运行实际近似未裁剪，不能推广为所有模型/设置都不触发。仓库中的阈值 (5\times10^{-5}) 来自既往实验脚本，不是论文要求。

**评价与反思。** 目前没有证据支持把更强的梯度裁剪作为 REAL-Q Adam 的改进：论文没有提出它，当前默认阈值在已测梯度尺度下不生效，而更低阈值可能选择性压扁少数坐标并改变 EMA。SPAM 的 clipping/reset 只针对经测量的梯度尖峰并配合 reset 后 warmup（Huang et al., ICLR 2025，[论文](https://proceedings.iclr.cc/paper_files/2025/hash/7a70ad3d9c704fb9b81b5c69eda722dc-Abstract-Conference.html)）；项目尚未证明存在相同尖峰机制。因此先记录每次 refresh 的 `max(|g|)`、被裁坐标比例、被裁梯度 L1/L2 mass、裁剪前后 Adam update cosine/L2，再决定是否做单快照对照；不开始阈值 sweep，也不将裁剪称为已验证优化。

**新文献审查：AdaR（未移植、未实验）。** Wang et al. 的 AdaR 用两套 一/二阶矩轮流承担更新：每个固定长度区间的中点重置备用状态并累积“较新的半区间”梯度，区间边界再交换；因此它不是单纯每隔若干步将当前状态归零。论文报告其在图像和小型语言模型上的实验结果，但区间长度经粗细网格搜索选择，论文建议长度为 12。这个机制与 REAL-Q 的 refresh/量化阶段变更有概念联系；然而 REAL-Q 不同线性模块的 refresh 数、活动列后缀和最终层 KL 路径并不统一。为适配而另设周期会变成模块特定超参；两套 (m,v) 状态也会增加活跃模块的优化器内存。基于这两点，本轮只将 AdaR 记为待参考工作，不直接移植、不声称有益，也不先做 λ 参数扫描。只有之后的逐 refresh 诊断确实表明“旧状态随时间变陈”并能指出合理的状态更新边界，才重新考虑小型机制测试。

**新文献审查：优化器状态低精度量化与重置（仅作机制对照）。** Topollai & Choromanska 研究 LLM 预训练中被低精度存储的 EMA：舍入会让名义更新多次无法改变已存状态，形成“staleness”；其重置策略是在观测到该低精度停滞机制后恢复响应性。当前 REAL-Q 的 Adam (m,v) 缓冲区是 FP32，且长程低精度 EMA 累积停滞的必要条件未见于代码，所以这篇工作不能作为本项目清 (m) 的理论证明，也不支持照搬其周期。它反而给出一个很好的反例筛查：只有测到具体 stale-state 机制，才有依据按 stale 触发重启。另一个相邻工作 ECO 将量化训练的权重量化残差反馈进 optimizer momentum；其目标是每步直接更新低精度权重，而 REAL-Q 的列块量化残差已通过 GPTQ 闭式补偿传播到浮点后缀，直接再注入 (m) 有重复补偿风险。两篇都未移植、未实验；后续若研究 residual-feedback，先从现有轨迹验证闭式补偿后是否还留有与后续梯度同向的系统残差，再设计最小成对测试。

## 实验清单和 artifact 索引

| 方向 | 结果/审计文件 |
|---|---|
| 文献与 REAL-Q Adam 适配审计 | [`adam_adaptation_literature_audit_20261005.md`](adam_adaptation_literature_audit_20261005.md) |
| Adam 状态隔离/偏置 age 重置 | [`adam_refresh_followup_design_20261005.md`](adam_refresh_followup_design_20261005.md)、`outputs/realq-adam-history-isolated-moments-seed{42,43}-20261005/summary.json` |
| Cautious 同快照/下一 GPTQ | [`cautious_transition_diagnostic_20261005.md`](cautious_transition_diagnostic_20261005.md) |
| Grams 方向替换 | `outputs/realq-adam-direction-direction-snapshots-full-20261005/summary.json` |
| Adam 总步长匹配的原始梯度方向 | `outputs/realq-adam-direction-normmatched-gradient-seed42-43-20261006/summary.json`、各目标的 `status.json` 与 `run.log` |
| Adam-aware shadow preview | [`shadow_quant_mechanism_audit_20260929.md`](shadow_quant_mechanism_audit_20260929.md) |
| 跨层 (v) 迁移/当前 shrinkage 配对 | [`adam_cross_module_transfer_screen_20261005.md`](adam_cross_module_transfer_screen_20261005.md)、`outputs/adam-vtransfer-shrinkage-seed*-*/summary.json` |
| 当前矩漂移诊断及 v-only age 修正筛选 | `outputs/adam-moment-drift-diagnostic-seed42-l8/diag/moment_drift_summary.md`、`outputs/realq-adam-history-v-only-vage-seed42-20261006/summary.json` |
| RAdam early-step rectification | `outputs/realq-adam-history-radam-seed{42,43,44}-20261006/summary.json` |
| AdaBelief prediction-residual second moment | `outputs/realq-adam-history-adabelief-seed{42,43,44}-20261006/summary.json`、`outputs/realq-adam-history-adabelief-full28-seed42-20261006/summary.json` |
| ADOPT stale-second-moment normalized momentum | `outputs/realq-adam-history-adopt-seed{42,43}-screen-20261006/summary.json`、`outputs/realq-adam-history-adopt-seed44-retry-screen-20261006/summary.json` |
| 修正 m-age 的一阶矩重启 | `outputs/realq-adam-history-corrected-m-restart-seed{42,43,44}-20261006/summary.json`、前27层历史筛选 `...corrected-m-restart-full28-seed{42,43,44}-20261006/summary.json`；严格全路径三种子 `outputs/realq-adam-history-allpath-m-restart-full28-seed42-retry-20261006/summary.json` 与 `...seed{43,44}-20261006/summary.json` |
| Adam reset scope 消融与中配置反证 | `outputs/realq-adam-history-scope-early0-2-seed42-20261006/summary.json`; `outputs/realq-adam-history-scope-mid0-26-seed{42,43,44}-20261006/summary.json`; `outputs/realq-adam-history-scope-mid0-26-medium64-seed42-20261006/summary.json` |
| B6 大配置六种子及 Qwen3-1.7B 跨模型尝试 | `outputs/realq-adam-history-allpath-m-restart-full28-n32-s128-e32x256-seed{42,43,44,45,46,47}-20261006/summary.json`；Qwen3-1.7B 两次无效尝试见 B6 小节 |

**检查点归档与磁盘恢复。** 为从实验导致的 D 盘空间不足中恢复，已完成的 AdaBelief checkpoint 文件从两个目录移动到 `C:\realq_adam_checkpoint_archive\`，通过 `manifest.json` 记录原路径、SHA-256 和字节数；对应项目中的 summary、status、run.log 仍保留。此处是可逆的文件归档，未删除检查点内容。新方向实验为避免重复占满磁盘，checkpoint 保存后先记录 SHA-256，再由 runner 删除生成的 checkpoint 文件；该实验的过程指标、日志、源码散列均保留。

## 文献目录

1. Kingma, D. P. & Ba, J. *Adam: A Method for Stochastic Optimization*. 2014. [arXiv](https://arxiv.org/abs/1412.6980).
2. Liu, L. et al. *On the Variance of the Adaptive Learning Rate and Beyond*. ICLR 2020 (RAdam). [arXiv](https://arxiv.org/abs/1908.03265).
3. Zhuang, J. et al. *AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients*. NeurIPS 2020. [Proceedings](https://proceedings.neurips.cc/paper/2020/hash/d9d4f495e875a2e075a1a4a6e1b9770f-Abstract.html).
4. Abuduweili, A. & Liu, C. *Revisiting the Initial Steps in Adaptive Gradient Descent Optimization*. CPAL/PMLR 280, 2025. [PMLR](https://proceedings.mlr.press/v280/abuduweili25a.html).
5. Ellis, B. et al. *Adam on Local Time: Addressing Nonstationarity in RL with Relative Adam Timesteps*. NeurIPS 2024. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f2733d3b0dde1d74995f35a9cf442d38-Abstract-Conference.html).
6. Cao, Y., Li, X. & Song, Z. *Grams: Gradient Descent with Adaptive Momentum Scaling*. 2024 preprint. [arXiv](https://arxiv.org/abs/2412.17107).
7. Liang, K. et al. *Cautious Optimizers: Improving Training with One Line of Code*. ICLR 2026. [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ae03bdef276132fae089692445725635-Abstract-Conference.html); [arXiv preprint](https://arxiv.org/abs/2411.16085).
8. Kim, J. *Second-Moment Memory in Coordinatewise Adam*. 2026 preprint. [arXiv:2608.15824](https://arxiv.org/abs/2608.15824).
9. Lee, J. et al. *Scheduling Weight Transitions for Quantization-Aware Training*. ICCV 2025. [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2025/html/Lee_Scheduling_Weight_Transitions_for_Quantization-Aware_Training_ICCV_2025_paper.html).
10. Taniguchi, S. et al. *ADOPT: Modified Adam Can Converge with Any \(\beta_2\) with the Optimal Rate*. NeurIPS 2024. [arXiv](https://arxiv.org/abs/2411.02853), [proceedings PDF](https://papers.neurips.cc/paper_files/paper/2024/file/84d286e32bbee8fa3a86ee9c50e00081-Paper-Conference.pdf).
11. Reddi, S. J., Kale, S. & Kumar, S. *On the Convergence of Adam and Beyond*. ICLR 2019. [OpenReview](https://openreview.net/pdf?id=H1x-x309tm).
12. Ghadimi, S. & Lan, G. *Stochastic First- and Zeroth-Order Methods for Nonconvex Stochastic Programming*. SIAM Journal on Optimization 23(4), 2013. [arXiv](https://arxiv.org/abs/1309.5549).
13. Asadi, K., Fakoor, R. & Sabach, S. *Resetting the Optimizer in Deep RL: An Empirical Study*. NeurIPS 2023. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e4bf5c3245fd92a4554a16af9803b757-Abstract-Conference.html).
14. Wang, Y., Ding, L. & Shi, P. *AdaR: An Adaptive Gradient Method with Cyclical Restarting of Moment Estimations*. IJCAI 2025, pp. 6478–6486. [Proceedings](https://www.ijcai.org/proceedings/2025/721), [PDF](https://www.ijcai.org/proceedings/2025/0721.pdf).
15. Topollai, K. & Choromanska, A. *Understanding Quantization of Optimizer States in LLM Pre-training: Dynamics of State Staleness and Effectiveness of State Resets*. arXiv:2603.16731v2, revised 2026-09-27. [arXiv](https://arxiv.org/abs/2603.16731).
16. Nikdan, M. et al. *ECO: Quantized Training without Full-Precision Master Weights*. arXiv:2601.22101, 2026. [arXiv](https://arxiv.org/abs/2601.22101).
17. Guo, J. *Delayed Optimizer-State Transport Shapes Short-Horizon Training Decisions*. arXiv:2608.24593, 2026. [arXiv](https://arxiv.org/abs/2608.24593).
18. Liu, R. et al. *G²PTQ: Improving LLM Post-Training Quantization with Generalized Gradient Compensation*. arXiv:2609.31009, 2026. [arXiv](https://arxiv.org/abs/2609.31009).

