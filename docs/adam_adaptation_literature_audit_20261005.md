# REAL-Q 的 Adam 适配：文献与现有证据审计

日期：2026-10-05。范围是更新适配（短轨迹、冷启动与梯度分布漂移），不是 Adam 状态显存优化。REAL-Q 仍是项目基线，不作为外部相关工作计数。

## 结论

此前关于 Adam 的代码和诊断投入很多，但文献覆盖不够完整。尤其漏掉了与 `warm_adam` 最贴近的 Abuduweili 与 Liu 的二阶矩非零初始化工作。原先主要引用 Adam、RAdam/AdaBelief 的一般机制与 Adafactor，对“利用数据初始化 v₀”的工作没有做及时的重合审查。因此，不能说我们此前已经充分查阅了 Adam 适配文献。

目前可以成立的项目证据是：REAL-Q 轨迹中的首个 Adam 更新更容易受坐标符号与预条件影响；某个 warm-start prior 与实际逐坐标 g² 存在相关性；但尚无足够的全模型质量证据表明 warm-start 改善了最终 KL。Adam-aware shadow preview 的已完成全模型对照反而变差，不能把它算作成功的 Adam 优化。

## REAL-Q 中要适配的实际问题

项目的 Block-GD 每次 refresh 对当前仍浮点的尾部做一次带 bias correction 的 Adam 更新，量化前缀被锁定；一层中的一组线性模块会经历多个 refresh，模块间则重置状态。这里的梯度来自当前量化轨迹上的代理损失/真实 KL，而不是标准训练中围绕固定参数分布采到的随机 mini-batch 梯度。梯度分布会随量化前缀、待更新列范围和样本批次改变，更新次数也远少于常规模型训练。

应分别检验三类机制：

1. **冷启动**：零初始 (m, v) 与 bias correction 后，首个坐标更新接近裁剪后的符号方向；这会丢失本来来自梯度幅值的差别。
2. **短轨迹状态失配**：(β₂ = 0.999) 的长记忆可能对模块内不断变化的梯度反应迟缓；这不同于“bias correction 算错”。在平稳 E[g²] 的假设下，标准偏置修正并不会留下简单的 (1 − β₂ᵗ) 系统性低估。
3. **离散量化保留**：连续权重更新即使局部改善，也可能被下一列块 GPTQ 舍入抵消；这要与 Adam 更新本身的方向质量分开测。

旧笔记中“50 步后修正过的 (v) 只有真实值 5%”的推导已经审计为错误。Adam 的 bias correction 在平稳矩假设下会抵消零初始化带来的期望缩放；仍可能存在的是非平稳梯度、有限样本方差与状态滞后，不应再用错误的偏差论证设计新算法。

## 相关文献与适用边界

| 工作 | 与 REAL-Q 相关的机制 | 不可直接外推之处 |
|---|---|---|
| Kingma & Ba, *Adam: A Method for Stochastic Optimization* (2014) | 定义 (m, v)、bias correction，并讨论非平稳目标 | 训练优化的通用基线；未处理逐模块重置、序贯离散量化或低更新预算 |
| Liu et al., *On the Variance of the Adaptive Learning Rate and Beyond* / RAdam (ICLR 2020) | 将早期自适应步长高方差作为问题，使用 rectification 稳定初期更新 | 主要针对随机优化训练的初期阶段；需确认 REAL-Q 每模块短轨迹下的 rectification 是否合理 |
| Zhuang et al., *AdaBelief* (NeurIPS 2020) | 用梯度相对 EMA 预测的残差更新分母，针对梯度方向变化调整步长 | 更换 (v) 的定义；不能当作不改变 REAL-Q 算法的实现细节，且与量化前缀引起的非平稳性仅机制相似 |
| Abuduweili & Liu, *Revisiting the Initial Steps in Adaptive Gradient Descent Optimization* (CPAL 2025；先有 OPT 2024 版本) | 直接研究零初始化 v₀ 的初期符号下降，并提出数据驱动或随机的非零 v₀；数据驱动估计包含均值梯度平方和梯度方差 | 证据来自常规长程训练任务；不证明该策略能改善 REAL-Q 的 KL，也没有处理 Adam 更新后的 GPTQ 舍入 |
| Kim, *Second-Moment Memory in Coordinatewise Adam* (2026 预印本) | 理论分析长 β₂ 二阶矩记忆可能如何滞留大梯度、压低后续有效更新 | 结果建立在特定随机 oracle/步长假设上；REAL-Q 梯度的实际漂移形态需实测，不能直接断言大 β₂ 是本项目瓶颈 |

主要来源： [Adam 原论文](https://arxiv.org/abs/1412.6980)、[RAdam](https://arxiv.org/abs/1908.03265)、[AdaBelief](https://arxiv.org/abs/2010.07468)、[二阶矩初始化（CPAL 2025）](https://proceedings.mlr.press/v280/abuduweili25a.html)、[二阶矩记忆（2026 预印本）](https://arxiv.org/abs/2608.15824)。

### 在“不额外预热反传”约束下补充的文献

| 工作 | 可复用的现有量 | 对 REAL-Q 的具体启发 | 局限/优先级 |
|---|---|---|---|
| Balles & Hennig, *Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients* (2018) | 当前梯度、动量和二阶矩估计 | 把 Adam 更新拆成方向/符号效应与方差缩放效应；可据此分别测 `sign(g)`、`sign(m)`、`m/sqrt(v)` 与实际更新的对齐，避免只看 raw-gradient cosine | 主要是机制分析，不是 REAL-Q 改进方案；优先用于诊断 |
| Liang et al., *Cautious Optimizers: Improving Training with One Line of Code* (ICLR 2026；2024 预印本) | 当前梯度 `g_t` 与基准优化器已有的更新 `u_t` | 对更新方向与当前梯度不一致的坐标做掩码，再按保留坐标比例缩放；不需要额外反传。可作为“动量是否把更新带偏”的一步反事实候选 | 训练任务证据不能推出 GPTQ/KL 改善；硬掩码可能删掉对下一次 GPTQ 舍入有用的坐标，故先做单 refresh/单真实量化块检查，不直接全模型推广 |
| Joo et al., *On Surprising Effectiveness of Masking Updates in Adaptive Optimizers* / Magma (2026 预印本) | 当前梯度与动量对齐度 | 以动量-梯度对齐控制更新掩码概率，给出与 Cautious 硬掩码不同的软/随机门控思路；可作为“少数冲突坐标如何处理”的近邻文献 | 针对 LLM 预训练；会引入随机性，REAL-Q 短轨迹需要固定随机种子并检查跨种子 KL，优先级低于确定性的 Cautious 对照 |
| Liu et al., *On the Variance of the Adaptive Learning Rate and Beyond* / RAdam (ICLR 2020) | Adam 当前 `v_t` 与步数 `t` | 通过初期 rectification 控制自适应步长的方差，不需要额外梯度；对逐模块重置后的早期步数尤其值得作机制对照 | 论文研究常规随机训练的初期统计；REAL-Q 每个模块样本/梯度非平稳，必须先确认本项目早期有效步长确实异常 |
| Cao et al., *Grams: Gradient Descent with Adaptive Momentum Scaling* (2024 预印本) | 当前梯度和已有动量/方差状态 | 解耦更新方向与动量幅度，作为“方向是否应跟随当前梯度、历史状态只负责缩放”的更大幅度替代设计 | 算法改动较大，离 REAL-Q 当前机制诊断更远；列入背景，不建议作为第一轮候选 |

以上工作均不需要为每个模块单独额外反传，计算改动主要发生在已有梯度/状态上的逐元素更新。但除 RAdam/Cautious/Magma 的一般训练证据外，目前没有证据能直接保证它们改善 REAL-Q 的最终 KL。直接可用的参考链接：[Dissecting Adam](https://arxiv.org/abs/1705.07774)、[Cautious Optimizers（ICLR 2026 论文）](https://openreview.net/pdf?id=zBPZeRjfgu)、[Magma](https://arxiv.org/abs/2602.15322)、[RAdam](https://arxiv.org/abs/1908.03265)、[Grams](https://arxiv.org/abs/2412.17107)。

### 与“离散量化/补偿之间沿用 Adam 状态”更接近的文献

目前检索到的文献分别覆盖了相邻部件，但没有找到直接研究 REAL-Q 这种完整组合（先量化一块、闭式补偿浮点后缀、再对后缀做 Adam 梯度刷新）的论文。

| 工作 | 它研究什么 | 与 REAL-Q 的关系与边界 |
|---|---|---|
| Ellis et al., *Adam on Local Time: Addressing Nonstationarity in RL with Relative Adam Timesteps* (NeurIPS 2024) | Adam 面对阶段/目标变化时的一阶、二阶矩与偏差修正。Adam-Rel 保留动量状态、重置局部时间步；论文也比较完整重置动量的 Adam-MR，并发现简单丢掉历史动量可能损失有用信息 | 最贴近“阶段切换后是否沿用历史状态”这个问题；可启发区分重置 `m/v` 与只调整局部 bias-correction 时间步。它的阶段变化来自 RL 目标/数据更新，不是 GPTQ 格点舍入和 Hessian 闭式补偿，不能直接移植结论 |
| Sahu et al., *On the Provable Suboptimality of Momentum SGD in Nonstationary Stochastic Optimization* (ICML 2026) | 分析目标最优点持续漂移时，动量平滑抑制噪声与追踪漂移之间的权衡 | 支持检查 REAL-Q 的梯度是否随离散补偿快速漂移，以及旧动量是否产生跟踪滞后；理论对象是 momentum SGD 和动态目标，并非 Adam 或量化补偿 |
| Lee et al., *Scheduling Weight Transitions for Quantization-Aware Training* (ICCV 2025) | QAT 中 latent FP 权重跨过舍入阈值后，量化权重发生离散跳变；论文显示 latent 更新幅度与量化权重实际跳变不一一对应，并按量化权重转移率调度更新 | 直接证明“连续 latent 更新”和“离散量化结果”的作用不能混为一谈；但它研究 QAT 的 STE/latent-weight 训练，不研究 GPTQ 后缀闭式补偿，也不专门研究 Adam 的 `m/v` 应否跨补偿保留 |
| Lapucci et al., *Projected Gradient Methods with Momentum* (JOTA 2026) | 在约束优化中分析如何把动量与投影步骤组合，指出不能把无约束更新后简单加投影当作无影响的实现细节 | 为“参数更新后施加硬约束/投影会改变动量法动力学”提供数学参照；约束是凸集合投影，和非凸离散量化、GPTQ 误差补偿不同 |

这些文献共同支持一个研究问题，而不是预先给出答案：REAL-Q 每次 GPTQ/补偿刷新是否构成足够大的梯度阶段变化，使跨阶段的 `m/v` 变成滞后状态？应实测补偿前后梯度方向、范数和 Adam 预条件更新的变化，再比较保留、调整时间步或重置状态对下一真实量化块 KL 的影响。来源：[Adam on Local Time（NeurIPS 2024）](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f2733d3b0dde1d74995f35a9cf442d38-Abstract-Conference.html)、[非平稳目标下动量的追踪界（ICML 2026）](https://proceedings.mlr.press/v306/sahu26a.html)、[QAT 离散权重转移（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Lee_Scheduling_Weight_Transitions_for_Quantization-Aware_Training_ICCV_2025_paper.html)、[约束投影与动量（JOTA 2026）](https://link.springer.com/article/10.1007/s10957-026-03054-7)。

## 本项目实现和实验审计

### `warm_adam`

现有 `ModuleGradSecondMoment` 通过 refresh backward 的逐 token 梯度和输入，计算权重梯度批均值的逐坐标二阶矩：

\[
E[g_{\mathrm{batch}}^2] = \bar g^2 + \operatorname{Var}_s(g_s)/B.
\]

它用 (K) 个真实 refresh batch 测量 prior，再令 `exp_avg_sq = prior * (1 - beta2**t0)`，并只给二阶矩 bias correction 加 `t0`；一阶矩仍从零开始。这与 2025 文献“从数据估计非零初始 v₀”高度重合，不能把“非零初始化”本身宣称为新算法。项目特定部分是按实际 refresh loss、当前模块输入/输出坐标和 batch size 估计 E[g_batch²]，并试图在逐模块有限轨迹下使用；但它需要额外的每模块 refresh backward，且可能在当前权重/量化上下文与后续状态之间失配。

证据边界：`warm_start_steps=0` 的全模型 null test 用来证明管线与原 Adam 等价，不是收益实验；t0=1 的 layer-0 align probe 显示 prior 与实际逐坐标梯度平方的 log-correlation 大约 0.50–0.81，但只量化 layer 0 且跳过评估；现有 `compare_queue.log` 没有完成可用的 adam/warm_adam 质量配对。因此目前是“prior 可以测到且有相关性”，不是“Adam 冷启动已被解决”。

### Shadow/Adam-aware 方向

`shadow_quant_mechanism_audit_20260929.md` 的同状态诊断中，首块 raw-gradient cosine 约 0.99996，但 Adam 更新 cosine 约 0.9876、坐标符号分歧约 0.80%；后续块差异下降。这说明 Adam 变换可能放大少量坐标符号变化，不能单凭 raw-gradient cosine 认定更新近似相同。它只覆盖少量层/块和一个种子，尚未证明该敏感性导致最终 KL 损失。

一次 Adam-aware shadow preview 通过额外预览下一待量化块来匹配影子状态，在全模型配对中 KL 从 0.225465 增至 0.228842（+1.50%），PPL 从 32.73832 增至 32.91666（+0.54%），量化时间增加约 16%，峰值显存不变。该版本应停止，不能用作 Adam 改善结果。参见 [shadow audit](shadow_quant_mechanism_audit_20260929.md)。

## 后续该怎么研究

先停止把“更多 Adam 变体”当作路线图。优先做一条由新文献和本项目机制共同约束的比较：

1. 在同一真实量化轨迹快照、同一 loss/列范围/样本、同一学习率与裁剪下，先比较原始 Adam、RAdam 式初期 rectification，以及复用当前梯度/状态的 Cautious 掩码。数据初始化 v₀ 因需要额外预热反传，不再作为主路线；AdaBelief 仅保留为机制对照。不能把文献的训练超参照抄到量化流程。
2. 先测每个模块/层的 refresh 数、梯度 batch 间 cosine 与 g² 尺度漂移、m/√v 的有效步长分布、实际更新后局部 KL、以及下一真实 GPTQ 量化后收益保留率。这样区分冷启动、二阶矩滞后和舍入抵消。
3. 增加“无额外 warm-up backward”的主对照。若 prior 必须通过额外反传取得，应单列为资源成本，并与同 GPU 时间预算的 REAL-Q 对照；也要检查能否从现有前向/反向或已缓存统计中取得有效 prior，不能把 Fisher 直接当 Adam 的 E[g_W²]。
4. 仅把机制诊断通过的候选推进到多层/全模型，多种子确认 KL；PPL 做参考。若只是前几次更新的方向变稳定而取整后 KL 无增益，就停止，不改名包装为贡献。

当前最值得做的下一步不是再加一种 optimizer，而是按这份审计补齐 `warm_adam` 与 Adam-init 文献的同协议对照和轨迹诊断。是否继续投入全模型资源，应由低成本的真实 refresh/真实下一块 GPTQ 结果决定。
