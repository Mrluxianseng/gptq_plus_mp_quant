# REAL-Q 改进方向：文献与现有证据复评

日期：2026-09-28。范围：阅读论文方法部分、附录与本地实验记录，提出可证伪的后续路线。本轮没有启动 GPU 实验。所有新方向均未证实能改善 KL。

**研究定位与建议**

基线是 REAL-Q，目标是其 future work 在有限显存下的工程实现与验证。质量主线继续围绕损失/优化目标；不把增加校准数据、参数搜索、原作者的 refined_mse 或开启原有滑动窗口当作贡献。建议只保留两个待筛选机制：样本敏感度与误差的配对关系，以及连续更新收益经过后续取整后的保留程度。先测机制，选择一个主攻。冻结前缀复用可作为独立的效率备选，不能冒充 KL 改善。

REAL-Q 附录 E 的三个方向分别是非线性代理保真度、短且非平稳轨迹上的优化器/状态迁移、系统内存优化。当前 block 的真实非线性已进入前向和反向；在损失里再用同一个 block 输出构造 MSE，并不能叫作补上遗漏的 SwiGLU。论文的 loss sliding window 已涉及相邻 block，单纯扩大到下一 block 不是新增机制。[REAL-Q §4、附录 E](https://arxiv.org/html/2609.00049v2)

**已有证据：不能只保留正结果，也不能扩大负结果适用范围**

| 路线 | 本地证据 | 当前处置 |
|---|---|---|
| QEP 式跨模块闭式补偿 | 修正公式后的五层配对 KL 0.0987066 → 0.1101848；局部线性预测变好而整体指标变差 | 当前版本停止；不声称否定 QEP 原论文 |
| YAQA 式双侧舍入移植 | 修正行残差传输后，27 层筛查仍比匹配对照 KL 高 34%，耗时约 2.5 倍 | 当前移植停止；不把修 bug 本身写成质量贡献 |
| Fisher + MSE | 有 KL −2.92%、−0.62% 的早期小评测配对；后续有 +1.83%、+2.01%、+2.41%；k=4 配对另有 −1.3947% | 结果混合，只保留一次有明确边界的复验，不再称稳定有效 |
| 自适应 Fisher 收缩 | 四个配对的 KL 均变差，约 +1.2% 至 +3.44% | 停止泛化收缩/谱补偿搜索 |
| 当前与后续模块联合 Adam | 旧 smoke 有明显负结果，但旧审计发现学习率日程等混杂 | 不能视作新空白方向，也不能据此证明所有联合优化必败；低优先级 |
| SwiGLU 响应匹配 | 只有设计，没有质量实验 | 暂缓；额外 FFN 计算以及妨碍上游误差补偿的风险尚无正面证据抵消 |

关键记录：

- [修正 cross 的配对](../outputs/resume_20260927_1216/CROSS_RESULT.md)
- [修正双侧残差传输后的筛查](../outputs/resume_20260927_ts_quality-screen8/TS_QUALITY_RESULT.md)
- [hybrid 早期验证](../outputs/resume_20260927_ffn_hybrid_validation/VALIDATION_RESULT.md)
- [hybrid 后续负结果](../outputs/ffn_hybrid_factorial_audit_20260928/results.json)
- [hybrid 另一评测窗口的负结果](../outputs/ffn_hybrid_third_replication_20260928/results.json)
- [k=4 配对](../outputs/ffn_hybrid_rademacher_pair_20260928/results.json)
- [收缩损失评估](../outputs/ffn_adaptive_shrinkage_assessment_20260928/ASSESSMENT.md)
- [历史实现审计](../outputs/overnight_20260927_0143/AUDIT.md)

k=4 的配对双方均 k=4，所以该配对本身有效；但 k 会影响 Fisher 和 saliency，不能把 k=0、k=4 的差异直接归因于 Fisher 质量。不能平均不同协议下的百分比来宣传总体收益。

**先清除会误导设计的解释和配置混杂**

1. 当前 `fisher_diag_mse` 用完整共享 Fisher；`legacy_fisher_diag_mse` 才是另一条逐 token 对角路径。用低秩矩阵替换现有完整矩阵本身不会增加跨通道信息。
2. k=4 正向 hybrid 配对实际是 32×256 校准、8×1024 评估，`rotate=False`、`loss_slide_window=False`、`a_loss_ratio=1`。它是小配置 REAL-Q 派生流程的筛查，不能直接声称优于论文完整机制。让双方具有相同的原方法机制是验证新增方法的必要控制，不是重新把复现原论文当贡献。
3. `is_fisher_backed_loss` / `is_full_fisher_backed_loss` 的集合未包含 hybrid 名称，相关截断和滑窗路由需要在启用前审查。上述禁用这些机制的配对未因此自动失效。
4. Hessian 累积时序取决于执行路径。当前 `gptq_plus_utils.py` 中 `layer_hessian_once = dynsal_refresh_mode == "per_layer"`；另一分支按顺序 group 重算。近期日志为 `per_boundary`；`realq/runner/layer_loop.py` 也按 group 重放收集 H。旧 summary 中“一律在所有模块量化前算一次”的描述不能当通用事实。模块内部列块循环则不会每次 Adam 后重建 H。
5. 旧 `D:/gptq_plus/adam_analysis_for_ptq.md` 关于“50 步后 bias-corrected v 只有真实值的 5%”的推导错误。若二阶矩平稳、v0=0，则 E[v_t]=(1−β2^t)E[g²]，除以同一因子后 E[vhat_t]=E[g²]。短轨迹仍可能有高方差、非平稳性和方向滞后，但不能用该错误偏差推导新优化器。梯度平方二阶矩也不等于精确 Hessian。[Adam 原论文](https://arxiv.org/abs/1412.6980)
6. 将共享输出度量从无约束线性最小二乘正规方程中消掉，需要对应的可逆性、线性结构及正则化形式。不能据此推出实际 REAL-Q 中 Adam 只对 Fisher 非对角项有价值。去掉 PSD Fisher 的对角后，剩余矩阵一般也不再 PSD。

**候选 A：保留 token 敏感度与量化误差的对应关系，优先验证**

REAL-Q §4.2 明确采用 mean-field 解耦：逐 token 曲率与其误差外积的联合期望，被共享曲率和误差分布的组合替代。该近似给出一个比“非线性强所以加 MSE”更具体的切入口。这里提出的是提高样本相关曲率代理的保真度，不声称已经建立高阶 SwiGLU 代理。

令 t 合并样本和 token 索引，δ_t 是真实当前 block 前向输出与 FP 目标之差。设 g_t^(k) 是正确采样和归一化的 teacher score-gradient 探针。一个诊断原型是：

    F_bar = mean_{t,k} g_t^(k) g_t^(k)^T
    L_pool = 1/2 mean_t δ_t^T F_bar δ_t
    L_match = 1/(2K) mean_t sum_k (g_t^(k)^T δ_t)^2

两者区别在于是否保留同一 token 的敏感度与误差配对。L_match 保留每个探针外积的跨通道信息，区别于 legacy 的逐 token 对角加权；但有限探针数会产生高方差与大零空间。它仍是局部二阶近似，仍可能忽略跨 token 曲率以及较大扰动的高阶项，不能称为精确 KL。

YAQA §3.2 的 Fisher sketch 讨论对本设计有直接警示：LLM 中 token 非独立，降低采样方差的近似可能引入偏差。FIMA-Q 的结构化 Fisher 近似说明可以将曲率表达与损失实现联合设计，但其验证对象是 ViT，且方法不同；两篇都没有证明本候选在 REAL-Q 上有效。[YAQA](https://arxiv.org/html/2505.22988) [FIMA-Q](https://arxiv.org/html/2506.11543)

首个实验应为固定状态的机制诊断：

- 从 REAL-Q 的真实列块量化轨迹保存前、中、后部若干层的 gate/up/down 状态、样本编号和 Adam 状态，不能只用“全模型 RTN + 第一层”的状态代替。
- 用相同探针计算 pooled 与 matched 损失，再将探针的 token 配对打乱作为控制。打乱控制保持探针集合及聚合 Fisher，不引入额外校准序列。
- 检查对实际 end-to-end KL 变化的排序、实际 Adam 更新的下降方向以及应用后的 KL。原始 surrogate 梯度余弦只是诊断，Adam 预条件后的实际步长还需单独检查。
- 如果 matched 与 shuffled 没有可重复差异，或实际 KL 下降没有改善，就停止该机制，不继续扫分组数和系数。

若机制成立再压缩实现：可保存少量投影统计，或构建按 teacher 状态条件化的少量 Fisher；先证明信息有用，再决定表示方式。一个保持 PSD 的谱子空间实现是 F_t^eff=U B_t U^T+F_perp，其中 U 是 F_bar 的一部分正交特征向量，F_perp=F_bar−UΛU^T，B_t 为投影 score 的二阶矩。若聚合口径相同，平均 B_t=Λ。这保留未替换子空间的原 Fisher；并非简单抬高所有弱特征值。

工程成本：不保存每 token 的 d×d 矩阵。诊断只保存少量层的探针；正式实现需要评估 CPU/磁盘缓存、重新计算探针和 GPU 分批传输的取舍。特征子空间在聚合后才确定，不能未经实现就承诺所有投影都能免费复用一次 Stage-0。现有 rademacher_k 也不能未经审查直接当成独立标签探针数。初次对照必须保持 saliency 和闭式 H 不变，仅改变 Adam 损失。

**候选 B：让损失看到即将发生的取整，作为有限备选**

REAL-Q 当前刷新在“已量化前缀 + 未量化浮点尾部”上求梯度。可检验的问题是：浮点尾部上的改善，是否经下一步实际 GPTQ 取整后消失或反向。当前有 horizon 更新时序加权和双侧舍入实现，但这两者不等价于在即将量化的状态上定义刷新损失。

AdaRound 把舍入选择纳入重建优化；EfficientQAT 把量化/反量化放入 block 计算图。由此可借鉴“针对量化后状态优化”的原则，在 REAL-Q 内只对下一待固定列块构造量化影子，其余列保留当前状态，仍用原 Fisher/滑窗目标和真实非线性前向。[AdaRound §3.3](https://proceedings.mlr.press/v119/nagel20a/nagel20a.pdf) [EfficientQAT §3.2、附录 B](https://arxiv.org/html/2407.11062v3)

先做诊断：同一状态分支 A 跳过一次 Adam、分支 B 执行原 Adam；两边都执行相同下一步真实量化，再测最终输出 KL。如果执行 Adam 的优势并未在取整后系统性消失，该方向不应优先投入。

候选实现保留 FP master、已固定列掩码和原量化尺度；下一列块在 loss 前向中临时 fake-quantize，反向可使用明确的 STE 或软舍入假设。它不转移跨模块坐标，也不要求增添校准样本。这是在改优化目标对应的量化状态，不能直接宣传为高阶非线性模型。

风险：STE 有偏；AdaRound 本身报告过其受限优化中 STE 弱于软舍入，不能只因 EfficientQAT 成功就认定短轨迹上也成功。廉价 nearest-rounding 影子也不等于包含 GPTQ 补偿的真实下一状态；候选必须用实际下一步量化结果验证。复用原前反向有望限制开销，但还未测量，不能承诺 GPU 时间只增几个百分点。

**现有方向还能怎样优化**

- Hybrid：更准确的定位是共享曲率的正则化。先确认 k=4 信号能否在相同协议、固定独立评测和多个校准种子下保留，并加入纯 Fisher 整体缩放控制；不继续堆 MSE 变体。启用标准滑窗/截断时先修正相同路由，再谈相对 REAL-Q 的增量。正负结果都保留。
- 响应匹配：Sobolev/Jacobian matching 的文献支持的是通用函数响应约束，不能自动推出其对 REAL-Q KL 有利。真实 FFN 输入漂移小的时候项可能近零；强制恢复 teacher 响应也可能阻碍已量化上游误差的补偿。暂排在 A、B 之后。
- 相邻模块联合更新：旧 joint 梯度铺开已覆盖部分思想，且存在 LR 日程混杂。若将来重启，必须先把“依赖图选择目标、同样更新预算、同样层级 LR、下一模块自身量化后的保留收益”定义清楚。当前不直接再跑完整联合更新。
- 冷启动：Adafactor 提醒了慢二阶矩响应、更新裁剪等问题，但换优化器或改 β 本身不足以构成贡献。先在真实轨迹上测 Adam 更新是否因旧状态变得有害；不把现有 Fisher 当作精确 Adam v，也不重新包装已经做过的 warm/closed-v。[Adafactor](https://proceedings.mlr.press/v80/shazeer18a.html)

**工程备选：冻结计算前缀的精确复用**

量化 down_proj 时，当前层 attention 和 gate/up 权重固定；可缓存 down 输入及残差，只重放与 down 更新有关的后缀。AutoFreeze 已研究冻结前缀的激活缓存及传输/重算成本权衡，方法参考明确。[AutoFreeze §3.2](https://arxiv.org/html/2102.01386)

本项目已有 `gptq_utils/frozen_prefix_replay.py`。down_proj 缓存的两组初始完整配对均逐位一致且小幅提速；随后额外配对中一组分叉并变慢，固定 cuBLAS 工作区的配对则重新逐位一致、只快 0.43%。真实 refresh 级别的两次 621 组对照（有/无 cuBLAS 设置）梯度、loss 和计数全部逐位相同，refresh 本身快约 13–15%；这不能解释那次完整量化分叉，故还不能声称稳定的端到端加速。全 FFN 扩展通过逐位前向核对，但无核对运行有一次完整状态分叉且没有稳定提速，因此目前不作为默认实现。详细结果见 [Frozen-prefix replay 审计](frozen_ffn_replay_audit_20260929.md)。

2026-09-29 两组初始完整配对使用同源码、同 seed、同参数，顺序分别为 candidate/control、control/candidate。最终 507 张量 SHA、KL、PPL 都相同，量化耗时下降 2.15%、1.25%，端到端下降 1.49%、1.19%。但后来一组无 cuBLAS 固定设置的完整配对出现模型状态和 KL 差异（且候选慢 3.66%）；再固定 cuBLAS 的配对状态、KL、PPL 完全相同，只有 0.43% 的量化耗时下降。两次真实 refresh 对照分别 621 组梯度逐位相同，但整模型存在一次未解释分叉。完整记录在 `outputs/frozen_prefix_timing_replication_20260929/extended_audit_20260929.json`。当前应把它称作局部正确、可能省少量时间的 opt-in 原型，不作为稳定的端到端加速结论。

同日新增的 `FrozenFFNPrefix` 原型把缓存扩展到 `gate_proj`、`up_proj` 和 `down_proj`。合成模型测试验证了输出与目标权重梯度；全模型核对验证了 7,560 次命中和 28 层量化后输出逐位一致。可是无核对全模型运行中 4 次有 1 次的最终状态哈希不同，KL 从 0.222646 增至 0.223470，且扩展没有可重复加速，故停止把它当作可交付优化，保留为待查原型。结果和停止理由见 [Frozen-prefix replay 审计](frozen_ffn_replay_audit_20260929.md)。

**执行顺序与停止规则**

1. 整理相同协议下的结果和候选代码路由，保存可重放的真实量化状态。这是研究基础设施，不单列为算法贡献。
2. 小规模筛查 A 的 matched/shuffled 与 B 的取整后收益；对新损失检查梯度、目标索引、teacher 隔离、冻结列和实际触发次数。
3. 只把机制筛查通过的一条推进到全层 W4 配对。初筛保持固定种子和评测开发集，不利用最终测试集选择候选。
4. 全层结果成立后再做多种子、更多固定评测序列、另一位宽和另一个模型/规模。报告配对差异及按序列/校准种子的变异，不将 token 当独立样本虚增置信度。8 GB 上通过小 microbatch、累计和有证据的 checkpoint/offload 控制峰值，不能只减少候选一侧的数据量。
5. 包含等更新次数对照；若候选明显增算力，再加等 GPU 时间的原 REAL-Q 对照。KL 为主要指标，PPL 和耗时/显存完整报告。
6. 正向机制证据不等于最终贡献；实际全模型 KL 持续不改善就停止该版本。负结果可以解释设计选择，但不能包装成正向方法成果。

建议论文工作收敛为一个可重复的质量改进机制，以及一个可测的低资源实现或效率贡献；是否满足学位要求需以学院标准和导师意见为准。现有记录不支持保证任何候选成功，也不支持因为几个候选失败就判定整个课题不可完成。
