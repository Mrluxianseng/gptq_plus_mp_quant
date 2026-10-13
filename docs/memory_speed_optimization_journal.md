# REAL-Q 显存优化与速度优化实验记录

> 本文是系统优化方向的单一持续记录。每项工作按“瓶颈—理论/文献依据—实现—配置与证据—评价—反思/后续”整理。REAL-Q 是基线和目标论文，不计作外部系统优化文献。本文持续追加已留下的代码、实验文档和对话摘要；它本身不表示本轮重新运行了所有历史实验。

最后盘点：2026-10-10。当前主线：**优先降低完整 REAL-Q 流程的每卡峰值 GPU 显存需求，使更大模型能完成校准与量化；速度是同一系统工作的次级指标。**已验证的核心工作主要来自 Qwen3-0.6B/W4A16 本地缩减配置。除特别标明外，不能将它们称为论文正式配置结果。

## 研究边界与验收原则

工程贡献应来自运算组织、内存生命周期、数据搬运/重算策略、kernel 融合或可复现的系统工具，而不是只改变超参数。除非专门研究量化算法，优化不得改变 REAL-Q 的量化顺序、闭式补偿、Block-GD 更新公式或目标函数。

主验收必须在相同模型、校准数据与样本、量化参数、种子、并行配置、软件版本和源码快照下配对。候选至少应记录：

1. 每个 rank 的 `max_memory_allocated`、`max_memory_reserved` 和阶段；另记录 NVML/device-used 峰值，区分 PyTorch allocator 与整卡占用。
2. 完整量化权重 state hash、逐层输出指纹，以及最终 KL、PPL。数值等价路径应达到预先定义的逐位或容差标准；有意改变随机抽样路径的候选必须单独评估质量，不能冒充精确等价。
3. 每阶段耗时、总量化时间和端到端时间；速度结论需空闲/受控 GPU 上至少三组配对重复，报告每组值、均值/中位数和波动。诊断插桩运行不用于速度结论。
4. CPU RAM、offload 搬运字节、GPU 数与通信量，避免把显存节省误报成系统总资源节省。

本项目对“显存优化”的强验收目标是：候选全流程峰值下降，且被比较的每个阶段/每张卡峰值不高于 control；如果候选峰值只是转移到另一阶段，必须用新的全局峰值判断。更少显存卡可运行是额外部署价值，不替代端到端峰值与正确性验收。

## 当前可陈述的结论

| 工作 | 当前证据 | 可以陈述 | 不能陈述 |
|---|---|---|---|
| 真实 KL token 行分块 | Qwen3-0.6B/W4，32×256 校准完整 28 层，3 组配对；state、KL/PPL 相同；allocated −53.2%、reserved −54.2% | 通过限制词表维 FP32 临时张量的存活范围，显著降低该配置的完整量化峰值显存 | 不能说全模型所有中间量都按相同比例降低；不能说加速；不能外推至所有模型/配置 |
| Triton 融合 GPTQ 列内循环 | 无竞争 CUDA 进程复测，3 组完整模型配对；量化加速中位数 2.143×、端到端 2.032×；state、KL/PPL 相同，allocator 峰值基本相同 | 受支持配置上融合减少列循环 launch 并显著加速量化 | 不是显存优化；不代表所有 kernel/config 都受支持 |
| KL 分块 + Triton 联合配置 | 完整模型逐层指纹 28/28 一致；state、KL/PPL 一致；allocated −53.8%、reserved −53.6% | 两项优化联合使用时，完整模型仍保持实测结果一致并降低 allocator 峰值 | 该 trace-hook 运行不作为速度测量；不能假设联合配置在其他硬件也有相同比例 |
| Packed Fisher 累积 | 0.6B/1.7B 缩减配置完整层结果一致；实际全程节省约 1–4% allocated，reserved 收益更小 | 对称 Fisher 的 GPU FP32 临时累积器可无损减半 | 不能把累积器节省 50%说成全程显存节省 50%；最终 Fisher cache 在当前实现中转到 CPU |
| Stage-0 saved-tensor CPU offload＋去重 | 2026-10-10 的 0.6B 2×1024、refresh microbatch1、完整28层三对：NVML均6469→3499 MiB（−45.91%），13缓存/28层输出/state/KL/PPL一致且对应精确区间无回升 | 长序列Stage-0主峰下的完整流程重复收益；详见 T/U 节；AI节另核对6臂跨重复一致 | 非正式规模、非多卡；RSS约增加60.5%；整卡时间加权平均只减少约0.37%，没有加速主张 |
| Stage-0整块重算＋评估CE分块完整组合 | 2026-10-10 AG/AI/AK，0.6B 2×1024、refresh microbatch2，共同offload_inps；完整3/3配对并交换顺序，13缓存/507-state/28层/KL/PPL精确相同，NVML均6461→4247 MiB，每对2961个已测阶段/边界对应项无回升 | 三对重复了峰值减少34.27%；allocated减少42.64%、reserved减少40.22%；RSS峰值接近对照 | 非正式规模、非两模型或4卡成果；NVML缺口导致平均占用未知，速度未测；Stage-2峰值相同；不与旧microbatch1结果拼接 |
| Stage-0 整块重算 | 0.6B 2×1024及1.7B 2×512各一对实际FSDP收集；13缓存及448模块梯度记录一致，已测区间无回升 | 两模型Stage-0峰值分别降低57.32%/32.18%；详见 V 节 | 尚不能当完整量化/速度重复结果；两模型序列长度不同，不能据此推断纯模型增长倍数 |
| 直接CPU创建输入缓存 | 两模型fresh prefix独立配对，缓存/状态/全部28层评估输出/质量一致，无区间回升 | 可安全改变捕获缓存的驻留位置，详见 AA 节 | 小输入实测全程峰值无明显下降；尚无大缓存完整REAL-Q容量结论 |

## 方法记录

### A. 真实 KL 的 token 行分块：当前最强显存结果

**瓶颈。** 末层 KL refresh 同时物化 student/teacher 的全词表 logits 相关 FP32 中间量。对于有效 token 行数 (R)、词表大小 (V)、chunk 行数 (C)，分块将大型临时量的活跃维度从 (R\times V) 限制到约 (C\times V)，总计算量仍随 (R\times V) 增长。

**理论/文献依据。** Wijmans 等人的 Cut Cross-Entropy 通过词表维分块/融合，避免完整物化大词表交叉熵中间量，提供了工程先例；REAL-Q 的 teacher-student KL 需要维持完整分布，不能直接照搬 CCE。当前实现采用 token 行分块与反向重算，保持 KL 定义和梯度语义。相关来源：[Cut Your Losses in Large-Vocabulary Language Models](https://arxiv.org/abs/2411.09009)；项目背景与局限见 [峰值显存盘点](peak_memory_inventory_literature_20261004.md)。

**方法/实现。** `--kl_logit_chunk_tokens 16` 分块计算原 PyTorch KL 表达式，并在反向逐块重算以取得 student-logit 梯度；teacher 不接收梯度，全词表/现有 top-k support 保留。默认 `0` 仍走原路径。实现和测试位于 `utils/loss_utils.py`、`gptq_utils/gptq_plus_utils.py`、`tests/test_chunked_kl.py` 与 `tests/test_chunked_refresh_loss.py`。

**实验。** WSL Ubuntu 24.04、RTX 4060 Laptop 8 GiB、Qwen3-0.6B、W4、32×256 校准、32 backward samples、完整 28 层、8×1024 评估。3 组配对都得到相同的 507-tensor state SHA `96e6281619f2304fb97c885962fcbae9e7dea8fe0e3f256aa225fc41201a8d4e`、KL `0.23623386025428772`、PPL `33.279949188232422`。每组 allocated 从 5,926,841,856 B 降至 2,774,529,024 B（−53.2%）；reserved 从 7,465,861,120 B 降至 3,418,357,760 B（−54.2%）。峰值从 layer-27 KL refresh 移到 Stage-0 Fisher backward，约 2.77 GiB。

**速度结果与限制。** 多次时间受运行条件/源码快照差异影响，配对量化时间比中位数约 0.976，端到端时间比中位数约 0.980；因此没有加速主张。chunk16 降低峰值临时量，但仍保留完整 BF16 logits 与部分梯度缓存。不能通过继续减小 chunk 绕过已经转为全程峰值的 Stage-0。

**评价。** 这是当前最稳健的工程显存结果：完整模型、多组重复、质量完全相同，且全程 allocated/reserved 均有明显下降。证据仍限于该本地配置；尚未完成论文 2048 样本设置及多模型规模的同协议重复，也没有高频 NVML 数据证明整卡 device-used 的每一项峰值都同样下降。

**反思/下一步。** 先把它作为系统基线保留，再以相同流程测 1.7B 及论文规模。追查峰值由 Stage-0 导致后的可扩展性瓶颈，优先优化模型参数常驻、保存激活与 Fisher 累积的生命周期，而不是继续缩小 KL chunk。原始数据：`outputs/kl16-final_paired_repetitions.json`；联合正确性/峰值审计：`outputs/combined-final-layer-audit-c-20261002-campaign/paired_repetitions.json`。

### B. Triton 融合 GPTQ 列内循环：当前最强速度结果

**瓶颈。** Nsight 在 `layers.0.mlp.down_proj` 的代表性 128 列范围观察到 2,556 次 CUDA kernel launch（约每列 20 次）；host launch API 时间约 19.6–28.9 ms，而该范围设备 kernel 累计约 5.48 ms，显示逐列 PyTorch 调度开销明显。

**理论/文献依据。** 将多个细粒度运算融合到单个 GPU kernel 可减少 launch 与中间张量往返。Triton 官方 kernel 教程/benchmark 文档可作为实现与测量的系统参考：[Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/)。Nsight profile 见 [GPTQ 列内核分析](gptq_inner_kernel_profile_20260929.md)。

**方法/实现。** 在 opt-in 的受支持路径中，将量化/反量化、误差计算、剩余权重二阶补偿及 `GHinv` rank-one 更新融合；列顺序与 block 结束后的 REAL-Q refresh 不变。通过独立输入/输出行步长、严格除法和关闭 FP fusion 对齐原 FP32 顺序。默认仍是 PyTorch。实现：`gptq_utils/triton_gptq_kernels.py`；验收驱动：`tools/run_triton_gptq_reproducibility.py`。

**实验。** 同一源码快照、Qwen3-0.6B/W4、完整 28 层、无可见 CUDA 计算进程下做了 3 组完整配对。每组 507 张量 state SHA、KL `0.23623386025428772`、PPL `33.27994918823242` 完全一致。control 量化时间为 699.70–725.14 s，Triton 为 333.16–335.19 s；配对量化加速比中位数 2.143×、端到端中位数 2.032×。峰值 allocated 只差 4,096 B，reserved 相同。r4 profile 的列内环累计时间约 417.432→23.431 s，真实梯度刷新约 194.479 vs 187.798 s，支持加速来自融合而不是跳过 refresh。

**评价。** 在本地 RTX 4060、当前特定配置下，加速幅度大且三次复现；它是速度贡献，不应算作显存贡献。适用边界包括 `block_gd`、对称逐行低位宽量化、FP32 内部权重、block size ≤128、非 atomic、无 two-sided rounding、无并行分组等；其他配置应显式拒绝或先验证。两组/额外 seed 的 microbenchmark 也显示代表形状逐位一致，但不替代端到端数字。

**反思/下一步。** 可作为同一系统论文的次级贡献；优先扩展正式量化配置和更多模型/卡型，记录候选平均 device-used 以及 reserved 峰值。不能把它宣传成“省显存”；本地两组 profiling 甚至观察到候选运行时平均 device memory 较高，但 PyTorch allocator 峰值不变。原始配对：`outputs/triton-uncontended-20261002-campaign/paired_repetitions.json`；正式配置脚本说明：`triton_paper_server_benchmark.md`。

### C. KL 分块与 Triton 联合使用：正确性已审，耗时未测

**方法。** `triton_fused` GPTQ 内循环与 chunk16 KL refresh 组合。逐层输出 audit 记录 28/28 层、每层 8 次 forward，shape、dtype、调用数和 SHA 指纹全部相同；state SHA、KL、PPL 相同。联合 allocated 由 5,913,079,296 B 降至 2,731,209,728 B（−53.8%），reserved 由 6,184,501,248 B 降至 2,868,903,936 B（−53.6%）。

**评价/反思。** 这是联合正确性与显存结果；该次带输出 trace hook，不能拿 wall time 评估速度。加速证据取自未插桩 Triton 配对，KL 分块没有加速主张。此配置仍需在目标正式设置下复测，并补足高频 device-level telemetry。

### D. 对称 Fisher 的 packed GPU 累积：组件减半，全程收益小

**瓶颈。** Stage-0 暂存每层对称 Fisher 的 FP32 GPU 累积器。完整矩阵用 (d^2) 元素，packed 下三角用 (d(d+1)/2) 元素；本地代码最终仍重建 full matrix，并转成 CPU BF16 cache。它优化的是临时 GPU 累积器，而非论文理论中的长期 BF16 全矩阵 GPU cache。

**依据。** 利用 Fisher 对称性仅存一份三角区域是无损存储布局优化；不近似或删除 cross-channel 项。研究背景与实际 scaling profile 见 [Fisher scaling 记录](realq_memory_scaling_packed_fisher_20261005.md)。

**实现。** `--static_fisher_packed_accumulation` 使用 compact vector 做 all-reduce，之后重建矩阵并沿用旧 cache/loss API；默认关闭。实现：`gptq_utils/fisher_storage.py` 和 Stage-0 collector。

**实验及评价。** 理论上 0.6B 临时 accumulator 112→56.055 MiB，1.7B 448→224.219 MiB，组件约减半。缩减配置全模型测试逐层输出/state/KL/PPL一致：0.6B 32×256 单配对 allocated −1.20%、reserved +0.08%；组合 memory run allocated −2.28%、reserved −5.39%。1.7B 8×64 full-28 单配对 allocated −3.72%、reserved −0.86%，量化时间 +0.67%，state/KL/PPL一致。没有重复速度证据。结果说明组件确实减半，但模型常驻和反向激活占大头，全程收益只有低个位数。

**反思/下一步。** 这是准确、干净的子组件工程优化，但单独作为论文主线力度有限；应与能削减主峰的措施联合。不要把“Fisher accumulator 减半”写成“模型显存减半”。原始数据：`outputs/local-fisher-packed-32x256-retry1-campaign/`、`outputs/fisher-scale-qwen3-17b-full8x64-20261005-campaign/`。

### E. Stage-0 autograd saved-tensor CPU offload：有条件有效，短序列全程不达标

**依据。** Activation checkpointing/rematerialization 以重算换显存，见 Chen et al., [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) 与 Jain et al., [Checkmate](https://arxiv.org/abs/1910.02653)。CPU offload 可参考 PyTorch [saved tensor hooks](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)；Capuchin 的 eviction/prefetch/recompute 是相关系统思路（[paper](https://www.microsoft.com/en-us/research/?p=690597)）。这些是设计依据，不保证此处收益。

**实现。** `--static_saved_tensor_offload_mib` 在 Stage-0 forward 注册 hooks，将达到阈值的 autograd-saved CUDA tensor 复制到 CPU，需要 backward 时再恢复。默认关闭。当前主要是同步 pageable CPU 搬运。

**实验。** 0.6B、32×256、full-28：候选搬运 14,400 个 tensors、累计 25.00 GiB payload；所有层指纹/state/KL/PPL相同，但全程 allocated 持平 5,913,079,296 B，reserved 增加约 100 MiB；量化时间 +6.19%。25 GiB 是全程累计搬运，不是同时驻留显存。另一个 seq-2048 collector-only profile 的 Stage-0 allocated 峰值下降 35.72%，但未覆盖完整量化。1.7B、8×128 full-28：Stage-0 峰值下降 5.7%，峰值转到 layer-27 quantization；全程 allocated 仅 −3.02%，reserved +0.90%，时间 +1.26%。另一组 1.7B 8×128 截至 layer0 的 packed+offload pilot 全程 allocated −4.27%、reserved −2.14%，但不是 full-28。

**评价。** 它在 Stage-0 本地有效，却在多个短序列完整配置中未通过严格的全程峰值验收，还带来搬运成本。要点是峰值迁移：Stage-0 节省的显存可能被后续 Hessian/量化峰值抵消。

**反思/后续。** 保持 default-off；不要继续扫 offload threshold 来凑结果。只有正式配置 profile 证明 Stage-0 是主峰时，才研究选择性 offload、pinned-memory 异步搬运/prefetch 或 checkpoint/recompute，并同时计 CPU RAM、搬运量和耗时。若 Stage-1 更高，则先处理 Stage-1。文档：[Stage-0 offload 研究](realq_stage0_saved_tensor_offload_20261004.md)。

### F. Fisher 标签 softmax/sample 分块：微基准有效，但量化随机路径变化

**实现/结果。** `--static_fisher_label_chunk_tokens` 分块执行类别概率 softmax/categorical sampling。CUDA 单算子三次中，峰值从 388,963,328 B 降至 97,256,448 B，中位耗时 9.203→4.131 ms。但分块改变 PyTorch RNG 调用形状，抽取的标签和完整量化 state 可能变化；只有一次完整配置质量结果，没有跨种子复现。

**评价/反思。** 该微基准不能证明全流程显存下降，也不能当成数值等价工程优化。若保留，应把它作为随机算法配置单独验证 KL/PPL 方差，并与相同随机预算的 control 配对；当前不纳入主结果。测试和总体边界见 `docs/thesis_final_acceleration_memory_optimization.md`。

### G. Hessian 对称化临时张量减量：有代码候选，缺 GPU 验证

**瓶颈/改动。** 对 FP32 Hessian 做 `0.5 * (H + H.T)` 时，旧表达式可能同时保留旧 H、加法结果和缩放结果。项目笔记按 Qwen3-1.7B `down_proj` 的 FP32 (Hin R^{4\times6144\times6144}) 估算，单矩阵约 576 MiB，旧表达式理论瞬时存活可到约 1.69 GiB。候选改为一个 `empty_like` 输出、`torch.add(..., out=buffer)`、原地缩放，保留加法/缩放顺序。

**证据。** 多个 CPU shape 的输出 bitwise equal；当前记录明确指出本地 CUDA profile 尚未完成，因此没有实测 GPU 峰值或耗时收益。位置与测量设计见 [Fisher scaling 记录](realq_memory_scaling_packed_fisher_20261005.md)。

**评价/反思。** 这是很有潜力的低风险工程点，尤其是模型变大时，但 CPU 相等不能替代 CUDA 峰值测试。下一步应在 1.7B 先用同 shape profile 确认峰值，再做 full-28；如果被别的更大张量掩盖，不能宣称端到端节省。

### H. 遥测、阶段定位和论文配置 runner：支撑性工程

历史实验修复了 telemetry 文件覆盖冲突、CSV 重复表头/残行解析、组件耗时文本截断解析，以及输出 trace wrapper 名称不一致；完整配对日志现在能归档状态、manifest、源码哈希、阶段峰值、GPU telemetry、component timing 和逐层 fingerprints。它们是研究结果可复现与诊断的必要设施，本身不等于省显存/加速结果。

目前仓库有大量未提交修改与未跟踪实验脚本，且一些旧 profile 的源码哈希不同。所有历史数据必须按来源文件和 manifest 引用，不能把不同快照的结果合并计算。建议在正式复测前固定干净 commit 或不可变 source hash。

### I. FSDP2 CPU offload 用于 Stage-0 Fisher 预计算：端到端峰值下降，单卡试验

**动机与外部依据。** 1.7B profile 显示 base 的最大 allocator 峰值出现在 Stage-0 静态 Fisher 反向，而不是 Hessian 累积；这要求减少反向阶段常驻 GPU 的模型参数。PyTorch FSDP2 的 `fully_shard` 在模块前后 all-gather/free 参数；`CPUOffloadPolicy` 把不参与计算的分片参数放到 CPU，属于已有系统能力，不是新量化算法。[PyTorch FSDP2 文档](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html) 描述其参数分片、计算前聚集和计算后释放；[FSDP2 教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) 说明其使用方式。此处将它用于 REAL-Q Stage-0 的工程适配。

**实现与遇到的问题。** 新增 `--fsdp_precompute --fsdp_cpu_offload --fsdp_meta_init` 的单卡 CPU-offload fallback：在 CPU 上实例化完整模型，再用 FSDP2 CPUOffloadPolicy 包装后执行 Stage-0 预计算，Stage-2 继续使用原普通量化路径。首次试验发现模型 loader 会把根层 embedding、norm、lm_head 留在 CUDA，导致 FSDP 报 “parameters should be materialized on CPU”；修复后 retry2 完成。实现位于 `utils/model_utils.py`。当前 fallback 是单 rank 专用，先完整加载 CPU 模型，未省 CPU RAM；多卡/分布式 meta-init 路径尚未通过验证，不能据此宣称服务器多卡兼容。

**配置与正确性。** 本地 WSL、RTX 4060 Laptop、Qwen3-1.7B、W4A16、WikiText-2 校准 8×64、backward_samples=8、block=128、groups=4、单 rank。regular Stage-0 与 FSDP CPU-offload Stage-0 生成的 cache 通过递归逐 key、逐 tensor 比较，所有 tensor 完全相同（0 个差异，文件均 236,643,375 B）。候选完整 Stage-2 输出 state SHA 为 `881b4510bd03ef24189a111d53b744320f7414e513ab865a270c55177a4e1c5c`，与已有 base full-28 run 相同；WikiText-2 KL/PPL 为 `0.5254853963851929 / 35.92361068725586`，与 base 完全相同。这说明在此配置下 Stage-0 offload 没改变量化输入缓存，也没有改变最终量化状态。

**全流程峰值与代价。** Base Stage-0 peak 为 4,849,651,712 B allocated / 4,884,267,008 B reserved；候选 Stage-0 为 2,425,411,584 / 2,810,183,680 B（局部下降约 50.0% / 42.5%）。但 Stage-2 接替为全流程峰值，候选全流程 max 为 4,669,303,296 / 4,762,632,192 B；对全流程 base max 4,849,651,712 / 4,886,364,160 B，allocated 降 3.72%，reserved 降 2.53%。因此该方案在这一小配置下达到“全程 allocator 峰值低于 base”，但幅度是低个位数，不能说显存减半。候选 Stage-0 耗时约 90.5 s，base Stage-0 约 41.1 s，offload 带来约 2.2× 阶段耗时；Stage-2 候选约 1,827 s，历史 base 约 1,858 s，但非同日配对计时，不能宣称速度提升。主机 RAM 与 CPU 搬运成本都应纳入部署评估。

**遥测限制。** Stage-2 有 100 ms `nvidia-smi` 外部采样，采样段峰值 5,299 MiB；它没有覆盖 Stage-2 启动前约 5 分钟，也没有同条件 base 外部采样。CUDA `mem_get_info` 边界快照的 base/candidate 最大值分别约 6.823/6.581 GB，与这段 NVML 采样的绝对值不一致。allocated/reserved 的 PyTorch allocator high-water mark 是可比的，并且 Stage-0 和 Stage-2 分开记录；整卡 device-used 暂不据此给出严格下降百分比。后续正式验收需要 base 与 candidate 从进程启动到退出全程同时采集 NVML/allocator 数据，并记录峰值时相位、rank 和进程。

**评价与下一步。** 这是目前最贴近“模型规模增大时减少 Stage-0 GPU 常驻参数”的可行方向，但当前只证明 1.7B、8×64 的单卡小配置和 allocator 低个位数端到端下降。Stage-0 本地完整 CPU model 在 1.7B 进程峰值约需 11.5 GB 主机 RAM；更大模型会增加主机 RAM 压力。接下来先用论文校准规模和至少一个更大模型验证 Stage-0 与后续 Stage-2 峰值迁移；严格维持 candidate allocated/reserved 全程都低于 base，否则记为未通过。多卡实现与性能配对必须单独验证。原始记录：`outputs/memory_profile_base-qwen3-17b-8x64-stage0-20261007/`、`outputs/memory_profile_fsdp-cpuoffload-qwen3-17b-8x64-stage0-retry2-20261007/`、`outputs/phase_profile_fsdp-cpuoffload-qwen3-17b-8x64-stage2-20261007/`；与既有 base full run 比对：`outputs/fisher-scale-qwen3-17b-full8x64-20261005-campaign/summary.json`。

**Stage-2 CPU-master 负向消融。** 为判断全模型 CPU 主副本能否继续压低迁移后的 Stage-2 峰值，使用相同 FSDP Stage-0 cache、参数和 eval 再跑一次 `--stage2_cpu_master`；实现由 `tools/run_stage2_cpu_master_memory_profile.py` 留下可重跑入口。完整 run 成功、源码哈希不变，state SHA 与普通 Stage-2 候选相同，KL/PPL 仍为 `0.5254853963851929 / 35.92361068725586`。峰值 allocated 与普通候选都为 `4,669,303,296 B`；reserved 为 `4,758,437,888 B`（CPU-master）和普通候选 `4,762,632,192 B`，只有约 4 MiB 的差异；NVML 100 ms 全程采样 CPU-master 最大 5,295 MiB，普通候选所采样区间最大 5,299 MiB，但普通 run 的 NVML 采样起始晚了约 5 分钟，故这组 NVML 数字不构成完整配对结论。CPU-master 没有降低 allocator 全程峰值，说明本配置的峰值被单层 Hessian/量化工作区主导；不把它列为有效增量。原始目录：`outputs/phase_profile_fsdp-stage2-cpumaster-qwen3-17b-8x64-20261007/`。

**Stage-2 延迟创建 Hessian 备用单位阵：正确性通过、未测出全程峰值收益。** 审查 `_compute_hessian_inverse_batched_with_fallback` 后发现，正常 Cholesky 成功路径也会提前构造只在 damping fallback 才用到的 identity；将 identity 创建移入 fallback 分支。这里实际单个 `down_proj` Hessian 是 6144×6144 FP32，单位阵大小为 150,994,944 B（144 MiB），不是 16 MiB；这是对前一版本记录中维度解释错误的更正。新增正常路径“不调用 `torch.eye`”及失败回退“只创建一次且结果为 identity”的检查；pytest 在 WSL 项目环境未安装，改用项目 venv 的直接函数调用验证，两个分支都通过。随后以同一份 FSDP Stage-0 cache 完整重跑 Qwen3-1.7B 28 层 Stage-2，8×64、W4A16、相同量化及评估配置：`returncode=0`、源码哈希未变，28 层输出指纹 JSON 与上一条普通候选逐项相等，state SHA 仍为 `881b4510bd03ef24189a111d53b744320f7414e513ab865a270c55177a4e1c5c`，KL/PPL 为 `0.5254853963851929 / 35.92361068725586`。峰值 allocated/reserved 为 `4,669,303,296 / 4,762,632,192 B`，与普通 Stage-2 候选相同；NVML 100 ms 全程 17,736 个采样点，峰值 5,299 MiB。该结果说明移除备用单位阵没有降低这个配置的全程 high-water mark，原因待结合峰值同相位的张量存活情况解释；不将它算作显存收益。完整候选流程仍使用已验证的 FSDP Stage-0 offload，其 allocator 全程峰值低于记录的 base（allocated 约 3.72%、reserved 约 2.53%）；这部分收益来自 Stage-0 offload。原始记录：`outputs/phase_profile_fsdp-lazy-eye-qwen3-17b-8x64-20261007/`。

**Stage-2 跳过未使用的完整 Hessian 逆矩阵：正确性通过，全程 allocator 峰值下降约 9.7%。** 峰值日志显示 `down_proj` 每个输出行组的 Hessian 为 `[6144, 6144]`，完整矩阵为 144 MiB。当前 `alpha=0` 时，`_compute_gradient_terms` 明确不消费 `Hinv_init`，但旧代码仍将完整 `Hinv_init` 作为每个 subgroup 的长期 GPU 状态保存；四组随 subgroup 处理而累积。候选只在 `enable_gradient_update and alpha > 0` 时返回/保存 `Hinv_init`，否则仍原样计算并保存 `Hinv`，Adam 的 block-GD 梯度更新路径不依赖这个被省略的矩阵。此优化来自逐调用点的 liveness 审计，属于算法不变的工程内存优化。

**配置与正确性。** WSL Ubuntu 24.04、RTX 4060 Laptop 8 GiB、Qwen3-1.7B、W4A16、WikiText-2 8×64 校准、8 backward samples、block=128、4 groups、完整 28 层，最终评估 2×128。候选完整运行 `returncode=0`、`sources_unchanged=true`；507-tensor state SHA `881b4510bd03ef24189a111d53b744320f7414e513ab865a270c55177a4e1c5c`；KL/PPL `0.5254853963851929 / 35.92361068725586`。28 层输出指纹逐项与原 base Stage-2 和上一条 lazy-identity run 相等，指纹 JSON SHA256 `969b54cda33ecefafc64229c19914fa72cfce0eecdb01e90b2b8f2fd05757a65`。

**全程峰值与代价。** 与同模型/同小配置 FSDP Stage-0 的 base Stage-2 记录相比，PyTorch `max_memory_allocated` 从 4,669,303,296 B 降到 4,216,318,464 B（−9.70%），`max_memory_reserved` 从 4,762,632,192 B 降到 4,303,355,904 B（−9.64%）。重新按 `nvidia-smi` CSV 的真实列序（`memory.used` 在 `memory.free` 前）核算：候选峰值 4,861 MiB、均值 3,155 MiB；base 峰值 5,299 MiB、均值 3,293 MiB，因此候选采样峰值低 8.3%、均值低 4.2%。这组日志的 100 ms 采样点覆盖完整 Stage-2 进程；均值仍会受桌面/其它进程波动影响。量化时间 1,789.4 s，旧 base 记录 1,827.4 s，但不是同期配对，不作加速主张。此前把 CSV 的空闲显存列误当成已用显存，故误报“峰值持平、均值较高”；本段已更正。原始候选：`outputs/phase_profile_fsdp-no-unused-hinvinit-qwen3-17b-8x64-20261007/`；对照：`outputs/phase_profile_fsdp-cpuoffload-qwen3-17b-8x64-stage2-20261007/`。

**评价与后续。** 这是算法不变、可独立复用的持久状态生命周期优化；已进一步测试 Hinv CPU 行暂存，完整配对结果见下一节。当前两项均只在 alpha=0 的配置成立。正式论文结论仍需扩大校准规模和模型覆盖，并维持 candidate 全程峰值不高于 base。

**Stage-2 Hinv 按 block 行暂存到 CPU：完整正确性通过，显存下降、速度略降。**

**依据与方法。** ZeRO-Offload/ZeRO-Infinity 为训练状态提供 CPU/GPU 分层存放的系统先例；PyTorch 的 pinned-memory 文档提醒，主线程手动 pinning 本身是阻塞操作。因此首个实现采用简单同步 CPU↔GPU 搬运，不增加异步生命周期风险。仅针对 `alpha=0` 的 REAL-Q block-GD 路径：每个 subgroup 的 `Hinv` 在 GPU 上照常计算，随后将完整 FP32 因子放入 CPU RAM；每个量化 block 仅将当前 `[blocksize, columns]` 行切片搬回 GPU，inner/outer GPTQ 运算仍按原算子和顺序执行。新参数 `--stage2_hinv_cpu_offload` 默认关闭，并对 alpha、更新模式、分组并行、Triton 和 preview 路径做显式限制。它不近似 Hessian，也不减少累计传输量，只缩短 GPU 上 Hinv 的同时驻留时间。

**配置、正确性与资源。** WSL Ubuntu 24.04、RTX 4060 Laptop 8 GiB、Qwen3-1.7B、W4A16、WikiText-2 8×64、8 backward samples、block=128、4 groups、完整 28 层；同一 Stage-0 cache、同一 eval 配置。候选 `returncode=0`、`sources_unchanged=true`；507-tensor state SHA 与 base 完全相同：`881b4510bd03ef24189a111d53b744320f7414e513ab865a270c55177a4e1c5c`；最终 KL/PPL 仍是 `0.5254853963851929 / 35.92361068725586`。28 层输出 fingerprints 与 base、跳过 `Hinv_init` 候选逐项相同，SHA256 为 `94b65cbf7c348c715a79571c98ae3c4bc84c92c84d23af36805071cbcf1de09e`。相对 base Stage-2，allocated `4,669,303,296 → 3,760,822,272 B`（−19.46%），reserved `4,762,632,192 → 3,846,176,768 B`（−19.24%）；NVML 整卡 used peak `5,299 → 4,425 MiB`（−16.49%），采样均值 `3,293 → 3,072 MiB`（−6.71%）。所以这次 allocator 与 NVML 采样峰值都低于 base。Hinv factor 每个 down_proj subgroup 604 MiB，主机进程 RSS 5 秒采样峰值 10.40 GiB；WSL 可用 RAM 最低仍约 16 GiB，未用 swap。每模块 H2D 切片累计合计 28.19 GB（26.25 GiB），日志测得搬运 wall time 合计 46.62 s；完整量化时间 1,910.67 s。相比旧 base 1,827.37 s 单次历史值慢 4.56%，而此时间并非同期配对，不能作为精确回归结论。

**评价与后续。** 当前 1.7B pilot 同时跨过“候选显存不高于 base”和“输出完全一致”两道门槛；省下的 GPU factor 驻留以约 10.4 GiB CPU RSS 和 26.25 GiB 累计 PCIe 搬运为代价，速度可能略降。CPU RSS 是主机容量成本，不能忽略，但本机此次仍有充足可用 RAM。先在相同源码快照下跑不开该 flag 的 control，再决定是否做重复配对；随后需在 0.6B 及更高校准规模确认收益随 Hessian 输入维度增长，并验证多卡路径。未完成同源码 paired repeat 前，不把单次时间差写成正式性能结论。原始候选：`outputs/phase_profile_fsdp-hinv-cpurow-qwen3-17b-8x64-20261007/`（含 `cpu_rss.csv`、NVML、phase profile、manifest 和逐层 fingerprints）；base：`outputs/phase_profile_fsdp-cpuoffload-qwen3-17b-8x64-stage2-20261007/`。

**同源码配对 control（补充）。** 之后在完全相同的 source hashes、runner、Stage-0 cache、量化/eval 参数下运行了关闭 `--stage2_hinv_cpu_offload` 的 control。Control 和候选 state SHA 都是 `881b4510bd03ef24189a111d53b744320f7414e513ab865a270c55177a4e1c5c`；逐层 fingerprint JSON SHA 都是 `94b65cbf7c348c715a79571c98ae3c4bc84c92c84d23af36805071cbcf1de09e`；KL/PPL 均为 `0.5254853963851929 / 35.92361068725586`。同配对下 allocator 峰值 allocated `4,103,072,256 → 3,760,822,272 B`（−8.35%），reserved `4,303,355,904 → 3,846,176,768 B`（−10.62%）；100 ms NVML used 峰值 `4,861 → 4,425 MiB`（−8.97%），样本均值 `3,137.67 → 3,063.68 MiB`（−2.36%）。量化时间 `1,810.61 → 1,910.67 s`（候选慢 5.53%），wall `1,872.77 → 1,973.41 s`（慢 5.37%）；CPU RSS 峰值约 `10.38 → 10.39 GiB`，相近。仅一组配对，显存方向明确、正确性精确、速度有可见代价；必须至少重复三次才能定量总结速度。全流程端到端峰值仍受 Stage-0 外的阶段控制，当前这组只覆盖 Stage-2，需结合已有同源 Stage-0/Stage-2 两段 allocator profile 判断 whole-run allocator max。

**Hessian CPU 暂存候选的同源码 control（2026-10-08）。** 新增 `--stage2_hessian_cpu_offload` 后，在同一 source hash/runner、同一 1.7B 8×64 profile 与 cache 上重跑了 no-offload control；候选和 control 的 5 个实际代码 source hashes 与 runner SHA 完全一致（entry manifest 仅 tag/path 不同）。Control 与 Hessian 暂存候选 state SHA、逐层 fingerprint SHA、KL/PPL 逐项相同，均为上述值。Control allocator 峰值 `4,216,318,464 / 4,303,355,904 B`，候选 `3,612,338,688 / 3,716,153,344 B`，下降 `14.33% / 13.69%`；NVML used 峰值 `4,861 → 4,301 MiB`（−11.52%），均值 `3,148.74 → 3,009.33 MiB`（−4.43%）。CPU RSS 样本峰值约 `10.63 → 10.38 GiB`。量化时间 control `2,076.70 s`、candidate `1,796.60 s`，差 13.5%；但单次同配对出现异常大的时间差，且 NVML 平均占用/利用率条件有波动，只作为诊断数据，不声称加速。阶段日志显示每个模块搬运一个完整 grouped Hessian 到主机、再按输出行组搬回；Qwen3-1.7B `mlp.down_proj` 的完整 grouped Hessian 是 `603,979,776 B`（576 MiB），D2H+H2D 约 0.17–0.2 s/模块。此组满足当前 Stage-2 峰值不高于 control 和数值完全一致，但仍只有一次配对；candidate/control repeats 继续进行。原始目录：`outputs/phase_profile_fsdp-hessian-cpurow-qwen3-17b-8x64-20261007/` 与 `outputs/phase_profile_fsdp-hessian-cpurow-control-v2-qwen3-17b-8x64-20261008/`。

**Hessian 暂存重复配对 2。** 交换运行顺序（control 先、candidate 后）再次跑同一组量化。各方 5 个代码文件 hash 与 runner 完全相同；state SHA、逐层指纹 SHA、KL/PPL 完全相同。峰值 allocated/reserved 与第一组相同；NVML max used 仍为 `4,861 → 4,301 MiB`，mean used `3,132.74 → 2,998.92 MiB`（−4.27%）。CPU RSS 峰值 `10.38 → 10.36 GiB`。量化时间 `1,796.26 → 1,810.00 s`，本配对候选慢 0.76%；第一组因 control 时间异常而显示候选快 13.5%，因此两组时间波动大，尚不能提出速度结论。两组都在低端 allocator/NVML 峰值上给出同方向且幅度一致的收益，正确性也逐位复现。第二组 artifacts：`outputs/phase_profile_fsdp-hessian-cpurow-control-p2-qwen3-17b-8x64-20261008/`、`outputs/phase_profile_fsdp-hessian-cpurow-candidate-p2-qwen3-17b-8x64-20261008/`。仍需第三组和另一个模型/正式校准配置，才能判断可推广性。

**Hessian 暂存重复配对 3（2026-10-08）。** control→candidate 顺序；完整 28 层成功，源文件未改变。两边 state SHA 均为 `881b4510bd03ef24189a111d53b744320f7414e513ab865a270c55177a4e1c5c`，KL/PPL 均为 `0.5254853963851929 / 35.92361068725586`，layer-output fingerprint SHA 与前两组相同。allocated/reserved 峰值仍为 `4,216,318,464 / 4,303,355,904 → 3,612,338,688 / 3,716,153,344 B`；NVML sampled max used `4,861 → 4,301 MiB`；本次平均 used `3,140.13 → 3,017.83 MiB`。这三组合并后，NVML 平均 used 约 `3,140.54 → 3,008.69 MiB`（−4.20%），但这是运行时采样整卡占用，不能代替 allocator 的逐 phase 峰值。量化时间 `1,818.58 → 1,825.31 s`（候选慢 0.37%），wall `1,883.77 → 1,890.09 s`（慢 0.34%）。三组的 median quant 时间 `1,818.58 → 1,810.00 s`（候选快 0.47%），median wall `1,883.77 → 1,884.32 s`（候选慢 0.03%）；速度变化都小于 1%，当前判为无实质变化。逐组时间范围较宽，可能受 GPU/CPU 频率与系统波动影响，不提出速度收益。第三组 artifacts：`outputs/phase_profile_fsdp-hessian-cpurow-control-p3-qwen3-17b-8x64-20261008/`、`outputs/phase_profile_fsdp-hessian-cpurow-candidate-p3-qwen3-17b-8x64-20261008/`。该 1.7B 8×64 pilot 的 Hessian CPU staging 现在有三组同源正确性和显存复现；正式论文规模、多模型和端到端独立 Stage-0 峰值仍未覆盖。

### H. Hessian + Hinv 分层暂存：降低全程峰值，付出约 5–7% 量化时间

**思路与适用边界。** Hessian CPU staging 三次精确复现后，profile 的剩余高点仍包含 GPU 上的 Hinv subgroup 因子。Hinv-only 已显示 0.45 GiB 左右的 peak reduction，但有约 5.5% 单次量化时间代价。将两个互不近似的只读矩阵都按 liveness 放到主机：H 在完成累积后 D2H，逐输出组 H2D；Hinv 照原 GPU 算法求出后留在 CPU，每个量化 block 只搬当前行片。两种 flag 都是 opt-in，保留默认路径；仍只支持当前已验证的 `alpha=0`、block-GD、无 group-parallel、PyTorch inner kernel、无 shadow preview 配置。依据是 ZeRO-Offload/ZeRO-Infinity 的分层状态放置先例；这里借鉴的是系统放置与按需暂存，不复用它们的训练算法，也不改变 REAL-Q 数学计算。

**正确性与同源码 Stage-2 对照。** WSL Ubuntu 24.04、RTX 4060 Laptop 8 GiB、Qwen3-1.7B、W4A16、WikiText-2 8×64、完整 28 层，固定 Stage-0 Fisher cache。当前源码 5 个核心文件 SHA 与同源码 control 完全相同；运行结果 `sources_unchanged=true`。两次 H+Hinv candidate 的 507-tensor state SHA 均为 `881b4510bd03ef24189a111d53b744320f7414e513ab865a270c55177a4e1c5c`，28 层输出指纹 SHA 均为 `94b65cbf7c348c715a79571c98ae3c4bc84c92c84d23af36805071cbcf1de09e`；KL/PPL 都与 control 相同：`0.5254853963851929 / 35.92361068725586`。

与 Stage-2 no-offload control（allocated/reserved `4,216,318,464 / 4,303,355,904 B`、NVML max `4,861 MiB`）相比，两次组合 candidate 峰值固定为 allocated/reserved `3,160,982,528 / 3,412,066,304 B`、NVML max `4,011 MiB`，分别降低 25.00%、20.71%、17.49%。相较 Hessian-only 三组峰值，组合再降低 allocated `451,356,160 B`（约 430 MiB）、reserved `304,467,040 B`（约 290 MiB）、NVML max 290 MiB。每次 Hinv H2D 累计约 `28.19 GB`，日志计时 31.08/33.95 s；一次外部 RSS 监测的进程峰值为 10.41 GiB，WSL 主机未用 swap。两次 candidate 量化耗时 `1,899.49 / 1,939.71 s`，同配置 control `1,818.58 s`；相对慢 4.45%/6.66%。因此这是显存优先配置，速度有真实代价；只有显存容量受限时才值得启用，不能称为全面胜过 base。原始目录：`outputs/phase_profile_fsdp-hessian-hinv-combined-qwen3-17b-8x64-20261008/` 和 `outputs/phase_profile_fsdp-hessian-hinv-combined-repeat1-qwen3-17b-8x64-20261008/`。

**当前源码全程峰值拼接核对。** 为确认“任一阶段都不超过 base”，另用当前源码和不同的全新 cache 路径重跑 Stage-0 control/offload precompute，并记录 100 ms NVML。两条 Fisher cache 文件大小都是 `236,643,375 B`，SHA256 完全相同：`FEDD810F447E6EB4B6028E5C087778A9908C537069A644DD99EDC2F792A9C78F`，证明 CPU-offload Stage-0 与 base 产出的缓存逐字节相同；两条 Stage-0 的核心源码 hashes 与 Stage-2 runs 也相同。Stage-0 base 峰值 allocated/reserved `4,849,651,712 / 4,884,267,008 B`、NVML max `5,281 MiB`；FSDP CPU-offload Stage-0 为 `2,425,411,584 / 2,810,183,680 B`、NVML max `3,303 MiB`。Stage-2 base 的峰值是 `4,216,318,464 / 4,303,355,904 B`，故 base 全流程最高点在 Stage-0；组合 candidate 的 Stage-2 峰值 `3,160,982,528 / 3,412,066,304 B` 高于其 Stage-0 的 `2,425,411,584 / 2,810,183,680 B`，故候选全流程最高点在 Stage-2。按这两个实际 phase 记录取全程最大值，allocated 从 `4,849,651,712` 降至 `3,160,982,528 B`（−34.82%），reserved 从 `4,884,267,008` 降至 `3,412,066,304 B`（−30.14%），NVML max 从 `5,281` 降至 `4,011 MiB`（−24.05%）。这是跨两个串行阶段的 profile maximum，不是同一 PID 的单次端到端 telemetry；但两阶段用同一源码、同模型/样本、逐字节相同缓存，且候选每一阶段各自低于对应 base 阶段，满足当前配置下“没有阶段峰值高于 base”的验收条件。原始 Stage-0 目录：`outputs/memory_profile_fsdp-stage0-current-base-freshcache-retry2-qwen3-17b-8x64-20261008/`、`outputs/memory_profile_fsdp-stage0-current-offload-freshcache-qwen3-17b-8x64-20261008/`；100 ms NVML CSV 在同级 `outputs/stage0-nvml-*.csv`。

**实验排除项与下一步。** 一次 Stage-0 base 重跑不小心沿用了已有 Fisher cache，profile 只记录 cache load，allocated 约 1 KiB；它不是 Stage-0 计算结果，已排除。后续改用全新 cache 路径重跑并核对 cache SHA。当前组合证据是 1.7B、8×64 pilot，而不是论文 2048 校准样本配置；论文结论前必须在正式样本量/多卡配置复验。下一步优先做更大样本与另一模型的 matched full-phase test，再用 profile 记录“峰值位置是否转移”。若显存容量允许，Hessian-only 速度更稳；组合 H+Hinv 仅作为低显存机器的部署选项。

## 失败与排除的实验记录

- **2026-10-03 服务器 W4 per-row/2048 配置 campaign。** 用户粘贴的 `...paper2048-perrow-03f0e48` 日志显示 control 与 Triton 的完整 state SHA、KL、PPL 不一致；后续 layer-0 fingerprint 也不同。runner 按安全条件停止后续配对。它是有效的“候选实现尚未跨环境复现”的证据，不是加速/显存数据。此前本机某些配置输出一致，不能代替服务器目标配置的一致性。
- **服务器早期多次失败的 campaign。** 日志出现子进程 `returncode=1`、`profile_summary_present=false`，因此没有完整 metrics；不能用于性能比较。失败日志和诊断应保留，但根因未被同等配置的有效运行证明时，不应将失败归结为显存优化有效或无效。
- **2026-10-05 `local-memory-combo-32x256-retry4`。** 因 `utils/loss_utils.py` 在运行期间改变、`sources_unchanged=false`，从正式 paired 证据中排除；随后同配置的 clean rerun 才被记入上文。
- **Stage-0 offload 的 no-op 尝试。** 1.7B 64-token 配置日志显示 offload tensor 数为 0；即使命令启用了参数，也没有实际搬运，故不视为 offload 实验结果。后续改用 128-token 运行才确认 hook 生效。
- **Triton 与 KL chunk 的时间异常。** 部分早期单次 A/B 遇到 GPU 竞争或不同源码快照，未纳入速度结论。有效 Triton 时间结论来自无可见 CUDA 进程且三次配对的复测；KL chunk 的时间波动仍只作限制记录。

## 服务器实验摘要（证据待归档）

对话中曾粘贴 `realq-memory-speed-server-20261004-041835` 的三组 campaign 摘要：`exact_state_kl_ppl_all_pairs=true`，`peak_allocated_and_reserved_never_above_control=true`，量化 speedup mean 1.0353（量化时间约降 3.35%），端到端 speedup mean 1.0413，mean device memory reduction 8.91%，CUDA allocated reduction 19.72%，reserved reduction 9.63%。用户还粘贴了 `status.json` 显示 `valid: true`。

**证据等级：暂定。** 该 campaign 的原始 `summary.json`、paired repetitions、配置 manifest、每次 run 日志及源码快照目前没有在本地工作区中找到；因此此处只记录为对话提供的结果，不并入本地已复核均值，也不确认它与论文 2048 样本/per-row 配置一致。若诊断包仍可取得，应补充环境、模型/样本配置、逐 pair 峰值和源哈希后再决定是否纳入论文结果。

## 文献路线与后续优先级

已整理的外部系统文献包括：Chen et al. 的 activation checkpointing（重算换显存）；Checkmate 的内存约束下重算计划；Capuchin 的 eviction/prefetch/recompute；Cut Cross-Entropy 的大词表 loss 中间量分块；FlashAttention 的 IO-aware attention tiling；PyTorch saved-tensor hooks/checkpoint API；Triton kernel 编写与 benchmark 文档。适用边界和 bibliographic details 见 `docs/peak_memory_inventory_literature_20261004.md`、`docs/realq_stage0_saved_tensor_offload_20261004.md` 与 `docs/thesis_final_acceleration_memory_optimization.md`。

下一轮按证据顺序推进：

1. **冻结公平的全程测量。** 同一设备、干净源码和论文配置，从模型加载起对 base/candidate 同时记录 allocator allocated/reserved、NVML device-used、CPU RSS、时间与峰值相位；阶段边界采样不能替代进程全程的外部采样。
2. **放大验证 FSDP2 Stage-0 CPU offload。** 当前 Qwen3-1.7B 小规模单卡 pilot 的 Stage-0 峰值显著下降，但全流程只下降低个位数。先做论文校准规模，再做 1.7B/更大模型及可用的多卡路径；确认 CPU RAM 受控、全流程峰值仍低于 base、cache 和最终 state/KL/PPL 一致后，再把它定为主贡献。
3. **继续优化 Stage-2 真正峰值。** candidate 的峰值从 Stage-0 移到 Stage-2 Hessian/量化阶段；依据 phase profile 测大 `down_proj` 的 (O(n^2)) Hessian、Block-GD workspace、权重/层物化，避免只优化已经不再是全程最大值的组件。Hessian 对称化临时 buffer 先做 CUDA 数值与生命周期核对，再做全模型配对。
4. **将已验证子项迁移到论文设置。** 对 KL chunk16、Triton kernel 和 Fisher packed 累积独立做正式配置复现，再测组合；每项都核对逐层输出、最终 state、KL/PPL 与全程峰值。
5. **选择性使用 checkpoint/recompute 或异步 offload。** 只有峰值张量 trace 指向可重算激活时才引入；用 CPU RAM、PCIe traffic、GPU peak 和 wall time 联合验收，不因局部 Stage-0 峰值下降就算成功。

速度优化是次级线：先保留 Triton 融合列内环的已验证收益，再把 profile 指出的真正瓶颈（Python dispatch、小 kernel launch、workspace materialization）逐个测试。速度和显存实验须使用各自适当的非插桩 benchmark 与诊断 profile，不混用计时。

## 参考文献与技术资料

1. Tianqi Chen, Bing Xu, Chiyuan Zhang, Carlos Guestrin. “Training Deep Nets with Sublinear Memory Cost.” 2016. [arXiv:1604.06174](https://arxiv.org/abs/1604.06174).
2. Paras Jain, Ajay Jain, Aniruddha Nrusimha, Amir Gholami, Pieter Abbeel, Kurt Keutzer, Ion Stoica, and Joseph E. Gonzalez. “Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization.” MLSys 2020. [arXiv:1910.02653](https://arxiv.org/abs/1910.02653).
3. Xuan Peng et al. “Capuchin: Tensor-based GPU Memory Management for Deep Learning.” ASPLOS 2020. [Microsoft Research publication page](https://www.microsoft.com/en-us/research/?p=690597).
4. Erik Wijmans et al. “Cut Your Losses in Large-Vocabulary Language Models.” ICLR 2025. [arXiv:2411.09009](https://arxiv.org/abs/2411.09009).
5. Tri Dao et al. “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.” NeurIPS 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135).
6. PyTorch Contributors. [Hooks for autograd saved tensors](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html); [Activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint).
7. Triton Contributors. [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/).
8. PyTorch Contributors. [FSDP2 `fully_shard` API](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html); [FSDP tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html).
9. REAL-Q论文仅作为被优化系统与问题定义来源，不能计作本研究的外部工程方法参考。具体版本及其 Future Work 见项目根目录论文 PDF。
10. Jie Ren et al. “ZeRO-Offload: Democratizing Billion-Scale Model Training.” USENIX ATC 2021. [USENIX](https://www.usenix.org/conference/atc21/presentation/ren-jie); [arXiv:2101.06840](https://arxiv.org/abs/2101.06840). Used only as a systems precedent for placing inactive state in host memory; its optimizer/gradient offload algorithm is not directly reused.
11. Samyam Rajbhandari et al. “ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning.” SC 2021. [arXiv:2104.07857](https://arxiv.org/abs/2104.07857). Relevant to tiered CPU/GPU state placement and transfer scheduling; it does not establish that repeatedly staging a GPTQ inverse factor is fast.
12. PyTorch Contributors. “A guide on good usage of `non_blocking` and `pin_memory()`.” [Official tutorial](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html). The tutorial notes that manually pinning in the main thread is blocking; initial correctness/performance prototype should use explicit synchronous copies before considering overlap.
13. Erik Wijmans, Brody Huval, Alexander Hertzberg, Vladlen Koltun, and Philipp Krähenbühl. “Cut Your Losses in Large-Vocabulary Language Models.” ICLR 2025. [arXiv:2411.09009](https://arxiv.org/abs/2411.09009). CCE avoids materializing the full token-vocabulary logits for CE by fusing indexed target-logit computation with an online log-sum-exp. The prototype below borrows the memory-lifetime goal, but currently chunks CE over already materialized BF16 head logits; it is not a CCE reimplementation.

### I. Hessian + Hinv 暂存跨模型核验：Qwen3-0.6B 未降低全局峰值

**问题与理论判断。** 1.7B 的 H/Hinv 是明显的 Stage-2 内存项，因此 CPU 分层暂存有机会降低全流程峰值；但这不意味着所有模型都由矩阵状态支配。应按 REAL-Q 各阶段峰值的位置判断，而不能将 1.7B 的收益外推到 0.6B。该判断属于峰值生命周期分析，符合 Chen et al. / Checkmate 的 rematerialization 系统视角与 ZeRO-Offload 的分层存放先例；这些论文不保证本工程配置会有收益。

**配置与复现。** WSL Ubuntu 24.04、RTX 4060 Laptop 8 GiB、Qwen3-0.6B W4A16、WikiText-2 32×256 校准、完整 28 层、8×1024 评估、PyTorch GPTQ kernel。Control 使用同一基础 manifest；候选仅增加 Hessian 与 Hinv CPU 暂存。核心源码哈希相同且 `sources_unchanged=true`。全层输出指纹文件 SHA-256 两边均为 `14592CE9E5AEA6475A2B8E1A4E0F6C13D602FCDA6F48E5CC8292B34BBA54A66F`；507-tensor state SHA 均为 `99589bb56bfe9f3c1bce8381b3d36cdc101e96367aedd9eacb4617fda1404b81`；KL/PPL 均为 `0.23308846354484558 / 32.618465423583984`，因此数值正确性通过。

**显存与速度。** 两边记录的全程 allocator peak allocated/reserved 都是 `2,561,811,968 / 2,604,662,784 B`，没有降低。100 ms NVML `memory.used` 峰值为 `5,881 → 5,897 MiB`，候选高 16 MiB，严格“任何候选峰值不得高于 base”的 device-used 验收也未通过。量化时间为 `1,165.08 → 1,235.02 s`（候选慢 6.0%）；完整 wall 为 `1,228.07 → 1,292.93 s`（候选慢 5.3%）。profile 原始事件把高点定位在 Stage-0 `static_fisher_batch_backward`，不是 Hessian/Hinv；因此 H/Hinv 暂存即使释放 Stage-2 矩阵，也不会降低这个配置的全局高水位。allocator 的逐阶段峰值相同；NVML 的 16 MiB 回升不应夸大解释为算法性增耗，但按预定严格门槛仍记未通过。

**`--offload_inps` 组合核验。** 继续将项目已有的输入 CPU 驻留与 H/Hinv 暂存组合跑完完整 28 层。逐层指纹 SHA 与 control 相同，state SHA、KL/PPL 也完全相同；peak allocated/reserved 仍为 `2,561,811,968 / 2,604,662,784 B`，全程峰值没有下降。NVML max `5,893 MiB`，仍高于 control 的 `5,881 MiB`；量化时间 `1,257.00 s`，比 control 慢 7.9%。它可能降低部分 Stage-2 张量驻留，但没有降低 Stage-0 主导的全程峰值，因此不通过总目标验收。

**评价与反思。** 这是完整、逐层、最终指标都一致的负向跨模型结果。结论是 H/Hinv 暂存只适用于 Hessian/Hinv 主导全程峰值的模型与校准配置；Qwen3-0.6B 32×256 的优先瓶颈转为 Stage-0 Fisher backward。无须继续把 `--offload_inps` 与其他 Stage-2 选项交叉组合来追全程峰值，因为它只改变 Stage-2 输入驻留，不能压低已观测到的 Stage-0 峰值。下一项改为针对 Stage-0 的 saved-tensor offload。

原始数据：`outputs/phase_profile_hessian-hinv-crossmodel-qwen3-06b-32x256-control-20261008/`、`outputs/phase_profile_hessian-hinv-crossmodel-qwen3-06b-32x256-candidate-20261008/`、`outputs/phase_profile_hessian-hinv-inputoffload-qwen3-06b-32x256-20261008/`。

### J. Stage-0 4 MiB saved-tensor offload + H/Hinv 暂存：量化峰值降低，评估峰值接管全程

**目的与参考依据。** 0.6B 的 H/Hinv 负向跨模型结果显示主峰在 Stage-0 Fisher backward。项目已有 `--static_saved_tensor_offload_mib`，利用 PyTorch autograd saved-tensor hooks 将大于门槛的非叶 CUDA saved tensors 暂存 CPU；这是 PyTorch 官方 saved-tensor hooks 的直接系统接口，相关 activation rematerialization/offload 背景见 Chen et al.、Checkmate、Capuchin（文档中的外部文献，不将 REAL-Q 作为优化来源）。这里复用已实现的 4 MiB 配置，不再扫阈值；与 H/Hinv 暂存组合，是为了避免 Stage-0 高点压下后 Stage-2 矩阵又成为新峰值。

**方法与配置。** WSL Ubuntu 24.04、RTX 4060 Laptop 8 GiB、Qwen3-0.6B W4A16、WikiText-2 32×256 校准、完整 28 层、8×1024 评估、PyTorch GPTQ kernel。control 与候选核心代码相同；candidate 加 `--static_saved_tensor_offload_mib 4 --stage2_hessian_cpu_offload --stage2_hinv_cpu_offload`。日志确认 offload hook 实际暂存 1,792 tensors、累计 payload 7.00 GiB（累计流量，不是同时驻留量）。外部 PID RSS sampled maximum `5.68 GiB`；量化时间 `1,234.01 s`，wall `1,288.06 s`。

**正确性。** 28 层全部输出指纹、507-tensor 量化 state SHA、KL/PPL 与 base 完全一致：state `99589bb56bfe9f3c1bce8381b3d36cdc101e96367aedd9eacb4617fda1404b81`；KL/PPL `0.23308846354484558 / 32.618465423583984`。`sources_unchanged=true`。

**显存与当前判断。** 完整运行的 allocator peak allocated/reserved 从 base `2,561,811,968 / 2,604,662,784 B` 降至 `2,433,359,360 / 2,480,930,816 B`（约 `−5.0% / −4.75%`）。但 NVML 100 ms 全调用采样最大值仍为 `5,893 MiB`，而 base 为 `5,881 MiB`；所以严格 device-used 峰值“不高于 base”未通过。峰值采样落在最后 KL/PPL 评估，日志同一时刻显示 `_get_logits` 清理后 CUDA memory 仅 `0.05 GB`；allocator 峰值仪表又只覆盖量化 profile 相位。现有证据表明最终全词表 head/KL 评估路径需要单独 profile，且系统级 used 值可能混有 allocator 外的上下文。候选量化时间较 base `1,165.08 s` 慢约 `5.9%`，完整 wall 较 base `1,228.07 s` 慢约 `4.9%`，因此不能据这单次运行宣称端到端全局显存优化或速度收益。

**评价与反思。** 4 MiB saved-tensor offload 能压低 Fisher backward 计入的 allocator peak，状态与结果精确保持；但将量化阶段峰值降低后，整次实验的 NVML 高点移动到最后评估，主流程最大设备占用没有得到证明。下一步先拆解 `_kl_ppl_eval` 的全词表 logits / `log_softmax` 中间量，验证可选 token-row chunk 是否保持 KL/PPL 数值、同时降低评估峰值；再以新评估代码下的同源码 control/candidate 配对重新测完整调用。如果最终评估峰值来自系统外部 GPU context，则应分别报告量化进程 allocator 峰值和整卡 NVML 峰值，不将二者混为一谈。

原始数据：`outputs/phase_profile_hessian-hinv-savedtensor4-qwen3-06b-32x256-20261008/`、`outputs/phase_profile_hessian-hinv-crossmodel-qwen3-06b-32x256-control-20261008/`。

### K. 全词表 KL 评估行分块 + Stage-0/Stage-2 暂存：局部 allocator 峰值通过，整卡峰值仍待达标

**瓶颈定位与工程依据。** 0.6B 的全流程 NVML peak 在原始评估函数的末尾；每个样本将完整序列 LM-head logits 转 FP32，并同时对 student/teacher 全词表 logits 做 log-softmax、概率和 KL。REAL-Q 的低显存目标不应被离线评估的全词表临时量抵消。外部工程参考包括大词表交叉熵的分块/融合（Cut Cross-Entropy）以及 PyTorch saved-tensor/offload API；这里不截断词表、不近似 KL，也不改量化/梯度，只按 token 行限制 FP32 KL 中间量。REAL-Q 仍是被优化对象而非外部方法依据。

**实现。** 新增默认关闭 `--eval_logit_chunk_tokens N`。candidate 对 teacher LM-head 使用与原实现相同的完整序列 BF16 GEMM，随后将 teacher logits 放到 CPU；student LM-head 也保持原始完整序列形状。PPL 保留原版全序列 cross-entropy reduction 形状；KL 按 N 行将 full-vocabulary FP32 `log_softmax`/probability 计算流式分块，只保留逐 token 标量 KL。这样避免因改变 BF16 GEMM 行数或 CE reduction 形状引入评分偏移。`tools/run_stage2_memory_profile.py` 的 tracked source 清单现包含 `utils/eval_utils.py`，后续 profile 会检查评估器源码哈希。

**失败实现与修正。** 第一版把 LM head 也按 16 行调用，GPU BF16 GEMM 因批量形状变化，使 PPL 从 `32.6184654` 漂到 `32.6292839`；随后只分块 CE 也因改变 CE kernel/reduction 形状出现不该接受的 PPL 差异。两版都被排除。最终实现恢复原版 head GEMM 与全序列 CE 数学/形状，仅分块 KL 中间量。Qwen 形状 CUDA 合成测试中，PPL NLL 精确相同，KL 绝对差 `9.1e-13`；成功的完整 Qwen 0.6B control KL/PPL 与原版全序列结果相同到打印精度。

**完整 control 验证。** WSL Ubuntu 24.04、RTX 4060 Laptop 8 GiB、Qwen3-0.6B W4A16、WikiText-2 32×256、full-28、8×1024 eval，固定相同代码/缓存。最终分块评估 control 的 state SHA 与 28 层逐层指纹均与原 base 相同；KL/PPL `0.2330884337425232 / 32.618465423583984`，原版为 `0.23308846354484558 / 32.618465423583984`。NVML 100 ms 全程 peak 从原评估的 `5,881 MiB` 降至 `3,505 MiB`；CUDA allocator peak allocated/reserved 仍是 `2,561,811,968 / 2,604,662,784 B`，说明此部分只降低评估的整卡临时量，不改变量化阶段 allocator 水位。新 control 量化/wall 时间 `1,178.31 / 1,241.89 s`，与原 base `1,165.08 / 1,228.07 s` 接近（约 +1.1%），不是速度优化。原始路径 `outputs/phase_profile_evalnorm16v2-control-qwen3-06b-32x256-20261008/`。

**组合 candidate 两次完整结果。** 与上述 chunk16 control 同源运行，candidate 使用 `--static_saved_tensor_offload_mib 4 --stage2_hessian_cpu_offload --stage2_hinv_cpu_offload`。saved-tensor hook 记录 1,792 tensors / 7.00 GiB 累计传输；第一次 CPU RSS sample peak `6,057,410,560 B`（5.64 GiB），第二次 `5,724,594,176 B`（5.33 GiB）。两次运行均完整覆盖 28 层，逐层输出指纹文件 SHA 与 control 完全相同（`14592ce9e5aea6475a2b8e1a4e0f6c13d602fcda6f48e5cc8292b34bba54a66f`），507-tensor state SHA、KL/PPL 也完全一致，分别为 `99589bb56bfe9f3c1bce8381b3d36cdc101e96367aedd9eacb4617fda1404b81` 和 `0.2330884337425232 / 32.618465423583984`；`sources_unchanged=true`。两次 allocator peak allocated/reserved 都是 `2,433,359,360 / 2,480,930,816 B`，相对 chunk16 base 下降约 `5.0% / 4.75%`。但两次 NVML 100 ms 全程 peak 都是 `3,517 MiB`，高于 control 的 `3,505 MiB` 12 MiB，且两次最高采样均发生在最终评估附近。因此该组合在预先约定的整卡显存标准下**重复未通过**，不能归因为偶发采样噪声。候选两次量化时间 `1,227.30 / 1,234.82 s`，wall `1,282.33 / 1,294.36 s`，分别比 control 的 `1,178.31 / 1,241.89 s` 慢；不声称速度收益。分析上，4 MiB saved-tensor offload 与 H/Hinv 暂存确实降低 allocator 账面高点，但未压低最终的 NVML 全程最大值；额外的 CPU-GPU 生命周期/复制也增加了耗时。

**评价与下一步。** 评估 KL 分块本身相对原始完整词表评估，把整卡峰值从约 `5,881 MiB` 降到 `3,505 MiB`，并保持 state、每层输出、KL/PPL；这是本节目前已被完整运行支持的整卡峰值优化。再叠加 Stage-0 与 H/Hinv 暂存后，allocator peak 下降约 5%，但 NVML peak 两次重复都升到 `3,517 MiB`，并且运行更慢，故叠加组合不得按用户要求的严格全程标准作为已成功优化提交。下一步应针对最终评估中仍保留在 GPU 上的完整 student logits/FP32 CE 工作集做生命周期优化；先拆分测量各评估子步骤的 allocated/reserved/NVML 峰值，再寻找不改变 full-sequence GEMM 与 CE 结果的低内存实现。不能只通过把 chunk 数继续减小就预先认定能降峰，因为现有 helper 在 KL 分块前仍创建完整 student logits，并保留全序列 CE 的 FP32 输入；必须实测峰值究竟来自 CE、logits 本体还是 KL 块。若需换成进程专属内存作指标，必须单独说明并遵循论文使用口径，不能把它说成整卡峰值。

原始数据：`outputs/phase_profile_evalnorm16v2-control-qwen3-06b-32x256-20261008/`、`outputs/phase_profile_evalnorm16v2-savedtensor4-qwen3-06b-32x256-20261008/`、`outputs/phase_profile_evalnorm16v2-savedtensor4-repeat2-qwen3-06b-32x256-20261008/`。

### L. 精确全序列 CE 峰值拆解与分块原型：定位到下一项显存优化

**瓶颈与文献依据。** K 节的全流程 NVML peak 在最终评估；逐步采样显示 KL 的 token chunk 大小不是峰值控制项。Cut Cross-Entropy（Wijmans et al., ICLR 2025）指出，大词表 CE 的主要临时开销来自 token×vocabulary logits/log-probability，并通过融合目标 logit 与在线 log-sum-exp 避免完整 logits 物化。REAL-Q 本身是被优化对象；这里的外部工程依据是 CCE 的张量生命周期设计。当前原型比完整 CCE 保守：继续保留完整序列 BF16 LM-head GEMM，避免 GEMM 形状改变带来输出漂移，只将原版全序列 FP32 CE 替换成逐 token 行块 CE，并以 FP64 累加 chunk sums。

**峰值定位探针。** 在 WSL Ubuntu 24.04、RTX 4060 Laptop GPU 上，使用 Qwen3-0.6B 对应的 `seq=1024 / hidden=1024 / vocab=151936 / BF16` 真实 head 维度进行独立 CUDA 生命周期探针。KL chunk 16、8、4 的 CE 前向峰值 allocated 均为 `2,501,350,400 B`，reserved 均为 `2,516,582,400 B`；最高点出现在全序列 `cross_entropy`，并非 KL chunk。因此单纯继续减小 `eval_logit_chunk_tokens` 不能解决此处峰值。原始探针：`outputs/eval_peak_lifecycle_qwen3_06b_20261008.json`。

**分块 CE 数值与峰值 smoke。** 随后用本机 Qwen3-0.6B BF16 实际模型和 LM head，取缓存 WikiText2 token window 并重复到 1024 token，运行真实模型前向得到 head logits。production helper 的全序列 PyTorch CE NLL 为 `0.7489542365`；按 256 token 行块计算并对 chunk sums 做 FP64 累加，NLL 为 `0.7489542563`，绝对误差 `1.98e-8`，打印 PPL 完全一致（`2.11478734`）。allocator peak allocated 从 `3,380,203,008 B` 降至 `1,980,653,056 B`（约 −41%）；前一独立 tensor-lifecycle probe 的 reserved peak 从 `3,755,999,232 B` 降至 `2,021,654,528 B`（约 −46%）。chunk=64、16 在该 smoke 上有约 `1.08e-4`、`1.14e-4` 的 NLL 差异，所以原型选择 256，而不是将任意较小块都宣称等价。该数据来自重复 token 的真实模型 smoke，并非完整校准/评估语料，也不构成完整 REAL-Q 正式结果。`tools/probe_eval_peak_lifecycle.py` 保存了可复现的形状与分块峰值探针；其 JSON 结果位于 `outputs/eval_peak_lifecycle_qwen3_06b_20261008.json`、`outputs/eval_ce_chunk_qwen3_realhead_20261008.json` 和 `outputs/eval_ce_chunk_production_helper_qwen3_06b_20261008.json`。

**代码与单测。** 在 `utils/eval_utils.py` 增加可选 `eval_ce_chunk_tokens` 路径，原默认路径保持不变；同时新增 `--eval_ce_chunk_tokens` 参数和分块 NLL 单测。对 `seq=1024 / vocab=151936` 的 CUDA 随机 BF16 logits，production helper 对 FP64 chunk sum 转回 FP32 后，最新 chunk=256 smoke 与 full CE 得到相同 NLL。相关文件通过 `py_compile` 和直接 CUDA smoke。当前 WSL venv 没有 pytest，所以尚未运行 pytest 套件；完整回归测试仍待运行。候选 manifest 已准备为 `outputs/manifest_qwen3-06b-32x256_evalchunk16_cechunk256_20261008.json`，只在 evalchunk16 control 上追加 `--eval_ce_chunk_tokens 256`，便于下一轮做干净 A/B。

**评价与边界。** 这是有明确显存空间、已定位当前瓶颈并通过真实 Qwen head smoke 的新工程候选，比继续叠加 Stage-0/Hessian offload 更直接地打在最终整卡峰值上。目前还没有将该新 CE chunk 参数放入完整 REAL-Q 28 层流程，不能宣称全程峰值、最终 quantized KL/PPL 或耗时已通过。前两次 Stage-0/H/Hinv candidate 在同源 full-run 下均与 control 的每层指纹、state、KL/PPL 完全相同，但 NVML peak 都是 `3,517 MiB > 3,505 MiB`，速度也慢；故不推荐把该组合单独作为优化结论。下一次完整成对运行应在 matched base/candidate 中都启用 KL chunk16，而 candidate 额外启用 CE chunk256；记录全程 NVML、allocator、逐层 fingerprint、state hash、最终 KL/PPL、RSS 与时间，并至少复现两对。严格门槛仍是候选 NVML 峰值不高于对应 base；若 chunk CE 造成实际量化结果差异，需先审查是否为 reduction 数值误差以及是否在预设容差内，再决定接受与否。

原始数据：`outputs/eval_peak_lifecycle_qwen3_06b_20261008.json`、`outputs/eval_ce_chunk_qwen3_realhead_20261008.json`；CE chunk full-run 尚待执行。

### M. 精确分块 CE 的完整 REAL-Q 首轮配对：0.6B 全流程 NVML 峰值下降 10.84%，复现与扩模进行中

**方法与边界。** 根据 L 节全序列 CE 工作集探针，候选只在评估时将完整序列 LM-head logits 的 CE reduction 按 256 个 token 分块；完整 BF16 LM-head GEMM、量化、Hessian/Fisher、Adam、KL 路径和 KL chunk16 均不变。它借鉴 Cut Cross-Entropy（Wijmans et al., ICLR 2025）减少大词表 CE 中间量生命周期的工程思路，但不是完整 CCE kernel；此处仍物化完整 logits，目标是降低 CE 临时工作区。外部依据：[Cut Cross-Entropy](https://arxiv.org/abs/2411.09009)。

**首轮配置。** WSL Ubuntu 24.04、RTX 4060 Laptop 8 GiB、Qwen3-0.6B、W4A16、WikiText-2 32×256、完整 28 层、评估 8×1024，单卡；两边相同 seed=44、KL logit chunk=16 和 source hash。control 不传 `--eval_ce_chunk_tokens`（默认 0），candidate 仅设 `--eval_ce_chunk_tokens 256`。两次均 `returncode=0`、`sources_unchanged=true`。

**正确性核对。** 28/28 层输出 fingerprints 全部逐项相同（0 个 mismatch）；量化 state SHA 均为 `99589bb56bfe9f3c1bce8381b3d36cdc101e96367aedd9eacb4617fda1404b81`，507 tensors。最终 KL/PPL 两边均为 `0.233088433742523 / 32.618465423584`。因此当前实测配置下，CE 分块没有改变量化产物、逐层输出或最终指标。

**整机峰值与速度。** 100 ms NVML 全进程采样峰值从 3505 MiB 降至 3125 MiB，降低 380 MiB（10.84%）；PyTorch `max_memory_allocated/reserved` 均不变，分别为 `2,561,811,968 / 2,604,662,784 B`，与“量化 allocator peak 未变、评估整机工作区峰值下降”的预期相符。量化时间 control/candidate 为 `1182.79 / 1178.98 s`，候选快约 0.32%；全程 wall 为 `1240.56 / 1247.10 s`，候选慢约 0.53%。单对噪声范围内，不宣称速度提升；当前证据只支持完整流程 NVML 峰值下降，且未超过 base。

**评价与反思。** 这是目前第一个同时通过“全流程整机峰值不高于 base”、逐层输出精确一致和最终 KL/PPL 一致的 CE 工作区优化完整 REAL-Q 试验。结果与 L 节真实模型 smoke 方向一致，但还只有一对，尚不能称为论文最终结论。继续按相同协议进行反序重复（candidate→control），再完成至少三对；之后将同一对照扩至 Qwen3-1.7B，并按该模型相同配置比较全程 NVML 峰值、state/fingerprints、KL/PPL、CPU/RSS 与耗时。若重复中峰值未稳定低于 base，就按失败/不稳定记录；不叠加已未通过严格整机峰值门槛的 Stage-0/H/Hinv offload 组合。

原始数据：`outputs/phase_profile_evalce-qwen06b-r1-control-20261008/`、`outputs/phase_profile_evalce-qwen06b-r1-ce256-20261008/`；manifest 分别为 `outputs/manifest_qwen3-06b-32x256_evalchunk16_control_20261008.json` 和 `outputs/manifest_qwen3-06b-32x256_evalchunk16_cechunk256_20261008.json`。反序重复及 1.7B 验证待完成。

**暂停记录（2026-10-08）。** 反序重复 candidate `evalce-qwen06b-r2-ce256-20261008` 按用户要求在搬动电脑前停止；停在 layer 24/28（约 86%），只有 7,961 个 NVML 采样点，前缀最大值 3,125 MiB。该值属于未完成运行，不计入峰值对照、均值或论文结果；此运行未产出完整 state hash、KL/PPL 或 28 层 fingerprints。其 `run.log`、`nvml_telemetry.csv` 和 `PAUSE_STATUS.md` 保留在对应输出目录。后续需以新 tag 从头重跑 candidate，再执行匹配 control。

**反序重复完成。** 后续以 fresh tag 从头跑 candidate，再跑 control。Candidate `evalce-qwen06b-r2-restart-ce256-20261008` 与 control `evalce-qwen06b-r2-restart-control-20261008` 均 28 层完成，`returncode=0`、`sources_unchanged=true`。两边 fingerprints 相同，且都与首对 state SHA `99589bb56bfe9f3c1bce8381b3d36cdc101e96367aedd9eacb4617fda1404b81`、KL/PPL `0.233088433742523 / 32.618465423584` 一致。整机 NVML peak 再现为 control `3505 MiB`、candidate `3125 MiB`（−10.84%）；PyTorch allocator peak 在两边均为 `2,561,811,968 / 2,604,662,784 B`。这使 0.6B 的两组成对试验均通过精确性和 peak 方向检查。两次配对运行耗时差别明显（首对约相同，第二对候选较快），与先后顺序/设备负载可能混淆；暂不作速度收益结论。原始数据：`outputs/phase_profile_evalce-qwen06b-r2-restart-control-20261008/`、`outputs/phase_profile_evalce-qwen06b-r2-restart-ce256-20261008/`。

**跨模型验证待完成。** 已准备相同量化参数的 Qwen3-1.7B 8×64（复用相同 static cache）、2×1024 WikiText-2 评估的 control/candidate manifest，二者仅差 `--eval_ce_chunk_tokens 256`。在 1.7B 上完成配对并核实 28 层 fingerprints、state、KL/PPL、全程 NVML 与 allocator peak 之前，不将该优化概括为跨规模普遍成立。

**Qwen3-1.7B 首对结果（Stage-2+eval，尚非完整 Stage-0 至评估配对）。** W4A16、WikiText-2 8×64 校准、full-28、2×1024 评估，control/candidate 共用同一份预计算 static cache，唯一差别是 CE chunk256。两边 `returncode=0`、源码哈希未变；28 层 fingerprints 相同、507-tensor state SHA 都为 `881b4510bd03ef24189a111d53b744320f7414e513ab865a270c55177a4e1c5c`，KL/PPL 都是 `0.900344729423523 / 37.77244567871094`。Allocator peak allocated 两边均为 `4,216,318,464 B`；reserved control/candidate 为 `4,299,161,600 / 4,303,355,904 B`。整卡 NVML sampled peak control/candidate 为 `4857 / 4861 MiB`，candidate 高 4 MiB，按严格标准本次不通过。两边峰值 timestamp 分别在 14:17:16 和 15:03:48；profile 显示最高 allocator 阶段是 `layer_27_mlp_down_proj_before_hessian_inverse`，NVML 高点出现在量化中段 down_proj，而非 CE evaluation。该配置量化峰值已压过评估工作集，故 CE 分块可省评估临时量，但无法降低此模型的全程 peak。另因此 A/B 复用了既有 static cache，NVML 数字只覆盖 Stage-2+eval；不得称为完整 REAL-Q Stage-0 至评估结果。原始数据：`outputs/phase_profile_evalce-qwen17b-p1-control-20261008/`、`outputs/phase_profile_evalce-qwen17b-p1-ce256-20261008/`；日志/trace 用于确认 peak phase。

**当前判断。** CE chunk 在 Qwen3-0.6B 已有两对完整运行支持整卡 NVML peak 从 3505 降到 3125 MiB（−10.84%），并精确保持 state/output/quality；在 Qwen3-1.7B 的 Stage-2+eval 配对中未降低端到端 peak。它可以作为“评估阶段按模型/序列规模按需分块”的次级工程能力，不能单独支撑跨模型的全流程显存主贡献。下一步必须处理 REAL-Q 量化中的 `down_proj_before_hessian_inverse` 峰值张量生命周期，并在不改变矩阵/补偿数值的前提下做完整配对；同时补做不复用 static cache 的整流程 NVML 对照，确保最终统计包括 Stage-0。

### N. Qwen3-1.7B Hinv CPU offload：Stage-2 峰值重复下降约 9%，输出完全一致

**问题与依据。** M 节定位出 1.7B 的整卡高点在量化阶段 `layer_27_mlp_down_proj_before_hessian_inverse`，而不是评估 CE。down_proj 的 Hessian inverse 工作集较大；项目现有 `--stage2_hinv_cpu_offload` 将 inverse factor 放在 CPU，并在列块迭代中按需分片传回 GPU。这是状态分层/CPU-GPU 搬运的系统工程方法，概念参考 ZeRO-Offload 与 ZeRO-Infinity；这两篇文献只提供分层存储的设计先例，并不证明本项目能省峰值或提速。REAL-Q 是被优化的系统，不列作外部方法参考。

**实验设置。** WSL Ubuntu 24.04、RTX 4060 Laptop 8 GiB、Qwen3-1.7B W4A16、WikiText-2 8×64 校准、完整 28 层、2×1024 评估、单 GPU。两边均复用同一个已生成的 Stage-0 static cache，所以测到的是 Stage-2 量化加评估，不含 Stage-0 预计算；control 使用默认 GPU Hinv，candidate 只增加 `--stage2_hinv_cpu_offload`。control 与 candidate 的 Hinv offload 代码路径以外的参数、源代码哈希和缓存保持相同。第二对按 candidate→control 反序运行，并在两侧都采集进程树 aggregate RSS。

**正确性。** 两组成对运行都 `returncode=0` 且 `sources_unchanged=true`。candidate 的 28/28 层输出指纹与 control 完全相同，state SHA 都为 `881b4510bd03ef24189a111d53b744320f7414e513ab865a270c55177a4e1c5c`（507 tensors），最终 WikiText-2 KL/PPL 两边均为 `0.900344729423523 / 37.77244567871094`。因此 Hinv 放置策略没有改变本配置下的量化状态或输出。

**显存与系统内存。** 两对 100 ms NVML `memory.used` peak 分别为 `4857→4425 MiB` 与 `4861→4425 MiB`，各下降 432/436 MiB（8.9%/9.0%），候选峰值均低于对应 control。PyTorch allocator peak allocated 每对由 `4,216,318,464 B` 降至 `3,763,333,632 B`（−432 MiB）；peak reserved 由 `4,299,161,600 / 4,303,355,904 B` 降至 `3,846,176,768 B`（−432/−436 MiB）。逐阶段 profile 在两臂均把峰值归到 `layer_27_mlp_down_proj_before_hessian_inverse`，与此前定位一致。第二对进程树 aggregate RSS 峰值 control/candidate 为 `12,305,260,544 / 12,302,147,584 B`（约 11.46 GiB，基本持平）；candidate 没有观测到更高的主机进程树峰值。该 RSS 是进程树各 PID RSS 之和，可能重复计算共享页，不能当作物理 RAM 精确占用。candidate 日志记录 197 次因子调度事件，分片 H2D wall time 累计约 `42.96 s`；因子最大 `603,979,776 B`（576 MiB）。

**速度与评价。** 第一对量化时间从 `2298.08 s` 到 `2443.64 s`（candidate 慢 6.3%），第二对从 `2185.37 s` 到 `2088.68 s`（candidate 快 4.4%）；运行次序和热/缓存状态造成的波动足以改变方向，故当前不宣称速度收益。现有两对支持一个范围受限但可复现的工程结论：在 Qwen3-1.7B、当前 8×64 校准以及复用 Stage-0 cache 的 Stage-2+eval 路径中，Hinv CPU offload 将整卡峰值降低约 9%，且输出完全一致。它达到“candidate peak 不高于 base”的本次门槛，但尚未达到用户提出的 10% 期望，也没有证明全流程 Stage-0 至评估峰值下降；因此只能作为有效的模型/阶段定向优化，不能写成普遍的 REAL-Q 显存缩放解法。

**完整 Stage-0→Stage-2→评估验证。** 为覆盖真实全流程，新建 `tools/run_full_pipeline_memory_profile.py`，在同一 NVML/RSS 时间窗中先执行 FSDP CPU-offload Stage-0 预计算，再使用本次生成的独立 cache 执行完整量化和评估。control 与 candidate 的 Stage-0 cache SHA-256 完全相同（`FEDD810F447E6EB4B6028E5C087778A9908C537069A644DD99EDC2F792A9C78F`），Stage-0 allocator peak allocated/reserved 两边也都为 `2,425,411,584 / 2,810,183,680 B`。全流程 NVML peak 从 `4861` 降到 `4425 MiB`（−436 MiB，−8.97%）；全流程任一阶段 allocator max allocated/reserved 从 `4,216,318,464 / 4,303,355,904 B` 降至 `3,763,333,632 / 3,846,176,768 B`。完整流程仍由 Stage-2 的 `layer_27_mlp_down_proj_before_hessian_inverse` 主导峰值，Stage-0 没有接管高点。28/28 层输出 fingerprint 无差异，state SHA、KL/PPL 同前；两边 Stage-0 cache 的逐字节相同也确认本次公平比较使用相同 Fisher/saliency 输入。全程 process-tree aggregate RSS control/candidate 为 `12,561,141,760 / 12,580,343,808 B`（候选约高 18.3 MiB，整体相近；RSS 会重复计入共享页）。该结果补齐了此前复用缓存的阶段性结论：Hinv CPU offload 在本配置的完整流程中仍满足候选 GPU 峰值低于 base，但节省幅度约 9%，未达 10% 目标。

**完整流程的速度代价。** 这组完整 paired run 中，量化时间 control/candidate 为 `2353.50 / 2932.80 s`（候选慢 24.6%）；从 Stage-0 开始的 total wall 为 `2505.38 / 3081.11 s`（候选慢 23.0%）。candidate 日志记录 197 次 Hinv 因子调度，H2D wall time 累计 `51.14 s`，不足以单独解释全部计时差；先前两组只覆盖 Stage-2+eval 的复跑则速度方向相反，因此 profile run 的计时尚不稳定。当前结论是显存收益可复现，速度退化风险未解决；不宣称加速，后续若继续此项，应先在同机器、无额外 profile 插桩的重复计时中确认速度代价，再考虑 pinned-buffer/pre-fetch 等可引用的传输优化，而不是盲目调参数。`tools/run_full_pipeline_memory_profile.py` 已在 WSL venv 通过 `py_compile` 与 `--help` 检查。

**结论与下一步。** 对 Qwen3-1.7B W4A16、8×64 校准，Hinv CPU offload 是一项有效但范围受限的 GPU 显存工程优化：在完整 Stage-0 到评估流程中，两次阶段复现及一次 fresh-cache 全流程均保持完全一致量化状态与输出，1.7B 全流程峰值下降约 9%；但 0.6B 上已知没有降低严格整机峰值，且当前完整流程速度样本显示明显退化。保留此方法作为按模型/阶段启用的候选，不作为默认路径或跨模型通用主结论。下一步优先拆解并复核这 23–25% 的全流程计时差异；如经非插桩重复确认退化，再参考 pinned-memory/transfer overlap 的工程文献做一个明确针对搬运开销的实现与对照。Stage-0 是否主导的疑问已由本次全流程配对回答：在此配置下不是。

原始数据：`outputs/phase_profile_evalce-qwen17b-p1-control-20261008/`、`outputs/phase_profile_qwen17b-p2-hinv-cpu/`、`outputs/phase_profile_qwen17b-p3-control/`、`outputs/phase_profile_qwen17b-p3-hinv-cpu/`、`outputs/full_pipeline_profile_qwen17b-e2e-control/`、`outputs/full_pipeline_profile_qwen17b-e2e-hinv-cpu/`。基础参数 manifest：`outputs/manifest_qwen3-17b-8x64-eval1024_chunk16_ce256_control_20261008.json`；Stage-0 manifest：`outputs/memory_profile_fsdp-stage0-current-offload-freshcache-qwen3-17b-8x64-20261008/manifest.json`。

**选择性 Hinv offload 后续实验（2026-10-08，首次配对）。** 为区分“只有末层需要 offload”与“所有 FFN down_proj 需要 offload”，新增 `--stage2_hinv_cpu_offload_modules`，按 `层号.模块名` 指定模块；未选中模块继续沿用原 GPU Hinv 路径。完整 runner 增加 `--hinv-cpu-offload-modules` 入口，并把选择器写入 run manifest。选择器通过编译检查，运行日志确认实际选择到预期模块。

先做仅 `27.mlp.down_proj` 的 fresh-cache 全流程试验，并紧接着用当前源码跑全 GPU base。两边 Stage-0 cache 字节 SHA 均为 `fedd810f447e6eb4b6028e5c087778a9908c537069a644dd99edc2f792a9c78f`；28/28 层 fingerprints、507-tensor state SHA `881b4510bd03ef24189a111d53b744320f7414e513ab865a270c55177a4e1c5c`、最终 KL/PPL `0.900344729423523 / 37.77244567871094` 均完全相同。base / 末层 offload 的 NVML peak 都是 `4861 MiB`，allocator peak allocated/reserved 也相同或仅相差约 1.6 MiB；故只搬末层不构成有效峰值优化。紧邻这次末层 selective 的 current base 量化/全程耗时分别为 `2982.18 / 3111.83 s`，末层 selective 为 `2909.52 / 3052.58 s`，相差约 2.4% 和 1.9%，不支持之前单独出现的 24% 时间退化；该差异也说明长时间量化受运行状态影响，单对不应作为速度结论。

随后完成“28 个 MLP `down_proj` 全部选择性 offload”与 current base 的一对完整 fresh-cache 全流程对照。candidate 和 base 均 `returncode=0`、`sources_unchanged=true`。Stage-0 cache SHA 逐字节相同；28/28 层指纹相同，state SHA、KL/PPL 均与上文一致。NVML 全流程 peak 从 `4861` 降至 `4429 MiB`（−432 MiB，−8.89%）；allocator peak allocated 从 `4,216,318,464` 降至 `3,763,333,632 B`，reserved 从 `4,303,355,904` 降至 `3,850,371,072 B`（均约 −432 MiB）。NVML sampled peak 与 allocator profile 均定位到 down_proj / Hessian inverse 工作阶段；candidate 全流程最大 allocator sampled phase 为 `layer_27_mlp_down_proj_before_hessian_inverse`。process-tree aggregate RSS base/candidate 为 `12,569,513,984 / 12,568,317,952 B`，基本持平。该结果支持“只对 MLP down_proj 迁移 Hinv 即可保留此前全模块 offload 的约 9% 峰值节省”，且严格未高于 base。

本次 down_proj candidate 量化/全程耗时为 `2133.41 / 2269.18 s`，current base 为 `2982.18 / 3111.83 s`，表观快约 28.5% / 27.1%；28 个模块的 H2D 分片搬运累计约 `20.37 s`。时间收益目前只有一对，且 NVML 平均 utilization 不支持把巨大差值直接解释成 H2D 优化的结果；之前单次 Hinv offload 也曾出现相反时间方向。故先把显存结果记录为强正向候选，速度结果只记作待复现，不能声称已确认加速。下一步按反序再跑 fresh-cache candidate→base，确认时间方向和 `≤ base` 峰值能否重现；之后在 Qwen3-0.6B 上用相同模块选择规则做跨规模检验。如果 0.6B 峰值由其他阶段主导，按日志确认是否需要按峰值模块选用，而不是强行推广。

**反序重复配对（2026-10-08）。** 为消除固定先后次序的偏差，另建一对全新 Stage-0 cache，执行顺序为 candidate→base；两边均为 Qwen3-1.7B W4A16、8×64 校准和 2×1024 评估配置，并刷新了全新 Fisher cache。两边 cache 文件逐个 SHA-256 完全相同，28/28 层输出 fingerprint 一致，507-tensor state SHA 同为 `881b4510bd03ef24189a111d53b744320f7414e513ab865a270c55177a4e1c5c`，KL/PPL 同为 `0.900344729423523 / 37.77244567871094`。全流程 NVML peak 再次为 base `4861 MiB`、candidate `4429 MiB`（少 `432 MiB`，−8.89%）；allocator allocated/reserved 也从 base `4,216,318,464 / 4,303,355,904 B` 降至 candidate `3,763,333,632 / 3,850,371,072 B`（均少 `432 MiB`）。因此峰值显存收益在第二个完整配对中复现，且没有超过 base。量化时间为 base `1851.65 s`、candidate `1875.39 s`（candidate 慢约 1.28%）；全程 wall 为 `1979.43 / 2001.64 s`（candidate 慢约 1.12%）。结合前一对明显相反的单次时间方向，当前应认定为**稳定降低约 9% GPU 峰值、速度中性/未证明加速**，不把第一对的巨大速度差当成优化效果。原始数据：`outputs/full_pipeline_profile_qwen17b-selective-hinv-all-downproj-r2/`（candidate）与 `outputs/full_pipeline_profile_qwen17b-selective-hinv-all-downproj-r2-base/`（base）。

**Qwen3-0.6B 跨模型配对（2026-10-09）。** 将相同的 28 个 `mlp.down_proj` 选择器用于 Qwen3-0.6B W4A16、WikiText-2 32×256 校准、完整 28 层、8×1024 评估。control 和 candidate 都通过 FSDP2 meta-init/CPU-offload 重算 Stage-0，并分别新建 static cache；13 个 cache 文件逐文件 SHA-256 相同。28/28 层输出 fingerprints 完全相同，507-tensor state SHA 同为 `99589bb56bfe9f3c1bce8381b3d36cdc101e96367aedd9eacb4617fda1404b81`，KL/PPL 均为 `0.2330884337425232 / 32.618465423583984`。全程 NVML peak 两边都是 `3521 MiB`，且峰值采样时间分别落在最后 KL/PPL 评估期间；因此候选没有高于 base，但也没有降低整卡峰值。allocator allocated 峰值由 base `1,811,390,976 B` 降到 candidate `1,779,633,152 B`（少约 `30.3 MiB`），reserved 两边均为 `2,491,416,576 B`；两次 Stage-0 allocated/reserved 峰值完全相同，为 `1,680,262,656 / 1,960,837,120 B`。allocator allocated 峰值仍出现在 `layer_27_mlp_down_proj_after_hessian_inverse`，整卡 NVML 峰值却由未受该优化影响的 KL/PPL 评估决定。量化时间 base/candidate `1182.40 / 1181.80 s`，全程 wall `1281.50 / 1274.75 s`，单对差距低于 1%，视为速度中性。结论是 down_proj Hinv offload 在 1.7B 上省下的约 9% 整卡峰值不能外推到 0.6B；它是依模型/阶段瓶颈选择的策略，0.6B 本组只减少少量 allocator allocated 高水位。

**运行器兼容性修正。** 这组 0.6B 测试还发现全流程 runner 假设 Stage-0/Stage-2 manifest 都预先含 `--static_cache_path`，并把 allocator profile 和 profile 文件列表硬编码在 `outputs/Qwen3-1.7B/` 下。修正后，缺少 cache path 时 runner 会补上其独占 fresh-cache 路径，并按实验名在各模型输出根目录查找阶段 profile。base run 已由修正后的解析器正常生成 Stage-0/Stage-2 allocator 字段；candidate 在运行结束后用相同 profile 扫描函数对原始 JSONL 做了后处理补齐，另保留 `status_reprocessed_after_runner_path_fix=true` 标记。两次 raw JSONL 与 NVML CSV 均保留，汇总字段可独立复核。WSL venv `py_compile` 通过。

原始数据：`outputs/full_pipeline_profile_qwen17b-selective-hinv-downproj/`、`outputs/full_pipeline_profile_qwen17b-selective-hinv-current-base/`、`outputs/full_pipeline_profile_qwen17b-selective-hinv-all-downproj/`。该三组实验使用上方同一 Stage-0 / Stage-2 manifest、Qwen3-1.7B W4A16、WikiText-2 8×64 校准、完整 28 层和 2×1024 评估，三次各自新建 Stage-0 cache。

**Qwen3-0.6B 完整流程反序重复（2026-10-09）。** 为确认前一对 full-pipeline 结果不是运行顺序偶然，新建独立 Stage-0 cache，先跑 CE chunk 256 candidate、再跑默认 CE base；模型/数据、W4A16、32×256 校准、28 层、8×1024 评估及其他量化参数相同。两份 Stage-0 Fisher cache 文件 SHA-256 均为 `47a4b7448d80fb45baf999e4296b770068774ed81dc7425c14d7a699dd801dae`；Stage-0 source manifest 哈希相同。Stage-2 命令的实质差异只有 candidate 的 `--eval_ce_chunk_tokens 256`（另有实验名和独立 cache 路径）。两边 `returncode=0`、`sources_unchanged=true`；28/28 层输出 fingerprint 完全相同，507-tensor state SHA 均为 `99589bb56bfe9f3c1bce8381b3d36cdc101e96367aedd9eacb4617fda1404b81`，KL/PPL 均为 `0.2330884337425232 / 32.618465423583984`。

**全流程显存与速度。** 100 ms NVML 采样的整卡 `memory.used` 峰值从 base `3521 MiB` 降至 candidate `3133 MiB`，少 `388 MiB`（`11.02%`），candidate 严格低于 base。base 峰值发生在最终 KL/PPL 评估附近；candidate 峰值转移到量化阶段，说明分块评估压低原评估峰值后，量化成为新的全程峰值。两边 CUDA allocator 峰值完全相同：Stage-2 allocated/reserved 为 `1,811,390,976 / 2,491,416,576 B`，Stage-0 为 `1,680,262,656 / 1,960,837,120 B`；因此收益是减少完整流程的整卡峰值，而不是降低 PyTorch allocator 峰值。进程树聚合 RSS candidate/base 为 `6,525,755,392 / 6,545,207,296 B`，差约 18.6 MiB，不能据此声称显著 RAM 节省。量化时间 candidate/base `1201.24 / 1186.83 s`，总 wall `1297.27 / 1283.76 s`，candidate 分别慢约 `1.21% / 1.05%`；计入前一对后，速度方向一致地小幅变慢，故不报告加速。

**结论与边界。** 第二个独立 full-pipeline fresh-cache 配对复现了 CE chunk 256 的全卡峰值收益：两对均把 NVML peak 从 `3521` 降到 `3133 MiB`（约 `−11.0%`），且输出与最终质量严格一致，满足“候选任一全程峰值不高于 base”的门槛，并达到当前设定的 10% 显存目标。该效果源于降低 PPL/CE 评估的临时 logits/loss 工作集；不降低量化 allocator 峰值，也未在 Qwen3-1.7B 上降低由量化阶段主导的全程峰值，因此应表述为 Qwen3-0.6B 当前配置下的有效评估期内存优化，不能外推为跨模型通用方案。速度没有收益，反而小幅退化；下一步应保留 chunk256 为可选评估内存路径，优先检查是否有无需增加峰值、且不改变数值的评估 kernel/lifetime 优化空间，并在论文主结果中如实同时报告 memory 与 wall-time。

原始数据：`outputs/full_pipeline_profile_qwen06b-evalce256-fullpipeline-candidate/`、`outputs/full_pipeline_profile_qwen06b-selective-hinv-all-downproj-xmodel-base/`（第一对），`outputs/full_pipeline_profile_qwen06b-evalce256-fullpipeline-r2-candidate/`、`outputs/full_pipeline_profile_qwen06b-evalce256-fullpipeline-r2-base-clean/`（反序重复）。

**CE chunk256 + selective down_proj Hinv 的追峰值组合实验（2026-10-09）。** CE chunk 两次完整配对后，全流程 NVML 峰值由 KL/PPL 评估转移到量化；逐阶段 JSONL 显示新高点在 `layer_27_mlp_down_proj_after_hessian_inverse`。因此在 CE chunk256 基础上，只对 28 个 MLP `down_proj` 暂存 Hinv，并与已完成的 CE-only candidate 比较；其他 Stage-2 配置完全相同。两次独立 Stage-0 Fisher cache SHA 均为 `47a4b7448d80fb45baf999e4296b770068774ed81dc7425c14d7a699dd801dae`。组合 candidate `returncode=0`、`sources_unchanged=true`，28/28 层指纹、507-tensor state SHA、KL/PPL 均与 CE-only candidate 完全相同（state `99589bb56bfe9f3c1bce8381b3d36cdc101e96367aedd9eacb4617fda1404b81`；KL/PPL `0.2330884337425232 / 32.618465423583984`）。

整卡 NVML peak 组合前后都是 `3133 MiB`，故 Hinv offload 没有继续降低 Qwen3-0.6B 的全流程设备峰值；但 allocator peak allocated 从 CE-only 的 `1,811,390,976 B` 降至 `1,779,633,152 B`（约 −30 MiB），reserved 都是 `2,491,416,576 B`。峰值事件转移到 `layer_23_mlp_down_proj_hessian_batch_accumulated`，说明 Hinv 离开 GPU 后 Hessian 累积成为 allocator 的新高点。量化时间 CE-only/组合为 `1201.24 / 1220.28 s`（组合慢约 1.58%），总 wall 为 `1297.27 / 1324.00 s`（慢约 2.06%）。因此这个组合只降低 allocator allocated 高水位，没有带来额外的整卡峰值节省，且增加耗时；不建议在 0.6B 配置默认启用。它也验证了下一步若要继续压低该设置的全流程峰值，重点应放在 Hessian 累积生命周期，而非重复优化已被移出 GPU 的 Hinv。原始数据：`outputs/full_pipeline_profile_qwen06b-evalce256-hinvdownproj-fullpipeline-r1-candidate/`，对照为 `outputs/full_pipeline_profile_qwen06b-evalce256-fullpipeline-r2-candidate/`。

**Hessian + Hinv CPU 暂存组合（2026-10-09，追量化峰值的单次全流程候选）。** CE chunk256 与 selective down_proj Hinv 后，profile 最大 allocated phase 指向 `layer_23_mlp_down_proj_hessian_batch_accumulated`。据此，在 CE chunk256 上追加项目已有的全模块 `--stage2_hessian_cpu_offload`，同时保留 28 个 down_proj 的 selective Hinv offload；Stage-0 cache SHA 与 CE-only 比较臂相同（`47a4b7448d80fb45baf999e4296b770068774ed81dc7425c14d7a699dd801dae`）。完整流程 `returncode=0`、`sources_unchanged=true`；28/28 层 fingerprints、state SHA、KL/PPL 完全一致（507 tensors；state `99589bb56bfe9f3c1bce8381b3d36cdc101e96367aedd9eacb4617fda1404b81`；KL/PPL `0.2330884337425232 / 32.618465423583984`）。

**结果与评价。** 整卡 NVML peak 为 `3125 MiB`，只比 CE-only 的 `3133 MiB` 少 `8 MiB`（约 0.26%）；虽仍低于原 REAL-Q base 的 `3521 MiB`，但新增 Hessian/Hinv offload 几乎没有额外的整卡峰值收益。allocator allocated/reserved 为 `1,780,040,704 / 2,483,027,968 B`，对 CE-only 分别少约 30 MiB / 8 MiB；profile 的 allocated 高点仍是 `layer_27_mlp_down_proj_hessian_batch_accumulated`（Hessian 150,994,944 B），说明完整 Hessian 暂存发生在累积完成后，无法消除累积瞬间的 GPU 高点。量化时间从 CE-only `1201.24 s` 增至 `1908.16 s`，完整 wall 从 `1297.27 s` 增至 `2021.17 s`；单次计时还受到 GPU 利用率差异影响，不能把全部差额归因于算法，但这个结果不支持采用该组合。日志总计记录 196 次 Hessian D2H、196 次 Hessian H2D、28 次 Hinv H2D；已记录 copy wall time 的总和只有约 12.7 s，因此额外耗时不应未经 profile 归因于这几项拷贝本身。结论：全模块 Hessian CPU 暂存会增加实现与运行负担，却未解决累积阶段峰值；暂不继续扫该开关，若要压低峰值，需要改成分片/主机端累积使 GPU 上从不同时驻留完整 Hessian，而这会改变计算顺序，必须先做严格数值与逐层输出验证。该候选只跑一对，因此作为否定方向筛选，不作稳定速度统计。

原始数据：`outputs/full_pipeline_profile_qwen06b-evalce256-hessian-hinvdownproj-fullpipeline-r1-candidate/`；对照：`outputs/full_pipeline_profile_qwen06b-evalce256-fullpipeline-r2-candidate/`。

**分组 Hessian 顺序累积（2026-10-09，否定结果）。** 目标是减少 `GPTQPlus.add_batch` 同时创建的 `[G,T,C]` 加权输入和 `[G,C,C]` Hessian 临时块。首版按 group 循环，用 `torch.mm` 单独计算每组 Hessian，再累加到持久 `H`。设计动机是常见的分块/流式计算：避免一次物化所有 group 的临时张量；相关系统先例可参照 Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022（其具体算法是 exact tiled attention，本实验只借鉴“减少中间张量驻留”的系统原则，不声称本方法来自或等同于 FlashAttention）。

**实验设置与可比性。** Qwen3-0.6B、W4A16、WikiText-2 32×256 校准、完整 28 层、8×1024 评估、单 RTX 4060 Laptop 8 GiB；候选在 CE chunk256 配置上追加 `--stage2_hessian_groupwise_accum`，其余 Stage-2 参数相同。候选重新执行 Stage-0 与 Stage-2；候选及 CE-only 参考的 Stage-0 cache SHA 均为 `47a4b7448d80fb45baf999e4296b770068774ed81dc7425c14d7a699dd801dae`。两次运行均 `returncode=0`、`sources_unchanged=true`。

**结果。** 首版候选 28 层输出 fingerprint 全部不同，507-tensor state SHA 从参考的 `99589bb56bfe9f3c1bce8381b3d36cdc101e96367aedd9eacb4617fda1404b81` 变为 `7ecf8197234d060c088dabd1d13d2bc203587683c52e25de0efbe2a876d92d8e`；KL/PPL 从 `0.2330884337 / 32.61846542` 变为 `0.2368731201 / 32.44405365`。因此即使指标数值接近，也不满足本项目要求的逐层输出和量化状态严格一致。NVML 全流程 peak 是 `2981 MiB`，比 CE-only candidate 的 `3133 MiB` 低 `152 MiB`（4.85%）；但 allocator peak allocated 为 `1,812,439,552 B`，反而比 CE-only 的 `1,811,390,976 B` 高约 1 MiB，reserved 降至 `2,332,033,024 B`（比参考少约 152 MiB）。量化耗时从 `1201.24 s` 增至 `2278.21 s`，约慢 89.7%；全程 wall 从 `1297.27 s` 增至 `2397.43 s`，约慢 84.8%。故本次不通过正确性/性能门槛，NVML 峰值下降不能作为有效收益结论。

**诊断与反思。** 两臂使用同一 Stage-0 cache，差异来自候选对 Hessian 乘法实现的改变。把参考路径的完整 batch `bmm` 改成单 group `mm` 或 batch-1 `bmm`，都得到完全相同的候选 state 与 KL/PPL，说明不是 `mm` 专有问题：逐组 GEMM 的归约/执行路径改变了 Hessian 累加轨迹，导致全部层的离散量化状态变化。batch-1 `bmm` 复测也未恢复逐层一致，且量化时间仍约为参考的 1.97 倍；因此不再继续调这个按组串行方案。两次候选结果完全相同（state SHA `7ecf8197234d060c088dabd1d13d2bc203587683c52e25de0efbe2a876d92d8e`，KL/PPL `0.2368731201 / 32.44405365`），都是 28/28 层 fingerprint 不同于参考。首轮原始数据：`outputs/full_pipeline_profile_qwen06b-evalce256-hessian-groupwise-fullpipeline-r1-candidate/`；batch-1 `bmm` 复测：`outputs/full_pipeline_profile_qwen06b-evalce256-hessian-groupwise-bmm-r2-candidate/`；对照：`outputs/full_pipeline_profile_qwen06b-evalce256-fullpipeline-r2-candidate/`。

**保留完整 batched `bmm` 的原位 Hessian 累加（2026-10-09，正确但无收益，淘汰）。** 为隔离前述不一致究竟来自逐组归约还是单纯的 H 输出分配，第三臂保持完整 batch `bmm` 及相同加权输入，试用 `torch.baddbmm(H, Xᵀ, Xw, beta=1, out=H)` 原位累计，意图避免临时 `[G,C,C]` 结果块。CUDA 微基准中该写法与 `bmm` 后 `add_` 的 8 次累计逐位相等，但这不足以推断全流程省显存。

**全流程结果。** 同为 Qwen3-0.6B、W4A16、WikiText-2 32×256 校准、完整 28 层、8×1024 评估和单 RTX 4060 Laptop 8 GiB。候选 28/28 层输出 fingerprint、507-tensor state SHA、KL/PPL 均与 CE-only 参考完全一致（state `99589bb56bfe9f3c1bce8381b3d36cdc101e96367aedd9eacb4617fda1404b81`；KL/PPL `0.2330884337425232 / 32.618465423584`）。但整卡 NVML 峰值为 `3229 MiB`，比 CE-only 的 `3133 MiB` 高 `96 MiB`；allocator allocated/reserved 为 `1,813,725,696 / 2,592,079,872 B`，参考为 `1,811,390,976 / 2,491,416,576 B`。量化时间 `2394.85 s` 对 `1201.24 s`，总 wall `2513.05 s` 对 `1297.27 s`。候选运行期间设备利用率也更高（75.8% 对 46.2%），因此 wall 差距受并发负载混杂，不能全部归因于 `baddbmm`；然而无论如何，峰值显存没有降低、也看不到速度收益，故仍不接受。原始候选：`outputs/full_pipeline_profile_qwen06b-evalce256-hessian-baddbmm-r1-candidate/`，参考：`outputs/full_pipeline_profile_qwen06b-evalce256-fullpipeline-r2-candidate/`。

**结论与后续。** 两个尝试都已从正式参数入口及 `GPTQPlus.add_batch` 移除：分组累积因精确性失败而终止；原位批量累积因没有通过峰值门槛而终止。今后若重访该瓶颈，先用全流程阶段峰值和同负载重复计时证明临时 `[G,C,C]` 是峰值来源，再考虑有证据的分配复用方案；不可仅凭微基准或 reserve 下降认定优化成功。对于更大模型，当前保留的正向候选仍是 Qwen3-1.7B 的所有 MLP `down_proj` Hinv CPU 暂存：两次完整配对精确，峰值由 4861 降到 4429 MiB（−432 MiB），量化时间差异约 1% 且重复间不稳定。下一轮优先复核峰值事件和可移植性，避免继续花数十分钟验证没有峰值证据的临时缓冲变体。

**Qwen3-1.7B、8×128 下的全部 MLP `down_proj` Hinv CPU 暂存复核（2026-10-09）。** 为检验 8×64 结果在更长校准序列上的可迁移性，使用同一 Qwen3-1.7B W4A16 配置，校准样本数保持 8、序列长度从 64 改为 128，完整 28 层量化，WikiText-2 2×1024 评估。唯一算法/内存策略差异是 candidate 对 28 个 `mlp.down_proj` 启用 `--stage2_hinv_cpu_offload_modules`；其余量化参数、seed、stage runner、source-manifest hash 相同。两份 Stage-0 Fisher cache SHA 完全相同：`5efc152f883a52215fdcf15fe33e5b75232267f6b10e014c0ae902ea53eefdf1`；所有共享源文件 hash 与复制的 entry 源码 hash 也相同。两次全流程均 `returncode=0`、`sources_unchanged=true`。

**精确性。** 两边 28/28 层输出 fingerprint JSON 逐项完全相等（文件 SHA256 均为 `0f1652cbd826bfbbaeed35f552d20c009b899c12def1e8686236a320b4a76776`）；507-tensor 量化模型 state SHA 都是 `9e2943f05177a7210753a616d8543624582db2601043cdb3ffea53d7ea33880e`。KL/PPL 两边均为 `0.7262135744094849 / 30.44288444519043`。这证明该 offload 对当前路径只改变 Hinv 的存储位置/搬运生命周期，没有改变量化权重及最终输出。

**全流程峰值与代价。** 100 ms NVML 全程采样：整卡 used 峰值由 control `4,879 MiB` 降至 candidate `4,447 MiB`（少 `432 MiB`，`−8.85%`）；整卡平均 used `3,135.04 → 3,053.22 MiB`（`−2.61%`）。PyTorch allocator Stage-2 峰值 allocated `4,220,558,336 → 3,767,573,504 B`（`−10.73%`），reserved `4,322,230,272 → 3,869,245,440 B`（`−10.48%`）；Stage-0 allocated/reserved 两边相同（`2,445,384,704 / 3,487,563,776 B`）。两边 allocator 最大 phase 都是 `layer_27_mlp_down_proj_before_hessian_inverse`，对应峰值由 Hinv 仍全部留在 GPU 的 control 降到 28 个 down_proj Hinv 已按策略暂存的 candidate；这说明本组全程差值来自量化阶段，而非评估或 Stage-0。主机聚合 RSS sampled peak 为 `12,566,523,904 → 12,589,957,120 B`（增加约 22.3 MiB、`+0.19%`），候选没有明显增加本机 RAM 峰值。量化时间 `3,243.17 → 3,321.75 s`（candidate 慢 `2.42%`），全流程 wall `3,384.47 → 3,460.40 s`（慢 `2.24%`）；当前只是一对运行，速度差异不作统计学结论，也不声称加速。

**评价与边界。** 这次复核维持了“输出精确一致、candidate 全流程整卡峰值低于 control”的门槛，且序列长度加倍后仍节省约 9% NVML 峰值/约 10.5% allocator 峰值，支持 Hinv CPU 暂存是面向大模型 Stage-2 峰值的有效工程优化。它不能代表 REAL-Q 论文正式 2048 样本规模：本次是 8×128 的可重复性/可运行性 pilot，速度还有约 2.4% 的单次回退。下一步应在 1.7B 更接近论文的校准规模做受控小批量扩展，并继续确认 host RAM 可承受；达不到资源预算时，以当前完整配对作为 pilot 证据，不把它冒充正式规模实验。原始数据：`outputs/full_pipeline_profile_qwen17b-hinv-all-downproj-8x128-control-20261009/`、`outputs/full_pipeline_profile_qwen17b-hinv-all-downproj-8x128-candidate-20261009/`；配置清单：`outputs/qwen17b_stage0_8x128_20261009.json`、`outputs/qwen17b_stage2_8x128_20261009.json`。

**Qwen3-1.7B、16×128 下全部 MLP `down_proj` 的 Hinv CPU 暂存反序复现（2026-10-09）。** 为检查 8×128 结果在更大校准 token 数下是否成立，校准改为 WikiText-2 16×128，W4A16、完整 28 层，最终评估仍为 2×1024；唯一差异是 candidate 对每层 `mlp.down_proj` 启用 `--stage2_hinv_cpu_offload_modules`。两组均重新运行完整 Stage-0 与 Stage-2；四份 Stage-0 Fisher cache SHA 都是 `C2062CEF98FBDEB802D35BB58F95787290427A802D5F1E969F6F5FD996C2651D`，源码未变化，所有运行 `returncode=0` 且 `sources_unchanged=true`。

**正确性。** 两组运行的 28 层输出指纹 JSON SHA 均为 `a84f75d55311575c79b314f0845d008b9291b2464ad13d9212c0c2b1e74e5d1a`，包含 28/28 层且逐层哈希相同；507-tensor 量化 state SHA 四次均为 `11d4c62cf4c58bd033e26b85d8ebb6408bae71108a11045512855d918d946c75`。KL/PPL 四次完全相同：`0.6299530863761902 / 27.082468032836914`。因此这次内存路径改变未影响量化模型或层输出。

**全程显存与主机内存。** 两组配对的 NVML 全程峰值都重复为 control `4919 MiB`、candidate `4451 MiB`，下降 `468 MiB`（`9.51%`），candidate 峰值未高于 base。整卡采样均值：第一对 `3221.75 → 3069.07 MiB`（下降 `4.74%`）；反序第二对 `3159.75 → 3116.21 MiB`（下降 `1.38%`）。Stage-2 allocator peak allocated/reserved 由 `4,228,946,944 / 4,364,173,312 B` 降为 `3,775,962,112 / 3,873,439,744 B`，即少 `432 / 468 MiB`；Stage-0 allocator peaks 四次完全一致，为 `2,445,384,704 / 3,487,563,776 B`。聚合进程 RSS（会重复计入共享页）第一对 candidate 低约 `5.08%`，反序第二对仅低约 `0.17%`，所以不能据此主张稳定的主机内存节省。

**速度、评价与下一步。** 两对量化时间分别为 control/candidate `3741.07 / 2400.72 s` 和反序 `2538.48 / 2813.02 s`；方向相反，且一次运行检测到 WSL/Hyper-V 时钟同步跳变、运行时长和采样间隔受影响，故不报告加速。当前可重复支持的是 1.7B、16×128 这一配置下约 `9.5%` 的全程 NVML 峰值降低，同时精确保留输出；这仍是 pilot，不等同于论文正式校准规模。接下来在已验证的 down_proj offload 上扩展到 gate/up 两个 MLP 投影，检查额外 GPU 峰值收益是否足以抵偿搬运与主机 RAM 代价；若收益不明显或正确性不一致，就保留只 offload down_proj 的方案，不再扩大范围。原始数据：`outputs/full_pipeline_profile_qwen17b-hinv-downproj-16x128-control-20261009/`、`outputs/full_pipeline_profile_qwen17b-hinv-downproj-16x128-candidate-20261009/`、`outputs/full_pipeline_profile_qwen17b-hinv-downproj-16x128-r2-candidate-20261009/`、`outputs/full_pipeline_profile_qwen17b-hinv-downproj-16x128-r2-control-20261009/`；配置清单：`outputs/qwen17b_stage0_16x128_20261009.json`、`outputs/qwen17b_stage2_16x128_20261009.json`。

**把 Hinv 暂存扩展到全部 MLP 投影（2026-10-09，未获得增量收益，停止该扩展）。** Candidate 对 28 层的 `mlp.gate_proj`、`mlp.up_proj` 和 `mlp.down_proj` 共 84 个模块启用同一 selective Hinv CPU offload；control 是前述同配置完整流程。Stage-0 cache SHA 仍为 `C2062CEF98FBDEB802D35BB58F95787290427A802D5F1E969F6F5FD996C2651D`，运行 `returncode=0` 且 `sources_unchanged=true`。28 层指纹 SHA、507-tensor state SHA、KL/PPL 均与 down_proj-only 和 control 完全相同（指纹 `a84f75d55311575c79b314f0845d008b9291b2464ad13d9212c0c2b1e74e5d1a`；state `11d4c62cf4c58bd033e26b85d8ebb6408bae71108a11045512855d918d946c75`；KL/PPL `0.6299530863761902 / 27.082468032836914`）。

**结果与反思。** NVML 全程峰值仍是 `4451 MiB`，与 down_proj-only candidate 完全相同，也仍比 control 的 `4919 MiB` 低 `468 MiB`；allocator peak allocated/reserved 仍为 `3,775,962,112 / 3,873,439,744 B`，逐项与 down_proj-only 相同。整卡峰值采样发生在量化末端的 layer 27；memory profile 指向 `layer_27_mlp_down_proj_before_hessian_inverse` 到 `layer_27_activation_replay_and_cleanup` 这一高水位区间。扩展 gate/up selector 没有降低该末端峰值，说明当前最大工作集受最大 down_proj Hessian/逆分解生命周期限制，额外暂存较小 MLP Hinv 并未改变峰值。主机聚合 RSS sampled peak `12,581,183,488 B`，与 down_proj-only 的 `12,584,140,800 B` 基本持平。量化时间 `2,579.57 s`、wall `2,834.02 s`；单次结果不作速度判断。按“增量 GPU 峰值收益必须可测”的标准，停止扩展 gate/up offload，不保留它为独立优化贡献。

**下一项有证据的实现方向。** profile 暴露的末端峰值落在 down_proj 的 Hessian 逆分解路径。代码当前为 `H_work = H_sub.clone()`、对 `H_work` 加 damping、随后构造 Cholesky 与逆-Cholesky 因子；这使 damping 阶段额外保留一份完整 Hessian。先做一个可选的原位 damping 实现：保留原始对角向量以支持 damping retry，将同样的 damping 加到当前 subgroup 的 `H_sub` 对角，避免复制整个 `H_work`；先用微型数值测试检查输出因子是否逐位/容差一致，再通过完整 1.7B 配对确认所有层指纹、state、KL/PPL 和 NVML 峰值。该优化仅在实测仍降低全程 NVML peak 时接受；否则移除。实验来源：`outputs/full_pipeline_profile_qwen17b-hinv-all-mlp-16x128-r1-20261009/`，对照见前述 down_proj-only control/candidate 两对。

**Hessian damping 原位实现的完整 REAL-Q 首轮配对（2026-10-09）。** 在 1.7B、W4A16、WikiText-2 16×128、全 28 层设置下，control 使用原来的 `H_work = H_sub.clone()` damping 路径；candidate 只额外开启 `--hessian-damping-inplace`，将 damping 写入私有 `H_sub` 并保存/恢复原对角以支持 retry。两边分别运行完整 Stage-0、Stage-2 与最终评估，`returncode=0`、`sources_unchanged=true`。CUDA SPD/indefinite retry 小测试也通过，针对 covered cases 检查了 Cholesky/逆因子逐位相同；candidate Stage-2 日志确认该开关为 `True`。

**正确性。** 两边 Stage-0 `static_cache` 共 13 个文件，按相对路径、长度和 SHA-256 全部一致。28 层前向指纹逐层相同，指纹 JSON SHA 均为 `a84f75d55311575c79b314f0845d008b9291b2464ad13d9212c0c2b1e74e5d1a`；量化 state SHA 均为 `11d4c62cf4c58bd033e26b85d8ebb6408bae71108a11045512855d918d946c75`（507 tensors）。最终 KL/PPL 两边完全相同：`0.6299530863761902 / 27.082468032836914`。所以本次内存路径改动保留了全层输出、量化权重和评估结果。

**全程显存与速度。** 100 ms NVML 全程采样峰值从 `4928 MiB` 降到 `4844 MiB`（少 `84 MiB`，约 `1.70%`），candidate 未超过 control。Stage-2 allocator peak allocated 从 `4,228,946,944` 降至 `4,140,037,120 B`（少 `84.8 MiB`），reserved 从 `4,364,173,312` 降至 `4,276,092,928 B`（少 `84 MiB`）；Stage-0 allocator peaks 两边相同（`2,445,384,704 / 3,487,563,776 B`）。量化时间 control/candidate 为 `2684.60 / 2127.14 s`，但只有一对且本地 WSL 时钟/运行速度存在波动，暂不把约 20.8% 的单次差值作为加速结论。

**评价与下一步。** 这是一项完整真实流程上通过逐层精确性、最终 KL/PPL 及“全程 NVML 峰值不高于 base”门槛的工程候选，说明避免 Hessian 临时副本确实能降低峰值；当前证据是一对 1.7B 小校准配置，降幅约 1.7%，远小于已复现两对的 down_proj-only Hinv CPU 暂存（约 9.5%）。因此记录为“首轮 work、待反序复现”，不单独宣称论文最终结果或稳定加速。先做反序 control→candidate 配对复现，至少再重复一对；若峰值差可重现且精确性保持，再评估它与 down_proj-only offload 的叠加效果，判断是否能在不抬高任何时机峰值的条件下继续降低显存。原始数据：`outputs/full_pipeline_profile_qwen17b-hinv-inplace-16x128-r2-control-20261009/`、`outputs/full_pipeline_profile_qwen17b-hinv-inplace-16x128-r2-candidate-20261009/`；配置清单：`outputs/qwen17b_stage0_16x128_20261009.json`、`outputs/qwen17b_stage2_16x128_20261009.json`。

**Hessian damping 原位实现反序复现（2026-10-09）。** 为排除先后顺序影响，按 candidate→control 顺序重新执行同一组 Qwen3-1.7B、W4A16、WikiText-2 16×128、完整 28 层流程。Candidate 仅开启 `--hessian-damping-inplace`；两边重新完成 Stage-0、Stage-2 和评估，均 `returncode=0` 且源码未变。28/28 层指纹无 mismatch，fingerprint JSON SHA 同为 `a84f75d55311575c79b314f0845d008b9291b2464ad13d9212c0c2b1e74e5d1a`；507-tensor state SHA 同为 `11d4c62cf4c58bd033e26b85d8ebb6408bae71108a11045512855d918d946c75`；KL/PPL 都是 `0.6299530863761902 / 27.082468032836914`。

**反序峰值与结论。** NVML 全程峰值 control/candidate 为 `4930 / 4844 MiB`，candidate 少 `86 MiB`（`1.74%`）；首对是 `4928 / 4844 MiB`，少 `84 MiB`（`1.70%`）。两对 Stage-2 allocator 峰值也重复一致：control allocated/reserved `4,228,946,944 / 4,364,173,312 B`，candidate `4,140,037,120 / 4,276,092,928 B`，差值分别约 `84.8 / 84.0 MiB`；Stage-0 峰值不变。主机 RSS 峰值两边近乎一样。两对均按反序或正序得到 candidate 峰值更低，因此该实现作为 1.7B/16×128 的小幅显存优化可重复 work，并且全层结果精确一致；平均峰值降低约 `85 MiB`、约 `1.72%`。这符合“峰值不能高于 base”，但若论文目标要求明显降低显存，它本身仍然太小，不能替代约 9.5% 的 down_proj-only Hinv CPU 暂存方案。

**速度边界与组合计划。** 首对量化时间 control/candidate `2684.60 / 2127.14 s`，反序对 `3198.11 / 2230.51 s`，测得方向都偏 candidate 快，但 WSL/Hyper-V 时钟跳变和单机负载使 wall/quant 时间存在明显波动，且目前仅两对，暂时不宣称加速；需要三对以上、检查时钟稳定后再评估。下一步把原位 damping 与已重复两对的 down_proj-only Hinv 暂存组合，在同一 1.7B/16×128 配置下先做一对完整对照；只有在逐层输出/state/KL/PPL 精确一致且组合 candidate 全程峰值不超过 base 时才保留。如果组合峰值没有比 Hinv 暂存单独方案更低，删除组合复杂度，不继续堆叠。原始数据：`outputs/full_pipeline_profile_qwen17b-hinv-inplace-16x128-r3-candidate-20261009/`、`outputs/full_pipeline_profile_qwen17b-hinv-inplace-16x128-r3-control-20261009/`。

**down_proj Hinv 暂存与 Hessian damping 原位实现组合首对（2026-10-09，增量峰值下降）。** 为验证第二项小优化能否叠加到已复现的 Hinv 暂存上，本次 control 对 28 层 `mlp.down_proj` 启用 selective Hinv CPU offload；candidate 保持同一组 Hinv selectors，仅多开 `--hessian-damping-inplace`。Qwen3-1.7B、W4A16、WikiText-2 16×128、full-28，两边完整重跑 Stage-0、Stage-2 与评估，均 `returncode=0`、`sources_unchanged=true`。

**严格正确性核对。** 两侧 28 层输出指纹逐层一致，0 个 mismatch，fingerprint JSON SHA 均为 `a84f75d55311575c79b314f0845d008b9291b2464ad13d9212c0c2b1e74e5d1a`。量化 state SHA 同为 `11d4c62cf4c58bd033e26b85d8ebb6408bae71108a11045512855d918d946c75`（507 tensors），最终 KL/PPL 完全一致：`0.6299530863761902 / 27.082468032836914`。

**内存效果。** 在 matched Hinv-only control 上，candidate 全程 NVML 峰值从 `4460` 降至 `4326 MiB`，少 `134 MiB`（`3.00%`），全程最大值低于 control。此前写为“没有任何已采样时刻超过 control”超出了证据，2026-10-10 修正：两次运行的时间轴和阶段长度不同，不能由全程最大值大小推出任何对应阶段或时刻均不回升。Stage-2 allocator allocated/reserved 从 `3,775,962,112 / 3,873,439,744 B` 降至 `3,624,991,744 / 3,732,930,560 B`，约少 `144 / 134 MiB`；Stage-0 峰值仍相同。主机聚合 RSS candidate 高约 39 MiB（约 `0.33%`），幅度小且该指标会重复统计共享页，尚不能据此判断主机内存退化。

**解释与边界。** 原地阻尼单独使用只比普通 base 少约 `85 MiB`（约 `1.72%`），确实属于小幅改善；组合后相对 Hinv-only 又少 `134 MiB`，本次 candidate 的 `4326 MiB` 峰值也比此前普通 REAL-Q base `4928/4930 MiB` 低约 `602/604 MiB`（约 `12.2%`）。这是有潜力的组合首对，但还没有反序复现，故先记为“首对 work、增量待复现”。量化时间 control/candidate `2909.82 / 3261.45 s`，单对且存在 WSL 时钟/运行波动，不宣称速度结论。原始数据：`outputs/full_pipeline_profile_qwen17b-hinv-downproj-inplace-16x128-r1-control-20261009/`、`outputs/full_pipeline_profile_qwen17b-hinv-downproj-inplace-16x128-r1-candidate-20261009/`。

**下一步。** 用反序 candidate→Hinv-only control 再跑一对，验证 134 MiB 的增量是否稳健、最终 peak 是否仍低于 base。若精确性保持、峰值增量重现，再扩展到 Qwen3-0.6B 或更接近论文校准规模；若组合中出现峰值回退或层输出差异，先检查峰值阶段与 damping retry 数值路径，不以单对结果作为论文最终结论。

**Hinv 暂存 + 原位阻尼组合的反序复现（2026-10-09）。** Candidate→control 顺序重复同一 1.7B、W4A16、WikiText-2 16×128、full-28 配对。两边均成功完成全流程，`returncode=0`、源码未变。Candidate 的 28 层输出相对 control 逐层完全一致（0 mismatch），fingerprint JSON SHA 两边均为 `a84f75d55311575c79b314f0845d008b9291b2464ad13d9212c0c2b1e74e5d1a`；507-tensor state SHA 均为 `11d4c62cf4c58bd033e26b85d8ebb6408bae71108a11045512855d918d946c75`，KL/PPL 均为 `0.6299530863761902 / 27.082468032836914`。

NVML 全程峰值 control/candidate 为 `4459 / 4325 MiB`，再次相差 `134 MiB`（约 `3.00%`）；首对是 `4460 / 4326 MiB`，同样相差 `134 MiB`。Stage-2 allocator allocated/reserved 的差值两对也相同，为约 `144 / 134 MiB`；Stage-0 allocator 峰值完全不变。两对组合 candidate 的峰值都低于 Hinv-only control，同时低于普通 REAL-Q base 的历史同配置 4928–4930 MiB。故当前可以认为“down_proj selective Hinv CPU 暂存 + 原位 Hessian damping”在 Qwen3-1.7B、16×128 pilot 上是一个有复现的全流程显存优化，且保持严格输出一致；全程 NVML 相对普通 base 下降约 `602–605 MiB`（约 `12.2%`）。这不是论文正式 2048 样本规模，也不能直接外推其他模型。量化耗时继续存在较大波动，不报告速度收益。原始数据：`outputs/full_pipeline_profile_qwen17b-hinv-downproj-inplace-16x128-r2-candidate-20261009/`、`outputs/full_pipeline_profile_qwen17b-hinv-downproj-inplace-16x128-r2-control-20261009/`。

**后续跨模型验证。** 因目标是降低全 REAL-Q 流程在不同模型上的需求，下一步将同一组合放到 Qwen3-0.6B 的匹配 16×128/全层配置做一对试验；若峰值阶段/比例变化，按 profile 定位实际工作集，不假定显存按参数量同比缩放。

**Qwen3-0.6B 组合跨模型结果（2026-10-09 完成，2026-10-10 核对）。** 沿用上节 1.7B 的配置，将模型替换为 Qwen3-0.6B：W4A16、WikiText-2 16×128 校准、完整 28 层、2×1024 评估。Control 对 28 个 `mlp.down_proj` 开启 Hinv CPU 暂存；candidate 在此基础上增加原位 Hessian damping。这是对原位 damping 增量的对照，control 已经过 Hinv 优化，不能将它称为未优化 REAL-Q base。方法、理论依据与参考文献沿用前述 Hinv 暂存和原位阻尼条目，未增加新的算法近似。

**正确性。** 两侧 Stage-0/Stage-2 都返回 0，源码未变。13 个 static cache 文件逐文件 SHA 相同；28/28 层输出指纹相同，0 个 mismatch，fingerprint JSON SHA 都为 `31d803335519f56a21769d7d0b3c29e64aef386a252124816f03c191e9b28a5e`。507-tensor state SHA 同为 `554af15e1a0c09eaa317921c1cc23ad1333ff0c784f4f59a19b13421e09174ee`；KL/PPL 完全相同，为 `0.2762773036956787 / 23.42369270324707`。

**实际效果。** 全流程 NVML peak control/candidate 都是 `3499 MiB`，降幅为 0。Stage-2 allocated peak 为 `1,405,792,256 / 1,405,005,824 B`，仅降低 `0.75 MiB`；reserved peak 为 `1,476,395,008 / 1,478,492,160 B`，candidate 反而高 `2 MiB`。两侧 Stage-0 allocated/reserved peak 均为 `1,220,594,688 / 1,572,864,000 B`，因此全流程 allocator reserved 高水位仍相同。量化时间为 `967.16 / 959.87 s`，全程 wall 为 `1131.45 / 1121.79 s`，只有一对、差距不足 1%，不作为加速证据。

**峰值来源与评价。** NVML 最大值的采样时间 control 为 `20:18:29.197–20:18:29.841`、candidate 为 `20:38:44.752–20:38:45.489`，均与各自的最终 KL/PPL 评估相吻合。原位 Hessian damping 作用于量化阶段，未覆盖此时的评估工作集，所以在 0.6B 上没有降低整卡峰值。这条增量优化在 1.7B 上已有两对正向复现，但本组 0.6B 为“正确性通过、全流程显存增量无收益”，不能宣称跨模型通用节省。Stage-2 reserved 还有 2 MiB 回升，故也不能用“全程 peak 相同”推导每个阶段或任何时刻均不超过 control。

**反思与下一步。** 接下来应按模型的实际峰值阶段启用策略：1.7B 优先降低 down_proj Hessian/Hinv 工作集；0.6B 优先复用此前已有两对正向证据的 CE token chunk 评估路径，并在当前 16×128 配置上做完整匹配对照，确认它能否压低新的全流程峰值。无峰值收益的原位 damping 不因其在 1.7B 有效就默认用于 0.6B。本轮为小校准规模，仍需正式校准配置和服务器多卡验证。原始数据：`outputs/full_pipeline_profile_qwen06b-hinv-downproj-inplace-16x128-r1-control-20261009/` 与 `outputs/full_pipeline_profile_qwen06b-hinv-downproj-inplace-16x128-r1-candidate-20261009/`。

**0.6B 当前峰值的 CE 分块验证协议（2026-10-10，运行中）。** 针对上节最终评估期 NVML 峰值，启动新 candidate，保持同一 16×128/full-28/2×1024 配置与 down_proj Hinv 暂存，仅增加 `--eval_ce_chunk_tokens 256`；两侧均不开原位 Hessian damping。理论依据是独立 token 的交叉熵可以分块计算，缩短全词表 FP32 中间张量的存活范围；外部参考为 Wijmans et al., *Cut Your Losses in Large-Vocabulary Language Models*（ICLR 2025，[论文](https://arxiv.org/abs/2411.09009)）。当前实现保留完整序列 LM-head GEMM，仅对 CE 工作集分块，与论文完整融合线性层和 CE 的内核存在实现区别。

本轮采用上节 Hinv-only control 为首对参照，启动前逐一核对其保存的所有 tracked source SHA 与当前文件完全相同，随后重算 fresh Stage-0 cache。必须在结束后比较 cache 文件、28 层指纹、state SHA、KL/PPL、全流程 NVML peak 及分阶段 allocated/reserved；峰值相同或某阶段增加时如实记录。此处尚无实验结果，不预设会重现旧配置的 11% 收益。候选 manifest 为 `outputs/qwen06b_stage2_16x128_ce256_20261010.json`；运行目录为 `outputs/full_pipeline_profile_qwen06b-hinv-downproj-ce256-16x128-r1-candidate-20261010/`，参照为 `outputs/full_pipeline_profile_qwen06b-hinv-downproj-inplace-16x128-r1-control-20261009/`。结果正向且正确后，再安排反序独立配对与更大校准工作集验证。

**CE256 当前配置首对结果（2026-10-10）。** 上述 candidate 成功完成 Stage-0、全 28 层 Stage-2 及评估，源码未变。13 个 static cache 文件与 control 逐文件 SHA 相同，28 层指纹全部相同，state SHA 为 `554af15e1a0c09eaa317921c1cc23ad1333ff0c784f4f59a19b13421e09174ee`（507 tensors），KL/PPL 为 `0.2762773036956787 / 23.42369270324707`，都与 control 精确相同。NVML 完整运行窗口峰值 control/candidate 为 `3499 / 2173 MiB`，少 `1326 MiB`（`37.90%`）。Stage-0 和量化期 allocator 高水位不变，量化 allocated/reserved 都是 `1,405,792,256 / 1,476,395,008 B`；Stage-0 allocated/reserved 为 `1,220,594,688 / 1,572,864,000 B`。

**效果评价与边界。** 该对是当前 16×128 pilot 的正向整卡峰值结果，重算 cache、量化权重和最终质量都保持一致。量化时间 control/candidate 为 `967.16 / 997.59 s`，完整 wall 为 `1131.45 / 1134.29 s`，不支持加速。审计得到 0 个同步阶段边界回升，但 5 ms sampled phase maxima 有 296 项差异；因候选只改最终 CE，量化阶段的这些短暂采样差异不能直接归因于算法增耗，需精确区间峰值复核。此对参照来自前一晚，虽源码相同，仍需用当前测量器独立反序配对，不将 37.90% 立即推广到正式规模或其他模型。原始审计目录：`outputs/audit-qwen06b-ce256-16x128-r1-20261010/`。

**测量缺口修复（2026-10-10）。** 上述旧 runner 的 `stage2_allocator_peak_*` 和 `allocator_peak_*_any_stage` 来自量化 profiler，截止在量化结束；它们没有覆盖最终评估。因此旧数据中这些字段不能称为“含评估的完整流程 allocator 峰值”。NVML 的外部采样窗口确实覆盖了加载、Stage-0、量化和评估，旧 NVML 结论仍保留。新 runner 在每个阶段完整进程返回后记录每 rank 的 `process_allocator_peak_stage*_rank*.json`，新增 `full_process_peak_allocated_bytes/reserved_bytes` 字段，并保留旧量化字段以明确范围。源文件快照覆盖 gptq_utils/utils Python 实现，记录 allocator/NCCL 环境；状态和指纹改为独占 rank 文件后核对全部 ranks，避免汇总文件覆盖掩盖 rank 漂移。多卡实现仍需实际多卡硬件复测，不能以本地单卡通过代替。

**精确区间 profiler。** 新增默认关闭的诊断环境 `REALQ_PHASE_PEAK_MODE=exact`，由 runner 的 `--exact-phase-peaks` 启用。每个同步 phase marker 读取 allocator 精确区间高水位，再重置区间计数；独立维护全进程高水位，通过 `get_process_cuda_memory_peaks()` 合并结束后的评估分配。该开关只用于诊断：直接读取原始 CUDA max counter 不能在此模式中代表全程峰值，调用者必须使用合并函数。8 个审计单测及 2 个真实 CUDA 测试通过，覆盖末层缺失、数据条件变化、原始 NVML 不符、阶段回升、评估范围缺失、瞬时分配释放后的峰值保留以及默认路径不重置计数。新协议为 `docs/full_pipeline_memory_validation_20261010.md`。

**长序列 Stage-0 计划（2026-10-10，运行中）。** 下一轮先在相同 2×1024 校准条件下比较既有 saved-tensor CPU offload（threshold=4 MiB）关闭/开启。理论依据为 autograd 保存张量可无损迁移到主机，在 backward 按需恢复；参考 PyTorch saved-tensors hooks、Capuchin 的激活分层管理先例，具体实现仍是本项目已有同步迁移原型，不预设有加速。目的为对准 REAL-Q 附录 E(3) 的 Stage-0 激活瓶颈，并检验更长序列的峰值迁移。先跑阶段诊断并比较所有 cache，再决定是否推进完整两模型配对。Stage-0-only 实验不能替代全流程、逐层量化或终稿正式配置数据。0.6B control 为 `outputs/full_pipeline_profile_qwen06b-stage0-2x1024-control-exact-20261010/`；manifest 为 `outputs/qwen06b_stage0_2x1024_20261010.json` 与 `outputs/qwen06b_stage0_2x1024_offload4_20261010.json`。

### O. 长序列 Stage-0 激活卸载：首对显著下降，继续完整流程验证

**瓶颈与方法依据。** 短序列 16×128 的主峰不一定在 Stage-0，不能说明 REAL-Q 端到端预计算激活没有规模瓶颈。固定 2 个校准样本、每样本长度 1024，0.6B 的精确区间 profiler 将主峰定位到 `static_fisher_batch_backward`。按 [PyTorch saved-tensor hooks](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html) 的接口，把 backward 保存的非叶 CUDA 张量无损迁移至 CPU，按需恢复；Capuchin 是张量迁移/重算管理的相关系统文献，并非本实现完整复现的算法。直接复用既有 4 MiB 阈值，不以扫阈值作为工程贡献；量化算法、Fisher、随机采样和精度不变。

**配置与实现。** Qwen3-0.6B、RTX 4060 Laptop 8 GiB、WSL、单 rank，2×1024 WikiText-2；双方 FSDP CPU-offload/meta-init、labelchunk16 相同，candidate 仅增加 `--static_saved_tensor_offload_mib 4`。本次 `--stage0-only --exact-phase-peaks` 是阶段诊断，未执行量化和最终评估。Runner 审计新增明确的 Stage-0-only 模式：强制比较缓存但不会把缺少权重/质量数据的运行称为全流程通过。

**Bug 审查。** 首次 control 的模型工作和缓存生成成功，但 `exit_after_precompute` 中的 `sys.exit(0)` 绕过了测量 footer，导致 wrapper 正确拒绝该次不完整监控。修复为仅捕获成功的 `SystemExit(None/0)`，失败退出仍传播；随后 control 与 candidate 都重跑并成功产生完整进程峰值记录，源码快照相同。新增阶段审计测试连同原测试共 10 项通过。

**实际效果与正确性。** 固定版本 control/candidate 的 NVML 全窗口峰值 `6469 / 2763 MiB`，下降 `3706 MiB`（`57.29%`）。完整 Stage-0 进程 allocated 高水位 `5,760,737,792 / 2,057,167,360 B`，reserved `6,121,586,688 / 2,235,564,032 B`。13 个缓存/准备文件按相对路径逐文件 SHA 完全一致，包括实际 Fisher/saliency cache；全部对应精确区间峰值和同步边界无回升，0 个 phase regressions，NVML 无采样时间缺口。首对审计通过 **Stage-0 阶段** 强门槛。

**代价、评价与反思。** 进程树 aggregate RSS 峰值 `6,266,798,080 / 13,066,547,200 B`（约 `5.84 / 12.17 GiB`），显存节省换来约 6.3 GiB 主机 RSS 增量，不能叫总内存节省；共享页可能被重复计数。candidate 累计卸载 900 张量 / 11.50 GiB payload，是传输总量而非同时驻留。Stage-0 wall `100.06 / 101.03 s`，完整诊断窗口 `112.50 / 112.16 s`；有 checkpoint 生成与插桩，不作速度结论。此结果比短序列试验更贴近 REAL-Q 的 memory-efficient backward future work，但还不是正式 2048×2048 配置，不能推断整个流程下降 57%。下一步在同一 2×1024 校准下做 fresh Stage-0→完整 28 层量化→2×1024 评估，控制其他策略不变；随后更大模型及独立反序复现。如峰值移到量化或评估，以新全局峰值判断，不把阶段收益冒充全程收益。

**原始证据。** `outputs/full_pipeline_profile_qwen06b-stage0-2x1024-control-exact-fixed-20261010/`、`outputs/full_pipeline_profile_qwen06b-stage0-2x1024-offload4-exact-fixed-20261010/`；审计 `outputs/audit-qwen06b-stage0-2x1024-offload4-exact-r1-20261010/`。完整 candidate 现已启动，目录 `outputs/full_pipeline_profile_qwen06b-full-2x1024-offload4-exact-r1-20261010/`，不复用旧 cache；Stage-2 不开启 Hinv CPU staging、原位 damping 或 CE chunk，以隔离 saved-activation offload 的作用。

**全流程首对完成（2026-10-10 02:51）。** 在同一 2×1024 校准、全 28 层量化、2×1024 held-out 评估的缩减配置下，以 candidate→control 顺序分别重算 Stage-0，源码快照、运行环境、物理 GPU 身份和全部算法参数一致，只有 Stage-0 saved-tensor offload 不同。双方共 13 个缓存/准备文件 SHA 相同，28 层输出逐项一致，507-tensor state SHA 同为 `3d2737bec0d9ce5ed6542fd0c88bb830dcd1e39e6591c9b1f59ab893bbcb0803`，KL/PPL 均为 `0.2903517782688141 / 23.759910583496094`。

**全程显存结果。** NVML peak 为 `6469→3499 MiB`，下降 `2970 MiB / 45.91%`；完整进程 allocated peak 为 `5,760,737,792→2,839,141,376 B`，reserved 为 `6,121,586,688→2,866,806,784 B`。量化阶段两侧 allocated/reserved 完全相同：`2,376,964,608 / 2,501,902,336 B`。候选的新全程峰值位于最终 KL/PPL 评估，已用独立 evaluation profiler 和完整进程 footer 确認，并非仅引用截止量化的旧字段。对所有匹配 phase 的每次 occurrence 分别检查精确 allocator 区间峰值，0 个 interval 回升、0 个同步边界回升；整卡 NVML 100 ms 采样无缺口。`strict_measured_memory_gate_passed=true`。这是“已测阶段/区间和完整峰值”的通过，不能数学上证明任意连续时刻的支配关系。

**运行代价与评价。** Control/candidate quantization wall `808.59 / 823.03 s`，完整诊断 wall `955.13 / 947.68 s`，有同步 profile 和哈希/逐层采集，只能作为代价筛查，不能宣称加速。进程树 RSS peak `6,351,527,936→13,079,711,744 B`，约 `5.92→12.18 GiB`。结论是 GPU 资源门槛的有效下降，伴随主机存储成本；不会把搬移后的数据称为总内存消失。当前只有一对、非论文正式配置；需要反序重复、多模型和服务器正式规模验证。

**证据与下一步。** 运行目录 `outputs/full_pipeline_profile_qwen06b-full-2x1024-control-exact-r1-20261010/` 与 `outputs/full_pipeline_profile_qwen06b-full-2x1024-offload4-exact-r1-20261010/`；审计 `outputs/audit-qwen06b-full-2x1024-offload4-exact-r1-20261010/`。先做 1.7B Stage-0 资源诊断，再补完整量化和重复配对。参考 SSDTrain III-B/III-C1 的 storage 身份去重原则，准备默认关闭的 CPU 副本缓存，先验证生命周期、版本/视图安全和 CUDA 梯度，再在真实 Stage-0 量化缓存上测命中、传输量及主机 RSS。没有真实模型收益就不保留，仅玩具测试正向不能写成成果。

**补充时间平均占用检查（2026-10-10 03:36）。** 对 O 节同一完整首对的原始 NVML 数据按相邻时间戳做梯形积分，不用简单样本均值；每 GPU 单独处理，超过五个采样间隔或时间倒退即拒绝。控制/候选监控跨度为 `940.782 / 924.034 s`，最大采样间隔 `0.172 / 0.174 s`，没有缺口。时间加权整卡占用为 `1800.6314 / 1833.2677 MiB`，候选高约 `1.81%`，目前不支持“全程平均占用减少10%”。累计占用积分约为 `1,694,001.589 / 1,694,001.684 MiB·s`，用独立 Decimal 算术复核一致。该窗口包含加载、checkpoint 写盘及退出，不是匹配 phase 的平均活跃张量；长校准预计算占比与论文正式配置不同。这不会推翻峰值/对应区间已通过的结论，但明确说明峰值与全程平均是不同目标，不把其中一个替代另一个。工具 `tools/analyze_memory_time_integral.py`、4 项积分/缺口测试及原始结果 `outputs/audit-qwen06b-full-2x1024-offload4-exact-r1-20261010/nvml_time_integral.json` 已保存。

### P. 1.7B 的 Stage-0 激活卸载：峰值下降但退出缓存回升

**理论与方法。** 与 O 节相同的同步保存张量 CPU 卸载，只替换模型并把校准设为 2×512，以在 8 GiB GPU 上先检查更大模型的 backward 路径；仍为单 rank、4 MiB threshold、FSDP CPU-offload/meta-init。两边原始量化/随机采样/labelchunk16 条件相同。该试验是 Stage-0-only，未执行量化和评估；序列长度不同于 O 节 0.6B，不能把两模型峰值之差归因于模型大小。

**实际效果与正确性。** 双方有效结束、源码未变、缓存逐文件 SHA 完全一致；NVML 全窗口 peak `4867→3301 MiB`（−1566 MiB，−32.18%）。完整 Stage-0 process allocated peak `4,174,578,176→2,565,223,424 B`，reserved `4,441,767,936→2,799,697,920 B`。进程树 RSS peak `13,255,196,672→15,665,582,080 B`；候选累计 offload 788 tensors / 4.72 GiB。Stage-0 wall `240.28 / 168.68 s` 包含从 Windows 文件系统生成准备 checkpoint，单对且大部分是加载/写盘，不能称为加速。

**门槛与诊断。** 唯一对应区间/同步边界回升在 `stage0_exit`：allocated 两侧均只剩 `1024 B`，reserved `1,145,044,992→1,497,366,528 B`，候选多 `352,321,536 B / 336 MiB`。因此虽然全程 peak 明显降低，`strict_measured_memory_gate_passed=false`，不能宣称所有已测阶段均低于 base。该边界的显著 reserved/allocated 差距明确指向 allocator 持有的空闲缓存，而非仍活跃的模型张量；需在完成 Stage-0 缓存持久化的进程交接处释放空闲 GPU 缓存，并在双方同源码下重新测量。不会把删除此阶段遥测或放松比较标准当作修复。

**评价、反思与下一步。** 激活卸载在更大模型预计算上有实测价值，当前实现仍不满足严格部署门槛；不据此称为完整 1.7B 量化成果。先做退出资源生命周期修复，再在同一 2×512 条件下完整配对；必要时与已独立验证的 down_proj Hinv 暂存组合，仍按完整区间门槛判断，不默认叠加就会正向。原始目录 `outputs/full_pipeline_profile_qwen17b-stage0-2x512-{control,offload4}-exact-r1-20261010/`，审计 `outputs/audit-qwen17b-stage0-2x512-offload4-exact-r1-20261010/`。

**交接释放修复与配对复测（2026-10-10 03:21）。** 在 Stage-0 cache 持久化之后、`stage0_exit` marker 之前，仅在预计算退出路径执行一次同步与 `torch.cuda.empty_cache()`；它释放 allocator 空闲缓存，不能释放仍活跃的张量，语义依据为 PyTorch [CUDA memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html)。双方共用同一修复，不向 refresh 热循环加入反复清空。旧峰值仍被 profiler 保留，未删除遥测或改门槛。此次使用可移植 entry，candidate 同时启用 Q 节的 CPU 副本去重。

双方退出前 allocated 均为 `1024 B`、reserved 仍为旧的 `1,145,044,992 / 1,497,366,528 B`；释放后 reserved 均为 `2,097,152 B`。13 个缓存文件 SHA 相同、完整 Stage-0 NVML/allocated/reserved 高水位仍为 `4867→3301 MiB`、`4,174,578,176→2,565,223,424 B`、`4,441,767,936→2,799,697,920 B`，全部对应精确区间与同步边界无回升，`strict_measured_memory_gate_passed=true`。该轮只证明修复后的 Stage-0，完整 1.7B 量化仍待验证。

去重请求 788 次 / `5,066,719,232 B`，实际复制 562 次 / `3,414,163,456 B`，226 命中，D2H 累计请求对应的副本数据少 `32.62%`；active CPU payload peak `1,707,081,728 B`，结束 live 为0。Candidate 进程树 RSS peak 为 `14,547,578,880 B`。它与旧普通卸载不是同源码直接配对，因此不将两轮 RSS 差值当作去重的独立因果降幅；本轮 GPU 收益是关闭卸载对照与“卸载+去重”的组合效果。

证据：`outputs/memory_campaign_qwen17b-stage0-handoff-dedup-2x512-20261010/pair1_audit/`，单日志 `combined.log`，有效状态已生成。串行队列 `tools/run_memory_followup_queue_20261010.py` 已在该门槛通过后启动 0.6B 三组完整配对，然后执行 1.7B 完整配对；新完整数据需要独立审核再追加，不能预先称为成功。

### Q. Stage-0 保存张量的 CPU 副本去重：真实模型首对正向

**动机、理论支持与参考。** O 节的 GPU 显存节省伴随主机 RSS 约翻倍；PyTorch autograd 可能为多个 backward 节点反复保存同一个连续 storage 区域。参考 Wu et al., *SSDTrain*（arXiv:2408.10013v2，2025，III-B/III-C1）的 storage 身份去重原则，减少同一保存值重复存储。该方法是无损资源生命周期管理，不修改目标函数、Fisher、随机标签或梯度更新。本实现不是 SSDTrain 的 SSD I/O、异步预取或权重排除方案。

**具体实现。** `utils/saved_tensor_offload.py` 为 storage 分配递增唯一 ID，避免裸 CUDA 地址复用；键包含 device/dtype/version，连续区域含 offset/numel，非连续视图另含 shape/stride。连续同区域 reshape 共享 CPU payload，unpack 恢复逻辑 shape；仅保存当前视图，不复制整个父/FSDP storage。注册表使用 payload 弱引用、不保留 GPU tensor/storage；最后一个图引用释放后删除 CPU 副本。默认关闭，入口 `--static_saved_tensor_offload_deduplicate`，要求启用正数 MiB threshold。其他 eligibility 与同步搬运保持原样。

**Bug 审查与检查。** 真实 CUDA fanout 图、retained graph 的 repeated-unpack 梯度与无卸载/普通卸载逐位相同；CPU 覆盖 shape/stride/offset、原位修改版本、新 storage、同连续区域 reshape、最后引用释放。审计和配置测试共 26 项通过（包含 3 个真实 CUDA 测试），没有与实际 GPU profile 并发。完整连续日志脚本首次真实执行两臂并通过审计；Stage-0-only 模式明确不宣称全层量化/质量已经验证。

**配置与实际效果（2026-10-10 03:06）。** Qwen3-0.6B，2×1024 校准，单 rank，threshold4 MiB/FSDP CPU-offload/meta-init/labelchunk16。Control 是普通 CPU 卸载，candidate 仅加副本去重；双方全部缓存逐文件 SHA 相同，NVML peak 均为 `2763 MiB`，allocated/reserved 均为 `2,057,167,360 / 2,235,564,032 B`，所有对应精确区间/同步边界无回升。请求 900 次、`12,348,030,976 B`；实际 CPU copies 618 次、`7,407,140,864 B`，282 次命中，累计 D2H payload 少约 40.01%。candidate 的活跃 payload 峰值 `3,703,570,432 B`，结束时 live payload 为 0；普通卸载首版没有这一组件计数，不能推导其组件峰值降幅。进程树 RSS peak `12,854,722,560→10,293,284,864 B`，少 `2,561,437,696 B`（约 19.93%）。

**评价与反思。** 去重在真实模型上减少主机驻留和重复传输，保留普通卸载的 GPU 节省；不是 GPU peak 的新增收益。Stage-0 wall `92.17 / 67.44 s` 受准备 checkpoint 生成/文件系统缓存影响且只有一对，不作为加速。这个方向值得保留为激活卸载的工程完善，下一步在完整量化和 1.7B 上验证，随后重复测量。不能把“累计 D2H payload −40%”写成“运行时间 −40%”或“全流程显存 −40%”。

**证据。** 原始目录 `outputs/full_pipeline_profile_qwen06b-2x1024-dedup-stage0-20261010-r1-{control,candidate}/`；审计 `outputs/memory_campaign_qwen06b-2x1024-dedup-stage0-20261010/pair1_audit/`，连续进展 `combined.log`，scope/status/paired records 都已生成。外部原文及哈希在 `outputs/literature_memory_20261010/`，参考文献见 `docs/references_memory_systems.bib`。

### R. 卸载与副本去重的完整重复验证

**理论、方法与文献。** O/Q 节的保存张量迁移与 SSDTrain storage 身份去重组合，不改 loss/Adam/随机标签；修复后的阶段交接释放在两侧共享。采用可移植 entry，每臂重新生成 Stage-0 cache，经过全部 28 层量化和实际 held-out 前向，检查最终状态及质量；0.6B 三组交替顺序，随后 1.7B 首个完整配对。候选 1.7B 另启用全部 down_proj 的 Hinv CPU 暂存，其组合效果不能单独归因于激活卸载。

**首对效果（2026-10-10 03:55 复核）。** 0.6B 2×1024 校准/2×1024 评估，control→candidate；cache 逐文件 SHA、507-tensor state `3d2737bec0d9ce5ed6542fd0c88bb830dcd1e39e6591c9b1f59ab893bbcb0803`、28 层指纹及 KL/PPL `0.2903517782688141 / 23.759910583496094` 全部相同。完整 GPU NVML peak `6469→3499 MiB`（−45.91%），allocated `5,760,737,792→2,839,141,376 B`，reserved `6,121,586,688→2,866,806,784 B`；全部已测对应区间/同步边界无回升，严格门槛通过。完整 aggregate RSS `6,516,658,176→10,359,582,720 B`；相比不卸载的 control 仍有主机代价，不能说总内存下降。去重仍为 900 请求、618 副本、282 命中、实际累计复制 `7,407,140,864 B`、结束 live 为0；这些指标不是 GPU peak 或运行时间的比例降幅。

**评价与后续。** 当前只有该三组 campaign 的第一组通过；第二组已经反序执行，不能提前声明三组都成功。整机环境独立归档：i9-13900H、WSL2、RTX4060 Laptop 8188 MiB、driver566.26、Python3.12.3、torch2.11.0+cu128；每 worker 的 runtime 软件记录另外保存，父进程环境不代替 worker 配置。已添加缺失/重复 rank 搬运计数、积分缺口、空层 hook 和审计绑定的检查，30 项 CPU 审计/协议检查通过；CPU 测试没有初始化 CUDA，没有并发启动其他 GPU 计算，但会使用主机 CPU，因此本批诊断时间仍不作干净速度结论。原有真实 CUDA 函数检查此前已通过，不在当前模型运行期间重复。

**证据。** `outputs/memory_campaign_qwen06b-full-offload-dedup-2x1024-20261010/pair1_audit/` 与该目录 `combined.log`；两臂目录 `outputs/full_pipeline_profile_qwen06b-full-offload-dedup-2x1024-20261010-r1-{control,candidate}/`。环境快照 `outputs/memory_followup_queue_20261010/environment_snapshot.json`，串行后续 driver 在两模型数值和强门槛通过后才开展 CE256 消融与 1.7B 重复。不推送远端，不在运行中改模型实现。

**新的速度候选依据，尚无结果。** 同步卸载的实际 Stage-0 前后向比关闭卸载慢，应单独记录通信代价而不是用文件系统加载波动宣称加速。PyTorch 当前在线文档的 `save_on_cpu(pin_memory=True)` 描述和本机 torch2.11 的实现存在版本差异；本机源码实际是 pinned `empty` 后同步 `copy_`，unpack 的 `.to(..., non_blocking=True)` 才异步。本轮尚未加入 pinned/async 新路径。后续若试验，先做真实 CUDA 值/梯度、版本/视图、异步副本生命周期，再检查真实 cache、host pinned pool/RSS 和各阶段 GPU 占用；不把“non_blocking”自动解释成 copy/compute overlap 或提速。依据：[官方保存 hooks](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)、[pinning/non_blocking guide](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)、[host allocator stats](https://docs.pytorch.org/docs/main/generated/torch.cuda.memory.host_memory_stats.html)。

**第二组反序复现（2026-10-10 04:28）。** Candidate→control 重新执行两侧全流程。第二组审计同样通过：13 个缓存文件、507-tensor state、28/28 层与最终 KL/PPL 精确相同，无区间/边界显存回升，完整 NVML `6469→3499 MiB`，allocated/reserved 与第一组完全相同。主机进程树 RSS `6,349,459,456→10,263,302,144 B`；诊断 quant `812.69→821.56 s`，不宣称加速。第三组已进入 fresh Stage-0，未提前纳入均值。证据：`outputs/memory_campaign_qwen06b-full-offload-dedup-2x1024-20261010/pair2_audit/`。

### S. 激活重算的梯度钩子兼容性：CPU 前置检查，不是显存成果

**理论与依据。** 激活重算用额外前向代替长期保存中间量，是 Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models*（2022）的相关工程路径。考虑它是因为 O/Q 的 CPU 卸载有主机驻留与 PCIe 搬运代价。REAL-Q 收集的是中间输出的梯度，不能默认 forward 变多就一定错误，也不能默认普通训练支持重算就说明本 collector 支持。必须检查真正累加统计的 grad hooks，而不是只比较前向调用次数。

**检查方法。** `tools/check_qwen_checkpoint_hook_compatibility.py` 在 CPU 上读取两模型真实 layer0 的全部权重，冻结参数，采用64-token合成输入和两个固定 VJP，包含 retained graph 的重复 backward。比较普通 block、外层 non-reentrant checkpoint 和原型 `tools/realq_checkpoint_forward_prototype.py` 的 bound-forward checkpoint。第三种保留原 module 对象和名字，避免标准 wrapper 引入 `_checkpoint_wrapped_module` 前缀，污染 saliency cache 的模块键。未接入真实 REAL-Q 流程，不修改正在运行的模型文件。

**实际结果。** Qwen3-0.6B 和 Qwen3-1.7B 在两种重算方式下均与普通 block 的输出、输入梯度以及 block-output/七个投影模块梯度逐字节相同；八个观测点的 grad hooks 均每次测试触发2次。部分投影 forward 从1次变3次（初次+两次重算），down_proj 和整个 block 的完整 forward hook 仍只有1次，因为 non-reentrant early stop 在后段结束重算。此检查说明在该 fixture 上没有重复累加输出梯度，不能推断实际全模型 collector、FSDP、多卡或 CUDA 同样通过。

**评价、风险与下一步。** 不给出 GPU 峰值或加速百分比；输入不是实际 WikiText 校准状态，CPU/eager 与实际 CUDA/math-SDPA 路径不同。已只读检查本机 torch2.11 FSDP2 的 pre-backward：它按需要梯度的输出安装 hook 并调用 unshard，而非仅按参数 requires_grad 决定；这给 frozen-parameter 重算适配提供代码依据，但不能替代真实实验。下一步在当前固定源码重复全部结束后，做真实 CUDA 单 block，再独立 fresh Stage-0/FSDP cache 配对，记录 hook 次数、资源与最终缓存；如果任一不一致，先定位，不推进完整量化。作为卸载的备选/组合方案，不把它称为当前已经 work 的系统结果。

**原始工件。** `outputs/qwen06b-cpu-checkpoint-hook-compatibility-bound-20261010.json`、`outputs/qwen17b-cpu-checkpoint-hook-compatibility-bound-20261010.json`。早期 outer-checkpoint 版本也保留，没有用后续版本覆盖。参考原文位于 `outputs/literature_memory_20261010/`；现有通过的全流程卸载结果仍以 R 节为准。

**文档核对与 CUDA 前置工具（2026-10-10 04:28，尚未执行）。** 核对本机对应的 [PyTorch 2.11 checkpoint 文档](https://docs.pytorch.org/docs/2.11/checkpoint.html)：non-reentrant 保留 autograd 图并支持多种 backward 调用；默认的 determinism check 只核对 shape/dtype/device，不能替代数值相等验证。因此新工具 `tools/check_qwen_checkpoint_cuda_compatibility.py` 另外检查每个模块梯度与真实输出 SHA，使用 math-SDPA、冻结真实权重和两个固定 VJP，记录单 block allocator peak。只做语法检查，当前真实模型配对尚在运行，未并发执行 CUDA 工具。单 block 的合成输入峰值不能作为全 REAL-Q 的显存节省结论。

**适配反思。** 重算必须保留原模块名称、梯度钩子角色和 cache 采样对应关系；Fisher 收集器的部分可选模式会在同一图上反向多次，重算成本随 backward 次数增加。当前主实验没有打开这些可选估计器，后续先验证原算法的单次 backward 路径，再独立验证 retained graph。SSDTrain 的预取与存储硬件条件也不同于本机同步 CPU 迁移，不能从其吞吐率直接推断本实现速度。重算、迁移、选择性组合分别实测，失败不得用更改 Fisher 统计来绕过。

**CUDA 前置结果与测量修正（2026-10-10 09:25）。** 真实 layer0 权重、math-SDPA、0.6B 1024-token 与1.7B 512-token检查完成：普通、外层 checkpoint、bound-forward checkpoint 的输出、输入梯度、八个模块输出梯度完全相同，grad hooks 每种都触发两次。第一版 allocator baseline 在普通与重算模式之间差32 MiB，符合首次 cuBLAS工作区创建，不能把第一版峰值差当作重算显存收益。工具已增加普通 backward 的测量外预热，待GPU空闲再复核一致baseline；预热仅属于诊断fixture，不加入REAL-Q模块的Adam预热。原始文件 `outputs/qwen06b-cuda-checkpoint-1024-20261010.json` 与 `outputs/qwen17b-cuda-checkpoint-512-20261010.json` 保留。

### T. 完整重复结果与1.7B reserved失败定位（2026-10-10）

**方法、理论与参考。** 沿用 R 节无损迁移、SSDTrain身份去重和按需 Hinv CPU暂存，不改变更新目标、Fisher统计或量化轨迹。当前对照双方共享的系统条件保持不变。所有worker独立fresh Stage-0→量化→评估。

**0.6B实际效果。** 三组交替完整28层对照全部通过：13个缓存文件、507-tensor状态、28层输出与最终KL/PPL `0.2903517782688141 / 23.759910583496094` 全部精确相同，所有已测区间/边界无回升。三组整卡峰值均 `6469→3499 MiB`，平均run-peak降低45.91%；allocated均 `5493.87→2707.62 MiB`（−50.72%），reserved均 `5838→2734 MiB`（−53.17%）。这里均值是每次运行最大值的均值，不是时间平均显存。主机RSS峰值均值 `6163.27→9893.98 MiB`（增加约60.54%），去重仍每次减少40.01%累计复制字节，相对没有卸载的对照不能称为主机内存节省。诊断量化时间分别 `805.64/803.63`、`812.69/821.56`、`810.68/810.40 s`，无稳定加速结论。

**1.7B实际效果及失败。** 2×512校准、2×1024评估完整28层首对，输出/cache/state/KL/PPL精确相同（KL/PPL `0.7481881380081177 / 32.389034271240234`）。全程 NVML `5321→4739 MiB`（−10.94%），allocated `4281.06→3718.92 MiB`，reserved `4556→3974 MiB`。但 layer0 activation replay 区间reserved `1619001344→1681915904 B`（+60 MiB），同步边界reserved `1562378240→1675624448 B`（+108 MiB）；allocated相同。仅一个区间/边界失败，仍不满足强标准。对应inactive split `97809920→211056128 B`，差恰为108 MiB，支持锁定allocator碎片，而非活跃张量增加。候选主峰仍在layer27 down_proj Hessian逆分解，故Stage-0节省不能外推完整同降幅。主机RSS `12556996608→14561599488 B`。

**评价与反思。** 0.6B已获得可重复的完整pilot成果；1.7B只有全峰值下降而严格门槛失败，不能提前合并为多模型全部work。原完成队列按要求停止，没有继续重复失败版本。下一项默认关闭的 `--stage2_hinv_cpu_reuse_buffer` 为每个offloaded subgroup复用一份CUDA行缓冲区，减少每个128列块反复申请释放的存储；仅改变复制目的地，不改矩阵值/布局/运算顺序。先在fresh Stage-0 +真实layer0轨迹中检查，再决定是否全层。这个修复未出结果，不宣称能消除碎片。

**证据。** 三组汇总 `outputs/summary-qwen06b-full-offload-dedup-threepairs-20261010/`，可导出图 `outputs/figures-qwen06b-full-offload-dedup-threepairs-20261010/`；1.7B负向报告 `outputs/summary-qwen17b-full-offload-dedup-hinv-failed-gate-20261010/`，原审计 `outputs/memory_campaign_qwen17b-full-offload-dedup-hinv-2x512-20261010/pair1_audit/`。新prefix诊断 `outputs/memory_campaign_qwen17b-layer0-hinv-reuse-fresh-2x512-20261010/`；不将prefix输出一致冒充全28层量化一致。

**平均占用补充。** 本三对完整NVML按原始时间戳梯形积分，候选平均占用减少分别−1.5435%、+1.3601%、+1.2876%，配对比例均值仅+0.3681%，不支持平均减少10%目标。最大间隔0.170/0.169、0.179/0.302、0.175/0.174秒，均无超出0.5秒阈值缺口。该统计包含初始化、checkpoint I/O和退出，不是对应phase的活跃张量平均。`time-integral-r{1,2,3}.json` 保留每对窗口/积分/原值，不能因为峰值下降45.91%就把它当作平均下降。

**修复设计的工程依据。** [PyTorch 2.11 CUDA memory说明](https://docs.pytorch.org/docs/2.11/notes/cuda.html)区分缓存reserved和活跃allocated，并提供memory_stats/snapshot诊断分配模式。本修复从代码中的重复行缓冲申请入手，不通过改变测量、删除回升点或只扫描allocator超参取得通过。每subgroup工作区大小固定 `min(blocksize, columns)×columns×sizeof(dtype)`，最后不足128列块使用同stride的前缀。复制/使用仍在同CUDA流顺序执行；不会并发覆盖尚在用的旧行。若真实prefix仍有inactive split回升，说明原因不只这段重复分配，应从长期驻留对象的segment位置继续定位，不能因数值正确就保留为成功优化。

### U. Hinv缓冲区首版的数值失败与布局修复（2026-10-10 09:45，修复验证进行中）

**理论依据和具体实现。** CUDA caching allocator 可因重复分配的生命周期与长寿命张量交错产生不可释放的 inactive split。尝试为每个 Hinv CPU-offload subgroup 保留一份128行CUDA工作区，循环内只复制新行，避免反复申请同尺寸存储。该变更默认关闭，只用于已启用Hinv CPU暂存的模块；不新增损失或梯度近似。参考仍为 T 节 PyTorch 2.11 allocator 文档，不把原论文作为外部方法参考。

**实际效果：首版不正确。** fresh Stage-0 + layer0量化 + 完整28层评估，13缓存文件精确相同，但最终state及28个评估输出指纹均不同。control KL/PPL `0.01714225485920906 / 15.934643745422363`；candidate `0.01654343493282795 / 15.89669418334961`。整卡峰值 `4867→4415 MiB`，仍有1个区间和1个边界reserved回升。数值差异使方案无效，即使KL较低也不能计为无损系统优化，更不能扩大为全模型重复实验。

**Bug定位和反思。** 原upper Cholesky Hinv是列优先布局，CPU行切片复制至GPU时会保持其stride顺序；首版 `torch.empty((rows, columns))` 是行优先。同样的元素值不代表后续GEMM浮点运算轨迹相同，量化舍入可能放大差异。修复改用 `empty_like(cpu_rows, device=CUDA)`；短末块用meta推断目标stride和as_strided，不额外分配GPU存储。布局是当前主要嫌疑，尚未通过实测，不能写为已证实唯一原因。复用cache开展两臂真实layer0数值冒烟，若通过再fresh重算所有cache核对显存。

**审计工具另一个问题。** 原审计把 `quant_stop_layer=0` 当作评估也只覆盖0层，错误拒绝了28层评估记录。现在从模型config读取总层数，要求包括未量化后缀的完整评估覆盖，另标记量化prefix范围和 `full_model_quantization_validated=false`。新增回归检查同时确认28层记录可接受、双方都漏末层仍拒绝。此修复不消除上述真实数值失败，旧负向审计文件保留。

**证据与评价。** 首版 `outputs/memory_campaign_qwen17b-layer0-hinv-reuse-fresh-2x512-20261010/pair1_audit/`；修复队列 `outputs/memory_repair_queue_20261010/`；代码 `gptq_utils/gptq_plus_utils.py` 的Hinv_row_workspace分支。旧后续队列已按数值失败退出。新的独立checkpoint和CE消融不启用该失败Hinv变更，分别记录，不混合归因。当前仍只有0.6B三对完整pilot满足强门槛，1.7B不能提前列为成功。

**布局修复数值冒烟（09:51完成）。** 相同已完成control的静态缓存，两臂重新执行真实layer0量化及28层评估，源码快照未变。state均为 `b9460192193265b3d9d13f1efd4579af7de489c3ab49051cc7e071bd3ebb865b`；KL/PPL均 `0.01714225485920906 / 15.934643745422363`；两侧完整28层指纹JSON的SHA均 `bf79abe85fa42b4c5ceef0ac653568d0ed4fbda0b470319a02db8f55745cfae5`。保留stride修复消除了该真实轨迹的数值差异。该检查复用cache，不能替代fresh cache审计，也没有完整进程allocator收尾，不能据此宣布强显存门槛通过。下一fresh prefix配对已启动；进一步的1.7B完整量化仍待prefix资源结果。工件 `outputs/phase_profile_qwen17b-layer0-hinv-layout-repair-{control,candidate}-20261010/`。

**Fresh修复验证（10:07完成）：数值正确，碎片修复无效。** 两侧fresh缓存13文件、state、28层评估输出和KL/PPL精确相同，源码/环境/设备相同。完整prefix流程NVML仍 `4867→4415 MiB`（−9.29%），但layer0 activation replay区间和边界仍分别多60/108 MiB reserved，allocated相同，数值与原失败点完全一致。故工作区复用不足以消除该分配碎片，严格门槛仍失败；不扩大为1.7B全层重复，不算成功组件。证据 `outputs/memory_campaign_qwen17b-layer0-hinv-layout-repair-fresh-2x512-20261010/pair1_audit/`。审计明确 `quantized_prefix_diagnostic=true`、`full_model_quantization_validated=false`，避免把未量化后缀的28层评估误写为28层量化。

### V. 选择性激活重算与按需工作集调研（2026-10-10，待实际collector验证）

**理论、文献与对应瓶颈。** Korthikanti et al. (2022) Sec.5 只重算attention core，避免重算QKV与FFN投影；其对象是长期保存的attention中间值，不是改变数学attention或Fisher。该研究的GPT-3/MT-NLG百分比不能直接套用Qwen。本项目先在实际layer0、math-SDPA、两个固定VJP下核对output/input及八个模块输出梯度，再在真实FSDP静态collector中检验缓存和钩子次数。

**具体实现及限制。** `tools/realq_checkpoint_research_entry.py` 是研究入口，默认不重算。`--research_stage0_checkpoint` 保留module名字/对象，包装bound-forward为non-reentrant checkpoint；另一个互斥marker `--research_stage0_attention_checkpoint` 只在collector范围内包装原SDPA函数，调用结束恢复原函数。无grad分支直接执行原函数，不重算teacher路径。观察钩子两臂相同；报告每模块forward/grad次数及真实被checkpoint的SDPA次数。如果请求attention重算但调用次数为0，拒绝实验。尚未整合到production CLI，不改变Adam，也不新增预热反传到量化过程。

**实际效果与评价。** CPU及初版CUDA完整block数值前置检查见S节；带测量外CUDA预热和attention模式的新版等待串行执行。真实collector、主机驻留、GPU完整峰值、缓存等价性尚无结果。研究manifest `outputs/selective-checkpoint-offload-stage0-manifests-20261010/` 比较共享卸载+去重下的attention重算；它是独立Stage-0消融，不能声称完整28层量化已验证。另一个完整block重算实验关闭卸载，用于分别观察两种机制，不能把它的效果归于选择性attention。

**按需矩阵驻留的新文献。** 本轮补读 Rajbhandari et al., *ZeRO-Infinity* (2021) Sec.5.1.3：顺序获取/释放小块降低算子的GPU工作集；Sec.5.2.2还强调临时迁移缓冲复用以减轻碎片。这支持Hessian/Fisher等平方矩阵的驻留管理方向。现有Hinv按行迁移与这种思想相近，但代码早已存在，不追溯为本轮阅读后新实现；只可引用作为设计论证。REAL-Q列顺序和GEMM形状影响浮点轨迹，后续优先管理现有分组的存储生命周期，不能直接拆矩阵乘法后宣布逐位等价。没有实现该论文的NVMe引擎、通信重叠、参数/梯度跨rank分区，不引用其吞吐率为本系统性能。

**证据与反思。** 原文及校验记录 `outputs/literature_memory_20261010/ZeRO_Infinity_2104.07857v1.{pdf,txt,provenance.json}`；选择性重算原文 `Reducing_Activation_Recomputation_2205.05198.{pdf,txt}`；BibTeX见 `docs/references_memory_systems.bib`。目前无新资源百分比，后续结果再按本格式补充。固定GPU8GB的目标是整个量化流程工作集下降，不能只靠单block示例或普通训练文献推断REAL-Q结果。

**新版CUDA前置实测（10:08完成）。** 五模式：ordinary、outer-block checkpoint、bound-forward block checkpoint、attention-core checkpoint、保存值卸载（非叶激活+已识别冻结参数）。两模型在两个固定VJP下均输出/input梯度/八模块输出梯度逐字节相同，grad hooks次数均为2。所有模式的baseline allocated一致：0.6B `111687168 B`；1.7B `177485312 B`，解决了S节初版32 MiB预热偏差。0.6B ordinary→block checkpoint allocated `462019072→430557696 B`，reserved `494927872→536870912 B`，reserved增加，不能称为满足强门槛。Attention-only的两模型allocated/reserved都略增加：0.6B `466213376 / 557842432 B`；1.7B `305465856 / 335544320 B`，ordinary分别 `303368704 / 322961408 B`。单block的瞬态增加不直接判定28层长期保存的累计效果，但必须保留，并在真实Stage-0各区间再核对。

**真实FSDP完整block重算首对（10:13完成，仅Stage-0）。** 不启用激活卸载，ordinary→block checkpoint，fresh校准2×1024，13个文件（含Fisher/saliency）SHA相同，源码/设备/环境一致，224模块均两次grad hook，总计448，所有已测区间/边界无回升。Stage-0整卡peak `6469→2761 MiB`（−57.32%）；allocated `5760737792→2116059648 B`；reserved `6121586688→2233466880 B`；aggregate RSS `6293164032→6291103744 B`，该首对没有激活卸载的主机驻留增幅。此对照是关闭卸载的baseline，不是与卸载+去重做匹配消融；两者内存/速度优劣需要独立对照，不能跨运行归因。`outputs/memory_campaign_qwen06b-checkpoint-stage0-fsdp-2x1024-20261010/pair1_audit/`；完整28层量化、最终KL/PPL、重复和多卡待验证。REAL-Q本身提到70B使用checkpoint，所以贡献应落在本项目收集器/FSDP/缓存身份的工程适配和系统验证，不能把激活重算本身写为新算法。

**真实选择性attention＋卸载首对（11:11完成）：严格失败。** 相同fresh 2×1024、两侧非叶卸载＋去重；13文件和448次模块梯度钩子一致，checkpoint SDPA实际执行。NVML都2763 MiB、reserved都2235564032 B，但candidate allocated `2057167360→2058215936 B`多1 MiB，两个batch前向/反向均有该回升。因此即使整卡peak看似不变，也不满足用户强标准；不加入正向组合。该结果仅在当前已经卸载大activation的路径上测得，不能否定其他负载下attention重算的作用。证据 `outputs/memory_campaign_qwen06b-selective-attention-offload-stage0-2x1024-20261010/pair1_audit/`。

**1.7B实际完整block重算首对（11:26完成，仅Stage-0）：通过。** fresh 2×512，关闭activation offload；对照/候选13个缓存文件SHA相同，模块梯度钩子都是448次，已测区间/同步边界无回升。整卡NVML `4867→3301 MiB`（−32.18%）；allocated `4174578176→2565223424 B`，reserved `4441767936→2799697920 B`。主机RSS `12559175680→12577837056 B`，增加约17.8 MiB，只有一对不作稳定主机性能结论。不是2×1024，与0.6B配置的序列长度不同，所以不能由两次峰值直接推断纯模型大小的增长倍数。工件 `outputs/memory_campaign_qwen17b-checkpoint-stage0-fsdp-2x512-20261010/pair1_audit/` 和两模型 `outputs/summary-qwen{06b,17b}-checkpoint-stage0-fsdp-20261010/`。完整量化、最终KL/PPL和重复仍待完成。

### W. 冻结权重的反向保存副本：身份与版本安全迁移原型

**瓶颈假设、理论与文献。** 本机Torch2.11官方 `Tensor.is_leaf` 语义规定所有requires_grad=False张量都是叶子；现有 `SavedTensorCpuCache.pack` 跳过全部leaf，因此冻结线性权重的保存视图也可能留在GPU。FSDP参数CPU offload不自动保证反向图保存的GPU副本释放。依据PyTorch saved-tensors hooks/save_on_cpu 的保存值迁移语义和 ZeRO-Infinity 工作集驻留原则，考虑无损迁移这些特定副本；当前是代码导出的假设，尚未证实实际Stage-0峰值主要由它们造成。

**具体方法。** `utils/frozen_parameter_offload_prototype.py` 中，前向前只登记真实二维冻结参数的live storage UID和版本。注册表用弱storage引用，释放后版本也清理，不保留GPU tensor/parameter。pack必须匹配登记的身份与版本、CUDA及原4 MiB阈值，才把冻结参数保存视图纳入现有无损CPU去重；其他leaf不因相同shape而纳入。复制保留dtype/stride，不改变参数更新、张量值和梯度公式。实验入口marker `--research_stage0_frozen_weight_offload` 默认关闭，要求共享非叶卸载+去重，且先不与checkpoint组合；生产CLI尚未整合。

**实际效果：身份与梯度前置检查通过，资源实测未完成。** 四项CPU检查覆盖转置共享身份/等值clone不匹配、原地版本修改、trainable参数排除和GPU存储不被注册表延长生命周期（生命周期fixture用CPU）；真实CUDA五模式检查的0.6B/1.7B均输出和梯度精确相同。单block确实分别识别5/7次参数保存请求，复制 `27262976 / 100663296 B`（26/96 MiB）。该CUDA模式同时启用非叶激活卸载，且模型本身保留常驻单block GPU参数，不能把其peak变化单独归因于参数副本迁移或推断真实FSDP收益。实际两模型Stage-0独立配对已准备，等待前一串行队列完成，不并发GPU。

**评价、反思和证据。** 候选针对随模型投影维度增长的驻留量，比只按0.6B峰值排序更符合跨模型目标；是否work以真实collector的cache、每phase完整峰值、主机RSS和传输统计为准。不使用shape猜测权重，不卸载需要训练梯度的leaf，不引用训练文献性能数字为本系统结果。`tests/test_frozen_parameter_registry.py`、`outputs/memory_repair_queue_20261010/qwen3-{06b,17b}-cuda-checkpoint-warmed-selective.json`、`outputs/frozen-weight-offload-qwen{06b,17b}-manifests-20261010/`、`outputs/memory_residency_followup_20261010/`。相关文献/接口URL：[is_leaf](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.is_leaf.html)、[saved-tensors hooks](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)、[ZeRO-Infinity](https://arxiv.org/abs/2104.07857)。

**机制修正（10:50）。** 已核对本机FSDP2的 `FSDPParam.free_unsharded_param`：它会主动释放/缩小参数storage，即使autograd仍持有旧视图，也不能据此断言该权重一直占用GPU。因而上述“leaf保存副本留在GPU”只能作为普通autograd路径假设，不能直接用于FSDP。单独迁移保存权重可能与正常backward重新unshard产生重复驻留；实际两模型独立配对正在运行，必须保留负向结果。

**受约束的下一原型，尚未实测。** 官方 `FSDPModule.set_unshard_in_backward(False)` 只适用于反向不再需要模块参数存储的情况；冻结参数本身不足以满足条件，因为线性层输入梯度仍需要权重。研究入口另外准备“迁移所有实际保存的冻结参数（包括小投影/归一化向量）＋关闭backward unshard”的互斥原型，保持非叶激活阈值不变。禁止与会重新读取模块参数的checkpoint组合，要求真实FSDP、全部参数冻结、参数组配置成功，并在结束恢复状态。工具辅助源码SHA绑定至复制的entry并在收集前后检查。先做CUDA保存值/梯度，再fresh Stage-0 cache和完整资源检查；没有结果前不算有效组件。官方依据：[FSDP2参数生命周期与专家接口](https://docs.pytorch.org/docs/2.11/distributed.fsdp.fully_shard.html)。本机实现和正式多卡仍需分别验证。

**0.6B实际Stage-0消融（10:52完成）：没有新增显存收益。** 相对于非叶卸载＋去重，两侧fresh 2×1024/FSDP，缓存13文件、源码/设备/环境匹配，所有已测区间/边界无回升。NVML都是2763 MiB，allocated/reserved都是 `2057167360 / 2235564032 B`。候选增加282次参数保存请求和 `2149056512 B`累计复制，活跃CPU payload峰值由3532增为4557 MiB；主机进程树RSS `10314166272→12115566592 B`。它数值有效、GPU节省为零，并有主机/传输代价，不纳入有效优化组合。实际工件 `outputs/memory_campaign_qwen06b-frozen-weight-offload-stage0-2x1024-20261010/pair1_audit/` 和candidate `checkpoint_hook_audit_rank0.json`。1.7B仍在独立缓存审计，不能提前外推。

**1.7B实际Stage-0消融（11:06完成）：显存增加，拒绝。** fresh 2×512/FSDP、全部缓存精确相同，但NVML `3301→3895 MiB`（+17.99%），allocated `2565223424→2781282816 B`（+206.05 MiB），reserved `2799697920→3422552064 B`（+594 MiB）。对应batch1反向allocated增加 `313786368 B`，增加发生在反向，不是初始模型装载。主机RSS `14565265408→19225485312 B`（+约4.34 GiB）；新增394次保存请求，参数累计复制 `6881804288 B`。结果与“迁移后的保存值在反向恢复，与正常FSDP恢复重叠”的机制相容，但计数尚不能证明每个重叠张量的具体生命周期。继续测试受约束的无backward-unshard组合，不重复扩大此单独迁移版本。工件 `outputs/memory_campaign_qwen17b-frozen-weight-offload-stage0-2x512-20261010/pair1_audit/`。

### X. 评估交叉熵分块的完整独立消融（2026-10-10 10:47完成）

**瓶颈、理论与文献。** T节0.6B卸载组合的完整allocated/reserved峰值落在最终KL/PPL评估。沿用按token分块管理词表损失中间值的思路，参考Cut Cross-Entropy（arXiv:2411.09009）关于词表中间量显存开销的分析；本实现仍保留原有完整lm-head输出，不等于复现该论文的融合线性/交叉熵kernel。只控制已有评估函数的CE临时量，不能声称量化Adam加速。

**方法与配置。** 两侧共享激活卸载4 MiB＋storage副本去重、FSDP CPU/meta、KL chunk16及原REAL-Q算法；candidate仅设 `--eval_ce_chunk_tokens 256`。Qwen3-0.6B，fresh 2×1024校准、全部28层量化、2×1024评估，一对，独立缓存、源码/环境/设备核对；Hinv复用和重算均关闭。

**实际效果。** 13个缓存文件、507-tensor最终状态、28层输出和KL/PPL `0.2903517782688141 / 23.759910583496094` 精确相同；所有已测对应区间/同步边界无回升。整卡NVML `3499→3151 MiB`（−9.9457%）；完整allocated `2839141376→2376964608 B`（−16.2788%），reserved `2866806784→2501902336 B`（−12.7286%）。两侧量化阶段的峰值本身相同；candidate消除了评估产生的更高峰值，新的全程allocator峰值在 `layer_27_mlp_down_proj_after_hessian_inverse`。Stage-0 allocated/reserved保持 `2057167360 / 2235564032 B`。主机进程树RSS `10059.36→9936.21 MiB`，单对轻微差异不作稳定主机优化结论。

**评价和反思。** 是通过严格门槛的完整单对工程消融，需要重复和其他模型验证；不是显存平均占用下降9.95%，也不是运行加速9.95%。相对于早期关闭卸载的6469 MiB，3151 MiB算术上低51.29%，但两项来自不同campaign，不将此算术拼接当作新组合对照的重复证据。峰值迁移至平方Hessian/Hinv相关工作集，后续应优先处理该随模型投影维度增长的对象。

**证据。** `outputs/memory_campaign_qwen06b-offload-dedup-ce256-full-ablation-20261010/pair1_audit/`、`outputs/summary-qwen06b-ce256-full-ablation-20261010/{summary.json,paired_values.csv,peak_locations.json}`、串行队列 `outputs/memory_repair_queue_20261010/`。同队列Hinv复用严格失败和其他仅Stage-0结果分别记录，不能因为队列overall=false就否定本独立消融，也不能因为本项通过就掩盖那些失败。

**更广CUDA一致性检查（11:12完成）：归约与kernel分支问题。** 在BF16随机logits、词表151936、长度513/1024、三个seed/尺度下，256分块有2/6个case的loss与原路径差一个FP32 ulp；保留逐token loss并执行原FP32 mean后这6个256 case全部相同。16分块的6/6 case仍不同，证明仅修改最终归约不足以恢复所有分块大小。原完整模型首对的精确相等仍是该轨迹的实测，但不能外推所有输入/配置都逐位等价。源文件 `utils/eval_utils.py` 的现有分块路径尚未改动；正在运行的独立collector对照保持冻结。

**定位依据和修复原型，尚未实测。** [PyTorch v2.11.0 SoftMax.cu](https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/aten/src/ATen/native/cuda/SoftMax.cu) 的 `SpatialSoftMax_getBlockSize` 在inner token维度≤64时切换到跨线程词表归约；>64时各线程顺序累加词表。CE输入为 `[1,V,T]`，因此过小tile或短tail改变浮点轨迹。`tools/check_ce_chunk_exactness_cuda_20261010.py --layout-preserving` 原型保证参考T>64时每个tile>64，合并短尾，保留全部O(T) FP32逐token loss并用原mean；参考T≤64时执行同形状完整路径。实际tile上界为 `max(requested,65)+64`，不隐瞒比请求大小更大的短尾/最小tile。新增长度33/65/66/129/258/777覆盖分支阈值和短尾；待当前模型队列全部结束才串行CUDA实测。该规则依据固定Torch版本，其他版本必须重新核对，不能宣称跨版本普遍证明。

**参考文献归档补齐。** CCE v2原文及text/provenance在 `outputs/literature_memory_20261010/Cut_Your_Losses_2411.09009v2.*`，PDF SHA `19026f0d3f943c7b47e50f27a0c3319aec809ed2f22ad11df0419b67b53947c7`；五位作者和版本元数据写入 `references_memory_systems.bib`。不采用该文的梯度截断，也不引用其训练吞吐率为本项目结果。

### Y. 扩展模型及校准工作集的容量预算：代码证据，非实测收益

**依据与待优化对象。** REAL-Q future work的资源门槛需要覆盖模型和校准规模扩展。代码 `gptq_fwrd` 先在GPU创建 `inps[n_local,S,H]`，再GPU clone为 `fp_inps`，最后才在 `offload_inps` 分支转CPU；所以该开关不能避免此前的GPU瞬态分配。之后的逐样本路径已经支持CPU输入再 `.to(dev)`。计划默认关闭的新路径在一开始将这两份缓存建到CPU，保留捕获和后续逐样本运算形状/次序。当前还未修改正在实测的核心代码，也没有实测资源结果。

**同条件的逻辑容量。** `tools/build_qwen_memory_growth_budget.py` 从两模型真实config读取H/FFN宽度/层数。在2048×2048校准、4个数据rank、BF16输入下，两份输入缓存的名义payload为每rank4096/8192 MiB；完整28块FP32 Fisher累积112/448 MiB、BF16 Fisher cache56/224 MiB；g=4且不按输出组分区时，最大FFN Hessian和全部Hinv各144/576 MiB；down_proj全后缀Adam的m/v各合计24/96 MiB。输入工作集按H增长2倍，平方矩阵及FFN权重对应4倍。这里没有包括allocator、workspaces、head/activation、复制和别名，也没有把各项相加当作同时峰值。尤其4个数据rank不代表矩阵也分成4份，实际rank分组策略需另行核对。

**评价与下一检查。** 该预算支持优先看输入瞬态及平方矩阵驻留，而不是依据0.6B单个最高峰盲目排序。保存值迁移、顺序工作集管理的参考依据仍为SSDTrain/ZeRO-Infinity；不用容量公式冒充已取得GPU节省、硬件可行性或真实4卡验证。候选实现后先在同数据的真实prefix比较cache/state/全部评估输出/质量，再扩大校准规模做资源压力实验；数据规模变化只是测试条件，两臂相同，不作为改校准覆盖的算法贡献。证据 `outputs/qwen-memory-logical-growth-budget-20261010/{budget.json,budget.md}`。

### Z. 2026-10-10 中午：补齐负向结论，修复 CE 并继续真实模型验证

**失败教训与实际效果。** W节“迁移全部保存冻结参数＋关闭 backward unshard”真实0.6B/FSDP Stage-0、2×256独立配对在11:32完成：缓存精确相同，但NVML `2401→2699 MiB`，allocated `1607370240→1763476992 B`，reserved `1855979520→2168455168 B`，主机RSS `7079428096→8036880384 B`。配置了29个FSDP参数组并恢复状态，不等于观测到29次实际unshard被消除；不能凭接口名称解释收益。严格失败，不扩展1.7B。工件 `outputs/memory_campaign_qwen06b-all-saved-frozen-no-backward-unshard-stage0-2x256-20261010/pair1_audit/`。

**平方矩阵卸载的独立负向结果。** fresh1.7B/2×512、layer0量化加全部28层评估的Hessian CPU分组驻留在11:51完成：13缓存、状态和输出/KL/PPL相同，NVML `4847→4291 MiB`（−11.471%）。但是 `layer_0_activation_replay_and_cleanup` 区间allocated相同，reserved `1619001344→2118123520 B`（+476 MiB），同步边界reserved `1562378240→2111832064 B`（+524 MiB），严格失败。不能与Hinv复用的60/108 MiB回升混淆；这是另一独立组件。证据 `outputs/memory_campaign_qwen17b-layer0-hessian-cpu-fresh-2x512-20261010/pair1_audit/comparison.json`。单纯减少矩阵活跃payload不能保证allocator保留空间也下降，后续必须改变驻留生命周期并实测，禁止扩大相同已失败版本。

**CE修复的理论支持和方法。** X节原型在11:52完成72个CUDA用例全部相同（8个长度×3个seed/尺度×3个请求tile）。根据固定Torch2.11 SoftMax.cu的归约分支，生产 `_causal_nll_from_logits` 已在所有GPU任务退出后修复：保持大序列每tile>64、合并短尾，保存O(T)逐token损失并采用原FP32 mean；默认chunk=0路径不变。物理tile可能达到 `max(requested,65)+64`，不虚报显存上界。接入后另外重跑72个CUDA用例，`current_all_exact=true`；28项相关检查通过（另4个subtest）。工件 `outputs/ce-production-repair-20261010/cuda-fidelity.json`。这里只证明这些函数级输入，接下来需要重新验证实际量化模型和重复实验。

**新输入缓存工程方案。** 新增默认关闭 `--stage2_input_cache_cpu_init`，强制同时启用 `--offload_inps`。在rank-local输入buffer首次分配时选CPU，参考clone也在CPU，保持BF16捕获值和后续逐样本GPU前向。依据Y节代码瞬态及ZeRO-Infinity的工作集驻留管理；没有近似权重或改损失。两侧共享offload_inps，候选仅提前选择存储位置。新队列独立验证两个模型fresh prefix；失败的输入组件不会加入后续完整组合。

**下一步与评价边界。** `tools/run_cpu_input_checkpoint_followup_20261010.py` 串行安排独立输入缓存配对，以及0.6B全28层三对整块checkpoint＋修复CE组合。整块checkpoint目前两个模型的Stage-0均正向，仍需完整状态/评估和全流程内存证据；不将Stage-0下降当全程下降。保留原路径control，每臂fresh统计，严格检查每phase和同步边界，时间仍为诊断计时；正式规模和真实4卡尚未验证。磁盘空间不足时保留所有工件，停止依赖写入，不能冒险把磁盘失败当GPU问题。

**源码归档。** 修复前 `outputs/runtime-sources-pre-ce-fidelity-repair-20261010.zip` SHA `74a02ec876e230b155b2afb9f416eef41ef2dedb7a343d7559b5158324284efd`；修复后 `outputs/runtime-sources-ce-cpuinput-repair-20261010.zip` SHA `2054067708f0f8adc0876fa24bca1538a8a414ce777548aabdbddd2e13ff2e09`，449份源码文件逐项hash/CRC验证。源码归档不含数据、权重和安装环境；历史实验自己的源码hash仍保留，不覆盖为当前版本。

### AA. 直接 CPU 创建输入缓存：两个真实模型的独立配对（2026-10-10）

**理论支持、文献与方法。** 延续 Y/Z 节输入瞬态分析，借鉴 ZeRO-Infinity 的异构存储工作集管理，不改量化损失/数据/权重。对照和候选均启用原有 offload_inps；候选仅增加 stage2_input_cache_cpu_init，在首次分配和 FP 参考 clone 时直接使用 CPU。共同使用激活保存值卸载4 MiB及去重；保留逐样本传入 GPU 的计算次序。新增选项默认关闭并要求 offload_inps。

**实际效果。** 0.6B fresh 2×1024、1.7B fresh 2×512，均 backward_bsz=2、仅量化 layer0、评估全部28层；各一对。两模型均13缓存文件、最终状态、28层评估输出、KL/PPL精确一致；已测区间/边界无显存回升。0.6B KL/PPL `0.0259459167718887 / 19.7213973999023`，整卡peak `3501→3499 MiB`，allocated均 `2839141376 B`，reserved `2868903936→2866806784 B`。1.7B KL/PPL `0.0180289838463068 / 15.88408088684082`，整卡peak均 `4855 MiB`，allocated/reserved均 `4208094720 / 4288675840 B`。1.7B这次microbatch为2，不能与以前microbatch=1的质量值直接比较。

**评价、反思与下一实验。** 正确性和不回升通过，但这两个小输入配对不能支持有意义的全程峰值节省。0.6B仅2 MiB差异也不能视为稳定收益。下一步独立测32×1024和512×2048真实输入捕获及参考clone，记录实际输入字节hash、失败位置和CPU/GPU峰值；输入捕获容量实验不执行Fisher/Adam/量化，不能把其成功写成完整REAL-Q在该规模可运行。完整量化质量仍由另一个28层重复campaign验证。

**证据。** `outputs/memory_campaign_qwen06b-cpu-input-init-layer0-2x1024-20261010/pair1_audit/` 和 `outputs/memory_campaign_qwen17b-cpu-input-init-layer0-2x512-20261010/pair1_audit/`；实现 `gptq_utils/gptq_plus_utils.py`、`process_args.py`；容量工具 `tools/probe_real_input_cache_capacity.py`。参考文献 [ZeRO-Infinity](https://arxiv.org/abs/2104.07857)。

### AB. 模块结束后的量化元数据驻留：有边界的生命周期假设

**理论、文献与瓶颈。** Z节 Hessian CPU 卸载减少全程peak却在 layer0 replay/cleanup 增加reserved，说明活跃payload和allocator保留空间不是同一指标。[PyTorch CUDA memory说明](https://docs.pytorch.org/docs/2.11/notes/cuda.html#memory-management)支持用memory_stats/snapshot检查。Checkmate（MLSys2020，arXiv:1910.02653v3）将保存/重算/释放时序作为时间—内存调度问题；这里仅借鉴生命周期管理，不实现其MILP。代码中每个完成模块的quantizer仍被返回字典保留；其小GPU scale/zero/maxq buffer可能延长大型allocator segment驻留，但目前尚未证明是本回升的原因。

**具体原型与前置门槛。** 复制的研究entry在GPTQPlus.free前观察quantizer buffer的shape/dtype/hash、GPU storage所在segment的total/active/inactive块。候选将这些完成模块的元数据无损迁至CPU，校验字节后再走原free。两侧共享同一观察器，观测开销不能算速度收益。先做1.7B/Hessian CPU共享基线的一对独立元数据消融，要求7个模块的buffer内容一致及原有全流程严格门槛；通过后再对原control做 Hessian CPU+元数据CPU 的新鲜配对。没有通过，不扩大全模型。

**实际效果、评价和反思。** 尚未完成真实元数据实验。不能因为变量很小就断言无用，也不能因为reserved回升就直接归因为它；segment观察和实际配对会区分假设。研究源码 `tools/prepare_quantizer_residency_pilot_20261010.py`，复制entry在 `outputs/quantizer-residency-prefix-v2-manifests-20261010/entry.py`。Checkmate原文/PDF/text/SHA出处在 `outputs/literature_memory_20261010/Checkmate_1910.02653v3.*`，BibTeX已记录。

### AC. 中断后的重新验证、计时门槛与工程存储记录（2026-10-10 14:03）

**运行事实。** 原全模型组合三对首个control日志最后记录12:36:20、layer11；14:03检查WSL无相关控制器/worker且GPU空闲，具体退出原因未确定。没有完成状态和配对审计，因此这一轮不能算完整结果或计时样本。保留原工件，不从半途缓存续量化。新tag `qwen06b-checkpoint-ce-full-2x1024-threepairs-20261010-restart1408` 使用原配置、每臂独立Stage-0重新启动；依赖队列改为新状态文件和唯一目录。

**验证与计时修复。** 40项资源/评估/协议检查通过。速度复测工具已要求完整28层量化审计，拒绝只量化prefix却评估28层的参照；另外要求每臂fresh Stage-0所有缓存SHA与已验证参照一致，并在CUDA同步的收集器和量化调用计时之外做hash审计。单独记录收集器时间，避免只看量化速度漏掉checkpoint重算代价。新增cache缺失/变更拒绝检查，10项及3个subtest通过。这些是计时协议的前置正确性检查，不是加速结果；实际速度复测等待新的完整组合通过。

**存储整理边界。** 为保留实验且有空间写新工件，仅对明确已完成、配置同模型、prepared model字节hash一致的成对文件做事后只读hardlink，保留两条原路径、统计与日志。0.6B模型byte SHA `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`；1.7B `98f42c93ec4b2b80bff038d836bc1ebe1bf0c889a7706b125e97151ab1e846a1`，各操作报告在 `outputs/disk-housekeeping-*.json`。这是磁盘管理，不是GPU内存优化；fresh收集时两侧未共享这些工件，失败未完成的运行不做去重。源码归档中的legacy stage字段未随新zip名称更新，应以真实文件清单/hash/时间和本文修复顺序解释，不能把错误字段当作版本证据。

**实时入口。** 新的单一汇总日志 `logs/realq-system-restart-20261010-1408.log` 串行转发三条队列的实际worker日志，标出来源和状态，不启动GPU工作。各campaign原始日志保留。完整模型、元数据假设和输入容量分别判定，不能以控制器正常结束代替每项数值和资源门槛。

**参照漂移核对（14:23）。** 新组合首个control完成全28层，state `7ad9b1a7a890faf5175b136b0c12ea30228bea96060a84f9ee734cd2f0c57ff3`，KL/PPL `0.30534985661506653 / 25.013607025146484`。核对实际保存的两轮manifest发现：旧卸载＋去重三对的 backward_bsz/final_layer_backward_bsz 均为1，新组合两侧均为2；另外两侧增加offload_inps。之前根据摘要假设旧轮microbatch为2不成立，以实际manifest为准。Stage-0统计文件大小及SHA完全相同 `9aa82a632f9c468e906127a92f29d05fe81fc44b972fc9e629795dcf68576a82`，因此参照变化并非该文件误复用/缺失。新组合必须独立配对和重复，不与旧微批量的质量、峰值或时间拼接；新微批量用于相同输入缓存pilot的工作集条件，不能把质量差异归因于工程方案。证据 `outputs/reference-drift-provenance-20261010-1423.json`。新候选尚未完成，不提前称其数值一致。

**归档补齐。** restart/followup与新计时协议源码457文件的zip在 `outputs/runtime-sources-restart-followups-20261010-1413.zip`，SHA `cdf220be3976c456c1f61c256283ae7acd102143d9114ba5d37e932f10d28d6c`，逐文件和CRC验证。归档工具现接受显式stage并记录UTC，保留旧归档原字节。WSL在14:09 uptime约5分钟，证实已有新的WSL运行实例；导致原实例退出的宿主事件未确定。

### AD. 全模型组合数值通过、阶段显存失败：不能用首层通过替代全流程（2026-10-10 14:38）

**理论支持与方法。** 同时应用 Z/AA 节的整块 Stage-0 checkpoint、保持归约路径的 CE256，以及 Stage-2 输入缓存直接 CPU 初始化。目标是改变工作集驻留而不改损失/量化结果。两个方法的单独 prefix 或 Stage-0 正向结果仅是前置门槛，组合必须对原系统 control 重新收集统计并验证完整量化。

**实际配置与数值。** tag `qwen06b-checkpoint-ce-full-2x1024-threepairs-20261010-restart1408`，Qwen3-0.6B、2×1024 校准、backward/final microbatch=2、评估2×1024、28层全部量化、W4A16/g4/b128、seed44、不旋转。两侧共享 offload_inps，control 不启用新增组件。首对13个缓存、507个状态tensor、28层评估输出、KL/PPL精确一致，state `7ad9b1a7a890faf5175b136b0c12ea30228bea96060a84f9ee734cd2f0c57ff3`，KL/PPL `0.30534985661506653 / 25.013607025146484`。

**显存结果与评价。** NVML全程peak `6461→4395 MiB`（−31.976%），全过程allocated `5760737792→3299191296 B`，reserved `6121586688→3814719488 B`。但 Stage-2 reserved `3659530240→3814719488 B` 增加148 MiB，另有1175个区间和1150个同步边界回升，严格失败，按规则首对后停跑而不重复失败版本。两臂worker均完成不等于组合通过。CPU RSS peak `6428282880→6441009152 B`。诊断计时含观测/同步，不能称为加速成功。

**反思、下一实验与文献。** 首层输入方案曾通过内存边界，实际全层却出现回升；需要覆盖长量化轨迹上的allocator历史。由于 Stage-0 两臂独立进程且缓存字节一致，下一轮保持 microbatch、offload_inps 等条件不变，仅取消候选 stage2_input_cache_cpu_init，独立验证 checkpoint＋CE。不能把这个怀疑当因果定论。[GMLake（ASPLOS2024）](https://arxiv.org/abs/2401.08156)讨论卸载/重算造成的不规则分配与碎片化，适合解释为何payload减少不必然降低reserved；本项目没有实现其VMS分配器。[PyTorch2.11 CUDA内存说明](https://docs.pytorch.org/docs/2.11/notes/cuda.html#memory-management)用于核对active/cache/segment。证据 `outputs/memory_campaign_qwen06b-checkpoint-ce-full-2x1024-threepairs-20261010-restart1408/pair1_audit/comparison.json`。

### AE. 真实输入捕获容量：两模型四个配置全部相同，但没有 baseline OOM（2026-10-10 14:43）

**理论、文献和方法。** 借鉴 ZeRO-Infinity 的异构存储工作集管理，独立执行真实 Qwen embedding/rotary 前缀、捕获首块输入、创建 FP 输入参考；control 在 CUDA 初始化和clone，candidate 在 CPU。校准行来自WikiText；所有捕获行及clone做bytehash。未收集Fisher、执行Adam或全层量化，不能作为完整REAL-Q可部署容量结论。

**实际结果。** 四配置两臂都完成，输入与FP clone SHA相同；四次baseline均没有CUDA OOM。以下peak是这一独立工作进程的 PyTorch CUDA allocated，不是整卡物理显存；RSS是after_clone的CPU进程值，字节转MiB。

| 模型/输入 | C allocated peak MiB | N allocated peak MiB | C clone后RSS MiB | N clone后RSS MiB |
| --- | ---: | ---: | ---: | ---: |
| 0.6B / 32×1024 | 456.75 | 334.02 | 2109.82 | 2173.25 |
| 1.7B / 32×1024 | 882.00 | 633.27 | 2407.51 | 2661.90 |
| 0.6B / 512×2048 | 4424.75 | 344.81 | 2109.48 | 6206.61 |
| 1.7B / 512×2048 | 8818.00 | 646.06 | 2408.42 | 10598.48 |

1.7B大配置单份BF16 cache 4294967296 B（4 GiB），两份8 GiB。C peak allocated `9246343680 B`、reserved `9273606144 B`；N分别 `677447168 / 683671552 B`；after_clone RSS C/N `2525413376 / 11113308160 B`。两臂输入 SHA `1953d1dacf1b8a45806e5f36d3b1364dbef6f23abe6cf10f9d68c7aa059a4452`。独立捕获wall C/N `9.7273 / 11.4792 s`，单次未重复，不能据此作速度结论。

**评价和反思。** 该组件将按 N×T×H 增长的双输入工作集迁至CPU，代价是系统RAM和拷贝；大配置的allocated显著下降是局部实测。C allocated超过设备报告8188 MiB仍完成，说明allocator逻辑值不能直接作为物理GPU需求或硬件OOM证明。WSL/Windows驱动fallback是可能解释，尚未定位本机驱动行为，不能断言已确认。若后续采用相同显式allocator预算，必须注明预算受控实验，与本次不设预算的实际结果分开；禁止把预期OOM写成已发生。完整全层加入该组件在 AD 节失败，因此目前不纳入最终组合。

**证据。** `outputs/input-capacity-restart-20261010-1408/status.json`、八个worker JSON与原始日志，工具 `tools/probe_real_input_cache_capacity.py`；[ZeRO-Infinity](https://arxiv.org/abs/2104.07857)。

### AF. 元数据实验入口修复与串行后续协议（2026-10-10 14:57）

**失败审查。** AB节研究entry错误引用 impl.GPTQ.free，但真实类是 GPTQPlus。v2首个 Stage-0 在14:38:22导入失败，零对完成；这是基础设施bug，没有执行待测量化方法，不能评价为方法无收益。旧目录/错误日志保留。原型改用 GPTQPlus.free，并重新生成 v3 的不可变entry和四个manifest。

**修复验证与方法。** 工具 `tools/check_quantizer_lifetime_runtime_20261010.py` 在实际CUDA Linear＋GPTQPlus＋WeightQuantizer上执行两臂free，小测试显示maxq/scale/zero字节完全相同；candidate三者均CPU，原free释放H。证据 `outputs/quantizer-api-preflight-20261010-1500/status.json`。该验证避免只编译却遗漏API错误，不能当模型级内存结果。真实1.7B元数据配对排队等待当前GPU实验完成，只在独立元数据方案严格通过后比较原control与HCPU＋元数据CPU；若不通过，再根据segment观测分析completed block驻留，不扩大相同失败版本。

**当前队列。** 0.6B新全模型tag `qwen06b-checkpoint-ce-original-input-full-threepairs-20261010-1457`，每臂fresh Stage-0、保留microbatch2等原条件，仅去掉直接CPUinput-init；首对严格通过后自动做三对、交替顺序。依赖控制器按状态文件串行启动，不并发占GPU。后续1.7B prefix通过后才扩大全层；0.6B完整内存三对通过后才验证Triton组合，再进行不含诊断同步/张量hash的重复速度计时，并单独计入Stage-0收集重算代价。未完成步骤均不声称成功。

**评价、记录和工程边界。** 核心生产源码在worker运行中保持不变；工具入口复制并保留SHA。磁盘整理只对已完成、prepared model SHA匹配的文件做事后只读hardlink，统计独立且不减少GPU内存。Windows长路径接口修复已在两个完成模型文件上实际通过。单一实时入口仍为 `logs/realq-system-restart-20261010-1408.log`，显式append且前一个转发器已退出，原始worker日志完整保留。最新结果将继续追加在本记录。

### AG. 去掉 CPUinput-init 后首个完整配对通过：checkpoint＋CE 的独立证据（2026-10-10 15:29）

**理论、参考与具体方法。** 接AD节，仅撤销候选的直接CPUinput-init，保持两侧既有offload_inps、microbatch2和其他量化条件不变。候选使用Stage-0整块non-reentrant checkpoint＋最终评估CE256；Stage-2量化计算及输入初始化走共同原路径。分别参考 Korthikanti 等的激活重算资源权衡和 CCE 的词表中间量分块思想；这里不实现CCE融合kernel，不改Fisher/Adam目标。两侧重新收集Stage-0统计，不复用已失败或不完整工件。

**实际效果与已验证范围。** tag `qwen06b-checkpoint-ce-original-input-full-threepairs-20261010-1457` 首对在15:29完成。W4A16/per-row、g4/b128、seed44、不旋转，2×1024校准、refresh microbatch2、2×1024评估、28层全部量化。13个缓存、507个状态tensor、28层held-out输出及最终KL/PPL精确一致；KL/PPL `0.30534985661506653 / 25.013607025146484`。2961个已测阶段/边界对应项中，区间和边界回升数量均0，global peaks及strict_measured_memory_gate均通过。

| 首对全流程指标 | control | candidate |
| --- | ---: | ---: |
| NVML整卡采样peak MiB | 6461 | 4247 |
| 完整进程allocated peak B | 5760737792 | 3304098304 |
| 完整进程reserved peak B | 6121586688 | 3659530240 |
| CPU aggregate RSS采样peak B | 6525022208 | 6322028544 |

NVML peak下降34.267%；Stage-0 allocated `5760737792→2116059648 B`，reserved `6121586688→2233466880 B`。Stage-2 allocated/reserved两侧完全相同 `3304098304 / 3659530240 B`；因此此次全程下降来自Stage-0，不声称Stage-2总peak也下降。与AD对照，改变input-init策略后Stage-2回升不再重现，支持allocator历史影响的解释，但只有一对，不是最终因果/重复结论。

**评价、反思与下一步。** 首对同时满足数值与已测资源门槛，是正向pilot；尚未完成三对，不能说已重复成功或终稿验证完成。第二对已自动启动并交换运行顺序。之后1.7B将使用与0.6B相同2×1024、microbatch2、eval2×1024，独立比较同一checkpoint＋CE组合，不将先前不同长度的两模型数据当纯模型大小效应。平方矩阵驻留修复另行验证，只有严格通过才加入组合。Triton组合和低开销速度复测仍待完成，诊断wall不作为加速证据。

**平均显存的限制。** 首对NVML设置为100 ms，两臂各有一个超过5个采样间隔的gap（最大1.189/1.199 s）。现有时间加权分析因此拒绝积分；不能调宽阈值后把未知时段补成0或把sample mean当可靠时间均值，更不能宣称平均显存已降低10%。记录的是全程峰值和已测区间，NVML采样也不证明连续每一时刻。

**工程留档。** comparison在 `outputs/memory_campaign_qwen06b-checkpoint-ce-original-input-full-threepairs-20261010-1457/pair1_audit/`；后续控制器SHA在 `outputs/research-continuation-20261010-1530/controller_sources.json`。GMLake v1原文17页已下载、提取和归档，PDF SHA `99903eef4b04b25eadd60061e5c724505b803fc16adf7d666e1203e712097eb8`，`outputs/literature_memory_20261010/GMLake_2401.08156v1.*`；参考文献BibTeX补齐。本地Python下载的证书链错误通过系统可信HTTPS下载路径解决，没有关闭TLS校验。

**记录器核对。** 新完成结果记录器先在AD的已完成失败pair上独立实测，修复相对路径bug后正确保留“1/3对、数值true、资源false、完整模型true”，未把失败摘要写成成功。测试输出 `outputs/completed-record-preflight-20261010-1526/`。正式记录器等待GPU/计时队列退出才解析原始profile并按本格式追加，避免报告工作争用速度实验CPU；自动摘要仍需人工分析，不代替问题原因审查。

### AH. 保留原 LM-head GEMM 的 KL logits 驻留原型（2026-10-10，待 CUDA/完整流程验证）

**瓶颈、理论与文献。** AG首对的Stage-2 allocated peak两臂相同3304098304 B，新全程峰值在末层KL刷新，Stage-0进一步重算不能压低该峰值。Capuchin（ASPLOS 2020，DOI 10.1145/3373376.3378505）按tensor访问与生命周期组织卸载/恢复，提供保存位置管理的工程参考；CCE（arXiv:2411.09009v2）解释词表中间量随token×vocab增长的开销。这里借用生命周期与词表工作集分析，不实现Capuchin自动调度器，也不把CCE的融合CE直接当成KL实现。REAL-Q仍是研究对象，不作为外部创新方法参考。

**具体方法。** 原 `hidden2logits` 的完整head前向、外部线性反向GEMM及loss逐token归约全部保留；只替换chunked KL节点保存的数据：forward把student/teacher logits字节副本保存到CPU，backward逐token块恢复到GPU，按原loss公式求局部梯度，再组装与原路径相同的完整logits梯度。无top-k截断、无新损失项、无Fisher/Adam参数变化。工具 `tools/cpu_saved_logits_kl_prototype.py`、`tools/check_cpu_saved_logits_kl.py`，独立flag `--research_stage2_cpu_saved_logits` 只在研究入口生效。当前核心生产代码不改动。

**为何与此前失败方法不同。** 先前重算head缩小GEMM行数导致BF16数值漂移；本原型不重算或切分head GEMM，仅管理KL节点保存的logits。它仍生成原完整logits，且仍分配原完整logits梯度，因此不能声称消除了所有O(token×vocab)显存。调用者/其他节点若继续持有GPU logits，卸载保存副本也未必释放原GPU张量；实际持有者与峰值必须由完整运行验证。

**已完成检查。** CPU小fixture（19行、H32、V127）的FP32/BF16两例loss与hidden梯度均torch.equal、最大差0，证据 `outputs/cpu-saved-logits-kl-cpu-fidelity-20261010-1540.json`。该fixture不是实际量化残差，不是CUDA、模型质量或显存收益证据。训练head的实际CUDA形状检查（两模型、2048行）已排在现有GPU及低开销计时队列之后，保持单GPU串行。两例精确通过才启动0.6B fresh Stage-0＋全部28层配对；双方共同checkpoint＋CE，只有candidate改变logits保存位置。随后检查13缓存、全部模型状态、28层输出、最终KL/PPL及严格已测内存门槛，失败则停止同一版本。

**评价、反思与资源风险。** 目前仅是待验证工程假设。大词表CPU副本/PCIe搬运可能增加主机内存与运行时间；尚未做异步预取或声称加速。前向完整logits还在，峰值的可降低幅度有下限。结果必须填入本节或后续同tag续记，不以CPU的数值一致推断完整方法work。新增控制器 `tools/run_cpu_saved_logits_followup_20261010.py`，队列 `outputs/cpu-saved-logits-followup-20261010-1545/`，等待原队列结束，避免干扰主实验。

**元数据假设的限定。** AF的实际小CUDA free预检查中，maxq/scale/zero合计仅520 B，所在allocator segment为small pool 2 MiB；它没有证明这些量能解释此前500 MiB量级large-pool残留。真实模块snapshot需要辨认权重、H/Hinv、Fisher等storage的所属segment。如果实际参数也落在small pool，应否定“这三个元数据钉住large segment”的解释，不能以prefix峰值下降替代归因证据。

### AI. 第二对重复通过与跨重复审计补充（2026-10-10 16:04）

**方法与文献。** 延续AG的同一checkpoint＋CE工程组合，无新增参数或量化目标；第二对交换candidate/control运行顺序。为避免只证明一对内一致却遗漏固定seed的跨次变化，新增只读工具 `tools/audit_memory_repeatability_20261010.py`，对完成的所有臂核对fresh cache SHA、完整state、28层fingerprint、KL/PPL及算法命令，按文件stat前后稳定性保护读取，hardlink只节省重复读取，不改变工件。

**实际效果。** 主tag的第二对已完成：exactness/full_model/strict_measured_memory_gate均true；对应区间与边界回升均0，NVML仍6461→4247 MiB（−34.267%）。两对的四次state SHA均 `7ad9b1a7a890faf5175b136b0c12ea30228bea96060a84f9ee734cd2f0c57ff3`，KL/PPL均0.30534985661506653 / 25.013607025146484。第三对正在执行，尚无3/3结论。

**审计工具的已完成核对。** 使用此前独立完成的卸载＋去重3对实际工件进行检查，6臂的全部fresh缓存、state、28层输出、KL/PPL、算法命令跨重复相同，valid=true；证据 `outputs/cross-repeat-offload-dedup-20261010-1605.json`。该证据属于旧microbatch1研究，不替代当前microbatch2第三对；新完整组合完成后仍需独立审查。

**评价和反思。** 两个交换顺序配对都重复了正向峰值结果，尚需第三对、1.7B、低开销速度及更大工作集。保持所有原值，不去掉极端值，也不把固定seed的重复当多seed质量泛化。CPU logits preflight另以与生产相同的CUBLAS配置/确定性设置重跑CPU两例，仍精确一致（`outputs/cpu-saved-logits-kl-deterministic-cpu-fidelity-20261010-1600.json`），其CUDA/完整量化仍待验证。源码留档 `outputs/runtime-sources-checkpoint-ce-logits-20261010-1600.zip`，468文件CRC与SHA验证通过，archive SHA `809be35a8f537547aff7251890741c2c41f2cfdb0d29bd1e1cb73ab2d0c7f4ad`；跨重复工具产生于该快照之后，应单独保存。

### AJ. 未消费整数导出副本的按需分配（2026-10-10，代码审查原型，未测GPU收益）

**瓶颈与文献。** 审查 `GPTQPlus.fasterquant` 发现，即使 `export_to_et=False` 且 `diagnostic_recorder=None`，仍创建完整W_int_final/Scale_final及各子组W_int_sub/Scale_sub；它们是整数码与scale的导出/诊断副本，不参与Q、Hinv或梯度补偿的数学计算。参考Capuchin按tensor消费者和生命周期管理资源的原则，尝试在没有消费者时不物化这些副本。不是复现其自动调度器，也不是新的量化算法。

**方法。** 研究工具 `tools/export_buffer_lifetime_prototype.py` 对原方法源码做有明确匹配计数的局部转换：每个边界必须恰好匹配一次，否则拒绝；不改核心生产源码。候选仅在不导出、无诊断消费者时省掉完整和子组副本，保留逐列整数码计算、block W_int1/Scale1临时量、Q写回、量化决策、闭式补偿与Adam。导出或diagnostic消费者存在时仍分配原数组。原模型真实权重、GEMM形状及列执行顺序都保持原样；生成后的实际方法源码保存在worker工件中。flag `--research_stage2_lazy_export_buffers`。

**规模动机（估算，不是峰值效果）。** full两副本加全部subgroup两副本，总计约4mn个FP32元素，即16mn B；Qwen3-0.6B FFN权重3072×1024约48 MiB，1.7B权重6144×2048约192 MiB，随权重矩阵增长4倍。这个数字是被省略数组的payload估算，不是完整流程省48/192 MiB的实测，也不包括必要的block临时量。allocator/峰值位置可能使最终收益更小、为零或出现回升，因此仍需严格配对。

**检查与排队。** 原方法源码已通过结构转换和compile，`outputs/lazy-export-source-check-20261010-1618.py`；没有CUDA运行证据。先用真实GPTQPlus API的小CUDA案例分别检查非导出/导出路径的weight、整数码、scale及quantizer精确一致。通过后先在1.7B layer0做共享HCPU的独立消融，再比较原control与HCPU＋lazy副本；前一项不通过，不扩大同一组合。另在已经完整接受的0.6B checkpoint＋CE基础上做fresh全28层独立组件配对，检查统计、state、逐层输出、KL/PPL及所有已测资源门槛。工具 `tools/run_lazy_export_buffer_followup_20261010.py`，当前队列 `outputs/lazy-export-buffer-followup-20261010-1628/` 等待CPU logits实验退出，只做串行GPU运行。

**评价与反思。** 比完成block CPU→GPU迁移更少搬运的潜在工程方案，但目前只有源码审查。它只覆盖常规fasterquant路径；其他batch/group并行路径若仍创建原副本，不能宣称已优化。GPU小API不覆盖实际Adam轨迹，不能替代完整模型验证。即使该组件局部节省，也必须查看是否压低新全程峰值；不因数组估算较大就直接宣称work或省显存10%。

**工程留档。** 初始等待器尚未启动GPU，随后补充事后只读model hardlink整理并重新启动新版等待器，旧等待状态保留 `outputs/lazy-export-buffer-followup-20261010-1625/status.json`（SIGTERM、jobs为空）；该中断不作为方法失败。只链接已完成且FP模型SHA一致的prepared文件，独立Fisher与输出工件不动。后续日志转发器等待前一写入者退出后继续相同总日志，保持单个实时入口。

### AK. 整块重算＋CE组合三对完成、跨重复审计与新峰值定位（2026-10-10 16:37—16:55）

**理论支持与方法。** 沿用AG的组合，没有新增量化目标或参数：按Checkmate的重算思想在真实FSDP统计收集期使用非重入整块checkpoint，保留模块对象、Fisher hooks及原梯度；评估CE分块控制词表维临时张量存活。参考方法的具体适配、CUDA分支一致性修复与负向组件见V/W/AG。REAL-Q是被改进对象，不当外部参考文献。新增的是结果审计与资源证据整理，未修改运行中的核心源码。

**实际设置。** 主tag `qwen06b-checkpoint-ce-original-input-full-threepairs-20261010-1457`；WSL、RTX4060 Laptop 8GiB、Qwen3-0.6B，W4A16/per-row、g4/b128、seed44、不旋转；校准2×1024、refresh/final microbatch2、评估2×1024；两侧共同offload_inps且保持原GPU输入缓存初始化。全部28层实际量化；三对顺序C→N、N→C、C→N。该配置是本地pilot，不是论文正式2048样本规模。

**实际效果。** campaign在16:37:26完成3/3且valid=true。每对13份fresh缓存、507个状态tensor、全部28层held-out输出、最终KL/PPL精确相同，每对2961个对应阶段项、区间回升0、同步边界回升0。三次NVML整卡峰值全部6461→4247 MiB（−34.2671%）；完整进程allocated全部5,760,737,792→3,304,098,304 B（5493.87→3151.03 MiB，−42.6445%），reserved全部6,121,586,688→3,659,530,240 B（5838→3490 MiB，−40.2193%）。保留全部原值，无剔除极端值。进程树RSS峰值三次均值6229.31→6145.55 MiB，配对减少比例均值1.35%，范围−0.0574%到3.1110%；不宣称稳定主机内存节省。

**跨重复正确性。** 另用只读审计工具核对六臂的78个独立缓存文件、完整权重state、28层输出、KL/PPL及算法命令，valid=true、errors=[]。六臂state均 `7ad9b1a7a890faf5175b136b0c12ea30228bea96060a84f9ee734cd2f0c57ff3`；KL/PPL均 `0.30534985661506653 / 25.013607025146484`。这是固定seed的重复一致性，不是多seed/跨数据泛化结论。

**峰值位置及弱势。** 三次control完整allocated峰值都由Stage-0 `static_fisher_batch_backward`捕获；三次candidate完整allocated/reserved峰值都由Stage-2 `layer_27_mlp_down_proj_after_hessian_inverse`捕获，profile区间峰值等于独立累计的完整进程高水位。因此Stage-0改善后，真正的新峰值是末层down_proj Hessian逆构造附近；Stage-2两侧峰值完全相同，不得写成该阶段也已下降。后续H/可选副本生命周期方案有明确针对性；logits保存位置方案只作独立机制检查，若不降低新全程峰值，不堆叠为主成果。

**平均与速度限制。** NVML最大采样间隔约1.19—1.21秒，触发预先设定的时间积分拒绝，平均占用尚未知，不能拿run peak均值或sample mean替代平均显存。当前诊断耗时包含同步/阶段profile/输出指纹/工件I/O，不作为速度证据。已测区间与边界无回升，不证明连续每一瞬间candidate都低于control；保持原强门槛，不放宽缺口阈值。

**评价、反思和下一步。** 这是可重复的完整单模型峰值工程收益，主机成本明显小于卸载＋去重方案；仍需相同工作量1.7B、低开销速度、正式规模及多卡。按新峰值优先研究Hessian与未消费副本的生命周期，拒绝数值一致但reserved回升的组件。1.7B机制配对已串行开始，未完成结果不提前归为成功。

**证据。** 原campaign `outputs/memory_campaign_qwen06b-checkpoint-ce-original-input-full-threepairs-20261010-1457/`；跨重复 `outputs/cross-repeat-checkpoint-ce-threepairs-20261010-1650.json`；每次原值/峰值位置 `outputs/summary-qwen06b-checkpoint-ce-threepairs-20261010-1650/`；PNG/PDF/SVG图及来源SHA `outputs/figures-qwen06b-checkpoint-ce-threepairs-20261010-1653/`，图已目视检查。代码留档 `outputs/runtime-sources-lazy-export-20261010-1635.zip`，474文件CRC/SHA验证，SHA `626b7ebaa7b11a764fd7792eef7b9494fea32135949afd9c6ba550346208856f`；其中logits/lazy副本原型尚无CUDA模型结果，快照不表示它们已work。

### AL. 量化元数据生命周期消融：正确但未压低峰值（2026-10-10 16:55）

**理论与待检验假设。** 依据Capuchin的tensor生命周期管理及GMLake的不规则分配/碎片化问题，检查Hessian卸载后仍驻留的小量化参数是否占住大segment。此前小API显示2MiB small-pool，已明确不能据此解释约500MiB的大segment；实际模型快照用于证伪或支持这个归因。没有实现两篇论文的完整调度器/分配器。

**方法和配置。** tag `qwen17b-quantizer-metadata-cpu-layer0-2x512-20261010-v3-1457`；fresh两臂Stage-0、Qwen3-1.7B、2×512校准、microbatch1、W4A16、仅量化layer0但前向评估全部28层。两侧共同Stage-0卸载4MiB＋去重及Hessian CPU暂存；候选仅在module完成后将maxq/scale/zero搬至CPU，逐字节核对七模块保存参数。这个消融不改变分组、补偿或Adam。它不是完整28层量化成果。

**实际效果。** 数值、缓存、state、28层held-out输出和质量精确一致；KL/PPL `0.01714225485920906 / 15.934643745422363`。metadata byte audit七模块一致，所有已测区间与边界回升0；但NVML峰值4283→4283 MiB、reserved峰值3,697,278,976→3,697,278,976 B，完整allocated仅3,612,503,552→3,612,353,024 B（减少150,528 B）。不把valid=true等同于有实质显存收益。

**机制审查。** 七模块的参数payload共163,896 B，真实snapshot中它们全部位于2,097,152 B small-pool segment，不位于大Hessian segment。down_proj完成后inactive_split约647,263,744→647,431,168 B，reserved仍相同：释放小参数后，它们只是变为尚未释放的小池空闲块。大碎片不能归因于这些参数。payload与peak差值不同也符合不同时间点的存活集合和allocator对齐，不能由小参数payload推断完整峰值下降。

**评价、反思及下一步。** 正确性通过、峰值收益为零，是对一个具体机制的负向结果。不重复相同独立消融或直接扩大全模型。另一次原control与HCPU＋元数据CPU组合已启动，用于核对原reserved回升是否修复；它若仍回升，保留失败并转向有真实大segment owner快照的completed-block驻留重整或无需搬运的可选导出副本消除。独立组合必须通过自己的门槛，不能用本组通过来替代。证据：对应campaign `pair1_audit/comparison.json`、`metadata_byte_audit.json`、两臂 `quantizer_metadata_lifetime_rank0.json`。

### AM. 控制器修复、真实CUDA前检及大segment所有者核对（2026-10-10 17:14—17:43）

**负向组合与归因。** 原control对照tag `qwen17b-hessian-metadata-cpu-original-control-layer0-20261010-v3-1457`数值完全一致、NVML4839→4283 MiB，但layer0重放区间reserved多499,122,176 B（476MiB），边界多549,453,824 B（524MiB），严格拒绝；元数据卸载没有修复大碎片。不因完整峰值下降11.49%而忽略该回升。

**测试/工程bug修复。** post-v3在启动任何GPU作业前因相对output路径无法转换Windows路径而退出（jobs=[]）；改为resolve绝对工作区路径，新队列v4启动，旧状态保留。CPU logits的1.7B前检实际上因要求单个safetensors文件而在加载时退出，没有测出梯度不一致；修复按index定位真实lm_head分片、限制路径在模型目录、记录index和head SHA。小GPTQ导出前检误传4×32梯度，真实权重是64×32；修复输入尺寸，不改生产公式。原始失败日志不删除，也不把入口错误写成方法数学失败。

**已完成CUDA正确性。** 修复后普通、整数导出及diagnostic三种真实GPTQPlus API消费者的C/N权重、整数码、scale和quantizer全部精确一致，`outputs/lazy-export-cuda-api-v2-20261010-1719/status.json`。1.7B原训练head/2048行CUDA测试loss及hidden梯度精确一致、差值0，`outputs/cpu-saved-logits-qwen17-sharded-cuda-fidelity-20261010-1723.json`；0.6B此前也通过。隐藏向量为合成fixture，小API不包含实际Adam，均不能代替完整模型或资源证据。原型helper未因前检修复改变。

**大segment机制实测。** tag `qwen17b-hessian-layer-residency-independent-layer0-20261010-v4-1720`，两侧共享HCPU及2×512/microbatch1，只量化layer0、全部28层评估。数值、缓存、state、28层输出、KL/PPL精确相同，KL/PPL0.01714225485920906 / 15.934643745422363；已测区间/边界无回升。两侧NVML4283MiB、全程allocated3,612,503,552 B、reserved3,697,278,976 B相同，故尚无完整峰值收益。

**生命周期方法与文献。** 根据Capuchin的消费者/驻留原则和GMLake的碎片问题，在完成block、无活跃反向时将该block CPU→GPU重新驻留，中间empty_cache释放不再含活跃权重的segment。真实快照发现down_proj权重占住603,979,776 B（576MiB）segment，578,813,952 B（552MiB）inactive。重整后block边界allocated不变1,464,568,320 B，reserved2,111,832,064→1,482,686,464 B（2014→1414MiB）、inactive_split647,263,744→18,118,144 B。权重/缓冲/梯度字节、shape/stride/dtype/requires_grad及Parameter身份均精确相同；一次边界诊断copy约0.0533s，不作为正式速度数据。

**评价及后续。** 这次实测支持大碎片由新写回的down_proj权重驻留引起，而不是小quantizer参数。它证明边界碎片修复，不证明全程峰值降低；正在与原control做fresh独立组合，只有该组合无回升才扩全层。另有同工作量1.7B checkpoint＋CE完整三对及0.6B低开销速度测试排在后面。新的串行队列 `outputs/post-memory-v4-followup-20261010-1720/`，随后logits-v2、lazy-v3。所有新内容仍按数值/资源门槛记录，不扩未通过方案。

**18:02已完成原control组合核对。** `qwen17b-hessian-layer-residency-original-control-layer0-20261010-v4-1720` 的fresh prefix配对exactness及strict_measured_memory_gate均true，区间/边界回升均0，NVML4839→4283MiB（−11.49%）。它修复了同配置HCPU＋元数据组合的476/524MiB回升，但仅为layer0量化、全部28层评估证据。按原门槛扩展完整1.7B checkpoint＋HCPU＋驻留重整＋CE三对，2×512/microbatch1；18:32首对control完成14/28层、正在layer14，未完成的全模型结果不提前推广。匹配0.6B工作量的2×1024/microbatch2完整1.7B实验仍是独立后续组。

### AN. 1.7B驻留组合全模型被拒绝、独立方向中断后重启（2026-10-10 21:20）

**方法、依据与完整结果。** AM的Hessian卸载＋完成block驻留重整＋Stage-0 checkpoint＋CE在 `qwen17b-checkpoint-hessian-compact-ce-full-threepairs-20261010-v4-1720` 完成首对全部28层，fresh 2×512/microbatch1。全部缓存、state、28层输出及KL/PPL精确一致（0.7481881380081177 / 32.389034271240234），但严格资源门槛失败，20:08:21按协议停止1/3，不继续同一失败方案。NVML全程5313→4741MiB（−10.766%），allocated4,489,012,736→3,733,775,872 B、reserved4,777,312,256→4,177,526,784 B下降，仍不能忽略538个区间、541个边界回升。最大reserved区间回升位于layer24 o_proj Hessian分配前，+121,634,816 B（116MiB），allocated相同。

**评价与反思。** Capuchin/GMLake提供生命周期和碎片的工程依据，但局部layer0重整成功没有控制全28层的分配历史；新的活跃tensor布局会改变后续小/大池segment组合。数值正确、完整峰值下降并不足以满足本研究更强的逐区间门槛。该组合不是成功的两模型推广，不能按三对成果写入终稿。后续若再研究它，须先证明回升phase的真实segment所有者，并改变设计；不重复当前未修改方案。完整失败audit和所有工件保留。

**中断事实。** 控制器随后自动进入独立匹配工作量的1.7B checkpoint＋CE（2×1024/microbatch2、相同0.6B工作量、无HCPU驻留组合）。control最后日志20:42:00，layer14 up_proj；21:18检查无worker/controller、GPU0MiB，WSL新boot ID `33ec080f-7597-4813-8eac-0ad94a8d533b`。无完整status、无配对audit。具体退出原因未知，不能据此称CUDA OOM或方法失败，也不能把半程数据当完整结果。

**修复与继续。** 控制器增加显式skip-hessian-repair，仅跳过已经完成且失败的Hessian组合，不改变原量化参数、loss、Adam或生产代码。新队列 `outputs/post-memory-v5-followup-20261010-2120/` 用新tag后缀`-v5-2120`重新生成统计，从头做独立1.7B完整三对，然后0.6B Triton完整精确性检查及低开销速度三对；后续logits-v3、lazy-v4等待该队列，单GPU串行。旧中断工件保留，不复用半程量化状态。只读整理已完整且SHA相同的prepared FP副本以保留磁盘空间，fresh统计不合并、不计GPU收益。记录 `outputs/restart-audit-20261010-2120/interruption_and_failed_method.json`；实时总日志入口不变。

<!-- completed-memory-followup:qwen06b-checkpoint-ce-original-input-full-threepairs-20261010-1457 -->
### 自动续记：qwen06b-checkpoint-ce-original-input-full-threepairs-20261010-1457（2026-10-10 17:14:33，已完成审计）

**理论支持、文献与具体方法。** 本轮配置与改动以保存manifest为准；checkpoint/CE组合延续Z/AD/AF节的重算和分块方法，元数据或完成block迁移延续AB节生命周期假设。[Korthikanti等](https://arxiv.org/abs/2205.05198)、[Checkmate](https://arxiv.org/abs/1910.02653)、[GMLake](https://arxiv.org/abs/2401.08156)支持资源调度/碎片化分析，不表示实现其所有算法。

**实际配置与效果。** 模型 `/mnt/d/llamaModels/Qwen3-0.6B`，校准 `2×1024`，refresh microbatch `2`，量化截止层 `27`。已审计 `3/3` 对，完整模型量化验证 `True`，campaign valid `True`。

| 指标 | control均值 | candidate均值 |
| --- | ---: | ---: |
| 整卡采样peak MiB | 6461.0000 | 4247.0000 |
| 完整进程allocated peak MiB | 5493.8677 | 3151.0337 |
| 完整进程reserved peak MiB | 5838.0000 | 3490.0000 |
| CPU RSS peak MiB | 6229.3125 | 6145.5521 |

首对KL/PPL C `0.30534985661506653 / 25.013607025146484`，N `0.30534985661506653 / 25.013607025146484`。每对数值门槛 `[True, True, True]`，资源门槛 `[True, True, True]`；区间回升数量 `[0, 0, 0]`、边界回升数量 `[0, 0, 0]`。

**评价与反思。** 所有请求配对通过已测门槛；仍只覆盖保存的本地pilot规模，不是正式论文规模/4卡证明。 prefix不能称为完整量化；NVML为采样，不能证明连续每一时刻。这里的均值是各次peak的均值，不是时间平均显存。诊断计时不能支持加速。原因分析需结合原始profile和代码，不以这个自动摘要替代人工审查。

**证据及后续。** `outputs/memory_campaign_qwen06b-checkpoint-ce-original-input-full-threepairs-20261010-1457/`、`outputs/completed-memory-records-20261010-1526/qwen06b-checkpoint-ce-original-input-full-threepairs-20261010-1457/summary.json`及原始每对comparison。完整量化与系统资源通过后才进入相同源码的低开销计时；尚未通过则先定位，不继续重复同一失败配置。

<!-- completed-memory-followup:qwen17b-quantizer-metadata-cpu-layer0-2x512-20261010-v3-1457 -->
### 自动续记：qwen17b-quantizer-metadata-cpu-layer0-2x512-20261010-v3-1457（2026-10-10 17:14:34，已完成审计）

**理论支持、文献与具体方法。** 本轮配置与改动以保存manifest为准；checkpoint/CE组合延续Z/AD/AF节的重算和分块方法，元数据或完成block迁移延续AB节生命周期假设。[Korthikanti等](https://arxiv.org/abs/2205.05198)、[Checkmate](https://arxiv.org/abs/1910.02653)、[GMLake](https://arxiv.org/abs/2401.08156)支持资源调度/碎片化分析，不表示实现其所有算法。

**实际配置与效果。** 模型 `/mnt/d/llamaModels/Qwen3-1.7B`，校准 `2×512`，refresh microbatch `1`，量化截止层 `0`。已审计 `1/1` 对，完整模型量化验证 `False`，campaign valid `True`。

| 指标 | control均值 | candidate均值 |
| --- | ---: | ---: |
| 整卡采样peak MiB | 4283.0000 | 4283.0000 |
| 完整进程allocated peak MiB | 3445.1519 | 3445.0083 |
| 完整进程reserved peak MiB | 3526.0000 | 3526.0000 |
| CPU RSS peak MiB | 13859.3672 | 13876.7852 |

首对KL/PPL C `0.01714225485920906 / 15.934643745422363`，N `0.01714225485920906 / 15.934643745422363`。每对数值门槛 `[True]`，资源门槛 `[True]`；区间回升数量 `[0]`、边界回升数量 `[0]`。

**评价与反思。** 所有请求配对通过已测门槛；仍只覆盖保存的本地pilot规模，不是正式论文规模/4卡证明。 prefix不能称为完整量化；NVML为采样，不能证明连续每一时刻。这里的均值是各次peak的均值，不是时间平均显存。诊断计时不能支持加速。原因分析需结合原始profile和代码，不以这个自动摘要替代人工审查。

**证据及后续。** `outputs/memory_campaign_qwen17b-quantizer-metadata-cpu-layer0-2x512-20261010-v3-1457/`、`outputs/completed-memory-records-20261010-1526/qwen17b-quantizer-metadata-cpu-layer0-2x512-20261010-v3-1457/summary.json`及原始每对comparison。完整量化与系统资源通过后才进入相同源码的低开销计时；尚未通过则先定位，不继续重复同一失败配置。

<!-- completed-memory-followup:qwen17b-hessian-metadata-cpu-original-control-layer0-20261010-v3-1457 -->
### 自动续记：qwen17b-hessian-metadata-cpu-original-control-layer0-20261010-v3-1457（2026-10-10 17:14:34，已完成审计）

**理论支持、文献与具体方法。** 本轮配置与改动以保存manifest为准；checkpoint/CE组合延续Z/AD/AF节的重算和分块方法，元数据或完成block迁移延续AB节生命周期假设。[Korthikanti等](https://arxiv.org/abs/2205.05198)、[Checkmate](https://arxiv.org/abs/1910.02653)、[GMLake](https://arxiv.org/abs/2401.08156)支持资源调度/碎片化分析，不表示实现其所有算法。

**实际配置与效果。** 模型 `/mnt/d/llamaModels/Qwen3-1.7B`，校准 `2×512`，refresh microbatch `1`，量化截止层 `0`。已审计 `1/1` 对，完整模型量化验证 `False`，campaign valid `False`。

| 指标 | control均值 | candidate均值 |
| --- | ---: | ---: |
| 整卡采样peak MiB | 4839.0000 | 4283.0000 |
| 完整进程allocated peak MiB | 4021.1519 | 3445.0083 |
| 完整进程reserved peak MiB | 4082.0000 | 3526.0000 |
| CPU RSS peak MiB | 13878.6602 | 13881.3320 |

首对KL/PPL C `0.01714225485920906 / 15.934643745422363`，N `0.01714225485920906 / 15.934643745422363`。每对数值门槛 `[True]`，资源门槛 `[False]`；区间回升数量 `[1]`、边界回升数量 `[1]`。

**评价与反思。** 本轮未通过完整请求/严格门槛；即使局部peak下降也不能作为合格最终方案，失败配对保留，不剔除。 prefix不能称为完整量化；NVML为采样，不能证明连续每一时刻。这里的均值是各次peak的均值，不是时间平均显存。诊断计时不能支持加速。原因分析需结合原始profile和代码，不以这个自动摘要替代人工审查。

**证据及后续。** `outputs/memory_campaign_qwen17b-hessian-metadata-cpu-original-control-layer0-20261010-v3-1457/`、`outputs/completed-memory-records-20261010-1526/qwen17b-hessian-metadata-cpu-original-control-layer0-20261010-v3-1457/summary.json`及原始每对comparison。完整量化与系统资源通过后才进入相同源码的低开销计时；尚未通过则先定位，不继续重复同一失败配置。

<!-- completed-memory-followup:qwen17b-checkpoint-hessian-compact-ce-full-threepairs-20261010-v4-1720 -->
### 自动续记：qwen17b-checkpoint-hessian-compact-ce-full-threepairs-20261010-v4-1720（2026-10-10 21:21:40，已完成审计）

**理论支持、文献与具体方法。** 本轮配置与改动以保存manifest为准；checkpoint/CE组合延续Z/AD/AF节的重算和分块方法，元数据或完成block迁移延续AB节生命周期假设。[Korthikanti等](https://arxiv.org/abs/2205.05198)、[Checkmate](https://arxiv.org/abs/1910.02653)、[GMLake](https://arxiv.org/abs/2401.08156)支持资源调度/碎片化分析，不表示实现其所有算法。

**实际配置与效果。** 模型 `/mnt/d/llamaModels/Qwen3-1.7B`，校准 `2×512`，refresh microbatch `1`，量化截止层 `27`。已审计 `1/3` 对，完整模型量化验证 `True`，campaign valid `False`。

| 指标 | control均值 | candidate均值 |
| --- | ---: | ---: |
| 整卡采样peak MiB | 5313.0000 | 4741.0000 |
| 完整进程allocated peak MiB | 4281.0562 | 3560.8062 |
| 完整进程reserved peak MiB | 4556.0000 | 3984.0000 |
| CPU RSS peak MiB | 11974.3906 | 11981.2812 |

首对KL/PPL C `0.7481881380081177 / 32.389034271240234`，N `0.7481881380081177 / 32.389034271240234`。每对数值门槛 `[True]`，资源门槛 `[False]`；区间回升数量 `[538]`、边界回升数量 `[541]`。

**评价与反思。** 本轮未通过完整请求/严格门槛；即使局部peak下降也不能作为合格最终方案，失败配对保留，不剔除。 prefix不能称为完整量化；NVML为采样，不能证明连续每一时刻。这里的均值是各次peak的均值，不是时间平均显存。诊断计时不能支持加速。原因分析需结合原始profile和代码，不以这个自动摘要替代人工审查。

**证据及后续。** `outputs/memory_campaign_qwen17b-checkpoint-hessian-compact-ce-full-threepairs-20261010-v4-1720/`、`outputs/completed-memory-v5-records-20261010-2120/qwen17b-checkpoint-hessian-compact-ce-full-threepairs-20261010-v4-1720/summary.json`及原始每对comparison。完整量化与系统资源通过后才进入相同源码的低开销计时；尚未通过则先定位，不继续重复同一失败配置。

### AN续记：21:28恢复队列与证据边界

**中断核对。** WSL boot ID已改变，原控制器及GPU worker均不存活；1.7B匹配工作量checkpoint＋CE的旧control日志止于layer14，没有完成status，不能把它算作正向或负向结果。没有证据把中断归因于CUDA OOM。原HCPU＋完成block重整全28层首对数值全部精确，但538个区间和541个边界资源回升，已判失败，不再重复该组合。

**实际恢复。** v5控制器在启动GPU前遇到27GiB磁盘余量保护，jobs为空。对已完成的、SHA一致的prepared FP文件做只读hardlink整理，保留两个路径及独立Fisher/stat工件，磁盘余量恢复约30GiB。两个旧1.7B副本整理成功；第三个由于Windows只读文件替换限制失败，原文件保留。这是磁盘维护，不计GPU优化成果。

**当前工作。** CPU保存KL logits的真实CUDA训练head检查在0.6B、1.7B均通过，loss与hidden梯度精确一致；随后实际启动0.6B完整28层新鲜独立配对，21:27 control运行到layer7。预检通过不等于完整量化收益通过。此配对完成之后，v6串行控制器先做独立的1.7B匹配工作量checkpoint＋CE全28层最多三对：两模型同2×1024校准、microbatch2、eval2×1024，移除已失败的HCPU/驻留重整组合。任一数值或资源门槛失败停止该配置重复，然后另行核对0.6B Triton组合与低开销速度。可选导出缓冲生命周期实验排在v6队列结束后，避免多个GPU实验互相干扰。

**位置。** 当前GPU队列 outputs/cpu-saved-logits-followup-v3-20261010-2120；后续 outputs/post-memory-v6-followup-20261010-2125；可选缓冲 outputs/lazy-export-buffer-followup-v5-20261010-2125。完成记录器 outputs/completed-memory-v6-records-20261010-2125。单一总日志仍为 logs/realq-system-restart-20261010-1408.log，原始worker日志保留；转发器重启的历史文本重放不算新实验。

### AO：以设备可运行门槛为首要目标（2026-10-10）

**原文核对。** REAL-Q第7节及附录E(3)明确把系统显存future work联系到Stage-0全精度模型临时驻留、逐层O(d²) Fisher与反向过程；D.1说明70B采用FSDP端到端预计算并缓存到磁盘，随后分层量化；D.6区分Fisher静态缓存和Stage-0反向瞬态，且70B已经采用activation checkpointing。原文 https://arxiv.org/html/2609.00049 。以上为研究对象的动机，不当作外部方法文献。

**工程目标及评价。** 优先降低在目标模型和正式校准配置下，决定最低每卡显存或最少GPU数的阶段；先分别报告Stage-0与正式量化的每rank峰值，并以两者最大值作为生成量化模型的资源门槛。离线KL/PPL评估另外报告，不将仅评估节省当作降低REAL-Q校准/量化门槛。完整执行仍需评估可运行。最强容量证据是固定模型/样本/序列/量化目标/质量与并行协议，在更少GPU或更低显存预算上baseline无法完整执行而candidate可以，且报告CPU RAM、通信和时间代价。未做这样的边界实验不得宣称实际减少GPU数量。任何已测时段显存不回升是用户要求的额外严格门槛，不伪称论文原文规定。

**反思。** 0.6B短配置用于机制和正确性，不以其峰值阶段替代大模型正式配置瓶颈；不同规模的Hessian、参数驻留和保存激活增长不同。已有checkpoint方案须与启用原论文checkpoint的公平基线对照，不把已有功能开关计作新算法。1.7B匹配工作量完整实验完成后，再决定主要资源扩展组件；速度作为次级指标，离线评估作为辅助工程完整性。

<!-- completed-memory-followup:qwen06b-cpu-saved-kl-logits-checkpoint-ce-full-20261010-1545 -->
### 自动续记：qwen06b-cpu-saved-kl-logits-checkpoint-ce-full-20261010-1545（2026-10-10 21:55:22，已完成审计）

**理论支持、文献与具体方法。** 本轮是AH节logits保存位置的独立消融：两臂共同checkpoint＋CE，候选仅把KL节点保存的student/teacher logits转到CPU，反向逐token块恢复，并保留原始完整LM-head前向/反向GEMM形状。参考[Capuchin](https://alchem.usc.edu/portal/static/download/capuchin.pdf)的tensor生命周期管理与[CCE](https://arxiv.org/abs/2411.09009v2)对词表中间量开销的分析；不实现其自动调度器或融合CE算法。CPU搬运代价及调用者额外持有logits的影响必须另外核对，不预设能够加速。

**实际配置与效果。** 模型 `/mnt/d/llamaModels/Qwen3-0.6B`，校准 `2×1024`，refresh microbatch `2`，量化截止层 `27`。已审计 `1/1` 对，完整模型量化验证 `True`，campaign valid `True`。

| 指标 | control均值 | candidate均值 |
| --- | ---: | ---: |
| 整卡采样peak MiB | 4247.0000 | 3653.0000 |
| 完整进程allocated peak MiB | 3151.0337 | 2570.9429 |
| 完整进程reserved peak MiB | 3490.0000 | 2896.0000 |
| CPU RSS peak MiB | 6237.8555 | 6507.6523 |

首对KL/PPL C `0.30534985661506653 / 25.013607025146484`，N `0.30534985661506653 / 25.013607025146484`。每对数值门槛 `[True]`，资源门槛 `[True]`；区间回升数量 `[0]`、边界回升数量 `[0]`。

**评价与反思。** 所有请求配对通过已测门槛；仍只覆盖保存的本地pilot规模，不是正式论文规模/4卡证明。 prefix不能称为完整量化；NVML为采样，不能证明连续每一时刻。这里的均值是各次peak的均值，不是时间平均显存。诊断计时不能支持加速。原因分析需结合原始profile和代码，不以这个自动摘要替代人工审查。

**证据及后续。** `outputs/memory_campaign_qwen06b-cpu-saved-kl-logits-checkpoint-ce-full-20261010-1545/`、`outputs/cpu-saved-logits-followup-v3-20261010-2120/completed-records/qwen06b-cpu-saved-kl-logits-checkpoint-ce-full-20261010-1545/summary.json`及原始每对comparison。完整量化与系统资源通过后才进入相同源码的低开销计时；尚未通过则先定位，不继续重复同一失败配置。

### AO续记：仅优先硬件门槛阶段（用户最新约束）

用户明确同意：其他阶段未触碰硬件门槛可先不处理。1.7B匹配工作量checkpoint＋CE的完整配对已经实际启动，保留其GPU worker继续完成；撤下尚未启动的可选导出缓冲队列，并通过 tools/defer_pending_memory_queue_20261010.py 在独立campaign执行期间只停止后续调度的父控制器，绝不停止GPU子进程/进程组。等当前campaign退出后关闭旧调度器，保留原campaign status/audit，标记后续Triton/纯速度与可选缓冲为因优先级调整而deferred，不计方法失败。该控制器退出动作不改变任何模型、损失、校准或量化源码，也不是新的GPU优化贡献。

当前优先：从1.7B同工作量完整配对获取Stage-0与正式量化每阶段峰值，确定控制设备可运行边界的阶段。离线评估与无门槛压力的阶段暂不展开实验。已有0.6B CPU保存logits独立配对已完成，其结果留档，但不因单个组件通过就自动扩大重复。

<!-- completed-memory-followup:qwen17b-checkpoint-ce-matched-full-threepairs-20261010-v6-2125 -->
### 自动续记：qwen17b-checkpoint-ce-matched-full-threepairs-20261010-v6-2125（2026-10-11 01:46:21，已完成审计）

**理论支持、文献与具体方法。** 本轮配置与改动以保存manifest为准；checkpoint/CE组合延续Z/AD/AF节的重算和分块方法，元数据或完成block迁移延续AB节生命周期假设。[Korthikanti等](https://arxiv.org/abs/2205.05198)、[Checkmate](https://arxiv.org/abs/1910.02653)、[GMLake](https://arxiv.org/abs/2401.08156)支持资源调度/碎片化分析，不表示实现其所有算法。

**实际配置与效果。** 模型 `/mnt/d/llamaModels/Qwen3-1.7B`，校准 `2×1024`，refresh microbatch `2`，量化截止层 `27`。已审计 `3/3` 对，完整模型量化验证 `True`，campaign valid `True`。

| 指标 | control均值 | candidate均值 |
| --- | ---: | ---: |
| 整卡采样peak MiB | 7925.0000 | 6629.0000 |
| 完整进程allocated peak MiB | 7158.8696 | 5605.1909 |
| 完整进程reserved peak MiB | 7558.0000 | 5872.0000 |
| CPU RSS peak MiB | 11985.0221 | 12002.7513 |

首对KL/PPL C `0.625495195388794 / 25.562191009521484`，N `0.625495195388794 / 25.562191009521484`。每对数值门槛 `[True, True, True]`，资源门槛 `[True, True, True]`；区间回升数量 `[0, 0, 0]`、边界回升数量 `[0, 0, 0]`。

**评价与反思。** 所有请求配对通过已测门槛；仍只覆盖保存的本地pilot规模，不是正式论文规模/4卡证明。 prefix不能称为完整量化；NVML为采样，不能证明连续每一时刻。这里的均值是各次peak的均值，不是时间平均显存。诊断计时不能支持加速。原因分析需结合原始profile和代码，不以这个自动摘要替代人工审查。

**证据及后续。** `outputs/memory_campaign_qwen17b-checkpoint-ce-matched-full-threepairs-20261010-v6-2125/`、`outputs/completed-memory-v6-records-20261010-2125/qwen17b-checkpoint-ce-matched-full-threepairs-20261010-v6-2125/summary.json`及原始每对comparison。完整量化与系统资源通过后才进入相同源码的低开销计时；尚未通过则先定位，不继续重复同一失败配置。

### AP：同一系统方案在两模型上完成三组验证（2026-10-11）

**瓶颈及理论依据。** 按AO的设备门槛优先级，验证Stage-0整块non-reentrant checkpoint与CE精确分块组合。两模型控制组的全程allocated峰值均定位于static_fisher_batch_backward；候选压低该阶段后，最高点均转移至Stage-2末层down_proj的Hessian求逆。外部工程依据继续为Korthikanti等（https://arxiv.org/abs/2205.05198）与Checkmate（https://arxiv.org/abs/1910.02653）。REAL-Q是目标与基线，不计独立参考文献；70B原方案已有checkpoint，不能把开启既有功能包装成新算法。本项目贡献须落在本地统计收集器、FSDP/元数据/钩子生命周期适配、精确分块与系统验证。

**具体方法及公平配置。** 不加新的loss，不改Adam/闭式补偿，不重新调参；candidate仅启用整块重算及CE256分块。两模型均为W4A16/per-row，cal2×1024、refresh microbatch2、eval2×1024，28层完整量化，seed44，单GPU，校准/样本顺序/数据/算法条件在各配对两臂一致。共同KL chunk16、FSDP CPU-offload/meta-init、offload_inps等条件见保存command。早先模板的explicit_shared_conditions仍有microbatch1文字残留，实际microbatch以command中的backward_bsz=2/final_layer_backward_bsz=2及scope_note为准；不能用陈旧描述代替执行参数。这是本地pilot，不是正式2048校准样本，也不是4卡结果。

**实际效果。** 两模型均完整3/3配对、每组cache/state/所有28层held-out输出/KL/PPL精确一致，并通过全程及所有已测对应区间/同步边界不回升门槛。交替配对顺序，未剔除任何结果。

| 模型 | 整卡NVML peak C→N（MiB） | 整卡peak下降 | allocated peak C→N（MiB） | reserved peak C→N（MiB） | 最终KL/PPL（两臂相同） |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3-0.6B | 6461→4247 | 34.27% | 5493.8677→3151.0337 | 5838→3490 | 0.30534985661506653 / 25.013607025146484 |
| Qwen3-1.7B | 7925→6629 | 16.35% | 7158.8696→5605.1909 | 7558→5872 | 0.625495195388794 / 25.562191009521484 |

0.6B三组peak相同，1.7B三组peak也相同。1.7B allocated下降21.70%，reserved下降22.31%；RSS峰值均值11985.02→12002.75 MiB，轻微增加约0.15%，不宣称节省主机内存。1.7B六臂跨重复审计valid=true/errors=[]，全部cache/state/layer/quality一致，73个独立inode内容被实际哈希（六份prepared FP权重经SHA核对后共享只读硬链接，其他Fisher/stat文件均独立）。0.6B跨重复复核见AK。峰值的三次均值不等于时间平均显存；时间平均仍受NVML采样缺口限制；上述诊断耗时不能用于宣称加速。

**评价、反思与下一步。** 现在可作两模型可重复的完整小配置显存成果；原1.7B HCPU＋驻留重整全28层失败不被删除，新的正向结果来自独立checkpoint＋CE方案，不能归因于已失败组件。充分验证还需扩展校准/评估覆盖、第二个数据种子、受限显存预算、精确省显存的时间代价，以及正式配置/实际多卡。下一步不探索纯速度kernel或无硬件压力的缓冲组件。

**证据。** 主0.6Bcampaign与summary见AK。1.7B：outputs/memory_campaign_qwen17b-checkpoint-ce-matched-full-threepairs-20261010-v6-2125/；summary/raw values/peak locations在outputs/completed-memory-v6-records-20261010-2125/qwen17b-checkpoint-ce-matched-full-threepairs-20261010-v6-2125/；跨重复outputs/cross-repeat-qwen17b-checkpoint-ce-threepairs-20261011-0223.json；图目录outputs/figures-qwen17b-checkpoint-ce-threepairs-20261011-0240/。

**两模型公平性补核。** outputs/two-model-main-evidence-20261011-0250.json将两个已完成summary、全部六臂/三对CSV与跨重复审计绑定SHA256，并实际比较两模型各臂Stage-0/Stage-2的算法命令：仅模型路径替换为统一占位符，其余算法/工作量条件归一化后完全相同。每模型三对身份完整且全部门槛通过。归一化沿用compare_full_pipeline_profiles.algorithm_command，系统参数被该函数排除；因此这里证明算法工作量匹配，系统改动仍由原始保存manifest审查。该汇总不重新启动GPU或改写原始工件，也不新增时间平均显存/速度结论。

### AQ：固定分配器预算的完整可运行门槛检查（进行中）

**支持与方法。** PyTorch 2.11的set_per_process_memory_fraction在分配超过visible-total×fraction时触发allocator OOM，官方说明https://docs.pytorch.org/docs/2.11/generated/torch.cuda.memory.set_per_process_memory_fraction.html 。它限制PyTorch分配器，不限制driver/NCCL；不是换成物理小显存卡，也不证明减少GPU数量。预算根据此前完整peak区间预先固定：1.7B 6400MiB、0.6B 4096MiB，两臂相同，不改变校准/量化/质量参数，不扫预算找最好loss。预算安装值、实际fraction、运行源码SHA、OOM阶段、全程峰值及原始日志均保存。候选必须完整量化/评估，且与已有未受限候选的13缓存、state、28层输出及KL/PPL精确一致；baseline必须是明确CUDA OOM，其他错误先停查bug。

**已做与未知。** CUDA API的64MiB预算/128MiB分配冒烟明确触发预期OOM，工件outputs/allocator-capacity-api-check-20261011-0230.json。OOM文本里本机WDDM的non-PyTorch in-use字段出现明显溢出值，不能拿该字段计算显存；使用allocator计数及NVML遥测。真实1.7B control在6400MiB预算下Stage-0已OOM；candidate通过预计算并进入全28层量化，目前未完成，不把预计算通过写为全流程成功。完整脚本tools/run_realq_allocator_capacity_20261011.py，队列outputs/realq-two-model-allocator-capacity-20261011-0228/。

**后续队列。** tools/run_two_model_memory_followup_20261011.py等待容量检查完整通过，随后各模型做cal8×1024/seed45、eval8×1024的独立完整配对，再对已经接受的cal2×1024方案各做三组无逐层hash/phase重置插桩的成本配对，测重算时间代价而非探索新速度kernel。成本结果仍需fresh-cache/state/KL/PPL精确绑定参考；计时结束后做hash，不将整个包含审计的进程wall当纯运行时间。任一新配置失败停止该配置扩大，保留原始负向工件。队列outputs/two-model-memory-followup-20261011-0240/。


<!-- two-model-allocator-capacity:realq-two-model-allocator-capacity-20261011-0228 -->
### AQ续记：两模型固定预算的实测结果

**理论、方法与配置。** 延续AQ的PyTorch allocator限额验证：1.7B6400MiB、0.6B4096MiB，校准/量化/数据与AP已接受参考一致。参考文献及实现见AQ，不是新算法或物理小显存卡实验。

**实际效果。**

| 模型 | 分配器预算MiB | baseline OOM阶段 | candidate完整完成 | 与参考cache/state/28层/KL/PPL精确一致 | 整卡NVML peak MiB |
| --- | ---: | --- | --- | --- | ---: |
| qwen17b | 6400 | ['stage0'] | True | True | 6629 |
| qwen06b | 未执行 | [] | False | False | 未知 |

**评价与反思。** 只有明确CUDA OOM且候选完整完成/精确一致时才算此固定预算可运行对比成立；非OOM错误、未完成或数值不一致均不算成功。预算是分配器限额，driver/NCCL不受该API限制，因此不把它写为物理4GB/6GB卡已经可运行或GPU数已经减少。OOM异常里的WDDM异常in-use数值不用于计量；native CUDA max计数可能被诊断阶段重置，全程峰值以profile status为准。原始失败日志全部保留。

**证据及后续。** /mnt/d/gptq_plus_realq/outputs/realq-two-model-allocator-capacity-20261011-0228；完整摘要/mnt/d/gptq_plus_realq/outputs/two-model-capacity-cost-records-20261011-0240/capacity_summary.json。下一步按已排队方案扩大校准与评估、换数据种子，并测重算时间成本。


<!-- two-model-recomputation-cost:two-model-memory-followup-20261011-0240 -->
### AR：省显存方案的重算成本复测

**理论支持与具体方法。** Checkmate与Korthikanti的重算是用计算换显存，应同时报告成本。这里沿用AP已接受完整方案，不探索Triton。每模型三组交替顺序、fresh-stat校验，collector和正式量化用CUDA同步wall timer，hash在计时外。未删除极端值。

**实际效果。**

| 模型 | 已完成配对 | collector C/N秒 | 量化 C/N秒 | collector＋量化 C/N秒 |
| --- | ---: | --- | --- | --- |
| qwen17b | 0 | 未知 / 未知 | 未知 / 未知 | 未知 / 未知 |
| qwen06b | 0 | 未知 / 未知 | 未知 / 未知 | 未知 / 未知 |

**评价与反思。** 只对完整通过参考cache/state/KL/PPL的计时配对作成本解释；未完成/失败不作加速或成本均值。collector＋量化不包含模型准备、离线评估或事后hash，不能称完整端到端时间。若重算更慢，照实写GPU显存与时间权衡，不把省显存等同加速。扩展cal8/eval8/seed45配对的数值和资源门槛独立于这些cal2/eval2成本结果，失败配置不被合并或剔除。

**证据及后续。** /mnt/d/gptq_plus_realq/outputs/two-model-capacity-cost-records-20261011-0240/cost_summary.json及各计时目录原始paired_repetitions.json；正式配置/实际多卡仍待后续验证。

### AS：down_proj 求逆临时矩阵早释放与已消费 H 存储复用（2026-10-11，组件通过、真实前缀进行中）

**瓶颈、理论支持与文献。** AP中两模型完整全程最高allocated均迁移至末层down_proj求逆。原代码依次调用cholesky(H_work)、cholesky_inverse(chol)、cholesky(Hinv_init, upper=True)，局部H_work与chol直到helper返回仍存活。各输出分组的H原存储在其私有副本完成求逆后不再被消费，却与保存的全部Hinv因子同时驻留。借鉴OLLA的张量生命周期与存储位置规划（https://arxiv.org/html/2210.12924v2 ，Sec.2.2/2.3），在确定的最后消费者之后释放/复用存储；本研究不是OLLA的ILP调度器。原文已下载到docs/literature/OLLA_2210.12924v2.pdf，SHA及版本在相邻provenance.json，BibTeX加入references_memory_systems.bib。Checkmate的释放/驻留分析继续作为背景，不称其重算算法为本组件。

**方法具体内容。** 第一项仅del最后消费者完成的H_work和chol，保留原阻尼、重试、finite检查、三个solver调用及返回结果。第二项把当前组已经消费的self.H[sub_idx]存储用as_strided视图复用于该组Hinv，copy原solver结果并严格保留其dtype/shape/stride；后续尚未消费H组的内容保持不变。正常上三角Cholesky因子是列主序stride(1,n)，不得改为contiguous后假称逐位等价；identity fallback的原stride同样保留。无CPU搬运、不换分解/近似/精度、不改变Adam、Fisher或GPTQ顺序。研究入口仅在down_proj且alpha0、顺序分组、无Hessian正则/无H或Hinv CPU卸载时复用；真实rank/batched路径尚未适配，不能声称本原型已获多卡验证。生产默认源码未更改；原型为tools/hessian_inverse_lifetime_prototype.py。

**实际本地组件效果。** CUDA checker tools/check_hessian_inverse_lifetime_cuda.py 先验证12个FP32/FP64、保留/不保留逆、原地/非原地阻尼、重试/非finite fallback例，输出、输入最终状态及layout均精确一致。随后从本地0.6B/1.7B实际首层权重、WikiText seed44、2×256真实token前向捕获SwiGLU后down_proj输入；组件采用uniform saliency、4组、W4/per-row、block128、act_order，只隔离GPTQ API，不含Adam/整模型质量证明。

| 模型 | 提前释放单项：组件allocated peak C→N MiB | 提前释放＋已消费H复用：组件allocated peak C→N MiB | 组合reserved peak C→N MiB |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 668.2368→596.2368 | 668.2368→488.2368 | 706→526 |
| Qwen3-1.7B | 2576.4702→2288.4702 | 2576.4702→1856.4702 | 2600→1880 |

两模型各4次真实维度逆调用均被观察，组合确实复用4个消费后的H分组；量化权重和maxq/scale/zero哈希完全一致。仅提前释放分别减少72/288MiB，组合分别减少180/720MiB（组件allocated约26.94%/27.95%）。这些是隔离down_proj组件的峰值，不是整卡/全流程/时间平均下降；不能把720MiB直接从已有完整峰值减去宣称新成果。首次模型调用与对照库初始化之外的诊断时间不是速度证据。

**评价与反思。** 与先前HCPU/驻留重整的失败方案不同，本项管理同GPU的大矩阵存储，避免新增D2H/H2D布局与驻留；但allocator reserved是否回升仍须真实轨迹实测。只在存储当前组的最后消费者完成后覆盖，禁止复用仍用于正则或其它消费者的H。保持原factor stride是前期Hinv workspace数值失败得到的直接教训。临时张量节省不等于全程硬件门槛降低；候选最高点可能再次迁移。

**真实REAL-Q对照及下一步。** 控制组为AP已接受的checkpoint＋CE系统，候选只叠加上述两项；本组件收益不重复计入checkpoint。两侧fresh静态统计、cal2×1024、refresh microbatch2、eval2×1024、seed44、alpha0且Adam block-GD保持开启。tools/run_hessian_inverse_lifetime_followup.py 已启动outputs/hessian-inverse-reuse-followup-20261011-0945/：先两模型layer0量化/28层评估的独立prefix屏障，通过数值及全程/区间/边界资源门槛才做两模型各3次完整28层配对。prefix不能称全28层量化。任一失败停止该配置扩大，保留失败、不剔除极端值；每worker记录helper SHA、实际112次（全模型）inverse/存储复用的覆盖和stride。完成记录自动追加，成本/正式样本/真实多卡仍需后续。单一总日志logs/realq-system-restart-20261010-1408.log已append接入新队列。

**证据。** 单项outputs/hessian-inverse-lifetime-cuda-20261011-0945/status.json；组合outputs/hessian-inverse-reuse-cuda-20261011-0940/status.json；每种保存patch源码，helper接口保留矩阵/重试和layout比较；新增H复用结果不可归因于旧失败CPU组件。

### AQ排障续记：0.6B限额检查误判，非模型OOM结果

1.7B6400MiB对比已在03:06完整完成，候选fresh13 cache/state/28层/KL/PPL与未受限参考精确一致，baseline Stage-0 CUDA OOM；此模型容量对比成立。随后0.6B启动前报allocator budget was not installed，没有保存budget证据、没有执行真实量化，故原两模型队列valid=false，扩大样本/成本队列未启动，不能声称它们正在跑或算作方法负向。根因是API将fraction转为整数字节限额，原脚本严格浮点==误拒绝1字节以内舍入差；已改为保存observed_budget_bytes后比较误差≤1B。修复源代码不篡改旧失败工件，0.6B修复后的实际CUDA及完整重跑仍待串行验证。最新用户优先要求down_proj求逆，当前GPU用于AS，不并跑预算排障。

<!-- completed-memory-followup:qwen17b-hinv-reuse-layer0-20261011-0945 -->
### 自动续记：qwen17b-hinv-reuse-layer0-20261011-0945（2026-10-11 10:05:38，已完成审计）

**理论支持、文献与具体方法。** 本轮是AS节down_proj Hessian求逆存储生命周期的独立消融。两侧共享已接受的checkpoint＋CE系统；候选提前释放最后一次使用后的H_work和chol；hinv-reuse标签还将每个已消费H分组的GPU存储复用于该组逆Cholesky因子，保留参考因子的shape/dtype/stride和全部solver调用。仅验证alpha0、顺序分组、无Hessian正则和无H/Hinv CPU卸载路径；不改变Adam或量化算法。参考[OLLA](https://arxiv.org/html/2210.12924v2#S2.SS2)的张量生命周期/内存位置思想，不实现其ILP调度器，也不将已有checkpoint收益再次计入本组件。

**实际配置与效果。** 模型 `/mnt/d/llamaModels/Qwen3-1.7B`，校准 `2×1024`，refresh microbatch `2`，量化截止层 `0`。已审计 `1/1` 对，完整模型量化验证 `False`，campaign valid `True`。

| 指标 | control均值 | candidate均值 |
| --- | ---: | ---: |
| 整卡采样peak MiB | 4975.0000 | 4399.0000 |
| 完整进程allocated peak MiB | 4116.6484 | 3540.3984 |
| 完整进程reserved peak MiB | 4218.0000 | 3642.0000 |
| CPU RSS peak MiB | 11986.1133 | 11987.3516 |

首对KL/PPL C `0.017350513488054276 / 15.920016288757324`，N `0.017350513488054276 / 15.920016288757324`。每对数值门槛 `[True]`，资源门槛 `[True]`；区间回升数量 `[0]`、边界回升数量 `[0]`。

**评价与反思。** 所有请求配对通过已测门槛；仍只覆盖保存的本地pilot规模，不是正式论文规模/4卡证明。 prefix不能称为完整量化；NVML为采样，不能证明连续每一时刻。这里的均值是各次peak的均值，不是时间平均显存。诊断计时不能支持加速。原因分析需结合原始profile和代码，不以这个自动摘要替代人工审查。

**证据及后续。** `outputs/memory_campaign_qwen17b-hinv-reuse-layer0-20261011-0945/`、`outputs/hessian-inverse-reuse-followup-20261011-0945/records-qwen17b-hinv-reuse-layer0-20261011-0945/qwen17b-hinv-reuse-layer0-20261011-0945/summary.json`及原始每对comparison。完整量化与系统资源通过后才进入相同源码的低开销计时；尚未通过则先定位，不继续重复同一失败配置。

<!-- completed-memory-followup:qwen06b-hinv-reuse-layer0-20261011-0945 -->
### 自动续记：qwen06b-hinv-reuse-layer0-20261011-0945（2026-10-11 10:16:55，已完成审计）

**理论支持、文献与具体方法。** 本轮是AS节down_proj Hessian求逆存储生命周期的独立消融。两侧共享已接受的checkpoint＋CE系统；候选提前释放最后一次使用后的H_work和chol；hinv-reuse标签还将每个已消费H分组的GPU存储复用于该组逆Cholesky因子，保留参考因子的shape/dtype/stride和全部solver调用。仅验证alpha0、顺序分组、无Hessian正则和无H/Hinv CPU卸载路径；不改变Adam或量化算法。参考[OLLA](https://arxiv.org/html/2210.12924v2#S2.SS2)的张量生命周期/内存位置思想，不实现其ILP调度器，也不将已有checkpoint收益再次计入本组件。

**实际配置与效果。** 模型 `/mnt/d/llamaModels/Qwen3-0.6B`，校准 `2×1024`，refresh microbatch `2`，量化截止层 `0`。已审计 `1/1` 对，完整模型量化验证 `False`，campaign valid `True`。

| 指标 | control均值 | candidate均值 |
| --- | ---: | ---: |
| 整卡采样peak MiB | 2753.0000 | 2753.0000 |
| 完整进程allocated peak MiB | 2018.0317 | 2018.0317 |
| 完整进程reserved peak MiB | 2130.0000 | 2130.0000 |
| CPU RSS peak MiB | 5995.2656 | 6008.3828 |

首对KL/PPL C `0.025945916771888733 / 19.721397399902344`，N `0.025945916771888733 / 19.721397399902344`。每对数值门槛 `[True]`，资源门槛 `[True]`；区间回升数量 `[0]`、边界回升数量 `[0]`。

**评价与反思。** 所有请求配对通过已测门槛；仍只覆盖保存的本地pilot规模，不是正式论文规模/4卡证明。 prefix不能称为完整量化；NVML为采样，不能证明连续每一时刻。这里的均值是各次peak的均值，不是时间平均显存。诊断计时不能支持加速。原因分析需结合原始profile和代码，不以这个自动摘要替代人工审查。

**证据及后续。** `outputs/memory_campaign_qwen06b-hinv-reuse-layer0-20261011-0945/`、`outputs/hessian-inverse-reuse-followup-20261011-0945/records-qwen06b-hinv-reuse-layer0-20261011-0945/qwen06b-hinv-reuse-layer0-20261011-0945/summary.json`及原始每对comparison。完整量化与系统资源通过后才进入相同源码的低开销计时；尚未通过则先定位，不继续重复同一失败配置。

<!-- completed-memory-followup:qwen17b-hinv-reuse-full-20261011-0945 -->
### 自动续记：qwen17b-hinv-reuse-full-20261011-0945（2026-10-11 12:54:01，已完成审计）

**理论支持、文献与具体方法。** 本轮是AS节down_proj Hessian求逆存储生命周期的独立消融。两侧共享已接受的checkpoint＋CE系统；候选提前释放最后一次使用后的H_work和chol；hinv-reuse标签还将每个已消费H分组的GPU存储复用于该组逆Cholesky因子，保留参考因子的shape/dtype/stride和全部solver调用。仅验证alpha0、顺序分组、无Hessian正则和无H/Hinv CPU卸载路径；不改变Adam或量化算法。参考[OLLA](https://arxiv.org/html/2210.12924v2#S2.SS2)的张量生命周期/内存位置思想，不实现其ILP调度器，也不将已有checkpoint收益再次计入本组件。

**实际配置与效果。** 模型 `/mnt/d/llamaModels/Qwen3-1.7B`，校准 `2×1024`，refresh microbatch `2`，量化截止层 `27`。已审计 `1/3` 对，完整模型量化验证 `True`，campaign valid `False`。

| 指标 | control均值 | candidate均值 |
| --- | ---: | ---: |
| 整卡采样peak MiB | 6629.0000 | 6089.0000 |
| 完整进程allocated peak MiB | 5605.1909 | 5027.6909 |
| 完整进程reserved peak MiB | 5872.0000 | 5332.0000 |
| CPU RSS peak MiB | 11995.5000 | 11998.0156 |

首对KL/PPL C `0.625495195388794 / 25.562191009521484`，N `0.625495195388794 / 25.562191009521484`。每对数值门槛 `[True]`，资源门槛 `[False]`；区间回升数量 `[14]`、边界回升数量 `[15]`。

**评价与反思。** 本轮未通过完整请求/严格门槛；即使局部peak下降也不能作为合格最终方案，失败配对保留，不剔除。 prefix不能称为完整量化；NVML为采样，不能证明连续每一时刻。这里的均值是各次peak的均值，不是时间平均显存。诊断计时不能支持加速。原因分析需结合原始profile和代码，不以这个自动摘要替代人工审查。

**证据及后续。** `outputs/memory_campaign_qwen17b-hinv-reuse-full-20261011-0945/`、`outputs/hessian-inverse-reuse-followup-20261011-0945/records-qwen17b-hinv-reuse-full-20261011-0945/qwen17b-hinv-reuse-full-20261011-0945/summary.json`及原始每对comparison。完整量化与系统资源通过后才进入相同源码的低开销计时；尚未通过则先定位，不继续重复同一失败配置。


<!-- inverse-lifetime-cross-repeat:20261011-0945 -->
### AS续记：完整队列退出后的独立复核

**理论与方法。** 延续AS的矩阵最后消费者与原布局存储复用。除每对审计之外，重新实际哈希已完成缓存，核对六臂跨重复身份与已接受checkpoint＋CE参考，逐worker核查112次down_proj inverse/复用调用覆盖。仅共享SHA一致的已完成prepared FP硬链接，校准/Fisher统计独立。

**实际效果及评价。** 完整独立审计valid=False；详见下列原始状态。缺失或未完成全层不列为成功；失败不剔除、不重复推广。诊断时间不计加速，正式校准规模和多卡仍待验证。

```json
{
  "valid": false,
  "parent_valid": false,
  "models": [
    {
      "model": "qwen17b",
      "complete_valid_full_campaign": false,
      "note": "Not a full positive; missing work is not a measured negative"
    },
    {
      "model": "qwen06b",
      "complete_valid_full_campaign": false,
      "note": "Not a full positive; missing work is not a measured negative"
    }
  ],
  "finished": "2026-10-11 12:54:05",
  "scope": "Independent full-model cross-repeat/reference and method-coverage audit; no speed/physical smaller-GPU claim"
}
```

**反思及证据。** 组件原值与完整资源值分别见AS和对应自动续记；跨重复cache/state/28层/KL/PPL、原参考身份、实际方法覆盖需同时成立。工件outputs/hessian-inverse-reuse-completed-audit-20261011-0945。


<!-- inverse-lifetime-manual-fragmentation:20261011 -->
### AS人工复核与下一步：实际张量省显存，但回放入口碎片回升

**理论与文献。** OLLA第2.3节区分张量驻留量和存储位置引起的碎片：https://arxiv.org/html/2210.12924v2 。PyTorch2.11 CUDA文档解释局部MemPool及池无活跃张量后析构回收；本轮尝试局部临时池，而非OLLA的ILP或自定义CUDA算子：https://docs.pytorch.org/docs/2.11/notes/cuda.html 。

**实际效果。** 1.7B第一对完成全28层，独立配对缓存、最终state、全部28层输出指纹、KL/PPL相等；全设备采样峰值6629→6089MiB（8.146%），进程allocated5605.1909→5027.6909MiB、reserved5872→5332MiB。112次inverse和112次候选存储复用实际执行，源码未变。控制为已接受checkpoint+CE系统，不是原始上游base。尚未完成三对重复；0.6B全模型未启动。

**严格评价。** 全部区间/边界allocated回升数量为0，14个区间、15个边界reserved回升，最大边界+540MiB。因此AS组合仍不合格，保留失败，不继续同配置第二/三对。第15层回放入口allocated均1392.120117MiB，inactive split由257.879883→797.879883MiB，差值正好540MiB；下一层入口两臂inactive split又同为157.932617MiB。支持分配布局导致被活跃块钉住的空闲碎片解释，尚没有逐块归属快照，不能说已证明是哪一个张量。原项目free已empty_cache，追加清缓存无法释放仍与活跃块共处的拆分段。

**下一方法。** 先以真实Qwen第一块的down_proj输入验证求逆私有池：临时求逆结果在池内计算，按原布局复制回已消费H组，确保无活跃私有池tensor逃逸后释放池；避免临时大矩阵与后续长寿命张量共享存储段。此脚本仅CUDA组件、均匀saliency、无Adam，不是REALQ全模型证据。明确记录全进程峰值，不能通过隐藏私有池占用获得假收益。显式同步属于诊断实现，其耗时需要后续单独审查。

**证据。** outputs/memory_campaign_qwen17b-hinv-reuse-full-20261011-0945/manual_fragmentation_diagnosis.json；新代码tools/check_hessian_inverse_pool_cuda.py，新预检outputs/hessian-inverse-pool-cuda-20261011-next/status.json。新结果待完成再按同格式追加，组件通过仍需真实Adam、多层与全层资源验证。


<!-- inverse-pool-preflight:v2-20261011-1335 -->
### AS续记：私有求逆池CUDA预检与1.7B真实轨迹计划

**理论支持与文献。** 沿用OLLA的生命周期与存储位置分离思想（https://arxiv.org/html/2210.12924v2），采用PyTorch2.11局部native MemPool而非其ILP： https://docs.pytorch.org/docs/2.11/notes/cuda.html 。有张量仍占据一个段时，空闲拆分块不能靠empty_cache释放；隔离短命求逆量与长命量是针对AS失败证据的工程改造。

**方法内容。** alpha0、顺序分组、无H/Hinv CPU卸载和Hessian正则的已验证路径。保留原Cholesky→inverse→upper Cholesky与dtype/stride/阻尼/重试/回退；临时量在私有池分配，结果复制到当前已消费H组的原布局视图后释放私有池；单worker仅量化一次，禁止私有池tensor逃逸。每调用同步和轻量pool快照仅用于诊断，不能声称加速。异常traceback循环若保持临时量，则仅在池仍有active块时gc并复核。

**实际效果。** v2六组CUDA SPD、FP32/FP64、inplace/非inplace、重试/单位阵fallback均与原算式逐字节相同；两模型实际第一层FFN输入的组件量化weight和quantizer完全一致，每个worker四组私有池均实际释放。组件allocated peak 0.6B668.2368→488.2368MiB、1.7B2576.4702→1856.4702MiB；reserved 706→584MiB、2600→2124MiB。此处是真实权重/输入+均匀saliency、无Adam的组件，不是完整REALQ/全局收益。v2绑定源码SHA 198fb21976e20516d6bf4ff6f1cfd1d1bb05c371759a74f0c5c365860d89978f。

**评价与反思。** isolated池reserved高于前一无池reuse组件（0.6B526MiB、1.7B1880MiB），说明隔离也有池并存代价，不能假定全层门槛必过；必须看实际已同步区间和边界。此前全层8.15%峰值改善仍因reserved回升失败，保留。

**后续真实实验。** outputs/hessian-inverse-pool-followup-20261011-1340：1.7B先量化layer0–2（覆盖此前最早layer2回放回升），两臂fresh Stage0 cal2×1024、micro2、eval2×1024、seed44，真实Adam blockGD启用；双方共享已接受checkpoint+CE，仅候选增加隔离池+生命周期/存储复用。prefix资源/数值通过后自动全28层三对，失败不继续同配置重跑；完成后独立哈希缓存、跨重复/旧参考身份及112次实际方法覆盖。总日志logs/realq-system-restart-20261010-1408.log。


<!-- fragmentation-literature-next-path:20261011-1340 -->
### AS补充文献与连续实验决策规则

**参考文献及应用判断。** 重读GMLake（Guo等，ASPLOS2024，arXiv2401.08156v1）第1/2/4节：重算、卸载会引入不规则分配，split缓存可能有大量空闲碎片；其实际方法是CUDA VMM stitch，我们当前没有实现该框架，不能将私有池等同于GMLake。 https://arxiv.org/html/2401.08156v1 。OLLA分别优化存活区间和内存位置，支持把AS组合拆开诊断的工程思路。

**连续工作规则。** 当前pool真实轨迹的同配置若失败，不重跑。只在已完成配对证明数值完全一致、严格资源门槛失败时，触发仅early-release的独立消融；其真实两模型CUDA预检已通过且controller核对源码，参数/真实Adam不变。若为基础设施或数值错误，必须先定位，不能被自动消融掩盖。若私有池通过前三层，先继续其28层三对，不同时占用GPU跑别的候选。完成后独立哈希参考/跨重复；不将单对、prefix或组件称为终稿成果。

**代码及评价。** tools/run_inverse_resource_failure_followup.py等待主要队列终止，实际资源失败证据记录到decision.json，随后串行下一方案。fallback到来才写触发文档和启动GPU；否则不执行。总日志在原reader退出后继续追加，原工件不覆盖。全套资源诊断不提供速度结论。


<!-- inverse-prefix-gradient-scope:20261011 -->
**prefix范围补充核对。** 保存日志中layer0 cosine grad_lr schedule的scale=0，因此layer0虽走实际REALQ blockGD/Adam路径，却不能单独证明非零Adam步下的改造等价。此前AS全28层第一对已经覆盖非零更新且质量相等；私有池当前screen选择layer0–2，实际layer1/2的effective_lr非零，同时覆盖此前最早layer2 reserved回升。这也是不继续仅layer0测试的原因。组件均匀saliency与完整Fisher/非零Adam证据继续分开标记。

<!-- completed-memory-followup:qwen17b-hinv-isolated-pool-layer2-20261011-1340 -->
### 自动续记：qwen17b-hinv-isolated-pool-layer2-20261011-1340（2026-10-11 14:02:23，已完成审计）

**理论支持、文献与具体方法。** 本轮为AS续记的down_proj求逆私有池修正；两侧共享已接受checkpoint＋CE，候选保持原求逆运算、精度和因子stride，提前释放临时矩阵并将结果复制到已消费H组；求逆临时量在独立CUDA native池分配，确认无活跃池张量后同步并释放池，避免临时大矩阵与后续长期存活张量混用存储段。参考[OLLA](https://arxiv.org/html/2210.12924v2)的存储位置/碎片分析及[PyTorch2.11局部MemPool文档](https://docs.pytorch.org/docs/2.11/notes/cuda.html)。不实现OLLA的ILP；不改变Adam/量化算法；同步代价属于待测项，诊断计时不作为加速证据。

**实际配置与效果。** 模型 `/mnt/d/llamaModels/Qwen3-1.7B`，校准 `2×1024`，refresh microbatch `2`，量化截止层 `2`。已审计 `1/1` 对，完整模型量化验证 `False`，campaign valid `False`。

| 指标 | control均值 | candidate均值 |
| --- | ---: | ---: |
| 整卡采样peak MiB | 4999.0000 | 4809.0000 |
| 完整进程allocated peak MiB | 4116.6484 | 3540.3984 |
| 完整进程reserved peak MiB | 4242.0000 | 4052.0000 |
| CPU RSS peak MiB | 12001.6797 | 11998.0977 |

首对KL/PPL C `0.1913219541311264 / 20.164043426513672`，N `0.1913219541311264 / 20.164043426513672`。每对数值门槛 `[True]`，资源门槛 `[False]`；区间回升数量 `[10]`、边界回升数量 `[1]`。

**评价与反思。** 本轮未通过完整请求/严格门槛；即使局部peak下降也不能作为合格最终方案，失败配对保留，不剔除。 prefix不能称为完整量化；NVML为采样，不能证明连续每一时刻。这里的均值是各次peak的均值，不是时间平均显存。诊断计时不能支持加速。原因分析需结合原始profile和代码，不以这个自动摘要替代人工审查。

**证据及后续。** `outputs/memory_campaign_qwen17b-hinv-isolated-pool-layer2-20261011-1340/`、`outputs/hessian-inverse-pool-followup-20261011-1340/records-qwen17b-hinv-isolated-pool-layer2-20261011-1340/qwen17b-hinv-isolated-pool-layer2-20261011-1340/summary.json`及原始每对comparison。完整量化与系统资源通过后才进入相同源码的低开销计时；尚未通过则先定位，不继续重复同一失败配置。


<!-- inverse-lifetime-cross-repeat:20261011-1340 -->
### AS续记：完整队列退出后的独立复核

**理论与方法。** 延续AS的矩阵最后消费者与原布局存储复用。除每对审计之外，重新实际哈希已完成缓存，核对六臂跨重复身份与已接受checkpoint＋CE参考，逐worker核查112次down_proj inverse/复用调用覆盖。仅共享SHA一致的已完成prepared FP硬链接，校准/Fisher统计独立。

**实际效果及评价。** 完整独立审计valid=False；详见下列原始状态。缺失或未完成全层不列为成功；失败不剔除、不重复推广。诊断时间不计加速，正式校准规模和多卡仍待验证。

```json
{
  "valid": false,
  "parent_valid": false,
  "models": [
    {
      "model": "qwen17b",
      "complete_valid_full_campaign": false,
      "note": "Not a full positive; missing work is not a measured negative"
    }
  ],
  "finished": "2026-10-11 14:02:24",
  "scope": "Independent full-model cross-repeat/reference and method-coverage audit; no speed/physical smaller-GPU claim"
}
```

**反思及证据。** 组件原值与完整资源值分别见AS和对应自动续记；跨重复cache/state/28层/KL/PPL、原参考身份、实际方法覆盖需同时成立。工件outputs/hessian-inverse-pool-full-audit-20261011-1340。


### AS进一步消融：私有池资源失败后的预设下一实验

**理论与参考。** OLLA将lifetime和location分开建模（https://arxiv.org/html/2210.12924v2）；GMLake指出重算/卸载引入不规则分配会加重碎片（https://arxiv.org/html/2401.08156v1）。不是复现原论文正确性，也不是扫参数。

**触发证据及方法。** 本次私有池版本数值正确但资源门槛失败，路径outputs/inverse-adaptive-resource-followup-20261011-1340/decision.json。下一候选仅提前释放H_work/chol，移除H缓冲复用及独立池；保持量化/Adam/校准参数。用已有真实CUDA两模型预检作为源码绑定前提，再fresh 1.7B layer0–2检查真实Adam轨迹；通过才运行完整28层三对。

**评价及反思。** 正向峰值不能覆盖局部资源回升，失败保留；本段只是实际触发后的下一方案，尚无下一实验效果。结果由完成记录器按方法/效果/评价/反思格式追加。

<!-- completed-memory-followup:qwen17b-hinv-early-release-layer2-20261011-1340-early -->
### 自动续记：qwen17b-hinv-early-release-layer2-20261011-1340-early（2026-10-11 14:31:48，已完成审计）

**理论支持、文献与具体方法。** 本轮是AS节down_proj Hessian求逆存储生命周期的独立消融。两侧共享已接受的checkpoint＋CE系统；候选提前释放最后一次使用后的H_work和chol；hinv-reuse标签还将每个已消费H分组的GPU存储复用于该组逆Cholesky因子，保留参考因子的shape/dtype/stride和全部solver调用。仅验证alpha0、顺序分组、无Hessian正则和无H/Hinv CPU卸载路径；不改变Adam或量化算法。参考[OLLA](https://arxiv.org/html/2210.12924v2#S2.SS2)的张量生命周期/内存位置思想，不实现其ILP调度器，也不将已有checkpoint收益再次计入本组件。

**实际配置与效果。** 模型 `/mnt/d/llamaModels/Qwen3-1.7B`，校准 `2×1024`，refresh microbatch `2`，量化截止层 `2`。已审计 `1/1` 对，完整模型量化验证 `False`，campaign valid `False`。

| 指标 | control均值 | candidate均值 |
| --- | ---: | ---: |
| 整卡采样peak MiB | 4999.0000 | 4999.0000 |
| 完整进程allocated peak MiB | 4116.6484 | 4116.6484 |
| 完整进程reserved peak MiB | 4242.0000 | 4242.0000 |
| CPU RSS peak MiB | 11988.4648 | 12006.9141 |

首对KL/PPL C `0.1913219541311264 / 20.164043426513672`，N `0.1913219541311264 / 20.164043426513672`。每对数值门槛 `[True]`，资源门槛 `[False]`；区间回升数量 `[2]`、边界回升数量 `[0]`。

**评价与反思。** 本轮未通过完整请求/严格门槛；即使局部peak下降也不能作为合格最终方案，失败配对保留，不剔除。 prefix不能称为完整量化；NVML为采样，不能证明连续每一时刻。这里的均值是各次peak的均值，不是时间平均显存。诊断计时不能支持加速。原因分析需结合原始profile和代码，不以这个自动摘要替代人工审查。

**证据及后续。** `outputs/memory_campaign_qwen17b-hinv-early-release-layer2-20261011-1340-early/`、`outputs/inverse-early-release-ablation-20261011-1340/records-qwen17b-hinv-early-release-layer2-20261011-1340-early/qwen17b-hinv-early-release-layer2-20261011-1340-early/summary.json`及原始每对comparison。完整量化与系统资源通过后才进入相同源码的低开销计时；尚未通过则先定位，不继续重复同一失败配置。


### AS后续：失败后继续存储位置优化（2026-10-11）

**理论支持与参考。** [OLLA](https://arxiv.org/html/2210.12924v2)分别讨论生命周期和存储位置；[PyTorch copy_](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.copy_.html)支持复制到既有且可不同dtype的目标。借鉴生命周期/位置原则，不实现OLLA求解器。

**上一轮实际效果。** 私有池layer0–2数值完全一致、整卡peak4999→4809MiB，但10区间/1边界reserved回升，求逆区间最大+542MiB，不能接受。仅提前释放layer0–2数值一致，peak4999→4999MiB，layer2回放reserved+48MiB、末分组求逆区间allocated+0.25MiB，仍不合格。两轮均未进入28层三对，不把prefix当全量结果。原始失败证据：outputs/hessian-stable-weight-followup-design-20261011.json。

**新方法具体内容。** 不重复私有池：保留已消费H存储复用，改变down_proj末尾写回，从重新分配新权重改成向已有BF16权重copy_；既有权重ptr/dtype/shape/stride保持。两侧均观察权重所在allocator段大小和inactive空间，检查长期权重是否占住此前临时矩阵所在段。这是待验证的代码归因，不能仅凭reserved差就断言由权重导致。

**审查与实验。** 新预检使用Qwen3-0.6B/1.7B真实权重和WikiText首层FFN输入、12组helper/回退核对、8组不同dtype/布局copy与原to转换核对，保存源码与哈希。首次预检因动态源码匹配缩进错误，在更改计算前中止，保留失败；修为显式匹配实际缩进后新建v2，不当作方法负结果。随后新鲜REAL-Q cal2×1024/refresh微批2/eval2×1024/seed44，先layer0–2实际Adam轨迹，再满足门槛后28层三对，并复核旧已接受参考。

**评价与反思。** 当前是实现与验证进行中，尚无新完整模型成功结果。数值失败先修bug；资源失败停止该候选并分析分配记录，再推进有机制依据的下一方案，不把单个候选失败当成结束整个研究。所有失败保留，不改变门槛/校准数量，不将旧checkpoint收益重复计入求逆组件。


### AS新候选组件结果与真实轨迹启动（2026-10-11）

**实际效果。** hessian-stable-weight-cuda-20261011-v2成功：12组helper/retained inverse/inplace/回退原值核对、8组FP16/BF16/FP32/FP64两布局转换核对，两模型实际FFN输入的量化权重与quantizer全部exact。0.6B组件allocated668.2368→488.2368MiB、reserved706→526MiB；1.7B组件allocated2576.4702→1856.4702MiB、reserved2600→1880MiB。此收益包含既有消费H复用，不归为单独写回贡献。

**机制证据。** 两模型candidate权重地址保持不变；1.7B组件control新权重位于36MiB段、其inactive为12MiB，candidate继续用原24MiB段、inactive为0。这是组件里的存储现象，不能证明已解决真实轨迹540MiB碎片。真实prefix/full会保存每层段归属，继续审查。源码SHA：6a84aae866064d25ccf28436e544be7757f4df7ae808692de054e0ad11a5b514.

**实际执行与评价。** 串行GPU队列outputs/hessian-stable-weight-followup-20261011-v2已启动，模型顺序1.7B、0.6B；先各layer0–2新鲜配对，再过严格资源/数值门槛后各28层三对。独立跨重复/reference复核为outputs/hessian-stable-weight-full-audit-20261011-v2。总日志logs/realq-system-restart-20261010-1408.log，只有一个append读取器。尚无该候选全层成功结论，原正式规模和多卡证明仍待做。

**失败后的连续改进。** outputs/hessian-workspace-adaptive-followup-20261011-v2已启动CPU等待器，不并行CUDA。仅当前队列结束且有完整数值正确的资源失败时，保存决定及原负结果、继续新的solver工作区out=调度：先两模型CUDA预检；通过后再新fresh真实轨迹和完整三对。它不会重试同一配置，不会降低门槛。新的工作区方法目前仅源码/入口编译核对，没有CUDA有效性结论。若当前方案完成所有请求门槛，则不额外启动替代方法；独立审计另行执行。

<!-- completed-memory-followup:qwen17b-hinv-stable-weight-layer2-20261011-v2 -->
### 自动续记：qwen17b-hinv-stable-weight-layer2-20261011-v2（2026-10-11 15:12:05，已完成审计）

**理论支持、文献与具体方法。** 本轮为AS后续的存储位置修正：两侧共享已接受checkpoint＋CE；候选保留已消费H组存储复用，不启用私有池，将down_proj最终量化权重copy_至既有权重存储，避免长期权重重新分配。保留原solver/因子布局/量化顺序/Adam/校准数据，附两侧相同的CPU allocator段归属观察。参考[OLLA](https://arxiv.org/html/2210.12924v2)的生命周期和存储位置原则；未实现其ILP。必须核对实际转换、缓存/每层输出/最终state/KL/PPL和全阶段资源；诊断计时不计加速。

**实际配置与效果。** 模型 `/mnt/d/llamaModels/Qwen3-1.7B`，校准 `2×1024`，refresh microbatch `2`，量化截止层 `2`。已审计 `1/1` 对，完整模型量化验证 `False`，campaign valid `True`。

| 指标 | control均值 | candidate均值 |
| --- | ---: | ---: |
| 整卡采样peak MiB | 4999.0000 | 4407.0000 |
| 完整进程allocated peak MiB | 4116.6484 | 3540.3984 |
| 完整进程reserved peak MiB | 4242.0000 | 3650.0000 |
| CPU RSS peak MiB | 11982.6094 | 11986.2695 |

首对KL/PPL C `0.1913219541311264 / 20.164043426513672`，N `0.1913219541311264 / 20.164043426513672`。每对数值门槛 `[True]`，资源门槛 `[True]`；区间回升数量 `[0]`、边界回升数量 `[0]`。

**评价与反思。** 所有请求配对通过已测门槛；仍只覆盖保存的本地pilot规模，不是正式论文规模/4卡证明。 prefix不能称为完整量化；NVML为采样，不能证明连续每一时刻。这里的均值是各次peak的均值，不是时间平均显存。诊断计时不能支持加速。原因分析需结合原始profile和代码，不以这个自动摘要替代人工审查。

**证据及后续。** `outputs/memory_campaign_qwen17b-hinv-stable-weight-layer2-20261011-v2/`、`outputs/hessian-stable-weight-followup-20261011-v2/records-qwen17b-hinv-stable-weight-layer2-20261011-v2/qwen17b-hinv-stable-weight-layer2-20261011-v2/summary.json`及原始每对comparison。完整量化与系统资源通过后才进入相同源码的低开销计时；尚未通过则先定位，不继续重复同一失败配置。

<!-- completed-memory-followup:qwen06b-hinv-stable-weight-layer2-20261011-v2 -->
### 自动续记：qwen06b-hinv-stable-weight-layer2-20261011-v2（2026-10-11 15:25:28，已完成审计）

**理论支持、文献与具体方法。** 本轮为AS后续的存储位置修正：两侧共享已接受checkpoint＋CE；候选保留已消费H组存储复用，不启用私有池，将down_proj最终量化权重copy_至既有权重存储，避免长期权重重新分配。保留原solver/因子布局/量化顺序/Adam/校准数据，附两侧相同的CPU allocator段归属观察。参考[OLLA](https://arxiv.org/html/2210.12924v2)的生命周期和存储位置原则；未实现其ILP。必须核对实际转换、缓存/每层输出/最终state/KL/PPL和全阶段资源；诊断计时不计加速。

**实际配置与效果。** 模型 `/mnt/d/llamaModels/Qwen3-0.6B`，校准 `2×1024`，refresh microbatch `2`，量化截止层 `2`。已审计 `1/1` 对，完整模型量化验证 `False`，campaign valid `True`。

| 指标 | control均值 | candidate均值 |
| --- | ---: | ---: |
| 整卡采样peak MiB | 2753.0000 | 2753.0000 |
| 完整进程allocated peak MiB | 2018.0317 | 2018.0317 |
| 完整进程reserved peak MiB | 2130.0000 | 2130.0000 |
| CPU RSS peak MiB | 6005.3281 | 6007.6406 |

首对KL/PPL C `0.08332214504480362 / 21.569381713867188`，N `0.08332214504480362 / 21.569381713867188`。每对数值门槛 `[True]`，资源门槛 `[True]`；区间回升数量 `[0]`、边界回升数量 `[0]`。

**评价与反思。** 所有请求配对通过已测门槛；仍只覆盖保存的本地pilot规模，不是正式论文规模/4卡证明。 prefix不能称为完整量化；NVML为采样，不能证明连续每一时刻。这里的均值是各次peak的均值，不是时间平均显存。诊断计时不能支持加速。原因分析需结合原始profile和代码，不以这个自动摘要替代人工审查。

**证据及后续。** `outputs/memory_campaign_qwen06b-hinv-stable-weight-layer2-20261011-v2/`、`outputs/hessian-stable-weight-followup-20261011-v2/records-qwen06b-hinv-stable-weight-layer2-20261011-v2/qwen06b-hinv-stable-weight-layer2-20261011-v2/summary.json`及原始每对comparison。完整量化与系统资源通过后才进入相同源码的低开销计时；尚未通过则先定位，不继续重复同一失败配置。

<!-- completed-memory-followup:qwen17b-hinv-stable-weight-full-20261011-v2 -->
### 自动续记：qwen17b-hinv-stable-weight-full-20261011-v2（2026-10-11 17:53:55，已完成审计）

**理论支持、文献与具体方法。** 本轮为AS后续的存储位置修正：两侧共享已接受checkpoint＋CE；候选保留已消费H组存储复用，不启用私有池，将down_proj最终量化权重copy_至既有权重存储，避免长期权重重新分配。保留原solver/因子布局/量化顺序/Adam/校准数据，附两侧相同的CPU allocator段归属观察。参考[OLLA](https://arxiv.org/html/2210.12924v2)的生命周期和存储位置原则；未实现其ILP。必须核对实际转换、缓存/每层输出/最终state/KL/PPL和全阶段资源；诊断计时不计加速。

**实际配置与效果。** 模型 `/mnt/d/llamaModels/Qwen3-1.7B`，校准 `2×1024`，refresh microbatch `2`，量化截止层 `27`。已审计 `1/3` 对，完整模型量化验证 `True`，campaign valid `False`。

| 指标 | control均值 | candidate均值 |
| --- | ---: | ---: |
| 整卡采样peak MiB | 6629.0000 | 6089.0000 |
| 完整进程allocated peak MiB | 5605.1909 | 5027.6909 |
| 完整进程reserved peak MiB | 5872.0000 | 5332.0000 |
| CPU RSS peak MiB | 12000.3164 | 11987.3633 |

首对KL/PPL C `0.625495195388794 / 25.562191009521484`，N `0.625495195388794 / 25.562191009521484`。每对数值门槛 `[True]`，资源门槛 `[False]`；区间回升数量 `[2]`、边界回升数量 `[0]`。

**评价与反思。** 本轮未通过完整请求/严格门槛；即使局部peak下降也不能作为合格最终方案，失败配对保留，不剔除。 prefix不能称为完整量化；NVML为采样，不能证明连续每一时刻。这里的均值是各次peak的均值，不是时间平均显存。诊断计时不能支持加速。原因分析需结合原始profile和代码，不以这个自动摘要替代人工审查。

**证据及后续。** `outputs/memory_campaign_qwen17b-hinv-stable-weight-full-20261011-v2/`、`outputs/hessian-stable-weight-followup-20261011-v2/records-qwen17b-hinv-stable-weight-full-20261011-v2/qwen17b-hinv-stable-weight-full-20261011-v2/summary.json`及原始每对comparison。完整量化与系统资源通过后才进入相同源码的低开销计时；尚未通过则先定位，不继续重复同一失败配置。


<!-- inverse-lifetime-cross-repeat:20261011-v2 -->
### AS续记：完整队列退出后的独立复核

**理论与方法。** 延续AS的矩阵最后消费者与原布局存储复用。除每对审计之外，重新实际哈希已完成缓存，核对六臂跨重复身份与已接受checkpoint＋CE参考，逐worker核查112次down_proj inverse/复用调用覆盖。仅共享SHA一致的已完成prepared FP硬链接，校准/Fisher统计独立。

**实际效果及评价。** 完整独立审计valid=False；详见下列原始状态。缺失或未完成全层不列为成功；失败不剔除、不重复推广。诊断时间不计加速，正式校准规模和多卡仍待验证。

```json
{
  "valid": false,
  "parent_valid": false,
  "models": [
    {
      "model": "qwen17b",
      "complete_valid_full_campaign": false,
      "note": "Not a full positive; missing work is not a measured negative"
    },
    {
      "model": "qwen06b",
      "complete_valid_full_campaign": false,
      "note": "Not a full positive; missing work is not a measured negative"
    }
  ],
  "finished": "2026-10-11 17:53:55",
  "scope": "Independent full-model cross-repeat/reference and method-coverage audit; no speed/physical smaller-GPU claim"
}
```

**反思及证据。** 组件原值与完整资源值分别见AS和对应自动续记；跨重复cache/state/28层/KL/PPL、原参考身份、实际方法覆盖需同时成立。工件outputs/hessian-stable-weight-full-audit-20261011-v2。


### AS条件续接：既有权重存储方案资源失败后继续工作区验证

**理论与参考。** OLLA的最后消费者/内存位置思想（https://arxiv.org/html/2210.12924v2）；PyTorch2.11 cholesky与cholesky_inverse的out=接口。

**触发证据及具体方法。** 完成的数值正确资源失败见outputs/hessian-workspace-adaptive-followup-20261011-v2/decision.json。保留原三步求逆和dtype/stride：H_work最后被Cholesky读取后用作逆矩阵目标；chol最后被inverse读取后用作最终upper因子目标，每步输入与输出存储不同。随后保留消费H组复用和既有权重写回，不启用私有池。原inplace damping分支回到原helper，实际目标轨迹不启用该分支。

**评价及反思。** 这是新分配调度的小实验，尚无正向结果。先实际CUDA两模型原值/回退/布局核对；精确失败保留并禁止上真实轨迹；成功后才fresh layer0–2，再过资源门槛才28层三对。不剔除失败，不改校准/Adam参数，诊断计时不算加速。


### AS实时复核：既有权重存储全28层首对及自动续接（2026-10-11 17:54）

**方法与参考。** 延续上一节OLLA生命周期/存储位置原则，保留消费H复用，仅将down_proj长期权重写回改为既有存储copy_；不启用私有池。

**实际效果。** 两模型layer0–2 fresh配对均exact且严格资源门槛通过。1.7B全28层第一对actual权重/state/28层输出、缓存和KL/PPL均exact；KL0.625495195388794，PPL25.562191009521484；整卡6629→6089MiB（8.1460%），完整进程allocated5605.1909→5027.6909MiB，reserved5872→5332MiB。比较的是已接受checkpoint＋CE系统，不是原上游。该候选全量三对没有完成：layer3、layer10回放区间reserved各+36MiB，虽无allocated/边界回升，仍严格拒绝。失败第一对保留，不剔除。

**机制核对。** 28次down_proj写回，C均更换存储，N均保持ptr。实际layer2/layer27 C新权重段144MiB、inactive120MiB；layer9 C新权重段576MiB、inactive520MiB。N对应layer2/layer9用24MiB原段、inactive0；layer27用128MiB原段、inactive24MiB。这支持长期权重重新分配会占住大段的代码假设，但仍有两区间reserved回升，不能断言所有碎片根因已解决。

**评价、反思及下一步。** 与先前非池复用版本的14区间/15边界回升相比，这版剩2区间/0边界（独立fresh轨迹，不能等同隔离写回的单变量统计效应）。严格标准不放宽。自动续接在17:54已启动新的solver工作区CUDA组件预检；维持三步求逆与F布局，以out=复用最后消费者后的两块不同输入/输出存储。0.6B与12组helper已核对exact，1.7B尚在验证，预检通过才进入新鲜真实轨迹。这不是该候选成功、正式2048样本证明或速度收益。

**证据。** outputs/memory_campaign_qwen17b-hinv-stable-weight-full-20261011-v2/pair1_audit/comparison.json；每臂inverse_lifetime_stage2_rank0.json；阶段复核outputs/live_phase_check_qwen17b-hinv-stable-weight-full-20261011-v2-r1.json。总日志转发已自动切换至workspace续接，只有一个writer。


### 开发实验流程反思：瓶颈筛查先于完整重跑（2026-10-11）

**问题与评价。** 用户指出每个候选fresh Stage-0、全量量化/评估及大型cache哈希过于耗时。对Stage-2-only存储调度改动，这样的开发验证成本确实过高；最终验收仍需完整流程。

**调整内容。** 后续按docs/bottleneck_validation_protocol.md分级：真实1.7B bottleneck数据一次捕获/局部重放 → 已验证静态缓存支持的短窗口联动检查 → 稳定候选单次全28层 → 最终三组重复。重点检查layer27 down_proj峰值和layer3/10回放回升。保留校准/算法/资源门槛；局部结果不宣称全流程成功。

**审查与反思。** 现有首层真实激活组件采用uniform saliency且无Adam，不能替代真实量化轨迹；quant_stop_layer不是任意起点恢复。张量检查点不能恢复CUDA allocator历史，尤其reserved碎片必须保留实际前缀/全流程核对。当前尚未实现完整快照恢复器和Stage-2-only窗口驱动，不能把此流程文档当作已运行的新实验。当前活动候选保持冻结manifest，不中途污染结果。

<!-- completed-memory-followup:qwen17b-hinv-workspace-stable-layer2-20261011-workspace-v2 -->
### 自动续记：qwen17b-hinv-workspace-stable-layer2-20261011-workspace-v2（2026-10-11 18:23:52，已完成审计）

**理论支持、文献与具体方法。** 本轮是数值正确资源失败后的solver工作区分配调度：保留原三步Cholesky/inverse/upper调用，用最后消费者后的H_work、chol作为out=目标，每步输入/输出不共享存储，保持原因子stride。仅非inplace damping使用工作区复用；原inplace分支回到原helper。之后仍复用消费H组存储、向既有down_proj权重copy_，不启用私有池。参考[OLLA](https://arxiv.org/html/2210.12924v2)的lifetime/location原则和PyTorch2.11 out=接口；不是算法替换，不改变REAL-Q的Adam、校准或量化参数；局部组件证明不能替代全程严格资源审计。

**实际配置与效果。** 模型 `/mnt/d/llamaModels/Qwen3-1.7B`，校准 `2×1024`，refresh microbatch `2`，量化截止层 `2`。已审计 `1/1` 对，完整模型量化验证 `False`，campaign valid `True`。

| 指标 | control均值 | candidate均值 |
| --- | ---: | ---: |
| 整卡采样peak MiB | 4999.0000 | 4407.0000 |
| 完整进程allocated peak MiB | 4116.6484 | 3540.3984 |
| 完整进程reserved peak MiB | 4242.0000 | 3650.0000 |
| CPU RSS peak MiB | 11990.4258 | 12006.5156 |

首对KL/PPL C `0.1913219541311264 / 20.164043426513672`，N `0.1913219541311264 / 20.164043426513672`。每对数值门槛 `[True]`，资源门槛 `[True]`；区间回升数量 `[0]`、边界回升数量 `[0]`。

**评价与反思。** 所有请求配对通过已测门槛；仍只覆盖保存的本地pilot规模，不是正式论文规模/4卡证明。 prefix不能称为完整量化；NVML为采样，不能证明连续每一时刻。这里的均值是各次peak的均值，不是时间平均显存。诊断计时不能支持加速。原因分析需结合原始profile和代码，不以这个自动摘要替代人工审查。

**证据及后续。** `outputs/memory_campaign_qwen17b-hinv-workspace-stable-layer2-20261011-workspace-v2/`、`outputs/hessian-workspace-full-followup-20261011-v2/records-qwen17b-hinv-workspace-stable-layer2-20261011-workspace-v2/qwen17b-hinv-workspace-stable-layer2-20261011-workspace-v2/summary.json`及原始每对comparison。完整量化与系统资源通过后才进入相同源码的低开销计时；尚未通过则先定位，不继续重复同一失败配置。


### 开发流程与条件候选（2026-10-11，待验证）

**理论支持、参考文献。** OLLA（https://arxiv.org/html/2210.12924v2）的张量寿命/存储位置原则，GMLake（https://arxiv.org/html/2401.08156v1）的碎片分析；只借鉴调度原则，未实现论文ILP或VMM系统。PyTorch copy_/cholesky/cholesky_inverse接口。

**方法具体内容。** 新真实solver捕获/重放工具，以及资源失败后才执行的所有线性模块稳定写回。后者保留消费H复用和求逆工作区，只改变最终量化权重存储，顺序前缀0–10层验证原3/10层风险，跳过重复Stage-0并校验缓存不变。详见 `docs/bottleneck_validation_protocol.md` 及 `docs/system_memory_active_queue_20261010.md`。

**实际效果。** 捕获CPU测试2项通过，真实生成方法/入口编译通过；新工具尚无CUDA捕获或all-module候选收益。活动workspace方案1.7B 0–2层严格通过，NVML4999→4407 MiB，后续正在运行。

**评价与反思。** 先前down_proj稳定写回已解决大量回放reserved回升，但完整运行仍有层3/10各36 MiB。需要分别检查求逆缓冲调度及其他长寿命权重的位置；其他模块是否导致残留问题还没有证据，不能先认定。新候选仅在前法资源失败后跑，不重跑已证伪同配置，不把CPU或局部组件通过当作终稿结论。

<!-- completed-memory-followup:qwen06b-hinv-workspace-stable-layer2-20261011-workspace-v2 -->
### 自动续记：qwen06b-hinv-workspace-stable-layer2-20261011-workspace-v2（2026-10-11 18:36:49，已完成审计）

**理论支持、文献与具体方法。** 本轮是数值正确资源失败后的solver工作区分配调度：保留原三步Cholesky/inverse/upper调用，用最后消费者后的H_work、chol作为out=目标，每步输入/输出不共享存储，保持原因子stride。仅非inplace damping使用工作区复用；原inplace分支回到原helper。之后仍复用消费H组存储、向既有down_proj权重copy_，不启用私有池。参考[OLLA](https://arxiv.org/html/2210.12924v2)的lifetime/location原则和PyTorch2.11 out=接口；不是算法替换，不改变REAL-Q的Adam、校准或量化参数；局部组件证明不能替代全程严格资源审计。

**实际配置与效果。** 模型 `/mnt/d/llamaModels/Qwen3-0.6B`，校准 `2×1024`，refresh microbatch `2`，量化截止层 `2`。已审计 `1/1` 对，完整模型量化验证 `False`，campaign valid `True`。

| 指标 | control均值 | candidate均值 |
| --- | ---: | ---: |
| 整卡采样peak MiB | 2753.0000 | 2753.0000 |
| 完整进程allocated peak MiB | 2018.0317 | 2018.0317 |
| 完整进程reserved peak MiB | 2130.0000 | 2130.0000 |
| CPU RSS peak MiB | 6001.6445 | 6001.6055 |

首对KL/PPL C `0.08332214504480362 / 21.569381713867188`，N `0.08332214504480362 / 21.569381713867188`。每对数值门槛 `[True]`，资源门槛 `[True]`；区间回升数量 `[0]`、边界回升数量 `[0]`。

**评价与反思。** 所有请求配对通过已测门槛；仍只覆盖保存的本地pilot规模，不是正式论文规模/4卡证明。 prefix不能称为完整量化；NVML为采样，不能证明连续每一时刻。这里的均值是各次peak的均值，不是时间平均显存。诊断计时不能支持加速。原因分析需结合原始profile和代码，不以这个自动摘要替代人工审查。

**证据及后续。** `outputs/memory_campaign_qwen06b-hinv-workspace-stable-layer2-20261011-workspace-v2/`、`outputs/hessian-workspace-full-followup-20261011-v2/records-qwen06b-hinv-workspace-stable-layer2-20261011-workspace-v2/qwen06b-hinv-workspace-stable-layer2-20261011-workspace-v2/summary.json`及原始每对comparison。完整量化与系统资源通过后才进入相同源码的低开销计时；尚未通过则先定位，不继续重复同一失败配置。

<!-- completed-memory-followup:qwen17b-hinv-workspace-stable-full-20261011-workspace-v2 -->
### 自动续记：qwen17b-hinv-workspace-stable-full-20261011-workspace-v2（2026-10-11 20:57:15，已完成审计）

**理论支持、文献与具体方法。** 本轮是数值正确资源失败后的solver工作区分配调度：保留原三步Cholesky/inverse/upper调用，用最后消费者后的H_work、chol作为out=目标，每步输入/输出不共享存储，保持原因子stride。仅非inplace damping使用工作区复用；原inplace分支回到原helper。之后仍复用消费H组存储、向既有down_proj权重copy_，不启用私有池。参考[OLLA](https://arxiv.org/html/2210.12924v2)的lifetime/location原则和PyTorch2.11 out=接口；不是算法替换，不改变REAL-Q的Adam、校准或量化参数；局部组件证明不能替代全程严格资源审计。

**实际配置与效果。** 模型 `/mnt/d/llamaModels/Qwen3-1.7B`，校准 `2×1024`，refresh microbatch `2`，量化截止层 `27`。已审计 `1/3` 对，完整模型量化验证 `True`，campaign valid `False`。

| 指标 | control均值 | candidate均值 |
| --- | ---: | ---: |
| 整卡采样peak MiB | 6629.0000 | 6089.0000 |
| 完整进程allocated peak MiB | 5605.1909 | 5027.6909 |
| 完整进程reserved peak MiB | 5872.0000 | 5332.0000 |
| CPU RSS peak MiB | 11987.1016 | 11983.8906 |

首对KL/PPL C `0.625495195388794 / 25.562191009521484`，N `0.625495195388794 / 25.562191009521484`。每对数值门槛 `[True]`，资源门槛 `[False]`；区间回升数量 `[2]`、边界回升数量 `[0]`。

**评价与反思。** 本轮未通过完整请求/严格门槛；即使局部peak下降也不能作为合格最终方案，失败配对保留，不剔除。 prefix不能称为完整量化；NVML为采样，不能证明连续每一时刻。这里的均值是各次peak的均值，不是时间平均显存。诊断计时不能支持加速。原因分析需结合原始profile和代码，不以这个自动摘要替代人工审查。

**证据及后续。** `outputs/memory_campaign_qwen17b-hinv-workspace-stable-full-20261011-workspace-v2/`、`outputs/hessian-workspace-full-followup-20261011-v2/records-qwen17b-hinv-workspace-stable-full-20261011-workspace-v2/qwen17b-hinv-workspace-stable-full-20261011-workspace-v2/summary.json`及原始每对comparison。完整量化与系统资源通过后才进入相同源码的低开销计时；尚未通过则先定位，不继续重复同一失败配置。

## 2026-10-11 21:10：末层峰值归因核对及有限值临时量方案

**理论支持与参考文献。** OLLA 的最后消费者与存储位置原则（https://arxiv.org/html/2210.12924v2）；PyTorch isfinite 返回与输入逐元素对应的布尔结果（https://docs.pytorch.org/docs/2.11/generated/torch.isfinite.html）。借鉴临时存储有界化，不声称实现 OLLA 的规划器。

**代码审查与定位修正。** CudaPhaseMemoryProfiler.mark(next_phase) 关闭旧区间后开始新名字的区间。此前将 `layer_27_mlp_down_proj_after_hessian_inverse` 最高峰统称为“求逆阶段”不准确。原始 exact jsonl 表明末组该区间直到 `layer_27_activation_replay_and_cleanup` 才关闭，包含后续量化、梯度刷新及清理。纯求逆是 before_hessian_inverse→after_hessian_inverse。只读审计脚本 tools/audit_hessian_peak_attribution.py，证据 outputs/hessian-peak-attribution-20261011-v1.json：control纯求逆4018.4668 MiB、后续5605.1909 MiB；candidate纯求逆3298.4668 MiB、后续5027.6909 MiB。这支持优化求逆留下的长寿命因子，但不支持把全部最高峰归因于数值分解的临时工作区。

**新方法具体内容。** tools/hessian_chunked_finite_prototype.py 将两个完整矩阵 isfinite().all() 改成沿连续维度的分块布尔 AND 归约。6144×6144布尔输出理论36MiB，分块输出预算1MiB；isfinite内部临时量的总体峰值必须实测。CUDA标量累积，每矩阵仍只有一次主机布尔判定，避免每块同步。保留全部三个数值分解、原dtype/stride、阻尼、重试、fallback，可分别对原solver或两缓冲区solver叠加。

**实际效果。** WSL CPU测试 tests/test_hessian_chunked_finite.py 3/3通过：NaN/正负Inf/空矩阵、FP32/FP64/BF16和两种布局，输入不变、mask大小约束，真实helper因子/stride/阻尼和identity fallback一致。GPU新收益未知。tools/replay_hessian_bounded_validation.py 在真实层27首个down_proj子组上对reference/workspace/finite/workspace_finite各独立进程3次，同时测峰值和诊断耗时；新CUDA语义检查包含两布局及NaN/Inf。排队器 outputs/hessian-bounded-last-layer-followup-20261011-v1，先等待现有捕获和all-weight串行队列结束，不并发CUDA。当前仅CPU与编译通过。

**评价与反思。** 工作区full首对仍层3/10 reserved各回升36MiB，和先前stable-write-only相同；两缓冲区API本身未解决剩余碎片，不重跑相同失败配置。36MiB恰与大布尔输出一致，仅是待检验线索，不能认定它就是回放区间回升的根因。新方法可能增加kernel启动耗时，或只减少局部临时量而不改变总峰值；必须先局部对照，再真实前缀及全流程验收。捕获所有来源/state/28层输出/KL/PPL尚在运行，局部重放不能恢复allocator历史，更不能作为终稿新增结果。

### 2026-10-11 22:04：真实末层Hessian捕获完成与独立轨迹核对

**理论、文献与方法。** 沿用真实solver捕获/组件重放开发协议，以及OLLA存储生命周期原则（https://arxiv.org/html/2210.12924v2）。从核验Stage-0缓存运行实际Stage-2，保存层3/10/27首个down_proj子组H及原求逆结果，保持原FP32行主序H和列主序因子，不改变量化/Adam算术。

**实际效果。** 三份6144×6144真实H已保存；均percdamp=.01、非inplace damping、need_hinv_init=False、无identity fallback。独立原始工件审查 `outputs/qwen17b-real-hessian-capture-20261011-v1/independent_trajectory_audit.json` valid=True：507个状态张量的state哈希、所有28层held-out输出和KL/PPL与已接受参考精确一致；41份绑定源码哈希无变化。KL .62549519538879395、PPL25.562191009521484。捕获控制器完成13项缓存前后内容核验后已进入solver回放；最终捕获/回放status尚未生成。

**评价与反思。** 当前证据证明真实fixture来自正确量化轨迹，不能证明新存储方案成功。原reference layer3首份重放已valid=True，其他方法/层仍在串行执行；不能提前汇总为全部通过。捕获包含CPU拷贝和大文件写入，时间与显存不用于新收益；fresh solver环境不恢复完整allocator历史。新有限值检查仍等待既有串行队列，GPU收益未知。

**组件阶段的部分结果（22:05，尚非全组通过）。** 层3三臂各3次均精确：reference allocated1116.0479/reserved1118MiB；early-release与workspace都allocated828.0479/reserved830MiB。两优化臂相同峰值，说明在此真实H的fresh环境里额外out=存储安排尚未超过早释放的峰值收益。是否有限值检查内部临时量限制进一步收益，应由后续bounded四臂实测判断，不能由矩阵尺寸反推成已确认归因。组件约0.67–0.69s/次，仅诊断，不计系统加速。


### 真实Hessian瓶颈捕获及组件重放

**理论支持与文献。** OLLA最后消费者与存储位置原则（https://arxiv.org/html/2210.12924v2）；PyTorch cholesky/cholesky_inverse 的out=接口。

**具体方法。** 复用已完成、来源绑定且内容核验的Stage-0缓存，只运行一次实际Stage-2，保存指定层首个down_proj子组的真实求逆输入/输出，保持dtype/stride。后续每方法独立进程重放。

**实际效果。** 见 `/mnt/d/gptq_plus_realq/outputs/qwen17b-real-hessian-capture-20261011-v1/status.json`，valid=True。捕获/重放不是新增全程优化收益；失败不从记录删除。

**评价与反思。** 解决开发阶段重复预计算与求逆筛查成本，未实现完整block/Adam状态恢复，不恢复CUDA allocator历史，不能据此认定层回放reserved回升已解决。只有全流程候选通过原资源与质量门槛，才可作为终稿收益。

### 2026-10-11 22:14：三处真实H求逆回放全部完成

**理论、文献与方法。** 按已记录OLLA最后消费者/存储位置原则，比较原helper、early-release和两缓冲区out=，保持三步求逆、dtype/stride及阻尼/回退。使用真实层3/10/27首个down_proj子组，独立进程各方法3次；无并发CUDA。

**实际效果。** 自动捕获status valid=True，cache_unchanged/source_unchanged=True。独立审核 `independent_replay_audit.json`：9个fixture-method组合、27次测量全部精确，H输入副作用、逆/因子hash及布局、阻尼/回退均满足worker门槛。三处reference每次allocated1116.0479/reserved1118MiB，early-release和workspace每次均allocated828.0479/reserved830MiB，局部allocated减少288MiB（约25.81%）。layer27也一致。完整轨迹与41源码核对见此前independent_trajectory_audit.json。

**评价与反思。** workspace的额外out=在这三处fresh solver环境均没有超过早释放的局部峰值收益；不继续重复相同代码。局部下降不能代表完整严格门槛已修复：已知两次full仍各有层3/10 reserved+36MiB。有限值检查的大临时量是假设，等待独立bounded对照，尚无其GPU结果。captured H的这些first-subgroup不能替代末组完整状态/allocator恢复。

**续接。** capture22:07:51结束后，all-weight cached screen已开始真实0–10层control，candidate尚待串行，需77写回覆盖及完整数值/已测资源审查。conditional full与bounded局部方案仍等待，未并发GPU；无新全程成功结论。

### 2026-10-11 22:38：有限值检查的实现依据与今晚验证约束

**理论支持与参考文献。** OLLA（https://arxiv.org/html/2210.12924v2）用于最后消费者后的缓冲复用；GMLake（https://arxiv.org/html/2401.08156v1）用于区分活跃张量与碎片，不声称复现其ILP/VMM系统。额外核对PyTorch v2.11.0原始实现 TensorCompare.cpp 的 isfinite：浮点路径为 `(self == self) * (self.abs() != infinity)`，除布尔结果还有abs/比较临时量。来源 https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/TensorCompare.cpp，已归档源码及SHA/provenance到 docs/literature，BibTeX项 pytorch211isfinite 是软件来源，不作为新研究论文。

**方法具体内容。** 针对真实1.7B 6144平方FP32求逆，保持原有分解/精度/阻尼/重试/回退，只分块执行相同有限值检查，CUDA标量AND累计；不每块读回CPU。完整abs理论144MiB、每布尔张量36MiB；这些尺寸说明分块有缩减临时量的机制，不能将全部相加直接当作实测峰值，也不能认定层3/10的reserved+36MiB就是该检查导致。求逆留下的因子与随后KL/Adam刷新仍须联动检查。

**实际效果。** 真实H捕获/原reference、early-release、workspace 27次回放已通过，见22:14记录。新finite仅CPU三项通过，四臂三次GPU筛查仍等待当前真实量化队列释放GPU；无新finite GPU收益。当前all-weight缓存0–10层control已结束，candidate已开始，尚待77次写回、全部28层held-out输出/state/KL/PPL、源码缓存及严格资源核对。PyTorch原始源码下载首次TLS失败，curl重试成功且文件35221字节；仅成功文件记录SHA，不把失败空文件当文献。

**评价与反思。** 候选仍沿用已接受checkpoint+CE系统，两臂的新增比较不能重复计入其历史收益。层级缓存开发不能宣称全流程峰值；独立组件不能证明allocator历史或末层后续刷新改善。user已授权继续整晚优化：局部通过后真实轨迹验收，失败先查实现与资源证据再选下一方法，维持GPU串行，不改活动冻结源码，不扫学习率/校准，不推送。平均速度仍为次级指标，只有数值和资源门槛通过后才做低开销计时。

### 2026-10-11 22:44：分块有限值检查的真实轨迹入口已准备，尚待GPU组件门槛

**理论/参考文献。** OLLA（生命周期/位置）、GMLake（碎片机制）、版本绑定PyTorch isfinite原始实现，沿用22:38来源。不修改Cholesky数值运算或REAL-Q Adam。

**具体方法。** 新 tools/hessian_bounded_solver_install.py 对顺序down_proj按开关安装workspace+分块有限值检查；其他模块及inplace damping走原helper，记录每次适用范围。入口构造核对installer、finite及workspace三个依赖SHA，原最终权重/H复用路径保持。新 tools/run_cached_bounded_finite_screen.py 接在现有GPU队列之后，只在实际layer27 workspace与workspace_finite各三次精确、两项局部peak均不回升且有下降时跑真实0–10层两臂；同时拒绝前一all-weight方案未查明的数值/缓存错误。候选组合含all-linear稳定写回，不能把组合全部收益单独归因finite。复用同一静态缓存，核对77次稳定写回、44次down_proj分块调用覆盖、28层输出/state/KL/PPL、exact区间/边界及NVML。通过后仍须fresh完整验收。

**实际效果。** WSL CPU新增installer检查2/2通过，证据 outputs/hessian-bounded-installer-cpu-20261011-v1.log：开关/目标module范围、inplace回退、因子dtype/stride/阻尼/输入副作用，真实归档入口替换编译及绑定边界失败检测。CLI编译/help通过。新的 outputs/qwen17b-cached-bounded-finite-screen-20261011-v1 控制器实际处于等待，尚未启动CUDA；活动CUDA仍为原all-weight候选（已到layer4）。没有此方法GPU或新全流程通过结论。

**评价与反思。** 提前实现必要接入以缩短串行组件通过后的等待，未提前放宽GPU/严格资源门槛。两模型正式规模、多卡及速度重复仍独立验证；该前缀工具不替代Stage0或最后层联动证明。更新同一heartbeat复查范围到新前缀及必要完整验收，继续按失败证据决策，避免只启动后台而不审查。

### 2026-10-11 22:47：边界验收缺口修正及既有成果补审

**理论、方法与来源。** 沿用PyTorch allocated/reserved定义与工程存储原则；本条是验收工具修正，不是新显存方法。旧phase_peaks将start/end两点取最大再比较，不能证明两个对应边界各自不回升。新只读 tools/audit_memory_boundary_points.py 按phase+出现次数+start/end+allocated/reserved逐值比较，不覆盖原审计；部分日志只定位问题、不能判完成。跨stage/rank/不同stream分别加身份，本次既有审计按每份文件的stage/kind/rank加键。

**实际效果。** 新工具CPU两项通过：较大起点不能隐藏终点回升，部分/缺失数据不得通过。对既有checkpoint+CE两模型各三对完整原始记录补审，6/6均valid=True，每点allocated/reserved无回升、无缺失；证据 outputs/accepted-two-model-start-end-boundary-audit-20261011-v1.json。既有全程成果没有被这一缺口推翻。

当前all-weight候选部分日志已出现layer3 activation replay结束reserved N比C高20MiB（allocated相同），证据 outputs/all-weight-start-end-boundary-partial-20261011-2245.json。86个逐值回升包含相邻区间重复记录的同一边界，不是86个不同高峰；部分数据未知后续质量。旧阶段max门槛当前也已找到55处部分差值回升，故该候选不会只因新补审而失败。继续完成当前前缀以核对所有state/28层/KL/PPL及来源，再审失败机制，不进入同配置完整重复。

**评价与反思。** 不能用时间平均/全程峰值下降替代用户逐阶段无回升要求。未改活动冻结比较器，新增point补审作为更严格验收；新finite前缀结束后必须补查每点才决定fresh全层。保留旧自动status及失败证据，最终报告应区分旧phase-max门槛与逐point门槛，而不是静默改口径。

### 2026-10-11 22:53：求逆之后仍存活的末组私有输入（待真实轨迹验证）

**理论支持、参考文献。** OLLA最后消费者与存储生命周期（https://arxiv.org/html/2210.12924v2）及Checkmate对存储/计算依赖的分析（https://arxiv.org/abs/1910.02653v3）。这里只实施最后消费者后的显式局部变量释放，不实现ILP/MILP，不重算或改变数值。

**代码依据与具体方法。** 原顺序fasterquant 4918/4932附近对每组H_sub clone/permute，4969求逆；5000的del H_sub仅在hessian_cpu_offload时执行。5030–5075将所需Hinv及其他状态入subgroup_states，5081后进入列块量化/Adam。AST审查这一边界之后没有H_sub读取。非CPU路径最后一份私有H_sub因此留到函数结束；6144平方FP32是144MiB，0.6B3072平方为36MiB，仅张量尺寸估算，不是已测峰值收益。新 tools/hessian_dead_subgroup_input_prototype.py 在所有组设置完成之后、块循环之前释放这个局部，限制down_proj/alpha0/非HCPU；保留全部因子，并检查H_sub不与retained factor共享底层存储。记录CPU元数据，不持有已死GPU张量。生成源显式来自绑定fasterquant，不把observer当原源码；若新代码增加后续消费者，AST拒绝补丁。

**实际效果。** WSL CPU2/2通过，证据 outputs/hessian-dead-subgroup-input-cpu-20261011-v2.log：真实顺序方法生成/编译、后续新消费者拒绝，原注入语句的weakref释放、因子值保留、alias错误拒绝。v1曾因测试构造未来消费者时缩进错误而失败，保留v1日志；修正测试插入位置及缩进后v2通过，不将此误记为CUDA方法失败。尚未接入真实GPU量化，没有声称降144MiB峰值；当前有限值组件/前缀顺序不改。

**评价与反思。** 这是直接面向after_hessian_inverse后续区间的生命周期候选，区别于纯分解临时量；原工程注释已指出私有输入没有后续消费者，存在清理机会并不意味着它已产生全程收益。是否降低NVML/allocated/reserved、是否引起碎片迁移，必须组件/真实前缀再全流程审查。分块finite结果出来后再根据失败位置决定独立或组合验证，不盲目堆叠方法，也不改正在运行的源码。

**验收工具补充。** start/end逐点工具已加入stage、日志类型及rank身份，三项CPU检查通过，避免多stage/多卡同名phase被聚合。原六对补审已在调用方添加相同身份，仍有效；不覆盖旧结果。

### 2026-10-11 23:00：all-weight前缀worker完成，独立质量与边界审查

**理论/方法。** 沿用OLLA位置/生命周期假设，对所有7个线性模块稳定最终权重写回，在既有workspace/消费H复用上组合；same校准、原REAL-Q算术，0–10层真实前缀。此条不是新增算法改进。

**实际效果。** 两侧worker已完成；独立读取原始metrics、28层held-out指纹、日志KL/PPL及77个写回记录：507个状态张量hash精确一致，所有28层输出完全一致，KL .32509875297546387、PPL20.781702041625977均相同；两臂各77写回，候选全部保持指针，控制开关全部关闭。按stage/log-kind/rank/phase/occurrence分开逐起点/终点审查4644个数值，无缺失，但316个reserved对应值回升（相邻边界可能重复记录，不等于316个独立高峰）。证据 outputs/qwen17b-cached-all-weight-screen-20261011-v1/independent_completed_worker_point_audit.json。全cache/source最终门槛仍在控制器末尾哈希中，暂不称所有审查完成。

**评价与反思。** 质量正确不能替代显存门槛。将稳定写回泛化到全部模块尚未达到严格无回升，不进入相同失败候选的完整三次重复。component finite接续仍等控制器正常写出失败状态和GPU释放后才执行，先追查局部临时量；有数值错误则改bug而非直接切换。已有checkpoint+CE六对逐点复查通过，不被本新失败推翻。

**CPU原型续记。** dead H_sub入口补充已有CPU-offload、非down_proj和alpha非零短路保护测试，v3两项仍通过，保留v1/v2日志；尚未GPU验证。start/end审计工具三项测试通过。


### 条件续接：所有线性模块权重存储位置

**理论/参考。** OLLA最后消费者与位置（https://arxiv.org/html/2210.12924v2），GMLake碎片化分析（https://arxiv.org/html/2401.08156v1）。

**具体方法。** 仅在两缓冲区全流程出现数值正确的资源失败、真实捕获/重放通过后，将稳定最终权重存储从down_proj推广到所有顺序量化线性模块，保持转换/补偿/Adam数学不变。两臂复用同一核验缓存，跑真实0–10层前缀保留Stage-2分配历史，覆盖先前层3/10问题。

**实际效果。** `/mnt/d/gptq_plus_realq/outputs/qwen17b-cached-all-weight-screen-20261011-v1/status.json`；valid=False，skipped=False。层27及Stage-0未验证，不能写成新全程收益。

**评价与反思。** 其他权重钉住大段是待验证假设；若失败先审查逐区间差值，不用校准/学习率调整掩盖资源失败，重复完整实验留给稳定候选。


### 所有模块稳定写回：条件完整验收

**理论/参考与方法。** 延续OLLA位置/寿命原则及GMLake碎片分析；前缀严格通过后，逐对fresh Stage-0+28层量化+评估，两模型各三对；任何一对失败不继续相同配置重复。

**实际效果。** `/mnt/d/gptq_plus_realq/outputs/all-weight-full-followup-20261011-v1/status.json`；valid=False，skipped=False，完成配对数=0。

**评价与反思。** 完整结果必须保留所有阶段峰值/区间、全部28层及canonical state/KL/PPL核对，局部通过不作完整结论。诊断计时不是速度证据；此处仍是本机pilot，不是正式2048样本或多卡验收。

### 2026-10-11 23:01：all-weight最终失败验收与下一组件实际启动

原控制器已完成：数值exact、缓存前后不变、write覆盖通过、全局显存峰门槛通过，NVML5033→4405MiB（这里只是Stage2-only前缀）。旧阶段max审计174项回升、逐point补审316项，故valid=False。39份绑定源码独立SHA核对全部不变，证据independent_final_controller_audit.json及原status。失败来自资源，未发现本轮数值/缓存/源码错误。原full等待器已按门槛拒绝启动任何全层GPU，不重跑原配置。

bounded组件已实际开始，目录 outputs/hessian-bounded-last-layer-followup-20261011-v1/replay/，当前layer27 reference独立worker；后续workspace/finite/workspace_finite各三次。新finite真实前缀仍只等待，有明确局部门槛，没有第二个GPU并发。组件收益暂未知，不混淆原all-weight失败与新finite成功。


### 末层真实求逆：分块有限值检查局部对照

**理论与文献。** OLLA张量寿命/临时存储调度（https://arxiv.org/html/2210.12924v2），PyTorch isfinite逐元素布尔判定；AND归约的精确等价，不替换数值分解。

**具体方法。** 捕获的layer27 down_proj首组Hessian，reference/workspace/finite/workspace_finite，各独立进程三次；保留三步分解、stride、阻尼、重试和fallback。分块检查在CUDA归约标量，每矩阵仅原有一次主机判定，不逐chunk同步。

**实际效果。** valid=True，证据 `/mnt/d/gptq_plus_realq/outputs/hessian-bounded-last-layer-followup-20261011-v1/status.json` 及replay各臂报告。局部结果不计全程显存/速度成果，不能修复末层后续KL刷新或层3/10碎片的证明缺口。

**评价与反思。** 6144平方的布尔输出理论36MiB，分块输出预算1MiB；isfinite内部临时量及kernel启动时间仍须实测。必须在完整真实轨迹各已测区间无回升后验收。

### 2026-10-11 23:05：真实末层有限值检查四臂三次局部对照完成

**理论支持、参考文献。** OLLA最后消费者/临时存储调度（https://arxiv.org/html/2210.12924v2）与版本绑定PyTorch TensorCompare.cpp浮点isfinite实现（https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/TensorCompare.cpp）；浮点abs/布尔比较临时量沿连续维度分块，同一CUDA标量AND合并，保留原有两次矩阵级主机判定、不每块同步。未替换分解、数值、阻尼/回退。

**方法具体内容。** 来自已独立验证正确量化轨迹的layer27 down_proj首个子组，真实6144平方FP32 H，各方法独立进程预热后连续三次。reference/workspace/finite/workspace_finite分别是原helper、两缓冲区out=、原helper加分块检查、两缓冲区加分块检查。仅局部fresh allocator环境，不恢复完整轨迹。

**实际效果。** 12次测量均精确，H输入副作用、逆/因子hash与stride、阻尼及fallback通过，各同方法三次峰值完全相同。3份绑定新源码哈希无变化；独立persisted报告审查valid=True，证据 outputs/hessian-bounded-last-layer-followup-20261011-v1/independent_component_audit.json。

| 方法 | allocated峰MiB（三次相同） | reserved峰MiB（三次相同） | 三次诊断秒数（全部保留） |
| --- | ---: | ---: | --- |
| reference | 1116.0479 | 1118 | .690386, .681842, .683318 |
| workspace | 828.0479 | 830 | .687352, .678415, .676548 |
| finite | 871.0210 | 888 | .937350, .685665, .949920 |
| workspace_finite | 720.0483 | 724 | .962128, .669962, .689679 |

相对workspace，新增finite的局部allocated下降107.9995MiB（约13.04%），reserved下降106MiB（约12.77%）。相对reference的组合降幅不是finite单项收益。finite独立也减少峰值；组合最终峰720MiB说明不能简单用252MiB内部临时量尺寸相减推断实际新峰，其他重叠分配成为局部上限。

**评价与反思。** 本轮证明有限值检查确有可削减的局部工作存储，不证明层3/10 reserved问题由它导致，不证明最后after区间KL刷新峰被同幅削减。诊断耗时finite有显著波动、workspace_finite首次数高，全部保留，不剔除或直接宣称加速。分块会增加kernel启动，需全流程通过后单独低开销计时。独立真实prefix/full必须每个start/end对应点、阶段峰、process/NVML峰均不回升，state/28层/KL/PPL/缓存源码一致。

**后续实际状态。** 新finite-prefix控制器已通过局部门槛并进入cache SHA准备，将串行跑C/N真实0–10层（组合包括all-linear稳定写回）。该组合效果不能归为finite单项，尚无新前缀/full通过结论。此前all-weight方案是数值正确、资源失败，未进行重复full。总进度reader已从旧队列交给bounded，再交给finite-prefix，单一日志继续logs/realq-system-restart-20261010-1408.log。


### 2026-10-11 23:14：逐点验收工具与finite组合的fresh全流程续接

**理论支持、参考文献。** 延续OLLA（https://arxiv.org/html/2210.12924v2）张量位置/寿命、GMLake（https://arxiv.org/html/2401.08156v1）碎片分析以及版本绑定PyTorch浮点isfinite实现（https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/TensorCompare.cpp）。这是系统存储调度组合的严格验证基础设施；不将新增验收脚本本身计作算法/显存收益。

**方法具体内容。** 新 `tools/run_bounded_finite_full_followup.py` 等当前finite-prefix结束，独立对start/end逐点比较，绑定prefix及后续入口/验收源码。仅质量、来源、所有区间和逐边界门槛通过才串行fresh Stage0+28层+评估；每对检查canonical state/全部28层/KL/PPL、196模块写回、112次down solver/有限值覆盖、区间/逐点/全局峰值，再决定下一次。避免直接复用旧不含finite的full入口或重复profile包装，保留原控制器与结果。完成prepared权重按字节核验后hard-link整理磁盘，Fisher和路径保留，此操作不计作GPU优化。

**实际效果。** WSL CPU三项通过，日志 `outputs/bounded-finite-full-gate-cpu-20261011-v1.log`：较大起点不能掩盖终点回升，遥测缺失拒绝，源码变化拒绝；真实未profile入口生成可编译且fresh Stage0不被禁用，依赖哈希全部注入。新条件控制器已启动等待；未启动full GPU或取得其收益。当前唯一GPU为finite前缀control，23:14推进layer4，尚待candidate。

**评价与反思。** 局部720MiB正向并不等价于真实量化轨迹reserved改善，必须保留前缀及fresh全流程门槛。旧边界max缺口已通过独立逐点工具补齐，并应用于新full准入/每对继续条件。若新前缀资源失败，不重复相同失败全层实验，而依据失败位置继续存储生命周期改进；若数值/缓存出错先修复。当前结果不可用于新终稿全程收益或加速结论。


### 2026-10-11 23:37：finite真实前缀的部分边界诊断（非最终验收）

**理论/参考、具体方法。** 沿用OLLA存储生命周期和GMLake碎片分析，对同一只读cache的C/N真实0–10层量化，独立比较每个start/end；不改变活动代码、校准或Adam参数。

**实际效果。** control已完成，state4114c7f0...、KL .32509875297546387/PPL20.781702041625977，与前轮相同control一致。候选推进layer3时部分1624个匹配值有68个reserved回升，最大18MiB；layer0首个after_symmetrize边界+4MiB、after_inverse+6MiB。证据outputs/finite-start-end-boundary-partial-20261011-2337.json。尚无候选最终数值、cache/source或全程验收，不将该部分观察记作方法最终成败。

**评价、反思。** 局部solver三次720MiB不能保证真实allocator历史下所有边界无回升。首子组求逆早期回升发生在拟释放末组H_sub之前，后者单独无法修复这些早期边界；不能无依据叠加后宣称完整修复。保留本轮直到质量/来源审查结束，以排除数值bug；完整资源失败则拒绝重复相同full，按实际位置设计生命周期或临时存储布局新候选。


### 2026-10-11 23:53：末组私有H输入释放的真实入口接入准备

**理论支持、参考文献。** 延续已归档OLLA（https://arxiv.org/html/2210.12924v2）的最后消费者/存储位置，以及Checkmate（https://arxiv.org/abs/1910.02653v3）的内存依赖调度思想。采用显式死亡点清理，不实现其优化求解器，不进行重算、CPU卸载或数值近似。

**方法具体内容。** 新 `tools/hessian_dead_subgroup_install.py` 接到已验证workspace/消费H复用/down_proj稳定写回的生成builder，在最后子组设置完成、列块量化/Adam开始前释放无消费者的私有H_sub，保留四组因子并保护alias，仅存CPU元数据。下一候选不推广all-linear稳定写回、不加入finite，便于针对原workspace层3/10回放问题及末组后续刷新区间筛查；不是给当前失败组合盲目堆叠。控制仍是既有checkpoint+CE系统、候选是组合；新释放增量需结合已有workspace结果区分归因。

**实际效果。** 两项WSL CPU接入检查通过，原生产builder生成方法可编译，末组释放位置/因子alias保护保留，真实入口源hash守卫及失败边界拒绝；证据 `outputs/hessian-dead-subgroup-installer-cpu-20261011-v1.log`。实际缓存profile入口生成也已编译，证据 `outputs/hessian-dead-subgroup-profile-entry-cpu-20261011-v1.json`；CLI通过 `outputs/dead-subgroup-screen-cli-20261011-v1.log`。只是CPU准备，无新GPU/显存收益。

**评价与反思。** 新 `tools/run_cached_dead_subgroup_screen.py` 条件等待当前finite前缀：质量/缓存/覆盖/来源全部正确且完整区间或独立逐点资源失败时，才串行执行新C/N0–10层；若finite严格通过，跳过此分支让其full继续。新前缀直接用独立start/end审计作准入条件，并核对11次私有输入清理、44次solver及11次down写回、所有state/28层/KL/PPL/cache/source。释放点之后可能降低144MiB活跃张量，但并非实测peak收益，更不能回溯修复当前finite发生在释放点之前的回升；真实前缀通过后仍需fresh完整首对，严格通过才重复/0.6B交叉验证。新等待器目录 `outputs/qwen17b-cached-dead-subgroup-screen-20261011-v1`，工具会话32295，当前GPU仍是finite候选layer10。


### 有限值临时量：真实0–10层联动检查

**理论/参考。** OLLA最后消费者与位置（https://arxiv.org/html/2210.12924v2），GMLake碎片化分析（https://arxiv.org/html/2401.08156v1）。

**具体方法。** 仅在两缓冲区全流程出现数值正确的资源失败、真实捕获/重放通过后，在workspace及所有线性模块稳定写回组合上加入分块有限值检查；仅真实layer27组件三次资源及数值门槛通过才启动。保持转换/补偿/Adam数学不变。两臂复用同一核验缓存，跑真实0–10层前缀保留Stage-2分配历史，覆盖先前层3/10问题。

**实际效果。** `/mnt/d/gptq_plus_realq/outputs/qwen17b-cached-bounded-finite-screen-20261011-v1/status.json`；valid=False，skipped=False。层27及Stage-0未验证，不能写成新全程收益。

**评价与反思。** 有限值内部临时量及长寿命位置共同影响allocator历史；组合通过不能单独归因于finite。若失败先审查逐区间差值，不用校准/学习率调整掩盖资源失败，重复完整实验留给稳定候选。


### 分块有限值组合：逐点门槛下的条件完整验收

**理论/参考与方法。** 延续OLLA位置/寿命原则及GMLake碎片分析，结合版本绑定PyTorch有限值临时量；前缀阶段峰及独立start/end严格通过后，逐对fresh Stage-0+28层量化+评估，两模型各三对；任何一对失败不继续相同配置重复。

**实际效果。** `/mnt/d/gptq_plus_realq/outputs/bounded-finite-full-followup-20261011-v1/status.json`；valid=False，skipped=False，完成配对数=0。

**评价与反思。** 完整结果必须保留所有阶段峰值/区间、全部28层及canonical state/KL/PPL核对，局部通过不作完整结论。诊断计时不是速度证据；此处仍是本机pilot，不是正式2048样本或多卡验收。


### 2026-10-11 23:56：finite前缀两臂完成，独立资源审查未通过

**理论/方法、参考。** 沿用OLLA临时量生命周期和版本绑定PyTorch有限值检查证据，workspace+消费H复用+all-linear稳定写回+分块finite的真实0–10层前缀；对应控制为既有checkpoint+CE系统，原算法/校准不变。

**实际效果。** 独立读取两臂完成worker，507状态、所有28层输出、KL .32509875297546387/PPL20.781702041625977均精确，77模块写回及44次finite solver覆盖通过，42源码SHA不变。4644对应start/end值无缺失，但316个reserved值回升，最大32MiB；旧区间max174项回升。NVML5033→4405MiB，process allocated4117.5854→3541.4951MiB、reserved4276→3648MiB，仅Stage2-only前缀。证据finite目录independent_completed_worker_audit.json。缓存最终控制器SHA尚在核对，暂不写全来源最终通过。

**评价、反思。** 全局峰下降仍不足以通过用户严格门槛；局部finite108MiB收益没有消除全轨迹早期/回放reserved回升，不进入相同配置fresh全层重复。该负向结果是资源问题的独立证据，不能归因为数值改变；缓存核对仍需完成。下一条件候选回到down-only原workspace组合并缩短末组私有H_sub死亡后的寿命，针对求逆后量化/刷新重叠峰；CPU接入通过不代表该方向GPU改善，必须新目录真实前缀后再完整验收。


23:56最终来源审查补充：finite controller已valid=False，cache_unchanged=True，独立42源码SHA仍不变，quality/coverage全部通过；证据independent_final_controller_audit.json。finite full按门槛拒绝，完成配对数0。dead条件控制器已通过此前质量/cache/source+资源失败检查，进入新缓存SHA准备，暂不占GPU；原GPU队列已结束，后续仍串行。


### 2026-10-12 00:06：末组私有输入方案推进及条件完整验收准备

**理论支持、参考文献。** 采用 [OLLA](https://arxiv.org/html/2210.12924v2) 的最后消费者/存储生命周期分析，以及 [Checkmate](https://arxiv.org/abs/1910.02653v3) 的内存依赖原则。这里只显式释放已死私有输入，不实现其优化求解器或重算策略，不进行数值近似。

**方法具体内容。** 新 `tools/run_dead_subgroup_full_followup.py` 生成不含finite/all-linear扩展的fresh入口，接续workspace+消费H复用+down-only稳定写回+末组私有H_sub释放。等新dead前缀最终状态、全部质量/cache/source/覆盖及区间/分别start-end显存门槛通过才执行fresh Stage-0+28层；先1.7B三对，再0.6B三对，每对严格验收后才下一对。每臂另与既有已接受参考核对canonical state、所有28层输出和KL/PPL，检查112次求逆、28次down写回及候选28次清理；清理字节数从实际FP32形状推导，分别144/36MiB。这是理论释放量，尚不是实测峰值降幅。

**实际效果。** 两项WSL CPU检查通过，包括两种模型宽度的覆盖/缺失拒绝、实际raw入口编译和fresh预计算保护，证据 `outputs/dead-subgroup-full-gate-cpu-20261012-v1.log`。条件完整队列目录 `outputs/dead-subgroup-full-followup-20261012-v1`，日志 `logs/dead-subgroup-full-followup-20261012-v1.log`，工具会话68833、等待器PID1830，目前只CPU等待，不占GPU。唯一CUDA任务仍新dead前缀控制组PID1759，00:03:47到layer2 down_proj；候选尚未测完，不能写新GPU收益。独立核对当前dead前缀44份绑定源码SHA无变化。

**评价与反思。** 上一finite前缀全局峰下降但逐点reserved回升，已拒绝full且保留完整负向证据。本方案依赖真实死亡点，不保证解决所有allocator回升；只有全部对应边界不高于控制才允许full。完整入口准备/后台排队不是实验成功；若前缀失败应独立查明位置继续有依据的工程改进，不重复相同失败配置。诊断计时不作速度结论；即使本机两模型完成也仍需正式2048校准样本和服务器多卡验证。


### 2026-10-12 00:20：新对照组的显存边界复现核对（部分）

**理论/参考与方法。** 继续使用OLLA生命周期工程候选；在候选结果出来之前，独立比较新dead control与已完成finite control的同阶段/同rank/同occurrence/start-end allocated和reserved，检查更换CPU入口观察器是否改变控制轨迹。没有新数值方法或新文献应用。

**实际效果。** 当前控制组到layer10 down_proj，4564个已对应显存边界数值与前次控制逐项相等，0项变化；80个后续边界尚未出现，不用于完成验收。证据outputs/dead-control-repeatability-partial-20261012-0020.json。44份活动绑定源码仍无变化；GPU仍仅PID1759，完整等待器PID1830仅CPU等待。

**评价、反思。** 对应控制显存轨迹在已测区间能复现，支持后续区分候选存储改变与基础对照漂移；不能把部分控制重复性当作新方法显存/速度收益，也没有确认候选数值或最终资源门槛。必须继续等待候选及完整source/cache验收。


### 2026-10-12 00:31：dead候选仍有回放瞬时reserved回升，准备最后消费者清理原型

**理论支持、参考文献。** 本轮重新核对 [OLLA](https://arxiv.org/abs/2210.12924) 的张量寿命/位置优化及 [Checkmate](https://arxiv.org/abs/1910.02653) 的依赖调度原则；并查阅 [PyTorch 2.11 CUDA内存管理](https://docs.pytorch.org/docs/2.11/notes/cuda.html)。活跃张量引用与allocator保留空间不同，因此释放已死引用能减少活跃需求，但不保证reserved同步下降。此处采用最后消费者分析，不实现ILP或重算算法。

**方法与代码依据。** 新dead控制组完整507状态/28层输出/KL/PPL与此前control精确一致；完整4644边界allocated/reserved也逐项一致。当前候选末组后的量化/刷新区间峰有下降，但layer3 activation_replay_and_cleanup瞬时reserved仍+36MiB，allocated峰相等、该区间起终点均未回升。这说明仅核对起终点会漏掉瞬时资源回升，也说明释放最后H_sub尚未解决此回放区间问题；具体36MiB分配根因未确认。

核心free()只清self.H/Losses并清cache；gptq字典还保留GPTQ对象，saliency_dict/gradients_dict亦在回放结束后才del。新的tools/activation_replay_stats_lifetime_prototype.py将原有del gptq、del saliency_dict/gradients_dict移动到after_layer_quantized hook之后、回放之前，AST拒绝后续消费者/重绑定，保持输入/参数/前向数学不变。存储引用图可能提前释放量化统计，是否解除碎片或峰值仍是假设；不能将36MiB直接归因于这些统计。先准备独立新候选，不改当前活动冻结入口。

**实际效果。** 独立部分证据outputs/dead-interval-and-boundary-partial-20261012-0031.json：424个已结束区间，1处reserved+36MiB、起终点0处回升；不是完成资源验收。候选仍在运行，最终state/cache/source未审完，full仍只CPU等待。原型CPU两项通过（两次层迭代的最后消费者/弱引用死亡点、真实生产函数生成编译及新增消费者拒绝），outputs/activation-replay-stats-lifetime-cpu-20261012-v2.log。v1为合成fixture移走del后留下空if的测试缩进错误，已修测试并保留日志；真实生产函数编译在v1也通过。尚未接到真实入口或测GPU，无新显存/速度结果。

**评价、反思。** 优先处理实际回升区间中已死对象的生命周期，不重复相同失败full，也不通过empty_cache或参数调整假装数值方法改善。当前dead必须先完成全部质量/缓存/来源审查；只有确认质量正确且资源失败才继续新原型的真实入口接入及串行小规模验收。局部降幅不归因纯求逆，也不替代终稿全流程/两模型三对及正式多卡验证。


### 2026-10-12 00:58：回放统计寿命接入与中断恢复

**理论支持、参考文献。** 沿用OLLA（https://arxiv.org/abs/2210.12924）的最后消费者/位置分析及Checkmate（https://arxiv.org/abs/1910.02653）的依赖原则；只前移原有删除操作，不实现其ILP/重算求解器，不改变REAL-Q数学。引用字段、活跃allocated与allocator reserved不同，统计持有字节不等于峰值释放量。

**具体方法。** tools/activation_replay_stats_install.py在原测量包装捕获original_fwrd之前接入生成函数，保留真实no_grad和实时模块globals。两臂共享不保留GPU张量的CPU观察器，按CUDA storage去重统计gptq.gradients/act_square及saliency/gradients_dict；候选在after_layer_quantized hook后、回放标记前执行原有del。新tools/run_cached_replay_stats_screen.py要求11层覆盖、同层统计storage字节相等及全部数值/源码/缓存/区间/分别起终点/全局门槛，不叠finite或all-linear扩展。

**实际效果。** 原型CPU2项、接入CPU4项通过，含真实入口编译、no_grad、实时globals、两层死亡点、去重且无张量保留；此前测试fixture空if及控制观察器锚点歧义已修并保留失败日志。实际生成入口仅编译通过，GPU收益未知。00:54核对发现旧dead候选在评估14/28附近中断，全部WSL实验/CPU等待器已消失；Windows主机未重启，具体中断原因未确认。已保存507状态与控制相同；层指纹文件列出28层，但仅0–14层完成且相同，15–27层为未计算的空指纹，不能称28层一致；44绑定源码无变化，但最终KL/PPL/进程峰/缓存复核缺失，见旧目录independent_interrupted_worker_audit-20261012-0058.json。不得补造完成status。

**评价与反思。** old dead已有layer3回放reserved瞬时+36MiB证据，统计生命周期仍只是假设，未确认该分配根因。本次相同dead配置仅为中断后完成质量与来源验证，不作为寻优重复或成功重复。tools/run_replay_stats_interrupted_recovery.py在新目录串行恢复两臂，随后才按确认数值正确而资源失败的门槛接续新统计方案；原文件保留，不并发GPU。完整验收还需新统计fresh入口，两模型各三对及正式多卡尚未证实。


00:59完整续接准备：tools/run_replay_stats_full_followup.py新入口CPU2项通过，保留fresh Stage0/单测量包装及28层stats缺失/字节改变拒绝。会话93826、CPU等待器61等待恢复stats-v2 status；目录outputs/replay-stats-full-followup-20261012-v1。前缀数值/cache/覆盖及每区间/每point/global严格通过才fresh两模型各三对，每对再核对canonical全部数值与112inverse/28write/28dead/28stats同层storage持有量。尚无新GPU收益，不使用原中断olddead/full队列。恢复控制组当前唯一CUDA51，controller17/dead19/reader21；每次实际ps更新。检查点outputs/realq-continuation-20261012-0100-recovery.json。旧指纹0–14完成相同、15–27空，28行列表不是28层验收；统计字节不是峰下降。


### 末组已死私有H输入：真实缓存前缀条件验证

**理论/参考。** OLLA最后消费者与位置（https://arxiv.org/html/2210.12924v2），Checkmate存储依赖与调度（https://arxiv.org/abs/1910.02653v3）。

**具体方法。** 仅在两缓冲区全流程出现数值正确的资源失败、真实捕获/重放通过后，回到workspace/消费H复用/down-only稳定写回组合，在块循环前释放最后已死私有H_sub，保留四份因子；不启用finite/all-linear扩展。保持转换/补偿/Adam数学不变。两臂复用同一核验缓存，跑真实0–10层前缀保留Stage-2分配历史，覆盖先前层3/10问题。

**实际效果。** `/mnt/d/gptq_plus_realq/outputs/qwen17b-cached-dead-subgroup-screen-20261012-recovery-v2/status.json`；valid=False，skipped=False。层27及Stage-0未验证，不能写成新全程收益。

**评价与反思。** 此处只证明释放点之后的存储生命周期；不声称可修复先前finite首子组早期回升。若失败先审查逐区间和逐点差值，不用校准/学习率调整掩盖资源失败，重复完整实验留给稳定候选。


### 2026-10-12 01:43：中断恢复完成，dead私有输入释放仍未通过区间门槛

**理论支持、参考文献。** OLLA最后消费者/存储位置与Checkmate依赖原则，沿用前述主文献；本次无新数值方法。只释放已死私有H_sub，保护四份因子，不能修复所有reserved碎片。

**方法具体内容。** 原v1评估被中断，v2新目录完整恢复同配置0–10层两臂以完成验收，不将此次运行视为寻优重复。独立复核全部状态/28层非空输出/KL/PPL、44solver11稳定写回11候选清理、44源码，以及精确区间和分别起终点；来源缓存另有独立SHA复核。

**实际效果。** outputs/qwen17b-cached-dead-subgroup-screen-20261012-recovery-v2/independent_completed_worker_and_controller_audit.json：507状态/all28/KL .32509875297546387/PPL20.781702041625977精确，覆盖和44源码不变，controller缓存核对通过。4644对应边界均无回升/缺失，但layer3、10 activation_replay_and_cleanup瞬时reserved各+36MiB，allocated相等，区间边界reserved反而-12MiB。全局NVML5033→4309MiB，allocated4117.5854→3397.4951MiB、reserved4276→3552MiB，仅Stage2-only前缀，非全模型全部量化结果。诊断quantization秒数998.966→1102.834，只有一对、不能宣称加速。顶层valid=False，不能进入相同配置full。

**评价、反思。** 删除末组私有输入后全局峰进一步下降，但未消除回放瞬时reserved门槛；必须保留区间峰与逐point双重核查，point全部通过不足以验收。本次没有算法质量问题，资源回升可复现；不能把后续峰归为纯求逆或以全局下降替代局部门槛。新统计死亡点方案已依前置门槛串行进入真实GPU控制组，未有新收益；两臂观察保持一致，假设与36MiB根因仍未确认，待完整C/N验收再判断。


01:44缓存独立补审：首次Windows普通路径枚举仅12份、漏掉长文件名Fisher，属于超过MAX_PATH的验收工具缺口，不是缓存变动；保留false证据。使用Windows extended path重新枚举/流式SHA全部13份，与原manifest逐项相同，见dead恢复目录independent_final_cache_extended_path_audit.json。源码/缓存与全部质量至此独立核验正确，失败原因仍仅layer3/10回放区间reserved各+36MiB。后续跨平台独立审核必须保留完整文件名/数量，不将is_file静默False当真实文件缺失。


### 2026-10-12 02:01：统计寿命观察器的真实控制覆盖

**理论/方法及参考。** 延续OLLA最后消费者与Checkmate存储依赖；两臂共享storage去重观察，控制原删除时机不变，区别持有量、释放量及reserved区间峰。

**实际效果。** 新stats-v2控制组完整结束，independent_completed_control_audit.json确认507state/全部28层非空输出/KLPPL与前控制精确，4644显存边界逐值相同，46绑定源码不变。每层回放前已死统计观察到48.0234375MiB CUDA storage，11层覆盖正常；这仅是持有量，不是实测峰下降或36MiB根因。候选唯一GPU299已从第0层实际运行，full61仍CPU等待。

**评价/反思。** 共享CPU元数据观察在这次控制组未改变显存边界或数值，支持两臂后续对照；仍需候选完整all28/状态/KLPPL、storage字节一致、cache/source及区间/逐point/global验收。不能由48MiB持有量推断reserved会减少48MiB，也不能将控制完成算成新方案成功。


### 回放前已死量化统计清理：真实缓存前缀条件验证

**理论/参考。** OLLA最后消费者与位置（https://arxiv.org/html/2210.12924v2），Checkmate存储依赖与调度（https://arxiv.org/abs/1910.02653v3）。

**具体方法。** 仅在两缓冲区全流程出现数值正确的资源失败、真实捕获/重放通过后，保持workspace/消费H复用/down-only稳定写回/最后私有H释放组合，在after_layer_quantized hook后、回放前前移原有gptq及统计dict删除；不启用finite/all-linear扩展。保持转换/补偿/Adam数学不变。两臂复用同一核验缓存，跑真实0–10层前缀保留Stage-2分配历史，覆盖先前层3/10问题。

**实际效果。** `/mnt/d/gptq_plus_realq/outputs/qwen17b-cached-replay-stats-screen-20261012-recovery-v2/status.json`；valid=True，skipped=False。层27及Stage-0未验证，不能写成新全程收益。

**评价与反思。** 提前统计死亡点是否消除回放reserved回升需要实测；统计bytes观察不等同真正释放bytes或全程峰值。若失败先审查逐区间和逐点差值，不用校准/学习率调整掩盖资源失败，重复完整实验留给稳定候选。


### 回放统计寿命清理：逐点门槛下的条件完整验收

**理论/参考与方法。** 延续OLLA最后消费者/寿命及Checkmate内存依赖原则；前缀阶段峰及独立start/end严格通过后，逐对fresh Stage-0+28层量化+评估，两模型各三对；任何一对失败不继续相同配置重复。

**实际效果。** `/mnt/d/gptq_plus_realq/outputs/replay-stats-full-followup-20261012-v1/status.json`；valid=False，skipped=False，完成配对数=0。

**评价与反思。** 完整结果必须保留所有阶段峰值/区间、全部28层及canonical state/KL/PPL核对，局部通过不作完整结论。诊断计时不是速度证据；此处仍是本机pilot，不是正式2048样本或多卡验收。


### 2026-10-12 02:26：统计寿命前缀通过，完整验收比较器接入修复

**理论支持、参考文献。** 应用OLLA（https://arxiv.org/abs/2210.12924）最后消费者/存储生命周期原则，参考Checkmate（https://arxiv.org/abs/1910.02653）依赖约束；只缩短已死私有统计对象引用寿命，不改Fisher/Adam/量化数学，不使用其ILP或重算求解器。

**具体方法。** 在workspace+消费H复用+down-only稳定写回+末组私有H输入清理组合上，移动原有gptq及统计dict删除到after_layer_quantized hook后、activation replay标记前。两臂CPU观察去重storage；11层44solver/11write/11dead及11stats覆盖，统计持有量两臂均48.0234375MiB/层。

**实际效果。** stats recovery-v2完整前缀valid=True；独立worker、controller+extended-path缓存审查全部通过。507状态、28层非空held-out输出、KL .32509875297546387/PPL20.781702041625977精确；1161区间/4644各自起终点无回升/缺失，46绑定源码和13缓存SHA不变。NVML5033→4309MiB（约14.39%），allocated4117.5854→3397.4951MiB、reserved4276→3552MiB，仍仅0–10量化且Stage2-only前缀。原dead在layer3/10的回放瞬时reserved各+36MiB已在此组合中消除，但没有证明具体36MiB唯一分配根因；全局峰与dead相同。诊断quant秒数1051.024→1114.804，不宣称加速。独立证据stats目录independent_completed_worker_audit.json及independent_final_controller_and_cache_audit.json；statusSHA f0f1981874d3a6fd93445f4bedd133d056f3e60af505d4fc988df511b6233dff。

**评价与反思。** 生命周期组合在真实Stage2分配历史下首次满足新solver候选的全部前缀门槛，但末层27、fresh Stage0及完整两模型重复尚未验收，不能称全程/终稿完成。刚接续的full-v1检测出比较器未登记新工程flag，会把仅存储变化误判为算法参数改变，因此主动在完整配对之前SIGTERM终止并保留源/日志/状态；0完成对，不是方法失败。没有编辑旧绑定源码。新adapter只登记research_replay_early_stats，仍严格比较学习率、样本、block、seed、dataset等；两个CPU检查包括实际生成命令和改算法参数拒绝。tools/run_replay_stats_full_followup_v2.py增加启动前算法命令核对、绑定adapter源码，使用新目录full-v2，完整fresh两模型逐对严格准入已恢复，尚无full收益结论。

### 2026-10-12 04:34：统计寿命组合的 1.7B 首对 fresh 全流程独立验收通过

**理论支持、参考文献。** 延续 OLLA（https://arxiv.org/abs/2210.12924）最后消费者与张量存储生命周期、Checkmate（https://arxiv.org/abs/1910.02653）存储依赖原则。应用的是工程寿命管理，不是论文的 ILP/重算优化器；未改 REAL-Q 的 Fisher、Adam 或量化数学。

**方法具体内容。** 已接受 checkpoint+CE 系统作 control；candidate 在其上加入 Hessian solver workspace、消费 H 复用、down_proj 稳定写回、末组已死私有 H_sub 清理，以及量化统计对象原有删除点前移到 activation replay 前。两臂 fresh Stage-0 + 全 28 层量化 + 评估，W4A16/每行、本机 cal2×1024/eval2×1024 pilot，未启用 finite/all-linear。两臂共享仅 CPU 整数 storage 观察器；112 down 求逆、28 写回、28 私有输入清理及 28 stats 观察完整覆盖，四份因子保留，每层私有输入 144MiB。stats 持有量两臂相同 48.0234375MiB/层，不是峰值下降量。

**实际效果。** tag=qwen17b-replay-stats-full-20261012-v2-p1；独立证据 outputs/replay-stats-full-followup-20261012-v2/independent_completed_pair_audit-20261012-0434.json valid=True。两臂全部 507 状态、28 层非空输出及 KL .625495195388794/PPL25.562191009521484 与 canonical accepted reference 精确一致；55 份绑定源码未变，13 份缓存用 extended Windows path 流式 SHA 逐字节一致，两个 stage 算法命令一致。保留 stage/logkind/rank/phase/occurrence 身份，2961 个 exact 区间、11844 个分别 start/end 指标全部无回升/缺失；全局及评估峰值无回升。NVML 6629→6089MiB，-540MiB/-8.146%；完整 allocated 5605.190918→4883.690918MiB，-721.5MiB/-12.872%；reserved5872→5332MiB，-540MiB/-9.196%。Stage-0 allocated/reserved 3008.533691/3410MiB 两臂相等，评估峰也相等。最大 allocated 仍在 layer_27_mlp_down_proj_after_hessian_inverse 区间，该区间含后续量化/梯度刷新，不能称为纯求逆峰。旧 workspace 的 layer3/10 回放 reserved+36MiB 在本组合完整流程中均未再发生。

**方法评价。** 首个完整新组合达到了所有已测区间/边界不高于 control 的门槛，解决了旧 workspace 数值正确但局部 reserved 回升的验收缺口。额外收益以 accepted checkpoint+CE 为 control，不能与原始 base 的已有下降重复相加。NVML 是采样，不证明连续每一时刻。尚仅 1.7B 一对完整 pilot，第二对已串行启动；三次重复、0.6B 交叉验证、正式 2048 样本和服务器多卡仍未完成。

**反思及后续。** 首对诊断 quantization_seconds=2908.441606→3635.631319；候选更慢，不宣称加速，也不能将密集测量时间当未经插桩的最终速度证据。显存主线已有正向完整证据，仍须重复确认稳定性，再做同一数学/校准配置的低扰动计时验证。尚未证明具体 +36MiB 分配根因；不能把 stats 持有量直接等同 reserved 下降。保留全部失败/中断/工具接入证据，不改活动 55 份绑定源码，不推送。

### 2026-10-12 06:52：1.7B 第二对 fresh 全流程独立验收通过

**理论支持、参考文献。** 沿用 OLLA（https://arxiv.org/abs/2210.12924）最后消费者/存储寿命和 Checkmate（https://arxiv.org/abs/1910.02653）依赖约束；本次为冻结同一工程组合的重复验证，无新数值算法。

**方法具体内容。** p2 与 p1 均使用 accepted checkpoint+CE control，候选增加 workspace/消费H复用/down-only稳定写回/末组私有H释放/回放前统计删除，两臂 fresh Stage0+全28层+评估。保持算法、数据、seed44、本机cal2×1024/eval2×1024 pilot不变；单GPU串行，全部112求逆/28写回/28dead/28stats覆盖逐项核对，未更改活动绑定源码。

**实际效果。** outputs/replay-stats-full-followup-20261012-v2/independent_completed_pair_audit-20261012-0652.json validTrue。canonical507状态hash237fd17bf88c35a522dc91f4a5c3c0bf519fd3d707967ce0a115350232053187、全部28层非空输出、KL .625495195388794/PPL25.562191009521484精确。55份源码SHA无变化/冲突，两stage算法命令一致；13份完整extended-path缓存独立流式SHA逐字节一致。保留stage/logkind/rank/phase/occurrence，2961 exact interval、11844各自start/end无回升/缺失，全局各stage、全流程及评估峰不高于control。NVML6629→6089MiB、allocated5605.190918→4883.690918MiB、reserved5872→5332MiB，与p1相同；Stage0与评估峰相等。C/N statusSHA分别1b870f975992f0ee1ad70873b812ef2632612a8c350882adec9e46427083af27/d66ea16fb84e5a5a095769589ca73c25f1a343d564186b8b5122d07cd7f01f10。

**评价与反思。** 两个fresh完整配对支持已测显存改善可重复，仍不是三次/两模型完成。p2诊断quantization_seconds3607.361559→3534.241313，方向与p1的2908.441606→3635.631319不同，不能据密集插桩时间声称加速或固定开销；重复完成后仍需低扰动计时。stats48.0234375MiB/层是持有量，末层after_inverse区间含后续刷新，NVML采样不是连续每刻证明；唯一+36MiB分配根因尚未确定。p3已自动串行开始，随后0.6B三对，逐对独立所有质量/source/cache/覆盖/每区间/每point/global门槛。保留原证据，不并发GPU、不推送。

### 2026-10-12 09:09：1.7B 三对 fresh 全流程独立重复验收通过

**理论支持、参考文献。** 沿用 OLLA（https://arxiv.org/abs/2210.12924）最后消费者/存储生命周期与 Checkmate（https://arxiv.org/abs/1910.02653）依赖原则；工程组合冻结后进行真实完整重复验证，不改变 Fisher、Adam、量化或校准数学。

**方法具体内容。** accepted checkpoint+CE 为 control，candidate 增加 Hessian solver workspace/消费H复用/down-only稳定写回/末组私有H输入释放/回放前统计删除。三对均 fresh Stage0+28层量化+评估，本机 cal2×1024/eval2×1024 pilot，单GPU串行。每对独立核对 canonical507状态、28层非空输出、KL/PPL，112 down 求逆、28写回、28dead、28stats覆盖，两stage命令、55源码和13完整extended-path缓存；区间身份保留stage/logkind/rank/phase/occurrence，各自start/end分别审查。

**实际效果。** 第三对独立 outputs/replay-stats-full-followup-20261012-v2/independent_completed_pair_audit-20261012-0902.json validTrue（实际生成09:08）；三对汇总 independent_three_pair_summary-20261012-0909.json validTrue。三对均507state hash237fd17bf88c35a522dc91f4a5c3c0bf519fd3d707967ce0a115350232053187、all28输出、KL .625495195388794/PPL25.562191009521484精确；源码/cache/方法覆盖正常。每对2961 exact区间/11844各自起终点无回升缺失，累计8883区间/35532起终点指标。三对NVML峰均6629→6089MiB（-540/-8.146%）、allocated均5605.190918→4883.690918MiB（-721.5/-12.872%）、reserved均5872→5332MiB（-540/-9.196%），Stage0/评估相等。第三对C/N statusSHA 8610e8977db3e7195a9f42b55d67e524f61a0917feb33793adbea9c4036964aa/ebf3b899be97fd4571e211d9762cba19cbb64632bfd8676d8000e84e75b67570。

**评价与反思。** 三对支持同一硬件/数据/校准配置下已测峰值收益可重复，旧workspace layer3/10回放reserved+36在三对完整组合中均被消除；唯一分配根因仍未证明。stats持有48.0234375MiB/层不是峰降幅，after_inverse区间含后续量化/梯度刷新，不作纯求逆归因，NVML采样不是连续每刻证明。第三对诊断quant3540.361688→3667.490090，三对方向和幅度不稳定；不宣称加速，后续低扰动计时仍必要。仅1.7B三对pilot完成，0.6B新组合尚未启动/验收，非正式2048样本/多卡。09:08 controller17/reader19/p3campaign571正在CPU最终缓存核对，无GPU进程是审核阶段；独立13文件流式SHA已通过，原控制器审核日志/status不能补造或覆盖。待原gate通过自动串行0.6B三对，继续逐对独立全部门槛。不编辑活动冻结源，不并发，不推送。


### 回放统计寿命清理：逐点门槛下的条件完整验收

**理论/参考与方法。** 延续OLLA最后消费者/寿命及Checkmate内存依赖原则；前缀阶段峰及独立start/end严格通过后，逐对fresh Stage-0+28层量化+评估，两模型各三对；任何一对失败不继续相同配置重复。

**实际效果。** `/mnt/d/gptq_plus_realq/outputs/replay-stats-full-followup-20261012-v2/status.json`；valid=False，skipped=False，完成配对数=3。

**评价与反思。** 完整结果必须保留所有阶段峰值/区间、全部28层及canonical state/KL/PPL核对，局部通过不作完整结论。诊断计时不是速度证据；此处仍是本机pilot，不是正式2048样本或多卡验收。

### 2026-10-12 09:57：0.6B 首对完整交叉验证失败，保留负向证据并准备大小感知写回

**理论支持、参考文献。** OLLA最后消费者/存储位置和Checkmate依赖原则继续适用，但小权重稳定位置可能保留共享段。新核对PyTorch2.11官方CUDA文档 https://docs.pytorch.org/docs/2.11/notes/cuda.html 及v2.11.0原生分配器源码 https://github.com/pytorch/pytorch/blob/v2.11.0/c10/cuda/CUDACachingAllocator.cpp ：1–10MiB请求在无合适空闲块时会拆分20MiB段，原生large_segment_size_mb默认20；缓存与活跃分割块须区分。此实现说明提供存储位置假设，不证明本次96MiB唯一根因。1.7B后半部也有共享128MiB段，不能把所有共享段简单等同失败。

**方法具体内容。** 0.6B新组合首对fresh Stage0+28量化+评估结束；不放宽原门槛，独立核对canonical507/all28非空/KLPPL、112inverse28write28dead28stats、55source/13extended-path完整缓存，两stage算法命令、2961exact区间/11844每point/global。见 outputs/replay-stats-full-followup-20261012-v2/independent_qwen06_completed_failure_audit-20261012-0951.json。

**实际效果。** 质量与全部来源/覆盖通过：state hash7ad9b1a7a890faf5175b136b0c12ea30228bea96060a84f9ee734cd2f0c57ff3，KL .30534985661506653/PPL25.013607025146484精确。NVML4247→4119MiB，full allocated3304098304→3156305408bytes，reserved3490→3362MiB，Stage0/评估相等。仍validFalse：layer25_activation_replay_and_cleanup区间reserved1088421888→1189085184bytes（1038→1134MiB，+96MiB）；两个对应point分别是该回放start_reserved及最后after_inverse区间end_reserved，同样+96MiB，其余point无回升/缺失。累计2961区间仅此一处回升、11844point仅2回升，不能由全局下降接受。C/N诊断quant852.381918→898.524426，不称加速。statusSHA C0624a141c2877fe46319b2311584c2126e58832b0d63707eb3126035ba40f0fa/Ncdfc856c789fd7f97357542fb349376787459f9bb84a48398e2b800f354a0823。

**评价与反思。** 1.7B三对已接受结论保留，不能称新组合两模型通用接受。原full-v2控制器09:46:48正确拒绝该0.6B配置，p2/p3未启动；旧status/log/source保留，所有实验进程结束，实际WSL已重新启动，不沿用17/770/885等旧PID。0.6B每层down权重6MiB，层25既有weight storage位于128MiB段，before inactive90MiB，构成进一步检查生命周期/位置的依据；不是分配根因确证。

**后续具体方案及CPU证据。** 新 tools/hessian_weight_size_policy_prototype.py 保留workspace/消费H复用/dead私有输入/early-stats，按权重目的storage大小>=20MiB使用稳定写回，否则使用原FP32→原dtype转换及分配。6MiB走原写回，24MiB保持已接受1.7B写回；阈值取原生段默认尺寸，不作超参数扫优。新共享CPU观察器显式区别requested_enabled和实际selected enabled，不保留CUDA张量；不能以仍申请stable flag假称28次稳定写回。CPU5项和真实fresh原始入口编译通过，source与dead释放、复用数学保留，缺失builder依赖拒绝，见 outputs/weight-size-policy-cpu-20261012-v1/status.json，helperSHA90a0d3a390ceffe69b2bcfcb62843b2da2988359ec2c1b1afe79ced09ff53daf。尚无新GPU收益，下一步需新绑定入口/覆盖审计接入，一对0.6B严格screen完成后才准入必要fresh重复；不继续相同失败配置，不用前缀0–10漏掉layer25。不编辑旧冻结源/原始结果、不并发GPU、不推送。此前11–12点结束估计不再作为保证，新工程验收时间需重新估计。

### 2026-10-12 10:03：大小感知写回接入真实0.6B完整screen

**理论、参考文献。** 继续09:57 OLLA/Checkmate位置寿命与PyTorch2.11原生分配器依据。官方CUDACachingAllocator.cpp已归档docs/literature/PyTorch_v2.11.0_CUDACachingAllocator.cpp，SHA1903454eff07a535ce580c075afb26e0e12ae2fe00d0a3050671c376b4a3c8ea；参考URL见新增bib条目。shared segment/live pin只是方法假设，不把层25+96归因已证实。

**具体方法。** 新tools/run_weight_size_policy_screen.py绑定旧来源+newhelper/rawentry，在olddead+earlystats真实原入口上接入size-policy installer，保留实时forward globals/后续wrapper。仅一对0.6B fresh Stage0+28量化+评估，不以0–10漏掉layer25，不自动重复或继续旧失败full-v2。两臂共享仅CPU元数据观察，requested stable与实际selected stable区分；6MiB weight选择原转换分配，仍保留112inverse/reuse/28dead/28stats。各层coverage审查必须检查requested、selected、大小/dtype、factor alias，全部canonical/source/cache/区间/每point/global原门槛保留。

**实际效果。** CPU两项新screen接入检查通过，包含真实fresh入口/单wrapper及错selected、错requested、缺失调用拒绝，outputs/weight-size-policy-screen-cpu-20261012-v1.log；不作为显存收益。真实新工具会话84210启动，out outputs/weight-size-policy-screen-20261012-v1，tag qwen06b-weight-size-policy-screen-20261012-v1，日志logs/weight-size-policy-screen-20261012-v1.log。10:02 actual WSL新controller17/唯一reader19/campaign20/profile21/唯一GPU31 controlStage0。56绑定source独立SHA未变，independent_launch_source_audit.json，只launch provenance，不是实验通过。原full-v2与全部旧结果仍保留。

**评价与反思。** 新配置改变小权重目的存储位置而不是算法/校准，GPU收益未取得；必须等完整两臂与独立最终审核再准入后续fresh重复。用户询问reserved是否主要瓶颈：reserved不是阶段，1.7B已证实大张量求逆/因子/后续刷新显存压力，0.6B reserved+96是当前通用验收卡点，不能泛化为整个REAL-Q唯一主要显存或速度瓶颈。新方法若失败继续依实际记录诊断，不重跑同失败配置。速度低扰动/正式2048/多卡均未验收，不并发GPU、不推送、不改冻结源。


### 2026-10-12 10:45：0.6B大小感知写回完整screen失败，独立证据保留

**理论支持、参考文献。** OLLA https://arxiv.org/abs/2210.12924 的寿命/位置分离、GMLake https://arxiv.org/abs/2401.08156 的碎片分析，以及PyTorch2.11 CUDA memory management/MemPool文档 https://docs.pytorch.org/docs/2.11/notes/cuda.html 。仅小down权重转换分配不足以稳定所有reserved区间；单一分配根因仍未证明。

**具体方法。** 两臂fresh Stage0+全28层量化+评估；候选保留workspace/消费H复用/dead私有输入/early-stats，6MiB down目的权重选择原转换分配（requestedTrue/selectedFalse），20MiB阈值未扫描。两臂共享CPU元数据观察，没有将求逆或因子转移到CPU，没有改变REAL-Q数学/校准。

**实际效果。** 控制器10:38:24完成validFalse，独立outputs/weight-size-policy-screen-20261012-v1/independent_completed_pair_audit-20261012-1042.json实际10:40生成validFalse。507state/all28非空输出/KL .30534985661506653/PPL25.013607025146484精确；58绑定源码无变化/冲突，13完整extended-path缓存流式SHA一致，两stage命令和112inverse/reuse28requested-selected-write28dead28stats覆盖通过。2961 exact区间无缺失，layer10/11/17回放reserved峰各+32MiB、layer20+64MiB；11844独立point无缺失，layer10/17回放start及其末组after_inverse区间end共4处+96MiB。原layer25回升未出现，但严格门槛仍失败。global NVML4247→4119MiB、allocated3304098304→3156305408B、reserved3490→3362MiB；Stage0/评估相等。诊断quant823.401686→1014.169016，不作速度结论。

**评价及反思。** 大小policy将回升位置转移，不能据layer25消失称修复或推广三对；不重跑同已完成失败配置。allocated减少不保证reserved逐区间减少，当前证据支持分配位置/历史问题，但尚未定位唯一活跃张量。1.7B此前三对完整正向保留，当前新两模型通用结论仍未成立。本机cal2×1024 pilot非正式2048/多卡，NVML为采样。所有旧源码/结果/失败日志保留，不推送。

**下一工程方向（尚未实现/无收益）。** 检查整层回放权重是否仍占据求逆/刷新短命量的共享段，考虑在量化与统计最后消费者完成后，将当前层FP16权重聚合到单一专用native MemPool存储，CPU释放当前层之后验证池无活跃引用再销毁。不同于此前失败的求逆临时池；集中的是回放仍活跃的权重而非求逆临时量。需要先验证真实来源入口、parameter身份/值/shape/stride、alias与pool生命周期，再完整长轨迹严格screen；不得把新假设、CPU准备或局部拷贝测试当GPU收益，不加入反复empty_cache或放宽门槛。当前全部GPU/控制器已经结束，以actualps为准。


### 2026-10-12 11:05：整层回放权重紧凑池组件通过，进入新0.6B完整screen

**理论支持、参考文献。** OLLA最后消费者与存储位置、GMLake不规则分配/碎片、PyTorch2.11 native MemPool文档 https://docs.pytorch.org/docs/2.11/notes/cuda.html ：池内存须等全部活跃tensor引用解除才可销毁。不同于此前已失败的求逆临时池，当前隔离的是量化结束到回放/CPU释放之间仍活跃的整层参数；没有确认旧reserved回升唯一根因。

**具体方法。** 新tools/replay_weight_compact_pool_prototype.py在原after_layer_quantized hook及统计删除后、回放mark前，把当前层11份FP16 Parameter存储复制到一份专用native池arena，保持Parameter对象身份、值、shape/contiguous stride，拒绝混合dtype/设备/非连续及共享原storage。CPU layer_manager释放后同步核对pool active bytes=0，立即删除ticket/pool，CPU记录不保留tensor。保留workspace/消费H复用/原dead/earlystats/小weight size-policy。新tools/replay_weight_cleanup_schedule_prototype.py保留最后down free()原H/Losses释放点，仅将其原有empty_cache+cleanup_memory尾部移动到紧凑权重放置之后；不添加调用次数、不每个refresh清空、不保留原GPTQ对象。这是原清理时序与存储位置实验，不称新量化算法。AST对原free确切四个语句绑定，错pending层拒绝。

**实际效果与准备证据。** 新CPU5项通过：真实gptq_fwrd最后消费者/CPU释放边界、rawentry/fresh Stage0/liveglobals单wrapper、原free清理调用次数与顺序、全层pool覆盖及遗漏/active引用/共享storage/额外cleanup拒绝。新真实CUDA组件outputs/replay-weight-compact-cuda-20261012-v1/status.json validTrue（10:55:49）：原FP16 Qwen3-0.6B layer25和1.7B layer27参数与固定32-token输出逐位相同，Parameter身份保留，CPU释放值相同，pool active0；不是量化/长分配历史或收益证据。0.6B payload31461888B、private segment33554432B；1.7B payload100672000B、segment102760448B，这是存储大小非显存峰降幅。helperSHA1fb8e2c6bef9a29adfc8f29d177d9accd326faf73c99d9e8d3f65535a329c484，cleanupSHAeb13ed85da9a4aeacec4d9ec52e2027c10dbfa17b74507ea32a47657cddde026。

**真实接入。** 新tools/run_replay_weight_compact_screen.py CPU2实际rawentry及missing/错误active/额外cleanup gate通过，真实工具会话72728已启动out outputs/replay-weight-compact-screen-20261012-v1，tag qwen06b-replay-weight-compact-screen-20261012-v1，日志logs/replay-weight-compact-screen-20261012-v1.log。11:04 actual新controller17/reader19/campaign20/profile21/唯一GPU31 control freshStage0；17不是已完成size-policy旧controller。60绑定源码独立SHA全部不变，independent_launch_source_audit.json仅来源启动，不是GPU收益。

**评价/反思与后续。** 一对fresh Stage0+28量化+评估覆盖全部旧回升层，不自动重复。必须canonical507/all28非空/KLPPL、source/13完整cache/112inverse28requested-selectedwrite28dead28stats、28紧凑参数规格与pool active0/原cleanup恰1次、每区间/每point/global全部过原门槛。候选原小down selectedFalse，不能旧28stablepointer gate误判；控制无pool/无清理重排。组件不证明真实数值/回放源或显存改善，也不宣称速度；实际CPU释放pool失败先查引用bug，不能改方法掩盖。原size-policy及更早所有失败证据保留，不重跑同完成失败配置，不编辑新活动绑定源、不并发、不推送。正式2048样本/多卡/低扰动速度及报告仍待。


### 2026-10-12 11:15：用户明确硬件容量目标，增加全局峰值主验收

**用户口径与理论。** 用户指出设备容量要求取决于完整流程全局峰值，局部reserved回升若不抬高global allocated/reserved峰值，不应单独否决容量优化。按此澄清增加硬件容量主验收：同一完整工作量及canonical507/all28非空/KLPPL、来源/cache/方法覆盖保持，candidate完整process allocated/reserved与整卡NVML采样峰均不高于control。逐区间和各自start/end继续审查并报告，作为分配/碎片诊断，不再单独决定硬件容量合格。不能把局部峰与自身全局峰的必然大小关系当新的独立证据。NVML仍是采样；实际更小卡能运行须另验证。本机pilot不泛化正式2048/多卡。

**具体审查及效果。** 新outputs/hardware-global-peak-reclassification-20261012-v1.json为已完成独立原审查的补充重新分类，逐一核对原独立证据SHA及actual完成statusSHA，没有重跑/修改原结果。1.7B原三对与两个不同0.6B单对均capacity_criterion_passedTrue。1.7B三对保留原strictTrue；0.6B原stable+stats和size-policy各一对保留原strictFalse/local回升，同时global NVML4247→4119、allocated3304098304→3156305408B、reserved3490→3362MiB，可列为全局峰值合格的单次pilot证据。两种0.6B方法不能混算重复；重复与低扰动速度尚待。

**评价、反思与后续。** 原strict失败是旧更强条件的真实负向，不能覆盖改为旧validTrue；新的容量条件需明确标注，而不是隐藏失败。活动compact screen仍冻结60source/旧比较器，完整两臂结束后另做独立硬件global验收，即使旧strictFalse也不能据其自动扩大/停止；必须质量/source/cache/method及完整global证据。若只有局部reserved变化而硬件global无新增改善，可停止追求局部零回升，优先完成最简可接受组合的0.6B同配置必要重复、速度成本验证与报告。原0.6B组合此前只有旧local条件失败，如按新的容量主验收进入必要同配置重复，须使用新目录/记录新准入理由，不能把它记成旧失败寻优重跑。继续单GPU、不推送、不修改活动冻结检查器。


### 2026-10-12 11:38：全局容量验收独立检查器与0.6B必要重复续接准备

**理论与口径。** 延续用户11:15明确的设备全局峰值目标及OLLA存储寿命原则。每个区间/start/end仍完整审查记录，局部回升作为诊断；不能覆盖旧strict结果，不能把两个不同方法的0.6B单对合并为重复。

**具体方法。** 新tools/audit_completed_memory_pair_capacity.py独立读取actual完整status/manifest、canonical507/all28非空/KLPPL、当前sourceSHA/13完整extended-path缓存流式SHA、真实112inverse/reuse28writes28dead28stats，以及2961exact区间/11844各point和基础设施身份。容量valid仅取上述完整性/数值/来源/方法门槛及full allocated/reserved/NVML各<=control；strict_interval_point_valid与全部局部回升分别保留。任何global回升、质量/source/cache/方法或遥测缺失仍拒绝，不受旧controller validFalse影响。CPU2项验证全局三个指标各回升及质量/来源/方法/遥测缺失均拒绝。

**已完成的独立效果。** 原0.6B stable+stats首对已用新native审查器实际重核所有55sources、13cache、507state/28输出/KLPPL/覆盖/区间point，outputs/replay-stats-full-followup-20261012-v2/independent_qwen06_hardware_capacity_audit-20261012-1135.json validTrue/strictFalse；NVML4247→4119、allocated3304098304→3156305408B、reserved3490→3362MiB。不是新GPU实验、不是三次重复或速度结果。

**后续续接。** 新tools/run_qwen06_hardware_capacity_followup.py原stable+stats同方法fresh repeat2/3（不含size-policy/compact），只在当前compact完整独立quality/source/cache/method/telemetry检查完成且GPU空闲后执行；legacy campaign对局部回升返回1保留，下一次准入须新的独立容量global检查通过，不用legacy返回码掩盖基础设施/质量错误。新目录out outputs/qwen06-hardware-capacity-followup-20261012-v1，日志logs/qwen06-hardware-capacity-followup-20261012-v1.log，工具会话54194/CPU等待器224（11:37实际）；等待outputs/replay-weight-compact-screen-20261012-v1/independent_completed_capacity_audit.json，未启动新GPU。当前唯一候选GPU165/compactcontroller17，active layer22附近，actualps持续更新。等待器自身及新auditor SHA已冻结，不改。已有首对准入依据是新用户全局容量条件，不记作重复旧失败寻优，原source/status/失败完整保留。

**评价与反思。** 实验收益与审查器/等待器接入区别；当前compact尚未结束，不因其启动或组件通过宣布成功。后续简方案两模型各三次容量完整证明、低扰动时间代价和报告仍待，不为局部reserved零回升无限增加方法。单GPU、不推送、不改活动冻结源。


### 2026-10-12 11:55：紧凑回放整层方案完整数值/显存通过，BF16规格独立补审

**理论与具体方法。** 延续OLLA存储位置/寿命、GMLake分配碎片与PyTorch2.11 native MemPool原则。此前文档把完整权重写作FP16须纠正：组件预检显式FP16，真实原模型config与两臂完整流程均为BF16，矩阵求逆仍FP32；紧凑arena按原Parameter dtype复制，无dtype转换、没有降低精度。整层11份参数的当前GPU存储集中到单native池，原最后down free清理尾部移动至集中之后，H/Losses原释放点及清理调用次数保持；CPU层释放后28次pool active0。

**验收工具缺陷与独立处理。** 原runner11:45:14 status.validFalse仅因compact_pool_coverage把真实BF16规格与预检FP16规格逐字比较，数值/来源/区间门槛已通过。不是量化或内存方法失败，不能改原冻结源/status。新tools/audit_replay_compact_bf16_completed.py先完整独立重核所有实际数值/来源/cache/方法/遥测，再按源模型config的BF16与hidden/head/intermediate架构构建11份name/shape/stride/bytes/dtype，逐层核对两臂真实参数及28次紧凑/active0/sharedstorage1/原cleanup1/pending空。component原FP16证据保留，不修改成虚构BF16预检；完整BF16 canonical精确结果才是实际dtype路径证明。

**实际效果。** outputs/replay-weight-compact-screen-20261012-v1/independent_completed_capacity_audit.json validTrue且strict_interval_point_validTrue。canonical全部507state hash7ad9b1a7a890faf5175b136b0c12ea30228bea96060a84f9ee734cd2f0c57ff3/all28非空输出/KL .30534985661506653/PPL25.013607025146484精确；62绑定源码不变/无冲突、13完整extended-path缓存流式SHA一致，基础设施/两stage算法命令/112inverse-reuse28requestedselectedwrite28dead28stats28compact28cleanup均通过。2961exact interval/11844独立start-end无回升/缺失。global NVML4247→4119MiB、allocated3304098304→3156305408B、reserved3659530240→3558866944B（3490→3394MiB）；Stage0/评估相同。C/N statusSHA676b79f1d4c56c40ff1b2842f71ad8fb157981e880a78ffa811a3b1934dbcc75/3cd37942a14b5c201af19fd595863da246a1b271b2d68ae684ccc26639b9f4bc。诊断quant831.726950→1298.993827，非低扰动速度证明。

**评价与反思。** 一对真实完整轨迹支持局部reserved回升被消除，但不能称单一旧+96根因已确认；位置集中与原清理时序同时改变。新用户硬件目标下，它相对原简stable+stats方案没有额外allocated/NVML峰下降，global reserved反而3362→3394多32MiB，且增加arena搬运/池/同步复杂度，故不推广额外完整重复；保留其系统机制结果和预检dtype不匹配工具反思。仅0.6单对pilot，1.7只有FP16组件，不混称两模型pool完整通过。正式2048/多卡/更小卡/低扰动时间仍待。

**后续实际状态。** 完整BF16独立审查文件已触发CPU followup224；正在原简首对重新核对cache/source的CPU准入阶段，无GPU不表示研究停止。新56source native独立SHA不变/原rawentry逐字相同，outputs/qwen06-hardware-capacity-followup-20261012-v1/independent_admission_source_check-20261012-1154.json，仍非新GPU成功。接续原stable+stats fresh repeat2/3的容量验证；不含size-policy/compact、不改旧原strict比较器，local回升单列诊断。旧compact17/GPU165/reader19均已结束，actualps更新；会话54194续接。所有失败与旧validFalse保留、不推送不并发、不改冻结waiter/auditor。


11:56 actual补充：followup224已完成原首对CPU独立readmission validTrue/strictFalse，原source/cache/覆盖不变；新p2已11:53附近串行启动campaign290/profile291/唯一GPU301 control freshStage0，不再是仅CPU等待。唯一总reader289，actualps更新；原compact17/165/reader19结束。新56source原rawentry exact已独立核对，p2未完成，不冒称三对。


### 2026-10-12 12:14：原简方案低扰动计时入口CPU准备（未启动GPU）

**理论与具体方法。** Checkmate/Korthikanti重算/存储优化需同时计时间成本。新tools/prepare_replay_stats_low_overhead_entry.py将已接受原stable+stats rawentry的workspace/reuse/dead/stable源码前缀与earlystats installer原hook绑定到既有低扰动timing入口；不修改数学、dtype、校准、seed或Adam。双方关闭可选memory/quant/event密集profiling，不跑NVML采样与区间reset；保留共享CPU方法/weightstorage/stats覆盖观察，因此不是零开销纯计算。量化与Stage0 collector分别CUDA同步wall timer，全部507state哈希和28层heldout fingerprint在主量化timer之外，KL/PPL/cache仍必验；保留fresh Stage0、stage2禁止新预计算。此collector+quant时间不含模型准备/离线评估/hash，不能称完整端到端延迟。

**实际准备证据。** CPU2项真实源/单timedwrapper/原hook及真实manifest仅删除profiling控制检查通过；实际生成outputs/replay-stats-low-overhead-entry-cpu-20261012-v1/entry.py和status.json validTrue，只CPU准备。未启动timingGPU、尚无正确性或速度结果，不使用密集profile秒数宣称加速或确定代价。新方法移植后必须真实fresh统计13cache/source/507state/all28输出/KLPPL及112inverse28writes28dead28stats再次核对；若任何数值/来源不一致先查接入问题。

**评价与后续。** 当前q06容量p2控制组GPU343第15层继续，原56绑定源码/auditor/controller不编辑；p3仍未启动/通过。计时完整runner与正式准入尚待两模型容量重复完毕，不并发GPU、不因CPU准备停止既有实验或冒称最终成果。低扰动少量/多次配对须如实标明样本量、顺序/方差与scope，不剔除慢结果，最终报告显存与时间权衡。


### 2026-10-12 12:44：两模型低扰动成本runner CPU准备，未启动GPU

**理论与方法。** 延续Checkmate/Korthikanti显存与时间权衡。新tools/run_replay_stats_low_overhead_cost.py只在原简组合0.6B容量三对完成后，对1.7B三对及0.6B三对重新独立完整capacity readmission；不是据旧valid扩大实验。复用既有串行timer，AST只替换唯一入口复制位置、绑定已准备原数学/earlystats入口，原timer源码不改。计划每模型三对交替C/N顺序、不删除慢值；同校准/seed/Adam/原BF16权重和FP32 solver。fresh Stage0缓存逐字核验、507state/KLPPL基础timer核验，再all28非空输出及完整112inverse/reuse28write28dead28stats元数据与对应参考核对，actual workerentry与全部source SHA一致。仅地址/segment/inactive等分配观察字段不作方法身份；pointer_preserved、参数bytes/dtype/stride和统计storagebytes仍必须一致。collector加quant不含模型准备、评估及hash，不称端到端或零观察开销。

**效果与CPU准备证据。** 新runner CPU2加入口CPU2，共4项通过，实际完整参考元数据用于测试；缺失group、pointer_preserved错误、factor/统计bytes变化、末层空输出、算法命令变化、source变化及缺失三对准入均拒绝。outputs/replay-stats-low-overhead-cost-cpu-20261012-v1/status.json仅CPU准备，无GPU计时、收益或正式正确性结论。没有改活动56绑定源、没有并发GPU。

**评价/反思及后续。** 当前q06容量p2候选GPU467正常，12:39层14附近，控制组12:23:57完整；独立control审查independent_completed_control_audit-20261012-1230.json通过507/all28/KLPPL/57source/13cache及2961interval/11844point自身遍历完整性，仅control不冒称配对通过。后续两臂对应区间/point及fullglobal完整验收后才p3，再低扰动成本与必要报告。D盘此时约11GiB可用，后续计时需先实际核算存储，不能以磁盘不足造成基础设施失败或删除旧证据；未执行任何dedup/删除。正式2048/多卡未验、不推送。


2026-10-12 12:51 计时存储只读核算：ordinary Windows路径再次漏MAX_PATH Fisher，12文件初稿证据保留；extended-path新独立plan完整13文件/model。未来6对若重复保存所有prepared约34GB，当前11GiB不够；1.7B旧六臂prepared已同inode，不能假算可再释放20GB。既有已完成SHA只读hardlink流程可保留全部路径/字节/Fisher独立，后续新timing需CPU准备/验证兼容其完整status准入的事后prepared链接，进入下一arm前腾重复磁盘占用，属于磁盘housekeeping非显存贡献；此刻未链接/删除或启动timingGPU。原不完整plan和extended完整plan均在outputs/replay-stats-low-overhead-cost-cpu-20261012-v1，当前p2 GPU健康，继续完整验收。


### 2026-10-12 13:01：0.6B第二对完整容量重复通过，第三对串行启动

**理论与口径。** 原stable+stats组合依照用户11:15全局硬件容量目标重复，OLLA存储寿命原则不改变量化数学；局部区间及起终点仍独立检查并保留诊断，旧strictFalse不覆盖。

**具体验收及效果。** native独立outputs/qwen06-hardware-capacity-followup-20261012-v1/independent_completed_pair_audit-20261012-1300.json实际13:00:10 capacityTrue/strictFalse；全部507state/all28非空fingerprint/KL .30534985661506653/PPL25.013607025146484精确。58当前source SHA不变/无冲突，13完整extended-path缓存流式SHA一致，两stage算法与环境/GPU身份、112inverse/reuse28stablewrite28dead28stats覆盖通过。2961exact区间/11844各point无缺失；仅layer25 replay reserved1038→1134MiB(+96)，回放start及末组after_inverse区间end两point同+96，与首对一致。全球NVML4247→4119MiB、allocated3304098304→3156305408B、reserved3490→3362MiB，所有全局条件通过；after_inverse区间含后续量化/刷新，不纯归求逆。C/NstatusSHA6642a6ea6c9cc9a9fba8748d21aaa9494deb85ec6e3416440e729990cf7ae7eb/f9f227e69213614fb12711396d26e3e49f3e842bdcdb64b0fab6f566a69f6d32。诊断quant1594.308061→1639.228607秒，不作低扰动速度结论。

**评价/反思与续接。** 0.6B现仅原简组合两对容量重复，不是三对最终完成；不同size/compact方法不混算。第三对12:58后串行开始，12:59 actualcontroller224/reader289/p3campaign532/profile533/唯一GPU543 controlfreshStage0；p2campaign290/GPU467结束，后续actualps更新。继续p3全部完整独立验收，再六对容量汇总、低扰动成本及报告。新成本runner仍仅CPU准备，磁盘35.4GB全复制需求与当前11GiB不匹配，须先CPU核验事后prepared只读硬链接housekeeping，全部原路径/字节/独立统计保持；不运行未经验收新timer。NVML为采样，pilot非正式2048/物理小GPU/多卡；不并发不改活动冻结源不推送。


### 2026-10-12 13:12：计时工件保真磁盘管理CPU接入

**方法与边界。** 新tools/timing_completed_prepared_disk.py延续既有已完成prepared权重只读hardlink流程；新计时arm须完整507/KLPPL及独立all28/method/cache验收，参考须full完成，才两份prepared逐字流式SHA/size/稳定mtime比对、同volume原子链接、再实际SHA。全部路径和字节保留，Fisher/统计不共享；这是GPU结束、所有timer之外的磁盘管理，不算显存贡献。新costrunner每arm启动前核对freshcache大小加1GiB余量，每完成arm验收后才事后housekeeping，之后下一arm；能在现存约8–10GiB空间留所有12份独立统计，不需要删除旧证据。此前35.4GB总复制投影不能作为实际磁盘收益或已执行结果。

**实际CPU证据。** 磁盘helper2+runner2+entry2共6项CPU通过；真实scratch文件路径/字节/同inode/已链接幂等、错误内容/缺失completeaudit/越界拒绝。未对真实已完成模型工件执行新链接、未启动timingGPU；WSL真实跨平台路径仍需首次arm完整实际验收，不把native scratch当真实量化证据。outputs/replay-stats-low-overhead-cost-cpu-20261012-v2/status.json仅CPU准备，原v1 status保留且旧runner按其SHA逐字归档original_cpu_runner_source_snapshot.py，未覆盖先前CPU证据。56活动capacity源码独立SHA全部不变。

**评价与后续。** 当前q06第三对actual13:09唯一控制GPU584，层17附近，controller224/reader289/p3campaign532正常，旧543结束。继续完整对应质量/source/cache/方法/区间point/global三指标验收；容量三对结束才六对重新admission及单GPU串行timing，逐对记录所有慢值。计时scope非端到端、共享CPU观察非零开销，正式2048/多卡未验，不推送。


### 2026-10-12 13:30：低扰动成本CPU等待器已接续，第三对原control完整核对

**方法/准入。** NEW tools/run_replay_stats_cost_after_capacity.py只CPU等待容量followup complete status及新native独立第三对最终文件independent_completed_capacity_repeat3_native.json（尚未生成，必须真实双臂完整审查才可写）。要求原简三对status及新容量audit所有质量/source/cache/method/telemetry检查真，之后才启动已CPU准备的成本controller，其还会重新完整六对readmission与GPU空闲检查。绑定waiter/cost/generator/磁盘helper与原56source，禁止修改这组冻结源；不会据启动审查或旧strictFalse自动扩大。

**实际准备。** 新waiter1+cost2+disk2+entry2共7项WSL CPU通过，未跑额外GPU。工具会话7219，actual13:23 CPUwaiter731，out outputs/replay-stats-cost-after-capacity-20261012-v1，logs/replay-stats-cost-after-capacity-20261012-v1.log，候选GPU665仍唯一，layer9附近；后续actualps更新。独立waiter 62source当前SHA不变，只来源不是速度结果。cost out未来outputs/replay-stats-low-overhead-cost-20261012-v1；每模型3交替timing对，完整每臂507/all28/KLPPL/13freshcache/方法及actualentrysource后才事后prepared byteSHA硬链接保磁盘，Fisher独立不删路径，所有时间scope明确。

**当前证据与反思。** p3控制完整独立outputs/full_pipeline_profile_qwen06b-replay-stats-hardware-20261012-v1-p3-r1-control/independent_completed_control_audit-20261012-1320.json通过507/all28/KLPPL/57source/13完整cache/112controlsolver28write28stats及2961interval11844point自身遍历，无pair正向冒称。控制NVML4247/alloc3304098304/res3659530240；诊断quant886.821615不是速度证据。候选与最终容量门槛仍待；后续必须先生成真实native第三对final audit再由CPU等待器串行接续。正式2048/多卡未验，未推送/未并发/未改活动源。


### 2026-10-12 14:11：主机重启中断归因与第三对恢复（覆盖旧PID/等待状态）

**实证归因。** Windows LastBootUpTime=2026-10-12T13:24:08.5000000+08:00，System/User32事件1074记录13:23:40 rundll32.exe、13:23:49 winlogon.exe发起restart；已保存independent_host_reboot_attribution-20261012-1333.json。这次与00:47主机未重启的未知中断不同，不能沿用旧归因。原candidate最后log13:23:45、10/28完成，最终status/state_metrics/全部28输出/KLPPL/阶段2方法coverage缺失。原controller224/reader289/GPU665及costwaiter731全部消失，原根status均缺失，不补造。独立independent_interrupted_third_pair_audit-20261012-1332.json validFalse保存所有原files SHA/mtime；完整13freshcache仍与control一致，64绑定source SHA不变/无冲突，但不是数值/全局验收。重启底层动机不据此称实验OOM/方法失败。

**恢复方法及实际启动。** NEWtools/run_qwen06_capacity_interrupted_recovery.py先独立重新实际审查已完成p1/p2的全部质量/cache/source/覆盖/2961interval11844point/global，未重跑前两对，然后只新目录fresh恢复未完成p3，原stable+stats rawentry逐字相同，算法/data两stage一致。工具会话24530，out outputs/qwen06-hardware-capacity-recovery-20261012-v2，日志同名；tag qwen06b-replay-stats-hardware-20261012-recovery-v2-p3。14:10 actualNEWcontroller17/reader19/campaign20/profile90/唯一candidateGPU129量化27/28附近（17不是旧full或旧screen）；独立58source/rawentry exact launch核查通过，source冻结不编辑。这是中断恢复，不算寻优重复已失败配置或方法成功重复。原已完成1.7三对/0.6两对正向保留。

**评价/续接。** 必须new两臂完整507/all28非空/KLPPL/真实112inverse/reuse28stablewrites28dead28stats/source/13cache/2961exact区间11844独立point及fullglobal容量门槛再接续。旧costwaiter session7219也消失，不再称运行；其期待旧p3tag/旧root不能直接重启。需要新versioned成本准入接新recovery root/实际第三对tag，保留旧cost/waiter冻结source/CPU证据且不编辑，不能伪造旧followup status以解锁。新timing还须六对独立readmission、单GPU idle、每arm完整质量/source/cache/method及事后保真磁盘管理；低扰动成本/报告未完成，正式2048/多卡仍未验、不推送。


### 2026-10-12 14:18：原简组合两模型各三对容量完整验收，新低扰动成本v2实际启动

**理论/准入。** 用户全局硬件容量目标下，原workspace/消费H复用/down-only稳定写回/私有H死亡点/早stats释放组合，两个模型同一方法各三对；中断旧p3排除，不改变校准/seed/Adam/数学/dtype，不混size/compact。采样NVML不能证明每刻，局部reserved仍完整保留诊断。

**完整实际效果。** 新恢复p3两臂已14:11:36完整结束，native独立newrecoveryroot/independent_completed_capacity_repeat3_native.json实际14:12:30 capacityTrue/strictFalse；507state/all28非空/KL .30534985661506653/PPL25.013607025146484精确、60sources/13完整extended-pathcache/命令环境GPUidentity/112inverse-reuse28stablewrites28dead28stats均过。2961exact区间/11844各point无缺失，layer25 interval1及对应start/finalafterinverse end2points reserved+96仍相同；global NVML4247→4119/alloc3304098304→3156305408B/res3490→3362MiB，重复前两对。C/NstatusSHAe4c83da630fc3019d0a6071c5a99016d4029f9835ed30282b4b988478c360743/4849e259fc4110c62b58932d9a52a144783d85667e2ec92085b6b58bf65a7a5d。诊断quant997.755932→870.810735秒，不作为低扰动速度结果。

**六对独立汇总。** outputs/two-model-hardware-capacity-accepted-20261012-v1/summary.json实际14:15:32 validTrue；六对重新native完整读取原status/source/cache/方法/所有2961interval11844points，累计17766区间/71064point values。各模型跨3次cache和globalpeaks均相同；1.7B三对NVML6629→6089MiB(-8.146%)、alloc5605.190918→4883.690918(-12.872%)、res5872→5332(-9.196%)，局部无回升；0.6B三对NVML4247→4119(-3.014%)、alloc3151.033691→3010.087402(-4.473%)、res3490→3362(-3.668%)，累计local3interval/6point+96诊断，oldstrictFalse全部保留不重写。都是cal2×1024pilot，非正式2048/物理小GPU/多卡，尚无速度结论。

**磁盘管理。** 单GPU已全部退出后，既有completed SHA只读hardlink工具对p2 C/N及恢复p3 C/N prepared实际逐字匹配f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b（每份1503300328B），保留全部路径/字节/Fisher独立，只改变重复磁盘存储，报告completed-prepared-disk-*.json在上述summaryroot。不是显存优化，原中断candidate未动；完成后D盘~10.15GB可用，成本每arm仍检查cache+1GiB余量。

**成本接入与评价。** NEW tools/run_replay_stats_low_overhead_cost_v2.py仅改admit读取恢复p3实际tag/root和p1/p2原bindings，其他timer/math/method/disk函数AST逐字结构相同，旧cost/waiter源/CPU记录保留不编辑。WSL7CPU通过，实际工具会话8929启动out outputs/replay-stats-low-overhead-cost-20261012-v2/log同名；初始CPU六对重新readmission，不是timingGPU成果。每模型3交替顺序完整timing对，所有13freshcache/507state/all28/KLPPL/方法/source/actualentry完整过后才事后prepared保真链接。collector及fullquant CUDAwalltimer共享CPU观察，prep/eval/hash不含，非零开销/非端到端；继续如实成本和必要最终报告，不把容量完成算速度完成、不剔除慢值、不并发不推送。


2026-10-12 14:25 actual成本v2已完成六对实际WSL完整readmission并启动首q06 control，freshStage0 13cache SHA匹配实际通过；controller17/CPUreader70/唯一GPU79 stage2第3层（14:23），17不是已结束recovery17。新60source/currentworkerentry与rootentry逐字相同、无变化，independent_launch_source_and_capacity_binding.json仅来源/准入/Stage0 cache，不是计时完整质量或速度成果。new entrySHA8559c96317b7eb58ddc58826bebe5469c8978a546a8edcaa5f7b02a19697755f以实际WSL冻结字节为准，跨平台CPU工件换行差异不当数学变化。总logs/realq-system-restart-20261010-1408.log已由NEW stream_low_overhead_cost_progress.py CPU reader会话46629串行append，不写旧reader；成本controller会话8929/out/log replay-stats-low-overhead-cost-20261012-v2。禁止编辑新活动costv2/generator/helper/原源；每臂全数值/方法审查后新timingpair才作结论。


### 2026-10-12 14:47：低扰动包装post-eval记录错误，版本化修复并重新准入

**独立实证与拒绝。** costv2首q06 control于14:36:26退出1，0完整timing pairs；最终_eval写layer_output_fingerprints.json时NameError root未定义。原raw fingerprint包装持有root，但低扰动入口的同义环境字段是destination，移植时遗漏全局绑定。全部507state SHA7ad9b1a7a890faf5175b136b0c12ea30228bea96060a84f9ee734cd2f0c57ff3/KL .30534985661506653/PPL25.013607025146484已经精确；独立完整13cache/61source SHA及112controlinverse28write28stats和dead-disabled元数据与对应参考精确，只有all28输出工件缺失。failure独立审查independent_completed_recording_failure_audit-20261012-1442.json validFalse，原status/source/log全部保留，不补造指纹，不计862.328966秒control-only为速度配对结果。并非量化数学或已接受六对容量方法失败；原容量summary不改。

**具体修复与CPU执行。** 新tools/prepare_replay_stats_low_overhead_entry_v2.py只将post-eval输出表达式root改destination，数学/量化/timer/earlystats/数据参数及所有其他AST一致。新增tests/test_low_overhead_fingerprint_destination.py实际提取运行原包装：旧函数复现NameError，新函数用28层×2forward CPU tensorfixture真写完整非空文件，异常eval仍原样抛出/所有hooks清理；不再只凭compile确认自由全局契约。newcostv3仅import新generator并加入其SHA绑定，其他admit/verify/timer源保持。WSL wrapper2+原admit1/disk2/runner2/entry2共9CPU通过；out outputs/replay-stats-low-overhead-record-repair-cpu-20261012-v1是准备不是GPU结果。新功能没有更改Hsolver/权重精度/Adam/校准，也没有改原冻结失败源。

**续接与反思。** actual工具会话75844启动新out outputs/replay-stats-low-overhead-cost-20261012-v3/log同名，初始CPU重核六对capacity/cache/source，再idle检查与fresh首control恢复计时；只修记录接入错误必要重试，不重复已完成失败方法寻优。NEWreader工具会话51254 append同总日志，只CPU；旧controller17/GPU79/reader70均已正常因异常结束，不沿用PID。每臂仍完整507/all28/KLPPL/13cache/source/方法之后才接受，对失败不修改方法遮掩；低扰动三对/model与最后报告仍待。原失败工件不dedup/删除，当前D盘~8.56GB，nextarm余量检查保持，completed prepared只事后保真链接，非显存贡献。

2026-10-12 15:16 低扰动v3首控制臂完成、记录修复真实路径验收：q06 p1 control于15:07:47完成，15:15:38 native fresh独立审查 outputs/replay-stats-low-overhead-cost-20261012-v3/independent_completed_control_native-20261012-1512.json validTrue。全部507state canonical、28层非空heldout fingerprint、KL .30534985661506653/PPL25.013607025146484、61绑定source/actual entry、13完整extended cache SHA、两stage算法命令及control112inverse/28write/28stats/dead-disabled完整metadata与已接受内存参考一致。真实GPU后eval已写全部28层工件，证实destination记录修复；原v2失败不改。control quant1096.531017082秒/collector2.773640983秒只是单臂，0完整timing pairs、不能声称加速或最终成本；CPU观察仍保留、计时不含prep/eval/hash。已完成arm prepared逐字SHA事后只读链接保所有路径/字节/独立Fisher，非显存贡献。actual controller28/reader37/唯一candidateGPU185；候选15:15已12/28量化，原controlGPU90结束。继续三timing对/model及必要报告，不改活动source、不并发、不推送。

2026-10-12 15:31 低扰动成本首个完整配对：costv3 q06 p1于15:27:11完成，NEW independent_completed_pair_native-20261012-1531.json实际15:30:51 native fresh validTrue。两臂各507state/全28非空输出/KL .30534985661506653/PPL25.013607025146484、61source/actualentry、13extended cache SHA、两stage算法命令及所有inverse/write/dead/stats metadata与对应已接受dense内存参考精确，原destination修复真实两臂验收。C/N quant1096.531017082→988.822695287秒，collector2.773640983→2.648890611，collector+quant1099.304658065→991.471585898秒，单对ratio1.108760627839206；这只是q06一对，不能据此称稳定加速或两模型成本结论，保留所有慢值和后续相反顺序。C/N statusSHA e12bb7e6d394f4ec24e84390946915cfcdfeddcd428e4112438d53691e1580d1 / 822836cd5e3736b4808c92cd79f6283ab25737a239f019c9975e75eb525f4fb3。CPU方法观察未关闭、timers不含prep/eval/hash非端到端；没有重复dense遥测，新显存结论仍来自已接受六对完整区间/逐point/global审查。每臂completed prepared事后保字节链接，不算GPU收益。actual15:29 controller28/reader37/唯一GPU275为q06 p2 candidate层2；p1GPU185结束，第二对N→C，第三对C→N，随后1.7三对。继续独立完整验收/必要报告，不修改61冻结source、不并发GPU、不推送。

### 2026-10-12 16:11：低扰动第二对相反顺序验收，速度方向反转

**方法/理论。** 继续同一已接受stable+stats工程，预定p2 N→C用于暴露执行顺序及运行状态影响；OLLA/Checkmate的存储生命周期依据不推导速度必然提升。保留全部时间，不剔除慢值、不扫描算法参数。

**实证/效果。** q06 p2已16:05:46附近完整结束并串行接p3；独立fresh native independent_completed_pair_native-20261012-1610.json实际16:10:09 validTrue：两臂各507state/全部28非空输出/KL .30534985661506653/PPL25.013607025146484/61source及实际入口/13完整extended cache SHA/两stage算法命令/全部inverse-write-dead-stats metadata精确，pair身份/顺序/原timer和统计加和完整。C/N quant811.464644209→1249.974339955秒，collector2.554306777→2.800949035，collector+quant814.018950986→1252.775288990秒，ratio0.649772515582。C/N statusSHA252a8fa0820dd2fbfc59efce60954d5211357a7c9d072e79c611ed8441749a90 / 7378c33217a9666727cc1b787cffb9e4856cfd7f81288f0dfd12eac474b0ac1e。

**评价/反思。** 第二对候选更慢，方向反于首对；低扰动不等于运行条件无波动，目前明确不能宣称稳定加速，未确认热状态/功耗/频率或后台因素唯一原因。不据速度负向否定已通过的容量/数值结论，也不把两对当三对，不为改善均值重跑/删值。CPU观察仍留、collector+quant排除prep/eval/hash不是端到端；dense每区间/逐point/global证据由已接受容量六对保留，timing没有新密集遥测。actual16:10 controller28/reader37/唯一GPU455为q06 p3 control层7，旧p2 GPU365结束；继续p3及q17三timing对与必要报告，不改61冻结source，不并发不推送。

### 2026-10-12 16:55：0.6B三对低扰动成本完整，未证明加速

**方法/理论。** 原简stable+stats工程按预先C→N/N→C/C→N三对，数学/FP32 Hsolver/BF16权重/校准Adam不变，CPU方法观察保留。存储生命周期理论与显存下降不保证walltime下降，时间证据与dense显存证据分开评价。

**实证/效果。** p3于16:44:53完整，native independent_completed_pair_native-20261012-1653.json实际16:53:05 validTrue。p3 C/N quant1002.827937251→1047.339430312秒，collector2.942660246→3.253914315，sum1005.770597497→1050.593344627；C/NstatusSHA1913588ea0c21ca699ac31299e9ec76278b8632f4f41589755120dab877f8086 / 32b82746475875f27bc3266184f3ba090718cbfc4b5b75a706720335f49ed968。NEW independent_three_pair_qwen06_summary-20261012-1655.json实际16:54:57 fresh六臂重新all28/507/KLPPL/source/入口/13extended cache流式SHA/方法/两stage命令/记录SHA验收validTrue。三对C quant[1096.531017082,811.464644209,1002.827937251]均值970.274532847/sampleSD145.294536509秒；N[988.822695287,1249.974339955,1047.339430312]均值1095.378821851/sampleSD137.043360776，候选均值+12.8937%；collector+quant均值+12.8720%。全部时间保留无剔除。

**评价/反思。** 0.6B容量已三对下降且全数值一致，速度三对一快两慢且CV约15.0%/12.5%，本机这组三对没有稳定加速证据，观测平均成本增加，不把它解释成已确认方法固有惩罚或唯一热/功耗原因。结果只cal2×1024pilot、collector+quant排除prep/eval/hash非端到端，正式2048/小GPU/多卡未验。q17成本尚零完整pairs，不能推广两模型速度结论。actual16:52 controller28/reader37/唯一GPU647为q17 p1 control层3，q06原556退出；继续q17三对必要完整验收和最后报告，不改61活动source，不并发不推送。已有六对内存17766interval/71064point与NVML采样global证据独立保留，不冒称计时有dense连续证明。

2026-10-12 17:23 新docs/realq_system_memory_research_report_20261012_draft_v1.md整理已接受两模型六对容量、归因/生命周期方法、0.6三对成本/负向分支/工具错误/文献和限制。明确草稿，1.7成本仍pending，不把论文可显存主目标讨论当停实验指令，不冒称正式2048/多卡/小GPU或稳定加速。未修改活动61source、原证据或GPU队列；实际q17p1 control层25正常，继续必要完整成本及最终报告。



### 2026-10-12 18:07：方法贡献整理v1（文档，不新增GPU实验）

**方法/理论。** 按用户收窄题目与整理贡献要求，新docs/realq_storage_method_contributions_20261012_v1.md将求逆工作区、已消费H接管、down稳定写回、私有H及回放统计死亡点组织为一个面向REAL-Q消费顺序/存储别名约束的管理方法。补充研究问题、不同存储计数、释放/复用条件、实现映射、已有技术归属和可写贡献草稿；REAL-Q/OLLA/Checkmate/PyTorch主来源核对，references新增REAL-Q v2元数据。不宣称通用最优、首创、五种独立算法或把checkpoint/CE/审查工具列作新增方法。

**效果/评价。** 引用已接受两模型各三对完整容量、17766interval/71064point和0.6B三对时间代价，保留0.6局部+96与原strictFalse、采样及pilot限制。新增document_consistency_and_source_audit.json只核对文档链接、汇总绑定六审查字节及活动61source当前SHA未变，不能替代507/28输出/cache完整重新验收或新增GPU结果。

**反思/后续。** 组合容量不等于各项独立贡献，组件/前缀不能拼成full消融；列出带依赖的必要消融和容量边界补强计划，未自动启动或重复旧失败。1.7B低扰动预定成本继续原串行链，最终成本及报告仍待。不修改活动源码/原status/草稿/失败证据，不并发GPU不推送。


### 2026-10-12 18:19：1.7B首个低扰动成本完整配对

**方法/理论。** 延续同一已接受stable+stats存储方法，预定p1 C→N，数学与算法参数不变，CPU方法观察保留。显存主目标与时间代价分别评价，不据单对建立稳定加速。

**实证/效果。** NEW independent_completed_pair_native-20261012-1818.json实际18:19:03 validTrue，fresh native两臂全部507state/all28非空/KL .625495195388794/PPL25.562191009521484精确，61source及actualentry/13完整extended cache SHA/两stage算法命令/全部inverse-write-dead-stats metadata与参考一致，pair身份顺序和timer原记录精确。C/N quant2498.588322917→2355.394039051秒（候选单对变化-5.7310%），collector4.139627128→4.640291513，sum2502.727950045→2360.034330564秒，ratio1.060462518546。C/NstatusSHA c44388507c4881e3e02294e5fba18a4a96ae6a5c3a7dc7d2b69cca9cf34222f3 / 62cf08a3c00920c6b9e8818d1a819da48f6ecfe65c3cad7c4288717f5df9ef52。

**评价/反思。** 只有q17一对；q06三对平均更慢的完整证据仍保留，不混成两模型稳定加速结论，不剔慢值。collector+quant排除prep/eval/hash非端到端、CPU观察非零开销；timing无新dense遥测，容量仍基于已接受六对17766区间/71064point/global。cal2×1024pilot非正式2048/小GPU/多卡。actual18:18 controller28/reader37/唯一GPU870已q17 p2 candidate第0层，p2 N→C后p3 C→N；旧772退出，不沿用。继续所有完整fresh独立审查及最终报告，不改61活动source，不并发不推送。方法贡献文档18:07已独立整理，论文40多页可按分析/方法/验证规划，但单项消融、正式规模和容量边界仍须补强，篇幅不等于证据充分。


2026-10-12 19:08 只读运行时观察：q17p2 candidate唯一GPU870继续第6层，controller28/reader37，无新completedpair。19:06日志tqdm elapsed48:36与实际ps worker elapsed11:54不一致；19:07 Linuxwallclock及perf_counter/monotonic/uptime独立记录于independent_clock_runtime_observation-20261012-1907.json。过去1h筛选主机电源/时间事件无返回，不证明无暂停/时钟调整，也不能据此确证热/频率/方法成本原因。原timer/慢值保留，不剔除或为改均值重跑，不改变活动源或队列；最终按实际计时scope与环境限制评价。此文件仅环境观察非数学/缓存完整验收，不新增GPU，不推送。


### 2026-10-12 20:17：1.7B第二个低扰动成本完整配对，方向反转

**方法/理论。** 按预定p2 N→C串行验证原简stable+stats工程，保持同校准/FP32求逆/BF16权重/Adam数学不变。生命周期存储优化不保证速度提升，时间代价保留全部原记录。

**实证/效果。** NEW independent_completed_pair_native-20261012-2016.json实际20:17:13 fresh native validTrue，两臂507/all28非空输出/KL .625495195388794/PPL25.562191009521484精确、61source/currententry/13完整extended cache SHA/两stage算法命令/完整inverse-write-dead-stats metadata均通过，pair顺序身份与原timer精确。C/Nquant1759.998356115→2052.422811679秒（候选本对+16.6150%），collector4.102322823→4.552469080，sum1764.100678938→2056.975280759秒，ratio0.857618803415。C/NstatusSHA 6cbe089cb861bf2d5aa828b7ec6c3b3b9684f307c776d7a11a56cfa328006173 / b6dce304485f618d34d9f79c8bae5b1b034701885e003549246e4e037bec0ed2。

**评价/反思。** q17只有两对，一快一慢，不宣称稳定加速或三对最终成本。19:07时钟观察保留；源码核对collector/quant使用CUDA同步的time.monotonic差值，不能以tqdm墙钟elapsed改写原timer或推导暂停原因，不确认唯一热/频率/方法惩罚，不剔慢值或重跑改善均值。CPU方法观察仍留，sum排除prep/eval/hash非端到端；timing无新dense峰证据，已接受容量六对17766interval/71064point及全部global保持原证据。pilot非正式2048/更小GPU/多卡。actual20:16 controller28/reader37/唯一GPU1040为q17 p3 control第0层，旧GPU958/870已结束；p3 C→N后fresh三对六臂summary和两模型成本/最终报告，不因队列进度停止，不改61活动源、不并发不推送。


### 2026-10-12 21:31：当前pilot研究链完整收尾（容量与时间成本）

**方法/理论。** 最终选择原简workspace/消费H复用/down-only稳定写回/末组私有H死亡点/earlystats统一存储方法，保留原FP32求逆、BF16权重、校准/seed/Adam数学。OLLA、Checkmate、REAL-Q与PyTorch2.11来源及创新边界详见NEW docs/realq_system_memory_research_report_20261012_final_v1.md，不将通用生命周期理念或既有checkpoint/CE记为新增算法。

**实证/效果。** q17 p3 C→N于21:20:00完成，NEW independent_completed_pair_native-20261012-2126.json实际21:27:15 validTrue，两臂507state/all28非空/KL .625495195388794/PPL25.562191009521484精确，61source/实际entry/13完整extended cache SHA/两stage命令/完整inverse-write-dead-stats/timer身份顺序通过。C/Nquant1756.192297697→1745.818439124秒、collector4.641346405→4.205734768、sum1760.833644102→1750.024173892；C/NstatusSHA793d20673d206591b511c5dbd9d6e7d4348a424cf7acd0834f40aa270e151edc / 4c46062ac8ace0f8ac9ec262023e7ad279b34a845896e5b6c56d63e48d431290。

NEW q17 independent_three_pair_qwen17_summary-20261012-2128.json及q06 independent_three_pair_qwen06_summary-final-20261012-2129.json分别21:28:38/21:28:46重新六臂完整通过。q17 C量化均值2004.926325576/sampleSD427.528066019，N2051.211763285/sampleSD304.789604457秒，N均值+2.3086%，collector+quant+2.3122%；q06 C970.274532847/SD145.294536509，N1095.378821851/SD137.043360776，N+12.8937%，sum+12.8720%。预定顺序及全部快慢值保留，无剔除。

NEW outputs/realq-final-complete-acceptance-20261012-v1/fresh_memory_six_pair_summary.json再次CPU重新执行六对内存507/all28/KLPPL/source/cache/method/2961区间峰与11844逐point/global验收，累计17766/71064，与原status/cache/全局峰一致；q17局部0、q06累计3区间6point各+96MiB保留旧strictFalse，按用户11:15硬件全局目标容量通过。q17 NVML6629→6089MiB/alloc5605.190918→4883.690918/res5872→5332，q06 NVML4247→4119/alloc3151.033691→3010.087402/res3490→3362。final_capacity_and_cost_summary.json绑定全六对容量、全六对计时及原controller记录SHA，validTrue。复审只CPU读取，没有额外GPU重跑。

**评价/反思。** 两模型三对均无稳定加速证据；显存为主结果，计时分布为成本，均值变化不确认固有惩罚或唯一热/功耗原因。19:07时钟观察保留，CUDA同步time.monotonic原timer不按tqdm墙钟改写。CPU观察仍留，collector+quant排除prep/eval/hash非端到端；NVML采样非连续证明。所有结果仅cal2×1024 pilot，正式2048/实际小GPU/多卡/跨架构及同工作量规范消融未完成。统计持有量非峰降、after_inverse含后续量化刷新，池局部零回升无额外global收益不推广，所有原失败/工具bug/重启中断证据保留。

**完成范围。** 原costv3 controller与GPU均正常退出，status实际validTrue/finished21:20:00；actual ps不沿用28/1122。NEW final报告链接/数字/来源再次核对后关闭本轮real-q自动复查。当前预定研究链与必要完整验收/报告完成，论文扩展研究列待办不冒称已验、不自动另启GPU。原草稿与贡献v1保留当时pending快照，以final报告更新；不推送。

### 2026-10-12 22:15：服务器正式配置入口整理（仅CPU准备）

方法/理论：NEW scripts/run_storage_method_formal.sh、tools/run_storage_method_formal.py及docs/run_storage_method_formal.md将已验收统一存储方法绑定至portable checkpoint入口，不读取历史outputs。rawentry与原已接受entry仅Stage2缓存保护报错文字不同，数学、消费者边界、原helper不变。参考旧formal_realq_compare.sh及本地论文表6：cal2048×2048/seed1/W4A16/4卡、q06LR3e-4/q17LR5e-4/末层1e-5，Adam32样本/cosine/滑窗/P95/P99、默认QA任务。明确两臂group_parallel_quant=none（本方法未接rank分组路径），保留4卡DP/Stage0FSDP，显式native分配器；不能假称旧rank路径及expandable设置不变或论文表逐位复现。

具体方法：默认串行q06 C/N后q17 C/N，各自fresh Stage0+28量化+全测试评估，Stage2拒绝重算缓存。control包含既有checkpoint/CE/输入卸载，candidate只新增workspace/consumeH/stable/downdead/earlystats。独立全rank507/all28/KLPPL/source/完整cache/112inverse-reuse28write-dead-stats、exact interval及逐point/global审查，局部回升保留诊断，各GPU NVML采样峰另核。正式参考值来自同服务器control，不复用pilot canonical。新目录拒绝覆盖，失败停止后续，保留原status/日志；不扫LR，不推送。

效果/评价：WSL CPU5tests通过（正式参数被真实production parser接受、actual源builder、单wrapper、禁用/缺rank/工作区未启用拒绝、算法数据参数变化拒绝），Bash语法及实际shell dry-run四臂通过。NEW outputs/storage-method-formal-cpu-20261012-v3/real_pilot_auditor_cpu_regression.json对既有q06完整pilot读后validTrue，2961interval/11844point及local1interval2point+96完整保留；此为新auditor CPU接入回归，不是新GPU收益。cpu_preparation_status.json绑定新文件SHA，未启动任何formal GPU。较早CPU parser揭示QA应为--lm_eval而非环境变量名--enable_qa_eval，修正后再次真实parser通过，旧CPU计划保留不覆盖。

反思/范围：4卡、旋转、全正式数据和全QA尚需用户在服务器实际验证，脚本准备不作成功或显存比例保证。用户此次明确要求先服务器0.6/1.7正式配置；仅交付方法说明及启动入口，不恢复本机自动实验。正式多卡cache按实际完整文件清单，不硬套单卡13文件。密集profile时间不是低扰动速度，NVML不是连续证明。此前pilot final报告和所有冻结源/负向证据保留。

### 2026-10-13：正式服务器双卡 CPU master post-quant 审查修复

方法/理论：用户提供首个0.6B control日志，Stage2量化至layer27后rank1在完整state哈希`.cpu()`遇到meta张量退出，torchrun终止rank0。原CPU master设计由rank0持有完整CPU模型并做KL/PPL，其他rank释放每层到meta；不应要求meta rank装回全模型来审查。此为正式记录入口接入错误，非该日志中的OOM或方法/数值失败，未完成全评估不能算有效正式pair。quant_stop_layer=27包含第28层，进度27/28不独立证明缺层或完成。

具体方法：新增tools/storage_method_distributed_audit.py与独立tools/run_storage_method_pipeline.py。只改生成entry量化之后的hash块，rank0完整507state原格式字节SHA及28层真实非空输出，所有rank真实布局SHA/name-shape-dtype、meta数量、rank/world_size/所有权；非所属rank明确state_tensors=0/state_sha256=null及layout_only，不广播/复制rank0数值证据。纠正22:15记录“全rank507/all28”不适用于CPU master的要求。普通复制模型模式仍每rank完整数值和评估一致。每rank inverse/write/dead/stats覆盖、fresh缓存、SOURCE/环境/命令、exact区间峰、独立每point及全局显存准入均保留，另核全部rank过程峰身份，owner记录从实际rank0文件读取不取最先打印的rank1日志。旧checkpoint entry、full pipeline runner、量化/helper及旧结果不改。

效果/验证：WSL真实两Gloo CPU进程，用CPU传输替代production CUDA-only广播，调用真实release_layer及实际生成measured_fwrd/held-out hook；原入口在相同meta权重复现NotImplementedError，新入口owner507/all28审查通过且rank1仍meta、无复制模型及虚构trace。完整owner含meta、缺失/重复rank、布局/所有权/world/count/digest错误、缺/空末层输出、伪造非owner字节哈希、非CPU模式缺rank及缺过程峰均拒绝。相关42tests通过（27.20秒），真实pipeline main以CPU worker/telemetry fixture验证owner选择和rank过程峰完整性。双卡四臂正式shell dry-run成功；NEW outputs/storage-method-distributed-audit-cpu-20261013-v1/cpu_preparation_status.json validTrue仅CPU/source。实际entry除了post-quant hash外AST相同；stage launch/profile读取函数、CUDA峰和评估包装相同，11份冻结源与已推送b587245字节相同，真实worker符合Python3.10语法。没有新增CUDA正式实验/显存收益。

评价/反思：此前单进程pilot及compile/parser测试漏掉多卡CPU master权重所有权，应以真实双进程释放和输出包装执行补覆盖。回归初版fixture误用production CUDA-only广播，另一次torch facade漏uint8，修复fixture后全部通过；native Git SHA核对因WSL不能解析Windows worktree gitdir改用Windows原生Python，未修改gitdir/源或伪造结果。CPU修复不能称服务器GPU已通过。旧server未设置save_qmodel_path，Stage0准备checkpoint非已量化模型；进程退出后不能仅补hash恢复，需要新目录fresh重启两臂，原失败目录保留。用户已授权推送，修复交付到mine/realq-warmadam；本机pilot全部正负向及最终报告保持原结论，正式结果仍待服务器完整验收。

### 2026-10-13：正式实验四臂实时总日志接入

方法/具体工程：用户要求沿用一个总日志看全部输出，核对eefa863后发现仅控制器stdout进入nohup总日志，worker仍各写stage0/stage2，先前nohup命令未自动汇总。新增storage_method_log_stream.LiveFileRelay，阶段原stdout/stderr保留后每0.25s转发到arm控制stdout，正式控制器再转发全部arm输出到nohup总日志；PYTHONUNBUFFERED=1，plan标记aggregate_worker_stdout。tqdm回车在总日志转为换行，原阶段文件字节不改，最终退出尾部同样排空，rank/tag/stage原输出保留。NVML CSV/区间JSONL仍独立完整原始工件，不向进度日志灌100ms采样。无数学/配置/模型/缓存/峰值门槛变化，无GPU操作；密集profile非低扰动速度结果。

效果/验证：新增tests/test_storage_method_log_stream.py验证缺失文件等待、增长UTF8跨块/回车、去重/截断拒绝，真实CPU子进程stage stdout+stderr及成功/失败退出尾部，旧四臂plan动态出现stage日志及最终失败status，新aggregate计划防重复汇总。日志、双进程meta审查和正式参数接入27tests通过（31.08秒）；真实Bash双GPU四臂dry-run生成新计划，只CPU，无新正式GPU结果。NEW outputs/storage-method-log-stream-cpu-20261013-v1保存计划/来源快照，原分阶段日志和此前失败/CPU验收全部保留。

评价/反思/在线用法：此前日志仅说明输出路径，不满足用户一份总日志的使用要求，应核对嵌套subprocess的两层重定向而非仅给nohup。新增单文件tools/stream_storage_method_formal.py可对已运行的旧plan只读汇总各arm pipeline/stage0/stage2，读已有输出后继续跟后续模型/阶段，保留身份，实际controller结束status出现才退出；不启动GPU、不重启实验。在线仅git fetch+git show提取reader至/tmp，不pull/修改任何活动绑定源。新正式程序在本次结束后pull供下次运行。用户原授权交付修复到mine/realq-warmadam；pilot结论不变。
