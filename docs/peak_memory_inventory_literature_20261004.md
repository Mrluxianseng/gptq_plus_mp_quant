# REAL-Q 全程峰值显存盘点与省显存文献（2026-10-04）

## 结论先行

现有阶段 profile 显示，显存最大项会随优化而迁移：

1. **原始 32×256 全模型路径：最后一层真实全词表 KL refresh 是全程峰值阶段。** 全程峰值 allocated 为 5,926,841,856 B（5.52 GiB），reserved 为 7,465,861,120 B（6.95 GiB）。阶段 profile 将峰值定位到 `layer_27_quantization_and_refresh`。
2. **启用逐 token 分块 KL 后：Stage-0 全模型 saliency/Fisher 反向成为新峰值。** 全程峰值 allocated 为 2,774,529,024 B（2.58 GiB），reserved 为 3,418,357,760 B（3.18 GiB），峰值阶段为 `static_end_to_end_saliency_fisher`；末层量化/refresh 阶段采样峰值约 2.38 GiB。
3. 因此，当前要继续降低全程峰值，优先次序应为 **Stage-0 的保存激活/反向图**，再做**避免完整 vocab logits 常驻的 KL 实现**。GPTQ Hessian、列块补偿、层间输入缓存和 Triton 内核不是这些 profile 中的最大阶段。

这里的“峰值项目”分为**阶段实测**与**按张量形状估算**。当前记录器能定位创下全程高水位的阶段，但没有记录峰值瞬间每个仍存活张量的分配栈和名称；所以不能把某个张量写成已测得的显存占比。

REAL-Q 的系统目标应理解为**降低整条校准与量化流程的 GPU 显存需求，使方法能在显存更少、GPU 数量更少的机器上完成**。不是只缩小最终量化模型，也不是只优化一个阶段。本文当前的 Qwen3-0.6B 数据用于定位机制，不能代替跨模型证据；所有优化都需要在多个规模上按各自论文配置复核。

## 证据和适用配置

主要证据来自本地 Qwen3-0.6B、W4A16、28 层完整量化的配对实验：32 个 WikiText-2 校准样本、每个 256 tokens、32 个 backward 样本、最终评测 8×1024 tokens、seed 44。control 与 KL-chunk16 candidate 的 507-tensor 权重状态 hash、最终 WikiText2 KL/PPL 相同；阶段 profile 的实验记录在 `docs/memory_optimization_assessment_20261001.md`。这不是论文的正式 2048 校准样本规模，结论只对这组本地配置直接成立。

2026-10-04 的 Stage-0 saved-tensor offload 配对实验也完成了同一 32×256 配置：全程 peak allocated 两边均为 5,913,079,296 B（5.51 GiB），peak reserved 从 6,184,501,248 B 增至 6,289,358,848 B。它证实 offload 原型保持了全部 28 层指纹、状态 hash、KL/PPL，却**没有降低全程峰值**，因为这个短序列配置仍由后续量化/refresh 阶段主导。详细结果见 `docs/realq_stage0_saved_tensor_offload_20261004.md`。

另有一组完整 32×256 实验的不同运行状态报告峰值 allocated 5,913,079,296 B（5.51 GiB）、reserved 6,184,501,248 B（5.76 GiB）。这些 profile 的阶段记录能说明峰值阶段，不应把不同日期、代码 hash 或运行状态的绝对字节数混成一组 A/B。

## 跨模型目标：REAL-Q 意义上的“省显存”

REAL-Q 附录 D.6 把**校准期 full Fisher 缓存**和**Stage-0 全精度前向/反向的瞬时内存**作为内存开销来讨论，并指出这使其峰值高于 vanilla GPTQ。它以 BF16 矩阵计算 Fisher 的理论缓存成本：全模型 `2×L×d²` 字节。论文表中 Qwen3-0.6B 是约 56 MB，而 LLaMA-3.1-70B 是约 10 GB；论文同时说明 70B 通过 FSDP 和 activation checkpointing 分摊 Stage-0 参数、梯度和 optimizer state。这个 Fisher 公式是跨模型可比较的静态缓存规模，不等同于运行时峰值，也不意味着小模型的 Fisher 矩阵就是它的最大项。[REAL-Q 附录 D.6 与系统级 Future Work](https://arxiv.org/html/2609.00049)

因此，论文级的验证目标应包括以下三个口径：

| 口径 | 记录项 | 它回答的问题 |
|---|---|---|
| 每卡峰值（首要） | 每个 rank 的 `max_memory_allocated`、`max_memory_reserved`、NVML/device-used 最大值及发生阶段 | 单卡是否会 OOM、最少需要多大显存的卡 |
| 集群资源（次要） | GPU 数、每卡峰值与运行时长；另报总 GPU-hours | 能否减少参与的 GPU 数，或降低整次校准/量化的资源成本 |
| 结构性缓存与代价 | Fisher bytes、模型/参数驻留量、CPU offload 量、峰值 host RAM、通信量 | 节省来自哪里，是否只是把 GPU 压力转移到 CPU/通信 |

**不能用各卡的平均显存代替每卡峰值。** 最忙的 rank 决定 OOM；多卡实验要逐卡报告峰值。Device-wide 峰值与 PyTorch allocator 峰值也需分开记录。为了证明最低显存需求确实下降，最终应逐步降低可用 GPU 显存或 GPU 数，验证 control 会失败/OOM 而 candidate 能完整运行；这项“可运行边界”测试要在不改变论文算法配置的前提下做。

模型规模会移动瓶颈：Stage-0 full-precision 参数驻留大体随参数量增长；Fisher 缓存随 `L×d²` 增长；反向保存激活主要随 rank-local batch、序列长度、层数和实现增长；末层 full-vocab KL logits 随 `batch×sequence×vocab×dtype_bytes` 增长；GPTQ 的 Hessian/当前线性层工作区则受输入维度平方和矩阵形状影响。因而不能把 Qwen3-0.6B 上“KL chunk 后 Stage-0 最大”外推到 8B/32B，或者把 70B 的 Fisher 存储量误当作所有规模的 transient peak。

建议的跨模型分层验证集是：一个小模型用于快速迭代、一个中等模型检验规模迁移、一个更大且论文也报告的模型检验显存可扩展性。具体型号和每项资源允许的规模由服务器可用卡决定；参数需逐模型对齐论文的 W4A16、per-row 量化、校准集/序列长度、batch 与并行配置。论文附录指出除 LLaMA-3.1-70B 使用 256 校准样本外，其余模型实验使用 2048 个校准样本，因此不能把“所有模型都 2048”误当成论文统一设置。

## 峰值阶段及其内部候选占用项

| 排名/状态 | 已测峰值阶段 | 已测数值 | 阶段内显存来源判断 | 证据等级 |
|---|---|---:|---|---|
| 1. 未启用 KL 分块 | `layer_27_quantization_and_refresh`，末层真实 KL refresh 创下全程高水位 | 全程 allocated 5.52 GiB；reserved 6.95 GiB | student/teacher LM-head logits、vocab 维 FP32 softmax/KL 中间张量，以及当前 refresh 的反向图；分阶段峰值能确认所在阶段，但没能给这些对象逐个做存活字节归因 | 阶段是实测；内部来源由代码路径和形状推断 |
| 2. 启用 `--kl_logit_chunk_tokens 16` | `static_end_to_end_saliency_fisher`（Stage-0） | 全程 allocated 2.58 GiB；reserved 3.18 GiB | 全模型 backward 的 autograd saved activations、当前模型参数、logits/NLL 与 Fisher/saliency hook 临时张量；代码冻结了所有参数，因此**不会**为模型参数生成/保存参数梯度缓冲区 | 阶段是实测；内部张量未做精确 census |
| 3. 两条路径都有的显存底座 | 模型参数、CUDA context、当前层/缓存和工作区 | 已包含在上述阶段峰值中，未单独拆账 | GPU 上模型权重和运行时工作区是底座；`inps`/`fp_inps` 一类 hidden-state 缓存按 `N×T×H×dtype` 增长。现有 32×256 Qwen3-0.6B 配置中，这类两份 BF16 缓存约 32 MiB，远小于 5.5 GiB 的峰值 | 代码路径及形状估算 |
| 4. 非峰值候选 | Hessian 累积、GPTQ 列块补偿、Triton 内循环 | 在已有 profile 中没有创下全程高水位 | 它们可能耗时长或在更大 hidden dimension 下占用增加，但当前配置不是第一显存优化目标 | 阶段 profile 排除为本配置峰值 |

### 全词表 KL 的尺寸估算

Qwen3-0.6B 的词表大小为 151,936。最后层真实 KL refresh 的 backward microbatch 为 4、序列长度为 256，因此一个完整 logits 张量含 `4×256×151,936 = 155,582,464` 个值：

* BF16 单份 logits 约 311 MB（0.29 GiB）；student 与 teacher 两份约 0.58 GiB。
* FP32 单份 cast 约 622 MB（0.58 GiB）。`log_softmax`、teacher 概率、差值和乘积等运算可能让多个 full-vocab FP32 临时量在一个阶段内同时存活。
* 这些数字是由元素数和 dtype 得出的**单对象大小估计**，不是 profile 对同时存活张量的直接测量。真实峰值还包含模型权重、反向保存对象、CUDA workspace 和 allocator 状态。

现有 `chunked_tokenwise_kl_from_logits` 把 FP32 KL 中间量限制在 token chunk 内，并在 backward 中逐 chunk 重算 KL 梯度；但它仍接收并保存完整的 student/teacher logits，以便外层 LM-head backward 使用。因此它解决了“大量 FP32 softmax/KL 中间量”，还没有解决“完整 logits 本身”的常驻问题。真实 refresh 形状的独立 CUDA microbenchmark 中，KL 函数 peak allocated 从 3.48 GiB 降到 0.92 GiB；完整 32×256 量化则把全程 peak allocated 从 5.52 GiB 降到 2.58 GiB，state/KL/PPL 精确相同。

### Stage-0 为什么会占显存

`collect_static_end_to_end_saliency_and_fisher()` 按 batch 对完整模型做 forward/backward，通过各层 hook 读取激活梯度来累积 saliency/Fisher。它将模型参数设置为 `requires_grad=False`，只在第一层输入重新开启 autograd，因此不会存模型级参数 `.grad`，但为反向传播仍必须保留或重算前向中间激活。32×256 实验的 `global_loss_bsz=1`；启用 CPU saved-tensor offload 时，Stage-0 共把 14,400 个 saved tensors、累计约 25 GiB 的张量载荷复制到 CPU。这里的 25 GiB 是跨整个 precompute 的**累计搬运量**，不是 GPU 同时峰值。

此前 seq-2048 的 collector-only 实验中，Stage-0 peak allocated 从 7.0645 GiB 降到 4.5410 GiB（−35.7%），输出 logits 完全相同；然而 32×256 完整管线的全程峰值仍在后续量化阶段。故长序列/大 batch 时 Stage-0 offload 有明确空间，短序列完整运行不能仅凭 Stage-0 下降宣称全程节省。

## 文献与可借鉴的方法

### 第一优先：融合/分块大词表 loss，避免保存整份 logits

Wijmans 等人的 *Cut Your Losses in Large-Vocabulary Language Models* 提出 Cut Cross-Entropy，在词表维计算 log-sum-exp 时不把所有 token×vocab logits 写入全局显存；论文报告 Gemma 2 2B 的 loss footprint 从 24 GB 降到 1 MB。它针对 cross-entropy，不能直接原样搬到 teacher-student KL，但它证明了“对词表分块并融合 loss/reduction、避免物化全 logits”是大词表内存瓶颈的成熟工程路线。

REAL-Q 的真实 KL 需要 teacher 与 student 的分布，其可借鉴版本是：对 vocab tiles 流式累计每个 token 的 log-partition、KL 和反向所需统计量，再用自定义 backward/recompute 计算 LM-head 输入及其权重所需的梯度；不能简单只算目标标签，因为 KL 需要整个分布。优先先做固定 logits 的 loss 与 logits-gradient 等价测试，再做真实 lm_head 端到端梯度核对。注意当前 quantization refresh 中 student 模型权重和 teacher FP 权重的语义、temperature/归约方式必须维持不变。

参考：Erik Wijmans, Brody Huval, Alexander Hertzberg, Vladlen Koltun, Philipp Krähenbühl, “Cut Your Losses in Large-Vocabulary Language Models,” ICLR 2025, [arXiv:2411.09009](https://arxiv.org/abs/2411.09009)。

### 第二优先：Stage-0 保存激活的 checkpoint/recompute 或有选择的 CPU offload

Chen 等人的 activation checkpointing 以重算前向换取更少保存中间激活，理论上将深网络训练的激活内存降到次线性；PyTorch checkpoint API 提供了分段重算路径。对本项目应从 transformer block 粒度开始，确认 eval/no-dropout 条件下重算确定性，逐层对齐 hook 梯度和最终 Fisher cache，再比峰值与 wall time。

另一条是 PyTorch `saved_tensors_hooks` 的 CPU offload。已有原型已验证数值一致，但同步 pageable CPU copy 增加了 Stage-0 时间；它也没降低 32×256 全程峰值。可以根据 saved tensor 大小/重复使用成本选择性 offload，并使用 pinned memory、异步拷贝及预取/回收来降低搬运停顿，但应先做内存和速度都记录的 prototype，不应预先假定会更快。

REAL-Q 论文的系统优化展望明确提到 custom memory-efficient backward kernels 与 activation offloading，因此这条直接承接论文 future work；需在论文中区分“REAL-Q 已提出方向”与“本项目设计/实现的调度或 kernel”。

参考：Tianqi Chen, Bing Xu, Chiyuan Zhang, Carlos Guestrin, “Training Deep Nets with Sublinear Memory Cost,” 2016, [arXiv:1604.06174](https://arxiv.org/abs/1604.06174)；PyTorch, [Activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html) 和 [saved tensor hooks](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)；Qian Zhang et al., “REAL-Q: E2E LLM Quantization via Dynamic Gradient Descent,” 2026, [arXiv:2609.00049](https://arxiv.org/abs/2609.00049)。

### 第三优先：长序列时优化 attention 中间量

FlashAttention 通过 IO-aware tiling 计算精确 attention，避免在 HBM 中物化完整 attention score/probability 矩阵。对于本地 seq_len=256 的 Qwen3-0.6B 峰值，目前证据指向 LM-head KL 或 Stage-0 保存激活，attention 不是当前首要峰值；当使用 2048 token 校准或更长上下文时，应重新 profile attention kernel 和保存激活的占比。

参考：Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré, “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness,” NeurIPS 2022, [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)。

### 暂缓：Hessian/量化状态的低精度或分片

目前它们没有造成全程最大阶段。继续压缩 Hessian、更新缓存或量化器状态之前，先证明目标配置的阶段高水位已经移到这些部分。低精度压缩可能改变 GPTQ 补偿，需要逐层输出、507-tensor state hash 和最终 KL/PPL 共同验收。

## 下一步测量要求

现有 profile 回答了“哪个阶段创下全程峰值”，没有回答“峰值那一刻哪些张量占了多少字节”。下一次正式诊断应在两个候选阶段分别添加可控的 allocation snapshot/recording：

1. `layer_27` KL refresh 的真实 forward/backward 内，在 LM-head logits 生成后、KL backward 期间记录 CUDA allocator snapshot；同时记录 logits shape/dtype、student/teacher logits、loss 临时项的生存时间。
2. Stage-0 batch forward 到 backward 的峰值位置记录 snapshot；把 saved-tensor hooks 的对象大小、对象类型、是否 offload、累计 payload 与峰值并列输出。
3. 确保诊断开销单独标注；先用小模型/单层 smoke 检查 trace 完整，再按同配置全模型配对。最终仍以峰值 allocated/reserved、层输出、完整 state hash 和 KL/PPL 为准。

4. 将同一诊断接入跨模型 paired runner。每个模型分别输出 control/candidate 的逐阶段、逐 rank 峰值表、最低可完成 GPU 配置、运行时间、CPU RAM/offload 和模型结果一致性；至少对小/中/大三个规模重复验证。按本项目已确定的严格验收标准，candidate 的每个阶段、每张卡的 allocator 峰值都不得高于 control，且全程峰值要下降；若 device-used 的高频 NVML 采样显示更高，也不能宣称设备级峰值优化。更少 GPU 或更小显存仍可完成是额外的部署价值，不替代该峰值验收。层输出、量化 state、KL/PPL 也必须通过一致性门槛。

只有取得以上对象级证据后，才适合在论文中写“某个具体张量占全程峰值的 X%”。当前可支持的表述是：**KL refresh 和 Stage-0 分别是优化前后的实测峰值阶段；它们的内部候选分配来源由代码路径和张量形状分析解释，尚无峰值瞬间对象级百分比分解。**
