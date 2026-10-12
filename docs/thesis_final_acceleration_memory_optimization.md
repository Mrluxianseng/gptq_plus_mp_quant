# REAL-Q 量化工程的 GPU 加速与显存优化

> 2026-10-10 范围说明：下文为历史受支持配置的组件/量化阶段实验记录，并非当前“含 fresh Stage-0、所有层量化及评估”的终稿结论。当前主线已调整为全流程显存门槛优先；正式规模、多模型与多卡仍按新审计逐项补齐。最新证据和保留/淘汰判断以 `memory_speed_optimization_journal.md` O–Q 节及 `full_pipeline_memory_validation_20261010.md` 为准。

**状态：** 工程实现、三组完整模型加速重复、三组显存配对重复及联合配置逐层输出审计均已完成。  
**验证平台：** WSL Ubuntu 24.04、RTX 4060 Laptop GPU（8 GiB）、Qwen3-0.6B、本地模型权重、W4、REAL-Q `block_gd`。

## 论文可陈述的工程问题

REAL-Q 的列序 GPTQ 补偿循环逐列执行多个小型 PyTorch 运算，产生大量 GPU kernel launch；真实 KL refresh 则在词表维同时物化 student/teacher 的 FP32 log-softmax 和概率张量。前者主要浪费 GPU launch 与 Python 调度时间，后者主要浪费峰值显存。这两种开销处于不同计算阶段，采用两项相互独立的实现优化，并保持默认路径不变。

本文工程贡献的边界是：改变运算组织方式以减少 kernel launch 和临时张量峰值，不改变 REAL-Q 的量化决策、GPTQ 列顺序、梯度更新公式或目标函数。评估采用完整模型输出权重哈希、最终 KL/PPL 和 GPU 资源计量，区分性能改善与量化质量变化。

## 优化一：融合 GPTQ 列内循环

在严格受限的受支持配置中，将每列的量化/反量化、误差计算、对剩余权重的二阶补偿、以及 `GHinv` rank-one 更新合并到一个 Triton kernel。列仍按原顺序处理，block 结束后的 REAL-Q 梯度 refresh 不变。实现使用独立输入/输出行步长，`div_rn` 和关闭 FP fusion，以匹配原 FP32 运算序列。

三组最终源码完整模型配对均使用相同 manifest、28 层、Qwen3-0.6B/W4 和完整评估。每组的 507 个量化张量 SHA-256 均为 `96e6281619f2304fb97c885962fcbae9e7dea8fe0e3f256aa225fc41201a8d4e`；control/candidate 的 KL=`0.23623386025428772`、PPL=`33.27994918823242` 完全一致。

| 配对 | PyTorch 量化时间 | Triton 量化时间 | 加速比 | PyTorch 端到端 | Triton 端到端 | 加速比 |
|---|---:|---:|---:|---:|---:|---:|
| r2 | 712.47 s | 328.26 s | 2.17× | 746.81 s | 359.44 s | 2.08× |
| r3 | 735.84 s | 332.91 s | 2.21× | 767.07 s | 367.48 s | 2.09× |
| r4 | 715.43 s | 332.16 s | 2.15× | 750.22 s | 368.32 s | 2.04× |
| **配对加速比中位数** |  |  | **2.17×** |  |  | **2.08×** |

三组中 Triton 量化时间为 328.26–332.91 s，端到端为 359.44–368.32 s。峰值 allocated 为 5,913,079,296 B（PyTorch）和 5,913,075,200 B（Triton），reserved 两者均为 6,184,501,248 B，说明加速没有实质增加显存。r4 同一详细 profile 下，顺序列补偿内循环累计 CUDA-event 时间为 410.47 s→24.03 s；true gradient-refresh 为 193.87 s→187.23 s，支持加速主要来自列内循环融合而非删减 REAL-Q 梯度阶段。该分项 profile 用于解释机制；总体论文数字仍以三组 wall-clock 配对结果为准。

### 2026-10-02 无竞争 CUDA 进程复测

为排除先前本机其他任务占用 GPU 对耗时的干扰，在同一源码快照、相同完整模型 manifest 和配置下重新执行三组完整配对。开始前 `nvidia-smi` 未见 CUDA 计算进程。独立 GPU 监控每约 5 秒采样一次，共 647 个有效采样点；GPU 利用率均值 42.8%、中位数 38%、最高 99%，温度 56–78°C，核心时钟 210–2520 MHz。监控记录的是整张卡的活动；配套进程日志未可靠解析出进程名，因此不能仅凭它断言整个运行期间完全没有其他 GPU 活动。实验后显存回到 0 MiB 使用。

| 配对 | PyTorch 量化时间 | Triton 量化时间 | 量化加速比 | PyTorch 端到端 | Triton 端到端 | 端到端加速比 |
|---|---:|---:|---:|---:|---:|---:|
| r2 | 699.70 s | 335.19 s | 2.087× | 735.07 s | 370.39 s | 1.985× |
| r3 | 715.09 s | 333.62 s | 2.143× | 750.90 s | 369.45 s | 2.032× |
| r4 | 725.14 s | 333.16 s | 2.177× | 760.12 s | 368.65 s | 2.062× |
| **中位数** |  |  | **2.143×** |  |  | **2.032×** |

三组 control 的量化耗时范围为 699.70–725.14 s，而 Triton 为 333.16–335.19 s；候选耗时在三次间仅相差约 2.03 s，改善幅度远大于本轮运行间波动。三组 507 个张量的 state SHA-256 均为 `96e6281619f2304fb97c885962fcbae9e7dea8fe0e3f256aa225fc41201a8d4e`，KL=`0.23623386025428772`、PPL=`33.27994918823242` 完全一致。峰值 allocated 为 5,913,079,296 B（control）与 5,913,075,200 B（Triton），峰值 reserved 两者均为 6,184,501,248 B。

r4 profile 中，Triton 将 GPTQ 列补偿循环累计时间从 417.432 s 降至 23.431 s；真实梯度刷新时间为 194.479 s 与 187.798 s。该分解与融合 kernel 的预期机制相符：收益来自减少逐列 PyTorch 调度和 kernel launch，而非跳过梯度刷新。此次同源码、完整模型、GPU 监控下的三组重复支持“加速是真实且可重复的”，但结论仍限于上文注明的 Qwen3-0.6B/W4 和受支持配置。

本轮原始数据：配对结果见 `outputs/triton-uncontended-20261002-campaign/paired_repetitions.json`；GPU 利用率/显存/时钟/温度采样见 `outputs/triton-recheck-gpu-monitor-20261002/gpu_telemetry.csv`，WSL 进程记录见同目录 `gpu_processes.log`。每次运行的 manifest 与状态日志位于 `outputs/phase_profile_triton-uncontended-20261002-r{2,3,4}-{control,triton}/`。

性能证据还包括：代表性 3072×128 非连续权重视图（输入行步长 4096、输出行步长 128）在五个追加 seed 下，量化整数、误差、每列权重状态和 `GHinv` 均逐位一致；每个 seed 各跑三次，取各自中位数后，控制路径中位数为 31.107 ms，Triton 为 7.732 ms，跨 seed 的中位加速比为 4.016×。Nsight 选定列循环的 host launch 从每 128 列 2,556 次降至 128 次。Nsight 时间受插桩影响，仅用于说明 launch 机制，不用于端到端加速比。

Triton 路径是 opt-in，默认仍为 PyTorch。当前只支持 `block_gd`、非 atomic quantization、不启用 two-sided rounding、对称逐行低位宽量化、FP32 内部权重、block size ≤128、无并行分组。其他路径会显式报错。证据见 [融合内循环验证记录](triton_gptq_inner_loop_optimization_20260929.md) 和 [内核 profile](gptq_inner_kernel_profile_20260929.md)。

## 优化二：真实 KL 的 token 行分块

新增 `--kl_logit_chunk_tokens N`。student 和 teacher 模型前向仍各执行一次；KL 损失对展平的 token 行分块计算，反向则逐块重算原 PyTorch KL 表达式并通过 autograd 得到 student-logit 梯度。这样把高峰值的 FP32 log-softmax/probability 临时量限制在 N 个 token 行内。teacher 不接收梯度；全词表与既有 top-k support 都保留。`N=0` 是默认旧路径，`N=16` 开启分块。

这里降低的是 KL 计算的临时中间张量，而不是整个 Transformer 的所有激活：完整 BF16 student/teacher logits 和 student logits 梯度仍需存在。若有效 token 行数为 (R)、词表大小为 (V)、chunk 为 (C)，KL 的 FP32 大型临时量从 (O(RV)) 降为 (O(CV))；总算术量仍是 (O(RV))，反向会重算分块 KL，因此该方案的目标是降低峰值显存，不能据此宣称加速。

完整模型 32 样本×256 token 配置的三组配对结果：

| 指标 | 未分块 control | 16 行分块 | 差异 |
|---|---:|---:|---:|
| 507 张量 state SHA-256 | `96e6281619f2304fb97c885962fcbae9e7dea8fe0e3f256aa225fc41201a8d4e` | 相同 | 逐位相同 |
| Wikitext-2 KL | 0.23623386025428772 | 相同 | 0 |
| Wikitext-2 PPL | 33.279949188232422 | 相同 | 0 |
| 峰值 allocated | 5,926,841,856 B（5.52 GiB） | 2,774,529,024 B（2.58 GiB） | −53.2%，三组完全相同 |
| 峰值 reserved | 7,465,861,120 B（6.95 GiB） | 3,418,357,760 B（3.18 GiB） | −54.2%，三组完全相同 |

独立 CUDA logits 测试使用实际 refresh 形状 `[4,256,151936]`、BF16 输入；allocated 从 3,736,600,576 B 降至 986,977,792 B（−2.56 GiB），loss 和 logits gradient 完全相同。真实 Qwen3 final RMSNorm/lm_head smoke 的 hidden gradient 最大差为 0。完整模型 8×128 配置也得到相同 state、KL/PPL，allocated 峰值减少 41.1%。

分块后全程峰值从末层真实 KL refresh 移到了 Stage-0 全模型 Fisher backward（2.77 GiB）。这说明 KL 路径已不再是该配置最大显存阶段；继续缩小 KL chunk 不会消除新的 Stage-0 下界。单次 32×256 wall time control 为 2,157.55 s、candidate 为 1,662.64 s，但机器存在外部 GPU 竞争，**不把该差异作为加速结论**。本优化的论文主张是严格输出等价下的峰值显存削减。

## 单独实验、暂不纳入主结论的 Fisher 标签分块

Stage-0 细分 profile 发现，静态 Fisher 的分类标签抽样会临时创建 `[样本, token, vocab]` FP32 概率。`--static_fisher_label_chunk_tokens` 分块做 softmax 与 categorical sampling，可降低这一操作自身的显存。CUDA 单算子三次实测，峰值由 388,963,328 B 降至 97,256,448 B；中位时间由 9.203 ms 降至 4.131 ms。按分块抽样仍服从同一 categorical 分布，但 PyTorch RNG 调用形状变化会改变抽到的标签，因此完整量化 state 不再与默认路径逐位相同。一次全配置候选 KL/PPL 有改善，但没有跨随机实现/数据种子的重复质量证据；故将它保留为实验性 opt-in，不与严格等价的 KL 分块优化混为一项，也不作为论文主结果。

## 最终重复性与联合配置验收

内存配对重复由 `tools/run_chunked_kl_reproducibility.py` 自动执行：采用相同 manifest、seed、校准及评估设置，先跑 control 再跑 chunk16。任一配对出现量化 state hash、KL 或 PPL 不一致，脚本立即终止后续运行。32×256 配置的三组完整配对都得到相同的 507 张量哈希、KL=`0.23623386025428772`、PPL=`33.279949188232422`。三组峰值 allocated 均为 control 5,926,841,856 B、chunk16 2,774,529,024 B；峰值 reserved 均为 control 7,465,861,120 B、chunk16 3,418,357,760 B，即分别下降 53.2% 和 54.2%。

运行时间波动较大：三组配对量化时间 control 为 2,119.85、1,003.90、992.64 s，candidate 为 1,570.53、979.96、980.75 s；配对时间比中位数 0.976。端到端时间比中位数为 0.980。首组 control 的 2,119.85 s 主要多在量化阶段，不是装载/评估耗时；但首组与后两组的 `gptq_plus_utils.py`、`process_args.py` 源码哈希不同（首组 control/candidate 内部一致，后两组内部一致）。因此该异常不能单独归因于 GPU 负载，也不能把它当作严格同源码重复；GPU 时钟、并发任务或 WSL 调度也可能造成波动，但当时没有足够遥测来判定具体原因。KL 分块没有加速主张，核心结论仍是跨配对重复的峰值显存下降与输出不变。各配对的源码哈希记录在对应 manifest 中。

最终联合配置审计：`torch + KL full` 对 `triton_fused + KL chunk16`。控制与候选的完整 state SHA-256 均为 `96e6281619f2304fb97c885962fcbae9e7dea8fe0e3f256aa225fc41201a8d4e`，KL/PPL 同为 `0.23623386025428772 / 33.27994918823242`。28 层评估输出均记录 8 次 forward，层输出 shape、dtype、调用次数和 SHA-256 全部逐层一致（28/28）。联合配置峰值 allocated 从 5,913,079,296 B 降到 2,731,209,728 B（−53.8%），reserved 从 6,184,501,248 B 降到 2,868,903,936 B（−53.6%）。这次带 CPU 输出哈希钩子的运行只用于正确性和显存审计，不用其 wall time 作为加速结论；加速数字引用上方不带 trace hook 的三组配对。

## 代码、开关与复核

- KL 计算及 custom backward：`utils/loss_utils.py`
- 参数定义/验证：`process_args.py`
- KL 分块接入 refresh 路径：`gptq_utils/gptq_plus_utils.py`
- Triton kernel：`gptq_utils/triton_gptq_kernels.py`
- 阶段显存记录：`utils/memory_utils.py`
- 数值测试：`tests/test_chunked_kl.py`
- 所有 KL refresh surrogate 接入测试：`tests/test_chunked_refresh_loss.py`
- 标签抽样测试：`tests/test_chunked_categorical_sampling.py`
- logits 峰值对照：`tools/benchmark_chunked_kl_memory.py`
- 可自动检查输出等价并在不一致时停跑的重复驱动：`tools/run_chunked_kl_reproducibility.py`
- Triton 完整模型重复与可选逐层输出指纹审计：`tools/run_triton_gptq_reproducibility.py`
- 三组完整模型 Triton/torch 原始配对数据：`outputs/triton-final-campaign/paired_repetitions.json`
- 三组完整模型 KL 分块配对数据：`outputs/kl16-final_paired_repetitions.json`
- 联合配置配对与逐层指纹结果：`outputs/combined-final-layer-audit-c-20261002-campaign/paired_repetitions.json`
- 联合配置逐层输出指纹：`outputs/phase_profile_combined-final-layer-audit-c-20261002-r2-control/layer_output_fingerprints.json` 和 `outputs/phase_profile_combined-final-layer-audit-c-20261002-r2-triton/layer_output_fingerprints.json`

完整复现实验仍通过同一驱动运行，`--pairs 3` 做三组无 trace 的速度配对；最后一次 `--trace-layer-outputs --candidate-kl-chunk-tokens 16` 用于组合配置正确性和显存审计。复核时应保留原 manifest 和独立 run tag，避免覆盖已有输出。

## 论文表述边界

可将 Triton 融合表述为“保持 REAL-Q 量化结果不变，减少 GPTQ 顺序列补偿循环中的 kernel launch，并在 Qwen3-0.6B/W4 完整模型测试中降低量化时间”；将 KL 分块表述为“保持 REAL-Q 输出逐位一致，限制词表维 FP32 中间量的存活范围，并显著降低峰值 GPU memory”。结果限定在已验证的软件、模型、位宽与参数配置；不推广到不受支持的量化模式，也不声称量化精度改善或普适加速。
