# REAL-Q 显存优化：当前基线与下一步

状态：方向已切换到显存优化。已添加 opt-in 阶段显存记录和 opt-in 的 chunked KL 原型。完整 28 层的 8×128 与原始 32×256 两组 control/candidate A/B 均已完成；两组的 507-tensor 量化 state hash、最终 KL 和 PPL 都完全相同。32×256 配置下 peak allocated 降低约 53%，默认路径仍未改变。

## 现在知道什么

Qwen3-0.6B、W4、完整 28 层已有干净 paired profile：

| 路径 | 峰值 allocated | 峰值 reserved | 来源 |
|---|---:|---:|---|
| PyTorch inner loop | 5,913,079,296 B（5.51 GiB） | 6,499,079,296 B（6.05 GiB） | Triton 内循环验证记录 |
| Triton fused inner loop | 5,913,075,200 B（5.51 GiB） | 6,561,988,608 B（6.11 GiB） | 同上 |

此前 Triton 改动主要降低 kernel launch 与时间，基本没有降低 allocated 峰值；reserved 多约 63 MB。它不应被算作显存优化。

这些历史 profile JSON 只保存整个运行的峰值，没有阶段归因。为补足它，本轮加入了 `--enable_memory_profile` 阶段记录器。此前 32×256 配置的 control 在 layer 27 q_proj 真实 KL refresh 期间出现 5.10 GiB allocator 峰值、6.00 GiB reserved，之后因运行过慢被中断；它没有最终评测结果，不能作为有效完整对照。随后改用同一 manifest 派生的 8×128、完整 28 层低资源 A/B，避开长时间停在末层的问题，并获得了完成的端到端数据。

被中断的 32×256 运行期间，Windows `nvidia-smi` 曾显示整卡约 7.7 GiB / 8 GiB 已用、GPU 利用率 90–100%；能看到 PID 26500 `GameAssistVm` 和系统进程，但无权查询它们分别占用多少显存。因此那次 wall time 不能用于速度归因。后续完整低资源 A/B 使用 PyTorch allocator 峰值作为进程内显存证据，并额外比较了量化 state hash。

旧记录器的退出回调曾把当前区间错误标为 `next_phase=complete`；已修复为区分显式正常完成与进程异常/中断退出。新 A/B 的 profile summary 都记录 `termination_status=quantization_complete`，两条完整运行均 return code 0。

## 已实现的内存优化原型

真实 KL 路径先为 student 与 teacher 各生成全词表 logits，再将两者转成 FP32，分别计算 `log_softmax`、teacher 概率和逐项 KL。对 Qwen3 大词表，这会在一个 refresh microbatch 内制造多份 `(B×T×V)` 临时张量。已有的 batch microbatch 只能减小 B，仍完整保留每个 microbatch 的 T×V 中间量。

我增加了可选 `--kl_logit_chunk_tokens N` 路径：模型仍只各自生成一次 student/teacher logits，KL 前向按展平 token 行分块计算；反向逐块重新执行原有 PyTorch KL 表达式并用 autograd 求梯度，避免同时物化全量 FP32 logits/log-softmax/probability 中间张量，同时保持与旧路径相同的反向归约语义。top-k 模式也沿用旧路径的 teacher top-k support。选项默认 0，保留旧路径；例如 `--kl_logit_chunk_tokens 16` 开启原型。

CPU 对照覆盖 FP32/BF16、全词表/top-k；测试 `tests/test_chunked_kl.py` 全部通过。CUDA 上用实际 refresh 形状 `[4,256,151936]` 的 BF16 logits 对照：原实现峰值 allocated 3,736,600,576 B（3.48 GiB），经修正的分块 16 token 为 986,977,792 B（0.92 GiB），减少 2,749,622,784 B（2.56 GiB）；loss 和 logit gradient 完全相同。真实 Qwen3-0.6B final RMSNorm 与 lm_head smoke 中，loss 最大差 `9.54e-7`，hidden gradient 的最大差和相对 L2 差均为 0。可复现脚本：`tools/benchmark_chunked_kl_memory.py` 和 `tests/test_chunked_kl.py`。

### 完整低资源 A/B 结果

运行器 `tools/run_chunked_kl_pilot.py` 从同一 manifest 派生 control 与 chunk16-autograd 两个 run，固定 seed 44、完整 28 层、W4、8 个校准样本×128 token、8 个 backward 样本、2 个 Wikitext2 评测样本×256 token；差别是 `--kl_logit_chunk_tokens=0` 或 `16`。除 `utils/loss_utils.py` 外，所有被跟踪源码 hash 相同；该文件在 control 启动后改过 chunked backward 的实现，但 control 使用默认 0 路径，不会调用新增的分块自定义 autograd 函数。两边的量化 state hash 和 KL/PPL 完全相同，提供了行为级核对。运行目录分别是 `outputs/chunked_kl_pilot_control/` 和 `outputs/chunked_kl_pilot_chunk16-autograd/`。

| 指标 | control | chunk16-autograd | 差异 |
|---|---:|---:|---:|
| 量化后 state | 507 tensors | 507 tensors | SHA256 完全相同：`057f6e16e3abd1188512b2b492cde831c1494ec69587119bf5f3ff7ffead2f73` |
| Wikitext2 KL | 0.31672397255897522 | 0.31672397255897522 | 0 |
| Wikitext2 PPL | 37.200477600097656 | 37.200477600097656 | 0 |
| CUDA peak allocated | 3,703,611,904 B（3.45 GiB） | 2,182,361,600 B（2.03 GiB） | −1,521,250,304 B（−41.1%） |
| CUDA peak reserved | 4,081,057,792 B（3.80 GiB） | 2,239,758,336 B（2.09 GiB） | −1,841,299,456 B（−45.1%） |
| end-to-end wall time | 1,259.04 s | 1,233.71 s | −2.0%，单次运行不能说明提速 |

阶段归因显示，control 的峰值位于 `layer_27_quantization_and_refresh`：阶段采样 3,680,804,864 B（3.43 GiB），全程精确峰值 3,703,611,904 B。chunk16 同阶段峰值为 2,034,247,680 B（1.89 GiB），全程峰值则转到前置 `static_end_to_end_saliency_fisher` 阶段，为 2,182,361,600 B（2.03 GiB）。这说明 KL 分块降低了末层 refresh 峰值，新的限制变成 Stage-0 Fisher 反向；当前低资源设置下再缩小 KL chunk 不会进一步降低全程峰值。

两个 `baseline-profile_metrics.json` 中保存的量化 state hash 完全相同，这是比最终两样本 KL/PPL 更强的逐权重一致性检查。按 block-metrics 文本逐行比较时只有日志浮点格式最后一位的打印差异；完整模型 state 本身相同。该 A/B 已证明低资源配置下，分块减少峰值显存且保持量化结果不变。常规 32×256 候选结果见下节，同配置 control 尚在运行；不同 chunk 大小下的速度成本和跨运行时间收益也未验证，因此仍保持 opt-in。

### 原始 32×256 扩容候选

使用 manifest 中的完整配置配对运行 control 和 chunk16 候选：32 个样本×256 token 校准、32 个 backward 样本、W4、全 28 层、8 个评测样本×1024 token。两次都 return code 0，量化相关源码 hash 完全相同（仅分块参数为 0 或 16）。

| 指标 | control | chunk16-autograd | 差异 |
|---|---:|---:|---:|
| 量化后 state | 507 tensors | 507 tensors | SHA256 完全相同：`96e6281619f2304fb97c885962fcbae9e7dea8fe0e3f256aa225fc41201a8d4e` |
| Wikitext2 KL | 0.23623386025428772 | 0.23623386025428772 | 0 |
| Wikitext2 PPL | 33.279949188232422 | 33.279949188232422 | 0 |
| CUDA peak allocated | 5,926,841,856 B（5.52 GiB） | 2,774,529,024 B（2.58 GiB） | −3,152,312,832 B（−53.2%） |
| CUDA peak reserved | 7,465,861,120 B（6.95 GiB） | 3,418,357,760 B（3.18 GiB） | −4,047,503,360 B（−54.2%） |
| wall time | 2,157.55 s | 1,662.64 s | control 慢 22.9%，但单次受外部 GPU 负载影响，不能宣称提速 |

control 的精确峰值来自 `layer_27_quantization_and_refresh` 阶段（采样峰值 allocated 5,853,572,096 B，精确累计 5,926,841,856 B）。chunk16 仍以 Stage-0 `static_end_to_end_saliency_fisher` 为全程峰值（2,774,529,024 B），末层 refresh 的采样峰值为 2,554,028,544 B。说明在常规配置下，chunking 把 KL-heavy 末层的峰值压低，并使全程瓶颈转移到 Stage-0 Fisher；若论文目标是继续降低内存，下一步应研究 Stage-0 激活 checkpoint/offload，而不是继续减小 KL chunk。

本组运行目录：`outputs/chunked_kl_pilot_full32x256-control-current-src/`、`outputs/chunked_kl_pilot_full32x256-autograd/`。8×128 与 32×256 两组结果都 bitwise 相同，已足以确认该实现的数值语义；速度需要空闲 GPU 下配对重复后才能评价。

## 初步代码审查

* `--offload_inps` 已存在，在 `gptq_plus_utils.py` 将 `inps` 和 FP 参考输入 `fp_inps` 移到 CPU。按当前小实验 32×256×1024、BF16 计算，每份缓存约 16 MiB，两份合计约 32 MiB，不足以解释 5.51 GiB allocated 峰值。样本数和序列长度增大时，其占用按 `N×T×H×dtype_bytes` 线性增长，届时 CPU offload 才可能很有效；单纯启用已有开关也不是代码贡献。
* 一个条件性的大分配位于 `gptq_plus_utils.py` 约 11886 行：若没有可复用的 `fp_inps_final` 缓存，fallback 会将整个 `inps` clone 并一次性 `.to(dev)` 到 GPU，然后虽然逐样本前向，仍把所有样本的 `scratch` 留在 GPU。使用全校准集且长序列时，这可能形成明显峰值；该分支是否为当前目标配置的主路径，需要从运行日志/配置确认。
* `--stage2_cpu_master` 是面向多进程/多 GPU Stage-2 模型常驻的 CPU master 模式；它要求静态预计算缓存等条件，不能直接当成 8GB 单卡 Stage-2 量化的显存解法。
* REAL-Q Stage 0 已有 FSDP 预计算和 CPU offload 选项，适用于多 GPU 的 end-to-end Fisher backward。论文所述 FSDP/activation checkpointing 系统路径与本地单卡 Stage-2 的量化峰值是不同阶段，需分开测量。
* `utils/memory_utils.py` 原有 CUDA allocation history/snapshot helper 没有调用点；新加的阶段 recorder 会标出峰值区间，不记录 allocation stack，因此不能单靠它定位某条分配语句。

## 阶段记录器

当前已经增加默认关闭的**阶段峰值显存诊断器**，只记录、不改变量化计算。对每个阶段记录：进入/退出时 allocated/reserved、后台采样得到的阶段高水位、未重置的全程 CUDA allocator 精确高水位、是否在本区间创下新全程高水位、设备总显存快照，以及时间戳与 layer/module/sample 配置。阶段覆盖：

1. 模型/运行模块加载与 layer materialization；
2. 校准输入捕获及 `inps`、`fp_inps`、`fp_inps_final` 准备；
3. Stage-0 全模型 saliency/Fisher backward（若开启）；
4. 每层 FP reference forward；
5. Hessian/saliency 统计与 `fasterquant.add_batch`；
6. 列块 GPTQ 补偿；
7. Block-GD refresh/backward；
8. 最终 KL/PPL 评估。

阶段边界会同步 CUDA；后台每 5ms 读取 allocator 当前值估算局部高水位。CUDA 全程 peak stats 不重置，保持现有 runner 的整体峰值统计有效。采样可能漏掉短于采样间隔的局部尖峰，但累计 high-water 字段能标出哪个大阶段首次创下全程新高。设备总显存只在边界采样；若后续必须量化非 PyTorch CUDA context/其他进程峰值，再加 NVML 轮询。profile 开销和同步成本须标出，不能和速度成绩混用。

## 最可能先审查的优化点

若 trace 确认输入/参考激活占主导，优先把 `fp_inps_final` fallback 改成分批或逐样本从 CPU 搬运、计算后写回 CPU，而不是持有全量 GPU `scratch`。当前代码本来逐样本执行前向，流式方案有望保留同样的 batch 形状与数值路径，并把这份临时显存从 `O(NTH)` 降至 `O(TH)`。必须比较每层参考 hidden 和最终 KL/PPL，确认浮点输出符合现有基准。

完整配置 A/B 已确认 KL 分块保持量化 state 与 KL/PPL 不变，并把 peak allocated 从 5.52 GiB 降到 2.58 GiB；新的峰值来自 Stage-0 Fisher。下一步可针对 Stage-0 激活做 checkpoint/offload 的小规模原型，并验证 state、KL/PPL；如果要报告速度，再在 GPU 空闲时做至少三次配对重复。不要先改 GPTQ 二阶补偿，它具有列间依赖。

## 验收方式

1. 以 Stage-0 Fisher 为目标，先设计不改变计算语义的 activation checkpoint/offload 原型；先做小层数 smoke，再跑完整模型。
2. 用 507-tensor state hash 与最终 KL/PPL 核验数值一致性，配合阶段 profile 确认峰值确实下降。
3. 若报告 wall-time，再在 GPU 空闲时做至少三次配对重复，给出中位数和波动；不从当前单次 A/B 推断加速。

## 现有依据

* 干净 full-28 memory/speed pair：`docs/triton_gptq_inner_loop_optimization_20260929.md`。
* Stage 2 profile JSON：`outputs/phase_profile_triton-inner-control-full28-recheck-timeout4h/baseline-profile_metrics.json`。
* `--offload_inps` 定义：`process_args.py`；激活存放切换：`gptq_utils/gptq_plus_utils.py` 约 11784–11820 行。
* `fp_inps_final` fallback：`gptq_utils/gptq_plus_utils.py` 约 11860–11910 行。
* Memory history/snapshot helper：`utils/memory_utils.py`，现有仓库搜索未发现调用者。
* 本轮运行记录：`outputs/memory_profile_qwen3-06b-full28-20261001/run.log` 与 `outputs/Qwen3-0.6B/memory_profile_qwen3-06b-full28-20261001/memory_profile/cuda_memory_quantization_rank0_pid33.jsonl`。进程在 layer 27 q_proj 的真实 KL 更新中收到 KeyboardInterrupt，不能用作完整结果。
