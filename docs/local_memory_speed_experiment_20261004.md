# REAL-Q 本地内存与速度优化实验（2026-10-04）

## 结论

在本机 RTX 4060 8 GB、Qwen3-0.6B 上，经过三组配对的全 28 层 W4A16 实验，候选方案通过逐层输出、最终权重、KL/PPL 的精确一致性检查。量化阶段平均 GPU 占用下降 21.8%，平均量化时间下降 37.3%；候选的量化阶段峰值 CUDA allocated/reserved 和 `nvidia-smi` 峰值均低于 control。全进程平均 GPU 占用只下降 6.8%，因为它把后续相同的完整 WikiText-2 评测也算进了均值；此口径也一并报告，不能与量化阶段均值混用。

此结论针对项目中的量化工作阶段。它不表示 REAL-Q 论文 2048 样本的正式复现：受本机资源限制，本次用 32 个 WikiText-2 样本、序列长度 512 做全模型配对；模型仍是本地 Qwen3-0.6B，量化为 W4A16，评测完整运行。

## 论文依据与改动

REAL-Q 的 Future Work 将系统级显存优化排在明确方向中：自定义省显存反向核、中间激活 offload、融合 Block-GD；论文还指出 Stage-0 全模型反向的峰值显存高于 vanilla GPTQ。速度并非该 Future Work 的首要表述。论文给出的 Qwen3-0.6B 聚合 Fisher 缓存约 56 MB，因此本机上首要优化对象是 Stage-0 反向的临时激活和模块驻留生命周期，而非只压缩这份 Fisher 缓存。[REAL-Q §7、Appendix D.6、Appendix E](https://arxiv.org/html/2609.00049)

候选相对 control 增加两项省显存处理，并启用本实验已有的速度路径：

- `--static_fisher_activation_checkpointing`：在 Stage-0 逐 Transformer block 使用 PyTorch non-reentrant activation checkpointing，反向时重算中间激活。PyTorch 文档将其定义为用计算换显存，并推荐 `use_reentrant=False`。[PyTorch checkpoint 文档](https://docs.pytorch.org/docs/stable/checkpoint.html)
- `--offload_unused_runtime_modules`：静态校准输入捕获后，把 embedding/前置运行模块释放回原设备；最后一层计算真实 KL 前才物化 final norm 和 lm_head。
- 速度路径：Triton fused GPTQ inner kernel 与 `symmetric_union_exact` 预裁剪。

论文还将 GPTQ 的 lazy batch update 描述为解决内存/吞吐瓶颈的实现方式之一；本次 Triton/预裁剪组合的速度收益按整体候选报告，不单独归因到 checkpointing 或按需驻留。[GPTQ](https://arxiv.org/html/2210.17323)

实现位置：`process_args.py` 中两个 opt-in 参数；`gptq_utils/gptq_plus_utils.py` 中 Stage-0 checkpoint 包装与运行模块延迟物化/释放（见参数名 `static_fisher_activation_checkpointing`、`offload_unused_runtime_modules`）。默认不启用。

## 配置与一致性方法

- 模型：`/mnt/d/llamaModels/Qwen3-0.6B`；单卡 RTX 4060 8 GB。
- W4A16，per-row（`w_groupsize=-1`），GPTQ block size 128，act-order 与 QuaRot rotation 开启。
- 校准集：WikiText-2，32 samples × 512 tokens；seed、microbatch、优化器和其余参数在配对两臂完全相同。
- 确定性：`REALQ_DETERMINISTIC_SDPA=1`；三对交错顺序运行，同一组 source revision。
- 每对检查：完整 WikiText-2 评测时记录所有 28 层输出的 SHA256、调用次数和形状；逐层指纹必须完全相同。再比较全部模型 state tensor 的最终哈希、精确 KL 和 PPL。任一不一致即停止后续配对。
- 显存：runner 每秒采集 CUDA allocated/reserved 和 `nvidia-smi`。量化窗口用各自 run.log 的 `-----GPTQPlus Quantization-----` / `-----GPTQPlus Quantization Done-----` 时间戳筛选；全进程均值另行保留。

## 三对完整结果

| 配对 | 量化时间 control → candidate | 量化窗口 `nvidia-smi` 均值 | CUDA allocated 均值下降 | CUDA reserved 均值下降 | 峰值 CUDA allocated / reserved | 28 层与 KL/PPL |
|---|---:|---:|---:|---:|---|---|
| 1 | 458.87 → 291.32 s | 2102.4 → 1645.0 MiB（21.76%） | 15.93% | 9.62% | 4.009/4.264 → 3.648/3.949 GB | 完全相同 |
| 2 | 462.21 → 288.65 s | 2074.0 → 1612.6 MiB（22.25%） | 15.03% | 8.59% | 4.009/4.264 → 3.648/3.949 GB | 完全相同 |
| 3 | 467.27 → 290.42 s | 2086.3 → 1642.6 MiB（21.27%） | 16.55% | 9.87% | 4.009/4.264 → 3.648/3.949 GB | 完全相同 |
| 平均 | 462.78 → 290.13 s（37.31% 更快） | 2087.6 → 1633.4 MiB（21.76% 更低） | 15.84% | 9.37% | candidate 从未高于 control | 3/3 通过 |

三次所有逐层 fingerprint 均为 28/28 相同。三对最终权重 SHA256 均为 `ff3ac5d8b9c97b590f739a07de4fbfae193ed17c9aa683888e45eaa62742ed2f`；KL 均为 `0.2727596163749695`，PPL 均为 `43.71205139160156`。每对 candidate 的峰值 allocated/reserved 均低于 control。

端到端 wall time 平均由 817.89s 降至 643.06s（约 21.4%）。若把评测阶段也纳入平均显存，整进程 `nvidia-smi` 平均只下降 6.76%；CUDA allocated / reserved 全进程均值分别下降 15.84% / 9.37%。因此若“平均内存”用量化窗口的卡占用定义，本次超过 10%；若用全进程设备占用定义，则没有超过 10%。

## 淘汰的分支

尝试把 `--static_fisher_activation_offload` 再叠加到已通过的方案。单层确定性 smoke 的权重 hash 从通过版 `f310…b4775` 变为 `4047…bdc`，并且平均 allocated/reserved 比 checkpoint-only smoke 更高，peak reserved 也更高。该组合不通过一致性门禁，未用于三对全模型候选。

## 产物

- 配对汇总：`outputs/local-qwen3-lazy-runtime-fast-20261004-campaign/summary.json`
- 原始配对数据与逐层输出指纹路径：同目录 `paired_repetitions.json`，以及 `outputs/phase_profile_local-qwen3-lazy-runtime-fast-20261004-r{1,2,3}-{control,triton}/layer_output_fingerprints.json`
- 遥测：同 campaign 的 `gpu_telemetry.csv`；每次的 CUDA 分配采样在对应 `phase_profile_.../cuda_allocator_telemetry_rank0.csv`
- 本地运行目录：`D:\gptq_plus_triton_paper_bench`，未推送远端。

## 限制与下一步

这是一组严格确定性、 reduced-calibration 的工程验证，不是论文 2048-sample 的正式配置复现。当前候选在本机量化窗口同时达到 >10% 平均设备内存、>10% 速度和峰值不高于 base；但在全进程设备均值口径上显存只降 6.8%，应在论文中明确按量化阶段报告，并另列端到端峰值/完整评测口径。下一步应在服务器上按论文 2048 样本配置复验，并至少再做三对；只有仍保持逐层精确一致、峰值不超 base 且量化窗口均值/耗时达标，才将其作为最终工程贡献。
