# REAL-Q 长序列激活卸载与主机副本去重设计

本工作对应 REAL-Q future work 的 memory-efficient backward / activation offloading。REAL-Q 是研究对象，不作为外部方法参考。阶段收益与全流程收益分别记录，不承诺所有配置同一降幅。

## 已有事实

Qwen3-0.6B、2×1024 校准、FSDP CPU-offload/meta-init、labelchunk16：保存张量同步 CPU 卸载的 Stage-0 首对 NVML 峰值为 6469→2763 MiB。实际 Fisher/saliency cache 与全部 13 个缓存/准备文件 SHA 相同；精确区间与同步边界没有回升。该实验未量化，不能据此宣称 KL/PPL 或完整流程已经验证。

候选进程树 aggregate RSS 峰值约从 5.84 增至 12.17 GiB，累计搬运 900 tensors / 11.50 GiB。这是继续优化卸载实现的依据：必须同时报告 GPU、主机存储与搬运，不把数据移动当作总资源消失。累计 payload 不等于同一时刻的活跃内存。

完整 28 层首对已完成：全流程 NVML 6469→3499 MiB（−45.91%），13 个缓存、28 层输出、模型 state 和 KL/PPL 精确相同，完整 allocator/对应区间门槛通过。仍只有一对缩减配置，不是正式规模或稳定速度结论。详见 `memory_speed_optimization_journal.md` O 节及 `outputs/audit-qwen06b-full-2x1024-offload4-exact-r1-20261010/`。

## 外部参考与适用边界

- PyTorch [saved-tensors hooks](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)：本实现的直接 API 依据，保持保存值并按需恢复；允许选择卸载范围。
- Kun Wu et al., [SSDTrain: An Activation Offloading Framework to SSDs for Faster Large Language Model Training](https://arxiv.org/abs/2408.10013v2), 2025 版本，III-B：张量缓存用底层 storage 的唯一标识消除重复存储。这里仅借鉴身份和去重管理，未实现 SSD、异步预取、transfer overlap 或论文的完整调度。
- Vijay Korthikanti et al., [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198), 2022：选择性重算是相关备选。REAL-Q 需要中间层梯度钩子收集 Fisher/saliency，重算会增加部分 forward 调用，但不能由此直接推断梯度统计会被重复累加。两模型真实权重的 CPU 前置检查已保留输出与梯度钩子统计；CUDA、FSDP 和完整缓存仍待实测，暂不作为本轮默认替代。检查范围见主记录 S 节。
- Xuan Peng et al., [Capuchin](https://alchem.usc.edu/portal/static/download/capuchin.pdf), ASPLOS 2020：张量分层管理的系统先例，不能用其训练效果替代本项目的量化实测。

下载的 SSDTrain v2 与 selective recomputation 原文、文本摘取位于 `outputs/literature_memory_20261010/`；保存 SHA 以便复核版本。作者原文的名称优先于聚合站点的拼写。

## 副本去重原型

实现：`utils/saved_tensor_offload.py`；兼容入口 `tools/prototype_saved_tensor_offload.py`；检查：`tests/test_saved_tensor_offload_cache.py`。2026-10-10 已通过真实 CUDA 检查并接入默认关闭的 REAL-Q 入口 `--static_saved_tensor_offload_deduplicate`，需要正数 `--static_saved_tensor_offload_mib`。

1. 为底层 storage 分配唯一标识，避免仅用 `data_ptr` 时被 allocator 地址复用混淆。
2. 缓存键还包含 device、dtype、storage offset 和版本。连续张量仅在同一区域（offset、numel 均相同）时允许 reshape 共享 payload，unpack 恢复逻辑 shape；非连续视图额外按 shape/stride 分开。不复制整个父 storage，也不假设相同 shape 就是相同内容。
3. 保存 CPU payload 的弱引用；多个相同保存值共享一份 CPU 副本。缓存键与注册表不保留原始 GPU tensor/storage，最后一个图引用释放后 CPU 副本也释放。
4. 仍同步复制，维持原来的非叶 CUDA / 4 MiB eligibility。重置统计、异步预取、排除 FSDP 权重视图均不在本原型中，不能混为已有收益。
5. 记录请求次数/字节、实际副本/字节、命中数、当前/峰值主机 payload；这些组件值与进程 RSS、整卡 NVML 分开。

CPU 测试通过同 storage 的相同逻辑视图复用、不同 offset/stride 分离、原位修改保留旧快照、新 storage 身份不同及最后引用释放。真实 CUDA 梯度/repeated-unpack 测试已通过（与无卸载、普通卸载逐位相同），运行时没有另一个模型占用 GPU。相关审计/配置/缓存/峰值测试共 26 项通过。

真实 0.6B 2×1024 Stage-0 首对：900 次请求变为 618 次实际复制，282 次去重命中，累计 D2H payload 12,348,030,976→7,407,140,864 B（约 −40%），host RSS peak 12,854,722,560→10,293,284,864 B（约 −20%）。两边全部缓存 SHA 相同，GPU NVML/allocated/reserved peak 不变，所有精确区间/同步边界无回升。统计结束时 `live_cpu_bytes=0`。目前仅阶段首对，不能称为完整量化或稳定加速；首版普通卸载没有活跃 payload 组件计数，不声称其组件峰值降幅。证据见 `outputs/memory_campaign_qwen06b-2x1024-dedup-stage0-20261010/pair1_audit/` 与主记录 Q 节。

## 接受条件与失败处理

2026-10-10 04:28，去重卸载与关闭卸载的完整 0.6B campaign 已通过前两组（含反序），第三组进行中。两组 NVML 均为 6469→3499 MiB，缓存/模型状态/28层/KL/PPL 精确相同，区间与边界无回升；主机驻留仍高于关闭卸载的对照。该完整对照检验组合方案，不替代上节普通卸载与去重卸载的组件消融。两者共同支持 GPU 门槛下降与主机/搬运成本改善，尚无干净加速证据。

先完成既有卸载与 base 的真实完整配对，再检查原型 CUDA 逐位梯度，最后在相同真实 Stage-0 数据下只对照普通卸载与去重卸载。

去重只有在真实请求产生足够命中、Fisher/saliency cache 完全相同、GPU 对应阶段无回升，且主机 payload/RSS 或搬运有实测减少时才保留。如果命中低或复杂度无可测收益，停止集成；不得把玩具图的 50% 副本减少写成模型收益。速度需要无额外诊断插桩的至少三组交替配对。
