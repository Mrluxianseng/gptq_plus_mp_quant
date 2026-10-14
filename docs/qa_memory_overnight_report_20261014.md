# QA 显存研究当晚报告（2026-10-14 08:00汇报定稿）

今晚已在本地真实 Qwen3-0.6B/1.7B 组件中保持配置 batch16、BF16、SDPA、完整词表、原 KV 与评分计算，显著降低 NVML 采样到的物理显存峰。正式服务器完整 QA 的最终峰还没有验证，不能把服务器此前约19GB直接替换成下面的本地结果。

## 方法

1. 在 continuation 对应的行上执行原 log_softmax，缩短评分中间存储的寿命；模型仍生成完整 logits。
2. 在 dispatch 之前选择逐层 CPU 权重放置，使用官方 DynamicCache 卸载，保持原 KV 计算与 Parameter 身份。
3. 保留 LM-head 原整块 GPU Linear/GEMM，只通过私有 MemPool 将输出存储映射到 pinned host memory。
4. 在每批最后消费者之后确认 active=0，删除该批私有池、确认 hostlive=0，再按需建立下一批的池。
5. 用16MiB CPU暂存流式计算完整字节指纹，减少验证自身的主机内存开销；这一步不改变 QA 的 GPU 计算。

这套方案将一部分存储需求转移到主机内存，并增加传输和分配开销。CUDA allocated/reserved 包含 host-backed CUDA 张量的逻辑存储，因此必须同时报告 native/private/callback/NVML 与主机代价。

## 已通过的真实模型组件

下表的长输入对照已包含 rows、分层权重及官方 KV 卸载，候选再叠加整块输出映射和逐批释放；它没有原始全驻留长输入对照。

| 范围 | 模型 | NVML C→N (MiB) | native allocated C→N (B) | native reserved C→N (B) |
|---|---|---:|---:|---:|
| 长输入，实际1235/1016/845，64请求 | 0.6B | 7047→1415 | 6446786048→6446786048 | 7251951616→7124025344 |
| 同上 | 1.7B | 7907→2363 | 6798093824→6798093824 | 8346664960→8218738688 |
| 短输入，最长131-token；自然批16/16/16/1，65请求 | 0.6B | 1245→637 | 970141696→970141696 | 1168113664→1124073472 |
| 同上 | 1.7B | 1687→1079 | 1286125568→1286125568 | 1631584256→1587544064 |

所有组件使用synthetic-token请求；自然尾批最后一次输入为1×3，配置batch仍为16。表中的长输入与短输入是不同工作量，不能用短输入峰估计长输入峰。

长组件两模型共724真实区间/2896独立start/end点，完整311 state、每次28层非空输出、full logits、全部64分数与greedy、输入/方法/源/模型/库SHA及私有池账完整独立通过。0.6B两个局部 allocated 区间峰微升（786432/721408 B）已保留诊断，每点无回升，三global门槛通过；1.7B局部也无回升。NVML采样不证明连续每刻的峰。

每批 callback pinned 存储峰约6.006GB（6006243328 B）。长组件单worker采样RSS：0.6B约8.77→13.67GB、1.7B约12.30→17.14GB；包含载入和摘要，并非总主机/PSS/连续最大或80G服务器容量证明。密集观测评分墙钟分别21.56→31.05s、30.81→42.84s，只作成本诊断，不作稳定速度或端到端结论。

07:01新增只读Linux进程树监测工具，分别保存RSS/PSS、共享/私有页、锁页与swap、PID启动时间/boot身份、系统和可见cgroup限额；真实32MiB共享页与父进程退出后worker继续运行的CPU测试通过。它没有重跑QA，也不能补造此前QA PSS。监测字段不可读明确保留，锁页字段不替代CUDA callback pinned计数。服务器恢复时可使用 `tools/run_qa_host_memory_observer_v2.py --pid <实际实验PID> --output <新的JSONL路径>`，保持原正式launcher默认不变。

07:12只读预算工具已核对两模型真实三次输出/head/KV物件：实际B16/L1235的full logits为6004510720B，全部28层逻辑K+V为2266234880B。它们不是峰值，也不是全都在CPU上的对象量。假设L2048时full logits约9.96GB，但该长度没有执行GPU或正式QA；不能用校准seq2048或模型配置上限代替正式请求的实际长度分布。

原 native dispatch 对照与完整组合的短包装fixture也已通过：0.6B NVML3359→637、1.7B5553→1079MiB。两模型共1186区间/4744独立点，535 state/28Had/full logits/65分数、真实尾批、实际方法与完整峰累计均通过。但基权重未旋转，本机使用torch_fallback；这不证明正确REAL-Q量化模型或服务器FHT质量。

## 最终独立复核

07:25在CUDA隐藏的fresh CPU进程中，8组已完成配对工件重新通过完整独立审查：长输入、自然尾批、原native对组合、实际官方控制头hook各两模型，共4058区间/16232独立边界点。全部数值与源/模型/库/方法/KV/私有池统计核对，三global门槛通过；长0.6B两个局部微升仍保留。它们是已有组件工件的复核，具有方法继承关系，没有新GPU重放。首版单池long的356/1424另列为拒绝，仍仅global reserved失败，不混入通过数。

[最终复核汇总](../outputs/qa-night-final-independent-20261014-v1/status.json)保留各新独立审查、日志及原parent status的SHA。另有[CPU来源收据](../outputs/qa-night-final-cpu-receipts-20261014-v1.json)核对9个准备根69条既有SHA全部未变，旧CLI失败状态未改。它不等同完整正式准入或新CPU/GPU测试。

## 拒绝的方法与保留的失败

- 关闭 KV 改变62/64分数及1个greedy，拒绝且未重跑。
- BF16 LM-head 256行切块改变首批完整logits字节，即使最终分数一致也拒绝。
- 长输入首版把一个映射池保留到全部三批结束，数值完全一致但全局 reserved 增加96MiB，拒绝。之后的逐批最后消费者释放是独立的生命周期改进，原失败未改判。
- 构建 anchor、观察源归因及测试缩进等接入失败原证据均保留；不把它们当算法资源收益。

## 正式接入状态与下一步

完整任务/输入/层/logits/分数/state观察器、自然尾批、各可见设备 exact 区间和原process峰累计已有CPU及本地单卡组件证据。06:35两模型的真实官方CUDA AlignDevicesHook控制头观察也已完整独立通过，原forward/Parameter/hook身份保留；各593区间/2372逐点，全部535 state/full logits/65分数精确。0.6B原parent False是预期函数摘要构造错误，CPU复现并在新v2独立审查中修正，原False保留，GPU没有重跑。这些仍不表示正式多卡准入完成。

服务器恢复后先只读核对实际源码、运行时、库、缓存和中断工件，只续缺失部分。随后验证正确旋转量化535状态及实际FHT/allocator后端，再做完整十QA配对、所有visible devices区间与每点、每设备NVML和原全流程峰累计验收。默认正式脚本目前没有启用该原型。

06:47已核对当前源码：量化和KL/PPL profiler在进入QA之前正常结束，未发现当前路径已重叠reset。新增CPU独占计数器准入5项通过，遇活动native/QA profiler则拒绝，保留原异常与原process峰；源码契约3项通过，核对正常交接、完整十任务、原batch参数与无limit。它们是接口准备，不能证明任意第三方reset或真实多卡。接入时继续明确所有权，累计footer不能代替区间完整性。

本地synthetic组件、实际最长1235-token的范围不能外推正式全部请求、2048×2048量化全流程或任意多卡/主机容量。QA之后仍需重新比较Stage0、量化、KL/PPL、QA与全流程峰，不能称QA是最后唯一瓶颈。

详细工件与方法见 [映射输出方法记录](qa_mapped_output_storage_20261014.md)、[本地夜间记录](qa_memory_local_night_20261014.md)、[正式恢复验收链](qa_layerwise_formal_validation_plan_20261014.md)。理论依据为 [CUDA12.8.1映射内存API](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/group__CUDART__MEMORY.html)、[PyTorch2.11 MemPool](https://docs.pytorch.org/docs/2.11/notes/cuda.html)、[NVIDIA WSL约束](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。

定稿时间（北京时间）：2026-10-14T07:58:48.956992+08:00。当晚本地研究与最终工件复核完成，研究CPU/GPU均已退出。服务器保持关闭；正式QA原型默认未启用。源码、原失败及独立审查文件继续保留。
