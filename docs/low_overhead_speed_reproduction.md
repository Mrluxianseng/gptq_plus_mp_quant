# 与显存诊断分开的速度复测

实现入口：`tools/run_low_overhead_speed_campaign.py` 与 `tools/realq_low_overhead_speed_entry.py`。截至 2026-10-10 03:49 已通过 CPU 端参数过滤、审计绑定和失败范围检查；**本入口的真实完整模型运行还未完成**。不能依据脚本存在而写入新的速度结果。

先完成同版本的完整显存配对。入口要求提供该配对的 control/candidate 目录与对应 `comparison.json`，拒绝 Stage-0-only、未通过强门槛、引用错实验的审计，以及模型实现发生变化的情况。速度复测仍独立建立 Stage-0 cache，每组交替执行，最终 state SHA 与 KL/PPL 必须分别等于已验证的参照，且所有 rank 一致；不默认小幅质量差异可接受。

2026-10-10补充：只量化prefix后评估28层的报告也必须拒绝，要求 `full_model_quantization_validated=true`。每臂新的Stage-0全部缓存文件SHA必须与自己的参照逐项一致，包括Fisher、saliency及prepared model；缓存hash在同步计时之外进行。工具为checkpoint研究选项保留原block对象并绑定forward，只在实际收集器的grad前向启用非重入重算，不加载诊断用的模块梯度观察器。

测量关闭逐层输出拷贝、NVML/RSS 周期采样、精确区间峰值同步和组件 event profile。量化计时用原完整 `gptq_fwrd` 调用前后的 CUDA 同步墙钟；最终 state SHA 在计时结束后生成。因此主速度字段是每次 `quantization_seconds_slowest_rank`。每个进程仍有启动、检查、写盘与计时后的 state audit；记录的整进程 wall 不称为无插桩端到端延迟，也不能据此承诺包含预计算的加速。

另外输出各rank `static_collection_timing_rank*.json`，收集器同步时间不包括模型checkpoint准备或外部hash审计。汇总同时列出Stage-0收集器、量化调用及两者之和（各阶段取最慢rank）；这个和仍不包括启动、模型准备、最终评估等，不能标成完整端到端时间。checkpoint可能以更多收集器计算换较低显存，这项代价须与显存收益同时呈现。

示例只在两臂全部结束、审计通过且 GPU 无其他工作时执行；当前显存队列运行时不要并发启动：

```bash
python -u tools/run_low_overhead_speed_campaign.py \
  --control-reference outputs/full_pipeline_profile_<tag>-r1-control \
  --candidate-reference outputs/full_pipeline_profile_<tag>-r1-candidate \
  --pair-audit outputs/memory_campaign_<tag>/pair1_audit/comparison.json \
  --tag <new-speed-tag> --pairs 3 --gpu-indices 0
```

外层 stdout 汇总全部 worker 进展；同一份归档在 `outputs/low_overhead_speed_<new-speed-tag>/combined.log`。CPU 构造 manifest 不会启动 GPU。设备启动检查连续三次 utilization 必须不超过3%；这只能发现明显忙碌，不能证明所有系统资源绝对空闲。`--allow-gpu-contention` 是明确记录的诊断覆盖，不能用于干净速度结论。若运行出现状态或质量差异，停止后续组并回到全层诊断。

每次原始时间、配对 speedup 和顺序全部保留；汇总均值及范围，不按结果删除异常值。同步高精度 profile 的旧 wall 只用于排查，不与这个入口的新时间拼接成一个平均。首次单对不能称为稳定加速；三对若方向变化或幅度接近波动，终稿如实报告未观察到可靠速度收益。

如需闭式补偿与梯度更新的组件计时，应另做两侧一致的 event-profile 诊断，独占 rank JSON 并选择互不重叠的 section；父子 inclusive sections 不能直接相加作为“总时间”。本入口未产生这些组件 event 数据，不能从缺失记录填0或从日志时间戳推导 kernel 时间。
