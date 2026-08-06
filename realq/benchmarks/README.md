# RealQ Benchmark

本目录是主线 `realq` 包的生成式推理评测子模块。它不复制量化内核，
而是在“量化完成或加载量化 checkpoint 之后”直接消费同一个主线模型和
`realq.config.Config`，因此 Block-GD、MoE 分流、checkpoint、KL/PPL、
lm-eval 和性能开关使用同一份运行时配置。

详细操作见
[REALQ_BENCHMARK_TUTORIAL.md](../../docs/REALQ_BENCHMARK_TUTORIAL.md)，完整
实施与测试流水账见
[REALQ_BENCHMARK_IMPLEMENTATION_20260726.md](../../docs/REALQ_BENCHMARK_IMPLEMENTATION_20260726.md)。

## 支持范围

| 任务 | 生成 | 判分 |
|---|---|---|
| GSM8K | 本进程、同一量化模型 | 数值答案等价，输出 `pass_at_1` |
| MATH-500 | 本进程、同一量化模型 | Math-Verify 0.9.0 符号等价 |
| HumanEval+ | 本进程、同一量化模型 | 导出 EvalPlus `samples.jsonl` |
| LiveCodeBench-lite | 本进程、同一量化模型 | 导出官方 custom evaluator JSON |

HumanEval+ 和 LiveCodeBench-lite 默认只生成候选，不执行候选代码。代码执行
必须进入独立安全沙箱后，通过
`python -m realq.benchmarks.official_eval` 显式确认。这个确认
开关不是沙箱本身。

## 固定评测协议

当前协议名为 `realq_zero_shot_v1`，目标是比较 FP/不同 RealQ 量化设置，
而不是声称复现所有公开 leaderboard 的模型专属 prompt：

- 四个任务均使用固定 zero-shot 英文指令。
- instruct/chat 模型默认应用 tokenizer 自带 chat template。
- Qwen3 默认 `enable_thinking=True`，采样参数为 temperature 0.6、
  top-p 0.95、top-k 20。
- 默认每题生成一次，因此数学任务报告 `pass_at_1`；多样本时额外报告
  `pass_at_n_oracle`，它不是无偏 pass@k。
- LiveCodeBench 默认固定累计 `release_v6`，共 1055 题；禁止
  `release_latest`。
- LiveCodeBench 官方协议通常采用 n=10、temperature 0.2 并报告 pass@1/
  pass@5。若本项目为了 12 小时预算使用 n=1，结果必须注明
  `realq_zero_shot_v1, n=1`，不能直接当作官方 leaderboard 数字。

原始生成参数、数据 SHA256、chat template SHA256、模型和量化设置、依赖
版本、逐任务耗时都会写入 `manifest.json`。

## 流水线位置

```text
加载基础模型或量化 checkpoint
  -> 旋转/RealQ 量化与运行时 A/K/V wrapper
  -> 可选保存量化 checkpoint
  -> 可选 KL/PPL
  -> 释放 torch distributed process group
  -> rank 0 将完整模型分配到可见 GPU
  -> 可选原 lm-eval
  -> 可选四项 reasoning eval
```

`--skip_eval true` 会关闭所有评测；如果只想跳过昂贵的 WikiText KL/PPL，
但仍运行 lm-eval/reasoning，应使用 `--skip_kl_ppl_eval true`。

当前 RealQ 不会无条件保存 checkpoint。只有显式设置
`--save_qmodel_path PATH` 时才保存版本化 checkpoint；之后可用
`--load_qmodel_path PATH` 跳过重新量化并运行评测。

## 主要实现

- `benchmarks/data.py`：四任务别名、本地数据优先、离线 fail-closed、数据
  SHA256、统一样本 schema。
- `benchmarks/generation.py`：chat template、Qwen3 thinking、lm-eval HFLM
  后端、固定 chunk seed、增量 JSONL 和安全恢复 fingerprint。
- `benchmarks/scoring.py`：GSM8K/Math-Verify 判分以及两类官方代码评测输入。
- `benchmarks/runner.py`：数据预检、manifest、逐任务计时和结果汇总。
- `benchmarks/prepare_data.py`：联网宿主机数据固化。LCB 被拆成小型 prompt
  文件和完整 official-test 文件，避免每个模型都解析 4.49 GB 测试数据。
- `benchmarks/official_eval.py`：代码执行安全门。
- `benchmarks/lcb_local_eval.py`：把本地固定 release 注入官方
  LiveCodeBench evaluator，避免离线容器回退到 `release_latest` 或访问网络。

断点恢复不只比较 prompt。每条 generation 还记录模型/checkpoint 路径、
W/A/K/V 配置、生成参数、seed、batch size、样本数和协议的 fingerprint；
这些设置变化后旧答案不会被误复用。

## 结果目录

默认目录为 `${output_dir}/${exp}/reasoning_eval/`：

```text
reasoning_eval/
├── manifest.json
├── CODE_EVALUATION.txt
├── gsm8k/
│   ├── generations.jsonl
│   └── scores.json
├── math_500/
│   ├── generations.jsonl
│   └── scores.json
├── humaneval_plus/
│   ├── generations.jsonl
│   ├── evalplus_samples.jsonl
│   └── scores.json
└── livecodebench_lite/
    ├── generations.jsonl
    ├── livecodebench_custom_outputs.json
    └── scores.json
```

## 模型要求

评测入口不绑定某个固定 model-zoo 清单；`cfg.model` 或已加载量化
checkpoint 对应的 tokenizer 必须可从本地离线读取。chat/instruct 模型应
提供有效 chat template；不要仅根据目录名是否带 `-Instruct` 判断模型类型。
