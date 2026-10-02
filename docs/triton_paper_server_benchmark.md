# REAL-Q Triton 正式配置服务器复测

脚本 `tools/run_triton_paper_benchmark.py` 用于在 Linux/WSL 服务器上复跑完整的 PyTorch control 与 Triton 候选配对。它调用项目现有的 profile/reproducibility runner，不改变待测 kernel；每组先 control、再 candidate，并在每一组后核对完整量化 state 哈希、KL 和 PPL。发现不一致会立即停止后续配对。

## 运行

在项目虚拟环境中执行。模型必须是本地 Qwen3-0.6B Hugging Face 目录，至少含 `config.json` 和权重文件；评测使用本地 Wikitext-2 数据缓存，运行时设置离线模式：

```bash
cd /path/to/gptq_plus_realq
source .venv/bin/activate
python tools/run_triton_paper_benchmark.py \
  --model-path /path/to/Qwen3-0.6B \
  --pairs 3 \
  --tag-prefix triton-thesis-server-20261002
```

`--tag-prefix` 每轮必须唯一。省略时脚本按本地时间自动生成。默认每个 arm 超时 4 小时，可用 `--timeout-seconds` 调整；`--python` 可指定虚拟环境解释器。脚本预检 CUDA、模型配置和可见的 CUDA 计算进程。如果 GPU 已有计算进程，默认停止；只有确认可以接受竞争时才传入 `--allow-gpu-contention`。

## 固定实验条件

默认正式配置与已验证的 Qwen3-0.6B/W4 全模型实验一致：28 层、Wikitext-2 校准 32×256 token、W4、4 groups、GPTQ block size 128、act-order、`block_gd`、Adam、REAL-Q `fisher_diag_mse` refresh、seed 44；评估使用 Wikitext-2 的 8 个样本、序列长度 1024。Control 使用 PyTorch 列循环，candidate 使用 `triton_fused`。两者使用相同精确 union 搜索路径。配置故意固定，脚本只暴露模型路径、重复数、唯一标签、解释器和 timeout，避免无意把参数扫点混入正式速度对比。

## 输出与停止条件

结果会写入 `outputs/<tag-prefix>-campaign/`，并在控制台逐组打印进度。关键文件如下：

- `summary.md`、`summary.json`：每组 control/candidate 耗时、配对加速比中位数/均值/标准差、state/KL/PPL 一致性及有效状态。
- `paired_repetitions.json`：每组原始量化与端到端秒数、完整权重 state SHA-256、507 个张量计数、KL/PPL、峰值 allocated/reserved 显存、列循环和梯度刷新 profile 分项。
- `config.json`、`environment.json`：固定配置、模型 config 哈希、Git commit/工作区状态、Python/PyTorch/CUDA/Triton 版本、GPU/驱动及启动前可见计算进程。
- `campaign.log`：配对驱动的实时完整输出。
- `gpu_telemetry.csv`、`gpu_processes.log`：约每 5 秒的 GPU 利用率、显存、时钟、功耗、温度采样；约每 30 秒记录一次可见 CUDA 计算进程。
- `control-current-source.json`、`candidate-current-source.json` 及 `outputs/phase_profile_<tag>/`：逐次运行命令、源码哈希、status、详细日志、profile 指标和 entry wrapper。

每组比较都会要求 state hash、张量数、KL 和 PPL 完全相等，且 state tensor 数为 507；失败时配对驱动会立即中止。整体成功还要求完成指定的全部配对。运行结束请检查 `status.json` 中 `valid: true`，再将 `summary.md` 和原始 JSON 一并归档。该脚本测量的是 REAL-Q **量化/校准过程**的工程耗时，不测模型推理吞吐。
