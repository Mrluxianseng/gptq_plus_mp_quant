# REAL-Q Triton 正式配置服务器复测

脚本 `tools/run_triton_paper_benchmark.py` 用于在 Linux/WSL 服务器上复跑完整的 PyTorch control 与 Triton 候选配对。它调用项目现有的 profile/reproducibility runner，不改变待测 kernel；每组先 control、再 candidate，并在每一组后核对完整量化 state 哈希、KL 和 PPL。发现不一致会立即停止后续配对。

## 运行

先从私人仓库检出这个干净分支，再进入项目虚拟环境。模型必须是本地 28 层 Qwen3 Hugging Face 目录，至少含 `config.json` 和权重文件。数据集不会提交到 Git；项目需要在 `datasets/wikitext` 找到本地 Wikitext 文件。如果数据集放在别处，先建立链接，例如 `mkdir -p datasets && ln -s /data/wikitext datasets/wikitext`。运行前确认模型、数据和虚拟环境都位于服务器本地：

```bash
git clone --branch codex/triton-paper-benchmark \
  git@github.com:Mrluxianseng/gptq_plus_mp_quant.git gptq_plus_triton_bench
cd gptq_plus_triton_bench
source /path/to/venv/bin/activate
python -c 'import torch, triton; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), triton.__version__)'
nvidia-smi
CUDA_VISIBLE_DEVICES=0 \
python tools/run_triton_paper_benchmark.py \
  --model-path /path/to/Qwen3-0.6B \
  --gpu-index 0 \
  --pairs 3 \
  --tag-prefix triton-thesis-server-20261002
```

如果选择物理 GPU 1 等其他卡，应同时将 `CUDA_VISIBLE_DEVICES` 和 `--gpu-index` 改成该卡的物理编号。`--tag-prefix` 每轮必须唯一。省略时脚本按本地时间自动生成。默认每个 arm 超时 4 小时；`--python` 可指定虚拟环境解释器。脚本预检 CUDA、模型配置和可见的 CUDA 计算进程。如果 GPU 已有计算进程，默认停止；只有确认可以接受竞争时才传入 `--allow-gpu-contention`。

正式规模有明显显存要求：项目既往正式运行在最后一层全词表 KL 路径上曾尝试约 37 GiB 的单次分配，而本分支没有集成 KL 投影分块优化。不要在 8 GB 卡上启动这组正式配置；建议使用至少 48 GB 空闲显存，最好是 80/96 GB 卡。脚本保留正式样本和 batch 数，不会自动降档。

## 固定实验条件

正式规模对齐项目已完成的 Qwen3-0.6B W4A16 REAL-Q 量化实验：校准 256×2048 tokens、seed 1、W4 权重 group size 128、GPTQ block size 128、Hessian accumulation batch 64；每次梯度刷新使用 32 个样本、反向 batch size 32。评估使用 WikiText-2 测试集 256×2048 tokens。设置包含 act-order、QuaRot、4 个 Hessian/saliency groups、`block_gd`、Adam、`loss_slide_window` 和 REAL-Q `fisher_diag_mse` refresh；学习率使用该正式行的 `5e-7`，旋转与 refresh seed 固定为 0。Control 使用 PyTorch 列循环，candidate 使用 `triton_fused`。这是把内核对照的样本规模和分块规格与项目正式 campaign 对齐；未在该 campaign 冻结的实现参数继续由干净锚点 `origin/zq` 提供，并记录在逐次运行命令及环境清单中，因此不将其宣称为论文表格的逐项复现。脚本固定配置，只开放模型路径、重复数、唯一标签、解释器和 timeout，不做参数扫描。

## 输出与停止条件

结果会写入 `outputs/<tag-prefix>-campaign/`，并在控制台逐组打印进度。关键文件如下：

- `summary.md`、`summary.json`：每组 control/candidate 耗时、配对加速比中位数/均值/标准差、state/KL/PPL 一致性及有效状态。
- `paired_repetitions.json`：每组原始量化与端到端秒数、完整权重 state SHA-256、507 个张量计数、KL/PPL、峰值 allocated/reserved 显存、列循环和梯度刷新 profile 分项。
- `config.json`、`environment.json`：固定配置、模型 config 哈希、Git commit/工作区状态、Python/PyTorch/CUDA/Triton 版本、GPU/驱动及启动前可见计算进程。
- `campaign.log`：配对驱动的实时完整输出。
- `gpu_telemetry.csv`、`gpu_processes.log`：约每 5 秒的 GPU 利用率、显存、时钟、功耗、温度采样；约每 30 秒记录一次可见 CUDA 计算进程。
- `control-current-source.json`、`candidate-current-source.json` 及 `outputs/phase_profile_<tag>/`：逐次运行命令、源码哈希、status、详细日志、profile 指标和 entry wrapper。

启动预检要求选定的虚拟环境可用 CUDA 与 Triton，GPU 上没有可见的竞争 CUDA 计算进程，模型结构为 28 层 Qwen3，且本地数据路径存在。每组比较都会要求 state hash、张量数、KL 和 PPL 完全相等，且 state tensor 数为 507；失败时配对驱动会立即中止。整体成功还要求完成指定的全部配对。运行结束请检查 `status.json` 中 `valid: true`，再将 `summary.md` 和原始 JSON 一并归档。该脚本测量的是 REAL-Q **量化/校准过程**的工程耗时，不测模型推理吞吐。
