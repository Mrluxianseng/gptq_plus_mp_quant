# REAL-Q Triton 正式配置服务器复测

脚本 `tools/run_triton_paper_benchmark.py` 用于在 Linux/WSL 服务器上复跑完整的 PyTorch control 与 Triton 候选配对。它调用项目现有的 profile/reproducibility runner，不改变待测 kernel；每组先 control、再 candidate，并在每一组后核对完整量化 state 哈希、KL 和 PPL。发现不一致会立即停止后续配对。

## 运行

先从私人仓库检出这个干净分支，再进入项目虚拟环境。模型必须是本地 28 层 Qwen3 Hugging Face 目录，至少含 `config.json` 和权重文件。数据集不会提交到 Git；项目需要在 `datasets/wikitext` 找到本地 Wikitext 文件。支持旧式本地 `wikitext.py` builder，也支持 `datasets/wikitext/wikitext-2-raw-v1/` 下按 split 保存的 Parquet 文件。如果数据集放在别处，先建立链接，例如 `mkdir -p datasets && ln -s /data/wikitext datasets/wikitext`。运行前确认模型、数据和虚拟环境都位于服务器本地：

```bash
git clone --branch codex/triton-paper-benchmark \
  git@github.com:Mrluxianseng/gptq_plus_mp_quant.git gptq_plus_triton_bench
cd gptq_plus_triton_bench
source /path/to/venv/bin/activate
python -c 'import torch, triton; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), triton.__version__)'
nvidia-smi
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python tools/run_triton_paper_benchmark.py \
  --model-path /path/to/Qwen3-0.6B \
  --gpu-indices 0,1,2,3 \
  --pairs 3 \
  --tag-prefix triton-thesis-server-20261002
```

这里会以单机 4-rank `torchrun` 启动每个 arm，control 跑完后再跑 candidate；四张卡同时参与各自的量化 run。脚本要求 PyTorch 恰好看到四张 CUDA 卡，并将 `--gpu-indices` 同时用于设备选择、遥测和竞争进程预检。如果四张卡的物理编号不是 `0,1,2,3`，两处都替换为实际编号。`--tag-prefix` 每轮必须唯一。省略时脚本按本地时间自动生成。默认每个 arm 超时 4 小时；`--python` 可指定虚拟环境解释器。如果任一卡已被其他 CUDA 进程占用，默认停止。

四卡以数据并行分 shard 处理样本，梯度刷新仍合计使用 32 个样本；每个 rank 都会各自加载模型，显存不会跨卡合并。项目既往单卡正式运行在末层全词表 KL 路径上曾遇到约 37 GiB 的分配；四卡后该 batch 沿 rank 切分，但 5090 的 32 GB 单卡显存是否足够仍需由真实 canary 确认。本分支没有集成 KL 投影分块优化，运行时也不会自动降低正式 batch。先确保四张卡都空闲，并保留充足余量；如果首个 control 因 OOM 失败，保存产物和日志后再诊断，不要直接缩小样本或 batch 后把结果当正式配置。

## 固定实验条件

正式规模沿用项目四卡 REAL-Q full-run profile：Qwen3-0.6B W4A16 全 28 层、WikiText-2 校准 256×2048 tokens、seed 1、W4 group size 128、GPTQ block size 128；全局 Hessian batch 128、Hessian accumulation batch 128、全局 loss batch 32、每次梯度刷新 32 个样本，均分到四个 rank。每卡的 Hessian accumulation microbatch 为 32；它只控制前向分块，所有 256 个校准样本仍参与 Hessian 累积。评估使用 WikiText-2 测试集 256×2048 tokens。设置包括 act-order、QuaRot、4 个 Hessian/saliency groups、rank-sharded group quantization、`block_gd`、Adam、`loss_slide_window` 和 REAL-Q `fisher_diag_mse` refresh，学习率为项目 W4A16 正式行的 `5e-7`。Control 使用 PyTorch 列循环，candidate 使用 `triton_fused`。此配置对齐项目既有四卡 campaign 的样本和分块规模，不宣称逐项复现论文表格。脚本固定配置，只开放模型路径、四卡编号、重复数、唯一标签、解释器和 timeout。

## 输出与停止条件

结果会写入 `outputs/<tag-prefix>-campaign/`，并在控制台逐组打印进度。关键文件如下：

- `summary.md`、`summary.json`：每组 control/candidate 耗时、配对加速比中位数/均值/标准差、state/KL/PPL 一致性及有效状态。
- `paired_repetitions.json`：每组原始量化与端到端秒数、完整权重 state SHA-256、507 个张量计数、KL/PPL、峰值 allocated/reserved 显存、逐层量化 GPU 时间、GPTQ 补偿路径及其分项、梯度刷新加 Adam 更新总时间，以及逐层输出指纹。补偿路径汇总包含列内循环、writeback 和外层更新；外层 `delta-W` 与 `block_gd` 梯度校正共用代码区，因此也保留各分项，不把总数误称为纯闭式求解时间。逐层输出以形状、dtype、SHA-256 和前向次数记录，不保存完整激活张量；任一层指纹不同会立即停止后续实验。
- `config.json`、`environment.json`：固定配置、模型 config 哈希、Git commit/工作区状态、Python/PyTorch/CUDA/Triton 版本、GPU/驱动及启动前可见计算进程。
- `campaign.log`：配对驱动的实时完整输出。
- `gpu_telemetry.csv`、`gpu_processes.log`：约每 5 秒的 GPU 利用率、显存、时钟、功耗、温度采样；约每 30 秒记录一次可见 CUDA 计算进程。
- `control-current-source.json`、`candidate-current-source.json` 及 `outputs/phase_profile_<tag>/`：逐次运行命令、源码哈希、status、详细日志、profile 指标、逐层 `layer_output_fingerprints.json` 和 entry wrapper。

启动预检要求选定的虚拟环境可用 CUDA 与 Triton，GPU 上没有可见的竞争 CUDA 计算进程，模型结构为 28 层 Qwen3，且本地数据路径存在。每组比较都会要求 state hash、507 个张量计数、KL、PPL 和全部 28 层输出指纹完全相等；任一输出不一致或记录缺失都会立即停止。整体成功还要求完成指定的全部配对。运行结束请检查 campaign `summary.json` 中 `valid: true`，再将 `summary.md` 和原始 JSON 一并归档。该脚本测量的是 REAL-Q **量化/校准过程**的工程耗时，不测模型推理吞吐。
