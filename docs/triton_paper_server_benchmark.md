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

这里会以单机 4-rank `torchrun` 启动每个 arm，control 跑完后再跑 candidate；四张卡同时参与各自的量化 run。脚本要求 PyTorch 恰好看到四张 CUDA 卡，并将 `--gpu-indices` 同时用于设备选择、遥测和竞争进程预检。如果四张卡的物理编号不是 `0,1,2,3`，两处都替换为实际编号。`--tag-prefix` 每轮必须唯一。省略时脚本按本地时间自动生成。默认每个 arm 超时 4 小时；`--python` 可指定虚拟环境解释器。如果任一卡已被其他 CUDA 计算进程占用，默认停止。启动时记下 tag 后，实时只需追踪外层日志：

```bash
tail -n 50 -F "logs/${TAG}.log"
```

内层 `run.log` 会保留逐 rank 的完整原始输出；现在它也会实时转发到外层日志。只有外层出现错误、需要查看某个 rank 的上下文时，才直接检查 `outputs/phase_profile_<run-tag>/run.log`。`campaign.log` 是配对调度器输出的归档副本，无需另行追踪。

四卡以数据并行分 shard 处理样本，梯度刷新仍合计使用 32 个样本；每个 rank 都会各自加载模型，显存不会跨卡合并。正式命令对真实 KL 刷新设置每卡 refresh microbatch=2，梯度按样本数累积后仍对完整 32 个全局样本执行一次梯度/Adam 更新，不降低样本数或更新次数。静态 Fisher 预计算另用全局 microbatch=8。两项分块针对不同显存热点；服务器完整复跑前，不宣称全流程已通过，也不宣称候选在每个测量时刻都低于原始 base。

## 固定实验条件

量化设置现已按论文明确给出的 Qwen3-0.6B W4A16 条件对齐：全 28 层、WikiText-2 校准 2048×2048 tokens、对称 per-row 权重量化（`w_groupsize=-1`）、seed 1、Adam 每次使用 32 个校准样本、block size 128、reverse-cosine 层学习率（base ratio 0.01）、非末层表列 LR `3e-4`、末层 LR `1e-5`、小型 Qwen 的 activation-loss clipping `0.95`、QuaRot，以及 4 个 saliency groups。论文明确 W4A16 使用 per-row 且通常用 2048 个 WikiText-2 校准样本；Qwen3-0.6B 的学习率来自论文表 6。全局 Hessian batch、Hessian accumulation batch、Fisher microbatch、KL refresh microbatch、梯度裁剪阈值等论文未完整指定的项目仍是工程实现选择，因此不能声称严格复现论文的全部环境与细节。四卡 RTX 5090 也与论文使用的 RTX Pro 6000 不同。

完整 2048 样本会比此前 256 样本配置显著增加运行时间；四卡按数据并行分 shard，每个 rank 处理校准集的一部分，但总计仍覆盖全部 2048 个样本。Hessian accumulation microbatch、静态 Fisher microbatch、真实 KL refresh microbatch 只限制单次计算分块，不减少校准样本总量或 32 样本梯度更新。由于 `triton_fused` 当前不支持 `group_parallel_quant=rank`，配对两侧都设为 `none`：统计量仍由四卡分布式累计，但每个 rank 量化完整输出行，保证 kernel 是唯一变量；这会增加重复量化计算，耗时只代表此兼容配置。Control 使用 PyTorch 列循环，candidate 使用 `triton_fused`，两臂除 kernel 外使用相同配置和数据顺序。

这是一组对论文明确披露的量化设置进行对齐的 kernel 对照实验，并非完整论文复现：目前自动核对 held-out WikiText-2 KL/PPL 和逐层输出，不运行论文报告的十项 zero-shot 任务；梯度裁剪和若干内部计算 batch 也未由论文完整披露。论文报告的硬件为 RTX Pro 6000，本实验使用服务器的 RTX 5090。

## 输出与停止条件

结果会写入 `outputs/<tag-prefix>-campaign/`，并在控制台逐组打印进度。关键文件如下：

- `summary.md`、`summary.json`：每组 control/candidate 耗时、配对加速比中位数/均值/标准差、state/KL/PPL 一致性及有效状态。
- `paired_repetitions.json`：每组原始量化与端到端秒数、完整权重 state SHA-256、动态记录的 state tensor 数量、KL/PPL、峰值 allocated/reserved 显存、逐层量化 GPU 时间、GPTQ 补偿路径及其分项、梯度刷新加 Adam 更新总时间，以及逐层输出指纹。补偿路径汇总包含列内循环、writeback 和外层更新；外层 `delta-W` 与 `block_gd` 梯度校正共用代码区，因此也保留各分项，不把总数误称为纯闭式求解时间。逐层输出以形状、dtype、SHA-256 和前向次数记录，不保存完整激活张量；任一层指纹不同会立即停止后续实验。
- `config.json`、`environment.json`：固定配置、模型 config 哈希、Git commit/工作区状态、Python/PyTorch/CUDA/Triton 版本、GPU/驱动及启动前可见计算进程。
- `campaign.log`：配对驱动的实时完整输出。
- `gpu_telemetry.csv`、`gpu_processes.log`：每 1 秒采样 GPU 利用率、显存、时钟、功耗和温度，并以 run tag 区分 control/candidate；约每 30 秒记录一次可见 CUDA 计算进程。
- `control-current-source.json`、`candidate-current-source.json` 及 `outputs/phase_profile_<tag>/`：逐次运行命令、源码哈希、status、详细日志、profile 指标、逐层 `layer_output_fingerprints.json` 和 entry wrapper。

启动预检要求选定的虚拟环境可用 CUDA 与 Triton，GPU 上没有可见的竞争 CUDA 计算进程，模型结构为 28 层 Qwen3，且本地数据路径存在。每组比较都会要求两臂的 state hash、实际 state tensor 数量、KL、PPL 和全部 28 层输出指纹完全相等；不再假设 state dict 固定包含 507 个张量。任一输出不一致或记录缺失都会立即停止。整体成功还要求完成指定的全部配对。运行结束请分别检查 campaign `summary.json` 中的 `valid` 和 `candidate_memory_peaks_below_control`；前者只表示量化比较完整且数值一致，后者是候选整轮峰值低于 control 的初筛。由于它仍是 1 秒采样与整轮峰值，不能单独证明每个时刻或每个阶段都低于未优化 base；还需阶段对齐的显存峰值检查，才可判定全流程显存优化成功。再将 `summary.md` 和原始 JSON 一并归档。该脚本测量的是 REAL-Q **量化/校准过程**的工程耗时，不测模型推理吞吐。
