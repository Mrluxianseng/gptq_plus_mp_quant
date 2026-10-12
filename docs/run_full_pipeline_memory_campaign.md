# REAL-Q 全流程显存实验入口

本入口从全新 Stage-0 缓存开始，随后量化全部 Transformer 层并评估。连续日志汇总加载、预计算、模块量化、最终 KL/PPL、审计和退出状态。默认三组配对，顺序为 control→candidate、candidate→control、control→candidate；不会删除异常值。诊断包含同步区间测量和逐层输出拷贝，不用于声称运行加速。

本地单卡已验证可移植 entry 的 Stage-0 和审计路径；完整可移植入口的重复配对正在验证。四卡配置通过参数/全局 batch 可整除检查，**还没有在当前版本的四卡硬件上实际运行**。旧 Triton 多卡结果不能替代本版的分阶段测试。先完成本地完整配对再提交服务器正式验证，避免以配置检查冒充部署通过。

## 创建 manifest

进入项目根目录并激活含 PyTorch/Transformers 的虚拟环境。`--output-dir` 和后续 `--tag` 必须是新名字，已有结果不会覆盖。

本地 WSL 缩减配置：

```bash
python tools/prepare_realq_memory_manifests.py \
  --model-path /mnt/d/llamaModels/Qwen3-0.6B \
  --output-dir outputs/memory-06b-pilot-manifests \
  --preset pilot --world-size 1 --activation-deduplicate
```

Pilot 默认 2×1024 校准，2×1024 held-out 评估，全 28 层，seed44、不开旋转；这不对齐论文质量表，只用于工程正确性/峰值验证。1.7B 本地初测用同一入口替换模型，并明确 `--seq-len 512`。仅当要测已验证的 Hinv 组合时另加 `--candidate-hinv-downproj-offload`，不要把组合结果记为单独激活卸载的效果。

服务器论文资源规模配置：

```bash
python tools/prepare_realq_memory_manifests.py \
  --model-path /openbayes/home/llmModels/Qwen3-0.6B \
  --output-dir outputs/memory-06b-paper-manifests \
  --preset paper --world-size 4 --activation-deduplicate
```

Paper preset：W4A16 per-row（group size−1）、WikiText-2 2048 个校准样本×2048 tokens、blocksize128、4 saliency groups、32 个 global gradient samples、QuaRot、seed1。0.6B 非末层/末层学习率 3e−4/1e−5；1.7B 5e−4/1e−5。评估默认 `--eval_samples 0`，使用完整测试集的 2048-token 窗口。默认 FSDP CPU-offload/meta-init、labelchunk16、KLchunk16、每 rank 微批量1和 gradient clipping 等为明确固定的共享系统条件，部分原论文未披露；**不声称精确重建论文表格**。当前 legacy 量化调度的差异见 `REALQ_PAPER_PROTOCOL.md`。

生成后先检查 `protocol.json`。生成使用当前环境的 Python 路径和当前项目根路径，因此应在目标服务器上重新生成；不复制本地 WSL manifest 的绝对路径。

Control 已共享 FSDP CPU-offload/meta-init 和 KL/label chunk16。Candidate 在此基础上仅给 Stage-0 增加 saved-tensor CPU offload4 MiB 和（显式指定时）副本去重；Hinv 是可选候选组件。结果必须称为相对这组共享系统对照的收益，不能将整个差值归因于未参与该配对的组件。

## 启动三组完整配对

服务器示例（本地改 GPU 列表为0、manifest目录为本地目录）：

```bash
mkdir -p logs
TAG=memory-qwen06b-paper-$(date +%Y%m%d-%H%M%S)
MANIFESTS=outputs/memory-06b-paper-manifests
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=8 \
  python -u tools/run_full_pipeline_memory_campaign.py \
  --control-stage0-manifest "$MANIFESTS/control-stage0.json" \
  --candidate-stage0-manifest "$MANIFESTS/candidate-stage0.json" \
  --control-stage2-manifest "$MANIFESTS/control-stage2.json" \
  --candidate-stage2-manifest "$MANIFESTS/candidate-stage2.json" \
  --tag "$TAG" --pairs 3 --gpu-indices 0,1,2,3 \
  --timeout-seconds 14400 \
  > "logs/${TAG}.log" 2>&1 < /dev/null &
echo $! | tee "logs/${TAG}.pid"
```

`NCCL_SHM_DISABLE=1` 只在目标环境存在已定位的 NCCL shared-memory 问题时使用，且双方一致；不是默认加速开关。GPU index 是物理监控列表，必须与可见 GPU 对齐。环境和软件版本分别保存在 manifest 与每 rank 的软件 JSON 中。

## 始终查看同一份实时日志

```bash
tail -n 50 -F "logs/${TAG}.log"
```

外层现在包含每个工作进程日志，不需正常运行时切换追踪另一个文件。相同连续内容也保存在 `outputs/memory_campaign_${TAG}/combined.log` 供归档；各臂原始日志用于追查，路径见 `active_run.json`。审计大 checkpoint 哈希时也会报告进展。`Ctrl+C` 退出 tail 只停止查看，不停止模型实验。

另开终端后先重新设置实际 TAG，不依赖上一终端的临时变量。退出阶段先检查：

```bash
cat "outputs/memory_campaign_${TAG}/status.json"
cat "outputs/memory_campaign_${TAG}/paired_repetitions.json"
```

每臂返回0不代表 campaign 已通过；必须检查 `valid=true` 和请求配对数全部完成。输出/质量不一致或任一已测阶段回升时该 campaign 自动停止并保存错误，不能继续跑第三组掩盖失败。

## 生成论文结果表和峰值位置

```bash
python tools/summarize_memory_campaign.py \
  --campaign "outputs/memory_campaign_${TAG}" \
  --output-dir "outputs/memory_summary_${TAG}"
```

输出 `paired_values.csv`（每次实验含 KL/PPL）、`summary.json/md`（每次 peak 的配对均值/范围）、`peak_locations.json`（每个 rank 的最大已测区间及是否对应完整进程峰值）、`offload_metrics.json`（每 rank 的实际搬运/去重/活跃副本计数）。没有计数器的旧实现记为未知，不记为0。这些均值是**多次运行峰值的平均**，不是时间平均占用。模型 state/所有层指纹与缓存 SHA 见 `pair*_audit`；实际 rank 文件独占保存；device NVML 和进程 allocator/host RSS 的原始遥测也不会共用同一文件。

主机 RSS 是进程树求和，可能重复计数共享页；必须报告这项代价，不能把 GPU 卸载称为总内存减少。NVML 是整卡100 ms采样，allocator interval maxima 是同步精确高水位；比较已测对应区间不等于数学上证明任意连续时刻候选都更低。完整进程含加载、Stage-0、量化、最终评估，不能只引用量化 profiler 的旧字段。

正式加速应另外进行无逐层拷贝/无精确区间同步的至少三组交替配对；本入口只构建显存与精确性证据，不给出未经测量的速度保证。
