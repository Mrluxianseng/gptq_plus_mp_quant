# REAL-Q存储管理方法：服务器正式配置启动说明

入口：[scripts/run_storage_method_formal.sh](../scripts/run_storage_method_formal.sh)。控制器：[tools/run_storage_method_formal.py](../tools/run_storage_method_formal.py)。配置参考原[scripts/formal_realq_compare.sh](../scripts/formal_realq_compare.sh)和仓库论文PDF表6；原脚本、量化源码及历史结果没有修改。

## 方法与比较对象

统一方法按照REAL-Q真实消费者顺序和存储别名约束，安排五个机制：两工作区求逆、已消费H存储接管保留因子、down_proj稳定写回、末组私有H死亡点释放、整层量化hook之后提前释放回放不再需要的统计。求逆仍在GPU以FP32计算，模型原权重BF16；没有改变数据、Adam、量化位宽或梯度更新。

默认一对/模型，顺序为0.6B control→candidate，随后1.7B control→candidate。control已含前期Stage0 block checkpoint、CE256及输入卸载；candidate在相同基础上加入上述方法。该比较测量新增方法的增量效果，不是全部工程相对早期基线的累计收益。脚本不跑warm_adam、不扫学习率，也不混入finite、all-linear写回、大小policy或compact池。

两臂各自fresh Stage0预计算→自己的完整缓存→28层量化→完整WikiText-2评估及默认QA任务。缓存不复用本机pilot结果，Stage2禁止偷偷重算预计算。正式参考值由服务器同配置control生成，不能拿cal2×1024的507state hash/KL/PPL当正式结果目标。

## 固定正式配置

| 设置 | 默认值 |
| --- | --- |
| 模型 | Qwen3-0.6B、Qwen3-1.7B，均28层 |
| 校准 | WikiText-2，2048条×2048 tokens，seed=1 |
| 量化 | W4A16/K16/V16，per-row，w_groupsize=-1，w_clip、act_order、QuaRot |
| 分组/列块 | num_groups=4，blocksize=128，alpha=0 |
| 优化器 | Adam，backward_samples=32，pre_gd_steps=0，block_gd |
| 非末层学习率 | 0.6B=3e-4；1.7B=5e-4（本地论文表6） |
| 末层学习率 | 两模型均1e-5 |
| 调度 | cosine，base_ratio=0.01；grad_clip/final_layer_grad_clip=1.0 |
| 损失 | fisher_diag_mse，loss_slide_window，global_loss，dp_global_shuffle |
| 辅助阈值 | a_loss_ratio=0.95，local_backward_chunk，saliency P99 |
| GPU | DEVICE=0,1,2,3；Stage0 FSDP，Stage2数据并行 |
| 分组执行 | **group_parallel_quant=none**，gptq_inner_kernel=torch |
| batch | bsz/global_loss_bsz/hessian_accum_bsz=8，backward_bsz/末层=4，末层stats=4 |
| logits/CE | KL和评估logit chunk=16，eval_ce_chunk_tokens=256 |
| Fisher标签 | static_fisher_label_chunk_tokens=0，保持旧正式脚本未分块策略 |
| 评估 | eval_seq_len=2048，eval_samples=0（全测试集），默认QA开启，batch16 |
| 分配器 | 显式backend:native，和本轮存储方法观察器保持一致 |

**与旧服务器启动路径的明确区别：**旧sweep默认rank分组并行及expandable_segments；本方法只接入顺序fasterquant路径，故两臂均明确none和native，不能保留rank并行却宣称方法启用了。多卡数据并行仍保留。Stage0采用已复审的FSDP预计算入口。DP shuffle及rank数影响刷新样本/数值，改变GPU数量须新建缓存并以同GPU数量control作比较。这里是正式工作量的服务器验证入口，不是已证明论文表格逐位复现。

## 服务器使用

把当前仓库源码整体同步到服务器，包括新增tools/helpers和scripts；不需要同步历史outputs、logs或本机venv。使用原来能运行REAL-Q的Python环境，额外沿用profile runner依赖psutil和nvidia-smi。数据和默认QA任务需要预先缓存，因为既有profile runner明确以offline环境运行。

```bash
cd /path/to/gptq_plus_realq
export PYTHON=/path/to/venv/bin/python
export MODEL_ROOT=/path/to/modelzoo/Qwen3
export DEVICE=0,1,2,3

# 只生成entry、完整命令及配置计划，不加载模型、不启动GPU。
bash scripts/run_storage_method_formal.sh --dry-run

# 正式运行四个串行臂。每次自动生成新目录，不覆盖旧实验。
bash scripts/run_storage_method_formal.sh --run
```

模型不在同一父目录时，分别设置MODEL06和MODEL17。MODEL_ROOT默认沿用此前服务器路径`/minimax-avatar-new/zhangqian/realq/gptq_plus/modelzoo/Qwen3`，实际启动前按你的服务器位置修改。

```bash
MODEL06=/models/Qwen3-0.6B MODEL17=/models/Qwen3-1.7B \
  DEVICE=0,1,2,3 bash scripts/run_storage_method_formal.sh --run

# 先只跑0.6B；不自动启动1.7B。
bash scripts/run_storage_method_formal.sh --run --only qwen06b

# 初次显存验证只保留全量WikiText-2 KL/PPL，暂不跑QA：明确记录该差异。
bash scripts/run_storage_method_formal.sh --run --skip-qa-eval

# 若要完整重复三对：各模型按C/N、N/C、C/N运行，失败即停止后续。
bash scripts/run_storage_method_formal.sh --run --repeats 3
```

默认每个阶段超时86400秒，可用`--timeout-seconds`调整。`--nsamples/--seq-len/--bsz/--backward-bsz/--seed/--lr06/--lr17`可显式覆盖；覆盖后plan会保存实际值，nsamples或seq_len偏离2048时不标作正式2048×2048工作量。学习率覆盖是用户配置，不存在自动扫描。

运行目录可用RUN_DIR指定，但必须尚不存在；OUTPUT_ROOT只控制计划、控制日志和独立审查目录。既有full-pipeline runner的每臂缓存/工件和PTQ日志仍放在仓库`outputs/full_pipeline_profile_<唯一tag>/`及`outputs/<模型>/<唯一exp>/`，因此该磁盘也必须有足够空间。所有路径在plan.json和控制台列明。

## 工件与准入

- `plan.json`：四臂/所有重复的命令、实际配置、模型config SHA、entry SHA及环境；dry-run不是GPU成功。
- `source_bindings.json`与`entry.py`：本次生成入口和源码绑定，运行中不编辑。
- `<tag>.log`及每臂`stage0.log/stage2.log`：控制日志和完整worker日志。
- 每臂`static_cache/`：独立fresh预计算缓存；审查完整目录所有文件的字节SHA，包含Fisher，正式多卡文件数量按实际清单记录，不硬套单卡13文件。
- `audit_<model>_pN.json`：CPU master所属rank0的完整507state、28层非空输出及KL/PPL；所有rank的state布局/身份、112inverse/reuse及28write/dead/stats、源/缓存/命令/环境/设备、精确区间峰和独立逐point/global验收。其他rank的meta布局记录不称为权重字节哈希。
- `status.json`：仅完整执行并通过所有pair后validTrue；GPU异常、数值/来源/覆盖/遥测/全局峰失败均保留证据并停止。

全局capacity主准入要求full-process allocated/reserved和各物理GPU的NVML采样峰不高于同正式control，且完整质量和方法审查通过。allocated/reserved为rank进程峰的最大值，不是多卡显存求和。区间与start/end局部回升保留为诊断，不单独否决用户11:15更新后的全局容量目标；旧strict比较结果同样完整保存。NVML采样不是连续证明，密集插桩时间不能称速度结论。

截至脚本交付，本机只完成CPU/source接入和命令计划验证，没有运行服务器2048×2048或四卡GPU实验。已有双模型各三对pilot收益不是新正式配置的保证；正式结果应以服务器生成的完整审查为准。

## 2026-10-13：双卡CPU master审查修复

用户提供的服务器日志表明，首个0.6B control在量化最后一层后，rank1的post-quant state哈希调用`.cpu()`遇到meta张量而退出；torchrun随后终止rank0。这是正式验收入口错误，不是日志中的OOM，也不能计作完整正式pair。`quant_stop_layer=27`采用0起始编号，包含第28层；进度条27/28不代表缺少最后一层，实际完整覆盖仍须工件审查。

原CPU master路径有意只让rank0保留完整CPU模型并做KL/PPL评估，其他rank按层释放至meta以节省内存。新增`storage_method_distributed_audit.py`只在完整模型所属rank上计算全部state的原格式字节SHA；所有rank记录真实state名称/形状/dtype布局SHA、meta数量、rank/world_size及所有权。非所属rank不复制meta，不广播/伪造rank0的数值哈希或28层输出。若完整模型rank含meta、少于507项、缺少任何rank记录或rank0有层没有真实输出，拒绝验收。普通非CPU master模式仍要求每rank完整state和评估结果一致。

正式控制器改用独立`run_storage_method_pipeline.py`，按上述所有权读实际工件；完整显存峰、所有rank覆盖、fresh缓存和每区间/每point审查继续保留。旧`realq_checkpoint_research_entry.py`、`run_full_pipeline_memory_profile.py`及研究helper均不修改，旧失败目录也不改。此修复不改量化数学、超参数、计时边界、求逆精度或GPU/CPU存储策略。

更新服务器后重新启动新目录：

```bash
git pull --ff-only https://github.com/Mrluxianseng/gptq_plus_mp_quant.git realq-warmadam
MODEL_ROOT=/openbayes/home/llmModels DEVICE=0,1 \
  bash scripts/run_storage_method_formal.sh --run
```

旧流程未设置`save_qmodel_path`，Stage0准备checkpoint不是已量化模型；进程退出后无法仅补一个哈希就恢复完整量化结果。保留旧outputs，新启动会重新执行两臂fresh完整流程。修复的CPU双进程回归替换CUDA传输为Gloo CPU传输，调用真实层释放函数并执行实际生成的哈希/评估包装；它不等于服务器双卡CUDA正式实验成功。
