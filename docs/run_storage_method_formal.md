# REAL-Q存储管理方法：服务器正式配置启动说明

入口：[scripts/run_storage_method_formal.sh](../scripts/run_storage_method_formal.sh)。控制器：[tools/run_storage_method_formal.py](../tools/run_storage_method_formal.py)。配置参考原[scripts/formal_realq_compare.sh](../scripts/formal_realq_compare.sh)和仓库论文PDF表6；原脚本、量化源码及历史结果没有修改。

## 方法与比较对象

统一方法按照REAL-Q真实消费者顺序和存储别名约束，安排五个机制：两工作区求逆、已消费H存储接管保留因子、down_proj稳定写回、末组私有H死亡点释放、整层量化hook之后提前释放回放不再需要的统计。求逆仍在GPU以FP32计算，模型原权重BF16；没有改变数据、Adam、量化位宽或梯度更新。

默认一对/模型，顺序为0.6B control→candidate，随后1.7B control→candidate。control已含前期Stage0 block checkpoint、CE256及输入卸载；candidate在相同基础上加入上述方法。该比较测量新增方法的增量效果，不是全部工程相对早期基线的累计收益。脚本不跑warm_adam、不扫学习率，也不混入finite、all-linear写回、大小policy或compact池。

默认`--cache-mode auto`：启动时在仓库`cache/`、`outputs/`和额外指定目录中查找旧的静态saliency/Fisher缓存。只有模型文件身份、数据、样本数、序列长度、seed、旋转、分组、loss batch、缓存schema和world size都匹配，且所有rank分片的层数/模块/形状/dtype/有限值与SHA校验通过，才会复用。四卡旧缓存只有rank0–rank2完整、rank3为空时会被拒绝并重算；双卡启动也不会误用四卡分片。

若完整rank缓存存在但没有完整旋转模型checkpoint，控制器会引用旧分片，并查找其他目录中匹配的prepared模型；没有匹配模型才新建一份。缓存命中保护会让该阶段在缓存缺失时直接失败，避免悄悄重算Fisher。若checkpoint和分片均完整，则跳过Stage0。auto模式默认cache-storage=shared，control和candidate及重复臂通过硬链接读取同一份经过验证的大文件（跨文件系统用文件软链接，不退回复制）；各实验只独立保存少量路径元数据、日志、配置和验收工件。每个正式比较的测量范围明确标记为stage2_cached，Stage0准备单独记入总日志和计划，不计入配对显存峰值。两臂仍各自运行完整28层量化和KL/PPL/QA评估，比较器不会伪造Stage0状态或峰值。

如需评估从空缓存开始的完整流程，可使用`--cache-mode fresh`，此时两臂各自重新运行Stage0，并按`full_pipeline`范围比较。正式参考值由服务器同配置control生成，不能拿cal2×1024的507state hash/KL/PPL当正式结果目标。

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

# 自动检查匹配缓存：完整命中则复用；缺失、不完整或配置不匹配时重算一次。
# 结果仍按control/candidate串行执行，所有输出进入一个nohup总日志。
bash scripts/run_storage_method_formal.sh --run

# 如需每个臂都重算Stage0，比较完整Stage0到评估的流程：
bash scripts/run_storage_method_formal.sh --run --cache-mode fresh
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

默认每个阶段超时86400秒，可用`--timeout-seconds`调整。`CACHE_ROOT`可增加旧缓存搜索目录（仓库内的`cache/`和`outputs/`仍会搜索）；也可重复传`--cache-search-root /some/cache/path`。`--nsamples/--seq-len/--bsz/--backward-bsz/--seed/--lr06/--lr17`可显式覆盖；覆盖后plan会保存实际值，nsamples或seq_len偏离2048时不标作正式2048×2048工作量。学习率覆盖是用户配置，不存在自动扫描。

运行目录可用RUN_DIR指定，但必须尚不存在；OUTPUT_ROOT只控制计划、控制日志和独立审查目录。既有full-pipeline runner的每臂缓存/工件和PTQ日志仍放在仓库`outputs/full_pipeline_profile_<唯一tag>/`及`outputs/<模型>/<唯一exp>/`，因此该磁盘也必须有足够空间。所有路径在plan.json和控制台列明。

## 工件与准入

- `plan.json`：四臂/所有重复的命令、实际配置、模型config SHA、entry SHA及环境；dry-run不是GPU成功。
- `source_bindings.json`与`entry.py`：本次生成入口和源码绑定，运行中不编辑。
- `<tag>.log`及每臂`stage0.log/stage2.log`：控制日志和完整worker日志。
- 每臂`static_cache/`：默认只引用共享来源文件，Fisher和prepared权重不增加重复的数据块；路径别名元数据单独保存，不写回共享来源。源文件使用前后及完整目录的字节SHA仍核对，包含Fisher，正式多卡文件数量按实际清单记录，不硬套单卡13文件。`shared_cache_reference.json`保存实际链接类型、源/目标和新增大文件字节数。显式`--cache-storage copy`才采用旧的独立复制策略。
- `audit_<model>_pN.json`：CPU master所属rank0的完整507state、28层非空输出及KL/PPL；所有rank的state布局/身份、112inverse/reuse及28write/dead/stats、源/缓存/命令/环境/设备、精确区间峰和独立逐point/global验收。其他rank的meta布局记录不称为权重字节哈希。
- `status.json`：仅完整执行并通过所有pair后validTrue；GPU异常、数值/来源/覆盖/遥测/全局峰失败均保留证据并停止。

全局capacity主准入要求full-process allocated/reserved和各物理GPU的NVML采样峰不高于同正式control，且完整质量和方法审查通过。allocated/reserved为rank进程峰的最大值，不是多卡显存求和。区间与start/end局部回升保留为诊断，不单独否决用户11:15更新后的全局容量目标；旧strict比较结果同样完整保存。NVML采样不是连续证明，密集插桩时间不能称速度结论。

截至脚本交付，本机只完成CPU/source接入和命令计划验证，没有运行服务器2048×2048或四卡GPU实验。auto复用实验不测量Stage0，因此只可报告`stage2_cached`显存范围；要评估空缓存完整工作流时显式使用fresh。已有双模型各三对pilot收益不是新正式配置的保证；正式结果应以服务器生成的完整审查为准。

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

## 一个总日志查看所有四臂

新启动计划记录`aggregate_worker_stdout=true`。Stage0/Stage2的完整stdout和stderr先保留原阶段文件，再实时转发至每臂控制日志，最后转发至正式控制器stdout。于是此前nohup命令的`$LOG`就包含四臂自动切换、Fisher进度、量化层/module/loss、cleanup、评估和异常；只需`tail -f "$LOG"`。转发将tqdm的回车转换为换行以便tail阅读，原阶段文件字节保留；NVML采样CSV、区间JSONL等原始数据继续单独保存，不把每100ms原始采样灌入进度总日志。新增转发无GPU操作，设置PYTHONUNBUFFERED=1；密集profile仍不是低扰动速度实验。

如果实验已用旧版本启动，不要中途pull改动绑定源码。只fetch并将独立reader取到/tmp，追加一个CPU日志跟踪进程；不会重启实验或额外启动GPU。以下自动选最新正式run目录，可手动将RUN_DIR改为控制台Plan对应目录：

```bash
cd /openbayes/home/gptq_plus
git fetch https://github.com/Mrluxianseng/gptq_plus_mp_quant.git realq-warmadam
STREAM_SCRIPT="/tmp/stream-storage-formal-$(date +%Y%m%d-%H%M%S).py"
git show FETCH_HEAD:tools/stream_storage_method_formal.py > "$STREAM_SCRIPT"
RUN_DIR=$(ls -td outputs/storage-formal-*/ | head -n 1)
CONTROLLER_LOG=$(ls -t storage-formal-20*.log | head -n 1)
LIVE_LOG="storage-formal-all-$(date +%Y%m%d-%H%M%S).log"
nohup /output/env/bin/python -u "$STREAM_SCRIPT" --run-directory "$RUN_DIR" --controller-log "$CONTROLLER_LOG" \
  > "$LIVE_LOG" 2>&1 < /dev/null &
echo "总日志=$LIVE_LOG reader_PID=$!"
tail -f "$LIVE_LOG"
```

reader会先读取当前run已有输出，再自动跟踪后续阶段/模型的日志，保留tag/stage身份；controller最终status出现后输出实际完成/失败状态并退出。它只读旧日志、不修改任何绑定文件。新版本已有完整总日志时reader拒绝重复汇总，直接看nohup总日志即可。新版本源码应在本次实验结束后pull，供下一次启动使用。


## 2026-10-13：Fisher缓存dtype校验修复

服务器03:14日志表明，四卡Stage0成功保存缓存并退出，随后独立缓存检查在layer0报Invalid full FP32 Fisher，尚未启动Stage2。生产collector实际执行static_fisher.append(fisher_mean.to(torch.bfloat16).cpu())：累积使用FP32，持久化矩阵使用BF16。新增校验器将累积精度误当存储精度，因此误拒新计算缓存；此前同一错误对旧四卡缓存的拒绝不能证明原文件损坏。生产数学不变，源数据不转换、不重写。

校验contract新增fisher_dtype=torch.bfloat16，通过AST核对生产保存语句；矩阵形状、完整28层/7模块、所有rank、有限值、prepared checkpoint及字节SHA检查保留。异常明确打印实际type/shape/dtype，生产保存语句改变时要求审查schema。回归直接执行真实collector保存转换及真实_to_save映射，创建四rank文件，重现旧FP32检查误拒，验证修复后查找/准入/独立复制及字节一致。相关51项CPU测试通过，没有本地新增GPU实验或服务器正式完整通过结论。

两次20261013四卡precompute目录中的缓存保留，更新后先只用CPU核验现有缓存，再启动新的正式run目录。不要将已生成的缓存删除后重新运行Stage0来绕过校验问题。


## 2026-10-13：默认共享大文件，取消每臂复制

默认cache-mode=auto、cache-storage=shared。匹配缓存和prepared权重通过硬链接复用同一份磁盘数据，跨文件系统使用文件软链接；两者均不可用时拒绝启动，不默默退回复制。static_cache下仍保留相同文件名，供原加载器和完整SHA审查使用；每臂单独保存source_model路径别名元数据，并在shared_cache_reference.json记录链接源/目标/类型以及additional_bulk_payload_bytes=0。一个目录单独统计的逻辑文件大小可能仍很大，实际新增的大文件数据块为零。

Fisher分片缺失或卡数改变时，控制器会独立查找相同模型身份、旋转配置、序列长度和架构的prepared模型，将它共享给一次性Stage0；该次只新生成所需Fisher，不重复保存已有prepared权重。来源字节使用前后完整核验；共享模型路径禁止prepared checkpoint写入，缓存命中禁止再次收集Fisher。共享只读指实验加载流程：外部程序仍不得写这些引用的源文件，修改会触发SHA拒绝。

日志、显存采样、配置、SHA清单和少量私有元数据仍占空间，首次缺失数据必须保存新缓存；显式cache-mode=fresh为了独立测量预计算仍生成各臂统计，不能称零磁盘增长。显式cache-storage=copy保留旧独立复制策略。默认启动无需新参数，若希望命令明确，可追加--cache-storage shared。57项初版、61项完整CPU回归通过；真实safetensors/torch.load（普通和mmap）加载后内存修改不改来源字节；无新本地GPU实验。原旧副本不自动去重或删除。


## 清理旧的大型生成文件

用户授权清理旧的生成缓存和prepared权重。独立工具tools/cleanup_generated_large_files.py只扫描仓库cache/和outputs/，默认选择100MiB以上的生成张量/数组文件，prepared模型权重不受大小阈值限制；原始/非prepared模型、下载数据目录、tokens、环境、代码、日志、JSON/CSV/JSONL验收数据保留。先用--plan生成准确清单，再--apply同一计划。apply检查运行中的formal/torchrun/PTQ进程、路径范围、每个文件inode/大小/mtime/链接目标未变化；新出现的文件不加入删除。目录软链接不遍历，文件软链接只解除引用，不删除指向的数据。硬链接释放量只计所有链接名字均被清除的数据块，不把逻辑重复计作释放量。

prepared权重删除时先取消其_SUCCESS完成标记，保证下一次重新构建而非加载一个不完整模型。清理会导致对应Fisher/prepared下次需要重新计算/生成一份，共享设置避免以后每臂重复复制。工具无递归删除，原有日志及验收证据保留；6项CPU测试通过，仅临时测试目录删除，没有对实际服务器执行清理。SSH root@ssh.openbayes.com:30516已尝试连接，认证失败；待用户安装清理专用公钥后才能实际盘点/执行。

服务器手动用法（确认无实验进程）：
```bash
cd /openbayes/home/gptq_plus
PLAN="cleanup-generated-$(date +%Y%m%d-%H%M%S).json"
/output/env/bin/python tools/cleanup_generated_large_files.py --root . --plan "$PLAN"
/output/env/bin/python tools/cleanup_generated_large_files.py --root . --apply "$PLAN"
```


### 2026-10-13：实际SSH清理结果（覆盖此前“尚未清理”状态）

已在用户授权范围内执行删除24项：8份Fisher、10份ref logits.cache、3份prepared权重及3个_SUCCESS。预计实际数据块释放35.44GiB，du仓库约37G→1.3G，整个/output56G→21G；模型、下载数据、环境和日志保留。54,951保护条目无变化，3份原始模型权重SHA一致。清理/保护计划和结果报告在仓库根，已取回本机outputs/server-generated-cleanup-20261013-v1。清理前无实验进程，未另启GPU。服务器当前HEAD a1823ec已启用共享默认；下次启动因为缓存已移除，仍必须新生成一份统计和prepared模型，再共享给各臂。

### 2026-10-13：正式控制器 job_index 未赋值修复

失败/归因：用户服务器Stage0生成缓存后，run_storage_method_formal.py剩余实验臂磁盘预检引用job_index，实际--run循环未赋值而UnboundLocalError退出。此前索引误加在--dry-run循环中；旧测试覆盖计划和子流程但没有执行外层--run，属于控制器接入错误，不是GPU OOM、缓存损坏或量化方法失败。原失败日志、缓存和结果保留。

方法/验证：实际运行循环改为enumerate(jobs)，dry-run恢复普通循环；数学、校准、模型、共享缓存和验收规则均不变。新增真实main --run的四项CPU调度回归，覆盖auto缓存命中/成功预计算及shared/copy，两个模型各三对、12臂、每模型remaining_arms=6和交替顺序。修复前在实际主循环复现原异常（1 failed），修复后相关65项CPU回归全部通过，46.09秒。进程/缓存/质量结果由CPU fixture模拟只验证调度，不能算正式GPU成功；meta所有权、缓存原生产保存schema及日志仍由相关独立测试覆盖。输出保存outputs/formal-controller-job-index-repair-20261013-v1，不覆盖旧证据。

评价/交付：此次遗漏应由外层控制器执行测试捕获，不能仅凭dry-run/compile宣布正式流程可靠。用户授权本地推送后服务器拉取；只在实际无实验进程时更新，不另启GPU、不清理新生成缓存。服务器SSH端口目前出现banner连接拒绝/超时，尚未完成拉取，后续以实际连接和HEAD核对为准。
2026-10-13交付续记：用户给出新SSH端口32327后，已密码认证连接实际原主机。执行前无formal/PTQ/torchrun进程，服务器realq-warmadam无tracked改动；git pull --ff-only origin realq-warmadam实际acfc5fa→e685c0f，完整HEAD与GitHub e685c0f739f7626cc2465fc24af155387681349c一致，服务器Python3.10 py_compile通过。服务器未安装pytest，额外CPU回归未执行（No module named pytest工件保留）；本地65项通过仍是测试依据，不冒称服务器测试成功或GPU正式通过。绑定总日志storage-formal-20261013-062852.log确认同一job_index异常，四个world4原分片每份3347217037bytes仍存在；旧失败目录、prepared及新缓存未改未删，不另启GPU。下次新目录auto启动按完整缓存准入复用，通过全部正式阶段后才能报告结果。