# REAL-Q 1.7B 瓶颈开发验证流程

目标：减少每个候选重复 Stage-0、完整量化、评估和大型缓存哈希的开发成本。显存优化的判定标准不放宽；局部筛查不代替最终完整流程验收。

## 当前事实

- 已有两模型的真实首层 FFN 激活组件实验。它采用 uniform saliency、无真实 Adam refresh，不等同于真实量化轨迹快照。
- 当前完整 profile runner 强制 fresh Stage-0 + Stage-2，每个配对重复预计算、准备 FP checkpoint、评估和13项缓存审计。
- 现有 `quant_stop_layer` 限制量化终止位置，但会从第0层开始，并不支持从任意层恢复。
- 完整显存 profile 保存分配器统计和权重段归属，未保存可恢复真实瓶颈前计算状态的完整检查点。不能宣称已有即时恢复器。
- 历史诊断 recorder 可保存 H、权重和梯度，但其 FP32转换、逐 block 大量落盘和 GPU/CPU复制会改变开销，不应直接把捕获运行当作无侵入峰值基准。

## 分级流程

### 1. 求逆与写回组件筛查

一次捕获真实 REAL-Q 轨迹的目标模块数据，后续复用。重点目标为1.7B的 down_proj；选择最后一层27的峰值条件，以及出现资源回升的层3、10。捕获应记录实际输入 Hessian、权重、梯度/必要量化状态、dtype/stride、分组、act_order、damping/retry条件及预期结果。捕获文件需绑定模型、配置、源代码和文件哈希；捕获过程另算，不用于测量收益。

改变 solver/workspace 时，优先重放该组件，核对因子数值、布局、量化结果及 allocated/reserved 峰值；正确性失败先修bug。不用缩小 Hessian 维度替代实际1.7B宽度。现有实际首层组件仍可作为基础筛查，但不能冒充层27真实轨迹。

### 2. 短窗口联动检查

至少包括：Hessian求逆 → 模块量化/Adam相关状态 → 权重写回 → Transformer block 输出回放与清理。只测 solver 内部会漏掉长期权重占住临时大段的问题。

复用经过验证的 Stage-0 静态缓存；该缓存只在模型/输入/校准/旋转/Fisher定义/相关Stage-0代码及运行条件不变时可复用。两臂使用同一只读缓存，并记录缓存身份。Stage-2代码改动不能自动使旧缓存合法，仍需判断是否影响Stage-0行为。

恢复张量值不恢复原 CUDA allocator 分配历史。短窗口结果只作筛查；需要明确采用重新构造的内存环境还是继承完整前缀的环境，不能用前者证明原来的 reserved 区间回升消失。若回升依赖历史，应保留较便宜的真实前缀检查，直到有已验证的分配轨迹重放办法。

开发窗口可以跳过全模型PPL/KL评估，核对本模块量化权重、block输出、refresh梯度/损失等局部预期结果；这不是最终质量证明。

### 3. 稳定候选的完整验收

局部正确且资源有收益之后，做一次完整28层配对，核对每层输出、state、KL/PPL、全阶段allocated/reserved/NVML。仍有局部回升先定位，不立刻进入三次全量重复。

只有稳定候选进入最终三组完整流程对照，报告峰值、时间和重复波动；正式配置/多模型/多卡独立验证。Stage-0复用的开发运行不用于宣称端到端加速或全流程峰值下降。

## 初始规划时尚待实现（下方记录后续实现）

- 小型真实轨迹捕获和组件重放器，保留dtype/stride而非全部转FP32。
- 缓存身份校验及Stage-2-only窗口驱动。
- 短窗口状态恢复正确性核对；allocator历史限制必须保留在报告中。

上述是流程调整与实现要求，不是已完成的新实验结果。当前正在运行的workspace候选仍遵循其已保存manifest，不中途改参数/源码污染实验；后续开发候选按本流程筛查。

## 2026-10-11 实现补充：真实求逆组件捕获

- `tools/hessian_bottleneck_capture.py`：在真实 REAL-Q 的求逆入口保存指定层首个 down_proj 子组，输入已经过死列处理与 act_order；保存 H、调用配置及原求逆结果，保留 dtype/stride，绑定文件与张量哈希。捕获是有侵入的诊断，不作为显存/时间基准。
- `tools/run_cached_hessian_capture.py`：等待已有队列结束、确认无 CUDA compute 进程后，核验完成的单卡全流程参考源码及13项缓存；复用缓存执行一次 Stage-2，禁止静默重新预计算，运行前后检查缓存不变。默认捕获层3/10/27，核对最终 state、全部28层输出及KL/PPL，再启动组件重放。原队列和源码未中途改变。
- `tools/replay_hessian_bottleneck.py`：每个方法/真实 H 单独启动CUDA进程，预热后测三次；比较原始、early-release及两缓冲区solver的因子、布局、H输入副作用、damping/fallback与保存预期一致性。统计 solver allocated/reserved 峰值，计时只作诊断。源码/缓存/运行库不匹配即停止。
- `tests/test_hessian_bottleneck_capture.py`：WSL CPU测试2项通过（BF16/列主序往返、捕获范围/首子组/副作用/参数及文件破坏检测）。真实GPU捕获与重放尚待实际运行；CPU通过不代表CUDA或轨迹通过。

本实现仅完成**求逆组件**数据复用，不是完整 block/Adam恢复器；短窗口联动及 allocator 历史复现仍待实现。缓存重用开发运行不用于声称Stage-0峰值或端到端速度收益。任意层恢复和多卡捕获仍未支持。

### 条件优化及后续完整验收

`tools/hessian_all_weight_storage_prototype.py` 将已测试的稳定最终权重写回扩展到所有顺序量化线性模块，算法/转换保持不变。假设其他模块新权重也可能钉住临时大段，尚未确认该归因或资源收益。

`tools/run_cached_all_weight_screen.py` 等待真实捕获/重放完成，且只在workspace完整队列出现数值正确资源失败时执行：复用完成的核验缓存，跑真实0–10层两臂前缀，核对77次模块写回覆盖、全部held-out层输出、state/KL/PPL、精确allocated/reserved区间及NVML峰值。此处保留Stage-2从层0的分配历史，仍不是从任意层恢复。workspace完整队列已通过则跳过；数值/基础设施失败不盲目启动。

`tools/run_all_weight_full_followup.py` 只在该前缀通过后，逐对跑独立fresh Stage-0+完整28层+评估：先1.7B，再0.6B，各三对；任何一对失败停止该配置的后续重复。另对canonical已接受参考的state/全部28层/KL/PPL及196次权重写回、112次down_proj求逆/因子复用做覆盖检查。使用未profile的入口生成完整测量入口，避免重复包装评估。完整结果仍是本机cal2×1024的pilot，未证明正式2048样本或多卡配置。

这些脚本已编译、生成方法和入口已做静态编译检查，等待真实GPU执行。它们的自动续接不会把“源码编译通过”当作“方法正确/资源通过”。


### 2026-10-11 23:14：有限值组合的条件完整验收入口

`tools/run_bounded_finite_full_followup.py` 是新版本续接入口，不修改活动前缀控制器或旧full入口。等待finite前缀最终status，只有数值/缓存/区间资源门槛通过且 `audit_memory_boundary_points.py` 对每个stage/log-kind/rank/phase/occurrence的start/end均无回升、无缺失，才允许fresh全流程。前缀资源失败也保留独立逐点审查，但不启动full。

使用原始未profile workspace入口注入绑定bounded installer及all-linear稳定写回；Stage-0允许真实预计算，Stage-2禁止静默重算缓存。全流程入口再由既有profile runner包装一次，避免重复评估包装。每对重新生成Stage-0缓存，先1.7B再0.6B，各三对；每一对需通过旧区间/全局门槛、新逐起点/终点门槛、canonical状态/28层输出/KL/PPL、196写回及112次solver/finite覆盖，方可后续重复。任何失败停止相同候选重复，保留证据并交回机制分析。

绑定prefix所用源码、生成入口、新控制器及验收/日志/磁盘整理工具，逐对核对哈希；单一总日志reader只在前缀完成后接续。完成的prepared模型只在字节身份核验后使用已有hard-link整理工具，保留路径和Fisher，不将磁盘整理计入GPU收益。CPU三项通过：终点回升/缺失拒绝、源码变化拒绝、真实入口fresh预计算及单次包装边界。证据 `outputs/bounded-finite-full-gate-cpu-20261011-v1.log`。当前控制器只等待，未取得新full GPU结果；仍是本机小校准pilot，正式配置/多卡待验证。


### 2026-10-11 23:53：dead-H_sub接入与条件前缀

新增tools/hessian_dead_subgroup_install.py接到实际stable-down builder，加入已审查死亡点的输入清理、alias保护及CPU元数据，控制保持原方法。CPU2项和真实profile入口生成通过；不是GPU验收。tools/run_cached_dead_subgroup_screen.py只在finite完整质量/来源正确且资源失败后串行验证workspace/down-only组合+清理，单GPU不并发。每点start/end为实际gate；11清理/44solver/11写回覆盖、cache/source/allstate/28层/KL/PPL及区间/全局峰必须全部通过才进入新的fresh完整入口。不直接重用含finite/all-linear的full入口。


### 2026-10-12 00:06：末组私有输入的fresh完整验收门槛

新tools/run_dead_subgroup_full_followup.py以未profile的workspace原始入口注入dead installer，禁止finite/all-linear扩展；Stage-0允许真实预计算，Stage-2仅消费对应fresh缓存。前缀旧区间门槛和独立每个start/end均通过才启动，每对再核对canonical state/28层/KL/PPL、全部区间和边界/global/NVML、源码绑定、112求逆/28写回/28清理（控制不清理、候选保留四组因子）。两模型各三对逐对准入，失败停止相同配置重复并保留证据。CPU两项通过但尚无GPU完整结果，条件等待器仅CPU；原日志reader在前缀完成后接续，不并发GPU、不修改旧活动冻结文件。


00:31补充：新的activation_replay_stats_lifetime_prototype.py仅CPU准备；AST验证回放之后不存在gptq/saliency_dict/gradients_dict消费者，再前移原有del。不得以弱引用CPU检查宣称GPU改善，不改变活动冻结源。必须当前实验最终质量/来源审查正常且资源失败，再生成新入口，用方法覆盖元数据及所有state/28层/KLPPL/cache/source/区间峰/每点/global严格门槛独立验证。


00:58恢复：原dead/旧full等待器/统计v1等待器均已消失，旧候选评估中断，不能称完成。新恢复工具会话25410，controller17/dead19/唯一reader21（实际ps为准），outputs/replay-stats-interrupted-recovery-20261012-v1；先outputs/qwen17b-cached-dead-subgroup-screen-20261012-recovery-v2，完成质量/cache/source及资源失败核对后才outputs/qwen17b-cached-replay-stats-screen-20261012-recovery-v2。新stats installer CPU4及prototype2通过，尚无GPU收益；保护no_grad/live globals/测量wrapper，两臂11层CPU观察。原中断证据保留，新源码/目录，不填原status。统计前缀全区间/每点通过才新fresh完整两模型三对；尚待新full入口。


00:59完整续接准备：tools/run_replay_stats_full_followup.py新入口CPU2项通过，保留fresh Stage0/单测量包装及28层stats缺失/字节改变拒绝。会话93826、CPU等待器61等待恢复stats-v2 status；目录outputs/replay-stats-full-followup-20261012-v1。前缀数值/cache/覆盖及每区间/每point/global严格通过才fresh两模型各三对，每对再核对canonical全部数值与112inverse/28write/28dead/28stats同层storage持有量。尚无新GPU收益，不使用原中断olddead/full队列。恢复控制组当前唯一CUDA51，controller17/dead19/reader21；每次实际ps更新。检查点outputs/realq-continuation-20261012-0100-recovery.json。旧指纹0–14完成相同、15–27空，28行列表不是28层验收；统计字节不是峰下降。


01:43实际续接：恢复dead-v2完整结束validFalse，507state/all28/KLPPL/44solver11writes11dead/cache/controller/source正确，4644point0回升，但layer3/10 replay interval reserved各+36MiB，禁止同配置full。新stats-v2实际controller208/唯一GPU219控制组/reader210，full CPU61仍等待；新绑定源码独立核对见outputs/realq-continuation-20261012-0143.json。新stats真实结果未完成，不将运行当改善。独立旧完成审查及最终缓存补审保存在dead恢复目录，下次按实际ps更新。


02:01新stats控制组完成，state/all28/KLPPL及4644边界与旧控制逐项相同，11层storage观察48.0234375MiB/层、46源码未变。独立control审查保存在stats恢复目录；仅持有量，无GPU收益结论。当前唯一候选GPU299第0层，controller208/reader210/fullCPU61，下次以实际ps为准。


02:26前缀stats recovery-v2独立全部通过：507/all28/KLPPL/46sources/13cache、1161interval及4644point无回升，layer3/10 replay+36被消除，但非全程。full-v1主动停在首对未完成，比较器漏登记工程flag将误报算法改变；旧source/artifact保留0pairs。新tools/run_replay_stats_memory_campaign.py仅登记该存储flag，CPU2含真实命令相等和算法改变拒绝；new tools/run_replay_stats_full_followup_v2.py绑定adapter/启动前命令核对，新目录outputs/replay-stats-full-followup-20261012-v2已启动fresh1.7B首对，随后三对及0.6B三对须逐对全部门槛。旧61已停止，不沿用旧PID；检查实际ps。证据outputs/realq-continuation-20261012-0226.json。

### 2026-10-12 04:34：首对完整 stats 生命周期组合严格通过，继续重复

NEW full-v2 的 1.7B p1 fresh Stage0+28 层+评估已完成。独立 outputs/replay-stats-full-followup-20261012-v2/independent_completed_pair_audit-20261012-0434.json validTrue：canonical507/all28非空/KLPPL精确，55源码不变、13 extended-path 缓存SHA一致；2961 exact interval/11844 各自start-end无回升/缺失，112inverse28write28dead28stats完整，所有stage/评估/global门槛通过。NVML6629→6089MiB，allocated5605.190918→4883.690918、reserved5872→5332；Stage0/评估无变化，非正式2048/多卡结果。诊断quant2908.442→3635.631，候选更慢，不声称加速；重复及低扰动计时尚待。

04:31实际进程：full-v2 controller17、reader19；p1 campaign20已结束，p2 campaign317/profile318、唯一GPU328控制Stage0。下次以实际ps更新；不是旧recovery PID17。首对通过不代表两模型三次通过，继续新目录p2/p3再0.6B三对逐对所有门槛。最大allocated区间仍是末层down after_inverse→后续刷新，不归纯求逆。原workspace layer3/10 replay+36在首对完整组合消除，但具体分配根因未确认。不更改活动冻结源/门槛、不并发GPU、不推送。

### 2026-10-12 06:52：第二对独立严格通过，继续第三对

full-v2 p2完整结束；独立 outputs/replay-stats-full-followup-20261012-v2/independent_completed_pair_audit-20261012-0652.json validTrue：canonical507/all28非空/KLPPL精确，55源码不变、13extended-path缓存SHA一致，2961exact区间/11844独立起终点无回升/缺失，112inverse28write28dead28stats及所有global/stage/评估通过。NVML6629→6089、allocated5605.190918→4883.690918、reserved5872→5332MiB，与p1相同，仅两对pilot；0.6B未验收。p2诊断quant3607.362→3534.241与p1方向不同，不能称加速，低扰动计时待做。

06:51实际：fullcontroller17/reader19，p2campaign317已结束；p3campaign571/profile572/唯一GPU582 control freshStage0。每次actualps更新，不能沿用旧GPU520/328。继续p3再0.6B三对；不编辑绑定源/门槛、不并发、不推送。最大after_inverse区间含后续刷新，stats持有量非峰下降，NVML采样非每刻证明。

### 2026-10-12 09:09：1.7B 三对独立完整通过，继续0.6B

第三对独立 independent_completed_pair_audit-20261012-0902.json（实际09:08）及 independent_three_pair_summary-20261012-0909.json 位于outputs/replay-stats-full-followup-20261012-v2，validTrue。三对canonical507/all28非空/KLPPL精确，逐对55source/13extended-path cache/112inverse28write28dead28stats通过；累计8883 exact区间/35532独立start-end均无回升/缺失，所有global/stage/评估通过。NVML三对均6629→6089、allocated5605.190918→4883.690918、reserved5872→5332MiB。仅1.7B三对pilot，0.6B及低扰动速度验证待完成；第三对诊断quant3540.362→3667.490，不作加速结论。

09:08 actual controller17/reader19/p3campaign571正在CPU最终cache SHA，GPU707/profile667已结束，不沿用；独立完整cache流式SHA已通过，仍待原controller审核后自动启动0.6B。GPU空闲属审核，不是停止研究；不能补造原status。继续0.6B三对所有独立门槛，再必要速度验证与报告；不修改活动冻结源/门槛、不并发、不推送。stats持有量不是峰降，after_inverse含后续刷新，NVML采样非每刻证明。

### 2026-10-12 09:57：0.6B资源门槛失败，旧队列结束，继续新工程候选

独立 outputs/replay-stats-full-followup-20261012-v2/independent_qwen06_completed_failure_audit-20261012-0951.json validFalse；507/all28非空/KLPPL/55source/13cache/覆盖均通过，全局下降，但layer25回放interval reserved+96MiB，回放start及最后after_inverse end共2point+96MiB，11844point无缺失。full-v2已09:46:48结束，仅接受1.7B三对，0.6B后两对未跑；禁止同失败配置重跑。实际WSL已重启，旧controller17/GPU885都不在，下一次actualps为准。

新tools/hessian_weight_size_policy_prototype.py及outputs/weight-size-policy-cpu-20261012-v1/entry.py已准备，CPU5与真实fresh入口编译通过，未跑GPU。保留workspace/reuse/dead/stats；down目的权重>=20MiB稳定写回，否则原转换分配（0.6B6MiB，1.7B24MiB）。依据PyTorch2.11原生allocator1–10MiB拆20MiB段官方源码/OLLA位置寿命，层25weight位于128MiB段是疑点非唯一根因。新观察器requested_enabled/selected enabled区分，不能用旧28stable gate误报；下一步独立新screen runner绑定helper+rawentry、保留live stats globals/单测量wrapper，必须覆盖layer25，全数值/source/cache/112solver28writes28dead28stats/所有区间point/global严格审查，通过才fresh必要重复。不能只轮询已结束队列或停止研究。速度/正式2048/多卡待，不并发不推送不修改旧绑定源。

### 2026-10-12 10:03：NEW size-policy0.6完整screen实际启动

新tools/run_weight_size_policy_screen.py已接入，CPU2通过，实际会话84210，out outputs/weight-size-policy-screen-20261012-v1；tag qwen06b-weight-size-policy-screen-20261012-v1；logs/weight-size-policy-screen-20261012-v1.log。10:02 actual 新controller17/reader19/campaign20/profile21/唯一GPU31 control freshStage0；17不是旧full-v2，后续按actualps。56绑定source独立SHA不变，independent_launch_source_audit.json只来源启动审查。新一对freshStage0+全28层+评估，not0–10/no自动重复；oldfull-v2冻结源/失败结果保留。

小down权重6MiB实际selected stableFalse、requested candidateTrue；保留112solver/reuse/28write记录28dead28stats，必须新policy_coverage而非旧28pointer gate。控制同观察器，算法参数一致；所有canonical/source/13完整cache/everyinterval/everypoint/global严格门槛通过再后续必要fresh重复，否则独立分析新失败。GPU收益尚无、不能称已修复96MiB。研究继续，单GPU不并发不推送不更改活动冻结源。


### 2026-10-12 10:45：size-policy首对失败，禁止相同配置重复

独立outputs/weight-size-policy-screen-20261012-v1/independent_completed_pair_audit-20261012-1042.json validFalse：507/all28/KLPPL、58source/13完整cache/112inverse28requested-selectedwrite28dead28stats与global通过，2961区间4处回放reserved+32至64MiB；11844point4处+96MiB、无缺失。layer25消失但layer10/11/17/20回升，拒绝推广。10:38:24控制器正常完成，GPU/旧PID17/147已结束，后续actualps确认；不重跑此失败配置。下一方向为整层回放权重专用紧凑native池位置/生命周期，尚未实现或GPU预检，不等同旧求逆临时池；先保护数值/参数身份/别名/池释放与真实入口，再新完整screen。详见主记录。速度/正式2048/多卡未验证，不并发不推送不改原冻结源码。


### 2026-10-12 11:05：NEW整层回放紧凑池screen启动，尚无全程收益

CPU5与两模型真实FP16 decoder组件通过（值/固定32token输出/Parameter身份/CPU释放值/pool active0），outputs/replay-weight-compact-cuda-20261012-v1/status.json；非量化完整验收。新helper compact所有11参数到单native池，只将原最后down free清理尾部移到复制之后（H/Losses原点释放、调用数不增），layer CPU释放后active0再销毁，保护liveglobals/单wrapper。新screen会话72728，out outputs/replay-weight-compact-screen-20261012-v1，tag qwen06b-replay-weight-compact-screen-20261012-v1。11:04 actual新17/reader19/campaign20/profile21/唯一GPU31 controlStage0，旧size17早已结束；60source launch独立SHA不变，仅启动来源。全fresh28层一对全部canonical/source/13cache/solver-reuse-write-dead-stats/pool28+cleanup28/interval/point/global严格门槛过后才后续必要重复，否则独立完整审查继续工程。不能重跑旧失败、放宽门槛、将pool32MiB持有量称峰降，组件不证明收益。不并发不推送不改活动绑定源。


### 2026-10-12 11:15：用户硬件目标澄清——全局峰值主验收，局部保留诊断

用户明确认为局部reserved回升不抬高全局allocated/reserved时可合格。新的容量主门槛为canonical507/all28非空/KLPPL/来源/cache/方法覆盖完整、full process allocated/reserved及NVML采样全峰不高于control；everyinterval/everypoint仍完整记录审查，局部变化单列而非否决容量。原strict结果不覆盖。outputs/hardware-global-peak-reclassification-20261012-v1.json已审证据/statusSHA补审重新分类：1.7B三对及0.6B两种方法各单对全局条件通过，原0.6strictFalse保留；不同方法不能合并重复。活动compact source/checker冻结照常完成，再独立新global验收。局部回升不再无限追零，应优先最简容量合格组合的必要同配置重复、低扰动速度成本和报告，重复须新准入理由/新目录（旧失败只local，非全局失败），不得遮掩质量/基础设施失败。NVML采样非连续证明、正式2048/多卡/更小GPU待。不并发不推送。


### 2026-10-12 11:38：原stable+stats容量独立复核通过，CPU续接等待器已启动

新auditor tools/audit_completed_memory_pair_capacity.py保留局部diagnostic，global/质量/source/cache/覆盖/完整遥测仍硬门槛；CPU2拒绝global回升/缺失。原q06首对独立55source/13完整cache/507state/all28/KLPPL/覆盖2961interval11844point实际重核，out旧full-v2/independent_qwen06_hardware_capacity_audit-20261012-1135.json validTrue/strictFalse，仅首对不是3对。新tools/run_qwen06_hardware_capacity_followup.py CPU等待会话54194/PID224（actual11:37），out outputs/qwen06-hardware-capacity-followup-20261012-v1，等待compact root/independent_completed_capacity_audit.json及其status，quality/source/cache/method/telemetry完整无GPU才fresh原stable+stats repeat2/3，使用旧full-v2原rawentry exact副本、不带size/compact。新auditor和waiter源码冻结，不编辑。当前唯一GPU165/compact17/reader19候选层22附近，actualps更新。旧局部strict return1作为诊断，必须独立全容量accept才下一对，global或数值/基础设施失败拒绝。下一次compact完整后先CLI新auditor --method compact --component outputs/replay-weight-compact-cuda-20261012-v1/status.json --bindings outputs/replay-weight-compact-screen-20261012-v1/source_bindings.json --tag qwen06b-replay-weight-compact-screen-20261012-v1 --output outputs/replay-weight-compact-screen-20261012-v1/independent_completed_capacity_audit.json；此文件触发CPU续接，不得在质量/覆盖缺失时提前写通过。重复新准入理由为用户全局目标，非旧失败寻优重跑。不并发不推送，速度/正式2048/多卡未验。


### 2026-10-12 11:55：compact完整BF16独立通过，优先简方案重复

原compact11:45 statusFalse仅FP16组件规格对比真实BF16导致，原source/status保留；新audit_replay_compact_bf16_completed.py依据源config架构BF16核对实际28层11param/池active0/shared1/原cleanup1，并重核全部canonical507/all28/KLPPL/62source/13cache/112inverse写回deadstats/2961interval11844points，out compactroot/independent_completed_capacity_audit.json validTrue/strictTrue，0回升缺失。NVML4247→4119/alloc3304098304→3156305408B/res3490→3394MiB。相对简方案res3362多32、alloc/NVML无额外收益，不推广pool重复；诊断831.727→1298.994不称速度。文档完整FP16措辞纠正实际BF16，FP16原组件不伪造BF16；无精度降低。文件已触发followup224，11:54仍CPU原简首对readmission cache审查，new56source SHA和rawentry exact通过，不是新GPU成功；旧compact17/165/19结束。通过后原stable+stats freshp2/p3，新的全局容量主门槛、local诊断/旧strict保留。继续低扰动速度/报告，不并发不推送不改frozen。


11:56 actual补充：followup224已完成原首对CPU独立readmission validTrue/strictFalse，原source/cache/覆盖不变；新p2已11:53附近串行启动campaign290/profile291/唯一GPU301 control freshStage0，不再是仅CPU等待。唯一总reader289，actualps更新；原compact17/165/reader19结束。新56source原rawentry exact已独立核对，p2未完成，不冒称三对。


2026-10-12 12:14 低扰动计时入口CPU准备：tools/prepare_replay_stats_low_overhead_entry.py保留原stable+stats数学和覆盖，关闭可选密集profiling，CUDA同步quant/collector计时，state与all28 heldout fingerprint/hash在主timer外，freshcache仍验。CPU2+实际生成out outputs/replay-stats-low-overhead-entry-cpu-20261012-v1/entry.py/status.json仅准备无GPU速度/正确性结果；sharedCPU观察仍保留，不称零开销/endtoend。当前容量p2控制GPU343层15正常，p3尚待；计时runner/准入待完整容量重复，不并发不改56source。


2026-10-12 12:44 新tools/run_replay_stats_low_overhead_cost.py CPU准备（runner2+入口2共4CPU通过）：容量p3完成后先独立六对完整readmission再双模型各3交替低扰动时间对，所有507/all28/KLPPL/source/13cache和真实方法records仍完整必验，参数/storagebytes变化拒绝，仅地址/段位置诊断不作身份；源timer不改、AST仅实际入口复制边界，尚未启动GPU/无速度结论。out outputs/replay-stats-low-overhead-cost-cpu-20261012-v1/status.json。当前唯一q06p2候选GPU467层14附近，控制57source/13cache及canonical/完整自身telemetry独立通过，仅control不算配对；p3待。D盘~11GiB需后续实际存储核算，原证据不删/未dedup，不并发不推送不改活动56source。


2026-10-12 12:51 计时存储只读核算：ordinary Windows路径再次漏MAX_PATH Fisher，12文件初稿证据保留；extended-path新独立plan完整13文件/model。未来6对若重复保存所有prepared约34GB，当前11GiB不够；1.7B旧六臂prepared已同inode，不能假算可再释放20GB。既有已完成SHA只读hardlink流程可保留全部路径/字节/Fisher独立，后续新timing需CPU准备/验证兼容其完整status准入的事后prepared链接，进入下一arm前腾重复磁盘占用，属于磁盘housekeeping非显存贡献；此刻未链接/删除或启动timingGPU。原不完整plan和extended完整plan均在outputs/replay-stats-low-overhead-cost-cpu-20261012-v1，当前p2 GPU健康，继续完整验收。


2026-10-12 13:01 0.6B原简repeat2完整native独立capacityTrue/strictFalse，out independent_completed_pair_audit-20261012-1300.json：507/all28/KLPPL/58source/13extended cache/覆盖及2961interval11844point完整；global NVML4247→4119/alloc3304098304→3156305408/res3490→3362，local layer25 interval1/point2 reserved+96诊断保留。原简仅两对非最终三对。第三对actual12:59 controller224/reader289/campaign532/profile533/唯一GPU543 freshcontrol，旧467结束。继续完整第三对、低扰动成本与报告，不并发不改56冻结source不推送。


2026-10-12 13:12 新timing_completed_prepared_disk.py与costrunner接入CPU6通过，out low-overhead-cost-cpu-20261012-v2仅CPU；只有每arm完整质量/method/cache/source之后prepared逐字SHA稳定核对再只读hardlink，保留全部路径/字节/Fisher独立、GPUtimer外磁盘管理，nextarm查缓存大小+1GiB余量，未对真实结果新链接/未启动timingGPU。原v1 runner按SHAsnapshot保留，56活动sources实际不变。当前第三对唯一GPU584 control层17附近，旧543结束，继续完整第三对才timing。不并发不推送。


2026-10-12 13:30 NEW cost CPUwaiter session7219/PID731（actual13:23）运行，等待followup status+independent_completed_capacity_repeat3_native.json真实完整通过再成本六对重新readmission，不提前写该文件。out outputs/replay-stats-cost-after-capacity-20261012-v1/log同名，62source实际不变并冻结，新waiter/cost/disk/generator禁改；cost out future outputs/replay-stats-low-overhead-cost-20261012-v1。WSL CPU7通过，仅CPU/no额外GPU；p3控制已独立507/all28/KLPPL/57source/13cache/覆盖及遥测完整通过，candidateGPU665仍运行，未称三对完成。结束后必须native完整capacity审查并保存指定final文件触发，不能只看旧valid。


2026-10-12 14:11 最高优先覆盖旧状态：Windows主机13:24:08重启（13:23:40/49 User32 1074），旧224/289/665/731全部结束，原p3候选10/28后无最终status/state/KLPPL/28输出/阶段2方法，旧rootstatus不补造。中断及host重启独立证据在oldcapacityroot；64sources/13cache仍精确不等于数值完成。NEW recovery工具会话24530/out outputs/qwen06-hardware-capacity-recovery-20261012-v2/tagqwen06b-replay-stats-hardware-20261012-recovery-v2-p3，先p1/p2独立完整readmission、只fresh恢复第三对。actual14:10 NEW17/reader19/campaign20/profile90/唯一候选GPU129层27；58source/rawentryexact独立launch通过，冻结不编辑。两臂全部canonical/source/cache/method/interval/point/global完整后再new版本cost准入，旧costwaiter已消失不能称运行/重用旧p3gate、不改旧源码/status。不并发不推送，速度/报告仍待。


2026-10-12 14:18 恢复p3完整native capacityTrue/strictFalse，60source/13cache/canonical/方法2961interval11844points完整，global重复原q06前两对/local+96保留。新two-model-hardware-capacity-accepted-20261012-v1/summary.json native六对重审validTrue(17766/71064)，1.7B三対与0.6B三对原简方法容量成立、正式2048/多卡/速度未验。原中断p3排除不补status。completed p2及recoveryp3 prepared已SHA逐字只读hardlink保路径/Fisher/字节，原中断工件未动。NEWcostv2 onlyrecoveredadmit change otherASTsame，WSL7CPU过、会话8929真实启动out/log replay-stats-low-overhead-cost-20261012-v2，初始CPU六对readmission后无GPU并发才3timing对/model；尚无速度结果。旧costwaiter已结束不要重启、不改旧源/新活动frozen，继续成本及报告不推送。


2026-10-12 14:25 actual成本v2已完成六对实际WSL完整readmission并启动首q06 control，freshStage0 13cache SHA匹配实际通过；controller17/CPUreader70/唯一GPU79 stage2第3层（14:23），17不是已结束recovery17。新60source/currentworkerentry与rootentry逐字相同、无变化，independent_launch_source_and_capacity_binding.json仅来源/准入/Stage0 cache，不是计时完整质量或速度成果。new entrySHA8559c96317b7eb58ddc58826bebe5469c8978a546a8edcaa5f7b02a19697755f以实际WSL冻结字节为准，跨平台CPU工件换行差异不当数学变化。总logs/realq-system-restart-20261010-1408.log已由NEW stream_low_overhead_cost_progress.py CPU reader会话46629串行append，不写旧reader；成本controller会话8929/out/log replay-stats-low-overhead-cost-20261012-v2。禁止编辑新活动costv2/generator/helper/原源；每臂全数值/方法审查后新timingpair才作结论。


2026-10-12 14:47 最高优先：costv2已14:36:26 firstcontrol后eval写工件NameError(root未定义)，0pairs。独立失败审查validFalse保存61source/13cache/507/KLPPL/method已精确但all28工件缺失，不补造/不记control862秒为速度结果。原六对容量不受影响。NEWgenerator_v2 onlypost-eval root→destination所有其他AST不变，真实wrapper旧错复现/28×2CPU文件写出/异常hook清理，WSL共9CPU过。NEWcostv3 import该generator+绑定SHA，工具75844 actual运行out/log replay-stats-low-overhead-cost-20261012-v3，初始CPU六对readmission，reader工具51254 append总log；旧17/79/70已结束每次actualps。不改旧source/status/log，失败工件不dedup，继续完整timing及报告，不并发不推送。

2026-10-12 15:16 低扰动v3首控制臂完成、记录修复真实路径验收：q06 p1 control于15:07:47完成，15:15:38 native fresh独立审查 outputs/replay-stats-low-overhead-cost-20261012-v3/independent_completed_control_native-20261012-1512.json validTrue。全部507state canonical、28层非空heldout fingerprint、KL .30534985661506653/PPL25.013607025146484、61绑定source/actual entry、13完整extended cache SHA、两stage算法命令及control112inverse/28write/28stats/dead-disabled完整metadata与已接受内存参考一致。真实GPU后eval已写全部28层工件，证实destination记录修复；原v2失败不改。control quant1096.531017082秒/collector2.773640983秒只是单臂，0完整timing pairs、不能声称加速或最终成本；CPU观察仍保留、计时不含prep/eval/hash。已完成arm prepared逐字SHA事后只读链接保所有路径/字节/独立Fisher，非显存贡献。actual controller28/reader37/唯一candidateGPU185；候选15:15已12/28量化，原controlGPU90结束。继续三timing对/model及必要报告，不改活动source、不并发、不推送。

2026-10-12 15:31 低扰动成本首个完整配对：costv3 q06 p1于15:27:11完成，NEW independent_completed_pair_native-20261012-1531.json实际15:30:51 native fresh validTrue。两臂各507state/全28非空输出/KL .30534985661506653/PPL25.013607025146484、61source/actualentry、13extended cache SHA、两stage算法命令及所有inverse/write/dead/stats metadata与对应已接受dense内存参考精确，原destination修复真实两臂验收。C/N quant1096.531017082→988.822695287秒，collector2.773640983→2.648890611，collector+quant1099.304658065→991.471585898秒，单对ratio1.108760627839206；这只是q06一对，不能据此称稳定加速或两模型成本结论，保留所有慢值和后续相反顺序。C/N statusSHA e12bb7e6d394f4ec24e84390946915cfcdfeddcd428e4112438d53691e1580d1 / 822836cd5e3736b4808c92cd79f6283ab25737a239f019c9975e75eb525f4fb3。CPU方法观察未关闭、timers不含prep/eval/hash非端到端；没有重复dense遥测，新显存结论仍来自已接受六对完整区间/逐point/global审查。每臂completed prepared事后保字节链接，不算GPU收益。actual15:29 controller28/reader37/唯一GPU275为q06 p2 candidate层2；p1GPU185结束，第二对N→C，第三对C→N，随后1.7三对。继续独立完整验收/必要报告，不修改61冻结source、不并发GPU、不推送。

### 2026-10-12 16:11：低扰动第二对相反顺序验收，速度方向反转

**方法/理论。** 继续同一已接受stable+stats工程，预定p2 N→C用于暴露执行顺序及运行状态影响；OLLA/Checkmate的存储生命周期依据不推导速度必然提升。保留全部时间，不剔除慢值、不扫描算法参数。

**实证/效果。** q06 p2已16:05:46附近完整结束并串行接p3；独立fresh native independent_completed_pair_native-20261012-1610.json实际16:10:09 validTrue：两臂各507state/全部28非空输出/KL .30534985661506653/PPL25.013607025146484/61source及实际入口/13完整extended cache SHA/两stage算法命令/全部inverse-write-dead-stats metadata精确，pair身份/顺序/原timer和统计加和完整。C/N quant811.464644209→1249.974339955秒，collector2.554306777→2.800949035，collector+quant814.018950986→1252.775288990秒，ratio0.649772515582。C/N statusSHA252a8fa0820dd2fbfc59efce60954d5211357a7c9d072e79c611ed8441749a90 / 7378c33217a9666727cc1b787cffb9e4856cfd7f81288f0dfd12eac474b0ac1e。

**评价/反思。** 第二对候选更慢，方向反于首对；低扰动不等于运行条件无波动，目前明确不能宣称稳定加速，未确认热状态/功耗/频率或后台因素唯一原因。不据速度负向否定已通过的容量/数值结论，也不把两对当三对，不为改善均值重跑/删值。CPU观察仍留、collector+quant排除prep/eval/hash不是端到端；dense每区间/逐point/global证据由已接受容量六对保留，timing没有新密集遥测。actual16:10 controller28/reader37/唯一GPU455为q06 p3 control层7，旧p2 GPU365结束；继续p3及q17三timing对与必要报告，不改61冻结source，不并发不推送。

### 2026-10-12 16:55：0.6B三对低扰动成本完整，未证明加速

**方法/理论。** 原简stable+stats工程按预先C→N/N→C/C→N三对，数学/FP32 Hsolver/BF16权重/校准Adam不变，CPU方法观察保留。存储生命周期理论与显存下降不保证walltime下降，时间证据与dense显存证据分开评价。

**实证/效果。** p3于16:44:53完整，native independent_completed_pair_native-20261012-1653.json实际16:53:05 validTrue。p3 C/N quant1002.827937251→1047.339430312秒，collector2.942660246→3.253914315，sum1005.770597497→1050.593344627；C/NstatusSHA1913588ea0c21ca699ac31299e9ec76278b8632f4f41589755120dab877f8086 / 32b82746475875f27bc3266184f3ba090718cbfc4b5b75a706720335f49ed968。NEW independent_three_pair_qwen06_summary-20261012-1655.json实际16:54:57 fresh六臂重新all28/507/KLPPL/source/入口/13extended cache流式SHA/方法/两stage命令/记录SHA验收validTrue。三对C quant[1096.531017082,811.464644209,1002.827937251]均值970.274532847/sampleSD145.294536509秒；N[988.822695287,1249.974339955,1047.339430312]均值1095.378821851/sampleSD137.043360776，候选均值+12.8937%；collector+quant均值+12.8720%。全部时间保留无剔除。

**评价/反思。** 0.6B容量已三对下降且全数值一致，速度三对一快两慢且CV约15.0%/12.5%，本机这组三对没有稳定加速证据，观测平均成本增加，不把它解释成已确认方法固有惩罚或唯一热/功耗原因。结果只cal2×1024pilot、collector+quant排除prep/eval/hash非端到端，正式2048/小GPU/多卡未验。q17成本尚零完整pairs，不能推广两模型速度结论。actual16:52 controller28/reader37/唯一GPU647为q17 p1 control层3，q06原556退出；继续q17三对必要完整验收和最后报告，不改61活动source，不并发不推送。已有六对内存17766interval/71064point与NVML采样global证据独立保留，不冒称计时有dense连续证明。

2026-10-12 17:23 新docs/realq_system_memory_research_report_20261012_draft_v1.md整理已接受两模型六对容量、归因/生命周期方法、0.6三对成本/负向分支/工具错误/文献和限制。明确草稿，1.7成本仍pending，不把论文可显存主目标讨论当停实验指令，不冒称正式2048/多卡/小GPU或稳定加速。未修改活动61source、原证据或GPU队列；实际q17p1 control层25正常，继续必要完整成本及最终报告。



### 2026-10-12 18:07：方法贡献整理v1（文档，不新增GPU实验）

**方法/理论。** 按用户收窄题目与整理贡献要求，新docs/realq_storage_method_contributions_20261012_v1.md将求逆工作区、已消费H接管、down稳定写回、私有H及回放统计死亡点组织为一个面向REAL-Q消费顺序/存储别名约束的管理方法。补充研究问题、不同存储计数、释放/复用条件、实现映射、已有技术归属和可写贡献草稿；REAL-Q/OLLA/Checkmate/PyTorch主来源核对，references新增REAL-Q v2元数据。不宣称通用最优、首创、五种独立算法或把checkpoint/CE/审查工具列作新增方法。

**效果/评价。** 引用已接受两模型各三对完整容量、17766interval/71064point和0.6B三对时间代价，保留0.6局部+96与原strictFalse、采样及pilot限制。新增document_consistency_and_source_audit.json只核对文档链接、汇总绑定六审查字节及活动61source当前SHA未变，不能替代507/28输出/cache完整重新验收或新增GPU结果。

**反思/后续。** 组合容量不等于各项独立贡献，组件/前缀不能拼成full消融；列出带依赖的必要消融和容量边界补强计划，未自动启动或重复旧失败。1.7B低扰动预定成本继续原串行链，最终成本及报告仍待。不修改活动源码/原status/草稿/失败证据，不并发GPU不推送。


### 2026-10-12 18:19：1.7B首个低扰动成本完整配对

**方法/理论。** 延续同一已接受stable+stats存储方法，预定p1 C→N，数学与算法参数不变，CPU方法观察保留。显存主目标与时间代价分别评价，不据单对建立稳定加速。

**实证/效果。** NEW independent_completed_pair_native-20261012-1818.json实际18:19:03 validTrue，fresh native两臂全部507state/all28非空/KL .625495195388794/PPL25.562191009521484精确，61source及actualentry/13完整extended cache SHA/两stage算法命令/全部inverse-write-dead-stats metadata与参考一致，pair身份顺序和timer原记录精确。C/N quant2498.588322917→2355.394039051秒（候选单对变化-5.7310%），collector4.139627128→4.640291513，sum2502.727950045→2360.034330564秒，ratio1.060462518546。C/NstatusSHA c44388507c4881e3e02294e5fba18a4a96ae6a5c3a7dc7d2b69cca9cf34222f3 / 62cf08a3c00920c6b9e8818d1a819da48f6ecfe65c3cad7c4288717f5df9ef52。

**评价/反思。** 只有q17一对；q06三对平均更慢的完整证据仍保留，不混成两模型稳定加速结论，不剔慢值。collector+quant排除prep/eval/hash非端到端、CPU观察非零开销；timing无新dense遥测，容量仍基于已接受六对17766区间/71064point/global。cal2×1024pilot非正式2048/小GPU/多卡。actual18:18 controller28/reader37/唯一GPU870已q17 p2 candidate第0层，p2 N→C后p3 C→N；旧772退出，不沿用。继续所有完整fresh独立审查及最终报告，不改61活动source，不并发不推送。方法贡献文档18:07已独立整理，论文40多页可按分析/方法/验证规划，但单项消融、正式规模和容量边界仍须补强，篇幅不等于证据充分。


2026-10-12 19:08 只读运行时观察：q17p2 candidate唯一GPU870继续第6层，controller28/reader37，无新completedpair。19:06日志tqdm elapsed48:36与实际ps worker elapsed11:54不一致；19:07 Linuxwallclock及perf_counter/monotonic/uptime独立记录于independent_clock_runtime_observation-20261012-1907.json。过去1h筛选主机电源/时间事件无返回，不证明无暂停/时钟调整，也不能据此确证热/频率/方法成本原因。原timer/慢值保留，不剔除或为改均值重跑，不改变活动源或队列；最终按实际计时scope与环境限制评价。此文件仅环境观察非数学/缓存完整验收，不新增GPU，不推送。


### 2026-10-12 20:17：1.7B第二个低扰动成本完整配对，方向反转

**方法/理论。** 按预定p2 N→C串行验证原简stable+stats工程，保持同校准/FP32求逆/BF16权重/Adam数学不变。生命周期存储优化不保证速度提升，时间代价保留全部原记录。

**实证/效果。** NEW independent_completed_pair_native-20261012-2016.json实际20:17:13 fresh native validTrue，两臂507/all28非空输出/KL .625495195388794/PPL25.562191009521484精确、61source/currententry/13完整extended cache SHA/两stage算法命令/完整inverse-write-dead-stats metadata均通过，pair顺序身份与原timer精确。C/Nquant1759.998356115→2052.422811679秒（候选本对+16.6150%），collector4.102322823→4.552469080，sum1764.100678938→2056.975280759秒，ratio0.857618803415。C/NstatusSHA 6cbe089cb861bf2d5aa828b7ec6c3b3b9684f307c776d7a11a56cfa328006173 / b6dce304485f618d34d9f79c8bae5b1b034701885e003549246e4e037bec0ed2。

**评价/反思。** q17只有两对，一快一慢，不宣称稳定加速或三对最终成本。19:07时钟观察保留；源码核对collector/quant使用CUDA同步的time.monotonic差值，不能以tqdm墙钟elapsed改写原timer或推导暂停原因，不确认唯一热/频率/方法惩罚，不剔慢值或重跑改善均值。CPU方法观察仍留，sum排除prep/eval/hash非端到端；timing无新dense峰证据，已接受容量六对17766interval/71064point及全部global保持原证据。pilot非正式2048/更小GPU/多卡。actual20:16 controller28/reader37/唯一GPU1040为q17 p3 control第0层，旧GPU958/870已结束；p3 C→N后fresh三对六臂summary和两模型成本/最终报告，不因队列进度停止，不改61活动源、不并发不推送。


### 2026-10-12 21:31：当前pilot研究链完整收尾（容量与时间成本）

**方法/理论。** 最终选择原简workspace/消费H复用/down-only稳定写回/末组私有H死亡点/earlystats统一存储方法，保留原FP32求逆、BF16权重、校准/seed/Adam数学。OLLA、Checkmate、REAL-Q与PyTorch2.11来源及创新边界详见NEW docs/realq_system_memory_research_report_20261012_final_v1.md，不将通用生命周期理念或既有checkpoint/CE记为新增算法。

**实证/效果。** q17 p3 C→N于21:20:00完成，NEW independent_completed_pair_native-20261012-2126.json实际21:27:15 validTrue，两臂507state/all28非空/KL .625495195388794/PPL25.562191009521484精确，61source/实际entry/13完整extended cache SHA/两stage命令/完整inverse-write-dead-stats/timer身份顺序通过。C/Nquant1756.192297697→1745.818439124秒、collector4.641346405→4.205734768、sum1760.833644102→1750.024173892；C/NstatusSHA793d20673d206591b511c5dbd9d6e7d4348a424cf7acd0834f40aa270e151edc / 4c46062ac8ace0f8ac9ec262023e7ad279b34a845896e5b6c56d63e48d431290。

NEW q17 independent_three_pair_qwen17_summary-20261012-2128.json及q06 independent_three_pair_qwen06_summary-final-20261012-2129.json分别21:28:38/21:28:46重新六臂完整通过。q17 C量化均值2004.926325576/sampleSD427.528066019，N2051.211763285/sampleSD304.789604457秒，N均值+2.3086%，collector+quant+2.3122%；q06 C970.274532847/SD145.294536509，N1095.378821851/SD137.043360776，N+12.8937%，sum+12.8720%。预定顺序及全部快慢值保留，无剔除。

NEW outputs/realq-final-complete-acceptance-20261012-v1/fresh_memory_six_pair_summary.json再次CPU重新执行六对内存507/all28/KLPPL/source/cache/method/2961区间峰与11844逐point/global验收，累计17766/71064，与原status/cache/全局峰一致；q17局部0、q06累计3区间6point各+96MiB保留旧strictFalse，按用户11:15硬件全局目标容量通过。q17 NVML6629→6089MiB/alloc5605.190918→4883.690918/res5872→5332，q06 NVML4247→4119/alloc3151.033691→3010.087402/res3490→3362。final_capacity_and_cost_summary.json绑定全六对容量、全六对计时及原controller记录SHA，validTrue。复审只CPU读取，没有额外GPU重跑。

**评价/反思。** 两模型三对均无稳定加速证据；显存为主结果，计时分布为成本，均值变化不确认固有惩罚或唯一热/功耗原因。19:07时钟观察保留，CUDA同步time.monotonic原timer不按tqdm墙钟改写。CPU观察仍留，collector+quant排除prep/eval/hash非端到端；NVML采样非连续证明。所有结果仅cal2×1024 pilot，正式2048/实际小GPU/多卡/跨架构及同工作量规范消融未完成。统计持有量非峰降、after_inverse含后续量化刷新，池局部零回升无额外global收益不推广，所有原失败/工具bug/重启中断证据保留。

**完成范围。** 原costv3 controller与GPU均正常退出，status实际validTrue/finished21:20:00；actual ps不沿用28/1122。NEW final报告链接/数字/来源再次核对后关闭本轮real-q自动复查。当前预定研究链与必要完整验收/报告完成，论文扩展研究列待办不冒称已验、不自动另启GPU。原草稿与贡献v1保留当时pending快照，以final报告更新；不推送。

### 2026-10-12 22:15：服务器正式配置入口整理（仅CPU准备）

方法/理论：NEW scripts/run_storage_method_formal.sh、tools/run_storage_method_formal.py及docs/run_storage_method_formal.md将已验收统一存储方法绑定至portable checkpoint入口，不读取历史outputs。rawentry与原已接受entry仅Stage2缓存保护报错文字不同，数学、消费者边界、原helper不变。参考旧formal_realq_compare.sh及本地论文表6：cal2048×2048/seed1/W4A16/4卡、q06LR3e-4/q17LR5e-4/末层1e-5，Adam32样本/cosine/滑窗/P95/P99、默认QA任务。明确两臂group_parallel_quant=none（本方法未接rank分组路径），保留4卡DP/Stage0FSDP，显式native分配器；不能假称旧rank路径及expandable设置不变或论文表逐位复现。

具体方法：默认串行q06 C/N后q17 C/N，各自fresh Stage0+28量化+全测试评估，Stage2拒绝重算缓存。control包含既有checkpoint/CE/输入卸载，candidate只新增workspace/consumeH/stable/downdead/earlystats。独立全rank507/all28/KLPPL/source/完整cache/112inverse-reuse28write-dead-stats、exact interval及逐point/global审查，局部回升保留诊断，各GPU NVML采样峰另核。正式参考值来自同服务器control，不复用pilot canonical。新目录拒绝覆盖，失败停止后续，保留原status/日志；不扫LR，不推送。

效果/评价：WSL CPU5tests通过（正式参数被真实production parser接受、actual源builder、单wrapper、禁用/缺rank/工作区未启用拒绝、算法数据参数变化拒绝），Bash语法及实际shell dry-run四臂通过。NEW outputs/storage-method-formal-cpu-20261012-v3/real_pilot_auditor_cpu_regression.json对既有q06完整pilot读后validTrue，2961interval/11844point及local1interval2point+96完整保留；此为新auditor CPU接入回归，不是新GPU收益。cpu_preparation_status.json绑定新文件SHA，未启动任何formal GPU。较早CPU parser揭示QA应为--lm_eval而非环境变量名--enable_qa_eval，修正后再次真实parser通过，旧CPU计划保留不覆盖。

反思/范围：4卡、旋转、全正式数据和全QA尚需用户在服务器实际验证，脚本准备不作成功或显存比例保证。用户此次明确要求先服务器0.6/1.7正式配置；仅交付方法说明及启动入口，不恢复本机自动实验。正式多卡cache按实际完整文件清单，不硬套单卡13文件。密集profile时间不是低扰动速度，NVML不是连续证明。此前pilot final报告和所有冻结源/负向证据保留。

### 2026-10-13：正式服务器双卡 CPU master post-quant 审查修复

方法/理论：用户提供首个0.6B control日志，Stage2量化至layer27后rank1在完整state哈希`.cpu()`遇到meta张量退出，torchrun终止rank0。原CPU master设计由rank0持有完整CPU模型并做KL/PPL，其他rank释放每层到meta；不应要求meta rank装回全模型来审查。此为正式记录入口接入错误，非该日志中的OOM或方法/数值失败，未完成全评估不能算有效正式pair。quant_stop_layer=27包含第28层，进度27/28不独立证明缺层或完成。

具体方法：新增tools/storage_method_distributed_audit.py与独立tools/run_storage_method_pipeline.py。只改生成entry量化之后的hash块，rank0完整507state原格式字节SHA及28层真实非空输出，所有rank真实布局SHA/name-shape-dtype、meta数量、rank/world_size/所有权；非所属rank明确state_tensors=0/state_sha256=null及layout_only，不广播/复制rank0数值证据。纠正22:15记录“全rank507/all28”不适用于CPU master的要求。普通复制模型模式仍每rank完整数值和评估一致。每rank inverse/write/dead/stats覆盖、fresh缓存、SOURCE/环境/命令、exact区间峰、独立每point及全局显存准入均保留，另核全部rank过程峰身份，owner记录从实际rank0文件读取不取最先打印的rank1日志。旧checkpoint entry、full pipeline runner、量化/helper及旧结果不改。

效果/验证：WSL真实两Gloo CPU进程，用CPU传输替代production CUDA-only广播，调用真实release_layer及实际生成measured_fwrd/held-out hook；原入口在相同meta权重复现NotImplementedError，新入口owner507/all28审查通过且rank1仍meta、无复制模型及虚构trace。完整owner含meta、缺失/重复rank、布局/所有权/world/count/digest错误、缺/空末层输出、伪造非owner字节哈希、非CPU模式缺rank及缺过程峰均拒绝。相关42tests通过（27.20秒），真实pipeline main以CPU worker/telemetry fixture验证owner选择和rank过程峰完整性。双卡四臂正式shell dry-run成功；NEW outputs/storage-method-distributed-audit-cpu-20261013-v1/cpu_preparation_status.json validTrue仅CPU/source。实际entry除了post-quant hash外AST相同；stage launch/profile读取函数、CUDA峰和评估包装相同，11份冻结源与已推送b587245字节相同，真实worker符合Python3.10语法。没有新增CUDA正式实验/显存收益。

评价/反思：此前单进程pilot及compile/parser测试漏掉多卡CPU master权重所有权，应以真实双进程释放和输出包装执行补覆盖。回归初版fixture误用production CUDA-only广播，另一次torch facade漏uint8，修复fixture后全部通过；native Git SHA核对因WSL不能解析Windows worktree gitdir改用Windows原生Python，未修改gitdir/源或伪造结果。CPU修复不能称服务器GPU已通过。旧server未设置save_qmodel_path，Stage0准备checkpoint非已量化模型；进程退出后不能仅补hash恢复，需要新目录fresh重启两臂，原失败目录保留。用户已授权推送，修复交付到mine/realq-warmadam；本机pilot全部正负向及最终报告保持原结论，正式结果仍待服务器完整验收。

### 2026-10-13：正式实验四臂实时总日志接入

方法/具体工程：用户要求沿用一个总日志看全部输出，核对eefa863后发现仅控制器stdout进入nohup总日志，worker仍各写stage0/stage2，先前nohup命令未自动汇总。新增storage_method_log_stream.LiveFileRelay，阶段原stdout/stderr保留后每0.25s转发到arm控制stdout，正式控制器再转发全部arm输出到nohup总日志；PYTHONUNBUFFERED=1，plan标记aggregate_worker_stdout。tqdm回车在总日志转为换行，原阶段文件字节不改，最终退出尾部同样排空，rank/tag/stage原输出保留。NVML CSV/区间JSONL仍独立完整原始工件，不向进度日志灌100ms采样。无数学/配置/模型/缓存/峰值门槛变化，无GPU操作；密集profile非低扰动速度结果。

效果/验证：新增tests/test_storage_method_log_stream.py验证缺失文件等待、增长UTF8跨块/回车、去重/截断拒绝，真实CPU子进程stage stdout+stderr及成功/失败退出尾部，旧四臂plan动态出现stage日志及最终失败status，新aggregate计划防重复汇总。日志、双进程meta审查和正式参数接入27tests通过（31.08秒）；真实Bash双GPU四臂dry-run生成新计划，只CPU，无新正式GPU结果。NEW outputs/storage-method-log-stream-cpu-20261013-v1保存计划/来源快照，原分阶段日志和此前失败/CPU验收全部保留。

评价/反思/在线用法：此前日志仅说明输出路径，不满足用户一份总日志的使用要求，应核对嵌套subprocess的两层重定向而非仅给nohup。新增单文件tools/stream_storage_method_formal.py可对已运行的旧plan只读汇总各arm pipeline/stage0/stage2，读已有输出后继续跟后续模型/阶段，保留身份，实际controller结束status出现才退出；不启动GPU、不重启实验。在线仅git fetch+git show提取reader至/tmp，不pull/修改任何活动绑定源。新正式程序在本次结束后pull供下次运行。用户原授权交付修复到mine/realq-warmadam；pilot结论不变。


### 2026-10-13：缓存接口真实保存格式回归要求

服务器03:14四卡Stage0正常结束，新增缓存检查误将FP32累积dtype要求施加于实际BF16持久化矩阵，误拒layer0；旧测试也用FP32手造缓存。新增contract按生产AST绑定BF16保存表达式，完整检查形状/层/module/rank/有限值/prepared/SHA，错误含实际dtype。回归必须执行生产转换及_to_save映射后真实torch.save，四rank完整文件准入/独立复制字节一致，错误dtype/缺失/损坏等仍拒绝；51项CPU测试通过。保留旧缓存并先CPU完整核验再重启正式任务，不为绕过接口bug重算或改生产精度；CPU通过非正式GPU完整验收。完整记录见主日志同日Fisher缓存准入误拒修复。


### 2026-10-13：共享磁盘输入的独立验收

默认正式auto缓存/模型大文件为shared引用，不再每臂复制；单独source_model元数据仍私有，shared_cache_reference.json记录link与additional_bulk_payload_bytes=0。共享source前后SHA、完整每rank/层/module/有限值、prepared身份/旋转/架构、真实数值与显存原验收继续执行，不能因两臂同源免除质量与method检查。生产加载的safetensors和torch.load普通/mmap内存修改不可写回来源；共享prepared写入和命中后再precompute禁止。61项CPU回归通过，无新GPU结果。日志/profile和新配置首次生成数据仍占空间；原副本不自动清理。

### 2026-10-13：正式控制器 job_index 未赋值修复

失败/归因：用户服务器Stage0生成缓存后，run_storage_method_formal.py剩余实验臂磁盘预检引用job_index，实际--run循环未赋值而UnboundLocalError退出。此前索引误加在--dry-run循环中；旧测试覆盖计划和子流程但没有执行外层--run，属于控制器接入错误，不是GPU OOM、缓存损坏或量化方法失败。原失败日志、缓存和结果保留。

方法/验证：实际运行循环改为enumerate(jobs)，dry-run恢复普通循环；数学、校准、模型、共享缓存和验收规则均不变。新增真实main --run的四项CPU调度回归，覆盖auto缓存命中/成功预计算及shared/copy，两个模型各三对、12臂、每模型remaining_arms=6和交替顺序。修复前在实际主循环复现原异常（1 failed），修复后相关65项CPU回归全部通过，46.09秒。进程/缓存/质量结果由CPU fixture模拟只验证调度，不能算正式GPU成功；meta所有权、缓存原生产保存schema及日志仍由相关独立测试覆盖。输出保存outputs/formal-controller-job-index-repair-20261013-v1，不覆盖旧证据。

评价/交付：此次遗漏应由外层控制器执行测试捕获，不能仅凭dry-run/compile宣布正式流程可靠。用户授权本地推送后服务器拉取；只在实际无实验进程时更新，不另启GPU、不清理新生成缓存。服务器SSH端口目前出现banner连接拒绝/超时，尚未完成拉取，后续以实际连接和HEAD核对为准。