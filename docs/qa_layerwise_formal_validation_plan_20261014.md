# QA 存储方法：服务器恢复后的最小验收链

本文件是验收计划，不是已完成的正式实验报告。服务器当前由用户关闭，今晚不连接或启动服务器；新 helper 尚未替换默认正式启动脚本，也未推送。

## 方法与保持条件

采用已接受的 scored-row 归一化寿命管理、逐层 CPU 权重放置和官方 DynamicCache 卸载组合。模型前向仍使用完整 GPU GEMM、BF16、SDPA、完整词表、batch16 和原 `use_cache=True`。不改变原请求、padding、continuation、评分公式或任务集合。

分层放置必须在原 `distribute_model` 添加 accelerate hooks **之前**选择。对照委托原设备分配，候选执行单 GPU 分层放置；不能在已有 dispatch hooks 上直接叠加原型。记录所有可见 GPU，即使候选主要使用一张卡。这个放置差异也必须在报告中说明，不能仅称“同样的多卡分配”。

完整 logits 仍在 GPU 上生成。关闭 KV 和 256 行切块 LM-head 已因真实数值不一致拒绝，不重新使用。流式 CPU 指纹仅减少观测开销，不改变前向张量或评分。

## 已完成证据与缺口

- 原模型两模型 context128/256 四臂组件已完整独立验收，311 state、完整 logits、28 层输出及全部分数精确。
- 两模型实际最长输入1235-token的候选单臂已可运行；没有相同长度原对照，不提供该范围的下降比例。
- online rotation wrapper 535-state fixture 已完整配对；基权重未旋转，且本机使用 torch_fallback。这不证明正确 REAL-Q 量化模型或服务器 FHT 后端的质量。
- 分层放置桥、完整流式字节观察、任务边界、各可见设备区间统计均有 CPU 准备证据。真实本地 CUDA transient 探针只验证单设备计数器。
- 原服务器完整十 QA 验证曾被关闭中断。恢复后先读取实际工件与源 SHA，完整成功臂保留；只恢复缺失部分，不能补造 status 或从参考复制分数。

## 顺序与准入

1. **只读来源盘点。** 记录 actual commit、Python/PyTorch/Transformers/lm-eval/FHT 版本与实际文件 SHA、GPU UUID/可见设备映射、模型配置/全部模型文件 SHA、现有完整与中断结果。核对实际可用 RAM/配额，不用宿主机文件系统总体空闲量替代容器配额。
2. **运行时接口短组件。** 用实际安装后端、真实旋转包装/量化状态验证选定方法。完整535 state前后、28 Had buffer 值与布局、Parameter身份与共享关系、每次28层非空输出、全部 full logits、每个 logprob/greedy 都要核对。任何精确性失败立即保存拒绝，不放宽容差。
3. **统计接口验收。** QA 区间统计独占 peak-counter reset。所有 visible devices 的 phase 必须带设备身份；每个 stage/logkind/rank/logical-phase/occurrence 都保留。独立审查区间真实 allocator maxima 和每个 start/end 点。每设备原 process maxima与QA累计 maxima显式合并到 footer；不能因重置计数器丢失既有峰。另保留每 GPU UUID 的 NVML 采样。
4. **同一正式模型完整 QA 配对。** 保持原十任务、batch16、原 tokenizer/请求与所有评分；不设 limit/samples。自动边界包装仅委托原 `simple_evaluate`。完整输入/层/logits/评分事件逐行落盘，使用16MiB CPU字节暂存；缺任务、缺 forward、缺层、NaN 或变 state 均拒绝。原准确率汇总一致不能代替完整数值。
5. **独立容量与全流程报告。** 数值、方法覆盖、来源/cache、所有设备统计完整后，才按全流程 allocated/reserved/NVML 三峰判定容量。保留全部局部回升诊断。QA改进后重新报告Stage0、量化、KL/PPL、QA各阶段和全流程主导峰；不得把QA阶段下降比例当全流程下降比例。

两模型顺序串行。验收失败先区分数学、记录接入、统计覆盖、资源或基础设施错误，原证据全部保留；不重复已完成失败方法。NVML是采样证据，不能声称连续每刻峰值证明。实际小显存卡是否能跑、正式2048×2048全流程、真实多卡主机容量仍分别需要实际运行。

## 接入时需要核对的代码

`qa_layerwise_placement_bridge.py` 负责选择放置与 finally 恢复；`qa_complete_observer.py` 负责完整字节/任务/forward证据；`qa_task_boundary_observer.py` 负责原任务委托；`qa_visible_device_phase_profile.py` 负责具有设备身份的 exact 区间。四者目前不是已经验收的正式 launcher。

生成的方法需分别记录外层 no_grad 包装与实际计算函数。动态 AST 函数没有实际源码文件时，sourceSHA应为null，记录实际 codeSHA/编译文件名，并另绑定原库源码、变换 helper 和生成入口；不能把外包装源码误标为实际计算源码。

后续默认脚本启用前，还需把上述来源、数值、实际方法、任务集合、完整设备统计与最终 footer 合并检查连成显式的 launch/acceptance gate。CPU tests、未旋转 wrapper fixture 或单卡探针均不能单独使正式 gate 通过。

04:06接入进展：qa_observed_placement_bridge.py现在已组合放置、完整字节/任务观测、全visible phase和PeakCarry累计。PeakCarry合并原CPU整数_process_profile_peaks表，原getter及已导入alias不替换。真实native dispatch对照/组合候选的两模型短GPUfixture均完整独立通过（433区间/1732逐point），并与此前accepted完整事件/selected方法code独立精确核对；仍是未旋转基权重、torch_fallback、单synthetic任务、实际1visible，formal_enabled=False。完整十QA/正确旋转量化状态/服务器FHT/实际多卡源和遥测准入尚未完成，不能直接将该prepared interface启用为正式默认launcher。

## 06:34 控制头原设备委托与正式准入边界

NEW qa_control_head_observer.py 可纯观察原 Linear 或实际安装的非offloading AlignDevicesHook/SequentialHook；验证官方partial/new_forward委托、_old_forward原Linear、实例方法/源码SHA、hook配置与Parameter身份，finally仅移除自己的PyTorch观察handles。它不剥离或改写原Accelerate放置。映射候选仍在dispatch之前选择分层放置，仍要求原Linear，不能与控制组设备hooks叠加。

真实CPU与单CUDA控制hook兼容不替代多卡：服务器恢复时必须检查实际head hook类型/是否offloading/执行设备与I/O回送设备、共享Parameter和实际库SHA，保存完整native-control dispatch与单CUDA候选的放置差异。若原head是offloading或其他wrapper，本版明确拒绝，不能降低guard以启动。所有visible设备的源/身份、exact区间/每point、累计process峰、NVML与主机成本仍须足额采集。

首q06完整组件原parent False属于独立SHA构造的接入问题，CPU重现同code对象实例化前后marshal摘要不同；新v2审查读取实际官方函数状态后fresh全审True，原False文件不变。这个修复不改变模型、评分或统计门槛。q17在同一新接口串行续验，正式default仍不启用。

## 2026-10-14 06:35 两模型官方控制头hook兼容完整完成

q17 qa-control-align-qwen17-20261014-v1 原parent status True，NEW independent_completed_audit_v2-20261014-v1.json fresh True。q06同名根的NEW v2独立审查True、原parent False保留且已CPU复现SHA构造归因。两模型完整535 state/28Had12×12FP32、实际input/full logits/28层4forward/65scoregreedy、原Parameter/官方CUDA AlignDevicesHook委托与hook身份、原native score/rows+官方cache实际selected code/source/library、private/native/callback与每批pool释放全部精确；各593interval/2372逐start-end全0回升缺失，累计1186/4744。NVML3359→637与5553→1079MiB，native allocated/reserved等此前接受组合（q06 2908462592→970141696、3397386240→1124073472B；q17 5158036992→1286125568、5697961984→1587544064B）。这是新控制观察接口兼容，不是新GPU算法收益；mapped candidate未叠accelerate。

会话43399、32216、83940及全部worker/CPU审查已退出，06:34actual ps无研究CPU/GPU，旧17/105/108等不再沿用。所有新旧helper/source/so继续冻结；没有为同已完成失败配置重跑GPU，旧q06 parent False/source/log完整保留。CPU回归7+1审查修正+1entry共准备完成，非多卡证明。

评价/后续：两模型实际官方单CUDA控制头接入缺口已补；真实多卡head offloading/跨设备I/O、正确REAL-Q旋转量化535、实际服务器FHT/allocator ABI、完整十QA与完整source/cache/multidevice/host admission仍未验证，formal默认仍未启用。正式profile若已有peak-counter owner，不能与新QA group同时reset；需显式交接并保留外层区间峰和原process maxima，不能只依赖累计footer掩盖外层interval遗漏。后续以CPU正式来源/任务/统计owner准入与08:00完整汇报为重点，已接受组件无需反复GPU回放追局部或速度值。

## 2026-10-14 06:47 正式源码正常交接核对与CPU计数器准入

文献/理论：读取当前实际 ptq.py、gptq_fwrd、CudaPhaseMemoryProfiler._close 和正式 pipeline EVALUATION_PROFILE_SETUP。当前量化正常 return 前 finish(quantization_complete)，KL/PPL包装在normal/exception两条路径finish，native close finally关闭文件并phase=None，ptq在原rank0 dispatch后调用原QA。当前源码未发现QA与上述native exact owner重叠；先前“若有外层owner会漏峰”是接入条件风险，不能误写为已观测正式漏峰。

具体方法：NEW qa_peak_counter_admission.py 在QA前后纯CPU检查已审查的native exact及VisibleDevicePhases实例状态、显式manifest绑定SHA和独占lease，active owner/重复lease/来源变化直接拒绝并保存False。它不query/sync/reset CUDA，不将full-state/tensor保存到记录；GC快照只在入/出准入期间短暂存在，输出仅CPU primitive元数据。NEW qa_mapped_observed_placement_bridge_v3.py只围绕已接受v2桥加lease，validation_only=True；不是默认formal安装。候选math/GEMM/KV/batch/放置与原峰累计逻辑不改。

效果：CPU5passed13.78s（qa-counter-admission-cpu-20261014-v1），包含已活动native区间/QAgroup拒绝、source坏SHA/嵌套lease/非validation拒绝、原异常保留与失败退出状态、真实HFLM两fixture任务完整state/input/28layers/full logits/scoregreedy/原footer精确。四fake-device只是CPU计数协议，不称真实4卡。GC isinstance触发已安装torch.distributed.reduce_op的第三方弃用warning，10条原日志保留；不隐藏为零warning或GPU失败，正式部署前仍需实际runtime范围审查。

NEW audit_qa_formal_source_handoff.py只读AST源码契约：原quant→KL→QA顺序、原QA参数与rank0-dispatch、native normal finish/inactive、KL normal/exception finish、完整十任务与无limit/samples、配置batch原传递。actual5source SHA齐全、source_contract.json True，CPU3passed1.69s含删除finish/改QA参数/删任务/加limit等拒绝。它不是任意控制流证明或GPU数值/容量/来源全部准入。两CPU根均有status、源快照及源/logSHA，原GPU绑定源码未编辑、没有启动新GPU。

评价/反思：准入策略选择“先结束前一个exact owner再独占QA”，不在仍活动的外层区间内reset，也不靠footer的累计值掩盖区间漏峰。它覆盖明确已审查的两类profiler，不宣称发现任意第三方直接CUDA reset；完整formal实际源/运行时链和所有设备覆盖仍待。两模型控制hookGPU接受证据不变，默认formal未启用，服务器仍关闭/noSSH/noPush。08:00草稿已补实际正常交接与CPU边界，后续必要来源/任务/主机成本准备及最后完整报告继续。

## 2026-10-14 07:01 主机存储成本观测准备（CPU，无新GPU）

文献/理论：官方Linux /proc文档说明PSS按共享映射比例分摊，smaps_rollup汇总每进程映射字段；RSS树相加可能重复共享页，PSS仍不等同系统总用量/连续最大，hugetlb与部分大页记账有边界。cgroup-v2 memory.current/max用于容器范围，root/隐藏层级或不可读字段必须明确unavailable，不能填0或推定服务器80G。文献已补bib linuxProcPss2026/linuxCgroupV2Memory2026，链接 https://docs.kernel.org/filesystems/proc.html 和 https://docs.kernel.org/admin-guide/cgroup-v2.html。

具体方法：NEW qa_host_memory_observer.py按actual PID+start_ticks扫描树，逐进程只读smaps_rollup RSS/PSS/shared/private/swap/Locked和status VmLck/VmPin，源/采样SHA、缺失/权限/身份/拓扑变化完整保存，缺PSS不拿RSS代填；meminfo与cgroup-v2目标/可见祖先limit独立记录。NEW v2增加boot/kernel身份和1MiB流式日志摘要，避免监测大JSONL自身增加大块RAM；NEW v3将根存活身份改为PID+start_ticks/boot，PPID仅用于拓扑，正常nohup父退出不视为任务结束。计数为顺序采样，非同一时刻树快照/总host/PSS连续峰，不导入CUDA/model，不触碰QA math。

效果/验证：CPU v1 3passed1.32s，含真实Linux两进程32MiB共享页RSS重复/PSS分摊、smaps缺失/单位/重用和现有输出拒绝。v2 2passed1.27s，包含强制禁止整文件read_bytes仍完整SHA相同、boot变更拒绝。v2 standalone首次缺仓库sys.path而ModuleNotFoundError，原CPUTrue/CLI失败日志保留；NEW run_qa_host_memory_observer.py只补CLIbootstrap/源绑定，真实外部cwd及不存在PID失败不补样本2passed2.81s。NEW run_qa_host_memory_observer_v2.py接稳定根身份v3，3passed1.46s含真实Linuxlauncher自然退出、worker reparent仍complete、start_ticks改动拒绝和actual独立CLI。最终standalone4/4完整样本、源/祖先SHA不变，只是CPU空载能力，不是QA PSS数据。

证据：outputs/qa-host-pss-cpu-20261014-v1/v2、qa-host-pss-cli-cpu-20261014-v1、qa-host-pss-reparent-cpu-20261014-v1，均有独立status/源快照/logSHA。只有v2原直接CLI失败False保留；后续用NEW run_qa_host_memory_observer_v2.py，不改旧helper/source。当前WSL23GiB/free22GiB只是actual07点空载环境，不可替代服务器容量。缺root cgroup memory.max在当前WSL明确记录unavailable；Locked/VmPin字段不替代CUDA callback pinned-host计数。

评价/反思：之前长QA只有singleworker sampled RSS，不能从新空载/共享页fixture反推旧实验PSS或将旧服务器343GB RSS视为物理需求。服务器恢复后需同时记录root-tree RSS/PSS、cgroup/system范围及mapped callback，仍保持完整CUDA native/private/NVML与每区间/逐point审查。首CLI缺路径和PPID不稳定说明必须真执行入口及真实进程生命周期，不能只compile/CPUimport。所有新旧源冻结，无QA GPU重跑/noSSH/noPush；08:00报告继续保留物理显存收益、HOST转移成本、失败与正式未验范围。

## 2026-10-14 07:12 物件字节预算与实际范围核对（CPU）

理论/文献：依据原Qwen3 model config与已有真实GPU full-logits/KV几何、BF16元素2B做物件字节算术；full logits=实际B×L×完整V×2，单层K+V=2×B×L×KVheads×head_dim×2，28层逻辑KV乘28。此处包含全部KV逻辑对象，不等同CPU驻留部分/allocator峰/NVML峰，也不包含权重、其他激活、传输、池对齐及CPU观察；不能作为硬件准入。

具体方法：NEW qa_storage_payload_budget.py只读已有long两模型C/N三次实际输出/head payload/KV几何，与绑定config SHA核对；precision/少KV/错误head字节/配置batch变化拒绝。实际工件和原独立审查SHA明确链接，标记不是fresh full re-audit、没有新GPU验收。CPU2passed0.76s和实际独立CLI两模型几何全同，outputs/qa-storage-payload-budget-cpu-20261014-v1含status/源快照/logSHA。

效果：两模型实际16×1235 full logits6004510720B（约6.0045GB），全部28层K+V逻辑2266234880B（约2.2662GB）；其余两次实际1016/845对应logits4939743232/4108349440B、KV1864368128/1550581760B。已测callback pinned6006243328B是池段持有量，和logits物件相差对齐等字节，不是NVML削减量。两个模型V=151936/KVheads8/head_dim128/layers28相同，头输出与KV物件字节相同，但hidden/权重和实测峰不同。

评价/假设范围：假设配置B16、L2048，full-logits物件9957277696B、全部KV3758096384B；仅CPU算术，没有执行该长度GPU或正式QA。model config上限40960的假设输出199145553920B/KV75161927680B也只用于展示长度代价，不是正式任务实际最大/已验证可行性或OOM结论。当前完整配对范围最长实际1235，不能把名义1270等同跑满1270，不能把calibration seq2048等同QA请求长度分布。

反思/接续：mapped改变输出存储驻留位置，仍保留完整CUDA GEMM及原数学，不能消除完整对象所需的主机空间。服务器恢复应先保存真实十任务经tokenizer/padding/grouping后的实际shape，再结合进程树PSS/cgroup/原callback与三GPU全局峰验收；不能只用配置最大长度或此预算替代真实容量。当前无GPU/无后台研究CPU，旧绑定源未改/noSSH/noPush，08:00报告已补物件与峰的区别。

## 2026-10-14 07:25 当晚最终CPU独立复核完成（无GPU重放）

理论/方法：最终报告使用完整工件重新审查，而不是读取旧valid标签或再跑已接受GPU配置。NEW audit_qa_night_final_chain.py串行在CUDA隐藏的fresh进程中调用冻结的四类独立auditor；native组合包含实际selected方法审查，control-align使用已CPU归因修正的v2实际官方函数摘要。每类两模型共8组配对工件，全数值/输入/311或535 state/28层/full logits/全部scoregreedy/KV/实际方法/模型库源SHA/私有池账、真实区间峰、各独立start/end点和三global门槛重新核对。它们有方法继承关系，不是8次新独立GPU重复实验。

效果/证据：outputs/qa-night-final-independent-20261014-v1/status.json True，8组全部通过，合计4058interval/16232独立points；长输入362/1448、自然尾批481/1924、native组合及官方control-hook各593/2372，每模型同数。NVML长输入0.6B7047→1415、1.7B7907→2363MiB；自然尾批1245→637/1687→1079；native/control-hook3359→637/5553→1079，和原接受结果一致。0.6B长输入两个局部allocated区间微升仍原样保留strict_local_nonincrease=False，三global通过，逐point无回升缺失。所有冻结auditor/source SHA在完整链前后相同。新dispatcher CPU2passed0.79s，独立根qa-night-final-chain-cpu-20261014-v1含源快照/logSHA/chain statusSHA。

负向/评价：首版单池长q06也fresh完整复核，valid=False，仅global_reserved失败（+96MiB），356interval/1424point留在独立拒绝记录，未纳入4058/16232通过数。它的原结果及control-hook q06原parent False均保存不改，control-hook新独立v2 True明确属于审查摘要构造修正，不是GPU重跑或数值容差放宽。首版单池失败NVML也低，说明不能只看物理采样下降而忽略native逻辑资源门槛。

来源收尾：outputs/qa-night-final-cpu-receipts-20261014-v1.json重新核对9个CPU准备根共69条已记录源/日志/工件SHA，全部未变；旧HOST v2 standalone False保持。它是来源收据，不是重新执行CPU测试/QA、不是正式全源准入。实际会话29437退出0，所有研究CPU/GPU已结束（不沿用19/200等PID）。无SSH/推送/模型复制，没有启用默认formal。

范围/反思：最长配对actual1235 synthetic、raw311或未旋转base的535包装、本地torch_fallback及1visible；不能把这些结果换算服务器19GB正式QA、完整十任务、正确REAL-Q旋转量化/FHT、真实多卡或80G主机容量。约6.006GB callback pinned和更高singleworker RSS、较慢诊断wall完整保留。08:00报告已补最终审查和恢复最小链；余下按期限整理报告，不重复已接受GPU组件或已完成失败配置。

## 2026-10-14 07:59 当晚报告已定稿，夜间本地工作完成

08:00汇报用 docs/qa_memory_overnight_report_20261014.md 于北京时间07:58:48定稿，完整保留方法、两模型各范围的三global统计、主机与诊断墙钟代价、拒绝方法及正式未验边界。outputs/qa-night-report-delivery-20261014-v1 保存定稿前草稿、最终报告快照及status/SHA收据；定稿工件完整性核对True，8组新独立audit/各日志/原parent status及冻结auditor源码SHA均未变，所有本地引用存在，单池资源失败和q06hook原parent False保持。此收尾没有重跑GPU或新的full audit，不增加此前组件验收范围。

当晚本地研究目标完成，报告不称正式QA最终容量已通过：最长actual1235 synthetic、raw311/未旋转535、torch_fallback/1visible；完整十任务、正确REAL-Q旋转量化/服务器FHT/allocator ABI、多卡与正式主机成本仍须恢复后验收。继续保留约6.006GB pinned段与更高worker RSS/更慢诊断墙钟、native逻辑存储计数和NVML物理采样差异，不把约1.4/2.3GiB本地峰换算服务器原19GB。当前无研究CPU/GPU，服务器仍关闭/noSSH/noPush/privatePDF未stage，正式默认未启用；不重复已接受或已完成失败GPU配置。
