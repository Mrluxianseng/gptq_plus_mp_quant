# QA 完整输出的存储位置与生命周期优化

截至北京时间 2026-10-14 05:25，本地两模型长输入配对均完整独立通过。方法保留配置 batch16、BF16、SDPA、完整词表、原 use_cache 与评分数学，让原整块 GPU LM-head GEMM 的输出存储由映射的 pinned 主机内存承载。输出仍是 CUDA Tensor；每批评分最后一个消费者结束后，检查私有池 active0，再释放池与主机映射段。下一批重新建立池。

## 已测结果

下表的对照已使用 scored-row 归一化、逐层 CPU 权重放置和官方 DynamicCache 卸载；下降表示在该已优化基线上叠加映射输出。NVML 为设备物理显存采样，单位 MiB，不是连续每刻最大值的证明。

| 配对组件 | 实际最长输入 | 对照 NVML | 候选 NVML | native allocated | native reserved |
|---|---:|---:|---:|---|---|
| 0.6B 短 online-wrapper fixture | 131 | 1243 | 635 | 两臂 970141696 B | 两臂 1166016512 B |
| 1.7B 短 online-wrapper fixture | 131 | 1685 | 1077 | 两臂 1286125568 B | 两臂 1629487104 B |
| 0.6B raw 长组件，逐批释放池 | 1235 | 7047 | 1415 | 两臂 6446786048 B | 7251951616 → 7124025344 B |
| 1.7B raw 长组件，逐批释放池 | 1235 | 7907 | 2363 | 两臂 6798093824 B | 8346664960 → 8218738688 B |

短组件基权重未旋转，仅具有真实 online wrapper，535 项 state，后端为本机 torch_fallback；长组件保持原 raw 模型 311 项 state，实际三次 forward 长度为 1235/1016/845。不同组件不能混成同一正式 workload 的重复验证或统一降幅。均为 synthetic token 请求，尚非完整十项 QA 或正确 REAL-Q 量化模型的准确率验收。

CUDA allocator 的 allocated/reserved **包括这部分 host-backed CUDA 存储的逻辑计数**。因此不能把它们全解读为物理 VRAM，也不能将私有池漏统计包装成节省。每个 head 输出都复核 private active/reserved、native snapshot 总计与地址包含、callback hostlive；映射池包含在 native 账内。

## 主机成本与运行代价

长组件两模型的 callback pinned 段峰都是 6006243328 B。第一批完整 logits 的对象量为 6004510720 B，不能把对象量当作显存峰下降量。每批都在输出所有消费者结束后检查 active0/hostlive0，三批实际 alloc/free 为 3/3。

| 长组件 | 对照 sampled worker RSS | 候选 sampled worker RSS | 插桩评分 wall 秒，对照 → 候选 |
|---|---:|---:|---:|
| 0.6B | 8768528384 B | 13670653952 B | 21.5566 → 31.0451 |
| 1.7B | 12303314944 B | 17140813824 B | 30.8068 → 42.8435 |

RSS 每 0.5 秒查询一次单 worker，包含模型载入与完整字节指纹开销，不是 total host/PSS、连续最大或服务器 80G 容量保证。计时含密集同步、区间统计、完整字节摘要、池快照与 pinned 分配/传输；它是成本诊断，不支持稳定速度或端到端时间结论。

## 完整数值与统计证据

长组件每模型都有新 C/N 两臂、parent 完成文件以及 CUDA 隐藏的 fresh 独立审查：全部 311 state 前后、完整 logits、28 层每次非空输出、全部 64 logprob/greedy、原请求 SHA、原 Linear forward、Parameter 身份、实际 K/V 几何与方法/源/模型/library SHA 精确一致，还逐字段等于此前同输入成功 raw 长参考。

每模型 362 个真实 exact 区间、1448 个独立 start/end 点均完整无缺失；0.6B 有两个局部 allocated 区间峰小幅回升（786432 B 与 721408 B），每点无回升，三全局峰通过。1.7B 区间和每点均无回升。局部诊断保留，按用户已确认的全局硬件容量口径接受。

证据目录为 `outputs/qa-mapped-long-batch-qwen06-ctx1270-20261014-v1` 与 `outputs/qa-mapped-long-batch-qwen17-ctx1270-20261014-v1`，各 `independent_completed_audit-20261014-v1.json` 为独立完整审核。`tools/summarize_qa_mapped_long_two_models.py` 另作两模型 fresh 全审查，`outputs/qa-mapped-long-two-model-summary-20261014-v1.json` 已实际完成 True，累计724区间/2896独立点，完整源与工件 SHA 再核对通过。

## 负向证据与方法选择

长输入首版保留同一映射池至三批结束，0.6B 全部数值精确、NVML7047→1415，但全局 native reserved 增加96MiB，原 `qa-mapped-long-qwen06-ctx1270-20261014-v1` 保留为拒绝。最大 reserved 出现在第三批，第一批大段留在池中与后续计算重叠。逐批最后消费者释放是有证据的生命周期更改，原源码/门槛/失败没有被覆盖或改判。

关闭 KV 与 256 行 BF16 LM-head 切块曾出现精确数值差异，均拒绝且未重跑。当前方案仍执行原完整 GEMM，不改变矩阵形状、精度、词表或评分；没有使用 UVM、CPU GEMM 或反复 empty_cache。

## 正式接入尚需完成

自然尾批 adapter `qa_mapped_head_natural_batch.py` 已有五项 CPU 准备测试：配置 batch16 下实际 1/7/16 input 原 Linear bytes 相同；实际 HFLM 与 28 层小模型 17 个请求产生 16/1 两批，完整 state/layer/logits/17scores 精确。配置8、非法0/17 input拒绝。真实 Qwen CUDA 自然尾批、服务器 runtime/allocator ABI/FHT、正确旋转量化535 state、完整十 QA、全部可见 GPU 的峰累计与 NVML仍待；默认正式 launcher 未启用此原型。

服务器保持用户关闭状态。本地新结果不能直接换算服务器原19GB完整QA峰或正式2048×2048全流程。恢复后的最小验收顺序见 `qa_layerwise_formal_validation_plan_20261014.md`；今晚08:00报告将同时列正向、失败、主机代价与未验范围。

CUDA 映射存储与私有池理论依据：[CUDA12.8.1 runtime memory API](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/group__CUDART__MEMORY.html)、[PyTorch2.11 MemPool](https://docs.pytorch.org/docs/2.11/notes/cuda.html)、[NVIDIA WSL约束](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。本机多GB pinned 已在上述明确范围实测，仍不能外推任意容量或服务器条件。

## 06:14接口续验

自然尾批已在两模型真实 CUDA 完整通过：新65请求保留原64并增加一个unique，configured16，实际16/16/16/1，最后输入1×3，全部311 state/28层4forward/full logits/65scores与真实KV精确。各481区间/1924逐点无回升缺失，best对照的NVML0.6B1245→637、1.7B1687→1079MiB；allocated相同，reserved分别1168113664→1124073472、1631584256→1587544064B，均少42MiB，四批pool active0/hostlive0后alloc-free4/4。原先“自然尾批只有CPU准备”保留为05:25状态，本节覆盖其pending，仍未完成真实服务器QA自然请求分布。

`qa_mapped_observed_placement_bridge.py`已在原native-control和完整观测接口中进行两模型单卡GPU复核。全535/28Had、实际输入SHA、每次28层和full logits/65scores、源/方法/KV/Param、593区间/2372逐点/peak carry全部True，C原native score完整归一化仍保留，N才组合rows/layerwise/cache/mapped。NVML0.6B3359→637、1.7B5553→1079MiB；这是原native对照与完整存储组合的比较，不等同best基线上单独mapped的增量。base未旋转，torch_fallback，本机1visible，不是正式REAL-Q质量。

新完整两模型fresh汇总 `outputs/qa-mapped-observed-two-model-summary-20261014-v1.json`实际06:06:01 True，1186区间/4744独立点。各根的`independent_selected_method_audit-20261014-v1.json`进一步独立对实际outer/selected函数、已安装和最终score codeSHA、原model_call/official cache方法身份全部True。原source/helper/so冻结。

早期运行时入口 `qa_mapped_early_runtime_entry.py` CPU3passed，保留真实formal源全部原AST，只在原QA rank0、Stage2、lm_eval场景提前设置已准入binary的mapHost runtime；非rank0/Stage0不新增context。它仅validation-only准备，未接到defaultformal。当前头检查仍拒绝实际accelerate多卡head-hook，需要新版本控制观察兼容与服务器足额验证。

逻辑 native 峰已包含host-backed存储，不能将其最大phase直接归为物理NVML峰；当前未精确对齐NVML与phase时钟，保留该归因限制。继续准备正式恢复验收链和08:00报告，服务器仍保持关闭。

## 2026-10-14 06:31 控制头观察首个GPU完成与独立审查修正

q06 NEW qa-control-align-qwen06-20261014-v1 两臂完整：全部535 state/28Had/full logits/28层4forward/65scoresgreedy/实际输入/KV/Parameter/方法/source、593interval/2372独立start-end和global均通过；实际官方CUDA AlignDevicesHook的forward/hook/Parameter身份保留。原parent status False只来自独立预期forward_code_SHA取了尚未实例化的嵌套code常量，其他字段全部True，原status/日志/worker/source不改。

NEW CPU serialization_diagnosis.json实际复现同一个code对象在函数实例化前后marshal摘要7ba25ecf…→8957f6e9…变化，实例化后的当前forward SHA正好等于worker记录，原Linear及Accelerate文件SHA相同。NEW audit_qa_control_align_completed_v2.py只将这一个expected摘要表达式改为实际已安装官方CPU-hook函数摘要；所有其他audit AST、完整数值/覆盖/source/区间/point/global门槛不变，CPU回归1passed10.69s。独立fresh independent_completed_audit_v2-20261014-v1.json实际True，NVML3359→637MiB，native峰同此前接受组合；没有GPU重跑、没有把旧False覆盖为True。此结果是新控制接口兼容证明，不是新算法或正式多卡收益。

NEW run_qa_control_align_component_v2.py仅接修正auditor并补来源绑定，CPU实际entry1passed1.70s；不改worker/math/配置。q06独立True和GPUidle后已会话83940串行接q17 NEW qa-control-align-qwen17-20261014-v1，日志同名，尚待实际完整结果与fresh v2独立审查。所有helper/原源已绑定冻结，不能沿用旧GPU105；每次actualps为准。

反思：直接对marshal code常量取摘要不保证等同函数实例化后的摘要；同一源码/同一code对象也必须匹配实际执行对象状态。保留失败并在CPU复现后修审查器，不能为记录接入错误重新消耗GPU或放宽数值。NEW docs/qa_memory_overnight_report_20261014.md已整理08:00草稿，正式server十QA/FHT/正确量化535/multiGPU仍待，defaultformal未启用，不SSH或推送。

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
