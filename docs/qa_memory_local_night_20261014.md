# 2026-10-14 本地QA显存研究

当前已接受的是原始BF16模型的有限长度评分组件，正式量化模型全QA尚未验收。服务器按用户要求保持关闭；本地使用RTX4060 Laptop 8GiB，WSL Ubuntu-24.04。

## 方法

固定batch16、BF16、SDPA、完整词表和原use_cache=True。四个独立进程分别验证原评分、仅continuation行log_softmax与logits最后消费者释放、逐层CPU权重加载/卸载，以及官方DynamicCache CPU卸载。保留原前向计算和Parameter身份，禁止与已有accelerate dispatch直接叠加。模型和数据不复制到实验目录。

KV卸载保留原K/V计算；官方实现末层之后会预取第0层，所以返回时第0层在CUDA、1–27在CPU。关闭KV的旧方法因62/64分数改变已拒绝，不采用。

参考：[Transformers v4.57.1 KV cache](https://huggingface.co/docs/transformers/v4.57.1/en/kv_cache)。实际安装的Cache/Qwen3/HFLM源码SHA随实验绑定，官方说明不能替代实测。

## 已完成效果

下表是NVML采样峰，单位MiB；不是连续每刻证明。输入为32组synthetic token上下文、两choices共64requests，continuation长度1或4，含padding与重复context，实际输入可能比名义context长3token。

| 模型/名义context | 原评分 | 行归一化 | 再分层权重 | 再卸载KV |
|---|---:|---:|---:|---:|
| Qwen3-0.6B / 128 | 3363 | 2243 | 1433 | 1243 |
| Qwen3-1.7B / 128 | 5551 | 4431 | 1865 | 1685 |
| Qwen3-0.6B / 256 | 5213 | 3051 | 2247 | 1831 |
| Qwen3-1.7B / 256 | 7473 | 5311 | 2755 | 2379 |

每组全部原模型311state前后字节SHA、28层各3次完整非空输出、3次全词表logits、64项分数/greedy逐位相同，实际28×3权重记录覆盖完整。每候选349exact区间和1396独立start/end point完整无回升；global allocated/reserved/NVML均下降，来源/模型文件SHA一致。v2同时直接观察K/V的device/shape/dtype/stride，同一物理GPU UUID通过独立核对。

证据目录：

- `outputs/qa-layerwise-local-qwen06-ctx128-20261014-v1/independent_complete_key_audit.json`
- `outputs/qa-layerwise-local-qwen17-ctx128-20261014-v2/independent_complete_kv_audit.json`
- `outputs/qa-layerwise-local-qwen06-ctx256-20261014-v2/independent_complete_kv_audit.json`
- `outputs/qa-layerwise-local-qwen17-ctx256-20261014-v2/independent_complete_kv_audit.json`

首组v1错误要求所有KV在CPU，原summary False保留。独立审查按真实循环预取源码纠正这一测量判据；v1只直接观察keys，不能假称values已实测。后续v2直接观察两者。增强审查CPU3项拒绝测试通过；真实28层小Qwen CPU3项通过。

## 今晚后续

`run_qa_lm_head_night_followup.py`只在1.7B ctx256完整独立审查True后，串行两模型ctx128的新LM-head配对。control使用上述已接受组合，candidate尝试BF16原Linear每256行GPU计算、完整logits即时移CPU、选continuation行后回原GPU评分。完整logits/28层/评分/原state/实际head分块与设备/KV/source/区间/point/global全部通过才下一模型。总日志`logs/qa-lmhead-night-followup-20261014-v1.log`，root `outputs/qa-lmhead-night-followup-20261014-v1`。

GEMM切块可能改变舍入。初始CPU FP32完整logits不一致已保存失败源及日志，新helper明确拒绝FP32；BF16 CPU3项通过仅允许真实GPUscreen，不证明真实GPU相等或通用等价。不得放宽容差或只比较最终准确率掩盖。

01:49更新：真实0.6B BF16 GPU配对已拒绝，第一批完整logits字节SHA不同（shape/stride相同），第二/第三批相同；64分数/greedy、311state和28层输出仍精确。NVML1243→711MiB只是失败候选的诊断，不能称接受收益。parent与fresh independent_completed_audit.json均False，后续1.7B未跑；原失败源与结果保留、不重复该失败配置。继续保持完整GEMM，优先验证已接受组合的长输入运行范围及峰归因。长输入缺旧baseline时只能报告单臂可行性。

02:02更新：两模型名义1024上下文的单臂组件均完成并独立审核，真实forward长度为1000/851/691，不是跑满1024。保持batch16和完整GPU GEMM；0.6B allocated5282044928B/reserved6046089216B/NVML5897MiB，1.7B allocated5626242048B/reserved7100956672B/NVML6903MiB。各349exact区间/1396独立point、原311state前后、28层所有forward与K/V几何/方法/全部来源通过。没有长输入旧对照，只能说明该组件在本地8GiB卡上的运行范围，不能计算下降百分比或替代QA准确率。

采样single-worker RSS分别14013440000/16686559232B，包含加载、全输出CPU字节摘要及评分开销，不是方法固有持有量或完整主机/PSS上限。实际max allocated均位于`forward1_layer27_end→forward1_logits_ready`，还包含norm/LM-head及model_call收尾；该forward完整logits4861952000B是对象量，不是纯GEMM峰或消除量。证据为两个`qa-layerwise-long-*-ctx1024-20261014-v1/independent_geometry_peak_audit-*.json`。随后串行验证名义1270范围，实际shape与最终状态另报。

新增文献评估：[PyTorch 2.11 MemPool说明](https://docs.pytorch.org/docs/2.11/notes/cuda.html)描述选择性allocator及UVM迁移成本，[NVIDIA WSL限制](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)说明full managed memory支持缺失。因此今晚未实现或宣称UVM超额显存收益；服务器原生Linux可用性仍要另做runtime与数值验证。

02:34更新：实际1235/1016/845-token输入的两模型单臂均通过增强独立审查。0.6B alloc6446786048/res7251951616B/NVML7047MiB；1.7B alloc6798093824/res8346664960B/NVML7907MiB。单worker sampled RSS18.53/22.04GB，包括完整输出摘要成本。新放置桥接7项CPU通过，尚未作为正式launcher启用。真实online rotation wrappers的两模型GPU兼容性fixture均独立通过：535state/28Had_K、full logits/28层/64scores/greedy及全部资源/来源精确；基权重未旋转、Had使用本机torch_fallback，不能当REAL-Q模型准确率或服务器FHT后端验证。

03:15更新：将完整指纹改为同C-order分块CPU字节摘要，6项CPU和4种CUDA张量（连续/转置/空/scalar）的旧新SHA完全一致。0.6B原1235-token同方法回放已fresh独立审核True，全部原state/输出/评分/方法不变，349区间/1396点及global allocated/reserved/NVML均无回升；worker sampled RSS18532884480→8770355200B。这是观测器主机内存开销降低，不是新的QA计算GPU显存收益，也不作稳定加速结论。1.7B同观测器正在串行复核。原observer与所有原峰/失败不修改。

## 边界与评价

短输入组件不能换算服务器原19GB全QA峰，也不是真实BoolQ准确率、十任务全QA、正式535包装state/2048×2048或多卡验证。原始模型完整state数量和正式量化包装模型数量不同。分层CPU权重/KV带来传输与主机内存成本；当前密集profile和CPU全输出摘要的时间不证明稳定加速，主机容量上限也未由这些测试证明。

QA目前是0.6B正式实验最大已观测峰，降低它后Stage0或量化峰可能成为全流程最大值；1.7B正式完整链仍待，不能称QA最后唯一瓶颈。服务器恢复后应先检查此前中断QA工件，只恢复缺失部分，再验证正式dispatch/旋转包装接口和所有device峰，必要完整验收后才能发布全流程结论。

夜间自动续接至北京时间08:00，仅重大变化通知，全部源与负向证据保留、单GPU串行、不扫描LR或进行昂贵训练。

03:36更新：1.7B stream observer已fresh独立True，原state/完整输出/分数/方法及349区间/1396point无变化，GPU三峰相同；worker sampled RSS22.04→12.18GB，与0.6B18.53→8.77GB共同支持HOST观测开销减少，非新GPU收益。新增完整QA流式事件接口CPU5passed12.18s（真实28层及实际HFLM/AST选定方法），保留两次源码归因断言失败；正式十任务与多device exact telemetry尚未接入，默认正式脚本不变。全本地，服务器关闭，继续08:00报告。

03:43更新：task原委托与四fake-device exact区间CPU6passed，真实本地CUDA64MiB transient探针验证区间峰不能由start/end替代，空闲device查询不初始化context。新完整observer+各device phase身份已接到原未旋转基权重online-wrapper GPUfixture：q06全535state/输出/score/source/KV及444区间1776point独立True，NVML3357→1243MiB；这是已接受方法的接入复核，不是新优化或正式QA。q17同接入正在串行，完成后另核实际独立工件。

2026-10-14 03:46最高优先覆盖q17 pending：新observer接入q17两臂及CUDA隐藏独立审查完整True，outputs/qa-complete-observer-rotation-qwen17-ctx128-20261014-v1/independent_completed_audit-20261014-v1.json。全部535state/28Had_K/full logits/28层3forward/64score和greedy、输入、actualKV几何/prefetch/实际方法/全部source/library均精确；新observer的state和旧原完整state摘要相同，444exact interval/1776独立point无回升缺失，all-device phase identity(当前实际1visible)完整。C/N allocated5158036992→1286125568B/res5695864832→1629487104B/NVML5551→1685MiB，全global通过，仍是已接受组合在新观测器下的wrapper兼容复核，不是新算法收益或正式QA/FHT/正确REAL-Q质量。
会话51047正常退出0，两模型GPU/control/candidate/CPU独立审查全结束；actualps没有研究worker，旧194/232均不再运行。组件绑定源冻结不改，失败方法拒绝不重跑。新增docs/qa_layerwise_formal_validation_plan_20261014.md整理服务器恢复后只读来源盘点、actual backend兼容、每device精确统计与process peak carry、完整十QA数值/方法和全流程硬件容量验收链，指出control原dispatch与candidate单设备放置差异；不包含擅自开启服务器指令，不启用默认formal。下一步必要正式接入的footer累计合并/launch前来源与完整task gate CPU准备，之后08:00报告；不要将队列完成当正式验收或夜间研究完成。

### 2026-10-14 04:06：完整观测放置桥组合及原native dispatch GPU复核

具体方法/峰累计：NEW qa_observed_placement_bridge.py组合原PlacementBridge、完整stream observer、TaskBoundaries和VisibleDevicePhases，只允许validation_only=True，formal_enabled固定False，不生成默认正式launcher。C仍原distribute_model；N在dispatch前CPU放置再使用原已验证rows/layerwise/KV。QA profile在方法安装前开始，原bridge finally清hooks/移CPU后才结束，cleanup属于统计范围。PeakCarry直接把group的每device累计max以CPU整数max合并到原memory_utils._process_profile_peaks（原CudaPhaseMemoryProfiler使用的累计表）；不替换getter，旧import alias及最终all-device footer仍能读到早期量化与新QA峰，避免reset后统计丢失。必须验证原表类型/值协议并要求独占counter owner，源文件不编辑，动态累计表不持有Tensor。

测试/失败分类：初始outputs/qa-observed-placement-cpu-20261014-v1保存source/test/log，3passed1failed10.74s。真实HFLM两臂完整数值已相同，但原C委托native没有候选的load/offload phase identity，CPU逐point比较缺身份而拒绝，不当GPU方法失败。新C只添加对应pre/post纯观测标记（无Tensor移动/forward替换），N原标记不变；对应语义允许C区间包含原dispatch开销。v2四项CPU passed11.44s，outputs/qa-observed-placement-cpu-20261014-v2/status.json只准备True：真实28层BF16/HFLM两fixture任务完整state/input/logits/28层/score配对、相同每区间/point、原异常/原getter alias保留、fake4device原0/1量化峰与已释放device3较高QA峰同时保留、finally恢复、拒绝formal开启和坏累计协议。不把fake4卡或常量fixture metadata当多卡容量/QA准确率。

真实GPU接入：NEW run_qa_observed_bridge_component.py两模型顺序串行，CLI实际source compile/--help过，启动前先CUDA隐藏fresh旧accepted wrapper observer readmission，绑定模型所有文件和实际HFLM/Cache/Qwen源码及所有新旧helper，不复制模型/data。旧架构真实online wrapper保持基权重未旋转；torch_fallback，本机1visible守卫，固定batch16/BF16/SDPA/64synthetic请求/原use_cache。C实际调用utils.dist_utils.distribute_model（保留真实native map）；N实际使用组合桥，并记录K/V实际设备/shape/stride/dtype、参数身份、full535字节和每forward方法/任务完整事件。单任务是synthetic_component，结果仅request_count，不称十任务或benchmark准确率。

两模型均完整且NEW独立两层审核True：roots outputs/qa-observed-native-bridge-qwen06-20261014-v1及qwen17同名。各independent_completed_audit-20261014-v1.json完整within-pair数值/方法/source/geometry/峰累计通过，各433exact interval/1732独立start/end point无回升/缺失；q06 NVML3357→1243MiB，allocated2908462592→970141696/res3395289088→1166016512B；q17 NVML5551→1685MiB，allocated5158036992→1286125568/res5695864832→1629487104B。它是已接受方法在更接近原dispatch接口下的复核，不是新增算法降幅，更不换算服务器19GB完整QA峰。

NEW audit_qa_observed_bridge_completed.py额外fresh old-reference来源/旧readmission、逐arm原先已接受的完整byte事件/state/score精确、实际selected code等于当前原HFLM C方法及实际生成N rows/cache方法、535+28Had_K12×12FP32布局全部通过；各independent_reference_selected_method_audit-20261014-v1.json True。旧/新observer区间数有意不同，只比较新C/N对应433区间，不拿旧444区间直接作零回升证明，不放宽数学。前述old/new complete-byte比较与selected-method身份独立支持没有改变数学或只靠汇总准确率。

实际状态/评价：会话42690串行GPU链退出0，CPU增强审查63426/13438均退出0；actualps无研究GPU/CPU，旧controller17/candidate133等均结束，不沿用。组装source绑定冻结不改。更新最小验收计划：累计合并接口已有CPU+本机1卡真实接入证据，但正式full ten QA、535正确旋转量化数学、actual服务器FHT/多卡、长序列匹配对照和NVML持续性仍未验，不启用formal默认脚本。下一步必要完整task/源/缓存/方法/所有device足额遥测的launch前准入准备和08:00最终夜间报告；服务器仍关闭不SSH，不推送/privatePDF不stage。QA不是最后唯一瓶颈的已证结论。

### 2026-10-14 04:54：整块GPU GEMM输出映射主机存储的新方法与0.6B完整短组件正向

文献/理论：CUDA12.8.1 runtime官方cudaHostAllocMapped/cudaHostGetDevicePointer允许具备UVA的设备访问mapped pinned-host地址，GPU GEMM本身不改。它与cudaMallocManaged/UVM页面迁移不同；此前WSL缺full managed支持结论保留，不能据此替代mapped runtime实测。WSL官方另说明pinned system memory可用量受限，因此64MiB可用不证明多GB或80G主机。参考 https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/group__CUDART__MEMORY.html 、https://docs.pytorch.org/docs/2.11/notes/cuda.html 、https://docs.nvidia.com/cuda/wsl-user-guide/index.html 。本机实际torch2.11.0+cu128/CUDA12.8，g++可用/nvcc无，不下载依赖。

具体方法：qa_mapped_host_allocator_probe.cpp使用cudaHostAlloc Portable+Mapped、cudaHostGetDevicePointer作为私有native MemPool的segment allocator，CUDA Tensor仍供原GPU整块Linear消费/产生，仅改变输出存储位置；CPU地址表保存指针/bytes整数不保留Tensor，active0后才删除pool并cudaFreeHost，记录alloc/free/hostlive。独立small probe现有代码/so/actual runtime库/SDK与PyTorch allocator头SHA均绑定；CUDAPluggableAllocator实际C++free_fn ABI四参数，Pythondoc字符串三参数不可靠，按已安装头编译。host-only shim的声明核对实际CUDA12.8 ABI，首次CUDA初始化前设置cudaDeviceMapHost；不改全局allocator、不启用OOM回退、不反复empty_cache。

最小效果/能力：outputs/qa-mapped-host-pool-probe-20261014-v1/status.json True，设备canMapHost/UVA都1。64MiB CUDA tensor读写/CPUcopy fullSHA精确；native allocated/reserved delta=64MiB=private active/reserved=callbackhostlive，未隐藏pool统计。私有active0后deletepool hostlive0、alloc/release1/1。原全块BF16 F.linear input[16,32,256]/weight[8192,256]/output[16,32,8192] fullSHA/torch.equal逐位同；这只是8MiB小输出，不是Qwen/fullQA/GPU显存收益，原source/so冻结。

CPU工程：NEW qa_mapped_head_storage_prototype.py仅nn.Linear原forward的pre/post scoped use_mem_pool，不替换forward、不切块、不改dtype/batch/词表/评分，拒绝被替换的forward/几何bias/错误batch或dtype。always_call失败也恢复poolcontext而不掩原异常；每head真实输出payload、private snapshot、callbackhostlive、native整个snapshot与allocated/reserved总计、private segment addresses包含于native记录独立核对，防较大的默认GPU分配遮住漏统计。CPU v1/v2各4passed3.07/3.09s，v3新增“native较大但私有segment缺失”拒绝后5passed3.08s，scope只是fake-pool/CPU BF16数学/context/lifecycle。证据outputs/qa-mapped-head-cpu-20261014-v1/v2/v3保留。不把fake allocator当CUDA证明。当前严格每head input batch16；真实任务自然尾批小于16尚未适配/验证，正式接口不能启用。

runner/工具反思：NEW run_qa_mapped_head_component.py在冻结complete-observer wrapper入口只新增mapped头/早期runtime flag，legacy state/layer/full-logits摘要改已验证stream，保留原完整GEMM和best rows+layerwise+officialKV两臂。baseline也是best存储，candidate才加mapped；不错误沿用旧all-resident-control coverage。初始源码builder weight_records anchor与旧hadamard_backend同一行导致CPU编译前拒绝，GPU0启动，原源码/error保存在outputs/qa-mapped-component-build-cpu-20261014-v1；纠正单anchor后实际worker source编译通过，原ancestor不改。初始构建错误不当方法失败。运行时先probe有效及全部SHA准入，两臂同map-host flag，模型/library/全部helper/CPP/so/runtime/header绑定，单GPU串行。

q06真实完整结果：outputs/qa-mapped-head-qwen06-ctx128-20261014-v1/status.json完整True；NEW audit_qa_mapped_head_completed.py independent_completed_audit-20261014-v1.json fresh独立True。raw未旋转基权重+online wrapper535state/28Had_K、完整full logits/28层各3forward/全部64scoregreedy/原输入/原Param identity/KV shape-stride-device/实际方法及source/library全精确，baseline与candidate还各自等此前accepted best完整byte事件。每head真实geometry与原nn.Linear.forward SHA、private/native统计不是只信bool而重新算，3calls；active0/releasehost0/alloc-free1对。451exactinterval/1804独立start/end point无回升缺失。

实际效果：相对已接受best存储baseline，q06 NVML1243→635MiB（下降608MiB，约48.91%），global native allocated970141696B/reserved1166016512B两臂完全相同。native CUDA allocator逻辑计数现在含映射的host-backed存储，不等同全是物理VRAM，必须与callback/private账及NVML联合报告；不将“少统计”为下降。峰确实少占物理设备采样，但只是本机ctx128/actual最长131、synthetic64、未旋转wrapper、torch_fallback短组件，不能换算服务器19GB完整QA/长输入/多卡/正确量化模型收益。不是旧失败的256行head chunk或noKV复跑，不改变GEMM舍入路径。

当前：会话71340 q06链已退出0，所有q06GPU结束；NEW会话11369在q06完整独立True且actualGPUidle后串行q17新目录outputs/qa-mapped-head-qwen17-ctx128-20261014-v1/log同名，两臂完成后自动CUDA隐藏fresh新同audit文件。q17尚未完整，下一次actualps，不并发新GPU、不改新绑定源。后续严格通过才必要长输入/主机代价范围，失败拒绝/保留，不重跑已完成失败配置。正式gate仍False，服务器关闭不SSH、不推送/privatePDF不stage，继续08:00报告。

2026-10-14 04:58重要交叉验证：q17映射头短组件两臂完整结束，NEW outputs/qa-mapped-head-qwen17-ctx128-20261014-v1/independent_completed_audit-20261014-v1.json fresh独立True。原535state前后/28Had_K12×12/full logits/28层3forward/64scoregreedy及所有模型/源/库/原nn.Linear forward/input/cache/方法精确，逐arm等旧accepted best byte事件，private/native真实原数值重算/segment包含/每头payload完整；私有active0、hostlive0、alloc-free相等。451exactinterval/1804逐point均0回升缺失。native allocated1286125568/res1629487104B两臂相同，NVML1685→1077MiB，减少608MiB约36.08%。q06新短组件1243→635亦独立True，不能混算比例或将short synthetic/raw未旋转wrapper/torch_fallback/实际1visible当服务器fullQA、正确REAL-Q或多卡。
会话11369及最后CPU独立已退出0，actualps无研究worker/GPU；旧controller18/GPU99结束，下一次实际核对。原mapped/probe/helper/所有run源与so/Header/库绑定继续冻结，不编辑。04:54“GPU0启动”是构建失败时启动GPUworker数量为0的简写，不表示GPU0上启动过该失败入口；原失败日志/source完整保留。
后续必要扩大范围：旧rawlong311与新short wrapper535不是同模型路径，不能直接比较；新long应保留rawlong原311数学/source/输入/精度/整块head，仅叠映射，先严格source/singlewrapper CPU构建和actual reference readmission，然后明确配对或候选单臂范围、native/private统计/RSS与区间point/source所有完整验收。多GB pinned容量及formal自然尾批<16仍未验，不据637MB成功推断6GB或80G。Mapped使用host RAM取代物理VRAM，本机长输入是否可行和CPU代价需新实测；allocator逻辑记数含host应如实报告，不只看NVML。正式足额task/正确quantized535/actualFHT/多device launch门槛仍未全，不启用默认formal。继续08:00夜间报告、服务器保持关闭。

### 2026-10-14 05:08：raw311长输入整块头映射配对已启动（尚无收益结论）

方法/工程：NEW tools/run_qa_mapped_long_component.py以冻结run_qa_long_stream_fingerprint生成的原raw311入口为基，只增加同两臂early mapHost flag、原nn.Linear pre/post scoped私有池与释放/实际callback峰观察。C仍已接受rows+逐层CPU权重+官方KV，N才加mapped输出；fresh C/N使用同16MiB完整字节观察、同输入/phase身份，保留batch16/BF16/SDPA/fullvocab/use_cache与原评分。旧short535未旋转wrapper不作为raw311直接数值对照。两臂使用独立实际worker VmRSS 0.5秒采样，不称total host/PSS或连续峰。

CPU准备：outputs/qa-mapped-long-cpu-20261014-v1四项passed4.44s，真实raw source编译、state/capture/measured_call/model pre/post AST完整未变、唯一mapped wrapper、缺原anchor拒绝；356预期exact区间包含28层×3的每load/compute/offload/end及3head边界、pool release，accounting缺segment/错payload/native/reserved/host/release拒绝。这仅CPU，不当GPU或正式准入。

实际启动：会话52815，root outputs/qa-mapped-long-qwen06-ctx1270-20261014-v1、logs同名。旧stream rawlong reference先fresh audit_qa_long_completed成功，来源模型库/原probe/so/新runner/auditor/helper及实际worker_source均绑定冻结；05:07 actual controller40、唯一control GPU82，尚无完整pair，不并发。新audit_qa_mapped_long_completed要求全311前后/state/full logits/28层所有forward/64scoresgreedy与旧同长度完整byte参考精确、实际K/V和layer payload/原Linear/private native callback重算、每356interval1424point/完整NVML UUID/global全部通过。新head markers与旧349身份不同，只比较新C/N对应356，不伪造旧memory配对；旧reference用于完整字节及来源准入。

理论/评价：继续CUDA12.8 mapped pinnedHost路径，保持整块GPU GEMM。多GB pinned容量此前未验，失败必须保存partial/source/日志、不重复该完成失败长度；即使NVML降而global native allocated/reserved升也拒绝主门槛。allocated/reserved逻辑包含host-backed CUDA存储，callback峰是segment持有量而非total RAM；记录真实长输入shape，不从637MB短组件推断6GB/80G或服务器19GB完整QA。正式十任务/正确旋转量化535/FHT/多卡与自然尾批未验，默认formal未开启，不SSH、不推送。

### 2026-10-14 05:14：长输入首版资源拒绝与最后消费者释放新版

完整负向证据：outputs/qa-mapped-long-qwen06-ctx1270-20261014-v1两臂正常完成，parent status与NEW independent_completed_audit-20261014-v1.json均False。actual1235/1016/845、原311state前后/full logits/28层各3forward/64scoresgreedy与旧stream同长度全部精确，source/model/library/KV/method/Param身份/private-native-callback统计/释放全部通过；356exact interval/1424独立point完整无缺失（局部回升按原工具保留）。global allocated6446786048B两臂相同，reserved7251951616→7352614912B（+96MiB）为唯一主验收失败，NVML7047→1415MiB只是被拒绝候选的物理采样效果，不宣称整体容量合格或推广1.7B旧方法。全源/日志/status保留，不重跑该配置。

主机代价：first mapped callback峰6006243328B，private/native segment都纳入CUDA逻辑统计；worker sampled VmRSS8770707456→14765682688B，包含载入/CPU摘要。说明物理VRAM转移给主机RAM，不是消除全部内存/80GiB主机容量证明。full output6004510720B是对象量，与采样峰差不能简单画等号。

定位与反思：reserved_lifetime_diagnostic-v2.txt确认首头native reserved7124025344B低于control global；最大7352614912在第三forward layer27 compute及后续。单私有池首批6GB段一直留到三批后，后续层计算与缓存段叠加是可操作的生命周期证据，不宣称96MiB唯一allocator分配根因已确认。首次只读diagnostic传str给要求Path的intervals造成AttributeError，原txt保留，v2仅修读路径类型、无GPU重跑/方法变化。独立完成审查有效，读诊断错误不当量化/数值失败。

新具体方法：NEW qa_mapped_head_batch_lifetime.py保持原nn.Linear完整GPU GEMM及BF16，只在已接受rows方法原del multi_logits后插一个callback。每batch最后原消费者已结束，先检查private active0，再delete pool/cudaFreeHost到hostlive0；下一批head lazy新pool。C同callback/phase只记录，不映射。没有反复empty_cache、没有CPU GEMM/切块或数据改变。拒绝尚活跃存储、重复/漏forward释放、cleanup不完整，不将原异常改成成功。原head/probe/旧runner/audit全部冻结不改。

CPU准备：batch helper实际原HFLM AST除del之后的callback外全等，3批fake池完整BF16 bytes/原异常及引用保护，4passed9.73s；新versioned runner/auditor actual raw source+batch完整覆盖共6passed16.64s，roots qa-mapped-batch-lifetime-cpu-20261014-v1及qa-mapped-long-batch-cpu-20261014-v1。仅CPU，不当新收益。新版tools/run_qa_mapped_long_batch_component.py / audit_qa_mapped_long_batch_completed.py保留旧数学/原验收，只增加每批active/host释放记录和362exact interval/1448point身份准入，所有三global仍原门槛。

真实新启动：工具会话25268、root outputs/qa-mapped-long-batch-qwen06-ctx1270-20261014-v1/log同名；先fresh旧成功raw311长参考完整readmission，再C/N单GPU串行，尚无新结果。新方法是已完整负向配置之后有证据的生命周期更改，不是原失败复跑/改门槛。完成必须fresh独立全审查新JSON，严格通过才必要q17同长度交叉；原long v1拒绝/short535两模型接受不改。服务器仍关闭，不SSH不push，继续08:00报告。

### 2026-10-14 05:19：0.6B长输入按批池释放完整通过；自然尾批CPU准备

NEW outputs/qa-mapped-long-batch-qwen06-ctx1270-20261014-v1/status.json及independent_completed_audit-20261014-v1.json均完整True。actual1235/1016/845，原311state前后/full logits/28层各3forward/64scoresgreedy与冻结旧rawlong参考精确，input/完整源库/Param/KV/真实layer payload/原Linear与mapped/private-native账全部通过。三批每次真实pool active0、hostlive0后释放，alloc/free3/3；callback peak6006243328B纳入逻辑CUDA计数。362exact区间/1448独立point完整无缺失，逐point0回升；区间有局部峰诊断，所以strict_local_nonincrease=False保留，主容量True不混称局部全0。

实际C/N native allocated6446786048B相同，reserved7251951616→7124025344B（减少122MiB），NVML7047→1415MiB（采样，非连续每刻）。单worker sampledRSS8768528384→13670653952B，主机约多4.9GB观测值含加载和指纹，不等于callback持有量或总host/PSS，不证明80G/正式任务。相比被拒绝长首版，只改变池最后消费者释放使native reserved回升消除，没有改数值/门槛。dense诊断评分wall21.5566→31.0451s（保留），包含密集profile/CPUhash/snapshot/pin-transfer，不作为稳定方法速度或端到端成本结论。

继续交叉：会话14577在q06fresh独立退出0后已串行新q17相同长组件root outputs/qa-mapped-long-batch-qwen17-ctx1270-20261014-v1/log同名，结束自动CUDA隐藏同独立审查。05:18 actualcontroller278/唯一controlGPU317，尚无新q17结果，不能声称两模型长输入均通过。旧72/153已结束，不沿用，冻结所有绑定src/so。

正式自然尾批工程：NEW qa_mapped_head_natural_batch.py允许原评测器在configured_batch_size=16下产生实际1..16尾批，不补padding/删request/改配置。只扩头input准入范围，完整nn.Linear/GEMM/输出shape/score仍原，逐batch释放继承已验证方法；configured batch8和非法0/17拒绝。CPU5passed8.34s，outputs/qa-mapped-natural-tail-cpu-20261014-v1，fake mapped池batch1/7/16全原BF16 bytes和释放；实际安装HFLM+真实28层小BF16模型17request两批16/1、完整state/28层/full logits/17scoresgreedy在完整observer配对精确。仅CPU（真实CUDA pool/原Qwen大模型自然尾批尚未验），不启用formal、不将此当降低batch到1的方法收益。

后续需保持同范围真实CUDA自然tail、足额源/所有device与完整十QA/正确旋转量化535/FHT接口验收，再服务器恢复后的最小链；当前仍synthetic raw311/本机1卡范围，不能换算服务器19GB或正式fullQA。主记录继续记录所有负向和成本，08:00报告，不SSH不推送。

### 2026-10-14 05:25：两模型raw长输入逐批映射池完整独立汇总通过

q17 NEW outputs/qa-mapped-long-batch-qwen17-ctx1270-20261014-v1/status.json和independent_completed_audit-20261014-v1.json完整True，原311state前后/28层各3forward/full logits/64scoresgreedy等旧成功同输入raw参考精确，所有源库/实际KV/原Linear/Param/native-private-callback/三批active0host0 alloc-free3/3全部通过。actual1235/1016/845，362exact interval/1448独立point0回升缺失；global allocated6798093824B相同，reserved8346664960→8218738688B（-122MiB），NVML7907→2363MiB。callback peak6006243328B，worker sampledRSS12303314944→17140813824B（非totalhost/PSS/连续最大）。dense评分wall30.8068→42.8435s仅诊断，不称稳定速度。

NEW tools/summarize_qa_mapped_long_two_models.py又fresh重新执行两模型完整auditor与全部源/model/library SHA，不只读旧valid。outputs/qa-mapped-long-two-model-summary-20261014-v1.json实际05:24:08 True，724interval/2896独立point，所有原工件SHA保留。q06已通过NVML7047→1415、allocated6446786048相同/res7251951616→7124025344B，0.6局部两个allocated区间分别+786432/+721408B完整保留、逐point0，无全局回升；q17局部全0。旧q06长单池全局reserved+96失败不改，当前新生命周期准入与原失败明确区分。

方法/效果/评价：支持本地8GiB卡原raw311实际最长1235-token synthetic组件的整块GPU GEMM输出host映射+最后消费者逐批释放；新物理VRAM峰下降没有修改数学/精度/batch/词表/cache。native allocated/reserved含host-backed CUDA逻辑存储，必须并列native/private/callback/NVML/workerRSS，不混称所有内存消除。每次6.006GB mapped callback持有量已实际测到，但80G服务器/更长长度/正式QA不能从此推断。

NEW docs/qa_mapped_output_storage_20261014.md汇总方法、各短/长范围、精确性、全局与局部、主机代价、首版失败和恢复验收缺口。短535未旋转wrapper与长311不混算同workload，完整服务器十QA/正确量化旋转/FHT/真实多device/fullpipeline上限仍未验，defaultformal未启用。当前14577全部GPU和CPU独立正常退出0，summary11096也退出0，05:25 actualps无研究worker，旧278/317/415全部结束，不沿用。

下一步继续必要自然尾批真实CUDA/足额正式准入CPU工程与08:00报告：自然tail helper5CPU只准备（configured16下实际16/1完整17请求精确），必须新fixture/新审查允许实际1..16并保持所有请求/配置16，不能用旧硬coded每shape16审查假放行或编辑冻结源。所有新旧绑定helper/run/audit/so继续冻结，服务器关闭不SSH、不推送、不stage私有PDF。队列结束不等于正式研究完结。

### 2026-10-14 05:51：自然尾批两模型CUDA完整通过与原生接口映射组合准备

NEW run_qa_mapped_natural_tail_component.py / audit_qa_mapped_natural_tail_completed.py先CPU7passed16.94s（actual raw入口单wrapper/原数学/65请求保留/尾批覆盖/错误配置拒绝），roots outputs/qa-mapped-natural-tail-entry-cpu-20261014-v1。两模型随后串行真实CUDA：outputs/qa-mapped-natural-tail-qwen06-ctx128-20261014-v1及qwen17同名，parent和各NEW independent_completed_audit-20261014-v1.json均True；会话3576全部正常退出0，actual不再有该GPU/CPU链。

实际配置batch16、65原请求（原64synthetic+一个unique短请求），forward真实batch16/16/16/1、尾批input1×3，原full词表/BF16/SDPA/use_cache与score不变，无padding/删请求/改配置。每模型全原311state前后、28层每4forward、完整logits/65scoresgreedy、真实KV/参数身份/所有源库/请求SHA、原NNLinear/native-private-callback账、四次pool active0/hostlive0/alloc-free4/4全部通过。每对481exact interval/1924独立point全0回升缺失。旧64long仅来源/方法准入，新不同请求不冒称完整bytes等旧64reference；数值对照为新freshC/N。

效果：q06 NVML1245→637MiB，allocated970141696B两臂同，reserved1168113664→1124073472B；q17 NVML1687→1079MiB，allocated1286125568B同/res1631584256→1587544064B，两模型均reserved-42MiB。callback峰637534208B。短natural组件不换算服务器完整QA或长1235范围，采样NVML非连续。原short535两模型/long311两模型已接受不改；本次raw311自然尾批支持已配置16下原HFLM自然不足16的输入兼容，不称降低batch到1的新方法。

早期运行时CPU工程：NEW qa_mapped_early_runtime_entry.py只生成validation_only入口prefix，明确拒绝formal开启；把原QA所属global/local rank0的mapHost准备置于所有原模块body之前（保留docstring/future import），Stage0/skip_eval/无lm_eval/非rank0不初始化mapped context。binarySHA与显式REALQ_QA_STORAGE_VALIDATION环境准入，原prepare_runtime拒绝已初始化CUDA。实际formal build_entry生成后的完整原AST（去prefix）完全一致，CPU3passed3.87s，outputs/qa-mapped-early-entry-cpu-20261014-v1，仅fake runtime时序/真实source编译，不是实际4rank/服务器ABI/全链验收，不修改defaultformal/旧source。

完整组合桥CPU：NEW qa_mapped_observed_placement_bridge.py叠原ObservedPlacementBridge，保持C原native score全部归一化，只在原batch loop后加纯观察callback（AST去callback原body全同）；N保持rows/cache，再最后consumer逐批private池释放。完整observer接实际输入/28层/logits/全scores/task，devicegroup+PeakCarry保原footer和所有device峰，headcleanup属于QA区间。原HFLM selected代码先比已接受原/rows，禁止叠无审查方法；private active仍有消费者时保owner/原异常，fail记录False。当前head入站仍要求原Linear forward，未适配实际多卡accelerate head-hook链，桥validationOnly，不启用formal。

失败/测试反思：CPUv1 3passed1failed11.48s，只有测试读inspect带缩进源直接ast.parse导致IndentationError；原helper实际原C和N全数值/两个CPUfixture任务/fake4device累计与原异常已经通过，GPU0worker启动（数量0）。原code/test/log保存outputs/qa-mapped-observed-bridge-cpu-20261014-v1；NEW test v2仅dedent正确源读，helper未改，4passed11.36s，outputs/qa-mapped-observed-bridge-cpu-20261014-v2。fake4device非真实多卡容量。新runner实际原native source/single composition/earlyMap时序/65请求/593期望phase CPU连同helper5passed19.03s，outputs/qa-mapped-observed-component-cpu-20261014-v1，仅准备。

真实组合正在新启动：tools/run_qa_mapped_observed_bridge_component.py保原native dispatch C，N rows+layerwise+officialKV+mapped，raw未旋转base加原online wrapper535/torch_fallback/65synthetic、configured16/实际natural tail；NEW audit_qa_mapped_observed_bridge_completed.py额外检查complete input/fullbytes/535+28Had/NNhead/每batchprivate账/593exactinterval2372独立point/global/carry。会话51079 root outputs/qa-mapped-observed-native-qwen06-20261014-v1/log同名，完成后自动CUDA隐藏NEW independent_completed_audit-20261014-v1.json。当前还没完整GPU结果，不冒称新组合修复或formal验收。先oldwrapper reference fresh完整readmission+probe所有实际源/so/headers库SHA，再single GPU。全部新source冻结不改，不并发。严格完整过后才必要q17相同组合新目录，任何数学/global/工具失败保存先分类，不改方法遮掩，不复跑失败。

服务器保持关闭，不SSH/推送/privatePDF不stage，仍继续08:00完整夜间报告。当前证据不能替代完整十任务/真实旋转量化535/服务器FHT/多device或正式全流程上限，QA非最后唯一瓶颈。

### 2026-10-14 06:14：原生QA接口映射组合两模型完整完审与实际方法身份补审

NEW roots outputs/qa-mapped-observed-native-qwen06-20261014-v1及qwen17同名，parent status、各independent_completed_audit-20261014-v1.json均True。全535state前后/28Had_K12×12FP32/实际model_input完整SHA/每次28层/full logits/全部65scoregreedy/原请求/Param身份/actualKV/原Linear/native-private-callback/fourpool active0host0 alloc-free4/4、真实native-distribute C与single-layerwise+rows+officialKV+mapped N、每device精确区间与PeakCarry最终footer全部通过。配置batch16，实际16/16/16/1，lastinput1×3；base仍未旋转、torch_fallback、本机1visible、synthetic，不当正式REAL-Q质量/多卡/十QA。

各593exact interval/2372独立start-end点0回升缺失，q06 C/N NVML3359→637MiB、allocated2908462592→970141696B、reserved3397386240→1124073472B；q17 NVML5553→1079MiB、allocated5158036992→1286125568B、reserved5697961984→1587544064B。这里C是原native完整score/放置，N是已接受存储组合+映射，不能把比例解读为在best基线上单独映射的增量；raw自然tail best-baseline1245→637/1687→1079与longraw best7047→1415/7907→2363分别保留，不混workload/模型state或方法对照。

NEW summarize_qa_mapped_observed_two_models.py再次fresh全两模型audit与所有源/model/librarySHA，outputs/qa-mapped-observed-two-model-summary-20261014-v1.json实际06:06:01 True，1186interval/4744points，所有原工件SHA保存。新独立audit_qa_mapped_observed_selected_methods.py又逐arm比current原HFLM C/native-callback与N rows-after-del-callback、当前original model_call C及official-cache N的完整method_record（outer/actual selected code/file/source）、helper记录installed/selected codeSHA。两模型各independent_selected_method_audit-20261014-v1.json全True，不只凭编译或旧valid认定实际方法。85514为CPU补审，GPU已全部退出；51079/36936/73305已退出0，actualps当前研究GPU空闲，旧17/134等不可沿用。

理论与评价：支持完整观测、原任务委托与CPU整数peak累计接口的单卡真实接入，映射原wholeGEMM输出由CPU pinned存储承载且每批最后消费者释放，source/method/numerical/native/private账全齐。allocated/reserved包含host-backed逻辑存储，所以native最大区间不能自动当作物理NVML峰区间；当前NVML没有与每phase建立精确时钟对应，不补造物理峰纯head归因。CPU early entry仅rank0提前flags/binarySHA/AST准备，实际server runtime/ABI/FHT/正式十QA/正确旋转量化535/多GPU仍待。原head原forward入站仍会拒绝accelerate多卡head-hook，默认formal未改；后续只在新版本适配原hook链及准入，不编辑绑定旧源。

负向CPU保存：qa-mapped-observed-bridge-cpu-20261014-v1只有测试侧inspect缩进读错，原helper实际数学和异常恢复已3项通过，原snapshot/log不改；v2测试onlydedent后4passed11.36s。不是GPU方法失败或QA数值放宽。新源/所有成果/失败/so继续冻结，服务器关闭不SSH/推送/privatePDF不stage。继续必要正式控制头hook兼容/足额源-task-device gate的CPU准备与08:00完整夜间报告，不因本地队列完结就称formal成功或QA唯一最后瓶颈。

## 2026-10-14 06:26 控制头 Accelerate 接口准备与必要 CUDA 接续

文献/理论：直接读取实际安装 accelerate/hooks.py 的 add_hook_to_module、AlignDevicesHook 和 SequentialHook 源码。它将原 Linear 委托保存在 _old_forward，当前 forward 是该官方 new_forward 的 partial；因此控制组只观察这个委托链，不能剥离设备 hook 或把任意 forward 包装视为原 Linear。源 SHA 18418fd0636991c75ccabaa95d38deccc7082cb34eed308004e080afd7ca80ed 已记录。

具体方法：NEW qa_control_head_observer.py 只安装自己的 pre/post(always_call)观察器，记录完整头输入/输出几何与设备、原始 forward/Parameter 身份、实际官方委托 code/library SHA 和每批纯观察标记；不更换 forward、不移权重、不创建 pool。只接受原 Linear 或实际已安装且非 offloading 的 AlignDevicesHook/SequentialHook，拒绝任意包装、变链、offloading 和非法自然 batch。NEW qa_mapped_observed_placement_bridge_v2.py 对冻结祖先仅 AST 替换一个 ticket factory：C 用纯控制观察，映射 N 仍走已接受原 Linear 的严格检查，不在候选叠 Accelerate。

效果/验证：真实已安装 Accelerate、完整 BF16 头输出、自然16/1/7批、既有hooks保留、异常清理、篡改拒绝和真实HFLM两fixture任务/full state/input/logits/28layers/scoregreedy/原footer的CPU6tests通过11.42s；加真实GPU entry生成编译与工作负载检查共7tests通过11.76s。outputs/qa-control-align-entry-cpu-20261014-v1/status.json 为 CPU 准备证据，含新源快照/源与日志SHA。此前两模型映射组合45/47绑定源独立全SHA未变。没有将fake4device或CPU结果当真实多卡。

评价/接续：NEW run_qa_control_align_component.py 在新的同65请求、535未旋转wrapper、torch_fallback、单CUDA设备fixture中给控制头实际添加官方 AlignDevicesHook(cuda:0,io_same_device=True)，保持原GEMM/输入/评分。目的仅验证新控制观察接口，不能作为新算法收益或正式多卡容量。06:25 actual controller17/control worker105（仅当时有效），会话43399，root outputs/qa-control-align-qwen06-20261014-v1；两臂完整后自动CUDA隐藏 audit_qa_control_align_completed.py fresh全部数值/方法/源/593interval/2372point/global及实际官方hook委托与保留身份。尚未完成独立GPU审查，不提前报告成功；后续actualps/status为准，源已冻结。

反思/未验：真实服务器可能存在offloading/其他hook组合，本版明确拒绝，不能仅放宽检查使其启动。正式十任务、正确REAL-Q旋转量化535状态、服务器FHT与allocator ABI、全部可见GPU和主机容量准入仍待，formal默认不变。服务器继续关闭、不SSH、不推送；08:00报告须区分CPU接入准备、组件物理峰收益和正式缺口。

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
