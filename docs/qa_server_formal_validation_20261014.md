# 2026-10-14 服务器正式QA存储验证

用户恢复OpenBayes并明确授权连接、推送和正式实验。夜间的“不SSH/不推送”限制仅适用于当时关闭的服务器；本次按新的授权部署。密码不写入仓库或实验文件。

## 当前部署与运行

- SSH：`root@ssh.openbayes.com:32666`。
- 仓库：`/output/gptq_plus`；部署分支`realq-warmadam`，提交`13af9dc`。
- Python：`/output/env/bin/python`；PyTorch`2.9.1+cu128`、Transformers`4.56.2`、lm-eval`0.4.4`、Accelerate`1.12.0`。
- 四张RTX5090，每张32607MiB；实际容器cgroup上限167772160000B。它不替代运行期PSS/系统/cgroup采样。
- 原正式控制器PID5728已在candidate失败后退出；恢复进程以末尾新记录和actual ps为准，不沿用PID。
- 控制器根：`/output/gptq_plus/outputs/storage-formal-qa-mapped-20261014-v4`。
- 总日志：`/output/gptq_plus/storage-formal-qa-mapped-20261014-v4.log`。

正式队列按0.6B control→candidate、1.7B control→candidate串行执行；各臂完整28层W4A16量化、KL/PPL和原十项QA，QA配置batch16/BF16/SDPA/完整词表/原KV。原量化超参数与原存储方法比较不变，候选QA增加rows寿命、逐层CPU权重、官方KV卸载、完整GPU Linear输出映射及逐批最后消费者释放。原默认脚本未更改，使用独立opt-in验证入口。

正式启动命令（已经执行，勿再开第二份）：

```bash
cd /output/gptq_plus
nohup env PYTHONPATH=/output/gptq_plus CUDA_VISIBLE_DEVICES=0,1,2,3 \
  HF_DATASETS_CACHE=/output/hf_cache/datasets HF_DATASETS_OFFLINE=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /output/env/bin/python -u tools/run_storage_formal_qa_mapped_v8.py \
  --runtime-root /output/gptq_plus/outputs/qa-formal-server-20261014-v1/runtime-v2 \
  --component-root /output/gptq_plus/outputs/qa-formal-server-20261014-v1/Qwen3-0.6B-interface-v2 \
  --component-root /output/gptq_plus/outputs/qa-formal-server-20261014-v1/Qwen3-1.7B-interface-v3 \
  --run --model06 /openbayes/home/llmModels/Qwen3-0.6B \
  --model17 /openbayes/home/llmModels/Qwen3-1.7B --devices 0,1,2,3 \
  --repeats 1 --cache-mode auto --cache-storage shared \
  --output-directory /output/gptq_plus/outputs/storage-formal-qa-mapped-20261014-v4 \
  > /output/gptq_plus/storage-formal-qa-mapped-20261014-v4.log 2>&1 < /dev/null &
```

先CPU逐rank校验原Fisher/prepared缓存，匹配则共享引用，不复制大模型/数据、不重复Fisher。原中断的QA工件保留；新组合的完整字节/28层/全部请求与评分/四设备统计要求比旧scored-row记录更完整，不能直接将旧control结果当新组合的配对证明。

## 已完成准入

十任务/61叶任务均离线加载通过。服务器原生CUDA12.8映射分配器重新编译，64MiB读写/整块BF16 GEMM字节精确，原生/private/callback统计和释放完整通过。PyTorch2.9没有`include_traces`参数，新增兼容层只委托原snapshot接口，不改变分配或数学。

两模型均以真实服务器FHT、四visible设备执行65条synthetic请求，实际batch16/16/16/1、完整535状态/28Had/全部logits和分数、实际K/V/源/库/原生与private账、每设备593区间和2372独立点通过。这只是服务器接口准入，基权重尚未旋转，不是正确W4模型或完整正式QA结果。

0.6B原status False只因在原多卡dispatch之前采集Parameter ID，原Accelerate放置重建Parameter。原False保留；独立CPU重审只将此项移为明确的放置诊断，全部完整数值/状态/头Parameter/原源码及资源门槛仍保持。没有重跑这份GPU工件。1.7B新入口直接检查dispatch之后到QA结束的所有Parameter身份，结果True。正式两臂也必须直接通过这项及共享Parameter拓扑检查，不能靠短组件读准入替代。

本地5项CPU入口/真实Linux主机窗口测试通过，实际命令参数测试另外1项通过；服务器Python3.10实际执行3项独立CPU检查通过。服务器未装pytest，最初pytest入口的`No module named pytest`原日志保留，随后直接运行原测试函数，没有安装或升级训练运行时。

## 完整正式验收

`audit_qa_server_mapped_v3.py`独立核对全部state/input/28层/full logits/每logprob及greedy/全部十任务、实际选择方法、KV、原头Linear、pool账与逐批释放、全设备每phase及每个start/end、Parameter身份和别名。原过程峰与QA累计峰继续合并；原formal全流程数值/来源/缓存及global容量审查保留。

每个QA窗口单独保存四GPU NVML/UUID和只读进程树RSS/PSS/cgroup。主机监测通过实际stop标记正常结束，缺失/部分样本保留，不能补造旧PSS或连续峰。QA物理峰和全流程物理峰分别报告；native allocated/reserved包含host-backed CUDA逻辑存储，不隐藏private映射段。

当前没有正式pair完成，不能把本地1.38/2.31GiB或服务器短组件峰当原约19GB完整QA的新峰。以完整两臂和独立审查结果报告；任何数值、来源、统计或global资源失败都保存拒绝，不修改旧工件、放宽容差或删任务。

## 已保留的接入错误

CPU早期setup插入顺序、来源清单str/Path、GitHub TLS拉取中断、原control pre-dispatch ID观察、argparse把`--run`当`--runtime-root`缩写、0.6B配置HF blob符号链接身份问题均按版本保留。GitHub远端已推送，TLS失败后服务器使用同提交Git bundle fast-forward，模型/数据未传输。正式v5/v6均在GPU开始前退出，v7真实CLI和实际模型文件SHA预检通过。相关GPU数值结果不因这些接入错误改判；没有复跑noKV/chunk或首版生命周期失败配置。

正式v7缓存已完整命中，但旧profile安装器依赖原metrics/fingerprint文本锚点，AST重排使其在GPU前拒绝。新v8仅保留原入口文本并插入早期运行时/桥，不重排原量化代码；本地真实formal→early QA→native profile→四设备footer集成测试1项通过（12.72s），服务器Python3.10同测试实际通过。新root v4使用独立不可覆盖的准入manifest；旧v3失败工件保留。当前控制器5728，需以actual ps和新总日志为准。

已实际确认正式GPU worker启动：0.6B control，tag qwen06b-storage-formal-20261014-014543-df96105a-p1-control，四rank从原缓存加载后进入第0/28层量化，阶段日志25752B且无Traceback；当时四卡各5578MiB。前述GPU前接入失败未计入成功pair。查看总日志：`tail -n 80 -F /output/gptq_plus/storage-formal-qa-mapped-20261014-v4.log`，Ctrl+C仅结束查看。

## 15:45后：正式candidate失败诊断与独立恢复

原0.6B control完整通过量化、KL/PPL和十QA；其QA窗口为北京时间11:34:32至13:04:15，截图12:51:54的19.1GB对应control BoolQ，GPU1采样峰19098MiB。control头实际在GPU3，不将GPU1物理峰直接归因于头。此control本身继承更早的系统优化，属于增量存储消融，不是用户期望的“全部优化之前”累计基线；最终报告必须分开两种问题，不把本轮差值当全部优化总收益。

原candidate量化完成（6127.820844218135s、535state SHA f8f7d45619437216b74ec14cfb22b63a33288ae6cf2ec14798534ea74cb82ccc），KL/PPL完成，PiQA已产生3676scores/229forwards。HellaSwag在总forward257、实际input16×123、head1024→151936处触发`Private mapped output/native accounting incomplete`；随后的“每批释放”检查失败并出现rank0 SIGSEGV，导致缺少rank0最终footer。原False、源码、日志和partial全部保留，1.7B尚未正式启动。原异常cleanup未写ticket内部raw row，不能从旧结果凭空恢复该计数，也不能声称旧失败数值已通过完整正式门槛。

先在原生GEMM初始化工作区后，按原257输入几何序列作独立head-only诊断，`outputs/qa-mapped-accounting-history-20261014-v2/run/status.json`精确在call257重现同一保护失败。payload/private_active598020096B、private_reserved/mapped_host_live599785472B，native与snapshot allocated922193920B、reserved2118123520B均精确相同，callback_error0。唯一失败项是private地址未出现在global原始snapshot中；private segment地址123725751601280和其首block地址123713546616832自身矛盾。异常退出栈展开后active0、callback alloc/free257/257、hostlive0，未删除活动池或伪造release。诊断使用synthetic hidden values，不是正式模型数值对照。

代码证据：PyTorch2.9.1过滤snapshot绑定将`PyTuple_GetItem`的borrowed引用放入owning `THPObjectPtr`，未增加引用；结合Python3.10小整数缓存和call257损坏，指向该过滤分支的引用寿命缺陷。参见[PyTorch2.9.1绑定](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/csrc/cuda/Module.cpp#L671)、[owning指针定义](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/csrc/utils/object_ptr.h)、[Python borrowed引用契约](https://docs.python.org/3.10/c-api/tuple.html#c.PyTuple_GetItem)。这是源码与复现实验支持的归因，不声称上游已确认或修复。

NEW `qa_server_runtime_compat_v2.py`完全避开带pool-ID的native绑定，读取不带ID的global原始snapshot后精确按`segment_pool_id`筛选；不改地址、不吞private计数、不改变池分配/GEMM/评分、不放宽旧guard。另检查每段各block地址连续且完整覆盖segment。旧snapshot helper和allocator/so及所有既有GPU绑定源未编辑。

NEW v3诊断两次真实几何周期共514次：166.23134306492284s、514/514释放、hostlive0、全部旧计数guard通过；calls1/255/256/257/258/513/514原完整GPU BF16 F.linear与映射原Linear逐完整输出字节SHA精确相同。这是修复接口压力与7次完整GEMM对照，未冒称514次全部正式QA数值或新的显存收益。v1新进程首GEMM把cuBLAS约8MiB持久工作区装入私有池而无法释放的False也保留；v2/v3先原生初始化，符合正式量化/PPL先于QA的顺序。Windows pytest收集因datasets连接WinError1920失败，改实际WSL8CPU通过；服务器未装pytest，第一次stub无spec导致导入失败记录保留，新CPU v2原测试函数8项通过且CUDA未初始化；独立raw pointer审查另2CPU通过。

恢复入口NEW `run_storage_qa_mapped_recovery_v2.py`只运行0.6B candidate，复用完成control与原共享Fisher/prepared缓存，完整原2048×2048、量化超参数、四rank、十QA、配置16/BF16/SDPA/fullvocab/KV不变。NEW journal逐批保存原guard row、actual output pointer、private/global所有segment与block、callback计数，失败也flush；独立`audit_qa_head_accounting_journal.py`逐字段重算、核对pointer完整覆盖和原head记录，加入最终主门槛。原完整数值/source/cache/interval/start-end/global审查仍全部保留。

上次无量化检查点，不能从SHA恢复qmodel，因此新candidate需要重做量化，不重跑control/Fisher。NEW opt-in checkpoint在量化完成、KL前保存完整qmodel和runtime manifest；实际reload逐全state字节与前后state/Parameter identity精确核对，路径`qwen06b_quantized_recovery.pt`及`.receipt.json`，不覆盖旧工件。保存/hash开销计入本次恢复wall，不能当稳定速度对比。default formal未启用新helper。

新恢复根`/output/gptq_plus/outputs/storage-qa-mapped-recovery-20261014-v1`，日志`/output/gptq_plus/storage-qa-mapped-recovery-20261014-v1.log`。15:45实际controller104676/pipeline104742已开始原共享缓存CPU逐rank核对；之后GPU阶段需actual ps和日志确认。旧正式v4保持False，恢复完整十QA及独立pair审查尚未完成；不宣布正式19GB已降到某数。

```bash
tail -n 80 -F /output/gptq_plus/storage-qa-mapped-recovery-20261014-v1.log
```
