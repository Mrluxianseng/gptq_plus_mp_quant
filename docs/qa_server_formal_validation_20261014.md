# 2026-10-14 服务器正式QA存储验证

用户恢复OpenBayes并明确授权连接、推送和正式实验。夜间的“不SSH/不推送”限制仅适用于当时关闭的服务器；本次按新的授权部署。密码不写入仓库或实验文件。

## 当前部署与运行

- SSH：`root@ssh.openbayes.com:32666`。
- 仓库：`/output/gptq_plus`；部署分支`realq-warmadam`，提交`13af9dc`。
- Python：`/output/env/bin/python`；PyTorch`2.9.1+cu128`、Transformers`4.56.2`、lm-eval`0.4.4`、Accelerate`1.12.0`。
- 四张RTX5090，每张32607MiB；实际容器cgroup上限167772160000B。它不替代运行期PSS/系统/cgroup采样。
- 当前正式控制器PID5728；不要跨重启沿用PID。
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
