# 两卡 REAL-Q v4 消融入口

本入口用于新服务器上的**同机两张完整 GPU**。它使用 `vendor/official_v4`
中已归档的官方基线和 v4 源码，未修改旧实验结果、旧源码工厂或旧四卡执行链。
导入模块、查看帮助和运行 CPU 单元测试均不会租赁服务器或启动 GPU 实验。

## 范围与正确性边界

- 默认矩阵：0.6B、1.7B 各执行完整 v4 对照，再分别关闭 Stage0 block checkpoint、
  CPU 输入流、求逆复用组合，共 **8 个串行实验臂**。另两项历史注册消融和 baseline
  可通过 `--variants` 显式选择；每个模型必须以 `full_v4` 开始。
- 原生 Config 完整来自归档配置：W4A16、N2048、T2048、全局 static batch4、
  Hessian batch8、backward samples/batch32、seed1/rotation0/refresh0、BF16、
  math SDPA、原学习率、P95/P99、TF32/Triton/fused 开关均不自动调整。
- **两卡与四卡不是相同数值分区。** 局部分位数统计人口、FSDP/collective
  舍入及每 rank 样本数可能变化；不宣称与历史四卡逐位相同。只在同机器、同两卡、
  同软件栈的新对照间作比较，不能把新旧耗时合并成一组重复实验。
- 默认 `--qa full` 保留官方 `qa_eval` 的完整十任务、batch16、BF16、native KV，
  使用新的两卡整层常驻 map，无模型/KV offload。`--qa none` 是**明确选择仅核心消融**，
  仍运行 KL/PPL，不代表新 QA 已通过。
- 该入口**不是旧四卡 mapped QA bridge / 独立65请求审查器的移植或豁免**。
  两卡 QA 采用两侧一致的原生评分。所有输出保留 historical formal acceptance=False；
  不能把执行成功冒充旧33门/65请求审查、论文终稿全部接受或GPU容量保证。
- 本机 CPU 检查只能验证下载、缓存、配置、串行控制和错误处理，不能证明新 GPU
  上 CUDA/FSDP/Triton/NCCL/完整模型数值正确。实际两卡运行需先完成环境与容量检查；
  任一非零、超时、缺层、统计/权重不一致或质量不合格均停发，无自动重试、降batch或调参。

## 持久存储与模型/数据下载

建议将 `--root` 放在有足够空间的持久磁盘，如 `/workspace/realq-work`。
所有下载、统计、旋转模型、量化模型和结果都在该目录下，**不放进 Git**。

```text
realq-work/
  shared/
    resources.json              # 实际模型/数据路径、固定commit、状态、逐文件SHA
    hf/hub/                     # Hub唯一模型/原始数据缓存；保留.incomplete供续传
    hf/datasets/                # 所有任务共享Arrow/处理缓存
    sources/<source-sha>/       # 校验后的版本化源码（很小）
    stage0/<scope-sha>/          # 单份tokens、rotation、reference、rank统计
    checkpoints/<state-sha>/    # 内容去重的实际量化checkpoint
    locks/
  runs/<unique-id>/
    plan.json / started.json / status.json
    00-q06-full_v4/
      spec.json / launch.json / wait.json / console.log
      host_nvml.jsonl / state.json / checkpoint.json / quality.json
      rank0/allocator.jsonl / configuration.json / result.json / transport.json
      rank1/...
```

模型不存在、目录不完整或不能读取时，`prepare` 从固定 Qwen Hub commit 下载。
已有路径可用 `--q06 /path/to/model --q17 /path/to/model` 指定。WikiText
可用 `--wikitext /path/to/snapshot` 指定，目录下需有
`wikitext-2-raw-v1/{train,validation,test}-*.parquet`。

下载前先落盘 `resources.json`；失败时保存 incomplete 状态，保留 Hub 的未完成块。
**再次执行同一条 prepare 命令即可继续下载**，不会重新复制已完成的模型。
下载采用同一 HF blob cache，验证固定 revision、大小及官方 Git blob/LFS SHA；
本地显式模型校验 config、tokenizer、safetensors header，登记实际字节身份。
成功后每次先校验本地内容；不需要每个实验重新下载。可在启动 Python 前设置
`HF_ENDPOINT` 指定可访问的兼容镜像，仍须通过固定内容身份检查。

`prepare --qa full` 会先解析并下载全部十任务数据。`run` 强制离线：
缺资源时直接留下失败记录并停止，不在GPU实验中临时下载。

共享 Stage0 key 包含模型/数据内容身份、完整数值配置、卡数、GPU UUID/驱动和软件版本。
因此两卡不会命中四卡缓存，换模型、数据或运行栈也不会误命中。

第一次完整 v4 对照产生统计并封存。CPU streams/solver/lifetime 等不影响
teacher 的消融读取该单份缓存。**关闭 Stage0 checkpoint 的消融和 baseline
必须重新计算 Stage0**，逐 rank 比对统计内容，但不再保存一套大统计。
其 mode 明确标记为 fresh-no-duplicate-save。缓存命中的 Stage0 时间/峰值只代表
加载，不能拿来与 fresh Stage0 计算比较；若新campaign复用已有完整缓存，其full-v4
Stage0也标为cache-hit，不能据此宣称重计算收益。

每次实验保留独立日志、配置、真实父wait、逐阶段allocator端点、采样NVML、主机PSS/RSS、
28层覆盖、全部state摘要、KL/PPL/所选QA、源码及缓存来源。遥测逐行写盘，不在内存中积累。
主机采样包含 partial 标记；NVML是采样值，不能冒充连续容量证明。
父进程wall含校验/解释器/CPU/QA等；`wall×2`只是在全程两卡独占假设下的GPU时折算。

量化 checkpoint 按实际全state内容存入共享目录，每臂只保存 `checkpoint.json` 引用；
数值相同的权重只保留一份，数值不同的权重单独保留。原始模型和旋转模型不复制到实验目录。
不会自动删除成功/失败记录；完整缓存损坏或上次producer中断未封存时拒绝覆盖，需人工核对。

## 安装（目标Linux服务器）

使用新的 Python 3.10 环境，不覆盖旧环境。确认GPU驱动支持CUDA 12.8，并有本地C++/CUDA
编译工具供 fast-hadamard-transform 构建。无需安装或启用FlashAttention，固定math SDPA。

```bash
python -m venv /workspace/realq-env
source /workspace/realq-env/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-build-isolation -r portable_requirements.txt
nvidia-smi -L
```

`prepare` 在下载模型前先检查固定运行栈，并在隐藏CUDA的独立子进程中实际导入
所选原生源码。若旧镜像的torchvision/torch等依赖不兼容，会在这里停止，
应修复新虚拟环境，而不是跳过检查或修改冻结算法源码。

## 准备与显式启动

下面两个 UUID 必须替换为**租赁实例实际分配的两张卡的完整 UUID**，不要使用旧服务器编号。
首先检查两卡空闲。示例仅展示命令，不表示已启动实验。

```bash
python -m portable_realq.campaign prepare \
  --root /workspace/realq-work \
  --devices GPU-REPLACE-A GPU-REPLACE-B \
  --models q06 q17 \
  --qa full
```

如果当前只执行核心消融，经确定范围后将上面 `--qa full` 改为 `--qa none`。
准备命令会输出新 `plan.json` 的路径。检查该计划后用它显式启动：

```bash
python -m portable_realq.campaign run \
  --plan /workspace/realq-work/runs/REPLACE-ID/plan.json
```

启动前逐文件检查资源、源码、运行栈和缓存，核对GPU UUID、实际主机/cgroup内存和磁盘。
当前版本使用保守的128GiB可用主机内存门以及至少40GiB剩余磁盘门（随原模型大小增加）；
这是最低准备门而非容量上界。建议持久空间预留150GiB或更多，以实际准备读数为准。
内核文件锁保证同一cache和选定GPU不被该入口的另一链并发使用；另核对实际活跃GPU进程。
只清理本次Popen创建的新进程组，不碰其他用户/旧实验进程。

失败后原计划不能再启动；没有自动实验重试。下载续传用prepare，实验修复需要新计划。
所有旧失败、部分文件、配置和结果保留。

## CPU验证

```bash
python -m pip install pytest
python -m pytest --rootdir=tests --confcutdir=tests tests/test_portable_realq.py -q
```

Windows旧工作区存在不可访问的datasets junction，故测试显式限制rootdir/confcutdir；
不删除或修改旧链接。CPU测试不加载真实模型/数据，不访问GPU，不下载大payload。
