# TurboBOA / YAQA_wclip Qwen3、Llama-3.1 公平 20 组实验记录（2026-08-21）

## 范围

- 模型：Qwen3-0.6B、Qwen3-4B、Qwen3-8B、Qwen3-32B、Llama-3.1-8B-Instruct；
- 每模型四个设定：W4A16KV16、W3A16KV16、W2A16KV16、W4A4KV4；
- 每种方法 20 个量化结果，后续统一测 KL、WikiText2 PPL、十项 QA、GSM8K、MATH500、HumanEval+ 和量化 GPU-hour；
- 两种方法都保存可重新加载的 checkpoint，评测不混入量化计时；
- EfficientQAT 另按论文的 weight-only 范围跑 15 组，W4A4KV4 为 N/A，不做 QuaRot。

## 共同冻结条件

- 校准集：各模型与 REALQ 20 组主表完全相同的 WikiText2 train token 制品；
- 每份制品严格为 256 条 `torch.int64[2048]`，按绝对路径和 SHA256 读取，禁止重新采样；
- calibration seed=1、global seed=1、`PYTHONHASHSEED=1`；
- W2/W3/W4 均为 signed symmetric uniform scalar quantization，natural-column G128；
- W4A4KV4 的 A/K/V 均为 symmetric per-token、groupsize=-1、clip=0.9；
- A16/KV16 是 BF16 identity，不实例化没有作用的激活量化器；
- 两台机器：`j-4mj21jb084-master-0` 与 `j-zogxxxduju-master-0`，各 8×L20C；
- 正式 Python：`.venv.py312-broken-20260729/bin/python`，并显式加入该 venv 的 `torch/lib`。

## TurboBOA-C 配置

- weight quantizer：仓库已有 `realq_mse` adapter，legacy Cartesian MSE/w_clip、strict first winner；
- `block_v=true, n_quant_rows=16, consider_dX=true, alpha=0.25`；
- `adaptive_qparam=true, refine_qparam=true, n_iters=1`；
- column act-order=true、row act-order=false；
- QuaRot=true、rotation seed=0；
- Qwen3 q_norm/k_norm output Hessian 使用已实现的 `mean_jacobian_kfac`；
- W4A4KV4 同时开启 activation-aware 与 K-cache-aware GPTQ；
- 计时包括 calibration load、precompute、QuaRot 和完整量化算法，排除 checkpoint IO 与评测；
- 方法名写作 `TurboBOA-C (RealQ-MSE, grouped-F3)`，披露 grouped F3 是受控适配而非 pristine upstream。

### 入口与测试

- 已增加精确 token 路径/SHA、禁止重采样入口；
- 已增加 Python/NumPy/PyTorch seed、cuBLAS workspace、deterministic algorithms、cuDNN deterministic、TF32=false；
- controlled-entrypoint 完整回归：`9 passed`；
- 新正式矩阵计划：`experiments/turboboa_fair20_20260821/plan.json`；
- 新 preflight、单卡队列 worker、launcher 与 fail-closed result gate 已实现；每张卡可在当前 EfficientQAT 进程退出后自动接续，正式 run 目录不可覆盖。

### 正式 20 组 v1

- 输出根目录：`/minimax-avatar-new/zhangqian/realq/experiment_data/turboboa_fair20_20260821_v1`；
- plan SHA256：`48ad6e6ca3df8c3af2194e09632e5b4db7380e9b2fabf414f666551db3d1e0cc`；
- preflight 通过，20 组已按模型/位宽排入两机队列；
- 与 YAQA 共用物理 GPU `flock`：队列只在前一方正式任务释放同卡锁后启动，避免 CUDA context 尚未建立时的空卡误判；
- 当前首组 `TB20-Q06-W4A4` 正在 `j-zogxxxduju-master-0` GPU7 运行。

### 已完成的量化结果

| run | 模型 | 设定 | algorithm GPU·h | precompute+rotation | quantization | checkpoint save | 状态 |
|---|---|---|---:|---:|---:|---:|---|
| TB20-Q06-W4A4 | Qwen3-0.6B | W4A4KV4 | 0.486136 | 210.903 s | 1539.187 s | 5.687 s（不计入） | 成功；1.5 GB checkpoint 安全重载通过 |
| TB20-Q06-W4 | Qwen3-0.6B | W4A16KV16 | 0.449602 | 见 immutable result/receipt | 见 immutable result/receipt | 不计入 | 成功 |
| TB20-Q06-W3 | Qwen3-0.6B | W3A16KV16 | 0.612925 | 见 immutable result/receipt | 见 immutable result/receipt | 不计入 | 成功 |
| TB20-Q06-W2 | Qwen3-0.6B | W2A16KV16 | 0.577745 | 见 immutable result/receipt | 见 immutable result/receipt | 不计入 | 成功 |
| TB20-Q4-W2 | Qwen3-4B | W2A16KV16 | 1.205710 | 见 immutable result/receipt | 见 immutable result/receipt | 不计入 | 成功 |

- 首组完成后，同队列已自动接续 `TB20-Q06-W4`，没有等待人工确认；
- CPU 上已验证 shared checkpoint 的 `weights_only=True` 安全反序列化、base-model identity、QuaRot/A/K/V runtime manifest 恢复与 SDPA 模型拓扑；CPU 有限值前向速度过慢，为不浪费控制机 CPU 已中止，改由正式 GPU 评测门禁执行。
- 14:32 CST，Qwen3-0.6B 队列前三组均成功，已自动进入最后一组
  `TB20-Q06-W2`；另有 `TB20-Q4-W2` 与 `TB20-L8-W4` 并行运行。其余 TurboBOA
  lane 已排在对应 EfficientQAT/恢复任务之后。

### Qwen3-0.6B W4A16 单层 canary v1

- 输出：`/minimax-avatar-new/zhangqian/realq/experiment_data/turboboa_fair20_20260821/_canary/qwen3-0.6b-w4-layer0-v1`；
- 设定：完整 256×2048 校准，W4 G128 symmetric REALQ-MSE，QuaRot seed=0，A16/KV16，完整确定性门禁，只量化 layer 0，跳过评测和 checkpoint；
- 状态：成功；
- 算法总时间：293.214099 秒（0.081448 GPU-hour）；
- 其中 precompute+rotation 214.402999 秒，layer-0 quantization 78.811100 秒；
- 没有 OOM、Traceback 或非有限值。

## YAQA_wclip 配置

- 直接使用已有 `lib/codebook/wclip.py`，不复制量化器；W2/W3/W4、signed symmetric、natural-column G128、legacy Cartesian MSE、strict first winner、V=1；
- Hessian：Sketch-B、power iteration=1、单卡 world size=1、batch=8、32 iterations，恰好消费 256 条序列；
- A16 三个 weight setting 共用每模型一份 Hessian；W4A4KV4 每模型另生成一份 activation-aware Hessian；
- raw quant：`ft_epochs=0, scale_override=1, td_x=1, td_y=128, sigma_reg=0.01`，不允许 recovery FT、skip、QTIP LUT 或 fallback；
- 保留 YAQA 原生双侧 randomized Hadamard、anti-diagonal LDLQ 和三项 error feedback，不额外叠加 QuaRot；
- 原生 RHT sign seed 固定为 `rotation_seed(0) + layer_index`；
- Hessian CLI 仍显示上游遗留的 `--seed 42`，但静态检查确认该参数在脚本中没有任何
  `args.seed` 读取，因而不进入数值路径；真正执行 Python/NumPy/Torch/CUDA 初始化的
  是 `--global_seed 1`。fake target 继续按方法原生的 calibration iteration index
  局部播种，RHT 则由 rotation seed 0 加 layer index 播种；
- weight work tensor 是 FP32；Hin/Hout normalization、RHT、regularization、block-LDL solver 是 FP64，随后转 FP32 进入 LDLQ，两个 dtype 独立写入 provenance；
- Hessian 采集的 fake target seed 保留论文实现的 calibration iteration index；
- 保存整数 code、compact scale、SU/SV 和 dense fake-dequantized weight，并做逐 projection 精确重构和双次 reload smoke。

### 已完成适配与 CPU 门禁

- 五模型 token 路径/SHA 表已冻结；
- Hessian 与 raw quant 均已增加 Python/NumPy/PyTorch/global seed、确定性 CUDA 合同和 manifest 校验；
- `qtip_kernels` 已改为 lazy import；dense `realq_wclip` 不再依赖 native QTIP CUDA 扩展；
- wclip、provenance、hfize 三个测试文件：`17 passed`；
- 无 QTIP 扩展导入冒烟：成功构造 `WClipQuantizer`。

### Qwen3-0.6B A16 Hessian layer-0 canary

- v1 输出：`.../yaqa_wclip_fair20_20260821/_canary/qwen3-0.6b-a16-hessian-layer0-v1`；
- v1 状态：入口瞬时失败；正式 venv 有 PyTorch module，但没有 `bin/torchrun` console wrapper，退出码 127；未加载模型、未占 GPU、不是 OOM；失败目录保留；
- v2 改为同一解释器的 `python -m torch.distributed.run`，其余算法与数据配置完全不变；
- v2 输出：`.../yaqa_wclip_fair20_20260821/_canary/qwen3-0.6b-a16-hessian-layer0-v2`；
- v2 状态：启动时与 TurboBOA 正式队列发生 CUDA context 初始化竞态；为不让两个算法共享同卡污染计时，已主动 TERM 父/子进程并保留目录。该 exit 1 是人工终止，不是算法失败或 OOM。

### 正式 20 组 v1

- 输出根目录：`/minimax-avatar-new/zhangqian/realq/experiment_data/yaqa_wclip_fair20_20260821_v1`；
- plan SHA256：`a7bbd0db640b6d87edbb286f640d744fa13c66c2adccc102d16be9253145bda1`；
- preflight 通过；wclip/provenance/hfize 回归为 `17 passed`，campaign 源码 `py_compile` 通过；
- 共 30 个 stage：5 模型 ×（A16 Hessian + A4-aware Hessian）共 10 个 Hessian stage，以及 20 个量化 stage；A16 的 W2/W3/W4 在同模型内共享 Hessian；
- 已按物理 GPU 全部排队。当前等待 EfficientQAT/TurboBOA 释放同卡锁，尚未计入 YAQA 算法 GPU-hour。

### 14:31 CST 队列恢复与顺序合同

为修复 EfficientQAT 的 FP32 门禁误判而插入恢复 worker 后，先前仅等待锁、尚未创建
run/stage manifest 的 TurboBOA/YAQA wrapper 被安全终止并重新排队。当前每条 lane 的
顺序冻结为：

- 有 EfficientQAT 恢复的 lane：恢复结束后运行 TurboBOA，再运行 YAQA；
- node0/GPU7 没有 TurboBOA 分配：EfficientQAT W4 恢复后直接运行 YAQA；
- node1/GPU5：正在运行的 TurboBOA Q4-W2 先完成，再运行 EfficientQAT Q06-W3
  恢复，最后运行 YAQA；
- node1/GPU6：EfficientQAT Q06-W2 已成功，TurboBOA Llama-W4 已于 14:29 启动，
  YAQA 在同一物理锁后等待；node1/GPU7 的 YAQA 排在当前 Q06 TurboBOA 队列之后。

14:43 CST，node0/GPU7 的 EfficientQAT W4 恢复成功并释放锁；`YH-Q06-A16` 于
14:43:21 自动开始。对应命令是正式 plan 固定的单卡
`python -m torch.distributed.run --nproc-per-node=1` Hessian producer，仍使用精确
Qwen3-0.6B 的 256×2048 token 制品，没有重新采样。

15:10 CST，`TB20-Q06-W2` 与 `TB20-Q4-W2` 均已成功；前者同卡链已进入
`YH-Q06-A4`，后者释放 GPU5 后 `EQ15-Q06-W3` 的审计恢复立即启动。15:13:16 CST，
`YH-Q06-A16` Hessian stage 成功并通过 validation，耗时与 provenance 写入其 immutable
stage receipt；对应 worker 将继续执行该 lane 的后续 YAQA stage。至此 TurboBOA 已完成
5/20，量化 failure 仍为 0。

串行 chain 使用原冻结 plan、原 plan SHA、同一正式 venv 和确定性环境；没有改动任一
算法配置。等待锁期间不计入算法 GPU-hour。

## 当前资源与进度（2026-08-21 13:37 CST）

- 16/16 张卡均有计算进程：15 张 EfficientQAT，1 张 TurboBOA；未见 OOM、Traceback 或 non-finite 错误；
- EfficientQAT Qwen3-0.6B W4/W2/W3 分别到 block 19/21/23；Qwen3-32B W2/W3 已进入 block 1，W4 仍在 block 0 打包；其他组日志持续增长；
- TurboBOA `TB20-Q06-W4A4` 已完成；`TB20-Q06-W4` 已量化到 block 11/28；无报错；
- YAQA 正式队列已启动，但受共享 GPU 锁保护，正常等待前序任务；
- 某一时刻的 `nvidia-smi` utilization=0% 对应 CPU 权重 pack/队列化阶段，显存、PID 和日志时间戳证明任务没有退出；
- 仍按约每五分钟人工检查日志、进程、显存与 GPU 利用率，不能只依赖守护程序。

### 2026-08-21 14:33 CST 更新

- 两机 16/16 条物理 lane 均已有当前任务或严格串行后继 worker；这表示没有未安排的
  实验槽位，不等价于每一秒都有 GPU kernel；
- node0 当次 5 秒采样均处在 EfficientQAT 的 CPU projection pack/materialize，SM 为
  0%，但 8 张卡对应显存、PID、CPU 活动和增长中的日志均存在；
- node1/GPU5 的 TurboBOA 在采样中 SM 为 10%—26%；GPU6/GPU7 的 TurboBOA 子进程
  分别处在 CPU 模型准备/旋转前置阶段，因此当时仅有很小或尚未建立 CUDA context；
- 未发现 OOM、Traceback、non-finite 或未经记录的瞬时退出。后续继续约每五分钟人工
  核对进程、日志新鲜度、GPU 采样和 immutable receipt/failure。

### 2026-08-21 15:12 CST 更新

- 12 条 EfficientQAT lane 已在 packed-weight byte-equivalent 优化和单线程 CPU 合同下
  重放；旧 partial artifact 全部归档，物理锁始终未释放，后继算法顺序未改变；
- node0 GPU7 运行 YAQA Qwen3-0.6B A16 Hessian；node1 GPU5 运行 EfficientQAT W3
  恢复，GPU6 运行 TurboBOA Llama-W4，GPU7 运行 YAQA Qwen3-0.6B A4 Hessian；其余
  12 卡运行 EfficientQAT 重放。两机 16/16 lane 均有当前正式工作；
- 采样时 node0 GPU2—6、node1 GPU2/5/6/7 有 CUDA kernel，Qwen3-32B 卡处于 checkpoint
  shard 加载；没有新增 failure。统一评测的 16 个等待式 worker 也已启动，但只会在各卡
  TurboBOA+YAQA 量化依赖成功并释放同一物理锁后开始，不与量化抢卡。

### Qwen3-0.6B A16 Hessian 干净重计时（2026-08-21 15:27—15:32 CST）

主动复核逐 iteration 时间发现，`YH-Q06-A16` 前 11/32 个 Hessian iteration 与旧版
EfficientQAT 的 12×144 CPU 线程 packing 重叠，约 100 s/iteration；清理超卖进程后立即
恢复到约 7 s/iteration。原 Hessian 张量数值有效，但其 1537.741447 s 计时不公平。

因此在 node1/GPU5 使用同一 token、seed、确定性合同和同一生产代码做一次独立干净
重放。结果为 327.021957 s，即 `0.090839432 GPU·h`；共 392 个 Hessian tensor 逐文件
SHA-256 与原正式 producer 完全一致，规范化 manifest 也一致。正式量化继续引用原
immutable Hessian tensor；公平表只把共享 A16 producer 的成本替换成该干净计时，原
`0.427150402 GPU·h` 作为受 CPU 争用污染的调试成本保留，不计入主表。

随后同一 GPU 插入 Qwen3-0.6B EfficientQAT 三组的干净重计时；物理锁未释放，统一评测
worker 仍等待。重计时完成后会自动重新启动该卡原计划中的 `YQ-Q4-W3/YQ-Q4-W2`
YAQA 队列，不改变冻结配置或其他 lane 的执行顺序。

### 早期 TurboBOA 结果的干净 GPU-hour 重放（2026-08-21 16:00—）

CPU 超卖窗口也覆盖了六个已成功 TurboBOA run：Qwen3-0.6B 的四个设定、Qwen3-4B
W2A16 和 Llama-3.1-8B W4A16。checkpoint 数值不受 CPU 调度影响，但这些 run 的墙钟
计时不能直接与 15:10 后运行的组公平比较。因此新增独立 clean-retime，不覆盖原 result
或 checkpoint：GPU5 串行重放 Q4-W2/L8-W4，GPU7 串行重放 Q06 四组。

重放严格复用原 plan SHA、token path/SHA、seed、rotation seed、quantizer、G128、位宽、
QuaRot 和全部 TurboBOA 参数，只把 OMP/MKL/OpenBLAS/NumExpr 线程固定为 1。每份新结果
先通过原 `validate_result`，再要求 source/clean configuration 完全相同，并以
`torch.load(weights_only=True,mmap=True)` 对 checkpoint primitive manifest、state_dict
键序、tensor dtype/shape/stride/value 逐项验证内容一致；只有全部通过才发布干净
GPU-hour。源 plan、worker、TurboBOA main 与 checkpoint writer 的 SHA 门禁及 GPU5/GPU7
`--check-only` 均通过。

16:00 CST，先利用原本等待 Q4 A16 Hessian 的 node1/GPU5 启动 Q4-W2/L8-W4 重放；该卡
原 YAQA/evaluator waiting worker 在尚未 claim stage/eval 时安全停止。supervisor 无论
重放成功或失败都会恢复原 YAQA GPU5 队列及修复后的统一 evaluator，因此不会形成后继
断链。16:01 CST，修复后的首个 Qwen3-8B suite 已原子发布 Exact-KL/PPL 与十项 QA
`quality_result.json`；随后在 GSM8K controlled-resume partial 安全点停止 evaluator，
16:02 CST 于 GPU7 启动 Q06 四组重放。重放 supervisor 完成后会自动用隔离的 EvalPlus
0.3.1 路径恢复 evaluator；已有质量结果与 reasoning partial 不会重算或丢失。

16:02 CST 全局量化进度为 TurboBOA 9/20 成功、6 个 run 运行、5 个等待，YAQA
4/30 stage 成功、4 个运行，二者 failure 均为 0。两条 clean-retime lane 也均已进入
真实量化，不与其他算法共享同卡计时。

### clean-retime 路径字段误判与无重算恢复（2026-08-21 16:11—）

GPU7 的第一组 `TB20-Q06-W4A4` 在 16:11 CST 已完整生成新 result 和 1.5 GB
checkpoint，干净算法计时为 561.842540 s，即 `0.156067372 GPU·h`，相比受线程超卖
污染的 `0.486136212 GPU·h` 明显缩短。随后审计 gate 报
`source and clean result configurations differ`。逐字段 diff 证明唯一差异是
`cache_dir/results_path/save_qmodel_path` 三个必然指向独立重放目录的输出路径；模型、
token、seed、QuaRot、quantizer、位宽、groupsize、clip/aware 和其余所有配置字段相同。
这不是算法失败，也不是 checkpoint 差异。

修正后的 gate 只从 configuration identity 中排除上述三个输出位置字段，仍要求其余字段
精确相同；checkpoint primitive manifest 与每个 tensor 的 dtype/shape/stride/value
门禁没有放宽。已有第一组不重算：保留原 `failure.json`，只有重新通过规范化配置和完整
checkpoint content identity 后，才会并列发布 `clean_replay_result.json`。修正 worker
SHA-256 为 `efadcf9d730e35768c87f071c646e48efe40c6dbeda14b7c0d269a5fb4c1235d`。

原 GPU7 supervisor 按 fail-safe 设计已于 16:11 自动恢复 evaluator；其在 16:01
quality safe point 被终止而遗留的 `.claim`，先验证 owner PID 已死亡，再完整改名保留为
`.claim.interrupted_for_turboboa_clean_retime_20260821T0802Z`，没有删除。evaluator 于
16:11 以 attempt 2 恢复 `efficientqat__EQ15-Q8-W4`，不会重算已发布 quality 结果。

node1/GPU4 当时因 YAQA `YQ-L8-W3` 等待跨卡 `YH-L8-A16` 而没有 CUDA 工作，但旧
YAQA worker 会在等待期间持有物理锁。16:16 CST 确认该 stage 尚未创建 manifest 后，
安全 TERM 仅等待锁的 worker，并把 Q06 clean-retime recovery 移至该空闲物理卡；使用
同型号 L20C、同一 node、同一单线程环境，run group 仍是原四个 Q06 设定。恢复 supervisor
脚本 SHA-256 为 `5e847a9b3ccfe188cb6ba1f2c0ec14336723f2dc4f13ec8b840d87b107244216`；
无论重计时成功或失败都会自动重启原 YAQA GPU4 队列。

上述 GPU4 supervisor 启动后，脚本又只做了 orchestration 泛化，使后续 GPU5 组也可在
任一经检查为空闲的 node1 物理卡上恢复，并动态重启同卡 YAQA worker；没有改变正在运行
的 GPU4 进程镜像。泛化版本通过 `bash -n`，SHA-256 为
`7be505a7e52d02675bbde596d6ec3a591a6259624628bbf5de207056981dec82`。

16:26 CST，GPU4 recovery 已完成两项严格门禁。W4A4 的既有 clean checkpoint 在不重算
的情况下确认 535 个 tensor 全相等，成本 `0.156067372 GPU·h`；随后 W4 新重放也确认
535 个 tensor 全相等，成本 `0.130901890 GPU·h`，原受污染值 `0.449602` 排除。队列已
进入 Q06-W3。上述 identity 比较还要求两份 checkpoint 文件字节数一致，并不只比较
最终质量指标。

### 2026-08-21 16:34 CST 人工巡检

- Canoe 只读门禁确认两台一周 sleep debug job 均为 `Running`。TurboBOA 正式矩阵为
  14/20 成功、5 个运行、1 个等待、0 个失败；YAQA 为 8/30 stage 成功、7 个运行、
  15 个等待、0 个失败。两机 16 条 lane 均有受管前驱或后继 worker；Hessian/LDLQ 的
  CPU 求解阶段允许瞬时低 SM，不能据单次 `nvidia-smi` 0% 判定任务退出。
- TurboBOA 32B W4A4/W4/W3 分别到约 8/64、8/64、10/64 block，W2 同步运行；
  Llama-8B W3 到 20/32。Q06 clean-retime W3 到 27/28，随后仍会自动执行 W2。
- node1/GPU5 的旧版 clean-retime worker 于 16:31:23 按预期在唯一的输出路径字段差异上
  fail closed，并自动恢复该卡 YAQA/evaluator。它已经完整生成 Q4-W2 clean result 与
  8.8 GB checkpoint，算法计时 `1851.139106 s = 0.514205307 GPU·h`，相对原受污染
  `1.205710 GPU·h` 显著下降；该值在修正版 worker 完成规范化配置与逐 tensor identity
  恢复前仍标为 provisional，不提前写入最终表，也不会重跑这次量化。
- Qwen3-8B A16/A4 Hessian 分别到 24/32、20/32。统一评测已有 2 个 standard quality
  result；Q32-W2 与 Q8-W4 的 GSM8K controlled-resume 分别到 192/1319、896/1319。
  node1/GPU3 的 Q32-W4 quality-only worker 已建立约 9.6 GB CUDA context 并继续运行，
  不是空闲或失联；完成后 supervisor 会恢复原 YAQA worker。

16:45 CST，机会式 `efficientqat__EQ15-Q32-W4` quality-only 完整通过冻结 evaluator
门禁并原子发布：Exact-KL=`0.1534895301`、WikiText2 PPL=`6.5034666`、十项 QA
Avg=`72.87`。它没有执行 reasoning，也没有占用量化 GPU-hour。supervisor 随即按合同
恢复 node1/GPU3 的 YAQA worker，`YQ-Q8-W3` 已使用刚发布的正式 A16 Hessian 启动；
同卡常驻 evaluator 继续等待物理锁，没有重复 claim。

同一时刻 `YH-Q8-A16` 的 32/32 iteration、tensor 保存、validation 与 receipt 全部成功，
成本为 `0.565212618 GPU·h`；A4-aware Hessian 仍在最后几个 iteration。上述两个完成事件
均由 immutable terminal 文件确认，而不是只依据进程退出或日志百分比。

16:46 CST，Q06 clean-retime 四组全部完成，`summary_gpu4_group7.json` 的内容状态为
`succeeded`。最后两组 W3/W2 的干净成本分别为 `0.169892067` 与 `0.175807419`
GPU·h，四组均保持 535 个 checkpoint tensor 内容相等。运行中的旧版 supervisor inode
在成功 JSON 打印之后，因旧日志字符串中的 shell command-substitution 写法误执行
`run-group-gpu`，把 supervisor 最终 rc 错记为 127；它没有改变 worker 已发布的成功
terminal，且仍按 finally 路径恢复了 YAQA。当前磁盘上的泛化脚本 SHA
`7be505a7e52d02675bbde596d6ec3a591a6259624628bbf5de207056981dec82`
已通过 `bash -n` 与 group5 `--check-only`，不存在该旧日志写法。

恢复后的 GPU4 YAQA 只在等待跨卡 `YH-L8-A16`，没有子进程、CUDA context 或
`YQ-L8-W3/W2` stage manifest。16:47 CST 安全 TERM 该等待 worker 后，以当前泛化脚本
在物理 GPU4 执行原 GPU5 clean-retime group。修正版 gate 在不重跑 Q4-W2 的情况下
确认 687 个 checkpoint tensor（16,090,311,936 tensor bytes）全部相等，正式干净成本
为 `0.514205307 GPU·h`，原受污染 `1.205709770 GPU·h` 排除。随后同一 supervisor
于 16:48 CST 启动 `TB20-L8-W4` 干净重放；结束后会再次恢复 GPU4 的 YAQA 队列。

16:53 CST 复核推理进度时发现，一次临时只读统计把 canonical task 目录误写成 `math`
而不是冻结协议中的 `math_500`，从而把正在运行的 MATH-500 临时显示为 0。正式
evaluator、manifest、reference contract 与 release builder 始终使用正确的 `math_500`，
没有配置或结果错误。改用 canonical 名称后确认 Q8-W4 的 GSM8K 已完成 1319/1319，
MATH-500 自 16:45 已运行到 96/500。

### 2026-08-21 17:00 CST 人工巡检

- TurboBOA 正式矩阵 15/20 成功、5 个运行、0 failure；Q32
  W4A4/W4/W3/W2 分别到 17/64、17/64、20/64、32/64，Llama-W2 到 6/32。
  Llama-W4 clean-retime 到 10/32。
- YAQA 为 11/30 stage 成功、7 个运行、12 个等待、0 failure；两份 Qwen3-8B
  Hessian 已 terminal，W4/W3/W4A4 三条量化 lane 均在运行。
- 统一评测保持 3 份 quality；Q8-W4 MATH-500 到 160/500，Q32-W2 与 Q32-W4
  GSM8K 分别到 576/1319、160/1319。official scorer 正常每 30 秒轮询，等待第一份
  HumanEval+ generation terminal。
- 两台机器 16/16 lane 均有受管进程；本次快照 node0/node1 各至少两卡在
  86%--100% SM，其他卡位于 YAQA 的 CPU/内存阶段。没有 zombie、OOM、Traceback、
  non-finite 或新的 failure terminal。

### Llama-W4 clean-retime 完成与第二次空窗 quality（2026-08-21 17:27—）

`TB20-L8-W4` 于 17:26:51 CST 完成干净重放并通过完整 identity gate：547 个
checkpoint tensor、30,019,266,560 tensor bytes 的 dtype/shape/stride/value 全部相等；
干净成本为 `0.628342185 GPU·h`，原受污染值 `0.950861098 GPU·h` 排除。group5
summary 同时包含已恢复的 Q4-W2，内容状态 `succeeded`、supervisor rc=0。至此六个受
早期 CPU 超卖污染的 TurboBOA run 都已有可供 release builder 强制校验的 clean terminal。

supervisor 恢复 node1/GPU4 YAQA 后，该 worker 仍只等待跨卡 `YH-L8-A16`；再次确认
`YQ-L8-W3/W2` 无目录/manifest、无子进程、无 CUDA context。17:27 CST 安全 TERM
等待 worker，并以 `opportunistic_worker --quality-only --check-only` 确认 lane dependency
未完成、可用 quality 列表和 reference fingerprint 均通过。17:28:24 CST，GPU4 claim
`efficientqat__EQ15-Q32-W3` 并开始 frozen quality-only；它只发布标准 quality，不执行
reasoning，结束后 supervisor 自动恢复同卡 YAQA。量化、teacher、evaluator、seed 与
reference manifest 均未修改。

### MATH-500 缺包的无生成恢复与 17:43 人工巡检

17:33:37 CST，`efficientqat__EQ15-Q8-W4` 已完整写出 500/500 条 MATH-500 generation，
随后只在 CPU scoring 的 `math_verify` 导入点失败。失败 manifest SHA-256 为
`be54a84e3b2206f1a7ab5d9ba1d14dceaaaf09c894d0723433f5018e86da879c`，原始
`generations.jsonl` SHA-256 为
`c03339b78c7573f01bf9c6b81fcaa40694f31a800dbd544e20cca60c02fe9c0a`；文件恰有
500 行且 500 个唯一 `(task, sample_id, sample_index)`，数据集仍为冻结的
`35dc4108...a06132`。因此这不是 checkpoint、显存或 generation 故障。

现有隔离 Python path 原先只发布 EvalPlus。17:36 CST 在同一目录精确加入
`math-verify==0.9.0`、`latex2sympy2_extended==1.11.0` 和
`antlr4_python3_runtime==4.13.2` 的六个 package/dist-info symlink，没有加入 vendor
根中的旧 `huggingface_hub`。两台 pod 分别用正式 venv 验证：上述三包与 EvalPlus
0.3.1 均从隔离路径导入，Transformers 4.56.2、huggingface_hub 0.36.2、Torch 2.9.1
仍来自正式 venv，已知 `\\boxed{2}` 样例评分为 true，CUDA 未初始化。一次预发布只读
validator 因 `Path.resolve()` 跟随 package symlink 而误判导入来源；修正为同时核对词法
导入路径和真实 target 后，500 条顺序、chunk seed、generation-config fingerprint、
dataset 与 package 门禁全部通过，期间没有修改任何评测产物。

17:43:14 CST 使用 scorer-only recovery 正式收口：不加载模型、不使用 CUDA、不重新生成
或修改候选，只调用仓库原 `score_or_export`，耗时 `1.692278 s`，结果
MATH-500=`284/500=56.80%`。完成 manifest SHA-256 为
`408d994bb91a5bde2d044ed04d3c31053c4785a3b7f0ed0b07a6c100635c32ad`，
scores SHA-256 为 `dde3c9c9f349858c02faaa22fb8d993f0e8168a06bee68d946987d9b1ce9480f`；
receipt 明确记录 `generation_reused_without_modification=true`、`model_loaded=false`、
`cuda_used=false`，并以 hardlink 保留原失败 manifest。执行源码 SHA-256 为
`ec0b0ffe9a94b22254e7006266cd0d1d12240db63ccff8ec0bd090f5e1d55488`。

同轮人工巡检确认 TurboBOA 为 16/20 成功、4 个 Qwen3-32B run 运行、0 failure；YAQA
为 11/30 stage 成功、9 个 stage 运行、10 个等待、0 failure。统一评测已有 5 份
quality、0 份完整 suite；Q32-W2/Q32-W4 GSM8K 分别为 1280/1319、864/1319，新的
TurboBOA Q8-W4A4 已完成 quality（Exact-KL=`0.3667215407`、PPL=`12.6266823`、
QA Avg=`60.73`）并进入 GSM8K。机会式 Q32-W3 quality 也已发布：
Exact-KL=`0.2585729957`、PPL=`7.6540432`、QA Avg=`68.73`，supervisor 随后恢复
YAQA。两机 16/16 lane 都有受管量化、Hessian 或评测进程；node1/GPU4 的新 YAQA worker
与 GPU7 的 evaluator 当时分别处于启动/CPU 阶段，单次 SM=0 不是空槽。Q8-W4 的下一次
suite attempt 将跳过已完成的 quality、GSM8K 和上述 MATH，直接续跑 HumanEval+。

17:45 CST 再次以 `build_release --allow-incomplete` 做只读预检，冻结 source/reference/plan
门禁全部通过并正确返回 `0/55 complete, 55 missing`；未指定 output，因而没有发布半成品。

17:48—17:50 CST，node1/GPU4 连续两轮仍为 0 MiB/0%：精确核对 YAQA worker PID 55390
无子进程、卡上无 compute PID，且 `YQ-L8-W3/W2` 目录均不存在；其唯一阻塞条件仍是
node0 正在生成的 `YH-L8-A16`。opportunistic `--check-only` 同时确认 lane dependency
未完成且存在未评 quality。为允许同一卡多次安全利用空窗，将 supervisor 的 YAQA
relaunch log 改为带安全 `RUN_TAG` 的不可覆盖路径；`bash -n` 通过，新 SHA-256 为
`6932e8b56644a26cff75d9969c929760d31bbe32c01d98c4e2826538c2024381`。
在上述五项只读门禁通过后 TERM 仅等待中的 worker；17:50:07 CST，GPU4 已 claim
`turboboa__TB20-Q8-W4` 做 frozen quality-only。supervisor 结束后会用独立日志
`gpu4.after_opportunistic_quality.quality3_20260821T0950Z.log` 自动恢复 YAQA，不改变任何
量化或评测数值配置。

17:52 CST 基于当前实测层速给出全部收口 ETA=`12--18 h`（预计次日上午）。TurboBOA
四个 32B run 约余 2--3 h；关键不确定项是尚未进入主体的 YAQA 32B 四设置及其长生成。
历史同模型证据中 YAQA 32B raw 每设置四卡约 4.1 h，32B MATH-500 单任务随 checkpoint
质量约 2.4--5.2 h；本轮为四设置分卡且量化/评测可流水重叠，因此采用区间而不是单点
承诺。后续将按本轮首个 32B Hessian/raw 批次的实测速度重新收紧。

17:59 CST 进一步逐 stage 核对冻结 plan 后，立即撤回上述 `12--18 h` 估计并更正为
`36--48 h`。上述首估错误地把历史“每设置 4 卡并行、约 4.1 h wall”的 YAQA 32B raw
直接当成本轮 wall；本轮每个 stage 实际是单卡，并且 `YQ-Q32-W3`、`YQ-Q32-W2` 都固定
在 node1/GPU1 串行。历史 raw 成本约 `16.45 GPU·h/设置`，也与本轮 Qwen3-8B 当前约
6.5 min/layer 的单卡实测按参数规模外推一致，因此 W3+W2 关键路径本身约 28--34 h，
其前还有单卡 A16 Hessian。其他卡会继续由其余量化与 55-suite 评测占用，但不能缩短这条
串行依赖；最终 ETA 将在本轮首个 Q32 Hessian 和 4 层 raw 后再次用实测更新。

### 2026-08-21 17:57 CST 人工巡检

- 两机 cgroup memory.current 分别约 `874/1000 GiB`、`829/1000 GiB`，两侧
  `memory.events max/oom/oom_kill=0/0/0`；16 卡均有受管上下文，未发现失联进程。
- TurboBOA 仍为 16/20、0 failure，Q32 W2/W3/W4/W4A4 分别推进到
  `53/40/39/39`（共 64）层；YAQA 仍为 11/30 terminal、0 failure，七条 raw 量化
  分别持续增长到约 11--28 层，Llama A16 Hessian 日志仍刷新。
- 统一评测 Q32-W2 MATH=`32/500`、Q32-W4 GSM=`1056/1319`、Turbo Q8-W4A4
  GSM=`160/1319`。机会式 Turbo Q8-W4 quality 已加载约 56 GiB 模型上下文并持有正确
  claim；单点 SM=0 对应 CPU/请求构造阶段，不是空槽或失败。

### 2026-08-21 18:05 CST 人工巡检

- `turboboa__TB20-Q8-W4` 的机会式 quality-only 已于 17:57:09 CST 原子发布并正常把
  node1/GPU4 交还 YAQA worker：Exact-KL=`0.0443699211`、WikiText2
  PPL=`10.0744648`、十项 QA Avg=`67.75`。跨卡 `YH-L8-A16` 随后于
  18:07:54 CST 通过 32/32 iteration、完整 validation 与 receipt 门禁，成本
  `0.495976710 GPU·h`；node1/GPU4 在下一秒启动 `YQ-L8-W3`，没有漏接或重复计算。
- TurboBOA 仍为 16/20 terminal、0 failure，四个 Qwen3-32B run 的 block 进度为
  W2/W3/W4/W4A4=`56/44/42/42`（共 64）。YAQA 仍为 11/30 terminal、0 failure；
  在途 Qwen3-4B/Qwen3-8B/Llama-8B raw 均继续增长，没有停滞或 non-finite。
- 统一评测现有 6 份 quality、0 份完整 suite。Q32-W2 已完成 GSM8K 1319/1319 并进入
  MATH-500（64/500）；Q32-W4 GSM8K=`1184/1319`，Turbo Q8-W4A4
  GSM8K=`288/1319`。Q8-W4 之前的 MATH 缺包 attempt 仍是唯一历史失败记录，已由
  scorer-only recovery 收口，下一次 claim 会从 HumanEval+ 续跑。
- 两机 `memory.events max/oom/oom_kill` 仍均为 `0/0/0`。node0 的
  `memory.current` 约 903 GiB，但其中约 689 GiB 是可回收 file cache、匿名内存约
  185 GiB；node1 结构类似，因此当前不是匿名内存逼近上限。16 卡中只有 node1/GPU4
  因上述跨卡依赖短暂无 CUDA context，其余 15 卡均有受管计算或模型上下文。

### 2026-08-21 19:13 CST 量化收口与 TurboBOA Q32-W3 异常

- TurboBOA 正式量化已达到 **20/20 terminal、0 failure**；YAQA 为 **15/30 stage
  terminal、9 个已 claim、0 failure**。统一评测已发布 9 份 quality。
- 新发布的 `TB20-Q32-W3` quality 明显反常：Exact-KL=`4.2886948586`、
  PPL=`385.1393433`、QA Avg=`34.36`；同模型更低 bit 的 `TB20-Q32-W2` 反而为
  KL=`0.9873412848`、PPL=`15.8305025`、QA Avg=`55.46`。因此 W3 结果暂时标记为
  **anomaly/不可直接用于公平结论**，待 checkpoint/loader 或算法稳定性复核。
- 第一轮只读核对没有发现任务层故障：W3 使用正确的 Qwen3-32B artifact、冻结
  calibration token SHA、W3/group128/symmetric/rotate/act-order 配置；量化
  `returncode=0`、64/64 block 完成、checkpoint 约 65.53 GB，日志中无 NaN、Inf、
  traceback 或 fallback。异常同时出现在 KL、PPL 和十项 QA，因而不像单一评测器
  的抄表错误；下一步需和 W4/W2 做 checkpoint tensor/statistics 与独立重载复算。
- 19:25 CST 继续逐层对照量化日志后，异常第一次出现于第 44 个 block 的
  `mlp.down_proj`：W3 的局部 surrogate loss 从正常的数百量级突然变为
  `2,660,257,536 -> 12,499,078`，随后第 45 层多个线性层目标转为大幅负值；同位置
  W4 为 `8,613,501 -> 1,090,292`，W2 仅为 `12,658 -> -13,093`。TurboBOA 的
  `consider_dX` 目标含线性项，负值本身不能直接判为数学错误，但 W3 在同一层的
  九个数量级跃迁与其下游退化高度相关，已作为 checkpoint/算法稳定性复核的首要
  定位点；在独立重载和 W4 quality 出来前仍不发布该点的公平排名。
- 为兼顾 REAL-Q debug，node1/GPU1 在等待 Q32 Hessian 且无 CUDA context 的空窗被
  暂借给 Qwen3-4B W4A16 精确旧最优 LR 复核；其余 15 lane 继续 YAQA 或统一评测。
  supervisor 会在复核结束后恢复 YAQA GPU1 队列，不停止任何正在计算的任务。

### 2026-08-21 20:12 CST 人工巡检

- TurboBOA 量化保持 20/20 成功、0 failure；YAQA 增至 16/30 stage terminal、
  0 failure。新 terminal 是 Llama-3.1-8B W4A4KV4 完成 raw、hfize、逐 projection
  重构与 reload validation 后发布，不是只看主进程退出码。
- Qwen3-32B A4/A16 两条 YAQA Hessian 均到 8/32 iteration。统一评测保持 9 份
  quality、0 份完整三任务 suite；EQ15-Q32-W2 已进入 HumanEval+（80 条），
  EQ15-Q32-W3 已完成 GSM8K 并进入 MATH-500，其他六条 generation 继续增长。
- REAL-Q debug 固定占 node1/GPU1 一卡，其余 15 卡继续三种新方法的量化/评测；
  4B-W3 all-off 到 11/36，约 129 s/layer。该 pair 后只插入一份约 4--5 分钟的
  current-code deterministic-SDPA cache producer，再接 8B allopts pair，最后自动
  归还 YAQA/evaluator。
- node0 cgroup 约 `1034.9/1073.7 GB`，既有 `memory.events max=60604` 未增加，
  `oom/oom_kill=0/0`。node1 在并行 mmap 旧/new cache 取样期间出现
  `max=39840`，但 `oom/oom_kill=0/0`，取样结束后约 943 GB；其中约 908 GB 是
  可回收 file cache，匿名内存约 36 GB。已停止后续并行 cache 扫描，只保留轻量状态
  查询和正式 GPU 任务。

### 2026-08-21 20:27 CST 人工巡检

- `TB20-Q32-W4A4` 已由统一 evaluator 原子发布 quality：Exact-KL=`0.4039516747`、
  WikiText2 PPL=`9.3451729`、十项 QA Avg=`66.76`，随后继续跑三项推理；统一评测
  因而增至 10 份 quality、0 份完整 suite。
- YAQA 保持 16/30 stage terminal、0 failure；Qwen3-32B A4/A16 两条 Hessian
  均推进到 11/32 iteration。两台机器的 16 个物理 GPU 锁全部由受管任务持有；
  REAL-Q debug 只占 node1/GPU1，其余 15 条 lane 继续三种新方法量化或评测。
- node0 的 `memory.current` 约 1044 GB，其中 `anon≈606.5 GB`、`file≈436.3 GB`
  （含 `shmem≈298.4 GB`）；既有 `memory.events max=60604` 仍未增长，
  `oom/oom_kill=0/0`。node1 同样无新增 memory event。当前不主动回收 cache 或改变
  冻结 batch 配置，继续以 OOM/kill、max 计数和进程存活作为干预门槛。

20:44:37 CST，`YQ-Q8-W3` 完成 raw、hfize 和独立 reload/reconstruction validation，
使 YAQA 达到 17/30 stage terminal、0 failure。冻结算法计时为
`14,015.991533 s = 3.893331 GPU·h`；同一 node1/GPU3 worker 随即开始队列下一项
`YQ-Q8-W2`，没有 GPU 空窗。20:46 巡检时两条 Q32 Hessian 均为 14/32，16/16
物理锁仍全部繁忙。

### 2026-08-21 20:47--20:59 CST TurboBOA Q32-W4 复核与资源巡检

- `TB20-Q32-W4` 的统一 quality 结果为 Exact-KL=`1.8178315163`、WikiText2
  PPL=`35.0668221`、十项 QA Avg=`43.17`。它不仅弱于 W4A4，也弱于同模型 W2，
  因而和 W3 一并标记为 **algorithmic anomaly**；三项推理 generation 继续保留。
  最终公平表按既定“纯算法崩坏也据实保留”口径列出原始结果并加异常注释，不用质量
  fallback 替换，也不把异常点静默删除。
- 四个 Q32 日志在第 44 个 block 的 `mlp.down_proj` 都出现目标不连续，但幅度强烈依赖
  setting：W2=`12,658 -> -13,093`、W3=`2,660,257,536 -> 12,499,078`、
  W4=`8,613,501 -> 1,090,293`、W4A4=`65,231 -> -13,896`。W3/W4 随后从第 45 个
  block 起出现大范围负目标，并分别对应 KL=`4.2887/1.8178`；这把首要嫌疑进一步收敛
  到 Q32 的 `consider_dX` 补偿/局部二次目标轨迹，而不是 evaluator 抄表或单个 W3
  checkpoint 路径错误。当前没有 NaN、Inf、traceback 或 fallback，仍需独立重载和张量
  统计后才能下最终结论。
- 20:59:01 CST，`YQ-Q4-W2` 完成 raw、hfize 和 reload validation，YAQA 增至
  **18/30 stage terminal、0 failure**；其量化成本（不含共享 Hessian）为
  `2.165929 GPU·h`，独立完整成本为 `2.505664 GPU·h`，摊销口径为
  `2.279174 GPU·h`。node1/GPU5 随后可由统一 evaluator 自动接管。
- 21:02:00 CST，`YQ-Q8-W4` 也完成完整 raw、hfize 与独立 reload/reconstruction
  validation，YAQA 继续增至 **19/30 stage terminal、0 failure**。其量化成本（不含
  共享 Hessian）为 `4.131928 GPU·h`，独立完整成本为 `4.697141 GPU·h`，摊销口径为
  `4.320333 GPU·h`；对应 lane 随即继续冻结队列。
- 21:06:41 CST，`YQ-Q8-W4A4` 完成相同完整门禁，YAQA 达到 **20/30 stage
  terminal、0 failure**。其量化成本（不含共享 Hessian）为 `4.132477 GPU·h`，独立
  完整与摊销口径均为 `4.698317 GPU·h`。21:08 巡检时两条 Q32 Hessian 同步到
  17/32；统一 evaluator 同时发布第二份完整 suite（`EQ15-Q32-W4`），正式 failure=0。
- 同轮两机仍为 16/16 物理锁受管；REAL-Q debug 固定在 node1/GPU1，其余 lane 跑
  EfficientQAT/TurboBOA/YAQA。node0 两条 Q32 Hessian 均到 17/32。两机
  `oom/oom_kill=0/0`；node0 因约 298 GB shmem 与大文件 cache 发生 direct reclaim，
  `memory.events max` 增至 114009，但没有进程退出或 failure terminal，继续按
  OOM/kill 与进程存活门槛监控，不改冻结 batch/算法配置。

历史独立实验提供了同源复现证据：
`docs/TurboBOA与YAQA_wclip_Qwen3重跑实验记录_20260727.md` 中，旧 Q32-W3 同样在
第 44 个 block 的 `mlp.down_proj` 发生灾难性范围扩张，旧 Exact-KL=`5.7259969711`，
本轮为 `4.2886948586`；旧/本轮 Q32-W4A4 的 Exact-KL 则分别为
`0.4029224515/0.4039516747`，绝对差仅 `0.0010292232`。旧记录当时已通过 checkpoint
严格加载、有限性和张量范围审计，并明确判定为纯 TurboBOA F2/F3 残差代理的深层不稳定。
这显著降低了本轮 calibration、loader 或 evaluator 配错的可能性；后续 checkpoint
张量统计用于完成审计闭环，而不是据结果优劣决定是否替换产物。

### 2026-08-21 21:14--21:30 CST YAQA Qwen3-8B 首批质量结果

- `YQ-Q8-W4` 的统一 evaluator 结果为 Exact-KL=`0.0322305337`、WikiText2
  PPL=`9.8533049`、十项 QA Avg=`67.46`。它直接绑定正式
  `stage_receipt.json`、共享旧 SDPA BF16 teacher SHA
  `f68f0312...e5decb` 和同一十项 QA 协议；当前继续生成三项推理。
- `YQ-Q8-W4A4` 的质量则发生跨指标灾难性退化：Exact-KL=`8.0090198517`、
  PPL=`16333.8125`、十项 QA Avg=`30.04`。这不是单列抄表异常；三项指标由同一次
  checkpoint reload 得到，且十项 QA 中 `lambada_openai=0`、其余多数接近随机水平。
- 第一轮身份审计确认该组使用正确的 Qwen3-8B token SHA
  `a7acbd90...d3b6f`、独立 `YH-Q8-A4` activation-aware Hessian、W4/G128/symmetric
  `realq_wclip` 权重量化，以及 A/K/V 4-bit、symmetric per-token、groupsize=-1、
  clip=0.9 的部署拓扑；252 个 projection 的整数重构和两次 reload 确定性门禁均通过，
  无 NaN、Inf、traceback 或 fallback。因此正式表暂按 **YAQA W4A4 algorithmic
  anomaly** 保留原值，继续完成三项推理，不以 W4A16 checkpoint 或关闭 activation
  wrapper 的结果替换。后续再用独立受控前向区分 activation-aware 权重与部署 A/K/V
  QDQ 各自的贡献。
- 21:30 人工巡检时，YAQA terminal 仍为 `20/30`、failure=0；两条 Qwen3-32B
  Hessian 分别到 `21/32` 与 `20/32`，REAL-Q Qwen3-8B 综合全关对照到 `3/36`。
  两机 16 张卡均有受管任务，`oom/oom_kill=0/0`；node1 GPU3/4 的低显存采样经
  PID、multiprocessing 子进程和同分钟持续增长的 quantize log 核验为逐层
  CPU/LDLQ 求解与卸载，不是 worker 退出或空卡。

### 2026-08-21 21:53--22:03 CST REAL-Q / 新算法并行巡检

- Canoe API 确认两台 debug job 均为 `Running`、retry=0；两个 pod 均 Ready。node0
  仅保留 11:53 启动前的短暂 `FailedMount` 重试事件，11:54 后持续运行；node1 的
  dead-node 快检无异常。MetaGod 因当前身份无读权限而跳过，不能据此声称已排除硬件
  故障，但 SSH、节点 `up{}`、GPU 采样与增长中的日志均证明本轮没有失联。
- 两次间隔约五分钟的人工检查中，16 张物理卡均由受管任务占用。REAL-Q 固定独占
  node1/GPU1，Qwen3-8B W4A4 综合全关对照由 `8/36` 推进到 `10/36`，稳定约
  `253 s/layer`；其余 15 条 lane 继续 YAQA 或统一评测。连续三秒 GPU 采样也确认
  Hessian/LDLQ 层间瞬时低利用率会在下一采样恢复，不是空卡。
- 两条 Qwen3-32B Hessian 同步推进到 `25/32`；Llama-8B W4 到第 29 层、Qwen3-8B
  W2 到第 11 层。22:03:25 CST，`YQ-L8-W3` 完成 raw、hfize、逐 projection 整数
  重构和双次 reload validation，YAQA 增至 **21/30 stage terminal、0 failure**。
  其量化主体为 `13,757.543477 s = 3.821540 GPU·h`，加入共享 A16 Hessian 的三组
  摊销后为 **`3.986865425 GPU·h`**；同一受锁 worker 随即启动 `YQ-L8-W2`。
- 统一评测当前为 `13/55` 份 quality、`3/55` 份 generation suite、`3/55` 份官方
  HumanEval+ terminal，formal failure=0。活跃任务均继续增长；例如
  `TB20-Q8-W4A4` HumanEval+ 由 48/164 推进到 80/164。
- 两机 `oom/oom_kill=0/0`。node0 `memory.current` 约 1.051/1.074 TB，表面余量约
  21 GiB；进一步拆分为 anon 约 605 GB、file 约 444 GB（其中 shmem 约 298 GB），
  仍有约 146 GB 非 shmem 文件页可回收。`memory.events max=222839` 在本轮多次采样间
  未增长，因此继续监控而不改变冻结 batch 或算法配置。

### 2026-08-21 22:21--22:28 CST 新 suite 与并行巡检

- `TB20-Q8-W4A4` 已完成同一冻结协议下的 PPL、十项 QA 和三项推理 generation，并由
  独立官方 scorer 闭合 HumanEval+ terminal。结果为 Exact-KL=`0.3667215407`、
  WikiText2 PPL=`12.6266823`、十项 QA Avg=`60.73`、GSM8K=`76.4215%`、
  MATH-500=`45.60%`、HumanEval base/plus=`23.7805%/20.7317%`。对应正式量化成本为
  `0.709578 GPU·h`；suite/official receipt 分别于 22:21:43/22:22:33 CST 原子发布。
- 原 lane 随即自动领取 `TB20-Q8-W3`，并于 22:28:48 CST 发布 quality：
  Exact-KL=`0.1632449776`、WikiText2 PPL=`10.8699217`、十项 QA Avg=`64.78`，
  正式量化成本=`0.781787 GPU·h`；随后继续同一 suite 的三项推理。统一评测因此达到
  **14/55 quality、4/55 generation suite、4/55 official terminal，formal
  failure=0**。该 lane 没有等待人工确认或空置；其余九个长期 claim 也都有存活
  runner 和持续增长的 generation 文件。
- REAL-Q 仍仅占 node1/GPU1，Qwen3-8B W4A4 综合全关对照推进到 `17/36`，稳定约
  `252--253 s/layer`。两条 YAQA Qwen3-32B A4/A16 Hessian 均到 `30/32`；
  Llama-8B W4 已完成 32 层 raw 量化主体（`15190.073921 s`），正在执行独立
  hfize/reload/reconstruction 门禁，尚未提前计为 terminal。
- 两机 16 条物理 lane 均由受管任务持有。node0 GPU0 的 Hessian iteration 边界和
  GPU6 的 checkpoint validation 会出现瞬时低 SM；PID、显存上下文和日志均继续更新。
  两侧 `oom/oom_kill=0/0`，不因短时低利用率改动冻结 batch、量化器或评测配置。

22:30:29 CST，`YQ-L8-W4` 通过 224 个 projection 的整数重构、双次 reload 确定性和
smoke-logit validation，YAQA 达到 **22/30 stage terminal、0 failure**。量化主体成本为
`4.219465 GPU·h`，加上 A16 Hessian 三组摊销后的 campaign 成本为
`4.384791 GPU·h`，单独完整成本为 `4.715442 GPU·h`。对应物理 lane 随即释放给统一
evaluator；上述 22:28 的“校验中”记录保留为时序证据，不把量化子进程退出提前当成功。

22:38:42 CST，统一 evaluator 发布 `YQ-Q8-W3` quality：Exact-KL=`0.1307958364`、
WikiText2 PPL=`10.4991980`、十项 QA Avg=`65.18`。其正式 campaign 摊销成本为
`4.081735 GPU·h`（量化主体 `3.893331 GPU·h`，共享 A16 Hessian 单独成本
`0.565213 GPU·h`），随后继续生成三项推理。统一评测更新为 **15/55 quality、
4/55 suite、4/55 official terminal，formal failure=0**。

23:02:13 CST，`YQ-Q8-W4` 完成三项推理 generation；23:03:12 CST，独立官方 scorer
发布 HumanEval+ terminal。完整结果为 GSM8K=`87.6422%`、MATH-500=`57.40%`、
HumanEval base/plus=`54.2683%/51.8293%`，与其 Exact-KL=`0.0322305337`、
PPL=`9.8533049`、十项 QA Avg=`67.46` 同向正常。正式 campaign 摊销成本为
`4.320333 GPU·h`；统一评测由此达到 **15/55 quality、6/55 suite、6/55 official
terminal，formal failure=0**。

同一时段 `TB20-Q8-W4` 也完成全部门禁：GSM8K=`86.2017%`、MATH-500=`56.60%`、
HumanEval base/plus=`43.9024%/42.0732%`；对应 Exact-KL=`0.0443699211`、
PPL=`10.0744648`、十项 QA Avg=`67.75`，量化成本=`0.652894 GPU·h`。suite/official
receipt 分别于 23:02:45/23:04:37 CST 发布。连同 `EQ15-Q32-W3` 的同期闭合，统一
评测实际更新为 **15/55 quality、8/55 suite、8/55 official terminal，formal
failure=0**；23:04 前一次 interim builder 的 7 组计数只是 scorer 落盘竞态，不是缺失。

### 2026-08-21 23:04--23:25 CST 新结果与 TurboBOA Llama 契约恢复

- 新发布三份 quality：`TB20-Q8-W2` 为 Exact-KL=`5.0765132904`、WikiText2
  PPL=`1013.2111206`、十项 QA Avg=`30.54`，量化成本=`0.765573 GPU·h`；
  `YQ-L8-W4A4` 为 Exact-KL=`3.6233930588`、PPL=`267.5844421`、QA Avg=`33.21`，
  campaign 摊销成本=`4.383406 GPU·h`；`EQ15-L8-W4` 为
  Exact-KL=`0.1389802992`、PPL=`6.8016219`、QA Avg=`66.03`。前两组的三个质量指标
  同向退化，按既定规则原样保留并继续三项推理，不做结果驱动 fallback。统一评测达到
  **19/55 quality、8/55 suite、8/55 official terminal**。
- 巡检发现 `TB20-L8-W4A4` 的前三次 evaluator attempt 均在模型推理前被同一严格门禁
  拒绝：checkpoint 真值为 `w_method=turboboa`，冻结 loader 却统一要求
  `turboboa_rmsnorm_mean_jacobian_kfac`。代码审计确认这不是量化配置偏差：TurboBOA
  只对具有 Q/K RMSNorm 的 Qwen3 使用后一个方法名；Llama 没有该 pullback，`main.py`
  按设计记录前一个方法名。Llama W4/W3/W2/W4A4 的正式命令仍全部固定同一
  RealQ-MSE 标量量化器、W group128、W/A/K/V 对称量化、rotate seed=0、calibration
  seed=1 与完全相同的 WikiText2 token 制品。
- 为避免改动正在运行且由 reference fingerprint 冻结的 v1 evaluator，新增独立、
  fail-closed 的 **manifest-alias-only recovery**：只把 Llama 的期望方法名更正为
  `turboboa`，模型重构、runtime manifest、KL/PPL、十项 QA、三项推理和官方 scorer
  全部复用冻结流程。恢复记录会绑定原失败日志 SHA、量化 result SHA、原 loader/recovery
  源码 SHA，以及 checkpoint 前后 inode/size/mtime/ctime；checkpoint 不修改。相关契约
  测试连同原 evaluator/TurboBOA 测试共 `17 passed`。
- 四组 Llama TurboBOA suite 已由 node0/GPU3 恢复队列原子预约；队列 PID=`105425`，
  当前等待该卡既有 `EQ15-L8-W4` suite 释放共享物理锁，随后串行补完四组。预约使其他
  worker 不会继续消耗无意义的旧 loader 重试预算，也不影响 node1/GPU1 上的 REAL-Q
  debug 或任一量化 lane。
- 同轮 REAL-Q Qwen3-8B W4A4 all-off 到 `31/36`，约 `253 s/layer`。YAQA Qwen3-32B
  A4/A16 Hessian 的 CPU-only 严格校验分别于 23:26:40/44 CST 完成：每份均覆盖
  896 个张量、64 层，随后在 23:26:42/46 发布 stage terminal；成本分别为
  `3.807910/3.859356 GPU·h`。YAQA 因而达到 **24/30 stage terminal、0 failure**，
  node0 GPU0/1 在同一分钟自动启动 `YQ-Q32-W4` 与 `YQ-Q32-W4A4`；此前无 CUDA
  context 的窗口约两分钟，不是漏调度。两机仍为 `oom/oom_kill=0/0`。

23:30:20 CST，`TB20-Q32-W2` 完成三项推理 generation；23:31:29 CST，官方
EvalPlus scorer 一次成功并发布 terminal。完整结果为 Exact-KL=`0.9873412848`、
WikiText2 PPL=`15.8305025`、十项 QA Avg=`55.46`、GSM8K=
`179/1319=13.5709%`、MATH-500=`23/500=4.60%`、HumanEval+ base/plus 均为
`1/164=0.6098%`，正式量化成本=`3.001077 GPU·h`。三项指标覆盖和官方样本数均通过
冻结 gate，统一评测更新为 **19/55 quality、9/55 suite、9/55 official terminal**。

23:30:49 CST，`TB20-Q32-W3` 完成三项推理 generation；23:32:11 CST，官方
EvalPlus scorer 一次成功并发布 terminal。该组与前述质量异常一致：GSM8K=
`15/1319=1.1372%`、MATH-500=`18/500=3.60%`、HumanEval+ base/plus 均为
`0/164=0%`；对应 Exact-KL=`4.2886948586`、WikiText2 PPL=`385.1393433`、
十项 QA Avg=`34.36`，正式量化成本=`2.905516 GPU·h`。原始异常值按冻结协议保留，
不触发结果驱动的 checkpoint 替换；统一评测达到 **19/55 quality、10/55 suite、
10/55 official terminal**。

23:35:50 CST，统一 evaluator 发布 `YQ-L8-W4` quality：Exact-KL=
`0.0283492338`、WikiText2 PPL=`7.4232306`、十项 QA Avg=`66.37`。正式 campaign
摊销成本=`4.384791 GPU·h`（量化主体 `4.219465 GPU·h`），随后继续同一冻结协议的
三项推理。统一评测因此更新为 **20/55 quality、10/55 suite、10/55 official
terminal**。

00:45:15 CST，`YQ-Q8-W2` 发布质量异常值：Exact-KL=`16.7574424744`、
WikiText2 PPL=`70114320.0`、十项 QA Avg=`31.73`，campaign 摊销成本=
`3.997343 GPU·h`。stage receipt 绑定 W2/G128 `realq_wclip`、A/K/V 16-bit、固定
Qwen3-8B 256×2048 token SHA `a7acbd90...d3b6f` 与共享 `YH-Q8-A16` Hessian SHA
`3e8bce07...e845`；validation 覆盖 36 层/252 个 projection，整数重构和 reload
确定性均为 true。该点无 fallback、NaN 或身份替换，按正式 2-bit 算法异常原样保留并
继续三项推理。其量化 worker 释放后，同卡评测 worker 已领取该 checkpoint；统一评测
更新为 **26/55 quality、16/55 suite、16/55 official terminal**，YAQA stage terminal
为 **25/30、0 failure**。

23:47:55 CST，REAL-Q 三组全优化开关诊断发布最后一个 pair terminal，原先专用的
node1/GPU1 随即自动归还 YAQA 队列并领取 `YQ-Q32-W3`；从此三种新算法重新占用全部
16 条物理 lane。23:52 CST，node0 上的 `YQ-Q32-W4/W4A4` 已同步从第 0 层进入第 1 层，
首层约 22 分钟；两份日志持续增长且无 traceback。`YQ-Q32-W3` 此时仍在 checkpoint
shard I/O，累计 `rchar` 从 79 GB 增至 238 GB、CPU time 持续增加，GPU 尚未建立
context。node1 cgroup 约 1 TB 内存几乎全部为可回收 file cache（anon 约 20 GB、
file 约 1.05 TB），`memory.events max` 因回收压力增长，但 `oom/oom_kill=0/0`；因此
保留正在推进的加载，不把它误判为量化挂起，也不为追求瞬时利用率修改冻结配置。
该加载最终累计读取约 293 GB，并于 23:58:45 CST 正常发布 `loaded model`、进入
`layer 0 gpu 0`；GPU context 随即建立，`oom/oom_kill` 仍为 0。由此确认此前约
11 分钟是受 file-cache 回收影响的有效 shard I/O，而非死锁或失败重试。

2026-08-22 00:13:24 CST，`TB20-Q8-W3` 完成三项推理 generation；00:14:07 CST，
官方 EvalPlus attempt001 发布 terminal。完整结果为 Exact-KL=`0.1632449776`、
WikiText2 PPL=`10.8699217`、十项 QA Avg=`64.78`、GSM8K=
`1034/1319=78.3927%`、MATH-500=`241/500=48.20%`、HumanEval+ base/plus=
`34/164=20.7317%` / `32/164=19.5122%`，正式量化成本=`0.781787 GPU·h`。
suite/official 样本覆盖和冻结 SHA 门禁均通过，统一评测更新为 **21/55 quality、
11/55 suite、11/55 official terminal**。原 node1/GPU7 worker 同一分钟自动领取
`YQ-L8-W3`；CUDA PID 与约 32 GB 显存上下文确认该 lane 已继续运行，并非空卡。

00:19:10 CST，`YQ-L8-W3` 发布质量异常值：Exact-KL=`2.6301767826`、
WikiText2 PPL=`100.7630768`、十项 QA Avg=`50.23`，campaign 摊销成本=
`3.986865 GPU·h`。严格身份复核确认 evaluator 绑定 `YQ-L8-W3` 的 stage receipt SHA
`5f408c66...d0f4`、validation SHA `25f88d3e...fc24` 和共享 Llama teacher SHA
`91a2a553...fd6`；正式命令为 W3/G128、`realq_wclip`、256×2048 WikiText2 token SHA
`8030125e...95e`、seed/global seed=`0/1`、rotation seed=`0`，并复用已验证的 A16
Hessian。validation 覆盖 32 层/224 个 projection，整数重构和双次 reload 确定性均为
true，runtime 为 dense fake quant、A/K/V 16-bit，无 fallback。因此该点按 **YAQA
algorithmic anomaly** 原样保留并继续三项推理；统一评测更新为 **22/55 quality、
11/55 suite、11/55 official terminal**。

2026-08-22 00:28:45 CST，`YQ-Q8-W3` 完成三项 generation；00:30:06 CST，
官方 EvalPlus attempt001 发布 terminal。完整结果为 Exact-KL=`0.1307958364`、
WikiText2 PPL=`10.4991980`、十项 QA Avg=`65.18`、GSM8K=
`1056/1319=80.0607%`、MATH-500=`265/500=53.00%`、HumanEval+ base/plus=
`39/164=23.7805%` / `36/164=21.9512%`，campaign 摊销成本=`4.081735 GPU·h`。
冻结覆盖和官方 scorer 门禁均通过；原 GPU6 worker 随即领取 `TB20-Q4-W4A4`。
连同同期闭合的 `EQ15-L8-W4`，统一评测更新为 **22/55 quality、13/55 suite、
13/55 official terminal**。

00:38:44 CST，`TB20-Q4-W4A4` 发布统一 quality：Exact-KL=`0.5950802565`、
WikiText2 PPL=`20.9556637`、十项 QA Avg=`51.66`，正式量化成本=
`0.403980 GPU·h`。三项质量指标同向退化，但 checkpoint terminal、共享 teacher、
W4/G128/symmetric RealQ-MSE 与 A/K/V 4-bit symmetric per-token runtime 门禁均成功；
原值继续三项推理，不触发 fallback。统一评测更新为 **24/55 quality、14/55 suite、
14/55 official terminal**。

2026-08-22 01:02:50/01:03:53 CST，`TB20-Q32-W4` 与 `TB20-Q8-W2` 分别完成
generation suite；官方 EvalPlus terminal 于 01:03:38/01:04:52 CST 发布。
`TB20-Q32-W4` 的完整结果为 Exact-KL=`1.8178315163`、PPL=`35.0668221`、
QA Avg=`43.17`、GSM8K=`16/1319=1.2130%`、MATH-500=`11/500=2.20%`、
HumanEval+ base/plus 均为 `0/164=0%`，正式量化成本=`2.828414 GPU·h`。

`TB20-Q8-W2` 为 Exact-KL=`5.0765132904`、PPL=`1013.2111206`、QA Avg=`30.54`、
GSM8K=`12/1319=0.9098%`、MATH-500=`21/500=4.20%`、HumanEval+ base/plus
均为 `0/164=0%`，量化成本=`0.765573 GPU·h`。两组推理退化均与其质量指标一致，
完整覆盖与冻结门禁通过；释放的 GPU 随即分别领取 `YQ-Q4-W4` 与 `EQ15-Q4-W3`，
统一评测更新为 **28/55 quality、18/55 suite、18/55 official terminal**。

00:51:20 CST，`TB20-Q4-W4` 发布统一 quality：Exact-KL=`0.0625278130`、
WikiText2 PPL=`14.2186518`、十项 QA Avg=`62.56`，正式量化成本=
`0.427313507 GPU·h`。它与同期 `EQ15-Q4-W4` 的 KL/PPL/QA 排序并不完全一致，表明
不能只凭单一质量指标筛选 checkpoint；两组均按冻结协议继续完整三项推理。统一评测
更新为 **28/55 quality、16/55 suite、16/55 official terminal**。

00:41:57 CST，`YQ-Q4-W4A4` 发布质量异常值：Exact-KL=`8.0512313843`、
WikiText2 PPL=`17795.234375`、十项 QA Avg=`30.23`，campaign 摊销成本=
`2.619174 GPU·h`。严格复核确认正式命令为 W4/G128 `realq_wclip`，使用 Qwen3-4B
固定 256×2048 token SHA `21210e19...a399`、独立 activation-aware `YH-Q4-A4`
Hessian SHA `5bc3d5ad...c7ee`，部署 A/K/V 均为 4-bit symmetric per-token、clip=0.9。
validation 覆盖 36 层/252 个 projection，整数重构与 reload 确定性均为 true；无 NaN、
fallback 或身份替换。该异常与 Q8/L8 W4A4 的退化形态一致，按 YAQA 正式算法结果原样
保留并继续三项推理。统一评测更新为 **25/55 quality、14/55 suite、14/55 official
terminal**。

2026-08-22 00:33:00 CST，`YQ-L8-W4` 完成三项 generation；00:33:50 CST，
官方 EvalPlus attempt001 发布 terminal。完整结果为 Exact-KL=`0.0283492338`、
WikiText2 PPL=`7.4232306`、十项 QA Avg=`66.37`、GSM8K=
`1094/1319=82.9416%`、MATH-500=`183/500=36.60%`、HumanEval+ base/plus=
`96/164=58.5366%` / `85/164=51.8293%`，campaign 摊销成本=`4.384791 GPU·h`。
所有冻结 gate 通过，原 GPU5 worker 随即领取 `YQ-Q4-W4A4`；统一评测更新为
**22/55 quality、14/55 suite、14/55 official terminal**。

01:08:40 CST，`YQ-Q4-W4` 发布统一 quality：Exact-KL=`0.0417815074`、
WikiText2 PPL=`13.9952888`、十项 QA Avg=`63.63`，campaign 摊销成本=
`2.378204 GPU·h`。stage receipt、共享 teacher、W4/G128 `realq_wclip` 与冻结 token/
Hessian 身份门禁均通过，随后继续三项推理；统一评测更新为 **29/55 quality、18/55
suite、18/55 official terminal**。

01:36:42/01:37:44 CST，`YQ-L8-W3` 与 `YQ-L8-W4A4` 分别发布三项推理 generation；
官方 EvalPlus terminal 于 01:37:44/01:38:27 CST 发布。两组完整结果为：

- `YQ-L8-W3`：Exact-KL=`2.6301767826`、PPL=`100.7630768`、QA Avg=`50.23`，
  GSM8K=`11/1319=0.8340%`、MATH-500=`12/500=2.40%`、HumanEval+ base/plus
  均为 `0/164=0%`，campaign 摊销成本=`3.986865 GPU·h`；
- `YQ-L8-W4A4`：Exact-KL=`3.6233930588`、PPL=`267.5844421`、QA Avg=`33.21`，
  GSM8K=`8/1319=0.6065%`、MATH-500=`2/500=0.40%`、HumanEval+ base/plus
  均为 `0/164=0%`，campaign 摊销成本=`4.383406 GPU·h`。

两组推理退化与前述 quality 异常同向，完整样本覆盖、冻结 artifact 和官方 scorer 门禁
均通过，继续按 YAQA 原始算法结果保留。统一评测达到 **30/55 quality、20/55 suite、
20/55 official terminal**。`YQ-L8-W4A4` 退出并确认 CUDA PID 清空后，node0/GPU4
短时交接给同-cache REAL-Q 快路径归因；该诊断完成后 supervisor 会自动恢复统一
evaluator，其余 15 条物理 lane 持续运行新算法任务。

01:43:02 CST，`TB20-Q4-W3` 发布统一 quality：Exact-KL=`0.2832130194`、
WikiText2 PPL=`16.5606079`、十项 QA Avg=`60.76`，正式量化成本=
`0.494840 GPU·h`。evaluator 绑定冻结的 TurboBOA Qwen3-4B W3/G128、对称
RealQ-MSE 标量量化器与同一 teacher/reference artifact，随后继续三项推理；统一评测
更新为 **31/55 quality、20/55 suite、20/55 official terminal**。

01:56:18 CST，`YQ-L8-W2` 通过 32 层/224 个 projection 的整数重构、双次 reload
确定性和 smoke-logit validation，YAQA 达到 **26/30 stage terminal、0 failure**。
量化主体耗时 `13631.168 s = 3.786435 GPU·h`，加上共享 A16 Hessian 三组摊销后的
campaign 成本为 `3.951761 GPU·h`，单独完整成本为 `4.282412 GPU·h`。该 lane 在
CPU-only hfize/validation 结束后约 30 秒内由统一 worker 领取 `YQ-L8-W2` 做质量与推理
评测，node1/GPU4 恢复计算占用；未把校验阶段的低显存窗口误报为空卡。

02:00:28 CST，node0/GPU3 在 `EQ15-L8-W2` suite 与 CUDA PID 均确认结束后安全交接给
四组 TurboBOA Llama recovery。首次 recovery 在模型加载前的 runtime metadata gate
立即退出：最初的后台启动命令把 `PYTHONPATH` 覆盖为 `.`，丢失了冻结 evaluator 使用的
`_runtime_dependencies/evalplus_only_0.3.1`，因此 `importlib.metadata.version`
找不到 EvalPlus。该尝试没有加载 checkpoint、没有 CUDA context，也没有发布 quality/
suite；四个显式 reservation claim 保留，未退回错误的旧 loader。

恢复队列随后改为显式加入并 preflight `evalplus==0.3.1` 的同一 audited runtime，且只
允许同 hostname/GPU、同 recovery kind、旧 PID 已不存在的 claim 原位续跑；两个新增
环境/claim 恢复测试 `2 passed`。02:04:06 CST retry1 重新取得 GPU3 并启动
`TB20-L8-W4A4`，约 20 秒后已进入 32 层模型 forward、GPU 利用率 `94%`，没有再次出现
metadata 或 loader gate 错误。第一次失败保留在独立日志中，作为基础设施 preflight
记录，不计作算法正式评测 failure。

01:56:04 CST，`YQ-Q8-W4A4` 发布三项 generation；01:57:15 CST，官方 EvalPlus
attempt001 发布 terminal。完整结果为 Exact-KL=`8.0090198517`、PPL=`16333.8125`、
QA Avg=`30.04`，GSM8K=`21/1319=1.5921%`、MATH-500=`21/500=4.20%`、
HumanEval+ base/plus 均为 `0/164=0%`，campaign 摊销成本=`4.698317 GPU·h`。
三项推理退化与该 W4A4 quality 异常同向，完整覆盖和官方 gate 通过，原值保留。

02:01:47/02:02:04 CST 又发布两份 YAQA quality：

- `YQ-L8-W2`：Exact-KL=`11.4592304230`、PPL=`685005.0625`、QA Avg=`32.97`，
  campaign 摊销成本=`3.951761 GPU·h`；
- `YQ-Q4-W3`：Exact-KL=`0.1520937383`、PPL=`14.6878729`、QA Avg=`59.83`，
  campaign 摊销成本=`2.264527 GPU·h`。

前者三个质量指标同向显示 W2 严重退化，后者处于可用区间；两者均绑定各自冻结
stage receipt、同模型 teacher 与正确 token/Hessian identity，继续三项推理且不做
结果驱动 fallback。连同同期闭合的 `EQ15-L8-W2`，统一评测更新为 **33/55 quality、
22/55 suite、22/55 official terminal**。

02:12:08 CST，修正 runtime preflight 后的 TurboBOA Llama contract recovery 首组
`TB20-L8-W4A4` 成功发布统一 quality：Exact-KL=`0.2138682604`、WikiText2
PPL=`8.8707447`、十项 QA Avg=`61.97`，正式量化成本=`0.655738 GPU·h`。该结果
绑定原始未改写 checkpoint、Llama 专属 `w_method=turboboa` 真值、W4/G128 对称
RealQ-MSE 标量量化器、A/K/V 4-bit 对称 per-token runtime，以及冻结的 Llama teacher
cache；说明 manifest-alias-only 恢复已穿过 loader、KL/PPL 和十项 QA 全部严格门禁，
没有再次出现 EvalPlus metadata 或 Qwen-only 方法名错误。该组已继续 GSM8K，恢复队列
会在其三项推理 terminal 后依次补齐 Llama W4/W3/W2。统一评测更新为 **34/55 quality、
22/55 suite、22/55 official terminal**。

02:16:40 CST 双机人工巡检确认 16/16 卡均有实际 CUDA 计算进程，两个 cgroup 的
`oom/oom_kill` 仍均为 `0/0`。资源分配为 1 卡执行 REAL-Q 同-SDPA/cache 快路径隔离、
3 卡执行 YAQA Qwen3-32B 剩余量化，其余 12 卡执行三种新算法的统一评测与 TurboBOA
Llama 恢复；REAL-Q 此时到 `22/36` 层，未挤停任何已有新算法子任务。

02:24:40 CST，`YQ-Q8-W2` 发布三项推理 generation；02:25:48 CST，官方
EvalPlus attempt001 发布 terminal。完整结果为前述 Exact-KL=`16.7574424744`、
PPL=`70114320.0`、QA Avg=`31.73`，GSM8K=`0/1319=0%`、MATH-500=
`0/500=0%`、HumanEval+ base/plus 均为 `0/164=0%`，campaign 摊销成本=
`3.997343 GPU·h`。三个推理任务的完全退化与其 KL/PPL 异常同向，样本覆盖和冻结
artifact gate 均通过，原值按算法正式结果保留。统一评测更新为 **34/55 quality、
23/55 suite、23/55 official terminal**；释放的评测 lane 已立即领取下一份待测
checkpoint。

02:33:33 CST，`TB20-Q4-W2` 发布统一 quality：Exact-KL=`1.8425840139`、
WikiText2 PPL=`65.9657745`、十项 QA Avg=`37.02`，正式量化成本=
`1.205710 GPU·h`。checkpoint terminal、Qwen3-4B teacher、W2/G128 对称 RealQ-MSE
标量量化器和 A/K/V 16-bit runtime 身份均通过冻结门禁；KL/PPL/QA 同向退化，原始
结果继续三项推理且不做 fallback。连同同期闭合的 EfficientQAT Q4-W4 suite 和
Q4-W2 quality，统一评测达到 **36/55 quality、24/55 suite、24/55 official
terminal**。

02:36:36 CST，`TB20-Q4-W4` 发布三项 generation；02:37:48 CST，官方
EvalPlus attempt001 发布 terminal。完整结果为 Exact-KL=`0.0625278130`、PPL=
`14.2186518`、QA Avg=`62.56`，GSM8K=`1071/1319=81.1979%`、MATH-500=
`278/500=55.60%`、HumanEval+ base=`55/164=33.5366%`、plus=
`52/164=31.7073%`，正式量化成本=`0.427314 GPU·h`。三项覆盖、checkpoint/teacher
SHA 和官方 scorer 门禁全部通过；统一评测更新为 **36/55 quality、25/55 suite、
25/55 official terminal**。

02:41:07 CST，REAL-Q 同-cache 诊断成功结束并把 node0/GPU4 交还统一 evaluator；首次
恢复进程随后因继承环境漏掉 audited `evalplus==0.3.1` metadata，在模型加载前让 10 个
Qwen3-0.6B 待测 suite 各产生三次同构 `PackageNotFoundError`。这 30 次均无 CUDA
context、quality/suite 产物或 checkpoint 改写，属于 handoff runtime 基础设施失败，
不计作算法评测结果。恢复脚本逐份校验 eval-id、node0/GPU4、返回码、错误文本和三个
文件 SHA 后，于 02:46:43 CST 将原 attempts 完整移动到独立 superseded 审计树并恢复
retry budget；supervisor 同时改为显式加入 `_runtime_dependencies/evalplus_only_0.3.1`
并在 exec 前强制 preflight 版本。02:46:57 CST 重启 worker 已使首组
`TB20-Q06-W4A4` 穿过该 gate、进入模型 forward 和 CUDA 计算，GPU4 恢复服务。

02:42:17 CST，`YQ-Q4-W2` 发布统一 quality：Exact-KL=`11.4570884705`、
WikiText2 PPL=`358920.1875`、十项 QA Avg=`31.15`，campaign 摊销成本=
`2.279174 GPU·h`。三个质量指标同向显示 2-bit 严重退化，checkpoint/teacher/token/
Hessian 身份均通过门禁后继续三项推理，不做 fallback。

02:43:08 CST，`YQ-Q4-W4` 发布三项 generation；02:44:02 CST，官方 EvalPlus
attempt001 发布 terminal。完整结果为 Exact-KL=`0.0417815074`、PPL=`13.9952888`、
QA Avg=`63.63`，GSM8K=`1128/1319=85.5193%`、MATH-500=`287/500=57.40%`、
HumanEval+ base=`72/164=43.9024%`、plus=`69/164=42.0732%`，campaign 摊销成本=
`2.378204 GPU·h`。所有冻结 gate 与样本覆盖通过；连同同期 EfficientQAT 的结果，统一
评测更新为 **38/55 quality、27/55 suite、27/55 official terminal**。

02:55:05 CST，修复 post-REALQ handoff runtime 后的首组 `TB20-Q06-W4A4` 已成功发布
统一 quality：Exact-KL=`4.5349602699`、WikiText2 PPL=`1294.5850830`、十项 QA
Avg=`30.93`。quality receipt 绑定的原始 contended 计时为 `0.486136 GPU·h`；公平
release 使用 checkpoint content-identical clean-retime 的 **`0.156067 GPU·h`**，原始
计时显式排除。该组已继续 GSM8K，证明恢复后的 GPU4 不仅穿过 metadata/model loader，
也完成了完整 KL/PPL 与十项 QA；统一 quality 更新为 **39/55**，suite/official 仍为
**27/55**。

02:56:57 CST，恢复后的第二条 lane 也成功发布 `YQ-Q06-W4A4` quality：Exact-KL=
`9.7596817017`、WikiText2 PPL=`246740.84375`、十项 QA Avg=`30.48`，campaign
摊销成本=`0.570312 GPU·h`。stage receipt、A4 Hessian、256×2048 固定 token、W4/G128
`realq_wclip` 与 A/K/V 4-bit 对称 per-token runtime 均通过门禁；该异常值继续完整推理，
不做 fallback。统一 quality 更新为 **40/55**，suite/official 仍为 **27/55**。

02:58 CST 对现有 27 个完整 suite 运行 fail-closed interim release builder，全部 artifact
SHA、teacher/reference fingerprint、PPL/KL 口径、十项 QA task 集、三项推理样本覆盖、
官方 scorer 以及 clean-retime accounting 均通过。当前可发布行按方法为 EfficientQAT
`11`、TurboBOA `8`、YAQA_wclip `8`；对应已完成行的公平 GPU-hour 合计为
`3.487732/12.072152/32.230994`。制品为
`interim_release_20260822T0258CST.json`，状态按预期为 `incomplete`、missing=`28`，
不冒充最终 55 组 release。

03:19:26 CST，`YQ-L8-W2` 完成三项推理 generation；03:20:37 CST，官方 EvalPlus
attempt001 发布 terminal。完整结果为前述 Exact-KL=`11.4592304230`、PPL=
`685005.0625`、十项 QA Avg=`32.97`，GSM8K=`0/1319=0%`、MATH-500=
`0/500=0%`、HumanEval+ base/plus 均为 `0/164=0%`，campaign 摊销成本=
`3.951761 GPU·h`。三项样本覆盖、checkpoint/teacher SHA 和官方 scorer 门禁全部
通过；推理完全失效与该 W2 checkpoint 的质量异常同向，原值保留。统一评测更新为
**40/55 quality、28/55 suite、28/55 official terminal**。GPU4 在 generation
terminal 后自动领取 `EQ15-Q06-W4`，约 75 秒后已进入实际 CUDA 计算，没有形成空卡。

03:29:06 CST，`TB20-Q4-W3` 完成三项推理 generation；03:29:59 CST，官方 EvalPlus
attempt001 发布 terminal。完整结果为前述 Exact-KL=`0.2832130194`、PPL=
`16.5606079`、十项 QA Avg=`60.76`，GSM8K=`1056/1319=80.0607%`、MATH-500=
`241/500=48.20%`、HumanEval+ base=`33/164=20.1220%`、plus=
`31/164=18.9024%`，正式量化成本=`0.494840 GPU·h`。完整覆盖、冻结 checkpoint/
teacher SHA 与官方 scorer 门禁均通过。统一评测更新为 **41/55 quality、29/55
suite、29/55 official terminal**；释放的 node1/GPU7 随即领取 `TB20-Q06-W4`，
两分钟内已进入 CUDA 计算。

03:34:02 CST，`TB20-Q06-W4` 发布统一 quality：Exact-KL=`0.0955235735`、
WikiText2 PPL=`22.6814442`、十项 QA Avg=`45.71`。checkpoint、Qwen3-0.6B
teacher、固定 token 与 W4/G128 对称 RealQ-MSE 标量量化器身份均通过冻结门禁；
公平 release 使用 checkpoint content-identical clean-retime 的 **`0.130902 GPU·h`**，
原始受 CPU 争用污染的 `0.449602 GPU·h` 显式排除。该组已继续三项推理，统一评测
更新为 **42/55 quality、29/55 suite、29/55 official terminal**。

03:39:30 CST，`YQ-Q4-W3` 完成三项推理 generation；03:40:34 CST，官方 EvalPlus
attempt001 发布 terminal。完整结果为前述 Exact-KL=`0.1520937383`、PPL=
`14.6878729`、十项 QA Avg=`59.83`，GSM8K=`1032/1319=78.2411%`、MATH-500=
`235/500=47.00%`、HumanEval+ base=`28/164=17.0732%`、plus=
`26/164=15.8537%`，campaign 摊销成本=`2.264527 GPU·h`。完整覆盖、冻结 stage/
teacher SHA 与官方 scorer 门禁均通过。统一评测更新为 **42/55 quality、30/55
suite、30/55 official terminal**；释放的 node0/GPU2 随即领取 `YQ-Q06-W4`，
约两分钟后已进入 CUDA 计算。

03:42 CST 对现有 30 个完整 suite 再次运行 fail-closed interim release builder，新增行
继续通过 artifact SHA、reference fingerprint、三项样本覆盖、官方 scorer 与公平计费
门禁。当前可发布行按方法为 EfficientQAT=`11`、TurboBOA=`9`、YAQA_wclip=`10`，
公平 GPU-hour 合计分别为 `3.487732/12.566993/38.447282`。制品为
`interim_release_20260822T0342CST.json`，状态按预期为 `incomplete`、missing=`25`；
本地登录节点的同名 venv 解析到 Python 3.6，首次命令在模块导入前即失败且未写产物，
随后在正式实验节点的冻结 Python 3.12 runtime 成功执行，不涉及模型或评测重算。

03:43:54 CST，`YQ-Q06-W4` 发布统一 quality：Exact-KL=`0.0720915496`、
WikiText2 PPL=`21.9715595`、十项 QA Avg=`46.45`。stage receipt、Qwen3-0.6B
teacher、固定 256×2048 token、W4/G128 `realq_wclip` 与 A/K/V 16-bit runtime 身份
均通过门禁。公平 release 使用 tensor byte-identical A16 Hessian clean-retime，按
`0.457562` 量化 GPU·h 加共享 Hessian `0.090839/3`，得到 **`0.487842 GPU·h`**；
原始受争用 campaign 口径 `0.599946 GPU·h` 排除。该组已继续三项推理，统一评测
更新为 **43/55 quality、30/55 suite、30/55 official terminal**。

04:03:00 CST，`TB20-Q32-W4A4` 完成三项推理 generation；04:04:03 CST，官方
EvalPlus attempt001 发布 terminal。完整结果为 Exact-KL=`0.4039516747`、PPL=
`9.3451729`、十项 QA Avg=`66.76`，GSM8K=`1203/1319=91.2055%`、MATH-500=
`281/500=56.20%`、HumanEval+ base=`40/164=24.3902%`、plus=
`39/164=23.7805%`，正式量化成本=`2.803941 GPU·h`。完整覆盖、冻结 checkpoint/
teacher SHA 与官方 scorer 门禁均通过。统一评测更新为 **44/55 quality、32/55
suite、32/55 official terminal**；释放的 node1/GPU6 随即领取 `TB20-Q06-W3`，
约两分钟后已进入 CUDA 计算。

04:07:52 CST，`TB20-Q06-W3` 发布统一 quality：Exact-KL=`0.4961514771`、
WikiText2 PPL=`31.1214542`、十项 QA Avg=`40.37`。checkpoint、Qwen3-0.6B
teacher、固定 token 与 W3/G128 对称 RealQ-MSE 标量量化器身份均通过冻结门禁；
公平 release 使用 checkpoint content-identical clean-retime 的 **`0.169892 GPU·h`**，
原始受 CPU 争用污染的 `0.612925 GPU·h` 显式排除。该组已继续三项推理。

04:12:38 CST，`YQ-Q06-W3` 发布统一 quality：Exact-KL=`0.3117986321`、
WikiText2 PPL=`26.0361137`、十项 QA Avg=`42.88`。stage receipt、Qwen3-0.6B
teacher、固定 256×2048 token、W3/G128 `realq_wclip` 与 A/K/V 16-bit runtime 身份
均通过门禁。公平 release 按量化主体 `0.510498 GPU·h` 加 byte-identical A16 Hessian
clean-retime `0.090839/3`，得到 **`0.540777 GPU·h`**；原始受争用 campaign 口径
`0.652881 GPU·h` 排除。该组已进入 GSM8K，统一评测更新为 **46/55 quality、
33/55 suite、33/55 official terminal**。

04:16:16 CST，`TB20-Q4-W2` 完成三项推理 generation；04:17:08 CST，官方
EvalPlus attempt001 发布 terminal。完整结果为前述 Exact-KL=`1.8425840139`、PPL=
`65.9657745`、十项 QA Avg=`37.02`，GSM8K=`0/1319=0%`、MATH-500=
`0/500=0%`、HumanEval+ base/plus 均为 `0/164=0%`。公平 release 使用 checkpoint
content-identical clean-retime 的 **`0.514205 GPU·h`**，原始受 CPU 争用污染的
`1.205710 GPU·h` 显式排除。完整覆盖、冻结 checkpoint/teacher SHA 与官方 scorer
门禁均通过；统一评测更新为 **46/55 quality、34/55 suite、34/55 official terminal**。

04:19:51 CST，`YQ-Q4-W2` 的官方 EvalPlus attempt001 发布 terminal，三项推理正式
闭合。完整结果为前述 Exact-KL=`11.4570884705`、PPL=`358920.1875`、十项 QA
Avg=`31.15`，GSM8K=`0/1319=0%`、MATH-500=`1/500=0.20%`、HumanEval+
base/plus 均为 `0/164=0%`，campaign 摊销成本=`2.279174 GPU·h`。完整样本覆盖、
冻结 stage/teacher SHA 与官方 scorer 门禁均通过；W2 严重退化按正式算法结果原样
保留。统一评测更新为 **46/55 quality、35/55 suite、35/55 official terminal**；
释放的 node1/GPU5 随即领取 `YQ-Q06-W2`，约一分钟内进入实际 CUDA 计算。

04:21:05 CST，`TB20-Q06-W2` 发布统一 quality：Exact-KL=`3.7238121033`、
WikiText2 PPL=`608.7858887`、十项 QA Avg=`30.71`。checkpoint、Qwen3-0.6B
teacher、固定 256×2048 token 与 W2/G128 对称 RealQ-MSE 标量量化器身份均通过冻结
门禁。公平 release 使用 checkpoint content-identical clean-retime 的
**`0.175807 GPU·h`**，原始受 CPU 争用污染的 `0.577745 GPU·h` 显式排除；该组已
继续三项推理且不做结果驱动 fallback。统一评测更新为 **47/55 quality、35/55
suite、35/55 official terminal**。

04:23:33 CST，`YQ-Q06-W2` 发布统一 quality：Exact-KL=`17.6170024872`、
WikiText2 PPL=`424796480.0`、十项 QA Avg=`31.89`。stage receipt、Qwen3-0.6B
teacher、固定 token、W2/G128 `realq_wclip` 与 A/K/V 16-bit runtime 身份均通过冻结
门禁。公平 release 按量化主体 `0.482035 GPU·h` 加 byte-identical A16 Hessian
clean-retime `0.090839/3`，得到 **`0.512315 GPU·h`**；原始受争用 campaign 口径
`0.624418 GPU·h` 排除。该严重退化 checkpoint 已继续三项推理，不做 fallback；统一
评测更新为 **48/55 quality、35/55 suite、35/55 official terminal**。

04:25:59 CST，node1/GPU4 的统一 worker 在成功发布 `EQ15-Q06-W4` suite 后退出；
只读日志确认退出原因是冻结 v1 worker 把正由独立契约恢复队列持有的
`TB20-L8-W4A4` 三次历史 alias failure 识别成 retry exhausted。当前恢复进程仍在
node0/GPU3 正常推进，checkpoint、quality、suite 与 official 产物均未受影响，因此这
不是算法或正式评测 failure。为保持 reference manifest 的 worker source SHA，不在活跃
实验中修改冻结 `worker.py`；等四个 TurboBOA Llama recovery terminal 依次发布后再
重启退出的 evaluator lane。GPU4 当前没有可安全领取的独立 checkpoint，直接重启只会
重复触发同一只读门禁。

04:41:07 CST，在 node1/GPU4 启动独立 CPU-only 延迟恢复 supervisor（PID 212996）。
它不获取物理 GPU 锁、不修改冻结 worker，只读等待四个 TurboBOA Llama
`suite_success.json`，逐一验证 `contract_recovery` kind、状态和 SHA；四个 terminal
齐全且 audited `evalplus==0.3.1` preflight 通过后，才以同一 PID `execve` 回原冻结
evaluator。对应实现的 incomplete/all-complete/wrong-kind 门禁测试为 `2 passed`；启动
日志与 receipt 位于 `queue_logs/j-zogxxxduju-master-0/`
`gpu4.wait_turboboa_recovery_then_eval.*`。这样后续 YAQA Qwen3-32B checkpoint 发布时
至少有一条跨 lane evaluator 会自动恢复，同时不会在当前 recovery 未闭合时重复消耗
retry budget。

04:45:29 CST，`YQ-Q06-W4` 完成三项推理 generation；04:46:26 CST，官方
EvalPlus attempt001 发布 terminal。完整结果为前述 Exact-KL=`0.0720915496`、PPL=
`21.9715595`、十项 QA Avg=`46.45`，GSM8K=`787/1319=59.6664%`、MATH-500=
`166/500=33.20%`、HumanEval+ base=`33/164=20.1220%`、plus=
`30/164=18.2927%`。公平量化成本按量化主体与 byte-identical A16 Hessian clean-retime
摊销为 **`0.487842 GPU·h`**；完整覆盖、冻结 stage/teacher SHA 与官方 scorer 门禁
均通过。统一评测更新为 **48/55 quality、37/55 suite、37/55 official terminal**。

该 suite 发布后，node0/GPU2 evaluator 也按同一 retry-exhausted 门禁退出且释放物理
GPU 锁。04:47:28 CST 已在该卡启动第二个同实现延迟 supervisor（PID 156579），与
node1/GPU4 的 PID 212996 分属不同 host/supervisor lock。两者均只等 recovery terminal、
当前不占 GPU；四个 TurboBOA Llama recovery 闭合后，两台机器将各自自动恢复至少一条
跨-lane evaluator，以并行承接随后发布的 YAQA Qwen3-32B checkpoint。

04:49:47 CST，`TB20-Q06-W4` 完成三项推理 generation；04:50:40 CST，官方
EvalPlus attempt001 发布 terminal。完整结果为前述 Exact-KL=`0.0955235735`、PPL=
`22.6814442`、十项 QA Avg=`45.71`，GSM8K=`803/1319=60.8795%`、MATH-500=
`179/500=35.80%`、HumanEval+ base=`37/164=22.5610%`、plus=
`35/164=21.3415%`。公平 release 使用 checkpoint content-identical clean-retime 的
**`0.130902 GPU·h`**，原始受 CPU 争用污染的 `0.449602 GPU·h` 显式排除；完整
覆盖、冻结 checkpoint/teacher SHA 与官方 scorer 门禁均通过。统一评测更新为
**48/55 quality、38/55 suite、38/55 official terminal**。

04:51:34 CST，`TB20-Q4-W4A4` 完成三项推理 generation；04:52:23 CST，官方
EvalPlus attempt001 发布 terminal。完整结果为 Exact-KL=`0.5950802565`、PPL=
`20.9556637`、十项 QA Avg=`51.66`，GSM8K=`500/1319=37.9075%`、MATH-500=
`99/500=19.80%`、HumanEval+ base/plus 均为 `6/164=3.6585%`，正式量化成本=
`0.403980 GPU·h`。完整覆盖、冻结 checkpoint/teacher SHA 与官方 scorer 门禁均通过；
W4A4 的质量与推理退化按算法正式结果原样保留。统一评测更新为 **48/55 quality、
40/55 suite、39/55 official terminal**；同期 `YQ-Q4-W4A4` generation 已发布，等待
同一官方 scorer 串行闭合。

04:53:06 CST，`YQ-Q4-W4A4` 的官方 EvalPlus attempt001 发布 terminal。完整结果为
Exact-KL=`8.0512313843`、PPL=`17795.234375`、十项 QA Avg=`30.23`，GSM8K=
`11/1319=0.8340%`、MATH-500=`14/500=2.80%`、HumanEval+ base/plus 均为
`0/164=0%`，campaign 摊销成本=`2.619174 GPU·h`。三项推理的严重退化与 KL/PPL/QA
同向；完整覆盖、冻结 stage/teacher SHA 与官方 scorer 门禁均通过，不做结果驱动
fallback。统一评测更新为 **48/55 quality、40/55 suite、40/55 official terminal**。

04:53 CST 对当前 40 个完整 suite 生成第三份 fail-closed interim release。全部 artifact
SHA、reference fingerprint、KL/PPL 口径、十项 QA task 集、三项样本覆盖、官方 scorer
与 clean-retime accounting 门禁通过；按方法为 EfficientQAT=`14`、TurboBOA=`13`、
YAQA_wclip=`13`，公平 GPU-hour 合计分别为 `3.729953/16.420021/43.833472`。制品为
`interim_release_20260822T0453CST.json`，状态明确为 `incomplete`、missing=`15`，仅作
过程审计，不用于渲染最终 20 设定总表。

04:52:37 CST，`EQ15-Q06-W3` 完成三项推理 generation；04:53:50 CST，官方
EvalPlus terminal 发布。完整结果为 Exact-KL=`0.5043333173`、PPL=
`26.7690525`、十项 QA Avg=`39.52`，GSM8K=`19/1319=1.4405%`、MATH-500=
`23/500=4.60%`、HumanEval+ base/plus 均为 `0/164=0%`。公平 release 使用
checkpoint byte-identical clean-retime 的 **`0.050731 GPU·h`**，原始受争用计时
`1.615332 GPU·h` 排除；量化与完整评测契约均通过。至此 EfficientQAT 要求的 15 个
纯权重量化设定已全部闭合。

04:57:47 CST，独立契约恢复完成 `TB20-L8-W4A4` generation；04:58:34 CST，官方
EvalPlus terminal 发布。完整结果为 Exact-KL=`0.2138682604`、PPL=`8.8707447`、
十项 QA Avg=`61.97`，GSM8K=`905/1319=68.6126%`、MATH-500=
`113/500=22.60%`、HumanEval+ base=`72/164=43.9024%`、plus=
`59/164=35.9756%`，正式量化成本=`0.655738 GPU·h`。恢复只修正冻结 evaluator 对
Llama checkpoint 的 TurboBOA manifest alias 预期；checkpoint inode/size/mtime、
数值 loader、样本与 scorer 均未改变，并由 `contract_recovery` receipt 和 SHA 门禁
记录。统一评测因此更新为 **48/55 quality、42/55 suite、42/55 official terminal**。

05:01 CST，在确认上述 W4A4 terminal 和 claim owner 后，先暂停的旧串行恢复父进程
已被终止并释放 GPU3 物理锁；其 W4A4 claim 原子移动到只读审计目录。新增的
`split_turboboa_llama_recovery.py` 仅将原队列剩余的 W4/W3/W2 claim 分别转交给
node0/GPU3、GPU5、GPU6，继续调用同一 `recover_turboboa_llama` 数值实现与冻结
runtime。owner/死 PID/suite recovery kind/物理 GPU 锁均 fail-closed，相关恢复与
supervisor 测试合计 **13 passed**，`git diff --check` 通过；三路在 05:03 CST 均已
进入模型 forward。另在 node1/GPU2、GPU7 增加 CPU-only 延迟 supervisor，与既有
node0/GPU2、node1/GPU4 一样，在四个 Llama terminal 齐全前不占 GPU，齐全后才
`execve` 原冻结 worker。

05:04 CST 对 42 个完整 suite 再次运行 fail-closed interim release builder，全部
artifact SHA、reference fingerprint、KL/PPL/QA、三项样本覆盖、官方 scorer 与公平
计费门禁通过。当前按方法为 EfficientQAT=`15`、TurboBOA=`14`、YAQA_wclip=`13`，
公平 GPU-hour 合计分别为 `3.780685/17.075759/43.833472`；制品为
`interim_release_20260822T0504CST.json`，状态为 `incomplete`、missing=`13`。

05:08 CST，并行拆分的三个 TurboBOA Llama A16 suite 均发布统一 quality：
`TB20-L8-W4` 为 Exact-KL=`0.0259965435`、PPL=`7.3992901`、十项 QA Avg=
`66.66`、量化成本=`0.950861 GPU·h`；`TB20-L8-W3` 为 Exact-KL=
`0.1094856113`、PPL=`8.0495634`、QA Avg=`64.08`、成本=
`0.725873 GPU·h`；`TB20-L8-W2` 为 Exact-KL=`0.9360691905`、PPL=
`18.1531010`、QA Avg=`39.96`、成本=`0.729881 GPU·h`。三者使用同一 Llama
teacher SHA、同一固定 WikiText2/QA 协议与各自原 checkpoint terminal，quality
从 48/55 增至 **51/55**；三路已继续 GSM8K/MATH-500/HumanEval+，不根据 W2
退化做 fallback。

05:05--05:10 CST 的 16 卡人工巡检中，12 张卡有实际 CUDA 负载，另外 4 张挂有
CPU-only 延迟 evaluator。没有强行把 `YQ-Q32-W2` 从 node1/GPU1 的冻结 W3→W2
串行队列提前并发：两台 pod 的 cgroup `memory.current` 分别约为
`992.5/1073.7 GB` 与 `1072.4/1073.7 GB`，其中绝大部分是可回收文件页缓存，但
`memory.events.max` 已有持续 reclaim 记录；新增一个 32B reader 会引入无必要的
内存/IO 抖动和阶段目录双 owner 风险。两机 `oom=0, oom_kill=0`，现有三路 32B
量化继续正常推进。同期所有活跃量化/评测日志均未命中 traceback、CUDA OOM、
assertion 或 non-finite；拆分的 Llama 三路显存均约 50 GB，已进入完整评测。

05:17:06/05:17:57 CST，`YQ-Q06-W3` 的 generation/官方 terminal 闭合。完整结果
为前述 Exact-KL=`0.3117986321`、PPL=`26.0361137`、十项 QA Avg=`42.88`，
GSM8K=`190/1319=14.4049%`、MATH-500=`20/500=4.00%`、HumanEval+
base/plus 均为 `0/164=0%`，公平量化成本=`0.540777 GPU·h`。05:21:57/
05:22:40 CST，`TB20-Q06-W3` 也闭合：Exact-KL=`0.4961514771`、PPL=
`31.1214542`、QA Avg=`40.37`，GSM8K=`216/1319=16.3760%`、MATH-500=
`20/500=4.00%`、HumanEval+ base/plus 均为 0，公平量化成本=
`0.169892 GPU·h`。两组均通过完整覆盖、冻结 checkpoint/teacher SHA 与官方 scorer
门禁，统一 suite/official 更新为 **44/55**。

05:22 CST，为消除 node1/GPU1 冻结队列中 W3→W2 的约 19 小时串行尾部，先仅对
队列父进程 PID 132939 发送 SIGSTOP；其 W3 quantizer PID 132942 及 CUDA 子进程
继续运行。`relocate_q32_w2.py` 随后验证父进程 stopped 状态/命令/物理锁、W3 子进程
PPID 与命令、原 plan/source/preflight/Hessian SHA、W2 stage 不存在，并发布 handoff
SHA `fb483f09860b…`。05:22:56 CST，node0/GPU2 取得共享物理锁并以冻结 W2 stage
的原始 seed、token、Hessian、量化器和命令启动；只把实际 CUDA 卡调度到 GPU2，
最终 terminal 会同时保留原 stage 与实际 host/GPU relocation receipt。相关定向测试
为 **7 passed**。该调度把四个 YAQA Qwen3-32B 设定变为并行，不改变数值合同。

05:25 CST 的 44 行 fail-closed interim release 再次通过全部门禁；按方法为
EfficientQAT=`15`、TurboBOA=`15`、YAQA_wclip=`14`，公平 GPU-hour 合计分别为
`3.780685/17.245651/44.374250`。制品为
`interim_release_20260822T0525CST.json`，状态仍为 `incomplete`、missing=`11`。

05:26:27/05:27:25 CST，`YQ-Q06-W2` 的 generation/官方 terminal 闭合。完整结果
为前述 Exact-KL=`17.6170024872`、PPL=`424796480.0`、十项 QA Avg=`31.89`，
GSM8K、MATH-500 与 HumanEval+ base/plus 均为 **0%**；公平量化成本=
`0.512315 GPU·h`。完整样本覆盖、冻结 stage/teacher SHA 与官方 scorer 门禁均
通过；该结果与质量指标的严重退化同向，原值保留。统一 suite/official 更新为
**45/55**。

05:35:31/05:37:02 CST，`TB20-Q06-W2` 的 generation/官方 terminal 闭合。完整结果
为前述 Exact-KL=`3.7238121033`、PPL=`608.7858887`、十项 QA Avg=`30.71`，
GSM8K=`3/1319=0.2274%`、MATH-500=`3/500=0.60%`、HumanEval+ base/plus
均为 `0/164=0%`。公平 release 使用 checkpoint content-identical clean-retime 的
**`0.175807 GPU·h`**，原始受争用计时 `0.577745 GPU·h` 排除；完整覆盖、冻结
checkpoint/teacher SHA 与官方 scorer 门禁均通过。统一 suite/official 更新为
**46/55**。

05:36 CST，针对前述 W3→W2 调度拆分补上独立 W3 terminal finalizer。审计发现 raw
quantizer 成功退出成为 zombie 后 `/proc/PID/cmdline` 可能为空，且被 SIGKILL 的旧
queue parent 即使尚未被 shell 立即 reap 也已经释放文件锁；finalizer 因此以不可复用的
PID start tick、PPID、wait status、父进程 stopped 状态和物理锁作为边界门禁，不依赖
这两个不稳定的 `/proc` 表象。定向 py_compile 与三文件测试共 **12 passed**；05:36:55
CST 在原 node1 启动 CPU-only supervisor PID 218790。它当前只读等待 W3 raw child，
成功后仅执行冻结 worker 原本的 hfize/validation/receipt 步骤并终止旧父进程，绝不进入
已经迁移的 W2 stage；量化 seed、token、Hessian、量化器和 GPU 均未改变。

05:38 CST 的人工巡检中，node0 八张卡均有实际 CUDA 计算：三张 YAQA Qwen3-32B、
三张 TurboBOA Llama 推理恢复和两张 Qwen3-0.6B W4A4 推理；node1 仅原
`YQ-Q32-W3` 有 CUDA 计算，其余卡暂时没有可独立启动的正式 checkpoint。两机
`oom=0, oom_kill=0`，活跃日志未见 CUDA OOM/traceback/non-finite。对现有 46 个
完整 suite 再运行 fail-closed interim builder，所有 SHA/reference/协议/覆盖与公平计费
门禁通过；方法计数为 EfficientQAT=`15`、TurboBOA=`16`、YAQA_wclip=`15`，公平
GPU-hour 合计分别为 `3.780685/17.421458/44.886564`。制品为
`interim_release_20260822T0538CST.json`，状态为 `incomplete`、missing=`9`。

05:52 CST，为兼顾 REAL-Q 根因诊断，先验证 node1/GPU0 的原统一 evaluator PID 27852
处于 `waiting_for_ready_eval completed=46/55`、没有 child/claim/CUDA context；随后启动
同一已测试延迟恢复 supervisor PID 219526，再终止该空闲 worker 释放物理锁。supervisor
不占 GPU，等待四个 TurboBOA Llama recovery terminal 后会用审计过的 EvalPlus 0.3.1
环境 `execve` 回冻结 worker。REAL-Q 的 SDPA/FA4 sampled-label paired 诊断只使用这张
已释放的 GPU0 和本来由 CPU-only supervisor 等待的 GPU2，06:00 前均返回成功并释放
GPU；期间 node0 八张正式实验卡持续计算，YAQA Q32-W3 继续独占 node1/GPU1，正式
checkpoint/评测 claim 均未被抢占。两机 `oom=0, oom_kill=0`。

06:02 CST 巡检时四个 YAQA Qwen3-32B stage 的 attention `o_proj` 标志分别推进到
W4A4 layer 21、W4 layer 22、W3 layer 20、迁移后的 W2 layer 1；W3 旧 queue parent
仍为 stopped，raw child 与
CPU-only finalizer 均正常。两组 Q06-W4A4 已完成 GSM8K/MATH-500 并进入 HumanEval+
（TurboBOA 112/164、YAQA 48/164）；三组 TurboBOA Llama A16 已完成 GSM8K，MATH-500
分别为 352/288/272。统一计数仍为 quality=`51/55`、suite/official=`46/55`，活跃日志
无 traceback、CUDA OOM 或 non-finite。

06:13--06:29 CST，四个后续 suite 依次完成 generation 和官方 EvalPlus：

- `TB20-Q06-W4A4`：Exact-KL=`4.5349602699`、PPL=`1294.5851`、QA Avg=
  `30.93`、GSM8K=`1/1319=0.0758%`、MATH-500=`4/500=0.80%`、HumanEval+
  base/plus 均为 0，content-identical clean-retime 公平成本=`0.156067 GPU·h`；
- `TB20-L8-W4`：Exact-KL=`0.0259965435`、PPL=`7.3992901`、QA Avg=`66.66`、
  GSM8K=`82.2593%`、MATH-500=`35.20%`、HumanEval+ base/plus=
  `59.1463%/54.2683%`，最终 clean-retime 成本=`0.628342 GPU·h`。05:08 过程记录的
  `0.950861 GPU·h` 已由 checkpoint-content-identical 的无争用复跑替代并显式排除；
- `TB20-L8-W3`：Exact-KL=`0.1094856113`、PPL=`8.0495634`、QA Avg=`64.08`、
  GSM8K=`75.8150%`、MATH-500=`29.00%`、HumanEval+ base/plus=
  `39.6341%/35.9756%`，公平成本=`0.725873 GPU·h`；
- `YQ-Q06-W4A4`：Exact-KL=`9.7596817017`、PPL=`246740.84375`、QA Avg=
  `30.48`、GSM8K=`11/1319=0.8340%`、MATH-500=`6/500=1.20%`、HumanEval+
  base/plus 均为 0，campaign 摊销成本=`0.570312 GPU·h`。

06:33 CST 的 50 行 interim release 通过全部 SHA/reference/协议/覆盖/计费门禁；方法
计数为 EfficientQAT=`15`、TurboBOA=`19`、YAQA_wclip=`16`，公平 GPU-hour 合计为
`3.780685/18.931740/45.456876`，制品为
`interim_release_20260822T0633CST.json`，missing=`5`。

06:35:16/06:36:03 CST，最后一个 TurboBOA 组 `TB20-L8-W2` 完成 generation/官方
terminal：Exact-KL=`0.9360691905`、PPL=`18.1531010`、QA Avg=`39.96`，GSM8K、
MATH-500、HumanEval+ base/plus 均为 0，公平量化成本=`0.729881 GPU·h`。恢复只修正
manifest alias，checkpoint 与数值路径未改变。至此 TurboBOA **20/20 全部闭合**；
06:37 的 51 行 interim release 中 EfficientQAT/TurboBOA/YAQA=`15/20/16`，公平
GPU-hour 合计=`3.780685/19.661621/45.456876`，missing 仅四个 YAQA Qwen3-32B。

为缩短最后四个 32B suite 的串行评测尾部，冻结并测试了
`experiments/yaqa_q32_eval_fanout_20260822/`。它不改 loader、reference、seed、样本、
scorer 或数值协议，只把每个 suite 的 quality、GSM8K、MATH-500、HumanEval+ 分发到
四张物理卡；四个 component receipt 全部通过后才原子发布原 schema 的
`suite_success.json`。plan fingerprint=`3fc37b91652cbeff5d8d876ae73ae565c6ae7c8519a7105d82fe6563c2d5e056`，
相关 fanout/evaluator/recovery 全套测试共 `32 passed`，其中 fanout 自测 `7 passed`。
06:34 CST 已预留四个 suite claim 并启动 16 个 CPU-only supervisor；checkpoint 发布
前不获取 GPU 锁。06:35 CST 又逐 PID 验证旧串行 evaluator 没有 child、claim 或 CUDA
context 后，以 SIGTERM 退出 node0 的 5 个、node1 的 7 个旧 worker/supervisor，审计
receipt 位于 `yaqa_q32_evaluation_fanout_20260822/legacy_worker_retirement/`。正在运行的
四条 Q32 quantizer、W3 finalizer、W2 relocation 和新 fanout supervisor 均未触碰。

06:37 CST 人工巡检时，Q32 W4A4/W4/W3/W2 已分别写出 attention `o_proj` artifact
到 layer `22/23/20/3`，后续组件仍持续落盘；两机 `oom=0, oom_kill=0`。量化继续
优先持有 node0 GPU0/1/2 与
node1 GPU1，fanout 只会在各自 artifact 原子发布并确认目标 GPU 无未跟踪进程后启动。

06:49--06:54 CST 利用 node1/GPU0/2 的空窗并行完成 REAL-Q 固定标签 signed-gradient
trace；两臂始终取得同一套物理锁，不与 node1/GPU1 的 `YQ-Q32-W3` 或 node0 三条 Q32
量化 lane 重叠，完成后两张卡与锁均释放。06:55 CST 人工巡检时 Q32
W4A4/W4/W3/W2 已分别写出 attention `o_proj` artifact 到 `24/25/22/4` 层
（W4A4/W3/W2 均正在下一层内部，W4 刚写出 layer 24 `o_proj`）；四份日志继续增长
且无 traceback、CUDA OOM 或 non-finite，
两机仍为 `oom=0, oom_kill=0`。16 个 fanout supervisor 数量严格保持 node0/node1=
`5/11`，在 artifact 发布前均为 CPU-only。此时未占用的卡没有可提前执行的独立正式
checkpoint；四个 Q32 artifact 完成后，会按冻结 plan 立即展开 quality（含 KL/PPL/十项
QA）、GSM8K、MATH-500、HumanEval+ 四路并行评测。

07:29--07:31 CST 再次人工巡检，两条 Canoe debug job 均为 `Running`、retry=`0`；
node0/node1 的 `memory.events` 仍分别为 `oom=0, oom_kill=0`。四个 Q32 stage 的
attention `o_proj` 完成层号更新为 W4A4/W4/W3/W2=`26/26/24/6`，均正在下一层内部
持续刷新日志，近期相邻 `o_proj` 标志平均用时分别约为
`18.44/18.04/18.96/17.97` 分钟，四份日志均未命中 traceback、
CUDA OOM、non-finite 或 assertion。物理锁严格保持 node0=`GPU0/1/2` 三把、node1=
`GPU1` 一把；W3 的 stopped parent、raw child 与 CPU-only finalizer 状态仍分别为
`T/S/S`，没有误恢复旧父进程。16 个 Q32 fanout supervisor 继续严格保持 node0/node1=
`5/11`，07:30 heartbeat 全部刷新且 checkpoint 发布前没有提前获取 GPU 锁；官方
EvalPlus scorer 也仍存活等待。此时实际 CUDA 计算为 node0 三卡、node1 一卡，其他
卡没有可独立执行的正式 checkpoint，保留给 artifact 发布后的四路评测，避免通过重复
加载同一 32B 模型制造不公平计时或内存/IO 争用。重新运行 fail-closed release 审计仍为
**51/55**，仅缺四个 YAQA Qwen3-32B；GPU-hour 汇总保持 EfficientQAT/TurboBOA/
YAQA=`3.780685/19.661621/45.456876`，没有结果或计费漂移。

08:00 CST 半小时节点审计中，四个 Q32 stage 的 attention `o_proj` 完成层号更新为
W4A4/W4/W3/W2=`27/28/26/8`；四条日志年龄均小于 30 秒且错误扫描为空。node0 三张
量化卡当次利用率为 `73%/90%/86%`，node1/GPU1 为 `59%`（处于层内阶段切换）；物理
锁仍严格为 node0=`3`、node1=`1`，W3 stopped parent/raw child/finalizer 与官方 scorer
均存活且状态符合预期。两机 `memory.current` 约为 `962/1043 GB`，但 `memory.events`
继续为 `oom=0, oom_kill=0`，`max` 计数也未增加。开发机侧 JFS attribute cache 一度把
fanout log mtime 显示为 07:55；直接在两台 pod 内读取同一文件确认最新 heartbeat 实际
已到 07:59:28，node0/node1 的 component supervisor 仍严格为 `5/11`。控制面再次确认
两条 Canoe job 均为 `Running`、retry=`0`。重新构建 incomplete release 仍严格为
**51/55**，missing 仅四个 YAQA Qwen3-32B，三方法 GPU-hour 合计保持
`3.780685/19.661621/45.456876`。

09:00 CST 小时审计时，Q32 W4A4/W4/W3/W2 的 attention `o_proj` 完成层号为
`31/31/29/11`；最近 8 个相邻 `o_proj` 标志的平均耗时分别为
`18.79/18.57/19.30/18.61` 分钟，未出现速度突变。四份日志持续刷新且错误扫描为空；
W4A4 在写出 layer 30 `o_proj` 后短暂进入换层窗口，30 秒复查
GPU0 已恢复到 `43%`，不是挂起。node0/node1 的物理锁保持 `3/1`，fanout supervisor
保持 `5/11`，W3 stopped parent/raw child/finalizer 和官方 scorer 均存活。node0/node1
的 `memory.current` 约为 `979/1062 GB`；node1 已接近 cgroup 上限，因此继续不新增
32B reader，但两机 `memory.events.max` 均未增加，`oom=0, oom_kill=0`。重新运行
fail-closed release builder 仍为 **51/55**，missing 仅四个 YAQA Qwen3-32B；GPU-hour
汇总仍为 EfficientQAT/TurboBOA/YAQA=`3.780685/19.661621/45.456876`。

10:00 CST 小时审计时，Q32 W4A4/W4/W3/W2 的 attention `o_proj` 完成层号更新为
`34/34/32/14`，最近 8 个相邻 `o_proj` 标志的平均耗时分别为
`18.85/18.51/18.98/18.86` 分钟，速度继续稳定；
四份量化日志均为秒级更新且错误扫描为空。node0 三张量化卡当次利用率为
`65%/92%/73%`，node1/GPU1 为 `39%`（刚写出 layer 32 `o_proj` 后的层内阶段切换）；物理锁
仍严格为 node0=`GPU0/1/2`、node1=`GPU1`，fanout supervisor 保持 `5/11`，尚无
component receipt，符合四个 checkpoint 尚未发布的预期。W3 stopped parent/raw child/
finalizer 与官方 scorer 均存活且状态正确。node0/node1 的 `memory.current` 约为
`963/1070 GB`；node1 其中约 `1066 GB` 为可回收 file cache，anon 仅约 `4.38 GB`，
两机 `memory.events.max` 均未增加，`oom=0, oom_kill=0`。重新构建的 fail-closed
incomplete release 仍严格为 **51/55**，missing 精确为四个 YAQA Qwen3-32B；GPU-hour
合计保持 EfficientQAT/TurboBOA/YAQA=`3.780685/19.661621/45.456876`，监控制品为
`/tmp/additional_methods_release_monitor_20260822_1000.json`。同时补跑 release/recovery/
fanout 及 REAL-Q backend/固定标签/gradient-trace 的核心测试共 `51 passed`，未触碰
正在运行的冻结量化配置。

11:00 CST 小时审计时，Q32 W4A4/W4/W3/W2 的 attention `o_proj` 完成层号为
`37/37/35/17`，W4 已进入 layer 38；最近 8 个相邻 `o_proj` 标志的平均耗时分别为
`18.86/18.52/19.13/18.80` 分钟，
未出现速度漂移。四份日志均为秒级更新且错误扫描为空；node0 三张量化卡当次利用率为
`89%/43%/89%`，node1/GPU1 为 `74%`。物理锁严格保持 node0=`GPU0/1/2`、node1=
`GPU1`；fanout supervisor 仍为 `5/11`，两侧最大心跳年龄均仅 `81` 秒、错误数为 0，
尚无 component receipt；官方 scorer 心跳年龄 `4` 秒且错误数为 0。W3 stopped parent/
raw child/finalizer 状态继续为 `T/S/S`。node0/node1 的 `memory.current` 约为
`969/1072 GB`，node1 约 `1068 GB` 为 file cache、anon 仅约 `4.38 GB`；两机
`memory.events.max` 未增加，`oom=0, oom_kill=0`。重新构建的 incomplete release
仍严格为 **51/55**，missing 仅四个 YAQA Qwen3-32B，GPU-hour 合计保持
EfficientQAT/TurboBOA/YAQA=`3.780685/19.661621/45.456876`，监控制品为
`/tmp/additional_methods_release_monitor_20260822_1100.json`。本小时还核验了四个 fanout
claim、冻结 plan/源码身份、最终 160 行表格列映射，并补跑 W2 relocation/W3 finalizer
专项测试 `9 passed`，均未发现配置、计费或交接漂移。

12:00 CST 小时审计时，Q32 W4A4/W4/W3/W2 的 attention `o_proj` 完成层号更新为
`40/41/38/20`；最近 8 个相邻 `o_proj` 标志的平均耗时分别为
`18.88/18.81/19.23/18.84` 分钟，按 layer 63 `o_proj` 落盘估算的时间约为 8 月 22 日
`19:07/18:52/19:53` 和 8 月 23 日 `01:12`。四份量化日志年龄均不超过 30 秒且错误
扫描为空；node0 三张量化卡当次
利用率为 `82%/84%/23%`（GPU2 正处 W2 层内阶段切换），node1/GPU1 为 `83%`。
量化进程严格保持 node0/node1=`3/1`，fanout supervisor 保持 `5/11`，16 份心跳
最大年龄 `88` 秒、错误数为 0，尚无 component receipt；官方 scorer 继续正常等待
generation。W3 stopped parent/raw child/finalizer 状态仍严格为 `T/S/S`，没有误恢复
旧父进程。node0/node1 的 `memory.current` 约为 `983/1072 GB`；node1 约
`1068 GB` 为可回收 file cache、anon 仅约 `4.38 GB`，`memory.events.max` 保持
`210384` 未继续增加，两机均为 `oom=0, oom_kill=0`。Canoe 只读控制面核验两条
debug job 均为 `Running`、retry=`0`，且当前 retry 各恰有一个 Running master pod。
重新构建的 fail-closed incomplete release 仍严格为 **51/55**，missing 仅四个 YAQA
Qwen3-32B，GPU-hour 合计保持 EfficientQAT/TurboBOA/YAQA=
`3.780685/19.661621/45.456876`，监控制品为
`/tmp/additional_methods_release_monitor_20260822_1200.json`。本小时还以 CPU-only
方式补跑 REAL-Q backend/固定标签/gradient-trace 测试 `16 passed` 和三种新算法的
release/recovery/fanout 核心测试 `51 passed`；目标源码 AST `79/79`、JSON `3/3`
可解析，冻结量化脚本/计划/fanout 身份及最终提交范围均无漂移。

12:26 CST 对 raw checkpoint 做逐组件结构审计后，进一步收紧监控口径：上述 06:02--
12:00 的层号来自 `*_o.pt`，准确含义是该层 attention `o_proj` 已写出，并不代表同层
后续 `gate/up/down/layernorm` 已全部完成。这个区别只影响进度措辞和 ETA，不影响量化
数值、checkpoint 内容或 GPU-hour。今后以每层 `q/k/v/o/up/gate/down/layernorm` 八个
非空文件全部存在、最终 `layernorm.pt` 落盘作为严格完整标志。按该口径，12:26 的
W4A4/W4/W3/W2 严格完整层号为 `40/41/38/21`，层 `0..highest` 均无缺层或缺件；
下一层 `41/42/39/22` 正在按预期部分写入。最近 8 个严格完整层平均耗时为
`18.81/18.84/19.32/18.84` 分钟，layer 63 `layernorm.pt` ETA 修正为 8 月 22 日
`19:21/19:08/20:11` 和 8 月 23 日 `01:27`。后续巡检统一报告这一严格口径，并可同时
括注正在写入的 partial layer。

13:00 CST 小时审计继续使用上述严格口径：Q32 W4A4/W4/W3/W2 的最高完整层号为
`42/43/40/23`，即分别完成 `43/44/41/24` 个 transformer block；正在写入的下一层
`43/44/41/24` 已分别具备 `q,k,v,o,up`、`q,k,v,o`、`q,k,v,o,up` 和 `q,k,v,o`
组件。最近 8 个完整层平均耗时为 `18.85/18.85/19.60/18.84` 分钟，按 layer 63
`layernorm.pt` 外推的 ETA 分别为 8 月 22 日 `19:22/19:08/20:18` 和 8 月 23 日
`01:27`。四份量化日志年龄均不超过 21 秒且错误扫描为空；边界阶段后的复采样中，
node0/GPU0--2 利用率为 `70%/72%/89%`，node1/GPU1 为 `84%`。量化进程严格保持
node0/node1=`3/1`，fanout supervisor 保持 `5/11`，16 份日志心跳年龄为
`92--125` 秒、错误数为 0，且 component receipt 仍为 0，符合完整 checkpoint 尚未
通过发布门禁的预期；官方 scorer 保持 `51/55` 并正常等待 generation。两机
`memory.current` 约为 `1000/1072 GB`，node1 其中约 `1068 GB` 为可回收 file cache、
anon 约 `4.41 GB`；node1 的 `memory.events.max` 较 12:00 增加 2 次到 `210386`，但
两机仍为 `oom=0, oom_kill=0`。Canoe 控制面再次确认两条 8-GPU debug job 均为
`RUNNING`、retry=`0`，且各自恰有一个当前 retry 的 Running master pod。冻结 builder
在实验 pod 内重建的 fail-closed incomplete release 仍严格为 **51/55**，missing 精确为
四个 YAQA Qwen3-32B；GPU-hour 合计保持 EfficientQAT/TurboBOA/YAQA=
`3.780685/19.661621/45.456876`，监控制品为
`/tmp/additional_methods_release_monitor_20260822_1300.json`。本小时同时复核三份 REAL-Q
backend/固定标签/gradient-trace 不可变结果的 SHA 与 plan fingerprint，均为
`status=succeeded`，没有 REAL-Q GPU 计算残留；新算法发布路径与 REAL-Q 诊断核心回归
本轮合计 `49 passed`，仅有不影响功能的 pytest cache 写权限 warning。

14:00 CST 小时审计时，Q32 W4A4/W4/W3/W2 的最高严格完整层号为 `45/46/43/26`，
即完成 `46/47/44/27` 个 transformer block；下一层 `46/47/44/27` 已分别写到
`gate/up/up/up` 阶段。最近 8 个完整层平均耗时为 `18.79/18.77/19.28/18.80`
分钟，layer 63 `layernorm.pt` ETA 分别为 8 月 22 日 `19:21/19:06/20:09` 和
8 月 23 日 `01:26`，没有出现速度漂移。四份量化日志年龄均不超过 19 秒且错误扫描
为空；node0 三张量化卡利用率为 `90%/88%/45%`，node1/GPU1 为 `57%`。
量化进程继续严格为 node0/node1=`3/1`，W3 stopped parent/raw child/finalizer 状态仍为
`T/Sl/Ss`，旧父进程没有误恢复。fanout supervisor 进程保持 `5/11`；其日志采用约
5 分钟批量心跳，首次采样恰在周期边界，30 秒复采样确认 16 份日志已于
`14:00:05--06` 全部刷新，当前年龄约 `75` 秒、错误数为 0，component receipt 仍为
0。官方 scorer 正常等待，正式 suite 保持 `51/55`。node0/node1 的
`memory.current` 约为 `1017/1072 GB`；node1 约 `1068 GB` 为可回收 file cache、
anon 约 `4.39 GB`，两机 `memory.events.max` 均未增加，仍为
`oom=0, oom_kill=0`。Canoe 控制面确认两条 8-GPU debug job 都是 `RUNNING`、
retry=`0`，且各自只有一个当前 retry 的 Running master pod。冻结 builder 在实验 pod
内生成的 fail-closed incomplete release 仍严格为 **51/55**，missing 仅四个 YAQA
Qwen3-32B；GPU-hour 合计保持 EfficientQAT/TurboBOA/YAQA=
`3.780685/19.661621/45.456876`，监控制品为
`/tmp/additional_methods_release_monitor_20260822_1400.json`。本小时还验证了严格 release
builder 与表格 renderer 均会拒绝 51/55 的 incomplete 输入且不生成输出，冻结量化/
fanout 源码身份和最终定向提交范围也无漂移。

15:00 CST 小时审计时，Q32 W4A4/W4/W3/W2 的最高严格完整层号为 `49/49/47/29`，
即完成 `50/50/48/30` 个 transformer block；下一层 `50/50/48/30` 已分别写到
`o/gate/v/gate` 阶段。最近 8 个完整层平均耗时为 `18.78/18.79/18.99/18.80`
分钟，layer 63 `layernorm.pt` ETA 分别为 8 月 22 日 `19:20/19:07/20:03` 和
8 月 23 日 `01:26`，继续没有速度退化。四份量化日志错误扫描为空；W2 当次处于单个
线性层求解段，日志年龄约 59 秒但 GPU 进程持续计算，其余日志年龄不超过 25 秒。
node0 三张量化卡利用率为 `25%/88%/31%`（GPU0/GPU2 为层内阶段边界），
node1/GPU1 为 `51%`。量化进程仍严格为 node0/node1=`3/1`；W3 stopped parent/raw
child/finalizer 保持 `T/Sl/Ss`，官方 scorer 存活。fanout supervisor 保持 `5/11`，
16 份日志心跳年龄为 `68--103` 秒、错误数为 0、component receipt 为 0，正式 suite
仍为 `51/55`。node0/node1 的 `memory.current` 约为 `1035/1073 GB`；node1 约
`1068.73 GB` 为可回收 file cache、anon 约 `4.38 GB`，两机
`memory.events.max` 均未增加且仍为 `oom=0, oom_kill=0`。Canoe 控制面再次确认两条
8-GPU debug job 都是 `RUNNING`、retry=`0`，且各自只有一个当前 retry 的 Running
master pod。冻结 builder 在实验 pod 内生成的 fail-closed incomplete release 保持
**51/55**，missing 仅四个 YAQA Qwen3-32B；GPU-hour 合计保持
EfficientQAT/TurboBOA/YAQA=`3.780685/19.661621/45.456876`，监控制品为
`/tmp/additional_methods_release_monitor_20260822_1500.json`。

16:00 CST 小时审计时，Q32 W4A4/W4/W3/W2 分别完成 **`53/54/51/33` 个严格完整
transformer block**，即最高完整层号为 `52/53/50/32`；下一层 `53/54/51/33`
分别已写出 `q,k,v,o`、`v`、`q,k,v` 和 `q,k,v,o,up,gate`。对所有完成层逐一检查
`q/k/v/o/up/gate/down/layernorm` 八个非空 artifact，缺失数为 0；四个 raw 目录的
零字节和 `.tmp/.part/.partial` 文件也均为 0。最近 8 个严格完整层平均耗时为
`18.78/18.82/19.08/18.78` 分钟，layer 63 `layernorm.pt` ETA 分别为 8 月 22 日
`19:21/19:07/20:06` 和 8 月 23 日 `01:25`，速度没有异常漂移。四份量化日志年龄均
不超过 29 秒且错误扫描为空；node0 三张量化卡利用率为 `85%/46%/90%`，
node1/GPU1 为 `53%`，四张卡均保持 `1965 MHz`。

量化进程继续严格为 node0/node1=`3/1`；W3 stopped parent/raw child/finalizer 状态为
`T/Sl/Ss`，没有误恢复旧父进程。Q32 fanout supervisor 保持 node0/node1=`5/11`，
16 把 supervisor lock 全部存在，最老 heartbeat 仅 47 秒、错误数为 0、component
receipt 为 0；官方 scorer 每 30 秒心跳并保持 `official_completed=51/55`。
node0/node1 的 `memory.current` 约为 `1052/1074 GB`，其中 file cache 约
`1040/1069 GB`、anon 约 `12.60/4.39 GB`；`memory.events.max` 分别为
`463449/210394`，两机仍为 `oom=0, oom_kill=0`。Canoe 只读控制面确认两条 debug job
均为 `Running`、retry=`0`，且各自只有一个当前 retry 的 Running master pod。

冻结 builder 在 node1 生成的 fail-closed incomplete release 仍严格为 **51/55**，
missing 精确为四个 YAQA Qwen3-32B；GPU-hour 合计保持 EfficientQAT/TurboBOA/YAQA=
`3.780685/19.661621/45.456876`，监控制品为
`/tmp/additional_methods_release_monitor_20260822_1600.json`。本小时进一步把最终 renderer
扩展为一致的八算法标题、方法/公平性说明、160 行表、三方法计费、整数命中数恢复以及
60 项推理唯一/并列第一统计；EfficientQAT 的 5 个 W4A4 格子严格为 N/A。合成完整
release 的临时渲染得到 160 行和 5 个 N/A，原表 300 个及新增已完成行 153 个推理百分比
均可唯一恢复为整数命中数；相关 release/fanout/renderer 合同测试 `16 passed`。正式
55/55 release 尚未完成，因此主表仍未提前改写。

17:34--17:42 CST 的两次人工巡检继续使用八组件严格完整口径。
17:42 时 Q32 W4A4/W4/W3/W2 分别完成 **`58/59/56/39` 个 transformer
block**，下一层分别已写出 `q,k,v,o,up` / `q,k,v,o,up` / `q,k,v,o` /
`q,k,v,o`。相比 17:34 的 `58/59/55/38` ，W3 和 W2 均在预期时间内新完成一层，
W4A4/W4 则在各自层内继续推进。四份量化日志年龄为 `0--18` 秒且错误扫描为 0；
node0 GPU0--2 与 node1/GPU1 当次利用率为 `7%/84%/81%/88%`，其中 GPU0
处于线性层边界的短暂切换，日志仍在 4 秒内刷新，不是挂起。四张计算卡均保持
`1965 MHz`。

16 个 fanout supervisor 的最老心跳为 `131` 秒，错误数为 0，component
receipt 仍为 0，与四个 checkpoint 尚未发布一致。official scorer 每 30 秒继续
刷新并保持 `official_completed=51/55`。W3 stopped parent/raw child/finalizer 仍为
`T/Sl/Ss`，旧父进程没有被误恢复。两机 `memory.events` 均为 `oom=0,
oom_kill=0`；这次巡检没有改变任何量化、评测或调度配置。

18:30 CST 左右，W3 的 stopped worker parent `132939` 意外从 `T` 变为 sleeping；旧
finalizer 按身份/状态门禁 fail-closed 退出，留下
`j-zogxxxduju-master-0_w3_finalizer.log`，没有执行 hfize、validation 或发布 receipt。
逐项核验 `/proc` 的 PID/PPID/start-ticks/cmdline 后确认：raw quantizer child
`132942` 始终是同一个进程、日志持续增长、没有第二份 quantizer，也没有发生数值重跑。
随后只对 parent `132939` 恢复 `SIGSTOP`，没有向 child 发信号；并为
`finalize_stopped_q32_w3.py` 加入 append-only attempt 制品命名及测试，18:40:48 CST
启动 retry2 finalizer `245082`。此后人工巡检持续确认 parent/child/finalizer 为
`T/Sl/Ss`。恢复动作只修复收尾调度状态，没有改变 W3 的校准集、量化参数、随机种子或
正在运行的 numerical child。

19:36--19:52 CST，Q32-W4 与 Q32-W4A4 先后完成全部 64 个 block、hfize 和严格
checkpoint validation；两组均通过 `integer_reconstruction_verified=true` 与
`reload_deterministic=true`。W4 量化本体为 `19.575498 GPU·h`，计入 A16 Hessian
三组摊销后的公平成本为 **`20.844802 GPU·h`**；W4A4 量化本体为
`19.858011 GPU·h`，计入其独占 A4 Hessian 后为 **`23.717367 GPU·h`**。
对应四路 fanout 均仅在 `stage_receipt.json` 原子发布后启动。W4 quality 已于
19:51 CST 完成：`KL×100=4.318970`、`PPL=7.713556`、十项 QA 平均 `71.67`；
GSM8K、MATH-500、HumanEval+ 继续并行运行。19:52 CST 严格层口径下 W3 已完成
`63/64`、正在最后一层，W2 已完成 `46/64`、正在 layer 46；两机继续保持
`oom=0, oom_kill=0`。

20:09:51 CST，Q32-W4A4 quality component 原子发布成功：
`KL=12.1912260056`（`KL×100=1219.122601`）、`PPL=1038383.875`、十项 QA
平均 `31.88`。十项分别为 ARC-Challenge=`25.85`、ARC-Easy=`25.63`、
BoolQ=`50.80`、C-Eval=`25.26`、HellaSwag=`26.69`、LAMBADA=`0.00`、
OpenBookQA=`30.00`、PIQA=`50.00`、Social-IQA=`34.19`、Winogrande=`50.36`。
component receipt SHA256 为
`94fe43328e3fc07a96791f87134202c7f6742b7082aadf68d2d97fd61447c09b`，绑定的
stage receipt/validation SHA256 分别为
`50cd8ea38daa7f1ba48ee592f91736d328e34a9806dd9ca918c9a833238948c7` 与
`6cddeffde7cedc3765b598dea30d728633ab807d921db16dd1df1b6c656b4d79`。
该异常与 2026-07-27 独立 Q32-W4A4 重跑的 `KL=12.304107`、
`PPL=1169179.875` 同量级，且本轮 checkpoint 已通过严格整数重构与确定性 reload；
因此按 YAQA_wclip 正式算法退化原样发布，不把它误判为 hfize 或 evaluator 故障。

20:05:39 CST，Q32-W3 的原始量化 child `132942` 正常结束并写出
`algorithm_elapsed_seconds=72413.565431`。finalizer retry2 在下一次 30 秒轮询中按冻结
PID/start-ticks/handoff SHA 接管，只终止原 stopped queue parent 后执行 hfize 与严格
validation。监控容器第一次读取刚关闭的 `63_down.pt` 时因共享文件系统属性缓存暂时看到
0 字节；同一时刻执行机 node1 对同一 inode 的权威 `stat` 为 `397376597` 字节，且
hfize 随后正常加载该投影与全部 64 层。这是跨客户端 metadata cache 延迟，不是损坏
artifact，也没有做文件修补或数值重跑。

20:27:49 CST，Q32-W3 原子发布 `stage_receipt.json`：448 个投影均通过整数代码/尺度/
RHT 符号的严格重构，两次 HF reload 与 GPU smoke 完全确定；量化本体为
`20.114879 GPU·h`，独立计入 A16 Hessian 为 `23.922789 GPU·h`，按三组共享摊销后的
公平成本为 **`21.384183 GPU·h`**。stage receipt、validation 与 finalization success
的 SHA256 分别为
`8302485e585644bdcb5a026ff516d1e178ffaf4348f62273c850d7ddb5fe5b49`、
`a0b7df22e779ad3fca157bee2d7a44d37d97a166c0935184a89113c3df23bbaa`、
`7f72ffe0570c1803dc988c6f1554410b4708d8228c5d58494fb00f122306078d`。
旧 parent/quantizer/finalizer PID 均已消失；四路正式评测随即分别在 node1/GPU1
（quality）、node0/GPU4（GSM8K）、node0/GPU7（MATH-500）和 node1/GPU7
（HumanEval+）启动。W3 finalizer、Q32 fanout、release builder 与总表 renderer 的
定向 CPU 合同测试同期为 **23 passed**。

20:41:26 CST，Q32-W3 quality component 完整发布：
`KL=0.1372597516`（`KL×100=13.725975`）、`PPL=8.003909`、十项 QA 平均
`69.85`。十项分别为 ARC-Challenge=`57.85`、ARC-Easy=`79.08`、
BoolQ=`85.44`、C-Eval=`81.13`、HellaSwag=`80.36`、LAMBADA=`68.70`、
OpenBookQA=`46.20`、PIQA=`79.82`、Social-IQA=`48.77`、Winogrande=`71.11`。
quality receipt SHA256 为
`da9828ca3ace7dca7d73dad4931cae4e6817f83bfd7cf2400e6b1e3329dad6ef`，
其 output SHA256 为
`59460f6084ced2ad3d3bb9b698b40b7f4ccdc5d1f6e7acb6b8a7e92cbedbf840`；
receipt 继续绑定上一段相同的 stage/validation SHA。与 Q32-W4 的 QA Avg=`71.67`
相比下降 1.82 点，但 KL/PPL 与 2026-07-27 独立重跑的
`0.1386846453/8.0212` 高度一致。三项推理仍按四路 fanout 继续，不提前发布 suite。

21:00 CST 小时审计时，Q32-W4/W4A4/W3 的 GSM8K、MATH-500、
HumanEval+ 生成批次分别为 `39/42, 19/32, 11/11`、
`18/42, 9/32, 9/11` 和 `13/42, 7/32, 7/11`；所有活跃日志的错误扫描为
0。Q32-W2 仍为最后关键路径：严格完成 `49/64` 层，layer 49 已写出
`q/k/v/o/up/gate`，量化日志持续实时刷新。两机当时共 9 张 GPU 执行已就绪的
独立组件，最高显存为 node1/GPU5 的 `160.5/183.4 GiB`；两机
`memory.events` 均为 `oom=0, oom_kill=0`。Canoe 控制面确认两条 debug job
均为 `Running`、retry=`0`，且各自只有一个当前 retry 的 Running master pod。
冻结 release builder 在 node1 生成的 fail-closed 中间发布仍严格为
**51/55**，missing 精确为四个 YAQA Qwen3-32B；已完成行的 GPU-hour 合计保持
EfficientQAT/TurboBOA/YAQA=`3.780685/19.661621/45.456876`，中间制品为
`/tmp/additional_methods_release_monitor_20260822_2100.json`。

21:04:20 CST，Q32-W4 GSM8K component 完整发布：
`1234/1319=93.5557%`，覆盖全部 1319 条、greedy 且 seed=`1234`。component receipt
SHA256 为 `a99e7092eef94d57a395c715ca22c182155d1ffaa08293a9e74b998097a896a4`，
绑定的 reasoning manifest SHA256 为
`07c60514956707693c0900b3ba5917c6f2ef1a1f523e0d79ae3c5daaf996147d`。W4 仍需等待
MATH-500 生成和 HumanEval+ 官方评分，因此不提前将该组计入完整 suite。

21:47:47 CST，Q32-W4 的全部质量与推理评测正式闭合：
`KL=0.0431896970`（`KL×100=4.318970`）、`PPL=7.713556`、十项 QA 平均
`71.67`；GSM8K=`1234/1319=93.5557%`、MATH-500=`299/500=59.80%`、
HumanEval+ base=`73/164=44.5122%`、plus=`70/164=42.6829%`。MATH component
receipt SHA256 为
`365a7e1d00d39f922c3e7384ef80724cbb602d2f400888e5d3b8274a919dcf0d`；
suite success 与 EvalPlus official success SHA256 分别为
`68c6459c053a0d2e525e5bcb70f5bfe30c9f177058b8ab4e4f4c88feb6747393` 和
`56b70e39826e4b92212445d386f44287169123b0bff1f11e8b87018fc34dba49`。公平量化
成本为 `20.844802 GPU·h`。正式 suite/official 计数因此从 `51/55`提升到
**`52/55`**。冻结 builder 的 52-suite 中间发布仅余 Q32-W4A4/W3/W2 三组，
当前 YAQA 17 个完整设定合计 `66.301678 GPU·h`，制品为
`/tmp/additional_methods_release_monitor_20260822_2148.json`。

22:00 CST 小时审计时，Q32-W3 的 GSM8K 也已完成：
`1218/1319=92.3427%`，component receipt SHA256 为
`fef032a28e0132299683cc6c91c4a5e62617a23e404a2dfc895fd0d267143492`。W3 仅余
MATH-500（`21/32`）；W4A4 的 GSM8K/MATH-500 为 `36/42`和 `18/32`，
两组 HumanEval+ 都已生成 164/164。Q32-W2 严格完成 `52/64` 层，
layer 52 已写出 `q/k/v/o/up/gate`，日志持续实时刷新。随着 W4 全部评测和
W3 GSM8K 结束，当前只有 4 个已就绪的独立 GPU 组件：W2 量化、W3
MATH 以及 W4A4 的 GSM/MATH；四张卡当次利用率为 `80%--94%`。两机
`memory.events` 继续为 `oom=0, oom_kill=0`，活跃日志错误扫描为 0；Canoe
控制面再次确认两条 debug job 均为 `Running`。正式 suite/official 严格保持
**`52/55`**，不把 W3/W4A4 的部分 reasoning component 提前写入 release。

22:36--23:28 CST，Q32-W3 与 Q32-W4A4 的四路评测先后闭合并通过官方
HumanEval+ 评分。Q32-W3 为 `KL=0.1372597516`、`PPL=8.003909`、十项 QA
平均 `69.85`、GSM8K=`1218/1319=92.3427%`、MATH-500=`297/500=59.40%`、
HumanEval+ base/plus=`71/164=43.2927%` / `66/164=40.2439%`，公平量化成本
`21.384183 GPU·h`；suite 与 official receipt SHA256 分别为
`5c063fdcb199c66f9ddb3eab3248b99ad973ff8c5f44f803cb547780cf02b7c8` 和
`0be3fa375fa536745b52717eb33d49aaff01b717ceeb992022358fdf4859e7a0`。
Q32-W4A4 明显退化：`KL=12.19122601`、`PPL=1038383.875`、QA Avg=`31.88`、
GSM8K=`17/1319=1.2889%`、MATH-500=`0/500`、HumanEval+ base/plus 均
`0/164`，公平量化成本 `23.717367 GPU·h`；suite/official SHA256 为
`18c3f2d5a501b14dcce8a9fa2c5757bcc3dfbfe272c789f61796d06dfb08eabe` 和
`b28dccf935eb906d021c730d8c4f4955011995edb83fe715eaded5f6545a78a2`。
严格完整 suite 因而达到 **54/55**。

02:22:13 CST，最后一个量化项 Q32-W2 原子发布并通过严格重构、HF reload 与 GPU
smoke；stage receipt SHA256 为
`99169305a447eeaf0a5a3588729f31e4bd6d83af78f957c0879e8197a42fecc2`。
量化本体 `20.024468 GPU·h`，共享 A16 Hessian 为 `3.807910 GPU·h`，按三组
共享摊销后的公平成本为 `21.293771 GPU·h`。其 quality 为
`KL=15.62452126`、`PPL=27065838.0`、十项 QA 平均 `31.56`；03:50:36 CST
完成 GSM8K=`14/1319=1.0614%`，05:34:19 CST 完成 MATH-500=`0/500`。
GSM/MATH component receipt SHA256 分别为
`c35943f1a938bf7a3fc5d712ab51f7be5f1be254a14779326d2d200347540c7a`、
`c24003f3d99fdce896947298673e8d78a75c6abd088a7797ebfc6450121ebae0`，对应输出
SHA256 为 `67eabdb57a435ad653e17bd6a67a87f19cf58820c5755dbea447e3ce2e073b02`、
`f8df22a819bf93bb6648b825a32e09f2cac0bc17ab942bc16ccb6bb58363838b`。
MATH 运行峰值为 `176306/178352 MiB`，但完整生成 500/500、无 OOM。

05:40 CST 的 fail-closed release builder 确认为 **54/55**，唯一缺项精确为
`yaqa_wclip__YQ-Q32-W2` 的 HumanEval+；该组件继续按冻结计划等待 node1/GPU4 上
REALQ-SDPA Q32 正式量化释放后自动接棒，未改数值配置或迁移 receipt 身份。MATH
释放的 node1/GPU2 已由 REALQ-SDPA 推理 worker 在物理锁保护下接管；这只是调度交接，
没有重用模型状态或修改任何算法产物。

10:57 CST 再次用当前源码运行 fail-closed incomplete release builder，结果仍严格为
**54/55**，`missing_eval_ids` 只有 `yaqa_wclip__YQ-Q32-W2`。已完成行的统一量化
GPU-hour 合计为 EfficientQAT=`3.7806847124`、TurboBOA=`19.6616211174`、
YAQA-wclip（19 组）=`111.4032278443`，中间制品为
`/tmp/additional_methods_release_monitor_20260823_1057.json`。YAQA-W2 的
HumanEval+ supervisor PID `225066` 与官方 score worker PID `14307` 均存活，仍只等待
绑定的 node1/GPU4 物理锁；两机 cgroup 继续为 `oom=0, oom_kill=0`。新增三方法 release、
Q32 fanout、恢复路径和 REALQ-SDPA 最终表 overlay 的定向合同测试合计
**33 passed**；用合成完整的 55+40 release 对当前主文档做端到端只读渲染，得到严格
160 行、5 个 EfficientQAT N/A 行、40 个 SDPA REALQ 覆盖行，20 组方法顺序全部通过，
未提前改写正式主表。

13:08:18 CST，node1/GPU4 上的 REALQ-SDPA Q32 formal 已先发布 terminal receipt
并释放物理锁；YAQA-W2 HumanEval+ 于 13:09:50 CST 才发布 running manifest，间隔约
92 秒，严格证明先释放、后接棒、无 GPU 并发。component attempt child PID `355189`
绑定原冻结 fan-out plan，13:13 已生成 16/164；13:19 的显存快照为约 108.5 GiB，
supervisor PID `225066` 与 official scorer PID `14307` 均存活。两机 cgroup 仍为
`oom=0, oom_kill=0`。正式 additional-method release 在 HumanEval+ generation 和
官方 EvalPlus receipt 均完成前继续严格保持 **54/55**，不得仅凭 running manifest
提前发布。

13:46 CST 人工巡检时，YAQA-W2 HumanEval+ 已推进到 **144/164**，generation
文件仍持续增长，child PID `355189` 占 node1/GPU4 约 130.8 GiB、GPU util 约
97%；supervisor PID `225066` 与 official scorer PID `14307` 均存活。两机仍为
`oom=0, oom_kill=0`。正式 release 继续保持 54/55，等待余下 20 条生成、官方
EvalPlus score receipt 和 suite assembly 三重 terminal gate 后再发布 55/55。

13:51--13:54 CST，最后 20 条 HumanEval+ 生成完成并发布 component receipt；
receipt SHA256=`91a091a3f033d56b2e3cd65b67bc8cc61076ec5353bda4b2161816ed67945a39`。
四组件随后原子组装 `suite_success.json`，SHA256=
`e5bcc24a36b3afb21d5a17d34366a05d1417b27b60eb6d2130534b0432761210`。
共享官方 scorer 在 attempt001 完成 EvalPlus：base/plus 均为
`0/164=0%`，official receipt SHA256=
`c8c55d6798d148760271147d64add4ccf1a4bf4551e6f8088dba23caa3f83973`，
official result SHA256=
`5a188a52bbce0b650251ec6fce12b30fb960411f8f4c285e7e7009c3bfc252a8`。
至此最后一行 YQ-Q32-W2 的完整指标为 KL=`15.6245212555`、
PPL=`27065838.0`、QA Avg=`31.56`、GSM8K=`14/1319=1.0614%`、
MATH-500=`0/500`、HumanEval+ base/plus=`0/164`，公平量化成本
`21.293771 GPU·h`。

fail-closed release builder 已严格发布 **55/55**，missing=0；release SHA256=
`349faf670b477ccbd2afddb069f2f906756caf9b53e7049ff9a7ce831bcb2a6f`。
方法计数/总量化成本为 EfficientQAT `15 / 3.7806847124 GPU·h`、TurboBOA
`20 / 19.6616211174 GPU·h`、YAQA-wclip `20 / 132.6969989696 GPU·h`。
定向 release/fanout/八算法 renderer/SDPA overlay 合同测试为 **20 passed**；
正式主文档已机械渲染为 20 个 comparison id、每项 8 行、合计 160 行，5 个
EfficientQAT W4A4KV4 行保持 N/A，文档 SHA256=
`157b3fe56f9e3e2173d88009a704dc852f626a69277e1036963d5cd6449b430a`。
REALQ 行当前仍是已发布的 run15/FA4 数据，待 SDPA 40 行 release 完成后由第二层
fail-closed renderer 覆盖；新增三方法本身已经完整终态。

## 第三方源码可重建补丁（2026-08-23）

TurboBOA 与 YAQA-wclip 均是主仓库内的嵌套 Git checkout。为避免最终只提交调度器
而遗漏实际算法改造，已把 tracked modifications 以及相关 untracked
源码、脚本和测试冻结到父仓库：

- TurboBOA base commit=`ea88f93cd4b3730a4731d2dce7bc49971896bb18`，补丁
  `third_party_patches/fair20_20260821/turboboa.patch`，SHA-256=
  `0365bcd2857ec09ca96b6737e19af6cdb597c533f16c730120b5aed32be67652`；
- YAQA-wclip base commit=`f9508723251ad839f0162326569f17fe70486fcc`，补丁
  `third_party_patches/fair20_20260821/yaqa_wclip.patch`，SHA-256=
  `71e3a9c6e133f52adb95805d454f13cf3f0d478d7b472cf78c0f0b6f4d54fff3`。

两份补丁均对当前 campaign 源码通过 `git apply --check --reverse`。仅排除
`__pycache__`、编译扩展及 `qtip-kernels/build` 等生成物；应用方式和固定 base
commit 记录在 `third_party_patches/fair20_20260821/README.md`。
