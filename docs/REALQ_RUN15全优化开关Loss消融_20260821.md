# REAL-Q run15 全优化开关 Loss 消融（2026-08-21）

## 目的

排查切到 run15 优化执行路径后，数值扰动是否足以改变全模型量化结果。比较的主指标
是统一 canonical evaluator 得到的 WikiText2 exact full-vocabulary KL；PPL 作为辅助
指标。该消融不重新调学习率，因为目标是验证实现路径差异，而不是分别比较两个路径
各自重新调参后的最优算法结果。

## 旧路径与 V6/run15 的实际差异审计

`realq_qwen3_4b_aloss_clip_independent_lr_20260817` 的旧 SDPA 命令已经显式打开：

- `quantizer_inner_fastpath=true`；
- `symmetric_union_exact` weight clip；
- `fisher_fp32_cache=true`；
- `prefix_q_trailing_w_exact` act-order stitch；
- `where_out` clip update 与 compact group qparam；
- prepared clamp-bound cache、Triton column block、fused Block-Adam。

旧 V3 与 V6 计划中 `realq_layer.py`、`block_gd.py`、`fisher_loss.py`、
`kl_loss.py`、`layer_loop.py` 等核心文件 SHA256 也完全相同。Qwen3 Q/K
RMSNorm+RoPE、SwiGLU fusion 和 CUDA Fisher fused equation 已存在并由旧实验使用。
因此从旧 a-loss 路径到 V6 并不是重新更换了上述 quantizer/Block-GD 实现。

已确认的实际变化是：

1. attention 从 deterministic SDPA 改为 deterministic FA4；
2. worker 环境从 `NVIDIA_TF32_OVERRIDE=0` 改为 unset，Hessian/Fisher GEMM 的局部
   TF32 policy 因而可以真正生效；
3. 因 FA4 teacher 数值不同，V6 在独立目录重新生成 static Fisher 和 reference
   logits cache。

V6 扩大了 plan 的 source snapshot，新增记录 `akv.py`、attention wrapper 与若干 Triton
文件；这不代表它们在 V6 才启用。上述文件的工作区 mtime 均为 2026-08-07/08，早于
旧 static cache（08-08 10:26/15:05 CST）和 08-17 a-loss 实验；旧命令也已经显式打开
column/stitch/fused-Adam 等公开开关。另有纯运行时差异：V6 设置
`PYTORCH[_CUDA]_ALLOC_CONF=expandable_segments:True`，formal 阶段跳过内联评测并改用
新 output/cache 目录；这些属于内存/调度和产物组织，不是新的量化算法。

固定未变的项目包括模型、WikiText2 256×2048 calibration token artifact、seed=1、
rotation seed=0、refresh seed=0、group size 128、W/A/K/V 对称性、act-order、rotate、
Block-GD scope、sliding window、batch 配置、a-loss ratio、LR 和调度器。

## 消融设计

三组覆盖模型规模、权重 bit 和低比特 activation：

| 设定 | 分支 | 冻结 LR | 调度器 |
|---|---|---:|---|
| Qwen3-0.6B W4A16 | full-block | `1.3335214321633242e-4` | inverse cosine |
| Qwen3-4B W3A16 | full-block | `1.3335214321633242e-5` | inverse cosine |
| Qwen3-8B W4A4KV4 | full-block | `7e-6` | fixed |

“全开”直接复用 source-locked V6 formal checkpoint，避免无意义地重复同一个确定性
量化。“全关”保持上表全部科学配置和 calibration/seeds 不变，切换为：SDPA、Hessian
FP32、Fisher eager FP32、Qwen3 Q/K/MLP eager equation、activation QDQ eager、
Fast-Hadamard 单独 scale、legacy Cartesian clip、expanded qparam、legacy act-order
stitch、Python column recurrence、unfused Adam，并关闭所有公开 run15 fast-path flags。

两个 checkpoint 最后都通过同一个 deterministic-SDPA/FP32/eager canonical evaluator
计算 exact KL/PPL；reference logits 按模型只生成一次并强制复用。这样不会把 FA4 与
SDPA teacher cache 的差异直接当成 checkpoint loss 差异。

冻结计划：
`../experiment_data/realq_allopts_loss_ablation_20260821_v1/plan.json`，fingerprint
`36d0d14e0157ce3759705509bc14917f131998adb80275dd609fa859381914e5`。

## 执行状态

- 2026-08-21 19:04 CST：计划与 runner 已冻结；等待三种新算法队列释放合适 GPU 后
  依次提交，不抢占正在计算的 YAQA/TurboBOA/EfficientQAT 进程。
- Qwen3-4B W4A16 的 V6 旧精确最优 LR `7.4989420933245589e-6` 另行在 node1 GPU1
  复核；该卡此前显存为 0、YAQA worker 仅等待 Qwen3-32B Hessian 依赖。复核完成后
  supervisor 会自动恢复 YAQA GPU1 lane，再恢复该 lane 的评测 worker。

### 旧最优 LR 在 run15 路径上的精确复核

2026-08-21 19:28 CST，Qwen3-4B W4A16 full-block 在 run15/FA4/TF32 路径上以旧路径
精确最优 LR `7.4989420933245589e-6` 完成：Exact-KL=`0.04658717289566994`、
PPL=`13.761168479919434`，单卡 wall/GPU time=`1555.5594 s`。它与新网格的
`7e-6` 点 KL=`0.0464841090` 只相差 `+0.000103064`（`+0.2217%`），但比新路径
零端点 KL=`0.04490520805` 差 `+0.001681965`（`+3.7456%`）。因此 Q4-W4 的
零端点选择不是因为粗网格漏掉 `7.5e-6` 附近的窄最优谷。

旧 deterministic-SDPA/全局 TF32-off 路径在同一精确 LR 的 KL 为
`0.04061254859`；切到 run15 执行环境后增加 `0.0059746243`（`+14.7113%`）。结合核心
Block-GD/quantizer 文件 SHA 与公开 fast-path flags 均未改变的审计，当前嫌疑已收敛为
attention/backend 生成的 Fisher/reference cache 与有效 TF32 policy 的交互，而不是
学习率候选点遗漏。对应不可变结果为
`../experiment_data/realq_fullblock_q4w4_lr0_diagnostic_20260821_v1/runs/`
`run15_exact_old_best_fused/result.json`。

同卡已在不等待人工确认的情况下自动接上 Qwen3-0.6B W4A16 的 all-on/all-off pair；
其余 15 张卡继续 YAQA、EfficientQAT/TurboBOA 统一评测。

2026-08-21 19:43 CST，Qwen3-0.6B W4A16 pair 完成。统一 canonical evaluator 下，
all-off 相比 all-on 的 Exact-KL 仅变化 `-0.00014241785`（`-0.2903%`），PPL 变化
`-0.01086044`。首组说明全部优化路径叠加造成的最终 loss 漂移处于千分之三量级，且
方向为 all-off 略好；它本身不足以解释上节 Qwen3-4B W4A16 同 LR 跨旧/new
Fisher/backend 环境的 `+14.7113%` KL 差异。all-off 正式量化耗时 `749.45 s`
（`0.2082 GPU·h`；不含两次 canonical 评测），而复用 checkpoint 对应的原始 all-on
formal 量化耗时 `479.91 s`，因此 all-off 慢 `1.5616×`。GPU1 已自动接续
Qwen3-4B W3A16，随后将接续 Qwen3-8B W4A4KV4；其余 15 卡保持三种新算法任务。

### Static cache 数值突变定位（2026-08-21 20:04 CST）

进一步 mmap 比较 Qwen3-4B 的历史 SDPA 与 run15 FA4 static cache。诊断脚本为
`experiments/realq_allopts_loss_ablation_20260821/compare_static_cache.py`：每层只读取
Fisher 首/中/尾固定 `16×16` block 与 128 个对角点，并读取七类 saliency 的固定
首/中/尾片段；每一对 cache 共比较 87,552 个 Fisher 值和 290,304 个 saliency 值，
不把完整 5.2 GB cache 同时读入内存。

同为 `global_loss_bsz=4` 的旧 `e306d5b89c27` 与 run15 `e306d5b89c27` 结果为：

| 张量 | 不相等样本比例 | 旧 RMS | run15 RMS | diff RMS / 旧 RMS | cosine |
|---|---:|---:|---:|---:|---:|
| block-boundary Fisher | 96.6009% | 0.000581418 | 0.019621043 | 33.1041 | 0.651492 |
| linear-output saliency | 99.9759% | 4,814,987.58 | 5,068,416.96 | 0.151672 | 0.990389 |

Fisher 差异集中在前半模型：layer 0 的旧/new sampled RMS 分别为
`0.00217888/0.06673748`，layer 16 的 diff/旧 RMS 仍为 `0.4123`，layer 17 降到
`0.02118`，后续约为 1% 量级。saliency 模块级 diff/旧 RMS 为 q_proj
`0.27150`、k_proj `0.28895`、v_proj `0.15149`、o_proj `0.12928`、gate/up/down
分别 `0.14144/0.14573/0.13291`；差异源首先出现在 attention 梯度，再向其余模块传播
的特征很明显。

时间序列排除了历史文件被后期覆盖：2026-08-08 与 2026-08-11 的 tuning cache 固定
样本逐值相同；08-11 到 08-18/run15 才发生上述突变。run15 自己按 bsz=4/8 独立生成的
两份 cache，saliency 固定样本逐值相同，Fisher diff/左 RMS 仅 `1.64e-7`，说明新路径
本身稳定可复现。旧 bsz=4/8 cache 则分别有 Fisher `0.4684`、saliency `0.2104` 的
相对 RMS 差；batch invariance 的历史缺口需要另行记录，但它不是本次旧/new 33×
Fisher 尺度突变的原因。

源码追溯确认 `realq/precompute/static_e2e.py` 与 `hooks.py` 当前 SHA256 分别为
`2dd9f02c...`、`01c9c06f...`，与 2026-08-06 commit `d5a0b93` 内容相同，早于三批
cache；因此 producer 数学源码没有在 08-11 到 08-18 之间改变。`realq.ptq` 和
reproducibility 又在 precompute 阶段全局关闭 CUDA matmul TF32，命令中的
`hessian_tf32` 只进入后续 RealQLayer/refresh，不影响 static cache。剩余主要变量是
deterministic FA4 vs deterministic SDPA。

当前还发现一个独立的 provenance 缺口：`realq/precompute/cache.py` 的 schema-v5 key
包含 model/token/seed/rotation/global-loss-bsz 等字段，但不包含
`attention_backend`。实测 SDPA/FA4 cache 数值显著不同，却会生成同一 key；若两种 backend
共用目录，后运行者会静默命中先前 backend 的 cache。run15 使用独立目录所以没有实际
串用旧 cache，但文件名无法证明 backend 身份。生产代码暂不修改，等待当前代码重新生成
一份 SDPA cache、与历史 SDPA cache 做配对确认后再决定修复。

同轮源码审计确认 reference-logit cache 也有同类缺口：`utils/eval_utils.py` 的
schema-v2 metadata 已包含 source model、rotation、精确 eval token、dtype、tokenizer 和
hidden-state shape，但没有 `attention_backend` 或 SDPA kernel policy。FA4/SDPA 因而也会
得到同一个 reference cache tag；若共用 `cache_dir`，同样可能静默复用另一 backend 的
FP hidden states。本轮旧实验、run15 与 canonical evaluator 使用彼此独立目录，因此没有
发生实际交叉；本消融的 canonical all-on/all-off 又强制共用一份新生成的 deterministic-
SDPA reference，指标口径不受影响。该字段将和 static cache key 一起在 source-locked
三组实验结束后修复，避免现在改源码导致冻结计划漂移。

现有 production-shape FA4 检查只验证单个 attention kernel：output/dQ/dK/dV 相对 L2
约 `0.197%/0.369%/0.404%/0.265%`。它没有覆盖 36 层 NLL backward 后的
block-boundary Fisher 尺度，因此原门禁通过不能排除上述逐层放大。对应取样结果保存在
`../experiment_data/realq_allopts_loss_ablation_20260821_v1/cache_sample_diff/`。

### Current-code SDPA 复现闭环（2026-08-21 21:10 CST）

冻结诊断在 node1/GPU1 用当前工作区代码、run15 producer 的同一命令，仅把
`attention_backend=flash_attention_4` 改为 deterministic `sdpa`，重新生成
Qwen3-4B `256×2048/global_loss_bsz=4` static cache；producer 返回码 0，端到端耗时
`182.146 s`。结果文件为
`../experiment_data/realq_static_backend_diagnostic_20260821_v1/sdpa_current_code/result.json`。

当前 SDPA 与历史 SDPA 的固定样本比较为：

| 张量 | 样本数 | 不相等比例 | diff RMS | cosine |
|---|---:|---:|---:|---:|
| block-boundary Fisher | 87,552 | **0** | **0** | 1.0 |
| linear-output saliency | 290,304 | **0** | **0** | 1.0 |

同一份当前 SDPA cache 与 run15 FA4 cache 比较则复现了原差异：Fisher 不相等
`96.6009%`、diff/SDPA RMS=`33.1041`、cosine=`0.651492`；saliency 不相等
`99.9759%`、diff/SDPA RMS=`0.151672`、cosine=`0.990389`。因此 08-11 到 run15 的
static cache 突变已明确收敛到 **attention backend**：当前代码走 SDPA 可以逐值复现
历史固定样本，排除了 producer 源码版本与 Stage-0 TF32 policy 作为该突变的解释。
这也证明 deterministic FA4 的“重复运行确定”不等于“与 deterministic SDPA 数值等价”；
其小的单层 kernel 差异至少在 categorical Fisher/36 层 backward 管线中被显著放大，
其中 sampled label 改变占多少仍需单独计数。后续还需
用同一 SDPA cache 比较 fast-path on/off，才能把非 attention fusion 的最终 KL 漂移单独
剥离；不把当前 Q4 综合 pair 的 6.61% 全归给某一个 fusion。

## 结果

| 设定 | all-on exact KL | all-off exact KL | off-on | 相对差 | all-on PPL | all-off PPL |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B W4A16 | 0.0490612574 | 0.0489188395 | -0.0001424178 | -0.2903% | 21.490572 | 21.479712 |
| Qwen3-4B W3A16 | 0.1394655555 | 0.1302433312 | -0.0092222244 | -6.6125% | 14.107749 | 14.128666 |
| Qwen3-8B W4A4KV4 | 0.2332827151 | 0.2357314080 | +0.0024486929 | +1.0497% | 10.843867 | 10.850874 |

2026-08-21 21:07 CST，Qwen3-4B W3A16 pair 正式闭合。all-off 量化 36/36 层、
返回码 0，耗时 `4738.897 s = 1.31636 GPU·h`；两侧 checkpoint 均由同一份新生成并
强制命中的 deterministic-SDPA reference、FP32 full-vocabulary evaluator 重载评测。
all-off 的 Exact-KL 比 all-on 低 `0.0092222244`（`-6.6125%`），但 PPL 高
`0.02091694`。因此 Q4 的综合开关差异已经超出普通舍入误差；由于 all-on checkpoint
来自 FA4 static Fisher、all-off 来自历史 SDPA static Fisher，这一 pair 本身还不能把
差异归因给某个单独 fast path。supervisor 已于 21:07:10 CST 自动启动 current-code
deterministic-SDPA cache producer；其与历史 SDPA/run15 FA4 的固定样本比较将完成下一步
归因，再接续 Qwen3-8B W4A4KV4 pair。

同一 formal 外层墙钟口径下，复用的 all-on run15 量化为
`1395.171 s = 0.387548 GPU·h`，all-off 为 `4738.897 s = 1.316360 GPU·h`；全关慢
`3.3966×`。该数字只描述综合 fast-path 的工程收益，不把 cache/backend 引起的质量差异
误归给某一项单独 fusion。

2026-08-21 23:47 CST，Qwen3-8B W4A4KV4 pair 正式闭合。all-off 于 23:45:39 CST
完成 `36/36`、checkpoint 原子保存和返回码 0，量化墙钟为
`9137.793 s = 2.538276 GPU·h`；随后与 all-on checkpoint 共用同一 deterministic-SDPA、
FP32 full-vocabulary canonical evaluator。all-off 的 Exact-KL 比 all-on 高
`0.0024486929`（`+1.0497%`），PPL 高 `0.00700665`。因此三组综合开关的相对 KL
变化分别为 `-0.2903%/-6.6125%/+1.0497%`：影响具有模型/设定依赖性，不能用一个
固定舍入误差界或固定方向概括；同时该综合 pair 仍混合了 checkpoint 生成时的
FA4-vs-SDPA static Fisher 差异，不能据此把 8B 的 1.05% 单独归因给某一项 fast path。

复用的 all-on run15 formal 量化为 `2851.700 s = 0.792139 GPU·h`，综合 all-off 慢
`3.2043×`。pair terminal 发布后，supervisor 无需人工确认即在 node1/GPU1 启动
`YQ-Q32-W3`，REAL-Q 不再占用实验卡；三种新算法恢复使用全部 16 条物理 lane。

### 同一 SDPA cache 的非 attention 快路径隔离（2026-08-22）

综合 pair 中 Qwen3-4B W3 的 `-6.6125%` 仍混入了 FA4/SDPA static Fisher 差异，不能
直接归因于 Triton、quantizer 或 fused Adam。为继续归因而不长期挤占三种新算法，新增
独立实验 `experiments/realq_sdpa_fastpath_isolation_20260822/`，只选择差异最大的
Qwen3-4B W3：冻结同一模型、token artifact、全部 seed、LR=`1.3335214321633242e-5`、
inverse-cosine 调度和 full-block scope，并让 fast-on 与既有 all-off checkpoint 强制
使用同一份 current-code 可逐值复现的 deterministic-SDPA static cache。fast-on 只保留
其余 run15 优化与 scoped TF32；最终 checkpoint 继续使用上一轮完全相同的 canonical
SDPA/FP32 reference cache 评测。

冻结计划位于
`../experiment_data/realq_sdpa_fastpath_isolation_20260822_v1/plan.json`，fingerprint 为
`313bd30d6eddb2c47c0ed82539e806dd15c08c48a75a8d6678974a3fa1d62fc6`；命令构造、环境
门禁与 canonical cache 复用测试 `3 passed`。01:28 CST 最初预约 node0/GPU7；01:36
CST 巡检发现 node0/GPU4 的 `YQ-L8-W4A4` 已到 HumanEval 160/164，明显会更早释放，
因此取消尚未取得锁的 GPU7 预约并立即恢复其父 worker，改为暂停 GPU4 的父 worker。
两个评测子任务始终继续自然运行；待 GPU4 suite 原子发布并确认 CUDA 进程清空后才
交接给本诊断，诊断退出后自动恢复该 GPU 的统一评测 worker。

02:39:20 CST，同-SDPA/cache fast-on 量化完成 36/36 层并原子保存 checkpoint，返回码
为 0；量化墙钟为 `3667.429 s = 1.018730 GPU·h`。随后用上一轮 all-off 完全相同的
deterministic-SDPA、FP32 full-vocabulary reference cache 重载评测，canonical evaluator
于 02:41:07 CST 返回成功。最终结果为：

| Qwen3-4B W3A16 路径 | Exact-KL | PPL | 量化 GPU·h |
|---|---:|---:|---:|
| 原 run15 FA4 fast-on | 0.1394655555 | 14.1077490 | 0.387548 |
| **同 SDPA cache fast-on** | **0.1293588132** | **14.0839081** | **1.018730** |
| 同 SDPA cache all-off | 0.1302433312 | 14.1286659 | 1.316360 |

在严格同 cache 比较中，all-off 减 fast-on 的 KL 为 `+0.0008845180`，即 all-off
相对 fast-on **高 `0.6838%`**；PPL 高 `0.04475784`（`+0.3178%`）。因此除 attention
backend 外的 run15 fastpath 与 scoped TF32 合计并没有造成上一轮表面上的 6.61% KL
劣化，方向反而是小幅改善。工程上，同-SDPA fast-on 比 all-off 快 `1.2922×`，但因
SDPA 本身比 FA4 慢，它仍比原 FA4 fast-on 慢 `2.6287×`。

反过来固定其余 fastpath、只比较 checkpoint 生成时的 backend/cache，原 FA4 fast-on
比同-SDPA fast-on 的 KL 高 `0.0101067424`，相对后者高 **`7.8130%`**；PPL 只高
`0.02384090`（`+0.1693%`）。这个 backend 项足以覆盖并反转同-cache fastpath 的
`0.6838%` 改善，定量解释了综合 all-on/all-off pair 为何呈现 all-off 好 `6.6125%`。
结论是：之前的“优化全开最终 loss 变差”主要不是 Triton、quantizer、fused Adam 等
快路径数值误差，而是 deterministic FA4 与 SDPA 在逐层 backward/Fisher 构造中的
数值差异被放大；PPL 对该差异明显不如 full-vocabulary KL 敏感。

不可变汇总为
`../experiment_data/realq_sdpa_fastpath_isolation_20260822_v1/pair_result.json`，绑定 plan
fingerprint `313bd30d6eddb2c47c0ed82539e806dd15c08c48a75a8d6678974a3fa1d62fc6`；
canonical 评测耗时 `106.835 s`。诊断 terminal 发布后 supervisor 已自动 exec 回
node0/GPU4 的三算法统一评测 worker，REAL-Q 不再占用实验卡。

该 handoff 首次 exec 随后暴露了一个与 REAL-Q 数值结果无关的恢复环境缺口：supervisor
继承的 `PYTHONPATH` 没有统一 evaluator 已审计的 `evalplus==0.3.1` metadata 目录，
worker 因而让 10 个尚未开始的 Qwen3-0.6B suite 各在模型加载前瞬时消耗三次 attempt。
所有日志均为同一 `PackageNotFoundError`，无 CUDA context、quality 或 suite 产物。
`recover_post_realq_evalplus_attempts.py` 对 30 份 manifest/result/log 的 hostname、GPU、
错误文本和 SHA 逐一门禁后，于 02:46:43 CST 将其原样移动到
`_superseded_missing_evalplus_pythonpath_20260822_post_realq_handoff/`，恢复正常 retry
budget；旧失败没有删除。supervisor 已改为先 preflight EvalPlus 版本再用显式环境
`execve` evaluator。02:46:57 CST GPU4 重新领取 `TB20-Q06-W4A4`，随后穿过 runtime
gate、进入 28 层模型 forward 并建立 CUDA context，确认该 lane 已恢复。

### FA4/SDPA sampled label 分歧闭环（2026-08-22 06:00 CST）

为继续解释“单 attention kernel 相对误差不到 0.5%，但 block-boundary Fisher sampled
RMS 可相差 33 倍”的放大链路，新增独立诊断
`experiments/realq_backend_label_diagnostic_20260822/`。它从同一 run15 producer receipt
构造 paired Stage-0 命令，严格固定 Qwen3-4B、同一 WikiText2 `256×2048` token
artifact、calibration seed=`1`、rotation seed=`0`、global-loss bsz=`4`、完整 vocabulary
和所有其余参数；两臂唯一数值差异为 `attention_backend=sdpa/flash_attention_4`。wrapper
调用 production `deterministic_categorical_labels` 后只把已经生成的 int64 label 复制到
审计文件，不改变 logits、label 或 loss，再继续原始 backward。

两臂均返回 0 并分别捕获精确 `256×2048=524,288` 个 label，原始 label SHA256 为
SDPA `ae23e11d9096…`、FA4 `f77045454851…`。逐位置比较结果为：

| 指标 | SDPA vs FA4 |
|---|---:|
| 不同 sampled label | **6,520 / 524,288** |
| 不同 label 比例 | **1.2435913%** |
| 至少一个 label 改变的序列 | **256 / 256（100%）** |
| 每条 2048-token 序列不同数 | 均值 **25.46875**，最少 11，最多 42 |

因此 backend 的小 logits 扰动并不只产生连续的小梯度扰动：它让约 1.24% token 的
categorical target 直接换成另一个 vocabulary ID，且覆盖全部校准序列。cross-entropy
中的 one-hot target 项随之发生离散跳变，提出了一个可能的放大通道；但仅凭标签计数
不能判断它在 33× Fisher 差异中的占比，必须用两 backend 强制共享同一标签的完整
backward 配对实验验证。它也进一步说明 deterministic FA4 和 deterministic SDPA
各自重复确定，并不意味着两者会生成相同 Fisher 监督信号。

冻结计划 fingerprint 为 `d0287668fb41…`，不可变比较为
`../experiment_data/realq_backend_label_diagnostic_20260822_v1/comparison.json`，SHA256
`672065f4d5e0…`。首次比较命令仅在收尾处把字符串传给了要求 `Path` 的 SHA helper；
两份已校验 capture 无需重跑，独立 adapter 绑定 frozen runner/plan/两臂 result SHA，
只做 `str→Path` 类型转换后发布 comparison，`numerical_contract_changed=false`，receipt
位于同目录 `comparison_adapter_receipt.json`。两臂执行期间 cgroup 仅发生 file-cache
reclaim，`oom=0, oom_kill=0`；完成后 GPU0/GPU2 均已释放。

### 固定 categorical labels 后的完整 Fisher 配对（2026-08-22 06:37 CST）

为直接测量上述离散标签通道的贡献，新增
`experiments/realq_fixed_label_backend_diagnostic_20260822/`。两臂继续使用完全相同的
Qwen3-4B、WikiText2 `256×2048` token、seed、rotation、global-loss bsz=4 和完整
Stage-0 producer，只把先前 native SDPA 捕获的 `524,288` 个 int64 labels 按原 64 次
调用的 sample-index/shape/offset 原样注入 SDPA 与 FA4。production sampler 不再调用；
每次调用身份、label SHA 和完整 cache terminal 都 fail-closed。冻结 plan fingerprint 为
`3c68518f685b8e2d319e4de8873c102c06d7927c008605102f1adb39a54d65a2`。

注入门禁首先得到一个关键正对照：native SDPA 与 fixed-label SDPA 的 87,552 个 Fisher
样本和 290,304 个 saliency 样本均逐值相同，diff RMS=`0`、cosine=`1.0`。因此 wrapper
确实重放了 native SDPA 的原标签，未改变其余 producer 数值路径。完整比较如下：

| 配对 | Fisher diff/左 RMS | Fisher cosine | saliency diff/左 RMS | saliency cosine |
|---|---:|---:|---:|---:|
| native SDPA → native FA4 | 33.1041 | 0.651492 | 0.151672 | 0.990389 |
| **fixed-label SDPA → fixed-label FA4** | **36.3402** | **0.678852** | **0.290022** | **0.986785** |
| native FA4 → fixed-label FA4 | 0.178476 | 0.989745 | 0.256116 | 0.983681 |
| native SDPA → fixed-label SDPA | 0 | 1.0 | 0 | 1.0 |

结论与上一节的初步猜测不同：强制两 backend 使用完全相同的 target 后，Fisher 绝对
diff RMS 从 `0.0192473` **增加**到 `0.0211288`（`+9.78%`），相对 SDPA RMS 从
`33.10×` 增至 `36.34×`；saliency 差异也从 `15.17%` 增至 `29.00%`。因此 1.24%
label flip 虽然会显著改变 FA4 自身 Fisher（native→fixed 的相对 diff `17.85%`），但
**不是 SDPA/FA4 巨大 backend gap 的主因，整体上反而部分抵消了同标签时的差异**。
主要放大发生在相同标签下的 36 层 forward/backward 数值传播：fixed-label layer 0
Fisher diff/SDPA RMS=`33.02×`，到 layer 16 为 `0.435×`、layer 17 为 `0.0382×`，
之后快速收敛到百分之几以下，符合反向传播深度累积而非某个离散标签边界的特征。

不可变结果为
`../experiment_data/realq_fixed_label_backend_diagnostic_20260822_v1/comparison.json`；
SDPA/FA4 两臂分别耗时 `203.23/151.02 s`，返回码均为 0。两臂使用 node1/GPU0/2
物理锁，期间 `oom=0, oom_kill=0`，完成后显存和锁均释放。下一步应在固定标签下对
attention output 与 dQ/dK/dV 做逐层/逐反向深度误差曲线，而不再把主要嫌疑放在 label
sampling。

### 固定标签下的逐层有符号 forward/gradient trace（2026-08-22 06:54 CST）

为继续定位上述同标签 backend gap，新增并冻结
`experiments/realq_fixed_label_gradient_trace_20260822/`。SDPA/FA4 两臂复用上一节完全
相同的 Qwen3-4B、WikiText2 `256×2048` token、rotation/seed、global-loss bsz=4 和
`524,288` 个固定 SDPA labels；唯一数值差异仍是 attention backend。只读 hook 对第一个
global batch（sample 0--3）在 36 层各采样 block output、q/k/v projection output、
`o_proj` input/output 的 12 个 token × 最多 64 个 feature，并同时保存 forward 与该张量
在本次 backward 收到的有符号梯度，共 `36×6×2=432` 个 tensor。production Fisher、
saliency 和量化代码均未改动；plan fingerprint=
`3a53cca957da0e729e62051e86ffef85f846279a3f5bbb28dddbeea6b3f85e00`。

第一处 forward 差异被明确定位在 **layer 0 attention 输出，即 `o_proj` input**：layer 0
q/k/v projection output 在两臂逐值相同，而 `o_proj` input 已有 `11.65%` 采样值不等，
diff/SDPA RMS=`0.1009%`；经过 `o_proj` 后为 `0.1128%`，block output 为 `0.3215%`。
因此差异不是由 q/k/v 线性层、token、固定 label 或 rotation 首先引入，而是在相同
Q/K/V 上执行 SDPA 与 FA4 attention kernel 时首次出现。

反向差异随后沿深度显著放大。下表是 block-output gradient 的代表层；原始 gradient
保留 production 的 loss scale=`1000`，但相对 RMS 与 cosine 不受该公共缩放影响：

| layer（反传方向 35→0） | gradient diff/SDPA RMS | cosine | SDPA RMS | FA4 RMS |
|---:|---:|---:|---:|---:|
| 35 | 3.31% | 0.999453 | 3.0336 | 3.0280 |
| 32 | 7.31% | 0.997322 | 3.6786 | 3.6680 |
| 20 | 8.54% | 0.996372 | 15.0638 | 14.8948 |
| 17 | 17.35% | 0.984848 | 21.4295 | 21.1756 |
| 16 | 24.23% | 0.970669 | 21.8920 | 21.9043 |
| 12 | 2.22× | 0.390161 | 21.6717 | 52.1787 |
| 8 | 2.29× | 0.361081 | 25.4107 | 62.4597 |
| 4 | 9.54× | -0.042084 | 48.9661 | 462.3540 |
| 0 | 6.34× | 0.110384 | 105.8718 | 674.5400 |

layer 0 内部也同向：q/k/v projection output 的 forward 全部逐值相同，但其 gradient
diff/SDPA RMS 分别已达 `3.00×/4.53×/8.31×`，其中 V-gradient cosine 仅
`1.11e-5`；`o_proj` input/output gradient 分别为 `6.69×/7.37×`。这与完整 Fisher
比较中早层巨大、约 layer 16--17 后快速减弱的曲线一致。当前证据因此把主因收窄为：
**FA4 与 SDPA 在相同 Q/K/V 和相同 loss target 下产生的小 attention forward 误差，经过
attention backward 与跨层反传后被放大；不是 label sampling，也不是其他量化快路径。**
这里的 signed trace 是固定位置抽样而非全 tensor 范数，所以用于定位首次分歧和放大趋势，
完整 Fisher 仍作为总体量级依据。

不可变结果为
`../experiment_data/realq_fixed_label_gradient_trace_20260822_v1/comparison.json`；
SDPA/FA4 两臂分别耗时 `195.48/169.07 s`，均返回 0，432 个 tensor 与 metadata 覆盖门禁
全部通过。两臂使用 node1/GPU0/2 的物理锁，结束后均已释放，期间
`oom=0, oom_kill=0`。

### FA4 是否存在 attention 语义错误的独立 FP32 对照（2026-08-22 16:24 CST）

针对“36 层后 Fisher 差异很大是否意味着 FA4 实际算错”的疑问，在不占用四条正式
YAQA-Q32 量化卡和 16 个评测 fanout reservation 的前提下，使用 node0/GPU3 物理锁补做
单算子审计。形状严格取 Qwen3-4B production geometry：batch=`1`、sequence=`2048`、
Q/KV heads=`32/8`、head dim=`128`、BF16 Q/K/V、causal attention，并以相同 BF16 输入
显式展开 GQA 后用 FP32 `QK^T -> causal mask -> softmax -> PV` 前反向作为独立参考。
正式复跑在进程创建前设置 `NVIDIA_TF32_OVERRIDE=0`、
`CUBLAS_WORKSPACE_CONFIG=:4096:8`，关闭 matmul TF32，并启用严格 deterministic
algorithms；第一次辅助脚本因 GQA repeat 的 `None` 维位置写错而在进入 attention 前
fail-fast，修正后才产生下述有效结果。

| BF16 路径相对显式 FP32 参考 | output rel-L2 | dQ rel-L2 | dK rel-L2 | dV rel-L2 |
|---|---:|---:|---:|---:|
| deterministic math-SDPA | 0.1612% | 0.2347% | 0.2354% | 0.2357% |
| FA4 `pack_gqa=true` | 0.2088% | 0.2978% | 0.2967% | 0.2835% |
| FA4 `pack_gqa=false` | 0.2088% | 0.2978% | 0.2967% | 0.2835% |

FA4 两种 `pack_gqa` 设置的 output/dQ/dK/dV 均逐 bit 相同，排除了当前 32:8 GQA
geometry 下强制 `pack_gqa` 导致 head 映射或 backward 语义变化。此前 production-shape
FA4-vs-SDPA 复测也再次得到 output/dQ/dK/dV rel-L2=
`0.1953%/0.3670%/0.4035%/0.2687%`，FA4 两次调用四个张量均逐 bit 相同。源码配置审计
同时确认 Qwen3-4B 的 `sliding_window=null`、dropout=`0`、无 softcap；adapter 正确传递
`softmax_scale`、`causal=true` 和 `[B,S,H,D]` 布局。因此现有证据**不支持 causal mask、
scale、GQA head 对应或布局写错这类 attention 算法语义 bug**。FA4 相对 FP32 确实比
math-SDPA 略不精确，但差距只有约 `0.05--0.06` 个百分点，量级符合 BF16 fused kernel
的数值舍入差异。

巨大 Fisher 差异不是单 attention 输出直接相差几十倍：固定标签 trace 中 layer-0
block gradient 的 `diff RMS / SDPA RMS` 已到 `6.34x`，且 FA4/SDPA gradient RMS
本身为 `674.54/105.87=6.37x`、cosine 仅 `0.1104`；梯度外积 `g g^T` 对梯度幅值是
二次的，因此 FA4 outer-product 的单样本尺度可以达到约 `6.37^2=40.6x`，与实测固定
标签 Fisher `36.34x` 在量级上相容。但这只是对“Fisher 为何还能继续放大”的解释，
**不是**对 `0.1--0.2%` attention 差异为何经过整网反传变成六倍梯度的证明。逐层 trace
显示 block-gradient 差异从 layer 35 的 `3.31%`、layer 20 的 `8.54%`、layer 16 的
`24.23%`，在 layer 13/12 从 `81.96%` 跳到 `2.22x`，到 layer 5 达 `11.30x` 后在
layer 0 为 `6.34x`；这个陡峭区间仍需用“FA4 forward+SDPA backward / SDPA
forward+FA4 backward”以及真实 QKV+相同上游梯度的局部 replay 区分 forward 状态敏感性
和 FA4 backward 边角问题。更准确的工程结论是：目前未发现 FA4 kernel 的明显语义性
算错，但
REAL-Q Stage-0 Fisher 对合法的 BF16 backend 误差异常敏感，因而 FA4 即使满足普通
forward/backward 容差，也不能自动视为 SDPA Fisher 的等价替代。另有一个已确认的
REAL-Q 工程 bug 是 static/reference cache identity 没有纳入 attention backend/kernel
policy；它没有造成这次隔离实验的数值差异，但会允许未来静默串用不同 backend cache。

### 完整前向终点差异与 run15 缓存串用审计（2026-08-22 17:05 CST）

进一步直接比较旧纯 SDPA 与 run15 纯 FA4 独立生成的 Qwen3-4B reference cache。两份
cache 的 metadata 完全相同，覆盖同一批 Wikitext2 test token；保存张量是 36 层及
final norm 后、lm_head 前的 BF16 hidden states，shape=`[146,2048,2560]`，共
`765,460,480` 个值。全量逐块统计结果为：SDPA/FA4 RMS=`2.844605/2.844627`，
diff RMS=`0.0565292`，即 `diff RMS / SDPA RMS=1.98724%`，cosine=`0.99980316`；
`87.3289%` 的 BF16 元素逐值不同。这个全评测集终点结果比固定标签 trace 的首四条
校准序列抽样 last-block `0.8944%` 更完整，但仍是前向 hidden-state 差异，不是
layer-0 的 `6.34x` 端到端梯度差异。

同时对 run15 正式产物做了逐 receipt 审计：五个 model cache producer 命令均显式
`--attention_backend flash_attention_4`；40/40 个 formal checkpoint 命令也均为 FA4、
均要求 `--require_static_cache_hit true`，且每个 execution log 都记录了恰好一次 cache
hit。Qwen3-4B 的两个 W2 formal run 因 cache identity 变化命中了后生成的
`..._abf27a685b1a_...pt`，其 sampled saliency 与原 FA4 producer
`..._e306d5b89c27_...pt` 逐值相同，Fisher relative RMS 仅
`1.64e-7`、cosine 约 `1`；它相对旧 SDPA `abf27...` 的 Fisher relative RMS 为
`42.66x`，因此不是 SDPA cache。共享 calibration artifact 也已实读为 256 个
shape=`[2048]` 的 `torch.int64` token-ID tensor（SHA256
`21210e1929aa90ea23572e7904f8ebda7b3af75ebf8d9037e8196b3d2c52a399`），不包含任何
attention hidden state。故这批 run15 没有发生“SDPA 前传 cache + FA4 正式量化反传”；
cache key 遗漏 backend 仍需修复为未来防护。

### Qwen3-4B layer-6 真实 QKV+dO 局部因果回放（2026-08-22 19:44 CST）

为区分“FA4 backward 在真实输入上算错”与“合法的小 backend 扰动被整网
Jacobian 放大”，新增独立实验
`experiments/realq_fa4_q4_true_tensor_replay_20260822/`。它在 Qwen3-4B 固定
SDPA labels 的首个 `B=4,S=2048` global-loss batch 上，从 layer 6 attention
backend 边界捕获 post-QK-norm/post-RoPE 的 Q/K/V，以及 attention output
在 reshape/o_proj 前收到的同一个 dO。捕获的 Q/K/V 是 production 的非连续
BHSD view，adapter transpose 后为连续 BSHD；dO 是连续 BSHD。存档只是连续
BHSD value container，每个 replay case 都用
`transpose(1,2).contiguous().transpose(1,2)` 重建 production stride；content SHA
与绑定 shape/dtype/stride/storage-offset/contiguous 状态的 layout SHA 全链路
fail-closed。实际 dO RMS/max=`40.9190/4672`，不是随机小梯度代用。

首次 immutable v1 在 capture 前向的第一层主动 fail-closed：wrapper 误假定
SDPA 收到 `attention_mask=None`，而当前 Transformers 因 packed-sequence detector
禁止 `is_causal` skip，production SDPA 实际传入 `[4,1,2048,2048]` 的 bool
下三角 mask。该次 `68.13 s` 失败未捕获任何 tensor、未 OOM，原始 plan/log/
receipt 完整保留在
`../experiment_data/realq_fa4_q4_true_tensor_replay_20260822_v1/`。fresh v2 严格验证
mask 为 inclusive lower triangle（true count=`8,392,704`），36 层原样传给
math-SDPA；它与 FA4 的 `attention_mask=None, causal=true` 数学语义完全相同。
v2 plan fingerprint=`4ed118476151199c25db3453847783c8a449186f4d1d9f47485ae74ba4114de5`。

v2 用显式 FP32 `QK^T -> causal mask -> softmax -> PV` GQA 作独立参考，对
output/dQ/dK/dV 的全部元素计算指标（无 token/feature sampling）：

| layer-6 真实 B4，相对左侧 | output rel-RMS | dQ rel-RMS | dK rel-RMS | dV rel-RMS |
|---|---:|---:|---:|---:|
| FP32 → math-SDPA | 0.1657% | 0.1649% | 0.1669% | 0.1652% |
| FP32 → FA4 default | 0.1744% | 0.5119% | 0.2944% | 0.2337% |
| **math-SDPA → FA4 default** | **0.1284%** | **0.5335%** | **0.3268%** | **0.2638%** |
| FA4 default → FA4 no-2CTA | 0 | 0.001552% | 0 | 0 |

math-SDPA → FA4 的 dQ/dK/dV RMS ratio 分别为
`0.999930/0.999987/0.999978`，cosine 均不低于 `0.9999858`；因此同一个
真实 QKV+dO 上只有亚百分比级误差，不是旧整网 trace 中 layer-6
q/k/v gradient 的 `3.28x/45.59x/21.42x`。sample 3 单独的
math-SDPA → FA4 output/dQ/dK/dV 也只有
`0.1257%/0.5392%/0.3477%/0.2704%`，幅度比约为 1，与其他三条样本
同量级。这直接排除了“sample 3 在 layer-6 的 FA4 局部 VJP 上触发巨大错误”。

FA4 default 与 `FA_DISABLE_2CTA=1` 分别在 fresh process 中运行。CUDA 12.8
下两者的 causal forward 都是 1CTA；head-dim 128 的 backward 分别是
2CTA/1CTA。两臂 output/dK/dV 逐 bit 相同，dQ 相对 RMS 只差
`1.552e-5`（`0.001552%`，最大 BF16 绝对差 `0.25`），所以 SM100 GQA ratio-4
2CTA backward 不是 6--10 倍幅度的来源。此外，FP32/math/FA4-default/
FA4-no2CTA 四条路径的 `B4 vs 4×B1`、`B4[3,0,1,2] vs 独立 permutation`、
`[sample3]×4 vs 独立 B1-sample3` 在 output/dQ/dK/dV 上均逐 bit 相同，排除
batch-size、batch-slot 或 companion-sample 串扰。

因此当前最强结论是：**FA4 与 math-SDPA 在每层引入的合法 BF16 小扰动
改变了后续 forward state 和上游 dO trajectory，Qwen3-4B 的特定 sample 在
layer 13→12 与 6→5 附近的病态整网 Jacobian 将其放大**；已有证据不支持
cache/backend 串用、loss scaling/hook 错误、GQA/mask/scale 语义错误、通用
FA4 backward 错误或 2CTA/batch-slot bug。不可变 comparison receipt 为
`../experiment_data/realq_fa4_q4_true_tensor_replay_20260822_v2/comparison/comparison/comparison_receipt.json`，
SHA256=`54f5f25a6a9040754ff166c486eac17996d17b41ac58fb280930b17cd978d0f0`；完整
campaign 总耗时约 `227.6 s`，GPU7 已自动释放。下一个最小必要实验是在同层
捕获 FA4-side QKV+dO 再做同样回放，用于排除只在已偏离 FA4 trajectory 上出现
的窄触发问题。

### Qwen3-4B layer-6 FA4-side QKV+dO 因果闭环（2026-08-22 20:03 CST）

上述最后一个窄替代解释已用独立目录
`experiments/realq_fa4_q4_fa4side_tensor_replay_20260822/` 排除。新实验保持模型、
rotation、token、固定 SDPA labels、`B=4,S=2048`、loss reduction 与 scale 全部不变，
但 36 层前反向均走 production FA4，再从同一个 layer-6 backend 边界捕获真实
Q/K/V+dO。随后仍以同一组显式 FP32、math-SDPA、FA4 default 和 FA4 no-2CTA
做全元素局部回放。冻结 plan fingerprint=
`b22c4ce9469172cf3590fe287da2c8229e01753a89aeb3bcda0924a24ff36399`，
node1/GPU7 五阶段均成功，stage 总耗时约 `179.7 s`，完成后 GPU7 和物理锁均释放。

最关键的结果出现在**进入 layer-6 attention backward 之前**。相对上一节 SDPA-side
捕获，FA4-side Q/K/V 仍是同幅的小 forward 扰动，而真实 dO 已经发生巨大、且高度
sample-conditional 的变化：

| SDPA-side → FA4-side layer-6 输入 | diff/SDPA RMS | FA4/SDPA RMS | cosine |
|---|---:|---:|---:|
| Q | 0.6633% | 0.99996 | 0.999978 |
| K | 0.4271% | 0.99999 | 0.999991 |
| V | 0.9131% | 1.00006 | 0.999958 |
| **dO** | **19.129×** | **18.852×** | **-0.25343** |

其中 sample 0/1/2 的 dO 幅度比分别只有 `1.080/1.039/1.029`，而 sample 3 的
SDPA/FA4 dO RMS=`50.946/1541.310`，幅度比 **`30.254×`**、diff/SDPA RMS=
`30.725×`、cosine=`-0.45844`。聚合 dO RMS/max 从 SDPA-side 的
`40.919/4672` 变为 FA4-side 的 `771.388/192512`。因此 layer-6 的巨大梯度绝不是
FA4 layer-6 backward 接到正常 dO 后才凭空产生；爆炸的上游信号在进入该局部 VJP
前已经由更深层的反向 trajectory 形成，且几乎全部由 sample 3 驱动。这与原逐层 trace
在 layer 13→12 首次陡增相吻合。

即使把这份 RMS=`771.388`、max=`192512` 的异常真实 dO 原样交给局部 kernel，
math-SDPA 与 FA4 仍只有通常的 BF16 误差，梯度幅度保持为 1：

| FA4-side 真实 B4，相对左侧 | output rel-RMS | dQ rel-RMS | dK rel-RMS | dV rel-RMS |
|---|---:|---:|---:|---:|
| FP32 → math-SDPA | 0.1658% | 0.1524% | 0.1637% | 0.1661% |
| FP32 → FA4 default | 0.1744% | 0.5237% | 0.2456% | 0.2537% |
| **math-SDPA → FA4 default** | **0.1282%** | **0.5486%** | **0.2834%** | **0.2905%** |

最后一行 dQ/dK/dV 的 FA4/math RMS ratio 分别为
`0.999388/0.999679/0.999740`，cosine 分别为
`0.9999851/0.9999960/0.9999958`，没有任何 6--30 倍的局部幅值放大。default
2CTA backward 与 no-2CTA 的 output/dK/dV 继续逐 bit 相同，dQ 也只差
`0.001044%`。所有 FP32/math/default/no-2CTA 的 `B4 vs 4×B1`、B4 permutation、
sample-3 四次 duplicate 检查均逐 bit 相同；sample 3 在这份 FA4-side 输入上的
局部 FA4-vs-FP32 dQ/dK/dV 也只有 `0.5237%/0.2454%/0.2539%`，与其他样本同量级。

为把“trajectory 变化”和“局部 backend 变化”数值拆开，CPU-only 后处理
`experiments/realq_fa4_q4_fa4side_cross_analysis_20260822.py` 又对两次不可变 replay
outcome 做了全元素交叉配对：

| layer-6 实际 endpoint：SDPA trajectory+math VJP → FA4 trajectory+FA4 VJP | diff/SDPA RMS | FA4/SDPA RMS | cosine |
|---|---:|---:|---:|
| dQ | 17.679× | 17.429× | -0.22326 |
| dK | 16.186× | 15.972× | -0.18407 |
| dV | 11.051× | 10.950× | -0.05509 |

只改变 trajectory、两侧都使用 math VJP 时，对应结果为
`17.690×/16.191×/11.054×`；只改变 trajectory、两侧都使用 FA4 VJP 时为
`17.680×/16.186×/11.051×`，与 production endpoint 几乎重合。反之，在同一
FA4-side trajectory 上切换 math↔FA4 只有上一表的 `0.28--0.55%`。逐样本 production
endpoint 也再次定位到 sample 3：其 dQ/dK/dV 幅度比分别为
`28.974×/28.420×/18.594×`，而 sample 0--2 仅约 `1.02--1.08×`。

因此因果结论已经闭合：**6--10 倍乃至更大的 gradient/Fisher gap 是合法的逐层 BF16
attention forward 小扰动改变深层状态后，被 Qwen3-4B/sample-3 的病态跨层
forward/backward Jacobian 放大；不是 FA4 局部 VJP、2CTA、GQA ratio-4、batch slot、
cache/backend 串用、mask/scale、loss scaling 或 hook/accumulation bug。** 这不意味着
FA4 可以直接用于 REAL-Q Fisher：Stage-0 对 backend 的条件数过差，工程上仍应固定
math-SDPA（或对边界梯度做稳定化）并把 backend 纳入 cache identity。

主 comparison receipt 为
`../experiment_data/realq_fa4_q4_fa4side_tensor_replay_20260822_v1/comparison/comparison/comparison_receipt.json`，
SHA256=`973a430b6e4208d6ae917307817dde4764b5ca692262a94195f5a4600c84062d`；
交叉分解 receipt 为
`../experiment_data/realq_fa4_q4_fa4side_cross_analysis_20260822_v1/receipt.json`，
SHA256=`95f05f0a855dc85c2ac508e10542ebae03fe3c426c361b9972e9b45ba00cfcc3`。
