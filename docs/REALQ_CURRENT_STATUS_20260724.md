# REAL-Q 阶段结论与当前状态

更新时间：2026-07-24 21:16 CST

## 一页结论

1. `main.tex`、外层旧实现和 `realq/` 重构实现的核心数值路径已经完成
   三方审查。修正后，在已经定义并执行的测试矩阵内，旧/新实现不是
   “误差小于 1%”，而是每个 Block-GD refresh loss、组件 loss、样本
   ID、slide alpha、Adam step、最终权重和 checkpoint manifest 全部
   逐 bit 一致。
2. 这不等于“当前代码完全复现论文”。仍有一个明确的高严重度论文/
   代码不一致：当前保留的历史 loss sliding window 与论文现有公式和
   伪代码不同。reverse-cosine 的终点分母也存在论文歧义。论文还没有
   披露若干会影响表格复现的超参数、数据和软件 revision。
3. reverse-cosine 的历史实现不是简单错一位。当前公式是
   `sin(pi*x/2)`，历史实现等价于 `sin²(pi*x/2)`。正式 Qwen3-4B
   W4A16 A/B 中，当前 sine 相对历史 cos² 的 KL 高 131.86%，PPL 高
   3.44%；这个差异很大，但改回历史曲线仍无法达到论文数值。
4. 对 Qwen3-8B W2A4KV4 aware，关闭层调度、使用常数 LR 的当前 aware
   实现最好：相对历史 cos²，KL/PPL 分别下降 5.54%/5.36%。因此当前
   “aware 模式不使用 reverse-cosine”的实现有实验支持。
5. 当前 `realq` 支持 A/K/V fake-quant，也支持让 weight-GPTQ
   aware/unaware 这些 fake-quant 的两类路径；同时支持
   独立的 `a_clip_ratio`、`k_clip_ratio`、`v_clip_ratio`。A/V aware
   由 `act_quant_aware_gptq` 控制，K aware 由
   `k_cache_quant_aware_gptq` 独立控制。代码和论文都没有独立的 Q
   fake-quant、`q_bits`、`q_clip` 或 Q-aware 模式；Q 只有旋转。
6. 四套代码原生数据集已经安装到实验机并完成“全新空缓存 + 完全
   离线”验证：WikiText-2、NeuralMagic calibration、NuminaMath-1.5、
   UltraChat 2k。8 个 payload 均固定 revision 和 SHA256。论文十项
   zero-shot task 的完整数据尚未做离线镜像；打开完整 `lm_eval` 时仍
   可能需要这些 task 的本地 cache。
7. 性能优化目前完成了正确性基线和同步层计时探针，还没有给任何优化
   候选宣布端到端加速。用户当前占用 GPU 0–3，因此隔离 GPU 计时暂停；
   此期间只进行 CPU/static 审查，不会碰用户进程。

## 当前代码与证据边界

| 项目 | 当前值 |
|---|---|
| 主 checkout | `/minimax-avatar-new/zhangqian/realq/gptq_plus` |
| 当前分支 | `zq` |
| 本报告前 HEAD | `6f462b2` |
| 当前 Canoe job | `j-7x9o0je4pk` |
| 任务名 | `zhangqian_debugging_0724_1301` |
| 队列 / 集群 | `minimax-avatar-h800new` / `pa-cne02-prod-01` |
| GPU | 8 × NVIDIA L20C，每卡 183,359 MiB |
| 用户文件保护 | `main.tex` 和 `output/` 保持未跟踪，未被加入提交 |

结论分为三种证据等级：

- **已确认**：有源码审查、独立数学 oracle、旧/新集成矩阵或正式实验
  证据。
- **CPU 已确认、GPU 待确认**：优化候选已通过逐 bit CPU 对抗测试，
  但尚未通过真实 Qwen3 CUDA checkpoint 和隔离计时。
- **未完成**：不能据此作正确性或性能结论。

## 功能逐项审查结论

| 功能 | 当前外层旧实现 | 当前 `realq` | 与论文关系 | 结论 |
|---|---|---|---|---|
| Aggregated Fisher MSE | 修正后使用完整 token 聚合、非对角 Fisher 二次型 | 同值、同梯度语义 | 与论文公式一致 | **已确认**；非对角独立 oracle 通过 |
| Adam 驱动 Block-GD | 每次 refresh 一个带 bias correction 的 Adam step；只更新 trailing columns；每个 linear 重置状态，在该 linear 内跨缩短 suffix 保留 | 同 moments、step、clipping、prefix lock 和状态寿命 | 核心更新一致；论文未披露 clip 阈值 | **已确认**；两步 Adam oracle 和端到端 trace 逐 bit |
| Loss sliding window | 保留历史“只有 trailing update 才计 refresh，倒数第二层不 slide 到最终层” | 与旧实现完全一致 | **与当前论文公式/伪代码不一致** | 依作者要求保留当前实现；这是已知论文 gap |
| Reverse cosine | 当前 checkout 已修为 `sin(pi*x/2)`；保留历史 cos² 仅用于 A/B arm | 相同；低比特 aware 模式用常数 LR | 字面公式一致；最终层单独处理导致分母终点歧义 | **已确认**；不是 off-by-one |
| QuaRot / rotate | tied embedding/head 会先正确 clone/untie，再做全局和局部变换 | 使用共享实现 | 与论文 QuaRot 要求一致 | **已确认**；Llama/Qwen3 整模 logits oracle 通过 |
| A 量化 | per-token linear-input fake quant；aware/unaware 均支持 | 相同 | 论文只给高层 per-token 要求，未规定所有 hook 细节 | **已确认** |
| V 量化 | per-token `v_proj` 输出 fake quant；aware/unaware | 相同 | 高层语义一致 | **已确认** |
| K 量化 | RoPE 后 K-cache fake quant；K aware 有独立开关 | 相同，并拒绝 K-aware + K16 的无效组合 | 论文未披露精确 hook 和独立开关 | **已确认** |
| Q 量化 | 无 | 无 | 论文 `A4KV4` 也未定义独立 Q 量化 | 不应把 A-aware 称为 Q-aware |
| A/K/V clip | 三个独立 ratio，低比特论文 arm 固定 0.9 | 相同 | 与论文 0.9 一致；rounding/range 未完整披露 | **已确认** |
| 激活值 a-loss clip | 历史 `local_backward_chunk` 与显式 `global_refresh` 两种 scope | 同值、同梯度、同 bf16→clip→fp32 顺序 | 论文给 P95，但未给 percentile population | 正式 paper-gap arm 固定历史 local scope |
| Weight per-row | 正确 | 正确 | W4A16 与论文一致 | **已确认** |
| Weight group-128 | 正确，包括 act-order 自然列映射 | 已修复 scale shape、短 tail、act-order 和 rank gather | W2/W3/A4KV4 的 group-128 与论文一致 | **已确认**；测试矩阵逐 bit |
| Weight clip | MSE range search | 默认历史 Cartesian 实现 | 论文未披露搜索细节 | 正确性已锁；加速候选仍默认关闭 |
| 最终层 KL | full-vocabulary FP32 KL | 相同 | 与论文一致 | **已确认** |
| WikiText KL/PPL | full-vocabulary KL、shifted-token PPL | 共享 evaluator | 数学一致；revision/context 等历史信息不足 | **已确认** |
| 十项 zero-shot | task 名称和 metric-key 处理正确 | rank-0 生命周期已恢复 | 任务列表一致 | task discovery 已确认；完整离线 scoring 未完成 |

## 原始实现是否“本来就完全正确”

不是。审查前的原始/共享实现至少存在以下实质问题：

- reverse-cosine 使用了 cos²/sin² 曲线，而不是论文文字公式的 sine；
- Fisher、最终 KL 和 evaluator 曾默认或硬编码 top-20，而不是完整词表；
- aware A/V/K 的 FP teacher/precompute/replay 在重构代码中被 fake quant
  污染；
- tied embedding/head 在 untie 后可能跳过应有的全局旋转；
- GuidedQuant group saliency 使用 channel mean，而论文要求 group
  `sum(grad²)`；
- 重构后的 group-128 scale/column shape、短 tail 和 act-order 映射不完整；
- 旧代码 Cholesky retry 会累积 damping；
- activation-loss clip 曾因 bf16/fp32 运算顺序不同造成 7.0138% 的真实
  旧/新 loss 偏差；
- cache identity、seed domain、checkpoint runtime 配置和多卡 lm-eval
  生命周期不够安全。

这些问题已在当前 checkout 中修复并由 oracle/矩阵覆盖。这里的
“旧/新完全一致”指**修正后的当前外层旧实现与当前 `realq`**，不能
倒推为最初 paper-era 代码没有问题。

## 旧/新一致性验证

最终一卡矩阵：

- 8 个代表 case；
- 每个 case 旧/新各跑两次；
- 32 个 comparison；
- 864 个 refresh comparison point；
- loss、current/next component、sample IDs、slide alpha、Adam step 全部
  逐 bit；
- 每个 comparison 的 30/30 canonical tensor keys 完全一致；
- 最大 weight absolute difference 为 0。

最终八卡矩阵：

- 代表性的 rotate+slide 和 fully-aware A4KV4 两类；
- 8 个 comparison，216 个 refresh point；
- 16 个全局样本 ID、loss/components、权重、manifest 均逐 bit；
- 最大 loss 和 weight difference 均为 0。

生产 evaluator smoke：

| Case | 旧实现 | 新实现 | 差异 |
|---|---:|---:|---:|
| full-vocab KL | 0.003448204603 | 0.003448204603 | 0 |
| shifted PPL | 570.77783203125 | 570.77783203125 | 0 |
| aware A4KV4 KL（final HEAD） | 0.006439953577 | 0.006439953577 | 0 |
| aware A4KV4 PPL（final HEAD） | 583.796020507813 | 583.796020507813 | 0 |

## 作者确认并固定的 paper-gap 参数

| 参数 | 固定值 |
|---|---|
| `grad_clip` | `1.0` |
| `final_layer_grad_clip` | `None`，含义是继承 `grad_clip=1.0`，不是关闭 |
| `percdamp` | `0.01` |
| `act_order` | `true` |
| `w_clip` | `true` |
| `grad_hessian_topk` / `kl_topk` | `-1` / `-1`，完整词表 |
| `a_loss_clip_scope` | `local_backward_chunk` |
| `eval_seq_len` | `2048` |
| calibration / rotation / refresh seed | `1 / 0 / 0` |
| `optimized_rotation_path` | `None` |
| reverse-cosine | 当前实现 |
| loss sliding window | 当前实现 |

性能相关 batch 值固定为：

| 参数 | Qwen3-4B W4A16 | Qwen3-8B W2A4KV4 aware |
|---|---:|---:|
| `global_loss_bsz` | 48（每 rank 12） | 32（每 rank 8） |
| `hessian_accum_bsz` | 每 rank 256 | 每 rank 64 |
| `backward_samples` | 全局 32 | 全局 32 |
| `backward_bsz` | 全局 128 | 全局 128 |
| `final_layer_backward_bsz` | 全局 32 | 全局 32 |
| `lm_eval_batch_size` | 64 | 64 |

Qwen3-4B 每 rank 16 的 Stage-0 probe 达到 182,624 MiB 后又申请
9.27 GiB 而 OOM；每 rank 12 的 probe 以 168,612 MiB 完成。因此 48
不是保守猜测，而是实测后的最高安全全局 batch。

## 正式 A/B 结果

### Qwen3-4B W4A16

| Arm | raw KL | PPL | 相对论文 anchor |
|---|---:|---:|---|
| 当前 literal sine | 0.221625000238 | 14.1338205338 | KL +307.40%，PPL +5.16% |
| 历史 cos² | 0.095586717129 | 13.6634597778 | KL +75.71%，PPL +1.66% |
| 论文 anchor | 0.0544 | 13.44 | — |

当前 sine 相对历史 cos²：

- KL：`+131.8575%`；
- PPL：`+3.4425%`。

结论：reverse-cosine 公式差异对该轨迹的 KL 影响巨大，不是“指标错一
位”；但仅恢复历史公式仍无法关闭论文 gap。

### Qwen3-8B W2A4KV4 aware

| Arm | raw KL | PPL |
|---|---:|---:|
| 当前 constant-aware | 0.919793128967 | 20.1894054413 |
| literal sine aware | 0.944964349270 | 20.7609272003 |
| 历史 cos² aware | 0.973785102367 | 21.3331642151 |
| 论文 anchor | 0.767 | 19.68 |

当前 constant-aware 相对历史 cos²：

- KL：`-5.5445%`；
- PPL：`-5.3614%`。

最好 arm 仍比论文 KL 高 19.92%、PPL 高 2.59%。因此 schedule 解释了
部分差异，但不是全部论文 gap。

五个正式 artifact 均通过主 validator；独立 validator 完成 782 个检查、
0 failure，并重新计算约 9.4 GB 关键 cache 的 SHA256。

## Column-block backward loss / 学习率诊断

为直接检查论文指标差异是否来自学习率轨迹，`realq` 新增默认关闭的
日志开关：

```bash
--log_column_block_loss true
```

开启后，只有全局 rank 0 会在每次 Block-GD refresh 后输出一行
`[realq.column_block_loss]`。其中：

- `loss` 是该 column block 的所有 `backward_samples` 在所有 DP rank
  上按样本数加权后的均值，对应实际参与梯度累积的 backward objective；
- 普通层的 `objective=fisher_mse`；最终层为 `objective=kl`；
- loss sliding window 开启时，`loss` 是实际反传的 blended loss，并同时
  输出 `loss_current`、`loss_next` 和 `slide_alpha`；
- 同一行输出 `layer`、`module`、`block`、`columns=[start,end)`、
  `adam_step`、层级 `lr`、Adam bias-correction 后的
  `adam_step_size=lr/(1-beta1^step)` 和 `global_samples`；
- `column_space=quant_order` 表明这个区间是 GPTQ 的量化顺序位置；
  `act_order=true` 时不能把它误读成原始权重的自然列号；
- 最后一个 column block 后面没有 trailing columns，算法不会为它执行
  refresh/backward，因此它没有可打印的 backward loss。这不是漏日志。

示意格式：

```text
[realq.column_block_loss] layer=3 module=mlp.down_proj block=0 columns=[0,128) column_space=quant_order adam_step=1 objective=fisher_mse loss=... loss_current=... loss_next=none slide_alpha=none lr=... adam_step_size=... global_samples=32
```

日志专用 loss 使用独立 all-reduce，不扩大或改写优化器原有的
`[gradient | sample_count]` collective buffer；默认 `false` 时不收集
loss、不增加 collective，也不发生 `.item()` 同步。Fisher、最终层 KL、
配置解析、DP 空 shard、trace/slide、日志开关前后 Adam/update 逐 bit
行为的针对性测试为 `14 passed`；当前完整 CPU suite 为
`203 passed, 5 skipped`。验证显式设置
`CUDA_VISIBLE_DEVICES=''`，没有占用实验 GPU。

## Multi-GPU refresh sampling 与 group-parallel 审查

### `dp_global_shuffle`

这个开关只控制 Block-GD/pre-GD 的 refresh backward 如何从已经生成的
校准集选择样本；它不控制 Stage-0 static Fisher/saliency 的数据成员，
也不控制 GPTQ Hessian 是否按 output row 并行。

设全局校准样本数为 `N`、一次 Adam refresh 的全局样本数为 `B`、DP
world size 为 `W`：

| 实现/模式 | scheduler 范围与 chunk | 每 rank 行为 | seed |
|---|---|---|---|
| 旧代码 `dp_global_shuffle=true` | 一个全局 `[0,N)` scheduler，每次先取 `B` 个 global IDs | 每个 rank 得到相同 IDs，再按自己连续的 `N/W` shard 过滤；local count 可为 0，所有 rank 合计仍为 `B` | 所有 rank 都是 `refresh_seed` |
| 旧代码默认 `false` | 每个 rank 一个本地 `[0,N/W)` scheduler，每次取 `B/W` 个 local IDs | 每个 rank 始终等量处理 `B/W` 个自己 shard 内的样本；不是每 rank 各取 `B` 个 | `refresh_seed + rank` |
| 新 `realq` | 只实现了第一行的全局模式 | Config/CLI 没有关闭开关，传 `--dp_global_shuffle` 会是未知参数 | 所有 rank 都是 `refresh_seed` |

“旧代码默认 false”指裸 `process_args.py` CLI；旧实验常用的
`gptq_plus_lr_sweep.sh` 把 `DP_GLOBAL_SHUFFLE` 默认设为 `1`，README
也要求固定为 1。因此当前新默认对应的是旧 intended launcher，而不是
旧裸 CLI。

用户对 true 模式的理解基本正确：先产生一组全局 sample IDs，再由各
rank 按校准样本所属 shard 过滤并贡献 partial gradient/count。这里分配
的是 **refresh backward 计算**，不是把不同 weight columns 分给 rank。

还有一个容易被名称掩盖的重要事实：旧/新 scheduler 的第一轮不是
shuffle。它以 `range(N)` 开始，只有完整耗尽一轮后才原地 shuffle。
例如正式 Qwen3-4B 的 `N=2048, B=32, W=4`：

- 第一轮共 64 个 refresh；
- 前 16 个 refresh 的 32 个样本全部在 rank 0，其他三张卡为空；
- 接下来 16 个全部在 rank 1，然后依次 rank 2、rank 3；
- 第二轮以后才是全局乱序，但每个 chunk 的 rank 负载仍不保证相等。

因此 true 模式可能造成严重 straggler/空卡，不只是“轻微不平衡”；
false 模式每次严格平衡 `B/W`。但 false 不是纯性能开关：多卡下两种
模式在每个 Adam step 使用的 sample membership/order 不同，Adam 更新
又不可交换，所以会改变 refresh loss、梯度轨迹、最终量化权重和指标。
在 local backward cap 更小、同一 rank 还需拆多个 backward chunks 的
其他配置中，样本分块还可能改变 local-chunk P95 clip；当前正式
`B=32`、非 final local cap 为 32。若两种模式恰好选中同一全局集合，
各 rank 都是 8 个样本且只形成一个 chunk，因此当前配置不靠这一效应
区分两种模式。world size 为 1 时两种构造才退化为同一 scheduler。

此前旧/新 alignment runner 在旧端**显式传了**
`--dp_global_shuffle`，新端使用固定全局模式；8 卡代表 case 的
sample IDs/loss/weights 逐 bit 一致。因此旧/新比较没有混用两种采样
策略。正式 schedule A/B 全部走新代码，也都固定为全局模式。论文没有
披露该选择；它是值得单独做复现 A/B 的 protocol 参数，但需要先给
`realq` 增加一个默认保持当前 `true` 的显式开关，不能把 false arm
当作不改数学的性能优化。

证据边界也要说明：既有 alignment 的一卡 case 在 `W=1` 时两模式天然
相同；八卡代表 case 使用 `N=B=16`，每次已经取完整校准集。因此它们
证明的是“旧 true 路径与新固定 true 路径一致”，没有实测
`W>1, B<N` 时 true/false 对正式 Qwen3 指标的影响。

### `group_parallel_quant=rank`

默认值必须区分入口：

| 入口 | 默认值 |
|---|---|
| 旧 Python CLI `process_args.py` | `none` |
| 旧实验常用 `scripts/gptq_plus_lr_sweep.sh` | `rank` |
| 新顶层 `realq.Config` / production runner | `rank` |
| 新低层 `RealQLayer(...)` / `quantize(...)` API | `none`，但顶层 runner 会显式传 Config 的 `rank` |

所以，如果“旧代码默认”指裸 Python CLI，答案是否；如果指论文实验常用
sweep launcher，答案是。新 production pipeline 把旧 launcher 的
`rank` 选择提升成了真实 Config 默认。

新 `realq` 的 `rank` 实现把每个 linear 的连续 output rows 分给 DP
ranks：

- 每个 rank 只对自己的 rows 做 per-row weight qparam search、inner
  column loop 和 outer GPTQ compensation；
- `num_groups>1` 时，每个 calibration batch 的 group Hessian 用
  reduce-scatter 汇总，owner 只常驻自己 rows 需要的 groups；
- Block-GD 前重建完整 stitched weight，所有 rank 计算同一全局
  refresh gradient/Adam state，随后只把 update 的 owned rows 应用到
  本地 working weight；
- 最终 all-gather 完整 Q，因此保存的不是 sharded checkpoint；
- 这个开关不改变 refresh scheduler、sample IDs 或 LR。

在精确实数和相同 Hessian/qparams 下，各 output row 的 GPTQ 与逐元素
Adam 更新相互独立，所以 `rank` 与 `none` 是同一数学算法。但它不是
通用的逐 bit no-op：

- `num_groups>1` 多卡 rank 使用“每 batch、每 group reduce-scatter”，
  none 使用“先各 rank 本地累加、最后全 H all-reduce”，浮点加法树不同；
- rank 使用 batched Cholesky/不同张量布局和 CUDA kernel；
- group-quant + `act_order=false` 会在前序更新后的 block weight 上重新
  观察动态 scale，前面的 ulp 差异可进入最终 dequantized weight。

CPU toy 对抗验证了这个边界。固定相同非对角 SPD Hessian，world size 1：

- 论文协议形态 `act_order=true, num_groups=4` 下，缩小尺寸的 per-row
  和 group toy 两组各 50 个 seed 的 rank/none 输出均逐 bit 相同；
- dynamic group、`act_order=false, num_groups=1` 下，50 个 seed 有
  34 个输出不逐 bit，115 个元素不同，最大绝对差
  `2.384185791015625e-7`。

这个反例没有改变目标函数或量化公式，但证明不能把 rank/none 宣称为
所有配置下的文件/权重逐 bit 等价。当前论文复现实验固定
`act_order=true`，不落入该 dynamic-group 反例。更强的现有证据是：
旧/新 alignment 的一卡 8 cases 和八卡代表 per-row/group-128 aware
case 两端都显式使用 rank，最终 loss/state/weights 逐 bit一致；这证明
“旧 rank vs 新 rank”的受测路径正确，不证明“rank vs none”在真实
Qwen3/CUDA 上逐 bit。

结论：保留 `rank` 作为当前默认是合理的性能实现，但任何性能 A/B 必须
把它固定在两边。若要切到 `none`，应归类为“实数算法相同、浮点轨迹
可能漂移”的数值 arm，重新比较 checkpoint 和指标，不能当成无损开关。
另有一个 provenance 缺口：checkpoint 当前没有记录
`group_parallel_quant`；这不影响加载后的推理行为，但不足以单凭
checkpoint manifest 还原量化过程。

## 与论文用时表格的现有比较

论文 Qwen3-4B W4A16 在 4 × RTX Pro 6000 上报告 72 分钟。当前 L20C
节点上最接近的 Stage-0 core + Stage-1 范围为：

| Arm | Stage 0 | Stage 1 | 合计 | 相对 72 分钟 |
|---|---:|---:|---:|---:|
| literal sine | 2:06 | 38:37 | 40:43 | 1.768×；少 43.45% |
| 历史 cos² | 2:06 | 38:59 | 41:05 | 1.753×；少 42.94% |

按更宽的 launch-to-complete artifact scope，速度约为论文的
1.55×。因此可辩护的范围是 **1.6×–1.8×**，但不能归因于重构代码：
GPU、CUDA/PyTorch、文件系统、batch 和历史 runner 都不同。

## 性能优化当前进度

### 已完成

1. 建立不允许改变数学、默认安全、分层正确性 gate 的性能协议。
2. 完成 Qwen3-4B layer-0 的四组 pre-optimization correctness baseline：

   | Case | 状态 | sampled peak |
   |---|---|---|
   | group-128 | PASS | 17,556 / 19,132 / 19,362 / 19,362 MiB |
   | 同卡 group-128 A/A repeat | PASS | 17,556 / 17,980 / 18,044 / 17,618 MiB |
   | per-row | PASS | 11,838 / 11,520 / 11,776 / 11,776 MiB |
   | A4K4V4 aware group-128 | PASS | 18,348 / 19,170 / 19,746 / 19,426 MiB |

3. group-128 A/A 比较了 435 个 tensor keys、399 个 weight keys：
   canonical state SHA256 完全相同，缺失/额外/mismatch 为 0，
   `max_abs_diff=0`。
4. 合入默认关闭的同步 layer timing probe；开始边界为
   `CUDA sync → distributed barrier → CUDA sync`，结束边界再次 sync。
   默认 `perf_measure_layer=None` 不增加 barrier、CUDA query 或文件写入。
5. 当前包含计时探针的完整 CPU suite 在实验容器为
   `202 passed`。
6. 隔离计时 harness 已合入：强制一组 warm-up + 至少三次串行重复，
   以四个 rank 的最大同步 layer elapsed 为 primary metric；严格校验
   physical GPU ID→UUID 顺序、config diff、source/cache hash、无 eval、
   无 checkpoint。`--prepare-only` 实机 dry-run 通过且确认没有生成
   `run.log` 或启动 torchrun/GPU 计算。

上述 correctness wave 是并行执行的，因此它的 process wall 不能用于
加速声明；可靠性能数字必须来自单独 warm-up + 至少 3 次串行重复。

### 优化候选

| 候选 | 当前状态 | 初步证据 | 尚缺 |
|---|---|---|---|
| P01 `quantizer_inner_fastpath` | 实现已完成，默认 `false`；独立审查进行中 | 独立 94/94 CPU targeted tests 通过，默认数值路径 no-op；原作者 CPU primitive：per-row 1.30×，grouped 84.26× | 更多 adversarial、CUDA checkpoint、隔离计时 |
| P02 `w_clip_search_impl=symmetric_union_exact` | 实现已完成，默认历史 Cartesian；尚未合入主线 | 每个 equal-width observer batch 的 625 个 QDQ/error candidate 降至最多 50；短 tail 是两个 batch，即 1250→100；候选 80 个相关测试和独立 98 个 CPU 对抗项均逐 bit 通过；CPU primitive 约 11.4×–12.5× | CUDA kernel 逐 bit、真实 checkpoint、峰值显存、隔离计时 |

P02 可能临时保留 `O(2M × rows × groups)` 的 errors/scales/keys；对大
linear 可能增加数百 MiB 到数 GiB，具体取决于层型。它在 GPU 显存
实测前不会被默认打开。

P01 默认关闭时模型 tensor 数值路径不变，但新增的 provenance 字段会
改变 checkpoint manifest/raw archive bytes；因此只能称“数值 no-op”，
不能称整个文件 artifact 逐字节 no-op。开启后还需防范通过 `.data` 或
底层 storage 绕过 Tensor `_version` 导致 prepared scale stale 的边界，
独立审查尚未结束。

### 当前暂停点

用户从 2026-07-24 20:35 CST 起临时占用 GPU 0–3。为避免干扰用户，也
因为隔离计时要求整机没有并发 GPU 负载：

- 没有启动 baseline timing；
- 没有在 GPU 4–7 上偷跑会污染计时或争用主机资源的任务；
- 只继续 CPU/static review、脚本 dry-run 和文档。

用户释放 0–3 后，先检查 8 卡 idle，再运行：

1. group-stress legacy warm-up + 3 次；
2. 相邻 candidate arm warm-up + 3 次；
3. checkpoint exact gate；
4. benefit 超过 A/A noise 后才做 Nsight Systems；
5. row 和 aware 代表 case 回归。

第一组可靠 isolated baseline 在 GPU 释放后预计 20–30 分钟；第一组
候选 A/B（含 exact checkpoint gate）预计再需 30–50 分钟。

## 本地数据集交付

代码原生四套数据已安装：

```text
datasets/wikitext
datasets/LLM_compression_calibration
datasets/NuminaMath-1.5
datasets/ultrachat_2k
```

实际 payload 位于：

```text
/minimax-avatar-new/zhangqian/realq/dataset_mirror_staging/
```

验证使用全新的 `/tmp` cache，并同时设置
`HF_DATASETS_OFFLINE=1`、`HF_HUB_OFFLINE=1`：

- cache 开始为 0 文件；
- 本地 payload 构建出 27 个 cache 文件、1,206,999,080 bytes；
- 4 个精确 `load_dataset(...)` 调用通过；
- 4 个 `utils/data_utils.py` wrapper 通过；
- temporary cache 已删除；
- 8 个 payload 的 bytes/SHA256 与固定 Hub revision/LFS OID 一致。

完整 revision、SHA、schema、安装方式和复验命令见
`docs/REALQ_DATASET_MIRROR.md`。

## 仍需解决或明确保留的事项

1. **论文 loss-slide 文本**：要么修改论文为历史实现，要么将代码改成
   论文公式；当前依作者指示保留代码，不能称其与论文一致。
2. **reverse-cosine 分母**：论文需明确 schedule 是跨全部 `L` 层还是
   仅跨 `L-1` 个非最终层。
3. **历史复现信息缺失**：精确 model/tokenizer/dataset/lm-eval revision、
   clip percentile 轴、若干 batch/seed/评估设置没有在论文中披露。
4. **完整 zero-shot 离线数据**：task 注册和生命周期已验证，但十套
   task 数据还没有全部镜像和跑分。
5. **性能候选**：没有通过 CUDA checkpoint + isolated timing 之前，
   CPU primitive 加速不能写成 REAL-Q 端到端加速。
6. **非确定性**：安装的 attention backward kernel 会发出 nondeterministic
   warning；正式 A/B 是单 trajectory，百分比是 effect size，不是统计
   显著性。

## 关键文档与 artifact

- 三方审查与验证总记录：`docs/REALQ_ALIGNMENT_WORKLOG.md`
- 论文披露/未披露设置：`docs/REALQ_PAPER_PROTOCOL.md`
- 正式 A/B 与论文用时比较：`docs/REALQ_AB_PERF_RESULTS.md`
- 固定实验协议：`docs/REALQ_SCHEDULE_ABLATION_MANIFEST.md`
- 性能优化流水账：`docs/REALQ_PERFORMANCE_WORKLOG.md`
- 本地数据镜像：`docs/REALQ_DATASET_MIRROR.md`
- 正式 A/B artifact：
  `/minimax-avatar-new/zhangqian/realq/experiment_data/schedule_ablation_localclip_formal_20260724`
- 性能 correctness baseline：
  `/minimax-avatar-new/zhangqian/realq/experiment_data/perf_stage1_debugging_zhangqian/j-7x9o0je4pk_20260724T121554Z_cb5c469794a4`

## 当前总判定

- **重构正确性**：在已执行的数值范围和代表 case 内，当前 `realq` 与
  修正后的外层实现逐 bit 一致，证据强。
- **论文一致性**：核心 Fisher、Adam、量化、rotate、KL/PPL 已对齐；
  loss-slide 明确不一致，reverse-cosine 仍有论文终点歧义，历史表格
  仍缺复现信息。
- **正式指标**：当前 schedule 修改的影响可以非常显著，但方向依场景；
  Q4 历史 cos² 更好，Q8 aware 的当前 constant LR 更好。
- **性能结论**：当前仅能报告正式实验相对论文硬件表的 1.6×–1.8×
  wall-time 区间；新的代码级优化尚无合格端到端 speedup 数字。
- **当前动作**：GPU 被用户占用期间只做 CPU/static 工作；释放后继续
  严格的 baseline → candidate → exact gate → Nsight 循环。
