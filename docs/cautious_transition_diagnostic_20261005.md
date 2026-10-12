# REAL-Q：Cautious 更新与下一次 GPTQ 的拆分诊断

## 目的与范围

此前 Grams 方向替换和完整 Cautious 在小型前缀筛查中没有稳定收益。本轮不增加优化器种类、不扫学习率，而是分离掩码与缩放，定位收益或损失发生在浮点更新阶段还是下一次离散量化阶段。

这是诊断实验。所有干预只发生在一个预先指定的 refresh；使用真实量化轨迹和真实下一块 GPTQ，不用影子近似。未将候选设为生产默认，未覆盖最终 transformer block 的独立 KL refresh 路径。

## 方法

令 d 为原 Adam 返回的实际减法更新量（已包含学习率），g 为该次梯度经过原有裁剪后的浮点后缀梯度。M=1[d*g>0]，alpha=N/(sum(M)+1)，其中 N 为当前浮点后缀坐标数。

| 实验臂 | 写入权重的更新 | 保留状态 |
|---|---|---|
| adam | W <- W-d | 原始 m/v/步数 |
| cautious_mask | W <- W-M*d | 同上 |
| cautious_scale | W <- W-alpha*d | 同上 |
| cautious | W <- W-alpha*M*d | 同上 |

正乘积掩码沿用 Cautious 论文算法 1 的严格 `>0` 写法，有限计数归一化沿用算法 2 的 N/(nnz+1)。论文算法 2 的掩码行写作 `>=0`，与算法 1 对零乘积的处理不同；本实现明确采用 `>0`，并记录零梯度比例。REAL-Q 原更新无额外权重衰减，本诊断亦不添加。

借鉴来源：Liang et al., *Cautious Optimizers: Improving Training with One Line of Code*, [arXiv:2411.16085v1](https://arxiv.org/html/2411.16085v1)。该方法通过当前梯度与更新的一致性筛选坐标；其充分小步长下的当前目标下降分析不保证 held-out KL，更不保证 GPTQ 舍入之后的质量。

三个记录时点：

1. 当前 column block 已完成 GPTQ 舍入和闭式补偿，即将计算 Adam refresh。
2. Adam 返回的更新实际写入 FP32 工作权重之后。
3. 下一个 column block 在原量化循环中完成舍入与后缀闭式补偿之后、下一次 refresh 之前。

观测器只保存张量，不在量化进程里额外前向/反向。离线独立进程重放同一代理损失，并使用真实全模型下游前向计算 KL(teacher || student)。此处“全模型下游”不表示已经量化完全部层，后续未量化层仍是原 FP 权重。

## 配置与预先指定的检查点

- Qwen3-0.6B，本地 RTX 4060 8GB，WSL 虚拟环境。
- W4、每输出行量化 scale；Fisher/Hessian 行分组 num_groups=4，column block=128，act_order=true，rotate=false。
- 单卡 group_parallel_quant=none；校准 train 8 条 x64 tokens，backward_samples=8，backward_bsz=1。
- seed 42、43；refresh_seed=0；固定原学习率、裁剪、bias correction、矩更新与 slide mixture。
- 检查点：layer 0 / self_attn.o_proj / block 1；layer 3 / mlp.down_proj / block 1、block 16。block 从 0 起数，对应模块内 Adam step 2、2、17。
- 每个检查点四臂，共 24 次捕获和独立离线评估。
- 评估使用 WikiText-2 validation 前 16 个不重叠窗口，每个 128 tokens；与校准 train 分离。评估样本少且来自连续文本，不能据此作总体统计显著性或正式论文性能结论。
- 仅空干预核对继续量化前 5 层并运行原 test 评估，其余捕获在第三时点主动正常退出。

## 正确性检查

- 开启观测、保留原始 Adam 并完成前缀量化，最终原 test KL=0.098157130181789398，PPL=58.043598175048828，与此前无观测的结果完全一致。
- 离线重算目标时点的代理损失必须与线上一致（容差 rtol=1e-6、atol=1e-9）。空干预实测差值为 0。
- FP 教师与同一 FP 学生的自比较 KL 为 0。
- 候选与 Adam 比较之前，检查完整 before 权重、g、m、v、step、Hinv、perm、scale、zero、已量化前缀、prefix/next 层 state、Fisher、reference、校准样本与评估 tokens。
- 用保存的 FP32 before 权重和原始 d/g 独立重建四种公式，要求实际 after 权重与被测公式逐元素相等。
- 三时点检查冻结前缀不变、权重有限；实验前后检查关键源码哈希不变。
- 实际下一块 Q 的候选/Adam 差异比例单独保存，用于识别离散舍入差异。

## 诊断指标的更正

旧日志名 `conflict_update_mass_fraction` 实际计算 sum(|g*d| on conflict)/sum(|g*d|)，应解释为冲突坐标的绝对一阶贡献占比，不能解释为更新 L1/L2 占比。新分析同时输出正确命名的该指标、更新 L1 占比和更新平方 L2 占比。

代理损失变好不等于真实 KL 变好；真实 KL 在当前浮点状态变好，也不等于下一次 GPTQ 后仍变好。所有比较必须同时给出对应时点及基准。

## 复现与文件

```bash
python -u tools/run_realq_cautious_transition.py \
  --model /mnt/d/llamaModels/Qwen3-0.6B \
  --tag cautious-factorial-20261005
python tools/analyze_cautious_transition.py \
  outputs/realq-adam-transition-cautious-factorial-20261005
```

- 捕获：`realq/refresh/adam_transition_probe.py`，挂接点为 `realq/quant/realq_layer.py` 和 `realq/runner/layer_loop.py`。
- 四种单点干预：`realq/refresh/block_gd.py`，由诊断环境变量显式启用。
- 离线评估：`tools/evaluate_adam_transition.py`。
- 编排与同状态审计：`tools/run_realq_cautious_transition.py`。
- 分析：`tools/analyze_cautious_transition.py`。
- 原始结果：`outputs/realq-adam-transition-cautious-factorial-20261005/`；每臂保存 run.log、evaluation.log、status.json 和 transition 张量。

## 完成结果

24/24 次捕获与离线评估成功；18/18 候选配对通过同状态审计；24 次 FP 自比较 KL 和代理损失重放差值均为 0；源码哈希未变化。离线评估 peak allocated=1,810,917,888 bytes，此数仅是诊断评估峰值，不是量化全程资源结果。

下表为完整 Cautious 相对同状态 Adam 的真实验证 KL 百分比变化，负数表示候选更好：

| 检查点 | seed | Adam 写入后 | 下一块 GPTQ 后 |
|---|---:|---:|---:|
| layer0 o_proj block1 | 42 | +1.5167% | -0.2003% |
| layer0 o_proj block1 | 43 | +0.9238% | +0.0392% |
| layer3 down_proj block1 | 42 | +0.2153% | -0.3239% |
| layer3 down_proj block1 | 43 | -0.4096% | -0.3382% |
| layer3 down_proj block16 | 42 | -0.2097% | +0.6268% |
| layer3 down_proj block16 | 43 | -0.2104% | +0.2664% |

所有候选在更新后的代理损失都比 Adam 更低，但对应 KL 并非都更低。完整 Cautious 在 FFN 早期点两个种子的下一块 KL 均改善约 0.33%，每个种子 16 个窗口中 11 个改善；这只是小型局部线索。后期点两个种子均在 Adam 写入后改善、下一块量化后恶化，是本轮最明确的重复现象。

只加 mask：6 个点中 4 个在 Adam 写入后 KL 更好，但下一块之后仅 1 个更好。只加 scale：下一块之后 2/6 更好，且都来自 layer0；不能把局部好转归因于冲突坐标筛选。完整 Cautious 的下一块结果为 3/6 更好。并无可直接推广到所有 refresh 的方案。

### 冲突规模

六个点的冲突坐标比例约 2.32%–5.26%，但其绝对一阶贡献占比仅 0.0196%–0.1813%；更新 L1 占比约 0.94%–2.26%，更新平方 L2 占比约 0.42%–1.16%。掩码后的缩放系数约 1.024–1.056，影响的是所有保留坐标。因此不能用坐标冲突率说明历史动量是主要损失来源。

这些点的梯度裁剪均未触发（raw/clipped 相对差为 0），排除了此次比较中梯度裁剪导致符号或尺度差异的解释。

### 不应把后续变化全部归咎于格点翻转

下一块权重的变化比例随候选为 0%–0.0809%。尤其 seed43/layer0/mask 的下一块量化权重与 Adam 完全相同，但该时点 KL 仍更差；后缀仍保留不同浮点更新，量化上下文也已变化。故此处只能确认“收益经过实际量化/补偿转换后改变”，不能断言全部来自下一块整数值翻转。

### 决策

不将完整 Cautious、mask-only 或 scale-only 设为默认，也不因一个检查点正向就开展全模型方法筛选。保留原 Adam。下一步诊断是读取这些已保存状态，比较真实下游 KL 梯度与当前代理梯度的方向，并区分 FP32 更新量和实际 BF16 前向权重变化；它不改变生产路径的反传次数。

还不能排除短校准/短上下文、FP16/BF16 有限精度、验证样本有限带来的影响。不能宣称否定了 Cautious 文献，也不能把本轮结果作为正式最终量化模型质量结论。

## 补充：真实下游 KL 梯度诊断（已完成）

使用 `tools/probe_transition_kl_gradient.py` 在六个 before 状态、同一组 validation 窗口上求真实完整下游 KL 的自动微分梯度，其他参数冻结。每窗口独立求导，CPU float64 累加；模型及叶权重仍为 BF16，不能把结果称为无限精度梯度。

六处 validation 前向 KL 与主实验重放的最大逐窗口差为 2.13e-8。该差异来自有梯度/无梯度计算和归约路径。诊断峰值 allocated 约 2.21GiB，没有 OOM。

| 检查点 | seed | 校准代理梯度与验证 KL 梯度 cosine |
|---|---:|---:|
| layer0 o_proj block1 | 42 | -0.0000183 |
| layer0 o_proj block1 | 43 | +0.0004419 |
| layer3 down_proj block1 | 42 | +0.0017484 |
| layer3 down_proj block1 | 43 | -0.0002921 |
| layer3 down_proj block16 | 42 | -0.0006723 |
| layer3 down_proj block16 | 43 | -0.0030859 |

在相同 refresh 校准样本索引上，额外从真实 KL 梯度的同一程序计算 Fisher 代理梯度，两边都对相同 active suffix 比较。cosine 分别为：

| 检查点 | seed | 同校准样本的代理梯度 / 真实下游 KL 梯度 cosine |
|---|---:|---:|
| layer0 o_proj block1 | 42 | 0.03140 |
| layer0 o_proj block1 | 43 | 0.04863 |
| layer3 down_proj block1 | 42 | 0.00089 |
| layer3 down_proj block1 | 43 | 0.00045 |
| layer3 down_proj block16 | 42 | 0.00173 |
| layer3 down_proj block16 | 43 | 0.00057 |

这比跨数据集比较公平，仍显示当前短配置下两种梯度的方向对齐很弱。真实 KL 梯度的参考分布由保存的 FP layer 输出经过未量化后缀和完整词表头得到；学生则从实际量化轨迹的 refresh 输入开始。因此它对应 REAL-Q 的成对 teacher/student 目标，但还不是最终全量化模型 KL 对参数的梯度，也不能外推到完整校准集或其他模型。不能据此推断“原论文 Fisher 普遍错误”或“全部失配来自非线性”。

进一步分别用 FP32 实际工作权重差、BF16 实际前向权重差与真实 KL 梯度做内积。二者的一阶预测都不能稳定预测本轮实际 KL 变化的符号。例如 seed43/layer3/down_proj/block1 的原 Adam：FP32 差预测 -8.30e-5，BF16 差预测 -2.64e-5，实测为 +7.82e-4。仅修正为 BF16 权重差仍未消除偏差，可能涉及下游有限精度、有限位移的高阶响应等；本实验尚未分离原因。

### 目标线性层的精度对照

`tools/compare_transition_precision.py` 先要求 BF16 分支的 before/update/next-quant KL 与主实验逐项重现，容差 1e-7；六个状态全部通过。随后只把目标 Linear 的权重和矩阵乘改为 FP32，再将该 Linear 输出转回 BF16，后续 Transformer、LayerNorm 与 LM head 仍是 BF16。比较相同 16 个验证窗口的阶段 KL 变化：

| 检查点 | seed | BF16 更新后 ΔKL | FP32 目标 Linear 更新后 ΔKL | BF16 下一块后 ΔKL | FP32 目标 Linear 下一块后 ΔKL |
|---|---:|---:|---:|---:|---:|
| layer0 o_proj block1 | 42 | +1.12e-5 | -8.06e-5 | +1.681e-3 | +1.305e-3 |
| layer0 o_proj block1 | 43 | -2.874e-4 | -6.82e-5 | +6.170e-4 | +9.572e-4 |
| layer3 down_proj block1 | 42 | -4.95e-6 | +1.218e-4 | -6.510e-4 | -1.103e-3 |
| layer3 down_proj block1 | 43 | +7.816e-4 | +1.306e-5 | -8.387e-4 | -1.084e-3 |
| layer3 down_proj block16 | 42 | -1.616e-4 | +4.525e-4 | -3.378e-4 | +5.906e-4 |
| layer3 down_proj block16 | 43 | -3.250e-4 | +1.714e-4 | +5.35e-5 | +2.648e-4 |

更新后 6 处有 4 处 BF16/FP32 预测符号不同，下一块量化后符号均相同。FP32 对照只提高目标 Linear 的权重/乘加精度，输出仍转 BF16；它不是全模型 FP32 结果。结果说明目标 Linear 的精度足以影响这几个小更新的即时 KL 判断，但不能单独证明生产 Adam 更新有 bug：REAL-Q 反传本身也使用 BF16 override 权重，且目标线性层的 FP32 master 会在 GPTQ 路径中继续量化。当前的合理结论是有限精度是一个实质混杂因素，需要进一步做双精度/有限差分检查，不能把前述 KL 梯度的一阶预测当作可靠判据。

在六处中四处，被代理梯度认为“冲突”的原 Adam 分量对验证 KL 的局部一阶贡献反而有利；另两处有害。这进一步说明，用代理梯度作硬掩码没有通用验证 KL 保证，但它也受上述不同数据/精度限制。

补充原始结果保存于 campaign 的 `true_kl_gradient_analysis.json` 和 `precision_comparison.json`；六份真实 KL 梯度保存在对应 Adam 臂的 transition 目录。该诊断不进入正式量化流程，不改变生产反传次数。

## 补充：真实 Adam 步的有限差分核对（已完成）

在六个存档 before 状态上，沿实际 Adam 减法方向设连续系数 \(t\)，比较目标 Transformer block、后续层、最终 norm 与 LM head 的平滑 FP32 执行参考 \(f_{32}(t)\)，与原生 BF16 前向 \(f_{16}(t)\)。目标线性层使用 FP32 master；目标 block 其余参数、未量化后缀和保存的输入/teacher activation 是从加载的 BF16 状态提升到 FP32 执行，并非原始 FP32 checkpoint。用 autograd 得到 \(f'_{32}(0)\)，再以中心差分核对；此外分别记录实际整步 \(f(1)-f(0)\)。样本索引与主诊断使用的验证窗口一致。

| 检查点 | seed | autograd \(f'_{32}(0)\) | 中心差分 h=0.5 相对误差 | FP32 整步 ΔKL | BF16 整步 ΔKL | BF16/FP32 整步绝对值比 |
|---|---:|---:|---:|---:|---:|---:|
| layer0 o_proj block1 | 42 | -6.983e-6 | 0.145% | -6.627e-6 | -7.765e-6 | 1.17x |
| layer0 o_proj block1 | 43 | -9.538e-6 | 0.276% | -9.081e-6 | -5.463e-5 | 6.02x |
| layer3 down_proj block1 | 42 | -4.022e-5 | 0.054% | -3.218e-5 | -1.048e-4 | 3.26x |
| layer3 down_proj block1 | 43 | -2.161e-4 | 0.038% | -2.030e-4 | -2.210e-4 | 1.09x |
| layer3 down_proj block16 | 42 | -3.494e-7 | 2.70% | -1.856e-7 | -1.544e-4 | 832x |
| layer3 down_proj block16 | 43 | -7.012e-6 | 0.908% | -6.642e-6 | +3.187e-4 | 47.98x |

中心差分与 FP32 autograd 梯度在六处均吻合，验证了该平滑 FP32 目标上的导数实现。BF16 路径中一处整步 ΔKL 反号，其余五处同号但幅值相差 1.09–832 倍；大比值主要出现在 FP32 整步变化本身接近零的后期 down_proj 点。BF16 的 \(f(t)\) 随 \(t\) 不是平滑单调曲线。这证明“平滑路径梯度正确”不等于“梯度能准确预测原生 BF16 路径的有限一步变化”。

随后直接比较 `before_update.pt` 与 `after_update.pt` 的 FP32 master 权重及二者转成 BF16 后的有效权重：

| 检查点 | seed | FP32 有变化坐标中，BF16 完全不变 | BF16 有效更新 / FP32 更新 cosine |
|---|---:|---:|---:|
| layer0 o_proj block1 | 42 | 91.43% | 0.1607 |
| layer0 o_proj block1 | 43 | 91.30% | 0.1609 |
| layer3 down_proj block1 | 42 | 38.47% | 0.6683 |
| layer3 down_proj block1 | 43 | 36.44% | 0.6759 |
| layer3 down_proj block16 | 42 | 38.56% | 0.6654 |
| layer3 down_proj block16 | 43 | 40.53% | 0.6550 |

这证实小更新经过 BF16 转换后会稀疏化并改变方向，尤其 o_proj 点明显；但它仍不能解释整条下游链路，因为 REAL-Q 的 master 权重保留 FP32，后续 GPTQ 量化也使用该状态。该统计只说明 BF16 前向所见到的更新与 master 更新不同，不能据此宣称优化器更新被永久丢弃。

这仍是受限诊断，不能把差异单独归因于 BF16 权重舍入：两条路径的整个目标 block 和下游计算精度都不同，FP32/BF16 的 KL 基线也不同；本实验没有逐层钩住激活来定位首次分歧。因此目前可下的结论是有限精度是重要混杂因素，尚不能断言它是 Adam 适配无收益的唯一原因，也不能据此推出新的更新规则。

## 当前决策

保留原 Adam，暂停 Cautious/Grams 和单纯精度变体的推广。本轮新增诊断已完成输出响应、权重梯度精度、目标 Linear/整块 FP32 前向、输出空间梯度及局部性检查，具体结果见后续小节。它们尚未形成能改善 KL 的候选方案；此时继续改 Adam 矩或按模块挑选规则缺少机制依据。

当前证据更支持把后续工作转到 REAL-Q 的输出侧反馈近似 \(F\Delta y\)：先查明它为什么不能复现同一目标 KL 的 \(\partial L/\partial y\)，再设计更好的 Fisher/Hessian-vector 近似。局部样本诊断不能作为最终论文质量结论，任何候选都要在新样本/层或模型上复现。

## 补充：逐层更新响应审计（已完成）

新增离线脚本 `tools/audit_transition_layer_outputs.py`，在同一组保存的校准输入、同一 before/after master 权重上，记录目标 Linear 输出以及目标 Transformer block 到最后一层的输出变化。BF16 replay 的 KL 步长与有限差分记录一致；平滑 FP32 replay 也通过相同 KL 对照。脚本只做离线前向，不改 REAL-Q 的量化或 Adam 路径。响应余弦比较的是各精度下的 `(after_output - before_output)`，不是两种精度的绝对激活。

| 检查点 | 目标 Linear 的 BF16/FP32 响应 RMS 比 | 目标 Linear 响应 cosine | 首个显著后缀放大点 | 最后一层 BF16/FP32 响应 RMS 比 |
|---|---:|---:|---|---:|
| layer0 o_proj block1，seed42 | 8.53x | 0.122 | layer2，约 5.52x | 37.17x |
| layer0 o_proj block1，seed43 | 7.75x | 0.134 | layer2，约 6.41x | 25.99x |
| layer3 down_proj block1，seed42 | 1.33x | 0.752 | layer4，约 2.34x | 9.89x |
| layer3 down_proj block1，seed43 | 1.30x | 0.770 | layer4，约 2.22x | 9.52x |
| layer3 down_proj block16，seed42 | 2.86x | 0.351 | layer4，约 10.56x | 53.41x |
| layer3 down_proj block16，seed43 | 2.92x | 0.342 | layer4，约 10.68x | 56.86x |

结果显示差异首先已经出现在目标 Linear 的 BF16 有效更新响应；随后在后续 Transformer 层中进一步放大。FFN 后期 down_proj 的放大在两个 seed 上都出现，方向余弦到 layer11–13 附近降至接近零或变负；因此这不只是某个随机种子下单个 KL 标量波动。o_proj 的目标层响应在 BF16 与 FP32 参考间余弦都只有约 0.13，说明前向可见的局部更新方向已明显受精度路径影响。

不能把这些 RMS 比直接解释为误差增长倍数：FP32 参考的一步响应在 layer3/down_proj/block16 很小，分母接近零会放大比值；而各层输出扰动也不是 KL 的直接度量。另一个边界是 FP32 参考从 BF16 checkpoint 提升执行，只用于与原生 BF16 路径作机制对照，不代表高精度模型的真实表现。后续若要提出数值格式适配方法，需要在更多层/独立 seed 上重放，并以实际下一块 GPTQ 后 KL 与全模型输出验证。

原始逐层数据保存在 campaign 的 `layer_output_audit.json`；BF16 master 权重坐标变化率与方向对照保存在上一节对应记录中。

## 补充：舍入、目标 Linear 精度和整块精度候选（已完成）

### 只把目标权重梯度累加保留 FP32

`tools/probe_bf16_master_gradient.py` 用自定义 autograd Linear 保持原 BF16 前向逐位一致，只让权重梯度累加保留 FP32。六处的 BF16/FP32 梯度相对 L2 差仅 **0.059%–0.162%**，方向余弦约为 1；基线梯度能重放线上快照。故单纯把梯度张量由 BF16 提升到 FP32不足以解释前面观察到的 KL 波动，也不是有希望的单点修复。

### 将目标 Linear 或整个 block 改为 FP32 执行

`tools/probe_precision_proxy_alignment.py` 只把目标 Linear 乘加改为 FP32 后输出转回 BF16；`tools/probe_fp32_block_proxy_alignment.py` 将目标 block 和下游平滑参考都提升到 FP32。表中是同一组 refresh 样本下代理梯度与对应精度路径的真实下游 KL 梯度 cosine：

| 检查点 | 原生 BF16 | 目标 Linear FP32 | 整个 block/suffix FP32 |
|---|---:|---:|---:|
| layer0 o_proj，seed42 | 0.03140 | 0.02447 | 0.03836 |
| layer0 o_proj，seed43 | 0.04863 | 0.04317 | 0.03844 |
| layer3 down_proj block1，seed42 | 0.000889 | 0.001399 | 0.000452 |
| layer3 down_proj block1，seed43 | 0.000455 | 0.000442 | -0.000072 |
| layer3 down_proj block16，seed42 | 0.001733 | 0.001645 | 0.000933 |
| layer3 down_proj block16，seed43 | 0.000570 | 0.000518 | 0.000641 |

改变目标 Linear 的精度会实质改变 o_proj 的梯度，但没有跨点一致地改善与 KL 梯度的对齐；down_proj 几乎不变或更差。整块 FP32 也没有稳定收益。由此不能把问题归结为目标 Linear 的 BF16 乘法，或简单归结为 SwiGLU 某个算子的舍入。

## 补充：直接核对输出空间反馈 \(\partial L/\partial y\)（已完成）

`tools/probe_output_gradient_alignment.py` 将当前 block 输出 \(y\) 设为求导叶变量，使用同一批八条、每条 64 tokens 的 refresh 样本，比较实际 Fisher refresh loss 的 \(\partial L_F/\partial y\) 与真实 teacher/student 下游 KL 的 \(\partial L_{KL}/\partial y\)。真实 KL 通过相同 BF16 后缀、norm 和 head 反传；代理 loss 包含 REAL-Q 当次 refresh 的 slide 当前项与下一层项。两边 loss 重放均与存档完全吻合。

| 检查点 | 合并 Fisher 反馈 cosine | 仅当前层 Fisher | 仅 slide 下一层 Fisher | Fisher/真实 KL 输出梯度 RMS 比 |
|---|---:|---:|---:|---:|
| layer0 o_proj，seed42 | 0.02408 | 0.01206 | 0.03492 | 0.088x |
| layer0 o_proj，seed43 | 0.02785 | 0.01113 | 0.04183 | 0.078x |
| layer3 down_proj block1，seed42 | -0.000299 | -0.000662 | -0.000174 | 5.96x |
| layer3 down_proj block1，seed43 | -0.000160 | -0.000238 | -0.000076 | 4.92x |
| layer3 down_proj block16，seed42 | -0.000268 | -0.000693 | -0.000237 | 6.82x |
| layer3 down_proj block16，seed43 | -0.000208 | -0.000299 | -0.000183 | 4.77x |

这一步完全绕开 \(\partial y/\partial W\)：下游 KL 的真实反馈与 Fisher 反馈已经近乎正交，down_proj 点的幅值还从偏小变成偏大。当前和 slide 下一层两项单独看也没有解决方向失配。因此主要问题已经定位到 \(F\Delta y\) 对 \(\partial L_{KL}/\partial y\) 的近似侧，而不是 Adam 的一阶/二阶矩如何更新，也不是只在 \(\partial y/\partial W\) 的 SwiGLU Jacobian 中才出现。

## 补充：误差是否只因离开局部展开点太远（已完成）

`tools/probe_fisher_locality_path.py` 沿 \(y(\lambda)=y_{FP}+\lambda(y_Q-y_{FP})\) 插值，在平滑 FP32 后缀上比较 Fisher 反馈和真实 KL 输出梯度，\(\lambda=0.1,0.25,0.5,1\)。在 \(\lambda=0.1\) 时，o_proj cosine 仍只有约 0.026–0.027；down_proj block1 为 -0.0038 至 -0.0074，block16 为 -0.0026 至 -0.0063。增大到 \(\lambda=1\) 后同样接近零。故在这些点上，简单缩短残差距离没有恢复对齐；证据更倾向于当前保存的 Fisher 矩阵本身不是所需的局部 KL 曲率方向。该观察是局部方向诊断，不证明所有层/数据都如此。

## 补充：teacher 预测分布 Fisher 小样本试探

由于 REAL-Q 预计算 Fisher 是从真实 token NLL 的输出梯度外积累积而来，一个可能解释是它与 teacher KL 的局部模型 Fisher 不同。`tools/probe_predictive_fisher_feedback.py` 在 layer3/down_proj/block1、seed42 上，用 512 个 token 位置、每位置一份 teacher 分布采样，构成一次 Monte Carlo predictive-Fisher 外积，并与原 Fisher 反馈在相同残差方向上比较。两次独立抽样的 cosine 分别为：

| 抽样 seed | 原 Fisher 到真实 KL 梯度 | predictive Fisher 到真实 KL 梯度 | predictive/original feedback RMS |
|---:|---:|---:|---:|
| 0 | -0.000644 | -0.000331 | 0.946 |
| 1 | -0.000644 | -0.000457 | 0.863 |

predictive Fisher 的方向仍接近正交，且只在一个检查点上试了两次随机抽样，不能称为改进。它也揭示了下一步的门槛：只换 Fisher 的采样分布不够；应先对一个候选曲率估计做独立点验证，确认它确实近似真实 KL 的输出 Hessian-vector product，再谈 Adam 更新或完整量化质量。原始输出空间梯度、局部插值和预测 Fisher 指标均写入 campaign 目录对应 JSON。

### 下一项实验

不再扩展 Adam 规则变体。下一项应检查 Fisher 的构造与目标 KL 曲率之间的差异，并做一个有对照的输出 Hessian-vector 近似试验：在新选的 layer/module 和独立校准样本上，比较原经验 Fisher、一个有理论依据的候选曲率向量，与真实 paired-KL 输出梯度的方向和尺度；通过后再把候选梯度送入 Adam 做实际 Adam-step + GPTQ 后验证。当前 predictive-Fisher 试探尚未通过方向对齐门槛，不应直接推广。

## 补充：精确输出 KL-HVP 与低存储候选（2026-10-05）

为了把“曲率矩阵不合适”与“二阶局部模型本身不够”区分开，新增
`tools/probe_true_kl_output_hvp.py`。它在同一 saved transition、同一批
8×64 tokens 上，把当前 block 的完整输出序列 (y) 当变量；以 FP 目标输出
(y_0) 为展开点，使用实际下游 transformer suffix 和最终 KL，双反传直接算
(H_y(y_0)(y_Q-y_0))，并与 (y_Q) 处精确 paired-KL 的
(partial L/partial y) 对照。目标 block 与 suffix 都用平滑 FP32 路径，
所以它是机制诊断，不是 REAL-Q 原生 BF16 的质量结果，也不是可直接部署的
低成本方法。

| 检查点 | 精确 KL-HVP 与真实 KL 梯度 cosine | HVP/真实梯度 RMS | 原 empirical Fisher 反馈 cosine |
|---|---:|---:|---:|
| layer0 o_proj，seed42 | 0.9905 | 0.952 | 0.00985 |
| layer0 o_proj，seed43 | 0.9407 | 0.677 | 0.00908 |
| layer3 down_proj block1，seed42 | 0.8560 | 0.646 | -0.00064 |
| layer3 down_proj block1，seed43 | 0.8528 | 0.802 | -0.00022 |
| layer3 down_proj block16，seed42 | 0.8666 | 0.609 | -0.00064 |
| layer3 down_proj block16，seed43 | 0.8492 | 0.817 | -0.00025 |

HVP 与学生点 KL 梯度在 6/6 个点同向；相较之下，原 Fisher 反馈近乎正交。
在 layer3/down_proj/seed42 的一处，中心差分直接核对 HVP：沿真实残差取
步长比例 0.1，HVP 与梯度中心差分余弦 0.9994，相对 L2 误差 2.06%。这支持
实现计算的是预期的全序列 KL 二阶向量积，而非普通 autograd 接线误差。

随后把 teacher categorical predictive Fisher 按输出 token 位置保存为
`T×H` 对角阵（相比每位置完整 `H×H` 矩阵显著省内存），并在相同六点比较。
方向 cosine 只有 0.0080–0.0214（down_proj 四点为 0.0080–0.0181；o_proj 两点
为 0.0102–0.0214），没有修复 Fisher 与真实梯度的方向失配。此结果只否定了
这个具体的“位置区分 + hidden 对角 + 单次类别抽样”估计器，不代表所有低秩、
分块或序列曲率近似都无效。完整 JSON 分别保存在 campaign 的
`true_kl_output_hvp.json` 和 `predictive_fisher_feedback.json`。

又加了一个不训练参数的留出检查：用 8 条 refresh 样本中的偶数编号样本构造
rank-1/2/4 的完整序列输出 score-gradient sketch，只在奇数编号样本上评估。
rank-1/2/4 的留出总体 cosine 在 down_proj 例子分别为 -0.00019/-0.00126/
0.00066；o_proj 为 0.00582/0.00137/0.00033。方向仍然无效，尺度也不稳定，
不能作为保留 sequence Fisher 路线的正结果。这和既有 sequence-Fisher 全模型
负结果一致，本轮不再做 rank 或混合系数搜索。

为区分 token 汇总是否是唯一原因，又用相同 score 梯度做偶数样本拟合、奇数样本
评估，对照共享 `H×H`、逐 token 位置完整 `H×H`、完整序列低秩算子。四个训练
样本估出的逐位置矩阵在 down_proj 两点 cosine 为 0.0007–0.0055，在 o_proj
两点为约 0–0.0043；当前样本量下没有改善。因为每个算子只由 4 个 refresh 输入、
每个输入 1 次 sampled-label backward 构造，这只能算负向筛查，不能证明 token
位置结构无效。

另外直接对精确 KL-HVP 做了 token 分块分解：按连续 8 个 token 分组，只保留
同组输入扰动到同组输出反馈的 block-diagonal 部分。先用两个序列试测后，又扩到
4 个 refresh 序列：down_proj 的 block-diagonal HVP 与真实梯度 cosine 从两例均值
0.697 到四例均值 0.624；o_proj 从 0.783 到四例均值 0.784。被删去的跨组
HVP RMS 占完整 HVP 的比例，在四例分别为 0.672 和 0.627。完整 HVP 与真实 KL
梯度仍分别达到 0.842 和 0.994，而同一小批上的经验 Fisher 反馈 cosine 仍约
为 -0.0005 和 0.0067。

这说明下游注意力等序列耦合确实贡献了可观反馈；block-diagonal HVP 仍保留部分
方向信息，当前经验 Fisher 却几乎没有。因而不能把失配只归因于 cross-token 项：
token 分组内的 Hessian 与共享经验 Fisher 的估计/聚合之间也仍有差距。规模只有
两个模块、每模块四条 refresh 序列，而且执行路径为 FP32 平滑诊断；这些结果用于
定位信息损失，不是量化质量结论。

这一分解通过 HVP 双反传得到，每个 token 分组需一次二阶向量积，代价远高于当前
REAL-Q refresh；它只用于定位信息损失，不符合逐模块额外 warm-up backward 的
工程约束。记录在 `true_kl_output_hvp.json` 的 `*-tokenblock8-n4` 项中；常规六点
HVP 数值仍保存在未带后缀的六个项中。

### 如何解释以及文献依据

这一轮排除了一个简单说法：不能笼统说“Taylor 二阶近似不行”。在这六个
平滑 FP32 检查点上，真实 KL 的局部 Hessian-vector 已能较好指向学生点梯度；
差距主要在 REAL-Q 当前存储/使用的 Fisher 矩阵并未保留这个全序列输出曲率
方向。REAL-Q 的静态 Fisher 来自每个 transformer block 输出梯度的
`(B·T,H)` 行外积后汇总成共享 `(H,H)` 矩阵；refresh 再将同一矩阵用于每个
token 的 (Delta y_t)。因此它丢弃了 token 位置间的结构、不同位置的曲率差别，
并将实际模型后缀 Jacobian 映射出的输出曲率压成一个共享 hidden 矩阵。当前
位置对角估计器只恢复了位置索引，仍丢掉 hidden 交叉项和全序列输出映射，结果
也没有改善。

此解释与已有方法文献的边界相符：Kunstner、Hennig、Balles 指出常用
empirical Fisher 一般不等同于真实 Fisher/Hessian；Botev、Mattan、Barber
指出 generalized Gauss–Newton 可用自动微分计算矩阵向量积，并以低秩或递归
结构降低显存/计算，但大输出情形仍需进一步近似。见
[Kunstner et al., NeurIPS 2019](https://papers.neurips.cc/paper_files/paper/2019/hash/46a558d97954d0692411c861cf78ef79-Abstract.html)
和
[Botev et al., Practical Gauss-Newton Optimisation for Deep Learning, ICML 2017](https://proceedings.mlr.press/v70/botev17a.html)。
这些文献支持继续研究 GGN/HVP 的可计算近似，不是 REAL-Q 上能改进量化的直接证据。

### 下一步决策

暂不把 exact HVP 放进逐模块 Adam：它需要经过真实下游 suffix 的二阶反传，正好违反
“不为每个模块额外 warm-up backward”的主要预算约束。当前可支持的结论是机制性
的：瓶颈确实在右侧曲率反馈 (F\Delta y)，而不是已验证出一个可用 Adam 新算法。
序列梯度外积、预测 Fisher 和按位置对角都未通过方向门槛，不再追加这些变体。

若继续开发，下一项应是对 **exact HVP 的低成本近似** 作一次固定设计的复核：
从同一次已有的静态 Fisher backward 中保留一个有界秩的输出梯度 sketch（或小块
token × hidden 统计），不增加反传次数；在未参与构造的 saved transition 上，
先与 exact HVP 方向比较。只有方向/尺度达到预设门槛后，才做单个真实 refresh
Adam-step 和下一块 GPTQ。该实验必须记录附加缓存字节、静态 precompute 时间和
逐层/最终 KL；若 sketch 仍达不到 HVP 方向，就停止这条近似，不将其包装成论文
贡献。

### 同一次 Stage-0 反传的序列梯度草图（2026-10-05）

新增 `tools/capture_static_sequence_sketch.py`，它在 REAL-Q 的 `static_e2e`
既有 Fisher hook 所挂的 block 输出上再挂一个只读 tensor hook；每个 batch 仍只做
原本的一次全模型 sampled-label CE backward。诊断只保存 layer0/layer3 的
`(N,T,H)` 输出梯度，随后将每条序列梯度展平，形成 rank 不超过样本数的序列
协方差作用 `G^T G Δy/(N·T)`。`tools/evaluate_static_sequence_sketch.py` 将其与
同层 saved transition 的真实 KL 梯度和 exact HVP 对照，也从同一批 hook 梯度构造
共享 `H×H` Fisher 作为对照。

第一轮捕获曾误用了 `rotate=True`，而这些 saved transition 明确为
`rotate=False`；因此那轮 JSON 不作为证据。修正后按 saved config 分别重跑 seed42
和 seed43，均为 Qwen3-0.6B、WikiText-2、8 条校准序列、每条 64 tokens、
`global_loss_bsz=1`、不量化、只执行 Stage-0。捕获张量形状为
`8×64×1024`，每层 FP32 文件约 2.10 MB。

| Transition | 序列协方差 vs 真实 KL 梯度 cosine | 序列协方差 vs exact HVP cosine | 序列协方差 / 真实梯度 RMS | 共享 Fisher vs exact HVP cosine |
|---|---:|---:|---:|---:|
| seed42，layer0/o_proj/b1 | 0.0322 | 0.0332 | 2.66× | 0.0097 |
| seed42，layer3/down_proj/b1 | 0.0474 | 0.0174 | 8.39× | -0.0097 |
| seed43，layer0/o_proj/b1 | 0.0470 | 0.0354 | 2.01× | 0.0097 |
| seed43，layer3/down_proj/b1 | -0.0121 | -0.0210 | 6.75× | -0.0043 |

序列协方差在四个独立检查点都没有对齐真实 KL-HVP；它的尺度也明显过大。共享
Fisher 对照仍接近正交，和此前结论一致。这说明“把同一次 backward 的各 token
梯度先拼成序列向量，再保留跨 token 外积”并未恢复目标 KL 曲率，当前不值得接
Adam 或跑完整 GPTQ。

这次数据只有 8 条序列、64-token smoke 规模，且是 sampled-label empirical
score covariance；它不能排除经过更好统计估计的其他结构。但它已否定当前这个
具体、无额外反传的草图作为可用改进。存储也有现实障碍：若保留 8 条完整 FP32
梯度，`T=2048,H=1024` 时约 64 MiB/层，28 层约 1.75 GiB；FP16 仍约 0.88 GiB，
远高于 REAL-Q 当前每层 BF16 `H×H` Fisher 约 2 MiB。故即使方向通过，它也需要
额外在线压缩才能满足内存目标，而本轮没有理由继续压缩一个未对齐的估计器。

文件位于 `outputs/static_sequence_sketch_qwen06b_n8_t64_seed42_norotate/`、
`outputs/static_sequence_sketch_qwen06b_n8_t64_seed43_norotate/`；HVP 的逐样本
向量保存在本 campaign 目录的 `*_true_kl_vectors.pt`。这一轮只改诊断脚本与文档，
没有修改 Adam 或正式量化行为，也没有量化质量收益结论。

### 共享真实 KL 反传的跨层复用与 microbatch 内存试验

为检查另一条路线，新增 `tools/probe_shared_kl_backward.py`。实验取 seed42 的
layer0/o_proj 一次实际 Adam 后权重作为学生扰动，使用同一组 8×64 校准序列；
从最终 paired KL 做一次反向，hook 到所有 28 个 block 输出，同时保留
layer0/o_proj 和 layer3/7/15/23 的 down_proj 权重梯度。也就是说，一个反传图确实
能给多个层输出和多个模块权重提供真实 KL 梯度，而无需逐模块分别回传。

整批 8 条一次反传耗时 0.72 s，峰值 allocated/reserved 分别为 5.75/5.80 GiB，
可用显存余量不适合直接外推到论文配置。将 batch 分为 8 个单样本 microbatch 后，
总耗时 1.79 s，allocated/reserved 峰值降为 3.29/3.30 GiB（allocated 下降
42.8%）；累积的同一组权重梯度与整批结果相对 L2 误差为 1.3e-4–1.7e-4，
28 层输出梯度最大相对 L2 误差为 2.11e-4，输出残差最大相对 L2 误差为
2.82e-4。两种 microbatch 方式得到的 KL 分别为 3.82428e-4 和 3.82427e-4。
因此 microbatch 累积在这个 smoke 配置下基本保持数值结果，并用约 2.5 倍时间
换取约 43% 的峰值显存下降。这里的时间只计学生侧 forward/backward，不含模型加载、
teacher 前向或完整量化。

本次单模块扰动沿后续网络传播后，block0 的真实 KL 输出反馈与该层 REAL-Q Fisher
反馈余弦为 0.0098；到 block15 约 0.0834，到 block26 约 0.1484，仍不足以把
Fisher 当成可靠替代。共享 KL backward 让这些真实梯度可同时拿到，但它本身是一种
额外完整后缀反传，当前数据没有证明最终 KL/PPL 或量化状态会改善，也没有证明它
比现有逐模块 refresh 更快。selected 权重梯度参数本身合计约 56 MiB FP32；完整
训练级使用还需计入优化器状态、更新顺序和后续 GPTQ 轨迹。

本试验说明这条路线的工程形态比“每模块额外预热反传”更可控：可在一个共享 KL
反传图中同时服务多个模块，并通过 microbatch 累积降低峰值；但准确梯度的价值仍
须通过一次真实 Adam 更新、后续 GPTQ 和最终 KL/PPL 对照来验证。下一步应先固定
一个 block 内的多模块更新范围，用共享反传梯度做一次同时 Adam 更新，再按原 GPTQ
顺序量化，并与同一初始状态的 REAL-Q 对照。不能把这次梯度可用性/显存试验当成
算法收益结论。
