# 八算法：20 设定公平横向总表

> 原五方法表生成于 2026-08-21；新增三方法 release 生成于 2026-08-23，REALQ-SDPA 最终 release 生成于 2026-08-23（Asia/Shanghai）。数值由原始 marker / final audit 自动交叉核验后汇总；不是手工抄表。

## 方法与列定义

- `ResComp-C`：`ResComp core + controlled REAL-Q backend`；使用官方 ResComp CAE 方程与 `full` P/R 布局，但不声称是 native 官方脚本复现。
- `REALQ-F`：量化当前线性层时，同时更新 transformer block 内其他线性层（full-block）。
- `REALQ-S`：默认分支，不更新 transformer block 内其他线性层（single-linear）。
- `EfficientQAT`：论文范围的 signed-symmetric G128 weight-only 两阶段 QAT；只覆盖 W4/W3/W2，W4A4KV4 为 N/A，不做 QuaRot。
- `TurboBOA`：`TurboBOA-C (RealQ-MSE, grouped-F3)`；保留 QuaRot、column act-order 与方法原生 grouped-F3。
- `YAQA-wclip`：`YAQA-wclip-C`；使用 RealQ 等价 w-clip 标量量化器，保留方法原生双侧 RHT、anti-diagonal LDLQ，不额外叠加全局 QuaRot。
- 推理三项为 GSM8K、MATH-500、HumanEval+；HumanEval+ 一列同时给官方 EvalPlus base/plus pass@1。
- `KL` 越低越好；`PPL` 越低越好；其余准确率/平均分越高越好。
- `GPU·h*` 是统一后的单卡 NVIDIA L20C numerical-core 成本，定义见下文。

## 公平性门禁

| 项目 | 审计结论 |
|---|---|
| 矩阵 | 5 个相同模型 × 4 个相同量化设定；TurboBOA、YAQA-wclip 与原五方法均覆盖 20 设定。EfficientQAT 按论文 weight-only 范围覆盖 W4/W3/W2 共 15 设定，5 个 W4A4KV4 格子为 N/A。 |
| 校准集 | 严格相同的 WikiText-2 train `256×2048=524,288` token tensor；八方法各轮均直接引用 `realq_20group_20260808/shared_cache` 的同一物理文件，解包 token 与顺序不变。 |
| 校准哈希 | qwen3-0.6b=aed4e972d312…；llama31-8b-instruct=8030125e31c8…；qwen3-4b=21210e1929aa…；qwen3-8b=a7acbd907d64…；qwen3-32b=749360f8fb36…；REALQ 两分支逐模型也完全相同。 |
| 原五方法量化公共项 | 单卡、G128 natural-column grouping、blocksize=128、对称 per-output-channel MSE weight clipping、act-order、QuaRot、`seed/rotation_seed/refresh_seed=1/0/0`；W4A4KV4 均为对称 per-token A/K/V-aware、groupsize=-1、clip=0.9。GPTAQ/GuidedQuant 的 legacy Cartesian search 与 ResComp-C/REALQ 的 optimized-exact search 在本次有限 FP32 对称域上产生相同 raw scale/zero。 |
| 新增三方法表示与边界 | 三者权重均为 signed-symmetric scalar、natural-column G128。TurboBOA/YAQA-wclip 的 W4A4KV4 使用对称 per-token A/K/V、groupsize=-1、clip=0.9；EfficientQAT 的 A/K/V 恒为 BF16。各自保留论文/方法定义的训练、变换和误差补偿，因此属于 complete-method benchmark，不是 solver-only 消融。 |
| A16 无效参数 | ResComp-C 的 A16KV16 组仍记录 clip=0.9，其他方法为 1.0；但 bits=16 直接返回原 tensor，aware=false 且不安装 K wrapper，因此该字段不进入数值路径。 |
| 质量评测 | WikiText-2 Exact KL/PPL、同一十项 QA 与同一平均口径。八方法全部直接命中 GPTAQ/GuidedQuant 的同一旧 SDPA BF16 teacher 物理文件；REALQ-SDPA 的 40/40 execution log 另有逐行 load-hit / no-regeneration 审计。 |
| 推理评测 | 完全相同的 `realq_zero_shot_v1`：SDPA、seed=1234、greedy、thinking on；GSM8K 1319、MATH-500 500、HumanEval+ 164，EvalPlus 0.3.1 官方评分。三项逐样本 prompt SHA-256 序列也已跨轮核验完全一致。 |
| 硬件 | 八方法均为单张 `NVIDIA L20C`；评测时间不计入 GPU-hour。 |
| 方法专属项 | GPTAQ `alpha=0.25`、Guided 四组 saliency、ResComp-C `alpha/alpha2=0.25/0.25` 与 W2=`org`/W3–W4=`allw`、REALQ 学习率/Block-GD 属于各方法定义；因此这是完整方法 benchmark，不是逐 kernel 消融。 |

### GPTAQ / GuidedQuant 实际命令复核

2026-08-21 不仅复核 plan，还从每个 `quant_success.json` 反向解析其指向的
canonical `config.json` 和实际 `argv`。共审计 GPTAQ `20/20`、
GuidedQuant `20/20`，以下门禁违反数为 **0**：

- 权重 `w_groupsize=128`、`blocksize=128`、`w_clip=true`；
- `--w_asym/--a_asym/--k_asym/--v_asym` 均未出现。这四个参数在
  `process_args.py` 中是默认 false 的 `store_true`；GPTAQ/Guided 权重量化器传入
  `sym=not args.w_asym`，A/V/K 量化器分别传入
  `sym=not args.a_asym/v_asym/k_asym`。因此所有实际低比特 W/A/K/V
  均是对称量化；A16/KV16 组不进入低比特数值路径。
- 需特别区分：**G128 只是权重 groupsize**。A/K/V 实际均为
  `groupsize=-1`，即公平协议约定的对称 per-token 量化，不是 A/K/V G128。
- `--rotate`、`--act_order`、`seed=1`、`rotation_seed=0`、
  `refresh_seed=0` 在 `40/40` 份实际命令中全部存在。
- W4A4KV4 的 `a/k/v_bits=4`、三者 clip=`0.9`、
  `act_quant_aware_gptq=true`、`k_cache_quant_aware_gptq=true`均为
  `10/10`；三种 A16 设定的 A/K/V bits=`16`、clip=`1.0`、aware=false
  均为 `30/30`。GPTAQ 另外在 `20/20` 命令中显式使用
  `alpha=0.25`，GuidedQuant 在 `20/20` 中均使用 `w_method=gptq_guided`。

7 月旧表涉及的 Qwen3-4B/32B × W3A16/W4A4KV4 共 8 个 GPTAQ/
GuidedQuant canonical checkpoint 也按同一门禁复核，为 `8/8` 通过、
`0` 违反。旧批次多了不改变算法定义的 `--offload_inps`；但旧新批次
的 shared quant/rotation utility 源码指纹不同，所以两批 checkpoint 不应解释为
bitwise reproduction。

### 量化数值后端的严格限制

GPTAQ、GuidedQuant、ResComp-C 与本表 REALQ 双分支均使用 SDPA attention；
REALQ 的 Hessian/Fisher 仍使用 TF32 input / FP32 accumulate，其他方法也保留各自
核心数值路径。因此这是校准数据、量化表示、teacher 与评测协议受控的完整方法
benchmark，不宣称是只替换 solver 方程的逐 bit 数值消融。EfficientQAT、
TurboBOA、YAQA-wclip 同样保留各自论文/方法定义的训练、变换与误差补偿。

### Exact-KL teacher 控制

八方法的 WikiText-2 Exact KL 均以 GPTAQ/GuidedQuant 的同一旧 SDPA BF16
reference-logits 物理文件为 teacher。REALQ-SDPA 的 quality cache audit 对 40/40
成功日志逐项要求精确 `Loading reference logits`，并禁止任何 `Generating
reference logits`。因此本表 KL 的 teacher 已满足 bitwise-controlled 条件。

### 八算法的 60 项推理第一名计数

加入 EfficientQAT、TurboBOA、YAQA-wclip 后，仍按每个设定的 GSM8K、
MATH-500、HumanEval+ **plus** 组成 `20×3=60` 个比较单元。EfficientQAT
按论文范围不做激活量化，因此在 5 个 W4A4KV4 设定（15 个推理单元）中
记为 N/A 并从该单元的候选方法中排除；其余方法正常参与。第一名从整数命中数
复算，避免显示百分比舍入制造假并列。

| 方法 | 含并列第一 | 唯一第一 |
|---|---:|---:|
| GPTAQ | 14 | 8 |
| GuidedQuant | 15 | 8 |
| ResComp-C | 6 | 1 |
| REALQ-F | 18 | 10 |
| REALQ-S | 23 | 15 |
| EfficientQAT | 8 | 3 |
| TurboBOA | 7 | 1 |
| YAQA-wclip | 9 | 3 |

60 个单元中有 49 个唯一第一、11 个并列第一；其中 5 个是所有可用方法均为 0 的退化并列。
因此跨方法判断优劣时优先看“唯一第一”；“含并列第一”用于完整记录并列，
其列和可以超过 60。

### REALQ-SDPA 复跑与解释边界

本表的 REALQ-F/REALQ-S 已由 run15 deterministic-FA4 checkpoint 全部替换为
deterministic math-SDPA checkpoint；学习率逐行复用 run15 冻结选择，没有针对
SDPA 重新调参。校准 token 是 GPTAQ/GuidedQuant 的同一批物理文件，质量评测
teacher 也是同一旧 SDPA BF16 物理文件，static/Fisher cache 则按 backend 隔离。

为通过单卡容量门禁，Llama-8B/Qwen3-8B 的 global-loss batch 使用 4，Qwen3-32B
使用 1，Qwen3-4B W2A16 使用 4；Qwen3-32B 仅把数学上独立的 SDPA batch 轴切为
4 并使用 non-reentrant checkpoint，逻辑 backward batch、Hessian/Adam 节奏和
量化配置不变。因此本表满足 token、teacher、seed、quantizer 与评测协议的严格
控制，但不把它描述成相对 run15 只改变一个浮点 kernel 的单变量消融。FA4
诊断及旧 checkpoint 结果保留在 run15 专项文档中。

### GPU-hour 的统一口径

- GPTAQ：原 `algorithm_core_v1`（fusion/rotation + quantization core）。
- GuidedQuant：原 `algorithm_core_v1` + 每模型 saliency producer 的 `1/4`（按本 4-setting campaign 摊销）。
- ResComp-C：每组原始 algorithm timer，含 LN fusion、QuaRot、calibration/H/dXXT、`full` P/R 预计算与所有 decoder weight quantization；无跨设定共享 producer，不再摊销。
- 每个 REALQ 分支：从 SDPA 正式日志的 `Fusing LN`、`Rotating`、`Quantising layers` 三个完整 phase timer 求和，再加该模型 deterministic-SDPA static/Fisher producer 的 `1/4`。
- EfficientQAT：Block-AP + E2E-QP；受 CPU 超卖污染的 Qwen3-0.6B 三组以 checkpoint byte-identical clean replay 计时替换。
- TurboBOA：每组完整算法 timer；六个受并发污染的早期结果以 checkpoint content-identical clean replay 计时替换。
- YAQA-wclip：raw quantization 加 Hessian；A16 Hessian 在同模型 W4/W3/W2 间按 `1/3` 摊销，W4A4-aware Hessian 完整计入；Qwen3-0.6B A16 使用 tensor byte-identical clean retime。
- 八者统一排除模型加载、checkpoint I/O 和全部评测。REALQ 的非零 phase elapsed 使用 tqdm 整秒值；仅对已完成但显示 `00:00` 的亚秒 phase，以最终 completion rate 恢复正时长。因此 `GPU·h*` 保留 6 位仅用于账本复算，不表示微秒级测量精度。

## 20 设定 × 8 算法总表

| ID | 模型 | 设定 | 算法 | KL ↓ | PPL ↓ | QA Avg ↑ | GSM8K ↑ | MATH-500 ↑ | HumanEval+ base/plus ↑ | GPU·h* ↓ |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| C01 | Qwen3-0.6B | W4A16 | GPTAQ | 0.0908936 | 22.4740 | 45.14 | 61.03 | 37.60 | 25.00 / 20.12 | 0.217318 |
| C01 | Qwen3-0.6B | W4A16 | GuidedQuant | 0.0649882 | 21.8767 | 46.01 | 58.23 | 35.80 | 28.05 / 25.61 | 0.443612 |
| C01 | Qwen3-0.6B | W4A16 | ResComp-C | 0.0961812 | 22.4961 | 45.99 | 58.45 | 33.20 | 25.61 / 22.56 | 0.106359 |
| C01 | Qwen3-0.6B | W4A16 | REALQ-F | 0.0489757 | 21.4584 | 45.81 | 59.51 | 36.20 | 20.12 / 18.29 | 0.228179 |
| C01 | Qwen3-0.6B | W4A16 | REALQ-S | 0.0581426 | 21.6000 | 46.79 | 61.64 | 37.00 | 31.10 / 28.66 | 0.219290 |
| C01 | Qwen3-0.6B | W4A16 | EfficientQAT | 0.2421670 | 17.4534 | 44.78 | 63.23 | 36.40 | 23.17 / 20.12 | 0.051653 |
| C01 | Qwen3-0.6B | W4A16 | TurboBOA | 0.0955236 | 22.6814 | 45.71 | 60.88 | 35.80 | 22.56 / 21.34 | 0.130902 |
| C01 | Qwen3-0.6B | W4A16 | YAQA-wclip | 0.0720915 | 21.9716 | 46.45 | 59.67 | 33.20 | 20.12 / 18.29 | 0.487842 |
| C02 | Qwen3-0.6B | W4A4KV4 | GPTAQ | 3.2861776 | 369.8404 | 31.78 | 0.30 | 1.40 | 0.00 / 0.00 | 0.241737 |
| C02 | Qwen3-0.6B | W4A4KV4 | GuidedQuant | 5.0477214 | 2050.4565 | 32.18 | 0.30 | 0.80 | 0.00 / 0.00 | 0.525772 |
| C02 | Qwen3-0.6B | W4A4KV4 | ResComp-C | 3.2300003 | 356.8748 | 31.82 | 0.23 | 0.80 | 0.00 / 0.00 | 0.101168 |
| C02 | Qwen3-0.6B | W4A4KV4 | REALQ-F | 0.6377696 | 33.0349 | 35.81 | 0.45 | 0.40 | 0.00 / 0.00 | 0.199290 |
| C02 | Qwen3-0.6B | W4A4KV4 | REALQ-S | 0.9937649 | 46.6778 | 35.37 | 0.68 | 0.80 | 0.00 / 0.00 | 0.196235 |
| C02 | Qwen3-0.6B | W4A4KV4 | EfficientQAT | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| C02 | Qwen3-0.6B | W4A4KV4 | TurboBOA | 4.5349603 | 1294.5851 | 30.93 | 0.08 | 0.80 | 0.00 / 0.00 | 0.156067 |
| C02 | Qwen3-0.6B | W4A4KV4 | YAQA-wclip | 9.7596817 | 246740.8438 | 30.48 | 0.83 | 1.20 | 0.00 / 0.00 | 0.570312 |
| C03 | Qwen3-0.6B | W3A16 | GPTAQ | 0.3475016 | 27.9098 | 39.49 | 19.79 | 4.80 | 0.00 / 0.00 | 0.291707 |
| C03 | Qwen3-0.6B | W3A16 | GuidedQuant | 0.2359724 | 24.5254 | 40.56 | 21.15 | 8.20 | 0.61 / 0.61 | 1.075958 |
| C03 | Qwen3-0.6B | W3A16 | ResComp-C | 0.3571067 | 28.7269 | 40.56 | 8.34 | 4.40 | 0.00 / 0.00 | 0.080069 |
| C03 | Qwen3-0.6B | W3A16 | REALQ-F | 0.1604749 | 23.1020 | 40.63 | 13.27 | 5.20 | 0.00 / 0.00 | 0.175768 |
| C03 | Qwen3-0.6B | W3A16 | REALQ-S | 0.1999220 | 24.0043 | 42.91 | 16.53 | 5.60 | 0.00 / 0.00 | 0.176306 |
| C03 | Qwen3-0.6B | W3A16 | EfficientQAT | 0.5043333 | 26.7691 | 39.52 | 1.44 | 4.60 | 0.00 / 0.00 | 0.050731 |
| C03 | Qwen3-0.6B | W3A16 | TurboBOA | 0.4961515 | 31.1215 | 40.37 | 16.38 | 4.00 | 0.00 / 0.00 | 0.169892 |
| C03 | Qwen3-0.6B | W3A16 | YAQA-wclip | 0.3117986 | 26.0361 | 42.88 | 14.40 | 4.00 | 0.00 / 0.00 | 0.540777 |
| C04 | Qwen3-0.6B | W2A16 | GPTAQ | 1.7180772 | 95.9490 | 31.45 | 0.53 | 1.60 | 0.00 / 0.00 | 0.413600 |
| C04 | Qwen3-0.6B | W2A16 | GuidedQuant | 1.8523664 | 95.8677 | 31.04 | 0.53 | 3.20 | 0.00 / 0.00 | 0.945503 |
| C04 | Qwen3-0.6B | W2A16 | ResComp-C | 1.8813474 | 112.1039 | 30.67 | 0.30 | 1.20 | 0.00 / 0.00 | 0.073716 |
| C04 | Qwen3-0.6B | W2A16 | REALQ-F | 0.7302257 | 36.4039 | 33.99 | 0.38 | 1.00 | 0.00 / 0.00 | 0.175203 |
| C04 | Qwen3-0.6B | W2A16 | REALQ-S | 1.0199944 | 47.7785 | 33.01 | 0.53 | 1.40 | 0.00 / 0.00 | 0.176034 |
| C04 | Qwen3-0.6B | W2A16 | EfficientQAT | 7.6466551 | 28953.5742 | 30.44 | 0.91 | 4.00 | 0.00 / 0.00 | 0.051758 |
| C04 | Qwen3-0.6B | W2A16 | TurboBOA | 3.7238121 | 608.7859 | 30.71 | 0.23 | 0.60 | 0.00 / 0.00 | 0.175807 |
| C04 | Qwen3-0.6B | W2A16 | YAQA-wclip | 17.6170025 | 424796480.0000 | 31.89 | 0.00 | 0.00 | 0.00 / 0.00 | 0.512315 |
| C05 | Llama-3.1-8B-Instruct | W4A16 | GPTAQ | 0.0296730 | 7.4266 | 66.17 | 80.52 | 37.60 | 60.98 / 57.32 | 0.965085 |
| C05 | Llama-3.1-8B-Instruct | W4A16 | GuidedQuant | 0.0282109 | 7.4091 | 66.39 | 83.62 | 37.60 | 58.54 / 55.49 | 2.242274 |
| C05 | Llama-3.1-8B-Instruct | W4A16 | ResComp-C | 0.0316809 | 7.4581 | 66.62 | 80.14 | 36.60 | 59.76 / 55.49 | 0.330392 |
| C05 | Llama-3.1-8B-Instruct | W4A16 | REALQ-F | 0.0239341 | 7.3765 | 66.39 | 83.40 | 35.60 | 60.37 / 56.10 | 1.505475 |
| C05 | Llama-3.1-8B-Instruct | W4A16 | REALQ-S | 0.0244894 | 7.3817 | 66.36 | 82.71 | 35.60 | 58.54 / 53.66 | 1.440197 |
| C05 | Llama-3.1-8B-Instruct | W4A16 | EfficientQAT | 0.1389803 | 6.8016 | 66.03 | 82.94 | 34.60 | 55.49 / 48.78 | 0.191680 |
| C05 | Llama-3.1-8B-Instruct | W4A16 | TurboBOA | 0.0259965 | 7.3993 | 66.66 | 82.26 | 35.20 | 59.15 / 54.27 | 0.628342 |
| C05 | Llama-3.1-8B-Instruct | W4A16 | YAQA-wclip | 0.0283492 | 7.4232 | 66.37 | 82.94 | 36.60 | 58.54 / 51.83 | 4.384791 |
| C06 | Llama-3.1-8B-Instruct | W4A4KV4 | GPTAQ | 0.2108026 | 8.8535 | 61.46 | 69.60 | 20.80 | 42.68 / 39.02 | 0.911918 |
| C06 | Llama-3.1-8B-Instruct | W4A4KV4 | GuidedQuant | 0.2572249 | 9.3783 | 61.24 | 68.92 | 22.00 | 34.76 / 30.49 | 2.120096 |
| C06 | Llama-3.1-8B-Instruct | W4A4KV4 | ResComp-C | 0.2164089 | 8.9312 | 61.59 | 68.76 | 22.80 | 40.24 / 35.98 | 0.346384 |
| C06 | Llama-3.1-8B-Instruct | W4A4KV4 | REALQ-F | 0.1880534 | 8.6319 | 62.30 | 65.20 | 17.40 | 29.27 / 25.00 | 1.609641 |
| C06 | Llama-3.1-8B-Instruct | W4A4KV4 | REALQ-S | 0.1954761 | 8.6847 | 61.40 | 68.69 | 22.00 | 37.20 / 35.37 | 1.479363 |
| C06 | Llama-3.1-8B-Instruct | W4A4KV4 | EfficientQAT | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| C06 | Llama-3.1-8B-Instruct | W4A4KV4 | TurboBOA | 0.2138683 | 8.8707 | 61.97 | 68.61 | 22.60 | 43.90 / 35.98 | 0.655738 |
| C06 | Llama-3.1-8B-Instruct | W4A4KV4 | YAQA-wclip | 3.6233931 | 267.5844 | 33.21 | 0.61 | 0.40 | 0.00 / 0.00 | 4.383406 |
| C07 | Llama-3.1-8B-Instruct | W3A16 | GPTAQ | 0.1186237 | 8.1196 | 64.17 | 70.28 | 23.60 | 31.71 / 29.27 | 1.561493 |
| C07 | Llama-3.1-8B-Instruct | W3A16 | GuidedQuant | 0.1168033 | 8.1115 | 64.49 | 72.10 | 29.00 | 39.63 / 35.37 | 3.951369 |
| C07 | Llama-3.1-8B-Instruct | W3A16 | ResComp-C | 0.1240302 | 8.1658 | 64.23 | 73.31 | 24.40 | 35.37 / 30.49 | 0.330496 |
| C07 | Llama-3.1-8B-Instruct | W3A16 | REALQ-F | 0.0884760 | 7.8877 | 65.25 | 72.86 | 26.40 | 43.90 / 38.41 | 1.485197 |
| C07 | Llama-3.1-8B-Instruct | W3A16 | REALQ-S | 0.0935208 | 7.9060 | 64.85 | 75.59 | 26.20 | 37.80 / 33.54 | 1.394086 |
| C07 | Llama-3.1-8B-Instruct | W3A16 | EfficientQAT | 0.2165287 | 7.9233 | 62.19 | 66.26 | 21.20 | 34.76 / 32.93 | 0.197385 |
| C07 | Llama-3.1-8B-Instruct | W3A16 | TurboBOA | 0.1094856 | 8.0496 | 64.08 | 75.82 | 29.00 | 39.63 / 35.98 | 0.725873 |
| C07 | Llama-3.1-8B-Instruct | W3A16 | YAQA-wclip | 2.6301768 | 100.7631 | 50.23 | 0.83 | 2.40 | 0.00 / 0.00 | 3.986865 |
| C08 | Llama-3.1-8B-Instruct | W2A16 | GPTAQ | 0.7193087 | 14.5583 | 42.31 | 1.52 | 1.60 | 0.00 / 0.00 | 1.568030 |
| C08 | Llama-3.1-8B-Instruct | W2A16 | GuidedQuant | 0.8518013 | 16.7646 | 41.80 | 1.06 | 2.00 | 0.00 / 0.00 | 4.457617 |
| C08 | Llama-3.1-8B-Instruct | W2A16 | ResComp-C | 0.7578120 | 15.1865 | 40.92 | 0.91 | 2.00 | 0.00 / 0.00 | 0.337421 |
| C08 | Llama-3.1-8B-Instruct | W2A16 | REALQ-F | 0.5319166 | 12.2137 | 49.59 | 3.03 | 5.60 | 0.00 / 0.00 | 1.499641 |
| C08 | Llama-3.1-8B-Instruct | W2A16 | REALQ-S | 0.5466840 | 12.3491 | 49.27 | 1.90 | 2.60 | 0.00 / 0.00 | 1.391308 |
| C08 | Llama-3.1-8B-Instruct | W2A16 | EfficientQAT | 3.6673908 | 257.2000 | 30.26 | 0.00 | 0.00 | 0.00 / 0.00 | 0.200823 |
| C08 | Llama-3.1-8B-Instruct | W2A16 | TurboBOA | 0.9360692 | 18.1531 | 39.96 | 0.00 | 0.00 | 0.00 / 0.00 | 0.729881 |
| C08 | Llama-3.1-8B-Instruct | W2A16 | YAQA-wclip | 11.4592304 | 685005.0625 | 32.97 | 0.00 | 0.00 | 0.00 / 0.00 | 3.951761 |
| C09 | Qwen3-4B | W4A16 | GPTAQ | 0.0534650 | 14.3889 | 62.66 | 84.91 | 57.60 | 44.51 / 41.46 | 0.685467 |
| C09 | Qwen3-4B | W4A16 | GuidedQuant | 0.0483016 | 14.0191 | 63.35 | 86.05 | 56.00 | 57.93 / 55.49 | 1.539205 |
| C09 | Qwen3-4B | W4A16 | ResComp-C | 0.0581144 | 14.5369 | 63.05 | 80.59 | 57.40 | 42.07 / 40.24 | 0.244697 |
| C09 | Qwen3-4B | W4A16 | REALQ-F | 0.0449630 | 13.7130 | 62.83 | 87.57 | 58.60 | 51.22 / 48.78 | 0.242727 |
| C09 | Qwen3-4B | W4A16 | REALQ-S | 0.0429256 | 13.6482 | 63.11 | 86.66 | 58.60 | 42.68 / 40.85 | 1.041616 |
| C09 | Qwen3-4B | W4A16 | EfficientQAT | 0.2202488 | 9.6839 | 64.10 | 82.41 | 55.80 | 35.37 / 32.93 | 0.138703 |
| C09 | Qwen3-4B | W4A16 | TurboBOA | 0.0625278 | 14.2187 | 62.56 | 81.20 | 55.60 | 33.54 / 31.71 | 0.427314 |
| C09 | Qwen3-4B | W4A16 | YAQA-wclip | 0.0417815 | 13.9953 | 63.63 | 85.52 | 57.40 | 43.90 / 42.07 | 2.378204 |
| C10 | Qwen3-4B | W4A4KV4 | GPTAQ | 0.4555909 | 18.9317 | 53.53 | 60.35 | 35.20 | 4.88 / 4.88 | 0.779974 |
| C10 | Qwen3-4B | W4A4KV4 | GuidedQuant | 0.4383935 | 17.7117 | 54.17 | 56.48 | 41.00 | 18.90 / 17.68 | 1.884250 |
| C10 | Qwen3-4B | W4A4KV4 | ResComp-C | 0.4167263 | 18.0522 | 55.02 | 66.41 | 40.60 | 10.37 / 9.15 | 0.261402 |
| C10 | Qwen3-4B | W4A4KV4 | REALQ-F | 0.2884524 | 14.6829 | 58.48 | 65.88 | 31.00 | 1.22 / 1.22 | 1.064116 |
| C10 | Qwen3-4B | W4A4KV4 | REALQ-S | 0.3316950 | 15.1985 | 56.93 | 76.88 | 44.80 | 24.39 / 23.17 | 0.988560 |
| C10 | Qwen3-4B | W4A4KV4 | EfficientQAT | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| C10 | Qwen3-4B | W4A4KV4 | TurboBOA | 0.5950803 | 20.9557 | 51.66 | 37.91 | 19.80 | 3.66 / 3.66 | 0.403980 |
| C10 | Qwen3-4B | W4A4KV4 | YAQA-wclip | 8.0512314 | 17795.2344 | 30.23 | 0.83 | 2.80 | 0.00 / 0.00 | 2.619174 |
| C11 | Qwen3-4B | W3A16 | GPTAQ | 0.1917843 | 15.6079 | 59.81 | 81.96 | 51.00 | 33.54 / 31.71 | 1.109987 |
| C11 | Qwen3-4B | W3A16 | GuidedQuant | 0.1727959 | 15.0495 | 60.74 | 82.64 | 54.80 | 36.59 / 34.76 | 2.886723 |
| C11 | Qwen3-4B | W3A16 | ResComp-C | 0.1980814 | 15.5336 | 60.56 | 78.32 | 44.40 | 29.27 / 27.44 | 0.245180 |
| C11 | Qwen3-4B | W3A16 | REALQ-F | 0.1293962 | 14.0844 | 61.34 | 77.79 | 47.60 | 21.95 / 20.12 | 1.033282 |
| C11 | Qwen3-4B | W3A16 | REALQ-S | 0.1459601 | 14.0182 | 60.73 | 86.43 | 53.60 | 29.88 / 28.66 | 0.998282 |
| C11 | Qwen3-4B | W3A16 | EfficientQAT | 0.3335487 | 13.1819 | 57.39 | 56.33 | 27.80 | 0.00 / 0.00 | 0.135227 |
| C11 | Qwen3-4B | W3A16 | TurboBOA | 0.2832130 | 16.5606 | 60.76 | 80.06 | 48.20 | 20.12 / 18.90 | 0.494840 |
| C11 | Qwen3-4B | W3A16 | YAQA-wclip | 0.1520937 | 14.6879 | 59.83 | 78.24 | 47.00 | 17.07 / 15.85 | 2.264527 |
| C12 | Qwen3-4B | W2A16 | GPTAQ | 1.0151397 | 28.8031 | 36.64 | 0.38 | 0.40 | 0.00 / 0.00 | 1.204253 |
| C12 | Qwen3-4B | W2A16 | GuidedQuant | 1.0710428 | 26.0585 | 35.88 | 1.06 | 1.60 | 0.00 / 0.00 | 3.414326 |
| C12 | Qwen3-4B | W2A16 | ResComp-C | 1.0711416 | 30.5105 | 36.37 | 0.23 | 0.80 | 0.00 / 0.00 | 0.248224 |
| C12 | Qwen3-4B | W2A16 | REALQ-F | 0.5412940 | 17.8443 | 45.36 | 0.23 | 1.40 | 0.00 / 0.00 | 1.023004 |
| C12 | Qwen3-4B | W2A16 | REALQ-S | 0.6549757 | 18.9461 | 43.64 | 0.91 | 3.20 | 0.00 / 0.00 | 0.991338 |
| C12 | Qwen3-4B | W2A16 | EfficientQAT | 4.5155420 | 666.1116 | 29.98 | 0.08 | 0.20 | 0.00 / 0.00 | 0.138810 |
| C12 | Qwen3-4B | W2A16 | TurboBOA | 1.8425840 | 65.9658 | 37.02 | 0.00 | 0.00 | 0.00 / 0.00 | 0.514205 |
| C12 | Qwen3-4B | W2A16 | YAQA-wclip | 11.4570885 | 358920.1875 | 31.15 | 0.00 | 0.20 | 0.00 / 0.00 | 2.279174 |
| C13 | Qwen3-8B | W4A16 | GPTAQ | 0.0376639 | 10.0189 | 67.67 | 87.19 | 56.80 | 43.90 / 42.68 | 0.967987 |
| C13 | Qwen3-8B | W4A16 | GuidedQuant | 0.0340581 | 9.8978 | 67.28 | 85.82 | 56.20 | 45.12 / 43.29 | 2.179010 |
| C13 | Qwen3-8B | W4A16 | ResComp-C | 0.0409121 | 10.0599 | 67.02 | 84.31 | 55.40 | 42.68 / 40.24 | 0.349820 |
| C13 | Qwen3-8B | W4A16 | REALQ-F | 0.0323554 | 9.8415 | 67.86 | 86.66 | 58.00 | 43.29 / 40.24 | 1.664674 |
| C13 | Qwen3-8B | W4A16 | REALQ-S | 0.0324821 | 9.8476 | 67.16 | 89.23 | 58.00 | 46.34 / 43.90 | 1.570507 |
| C13 | Qwen3-8B | W4A16 | EfficientQAT | 0.1559180 | 7.9721 | 67.54 | 83.24 | 56.80 | 45.73 / 43.29 | 0.224545 |
| C13 | Qwen3-8B | W4A16 | TurboBOA | 0.0443699 | 10.0745 | 67.75 | 86.20 | 56.60 | 43.90 / 42.07 | 0.652894 |
| C13 | Qwen3-8B | W4A16 | YAQA-wclip | 0.0322305 | 9.8533 | 67.46 | 87.64 | 57.40 | 54.27 / 51.83 | 4.320333 |
| C14 | Qwen3-8B | W4A4KV4 | GPTAQ | 0.3056938 | 12.1756 | 60.49 | 82.94 | 50.40 | 24.39 / 23.17 | 0.977770 |
| C14 | Qwen3-8B | W4A4KV4 | GuidedQuant | 0.3093038 | 11.7392 | 60.99 | 72.63 | 47.40 | 18.90 / 17.07 | 2.575195 |
| C14 | Qwen3-8B | W4A4KV4 | ResComp-C | 0.2961951 | 11.9966 | 60.68 | 80.59 | 47.60 | 24.39 / 22.56 | 0.354657 |
| C14 | Qwen3-8B | W4A4KV4 | REALQ-F | 0.2313732 | 10.8267 | 62.25 | 67.17 | 38.60 | 1.22 / 1.22 | 1.652729 |
| C14 | Qwen3-8B | W4A4KV4 | REALQ-S | 0.2530467 | 11.2904 | 63.09 | 85.67 | 51.40 | 35.98 / 34.15 | 1.566063 |
| C14 | Qwen3-8B | W4A4KV4 | EfficientQAT | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| C14 | Qwen3-8B | W4A4KV4 | TurboBOA | 0.3667215 | 12.6267 | 60.73 | 76.42 | 45.60 | 23.78 / 20.73 | 0.709578 |
| C14 | Qwen3-8B | W4A4KV4 | YAQA-wclip | 8.0090199 | 16333.8125 | 30.04 | 1.59 | 4.20 | 0.00 / 0.00 | 4.698317 |
| C15 | Qwen3-8B | W3A16 | GPTAQ | 0.1313733 | 10.6531 | 64.84 | 81.96 | 49.00 | 18.29 / 16.46 | 1.522448 |
| C15 | Qwen3-8B | W3A16 | GuidedQuant | 0.1226893 | 10.4331 | 65.07 | 79.91 | 49.60 | 17.68 / 16.46 | 4.169223 |
| C15 | Qwen3-8B | W3A16 | ResComp-C | 0.1432403 | 10.8061 | 64.94 | 83.93 | 49.20 | 14.63 / 13.41 | 0.338383 |
| C15 | Qwen3-8B | W3A16 | REALQ-F | 0.1040968 | 10.2504 | 65.33 | 78.85 | 48.60 | 15.24 / 13.41 | 1.607452 |
| C15 | Qwen3-8B | W3A16 | REALQ-S | 0.1144490 | 10.3462 | 65.17 | 85.60 | 51.60 | 29.27 / 26.83 | 1.512452 |
| C15 | Qwen3-8B | W3A16 | EfficientQAT | 0.2544065 | 10.1282 | 63.12 | 74.98 | 48.80 | 15.24 / 14.02 | 0.212620 |
| C15 | Qwen3-8B | W3A16 | TurboBOA | 0.1632450 | 10.8699 | 64.78 | 78.39 | 48.20 | 20.73 / 19.51 | 0.781787 |
| C15 | Qwen3-8B | W3A16 | YAQA-wclip | 0.1307958 | 10.4992 | 65.18 | 80.06 | 53.00 | 23.78 / 21.95 | 4.081735 |
| C16 | Qwen3-8B | W2A16 | GPTAQ | 0.7033842 | 16.6954 | 43.43 | 1.36 | 4.40 | 0.00 / 0.00 | 1.703465 |
| C16 | Qwen3-8B | W2A16 | GuidedQuant | 0.7389094 | 15.6147 | 41.20 | 1.14 | 3.60 | 0.00 / 0.00 | 4.747457 |
| C16 | Qwen3-8B | W2A16 | ResComp-C | 0.6828204 | 16.1136 | 41.94 | 1.36 | 2.00 | 0.00 / 0.00 | 0.351449 |
| C16 | Qwen3-8B | W2A16 | REALQ-F | 0.4619108 | 13.2912 | 49.83 | 3.87 | 4.00 | 0.00 / 0.00 | 1.627174 |
| C16 | Qwen3-8B | W2A16 | REALQ-S | 0.5616412 | 14.9065 | 47.84 | 1.59 | 3.60 | 0.00 / 0.00 | 1.521063 |
| C16 | Qwen3-8B | W2A16 | EfficientQAT | 4.0143013 | 353.3516 | 29.77 | 1.14 | 4.00 | 0.00 / 0.00 | 0.218340 |
| C16 | Qwen3-8B | W2A16 | TurboBOA | 5.0765133 | 1013.2111 | 30.54 | 0.91 | 4.20 | 0.00 / 0.00 | 0.765573 |
| C16 | Qwen3-8B | W2A16 | YAQA-wclip | 16.7574425 | 70114320.0000 | 31.73 | 0.00 | 0.00 | 0.00 / 0.00 | 3.997343 |
| C17 | Qwen3-32B | W4A16 | GPTAQ | 0.0595478 | 7.7030 | 71.13 | 92.27 | 61.40 | 50.00 / 48.78 | 2.915012 |
| C17 | Qwen3-32B | W4A16 | GuidedQuant | 0.0493096 | 7.7314 | 71.59 | 92.80 | 62.20 | 46.95 / 45.73 | 7.573424 |
| C17 | Qwen3-32B | W4A16 | ResComp-C | 0.0518231 | 7.7295 | 71.63 | 93.48 | 61.40 | 51.22 / 50.00 | 1.289660 |
| C17 | Qwen3-32B | W4A16 | REALQ-F | 0.0449553 | 7.6123 | 71.63 | 93.93 | 62.60 | 59.15 / 56.71 | 13.386806 |
| C17 | Qwen3-32B | W4A16 | REALQ-S | 0.0445755 | 7.6006 | 71.56 | 93.71 | 60.20 | 51.83 / 49.39 | 12.776251 |
| C17 | Qwen3-32B | W4A16 | EfficientQAT | 0.1534895 | 6.5035 | 72.87 | 93.93 | 61.20 | 52.44 / 50.00 | 0.670817 |
| C17 | Qwen3-32B | W4A16 | TurboBOA | 1.8178315 | 35.0668 | 43.17 | 1.21 | 2.20 | 0.00 / 0.00 | 2.828414 |
| C17 | Qwen3-32B | W4A16 | YAQA-wclip | 0.0431897 | 7.7136 | 71.67 | 93.56 | 59.80 | 44.51 / 42.68 | 20.844802 |
| C18 | Qwen3-32B | W4A4KV4 | GPTAQ | 0.4107974 | 9.4401 | 64.87 | 88.25 | 54.40 | 16.46 / 15.24 | 3.097557 |
| C18 | Qwen3-32B | W4A4KV4 | GuidedQuant | 0.4482003 | 9.8140 | 63.79 | 88.70 | 54.40 | 18.29 / 17.07 | 7.467291 |
| C18 | Qwen3-32B | W4A4KV4 | ResComp-C | 0.4012597 | 9.4089 | 65.20 | 90.07 | 55.40 | 16.46 / 15.85 | 1.391882 |
| C18 | Qwen3-32B | W4A4KV4 | REALQ-F | 0.2425114 | 8.2157 | 69.06 | 93.56 | 58.20 | 37.80 / 35.98 | 13.471251 |
| C18 | Qwen3-32B | W4A4KV4 | REALQ-S | 0.2548215 | 8.2656 | 68.88 | 94.24 | 61.00 | 40.24 / 39.02 | 12.775418 |
| C18 | Qwen3-32B | W4A4KV4 | EfficientQAT | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| C18 | Qwen3-32B | W4A4KV4 | TurboBOA | 0.4039517 | 9.3452 | 66.76 | 91.21 | 56.20 | 24.39 / 23.78 | 2.803941 |
| C18 | Qwen3-32B | W4A4KV4 | YAQA-wclip | 12.1912260 | 1038383.8750 | 31.88 | 1.29 | 0.00 | 0.00 / 0.00 | 23.717367 |
| C19 | Qwen3-32B | W3A16 | GPTAQ | 0.1558156 | 8.0272 | 70.97 | 93.48 | 59.00 | 46.95 / 43.90 | 4.516529 |
| C19 | Qwen3-32B | W3A16 | GuidedQuant | 0.1496158 | 8.0312 | 70.66 | 92.87 | 58.20 | 37.20 / 36.59 | 14.328829 |
| C19 | Qwen3-32B | W3A16 | ResComp-C | 0.1524957 | 8.0018 | 71.40 | 91.51 | 57.20 | 36.59 / 34.76 | 1.358441 |
| C19 | Qwen3-32B | W3A16 | REALQ-F | 0.1272063 | 7.7466 | 69.82 | 86.05 | 56.40 | 17.07 / 15.85 | 13.352918 |
| C19 | Qwen3-32B | W3A16 | REALQ-S | 0.1303551 | 7.7551 | 70.14 | 93.33 | 59.40 | 44.51 / 41.46 | 12.763751 |
| C19 | Qwen3-32B | W3A16 | EfficientQAT | 0.2585730 | 7.6540 | 68.73 | 84.91 | 53.60 | 35.37 / 33.54 | 0.667888 |
| C19 | Qwen3-32B | W3A16 | TurboBOA | 4.2886949 | 385.1393 | 34.36 | 1.14 | 3.60 | 0.00 / 0.00 | 2.905516 |
| C19 | Qwen3-32B | W3A16 | YAQA-wclip | 0.1372598 | 8.0039 | 69.85 | 92.34 | 59.40 | 43.29 / 40.24 | 21.384183 |
| C20 | Qwen3-32B | W2A16 | GPTAQ | 0.7156695 | 12.1626 | 53.02 | 5.61 | 2.00 | 0.00 / 0.00 | 5.098336 |
| C20 | Qwen3-32B | W2A16 | GuidedQuant | 0.6648021 | 11.4255 | 51.37 | 12.66 | 4.00 | 0.61 / 0.61 | 17.569220 |
| C20 | Qwen3-32B | W2A16 | ResComp-C | 0.6412292 | 11.3898 | 50.95 | 4.09 | 3.20 | 0.00 / 0.00 | 1.377077 |
| C20 | Qwen3-32B | W2A16 | REALQ-F | 0.4446139 | 9.5658 | 59.86 | 32.45 | 6.40 | 4.88 / 4.88 | 13.339584 |
| C20 | Qwen3-32B | W2A16 | REALQ-S | 0.4857540 | 9.8401 | 58.95 | 19.03 | 4.80 | 1.22 / 0.61 | 12.744862 |
| C20 | Qwen3-32B | W2A16 | EfficientQAT | 2.7352650 | 82.7267 | 31.94 | 0.45 | 0.40 | 0.00 / 0.00 | 0.629704 |
| C20 | Qwen3-32B | W2A16 | TurboBOA | 0.9873413 | 15.8305 | 55.46 | 13.57 | 4.60 | 0.61 / 0.61 | 3.001077 |
| C20 | Qwen3-32B | W2A16 | YAQA-wclip | 15.6245213 | 27065838.0000 | 31.56 | 1.06 | 0.00 | 0.00 / 0.00 | 21.293771 |


## GPU-hour 汇总（20 设定）

| 算法 | 有效设定数 | 统一 GPU·h* 合计 | 平均每有效设定 GPU·h* |
|---|---:|---:|---:|
| GPTAQ | 20 | 30.749673 | 1.537484 |
| GuidedQuant | 20 | 86.096354 | 4.304818 |
| ResComp-C | 20 | 9.516878 | 0.475844 |
| REALQ-F | 20 | 70.344111 | 3.517206 |
| REALQ-S | 20 | 67.722980 | 3.386149 |
| EfficientQAT | 15 | 3.780685 | 0.252046 |
| TurboBOA | 20 | 19.661621 | 0.983081 |
| YAQA-wclip | 20 | 132.696999 | 6.634850 |

## 数据来源与审计身份

- GPTAQ/GuidedQuant：`docs/GPTAQ_GuidedQuant_Llama31_Qwen3_20组实验记录_20260809.md`；plan fingerprint `46de88a0f17fe8cf481204d1040458944b52f3753d36cb47604e6dab06093211`。40/40 quant、40/40 quality、120/120 reasoning generation、40/40 EvalPlus official。
- ResComp-C：`docs/ResComp_Llama31_Qwen3_20组量化评测_20260809.md`；plan fingerprint `670d13cc6ef4884098fb2f34ed99b1c238d0019fe4fce73ae00d05d30fba73cb`，audit fingerprint `5814da6be832cec0b6b373ff9fbd1295d6062c8fbfbee8f7f24ad06c87d845ce`。20/20 quant、20/20 quality、60/60 reasoning generation、20/20 EvalPlus official。
- REALQ 双分支（本表权威值）：`docs/REALQ_SDPA双分支20设定复跑_20260822.md`；release `/minimax-avatar-new/zhangqian/realq/experiment_data/realq_sdpa_frozen_lr20_20260822_v2_memory/final_release_v5_merged.json`，fingerprint `a2c700b877ce3ba80a39a9b2619f334bb0e23982c7c85264a2c239db659d05bc`。formal/quality/reasoning audit 分别为 `5b9de2dfda83b9705c1f3b5a0ac9de93b53489200a1007c20bd17b40148f20cf`、`95865b470cef055241fe2ae29dbe7c603e91c29e3afec93e7300d5cdfed88941`、`a09dd654dd0c6ff3f3fe265c1b8c8af2bbe9f334e4e1aa9b3f092c5f5ce554b6`；formal 与 quality 的 exact-cache audit 均为 40/40。历史 run15/FA4 数值不再进入本表主行，仍保留在原实验记录中。
- EfficientQAT：`docs/EfficientQAT_Qwen3_Llama31_15组权重量化实验记录_20260821.md`；15/15 quant、quality、三项 reasoning 与官方 EvalPlus 完整；嵌套 upstream 的实际算法改动保存在 `third_party_patches/fair20_20260821/efficientqat.patch`。
- TurboBOA / YAQA-wclip：`docs/TurboBOA_YAQA_wclip_Qwen3_Llama31_公平20组实验记录_20260821.md`；两方法各 20/20 quant、quality、三项 reasoning 与官方 EvalPlus 完整；嵌套 upstream 改动分别保存在 `third_party_patches/fair20_20260821/turboboa.patch` 与 `third_party_patches/fair20_20260821/yaqa_wclip.patch`。
- 新增三方法 release：55/55 suite，reference fingerprint `3ecffa5d4cb97efb24aac4a668b6570d8e8193f4f4fa21f98825c2d9d238be05`；每行绑定量化 terminal、checkpoint validation、quality result 与官方 HumanEval+ receipt 的 SHA-256。
- 生成时再次核验：baseline 40 行均与 `quant_success.json`、`quality_success.json`、HumanEval+ `official_success.json` 一致；ResComp-C final audit 为 `complete` 且覆盖 20 行；REALQ 三份 final audit 均为 `complete` 且各覆盖 40 行。
