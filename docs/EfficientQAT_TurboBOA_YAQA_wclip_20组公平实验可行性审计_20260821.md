# EfficientQAT / TurboBOA / YAQA_wclip：20 组公平实验可行性审计

> 审计日期：2026-08-21。目标矩阵与
> `REALQ_GPTAQ_GuidedQuant_四算法20设定公平总表_20260821.md` 相同：
> 5 个模型 × 4 个量化设定。本文只判断代码和实验协议是否具备正式铺开的条件；
> 尚未把未运行的数据写入结果大表。2026-08-21 新增的用户决策以本文
> “EfficientQAT 代码审计”一节为准：EfficientQAT 只做论文实际评测过的
> weight-only 设定，不做激活/KV 量化，不做 QuaRot。

## 结论

三种算法都能在当前模型范围中建立可审计的完整方法对比，但 EfficientQAT
的论文主实验只是 weight-only，因此实际需要启动的数量是
`20 (TurboBOA) + 20 (YAQA_wclip) + 15 (EfficientQAT) = 55` 组，不是 60 组：

| 方法 | 当前可行性 | 正式铺开前必须完成的工作 | 建议正式标签 |
|---|---|---|---|
| EfficientQAT | 可行，只跑 15 个唯一 weight-only 设定 | 保留论文 Block-AP + E2E-QP 配方；绑定 256×2048 WikiText-2；将权重量化器限定为 G128 signed-symmetric uniform scalar；不量化 A/K/V；不做 QuaRot；打开确定性门禁 | `EfficientQAT-Sym-WO` |
| TurboBOA | 最接近可运行，改造量小 | 直接绑定五份主表 token 制品及 SHA；补统一确定性入口；修复一个过期 mock 单测；做 5 模型 smoke 和 Qwen3-32B 有限性 canary | `TurboBOA-C` |
| YAQA_wclip | 有条件可行，不需要重写 quantizer | 直接使用已有 RealQ 等价 w-clip 标量量化器；去掉旧模型/token 硬编码；dense 路径 lazy-import QTIP kernel；显式记录求解精度；补 0.6B/8B/Llama-3.1 门禁；单卡 smoke | `YAQA-wclip-C` |

这里的“公平”是大表现有定义的 complete-method benchmark：公共数据、量化表示、
激活/KV 量化、评测和计时边界一致；各方法自己的优化方程、变换和训练预算保留并披露。
它不是只替换一个 solver、其余浮点 kernel 逐指令相同的消融。TurboBOA 必须用 eager
attention 取得 attention matrix，YAQA 使用原生双侧 RHT，EfficientQAT 使用 QAT，强行抹平
这些差异会改变方法本身。

## 统一的 20 组合同

### 模型与设定

模型固定为：

1. Qwen3-0.6B；
2. Llama-3.1-8B-Instruct；
3. Qwen3-4B；
4. Qwen3-8B；
5. Qwen3-32B。

每个模型固定四组：`W4A16KV16`、`W4A4KV4`、`W3A16KV16`、
`W2A16KV16`。所有目标 projection 的输入宽度都可被 128 整除：

| 模型 | hidden | MLP intermediate | Q/O 宽度 | G128 结构门禁 |
|---|---:|---:|---:|---|
| Qwen3-0.6B | 1024 | 3072 | 2048 | 通过 |
| Llama-3.1-8B-Instruct | 4096 | 14336 | 4096 | 通过 |
| Qwen3-4B | 2560 | 9728 | 4096 | 通过 |
| Qwen3-8B | 4096 | 12288 | 4096 | 通过 |
| Qwen3-32B | 5120 | 25600 | 8192 | 通过 |

EfficientQAT 是该矩阵的 weight-only 子集：每个模型只跑
`W4A16KV16`、`W3A16KV16`、`W2A16KV16` 三组，共 15 组。原矩阵的
`W4A4KV4` 对 EfficientQAT 记为 `N/A（论文未做 A/K/V 量化）`，不把
`W4A16KV16` 的同一个 checkpoint 重复跑一次或复制数值冒充 W4A4。

### 校准数据

- WikiText-2 train，`256 × 2048 = 524,288` token，采样 seed=1；
- 必须直接读取 `experiment_data/realq_20group_20260808/shared_cache` 下五份
  既有物理 `.pt` 文件，不允许按相同 seed 重新采样，也不允许只比较文件名；
- 必须同时校验 archive SHA-256、解包后的 tensor SHA-256、容器类型、长度、dtype、
  shape 和顺序；
- 五份 archive SHA-256 分别为：
  `aed4e972d312…`、`8030125e31c8…`、`21210e1929aa…`、
  `a7acbd907d64…`、`749360f8fb36…`；
- EfficientQAT 的 Block-AP 与 E2E-QP 只能重复遍历这同一份制品，不能引入
  RedPajama、Alpaca、DEITA 或第二份 WikiText-2 抽样。因公共 token 制品的单条长度为
  2048，E2E-QP 也使用 2048 context，这是相对论文 4096-context E2E 数据的显式受控覆盖；
- YAQA 的 Hessian producer 与 raw quantizer 必须绑定同一 token 路径和 SHA；
  W4A4 必须单独生成 A/K/V-aware Hessian，不能复用 A16 Hessian。

历史 Qwen3-4B Turbo/YAQA 记录使用过 archive SHA `e87b6d8f…`，而本轮主表为
`21210e19…`。即使解包 token 可能相同，也不能把历史结果直接挪入本表；本轮要求同一
物理制品。

### 权重量化器

公共叶子量化表示固定为：

- W2/W3/W4 signed symmetric 整数域分别为 `[-2,1]`、`[-4,3]`、`[-8,7]`；
- per-output-row × contiguous natural-input-column G128；
- 每组一个正 scale，zero 固定为 0；若使用 unsigned packed 表示，只允许固定偏移
  `2^(bits-1)`，不得学习 zero；
- TurboBOA 与 YAQA_wclip 的 scale 初值由同一 FP32 MSE/w-clip 定义产生：
  `norm=2.4`、`grid=50`、`maxshrink=0.5`、strict first-winner tie；
- EfficientQAT 按论文/官方代码的 uniform scalar QAT 配方，使用对称 max-abs
  初始 scale，随后在 Block-AP 和 E2E-QP 中学习 scale。这不是相同初始量化器的
  solver-only 消融，必须在大表中披露为 complete-method 差异；
- `cartesian_legacy` 与 `symmetric_union_exact` 只在已证明 raw scale/code 完全等价时
  视为实现优化，不视为不同量化器；正式 plan 必须记录实际 search implementation 和
  compact qparam layout，优先使用 optimized-exact；保留 legacy 的方法必须通过全 bit/G128
  parity 门禁；
- 对所有 decoder layer 的 q/k/v/o/gate/up/down 七个 projection 量化；embedding、
  lm_head、所有 RMSNorm 以及 Qwen3 q_norm/k_norm 保持原值；禁止 skip/fallback；
- 算法可按自身定义修改权重或后续优化 scale，但必须在 provenance 中明确记录。
  TurboBOA 的 grouped F3 scale adaptation、EfficientQAT 的 learnable scale 属于方法核心，
  不能在结果不好时关闭或切换。

### A/K/V 量化

- A16/KV16 组为恒等路径，clip 统一记录 1.0；
- W4A4KV4 为 signed symmetric、per-token `groupsize=-1`、clip=0.9；
- 七个 projection 输入做 A4，v_proj 输出做 V4，RoPE 后 K cache 做 K4；Q cache 不再
  单独 QDQ；
- W4A4 的曲率/重构/训练阶段必须已经看到这套 A/K/V QDQ，不能训练结束后才以
  unaware 方式挂载；checkpoint reload 与全部评测继续恢复同一拓扑。

上述 W4A4 要求适用于实际支持该设定的 TurboBOA 与 YAQA_wclip。EfficientQAT
本轮不创造论文未实验的激活量化变体；其 A/K/V 全部为 BF16 恒等路径，
因此“A 对称性”记为 `N/A`，而不是虚构一个没有生效的对称激活 quantizer。

### 随机性、数值后端与评测

- calibration seed=1、rotation seed=0、推理 seed=1234；每种方法的其他随机流都从
  plan 中的统一 global seed 派生并写入 manifest；
- 开启 PyTorch deterministic algorithms，关闭 cuDNN benchmark；禁止未记录的随机
  fallback。正式前每种方法至少选一组做两次 cold replay，要求 checkpoint 的量化码、
  scale 和核心指标可复现；
- 三种新增方法的质量评测统一使用同一 attention backend、同一 full-vocabulary BF16
  teacher cache、WikiText-2 PPL/KL 和同一十项 QA evaluator；
- 三项推理继续使用 `realq_zero_shot_v1`：SDPA、thinking on、greedy、GSM8K 1319、
  MATH-500 500、HumanEval+ 164，并使用 EvalPlus 0.3.1 官方评分；
- 当前大表已经注明 REALQ run15 与旧 baseline 的 teacher/backend 不同。新增三方法即使
  彼此严格共享 teacher，写回旧表时 KL 仍须保留 `KL†` 标记，除非连既有方法一起重算。

### GPU-hour

- 每组均按单张 NVIDIA L20C 运行；两台 8 卡机器可以并发 16 个独立单卡 run；
- 计时包含该方法必需的 fusion/rotation、Hessian/static producer、训练/量化 core；
  排除模型加载、checkpoint I/O、hfize/格式转换和全部评测；
- TurboBOA 每组独立计时；EfficientQAT 计 Block-AP + E2E-QP；
- YAQA A16 Hessian 可在同模型 W4/W3/W2 三组间共享，按 `1/3` 摊销；W4A4-aware
  Hessian 只服务一个设定，完整计入该组；raw YAQA rounding 各组完整计入；
- 若实际执行选择每组 fresh Hessian，也必须另列“未摊销实耗”，不能与主表摊销值混用。

## EfficientQAT 代码审计

### 论文是否做了激活量化

结论是：**EfficientQAT 论文的实证结果是 weight-only**。证据如下：

- 主实验表只列 `W2G64/G128`、`W3G128`、`W4G128`，没有 W4A4/W8A8 行；
- 附录明确把 group size 称为“weight-only quantization”的超参，并说只量化
  transformer block 内的 Linear 权重，norm/embedding/head 保留 FP16；
- 官方 `UniformAffineQuantizer` 只挂在 `QuantLinear.weight`，代码中没有 activation
  quantizer/observer 或 A/K/V bit 入口；
- 正文虽有一句“EfficientQAT supports both weight and activation quantization”，但紧接着引用的是
  后续独立方法 PrefixQuant；官方 README 也把 PrefixQuant 单独称为新的
  weight-activation 算法。这句话不能当成本论文已经做了激活量化实验的证据。

### 本轮冻结配置

- 方法主体保留论文两阶段：Block-AP 按 transformer block 顺序训练所有模型
  weight 与量化参数，再用 E2E-QP 只训练全模型 scale。按论文/官方实现，
  Block-AP 内 RMSNorm weight 也属于“all parameters”，不再为了和 PTQ 方法形式一致而冻结它；
- Block-AP：batch size=2，epoch=2，quantizer LR=`1e-4`，weight LR 为 W2
  `2e-5`、W3/W4 `1e-5`，cosine 降到初值的 1/20，FP16 AMP；
- E2E-QP：epoch=1，effective batch size=32，只训练 scale，LR 为 W2 `2e-5`、
  W3/W4 `1e-5`，cosine、warmup ratio=0.03、max grad norm=0.3、BF16；
- 环境继续使用先前已确认的 `torch.optim.AdamW` 兼容替代；官方锁定的
  `bitsandbytes==0.41.0` 不存在于当前 Torch 2.9/CUDA 12.8 共享环境。LR、WD 与
  scheduler 不变，manifest 必须披露这项兼容替代；
- 受控覆盖：每模型使用主表同一份 256×2048 WikiText-2 token 制品；
  `groupsize=128`；W2/W3/W4 signed-symmetric uniform scalar，zero 固定，scale 可学习；
- A/K/V 保持 BF16，不安装训练期或训练后 activation quantizer；不做 QuaRot；
- 不做额外 LR 调参，不使用评测结果挑超参；不使用验证集选 checkpoint。

### 现有能力与启动前门禁

- `EfficientQAT/quantize/quantizer.py` 已有 symmetric 分支：signed grid 通过固定 packed
  zero 表示，zero 不是 parameter；W2/W3/W4 与 G128 pack/materialize 已有测试；
- `experiments/efficientqat_compare/run_one.py` 能直接加载带 SHA 的精确 token artifact，
  逐 block 运行 Block-AP，再对所有 scale 做 E2E-QP；
- Qwen3 和 Llama 的七 projection topology、packed checkpoint 以及 dense BF16
  materialization 路径已经存在；Qwen3-4B/32B 也有历史成功实跑；
- 旧 runner 的 `cudnn.benchmark=True` 需要改为确定性配置，并固定 Python/NumPy/Torch
  seed、CUBLAS workspace、TF32 状态；
- 正式前做一个 symmetric G128 quantizer/pack/materialize CPU 门禁，再做一个最小
  Qwen3 block GPU smoke。不再做 W4A4 aware、QuaRot、MSE/w-clip 初始化或冻结
  RMSNorm 等与新决策相反的改造。

上述门禁通过后启动 15 组。

## TurboBOA 代码审计

### 已有能力

- `turboBOA/quantizers/realq_mse.py` 直接继承共享 `WeightQuantizer`，现有单测验证
  scale、zero、codes 与 QDQ byte-exact；
- grouped natural-column act-order 已保留 `column // 128` 的组归属；
- W2/W3/W4、G128、QuaRot、A/V-aware、post-RoPE K-aware、Qwen3 q_norm/k_norm
  mean-Jacobian KFAC 扩展、checkpoint 和统一 evaluator 均已存在；
- Llama/Qwen3 两种模型类型均由入口 fail-closed 限定；Qwen3-4B/32B 和 Llama
  小模型已有完整实跑证据。

### 需要补的门禁

1. **2026-08-21 已完成**：入口新增
   `--calib_tokens_file/--calib_tokens_sha256`；显式制品路径会校验 SHA、list 类型、
   256×2048、CPU int64 和逐条 shape，并完全绕过 cache/resample 路径；
2. **2026-08-21 已完成**：新增 formal deterministic 入口，固定
   Python/NumPy/Torch seed、deterministic algorithms、CUBLAS workspace、cuDNN
   benchmark/deterministic 与 TF32；`PYTHONHASHSEED` 不匹配时在加载模型前失败；
3. W4A4 必须显式传 `act_quant_aware_gptq=true` 和
   `k_cache_quant_aware_gptq=true`，不能使用当前 false 默认值；
4. `RealQMSEQuantizer` 当前默认 legacy Cartesian；可显式改为
   `symmetric_union_exact`，并以全 bit/G128 parity 测试证明不改变 scale/code；
5. F3 对 G128 的 refine 是本仓库扩展，不是 pristine upstream TurboBOA；正式方法必须标为
   `TurboBOA-C (RealQ-MSE, grouped-F3)`；
6. Qwen3-32B 历史纯算法在 block 43/44 的 down_proj 出现过灾难性范围扩张。禁止
   quality-driven fallback，但正式前必须逐 projection 记录 finite/range/scale gate；canary
   失败就报告算法失败，不能换成 GPTQ 权重；
7. **2026-08-21 已修复测试夹具**：
   `test_entrypoint_orders_reference_before_transform_and_eval_last` 现在显式 mock 后来新增的
   post-RoPE observer 安装；同批定向测试中新增的精确 token 制品测试已通过，首次回归的
   唯一失败就是该既知 mock 漂移，修复后还需在正式 Python 环境复跑完整文件并做 GPU
   smoke。

完成这些小改动和五模型 smoke 后，TurboBOA 可以最先铺开 20 组。

## YAQA_wclip 代码审计

### 已有能力

- `YAQA_wclip/lib/codebook/wclip.py` 实现 signed symmetric W2/W3/W4、natural-column
  G128、公共 `norm/grid/maxshrink`、legacy Cartesian 和 strict tie；现有 parity test
  覆盖 scale、codes、QDQ 与 short-tail；
- 因此本轮**直接使用这个已有 quantizer**，不再重写、不再引入第二份
  “类 RealQ”实现；启动前只做已有 parity 门禁；
- YAQA 保留 Sketch-B、power iteration=1、双侧 randomized Hadamard、anti-diagonal
  LDLQ 和三个 error-feedback term；
- A/K/V adapter 直接复用共享 `ActQuantizer`，支持 symmetric per-token clip=0.9，
  Hessian-aware 与 reload 后部署拓扑都有门禁；
- Llama/Qwen3 loader、Qwen3 q_norm/k_norm、Qwen 特殊 Hadamard 维度、dense HF
  checkpoint 和逐 projection 整数重构已经实现；Qwen3-4B/32B 有完整历史实跑。

### 2026-08-21 已完成的正式化改造

- `preflight_wclip.py` 已冻结五个主表模型各自的 WikiText2 256×2048 token 路径和
  SHA256，不再让 Qwen3-0.6B/Qwen3-8B/Llama-3.1-8B 回落到历史 Llama-3.2 合同；
- Hessian 生成与 raw quant 入口都增加同一套 deterministic contract：global seed=1、
  Python/NumPy/PyTorch seed=1、`PYTHONHASHSEED=1`、cuBLAS workspace、deterministic
  algorithms、cuDNN deterministic、TF32 关闭；原生双侧 RHT 的 sign seed 固定为
  `rotation_seed(0) + layer_index`。Hessian manifest 与量化 `quip_params` 都逐字段记录并
  互相校验；
- `lib/codebook/__init__.py` 已改为按需加载 `qtip_kernels`：dense `realq_wclip` 可以在
  无 QTIP 扩展的正式环境导入，只有 native packed-QTIP 路径真正实例化时才 fail-fast；
- raw manifest 新增 `weight_work_dtype` 与 `hessian_solver_dtype`。本轮明确记录 weight
  work tensor 为 FP32，而 Hin/Hout 的归一化、双侧 RHT、正则化与 block-LDL solver
  为 FP64，之后再转 FP32 进入 LDLQ；不再把它含糊描述成“全 FP32”。

### 剩余阻塞点

1. YAQA 没有全局 QuaRot/act-order，而是方法原生的双侧 RHT/anti-diagonal 顺序。若强加
   QuaRot，应另标 `YAQA-wclip-C+QuaRot`；建议主行保留原生变换并作为 complete-method
   差异披露；
2. W4A4 不做共享 QuaRot 的 QK Hadamard，`extra_qk_hadamard=false`。A/K/V 的 bit、
   groupsize、sym、clip 和 aware 仍可严格统一，但 transform 不是 solver-only 对齐；
3. 旧 audit/campaign 主要面向 Llama-3.2 两模型或 Qwen3-4B/32B 两设定，需新建五模型
   四设定 plan，不能在旧脚本里用别名冒充；
4. 32B YAQA_wclip 历史总 GPU-hour 很高，且旧 raw quant 使用 4 GPU。新主表要求每组
   单卡，必须先验证 world_size=1、batch=8、CPU offload 的完整一层与内存峰值，再估算 ETA。

### 正式适配要求

- 以 plan 表驱动五个模型的 token path/SHA、层数、projection shapes 和 Hessian 目录；
- dense w-clip 路径彻底绕开 QTIP CUDA kernel import；
- 单卡 Hessian 固定 `world_size=1, batch_size=8, iterations=32`，保持 256 条 token；
- A16 每模型生成一份 Hessian供三组共享；W4A4 每模型生成独立 aware Hessian；
- raw quant 固定 `ft_epochs=0, scale_override=1, V=1, td_x=1, td_y=128,
  sigma_reg=0.01`，不允许 skip、recovery FT、QTIP LUT 或结果 fallback；
- 每组保存 qweight/scale/SU/SV 和 dense weight，并逐 projection 验证整数重构及 reload
  deterministic smoke。

完成这些适配后可以跑 20 组；方法成本预计仍明显高于 TurboBOA 和 EfficientQAT。

## 建议执行顺序

1. 建立一份 55-run 只读 plan，冻结模型、适用设定、五份 token、量化器、随机种子、
   evaluator、计时边界和输出目录；生成 canonical plan fingerprint。
2. EfficientQAT 先按本文冻结的 15 组 weight-only 协议完成 deterministic/token
   hardening 并启动；TurboBOA 补 token/determinism/test hardening；YAQA 补 token 表与 lazy import。
3. CPU 单测必须全部通过；对每个算法做量化器 byte-exact、G128 natural-column、A/K/V
   topology、checkpoint reload 与 provenance 测试。
4. GPU smoke 覆盖五种模型结构；EfficientQAT 额外跑 Qwen3-0.6B W2 与
   Qwen3-32B W3 canary，不跑 W4A4 canary。Turbo 32B 必须检查历史异常层。
5. 每种方法选一个代表组 cold replay 两次，确认确定性；未通过则不启动其余正式组。
6. 门禁全部通过后再利用两台实验机并发正式 55 组。任何失败保留完整 artifact，不按
   结果切算法或 fallback。
7. 量化 checkpoint 通过验证后，统一补 KL/PPL、十项 QA、三项推理和 GPU-hour。
   大表保留 8 方法 × 20 设定的对齐视图，但 EfficientQAT 的五个 W4A4 单元格写
   `N/A`；实际新增有效数据行是 55，不把重复 W4 checkpoint 计成新实验。

## 本次静态与测试证据

- TurboBOA 完整 controlled-entrypoint 回归在修复测试夹具后为 `9 passed`，覆盖入口顺序、
  精确 token 路径/SHA、禁止重采样、固定任务面与 CLI 门禁；
- YAQA 完成 lazy import 后，无需注入空 `qtip_kernels` module，w-clip、provenance、hfize
  三个选定测试文件直接为 `17 passed`；dense w-clip 导入冒烟也验证
  `QTIP_KERNELS_AVAILABLE=false` 时量化器可正常构造；
- 五个模型的 hidden/intermediate/Q-O 输入宽度均通过 G128 整除静态检查；
- 历史实跑证明 TurboBOA 与 YAQA_wclip 已覆盖 Qwen3-4B/32B，EfficientQAT 已覆盖
  Qwen3-4B/32B；0.6B、8B 与本轮 Llama-3.1 仍必须按上述 plan 做正式 smoke。

## 统一评测 release 的跨 pod 文件身份修正（2026-08-21 14:47 CST）

五份旧 SDPA BF16 teacher cache 的最终发布在 node1 完成后，node1 全部门禁通过，node0
则全部被旧门禁拒绝。逐字段比对确认两机的 path、SHA-256、size、mtime_ns、ctime_ns 与
inode 完全相同，唯一差异是同一共享挂载在两个 pod mount namespace 中分别暴露
`st_dev=133` 与 `st_dev=22020131`。`st_dev` 因此不能作为跨 namespace 的 portable
identity。

修正后每个 consumer 仍严格核对 SHA release、size、mtime_ns、ctime_ns 和 inode；发布者
看到的 st_dev 继续写入 manifest 作为审计观察值，但不要求另一 mount namespace 数值相等。
针对该行为新增测试：仅 device 变化必须通过，inode 变化必须失败。统一 evaluator 合同测试
由 `7 passed` 更新为 `8 passed`。错误的 schema-2 manifest 已另存为
`reference_manifest.schema2_superseded_crosspod_stdev.json`，不会用于正式评测；修正后的
manifest 重新从五份 cache 内容做完整 SHA-256 发布，不沿用未验证的手工值。

最终 portable-stat reference manifest 于 14:49 CST 发布，SHA-256 为
`67645db80823219bf38ad24f43f69006e736ba37a216862b723421a4a2d54bb9`，内部 canonical
fingerprint 为 `3ecffa5d4cb97efb24aac4a668b6570d8e8193f4f4fa21f98825c2d9d238be05`。
node0 与 node1 分别调用同一 `load_reference_manifest()` 做完整 sources/reference 校验，
均返回该 fingerprint。14:54 CST，两机各启动 8 个动态统一评测 worker；node1 另启动唯一
official HumanEval+ CPU scorer。worker 先等待本 lane 的 TurboBOA 与 YAQA terminal，再持有
同一物理 GPU lock 执行评测，故提前安装不会占用量化显存或污染算法 GPU-hour。

## 统一评测 EvalPlus 运行路径修复（2026-08-21 15:47—15:55 CST）

第一条量化 lane 释放后，统一 suite 在进入模型加载前瞬时失败。完整 traceback 证明根因
不是 checkpoint：`importlib.metadata.version("evalplus")` 在正式 venv 中找不到包。历史
官方评分没有修改 venv，而是由隔离脚本把仓库固定的 EvalPlus 0.3.1 vendor 目录加入
`PYTHONPATH`；首轮 launcher 遗漏了这项运行时路径。

直接加入整个 vendor 根的第一次修复 canary 又被 fail-closed 拒绝：该根同时含旧版
`huggingface_hub==0.33.4`，会遮蔽正式 venv 的 0.36.2，不满足 Transformers 4.56.2 的
`>=0.34,<1.0` 合同。没有放宽版本门禁；改为建立只含 `evalplus/` 与
`evalplus-0.3.1.dist-info/` 两个只读 symlink 的隔离 Python path。修复后的运行时预检同时
得到 EvalPlus 0.3.1、Transformers 4.56.2、huggingface_hub 0.36.2、Torch
2.9.1+cu128，且完整 deterministic/CUDA 单卡门禁通过。

所有缺包/遮蔽失败 attempt 均按原因移动到
`_superseded_missing_evalplus_pythonpath_20260821/` 与
`_superseded_evalplus_full_vendor_shadow_20260821/`，没有删除或覆盖；它们发生在模型数值
评测前，不计入结果。16 个 GPU worker 随后用隔离路径重新启动；reference manifest、
冻结 evaluator 源码及其 SHA-256 均未修改。15:54 CST 的新 canary 已越过原 5—7 秒失败
点，成功加载 Qwen3-8B EfficientQAT checkpoint 和 teacher reference，正式 suite 继续运行。

### 统一评测恢复 claim 与机会式空槽门禁（2026-08-21 16:08—16:16 CST）

为插入受污染 TurboBOA GPU-hour 的干净重放，GPU7 evaluator 曾在首个 suite 已原子发布
`quality_result.json`、GSM8K 处于可恢复 partial 时被安全终止。16:08 CST 核对 claim
owner PID 27428 已死亡后，没有删除该目录，而是将其改名保留为
`.claim.interrupted_for_turboboa_clean_retime_20260821T0802Z`。重放 supervisor 在
16:11 自动启动的新 evaluator 随即以 attempt 2 claim 同一 suite；冻结 source SHA、
reference fingerprint 和已有 quality result 均未改变。

16:09 CST，node1/GPU2 对应的 TurboBOA/YAQA 全部 lane dependency 已 terminal，常驻
evaluator 自动开始 `efficientqat__EQ15-Q32-W2`。为了只利用“跨 lane Hessian 尚未就绪”
造成的真空槽，新增 orchestration-only 单-suite worker；它持有相同物理 GPU flock，调用
冻结的 `_run_one/run_suite`，不改变 evaluator、数据、seed 或数值代码，并在整条量化 lane
已就绪时主动跳过。脚本 SHA-256 为
`2351f0082ab6e20f51da65a7b6de17d204b2487cd4d6baf366d9797e12921339`；GPU2
check-only 正确判定 lane 已完成而不接管，GPU4 check-only 列出可评 suite 但因 YAQA 旧
worker 持锁没有越权并发。随后 GPU4 优先用于必须完成的 TurboBOA clean-retime，评测
机会式 worker 尚未执行正式 suite。

为避免 55 组结束后手抄数值，新增只读 release builder：逐 suite 复核 terminal/validation
SHA、共享 teacher fingerprint、Exact-KL 定义、十项 QA 任务集合、三项推理协议与覆盖、
EvalPlus official 164/164 覆盖，并把三类已证明 checkpoint/tensor 等价的干净重计时作为
显式 accounting override。脚本 SHA-256 为
`04771f115c400b440b54980991e06bb99c933143d38b83d1f419a16ad49b0d33`；16:24 CST
以 `--allow-incomplete` 门禁运行，正确报告 `0/55` 完整 suite、55 个缺失，不提前发布
半成品 release。正式模式会在任一 suite 或 official score 缺失时 fail closed。

16:29 CST，node1/GPU3 的 `YQ-Q8-W3/W2` 仍在等待 node0 的 `YH-Q8-A16`；两组 stage
均无 manifest，GPU3 为 0 MiB/0% 的真空槽，而 Hessian 尚余约 15 个 iteration。为避免
启动一个可能阻塞 YAQA 一小时以上的完整 reasoning suite，将机会式 worker 扩展为
`quality-only`：直接调用冻结 `run_suite._configure_runtime/_run_quality`，只预计算
Exact-KL/PPL 和十项 QA，原子发布标准 `quality_result.json` 后退出。冻结 evaluator 源码、
teacher、seed 和数值协议均未修改；扩展后 orchestration SHA-256 为
`d5cf336a4c0e99e12e979073e0dee095993af635c8510fbee458aa176b07630c`。

确认 node1/GPU3 的 YAQA worker PID 32595 只在等待、没有子进程和 stage manifest 后将其
TERM；16:29:41 CST，机会式 quality supervisor 开始
`efficientqat__EQ15-Q32-W4`。supervisor SHA-256 为
`7fbf7eb589edc70cb5a4f605043486a1aaf1939e596e16a69435412d0ee2ad40`；成功、失败或
lane 刚好变为 ready 三种退出均会重启同卡原 YAQA worker。常驻 evaluator 没有停止，
suite claim 与物理 flock 同时避免重复评测和并卡。

## MATH-500 scorer-only 恢复审计（2026-08-21 17:33—17:43 CST）

`efficientqat__EQ15-Q8-W4` 在 500/500 条 MATH-500 generation 全部持久化后，仅因运行
路径缺少 `math-verify` 而在 CPU scoring 失败。原失败 manifest/generation SHA-256 分别为
`be54a84e...da879c` 与 `c03339b7...c0a`；dataset SHA-256 为
`35dc4108...a06132`。隔离依赖目录只增加 math-verify 0.9.0、
latex2sympy2_extended 1.11.0 与 antlr4 runtime 4.13.2，未暴露 vendor 中不兼容的旧
huggingface_hub。两台 pod 的版本、模块路径、已知样例和 `cuda_initialized=false` 门禁
一致通过；现有 evaluator 的 `PYTHONPATH` 指向同一个目录，因此后续 MATH scoring 无需
重启 worker。

正式恢复源码 SHA-256=`ec0b0ffe9a94b22254e7006266cd0d1d12240db63ccff8ec0bd090f5e1d55488`。
它原子占用 suite claim，逐条校验 500 个 sample ID、顺序、chunk index/seed、
generation-config fingerprint、dataset 与 package 身份，随后以 `CUDA_VISIBLE_DEVICES=-1`
调用冻结的 `score_or_export`。结果为 `284/500=56.80%`，CPU scoring=`1.692278 s`；
generation SHA 在评分前后不变，原失败 manifest 以 hardlink 保留。正式 receipt 记录
`generation_reused_without_modification=true`、`model_loaded=false`、`cuda_used=false`，
完成 manifest/scores SHA-256 分别为 `408d994b...32ad` 与 `dde3c9c9...480f`。一次在发布前
发生的只读路径 validator 误判仅源于解析 symlink target，修正后重跑全门禁通过；没有
生成第三份候选，也没有把该只读误判记为正式 attempt。

17:45 CST 的 incomplete release 预检重新验证全部冻结 source、reference fingerprint 与
三份量化 plan，按预期只报告 `0/55` 完成且不写 output。17:48 CST 再次确认
node1/GPU4 仅由无子进程、无 CUDA context、未创建 `YQ-L8-W3/W2` manifest 的 YAQA
等待 worker 占锁。重复空窗 supervisor 改用经过字符门禁的唯一 `RUN_TAG` 日志名，避免
覆盖前次 relaunch 证据；脚本 `bash -n` 通过，SHA-256=`6932e8b5...4381`。释放该等待
worker 后，17:50:07 CST 机会式 claim `turboboa__TB20-Q8-W4`，仍只运行冻结 quality
入口，结束后自动恢复同卡 YAQA。

为最终总表避免手抄 55 行，新增 fail-closed table renderer，源码 SHA-256=
`f3783a3dc0e7ab83a529ce90bb67e90d682fa137fa01476afd7798839e568508`。它只接受
`status=complete, completed=55/55, missing=[]` 的 release，并严格要求原表恰有
20×5 行、新 release 恰有 EfficientQAT 15 + TurboBOA 20 + YAQA-wclip 20 行；五个
W4A4 单元显式写 N/A。它还逐行核对 comparison ID 对应的 model/setting、共享 reference
fingerprint 与三方法 GPU-hour 汇总。内存合成合同测试已通过，结果恰为 160 行、5 个
EfficientQAT N/A；正式 release 尚未完成，因此没有提前改写主表。
