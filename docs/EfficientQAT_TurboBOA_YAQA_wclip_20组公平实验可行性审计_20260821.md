# EfficientQAT / TurboBOA / YAQA_wclip：20 组公平实验可行性审计

> 审计日期：2026-08-21。目标矩阵与
> `REALQ_GPTAQ_GuidedQuant_四算法20设定公平总表_20260821.md` 相同：
> 5 个模型 × 4 个量化设定。本文只判断代码和实验协议是否具备正式铺开的条件；
> 尚未把未运行的数据写入结果大表。

## 结论

三种算法都可以在当前 20 组矩阵上做成“完整方法、公共量化表示受控”的公平实验，
但不能原样启动 60 组：

| 方法 | 当前可行性 | 正式铺开前必须完成的工作 | 建议正式标签 |
|---|---|---|---|
| EfficientQAT | 有条件可行，改造量中等 | 换成公共 MSE/w-clip 初始化；冻结 RMSNorm；W4A4 训练期 aware；固定正 scale 语义；接入公共 QuaRot；打开确定性门禁 | `EfficientQAT-C` |
| TurboBOA | 最接近可运行，改造量小 | 直接绑定五份主表 token 制品及 SHA；补统一确定性入口；修复一个过期 mock 单测；做 5 模型 smoke 和 Qwen3-32B 有限性 canary | `TurboBOA-C` |
| YAQA_wclip | 有条件可行，工程改造量中等 | 去掉旧模型/token 硬编码；dense w-clip 路径 lazy-import QTIP kernel；显式记录求解精度；补 0.6B/8B/Llama-3.1 门禁；单卡 smoke | `YAQA-wclip-C` |

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
  RedPajama、Alpaca、DEITA 或第二份 WikiText-2 抽样；
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
- scale 初值由同一 FP32 MSE/w-clip 定义产生：`norm=2.4`、`grid=50`、
  `maxshrink=0.5`、strict first-winner tie；
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

### 已有能力

- `EfficientQAT/quantize/quantizer.py` 已有 symmetric 分支：signed grid 通过固定 packed
  zero 表示，zero 不是 parameter；W2/W3/W4 与 G128 pack/materialize 已有测试；
- `experiments/efficientqat_compare/run_one.py` 能直接加载带 SHA 的精确 token artifact，
  逐 block 运行 Block-AP，再对所有 scale 做 E2E-QP；
- Qwen3 和 Llama 的七 projection topology、packed checkpoint 以及 dense BF16
  materialization 路径已经存在；Qwen3-4B/32B 也有历史成功实跑。

### 当前不公平或不安全的点

1. symmetric 初始化是 `max(abs(min), abs(max))/qmax`，不是公共 MSE/w-clip；
2. `set_weight_parameters()` 按参数名包含 `weight` 来选参数，会把 block 内两个 RMSNorm
   weight 一起训练并保存；
3. 现有 W4A4 是完成 Block-AP/E2E 后再挂 A/K/V，属于 unaware，和主表 aware 语义不同；
4. 当前正式 recipe 不做 QuaRot；若直接运行，只能作为“native transform 不同”的完整方法
   行，不能声称与主表的公共 QuaRot 受控；
5. runner 设置 `torch.backends.cudnn.benchmark=True`，没有开启完整 deterministic gate；
6. E2E-QP 直接训练 packed scale，历史上出现过 0/负 scale；虽有保值 recovery，但这会让
   “正 scale + 固定 zero”的公共量化表示变得含糊；
7. 旧 runner 只对 `W2A4KV4` 安装 post-training A/K/V，新的矩阵是 `W4A4KV4`，不能靠
   修改 run 名称复用。

### 正式适配要求

- 在 symmetric quantizer 中直接调用共享 `WeightQuantizer` 产生 MSE/w-clip 初始 scale；
- 参数白名单只允许七个 Linear FP weight 和各自 scale，显式断言 Norm/q_norm/k_norm
  均不在 optimizer；
- 使用 positive parameterization 或每步等价投影，禁止靠事后 signed-scale recovery 才
  获得可加载结果；
- 模型进入 Block-AP 前完成公共 LN fusion/QuaRot，rotation seed=0；
- 在 Block-AP target/input replay 和 E2E CE 两阶段都安装 W4A4 aware A/K/V；
- 保留 EfficientQAT 的两阶段训练和 learnable scale/weight 作为方法核心，但 LR、epoch、
  optimizer steps 必须写入每组 manifest；不使用验证集挑结果；
- 对 0.6B、Llama-3.1-8B、4B、8B、32B 各做一层 smoke，再做完整 Qwen3-0.6B
  W4A4 与 Qwen3-32B W3 canary。

完成以上适配后可以跑 20 组；原样代码不应直接写入大表。

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

1. 入口目前根据 cache_dir 和模型 identity 找 token；正式 20 组应增加
   `--calib_tokens_file/--calib_tokens_sha256`，直接绑定主表物理文件，禁止 cache miss
   重新生成；
2. 全局 reproducibility 入口不完整，需固定 Python/NumPy/Torch seed、deterministic
   algorithms、cuDNN benchmark 和 TF32 状态；
3. W4A4 必须显式传 `act_quant_aware_gptq=true` 和
   `k_cache_quant_aware_gptq=true`，不能使用当前 false 默认值；
4. `RealQMSEQuantizer` 当前默认 legacy Cartesian；可显式改为
   `symmetric_union_exact`，并以全 bit/G128 parity 测试证明不改变 scale/code；
5. F3 对 G128 的 refine 是本仓库扩展，不是 pristine upstream TurboBOA；正式方法必须标为
   `TurboBOA-C (RealQ-MSE, grouped-F3)`；
6. Qwen3-32B 历史纯算法在 block 43/44 的 down_proj 出现过灾难性范围扩张。禁止
   quality-driven fallback，但正式前必须逐 projection 记录 finite/range/scale gate；canary
   失败就报告算法失败，不能换成 GPTQ 权重；
7. 当前 `test_entrypoint_orders_reference_before_transform_and_eval_last` 没有 mock 后来新增的
   post-RoPE observer 安装，导致 1 个过期测试失败；这是测试夹具漂移，但正式代码门禁仍应
   先恢复全绿。

完成这些小改动和五模型 smoke 后，TurboBOA 可以最先铺开 20 组。

## YAQA_wclip 代码审计

### 已有能力

- `YAQA_wclip/lib/codebook/wclip.py` 实现 signed symmetric W2/W3/W4、natural-column
  G128、公共 `norm/grid/maxshrink`、legacy Cartesian 和 strict tie；现有 parity test
  覆盖 scale、codes、QDQ 与 short-tail；
- YAQA 保留 Sketch-B、power iteration=1、双侧 randomized Hadamard、anti-diagonal
  LDLQ 和三个 error-feedback term；
- A/K/V adapter 直接复用共享 `ActQuantizer`，支持 symmetric per-token clip=0.9，
  Hessian-aware 与 reload 后部署拓扑都有门禁；
- Llama/Qwen3 loader、Qwen3 q_norm/k_norm、Qwen 特殊 Hadamard 维度、dense HF
  checkpoint 和逐 projection 整数重构已经实现；Qwen3-4B/32B 有完整历史实跑。

### 当前阻塞点

1. `preflight_wclip.py` 只硬编码 Qwen3-4B/32B；其他模型会回落到旧
   Llama-3.2-1B token 合同，Qwen3-0.6B、Qwen3-8B 和 Llama-3.1-8B 不能直接正式跑；
2. `lib/codebook/__init__.py` 无条件导入 `qtip_kernels`，但 dense `realq_wclip` 路径不用
   低比特 kernel；当前两个本地 venv 都缺该扩展，导致测试收集失败。应做 lazy import，
   仅 native QTIP kernel 路径要求扩展；
3. CLI 的 `use_fp64=false` 只控制 weight tensor；`finetune.py` 仍把 Hin/Hout 转 FP64 做
   归一化、RHT、正则化与 block-LDL，然后转回 FP32。必须增加独立
   `hessian_solver_dtype` 字段并如实写 manifest，不能把当前路径描述成全 FP32；
4. YAQA 没有全局 QuaRot/act-order，而是方法原生的双侧 RHT/anti-diagonal 顺序。若强加
   QuaRot，应另标 `YAQA-wclip-C+QuaRot`；建议主行保留原生变换并作为 complete-method
   差异披露；
5. W4A4 不做共享 QuaRot 的 QK Hadamard，`extra_qk_hadamard=false`。A/K/V 的 bit、
   groupsize、sym、clip 和 aware 仍可严格统一，但 transform 不是 solver-only 对齐；
6. 旧 audit/campaign 主要面向 Llama-3.2 两模型或 Qwen3-4B/32B 两设定，需新建五模型
   四设定 plan，不能在旧脚本里用别名冒充；
7. 32B YAQA_wclip 历史总 GPU-hour 很高，且旧 raw quant 使用 4 GPU。新主表要求每组
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

1. 建立一份 60-run 只读 plan，冻结模型、四设定、五份 token、公共量化器、随机种子、
   evaluator、计时边界和输出目录；生成 canonical plan fingerprint。
2. 先完成 TurboBOA 的 token/determinism/test hardening；再完成 YAQA 的 token 表与
   lazy import；最后完成 EfficientQAT 的 quantizer/Norm/aware/QuaRot 改造。
3. CPU 单测必须全部通过；对每个算法做量化器 byte-exact、G128 natural-column、A/K/V
   topology、checkpoint reload 与 provenance 测试。
4. GPU smoke 覆盖五种模型结构；额外跑 Qwen3-0.6B W2、Llama W4A4、Qwen3-8B
   W4A4、Qwen3-32B W3/W4A4 canary。Turbo 32B 必须检查历史异常层。
5. 每种方法选一个代表组 cold replay 两次，确认确定性；未通过则不启动其余正式组。
6. 门禁全部通过后再利用两台实验机并发正式 60 组。任何失败保留完整 artifact，不按
   结果切算法或 fallback。
7. 量化 checkpoint 通过验证后，统一补 KL/PPL、十项 QA、三项推理和 GPU-hour，最后把
   三种方法各 20 行追加到大表，并把表头更新为 8 方法、160 行。

## 本次静态与测试证据

- TurboBOA + EfficientQAT 相关选定单测：`31 passed, 1 failed`；唯一失败是 TurboBOA
  入口顺序测试未 mock 新增的 `install_post_rope_observers()`，实际代码没有进入数值 run；
- YAQA 在当前裸 venv 中因缺 `qtip_kernels` 有 3 个 collection error，证明 lazy-import
  门禁确实需要修；
- 仅在测试进程注入不参与 dense w-clip 数学的空 kernel module 后，YAQA w-clip、
  provenance、hfize 选定测试为 `17 passed`；这不是正式运行方案，只用于隔离依赖问题；
- 五个模型的 hidden/intermediate/Q-O 输入宽度均通过 G128 整除静态检查；
- 历史实跑证明 TurboBOA 与 YAQA_wclip 已覆盖 Qwen3-4B/32B，EfficientQAT 已覆盖
  Qwen3-4B/32B；0.6B、8B 与本轮 Llama-3.1 仍必须按上述 plan 做正式 smoke。
