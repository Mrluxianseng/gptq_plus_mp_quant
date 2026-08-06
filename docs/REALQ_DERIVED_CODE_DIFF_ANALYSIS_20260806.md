# REAL-Q 主代码与衍生代码差异分析

> 分析日期：2026-08-06
>
> 主代码：`realq/`
>
> 衍生代码：`realq-plus/`、`realq_plus`、`realq_moe/`、`realq_benchmark/`、`../final_perf_integration_20260724/`

## 1. 结论先行

这几套代码虽然都从 `realq/` 衍生，但解决的问题并不相同：

| 目录 | 相对主代码的核心变化 | 本质定位 |
|---|---|---|
| `realq-plus/` | 把 Block-GD 从“只更新当前 linear 的未量化列”扩大为“更新当前 Transformer block、以及滑窗下一 block 中所有尚未量化的 linear”；另有可选 Lanczos Block-Hessian 层级补偿 | **量化算法变化** |
| `realq_plus` | 指向 `realq-plus` 的符号链接，没有第二份代码 | **Python 包名兼容入口** |
| `realq_moe/` | 增加 Qwen3-MoE 架构适配、自然路由统计、ragged expert Hessian、全 expert 锁步 GPTQ、joint/Jacobi Block-GD、packed grouped-mm 和全 GPU 驻留生命周期 | **模型结构扩展 + MoE 专用调度/性能优化** |
| `realq_benchmark/` | 在量化完成或加载量化 checkpoint 后，增加 GSM8K、MATH-500、HumanEval+、LiveCodeBench-lite 的生成、恢复、判分/导出和审计 | **评测能力扩展** |
| `../final_perf_integration_20260724/` | 引入 P01–P06/P10 量化热路径优化、严格正确性/性能证据链、可选严格 Block-GD 确定性；这些优化的大部分后来已合入当前 `realq/` | **性能集成与实验基础设施分支** |

最重要的关系是：

1. `realq-plus` 改了优化目标实际作用到哪些权重，输出轨迹与主代码不等价。
2. `realq_moe` 不是给 dense 模型加一个小补丁，而是为稀疏 expert 建立了独立的 Stage-0/Stage-1 数据结构和 runner。
3. `realq_benchmark` 基本不改 REAL-Q 的数学目标，主要把量化后评测扩成可复现的生成式 benchmark；但它复制自较早快照，量化内核落后于当前主代码。
4. `final_perf_integration_20260724` 是当前主代码多项默认性能实现的来源/证据归档。若严格按当前文件逐行比较，它反而缺少主代码后来新增的 Triton column-block、强制 cache-hit 和统一 token artifact 等能力。

## 2. 分析口径与代码快照

### 2.1 比较方法

本分析采用当前工作区的实际文件，而不是只比较 Git HEAD：

- 排除 `__pycache__`、`.pyc` 等生成物；
- 先比较文件集合；
- 再把 `realq_plus`、`realq_moe`、`realq_benchmark` 的包名前缀归一为 `realq`，区分“仅 namespace 变化”和“真实语义变化”；
- 对新增文件阅读数据结构、入口、更新时机和 fail-closed 条件；
- 对 `final_perf_integration_20260724` 同时阅读其 `realq/`、外层 `utils/`、`tools/`、`tests/` 和性能文档，因为该分支的优化实现并不全在 Python 包内部。

当前工作树并不干净，且几个衍生目录本身是未跟踪实验资产。因此本文描述的是 **2026-08-06 当前文件快照**，不应被理解为某个干净 commit 间的永久 diff。

代码规模如下：

| 代码树 | Python 文件 | Shell 文件 | Python 行数 | 说明 |
|---|---:|---:|---:|---|
| `realq/` | 33 | 2 | 7,332 | 当前 dense 主代码 |
| `realq-plus/` | 35 | 2 | 9,572 | 主代码副本 + full-block/层级 Hessian |
| `realq_moe/` | 50 | 2 | 18,818 | 含 8 个包内 MoE 测试文件 |
| `realq_benchmark/` | 44 | 2 | 9,342 | 含 9 个 benchmark 模块和 3 个测试/运行文件 |
| `final_perf.../realq/` | 32 | 2 | 7,335 | 性能 checkout 内的 REAL-Q 包 |
| `final_perf.../` 全仓 | 103 | 23 | 56,376 | 包含外层工具、测试、旧入口和文档配套代码 |

为方便后续判断工作区是否发生漂移，本次按“排序后的 `.py/.sh` 文件路径及内容 SHA256 列表”得到的树指纹为：

| 代码树 | 指纹 |
|---|---|
| `realq/` | `910be6cc53c1410cec50ac76d27b806459aea0175e84253390e56ea2eb22c82e` |
| `realq-plus/` | `c85adc22d4933e2da6607a0ddd8c6e5568e4a038718b46dd8185169db3de5fd0` |
| `realq_moe/` | `6aaede9aa65b8c5b9898659a7a0f5778fd08e12539ee97bbae11f1a29ba1fe60` |
| `realq_benchmark/` | `e984c62b66c69877b9b8a579e7947f991b7c6e99d23a2afb249e123fb347e8f2` |
| `final_perf.../realq/` | `977fdf01d4fa52d6a268de1ef8ebcca54eebb1286b2b3c7f7962b232e881a6f9` |

这些指纹包含相对路径，只用于识别本次快照，不能当作跨目录纯内容等价证明。

### 2.2 `realq-plus` 与 `realq_plus` 的关系

工作区实际关系是：

```text
realq_plus -> realq-plus
```

原因是 Python import/package 名不能含 `-`。代码物理目录叫 `realq-plus`，代码内部包名和启动入口使用 `realq_plus`。因此：

- 两者不是两个衍生算法；
- 修改任意一边看到的是同一文件；
- 正确启动形式是 `python -m realq_plus.ptq ...`；
- 统计 diff 时只应统计一次。

类似地，Plus 的层级 Hessian 依赖：

```text
realq_block_hessian -> realq-block-hessian
```

`realq-plus/quant/realq_layer.py` 和 `realq-plus/second_order/hierarchical_block_hessian.py` 会直接 import 这个包。即使层级开关默认关闭，当前模块顶层 import 仍要求该兼容入口存在。

## 3. `realq/` 主代码基线

### 3.1 总体流程

当前主代码的主流程位于 `realq/pipeline.py`：

```text
加载模型/可选加载量化 checkpoint
  -> 准备 KL/PPL reference（未 skip 时）
  -> fuse/rotate，安装 aware quant 所需 wrapper
  -> Stage 0：全模型端到端反传，预计算 saliency + aggregated full Fisher
  -> 捕获 layer-0 输入，维护 FP/student 两条 activation stream
  -> Stage 1：逐 Transformer layer、逐 module group、逐 linear 量化
  -> 可选保存 fake-quantized checkpoint
  -> KL/PPL 与 lm-eval
```

核心不是普通 GPTQ 的纯二阶闭式更新，而是两种机制交替：

1. **静态 saliency-weighted GPTQ**：对当前 linear 累积输入 Hessian，按列或列块量化，并用 Hessian inverse 对 block 内和 block 外的后续列补偿。
2. **动态 Block-GD**：每个非末尾 column block 完成后，用端到端预计算的 Fisher MSE surrogate 反向一次，以 Adam 更新当前 linear 尚未量化的 trailing columns。

最后一个 Transformer block 不再使用 Fisher surrogate，而是经过 final norm + LM head 对 teacher/student logits 计算真实 KL。

### 3.2 Stage 0：saliency 与 Fisher

关键文件：

| 文件 | 职责 |
|---|---|
| `precompute/static_e2e.py` | 分片校准数据、模型端到端 backward、cache 读写、返回 `StaticStats` |
| `precompute/hooks.py` | 对各 linear 输出收集分组 saliency，对 block 输出累积 Fisher outer product |
| `precompute/cache.py` | 构建身份相关 cache key，按 rank 原子保存/加载 |
| `precompute/labels.py` | 与 batch/world-size 解耦的确定性 categorical labels |

主线的 `StaticStats` 包含：

- `saliency[layer][module]`：每个 calibration sample/token、每个输出 row group 的梯度平方范数；
- `fisher[layer]`：block hidden 维度上的完整、非对角 aggregated empirical Fisher。

虽然历史配置名仍叫 `fisher_diag_mse`，当前 `refresh/fisher_loss.py` 使用的是完整二次型：

\[
\frac{1}{2}\Delta Y^\top F\Delta Y.
\]

### 3.3 Stage 1：逐层/逐线性层量化

关键文件：

| 文件 | 职责 |
|---|---|
| `runner/layer_loop.py` | 按 `q/k/v -> o -> up/gate -> down` 组织量化、Hessian 累积、refresh 和最终 replay |
| `runner/module_groups.py` | 固定 module group 顺序 |
| `runner/streams.py` | 捕获并推进 student `inps` 与 FP `fp_inps` |
| `quant/realq_layer.py` | `RealQLayer`：Hessian、act-order、量化参数、GPTQ block loop、refresh 接口 |
| `quant/hessian.py` | 带 damp fallback 的 Cholesky inverse |
| `quant/triton_column_block.py` | 单卡 column-block 内逐列量化与补偿融合 kernel |
| `refresh/block_gd.py` | 非末层 Fisher-MSE Block-GD |
| `refresh/kl_loss.py` | 末层真实 KL Block-GD |

主代码中一次 refresh 的更新域是：

```text
当前 linear 的当前 stitched FP32 weight
  ├─ 已量化 prefix：锁定
  └─ trailing columns：求梯度并做 Adam update
```

即使 forward/backward 经过整个 Transformer block，`functional_call` 只 override 当前 linear 的一个 weight，`autograd.grad` 也只对这个 weight leaf 求导。这正是 Plus 要改变的基线语义。

### 3.4 主代码当前已有的性能实现

当前 `Config` 有 87 个字段，并默认开启：

- P01 `quantizer_inner_fastpath=true`；
- P02 `w_clip_search_impl=symmetric_union_exact`；
- P03 `w_clip_update_impl=where_out`；
- P04 `w_group_param_layout=compact`；
- P05 `fisher_fp32_cache=true`；
- P06 `act_order_stitch_impl=prefix_q_trailing_w_exact`；
- P10 `prepared_clamp_bound_cache=true`；
- `triton_column_block=true`。

这很重要：分析 `final_perf_integration_20260724` 时，不能把已经进入当前主线的 P01–P06/P10 再描述为“相对当前主代码独有的新功能”。它们是该历史性能分支的主要成果，但当前差异更多是证据工具和版本先后差异。

## 4. `realq-plus/`：全 Block-GD 与层级 Block-Hessian

### 4.1 文件级 diff

把包名 `realq_plus` 归一成 `realq` 后，绝大多数复制文件与主代码完全一致。真正有语义变化的公共文件只有：

| 文件 | 语义变化 |
|---|---|
| `alignment.py` | trace schema 3→4；新增 backward invocation、chunk、global count 和 active-weight 审计 |
| `config.py` | 新增 11 个 hierarchical Block-Hessian 配置与约束 |
| `quant/realq_layer.py` | 接收跨 linear 累积的 FP32 master；支持复用 provisional local Hessian state；接入层级补偿器 |
| `refresh/block_gd.py` | 新增 `BlockRefreshState`，实现 full-block 多 weight backward/Adam |
| `refresh/kl_loss.py` | 最终层 KL 同样扩成 full-block 更新 |
| `runner/layer_loop.py` | 创建、跨 layer 保存、移交和释放 `BlockRefreshState`；可准备 hierarchical Hessian |
| `refresh/__init__.py` | 导出新的 full-block 状态/API |

新增文件只有：

- `second_order/__init__.py`
- `second_order/hierarchical_block_hessian.py`

其余公共文件归一包名后均一致，包括 Stage-0、FSDP、CPU master、Triton kernel、checkpoint/eval 接线等。因此 Plus 是一个相对聚焦的算法 fork，而不是整套 runtime 重写。

### 4.2 Full-block Block-GD 的准确语义

主代码和 Plus 的关键差别如下：

| 项目 | 主 `realq` | `realq-plus` |
|---|---|---|
| 当前 linear | 只更新 trailing columns | 同样只更新 trailing columns |
| 当前 block 后续 linears | 不参与梯度/更新 | 整个 weight 参与梯度并更新 |
| 当前 block 已完成 linears | 固定 | 固定 |
| slide-window 下一 block | 只作为 loss 的 forward graph | 下一 block 所有未量化 linears 也参与梯度/更新 |
| Adam state | 每个当前 linear 创建/使用 | 每个待量化 linear 从成为 future weight 起持续保存 |
| FP32 master | 当前 linear 的临时 working weight | 每个 future linear 都有持久 FP32 master |
| 下一 linear 开始 GPTQ 时 | 从 module 当前 BF16/FP weight clone 开始 | 接管此前所有 full-block Adam 更新后的 FP32 master |

`BlockRefreshState` 为一个 Transformer block 的量化序列建立 `_WeightRefreshState`：

- `master`：future weight 的 FP32 主副本；
- `exp_avg` / `exp_avg_sq`：持久 Adam moments；
- `step`：该 weight 已经历的更新步数；
- `status`：future、current、finished 的生命周期；
- `parameter_name` / storage identity：确保 backward 使用和被写入的是同一存储。

一次 refresh 的实际过程是：

1. 当前 linear 把“已量化 prefix + working suffix”拼成 FP32 candidate。
2. `BlockRefreshState.make_overrides()` 为所有 active weights 生成 leaf：
   - 当前 linear 只开放 trailing columns；
   - 当前 block future linears 开放完整矩阵；
   - slide alpha `< 1` 时，下一 block future linears 也开放完整矩阵。
3. 一次 `functional_call` 重放当前 block；需要 slide 时再重放下一 block。
4. 一次 `torch.autograd.grad(loss, leaves)` 同时得到所有 active weights 的梯度。
5. 分布式时把所有梯度、used flag、sample count 和可选 loss sums 一起 SUM all-reduce。
6. 每个 weight 用自己的持久 step/moments 做 Adam：
   - 当前 weight 的 update 返回给 `RealQLayer`，只减到 trailing columns；
   - future weight 的 update 直接减到其 FP32 master。
7. 当前 linear 完成后 `finish_quantization()` 锁定并释放其 optimizer state；block 完成时 `assert_complete()` 检查没有遗留 active weight。

这样做的功能目的，是让一次已经覆盖完整 block 的梯度图真正更新图中的全部可量化权重，给后续尚未离散化的 linears 更多自由度来吸收前面量化产生的误差。

### 4.3 最后一层 KL 也扩大更新域

主代码 `refresh/kl_loss.py` 只对当前 linear 求最终 logits KL 梯度。Plus 版本使用同一 `BlockRefreshState`：

- 仍经过 final norm + lm head 计算 KL；
- 当前 linear 只开放 suffix；
- 最后一个 Transformer block 内未来 linears 开放完整矩阵；
- 一次 backward 后统一更新。

因此 Plus 的 full-block 语义不仅作用于 Fisher surrogate，也作用于最后一层真实 KL。

### 4.4 Trace 与审计增强

`alignment.py` 的 schema 从 3 升到 4，并兼容读取 schema 3。新增信息包括：

- `objective`：如 `fisher_mse_full_block`、`kl_full_block`；
- `backward_invocation_id`；
- `backward_chunk_sizes`、`backward_bsz`、`global_count`；
- 每个 active weight 的 scope、参数名、是否 current、active column 数；
- 更新前后 optimizer step；
- storage identity；
- update norm、source norm 和是否真实写回。

这不是量化质量机制，但它解决了一个重要的可验证性问题：仅看到 loss/backward 发生，不能证明 future weights 真正进入了 autograd，也不能证明 Adam 更新落在同一 FP32 master 上。schema 4 给出了逐 backward 的运行时证据。

### 4.5 可选 hierarchical Block-Hessian

Plus 还增加了第二层、默认关闭的实验路径：

```text
block_hessian_hierarchical=false  # 默认
```

相关配置包括样本数、HVP micro-batch、Lanczos steps/rank、seed、finite-difference fallback、FD scale 和 trace path。

该路径在每个 Transformer block 正式量化前：

1. 对每个 linear 做 provisional local GPTQ，冻结/保存 `LocalHessianState`；
2. 在 FP32 master reference point 上构造 block HVP operator；
3. 使用 exact forward-AD JVP + reverse VJP 做 Lanczos，保留正且有限的低秩谱模；
4. 丢弃 provisional 离散权重，恢复 live FP32 masters；
5. 正式量化复用 provisional local Hessian state，避免重复累积；
6. column block 完成后提交谱补偿；
7. 一个完整 linear 完成后，根据完整残差 `Q - W_entry` 联合更新所有 future linears；
8. 每个 column-block 的二阶提交之后，再独立执行原 Plus Block-GD backward/Adam。Lanczos 与 Block-GD 不共享梯度或计算图。

其跨 linear 代理可概括为：对当前已完成 linear 的残差

\[
E_l=Q_l-W_{l,\mathrm{entry}},\qquad c_l=U_l^\top E_l,
\]

对 future 集合构造

\[
Z_F=B_{FF}^{-1}U_F,\qquad
M_F=\Lambda^{-1}+U_F^\top Z_F,
\]

并一次写入

\[
\Delta W_F=-Z_FM_F^{-1}c_l.
\]

这里 `B_FF` 是各 future linear 的 local GPTQ 曲率组成的 block diagonal，`UΛUᵀ` 是 Lanczos 低秩谱项。它是代理模型下的联合条件补偿，不应描述为真实端到端 KL Hessian 的精确解。

### 4.6 Hierarchical 路径的硬限制

`Config.__post_init__` 和运行时共同限制：

- 必须 `grad_lr > 0`，即必须与 Plus Block-GD 组合；
- 必须 `group_parallel_quant='rank'`；
- 必须 `num_groups > 1`；
- 当前只允许单 rank/单 GPU；
- 必须关闭 `triton_column_block`；
- 正式路径禁止 finite-difference fallback，只接受 exact forward-AD；
- `block_hessian_rank <= block_hessian_steps`；
- Hessian 样本数不能超过校准样本数。

因此 hierarchical 不是主代码 Triton 快路径上的免费增强，而是一条成本更高、约束更强的研究路径。

### 4.7 已有实验反映的功能收益与边界

仓库已有正式记录显示，单独的 full-block Plus 在 Llama-3.2-1B/3B-Instruct、W3A16/W2A16 四组中：

- exact KL 相对原 REAL-Q 降低 `10.89%–21.83%`；
- PPL 变化 `-0.92%–-16.04%`；
- QA 平均变化 `-0.02–+2.70 pp`；
- 量化 GPU-hour 变化 `-0.24%–+7.61%`。

这与代码目标一致：扩大更新自由度主要改善量化误差/KL，而没有数量级成本增长。

但 hierarchical Lanczos 组合并未稳定优于普通 Plus：已有四组组合实验的 KL 均略差，且成本增加约 `14.86%–18.97%`。进一步的真实端到端 KL Hessian cosine 诊断表明，边界代理 Lanczos 与真实 KL Hessian 的相关性较低。因此：

- **full-block Block-GD 是 Plus 的主要有效功能**；
- **hierarchical Block-Hessian 是保留的实验分支，不适合作为默认推荐**。

## 5. `realq_moe/`：Qwen3-MoE 支持与全 expert 联合优化

### 5.1 为什么不能直接复用 dense runner

Dense REAL-Q 假设每个 linear 对全部 `(sample, token)` 产生规则三维激活，Hessian 和 saliency 可以按固定 batch/token 形状对齐。

MoE expert 的输入是 router 产生的 ragged assignments：

- 不同 expert 命中的 token 数不同；
- 同一 expert 在不同 batch/rank 的命中数不同；
- 某个 expert 可能在 calibration 中完全没有路由；
- 逐 expert 重新跑完整层会重复 router、attention、其他 experts 和下一层，成本按 expert 数放大。

所以 `realq_moe` 同时改变了统计表示、Hessian 累积、量化调度和 Block-GD replay。

### 5.2 文件级 diff

归一包名后，公共文件中有真实语义变化的包括：

| 文件 | 主要变化 |
|---|---|
| `akv.py` | rotate/aware wrapper 后修复 MoE down projection wrapper |
| `config.py` | 新增 11 个 MoE policy/lifecycle 配置及 fail-closed 校验 |
| `pipeline.py` | 稀疏架构探测、整模型 GPU residency、GPU-native KL/PPL、禁 offload 生命周期 |
| `precompute/cache.py` | schema 5→8，保存/验证 routes、counts、coverage，支持 map 到 CUDA |
| `precompute/hooks.py` | 对 ragged expert 输出收集 assignment-aligned saliency |
| `precompute/static_e2e.py` | 路由捕获、packed saliency finalize、coverage gate、GPU-resident `StaticStats` |
| `quant/realq_layer.py` | dense saliency 也保持 GPU resident；Triton 从主 `realq` 包导入 |
| `runner/layer_loop.py` | dense attention + sparse expert 分流、静态统计逐层释放、MoE trace |
| `runner/streams.py` | GPU-resident layer-0 capture，token shard 一次上传，拒绝 CPU manager |

新增的核心模块为：

| 新文件 | 职责 |
|---|---|
| `model_adapter.py` | Qwen3-MoE 架构识别、dense/sparse layer plan、expert projection 路径、wrapper 修复 |
| `precompute/routed_stats.py` | 自然路由 capture、expert-major CSR、saliency clipping、全局 coverage |
| `quant/routed_realq_layer.py` | ragged expert Hessian 累积与 zero-route RTN Hessian |
| `quant/joint_column_quant.py` | 全 experts 同 projection 同 column block 锁步 GPTQ/Triton stepper |
| `refresh/fixed_route_moe.py` | packed fixed-route grouped-mm replay |
| `refresh/joint_batched_adam.py` | `(E,R,C)` 连续 Adam moments 与批量 suffix update |
| `refresh/joint_moe_block_gd.py` | 一次 all-expert forward/backward、Jacobi commit、full-slide |
| `refresh/moe_block_gd.py` | 逐 expert contribution decomposition，保留为 serial correctness oracle |
| `runner/moe_layer_loop.py` | sparse MLP capture、coverage、三 projection 联合量化主循环 |
| `utils/module_capture.py` | 捕获 wrapper 内部 Linear 输入，并在 GEMM 前提前终止 |

包内还新增 8 个聚焦测试，覆盖 fixed-route、GPU residency、joint Adam、joint GPTQ、joint runner、dense lock、runtime numerics 等。

### 5.3 架构适配范围

`model_adapter.py` 当前显式支持：

```text
Qwen3MoeForCausalLM
```

目标是 Transformers 4.56.2 的未 fused Hugging Face Qwen3-MoE 结构，要求：

- `layer.mlp.experts` 是 `nn.ModuleList`；
- `layer.mlp.gate` 是 router linear；
- `mlp.num_experts`、`mlp.top_k` 合法；
- expert 有 `up_proj`、`gate_proj`、`down_proj`；
- 被 activation wrapper 包裹时能解析到底层 `nn.Linear`。

Sparse layer 的 attention 仍沿用 dense REAL-Q 的 `q/k/v -> o` 分组；expert MLP 不再走 dense `up/gate -> down` module group，而由 MoE runner 接管。Dense layer 仍可走原有分组，因此 adapter 本身支持模型内 dense/sparse layer 的区分；正式运行契约主要针对全 sparse Qwen3-MoE 目标。

### 5.4 Stage 0：自然路由、ragged saliency 与 coverage

`RouteCaptureManager` 在 router 和 expert 前后挂 hook，按 Qwen3-MoE 原生顺序记录：

- `expert_offsets`；
- `flat_token_indices`；
- `topk_slots`；
- `route_weights`。

`PackedRouteBatch`/`PackedLayerRoutes` 使用 expert-major CSR。每个 expert 内保持“top-k slot 优先，再按 flattened sample/token 顺序”，多 batch 按 forward 顺序追加。四个主要 tensor 留在 router 的 CUDA device，避免 Stage-0 热路径产生 `3 × num_experts` 小对象和 D2H。

MoE `StaticStats` 在主线 `saliency + fisher` 之外增加：

- `routes[layer]`；
- `expert_global_assignment_counts[layer]`；
- `expert_global_coverage[layer]`，含 assignment、unique token、unique sample、affinity mass。

数据形状也不同：

- dense module saliency：`(N_local, T, G)`；
- routed expert saliency：`(A_local, 1, G)`，其中 `A_local` 是本 rank 的真实 assignment 数；
- block Fisher 仍是 `(H,H)` 全局统计。

### 5.5 Coverage gate 与 zero-route RTN

代码把两种“冷 expert”明确区分：

1. **正 assignment 但低于阈值**：fail closed。不能用不足统计悄悄构造 Hessian。
2. **全局 assignment 恰好为 0**：没有任何 routed Hessian 可估，走显式 `moe_zero_route_fallback='rtn'`。

对 zero-route expert：

- 三个 projection 使用相同 W-bit、对称、grouped、MSE clipping 的 direct RTN；
- `RoutedRealQLayer.finalize_rtn_fallback()` 使用单位 Hessian，使 GPTQ 补偿退化为逐列独立 rounding；
- joint Block-GD 对该 expert lane 的 update 强制为零；
- teacher-zero 与 student-zero 取并集；
- 不伪造 token、不复制别的 expert Hessian、不改变 router。

配置名 `moe_fail_on_teacher_cold/student_cold` 仍被强制为 true，表示对“不满足 coverage 的正样本 expert”保持 fail-closed；exact zero-route 是单独、显式且可审计的 RTN 例外。

### 5.6 Stage 1：expert 锁步 GPTQ

`JointMoeProjectionStepper` 把调度从逐 expert 改为逐 projection：

```text
所有 experts 的 up_proj
  -> 所有 experts 的 gate_proj
  -> 所有 experts 的 down_proj
```

在每个 projection 内：

1. 为每个 expert 保留独立 Hessian、quantizer、scale/zero 和 working weight；
2. 将所有 experts 推进到相同 column-block boundary；
3. 把 expert×row 展平成联合 batch；
4. Cholesky、clip observer、逐列扫描、误差补偿和 writeback 尽可能批量执行；
5. 单卡且 fastpath 可用时调用 Triton column-block kernel；
6. 非末尾 block 产生一个 `JointProjectionBoundary`，等待一次 joint Block-GD update 后继续。

这是“数值状态独立、执行调度联合”：没有让不同 expert 共享 Hessian 或量化参数，只是消除 Python/launch/replay 的 expert 维重复。

### 5.7 Joint/Jacobi Block-GD

主执行路径 `make_joint_moe_block_gd_refresh_fn` 的语义是：

- 同一 projection 的所有 expert weights 在同一 candidate 状态上求梯度；
- 一次 `functional_call`/一次 tuple-valued `autograd.grad` 得到所有 expert 梯度；
- 所有 expert 的 Adam update 同步提交，是 Jacobi update；
- 不是逐 expert 更新后让下一个 expert 看见新状态的 Gauss-Seidel 轨迹；
- 所有 expert 共享一次 scheduler 取样并同步推进 Adam step，包括 unused/zero-gradient expert；
- `backward_samples == backward_bsz` 时，每个 joint block 只需一次 backward。

当前层 fast path 不必重跑 attention：runner 捕获 post-attention residual 和 post-attention-layernorm MLP input，只重放完整 sparse MLP，再加 residual。另保留完整当前层 replay 作为 oracle。

`loss_slide_window=true` 时：

- 当前层可用固定 student route 做 packed replay；
- 当前 candidate 输出直接进入下一层；
- 下一层执行完整自然 forward，router 会重新计算；
- 因此 full-slide 没有错误复用下一层旧 route。

对目标 Qwen3-30B-A3B 的 128 experts、列宽配置，正式记录中 expert refresh 从逐 expert 的数量级降到每层 `15 + 15 + 5 = 35` 次，而不是 `35 × 128`。

### 5.8 Packed fixed-route grouped-mm

当前层每个 column block 都重新执行 router/top-k/one-hot/expert discovery 浪费很大。`fixed_route_moe.py` 将已捕获 student assignments 重排为 expert-major、sample-major compact rows：

- payload 只有真实 `A` 行，不构造 `[E,max_tokens,*]` padding；
- selected sample 的真实 assignment 在 GPU 上生成 cumulative `offs`；
- cold expert 用重复 offset 表示；
- `torch._grouped_mm` 一次计算所有 active experts；
- 一次 expert-major `index_add_` 混回 token hidden states；
- candidate 是单一 `(E,R,C)` leaf，便于一次 backward 和批量 Adam。

Up/gate projection 还缓存不变的另一路乘法因子；down projection 捕获 wrapper 内部 Linear 真正看到的输入，支持 identity 和 formal `online_full_had` 变换。

实现对 grouped-mm 能力 fail closed：不能只检查 `hasattr(torch, '_grouped_mm')`，因为 unsupported shape/dtype 可能进入带 host offset 的 fallback，违反无 D2H 和性能契约。

### 5.9 GPU-resident 生命周期

MoE 正式路径强制：

- 单 GPU、`world_size=1`；
- 模型、校准 inputs、routes、saliency、Fisher、H/Hinv、W/Q、Adam、grad、slide graph 全部在 CUDA；
- 拒绝 `cpu_master=true`；
- 拒绝 `fsdp_cpu_offload=true`；
- 拒绝当前会在 Stage-0 后 `reload_on_cpu` 的 FSDP 路径；
- layer-0 token shard 一次上传，不在 per-sample loop 反复 `.to(cuda)`；
- 每个 MoE layer 完成后立即释放该层已消费的 saliency/Fisher/routes/counts/coverage，降低后续峰值；
- GPU-native KL/PPL 避免调用主线的 CPU layer streamer。

这种设计以速度和避免重复搬运为优先，但代价是非常高的单卡显存需求。已有 Qwen3-30B-A3B W4A16、256×2048 完整 48 层运行记录为：

- precompute `1518.517 s`；
- quantization `7734.592 s`；
- 算法合计 `9253.110 s`；
- 200 ms `nvidia-smi` 已观测峰值 `182612 MiB`；
- 19 个 layer 共 27 个 `(layer, expert)` RTN fallback；
- 每层 joint refresh 均为预期 `35/35`。

该次运行 `skip_eval=true`，只能证明完整量化执行、显存和调度，不能据此报告精度。

### 5.10 与主代码的耦合和限制

`realq_moe` 并非完全自包含：

- `realq_moe/quant/realq_layer.py` 和 `quant/joint_column_quant.py` 直接从 `realq.quant.triton_column_block` 导入 Triton kernel；
- 全部包仍依赖仓库外层共享 `utils.*`；
- 因此主 `realq` 的 Triton 文件或外层 quantizer API 变化可能影响 MoE，即使 `realq_moe/` 自身未改。

其他限制：

- 当前只明确支持 Qwen3-MoE unfused 结构；
- 正式 joint path 是单卡；
- 需要极大显存；
- serial expert path 只是 correctness/quality oracle，不应被静默选为正式性能路径；
- dense attention 仍走主线语义，但 saliency 生命周期改为 GPU resident；
- cache schema 与 dense 主线不兼容，不能混用。

## 6. `realq_benchmark/`：生成式推理评测扩展

### 6.1 量化算法改动很少

归一包名后，公共文件只有 4 个有语义差异：

| 文件 | 变化 |
|---|---|
| `config.py` | 新增 21 个 reasoning 配置和 `skip_kl_ppl_eval`；性能默认值保留较早的 legacy 选择 |
| `pipeline.py` | 在量化或 checkpoint load 后接入 reasoning eval；允许只跳过 KL/PPL |
| `quant/realq_layer.py` | 来自较早快照：没有当前 P10 runtime audit/fallback 和 Triton column-block |
| `runner/layer_loop.py` | 不向 `RealQLayer.quantize` 传 P10/Triton 参数 |

其他公共文件在包名归一后等价。新增 9 个 `benchmarks/*.py` 模块、README、依赖文件和 3 个测试/真实 GPU 运行文件。

因此它的主要目的不是改量化质量，而是在同一个模型对象上建立量化后生成评测。

### 6.2 Pipeline 接入位置

新增流程是：

```text
加载基础模型或量化 checkpoint
  -> 可选执行 RealQ 量化
  -> 可选保存 checkpoint
  -> 可选 KL/PPL
  -> 销毁 torch distributed process group
  -> rank 0 将模型 dispatch 到可见 GPU
  -> 可选 lm-eval
  -> 可选 reasoning benchmark
```

开关语义：

- `skip_eval=true`：关闭 KL/PPL、lm-eval 和 reasoning 全部评测；
- `skip_kl_ppl_eval=true`：只跳过 WikiText KL/PPL，lm-eval/reasoning 仍可运行；
- `reasoning_eval=true`：运行配置中的 reasoning tasks；
- `load_qmodel_path`：可跳过重复量化，单独评测已保存 checkpoint。

### 6.3 支持的四项任务

| 任务 | 数据/Prompt | 生成后处理 | 当前判分 |
|---|---|---|---|
| GSM8K | 固定 zero-shot 指令 | 提取 final answer/最后数值 | Fraction 数值等价，`pass_at_1` |
| MATH-500 | 固定 zero-shot 数学指令 | 保留模型文本 | Math-Verify 0.9.0 符号等价 |
| HumanEval+ | 函数签名/题目 prompt | 提取 Python code，去除 thinking 文本 | 导出 EvalPlus samples，默认不执行 |
| LiveCodeBench-lite | pinned release 的题目/starter code | 导出 custom evaluator JSON | 在外部沙箱调用固定源码 evaluator |

统一协议名是 `realq_zero_shot_v1`：

- 默认使用 tokenizer chat template；
- Qwen3 默认 `enable_thinking=true`；
- 默认 temperature `0.6`、top-p `0.95`、top-k `20`；
- 默认每题一次生成；
- 多样本只额外报告 `pass_at_n_oracle`，不冒充无偏 pass@k；
- LiveCodeBench 拒绝 `release_latest`，要求固定如 `release_v6`。

### 6.4 离线数据与可复现性

`benchmarks/data.py` 优先读取本地 materialized 数据；没有本地文件时才尝试 Hugging Face。Canoe 无网环境必须先在宿主机运行 `prepare_data.py`。

数据准备会：

- 固化 GSM8K/MATH-500/HumanEval+/LCB release；
- 记录文件 SHA256；
- 把 LCB 拆成小型 prompt 文件和完整 official-test 文件，避免每次生成解析数 GB 隐藏测试；
- 对 LCB official evaluator 使用 pinned source checkout，而不是会缺 evaluator 模块的普通 wheel。

`manifest.json` 记录：

- 模型/checkpoint 与 W/A/K/V 配置；
- 数据 SHA256；
- chat template SHA256；
- thinking、sampling、max tokens、batch size、seed、n；
- 依赖版本；
- 逐任务耗时和状态。

### 6.5 断点恢复

`generation.py` 每个 batch 追加并 flush/fsync `generations.jsonl`。恢复 key 是 `(task, sample_id, sample_index)`，但是否复用还受 generation fingerprint 约束，fingerprint 覆盖：

- prompt/chat/template；
- 模型/checkpoint；
- 量化设置；
- thinking 和 sampling 参数；
- max tokens、batch size、num samples；
- seed 和协议版本。

损坏的最后一行会被忽略；重复 key 以最后一条完整记录为准。参数变化不会误复用旧答案。

### 6.6 代码评测安全边界

HumanEval+ 与 LCB 默认只导出候选，不在量化/GPU job 内执行模型生成代码。

`official_eval.py` 要求显式：

```text
--i-understand-generated-code-will-run
```

但这个 flag 只是防误触，不是沙箱。正确使用仍要求外部隔离容器/VM、禁网、资源限制、无敏感可写挂载。

`lcb_local_eval.py` 还要求环境变量 `REALQ_ALLOW_UNTRUSTED_CODE=1`，并把本地 pinned release 注入官方 evaluator，避免其联网或回退到 `release_latest`。

### 6.7 相对当前主代码的版本滞后

`realq_benchmark` 的 Config 有 106 个字段，但量化内核是较早快照：

- `quantizer_inner_fastpath=false`；
- `w_clip_search_impl=cartesian_legacy`；
- `fisher_fp32_cache=false`；
- `act_order_stitch_impl=full_weight_legacy`；
- `w_clip_update_impl=guarded`；
- `w_group_param_layout=expanded`；
- 没有 `prepared_clamp_bound_cache`；
- 没有 `triton_column_block.py`，runner 也不传 Triton 开关。

所以 benchmark 包适合复现它建立时的评测闭环，但不应直接被当作“当前最快/最新的 REAL-Q + benchmark”。如果要用于新的主线实验，更合理的做法是把 `benchmarks/` 和 pipeline 的 reasoning hook 前移到当前 `realq/`，而不是长期维护一份旧量化 fork。

另一个边界是 checkpoint 仍保存 fake-quantized BF16 weights 和 runtime manifest，不是压缩成 2/3/4-bit 的部署格式；推理显存不能按真实低比特 packed 权重估算。

## 7. `../final_perf_integration_20260724/`：性能优化与证据链

### 7.1 目录性质

该目录不是一个与 `realq-plus` 类似的单包副本，而是一份完整 checkout：

- 当前分支：`codex/final-perf-integration-20260724`；
- HEAD：`15a2205`；
- checkout 自身也有少量未提交文档/测试/config 修改；
- 内含 `realq/`、外层 `utils/`、`tools/`、`tests/`、运行脚本和大量证据文档。

它的价值有两部分：

1. P01–P06/P10 性能实现本身；
2. 用来证明优化有没有执行、是否逐字节等价、计时是否可信的完整工具链。

只比较内部 `realq/` 会漏掉 `utils/quant_utils.py` 中的 P01/P02/P03/P04/P10，以及 checkpoint/provenance 和比较器。

### 7.2 P01–P06/P10 优化内容

| ID | 开关 | 实现 | 主要收益 | 数值/适用边界 |
|---|---|---|---|---|
| P01 | `quantizer_inner_fastpath` | 每个 GPTQ block 一次性验证 scale/zero、自然列映射和输入域，内循环调用私有 prevalidated fake-quant primitive | 去掉逐列重复校验/映射 | 版本计数不可用时结构化回退 legacy，并记录原因 |
| P02 | `w_clip_search_impl=symmetric_union_exact` | 对有限、对称量化，将 Cartesian `max(a_i,b_j)` 候选化简为端点并集，保留旧 tie order | 每 row/group 的 625 次 QDQ/error 评估降到最多 50 | 非有限、非对称或 unsupported shape 回退 Cartesian |
| P03 | `w_clip_update_impl=where_out` | 用固定 shape `torch.where(..., out=)` 更新 winner，去掉 Python `torch.any` guard | 减少 host-visible CUDA predicate/sync | P02 命中时 Cartesian winner 分支被绕过，P03 被 shadow |
| P04 | `w_group_param_layout=compact` | grouped weight scale/zero 从每自然列复制一次改为每自然 group 保存一次，使用时映射展开 | qparam 存储/collective 宽度约从 `2RC` 降到 `2R ceil(C/G)` | act-order、short tail、API 映射必须保持一致 |
| P05 | `fisher_fp32_cache` | 每层一次把 BF16 Fisher 扩成 FP32并驻留，所有 refresh 复用 | 去掉每个 module/chunk 重复 cast/分配 | 数值 class N：会改变 allocator/workspace 历史，不能预承诺 checkpoint bit-exact |
| P06 | `act_order_stitch_impl=prefix_q_trailing_w_exact` | 多 rank act-order refresh 用已 gather 的 Q prefix 与 W suffix 重建完整 permuted weight | 去掉一次冗余 full-weight all-gather，减少 logical payload | 主要是多 rank 收益，边界/collective 对称性需严格检查 |
| P10 | `prepared_clamp_bound_cache` | 在 P01 prepared context 中每 block 只计算一次 invariant tensor `-(maxq+1)` | 减少逐列 integer add/neg kernel launch | 必须依赖 P01；保留 tensor clamp overload 和旧算术顺序 |

该分支最终默认启用 P01/P02/P04/P05/P06/P10，P03 保持 `guarded`，原因是正常 finite symmetric 主路径已被 P02 接管，开启 P03 不代表它实际执行。

### 7.3 可选严格 Block-GD 确定性

该 checkout 独有而当前主代码已不再保留的一个配置是：

```text
strict_block_gd_determinism=false
```

开启时，仅在 Fisher-MSE 和 final-KL 的 `torch.autograd.grad` 周围临时启用 strict deterministic algorithms。它不改变其他 forward 选择。

这个开关被归为 numerical class N：memory-efficient attention backward 可能改变 split-key reduction 和浮点累加顺序。它用于诊断/收紧重复性，不是宣称免费 bit-exact 的性能优化。

### 7.4 运行时优化审计

`final_perf.../realq/runner/layer_loop.py` 比当前主线多出约 200 行 P01/P10 审计逻辑：

- 记录 requested 与 actual prepared/fallback；
- 合并每个 module 的 prepared block、cached bound、fallback reason；
- 只有全部 module 有审计、prepared>0、fallback=0 时才标记 P01 active；
- 只有所有 prepared block 都使用 cached bound 时才标记 P10 active；
- 在 `quant_stop_layer`/`perf_measure_layer` 诊断运行中原子写每 rank JSON；
- 正常全模型运行不做每层 fsync，避免观测污染性能。

这解决了“配置打开但 runtime 实际 fallback，报告却仍声称优化生效”的问题。

### 7.5 更严格的计时边界

该 checkout 的 performance measurement 把一层调用拆成：

```text
algorithm interval:
  layer quantization + final replay
  -> CUDA synchronize
  -> algorithm endpoint

teardown interval:
  algorithm endpoint -> quantize_one_layer return

post-call barrier:
  不计入上述 interval
```

`quantize_one_layer` 接收内部 callback，在 `layer.final_replay` 完成、`layer.teardown` 开始前恰好调用一次。JSON schema 记录 algorithm/full-call/teardown 三种时间，防止慢 rank 的 post-call barrier 或 JSON/fsync 污染主指标。

### 7.6 正确性与性能证据工具

该 checkout 新增/强化的工具覆盖：

- `run_performance_correctness.sh`：同 GPU/topology 的 baseline/candidate checkpoint；
- `compare_performance_checkpoints.py`：逐 tensor raw-byte、config delta、source/cache/topology/provenance；
- `run_performance_timing.sh`：warm-up、重复测量、GPU telemetry lifecycle；
- `compare_performance_timings.py`：A/A repeatability、A/B envelope、fail-closed schema；
- `p03_p04_cuda_probe.py`：CUDA raw-byte 与 compact layout；
- `p06_distributed_cpu_probe.py`：distributed stitch/collective；
- `compare_column_block_loss_logs.py`：按 layer/module/block/column/Adam step 对齐 loss；
- P01/P02 adversarial tests、P04/P06/P10 tests、strict determinism tests。

证据协议显式区分：

- `X`：意图逐字节等价的结构优化；
- `N`：数学公式不变但可能因 allocator/kernel/reduction 历史产生数值漂移；
- operator raw-byte gate；
- one-layer checkpoint gate；
- full-model quality；
- diagnostic timing 与可正式声明 timing。

这套证据链是该目录相对当前 `realq/` 最值得保留、但尚未完整回合主线的部分。

### 7.7 历史 A/B 结论

Llama-3.2-3B W4A4KV4 full-model concurrent A/B 中：

- baseline 28-layer loop `1467 s`；
- all-opt `1304 s`；
- 时间减少 `11.11%`，throughput 约 `1.125×`；
- 最终 KL/PPL 在当时打印精度内接近；
- 但 5628 个主 loss 中只有前 60 个完全相同，之后大部分出现漂移；
- 没有完成 checkpoint raw-byte 对比和 P05 隔离；
- 同节点并发、loss logging 和无三重复意味着它是诊断结果，不是严格隔离 speed claim。

用户当时接受该精度范围后，P01/P02/P04/P05/P06/P10 被提升为默认。当前主代码随后又把 P03 默认设为 `where_out`，并增加 Triton column-block。

### 7.8 严格按“当前主代码”比较时的差异

当前 `realq/` 已吸收该性能分支大部分实现，因此现在的真实差异是：

| 能力 | 当前 `realq/` | `final_perf.../realq/` |
|---|---|---|
| P01/P02/P04/P05/P06/P10 | 默认启用 | 默认启用 |
| P03 | 默认 `where_out` | 默认 `guarded` |
| Triton column-block | 有，默认启用 | 没有 |
| `strict_block_gd_determinism` | 没有 | 有，默认关闭 |
| P01/P10 per-layer 持久审计 | 主线仅保留较轻 runtime audit | 有完整 fail-closed JSON 审计 |
| 算法/teardown 精确计时边界 | 较简版 schema | 有 callback 和 schema 2 |
| `require_static_cache_hit` | 有 | 没有 |
| `require_reference_cache_hit` | 有 | 没有 |
| `tokens_cache_file` | 有 | 没有，只从 `tokens_cache_path` 推导 |
| grouped weight 组合 | 主线支持更多 act-order/group size 组合 | 较早限制为 `w_groupsize=-1` 或等于 blocksize |
| cache-hit 后 non-FSDP model 状态 | 主线补了回 CPU/cleanup 对齐 | 较早行为 |

所以该 checkout 不应该整体覆盖回当前主代码。正确做法是按需移植其 **审计/比较/计时基础设施**，而不是回退量化内核。

## 8. 横向对比：每个衍生版本到底优化了什么

| 维度 | 主 `realq` | Plus | MoE | Benchmark | Final perf |
|---|---|---|---|---|---|
| Dense GPTQ 数学 | 基线 | 保留 | dense attention 保留 | 保留较早快照 | 保留 |
| Block-GD 更新域 | 当前 linear suffix | 当前/下一 block 全部未量化 weights | 同 projection 全 experts joint | 与早期主线一致 | 与主线一致 |
| 新模型结构 | dense decoder | 无 | Qwen3-MoE | 无 | 无 |
| Stage-0 数据 | saliency + Fisher | 相同 | + routes/counts/coverage/ragged saliency | 相同 | 相同 |
| 主要性能手段 | P01–P10 + Triton | 继承主线；full-block 会增加 active weights | expert batching、fixed-route CSR、grouped-mm、GPU residency | generation batching/resume | P01–P06/P10 与证据工具 |
| 主要质量目标 | 降低 KL/PPL | 用更多 future weights 吸收误差 | 让 sparse experts 可量化且保持自然路由 | 测 reasoning/code 能力 | 不改目标，缩短运行时间 |
| 分布式 | DP/rank group/FSDP/CPU master | full-block 可分布式；hierarchical 仅单卡 | 正式 joint 仅单卡 | 量化可继承旧 DP；生成由 rank0 dispatch | 重点验证 world4/world8 |
| 最大风险 | 复杂配置/显存 | 算法轨迹改变；hierarchical 代理失配 | 极高显存、单架构、route coverage | 旧量化内核、生成成本、代码执行安全 | 证据分支较旧，P05 数值漂移 |

## 9. 组合与复用建议

### 9.1 哪些可以直接组合

- **当前 `realq` + benchmark modules**：概念上最自然。量化核心保持最新，只移植 `benchmarks/`、reasoning Config 字段和 pipeline hook。
- **Plus + 主线性能内核**：当前 Plus 已复制主线 P01–P10/Triton；普通 full-block 路径可用。Hierarchical 必须关闭 Triton。
- **MoE + 主线 Triton**：当前已经通过跨包 import 组合，但需要把这个依赖显式化或复制成 MoE 自有稳定 API。
- **final perf 工具 + 当前主线**：checkpoint comparator、timing comparator、runtime audit 可按功能移植。

### 9.2 哪些不能简单叠加

- Plus full-block 与 MoE joint Block-GD 不是同一抽象：前者跨当前/下一 Transformer block 的未来 dense linears，后者跨同 projection 的所有 experts。直接复制 `BlockRefreshState` 到 MoE 会改变 joint/Jacobi 调度和显存布局。
- Plus hierarchical 当前要求 `triton_column_block=false`，不能与主线默认 Triton 同时开。
- MoE 正式路径拒绝 CPU master/FSDP offload；不能套用 dense 大模型的两阶段 offload 生命周期。
- Benchmark 目录不能仅通过替换 import 名就获得当前主线 Triton，因为它的 `RealQLayer`/runner 接口缺少 P10/Triton 参数。
- `final_perf` 整仓不能直接作为新基线：会丢掉当前主线较新的 cache、group quant 和 Triton 功能。

### 9.3 推荐的长期代码组织

目前多个目录都是整包复制，容易发生已观察到的快照漂移。建议逐步改成：

```text
realq/                         # 唯一 dense quant core
realq/extensions/full_block/  # BlockRefreshState / Plus policy
realq/extensions/moe/         # adapter、routed stats、joint runner
realq/benchmarks/              # generation/scoring，不复制 quant core
tools/perf/                    # final_perf 的 comparator/harness
```

优先抽象的接口：

1. `WeightUpdatePolicy`：single-linear suffix / full-block future / all-expert joint；
2. `LayerPlan`：dense module groups / sparse expert projections；
3. `StaticStats` 扩展字段与 cache schema registry；
4. `ColumnBlockStepper`：PyTorch / Triton / MoE expert-batched；
5. `PostQuantEvaluator`：KL/PPL、lm-eval、reasoning；
6. `OptimizationAuditSink`：运行时分支证据，不侵入数学代码。

这样能减少整包复制导致的四类问题：默认值漂移、bug fix 漏同步、checkpoint schema 分叉、共享 `utils` 的隐式 ABI 依赖。

## 10. 文件差异附录

### 10.1 Plus 独有配置

相对主线新增：

```text
block_hessian_hierarchical
block_hessian_samples
block_hessian_hvp_bsz
block_hessian_steps
block_hessian_rank
block_hessian_seed
block_hessian_allow_fd_fallback
block_hessian_fd_scale
block_hessian_trace_path
```

Plus `Config` 共 96 个字段，比主线多 9 个，正好对应上述新增字段。

### 10.2 MoE 独有配置

```text
moe_route_pack_impl
moe_expert_chunk_assignments
moe_min_expert_assignments
moe_min_expert_unique_tokens
moe_min_expert_unique_samples
moe_fail_on_teacher_cold
moe_fail_on_student_cold
moe_zero_route_fallback
moe_gpu_resident
moe_joint_column_block
moe_expert_loss_slide_window
```

MoE `Config` 共 98 个字段。

### 10.3 Benchmark 独有配置

```text
reasoning_eval
reasoning_tasks
reasoning_data_dir
reasoning_output_dir
reasoning_batch_size
reasoning_limit
reasoning_max_new_tokens
reasoning_num_samples
reasoning_apply_chat_template
reasoning_enable_thinking
reasoning_do_sample
reasoning_temperature
reasoning_top_p
reasoning_top_k
reasoning_seed
reasoning_resume
reasoning_protocol
reasoning_system_prompt
reasoning_lcb_release
reasoning_lcb_source_dir
skip_kl_ppl_eval
```

Benchmark `Config` 共 106 个字段。它没有当前主线的 P10/Triton 两字段，所以总字段差不是简单的 `87 + 21`。

### 10.4 Final perf 独有/缺失配置

相对当前主线独有：

```text
strict_block_gd_determinism
```

相对当前主线缺少：

```text
require_static_cache_hit
require_reference_cache_hit
tokens_cache_file
triton_column_block
```

Final perf `Config` 共 84 个字段。

### 10.5 仅 namespace 变化的公共代码

Plus 中除第 4.1 节列出的文件外，公共 `.py/.sh` 都只需把 `realq_plus` 归一为 `realq` 即可与主线匹配。

MoE 中以下公共代码归一 namespace 后与主线一致：

```text
alignment.py
fsdp.py
parallel/*
precompute/labels.py
ptq.py
quant/hessian.py
refresh/block_gd.py
refresh/fisher_loss.py
refresh/kl_loss.py
runner/module_groups.py
scripts/*
utils/log.py
utils/memory.py
utils/nvtx.py
```

这意味着 MoE 没有修改 dense Block-GD/Fisher/KL 数学函数本身，而是在新的 MoE runner/refresh 中组合或旁路它们。

Benchmark 中除 `config.py`、`pipeline.py`、`quant/realq_layer.py`、`runner/layer_loop.py` 外，公共代码归一 namespace 后与主线一致。

### 10.6 容易误读的几点

1. `realq_plus` 不是另一版 Plus。
2. `fisher_diag_mse` 在当前代码中是 full non-diagonal aggregated Fisher，不应因名字写成 diagonal 方法。
3. Plus 的“full block”不是一次把已量化 weights 重新训练；已完成 weights 永久锁定。
4. Plus slide-window 只有在下一 arm 实际参与，即 `alpha < 1` 时，才把下一 block future weights 加入 active leaves。
5. MoE 的 zero-route RTN 只允许 assignment 真正为 0；少量正 assignment 仍 fail closed。
6. MoE 的 joint update 是 Jacobi，不等价于 serial expert Gauss-Seidel。
7. MoE fixed-route 只固定当前层已捕获 route；下一层 slide 仍自然 reroute。
8. Benchmark 的 n>1 oracle 不是官方 pass@k。
9. `official_eval` 的确认 flag 不是安全沙箱。
10. Final perf 的历史优化收益不能直接加到当前主线：大部分优化已在主线默认开启。
11. Triton inner-loop 的微基准加速不能直接当成端到端加速；Stage-0、Hessian、Block-GD、replay 和 eval 仍占主要时间。

## 11. 最终判断

如果按“相对 `realq` 到底实现了什么”给每个目录一句最准确的定义：

- `realq-plus/`：**让一次 Block-GD backward 真正优化完整反向图里尚可变的 Transformer 权重，并实验性叠加低秩 Block-Hessian 跨 linear 补偿。**
- `realq_plus`：**同一 Plus 代码的合法 Python 包名。**
- `realq_moe/`：**把 REAL-Q 从规则 dense linears 扩展到自然路由的 ragged experts，并通过全 expert 锁步、fixed-route grouped-mm、joint backward 和 GPU-resident 生命周期把复杂度压到可运行。**
- `realq_benchmark/`：**给量化模型增加可复现、可恢复、带安全边界的数学/代码生成评测闭环。**
- `final_perf_integration_20260724/`：**P01–P06/P10 性能实现及其严谨证据工具的历史集成点；算法优化大多已进入当前主线，审计/计时工具仍有独立价值。**

从当前代码维护角度，主线应继续以 `realq/` 为准；Plus/MoE 中真正改变算法或结构的模块应逐步变成扩展策略，benchmark 应前移为共享评测层，final-perf 应保留为证据与工具来源，而不应继续让四份量化核心独立漂移。
