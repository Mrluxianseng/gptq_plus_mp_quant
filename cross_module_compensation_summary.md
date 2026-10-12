# 跨模块补偿方向总结（cross-module compensation）

> 更新于 2026-09-27。本文档记录「同层内跨模块补偿」方向的所有方法、实验数据、伪代码、方法论与理论，供新会话快速恢复上下文。
> **代码在 worktree `D:\gptq_plus_realq`（分支 realq-warmadam），全部未提交**。主工作目录 `D:\gptq_plus` 里的 exact_v/warm_adam/closed_v 是更早一轮，别混。

---

## 1. 背景与动机

REAL-Q 的 block-GD 逐模块量化，每量化一块列，用 **layer 级 refresh loss（fisher_diag_mse）** 的梯度刷新**当前模块**的未量化列。一个自然的问题是：同层内**先量化**的模块（feeder，如 `v_proj` / `up_proj`）产生的量化误差，会通过残差流 / 非线性传播到**后续模块**（target，如 `o_proj` / `down_proj`）。能不能**显式地"跨模块"补偿**这个误差——直接改 target 的权重来抵消 feeder 的误差？

方向参考了 **QEP**（Quantization Error Propagation: Revisiting Layer-Wise PTQ, NeurIPS 2025, Fujitsu/东京大学, **arXiv:2504.09629**），其闭式跨层补偿：

```
W_l* = W_l + α_l · W_l · δ_l · X̂_lᵀ · Ĥ_l⁻¹，   δ_l = X_l − X̂_l（上游累积误差）
```

本方向试图把它移植到"同层内跨模块"（SwiGLU 的 up→down、attention 的 v→o）。

---

## 2. 试过的方法

两条路线：**跨模块闭式解**（closed-form）和**跨模块梯度累积/铺开**（gradient）。全部需要 `--g_update_mode block_gd`。

### 2.1 跨模块闭式解

#### A. 顺序闭式 `--cross_module_compensate`
feeder 量化后，闭式补到**下一个**残差流模块（固定配对 `v_proj→o_proj`、`up_proj→down_proj`），**单个 feeder 的净误差**（不是累积 Δy）。

```
量化 feeder（v_proj / up_proj）后，量化 target（o_proj / down_proj）之前：
  δout = out(feeder 量化) − out(feeder FP)     # functional_call 只替换 feeder，跑整个 layer
  h    = target 输入（forward hook 捕获）
  ΔW   = −(δout·F)ᵀ · h · (hᵀh + λI)⁻¹        # 闭式最小二乘；F=layer output fisher（本配置下传 None → 退化为等权对角）
  target.weight += ΔW
```

#### B. 闭式铺开 `--cross_module_compensate_all`
feeder 量化后，闭式补到**所有**未量化模块（残差流精确 + SwiGLU 线性化 + 均分 `δout/m` 防过度补偿）。

```
量化 feeder 后：
  δout = out(feeder 量化) − out(feeder FP)
  m = 未量化 target 数量
  for target in 后续未量化模块:
    if target 是残差流模块（o_proj / down_proj）:
      ΔW = −(δout/m)ᵀ · h · (hᵀh + λI)⁻¹
    elif target 是 SwiGLU（up / gate）:
      ā_up   = mean_t SiLU(gate_out_t)
      ā_gate = mean_t [SiLU'(gate_out_t) ⊙ up_out_t]
      A = W_down · diag(ā)                       # 线性化 SiLU⊙gate⊙down
      ΔW = −(AᵀA + λI)⁻¹ · Aᵀ · (δout/m)ᵀ · x · (xᵀx + λI)⁻¹
    target.weight += ΔW
```

### 2.2 跨模块梯度累积 / 铺开

#### C. SGD 铺开 `--eager_cross_compensate`
每个模块量化后，对所有后续未量化模块做一步 **plain-SGD**（`eager_cross_lr`）。

#### D. Adam 铺开 `--joint_block_gd`
每个 block 边界，对所有后续未量化模块做一步 **Adam**（persistent state 跨层预热）。

```
# eager 伪代码
量化模块 name 后：
  for target in 后续未量化模块:
    g = ∂(refresh loss)/∂W_target    # 当前状态（feeder 已量化）梯度
    W_target −= g · eager_cross_lr
```

> 另有更早的「梯度累积」探针（Grad Accumulation，见 memory）：否证"提前共享反传累积未量化模块自身梯度"——cos(first,cur) 浅层≈噪声底、深层塌到 0.00–0.28，mean_beats_latest 14/18 no。

---

## 3. 实验数据

### 3.1 smoke 配置（`QUANT_STOP_LAYER=4`, `nsamples=4`, `seq_len=256`, `grad_lr=5e-5`, fisher_diag_mse, global_loss）

| 方法 | KL | PPL | vs baseline |
|---|---|---|---|
| baseline（纯 block_gd） | 1.13e-01 | 26.69 | — |
| cross（顺序闭式） | 1.12e-01 | 26.61 | −0.4% |
| all·残差流铺开 | 1.14e-01 | 26.75 | +0.9% |
| all·含 SwiGLU | 1.15e-01 | 26.70 | +1.8% |
| joint（Adam 铺开） | 2.95e-01 | 30.34 | 明显更差 |
| eager（SGD 铺开，默认 lr） | 1.24e-01 | 26.32 | 略差 |
| eager_2e-5 | 2.60e-01 | 30.02 | 差 |
| eager_5e-5 / 1e-4 | 5.08 / 2.28 | 3006 / 213 | 发散 |

### 3.2 full 配置（无 QUANT_STOP_LAYER，`nsamples=32`, `seq_len=256`，其余同 smoke）

| 方法 | KL（精确） | PPL |
|---|---|---|
| baseline | 0.2226463 | 27.9890 |
| cross | 0.2229371 | 27.9932（**+0.13%**） |

### 3.3 关键诊断

- cross 补偿**确实生效**（不是 no-op）：触发 56 次（28 层 × 2 对），补偿 norm 0.0008~0.62。
- cross 让 layer0 `o_proj` 的 refresh loss（block 0）从 **0.0101397 → 0.0099602（−1.8%）**，但最终 KL **+0.13%**——**surrogate 下降没有传导为最终指标改善**。
- SwiGLU 补偿 norm 失控：gate 0.41~2.74 vs 残差流 0.04~0.13（`AᵀA` 病态：`A=W_down·diag(ā)` 秩 ≤1024 < 3072，`SiLU'(g)` 大量近零通道）。

---

## 4. 方法论

### 4.1 数学结构（GPTQ 模块内补偿的跨模块推广）

| | GPTQ 模块内 | 跨模块闭式（cross） |
|---|---|---|
| 误差量 | 列误差 `Err`（本模块） | feeder 输出误差投影 `δout`（残差流实测） |
| 相关系数 | `H = XᵀX`（输入二阶矩） | `hᵀh`（target 输入二阶矩） |
| 解 | `δW = −Err·H⁻¹` | `ΔW = −(δoutᵀh)(hᵀh+λI)⁻¹` |

### 4.2 关键理论发现

1. **闭式解与输出 fisher F 无关（F 求导消掉）**。目标 `min_ΔW tr(F(ΔW hᵀ+δoutᵀ)(·)ᵀ)` 求导得 `F·ΔW·(hᵀh) = −F·δoutᵀh`，左乘 F⁻¹ → `ΔW = −δoutᵀh(hᵀh)⁻¹`。**ΔW 行可分，逐输出通道独立**。⟹ 闭式跨模块补偿**用不上**输出 fisher 的跨通道耦合（非对角）；跨通道耦合只在**有限步梯度**（block-GD）里起作用。

2. **GPTQ Hessian 用跨层污染输入 `inps`，但看不到同层 feeder 误差**。代码时序：
   - `inps` = 跨层污染输入（`layer.quantized_replay_forward` 逐层更新，line ~12797）；
   - `fp_inps` = FP 输入（`disable_fp_path_quant` 更新，line ~11221）；
   - GPTQ Hessian 累积（`accumulate_hessian_for_gptq`）在 `for group_names in sequential` **之前** → 用 `inps` forward **FP 状态**的整层，抓所有模块输入。
   ⟹ 它刻画的是**跨层污染后的输入分布**，**未经历同层 feeder 量化后的变化**。

3. **非对角定理**（memory `project-offdiag-insight`）：对角行度量闭式解最优，block-GD 刷新的全部价值来自输出 Fisher 的非对角部分。

### 4.3 两种污染（必须分开）

- **跨层污染** `x_FP → x̂_layer`：前面已量化层传播 → 本层输入。GPTQ 的 Hessian 用 `inps` **已处理**。
- **同层模块间影响** `x̂_layer → ŷ_feeder → ŷ_target`：同层 feeder 量化 → target 输入 `ĥ_B = h_B + δh_B`。GPTQ 的 Hessian **看不到**（层开始时累积），block-GD 梯度**能看到**（当前状态），但"能看到" ≠ "有限步已充分补偿"。

---

## 5. 优势与劣势

### 优势
- 闭式解**有界**、无需迭代，不像梯度铺开会发散（eager 发散、joint 明显有害）。
- 分摊（`δout/m`）从机制上避免了 eager 的过度补偿。
- 数学结构清晰，是 GPTQ 模块内闭式补偿的自然推广。

### 劣势
- **无正收益**：顺序闭式（cross）full 配置 +0.13%；铺开（all）smoke +0.9%~+1.8%；梯度铺开发散/明显有害。
- **surrogate 下降不传导**：cross 降 refresh loss 1.8%，最终 KL 反升 0.13%。
- SwiGLU 线性化数值病态（gate 补偿 norm 失控）。
- 闭式解逐输出通道独立，**拿不到 fisher 跨通道耦合**（只有梯度能拿）。

---

## 6. 待验证假设与下一步（重要）

**尚未定论、不要当结论**：
- ❌ 之前曾断言"cross 冗余 / GPTQ 已处理跨层漂移 ⟹ QEP 式修正无增量 / block-GD 已覆盖 cross"——**这些都是待验证假设，不是已证结论**（尤其"GPTQ 用污染输入 ≠ 已处理同层 feeder 误差"）。

下一步按序做（用户 2026-09-27 定的计划）：

1. **实验 1（最先，成本低）**：测 feeder 量化前后 target 梯度变了多少。固定 layer、feeder→target 对、校准样本、seed；状态 A=feeder 未量化、状态 B=feeder 已量化，分别算 target 梯度，记录范数、cosine、沿梯度试探更新后的固定评估 loss 变化。判据：状态 B 的梯度是否**真的朝着降低 feeder 量化后输出误差的方向走**。
2. **实验 2**：最小梯度时序消融。A=feeder 量化后重算梯度（当前）、B=冻结 feeder 量化前的缓存梯度、C=不做 target block-GD。A vs B = 梯度是否感知 feeder 量化状态；A vs C = block-GD 更新本身的贡献。
3. **实验 3**：cross 是否有独立增量。baseline vs baseline+cross 配对对照，报告局部 loss、最终 KL/PPL、修正量与 block-GD 更新量的 cosine。

实验纪律：先固定代表性 layer + 少量 feeder→target 对；预先写判断标准；最终 KL/PPL 用独立评估集，局部 surrogate 只作诊断。

---

## 7. 相关文件/日志

- 代码：`gptq_utils/gptq_plus_utils.py`（`compute_cross_module_compensation` ~8642 行；`compute_cross_module_compensation_all` 其后；触发逻辑在 `gptq_fwrd` 的模块循环里）、`process_args.py`（4 个 flag）
- 脚本：`scripts/smoke_situation_b.sh`（smoke 各 arm）、`scripts/smoke_cross_compare.sh`（三方队列）、`scripts/cross_full_compare.sh`（full 两臂）
- 日志：`outputs/smoke_*.log`、`outputs/full_*.log`
- 环境：WSL `/home/uky/gptq_plus/.venv`（Python 3.12 + torch 2.11.0+cu128），代码 `/mnt/d/gptq_plus_realq`，模型 `/mnt/d/llamaModels/Qwen3-0.6B`
