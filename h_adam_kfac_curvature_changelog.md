# H-Adam KFAC 曲率对角接入 —— 修改说明与工作量记录

> 关联设计文档：[`adam_analysis_for_ptq.md`](./adam_analysis_for_ptq.md) 方向 2（H-Adam：用 Hessian/Fisher 对角替换 Adam 二阶矩）+ 方向 4（格点感知阻尼的接入点）。
> 目标：让 `run_pre_quant_gd` 的 `h_adam` 优化器分支真正用上 **完整 KFAC 曲率对角**（output-Fisher ⊗ input-Hessian），并修复原占位调用的多个缺陷。

---

## 一、背景与数据通路结论

原调用是坏的：

```python
apply_dense_optimizer_step(
    module.weight.data, grad, lr=grad_lr,
    curvature_diag,          # 语法错误：位置参数跟在关键字参数后
    optimizer=grad_optimizer, ...
)
```

三个问题：
1. **语法错误**：`lr=` 之后又出现位置参数 `curvature_diag`。
2. **`curvature_diag` 未定义**：`run_pre_quant_gd` 作用域内根本没有这个变量（占位符）。
3. **`h_adam` 分支 bug**：`apply_dense_optimizer_step` 里整段更新逻辑被写在 `if opt_state is None:` 内部，导致第 2 步起落入原生 Adam 分支并因缺失 `exp_avg_sq` 而 `KeyError`。

**数据通路关键发现**：
`run_pre_quant_gd`（`gptq_fwrd` 中约 line 6744）在 **`GPTQPlus` 实例创建（约 line 6890）之前** 执行，因此 GPTQ 的输入 Hessian（`GPTQPlus.H` / `act_square`）此时 **尚不存在**。

因此 KFAC 两个因子的来源为：

| KFAC 因子 | 数学含义 | 本次实现的来源 |
|---|---|---|
| 输入因子 diag(A) | `E[x_j²]` 每输入通道激活能量 | **新增**：pre-GD 阶段一次带 hook 的前向，累积 `Σ x_j²`（复现 `act_square` 的对角，因为此时 GPTQ H 还没算） |
| 输出因子 diag(B) | `E[g_i²]` 输出端 Fisher 对角 | 复用已有的 `layer_output_fisher_by_module`（block 输出 (H,H) Fisher）的对角 |

**维度限制（已在代码中处理）**：`layer_output_fisher` 是在 **block 输出（残差流 hidden_size）** 上的 Fisher，只有 `o_proj / down_proj`（输出维 == H）能直接用完整外积；`q/k/v/gate/up_proj` 输出维 ≠ H，自动回退为 **input-only** 对角按输出行广播。

---

## 二、KFAC 曲率对角公式

```
curvature[i, j] ≈ E[g_i²] · E[x_j²] = diag(output_fisher)[i] · input_sq[j]
                = outer( b_norm , a_norm ) + damping
```

- 两个因子各自 **归一化到均值 1** → 组装后 `curvature` 均值 ≈ 1，`sqrt(curvature)` 量级 ≈ 1，从而 `lr` 语义与原生 Adam 保持一致（不需重调学习率）。
- `damping`（默认 0.1，相对量级）防止近零通道处 `sqrt(denom)` 爆炸，同时是方向 4「量化格点感知阻尼」的自然接入点。
- DP 下输入因子做 `allreduce_sum_`，保证各 rank 曲率一致、权重逐位同步。
- 显存：H-Adam 去掉了 `exp_avg_sq`，state 由 Adam 的 2× 权重降为 1×；曲率对角额外 1×（方阵模块）或极小（回退模块），整体 ≤ Adam。

---

## 三、逐项修改清单（文件：`gptq_utils/gptq_plus_utils.py`）

| # | 位置 | 修改 | 类型 |
|---|---|---|---|
| 1 | `apply_dense_optimizer_step` 的 `h_adam` 分支 | 把更新逻辑移出 `if opt_state is None:`（修 KeyError bug）；改用归一化 KFAC 曲率对角做分母；`grad_step` 转 fp32；设备/精度对齐 | 逻辑修复 + 功能 |
| 2 | 新增 `collect_module_input_second_moment(...)` | 带 forward hook 的单次前向，逐 module 累积 `Σ x_j²`（KFAC 输入因子），跨 DP rank 求和 | 新函数 |
| 3 | 新增 `build_kfac_curvature_diag(...)` | 组装 `outer(diag(B)/mean, diag(A)/mean) + damping`，方阵模块用完整外积、其余回退 input-only 广播 | 新函数 |
| 4 | `run_pre_quant_gd` 签名 | 新增 `curvature_damping=0.1` 形参 | 接口 |
| 5 | `run_pre_quant_gd` setup 块 | `h_adam` 时预计算 `curvature_diag_by_module`（每层一次） | 功能 |
| 6 | `run_pre_quant_gd` optimizer_step 调用 | 修语法错误，改为 `curvature_diag=curvature_diag_by_module.get(module_name)` | Bug 修复 |
| 7 | `gptq_fwrd` 调用 `run_pre_quant_gd` 处 | 透传 `curvature_damping=getattr(args, "h_adam_curvature_damping", 0.1)` | 接口 |

---

## 四、代码工作量统计

- **改动文件**：1 个（`gptq_utils/gptq_plus_utils.py`）
- **新增函数**：2 个
  - `collect_module_input_second_moment` —— 约 **68** 行（4493–4560）
  - `build_kfac_curvature_diag` —— 约 **41** 行（4562–4602）
- **修改函数**：2 个（`apply_dense_optimizer_step`、`run_pre_quant_gd`）+ 1 处调用点（`gptq_fwrd`）
- **本次新增净代码行**：约 **155 行**
  （2 个新函数 ≈109 + h_adam 分支重写净 +12 + setup 预计算块 +32 + 签名/透传 +2）
- **编辑落点**：7 处（2 新函数 + 5 处原地修改）
- **语法校验**：`python -m ast` 解析通过 ✅

> 注：`git diff --numstat` 显示 `+270 / -16`，其中包含本文件在本次会话开始前 **已存在的未提交改动**；上面 155 行为本次任务实际贡献的净新增量。

---

## 五、命令行接入（`process_args.py`）

- `--pre_grad_optimizer` / `--pre_final_layer_grad_optimizer` 的 `choices` 补上 **`h_adam`**（此前只有 `sgd/adam`，导致 h_adam 根本无法从命令行选中）。
- 新增 `--h_adam_curvature_damping`（`float`，默认 `0.1`），并加入非负校验。
- `gptq_fwrd` 处透传 `getattr(args, "h_adam_curvature_damping", 0.1)`。

示例：
```bash
--pre_grad_optimizer h_adam --h_adam_curvature_damping 0.1
```

## 六、可选后续（未在本次实现）

1. 内层模块（q/k/v/gate/up）的真实输出端 Fisher：可专门采集「每 module 输出处」的 Fisher 对角以启用完整 KFAC，替换现在的 input-only 回退。
2. 方向 4：把「权重到最近量化格点距离」并入 `build_kfac_curvature_diag` 的 `damping` 项，实现真正的量化格点感知阻尼。
