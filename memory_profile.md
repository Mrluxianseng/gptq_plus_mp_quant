# GPTQ+ 全流程显存峰值分析

## 符号与常量

| 符号 | 含义 | 典型值（Qwen3-4B） |
|------|------|------------------|
| `P` | 模型参数总数 | 4 × 10⁹ |
| `L` | transformer 层数 | 36 |
| `d` | hidden dim | 2560 |
| `d_ff` | MLP 中间维度 | 9728 |
| `V` | vocab size | 151936 |
| `T` | 校准序列长度 `SEQ_LEN` | 2048 |
| `N` | 校准样本数 `N_SAMPLES` | 768 |
| `n_local` | 单 rank 的样本数 = `N / dp_world` | 384（2 卡） |
| `NG` | `NUM_GROUPS`（Hessian 按输出行分组数） | 4 |
| `FNG` | `FISHER_NUM_GROUPS` | 512 |
| `GHTK` | `GRAD_HESSIAN_TOPK` | 20 |
| `M` | `BLOCKSIZE` | 256 |
| `GL_BSZ` | `GLOBAL_LOSS_BSZ` | 4 |
| `BSZ` | stats 收集 batch size | 32 |
| `FSTATS_BSZ` | `FINAL_LAYER_STATS_BSZ` | 4 |
| `HA_BSZ` | `HESSIAN_ACCUM_BSZ` | 128 |
| `BWD_BSZ` | `BACKWARD_BSZ`（refresh batch） | 32 |
| `BSAMP` | `BACKWARD_SAMPLES`（refresh 采样数） | 32 |

**dtype 字节数**：bf16 = 2 B，fp32 = 4 B。
**前提**：`--offload_inps` 未开（inps/fp_inps 在 GPU）。

---

## 全流程阶段划分

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 0: 载入模型 + input capture                                │
├─────────────────────────────────────────────────────────────────┤
│ Stage 1: 预计算 static saliency + fisher（GLOBAL_LOSS=1）        │
├─────────────────────────────────────────────────────────────────┤
│ Stage 2: 预计算 fp_inps_final（residual_kl / refined_mse）       │
├─────────────────────────────────────────────────────────────────┤
│ 每层循环 (× L):                                                   │
│   Stage 2.5: refined_mse 每层 grad pool 采集（仅 refined_mse）    │
│   Stage 3a: Reference FP forward                                 │
│   Stage 3b: Slide window 的 next-FP forward                      │
│   Stage 3c: Stats 收集（saliency NLL + reference loss backward） │
│   Stage 3d: Hessian accumulation（add_batch hook forward）       │
│   Stage 3e: finalize_hessian（allreduce）                        │
│   Stage 3f: 每个 Linear 的 fasterquant（含 block refresh）       │
├─────────────────────────────────────────────────────────────────┤
│ Stage 4: lm_eval QA                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

# 版本 A：`GRAD_REFRESH_LOSS = fisher_diag_mse`（默认）

## Stage 0 — input capture

| 对象 | 公式 | 4B |
|------|------|-----|
| 全模型常驻 GPU（bf16） | `2P` | 8 GB |
| `inps` 缓冲（bf16） | `2·n_local·T·d` | 4 GB |
| 单次 forward 激活 | ~`5·1·T·d·2` | 0.03 GB |

**峰值 ≈ `2P + 2·n_local·T·d` ≈ 12 GB**

## Stage 1 — 预计算 saliency + fisher

整模型 backward，把 end-to-end NLL loss 对每层输出 + 每个 Linear 输出的梯度 squared 收集下来。

循环步长 `GL_BSZ`。每个 batch 内：

| 对象 | 公式 | 4B, GL_BSZ=4 |
|------|------|------|
| 全模型常驻 | `2P` | 8 GB |
| 所有 L 层前向激活图（autograd 保留） | `~10·GL_BSZ·T·d·L·2` | 15 GB |
| 全模型权重梯度（fp32） | `4P` | 16 GB（峰值在 backward 时） |
| lm_head 的 logits | `GL_BSZ·T·V·2` | 2.5 GB |
| saliency hook GPU 临时 `grad.float()` | `GL_BSZ·T·d·4` + reshape | ~0.1 GB |
| fisher hook GPU 临时 | `GL_BSZ·T·d·4` | ~0.1 GB |

**峰值 ≈ `2P + 10·GL_BSZ·T·d·L·2 + 4P + GL_BSZ·T·V·2`**
**4B ≈ 8 + 15 + 16 + 2.5 ≈ 42 GB**

调 `GL_BSZ` 是唯一杠杆（线性影响激活图大小 + logits）。4B 上 `GL_BSZ=4` 已经接近 A100-40G 上限。

## Stage 2 — fp_inps_final 预计算（`fisher_diag_mse` 路径**不触发**）

此 stage 跳过。✅

## Stage 3a — Reference FP forward

只在当前层做一次 FP 前向，bs=1 循环（未做 batching）。

**峰值增量：~0.5 GB**（激活 + 当前层参数）

## Stage 3b — Slide window next-FP forward（`LOSS_SLIDE_WINDOW=1` 时）

| 对象 | 公式 | 4B |
|------|------|-----|
| `slide_fp_inps_next` 缓冲（bf16） | `2·n_local·T·d` | **+4 GB** |
| `slide_next_layer` 参数常驻 GPU | `2·P/L` | **+0.2 GB** |

**slide_window 此阶段额外开销 ≈ +4.2 GB**（整层循环期间常驻）

## Stage 3c — Stats 收集

循环步长 `BSZ`（最后一层 `FSTATS_BSZ`）。每 batch 做 1 次 forward + 2 次 backward（saliency NLL + fisher_diag_mse 参考梯度）。

| 对象 | 公式 | 4B, BSZ=32 |
|------|------|------|
| 单层前向激活图（autograd） | `~10·BSZ·T·d·2` | 10 GB |
| `logits`（layer out → lm_head） | `BSZ·T·V·2` | 20 GB |
| `logits_fp` | `BSZ·T·V·2` | 20 GB |
| topk 切片 `grad_hessian_logits*` | `BSZ·T·GHTK·2` × 2 | 0.01 GB（可忽略） |
| saliency NLL 反传，lm_head 权重梯度（fp32） | `~V·d·4` | 1.5 GB |
| 各 Linear 权重梯度 Cache（fp32） | `~Σ rows·cols·4` | ~0.5 GB |
| `fisher_diag_mse` 参考 loss | `delta²` = `BSZ·T·d·4` | 0.7 GB |
| 当前层参数 | `2·P/L` | 0.2 GB |
| `inps + fp_inps` 常驻 | `4·n_local·T·d` | 8 GB |
| slide window 缓冲（若开） | +4 GB | +4 GB |

**峰值 ≈ 60 GB（不开 slide）/ 64 GB（开 slide）**（4B, BSZ=32）

**注意** V=150k 很大，`logits` 占 20 GB × 2。如果 BSZ 减到 16，logits 减到 10 GB × 2，峰值降到 40 GB。最后一层建议压 `FSTATS_BSZ ≤ 4`。

## Stage 3d — Hessian accumulation

`add_batch` hook 下的单 Linear（down_proj 最痛），fp32 中间量：

| 对象 | 公式 | 4B, HA_BSZ=128 |
|------|------|------|
| `weighted = [NG, B*T, C]` fp32 | `4·NG·HA_BSZ·T·d_ff` | **40 GB ⚠️** |
| `inp.float()` 临时拷贝 | `4·HA_BSZ·T·d_ff` | 10 GB |
| `block = [NG, C, C]` fp32 | `4·NG·d_ff²` | 1.5 GB |
| 所有 Linear 累积的 H（fp32） | `4·NG·(6d² + d_ff²)` | 2.2 GB |
| 当前层参数 | `2·P/L` | 0.2 GB |
| `inps + fp_inps` 常驻 | `4·n_local·T·d` | 8 GB |
| 前向激活（无 backward） | `~3·HA_BSZ·T·d·2` | 4 GB |

**峰值 ≈ 66 GB**（4B, HA_BSZ=128）

**HA_BSZ 直接杠杆 weighted**，想压到 20 GB 以内 → HA_BSZ ≤ 32。

## Stage 3e — finalize_hessian

仅 allreduce，新增 <100 MB。✅

## Stage 3f — fasterquant per Linear（block_gd）

### 3f-1: fasterquant 初始化（单 Linear，以 down_proj 为例）

| 对象 | 公式 | 4B, down_proj |
|------|------|------|
| `H`（已存在） | `4·NG·d_ff²` | 1.5 GB |
| `Hinv_init + Hinv`（Cholesky） | `~2 × H` | 3 GB |
| `GHinv + Z`（fp32） | `8·rows·d_ff` | 0.2 GB（rows=5120） |
| 各 subgroup 副本（`W_sub, Hinv 独立拷贝`） | `×NG` | ~0.5 GB |
| `anchor_weight + full_precision_weight` 等 snapshot | `~4·rows·d_ff` | 0.2 GB |
| Adam state（`exp_avg, exp_avg_sq` fp32） | `8·rows·d_ff` | 0.2 GB |

**fasterquant 本身峰值（down_proj） ≈ 5.5 GB**

### 3f-2: Block refresh（block_gd，无 slide_window）

每个 block 尾做一次 `gradient_refresh_fn`：用 `BWD_BSZ` 样本做 forward+backward。

| 对象（每 block） | 公式 | 4B, BWD_BSZ=32 |
|------|------|------|
| 单层 forward 图（带 autograd） | `~10·BWD_BSZ·T·d·2` | 10 GB |
| `override_weight`（fp32，替换权重） | `4·rows·cols` | 0.2 GB（down_proj） |
| refresh loss 反传临时 | | |
| ↳ `fisher_batch`（fp32，slice 自 static） | `4·BWD_BSZ·T·FNG` | 0.25 GB |
| ↳ `delta²` 计算 | `4·BWD_BSZ·T·d` | 0.65 GB |
| 权重梯度（fp32） | `4·rows·cols` | 0.2 GB |
| `refreshed_grad` 副本 | `4·rows·cols` | 0.2 GB |
| `weight_snapshot` | `4·rows·cols` | 0.2 GB |

**Block refresh 峰值（fisher_diag_mse, 无 slide） ≈ +11.7 GB**

**叠加到 fasterquant 的总峰值**：
fasterquant（5.5）+ inps/fp_inps（8）+ block refresh（11.7）+ 静态 H 余量（2.2）+ 层参数（0.2）
**≈ 27–30 GB**（不含 slide_window）

### 3f-3: Block refresh **开启** slide_window（`LOSS_SLIDE_WINDOW=1`）

slide 会对每个 block 额外：
1. 对 `next_layer` 再做一次 forward（基于 `out_hidden`）
2. 对下一层 fisher 再做一次 `delta_next²` 计算
3. 第二次 backward（属于同一计算图，实际是 combined loss 的 backward，autograd 图更大）

| 额外对象 | 公式 | 4B, BWD_BSZ=32 |
|------|------|------|
| `slide_next_layer` forward 激活 | `~10·BWD_BSZ·T·d·2` | **+10 GB** |
| `next_layer_output_fisher` batch slice | `4·BWD_BSZ·T·FNG` | +0.25 GB |
| `slide_fp_inps_next[batch]` GPU 切片 | `2·BWD_BSZ·T·d` | +0.3 GB |
| `delta_next²` 临时 | `4·BWD_BSZ·T·d` | +0.65 GB |
| `slide_next_layer` 参数 | `2·P/L` | +0.2 GB |
| 额外 `slide_fp_inps_next` 全局缓冲（已计入 3b） | - | - |

**slide_window 对 block refresh 额外增加 ≈ +11.4 GB**

**fasterquant 总峰值（fisher_diag_mse + slide） ≈ 38–42 GB**

## Stage 4 — lm_eval

量化后推理。峰值无 autograd：

| 对象 | 公式 | 4B, LM_EVAL_BSZ=32 |
|------|------|------|
| 模型权重（fake-quant bf16） | `2P` | 8 GB |
| KV cache | `2·LM_EVAL_BSZ·T_task·d·L·2` | 5–6 GB |
| 单步激活（无 graph） | `~3·LM_EVAL_BSZ·T_task·d·2` | 0.3 GB |
| logits | `LM_EVAL_BSZ·T_task·V·2` | 5 GB |

**峰值 ≈ 19 GB**（T_task ≈ 500）

---

## 版本 A 各阶段峰值汇总（4B 默认 sweep 配置）

| Stage | 不开 slide | 开 slide |
|-------|-----------|---------|
| 0 input capture | 12 GB | 12 GB |
| 1 static precompute | 42 GB | 42 GB |
| 2 fp_inps_final | — | — |
| 3c stats collection | 60 GB | **64 GB** |
| 3d Hessian accum（HA=128） | 66 GB | 70 GB |
| 3f fasterquant + refresh | 28 GB | **40 GB** |
| 4 lm_eval | 19 GB | 19 GB |

**全流程峰值（fisher_diag_mse + slide）≈ 70 GB** — A100-80G 可撑住，A100-40G 需要 `BSZ=16 / HA_BSZ=32`。

**slide_window 相对不开的额外显存**：
- Stage 3b 常驻：**+4.2 GB**（整层循环期间不释放）
- Stage 3f block refresh 峰值：**+11.4 GB**（每 block 发生）
- **合计增量 ≈ +15 GB**

---

# 版本 B：`GRAD_REFRESH_LOSS = residual_kl`

与版本 A 的差异全部集中在**凡是计算 refresh/reference loss 的地方**，都会把 `delta²` / fisher-加权替换成**再过一次 lm_head 的 KL loss**，从而引入**额外的 `B·T·V` 量级 logits**。

## Stage 0, 1 — 不变

## Stage 2 — fp_inps_final 预计算（仅 residual_kl 触发）⚡

```python
scratch = inps.clone()  # [n_local, T, d] bf16
for idx in range(L):
    lay = layers[idx].to(dev)
    for j in range(n_local):  # bs=1 循环！
        scratch[j] = lay(scratch[j].unsqueeze(0), ...)
```

| 对象 | 公式 | 4B |
|------|------|-----|
| `scratch` 缓冲（bf16） | `2·n_local·T·d` | 4 GB |
| 单层参数（逐层切换） | `2·P/L` | 0.2 GB |
| 单样本 forward 激活 | `~3·T·d·2` | 0.03 GB |
| 全局 `fp_inps_final` 永久缓冲（bf16） | `2·n_local·T·d` | **+4 GB 常驻** |

**Stage 2 峰值 ≈ 12 GB**（期间），**永久开销 +4 GB**

4B 上是 bs=1 循环，**慢**（~30 秒 × L = 18 分钟），但不爆显存。

## Stage 3c — Stats 收集（residual_kl reference loss）⚡

在版本 A 的 2 个 full-V logits 基础上，`compute_refresh_loss(residual_kl)` 里会再**多两次 `hidden2logits`**：

```python
final_with_delta = fp_final_hidden + delta       # [B, T, d]
logits_perturbed = hidden2logits(final_with_delta, analyzer)  # [B, T, V]
logits_fp = hidden2logits(fp_final_hidden, analyzer)          # [B, T, V]
```

| 对象 | 公式 | 4B, BSZ=32 |
|------|------|------|
| saliency 路径 `logits + logits_fp`（retain_graph 保留） | `2·BSZ·T·V·2` | 40 GB |
| **residual_kl 路径 `logits_perturbed + logits_fp`** | **`2·BSZ·T·V·2`** | **+40 GB** |
| autograd 图里的 intermediate（`final_with_delta`、softmax 输出等） | `~4·BSZ·T·V·2` 额外 | +20 GB |
| 其他（见版本 A） | | ~20 GB |

**stats 峰值 ≈ 120 GB**（4B, BSZ=32, V=150k） — **A100-80G 也会 OOM** ❌

**residual_kl 下 BSZ 必须压到 8 甚至 4**。BSZ=4 时 logits 共 ~10 GB，峰值 30–40 GB，勉强可行。

## Stage 3d — Hessian accumulation — 与版本 A 完全一致

## Stage 3f — fasterquant with residual_kl refresh ⚡

`compute_refresh_loss(residual_kl)` 用在 block refresh 里：每 block 会生成 2 个 full-V logits。

| 对象（每 block） | 公式 | 4B, BWD_BSZ=32 |
|------|------|------|
| 单层 forward 图 | `~10·BWD_BSZ·T·d·2` | 10 GB |
| **`logits_perturbed + logits_fp`（residual_kl）** | **`2·BWD_BSZ·T·V·2`** | **40 GB ⚠️** |
| autograd 图里 softmax/log_softmax 中间量 | `~2·BWD_BSZ·T·V·2` | +20 GB |
| `override_weight + refreshed_grad + snapshot + Adam` | `~20·rows·cols` | ~1 GB |
| `fp_inps_final[batch]` 切片 | `2·BWD_BSZ·T·d` | 0.3 GB |

**Block refresh 峰值（residual_kl, 无 slide） ≈ +72 GB**

→ **OOM 概率极高**。建议 `BWD_BSZ ≤ 4`（logits 降到 5 GB × 2 = 10 GB），block refresh 峰值降到 ~20 GB。

### Block refresh 开启 slide_window（residual_kl） ⚡⚡

slide 会**再生成一对** logits 用于 next_layer 的 residual_kl loss：

| 额外对象 | 公式 | 4B, BWD_BSZ=32 |
|------|------|------|
| `slide_next_layer` forward 激活 | `~10·BWD_BSZ·T·d·2` | +10 GB |
| **next-layer 的 `logits_perturbed + logits_fp`** | **`2·BWD_BSZ·T·V·2`** | **+40 GB ⚠️** |
| autograd 图中 softmax 中间量 | `~2·BWD_BSZ·T·V·2` | +20 GB |
| 其他 slide 缓冲 | ~0.5 GB | |

**slide_window 对 residual_kl block refresh 额外增加 ≈ +71 GB**

**residual_kl + slide_window 是最极端配置**：单 block refresh 峰值 **≥ 140 GB**（4B, BWD_BSZ=32）。**几乎任何常见显卡都 OOM**。

---

## 版本 B 各阶段峰值汇总（4B 默认 sweep 配置，**大概率 OOM**）

| Stage | 不开 slide | 开 slide |
|-------|-----------|---------|
| 0 input capture | 12 GB | 12 GB |
| 1 static precompute | 42 GB | 42 GB |
| **2 fp_inps_final 预计算** | **12 GB（期间） + 4 GB 永久** | 同 |
| 3c stats collection（BSZ=32） | **120 GB ❌** | 124 GB ❌ |
| 3c stats collection（BSZ=4） | 40 GB | 44 GB |
| 3d Hessian accum（HA=128） | 66 GB | 70 GB |
| **3f fasterquant + refresh（BWD=32）** | **~100 GB ❌** | **~170 GB ❌❌** |
| 3f fasterquant + refresh（BWD=4） | ~28 GB | ~40 GB |
| 4 lm_eval | 19 GB | 19 GB |

**residual_kl + slide 的 slide_window 相对不开的额外显存**：
- Stage 3b 常驻：**+4.2 GB**
- Stage 3f block refresh 峰值：**+71 GB**（是 fisher_diag_mse 的 ~6 倍！）
- **合计增量 ≈ +75 GB**

---

# 版本 C：`GRAD_REFRESH_LOSS = refined_mse`

refined_mse = fisher_diag_mse（二阶项）+ 一阶项 $g\cdot\Delta y$ 。详见 [README 的 refined_mse 详解](README.md#refined_mse-详解)。显存特征可以这样理解：

- **二阶项**和版本 A 完全一致（共用 fisher 预计算、共用 `compute_refresh_loss` 里的 fisher_diag_mse 计算路径）。
- **一阶项**引入两部分新开销：(a) 每 layer 量化前的一次端到端反传采 grad pool；(b) per-mini-batch refresh 时按 pool 查表取精准 grad / mean grad。
- **不额外触发 `hidden2logits`**，block refresh 的峰值接近版本 A（不是版本 B）。

## Stage 0, 1 — 与版本 A 相同

Stage 1 precompute 里 `collect_fisher=True` 触发条件从 `fisher_diag_mse` 拓展到 `("fisher_diag_mse", "refined_mse")`，显存峰值不变。

## Stage 2 — `fp_inps_final` 预计算 ⚡

refined_mse 需要 teacher 的 final hidden（用于端到端 KL），同版本 B 一样触发。

| 对象 | 公式 | 4B |
|------|------|-----|
| `scratch` 缓冲（bf16） | `2·n_local·T·d` | 4 GB |
| 全局 `fp_inps_final` 永久缓冲（bf16） | `2·n_local·T·d` | **+4 GB 常驻** |

**Stage 2 峰值 ≈ 12 GB**（期间），**永久开销 +4 GB**

## Stage 2.5 — refined_mse grad pool 采集（每层都做） ⚡⚡

这是 refined_mse 独有的阶段，每量化一层前触发一次。伪代码：

```python
# 下游 layer + lm_head + final norm 临时搬 dev
for k in range(i+1, L):
    layers[k].to(dev)
# 关掉 layer[i..L-1] 的 act-quant wrapper（FP 前向）
# 冻结全部 param，只让 activation 有 grad
for start in range(0, N_pool, global_loss_bsz_local):
    h = inps[batch].detach().clone().requires_grad_(True)
    out_i = layer(h, ...)[0]
    for k in range(i+1, L):
        out_i = layers[k](out_i, ...)[0]
    logits_student = hidden2logits(out_i, analyzer)         # (B, T, V)
    logits_teacher = hidden2logits(fp_inps_final[batch])    # (B, T, V)
    kl = F.kl_div(log_softmax, softmax).sum(-1).mean()
    grad = torch.autograd.grad(kl, out_i)[0]                # (B, T, d)
    grad_pool[start:start+B] = grad.to(bf16).cpu()
# 下游 layer 搬回 CPU；act-quant wrapper 还原
```

每 batch 内的显存：

| 对象 | 公式 | 4B, GL_BSZ=4, N_pool=32 |
|------|------|------|
| 全下游 layer 常驻 GPU（bf16） | `2·P·(L-i)/L` | 峰值在 i=0 时 ≈ `2·P` = **8 GB** |
| 反传用 `h_in + out_i + 下游激活图` | `~10·GL_BSZ·T·d·(L-i)·2` | 峰值 i=0 ≈ **15 GB** |
| `logits_student + logits_teacher` | `2·GL_BSZ·T·V·2` | **2.5 GB** |
| autograd softmax 中间量 | `~2·GL_BSZ·T·V·2` | 2.5 GB |
| `fp_inps_final[batch]` 切片 | `2·GL_BSZ·T·d` | 0.05 GB |
| **grad_pool**（CPU bf16，不占 GPU） | `2·N_pool·T·d` | 0.3 GB（CPU，逐 layer 丢弃） |
| `mean_grad`（fp32 on dev） | `4·d` | 10 KB（忽略） |
| 上一 layer 的 inps/fp_inps 常驻 | `4·n_local·T·d` | 8 GB |

**Stage 2.5 峰值（layer i=0） ≈ `2P + 10·GL_BSZ·T·d·L·2 + 2·GL_BSZ·T·V·2·2 + 4·n_local·T·d`**
**4B (GL_BSZ=4, N_pool=32) ≈ 8 + 15 + 5 + 8 = 36 GB**（i=0）  
**layer i=L/2 时 ≈ 8 + 7.5 + 5 + 8 ≈ 28 GB**，layer i=L-1 时退化到 ≈ 8 + 0 + 5 + 8 = 21 GB（只有 lm_head/norm，没下游 block）。

**压这一阶段显存的主杠杆**：
- `GLOBAL_LOSS_BSZ`（直接线性影响激活图 + logits）
- `N_pool = NUM_SAMPLES_FOR_REFINED_MSE`（不影响单次峰值，只影响 batch 循环次数）
- `kl_topk > 0` 会让 student 侧用 gather 瘦下来，显存略少

**注意**：每次进 Stage 2.5 前下游 layer 还在 CPU，进入后需要瞬时搬 ~`2P(L-i)/L` bf16 的权重到 dev。7B 模型从 layer 0 开始 ≈ 14 GB 的 CPU→GPU 拷贝，~1 秒（PCIe 4.0）或更久，逐层递减。

## Stage 3a/3b/3c/3d — 与版本 A 相同

Stage 3c 的 stats 收集用 `gptq_reference_loss_type`，对 refined_mse 来讲 `enable_gptq_plus=0` 是硬性要求（argparse 校验），所以 `skip_ref_backward=True`，`gradient_loss` 这条路不跑。→ **比版本 A 的 3c 还少一次 backward**，峰值不变但常数时间更少。

fisher 切片路径（`batch_layer_output_fisher`）在 `layer_refresh_loss_type in ("fisher_diag_mse", "refined_mse")` 时走得一模一样，显存一致。

## Stage 3f — fasterquant + block refresh

### 3f-1: fasterquant 初始化 — 与版本 A 相同

### 3f-2: Block refresh（refined_mse，**禁止** slide_window）

每个 block 尾做一次 `gradient_refresh_fn`：用 `BWD_BSZ` 样本做 forward+backward。比版本 A 额外新增的只是一阶项的 pool 查表 / mean 计算：

| 对象（每 block） | 公式 | 4B, BWD_BSZ=32 |
|------|------|------|
| 单层 forward 图（与 A 相同） | `~10·BWD_BSZ·T·d·2` | 10 GB |
| `override_weight`（fp32） | `4·rows·cols` | 0.2 GB |
| **`fisher_batch`**（fp32 slice）— 与 A 完全相同 | `4·BWD_BSZ·T·FNG` | 0.25 GB |
| **`delta²` 二阶项**（fp32） | `4·BWD_BSZ·T·d` | 0.65 GB |
| **`refined_mse_exact_batch`** — pool 命中样本的精准 grad，cast 到 bf16 | `2·B_pool·T·d` | ~0.2 GB（B_pool ≤ 32） |
| **`delta.index_select(0, pool_positions)`** 临时 | `2·B_pool·T·d` | ~0.2 GB |
| **`delta[non_pool].mean(1)`** 临时 | `2·B_nonpool·d` | 0.002 GB（忽略） |
| **`mean_grad`** 常驻（fp32 on dev） | `4·d` | 10 KB（忽略） |
| 权重梯度 / snapshot / Adam state | `~20·rows·cols` | ~1 GB |

**Block refresh 峰值（refined_mse） ≈ +12.3 GB**（比 fisher_diag_mse 多 ~0.4 GB，基本持平）

**叠加到 fasterquant 的总峰值**（与 A 相比，只多 pool 相关 ≤0.5 GB）：
**≈ 28–30 GB**（4B, BWD_BSZ=32）

### 3f-3: slide_window ❌

refined_mse + slide_window 在 argparse 层面被拒绝（next-layer grad pool 没实现，v1 不支持）。

## Stage 4 — 与版本 A 相同

---

## 版本 C 各阶段峰值汇总（4B 默认 sweep 配置）

| Stage | refined_mse |
|-------|-------------|
| 0 input capture | 12 GB |
| 1 static precompute | 42 GB |
| **2 fp_inps_final** | **12 GB（期间） + 4 GB 永久** |
| **2.5 grad pool 采集**（i=0，峰值） | **36 GB** |
| 2.5 grad pool 采集（i=L/2） | 28 GB |
| 3c stats collection（BSZ=32） | 60 GB |
| 3d Hessian accum（HA=128） | 66 GB |
| 3f fasterquant + refresh | 28 GB |
| 4 lm_eval | 19 GB |

**全流程峰值 ≈ 66 GB**（Stage 3d 仍然是瓶颈，和版本 A 一致）—— A100-80G 可撑住。

**相对版本 A（fisher_diag_mse）的增量**：
- Stage 2 `fp_inps_final` 永久：**+4 GB**
- Stage 2.5 grad pool 每层采集：**~30 GB 瞬时**（但不常驻，采完就释放下游 layer 权重）
- Stage 3f block refresh 瞬时：**+0.5 GB**（index_select + cast）
- Stage 3c 反而少一次 `gradient_loss` backward（`enable_gptq_plus=0` 要求）

**关键区别 vs 版本 B (residual_kl)**：block refresh 里**不跑 `hidden2logits`**，因此没有 B 路径里那个致命的 `B·T·V` logits 爆炸。refined_mse 的显存特征基本是"A + 一次性 grad pool 采集"，block refresh 阶段对 B 显存压力远小于 residual_kl。

---

## 推荐配置（refined_mse）

| 显卡 | BSZ | HA_BSZ | BWD_BSZ | GL_BSZ | N_pool |
|------|-----|--------|---------|--------|--------|
| A100-80G | 32 | 32–64 | 32 | 4 | 32 |
| A100-40G | 16 | 16 | 16 | 2 | 32 |
| 24G（3090/4090） | 8 | 8 | 8 | 1 | 16（减 pool 也能减 Stage 2.5 次数→省累计时间） |

主要压显存杠杆：
1. **GLOBAL_LOSS_BSZ** — 同时影响 Stage 1 precompute 和 Stage 2.5 的激活图 / logits 峰值
2. **BSZ** / **HA_BSZ** — 同版本 A
3. **NUM_SAMPLES_FOR_REFINED_MSE** — 不影响单次峰值，只影响 Stage 2.5 累计时间（采 batch 数 = `N_pool / (GL_BSZ // world)`）

---

## 跨版本推荐配置

### fisher_diag_mse（主推）

| 显卡 | BSZ | HA_BSZ | BWD_BSZ | slide |
|------|-----|--------|---------|-------|
| A100-80G | 32 | 32–64 | 32 | ✅ |
| A100-40G | 16 | 16 | 16 | ✅ |
| 24G（3090/4090） | 8 | 8 | 8 | ⚠️ 或关 |

### residual_kl（显存重度消耗）

| 显卡 | BSZ | HA_BSZ | BWD_BSZ | slide |
|------|-----|--------|---------|-------|
| A100-80G | 4 | 32 | 4 | ⚠️ 需实测 |
| A100-40G | 4 | 16 | 2 | ❌ 建议关 |
| 24G | 2 | 8 | 1 | ❌ |

**关键观察**：`residual_kl` 的显存代价主要来自 `hidden2logits`。`d`、`V` 越大，代价越高。Qwen3-4B V=150k 比 Llama2-7B V=32000 吃显存多 ~5 倍。如果你用小 V 的模型（早期 Llama、Phi 系列），residual_kl 的配置可以比上表宽松。

---

## 杠杆优先级速查

按影响力排序（每增 1 单位显存成本）：

1. **residual_kl 的 BSZ/BWD_BSZ** — 影响 `B·T·V`，最凶
2. **stats 的 BSZ** — 影响 `B·T·V`（logits），次凶
3. **HA_BSZ** — 影响 `NG·B·T·d_ff`（weighted），down_proj 时很重
4. **GL_BSZ** — 影响 `B·T·d·L`（全模型激活图），对 refined_mse 是 Stage 2.5 峰值的主杠杆
5. **slide_window** — 对 residual_kl 加倍 block refresh 峰值；fisher_diag_mse 影响适中；refined_mse 不支持
6. **NG** — 影响所有 `H` 和 `weighted`（倍数为 4）
7. **LM_EVAL_BSZ** — 只影响 eval 阶段，隔离度好
8. **BACKWARD_SAMPLES / NUM_SAMPLES_FOR_REFINED_MSE** — 不影响单次峰值，只影响迭代 / Stage 2.5 累计时间

---

## 附：跨模型规模速查（fisher_diag_mse + slide, 默认 sweep 配置）

| 模型 | P | d | d_ff | V | 3c 峰值 | 3d 峰值 | 3f 峰值 | 最小建议显卡 |
|------|-----|-----|------|-----|--------|--------|--------|------------|
| Qwen3-0.6B | 0.6B | 1024 | 3072 | 151k | 35 GB | 18 GB | 22 GB | 24G |
| Qwen3-4B | 4B | 2560 | 9728 | 152k | 64 GB | 70 GB | 40 GB | A100-80G |
| Llama2-7B | 7B | 4096 | 11008 | 32k | 45 GB | 90 GB | 42 GB | A100-80G |
| Llama3-8B | 8B | 4096 | 14336 | 128k | 90 GB | 120 GB | 55 GB | H100-80G 紧 |
| Qwen3-14B | 14B | 5120 | 17408 | 152k | 115 GB | 175 GB | 75 GB | 多卡 |

（3c 峰值 = BSZ=32 情况下；V 越大越贵；d_ff 越大 HA 阶段越贵）
