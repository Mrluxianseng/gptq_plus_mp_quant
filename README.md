# 算法流程

1. rotate

2. 预计算整个模型的Saliency（用来得到每一层对端到端loss的hessian）和fisher系数（fisher diag mse loss使用）

3. 开始量化每个layer，计算hessian和此时的loss、gradient。

4. 开始量化layer内每个线性层，先调用find_params确定per-channel的scale/zero（groupsize=-1），然后对列做act_order排序

5. 将列维度切成block，开始block循环，每次对block内逐列量化并调整block内其他权重进行补偿（gptq式）

6. block外的权重进行gptq式的补偿更新，补偿block内的量化误差

7. 从校准集中取BACKWARD_SAMPLES条样本，计算loss，反向传播得到梯度，做梯度下降。
   
8. 循环结束后还原列顺序

# 参数说明

## 运行脚本：scripts/gptq_plus_lr_sweep.sh

- NUM_GROUPS：调整hessian的共享group数，默认4表示一个线性层保留4个hessian矩阵(num_col * num_col)
- GRAD_LRS_STR：梯度下降学习率
- N_SAMPLES：校准集总样本量
- SEQ_LEN：校准集样本长度
- BSZ：每层计算saliency/hessian/gradient等用的batch_size
- FINAL_LAYER_STATS_BSZ：最后一层单独设置计算saliency/hessian/gradient等用的batch_size，因为最后一层有输出头比较重
- HESSIAN_ACCUM_BSZ：stats阶段跑完之后，还要再挂`add_batch` hook把inps过一遍layer累hessian，这个就是那个forward循环的batch_size。和BSZ解耦（stats阶段吃logits反传显存，这个阶段吃`add_batch`里`weighted`张量的显存），留空则复用BSZ。默认128。OOM时单独调小。
- BACKWARD_SAMPLES：每block后用来计算loss的样本条数
- BACKWARD_BSZ：每block后用来反传的batchsize
- FINAL_LAYER_BACKWARD_BSZ：最后一层反传的batchsize
- FINAL_LAYER_FULL_BACKWARD：最后一层是否每个block都用上所有校准样本
- BLOCKSIZE：列切块的大小
- BLOCK_ATOMIC_QUANT：开启后一个block同时量化，内部不再列循环
- GRAD_OPTIMIZER：梯度下降用优化器
- FINAL_LAYER_GRAD_OPTIMIZER：最后一层用的优化器
- GRAD_CLIP：梯度下降用的逐元素clip
- GRAD_REFRESH_LOSS：梯度下降用的loss类型
- FINAL_LAYER_GRAD_LR：最后一层的学习率
- PRE_GD_STEPS：弃用了
- PRE_GRAD_LR：弃用
- PRE_FINAL_LAYER_GRAD_LR：弃用
- PRE_FINAL_LAYER_GRAD_OPTIMIZER：弃用
- GRAD_REG_STRATEGY：梯度下降用的正则项（有时间可以测测，感觉没啥用）
- GRAD_REG_LAMBDA：l2和hessian正则的系数
- GRAD_GATE_FLOOR：quant_error_gate和quant_error_gate_optimized用的系数
- GRAD_GATE_SHARPNESS：quant_error_gate和quant_error_gate_optimized用的系数
- GRAD_GATE_SINE_AMP：quant_error_gate_optimized用的系数
- GRAD_HESSIAN_TOPK：计算saliency和fisher用的logits topk（如果用的不是端到端的kl(global_loss=0)，一定要设成-1，因为逐层的输出头的topk不一定是一样的）
- SALIENCY_CLIP_PERCENTILE：在precompute阶段把每个token的saliency（`grad²(NLL_loss, module_output).mean(group)`）裁到这个分位数，默认0.99。用来压掉深层（layer 24+）NLL backward产生的极端gradient outlier（有几个token的saliency可以比中位数大10-12个数量级）。不裁的话这些outlier会让后续的`inp.T @ diag(s) @ inp`变成近rank-1的病态矩阵，Cholesky即使damp涨到0.5+还是失败。设成1.0可以关掉裁剪。改了这个值会让static_cache_path的cache key变（key里带`salclip{value}`），重新预计算。
- PROJ_LR_SCALE：调整o_proj层用的学习率
- DOWN_PROJ_LR_SCALE：调整down_proj层用的学习率（这个层一般比较爆炸）
- SECOND_ORDER_SCALE：调整gptq式二阶更新的scale，固定为1就行
- FISHER_NUM_GROUPS：每layer输出空间的fisher系数保留多少组（组内共享系数）
- NUM_SAMPLES_FOR_REFINED_MSE：`GRAD_REFRESH_LOSS=refined_mse` 时才用。每 layer 端到端反传采集 grad 的 per-rank pool 大小，默认 32。需要 `<= nsamples // world` 且 `% (global_loss_bsz // world) == 0`。和 `LOSS_SLIDE_WINDOW=1` 共同开启时，同一次反传会追加采集下一层输出的 grad pool（同一批样本、同一个 graph），供 slide-window 里的下一层 refresh loss 用，不另外跑 forward。
- REFINED_MIX_SPLIT_LAYER：`GRAD_REFRESH_LOSS=refined_mix` 时才用。前 `SPLIT` 层用 refined_mse，后 `N-1-SPLIT` 层用 refined_residual_kl，final layer 照旧 kl。空值 = 运行时取 `N // 2`。改这个值会让 cache key 变（后缀 `_mixsplit{N}`），需要重新 precompute。
- REFINED_MIX_RKL_LR_RATIO：`GRAD_REFRESH_LOSS=refined_mix` 时才用。后半段 layer (refined_residual_kl) 的 `GRAD_LR` 和 `PRE_GRAD_LR` 都乘上这个标量，默认 1.0（两段同 lr）。Final layer 不走这个比例，用 `FINAL_LAYER_GRAD_LR` 独立控制。
- PRE_CLIP：弃用，固定为0
- GLOBAL_LOSS：调整gptq的hessian估计以及fisher mse loss的fisher系数使用端到端的kl loss，固定为1就行
- GLOBAL_LOSS_BSZ：预计算整个模型反传时的batchsize
- LOSS_SLIDE_WINDOW：开启后会同时计算本层和下一层的loss并线性配比
- DP_GLOBAL_SHUFFLE：开启dp后的数据shuffle模式，固定为1就行
- GRAD_LR_LAYER_SCHEDULE：跨layer的学习率调度器
- ALPHA：中兴的算法用的，固定为0就行
- KL_TOPK：计算kl和res kl用的topk
- LM_EVAL_BATCH_SIZE：eval时batchsize
- ENABLE_QA_EVAL：开启qa_eval打分，慢
- BASE_EXP：实验名
- OUTPUT_ROOT：实验日志输出
- ENABLE_GPTQ_PLUS：GPTQ+一阶项开关，完全等价于把alpha调成0。设0时只关掉fasterquant内循环/外更新里的GHinv一阶项（GHinv/Z/beta全部变0），其他所有机制（block_gd梯度下降、loss_slide_window、pre_gd_steps、fisher预计算、residual_kl的fp_inps_final预计算）照常运行，与alpha的语义解耦。作为性能优化，stats阶段会跳过reference loss的backward（因为它算出来的权重梯度会被beta=0乘掉，无用）。做纯GPTQ一阶/二阶对照实验时设为0即可。
- FSDP_PRECOMPUTE：在`collect_static_end_to_end_saliency_and_fisher`里用FSDP2把模型权重+梯度分片到各个rank上，给单卡放不下整模型backward的大模型用。权重dtype保持不变，grad dtype fp32；因为precompute里所有param都会被冻结（`requires_grad=False`），所以FSDP的reduce_scatter路径不触发，只走param的all_gather。**precompute结束后会自动unwrap FSDP**（用进FSDP前的CPU快照把DTensor param复原成普通Tensor，并摘掉forward/backward hooks），所以同一个run可以直接接着跑正常的per-layer量化，不需要分两阶段跑。但unwrap靠的是CPU snapshot，需要一份完整模型大小的额外CPU RAM：4B≈+8GB、7B≈+14GB、13B≈+26GB可以直接跑；**70B需要+140GB CPU RAM，大概率爆cgroup，这种情况必须走两阶段**（`EXIT_AFTER_PRECOMPUTE=1 STATIC_CACHE_PATH=...`先存盘退出，再换一次run读盘量化）。
- FSDP_CPU_OFFLOAD：开了FSDP后，把param的shard放在pinned CPU内存里，每次forward前all_gather到GPU、forward后释放。进一步省GPU显存，代价是多一轮CPU↔GPU带宽。
- STATIC_CACHE_PATH：precompute结果的磁盘缓存目录。key绑定`model/dataset/nsamples/seq_len/rotate/num_groups/fisher_num_groups/grad_hessian_topk/global_loss_bsz/seed/world_size/rank`，每rank存自己的分片（`_world{W}_rank{R}.pt`）。cache命中时跳过precompute直接读盘，在sweep不同lr之间复用同一份saliency/fisher、或者70B走两阶段工作流时用。要复用cache必须用相同的world_size和rank分配，否则cache miss重算。
- EXIT_AFTER_PRECOMPUTE：precompute完成+结果存盘后直接退出（跳过量化和eval），专门给70B的两阶段流程用。默认0。
- BASE_EXP：实验名
- OUTPUT_ROOT：实验日志输出

## FSDP使用说明

### 何时开启

FSDP只在有帮助的时候开。它**只shard参数+梯度，不shard激活值**，所以对激活主导的小模型收益很小：

| 模型规模 | 建议 |
|---------|-----|
| ≤13B | 不开FSDP。小模型激活占显存大头，参数分片省不了多少。直接跑就行，甚至降`GLOBAL_LOSS_BSZ=1`更有效 |
| 30B左右 | 可开可不开。A100-80G够用时不开；更小卡用FSDP会有帮助 |
| **≥70B** | **必须开FSDP**。单机单卡放不下140GB权重+280GB fp32梯度。**且必须走两阶段**，否则in-process unwrap的CPU快照会爆内存 |

### 一阶段用法（4B / 7B / 13B，默认）

直接开`FSDP_PRECOMPUTE=1`跑就行，precompute完会自动unwrap：
```bash
FSDP_PRECOMPUTE=1 bash scripts/gptq_plus_lr_sweep.sh <model_path> 4 0,1
```

不过这个规模**开FSDP通常不会让你更快或更省**，因为激活内存才是瓶颈。真想压显存优先考虑`GLOBAL_LOSS_BSZ=1`。

### 两阶段用法（70B必用）

70B整模型放GPU做backward需要超过400GB显存，单卡搞不定；而in-process unwrap又需要140GB CPU RAM做快照，也会爆。解决办法：**第一个run用FSDP跑precompute、存盘、退出；第二个run不用FSDP、读盘、跑量化**。

**Stage 1：FSDP下precompute + 存盘 + 退出**
```bash
# 8×80G跑precompute，按rank分片存到./cache/static_stats
FSDP_PRECOMPUTE=1 FSDP_CPU_OFFLOAD=1 \
  EXIT_AFTER_PRECOMPUTE=1 \
  STATIC_CACHE_PATH=./cache/static_stats \
  GLOBAL_LOSS_BSZ=1 \
  bash scripts/gptq_plus_lr_sweep.sh /path/to/Llama-2-70b-hf 4 0,1,2,3,4,5,6,7
```

**Stage 2：关FSDP，读盘接着量化**
```bash
FSDP_PRECOMPUTE=0 \
  STATIC_CACHE_PATH=./cache/static_stats \
  bash scripts/gptq_plus_lr_sweep.sh /path/to/Llama-2-70b-hf 4 0,1,2,3,4,5,6,7
```

两阶段必须用相同的`N_SAMPLES / SEQ_LEN / NUM_GROUPS / FISHER_NUM_GROUPS / GRAD_HESSIAN_TOPK / SALIENCY_CLIP_PERCENTILE / GLOBAL_LOSS_BSZ / 种子 / rotate开关 / world_size`，否则cache key对不上会重算precompute。

# 核心的消融/创新点

## blockwise梯度下降

- 缓解误差跨层累积，允许后续层修正前面层的误差，且数值精确不依赖低精度的hessian

学习率需要好好调一下，0.6b上对学习率挺敏感的。有时间的话最后一层学习率要单独调一下，不然直接跟前面层保持一致就行。

优化器也可以实验一下

## loss type

- fisher_diag_mse：将最后一层（开启global loss）输出的kl loss在当前层的输出隐空间上二阶展开： $loss = \frac{1}{2} \Delta y ^T H \Delta y$ ，其中H使用fisher矩阵估算对角项。

这里之前做过逐样本group维度l2norm为1的标准化，现在删掉了（效果更好），就是严格二阶展开式，不加任何归一化。

- refined_mse：在fisher_diag_mse的基础上再加一阶项 $g \cdot \Delta y$ ，即端到端kl loss对当前layer输出的完整Taylor二阶展开式。详见 [refined_mse 详解](#refined_mse-详解)。支持和 loss_slide_window 同开（同一次反传顺带采下一层 grad pool），但要求 `--enable_gptq_plus 0`（v1 限制，理由见下）。

- res_kl：假设当前层的全精度模型输出为x，量化模型输出为x+Δx，假设后续层的变换可以近似为恒等变换加上一个较小的函数f，则全精度模型最终输出： $x+f(x)$ ，量化模型输出： $x+Δx+f(x+Δx)$ 约等于 $x+Δx+f(x)$ ，因此可以考虑直接比较这两个量过输出头之后的kl loss。其中 $x+f(x)$ 一开始就缓存下来了（全精度模型的激活值）。

- refined_res_kl：在res_kl中，我们近似认为 $x+Δx+f(x+Δx)$ 约等于 $x+Δx+f(x)$ ，现在我们多近似一阶： $x+Δx+f(x+Δx)$ 约等于 $x+Δx+f(x)+Δx \cdot \nabla f(x)$ 。而这个 $\nabla f(x)$ 我们用一个常量Jacobi矩阵J去近似。在量化开始前预计算全模型反向传播时，对于每一layer，假设输出隐向量梯度为dx，假设最后一层输出隐向量梯度为dy，则有： $dx=(I+J)dy$ 即 $dx-dy = Jdy$ ，这个可以用最小二乘法直接拟合，从而得到每一层的J矩阵并存在cpu。除此之外定义num_A表示可以保存多少个样本的J矩阵。刚刚假设所有样本共用一个，现在改成总共有num_A个分组共用。

- refined_mix：前一半 transformer layer 用 refined_mse，后一半用 refined_res_kl，最后一层沿用现有规则强制 kl。直觉是前半段层输出一阶项 $g$ 数值小、采样噪声占优，refined_mse 的"每层实采 NUM_SAMPLES_FOR_REFINED_MSE 个样本做端到端 backward"很贵且边际收益有限；而后半段层 $g$ 变大、refined_res_kl 的线性 Jacobi 近似误差反而更明显，所以用 refined_mse 更精细。两种 loss 的预计算只各收自己需要的那一半（fisher 只收前半层、A 只拟合后半层，和 refined_mse / refined_res_kl 的 precompute 是同一次端到端 backward，只是 hook 挂的 layer 不同），相应 CPU RAM 约各省一半。切换边界 layer `i = split - 1` 上 `loss_slide_window` 自动禁用（下一层 loss type 变了、pool / A 形态对不上）。详见 [refined_mix 详解](#refined_mix-详解)。要求 `--global_loss` + `--enable_gptq_plus 0`（继承 refined_mse 的 v1 限制）。

可以做一下实验看一下这个loss（包括fisher mse loss）的梯度对真正kl loss的梯度的近似水平好不好，可以通过cosine相关性来测一下。

## refined_mse 详解

### 数学定义

记端到端 KL loss 为 $L = \text{KL}(\text{student}\_{\text{logits}}(y_i)\,\|\,\text{teacher}\_{\text{logits}}(y_i^{\text{fp}}))$ ，其中 $y_i$ 是 layer $i$ 的输出隐向量。把 $L$ 在 fp 输出 $y_i^{\text{fp}}$ 处做二阶 Taylor 展开：

$$
L(y_i^{\text{fp}}+\Delta y_i) \;\approx\; L(y_i^{\text{fp}}) \;+\; g_i^{T}\Delta y_i \;+\; \tfrac{1}{2}\,\Delta y_i^{T} H_i\, \Delta y_i
$$

其中 $g_i = \left.\partial L/\partial y_i\right|\_{y_i=y_i^{\text{fp}}}$ ，$H_i$ 是 $L$ 在 $y_i^{\text{fp}}$ 处的 Hessian。 $L(y_i^{\text{fp}})$ 对当前层权重不依赖、求梯度为 0，舍去。 $H_i$ 用 **empirical Fisher** 的对角近似（和 fisher_diag_mse 完全复用，按 `fisher_num_groups` 切 group）。忽略一阶项就是 fisher_diag_mse；refined_mse 把一阶项加回来。

一阶项 $g_i$ 取决于"从当前 layer $i$ 输出到端到端 KL 的反传"。在 fp 模型上 $g_i\equiv 0$（student 完全等于 teacher，KL=0，梯度处处为 0）——所以**必须在量化进行到 layer $i$ 的那一刻、上游 $0..i-1$ 已经量化的状态下**去采 $g_i$ ，才能拿到非零值。这一点是 refined_res_kl 在预处理一次就把 A 矩阵全收完的做法做不到的。

### 实现细节（代码怎么做）

整个流程完全嵌进主量化循环（[gptq_fwrd](gptq_utils/gptq_plus_utils.py)），每一层独立算一次 grad pool 然后用于该层的 refresh：

**Step 1 — 采样（[gptq_plus_utils.py:4738-4808](gptq_utils/gptq_plus_utils.py#L4738-L4808)）**。开始量化 layer $i$ 之前：
- 用 `random.Random(args.seed + i)` 建独立 RNG，从 `[0, n_local)` 无放回抽 `num_samples_for_refined_mse` 个 rank-local sample id（sorted）。
- 每 layer 独立抽、每 layer 不同样本。layer 0 直接跳过采样（grad 恒为 0），`mean_grad = zeros(H)`。

**Step 2 — 端到端反传（[collect_layer_output_grad_for_refined_mse](gptq_utils/gptq_plus_utils.py#L3192)）**：
- 把 layer $i+1..N-1$ 搬到 `dev`（主循环里下游 layer 平时在 CPU；`layernorm_before_head` / `lm_head` 常驻 dev）。
- `quant_utils.disable_act_quant` 在 layer $i$ 和下游每一个 block 上关掉 act-quant wrapper，使得这次 forward 纯 FP。
- `temporary_requires_grad(...)` 把上述所有模块的 param 冻结（`requires_grad=False`），只让 activation 保留 grad。
- 按 `global_loss_bsz // world` 的 batchsize 循环：
  - `h_in = inps[batch_ids].detach().clone().requires_grad_(True)` 启动 graph
  - `out_i = layer(h_in, ...)[0]`（FP）
  - 逐 block forward 到最后
  - `logits_student = hidden2logits(h, analyzer)`
  - `logits_teacher = hidden2logits(fp_inps_final[batch_ids], analyzer).detach()`
  - `kl_topk > 0` 时 gather 成 top-k
  - `kl = F.kl_div(log_softmax(...), softmax(...), reduction='none').sum(-1).mean()`
  - `grad = torch.autograd.grad(kl, out_i, retain_graph=False)[0]`（**只反传到 `out_i`，不再往前**）
- 所有 batch 的 grad 拼成 `grad_pool` shape `(N_pool, seq, H)`，bf16 存 CPU；顺便算 `mean_grad = grad_pool.float().mean(dim=(0,1))` shape `(H,)` fp32 on dev。
- finally 把下游 layer 搬回 CPU、把 act-quant wrapper 恢复。

**Step 3 — refresh loss 计算（[compute_refresh_loss `refined_mse` 分支](gptq_utils/gptq_plus_utils.py#L3563)）**。block_gd 里每做一次 refresh，`collect_true_weight_gradient` 会按 `batch_indices` 切 mini-batch，每 batch 里：

1. 建 `refined_mse_pool_lookup = {local_idx: pool_pos}`（一次，缓存整个 refresh）
2. 每 mini-batch 对本 batch 样本查表，分两拨：
   - 命中 pool 的样本 → `layer_output_grad_exact` 是 `grad_pool` 相应行的 slice shape `(B_pool, seq, H)`，`pool_positions` 记录这些样本在 batch 里的 row id
   - 没命中的样本 → 用全 pool 的 `mean_grad`
3. `compute_refresh_loss('refined_mse', ...)` 的公式：

$$
\underbrace{\tfrac{1}{2}\sum_g f_g \sum_{i\in g}\Delta y_i^2}\_{\text{fisher\_diag\_mse}} \;+\; \underbrace{\tfrac{1}{B}\Big[\sum\_{b\in\text{pool}}\tfrac{1}{T}\sum\_t g^{\text{exact}}\_{b,t}\cdot \Delta y\_{b,t} + \bar{g}\cdot\sum\_{b\notin\text{pool}}\overline{\Delta y\_b}\Big]}\_{\text{一阶项}}
$$

$\overline{\Delta y\_b} = \tfrac{1}{T}\sum\_t\Delta y\_{b,t}$ 。非 pool 样本用 `mean_grad @ delta.mean(1).t()` 一个 matmul 直接算，**不展开成 `(B, seq, H)` 大张量**。pool 样本的一阶项用 element-wise `(grad_exact * delta_pool).sum(-1).mean(-1).sum()`。

**Step 4 — 资源释放（[gptq_plus_utils.py:5417](gptq_utils/gptq_plus_utils.py#L5417)）**。当前 layer 量化结束时：
```python
del layer_refined_mse_grad_pool
del layer_refined_mse_mean_grad
del layer_refined_mse_pool_ids
# slide_window 开启时还会多 del 下一层的 pool/mean_grad
memory_utils.cleanup_memory()
```
下一层开始时重新采、重新传；不跨 layer 留存。

### 和 loss_slide_window 同开

开启 `loss_slide_window` + `refined_mse` 时，Step 2 的端到端反传会在同一次 `torch.autograd.grad(kl, [out_i, out_{i+1}])` 里**一次性**采出 $g_{i+1}$，存成 `layer_refined_mse_grad_pool_next` / `layer_refined_mse_mean_grad_next`（同一批样本、同一 graph，pool_ids 复用，没有额外 forward）。Block_gd 里 refresh 时，`compute_refresh_loss('refined_mse', ...)` 被调两次：一次用当前层的 $g_i$ 算 `refresh_loss_current`，一次用下一层的 $g_{i+1}$、下一层的 fisher、以及 $\Delta y_{i+1} = \text{next\_layer}(\text{out\_hidden}) - \text{fp\_inps\_next}$ 算 `refresh_loss_next`，最后按 `slide_alpha * current + (1 - slide_alpha) * next` 混。slide_window 只在 `i <= final_layer_idx - 2` 激活（和其他 loss 类型一致）。至 layer 0 时两份 pool 都塞 zeros（student == teacher，两层 KL 梯度都恒零）。

### Fisher 复用

refined_mse 的二阶项走 fisher_diag_mse 完全一样的路径（precompute 阶段把 layer output 的 empirical Fisher 做 grad² 存 CPU bf16），代码里 5 处 `layer_refresh_loss_type == "fisher_diag_mse"` 的硬编码被改成 `in ("fisher_diag_mse", "refined_mse")`：
- `want_fisher`（[gptq_plus_utils.py:4455](gptq_utils/gptq_plus_utils.py#L4455)）— precompute 阶段是否收 fisher
- `need_layer_output_fisher_collection`（[gptq_plus_utils.py:3904](gptq_utils/gptq_plus_utils.py#L3904)）— 兜底 per-layer 收 fisher
- `batch_layer_output_fisher` 切片（[gptq_plus_utils.py:4009](gptq_utils/gptq_plus_utils.py#L4009)）
- `pre_gd_refresh_loss_type` 读 fisher（[gptq_plus_utils.py:4917](gptq_utils/gptq_plus_utils.py#L4917)）
- `precomputed_layer_output_fisher` 传入 `collect_layer_grad_hessian_stats`（[gptq_plus_utils.py:5025](gptq_utils/gptq_plus_utils.py#L5025)）

### layer 0 的退化

layer 0 上游还没量化，student == teacher，KL=0，$g_i\equiv 0$。代码里直接跳过 Step 1-2，把 `pool_ids=empty`、`grad_pool=None`、`mean_grad=zeros(H)` 传下去。`compute_refresh_loss('refined_mse', ...)` 里一阶项整体化为 0，loss 与 fisher_diag_mse 数值一致（已经单元测试验证）。这样 layer 0 的行为既不多花时间收无意义的 grad、也不破坏对 fisher_diag_mse 的向下兼容。

### 超参

| 参数 | 默认 | 约束 | 说明 |
|------|------|------|------|
| `--num_samples_for_refined_mse` | 32 | `>0`, `<= nsamples // world`, `% (global_loss_bsz // world) == 0` | per-rank pool 大小 |
| `--grad_refresh_loss refined_mse` | — | 要求 `--global_loss`，要求 `--enable_gptq_plus 0`；与 `--loss_slide_window` 兼容 | loss 选择 |

### 为什么要求 `enable_gptq_plus=0`（v1 限制）

开 GPTQ+（alpha≠0）时，`collect_layer_grad_hessian_stats` 里会用 `gptq_reference_loss_type` 再跑一次参考梯度反传（用于 GPTQ+ 一阶项的 reference grad）。若此时 `gptq_reference_loss_type == "refined_mse"`，该函数内部调的 `compute_refresh_loss('refined_mse', ...)` 拿不到 pool 参数会 raise。把 pool 参数穿进 `collect_layer_grad_hessian_stats` 才能解锁这一组合，目前暂不做——做纯 refresh loss 对照实验时 `ENABLE_GPTQ_PLUS=0` 是默认选项，影响不大。

### 显存代价

见 [memory_profile.md](memory_profile.md) 的「版本 C: refined_mse」小节。

## refined_mix 详解

### 想法

前半段 transformer layer 的端到端梯度 $g_i$ 本身数值小、采样噪声大，refined_mse 的一阶项收益有限但每层都要付"搬下游 layer 到 GPU → 端到端 backward NUM_SAMPLES_FOR_REFINED_MSE 个样本 → 搬回 CPU"的开销；后半段 layer 的 $g_i$ 变大、refined_res_kl 的线性 Jacobi 近似开始失真，这时补一个一阶 $g\cdot\Delta y$ 更值得。refined_mix 就按深度切一刀，两段各用"当前更合适"的那一种。

### 分层规则

- 默认切分点 `split = N // 2`（N 是 transformer block 总数，不含 lm_head），可用 `--refined_mix_split_layer` 覆盖。
- layer `i ∈ [0, split)`：`refined_mse`
- layer `i ∈ [split, N-1)`：`refined_residual_kl`
- layer `i = N-1`（final layer）：沿用 [get_effective_refresh_loss_type](gptq_utils/gptq_plus_utils.py#L58) 的现有规则，强制 `kl`

### precompute 只收一半 stats

`collect_static_end_to_end_saliency_and_fisher` 的那次端到端 backward 照常跑一次，但：

- fisher 的 grad² 捕获 hook 只挂在 `[0, split)` 的前半段 layer 上 —— 后半段 layer 的 `fisher_data[i]` 留空，aggregate 阶段直接填 `None`。refined_res_kl 不读 fisher，所以后半没这份 stats 不影响正确性。
- refined_res_kl 的 per-layer dy·delta 捕获 hook 只挂在 `[split, N-1)` 的后半段 layer 上（last-layer 的 dy 捕获 hook 仍然挂，它只是公共 buffer 源）。前半段 `static_refined_A[i]` 全 None，主循环里 `layer_refined_A_list=None`，refined_mse 分支不读 A，也不影响。
- **节省估算**（以 Qwen3-4B，28 层，H=2560，seq=2048，512 samples，fisher_num_groups=512 为例）：fisher 每层 CPU bf16 ≈ 1 GB × 28 = 28 GB，砍一半省约 14 GB。A 的 CPU fp32 accumulator 峰值 `2 × H² × 4B × L/2` 省一半约 1.5 GB，加上 bf16 A stack 省约 185 MB。
- Cache key 在 mix 模式下追加 `_mixsplit{N}` 后缀，避免和非 mix run 的 cache 混用。

### 后半段 layer 不再跑 refined_mse 的端到端 backward

主循环里 refined_mse 的 grad pool 采集（[gptq_plus_utils.py:5144](gptq_utils/gptq_plus_utils.py#L5144)）是按 `layer_refresh_loss_type == "refined_mse"` 分派的——mix 模式下后半段 layer 的 `layer_refresh_loss_type` 变成 `refined_residual_kl`，这一大段（搬下游 layer → 端到端 FP forward + backward → collect pool → 搬回 CPU）自动跳过。所以**"把后半段的每层端到端 backward 全部省掉"** 不需要额外代码，只是 loss type 路由的副产品。

### 切换边界 layer 的 slide_window 行为

- 打开 `--loss_slide_window` 时，layer `i` 会同时计算自身的 loss 和 layer `i+1` 的 loss，按 `slide_alpha: 1→0` 线性混合。
- mix 下 `i = split - 1` 那一层，当前 loss 是 refined_mse、下一层 loss 是 refined_residual_kl，两者依赖的 per-layer state 完全不同（refined_mse 要 fisher[i+1] + grad_pool_next；refined_residual_kl 要 A[i+1] + fp_inps_final），也无法跨类型线性混合。所以这一层的 slide_window 自动关闭（`next_layer_same_type` 判空）。其他 layer 的 slide_window 照常。

### 学习率两段制

`--refined_mix_rkl_lr_ratio r`（默认 1.0）：后半段 layer（refined_residual_kl）的 block_gd LR 和 pre_gd LR 都等于 `r × --grad_lr` / `r × --pre_grad_lr`。前半段用原始 `--grad_lr`。Final layer 仍独立使用 `--final_layer_grad_lr` / `--pre_final_layer_grad_lr`。Cosine / linear LR schedule (`--grad_lr_layer_schedule`) 在两段各自内部正常生效。

### 约束

- 必须 `--global_loss`（两半都依赖 precompute 的产物）
- 必须 `--enable_gptq_plus 0`（继承 refined_mse 的 v1 限制，理由见 [refined_mse 详解](#为什么要求-enable_gptq_plus0v1-限制)）
- `--num_samples_for_refined_mse` 的所有整除约束保留（只对前半段有效但按全局校验）
- `--refined_rkl_num_A > 1` 时仍需 `nsamples % num_A == 0`

## loss slide window

- 原本的方法中每个layer都有一个layer wise的loss，但是一旦开始量化下一个layer那么这个loss就会突变导致优化目标瞬移。因此引入了一个slide window loss，具体来说就是在量化layer i的时候同时计算layer i+1的loss，并且在量化layer i的时候逐步调节layer i和layer i+1所占的比重从(1,0)到(0,1)，实现loss的平滑过渡。这在理论上还有一个好处就是使得layer i在量化时不仅关注于降低layer i及以前的量化误差，同时也关注于layer i的量化不要对后面的层产生太大的量化误差积累影响。

目前支持的 refresh loss：`fisher_diag_mse` / `residual_kl` / `refined_residual_kl` / `refined_mse` / `refined_mix`（`refined_mse` 的做法见 [refined_mse 详解 · 和 loss_slide_window 同开](#和-loss_slide_window-同开)；`refined_mix` 在切换边界 layer 上会自动禁用 slide_window，见 [refined_mix 详解](#refined_mix-详解)）。

这个一定要测一下，我感觉效果挺好的。

## 正则化和学习率调度器（次要）

- l2正则项： $0.5\lambda(W-W_{fp})^2$
- 根据hessian的正则项： $0.5\lambda(W-W_{fp})^T H (W-W_{fp})$
- quant error gate：根据目前权重与最近量化格点的距离动态调整学习率：

定义归一化距离d：

$$
d = \frac{|w-q_{nearest}|}{quant\_scale}
$$

学习率门控系数g：

$$
g(d)=f + (1-f)(1-e^{-k d})
$$

其中k和f为超参。

- quant error gate optimized：添加一个cosine曲线的loss正则，波谷与量化格点对齐，相当于减小权重被量化产生的量化误差的期望。但其实很多误差来源于w_clip。所以感觉没啥用。

有时间就测一下。

# TODO

按重要性：

- 测fisher diag mse loss和res kl（llama2-7b/13b/70b）
- 测loss slide window
- 试试给res kl加二阶修正
- 测一些极低精度/激活值量化的数据 W4A4 W2A4等
- 测一下正则化
- refined_mse 现在采集 grad 时是每 GPU 存一份完整模型权重、把下游 transformer block 临时搬到 dev 做端到端反传。大模型（70B+）装不下时需要像 `collect_static_end_to_end_saliency_and_fisher` 那样上 FSDP2 shard。
- 测一下调度器