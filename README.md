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
- GROUP_PARALLEL_QUANT：GPTQ+线性层内输出group并行量化模式。这里的group是`NUM_GROUPS`对应的Hessian/saliency输出通道group，不是`W_GROUPSIZE`对应的量化参数列group。`none`保持旧的逐group循环；`tensor`在单rank内把group堆成torch tensor并行跑block内/外更新和block_gd；`rank`要求每个线性层的输出行数能被`WORLD_SIZE`整除，在DP多rank时按连续输出行切分block内量化，每行仍映射回所属Hessian group使用对应Hessian，block结束后同步block结果，block外更新和block_gd仍在每个rank本地用完整group tensor并行执行。rank模式下线性层开始前的per-row weight quant params也按同一输出行段分给不同rank计算，随后同步scale/zero。
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
- SALIENCY_CLIP_PERCENTILE：在precompute阶段把每个中间层token的saliency（`grad²(NLL_total, module_output).mean(group)`）裁到这个分位数，默认0.99。这里 `NLL_total` 是对输出sample×输出token求和的 sampled NLL；`mean(group)` 只是在模块输出通道group内平均，不是在中间层token维平均。用来压掉深层（layer 24+）NLL backward产生的极端gradient outlier（有几个token的saliency可以比中位数大10-12个数量级）。不裁的话这些outlier会让后续的`inp.T @ diag(s) @ inp`变成近rank-1的病态矩阵，Cholesky即使damp涨到0.5+还是失败。设成1.0可以关掉裁剪。改了这个值会让static_cache_path的cache key变（key里带`salclip{value}`），重新预计算。
- PROJ_LR_SCALE：调整o_proj层用的学习率
- DOWN_PROJ_LR_SCALE：调整down_proj层用的学习率（这个层一般比较爆炸）
- SECOND_ORDER_SCALE：调整gptq式二阶更新的scale，固定为1就行
- NUM_SAMPLES_FOR_REFINED_MSE：`GRAD_REFRESH_LOSS=refined_mse` 时才用。每 layer 端到端反传采集 grad 的 per-rank pool 大小，默认 32。需要 `<= nsamples // world` 且 `% (global_loss_bsz // world) == 0`。和 `LOSS_SLIDE_WINDOW=1` 共同开启时，同一次反传会追加采集下一层输出的 grad pool（同一批样本、同一个 graph），供 slide-window 里的下一层 refresh loss 用，不另外跑 forward。
- REFINED_MIX_SPLIT_LAYER：`GRAD_REFRESH_LOSS=refined_mix` 时才用。前 `SPLIT` 层用 fisher_diag_mse，后 `N-1-SPLIT` 层用 refined_residual_kl，final layer 照旧 kl。空值 = 运行时取 `N // 2`。改这个值会让 cache key 变（后缀 `_mixsplit{N}`），需要重新 precompute。
- REFINED_MIX_RKL_LR_RATIO：`GRAD_REFRESH_LOSS=refined_mix` 时才用。后半段 layer (refined_residual_kl) 的 `GRAD_LR` 和 `PRE_GRAD_LR` 都乘上这个标量，默认 1.0（两段同 lr）。Final layer 不走这个比例，用 `FINAL_LAYER_GRAD_LR` 独立控制。
- PRE_CLIP：弃用，固定为0
- GLOBAL_LOSS：调整gptq的hessian估计以及fisher mse loss的fisher系数使用端到端的kl loss，固定为1就行
- GLOBAL_LOSS_BSZ：预计算整个模型反传时的batchsize
- LOSS_SLIDE_WINDOW：开启后会同时计算本层和下一层的loss并线性配比
- DP_GLOBAL_SHUFFLE：开启dp后的数据shuffle模式，固定为1就行
- GRAD_LR_LAYER_SCHEDULE：跨layer的学习率调度器。`cosine` 时非 final layer 的有效学习率为 `base_lr + (lr - base_lr) * 0.5 * (1 - cos(pi * layer_idx / (num_layers - 1)))`
- GRAD_LR_LAYER_BASE_RATIO：跨layer调度器的起始学习率比例，默认0.01，即 `base_lr = 0.01 * GRAD_LR` / `0.01 * PRE_GRAD_LR`。设成0可恢复旧的从0起步的cosine
- ALPHA：中兴的算法用的，固定为0就行
- KL_TOPK：计算kl和res kl用的topk
- LM_EVAL_BATCH_SIZE：eval时batchsize
- ENABLE_QA_EVAL：开启qa_eval打分，慢
- BASE_EXP：实验名
- OUTPUT_ROOT：实验日志输出
- ENABLE_GPTQ_PLUS：GPTQ+一阶项开关，完全等价于把alpha调成0。设0时只关掉fasterquant内循环/外更新里的GHinv一阶项（GHinv/Z/beta全部变0），其他所有机制（block_gd梯度下降、loss_slide_window、pre_gd_steps、fisher预计算、residual_kl的fp_inps_final预计算）照常运行，与alpha的语义解耦。作为性能优化，stats阶段会跳过reference loss的backward（因为它算出来的权重梯度会被beta=0乘掉，无用）。做纯GPTQ一阶/二阶对照实验时设为0即可。
- FSDP_PRECOMPUTE：只用于Stage 1 static saliency/fisher precompute，在`collect_static_end_to_end_saliency_and_fisher`里用FSDP2把模型权重分片到各个rank上，给单卡放不下整模型backward的大模型用。当前支持的稳定工作流是两阶段：Stage 1必须配合`EXIT_AFTER_PRECOMPUTE=1 STATIC_CACHE_PATH=...`写cache后退出，Stage 2重新启动普通量化进程读cache。
- FSDP_CPU_OFFLOAD：开了FSDP后，把param的shard放在pinned CPU内存里，每次forward前all_gather到GPU、forward后释放。进一步省GPU显存，代价是多一轮CPU↔GPU带宽。
- FSDP_META_INIT：precompute阶段的CPU内存优化，默认跟随`FSDP_PRECOMPUTE`开启。它先在rank0准备一份可直接加载的checkpoint，然后所有rank用meta model初始化、先FSDP分片、再把checkpoint加载到本rank shard，避免torchrun每个rank各自持有一整份CPU权重。`rotate=1`时rank0会先单独加载一份完整模型并执行现有fuse/rotate，再把旋转后的checkpoint写到`STATIC_CACHE_PATH/_prepared_checkpoints/`；其他rank不加载完整CPU模型。非rotate且源模型缺少独立`lm_head.weight`时，也会rank0先写一份untied prepared checkpoint。
- STATIC_CACHE_PATH：precompute结果的磁盘缓存目录。key绑定`model/dataset/nsamples/seq_len/rotate/num_groups/full_fisher/grad_hessian_topk/global_loss_bsz/seed/world_size/rank`，每rank存自己的分片（`_world{W}_rank{R}.pt`）。cache命中时跳过precompute直接读盘，在sweep不同lr之间复用同一份saliency/fisher、或者70B走两阶段工作流时用。要复用cache必须用相同的world_size和rank分配，否则cache miss重算。
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

### 两阶段用法（FSDP必用）

70B整模型放GPU做backward需要超过400GB显存，单卡搞不定；普通torchrun启动还会让每个rank先在CPU各加载一整份权重。解决办法：**第一个run用FSDP + meta-init跑precompute、存盘、退出；第二个run不用FSDP、读盘、跑量化**。这样precompute阶段不会出现`world_size × full model`的CPU权重副本；`rotate=1`时仍会由rank0短暂持有一份完整CPU模型来生成已旋转checkpoint。

**Stage 1：FSDP下precompute + 存盘 + 退出**
```bash
# 8×80G跑precompute，按rank分片存到./cache/static_stats
FSDP_PRECOMPUTE=1 FSDP_META_INIT=1 FSDP_CPU_OFFLOAD=1 \
  EXIT_AFTER_PRECOMPUTE=1 \
  STATIC_CACHE_PATH=./cache/static_stats \
  GLOBAL_LOSS_BSZ=8 \
  bash scripts/gptq_plus_lr_sweep.sh /path/to/Llama-2-70b-hf 4 0,1,2,3,4,5,6,7
```

**Stage 2：关FSDP，读盘接着量化**
```bash
FSDP_PRECOMPUTE=0 \
  STATIC_CACHE_PATH=./cache/static_stats \
  bash scripts/gptq_plus_lr_sweep.sh /path/to/Llama-2-70b-hf 4 0,1,2,3,4,5,6,7
```

两阶段必须用相同的`N_SAMPLES / SEQ_LEN / NUM_GROUPS / GRAD_HESSIAN_TOPK / SALIENCY_CLIP_PERCENTILE / GLOBAL_LOSS_BSZ / 种子 / rotate开关 / world_size`，否则cache key对不上会重算precompute。

# 核心的消融/创新点

## blockwise梯度下降

- 缓解误差跨层累积，允许后续层修正前面层的误差，且数值精确不依赖低精度的hessian

学习率需要好好调一下，0.6b上对学习率挺敏感的。有时间的话最后一层学习率要单独调一下，不然直接跟前面层保持一致就行。

优化器也可以实验一下

## loss type

- fisher_diag_mse（名字带diag但现在优化后不再只考虑对角项了，是个遗留问题）：将最后一层（开启global loss）输出的kl loss在当前层的输出隐空间上二阶展开： $loss = \frac{1}{2} \Delta y ^T H \Delta y$ ，其中H使用fisher矩阵估算。

- res_kl：假设当前层的全精度模型输出为x，量化模型输出为x+Δx，假设后续层的变换可以近似为恒等变换加上一个较小的函数f，则全精度模型最终输出： $x+f(x)$ ，量化模型输出： $x+Δx+f(x+Δx)$ 约等于 $x+Δx+f(x)$ ，因此可以考虑直接比较这两个量过输出头之后的kl loss。其中 $x+f(x)$ 一开始就缓存下来了（全精度模型的激活值）。

- refined_res_kl：在res_kl中，我们近似认为 $x+Δx+f(x+Δx)$ 约等于 $x+Δx+f(x)$ ，现在我们多近似一阶： $x+Δx+f(x+Δx)$ 约等于 $x+Δx+f(x)+Δx \cdot \nabla f(x)$ 。而这个 $\nabla f(x)$ 我们用一个常量Jacobi矩阵J去近似。在量化开始前预计算全模型反向传播时，对于每一layer，假设输出隐向量梯度为dx，假设最后一层输出隐向量梯度为dy，则有： $dx=(I+J)dy$ 即 $dx-dy = Jdy$ ，这个可以用最小二乘法直接拟合，从而得到每一层的J矩阵并存在cpu。除此之外定义num_A表示可以保存多少个样本的J矩阵。刚刚假设所有样本共用一个，现在改成总共有num_A个分组共用。

- fisher mse loss效果更好，现在完全用fisher mse

## loss slide window

- 原本的方法中每个layer都有一个layer wise的loss，但是一旦开始量化下一个layer那么这个loss就会突变导致优化目标瞬移。因此引入了一个slide window loss，具体来说就是在量化layer i的时候同时计算layer i+1的loss，并且在量化layer i的时候逐步调节layer i和layer i+1所占的比重从(1,0)到(0,1)，实现loss的平滑过渡。这在理论上还有一个好处就是使得layer i在量化时不仅关注于降低layer i及以前的量化误差，同时也关注于layer i的量化不要对后面的层产生太大的量化误差积累影响。

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

- 正则化现在完全不用，没用

- cosine调度器：学习率随着层数变大而变大，防止在量化前期层的时候梯度更新过大导致量化误差积累到下游。要开，有用，但是激活值量化时从第一层开始就有较大量化误差需要补偿，此时不开。
