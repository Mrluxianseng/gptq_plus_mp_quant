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
- PROJ_LR_SCALE：调整o_proj层用的学习率
- DOWN_PROJ_LR_SCALE：调整down_proj层用的学习率（这个层一般比较爆炸）
- SECOND_ORDER_SCALE：调整gptq式二阶更新的scale，固定为1就行
- FISHER_NUM_GROUPS：每layer输出空间的fisher系数保留多少组（组内共享系数）
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
- ENABLE_GPTQ_PLUS：总开关，0=走纯GPTQ基线，1=开启所有GPTQ+扩展（默认1）。设0后会自动旁路以下计算：stats阶段的reference loss反传（省一次backward）、fisher预计算（省CPU RAM和时间）、residual_kl的fp_inps_final预计算、pre-quantization GD（`run_pre_quant_gd`）、block间的gradient refresh（`gradient_refresh_fn`）、fasterquant逐列内循环的GHinv/Z一阶项（`enable_gradient_update=False`+`alpha=0`使`beta=0`）、loss slide window。等价于把`g_update_mode`强制成`frozen`、`pre_gd_steps=0`、`alpha=0`、`loss_slide_window=0`。适合用来跑时间/精度的GPTQ基线做ablation。

# 核心的消融/创新点

## blockwise梯度下降

- 缓解误差跨层累积，允许后续层修正前面层的误差，且数值精确不依赖低精度的hessian

学习率需要好好调一下，0.6b上对学习率挺敏感的。有时间的话最后一层学习率要单独调一下，不然直接跟前面层保持一致就行。

优化器也可以实验一下

## loss type

- fisher_diag_mse：将最后一层（开启global loss）输出的kl loss在当前层的输出隐空间上二阶展开： $loss = \frac{1}{2} \Delta y ^T H \Delta y$ ，其中H使用fisher矩阵估算对角项。

这里为了保持数值稳定，fisher系数做了逐样本的group维度标准化l2norm为1。

- res_kl：假设当前层的全精度模型输出为x，量化模型输出为x+Δx，假设后续层的变换可以近似为恒等变换加上一个较小的函数f，则全精度模型最终输出： $x+f(x)$ ，量化模型输出： $x+Δx+f(x+Δx)$ 约等于 $x+Δx+f(x)$ ，因此可以考虑直接比较这两个量过输出头之后的kl loss。其中 $x+f(x)$ 一开始就缓存下来了（全精度模型的激活值）。

可以做一下实验看一下这个loss（包括fisher mse loss）的梯度对真正kl loss的梯度的近似水平好不好，可以通过cosine相关性来测一下。

## loss slide window

- 原本的方法中每个layer都有一个layer wise的loss，但是一旦开始量化下一个layer那么这个loss就会突变导致优化目标瞬移。因此引入了一个slide window loss，具体来说就是在量化layer i的时候同时计算layer i+1的loss，并且在量化layer i的时候逐步调节layer i和layer i+1所占的比重从(1,0)到(0,1)，实现loss的平滑过渡。这在理论上还有一个好处就是使得layer i在量化时不仅关注于降低layer i及以前的量化误差，同时也关注于layer i的量化不要对后面的层产生太大的量化误差积累影响。

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
- 测一下调度器