# GPTQ+ 优化方案设计

# 第一周

## 原方案问题

原方案会遇到梯度更新量爆炸问题，从而不得不通过调整α来间接处理这个困难。α太小就会导致一阶项实际上没有太发挥作用。

梯度的来源是逐层积累的误差，其scale远大于hessian的尺度，因此 $gH^{-1}$ 的尺度非常大，导致一步的步长非常大。当更新较大时，会对hessian和grad产生较大的变动，而在原始算法中，hessian和grad在量化同一层过程中不会更新，这将会导致模型认定一个梯度方向后，沿着这个方向走num_cols步，从而导致更新量爆炸。

因此，想法是每量化一段时间，就要更新hessian和grad。其中hessian的更新比较困难，并且hessian的估计本身就非常不准（尤其是假定group内共享hessian，通过实验发现group内hessian的相关度非常差）因此想法是通过跑一步反向传播来更新g。

## 解决方案

在逐列量化每一个block结束后，通过一次反向传播（使用训练集中BACKWARD_SAMPLES个样本）计算真实梯度。

一开始曾尝试直接用这个量去更新gptq+更新式中的梯度g，但是发现由于 $gH^{-1}$ 是绑定出现的，因此只更新g不更新H反而导致量化结果变差。

所以最后方案是在每一个block结束后，反传得到真实梯度g后直接运行一步梯度下降，并且这个真实梯度不会影响gptq+更新用的 $gH^{-1}$ 本身，也就是 $gH^{-1}$ 在整个线性层更新过程中不做任何更新。优化器就用sgd（adam会过拟合）。

（在block内维持原始的更新方式不变）

## 遇到的问题

- 梯度下降会导致比较严重的过拟合，可能需要想一些正则化手段

- Hessian的group内共享导致的误差应该非常大（但是确实有效防止了过拟合，目前还不太理解为什么偏差这么大的Hessian居然可以这么有效）

- 开启rotate后最后一层grad爆大，比倒数第二层之前大100到1000倍。没开就不会有这个现象，不做梯度下降也不会有这个现象，这两个事情叠加在一起才出问题。

- 开启rotate后down_proj和o_proj的梯度爆大。可以考虑动态调整学习率。（没做实验）

## Ablation

参数：

- N_SAMPLES=512
- SEQ_LEN=1024
- BACKWARD_SAMPLES=32
- ALPHA=0.05
- GROUP_SIZE=4
- ROTATE=false

---

- GRAD_LR=0.0 BLOCKSIZE=256:(baseline)

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.88e-01 | 24.77 | 7.60e-02 | 5.07 | 1.35e-01 | 8.80 |

- GRAD_LR=0.01 BLOCKSIZE=256:

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.90e-01 | 24.76 | 7.56e-02 | 5.07 | 1.37e-01 | 8.93 |

- GRAD_LR=0.05 BLOCKSIZE=256:

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.88e-01 | 24.76 | 7.38e-02 | 5.09 | 1.36e-01 | 8.92 |

- GRAD_LR=0.1 BLOCKSIZE=256:

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.90e-01 | 24.66 | 7.46e-02 | 5.01 | 1.33e-01 | 8.66 |

- GRAD_LR=0.2 BLOCKSIZE=256:

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.82e-01 | 24.39 | 7.33e-02 | 5.02 | 1.33e-01 | 8.71 |

- GRAD_LR=0.3 BLOCKSIZE=256:

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.87e-01 | 24.69 | 7.17e-02 | 5.00 | 1.32e-01 | 8.63 |

- GRAD_LR=0.5 BLOCKSIZE=256:

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.83e-01 | 24.53 | 7.19e-02 | 5.03 | 1.30e-01 | 8.73 |

- GRAD_LR=0.7 BLOCKSIZE=256:

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.90e-01 | 24.87 | 7.24e-02 | 5.03 | 1.31e-01 | 8.67 |

- GRAD_LR=1.0 BLOCKSIZE=256:

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.80e-01 | 24.16 | 7.19e-02 | 4.98 | 1.31e-01 | 8.67 |

- GRAD_LR=2.0 BLOCKSIZE=256:

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 3.05e-01 | 29.32 | 1.37e-01 | 5.47 | 2.28e-01 | 9.78 |

## 加入正则化

### 方案一

l2正则项： $0.5\lambda(W-W_{全精度})^2$

### Ablation

- GRAD_LR=1.0 BLOCKSIZE=256 lambda=0.001:

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.88e-01 | 24.53 | 7.23e-02 | 5.05 | 1.30e-01 | 8.67 |

- GRAD_LR=1.0 BLOCKSIZE=256 lambda=0.005:

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.91e-01 | 24.68 | 7.43e-02 | 5.07 | 1.30e-01 | 8.68 |

- GRAD_LR=1.0 BLOCKSIZE=256 lambda=0.01:

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.77e-01 | 23.77 | 7.28e-02 | 4.99 | 1.32e-01 | 8.66 |

- GRAD_LR=1.0 BLOCKSIZE=256 lambda=0.05:

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.79e-01 | 24.02 | 7.17e-02 | 4.99 | 1.29e-01 | 8.62 |

- GRAD_LR=2.0 BLOCKSIZE=256 lambda=0.05:

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 2.06e-01 | 24.73 | 8.76e-02 | 5.02 | 1.45e-01 | 8.52 |

## 方案二

根据hessian的正则项： $0.5\lambda(W-W_{全精度})^T H (W-W_{全精度})$

### Ablation

- GRAD_LR=1.0 BLOCKSIZE=256 lambda=3.0:

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.93e-01 | 25.00 | 7.42e-02 | 5.04 | 1.31e-01 | 8.65 |

- GRAD_LR=1.0 BLOCKSIZE=256 lambda=10.0:

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.88e-01 | 24.40 | 7.25e-02 | 5.02 | 1.36e-01 | 8.71 |

- GRAD_LR=1.0 BLOCKSIZE=256 lambda=30.0:

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.94e-01 | 24.54 | 7.79e-02 | 5.15 | 1.46e-01 | 8.82 |

## 方案三

根据目前权重与最近量化格点的距离动态调整学习率：

定义归一化距离d：

$$
d = \frac{|w-q_{nearest}|}{quant\_scale}
$$

学习率门控系数g：

$$
g(d)=f + (1-f)(1-e^{-k d})
$$

其中k和f为超参。

### Ablation

- GRAD_LR=1.0 BLOCKSIZE=256 f=0.0 k=8.0:



- GRAD_LR=1.0 BLOCKSIZE=256 f=0.1 k=8.0:

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.90e-01 | 24.76 | 7.21e-02 | 5.01 | 1.31e-01 | 8.74 |

## Adam优化器

- GRAD_LR=0.0001 BLOCKSIZE=256 l2_reg=10.0:

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.88e-01 | 24.75 | 7.33e-02 | 5.03 | 1.31e-01 | 8.70 |

拟合效果较好，但过拟合

# 第二周

## 重新跑一下l2正则的实验

- GRAD_LR=0.3 BLOCKSIZE=256 lambda=0.01:

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.86e-01 | 24.77 | 7.33e-02 | 5.06 | 1.33e-01 | 8.70 |

- GRAD_LR=0.3 BLOCKSIZE=256 lambda=0.03:

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.87e-01 | 24.65 | 7.16e-02 | 5.00 | 1.31e-01 | 8.65 |

- GRAD_LR=0.3 BLOCKSIZE=256 lambda=0.05:

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.81e-01 | 24.26 | 7.20e-02 | 5.05 | 1.33e-01 | 8.76 |

- GRAD_LR=0.3 BLOCKSIZE=256 lambda=0.07:

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.85e-01 | 24.71 | 7.32e-02 | 5.06 | 1.33e-01 | 8.79 |

- GRAD_LR=0.3 BLOCKSIZE=256 lambda=0.1:

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.90e-01 | 24.77 | 7.22e-02 | 4.97 | 1.31e-01 | 8.60 |

## 测一下quant error gate搭配block原子化更新

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.91e-01 | 24.85 | 7.29e-02 | 5.03 | 1.33e-01 | 8.83 |

## 最后一层更重要，调整为跑全部样本：

- GRAD_LR=0.3 BLOCKSIZE=256:

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.87e-01 | 24.57 | 7.45e-02 | 5.06 | 1.32e-01 | 8.75 |

## loss优化

再做一个优化，由于对整个模型除了最后一层之外的kl loss都是外接一个不准的输出头去测的，所以肯定不准。但是这个kl loss可以更容易估计hessian。因此对于梯度下降过程，可以换用更精确的loss。

- 假设输出的kl loss为这一层输出的变化的二阶项： $loss = \frac{1}{2} \Delta y ^T H \Delta y$ ，这里的H可以用fisher矩阵估算对角项，也就是逐元素的梯度平方的均值，这个可以在预处理的时候得到。

直接这样取loss数值会爆炸，因为不同线性层的loss尺度不一样且hessian数值波动。改成加权系数归一化。后面可以试试给H加一个λI稳定数值。

- GRAD_LR=0.00007 adam

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.77e-01 | 24.32 | 6.82e-02 | 5.09 | 1.29e-01 | 8.88 |

- GRAD_LR=0.0001 adam

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.76e-01 | 24.42 | 6.74e-02 | 5.05 | 1.28e-01 | 8.90 |

- GRAD_LR=0.0003 adam (best)

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.70e-01 | 23.94 | 6.75e-02 | 5.04 | 1.24e-01 | 8.83 |

- GRAD_LR=0.0005 adam

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.76e-01 | 24.36 | 7.07e-02 | 5.07 | 1.27e-01 | 8.86 |

- GRAD_LR=0.0007 adam

| KL-wikitext2 | PPL-wikitext2 | KL-ultrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath |
| --- | --- | --- | --- | --- | --- |
| 1.83e-01 | 24.80 | 7.55e-02 | 5.10 | 1.33e-01 | 8.94 |

## quant边界优化：

（暂时没动groupsize！=-1的情况）

在quant边界确定后先裁剪再算sg H g等，blockwise更新时顺序：先更新block外的gptq+二阶项，再裁剪，再反传梯度下降

- GRAD_LR=0.0003 adam



## 正弦周期正则化