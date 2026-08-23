# REAL-Q Qwen3 refresh 融合与 Fisher TF32 实现记录（2026-08-08）

## 1. 任务范围

本轮按 run13 nsys 热点实现两个最高性价比的模型算子融合：

1. Qwen3 Q/K head-wise RMSNorm 与 RoPE 的联合 forward/backward；
2. SwiGLU `silu(gate) * up` 的联合 forward/backward。

用户随后明确允许 Fisher loss 的乘法采用 TF32 输入精度，但要求 FP32
累积，因此同时把 Fisher 二次型改为局部 TF32-input/FP32-accumulate 路径。

本轮不修改 Transformers 安装包，不改变模型参数名、checkpoint key、activation
quant、Fast Hadamard、FA4 或 GPTQ column-inner 数学。

## 2. nsys 归因修正

此前按 CUDA kernel 名把所有 `MulFunctor` 归到模型 elementwise，得到的
`5.377 s` 不是纯 norm/RoPE/SwiGLU：Fisher 二次型 backward 展开的 FP32
逐元素乘法也使用相同 kernel 名。run13 中可精确归因的两次 forward 合计为：

| 流程 | 当前层 forward | 下一层 forward | 每 refresh 合计 |
|---|---:|---:|---:|
| RMSNorm | 5.17 ms | 5.11 ms | 10.28 ms |
| RoPE（含 rotate_half/cat） | 4.83 ms | 4.74 ms | 9.57 ms |
| SwiGLU | 1.10 ms | 1.09 ms | 2.19 ms |
| residual | 0.29 ms | 0.29 ms | 0.58 ms |

因此两个 fusion 的首要目标是减少完整 activation 中间张量和 HBM traversal，
不是把 `5.377 s` 当成可以全部消除的上界。

## 3. 实现设计

### 3.1 Q/K RMSNorm + RoPE

新增 `utils/triton_qwen3_fusions.py`。Qwen3 原图为：

```text
q_proj/k_proj
  -> head-wise RMSNorm (FP32 reduction/normalization)
  -> cast deployed dtype + norm weight
  -> rotate_half: neg + cat
  -> x*cos + rotate_half(x)*sin
```

接入利用已有 `QKRotationWrapper`：将 Q/K Norm instance forward 延迟成 identity，
wrapper 在原 RoPE site 获取 raw Q/K、norm weight/epsilon 与 cos/sin。CUDA 合法
布局走 fused custom autograd，CPU/不支持布局调用保存的原 RMSNorm forward 和
原 RoPE 函数。

为避免 wrapper 重复注册 Q/K Norm 并产生新的 state_dict key，wrapper 只保存
weakref。Block-GD 从不优化 Norm 参数；安装时显式冻结 Q/K Norm weight，custom
backward 因而只计算 Q/K activation gradient，不执行 dweight reduction。

Forward 每个 Q/K 各一个 kernel：warp reduction 得到 FP32 inverse RMS，把
normalization、affine、half-index 交换、RoPE mul/add 保持在寄存器中。Backward
直接应用 RoPE transpose，然后在同一 kernel 中完成 RMSNorm Jacobian-vector
product。Qwen3 的一行只有 128 个元素，因此一个 CTA 同时处理 16 个相互独立的
head/token row，把 production shape 的约 209 万个单行 CTA 降到约 13 万个，且
不引入跨行同步或额外 global-memory pass。

数值口径保留 eager 的关键舍入点：FP32 normalization 后先 cast BF16/FP16，
再执行 norm weight；RoPE 两个乘积和最终加法也显式 cast deployed dtype。

### 3.2 SwiGLU

只 patch dense `Qwen3MLP.forward` 的表达式，`gate_proj/up_proj/down_proj` module
对象及其 activation quant wrapper 不变：

```text
forward:  silu(gate) * up                     -> one kernel
backward: dgate, dup                          -> one kernel/two outputs
```

CPU/Triton 不可用时执行原 `torch.nn.functional.silu(gate) * up`。CUDA kernel
在 SiLU 后显式 cast deployed dtype，再乘 up，保留原 BF16/FP16 中间舍入点。

### 3.3 Fisher TF32

TF32 不是 tensor dtype；`torch.backends.cuda.matmul.allow_tf32` 只影响 cuBLAS
矩阵乘，不能把普通 elementwise `*` 自动变成 TF32。实现采用两部分：

1. `delta @ fisher` 和 backward 的 `delta @ fisher.T` 在局部 TF32 policy 下
   dispatch，输入/输出 storage 为 FP32，Tensor Core accumulator 为 FP32；
2. 新增 `realq/refresh/triton_fisher.py`，row-wise inner product 用 PTX
   `cvt.rna.tf32.f32` 显式把两个 FP32-storage operand 舍入到 TF32 mantissa，
   乘法和 reduction accumulator 使用 FP32。

Fisher custom backward 直接使用解析梯度：保存 forward 的 `delta @ F`，backward
计算 `delta @ F.T`，一个 fused kernel 完成两个分支的 TF32-input scale 和 FP32
相加。这样不再让 PyTorch autograd 展开原先的大量 FP32 `MulFunctor/AddFunctor`。
全局 `allow_tf32` 在每次 GEMM dispatch 后恢复，其他 FP32 matmul 不受影响。

用户要求“所有 Fisher multiplication 都可使用 TF32、只保留 FP32 累积”后，
gradient kernel 也显式舍入 `left/right/grad_output/scale factor`；每次乘法的输入
均为 TF32 mantissa，两个梯度分支的相加和输出仍为 FP32。forward row inner 的
两个输入同样显式舍入，`tl.sum` 和后续 mean 都是 FP32 reduction。`0.5` 只是精确
的二进制指数缩放，不损失 accumulator 的尾数。

旧 `tools/profile_overrides/fisher_loss_tf32/sitecustomize.py` 已降级成兼容 no-op。
它不再包裹全局 `torch.autograd.grad`，避免把 refresh 内与 Fisher 无关的 FP32
matmul 意外切到 TF32。旧 launcher 即使仍设置该环境变量，也会走生产代码中的
局部精度策略。

CPU 继续保留旧 FP32 方程，作为 paper-numerics 精确 oracle。

## 4. 硬件选择

- Q/K RMSNorm、RoPE、SwiGLU 是 reduction/streaming elementwise，不适合
  WGMMA/TCGen05；主要优化是寄存器复用、warp reduction 和合并 HBM pass。
- RoPE cos/sin 在 head 间有复用，但 head_dim=128；首版使用 coalesced load，
  不为一次消费强制增加 TMA descriptor/shared-memory barrier。
- Fisher 两个大矩阵乘继续由 cuBLAS Tensor Core TF32 执行；最终 row inner
  没有合适的 M/N tile，使用 fused streaming reduction，避免构造完整 product。

## 5. 变更文件

- `utils/triton_qwen3_fusions.py`：Q/K Norm+RoPE 与 SwiGLU kernels/custom autograd；
- `utils/rotation_utils.py`：deferred Q/K Norm 接线和 exact fallback；
- `gptq_utils/quant_aware_utils.py`：给 QK wrapper 传入 Q/K Norm；
- `realq/akv.py`：幂等安装 dense Qwen3 SwiGLU patch；
- `realq/refresh/triton_fisher.py`：TF32 row inner/gradient fusion；
- `realq/refresh/fisher_loss.py`：局部 TF32 GEMM 与解析 custom backward；
- `tests/test_qwen3_fused_ops.py`、`tests/test_fisher_loss_tf32.py`：CPU/CUDA
  forward、gradient、fallback、state_dict、policy restoration 测试。

## 6. 验证记录

### 6.1 重启前 CPU 回归

- `tests/test_qwen3_fused_ops.py -k 'not cuda'`：2 passed，2 CUDA skipped；
- Fisher loss 与既有 AKV/paper numerics/Fisher cache 回归：53 passed，1 CUDA
  skipped；
- 合计 55 passed，3 个 CUDA case 因开发机无 GPU 跳过。

CPU 测试覆盖 wrapper fallback、deferred Norm、state_dict 不新增 alias、SwiGLU
patch 幂等、原 Fisher FP32 oracle 的 value/gradient bit-exact。CUDA tests 已写好，
覆盖两个 fusion 和 Fisher 的 forward/backward、TF32 全局 policy 恢复与误差。

### 6.2 2026-08-08 重启后静态与 Blackwell AOT 检查

- 10 个本轮 Python 文件 AST/py_compile 通过；
- `git diff --check` 通过；
- 使用环境里的 Triton 3.5.1，不连接 GPU driver，目标设为
  `GPUTarget("cuda", 100, 32)`，对 production shape 常量离线编译：
  - Q/K RMSNorm+RoPE forward：通过，PTX 45,297 chars；
  - Q/K RMSNorm+RoPE backward：通过，PTX 55,585 chars；
  - SwiGLU forward：通过，PTX 11,613 chars；
  - SwiGLU backward：通过，PTX 14,557 chars；
  - Fisher row inner：通过，PTX 20,357 chars；
  - Fisher analytic gradient：通过；最终版本 PTX 13,507 chars。
- Fisher 两个 kernel 的 PTX 均包含 `cvt.rna.tf32.f32`；最终 gradient PTX 中
  检出 11 个静态 TF32 conversion site，说明 left/right/scalar 链的显式舍入
  没有在 lowering 时丢失。

AOT 结果证明 Triton AST、SM100 lowering 和 PTX assembly 可以生成，但不能证明
真卡上的数值、寄存器占用/occupancy 或端到端速度。服务器重启后当前开发容器
导入 Torch 会卡住，且现有工具没有进入旧 Canoe debug pod 执行命令的通道，因此
本轮尚未伪造或外推 GPU 数值/性能结论。

### 6.3 真卡待执行项

`tools/benchmark_qwen3_refresh_fusions.py` 已同时覆盖 quick 与 production
`B=32,S=2048`，会输出三个路径的 forward 和 forward+backward latency、speedup、
output relative-L2 及 gradient relative-L2。待 B100 可执行后，应依次运行：

```bash
python tools/benchmark_qwen3_refresh_fusions.py
python tools/benchmark_qwen3_refresh_fusions.py --production
```

通过 microbenchmark 数值门槛后，再运行原 layer 5--8 nsys capture；最终是否保留
`BLOCK_ROWS=16` 应由 B100 的 register/occupancy 与实测 latency 决定，而不是只凭
离线 PTX 大小判断。

### 6.4 `j-beiyvye88v` SM100 真卡实测

2026-08-08 进入指定 debug job 的 `master-0` 实测。容器报告：

- device：`NVIDIA L20C`，compute capability `(10, 0)`；
- Torch `2.9.1+cu128`，Triton `3.5.1`；
- 使用 job 原有 `.venv.py312-broken-20260729`，没有安装或修改系统依赖；
- dead-node 快检通过，节点 `up{}` 持续为 1，无关键 pod event。

CUDA 单测结果：`tests/test_qwen3_fused_ops.py` 和
`tests/test_fisher_loss_tf32.py` 共 **6 passed in 16.77 s**。

#### Production microbenchmark

命令：

```bash
.venv.py312-broken-20260729/bin/python \
  tools/benchmark_qwen3_refresh_fusions.py \
  --production --warmup 10 --repetitions 50
```

shape 为正常 refresh 的 `backward_bsz=32, seq_len=2048`；Q/K heads 为
`32/8`、head dim `128`、SwiGLU intermediate `9728`、Fisher hidden `2560`。
时间是 CUDA event 的 50 次稳态均值，forward 测量保留 autograd graph 构建，因为这正是
refresh forward 的实际执行方式。

| 路径 | eager forward | fused forward | forward 加速 | eager fwd+bwd | fused fwd+bwd | fwd+bwd 加速 |
|---|---:|---:|---:|---:|---:|---:|
| Q/K RMSNorm+RoPE | 8.658 ms | 0.361 ms | **24.01x** | 20.719 ms | 0.723 ms | **28.67x** |
| SwiGLU | 1.063 ms | 0.635 ms | **1.67x** | 2.819 ms | 1.557 ms | **1.81x** |
| Fisher quadratic | 1.979 ms | 1.795 ms | **1.10x** | 4.328 ms | 3.368 ms | **1.29x** |

Production 数值：

- Q/K output relative-L2：Q `2.145e-3`，K `2.177e-3`；
- Q/K input-gradient relative-L2：dQ `2.593e-3`，dK `2.601e-3`；
- SwiGLU output bit-exact；dUp bit-exact，dGate relative-L2 `1.793e-6`；
- 随机对称 Fisher case：loss relative delta `9.628e-6`，经 BF16
  `q_out` 链路后的 gradient relative-L2 `1.433e-3`。

Q/K 测试使用独立高斯 cos/sin，幅度比真实 RoPE 的 `[-1, 1]` 更激进；误差仍低于
BF16 unit roundoff 的量级。SwiGLU forward 达到 bit-exact。

#### Fisher loss 三精度口径对照

新增 `tools/validate_fisher_tf32_precision.py`，在完整
`B=32,S=2048,H=2560` 上构造确定性的 PSD Fisher，并用三档 BF16 delta RMS
比较：

1. 所有乘法 FP32；
2. 仅两次 GEMM 使用 TF32，elementwise inner/gradient 保持 FP32；
3. 新实现：所有 Fisher 乘法使用 TF32 输入，所有 reduction/addition 使用 FP32。

| delta RMS | FP32 loss | all-TF32 loss | loss rel. vs FP32 | grad rel-L2 vs FP32 | all vs GEMM-only grad rel-L2 |
|---:|---:|---:|---:|---:|---:|
| 0.999953 | 1279.879639 | 1279.867432 | `9.538e-6` | `3.007e-4` | `2.070e-4` |
| 0.100014 | 12.802535 | 12.802428 | `8.343e-6` | `3.012e-4` | `2.078e-4` |
| 0.010136 | 0.1315203 | 0.1315196 | `5.665e-6` | `3.008e-4` | `2.074e-4` |

三组中：

- all-TF32 相对全 FP32 的最大 loss relative error 为 **`9.538e-6`**；
- FP32 `delta` 解析梯度的最大 relative-L2 为 **`3.012e-4`**；
- all-TF32 相比 GEMM-only TF32，loss 最大额外 relative difference 仅
  **`7.449e-8`**，两组因最终 FP32 scalar 舍入而逐 bit 相等；
- public `fisher_mse_loss` 与直接 fused core 在三组中均逐 bit 相等；
- 测试前后的全局 `torch.backends.cuda.matmul.allow_tf32` 状态一致。

结论：把 Fisher 剩余乘法也切到 TF32 后，新增的数值扰动主要体现在梯度的
`~2.1e-4` relative-L2；总计相对全 FP32 为 `~3.0e-4`。loss 标量误差稳定在
`1e-5` 以下，没有随 delta 从 1 缩小到 0.01 而恶化。性能上 50 次稳态
production forward+backward 提升 1.285x；该实现可以保留。

### 6.5 正常 W4A4 配置端到端 profile

在同一台 `j-beiyvye88v`、同一张 SM100 GPU 上重新运行 layer 5 profile。
run13 与 run14 的公共配置为：W4A4、所有 weight/activation/K/V group size 均为
128、WikiText2 256 samples、seq len 2048、前向 batch size 128、
`backward_samples=backward_bsz=32`、FA4、A-loss clip 关闭、Hessian TF32，且都只
采 layer 5、完成 layer 6 后停止。layer 0--4 负责预热，所以 layer 5 不包含 Triton
首次编译开销。

- 直接基线：`20260807_run13_allopts_layer5`，包含此前的 fused fake quant、compact
  scale、Fast Hadamard、column-inner 和 Hessian TF32 优化；
- 新结果：`20260808_run14_fusions_fisher_tf32_layer5`，在 run13 上增加本文件的
  Q/K RMSNorm+RoPE、SwiGLU 融合，以及 Fisher 全乘法 TF32；
- 两份结果各有 201 次 `block.refresh`，比较口径完全一致。

| 指标 | run13 | run14 | 节省 | 加速 |
|---|---:|---:|---:|---:|
| layer 5 完整量化 | 42.810 s | **35.614 s** | 7.196 s / 16.81% | **1.202x** |
| 201 次 refresh 合计 | 35.404 s | **28.418 s** | 6.986 s / 19.73% | **1.246x** |
| 单次 refresh 平均 | 176.139 ms | **141.384 ms** | 34.755 ms / 19.73% | **1.246x** |
| layer 非 refresh 部分 | 7.406 s | 7.196 s | 0.210 s / 2.84% | 1.029x |

本轮整层节省的 97.08% 来自 refresh。端到端整层加速低于算子 microbenchmark，
原因是 Q/K 和 SwiGLU 只覆盖 refresh 的两次 forward/联合 backward；Hessian、Hinv、
final replay、materialize 和 teardown 等约 7.2 s 不在本轮优化范围内。

按被量化模块拆分，单次 refresh 平均耗时如下：

| module | run13 | run14 | 降幅 | 加速 |
|---|---:|---:|---:|---:|
| `self_attn.q_proj` | 201.491 ms | 157.654 ms | 21.76% | 1.278x |
| `self_attn.k_proj` | 190.553 ms | 157.244 ms | 17.48% | 1.212x |
| `self_attn.v_proj` | 186.781 ms | 152.838 ms | 18.17% | 1.222x |
| `self_attn.o_proj` | 181.868 ms | 148.215 ms | 18.50% | 1.227x |
| `mlp.up_proj` | 176.683 ms | 142.198 ms | 19.52% | 1.243x |
| `mlp.gate_proj` | 172.594 ms | 139.060 ms | 19.43% | 1.241x |
| `mlp.down_proj` | 161.762 ms | 127.901 ms | 20.93% | 1.265x |

整段 capture 的 CUDA kernel launch 从 374,042 降到 328,445，减少 45,597 次
（12.19%）；所有 kernel duration 之和从 38.904 s 降至 31.713 s（减少 18.48%），
memcpy duration 从 0.487 s 降至 0.368 s（减少 24.41%）。trace 中实际出现：

- `_qk_rmsnorm_rope_forward_kernel` 2,370 次、170.978 ms；
- `_qk_rmsnorm_rope_backward_kernel` 476 次、87.412 ms；
- `_swiglu_forward_kernel` 1,185 次、313.491 ms；
- `_swiglu_backward_kernel` 326 次、306.910 ms；
- `_tf32_row_inner_kernel` 401 次、76.336 ms；
- `_tf32_quadratic_grad_kernel` 401 次、114.679 ms。

这说明端到端加速来自生产路径真实命中，而不是独立 benchmark 的外推。

同一对 profile 的 7 层量化循环（含 layer 0 首次编译、layer 5 nsys capture）从
5:32 降到 4:34，实测 **1.212x / 17.47%**。若只按稳态 layer 5 线性外推 36 层，
量化主体约从 25.69 min 降到 21.37 min，节省约 4.32 min；该数值是外推，不是
完整 36 层实跑。

作为累计收益参考，较 FA4-only 的 run8，layer 5 从 89.327 s 降到 35.614 s，
累计 **2.508x / 60.13%**；较最初修正为 `backward_samples=backward_bsz=32` 的
run6，layer 5 从 126.625 s 降到 35.614 s，累计 **3.556x / 71.87%**。后一个口径
还包含关闭 A-loss clip、接入 FA4、Hessian TF32 等此前多轮优化，不能归因于本轮
三个新 kernel。

端到端 loss 的第一处可比点（layer 0、q_proj、block 0）从
`0.0329070389` 变为 `0.0329321176`，相对差 `7.62e-4`。后续 block 会因优化器轨迹
和前层量化误差累积而逐渐分叉，不能再用逐 block loss 直接代表单算子误差；隔离
测试中 Fisher loss 误差仍小于 `1e-5`，Q/K output/gradient relative-L2 分别约为
`2.2e-3/2.6e-3`，SwiGLU forward bit-exact。

### 6.6 run14 剩余瓶颈

run14 中 201 次 refresh 仍占完整 layer 5 的 `28.418/35.614=79.79%`。按 kernel
start timestamp 归入互不重叠的 `block.refresh` NVTX 区间后，refresh 内 CUDA
kernel duration 合计 27.642 s，是 refresh wall 的 97.27%。因此剩余问题不是
CPU 空等；即使用 CUDA Graph 完全消除 host launch gap，独立上限也只有约 2.7%。
必须减少 GPU 上实际执行的数据搬运、临时张量和计算。

| refresh 内 kernel 类别 | GPU 时间 | refresh wall 占比 | 判断 |
|---|---:|---:|---|
| GEMM/BMM | 10.598 s | 37.29% | 最大绝对项；主要是模型 fwd/bwd BF16 GEMM，另有约 0.741 s Fisher TF32 GEMM；Blackwell tuned Tensor Core 路径，继续手写 GEMM 的收益有限 |
| copy/cat/index | 3.249 s | 11.43% | direct copy/cast 约 2.43 s，indexing backward 约 0.93 s；cat 已只剩约 0.9 ms |
| Fast Hadamard | 2.759 s | 9.71% | 已使用专用扩展；下一步需要跨算子保留 rotated layout 或与相邻 GEMM/quant 融合，单独替换 kernel 空间较小 |
| 其他 elementwise | 2.536 s | 8.92% | BF16/FP32 mul/add、residual、Adam/反传链；仍有大量独立 pass |
| fill/masked-fill | 2.263 s | 7.96% | 约 70k launches；主要是 BF16/FP32 临时量与梯度/累积 buffer 清零 |
| dynamic min/max | 1.891 s | 6.65% | min 和 max 仍是两个 reduction pass；可合为一次成对 reduction |
| FA4 attention | 1.532 s | 5.39% | 已命中 SM100 FA4，继续换 attention backend 不是最高优先级 |
| fused fake quant | 0.982 s | 3.45% | arithmetic 已融合；主要剩第二遍读写，受前置动态 qparam reduction 约束 |
| fused SwiGLU | 0.580 s | 2.04% | 已不是主要项 |
| fused Q/K Norm+RoPE | 0.235 s | 0.83% | 已不是主要项 |
| Fisher fused elementwise | 0.191 s | 0.67% | 已不是主要项 |

最可操作的下一目标是 copy/fill/elementwise，三者合计 8.047 s、占 refresh wall
28.31%。当前正式配置 `backward_samples=backward_bsz=32`，每次 refresh 只有一个
backward chunk，但 full-block 路径仍预分配 FP32 `partial_grad_sums`、执行
`zeros_like/add_/copy/index_select/index_copy`，并为 selected-column Adam 构造 full
zero update。增加 single-chunk/single-rank direct-consume 路径、让 Adam 直接更新
连续 active suffix、复用 accumulation buffer，预期比继续优化已经很小的 Q/K 或
Fisher fused kernel 更有价值；实现时必须保持 FP32 master/moment 和 act-order
坐标语义。

动态量化是第二个相对独立的目标：把 min/max 合成一个 Triton reduction 可少一次
activation 全读；随后仍需第二 pass fake quant，因此不能把理论收益写成 2.873 s
全部消失。Fast Hadamard 若不改变跨算子 layout，standalone kernel 已较成熟。

非 refresh 部分合计 7.196 s（20.21%）：Hessian accumulate 2.200 s（6.18%）、
`quant.hinv` 1.072 s（3.01%）、final/FP/next-FP replay 合计 2.496 s（7.01%）、
teardown 1.023 s（2.87%）。Hessian 的 TF32 matmul 本体约 0.724 s，剩余主要是
layer forward、weighted-input 构造、cast/copy 和 accumulation；它已不再是第一
瓶颈。Hinv 和 replay 还有空间，但即使完全删除任一单项，对整层也只有约 3%--7%
的上限。

### 6.7 copy / cast / index 一次性优化与真卡结果

本轮以 run14 为基线，集中处理 6.6 节中 full-block refresh 的
copy/cast/index 临时量。最终保留的实现由三部分组成。

#### 实现

1. **单卡单 chunk 直接消费 autograd gradient**：正常配置是单卡且
   `backward_samples=backward_bsz=32`，loss 已经返回 32 个样本的 batch mean。
   此时不再为每个 active weight 分配完整 FP32 `partial_grad_sum`，也不再执行
   `BF16 grad -> FP32 -> *32 -> /32` 的等价链；原始 autograd gradient 直接交给
   fused Adam。多卡、多 chunk 和 trace audit 路径仍走原 FP32 accumulation oracle。
2. **current/future weight 的一遍 fused Adam**：新增
   `realq/refresh/triton_block_adam.py`。kernel 在寄存器中完成 gradient 转 FP32、
   scale、clip、一二阶 moment、bias correction、sqrt/div 和 update。future weight
   直接原地更新 FP32 master；current weight 只输出 GPTQ quant-order 的 active
   suffix，不再物化 full-size zero update，也不再做
   `index_select/index_copy/indexing_backward`。FP32 master、`exp_avg` 和
   `exp_avg_sq` 的精度没有降低。
3. **act-order 一遍 stitch**：新增 `realq/quant/triton_refresh_stitch.py`，根据
   `invperm` 直接从 permuted Q/W 生成 natural-order `[Q prefix, W suffix]`，替代
   `Q[:, invperm] + W[:, invperm] + clone + indexed scatter` 四段临时量。refresh
   callback 增加 `trailing_quant_order_full_block` 布局，`RealQLayer` 直接消费 compact
   suffix；unmarked legacy callback 的 natural-order/full-update 兼容分支也有独立测试。

此外，FP32 master 转 module dtype 时，`.to()` 已经产生独立存储便不再无条件
`.clone()`；配置、launcher、trace 和 checkpoint provenance 均记录
`fused_block_adam`。该开关默认开启，历史 checkpoint 缺字段时恢复为 `false`，避免
静默改变旧 artifact 的 build provenance。

这两个 Triton kernel 没有强行使用 TMA/WGMMA。act-order stitch 是每个输出只读
一个随机列元素的一遍 gather，Adam 是低算术强度的 streaming elementwise/update；
二者都没有可复用的矩阵 tile，也没有矩阵乘累加。future Adam 的自然列访问已完全
coalesced，current Adam 的随机列语义由 act-order 决定。TMA staging 或 WGMMA 在这里
只会增加同步与搬运，不会提高有效带宽；SM100 Tensor Core 仍由真正的 GEMM 路径使用。

#### 隔离性能与精度

在 `j-beiyvye88v` 的 SM100 GPU 上，以 `2560 x 2560`、trailing start 128、
10 warmup + 50 repetitions 测得：

| 路径 | 旧实现 | 新实现 | 加速 | 精度 |
|---|---:|---:|---:|---|
| act-order stitch | 0.08899 ms | 0.02299 ms | **3.871x** | FP32 bit-exact |
| current Adam + compact suffix | 0.34555 ms | 0.19494 ms | **1.773x** | update relative-L2 `1.110e-7`，max abs `2.728e-12` |
| future Adam + master update | 0.15841 ms | 0.02554 ms | **6.202x** | FP32 oracle 范围内一致 |
| master-to-leaf（删除冗余 clone） | 0.01245 ms | 0.00658 ms | **1.892x** | BF16 bit-exact |

current Adam 的 `exp_avg` relative-L2 为 0，`exp_avg_sq` relative-L2 为
`4.139e-8`。GPU 单测还覆盖三个 trailing boundary 的 stitch 逐 bit 对照、current 和
future Adam、act-order/no-act-order compact callback；CPU full update 与 compact
update/master 对照为逐 bit 相等。最终相关回归集合为 **190 passed in 99.89 s**。

#### 正常 W4A4 profile：run14 -> run15

run15 tag 为 `20260808_run15_copy_cast_index_layer5`。配置与 run14 完全相同：
W/A/K/V 4 bit、所有 group size 128、WikiText2 256 samples、seq len 2048、forward
bsz 128、`backward_samples=backward_bsz=32`、FA4、A-loss clip 关闭、Fisher/Hessian
TF32，采 layer 5，完成 layer 6 后停止。

| 稳定主体指标 | run14 | run15 | 降幅 | 加速 |
|---|---:|---:|---:|---:|
| 7 个 `module.quantize` 合计 | 29.8600 s | **27.8393 s** | **6.77%** | **1.073x** |
| 201 次 `block.refresh` | 28.4181 s | **26.5544 s** | **6.56%** | **1.070x** |
| 单次 refresh 平均 | 141.384 ms | **132.111 ms** | **6.56%** | **1.070x** |
| refresh 内 CUDA kernel（不含 memcpy/memset） | 27.6247 s | **26.0025 s** | **5.87%** | **1.062x** |
| refresh 内 kernel launch | 197,349 | **134,649** | **31.77%** | -- |

每个模块都变快：q/k/v/o/up/gate/down 的 `module.quantize` 分别从
`3.092/3.066/2.988/4.761/2.807/2.751/10.395 s` 降到
`2.904/2.935/2.849/4.452/2.625/2.549/9.525 s`，说明收益并非来自某一个模块的
偶然波动。

copy/index 类 GPU 时间从 3.2491 s 降到 2.2112 s，减少 **1.0379 s / 31.94%**：

- `indexing_backward`：0.9252 s / 804 launches -> 0；
- scatter：0.0226 s / 603 -> 0；
- cat：0.0230 s / 804 -> 0；
- direct copy：2.2332 s / 11,621 -> 2.1435 s / 9,471；
- gather：0.0676 s / 602 基本不变，它主要是每个 refresh 必需的样本/目标选择；
- FP32 fill：0.6501 s / 41,845 -> 0.5439 s / 30,603；BF16 fill launch 也减少
  2,036 次；
- 新 fused Adam 本身为 0.2404 s / 2,036 launches。它用一次 streaming pass
  替换的旧 index/cast/fill/elementwise 链远大于自身成本。

run15 的 raw `layer_5` 是 39.414 s，不能直接与 run14 的 35.614 s 比较：
`layer.teardown` 中有一次孤立 `cuMemUnmap` 花费 5.068 s，使 teardown 从 1.023 s
跳到 5.992 s。去掉这一明确的 driver/allocator outlier 后为约 34.346 s，即整层
约 **1.037x / 3.56%**；这是去异常值口径，不冒充无条件 raw wall speedup。
稳定且直接受本轮影响的端到端口径应采用上表 `module.quantize` 的 **6.77%** 和
`block.refresh` 的 **6.56%**。在 nsys 下，一层量化约 34.3 s，其中量化模块主体
约 27.84 s。

#### 数值正确性

run14、run15 的第一个端到端可比 loss（layer 0/q_proj/block 0）均为
`0.0329321175814`，逐 bit 相同。block 1/2 的相对差分别为 `1.066e-4` 和
`9.67e-5`；之后 low-bit 阈值与 Adam 轨迹会放大微小算子舍入，逐 block loss 不再是
单算子误差指标。因此最终接受标准是：首点相同、隔离 stitch bit-exact、CPU contract
bit-exact、GPU Adam 的 update relative-L2 `1.11e-7`，而不是要求整条离散量化轨迹
逐 bit 不分叉。

#### 被否决的 persistent BF16 mirror

为继续消除 master-to-leaf cast，还实作并 profile 了 run16：future weight 常驻一份
BF16 mirror，由 fused Adam 在写 FP32 master 的同一 pass 顺带写 mirror。它把
`bfloat16_copy` 从 5,406 次降到 3,578 次，但 GPU 时间只从 0.4640 s 降到
0.4378 s（省 26.3 ms）；Adam 因额外 BF16 store 从 0.2404 s 增到 0.2525 s。
最终 refresh wall 仅从 26.5544 s 到 26.5172 s，只有 **0.14%**，而
`module.quantize` 反而从 27.8393 s 到 28.0988 s。run16 teardown 又出现累计
18.384 s 的 `cuMemUnmap` 长尾；不能仅凭一次 profile 断言它由 mirror 导致，但该方案
既没有稳定主体收益，又增加常驻显存和 allocator 压力，因此已从最终代码回退。

最终保留版本对应 run15 的算法路径，并额外包含不影响正式 full-block 配置的 legacy
callback 布局兼容修正。剩下的 copy/cast 主要位于模型 forward/backward 内：按 CUDA
runtime correlation 归属，direct copy 为 backward 0.686 s、current forward
0.409 s、next-layer forward 0.401 s、loss 0.184 s；BF16 copy 分别约
0.199/0.118/0.117 s。这些是 FA4、autograd、模型算子边界的布局/精度转换，已不再是
独立 Block-Adam/index 临时量。进一步删除需要改变模型级 layout 或继续融合
forward/backward，而不是再在 refresh bookkeeping 中做局部 copy 微调。
