# REAL-Q Llama 3.1 / Qwen 3 二十组量化与推理评测记录（2026-08-08）

## 1. 任务范围

本 campaign 使用 `tools/realq_auto_tune.py` 对 5 个 post-trained/Instruct 模型、4 种量化设定共 20 组实验独立搜索学习率，随后以各组最佳 LR 做正式单卡 B100 量化并保存 checkpoint，再运行三项推理评测。本文是唯一实时计划与结果表；配置、机器、异常、重试和最终门禁均在此追踪。

目标模型：

1. Qwen3-0.6B（不带 Base 后缀的 post-trained/Instruct 版本）
2. Llama-3.1-8B-Instruct
3. Qwen3-4B（不带 Base 后缀的 post-trained/Instruct 版本）
4. Qwen3-8B（不带 Base 后缀的 post-trained/Instruct 版本）
5. Qwen3-32B（不带 Base 后缀的 post-trained/Instruct 版本）

目标设定：W4A16、W4A4KV4、W3A16、W2A16。W4A4KV4 同时采用 K4/V4；其余设定采用 K16/V16。用户已确认全部位宽、aware 和评测协议，提交任务时不依赖默认值推断。

## 2. 已冻结配置

| 配置项 | 拟冻结值 | 当前状态 | 证据/备注 |
|---|---:|---|---|
| 校准集 | 256 条，seq_len=2048 | 已指定 | 用户要求 |
| tuning / 正式 weight group size | 128 | **已冻结** | 用户确认本 campaign 调参与正式量化均为 128；tuner 增加显式 campaign override |
| W/A/K/V quantizer | symmetric | **已冻结并显式渲染** | `w/a/k/v_asym=false` |
| rotation | true | 已指定 | 用户要求 |
| act order | true | **已冻结并显式渲染** | `act_order=true` |
| weight clipping | true | **已冻结并显式渲染** | `w_clip=true` |
| A/K/V aware | 仅 W4A4KV4 开启 | **已冻结** | W4A4KV4：A/V-aware 与 K-aware 均开启；A16/K16/V16 三种配置关闭 |
| A/K/V quantizer | symmetric per-token，groupsize=-1 | **已冻结** | 仅 W4A4KV4 真正量化 A/K/V；与 weight groupsize=128 区分 |
| A/K/V clip（激活量化） | 0.9 | **已冻结** | 仅 W4A4KV4 |
| a_loss_ratio | 1 | **已冻结并显式渲染** | 所有模型、所有配置均为 1 |
| block 内 Adam 更新 | true | **已冻结并显式渲染** | `fused_block_adam=true`，使用新 fused 实现 |
| tuner profile | group=128, block=256, backward_samples/backward_bsz=16, global_loss_bsz 按 producer 的 8/4/2/1 容量阶梯，loss_slide_window=false，量化 ceil(1/4) 层 | **已冻结并已实现 override** | 实际成功档：Qwen3-0.6B=8、Llama/Qwen3-4B/8B=4、Qwen3-32B=1；backward 固定 16 |
| tuning Hessian accumulation | 默认 64；仅 Qwen3-32B=32 | **已冻结并实机越过旧 OOM 点** | 用户仅授权降低 Hessian accum bsz；其他模型保持 64，所有 backward/global-loss 参数不变 |
| 正式 profile | group=128, block=128, backward_samples/backward_bsz=32, loss_slide_window=true | **已冻结并显式渲染** | global_loss_bsz 初值 32；OOM 时只按 32/16/8/4/2/1 下调它 |
| final-KL 投影显存 tile | 固定 512 tokens | **已实现并实机越过旧 OOM 点** | 仅分块重计算 `norm + lm_head + KL` 的词表投影；仍选取 32×2048 tokens、形成同一个 mean KL、只做一次 Adam 更新，不改变任何训练 batch/config |
| formal GPU | 单卡 B100 | 已指定 | 每个正式量化 invocation 只占一张 B100 |
| 三项推理评测 | GSM8K / MATH-500 / HumanEval+ | **已冻结** | 明确不跑 LCB |
| thinking / decoding protocol | thinking-on、greedy、n=1、全量 | **已冻结** | max_new_tokens：GSM8K=1024、MATH-500/HumanEval+=2048 |
| HumanEval+ 官方评分 | Canoe CPU-only、每节点 2 组并发、每组 32 worker | **已实现并通过回归测试** | 164 个唯一 task + 30 秒样本稳定门禁；uid/gid 65534、清 capabilities、no_new_privs、seccomp 禁网、64 GiB 地址空间限制；base/plus 分开审计 |
| checkpoint 格式 | versioned quantized checkpoint + runtime manifest | **已冻结** | 原子保存后检查非空，并等待 60 秒确认 size/mtime 稳定 |
| 模型架构支持 | 五个目标均支持 | 已核对 | Llama 为 `LlamaForCausalLM`；四个 Qwen 均为 `Qwen3ForCausalLM` |
| Transformers 兼容性 | 当前 4.56.2 支持 | 已核对 | 本地三个 Qwen3 checkpoint 可由当前环境识别 |

## 3. 二十组实验总表

状态枚举：配置审计中 → 待调参 → 调参中 → 待正式量化 → 正式量化中 → 待评测 → 评测中 → 完成；任何不满足身份/题数门禁的结果记为失败待诊断，不能填写成绩。

| ID | 模型 | W/A/K/V | LR / tuning KL | 正式 checkpoint | GSM8K | MATH-500 | HumanEval+ base/plus | Job / GPU | 状态 | 备注 |
|---:|---|---|---|---|---|---|---|---|---|---|
| 01 | Qwen3-0.6B | 4/16/16/16 | `5e-7 / 0.02661517` | `quantized.pt` / 1,503,480,592 B | `57.01% (752/1319)` | `36.20% (181/500)` | `24.39% (40/164) / 21.95% (36/164)` | `j-p1agegu8nh` / GPU0 | 完成 | formal glb=1；final-KL tile smoke rc=0；strict bracket=`2e-7,5e-7,1e-6` |
| 02 | Qwen3-0.6B | 4/4/4/4 | `5e-5 / 2.21603799` | `quantized.pt` / 1,503,485,968 B | `0.76% (10/1319)` | `0.80% (4/500)` | `0.00% (0/164) / 0.00% (0/164)` | `j-xz8xghpcqh` / GPU0 | 完成 | formal glb=1；activation/KV aware + clip=0.9；strict bracket=`2e-5,5e-5,1e-4` |
| 03 | Qwen3-0.6B | 3/16/16/16 | `2e-7 / 0.09138498` | `quantized.pt` / 1,503,480,592 B | `9.02% (119/1319)` | `6.40% (32/500)` | `0.00% (0/164) / 0.00% (0/164)` | `j-p1agegu8nh` / GPU0 | 完成 | formal glb=1；strict bracket=`1e-7,2e-7,5e-7` |
| 04 | Qwen3-0.6B | 2/16/16/16 | `5e-6 / 0.63418907` | `quantized.pt` / 1,503,480,592 B | `0.23% (3/1319)` | `1.00% (5/500)` | `0.00% (0/164) / 0.00% (0/164)` | `j-xz8xghpcqh` / GPU1 | 完成 | formal glb=1；strict bracket=`1e-6,5e-6,1e-5` |
| 05 | Llama-3.1-8B-Instruct | 4/16/16/16 | `5e-6 / 0.00660720` | `quantized.pt` / 16,060,829,300 B | `83.32% (1099/1319)` | `38.80% (194/500)` | `61.59% (101/164) / 59.15% (97/164)` | `j-xz8xghpcqh` / GPU4 | 完成 | formal glb=8；rc=0，3328.57s；strict bracket=`1e-6,5e-6,1e-5` |
| 06 | Llama-3.1-8B-Instruct | 4/4/4/4 | `1e-6 / 0.19385381` | `quantized.pt` / 16,060,835,444 B | `65.50% (864/1319)` | `21.00% (105/500)` | `37.80% (62/164) / 32.93% (54/164)` | `j-p1agegu8nh` / GPU5 | 完成 | formal glb=8；rc=0，3704.58s；activation/KV aware + clip=0.9 |
| 07 | Llama-3.1-8B-Instruct | 3/16/16/16 | `1e-5 / 0.02330919` | `quantized.pt` / 16,060,829,300 B | `73.16% (965/1319)` | `25.20% (126/500)` | `35.98% (59/164) / 31.71% (52/164)` | `j-xz8xghpcqh` / GPU5 | 完成 | formal glb=8；rc=0，3335.19s；strict bracket=`5e-6,1e-5,2e-5` |
| 08 | Llama-3.1-8B-Instruct | 2/16/16/16 | `2e-5 / 0.14982496` | `quantized.pt` / 16,060,829,300 B | `0.00% (0/1319)` | `0.00% (0/500)` | `0.00% (0/164) / 0.00% (0/164)` | `j-p1agegu8nh` / GPU5 | 完成 | formal glb=8；rc=0，3288.68s；strict bracket=`1e-5,2e-5,5e-5` |
| 09 | Qwen3-4B | 4/16/16/16 | `5e-6 / 0.01076816` | `quantized.pt` / 8,823,939,012 B | `86.35% (1139/1319)` | `55.80% (279/500)` | `44.51% (73/164) / 42.68% (70/164)` | `j-p1agegu8nh` / GPU7 | 完成 | formal glb=4；strict bracket=`1e-6,5e-6,1e-5` |
| 10 | Qwen3-4B | 4/4/4/4 | `5e-6 / 0.36080462` | `quantized.pt` / 8,823,945,860 B | `40.56% (535/1319)` | `20.60% (103/500)` | `0.61% (1/164) / 0.00% (0/164)` | `j-xz8xghpcqh` / GPU6 | 完成 | formal glb=4；activation/KV aware + clip=0.9 |
| 11 | Qwen3-4B | 3/16/16/16 | `0 / 0.03629872` | `quantized.pt` / 8,823,939,012 B | `76.50% (1009/1319)` | `50.20% (251/500)` | `26.22% (43/164) / 25.00% (41/164)` | `j-p1agegu8nh` / GPU0 | 完成 | formal glb=4；audited physical-zero boundary；8 个不同正 LR 均更差 |
| 12 | Qwen3-4B | 2/16/16/16 | `1e-7 / 0.19021450` | `quantized.pt` / 8,823,939,012 B | `0.99% (13/1319)` | `2.00% (10/500)` | `0.00% (0/164) / 0.00% (0/164)` | `j-xz8xghpcqh` / GPU7 | 完成 | formal glb=8；strict bracket=`0,1e-7,2e-7` |
| 13 | Qwen3-8B | 4/16/16/16 | `5e-6 / 0.00733649` | `quantized.pt` / 16,381,751,564 B | `88.86% (1172/1319)` | `56.80% (284/500)` | `48.17% (79/164) / 45.73% (75/164)` | `j-xz8xghpcqh` / GPU0 | 完成 | formal glb=8；rc=0，3590.48s；strict bracket=`1e-6,5e-6,1e-5` |
| 14 | Qwen3-8B | 4/4/4/4 | `1e-6 / 0.25699314` | `quantized.pt` / 16,381,758,476 B | `59.14% (780/1319)` | `24.60% (123/500)` | `1.22% (2/164) / 0.00% (0/164)` | `j-p1agegu8nh` / GPU1 | 完成 | formal glb=8；rc=0，3642.12s；activation/KV aware + clip=0.9 |
| 15 | Qwen3-8B | 3/16/16/16 | `5e-7 / 0.02373810` | `quantized.pt` / 16,381,751,564 B | `86.81% (1145/1319)` | `55.00% (275/500)` | `33.54% (55/164) / 31.71% (52/164)` | `j-xz8xghpcqh` / GPU1 | 完成 | formal glb=8；rc=0，3585.07s；strict bracket=`2e-7,5e-7,1e-6` |
| 16 | Qwen3-8B | 2/16/16/16 | `1e-5 / 0.12215233` | `quantized.pt` / 16,381,751,564 B | `1.97% (26/1319)` | `3.80% (19/500)` | `0.00% (0/164) / 0.00% (0/164)` | `j-p1agegu8nh` / GPU2 | 完成 | formal glb=8；rc=0，3639.91s；strict bracket=`5e-6,1e-5,2e-5` |
| 17 | Qwen3-32B | 4/16/16/16 | `1e-5 / 0.00705344` | `quantized.pt` / 65,527,268,396 B | `93.48% (1233/1319)` | `61.40% (307/500)` | `45.12% (74/164) / 43.29% (71/164)` | `j-p1agegu8nh` / GPU3 | 完成 | formal glb=2；Hessian accum=32；rc=0，19133.33s；strict bracket=`5e-6,1e-5,2e-5` |
| 18 | Qwen3-32B | 4/4/4/4 | `1e-5 / 0.37795773` | `quantized.pt` / 65,527,280,684 B | `92.49% (1220/1319)` | `59.00% (295/500)` | `38.41% (63/164) / 36.59% (60/164)` | `j-xz8xghpcqh` / GPU2 | 完成 | formal glb=2；Hessian accum=32；rc=0，19520.82s；activation/KV aware；strict bracket=`5e-6,1e-5,2e-5` |
| 19 | Qwen3-32B | 3/16/16/16 | `1e-5 / 0.01902595` | `quantized.pt` / 65,527,268,396 B | `92.87% (1225/1319)` | `57.00% (285/500)` | `45.73% (75/164) / 43.90% (72/164)` | `j-p1agegu8nh` / GPU4 | 完成 | formal glb=2；Hessian accum=32；rc=0，19020.10s；strict bracket=`5e-6,1e-5,2e-5` |
| 20 | Qwen3-32B | 2/16/16/16 | `1e-5 / 0.07442397` | `quantized.pt` / 65,527,268,396 B | `30.33% (400/1319)` | `8.60% (43/500)` | `0.00% (0/164) / 0.00% (0/164)` | `j-xz8xghpcqh` / GPU3 | 完成 | formal glb=2；Hessian accum=32；rc=0，19482.29s；strict bracket=`5e-6,1e-5,2e-5` |

### 3.1 正式量化 GPU 时间

下表只统计 20 个 formal run 的单卡 wall time；不含调参、共享 precompute 和三项推理评测。“成功计算量”只累计最终 rc=0 attempt，“含容量重试”还累计 formal 阶段因 OOM 结束的 attempt。

| 模型 | 4 次成功量化 GPU·h | 平均每次 GPU·h | 含 formal 容量重试 GPU·h |
|---|---:|---:|---:|
| Qwen3-0.6B | 0.409 | 0.102 | 1.964 |
| Llama-3.1-8B-Instruct | 3.794 | 0.948 | 3.794 |
| Qwen3-4B | 1.895 | 0.474 | 3.107 |
| Qwen3-8B | 4.016 | 1.004 | 4.016 |
| Qwen3-32B | 21.432 | 5.358 | 21.432 |
| **合计** | **31.546** | **1.577（20 次平均）** | **34.313** |

### 3.2 WikiText2 PPL 与论文十项 QA（checkpoint 补测）

口径为 WikiText2 256×2048 全测试 token、全词表 FP32 CE/KL；QA 使用 lm-eval `0.4.4`，任务顺序固定为 PIQA、HellaSwag、ARC-Easy、ARC-Challenge、WinoGrande、LAMBADA、C-Eval-valid、BoolQ、OpenBookQA、Social-IQA。每项优先 `acc_norm,none`，否则 `acc,none`；先把单项百分比四舍五入到两位，再取十项算术平均。每个 checkpoint 只加载一次，连续完成 PPL 与十项 QA。

| ID | 模型 | W/A/K/V | PPL | QA Avg | PIQA | Hella | ARC-E | ARC-C | Wino | LAMBADA | C-Eval | BoolQ | OBQA | SIQA | 状态 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 01 | Qwen3-0.6B | 4/16/16/16 | 21.7350 | 44.92 | 65.34 | 45.36 | 50.55 | 32.59 | 56.59 | 37.18 | 35.44 | 54.98 | 31.60 | 39.56 | 完成 |
| 02 | Qwen3-0.6B | 4/4/4/4 | 37.9270 | 35.99 | 58.16 | 34.29 | 38.38 | 24.32 | 50.83 | 12.56 | 24.44 | 53.36 | 27.60 | 35.98 | 完成 |
| 03 | Qwen3-0.6B | 3/16/16/16 | 24.7696 | 41.39 | 63.11 | 41.46 | 46.00 | 27.90 | 55.88 | 33.63 | 27.56 | 51.80 | 29.20 | 37.36 | 完成 |
| 04 | Qwen3-0.6B | 2/16/16/16 | 51.3172 | 32.08 | 53.86 | 29.39 | 33.63 | 22.01 | 49.25 | 4.52 | 25.71 | 40.92 | 26.40 | 35.06 | 完成 |
| 05 | Llama-3.1-8B-Instruct | 4/16/16/16 | 7.3739 | 66.66 | 80.58 | 78.31 | 78.54 | 55.29 | 72.77 | 72.15 | 52.23 | 83.85 | 43.80 | 49.13 | 完成 |
| 06 | Llama-3.1-8B-Instruct | 4/4/4/4 | 8.6284 | 62.38 | 78.02 | 74.18 | 74.79 | 48.72 | 69.30 | 66.89 | 43.76 | 80.73 | 40.20 | 47.24 | 完成 |
| 07 | Llama-3.1-8B-Instruct | 3/16/16/16 | 7.8874 | 65.07 | 79.49 | 75.72 | 79.34 | 52.82 | 70.88 | 70.44 | 48.66 | 83.09 | 42.20 | 48.06 | 完成 |
| 08 | Llama-3.1-8B-Instruct | 2/16/16/16 | 11.5613 | 50.54 | 68.99 | 57.75 | 57.74 | 33.87 | 61.96 | 48.42 | 25.48 | 73.09 | 35.80 | 42.27 | 完成 |
| 09 | Qwen3-4B | 4/16/16/16 | 13.7644 | 63.04 | 73.61 | 67.37 | 74.87 | 50.43 | 63.77 | 58.08 | 68.57 | 85.99 | 39.00 | 48.67 | 完成 |
| 10 | Qwen3-4B | 4/4/4/4 | 14.6810 | 56.62 | 72.42 | 61.24 | 63.85 | 41.30 | 61.17 | 50.15 | 55.13 | 80.86 | 35.80 | 44.27 | 完成 |
| 11 | Qwen3-4B | 3/16/16/16 | 13.9629 | 59.71 | 73.78 | 63.70 | 69.11 | 46.84 | 63.06 | 54.41 | 60.62 | 83.85 | 35.40 | 46.37 | 完成 |
| 12 | Qwen3-4B | 2/16/16/16 | 19.8435 | 41.49 | 63.11 | 44.64 | 42.13 | 27.82 | 54.22 | 27.15 | 27.79 | 61.28 | 29.60 | 37.21 | 完成 |
| 13 | Qwen3-8B | 4/16/16/16 | 9.8379 | 67.44 | 78.18 | 74.03 | 79.50 | 55.55 | 68.90 | 63.75 | 75.93 | 86.18 | 41.20 | 51.18 | 完成 |
| 14 | Qwen3-8B | 4/4/4/4 | 10.9895 | 60.49 | 74.59 | 66.81 | 73.61 | 47.78 | 63.54 | 54.03 | 60.77 | 80.31 | 39.00 | 44.47 | 完成 |
| 15 | Qwen3-8B | 3/16/16/16 | 10.2774 | 64.53 | 76.39 | 70.68 | 77.06 | 52.22 | 66.30 | 61.34 | 69.32 | 86.21 | 39.80 | 45.96 | 完成 |
| 16 | Qwen3-8B | 2/16/16/16 | 13.4056 | 49.19 | 69.80 | 55.31 | 57.87 | 35.75 | 60.14 | 39.49 | 27.71 | 68.78 | 34.80 | 42.22 | 完成 |
| 17 | Qwen3-32B | 4/16/16/16 | 7.6608 | 71.58 | 81.34 | 82.25 | 83.33 | 61.01 | 73.40 | 68.08 | 84.70 | 85.32 | 45.00 | 51.33 | 完成 |
| 18 | Qwen3-32B | 4/4/4/4 | 8.2656 | 68.57 | 79.05 | 78.74 | 79.29 | 56.57 | 68.67 | 63.69 | 79.12 | 84.34 | 46.20 | 50.05 | 完成 |
| 19 | Qwen3-32B | 3/16/16/16 | 7.8415 | 70.23 | 79.92 | 80.77 | 79.67 | 60.41 | 70.09 | 66.85 | 80.91 | 87.95 | 46.00 | 49.74 | 完成 |
| 20 | Qwen3-32B | 2/16/16/16 | 9.5914 | 59.05 | 76.17 | 70.31 | 69.07 | 45.90 | 64.25 | 58.84 | 43.68 | 77.95 | 41.40 | 42.94 | 完成 |

### 3.3 BF16 基线（5 个唯一模型映射到 20 行）

同一基础模型的 BF16 权重不随对应 W4/W3/W2 量化行变化，因此每个模型只测一次，再映射到该模型的四行；这既保持比较口径，又避免把完全相同的 BF16 模型重复跑四遍。三项推理沿用主实验的 thinking-on、greedy、n=1 全量协议；HumanEval+ 报告官方 base/plus pass@1。

| 模型 | 对应量化行 | PPL | QA Avg | PIQA | Hella | ARC-E | ARC-C | Wino | LAMBADA | C-Eval | BoolQ | OBQA | SIQA | GSM8K | MATH-500 | HumanEval+ base/plus | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-0.6B | 01–04 | 20.9597 | 47.84 | 67.30 | 47.30 | 55.89 | 33.70 | 56.43 | 39.84 | 43.31 | 63.82 | 31.80 | 39.05 | 68.01% (897/1319) | 51.00% (255/500) | 34.76% (57/164) / 31.71% (52/164) | 完成 |
| Llama-3.1-8B-Instruct | 05–08 | 7.2143 | 67.21 | 80.96 | 79.20 | 79.67 | 55.12 | 73.56 | 73.06 | 54.01 | 84.10 | 43.00 | 49.39 | 85.67% (1130/1319) | 40.60% (203/500) | 58.54% (96/164) / 53.05% (87/164) | 完成 |
| Qwen3-4B | 09–12 | 13.6580 | 64.52 | 74.92 | 68.46 | 78.20 | 53.58 | 65.27 | 59.36 | 70.36 | 84.95 | 40.20 | 49.90 | 87.19% (1150/1319) | 60.20% (301/500) | 60.98% (100/164) / 57.93% (95/164) | 完成 |
| Qwen3-8B | 13–16 | 9.7232 | 68.20 | 77.80 | 74.90 | 80.85 | 56.74 | 68.11 | 64.12 | 79.57 | 86.61 | 41.60 | 51.69 | 85.37% (1126/1319) | 56.80% (284/500) | 53.05% (87/164) / 50.61% (83/164) | 完成 |
| Qwen3-32B | 17–20 | 7.6099 | 71.96 | 81.94 | 82.59 | 83.25 | 60.84 | 73.40 | 67.13 | 85.81 | 86.39 | 45.80 | 52.41 | 93.10% (1228/1319) | 61.40% (307/500) | 51.22% (84/164) / 48.17% (79/164) | 完成 |

## 4. 调度计划

1. 先完成 CLI/default/config 语义审计；所有用户指定项必须显式出现在冻结 manifest，不能依赖不透明默认值。
2. 在宿主机下载缺失模型和离线依赖；Canoe pod 不联网，不在宿主机加载模型或运行量化/生成/代码评分。
3. 申请两台 8×B100 debug 机器。调参每个候选单卡运行，两台机器最多并行 16 个独立候选；正式量化严格单卡 B100，每卡一个独立 writer。
4. 各组 tuner 先运行一次 campaign 专属 producer，所有候选只允许 cache hit；候选由 agent 按小批次显式放行，同组 LR 从低到高，单组最多 20 次 launch。每批结束后检查 Exact KL、高侧证据和失败类型，再决定继续扩展、局部细化或选择。
5. 只有 agent 选择记录通过“已观测全局最低点 + 更低/更高 LR 严格邻居”门禁后才允许启动正式量化；若唯一最优点恰为非负 LR 的物理下界 0，则要求至少 3 个不同正 LR 均严格更差并以独立 decision kind 留档。若单卡 B100 OOM，只降低 `global_loss_bsz`，保留 `backward_bsz`；每次 immutable retry 单独留档。
6. checkpoint 完整且大小稳定后，按独立 task/output/port 运行三项评测；HumanEval+ 生成与官方评分解耦。
7. 最终逐组检查配置 fingerprint、题数、状态、结果哈希、checkpoint 和 manifest，再填写成绩。

## 5. 完成门禁

- [x] 20/20 组都有独立、收敛且可审计的 best LR。
- [x] 20/20 正式量化均在单卡 B100 完成并保存稳定 checkpoint。
- [x] 每组配置显式满足 group=128、symmetric、aware、clip、act-order、w-clip、rotate、block-Adam 和 a_loss_ratio=1 要求。
- [x] GSM8K 每组 1319 个当前身份 generation，状态 completed/scored。
- [x] MATH-500 每组 500 个当前身份 generation，状态 completed/scored。
- [x] HumanEval+ 每组 164 个 generation，官方 result 覆盖 164 task，base/plus 分开报告。
- [x] 无宿主机高负载模型、生成或代码判分进程；所有 writer、GPU 和端口唯一。
- [x] 总表、原始路径、job/GPU、异常和重试记录完整。
- [x] 20/20 checkpoint 的 WikiText2 PPL 与 200/200 个论文 QA 单项通过新增质量审计。
- [x] 5/5 唯一 BF16 模型的 PPL、50 个 QA 单项、15 个推理任务与 5 个官方 HumanEval+ 结果通过审计，并映射回 20 行。

## 6. 实时记录

### 2026-08-08：启动与初步审计

- 当前矩阵经两次用户变更后为 5 模型 × 4 设置 = 20 组。
- 本地已存在 `modelzoo/Qwen3/Qwen3-0.6B`、`Qwen3-4B`、`Qwen3-8B`、`Qwen3-32B`；配置均为受支持的 `Qwen3ForCausalLM`，且 tokenizer 均有 chat template。只缺 Llama-3.1-8B-Instruct，后续从 ModelScope 下载。
- `调参方法.md` 的旧默认对 Qwen 4B 使用 `a_loss_ratio=0.95`，但用户本轮明确要求全部固定为 1，本 campaign 将显式覆盖旧特例。
- 推理复盘文档的当前完成门禁是 GSM8K、MATH-500、HumanEval+；LCB 被明确排除。仍需核对集成后的 RealQ CLI/launcher，确认本轮“三项”没有新的定义。
- 尚未申请机器或启动实验；必须先完成全部配置歧义审计。

### 2026-08-08：已撤销的 Qwen3.5 硬阻塞与仍待确认的配置冲突

- 原目标曾为 `LLM-Research/Meta-Llama-3.1-8B-Instruct` 和三种 Qwen3.5。用户随后将三种 Qwen3.5 改为本地已有的 Qwen3-4B/8B/32B，并追加 Qwen3-0.6B；下面的 Qwen3.5 硬阻塞仅作为审计历史保留，不再影响当前 campaign。
- Qwen3.5 不是 Qwen3 的同构更名。官方元数据将其标为 `qwen3_5`，文本骨干混合 Gated DeltaNet 与 Gated Attention，并带视觉编码器。当前 `utils/model_utils.py::ModelAnalyzer` 对所有结构访问都只接受 `LlamaForCausalLM`、`Qwen3ForCausalLM`、`Qwen3MoeForCausalLM`；旋转、逐层模块枚举、attention/KV aware 和 checkpoint wrapper 均没有 Qwen3.5 分支。
- 仓库 `.venv` 的 Transformers 为 4.56.2，`transformers.models.qwen3_5` 不存在；ModelScope 官方 Qwen3.5 页面要求最新 Transformers 主线。因此三种 Qwen3.5 模型目前连受支持的 HF model class 都无法在本环境构造，不能直接进入 RealQ。
- 当时结论是 Qwen3.5 需要独立架构适配，不能靠添加 architecture 字符串绕过；当前已通过更换目标模型解除。
- 自动调参器按上一轮冻结协议强制 `w_groupsize=-1, blocksize=256`，而本轮及 `对比实验方法.md` 明确所有实验 `w_groupsize=128`；若搜索阶段仍用 -1，最佳 LR 与正式 group=128 不完全同构，需要用户明确本轮是保留旧 tuner profile，还是把 campaign 的 tuning profile 改为 group=128。
- `k_cache_quant_aware_gptq=true` 在 `k_bits=16` 时会被配置校验直接拒绝。因此“激活/K/V aware”只能自然地解释为 W4A4KV4 开启 A/V-aware 与 K-aware；W4A16/W3A16/W2A16 必须关闭 aware 开关。该解释需用户确认。
- 推理评测代码默认含四项任务并默认采样（temperature=0.6、top-p=0.95、max_new_tokens=4096）；复盘文档的当前三项完成门禁则是 GSM8K/MATH-500/HumanEval+，并采用 thinking-on、greedy、各生成一次及任务特定上限。必须显式冻结，不能沿用代码默认值。

### 2026-08-08：用户确认冻结协议

- 用户确认 tuner 搜索阶段与正式阶段均使用 `w_groupsize=128`。本 campaign 将通过新显式参数覆盖 tuner 的通用默认 `-1`，不静默修改历史 campaign 语义。
- 用户确认仅 W4A4KV4 启用 A/V-aware 与 K-aware；A/K/V 为 symmetric per-token（groupsize=-1）、clip=0.9。W4A16、W3A16、W2A16 的 A/K/V 均为 16 bit 且 aware 开关关闭。
- 用户确认推理评测仅 GSM8K、MATH-500、HumanEval+，thinking-on、greedy、n=1、全量；生成上限依次 1024、2048、2048，不跑 LCB。

### 2026-08-08：启动器、机器与模型准备

- 新增 `experiments/realq_20group_20260808/campaign.py`：20 组固定矩阵均匀分配给两个节点（每节点 10 组），每个量化、调参候选和评测 invocation 均为单卡 world-size=1；最多使用每节点 8 个独立 GPU lane。
- 启动前复核 `调参方法.md` 后，撤销了“连续自动扩展直到收敛”的方案。本 campaign 使用 `tools/realq_auto_tune.py` 新增的 controlled batch 入口：每批只运行 agent 显式给出的低到高 LR，运行后停在 `awaiting_agent_decision`；每组最多 20 次，选择理由和三点 bracket 原子写入 `controlled_decision.json`。连续模式实现保持不变，但本 campaign 不调用。
- 首批调度按“一个 LR = 一个原子 worker”滚动补位；同组候选必须看到前一个 candidate success marker 才可运行，因此严格保持低到高，同时避免一个组占卡连续跑完整批次造成 6 卡拖尾。tuning cache 与 formal cache 也拆成独立 producer/marker：首批只计算 tuning cache，formal cache 延后到 LR 选择完成后。
- 每个模型只允许一个加锁的 tuning/formal cache producer；消费者必须等待对应 success marker。正式 OOM 重试只降低 `global_loss_bsz`，阶梯固定为 32/16/8/4/2/1，`backward_bsz=32` 不变。
- 新增模型完整性硬门禁：读取 `model.safetensors.index.json` 并逐一核对分片存在且非空；任何 `.incomplete`、缺失 tokenizer 或缺失分片都会阻止生成 plan/启动该模型。
- 定向测试 `tests/test_realq_auto_tune.py` 与 `tests/test_realq_20group_20260808.py` 在 LR 选择协议重建修复后共 39 项全部通过；仅有现存 `.pytest_cache` 无写权限警告，不影响测试结果。
- 当前已识别两项 8-GPU 资源：`j-adb1691wsv`（Running）与 `j-5s5kdonxda`（初始 Inqueue）。平台内部/NVML 名称显示 `NVIDIA L20C`；本仓库现有 benchmark 记录注明用户确认该标签对应本实验使用的 B100 资源。MetaGod CMDB 查询返回 403，按工具约束未重试，因此最终仍以每个实际 job 保存的 `nvidia-smi` 型号、显存和 UUID 快照为审计事实。
- 由于 Canoe debug job 不提供可用的 pod exec，已有 sleep job 不能原地替换命令。`j-adb1691wsv` 与 `j-5s5kdonxda` 已经安全校验 owner/project 后停止；它们只运行 sleep，没有产生实验结果。
- 第一版正式等待 job `j-i948klhsw8` / `j-4yg3d24zgs` 在启动前发现 campaign 根误指向 root-owned 的仓库 `output/`，分别处于 Inqueue/等待状态，未启动调参即安全停止。campaign 根已更正为可写共享目录 `/minimax-avatar-new/zhangqian/realq/experiment_data/realq_20group_20260808`。
- 更正后的首批两节点 job 为 `j-manzaxchig`（node 0）和 `j-nthxsmqg93`（node 1）；命令在 `plan.json` 与 `batches/initial.json` 原子发布前每 30 秒等待，发布后运行 controlled batch `0,1e-6,5e-6`。
- 仓库 `.venv` 安装了 `modelscope==1.39.1` / `modelscope-hub==0.2.0`。首次下载误包含冗余的 16.1GB `original/consolidated.00.pth`，已中断并改为 `--exclude '*.pth'` 的 safetensors-only 下载。
- ModelScope 的断点续传把 HTTP “剩余长度”写成新 partial，导致截断 shard 经哈希失败后再次从零下载。3 个损坏的 safetensors partial 和 `.pth` partial（合计约 3.7 GB）已精确移动到 campaign 的 `download_remainders/modelscope_broken_resume_20260808/`，可恢复、未删除；完整 shard 4、tokenizer 与配置保留，剩余 shard 从零下载。模型门禁还会核对 index `total_size` 与所有 shard 文件总大小，截断文件不可能进入 plan。
- 随后启用 `modelscope-hub` 默认关闭的 HTTP Range 并行：每次一个 shard、8 个 range worker、160 MB part；part 自身可安全续传，完整文件仍按远端 SHA256 校验。
- HumanEval+ 普通 GPU job 只负责生成 164 题候选；官方 base/plus 代码执行必须进入独立隔离沙箱，未完成官方评分前该列不能标为完成。

### 2026-08-08：模型就绪、首轮入口故障与兼容修复

- Llama-3.1-8B-Instruct 的 4 个 safetensors 分片已全部下载完成，总文件字节数 `16,060,556,376`，index 声明 tensor bytes 为 `16,060,522,496`，头部开销 `33,880` 字节；4 个分片的 safetensors header 均可解析。五个模型的 config/tokenizer/index/shard 门禁以及三份评测数据的行数/hash 门禁全部通过。
- 首份已发布 plan 因随后发现容器 Python 不兼容而作废，未产生任何 tuning/formal/eval marker。修复后的冻结计划为 campaign `realq-llama31-qwen3-20group-20260808-v2`，plan fingerprint=`e7a178071b63a95f9de9342a4b5b98df2add8bbea192164dcfb2b33aaea8e30e`，明确记录 20 个完整 resolved config 和 Python 路径。
- `j-manzaxchig` 与 `j-nthxsmqg93` 在 plan 发布后均于任何 Python import、GPU 计算或结果写入前失败。跨 Pod mlog 的最早错误为 `2026-08-08 17:42:45 +08:00 /bin/bash: .venv/bin/python: No such file or directory`，退出码 127；两任务在 retry=10 后分别 Failed。`dead_node_check.py` 确认节点 `up{}` 持续为 1 且没有高危 pod event；MetaGod 因当前身份 403 无权限而跳过，因此硬件层未由 CMDB 覆盖，但已排除 dead-node 证据。
- 原因是宿主重建后的 canonical `.venv/bin/python3 -> /usr/local/bin/python3`，而冻结 Canoe 镜像只提供 `/usr/bin/python3.12`。未覆盖系统 Python、未改写 canonical `.venv`；campaign 仅增加精确白名单兼容入口 `.venv.py312-broken-20260729/bin/python`。这是仓库内此前在同一镜像验证过的保留环境，节点入口会在分配 GPU 后、任何实验前强制核验 `sys.prefix`、Python、torch=`2.9.1+cu128`、transformers=`4.56.2`、lm-eval=`0.4.4`，并将快照写入 node manifest。
- 初始受控批次固定每组 `[0, 1e-6, 5e-6]`。当前 `batches/initial_v2.json` 同时绑定 plan fingerprint，batch fingerprint=`5315b71d2fccb9b73292eb0f9ec03b5e8e04678276b6e22ef505c6d3199c82af`；先前未绑定 plan 的同名文件已可恢复地归档为 `initial_v2_unbound_archived_20260808.json`，不会被执行。
- 修复后的两台 8 卡任务为 node0=`j-zrjcfdird8`、node1=`j-4drxxvck1a`，均于 `17:52:34 +08:00` 创建，初始状态 Pending/retry0；入口绑定 `initial_v2.json` 和兼容 Python，OnFailure 重试上限收紧为 2。只有 node manifest 的版本门禁和 8×GPU 快照落盘后才算有效开跑。

### 2026-08-08：首个有效 GPU 入口审计与 deterministic producer 重签

- v2 两节点均成功写入 Python runtime 与 8×`NVIDIA L20C` 快照，证明兼容环境门禁通过。随后首个 worker 因 scheduler 用脚本路径启动而缺失仓库根 `sys.path`，报 `ModuleNotFoundError: No module named 'tools'`；改为 `python -m experiments.realq_20group_20260808.campaign` 后，node0 retry1 同时启动 Qwen3-0.6B/4B/32B 三个 tuning producer，模型加载、WikiText2 256×2048 取样、旋转与 static precompute 均进入真实 GPU 路径。
- 真实日志显示 tuning producer 的 attention backward 出现 nondeterministic algorithm 警告，而受控 LR candidate 在 `realq_auto_tune.py` 中明确设置 `REALQ_DETERMINISTIC_SDPA=1`。这会让共享 cache 与候选的数值语义不完全一致，因此 v2 producer 结果不采纳；当时尚无 `tuning_success.json`、trial、formal 或 eval marker。
- node0=`j-zrjcfdird8` 与仍 Pending 的替代 node1=`j-dy9z6qr9la` 经 project/creator 双重校验后安全停止。所有 v2 partial precompute、shared cache、node manifest 和 plan 已整体移动到 `diagnostics/protocol_v2_nondeterministic_precompute/`，保留可审计、未删除，也不会被 v3 路径命中。
- v3 显式冻结 `tuning_deterministic_sdpa=true`、`formal_deterministic_sdpa=false`，分别与 tuner candidate 和正式 main 的既有语义一致。campaign=`realq-llama31-qwen3-20group-20260808-v3`，plan fingerprint=`fa29f11305bf4db3485d03cd0b822b00eefb4ec3c6d17eae19049c696bcfbf1d`；`initial_v3` batch fingerprint=`a349c6ec39b2d1bb9cee30b97475e05e15cc0b978ae4bfabd781949f57e5d226`，绑定同一 plan，覆盖 20 组各 `[0,1e-6,5e-6]`。
- v3 两节点已提交：node0=`j-in9peno0i4`、node1=`j-z6788pvkpy`，均请求 8 GPU、单卡 worker lanes、retry=2；等待资源后重新从空的 v3 precompute/cache 目录开始。

### 2026-08-08：tuning OOM 阶梯与 v4 有效首批

- v3 在真实单卡上证明 deterministic tuning producer 的 `global_loss_bsz=8` 对 Qwen3-4B 已不足：SDPA static/reference 前向请求额外约 4 GiB 时单卡 178.35 GiB 已使用约 177.9 GiB。该 OOM 发生在 producer 前向，不是 backward，因此不能通过改变 `backward_bsz=16` 解决。node0=`j-in9peno0i4`、node1=`j-z6788pvkpy` 经身份校验后安全停止；v3 partial 与日志整体可恢复地归档到 `diagnostics/protocol_v3_tuning_glbsz8_oom/`，不参与后续结果。
- v4 为 tuning producer 增加唯一允许的 OOM 阶梯 `global_loss_bsz=8/4/2/1`，每一步保持 `backward_samples=16`、`backward_bsz=16` 和其余协议不变；producer success marker 记录实际成功值，所有同模型受控候选读取该值并写入各自 protocol。正式量化仍使用独立的 `32/16/8/4/2/1` 阶梯并固定 `backward_bsz=32`。
- 当前冻结 campaign=`realq-llama31-qwen3-20group-20260808-v4`，plan fingerprint=`13782883a21e69e84b63e2f7fafe1f775b4f3d8e8e2648b8b69e80f0fb1e147e`；`batches/initial_v4.json` batch fingerprint=`cfc00972b3f4d58d3c9f50919045e0b1418c81fc339618dcf569375c48a8831a`，绑定该 plan，覆盖 20 组各 `[0,1e-6,5e-6]`。
- v4 两台 8 卡任务为 node0=`j-4pcw0h1sce`、node1=`j-uurbf11cgu`。截至约 `18:28 +08:00` 两者均为 Running、retry=0，并已通过 Python/GPU runtime 门禁。Qwen3-0.6B 在 tuning `global_loss_bsz=8` 成功；Qwen3-4B 的 8 档按预期 OOM 后在 4 档成功。Llama-3.1-8B-Instruct 与 Qwen3-8B 已从 8 档降到 4 档继续，Qwen3-32B 的 8 档仍在运行。
- Qwen3-0.6B 四种设置的首个 `lr=0` 均已成功并写入 Exact KL：W4A16=`0.026647988706827164`、W4A4KV4=`4.996985912322998`、W3A16=`0.09165721386671066`、W2A16=`0.7600040435791016`。这些仅是受控搜索观测点；没有低侧邻居，当前不能据此选择 LR。四组均已按协议进入 `lr=1e-6`。
- Qwen3-32B 依次在 tuning `global_loss_bsz=8/4/2/1` OOM；最后一档已经进入首个 static backward，只因申请 1 GiB 时设备空闲约 792 MiB 而失败，同时 PyTorch 报告约 2.89 GiB reserved-but-unallocated，并明确建议 `expandable_segments:True`。这不是继续降低 backward 的授权依据。失败 node0 `j-4pcw0h1sce` 最终 Failed/retry2；它的 32B 日志和 partial output 已可恢复地移动到 `diagnostics/protocol_v4_qwen32b_allocator_fragmentation/`，其余模型的成功 marker 未移动。
- campaign worker 现显式设置同值的 `PYTORCH_ALLOC_CONF` 与兼容别名 `PYTORCH_CUDA_ALLOC_CONF` 为 `expandable_segments:True`；只改变 CUDA allocator 的分段策略，不改变 batch、backward、精度、随机性或量化参数。新日志 header 与每次 producer attempt 都记录该值，定向测试更新为 34 项通过。替代 node0=`j-l1yyvjedrh` 已提交并从现有候选 marker 断点继续；node1=`j-uurbf11cgu` 全程未停止。
- `j-l1yyvjedrh` 的两次 28 秒运行经 mlog 定位为同一恢复门禁，而非新的 32B OOM：原 node0 退出时同时 SIGTERM 了 Llama W4A4KV4/W2A16 的 `lr=1e-6` worker，二者没有原子 result，但 state 保留旧 PID。tuner 按设计拒绝自动重跑以避免重复计算，导致 node0 替代任务 Failed/retry2。
- 新增独立的 agent 授权 `recover-interrupted-trial` 入口：必须持有非阻塞 controller lock、必须没有原子 result、必须匹配 state/spec/LR/protocol，并强制写审计理由；旧 trial 目录原子移动到 `diagnostics/interrupted_trials/`，且中断 launch 继续计入每组 20 次硬上限。两个 stale trial 已以 `j-4pcw0h1sce` terminal Failed/retry2 为证据完成恢复；测试现为 36 项通过。新的 node0=`j-urh55l2qe8` 继承 allocator 配置并从 marker 继续。
- 后续恢复暴露出同一次原 node0 退出还中断了 Qwen3-4B W4/W3 的 `5e-6` 和 Qwen3-8B W4A4KV4/W2A16 的 `1e-6`；第一次恢复后的两个 Llama worker 又被 `j-urh55l2qe8` 因 Qwen3-4B stale 门禁退出而二次中断。已枚举全部 20 组 state，逐项确认无原子 result 后按上述入口恢复；所有中断 launch 均保留并计数。恢复后全矩阵 `running` map 为空。
- scheduler 内部 worker 失败策略已改为停止派发新任务、等待所有已启动 lane 到达原子 result 边界后再让 node 失败，不再向无关 lane 发 SIGTERM；这只增强故障隔离，不改变量化或调参计算。测试更新为 37 项通过。新两节点为 node0=`j-ocn1d0inn2`、node1=`j-ip0k4e59tu`，均显式携带 expandable-segments allocator 配置。
- 新两节点首次并行启动时，Qwen3-32B producer 日志在 `11:05:20Z` 与 `11:05:22Z` 出现两个不同 Pod 的 command header，证明 CPFS 上的 `flock` 不能作为可靠的跨节点排他锁。node1=`j-ip0k4e59tu` 经项目/创建者校验后立即安全停止；两 producer 只重叠约 2 秒，均尚未进入 cache 写阶段。campaign 已禁用跨节点自动 failover：各模型 producer 固定由 model-index parity 主节点负责，另一节点只等待 success marker。测试更新为 38 项通过；node0 继续运行，替代 node1=`j-hs9cyom375` 使用 no-failover 代码恢复其余候选。
- Qwen3-32B 在启用 expandable-segments 后对 tuning `global_loss_bsz=8/4/2` 的失败仍均为真实容量 OOM；固定 `backward_samples=16`、`backward_bsz=16` 后，`global_loss_bsz=1` 于 `11:29Z` 成功写入第五个模型的 `tuning_success.json`。至此五个模型的共享 tuning producer 全部完成，首批候选为 `48/60`，其余 12 个均为 Qwen3-32B；四个 32B 设置的首个候选已经派发并开始各自的量化前置缓存。
- 首次尝试为 Llama-3.1-8B-Instruct W4A4KV4 写入 LR 选择时，门禁正确拒绝了 protocol fingerprint 不一致。根因是 `tuner_select_command` 重建 protocol 时漏带了与 tuning 命令一致的 14 个 formal tail 参数；现已复用同一组参数并增加“选择命令重建相同 protocol base”的回归测试，总计 39 项通过。修复不改变任何已运行候选的计算或结果。
- Llama-3.1-8B-Instruct W4A4KV4 已在完整三点严格 bracket 上选择 `lr=1e-6`：Exact KL 分别为 `lr=0: 0.23139838874340057`、`1e-6: 0.19385381042957306`、`5e-6: 0.22385969758033752`。`controlled_decision.json` 与 state 均原子记录 `selected_lr=1e-6`、`status=tuning_complete`、protocol fingerprint=`9273bb4db6d51fc468b590bfe0e2d21d7880aab7a72fa88cdf68e38705e0c85f`；三次成功候选加两次已审计中断共计 5 次 launch。
- 当前有效作业为 node0=`j-ocn1d0inn2`、node1=`j-hs9cyom375`；最近一次平台查询均为 Running、retry=0。后续先等待 Qwen3-32B 的 12 个首批点到达原子结果，再根据全部 Exact KL 分别生成高侧扩展或 `0..1e-6` 内部细化批次；除已具备严格 bracket 的上述 Llama W4A4KV4 外，不提前选择边界最优点。

### 2026-08-08：Qwen3-32B Hessian 容量 OOM 与 v5 单一例外

- 非 32B 的 v4 首批最终为 `48/48` 个成功候选。除 Llama-3.1-8B-Instruct W4A4KV4 外，Qwen3-8B W4A4KV4 也形成严格三点 bracket：`lr=0 / 1e-6 / 5e-6` 的 Exact KL 为 `0.28341418504714966 / 0.2569931447505951 / 0.3620162904262543`，已原子选择 `lr=1e-6`；三次成功加一次已审计中断共 4 次 launch，protocol fingerprint=`cc67a422b4b9ba2bc1076e5177e4a374f11fc7330844551111f4b9fd2e0e7f68`。
- node0=`j-ocn1d0inn2` 与 node1=`j-hs9cyom375` 最终均为 Failed/retry2。四个 Qwen3-32B 设置的 `lr=0` 失败 sidecar 完全一致：`global_loss_bsz=1`、`backward_samples=16`、`backward_bsz=16`、`hessian_accum_bsz=64`；在 layer-0 Hessian accumulation 的 deterministic SDPA 前向申请 16 GiB 时，单卡仅余 8.17 GiB。PyTorch reserved-but-unallocated 仅 74.85 MiB，因此这是容量 OOM，不是 allocator fragmentation，也不发生在 final-layer/backward。
- 用户明确授权“可以降 hessian accum bsz，别的不要改”。v5 因此只对 `qwen3-32b` 固定 `hessian_accum_bsz=32`；Qwen3-0.6B/4B/8B 与 Llama-3.1-8B-Instruct 继续为 64，`global_loss_bsz=1`、`backward_bsz=16`、所有校准/量化/随机性参数均不变。代码增加逐模型冻结断言与 v4 producer adoption 门禁；语法检查及两份定向测试共 `41 passed`，仅有历史 `.pytest_cache` 权限警告。
- v4 plan、两个 batch、node manifest 和四个 32B 失败 run 均可恢复地归档在 `diagnostics/protocol_v4_qwen32b_hessian_accum64_oom/`；旧 plan 文件 SHA256=`35e4530ea07871579959538c52d9fb975a757ab614b05a7226e2985f060f4c27`。没有删除失败 sidecar，也没有移动或修改 16 个非 32B run。
- 当前冻结 campaign=`realq-llama31-qwen3-20group-20260808-v5`，plan fingerprint=`e187159ad237ac40d06b45aa3271b91ed104112d495d944d7374fdec2732f538`。五个 v4 tuning producer marker 均重新计算并匹配具体模型 artifact/cache identity 后原子 adoption 为 v5；旧 marker 原文保存在 `diagnostics/campaign_v5_precompute_marker_adoption/`，不重跑共享 GPU producer。
- v5 合并恢复批次为 `batches/recovery_refine_v5.json`，batch fingerprint=`910b6e14636843429f93efa4eda727825a2a015cd0c99af9630caf86fc73aa15`，覆盖 18 个未完成组、36 个候选：四个 32B 设置重跑 `[0,1e-6,5e-6]`；9 个非 32B 高边界组只先加 `1e-5`；5 个零边界组在既有 `0..1e-6` 内加 `[1e-7,2e-7,5e-7]`。两组已选定的 W4A4KV4 不重复运行。
- v5 loader 已重新验证 batch campaign/plan/fingerprint；node0 与 node1 均精确分到 18 个候选（分别覆盖 8/10 个 run），避免一侧因候选数不平衡提前空转。目标固定队列 `minimax-avatar-h800new` 最近一次查询为 allocatable 664 GPU、allocated 570 GPU，足以同时提交两个 8-GPU node。
- 用户确认提交后，v5 两节点恢复任务已创建并于 `20:34 +08:00` 进入 Running：node0=`j-w8tdjymb3w`、node1=`j-p5gywa37yz`。两者均为 minimax-avatar 项目、单 Pod/8 GPU、retry=0，入口分别绑定 `run-node --node 0/1 --stage tune --batch-file batches/recovery_refine_v5.json`；两个 Pod 均 Ready，且已各自写入 campaign v5 node manifest。
- 启动后的实际 `spec.json` 审计确认：当前物化的 Qwen3-32B worker 为 `hessian_accum_bsz=32`、`global_loss_bsz=1`、`backward_samples=16`、`backward_bsz=16`、`blocksize=256`；其余模型仍为 `hessian_accum_bsz=64`，对应 global-loss/backward/block 参数均未改变。16 张卡已滚动派发非 32B refinement 与首批 32B W4A16/W3A16 `lr=0`，后续以原子 candidate result 补位。
- `20:50 +08:00`，四个 Qwen3-32B `lr=0` 候选均已越过 v4 的原始失败点：W2A16/W4A4KV4 完成 layer 0 分别约 `40.63s/39.31s`，随后推进到 layer 3/16；W3A16/W4A16 完成 layer 0 分别约 `49.01s/48.35s`。实时显存抽样中 W2A16 所在卡约用 119.8 GiB、仍余 62.8 GiB；全过程无 OOM/failure sidecar。该证据确认仅将 32B Hessian accumulation 从 64 降至 32 已解除四种设置的 layer-0 容量 OOM。
- 非 32B 的 v5 refinement 为 `24/24` 成功、零 failure；结合 v4 首批证据，新增 9 个严格 bracket 选择：Llama W4A16=`5e-6`，Qwen3-0.6B W4/W3/W2=`5e-7/2e-7/5e-6`，Qwen3-4B W4/W4A4KV4/W2=`5e-6/5e-6/1e-7`，Qwen3-8B W4/W3=`5e-6/5e-7`。连同先前 Llama 与 Qwen3-8B 的 W4A4KV4=`1e-6`，当前非 32B 已选择 `11/16`；Llama W3/W2、Qwen3-0.6B W4A4KV4、Qwen3-8B W2 保留为 `1e-5` 高边界，Qwen3-4B W3 保留为 0 低边界，不提前选择。
- 四个 Qwen3-32B `lr=0` 均写出原子成功结果，Exact KL 为 W4A16=`0.0071783089`、W4A4KV4=`0.4288357496`、W3A16=`0.0200191177`、W2A16=`0.0832047164`；随后严格按低到高顺序进入 `lr=1e-6`。截至该时点 v5 总计 `28/36` 个候选成功、零失败。
- 为复用两条任务在 32B 同组候选串行期间的空闲卡，新增受控 batch `boundary_refine_v5.json`，fingerprint=`5ff101961e83c0b42b324e25037be3d4c4cbd3c9979ee8986bcdd8d1bf408517`。四个 `1e-5` 高边界组各只增加 `2e-5`；Qwen3-4B W3A16 按升序增加 `[1e-8,2e-8,5e-8]`，合计 7 点。任务直接运行在现有两 Pod 的 GPU2–4，32B 保持 GPU0/1；实际 spec 复核确认非 32B Hessian=64、原 global-loss 档、backward=16 均不变，未提交额外机器。
- 四个 Qwen3-32B `lr=1e-6` 也全部原子成功，Exact KL 为 W4A16=`0.0071643288`、W4A4KV4=`0.4145039320`、W3A16=`0.0198655184`、W2A16=`0.0797107890`；v5 主批累计 `32/36`、零失败，四组均已在原 GPU0/1 自动进入最后的 `lr=5e-6`。
- 非 32B 边界补点全部成功：Llama W3 在 `2e-5` 变差后选择 `1e-5`；Qwen3-8B W2 在 `2e-5` 变差后选择 `1e-5`；Llama W2 的 `2e-5` 优于 `1e-5`，而 `5e-5` 又变差，因此选择 `2e-5`；Qwen3-0.6B W4A4KV4 在 `2e-5/5e-5/1e-4` 中以 `5e-5` 最优。
- Qwen3-4B W3A16 的 `1e-8/2e-8/5e-8` Exact KL 分别为 `0.0369053520/0.0365462303/0.0366318449`，连同既有 `1e-7..5e-6` 共 8 个不同正 LR 全部严格劣于 `lr=0` 的 `0.0362987220`。tuner 新增显式 `physical_zero_boundary` 决策：只允许 0 为唯一观测全局最优且至少 3 个正 LR 全部更差；不制造不存在的低侧邻点。语法检查与定向回归测试为 `42 passed`，不改任何训练超参。
- 至此 16 个非 32B run 均已原子写入 `status=tuning_complete`，其中 15 组为严格双侧邻点选择、Qwen3-4B W3A16 为上述物理零下界选择；主表已全部更新为待正式量化。
- 四个 Qwen3-32B `lr=5e-6` 最终也全部原子成功，Exact KL 为 W4A16=`0.0070993565`、W4A4KV4=`0.4136393666`、W3A16=`0.0194189772`、W2A16=`0.0769282803`；四组都继续优于 `1e-6`，因此不能提前选择中点。两条 v5 job 分别于 `21:58/21:59 +08:00` 正常 Completed、retry=0，主批完整结果为 `36/36` 成功、零失败。
- 已生成只含四个 Qwen3-32B `lr=1e-5` 的下一受控 batch `q32_boundary_refine_v5.json`，batch fingerprint=`36b4ccfb01687e0b71d6d42e33ae966628b412c63c6333c58d90ac9b2d97fea9`，继续绑定 v5 plan fingerprint。该批不改变 Hessian=32、global-loss=1、backward=16 或任何其余训练参数，等待新的 Canoe 提交确认。
- 对该追加 batch 做了 controller→tuning worker 的完整 resolved-config 预演：node0 仅含 W4A16/W3A16，node1 仅含 W4A4KV4/W2A16，每侧各两个单卡候选；四组均为 group=128、block=256、global-loss=1、backward_samples/backward_bsz=16、Hessian accum=32、a_loss_ratio=1、act-order/w-clip/rotate=true。W4A4KV4 继续使用 A/K/V4、clip=0.9、schedule=none；其余三组 A/K/V16、clip=1、cosine。计算映射本身每侧只有两个 worker，但冻结 launcher 的完整节点门禁仍要求申请 8 卡，详见下一条异常记录。
- 上一条关于 2 卡资源可行性的判断不成立：campaign 在 scheduler 前还有 `gpu_snapshot()` 强门禁，固定要求恰好看到 GPU `0..7`。首次直接把 campaign 绑进 2 卡 Pod 入口的 `j-kaz2giboif`/`j-wzdokv1559` 因此均在数秒内 Failed/retry=2；两者都没有写 node manifest、trial spec 或原子候选结果，故没有训练启动、重复计算或 stale trial。服务端更早的两次带 `--hw-check` 的 2 卡请求也在创建 Job 前以 `hardware check not support` 被拒绝。
- 用户指出并重新确认 `AGENT.md` 规定的标准流程：参考 `j-7x9o0je4pk` 提交 8 卡长 sleep debugging job，再 SSH 进入 Pod 执行实验；同时授权本 campaign 后续任务提交无需逐次确认。已按该流程创建 node0=`j-p1agegu8nh`、node1=`j-xz8xghpcqh`，两者均为 sleep 604800、8 GPU、hw-check、retry=10，并于 `22:49 +08:00` 进入 Running/retry=0；宿主分别为 `e01-cn-rij4mvgjj0i` 与 `e01-cn-8h74p0pxk0t`。
- 两个 SSH 会话内均验证 Python 3.12.3 和完整 8×NVIDIA L20C（每卡 183359 MiB），随后分别启动 node0/node1 的 `q32_boundary_refine_v5`。两份 node manifest 与四份 `lr=1e-5` spec 已物化；实际 spec 再次证明 Hessian=32、global-loss=1、backward=16、block=256、group=128、a_loss_ratio=1，W4A4KV4 A/K/V4 clip=0.9，其余 A/K/V16 clip=1，act-order/w-clip/rotate 均为 true。
- `23:08 +08:00`，四个 Qwen3-32B `lr=1e-5` worker 均持续运行且无 OOM/failure：W4A16/W3A16 已完成约 56% 的层，W2A16 约 56%，W4A4KV4 约 44%。它们分别固定在 node0 GPU1/2 与 node1 GPU0/1；sleep job 本身仍为 Running/retry=0。
- 为利用同两台 sleep Pod 的其余 12 张卡，已提前按正式协议为 16 个完成选点的非 32B run 生成每模型共享 formal cache。只沿用户授权的 formal `global_loss_bsz=32/16/8/...` 阶梯降档，`backward_samples=32`、`backward_bsz=32`、block=128 和其余量化参数均未改变：Qwen3-0.6B 在 32 OOM 后于 16 成功；Qwen3-4B 在 32/16 OOM 后于 8 成功；Llama-3.1-8B-Instruct 与 Qwen3-8B 均在 32/16 OOM 后于 8 成功。每档日志和最终 `formal_success.json` 已原子留档。
- 上述四个模型的 16 个正式量化已于 `23:07–23:09 +08:00` 在不与 32B worker 冲突的 GPU lane 上启动：node0 使用 GPU0/3–7 运行 Qwen3-0.6B 与 Qwen3-4B，node1 使用 GPU2–7 运行 Llama-3.1-8B-Instruct 与 Qwen3-8B；每个 invocation 仍为单卡 world-size=1。因每节点当前只有 6 张空闲卡，各有两个 run 串接在先完成的 lane 后，避免新增机器和 GPU 空转。
- 四个 Qwen3-32B `lr=1e-5` 候选已全部原子成功、零 OOM：W4A16=`0.0070534423`、W4A4KV4=`0.3779577315`、W3A16=`0.0190259498`、W2A16=`0.0744239688`。四者都严格优于各自 `5e-6`，仍是观测高边界，因而不能提前选择。node0 的 W4/W3 通过 batch `q32_high_extend_node0_v5`（fingerprint=`bc3ec5cfb54b3d324c503a151733f23254b00534440ef93cbbd2f59ee55ec593`）进入 `2e-5`；node1 的 W2/KV4 分别通过单 run batch 在原空闲 GPU1/0 进入 `2e-5`，仍由同一 8-GPU sleep Pod 的 SSH 会话执行。
- Qwen3-0.6B 的共享 formal precompute 在 `global_loss_bsz=16` 成功，但四个完整正式 run 在最终 KL loss 处均因申请约 37.09 GiB OOM；自动降到 8 后仍 OOM，现已按冻结阶梯进入 4 档。每次降档都先生成匹配 cache，`backward_samples/backward_bsz=32`、Hessian=64、block=128 以及其他训练参数保持不变；因此这属于获准的 formal global-loss 容量重试，不是任务级瞬时失败。
- Qwen3-0.6B 四组随后在 formal `global_loss_bsz=4/2/1` 也全部于同一 final-layer KL refresh 表达式 OOM，阶梯已完整耗尽。各档都尝试分配相同的 37.09 GiB 张量；源码路径为 `kl_topk_loss -> tokenwise_kl_from_logits`，形状由冻结的 `backward_bsz=32 × seq_len=2048 × vocab=151936` 决定，而非 `global_loss_bsz`。因此继续降 global-loss 无法处理该 OOM；按用户约束没有降低 backward，也没有修改 KL/top-k/序列长度或其他参数。四个失败 run 均完整保留 16/8/4/2/1 日志，没有 formal success/checkpoint 被误标。
- Qwen3-32B 的 `2e-5` 补点中，W4A16=`0.0093044350`、W3A16=`0.0208995640`，均相对 `1e-5` 明显变差；W2A16=`0.0745936185` 也略差于 `1e-5` 的 `0.0744239688`。三组已分别以 `[5e-6,1e-5,2e-5]` 严格 bracket 原子选择 `lr=1e-5`，trial launch count 均为 5；KV4 的 `2e-5` 仍在运行。
- node0 原 SSH 控制连接被远端关闭时，两个已完成 Qwen3-4B `glbsz=8` OOM 的 run 正好刚进入 4 档，留下只有 command header、没有 `run_logged` footer 的中断 attempt。新增的 orchestration-only 恢复门禁会核对当前 formal config：终态 OOM 日志直接纳入 attempts 后跳到下一档；无 footer 的目录和任何 checkpoint 可恢复地移入 `diagnostics/interrupted_formal_attempts/`，再重跑当前档；每个 run 另加非阻塞 controller lock。未改任何训练参数，定向测试现为 `43 passed`（另有历史 pytest cache 权限 warning）。两个 4 档中断目录已归档，四个 4B formal run 已用 detached SSH worker 在 GPU0/3/6/7 恢复。
- 三个已选 Qwen3-32B run 释放的卡已用于 formal 阶段：node0 GPU1 正在生成该模型共享 formal cache，成功后 GPU1/2 自动接续 W4A16/W3A16 正式量化；node1 GPU1 的 W2A16 watcher 等待同一 cache marker 后接续。KV4 watcher同时等待其 LR 决策和 cache，不会与当前 GPU0 tuning worker 抢卡。
- 随后 Qwen3-4B W3A16 的首个完整 formal 档也在同一 final-layer KL 位置 OOM，证实问题跨设置且不由 LR 是否为 0 决定。为避免 4B/8B/32B 继续把每个 global-loss 档整轮白跑，在不停止两条 sleep job 的前提下，已精确终止现有 formal controller/child 进程及 32B formal 自动接续 watcher；所有对应中断目录保留，恢复门禁会归档后续跑。Qwen3-32B `2e-5` KV4 tuning 与其共享 formal precompute 保持运行，未受影响。恢复正式量化需要明确允许不改变逻辑 `backward_bsz=32` 的 final-KL 投影分块实现，或另行允许降低 backward；当前没有擅自采用任何一种。
- Qwen3-32B W4A4KV4 的 `2e-5` 最终 Exact KL=`0.5778760910`，显著差于 `1e-5` 的 `0.3779577315`；已用同一严格 bracket 选择 `lr=1e-5`。至此 20/20 run 均为 `tuning_complete`，四个 32B run 的选择均为 `1e-5`，主表和完成门禁已更新。
- Qwen3-32B 共享 formal precompute 已完成：`global_loss_bsz=32/16/8/4` 容量 OOM，2 档成功，Hessian accum=32、`backward_samples/backward_bsz=32` 保持冻结；最终 marker 为 `formal_global_loss_bsz=2`。自动正式量化接续已提前解除，因此该 marker 不会越过上述 final-KL blocker 擅自启动 run。
- final-KL 的 37.09 GiB 单次分配已用等价的投影显存 tile 处理：`norm + lm_head + KL` 在展平后的 token 维按固定 512 tokens 分块，并用 non-reentrant checkpoint 丢弃/反向重算大词表中间量。外层仍一次选取并前向 `backward_bsz=32`、共 65,536 tokens，按 token 数加权得到同一个 mean KL，只调用一次梯度裁剪和 Adam；`global_loss_bsz`、Hessian、block/group、LR、KL top-k、序列长度、样本与随机种子均未修改。新增单元测试同时覆盖 full-vocab/top-k 的 loss 和上游梯度等价、shape 门禁；相关 refresh/numerics 测试为 `38 passed`，campaign+tuner 测试为 `43 passed`，仅有历史 pytest cache 权限 warning。
- 真实单卡 smoke 复跑 Qwen3-0.6B W4A16 的末级 `global_loss_bsz=1`，物化 config 审计为 `backward_samples/backward_bsz/final_layer_backward_bsz=32`、Hessian=64、block/group=128、KL top-k=-1；日志确认 `tokens=65536, token_chunk=512`。该 run 越过此前固定 OOM 位置，以 `returncode=0` 完成 28/28 层并写出 1,503,480,592-byte checkpoint；60 秒 size/mtime 稳定性门禁后已原子发布 `formal_success.json`。
- 其余三个 Qwen3-0.6B run 的 glb=1 旧版完整 OOM 目录已可恢复地移入 `diagnostics/pre_kl_projection_chunk_formal/`，此前 16/8/4/2/1 证据均未删除。node0=`j-p1agegu8nh`、node1=`j-xz8xghpcqh` 继续保持 8-GPU sleep 入口；已通过 SSH 在空闲 GPU0/1 直接重跑这三个末级档，同时两侧 formal scheduler 正在排空已经启动的 Llama/Qwen3-4B run。调度器完成当前原子 run 后会按 success marker 断点恢复其余 8B/32B 正式量化。
- 三个末级重试也均以 rc=0 越过 final-KL，并在 checkpoint 60 秒稳定门禁后发布 success；至此 Qwen3-0.6B formal 为 4/4。为避免排空期间卡空转，已取消仅负责 second-pass 的无 GPU watcher，直接在其余 GPU lane 启动尚未覆盖的 Qwen3-4B W3、四个 Qwen3-8B 和四个 Qwen3-32B；连同两侧原调度器正在排空的 7 个 Llama/4B run，formal 一度达到两节点各 8 个 controller/CUDA lane。
- 四个 0.6B marker 发布后，两条更早遗留的 eval lane watcher（node0 PID 11880/11881）提前在 GPU4/5 启动了 W3/W2 的 GSM8K，分别写出 160/96 条可恢复 generation，并与当前 formal lane 发生资源重叠。发现后已精确终止这两个 eval controller/child 及 node1 对应的旧 watcher PID 7106/7107；没有 generation success 被误标，部分输出保留且配置为 `reasoning_resume=true`。新的 formal→eval watcher只在各节点 10/10 formal success 后启动，因此后续会从已写 generation 续跑而不再与量化抢卡。
- Qwen3-4B W3A16 恢复时直接采纳旧 glb=8 的完整 OOM footer，并在 glb=4 重新运行；实际层量化约 4 分钟，随后同样通过 65,536-token final-KL tile，以 rc=0 写出 8,823,939,012-byte checkpoint并通过 60 秒稳定门禁。当前 formal success 为 5/20；其余 15 个 run 都有唯一 controller，Llama/8B/32B 与另外三个 4B 正常推进，且两节点已复核无 reasoning 进程。
- HumanEval+ 官方评分已在 campaign 中闭环：scorer 不再假设容器内不可用的 `.venv/bin/python`，而是 fail-closed 接受已审计 `REALQ_PYTHON`；`score` stage 先核对三项 generation success，再对每组验证 164 个唯一 task 与本地 v0.1.10 数据集完全一致、样本 size/mtime 稳定，最后在现有禁网/降权/资源限制 runner 中执行官方 evaluator并要求结果精确覆盖 164 题、每题一个 candidate、base/plus 状态合法。结果 marker记录样本、generation receipt、官方 result 的 SHA256、字节数和 base/plus pass@1。相关 campaign/tuner/final-KL 回归为 `49 passed`，另有既有 pytest cache 权限 warning；`bash -n` 与 `git diff --check` 均通过。
- 两台 sleep Pod 已各布置一条 CPU-only score watcher（node0 PID 23791、node1 PID 13401）：等待对应 formal→eval 调度器完整退出后才调用 `run-node --stage score`，而 score stage 自身还会再次检查本节点 10×3 个 generation success；每节点最多同时运行 2 个 EvalPlus scorer，不占 GPU。首次试图直接在 watcher shell 展开 30 个文件条件时命令被 SSH 交互网关截断在 continuation prompt，未产生进程或输出；已改为上述短命令加 campaign 内部完整门禁。
- Qwen3-4B 其余三组也均通过 final-KL 并发布稳定 checkpoint：W4A16/KV4 在 glb=4 成功，W2A16 在首个完整 glb=8 档成功；结合先前 W3A16 glb=4，Qwen3-4B formal 为 4/4、campaign 总计 8/20。四个 checkpoint 约 8.824 GB，具体字节数和 job/GPU 已写入主表。两节点当前各剩 6 个 formal controller（Llama/8B/32B），无 reasoning 进程。
- `01:10 +08:00` 再次同时核验两台 sleep Pod：`j-p1agegu8nh`、`j-xz8xghpcqh` 均为 Running/retry=0，入口仍是 8 卡 `sleep 604800`；12 个剩余 formal run 各只有一个 controller。Llama 四组推进到 22–25/32 层，Qwen3-8B 四组均到 18/36，Qwen3-32B 四组均到 5/64；各 CUDA 进程持续占卡，无 OOM/Traceback，无 reasoning 进程，formal→eval 与 eval→score watcher 均存活。
- 新增 `experiments/realq_20group_20260808/audit.py` 作为最终 fail-closed 门禁：固定绑定 v5 plan fingerprint，逐组重建并比对调参/正式/评测 resolved config，验证 selected LR 为成功观测中的 Exact-KL 全局最优、formal OOM 降档顺序、rc=0 日志 footer、final-KL tile 证据、checkpoint stat，以及三项 generation 的题目身份、唯一性、配置 hash、manifest、数学评分明细和 HumanEval+ 官方 164 题结果/hash。只有 20 个 checkpoint、60 个 generation 和 20 个官方评分全部通过才原子写出 `final_audit.json`；4 个定向新测试通过，未完成 campaign 的实机 dry audit 已成功通过 plan、调参与首组 formal 检查，并按预期在缺少 GSM8K success marker 处拒绝完成。
- 最终审计已作为 node0 的 CPU-only watcher PID `24627` 部署，driver=`orchestration/final_audit_after_scores.driver.log`；它只轮询 20 个 `official_success.json`，齐全后才执行上述 audit，不加载模型、不占 GPU。campaign、审计、final-KL 与 tuner 相关回归合计 `53 passed`，仅保留既有 pytest cache 权限 warning。
- `01:29–01:34 +08:00` 的逐卡复核发现 formal 已完成 11/20 后，两台 sleep Pod 只剩 9 个 formal CUDA 进程（node0 5/8、node1 4/8），其余 7 卡因原 formal→eval watcher 要等待本节点全部 10 个 formal marker 而空闲。为避免手工补位和后续正常 scheduler 重复写 generation，`run_evaluation` 增加每 run 的阻塞式 `.eval.lock`；第二个 entrant 只会在锁释放后依据原子 task success marker 跳过/续跑，不改变任何训练或评测参数。定向锁测试、`py_compile` 与 `git diff --check` 均通过。
- 已在 7 张空闲卡启动 reasoning eval 队列，并把已完成的 11 个 run 全部纳入当前/后续串行队列。期间 Llama W4A4KV4 formal 完成使总进度升至 12/20、node0 GPU5 释放；首次补该 lane 时因 SSH 命令漏掉仓库 `cwd`，仅在 module import 前以 `ModuleNotFoundError: experiments` 瞬退，没有加载 checkpoint、占用 GPU 或写评测结果。按标准 sleep-job→SSH 流程补上 `cd /minimax-avatar-new/zhangqian/realq/gptq_plus` 后原命令直接重提，配置未变并进入 GSM8K 生成。
- `01:33:45 +08:00` 最终逐卡 `nvidia-smi`/`pmon` 证据：node0 GPU0–7 与 node1 GPU0–7 均各有唯一 CUDA PID，总计 16/16 卡在工作；组合为 8 个剩余 formal 与 8 个 reasoning eval。node0 瞬时利用率为 `77/99/100/100/100/2/58/55%`，node1 最近一次完整样本为 `100/100/99/100/42/63/54/87%`；生成阶段利用率会随 decode 波动，但每卡显存和 CUDA 进程均已确认存在。两条 Canoe sleep job 仍为 `j-p1agegu8nh` / `j-xz8xghpcqh`，任务内通过 SSH 执行实验。
- 为防止接近完成的 4 个 Qwen3-8B formal 释放 GPU 后再次空转，剩余 8 个 Qwen3-8B/32B run 均新增“等待自身 formal marker→在原 GPU 执行自身 eval”的 detached watcher。原先等待本节点全部 formal 的 eval watcher及依赖其 PID 的 score watcher仅处于 sleep 状态，已核对命令后 TERM，并替换为“全局 60 个 generation marker 齐全→各节点 `run-node --stage score`”的 CPU watcher；不会提前评分。`01:37 +08:00` 两侧各 4 个 formal→eval watcher和 1 个 score watcher均存活，旧 watcher 均已退出；再次抽样两台仍为 8/8 CUDA lane 忙碌。
- `01:40–01:43 +08:00`，Qwen3-8B 四组均完成 36/36 层、bounded final-KL、checkpoint 保存和 60 秒 size/mtime 稳定门禁，正式 invocation 实测分别为 W4A16=`3590.48s`、KV4=`3642.12s`、W3A16=`3585.07s`、W2A16=`3639.91s`，全部 rc=0、formal glb=8、无 OOM/Traceback；checkpoint 字节数已回填主表。formal 总进度升至 16/20，四条 watcher 随 marker 在原 GPU 成功切换到 GSM8K，逐卡复核仍为 16/16 CUDA lane（4 个 Qwen3-32B formal + 12 个 reasoning eval）。
- 同时根据 Qwen3-32B 当前 10–11/64 层的真实单卡速率估算：W4/W3 约 5.3 GPU·h，W2 约 5.4–5.5 GPU·h，KV4 约 6.1–6.2 GPU·h，四组总计约 22–23 GPU·h、四卡并行墙钟约 6.2 小时。该口径另核对了共享 formal precompute：已知成功 glb=2 的 clean attempt 为 `481.54s`，四组共用，不重复计入每个 run。
- `01:52 +08:00`，12 条并行评测均持续写入 GSM8K 可恢复输出，累计 `7,296` 条、单 run 均未超过数据集门禁 `1,319`；最快的 Qwen3-0.6B W3A16 为 `1,248/1,319`。当前尚无完整 task marker，故主表成绩仍保留空值，不能提前填分。两机继续 16/16 卡占用，活跃评测及 Qwen3-32B formal 日志均无 OOM、Traceback 或非零 returncode。
- `01:53:40 +08:00`，首个完整 generation marker 发布：Qwen3-0.6B W3A16 的 GSM8K 精确覆盖 `1,319/1,319`，manifest=`completed`，数据集 SHA256=`752adc99...abafbac`，pass@1=`0.0902198635`（`119/1319`）。该结果已回填主表；同一 controller 随后在原 GPU6 自动进入 MATH-500（batch=16、max_new_tokens=2048），其余 15 卡任务未受影响。
- `01:56 +08:00`，Qwen3-0.6B W4A16 与 W2A16 也分别发布完整 GSM8K marker：W4A16=`0.5701288855`（`752/1319`），W2A16=`0.0022744503`（`3/1319`）；两项 manifest 均为 scored/completed 且各 `1,319` 个唯一 generation。两条 controller 已分别在 node0 GPU0、node1 GPU5 切入 MATH-500，逐卡复核仍无空闲卡。
- `01:57 +08:00`，两侧负载拆分为 node0=`2 formal + 4 GSM8K + 2 MATH-500`、node1=`2 formal + 5 GSM8K + 1 MATH-500`，恰好各 8 个 CUDA lane。三个已完成 GSM8K 的 MATH-500 分别推进到 16/48/16 条；Qwen3-4B W4/W2 的 GSM8K 均为 992/1319。Qwen3-32B W4/KV4/W3/W2 分别推进到 `15/13/15/14` 层，所有活跃日志继续无异常。
- `02:06–02:07 +08:00`，Qwen3-4B W4A16、W2A16 的 GSM8K 均精确完成 `1,319/1,319` 并发布原子 marker；manifest pass@1 分别为 `0.8635329795`（`1139/1319`）与 `0.0098559515`（`13/1319`）。成绩已回填主表，两条 controller 均已创建 MATH-500 输出并在原 GPU 继续。
- `02:11–02:17 +08:00`，再次通过 Canoe API、dead-node 快检及两台容器内连续 5 秒 `nvidia-smi dmon`/`pmon` 交叉核验资源：`j-p1agegu8nh`、`j-xz8xghpcqh` 均为 `Running/retry0`，宿主分别为 `e01-cn-rij4mvgjj0i`、`e01-cn-8h74p0pxk0t`；两台各 8 张卡均有唯一 Python CUDA PID且连续样本 SM 利用率非零，即 `16/16` 卡在工作。两节点 `up{}` 全窗口持续为 1、pod events 无高危 reason；MetaGod 因当前身份无读权限跳过，因此硬件 CMDB 层仍未覆盖。
- 同轮产物核验仍为 formal `16/20`、generation `5/60`、HumanEval+ official `0/20`。Qwen3-32B W4A16/KV4/W3A16/W2A16 分别推进至 `18/17/18/18` 层，单层滚动均值约 `288–294s`，四份活跃日志对 `Traceback|CUDA out of memory|OutOfMemory|RuntimeError` 命中均为 0；其冻结参数仍为 formal `global_loss_bsz=2`、`hessian_accum_bsz=32`、`backward_bsz=32`。评测侧 Qwen3-8B W4A16/W3A16 已到 `1248/1319`，三条 Qwen3-0.6B MATH-500 分别到 `240/288/256`，所有 lane 继续原任务或原队列运行。
- `02:16:50–02:16:54 +08:00`，Qwen3-8B W3A16、W4A16 同时完成 GSM8K 并发布第 6、7 个 generation marker。两份 manifest 均为 `completed/scored`，各精确含 `1,319` 条 generation，数据集 SHA256 均为冻结值 `752adc99...abafbac`；W4A16 pass@1=`0.8885519333`（`1172/1319`），W3A16=`0.8680818802`（`1145/1319`），已回填主表。两条 controller 已在 node1 GPU0/1 原位创建 MATH-500 输出；`02:18 +08:00` 两卡新 Python PID 分别占用约 `21.1/21.2 GiB`、SM=`79/83%`，确认不是只发布 marker 后空转。此时总 generation 进度为 `7/60`，两机仍为 `16/16` CUDA lane。
- `02:19:33 +08:00`，Qwen3-8B W2A16 发布第 8 个 generation marker：GSM8K manifest=`completed/scored`、`1,319/1,319`，pass@1=`0.0197119030`（`26/1319`），同一冻结数据集 hash；成绩已回填主表。原 node0 GPU2 controller 已创建 MATH-500 输出继续运行。同期三条 Qwen3-0.6B MATH-500 已分别到 `320/352/320`，Qwen3-4B W4/W2 到 `112/128`，新进入的 Qwen3-8B W4/W3 各到 `16`。
- `02:24 +08:00`，Qwen3-32B W4A16/W3A16/W2A16 推进至 `20/64`、KV4 至 `19/64`；四条 formal 与全部当前 reasoning `execution.log` 的同轮关键错误扫描总命中为 0。
- `02:31:41–02:33:57 +08:00`，首批三条 Qwen3-0.6B MATH-500 全部发布完整 marker：W4A16=`0.362`（`181/500`）、W3A16=`0.064`（`32/500`）、W2A16=`0.01`（`5/500`）。三份 manifest 均为 `completed/scored`、精确 `500/500`，数据集 SHA256 均为冻结值 `35dc4108...5a06132`，成绩已回填主表。三条 controller 均已创建 HumanEval+ 生成目录并在原 GPU 接力；核验时 W3 已落盘 `16/164`，W4/W2 正在首批生成。总 generation marker 由 `8/60` 升至 `11/60`；official EvalPlus 仍为 `0/20`，按门禁不以生成分数替代官方 base/plus 执行结果。
- `02:35–02:36 +08:00`，再次逐卡核验两台均有 8 个唯一 Python CUDA PID，合计 `16/16`；三条 Qwen3-0.6B HumanEval+ 已推进至 `16/48/16`。同一时段 Llama-3.1-8B-Instruct W4A4KV4 完成 GSM8K 并发布第 12 个 generation marker：manifest=`completed/scored`、`1,319/1,319`、pass@1=`0.6550416983`（`864/1319`），冻结数据集 hash 正确；成绩已回填主表，controller 已在 node0 GPU5 创建 MATH-500 输出继续。
- `02:41:08 +08:00`，Qwen3-0.6B W4A4KV4 完成 GSM8K 并发布第 13 个 generation marker：manifest=`completed/scored`、`1,319/1,319`、pass@1=`0.0075815011`（`10/1319`），冻结数据集 hash 正确；成绩已回填主表。原 node1 GPU4 controller 已创建 MATH-500 输出继续；同刻 Qwen3-0.6B W3A16 HumanEval+ 已到 `128/164`。
- `02:44:51 +08:00`，Qwen3-0.6B W3A16 完成 HumanEval+ 候选生成：manifest=`completed`、结果状态=`generated_unscored`、精确 `164/164`，冻结数据集 SHA256=`272720b9...9f101`，generation marker 明确 `official_code_score_pending=true`。原 node0 GPU6 队列已无缝切换到 Qwen3-4B W3A16 GSM8K，新 PID/显存/SM 均已确认非零。
- 未等待全局 `60/60`，在同一 sleep 容器以 CPU 直接执行该 run 的受审计 `_score`；其 164 题身份/唯一性、样本 30 秒稳定性、官方结果全覆盖及 base/plus 状态门禁全部通过，rc=0。`02:47:42 +08:00` 发布首个 official marker：Qwen3-0.6B W3A16 HumanEval+ base=`0/164`、plus=`0/164`，官方结果覆盖 `164` 个 task、parallel=`32`、elapsed=`38.92s`；成绩已回填主表并将该 run 标为完成。全局后续 scorer 会依据同一 marker 幂等跳过，当前 official 进度为 `1/20`。
- Qwen3-0.6B W4A16/W2A16 随后也完成 `164/164` HumanEval+ 候选，generation 总进度升至 `16/60`。两台机器分别并行执行官方 scorer：W4A16 正常发布 base=`40/164`、plus=`36/164`；W2A16 的官方结果对 7 题返回 `timeout`，原 campaign 审计器因只接受 `pass/fail` 而拒绝 marker。核对 vendored EvalPlus 0.3.1 源码后确认其完整正式枚举为 `pass/fail/timeout`，且官方 pass@k 只把 `pass` 计正确；W2 状态分布为 base/plus 各 `157 fail + 7 timeout`，这是合法非通过结果而不是结果损坏。
- campaign 状态门禁已精确扩为 `pass/fail/timeout`，`timeout` 仍按非通过计数；定向测试加入 timeout fixture 后 `1 passed`，`py_compile` 与 `git diff --check` 通过（仅既有 pytest cache 权限告警）。W2A16 重评分 rc=0 并发布 official marker：base=`0/164`、plus=`0/164`、elapsed=`77.98s`。至 `02:57 +08:00` official=`3/20`，三条 Qwen3-0.6B 非 KV run 已完整闭环；两侧 GPU 队列已分别接上 Llama W2/W3 和 Qwen3-4B W3，逐卡复核仍为 `16/16` 唯一 CUDA PID。
- `02:59:20–03:00:18 +08:00`，Qwen3-4B W2A16/W4A16 的 MATH-500 与 W4A4KV4 的 GSM8K 连续发布第 17–19 个 generation marker。三份 manifest 均为 `completed/scored` 且行数精确命中冻结全集：W2 MATH=`0.02`（`10/500`）、W4 MATH=`0.558`（`279/500`）、KV4 GSM=`0.4056103108`（`535/1319`），数据集 hash 均正确；成绩已回填主表。W2/W4 已在原 GPU 创建 HumanEval+ 输出，KV4 已创建 MATH-500 输出，核验时均处于首批生成。
- `03:04–03:07 +08:00`，再次按 Canoe 排障门禁复核两条 sleep 作业：`j-p1agegu8nh`、`j-xz8xghpcqh` 均为 `Running/retry=0`，入口仍分别在 8 卡 Pod 中执行 `sleep 604800`；`02:11–03:05` 的节点 `up{}` 持续为 1，Pod events 无高危 reason，MetaGod 仍因读权限不足跳过。容器内逐卡 `nvidia-smi`/`pmon` 证据显示两机各 8 个唯一 Python CUDA PID，16 张卡瞬时 SM 均非零（node0=`87/49/89/99/99/36/86/86%`，node1=`87/50/97/99/41/86/47/66%`）。Qwen3-32B W4A16/W3A16/W2A16 均到 `29/64`，KV4 到 `28/64`，四份日志错误扫描仍为 0；按当前滚动速率剩余墙钟约 `2.8–2.9h`。当前原子产物计数保持 formal=`16/20`、generation=`19/60`、official=`3/20`；12 条活跃评测均继续推进，其中 Qwen3-8B KV4 GSM=`1248/1319`、三条 Qwen3-8B MATH=`448–480/500`、Llama W2/W3 GSM=`896/1088`，未发现空卡。
- `03:11 +08:00`，将原先“全局 60/60 generation 后再评分”的两条 CPU-only score watcher，安全替换为按节点各自 10 个 run 逐项轮询并即时执行 `_score` 的 watcher：node0 PID=`39844`、node1 PID=`26882`，driver 分别为 `orchestration/immediate_score_node0.driver.log` 与 `immediate_score_node1.driver.log`。替换前精确核对并仅 TERM 旧 watcher PID `27390/16686`；新 watcher 对已有 official marker 幂等跳过、每节点至多一个 scorer 串行运行、失败会留日志并重试，不占 GPU、不改变 generation 或官方评分协议，也消除了最后一个 run 与全局 scorer 争抢 `.official.lock` 的竞态。
- `03:10:29–03:13:15 +08:00`，连续发布 5 个新 generation marker，总进度升至 `24/60`。逐项核对 manifest=`completed`、result=`scored`、精确唯一行数与冻结数据集 hash 后，结果为：Qwen3-8B W4A16/W3A16/W2A16 的 MATH-500 分别 `56.80% (284/500)`、`55.00% (275/500)`、`3.80% (19/500)`；Qwen3-8B W4A4KV4 的 GSM8K=`59.14% (780/1319)`；Llama-3.1-8B-Instruct W3A16 的 GSM8K=`73.16% (965/1319)`。五条 controller 均已在原 GPU 原位切入下一任务（前三条非 KV Qwen3-8B 进入 HumanEval+，KV4 与 Llama W3 进入 MATH-500），主表已回填。
- `03:14:56–03:14:57 +08:00`，在上述五次任务切换后再次逐卡取样：两台各有 8 个唯一 Python CUDA PID，16 张卡显存均非零且瞬时 SM 全部非零；node0=`81/55/83/99/99/51/64/87%`，node1=`73/87/99/99/33/90/47/64%`。因此切换没有造成 lane 丢失或空卡。
- `03:16:43 +08:00`，Llama-3.1-8B-Instruct W2A16 发布完整 GSM8K marker，generation 总进度=`25/60`。该项 manifest=`completed`、result=`scored`，冻结数据集 hash 正确且精确覆盖 `1,319/1,319`，pass@1=`0.00% (0/1319)`；这是审计通过的真实低分而非缺失结果，已回填主表。controller 已在原 node0 GPU0 进入 MATH-500。
- `03:17:42–03:20:21 +08:00`，Qwen3-4B W2A16/W4A16 先后完成 HumanEval+ 候选生成，均通过 manifest、冻结数据集 hash、164 个唯一 task 与 `official_code_score_pending=true` 门禁，generation 总进度=`27/60`。新即时 watcher 分别在 node1/node0 自动启动 CPU-only 官方评分并均 rc=0：W2 base/plus=`0/164, 0/164`（elapsed=`37.69s`）；W4 base=`73/164`、plus=`70/164`，即 `44.51% / 42.68%`（elapsed=`40.39s`）。官方结果各覆盖 164 task 且 hash/字节数已写入 marker，official 总进度=`5/20`；两组现均完整闭环，主表状态改为完成。
- `03:21:28 +08:00`，Qwen3-4B W3A16 发布完整 GSM8K marker：manifest=`completed`、result=`scored`、冻结 hash 正确且 `1,319/1,319`，pass@1=`76.50% (1009/1319)`；generation 总进度=`28/60`，成绩已回填，controller 已转入 MATH-500。
- 由于两条 Qwen3-4B HumanEval+ 完成后只剩 15 个 run 级 controller 可运行，新增严格输出目录级 `_eval-task` 尾部调度入口：只允许冻结的三项任务，每个 task 使用独立阻塞式 `.generation.lock`、原 `eval_args`、原 audit 和同一原子 success marker；正常 `_eval` 也复用该单 task 实现。`py_compile`、3 个定向测试和 `git diff --check` 均通过，仅有既有 pytest cache 权限告警；没有修改任何模型、量化或 decoding 配置。
- 为避免旧进程稍后与 task 级 worker 重复写入，先精确核对并仅 TERM Qwen3-0.6B KV4 的旧串行 MATH child/controller/wrapper PID `22625/14557/14502`；三者均退出，未发布伪 marker，已有 `208/500` 行可恢复输出完整保留。随后在 node1 GPU4 以新入口从该断点续跑 MATH-500（PID `29299`），并在空闲 node0 GPU7 并行启动同一 immutable checkpoint 的独立 HumanEval+（PID `42910`）；此前未启动的 Llama W4 已补到 node1 GPU7（PID `28563`）。`03:27:42 +08:00` 两台逐卡 `pmon` 再次显示各 8 个唯一 Python CUDA PID，16/16 张卡显存和瞬时 SM 均非零。
- `03:29:24–03:32:59 +08:00`，Qwen3-8B W4A16/W3A16/W2A16 三组 HumanEval+ 先后生成并官方评分完成。三组 generation 均为冻结 hash 下的 `164/164` 唯一 task，generation 总进度=`31/60`；即时 watcher 串行发布 official marker：W4 base/plus=`79/164,75/164`（`48.17%/45.73%`，elapsed=`37.84s`），W3=`55/164,52/164`（`33.54%/31.71%`，elapsed=`38.26s`），W2=`0/164,0/164`（elapsed=`37.91s`）。每份官方 result 均精确覆盖 164 task 并记录 hash/字节数，official 总进度=`8/20`；三组主表状态均改为完成。
- `03:32 +08:00`，Qwen3-32B 四条正式量化仍无错误，W4/W3/W2=`34/64`、KV4=`33/64`；按日志实时 ETA 尚余约 `2.4–2.5h`。formal 总进度仍为 `16/20`。
- 三个 Qwen3-8B 非 KV run 完成后释放的 lane 已继续用于 task 级尾部并行。Qwen3-8B KV4 旧串行 MATH child/controller PID `39919/27386` 在精确核对命令后 TERM，保留 `80/500` 行且无伪 marker；随后 node0 GPU1 从断点续跑其 MATH、GPU2 并行运行其 HumanEval+（PID `44464/44465`）。Llama W4 的新 evaluator 本身已使用 task lock，因此无需中断其 GPU7 GSM8K，直接在 node1 GPU0/GPU1 并行启动该 run 的 MATH/HumanEval+（PID `30930/30931`）；其原 controller 日后会等待对应 task lock并依据 marker 幂等跳过。
- `03:36:15 +08:00`，上述四个补位 worker 均已真实加载 checkpoint。两台机器再次各有 8 个唯一 Python CUDA PID，16/16 卡显存与瞬时 SM 全部非零：node0=`90/33/35/99/99/38/80/46%`，node1=`79/60/99/99/38/57/48/85%`。task 级改动的完整 campaign 测试文件最终为 `26 passed`（`268.37s`），仅有既有 pytest cache 权限 warning。
- `03:37:40–03:39:14 +08:00`，Llama-3.1-8B-Instruct W4A16 的 HumanEval+ 生成和官方评分依次完成：冻结数据集下精确生成 `164/164` 个唯一 task，官方 base=`101/164`（`61.59%`）、plus=`97/164`（`59.15%`），官方结果文件为 `77,080` 字节、SHA256=`2a4f8010...688b182`，scorer elapsed=`37.60s`。generation 总进度升至 `32/60`，official 升至 `9/20`；主表已回填，但该 run 的 GSM8K/MATH-500 仍在运行，故状态保持“评测中”。
- `03:39–03:40 +08:00`，按 Canoe 标准排障顺序重新核验两条 sleep job：`j-p1agegu8nh`、`j-xz8xghpcqh` 均为 `Running/retry=0`，入口仍是 `sleep 604800`，Pod 分别位于 `e01-cn-rij4mvgjj0i`、`e01-cn-8h74p0pxk0t`。从两 job 的 Running 起点至当前，节点 `up{}` 持续为 1、Pod events 无高危 reason；MetaGod 仍因当前身份无读权限跳过。Llama W4A16 HumanEval+ 完成后 node1 GPU1 短暂释放，抽样为 `15/16` 个 CUDA lane。
- `03:42–03:43 +08:00`，在核对 Qwen3-4B W4A4KV4 的 MATH-500 仍只有 `144/500`、其 HumanEval+ 尚未启动且不存在同 task writer 后，直接用新的 `_eval-task` 在 node1 GPU1 启动 HumanEval+（controller PID=`32630`、CUDA PID=`32632`）；随后 `pmon` 确认该卡 SM=`33%`，两机恢复 `16/16` CUDA lane。同期 Qwen3-32B W4A16/KV4/W3A16/W2A16 分别到 `36/35/37/36` 层，日志 ETA 分别尚余约 `2:16/2:20/2:09/2:17`，四份日志仍无 OOM 或非零退出。
- `03:48–03:49 +08:00`，为消除 task 拆分与旧串行 evaluator 的后续竞态，精确核对并 TERM Qwen3-4B W3A16、W4A4KV4 的旧 MATH child/controller/wrapper；分别完整保留 `240/500`、`176/500` 行且无 success marker，随后在原 node0/node1 GPU6 以 `_eval-task math_500` 从断点重启。两条新 CUDA PID=`45935/34014` 均已加载 checkpoint，16 卡占用未丢失。
- `03:49 +08:00`，Llama-3.1-8B-Instruct W4A16 的 GSM8K 发布完整 marker：manifest=`completed/scored`、冻结数据集下 `1,319/1,319`，pass@1=`83.32% (1099/1319)`；generation 总进度升至 `33/60`，主表已回填。
- 同一 run 的正常 evaluator 随后在 node1 GPU7 进入 MATH-500，而此前拆出的 GPU0 task worker仍在写同一输出，实机证明共享 CPFS 上的 `flock` 即使在同一 Pod 也不能提供本 campaign 所需的排他语义。较晚的 GPU7 writer 在约 140 秒时被精确终止；检查发现其已经产生一个 16 题重叠 batch，污染文件最终为 `224` 行但仅 `208` 个唯一 `sample_id`，不能继续采纳。
- `03:54–03:55 +08:00`，停止剩余 Llama W4 MATH writer后，将整个污染 task 目录原样、可恢复地移动到 `diagnostics/duplicate_generation_writer/llama31-8b-instruct_w4a16/math_500_20260808T195439Z/`，没有手工删行或挑选 generation；随后在 node1 GPU0 从空 task 目录按同一冻结配置重跑。campaign 的 per-task 排他已从 `flock` 改为共享文件系统原子 `mkdir(.generation.claim)`：owner 记录 host/PID/token，重复 entrant 只等待 success，死亡 owner 不自动抢占而是超时失败。定向测试 `1 passed`、`py_compile` 和 `git diff --check` 均通过；实机新 worker PID=`35421` 已写出 owner claim，CUDA PID=`35423` 在算，两机再次为 `16/16`。
- `03:55 +08:00`，Llama-3.1-8B-Instruct W3A16 的 MATH-500 发布完整 marker：manifest=`completed/scored`、`500/500`，pass@1=`25.20% (126/500)`；generation 总进度升至 `34/60`。原串行 controller随后在 node1 GPU5 唯一进入 HumanEval+，新 CUDA PID=`35911` 已确认在算，未与任何拆分 worker 重叠。
- 已为 Qwen3-32B 的 8 个附加 task 预置尾部 watcher：每组 formal 原卡上的既有 watcher负责 GSM8K，另外 8 条分别等待指定非 32B 当前 controller 退出、对应 `formal_success.json` 出现并再次确认目标 GPU 空闲后，运行该组 MATH-500/HumanEval+。node0 watcher PID=`46740–46743`，node1=`36558–36561`；实际启动时会加载新的原子目录 claim，若正常 evaluator 已先完成 task 则依据 marker 幂等退出。该安排不占等待期间 GPU，也不改变任何评测参数。
- `03:59 +08:00`，两机逐卡 `pmon` 再次为 `16/16` 唯一 CUDA PID；campaign 原子计数 formal=`16/20`、generation=`34/60`、official=`9/20`。Qwen3-32B W4A16/KV4/W3A16/W2A16 分别推进到 `40/38/40/39` 层，日志剩余 ETA 约 `1:56/2:05/1:55/2:02`，四卡持续高利用率。
- `04:01–04:02 +08:00`，为在 Qwen3-0.6B KV4 HumanEval+ 完成释放 node0 GPU7 后立即补位，将仍处旧串行实现的 Llama W4A4KV4 MATH 精确停止于 `352/500`、移除其 run-level controller，并在原 GPU5 以新 claim 的 `_eval-task math_500` 断点续跑（controller/CUDA PID=`47529/47543`）。另预置 PID=`47530` 等待 GPU7 当前 controller `42910` 退出后在该卡启动同一 Llama checkpoint 的 HumanEval+；MATH 与 HumanEval+ 输出目录独立，均不改变评测配置。
- `04:05–04:06 +08:00`，再次按 Canoe 标准顺序检查两条 sleep job：`j-p1agegu8nh`、`j-xz8xghpcqh` 均为 `Running/retry=0`，Pod Ready，完整 Running 窗口内节点 `up{}` 持续为 1、无高危 pod event；MetaGod 因当前身份无读权限仍未覆盖。容器内逐卡 `pmon` 显示 node0 为 `8/8` CUDA lane，node1 为 `7/8`，空闲的是 GPU5。此时全部 11 个依赖已满足且未完成的非 32B generation task 均已有唯一 writer，另 4 卡运行 Qwen3-32B formal；剩余 12 个 Qwen3-32B generation task 均被未完成 checkpoint 硬门禁阻塞，因此安全最大并行度暂为 `15/16`，没有为了表面满卡而启动重复 writer。
- `04:06–04:13 +08:00`，Qwen3-0.6B KV4、Llama W4A4KV4、Llama W3A16、Qwen3-4B W3A16 的 HumanEval+ 候选与官方评分陆续完成；base/plus 分别为 `0/0`、`62/54`、`59/52`、`43/41`（分母均为 164）。Llama W2A16 的 MATH-500 也完成为 `0/500`。原子进度由 formal=`16/20`、generation=`34/60`、official=`9/20` 更新到 `16/20`、`39/60`、`13/20`，相关成绩已回填主表。
- 同轮部分完成审计发现最终 `audit.py` 仍把 EvalPlus `timeout` 误判为非法，而正式 scorer 已按 vendored EvalPlus 0.3.1 的 `pass/fail/timeout` 完整终态集合正确处理。审计器现与 scorer 共用冻结状态集合，`timeout` 只计为 non-pass；新增回归测试后定向测试 `2 passed`、`py_compile` 通过。随后以 root 只读权限逐项重验当时已发布的 16 个 formal 和 39 个 generation marker，输出 `PARTIAL_AUDIT_OK formal=16 generation=39`；未改写任何已有结果。
- `04:03–04:06 +08:00` 的 Qwen3-32B 实时层进度为 W4A16=`42/64`、W4A4KV4=`40/64`、W3A16=`42/64`、W2A16=`41/64`，滚动 ETA 仍约 `1.8–2.0h`。四张卡的显存/SM 活动正常，日志对 OOM、Traceback、exception 和非零失败均无命中；冻结参数保持 formal `global_loss_bsz=2`、`hessian_accum_bsz=32`、`backward_bsz=32`。
- `04:17–04:18 +08:00`，Llama W2A16 HumanEval+ 官方评分发布 base/plus=`0/164,0/164`，Qwen3-4B W3A16 MATH-500 发布 `50.20% (251/500)`；两组至此均三项闭环并回填主表。完整 campaign 回归测试为 `27 passed`（仅既有 pytest cache 权限 warning），`git diff --check` 通过；原子进度更新为 formal=`16/20`、generation=`41/60`、official=`14/20`。
- `04:20–04:21 +08:00`，Qwen3-8B W4A4KV4 HumanEval+ 完成 `164/164` 唯一候选，node0 即时 scorer 自动执行并发布 official marker：base=`2/164`（`1.22%`）、plus=`0/164`，elapsed=`38.33s`。原子进度更新为 formal=`16/20`、generation=`42/60`、official=`15/20`；该组 MATH-500 仍由唯一 writer 继续。
- `04:28–04:30 +08:00`，Qwen3-4B W4A4KV4 HumanEval+ 完成 `164/164` 唯一候选，node1 即时 scorer 自动发布 official marker：base=`1/164`（`0.61%`）、plus=`0/164`，elapsed=`38.60s`。原子进度更新为 formal=`16/20`、generation=`43/60`、official=`16/20`；其 MATH-500 仍继续运行。
- `04:30:30 +08:00`，Qwen3-0.6B W4A4KV4 的 MATH-500 发布完整 marker：`0.80% (4/500)`；该组至此三项与官方评分全部闭环，generation=`44/60`，主表状态改为完成。
- `04:37:59 +08:00`，Llama W4A4KV4 的 MATH-500 发布完整 marker：`21.00% (105/500)`；该组至此三项与官方评分全部闭环，generation=`45/60`，主表状态改为完成。
- `04:39:55 +08:00`，Llama W4A16 的 clean-restart MATH-500 发布完整 marker：`38.80% (194/500)`；该组至此三项与官方评分全部闭环，generation=`46/60`，主表状态改为完成。随后部分完成审计再次通过：`PARTIAL_AUDIT_OK formal=16 generation=45`（该审计启动时新 marker 尚未发布）。
- `04:51 +08:00`，两条 Canoe job 再次为 `Running/retry=0`；`04:05–04:51` 的 dead-node 快检仍显示节点 `up{}` 持续为 1、无高危 pod event，MetaGod 仍因无读权限跳过。此时非 32B 只剩 Qwen3-4B/8B 两个 KV4 MATH 唯一 writer，另有 4 个 Qwen3-32B formal，逐卡实际为 `7/16` CUDA 进程；其余 9 卡没有依赖已满足的未启动任务。node1 的平台 `job_hang=true` 是 sleep job 在大量 lane 完成后的低平均利用率粗筛信号，与两张 Qwen3-32B formal 卡持续 `98–99% SM`、节点存活和日志前进证据不矛盾。
- `05:15:22 +08:00`，Qwen3-4B W4A4KV4 的 MATH-500 发布完整 marker：`20.60% (103/500)`；该组至此三项与官方评分全部闭环，generation=`47/60`，主表状态改为完成。非 32B 现仅剩 Qwen3-8B W4A4KV4 的 MATH-500（当时 `464/500`）。
- `05:25:30 +08:00`，Qwen3-8B W4A4KV4 的 MATH-500 发布完整 marker：`24.60% (123/500)`；该组至此三项与官方评分全部闭环。前 16 个非 32B run 现全部完成，原子进度为 formal=`16/20`、generation=`48/60`、official=`16/20`；剩余项仅为 4 个 Qwen3-32B formal 及其 12 个 generation/4 个 official。
- `05:57:56–06:06:17 +08:00`，Qwen3-32B W3A16、W4A16、W2A16、W4A4KV4 依次完成 checkpoint 保存、60 秒稳定性门禁并发布 formal marker；四组均为 glb=`2`、Hessian accum=`32`、backward bsz=`32`、单 attempt `rc=0`，且逐个通过 `audit_formal`。真实单卡 GPU time 分别为 W3=`5.2834h`、W4=`5.3148h`、W2=`5.4117h`、KV4=`5.4224h`，合计 `21.4324 GPU·h`；此前基于中途层速率给出的 `22–23 GPU·h` 是保守估计。formal 计数至此达到 `20/20`。
- 四个 formal marker 发布后，原卡上的 4 条 GSM8K evaluator 与预置的 8 条 MATH/HumanEval+ watcher 全部启动。`06:06–06:09 +08:00` 实机核对 node0 六条 task 均已进入 CUDA 计算；node1 六条进程均已创建，其中 W2 三卡进入 CUDA 计算、KV4 三卡仍处 65.5GB checkpoint 冷加载阶段并已建立 CUDA context。当前全部 12 个 Qwen3-32B generation task 均有唯一 controller/claim；每机剩余 GPU6/7 没有额外独立 task，因此安全最大并行度为 `12/16`。
- `06:43–06:47 +08:00`，Qwen3-32B W3A16、W4A16 的 HumanEval+ 先后完成 `164/164` 唯一候选并由 node0 即时 scorer 串行完成官方评分。逐项 `audit_evaluation` 通过；W4 base/plus=`74/164,71/164`（`45.12%/43.29%`），W3=`75/164,72/164`（`45.73%/43.90%`）。全局原子进度更新为 formal=`20/20`、generation=`50/60`、official=`18/20`；两组的 GSM8K/MATH 仍并行运行。
- `06:51–06:53 +08:00`，Qwen3-32B W2A16 HumanEval+ 完成 `164/164` 并通过官方评分与 `audit_evaluation`：base/plus=`0/164,0/164`。全局原子进度更新为 formal=`20/20`、generation=`51/60`、official=`19/20`；仅 KV4 HumanEval+ 及四组 GSM8K/MATH 尚未闭环。
- `07:25–07:27 +08:00`，Qwen3-32B W3A16、W4A16 的 GSM8K 先后发布完整 marker并逐项通过 `audit_evaluation`：W3=`92.87% (1225/1319)`，W4=`93.48% (1233/1319)`。全局 generation=`53/60`；对应原卡 evaluator 后续遇到各自 MATH 的既有原子 claim，会等待尾部 worker 完成而不会重复写入。
- `07:32–07:34 +08:00`，Qwen3-32B W4A4KV4 的 HumanEval+ 完成 `164/164` 唯一候选并通过官方评分与 `audit_evaluation`：base=`38.41% (63/164)`、plus=`36.59% (60/164)`。至此四个 32B run 的官方评分全部闭环，official=`20/20`，generation=`54/60`。
- `07:34–07:36 +08:00`，Qwen3-32B W2A16 的 GSM8K 发布完整 marker并通过 `audit_evaluation`：`30.33% (400/1319)`。全局原子进度为 formal=`20/20`、generation=`55/60`、official=`20/20`；剩余恰为 W4/W3/W2 的 MATH-500 与 KV4 的 GSM8K/MATH-500，实时逐卡进程也对应 node0 两卡、node1 三卡，无遗漏或重复 writer。
- `07:40–07:44 +08:00`，两条 sleep job 继续为 `Running/retry=0`；`06:40–07:40` 的 dead-node 快检确认两节点 `up{}` 持续为 1、无高危 pod event，MetaGod 仍因权限不足跳过。5 个剩余任务的 5 个 CUDA 进程连续采样 SM=`69–97%`，活跃日志错误扫描为 0。对当前全部 20 个 formal 和 55 个 generation marker 重新执行 fail-closed 审计，结果为 `PARTIAL_AUDIT_OK formal=20 generation=55`。原 final-audit watcher 仅以 20 个 official marker 为启动门禁，因此提前启动并按预期拒绝缺失的 W4 MATH marker；未写出 `final_audit.json`，待 generation=`60/60` 后由主流程重新执行完整审计。
- `07:44–07:46 +08:00`，新增轻量 `watch_completion.py`，门禁精确要求 formal=`20`、generation=`60`、official=`20` 三类原子 marker 同时齐全，才调用同一 `audit_campaign`；计数门禁定向测试 `1 passed`，`py_compile` 与 `git diff --check` 通过（仅既有 pytest cache 权限 warning）。按 AGENT.md 的 sleep-job→SSH 方式在 node0 启动 CPU-only watcher PID=`59530`，首条 receipt 为 `20/55/20`，driver=`orchestration/final_audit_after_all_generation.driver.log`；不加载模型、不占 GPU。
- `07:47–07:50 +08:00`，包含 completion watcher 在内的完整 campaign 回归为 `28 passed`（`119.57s`），仅有既有 `.pytest_cache` 权限 warning；宿主机同时复核无 `realq.ptq`、generation 或 EvalPlus 高负载进程。测试期间 5 条 GPU generation 均继续前进，最新为 W4/W3 MATH=`416/500`、W2 MATH=`384/500`、KV4 GSM=`864/1319`、KV4 MATH=`224/500`。
- `08:09:02–08:09:33 +08:00`，Qwen3-32B W4A16、W3A16 的 MATH-500 先后发布完整 marker，并分别通过题目身份、500 个唯一 generation、冻结 config/manifest、明细评分和成功日志 footer 的 `audit_evaluation`。W4=`61.40% (307/500)`，W3=`57.00% (285/500)`；全局 generation 从 `55/60` 依次升至 `57/60`，completion watcher 正确记录两次计数变化。这两组现均三项闭环并在总表标记完成。
- `08:16:34 +08:00`，Qwen3-32B W2A16 的 MATH-500 发布完整 marker并通过同一 fail-closed 逐项审计：`8.60% (43/500)`，500 个题目身份与评分明细完整，generation hash/manifest/config/日志 footer 均匹配。全局 generation=`58/60`，该组也已三项闭环；仅剩 W4A4KV4 的 GSM8K 与 MATH-500 两项。
- `08:44:08 +08:00`，Qwen3-32B W4A4KV4 的 GSM8K 完成 `1,319/1,319` 并通过逐项审计：`92.49% (1220/1319)`，题目集合、唯一 generation、冻结 decoding/config、manifest、评分明细与成功 footer 全部一致。全局 generation=`59/60`，completion watcher 正确接收该计数；campaign 仅剩同组 MATH-500。
- `09:53:50 +08:00`，Qwen3-32B W4A4KV4 的 MATH-500 完成 `500/500` 并通过逐项 `audit_evaluation`：`59.00% (295/500)`，生成耗时 `13,404.35s`；题目身份、唯一 generation、冻结 config/manifest、评分明细、文件 hash 与 rc=0 日志 footer 全部一致。至此 generation=`60/60`，20 组的三项评测全部闭环。
- `09:54:02 +08:00`，completion watcher 在精确观察到 formal=`20`、generation=`60`、official=`20` 后自动执行全量 fail-closed 审计并原子写出 `experiment_data/realq_20group_20260808/final_audit.json`。最终状态=`complete`，计数为 runs=`20`、formal=`20`、generation=`60`、official EvalPlus=`20`；冻结 plan fingerprint=`e187159ad237ac40d06b45aa3271b91ed104112d495d944d7374fdec2732f538`，audit fingerprint=`97855ba3f73e4136d9876d89eccf3b9e180a57ff59a4beebcd42bddd330e8d6a`。
- `09:55–09:56 +08:00`，最终资源复核确认 `j-p1agegu8nh` 与 `j-xz8xghpcqh` 仍为 `Running/retry=0`，当前 retry 的 pod 均为 Ready/Running；`08:40–10:00` dead-node 快检中两节点 `up{}` 持续为 1、无高危 pod event（MetaGod 因当前身份无读权限仍未覆盖）。两台实验机连续两次 `pmon` 均为 8/8 卡无 CUDA 进程，容器与宿主机均无 RealQ、generation、EvalPlus 或 watcher 高负载进程；所有实验 writer 已退出。

### 2026-08-09：checkpoint PPL/论文十项 QA 与 BF16 基线补测

- 新增冻结质量计划 `quality_plan.json`，fingerprint=`7322f455451ae3e5378fca2561bb4046f08eab0f66d0c636b7182f0eb9f4496f`，绑定原 campaign plan/audit fingerprint、20 个 checkpoint 的 path/size/mtime、formal marker hash、完整 resolved config、lm-eval=`0.4.4` 与论文十项任务顺序。每个 checkpoint 单次加载后依次计算 WikiText2 全词表 PPL/KL 和十项 QA，强制 `require_reference_cache_hit=true`；独立原子 claim、attempt、result/success hash 与 `quality_final_audit.json` 防止重复 writer 或不完整表格。
- 新增去重 BF16 计划 `bf16_plan.json`，fingerprint=`8f2a8d0d42cb09d8d9526110fa8ba2b6e6cb1ca92c5b932160a9623e76293edb`。5 个唯一 BF16 基础模型各运行一次 PPL/十项 QA，并各运行 GSM8K、MATH-500、HumanEval+；HumanEval+ 候选继续进入官方 EvalPlus。计划显式记录 5→20 行映射，完整工作量为 20 个 BF16 子任务而不是重复测 20 份相同模型。
- 代码与冻结配置门禁通过：新增定向测试连同原 campaign/audit 回归为 `25 passed`，`py_compile` 和 `git diff --check` 通过。高负载只在 Canoe sleep Pod 内运行，宿主机只做计划、状态与文件审计。
- 启动前 `j-p1agegu8nh` 的 sleep Pod 于 `13:09 +08:00` 自动进入 retry1，新 Pod 仍调度到 `e01-cn-rij4mvgjj0i` 并 Ready/Running；`j-xz8xghpcqh` 保持 Running/retry0、节点 `e01-cn-8h74p0pxk0t`。`13:00–13:25` dead-node 快检确认两节点 `up{}` 持续为 1，当前 Pod events 均无 NodeNotReady/OOM/Evicted 等高危 reason；重启发生在本轮启动前，没有中断新增结果。
- `13:20–13:22 +08:00`，按 sleep-job→SSH 标准流程分别启动 node0/node1 统一调度器与 node0 CPU watcher。首波严格每节点 8 个 checkpoint worker：两台均为 2×Qwen3-32B、2×Qwen3-8B、2×Qwen3-4B、2×Llama-3.1-8B，随后自动回填各自剩余 2 个 Qwen3-0.6B checkpoint 和 10 个 BF16 子任务。两台 `pmon` 均显示 GPU0–7 各有唯一 Python CUDA PID，即 16/16 lane 已占用；初始样本处于模型/checkpoint 冷加载，尚无成功 marker，表格不提前填写。
- `13:37–14:58 +08:00`，量化 checkpoint 质量评测最终达到 `20/20`。BF16 首批两个 quality worker 因 BF16 dtype 对应的 reference-logit cache 尚未生成而 fail-closed，错误为 `Required reference-logit cache hit was unavailable`，不是 OOM；随后 5 个唯一模型的 BF16 reference producer 全部 rc=0。首次手工补填 9 个推理 worker 又在模型加载前因内部默认 `.venv/bin/python` 已失效而立即退出，未产生 GPU 计算或结果；恢复入口显式绑定审计过的 Python 3.12 环境，并增加 claim/GPU-aware 调度，保留失败证据后从新 attempt 恢复。
- `14:58–15:41 +08:00`，恢复 driver 在不打断既有 Qwen3-32B 推理 claim 的前提下把两节点重新填至 15/16 lane；唯一暂空卡由冻结 node 分配造成，node0 lane 释放后自动回填最后一项。恢复后未产生新的 failure，BF16 进度从 `4/20` 连续闭合到 `20/20`。
- `15:41–15:51 +08:00`，CPU watcher 在精确观察到量化质量=`20/20`、BF16 work=`20/20` 后执行双重 fail-closed audit。`quality_final_audit.json` 状态=`complete`，覆盖 PPL=`20`、论文 QA=`200`，fingerprint=`513a8a0c01435966398bfaf5627e83c85f63fe6ba84918b9ecc1e880da05f116`；`bf16_final_audit.json` 状态=`complete`，覆盖 5 个唯一模型、20 行映射、PPL=`5`、QA=`50`、推理=`15`、官方 EvalPlus=`5`，fingerprint=`09e5e9fe3cbb215399db454cd3cb593ca178a3a883e8a06a34cf0ef5a6028e25`。人工复核两台 Job/Pod 均为 Running，节点 `up{}` 持续为 1、无高危 Pod event；Canoe 的 `job_hang=true` 来自 sleep-job 入口而非实验进程挂死。
