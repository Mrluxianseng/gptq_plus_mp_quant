# ResComp Llama 3.1 / Qwen 3 二十组量化与评测记录（2026-08-09）

## 1. 任务与当前门禁

本 campaign 使用 **ResComp core + controlled REAL-Q backend**，覆盖 5 个模型 × 4 个量化设定共 20 组。每组正式量化必须是单进程、单卡，保存版本化 fake-quant checkpoint，并分别完成 WikiText-2 full-vocabulary KL/PPL、论文十项 0-shot QA，以及 GSM8K、MATH-500、HumanEval+ 三项推理评测。

代码继续隔离在 `rescomp_controlled/`；campaign 调度代码计划放在 `experiments/rescomp_controlled_20group_20260809/`。不修改 `realq/`、`gptq_utils/`、`utils/` 或官方 `resComp/` checkout。正式产物根目录冻结为：

`/minimax-avatar-new/zhangqian/realq/experiment_data/rescomp_controlled_20group_20260809`

当前状态：**20/20 正式量化、20/20 质量评测、60/60 推理 generation 和 20/20 官方 HumanEval+ 评分均已完成并通过两次全量审计**。最终 audit fingerprint=`5814da6be832cec0b6b373ff9fbd1295d6062c8fbfbee8f7f24ad06c87d845ce`，对应 plan fingerprint=`670d13cc6ef4884098fb2f34ed99b1c238d0019fe4fce73ae00d05d30fba73cb`。20 个稳定 checkpoint 与全部日志/评测产物保留在共享盘；8 卡 Canoe debug 任务 `j-m1krfnr90y` 已按用户要求安全停止并只读复核为 `Aborted`。用户确认的 ResComp P/R 预计算始终使用官方 `full` 实现，见第 3 节。

## 2. 已冻结配置

| 配置项 | 冻结值 | 依据/门禁 |
|---|---|---|
| 方法标签 | ResComp core + controlled REAL-Q backend | 不声称 native ResComp 官方脚本复现 |
| 模型 | Qwen3-0.6B、Llama-3.1-8B-Instruct、Qwen3-4B、Qwen3-8B、Qwen3-32B | 均为本地 dense checkpoint；架构只允许 `LlamaForCausalLM` / `Qwen3ForCausalLM` |
| 量化组合 | W4A16KV16、W4A4KV4、W3A16KV16、W2A16KV16 | 仅允许这四个整体 tuple；禁止 per-module 位宽覆盖、int8 down-proj、outlier 高精度和向量码本 |
| calibration | WikiText-2 train，256×2048，seed=1 | 精确读取已保存的 `torch.int64 [256,2048]`，禁止重新采样/tokenize |
| weight quantizer | symmetric scalar REAL-Q `w_clip` | `symmetric_union_exact` + `where_out` + compact natural-group qparams |
| weight groupsize | 128 | natural columns before act-order |
| blocksize / act-order | 128 / true | 显式传参，不依赖 parser 默认值 |
| rotation | REAL-Q QuaRot=true，rotation_seed=0 | 不允许 optimized/learned rotation |
| A/K/V | symmetric per-token，groupsize=-1，clip=0.9 | W4A4KV4 开 A/V-aware 与 post-RoPE K-aware；A16KV16 三组 aware=false |
| ResComp | alpha=0.25，alpha2=0.25，percdamp=0.01 | 官方固定超参；ResComp 无需学习率搜索 |
| ResComp bit branch | W2=`org`；W3/W4=`allw` | 用户已确认；`allw` 按 natural group 对每列 clamp |
| 混合精度量化 | 禁止 | 所有 decoder q/k/v/o/gate/up/down projection 使用同一个 W bit；embedding/norm/rotary/lm_head 保持 BF16，FP32 统计仅是计算精度 |
| formal GPU | 每个 invocation 恰好一张物理 GPU，world size=1 | 可在同一 8 卡节点并行 8 个独立单卡 invocation |
| input offload | 默认关闭 | 先以真实 183,359 MiB/card 资源做无 offload smoke；只有单卡容量 OOM 才按用户授权为该组开启，并保留失败证据和例外记录 |
| algorithm timer | 包含 LN fusion、QuaRot、A/K/V site 配置、calibration/H/dXXT/P/R 和所有 decoder weight quantization | 排除模型/数据加载、reference 准备、checkpoint 写盘和所有 evaluation；单卡 GPU-hour=wall/3600 |
| checkpoint | `utils.checkpoint_utils` versioned quantized checkpoint + runtime manifest | 原子保存，记录 stat；稳定后才能被评测加载 |
| KL/PPL | WikiText-2 test，seq_len=2048，full vocabulary FP32 KL | 复用上一轮同模型 frozen reference cache，并要求 cache hit；禁止 top-k KL |
| 十项 QA | PIQA、HellaSwag、ARC-E、ARC-C、WinoGrande、LAMBADA、C-Eval-valid、BoolQ、OpenBookQA、Social-IQA | lm-eval 0.4.4，0-shot，batch=32；逐项准确率及十项均值均落盘 |
| 三项推理 | GSM8K 1319、MATH-500 500、HumanEval+ 164 | thinking-on、greedy、n=1、limit=-1、chat template=true、seed=1234、resume=true |
| 推理 cap / batch | GSM8K=1024/32；MATH-500=2048/16；HumanEval+=2048/16 | 与 REAL-Q 二十组 campaign 相同；每项独立 invocation 和输出目录 |
| HumanEval+ | 生成与官方 EvalPlus 评分分离 | base/plus 分开报告；官方结果必须覆盖 164 个 task |

## 3. P/R 预计算布局（已确认）

`rescomp_controlled` 支持两个代数等价的实现：

| 选项 | 定义 | 优点 | 代价 |
|---|---|---|---|
| `full` | 完整物化 `triu(S @ Hinv.T, 1) @ Hinv` | 最贴近官方 released code；大 GEMM 通常更高效 | 高峰显存更高，尤其 32B MLP down-proj |
| `block_rows` | 每 128 行计算同一表达式，结果不变 | 峰值显存低，仅保留作显式诊断路径 | 分块可能改变 GPU-hour，且不是官方的内存布局 |

用户于 2026-08-09 确认“按照 ResComp 的官方实现来”。因此 20/20 正式组统一显式使用 `full`，不使用 adapter 默认值，也不按模型自动切换。若 `full` 真实 OOM，保留完整证据并再次请求授权；不能自行退到 `block_rows`。

## 4. 模型、calibration 与 reference 身份

| 模型 | 本地路径 | calibration archive SHA256 | token tensor SHA256 | reference cache |
|---|---|---|---|---|
| Qwen3-0.6B | `modelzoo/Qwen3/Qwen3-0.6B` | `aed4e972d31207526d57a4bb62414c24b6c92ff51eda1fa9d70c36c1dfea5d34` | `0f26c1d0544b6e501ae6f071cd35e6740d6aff09479bf20c0247b47f2b431379` | `Qwen3-0.6B_wikitext2_test_2048_01cda27c21a7c9b39866.cache` |
| Llama-3.1-8B-Instruct | `modelzoo/Llama/Llama-3.1-8B-Instruct` | `8030125e31c8738d6cea839d9afd5252617acf9517a226b56ebd1c69804be95e` | `d11551cb3740182cd385427addabf096f46938f79a03c8940cacdc15105ef31f` | `Llama-3.1-8B-Instruct_wikitext2_test_2048_714db90097dcf1551b0f.cache` |
| Qwen3-4B | `modelzoo/Qwen3/Qwen3-4B` | `21210e1929aa90ea23572e7904f8ebda7b3af75ebf8d9037e8196b3d2c52a399` | `0f26c1d0544b6e501ae6f071cd35e6740d6aff09479bf20c0247b47f2b431379` | `Qwen3-4B_wikitext2_test_2048_dd66e7aecb5edc53ca7d.cache` |
| Qwen3-8B | `modelzoo/Qwen3/Qwen3-8B` | `a7acbd907d640eb8ab33a52153da1bf36e4eb550cbc107fe081c9bdc047d3b6f` | `0f26c1d0544b6e501ae6f071cd35e6740d6aff09479bf20c0247b47f2b431379` | `Qwen3-8B_wikitext2_test_2048_9b159923d4e4bfa3dc47.cache` |
| Qwen3-32B | `modelzoo/Qwen3/Qwen3-32B` | `749360f8fb36a6ac91936954f61a7ec688fa8eb0b0d17a61fe4349ff6ec06364` | `0f26c1d0544b6e501ae6f071cd35e6740d6aff09479bf20c0247b47f2b431379` | `Qwen3-32B_wikitext2_test_2048_dbae252da01bcfe1e2fa.cache` |

说明：四个 Qwen3 tokenizer 相同，因此 calibration token tensor 的语义 SHA 相同；Llama tokenizer 不同，必须使用它自己与 REAL-Q 对应的 token artifact。五个 artifact 均为 524,288 tokens。

## 5. 二十组主结果表

状态枚举：配置门禁 → 待 smoke → 待量化 → 量化中 → checkpoint 完成 → 质量评测中 → 推理评测中 → 完成；任何题数、身份或 hash 不满足门禁的结果不得填分。

| ID | 模型 | W/A/K/V | Offload | 量化 wall / GPU·h | Checkpoint | KL / PPL | QA Avg | GSM8K | MATH-500 | HumanEval+ base/plus | Job/GPU | 状态/异常 |
|---:|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 | Qwen3-0.6B | 4/16/16/16 | off | 382.892 s / 0.106359 | 1,503,480,464 B | 0.096181 / 22.4961 | 45.99 | 58.45% | 33.20% | 25.61% / 22.56% | `j-m1krfnr90y`/GPU0 | 三项推理完成；gate/reload 通过 |
| 02 | Qwen3-0.6B | 4/4/4/4 | off | 364.204 s / 0.101168 | 1,503,485,776 B | 3.230000 / 356.8748 | 31.82 | 0.23% | 0.80% | 0.00% / 0.00% | `j-m1krfnr90y`/GPU0 | 完成 |
| 03 | Qwen3-0.6B | 3/16/16/16 | off | 288.250 s / 0.080069 | 1,503,480,464 B | 0.357107 / 28.7269 | 40.56 | 8.34% | 4.40% | 0.00% / 0.00% | `j-m1krfnr90y`/GPU1 | 三项推理完成 |
| 04 | Qwen3-0.6B | 2/16/16/16 | off | 265.378 s / 0.073716 | 1,503,480,464 B | 1.881347 / 112.1039 | 30.67 | 0.30% | 1.20% | 0.00% / 0.00% | `j-m1krfnr90y`/GPU2 | 三项推理完成 |
| 05 | Llama-3.1-8B-Instruct | 4/16/16/16 | off | 1,189.411 s / 0.330392 | 16,060,829,172 B | 0.031681 / 7.4581 | 66.62 | 80.14% | 36.60% | 59.76% / 55.49% | `j-m1krfnr90y`/GPU7 | 完成 |
| 06 | Llama-3.1-8B-Instruct | 4/4/4/4 | off | 1,246.983 s / 0.346384 | 16,060,835,252 B | 0.216409 / 8.9312 | 61.59 | 68.76% | 22.80% | 40.24% / 35.98% | `j-m1krfnr90y`/GPU2 | 完成 |
| 07 | Llama-3.1-8B-Instruct | 3/16/16/16 | off | 1,189.785 s / 0.330496 | 16,060,829,172 B | 0.124030 / 8.1658 | 64.23 | 73.31% | 24.40% | 35.37% / 30.49% | `j-m1krfnr90y`/GPU1 | 完成 |
| 08 | Llama-3.1-8B-Instruct | 2/16/16/16 | off | 1,214.717 s / 0.337421 | 16,060,829,172 B | 0.757812 / 15.1865 | 40.92 | 0.91% | 2.00% | 0.00% / 0.00% | `j-m1krfnr90y`/GPU0 | 完成 |
| 09 | Qwen3-4B | 4/16/16/16 | off | 880.910 s / 0.244697 | 8,823,938,820 B | 0.058114 / 14.5369 | 63.05 | 80.59% | 57.40% | 42.07% / 40.24% | `j-m1krfnr90y`/GPU3 | 完成 |
| 10 | Qwen3-4B | 4/4/4/4 | off | 941.048 s / 0.261402 | 8,823,945,668 B | 0.416726 / 18.0522 | 55.02 | 66.41% | 40.60% | 10.37% / 9.15% | `j-m1krfnr90y`/GPU4 | 完成 |
| 11 | Qwen3-4B | 3/16/16/16 | off | 882.647 s / 0.245180 | 8,823,938,820 B | 0.198081 / 15.5336 | 60.56 | 78.32% | 44.40% | 29.27% / 27.44% | `j-m1krfnr90y`/GPU5 | 完成 |
| 12 | Qwen3-4B | 2/16/16/16 | off | 893.607 s / 0.248224 | 8,823,938,820 B | 1.071142 / 30.5105 | 36.37 | 0.23% | 0.80% | 0.00% / 0.00% | `j-m1krfnr90y`/GPU6 | 完成 |
| 13 | Qwen3-8B | 4/16/16/16 | off | 1,259.352 s / 0.349820 | 16,381,751,372 B | 0.040912 / 10.0599 | 67.02 | 84.31% | 55.40% | 42.68% / 40.24% | `j-m1krfnr90y`/GPU3 | 完成 |
| 14 | Qwen3-8B | 4/4/4/4 | off | 1,276.763 s / 0.354657 | 16,381,758,284 B | 0.296195 / 11.9966 | 60.68 | 80.59% | 47.60% | 24.39% / 22.56% | `j-m1krfnr90y`/GPU5 | 完成 |
| 15 | Qwen3-8B | 3/16/16/16 | off | 1,218.180 s / 0.338383 | 16,381,751,372 B | 0.143240 / 10.8061 | 64.94 | 83.93% | 49.20% | 14.63% / 13.41% | `j-m1krfnr90y`/GPU6 | 完成 |
| 16 | Qwen3-8B | 2/16/16/16 | off | 1,265.216 s / 0.351449 | 16,381,751,372 B | 0.682820 / 16.1136 | 41.94 | 1.36% | 2.00% | 0.00% / 0.00% | `j-m1krfnr90y`/GPU4 | 完成 |
| 17 | Qwen3-32B | 4/16/16/16 | off | 4,642.777 s / 1.289660 | 65,527,268,204 B | 0.051823 / 7.7295 | 71.63 | 93.48% | 61.40% | 51.22% / 50.00% | `j-m1krfnr90y`/GPU7 | 完成 |
| 18 | Qwen3-32B | 4/4/4/4 | off | 5,010.776 s / 1.391882 | 65,527,280,492 B | 0.401260 / 9.4089 | 65.20 | 90.07% | 55.40% | 16.46% / 15.85% | `j-m1krfnr90y`/GPU1 | 完成 |
| 19 | Qwen3-32B | 3/16/16/16 | off | 4,890.387 s / 1.358441 | 65,527,268,204 B | 0.152496 / 8.0018 | 71.40 | 91.51% | 57.20% | 36.59% / 34.76% | `j-m1krfnr90y`/GPU2 | 完成 |
| 20 | Qwen3-32B | 2/16/16/16 | off | 4,957.478 s / 1.377077 | 65,527,268,204 B | 0.641229 / 11.3898 | 50.95 | 4.09% | 3.20% | 0.00% / 0.00% | `j-m1krfnr90y`/GPU0 | 完成 |

## 6. 十项 QA 明细表

| ID | PIQA | Hella. | ARC-E | ARC-C | Wino. | LAMB. | C-Eval | BoolQ | OBQA | SIQA | Avg |
|---:|---|---|---|---|---|---|---|---|---|---|---|
| 01 | 66.49 | 45.25 | 55.47 | 32.08 | 56.12 | 36.06 | 39.08 | 57.00 | 32.20 | 40.12 | 45.99 |
| 02 | 54.95 | 29.04 | 31.06 | 21.93 | 50.91 | 2.58 | 22.44 | 42.48 | 26.60 | 36.18 | 31.82 |
| 03 | 62.57 | 40.40 | 41.33 | 27.13 | 53.28 | 29.23 | 27.64 | 57.22 | 30.20 | 36.64 | 40.56 |
| 04 | 50.60 | 27.78 | 30.26 | 24.74 | 47.28 | 3.43 | 24.74 | 40.52 | 23.80 | 33.57 | 30.67 |
| 05 | 81.12 | 78.06 | 78.70 | 53.84 | 73.40 | 72.46 | 52.75 | 83.73 | 43.20 | 48.93 | 66.62 |
| 06 | 77.97 | 73.83 | 74.12 | 48.72 | 66.54 | 66.99 | 41.98 | 81.13 | 38.00 | 46.57 | 61.59 |
| 07 | 78.24 | 74.53 | 78.24 | 51.28 | 71.98 | 69.34 | 44.95 | 84.19 | 41.20 | 48.31 | 64.23 |
| 08 | 61.81 | 45.76 | 38.01 | 27.05 | 58.48 | 27.48 | 24.15 | 59.11 | 29.20 | 38.13 | 40.92 |
| 09 | 74.86 | 66.44 | 73.19 | 48.72 | 66.54 | 59.65 | 67.46 | 85.14 | 40.20 | 48.26 | 63.05 |
| 10 | 70.08 | 59.42 | 63.80 | 42.66 | 57.70 | 46.38 | 53.71 | 78.23 | 34.80 | 43.40 | 55.02 |
| 11 | 73.94 | 61.74 | 73.82 | 47.87 | 64.01 | 55.42 | 61.29 | 83.67 | 37.00 | 46.88 | 60.56 |
| 12 | 60.55 | 38.10 | 36.78 | 25.34 | 51.38 | 21.13 | 23.63 | 43.18 | 27.20 | 36.39 | 36.37 |
| 13 | 77.91 | 73.27 | 79.25 | 56.23 | 67.32 | 62.84 | 76.00 | 85.50 | 40.80 | 51.07 | 67.02 |
| 14 | 74.76 | 67.71 | 73.02 | 46.59 | 61.80 | 51.14 | 63.89 | 82.35 | 39.80 | 45.75 | 60.68 |
| 15 | 75.90 | 69.96 | 78.11 | 53.24 | 68.67 | 64.22 | 67.38 | 85.60 | 39.60 | 46.67 | 64.94 |
| 16 | 63.38 | 47.81 | 43.39 | 27.65 | 53.12 | 31.11 | 23.18 | 62.35 | 29.00 | 38.38 | 41.94 |
| 17 | 81.12 | 82.00 | 82.53 | 58.45 | 73.64 | 68.12 | 84.32 | 87.25 | 46.80 | 52.05 | 71.63 |
| 18 | 75.95 | 76.21 | 75.25 | 52.47 | 66.69 | 62.00 | 70.28 | 81.25 | 43.80 | 48.06 | 65.20 |
| 19 | 80.74 | 80.30 | 82.11 | 60.67 | 73.72 | 68.93 | 79.94 | 88.20 | 47.80 | 51.54 | 71.40 |
| 20 | 70.57 | 65.15 | 53.49 | 35.49 | 59.51 | 46.69 | 27.19 | 72.26 | 37.60 | 41.50 | 50.95 |

## 7. 调度计划

1. P/R layout 已确认统一为 `full`；生成 immutable plan、resolved command、source/config/model/token/reference/dataset fingerprint，并让 launcher 对所有未显式覆盖的 parser 默认 fail closed。
2. 为隔离 adapter 增加版本化 checkpoint 保存，并用 `realq.ptq --load_qmodel_path` 做 checkpoint reload smoke；量化和所有评测分开，保证 eval 不计入 GPU-hour。
3. 按 `AGENT.md` 参考 sleep job 申请一台 8 卡 debug 节点，SSH 后核验 8×GPU、每卡 183,359 MiB、约 1 TiB 以上 RAM、仓库 `.venv` 版本和共享挂载。
4. 先运行 Qwen3-0.6B W4A16 正式组并把它兼作 gated smoke：默认不 offload，检查首层/末层、checkpoint 保存与量化审计；然后以独立 `realq.ptq` 进程只读加载 checkpoint（不执行 eval）。两步都通过后该量化结果直接进入正式表，才放行其余 19 组。质量/推理仍须完整执行，不能用小样本替代。
5. 正式 Wave 1：Qwen3-0.6B 四组 + Qwen3-4B 四组，占满 8 卡。Wave 2：Llama-3.1-8B-Instruct 四组 + Qwen3-8B 四组，占满 8 卡。Wave 3：Qwen3-32B 四组；空闲卡只运行已有稳定 checkpoint 的独立质量/推理 task，不制造重复 writer。
6. 每组 checkpoint 原子完成并稳定后，单独加载一次完成 KL/PPL+十项 QA；三项推理每项独立 invocation。GSM8K/MATH 在当前进程评分；HumanEval+ GPU 仅生成，官方 EvalPlus 在 Canoe 节点 CPU 安全 runner 中评分。
7. 运行中每次层进度、wall、ETA、OOM/NaN/Cholesky/身份异常、offload 例外和 job/GPU 映射实时追加到本文；不能以 raw JSONL 行数代替当前 generation identity 的唯一题数。
8. 最终审计要求 20 checkpoint、20 KL/PPL、200 QA task results、60 reasoning generation success、20 HumanEval+ official result 全部通过后才把 campaign 标成完成。

## 8. 完成门禁

- [x] P/R layout 已由用户确认；冻结为官方 `full`，后续写入 20/20 resolved config。
- [x] 20/20 量化均为单卡、统一 W bit、无混合精度量化，且保存稳定 checkpoint。
- [x] 20/20 timer 包含规定算法阶段、不含 load/save/eval，并记录 wall 与 GPU-hour。
- [x] 20/20 calibration archive 和 token tensor SHA 与第 4 节一致，无重采样。
- [x] 20/20 full-vocab WikiText-2 KL/PPL 完整，reference cache identity 命中。
- [x] 20/20 十项 QA 均覆盖固定十任务，逐项值及平均值算术一致。
- [x] GSM8K 20×1319、MATH-500 20×500 均为当前身份完整结果。
- [x] HumanEval+ 20×164 generation 与官方 base/plus 结果完整。
- [x] checkpoint/runtime/model/dataset/generation config/hash、job/GPU、日志和异常均可追溯。
- [x] 开发宿主机无模型加载、量化、lm-eval、reasoning generation 或 EvalPlus 高负载进程。
- [x] 全部审计与文档落盘后停止 Canoe debug 任务 `j-m1krfnr90y`，并复核其不再运行。

## 9. 实时记录

### 2026-08-09：启动审计与计划表

- 读取 `AGENT.md`、`对比实验方法.md`、`docs/REALQ_Llama31_Qwen3_20组量化评测_20260808.md`、`docs/推理评测实践经验与故障复盘_20260727.md`、两份 ResComp 分析/实现文档，以及隔离 adapter 的 config/runner/solver。
- 核验 5 个模型 checkpoint、5 个 model-specific calibration artifact 和上一轮同模型 WikiText-2 reference cache均已存在。四个 Qwen calibration token tensor SHA 相同；Llama 因 tokenizer 不同使用独立 artifact。
- 只读查询 Canoe 模板 `j-8j1en3m0aq`：镜像为 `avatar_gen3:v3.0_ngc2501_main`，单 Master Pod 请求 8 GPU、1000 GiB RAM，挂载 `/minimax-avatar-new`，入口为仓库目录下 `sleep 604800`；该历史任务当前已 Aborted。按 creator/name 搜索未发现仍在 Running 的 `zhangqian_debug*` 任务。
- 启动审计时隔离 adapter 尚不保存 checkpoint，并且 README 默认命令带 `--offload_inps`；这两点不满足本 campaign，因此只在隔离目录增加 checkpoint/campaign 支持，正式默认关闭 offload。
- 发现唯一未冻结默认是 ResComp P/R layout。正式实验停在配置门禁，等待用户选择 `full` 或 `block_rows`。

### 2026-08-09：用户确认官方 P/R 实现

- 用户确认“按照 ResComp 的官方实现来”。正式布局冻结为 `full`，20 组统一完整物化 P/R，不使用 `block_rows`。
- 配置门禁解除。若 `full` 在正式单卡上容量 OOM，只记录并上报，不自行改变布局。

### 2026-08-09：隔离 checkpoint 与 campaign 实现

- `rescomp_controlled` 已增加 `--require-checkpoint` 门禁：量化计时结束后才原子保存 `utils.checkpoint_utils` 版本化 fake-quant checkpoint，并把文件状态、格式和版本写入 manifest/progress/result；仍拒绝从隔离量化入口加载 checkpoint。
- 正式默认与 README 已统一为 `full`，命令显式关闭 offload；weight audit 仍要求 decoder 投影全部覆盖且 W bits 集合唯一，禁止任何混合精度量化。
- 新增隔离 campaign `experiments/rescomp_controlled_20group_20260809/`：冻结 20 组 resolved command/fingerprint，单卡 worker 原子认领，量化/质量/三项推理/官方 EvalPlus 分阶段调度，并对 checkpoint、calibration、reference cache、题目 identity 和 source hash fail closed。
- 校验时发现 controlled parser 会把 `output_dir` 规范化为 `<base>/<ModelName>/formal`；campaign 已改为使用 parser 的 resolved output path 查找产物，避免成功量化被误报为缺失文件。
- scheduler 已把 Qwen3-0.6B W4A16 设为硬 gate：先完成该正式量化，再用单独进程验证 checkpoint 身份、wrapper topology 和 state_dict 严格加载；只有 gate marker 通过后才会并发调度其余 19 组。
- dry-run 发现共享 `process_args.parse_gen` 即使只解析参数也会创建 timestamp log。`parse_controlled_args(..., initialize_logging=False)` 现只在 immutable-plan 解析期间抑制这一副作用，正式执行仍保留日志。确认此前正式根只有 43 个由本轮 dry-run 产生的 parser 日志、无 plan/checkpoint/result 后，将整个目录可恢复地移到同级 `rescomp_controlled_20group_20260809_parser_dryrun_quarantine`；没有删除文件或接触既有 RealQ 结果。
- 最终纯解析验证确认 20/20 resolved config、21 个实现文件、5 个模型、5 个精确 calibration、5 个 reference cache、3 个推理数据集均通过，且 build 无文件系统副作用。immutable `plan.json` fingerprint 为 `99a139842cc181e9f86c3994f00b4eedfb73ce0c9c12f7a2a6ebc8a60f75b38a`。
- 按模板配置提交 Canoe sleep job `j-m1krfnr90y`（`zhangqian_debug_rescomp_0809`）：8 GPU、100 CPU、1000 GiB RAM、RDMA、相同镜像与共享挂载、7 天 sleep。13:59:31 +08:00 查询状态为 `Starting`；GPU campaign 尚未启动。

### 2026-08-09：Canoe 环境门禁与 plan v2

- Pod `j-m1krfnr90y-master-0` 于 13:59:57 +08:00 Ready。实测 8×NVIDIA L20C、每卡 183359 MiB，启动前各卡仅 1–4 MiB/0%，无 compute process；`MemTotal=3168481808 kB`（约 3.0 TiB），共享盘可读。
- 首次环境门禁发现仓库 `.venv` 来自宿主 Python 3.10，`python3 -> /usr/local/bin/python3`，而该镜像没有这个解释器；系统 Python 3.12.3 只有 torch `2.6.0a0`、Transformers `4.48.3` 且无 lm-eval。因此未启动任何 GPU 进程，也未接受系统环境降级。
- 曾创建新的外部隔离 venv，但 PyPI/NGC 直连超时；在安装任何依赖前主动中止。随后发现仓库已有、此前正式实验使用的容器环境 `.venv.py312-broken-20260729`，实机版本精确为 Python 3.12.3、torch `2.9.1+cu128`、Transformers `4.56.2`、lm-eval `0.4.4`。新建空 venv 已完整移到 `realq/envs/rescomp_controlled_py312_20260809_unused_quarantine`，没有删除。
- campaign 的固定解释器改为上述正式环境。旧 plan 当时是正式根唯一文件、没有任何 run/checkpoint/result，已完整移到 `rescomp_controlled_20group_20260809_plan_v1_environment_mismatch`。plan v2 fingerprint=`fc1205e548206c6040d060e472041cb6c217ab2a77535df6f16de31a320ccc8b`；Canoe 内 `load_plan` 重建对账通过。
- Canoe 正式环境复跑 `rescomp_controlled/tests`：`15 passed in 10.09s`。硬件、版本、plan、共享挂载与代码测试门禁全部通过；下一步才允许启动 Qwen3-0.6B W4A16 gated formal quantization。

### 2026-08-09：gated smoke attempt001 与 plan v3

- 量化 orchestrator PID=`1004`，只放行 gate worker PID=`1218` / quant PID=`1373` 到 GPU0；GPU1–7 全程空闲。resolved command 明确为 Qwen3-0.6B、W4A16KV16、g128 symmetric `w_clip`、official `full`、no offload、exact calibration、checkpoint required。
- attempt001 在 exact calibration identity、LN fusion、QuaRot、A/K/V site 配置完成后，刚进入 layer 0 的第一次 FP replay 即退出：`RuntimeError: decoder layer returned unexpected hidden shape (2048, 1024)`。此时 `0/28` layer 完成，没有任何 decoder weight 被量化，没有 checkpoint，也没有放行其余组；GPU 随即释放。
- 根因是当前 Transformers 4.56 的 `Qwen3DecoderLayer` / `LlamaDecoderLayer` 直接返回形状 `[1, seq, hidden]` 的 tensor，而官方旧 ResComp replay 写法假设 tuple 并无条件取 `[0]`。隔离 adapter 也沿用了无条件 `[0]`，于是错误删掉 batch 维后再把合法 `[seq, hidden]` 判错。修复只做返回契约适配：若结果本身是 tensor 就直接使用；旧 tuple 路径仍取第一个 tensor；两者统一严格要求 batch=1 后 squeeze。算法、P/R、bits/group/clip/rotation/calibration 均未变化。
- 新增 current tensor、legacy tuple 和非法 shape 三项回归。宿主完整测试更新为 `18 passed`，Canoe 正式环境为 `18 passed in 12.67s`。
- attempt001 的 launch/failure/execution log/failed timing/manifest/progress、runtime snapshot 和 orchestrator log 全部原样保留。plan v2 作为当时唯一 plan 移到 `rescomp_controlled_20group_20260809_plan_v2_gate_tensor_contract_failure`。plan v3 固定 gate retry=`attempt002`、其余 19 组仍为 `attempt001`，fingerprint=`bae7582dd5144f51068c50fbb939a8bc87f7b563e9442314706a01d8fc3606f0`；Canoe 内重建对账通过。
- attempt002 完成 28/28 decoder layers 与 196 个 q/k/v/o/gate/up/down Linear；uniform-bit audit=`pass`、唯一 W bits=`4`、mixed precision=`false`、groupsize/blocksize=`128/128`、mode=`allw`、layout=`full`、offload=`false`。精确 algorithm wall=`382.8921574386768 s`，GPU-hour=`0.10635893262185467`（日志中的 `quant_elapsed=376.9s` 只从 layer loop 起算，不是正式 timer）。
- checkpoint=`1503480464 bytes`，通过 60 秒 stat 稳定窗后发布；独立 `realq.ptq` reload wall=`19.70998029317707 s`，status=`checkpoint_reload_passed`，job=`j-m1krfnr90y`。calibration 仍为 `int64 [256,2048]`、524288 tokens、token SHA=`0f26c1d0...1379`。至此 gate 闭合，scheduler 才放行后续 worker。
- 第一批后续并发实际为：Qwen3-0.6B 余下三组、Qwen3-4B 四组、Llama-3.1-8B-Instruct W4A16，各占一张独立 GPU。由于 gate 已先释放 GPU0，scheduler 用下一顺位 Llama W4 填满第八张卡；这只改变排队顺序，不改变任何组的命令或计时。
- 四个 Qwen3-4B 进程均出现 `lm_head.weight` newly initialized 提示。只读模型身份已知该 checkpoint `tie_word_embeddings=true`，index 只存共享的 `model.embed_tokens.weight`，Transformers 加载后重新绑定 tied lm_head；与上一轮相同，记录提示但不视为缺 shard 或随机独立 head。最终 checkpoint identity/load 与评测仍会 fail closed。

### 2026-08-09：并发 CPU 过订阅与 plan v4

- 放行后的 8 个 worker 默认各自继承约 100 个 CPU threads；在 100-CPU Pod 上形成严重过订阅。证据：单独 gate 的 Qwen3-0.6B W4 首层约 11.5 s，而并发 Qwen3-0.6B W2/W3 首层分别为 `114.6 s` / `104.9 s`，GPU 利用率快照仍低，滚动 ETA 分别达到约 51/47 分钟。这不是 W bit 算法本身足以解释的十倍差异，会把调度争用错误计入 GPU-hour。
- 发现时后续 8 组最多完成 1 层，均没有 checkpoint/quant success。只向已核验的 orchestrator process group `PGID=2166` 发送 TERM；全部 parent/worker/quant child 正常退出，8 卡回到空闲，没有影响已经发布的 gate checkpoint。
- 所有 partial attempt001 原样保留，不作为结果。涉及 Qwen3-0.6B W4A4/W3/W2、Qwen3-4B 四组、Llama-3.1-8B W4；它们固定重试为 attempt002。worker 环境统一冻结 `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=8`，并关闭 tokenizer 并行；该设置只消除跨卡 CPU 争用，不改变模型、量化算法或 GPU 数值路径。
- plan v3、runtime snapshot 与 orchestrator log 已完整归档到 `rescomp_controlled_20group_20260809_plan_v3_cpu_oversubscription`。plan v4 fingerprint=`670d13cc6ef4884098fb2f34ed99b1c238d0019fe4fce73ae00d05d30fba73cb`；Canoe 内 `load_plan`、四项 thread env 和 gate 的 archived-plan acceptance 对账通过。gate 的 solver/runner 实现未变化，因此数值/checkpoint 保留；其独跑 timer 与并发组分开解释，不伪装成同一并发条件。

### 2026-08-09：plan v4 正式量化进度

- plan v4 orchestrator PID=`7143`；每个 worker/量化 child 的 `CUDA_VISIBLE_DEVICES` 只暴露一张物理卡，并在进程环境中验证四个 CPU thread 变量均为 `8`。重启后 Qwen3-0.6B W2/W3 在 6 层处分别约 `56.8 s` / `63.6 s`，恢复到约 9–11 秒/层，确认先前的十倍变慢来自 CPU 过订阅而非算法差异。
- Qwen3-0.6B 四组现已全部发布 `quant_success.json` 并通过 60 秒 checkpoint stat 稳定窗。正式 algorithm wall/GPU-hour 分别为：W4A16 `382.8921574386768 s / 0.10635893262185467`，W4A4KV4 `364.20409460691735 s / 0.10116780405747704`，W3A16 `288.24959742836654 s / 0.08006933261899071`，W2A16 `265.37779767392203 s / 0.07371605490942279`；四者均为 `physical_gpu_count=1`、`offload=false`、official `full`。
- 四组 decoder 投影均为统一 W bit、g128 symmetric `w_clip`，没有 per-module 位宽覆盖或混合精度量化。仅 W4A4KV4 按冻结口径启用 `act_quant_aware_gptq=true` 与 `k_cache_quant_aware_gptq=true`；其余 A16KV16 三组均为 false。checkpoint 大小依次为 `1,503,480,464`、`1,503,485,776`、`1,503,480,464`、`1,503,480,464` bytes。
- 07:02 UTC 快照：Qwen3-4B W4/W3 约 24/36 层，W4A4/W2 约 23/36 层；Llama W4 16/32 层，W4A4/W3 约 5/32 层，W2 3/32 层。8 张 GPU 仍各自只有一个正式 worker；没有 OOM、NaN、Cholesky 或 offload 例外。调度器会在每组 checkpoint 稳定后自动用后续 Qwen3-8B/32B 组填补空闲卡。
- 07:09 UTC：Qwen3-4B 四组均完成 36/36 层、稳定 checkpoint 与 `quant_success` 审计，累计成功 `8/20`。W4A16、W4A4KV4、W3A16、W2A16 的 algorithm wall/GPU-hour 分别为 `880.9096492696553 s / 0.2446971247971265`、`941.0482583083212 s / 0.2614022939745337`、`882.6473259930499 s / 0.2451798127758472`、`893.606516927015 s / 0.2482240324797264`；全部 `physical_gpu_count=1`、`offload=false`、official `full`，checkpoint 大小依次为 `8,823,938,820`、`8,823,945,668`、`8,823,938,820`、`8,823,938,820` bytes。
- Qwen3-4B 的四张释放卡已立即承接 Qwen3-8B 四组；同时 Llama 四组继续在另外四卡运行。新的 8 卡快照仍是一张物理 GPU 对应一个量化 worker，没有重复 writer。
- 07:13 UTC：Llama-3.1-8B-Instruct W4A16 完成并通过稳定门禁，algorithm wall=`1189.4106988436542 s`、GPU-hour=`0.33039186078990396`、checkpoint=`16,060,829,172 bytes`，GPU7、单卡、no offload、official `full`。成功数增至 `9/20`；GPU7 随即开始 Qwen3-32B W4A16，未与其它 invocation 共卡。
- 07:22 UTC：Llama W4A4KV4、W3A16、W2A16 也全部通过，algorithm wall/GPU-hour 分别为 `1246.9833952360786 s / 0.3463842764544663`、`1189.784519652836 s / 0.3304956999035656`、`1214.7166453315876 s / 0.33742129036988544`；checkpoint 分别为 `16,060,835,252`、`16,060,829,172`、`16,060,829,172` bytes。四个 Llama 结果均为单卡、no offload、uniform W bit、official `full`；累计成功 `12/20`。
- 此后量化节点恰好运行 Qwen3-8B 四组与 Qwen3-32B 四组。32B 的 W4/W4A4/W3/W2 分别落在 GPU7/GPU1/GPU2/GPU0；W4 首层 `70.6 s`，滚动 ETA 约 74 分钟，表明本地 32B checkpoint 在 official `full`、无 offload 下已越过模型加载、LN fusion、QuaRot 和第一个完整 decoder layer。
- 07:32 UTC：Qwen3-8B 四组全部通过。W4A16、W4A4KV4、W3A16、W2A16 的 algorithm wall/GPU-hour 分别为 `1259.35171145387 s / 0.3498199198482972`、`1276.7634437978268 s / 0.354656512166063`、`1218.180246845819 s / 0.3383834019016164`、`1265.2159542730078 s / 0.3514488761869466`；checkpoint 分别为 `16,381,751,372`、`16,381,758,284`、`16,381,751,372`、`16,381,751,372` bytes。均为单卡、no offload、uniform W bit、official `full`，累计量化成功 `16/20`。
- Qwen3-8B 释放的 GPU3/4/5/6 随后分别启动 Qwen3-0.6B W4/W2/W4A4/W3 的独立质量评测；Qwen3-32B 继续占 GPU7/0/1/2。质量 worker 只读稳定 checkpoint，并重新 fail-closed 校验量化 identity、reference-cache hit、full-vocab KL/PPL 和固定十项 QA；两类 invocation 不共卡，量化 timer 也不含质量评测。
- 07:42 UTC：Qwen3-0.6B 四组质量评测全部成功。W4A16/W4A4KV4/W3A16/W2A16 的 KL 分别为 `0.096181221306324` / `3.2300002574920654` / `0.3571067452430725` / `1.881347417831421`，PPL 为 `22.496057510375977` / `356.8748474121094` / `28.726943969726562` / `112.10385131835938`，十项 QA 均值为 `45.99` / `31.82` / `40.56` / `30.67`。逐项结果已填入第 6 节，质量 marker 对 execution-log hash 与 checkpoint stat 的复审通过。
- 质量日志中的 `fatal: detected dubious ownership` 是 lm-eval 尝试读取 Git 版本号的非致命辅助命令；证据是同一进程随后继续逐项输出准确率、最终 `returncode=0` 并通过完整 marker audit。没有修改全局 Git safe-directory，也没有因此重跑或改变任何分数。
- GPU3/4/5/6 随即改跑 Qwen3-4B W4/W2/W4A4/W3 的质量评测；GPU0/1/2/7 的四个 Qwen3-32B 量化继续推进，仍无共卡。
- 07:48 UTC：Qwen3-4B W4A16/W3A16/W2A16 质量成功。KL=`0.05811435729265213` / `0.19808140397071838` / `1.0711416006088257`，PPL=`14.536905288696289` / `15.533590316772461` / `30.51050567626953`，十项 QA 均值=`63.05` / `60.56` / `36.37`；W4A4KV4 仍在正常计算 HellaSwag，不提前填分。对应三张释放卡已转入 Llama W4/W3/W2 质量评测。
- 07:54 UTC：Qwen3-4B W4A4KV4 也通过，KL=`0.4167262613773346`、PPL=`18.05219268798828`、十项 QA 均值=`55.02`，完整质量 wall=`613.9391793422401 s`。至此 Qwen3-4B 四组质量均完成，累计 `8/20`；GPU5 转入 Llama W4A4KV4 质量评测。
- 07:57 UTC：Llama W4A16/W3A16/W2A16 质量通过。KL=`0.0316808745265007` / `0.1240302100777626` / `0.757811963558197`，PPL=`7.458139419555664` / `8.165764808654785` / `15.186476707458496`，十项 QA 均值=`66.62` / `64.23` / `40.92`。对应 GPU3/6/4 已转入 Qwen3-8B W4/W3/W2 质量，Llama W4A4KV4 继续在 GPU5 运行；质量累计 `11/20`。
- 08:05 UTC：Llama W4A4KV4 质量通过，KL=`0.2164088636636734`、PPL=`8.931207656860352`、十项 QA 均值=`61.59`；Llama 四组质量全部完成。Qwen3-8B W4A16/W3A16/W2A16 也相继成功，KL=`0.04091212898492813` / `0.1432403326034546` / `0.6828204393386841`，PPL=`10.059866905212402` / `10.806148529052734` / `16.113557815551758`，QA 均值=`67.02` / `64.94` / `41.94`。质量累计 `15/20`，GPU5 继续 Qwen3-8B W4A4KV4；其余三张评测卡开始 Qwen3-0.6B W4/W3/W2 的 GSM8K。
- 08:16 UTC：Qwen3-8B W4A4KV4 质量成功，KL=`0.29619506001472473`、PPL=`11.996639251708984`、十项 QA 均值=`60.68`、wall=`626.2468319553882 s`。至此除尚在量化的 Qwen3-32B 外，16 组质量均完成；GPU5 加入 Qwen3-0.6B W4A4KV4 GSM8K，形成四个 32B 量化 + 四个 0.6B GSM8K 的一卡一任务布局。
- 08:34 UTC：Qwen3-32B W4A16 通过 64/64 层、checkpoint 稳定与审计；algorithm wall=`4642.777320703957 s`、GPU-hour=`1.2896603668622104`、checkpoint=`65,527,268,204 bytes`、GPU7、no offload、official `full`。量化累计 `17/20`，GPU7 随即独立加载该 checkpoint 做质量评测。
- 同时 Qwen3-0.6B W4A16/W3A16/W2A16 的 GSM8K 均覆盖精确 `1319/1319` generations，pass@1=`0.5845337376800607` / `0.08339651250947688` / `0.003032600454890068`；主表按百分比显示为 `58.45%` / `8.34%` / `0.30%`。三张释放卡转入对应 MATH-500；W4A4KV4 GSM8K 继续在 GPU5 运行。
- 08:47 UTC：Qwen3-32B W4A4KV4/W3A16/W2A16 全部发布成功 marker，algorithm wall/GPU-hour 分别为 `5010.775659217965 s / 1.3918821275605457`、`4890.3871433320455 s / 1.3584408731477904`、`4957.47828544816 s / 1.3770773015133777`；checkpoint=`65,527,280,492` / `65,527,268,204` / `65,527,268,204 bytes`。全程单卡、no offload、official `full`。
- quant orchestrator 正常退出；在 Canoe 正式环境对 20 个 marker 逐组调用 `audit_quant`，输出 `audit_quant_passed 20`。因此统一 W bit、mixed-precision=false、g128 symmetric `w_clip`、exact calibration hash、timer/physical GPU、checkpoint stat/source identity 三项量化完成门禁可以勾选。GPU0/1/2/7 随即分别运行 32B W2/W4A4/W3/W4 质量评测，GPU3/4/5/6 继续推理任务。
- 08:52 UTC：Qwen3-32B W4A16 质量成功，KL=`0.05182306095957756`、PPL=`7.729452133178711`、十项 QA 均值=`71.63`、wall=`1037.4150350252166 s`；逐项结果已填表，质量累计 `17/20`。GPU7 改做 Qwen3-0.6B W4A16 HumanEval+ 164 题生成；其官方 EvalPlus 评分仍会在 generation 完整并稳定后单独执行。
- 09:03 UTC：Qwen3-32B W3A16 质量成功，KL=`0.15249569714069366`、PPL=`8.001798629760742`、十项 QA 均值=`71.40`、wall=`1005.5265863970853 s`；逐项结果已填表，质量累计 `18/20`。释放的 GPU2 随即开始 Qwen3-0.6B W3A16 HumanEval+，仍保持一卡一任务。
- 09:08 UTC：Qwen3-32B W2A16 质量成功，KL=`0.6412291526794434`、PPL=`11.389839172363281`、十项 QA 均值=`50.95`、wall=`1210.7041887538508 s`；逐项结果已填表，质量累计 `19/20`。释放的 GPU0 随即开始 Qwen3-0.6B W2A16 HumanEval+；GPU7 在 W4 HumanEval+ 生成完成后转入 Qwen3-4B W4A16 GSM8K，官方 EvalPlus 评分在 CPU 独立执行。
- 09:12 UTC：Qwen3-32B W4A4KV4 质量成功，KL=`0.40125972032546997`、PPL=`9.408913612365723`、十项 QA 均值=`65.20`、wall=`1425.1868594870903 s`。节点内逐组 `audit_quality` 输出 `audit_quality_passed 20`，至此 full-vocab KL/PPL 与固定十项 QA 两项完整门禁闭合；GPU1 转入 Qwen3-0.6B W4A4KV4 HumanEval+。
- Qwen3-0.6B W4A16 的 MATH-500=`33.20%`；HumanEval+ 164/164 生成后经官方 EvalPlus `parallel=32` 独立评分，base/plus pass@1=`25.61% / 22.56%`。W3A16/W2A16 的 MATH-500 分别为 `4.40% / 1.20%`；相关 task identity、题数和文件 hash 均已由 generation marker 复核。
- Qwen3-0.6B W3A16/W2A16 的 HumanEval+ 也分别覆盖 164/164，并由官方 EvalPlus 独立评分；两者 base/plus pass@1 均为 `0.00% / 0.00%`。这两个低位宽结果按原值保留，没有用内部启发式分数替代或删题重跑。
- Qwen3-0.6B W4A4KV4 GSM8K 完成 1319/1319，pass@1=`0.002274450341167551`（表中 `0.23%`），wall=`4258.228442668915 s`。GPU5 随即转入同组 MATH-500；低准确率按冻结配置原样记录，没有改变 A/K/V 量化或重跑筛选。
- Qwen3-4B W4A16 GSM8K 完成 1319/1319，pass@1=`0.8059135708870356`（表中 `80.59%`），wall=`2088.1253321170807 s`。隔离调度器首次完成自动补卡验证：发现 GPU7 的原 worker 成功退出并已有 marker 后，才在同一卡启动该组 HumanEval+，没有出现重叠 writer。
- Qwen3-4B W3A16/W2A16 GSM8K 也各自完成 1319/1319，pass@1=`0.7831690674753601` / `0.002274450341167551`（表中 `78.32% / 0.23%`），wall=`2119.116674423218 s / 2120.8382663726807 s`。两张释放卡与刚完成 0.6B W4A4 HumanEval+ 的 GPU1，分别接续 Qwen3-4B W4A4 的三项推理。
- Qwen3-0.6B W4A4KV4 HumanEval+ 完成 164/164，官方 EvalPlus base/plus pass@1=`0.00% / 0.00%`，评分 wall=`38.153701685834676 s`。至此该组仅剩仍在 GPU5 运行的 MATH-500；0.6B 其余三组的三项推理均已完整结束。
- Qwen3-4B W4A16 HumanEval+ 完成 164/164，官方 EvalPlus base/plus pass@1=`0.42073170731707316 / 0.4024390243902439`（表中 `42.07% / 40.24%`），base/plus 通过题数=`69/66`。GPU7 已接续 Qwen3-4B W3A16 HumanEval+，该组仅剩 MATH-500。
- Qwen3-4B W4A16 MATH-500 完成 500/500，pass@1=`0.574`（表中 `57.40%`），wall=`3162.5043613910675 s`；该组质量与三项推理至此全部完成。GPU6 随即接续 Qwen3-4B W2A16 HumanEval+。
- Qwen3-4B W3A16 MATH-500 完成 500/500，pass@1=`0.444`（表中 `44.40%`），wall=`3288.734857559204 s`；该组仅剩 GPU7 上的 HumanEval+。释放的 GPU2 已开始 Llama-3.1-8B-Instruct W4A16 GSM8K。
- Qwen3-4B W2A16 MATH-500 完成 500/500，pass@1=`0.008`（表中 `0.80%`），wall=`3144.143942117691 s`；该组仅剩 GPU6 上的 HumanEval+。GPU0 已开始 Llama-3.1-8B-Instruct W4A16 MATH-500。
- Qwen3-4B W3A16 HumanEval+ 完成 164/164，官方 EvalPlus base/plus pass@1=`0.2926829268292683 / 0.27439024390243905`（表中 `29.27% / 27.44%`），base/plus 通过题数=`48/45`；该组至此完整结束。GPU7 已接续 Llama-3.1-8B-Instruct W4A16 HumanEval+。
- Llama-3.1-8B-Instruct W4A16 HumanEval+ 完成 164/164，官方 EvalPlus base/plus pass@1=`0.5975609756097561 / 0.5548780487804879`（表中 `59.76% / 55.49%`），base/plus 通过题数=`98/91`。GPU7 已接续 Llama W4A4KV4 GSM8K；W4A16 的 GSM8K/MATH-500 继续独立运行。
- Qwen3-4B W2A16 HumanEval+ 完成 164/164，官方 EvalPlus base/plus pass@1=`0.00% / 0.00%`；该组至此完整结束。GPU6 已接续 Llama W4A4KV4 MATH-500。
- Qwen3-4B W4A4KV4 HumanEval+ 完成 164/164，官方 EvalPlus base/plus pass@1=`0.10365853658536585 / 0.09146341463414634`（表中 `10.37% / 9.15%`），base/plus 通过题数=`17/15`；该组 GSM8K/MATH-500 继续运行。GPU3 已接续 Llama W4A4KV4 HumanEval+。
- Llama-3.1-8B-Instruct W4A16 GSM8K 完成 1319/1319，pass@1=`0.8013646702047005`（表中 `80.14%`），wall=`1580.7232184410095 s`；该组仅剩 MATH-500。GPU2 已接续 Llama W3A16 GSM8K。
- 10:44 UTC 对账：主表已填入当前全部 20 组量化/质量结果，以及所有已发布推理分数；完成度为 generation `23/60`、官方 HumanEval+ `9/20`。尚未形成完整 marker 的任务只显示“—/运行中”，没有把中间 JSONL 行数写成最终成绩。后台调度器 PID=`80940` 心跳正常，8 张卡均各自只有一个 worker。
- Llama-3.1-8B-Instruct W4A4KV4 HumanEval+ 完成 164/164，官方 EvalPlus base/plus pass@1=`0.4024390243902439 / 0.3597560975609756`（表中 `40.24% / 35.98%`），base/plus 通过题数=`66/59`。GPU3 已接续 Llama W3A16 MATH-500；完成度更新为 generation `24/60`、官方 HumanEval+ `10/20`。
- Llama-3.1-8B-Instruct W4A16 MATH-500 完成 500/500，pass@1=`0.366`（表中 `36.60%`），wall=`2609.3534133434296 s`；该组质量与三项推理至此全部完成。GPU0 已接续 Llama W3A16 HumanEval+；完成度更新为 generation `25/60`。
- Llama-3.1-8B-Instruct W3A16 GSM8K 完成 1319/1319，pass@1=`0.733131159969674`（表中 `73.31%`），wall=`1428.0307278633118 s`；HumanEval+ 完成 164/164，官方 base/plus pass@1=`0.35365853658536583 / 0.3048780487804878`（表中 `35.37% / 30.49%`），通过题数=`58/50`。该组仅剩 MATH-500；GPU0/GPU2 已分别接续 Llama W2A16 GSM8K/MATH-500，完成度更新为 generation `27/60`、官方 `11/20`。
- Qwen3-0.6B W4A4KV4 MATH-500 完成 500/500，pass@1=`0.008`（表中 `0.80%`），wall=`6299.32799243927 s`；该组至此完整结束。Qwen3-4B W4A4KV4 GSM8K 也完成 1319/1319，pass@1=`0.6641394996209249`（表中 `66.41%`），wall=`5093.350710868835 s`，该组仅剩 MATH-500。释放的 GPU5/GPU1 已接续 Llama W2A16 HumanEval+ 与 Qwen3-8B W4A16 GSM8K；完成度更新为 generation `29/60`。
- 用户追加要求：所有 60 个 generation、20 个官方评分、最终 campaign 审计和本文档落盘后，停止 Canoe debug 任务 `j-m1krfnr90y` 并复核停止状态；不得在结果尚未闭合时提前停机。
- Llama-3.1-8B-Instruct W4A4KV4 GSM8K 完成 1319/1319，pass@1=`0.6876421531463229`（表中 `68.76%`），wall=`3521.0311233997345 s`；该组仅剩 MATH-500。GPU7 已接续 Qwen3-8B W4A16 MATH-500；完成度更新为 generation `30/60`。
- Llama-3.1-8B-Instruct W3A16 MATH-500 完成 500/500，pass@1=`0.244`（表中 `24.40%`），wall=`2644.1636292934418 s`，该组至此完整结束；GPU3 已接续 Qwen3-8B W4A16 HumanEval+。Llama W2A16 HumanEval+ 也完成 164/164，官方 base/plus pass@1=`0.0 / 0.0`（表中 `0.00% / 0.00%`），通过题数=`0/0`，官方评分 wall=`77.05080821085721 s`；GPU5 已接续 Qwen3-8B W4A4KV4 GSM8K。完成度更新为 generation `32/60`、官方 HumanEval+ `12/20`。
- Llama-3.1-8B-Instruct W2A16 GSM8K 完成 1319/1319，pass@1=`0.009097801364670205`（表中 `0.91%`），wall=`1764.3862087726593 s`；该组仅剩 MATH-500。GPU0 已接续 Qwen3-8B W4A4KV4 MATH-500，完成度更新为 generation `33/60`。
- Qwen3-8B W4A16 HumanEval+ 完成 164/164，generation wall=`1095.1570389270782 s`；官方 EvalPlus base/plus pass@1=`0.4268292682926829 / 0.4024390243902439`（表中 `42.68% / 40.24%`），通过题数=`70/66`，官方评分 wall=`37.15584997320548 s`。GPU3 已接续同模型 W4A4KV4 HumanEval+；完成度更新为 generation `34/60`、官方 `13/20`。
- Llama-3.1-8B-Instruct W2A16 MATH-500 完成 500/500，pass@1=`0.02`（表中 `2.00%`），wall=`2673.3286945819855 s`，该组至此完整结束；GPU2 已接续 Qwen3-8B W3A16 GSM8K。Qwen3-8B W4A16 GSM8K 也完成 1319/1319，pass@1=`0.8430629264594389`（表中 `84.31%`），wall=`2104.8867149353027 s`，该组仅剩 MATH-500；GPU1 已接续同模型 W3A16 MATH-500。完成度更新为 generation `36/60`。
- Qwen3-4B W4A4KV4 MATH-500 完成 500/500，pass@1=`0.406`（表中 `40.60%`），wall=`8138.42812204361 s`，该组至此完整结束。GPU4 已接续 Qwen3-8B W3A16 HumanEval+；完成度更新为 generation `37/60`。
- Qwen3-8B W4A16 MATH-500 完成 500/500，pass@1=`0.554`（表中 `55.40%`），wall=`3161.826417207718 s`，该组至此完整结束。GPU7 已接续同模型 W2A16 GSM8K；完成度更新为 generation `38/60`。
- Llama-3.1-8B-Instruct W4A4KV4 MATH-500 完成 500/500，pass@1=`0.228`（表中 `22.80%`），wall=`6673.592164754868 s`，该组至此完整结束。GPU6 已接续 Qwen3-8B W2A16 MATH-500；完成度更新为 generation `39/60`。
- Qwen3-8B W3A16 HumanEval+ 完成 164/164，generation wall=`1081.2026462554932 s`；官方 EvalPlus base/plus pass@1=`0.14634146341463414 / 0.13414634146341464`（表中 `14.63% / 13.41%`），通过题数=`24/22`，官方评分 wall=`37.59123429702595 s`。GPU4 已接续同模型 W2A16 HumanEval+；完成度更新为 generation `40/60`、官方 `14/20`。
- Qwen3-8B W3A16 GSM8K 完成 1319/1319，pass@1=`0.8392721758908264`（表中 `83.93%`），wall=`2137.2782728672028 s`；该组仅剩 MATH-500。GPU2 已启动首项 Qwen3-32B W4A16 GSM8K；完成度更新为 generation `41/60`。
- Qwen3-8B W4A4KV4 HumanEval+ 完成 164/164，generation wall=`2571.7953686714172 s`；官方 EvalPlus base/plus pass@1=`0.24390243902439024 / 0.22560975609756098`（表中 `24.39% / 22.56%`），通过题数=`40/37`，官方评分 wall=`37.624685008078814 s`。GPU3 已接续 Qwen3-32B W4A16 MATH-500；完成度更新为 generation `42/60`、官方 `15/20`。
- Qwen3-8B W2A16 HumanEval+ generation 完成 164/164，GPU4 已接续 Qwen3-32B W4A16 HumanEval+，官方 EvalPlus 评分仍在运行；Qwen3-8B W3A16 MATH-500 也完成 500/500，pass@1=`0.492`（表中 `49.20%`），wall=`3144.6301851272583 s`，该组至此完整结束。GPU1 已接续 Qwen3-32B W4A4KV4 GSM8K；generation 完成度更新为 `44/60`，官方仍为 `15/20`。
- Qwen3-8B W2A16 HumanEval+ 官方 EvalPlus base/plus pass@1=`0.0 / 0.0`（表中 `0.00% / 0.00%`），通过题数=`0/0`；generation wall=`1090.1728146076202 s`，官方评分 wall=`37.31717117317021 s`。官方完成度更新为 `16/20`，该组仍剩 GSM8K/MATH-500。
- Qwen3-8B W2A16 GSM8K 完成 1319/1319，pass@1=`0.013646702047005308`（表中 `1.36%`），wall=`2204.4437251091003 s`；该组仅剩 MATH-500。GPU7 已接续 Qwen3-32B W4A4KV4 MATH-500；完成度更新为 generation `45/60`。
- Qwen3-8B W4A4KV4 GSM8K 完成 1319/1319，pass@1=`0.8059135708870356`（表中 `80.59%`），wall=`5098.5495018959045 s`；该组仅剩 MATH-500。GPU5 已接续 Qwen3-32B W4A4KV4 HumanEval+；完成度更新为 generation `46/60`。
- Qwen3-8B W2A16 MATH-500 完成 500/500，pass@1=`0.02`（表中 `2.00%`），wall=`3196.916113615036 s`，该组至此完整结束。GPU6 已接续 Qwen3-32B W3A16 GSM8K；完成度更新为 generation `47/60`。
- Qwen3-32B W4A16 HumanEval+ 完成 164/164，generation wall=`2589.304492712021 s`；官方 EvalPlus base/plus pass@1=`0.5121951219512195 / 0.5`（表中 `51.22% / 50.00%`），通过题数=`84/82`，官方评分 wall=`37.810247727669775 s`。GPU4 已接续同模型 W3A16 MATH-500；完成度更新为 generation `48/60`、官方 `17/20`。
- Qwen3-8B W4A4KV4 MATH-500 完成 500/500，pass@1=`0.476`（表中 `47.60%`），wall=`7691.663921117783 s`；该组及 Qwen3-8B 四组至此全部闭合。GPU0 已接续 Qwen3-32B W3A16 HumanEval+；完成度更新为 generation `49/60`。
- Qwen3-32B W4A16 GSM8K 完成 1319/1319，pass@1=`0.9347990902198635`（表中 `93.48%`），wall=`5095.411758184433 s`；该组仅剩 MATH-500。GPU2 已接续同模型 W2A16 GSM8K；完成度更新为 generation `50/60`。
- Qwen3-32B W4A4KV4 HumanEval+ 完成 164/164，generation wall=`5072.860184907913 s`；官方 EvalPlus base/plus pass@1=`0.16463414634146342 / 0.15853658536585366`（表中 `16.46% / 15.85%`），通过题数=`27/26`，官方评分 wall=`37.728160210419446 s`。GPU5 已接续同模型 W2A16 MATH-500；完成度更新为 generation `51/60`、官方 `18/20`。
- Qwen3-32B W3A16 HumanEval+ 完成 164/164，generation wall=`2569.418146133423 s`；官方 EvalPlus base/plus pass@1=`0.36585365853658536 / 0.3475609756097561`（表中 `36.59% / 34.76%`），通过题数=`60/57`，官方评分 wall=`37.835426721256226 s`。GPU0 已接续同模型 W2A16 HumanEval+；完成度更新为 generation `52/60`、官方 `19/20`。
- Qwen3-32B W3A16 GSM8K 完成 1319/1319，pass@1=`0.9150871872630781`（表中 `91.51%`），wall=`5171.058809518814 s`；该组仅剩 MATH-500，完成度更新为 generation `53/60`。
- Qwen3-32B W4A16 MATH-500 完成 500/500，pass@1=`0.614`（表中 `61.40%`），wall=`7718.720716714859 s`；该组的质量与三项推理至此全部闭合，完成度更新为 generation `54/60`。
- Qwen3-32B W4A4KV4 GSM8K 完成 1319/1319，pass@1=`0.9006823351023503`（表中 `90.07%`），wall=`9044.082765817642 s`；该组仅剩 MATH-500，完成度更新为 generation `55/60`。
- Qwen3-32B W2A16 HumanEval+ generation 完成 164/164，最后一次官方 EvalPlus 评分随即启动；该组 GSM8K/MATH-500 继续运行，generation 完成度更新为 `56/60`。官方分数在评分 marker 发布前不提前填表。
- Qwen3-32B W2A16 HumanEval+ 官方 EvalPlus base/plus pass@1=`0.0 / 0.0`（表中 `0.00% / 0.00%`），通过题数=`0/0`；generation wall=`2594.637267589569 s`，官方评分 wall=`38.64045147225261 s`。至此官方 HumanEval+ 完成度达到 `20/20`；该组仍剩 GSM8K/MATH-500。
- Qwen3-32B W2A16 GSM8K 完成 1319/1319，pass@1=`0.04094010614101592`（表中 `4.09%`），wall=`5225.30905175209 s`；该组仅剩 MATH-500，完成度更新为 generation `57/60`。
- Qwen3-32B W3A16 MATH-500 完成 500/500，pass@1=`0.572`（表中 `57.20%`），wall=`7736.403527021408 s`；该组的质量与三项推理至此全部闭合，完成度更新为 generation `58/60`。
- Qwen3-32B W2A16 MATH-500 完成 500/500，pass@1=`0.032`（表中 `3.20%`），wall=`7681.739715576172 s`；该组的质量与三项推理至此全部闭合，完成度更新为 generation `59/60`。
- Qwen3-32B W4A4KV4 MATH-500 完成 500/500，pass@1=`0.554`（表中 `55.40%`），pipeline wall=`13896.965138435364 s`；该组及全部 20 组质量与三项推理至此闭合，generation 完成度达到 `60/60`、官方 HumanEval+ 为 `20/20`。
- 运行时调度器正常记录 `all_workers_complete`，随后全量审计通过：quant=`20`、quality=`20`、paper QA results=`200`、reasoning=`60`、official EvalPlus=`20`、runs=`20`，audit fingerprint=`5814da6be832cec0b6b373ff9fbd1295d6062c8fbfbee8f7f24ad06c87d845ce`。独立再次执行 `campaign audit --root` 得到相同计数与 fingerprint，状态=`complete`。
- 停机前节点核验：8 张 GPU 均为 `0 MiB / 0%`，Canoe 节点和开发宿主机均无量化、lm-eval、reasoning generation、EvalPlus 或 campaign audit 残留进程。正式根内唯一 failure marker 是已完整保留并在前文说明的 Qwen3-0.6B W4 gate `attempt001` tensor-return-contract 失败；成功的 `attempt002`、checkpoint reload 和最终全量审计均通过。
- 停机前在 Canoe 正式 Python 3.12 环境复跑 scheduler/campaign 语法检查和 `rescomp_controlled/tests`，结果为 `18 passed in 12.26s`；本文两张 20 行结果表均无“—”缺项，宿主机 `git diff --check` 通过。宿主机同名 `.venv.py312-broken-20260729` 仍是前文已记录的不可执行挂载环境，因此不把宿主侧失败调用误报成代码测试失败。
- 使用 `canoe-ababvideo-submit` 的 `safe_stop.py` 对 `j-m1krfnr90y` 执行安全停止；脚本先验证项目为 minimax-avatar、任务创建者与当前用户匹配，再完成停止。随后按 `canoe-info` 只读查询复核：job name=`zhangqian_debug_rescomp_0809`、state=`Aborted`、reason=`AbortedBy`、message=`aborted by 章迁`、creator_id=`u-7s51rmjf43`。共享盘 checkpoint、日志、generation 与 `final_audit.json` 均未删除。
