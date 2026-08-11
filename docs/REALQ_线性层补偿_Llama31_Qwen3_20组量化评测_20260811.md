# REAL-Q 单线性层补偿 Llama 3.1 / Qwen 3 二十组量化与评测记录（2026-08-11）

## 1. 任务范围

本 campaign 复用 `docs/REALQ_Llama31_Qwen3_20组量化评测_20260808.md` 的模型、位宽、校准、学习率搜索和评测协议，唯一算法消融为关闭 REAL-Q Plus 的 Transformer-block 整体权重更新，恢复原论文/旧实现的单线性层更新：每次 Block-GD 只更新当前线性层的未量化后缀，不更新当前或下一 Transformer block 中的其它线性层。

目标模型为 Qwen3-0.6B、Llama-3.1-8B-Instruct、Qwen3-4B、Qwen3-8B、Qwen3-32B；每个模型运行 W4A16、W4A4KV4、W3A16、W2A16，共 20 组。

## 2. 冻结配置

| 配置项 | 冻结值 | 状态 |
|---|---:|---|
| `full_block_refresh` | `false` | 已显式写入并 fail-closed 校验 |
| 权重更新范围 | `current_linear_trailing_columns` | 已冻结 |
| 校准集 | WikiText2，256 条 × 2048 tokens | 已冻结 |
| weight group size | 128 | 已冻结 |
| W/A/K/V | W4A16、W4A4KV4、W3A16、W2A16 | 已冻结 |
| W/A/K/V quantizer | symmetric | 已冻结 |
| A/K/V aware 与 clip | 仅 W4A4KV4 开启 aware，clip=0.9 | 已冻结 |
| rotation / act-order / w-clip | true / true / true | 已冻结 |
| `a_loss_ratio` | 1 | 已冻结 |
| tuning profile | group=128、block=256、backward=16、loss_slide_window=false | 已冻结 |
| formal profile | group=128、block=128、backward=32、loss_slide_window=true | 已冻结 |
| Hessian accum | 默认 64；仅 Qwen3-32B=32 | 沿用用户授权 |
| OOM 处理 | 只降低获准的容量 batch，不改其它量化参数 | 已冻结 |
| WikiText2 质量 | full-vocab FP32 KL/PPL | 20/20 完成并审计 |
| 十项 QA | PIQA、HellaSwag、ARC-E、ARC-C、WinoGrande、LAMBADA、C-Eval、BoolQ、OBQA、SIQA | 200/200 完成并审计 |
| 三项推理 | GSM8K、MATH-500、HumanEval+；thinking-on、greedy、n=1、全量 | 60/60 generation、20/20 官方评分完成并审计 |

## 3. 二十组结果

| ID | 模型 | W/A/K/V | LR / tuning KL | 正式 KL | PPL | QA Avg | GSM8K | MATH-500 | HumanEval+ base/plus | GPU·h | 状态 |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 01 | Qwen3-0.6B | 4/16/16/16 | `5e-6 / 0.02643203` | 0.06334694 | 21.7094 | 45.88 | 59.67 | 35.20 | 25.00 / 22.56 | 0.099 | 全部完成 |
| 02 | Qwen3-0.6B | 4/4/4/4 | `2e-4 / 2.27085066` | 0.98816073 | 46.0317 | 34.76 | 0.38 | 0.60 | 0.00 / 0.00 | 0.094 | 全部完成 |
| 03 | Qwen3-0.6B | 3/16/16/16 | `1e-5 / 0.09054436` | 0.22560166 | 24.6261 | 40.89 | 21.00 | 7.40 | 1.22 / 1.22 | 0.092 | 全部完成 |
| 04 | Qwen3-0.6B | 2/16/16/16 | `5e-6 / 0.67557061` | 1.57509351 | 82.0018 | 31.78 | 0.68 | 2.00 | 0.00 / 0.00 | 0.099 | 全部完成 |
| 05 | Llama-3.1-8B-Instruct | 4/16/16/16 | `5e-6 / 0.00687024` | 0.02496048 | 7.3821 | 66.25 | 81.58 | 42.40 | 57.93 / 54.88 | 0.818 | 全部完成 |
| 06 | Llama-3.1-8B-Instruct | 4/4/4/4 | `1e-6 / 0.19599727` | 0.19640522 | 8.6909 | 61.72 | 68.92 | 22.60 | 38.41 / 33.54 | 0.924 | 全部完成 |
| 07 | Llama-3.1-8B-Instruct | 3/16/16/16 | `1e-5 / 0.02443988` | 0.09564172 | 7.9285 | 65.01 | 71.49 | 26.20 | 45.12 / 41.46 | 0.801 | 全部完成 |
| 08 | Llama-3.1-8B-Instruct | 2/16/16/16 | `2e-5 / 0.16160478` | 0.56603551 | 12.5885 | 46.59 | 0.00 | 0.00 | 0.00 / 0.00 | 0.799 | 全部完成 |
| 09 | Qwen3-4B | 4/16/16/16 | `1e-6 / 0.01095654` | 0.04346002 | 13.6937 | 62.74 | 87.34 | 60.00 | 47.56 / 44.51 | 0.551 | 全部完成 |
| 10 | Qwen3-4B | 4/4/4/4 | `1e-5 / 0.35550311` | 0.32651350 | 15.0027 | 56.77 | 76.12 | 44.60 | 25.61 / 24.39 | 0.532 | 全部完成 |
| 11 | Qwen3-4B | 3/16/16/16 | `1e-7 / 0.03627279` | 0.15423504 | 14.1615 | 61.12 | 83.62 | 53.60 | 41.46 / 38.41 | 0.541 | 全部完成 |
| 12 | Qwen3-4B | 2/16/16/16 | `1e-6 / 0.18904847` | 0.78348303 | 20.9588 | 40.03 | 1.21 | 1.20 | 0.00 / 0.00 | 0.535 | 全部完成 |
| 13 | Qwen3-8B | 4/16/16/16 | `0 / 0.00736449` | 0.03278381 | 9.8711 | 67.34 | 87.11 | 57.60 | 43.90 / 42.07 | 0.232 | 全部完成 |
| 14 | Qwen3-8B | 4/4/4/4 | `5e-6 / 0.25885999` | 0.26469281 | 11.4523 | 62.72 | 81.58 | 46.00 | 30.49 / 26.83 | 0.930 | 全部完成 |
| 15 | Qwen3-8B | 3/16/16/16 | `0 / 0.02391553` | 0.11800054 | 10.3457 | 65.23 | 85.90 | 53.40 | 19.51 / 18.29 | 0.225 | 全部完成 |
| 16 | Qwen3-8B | 2/16/16/16 | `1e-7 / 0.12194353` | 0.65187836 | 15.0857 | 44.11 | 1.14 | 3.40 | 0.00 / 0.00 | 0.906 | 全部完成 |
| 17 | Qwen3-32B | 4/16/16/16 | `1e-6 / 0.00713391` | 0.04527105 | 7.6046 | 71.97 | 93.48 | 62.80 | 51.83 / 49.39 | 4.673 | 全部完成 |
| 18 | Qwen3-32B | 4/4/4/4 | `5e-6 / 0.41599941` | 0.25566542 | 8.2580 | 68.71 | 93.18 | 62.20 | 35.98 / 35.37 | 4.796 | 全部完成 |
| 19 | Qwen3-32B | 3/16/16/16 | `5e-6 / 0.01971932` | 0.13114163 | 7.7452 | 70.90 | 93.56 | 61.80 | 46.95 / 45.12 | 4.718 | 全部完成 |
| 20 | Qwen3-32B | 2/16/16/16 | `5e-6 / 0.07927571` | 0.51529497 | 10.1171 | 57.60 | 36.62 | 10.00 | 3.66 / 3.66 | 4.800 | 全部完成 |

### 3.1 WikiText2 与十项 QA 明细

口径为 WikiText2 256×2048 全测试 token、全词表 FP32 CE/KL；QA 使用 lm-eval `0.4.4`，任务顺序固定为 PIQA、HellaSwag、ARC-Easy、ARC-Challenge、WinoGrande、LAMBADA、C-Eval-valid、BoolQ、OpenBookQA、Social-IQA。每项优先 `acc_norm,none`，否则 `acc,none`；先把单项百分比四舍五入到两位，再取十项算术平均。每个 checkpoint 只加载一次，连续完成 KL/PPL 与十项 QA。

| ID | 模型 | W/A/K/V | 正式 KL | PPL | QA Avg | PIQA | Hella | ARC-E | ARC-C | Wino | LAMBADA | C-Eval | BoolQ | OBQA | SIQA | 状态 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 01 | Qwen3-0.6B | 4/16/16/16 | 0.06334694 | 21.7094 | 45.88 | 66.43 | 45.48 | 48.23 | 32.17 | 55.88 | 36.74 | 34.62 | 65.14 | 33.40 | 40.69 | 完成 |
| 02 | Qwen3-0.6B | 4/4/4/4 | 0.98816073 | 46.0317 | 34.76 | 57.40 | 32.86 | 37.21 | 24.23 | 48.70 | 9.74 | 23.40 | 51.56 | 27.40 | 35.11 | 完成 |
| 03 | Qwen3-0.6B | 3/16/16/16 | 0.22560166 | 24.6261 | 40.89 | 62.84 | 41.36 | 42.47 | 27.65 | 55.01 | 32.41 | 23.40 | 57.52 | 29.20 | 37.00 | 完成 |
| 04 | Qwen3-0.6B | 2/16/16/16 | 1.57509351 | 82.0018 | 31.78 | 52.99 | 28.65 | 31.23 | 21.42 | 50.83 | 4.83 | 22.96 | 46.18 | 24.40 | 34.34 | 完成 |
| 05 | Llama-3.1-8B-Instruct | 4/16/16/16 | 0.02496048 | 7.3821 | 66.25 | 81.28 | 78.25 | 77.90 | 53.58 | 73.56 | 72.11 | 52.15 | 83.36 | 43.60 | 46.67 | 完成 |
| 06 | Llama-3.1-8B-Instruct | 4/4/4/4 | 0.19640522 | 8.6909 | 61.72 | 77.53 | 74.60 | 73.57 | 47.44 | 67.48 | 66.29 | 42.20 | 79.60 | 43.00 | 45.45 | 完成 |
| 07 | Llama-3.1-8B-Instruct | 3/16/16/16 | 0.09564172 | 7.9285 | 65.01 | 80.03 | 75.49 | 79.25 | 51.88 | 70.56 | 70.41 | 46.88 | 83.27 | 43.40 | 48.93 | 完成 |
| 08 | Llama-3.1-8B-Instruct | 2/16/16/16 | 0.56603551 | 12.5885 | 46.59 | 64.36 | 55.13 | 43.39 | 30.12 | 59.98 | 42.73 | 23.63 | 70.03 | 36.00 | 40.48 | 完成 |
| 09 | Qwen3-4B | 4/16/16/16 | 0.04346002 | 13.6937 | 62.74 | 75.19 | 67.30 | 72.47 | 48.29 | 65.67 | 59.58 | 67.16 | 85.47 | 38.80 | 47.49 | 完成 |
| 10 | Qwen3-4B | 4/4/4/4 | 0.32651350 | 15.0027 | 56.77 | 71.27 | 61.52 | 68.31 | 42.92 | 59.91 | 49.58 | 55.27 | 78.93 | 35.20 | 44.78 | 完成 |
| 11 | Qwen3-4B | 3/16/16/16 | 0.15423504 | 14.1615 | 61.12 | 73.88 | 63.22 | 75.13 | 49.06 | 63.46 | 55.64 | 61.07 | 84.74 | 39.20 | 45.85 | 完成 |
| 12 | Qwen3-4B | 2/16/16/16 | 0.78348303 | 20.9588 | 40.03 | 61.48 | 42.52 | 42.72 | 28.24 | 52.01 | 23.91 | 26.52 | 56.24 | 29.60 | 37.10 | 完成 |
| 13 | Qwen3-8B | 4/16/16/16 | 0.03278381 | 9.8711 | 67.34 | 77.75 | 73.84 | 79.42 | 55.63 | 69.61 | 63.46 | 76.75 | 86.09 | 40.20 | 50.67 | 完成 |
| 14 | Qwen3-8B | 4/4/4/4 | 0.26469281 | 11.4523 | 62.72 | 74.86 | 68.70 | 77.48 | 52.05 | 61.48 | 55.09 | 67.46 | 82.78 | 40.00 | 47.34 | 完成 |
| 15 | Qwen3-8B | 3/16/16/16 | 0.11800054 | 10.3457 | 65.23 | 75.63 | 70.92 | 78.87 | 54.01 | 68.27 | 62.90 | 71.84 | 84.80 | 38.40 | 46.62 | 完成 |
| 16 | Qwen3-8B | 2/16/16/16 | 0.65187836 | 15.0857 | 44.11 | 65.83 | 49.21 | 47.47 | 30.03 | 54.30 | 32.72 | 24.52 | 67.34 | 30.60 | 39.10 | 完成 |
| 17 | Qwen3-32B | 4/16/16/16 | 0.04527105 | 7.6046 | 71.97 | 81.66 | 82.10 | 82.66 | 59.64 | 73.72 | 67.09 | 86.03 | 88.17 | 46.00 | 52.66 | 完成 |
| 18 | Qwen3-32B | 4/4/4/4 | 0.25566542 | 8.2580 | 68.71 | 78.67 | 79.38 | 80.35 | 56.66 | 69.61 | 62.70 | 78.53 | 85.50 | 44.80 | 50.87 | 完成 |
| 19 | Qwen3-32B | 3/16/16/16 | 0.13114163 | 7.7452 | 70.90 | 81.12 | 80.71 | 81.02 | 59.47 | 72.45 | 67.01 | 82.02 | 88.44 | 46.40 | 50.36 | 完成 |
| 20 | Qwen3-32B | 2/16/16/16 | 0.51529497 | 10.1171 | 57.60 | 75.79 | 69.85 | 63.80 | 42.58 | 64.88 | 56.18 | 41.23 | 77.68 | 40.40 | 43.65 | 完成 |

## 4. 调度与完成门禁

- 两台 8-GPU Canoe debug sleep job，实验通过 SSH 在 Pod 内启动；最多 16 个独立单卡 lane。
- 每组先运行共享 producer，再由 agent 小批次显式放行 LR；单组最多 20 次 launch。
- 选择点必须是已观测全局最低点并有严格左右邻居；若最优为物理下界 0，至少三个不同正 LR 必须更差。
- 20/20 正式单卡量化保存稳定 checkpoint；20/20 KL/PPL、200/200 QA 单项、60/60 推理 generation 与 20/20 HumanEval+ 官方评分通过审计。
- 实验结束后停止两台 debug job。

## 5. 实时记录

- 2026-08-11：用户确认关闭 Transformer-block 整体更新，默认恢复单线性层更新；核心实现与 17 项回归测试通过，commit `bfa66e4` 已推送到 `origin/zq`。
- 2026-08-11：创建独立 campaign `realq-linear-refresh-llama31-qwen3-20group-20260811-v1`，输出根为 `/minimax-avatar-new/zhangqian/realq/experiment_data/realq_linear_refresh_20group_20260811`。
- 2026-08-11 10:24：两台 8×L20C/B100 口径 sleep debug job `j-tc64iczh8w`（node0）与 `j-jd9dtz5a1h`（node1）均进入 `Running/retry=0`；真实 plan fingerprint=`6c749049865856fcda739d07e16bbcab84ddb8349e618d83b9f0f7449ae060e1`。
- 2026-08-11 10:29：两侧 tune controller 已通过 SSH 启动。node0 固定负责 Qwen3-0.6B/4B/32B producer，node1 固定负责 Llama-3.1-8B-Instruct/Qwen3-8B producer；实机命令均显式包含 `--full_block_refresh false`，首批 20 组各放行 `[0,1e-6,5e-6]`。
- 2026-08-11 10:36：campaign 与刷新范围回归测试合并门禁为 `45 passed`（仅历史 `.pytest_cache` 权限 warning）。两台 job 仍为 `Running/retry=0`，两侧 controller PID `272/189` 均存活；Qwen3-0.6B 在 tuning `global_loss_bsz=8` 成功，Qwen3-4B 在 8 档容量 OOM 后按冻结阶梯于 4 档成功，Llama-3.1-8B-Instruct 与 Qwen3-8B 的 8 档也仅出现容量 OOM并自动进入 4 档，未改变 Hessian/backward/位宽或其它算法参数。
- 2026-08-11 10:38：首轮人工 GPU 巡检中，node0 有 5 条计算 lane、node1 有 6 条计算 lane，共 11/16 卡处于 producer 或候选计算；其余 lane 正等待 Llama/Qwen3-8B/Qwen3-32B 的共享 producer 门禁，并非 controller 退出。首批 Qwen3-0.6B 的四个 `lr=0` 已原子成功：W4A16=`0.0266479887`、W4A4KV4=`4.9969859123`、W3A16=`0.0916572139`、W2A16=`0.7600040436`；四组均继续按升序运行 `1e-6`，未提前选点。
- 2026-08-11 10:42：第二轮人工巡检确认两台 job 仍为 `Running/retry=0`，两侧均已派发 8 条 worker lane；瞬时 `pmon` 会因模型加载、共享缓存与 CPU 阶段低于 16 条计算进程，但不存在可越过依赖门禁的待派发工作。Llama-3.1-8B-Instruct 与 Qwen3-8B 已分别在 tuning `global_loss_bsz=4` 成功写入 producer marker，当前 producer 完成为 4/5；Qwen3-32B 仍在 8 档继续。Qwen3-0.6B 四组 `lr=1e-6` 均成功：W4A16=`0.0267704912`、W4A4KV4=`4.9364576330`、W3A16=`0.0936128795`、W2A16=`0.7558585405`，随后进入 `5e-6`。W4/W3 的 0 暂时更优但尚不满足物理零边界至少三个正 LR 均更差的门禁，因此没有提前选择。
- 2026-08-11 10:52：Qwen3-32B producer 的 tuning `global_loss_bsz=8/4/2` 均在冻结 profile 下发生真实容量 OOM，控制器按唯一获准的容量阶梯进入 1 档；`hessian_accum_bsz=32`、backward=16、block=256 与所有量化参数均未变化。1 档已稳定推进到 static precompute `81/256`，越过前三档的即时失败点。Qwen3-0.6B W4/W3 的 `5e-6` 分别为 `0.0264320280/0.0915320814`，两者均为当前高边界最优，下一批必须先向高侧扩展，不能在 `5e-6` 提前选点。
- 2026-08-11 10:54：Qwen3-0.6B W4A4KV4/W2A16 的 `5e-6` 也成功，Exact KL=`4.6533126831/0.6755706072`；至此该模型四组的 0/1e-6/5e-6 均完整，且四组都以 5e-6 为高边界最低点。已冻结下一受控 batch `q06_high_extend_v1`（fingerprint=`b45929c870a9e805d982b8b9132c36e59881ba56bfebd77c1997abfd0988d0c9`），每组只新增 `1e-5`；暂不并发启动第二个 scheduler，以免与仍持有 GPU lane 的 initial controller 冲突。
- 2026-08-11 10:59：Qwen3-4B W4A16 在 `0/1e-6/5e-6` 的 Exact KL=`0.0111377584/0.0109565360/0.0110949529`，以严格 bracket 原子选择 `lr=1e-6`，protocol fingerprint=`5760e1a490e71cfb32814c0e8294d587e8501e3cd103b321db5269ac5826ec73`。首次手工 selector 因 SSH 会话未带 controller 的 `REALQ_PYTHON` 而在启动子进程前失败，没有修改 state；补齐已审计的 Pod Python 3.12 路径后成功。Qwen3-4B W3A16 的 0 暂为最低但仅有两个正 LR 更差，已冻结 `q4_w3_zero_refine_v1`（fingerprint=`e4315c7dd8c959c74a34ba350f0091cebac159f59fdcd23586bdf490a7ae32be`），待补 `1e-7/2e-7/5e-7`。
- 2026-08-11 11:01：Qwen3-32B tuning producer 在 `global_loss_bsz=1` 成功，rc=0、elapsed=`665.07s`，五模型 producer 达到 5/5；四个 32B `lr=0` 已分别在两机启动。人工核对 node0 initial controller 的完整 active/pending 映射后，确认 GPU0–5 覆盖四个非 32B 最后候选和两条 32B 串行链，而 GPU6/7 不会被其后续 pending 选中；因此用受控 `_tune` 入口在 GPU6/7 启动 Qwen3-0.6B W4/W3 的 `1e-5`，对应 PID=`20227/20228`，使 node0 回到 8 条 lane，未启动会争抢 GPU 的第二个 scheduler。
- 2026-08-11 11:05：Llama-3.1-8B-Instruct W4A4KV4 在 `0/1e-6/5e-6` 的 Exact KL=`0.2313983887/0.1959972680/0.2295626253`，严格选择 `1e-6`；Qwen3-4B W2A16 对应为 `0.1925265193/0.1890484691/0.2000547200`，同样严格选择 `1e-6`。Llama W2A16 的 `5e-6` 仍为高边界最低点，已冻结 `llama_w2_high_extend_v1`（fingerprint=`b3c2492500ce6dab49c2ac2a070c7a7c04dc773a8b2ef177e4116a6d89cf8752`）并在 node0 GPU5 启动 `1e-5`；Qwen3-4B W3A16 的首个低侧补点 `1e-7` 在 GPU4 启动。此时 node0 的 GPU0–3 由 initial controller 使用、GPU4–7 为受控补点，8/8 lane 均已占用。
- 2026-08-11 11:07：Qwen3-0.6B W4A16 的 `1e-5` Exact KL=`0.0265138578`，因此 `5e-6` 在 `1e-6/1e-5` 间形成严格 bracket 并原子选择，四次 launch。W3A16 的 `1e-5=0.0905443579` 仍为高边界最低点，已冻结 `q06_w3_high_extend_2e5_v1`（fingerprint=`964dea3121b92c6e26814d55ffb5d075a837ee1718c8ba5228c7b3d243ad3a14`），下一步只补 `2e-5`。node0 GPU6/7 随即复用为 Qwen3-0.6B W2A16/W4A4KV4 的 `1e-5`，PID=`22275/22276`。
- 2026-08-11 11:15：两台 Canoe job 均为 `Running/retry=0`，两侧 initial controller PID=`272/189` 存活。Qwen3-0.6B W2A16 的 `1e-5=0.7320384383` 高于 `5e-6=0.6755706072`，已在 `1e-6/1e-5` 间严格选择 `5e-6`，protocol fingerprint=`79e1dfdae5592bc70d73445d7dbc62544aad470c15544329eed477fc8c24dd9d`。同期观测到 Qwen3-0.6B W4A4KV4、W3A16，Llama W4/W3/W2，Qwen3-4B W4A4KV4，Qwen3-8B W4A4KV4 仍需向高侧扩展；Qwen3-8B W4/W3/W2 的 0 点最低但各仅有两个正 LR 更差，需要第三个正点；Qwen3-4B W3A16 的 `1e-7=0.0362727866` 略优于 0，需继续补 `2e-7`。已冻结 `refine_round2_v1`，并通过受控 `_tune` 入口在 node0 GPU0/1/4/5/6/7 和 node1 GPU0–4 启动 11 个新候选；加上四个 32B `lr=0`，共占用 15/16 lane。node1 GPU5 保留给最先完成候选的顺序后继，避免越过候选依赖。
- 2026-08-11 11:17：node0 的 Qwen3-32B W4/W3 `lr=0` 成功完成，Exact KL=`0.0071783089/0.0200191177`。旧 initial controller 随后按其内部空闲表把 W4 的 `1e-6` 指向 GPU0，但该卡已被受控补点占用；`realq_auto_tune` 的 GPU 进程门禁在启动量化前以 rc=2 fail-closed，未创建 `lr=1e-6` trial、未写错误结果或 checkpoint。为消除动态抢卡，node1 controller PID 189 已 `SIGSTOP`，其现有两个 32B 子进程继续运行；node0 W4/W3 的 `1e-6` 分别改在确认空闲的 GPU2/3 重新启动。该事件只影响调度，不改变任何实验配置或 trial 计数。
- 2026-08-11 11:25：Qwen3-4B W3A16 的 `2e-7=0.0366561264` 高于 `1e-7=0.0362727866`，故 `1e-7` 已在物理 0 与 `2e-7` 间形成严格 bracket 并选定，protocol fingerprint=`8f95a3d3798b35a67538bbdef8f0172827f561c0833ccaa47bd8002bb8c5822d`。Llama W4A16 的 `1e-5=0.0071000070` 也使 `5e-6=0.0068702442` 形成严格 bracket，已选定，fingerprint=`2a021f2c34a5b4526083a0d9b1f2d03139a4af1ca1f5853846bc1ed2ceb03000`。Llama W3/W2 与 Qwen3-4B W4A4KV4 的最新高边界仍改善，已冻结 `refine_round3_v1` 并补发 `2e-5/5e-5/2e-5`。Qwen3-32B W4A4KV4/W2A16 的 `lr=0` 成功为 `0.4288357496/0.0832047164`，node1 controller 保持暂停，其 `1e-6` 已固定在 GPU6/7 续跑。为利用三个短时空闲 lane，已提前启动三个已选 run 的正式阶段：Qwen3-0.6B W4A16、Qwen3-4B W4A16、Llama W4A4KV4；各自只使用冻结 formal profile，合计回到 16/16 lane。
- 2026-08-11 11:29：Qwen3-8B W3/W4 的 `1e-7` Exact KL=`0.0239434373/0.0074722874`，均严格高于 0；结合 `1e-6/5e-6`，两组满足“0 为唯一观测全局最低且三个不同正 LR 均更差”，分别以 physical-zero 门禁选择 `lr=0`，fingerprint=`d62ab8bb5cf7ef06a45d98c78e0d5f7d92a23ce9c85263e0e5529ebf5f5d2c03/b042ae1eef9733bf23069be83659d7d3ee23da6a93afc57d2359ce589563c59b`。Qwen3-8B W4A4KV4 的 `1e-5=0.2672387660` 高于 `5e-6=0.2588599920`，已严格选择 `5e-6`。W2A16 的 `1e-7=0.1219435260` 优于 0，已冻结 `q8_w2_low_refine_v1` 并补发 `2e-7`。三个已选 Qwen3-8B run 同时进入 formal（共享 producer 以文件锁串行生成一次），两机重新达到 16/16 个受控进程。
- 2026-08-11 11:30：Qwen3-0.6B W3A16 的完整局部曲线为 `5e-6=0.0915320814, 1e-5=0.0905443579, 2e-5=0.0914857239, 5e-5=0.0919326320`，已严格选择 `1e-5`，fingerprint=`33eb4e47b4c029907be63221837ed6390783466965800d0963f377425c0569d4`。人工首次把 `2e-5` 误读为最低时，selector 以“best is lr=1e-05”在写状态前拒绝，随后按全曲线正确提交；无错误 decision/state/checkpoint。W4A4KV4 的 `5e-5=3.4355101585` 仍为高边界最低点，已补发 `1e-4`；另用同卡池启动 Qwen3-0.6B W2A16 formal，继续保持 16/16 个受控进程。
- 2026-08-11 11:35：Llama W3A16/W2A16 与 Qwen3-4B W4A4KV4 的高侧补点分别为 `2e-5=0.0245296005`、`5e-5=0.1881942451`、`2e-5=0.3708584607`，因此严格选择 `1e-5/2e-5/1e-5`，fingerprint=`a4cc5dd566429ccd5d3d62fb193b9d6d5eb3fcfecac5ccea7bb556c7ff418450/1747d1157bff2545edb7e89a8a0be140a732e672c2c453a6ca37406ece687d91/0807cf8c60496871505aed8d554772da4e452155ce80151a40d81a073bc4d14d`，三者均已进入 formal。Qwen3-0.6B W4A4KV4 的 `1e-4=2.5195648670` 仍继续改善，顺序补发 `2e-4`。四个已启动模型的 formal producer 在 `global_loss_bsz=32` 均只出现容量 OOM，并按冻结阶梯下降；Qwen3-0.6B 的 16 档已 rc=0。两台作业均 `Running/retry=0`，16 个受控进程存活；瞬时约 11 张卡在 GPU 计算，其余处于共享 formal producer 锁等待或模型加载阶段。
- 2026-08-11 11:40：Qwen3-0.6B W4A16/W2A16 formal 均在 `global_loss_bsz=16` 完成，checkpoint 各 `1,503,480,592 B`，稳定性门禁通过，完成时间为 `11:36/11:37`。空闲 GPU 已回填 Qwen3-0.6B W3A16、Qwen3-4B W3A16/W2A16 formal。Qwen3-0.6B W4A4KV4 的 `2e-4=2.2708506584` 仍为高边界最低点；为避免阻塞 tuning 临界路径，临时 `SIGSTOP` 尚未占用 GPU 的 Qwen3-8B W4 formal 等锁进程，在确认 GPU3 无 compute context 后于 node1 GPU3 补发 `5e-4`，完成后恢复原 formal。两台 job 仍为 `Running/retry=0`。
- 2026-08-11 11:45：上一条 GPU3 复用并未启动错误 trial：Qwen3-8B W4 formal 的 `realq.ptq` 子进程已在父进程暂停前创建，但在检查时尚未建立 CUDA context，随后开始占卡；Qwen3-0.6B KV4 `5e-4` 的 `realq_auto_tune` GPU 门禁在启动量化前检测到该子进程并 rc=2 fail-closed，无 `lr=5e-4` trial/result/checkpoint。formal 父进程已 `SIGCONT`，现有子进程继续正常计算；KV4 补点留待下一张真正释放的固定 GPU。Qwen3-8B W2A16 的 `2e-7=0.1227021813` 高于 `1e-7=0.1219435260`，已在 `0/2e-7` 间严格选择 `1e-7`，fingerprint=`64670c38133ecc1d977803886889f2ea0b7b95087966d93bffe47e194def5ce9`，并进入 formal。两台 job 仍为 `Running/retry=0`。
- 2026-08-11 11:50：node0 GPU2 释放后，Qwen3-0.6B W4A4KV4 的 `5e-4` 固定卡重跑成功，Exact KL=`3.2241005898`；`2e-4=2.2708506584` 因而在 `1e-4=2.5195648670` 与 `5e-4` 间形成严格 bracket。Qwen3-32B W4/W3 的 `5e-6` 分别在 node0 GPU0/3 启动；此时 node1 8/8 张卡均为高 GPU 利用率，node0 在补发 W4 后也恢复 8 条受控 lane。
- 2026-08-11 11:55：Qwen3-0.6B W4A4KV4 已严格选择 `lr=2e-4`，9 次有效 launch，protocol fingerprint=`5658757206f18a9bbfdf568445cfbc2f10cfb51fa2b18050a54ff5bec8b86ee0`，随即在 node0 GPU2 启动 formal。Qwen3-0.6B W3A16 与 Qwen3-8B W3A16 formal 已成功写入稳定 checkpoint；空出的 node1 GPU2 已回填此前尚未启动的 Llama W4A16 formal。两台 Canoe job 均保持 `Running/retry=0`。
- 2026-08-11 12:00：Qwen3-8B W4A16 formal 也在 `global_loss_bsz=8` 完成，当前稳定 checkpoint 为 5/20；node1 GPU3 已启动 Qwen3-8B W3A16 的三项推理评测。Qwen3-32B W2A16/KV4 的 `1e-6` Exact KL=`0.0807364434/0.4161804616`，均优于物理 0 的 `0.0832047164/0.4288357496`，已分别在 node1 GPU7/6 顺序补发 `5e-6`。人工巡检时 node0 8/8 张卡均为 `99–100%` 利用率；node1 GPU6 在 KV4 候选完成后短暂释放并立即回填，恢复两机共 16 条受控任务。
- 2026-08-11 12:02：Qwen3-0.6B W4A4KV4 formal 在 `global_loss_bsz=16` 成功，checkpoint `1,503,480,592 B`，稳定性门禁通过，当前 checkpoint 6/20；其释放的 node0 GPU2 已立即回填该 run 的三项推理评测。Qwen3-32B 下一高侧单点 batch `q32_high_extend_1e5_v1` 已预冻结，fingerprint=`80aabdd03d4b6795cc04b2ebfb700ef200f3737605ad84e17e38eee537cfc62d`，只会在各 run 的 `5e-6` 结果证明有必要后逐项放行。
- 2026-08-11 12:11：Qwen3-4B W4A16/W4A4KV4 formal 均在 `global_loss_bsz=8` 成功并通过 checkpoint 稳定性门禁，当前 checkpoint 8/20。两条 formal 子进程完成后，未等待父进程的 60 秒文件稳定观察占用空卡：node1 GPU0 与 node0 GPU6 已分别安全回填已有稳定 checkpoint 的 Qwen3-0.6B W3A16/W4A16 三项推理评测；被观察的 Qwen3-4B checkpoint 文件未被这些独立只读任务触碰。
- 2026-08-11 12:14：Qwen3-4B W3A16/W2A16 formal 同样在 `global_loss_bsz=8` 完成并通过稳定性门禁，checkpoint 达到 10/20。Qwen3-32B W3A16 的 `5e-6` Exact KL=`0.0197193194`，优于 `0=0.0200191177` 与 `1e-6=0.0200416595`，仍是高边界最低点，故按预冻结 batch 在原 GPU3 继续 `1e-5`；另外两张释放卡回填 Qwen3-0.6B W2A16 与 Qwen3-8B W4A16 三项推理评测。12:15 人工巡检时两台 job 都是 `Running/retry=0`，各 8 条任务；新回填的三条任务尚在模型加载阶段，其余卡持续计算。
- 2026-08-11 12:17：Qwen3-32B W4A16 的 `5e-6=0.0071388776` 略高于 `1e-6=0.0071339123`，后者在 `0=0.0071783089` 与 `5e-6` 间形成严格 bracket，已选定 `lr=1e-6`，protocol fingerprint=`41a95dc6e40e36fa345f552cc99856f802913d55ddb440516d5291cfda24ef84`；该 run 已在 node0 GPU0 启动 formal，并首先生成 Qwen3-32B 共享 formal producer。LR 选择达到 17/20。
- 2026-08-11 12:20：Qwen3-32B formal producer 的 `global_loss_bsz=32` 在约 `145.06s` 后发生已识别容量 OOM，随后自动进入冻结阶梯的 16 档；`hessian_accum_bsz=32`、backward=32、block=128 和全部量化参数保持不变。人工巡检两台 job 均为 `Running/retry=0`，16 张卡实际利用率均非零，无任务丢失。
- 2026-08-11 12:30：Qwen3-32B formal producer 的 16/8/4 档继续仅出现已识别容量 OOM，现已进入 2 档；除 `global_loss_bsz` 外配置未变。Qwen3-32B W2A16/KV4 的 `5e-6` Exact KL=`0.0792757124/0.4159994125`，均仍为高边界最低点，已在原 GPU7/6 继续 `1e-5`。Llama W3A16/W2A16 formal 在 `global_loss_bsz=8` 成功，checkpoint 达到 12/20；其 node0 GPU4/5 已回填两条三项推理评测。
- 2026-08-11 12:35：Qwen3-32B formal producer 在 `global_loss_bsz=2` 成功，W4A16 已直接进入同一成功档的正式量化，没有重复 32/16/8/4 容量尝试。Llama W4A4KV4 formal 在 `global_loss_bsz=8` 成功，checkpoint 达到 13/20；node1 GPU5 已用此前稳定的 Qwen3-4B W4A16 推理评测提前回填。Qwen3-8B W3A16 的 GSM8K 1319/1319 generation 完成并通过任务级 marker，继续同一 GPU 上的 MATH-500。
- 2026-08-11 12:38：Qwen3-8B W2A16/W4A4KV4 formal 均在 `global_loss_bsz=8` 成功并通过稳定性门禁，checkpoint 达到 15/20；两条 run 已分别在 node1 GPU1/GPU4 回填三项推理评测。Qwen3-0.6B W4A16/W3A16 的 GSM8K generation 也已完成，当前共有 3 条 GSM8K 任务级成功标记。两台 Canoe job 仍为 `Running/retry=0`。
- 2026-08-11 12:42：Qwen3-32B W3A16 的 `1e-5=0.0201575998` 高于 `5e-6=0.0197193194`，后者在 `1e-6=0.0200416595` 与 `1e-5` 间形成严格 bracket，已选定并在 node0 GPU3 进入 `global_loss_bsz=2` 的正式量化，protocol fingerprint=`9f13c415eac685c231a1ac814a171b02d83877e5361dafc69b994cfcfd6c7e93`。LR 选择达到 18/20。人工巡检两台 job 均为 `Running/retry=0`，node0/node1 各 8 条受控任务，16/16 卡均有实际 GPU 负载。
- 2026-08-11 12:47：Llama W4A16 formal 在 `global_loss_bsz=8` 成功并通过稳定性门禁，checkpoint 达到 16/20；node1 GPU2 释放后立即回填该 run 的三项推理评测。12:47 人工巡检确认两台 job 均为 `Running/retry=0`；回填前 node0 8 条、node1 7 条受控任务，回填后恢复 16/16 lane。
- 2026-08-11 12:53：Qwen3-32B W2A16 的 `1e-5=0.0796049982` 高于 `5e-6=0.0792757124`，后者在 `1e-6=0.0807364434` 与 `1e-5` 间形成严格 bracket，已选定并在 node1 GPU7 进入 `global_loss_bsz=2` 的正式量化，protocol fingerprint=`eb164276a27b95008e8c280d0d0c4aeb05a7fbfb997a468b6c8f0ecfa9b24a5d`。LR 选择达到 19/20。同期 Qwen3-8B W4A16 的 GSM8K 1319/1319 完成，准确率=`87.11%`，已回填主表。
- 2026-08-11 13:00：Qwen3-32B W4A4KV4 的 `1e-5=0.4171571732` 高于 `5e-6=0.4159994125`，后者在 `1e-6=0.4161804616` 与 `1e-5` 间形成严格 bracket，已选定并在 node1 GPU6 进入 `global_loss_bsz=2` 的正式量化，protocol fingerprint=`b3f7e23446ff7c9a94f45b3e509a689c80509fed74f812abaac84cb5d6caa65a`。至此 20/20 LR 全部按冻结门禁选定，四条 32B 正式量化并行；12:59 巡检时两台平台均 `Running/retry=0`，GPU6 的短暂收尾空窗已完成回填。
- 2026-08-11 13:16：推理 generation marker 达到 13/60；已发布并回填的 GSM8K 包括 Qwen3-0.6B W4/W3/W2/KV4=`59.67/21.00/0.68/0.38%`、Llama W4/W3/W2=`81.58/71.49/0.00%`、Qwen3-4B W4=`87.34%`、Qwen3-8B W4/W3/W2=`87.11/85.90/1.14%`。Qwen3-0.6B W4/W3 的 MATH-500 分别为 `35.20/7.40%`。13:14 人工巡检确认两台 job 均 `Running/retry=0`，16/16 lane 持续有实际负载。
- 2026-08-11 13:36：Qwen3-0.6B W4/W3/W2 三组均已完成三项 generation 与 EvalPlus 官方评分；HumanEval+ base/plus 分别为 `25.00/22.56%`、`1.22/1.22%`、`0.00/0.00%`，对应 MATH-500 为 `35.20/7.40/2.00%`。三条释放 lane 已依次回填 Qwen3-4B W3、Qwen3-4B W2、Llama W4A4KV4 evaluator，官方 scorer 均为 CPU-only 且未占 GPU。13:34 巡检的瞬时 node0 `7/8` 已在同一分钟恢复，node1 保持 `8/8`。
- 2026-08-11 13:52：Qwen3-8B W3A16 三项推理全部完成，GSM8K/MATH-500/HumanEval+ base/plus=`85.90/53.40/19.51/18.29%`。Llama W3A16 已完成 HumanEval+ generation 并启动 CPU-only 官方评分；node0 GPU4 释放后，已用任务级原子 claim 回填 Qwen3-4B W4A4KV4 的 MATH-500，与该 run 在 node1 GPU3 上串行进行的 GSM8K 互不重叠。
- 2026-08-11 13:54：人工巡检确认两台 Canoe job 均为 `Running/retry=0`；node0/node1 的 8 张 GPU 均存在独立 CUDA 计算进程，16/16 卡实际利用率为 `38–100%`，存储占用符合当前 32B formal 或推理评测。当前任务级 generation/official marker=`23/60` 与 `5/20`；Llama W3A16 官方 HumanEval+ base/plus=`45.12/41.46%`，三项推理已完成。
- 2026-08-11 14:03：Llama W4A16/W2A16 三项推理完成；W4A16 的 GSM8K/MATH-500/HumanEval+ base/plus=`81.58/42.40/57.93/54.88%`，W2A16 为 `0/0/0/0%`。两条 CPU-only 官方评分均已写入原子 marker。释放的 node0 GPU5/node1 GPU2 已经分别回填 Llama W4A4KV4 的 MATH-500/HumanEval+，与其原有 GSM8K 并行且任务输出互斥。
- 2026-08-11 14:04：人工巡检确认两台 job 仍为 `Running/retry=0`，两侧均有 8/8 条 CUDA 计算进程；刚回填的 node0 GPU5/node1 GPU2 已完成模型加载并进入计算，16/16 卡无空置。generation/official marker=`27/60` 与 `7/20`；Qwen3-4B W3A16 GSM8K=`83.62%`，已继续 MATH-500。
- 2026-08-11 14:09：Llama W4A4KV4 HumanEval+ base/plus=`38.41/33.54%`；Qwen3-8B W4A16 HumanEval+ base/plus=`43.90/42.07%`，该 run 三项推理完成。新增 Qwen3-4B W4A16 MATH-500=`60.00%`、Qwen3-4B W2A16 GSM8K=`1.21%`、Qwen3-8B W4A4KV4 GSM8K=`81.58%`。两条评测释放的 node1 GPU2/node0 GPU7 已继续回填 Qwen3-4B W2/W3 的 HumanEval+；generation/official marker=`32/60` 与 `9/20`。
- 2026-08-11 14:10：人工巡检确认两台 job 均为 `Running/retry=0`，16/16 张卡均有计算进程，瞬时利用率=`35–100%`。Qwen3-32B W4/W3/W2/KV4 formal 进度=`21/20/16/15` of 64，无错误。Qwen3-8B W2A16 MATH-500=`3.40%`，已进入 HumanEval+ generation。
- 2026-08-11 14:15：人工巡检确认两台 job 继续 `Running/retry=0`，node0/node1 均为 8/8 张卡实际计算，瞬时利用率=`38–99%`。Qwen3-32B W4/W3/W2/KV4 formal 进度=`23/21/18/16` of 64；generation/official marker 仍为 `33/60` 与 `9/20`，12 条进行中的推理 lane 均在持续增加样本。
- 2026-08-11 14:20：人工巡检确认两台 job 仍为 `Running/retry=0`，16/16 张卡均有 CUDA 计算进程，瞬时利用率=`35–100%`。Qwen3-32B W4/W3/W2/KV4 formal 进度=`24/22/19/17` of 64。四条未完成 HumanEval+ 样本数为 Qwen3-4B W4/W3/W2=`128/112/112` of 164、Qwen3-8B W2=`96/164`，全部持续前进。
- 2026-08-11 14:25：人工巡检时两台 job 仍为 `Running/retry=0`；node1 GPU5 恰在 Qwen3-4B W4A16 HumanEval+ 完成后释放，已立即回填 Qwen3-0.6B W4A4KV4 HumanEval+。随后 Qwen3-4B W2A16 HumanEval+ 完成，node1 GPU2 继续回填 Qwen3-4B W4A4KV4 HumanEval+。Qwen3-32B W4/W3/W2/KV4 formal 进度=`25/23/20/18` of 64。
- 2026-08-11 14:29：Qwen3-4B W4/W3/W2 HumanEval+ base/plus 官方结果分别为 `47.56/44.51%`、`41.46/38.41%`、`0/0%`；W4 三项推理已全部完成。W3 释放的 node0 GPU7 已回填最后一条未启动任务 Qwen3-8B W4A4KV4 HumanEval+；至此所有当前 checkpoint 可执行的推理子任务都已启动，generation/official marker=`36/60` 与 `12/20`。
- 2026-08-11 14:30：人工巡检确认两台 job 均为 `Running/retry=0`。Qwen3-8B W2A16 HumanEval+ generation 完成并启动 CPU-only 官方评分，generation marker=`37/60`；其 node1 GPU1 当前为唯一空闲卡，因其余非 32B 任务已全部有原子 claim，而 32B 评测尚等待 checkpoint，无可安全重复的任务。Qwen3-32B W4/W3/W2/KV4 formal 进度=`26/24/21/20` of 64，其余 15 张卡均在实际计算。
- 2026-08-11 14:32：Qwen3-8B W2A16 HumanEval+ base/plus 官方结果=`0/0%`，其 GSM8K/MATH-500/HumanEval+ 三项结果已全部完成。official marker=`13/20`。
- 2026-08-11 14:36：人工巡检确认两台 job 均为 `Running/retry=0`，共 15/16 张卡实际计算；node1 GPU1 继续因“所有可运行子任务均已 claim、32B checkpoint 尚未完成”的依赖门禁空闲。Qwen3-32B W4/W3/W2/KV4 formal 进度=`27/26/22/21` of 64；Llama W4A4KV4 GSM8K=`1280/1319`，接近完成。
- 2026-08-11 14:39：Llama W4A4KV4 GSM8K 1319/1319 完成，准确率=`68.92%`。该 run 的 MATH-500 已在另一条任务级 lane 运行，HumanEval+ 也已完成，故原串行 evaluator 正常结束且不重复生成。generation marker=`38/60`。
- 2026-08-11 14:42：人工巡检确认两台 job 均为 `Running/retry=0`，当前 14/16 张卡实际计算；node0 GPU1 因 Llama KV4 GSM8K 完成而释放，node1 GPU1 为先前 Qwen3-8B W2A16 完成后释放，两者均无未 claim 且依赖已就绪的任务。Qwen3-32B W4/W3/W2/KV4 formal 进度=`29/27/24/22` of 64，其余推理 lane 均在运行。
- 2026-08-11 14:47：人巡确认两台 job 继续 `Running/retry=0`，14/16 卡计算，空闲卡仍只是已完成任务后的依赖空窗。Qwen3-32B W4/W3/W2/KV4 formal 进度=`30/28/25/23` of 64。推理近尾进度包括 Qwen3-4B W3 MATH-500=`416/500`、Qwen3-0.6B KV4 MATH-500=`432/500`、HumanEval+=`96/164`，无错误。
- 2026-08-11 14:52：人工巡检确认两台 job 均为 `Running/retry=0`，14/16 卡计算，无新的可运行依赖项。Qwen3-32B W4/W3/W2/KV4 formal 进度=`31/30/26/25` of 64。Qwen3-4B W3 与 Qwen3-0.6B KV4 的 MATH-500 均已达 `464/500`，其余活跃任务亦持续增长。
- 2026-08-11 14:57：人巡确认两台 job 继续 `Running/retry=0`。Qwen3-4B W3A16 MATH-500 完成，准确率=`53.60%`，该 run 三项推理完成；node0 GPU6 释放后因无未 claim 任务而进入依赖空窗，当前 13/16 卡计算。Qwen3-32B W4/W3/W2/KV4 formal 进度=`32/31/27/26` of 64，generation marker=`39/60`。
- 2026-08-11 15:00：Qwen3-0.6B W4A4KV4 HumanEval+ generation 完成，CPU-only 官方评分 base/plus=`0/0%`；该 run 的 MATH-500 仍在最后 `496/500`。generation/official marker=`40/60` 与 `14/20`。
- 2026-08-11 15:03：人巡确认两台 job 均为 `Running/retry=0`。Qwen3-0.6B W4A4KV4 与 Qwen3-4B W2A16 的 MATH-500 均完成，准确率=`0.60/1.20%`，两个 run 的三项推理均已齐全；generation marker=`42/60`。当前 10/16 卡实际计算，其余卡的空闲原因均为可运行任务已穷尽。Qwen3-32B W4/W3/W2/KV4 formal 进度=`34/32/28/27` of 64。
- 2026-08-11 15:08：人巡确认两台 job 继续 `Running/retry=0`，10/16 卡实际计算，无可补的未 claim 任务。Qwen3-32B W4/W3/W2/KV4 formal 进度=`35/33/30/28` of 64。Qwen3-4B/Qwen3-8B KV4 HumanEval+ 分别为 `160/164` 与 `144/164`，均处于尾批次。
- 2026-08-11 15:12：人巡确认两台 job 均为 `Running/retry=0`。Qwen3-4B KV4 HumanEval+ generation 完成，官方 base/plus=`25.61/24.39%`；Qwen3-8B KV4 HumanEval+ 同期完成生成并进入 CPU 评分。巡检时 9/16 卡计算，Qwen3-32B W4/W3/W2/KV4 formal 进度=`36/34/31/29` of 64。
- 2026-08-11 15:15：Qwen3-8B KV4 HumanEval+ 官方 base/plus=`30.49/26.83%`；generation/official marker=`44/60` 与 `16/20`。非 32B 剩余任务为 Llama KV4 MATH-500、Qwen3-4B KV4 GSM8K/MATH-500、Qwen3-8B KV4 MATH-500。
- 2026-08-11 15:18：人巡确认两台 job 均为 `Running/retry=0`。Qwen3-4B KV4 GSM8K 完成，准确率=`76.12%`；非 32B 只剩三条 MATH-500，generation marker=`45/60`。当前 7/16 卡实际计算（四条 32B formal+三条 MATH），Qwen3-32B W4/W3/W2/KV4 formal 进度=`37/35/32/30` of 64。
- 2026-08-11 15:23：人巡确认两台 job 继续 `Running/retry=0`，当前 7/16 卡运行全部可执行任务。Qwen3-32B W4/W3/W2/KV4 formal 进度=`38/37/33/32` of 64。Llama/Qwen3-4B/Qwen3-8B KV4 MATH-500 进度=`368/352/272` of 500，三条均正常增长。
- 2026-08-11 15:27：人巡确认两台 job 均为 `Running/retry=0`，7/16 卡运行全部可执行任务。Qwen3-32B W4/W3/W2/KV4 formal 进度=`40/38/34/33` of 64。Llama/Qwen3-4B/Qwen3-8B KV4 MATH-500 进度=`400/368/288` of 500，未发现错误或停滞。
- 2026-08-11 15:33：人巡确认两台 job 均为 `Running/retry=0`，7/16 卡运行全部可执行任务。Qwen3-32B W4/W3/W2/KV4 formal 进度=`41/39/35/34` of 64。Llama/Qwen3-4B/Qwen3-8B KV4 MATH-500 进度=`432/384/304` of 500，所有进程与样本计数正常。
- 2026-08-11 15:38：人巡确认两台 job 均为 `Running/retry=0`，7/16 卡运行全部可执行任务。Qwen3-32B W4/W3/W2/KV4 formal 进度=`42/40/37/35` of 64。Llama/Qwen3-4B/Qwen3-8B KV4 MATH-500 进度=`464/400/336` of 500，无错误。
- 2026-08-11 15:44：人巡确认两台 job 均为 `Running/retry=0`，7/16 卡运行全部可执行任务。Qwen3-32B W4/W3/W2/KV4 formal 进度=`43/41/38/36` of 64。Llama/Qwen3-4B/Qwen3-8B KV4 MATH-500 进度=`480/432/352` of 500，Llama 进入最后 20 条。
- 2026-08-11 15:49：人巡确认两台 job 均为 `Running/retry=0`。Llama KV4 MATH-500 完成，准确率=`22.60%`，该 run 三项推理已齐全；generation marker=`46/60`。Qwen3-4B/8B KV4 MATH-500 进度=`448/368` of 500。Qwen3-32B W4/W3/W2/KV4 formal 进度=`45/43/39/38` of 64；Llama lane 退出后为 6/16 卡实际计算。
- 2026-08-11 15:55：人巡确认两台 job 均为 `Running/retry=0`，当前 6/16 卡运行全部已通过依赖门禁的任务；其余卡无未 claim 且 checkpoint 就绪的任务，不是 OOM 或任务失败。Qwen3-32B W4/W3/W2/KV4 formal 进度=`46/44/40/39` of 64。Qwen3-4B/8B KV4 MATH-500 进度=`480/384` of 500，均持续前进。
- 2026-08-11 16:00：人巡确认两台 job 均为 `Running/retry=0`，6/16 卡继续运行全部可执行任务。Qwen3-32B W4/W3/W2/KV4 formal 进度=`47/45/42/40` of 64；Qwen3-4B/8B KV4 MATH-500 进度=`496/416` of 500。两条 MATH generation 仍在增长，未发现错误。
- 2026-08-11 16:05：人巡确认两台 job 均为 `Running/retry=0`。Qwen3-4B KV4 MATH-500 完成，准确率=`44.60%`，该 run 三项推理已齐全；generation marker=`47/60`。Qwen3-8B KV4 MATH-500=`432/500`。Qwen3-32B W4/W3/W2/KV4 formal 进度=`49/46/43/41` of 64；5 条依赖就绪的 GPU lane 全部在运行，W3 在采样瞬间处于层间 CPU 边界。
- 2026-08-11 16:11：人巡确认两台 job 均为 `Running/retry=0`，5 条依赖就绪的 lane 全部存活。Qwen3-32B W4/W3/W2/KV4 formal 进度=`50/48/44/43` of 64；Qwen3-8B KV4 MATH-500=`448/500`。GPU7 在采样瞬间处于层间边界但保有正式量化上下文，日志继续推进，无失败。
- 2026-08-11 16:16：人巡确认两台 job 均为 `Running/retry=0`，5 条可执行 lane 全部在跑。Qwen3-32B W4/W3/W2/KV4 formal 进度=`51/49/45/44` of 64；四份当前 `glbsz2` 执行日志中的 `Traceback/OOM/CUDA error` 计数均为 0。Qwen3-8B KV4 MATH-500=`464/500`，继续正常增长。
- 2026-08-11 16:21：人巡确认两台 job 均为 `Running/retry=0`，5 条可执行 lane 仍全部存活。Qwen3-32B W4/W3/W2/KV4 formal 进度=`52/50/46/45` of 64；Qwen3-8B KV4 MATH-500=`480/500`。node1 GPU7 采样瞬间处于层间边界但保留约 `119116 MiB` 上下文，日志未停滞。
- 2026-08-11 16:26：人巡确认两台 job 均为 `Running/retry=0`。Qwen3-8B KV4 MATH-500 完成，准确率=`46.00%`，至此 16 个非 32B checkpoint 的三项推理全部完成；generation/official marker=`48/60` 与 `16/20`。Qwen3-32B W4/W3/W2/KV4 formal 进度=`53/51/48/46` of 64；当前 4/16 卡运行所有依赖就绪任务，其余任务严格等待 32B checkpoint。
- 2026-08-11 16:31：人巡确认两台 job 均为 `Running/retry=0`，四张正式量化卡均为 `99–100%` 瞬时利用率，其余卡处于 checkpoint 依赖空窗。Qwen3-32B W4/W3/W2/KV4 formal 最新进度=`55/52/49/47` of 64，四组尚未发布 `formal_success.json`，无任务被漏派。
- 2026-08-11 16:36：人巡确认两台 job 均为 `Running/retry=0`，四张正式量化卡继续保持 `99–100%` 瞬时利用率。Qwen3-32B W4/W3/W2/KV4 formal 进度=`56/54/50/48` of 64，按当前层速率首个 checkpoint 仍预计约 `17:11–17:15` 发布。
- 2026-08-11 16:41：人巡确认两台 job 均为 `Running/retry=0`。Qwen3-32B W4/W3/W2/KV4 formal 进度=`57/55/51/50` of 64；node0 两张 formal 卡为 `100/100%`，node1 为 `99/41%`（后者处于层内阶段切换且显存上下文约 `105256 MiB`），四条日志均持续推进。
- 2026-08-11 16:46：人巡确认两台 job 均为 `Running/retry=0`，四张正式量化卡为 `99–100%` 瞬时利用率。Qwen3-32B W4/W3/W2/KV4 formal 进度=`58/56/52/51` of 64，尚无 `formal_success.json`；W4 剩余 6 层，日志 ETA 约 25.5 分钟。
- 2026-08-11 16:51：人巡确认两台 job 均为 `Running/retry=0`，四张正式量化卡均为 `100%` 瞬时利用率。Qwen3-32B W4/W3/W2/KV4 formal 进度=`59/57/53/52` of 64；W4 剩余 5 层，日志 ETA 约 22 分钟。
- 2026-08-11 16:56：人巡确认两台 job 均为 `Running/retry=0`，四张正式量化卡均为 `100%` 瞬时利用率。Qwen3-32B W4/W3/W2/KV4 formal 进度=`60/58/55/53` of 64；W4 剩余 4 层，日志 ETA 约 17.5 分钟。
- 2026-08-11 17:01：人巡确认两台 job 均为 `Running/retry=0`，四张正式量化卡均为 `100%` 瞬时利用率。Qwen3-32B W4/W3/W2/KV4 formal 进度=`61/59/56/54` of 64；W4 剩余 3 层，日志 ETA 约 13 分钟。
- 2026-08-11 17:06：人巡确认两台 job 均为 `Running/retry=0`，四张正式量化卡保持 `99–100%`。Qwen3-32B W4/W3/W2/KV4 formal 进度=`63/61/57/55` of 64；W4 已进入最后 1 层并执行冻结的 bounded final-KL projection，尚未发布 checkpoint marker。
- 2026-08-11 17:11：人巡确认两台 job 均为 `Running/retry=0`，四张 formal 卡为 `99–100%`。Qwen3-32B W4/W3/W2/KV4 最新进度=`63/62/58/57` of 64；W4 的最后一层仍在 bounded final-KL projection，GPU0 约 `113496 MiB` 且 `99%`，无停滞或失败迹象。
- 2026-08-11 17:16：Qwen3-32B W4A16 64/64 层、62 GiB checkpoint 写入与 60 秒稳定性门禁全部通过，`formal_success.json` 已发布；正式量化实测=`4.673 GPU·h`，checkpoint=`65,527,268,396 B`。首次三条 tail launcher 因 SSH 默认目录 `/output` 无法导入 `experiments` 而在模块导入前 rc=1，未创建 task claim、未占 GPU、未生成样本；切换到仓库目录并先做 import 预检后，GSM8K/MATH-500/HumanEval+ 已分别在 node0 GPU0/1/2 重提并存活。
- 2026-08-11 17:21：人巡确认两台 job 均为 `Running/retry=0`。W4A16 三项推理已完成 62 GiB checkpoint 加载并在 node0 GPU0/1/2 建立 `64–73 GiB` 上下文；当前仍在首批次准备，generation=`0/0/0`。node0 GPU3 的 W3 formal 已到 63/64 并进入 final-KL projection；W2/KV4 formal=`60/59` of 64。当前 6/16 张卡覆盖全部依赖就绪任务。
- 2026-08-11 17:26：人巡确认两台 job 均为 `Running/retry=0`。Qwen3-32B W3A16 checkpoint 与稳定性门禁通过，正式量化=`4.718 GPU·h`、checkpoint=`65,527,268,396 B`；三项推理已在 node0 GPU3/4/5 启动。W4A16 GSM8K/MATH-500/HumanEval+ generation=`96/16/16`；W2/KV4 formal=`61/60` of 64。派发后共有 8/16 条可执行 GPU lane，node0 6 条推理、node1 2 条 formal。
- 2026-08-11 17:31：人巡确认两台 job 均为 `Running/retry=0`。node0 GPU0–5 的六条 W4/W3 推理 lane 均已进入 `91–95%` GPU 计算；W4 generation=`160/32/32`，W3 仍在首批。node1 W2 formal 已到 63/64 并进入 final-KL projection，KV4=`61/64`；两张 formal 卡=`100/99%`。当前 8/16 卡覆盖全部依赖就绪任务。
- 2026-08-11 17:36：人巡确认两台 job 均为 `Running/retry=0`。node0 六条推理 lane 均在计算，W4 generation=`224/48/64`，W3=`96/16/16`。node1 W2/KV4 formal 均已到 63/64 并进入 final-KL projection，两卡=`100/99%`；当前 8/16 卡覆盖全部依赖就绪任务。
- 2026-08-11 17:41：人巡确认两台 job 均为 `Running/retry=0`。node0 六条推理 lane=`89–94%`，W4 generation=`320/80/80`、W3=`160/32/32`。node1 W2 已完成 64/64 并转入 checkpoint 保存/稳定性门禁（GPU7 0%、约 `21526 MiB` 上下文），KV4 final projection 仍为 GPU6 `100%`；W2 marker 尚未发布。
- 2026-08-11 17:43：Qwen3-32B W2A16 checkpoint 与稳定性门禁通过，正式量化=`4.800 GPU·h`、checkpoint=`65,527,268,396 B`；三项推理已在 node1 GPU0/1/2 启动并建立原子 claim。此时 19/20 checkpoint 完成，KV4 final projection 继续在 node1 GPU6 运行。
- 2026-08-11 17:46：人巡确认两台 job 均为 `Running/retry=0`。node0 六路推理持续计算，W4 generation=`384/96/96`、W3=`256/64/64`；node1 W2 三路已完成加载并在 GPU0/1/2 建立上下文，首批仍在生成。KV4 已完成 64/64 并进入 checkpoint 保存/稳定性门禁（GPU6 0%、约 `21506 MiB` 上下文），marker 尚未发布。
- 2026-08-11 17:49：Qwen3-32B KV4 checkpoint 与稳定性门禁通过，正式量化=`4.796 GPU·h`、checkpoint=`65,527,280,684 B`；三项推理已在 node1 GPU3/4/5 启动并建立原子 claim。至此 20/20 正式 checkpoint 全部完成，12 条 32B 推理 generation 子任务均已并行派发。
- 2026-08-11 17:51：人巡确认两台 job 均为 `Running/retry=0`。node0 六路为 `93–97%`，W4 generation=`448/112/112`、W3=`320/80/80`；node1 W2 三路为 `91–93%`，generation=`64/16/16`。KV4 三路仍在并发加载 checkpoint，claim/父进程存活但尚未建立 GPU 上下文；当前 9 张卡实算，加载完成后预计增至 12 张。
- 2026-08-11 17:56：人巡确认两台 job 均为 `Running/retry=0`。12 条 32B 推理 lane 已全部进入 GPU 计算：node0 GPU0–5=`75–97%`，node1 GPU0–5=`50–97%`。W4 generation=`544/144/128`、W3=`384/96/96`、W2=`128/32/32`；KV4 三路完成加载但仍在首批。GPU6/7 无未 claim 任务可补。
- 2026-08-11 18:01：人巡确认两台 job 均为 `Running/retry=0`，node0/node1 GPU0–5 的 12 条推理 lane 瞬时利用率=`63–98%`。W4 generation=`608/160/160`、W3=`480/112/112`、W2=`192/48/48`、KV4=`32/0/0`；所有计数持续增长，W4 HumanEval+ 进入最后 4 条。
- 2026-08-11 18:04：Qwen3-32B W4A16 HumanEval+ generation 164/164 完成，generation marker=`49/60`；node0 GPU2 正常释放。CPU-only EvalPlus 官方评分已立即启动，GPU2 因其余 11 条 generation 均已有原子 claim 而进入依赖空窗。
- 2026-08-11 18:05：Qwen3-32B W4A16 EvalPlus 官方评分完成，HumanEval+ base/plus=`51.83/49.39%`（`85/81` of 164），official marker=`17/20`；主表已回填。
- 2026-08-11 18:06：人巡确认两台 job 均为 `Running/retry=0`。W4 GSM8K/MATH-500=`704/176`；W3=`544/144/144`，W2=`288/80/64`，KV4=`96/16/16`。除已完成的 W4 HumanEval+ 外，其余 11 条 generation lane 全部在 GPU0–5 上持续计算；无未 claim 任务可补到空卡。
- 2026-08-11 18:11：人巡确认两台 job 均为 `Running/retry=0`。W4 GSM8K/MATH-500=`768/192`；W3=`640/160/160`，W2=`352/96/80`，KV4=`128/32/32`。11 条未完成 generation lane 保持计算，W3 HumanEval+ 进入最后 4 条。
- 2026-08-11 18:13：Qwen3-32B W3A16 HumanEval+ generation 164/164 完成并通过原子 marker，generation marker=`50/60`；node0 GPU5 释放，CPU-only EvalPlus 官方评分已立即启动。其余 10 条 generation lane 均已有独立 claim 并继续运行。
- 2026-08-11 18:15：Qwen3-32B W3A16 EvalPlus 官方评分完成，HumanEval+ base/plus=`46.95/45.12%`（`77/74` of 164），official marker=`18/20`；主表已回填。
- 2026-08-11 18:17：人巡确认两台 job 均为 `Running/retry=0`。Qwen3-32B W4 GSM8K/MATH-500=`864/208`，W3=`736/176`，W2=`480/112`、HumanEval+=`112/164`，KV4=`192/48/48`。当前 node0 GPU0/1/3/4 与 node1 GPU0–5 共 `10/16` 卡持续生成；其余 6 卡对应已经完成的 W4/W3 HumanEval+ 或没有未 claim 的独立任务，未发现任务掉线。
- 2026-08-11 18:22：人巡确认两台 job 继续为 `Running/retry=0`，10 条活跃 lane 均保有约 `70–94%` 瞬时计算。W4 GSM8K/MATH-500=`928/224`，W3=`800/192`，W2=`512/128/128`，KV4=`224/48/48`；所有组相对上一轮均已有新批次落盘，10 份执行日志的 `Traceback/OOM/CUDA error` 计数均为 0。
- 2026-08-11 18:27：人巡确认两台 job 均为 `Running/retry=0`。node0 四条活跃 lane=`86–94%`，node1 六条=`49–94%`；W4=`1024/256`、W3=`896/208`、W2=`608/160/144`、KV4=`288/64/64`，全部继续正常增长。W2 HumanEval+ 剩余 20 条，完成后将立即触发 CPU-only 官方评分。
- 2026-08-11 18:31：Qwen3-32B W2A16 HumanEval+ generation `164/164` 完成，任务 rc=0 且原子 marker 已发布；node1 GPU2 正常释放，CPU-only EvalPlus 官方评分已在同机立即启动。同期 W4=`1088/272`、W3=`960/240`、W2 GSM8K/MATH-500=`672/176`、KV4=`320/64/64`。
- 2026-08-11 18:33：人巡确认两台 job 均为 `Running/retry=0`，余下 9 条 generation lane 均有实际 GPU 计算；W4=`1088/272`、W3=`960/240`、W2=`704/176`、KV4=`320/80/80`。Qwen3-32B W2A16 EvalPlus 官方评分同时完成，HumanEval+ base/plus=`3.66/3.66%`（`6/6` of 164），official marker=`19/20`；主表已回填。
- 2026-08-11 18:38：人巡确认两台 job 均为 `Running/retry=0`，9 条剩余 generation lane 全部存活并有实际计算（node0=`93–97%`，node1 非 KV 两条=`94–98%`、KV 三条=`49–95%`）。W4=`1184/288`、W3=`1056/256`、W2=`768/192`、KV4=`384/80/80`；全部执行日志继续无 `Traceback/OOM/CUDA error`。
- 2026-08-11 18:43：人巡确认两台 job 均为 `Running/retry=0`，9 条 generation lane 持续计算；node0 四条=`90–94%`，node1 W2 两条=`92–98%`、KV4 三条=`62–78%`。W4=`1248/320`、W3=`1120/272`、W2=`864/208`、KV4=`416/96/96`，未发现错误或停滞。
- 2026-08-11 18:46：Qwen3-32B W4A16 GSM8K `1319/1319` generation、任务级评分与原子 marker 全部完成，准确率=`93.48%`；主表已回填，node0 GPU0 正常释放。该 run 只剩 MATH-500 后即可完成三项推理。
- 2026-08-11 18:48：人巡确认两台 job 均为 `Running/retry=0`，剩余 8 条 generation lane 全部有实际 GPU 计算。W4 MATH-500=`336/500`，W3=`1216/288`，W2=`928/240`，KV4=`448/112/112`；已完成任务释放后的 8 张空卡无未 claim 的独立任务可安全回填。
- 2026-08-11 18:53：人巡确认两台 job 仍为 `Running/retry=0`，8 条剩余 lane 全部有实际计算。W4 MATH-500=`352/500`，W3=`1280/320`，W2=`1024/256`，KV4=`512/112/112`；W3 GSM8K 剩余 39 条，未发现错误或停滞。
- 2026-08-11 18:55：Qwen3-32B W3A16 GSM8K `1319/1319` generation、任务级评分与 marker 完成，准确率=`93.56%`；主表已回填，node0 GPU3 正常释放。W3 只剩 MATH-500 后即可完成三项推理。
- 2026-08-11 18:58：人巡确认两台 job 均为 `Running/retry=0`，7 条剩余 generation lane 全部有实际计算：node0 两条 MATH=`94–97%`，node1 五条=`66–97%`。W4 MATH-500=`368`，W3 MATH-500=`336`，W2=`1088/272`，KV4=`544/128/128`；无错误或停滞。
- 2026-08-11 19:03：人巡确认两台 job 均为 `Running/retry=0`，7 条剩余 lane 全部有实际计算。W4 MATH-500=`400`，W3 MATH-500=`352`，W2=`1184/288`，KV4=`608/144/144`；执行日志无 `Traceback/OOM/CUDA error`。KV4 HumanEval+ 剩余 20 条，完成后立即官方评分。
- 2026-08-11 19:08：人巡确认两台 job 均为 `Running/retry=0`，7 条剩余 lane 继续实际计算。W4/W3 MATH-500=`416/368`，W2=`1248/304`，KV4=`640/144/144`；KV4 HumanEval+ GPU5 worker PID 58230 状态=`Rl`、约 `81% GPU`，确认尾批仍在活跃计算而非卡死。
- 2026-08-11 19:13：人巡确认两台 job 均为 `Running/retry=0`。Qwen3-32B W2A16 GSM8K `1319/1319` generation、任务级评分与 marker 完成，准确率=`36.62%`，node1 GPU0 释放；主表已回填。剩余 W4/W3/W2 MATH-500=`432/400/336`，KV4=`672/160/160`，6 条 lane 均实际计算。
- 2026-08-11 19:18：人巡确认两台 job 均为 `Running/retry=0`。W4/W3/W2 MATH-500=`464/416/352`，KV4 GSM8K/MATH-500=`736/176`；KV4 HumanEval+ generation 已达 `164/164`，等待父进程发布原子 marker 后再启动官方评分，未越过门禁。
- 2026-08-11 19:20：Qwen3-32B KV4 HumanEval+ generation marker 与 CPU-only EvalPlus 官方评分全部完成，base/plus=`35.98/35.37%`（`59/58` of 164），official marker=`20/20`；主表已回填。至此全部 20 组 HumanEval+ 官方评分完成。
- 2026-08-11 19:23：人巡确认两台 job 均为 `Running/retry=0`，最后 5 条 generation lane 全部有实际计算。W4/W3/W2 MATH-500=`480/432/368`，KV4 GSM8K/MATH-500=`768/192`；W4 MATH-500 剩余 20 条，日志无错误。
- 2026-08-11 19:29：人巡确认两台 job 均为 `Running/retry=0`。Qwen3-32B W4A16 MATH-500 `500/500` generation、任务级评分与 marker 完成，准确率=`62.80%`，node0 GPU1 释放；该 run 三项推理全部完成，主表已回填。其余 W3/W2 MATH-500=`464/384`，KV4 GSM8K/MATH-500=`832/192`，4 条 lane 均计算。
- 2026-08-11 19:34：人巡确认两台 job 均为 `Running/retry=0`，最后 4 条 generation lane 均保持实际计算。W3/W2 MATH-500=`480/416`，KV4 GSM8K/MATH-500=`864/208`；四份日志无错误，W3 MATH-500 剩 20 条。
- 2026-08-11 19:39：人巡确认两台 job 均为 `Running/retry=0`。Qwen3-32B W3A16 MATH-500 `500/500` generation、任务级评分与 marker 完成，准确率=`61.80%`；该 run 三项推理全部完成，主表已回填，node0 8 卡全部进入依赖空窗。剩余 node1 三条任务为 W2 MATH-500=`432`、KV4 GSM8K/MATH-500=`896/224`。
- 2026-08-11 19:44：人巡确认两台 job 均为 `Running/retry=0`。node1 最后三条 generation lane 均有实际计算（`69–94%`），W2 MATH-500=`464`、KV4 GSM8K/MATH-500=`960/224`；node0 因最终审计/质量评测依赖这三条结果而处于严格门禁空窗。
- 2026-08-11 19:50：人巡确认两台 job 均为 `Running/retry=0`。node1 最后三条 lane=`81–96%`，W2 MATH-500=`480`、KV4 GSM8K/MATH-500=`992/240`；W2 剩余 20 条，node0 继续等待最终审计门禁。
- 2026-08-11 19:55：人巡确认两台 job 均为 `Running/retry=0`。Qwen3-32B W2A16 MATH-500 `500/500` generation、任务级评分与 marker 完成，准确率=`10.00%`；该 run 三项推理完成，主表已回填，node1 GPU1 释放。现在仅剩 KV4 GSM8K/MATH-500=`1056/256` 两条 lane。
- 2026-08-11 20:00：人巡确认两台 job 均为 `Running/retry=0`。最后两条 KV4 generation lane 在 node1 GPU3/4 以 `63/54%` 瞬时利用率运行，GSM8K/MATH-500=`1088/272`；两份日志无 `Traceback/OOM/CUDA error`。node0 继续等待最终审计门禁。
- 2026-08-11 20:05：人巡确认两台 job 均为 `Running/retry=0`。KV4 GSM8K/MATH-500=`1120/272`，node1 GPU3/4 瞬时利用率=`84/90%`；两条任务均保持实际计算，node0 继续等待最终审计门禁。
- 2026-08-11 20:10：人巡确认两台 job 均为 `Running/retry=0`。KV4 GSM8K/MATH-500=`1184/288`，node1 GPU3/4 瞬时利用率=`50/87%`；两个 worker 均保有约 `74–76 GiB` 上下文并继续计算。
- 2026-08-11 20:15：人巡确认两台 job 均为 `Running/retry=0`。KV4 GSM8K/MATH-500=`1216/304`，node1 GPU3/4 瞬时利用率=`59/49%`；两条任务相对上一轮均有新批次落盘，未发现错误。
- 2026-08-11 20:20：人巡确认两台 job 均为 `Running/retry=0`。KV4 GSM8K/MATH-500=`1248/304`，node1 GPU3/4 瞬时利用率=`80/88%` 且保有上下文；两个长批次尚未整批落盘，无错误或停滞证据。
- 2026-08-11 20:25：人巡确认两台 job 均为 `Running/retry=0`。KV4 GSM8K/MATH-500=`1280/320`，node1 GPU3/4 瞬时利用率=`93/70%`；GSM8K 剩余 39 条，两个 worker 均持续计算。
- 2026-08-11 20:29：Qwen3-32B KV4 GSM8K `1319/1319` generation、任务级评分与 marker 完成，准确率=`93.18%`；主表已回填，node1 GPU3 释放。现在只剩 KV4 MATH-500 一条 generation 任务。
- 2026-08-11 20:31：人巡确认两台 job 均为 `Running/retry=0`。最后一条 KV4 MATH-500=`336/500`，node1 GPU4 保有约 `76.5 GiB` 上下文并以 `49%` 瞬时利用率计算；其余 15 卡等待最终审计/质量门禁。
- 2026-08-11 20:36：人巡确认两台 job 均为 `Running/retry=0`。最后一条 KV4 MATH-500 仍为 `336/500`，但 node1 GPU4 瞬时利用率=`93%`、显存约 `76.5 GiB`，执行日志错误计数=0；当前长批仍在实际计算。
- 2026-08-11 20:41：人巡确认两台 job 均为 `Running/retry=0`。最后一条 KV4 MATH-500=`352/500`，node1 GPU4 瞬时利用率=`84%`、显存约 `76.5 GiB`；其余 15 卡继续等待最终审计/质量门禁。
- 2026-08-11 20:46：人巡确认两台 job 均为 `Running/retry=0`。最后一条 KV4 MATH-500=`368/500`，node1 GPU4 瞬时利用率=`64%`、显存约 `76.5 GiB`；剩余 132 条，任务继续稳定计算。
- 2026-08-11 20:52：人巡确认两台 job 均为 `Running/retry=0`。最后一条 KV4 MATH-500=`384/500`，node1 GPU4 瞬时利用率=`46%`、显存约 `76.5 GiB`；剩余 116 条，任务继续稳定计算。
- 2026-08-11 20:57：人巡确认两台 job 均为 `Running/retry=0`。最后一条 KV4 MATH-500 仍为 `384/500`，但 GPU4=`84%`、显存约 `76.5 GiB`，worker PID 58229 状态=`Rl`、CPU=`127%`，日志错误计数=0；确认当前批仍在真实计算。
- 2026-08-11 21:02：人巡确认两台 job 均为 `Running/retry=0`。最后一条 KV4 MATH-500=`400/500`，node1 GPU4 瞬时利用率=`62%`、显存约 `76.5 GiB`；剩余 100 条，任务继续计算。
- 2026-08-11 21:07：人巡确认两台 job 均为 `Running/retry=0`。最后一条 KV4 MATH-500=`416/500`，node1 GPU4 瞬时利用率=`51%`、显存约 `76.5 GiB`；剩余 84 条，任务继续计算。
- 2026-08-11 21:12：人巡确认两台 job 均为 `Running/retry=0`。最后一条 KV4 MATH-500 仍为 `416/500`，node1 GPU4 瞬时利用率=`89%`、显存约 `76.5 GiB`，日志错误计数=0；当前长批仍在实际计算。
- 2026-08-11 21:17：人巡确认两台 job 均为 `Running/retry=0`。最后一条 KV4 MATH-500=`432/500`，node1 GPU4 瞬时利用率=`68%`、显存约 `76.5 GiB`；剩余 68 条，任务继续计算。
- 2026-08-11 21:22：人巡确认两台 job 均为 `Running/retry=0`。最后一条 KV4 MATH-500=`448/500`，node1 GPU4 瞬时利用率=`54%`、显存约 `76.5 GiB`；剩余 52 条，任务继续计算。
- 2026-08-11 21:27：人巡确认两台 job 均为 `Running/retry=0`。最后一条 KV4 MATH-500 仍为 `448/500`，node1 GPU4 瞬时利用率=`93%`、显存约 `76.5 GiB`；当前长批仍在真实计算，无失败迹象。
- 2026-08-11 21:33：人巡确认两台 job 均为 `Running/retry=0`。最后一条 KV4 MATH-500=`464/500`，node1 GPU4 瞬时利用率=`74%`、显存约 `76.5 GiB`；剩余 36 条，任务继续计算。
- 2026-08-11 21:39：人巡确认两台 job 均为 `Running/retry=0`。最后一条 KV4 MATH-500=`480/500`，node1 GPU4 瞬时利用率=`67%`、显存约 `76.5 GiB`；剩余 20 条，任务继续计算尾部批次。
- 2026-08-11 21:47：人巡确认两台 job 均为 `Running/retry=0`。最后一条 KV4 MATH-500=`496/500`，node1 GPU4 瞬时利用率=`41%`、显存约 `76.5 GiB`，worker PID 58229 状态=`Rl`、CPU=`121%`；最后 4 条仍在实际生成。
- 2026-08-11 21:51：Qwen3-32B KV4 MATH-500 `500/500` generation、任务级评分与 marker 完成，准确率=`62.20%`，GPU4 正常释放；该 run 三项推理完成，主表已回填。至此 `60/60` generation 与 `20/20` EvalPlus 官方评分全部完成，两机 16 卡均已清空，立即进入最终审计。
- 2026-08-11 21:52：最终 fail-closed 审计 rc=0，计数=`20/20 formal`、`60/60 generation`、`20/20 official`，audit fingerprint=`9ff8b504bc75c03ee755f0dd3cf6bb09424b14f004b5b330775826570e3d291d`；指纹已写入质量门禁与对应测试。
- 2026-08-11 21:53：门禁测试 `32 passed`；20 组质量计划生成成功，quality plan fingerprint=`4c187d78a58dc0c6c5973035b2b93777447e799e4c8a599e67b3fce20ba4b568`。node0/node1 两个质量调度器已同时启动，每机负责 10 个 checkpoint 并使用 8 卡动态回填。
- 2026-08-11 21:55：人巡确认两台 job 均为 `Running/retry=0`。两边质量调度器各有 8 个 `_worker`，共 `16/16` 卡均已建立 attempt001/原子 claim；当前 16 组状态=`running`，4 个 Qwen3-0.6B 组按 largest-first 调度留作完成后的自动回填。GPU 正处于并发 checkpoint/参考模型加载阶段。
- 2026-08-11 21:58：人巡确认两台 job 均为 `Running/retry=0`，两边仍各有 8 个质量 worker，状态=`0 succeeded / 16 running / 4 pending`。16 卡显存均已建立上下文或加载上下文（约 `0.6–21.3 GiB`），部分 lane 开始计算；16 份 attempt 日志错误计数均为 0。
- 2026-08-11 22:03：人巡确认两台 job 均为 `Running/retry=0`，`0 succeeded / 16 running / 4 pending`、failure=`0`。16 个 `_worker` 及其 `realq.ptq` 子进程全部存活；node0 8 卡已全部建立模型上下文（Qwen3-32B 两卡约 `69.2 GiB`），node1 的若干 lane 正处于 CPU tokenization/QA 任务切换，GPU 利用率因此呈间歇性。日志中的 Git `safe.directory` 文本发生在单项 QA 完成后的元信息采集，后续任务继续运行，确认是无害告警而非任务失败。
- 2026-08-11 22:04：首批 3 组质量评测通过单组审计并发布原子 marker：Llama W2（KL/PPL/QA Avg=`0.56603551/12.5885/46.59`）、Qwen3-4B W3（`0.15423504/14.1615/61.12`）、Qwen3-4B W4（`0.04346002/13.6937/62.74`）。failure=`0`，释放的 lane 已自动回填 Qwen3-0.6B W3/W4，状态变为 `3 succeeded / 15 running / 2 pending`。
- 2026-08-11 22:08：人巡确认两台 job 均为 `Running/retry=0`，状态=`7 succeeded / 12 running / 1 pending / 0 failed`。新增完成 Llama KV4（KL/PPL/QA Avg=`0.19640522/8.6909/61.72`）、Qwen3-4B W2（`0.78348303/20.9588/40.03`）、Qwen3-8B W2（`0.65187836/15.0857/44.11`）与 Qwen3-8B KV4（`0.26469281/11.4523/62.72`）。node0 已完成 6 个固定归属组，剩余 4 组全部在跑；node1 保持 8 条 active，最后一个 Qwen3-0.6B KV4 等其首张卡释放后自动回填。固定节点调度下没有遗漏任务，日志错误计数仍为 0。
- 2026-08-11 22:13：人巡确认两台 job 均为 `Running/retry=0`，状态=`14 succeeded / 6 running / 0 pending / 0 failed`。新增完成 Qwen3-0.6B W4/W3（KL/PPL/QA Avg=`0.06334694/21.7094/45.88`、`0.22560166/24.6261/40.89`）、Llama W4/W3（`0.02496048/7.3821/66.25`、`0.09564172/7.9285/65.01`）、Qwen3-4B KV4（`0.32651350/15.0027/56.77`）以及 Qwen3-8B W4/W3（`0.03278381/9.8711/67.34`、`0.11800054/10.3457/65.23`）。剩余任务仅为四个 Qwen3-32B 组及 Qwen3-0.6B W2/KV4；node0 两条 32B lane 瞬时均为 `99%`，node1 四条 lane 在 QA 子项边界呈间歇负载，进程与显存上下文正常。
- 2026-08-11 22:18：人巡确认两台 job 均为 `Running/retry=0`，状态=`18 succeeded / 2 running / 0 pending / 0 failed`。新增完成 Qwen3-0.6B W2（KL/PPL/QA Avg=`1.57509351/82.0018/31.78`）与 Qwen3-32B W4/W3/W2（`0.04527105/7.6046/71.97`、`0.13114163/7.7452/70.90`、`0.51529497/10.1171/57.60`）。现在只剩 node1 上的 Qwen3-32B KV4 与 Qwen3-0.6B KV4；node0 的固定 10 组已全部完成并清卡。
- 2026-08-11 22:23：人巡确认两台 job 均为 `Running/retry=0`，状态=`19 succeeded / 1 running / 0 failed`。Qwen3-0.6B KV4 完成，KL/PPL/QA Avg=`0.98816073/46.0317/34.76`。最后一组 Qwen3-32B KV4 已越过 C-Eval 数据预处理阶段，在 node1 GPU1 运行 BoolQ=`1601/6540`，瞬时利用率=`98%`、显存约 `99.3 GiB`；之后只剩 OBQA 与 SIQA。
- 2026-08-11 22:26：Qwen3-32B KV4 最后一项 SIQA 完成，子进程 rc=0、单组审计与原子 marker 均通过；正式 KL/PPL/QA Avg=`0.25566542/8.2580/68.71`。至此质量评测=`20/20`、QA 单项=`200/200`、failure=`0`，两机 GPU 全部释放。
- 2026-08-11 22:27：质量 fail-closed 总审计 rc=0，计数=`20 runs / 20 PPL / 200 QA`，quality audit fingerprint=`0ecc1f4b3528d83698cbcfeea96b81f69a745c3a447e4d7c3dc8ee690d950a88`；主表与十项 QA 明细均从该审计 JSON 回填。
- 2026-08-11 22:29：主表 20 行的 KL/PPL/QA Avg 与十项 QA 明细 20 行均逐字段对比审计 JSON，无差异、无占位符；相关门禁测试=`32 passed`，`git diff --check` 通过。
- 2026-08-11 22:36：确认两机 16 卡均为 `0 MiB/0%` 且无 CUDA 进程后，执行 `mmctl jobs abort j-tc64iczh8w` 与 `mmctl jobs abort j-jd9dtz5a1h`，两条命令均成功；控制面复核两条 debug/sleep job 均为 `Aborted/retry=0`。共享盘 checkpoint、日志、generation、正式审计与质量审计产物均保留。
