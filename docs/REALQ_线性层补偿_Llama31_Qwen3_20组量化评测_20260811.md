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
| WikiText2 质量 | full-vocab FP32 KL/PPL | 待运行 |
| 十项 QA | PIQA、HellaSwag、ARC-E、ARC-C、WinoGrande、LAMBADA、C-Eval、BoolQ、OBQA、SIQA | 待运行 |
| 三项推理 | GSM8K、MATH-500、HumanEval+；thinking-on、greedy、n=1、全量 | 待运行 |

## 3. 二十组结果

| ID | 模型 | W/A/K/V | LR / tuning KL | 正式 KL | PPL | QA Avg | GSM8K | MATH-500 | HumanEval+ base/plus | GPU·h | 状态 |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 01 | Qwen3-0.6B | 4/16/16/16 | — | — | — | — | — | — | — | — | 待调参 |
| 02 | Qwen3-0.6B | 4/4/4/4 | — | — | — | — | — | — | — | — | 待调参 |
| 03 | Qwen3-0.6B | 3/16/16/16 | — | — | — | — | — | — | — | — | 待调参 |
| 04 | Qwen3-0.6B | 2/16/16/16 | — | — | — | — | — | — | — | — | 待调参 |
| 05 | Llama-3.1-8B-Instruct | 4/16/16/16 | — | — | — | — | — | — | — | — | 待调参 |
| 06 | Llama-3.1-8B-Instruct | 4/4/4/4 | — | — | — | — | — | — | — | — | 待调参 |
| 07 | Llama-3.1-8B-Instruct | 3/16/16/16 | — | — | — | — | — | — | — | — | 待调参 |
| 08 | Llama-3.1-8B-Instruct | 2/16/16/16 | — | — | — | — | — | — | — | — | 待调参 |
| 09 | Qwen3-4B | 4/16/16/16 | — | — | — | — | — | — | — | — | 待调参 |
| 10 | Qwen3-4B | 4/4/4/4 | — | — | — | — | — | — | — | — | 待调参 |
| 11 | Qwen3-4B | 3/16/16/16 | — | — | — | — | — | — | — | — | 待调参 |
| 12 | Qwen3-4B | 2/16/16/16 | — | — | — | — | — | — | — | — | 待调参 |
| 13 | Qwen3-8B | 4/16/16/16 | — | — | — | — | — | — | — | — | 待调参 |
| 14 | Qwen3-8B | 4/4/4/4 | — | — | — | — | — | — | — | — | 待调参 |
| 15 | Qwen3-8B | 3/16/16/16 | — | — | — | — | — | — | — | — | 待调参 |
| 16 | Qwen3-8B | 2/16/16/16 | — | — | — | — | — | — | — | — | 待调参 |
| 17 | Qwen3-32B | 4/16/16/16 | — | — | — | — | — | — | — | — | 待调参 |
| 18 | Qwen3-32B | 4/4/4/4 | — | — | — | — | — | — | — | — | 待调参 |
| 19 | Qwen3-32B | 3/16/16/16 | — | — | — | — | — | — | — | — | 待调参 |
| 20 | Qwen3-32B | 2/16/16/16 | — | — | — | — | — | — | — | — | 待调参 |

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
