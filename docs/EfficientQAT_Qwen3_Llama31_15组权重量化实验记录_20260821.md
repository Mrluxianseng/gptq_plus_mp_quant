# EfficientQAT：Qwen3 / Llama-3.1 15 组权重量化实验记录

> 日期：2026-08-21。状态：15/15 量化、WikiText-2 KL/PPL、十项 QA、三项推理与
> 官方 HumanEval+ 评分全部完成。本文是本轮正式运行的唯一流水记录；数值只通过
> 55-suite fail-closed release 写入公平总表，不手工抄录。

## 实验范围与论文依据

本轮不运行 EfficientQAT 的激活量化，也不运行 QuaRot。检查本地论文 LaTeX、附录、
官方 README 和实现后，结论是 EfficientQAT 论文的主实验只有 weight-only：结果表只有
W2/W3/W4 与 G64/G128，附录也明确说明只量化 transformer block 内 Linear 的权重；
官方量化器仅挂在 `QuantLinear.weight`。正文关于支持 weight-activation 的一句话指向后续
独立方法 PrefixQuant，不能作为 EfficientQAT 已做激活量化实验的证据。

因此原 20 设定中只运行五模型 × `W4A16KV16/W3A16KV16/W2A16KV16`，共 15 个唯一
checkpoint。五个 `W4A4KV4` 格子最终统一写为
`N/A（EfficientQAT 论文未做 A/K/V 量化）`，不会复制 W4A16 数值冒充结果。

## 冻结配置

- 模型：Qwen3-0.6B、Qwen3-4B、Qwen3-8B、Qwen3-32B、
  Llama-3.1-8B-Instruct；
- 数据：直接复用公平总表五份物理 token 制品，WikiText-2 train、seed=1、
  `256 × 2048 = 524,288` token；同时锁 archive SHA、解包 tensor SHA、dtype、shape
  与顺序；Block-AP 和 E2E-QP 遍历同一制品；
- 权重量化：W2/W3/W4 signed-symmetric uniform scalar，natural input-column G128，
  每 output-row/group 一个正 scale，zero 固定为 0；不使用 QuaRot；
- A/K/V：BF16 恒等路径，不安装 activation/KV quantizer；“A 对称量化”在本方法中为
  N/A；
- Block-AP：batch size=2、epoch=2、quantizer LR=`1e-4`、weight LR 为 W2
  `2e-5`、W3/W4 `1e-5`、cosine 到初值 1/20、FP16 AMP；按论文训练 block 内
  Linear weight、scale 与 RMSNorm weight；
- E2E-QP：micro batch=4、gradient accumulation=8、effective batch=32、epoch=1、
  scale-only；W2 LR=`2e-5`、W3/W4 LR=`1e-5`、cosine、warmup=0.03、
  max grad norm=0.3、BF16；
- 兼容替代：使用 `torch.optim.AdamW` 代替当前 CUDA/Torch 环境不可用的论文旧版
  `bitsandbytes==0.41.0`；其余 LR、WD 与 scheduler 不变；
- 确定性：Python hash seed=1；PyTorch deterministic algorithms 开启；
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`；cuDNN benchmark 关闭、cuDNN deterministic
  开启、TF32 关闭；
- GPU-hour：只计 Block-AP + E2E-QP 的单卡墙钟时间，排除模型加载、checkpoint I/O、
  dense materialization 与后续评测。

以上属于 complete-method 对比：EfficientQAT 保留论文的两阶段 QAT 和 max-abs
scale 初始化；没有为了伪造 solver-only 对比而替换成 RealQ 的 w-clip 初始化。

## 运行环境与任务分配

两台实验机均通过 7 天 8-GPU sleep debug job 提交后 SSH 进入容器执行：

| job | pod | GPU | 分配 |
|---|---|---:|---|
| `j-4mj21jb084` | `j-4mj21jb084-master-0` | 8 × NVIDIA L20C | Qwen3-32B W4/W3；Qwen3-8B W4/W3；Qwen3-4B W4/W3；Llama W4；Qwen3-0.6B W4 |
| `j-zogxxxduju` | `j-zogxxxduju-master-0` | 8 × NVIDIA L20C | Qwen3-32B W2；Qwen3-8B W2；Qwen3-4B W2；Llama W3/W2；Qwen3-0.6B W3/W2；GPU7 作为补跑/评测预留 |

正式环境为仓库历史 Canoe 实验已验证的
`.venv.py312-broken-20260729/bin/python`：Python 3.12.3、Torch 2.9.1+cu128、
Transformers 4.56.2、Datasets 3.6.0、Accelerate 1.12.0、lm-eval 0.4.4。
目录名中的 `broken` 只表示它不能在宿主机直接运行；在 Canoe Python 3.12 镜像内是已验证
环境。容器内 `.venv/bin/python` 的 `/usr/local/bin/python3` 目标不存在，故未使用该失效
链接，也没有修改共享 `.venv`。

## 启动前门禁

- 15 个 run 必须严格等于五模型 × W2/W3/W4，全部 A16/KV16、rotation=false；
- 五份模型 config/tokenizer/checkpoint inventory 与五份校准 token 制品逐项校验通过；
- tokenizer 与模型词表分别校验：Llama 为 128256/128256；Qwen tokenizer 为
  151669、最大 token id=151668，模型 padding vocab 为 151936；
- 两机运行时版本与 CUDA L20C matmul 通过；启动前两机 GPU 均为空闲；
- 当前正式 v3 plan SHA-256：
  `ceffe64189f07a5753a4feffa21553d8c802fa4c798af645ca6199b0fa271032`；
- v3 preflight：`status=succeeded`，完成时间
  `2026-08-21T04:14:24.036757+00:00`；其中 Block-AP 的
  `minimum_lr_ratio` 被 fail-closed 固定为 0.05，对应论文 `min_lr_factor=20`。

## 首轮门禁失败与修复

12:01 左右的 v1 首轮在进入训练前失败；Qwen 的错误统一为
`Tokenizer vocabulary ... 151669 != 151936`。这不是 OOM，也没有生成量化 checkpoint。
原因是计划误把 Qwen 的真实 tokenizer 长度与为 tensor 对齐而 padding 的模型 embedding
行数写成必须相等。三个仍处于模型加载阶段的 Llama 进程随后被主动 TERM，避免形成混合
批次。

修复方式不是放宽为不校验，而是把两个合同拆开：tokenizer 必须精确等于 151669，模型
config/input embedding/lm_head 必须精确等于 151936，同时增加 `max_token_id <
model_vocab_size` 门禁。v1 的 failure manifest 与日志完整保留在
`experiment_data/efficientqat_weightonly_15group_20260821`，不删除、不覆盖；正式运行写入全新
目录。

12:09 左右启动的 v2 随后由主动日志监控发现第二个配置错误：计划把论文参数
`min_lr_factor=20` 错填入语义为 `minimum_lr_ratio` 的字段，runner 再取倒数后得到
`min_lr_factor=0.05`，使 quantizer LR 从 `1e-4` 上升至 `2e-3`，而不是下降到
`5e-6`。证据是首个 Qwen3-0.6B block 的日志直接记录了 `quant_lr:0.002`。发现后立即
TERM 全部 15 个 worker；v2 不生成有效结果，也不进入总表。代码新增
`0 < minimum_lr_ratio <= 1` 运行时门禁，preflight 进一步要求其精确等于 0.05。
v2 partial manifest/log 完整保留在
`experiment_data/efficientqat_weightonly_15group_20260821_v2`。

## 正式运行进度

- 2026-08-21 12:15（Asia/Shanghai）：v3 的 15 个 worker 全部启动；node0 占 8/8 卡，
  node1 占 7/8 卡；输出根目录为
  `experiment_data/efficientqat_weightonly_15group_20260821_v3`；
- 2026-08-21 12:18：动态 LR 门禁通过。多个 run 的 block 0 实测 quantizer LR
  依次为 `1e-4`（初值）、`5.192416827251822e-5`（epoch 0 末）和 `5e-6`
  （epoch 1 末），符合 cosine 降到初值 1/20；相应 reconstruction loss 有限且下降；
- 当前阶段：15 个 run 并行进行 Block-AP，未见 OOM、Traceback 或非有限 loss。
  结果、GPU-hour 与评测将在完成后继续补入本节。

## FP32 scale 下限误判与可审计恢复（2026-08-21 13:55—）

- `EQ15-Q06-W3` 在 Block-AP 成功、E2E-QP 完整训练结束后，于 13:55 CST
  被后置合同错误拒绝；`EQ15-Q06-W4` 于 14:18 CST 命中同一错误。错误均为
  `positive scale projection contract failed: 9.999999747378752e-05 < 0.0001`，
  不是 OOM、非有限值或训练失败；
- 根因是 `clamp_(min=1e-4)` 写入 FP32 后的精确可表示值为
  `9.999999747378752e-05`，旧门禁却把它转回 Python float 后与十进制
  `0.0001` 直接比较。修复只把门禁阈值转换为各参数 dtype 的精确可表示值；
  训练值、学习率、随机种子、优化器、数据和量化算法均未改变；
- 原失败目录和 failure manifest 完整保留。恢复运行写入独立的
  `experiment_data/efficientqat_weightonly_15group_20260821_v4_recovery`，严格复用
  已成功且 stat identity 前后不变的 Block-AP checkpoint，只重跑 E2E-QP 和 CPU
  materialize；算法 GPU-hour 计为原 Block-AP 加恢复 E2E-QP。被该错误门禁废弃的首轮
  E2E 开销不进入主表，但作为 superseded debugging cost 单列保留；
- 14:22 CST，`EQ15-Q06-W2` 已正常成功，算法耗时 `1.757293085 GPU·h`，证明该
  错误只在 learned scale 恰好被投影到下限的 run 上触发；
- 14:43 CST，W4 恢复全部成功：8 个 E2E optimizer steps 用时 58.4151 s，最终
  `2.037462067 GPU·h`；旧错误门禁废弃的 E2E wall time 为 208.405669 s，已单独记录
  且不计入主表。恢复 terminal、stage accounting、parent failure/block-stage SHA、
  materialization manifest 及 run/spec 绑定均通过统一 evaluator 的严格门禁；
- W3 恢复仍等待同卡前序 TurboBOA 释放物理锁。其余 12 组继续在原 v3 Block-AP 中
  推进，未见新错误。

15:12:38 CST，W3 恢复也已成功：原 Block-AP 计时 5776.056770 s，恢复 E2E-QP
39.139222 s，合计 `1.615332220 GPU·h`；被旧错误门禁废弃的 E2E wall time
181.597620 s 单独保留且不计入主表。统一 evaluator 随后对 Q06-W4/W3/W2 三个 terminal
逐一执行 recovery policy、stage/accounting、artifact SHA 与 materialization binding 门禁，
三组全部通过；对应正式 GPU-hour 分别为 2.037462067、1.615332220、1.757293085。

## 并发运行的阶段性低 GPU 利用率说明

14:33 CST 对两机各连续采样 5 秒：16 条物理 GPU lane 均有正式任务或严格串行的后继
队列占用，但 EfficientQAT 多数 lane 当时处于 CPU `pack quantized ...`，Q06-W4 恢复
处于 CPU packed-weight 解码/materialize，因而 SM 利用率可连续为 0%。对应 PID 的 CPU
占用约 5—13 个核，日志时间戳持续增长；例如 Qwen3-8B W4 在 14:33 又完成了当前 block
的 q/k/v projection pack。这是 EfficientQAT 当前实现的 CPU 阶段，不是进程退出、锁死
或 OOM。后继任务按 `EfficientQAT/恢复 -> TurboBOA -> YAQA` 严格串行接续，释放锁后
无需再次人工确认。

## CPU packing 线程超卖与等价加速重放（2026-08-21 14:49—15:12 CST）

主动检查 Qwen3-32B 日志后确认，前述低利用率不是正常且可接受的短暂阶段，而是会把
整轮实验拖到数天的实现级性能问题：旧 `QuantLinear.pack` 对每个 input feature 单独发起
一次 PyTorch CPU 算子；容器中每个进程又默认启用 144 个 intra-op/interop 线程。12 个
并行 Block-AP 进程因此在 288 核节点上形成严重线程超卖，node0 load average 一度约 486。
实测 Qwen3-32B 一个 block 的 `mlp.down_proj` packing 约 38.5 分钟，而两轮 GPU 训练只需
约 42 秒；按 64 层外推，继续等待并不合理。

修复仅改变 packed integer 的计算批次，不改变任何量化数学：把逐 input-column 的
elementwise `add/divide/round/to(int32)` 合并为连续 256 列一块，后续 W2/W3/W4 bit packing
循环保持原样；重放进程另将 `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS` 全部固定为 1，消除
12 个进程各自使用 144 线程的超卖。正式 plan、校准 token、seed、学习率、epoch、
optimizer、quantizer、G128、对称性和 GPU-hour 计时边界均未改变。

等价门禁分两层通过：

- 4 个 seed/3 个 bit/4 个含 G128 边界与 short-tail 的形状，共 48 组，优化前后的
  `intweight` 与最终 packed `qweight` 逐字节相同；代表性 4096×1024 矩阵在单线程下
  该阶段由 0.593849 s 降至 0.099995 s（5.94×）；
- 直接实例化正式 `QuantLinear.pack` 再覆盖 3 个 seed/3 个 bit/4 个形状，共 36 组，
  `qweight/qzeros/scales/bias` 与旧参考实现逐字节相同；优化源码 SHA-256 为
  `d259af6def9db43766ecd704a6fc03f04913c8156d81d530f62777fe5defc2dc`。

15:09 CST 对 12 个尚未完成的旧 worker 逐 PID/完整命令核对后发送 TERM，均在 30 秒内
正常退出；没有删除或覆盖数据。其 partial run 目录原子移动到
`_superseded_cpu_thread_oversubscription_20260821/`，不计入正式 GPU-hour。原有 recovery
watcher 继续持有每卡共享物理锁，因此 TurboBOA/YAQA/evaluator 没有机会在切换窗口抢占
这些卡。15:10:52 CST，两机分别重放 7+5 个 run；launcher receipt 位于
`_cpu_pack_replay/queue_logs/<pod>/launcher_receipt.json`，记录 plan/source SHA、线程合同、
归档路径与新 PID。

重放启动约 1 分钟后，Qwen3-4B W3/W4 已进入 block 1，Llama/Qwen3-8B 的 projection
packing 以秒级推进；此前旧进程运行约 2 小时 45 分钟仅到 block 2—7。12 个重放 worker
均存活，未见 OOM、Traceback 或 non-finite；Qwen3-32B 三组当时仍在加载 17 个 checkpoint
shard。后续正式结果只接受重放目录产生、且 manifest 记录上述优化源码 SHA 的 terminal。

## Qwen3-0.6B 污染计时的干净重放（2026-08-21 15:33—）

Qwen3-0.6B 三组虽然在 packing 优化前已经产生成功 checkpoint，但它们的 Block-AP 与
12 个各占 144 CPU 线程的旧 worker 重叠，算法墙钟同样不能作为公平 GPU-hour。为避免
重新定义 checkpoint，另在 node1/GPU5 串行重放 W4/W3/W2：使用原冻结 plan、原 token、
seed、学习率、两阶段配置和优化后的 byte-equivalent pack，并把四个 CPU 线程环境变量
全部固定为 1。只有重放 `model.safetensors` 与原成功 checkpoint 的 SHA-256 完全相同，
干净计时才允许进入总表；原 checkpoint 继续作为评测输入。

首次 v1 启动遗漏 `CUDA_VISIBLE_DEVICES=5`，在 runtime gate 看到 8 张卡后瞬时失败；未
加载模型、未执行 GPU 算法，failure 保留在
`experiment_data/efficientqat_q06_clean_retime_20260821_v1`。修正环境后使用全新 v2
immutable output；这次入口门禁通过。

15:40 CST，W4 重放成功：Block-AP 147.742908 s、E2E-QP 38.209360 s，合计
`0.051653408 GPU·h`；新旧 `model.safetensors` SHA-256 均为
`14fc26b44da22cd1bfdd322e5427472e07c0f012b0c8a166110f08854aee0357`，逐字节一致。
原受污染的 `2.037462067 GPU·h` 将从公平主表排除，但保留在审计记录中。

15:43 CST，W3 也通过相同门禁：Block-AP 150.145366 s、E2E-QP 32.487671 s，合计
`0.050731399 GPU·h`；新旧 checkpoint SHA-256 均为
`7985bf364ab6033d9964443cc7e3304757b9553e46752c97f575531655d16e1f`。原受污染的
`1.615332220 GPU·h` 同样只保留为审计数据。

15:47 CST，W2 完成，干净算法耗时 `0.051758279 GPU·h`，并通过新旧 checkpoint
byte identity。三组汇总如下；三份替代计时的离散仅约 2%，也与模型大小相符：

| run | 受污染原 GPU·h（排除） | 干净 GPU·h（主表） | checkpoint byte identity |
|---|---:|---:|---|
| EQ15-Q06-W4 | 2.037462067 | 0.051653408 | true |
| EQ15-Q06-W3 | 1.615332220 | 0.050731399 | true |
| EQ15-Q06-W2 | 1.757293085 | 0.051758279 | true |

v2 clean-retime summary 状态为 `succeeded`，三组均固定 pack 源码 SHA 与四项
单线程环境合同。worker 在释放物理锁后于 15:47:05 CST 自动恢复 node1/GPU5 的原
YAQA 队列，未修改 YAQA plan 或评测输入。

## 15/15 正式量化完成（2026-08-21 16:14 CST）

最后三组 Qwen3-32B 均完成 optimized replay、E2E-QP、CPU materialize 与 terminal
门禁：W2 于 16:06:46、W3 于 16:13:13、W4 于 16:13:48 CST 发布。三组算法成本分别为
`0.629703916/0.667888369/0.670817303 GPU·h`；materialize 的 CPU wall time 不计入
GPU-hour。15 组最终可用于公平主表的量化成本如下，其中 Qwen3-0.6B 三组明确使用上节
checkpoint byte-identical 的干净重计时：

| run | GPU·h |
|---|---:|
| EQ15-Q32-W4 | 0.670817303 |
| EQ15-Q32-W3 | 0.667888369 |
| EQ15-Q32-W2 | 0.629703916 |
| EQ15-Q8-W4 | 0.224545391 |
| EQ15-Q8-W3 | 0.212619553 |
| EQ15-Q8-W2 | 0.218339577 |
| EQ15-L8-W4 | 0.191680347 |
| EQ15-L8-W3 | 0.197384824 |
| EQ15-L8-W2 | 0.200823429 |
| EQ15-Q4-W4 | 0.138702688 |
| EQ15-Q4-W3 | 0.135226681 |
| EQ15-Q4-W2 | 0.138809548 |
| EQ15-Q06-W4 | 0.051653408 |
| EQ15-Q06-W3 | 0.050731399 |
| EQ15-Q06-W2 | 0.051758279 |

正式量化新增 failure 为 0；Q06 W4/W3 的旧 FP32 下限门禁 failure 仍只作为已解释且
已恢复的审计制品保留。统一 evaluator 已开始消费 terminal checkpoint，量化目录不再修改。

## 统一质量与推理评测进度（2026-08-21 19:33 CST）

15 个权重量化 checkpoint 均由与其他方法共用的冻结 evaluator 消费；当前已原子发布
5 份 WikiText2 exact KL/PPL 与十项 QA 结果：

| run | Exact-KL | PPL | 十项 QA Avg |
|---|---:|---:|---:|
| EQ15-Q32-W4 | 0.1534895301 | 6.5034666 | 72.87 |
| EQ15-Q32-W3 | 0.2585729957 | 7.6540432 | 68.73 |
| EQ15-Q32-W2 | 2.7352650166 | 82.7267151 | 31.94 |
| EQ15-Q8-W4 | 0.1559179723 | 7.9720798 | 67.54 |
| EQ15-Q8-W3 | 0.2544064820 | 10.1282463 | 63.12 |

推理 generation 与量化并行推进：Q32-W4、Q8-W4 的 GSM8K 已完成；Q8-W4 的
MATH-500 也已完成并通过 scorer-only 缺包恢复，当前 Q32-W4/Q32-W2 在 MATH-500、
Q32-W3 在 GSM8K 继续生成。尚无完整三任务 suite，是因为每个 checkpoint 还需完整
HumanEval+；不是 quality failure。其余 GPU 同时推进 YAQA 量化与 TurboBOA 评测，
REAL-Q 路径排查固定只占一条 lane。

## 首份完整三项推理结果（2026-08-21 20:33 CST）

`EQ15-Q32-W2` 已完成全部 generation 与官方 scorer：GSM8K=`6/1319=0.4549%`、
MATH-500=`2/500=0.40%`；HumanEval+ 的 164 条候选经受限、断网的官方 EvalPlus
逐题执行后，base/plus 均为 `0/164=0%`。HumanEval+ generation manifest SHA-256 为
`ae742133df2153bebec2947751a463d5fe4981dd9b490091ed25ddc4fdccba85`，正式 samples
SHA-256 为 `71e4f972324fdffd69aa9e2d58deee0220eadaae2d47f0d5f36267d1331fd6fd`；
scorer attempt001 一次成功，官方执行耗时 38.4953 s，绑定统一 reference fingerprint
`3ecffa5d...be05`。该结果与其 quality 的 KL=`2.7352650`、PPL=`82.7267`、QA Avg=
`31.94` 同向退化，不是单项 scorer 异常。

评测 worker 在发布 generation terminal 后立即把原 GPU lane 交给
`TB20-Q32-W4`；官方 scorer 独立使用 CPU，不占 GPU。`suite_success.json` 按冻结协议
保留 `generation_succeeded_official_humaneval_pending` 状态，最终完整性由并列的
`official_success.json` 收口，release builder 会同时强制验证两份 terminal。

## 第二份完整三项推理结果（2026-08-21 21:08 CST）

`EQ15-Q32-W4` 已完成全部 generation 与官方 scorer：GSM8K=
`1239/1319=93.9348%`，MATH-500=`306/500=61.20%`；HumanEval+ 官方 EvalPlus
attempt001 一次成功，base=`86/164=52.4390%`、plus=`82/164=50.00%`，官方执行
耗时 `37.8062 s`。对应 quality 为 Exact-KL=`0.1534895301`、PPL=`6.5034666`、
十项 QA Avg=`72.87`，量化成本=`0.670817303 GPU·h`。三项 generation 覆盖分别为
1319/500/164，reference fingerprint、checkpoint terminal/validation SHA 与统一 evaluator
门禁均通过，正式评测 failure 仍为 0。

## 第三份完整三项推理结果（2026-08-21 21:17 CST）

`EQ15-Q8-W4` 已完成统一质量、三项 generation 和官方 scorer。质量结果为
Exact-KL=`0.1559179723`、WikiText2 PPL=`7.9720798`、十项 QA Avg=`67.54`；
三项推理为 GSM8K=`1098/1319=83.2449%`、MATH-500=`284/500=56.80%`、
HumanEval+ base=`75/164=45.7317%`、plus=`71/164=43.2927%`。量化成本为
`0.224545391 GPU·h`。

该组 MATH-500 使用前述 scorer-only 缺包恢复：500 条冻结 generation 的文件内容、
顺序和 SHA 均未改变，恢复过程没有加载模型或使用 CUDA；GSM8K 和 HumanEval+ 则由
同一 suite attempt 正常完成。官方 EvalPlus attempt001 一次成功，耗时
`37.0879 s`。checkpoint terminal/validation SHA、共享 Qwen3-8B teacher SHA
`f68f0312...e5decb`、reference fingerprint 和三项样本覆盖均通过 release builder
的只读验证；至此完整可发布套件为 `3/55`。

## 第四份完整三项推理结果（2026-08-21 22:58 CST）

`EQ15-Q8-W3` 已完成统一质量、三项 generation 和官方 scorer。质量结果为
Exact-KL=`0.2544064820`、WikiText2 PPL=`10.1282463`、十项 QA Avg=`63.12`；
三项推理为 GSM8K=`74.9810%`、MATH-500=`48.80%`、HumanEval+ base/plus=
`15.2439%/14.0244%`。量化成本为 `0.212619553 GPU·h`。

suite 与 official receipt 分别于 22:57:11/22:57:57 CST 原子发布，EvalPlus
attempt001 完成 164 条正式样本；checkpoint terminal、共享 Qwen3-8B teacher、reference
fingerprint 和完整样本数均由同一 release gate 约束，未发生 fallback 或正式 failure。

## 第五份完整三项推理结果（2026-08-21 23:04 CST）

`EQ15-Q32-W3` 的完整结果为 Exact-KL=`0.2585729957`、WikiText2
PPL=`7.6540432`、十项 QA Avg=`68.73`；GSM8K=`84.9128%`、MATH-500=`53.60%`、
HumanEval+ base/plus=`35.3659%/33.5366%`。量化成本为 `0.667888369 GPU·h`。
suite/official receipt 分别于 23:02:57/23:03:54 CST 发布，官方 EvalPlus attempt001
覆盖 164 条样本、耗时 `37.1742 s`，所有冻结身份和 terminal SHA 门禁通过。

同轮 `EQ15-Q8-W2` quality 于 23:04:01 CST 发布：Exact-KL=`4.0143013000`、
WikiText2 PPL=`353.3515930`、十项 QA Avg=`29.77`，量化成本=
`0.218339577 GPU·h`。KL/PPL/QA 同向显示 2-bit checkpoint 严重退化，但 terminal、
checkpoint reconstruction 和 evaluator 均成功；正式记录保留原始算法结果并继续三项
推理，不采用结果驱动的 checkpoint fallback。

23:20:44 CST，`EQ15-L8-W4` 也完成统一 quality：Exact-KL=`0.1389802992`、
WikiText2 PPL=`6.8016219`、十项 QA Avg=`66.03`，正式量化成本仍为
`0.191680347 GPU·h`；随后继续冻结三项推理协议。至 23:25 CST，整个 55-checkpoint
统一评测共有 `19` 份 quality、`8` 份 generation suite 和 `8` 份官方 HumanEval+
terminal；EfficientQAT 量化仍为 15/15 terminal、无新增算法 failure。

23:37:02 CST，`EQ15-L8-W3` 发布统一 quality：Exact-KL=`0.2165286541`、
WikiText2 PPL=`7.9232922`、十项 QA Avg=`62.19`，正式量化成本=
`0.197384824 GPU·h`。checkpoint terminal、Llama-3.1-8B 共享 teacher 与冻结
reference fingerprint 均通过门禁，随后继续三项推理；整个统一评测更新为
**21/55 quality、10/55 generation suite、10/55 official terminal**。

2026-08-22 00:28:31 CST，`EQ15-L8-W4` 完成三项 generation；00:29:24 CST，
官方 EvalPlus attempt001 发布 terminal。完整结果为 Exact-KL=`0.1389802992`、
WikiText2 PPL=`6.8016219`、十项 QA Avg=`66.03`、GSM8K=
`1094/1319=82.9416%`、MATH-500=`173/500=34.60%`、HumanEval+ base/plus=
`91/164=55.4878%` / `80/164=48.7805%`，正式量化成本=`0.191680 GPU·h`。
三项覆盖、checkpoint SHA 与冻结 reference fingerprint 均通过门禁；原 GPU3 worker
随即领取 `EQ15-L8-W2`，没有空置。

00:34:40 CST，`EQ15-L8-W2` 发布统一 quality：Exact-KL=`3.6673908234`、
WikiText2 PPL=`257.1999512`、十项 QA Avg=`30.26`，正式量化成本=
`0.200823429 GPU·h`。KL/PPL/QA 与其他 2-bit EfficientQAT checkpoint 同向严重退化；
checkpoint、共享 teacher、reference fingerprint 和 10 项 QA 覆盖均通过门禁，因此
原值继续三项推理，不做结果驱动 fallback。统一评测达到 **23/55 quality、14/55
suite、14/55 official terminal**。

2026-08-22 00:44:05/00:44:57 CST，`EQ15-L8-W3` 与 `EQ15-Q8-W2` 先后发布
generation suite；官方 EvalPlus terminal 分别于 00:45:06/00:45:51 CST 发布。
`EQ15-L8-W3` 的三项推理为 GSM8K=`874/1319=66.2623%`、MATH-500=
`106/500=21.20%`、HumanEval+ base/plus=`57/164=34.7561%` /
`54/164=32.9268%`，对应 Exact-KL=`0.2165286541`、PPL=`7.9232922`、QA Avg=
`62.19`、量化成本=`0.197385 GPU·h`。

`EQ15-Q8-W2` 则为 GSM8K=`15/1319=1.1372%`、MATH-500=`20/500=4.00%`、
HumanEval+ base/plus 均为 `0/164=0%`，与 Exact-KL=`4.0143013000`、PPL=
`353.3515930`、QA Avg=`29.77` 的严重退化一致；量化成本=`0.218340 GPU·h`。
两组完整覆盖与冻结身份门禁均通过，统一评测更新为 **25/55 quality、16/55 suite、
16/55 official terminal**。

00:50:00 CST，`EQ15-Q4-W4` 发布统一 quality：Exact-KL=`0.2202488333`、
WikiText2 PPL=`9.6839247`、十项 QA Avg=`64.10`，正式量化成本=
`0.138702688 GPU·h`。checkpoint、共享 Qwen3-4B teacher 和冻结十项 QA 覆盖均通过，
随后继续三项推理；统一评测更新为 **27/55 quality、16/55 suite、16/55 official
terminal**。

01:09:43 CST，`EQ15-Q4-W3` 发布统一 quality：Exact-KL=`0.3335486650`、
WikiText2 PPL=`13.1818762`、十项 QA Avg=`57.39`，正式量化成本=
`0.135226681 GPU·h`。冻结 checkpoint/teacher/QA 覆盖门禁通过并继续三项推理；统一
评测更新为 **30/55 quality、18/55 suite、18/55 official terminal**。

01:59:26 CST，`EQ15-L8-W2` 发布三项 generation；02:00:28 CST，官方 EvalPlus
attempt001 发布 terminal。完整结果为 Exact-KL=`3.6673908234`、PPL=
`257.1999512`、QA Avg=`30.26`，GSM8K=`0/1319=0%`、MATH-500=`0/500=0%`、
HumanEval+ base/plus 均为 `0/164=0%`，正式量化成本=`0.200823 GPU·h`。全部样本
覆盖、冻结 checkpoint/teacher SHA 和官方 scorer 门禁均通过；推理完全失效与该 W2
checkpoint 的质量异常同向，原值保留。统一评测达到 **31/55 quality、21/55 suite、
21/55 official terminal**。该 suite 及 CUDA 退出经核验后，node0/GPU3 的物理锁交接
给四组 TurboBOA Llama manifest-alias recovery。

02:27:32 CST，`EQ15-Q4-W4` 发布三项 generation；02:28:32 CST，官方
EvalPlus attempt001 发布 terminal。完整结果为 Exact-KL=`0.2202488333`、PPL=
`9.6839247`、QA Avg=`64.10`，GSM8K=`1087/1319=82.4109%`、MATH-500=
`279/500=55.80%`、HumanEval+ base=`58/164=35.3659%`、plus=
`54/164=32.9268%`，正式量化成本=`0.138703 GPU·h`。完整样本覆盖、checkpoint/
teacher SHA 与官方 scorer 门禁均通过。

02:30:32 CST，释放的 lane 已发布 `EQ15-Q4-W2` quality：Exact-KL=
`4.5155420303`、WikiText2 PPL=`666.1116333`、十项 QA Avg=`29.98`，正式量化成本=
`0.138810 GPU·h`。三个质量指标再次同向显示 2-bit 严重退化，身份和覆盖门禁通过后
继续三项推理，不采用 fallback。整个统一评测更新为 **35/55 quality、24/55 suite、
24/55 official terminal**。

02:44:18 CST，`EQ15-Q4-W3` 发布三项 generation；02:45:15 CST，官方
EvalPlus attempt001 发布 terminal。完整结果为 Exact-KL=`0.3335486650`、PPL=
`13.1818762`、QA Avg=`57.39`，GSM8K=`743/1319=56.3306%`、MATH-500=
`139/500=27.80%`、HumanEval+ base/plus 均为 `0/164=0%`，正式量化成本=
`0.135227 GPU·h`。完整样本覆盖和冻结身份门禁通过。

02:47:23 CST，`EQ15-Q06-W2` 发布统一 quality：Exact-KL=`7.6466550827`、
WikiText2 PPL=`28953.5742188`、十项 QA Avg=`30.44`。quality receipt 中绑定的原始
contended 计时为 `1.757293 GPU·h`；公平 release 按前述 checkpoint byte-identical v2
clean-retime 使用 **`0.051758 GPU·h`**，原计时显式列入 excluded accounting，不会进入
主表。质量指标同向严重退化，原值继续三项推理。连同同期 YAQA 的新结果，统一评测
达到 **38/55 quality、27/55 suite、27/55 official terminal**。

03:23:54 CST，`EQ15-Q06-W4` 发布统一 quality：Exact-KL=`0.2421669662`、
WikiText2 PPL=`17.4534340`、十项 QA Avg=`44.78`。checkpoint、Qwen3-0.6B teacher、
固定 256×2048 calibration token 与 W4/G128 对称量化身份均通过冻结门禁；公平 release
使用 checkpoint byte-identical v2 clean-retime 的 **`0.051653 GPU·h`**，原始受 CPU
争用污染的 `2.037462 GPU·h` 显式排除。该组已继续三项推理，统一评测更新为
**41/55 quality、28/55 suite、28/55 official terminal**。

03:47:06 CST，`EQ15-Q06-W2` 完成三项推理 generation；03:47:59 CST，官方
EvalPlus attempt001 发布 terminal。完整结果为前述 Exact-KL=`7.6466550827`、PPL=
`28953.5742188`、十项 QA Avg=`30.44`，GSM8K=`12/1319=0.9098%`、MATH-500=
`20/500=4.00%`、HumanEval+ base/plus 均为 `0/164=0%`。公平量化成本使用 checkpoint
byte-identical v2 clean-retime 的 **`0.051758 GPU·h`**；完整覆盖、冻结 checkpoint/
teacher SHA 与官方 scorer 门禁均通过。统一评测更新为 **43/55 quality、31/55
suite、31/55 official terminal**；释放的 node1/GPU2 随即领取 `EQ15-Q06-W3`，
约两分钟后已进入 CUDA 计算。

03:51:34 CST，`EQ15-Q06-W3` 发布统一 quality：Exact-KL=`0.5043333173`、
WikiText2 PPL=`26.7690525`、十项 QA Avg=`39.52`。checkpoint、Qwen3-0.6B teacher、
固定 token 与 W3/G128 对称权重量化身份均通过冻结门禁；公平 release 使用 checkpoint
byte-identical v2 clean-retime 的 **`0.050731 GPU·h`**，原始受 CPU 争用计时排除。
该组已进入 GSM8K，统一评测更新为 **44/55 quality、31/55 suite、31/55 official
terminal**。

04:08:13 CST，`EQ15-Q4-W2` 完成三项推理 generation；04:09:24 CST，官方
EvalPlus attempt001 发布 terminal。完整结果为前述 Exact-KL=`4.5155420303`、PPL=
`666.1116333`、十项 QA Avg=`29.98`，GSM8K=`1/1319=0.0758%`、MATH-500=
`1/500=0.20%`、HumanEval+ base/plus 均为 `0/164=0%`，正式量化成本=
`0.138810 GPU·h`。完整覆盖、冻结 checkpoint/teacher SHA 与官方 scorer 门禁均
通过；推理严重退化与质量指标一致。统一评测更新为 **45/55 quality、33/55 suite、
33/55 official terminal**。

04:25:57 CST，`EQ15-Q06-W4` 完成三项推理 generation；04:27:07 CST，官方
EvalPlus attempt001 发布 terminal。完整结果为前述 Exact-KL=`0.2421669662`、PPL=
`17.4534340`、十项 QA Avg=`44.78`，GSM8K=`834/1319=63.2297%`、MATH-500=
`182/500=36.40%`、HumanEval+ base=`38/164=23.1707%`、plus=
`33/164=20.1220%`。公平量化成本使用 checkpoint byte-identical v2 clean-retime 的
**`0.051653 GPU·h`**，原始受 CPU 争用污染的 `2.037462 GPU·h` 显式排除。完整
样本覆盖、冻结 checkpoint/teacher SHA 与官方 scorer 门禁均通过；统一评测更新为
**48/55 quality、36/55 suite、36/55 official terminal**。

04:52:37 CST，最后一组 `EQ15-Q06-W3` 完成三项 generation；04:53:50 CST，
官方 EvalPlus terminal 发布。完整结果为前述 Exact-KL=`0.5043333173`、PPL=
`26.7690525`、十项 QA Avg=`39.52`，GSM8K=`19/1319=1.4405%`、MATH-500=
`23/500=4.60%`、HumanEval+ base/plus 均为 `0/164=0%`。公平量化成本使用
checkpoint byte-identical v2 clean-retime 的 **`0.050731 GPU·h`**，原始受争用
计时 `1.615332 GPU·h` 排除；完整覆盖、冻结 checkpoint/teacher SHA 与官方 scorer
门禁均通过。至此本实验要求的 EfficientQAT **15/15 组质量与三项推理均闭合**。

## 15 组最终公平计费汇总（2026-08-22）

55-suite release builder 已逐行重新验证 EfficientQAT 的量化 terminal、dense checkpoint
validation、WikiText-2 full-vocabulary KL/PPL、十项 QA 固定顺序、三项推理完整覆盖和
官方 HumanEval+ 164/164 receipt。EfficientQAT 子矩阵为 **15/15**，公平量化成本合计
`3.780684712 GPU·h`，平均 `0.252045647 GPU·h/设定`；五个 W4A4KV4 格子保持
`N/A（论文未做 A/K/V 量化）`。所有行绑定统一 reference fingerprint
`3ecffa5d4cb97efb24aac4a668b6570d8e8193f4f4fa21f98825c2d9d238be05`。

## 第三方源码可重建补丁（2026-08-23）

EfficientQAT 是主仓库内的嵌套 Git checkout，父仓库无法把其 dirty 文件作为普通
源码提交。为保证远程 `zq` 分支可重建本实验实现，已从 upstream base commit
`39f37f3b6053681c9b1cd4c9dcaf692d8999459e` 冻结完整 tracked source diff 到
`third_party_patches/fair20_20260821/efficientqat.patch`；补丁 SHA-256=
`a403cd7a7a3fdcbba669e9dfecb2ddb31eef2c720a5dcec5d2e9dc3f8e9336e7`。
补丁排除的只有 Python cache；对当前实际量化源码执行
`git apply --check --reverse` 已通过，证明该补丁从上述 base commit 应用后可还原
生成本轮 checkpoint 的四个修改文件。
