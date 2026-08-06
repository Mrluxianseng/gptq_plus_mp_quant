# REAL-Q 主线功能整合与 8 卡验证记录

> 日期：2026-08-06
> 工作分支：`zq`
> 目标远端：`origin/zq`
> 工作目录：`/minimax-avatar-new/zhangqian/realq/gptq_plus`

## 1. 目标

按以下顺序完成主线整合：

1. 将当前 `realq/` 目录的全部工作区改动单独提交并推送；
2. 从 `realq-plus/` 整合 full-block Block-GD：更新当前 Transformer block 和滑窗下一 block 中所有尚未量化的权重，但明确排除 Lanczos/层级 Block-Hessian；
3. 整合 `realq_moe/` 的 MoE 支持；
4. 整合 `realq_benchmark/` 的推理集评测支持；
5. 保持 P01–P06、P10 和 Triton 等性能路径默认开启；
6. 完成本地检查后，按 `AGENT.md` 申请或复用 8 卡实验机，执行运行和数值检查。

## 2. 必须保持的算法与实验边界

- Stage 0 使用端到端 backward 预计算 grouped saliency 和每 Transformer block 的完整、非对角 aggregated Fisher；
- Stage 1 保留 saliency-weighted GPTQ、block 外二阶补偿、逐 column-block Adam refresh、最后一层真实 KL；
- full-block Block-GD 只扩大一阶 refresh 的可更新权重集合，不允许重新更新已量化权重；
- slide-window 下一 block 只有在下一 loss arm 实际参与时才加入可更新集合；
- 不引入 `realq-plus/second_order/`、`realq_block_hessian` 或任何 Lanczos 配置、trace、补偿调用；
- 评测与量化结构扩展不得把性能优化默认值改回 legacy/default-off；
- 正式数值检查优先看相对 BF16 的 Exact KL，并记录 PPL、运行时间和异常。

## 3. 初始状态

### 3.1 Git 状态

- 本地分支：`zq`；
- 初始 HEAD：`b2da01754c95ad06e61159603711dbeb24d31b17`；
- 初始远端：`origin/zq = 847465c2cbabd83782d318fc7ce8c2c8fbd7beda`；
- 本地相对远端：ahead 55、behind 0；
- 工作树包含大量任务外的用户改动和未跟踪实验资产，所有提交必须使用显式路径，禁止 `git add -A`。

### 3.2 首个提交限定范围

首个提交只包含用户明确指定的当前 `realq/` 改动：

```text
realq/config.py
realq/pipeline.py
realq/precompute/static_e2e.py
realq/quant/realq_layer.py
realq/quant/triton_column_block.py
realq/runner/layer_loop.py
```

任务外的 `utils/`、`gptq_utils/`、`tests/`、`.gitignore` 和既有 `docs/` 改动不进入该提交。

### 3.3 发布环境检查

- `origin`：`https://github.com/QianerPKU/gptq_plus.git`；
- 当前环境未安装 `gh`，因此不能使用 GitHub CLI 做认证状态/PR 检查；
- 用户当前明确要求的是提交并推送 `zq`，不要求创建 PR。推送前通过只读远端查询和 Git HTTPS 实际 push 结果验证权限。

## 4. 操作日志

### 2026-08-06：准备阶段

- 阅读 `AGENT.md`、`README.md`、`main.tex`、`对比实验方法.md`、`调参方法.md`；
- 刷新并核对 `origin/zq`；
- 确认当前性能改动把 P01–P06 默认打开，并新增默认开启的 P10/Triton；
- 确认 `realq/quant/realq_layer.py` 直接依赖新文件 `realq/quant/triton_column_block.py`，两者必须一起提交；
- 创建本文档，后续提交、测试、实验机和异常结果继续追加在这里。

## 5. 待填验证矩阵

| 阶段 | 检查 | 状态 | 证据/结果 |
|---|---|---|---|
| 当前 `realq/` 发布前 | diff 范围、whitespace、相关单测 | 通过 | `214 passed, 4 skipped`；`git diff --check -- realq` 无输出 |
| Plus 整合 | full-block active leaves、锁定已量化权重、排除 Lanczos | 通过 | `12 passed`；代码扫描无层级/Lanczos 依赖 |
| MoE 整合 | 路由统计、zero-route RTN、expert 锁步/joint update | 通过 | 本地 CPU 门禁通过；L20C 上包含 CUDA grouped-mm/joint runner 的整合套件通过 |
| Benchmark 整合 | 数据 manifest、resume、数学/代码任务 | 通过 | 单测通过；Qwen3-0.6B 四任务各 1 条真实 GPU 生成、判分/导出通过 |
| 本地总回归 | import、配置默认值、单元/数值测试 | 通过 | `331 passed, 45 skipped` |
| 8 卡 | 环境、分布式运行、数值对齐、真实量化 | 通过 | job `j-24iddqgmk3`；104 项 GPU 回归、1-rank/8-rank 数值对齐、Qwen3 layer-0 full-block+slide smoke 均通过 |

### 2026-08-06：当前 `realq/` 发布前测试

使用仓库 `.venv` 执行：

```bash
python -m pytest -q \
  tests/test_triton_column_block.py \
  tests/test_paper_numerics.py \
  tests/test_fisher_fp32_cache.py \
  tests/test_p04_compact_qparams.py \
  tests/test_p06_act_order_stitch.py \
  tests/test_realq_weight_group_quant.py
```

结果：`214 passed, 4 skipped, 1 warning in 443.28s`。4 个 skip 为当前宿主缺少相应 GPU 执行条件；唯一 warning 是既有 `.pytest_cache` 目录无写权限，不影响测试结果。

### 2026-08-06：当前 `realq/` 提交与推送

- 使用显式路径暂存第 3.2 节列出的 6 个文件；
- 暂存区统计：`6 files changed, 631 insertions(+), 142 deletions(-)`；
- `git diff --cached --check` 通过；
- 提交：`e9f80ad enable optimized REAL-Q defaults and Triton path`；
- 推送：`847465c..e9f80ad  zq -> zq`；
- 本地 `zq` 已重新设置为跟踪 `origin/zq`。

发布时 GitHub HTTPS 首次认证阶段出现约 30 秒延迟；使用无数据的 curl trace 确认代理、TLS 和 credential helper 均正常后，push 成功。trace 中认证头被 Git 自动隐藏，未记录凭据。

### 2026-08-06：REALQ-Plus full-block Block-GD 定向移植

- `BlockRefreshState` 为一个 Transformer block 内所有可量化线性层保存 FP32 master、Adam 一阶/二阶矩和持久 step；
- 当前线性层仅开放尚未量化的列，后续线性层开放全部列，已经完成量化的线性层永久锁定；
- 滑窗损失实际启用下一层分量（`alpha < 1`）时，下一 Transformer block 的全部未量化线性层也进入同一次 `functional_call + autograd.grad`；
- 当前线性层开始 GPTQ 前接管此前累积更新过的 FP32 master；act-order 路径在自然列坐标维护优化器状态，返回后仅排列并应用工作后缀；
- trace schema 升级到 4，同时兼容读取 schema 3，并记录每次真实 backward、chunk、全局样本数和 active weight 审计；
- 有意排除：`realq-plus/second_order`、`realq_block_hessian`、hierarchical compensator、Lanczos/Power 及其配置项；
- 新增主线测试：`tests/test_realq_full_block_refresh.py`、`tests/test_realq_full_block_layer_contract.py`；
- `.venv/bin/python -m pytest -q tests/test_realq_full_block_refresh.py tests/test_realq_full_block_layer_contract.py`：`12 passed, 1 warning in 163.33s`；warning 仍仅为 `.pytest_cache` 无写权限。

### 2026-08-06：MoE 与推理集评测整合

MoE 采用“主线入口自动分流、稀疏实现保持专用调度”的组合方式：

- `realq.pipeline.run` 在加载 dense 主线前识别 `Qwen3MoeForCausalLM`，转入 `realq_moe.pipeline.run`；不可解析的占位/远端配置不会在预探测阶段改变原有加载错误或测试 mock 点；
- dense 模型继续使用主线 full-block Block-GD、P01–P06/P10 与 Triton；MoE 保留自然 ragged route、expert-major CSR、zero-route RTN、全 expert 锁步 GPTQ、Jacobi joint Block-GD、fixed-route grouped-mm 和 fully GPU-resident 生命周期；
- 主 `Config` 增加 11 个 MoE policy/lifecycle 字段，正式稀疏运行强制 full-slide、joint、GPU resident，并拒绝 CPU master/FSDP offload；
- 主线和 `realq_moe.Config` 的八项性能默认值均核对为开启/优化实现：P01、P02、P03、P04、P05、P06、P10、Triton。

Benchmark 不复制旧量化内核，而是把 `realq_benchmark/benchmarks` 前移到 `realq/benchmarks`：

- 支持 GSM8K、MATH-500、HumanEval+、LiveCodeBench-lite；
- 支持离线 materialization、SHA256 manifest、generation fingerprint、JSONL flush/fsync 和断点恢复；
- HumanEval+/LCB 默认只导出候选，官方执行入口仍要求显式危险确认，LCB 还要求 `REALQ_ALLOW_UNTRUSTED_CODE=1`；
- `skip_kl_ppl_eval` 只跳过 WikiText KL/PPL，不影响 lm-eval/reasoning；`skip_eval` 仍关闭全部评测；
- 推理评测直接消费当前主 `Config` 和量化模型，因此不会回退 `realq_benchmark` 旧快照里的性能默认值。

本地验证：

- 使用独立 `/tmp` pycache 做全源文件静态编译，绕开派生目录中既有 root-owned `__pycache__`；
- `.venv/bin/python -m pytest -q realq_moe/tests tests/test_realq_reasoning_eval.py`：`46 passed, 41 skipped, 1 warning in 268.58s`；skip 均由 `requires CUDA`/CUDA grouped-mm 条件标记；
- 新增 `tests/test_realq_moe_dispatch.py`，验证主配置优化/MoE 默认值以及入口在 dense load 前正确分流；
- 交叉回归首次发现 `tokens_cache_file` 对旧式轻量 config 的兼容问题以及 MoE 预探测对测试占位模型名的提前查询问题；分别用 `getattr(..., None)` 和预探测 `OSError` 回退修复；
- 修复后失败用例定向重跑：`7 passed, 1 warning in 136.42s`。

### 2026-08-06：本地发布候选总门禁

首次组合主线旧测试与 full-block 后发现两类公开调用契约需要同时保留：

- 旧调用允许不传 `module_name/BlockRefreshState`，并期望 `(rows, trailing_cols)` update；
- full-block 工厂 closure 返回 `(rows, all_columns)` 的自然列 update，只允许应用 active suffix。

最终实现显式双协议：

- `make_grad_refresh_fn` / `make_kl_refresh_fn` 在未传 block state 时进入保留的 single-linear legacy path；
- full-block 工厂 closure 标记 `_realq_update_layout="full_natural"`；legacy closure 标记 `"trailing_quant_order"`；未标记的外部自定义 callback 保持 legacy 语义；
- `RealQLayer` 按显式协议决定调用坐标与 update 切片，不通过 tensor shape 猜测；
- legacy `RefreshContext` moment 和 schema-3/旧 trace 行为保留，主 runner 仍只走 full-block。

定向验证：`tests/test_fisher_fp32_cache.py` 相关用例、完整 P06、`tests/test_refresh_trace_runtime.py` 合计 `26 passed`。

最终发布候选命令覆盖：P01/P02/P04/P05/P06/P10/Triton、paper numerics、full-block、MoE、reasoning、distributed cache、checkpoint、refresh trace。结果：

```text
331 passed, 45 skipped, 1 warning in 270.54s
```

45 个 skip 均来自本机缺少 CUDA/Triton/grouped-mm 执行条件；唯一 warning 仍为既有 `.pytest_cache` 无写权限。CUDA skip 将在 `AGENT.md` 指定的 8 卡实验机阶段补测。

### 2026-08-06：集成提交与远端发布

- 使用显式路径暂存主线 full-block、MoE、benchmark、对应测试和本文档；
- 提交：`d5a0b93 integrate full-block MoE and reasoning evals`；
- 提交规模：`78 files changed, 24741 insertions(+), 67 deletions(-)`；
- 推送：`e9f80ad..d5a0b93  zq -> zq`；
- `realq_moe/precompute/cache.py` 命中用户全局 `cache*` ignore 规则，因此仅对这个明确源码文件使用 `git add -f`；未暂存任何运行缓存或 `__pycache__`；
- 提交前 `git diff --cached --check` 通过，主 `realq/` 未出现 `Lanczos`、`hierarchical` 或 `realq_block_hessian` 依赖。

## 6. 8 卡实验机申请与环境核验

### 6.1 按 `AGENT.md` 检查既有任务

- 优先任务 `j-8j1en3m0aq`：`Aborted`，唯一历史 pod 已失败，不可复用；
- 参考任务 `j-7x9o0je4pk`：`Aborted`，但其配置正是仓库要求的 8 卡 sleep 调试任务；
- 目标队列：`pa-cne02-prod-01 / minimax-avatar-h800new`；申请前查询到 `allocatable=656`、`allocated=615` GPU，具备至少 8 卡余量；
- 按 `canoe-ababvideo-submit` 的安全重提脚本原样复制参考任务，未直接调用写接口：

```text
python3 .../safe_resubmit.py j-7x9o0je4pk
job j-24iddqgmk3 recreated.
```

### 6.2 新任务身份

| 字段 | 值 |
|---|---|
| job | `j-24iddqgmk3` |
| pod | `j-24iddqgmk3-master-0` |
| task | `t-4k95m7p4k0` |
| node | `e01-cn-2ml4muulc3s` |
| cluster / queue | `cne02_PA / minimax-avatar-h800new` |
| image | `harbor.xaminim.com/minimax/avatar_gen3:v3.0_ngc2501_main` |
| source commit | `d5a0b932dfd8` |
| actual GPU | `8 × NVIDIA L20C, 183359 MiB` |
| driver / CUDA | `580.95.05 / 12.8` |
| state | `Running`, retry 0, Ready/ContainersReady=True |

队列名称包含 H800，但 Canoe 队列类型和仓库历史参考任务实际都分配 L20C；本次没有把队列名误写成硬件结论。

### 6.3 `.venv` 环境漂移

当前共享 `.venv` 并不是历史文档所记录的可用 Python 3.12 环境：

- `.venv/pyvenv.cfg` 当前写的是 Python 3.10.17；
- `.venv/bin/python3 -> /usr/local/bin/python3`，而目标容器没有这个路径；
- 激活后 Bash 会跳过断链，回退到 `/usr/bin/python`，从而加载镜像系统包：PyTorch 2.6.0a0、Transformers 4.48.3，且缺 `loguru`、Qwen3/Qwen3-MoE；
- 因此第一次全量 pytest 在 collection 阶段出现依赖错误，没有进入任何合并代码断言。这个结果不能归因于代码回归。

为避免覆盖用户共享环境或修改系统 Python，本次只读复用仓库中保留的原隔离环境：

```text
.venv.py312-broken-20260729/bin/python
Python 3.12.3
PyTorch 2.9.1+cu128
Transformers 4.56.2
pytest 9.1.1
datasets 3.6.0
loguru 0.7.3
```

虽然目录名带 `broken`，但其解释器、`sys.prefix` 和 site-packages 均自洽；“broken”来自它被后续 `.venv` 修复操作重命名，而不是该 Python 3.12 环境无法执行。全部正式 GPU 验收直接调用此仓库内解释器，不使用系统 Python，也没有安装/升级任何依赖。

## 7. GPU 与分布式验收结果

### 7.1 聚焦 CUDA/Triton/MoE 回归

命令显式覆盖本次新增测试、主线 Triton 内核和 `realq_moe/tests`，并关闭 pytest cache 写入：

```bash
CUDA_VISIBLE_DEVICES=0 \
  .venv.py312-broken-20260729/bin/python -m pytest -q \
  -p no:cacheprovider \
  tests/test_realq_full_block_layer_contract.py \
  tests/test_realq_full_block_refresh.py \
  tests/test_realq_moe_dispatch.py \
  tests/test_realq_reasoning_eval.py \
  tests/test_triton_column_block.py \
  realq_moe/tests
```

结果：`104 passed in 41.27s`，无 skip。由此在真实 CUDA 上补齐本地被跳过的 Triton、packed grouped-mm、joint batched Adam、joint column quant、joint MoE Block-GD、MoE runner 和 GPU-resident numerics。

仓库中所有派生 QTIP 测试不能合并成一个 pytest 进程收集：若把 `tests/` 下 552 项全部同时收集，四个互斥的 QTIP monkey-patch 模块会在 collection 阶段报 `QTIP prepare was already replaced`。这属于既有派生测试隔离约束；与本次 `realq`/MoE/benchmark 集成无关，因此正式门禁采用上面的任务相关集合。

### 7.2 1-rank 与 8-rank full-block 数值等价

新增 [validate_realq_full_block_8gpu.py](../tools/validate_realq_full_block_8gpu.py) 作为可重复验收工具。固定 8 个全局样本和两个各含两层线性层的 toy Transformer block，分别在 world size 1 和 8 下运行同一次 full-block sliding refresh。工具强制验证：

- 当前线性 act-order 已量化自然列前缀 update 严格为 0；
- 当前线性的活动后缀 update 非零且全部有限；
- 当前 block 后续线性 FP32 master 非零更新；
- 下一滑窗 block 的两层线性 FP32 master 都非零更新；
- 四个 optimizer step 从 0 正确推进到 1；
- functional student 不污染两个 teacher module 的参数存储；
- 8 个 rank 的四类结果张量逐字节一致；
- 8-rank 全局样本聚合与 1-rank 参考在明确容差内一致；
- 主 `Config` 的 P01/P02/P03/P04/P05/P06/P10/Triton、slide-window 和 MoE joint/GPU-resident 默认值保持优化态。

运行结果：

| 张量 | 1-rank vs 8-rank max abs | mean abs | 门限 |
|---|---:|---:|---:|
| 当前线性 update | `9.313225746e-10` | `8.731149137e-11` | `3e-05` |
| 当前 block 后续 master | `1.490116119e-08` | `3.783497959e-10` | `3e-05` |
| 下一 block first master | `7.450580597e-09` | `1.746229827e-10` | `3e-05` |
| 下一 block second master | `2.980232239e-08` | `8.149072528e-10` | `3e-05` |

更新幅度也不是舍入噪声：当前活动 update L2=`0.06324547`，当前后续 master delta L2=`0.07999897`，下一 block 两个 master delta L2=`0.07999581/0.07998147`。

### 7.3 主线 benchmark 真实模型生成

新增 [validate_realq_reasoning_gpu.py](../tools/validate_realq_reasoning_gpu.py)，直接调用主线 `realq.benchmarks.run_reasoning_eval`，不经过旧 `realq_benchmark` 包。使用本地 Qwen3-0.6B、GPU0、确定性 non-thinking、每任务 1 题、每题最多 32 token：

- GSM8K：生成和 `scores.json` 落盘；
- MATH-500：使用离线 `math_verify==0.9.0` overlay 判分并落盘；
- HumanEval+：生成并导出 `evalplus_samples.jsonl`；
- LiveCodeBench-lite：生成并导出 `livecodebench_custom_outputs.json`；
- 写出 completed manifest 和 `CODE_EVALUATION.txt`；
- 没有在调试 pod 执行任何模型生成代码。

结果：四任务顺序和文件断言全部通过，manifest=`completed`；将工具参数从弃用别名 `torch_dtype` 改为当前 `dtype` 后重新从空输出目录运行，总耗时 `4.6344s`。

### 7.4 Qwen3-0.6B 真实 8 卡 layer-0 量化

为覆盖 toy probe 之外的模型结构接线，使用 8 个真实 Qwen tokenizer 校准样本（每样本 128 token）运行主线 `python -m realq.ptq`：

- world size 8 / NCCL，每 rank 1 个连续样本；
- 保留默认 `rotate=True`、W4、act-order、rank group-parallel、full slide 和全部性能优化开关；
- `quant_stop_layer=0`，完整量化 Qwen3 layer 0，并把 layer 1 作为滑窗下一 block；
- `backward_samples=8`、全局 `backward_bsz=8`；
- `skip_eval=True`，因为这是结构/数值 smoke，而不是论文质量跑数；
- Stage 0 对 28 个 Transformer block 生成真实 saliency/Fisher，`fisher[0].shape=(1024, 1024)`；
- layer 0 量化用时 `3.61s`，torchrun 正常结束。

schema-4 runtime trace 的自动检查结果：

```json
{
  "records": 74,
  "steps": 73,
  "objectives": ["fisher_mse_full_block"],
  "min_loss": 2.051610135822557e-05,
  "max_loss": 0.0010213757632300258,
  "current_updates": 171,
  "next_updates": 504,
  "max_update_l2": 0.005011774133890867
}
```

73 个 loss 全部为有限数；trace 明确记录 Qwen3 当前 block 的 q/k/v/o/up/gate/down 和下一 block 七个线性层的 active weight/storage/Adam 更新。第一个 q_proj column block 的 slide alpha 为 1，因此只更新当前 block；从第二个 column block 起 alpha<1，next-block 七个 master 均进入同一次 backward，符合设计。

本 smoke 没有报告 Exact KL/PPL：只量化 layer 0、其余层保持 FP，且使用 8×128 的快速校准协议；把这种数字写成论文质量指标没有意义。数值正确性由真实 trace 的有限 loss/update、1-rank/8-rank 张量对齐、最终层 KL full-block 单测和 GPU kernel/MoE 回归共同覆盖。正式质量实验仍应使用完整层数、论文校准规模和 WikiText-2 Exact KL/PPL 协议。

### 7.5 资源收尾

- 所有测试子进程均已退出；
- 8 张卡复核均为 `0 MiB / 0%`，无 compute process；
- Canoe job `j-24iddqgmk3` 保持 `Running`、retry 0，sleep 主进程继续存活，未擅自终止用户申请的调试机；
- 运行产物都写在 pod 本地 `/tmp/realq-j-24iddqgmk3-*`，没有混入 git 工作树或正式实验结果目录。

## 8. 最终结论

主线发布提交 `d5a0b93` 已包含用户指定的三类功能，并满足以下边界：

1. full-block/next-window Block-GD 已进入 `realq`，且没有 Lanczos/层级补偿；
2. Qwen3-MoE 由主入口自动分流到专用稀疏实现；
3. 四项推理集评测直接挂到主线量化模型；
4. 所有要求的性能优化仍默认开启；
5. 本地 331 项回归、真实 GPU 104 项回归、1/8-rank 数值对齐、四任务真实生成和 Qwen3 真实 8 卡 layer-0 量化全部通过；
6. 唯一需要另行维护的是共享 `.venv` 断链；本次没有修改它，且已用仓库内原 Python 3.12 隔离环境完成验证。
