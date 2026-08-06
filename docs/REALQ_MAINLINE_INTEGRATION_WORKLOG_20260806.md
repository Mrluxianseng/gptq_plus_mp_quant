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
| MoE 整合 | 路由统计、zero-route RTN、expert 锁步/joint update | 本地通过 | MoE/Benchmark 联合套件中 CPU 可执行项全部通过；CUDA 项待 8 卡机 |
| Benchmark 整合 | 数据 manifest、resume、数学/代码任务 | 通过 | `tests/test_realq_reasoning_eval.py` 纳入联合套件 |
| 本地总回归 | import、配置默认值、单元/数值测试 | 通过 | `331 passed, 45 skipped` |
| 8 卡 | 环境、分布式运行、数值对齐、Exact KL/PPL | 待执行 | 待补 |

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
