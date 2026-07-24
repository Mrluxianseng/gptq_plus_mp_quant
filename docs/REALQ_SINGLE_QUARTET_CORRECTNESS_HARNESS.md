# REAL-Q 单 quartet correctness/checkpoint A/B harness

## 目的与边界

`tools/run_performance_correctness.sh` 用一个显式指定的 4 卡集合运行一个
Stage-1 case，并保存完整 checkpoint。它用于验证默认实现与 P01/P02/P03/
P04/P06 等性能候选是否保持逐 tensor 的 raw-byte 一致性，不用于得出性能
加速结论。

脚本不会隐式选择 GPU，也没有“另一组卡”的概念：

- 必须传 `--gpus I,J,K,L`，且恰好是 4 个不同的物理卡号；
- GPU inventory、空闲检查、telemetry 都只查询这 4 张卡；
- `torchrun` 固定为 4 rank，`CUDA_VISIBLE_DEVICES` 等于校验后的同一列表；
- 默认是 prepare-only dry-run；只有显式 `--run` 才启动 GPU 进程。

## 固定 fixture

case、数据、seed 和主要数学配置沿用
`tools/run_performance_timing.sh`/`tools/run_performance_baseline.sh` 的
Qwen3-4B Stage-1 fixture：

- W2，`nsamples=4`，`seq_len=128`，只量化 layer 0；
- `act_order=true`，`group_parallel_quant=rank`；
- `skip_eval=true`、`lm_eval=false`；
- 使用已验证的 wikitext2 token cache 和四个 Stage-0 rank cache；
- 每个 run 保存 `checkpoint/model.pt`。

支持的 case：

- `group_stress`：group-128，A/K/V 16-bit，unaware；
- `per_row`：per-row，A/K/V 16-bit，unaware；
- `group_akv_aware`：group-128，A4/K4/V4，clip 0.9，aware；
- `block_gd_stress`：group-128，Block-GD 与 loss sliding window。

`--candidate-arg NAME=VALUE` 只能使用已审查性能开关：

- `quantizer_inner_fastpath`（P01）
- `w_clip_search_impl`（P02）
- `w_clip_update_impl`（P03）
- `w_group_param_layout`（P04）
- `fisher_fp32_cache`（P05）
- `act_order_stitch_impl`（P06）

数学、数据、seed、路径、日志和执行协议字段不能通过 candidate arg 修改。
脚本会解析 baseline/candidate 两份 `Config`，要求实际差异与声明的 candidate
字段完全相等；选择当前默认值形成 no-op arm 也会失败。

## 推荐执行方式

先对 baseline 做 dry-run，核对 `command.txt`、`resolved_config.json` 和物理
GPU UUID 映射。下列命令不会启动 GPU 进程：

```bash
bash tools/run_performance_correctness.sh \
  --case group_stress \
  --gpus 4,5,6,7 \
  --baseline-id p01-p06-correctness-20260724 \
  --label legacy
```

确认后重新执行并显式加 `--run`：

```bash
bash tools/run_performance_correctness.sh --run \
  --case group_stress \
  --gpus 4,5,6,7 \
  --master-port 30420 \
  --baseline-id p01-p06-correctness-20260724 \
  --label legacy
```

候选 arm 使用同一 commit、case、GPU 顺序和 baseline-id，只改变声明的开关：

```bash
bash tools/run_performance_correctness.sh --run \
  --case group_stress \
  --gpus 4,5,6,7 \
  --master-port 30421 \
  --baseline-id p01-p06-correctness-20260724 \
  --label p01-p02 \
  --candidate-arg quantizer_inner_fastpath=true \
  --candidate-arg w_clip_search_impl=symmetric_union_exact
```

每次 invocation 都生成一个新的 run root。脚本不会恢复或覆盖已有目录。

## 严格 checkpoint 对比

每个成功 run 会产生：

- `checkpoint/state_manifest.json`：每个 canonical tensor 的 key、dtype、
  shape、nbytes 和 raw-byte SHA256；
- `summary.json`：完整 state SHA256、checkpoint archive SHA256、验证结果；
- `manifest.json`：commit、config、cache、model、GPU mapping 和 harness hash。

比较 baseline/candidate 时应传两个 run root：

```bash
python tools/compare_performance_checkpoints.py compare \
  BASELINE_RUN_ROOT \
  CANDIDATE_RUN_ROOT \
  -o checkpoint_compare.json
```

通过条件是：

1. canonical tensor key 集合完全相同；
2. 每个 tensor 的 dtype、shape、nbytes 和 raw bytes 完全相同；
3. 完整 canonical-state SHA256 相同；
4. 两个 run 都成功，且 commit、case、baseline-id、物理 GPU ID/UUID 顺序
   相同；
5. world size、model artifact identity、harness/comparator SHA256 和输入
   source-cache identity 相同且字段完整；
6. 忽略 artifact 路径后，两份 resolved config 的差异恰好等于声明的
   candidate 字段。

`torch.save` archive SHA256 不要求相同，因为 archive 元数据可能不稳定。
性能开关会进入 checkpoint provenance，因此 checkpoint metadata 的预期差异
只记录为 audit 信息，不代替 tensor-byte 门禁。

也可以传两个裸 `model.pt`；此时仍做严格 tensor-byte 对比，但无法验证 A/B
run provenance，报告会明确标记 compatibility 不可用。

## Provenance 与 cache 门禁

run root 记录：

- Git commit、tracked/staged binary diff hash 与字节数；
- untracked status（仅 audit，不因用户在共享目录新增 untracked 输出而失败）；
- harness 与 checkpoint comparator 的 SHA256；
- resolved config、config diff、命令和 Python package 版本；
- model artifact identity；
- smoke cache 来源、run-local cache 副本的 size/mtime/mode/SHA256；
- selected physical GPU ID/UUID、PCI bus、driver、显存和 rank 映射；
- 运行日志、selected-only telemetry、wall time、checkpoint/state hash。

缓存源文件从不作为写入目标。验证过的 cache 通过 reflink/copy 放入 run-local
目录，文件和目录设置为只读；运行前后同时比较 source 与副本的
size/mtime/mode/SHA256。`runtime_cache` 是单独的可写目录。任何 Stage-0
recompute、cache write、token save 或数据下载日志都会令验证失败。

最终 source gate 严格要求 commit、tracked/staged bytes、harness、
comparator 和 model identity 在运行前后不变。untracked churn 只保留在审计
字段中。

GPU 模式安装了 EXIT cleanup trap。telemetry 和 torchrun 的 `setsid`
process-group PID 会在启动后立即登记；任何异常 shell 退出都会只终止登记的
harness-owned process group 并写入 `ABORTED`。已经 `wait`/reap 的 command
PID 会立即从登记表移除；若进程组尚未来得及建立，正 PID fallback 也必须先
通过 `/proc/<pid>/status` 证明它仍是 harness shell 的直接子进程。

## 已知风险

- 这是 layer-0 correctness fixture，不能替代最终完整模型的量化与评测；
- raw-byte 相等是强门禁；不相等时需要定位数值差异，不能用“误差很小”直接
  宣称数学等价；
- checkpoint comparator 针对 REAL-Q 的普通 dense fake-quantized
  `state_dict`；如果未来保存真正的 PyTorch quantized/sparse tensor，需要
  扩展 byte-view 逻辑；
- correctness run 的 wall time 只作诊断记录，不能与隔离 timing harness 的
  主指标混用。
