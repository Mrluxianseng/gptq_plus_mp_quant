# Low-bit activation experiment control files

`plan.json` is the machine-readable companion to
`docs/低比特激活值实验记录.md`. The Markdown document remains the human-facing
source of truth for progress and results; this directory prevents command-line
parameters from being reconstructed from memory.

The plan is deliberately fail-closed. Protocol choices are now resolved;
operational fields remain unresolved until the Canoe job is explicitly
submitted and the target-container preflight passes.
`tools/lowbit_activation_runner.py validate` reports every unresolved item,
and command rendering refuses to cross a relevant unresolved boundary.

Examples:

```bash
REALQ_EXPERIMENT_VENV=/minimax-avatar-new/zhangqian/realq/gptq_plus/.venv
source "$REALQ_EXPERIMENT_VENV/bin/activate"

python3 tools/lowbit_activation_runner.py validate
python3 tools/lowbit_activation_runner.py matrix
python3 tools/lowbit_activation_runner.py render \
  --phase tune --method realq --model qwen3-4b --setting 3W16A \
  --grad-lr 5e-5 --cuda-devices 0

# Build each phase/model static cache once before concurrent LR arms.
python3 tools/lowbit_activation_runner.py render \
  --phase precompute --method realq_static --target-phase tune \
  --model qwen3-4b --cuda-devices 0

# GuidedGPTQ requires one saliency precompute per model.
python3 tools/lowbit_activation_runner.py render \
  --phase precompute --method guided_saliency \
  --model qwen3-4b --cuda-devices 0
python3 tools/validate_guided_saliency.py \
  --model modelzoo/Qwen3/Qwen3-4B --sha256 \
  --write-manifest output/lowbit_activation/qwen3-4b/guided_saliency_validation.json
```

The renderer never submits a Canoe job and never launches a process. The
execution wrapper is also dry-run by default; only an explicit `--execute`
launches the rendered argv. It writes a locked, atomic provenance manifest and
merged log in the final run directory:

```bash
python3 tools/lowbit_activation_execute.py \
  --phase tune --method realq --model qwen3-4b --setting 3W16A \
  --grad-lr 5e-5 --cuda-devices 0

# Add only after reviewing the dry-run payload:
python3 tools/lowbit_activation_execute.py --execute \
  --phase tune --method realq --model qwen3-4b --setting 3W16A \
  --grad-lr 5e-5 --cuda-devices 0
```

The proxy command uses per-row `w_groupsize=-1, blocksize=256` as explicitly
requested. Formal runs restore `w_groupsize=128, blocksize=128`.

For Qwen3-32B, `fsdp=false` is non-negotiable. The refactored pipeline does
not consume `bsz`, so OOM fallback uses only effective knobs:
`global_loss_bsz`, `hessian_accum_bsz`, `backward_bsz`,
`final_layer_backward_bsz`, and (during QA evaluation)
`lm_eval_batch_size`.

Before launching, run the target-container preflight with actual task dataset
loading. The preflight and `--execute` both fail closed unless the exact
`AGENT.md` venv above is activated:

```bash
python3 tools/lowbit_activation_preflight.py --load-task-datasets \
  --canoe-job-id j-abmvtvxw97 \
  --write-manifest \
  output/lowbit_activation/runtime_preflight_j-abmvtvxw97.json
```
