# EfficientQAT controlled comparison launcher

This directory contains the immutable experiment matrix and a fail-closed
launcher for Canoe job `j-7x9o0je4pk`. The user has resolved the training
precision, validation, and optimizer decisions. The plan was enabled only
after the runner, static tests, representative GPU smoke, exact reference
caches, runtime versions, evaluator identity, and controlled-source hashes
all passed.

Enter the pod interactively:

```bash
ssh -F /dev/null -tt \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  u-7s51rmjf43+j-7x9o0je4pk-master-0.canoe-jobs.pod@pa-cne02-prod-01-ssh.xaminim.com
```

The SSH gateway requires a PTY; supplying a one-shot remote command does not
produce command output. Once inside the pod:

```bash
cd /minimax-avatar-new/zhangqian/realq/gptq_plus
.venv/bin/python experiments/efficientqat_compare/launcher.py plan
.venv/bin/python experiments/efficientqat_compare/launcher.py launch
.venv/bin/python experiments/efficientqat_compare/launcher.py status
```

With no subcommand, the launcher defaults to `plan`. `launch` is a dry run and
only prints all six commands. Actual execution additionally requires:

1. every required confirmation in `plan.json` to be `resolved`;
2. `launch_enabled=true`;
3. the runner module to exist;
4. execution from the exact Canoe pod;
5. byte verification of models and the shared token artifact;
6. both `--execute` and the exact
   `--confirm-plan-sha256 <digest>` argument.

There is intentionally no stop, delete, kill, overwrite, or relaunch command.

The resolved compatibility contract is:

- Block-AP uses the released FP16 AMP behavior.
- E2E-QP uses BF16.
- Validation is disabled (`validation_size=0`); all 2048 saved sequences are
  training data.
- E2E-QP uses `torch.optim.AdamW` as a disclosed, user-confirmed compatibility
  substitution for the unavailable CUDA-incompatible `bitsandbytes==0.41.0`.
- QA persists exactly the current shared RealQ evaluator output: each task is
  first rounded to a two-decimal percentage, then the ten-task mean is rounded
  to two decimals. No separate unrounded score path is introduced.
- BF16 weight materialization is CPU-only. Its wall time is recorded, but it
  contributes zero GPU-seconds; quantization GPU-hours cover Block-AP and
  E2E-QP only.

The launcher invokes the future runner with this interface:

```text
python -m experiments.efficientqat_compare.run_one \
  --plan-file <absolute-plan-path> \
  --expected-plan-sha256 <launcher-pinned-plan-sha256> \
  --run-id <run-id> \
  --train-device cuda:0 \
  --eval-device cuda:1 \
  --physical-train-gpu <0..5> \
  --physical-eval-gpu <6|7>
```

Each worker receives exactly two physical GPUs in `CUDA_VISIBLE_DEVICES`.
Consequently, `cuda:0` and `cuda:1` above are worker-local device indices.
Evaluation must also take the plan-defined exclusive lock for its physical
evaluation GPU before loading a model there.
