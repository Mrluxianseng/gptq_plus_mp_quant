#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/gptq_plus_realq
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/uky/gptq_plus/.venv/bin/python
RUN=outputs/overnight_20260927_0143
mkdir -p "$RUN"
case "${1:-preflight}" in
  followup-queue)
    "$PY" -u tools/overnight_followup_queue.py 2>&1 | tee "$RUN/followup_queue.log"
    ;;
  preflight)
    "$PY" -u -c 'import sys,torch,transformers; print(sys.version); print("torch",torch.__version__,"cuda",torch.version.cuda,"transformers",transformers.__version__); print(torch.cuda.get_device_name()); x=torch.randn(512,512,device="cuda",requires_grad=True); (x@x.T).square().mean().backward(); torch.cuda.synchronize(); print("CUDA_FORWARD_BACKWARD_OK",float(x.grad.norm()),"peak",torch.cuda.max_memory_allocated())' 2>&1 | tee "$RUN/preflight.log"
    ;;
  prefix-unit)
    "$PY" -u tools/overnight_prefix_unit.py 2>&1 | tee "$RUN/prefix_unit.log"
    ;;
  papers)
    "$PY" -u tools/overnight_fetch_papers.py 2>&1 | tee "$RUN/papers.log"
    ;;
  math-audit)
    "$PY" -u tools/overnight_math_audit.py 2>&1 | tee "$RUN/math_audit.log"
    ;;
  baseline|baseline-smoke|baseline-profile|cross-audit|cross-fixed|prefix-verify|prefix-fast)
    MODE="$1"
    test ! -e "$RUN/$MODE.log"
    ENTRY=(./ptq.py)
    EXTRA=()
    if [[ "$MODE" == cross-* ]]; then
      ENTRY=(tools/overnight_ptq_probe.py "$MODE")
      EXTRA=(--cross_module_compensate --quant_stop_layer 4)
    elif [[ "$MODE" == baseline-smoke ]]; then
      EXTRA=(--quant_stop_layer 4)
    elif [[ "$MODE" == baseline-profile ]]; then
      ENTRY=(tools/overnight_ptq_probe.py "$MODE")
    elif [[ "$MODE" == prefix-* ]]; then
      ENTRY=(tools/overnight_ptq_probe.py "$MODE")
      if [[ "$MODE" == prefix-verify ]]; then
        EXTRA=(--quant_stop_layer 1 --nsamples 4 --backward_samples 4)
      fi
    fi
    timeout --signal=TERM --kill-after=30s 3600 "$PY" -u -m torch.distributed.run \
      --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=localhost:29427 "${ENTRY[@]}" \
      --model /mnt/d/llamaModels/Qwen3-0.6B --exp "overnight_20260927_$MODE" \
      --dataset wikitext2 --nsamples 32 --seq_len 256 \
      --w_method gptq_plus --w_bits 4 --w_clip --num_groups 4 --blocksize 128 --act_order \
      --kl_topk -1 --bsz 1 --final_layer_stats_bsz 4 --alpha 0.0 --enable_gptq_plus 0 \
      --backward_samples 32 --backward_bsz 4 --final_layer_backward_bsz 4 \
      --g_update_mode block_gd --grad_lr 5e-5 --grad_optimizer adam \
      --grad_refresh_loss fisher_diag_mse --global_loss --global_loss_bsz 1 \
      --grad_clip 5e-5 --final_layer_grad_clip 5e-4 --final_layer_grad_lr 1e-6 \
      --group_parallel_quant none --eval_seq_len 1024 --eval_datasets wikitext2 \
      "${EXTRA[@]}" 2>&1 | tee "$RUN/$MODE.log"
    ;;
  *) echo 'Unknown action' >&2; exit 2 ;;
esac
