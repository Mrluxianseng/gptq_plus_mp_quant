#!/bin/bash
# Three-arm smoke comparison, run sequentially in one go:
#   baseline : plain block_gd (no cross-module compensation)
#   cross    : sequential closed-form (feeder -> next residual-stream module)
#   all      : closed-form spread over EVERY not-yet-quantised module (split)
# Each arm writes outputs/smoke_<mode>.log via smoke_situation_b.sh.
#
# Usage (WSL, from the worktree root):
#   QUANT_STOP_LAYER=4 nohup bash scripts/smoke_cross_compare.sh \
#       > outputs/smoke_compare_queue.log 2>&1 &
#   tail -F outputs/smoke_compare_queue.log
set -euo pipefail

export QUANT_STOP_LAYER="${QUANT_STOP_LAYER:-4}"

for MODE in baseline cross all; do
    echo "===== arm ${MODE} start $(date +%H:%M:%S) ====="
    bash scripts/smoke_situation_b.sh "${MODE}"
    echo "===== arm ${MODE} done  $(date +%H:%M:%S) ====="
done

echo "===== all arms done ====="
