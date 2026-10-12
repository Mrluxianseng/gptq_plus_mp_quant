#!/usr/bin/env bash
# Formal W4A16: serial 0.6B C/N then 1.7B C/N, fresh cache per arm.
set -Eeuo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$ROOT_DIR"
PYTHON_BIN=${PYTHON:-python}
MODEL_ROOT=${MODEL_ROOT:-/minimax-avatar-new/zhangqian/realq/gptq_plus/modelzoo/Qwen3}
MODEL06=${MODEL06:-$MODEL_ROOT/Qwen3-0.6B}
MODEL17=${MODEL17:-$MODEL_ROOT/Qwen3-1.7B}
DEVICE=${DEVICE:-0,1,2,3}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT_DIR/outputs}
MODE=--dry-run
if [[ ${1:-} == --run || ${1:-} == --dry-run ]]; then
    MODE=$1
    shift
fi
# The controller refuses an existing root; every invocation keeps its evidence.
RUN_DIR=${RUN_DIR:-$OUTPUT_ROOT/storage-formal-$(date +%Y%m%d-%H%M%S)-$$}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-1}
exec "$PYTHON_BIN" -u tools/run_storage_method_formal.py "$MODE" \
    --model06 "$MODEL06" --model17 "$MODEL17" --devices "$DEVICE" \
    --output-directory "$RUN_DIR" "$@"
