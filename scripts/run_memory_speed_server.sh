#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_PATH="${MODEL_PATH:-/openbayes/home/llmModels/Qwen3-0.6B}"
GPU_INDICES="${GPU_INDICES:-0,1,2,3}"
PYTHON="${PYTHON:-$(command -v python)}"
PAIRS="${PAIRS:-3}"
TAG_PREFIX="${TAG_PREFIX:-realq-memory-speed-$(date +%Y%m%d-%H%M%S)}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

IFS=',' read -r -a GPU_INDEX_ARRAY <<< "$GPU_INDICES"
if [[ "${#GPU_INDEX_ARRAY[@]}" -ne 4 ]]; then
  echo "ERROR: GPU_INDICES must name exactly four devices; got: $GPU_INDICES" >&2
  exit 2
fi
declare -A SEEN_GPU_INDICES=()
for gpu_index in "${GPU_INDEX_ARRAY[@]}"; do
  if [[ ! "$gpu_index" =~ ^[0-9]+$ || -n "${SEEN_GPU_INDICES[$gpu_index]:-}" ]]; then
    echo "ERROR: GPU_INDICES must contain four distinct numeric indices; got: $GPU_INDICES" >&2
    exit 2
  fi
  SEEN_GPU_INDICES[$gpu_index]=1
done

mkdir -p logs outputs
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "ERROR: model config not found: $MODEL_PATH/config.json" >&2
  exit 2
fi
if [[ ! -d "$ROOT/datasets/wikitext" ]]; then
  echo "ERROR: WikiText-2 directory not found: $ROOT/datasets/wikitext" >&2
  exit 2
fi
if [[ -e "$ROOT/outputs/${TAG_PREFIX}-campaign" ]]; then
  echo "ERROR: campaign already exists; choose a new TAG_PREFIX: $ROOT/outputs/${TAG_PREFIX}-campaign" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU_INDICES"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"

echo "=== REAL-Q 4-GPU preflight ==="
echo "tag=$TAG_PREFIX"
echo "model=$MODEL_PATH"
echo "dataset=$ROOT/datasets/wikitext"
echo "visible_physical_gpus=$CUDA_VISIBLE_DEVICES"
echo "python=$PYTHON"
echo "omp_num_threads=$OMP_NUM_THREADS"
echo "tqdm_mininterval=$TQDM_MININTERVAL"
echo "nccl_shm_disable=$NCCL_SHM_DISABLE"
"$PYTHON" - <<'PY'
import torch
import triton

print(f"python_torch={torch.__version__} torch_cuda={torch.version.cuda}")
print(f"triton={triton.__version__} cuda_available={torch.cuda.is_available()}")
print(f"visible_cuda_devices={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"cuda:{i}={torch.cuda.get_device_name(i)}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
    raise SystemExit("ERROR: this benchmark requires exactly four visible CUDA devices")
PY

echo "=== 4-rank NCCL collective preflight ==="
"$PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  tools/test_distributed_cuda.py

echo "=== 4-rank Triton kernel parity smoke ==="
"$PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  tools/test_triton_rank_kernel.py

echo "=== Paper-config paired full-model benchmark ==="
"$PYTHON" -u tools/run_triton_paper_benchmark.py \
  --model-path "$MODEL_PATH" \
  --gpu-indices "$GPU_INDICES" \
  --pairs "$PAIRS" \
  --tag-prefix "$TAG_PREFIX"
