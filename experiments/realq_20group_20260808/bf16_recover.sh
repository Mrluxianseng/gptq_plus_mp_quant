#!/usr/bin/env bash
set -u

node="${1:?usage: bf16_recover.sh NODE OUTPUT_ROOT}"
output_root="${2:?usage: bf16_recover.sh NODE OUTPUT_ROOT}"
workspace="/minimax-avatar-new/zhangqian/realq/gptq_plus"
python_bin="${workspace}/.venv.py312-broken-20260729/bin/python"
export REALQ_PYTHON="${python_bin}"
module="experiments.realq_20group_20260808.bf16"
driver_module="experiments.realq_20group_20260808.quality"
log_dir="${output_root}/orchestration"

mkdir -p "${log_dir}"
cd "${workspace}"

case "${node}" in
    0)
        producers=("qwen3-32b:0" "qwen3-4b:1" "qwen3-0.6b:2")
        ;;
    1)
        producers=("qwen3-8b:0" "llama31-8b-instruct:1")
        ;;
    *)
        echo "node must be 0 or 1" >&2
        exit 2
        ;;
esac

pids=()
labels=()
for producer in "${producers[@]}"; do
    model="${producer%%:*}"
    gpu="${producer##*:}"
    echo "[$(date --iso-8601=seconds)] starting BF16 reference producer model=${model} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m "${module}" _reference \
        --root "${output_root}" --model "${model}" \
        > "${log_dir}/bf16_reference_${model}.log" 2>&1 &
    pids+=("$!")
    labels+=("${model}")
done

failed=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "[$(date --iso-8601=seconds)] BF16 reference ready model=${labels[$index]}"
    else
        rc=$?
        echo "[$(date --iso-8601=seconds)] BF16 reference failed model=${labels[$index]} rc=${rc}" >&2
        failed=1
    fi
done
if [[ ${failed} -ne 0 ]]; then
    exit 1
fi

echo "[$(date --iso-8601=seconds)] starting claim-aware recovery driver node=${node}"
exec "${python_bin}" -m "${driver_module}" run-node \
    --node "${node}" --root "${output_root}"
