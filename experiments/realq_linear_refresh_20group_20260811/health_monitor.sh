#!/usr/bin/env bash
set -u

node="${1:?usage: health_monitor.sh NODE OUTPUT_ROOT [INTERVAL_SECONDS]}"
output_root="${2:?usage: health_monitor.sh NODE OUTPUT_ROOT [INTERVAL_SECONDS]}"
interval_seconds="${3:-300}"
workspace="/minimax-avatar-new/zhangqian/realq/gptq_plus"
python_bin="${workspace}/.venv.py312-broken-20260729/bin/python"
module="experiments.realq_linear_refresh_20group_20260811.quality"

cd "${workspace}"

while true; do
    timestamp="$(date --iso-8601=seconds)"
    quality_json="$(${python_bin} -m "${module}" status --root "${output_root}" 2>&1)"
    quality_rc=$?
    if [[ ${quality_rc} -eq 0 ]]; then
        quality_success="$(jq -r '.success' <<<"${quality_json}")"
        quality_states="$(jq -c '[.rows[].status] | group_by(.) | map({(.[0]): length}) | add' <<<"${quality_json}")"
    else
        quality_success="status-error"
        quality_states="$(jq -Rn --arg value "${quality_json}" '$value')"
    fi
    driver_count="$(pgrep -af "${module} run-node --node ${node}" | wc -l)"
    worker_count="$(pgrep -af 'realq\.ptq|realq_linear_refresh_20group_20260811\.quality _worker' | wc -l)"
    error_log_count="$(rg -l 'Traceback|CUDA out of memory|OutOfMemoryError|returncode=[1-9]' \
        "${output_root}/runs" -g execution.log 2>/dev/null | wc -l)"
    failure_json_count="$(rg --files "${output_root}/runs" 2>/dev/null | \
        rg -c '/failure\.json$' || true)"

    printf '[%s] node=%s host=%s quality=%s/20 quality_states=%s drivers=%s workers=%s error_logs=%s failure_json=%s\n' \
        "${timestamp}" "${node}" "$(hostname)" "${quality_success}" "${quality_states}" \
        "${driver_count}" "${worker_count}" "${error_log_count}" "${failure_json_count}"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw \
        --format=csv,noheader,nounits | sed 's/^/gpu=/'

    if [[ "${quality_success}" == "20" ]]; then
        printf '[%s] node=%s monitoring complete\n' "${timestamp}" "${node}"
        exit 0
    fi
    sleep "${interval_seconds}"
done
