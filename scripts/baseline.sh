#!/bin/bash

set -o pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <MODEL_PATH> <DEVICE> [NUM_GROUPS]"
    echo "Example: $0 ./modelzoo/Qwen3/Qwen3-0.6B 0 4"
    exit 1
fi

MODEL_PATH=${1}
DEVICE=${2}
NUM_GROUPS=${3:-4}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_NAME=$(basename "${MODEL_PATH}")
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/baseline_${MODEL_NAME}_g${NUM_GROUPS}_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

run_and_log() {
    local script_name=$1

    echo "[$(date +"%F %T")] Running ${script_name}" | tee -a "${LOG_FILE}"

    if [[ "${script_name}" == "gptq_guided.sh" || "${script_name}" == "save_grads.sh" ]]; then
        bash "${SCRIPT_DIR}/${script_name}" "${MODEL_PATH}" "${NUM_GROUPS}" "${DEVICE}" 2>&1 | tee -a "${LOG_FILE}"
    else
        bash "${SCRIPT_DIR}/${script_name}" "${MODEL_PATH}" "${DEVICE}" 2>&1 | tee -a "${LOG_FILE}"
    fi

    local exit_code=${PIPESTATUS[0]}
    if [[ ${exit_code} -ne 0 ]]; then
        echo "[$(date +"%F %T")] ${script_name} failed with exit code ${exit_code}" | tee -a "${LOG_FILE}"
        exit "${exit_code}"
    fi
}

echo "[$(date +"%F %T")] Baseline run started" | tee "${LOG_FILE}"
echo "MODEL_PATH=${MODEL_PATH}" | tee -a "${LOG_FILE}"
echo "DEVICE=${DEVICE}" | tee -a "${LOG_FILE}"
echo "NUM_GROUPS=${NUM_GROUPS}" | tee -a "${LOG_FILE}"
echo "LOG_FILE=${LOG_FILE}" | tee -a "${LOG_FILE}"

run_and_log "bf16.sh"
run_and_log "rtn.sh"
run_and_log "gptq.sh"
run_and_log "gptaq.sh"
run_and_log "save_grads.sh"
run_and_log "gptq_guided.sh"

echo "[$(date +"%F %T")] Baseline run finished successfully" | tee -a "${LOG_FILE}"
