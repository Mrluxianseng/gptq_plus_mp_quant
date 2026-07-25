#!/bin/bash

set -u
set -o pipefail

ROOT_DIR=${ROOT_DIR:-/root/lh/gptq_plus}
MODEL_ROOT=${MODEL_ROOT:-/root/lh/llmModels}
DEVICE=${DEVICE:-0}
REST_SECONDS=${REST_SECONDS:-60}
MODEL_NAME=${MODEL_NAME:-Qwen3-0.6B}

if [[ ! -d "${ROOT_DIR}" ]]; then
    echo "ROOT_DIR does not exist: ${ROOT_DIR}"
    exit 1
fi

if [[ ! -d "${MODEL_ROOT}" ]]; then
    echo "MODEL_ROOT does not exist: ${MODEL_ROOT}"
    exit 1
fi

MODEL_PATH="${MODEL_ROOT}/${MODEL_NAME}"
SCRIPT_DIR="${ROOT_DIR}/scripts"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/baselines_2048_${MODEL_NAME}_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

FAILED_STEPS=()

run_and_log() {
    local script_name="$1"

    echo "============================================================" | tee -a "${LOG_FILE}"
    echo "[$(date +"%F %T")] Running ${script_name}" | tee -a "${LOG_FILE}"
    echo "============================================================" | tee -a "${LOG_FILE}"

    (
        cd "${ROOT_DIR}" || exit 1
        bash "${SCRIPT_DIR}/${script_name}" "${MODEL_PATH}" "${DEVICE}"
    ) 2>&1 | tee -a "${LOG_FILE}"

    local exit_code=${PIPESTATUS[0]}
    if [[ ${exit_code} -ne 0 ]]; then
        echo "[$(date +"%F %T")] ${script_name} failed with exit code ${exit_code}" | tee -a "${LOG_FILE}"
        FAILED_STEPS+=("${script_name}")
        return "${exit_code}"
    fi

    echo "[$(date +"%F %T")] ${script_name} finished successfully" | tee -a "${LOG_FILE}"
    if [[ "${REST_SECONDS}" -gt 0 ]]; then
        echo "[$(date +"%F %T")] Resting for ${REST_SECONDS} seconds..." | tee -a "${LOG_FILE}"
        sleep "${REST_SECONDS}"
    fi
}

echo "[$(date +"%F %T")] Baselines 2048 run started" | tee "${LOG_FILE}"
echo "MODEL_PATH=${MODEL_PATH}" | tee -a "${LOG_FILE}"
echo "DEVICE=${DEVICE}" | tee -a "${LOG_FILE}"
echo "LOG_FILE=${LOG_FILE}" | tee -a "${LOG_FILE}"

run_and_log "bf16.sh" || true
run_and_log "rtn.sh" || true
run_and_log "gptq.sh" || true
run_and_log "gptaq.sh" || true

echo "============================================================" | tee -a "${LOG_FILE}"
echo "[$(date +"%F %T")] All steps finished" | tee -a "${LOG_FILE}"
if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
    echo "[$(date +"%F %T")] Failed steps:" | tee -a "${LOG_FILE}"
    printf '  - %s\n' "${FAILED_STEPS[@]}" | tee -a "${LOG_FILE}"
    exit 1
fi
echo "[$(date +"%F %T")] All steps completed successfully." | tee -a "${LOG_FILE}"
