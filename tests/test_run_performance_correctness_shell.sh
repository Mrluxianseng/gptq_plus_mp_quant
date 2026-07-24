#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly HARNESS="$REPO_ROOT/tools/run_performance_correctness.sh"
readonly COMPARE_TOOL="$REPO_ROOT/tools/compare_performance_checkpoints.py"
readonly TEST_TMP="$(mktemp -d)"
declare -a TEST_PROCESS_GROUPS=()

cleanup_test() {
    local pid
    for pid in "${TEST_PROCESS_GROUPS[@]:-}"; do
        kill -TERM -- "-$pid" 2>/dev/null ||
            kill -TERM "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    rm -rf -- "$TEST_TMP"
}
trap cleanup_test EXIT

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

bash -n "$HARNESS"
python -m py_compile "$COMPARE_TOOL"

help_output="$(bash "$HARNESS" --help)"
[[ "$help_output" == *"default is a dry-run"* ]] ||
    fail "help must document default dry-run"
[[ "$help_output" == *"--run is supplied explicitly"* ]] ||
    fail "help must document explicit --run"
[[ "$help_output" == *"--candidate-arg NAME=VALUE"* ]] ||
    fail "help must document repeatable candidate args"

set +e
bash "$HARNESS" \
    --case group_stress --gpus 4,5,6 --label invalid \
    >"$TEST_TMP/invalid-count.log" 2>&1
invalid_count_rc=$?
bash "$HARNESS" \
    --case group_stress --gpus 4,5,5,7 --label invalid \
    >"$TEST_TMP/duplicate.log" 2>&1
duplicate_rc=$?
bash "$HARNESS" --run --prepare-only \
    --case group_stress --gpus 4,5,6,7 --label invalid \
    >"$TEST_TMP/modes.log" 2>&1
modes_rc=$?
bash "$HARNESS" >"$TEST_TMP/default-mode.log" 2>&1
default_mode_rc=$?
set -e

((invalid_count_rc != 0)) || fail "three-GPU list was accepted"
((duplicate_rc != 0)) || fail "duplicate GPU list was accepted"
((modes_rc != 0)) || fail "conflicting modes were accepted"
((default_mode_rc != 0)) || fail "missing required case was accepted"
rg -q -- "--gpus must contain exactly four" "$TEST_TMP/invalid-count.log" ||
    fail "wrong invalid-count diagnostic"
rg -q -- "duplicate physical GPU index" "$TEST_TMP/duplicate.log" ||
    fail "wrong duplicate-GPU diagnostic"
rg -q -- "choose at most one mode" "$TEST_TMP/modes.log" ||
    fail "wrong conflicting-mode diagnostic"
rg -q -- "--case is required" "$TEST_TMP/default-mode.log" ||
    fail "default invocation unexpectedly required an explicit mode"

rg -q -F 'MODE="prepare"' "$HARNESS" ||
    fail "default mode is not prepare"
rg -q -F '[[ -n "$GPU_CSV" ]] || die "--gpus is required' "$HARNESS" ||
    fail "explicit GPU quartet is not required"
rg -q -F '"CUDA_VISIBLE_DEVICES=$GPU_CSV"' "$HARNESS" ||
    fail "launch is not scoped by the validated GPU CSV"
rg -q -F 'nvidia-smi --id="$GPU_CSV"' "$HARNESS" ||
    fail "selected-only nvidia-smi mapping/telemetry is missing"
if rg -q -- 'nvidia-smi[[:space:]]+(-L|topo)' "$HARNESS"; then
    fail "harness inventories GPUs outside the selected quartet"
fi
if rg -q -- 'CUDA_VISIBLE_DEVICES=[0-9]' "$HARNESS"; then
    fail "harness contains a hard-coded CUDA device mapping"
fi
if rg -q -- 'nvidia-smi[[:space:]]+--id=[0-9]' "$HARNESS"; then
    fail "harness contains a hard-coded nvidia-smi target"
fi
rg -q -F 'trap cleanup_on_exit EXIT' "$HARNESS" ||
    fail "harness-owned processes have no EXIT cleanup trap"
rg -q -F 'registered_pid_is_direct_child "$pid"' "$HARNESS" ||
    fail "positive-PID cleanup fallback lacks direct-child verification"
rg -q -F 'if [[ "$value" == "$$" ]]; then' "$HARNESS" ||
    fail "direct-child verification does not bind PPid to the harness shell"

mapfile -t telemetry_registration < <(
    sed -n '/^telemetry_pid=\$!$/,/^start_ns=/p' "$HARNESS"
)
[[ "${telemetry_registration[0]:-}" == 'telemetry_pid=$!' ]] ||
    fail "could not locate telemetry PID capture"
[[ "${telemetry_registration[1]:-}" == \
    'ACTIVE_PIDS=("$telemetry_pid")' ]] ||
    fail "telemetry PID is not registered immediately after capture"
mapfile -t command_registration < <(
    sed -n '/^command_pid=\$!$/,/^set +e$/p' "$HARNESS"
)
[[ "${command_registration[0]:-}" == 'command_pid=$!' ]] ||
    fail "could not locate command PID capture"
[[ "${command_registration[1]:-}" == \
    'ACTIVE_PIDS=("$command_pid" "$telemetry_pid")' ]] ||
    fail "command PID is not registered immediately after capture"

# Execute the exact cleanup functions from the harness against two disjoint
# setsid process groups. Only the registered group may be terminated.
process_functions="$(
    sed -n \
        '/^stop_active_processes() {$/,/^on_interrupt() {$/p' \
        "$HARNESS" |
        sed '$d'
)"
eval "$process_functions"
note() { :; }
RUN_ROOT="$TEST_TMP/cleanup-direct"
mkdir -p "$RUN_ROOT"
setsid sleep 300 &
sentinel_pid=$!
TEST_PROCESS_GROUPS+=("$sentinel_pid")
setsid sleep 300 &
owned_pid=$!
TEST_PROCESS_GROUPS+=("$owned_pid")
ACTIVE_PIDS=("$owned_pid")
true
cleanup_on_exit
if kill -0 "$owned_pid" 2>/dev/null; then
    fail "EXIT cleanup left its registered process group alive"
fi
kill -0 "$sentinel_pid" 2>/dev/null ||
    fail "EXIT cleanup killed an unregistered process group"
[[ -f "$RUN_ROOT/ABORTED" ]] ||
    fail "EXIT cleanup did not record ABORTED"
[[ ${#ACTIVE_PIDS[@]} -eq 0 ]] ||
    fail "EXIT cleanup did not clear the owned PID registry"

# Simulate a stale/reused numeric PID that belongs to another parent and is
# not itself a process-group leader. The guarded positive-PID fallback must
# leave it alive.
foreign_pid_file="$TEST_TMP/foreign-child.pid"
setsid bash -c \
    'sleep 300 & printf "%s\n" "$!" >"$1"; wait' \
    _ "$foreign_pid_file" &
foreign_supervisor_pid=$!
TEST_PROCESS_GROUPS+=("$foreign_supervisor_pid")
for _ in {1..100}; do
    [[ -s "$foreign_pid_file" ]] && break
    sleep 0.01
done
[[ -s "$foreign_pid_file" ]] ||
    fail "could not start foreign-parent cleanup fixture"
foreign_pid="$(<"$foreign_pid_file")"
RUN_ROOT="$TEST_TMP/cleanup-foreign"
mkdir -p "$RUN_ROOT"
ACTIVE_PIDS=("$foreign_pid")
true
cleanup_on_exit
kill -0 "$foreign_pid" 2>/dev/null ||
    fail "cleanup killed a PID that was not a direct harness child/group"

RUN_ROOT="$TEST_TMP/cleanup-status"
mkdir -p "$RUN_ROOT"
set +e
(
    # Direct termination behavior is covered above. A nonexistent registered
    # PID exercises the trap's original-status preservation without creating
    # another child that this subshell cannot reap.
    ACTIVE_PIDS=("99999999")
    trap cleanup_on_exit EXIT
    exit 37
)
cleanup_status_rc=$?
set -e
[[ "$cleanup_status_rc" -eq 37 ]] ||
    fail "EXIT cleanup changed the original exit status"
rg -q -F 'aborted rc=37 ' "$RUN_ROOT/ABORTED" ||
    fail "EXIT cleanup did not preserve the original status in ABORTED"

rg -q -F 'status_porcelain_v1_audit_only' "$HARNESS" ||
    fail "untracked status is not retained for audit"
rg -q -F '"tracked_diff_sha256"' "$HARNESS" ||
    fail "tracked source hash gate is missing"
rg -q -F '"compare_tool_sha256"' "$HARNESS" ||
    fail "checkpoint comparator hash gate is missing"
rg -q -F '"mtime_ns"' "$HARNESS" ||
    fail "cache rewrite detection does not include mtime"
for identity in \
    world_size \
    model_artifact_identity \
    harness_sha256 \
    checkpoint_compare_tool_sha256 \
    source_cache_identity
do
    rg -q -F "\"${identity}_equal\"" "$COMPARE_TOOL" ||
        fail "cross-arm compatibility gate is missing $identity"
done
for candidate in \
    quantizer_inner_fastpath \
    w_clip_search_impl \
    w_clip_update_impl \
    w_group_param_layout \
    fisher_fp32_cache \
    act_order_stitch_impl
do
    rg -q -F "[$candidate]=1" "$HARNESS" ||
        fail "reviewed candidate allow-list is missing $candidate"
done

printf 'single-quartet correctness harness shell/static tests passed\n'
