#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly HARNESS="$REPO_ROOT/tools/run_performance_correctness.sh"
readonly COMPARE_TOOL="$REPO_ROOT/tools/compare_performance_checkpoints.py"
readonly TEST_TMP="$(mktemp -d)"
trap 'rm -rf -- "$TEST_TMP"' EXIT

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
rg -q -F 'status_porcelain_v1_audit_only' "$HARNESS" ||
    fail "untracked status is not retained for audit"
rg -q -F '"tracked_diff_sha256"' "$HARNESS" ||
    fail "tracked source hash gate is missing"
rg -q -F '"compare_tool_sha256"' "$HARNESS" ||
    fail "checkpoint comparator hash gate is missing"
rg -q -F '"mtime_ns"' "$HARNESS" ||
    fail "cache rewrite detection does not include mtime"
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
