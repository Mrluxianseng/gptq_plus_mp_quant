"""User's scalar global capacity policy, retaining the old per-GPU rejection.

This versioned adapter does not change the frozen formal runner, algorithm,
completed statuses or telemetry. Individual GPU/interval/point regressions stay
visible as diagnostics; exact quality, sources and complete telemetry remain
mandatory. The old audit is embedded without editing its checks or valid flag.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_storage_method_formal as formal
from tools.storage_process_device_peaks import valid_record

ORIGINAL_AUDIT = formal.audit_pair
PER_GPU_CHECK = 'sampled_each_physical_gpu_peak_nonincreasing'
REQUIRED_CHECKS = {
    'complete_exact_numerics_sources_commands_runtime_gpu',
    'complete_prequant_state_and_all_rank_ownership', 'owner_all28_nonempty_exact',
    'complete_cache_inventory', 'method_coverage', 'stats_holding_bytes_same',
    'complete_exact_intervals', 'separate_points_complete', 'every_stage_rank_has_intervals',
    'all_full_global_peaks_nonincreasing', 'bound_sources_unchanged',
    'all_visible_device_process_peaks_complete',
}


def global_policy(original, *, all_visible_device_peaks_complete=None):
    checks = dict(original['checks'])
    if all_visible_device_peaks_complete is not None:
        checks['all_visible_device_process_peaks_complete']=all_visible_device_peaks_complete
    diagnostic = checks.pop(PER_GPU_CHECK)
    checks['all_required_checks_present'] = REQUIRED_CHECKS <= checks.keys()
    devices = original['sampled_nvml_peaks_by_uuid']
    checks['physical_gpu_identities_complete_and_same'] = bool(devices[0]) and devices[0].keys() == devices[1].keys()
    # Recompute the three inequalities from raw values, not a stored valid flag.
    peaks = original['full_global_peaks']
    metrics = ('full_process_peak_allocated_bytes', 'full_process_peak_reserved_bytes', 'max_nvml_memory_used_mib')
    checks['all_full_global_peaks_nonincreasing'] = (
        set(peaks) == set(metrics) and all(len(peaks[k]) == 2 and
        all(type(v) is int and v > 0 for v in peaks[k]) and
        peaks[k][1] <= peaks[k][0] for k in metrics))
    valid = all(checks.values())
    return dict(valid=valid, capacity_valid=valid,
                scope=original.get('scope'), precompute_measured=original.get('precompute_measured'),
                policy='user_global_process_and_device_maxima_v1', checks=checks,
                per_gpu_peak_nonincreasing_diagnostic=diagnostic,
                full_global_peaks=peaks, sampled_nvml_peaks_by_uuid=devices,
                interval_count=original['interval_count'], points=original['points'],
                strict_interval_point_valid=valid and not original['interval_regressions'] and original['points']['valid'],
                original_per_gpu_gate_report=original,
                criterion='Exact quality/source/cache/method/complete telemetry plus global allocated/reserved/device-max NVML <= control. Each GPU, interval and point regression retained as diagnostics; NVML sampled.')


def audit_pair(control, candidate, ranks, bindings):
    # Fresh full numerical/provenance/cache/coverage/interval/point audit first.
    original=ORIGINAL_AUDIT(control, candidate, ranks, bindings)
    complete=True
    for root in (control,candidate):
        status=formal.read(root/'status.json')
        stages=('stage2',) if status.get('scope')=='stage2_cached' else ('stage0','stage2')
        for stage in stages:
            rows=status.get(f'{stage}_process_allocator_peaks_by_rank',[])
            complete &= len(rows)==ranks and all(valid_record(row,ranks) for row in rows)
    return global_policy(original,all_visible_device_peaks_complete=bool(complete))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--ranks', type=int, required=True)
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit_pair(args.control, args.candidate, args.ranks, formal.read(args.bindings))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(dict(valid=report['valid'], checks=report['checks'], full_global_peaks=report['full_global_peaks']), indent=2))
    if not report['valid']:
        raise SystemExit('Global capacity or required complete evidence failed; original artifacts retained.')


if __name__ == '__main__':
    main()
