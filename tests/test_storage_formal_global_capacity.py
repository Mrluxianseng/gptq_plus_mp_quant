from copy import deepcopy

import pytest

from tools.audit_storage_formal_global_capacity import global_policy


def complete_report():
    return dict(valid=False, checks={
        'complete_exact_numerics_sources_commands_runtime_gpu': True,
        'complete_prequant_state_and_all_rank_ownership': True,
        'owner_all28_nonempty_exact': True, 'complete_cache_inventory': True,
        'method_coverage': True, 'stats_holding_bytes_same': True,
        'complete_exact_intervals': True, 'separate_points_complete': True,
        'every_stage_rank_has_intervals': True, 'all_full_global_peaks_nonincreasing': True,
        'sampled_each_physical_gpu_peak_nonincreasing': False, 'bound_sources_unchanged': True,
        'all_visible_device_process_peaks_complete': True},
        full_global_peaks={'full_process_peak_allocated_bytes': [6713250816,6712304640],
                          'full_process_peak_reserved_bytes': [9254731776,9254731776],
                          'max_nvml_memory_used_mib': [19104,19104]},
        sampled_nvml_peaks_by_uuid=[{'gpu0':7868,'gpu1':19104},{'gpu0':9912,'gpu1':19104}],
        interval_count=110562, points={'valid':False,'matched_values':442248,'missing':[]},
        interval_regressions=[{'metric':'reserved'}])


def test_only_per_gpu_regression_becomes_diagnostic_old_evidence_unchanged():
    original=complete_report()
    before=deepcopy(original)
    report=global_policy(original)
    assert report['valid'] and not report['per_gpu_peak_nonincreasing_diagnostic']
    assert not report['strict_interval_point_valid']
    assert report['original_per_gpu_gate_report'] == original == before
    assert not original['valid']


@pytest.mark.parametrize('metric', ['full_process_peak_allocated_bytes','full_process_peak_reserved_bytes','max_nvml_memory_used_mib'])
def test_each_global_increase_is_rejected_even_if_old_check_claims_true(metric):
    original=complete_report()
    original['full_global_peaks'][metric][1] = original['full_global_peaks'][metric][0]+1
    assert not global_policy(original)['valid']


@pytest.mark.parametrize('check', [k for k in complete_report()['checks'] if k not in (
    'sampled_each_physical_gpu_peak_nonincreasing','all_full_global_peaks_nonincreasing')])
def test_every_quality_source_method_coverage_and_telemetry_failure_still_rejected(check):
    original=complete_report(); original['checks'][check]=False
    assert not global_policy(original)['valid']


def test_missing_gpu_identity_and_missing_global_measurement_rejected():
    original=complete_report(); del original['sampled_nvml_peaks_by_uuid'][1]['gpu0']
    assert not global_policy(original)['valid']
    original=complete_report(); original['full_global_peaks']['max_nvml_memory_used_mib']=[0,0]
    assert not global_policy(original)['valid']


def test_missing_mandatory_check_is_not_silently_accepted():
    original=complete_report(); del original['checks']['method_coverage']
    assert not global_policy(original)['valid']


def test_adapter_executes_original_main_and_binds_new_sources(monkeypatch, tmp_path):
    from tools import run_storage_formal_global_capacity as adapter
    original_save=adapter.formal.save
    original_sources=adapter.formal.FORMAL_TOOL_SOURCES
    monkeypatch.setattr(adapter.formal, 'save', original_save)
    monkeypatch.setattr(adapter.formal, 'FORMAL_TOOL_SOURCES', original_sources)
    monkeypatch.setattr(adapter.formal, 'audit_pair', adapter.formal.audit_pair)
    def run():
        assert adapter.formal.audit_pair is adapter.audit_pair
        assert set(adapter.formal.FORMAL_TOOL_SOURCES)-set(original_sources) == {
            'tools/audit_storage_formal_global_capacity.py','tools/run_storage_formal_global_capacity.py',
            'tools/storage_process_device_peaks.py','tools/run_storage_formal_multidevice_profile.py'}
        adapter.formal.save(tmp_path/'plan.json', {'jobs':[], 'optional_precompute_jobs':{}})
    monkeypatch.setattr(adapter.formal,'main',run)
    adapter.main()
    assert adapter.formal.read(tmp_path/'plan.json')['acceptance_policy']=='user_global_process_and_device_maxima_v1'
