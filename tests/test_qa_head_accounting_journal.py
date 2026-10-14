import copy
from tools.audit_qa_head_accounting_journal import check_row


def row():
    segment = dict(device=0, segment_pool_id=[0, 257], address=1000,
        total_size=100, active_size=80,
        blocks=[dict(address=1000, size=80, state='active_allocated'),
                dict(address=1080, size=20, state='inactive')])
    return dict(pool_id=[0, 257], private_segments=[segment], native_segments=[segment],
        output_device='cuda:0', current_device=0, output_data_ptr=1000, output_payload_bytes=80,
        native_allocated=80, snapshot_native_allocated=80, native_reserved=100,
        snapshot_native_reserved=100, private_reserved=100, mapped_host_live=100,
        private_active=80, accounting_complete=True, private_segments_in_native=True,
        callback_error=0, callback_allocations=257, callback_releases=256)


def test_raw_addresses_and_exact_counter_coverage():
    good = row()
    assert check_row(good, dict(native_allocated=80))
    for key, bad in (('output_data_ptr', 1001), ('native_allocated', 81),
                     ('mapped_host_live', 99), ('callback_error', 1),
                     ('current_device', 1), ('callback_releases', 255)):
        candidate = copy.deepcopy(good)
        candidate[key] = bad
        assert not check_row(candidate, {})
    corrupted = copy.deepcopy(good)
    corrupted['private_segments'][0]['address'] = 9000
    assert not check_row(corrupted, {})


def test_recovery_journal_gate_generated_actual_source():
    from tools.run_storage_qa_mapped_recovery_v2 import source
    text = source()
    compile(text, 'recovery_v2_test', 'exec')
    assert "pair['checks']['independent_complete_raw_pointer_journal'] = journal['valid']" in text
    assert text.index('journal = audit_journal') < text.index("pair['valid'] = pair['capacity_valid']")
