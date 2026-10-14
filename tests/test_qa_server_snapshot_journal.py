import json
import torch
from tools import qa_server_mapped_telemetry_v2 as revised
from tools.qa_mapped_head_natural_batch import NaturalBatchMappedHeadStorage
from test_qa_mapped_head_storage import Library, Pool, CUDA


def fixture(tmp_path, missing=False):
    head = torch.nn.Linear(64, 256, bias=False).bfloat16().eval()
    library = Library()
    cuda = CUDA(missing)
    cuda.current_device = lambda: 0
    def pool():
        result = Pool(library, 16*2*256*2)
        result.id = (0, 257)
        return result
    ticket = revised.JournaledHead(head, journal=tmp_path/'journal.jsonl',
        enabled=True, expected_shape=(256, 64), library=library, cuda=cuda,
        require_cuda=False, pool_factory=pool)
    return head, ticket, library


def test_journal_retains_failed_raw_guard_before_cleanup(tmp_path):
    head, ticket, library = fixture(tmp_path, missing=True)
    error = None
    try:
        with torch.no_grad():
            head(torch.randn(16, 2, 64, dtype=torch.bfloat16))
    except RuntimeError as exc:
        error = str(exc)
    assert error == 'Private mapped output/native accounting incomplete'
    row = json.loads((tmp_path/'journal.jsonl').read_text())
    assert row['accounting_complete'] is False and row['native_allocated'] == 0
    assert row['pool_id'] == [0, 257] and row['private_segments'] and row['native_segments']
    assert len(ticket.batch_releases) == 0
    for handle in ticket.handles: handle.remove()
    ticket.pool.active = 0
    ticket.pool = None
    assert library.qa_live_bytes() == 0


def test_journal_preserves_whole_linear_and_last_consumer(tmp_path):
    head, ticket, library = fixture(tmp_path)
    value = torch.randn(16, 2, 64, dtype=torch.bfloat16)
    with torch.no_grad():
        expected = torch.nn.functional.linear(value, head.weight)
        actual = head(value)
    assert torch.equal(actual, expected)
    row = json.loads((tmp_path/'journal.jsonl').read_text())
    assert row['accounting_complete'] and row['output_data_ptr'] == actual.data_ptr()
    del actual
    ticket.pool.active = 0
    ticket.release_batch()
    result = ticket.close()
    assert result['calls'] == 1 and result['final_host_live'] == 0
    assert library.qa_allocation_count() == library.qa_release_count()
    assert revised.JournaledHead.pre is NaturalBatchMappedHeadStorage.pre
    assert not torch.cuda.is_initialized()


def test_private_composition_does_not_modify_frozen_module_globals():
    from tools import qa_mapped_observed_placement_bridge_v2 as old
    from tools import qa_server_mapped_telemetry as old_telemetry
    assert old.namespace['_qa_admitted_head_ticket'] is old.head_ticket
    assert revised.TelemetryServerBridge is not old_telemetry.TelemetryServerBridge
    assert revised.install.__globals__['SnapshotCUDA'] is revised.SnapshotCUDA
    assert old_telemetry.install.__globals__['SnapshotCUDA'] is not revised.SnapshotCUDA
