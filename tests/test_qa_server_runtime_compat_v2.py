from types import SimpleNamespace
from tools.qa_server_runtime_compat_v2 import SnapshotCUDA, pool_segments


def segment(identity, address=1024):
    return dict(segment_pool_id=identity, address=address, total_size=4096,
                blocks=[dict(address=address, size=4096)])


def test_no_filtered_native_snapshot_for_large_ids():
    for identity in ((0, 1), (0, 256), (0, 257), (0, 65537)):
        raw = [segment((0, 0), 8192), segment(identity)]
        class Pool:
            def __init__(self): self.id = identity
            def snapshot(self): raise AssertionError('Unsafe filtered binding called')
        def snapshot(): return raw
        facade = SnapshotCUDA(SimpleNamespace(MemPool=Pool, memory_snapshot=snapshot))
        selected = facade.MemPool().snapshot()
        assert selected == [raw[1]] and selected[0] is raw[1]
        assert facade.memory_snapshot() is raw


def test_corrupt_addresses_fail_closed():
    raw = segment((0, 257))
    raw['blocks'][0]['address'] += 1
    try:
        pool_segments([raw], (0, 257))
    except RuntimeError:
        pass
    else:
        raise AssertionError('Corrupt native snapshot accepted')


def test_new_snapshot_signature_no_pool_id():
    calls = []
    class Pool:
        id = (0, 257)
    def snapshot(*, include_traces=True):
        calls.append(include_traces)
        return [segment((0, 257))]
    facade = SnapshotCUDA(SimpleNamespace(MemPool=Pool, memory_snapshot=snapshot))
    assert facade.MemPool().snapshot()[0]['segment_pool_id'] == (0, 257)
    assert calls == [False]
