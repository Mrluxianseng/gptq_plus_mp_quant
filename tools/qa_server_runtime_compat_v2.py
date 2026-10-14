"""Read private pool segments from an unfiltered native snapshot.

PyTorch 2.9.1's filtered snapshot binding wraps borrowed tuple entries in
owning THPObjectPtr objects. A non-small Python pool ID (first 257) can be
decref'd while still referenced, corrupting snapshot integers. Avoid that
binding branch entirely. Never rewrite addresses or allocation counters.
"""
import inspect


def pool_segments(segments, identity):
    identity = tuple(identity)
    if len(identity) != 2 or any(type(value) is not int or value < 0 for value in identity):
        raise RuntimeError('Invalid private native pool identity')
    rows = []
    for segment in segments:
        if tuple(segment['segment_pool_id']) == identity:
            address = segment['address']
            for block in segment['blocks']:
                if block['address'] != address:
                    raise RuntimeError('Native snapshot block/segment address inconsistent')
                address += block['size']
            if address != segment['address'] + segment['total_size']:
                raise RuntimeError('Native snapshot segment coverage incomplete')
            rows.append(segment)
    return rows


class SnapshotCUDA:
    def __init__(self, cuda):
        self.cuda = cuda
        self.include_traces = 'include_traces' in inspect.signature(cuda.memory_snapshot).parameters
        facade = self

        class CompatiblePool(cuda.MemPool):
            def snapshot(self, *, include_traces=False):
                if include_traces is not False:
                    raise RuntimeError('Only allocator segment snapshots are supported')
                # No mempool_id goes through the native Python binding. The
                # allocator itself, tensor pointers and returned rows stay raw.
                return pool_segments(facade.memory_snapshot(), self.id)

        self.MemPool = CompatiblePool

    def memory_snapshot(self, *, include_traces=False):
        if include_traces is not False:
            raise RuntimeError('Only allocator segment snapshots are supported')
        if self.include_traces:
            return self.cuda.memory_snapshot(include_traces=False)
        return self.cuda.memory_snapshot()

    def __getattr__(self, name):
        return getattr(self.cuda, name)
