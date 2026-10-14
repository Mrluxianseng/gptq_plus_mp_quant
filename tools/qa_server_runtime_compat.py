"""Snapshot API compatibility only; never changes CUDA allocation or GEMM."""
import inspect


class SnapshotCUDA:
    def __init__(self, cuda):
        self.cuda = cuda
        pool_base = cuda.MemPool
        if 'include_traces' in inspect.signature(pool_base.snapshot).parameters:
            self.MemPool = pool_base
        else:
            class CompatiblePool(pool_base):
                def snapshot(self, *, include_traces=False):
                    if include_traces is not False:
                        raise RuntimeError('Only the reviewed allocator segment snapshot is supported')
                    return super().snapshot()
            self.MemPool = CompatiblePool

    def memory_snapshot(self, *, include_traces=False):
        if include_traces is not False:
            raise RuntimeError('Only allocator segment snapshots are supported')
        if 'include_traces' in inspect.signature(self.cuda.memory_snapshot).parameters:
            return self.cuda.memory_snapshot(include_traces=False)
        return self.cuda.memory_snapshot()

    def __getattr__(self, name):
        return getattr(self.cuda, name)


def probe_source():
    from pathlib import Path
    path = Path(__file__).with_name('qa_mapped_host_pool_probe.py')
    text = path.read_text()
    anchor = '    import torch\n'
    if text.count(anchor) != 1:
        raise RuntimeError('Frozen runtime probe import changed')
    text = text.replace(anchor, anchor + '    from tools.qa_server_runtime_compat import SnapshotCUDA\n'
                        '    from types import SimpleNamespace\n'
                        '    torch = SimpleNamespace(**dict(torch.__dict__, cuda=SnapshotCUDA(torch.cuda)))\n')
    anchor = '    files=[Path(__file__),source,binary,runtime,'
    if text.count(anchor) != 1:
        raise RuntimeError('Frozen runtime probe source-binding changed')
    text = text.replace(anchor, '    files=[Path(__file__),Path("tools/qa_mapped_host_pool_probe.py"),'
                        'Path("tools/qa_server_runtime_compat.py"),source,binary,runtime,')
    compile(text, 'qa_server_runtime_probe', 'exec')
    return text


if __name__ == '__main__':
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    exec(compile(probe_source(), 'qa_server_runtime_probe', 'exec'),
         dict(__name__='__main__', __file__=str(Path(__file__).with_name('qa_mapped_host_pool_probe.py'))))
