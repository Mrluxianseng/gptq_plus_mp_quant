"""Server QA interface: frozen math, actual cache and runtime provenance.

This is an explicit validation entry; it does not enable the default launcher.
"""
import hashlib
import inspect
import json
from pathlib import Path

from tools.qa_mapped_observed_placement_bridge_v3 import MappedObservedPlacementBridge
from tools.qa_server_runtime_compat import SnapshotCUDA


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_runtime(root):
    root = Path(root)
    row = json.loads((root/'status.json').read_text())
    bindings = json.loads((root/'source_bindings.json').read_text())
    if not row['valid'] or not row['native_accounting_complete'] or not bindings:
        raise RuntimeError('Server mapped runtime/accounting admission missing')
    if not all(sha(p) == v for p, v in bindings.items()):
        raise RuntimeError('Server runtime source/binary/library changed')
    return root/'mapped_host_probe.so', bindings


class ServerBridge(MappedObservedPlacementBridge):
    def observed_eval(self, model, tokenizer, *args, **kwargs):
        rows = []
        pending = []
        def before(module, inputs, keywords):
            cache = keywords.get('past_key_values')
            pending.append(dict(use_cache=bool(keywords.get('use_cache', module.config.use_cache)),
                cache_type=None if cache is None else type(cache).__qualname__,
                cache_offloading=bool(getattr(cache, 'offloading', False))))
        def after(module, inputs, result):
            cache = result.past_key_values
            row = pending.pop()
            if not row['use_cache'] or cache is None:
                raise RuntimeError('Original QA KV computation is required')
            row['output_cache_type'] = type(cache).__qualname__
            row['layers'] = []
            for layer in cache.layers:
                k, v = layer.keys, layer.values
                if k is None or v is None or not k.numel() or not v.numel():
                    raise RuntimeError('Incomplete actual K/V geometry')
                row['layers'].append(dict(key_shape=list(k.shape), value_shape=list(v.shape),
                    key_dtype=str(k.dtype), value_dtype=str(v.dtype),
                    key_device=str(k.device), value_device=str(v.device)))
            if len(row['layers']) != 28:
                raise RuntimeError('All 28 actual cache layers are required')
            rows.append(row)
        handles = [model.register_forward_pre_hook(before, with_kwargs=True),
                   model.register_forward_hook(after)]
        completed = False
        try:
            result = super().observed_eval(model, tokenizer, *args, **kwargs)
            completed = True
            return result
        finally:
            for handle in handles:
                handle.remove()
            record = dict(completed=completed and bool(rows) and not pending,
                calls=rows, pending=len(pending),
                original_linear_source_sha256=sha(inspect.getfile(__import__('torch').nn.Linear)),
                scope='Actual original KV/geometry observations, not cache numerical equivalence by itself')
            with (self.root/'qa_actual_cache_rank0.json').open('x') as stream:
                json.dump(record, stream, indent=2)


def install(eval_module, dist_module, lm_eval, memory_module, torch, *, enabled,
            destination, expected_tasks, runtime_root, bindings):
    from tools.qa_mapped_head_storage_prototype import prepare_runtime
    binary, runtime_bindings = verify_runtime(runtime_root)
    # Early setup must have happened in a fresh process before model CUDA use.
    library = __import__('ctypes').CDLL(str(binary))
    for name in ('qa_live_bytes', 'qa_peak_bytes', 'qa_allocation_count', 'qa_release_count'):
        getattr(library, name).restype = __import__('ctypes').c_size_t
    allocator = (torch.cuda.memory.CUDAPluggableAllocator(str(binary), 'qa_mapped_alloc', 'qa_mapped_free')
                 if enabled else None)
    return ServerBridge(eval_module, dist_module, lm_eval, memory_module, SnapshotCUDA(torch.cuda),
        enabled=enabled, destination=destination, expected_tasks=expected_tasks,
        validation_only=True, exclusive_peak_owner=True, mapped_enabled=enabled,
        library=library, allocator=allocator, counter_source_bindings={**runtime_bindings, **bindings})
