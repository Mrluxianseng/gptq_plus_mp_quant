"""Pure control-head observation with verified installed Accelerate delegation.

No forward replacement, pool, dispatch or weight move. Offloading hooks and
arbitrary forward wrappers remain unsupported. This is validation preparation,
not a formal multi-GPU admission.
"""
import functools
import hashlib
import inspect
import marshal
import types
from pathlib import Path

import torch
from accelerate import hooks as accelerate_hooks


def code_sha(function):
    return hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()


def forward_chain(head):
    forward = head.forward
    if getattr(forward, '__func__', None) is torch.nn.Linear.forward:
        if hasattr(head, '_hf_hook') or hasattr(head, '_old_forward'):
            raise RuntimeError('Inconsistent native head hook chain')
        return dict(kind='original_linear', forward_code_sha256=code_sha(torch.nn.Linear.forward))
    expected = [value for value in accelerate_hooks.add_hook_to_module.__code__.co_consts
                if isinstance(value, types.CodeType) and value.co_name == 'new_forward']
    if (len(expected) != 1 or not isinstance(forward, functools.partial)
            or forward.args != (head,) or forward.keywords
            or not inspect.isfunction(forward.func)
            or marshal.dumps(forward.func.__code__) != marshal.dumps(expected[0])
            or forward.func.__globals__ is not accelerate_hooks.__dict__
            or getattr(getattr(head, '_old_forward', None), '__func__', None) is not torch.nn.Linear.forward):
        raise RuntimeError('Unreviewed head forward delegation')
    hook = getattr(head, '_hf_hook', None)
    hooks = hook.hooks if type(hook) is accelerate_hooks.SequentialHook else (hook,)
    if not hooks or any(type(item) is not accelerate_hooks.AlignDevicesHook or item.offload
                        or any(getattr(getattr(item, name), '__func__', None)
                               is not getattr(accelerate_hooks.AlignDevicesHook, name)
                               for name in ('pre_forward', 'post_forward')) for item in hooks):
        raise RuntimeError('Requires genuine nonoffloading AlignDevicesHook chain')
    if type(hook) is accelerate_hooks.SequentialHook and any(
            getattr(getattr(hook, name), '__func__', None)
            is not getattr(accelerate_hooks.SequentialHook, name) for name in ('pre_forward', 'post_forward')):
        raise RuntimeError('Unreviewed sequential hook delegation')
    path = Path(inspect.getfile(accelerate_hooks))
    return dict(kind='accelerate_align_devices', forward_code_sha256=code_sha(forward.func),
                original_linear_code_sha256=code_sha(torch.nn.Linear.forward),
                accelerate_source=str(path), accelerate_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                hooks=[dict(type=type(item).__name__, execution_device=str(item.execution_device),
                            offload=bool(item.offload), io_same_device=bool(item.io_same_device),
                            no_grad=bool(item.no_grad), place_submodules=bool(item.place_submodules)) for item in hooks])


class ControlHeadObserver:
    def __init__(self, head, *, expected_shape, configured_batch_size=16,
                 mark=lambda phase: None, require_cuda=True):
        if (type(head) is not torch.nn.Linear or head.training or head.bias is not None
                or head.weight.dtype != torch.bfloat16 or head.weight.is_meta
                or tuple(head.weight.shape) != tuple(expected_shape) or configured_batch_size != 16):
            raise RuntimeError('Requires original eval BF16 full Linear and configured batch16')
        if getattr(head, '_qa_control_head_observer', False) or getattr(head, '_qa_mapped_head_storage', False):
            raise RuntimeError('Head storage observation already installed')
        self.chain = forward_chain(head)
        self.head, self.shape, self.mark = head, tuple(expected_shape), mark
        self.require_cuda = require_cuda
        self.forward = head.forward
        self.hf_hook = getattr(head, '_hf_hook', None)
        self.spec = (id(head.weight), tuple(head.weight.shape), str(head.weight.dtype))
        self.records, self.batch_releases = [], []
        self.calls, self.pending, self.closed = 0, False, False
        self.forward_source_sha256 = hashlib.sha256(inspect.getsource(torch.nn.Linear.forward).encode()).hexdigest()
        self.handles = [head.register_forward_pre_hook(self.pre),
                        head.register_forward_hook(self.post, always_call=True)]
        head._qa_control_head_observer = True

    def check_chain(self):
        current = self.head.forward
        identical = (current is self.forward if isinstance(self.forward, functools.partial) else
                     getattr(current, '__func__', None) is self.forward.__func__
                     and getattr(current, '__self__', None) is self.head)
        if (not identical or getattr(self.head, '_hf_hook', None) is not self.hf_hook
                or forward_chain(self.head) != self.chain
                or (id(self.head.weight), tuple(self.head.weight.shape), str(self.head.weight.dtype)) != self.spec):
            raise RuntimeError('Control head forward/hook/Parameter changed')

    def pre(self, module, inputs):
        self.check_chain()
        if self.closed or self.pending or self.calls != len(self.batch_releases) or len(inputs) != 1:
            raise RuntimeError('Control head requires one original call per scored batch')
        value = inputs[0]
        if (not isinstance(value, torch.Tensor) or value.ndim != 3 or not 1 <= value.shape[0] <= 16
                or value.shape[2] != self.shape[1] or value.dtype != torch.bfloat16
                or value.is_meta or (self.require_cuda and value.device.type != 'cuda')):
            raise RuntimeError('Requires natural 1..16 BF16 full-head input')
        self.calls += 1
        self.pending = True
        self.input_shape, self.input_device = list(value.shape), str(value.device)
        self.mark(f'head{self.calls}_storage_start')

    def post(self, module, inputs, output):
        if not self.pending:
            return  # Preserve an original pre-hook/forward exception.
        self.pending = False
        if output is None:
            self.records.append(dict(call=self.calls, enabled=False, completed=False))
            return
        self.check_chain()
        if (not isinstance(output, torch.Tensor) or output.dtype != torch.bfloat16 or output.is_meta
                or list(output.shape) != [self.input_shape[0], self.input_shape[1], self.shape[0]]
                or (self.require_cuda and output.device.type != 'cuda')):
            raise RuntimeError('Control full-head output changed')
        self.records.append(dict(call=self.calls, enabled=False, completed=True,
            input_shape=self.input_shape, input_device=self.input_device, output_device=str(output.device),
            output_shape=list(output.shape), output_stride=list(output.stride()), dtype=str(output.dtype),
            output_payload_bytes=output.numel()*output.element_size()))
        self.mark(f'head{self.calls}_storage_end')

    def release_batch(self):
        self.check_chain()
        if (self.closed or self.pending or self.calls != len(self.batch_releases)+1
                or not self.records or not self.records[-1]['completed']):
            raise RuntimeError('Control batch observation incomplete')
        self.mark(f'head{self.calls}_storage_release_start')
        self.batch_releases.append(dict(call=self.calls, enabled=False))
        self.mark(f'head{self.calls}_storage_release_end')

    def close(self):
        if self.closed:
            raise RuntimeError('Control observer already closed')
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.head._qa_control_head_observer = False
        self.closed = True
        self.check_chain()
        if self.pending or self.calls != len(self.batch_releases) or any(not row['completed'] for row in self.records):
            raise RuntimeError('Control scored-batch observation incomplete')
        return dict(enabled=False, calls=self.calls, records=self.records, batch_releases=self.batch_releases,
                    configured_batch_size=16, actual_batch_sizes=[row['input_shape'][0] for row in self.records],
                    original_forward_sha256=self.forward_source_sha256, forward_chain=self.chain)
