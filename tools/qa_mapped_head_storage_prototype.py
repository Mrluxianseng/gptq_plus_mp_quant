"""Whole original Linear output placement; preparation/probe only, not formal QA.

No forward replacement or GEMM slicing. A private native caching pool delegates
its segments to the previously probed mapped pinned-host allocator.
"""
import ctypes
import gc
import inspect
import hashlib
import torch


def prepare_runtime(binary):
    if torch.cuda.is_initialized():
        raise RuntimeError('Mapped runtime must be prepared before CUDA initialization')
    library=ctypes.CDLL(str(binary))
    for name in ('qa_live_bytes','qa_peak_bytes','qa_allocation_count','qa_release_count'):
        getattr(library,name).restype=ctypes.c_size_t
    error=library.qa_enable_map_host()
    if error!=0: raise RuntimeError('Map-host context flag rejected: '+str(error))
    attributes={name:library.qa_device_attribute(number,0) for name,number in
                (('can_map_host_memory',19),('unified_addressing',41))}
    if any(value!=1 for value in attributes.values()):
        raise RuntimeError('Mapped-host capability missing')
    return library,attributes


class MappedHeadStorage:
    def __init__(self, head, *, enabled, expected_shape, library, pool=None,
                 cuda=torch.cuda, mark=lambda phase:None, require_cuda=True):
        if type(head) is not torch.nn.Linear or head.training or head.weight.dtype!=torch.bfloat16:
            raise RuntimeError('Requires the original eval BF16 Linear head')
        if getattr(head.forward,'__func__',None) is not torch.nn.Linear.forward:
            raise RuntimeError('Do not replace the original Linear forward')
        if tuple(head.weight.shape)!=tuple(expected_shape) or head.bias is not None:
            raise RuntimeError('Full original LM-head geometry/bias contract changed')
        if getattr(head,'_qa_mapped_head_storage',False): raise RuntimeError('Mapped head already installed')
        if enabled and pool is None: raise RuntimeError('Mapped candidate requires the admitted private pool')
        self.head=head;self.enabled=bool(enabled);self.shape=tuple(expected_shape)
        self.library=library;self.pool=pool;self.cuda=cuda;self.mark=mark;self.require_cuda=require_cuda
        self.context=None;self.records=[];self.calls=0;self.closed=False
        self.spec=(id(head.weight),tuple(head.weight.shape),str(head.weight.dtype))
        self.forward_source_sha256=hashlib.sha256(inspect.getsource(type(head).forward).encode()).hexdigest()
        self.handles=[head.register_forward_pre_hook(self.pre),
            head.register_forward_hook(self.post,always_call=True)]
        head._qa_mapped_head_storage=True

    def pre(self,module,inputs):
        if self.context is not None or len(inputs)!=1:
            raise RuntimeError('Nested/nonoriginal head call refused')
        value=inputs[0]
        if (not isinstance(value,torch.Tensor) or value.ndim!=3 or value.shape[0]!=16
                or value.shape[2]!=self.shape[1] or value.dtype!=torch.bfloat16
                or (self.require_cuda and value.device.type!='cuda')):
            raise RuntimeError('Requires unchanged batch16/BF16/full-head input')
        self.calls+=1
        self.input_shape=list(value.shape)
        self.mark(f'head{self.calls}_storage_start')
        if self.enabled:
            self.context=self.cuda.use_mem_pool(self.pool)
            self.context.__enter__()

    def post(self,module,inputs,output):
        if self.context is not None:
            context=self.context;self.context=None
            context.__exit__(None,None,None)
        if output is None:
            self.records.append(dict(call=self.calls,enabled=self.enabled,completed=False))
            return # Original forward exception remains the primary error.
        if (not isinstance(output,torch.Tensor) or output.dtype!=torch.bfloat16
                or list(output.shape)!=[self.input_shape[0],self.input_shape[1],self.shape[0]]):
            raise RuntimeError('Original full-head output geometry changed')
        row=dict(call=self.calls,enabled=self.enabled,completed=True,input_shape=self.input_shape,
            output_shape=list(output.shape),output_stride=list(output.stride()),dtype=str(output.dtype),
            output_payload_bytes=output.numel()*output.element_size())
        if self.enabled:
            self.cuda.synchronize()
            segments=self.pool.snapshot(include_traces=False)
            active=sum(segment['active_size'] for segment in segments)
            reserved=sum(segment['total_size'] for segment in segments)
            device=0 if output.device.index is None else output.device.index
            all_segments=[segment for segment in self.cuda.memory_snapshot(include_traces=False)
                          if segment['device']==device]
            all_allocated=sum(block['size'] for segment in all_segments for block in segment['blocks']
                              if block['state']=='active_allocated')
            all_reserved=sum(segment['total_size'] for segment in all_segments)
            private_in_native={segment['address'] for segment in segments}.issubset(
                {segment['address'] for segment in all_segments})
            row.update(private_active=active,private_reserved=reserved,
                native_allocated=int(self.cuda.memory_allocated()),native_reserved=int(self.cuda.memory_reserved()),
                mapped_host_live=int(self.library.qa_live_bytes()),snapshot_native_allocated=all_allocated,
                snapshot_native_reserved=all_reserved,private_segments_in_native=private_in_native)
            row['accounting_complete']=(active>=row['output_payload_bytes']
                and row['native_allocated']>=active and row['native_reserved']>=reserved
                and row['mapped_host_live']==reserved and self.library.qa_last_error()==0
                and row['native_allocated']==all_allocated and row['native_reserved']==all_reserved
                and private_in_native)
            if not row['accounting_complete']:
                self.records.append(row)
                raise RuntimeError('Private mapped output/native accounting incomplete')
        self.records.append(row)
        self.mark(f'head{self.calls}_storage_end')

    def close(self):
        if self.closed: raise RuntimeError('Mapped head already closed')
        for handle in self.handles: handle.remove()
        self.handles=[];self.head._qa_mapped_head_storage=False
        if self.context is not None:
            context=self.context;self.context=None;context.__exit__(None,None,None)
        if (id(self.head.weight),tuple(self.head.weight.shape),str(self.head.weight.dtype))!=self.spec:
            raise RuntimeError('LM-head parameter identity/geometry changed')
        result=dict(calls=self.calls,records=self.records,enabled=self.enabled,
                    original_forward_sha256=self.forward_source_sha256)
        if self.enabled:
            self.cuda.synchronize()
            active=sum(segment['active_size'] for segment in self.pool.snapshot(include_traces=False))
            result['released_active']=active
            if active!=0: raise RuntimeError('Mapped output still active; preserve pool until no consumers')
            self.pool=None;gc.collect();self.cuda.synchronize()
            result.update(final_host_live=int(self.library.qa_live_bytes()),
                allocations=int(self.library.qa_allocation_count()),releases=int(self.library.qa_release_count()))
            if result['final_host_live']!=0 or result['allocations']!=result['releases'] or self.library.qa_last_error()!=0:
                raise RuntimeError('Mapped pool release incomplete')
        self.closed=True
        return result
