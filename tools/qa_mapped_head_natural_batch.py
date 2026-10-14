"""Prepared natural-tail adapter; configured QA batch remains 16.

This does not enable formal QA. Original evaluators may produce their final
batch with fewer requests. No padding, request truncation or batch setting is
changed, and the full original Linear receives exactly that input shape.
"""
import torch
from tools.qa_mapped_head_batch_lifetime import BatchMappedHeadStorage,build_batch_method


class NaturalBatchMappedHeadStorage(BatchMappedHeadStorage):
    def __init__(self,head,*,configured_batch_size=16,**kwargs):
        if configured_batch_size!=16: raise RuntimeError('Configured QA batch must remain 16')
        self.configured_batch_size=configured_batch_size
        super().__init__(head,**kwargs)

    def pre(self,module,inputs):
        if self.context is not None or len(inputs)!=1:
            raise RuntimeError('Nested/nonoriginal head call refused')
        value=inputs[0]
        if (not isinstance(value,torch.Tensor) or value.ndim!=3 or not 1<=value.shape[0]<=16
                or value.shape[2]!=self.shape[1] or value.dtype!=torch.bfloat16
                or (self.require_cuda and value.device.type!='cuda')):
            raise RuntimeError('Requires natural 1..16 BF16 batch under configured batch16')
        if self.enabled and self.pool is None: self.pool=self.pool_factory()
        self.calls+=1;self.input_shape=list(value.shape)
        self.mark(f'head{self.calls}_storage_start')
        if self.enabled:
            self.context=self.cuda.use_mem_pool(self.pool)
            self.context.__enter__()

    def close(self):
        result=super().close()
        result['configured_batch_size']=self.configured_batch_size
        result['actual_batch_sizes']=[r['input_shape'][0] for r in self.records if r['completed']]
        return result
