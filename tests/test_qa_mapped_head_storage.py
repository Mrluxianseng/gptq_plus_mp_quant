import weakref
import pytest
import torch
from tools.qa_mapped_head_storage_prototype import MappedHeadStorage


class Library:
    def __init__(self): self.live=0;self.allocs=0;self.frees=0
    def qa_live_bytes(self): return self.live
    def qa_last_error(self): return 0
    def qa_allocation_count(self): return self.allocs
    def qa_release_count(self): return self.frees


class Pool:
    def __init__(self,library,payload):
        self.library=library;self.active=payload;self.reserved=2*1024*1024
        library.live=self.reserved;library.allocs+=1
    def snapshot(self,include_traces=False):
        return [dict(active_size=self.active,total_size=self.reserved,address=1,device=0,
                     blocks=[dict(state='active_allocated',size=self.active)])]
    def __del__(self): self.library.live=0;self.library.frees+=1


class CUDA:
    def __init__(self,missing=False): self.pool=None;self.events=[];self.missing=missing
    def use_mem_pool(self,pool):
        self.pool=weakref.ref(pool);cuda=self
        class Context:
            def __enter__(self): cuda.events.append('enter')
            def __exit__(self,*args): cuda.events.append('exit')
        return Context()
    def synchronize(self): pass
    def memory_allocated(self): return 0 if self.missing else self.pool().active
    def memory_reserved(self): return self.pool().reserved
    def memory_snapshot(self,include_traces=False): return self.pool().snapshot()


def fixture(missing=False):
    torch.manual_seed(1)
    head=torch.nn.Linear(64,256,bias=False).bfloat16().eval()
    inputs=torch.randn(16,2,64,dtype=torch.bfloat16)
    library=Library();cuda=CUDA(missing)
    ticket=MappedHeadStorage(head,enabled=True,expected_shape=(256,64),library=library,
        pool=Pool(library,16*2*256*2),cuda=cuda,require_cuda=False)
    return head,inputs,library,cuda,ticket


def test_original_full_linear_bytes_and_context_restoration_with_fake_pool():
    head,inputs,library,cuda,ticket=fixture()
    before=head.weight.detach().clone()
    with torch.no_grad():
        expected=torch.nn.functional.linear(inputs,head.weight)
        result=head(inputs)
    assert torch.equal(result,expected) and torch.equal(head.weight,before)
    assert ticket.records[0]['accounting_complete']
    assert cuda.events==['enter','exit']
    with pytest.raises(RuntimeError,match='still active'): ticket.close()
    # Simulate the fake allocator observing release after the last consumer.
    del result;ticket.pool.active=0
    report=ticket.close()
    assert report['final_host_live']==0 and report['allocations']==report['releases']==1
    assert not head._forward_hooks and not head._forward_pre_hooks
    assert not torch.cuda.is_initialized()


def test_untracked_private_accounting_refused():
    head,inputs,library,cuda,ticket=fixture(missing=True)
    with pytest.raises(RuntimeError,match='accounting incomplete'):
        with torch.no_grad(): head(inputs)
    assert ticket.records[-1]['accounting_complete'] is False
    assert cuda.events==['enter','exit']
    ticket.pool.active=0;ticket.close()


def test_original_forward_failure_exits_pool_and_is_not_masked(monkeypatch):
    head,inputs,library,cuda,ticket=fixture()
    def fail(*args,**kwargs): raise ValueError('original Linear error')
    monkeypatch.setattr(torch.nn.functional,'linear',fail)
    with pytest.raises(ValueError,match='original Linear error'): head(inputs)
    assert cuda.events==['enter','exit'] and ticket.records[-1]['completed'] is False
    ticket.pool.active=0;ticket.close()


def test_wrong_batch_and_forward_replacement_refused():
    head,inputs,library,cuda,ticket=fixture()
    with pytest.raises(RuntimeError,match='batch16'): head(inputs[:8])
    assert cuda.events==[]
    ticket.pool.active=0;ticket.close()
    head.forward=lambda value:value
    with pytest.raises(RuntimeError,match='replace'):
        MappedHeadStorage(head,enabled=False,expected_shape=(256,64),library=library)


def test_large_default_allocator_cannot_hide_missing_private_segments():
    head,inputs,library,cuda,ticket=fixture()
    cuda.memory_allocated=lambda:64*1024*1024
    cuda.memory_reserved=lambda:128*1024*1024
    cuda.memory_snapshot=lambda **kwargs:[dict(device=0,address=2,total_size=128*1024*1024,
        blocks=[dict(state='active_allocated',size=64*1024*1024)])]
    with pytest.raises(RuntimeError,match='accounting incomplete'):
        with torch.no_grad(): head(inputs)
    assert ticket.records[-1]['private_segments_in_native'] is False
    ticket.pool.active=0;ticket.close()
