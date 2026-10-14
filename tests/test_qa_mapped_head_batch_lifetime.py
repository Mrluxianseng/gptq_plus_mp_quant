import ast
import inspect
import pytest
import torch
from tools.qa_mapped_head_batch_lifetime import BatchMappedHeadStorage,build_batch_source
from tools.qa_score_lifetime_prototype import build_source
from lm_eval.models.huggingface import HFLM
from test_qa_mapped_head_storage import Library,Pool,CUDA


def fixture():
    head=torch.nn.Linear(64,256,bias=False).bfloat16().eval()
    value=torch.randn(16,2,64,dtype=torch.bfloat16)
    library=Library();cuda=CUDA()
    ticket=BatchMappedHeadStorage(head,enabled=True,expected_shape=(256,64),library=library,
        cuda=cuda,require_cuda=False,pool_factory=lambda:Pool(library,16*2*256*2))
    return head,value,library,cuda,ticket


def test_actual_ast_only_adds_callback_after_last_delete():
    original=inspect.getsource(HFLM._loglikelihood_tokens)
    old,new=ast.parse(build_source(original)),ast.parse(build_batch_source(original))
    class Remove(ast.NodeTransformer):
        def visit_Expr(self,node):
            return None if ast.unparse(node)=='_qa_release_batch()' else node
    restored=Remove().visit(new)
    assert ast.dump(old,include_attributes=False)==ast.dump(restored,include_attributes=False)
    text=build_batch_source(original)
    assert text.count('_qa_release_batch()')==1
    assert 'del multi_logits\n        _qa_release_batch()' in text


def test_release_three_batches_before_next_decoder_no_byte_change():
    head,value,library,cuda,ticket=fixture()
    for call in range(1,4):
        with torch.no_grad():
            expected=torch.nn.functional.linear(value,head.weight);output=head(value)
        assert torch.equal(output,expected)
        ticket.pool.active=0;del output
        ticket.release_batch()
        assert library.live==0 and library.allocs==library.frees==call and ticket.pool is None
    report=ticket.close()
    assert report['enabled'] and report['calls']==3 and len(report['batch_releases'])==3
    assert not head._forward_hooks and not head._forward_pre_hooks and not torch.cuda.is_initialized()


def test_live_output_cannot_be_released_or_finalized():
    head,value,library,cuda,ticket=fixture()
    with torch.no_grad(): output=head(value)
    with pytest.raises(RuntimeError,match='still has a consumer'): ticket.release_batch()
    assert library.live>0 and ticket.pool is not None
    with pytest.raises(RuntimeError,match='Every scored batch'): ticket.close()
    ticket.pool.active=0;del output;ticket.release_batch();ticket.close()


def test_double_or_missing_forward_release_refused():
    head,value,library,cuda,ticket=fixture()
    with pytest.raises(RuntimeError,match='exactly one'): ticket.release_batch()
    with torch.no_grad(): output=head(value)
    ticket.pool.active=0;del output;ticket.release_batch()
    with pytest.raises(RuntimeError,match='exactly one'): ticket.release_batch()
    ticket.close()
