import hashlib
import pytest
import torch
from tools.qa_tensor_fingerprint_stream import tensor_sha256,update_tensor_bytes


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16,torch.float16,torch.int64,torch.bool])
def test_scalar_empty_and_noncontiguous_c_order_bytes(dtype):
    base=torch.arange(2*7*11).reshape(2,7,11).to(dtype)
    for value in (base,base.transpose(0,2),base[:,:0,:],base[0,0,0]):
        expected=hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        assert tensor_sha256(value,max_bytes=32)==expected
    assert not torch.cuda.is_initialized()


def test_block_bound_and_cumulative_header_digest():
    class Recorder:
        def __init__(self): self.digest=hashlib.sha256();self.lengths=[]
        def update(self,block): self.lengths.append(len(block));self.digest.update(block)
    value=torch.arange(3*13*19).reshape(3,13,19).bfloat16().transpose(1,2)
    recorder=Recorder();recorder.update(b'original-header')
    update_tensor_bytes(recorder,value,max_bytes=64)
    expected=hashlib.sha256(b'original-header'+value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    assert recorder.digest.hexdigest()==expected and max(recorder.lengths)<=64
    with pytest.raises(ValueError): tensor_sha256(value,max_bytes=1)
