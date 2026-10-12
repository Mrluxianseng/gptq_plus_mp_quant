"""CPU exact-forward/gradient and cache-input-invalidation tests."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.func import functional_call
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer,Qwen3RotaryEmbedding
from transformers import Qwen3Config
from gptq_utils.frozen_prefix_replay import FrozenDownPrefix

torch.set_num_threads(2)
torch.manual_seed(31)
config=Qwen3Config(hidden_size=64,intermediate_size=128,num_attention_heads=4,
    num_key_value_heads=2,head_dim=16,num_hidden_layers=1,attention_dropout=0.0)
config._attn_implementation='eager'
layer=Qwen3DecoderLayer(config,0).eval()
for p in layer.parameters(): p.requires_grad_(False)
rotary=Qwen3RotaryEmbedding(config)
x=torch.randn(2,16,64)
pos=torch.arange(16).unsqueeze(0)
emb=rotary(x,pos)
kwargs={'position_embeddings':emb,'position_ids':pos}
cache=FrozenDownPrefix(layer)
for step in range(4):
    if step==3:
        x=x+0.1  # same key with changed data must miss
    w=(layer.mlp.down_proj.weight.detach()+torch.randn_like(layer.mlp.down_proj.weight)*0.02).requires_grad_()
    expected=functional_call(layer,{'mlp.down_proj.weight':w},(x,),kwargs)
    expected_y=expected[0] if isinstance(expected,tuple) else expected
    ge=torch.autograd.grad(expected_y.square().mean(),w)[0]
    with cache.activate([(0,1)],verify=True):
        actual=functional_call(layer,{'mlp.down_proj.weight':w},(x,),kwargs)
        actual_y=actual[0] if isinstance(actual,tuple) else actual
        ga=torch.autograd.grad(actual_y.square().mean(),w)[0]
    torch.testing.assert_close(actual_y,expected_y,rtol=0,atol=0)
    torch.testing.assert_close(ga,ge,rtol=0,atol=0)
assert cache.hits==2 and cache.misses==2
assert 'forward' not in layer.__dict__
print('PREFIX_CPU_EXACT_OK',{'hits':cache.hits,'misses':cache.misses,'bytes':cache.bytes})

# Independent metadata mutation: same sample ids/input, different positions.
old_misses=cache.misses
kwargs['position_embeddings']=rotary(x,pos+2)
with cache.activate([(0,1)],verify=True):
    functional_call(layer,{'mlp.down_proj.weight':w},(x,),kwargs)
assert cache.misses==old_misses+1

# Hard cache bound: an oversized item is computed normally but never retained.
tiny=FrozenDownPrefix(layer,max_bytes=1)
for _ in range(2):
    with tiny.activate([(0,1)]):
        layer(x,**kwargs)
assert tiny.bytes==0 and not tiny.entries and tiny.misses==2

# BF16 mirrors production's forward precision; never replace it with a
# mathematically equivalent FP32 sufficient-statistic calculation.
layer=layer.to(torch.bfloat16)
x=x.to(torch.bfloat16)
kwargs['position_embeddings']=rotary(x,pos)
bf=FrozenDownPrefix(layer)
for step in range(2):
    w=(layer.mlp.down_proj.weight.detach()+0.02*torch.randn_like(layer.mlp.down_proj.weight)).requires_grad_()
    y=functional_call(layer,{'mlp.down_proj.weight':w},(x,),kwargs)
    y=y[0] if isinstance(y,tuple) else y
    g=torch.autograd.grad(y.float().square().mean(),w)[0]
    with bf.activate([(0,1)],verify=True):
        z=functional_call(layer,{'mlp.down_proj.weight':w},(x,),kwargs)
        z=z[0] if isinstance(z,tuple) else z
        gz=torch.autograd.grad(z.float().square().mean(),w)[0]
    torch.testing.assert_close(y,z,rtol=0,atol=0)
    torch.testing.assert_close(g,gz,rtol=0,atol=0)
print('PREFIX_METADATA_CAP_BF16_OK')

# Same samples in a different order must hit without multiplying cache size.
prior_bytes=bf.bytes
prior_hits=bf.hits
rev=x.flip(0)
with bf.activate([(1,0)],verify=True):
    cached=functional_call(layer,{'mlp.down_proj.weight':w},(rev,),kwargs)
assert bf.hits==prior_hits+1 and bf.bytes==prior_bytes
print('PREFIX_SAMPLE_REORDER_OK')
