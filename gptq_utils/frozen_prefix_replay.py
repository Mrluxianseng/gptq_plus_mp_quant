"""Replay a Qwen3 decoder's final projection while its prefix is frozen.

Experimental execution optimization, not a changed quantization objective.
The caller must discard the cache whenever any prefix weight/input changes.
Cache hits additionally check the actual input tensor. Only the standard
Qwen3 residual + down_proj tail is supported; other architectures fail closed.
"""
from contextlib import contextmanager
import torch


def _same(a,b):
    if torch.is_tensor(a) and torch.is_tensor(b):
        return a.shape==b.shape and a.dtype==b.dtype and a.device==b.device and torch.equal(a,b)
    if isinstance(a,(tuple,list)) and isinstance(b,type(a)):
        return len(a)==len(b) and all(_same(x,y) for x,y in zip(a,b))
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()==b.keys() and all(_same(a[k],b[k]) for k in a)
    if torch.is_tensor(a) or torch.is_tensor(b):
        return False
    return a==b


def _snapshot(value):
    if torch.is_tensor(value): return value.detach().clone()
    if isinstance(value,tuple): return tuple(_snapshot(x) for x in value)
    if isinstance(value,list): return [_snapshot(x) for x in value]
    if isinstance(value,dict): return {k:_snapshot(v) for k,v in value.items()}
    if value is None or isinstance(value,(bool,int,float,str)): return value
    raise ValueError(f'unsupported cache metadata {type(value)}')


def _nbytes(value):
    if torch.is_tensor(value): return value.numel()*value.element_size()
    if isinstance(value,(tuple,list)): return sum(_nbytes(v) for v in value)
    if isinstance(value,dict): return sum(_nbytes(v) for v in value.values())
    return 0


def _sample(value,index,batch_size):
    """Slice batch-dependent metadata, preserving shared/one-dimensional data."""
    if torch.is_tensor(value):
        return value[index:index+1] if value.ndim>=2 and value.shape[0]==batch_size else value
    if isinstance(value,tuple): return tuple(_sample(v,index,batch_size) for v in value)
    if isinstance(value,list): return [_sample(v,index,batch_size) for v in value]
    if isinstance(value,dict): return {k:_sample(v,index,batch_size) for k,v in value.items()}
    return value


class FrozenDownPrefix:
    def __init__(self, layer, max_bytes=128*1024*1024):
        if type(layer).__name__ != 'Qwen3DecoderLayer':
            raise ValueError('FrozenDownPrefix supports Qwen3DecoderLayer only')
        if layer.training:
            raise ValueError('evaluation mode required')
        self.layer = layer
        self.max_bytes = max_bytes
        self.entries = {}
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.verified_hits = 0

    @contextmanager
    def activate(self, batch_keys, verify=False):
        """Batch keys encode sample ids; changing masks/positions requires reset.

        `verify` compares every hit with the complete original forward before
        returning the cached result, and is for correctness runs only.
        """
        original = self.layer.forward
        had_instance_forward = 'forward' in self.layer.__dict__
        saved_instance_forward = self.layer.__dict__.get('forward')
        cursor = 0

        def forward(hidden_states, *args, **kwargs):
            nonlocal cursor
            if kwargs.get('past_key_values') is not None or kwargs.get('use_cache',False):
                raise ValueError('KV-cache mutation is unsupported')
            keys = batch_keys[cursor % len(batch_keys)]
            cursor += 1
            batch_size=hidden_states.shape[0]
            if len(keys)!=batch_size:
                raise ValueError('sample identity count does not match forward batch')
            metadata=[_sample((args,kwargs),i,batch_size) for i in range(batch_size)]
            items=[self.entries.get(key) for key in keys]
            if all(item is not None and _same(hidden_states[i:i+1],item['x'])
                   and _same(metadata[i],item['metadata']) for i,item in enumerate(items)):
                self.hits += 1
                residual=torch.cat([item['residual'] for item in items],dim=0)
                h=torch.cat([item['h'] for item in items],dim=0)
                y = residual + self.layer.mlp.down_proj(h)
                result = (y,) if items[0]['tuple'] else y
                if verify:
                    expected = original(hidden_states,*args,**kwargs)
                    expected_y = expected[0] if isinstance(expected,tuple) else expected
                    if not torch.equal(y,expected_y):
                        error = float((y.float()-expected_y.float()).abs().max())
                        raise AssertionError(f'cached forward differs, max_abs={error}')
                    self.verified_hits += 1
                return result
            self.misses += 1
            captured = {}
            def down_hook(module, inputs):
                captured['h'] = inputs[0].detach()
            def norm_hook(module, inputs):
                captured['residual'] = inputs[0].detach()
            hooks = [self.layer.mlp.down_proj.register_forward_pre_hook(down_hook),
                     self.layer.post_attention_layernorm.register_forward_pre_hook(norm_hook)]
            try:
                result = original(hidden_states,*args,**kwargs)
            finally:
                for hook in hooks:
                    hook.remove()
            if isinstance(result,tuple) and len(result)!=1:
                raise ValueError('cache/attention outputs are unsupported')
            for i,key in enumerate(keys):
                item = {'x':hidden_states[i:i+1].detach().clone(),
                        'h':captured['h'][i:i+1].clone(),
                        'residual':captured['residual'][i:i+1].clone(),
                        'metadata':_snapshot(metadata[i]),
                        'tuple':isinstance(result,tuple)}
                size = _nbytes(item)
                old = self.entries.pop(key,None)
                if old is not None:
                    self.bytes -= old['bytes']
                if self.bytes+size<=self.max_bytes:
                    item['bytes']=size
                    self.entries[key]=item
                    self.bytes+=size
            return result

        self.layer.forward = forward
        try:
            yield
        finally:
            if had_instance_forward:
                self.layer.forward=saved_instance_forward
            else:
                del self.layer.forward
