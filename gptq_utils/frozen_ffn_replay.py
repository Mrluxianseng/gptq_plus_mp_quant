"""Cache the frozen prefix of Qwen3 FFN weight-refresh forwards.

For gate/up refreshes this replays the cached post-attention MLP input and
residual, then reruns the complete MLP with the live functional weight
override. For down_proj refreshes it also caches the current gate/up product.
The down-projection cache is invalidated when entering that phase because the
gate/up weights may have changed since earlier refreshes.

This changes execution only. Callers must restrict its lifetime to one frozen
attention/FFN prefix and disable it for joint updates or sliding-window paths.
"""
from contextlib import contextmanager

import torch

from gptq_utils.frozen_prefix_replay import _nbytes, _same, _sample, _snapshot


class FrozenFFNPrefix:
    def __init__(self, layer, max_bytes=128 * 1024 * 1024):
        if type(layer).__name__ != 'Qwen3DecoderLayer':
            raise ValueError('FrozenFFNPrefix supports Qwen3DecoderLayer only')
        if layer.training:
            raise ValueError('evaluation mode required')
        self.layer = layer
        self.max_bytes = int(max_bytes)
        self.entries = {}
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.verified_hits = 0
        self.mode = None

    @contextmanager
    def activate(self, batch_keys, module_name, verify=False):
        if module_name not in ('mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'):
            raise ValueError(f'unsupported FFN target {module_name!r}')
        if self.mode != module_name:
            if module_name == 'mlp.down_proj':
                # gate/up may have been modified after the preceding FFN stage.
                for item in self.entries.values():
                    old_size = item.pop('h_bytes', 0)
                    item.pop('h', None)
                    item['bytes'] -= old_size
                    self.bytes -= old_size
            self.mode = module_name

        original = self.layer.forward
        had_instance_forward = 'forward' in self.layer.__dict__
        saved_instance_forward = self.layer.__dict__.get('forward')
        cursor = 0

        def forward(hidden_states, *args, **kwargs):
            nonlocal cursor
            if kwargs.get('past_key_values') is not None or kwargs.get('use_cache', False):
                raise ValueError('KV-cache mutation is unsupported')
            keys = batch_keys[cursor % len(batch_keys)]
            cursor += 1
            batch_size = hidden_states.shape[0]
            if len(keys) != batch_size:
                raise ValueError('sample identity count does not match forward batch')
            metadata = [_sample((args, kwargs), i, batch_size) for i in range(batch_size)]
            items = [self.entries.get(key) for key in keys]

            def usable(item, i):
                return (item is not None
                        and _same(hidden_states[i:i + 1], item['x'])
                        and _same(metadata[i], item['metadata'])
                        and (module_name != 'mlp.down_proj' or item.get('h') is not None))

            if all(usable(item, i) for i, item in enumerate(items)):
                self.hits += 1
                residual = torch.cat([item['residual'] for item in items], dim=0)
                if module_name == 'mlp.down_proj':
                    h = torch.cat([item['h'] for item in items], dim=0)
                    y = residual + self.layer.mlp.down_proj(h)
                else:
                    mlp_input = torch.cat([item['mlp_input'] for item in items], dim=0)
                    y = residual + self.layer.mlp(mlp_input)
                result = (y,) if items[0]['tuple'] else y
                if verify:
                    expected = original(hidden_states, *args, **kwargs)
                    expected_y = expected[0] if isinstance(expected, tuple) else expected
                    if not torch.equal(y, expected_y):
                        error = float((y.float() - expected_y.float()).abs().max())
                        raise AssertionError(f'cached FFN forward differs, max_abs={error}')
                    self.verified_hits += 1
                return result

            self.misses += 1
            captured = {}

            def down_hook(module, inputs):
                captured['h'] = inputs[0].detach()

            def norm_pre_hook(module, inputs):
                captured['residual'] = inputs[0].detach()

            def norm_forward_hook(module, inputs, output):
                captured['mlp_input'] = output.detach()

            hooks = [
                self.layer.mlp.down_proj.register_forward_pre_hook(down_hook),
                self.layer.post_attention_layernorm.register_forward_pre_hook(norm_pre_hook),
                self.layer.post_attention_layernorm.register_forward_hook(norm_forward_hook),
            ]
            try:
                result = original(hidden_states, *args, **kwargs)
            finally:
                for hook in hooks:
                    hook.remove()
            if isinstance(result, tuple) and len(result) != 1:
                raise ValueError('cache/attention outputs are unsupported')

            for i, key in enumerate(keys):
                item = {
                    'x': hidden_states[i:i + 1].detach().clone(),
                    'residual': captured['residual'][i:i + 1].clone(),
                    'mlp_input': captured['mlp_input'][i:i + 1].clone(),
                    'metadata': _snapshot(metadata[i]),
                    'tuple': isinstance(result, tuple),
                }
                size = _nbytes(item)
                old = self.entries.pop(key, None)
                if old is not None:
                    self.bytes -= old['bytes']
                if module_name == 'mlp.down_proj':
                    item['h'] = captured['h'][i:i + 1].clone()
                    item['h_bytes'] = _nbytes(item['h'])
                    size += item['h_bytes']
                if self.bytes + size <= self.max_bytes:
                    item['bytes'] = size
                    self.entries[key] = item
                    self.bytes += size
            return result

        self.layer.forward = forward
        try:
            yield
        finally:
            if had_instance_forward:
                self.layer.forward = saved_instance_forward
            else:
                del self.layer.forward
