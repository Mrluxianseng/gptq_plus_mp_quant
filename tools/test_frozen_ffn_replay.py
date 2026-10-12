"""Numerical correctness checks for the opt-in Qwen3 FFN replay prototype."""
import unittest
import sys
from pathlib import Path

import torch
from torch import nn
from torch.func import functional_call

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gptq_utils.frozen_ffn_replay import FrozenFFNPrefix


class ToyMLP(nn.Module):
    def __init__(self, hidden=8, intermediate=12):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


def _forward(self, hidden_states, attention_mask=None):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(hidden_states)
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    return (residual + self.mlp(hidden_states),)


Qwen3DecoderLayer = type('Qwen3DecoderLayer', (nn.Module,), {'forward': _forward})


class FrozenFFNReplayTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.layer = Qwen3DecoderLayer()
        self.layer.input_layernorm = nn.LayerNorm(8)
        self.layer.self_attn = nn.Linear(8, 8, bias=False)
        self.layer.post_attention_layernorm = nn.LayerNorm(8)
        self.layer.mlp = ToyMLP()
        self.layer.eval()
        self.x = torch.randn(2, 5, 8)
        self.mask = torch.zeros(2, 1, 5, 5)
        self.cache = FrozenFFNPrefix(self.layer, max_bytes=1 << 20)
        self.keys = [(100, 101)]

    def _call(self, key, param, mask=None):
        return functional_call(
            self.layer, {key: param}, (self.x,), {'attention_mask': self.mask if mask is None else mask}
        )[0]

    def _assert_cached_output_and_grad(self, module_name, parameter_name):
        key = parameter_name
        original = dict(self.layer.named_parameters())[key]
        candidate = original.detach().clone().requires_grad_()
        with self.cache.activate(self.keys, module_name):
            # First call populates the cache; second differentiates through the
            # cached path while preserving the functional target weight.
            self._call(key, candidate)
            cached = self._call(key, candidate)
            cached_grad = torch.autograd.grad(cached.square().mean(), candidate)[0]
        reference = original.detach().clone().requires_grad_()
        expected = self._call(key, reference)
        expected_grad = torch.autograd.grad(expected.square().mean(), reference)[0]
        torch.testing.assert_close(cached, expected, rtol=0, atol=0)
        torch.testing.assert_close(cached_grad, expected_grad, rtol=0, atol=0)
        self.assertEqual(self.cache.verified_hits, 0)

    def test_gate_up_and_down_replay_preserve_forward_and_target_gradient(self):
        self._assert_cached_output_and_grad('mlp.up_proj', 'mlp.up_proj.weight')
        self._assert_cached_output_and_grad('mlp.gate_proj', 'mlp.gate_proj.weight')

        # Gate/up have changed before down_proj starts, so the down-input `h`
        # captured during their earlier refreshes must be discarded.
        with torch.no_grad():
            self.layer.mlp.gate_proj.weight.add_(0.01)
            self.layer.mlp.up_proj.weight.sub_(0.02)
        misses_before = self.cache.misses
        self._assert_cached_output_and_grad('mlp.down_proj', 'mlp.down_proj.weight')
        self.assertGreater(self.cache.misses, misses_before)
        self.assertLessEqual(self.cache.bytes, self.cache.max_bytes)

    def test_verify_checks_cached_gate_and_down_paths(self):
        with self.cache.activate(self.keys, 'mlp.gate_proj', verify=True):
            self._call('mlp.gate_proj.weight', self.layer.mlp.gate_proj.weight)
            self._call('mlp.gate_proj.weight', self.layer.mlp.gate_proj.weight)
        self.assertEqual(self.cache.verified_hits, 1)
        with self.cache.activate(self.keys, 'mlp.down_proj', verify=True):
            self._call('mlp.down_proj.weight', self.layer.mlp.down_proj.weight)
            self._call('mlp.down_proj.weight', self.layer.mlp.down_proj.weight)
        self.assertEqual(self.cache.verified_hits, 2)


if __name__ == '__main__':
    unittest.main()
