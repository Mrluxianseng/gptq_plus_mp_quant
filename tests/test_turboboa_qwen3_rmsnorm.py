import pytest
import torch
from torch import nn


class _RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.7, 1.3, width))
        self.variance_epsilon = eps

    def forward(self, value):
        inv_r = torch.rsqrt(
            value.float().square().mean(dim=-1, keepdim=True)
            + self.variance_epsilon
        )
        return self.weight * (value.float() * inv_r)


def test_mean_rmsnorm_jacobian_matches_explicit_autograd():
    from turboBOA.utils.hessian_utils import (
        MeanRMSNormJacobianCollector,
    )

    torch.manual_seed(3)
    heads, tokens, width = 3, 5, 4
    norm = _RMSNorm(width, eps=1e-5)
    values = torch.randn(tokens, heads, width)
    collector = MeanRMSNormJacobianCollector(
        norm, n_heads=heads, head_dim=width
    )
    collector.cache_fp = False
    collector.compute_batch(
        None, None, values.reshape(1, tokens, heads * width)
    )

    explicit = []
    for head in range(heads):
        per_token = []
        for token in range(tokens):
            sample = values[token, head].detach().requires_grad_(True)
            per_token.append(
                torch.autograd.functional.jacobian(norm, sample)
            )
        explicit.append(torch.stack(per_token).mean(dim=0))
    explicit = torch.stack(explicit)
    torch.testing.assert_close(
        collector.mean_jacobian(),
        explicit,
        rtol=2e-5,
        atol=2e-5,
    )


def test_mean_rmsnorm_jacobian_ignores_teacher_stream():
    from turboBOA.utils.hessian_utils import (
        MeanRMSNormJacobianCollector,
    )

    norm = _RMSNorm(4, eps=1e-6)
    collector = MeanRMSNormJacobianCollector(
        norm, n_heads=2, head_dim=4
    )
    collector.compute_batch(None, None, torch.randn(1, 7, 8))
    assert collector.n_data == 0
    with pytest.raises(RuntimeError, match="no quantized-stream"):
        collector.mean_jacobian()


def _tiny_qwen3():
    # Instantiate before importing TurboBoA model_utils: the upstream module
    # intentionally disables torch initializer functions while loading real
    # checkpoints.
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    return model


@torch.no_grad()
def test_qwen3_post_rope_observer_is_forward_exact():
    model = _tiny_qwen3()
    input_ids = torch.tensor([[1, 5, 9, 2, 7]])
    before = model(input_ids, use_cache=False).logits

    from turboBOA.utils.model_utils import install_post_rope_observers

    install_post_rope_observers(model)
    attention = model.model.layers[0].self_attn
    seen = {}

    def record(name):
        def hook(_module, _inputs, output):
            seen.setdefault(name, tuple(output.shape))

        return hook

    handles = [
        attention.rot_out_Q.register_forward_hook(record("q")),
        attention.rot_out_K.register_forward_hook(record("k")),
    ]
    after = model(input_ids, use_cache=False).logits
    for handle in handles:
        handle.remove()

    torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert seen == {"q": (1, 4, 5, 8), "k": (1, 2, 5, 8)}
    install_post_rope_observers(model)


@torch.no_grad()
def test_qwen3_observer_preserves_existing_qk_aware_wrapper():
    model = _tiny_qwen3()
    attention = model.model.layers[0].self_attn
    from utils.rotation_utils import (
        add_qk_rotation_wrapper_after_function_call_in_forward,
    )

    wrapper = add_qk_rotation_wrapper_after_function_call_in_forward(
        attention,
        "apply_rotary_pos_emb",
        head_dim=8,
        k_bits=4,
        k_groupsize=-1,
        k_sym=True,
        k_clip_ratio=0.9,
        k_quant_enabled=True,
    )
    input_ids = torch.tensor([[2, 4, 6, 8]])
    before = model(input_ids, use_cache=False).logits

    from turboBOA.utils.model_utils import install_post_rope_observers

    install_post_rope_observers(model)
    after = model(input_ids, use_cache=False).logits
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert (
        attention.apply_rotary_pos_emb_qk_rotation_wrapper is wrapper
    )


@torch.no_grad()
def test_block_v_captures_weights_from_stock_qwen_attention():
    model = _tiny_qwen3()
    from turboBOA.quantize import (
        _register_attention_weight_capture,
        _take_attention_weights,
    )

    handles = []
    captured = _register_attention_weight_capture(
        model.model.layers[0], handles
    )
    model(torch.tensor([[1, 2, 3]]), use_cache=False)
    weights = _take_attention_weights(captured)
    for handle in handles:
        handle.remove()
    assert tuple(weights.shape) == (1, 4, 3, 3)
