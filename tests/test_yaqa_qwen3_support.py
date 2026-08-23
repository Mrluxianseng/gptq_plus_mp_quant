import torch


def _tiny_qwen3():
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
    return Qwen3ForCausalLM(config).eval()


@torch.no_grad()
def test_yaqa_aware_runtime_keeps_qwen3_headwise_norms():
    model = _tiny_qwen3()
    attention = model.model.layers[0].self_attn
    q_norm_id = id(attention.q_norm)
    k_norm_id = id(attention.k_norm)
    norm_calls = {"q": 0, "k": 0}
    handles = [
        attention.q_norm.register_forward_hook(
            lambda *_: norm_calls.__setitem__(
                "q", norm_calls["q"] + 1
            )
        ),
        attention.k_norm.register_forward_hook(
            lambda *_: norm_calls.__setitem__(
                "k", norm_calls["k"] + 1
            )
        ),
    ]

    from experiments.yaqa_compare.akv_aware import (
        install_akv_quantization,
    )

    summary = install_akv_quantization(
        model,
        a_bits=4,
        k_bits=4,
        v_bits=4,
        groupsize=-1,
        symmetric=True,
        clip_ratio=0.9,
        mode="aware",
    )
    logits = model(
        torch.tensor([[1, 3, 5, 7]]), use_cache=False
    ).logits
    for handle in handles:
        handle.remove()

    assert torch.isfinite(logits).all()
    assert id(attention.q_norm) == q_norm_id
    assert id(attention.k_norm) == k_norm_id
    assert norm_calls == {"q": 1, "k": 1}
    assert summary.activation_input_sites == 7
    assert summary.value_output_sites == 1
    assert summary.post_rope_k_sites == 1
    assert summary.query_quantized is False
    assert summary.extra_qk_hadamard is False


def test_custom_linear_can_take_mode_from_context():
    from YAQA_wclip.hessian_llama.custom_linear_B import CustomLinear
    from YAQA_wclip.hessian_llama.hessian_context import (
        use_hessian_mode,
    )

    layer = CustomLinear(
        0,
        False,
        "/tmp/unused_yaqa_context_test",
        False,
        False,
        4,
        3,
    )
    with torch.no_grad():
        layer.weight.copy_(
            torch.arange(12, dtype=torch.float32).reshape(3, 4)
        )
    value = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    explicit = layer(value, (0, True, True))
    with use_hessian_mode((0, True, True)):
        contextual = layer(value)
    torch.testing.assert_close(contextual, explicit)


def test_hessian_mode_process_fallback_survives_empty_context():
    from contextvars import Context

    from YAQA_wclip.hessian_llama.hessian_context import (
        current_hessian_mode,
        use_hessian_mode,
    )

    expected = (0, True, False)
    with use_hessian_mode(expected):
        assert Context().run(current_hessian_mode) == expected


def test_qwen3_mlp_dimensions_use_exact_shared_hadamard_factors():
    from lib.utils.matmul_had import (
        get_hadK,
        matmul_hadU,
        matmul_hadUt,
    )

    generator = torch.Generator().manual_seed(20260727)
    for dimension, expected_factor in ((9728, 76), (25600, 100)):
        hadamard, factor = get_hadK(dimension)
        transposed, transposed_factor = get_hadK(
            dimension, transpose=True
        )
        assert factor == expected_factor
        assert transposed_factor == expected_factor
        assert hadamard.shape == (expected_factor, expected_factor)
        torch.testing.assert_close(transposed, hadamard.T)
        torch.testing.assert_close(
            hadamard @ hadamard.T,
            expected_factor * torch.eye(expected_factor),
        )

        value = torch.randn(
            2, dimension, dtype=torch.float64, generator=generator
        )
        restored = matmul_hadUt(matmul_hadU(value))
        # YAQA's historical normalization constructs sqrt(n) in FP32 even
        # when the Hessian transform is FP64.
        torch.testing.assert_close(restored, value, rtol=1e-6, atol=2e-7)
