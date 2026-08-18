from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from realq.attention import (
    configure_attention_backend,
    flash_attention_4_forward,
)
from realq.config import Config


def test_attention_backend_config_validation() -> None:
    assert Config(attention_backend="sdpa").attention_backend == "sdpa"
    assert (
        Config(attention_backend="flash_attention_4").attention_backend
        == "flash_attention_4"
    )
    with pytest.raises(ValueError, match="attention_backend"):
        Config(attention_backend="unknown")


def test_fa4_rejects_mask_before_kernel_import() -> None:
    q = torch.empty(1, 2, 4, 8)
    with pytest.raises(ValueError, match="attention mask"):
        flash_attention_4_forward(
            SimpleNamespace(is_causal=True),
            q,
            q,
            q,
            torch.ones(1, 1, 4, 4),
        )


def test_fa4_rejects_cpu_tensors_before_kernel_import() -> None:
    q = torch.empty(1, 2, 4, 8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA"):
        flash_attention_4_forward(
            SimpleNamespace(is_causal=True), q, q, q, None
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fa4_qwen3_forward_backward_matches_sdpa() -> None:
    pytest.importorskip("flash_attn.cute")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        use_cache=False,
    )
    # Transformers retains the supplied config object.  Use independent
    # copies so configuring the candidate cannot also switch the reference
    # away from SDPA.
    reference = Qwen3ForCausalLM(copy.deepcopy(config)).to(
        device="cuda", dtype=torch.bfloat16
    ).eval()
    candidate = Qwen3ForCausalLM(copy.deepcopy(config)).to(
        device="cuda", dtype=torch.bfloat16
    ).eval()
    candidate.load_state_dict(reference.state_dict())
    configure_attention_backend(reference, "sdpa")
    configure_attention_backend(candidate, "flash_attention_4")

    input_ids = torch.randint(0, config.vocab_size, (2, 64), device="cuda")
    with torch.no_grad():
        expected = reference(input_ids).logits.float()
    actual = candidate(input_ids).logits.float()

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    actual.square().mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in candidate.parameters()
    )
