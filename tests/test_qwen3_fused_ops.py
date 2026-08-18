from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from utils import hadamard_utils, rotation_utils, triton_qwen3_fusions  # noqa: E402


def _rmsnorm(x, weight, eps):
    fp32 = x.float()
    normalised = fp32 * torch.rsqrt(fp32.pow(2).mean(-1, keepdim=True) + eps)
    return weight * normalised.to(x.dtype)


def _rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    del position_ids
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    return (
        q * cos + rotate_half(q) * sin,
        k * cos + rotate_half(k) * sin,
    )


class _Norm(torch.nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(width))
        self.variance_epsilon = eps

    def forward(self, x):
        return _rmsnorm(x, self.weight, self.variance_epsilon)


def _hadamard(x):
    return hadamard_utils.scaled_hadamard_transform(
        x.float(), scale=1.0 / math.sqrt(x.shape[-1])
    ).to(x.dtype)


def test_qk_wrapper_cpu_fallback_defers_norm_without_state_dict_aliases():
    torch.manual_seed(20260808)
    q_norm = _Norm(4, 1e-6)
    k_norm = _Norm(4, 2e-6)
    q_weight = q_norm.weight.detach().clone()
    k_weight = k_norm.weight.detach().clone()
    wrapper = rotation_utils.QKRotationWrapper(
        _rope,
        head_dim=4,
        q_norm=q_norm,
        k_norm=k_norm,
        k_bits=16,
    )
    assert not q_norm.weight.requires_grad
    assert not k_norm.weight.requires_grad
    deferred_probe = torch.randn(1, 1, 2, 4)
    assert q_norm(deferred_probe) is deferred_probe
    assert not any("q_norm" in key or "k_norm" in key for key in wrapper.state_dict())

    q = torch.randn(2, 3, 5, 4, requires_grad=True)
    k = torch.randn(2, 1, 5, 4, requires_grad=True)
    cos = torch.randn(2, 5, 4)
    sin = torch.randn(2, 5, 4)
    expected_q, expected_k = _rope(
        _rmsnorm(q, q_weight, 1e-6),
        _rmsnorm(k, k_weight, 2e-6),
        cos,
        sin,
    )
    expected_q, expected_k = _hadamard(expected_q), _hadamard(expected_k)
    actual_q, actual_k = wrapper(q, k, cos, sin)
    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)


class Qwen3MLP(torch.nn.Module):
    def __init__(self, hidden=8, intermediate=12):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = torch.nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


def test_swiglu_installer_is_idempotent_and_cpu_exact():
    torch.manual_seed(20260808)
    model = torch.nn.Sequential(Qwen3MLP())
    x = torch.randn(2, 3, 8, requires_grad=True)
    expected = model(x)
    expected.sum().backward()
    expected_x_grad = x.grad.clone()
    expected_weight_grads = [p.grad.clone() for p in model.parameters()]

    model.zero_grad(set_to_none=True)
    x.grad = None
    assert triton_qwen3_fusions.install_qwen3_swiglu_fusion(model) == 1
    assert triton_qwen3_fusions.install_qwen3_swiglu_fusion(model) == 0
    actual = model(x)
    actual.sum().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, expected_x_grad, rtol=0, atol=0)
    for parameter, expected_grad in zip(model.parameters(), expected_weight_grads):
        torch.testing.assert_close(parameter.grad, expected_grad, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_qwen3_fusions.is_available(),
    reason="requires CUDA Triton",
)
def test_fused_qk_rmsnorm_rope_forward_backward_matches_eager():
    torch.manual_seed(20260808)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    base_q = torch.randn(3, 11, 4, 128, device=device, dtype=dtype)
    base_k = torch.randn(3, 11, 2, 128, device=device, dtype=dtype)
    q = base_q.transpose(1, 2).detach().requires_grad_(True)
    k = base_k.transpose(1, 2).detach().requires_grad_(True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    q_weight = torch.randn(128, device=device, dtype=dtype)
    k_weight = torch.randn(128, device=device, dtype=dtype)
    cos = torch.randn(3, 11, 128, device=device, dtype=dtype)
    sin = torch.randn(3, 11, 128, device=device, dtype=dtype)
    grad_q = torch.randn(q.shape, device=device, dtype=dtype)
    grad_k = torch.randn(k.shape, device=device, dtype=dtype)

    actual_q, actual_k = triton_qwen3_fusions.fused_qk_rmsnorm_rope(
        q,
        k,
        q_weight,
        k_weight,
        cos,
        sin,
        q_eps=1e-6,
        k_eps=1e-6,
    )
    expected_q, expected_k = _rope(
        _rmsnorm(q_ref, q_weight, 1e-6),
        _rmsnorm(k_ref, k_weight, 1e-6),
        cos,
        sin,
    )
    torch.testing.assert_close(actual_q, expected_q, rtol=2e-2, atol=3.125e-2)
    torch.testing.assert_close(actual_k, expected_k, rtol=2e-2, atol=3.125e-2)

    torch.autograd.backward((actual_q, actual_k), (grad_q, grad_k))
    torch.autograd.backward((expected_q, expected_k), (grad_q, grad_k))
    torch.testing.assert_close(q.grad, q_ref.grad, rtol=3e-2, atol=6.25e-2)
    torch.testing.assert_close(k.grad, k_ref.grad, rtol=3e-2, atol=6.25e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_qwen3_fusions.is_available(),
    reason="requires CUDA Triton",
)
def test_fused_swiglu_forward_backward_matches_eager():
    torch.manual_seed(20260808)
    shape = (4, 17, 9728)
    gate = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    gate_ref = gate.detach().clone().requires_grad_(True)
    up_ref = up.detach().clone().requires_grad_(True)
    grad = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    actual = triton_qwen3_fusions.fused_swiglu(gate, up)
    expected = torch.nn.functional.silu(gate_ref) * up_ref
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=3.125e-2)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(gate.grad, gate_ref.grad, rtol=3e-2, atol=6.25e-2)
    torch.testing.assert_close(up.grad, up_ref.grad, rtol=0, atol=0)
