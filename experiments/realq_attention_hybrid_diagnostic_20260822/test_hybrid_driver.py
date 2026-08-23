from __future__ import annotations

from types import SimpleNamespace

import torch

from experiments.realq_attention_hybrid_diagnostic_20260822 import hybrid_driver


def test_forward_value_backward_source_routes_exactly() -> None:
    value = torch.tensor([1.25, -3.5])
    gradient_source = torch.tensor([7.0, 9.0], requires_grad=True)
    output = hybrid_driver._ForwardValueBackwardSource.apply(
        value,
        gradient_source,
    )
    assert torch.equal(output, value)
    (output * torch.tensor([2.0, -4.0])).sum().backward()
    assert torch.equal(gradient_source.grad, torch.tensor([2.0, -4.0]))


def test_hybrid_uses_selected_value_and_jacobian(monkeypatch) -> None:
    calls: list[tuple[str, bool]] = []

    def fa4(**kwargs):
        calls.append(("fa4", torch.is_grad_enabled()))
        return kwargs["query"] * 2, None

    def sdpa(**kwargs):
        calls.append(("sdpa", torch.is_grad_enabled()))
        return kwargs["query"] * 3, None

    monkeypatch.setattr(hybrid_driver, "_FA4_FORWARD", fa4)
    monkeypatch.setattr(hybrid_driver, "_sdpa_math_forward", sdpa)
    common = dict(
        module=SimpleNamespace(),
        key=torch.tensor([0.0]),
        value=torch.tensor([0.0]),
        attention_mask=None,
    )

    query = torch.tensor([5.0], requires_grad=True)
    counter = {"attention_calls": 0}
    output, _ = hybrid_driver._hybrid_forward(
        "fa4_forward_sdpa_backward", counter
    )(query=query, **common)
    assert torch.equal(output, torch.tensor([10.0]))
    output.sum().backward()
    assert torch.equal(query.grad, torch.tensor([3.0]))
    assert calls == [("fa4", False), ("sdpa", True)]
    assert counter == {"attention_calls": 1}

    calls.clear()
    query = torch.tensor([5.0], requires_grad=True)
    counter = {"attention_calls": 0}
    output, _ = hybrid_driver._hybrid_forward(
        "sdpa_forward_fa4_backward", counter
    )(query=query, **common)
    assert torch.equal(output, torch.tensor([15.0]))
    output.sum().backward()
    assert torch.equal(query.grad, torch.tensor([2.0]))
    assert calls == [("sdpa", False), ("fa4", True)]
    assert counter == {"attention_calls": 1}
