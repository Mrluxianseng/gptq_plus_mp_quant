from types import SimpleNamespace

import torch

from realq.refresh.kl_loss import _legacy_kl_adam_update


def _ctx(step=3, v_step=3):
    return SimpleNamespace(
        exp_avg=torch.tensor([[0.4, -0.2, 0.7]], dtype=torch.float32),
        exp_avg_sq=torch.tensor([[0.3, 0.5, 0.8]], dtype=torch.float32),
        adam_step=step,
        adam_v_step=v_step,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        layer_lr=0.05,
    )


def test_legacy_kl_keep_matches_original_adam_expression(monkeypatch):
    monkeypatch.delenv("REALQ_ADAM_HISTORY_POLICY", raising=False)
    ctx = _ctx()
    grad = torch.tensor([[0.2, -0.4]], dtype=torch.float32)
    trailing_start = 1

    old_m = ctx.exp_avg[:, trailing_start:].clone()
    old_v = ctx.exp_avg_sq[:, trailing_start:].clone()
    expected_m = 0.9 * old_m + 0.1 * grad
    expected_v = 0.999 * old_v + 0.001 * grad.square()
    bc1 = 1.0 - 0.9**ctx.adam_step
    bc2 = 1.0 - 0.999**ctx.adam_step
    expected = (ctx.layer_lr / bc1) * (
        expected_m / (expected_v.sqrt() / bc2**0.5 + ctx.eps)
    )

    actual = _legacy_kl_adam_update(ctx, grad, trailing_start)

    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-8)
    assert ctx.adam_step == 3


def test_legacy_kl_m_restart_uses_local_m_age_and_persistent_v_age(monkeypatch):
    monkeypatch.setenv(
        "REALQ_ADAM_HISTORY_POLICY", "reset_m_and_m_age_each_refresh"
    )
    ctx = _ctx(step=6, v_step=7)
    grad = torch.tensor([[0.2, -0.4]], dtype=torch.float32)
    trailing_start = 1
    old_v = ctx.exp_avg_sq[:, trailing_start:].clone()
    expected_v = 0.999 * old_v + 0.001 * grad.square()
    expected_m = 0.1 * grad
    vhat = expected_v / (1.0 - 0.999**7)
    expected = ctx.layer_lr * grad / (vhat.sqrt() + ctx.eps)

    actual = _legacy_kl_adam_update(ctx, grad, trailing_start)

    assert ctx.adam_step == 1
    assert ctx.adam_v_step == 7
    assert torch.equal(ctx.exp_avg[:, :trailing_start], torch.tensor([[0.4]]))
    assert torch.equal(ctx.exp_avg[:, trailing_start:], expected_m)
    assert torch.allclose(ctx.exp_avg_sq[:, trailing_start:], expected_v)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-7)


def test_legacy_kl_respects_early_only_history_scope(monkeypatch):
    monkeypatch.setenv(
        "REALQ_ADAM_HISTORY_POLICY", "reset_m_and_m_age_each_refresh"
    )
    monkeypatch.setenv("REALQ_ADAM_HISTORY_LAYERS", "0-26")
    ctx = _ctx(step=6, v_step=7)
    ctx.trace_layer = 27
    grad = torch.tensor([[0.2, -0.4]], dtype=torch.float32)
    trailing_start = 1
    old_m = ctx.exp_avg[:, trailing_start:].clone()
    old_v = ctx.exp_avg_sq[:, trailing_start:].clone()
    expected_m = 0.9 * old_m + 0.1 * grad
    expected_v = 0.999 * old_v + 0.001 * grad.square()
    bc1 = 1.0 - 0.9**6
    bc2 = 1.0 - 0.999**6
    expected = (ctx.layer_lr / bc1) * (
        expected_m / (expected_v.sqrt() / bc2**0.5 + ctx.eps)
    )

    actual = _legacy_kl_adam_update(ctx, grad, trailing_start)

    assert ctx.adam_step == 6
    assert ctx.adam_v_step == 7
    assert torch.allclose(ctx.exp_avg[:, trailing_start:], expected_m)
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-8)


if __name__ == "__main__":
    class _Environment:
        @staticmethod
        def setenv(name, value):
            import os
            os.environ[name] = value

        @staticmethod
        def delenv(name, raising=False):
            import os
            os.environ.pop(name, None)

    test_legacy_kl_keep_matches_original_adam_expression(_Environment())
    test_legacy_kl_m_restart_uses_local_m_age_and_persistent_v_age(
        _Environment()
    )
    test_legacy_kl_respects_early_only_history_scope(_Environment())
    print("KL Adam history scope checks passed")
