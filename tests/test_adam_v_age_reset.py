import torch

from realq.refresh.block_gd import (
    _WeightRefreshState,
    _adam_history_policy_for_layer,
    _adam_update_selected,
    _match_update_l2_norm,
    _norm_matched_gradient_update,
)


def _state() -> _WeightRefreshState:
    zero = torch.zeros((1, 1), dtype=torch.float32)
    return _WeightRefreshState(
        name="probe",
        module=None,
        parameter_name="weight",
        shape=(1, 1),
        master=zero.clone(),
        exp_avg=zero.clone(),
        exp_avg_sq=zero.clone(),
    )


def test_v_only_restart_resets_its_age_but_preserves_momentum(monkeypatch):
    state = _state()
    monkeypatch.delenv("REALQ_ADAM_HISTORY_POLICY", raising=False)
    _adam_update_selected(
        state, torch.tensor([[2.0]]), None, lr=1.0, grad_clip=0.0
    )
    m_before = state.exp_avg.clone()

    monkeypatch.setenv(
        "REALQ_ADAM_HISTORY_POLICY", "reset_v_and_v_age_each_refresh"
    )
    update = _adam_update_selected(
        state, torch.tensor([[4.0]]), None, lr=1.0, grad_clip=0.0
    )

    assert state.step == 2  # m still uses its real two-refresh age
    assert state.v_step == 1  # v is corrected as a fresh one-step estimate
    assert torch.equal(state.exp_avg, 0.9 * m_before + 0.1 * 4.0)
    vhat = state.exp_avg_sq / (1.0 - 0.999**state.v_step)
    assert torch.allclose(vhat, torch.tensor([[16.0]]), rtol=1e-5)
    expected_mhat = state.exp_avg / (1.0 - 0.9**state.step)
    expected_update = expected_mhat / (vhat.sqrt() + 1e-8)
    assert torch.allclose(update, expected_update, rtol=1e-5, atol=1e-7)


def test_legacy_v_reset_keeps_old_age_for_audit(monkeypatch):
    state = _state()
    monkeypatch.delenv("REALQ_ADAM_HISTORY_POLICY", raising=False)
    _adam_update_selected(
        state, torch.tensor([[2.0]]), None, lr=1.0, grad_clip=0.0
    )

    monkeypatch.setenv("REALQ_ADAM_HISTORY_POLICY", "reset_v_each_refresh")
    _adam_update_selected(
        state, torch.tensor([[4.0]]), None, lr=1.0, grad_clip=0.0
    )

    assert state.step == 2
    assert state.v_step == 2
    vhat = state.exp_avg_sq / (1.0 - 0.999**state.v_step)
    assert not torch.allclose(vhat, torch.tensor([[16.0]]), rtol=1e-3)


def test_m_only_restart_resets_its_age_and_preserves_v_history(monkeypatch):
    state = _state()
    monkeypatch.delenv("REALQ_ADAM_HISTORY_POLICY", raising=False)
    _adam_update_selected(
        state, torch.tensor([[2.0]]), None, lr=1.0, grad_clip=0.0
    )
    old_v = state.exp_avg_sq.clone()
    old_v_step = state.v_step

    monkeypatch.setenv(
        "REALQ_ADAM_HISTORY_POLICY", "reset_m_and_m_age_each_refresh"
    )
    update = _adam_update_selected(
        state, torch.tensor([[4.0]]), None, lr=1.0, grad_clip=0.0
    )

    assert state.step == 1
    assert state.v_step == old_v_step + 1
    expected_v = old_v * 0.999 + torch.tensor([[16.0]]) * 0.001
    assert torch.allclose(state.exp_avg_sq, expected_v, rtol=1e-6)
    assert torch.allclose(state.exp_avg, torch.tensor([[0.4]]), rtol=1e-6)
    mhat = state.exp_avg / (1.0 - 0.9**state.step)
    vhat = state.exp_avg_sq / (1.0 - 0.999**state.v_step)
    expected_update = mhat / (vhat.sqrt() + 1e-8)
    assert torch.allclose(update, expected_update, rtol=1e-5)


def test_history_layer_scope_selects_only_requested_range(monkeypatch):
    monkeypatch.setenv(
        "REALQ_ADAM_HISTORY_POLICY", "reset_m_and_m_age_each_refresh"
    )
    monkeypatch.setenv("REALQ_ADAM_HISTORY_LAYERS", "0-26")
    assert (
        _adam_history_policy_for_layer(26)
        == "reset_m_and_m_age_each_refresh"
    )
    assert _adam_history_policy_for_layer(27) == "keep"


def test_scoped_m_restart_keeps_momentum_outside_layer_range(monkeypatch):
    state = _state()
    monkeypatch.delenv("REALQ_ADAM_HISTORY_POLICY", raising=False)
    monkeypatch.delenv("REALQ_ADAM_HISTORY_LAYERS", raising=False)
    _adam_update_selected(
        state, torch.tensor([[2.0]]), None, lr=1.0, grad_clip=0.0,
        layer_idx=27,
    )
    old_m = state.exp_avg.clone()
    old_v = state.exp_avg_sq.clone()

    monkeypatch.setenv(
        "REALQ_ADAM_HISTORY_POLICY", "reset_m_and_m_age_each_refresh"
    )
    monkeypatch.setenv("REALQ_ADAM_HISTORY_LAYERS", "0-26")
    _adam_update_selected(
        state, torch.tensor([[4.0]]), None, lr=1.0, grad_clip=0.0,
        layer_idx=27,
    )

    assert state.step == 2
    assert state.v_step == 2
    assert torch.equal(state.exp_avg, 0.9 * old_m + 0.1 * 4.0)
    assert torch.equal(
        state.exp_avg_sq, 0.999 * old_v + 0.001 * 16.0
    )


def test_radam_uses_momentum_only_before_variance_is_rectifiable(monkeypatch):
    state = _state()
    monkeypatch.setenv("REALQ_ADAM_HISTORY_POLICY", "radam")
    update = _adam_update_selected(
        state, torch.tensor([[2.0]]), None, lr=0.5, grad_clip=0.0
    )

    # At t=1, rho_t is below the rectification threshold; RAdam falls back to
    # the bias-corrected first moment without coordinatewise v normalization.
    assert state.step == state.v_step == 1
    assert torch.allclose(update, torch.tensor([[1.0]]), rtol=1e-6)


def test_adabelief_uses_current_gradient_minus_updated_momentum(monkeypatch):
    state = _state()
    monkeypatch.setenv("REALQ_ADAM_HISTORY_POLICY", "adabelief")
    grad = torch.tensor([[2.0]])
    update = _adam_update_selected(
        state, grad, None, lr=0.5, grad_clip=0.0
    )

    m = 0.1 * grad
    residual = grad - m
    expected_vhat = residual.square()
    actual_vhat = state.exp_avg_sq / (1.0 - 0.999**state.v_step)
    assert torch.allclose(actual_vhat, expected_vhat, rtol=1e-5)
    expected_update = 0.5 * grad / (residual.abs() + 1e-8)
    assert torch.allclose(update, expected_update, rtol=1e-5)


def test_adopt_normalizes_current_gradient_with_previous_second_moment(monkeypatch):
    state = _state()
    monkeypatch.setenv("REALQ_ADAM_HISTORY_POLICY", "adopt")
    first = _adam_update_selected(
        state, torch.tensor([[2.0]]), None, lr=0.5, grad_clip=0.0
    )
    # The first refresh reuses g_1 to initialize v_0; this matches Adam's
    # sign-like first update without an extra backward call.
    assert torch.allclose(first, torch.tensor([[0.5]]), rtol=1e-6)
    assert torch.allclose(state.exp_avg, torch.tensor([[1.0]]), rtol=1e-6)
    assert torch.allclose(state.exp_avg_sq, torch.tensor([[4.0]]), rtol=1e-6)

    second = _adam_update_selected(
        state, torch.tensor([[4.0]]), None, lr=0.5, grad_clip=0.0
    )
    # ADOPT uses the previous v (=4) to normalize g (=4), then updates v.
    expected_m = 0.9 * 1.0 + 0.1 * (4.0 / 2.0)
    expected_v = 0.999 * 4.0 + 0.001 * 16.0
    assert torch.allclose(second, torch.tensor([[0.5 * expected_m]]), rtol=1e-6)
    assert torch.allclose(state.exp_avg, torch.tensor([[expected_m]]), rtol=1e-6)
    assert torch.allclose(state.exp_avg_sq, torch.tensor([[expected_v]]), rtol=1e-6)


def test_norm_matched_gradient_preserves_adam_l2_norm_and_uses_gradient_direction():
    adam_update = torch.tensor([[3.0, 4.0]])
    gradient = torch.tensor([[6.0, 8.0]])
    update = _norm_matched_gradient_update(adam_update, gradient)
    assert torch.allclose(torch.linalg.vector_norm(update), torch.tensor(5.0))
    assert torch.allclose(update / torch.linalg.vector_norm(update), gradient / 10.0)
    assert torch.equal(
        _norm_matched_gradient_update(adam_update, torch.zeros_like(gradient)),
        torch.zeros_like(gradient),
    )


def test_reset_m_direction_and_adam_direction_can_be_norm_matched():
    adam_update = torch.tensor([[3.0, 4.0]])
    reset_m_update = torch.tensor([[1.0, 0.0]])
    reset_direction = _match_update_l2_norm(reset_m_update, adam_update)
    adam_direction = _match_update_l2_norm(adam_update, reset_m_update)

    assert torch.allclose(torch.linalg.vector_norm(reset_direction), torch.tensor(5.0))
    assert torch.equal(reset_direction / 5.0, torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(torch.linalg.vector_norm(adam_direction), torch.tensor(1.0))
    assert torch.allclose(adam_direction, torch.tensor([[0.6, 0.8]]))
    assert torch.equal(
        _match_update_l2_norm(torch.zeros_like(reset_m_update), adam_update),
        torch.zeros_like(reset_m_update),
    )


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

    test_v_only_restart_resets_its_age_but_preserves_momentum(_Environment())
    test_legacy_v_reset_keeps_old_age_for_audit(_Environment())
    test_m_only_restart_resets_its_age_and_preserves_v_history(_Environment())
    test_radam_uses_momentum_only_before_variance_is_rectifiable(_Environment())
    test_adabelief_uses_current_gradient_minus_updated_momentum(_Environment())
    test_adopt_normalizes_current_gradient_with_previous_second_moment(
        _Environment()
    )
    test_norm_matched_gradient_preserves_adam_l2_norm_and_uses_gradient_direction()
    test_reset_m_direction_and_adam_direction_can_be_norm_matched()
    test_history_layer_scope_selects_only_requested_range(_Environment())
    test_scoped_m_restart_keeps_momentum_outside_layer_range(_Environment())
    print("Adam refresh policy arithmetic checks passed")
