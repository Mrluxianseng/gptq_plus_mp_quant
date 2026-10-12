import torch

from realq.refresh import adam_transfer_diagnostic as diagnostic


def test_normalized_row_prior_seeds_only_when_profile_gate_passes(monkeypatch):
    module = "mlp.down_proj"
    profile = torch.log(torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64))
    diagnostic._DONOR[(module, 6, 0)] = {
        "profile": profile,
        "vhat_mean": 2.0,
        "grad2_mean": 2.0,
        "adam_step": 1,
        "refresh_index": 0,
        "active_input_columns": 3,
    }
    monkeypatch.setenv("REALQ_ADAM_V_TRANSFER_MODE", "normalized_row_prior")
    monkeypatch.setenv("REALQ_ADAM_V_TRANSFER_TARGET_LAYER", "7")
    monkeypatch.setenv("REALQ_ADAM_V_TRANSFER_TARGET_MODULE", module)
    monkeypatch.setenv("REALQ_ADAM_V_TRANSFER_MIN_CORR", "0.9")

    row_scale = profile.exp().float()
    grad = row_scale[:, None].expand(4, 3).clone()
    v = torch.zeros_like(grad)
    info = diagnostic.initialize_v_from_previous_row_profile(
        layer=7,
        module=module,
        refresh_index=0,
        grad=grad,
        exp_avg_sq=v,
        beta2=0.999,
    )

    assert info is not None
    assert info["correlation"] > 0.99
    assert info["donor_age"] == 1
    assert info["target_v_step"] == 2
    assert torch.all(v > 0)
    assert torch.allclose(v[:, 0], v[:, 1])
    assert torch.isclose(v.mean(), (1.0 - 0.999) * grad.square().mean())


def test_normalized_row_prior_leaves_state_unchanged_when_gate_fails(
    monkeypatch, caplog
):
    module = "self_attn.o_proj"
    diagnostic._DONOR[(module, 2, 0)] = {
        "profile": torch.log(torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64)),
        "vhat_mean": 1.0,
        "grad2_mean": 1.0,
        "adam_step": 1,
        "refresh_index": 0,
        "active_input_columns": 3,
    }
    monkeypatch.setenv("REALQ_ADAM_V_TRANSFER_MODE", "normalized_row_prior")
    monkeypatch.setenv("REALQ_ADAM_V_TRANSFER_TARGET_LAYER", "3")
    monkeypatch.setenv("REALQ_ADAM_V_TRANSFER_TARGET_MODULE", module)
    monkeypatch.setenv("REALQ_ADAM_V_TRANSFER_MIN_CORR", "0.99")

    grad = torch.ones((4, 3), dtype=torch.float32)
    v = torch.zeros_like(grad)
    before = v.clone()
    info = diagnostic.initialize_v_from_previous_row_profile(
        layer=3,
        module=module,
        refresh_index=0,
        grad=grad,
        exp_avg_sq=v,
        beta2=0.999,
    )

    assert info is None
    assert torch.equal(v, before)
    assert "reason=correlation_below_gate" in caplog.text


def test_zero_shrinkage_reproduces_baseline_first_step_vhat(monkeypatch):
    module = "mlp.down_proj"
    profile = torch.log(torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64))
    diagnostic._DONOR[(module, 6, 0)] = {
        "profile": profile,
        "vhat_mean": 2.0,
        "grad2_mean": 2.0,
        "adam_step": 1,
        "refresh_index": 0,
        "active_input_columns": 3,
    }
    monkeypatch.setenv("REALQ_ADAM_V_TRANSFER_MODE", "normalized_row_prior")
    monkeypatch.setenv("REALQ_ADAM_V_TRANSFER_TARGET_LAYER", "7")
    monkeypatch.setenv("REALQ_ADAM_V_TRANSFER_TARGET_MODULE", module)
    monkeypatch.setenv("REALQ_ADAM_V_TRANSFER_MIN_CORR", "-1")
    monkeypatch.setenv("REALQ_ADAM_V_TRANSFER_SHRINKAGE", "0")

    grad = torch.tensor(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
        dtype=torch.float32,
    )
    v = torch.zeros_like(grad)
    info = diagnostic.initialize_v_from_previous_row_profile(
        layer=7,
        module=module,
        refresh_index=0,
        grad=grad,
        exp_avg_sq=v,
        beta2=0.999,
    )
    assert info is not None and info["shrinkage"] == 0.0

    # Reproduce the production EMA transition with the virtual age=2. When
    # the initial v prior equals this refresh's g^2, corrected v remains g^2.
    v.mul_(0.999).addcmul_(grad, grad, value=1.0 - 0.999)
    vhat = v / (1.0 - 0.999**2)
    assert torch.allclose(vhat, grad.square(), rtol=2e-4, atol=1e-8)


def test_moment_drift_observer_is_read_only_and_matches_first_step(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("REALQ_ADAM_MOMENT_DRIFT_DIAG_DIR", str(tmp_path))
    grad = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.2, 0.4, 0.6, 0.8]],
        dtype=torch.float32,
    )
    beta2 = 0.999
    v = (1.0 - beta2) * grad.square()
    v_before = v.clone()
    record = diagnostic.record_moment_drift(
        layer=0,
        module="mlp.down_proj",
        refresh_index=0,
        adam_step=1,
        grad=grad,
        exp_avg_sq=v,
        beta2=beta2,
    )

    assert record is not None
    assert abs(record["mean_vhat_to_g2_ratio"] - 1.0) < 1e-5
    assert record["row_log_profile_pearson"] > 0.999
    assert record["tile_log_profile_pearson"] > 0.999
    assert torch.equal(v, v_before)
    assert (tmp_path / "moment_drift.jsonl").is_file()
