from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from realq.alignment import RefreshTraceWriter, load_refresh_trace
from realq.refresh.block_gd import (
    RefreshContext,
    _SharedSampleScheduler,
    _aggregate_refresh_sums,
    make_grad_refresh_fn,
)
from realq.refresh.kl_loss import make_kl_refresh_fn
from utils import dist_utils


def test_packed_distributed_refresh_uses_global_sample_mean(monkeypatch):
    """Simulate a second rank without starting a process group.

    Local rank contributes two samples with loss sum 6; the synthetic remote
    rank contributes three samples with loss sum 15. The traced mean must be
    (6 + 15) / (2 + 3), never either rank's local mean.
    """

    grad_sum = torch.tensor([[1.0, 2.0]])
    loss_sums = torch.tensor([6.0, 4.0, 8.0])
    remote_pack = torch.tensor([3.0, 4.0, 3.0, 15.0, 9.0, 21.0])

    monkeypatch.setattr(dist_utils, "get_world_size", lambda: 2)

    def fake_allreduce_sum_(packed):
        assert packed.numel() == remote_pack.numel()
        packed.add_(remote_pack)
        return packed

    monkeypatch.setattr(dist_utils, "allreduce_sum_", fake_allreduce_sum_)
    global_count, global_loss_sums = _aggregate_refresh_sums(
        grad_sum,
        partial_count=2,
        partial_loss_sums=loss_sums,
    )

    assert global_count == 5
    assert torch.equal(grad_sum, torch.tensor([[4.0, 6.0]]))
    assert torch.equal(global_loss_sums, torch.tensor([21.0, 13.0, 29.0]))
    assert global_loss_sums[0].item() / global_count == pytest.approx(4.2)


def test_trace_disabled_preserves_original_grad_count_pack(monkeypatch):
    grad_sum = torch.tensor([[1.0, 2.0]])
    monkeypatch.setattr(dist_utils, "get_world_size", lambda: 2)

    def fake_allreduce_sum_(packed):
        # Historical layout is exactly [flattened gradient | sample count].
        assert packed.numel() == grad_sum.numel() + 1
        packed.add_(torch.tensor([3.0, 4.0, 2.0]))
        return packed

    monkeypatch.setattr(dist_utils, "allreduce_sum_", fake_allreduce_sum_)
    global_count, global_loss_sums = _aggregate_refresh_sums(
        grad_sum,
        partial_count=2,
    )
    assert global_count == 4
    assert global_loss_sums is None
    assert torch.equal(grad_sum, torch.tensor([[4.0, 6.0]]))


class _ToyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 2, bias=False)

    def forward(self, hidden_states, **_kwargs):
        return (self.proj(hidden_states),)


class _ToyNextLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)

    def forward(self, hidden_states, **_kwargs):
        return (self.proj(hidden_states),)


def test_fisher_refresh_writes_global_step_identity_and_loss(tmp_path):
    torch.manual_seed(0)
    layer = _ToyLayer()
    state = SimpleNamespace(
        inps=torch.arange(32, dtype=torch.float32).reshape(4, 2, 4) / 32.0,
        attention_mask=None,
        position_ids=None,
        position_embeddings=None,
    )
    trace_path = tmp_path / "realq.jsonl"
    writer = RefreshTraceWriter(
        str(trace_path),
        implementation="realq",
        run_id="toy",
        config={"seed": 0},
    )
    ctx = RefreshContext(
        module=layer.proj,
        layer_lr=1e-3,
        grad_clip=1.0,
        backward_bsz=1,
        scheduler=_SharedSampleScheduler(4, 2, seed=0),
        trace_writer=writer,
        trace_layer=1,
        trace_module="self_attn.q_proj",
        blocksize=2,
    )
    refresh = make_grad_refresh_fn(
        layer=layer,
        module=layer.proj,
        layer_state=state,
        fp_out_for_this_layer=torch.zeros(4, 2, 2),
        fisher=torch.eye(2),
        ctx=ctx,
    )

    update = refresh(layer.proj.weight.detach().float().clone(), 2)
    writer.close()

    assert update is not None
    assert update.shape == (2, 2)
    _, steps = load_refresh_trace(str(trace_path))
    step = next(iter(steps.values()))
    assert step.identity == (1, "self_attn.q_proj", 0, 0, 2, 1)
    assert step.sample_indices == (0, 1)
    assert step.loss > 0
    assert step.loss_current == pytest.approx(step.loss)
    assert step.loss_next is None
    assert step.slide_alpha is None


def test_block_gd_update_is_bias_corrected_adam_on_trailing_columns():
    torch.manual_seed(31)
    layer = _ToyLayer()
    state = SimpleNamespace(
        inps=torch.tensor(
            [
                [[1.0, -2.0, 0.5, 3.0]],
                [[-1.0, 0.25, 2.0, -0.5]],
            ]
        ),
        attention_mask=None,
        position_ids=None,
        position_embeddings=None,
    )
    learning_rate = 3e-4
    ctx = RefreshContext(
        module=layer.proj,
        layer_lr=learning_rate,
        grad_clip=1.0,
        backward_bsz=2,
        scheduler=_SharedSampleScheduler(2, 2, seed=7),
    )
    refresh = make_grad_refresh_fn(
        layer=layer,
        module=layer.proj,
        layer_state=state,
        fp_out_for_this_layer=torch.zeros(2, 1, 2),
        fisher=torch.eye(2),
        ctx=ctx,
    )
    weight = layer.proj.weight.detach().float().clone()
    update = refresh(weight, trailing_col_start=2)

    # For 0.5 * mean ||W x||², dL/dW = Y^T X / token_count.
    x = state.inps.reshape(-1, 4)
    y = x @ weight.T
    full_grad = y.T @ x / x.shape[0]
    grad = full_grad[:, 2:].clamp(-ctx.grad_clip, ctx.grad_clip)
    exp_avg = (1.0 - ctx.beta1) * grad
    exp_avg_sq = (1.0 - ctx.beta2) * grad.square()
    expected = (
        learning_rate
        / (1.0 - ctx.beta1)
        * exp_avg
        / (
            exp_avg_sq.sqrt()
            / (1.0 - ctx.beta2) ** 0.5
            + ctx.eps
        )
    )

    assert update.shape == (2, 2)
    assert ctx.adam_step == 1
    torch.testing.assert_close(
        ctx.exp_avg[:, 2:], exp_avg, rtol=1e-6, atol=1e-8
    )
    torch.testing.assert_close(
        ctx.exp_avg_sq[:, 2:], exp_avg_sq, rtol=1e-6, atol=1e-8
    )
    torch.testing.assert_close(update, expected, rtol=1e-6, atol=1e-8)
    # The already-quantized prefix remains locked.
    assert torch.count_nonzero(ctx.exp_avg[:, :2]) == 0
    assert torch.count_nonzero(ctx.exp_avg_sq[:, :2]) == 0

    first_exp_avg = ctx.exp_avg.clone()
    first_exp_avg_sq = ctx.exp_avg_sq.clone()
    next_weight = weight.clone()
    next_weight[:, 2:].sub_(update)
    update_2 = refresh(next_weight, trailing_col_start=3)

    y_2 = x @ next_weight.T
    full_grad_2 = y_2.T @ x / x.shape[0]
    grad_2 = full_grad_2[:, 3:].clamp(
        -ctx.grad_clip, ctx.grad_clip
    )
    exp_avg_2 = (
        ctx.beta1 * first_exp_avg[:, 3:]
        + (1.0 - ctx.beta1) * grad_2
    )
    exp_avg_sq_2 = (
        ctx.beta2 * first_exp_avg_sq[:, 3:]
        + (1.0 - ctx.beta2) * grad_2.square()
    )
    expected_2 = (
        learning_rate
        / (1.0 - ctx.beta1**2)
        * exp_avg_2
        / (
            exp_avg_sq_2.sqrt()
            / (1.0 - ctx.beta2**2) ** 0.5
            + ctx.eps
        )
    )
    assert ctx.adam_step == 2
    torch.testing.assert_close(update_2, expected_2, rtol=1e-6, atol=1e-8)
    # Column 2 has become part of the locked prefix and its moments no longer
    # decay or absorb gradients; column 3 carries state into Adam step 2.
    assert torch.equal(ctx.exp_avg[:, 2], first_exp_avg[:, 2])
    assert torch.equal(ctx.exp_avg_sq[:, 2], first_exp_avg_sq[:, 2])
    torch.testing.assert_close(
        ctx.exp_avg[:, 3:], exp_avg_2, rtol=1e-6, atol=1e-8
    )
    torch.testing.assert_close(
        ctx.exp_avg_sq[:, 3:], exp_avg_sq_2, rtol=1e-6, atol=1e-8
    )


def test_local_backward_chunk_clip_skips_global_percentile_prepass(
    monkeypatch,
):
    torch.manual_seed(41)
    layer = _ToyLayer()
    state = SimpleNamespace(
        inps=torch.randn(2, 2, 4),
        attention_mask=None,
        position_ids=None,
        position_embeddings=None,
    )
    ctx = RefreshContext(
        module=layer.proj,
        layer_lr=1e-3,
        grad_clip=1.0,
        backward_bsz=1,
        scheduler=_SharedSampleScheduler(2, 2, seed=0),
    )

    def fail_global_percentile(*_args, **_kwargs):
        raise AssertionError("local scope must not run a global P95 prepass")

    monkeypatch.setattr(
        "realq.refresh.block_gd.global_percentile",
        fail_global_percentile,
    )
    refresh = make_grad_refresh_fn(
        layer=layer,
        module=layer.proj,
        layer_state=state,
        fp_out_for_this_layer=torch.zeros(2, 2, 2),
        fisher=torch.eye(2),
        ctx=ctx,
        a_loss_ratio=0.95,
        a_loss_clip_scope="local_backward_chunk",
    )
    assert refresh(
        layer.proj.weight.detach().float().clone(), 2
    ) is not None


def test_global_refresh_clip_runs_one_shared_percentile_prepass(
    monkeypatch,
):
    import realq.refresh.block_gd as block_gd

    torch.manual_seed(43)
    layer = _ToyLayer()
    state = SimpleNamespace(
        inps=torch.randn(2, 2, 4),
        attention_mask=None,
        position_ids=None,
        position_embeddings=None,
    )
    ctx = RefreshContext(
        module=layer.proj,
        layer_lr=1e-3,
        grad_clip=1.0,
        backward_bsz=1,
        scheduler=_SharedSampleScheduler(2, 2, seed=0),
    )
    original = block_gd.global_percentile
    calls = []

    def record_global_percentile(values, ratio):
        calls.append((values.numel(), ratio))
        return original(values, ratio)

    monkeypatch.setattr(
        block_gd,
        "global_percentile",
        record_global_percentile,
    )
    refresh = make_grad_refresh_fn(
        layer=layer,
        module=layer.proj,
        layer_state=state,
        fp_out_for_this_layer=torch.zeros(2, 2, 2),
        fisher=torch.eye(2),
        ctx=ctx,
        a_loss_ratio=0.95,
        a_loss_clip_scope="global_refresh",
    )
    assert refresh(
        layer.proj.weight.detach().float().clone(), 2
    ) is not None
    assert calls == [(8, 0.95)]


def test_fisher_refresh_records_explicit_first_and_blended_slide_steps(tmp_path):
    torch.manual_seed(1)
    layer = _ToyLayer()
    next_layer = _ToyNextLayer()
    state = SimpleNamespace(
        inps=torch.arange(32, dtype=torch.float32).reshape(4, 2, 4) / 32.0,
        attention_mask=None,
        position_ids=None,
        position_embeddings=None,
    )
    trace_path = tmp_path / "slide.jsonl"
    writer = RefreshTraceWriter(
        str(trace_path),
        implementation="realq",
        run_id="slide",
        config={"seed": 0},
    )
    ctx = RefreshContext(
        module=layer.proj,
        layer_lr=1e-3,
        grad_clip=1.0,
        backward_bsz=1,
        scheduler=_SharedSampleScheduler(4, 2, seed=0),
        trace_writer=writer,
        trace_layer=0,
        trace_module="mlp.up_proj",
        blocksize=1,
    )
    alpha_values = iter((1.0, 0.5))
    refresh = make_grad_refresh_fn(
        layer=layer,
        module=layer.proj,
        layer_state=state,
        fp_out_for_this_layer=torch.zeros(4, 2, 2),
        fisher=torch.eye(2),
        ctx=ctx,
        next_layer=next_layer,
        next_fp_out=torch.zeros(4, 2, 2),
        next_fisher=torch.eye(2),
        slide_alpha_fn=lambda: next(alpha_values),
    )
    weight = layer.proj.weight.detach().float().clone()
    assert refresh(weight, 1) is not None
    assert refresh(weight, 2) is not None
    writer.close()

    _, by_identity = load_refresh_trace(str(trace_path))
    steps = [by_identity[key] for key in sorted(by_identity)]
    assert [step.identity for step in steps] == [
        (0, "mlp.up_proj", 0, 0, 1, 1),
        (0, "mlp.up_proj", 1, 1, 2, 2),
    ]
    assert steps[0].slide_alpha == 1.0
    assert steps[0].loss_current == pytest.approx(steps[0].loss)
    assert steps[0].loss_next is None
    assert steps[1].slide_alpha == 0.5
    assert steps[1].loss_next is not None
    assert steps[1].loss == pytest.approx(
        0.5 * steps[1].loss_current + 0.5 * steps[1].loss_next,
        rel=1e-6,
    )
    assert steps[0].sample_indices == (0, 1)
    assert steps[1].sample_indices == (2, 3)


class _ToyAnalyzer:
    def __init__(self) -> None:
        self.norm = nn.Identity()
        self.head = nn.Linear(2, 5, bias=False)

    def get_layernorm_before_head(self):
        return self.norm

    def get_lm_head(self):
        return self.head


def test_final_kl_refresh_records_the_loss_used_by_adam(tmp_path):
    torch.manual_seed(2)
    layer = _ToyLayer()
    state = SimpleNamespace(
        inps=torch.arange(32, dtype=torch.float32).reshape(4, 2, 4) / 32.0,
        attention_mask=None,
        position_ids=None,
        position_embeddings=None,
    )
    trace_path = tmp_path / "kl.jsonl"
    writer = RefreshTraceWriter(
        str(trace_path),
        implementation="realq",
        run_id="kl",
        config={"seed": 0},
    )
    ctx = RefreshContext(
        module=layer.proj,
        layer_lr=1e-3,
        grad_clip=1.0,
        backward_bsz=2,
        scheduler=_SharedSampleScheduler(4, 2, seed=0),
        trace_writer=writer,
        trace_layer=2,
        trace_module="self_attn.o_proj",
        blocksize=2,
    )
    refresh = make_kl_refresh_fn(
        layer=layer,
        module=layer.proj,
        layer_state=state,
        fp_out_for_this_layer=torch.zeros(4, 2, 2),
        analyzer=_ToyAnalyzer(),
        kl_topk=-1,
        ctx=ctx,
    )
    assert refresh(layer.proj.weight.detach().float().clone(), 2) is not None
    writer.close()

    _, steps = load_refresh_trace(str(trace_path))
    step = next(iter(steps.values()))
    assert step.identity == (2, "self_attn.o_proj", 0, 0, 2, 1)
    assert step.loss > 0
    assert step.loss_current == pytest.approx(step.loss)
    assert step.loss_next is None
    assert step.slide_alpha is None
