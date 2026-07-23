from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from gptq_utils.gptq_plus_utils import (  # noqa: E402
    GPTQPlus,
    compute_layer_lr_scale,
    compute_refresh_loss,
)
from realq.config import Config, parse_cli  # noqa: E402
from realq.precompute.hooks import (  # noqa: E402
    FisherHookManager,
    LOSS_GRAD_SCALE,
)
from realq.refresh.block_gd import layer_lr_for_schedule  # noqa: E402
from realq.refresh.fisher_loss import fisher_mse_loss  # noqa: E402
from realq.refresh.kl_loss import kl_topk_loss  # noqa: E402
from realq.quant.hessian import (  # noqa: E402
    cholesky_inverse_batched_with_damp,
)
from utils.loss_utils import tokenwise_kl_from_logits  # noqa: E402
from utils.saliency_utils import global_percentile  # noqa: E402


def test_old_and_new_reverse_cosine_match_paper_equation():
    target_lr = 1e-3
    base_ratio = 0.01
    num_layers = 5

    for layer_idx in range(num_layers):
        x = layer_idx / (num_layers - 1)
        expected_scale = math.sin(math.pi * x / 2.0)
        expected_lr = target_lr * (
            base_ratio + (1.0 - base_ratio) * expected_scale
        )
        assert compute_layer_lr_scale(
            layer_idx, num_layers, "cosine"
        ) == pytest.approx(expected_scale)
        assert layer_lr_for_schedule(
            target_lr, layer_idx, num_layers, base_ratio, "cosine"
        ) == pytest.approx(expected_lr)


def test_aggregated_fisher_and_non_diagonal_quadratic_match_paper_oracle():
    # Deliberately non-diagonal gradients: a diagonal-only or grouped Fisher
    # implementation cannot pass this oracle.
    score = torch.tensor(
        [
            [[1.0, 2.0], [3.0, -1.0]],
            [[-2.0, 4.0], [0.5, 1.5]],
        ],
        dtype=torch.float32,
    )
    manager = FisherHookManager()
    manager._sums = [None]
    manager._make_grad_hook(0)(score * LOSS_GRAD_SCALE)
    manager.add_token_count(score.shape[0] * score.shape[1])
    sums, token_count = manager.finalize()

    flat_score = score.reshape(-1, 2)
    expected_fisher = sum(
        torch.outer(row, row) for row in flat_score
    ) / token_count
    actual_fisher = sums[0] / token_count
    torch.testing.assert_close(
        actual_fisher, expected_fisher, rtol=0, atol=0
    )
    assert actual_fisher[0, 1] != 0

    q_out = torch.tensor(
        [[[2.0, -1.0], [0.0, 3.0]]],
        requires_grad=True,
    )
    fp_out = torch.tensor([[[0.5, 0.5], [1.0, 1.0]]])
    loss = fisher_mse_loss(q_out, fp_out, actual_fisher)
    delta = q_out - fp_out
    expected_terms = torch.stack(
        [
            0.5 * row @ expected_fisher @ row
            for row in delta.reshape(-1, 2)
        ]
    )
    expected_loss = expected_terms.mean()
    expected_grad = (
        delta.detach().reshape(-1, 2) @ expected_fisher
        / delta.reshape(-1, 2).shape[0]
    ).reshape_as(delta)
    actual_grad = torch.autograd.grad(loss, q_out)[0]
    torch.testing.assert_close(loss, expected_loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        actual_grad, expected_grad, rtol=1e-6, atol=1e-7
    )


def test_activation_aware_uses_reported_constant_lr_but_fp16_flag_is_noop():
    target_lr = 1e-3
    base_ratio = 0.01
    for layer_idx in range(4):
        assert layer_lr_for_schedule(
            target_lr,
            layer_idx,
            4,
            base_ratio,
            "cosine",
            activation_aware=True,
        ) == pytest.approx(target_lr)

    assert Config(
        act_quant_aware_gptq=True,
        a_bits=16,
        v_bits=16,
    ).activation_aware_quantization_enabled is False
    assert Config(
        act_quant_aware_gptq=True,
        a_bits=4,
    ).activation_aware_quantization_enabled is True


def test_paper_defaults_and_conditional_akv_clip_presets():
    fp = Config()
    assert fp.grad_hessian_topk <= 0
    assert fp.kl_topk <= 0
    assert fp.saliency_clip_percentile == pytest.approx(0.99)
    assert fp.a_loss_ratio == pytest.approx(1.0)
    assert (fp.a_clip_ratio, fp.k_clip_ratio, fp.v_clip_ratio) == (
        1.0,
        1.0,
        1.0,
    )

    low = Config(a_bits=4, k_bits=4, v_bits=4)
    assert (low.a_clip_ratio, low.k_clip_ratio, low.v_clip_ratio) == (
        0.9,
        0.9,
        0.9,
    )
    explicit = parse_cli(
        [
            "--a_bits",
            "4",
            "--a_clip_ratio",
            "1.0",
            "--final_layer_grad_clip",
            "0.0005",
        ]
    )
    assert explicit.a_clip_ratio == pytest.approx(1.0)
    assert explicit.final_layer_grad_clip == pytest.approx(5e-4)
    assert fp.final_layer_backward_bsz == fp.backward_bsz == 32


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"w_bits": 1}, "w_bits"),
        ({"w_asym": True}, "w_asym"),
        ({"w_groupsize": 64, "blocksize": 128}, "w_groupsize"),
        ({"group_parallel_quant": "tensor"}, "group_parallel_quant"),
    ],
)
def test_refactored_weight_config_rejects_unsupported_modes(
    kwargs, message
):
    with pytest.raises(ValueError, match=message):
        Config(**kwargs)


def test_legacy_single_hessian_retry_does_not_accumulate_damping():
    # Eigenvalues are 2.025 and -0.025, so pristine-H retries with diagonal
    # damping 0.01, 0.02, 0.03 must first succeed at 0.03. The old cumulative
    # bug succeeded on the second try because it accidentally applied
    # 0.01 + 0.02 while reporting 0.02.
    hessian = torch.tensor([[1.0, 1.025], [1.025, 1.0]])
    solver = object.__new__(GPTQPlus)
    _hinv_init, _hinv, damp_used, fallback = (
        solver._compute_hessian_inverse_with_fallback(
            hessian,
            percdamp=0.01,
            damp_auto_increment=0.01,
        )
    )
    assert damp_used == pytest.approx(0.03)
    assert fallback is False


def test_legacy_batched_hessian_retry_does_not_accumulate_damping():
    hessians = torch.stack(
        [
            torch.eye(2),
            torch.tensor([[1.0, 1.025], [1.025, 1.0]]),
        ]
    )
    solver = object.__new__(GPTQPlus)
    _hinv_init, _hinv, damp_used, fallback = (
        solver._compute_hessian_inverse_batched_with_fallback(
            hessians,
            percdamp=0.01,
            damp_auto_increment=0.01,
        )
    )
    torch.testing.assert_close(
        damp_used,
        torch.tensor([0.01, 0.03]),
        rtol=1e-5,
        atol=1e-7,
    )
    assert fallback.tolist() == [False, False]


def test_refactored_batched_hessian_is_operation_exact_with_legacy():
    torch.manual_seed(17)
    base = torch.randn(4, 8, 8)
    hessians = base @ base.transpose(-1, -2)
    solver = object.__new__(GPTQPlus)
    _initial, legacy, _damp, _fallback = (
        solver._compute_hessian_inverse_batched_with_fallback(
            hessians.clone(),
            percdamp=0.01,
            damp_auto_increment=0.0015,
        )
    )
    refactored = cholesky_inverse_batched_with_damp(
        hessians.clone(),
        percdamp=0.01,
        damp_auto_increment=0.0015,
    )
    assert torch.equal(legacy, refactored)


def test_global_a_loss_threshold_is_microbatch_partition_invariant():
    q_out = torch.tensor(
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[4.0, 5.0], [6.0, 100.0]],
            [[8.0, 9.0], [10.0, 11.0]],
            [[12.0, 13.0], [14.0, 15.0]],
        ],
        requires_grad=True,
    )
    fp_out = torch.zeros_like(q_out)
    fisher = torch.eye(2)
    threshold = global_percentile(
        (q_out - fp_out).detach().abs(), 0.95
    )
    full = fisher_mse_loss(
        q_out,
        fp_out,
        fisher,
        a_loss_ratio=0.95,
        a_loss_threshold=threshold,
    )
    split = sum(
        fisher_mse_loss(
            q_out[start : start + 1],
            fp_out[start : start + 1],
            fisher,
            a_loss_ratio=0.95,
            a_loss_threshold=threshold,
        )
        for start in range(q_out.shape[0])
    ) / q_out.shape[0]
    torch.testing.assert_close(full, split, rtol=0, atol=0)

    # The historical per-microbatch P95 is observably a different objective.
    local_split = sum(
        fisher_mse_loss(
            q_out[start : start + 1],
            fp_out[start : start + 1],
            fisher,
            a_loss_ratio=0.95,
        )
        for start in range(q_out.shape[0])
    ) / q_out.shape[0]
    assert not torch.isclose(full, local_split)


def test_activation_loss_clip_matches_legacy_bf16_value_and_gradient():
    torch.manual_seed(9)
    fp_out = torch.randn(4, 3, 8, dtype=torch.bfloat16)
    noise = 0.1 * torch.randn_like(fp_out.float())
    q_old = (fp_out.float() + noise).to(torch.bfloat16).requires_grad_()
    q_new = q_old.detach().clone().requires_grad_()
    base = torch.randn(8, 8)
    fisher = base @ base.T
    threshold = global_percentile(
        (q_old - fp_out).detach().float().abs(), 0.95
    )

    old_loss = compute_refresh_loss(
        "fisher_diag_mse",
        q_old,
        fp_out,
        None,
        -1,
        layer_output_fisher=fisher,
        a_loss_ratio=0.95,
        a_loss_threshold=threshold,
    )
    new_loss = fisher_mse_loss(
        q_new,
        fp_out,
        fisher,
        a_loss_ratio=0.95,
        a_loss_threshold=threshold,
    )
    old_grad = torch.autograd.grad(old_loss, q_old)[0]
    new_grad = torch.autograd.grad(new_loss, q_new)[0]

    assert torch.equal(old_loss, new_loss)
    assert torch.equal(old_grad, new_grad)


class _TinyKLAnalyzer:
    def __init__(self):
        self.norm = torch.nn.Identity()
        self.head = torch.nn.Linear(
            3, 7, bias=False, dtype=torch.bfloat16
        )

    def get_layernorm_before_head(self):
        return self.norm

    def get_lm_head(self):
        return self.head


def test_full_vocab_kl_matches_closed_form_value_and_student_gradient():
    # Teacher p=(3/4, 1/4), student q=(1/2, 1/2).  This closed-form oracle
    # deliberately does not call softmax/log_softmax outside the production
    # primitive under test.
    student = torch.tensor([[0.0, 0.0]], requires_grad=True)
    teacher = torch.tensor([[math.log(3.0), 0.0]])
    loss = tokenwise_kl_from_logits(student, teacher).sum()
    gradient = torch.autograd.grad(loss, student)[0]

    expected_loss = 0.75 * math.log(1.5) + 0.25 * math.log(0.5)
    torch.testing.assert_close(
        loss, torch.tensor(expected_loss), rtol=1e-6, atol=1e-7
    )
    torch.testing.assert_close(
        gradient,
        torch.tensor([[-0.25, 0.25]]),
        rtol=1e-6,
        atol=1e-7,
    )


def test_old_new_final_kl_use_same_fp32_value_and_gradient():
    torch.manual_seed(4)
    analyzer = _TinyKLAnalyzer()
    fp = torch.randn(2, 3, 3, dtype=torch.bfloat16)
    q_old = fp.clone().requires_grad_(True)
    q_new = fp.clone().requires_grad_(True)
    with torch.no_grad():
        q_old[0, 0, 0] += torch.tensor(
            0.03125, dtype=torch.bfloat16
        )
        q_new.copy_(q_old)

    old_loss = compute_refresh_loss(
        "kl", q_old, fp, analyzer, kl_topk=-1
    )
    new_loss = kl_topk_loss(q_new, fp, analyzer, kl_topk=-1)
    old_grad = torch.autograd.grad(old_loss, q_old)[0]
    new_grad = torch.autograd.grad(new_loss, q_new)[0]

    assert old_loss.item() >= 0
    torch.testing.assert_close(old_loss, new_loss, rtol=0, atol=0)
    torch.testing.assert_close(old_grad, new_grad, rtol=0, atol=0)
