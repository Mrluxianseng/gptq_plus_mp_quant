from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from gptq_utils.gptq_plus_utils import (  # noqa: E402
    GPTQPlus,
    SaliencyCache,
    _STATIC_SALIENCY_SCHEMA_TAG,
    compute_layer_lr_scale,
    compute_refresh_loss,
    refresh_dynamic_saliency,
)
from realq.config import Config, parse_cli  # noqa: E402
from realq.precompute.hooks import (  # noqa: E402
    FisherHookManager,
    LOSS_GRAD_SCALE,
    SaliencyHookManager,
)
from realq.quant.realq_layer import RealQLayer  # noqa: E402
from realq.refresh.block_gd import layer_lr_for_schedule  # noqa: E402
from realq.refresh.fisher_loss import fisher_mse_loss  # noqa: E402
from realq.refresh.kl_loss import kl_topk_loss  # noqa: E402
from realq.quant.hessian import (  # noqa: E402
    cholesky_inverse_batched_with_damp,
)
from utils.loss_utils import tokenwise_kl_from_logits  # noqa: E402
from utils.saliency_utils import (  # noqa: E402
    global_percentile,
    grouped_channel_gram,
    grouped_gradient_norm_squared,
)


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


def test_grouped_saliency_is_paper_squared_norm_not_channel_mean():
    gradient = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 2.0, -2.0]]]
    )
    actual = grouped_gradient_norm_squared(gradient, num_groups=2)
    expected = torch.tensor([[[5.0, 25.0], [1.0, 8.0]]])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, expected / 2.0)


def test_dynamic_saliency_gram_uses_channel_sum_scale():
    matrix = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    )
    actual = grouped_channel_gram(matrix, num_groups=2)
    expected = torch.stack(
        [matrix[:2].T @ matrix[:2], matrix[2:].T @ matrix[2:]]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, expected / 2.0)


def test_dynamic_saliency_sum_scale_survives_hessian_finalization():
    projection = torch.eye(4)
    channel_gram = grouped_channel_gram(projection, num_groups=2)
    dyn_entry = {
        "U_Sigma": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        "W_cross_packed": channel_gram,
        "R_eff": 4,
    }
    scale = 1_000_000.0
    saliency = refresh_dynamic_saliency(
        dyn_entry=dyn_entry,
        static_saliency=torch.zeros(1, 1, 2),
        N_global=1,
        P_delta=torch.tensor([[[5.0, 6.0, 7.0, 8.0]]]),
        num_groups=2,
        dev=torch.device("cpu"),
        static_saliency_scale=scale,
    )
    expected_saliency = torch.tensor([[[34.0, 106.0]]]) * scale
    torch.testing.assert_close(
        saliency, expected_saliency, rtol=0, atol=0
    )

    solver = GPTQPlus(
        torch.nn.Linear(2, 4, bias=False),
        saliency,
        gradient=torch.zeros(2, 2),
        num_groups=2,
        alpha=0.0,
        reference_loss=0.0,
        hessian_saliency_scale=scale,
    )
    inputs = torch.tensor([[[2.0, 3.0]]])
    solver.add_batch(inputs, out=None)
    solver.finalize_hessian()
    outer = inputs.reshape(-1, 2).T @ inputs.reshape(-1, 2)
    expected_hessian = torch.stack([34.0 * outer, 106.0 * outer])
    torch.testing.assert_close(
        solver.H, expected_hessian, rtol=0, atol=0
    )


def test_old_and_new_saliency_collectors_feed_exact_paper_hessian():
    gradient = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 2.0, -2.0]]]
    )
    expected_saliency = torch.tensor([[[5.0, 25.0], [1.0, 8.0]]])

    legacy_output = torch.zeros_like(gradient, requires_grad=True)
    legacy = SaliencyCache(["proj"], num_groups=2)
    legacy.hooks_enabled = True
    legacy.cache_saliency(None, None, legacy_output, "proj")
    legacy_output.backward(gradient)
    legacy_saliency = legacy.saliency_cache["proj"][0]

    module = torch.nn.Linear(4, 4, bias=False)
    refactored = SaliencyHookManager(
        num_groups=2, clip_percentile=None
    )
    refactored.attach([{"proj": module}])
    refactored_output = module(torch.zeros_like(gradient))
    refactored_output.backward(gradient)
    refactored_saliency = refactored.finalize()[0]["proj"]
    refactored.remove()

    torch.testing.assert_close(
        legacy_saliency, expected_saliency, rtol=0, atol=0
    )
    torch.testing.assert_close(
        refactored_saliency, expected_saliency, rtol=0, atol=0
    )

    inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    realq = RealQLayer(
        torch.nn.Linear(2, 4, bias=False),
        refactored_saliency,
        quantizer=None,
        num_groups=2,
        dev=torch.device("cpu"),
    )
    realq.add_batch(inputs)
    flat_inputs = inputs.reshape(-1, 2)
    expected_hessian = torch.stack(
        [
            flat_inputs.T
            @ torch.diag(expected_saliency.reshape(-1, 2)[:, group])
            @ flat_inputs
            for group in range(2)
        ]
    )
    torch.testing.assert_close(realq.H, expected_hessian, rtol=0, atol=0)
    # v1 already existed while the optional dynamic correction still used a
    # channel mean. v2 is required to invalidate those payloads.
    assert _STATIC_SALIENCY_SCHEMA_TAG == "salsumv2"


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
    with pytest.raises(ValueError, match="requires `k_bits < 16`"):
        Config(
            k_cache_quant_aware_gptq=True,
            k_bits=16,
        )


def test_paper_defaults_and_conditional_akv_clip_presets():
    fp = Config()
    assert fp.quantizer_inner_fastpath is True
    assert fp.prepared_clamp_bound_cache is True
    assert fp.triton_column_block is True
    assert fp.fused_block_adam is True
    assert fp.w_clip_search_impl == "symmetric_union_exact"
    assert fp.fisher_fp32_cache is True
    assert fp.act_order_stitch_impl == "prefix_q_trailing_w_exact"
    assert fp.w_clip_update_impl == "where_out"
    assert fp.w_group_param_layout == "compact"
    assert fp.grad_hessian_topk <= 0
    assert fp.kl_topk <= 0
    assert fp.saliency_clip_percentile == pytest.approx(0.99)
    assert fp.a_loss_ratio == pytest.approx(1.0)
    assert fp.a_loss_clip_scope == "local_backward_chunk"
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
    assert parse_cli(
        ["--quantizer_inner_fastpath", "true"]
    ).quantizer_inner_fastpath is True
    assert (
        parse_cli(
            [
                "--act_order_stitch_impl",
                "prefix_q_trailing_w_exact",
            ]
        ).act_order_stitch_impl
        == "prefix_q_trailing_w_exact"
    )
    assert parse_cli(
        ["--w_clip_update_impl", "where_out"]
    ).w_clip_update_impl == "where_out"


def test_activation_loss_clip_scope_is_explicit_and_validated():
    assert parse_cli(
        ["--a_loss_clip_scope", "global_refresh"]
    ).a_loss_clip_scope == "global_refresh"
    assert Config(
        a_loss_clip_scope="local_backward_chunk"
    ).a_loss_clip_scope == "local_backward_chunk"
    with pytest.raises(ValueError, match="a_loss_clip_scope"):
        Config(a_loss_clip_scope="per_token")


def test_required_static_cache_hit_needs_an_explicit_cache_path():
    with pytest.raises(ValueError, match="requires `static_cache_path`"):
        Config(require_static_cache_hit=True)
    assert Config(
        static_cache_path="/tmp/static-cache",
        require_static_cache_hit=True,
    ).require_static_cache_hit is True
    assert Config(
        require_reference_cache_hit=True
    ).require_reference_cache_hit is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"w_bits": 1}, "w_bits"),
        ({"w_asym": True}, "w_asym"),
        ({"w_groupsize": 96, "blocksize": 128}, "integer multiple"),
        (
            {
                "w_groupsize": 64,
                "blocksize": 128,
                "act_order": False,
            },
            "act_order=True",
        ),
        ({"w_clip_search_impl": "unordered"}, "w_clip_search_impl"),
        ({"w_clip_update_impl": "unordered"}, "w_clip_update_impl"),
        ({"group_parallel_quant": "tensor"}, "group_parallel_quant"),
        ({"quantizer_inner_fastpath": 1}, "quantizer_inner_fastpath"),
        ({"act_order_stitch_impl": "unordered"}, "act_order_stitch_impl"),
    ],
)
def test_refactored_weight_config_rejects_unsupported_modes(
    kwargs, message
):
    with pytest.raises(ValueError, match=message):
        Config(**kwargs)


def test_refactored_weight_config_accepts_multiple_static_groups_per_block():
    cfg = Config(w_groupsize=128, blocksize=512, act_order=True)
    assert cfg.w_groupsize == 128
    assert cfg.blocksize == 512


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


def test_historical_local_a_loss_clip_matches_without_shared_threshold():
    torch.manual_seed(19)
    fp_out = torch.randn(3, 4, 8, dtype=torch.bfloat16)
    q_old = (
        fp_out.float() + 0.2 * torch.randn_like(fp_out.float())
    ).to(torch.bfloat16).requires_grad_()
    q_new = q_old.detach().clone().requires_grad_()
    base = torch.randn(8, 8)
    fisher = base @ base.T

    old_loss = compute_refresh_loss(
        "fisher_diag_mse",
        q_old,
        fp_out,
        None,
        -1,
        layer_output_fisher=fisher,
        a_loss_ratio=0.95,
    )
    new_loss = fisher_mse_loss(
        q_new,
        fp_out,
        fisher,
        a_loss_ratio=0.95,
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
