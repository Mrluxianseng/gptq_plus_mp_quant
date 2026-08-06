from __future__ import annotations

from collections.abc import Mapping

import pytest

torch = pytest.importorskip("torch")

from torch.func import functional_call
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
)

from realq_moe.precompute import cache as cache_mod
from realq_moe.precompute.hooks import SaliencyHookManager
from realq_moe.precompute.routed_stats import (
    PackedLayerRoutes,
    RouteCaptureManager,
)
from realq_moe.quant.realq_layer import LOSS_GRAD_SCALE
from realq_moe.quant.routed_realq_layer import RoutedRealQLayer
from realq_moe.refresh.block_gd import RefreshContext, _SharedSampleScheduler
from realq_moe.refresh.fisher_loss import fisher_mse_loss
from realq_moe.refresh.moe_block_gd import (
    _underlying_weight_name,
    make_moe_expert_refresh_fn,
)
from utils.quant_utils import ActQuantWrapper


def _tiny_sparse_layer() -> Qwen3MoeDecoderLayer:
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        decoder_sparse_step=1,
        moe_intermediate_size=8,
        num_experts_per_tok=2,
        num_experts=4,
        norm_topk_prob=True,
        attention_dropout=0.0,
        use_cache=False,
    )
    return Qwen3MoeDecoderLayer(config, layer_idx=0).eval()


def _official_routes(
    layer: Qwen3MoeDecoderLayer,
    hidden: torch.Tensor,
    *,
    flat_offset: int,
) -> dict[int, dict[str, torch.Tensor]]:
    mlp = layer.mlp
    flat = hidden.reshape(-1, hidden.shape[-1])
    logits = mlp.gate(flat)
    weights = torch.softmax(logits, dim=1, dtype=torch.float32)
    weights, selected = torch.topk(weights, mlp.top_k, dim=-1)
    if mlp.norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights.to(hidden.dtype)
    mask = torch.nn.functional.one_hot(
        selected, num_classes=mlp.num_experts
    ).permute(2, 1, 0)
    out: dict[int, dict[str, torch.Tensor]] = {}
    for expert_idx in range(mlp.num_experts):
        slot, token = torch.where(mask[expert_idx])
        out[expert_idx] = {
            "flat_token_indices": token + flat_offset,
            "topk_slots": slot,
            "route_weights": weights[token, slot],
        }
    return out


def _capture_routes(
    layer: Qwen3MoeDecoderLayer,
    batches: list[torch.Tensor],
) -> tuple[
    list[torch.Tensor],
    Mapping[int, Mapping[str, torch.Tensor]],
]:
    manager = RouteCaptureManager([layer])
    outputs: list[torch.Tensor] = []
    sample_start = 0
    try:
        for hidden in batches:
            manager.begin_batch(
                local_start=sample_start,
                batch_size=int(hidden.shape[0]),
                seq_len=int(hidden.shape[1]),
            )
            outputs.append(layer.mlp(hidden)[0])
            manager.release_current()
            sample_start += int(hidden.shape[0])
    finally:
        manager.remove()
    route_layers = manager.finalize()
    assert len(route_layers) == 1
    return outputs, route_layers[0]


def test_route_capture_matches_official_order_weights_and_output() -> None:
    torch.manual_seed(17)
    layer = _tiny_sparse_layer()
    batches = [
        torch.randn(2, 3, 16),
        torch.randn(1, 3, 16),
    ]
    actual_outputs, routes = _capture_routes(layer, batches)
    assert isinstance(routes, PackedLayerRoutes)
    assert routes.expert_offsets.dtype == torch.int64
    assert routes.flat_token_indices.dtype == torch.int32
    assert routes.topk_slots.dtype == torch.uint8
    assert routes.route_weights.dtype == batches[0].dtype
    for tensor in (
        routes.expert_offsets,
        routes.flat_token_indices,
        routes.topk_slots,
        routes.route_weights,
    ):
        assert tensor.device == batches[0].device

    expected_chunks = {expert_idx: [] for expert_idx in range(4)}
    sample_start = 0
    for hidden in batches:
        expected = _official_routes(
            layer,
            hidden,
            flat_offset=sample_start * hidden.shape[1],
        )
        for expert_idx, payload in expected.items():
            expected_chunks[expert_idx].append(payload)
        sample_start += int(hidden.shape[0])

    expected_counts = []
    for expert_idx in range(4):
        expected = {
            key: torch.cat(
                [chunk[key] for chunk in expected_chunks[expert_idx]], dim=0
            )
            for key in (
                "flat_token_indices",
                "topk_slots",
                "route_weights",
            )
        }
        expected_counts.append(
            int(expected["flat_token_indices"].numel())
        )
        torch.testing.assert_close(
            routes[expert_idx]["flat_token_indices"],
            expected["flat_token_indices"].to(dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            routes[expert_idx]["topk_slots"],
            expected["topk_slots"].to(dtype=torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            routes[expert_idx]["route_weights"],
            expected["route_weights"].to(dtype=batches[0].dtype),
            rtol=0,
            atol=0,
        )
        for field_name, backing in (
            ("flat_token_indices", routes.flat_token_indices),
            ("topk_slots", routes.topk_slots),
            ("route_weights", routes.route_weights),
        ):
            assert (
                routes[expert_idx][field_name]
                .untyped_storage()
                .data_ptr()
                == backing.untyped_storage().data_ptr()
            )

    expected_offsets = torch.tensor(
        [0, *torch.tensor(expected_counts).cumsum(0).tolist()],
        dtype=torch.int64,
        device=batches[0].device,
    )
    torch.testing.assert_close(
        routes.expert_offsets,
        expected_offsets,
        rtol=0,
        atol=0,
    )

    all_hidden = torch.cat(batches, dim=0)
    reconstructed = torch.zeros_like(all_hidden).reshape(-1, 16)
    flat_hidden = all_hidden.reshape(-1, 16)
    for expert_idx in range(4):
        payload = routes[expert_idx]
        ids = payload["flat_token_indices"].long()
        contribution = layer.mlp.experts[expert_idx](
            flat_hidden.index_select(0, ids)
        )
        contribution = contribution * payload["route_weights"].reshape(-1, 1)
        reconstructed.index_add_(0, ids, contribution)
    torch.testing.assert_close(
        reconstructed.reshape_as(all_hidden),
        torch.cat(actual_outputs, dim=0),
        # The captured route itself is checked bit-exactly above.  Replaying
        # all batches as one larger per-expert GEMM may select a different
        # accumulation shape than the two original forwards.
        rtol=1e-6,
        atol=5e-8,
    )


def test_packed_routes_cache_round_trip_preserves_canonical_storage(
    tmp_path,
) -> None:
    layer_routes = PackedLayerRoutes(
        expert_offsets=torch.tensor([0, 2, 3], dtype=torch.int64),
        flat_token_indices=torch.tensor([1, 4, 7], dtype=torch.int32),
        topk_slots=torch.tensor([0, 1, 0], dtype=torch.uint8),
        route_weights=torch.tensor([0.6, 0.4, 0.8]),
        _offset_values=(0, 2, 3),
    )
    global_counts = torch.tensor([2, 1], dtype=torch.int64)
    payload = {
        "saliency": [{}],
        "fisher": [torch.eye(2, dtype=torch.bfloat16)],
        "routes": [layer_routes],
        "expert_global_assignment_counts": [global_counts],
        "expert_global_coverage": [
            {
                "assignment_count": global_counts.clone(),
                "unique_token_count": global_counts.clone(),
                "unique_sample_count": torch.tensor(
                    [2, 1], dtype=torch.int64
                ),
                "affinity_mass": torch.tensor(
                    [1.0, 0.8], dtype=torch.float64
                ),
            }
        ],
    }
    cache_mod.save(str(tmp_path), "packed", 1, 0, payload)
    loaded = cache_mod.try_load(
        str(tmp_path),
        "packed",
        1,
        0,
        map_location="cpu",
    )
    assert loaded is not None
    restored = loaded["routes"][0]
    assert isinstance(restored, PackedLayerRoutes)
    restored.validate()
    assert restored.flat_token_indices.dtype == torch.int32
    assert restored.topk_slots.dtype == torch.uint8
    assert (
        restored[0]["flat_token_indices"].untyped_storage().data_ptr()
        == restored.flat_token_indices.untyped_storage().data_ptr()
    )

    legacy_payload = {
        **payload,
        "routes": [
            {
                expert_idx: {
                    field_name: layer_routes[expert_idx][field_name].clone()
                    for field_name in (
                        "flat_token_indices",
                        "topk_slots",
                        "route_weights",
                    )
                }
                for expert_idx in range(len(layer_routes))
            }
        ],
    }
    cache_mod.save(str(tmp_path), "legacy", 1, 0, legacy_payload)
    assert (
        cache_mod.try_load(
            str(tmp_path),
            "legacy",
            1,
            0,
            map_location="cpu",
        )
        is None
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_packed_route_capture_stays_cuda_resident() -> None:
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    layer = _tiny_sparse_layer().to(device)
    hidden = torch.randn(2, 3, 16, device=device, dtype=torch.float32)
    _, routes = _capture_routes(layer, [hidden])
    assert isinstance(routes, PackedLayerRoutes)
    routes.assert_resident_on(device)
    for tensor in (
        routes.expert_offsets,
        routes.flat_token_indices,
        routes.topk_slots,
        routes.route_weights,
    ):
        assert tensor.device == device


def test_routed_saliency_rows_align_with_teacher_routes() -> None:
    torch.manual_seed(29)
    layer = _tiny_sparse_layer()
    modules = {}
    expert_module_indices = {}
    for expert_idx, expert in enumerate(layer.mlp.experts):
        for projection in ("up_proj", "gate_proj", "down_proj"):
            name = f"mlp.experts.{expert_idx}.{projection}"
            modules[name] = getattr(expert, projection)
            expert_module_indices[name] = expert_idx

    route_manager = RouteCaptureManager([layer])
    saliency_manager = SaliencyHookManager(
        num_groups=4,
        clip_percentile=None,
    )
    saliency_manager.attach(
        [modules],
        expert_module_indices=[expert_module_indices],
        route_manager=route_manager,
    )
    hidden = torch.randn(3, 5, 16, requires_grad=True)
    try:
        route_manager.begin_batch(local_start=0, batch_size=3, seq_len=5)
        output = layer.mlp(hidden)[0]
        route_manager.release_current()
        output.square().sum().backward()
    finally:
        saliency_manager.remove()
        route_manager.remove()

    saliency = saliency_manager.finalize()[0]
    routes = route_manager.finalize()[0]
    for expert_idx in range(4):
        assignment_count = int(
            routes[expert_idx]["flat_token_indices"].numel()
        )
        for projection in ("up_proj", "gate_proj", "down_proj"):
            name = f"mlp.experts.{expert_idx}.{projection}"
            value = saliency[name]
            assert value.shape == (assignment_count, 1, 4)
            assert torch.isfinite(value).all()
            assert (value >= 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_routed_hessian_matches_direct_weighted_outer_product() -> None:
    torch.manual_seed(41)
    device = torch.device("cuda")
    groups = 4
    assignments = 7
    columns = 6
    linear = torch.nn.Linear(columns, 8, bias=False, device=device)
    saliency = torch.rand(
        assignments, 1, groups, device=device
    )
    inputs = torch.randn(assignments, columns, device=device)
    normalization_tokens = 11
    routed = RoutedRealQLayer(
        linear=linear,
        saliency=saliency,
        quantizer=object(),
        num_groups=groups,
        dev=device,
        normalization_token_count=normalization_tokens,
    )
    routed.add_batch(inputs[:2])
    routed.add_batch(inputs[2:5])
    routed.add_batch(inputs[5:])
    routed.finalize_hessian()

    expected = torch.einsum(
        "ag,ac,ad->gcd",
        saliency[:, 0].to(device),
        inputs,
        inputs,
    )
    expected = expected / normalization_tokens
    expected = expected / (LOSS_GRAD_SCALE * LOSS_GRAD_SCALE)
    expected = 0.5 * (expected + expected.transpose(-1, -2))
    torch.testing.assert_close(routed.H, expected, rtol=2e-5, atol=2e-8)
    torch.testing.assert_close(
        routed.act_square,
        inputs.square().sum(dim=0) / assignments,
        rtol=2e-5,
        atol=2e-6,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("wrapped", [False, True])
def test_contribution_refresh_gradient_matches_full_moe_forward(
    wrapped: bool,
) -> None:
    torch.manual_seed(53 + int(wrapped))
    device = torch.device("cuda")
    layer = _tiny_sparse_layer().to(device)
    hidden = torch.randn(4, 3, 16, device=device)
    _, routes = _capture_routes(layer, [hidden])
    expert_idx = max(
        routes,
        key=lambda idx: routes[idx]["flat_token_indices"].numel(),
    )
    expert = layer.mlp.experts[expert_idx]
    if wrapped:
        expert.up_proj = ActQuantWrapper(expert.up_proj)
        target_linear = expert.up_proj.module
    else:
        target_linear = expert.up_proj

    with torch.no_grad():
        y_running = layer.mlp(hidden)[0]
        ids = routes[expert_idx]["flat_token_indices"].to(
            device=device, dtype=torch.long
        )
        old_outputs = expert(
            hidden.reshape(-1, 16).index_select(0, ids)
        )
        fp_target = y_running + 0.03 * torch.randn_like(y_running)

    fisher = torch.randn(16, 16, device=device)
    fisher = fisher.transpose(0, 1) @ fisher / 16
    scheduler = _SharedSampleScheduler(
        n_total=hidden.shape[0],
        chunk_size=hidden.shape[0],
        seed=0,
    )
    ctx = RefreshContext(
        module=target_linear,
        layer_lr=0.25,
        grad_clip=0.0,
        backward_bsz=2,
        scheduler=scheduler,
    )
    candidate = (
        target_linear.weight.detach().float()
        + 0.02 * torch.randn_like(target_linear.weight, dtype=torch.float32)
    )
    refresh = make_moe_expert_refresh_fn(
        expert=expert,
        target_linear=target_linear,
        student_route=routes[expert_idx],
        mlp_inputs=hidden,
        y_running=y_running,
        old_expert_outputs=old_outputs,
        fp_out_for_this_layer=fp_target,
        fisher=fisher,
        ctx=ctx,
        seq_len=hidden.shape[1],
        global_sample_start=0,
        hit_sample_probability=1.0,
        a_loss_ratio=1.0,
    )
    update = refresh(candidate, 0)

    override = candidate.to(
        device=device,
        dtype=target_linear.weight.dtype,
    ).requires_grad_(True)
    expert_weight_name = _underlying_weight_name(expert, target_linear)
    full_weight_name = f"experts.{expert_idx}.{expert_weight_name}"
    with torch.enable_grad():
        candidate_output = functional_call(
            layer.mlp,
            {full_weight_name: override},
            (hidden,),
            strict=False,
        )[0]
        full_loss = fisher_mse_loss(
            candidate_output,
            fp_target,
            fisher,
            a_loss_ratio=1.0,
        )
        (full_grad,) = torch.autograd.grad(full_loss, override)

    recovered_grad = ctx.exp_avg / (1.0 - ctx.beta1)
    torch.testing.assert_close(
        recovered_grad,
        full_grad.float(),
        rtol=3e-4,
        atol=2e-6,
    )
    expected_update = ctx.layer_lr * full_grad.float() / (
        full_grad.float().abs() + ctx.eps
    )
    torch.testing.assert_close(
        update,
        expected_update,
        rtol=3e-4,
        atol=3e-5,
    )
