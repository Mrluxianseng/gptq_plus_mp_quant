from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from transformers.models.qwen3_moe.configuration_qwen3_moe import (
    Qwen3MoeConfig,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock,
)

from realq_moe import model_adapter
from realq_moe.precompute.routed_stats import PackedLayerRoutes
from realq_moe.runner import layer_loop, moe_layer_loop
from realq_moe.runner.moe_layer_loop import SparseReplay
from realq_moe.runner.streams import LayerInputs
from utils import hadamard_utils
from utils.quant_utils import ActQuantWrapper


class _CountingScheduler:
    def __init__(self, n_total: int) -> None:
        self.n_total = int(n_total)
        self.calls = 0

    def next_indices(self) -> list[int]:
        self.calls += 1
        return list(range(self.n_total))


class _TinyQwenMoeLayer(torch.nn.Module):
    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.mlp = Qwen3MoeSparseMoeBlock(config)
        self.post_attention_layernorm = torch.nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        **_: object,
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return (residual + self.mlp(hidden_states)[0],)


def test_release_consumed_moe_static_layer_keeps_indices_and_no_cpu_copy() -> None:
    device = torch.device("cpu")
    untouched = torch.ones(3, device=device)
    static = SimpleNamespace(
        saliency=[{"expert": torch.ones(2, device=device)}, {"next": untouched}],
        fisher=[torch.ones(2, 2, dtype=torch.bfloat16), untouched],
        routes=[{0: {"flat_token_indices": torch.arange(2)}}, {"next": untouched}],
        expert_global_assignment_counts=[
            torch.ones(1, dtype=torch.int64),
            untouched,
        ],
        expert_global_coverage=[
            {"assignment_count": torch.ones(1, dtype=torch.int64)},
            {"next": untouched},
        ],
    )

    layer_loop._release_consumed_moe_static_layer(static, 0, device)

    assert static.saliency[0] == {}
    assert static.fisher[0].numel() == 0
    assert static.fisher[0].device == device
    assert static.routes[0] == {}
    assert static.expert_global_assignment_counts[0].numel() == 0
    assert static.expert_global_assignment_counts[0].device == device
    assert static.expert_global_coverage[0] == {}
    assert static.saliency[1]["next"] is untouched
    assert static.fisher[1] is untouched
    assert static.routes[1]["next"] is untouched


def _route_payload(
    *,
    token_count: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "flat_token_indices": torch.arange(
            token_count,
            dtype=torch.int32,
            device=device,
        ),
        "topk_slots": torch.zeros(
            token_count,
            dtype=torch.uint8,
            device=device,
        ),
        "route_weights": torch.ones(
            token_count,
            dtype=torch.bfloat16,
            device=device,
        ),
    }


def _packed_routes(
    *,
    token_count: int,
    num_experts: int,
    device: torch.device,
) -> PackedLayerRoutes:
    payloads = [
        _route_payload(token_count=token_count, device=device)
        for _ in range(num_experts)
    ]
    offsets = tuple(
        expert_idx * token_count for expert_idx in range(num_experts + 1)
    )
    return PackedLayerRoutes(
        expert_offsets=torch.tensor(
            offsets,
            dtype=torch.int64,
            device=device,
        ),
        flat_token_indices=torch.cat(
            [payload["flat_token_indices"] for payload in payloads],
            dim=0,
        ),
        topk_slots=torch.cat(
            [payload["topk_slots"] for payload in payloads],
            dim=0,
        ),
        route_weights=torch.cat(
            [payload["route_weights"] for payload in payloads],
            dim=0,
        ),
        _offset_values=offsets,
    )


def _runtime_cfg(*, nsamples: int, seq_len: int) -> SimpleNamespace:
    return SimpleNamespace(
        moe_joint_column_block=True,
        moe_expert_loss_slide_window=True,
        hessian_accum_bsz=nsamples,
        moe_min_expert_assignments=1,
        moe_min_expert_unique_tokens=1,
        moe_min_expert_unique_samples=1,
        moe_zero_route_fallback="rtn",
        moe_expert_chunk_assignments=64,
        nsamples=nsamples,
        seq_len=seq_len,
        num_groups=4,
        group_parallel_quant="rank",
        w_bits=4,
        w_asym=False,
        w_clip=False,
        w_groupsize=4,
        w_clip_search_impl="cartesian_legacy",
        w_clip_update_impl="guarded",
        w_group_param_layout="expanded",
        blocksize=4,
        percdamp=0.01,
        act_order=True,
        log_column_block_loss=False,
        kl_topk=-1,
        a_loss_ratio=1.0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("slide_enabled", [False, True])
def test_joint_moe_runner_refreshes_once_per_projection_block(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    slide_enabled: bool,
) -> None:
    caplog.set_level(logging.INFO)
    torch.manual_seed(1701 + int(slide_enabled))
    device = torch.device("cuda")
    dtype = torch.bfloat16
    nsamples = 2
    seq_len = 3
    hidden_size = 8
    num_experts = 4
    config = Qwen3MoeConfig(
        hidden_size=hidden_size,
        intermediate_size=16,
        moe_intermediate_size=hidden_size,
        num_experts=num_experts,
        num_experts_per_tok=2,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    layer = _TinyQwenMoeLayer(config).to(
        device=device,
        dtype=dtype,
    ).eval()
    next_layer = _TinyQwenMoeLayer(config).to(
        device=device,
        dtype=dtype,
    ).eval()
    layer.requires_grad_(False)
    next_layer.requires_grad_(False)

    inputs = torch.randn(
        nsamples,
        seq_len,
        hidden_size,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        fp_outs = layer(inputs)[0]
        next_fp_outs = next_layer(fp_outs)[0]
    state = LayerInputs(
        inps=inputs,
        fp_inps=inputs.clone(),
        attention_mask=None,
        position_ids=None,
        position_embeddings=None,
    )

    routes = _packed_routes(
        token_count=nsamples * seq_len,
        num_experts=num_experts,
        device=device,
    )
    saliency = {}
    for expert_idx in range(num_experts):
        for projection in ("up_proj", "gate_proj", "down_proj"):
            path = (
                f"mlp.experts.{expert_idx}.{projection}"
            )
            saliency[path] = torch.ones(
                nsamples * seq_len,
                1,
                4,
                device=device,
                dtype=dtype,
            )
    static = SimpleNamespace(
        routes=[routes],
        saliency=[saliency],
        expert_global_assignment_counts=[
            torch.full(
                (num_experts,),
                nsamples * seq_len,
                device=device,
                dtype=torch.int64,
            )
        ],
    )

    def fake_capture(
        _layer,
        _state,
        *,
        batch_size: int,
        capture_student_routes: bool,
        capture_layer_outputs: bool,
    ) -> SparseReplay:
        assert batch_size == nsamples
        assert capture_student_routes
        assert not capture_layer_outputs
        return SparseReplay(
            mlp_inputs=inputs,
            mlp_residuals=inputs,
            layer_outputs=None,
            routes=routes,
        )

    monkeypatch.setattr(
        moe_layer_loop,
        "_capture_sparse_replay",
        fake_capture,
    )
    scheduler = _CountingScheduler(nsamples)
    alpha_calls: list[float] = []

    def alpha_fn() -> float:
        alpha = 1.0 - len(alpha_calls) / 2.0
        alpha_calls.append(alpha)
        return alpha

    weights_before = {
        name: parameter.detach().clone()
        for name, parameter in layer.named_parameters()
        if ".experts." in f".{name}" and name.endswith("weight")
    }
    moe_layer_loop.quantize_sparse_experts(
        cfg=_runtime_cfg(nsamples=nsamples, seq_len=seq_len),
        layer_idx=0,
        layer=layer,
        static=static,
        state=state,
        fp_outs=fp_outs,
        fisher=torch.eye(hidden_size, device=device, dtype=dtype),
        layer_lr=1e-3,
        grad_clip=1.0,
        refresh_bsz_local=nsamples,
        sample_scheduler=scheduler,
        block_gd_enabled=True,
        use_kl_refresh=False,
        analyzer=None,
        trace_writer=None,
        dev=device,
        next_layer=next_layer if slide_enabled else None,
        next_fp_outs=next_fp_outs if slide_enabled else None,
        next_fisher=(
            torch.eye(hidden_size, device=device, dtype=dtype)
            if slide_enabled
            else None
        ),
        slide_alpha_fn=alpha_fn if slide_enabled else None,
    )

    # C=8/blocksize=4 gives one non-final boundary for each of up, gate,
    # and down.  The count is three, never three times num_experts.
    assert scheduler.calls == 3
    assert len(alpha_calls) == 3 * int(slide_enabled)
    completion_logs = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith(
            "[realq_moe.joint_complete]"
        )
    ]
    assert completion_logs == [
        (
            "[realq_moe.joint_complete] scope=projection layer_idx=0 "
            "projection=up_proj actual_refreshes=1 "
            "expected_refreshes=1"
        ),
        (
            "[realq_moe.joint_complete] scope=projection layer_idx=0 "
            "projection=gate_proj actual_refreshes=1 "
            "expected_refreshes=1"
        ),
        (
            "[realq_moe.joint_complete] scope=projection layer_idx=0 "
            "projection=down_proj actual_refreshes=1 "
            "expected_refreshes=1"
        ),
        (
            "[realq_moe.joint_complete] scope=layer layer_idx=0 "
            "total_actual_refreshes=3 total_expected_refreshes=3"
        ),
    ]
    changed = 0
    for name, parameter in layer.named_parameters():
        if name not in weights_before:
            continue
        assert parameter.device.type == "cuda"
        assert torch.isfinite(parameter).all()
        changed += int(not torch.equal(parameter, weights_before[name]))
    assert changed == 3 * num_experts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_joint_capture_stops_after_natural_student_router() -> None:
    torch.manual_seed(1703)
    device = torch.device("cuda")
    nsamples, seq_len, hidden_size, num_experts, top_k = 2, 3, 8, 4, 2
    config = Qwen3MoeConfig(
        hidden_size=hidden_size,
        intermediate_size=16,
        moe_intermediate_size=hidden_size,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    layer = _TinyQwenMoeLayer(config).to(device).eval()
    layer.requires_grad_(False)
    inputs = torch.randn(
        nsamples,
        seq_len,
        hidden_size,
        device=device,
    )
    state = LayerInputs(
        inps=inputs,
        fp_inps=inputs.clone(),
        attention_mask=None,
        position_ids=None,
        position_embeddings=None,
    )
    expert_calls = {"count": 0}

    def count_expert(_module, _inputs):
        expert_calls["count"] += 1

    handles = [
        expert.register_forward_pre_hook(count_expert)
        for expert in layer.mlp.experts
    ]
    try:
        replay = moe_layer_loop._capture_sparse_replay(
            layer,
            state,
            batch_size=nsamples,
            capture_student_routes=True,
            capture_layer_outputs=False,
        )
    finally:
        for handle in handles:
            handle.remove()

    assert expert_calls["count"] == 0
    assert replay.layer_outputs is None
    assert replay.routes is not None
    assert sum(
        int(replay.routes[expert_idx]["flat_token_indices"].numel())
        for expert_idx in range(num_experts)
    ) == nsamples * seq_len * top_k
    torch.testing.assert_close(replay.mlp_inputs, inputs)
    torch.testing.assert_close(replay.mlp_residuals, inputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("wrapped_online_hadamard", [False, True])
def test_direct_down_hessian_input_skips_down_projection(
    wrapped_online_hadamard: bool,
) -> None:
    torch.manual_seed(1704)
    device = torch.device("cuda")
    config = Qwen3MoeConfig(
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=8,
        num_experts=2,
        num_experts_per_tok=1,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    expert = Qwen3MoeSparseMoeBlock(config).experts[0].to(device).eval()
    expert.requires_grad_(False)
    if wrapped_online_hadamard:
        wrapper = ActQuantWrapper(expert.down_proj)
        had_k, k = hadamard_utils.get_hadK(wrapper.module.in_features)
        wrapper.online_full_had = True
        wrapper.had_K = (
            None
            if had_k is None
            else had_k.to(
                device=wrapper.module.weight.device,
                dtype=wrapper.module.weight.dtype,
            )
        )
        wrapper.K = k
        wrapper.fp32_had = False
        expert.down_proj = wrapper
        down_linear = wrapper.module
    else:
        down_linear = expert.down_proj
    x = torch.randn(7, config.hidden_size, device=device)
    observed: list[torch.Tensor] = []
    completed_down_gemms = {"count": 0}

    def capture_down_input(_module, inputs):
        observed.append(inputs[0].detach().clone())

    def count_completed_down_gemm(_module, _inputs, _output):
        completed_down_gemms["count"] += 1

    capture_handle = down_linear.register_forward_pre_hook(capture_down_input)
    completion_handle = down_linear.register_forward_hook(
        count_completed_down_gemm
    )
    try:
        with torch.no_grad():
            expert(x)
            assert len(observed) == 1
            expected = observed.pop()
            assert completed_down_gemms["count"] == 1
            completed_down_gemms["count"] = 0
            actual = moe_layer_loop._down_input_for_hessian(
                expert=expert,
                x=x,
            )
    finally:
        capture_handle.remove()
        completion_handle.remove()

    # The inner pre-hook observes the transformed input, but the private
    # capture exception prevents the Linear forward hook (and therefore its
    # GEMM) from completing.
    assert len(observed) == 1
    torch.testing.assert_close(observed.pop(), expected)
    assert completed_down_gemms["count"] == 0
    torch.testing.assert_close(actual, expected)
    if wrapped_online_hadamard:
        raw = expert.act_fn(expert.gate_proj(x)) * expert.up_proj(x)
        assert not torch.equal(actual, raw)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_repaired_moe_down_hadamard_is_cuda_resident_and_numerical() -> None:
    torch.manual_seed(1705)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    config = Qwen3MoeConfig(
        hidden_size=8,
        intermediate_size=16,
        # Non-power-of-two width exercises an explicit had_K tensor rather
        # than the implicit K=1 transform.
        moe_intermediate_size=12,
        num_experts=2,
        num_experts_per_tok=1,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    layer = _TinyQwenMoeLayer(config).to(device=device, dtype=dtype).eval()
    layer.requires_grad_(False)
    for expert in layer.mlp.experts:
        expert.down_proj = ActQuantWrapper(expert.down_proj)

    # Match the production lifecycle: wrapper installation is followed by the
    # one whole-model residency transition before Stage 0.  Stage-1 AKV setup
    # then re-enters the repair function without constructing new wrappers.
    layer.to(device)
    analyzer = SimpleNamespace(get_layers=lambda: [layer])
    model_adapter.repair_moe_down_rotation_wrappers(analyzer)
    model_adapter.repair_moe_down_rotation_wrappers(analyzer)
    layer_device = next(layer.parameters()).device
    for _name, buffer in layer.named_buffers():
        assert buffer.device == layer_device

    raw = torch.randn(
        5,
        config.moe_intermediate_size,
        device=device,
        dtype=dtype,
    )
    expected_had_k, expected_k = hadamard_utils.get_hadK(
        config.moe_intermediate_size
    )
    assert expected_had_k is not None
    assert expected_k > 1
    expected_had_k = expected_had_k.to(device=device, dtype=dtype)

    for expert in layer.mlp.experts:
        wrapper = expert.down_proj
        assert isinstance(wrapper, ActQuantWrapper)
        assert wrapper.online_full_had
        assert not wrapper.fp32_had
        assert wrapper.K == expected_k
        assert wrapper.had_K is not None
        assert wrapper.had_K.device == expert.down_proj.module.weight.device
        assert wrapper.had_K.dtype == expert.down_proj.module.weight.dtype
        for _name, buffer in wrapper.named_buffers():
            assert buffer.device == wrapper.module.weight.device

        with torch.no_grad():
            expected_input = hadamard_utils.matmul_hadU_cuda(
                raw,
                expected_had_k,
                expected_k,
            )
            expected = torch.nn.functional.linear(
                expected_input,
                wrapper.module.weight,
                wrapper.module.bias,
            ).to(dtype)
            actual = wrapper(raw)
        torch.testing.assert_close(actual, expected)
