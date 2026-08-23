from __future__ import annotations

import math
import io

import pytest
import torch
import torch.nn.functional as F

from experiments.realq_fa4_q4_true_tensor_replay_20260822 import (
    capture_driver,
    common,
    compare_worker,
    runner,
)


def test_minimal_layer_and_case_contract() -> None:
    assert common.ALL_REQUESTED_LAYERS == (0, 5, 6, 12, 13, 35)
    assert common.SELECTED_LAYERS == (6,)
    definitions = common.case_definitions()
    assert [item["name"] for item in definitions] == [
        "real_b4",
        "real_b1_s0",
        "real_b1_s1",
        "real_b1_s2",
        "real_b1_s3",
        "real_b4_perm_s3_to_s0",
        "real_b4_s3_repeated",
        "random_b4",
    ]
    assert sum(int(item["batch"]) for item in definitions[1:5]) == 4
    assert definitions[5]["source_indices"] == [3, 0, 1, 2]
    assert definitions[6]["source_indices"] == [3, 3, 3, 3]
    assert common.b4_equivalent_case_count() == 5.0


def test_layout_hash_and_torch_roundtrip_bind_stride() -> None:
    bshd = torch.arange(2 * 7 * 4 * 5, dtype=torch.float32).reshape(2, 7, 4, 5)
    production_bhsd = bshd.transpose(1, 2)
    contiguous_bhsd = production_bhsd.contiguous()
    assert torch.equal(production_bhsd, contiguous_bhsd)
    assert common.tensors_semantic_sha256((production_bhsd,)) == common.tensors_semantic_sha256(
        (contiguous_bhsd,)
    )
    assert common.named_tensors_layout_sha256(
        {"q": production_bhsd}
    ) != common.named_tensors_layout_sha256({"q": contiguous_bhsd})
    assert production_bhsd.stride() == (140, 5, 20, 1)
    assert production_bhsd.transpose(1, 2).is_contiguous()

    buffer = io.BytesIO()
    torch.save({"q": production_bhsd}, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, map_location="cpu", weights_only=True)["q"]
    assert torch.equal(loaded, production_bhsd)
    assert loaded.stride() == production_bhsd.stride()
    assert loaded.storage_offset() == production_bhsd.storage_offset()


def test_capture_binds_canonical_sdpa_mask(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(common, "BATCH_SIZE", 2)
    monkeypatch.setattr(common, "SEQUENCE_LENGTH", 5)
    controller = capture_driver.CaptureController(tmp_path, "test")
    query = torch.empty(2, 4, 5, 3)
    positions = torch.arange(5)
    mask = (positions[:, None] >= positions[None, :]).view(1, 1, 5, 5)
    mask = mask.expand(2, -1, -1, -1)
    controller._validate_attention_mask(mask, query)
    controller._validate_attention_mask(mask, query)
    assert controller.attention_mask_calls == 2
    assert controller.attention_mask_contract is not None
    assert controller.attention_mask_contract["true_values"] == 30
    bad = mask.clone()
    bad[0, 0, 0, 4] = True
    other = capture_driver.CaptureController(tmp_path, "test-bad")
    with pytest.raises(common.ReplayDiagnosticError, match="lower-triangular"):
        other._validate_attention_mask(bad, query)


def test_slice_permute_repeat_and_kernel_reconstruction_preserve_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(common, "SEQUENCE_LENGTH", 3)
    monkeypatch.setattr(common, "QUERY_HEADS", 4)
    monkeypatch.setattr(common, "KEY_VALUE_HEADS", 2)
    monkeypatch.setattr(common, "HEAD_DIM", 2)
    archive = {
        "q": torch.arange(4 * 4 * 3 * 2, dtype=torch.float32)
        .reshape(4, 4, 3, 2)
        .to(torch.bfloat16),
        "k": torch.arange(4 * 2 * 3 * 2, dtype=torch.float32)
        .reshape(4, 2, 3, 2)
        .to(torch.bfloat16),
        "v": torch.arange(4 * 2 * 3 * 2, dtype=torch.float32)
        .reshape(4, 2, 3, 2)
        .to(torch.bfloat16),
        "dout": torch.arange(4 * 3 * 4 * 2, dtype=torch.float32)
        .reshape(4, 3, 4, 2)
        .to(torch.bfloat16),
    }
    definitions = {item["name"]: item for item in common.case_definitions()}
    permuted = common.slice_case(archive, definitions["real_b4_perm_s3_to_s0"])
    repeated = common.slice_case(archive, definitions["real_b4_s3_repeated"])
    torch.testing.assert_close(permuted["q"][0], archive["q"][3])
    torch.testing.assert_close(repeated["q"][0], archive["q"][3])
    torch.testing.assert_close(repeated["q"][3], archive["q"][3])

    kernel = common.reconstruct_kernel_qkvd(permuted, batch=4)
    assert kernel["q"].stride() == (24, 2, 8, 1)
    assert not kernel["q"].is_contiguous()
    assert kernel["q"].transpose(1, 2).is_contiguous()
    assert kernel["dout"].is_contiguous()


def test_frozen_token_serialization_and_semantics() -> None:
    assert common.file_sha256(common.TOKEN_PATH) == common.TOKEN_SERIALIZATION_SHA256
    tokens = torch.load(common.TOKEN_PATH, map_location="cpu", weights_only=True)
    assert isinstance(tokens, list)
    assert len(tokens) == 256
    assert {tuple(item.shape) for item in tokens} == {(2048,)}
    assert {item.dtype for item in tokens} == {torch.int64}
    assert common.tensors_semantic_sha256(tokens) == common.TOKEN_SEMANTIC_SHA256


def test_explicit_fp32_reference_matches_cpu_sdpa_forward_backward() -> None:
    generator = torch.Generator(device="cpu").manual_seed(123)
    q = torch.randn(2, 4, 7, 5, generator=generator, dtype=torch.float32)
    k = torch.randn(2, 2, 7, 5, generator=generator, dtype=torch.float32)
    v = torch.randn(2, 2, 7, 5, generator=generator, dtype=torch.float32)
    dout = torch.randn(2, 7, 4, 5, generator=generator, dtype=torch.float32)
    scale = 5**-0.5

    explicit_inputs = [item.clone().requires_grad_(True) for item in (q, k, v)]
    explicit_output = common.explicit_fp32_attention(
        *explicit_inputs, scaling=scale
    )
    explicit_gradients = torch.autograd.grad(
        explicit_output, explicit_inputs, grad_outputs=dout
    )

    sdpa_inputs = [item.clone().requires_grad_(True) for item in (q, k, v)]
    sdpa_output = F.scaled_dot_product_attention(
        *sdpa_inputs,
        dropout_p=0.0,
        is_causal=True,
        scale=scale,
        enable_gqa=True,
    ).transpose(1, 2).contiguous()
    sdpa_gradients = torch.autograd.grad(sdpa_output, sdpa_inputs, grad_outputs=dout)

    torch.testing.assert_close(explicit_output, sdpa_output, rtol=2e-6, atol=2e-6)
    for explicit, sdpa in zip(explicit_gradients, sdpa_gradients, strict=True):
        torch.testing.assert_close(explicit, sdpa, rtol=3e-6, atol=3e-6)


def test_full_tensor_metrics_uses_every_value() -> None:
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    right = torch.tensor([[1.0, 1.0], [5.0, 4.0]], dtype=torch.float64)
    metrics = common.full_tensor_metrics(left, right, chunk_elements=2)
    assert metrics["all_values"] is True
    assert metrics["values"] == 4
    assert metrics["unequal_values_after_fp64_cast"] == 2
    assert metrics["difference_rms"] == pytest.approx(math.sqrt(5 / 4))
    assert metrics["max_abs_difference"] == 2.0


def test_causal_case_metrics_are_direct_and_sample_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(common, "OUTPUT_FIELDS", ("output", "dq", "dk", "dv"))
    base = {
        field: torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
        for field in common.OUTPUT_FIELDS
    }
    retained: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    permutation = torch.tensor([3, 0, 1, 2], dtype=torch.int64)
    for name in ("fp32_reference", *common.ARMS):
        arm_b4 = {field: value.clone() for field, value in base.items()}
        if name == "fa4_default":
            for value in arm_b4.values():
                value[3].add_(1.0)
        retained[name] = {
            "real_b4": arm_b4,
            "real_b1_s3": {
                field: arm_b4[field][3:4].clone()
                for field in common.OUTPUT_FIELDS
            },
            "real_b4_perm_s3_to_s0": {
                field: arm_b4[field].index_select(0, permutation)
                for field in common.OUTPUT_FIELDS
            },
            "real_b4_s3_repeated": {
                field: arm_b4[field][3:4].repeat(4, 1)
                for field in common.OUTPUT_FIELDS
            },
        }

    metrics = compare_worker._causal_case_metrics(retained)
    for arm_metrics in metrics["permutation_equivariance"].values():
        assert all(
            field_metrics["max_abs_difference"] == 0.0
            for field_metrics in arm_metrics["fields"].values()
        )
    for arm_metrics in metrics["repeated_s3_vs_independent_b1_s3"].values():
        for slot in arm_metrics["slots"].values():
            assert all(
                field_metrics["max_abs_difference"] == 0.0
                for field_metrics in slot["fields"].values()
            )
    fa4_samples = metrics["real_b4_per_sample_vs_fp32"]["fa4_default"][
        "samples"
    ]
    assert fa4_samples["0"]["fields"]["dq"]["max_abs_difference"] == 0.0
    assert fa4_samples["3"]["fields"]["dq"]["max_abs_difference"] == 1.0


def test_source_command_is_q4_b4_math_capture() -> None:
    command = runner._capture_command()
    flags = runner._flags(command)
    assert command[2].endswith("true_tensor_replay_20260822.capture_driver")
    assert flags["--model"] == str(common.MODEL_PATH)
    assert flags["--global_loss_bsz"] == "4"
    assert flags["--nsamples"] == "256"
    assert flags["--seq_len"] == "2048"
    assert flags["--attention_backend"] == "sdpa"
    assert flags["--grad_hessian_topk"] == "-1"
    assert flags["--rotate"] == "true"
    assert flags["--exit_after_precompute"] == "true"
    assert flags["--static_cache_path"] == ""


def test_fa4_2cta_arms_are_process_and_environment_isolated() -> None:
    commands = {arm: runner._worker_command(arm) for arm in common.ARMS}
    assert len({tuple(command) for command in commands.values()}) == 3
    assert commands["fa4_default"][-1] == "fa4_default"
    assert commands["fa4_no2cta"][-1] == "fa4_no2cta"
    assert common.PAIRWISE_COMPARISONS == (
        ("fp32_reference", "math_sdpa"),
        ("fp32_reference", "fa4_default"),
        ("fp32_reference", "fa4_no2cta"),
        ("math_sdpa", "fa4_default"),
        ("math_sdpa", "fa4_no2cta"),
        ("fa4_default", "fa4_no2cta"),
    )
    plan = runner.build_plan(7, verify_model_hashes=False)
    assert plan["replay"]["arms"]["fa4_default"]["expected_forward_2cta_on_cuda12"] is False
    assert plan["replay"]["arms"]["fa4_default"]["expected_backward_2cta_on_sm100_hd128"] is True
    assert plan["replay"]["arms"]["fa4_no2cta"]["expected_forward_2cta_on_cuda12"] is False
    assert plan["replay"]["arms"]["fa4_no2cta"]["expected_backward_2cta_on_sm100_hd128"] is False
    assert plan["resource_contract"]["raw_outcome_bytes_per_arm"] == (
        common.expected_capture_bytes_per_kind() * 5
    )
    assert set(plan["replay"]["causal_metrics"]) == {
        "permutation",
        "duplicate",
        "sample_localization",
    }
    snapshot_paths = {
        item["path"] for item in plan["source_snapshot"]["files"]
    }
    assert (
        "experiments/realq_allopts_loss_ablation_20260821/cache_backend_diagnostic.py"
        in snapshot_paths
    )


def test_third_party_snapshot_resolves_environment_without_symlink_target() -> None:
    _, command = runner._source_command()
    snapshot = runner._third_party_snapshot(command[0])
    assert len(snapshot["files"]) == len(runner.THIRD_PARTY_RELATIVE_FILES)
    assert all(item["sha256"] for item in snapshot["files"])
