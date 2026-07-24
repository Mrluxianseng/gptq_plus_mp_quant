from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tools.compare_performance_checkpoints import (  # noqa: E402
    build_state_manifest,
    compare_checkpoints,
)


def _save(path: Path, state: dict[str, torch.Tensor], **metadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "realq.quantized_model",
            "format_version": 1,
            "base_model": "fixture/model",
            "runtime_quantization": {"a_bits": 16},
            "weight_quantization": {"w_bits": 2},
            "model": state,
            **metadata,
        },
        path,
    )


def _run_root(
    root: Path,
    *,
    value: bool,
    label: str,
    candidate_specs: list[str],
) -> Path:
    checkpoint = root / "checkpoint" / "model.pt"
    _save(
        checkpoint,
        {
            "layer.weight": torch.tensor([[1.0, -2.0]], dtype=torch.float32),
            "scalar": torch.tensor(3, dtype=torch.int64),
        },
        weight_quantization={
            "w_bits": 2,
            "quantizer_inner_fastpath": value,
        },
    )
    manifest = {
        "status": "passed",
        "baseline_id": "fixture-ab",
        "label": label,
        "case": "group_stress",
        "git_commit": "a" * 40,
        "world_size": 4,
        "model_artifact_identity": "model-identity-a",
        "harness_sha256": "harness-a",
        "checkpoint_compare_tool_sha256": "comparator-a",
        "source_cache_identity": [
            {
                "kind": "static",
                "rank": 0,
                "size_bytes": 123,
                "mtime_ns": 456,
                "mode": 0o444,
                "sha256": "cache-a",
            }
        ],
        "physical_gpu_ids": [4, 5, 6, 7],
        "physical_gpu_uuids_in_rank_order": [
            "GPU-a",
            "GPU-b",
            "GPU-c",
            "GPU-d",
        ],
        "candidate_arg_specs": candidate_specs,
    }
    config = {
        "model": "fixture/model",
        "w_bits": 2,
        "quantizer_inner_fastpath": value,
        "cache_dir": str(root / "runtime_cache"),
        "output_dir": str(root / "program_output"),
        "save_qmodel_path": str(checkpoint),
        "static_cache_path": str(root / "static"),
        "tokens_cache_path": str(root / "tokens"),
        "exp": label,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (root / "resolved_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    return root


def test_manifest_canonicalizes_only_raw_byte_identical_wrapper_aliases(
    tmp_path,
):
    path = tmp_path / "alias.pt"
    value = torch.tensor([0.0, float("nan")], dtype=torch.float32)
    _save(
        path,
        {
            "layer.weight": value.clone(),
            "layer.module.weight": value.clone(),
            "layer.bias": torch.tensor(1.0),
        },
    )
    manifest = build_state_manifest(path)
    assert manifest["canonical_tensor_count"] == 2
    assert set(manifest["tensors"]) == {"layer.weight", "layer.bias"}
    assert len(manifest["canonical_state_sha256"]) == 64


def test_manifest_rejects_value_equal_but_byte_different_aliases(tmp_path):
    path = tmp_path / "signed-zero-alias.pt"
    _save(
        path,
        {
            "layer.weight": torch.tensor([0.0]),
            "layer.module.weight": torch.tensor([-0.0]),
        },
    )
    with pytest.raises(
        RuntimeError, match="canonicalization collision with unequal values"
    ):
        build_state_manifest(path)


def test_manifest_preserves_unique_legitimate_module_key(tmp_path):
    path = tmp_path / "unique-module.pt"
    _save(
        path,
        {
            "encoder.module.weight": torch.tensor([1.0]),
            "encoder.bias": torch.tensor([2.0]),
        },
    )
    manifest = build_state_manifest(path)
    assert set(manifest["tensors"]) == {
        "encoder.module.weight",
        "encoder.bias",
    }


def test_bare_checkpoint_comparison_is_strict_per_tensor_bytes(tmp_path):
    left = tmp_path / "left.pt"
    right = tmp_path / "right.pt"
    _save(left, {"weight": torch.tensor([1.0, 2.0])})
    _save(right, {"weight": torch.tensor([1.0, 2.0])})

    equal = compare_checkpoints(left, right)
    assert equal["passed"] is True
    assert equal["all_tensor_bytes_equal"] is True
    assert equal["run_compatibility"]["available"] is False

    _save(right, {"weight": torch.tensor([1.0, 2.5])})
    unequal = compare_checkpoints(left, right)
    assert unequal["passed"] is False
    assert unequal["all_tensor_bytes_equal"] is False
    assert unequal["mismatched_tensors"][0]["key"] == "weight"
    assert "sha256" in unequal["mismatched_tensors"][0]["differences"]


def test_run_root_comparison_gates_provenance_and_exact_declared_diff(
    tmp_path,
):
    baseline = _run_root(
        tmp_path / "baseline",
        value=False,
        label="legacy",
        candidate_specs=[],
    )
    candidate = _run_root(
        tmp_path / "candidate",
        value=True,
        label="p01",
        candidate_specs=["quantizer_inner_fastpath=true"],
    )
    report = compare_checkpoints(baseline, candidate)
    assert report["all_tensor_bytes_equal"] is True
    assert report["checkpoint_metadata_equal"] is False
    assert report["run_compatibility"]["available"] is True
    assert report["run_compatibility"]["passed"] is True
    assert report["passed"] is True

    config_path = candidate / "resolved_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["w_bits"] = 3
    config_path.write_text(json.dumps(config), encoding="utf-8")
    incompatible = compare_checkpoints(baseline, candidate)
    assert incompatible["all_tensor_bytes_equal"] is True
    assert incompatible["run_compatibility"]["passed"] is False
    assert incompatible["passed"] is False


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("world_size", 2),
        ("model_artifact_identity", "model-identity-b"),
        ("harness_sha256", "harness-b"),
        ("checkpoint_compare_tool_sha256", "comparator-b"),
        (
            "source_cache_identity",
            [
                {
                    "kind": "static",
                    "rank": 0,
                    "size_bytes": 123,
                    "mtime_ns": 456,
                    "mode": 0o444,
                    "sha256": "cache-b",
                }
            ],
        ),
    ],
)
def test_run_root_comparison_rejects_cross_arm_provenance_drift(
    tmp_path, field, replacement
):
    baseline = _run_root(
        tmp_path / "baseline",
        value=False,
        label="legacy",
        candidate_specs=[],
    )
    candidate = _run_root(
        tmp_path / "candidate",
        value=True,
        label="p01",
        candidate_specs=["quantizer_inner_fastpath=true"],
    )
    manifest_path = candidate / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = replacement
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = compare_checkpoints(baseline, candidate)
    check_name = f"{field}_equal"
    assert report["all_tensor_bytes_equal"] is True
    assert report["run_compatibility"]["checks"][check_name] is False
    assert report["run_compatibility"]["passed"] is False
    assert report["passed"] is False


@pytest.mark.parametrize(
    "field",
    [
        "world_size",
        "model_artifact_identity",
        "harness_sha256",
        "checkpoint_compare_tool_sha256",
        "source_cache_identity",
    ],
)
def test_run_root_comparison_rejects_missing_identity_on_both_arms(
    tmp_path, field
):
    baseline = _run_root(
        tmp_path / "baseline",
        value=False,
        label="legacy",
        candidate_specs=[],
    )
    candidate = _run_root(
        tmp_path / "candidate",
        value=True,
        label="p01",
        candidate_specs=["quantizer_inner_fastpath=true"],
    )
    for root in (baseline, candidate):
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.pop(field)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = compare_checkpoints(baseline, candidate)
    assert report["all_tensor_bytes_equal"] is True
    assert report["run_compatibility"]["checks"][f"{field}_equal"] is False
    assert report["run_compatibility"]["passed"] is False
    assert report["passed"] is False
