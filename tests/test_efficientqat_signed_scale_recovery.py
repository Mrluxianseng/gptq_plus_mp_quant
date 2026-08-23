from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from experiments.efficientqat_compare import materialize
from experiments.efficientqat_compare import signed_scale_recovery as recovery


def _pack_dim0(values: torch.Tensor, bits: int) -> torch.Tensor:
    lanes = 32 // bits
    words = torch.zeros(
        (math.ceil(values.shape[0] / lanes), values.shape[1]),
        dtype=torch.int64,
    )
    for row in range(values.shape[0]):
        words[row // lanes] |= (
            values[row].to(torch.int64) << ((row % lanes) * bits)
        )
    return (words & 0xFFFFFFFF).to(torch.int32)


def _pack_dim1(values: torch.Tensor, bits: int) -> torch.Tensor:
    lanes = 32 // bits
    words = torch.zeros(
        (values.shape[0], math.ceil(values.shape[1] / lanes)),
        dtype=torch.int64,
    )
    for column in range(values.shape[1]):
        words[:, column // lanes] |= (
            values[:, column].to(torch.int64)
            << ((column % lanes) * bits)
        )
    return (words & 0xFFFFFFFF).to(torch.int32)


class _PackedLinear(nn.Module):
    def __init__(
        self,
        codes: torch.Tensor,
        zeros: torch.Tensor,
        scales: torch.Tensor,
        *,
        bits: int = 4,
        group_size: int = 2,
    ) -> None:
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.infeatures = codes.shape[0]
        self.outfeatures = codes.shape[1]
        self.register_buffer("qweight", _pack_dim0(codes, bits))
        self.register_buffer("qzeros", _pack_dim1(zeros, bits))
        self.register_parameter(
            "scales", nn.Parameter(scales.clone(), requires_grad=False)
        )
        self.register_buffer(
            "g_idx",
            torch.arange(self.infeatures, dtype=torch.int32) // group_size,
        )
        self.bias = None


class _Model(nn.Module):
    def __init__(self, packed: nn.Module) -> None:
        super().__init__()
        self.q_proj = packed


def _packed_with_scales(scales: torch.Tensor) -> tuple[_Model, torch.Tensor]:
    codes = torch.tensor(
        [[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.int64
    )
    zeros = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    model = _Model(_PackedLinear(codes, zeros, scales))
    expected = (
        (
            codes.reshape(2, 2, 2).float()
            - zeros[:, None, :].float()
        )
        * scales[:, None, :]
    ).reshape(4, 2).T.contiguous().to(torch.bfloat16)
    return model, expected


def test_signed_and_zero_scales_decode_without_transform():
    scales = torch.tensor([[-0.5, 0.0], [0.25, -0.125]])
    model, expected = _packed_with_scales(scales)

    with recovery.signed_scale_validator_override():
        report = materialize.materialize_efficientqat_model(
            model,
            expected_count=1,
            expected_bits=4,
            expected_group_size=2,
            expected_module_names=["q_proj"],
        )

    assert report["module_count"] == 1
    assert type(model.q_proj) is nn.Linear
    assert torch.equal(model.q_proj.weight, expected)
    # Zero scales must produce exactly zero decoded columns/elements rather
    # than an epsilon replacement.
    zero_scale_group = model.q_proj.weight.detach()[1, :2]
    assert torch.equal(zero_scale_group, torch.zeros_like(zero_scale_group))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_signed_validator_still_rejects_nonfinite_scales(bad):
    scales = torch.tensor([[bad, 0.0], [0.25, -0.125]])
    model, _ = _packed_with_scales(scales)

    with recovery.signed_scale_validator_override():
        with pytest.raises(ValueError, match="non-finite"):
            materialize.decode_efficientqat_weight_cpu(
                model.q_proj,
                expected_bits=4,
                expected_group_size=2,
                name="q_proj",
            )


def test_original_validator_identity_and_source_are_unchanged():
    original = materialize._validate_packed_module
    source = Path(materialize.__file__).resolve()
    digest_before = hashlib.sha256(source.read_bytes()).hexdigest()
    model, _ = _packed_with_scales(
        torch.tensor([[0.0, 0.25], [0.25, 0.25]])
    )

    with pytest.raises(ValueError, match="non-positive"):
        materialize.decode_efficientqat_weight_cpu(
            model.q_proj,
            expected_bits=4,
            expected_group_size=2,
            name="q_proj",
        )
    with recovery.signed_scale_validator_override():
        assert materialize._validate_packed_module is (
            recovery.validate_finite_signed_packed_module
        )
    assert materialize._validate_packed_module is original
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest_before
    with pytest.raises(ValueError, match="non-positive"):
        materialize.decode_efficientqat_weight_cpu(
            model.q_proj,
            expected_bits=4,
            expected_group_size=2,
            name="q_proj",
        )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _eligible_contract(tmp_path: Path):
    output_root = tmp_path / "formal"
    run = {
        "run_id": "EQAT-test",
        "output_subdir": "model/setting",
        "evaluation_gpu": 6,
        "w_bits": 4,
    }
    plan_sha = "a" * 64
    plan = {
        "runtime": {"output_root": str(output_root)},
        "calibration_contract": {"num_samples": 2048},
        "method_contract": {
            "e2e_qp": {
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 8,
                "epochs": 1,
            }
        },
    }
    run_dir = output_root / run["output_subdir"]
    block_output = run_dir / "block_ap" / "packed_model"
    e2e_output = run_dir / "e2e_qp" / "packed_model"
    block_output.mkdir(parents=True)
    e2e_output.mkdir(parents=True)
    _write_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": 1,
            "status": "running",
            "run_id": run["run_id"],
            "run": run,
            "plan_sha256": plan_sha,
            "started_at": "2026-07-25T00:00:00+00:00",
        },
    )
    _write_json(
        run_dir / "launcher.json",
        {
            "schema_version": 1,
            "run_id": run["run_id"],
            "plan_sha256": plan_sha,
        },
    )
    _write_json(
        run_dir / "block_ap" / "stage.json",
        {
            "schema_version": 1,
            "stage": "block_ap",
            "status": "succeeded",
            "started_at": "2026-07-25T00:00:01+00:00",
            "ended_at": "2026-07-25T00:01:00+00:00",
            "wall_seconds": 59.0,
            "gpu_seconds": 59.0,
            "output": str(block_output),
        },
    )
    _write_json(
        run_dir / "e2e_qp" / "stage.json",
        {
            "schema_version": 1,
            "stage": "e2e_qp",
            "status": "succeeded",
            "started_at": "2026-07-25T00:01:00+00:00",
            "ended_at": "2026-07-25T00:02:00+00:00",
            "wall_seconds": 60.0,
            "gpu_seconds": 60.0,
            "optimizer_steps": 64,
            "output": str(e2e_output),
        },
    )
    error = (
        "model.layers.0.self_attn.q_proj.scales contains a non-positive "
        "learned quantization step."
    )
    _write_json(
        run_dir / "failure.json",
        {
            "schema_version": 1,
            "status": "failed",
            "error_type": "ValueError",
            "error": error,
            "traceback": f"_validate_packed_module\nValueError: {error}",
            "ended_at": "2026-07-25T00:02:01+00:00",
        },
    )
    return plan, run, plan_sha, run_dir


def test_original_contract_rejects_non_positivity_unrelated_failure(tmp_path):
    plan, run, plan_sha, run_dir = _eligible_contract(tmp_path)
    failure_path = run_dir / "failure.json"
    failure = json.loads(failure_path.read_text())
    failure["error"] = "CUDA out of memory"
    failure["traceback"] = "_validate_packed_module\nCUDA out of memory"
    _write_json(failure_path, failure)

    with pytest.raises(recovery.RecoveryError, match="positivity gate"):
        recovery.validate_original_run_contract(
            plan=plan,
            run=run,
            plan_sha256=plan_sha,
            original_run_dir=run_dir,
        )


def test_original_contract_rejects_wrong_optimizer_steps(tmp_path):
    plan, run, plan_sha, run_dir = _eligible_contract(tmp_path)
    stage_path = run_dir / "e2e_qp" / "stage.json"
    stage = json.loads(stage_path.read_text())
    stage["optimizer_steps"] = 63
    _write_json(stage_path, stage)

    with pytest.raises(recovery.RecoveryError, match="64 optimizer steps"):
        recovery.validate_original_run_contract(
            plan=plan,
            run=run,
            plan_sha256=plan_sha,
            original_run_dir=run_dir,
        )


def test_checkpoint_manifest_detects_any_source_change(tmp_path):
    checkpoint = tmp_path / "packed"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "tokenizer.json").write_text("{}")
    (checkpoint / "tokenizer_config.json").write_text("{}")
    manifest = recovery.build_checkpoint_manifest(checkpoint)

    (checkpoint / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(recovery.RecoveryError, match="changed"):
        recovery.assert_checkpoint_unchanged(manifest, checkpoint)


def test_safe_default_is_dry_validation_and_never_executes(
    tmp_path, monkeypatch
):
    recovery_dir = tmp_path / "recovery" / recovery.RECOVERY_VARIANT
    preflight = SimpleNamespace(
        recovery_dir=recovery_dir,
        evidence={
            "schema_version": 1,
            "format": recovery.RECOVERY_FORMAT,
            "format_version": recovery.RECOVERY_VERSION,
        },
    )
    monkeypatch.setattr(
        recovery, "build_recovery_preflight", lambda args: preflight
    )

    def forbidden(_):
        raise AssertionError("execute_recovery must not run in dry mode")

    monkeypatch.setattr(recovery, "execute_recovery", forbidden)
    args = SimpleNamespace(execute=False)
    result = recovery.run_from_args(args)

    assert result["mode"] == "dry_validate"
    assert result["executed"] is False
    assert result["status"] == "succeeded"
    assert not recovery_dir.exists()


@pytest.mark.parametrize("relation", ["inside", "parent"])
def test_recovery_root_must_be_disjoint_from_formal_output(
    tmp_path, relation
):
    formal = tmp_path / "formal"
    original = formal / "model" / "setting"
    original.mkdir(parents=True)
    if relation == "inside":
        recovery_root = formal / "recovery"
    else:
        recovery_root = tmp_path

    with pytest.raises(recovery.RecoveryError, match="must be disjoint"):
        recovery._resolve_recovery_dir(
            recovery_root=recovery_root,
            original_output_root=formal,
            original_run_dir=original,
            output_subdir="model/setting",
        )


def test_frozen_offline_environment_is_installed_exactly(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    plan = {
        "runtime": {
            "offline_environment": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        }
    }

    installed = recovery.apply_frozen_offline_environment(plan)

    assert installed == plan["runtime"]["offline_environment"]
    assert recovery.os.environ["HF_HUB_OFFLINE"] == "1"
    assert recovery.os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_execution_boundary_rehashes_all_frozen_inputs(monkeypatch):
    calls = []
    plan_path = Path("/frozen/plan.json")
    checkpoint_path = Path("/frozen/packed")
    expected_plan_sha = "a" * 64
    expected_script_sha = "b" * 64
    expected_materialize_sha = "c" * 64
    expected_run_one_sha = "d" * 64
    offline = {"HF_HUB_OFFLINE": "1"}
    original = {"contract": "unchanged"}
    checkpoint = {"files_sha256": "checkpoint-sha"}
    preflight = SimpleNamespace(
        plan_path=plan_path,
        plan={"runtime": {"offline_environment": offline}},
        run={"run_id": "EQAT-test"},
        original_run_dir=Path("/frozen/original"),
        packed_checkpoint=checkpoint_path,
        script_sha256=expected_script_sha,
        checkpoint_manifest=checkpoint,
        evidence={
            "offline_environment": offline,
            "original": original,
            "identity": {
                "original_plan_sha256": expected_plan_sha,
                "original_materialize_sha256": expected_materialize_sha,
                "original_run_one_sha256": expected_run_one_sha,
            },
        },
    )

    monkeypatch.setattr(
        recovery.launcher,
        "plan_sha256",
        lambda path: expected_plan_sha,
    )
    monkeypatch.setattr(
        recovery,
        "apply_frozen_offline_environment",
        lambda plan: dict(offline),
    )
    monkeypatch.setattr(
        recovery.launcher,
        "verify_artifacts",
        lambda plan: calls.append("artifacts"),
    )
    monkeypatch.setattr(
        recovery.launcher,
        "verify_reference_cache_payloads",
        lambda plan: calls.append("reference_payloads"),
    )
    monkeypatch.setattr(
        recovery,
        "validate_original_run_contract",
        lambda **kwargs: dict(original),
    )
    monkeypatch.setattr(
        recovery,
        "assert_checkpoint_unchanged",
        lambda manifest, path: calls.append("checkpoint") or dict(checkpoint),
    )

    def fake_sha(path):
        resolved = Path(path).resolve()
        if resolved == Path(recovery.__file__).resolve():
            return expected_script_sha
        if resolved == Path(materialize.__file__).resolve():
            return expected_materialize_sha
        if resolved == Path(recovery.run_one.__file__).resolve():
            return expected_run_one_sha
        raise AssertionError(f"unexpected SHA request: {resolved}")

    monkeypatch.setattr(recovery, "_sha256_file", fake_sha)

    report = recovery.assert_preflight_inputs_unchanged(
        preflight,
        verify_reference_payloads=True,
        phase="unit_boundary",
    )

    assert calls == ["artifacts", "reference_payloads", "checkpoint"]
    assert report["phase"] == "unit_boundary"
    assert report["reference_cache_payloads_rehashed"] is True
    assert report["checkpoint_files_sha256"] == "checkpoint-sha"


def test_execute_uses_frozen_materializer_and_evaluator(
    tmp_path, monkeypatch
):
    recovery_dir = tmp_path / "recovery" / recovery.RECOVERY_VARIANT
    original_validator = materialize._validate_packed_module
    observed = []
    preflight = SimpleNamespace(
        plan={"runtime": {"output_root": str(tmp_path / "formal")}},
        run={"run_id": "EQAT-test"},
        recovery_dir=recovery_dir,
        packed_checkpoint=tmp_path / "packed",
        checkpoint_manifest={"files_sha256": "checkpoint-sha"},
        physical_eval_gpu=6,
        evidence={
            "schema_version": 1,
            "format": recovery.RECOVERY_FORMAT,
            "format_version": recovery.RECOVERY_VERSION,
            "identity": {"test": "identity"},
            "preflight_timing": {
                "started_at": "2026-07-25T00:00:00+00:00",
                "ended_at": "2026-07-25T00:00:01+00:00",
                "wall_seconds": 1.0,
            },
            "scale_statistics": {
                "negative_count": 1,
                "zero_count": 0,
            },
            "original": {
                "stages": {"e2e_qp": {"status": "succeeded"}},
                "failure": {"error_type": "ValueError"},
            },
        },
    )

    class Logger:
        def info(self, *args, **kwargs):
            return None

    monkeypatch.setattr(recovery, "_assert_execute_runtime", lambda _: None)
    monkeypatch.setattr(
        recovery,
        "assert_preflight_inputs_unchanged",
        lambda preflight, **kwargs: {
            "phase": kwargs["phase"],
            "reference_cache_payloads_rehashed": kwargs[
                "verify_reference_payloads"
            ],
        },
    )
    monkeypatch.setattr(
        recovery,
        "assert_checkpoint_unchanged",
        lambda manifest, path: dict(manifest),
    )
    monkeypatch.setattr(
        recovery.run_one,
        "_configure_logging",
        lambda path: Logger(),
    )

    def fake_materialize(plan, run, run_dir, checkpoint):
        observed.append(
            (
                "materialize",
                materialize._validate_packed_module
                is recovery.validate_finite_signed_packed_module,
            )
        )
        output = run_dir / "materialize" / "hf_model"
        output.mkdir(parents=True)
        return output, {"wall_seconds": 2.0, "status": "succeeded"}

    def fake_evaluation(
        plan,
        run,
        run_dir,
        materialized_dir,
        eval_device,
        physical_eval_gpu,
        logger,
    ):
        observed.append(
            (
                "evaluation",
                eval_device,
                physical_eval_gpu,
                materialized_dir,
            )
        )
        return (
            {"ppl": 1.25},
            {"wall_seconds": 3.0, "status": "succeeded"},
        )

    monkeypatch.setattr(recovery.run_one, "run_materialize", fake_materialize)
    monkeypatch.setattr(recovery.run_one, "run_evaluation", fake_evaluation)
    monkeypatch.setattr(
        recovery,
        "_timing_summary",
        lambda *args, **kwargs: {"effective_wall_seconds": 6.0},
    )

    result = recovery.execute_recovery(preflight)

    assert result["status"] == "succeeded"
    assert result["metrics"] == {"ppl": 1.25}
    assert observed[0] == ("materialize", True)
    assert observed[1][:3] == ("evaluation", "cuda:0", 6)
    assert materialize._validate_packed_module is original_validator
    assert (recovery_dir / "result.json").is_file()
    assert json.loads(
        (recovery_dir / "recovery_manifest.json").read_text()
    )["status"] == "succeeded"
