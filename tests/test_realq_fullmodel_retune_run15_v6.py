from __future__ import annotations

import copy
import os

import pytest

from experiments.realq_fullmodel_retune_20260817 import campaign as base
from experiments.realq_fullmodel_retune_20260817 import campaign_v6_run15 as v6


def _command(*, branch: str, config: str) -> list[str]:
    model = next(
        slug for slug in base.MODEL_SLUGS if config.startswith(f"{slug}_")
    )
    static_root, runtime_root = v6._cache_paths(model)
    values = {
        "--dataset": "wikitext2",
        "--eval_datasets": "wikitext2",
        "--seed": "1",
        "--rotation_seed": "0",
        "--refresh_seed": "0",
        "--nsamples": "256",
        "--seq_len": "2048",
        "--eval_seq_len": "2048",
        "--w_groupsize": "128",
        "--blocksize": "128",
        "--backward_samples": "32",
        "--backward_bsz": "32",
        "--loss_slide_window": "true",
        "--full_block_refresh": base.BRANCH_VALUES[branch],
        "--grad_lr_layer_schedule": base._schedule_for(config),
        "--a_loss_ratio": "1",
        "--hessian_accum_bsz": "32" if model == "qwen3-32b" else "64",
        "--rotate": "true",
        "--static_cache_path": str(static_root),
        "--cache_dir": str(runtime_root),
        "--require_static_cache_hit": "true",
        "--require_reference_cache_hit": "true",
        **v6.RUN15_FLAGS,
    }
    command = ["python", "-m", "realq.ptq"]
    for flag, value in values.items():
        command.extend((flag, value))
    return command


@pytest.mark.parametrize("branch", ["full_block", "single_linear"])
@pytest.mark.parametrize("config", ["qwen3-4b_w4a4kv4", "qwen3-32b_w3a16"])
def test_run15_gate_requires_fa4_tf32_and_q32_only_hessian_exception(
    branch: str, config: str
):
    command = _command(branch=branch, config=config)
    v6._validate_full_profile(command, branch=branch, config=config)

    sdpa = copy.copy(command)
    base._set_arg(sdpa, "--attention_backend", "sdpa")
    with pytest.raises(base.CampaignError, match="attention_backend"):
        v6._validate_full_profile(sdpa, branch=branch, config=config)

    tf32_off = copy.copy(command)
    base._set_arg(tf32_off, "--hessian_tf32", "false")
    with pytest.raises(base.CampaignError, match="hessian_tf32"):
        v6._validate_full_profile(tf32_off, branch=branch, config=config)

    if config.startswith("qwen3-32b_"):
        wrong_hessian_batch = copy.copy(command)
        base._set_arg(wrong_hessian_batch, "--hessian_accum_bsz", "64")
        with pytest.raises(base.CampaignError, match="hessian_accum_bsz=32"):
            v6._validate_full_profile(
                wrong_hessian_batch, branch=branch, config=config
            )


def test_run15_configuration_changes_only_branch_scope_between_methods():
    config = "qwen3-4b_w3a16"
    source = _command(branch="full_block", config=config)
    left = v6._configure_run15_command(
        copy.copy(source), branch="full_block", config=config
    )
    right = v6._configure_run15_command(
        copy.copy(source), branch="single_linear", config=config
    )
    left_flags = dict(zip(left[3::2], left[4::2]))
    right_flags = dict(zip(right[3::2], right[4::2]))
    assert left_flags.pop("--full_block_refresh") == "true"
    assert right_flags.pop("--full_block_refresh") == "false"
    assert left_flags == right_flags


def test_run15_worker_environment_unsets_math_sdpa_and_global_tf32_blocker(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("REALQ_DETERMINISTIC_SDPA", "1")
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    monkeypatch.setattr(base, "WORKER_ENV_OVERRIDES", dict(v6.RUN15_ENV_OVERRIDES))
    monkeypatch.setattr(base, "WORKER_ENV_UNSET", v6.RUN15_ENV_UNSET)
    environment = base._worker_environment("3")
    assert environment["CUDA_VISIBLE_DEVICES"] == "3"
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert "REALQ_DETERMINISTIC_SDPA" not in environment
    assert "NVIDIA_TF32_OVERRIDE" not in environment
    assert os.environ["REALQ_DETERMINISTIC_SDPA"] == "1"
    assert os.environ["NVIDIA_TF32_OVERRIDE"] == "0"
