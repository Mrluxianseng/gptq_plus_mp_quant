from __future__ import annotations

from experiments.realq_allopts_loss_ablation_20260821 import runner as prior
from experiments.realq_sdpa_fastpath_isolation_20260822 import continue_eval_lane
from experiments.realq_sdpa_fastpath_isolation_20260822 import runner


def test_fast_on_changes_only_backend_and_artifact_paths() -> None:
    row = runner._source_row()
    command = runner._fast_on_quant_command(row)
    flags = prior._flags(command)
    assert flags["--attention_backend"] == "sdpa"
    assert flags["--static_cache_path"] == str(
        prior.LEGACY_CACHE_ROOT / runner.MODEL / "static"
    )
    assert flags["--hessian_tf32"] == "true"
    for flag, expected in prior.ALL_ON_FLAGS.items():
        if flag != "--attention_backend":
            assert flags[flag] == expected
    runner._validate_commands(row, command, runner._canonical_command(row))


def test_fast_on_environment_is_deterministic_sdpa_without_global_tf32_veto() -> None:
    environment = runner._fast_on_environment(3)
    assert environment["CUDA_VISIBLE_DEVICES"] == "3"
    assert environment["REALQ_DETERMINISTIC_SDPA"] == "1"
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert "NVIDIA_TF32_OVERRIDE" not in environment


def test_canonical_reuses_prior_reference_and_loads_new_checkpoint() -> None:
    flags = prior._flags(runner._canonical_command(runner._source_row()))
    assert flags["--attention_backend"] == "sdpa"
    assert flags["--hessian_tf32"] == "false"
    assert flags["--load_qmodel_path"] == str(runner.CHECKPOINT)
    assert flags["--require_reference_cache_hit"] == "true"
    assert flags["--cache_dir"] == str(prior.CANONICAL_CACHE_ROOT / runner.MODEL)


def test_restored_eval_lane_includes_audited_evalplus_runtime() -> None:
    environment = continue_eval_lane._evaluation_environment()
    paths = environment["PYTHONPATH"].split(":")
    assert str(continue_eval_lane.EVALPLUS_RUNTIME) in paths
    assert str(runner.REPO_ROOT) in paths
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert environment["PYTHONHASHSEED"] == "1234"
