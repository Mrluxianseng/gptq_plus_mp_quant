from pathlib import Path

from experiments.realq_fullmodel_retune_20260817 import campaign as base
from experiments.realq_sdpa_frozen_lr20_20260822 import eager_reasoning_nonq32 as eager
from experiments.realq_sdpa_frozen_lr20_20260822 import formal_merged_v5 as formal
from experiments.realq_sdpa_frozen_lr20_20260822 import reasoning_v5_merged as reasoning


def test_eager_reasoning_matrix_is_exact_nonq32_subset():
    pairs = eager._pairs()
    works = eager._works()
    assert len(pairs) == len(set(pairs)) == 31
    assert len(works) == len(set(works)) == 93
    assert eager.EXCLUDED_RUNNING_ROWS.isdisjoint(pairs)
    assert all("qwen3-32b" not in config for _, config in pairs)
    assert {
        task for _, _, task in works
    } == set(reasoning.TASKS)


def test_eager_reasoning_plan_binds_commands_tokens_and_datasets():
    plan = eager._build_plan()
    assert plan["protocol"]["generation_works"] == 93
    assert plan["protocol"]["exact_gptaq_guidedquant_token_files"] is True
    assert set(plan["datasets"]) == set(reasoning.TASKS)
    assert len(plan["rows"]) == 93
    for row in plan["rows"]:
        flags = formal._flags(row["command"])
        model = eager.campaign._model_for_config(row["config"])
        contract = plan["cache_contracts"][model]
        assert flags["--tokens_cache_path"] == str(
            Path(contract["token"]["path"]).parent
        )
        assert flags["--cache_dir"] == contract["reference_runtime_root"]
        assert flags["--reasoning_output_dir"] == str(
            reasoning._output_dir(row["branch"], row["config"], row["task"])
        )
        assert flags["--attention_backend"] == "sdpa"
        assert flags["--reasoning_seed"] == "1234"
        assert flags["--reasoning_do_sample"] == "false"
        assert base._canonical_sha256(row["command"]) == row["command_sha256"]


def test_only_pipeline_output_is_normalized_for_adoption():
    plan = eager._build_plan()
    row = plan["rows"][0]
    command = list(row["command"])
    base._set_arg(command, "--output_dir", "/different/eager/pipeline-output")
    assert reasoning._normalized_command(command) == reasoning._normalized_command(
        row["command"]
    )
    assert formal._flags(command)["--reasoning_output_dir"] == formal._flags(
        row["command"]
    )["--reasoning_output_dir"]
