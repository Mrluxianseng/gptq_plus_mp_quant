from pathlib import Path

import pytest


def test_v2_matrix_lr_and_exact_baseline_cache_contracts():
    from experiments.realq_sdpa_frozen_lr20_20260822 import campaign_v2_memory as c

    plan = c.base._read_json(c.PLAN_PATH)
    selection = c.base._read_json(c.OUTPUT_ROOT / "selections.json")
    assert len(plan["configurations"]) == len(selection["rows"]) == 40
    source = c._source_selection()
    expected_lr = {
        (row["branch"], row["config"]): float(row["selected_lr"])
        for row in source["rows"]
    }
    assert {
        (row["branch"], row["config"]): float(row["selected_lr"])
        for row in selection["rows"]
    } == expected_lr
    baseline = c.v1._baseline_cache_contracts()
    for model, contract in plan["calibration_and_reference_contracts"].items():
        assert contract == baseline[model]
        assert Path(contract["tokens"]["path"]).is_file()
        assert Path(contract["reference_logits"]["path"]).is_file()


def test_v2_commands_have_only_frozen_capacity_delta_and_same_tokens():
    from experiments.realq_sdpa_frozen_lr20_20260822 import campaign_v2_memory as c

    plan = c.base._read_json(c.PLAN_PATH)
    source = c.base._read_json(c.SOURCE_PLAN_PATH)
    for key, row in plan["configurations"].items():
        branch, config = key.split("/", 1)
        model = c._model_for_config(config)
        command = row["source_command"]
        c._validate_full_profile(command, branch=branch, config=config)
        before = c.v1._flags(source["configurations"][key]["source_command"])
        after = c.v1._flags(command)
        changed = {
            flag for flag in set(before) | set(after) if before.get(flag) != after.get(flag)
        }
        expected = {"--static_cache_path"}
        if before["--global_loss_bsz"] != str(c.GLOBAL_LOSS_BSZ[model]):
            expected.add("--global_loss_bsz")
        assert changed == expected
        token = Path(plan["calibration_and_reference_contracts"][model]["tokens"]["path"])
        assert sorted(Path(after["--tokens_cache_path"]).glob("*.pt")) == [token]


def test_formal_plan_is_controller_first_import_order_invariant():
    from experiments.realq_sdpa_frozen_lr20_20260822 import controller_v2

    controller_v2._activate()
    plan = controller_v2.formal._load_plan()
    assert plan["formal_id"] == controller_v2.formal.FORMAL_ID
    assert len(plan["rows"]) == 40


def test_eval_adapters_bind_v2b_formal_and_sdpa_protocol():
    from experiments.realq_sdpa_frozen_lr20_20260822 import formal_v2
    from experiments.realq_sdpa_frozen_lr20_20260822 import quality_v2
    from experiments.realq_sdpa_frozen_lr20_20260822 import reasoning_v2

    quality_v2._activate()
    assert quality_v2.core.formal is formal_v2
    assert quality_v2.core.merged.OUTPUT_ROOT == formal_v2.campaign.OUTPUT_ROOT
    assert quality_v2.QUALITY_PLAN_PATH.name == "quality_plan_v2b.json"
    reasoning_v2._activate()
    assert reasoning_v2.core.formal is formal_v2
    assert reasoning_v2.core.merged.OUTPUT_ROOT == formal_v2.campaign.OUTPUT_ROOT
    assert reasoning_v2.REASONING_PLAN_PATH.name == "reasoning_plan_v2b.json"
    for module in (quality_v2, reasoning_v2):
        snapshot = module._code_snapshot()
        assert len(snapshot["files"]) == 5
        assert snapshot["sha256"]


def test_runtime_cache_audit_binds_all_40_rows_to_baseline_sources():
    from experiments.realq_sdpa_frozen_lr20_20260822 import campaign_v2_memory as c
    from experiments.realq_sdpa_frozen_lr20_20260822 import formal_v2
    from experiments.realq_sdpa_frozen_lr20_20260822 import runtime_cache_audit

    campaign_plan = c.base._read_json(c.PLAN_PATH)
    formal_plan = formal_v2._load_plan()
    rows = runtime_cache_audit._planned_row_audit(campaign_plan, formal_plan)
    assert len(rows) == 40
    assert all(row["token_path"] for row in rows)
    assert all(row["reference_logits_path"] for row in rows)


def test_v3_merged_formal_routes_only_q32_to_haccum16_recovery():
    from experiments.realq_sdpa_frozen_lr20_20260822 import formal_merged_v3 as merged
    from experiments.realq_sdpa_frozen_lr20_20260822 import quality_v3_merged
    from experiments.realq_sdpa_frozen_lr20_20260822 import reasoning_v3_merged

    plan = merged._build_plan()
    assert len(plan["rows"]) == 40
    assert sum(row["source"] == "v3_q32_haccum16" for row in plan["rows"]) == 8
    assert sum(row["source"] == "v2b_non_q32" for row in plan["rows"]) == 32
    for row in plan["rows"]:
        flags = merged._flags(row["command"])
        is_q32 = row["config"].startswith("qwen3-32b_")
        assert flags["--hessian_accum_bsz"] == ("16" if is_q32 else "64")
        assert flags["--attention_backend"] == "sdpa"
        assert flags["--require_static_cache_hit"] == "true"
        assert flags["--require_reference_cache_hit"] == "true"
        assert Path(row["checkpoint"]) == merged._checkpoint_path(
            row["branch"], row["config"]
        )

    quality_v3_merged._activate()
    assert quality_v3_merged.core.formal is merged
    assert quality_v3_merged.QUALITY_PLAN_PATH.name == "quality_plan_v3_merged.json"
    reasoning_v3_merged._activate()
    assert reasoning_v3_merged.core.formal is merged
    assert (
        reasoning_v3_merged.REASONING_PLAN_PATH.name
        == "reasoning_plan_v3_merged.json"
    )


def test_v5_merged_formal_routes_q32_to_exact_tiled_sdpa():
    from experiments.realq_sdpa_frozen_lr20_20260822 import formal_merged_v5 as merged
    from experiments.realq_sdpa_frozen_lr20_20260822 import quality_v5_merged
    from experiments.realq_sdpa_frozen_lr20_20260822 import reasoning_v5_merged

    plan = merged._build_plan()
    assert len(plan["rows"]) == 40
    assert sum(row["source"] == "v5_q32_sdpa_tiled" for row in plan["rows"]) == 8
    assert sum(row["source"] == "v2b_non_q32" for row in plan["rows"]) == 32
    for row in plan["rows"]:
        flags = merged._flags(row["command"])
        is_q32 = row["config"].startswith("qwen3-32b_")
        assert flags["--hessian_accum_bsz"] == ("32" if is_q32 else "64")
        assert flags["--backward_bsz"] == "32"
        assert flags["--attention_backend"] == "sdpa"
        assert flags["--require_static_cache_hit"] == "true"
        assert flags["--require_reference_cache_hit"] == "true"
        if is_q32:
            assert row["command"][2].endswith("q32_memory_entry")

    quality_v5_merged._activate()
    assert quality_v5_merged.core.formal is merged
    assert quality_v5_merged.QUALITY_PLAN_PATH.name == "quality_plan_v5_merged.json"
    reasoning_v5_merged._activate()
    assert reasoning_v5_merged.core.formal is merged
    assert (
        reasoning_v5_merged.REASONING_PLAN_PATH.name
        == "reasoning_plan_v5_merged.json"
    )


def test_v5_quality_commands_bind_exact_baseline_token_and_reference_caches():
    from experiments.realq_sdpa_frozen_lr20_20260822 import campaign_v2_memory as c
    from experiments.realq_sdpa_frozen_lr20_20260822 import formal_merged_v5 as merged
    from experiments.realq_sdpa_frozen_lr20_20260822 import quality_cache_audit_v5_merged
    from experiments.realq_sdpa_frozen_lr20_20260822 import quality_v5_merged

    formal_plan = merged._build_plan()
    contracts = c.v1._baseline_cache_contracts()
    quality_v5_merged._activate()
    for row in formal_plan["rows"]:
        branch, config = row["branch"], row["config"]
        command = quality_v5_merged.core._quality_command(
            formal_plan,
            branch,
            config,
            Path("/quality/attempt/realq_output"),
        )
        model = c._model_for_config(config)
        quality_cache_audit_v5_merged._validate_command(
            command,
            key=f"{branch}/{config}",
            model=model,
            contract=contracts[model],
        )


def test_v5_release_duration_and_phase_parser():
    from experiments.realq_sdpa_frozen_lr20_20260822 import (
        build_release_v5_merged as release,
    )

    text = (
        "Fusing LN: 100%|x| 36/36 [01:13<00:00, 2.0s/it]\r"
        "Fusing LN: 100%|x| 36/36 [01:13<00:00, 2.0s/it]\n"
        "Rotating: 100%|x| 36/36 [01:30<00:00, 2.5s/it]\r"
        "Quantising layers: 100%|x| 36/36 [1:10:53<00:00, 1s/it]\n"
    )
    parsed = release._parse_phase_timing(text, expected_layers=36)
    assert parsed["Fusing LN"]["seconds"] == 73
    assert parsed["Fusing LN"]["completed_match_count"] == 2
    assert parsed["Rotating"]["seconds"] == 90
    assert parsed["Quantising layers"]["seconds"] == 4253
    assert parsed["phase_seconds_sum"] == 4416


def test_v5_release_phase_parser_recovers_subsecond_tqdm_completion():
    from experiments.realq_sdpa_frozen_lr20_20260822 import (
        build_release_v5_merged as release,
    )

    text = (
        "Fusing LN: 100%|x| 28/28 [00:00<00:00, 87.40it/s]\n"
        "Rotating: 100%|x| 28/28 [00:02<00:00, 10.80layer/s]\n"
        "Quantising layers: 100%|x| 28/28 [09:46<00:00, 20.94s/it]\n"
    )

    parsed = release._parse_phase_timing(text, expected_layers=28)

    assert parsed["Fusing LN"]["seconds"] == pytest.approx(28 / 87.40)
    assert parsed["Fusing LN"]["timing_source"].startswith(
        "tqdm completion rate fallback"
    )
    assert parsed["Rotating"]["seconds"] == 2
    assert parsed["Quantising layers"]["seconds"] == 586


@pytest.mark.parametrize("adopted_v1", [False, True])
def test_v5_release_producer_timing_accepts_both_frozen_marker_schemas(
    tmp_path, monkeypatch, adopted_v1
):
    from experiments.realq_sdpa_frozen_lr20_20260822 import (
        build_release_v5_merged as release,
    )

    model = "qwen3-0.6b"
    output_root = tmp_path / "output"
    marker_path = output_root / "shared_cache" / model / "producer_success.json"
    marker_path.parent.mkdir(parents=True)
    tokens_root = tmp_path / "tokens"
    result = {
        "status": "succeeded",
        "returncode": 0,
        "elapsed_seconds": 120.0,
        "command": [
            "python",
            "-m",
            "realq.ptq",
            "--attention_backend",
            "sdpa",
            "--tokens_cache_path",
            str(tokens_root),
            "--require_reference_cache_hit",
            "true",
        ],
    }
    marker = {
        "status": "succeeded",
        "model": model,
        "baseline_reference_unchanged": True,
        "token_contract": {
            "path": str(tokens_root / "tokens.pt"),
            "same_physical_source_as_gptaq_guidedquant": True,
        },
    }
    if adopted_v1:
        source_result = tmp_path / "source_result.json"
        release.base._atomic_json(source_result, result)
        marker.update(
            stage="adopted-exact-v1-deterministic-sdpa-static-cache",
            source_result={
                "path": str(source_result),
                "sha256": release.base._file_sha256(source_result),
            },
        )
        expected_result = source_result
    else:
        marker.update(
            result,
            stage="deterministic-sdpa-static-cache-producer",
        )
        expected_result = marker_path
    release.base._atomic_json(marker_path, marker)
    monkeypatch.setattr(release.campaign, "OUTPUT_ROOT", output_root)

    timing = release._producer_timing(model)

    assert timing["producer_seconds"] == 120.0
    assert timing["per_branch_setting_allocation_seconds"] == 30.0
    assert timing["source_result"]["path"] == str(expected_result)
    assert timing["source_result"]["sha256"] == release.base._file_sha256(
        expected_result
    )


def test_v5_release_config_identity_covers_exact_comparison_matrix():
    from experiments.realq_fullmodel_retune_20260817 import campaign as base
    from experiments.realq_sdpa_frozen_lr20_20260822 import (
        build_release_v5_merged as release,
    )

    identities = [release._config_identity(config) for config in base.CONFIG_IDS]
    assert [identity[2] for identity in identities] == [
        f"C{index:02d}" for index in range(1, 21)
    ]
    assert {identity[0] for identity in identities} == set(release.MODEL_LAYERS)
    assert {identity[1] for identity in identities} == set(release.SETTING_NAMES)


def test_eager_quality_matrix_is_exactly_the_31_frozen_non_q32_rows():
    from experiments.realq_sdpa_frozen_lr20_20260822 import (
        eager_quality_nonq32 as eager,
    )

    pairs = eager._pairs()
    assert len(pairs) == len(set(pairs)) == 31
    assert {branch for branch, _config in pairs} == {
        "full_block",
        "single_linear",
    }
    assert all(not config.startswith("qwen3-32b_") for _branch, config in pairs)
    assert not eager.EXCLUDED_RUNNING_ROWS.intersection(pairs)
    assert eager.EXCLUDED_RUNNING_ROWS == {
        ("full_block", "llama31-8b-instruct_w4a16")
    }
    assert len(eager._source_snapshot()["files"]) == 5


def test_eager_quality_command_and_reference_cache_gates_fail_closed():
    from experiments.realq_sdpa_frozen_lr20_20260822 import (
        eager_quality_nonq32 as eager,
    )

    reference = "/baseline/runtime/ref_logits/model.cache"
    contract = {
        "token": {"path": "/baseline/tokens/calibration.pt"},
        "reference_logits": {
            "path": reference,
            "runtime_root": "/baseline/runtime",
        },
    }
    command = [
        "python",
        "-m",
        "realq.ptq",
        "--dataset",
        "wikitext2",
        "--eval_datasets",
        "wikitext2",
        "--seed",
        "1",
        "--rotation_seed",
        "0",
        "--refresh_seed",
        "0",
        "--nsamples",
        "256",
        "--seq_len",
        "2048",
        "--eval_seq_len",
        "2048",
        "--rotate",
        "true",
        "--attention_backend",
        "sdpa",
        "--skip_eval",
        "false",
        "--skip_kl_ppl_eval",
        "false",
        "--lm_eval",
        "true",
        "--reasoning_eval",
        "false",
        "--require_reference_cache_hit",
        "true",
        "--cache_dir",
        "/baseline/runtime",
        "--tokens_cache_path",
        "/baseline/tokens",
    ]
    eager._validate_command_contract(command, model="model", contract=contract)
    eager._reference_log_gate(
        f"Loading reference logits for wikitext2 from {reference}", reference
    )
    bad = list(command)
    bad[bad.index("--seed") + 1] = "2"
    with pytest.raises(eager.EagerQualityError):
        eager._validate_command_contract(bad, model="model", contract=contract)
    with pytest.raises(eager.EagerQualityError):
        eager._reference_log_gate(
            f"Generating reference logits for wikitext2 at {reference}", reference
        )


def test_eager_quality_adopted_result_binds_final_plan_identity():
    from experiments.realq_sdpa_frozen_lr20_20260822 import (
        eager_quality_nonq32 as eager,
    )

    checkpoint = {"path": "/checkpoint.pt", "size_bytes": 1, "mtime_ns": 2}
    eager_result = {
        "branch": "full_block",
        "config": "qwen3-4b_w4a16",
        "checkpoint": checkpoint,
        "command": ["python", "-m", "realq.ptq", "--output_dir", "/eager"],
        "command_sha256": "command-sha",
        "gpu": {"name": "gpu"},
        "hostname": "host",
        "pid": 7,
        "started_at": "start",
        "elapsed_seconds": 3.0,
        "finished_at": "finish",
        "log": {"path": "/log", "sha256": "log-sha"},
        "metrics": {"wikitext2": {"kl": 1.0, "ppl": 2.0}},
    }
    value = eager._adopted_result(
        eager_result=eager_result,
        official_plan={"quality_plan_fingerprint": "final-fingerprint"},
        official_row={"checkpoint": checkpoint},
        eager_identity={"kind": "eager_nonq32_quality_adoption"},
    )
    assert value["quality_plan_fingerprint"] == "final-fingerprint"
    assert value["checkpoint"] == checkpoint
    assert value["command"] == eager_result["command"]
    assert value["log"] == eager_result["log"]
    assert value["eager_adoption"]["kind"] == "eager_nonq32_quality_adoption"


def test_all_eager_waves_partition_the_canonical_40_by_120_matrix():
    from experiments.realq_sdpa_frozen_lr20_20260822 import (
        eager_quality_final_wave3 as quality_wave3,
    )
    from experiments.realq_sdpa_frozen_lr20_20260822 import (
        eager_quality_nonq32 as quality_nonq32,
    )
    from experiments.realq_sdpa_frozen_lr20_20260822 import (
        eager_quality_q32_wave1 as quality_wave1,
    )
    from experiments.realq_sdpa_frozen_lr20_20260822 import (
        eager_quality_q32_wave2 as quality_wave2,
    )
    from experiments.realq_sdpa_frozen_lr20_20260822 import (
        eager_quality_q32_wave4 as quality_wave4,
    )
    from experiments.realq_sdpa_frozen_lr20_20260822 import (
        eager_reasoning_final_wave3 as reasoning_wave3,
    )
    from experiments.realq_sdpa_frozen_lr20_20260822 import (
        eager_reasoning_nonq32 as reasoning_nonq32,
    )
    from experiments.realq_sdpa_frozen_lr20_20260822 import (
        eager_reasoning_q32_wave1 as reasoning_wave1,
    )
    from experiments.realq_sdpa_frozen_lr20_20260822 import (
        eager_reasoning_q32_wave2 as reasoning_wave2,
    )
    from experiments.realq_sdpa_frozen_lr20_20260822 import (
        eager_reasoning_q32_wave4 as reasoning_wave4,
    )
    from experiments.realq_sdpa_frozen_lr20_20260822 import formal_merged_v5
    from experiments.realq_sdpa_frozen_lr20_20260822 import reasoning_v5_merged

    quality_waves = (
        quality_nonq32._pairs(),
        quality_wave1._pairs(),
        quality_wave2._pairs(),
        quality_wave3._pairs(),
        quality_wave4._pairs(),
    )
    flattened_pairs = [pair for wave in quality_waves for pair in wave]
    assert [len(wave) for wave in quality_waves] == [31, 2, 1, 2, 4]
    assert len(flattened_pairs) == len(set(flattened_pairs)) == 40
    assert set(flattened_pairs) == set(formal_merged_v5._balanced_pairs())

    reasoning_waves = (
        reasoning_nonq32._works(),
        reasoning_wave1._works(),
        reasoning_wave2._works(),
        reasoning_wave3._works(),
        reasoning_wave4._works(),
    )
    flattened_works = [work for wave in reasoning_waves for work in wave]
    assert [len(wave) for wave in reasoning_waves] == [93, 6, 3, 6, 12]
    assert len(flattened_works) == len(set(flattened_works)) == 120
    assert set(flattened_works) == {
        (branch, config, task)
        for branch, config in formal_merged_v5._balanced_pairs()
        for task in reasoning_v5_merged.TASKS
    }
