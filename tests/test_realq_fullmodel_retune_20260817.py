from __future__ import annotations

import copy

import pytest

from experiments.realq_fullmodel_retune_20260817 import campaign
from experiments.realq_fullmodel_retune_20260817 import campaign_v2
from experiments.realq_fullmodel_retune_20260817 import campaign_v3
from experiments.realq_fullmodel_retune_20260817 import selection


def _full_command(*, branch: str, config: str) -> list[str]:
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
        "--full_block_refresh": campaign.BRANCH_VALUES[branch],
        "--grad_lr_layer_schedule": campaign._schedule_for(config),
        "--a_loss_ratio": campaign._a_loss_ratio_text(config),
        "--hessian_accum_bsz": "32" if config.startswith("qwen3-32b_") else "64",
        "--rotate": "true",
        "--attention_backend": "sdpa",
    }
    command = ["python", "-m", "realq.ptq"]
    for flag, value in values.items():
        command.extend((flag, value))
    return command


def test_matrix_is_exactly_two_branches_times_twenty_configs():
    assert len(campaign.CONFIG_IDS) == 20
    assert len(set(campaign.CONFIG_IDS)) == 20
    assert set(campaign.BRANCH_VALUES) == {"full_block", "single_linear"}
    assert len(campaign.CONFIG_IDS) * len(campaign.BRANCH_VALUES) == 40
    assert campaign.INITIAL_LRS == (0.0, 1e-7, 1e-6, 1e-5, 1e-4)


@pytest.mark.parametrize("branch", ["full_block", "single_linear"])
@pytest.mark.parametrize(
    "config",
    ["qwen3-4b_w3a16", "qwen3-32b_w4a4kv4"],
)
def test_full_profile_gate_accepts_only_the_frozen_formal_profile(branch, config):
    command = _full_command(branch=branch, config=config)
    campaign._validate_full_profile(command, branch=branch, config=config)

    bad = copy.copy(command)
    campaign._set_arg(bad, "--blocksize", "256")
    with pytest.raises(campaign.CampaignError, match="blocksize"):
        campaign._validate_full_profile(bad, branch=branch, config=config)

    shallow = copy.copy(command) + ["--quant_stop_layer", "9"]
    with pytest.raises(campaign.CampaignError, match="full-model"):
        campaign._validate_full_profile(shallow, branch=branch, config=config)


def test_profile_gate_freezes_qwen4_clip_and_qwen32_hessian_only_exception():
    qwen4 = _full_command(branch="single_linear", config="qwen3-4b_w4a16")
    campaign._set_arg(qwen4, "--a_loss_ratio", "1.0")
    with pytest.raises(campaign.CampaignError, match="a_loss_ratio"):
        campaign._validate_full_profile(
            qwen4, branch="single_linear", config="qwen3-4b_w4a16"
        )

    qwen32 = _full_command(branch="full_block", config="qwen3-32b_w2a16")
    campaign._set_arg(qwen32, "--hessian_accum_bsz", "64")
    with pytest.raises(campaign.CampaignError, match="hessian_accum_bsz=32"):
        campaign._validate_full_profile(
            qwen32, branch="full_block", config="qwen3-32b_w2a16"
        )


def test_trial_identity_uses_full_precision_lr_and_explicit_replicates():
    base = {
        "branch": "single_linear",
        "config": "qwen3-4b_w3a16",
        "schedule": "cosine",
    }
    left = campaign._trial_id({**base, "lr": 0.0000085857148})
    right = campaign._trial_id({**base, "lr": 0.00000858571483421127})
    repeated = campaign._trial_id(
        {**base, "lr": 0.00000858571483421127, "replicate": "confirm1"}
    )
    assert left != right
    assert right != repeated
    assert repeated.endswith("__rep_confirm1")


def test_historical_full_block_command_gets_explicit_current_flag():
    command = ["python", "-m", "realq.ptq", "--seed", "1"]
    campaign._set_or_append_arg(command, "--full_block_refresh", "true")
    assert campaign._arg_value(command, "--full_block_refresh") == "true"
    campaign._set_or_append_arg(command, "--full_block_refresh", "false")
    assert campaign._arg_value(command, "--full_block_refresh") == "false"
    assert campaign._arg_indices(command, "--full_block_refresh") == [5]


def test_invalid_or_negative_lr_and_unsafe_replicate_fail_closed():
    base = {
        "branch": "full_block",
        "config": "qwen3-8b_w4a16",
        "schedule": "cosine",
    }
    with pytest.raises(campaign.CampaignError, match="non-negative"):
        campaign._trial_id({**base, "lr": -1e-6})
    with pytest.raises(campaign.CampaignError, match="replicate"):
        campaign._trial_id({**base, "lr": 1e-6, "replicate": "bad/value"})


def test_directed_trial_parser_does_not_change_schedule():
    weight_only = campaign._parse_trial_text(
        "full_block/qwen3-8b_w3a16/3.162277660168379e-6/r1"
    )
    activation = campaign._parse_trial_text(
        "single_linear/qwen3-8b_w4a4kv4/1e-5"
    )
    assert weight_only["schedule"] == "cosine"
    assert activation["schedule"] == "none"
    assert weight_only["replicate"] == "r1"


def test_analysis_requires_two_measured_worse_high_side_points(monkeypatch):
    results = [
        {
            "branch": "full_block",
            "config": "qwen3-4b_w3a16",
            "status": "succeeded",
            "lr": lr,
            "kl": kl,
        }
        for lr, kl in [(0.0, 0.20), (1e-6, 0.15), (1e-5, 0.10), (1e-4, 0.12)]
    ]
    monkeypatch.setattr(campaign, "BRANCH_VALUES", {"full_block": "true"})
    monkeypatch.setattr(campaign, "CONFIG_IDS", ("qwen3-4b_w3a16",))
    monkeypatch.setattr(campaign, "_iter_results", lambda: iter(results))
    row = campaign._analysis_rows()[0]
    assert row["best"]["lr"] == 1e-5
    assert row["worse_high_side_points"] == [1e-4]
    assert row["has_two_high_side_points"] is False

    results.append(
        {
            "branch": "full_block",
            "config": "qwen3-4b_w3a16",
            "status": "succeeded",
            "lr": 1e-3,
            "kl": 0.30,
        }
    )
    row = campaign._analysis_rows()[0]
    assert row["has_two_high_side_points"] is True


def test_oom_is_recorded_as_failure_not_high_side_evidence(monkeypatch):
    results = [
        {
            "branch": "single_linear",
            "config": "qwen3-0.6b_w2a16",
            "status": "succeeded",
            "lr": 1e-5,
            "kl": 0.4,
        },
        {
            "branch": "single_linear",
            "config": "qwen3-0.6b_w2a16",
            "status": "failed",
            "failure_class": "oom",
            "lr": 1e-4,
        },
    ]
    monkeypatch.setattr(campaign, "BRANCH_VALUES", {"single_linear": "false"})
    monkeypatch.setattr(campaign, "CONFIG_IDS", ("qwen3-0.6b_w2a16",))
    monkeypatch.setattr(campaign, "_iter_results", lambda: iter(results))
    row = campaign._analysis_rows()[0]
    assert row["failures"] == 1
    assert row["worse_high_side_points"] == []
    assert row["recommendation"] == {"action": "extend_high", "suggested_lr": 1e-4}


def _selection_row(lr, kl, identity):
    return {
        "identity": identity,
        "status": "succeeded",
        "lr": lr,
        "kl": kl,
        "ppl": 10.0,
        "gpu": {"uuid": "GPU-test"},
        "result_path": f"/{identity}.json",
        "result_sha256": identity.ljust(64, "0")[:64],
    }


def test_selection_requires_narrow_bracket_two_high_points_and_repeats(monkeypatch):
    rows = [
        _selection_row(9e-6, 0.1010, "lower0"),
        _selection_row(9e-6, 0.1010, "lower1"),
        _selection_row(1e-5, 0.1000, "best0"),
        _selection_row(1e-5, 0.1000, "best1"),
        _selection_row(1.2e-5, 0.1020, "high0"),
        _selection_row(2e-5, 0.1100, "high1"),
    ]
    monkeypatch.setattr(selection, "_result_rows", lambda branch, config: rows)
    result = selection._analyze_one("full_block", "qwen3-4b_w3a16")
    assert result["bracket"]["width_dex"] < 0.30
    assert result["high_side_gate"] is True
    assert result["replicate_gate"] is True
    assert result["ready"] is True
    assert result["selected_lr"] == 1e-5


def test_selection_uses_observed_repeat_noise_and_conservative_lower_lr(monkeypatch):
    rows = [
        _selection_row(9e-6, 0.1005, "lower0"),
        _selection_row(9e-6, 0.1005, "lower1"),
        _selection_row(1e-5, 0.0990, "best0"),
        _selection_row(1e-5, 0.1010, "best1"),
        _selection_row(1.2e-5, 0.1030, "high0"),
        _selection_row(2e-5, 0.1100, "high1"),
    ]
    monkeypatch.setattr(selection, "_result_rows", lambda branch, config: rows)
    result = selection._analyze_one("single_linear", "qwen3-4b_w3a16")
    assert result["replicate_gate"] is False
    assert result["suggestions"] == [
        {
            "reason": "repeat_top_two_for_noise_gate",
            "lr": 1e-5,
            "replicate": "confirm2",
        }
    ]

    rows.append(_selection_row(1e-5, 0.1000, "best2"))
    result = selection._analyze_one("single_linear", "qwen3-4b_w3a16")
    assert result["replicate_gate"] is True
    assert result["noise_tie"] is True
    assert result["selected_lr"] == 9e-6


def test_selection_releases_first_positive_only_after_zero_endpoint(monkeypatch):
    rows = [_selection_row(0.0, 0.10, "zero0")]
    monkeypatch.setattr(selection, "_result_rows", lambda branch, config: rows)
    result = selection._analyze_one("full_block", "qwen3-4b_w3a16")
    assert result["suggestions"] == [
        {"reason": "first_positive_candidate", "lr": 1e-7}
    ]


def test_selection_refines_toward_zero_when_first_positive_is_worse(monkeypatch):
    rows = [
        _selection_row(0.0, 0.10, "zero0"),
        _selection_row(1e-7, 0.11, "positive0"),
    ]
    monkeypatch.setattr(selection, "_result_rows", lambda branch, config: rows)
    result = selection._analyze_one("single_linear", "qwen3-4b_w3a16")
    assert result["best"]["lr"] == 0.0
    assert result["suggestions"] == [
        {"reason": "refine_physical_zero_boundary", "lr": 1e-8}
    ]


def test_v2_balanced_zero_matrix_contains_each_branch_config_once():
    pairs = campaign_v2._balanced_pairs()
    assert len(pairs) == 40
    assert len(set(pairs)) == 40
    assert set(pairs) == {
        (branch, config)
        for branch in campaign.BRANCH_VALUES
        for config in campaign.CONFIG_IDS
    }
    # The first 16 lanes mix five model sizes instead of placing all eight
    # Llama tokenizers on one node as V1 did.
    first_models = {
        next(model for model in campaign.MODEL_SLUGS if config.startswith(model))
        for _, config in pairs[:16]
    }
    assert len(first_models) == 5


def test_v2_paired_flag_map_allows_only_explicit_branch_difference():
    full = _full_command(branch="full_block", config="qwen3-4b_w3a16")
    single = _full_command(branch="single_linear", config="qwen3-4b_w3a16")
    left = campaign_v2._flag_map(full)
    right = campaign_v2._flag_map(single)
    assert left.pop("--full_block_refresh") == "true"
    assert right.pop("--full_block_refresh") == "false"
    assert left == right


def test_v3_does_not_treat_grad_lr_zero_as_cross_branch_physical_zero(monkeypatch):
    template = {
        "campaign_id": campaign_v2.CAMPAIGN_ID,
        "output_root": str(campaign_v2.OUTPUT_ROOT),
        "initial_lrs": [0.0, 1e-7, 1e-6, 1e-5, 1e-4],
        "selection_protocol": {"paired_zero_control_gate": "invalid"},
        "code_snapshot": {"sha256": "old"},
        "protocol_fingerprint": "old-fingerprint",
        "created_at": "old-time",
    }
    monkeypatch.setattr(campaign_v2, "_build_plan", lambda: copy.deepcopy(template))
    monkeypatch.setattr(
        campaign, "_code_snapshot", lambda: {"sha256": "v3-code"}
    )
    monkeypatch.setattr(campaign, "_utc_now", lambda: "v3-time")

    plan = campaign_v3._build_plan()
    protocol = plan["selection_protocol"]
    assert plan["campaign_id"] == campaign_v3.CAMPAIGN_ID
    assert plan["output_root"] == str(campaign_v3.OUTPUT_ROOT)
    assert plan["initial_lrs"] == [0.0]
    assert protocol["cross_branch_zero_equality_gate"] is False
    assert "final_layer_grad_lr" in protocol["cross_branch_zero_equality_reason"]
    assert plan["v2_invalid_gate_evidence"]["final_layer_grad_lr"] == 1e-5
    stable = dict(plan)
    fingerprint = stable.pop("protocol_fingerprint")
    stable.pop("created_at")
    assert campaign._canonical_sha256(stable) == fingerprint
