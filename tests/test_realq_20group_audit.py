from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


campaign = importlib.import_module(
    "experiments.realq_20group_20260808.campaign"
)
audit = importlib.import_module(
    "experiments.realq_20group_20260808.audit"
)


def generation_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        model="/model",
        load_qmodel_path="/checkpoint/quantized.pt",
        reasoning_max_new_tokens=32,
        reasoning_do_sample=False,
        reasoning_temperature=0.0,
        reasoning_top_p=1.0,
        reasoning_top_k=0,
        reasoning_seed=1234,
        reasoning_batch_size=2,
        reasoning_num_samples=1,
        reasoning_protocol="realq_zero_shot_v1",
        w_bits=4,
        w_groupsize=128,
        a_bits=16,
        a_groupsize=-1,
        k_bits=16,
        k_groupsize=-1,
        v_bits=16,
        v_groupsize=-1,
        rotate=True,
    )


def test_generation_audit_requires_exact_unique_identity_and_fingerprint(
    tmp_path: Path, monkeypatch
):
    from realq.benchmarks.generation import (
        generation_config_sha256,
        generation_kwargs,
    )

    monkeypatch.setattr(campaign, "TASKS", {"gsm8k": (32, 2, 2)})
    cfg = generation_cfg()
    fingerprint = generation_config_sha256(
        "gsm8k", cfg, generation_kwargs(cfg)
    )
    path = tmp_path / "generations.jsonl"
    rows = [
        {
            "schema_version": 1,
            "task": "gsm8k",
            "sample_id": str(index),
            "sample_index": 0,
            "prompt_sha256": f"{index + 1:064x}",
            "generation_config_sha256": fingerprint,
            "chunk_index": index,
            "chunk_seed": 100 + index,
            "output": f"Final answer: {index}",
        }
        for index in range(2)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    examples = [SimpleNamespace(sample_id=str(index)) for index in range(2)]
    inventory, by_id = audit.audit_generation_records(
        path, "gsm8k", cfg, examples
    )
    assert inventory["rows"] == 2
    assert inventory["generation_config_sha256"] == fingerprint
    assert set(by_id) == {"0", "1"}

    with path.open("a") as handle:
        handle.write(json.dumps(rows[0]) + "\n")
    with pytest.raises(campaign.CampaignError, match="generation count mismatch"):
        audit.audit_generation_records(path, "gsm8k", cfg, examples)


def test_math_score_audit_recomputes_aggregate_from_exact_details(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(campaign, "TASKS", {"gsm8k": (32, 2, 2)})
    examples = [SimpleNamespace(sample_id="0"), SimpleNamespace(sample_id="1")]
    records = {"0": {"output": "a"}, "1": {"output": "b"}}
    path = tmp_path / "scores.json"
    payload = {
        "summary": {
            "task": "gsm8k",
            "status": "scored",
            "num_examples": 2,
            "num_generations": 2,
            "pass_at_1": 0.5,
            "pass_at_n_oracle": 0.5,
        },
        "details": [
            {
                "sample_id": str(index),
                "sample_count": 1,
                "pass_at_1": index == 0,
                "pass_at_n_oracle": index == 0,
                "candidates": [
                    {
                        "sample_index": 0,
                        "correct": index == 0,
                        "gold_answer": str(index),
                        "predicted_answer": str(index),
                    }
                ],
            }
            for index in range(2)
        ],
    }
    path.write_text(json.dumps(payload))
    result = audit.audit_math_scores(
        path, "gsm8k", examples, records
    )
    assert result["pass"] == 1
    assert result["pass_at_1"] == 0.5

    payload["summary"]["pass_at_1"] = 1.0
    path.write_text(json.dumps(payload))
    with pytest.raises(campaign.CampaignError, match="does not match details"):
        audit.audit_math_scores(path, "gsm8k", examples, records)


def test_tuning_audit_requires_selected_observed_global_minimum(tmp_path: Path):
    root = tmp_path / "campaign"
    run = campaign.RUNS[0]
    tuning = campaign.run_root(root, run) / "tuning"
    protocol = "a" * 64
    state = {
        "status": "tuning_complete",
        "selected_lr": 1e-6,
        "selected_kl": 0.1,
        "protocol_fingerprint": protocol,
        "trials": {
            "0": {
                "status": "succeeded",
                "returncode": 0,
                "lr": 0.0,
                "kl": 0.2,
            },
            "1e-06": {
                "status": "succeeded",
                "returncode": 0,
                "lr": 1e-6,
                "kl": 0.1,
            },
            "2e-06": {
                "status": "succeeded",
                "returncode": 0,
                "lr": 2e-6,
                "kl": 0.3,
            },
        },
    }
    decision = {
        "selected_lr": 1e-6,
        "selected_kl": 0.1,
        "protocol_fingerprint": protocol,
        "trial_launch_count": 3,
    }
    campaign.atomic_json(tuning / "state.json", state)
    campaign.atomic_json(tuning / "controlled_decision.json", decision)
    (tuning / "best_lr.txt").write_text("1e-6\n")
    assert audit.audit_tuning(root, run)["selected_lr"] == 1e-6

    state["selected_lr"] = 0.0
    state["selected_kl"] = 0.2
    decision["selected_lr"] = 0.0
    decision["selected_kl"] = 0.2
    campaign.atomic_json(tuning / "state.json", state)
    campaign.atomic_json(tuning / "controlled_decision.json", decision)
    (tuning / "best_lr.txt").write_text("0\n")
    with pytest.raises(campaign.CampaignError, match="global Exact-KL minimum"):
        audit.audit_tuning(root, run)


def test_frozen_audit_identity_matches_live_campaign_plan():
    assert audit.FROZEN_PLAN_FINGERPRINT == (
        "e187159ad237ac40d06b45aa3271b91ed104112d495d944d7374fdec2732f538"
    )
