from __future__ import annotations

import json

from experiments.turboboa_yaqa_qwen3_rerun.run_reasoning_group import (
    _audit_evalplus,
    _refresh_completed_result_evalplus,
)


def test_evalplus_plus_requires_base_and_extra_tests(tmp_path):
    evaluations = {}
    for index in range(164):
        base_status = "fail"
        plus_status = "fail"
        if index == 0:
            base_status = plus_status = "pass"
        elif index == 1:
            base_status = "pass"
        elif index == 2:
            # EvalPlus may report that the extra tests passed even though a
            # base test failed.  This is not a HumanEval+ pass.
            plus_status = "pass"
        evaluations[f"HumanEval/{index}"] = [
            {
                "task_id": f"HumanEval/{index}",
                "base_status": base_status,
                "plus_status": plus_status,
            }
        ]

    result_path = tmp_path / "samples_eval_results.json"
    result_path.write_text(
        json.dumps({"eval": evaluations}),
        encoding="utf-8",
    )

    summary = _audit_evalplus(tmp_path)

    assert summary["base_pass"] == 2
    assert summary["plus_pass"] == 1
    assert summary["base_pass_at_1"] == 2 / 164
    assert summary["plus_pass_at_1"] == 1 / 164


def test_refresh_completed_result_replaces_stale_plus_summary(tmp_path):
    scorer_root = tmp_path / "scorer"
    scorer_root.mkdir()
    evaluations = {
        f"HumanEval/{index}": [
            {
                "task_id": f"HumanEval/{index}",
                "base_status": "pass" if index in {0, 1} else "fail",
                "plus_status": "pass" if index in {0, 2} else "fail",
            }
        ]
        for index in range(164)
    }
    official_result = scorer_root / "samples_eval_results.json"
    official_result.write_text(
        json.dumps({"eval": evaluations}),
        encoding="utf-8",
    )
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "succeeded",
                "tasks": {
                    "humaneval_plus": {
                        "official": {
                            "official_result": str(official_result),
                            "base_pass": 2,
                            "plus_pass": 2,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    group_state_path = tmp_path / "group_state.json"
    group_state_path.write_text(
        json.dumps(
            {
                "status": "succeeded",
                "result": str(result_path),
                "result_sha256": "stale",
            }
        ),
        encoding="utf-8",
    )

    refreshed = _refresh_completed_result_evalplus(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    group_state = json.loads(group_state_path.read_text(encoding="utf-8"))

    assert refreshed["base_pass"] == 2
    assert refreshed["plus_pass"] == 1
    assert result["tasks"]["humaneval_plus"]["official"] == refreshed
    assert result["evalplus_summary_refreshed_at"]
    assert group_state["result_sha256"] != "stale"
    assert group_state["evalplus_summary_refreshed_at"]
