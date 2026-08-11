from pathlib import Path

from experiments.realq_linear_refresh_20group_20260811 import campaign as c
from experiments.realq_linear_refresh_20group_20260811 import quality as q


ROOT = Path(
    "/minimax-avatar-new/zhangqian/realq/experiment_data/"
    "realq_linear_refresh_20group_20260811"
)


def test_paper_qa_protocol_is_exact_and_ordered():
    assert q.PAPER_QA_TASKS == (
        "piqa",
        "hellaswag",
        "arc_easy",
        "arc_challenge",
        "winogrande",
        "lambada_openai",
        "ceval-valid",
        "boolq",
        "openbookqa",
        "social_iqa",
    )


def test_quant_quality_args_load_once_and_enable_ppl_and_qa():
    run = c.RUNS[0]
    cfg = c.validate_args(q.quality_args(ROOT, run, Path("/tmp/quality")))
    assert cfg.load_qmodel_path == str(q.checkpoint_path(ROOT, run))
    assert cfg.skip_eval is False
    assert cfg.skip_kl_ppl_eval is False
    assert cfg.lm_eval is True
    assert cfg.lm_eval_batch_size == 32
    assert cfg.reasoning_eval is False
    assert cfg.require_reference_cache_hit is True
    assert cfg.kl_topk == -1
    assert cfg.full_block_refresh is False


def test_quant_node_assignment_is_balanced_and_largest_first():
    for node in (0, 1):
        rows = q.expected_runs_for_node(node)
        assert len(rows) == 10
        assert all(run.node == node for run in rows)
        assert [run.model_index for run in rows] == sorted(
            (run.model_index for run in rows), reverse=True
        )


def test_quality_identity_is_campaign_specific_and_audit_is_not_stale():
    assert q.QUALITY_ID == (
        "realq-linear-refresh-20group-wikitext2-paperqa-20260811-v1"
    )
    assert q.EXPECTED_PLAN_FINGERPRINT == (
        "6c749049865856fcda739d07e16bbcab84ddb8349e618d83b9f0f7449ae060e1"
    )
    assert q.EXPECTED_FINAL_AUDIT_FINGERPRINT == (
        "9ff8b504bc75c03ee755f0dd3cf6bb09424b14f004b5b330775826570e3d291d"
    )
