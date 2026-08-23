from pathlib import Path

from experiments.realq_20group_20260808 import bf16
from experiments.realq_20group_20260808 import campaign as c
from experiments.realq_20group_20260808 import quality as q


ROOT = Path(
    "/minimax-avatar-new/zhangqian/realq/experiment_data/"
    "realq_20group_20260808"
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


def test_quant_node_assignment_is_balanced_and_largest_first():
    for node in (0, 1):
        rows = q.expected_runs_for_node(node)
        assert len(rows) == 10
        assert all(run.node == node for run in rows)
        assert [run.model_index for run in rows] == sorted(
            (run.model_index for run in rows), reverse=True
        )


def test_bf16_is_five_unique_models_mapped_to_twenty_rows():
    assert len(bf16.WORKS) == 20
    assert {work.model.slug for work in bf16.WORKS} == {
        model.slug for model in c.MODELS
    }
    assert {work.kind for work in bf16.WORKS} == set(bf16.KINDS)
    assert [len(bf16.expected_works_for_node(node)) for node in (0, 1)] == [10, 10]
    assert set(bf16.MODEL_BY_SLUG) == {model.slug for model in c.MODELS}


def test_bf16_configs_are_unquantized_and_protocol_exact():
    for work in bf16.WORKS:
        cfg = c.validate_args(
            bf16.work_args(ROOT, work, Path("/tmp/bf16-quality"))
        )
        assert (cfg.w_bits, cfg.a_bits, cfg.k_bits, cfg.v_bits) == (16, 16, 16, 16)
        assert cfg.rotate is False
        assert cfg.load_qmodel_path is None
        if work.kind == "quality":
            assert cfg.skip_eval is False
            assert cfg.skip_kl_ppl_eval is False
            assert cfg.lm_eval is True
            assert cfg.require_reference_cache_hit is True
            assert cfg.reasoning_eval is False
        else:
            assert cfg.skip_eval is True
            assert cfg.lm_eval is False
            assert cfg.reasoning_eval is True
            assert cfg.reasoning_tasks == [work.kind]
            assert cfg.reasoning_limit == -1
            assert cfg.reasoning_num_samples == 1
            assert cfg.reasoning_do_sample is False
            assert cfg.reasoning_protocol == "realq_zero_shot_v1"
