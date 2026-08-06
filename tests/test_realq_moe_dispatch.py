from __future__ import annotations

from realq import pipeline
from realq.config import Config
from realq_moe import pipeline as moe_pipeline


def test_main_config_keeps_optimized_and_moe_defaults_enabled() -> None:
    cfg = Config()

    assert cfg.quantizer_inner_fastpath is True
    assert cfg.w_clip_search_impl == "symmetric_union_exact"
    assert cfg.fisher_fp32_cache is True
    assert cfg.act_order_stitch_impl == "prefix_q_trailing_w_exact"
    assert cfg.w_clip_update_impl == "where_out"
    assert cfg.w_group_param_layout == "compact"
    assert cfg.prepared_clamp_bound_cache is True
    assert cfg.triton_column_block is True
    assert cfg.moe_gpu_resident is True
    assert cfg.moe_joint_column_block is True
    assert cfg.moe_expert_loss_slide_window is True


def test_main_pipeline_dispatches_supported_moe_before_dense_load(
    monkeypatch,
) -> None:
    cfg = Config(model="local-qwen3-moe", skip_eval=True)
    sentinel = object()
    calls = []

    monkeypatch.setattr(
        pipeline,
        "_model_source_declares_sparse_moe",
        lambda model: model == cfg.model,
    )

    def _run_moe(received):
        calls.append(received)
        return sentinel

    monkeypatch.setattr(moe_pipeline, "run", _run_moe)

    assert pipeline.run(cfg) is sentinel
    assert calls == [cfg]
