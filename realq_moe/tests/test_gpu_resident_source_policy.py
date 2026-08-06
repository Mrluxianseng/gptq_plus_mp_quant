from __future__ import annotations

import ast
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]


def _source(relative: str) -> str:
    return (_ROOT / relative).read_text(encoding="utf-8")


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _attribute_calls(node: ast.AST, attribute: str) -> list[ast.Call]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == attribute
    ]


def test_gpu_eval_helpers_have_no_cpu_or_dense_dispatch_calls() -> None:
    tree = ast.parse(_source("realq_moe/pipeline.py"))

    for name in (
        "_activate_moe_gpu_resident_runtime",
        "_capture_gpu_hidden_states",
        "_setup_moe_gpu_eval",
        "_moe_gpu_kl_ppl_eval",
    ):
        function = _function(tree, name)
        assert not _attribute_calls(function, "cpu"), name
        assert not any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "distribute_model"
            for child in ast.walk(function)
        ), name


def test_sparse_runner_uploads_one_token_shard_and_keeps_next_state_cuda() -> None:
    streams_source = _source("realq_moe/runner/streams.py")
    layer_loop_source = _source("realq_moe/runner/layer_loop.py")

    assert "gpu_resident: bool = False" in streams_source
    assert "sample_tensor = sample_tensor.to(" in streams_source
    assert "model(s.view(1, -1))" in streams_source
    assert "gpu_resident=gpu_resident_moe" in layer_loop_source
    assert "rank_samples = rank_samples.to(device=dev, dtype=torch.long)" not in (
        layer_loop_source
    )
    assert "if next_layer is not None and not moe_gpu_resident:" in (
        layer_loop_source
    )
    assert "_assert_layer_inputs_cuda(next_state, dev)" in layer_loop_source


def test_joint_full_slide_arguments_are_forwarded_once_per_projection() -> None:
    source = _source("realq_moe/runner/layer_loop.py")

    assert "if sparse_moe_layer and cfg.moe_expert_loss_slide_window:" in source
    assert "for projection in model_adapter.EXPERT_PROJECTION_ORDER:" in source
    for keyword in (
        "next_layer=",
        "next_fp_outs=",
        "next_fisher=",
        "slide_alpha_fn=",
    ):
        assert keyword in source
    assert "expert_slide_enabled = (" in source
    assert "block_gd_enabled" in source


def test_runtime_files_do_not_import_forbidden_general_optimizations() -> None:
    for relative in (
        "realq_moe/config.py",
        "realq_moe/pipeline.py",
        "realq_moe/runner/streams.py",
        "realq_moe/runner/layer_loop.py",
    ):
        assert "final_perf_integration_20260724" not in _source(relative)
