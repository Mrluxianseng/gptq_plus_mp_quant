from __future__ import annotations

import ast
import copy
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "lowbit_activation_runner.py"
SPEC = importlib.util.spec_from_file_location("lowbit_activation_runner", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def load():
    return runner.load_plan(
        ROOT / "experiments" / "lowbit_activation" / "plan.json"
    )


def select_lr(plan, model, setting, value):
    plan["selected_grad_lr_by_model_setting"][model][setting] = value
    return plan


def test_matrix_has_45_quantized_rows_plus_three_bf16_rows():
    plan = load()
    rows = runner.matrix_rows(plan)
    assert len(rows) == 48
    assert sum(row["method"] == "realq" for row in rows) == 15
    assert sum(row["method"] == "gptaq" for row in rows) == 15
    assert sum(row["method"] == "guided_gptq" for row in rows) == 15
    assert sum(row["method"] == "bf16" for row in rows) == 3
    assert all(row["model"] != "llama3.2-2b" for row in rows)


def test_plan_locks_user_confirmed_final_protocol():
    plan = load()
    assert plan["final"]["nsamples"] == 256
    assert plan["final_layer_grad_lr_by_model"] == {
        "llama3.2-3b": 1e-5,
        "qwen3-4b": 1e-5,
        "qwen3-32b": 1e-6,
    }
    assert plan["fixed_numerics"]["fsdp"] is False
    assert plan["runtime_requirements"] == {
        "minimum_gpu_count": 8,
        "minimum_gpu_memory_gib": 120,
    }
    assert plan["paper_zero_shot_tasks"] == list(runner.EXPECTED_TASKS)
    assert plan["models"] == runner.EXPECTED_MODELS
    assert plan["model_signatures"] == runner.EXPECTED_MODEL_SIGNATURES
    assert plan["runtime_versions"] == runner.EXPECTED_RUNTIME_VERSIONS
    assert (
        plan["runtime_environment"]
        == runner.EXPECTED_RUNTIME_ENVIRONMENT
    )
    assert all(
        value is None
        for model_values in plan[
            "selected_grad_lr_by_model_setting"
        ].values()
        for value in model_values.values()
    )
    assert plan["tuning"]["max_attempts_per_model_setting"] == 20
    assert plan["tuning"]["search_policy"] == {
        "strategy": "adaptive_bracket_then_canonical_refine",
        "canonical_mantissas": [1, 2, 3, 5, 7],
        "boundary_expansion_batch_size": 2,
        "reserved_refinement_attempts": 3,
        "zero_boundary_probe_ratios": [0.1, 0.3, 0.7],
        "fixed_upper_lr_ceiling": None,
        "require_complete_coarse_round": True,
        "require_local_refinement": True,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda plan: plan["resolutions"].pop("runtime_preflight_manifest"),
        lambda plan: plan["resolutions"].pop("canoe_job_id"),
        lambda plan: plan["resolutions"].__setitem__(
            "canoe_submit_confirmed", False
        ),
        lambda plan: plan["models"].__setitem__(
            "llama3.2-3b", "modelzoo/Llama/Llama-3.2-1B"
        ),
        lambda plan: plan["model_signatures"]["llama3.2-3b"].__setitem__(
            "hidden_size", 2048
        ),
        lambda plan: plan["settings"]["3W16A"].__setitem__("w_bits", 2),
        lambda plan: plan["fixed_numerics"].__setitem__("dataset", "c4"),
        lambda plan: plan["fixed_numerics"].__setitem__("seed", 999),
        lambda plan: plan["tuning"].__setitem__("seq_len", 1024),
        lambda plan: plan["final"].__setitem__("global_loss_bsz", 4),
        lambda plan: plan["qwen3_32b_memory_policy"][
            "tuning_ladders"
        ].__setitem__("global_loss_bsz", [8, 1]),
        lambda plan: plan["runtime_versions"].__setitem__(
            "transformers", "0.0.0"
        ),
        lambda plan: plan["runtime_environment"].__setitem__(
            "HF_DATASETS_OFFLINE", "0"
        ),
    ],
)
def test_structure_rejects_every_locked_protocol_mutation(mutation):
    plan = copy.deepcopy(load())
    mutation(plan)
    with pytest.raises(runner.PlanError):
        runner.validate_structure(plan)


def test_tuning_render_uses_per_row_weights_with_proxy_block256():
    plan = load()
    rendered = runner.render_realq(
        plan,
        phase="tune",
        model="qwen3-4b",
        setting_name="3W16A",
        grad_lr=5e-5,
        cuda_devices="0",
    )
    command = rendered.shell()
    assert "--w_groupsize -1" in command
    assert "--blocksize 256" in command
    assert "--act_order true" in command
    assert "--nsamples 256" in command
    assert "--require_static_cache_hit true" in command
    assert "--require_reference_cache_hit true" in command
    assert "--lm_eval false" in command
    assert rendered.argv[:3] == [
        "/minimax-avatar-new/zhangqian/realq/gptq_plus/.venv/bin/python",
        "-m",
        "realq.ptq",
    ]


@pytest.mark.parametrize("bad_lr", [float("nan"), float("inf"), -1e-5])
def test_tuning_render_rejects_nonfinite_or_negative_lr(bad_lr):
    with pytest.raises(runner.PlanError, match="finite non-negative"):
        runner.render_realq(
            load(),
            phase="tune",
            model="qwen3-4b",
            setting_name="3W16A",
            grad_lr=bad_lr,
            cuda_devices="0",
        )


@pytest.mark.parametrize("bad_devices", ["", "foo", "0, 1", "0,0", "-1"])
def test_render_rejects_invalid_cuda_indices(bad_devices):
    with pytest.raises(runner.PlanError):
        runner.render_realq(
            load(),
            phase="tune",
            model="qwen3-4b",
            setting_name="3W16A",
            grad_lr=1e-5,
            cuda_devices=bad_devices,
        )


@pytest.mark.parametrize(
    ("target_phase", "cuda_devices", "expected_group", "expected_block"),
    [
        ("tune", "0", "-1", "256"),
        ("final", "0,1,2,3", "128", "128"),
    ],
)
def test_static_precompute_builds_the_exact_phase_cache_once(
    target_phase,
    cuda_devices,
    expected_group,
    expected_block,
):
    plan = load()
    if target_phase == "final":
        select_lr(plan, "qwen3-4b", "3W16A", 5e-5)
    rendered = runner.render_realq_static_precompute(
        plan,
        target_phase=target_phase,
        model="qwen3-4b",
        cuda_devices=cuda_devices,
    )
    command = rendered.shell()
    assert f"--w_groupsize {expected_group}" in command
    assert f"--blocksize {expected_block}" in command
    assert "--skip_eval false" in command
    assert "--require_static_cache_hit false" in command
    assert "--require_reference_cache_hit false" in command
    assert "--lm_eval false" in command
    assert "--exit_after_precompute true" in command
    assert "--log_column_block_loss false" in command
    assert rendered.run_id == (
        f"precompute_realq_static_{target_phase}_qwen3-4b"
    )


def test_static_precompute_allows_only_effective_global_loss_oom_override():
    plan = load()
    rendered = runner.render_realq_static_precompute(
        plan,
        target_phase="tune",
        model="qwen3-32b",
        cuda_devices="0",
        overrides={"global_loss_bsz": 4},
    )
    assert "--global_loss_bsz 4" in rendered.shell()
    assert "tune_world1_glbsz4" in rendered.shell()
    assert rendered.run_id.endswith("_global_loss_bsz4")

    with pytest.raises(runner.PlanError, match="only a global_loss_bsz"):
        runner.render_realq_static_precompute(
            plan,
            target_phase="tune",
            model="qwen3-32b",
            cuda_devices="0",
            overrides={"backward_bsz": 4},
        )


def test_final_a4_render_is_akv_aware_without_fake_q_flag():
    plan = load()
    select_lr(plan, "qwen3-4b", "4W4A", 5e-5)
    rendered = runner.render_realq(
        plan,
        phase="final",
        model="qwen3-4b",
        setting_name="4W4A",
        grad_lr=5e-5,
        cuda_devices="0,1,2,3",
    )
    command = rendered.shell()
    assert "--a_bits 4" in command
    assert "--k_bits 4" in command
    assert "--v_bits 4" in command
    assert "--act_quant_aware_gptq true" in command
    assert "--k_cache_quant_aware_gptq true" in command
    assert "--grad_lr_layer_schedule none" in command
    assert "--final_layer_grad_lr 1e-05" in command
    assert "--nsamples 256" in command
    assert "--eval_datasets wikitext2" in command
    assert "--lm_eval true" in command
    assert "--q_bits" not in command
    assert "--q_clip" not in command


def test_final_a16_render_is_weight_only_and_does_not_fake_aware():
    plan = load()
    select_lr(plan, "llama3.2-3b", "3W16A", 5e-5)
    rendered = runner.render_realq(
        plan,
        phase="final",
        model="llama3.2-3b",
        setting_name="3W16A",
        grad_lr=5e-5,
        cuda_devices="0,1,2,3",
    )
    command = rendered.shell()
    assert "--a_bits 16" in command
    assert "--k_bits 16" in command
    assert "--v_bits 16" in command
    assert "--act_quant_aware_gptq false" in command
    assert "--k_cache_quant_aware_gptq false" in command
    assert "--grad_lr_layer_schedule cosine" in command


def test_qwen32_final_forbids_fsdp_and_uses_fixed_final_lr():
    plan = load()
    select_lr(plan, "qwen3-32b", "2W4A", 5e-6)
    rendered = runner.render_realq(
        plan,
        phase="final",
        model="qwen3-32b",
        setting_name="2W4A",
        grad_lr=5e-6,
        cuda_devices="4,5,6,7",
        overrides={
            "global_loss_bsz": 8,
            "hessian_accum_bsz": 32,
            "lm_eval_batch_size": 4,
        },
    )
    command = rendered.shell()
    assert "--nproc-per-node=4" in command
    assert rendered.argv[:3] == [
        "/minimax-avatar-new/zhangqian/realq/gptq_plus/.venv/bin/python",
        "-m",
        "torch.distributed.run",
    ]
    assert "--fsdp false" in command
    assert "--cpu_master false" in command
    assert "--final_layer_grad_lr 1e-06" in command
    assert "--bsz 128" in command
    assert "--global_loss_bsz 8" in command
    assert "--hessian_accum_bsz 32" in command
    assert "--lm_eval_batch_size 4" in command
    assert "--quantizer_inner_fastpath false" in command
    assert "--fisher_fp32_cache false" in command
    assert "--w_clip_search_impl cartesian_legacy" in command
    assert "--act_order_stitch_impl full_weight_legacy" in command
    assert "--w_clip_update_impl guarded" in command
    assert "--w_group_param_layout expanded" in command
    assert rendered.run_id.endswith(
        "_global_loss_bsz8_hessian_accum_bsz32_lm_eval_batch_size4"
    )


@pytest.mark.parametrize("method,expected", [("gptaq", "gptaq"), ("guided_gptq", "gptq_guided")])
def test_a4_baselines_use_paper_aware_protocol(method, expected):
    plan = load()
    rendered = runner.render_baseline(
        plan,
        method=method,
        model="qwen3-4b",
        setting_name="3W4A",
        cuda_devices="0",
    )
    command = rendered.shell()
    assert f"--w_method {expected}" in command
    assert "--w_bits 3" in command
    assert "--w_groupsize 128" in command
    assert "--a_bits 4" in command
    assert "--k_bits 4" in command
    assert "--v_bits 4" in command
    assert "--act_quant_aware_gptq" in command
    assert "--k_cache_quant_aware_gptq" in command
    assert "--eval_datasets wikitext2" in command
    assert "--cache_dir cache/lowbit_activation/legacy" in command
    assert "--kl_topk -1" in command
    assert "--offload_inps" not in command
    assert "--q_bits" not in command


def test_bf16_is_parser_legal_and_is_not_rotated_or_quantized():
    plan = load()
    rendered = runner.render_baseline(
        plan,
        method="bf16",
        model="llama3.2-3b",
        setting_name=None,
        cuda_devices="0",
    )
    command = rendered.shell()
    assert "--w_bits 16" in command
    assert "--w_method bf16" not in command
    assert "--w_method" not in command
    assert "--rotate" not in command
    assert "--w_clip" not in command
    assert "--act_quant_aware_gptq" not in command
    assert "--k_cache_quant_aware_gptq" not in command
    assert rendered.output_dir == Path(
        "output/lowbit_activation/llama3.2-3b/BF16/bf16/final/"
        "Llama-3.2-3B/final_bf16_llama3.2-3b_bf16"
    )


def test_gptaq_records_effective_alpha():
    plan = load()
    rendered = runner.render_baseline(
        plan,
        method="gptaq",
        model="qwen3-4b",
        setting_name="3W16A",
        cuda_devices="0",
    )
    assert "--alpha 0.25" in rendered.shell()


def test_legacy_baseline_callers_forward_reviewed_alpha_and_blocksize():
    expected_keywords = {
        "gptaq_utils.py": {"alpha", "blocksize"},
        "gptq_guided_utils.py": {"blocksize"},
    }
    for filename, required in expected_keywords.items():
        tree = ast.parse(
            (ROOT / "gptq_utils" / filename).read_text(encoding="utf-8")
        )
        forwarded = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "fasterquant"
            ):
                forwarded.update(
                    keyword.arg
                    for keyword in node.keywords
                    if keyword.arg is not None
                )
        assert required <= forwarded


def test_guided_saliency_precompute_is_explicit_and_seeded():
    plan = load()
    rendered = runner.render_guided_saliency(
        plan,
        model="qwen3-32b",
        cuda_devices="0,1,2,3",
    )
    command = rendered.shell()
    assert (
        "/minimax-avatar-new/zhangqian/realq/gptq_plus/.venv/bin/python "
        "save_grads.py"
    ) in command
    assert "--dataset wikitext2" in command
    assert "--nsamples 256" in command
    assert "--seq_len 2048" in command
    assert "--seed 1" in command
    assert "--rotation_seed 0" in command
    assert "--num_groups 4" in command
    assert "--rotate" in command


def test_baseline_rejects_fake_batch_override():
    plan = load()
    with pytest.raises(runner.PlanError, match="do not consume"):
        runner.render_baseline(
            plan,
            method="guided_gptq",
            model="qwen3-32b",
            setting_name="2W4A",
            cuda_devices="0",
            overrides={"bsz": 1},
        )


def test_baseline_allows_only_lm_eval_batch_fallback():
    plan = load()
    rendered = runner.render_baseline(
        plan,
        method="guided_gptq",
        model="qwen3-32b",
        setting_name="2W4A",
        cuda_devices="0,1,2,3",
        overrides={"lm_eval_batch_size": 2},
    )
    assert "--lm_eval_batch_size 2" in rendered.shell()
    assert rendered.run_id.endswith("_lm_eval_batch_size2")


def test_nominal_override_does_not_create_a_fake_retry_identity():
    plan = load()
    select_lr(plan, "qwen3-4b", "3W16A", 5e-5)
    without_override = runner.render_realq(
        plan,
        phase="final",
        model="qwen3-4b",
        setting_name="3W16A",
        grad_lr=5e-5,
        cuda_devices="0,1,2,3",
    )
    nominal_override = runner.render_realq(
        plan,
        phase="final",
        model="qwen3-4b",
        setting_name="3W16A",
        grad_lr=5e-5,
        cuda_devices="0,1,2,3",
        overrides={"global_loss_bsz": 32},
    )
    assert nominal_override.run_id == without_override.run_id
    assert nominal_override.output_dir == without_override.output_dir


def test_realq_rejects_noop_bsz_override_and_nondivisible_dp_batch():
    plan = load()
    select_lr(plan, "qwen3-32b", "2W4A", 5e-6)
    with pytest.raises(runner.PlanError, match="does not consume bsz"):
        runner.render_realq(
            plan,
            phase="final",
            model="qwen3-32b",
            setting_name="2W4A",
            grad_lr=5e-6,
            cuda_devices="0,1,2,3",
            overrides={"bsz": 32},
        )
    with pytest.raises(runner.PlanError, match="must be divisible"):
        runner.render_realq(
            plan,
            phase="final",
            model="qwen3-32b",
            setting_name="2W4A",
            grad_lr=5e-6,
            cuda_devices="0,1,2,3",
            overrides={"backward_bsz": 3},
        )


def test_formal_realq_requires_the_recorded_refined_best_lr():
    plan = load()
    with pytest.raises(runner.PlanError, match="best grad_lr is not recorded"):
        runner.render_realq(
            plan,
            phase="final",
            model="qwen3-4b",
            setting_name="3W16A",
            grad_lr=5e-5,
            cuda_devices="0,1,2,3",
        )

    select_lr(plan, "qwen3-4b", "3W16A", 5e-5)
    with pytest.raises(runner.PlanError, match="must equal the recorded best"):
        runner.render_realq(
            plan,
            phase="final",
            model="qwen3-4b",
            setting_name="3W16A",
            grad_lr=1e-4,
            cuda_devices="0,1,2,3",
        )
