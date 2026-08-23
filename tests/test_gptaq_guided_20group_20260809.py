from pathlib import Path

from experiments.gptaq_guided_20group_20260809 import campaign as c


def option(argv, name):
    index = argv.index(name)
    return argv[index + 1]


def test_matrix_and_node_balance():
    assert len(c.RUNS) == 40
    assert {run.method for run in c.RUNS} == {"gptaq", "guided_gptq"}
    assert {run.quant.slug for run in c.RUNS} == {"w4a16", "w4a4kv4", "w3a16", "w2a16"}
    assert sum(run.node == 0 for run in c.RUNS) == 20
    assert sum(run.node == 1 for run in c.RUNS) == 20
    q32_guided = [run for run in c.RUNS if run.method == "guided_gptq" and run.model.slug == "qwen3-32b"]
    assert sum(run.node == 0 for run in q32_guided) == 2
    assert sum(run.node == 1 for run in q32_guided) == 2


def test_legacy_protocol_is_frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(c.base, "model_path", lambda model: f"/models/{model.slug}")
    monkeypatch.setattr(c.base, "PYTHON", Path("/audited/venv/bin/python"))
    plan = c.legacy_plan(tmp_path)
    for run in c.RUNS:
        rendered = c.legacy_runner.render_baseline(
            plan,
            method=run.method,
            model=run.model.slug,
            setting_name=c.SETTING_NAMES[run.quant.slug],
            cuda_devices="0",
        )
        argv = rendered.argv
        assert option(argv, "--nsamples") == "256"
        assert option(argv, "--seq_len") == "2048"
        assert option(argv, "--w_groupsize") == "128"
        assert option(argv, "--blocksize") == "128"
        assert "--w_clip" in argv and "--act_order" in argv and "--rotate" in argv
        assert "--offload_inps" not in argv
        assert option(argv, "--w_bits") == str(run.quant.w_bits)
        assert option(argv, "--a_bits") == str(run.quant.a_bits)
        assert option(argv, "--k_bits") == str(run.quant.k_bits)
        assert option(argv, "--v_bits") == str(run.quant.v_bits)
        if run.quant.slug == "w4a4kv4":
            assert "--act_quant_aware_gptq" in argv
            assert "--k_cache_quant_aware_gptq" in argv
            assert option(argv, "--a_clip_ratio") == "0.9"
        else:
            assert "--act_quant_aware_gptq" not in argv
            assert "--k_cache_quant_aware_gptq" not in argv
            assert option(argv, "--a_clip_ratio") == "1.0"
        if run.method == "gptaq":
            assert option(argv, "--w_method") == "gptaq"
            assert option(argv, "--alpha") == "0.25"
        else:
            assert option(argv, "--w_method") == "gptq_guided"
            assert "--alpha" not in argv


def test_reasoning_scope_matches_realq_campaign():
    assert tuple(c.base.TASKS) == ("gsm8k", "math_500", "humaneval_plus")
    assert tuple(c.quality.PAPER_QA_TASKS) == (
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
