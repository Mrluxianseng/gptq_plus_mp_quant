from __future__ import annotations

import dataclasses
import gzip
import importlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


campaign = importlib.import_module(
    "experiments.realq_linear_refresh_20group_20260811.campaign"
)
audit = importlib.import_module("experiments.realq_linear_refresh_20group_20260811.audit")
watch_completion = importlib.import_module(
    "experiments.realq_linear_refresh_20group_20260811.watch_completion"
)


def fake_models(tmp_path: Path, monkeypatch) -> Path:
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(campaign, "WORKSPACE", workspace)
    for model in campaign.MODELS:
        directory = workspace / model.path
        directory.mkdir(parents=True)
        (directory / "config.json").write_text("{}")
        (directory / "tokenizer_config.json").write_text("{}")
        (directory / "tokenizer.json").write_text("{}")
        (directory / "model.safetensors").write_bytes(b"x")
    for task, relative in campaign.DATASET_PATHS.items():
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        expected = campaign.TASKS[task][2]
        if path.suffix == ".gz":
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write("{}\n" * expected)
        else:
            path.write_text("{}\n" * expected)
    return workspace


def test_matrix_has_twenty_unique_balanced_runs():
    assert len(campaign.MODELS) == 5
    assert len(campaign.QUANTS) == 4
    assert len(campaign.RUNS) == 20
    assert len({run.run_id for run in campaign.RUNS}) == 20
    assert [sum(run.node == node for run in campaign.RUNS) for node in (0, 1)] == [10, 10]
    for quant_index in range(4):
        counts = [
            sum(run.quant_index == quant_index and run.node == node for run in campaign.RUNS)
            for node in (0, 1)
        ]
        assert sorted(counts) == [2, 3]


def test_configured_python_accepts_only_two_audited_repo_venvs(monkeypatch):
    monkeypatch.setenv("REALQ_PYTHON", str(campaign.CONTAINER_FALLBACK_PYTHON))
    assert campaign.configured_python() == campaign.CONTAINER_FALLBACK_PYTHON
    monkeypatch.setenv("REALQ_PYTHON", str(campaign.WORKSPACE / "unknown" / "python"))
    with pytest.raises(RuntimeError, match="audited repository venv"):
        campaign.configured_python()


def test_worker_module_entry_preserves_repository_import_path():
    assert campaign.MODULE == "experiments.realq_linear_refresh_20group_20260811.campaign"
    assert str(campaign.WORKSPACE) in campaign.sys.path


def test_worker_env_enables_expandable_cuda_segments():
    env = campaign.worker_env(3)
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_completed_logged_run_requires_a_terminal_footer(tmp_path: Path):
    log = tmp_path / "execution.log"
    log.write_text(
        "[2026-08-08T00:00:00+00:00] command=[]\n"
        "worker output\n"
        "[2026-08-08T00:00:01+00:00] returncode=1 elapsed_seconds=1.250000\n"
    )
    assert campaign.completed_logged_run(log) == (1, 1.25)

    # A newer header after the footer represents an interrupted retry and
    # must not adopt the historical completed result.
    with log.open("a") as handle:
        handle.write("[2026-08-08T00:00:02+00:00] command=[]\npartial output\n")
    assert campaign.completed_logged_run(log) is None


def test_interrupted_trial_recovery_is_locked_archived_and_audited(
    tmp_path: Path,
):
    root = tmp_path / "campaign"
    run = campaign.RUNS[0]
    tuning = campaign.run_root(root, run) / "tuning"
    trial = tuning / "trials" / "lr_1e_06"
    trial.mkdir(parents=True)
    (trial / "spec.json").write_text(
        json.dumps({"lr": 1e-6, "protocol_fingerprint": "fp"})
    )
    (trial / "execution.log").write_text("interrupted")
    campaign.atomic_json(
        tuning / "state.json",
        {
            "protocol_fingerprint": "fp",
            "running": {
                "1e-06": {
                    "worker_pid": 123,
                    "gpu": "2",
                    "started_at": "earlier",
                }
            },
        },
    )
    recovery = campaign.recover_interrupted_trial(
        root, run, 1e-6, "Canoe job is terminal Failed."
    )
    assert recovery["launch_count_charged"] is True
    assert not trial.exists()
    archive = Path(recovery["archive_path"])
    assert (archive / "spec.json").is_file()
    assert (archive / "execution.log").read_text() == "interrupted"
    state = campaign.read_json(tuning / "state.json")
    assert state["running"] == {}
    assert state["interrupted_trial_recoveries"] == [recovery]


def test_scheduler_failure_path_drains_instead_of_terminating_workers():
    source = Path(campaign.__file__).read_text()
    failure_block = source.split("if failures:", 1)[1].split(
        'raise CampaignError(f"node {node} worker failures: {failures}")', 1
    )[0]
    assert "if active:" in failure_block
    assert ".terminate()" not in failure_block


def test_scheduler_does_not_launch_cross_node_failover_producers():
    source = Path(campaign.__file__).read_text()
    assert "failover_precomputes" not in source
    assert "Cross-Pod" in source


def test_frozen_configs_cover_user_contract(tmp_path: Path, monkeypatch):
    fake_models(tmp_path, monkeypatch)
    for run in campaign.RUNS:
        cfg = campaign.validate_args(
            campaign.common_args(tmp_path / "root", run.model, run.quant, global_loss_bsz=32)
        )
        assert cfg.w_groupsize == 128
        assert cfg.w_asym is False and cfg.w_clip is True
        assert cfg.act_order is True and cfg.rotate is True
        assert cfg.nsamples == 256 and cfg.seq_len == 2048
        assert cfg.a_loss_ratio == 1
        assert cfg.fused_block_adam is True
        assert cfg.backward_samples == cfg.backward_bsz == 32
        assert cfg.final_layer_backward_bsz == 32
        assert cfg.global_loss_bsz == 32
        assert cfg.hessian_accum_bsz == (
            32 if run.model.slug == "qwen3-32b" else 64
        )
        assert cfg.a_groupsize == cfg.k_groupsize == cfg.v_groupsize == -1
        assert cfg.a_asym is cfg.k_asym is cfg.v_asym is False
        if run.quant.slug == "w4a4kv4":
            assert (cfg.a_bits, cfg.k_bits, cfg.v_bits) == (4, 4, 4)
            assert cfg.act_quant_aware_gptq is True
            assert cfg.k_cache_quant_aware_gptq is True
            assert cfg.a_clip_ratio == cfg.k_clip_ratio == cfg.v_clip_ratio == 0.9
            assert cfg.grad_lr_layer_schedule == "none"
        else:
            assert (cfg.a_bits, cfg.k_bits, cfg.v_bits) == (16, 16, 16)
            assert cfg.act_quant_aware_gptq is False
            assert cfg.k_cache_quant_aware_gptq is False
            assert cfg.a_clip_ratio == cfg.k_clip_ratio == cfg.v_clip_ratio == 1.0
            assert cfg.grad_lr_layer_schedule == "cosine"


def test_hessian_accumulation_exception_is_qwen32_only(
    tmp_path: Path, monkeypatch
):
    fake_models(tmp_path, monkeypatch)
    values = {
        model.slug: campaign.validate_args(
            campaign.common_args(
                tmp_path / "root", model, campaign.QUANTS[0], global_loss_bsz=1
            )
        ).hessian_accum_bsz
        for model in campaign.MODELS
    }
    assert values == {
        "qwen3-0.6b": 64,
        "qwen3-4b": 64,
        "qwen3-8b": 64,
        "llama31-8b-instruct": 64,
        "qwen3-32b": 32,
    }


def test_tuner_and_formal_are_world_one_and_grouped(tmp_path: Path, monkeypatch):
    fake_models(tmp_path, monkeypatch)
    run = campaign.RUNS[0]
    command = campaign.tuner_command(tmp_path, run, 7, [0.0, 1e-6, 5e-6])
    assert "torchrun" not in command
    assert command[command.index("--parallelism") + 1] == "1"
    assert command[command.index("--tuning-w-groupsize") + 1] == "128"
    assert command[command.index("--tuning-global-loss-bsz") + 1] == "8"
    assert command[command.index("--cuda-ids") + 1] == "7"
    assert [
        float(value)
        for value in command[command.index("--controlled-candidates") + 1].split(",")
    ] == [0.0, 1e-6, 5e-6]
    assert "--run-main" not in command
    argv = campaign.formal_args(
        tmp_path,
        run,
        grad_lr=1e-6,
        global_loss_bsz=16,
        attempt_dir=tmp_path / "attempt",
    )
    cfg = campaign.validate_args(argv)
    assert cfg.global_loss_bsz == 16
    assert cfg.backward_bsz == 32
    assert cfg.require_static_cache_hit is True
    assert cfg.save_qmodel_path.endswith("quantized.pt")


def test_tuner_selection_reconstructs_identical_protocol_base(
    tmp_path: Path, monkeypatch
):
    fake_models(tmp_path, monkeypatch)
    root = tmp_path / "campaign"
    run = campaign.RUNS[0]
    marker = root / "precompute" / run.model.slug / "tuning_success.json"
    campaign.atomic_json(marker, {"tuning_global_loss_bsz": 4})
    candidate = campaign.tuner_command(root, run, 0, [0.0], 4)
    selection = campaign.tuner_select_command(root, run, 0.0, "test bracket")
    assert candidate[candidate.index("--") + 1 :] == selection[
        selection.index("--") + 1 :
    ]
    zero_boundary = campaign.tuner_select_command(
        root,
        run,
        0.0,
        "physical lower bound",
        physical_zero_boundary=True,
    )
    assert "--allow-physical-zero-boundary" in zero_boundary
    assert zero_boundary[zero_boundary.index("--") + 1 :] == candidate[
        candidate.index("--") + 1 :
    ]


def test_eval_protocol_is_three_tasks_greedy_and_task_specific(tmp_path: Path, monkeypatch):
    fake_models(tmp_path, monkeypatch)
    run = campaign.RUNS[0]
    assert set(campaign.TASKS) == {"gsm8k", "math_500", "humaneval_plus"}
    for task, (cap, batch, expected) in campaign.TASKS.items():
        cfg = campaign.validate_args(
            campaign.eval_args(tmp_path, run, task, tmp_path / "eval" / task)
        )
        assert cfg.reasoning_tasks == [task]
        assert cfg.reasoning_enable_thinking is True
        assert cfg.reasoning_do_sample is False
        assert cfg.reasoning_num_samples == 1
        assert cfg.reasoning_limit == -1
        assert cfg.reasoning_max_new_tokens == cap
        assert cfg.reasoning_batch_size == batch
        assert expected in (1319, 500, 164)


def test_evalplus_scorer_uses_audited_python_override():
    source = (campaign.WORKSPACE / "tools/lowbit_activation_evalplus_canoe.sh").read_text()
    assert 'PYTHON_BIN=${REALQ_PYTHON:-$REPO_ROOT/.venv/bin/python}' in source
    assert '"$PYTHON_BIN" -u "$ENTRYPOINT"' in source
    assert 'PATH="$PYTHON_BIN_DIR:/usr/local/bin:/usr/bin:/bin"' in source


def test_official_evalplus_score_requires_and_audits_all_164_tasks(
    tmp_path: Path, monkeypatch
):
    workspace = fake_models(tmp_path, monkeypatch)
    monkeypatch.setattr(campaign, "EVALPLUS_SAMPLE_STABILITY_SECONDS", 0)
    root = tmp_path / "campaign"
    run = campaign.RUNS[0]
    reasoning = campaign.run_root(root, run) / "reasoning" / "humaneval_plus"
    campaign.atomic_json(reasoning / "generation_success.json", {"status": "ok"})
    samples = reasoning / "humaneval_plus" / "evalplus_samples.jsonl"
    samples.parent.mkdir(parents=True)
    expected_ids = [f"HumanEval/{index}" for index in range(164)]
    samples.write_text(
        "".join(
            json.dumps({"task_id": task_id, "solution": "def f():\n    pass"})
            + "\n"
            for task_id in expected_ids
        )
    )
    dataset = workspace / campaign.DATASET_PATHS["humaneval_plus"]
    with gzip.open(dataset, "wt", encoding="utf-8") as handle:
        for task_id in expected_ids:
            handle.write(json.dumps({"task_id": task_id}) + "\n")

    def fake_run(command, **kwargs):
        assert command[0] == "bash"
        assert Path(command[1]).name == "lowbit_activation_evalplus_canoe.sh"
        assert Path(command[2]) == samples
        assert command[4] == str(campaign.EVALPLUS_PARALLEL)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["REALQ_PYTHON"] == str(campaign.PYTHON)
        official = Path(command[3])
        campaign.atomic_json(
            official / "evalplus_samples_eval_results.json",
            {
                "eval": {
                    task_id: [
                        {
                            "base_status": (
                                "pass"
                                if index % 2 == 0
                                else "timeout" if index == 1 else "fail"
                            ),
                            "plus_status": (
                                "pass"
                                if index % 4 == 0
                                else "timeout" if index == 1 else "fail"
                            ),
                        }
                    ]
                    for index, task_id in enumerate(expected_ids)
                }
            },
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(campaign.subprocess, "run", fake_run)
    campaign.run_evalplus_score(root, run)
    result = campaign.read_json(reasoning / "official_eval/official_success.json")
    assert result["status"] == "official_scored"
    assert result["task_count"] == result["base_total"] == result["plus_total"] == 164
    assert result["base_pass"] == 82
    assert result["plus_pass"] == 41
    assert result["base_pass_at_1"] == 0.5
    assert result["plus_pass_at_1"] == 0.25
    assert len(result["samples_sha256"]) == 64
    assert len(result["official_result_sha256"]) == 64


def test_completion_audit_accepts_evalplus_timeout_as_terminal_nonpass():
    assert audit.valid_evalplus_candidate(
        [{"base_status": "timeout", "plus_status": "timeout"}]
    )
    assert not audit.valid_evalplus_candidate(
        [{"base_status": "error", "plus_status": "fail"}]
    )


def test_score_stage_and_worker_do_not_require_a_gpu_argument():
    node = campaign.parser().parse_args(
        ["run-node", "--node", "0", "--stage", "score"]
    )
    assert node.stage == "score"
    worker = campaign.parser().parse_args(
        ["_score", "--root", "/tmp/root", "--run-id", campaign.RUNS[0].run_id]
    )
    assert worker.command == "_score"
    assert not hasattr(worker, "gpu")
    assert campaign.SCORE_WORKER_SLOTS == 2


def test_evaluation_uses_a_blocking_per_run_lock_before_task_dispatch():
    source = inspect.getsource(campaign.run_evaluation)
    assert 'reasoning / ".eval.lock"' in source
    assert "fcntl.LOCK_EX" in source
    assert "fcntl.LOCK_NB" not in source
    assert "_run_evaluation_locked(root, run, gpu)" in source


def test_task_scoped_evaluation_worker_has_its_own_atomic_claim(tmp_path: Path):
    run = campaign.RUNS[0]
    worker = campaign.parser().parse_args(
        [
            "_eval-task",
            "--root",
            "/tmp/root",
            "--run-id",
            run.run_id,
            "--gpu",
            "7",
            "--task",
            "humaneval_plus",
        ]
    )
    assert worker.command == "_eval-task"
    assert worker.task == "humaneval_plus"
    source = inspect.getsource(campaign.run_evaluation_task)
    assert "generation_claim(output, success)" in source
    assert "fcntl.LOCK_EX" not in source
    assert "generation_success.json" in source

    output = tmp_path / "task"
    output.mkdir()
    success = output / "generation_success.json"
    with campaign.generation_claim(output, success) as acquired:
        assert acquired is True
        assert (output / ".generation.claim/owner.json").is_file()
    assert not (output / ".generation.claim").exists()

    campaign.atomic_json(success, {"status": "complete"})
    with campaign.generation_claim(output, success) as acquired:
        assert acquired is False


def test_plan_serializes_full_resolved_config(tmp_path: Path, monkeypatch):
    fake_models(tmp_path, monkeypatch)
    payload = campaign.write_plan(tmp_path / "campaign")
    assert len(payload["runs"]) == 20
    assert len(payload["fingerprint"]) == 64
    fields = {field.name for field in dataclasses.fields(campaign.validate_args([]))}
    assert fields <= set(payload["runs"][0]["formal_config"])
    assert set(payload["model_inventories"]) == {model.slug for model in campaign.MODELS}
    assert {
        task: inventory["rows"]
        for task, inventory in payload["dataset_inventories"].items()
    } == {task: values[2] for task, values in campaign.TASKS.items()}
    assert payload["search"]["mode"] == "agent_supervised_controlled_batches"
    assert payload["search"]["max_trial_launches_per_run"] == 20
    assert payload["search"]["tuning_deterministic_sdpa"] is True
    assert payload["search"]["formal_deterministic_sdpa"] is False
    assert payload["tuning_global_loss_ladder"] == (8, 4, 2, 1)
    assert payload["campaign_id"].endswith("-v1")
    assert payload["full_block_refresh"] is False
    assert payload["weight_update_scope"] == "current_linear_trailing_columns"
    assert all(
        row["formal_config"]["full_block_refresh"] is False
        for row in payload["runs"]
    )
    assert payload["hessian_accum_bsz_by_model"] == {
        "qwen3-0.6b": 64,
        "qwen3-4b": 64,
        "qwen3-8b": 64,
        "llama31-8b-instruct": 64,
        "qwen3-32b": 32,
    }


def test_v5_adopts_only_valid_v4_tuning_precompute(
    tmp_path: Path, monkeypatch
):
    fake_models(tmp_path, monkeypatch)
    from realq.precompute import cache as cache_mod

    root = tmp_path / "campaign"
    model = next(model for model in campaign.MODELS if model.slug == "qwen3-32b")
    selected = 1
    marker = root / "precompute" / model.slug / "tuning_success.json"
    old = {
        "campaign_id": campaign.PREVIOUS_CAMPAIGN_ID,
        "profile": "tuning",
        "model": dataclasses.asdict(model),
        "tuning_attempts": [
            {
                "global_loss_bsz": selected,
                "returncode": 0,
                "oom": False,
            }
        ],
        "tuning_global_loss_bsz": selected,
        "completed_at": "earlier",
    }
    campaign.atomic_json(marker, old)
    cfg = campaign.validate_args(
        campaign.tuning_precompute_args(root, model, selected)
    )
    key = cache_mod.build_cache_key(cfg, 1)
    cache_path = Path(
        cache_mod.cache_path(
            str(campaign.cache_root(root, model) / "static"), key, 1, 0
        )
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"complete")
    monkeypatch.setattr(
        campaign,
        "run_logged",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("compatible producer must not rerun")
        ),
    )
    adopted = campaign.ensure_tuning_precompute(root, model, 0)
    assert adopted["campaign_id"] == campaign.CAMPAIGN_ID
    assert adopted["reused_from_campaign_id"] == campaign.PREVIOUS_CAMPAIGN_ID
    assert adopted["reused_cache_path"] == str(cache_path)
    archive = (
        root
        / "diagnostics/campaign_v5_precompute_marker_adoption"
        / f"{model.slug}_tuning_success_v4.json"
    )
    assert campaign.read_json(archive) == old


def test_tuning_precompute_uses_deterministic_sdpa(tmp_path: Path, monkeypatch):
    fake_models(tmp_path, monkeypatch)
    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return 0, 1.0

    monkeypatch.setattr(campaign, "run_logged", fake_run)
    campaign.ensure_tuning_precompute(tmp_path / "campaign", campaign.MODELS[0], 0)
    assert captured["deterministic_sdpa"] is True


def test_tuning_precompute_oom_only_lowers_global_loss_batch(
    tmp_path: Path, monkeypatch
):
    fake_models(tmp_path, monkeypatch)
    seen = []

    def fake_run(command, log, gpu, **kwargs):
        value = int(command[command.index("--global_loss_bsz") + 1])
        seen.append((value, command[command.index("--backward_bsz") + 1]))
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("CUDA out of memory" if value > 2 else "ok")
        return (1 if value > 2 else 0), 1.0

    monkeypatch.setattr(campaign, "run_logged", fake_run)
    payload = campaign.ensure_tuning_precompute(
        tmp_path / "campaign", campaign.MODELS[0], 0
    )
    assert seen == [(8, "16"), (4, "16"), (2, "16")]
    assert payload["tuning_global_loss_bsz"] == 2


def test_controlled_batch_manifest_is_fingerprinted_and_run_scoped(
    tmp_path: Path, monkeypatch
):
    fake_models(tmp_path, monkeypatch)
    root = tmp_path / "campaign"
    plan = campaign.write_plan(root)
    path = campaign.write_batch(
        root,
        batch_id="initial",
        candidates={
            campaign.RUNS[0].run_id: [0.0, 1e-6, 5e-6],
            campaign.RUNS[1].run_id: [0.0, 1e-6, 5e-6],
        },
        rationale="Initial low-to-high supervised probe.",
    )
    loaded = campaign.load_batch(path)
    assert loaded["batch_id"] == "initial"
    assert loaded["plan_fingerprint"] == plan["fingerprint"]
    assert len(loaded["fingerprint"]) == 64
    assert loaded["candidates"][campaign.RUNS[0].run_id] == [0.0, 1e-6, 5e-6]


def test_tuning_and_formal_precompute_publish_independent_markers(
    tmp_path: Path, monkeypatch
):
    fake_models(tmp_path, monkeypatch)
    root = tmp_path / "campaign"
    monkeypatch.setattr(campaign, "run_logged", lambda *args, **kwargs: (0, 1.25))
    model = campaign.MODELS[0]
    tuning = campaign.ensure_tuning_precompute(root, model, 0)
    assert tuning["profile"] == "tuning"
    assert tuning["tuning_global_loss_bsz"] == 8
    assert (root / "precompute" / model.slug / "tuning_success.json").is_file()
    assert not (root / "precompute" / model.slug / "formal_success.json").exists()
    formal = campaign.ensure_formal_precompute(root, model, 0)
    assert formal["profile"] == "formal"
    assert formal["formal_global_loss_bsz"] == 32
    assert (root / "precompute" / model.slug / "formal_success.json").is_file()


def test_controlled_candidate_cannot_skip_its_predecessor(tmp_path: Path, monkeypatch):
    fake_models(tmp_path, monkeypatch)
    with pytest.raises(campaign.CampaignError, match="low-to-high"):
        campaign.run_tuning_candidate(
            tmp_path / "campaign",
            campaign.RUNS[0],
            0,
            batch_id="initial",
            candidates=[0.0, 1e-6],
            candidate_index=1,
        )


def test_model_inventory_rejects_partial_or_missing_index_shards(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(campaign, "WORKSPACE", workspace)
    model = campaign.MODELS[0]
    directory = workspace / model.path
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    (directory / "tokenizer_config.json").write_text("{}")
    (directory / "tokenizer.json").write_text("{}")
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 2}, "weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}})
    )
    (directory / "model-00001-of-00002.safetensors").write_bytes(b"x")
    try:
        campaign.model_inventory(model)
    except campaign.CampaignError as exc:
        assert "missing or empty" in str(exc)
    else:
        raise AssertionError("missing indexed shard was accepted")
    (directory / "model-00002-of-00002.safetensors.incomplete").write_bytes(b"x")
    try:
        campaign.model_inventory(model)
    except campaign.CampaignError as exc:
        assert "incomplete shards" in str(exc)
    else:
        raise AssertionError("partial shard was accepted")


def test_completion_watcher_requires_formal_generation_and_official_markers(
    tmp_path: Path, monkeypatch
):
    run = SimpleNamespace(run_id="one-run")
    monkeypatch.setattr(campaign, "RUNS", (run,))
    monkeypatch.setattr(campaign, "TASKS", {"one-task": (1, 1, 1)})
    root = tmp_path / "campaign"
    run_dir = campaign.run_root(root, run)

    assert watch_completion.expected_counts() == {
        "formal": 1,
        "generation": 1,
        "official": 1,
    }
    assert watch_completion.completion_counts(root) == {
        "formal": 0,
        "generation": 0,
        "official": 0,
    }

    paths = (
        run_dir / "formal_success.json",
        run_dir / "reasoning" / "one-task" / "generation_success.json",
        run_dir
        / "reasoning"
        / "humaneval_plus"
        / "official_eval"
        / "official_success.json",
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")

    assert watch_completion.completion_counts(root) == {
        "formal": 1,
        "generation": 1,
        "official": 1,
    }
