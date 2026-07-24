import math
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from realq.config import Config
from tools import run_schedule_ablation_variant as entrypoint
from tools.run_schedule_ablation_variant import (
    _dataset_pin,
    _dispatch_schedule,
    _validate_experiment_protocol,
    _validate_measured_arm,
    schedule_ablation_lr,
)


def _expected_sin(x: float, base_ratio: float = 0.01) -> float:
    return base_ratio + (1.0 - base_ratio) * math.sin(math.pi * x / 2.0)


def _expected_cos2(x: float, base_ratio: float = 0.01) -> float:
    return base_ratio + (1.0 - base_ratio) * 0.5 * (
        1.0 - math.cos(math.pi * x)
    )


@pytest.mark.parametrize("layer_idx", [0, 1, 9, 18, 27, 35])
def test_weight_only_schedule_arms_match_their_closed_forms(layer_idx):
    x = layer_idx / 35
    paper = schedule_ablation_lr(
        "paper", 1.0, layer_idx, 36, 0.01, "cosine",
        activation_aware=False,
    )
    sin_scheduled = schedule_ablation_lr(
        "paper_sin_scheduled", 1.0, layer_idx, 36, 0.01, "cosine",
        activation_aware=False,
    )
    legacy = schedule_ablation_lr(
        "legacy_cos2_scheduled", 1.0, layer_idx, 36, 0.01, "cosine",
        activation_aware=False,
    )
    assert paper == pytest.approx(_expected_sin(x), abs=1e-15)
    assert sin_scheduled == pytest.approx(paper, abs=1e-15)
    assert legacy == pytest.approx(_expected_cos2(x), abs=1e-15)


@pytest.mark.parametrize("layer_idx", [0, 1, 9, 18, 27, 34])
def test_aware_three_arm_ablation_separates_curve_and_disable(layer_idx):
    paper = schedule_ablation_lr(
        "paper", 5e-6, layer_idx, 36, 0.01, "cosine",
        activation_aware=True,
    )
    sin_scheduled = schedule_ablation_lr(
        "paper_sin_scheduled", 5e-6, layer_idx, 36, 0.01, "cosine",
        activation_aware=True,
    )
    legacy = schedule_ablation_lr(
        "legacy_cos2_scheduled", 5e-6, layer_idx, 36, 0.01, "cosine",
        activation_aware=True,
    )
    x = layer_idx / 35
    assert paper == 5e-6
    assert sin_scheduled == pytest.approx(
        5e-6 * _expected_sin(x), abs=1e-20
    )
    assert legacy == pytest.approx(
        5e-6 * _expected_cos2(x), abs=1e-20
    )


@pytest.mark.parametrize(
    "variant",
    ["paper", "paper_sin_scheduled", "legacy_cos2_scheduled"],
)
def test_none_schedule_is_constant_for_every_arm(variant):
    assert schedule_ablation_lr(
        variant, 3e-4, 0, 36, 0.01, "none",
        activation_aware=False,
    ) == 3e-4


def test_unknown_schedule_or_variant_is_rejected():
    with pytest.raises(ValueError, match="Unknown schedule ablation variant"):
        schedule_ablation_lr(
            "other", 1.0, 0, 36, 0.01, "cosine",
            activation_aware=False,
        )
    with pytest.raises(ValueError, match="Unknown grad_lr_layer_schedule"):
        schedule_ablation_lr(
            "paper", 1.0, 0, 36, 0.01, "linear",
            activation_aware=False,
        )


def test_paper_dispatch_delegates_to_production_function():
    calls = []

    def production(*args, **kwargs):
        calls.append((args, kwargs))
        return 7.25e-6

    value = _dispatch_schedule(
        "paper",
        production,
        5e-6,
        3,
        36,
        0.01,
        "cosine",
        activation_aware=True,
    )
    assert value == 7.25e-6
    assert calls == [
        (
            (5e-6, 3, 36, 0.01, "cosine"),
            {"activation_aware": True},
        )
    ]

    calls.clear()
    _dispatch_schedule(
        "legacy_cos2_scheduled",
        production,
        5e-6,
        3,
        36,
        0.01,
        "cosine",
        activation_aware=True,
    )
    assert calls == []


def test_model_state_hash_covers_bfloat16_parameters_and_buffers():
    model = torch.nn.Linear(4, 3, bias=False).to(torch.bfloat16)
    model.register_buffer(
        "marker",
        torch.tensor([1, 2, 3], dtype=torch.int32),
    )
    before = entrypoint._model_state_sha256(model)
    assert before == entrypoint._model_state_sha256(model)
    with torch.no_grad():
        model.weight.view(-1)[0] += 1
    assert entrypoint._model_state_sha256(model) != before


def _rank_metrics(rank: int, kl: float, ppl: float):
    return {
        "rank": rank,
        "metrics": {
            "wikitext2": {
                "kl_raw": kl,
                "kl_x100": kl * 100.0,
                "ppl": ppl,
            }
        },
    }


def test_metric_spread_accepts_only_declared_kernel_noise():
    report = entrypoint._metric_spread_report(
        [
            _rank_metrics(0, 0.05432198, 13.45678),
            _rank_metrics(1, 0.05432200, 13.45679),
        ]
    )
    assert report["wikitext2"]["kl_raw"]["spread"] > 0
    assert (
        report["wikitext2"]["kl_raw"]["spread"]
        < report["wikitext2"]["kl_raw"]["allowed_spread"]
    )

    with pytest.raises(RuntimeError, match="differ materially"):
        entrypoint._metric_spread_report(
            [
                _rank_metrics(0, 0.0543, 13.45),
                _rank_metrics(1, 0.0553, 13.55),
            ]
        )
    with pytest.raises(RuntimeError, match="non-finite"):
        entrypoint._metric_spread_report(
            [
                _rank_metrics(0, 0.0543, 13.45),
                _rank_metrics(1, float("nan"), 13.45),
            ]
        )


def test_exclusive_json_lock_never_overwrites(tmp_path):
    lock = tmp_path / "run.lock"
    entrypoint._acquire_exclusive_json_lock(lock, {"owner": "first"})
    with pytest.raises(FileExistsError):
        entrypoint._acquire_exclusive_json_lock(lock, {"owner": "second"})
    assert json.loads(lock.read_text()) == {"owner": "first"}


def test_experiment_parser_does_not_abbreviate_production_exp_flag(tmp_path):
    experiment, remaining = entrypoint._parse_experiment_args(
        [
            "--experiment_case",
            "qwen3_4b_w4a16",
            "--experiment_root",
            str(tmp_path),
            "--schedule_ablation_variant",
            "paper",
            "--wikitext_parquet_dir",
            str(tmp_path),
            "--rotation_fingerprint_path",
            str(tmp_path / "rotation.json"),
            "--exp",
            "qwen3_4b_w4a16__warm",
        ]
    )
    assert experiment.experiment_case == "qwen3_4b_w4a16"
    assert remaining == ["--exp", "qwen3_4b_w4a16__warm"]


@pytest.mark.parametrize(
    ("warm", "variant"),
    [
        (True, None),
        (False, "legacy_cos2_scheduled"),
    ],
)
def test_launcher_mechanically_expands_locked_case(
    tmp_path,
    warm,
    variant,
):
    from tools.launch_schedule_ablation import build_command
    from realq.config import parse_cli

    model = tmp_path / "Qwen3-4B"
    dataset = tmp_path / "wikitext"
    model.mkdir()
    dataset.mkdir()
    args = SimpleNamespace(
        experiment_root=(tmp_path / "experiments"),
        wikitext_parquet_dir=dataset,
        model=model,
        case="qwen3_4b_w4a16",
        variant=variant,
        warm=warm,
        gpus="0,1,2,3",
        master_port=29611,
    )
    command, env = build_command(args)
    start = command.index("tools.run_schedule_ablation_variant") + 1
    experiment, realq_argv = entrypoint._parse_experiment_args(
        command[start:]
    )
    cfg = parse_cli(realq_argv)
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert cfg.global_loss_bsz == 48
    assert cfg.hessian_accum_bsz == 256
    assert cfg.grad_clip == 1.0
    assert cfg.final_layer_grad_clip is None
    assert cfg.exit_after_precompute is warm
    assert experiment.allow_create_rotation_fingerprint is warm
    assert experiment.require_precomputed_caches is (not warm)
    assert (experiment.metrics_json_path is None) is warm


def _write_test_parquets(tmp_path: Path, monkeypatch) -> Path:
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    dataset_dir = tmp_path / "wikitext"
    dataset_dir.mkdir()
    observed = {}
    for index, (split, filename) in enumerate(
        entrypoint._SPLIT_FILES.items()
    ):
        path = dataset_dir / filename
        parquet.write_table(
            pyarrow.table({"text": [f"{split}-row-0", f"{split}-row-{index + 1}"]}),
            path,
        )
        observed[split] = entrypoint._sha256(path)
    monkeypatch.setattr(entrypoint, "_EXPECTED_SPLIT_SHA256", observed)
    return dataset_dir


def test_dataset_pin_rejects_even_one_changed_byte(tmp_path, monkeypatch):
    dataset_dir = _write_test_parquets(tmp_path, monkeypatch)
    _dataset_pin(dataset_dir)
    train = dataset_dir / entrypoint._SPLIT_FILES["train"]
    with train.open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _dataset_pin(dataset_dir)


def test_measured_arm_requires_metrics_and_eval():
    with pytest.raises(ValueError, match="metrics_json_path"):
        _validate_measured_arm(Config(model="fake"), None)
    with pytest.raises(ValueError, match="skip_eval"):
        _validate_measured_arm(
            Config(model="fake", skip_eval=True),
            Path("metrics.json"),
        )
    _validate_measured_arm(
        Config(model="fake", exit_after_precompute=True, skip_eval=True),
        None,
    )
    with pytest.raises(ValueError, match="dataset wikitext2"):
        _validate_measured_arm(
            Config(model="fake", dataset="neuralmagic"),
            Path("metrics.json"),
        )
    with pytest.raises(ValueError, match="eval_datasets wikitext2"):
        _validate_measured_arm(
            Config(model="fake", eval_datasets=["wikitext2", "other"]),
            Path("metrics.json"),
        )
    with pytest.raises(ValueError, match="grad_lr_layer_schedule cosine"):
        _validate_measured_arm(
            Config(model="fake", grad_lr_layer_schedule="none"),
            Path("metrics.json"),
        )


def _manifest_case(
    tmp_path: Path,
    *,
    case: str = "qwen3_4b_w4a16",
    variant: str = "paper",
    warm: bool = False,
):
    protocol = entrypoint._CASE_PROTOCOLS[case]
    paths = entrypoint._case_paths(
        tmp_path,
        case,
        variant,
        warm=warm,
    )
    config_values = {
        **entrypoint._COMMON_CASE_CONFIG,
        **protocol["config"],
        "model": str((tmp_path / protocol["model_name"]).resolve()),
        "static_cache_path": str(paths["static_cache_path"]),
        "tokens_cache_path": str(paths["tokens_cache_path"]),
        "cache_dir": str(paths["cache_dir"]),
        "output_dir": str(paths["output_dir"]),
        "exp": (
            f"{case}__warm"
            if warm
            else f"{case}__{variant}"
        ),
        "exit_after_precompute": warm,
    }
    cfg = Config(**config_values)
    experiment = SimpleNamespace(
        experiment_case=case,
        experiment_root=tmp_path.resolve(),
        schedule_ablation_variant=variant,
        wikitext_parquet_dir=(tmp_path / "wikitext").resolve(),
        metrics_json_path=(
            None if warm else paths["metrics_json_path"].resolve()
        ),
        rotation_fingerprint_path=paths[
            "rotation_fingerprint_path"
        ].resolve(),
        allow_create_rotation_fingerprint=warm,
        require_precomputed_caches=not warm,
        overwrite_metrics=False,
    )
    return cfg, experiment, protocol["model_artifact_identity"]


def test_manifest_case_protocol_accepts_only_exact_case(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    cfg, experiment, model_identity = _manifest_case(tmp_path)
    _validate_experiment_protocol(
        cfg,
        experiment,
        model_artifact_identity=model_identity,
    )

    cfg.global_loss_bsz = 16
    with pytest.raises(ValueError, match="global_loss_bsz"):
        _validate_experiment_protocol(
            cfg,
            experiment,
            model_artifact_identity=model_identity,
        )


def test_manifest_warm_and_measured_cache_contracts(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    warm_cfg, warm_experiment, model_identity = _manifest_case(
        tmp_path,
        warm=True,
    )
    _validate_experiment_protocol(
        warm_cfg,
        warm_experiment,
        model_artifact_identity=model_identity,
    )

    warm_experiment.allow_create_rotation_fingerprint = False
    with pytest.raises(
        ValueError,
        match="allow_create_rotation_fingerprint",
    ):
        _validate_experiment_protocol(
            warm_cfg,
            warm_experiment,
            model_artifact_identity=model_identity,
        )

    measured_cfg, measured_experiment, model_identity = _manifest_case(
        tmp_path,
    )
    measured_experiment.require_precomputed_caches = False
    with pytest.raises(ValueError, match="require_precomputed_caches"):
        _validate_experiment_protocol(
            measured_cfg,
            measured_experiment,
            model_artifact_identity=model_identity,
        )


def test_manifest_rejects_wrong_model_identity_and_world(
    tmp_path,
    monkeypatch,
):
    cfg, experiment, model_identity = _manifest_case(tmp_path)
    monkeypatch.setenv("WORLD_SIZE", "8")
    with pytest.raises(ValueError) as error:
        _validate_experiment_protocol(
            cfg,
            experiment,
            model_artifact_identity=f"wrong-{model_identity}",
        )
    message = str(error.value)
    assert "model_artifact_identity" in message
    assert "world_size" in message


def test_wrapper_sets_deterministic_env_before_torch_import():
    code = (
        "import os; "
        "os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None); "
        "os.environ.pop('HF_DATASETS_TRUST_REMOTE_CODE', None); "
        "import tools.run_schedule_ablation_variant; "
        "assert 'torch' not in __import__('sys').modules; "
        "print(os.environ['CUBLAS_WORKSPACE_CONFIG']); "
        "print(os.environ['HF_DATASETS_TRUST_REMOTE_CODE'])"
    )
    env = dict(os.environ)
    env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    env.pop("HF_DATASETS_TRUST_REMOTE_CODE", None)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.stdout.splitlines() == [":4096:8", "1"]


def test_main_wires_schedule_pinned_cache_and_exact_metrics(
    tmp_path,
    monkeypatch,
):
    dataset_dir = _write_test_parquets(tmp_path, monkeypatch)
    metrics_path = tmp_path / "metrics.json"
    token_dir = tmp_path / "tokens"
    static_dir = tmp_path / "static"
    old_token_path = token_dir / "fake_wikitext2_train_n4_sl8_seed1.pt"
    old_token_path.parent.mkdir()
    old_token_path.write_bytes(b"historical-cache-must-not-be-used")

    from realq import config as config_mod
    from realq.parallel import env as parallel_env
    from realq.precompute import cache as static_cache
    import realq.precompute.static_e2e as static_e2e
    import realq.pipeline as pipeline_mod
    import realq.ptq as ptq
    import realq.refresh.block_gd as block_gd
    import realq.runner.layer_loop as layer_loop
    from utils.cache_identity import artifact_identity
    from utils import data_utils, dist_utils, eval_utils

    original_schedule = lambda *args, **kwargs: -123.0
    monkeypatch.setattr(block_gd, "layer_lr_for_schedule", original_schedule)
    monkeypatch.setattr(layer_loop, "layer_lr_for_schedule", original_schedule)
    monkeypatch.setattr(static_cache, "build_cache_key", lambda cfg, world: "base")
    monkeypatch.setattr(pipeline_mod, "_maybe_rotate", lambda cfg, analyzer: None)
    monkeypatch.setattr(parallel_env, "get_world_size", lambda: 1)
    monkeypatch.setattr(dist_utils, "is_main", lambda: True)
    monkeypatch.setattr(
        eval_utils,
        "_kl_ppl_eval",
        lambda *args, **kwargs: (13.456789, 0.054321987),
    )
    reference_metadata = {"schema": "test-reference-v1"}
    monkeypatch.setattr(
        eval_utils,
        "_reference_cache_metadata",
        lambda *args, **kwargs: reference_metadata,
    )
    monkeypatch.setattr(
        eval_utils,
        "_load_reference_cache",
        lambda path, metadata: (
            torch.tensor([1.0]) if Path(path).is_file() else None
        ),
    )
    monkeypatch.setattr(eval_utils, "pretty_print_results", lambda results: None)
    monkeypatch.setattr(
        entrypoint,
        "_validate_experiment_protocol",
        lambda *args, **kwargs: None,
    )

    seen_cache_paths = []

    def fake_get_tokens(
        dataset_name,
        split,
        tokenizer,
        seq_len,
        num_samples,
        save_path=None,
        seed=0,
    ):
        seen_cache_paths.append(save_path)
        assert save_path != str(old_token_path)
        assert data_utils._get_wikitext2("train") == [
            "train-row-0",
            "train-row-1",
        ]
        return [torch.arange(seq_len, dtype=torch.long) for _ in range(num_samples)]

    monkeypatch.setattr(data_utils, "get_tokens", fake_get_tokens)

    class FakeTokenizer:
        name_or_path = "fake-tokenizer-artifact"

    observed_lr = {}
    fake_model = torch.nn.Linear(3, 2, bias=False).to(torch.bfloat16)

    runtime_cache_dir = tmp_path / "runtime-cache"

    def fake_get_ref_logits(args, analyzer, dataset, dataloader):
        metadata = eval_utils._reference_cache_metadata(
            args,
            analyzer,
            dataset,
            dataloader,
        )
        cache_tag = __import__("hashlib").sha256(
            json.dumps(metadata, sort_keys=True).encode()
        ).hexdigest()[:20]
        path = (
            Path(args.cache_dir)
            / "ref_logits"
            / (
                f"{args.model_name}_{dataset}_test_"
                f"{args.eval_seq_len}_{cache_tag}.cache"
            )
        )
        cached = eval_utils._load_reference_cache(str(path), metadata)
        assert cached is not None
        return cached, object()

    monkeypatch.setattr(eval_utils, "get_ref_logits", fake_get_ref_logits)

    def fake_pipeline_run(cfg):
        analyzer = SimpleNamespace(model=fake_model, tokenizer=FakeTokenizer())
        ref_logits, orig_lm_head = eval_utils.get_ref_logits(
            cfg,
            analyzer,
            "wikitext2",
            object(),
        )
        pipeline_mod._maybe_rotate(cfg, analyzer)
        key = static_cache.build_cache_key(cfg, 1)
        cached = static_cache.try_load(
            cfg.static_cache_path,
            key,
            1,
            0,
        )
        assert static_e2e._all_ranks_have_cache(cached is not None, 1)
        assert block_gd.layer_lr_for_schedule is layer_loop.layer_lr_for_schedule
        observed_lr["value"] = layer_loop.layer_lr_for_schedule(
            cfg.grad_lr,
            0,
            2,
            cfg.grad_lr_layer_base_ratio,
            cfg.grad_lr_layer_schedule,
            activation_aware=cfg.activation_aware_quantization_enabled,
        )
        save_path = os.path.join(
            cfg.tokens_cache_path,
            "fake_wikitext2_train_n4_sl8_seed1.pt",
        )
        data_utils.get_tokens(
            "wikitext2",
            "train",
            FakeTokenizer(),
            8,
            4,
            save_path,
            1,
        )
        # A second lookup must have exactly the same sampled-token identity.
        data_utils.get_tokens(
            "wikitext2",
            "train",
            FakeTokenizer(),
            8,
            4,
            save_path,
            1,
        )
        eval_utils.kl_ppl_eval(
            cfg,
            analyzer,
            orig_lm_head,
            {"wikitext2": object()},
            {"wikitext2": ref_logits},
        )

    monkeypatch.setattr(pipeline_mod, "run", fake_pipeline_run)

    def fake_ptq_main():
        cfg = config_mod.parse_cli()
        pipeline_mod.run(cfg)

    monkeypatch.setattr(ptq, "main", fake_ptq_main)

    # Build the exact immutable outputs of a completed warm run.  This makes
    # the integration test exercise the measured arm's O_EXCL/preflight,
    # real loader-hit checks, and warm-provenance comparisons.
    _, dataset_sha256, dataset_identity = _dataset_pin(dataset_dir)
    pinned_static_key = f"base_dataset{dataset_identity[:12]}"
    static_cache_file = Path(
        static_cache.cache_path(
            str(static_dir),
            pinned_static_key,
            1,
            0,
        )
    )
    static_cache_file.parent.mkdir(parents=True)
    torch.save({"saliency": [], "fisher": []}, static_cache_file)

    tokenizer_identity = artifact_identity(FakeTokenizer.name_or_path)
    scoped_token_path = Path(
        entrypoint._revision_scoped_token_path(
            str(old_token_path),
            dataset_identity=dataset_identity,
            tokenizer_identity=tokenizer_identity,
        )
    )
    scoped_token_path.write_bytes(b"fixed-token-cache")
    sampled_tokens = [
        torch.arange(8, dtype=torch.long)
        for _ in range(4)
    ]
    token_sha256 = entrypoint._token_tensor_identity(sampled_tokens)

    reference_tag = __import__("hashlib").sha256(
        json.dumps(reference_metadata, sort_keys=True).encode()
    ).hexdigest()[:20]
    reference_path = (
        runtime_cache_dir
        / "ref_logits"
        / f"fake_wikitext2_test_2048_{reference_tag}.cache"
    )
    reference_path.parent.mkdir(parents=True)
    reference_path.write_bytes(b"fixed-reference-cache")

    rotation_path = tmp_path / "rotation.json"
    rotation_state_sha256 = entrypoint._model_state_sha256(fake_model)
    rotation_payload = {
        "schema_version": 1,
        "experiment_case": "qwen3_4b_w4a16",
        "model_artifact_identity": artifact_identity("fake"),
        "rotate": True,
        "rotation_seed": 0,
        "optimized_rotation_path": None,
        "post_rotation_model_state_sha256": rotation_state_sha256,
        "all_rank_model_state_sha256": [rotation_state_sha256],
    }
    entrypoint._atomic_json_dump(rotation_payload, rotation_path)

    warm_path = (
        tmp_path
        / "qwen3_4b_w4a16"
        / "cache"
        / "warm_complete.json"
    )
    source = entrypoint._immutable_source(
        entrypoint._source_snapshot(Path.cwd())
    )
    warm_payload = {
        "schema_version": 1,
        "experiment_case": "qwen3_4b_w4a16",
        "protocol_config": {
            **entrypoint._COMMON_CASE_CONFIG,
            **entrypoint._CASE_PROTOCOLS["qwen3_4b_w4a16"]["config"],
        },
        "dataset": {
            "repo": "Salesforce/wikitext",
            "revision": entrypoint._DATASET_REVISION,
            "config": "wikitext-2-raw-v1",
            "parquet_dir": str(dataset_dir.resolve()),
            "sha256": dataset_sha256,
            "identity": dataset_identity,
        },
        "model_artifact_identity": artifact_identity("fake"),
        "source": source,
        "calibration_tokens": {
            "train": {
                "dataset": "wikitext2",
                "split": "train",
                "num_samples": 4,
                "seq_len": 8,
                "seed": 1,
                "sha256": token_sha256,
                "cache_path": str(scoped_token_path),
                "tokenizer_artifact_identity": tokenizer_identity,
                "cache_end": entrypoint._path_manifest(scoped_token_path),
            }
        },
        "rotation_fingerprint": entrypoint._path_manifest(rotation_path),
        "static_cache": {
            "key": pinned_static_key,
            "files": [
                {
                    "rank": 0,
                    **entrypoint._path_manifest(static_cache_file),
                }
            ],
        },
        "reference_cache": [
            {
                "dataset": "wikitext2",
                "path": str(reference_path.resolve()),
                "metadata_sha256": __import__("hashlib").sha256(
                    json.dumps(reference_metadata, sort_keys=True).encode()
                ).hexdigest(),
                "end": entrypoint._path_manifest(reference_path),
            }
        ],
    }
    entrypoint._atomic_json_dump(warm_payload, warm_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_schedule_ablation_variant.py",
            "--experiment_case",
            "qwen3_4b_w4a16",
            "--experiment_root",
            str(tmp_path.resolve()),
            "--schedule_ablation_variant",
            "legacy_cos2_scheduled",
            "--wikitext_parquet_dir",
            str(dataset_dir.resolve()),
            "--metrics_json_path",
            str(metrics_path.resolve()),
            "--rotation_fingerprint_path",
            str(rotation_path.resolve()),
            "--require_precomputed_caches",
            "--model",
            "fake",
            "--nsamples",
            "4",
            "--seq_len",
            "8",
            "--quant_stop_layer",
            "0",
            "--tokens_cache_path",
            str(token_dir),
            "--static_cache_path",
            str(static_dir),
            "--cache_dir",
            str(runtime_cache_dir),
        ],
    )

    entrypoint.main()

    expected_lr = schedule_ablation_lr(
        "legacy_cos2_scheduled",
        Config().grad_lr,
        0,
        2,
        Config().grad_lr_layer_base_ratio,
        Config().grad_lr_layer_schedule,
        activation_aware=False,
    )
    assert observed_lr["value"] == pytest.approx(expected_lr)
    assert len(seen_cache_paths) == 2
    assert seen_cache_paths[0] == seen_cache_paths[1]
    assert ".dataset-" in seen_cache_paths[0]
    assert ".tokenizer-" in seen_cache_paths[0]

    payload = json.loads(metrics_path.read_text())
    assert payload["schema_version"] == 4
    assert payload["experiment_case"] == "qwen3_4b_w4a16"
    assert payload["metrics"]["wikitext2"] == {
        "kl_raw": 0.054321987,
        "kl_x100": 5.4321987,
        "ppl": 13.456789,
    }
    assert payload["schedule_calls"][0]["effective_lr"] == pytest.approx(
        expected_lr
    )
    assert payload["dataset"]["calibration_tokens"]["train"]["sha256"]
    assert payload["static_cache"]["key"].startswith("base_dataset")
    assert payload["static_cache"]["runtime"]["local_hit"] is True
    assert payload["static_cache"]["runtime"]["all_ranks_hit"] is True
    assert payload["post_rotation_model"][
        "post_rotation_model_state_sha256"
    ]
    assert payload["warm_completion"]["manifest"]["sha256"]
    assert not (tmp_path / "metrics.json.lock").exists()
    assert payload["source"]["start"]["wrapper_sha256"] == entrypoint._sha256(
        Path(entrypoint.__file__).resolve()
    )
