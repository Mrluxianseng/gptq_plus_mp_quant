from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import process_args
from gptq_utils.gptq_plus_utils import BackwardSampleScheduler
from realq.config import Config, parse_cli
from realq.precompute import cache as cache_mod
from realq.refresh.block_gd import _SharedSampleScheduler
from utils import data_utils, model_utils, rotation_utils
from utils.reproducibility import configure_reproducibility


def _scheduler_trace(cls, seed: int) -> list[list[int]]:
    if cls is BackwardSampleScheduler:
        scheduler = cls(total_samples=8, chunk_size=2, seed=seed)
    else:
        scheduler = cls(n_total=8, chunk_size=2, seed=seed)
    # The first epoch is deliberately ordered. Cross the epoch boundary so
    # the seed-controlled shuffle is observable.
    return [scheduler.next_indices() for _ in range(8)]


def _rotation_draws(
    monkeypatch,
    *,
    calibration_seed: int,
    rotation_seed: int,
    refresh_seed: int,
) -> list[torch.Tensor]:
    draws: list[torch.Tensor] = []

    def fake_matrix(size, _mode, device="cuda", generator=None):
        del device
        value = torch.randint(0, 2, (size,), generator=generator)
        draws.append(value.clone())
        return value

    monkeypatch.setattr(rotation_utils, "get_orthogonal_matrix", fake_matrix)
    monkeypatch.setattr(rotation_utils, "rotate_embeddings", lambda *_args: None)
    monkeypatch.setattr(rotation_utils, "rotate_head", lambda *_args: None)
    monkeypatch.setattr(
        rotation_utils, "rotate_attention_mlp_inputs", lambda *_args: None
    )
    monkeypatch.setattr(
        rotation_utils, "rotate_attention_mlp_output", lambda *_args: None
    )
    monkeypatch.setattr(rotation_utils, "rotate_down_proj", lambda *_args: None)
    monkeypatch.setattr(rotation_utils, "rotate_ov_proj", lambda *_args, **_kw: None)
    monkeypatch.setattr(rotation_utils.memory_utils, "cleanup_memory", lambda: None)

    analyzer = SimpleNamespace(
        hidden_size=16,
        head_dim=8,
        get_layers=lambda: [object(), object()],
    )
    args = SimpleNamespace(
        seed=calibration_seed,
        rotation_seed=rotation_seed,
        refresh_seed=refresh_seed,
        optimized_rotation_path=None,
    )
    rotation_utils.rotate_model(args, analyzer)
    return draws


def _tensor_samples(seed: int) -> list[list[int]]:
    class Tokenizer:
        def __call__(self, _text, return_tensors):
            assert return_tensors == "pt"
            return SimpleNamespace(input_ids=torch.arange(256).view(1, -1))

    samples = data_utils._sample_concat_and_tokenize(
        ["a", "b", "c", "d"],
        Tokenizer(),
        seq_len=16,
        num_samples=4,
        seed=seed,
    )
    return [sample.tolist() for sample in samples]


def test_seed_domains_change_only_their_own_products(monkeypatch):
    base_calibration = _tensor_samples(11)
    base_rotation = _rotation_draws(
        monkeypatch,
        calibration_seed=11,
        rotation_seed=22,
        refresh_seed=33,
    )
    base_refresh = _scheduler_trace(_SharedSampleScheduler, 33)

    # Calibration sweep: token sequences change; algorithmic artifacts stay.
    assert _tensor_samples(12) != base_calibration
    swept_rotation = _rotation_draws(
        monkeypatch,
        calibration_seed=12,
        rotation_seed=22,
        refresh_seed=33,
    )
    assert all(torch.equal(a, b) for a, b in zip(base_rotation, swept_rotation))
    assert _scheduler_trace(_SharedSampleScheduler, 33) == base_refresh

    # Rotation sweep: only the generated rotation changes.
    assert _tensor_samples(11) == base_calibration
    changed_rotation = _rotation_draws(
        monkeypatch,
        calibration_seed=11,
        rotation_seed=23,
        refresh_seed=33,
    )
    assert any(not torch.equal(a, b) for a, b in zip(base_rotation, changed_rotation))
    assert _scheduler_trace(_SharedSampleScheduler, 33) == base_refresh

    # Refresh sweep: only the post-epoch sample order changes.
    assert _tensor_samples(11) == base_calibration
    stable_rotation = _rotation_draws(
        monkeypatch,
        calibration_seed=11,
        rotation_seed=22,
        refresh_seed=34,
    )
    assert all(torch.equal(a, b) for a, b in zip(base_rotation, stable_rotation))
    assert _scheduler_trace(_SharedSampleScheduler, 34) != base_refresh


def test_old_and_new_refresh_schedulers_are_seed_equivalent():
    for refresh_seed in (0, 1, 20260724):
        assert _scheduler_trace(
            BackwardSampleScheduler, refresh_seed
        ) == _scheduler_trace(_SharedSampleScheduler, refresh_seed)


def test_calibration_sampling_does_not_mutate_global_python_rng():
    random.seed(919)
    before = random.getstate()
    first = _tensor_samples(7)
    after = random.getstate()
    second = _tensor_samples(7)

    assert after == before
    assert second == first


def test_token_cache_hit_and_miss_are_identical_and_rng_local(tmp_path, monkeypatch):
    class Tokenizer:
        def __call__(self, _text, return_tensors):
            assert return_tensors == "pt"
            return SimpleNamespace(input_ids=torch.arange(256).view(1, -1))

    monkeypatch.setattr(
        data_utils,
        "_get_dataset",
        lambda _tokenizer, _dataset, _split: ["a", "b", "c", "d"],
    )
    cache_path = tmp_path / "tokens" / "calibration_seed7.pt"

    random.seed(1234)
    state = random.getstate()
    miss = data_utils.get_tokens(
        "wikitext2", "train", Tokenizer(), 16, 4, str(cache_path), seed=7
    )
    assert random.getstate() == state
    assert cache_path.is_file()
    assert not list(cache_path.parent.glob("*.tmp.*"))

    monkeypatch.setattr(
        data_utils,
        "_get_dataset",
        lambda *_args: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    hit = data_utils.get_tokens(
        "wikitext2", "train", Tokenizer(), 16, 4, str(cache_path), seed=7
    )
    assert random.getstate() == state
    assert all(torch.equal(left, right) for left, right in zip(miss, hit))


def _static_cache_cfg(model: Path, **overrides):
    values = {
        "model": str(model),
        "model_name": "tiny",
        "dataset": "wikitext2",
        "nsamples": 8,
        "seq_len": 16,
        "rotate": True,
        "optimized_rotation_path": None,
        "rotation_seed": 0,
        "refresh_seed": 0,
        "num_groups": 4,
        "saliency_clip_percentile": 0.99,
        "grad_hessian_topk": -1,
        "global_loss_bsz": 4,
        "seed": 7,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_static_cache_key_tracks_only_numerically_relevant_seed_domains(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"hidden_size": 16}')

    base = _static_cache_cfg(model)
    key = cache_mod.build_cache_key(base, world_size=2)
    assert key != cache_mod.build_cache_key(
        _static_cache_cfg(model, seed=8), world_size=2
    )
    assert key != cache_mod.build_cache_key(
        _static_cache_cfg(model, rotation_seed=1), world_size=2
    )
    # Refresh happens after the FP static pass.
    assert key == cache_mod.build_cache_key(
        _static_cache_cfg(model, refresh_seed=99), world_size=2
    )

    # With rotation disabled there is no rotation artifact to invalidate.
    no_rotate = cache_mod.build_cache_key(
        _static_cache_cfg(model, rotate=False, rotation_seed=0), world_size=2
    )
    assert no_rotate == cache_mod.build_cache_key(
        _static_cache_cfg(model, rotate=False, rotation_seed=99), world_size=2
    )

    # Optimized rotations ignore rotation_seed but track artifact mutation.
    optimized = tmp_path / "rotation.pt"
    optimized.write_bytes(b"rotation-a")
    optimized_key = cache_mod.build_cache_key(
        _static_cache_cfg(model, optimized_rotation_path=str(optimized)),
        world_size=2,
    )
    assert optimized_key == cache_mod.build_cache_key(
        _static_cache_cfg(
            model, optimized_rotation_path=str(optimized), rotation_seed=99
        ),
        world_size=2,
    )
    optimized.write_bytes(b"rotation-b-is-different")
    assert optimized_key != cache_mod.build_cache_key(
        _static_cache_cfg(model, optimized_rotation_path=str(optimized)),
        world_size=2,
    )


def test_refactored_static_cache_schema_invalidates_old_saliency_mean(
    tmp_path, monkeypatch
):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"hidden_size": 16}')
    cfg = _static_cache_cfg(model)

    assert cache_mod._CACHE_SCHEMA_VERSION == 5
    sum_key = cache_mod.build_cache_key(cfg, world_size=1)
    monkeypatch.setattr(cache_mod, "_CACHE_SCHEMA_VERSION", 4)
    mean_key = cache_mod.build_cache_key(cfg, world_size=1)
    assert sum_key != mean_key


def test_prepared_rotation_identity_excludes_calibration_seed(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"hidden_size": 16}')
    (model / "model.safetensors").write_bytes(b"weights-a")
    base = _static_cache_cfg(model=model, rotation_seed=5)
    changed_calibration = _static_cache_cfg(
        model=model, seed=999, rotation_seed=5
    )
    changed_rotation = _static_cache_cfg(
        model=model, seed=7, rotation_seed=6
    )

    assert model_utils.rotation_cache_identity(base) == (
        model_utils.rotation_cache_identity(changed_calibration)
    )
    assert model_utils.rotation_cache_tag(base) == (
        model_utils.rotation_cache_tag(changed_calibration)
    )
    assert model_utils.rotation_cache_identity(base) != (
        model_utils.rotation_cache_identity(changed_rotation)
    )

    source_identity = model_utils.source_model_cache_identity(base)
    (model / "model.safetensors").write_bytes(b"weights-b-is-different")
    assert source_identity != model_utils.source_model_cache_identity(base)


def test_cli_defaults_hold_algorithm_seeds_fixed_and_legacy_cache_is_seeded(
    tmp_path, monkeypatch
):
    cfg = parse_cli(["--model", "unused", "--seed", "123"])
    assert cfg.seed == 123
    assert cfg.rotation_seed == 0
    assert cfg.refresh_seed == 0

    monkeypatch.setattr(process_args, "init_logging", lambda *_args, **_kw: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ptq.py",
            "--model",
            "unused",
            "--exp",
            "seed-test",
            "--seed",
            "123",
            "--cache_dir",
            str(tmp_path / "cache"),
            "--output_dir",
            str(tmp_path / "out"),
        ],
    )
    args = process_args.parse_gen()
    assert args.rotation_seed == 0
    assert args.refresh_seed == 0
    assert args.tokens_cache_path.endswith("_blk2048_seed123.pt")
    assert args.saliency_cache_path.endswith("_g4_salsumv1")


def test_reproducibility_helper_replays_python_numpy_and_torch_rngs():
    configure_reproducibility(77)
    first = (
        random.random(),
        float(np.random.rand()),
        torch.rand(4),
    )
    configure_reproducibility(77)
    second = (
        random.random(),
        float(np.random.rand()),
        torch.rand(4),
    )

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] in {":4096:8", ":16:8"}


def test_entrypoints_set_cublas_workspace_before_importing_torch():
    root = Path(__file__).resolve().parents[1]
    for relative in ("ptq.py", "realq/ptq.py"):
        source = (root / relative).read_text()
        assert source.index("CUBLAS_WORKSPACE_CONFIG") < source.index("\nimport torch")
