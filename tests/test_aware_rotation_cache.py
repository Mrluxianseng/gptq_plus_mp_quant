from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from realq import akv
from realq.config import Config
from realq.precompute import cache as cache_mod
from realq.runner import layer_loop
from tools.create_tiny_alignment_model import build_tokenizer
from utils import model_utils, rotation_utils


def _cache_cfg(model, **overrides):
    values = {
        "model": str(model),
        "model_name": "same-basename",
        "dataset": "wikitext2",
        "nsamples": 8,
        "seq_len": 32,
        "rotate": True,
        "optimized_rotation_path": None,
        "num_groups": 4,
        "saliency_clip_percentile": 0.99,
        "grad_hessian_topk": -1,
        "global_loss_bsz": 4,
        "seed": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_pipeline_configures_aware_quant_only_after_fp_stage0(monkeypatch):
    from realq import pipeline

    events = []
    analyzer = SimpleNamespace(model=SimpleNamespace(), tokenizer=object())
    cfg = Config(
        model="unused",
        skip_eval=True,
        rotate=False,
        fsdp=False,
        act_quant_aware_gptq=True,
        a_bits=4,
        tokens_cache_path=None,
        save_qmodel_path=None,
        lm_eval=False,
    )

    monkeypatch.setattr(pipeline.model_utils, "ModelAnalyzer", lambda *_args, **_kw: analyzer)
    monkeypatch.setattr(pipeline, "_maybe_rotate", lambda *_args: events.append("rotate"))

    def precompute(_cfg, current):
        assert current is analyzer
        assert not getattr(current.model, "aware_enabled", False)
        events.append("stage0_fp")
        return SimpleNamespace(fisher=[], saliency=[])

    def setup_aware(current, _cfg):
        current.model.aware_enabled = True
        events.append("aware_student")

    def quantize(_cfg, current, _static, _tokens):
        assert current.model.aware_enabled
        events.append("weight_quant")

    monkeypatch.setattr(pipeline.precompute, "run", precompute)
    monkeypatch.setattr(pipeline.akv, "setup_aware_pre_quant", setup_aware)
    monkeypatch.setattr(
        pipeline.akv,
        "setup_unaware_post_quant",
        lambda *_args: events.append("unaware_runtime"),
    )
    monkeypatch.setattr(pipeline.data_utils, "get_tokens", lambda *_args, **_kw: [])
    monkeypatch.setattr(pipeline.runner, "quantize_all_layers", quantize)
    monkeypatch.setattr(
        pipeline.parallel_env,
        "is_dist_available_and_initialized",
        lambda: False,
    )

    pipeline.run(cfg)

    assert events == [
        "rotate",
        "stage0_fp",
        "aware_student",
        "weight_quant",
        "unaware_runtime",
    ]


def test_fp_replay_temporarily_disables_aware_quant_and_restores_student(monkeypatch):
    layer = SimpleNamespace(aware_enabled=True)
    seen = []
    sentinel = torch.tensor([3.0])

    @contextmanager
    def fake_disable(current):
        assert current.aware_enabled
        current.aware_enabled = False
        try:
            yield
        finally:
            current.aware_enabled = True

    def fake_replay(current, state, bsz, inps):
        assert current is layer
        assert not current.aware_enabled
        seen.append((state, bsz, inps))
        return sentinel

    monkeypatch.setattr(layer_loop, "disable_fp_path_quant", fake_disable)
    monkeypatch.setattr(layer_loop.streams, "replay_layer", fake_replay)

    state = object()
    inps = torch.tensor([1.0])
    out = layer_loop._replay_fp_layer(layer, state, bsz=1, inps=inps)

    assert out is sentinel
    assert layer.aware_enabled
    assert seen == [(state, 1, inps)]


def test_aware_setup_does_not_double_wrap_rotation_wrappers(monkeypatch):
    model = SimpleNamespace(_gptqplus_rotation_wrappers_installed=True)
    analyzer = SimpleNamespace(model=model)

    def fail_double_wrap(_analyzer):
        raise AssertionError("rotation-installed ActQuantWrapper was wrapped twice")

    monkeypatch.setattr(akv.quant_utils, "add_actquant", fail_double_wrap)
    akv.install_actquant_wrappers(analyzer)

    assert model._realq_actquant_wrappers_installed is True


def test_static_cache_key_tracks_exact_model_and_rotation_artifacts(tmp_path):
    model_a = tmp_path / "left" / "same-model"
    model_b = tmp_path / "right" / "same-model"
    model_a.mkdir(parents=True)
    model_b.mkdir(parents=True)
    (model_a / "config.json").write_text('{"hidden_size": 4}')
    (model_b / "config.json").write_text('{"hidden_size": 4}')

    key_a = cache_mod.build_cache_key(_cache_cfg(model_a), world_size=2)
    key_b = cache_mod.build_cache_key(_cache_cfg(model_b), world_size=2)
    assert key_a != key_b

    (model_a / "config.json").write_text('{"hidden_size": 8}')
    key_a_modified = cache_mod.build_cache_key(_cache_cfg(model_a), world_size=2)
    assert key_a_modified != key_a

    rotation_a = tmp_path / "rot-a" / "rotation.pt"
    rotation_b = tmp_path / "rot-b" / "rotation.pt"
    rotation_a.parent.mkdir()
    rotation_b.parent.mkdir()
    rotation_a.write_bytes(b"rotation-a")
    rotation_b.write_bytes(b"rotation-a")
    rot_key_a = cache_mod.build_cache_key(
        _cache_cfg(model_a, optimized_rotation_path=str(rotation_a)),
        world_size=2,
    )
    rot_key_b = cache_mod.build_cache_key(
        _cache_cfg(model_a, optimized_rotation_path=str(rotation_b)),
        world_size=2,
    )
    assert rot_key_a != rot_key_b

    rotation_a.write_bytes(b"rotation-a-replaced")
    rot_key_a_modified = cache_mod.build_cache_key(
        _cache_cfg(model_a, optimized_rotation_path=str(rotation_a)),
        world_size=2,
    )
    assert rot_key_a_modified != rot_key_a


def test_static_fp_cache_ignores_student_aware_settings_but_tracks_seed(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    base = _cache_cfg(
        model,
        a_bits=16,
        v_bits=16,
        k_bits=16,
        act_quant_aware_gptq=False,
        k_cache_quant_aware_gptq=False,
    )
    aware = _cache_cfg(
        model,
        a_bits=4,
        v_bits=4,
        k_bits=4,
        act_quant_aware_gptq=True,
        k_cache_quant_aware_gptq=True,
    )
    assert cache_mod.build_cache_key(base, 1) == cache_mod.build_cache_key(aware, 1)
    assert cache_mod.build_cache_key(base, 1) != cache_mod.build_cache_key(
        _cache_cfg(model, seed=1), 1,
    )


def test_cloned_lm_head_is_not_reported_as_actually_tied():
    embedding = torch.nn.Parameter(torch.randn(5, 4))
    cloned_head = torch.nn.Parameter(embedding.detach().clone())
    tied_head = embedding
    shared_view = torch.nn.Parameter(embedding.detach())

    assert not model_utils.parameters_share_storage(embedding, cloned_head)
    assert model_utils.parameters_share_storage(embedding, tied_head)
    assert model_utils.parameters_share_storage(embedding, shared_view)


def test_untied_clone_runs_all_global_rotation_steps(monkeypatch):
    embedding = torch.nn.Parameter(torch.randn(5, 4))
    head = torch.nn.Parameter(embedding.detach().clone())
    layer = object()
    analyzer = SimpleNamespace(
        tie_word_embeddings=model_utils.parameters_share_storage(embedding, head),
        hidden_size=4,
        head_dim=2,
        get_layers=lambda: [layer],
    )
    args = SimpleNamespace(optimized_rotation_path=None, seed=0)
    calls = []

    monkeypatch.setattr(
        rotation_utils,
        "get_orthogonal_matrix",
        lambda size, mode, device="cuda", generator=None: (size, mode),
    )
    monkeypatch.setattr(
        rotation_utils, "rotate_embeddings",
        lambda current, _r1: calls.append(("embeddings", current.tie_word_embeddings)),
    )
    monkeypatch.setattr(
        rotation_utils, "rotate_head",
        lambda current, _r1: calls.append(("head", current.tie_word_embeddings)),
    )
    monkeypatch.setattr(
        rotation_utils, "rotate_attention_mlp_inputs",
        lambda current, current_layer, _r1: calls.append(("inputs", current_layer)),
    )
    monkeypatch.setattr(
        rotation_utils, "rotate_attention_mlp_output",
        lambda current, current_layer, _r1: calls.append(("outputs", current_layer)),
    )
    monkeypatch.setattr(
        rotation_utils, "rotate_down_proj",
        lambda current, current_layer: calls.append(("down", current_layer)),
    )
    monkeypatch.setattr(
        rotation_utils, "rotate_ov_proj",
        lambda current, current_layer, R2=None: calls.append(("ov", current_layer)),
    )
    monkeypatch.setattr(rotation_utils.memory_utils, "cleanup_memory", lambda: None)

    rotation_utils.rotate_model(args, analyzer)

    assert calls == [
        ("embeddings", False),
        ("head", False),
        ("inputs", layer),
        ("outputs", layer),
        ("down", layer),
        ("ov", layer),
    ]


@pytest.mark.parametrize(
    ("model_family", "config_cls", "model_cls"),
    [
        ("llama", LlamaConfig, LlamaForCausalLM),
        ("qwen3", Qwen3Config, Qwen3ForCausalLM),
    ],
)
@pytest.mark.parametrize("tie_word_embeddings", [False, True])
def test_tiny_quarot_preserves_full_model_logits(
    model_family,
    config_cls,
    model_cls,
    tie_word_embeddings,
):
    if not torch.cuda.is_available():
        pytest.skip("production rotation utilities require CUDA")

    torch.manual_seed(3)
    config_kwargs = dict(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=tie_word_embeddings,
    )
    if model_family == "qwen3":
        config_kwargs["head_dim"] = 16
    config = config_cls(**config_kwargs)
    config.architectures = [model_cls.__name__]
    model = model_cls(config).double().eval().cuda()
    analyzer = model_utils.ModelAnalyzer(
        model,
        seq_len=8,
        tokenizer_source=build_tokenizer(64),
    )
    assert analyzer.source_tie_word_embeddings is tie_word_embeddings
    assert analyzer.tie_word_embeddings is False
    assert not model_utils.parameters_share_storage(
        analyzer.get_embed_layer().weight,
        analyzer.get_lm_head().weight,
    )
    input_ids = torch.randint(4, 64, (1, 8), device="cuda")
    with torch.no_grad():
        before = model(input_ids).logits.cpu()

    rotation_utils.prepare_model_for_rotated_quantization(
        SimpleNamespace(
            rotate=True,
            optimized_rotation_path=None,
            rotation_seed=5,
        ),
        analyzer,
    )
    model.cuda().eval()
    with torch.no_grad():
        after = model(input_ids).logits.cpu()

    # Double precision isolates the mathematical equivalence from bf16
    # requantization noise while still executing every production transform
    # and online Hadamard wrapper.
    torch.testing.assert_close(before, after, rtol=1e-6, atol=1e-7)
