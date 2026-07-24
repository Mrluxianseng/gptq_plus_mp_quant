from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from gptq_utils.quant_aware_utils import (  # noqa: E402
    configure_activation_quantizers_for_gptq,
)
from realq import akv  # noqa: E402
from realq.config import Config, parse_cli  # noqa: E402
from utils import checkpoint_utils, hadamard_utils, quant_utils, rotation_utils  # noqa: E402


class _RuntimeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = quant_utils.ActQuantWrapper(
            torch.nn.Linear(4, 4, bias=False)
        )
        self.v_proj = quant_utils.ActQuantWrapper(
            torch.nn.Linear(4, 4, bias=False)
        )

    def forward(self, x):
        return self.v_proj(self.q_proj(x))


class _Tokenizer:
    def __init__(self, vocab):
        self._vocab = vocab
        self.special_tokens_map = {"eos_token": "</s>"}

    def get_vocab(self):
        return self._vocab


def _runtime_cfg(**overrides):
    values = dict(
        model="base/model",
        rotate=False,
        a_bits=4,
        a_groupsize=2,
        a_asym=False,
        a_clip_ratio=0.9,
        v_bits=4,
        v_groupsize=-1,
        v_asym=True,
        v_clip_ratio=0.8,
        k_bits=4,
        k_groupsize=-1,
        k_asym=False,
        k_clip_ratio=0.9,
        act_quant_aware_gptq=False,
        k_cache_quant_aware_gptq=False,
    )
    values.update(overrides)
    return Config(**values)


def _rope(q, k):
    return q + 0.25, k - 0.5


def apply_rotary_pos_emb(q, k):
    return _rope(q, k)


class _Attention(torch.nn.Module):
    def forward(self, q, k):
        return apply_rotary_pos_emb(q, k)


class _AttentionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()


class _AttentionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = _AttentionLayer()


class _AttentionAnalyzer:
    head_dim = 4

    def __init__(self):
        self.model = _AttentionModel()

    def get_layers(self):
        return [self.model.layer]

    def get_quantizable_modules(self, _layer):
        return {}


def test_checkpoint_is_weights_only_safe_and_restores_exact_a_v_runtime(
    tmp_path, monkeypatch
):
    torch.manual_seed(7)
    source = _RuntimeModel()
    cfg = _runtime_cfg(quantizer_inner_fastpath=True)
    configure_activation_quantizers_for_gptq(cfg, source)
    x = torch.tensor([[[1.0, -2.0, 3.5, -4.0]]])
    expected = source(x)

    # A bare basename used to call os.makedirs("") in the legacy saver.
    monkeypatch.chdir(tmp_path)
    checkpoint_utils.save_quantized_checkpoint("quantized.pt", source, cfg)

    raw = torch.load("quantized.pt", map_location="cpu", weights_only=True)
    assert raw["format"] == checkpoint_utils.CHECKPOINT_FORMAT
    assert "w_quantizers" not in raw
    assert not any(
        key.endswith((".quantizer.maxq", ".quantizer.scale", ".quantizer.zero"))
        for key in raw["model"]
    )

    loaded = checkpoint_utils.load_quantized_checkpoint("quantized.pt")
    restored_cfg = _runtime_cfg(
        a_bits=16,
        a_groupsize=-1,
        a_clip_ratio=1.0,
        v_bits=16,
        v_clip_ratio=1.0,
        k_bits=16,
        k_clip_ratio=1.0,
        act_quant_aware_gptq=True,
        k_cache_quant_aware_gptq=False,
        w_bits=16,
        quantizer_inner_fastpath=False,
    )
    assert checkpoint_utils.apply_runtime_manifest(restored_cfg, loaded)
    assert restored_cfg.a_bits == 4
    assert restored_cfg.a_groupsize == 2
    assert restored_cfg.v_bits == 4
    assert restored_cfg.v_asym is True
    assert restored_cfg.act_quant_aware_gptq is False
    assert restored_cfg.w_bits == 4
    assert raw["weight_quantization"]["quantizer_inner_fastpath"] is True
    assert restored_cfg.quantizer_inner_fastpath is True

    target = _RuntimeModel()
    configure_activation_quantizers_for_gptq(restored_cfg, target)
    checkpoint_utils.load_model_state(target, loaded)
    torch.testing.assert_close(target(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("aware", [False, True])
def test_manifest_reconstructs_exact_k_cache_behavior_for_aware_and_unaware(
    tmp_path, aware
):
    cfg = _runtime_cfg(k_cache_quant_aware_gptq=aware)
    source = _RuntimeModel()
    path = tmp_path / f"k-aware-{aware}.pt"
    checkpoint_utils.save_quantized_checkpoint(path, source, cfg)

    restored_cfg = _runtime_cfg(
        k_bits=16,
        k_clip_ratio=1.0,
        k_cache_quant_aware_gptq=False,
    )
    loaded = checkpoint_utils.load_quantized_checkpoint(path)
    checkpoint_utils.apply_runtime_manifest(restored_cfg, loaded)

    original = rotation_utils.QKRotationWrapper(
        _rope,
        head_dim=4,
        k_bits=cfg.k_bits,
        k_groupsize=cfg.k_groupsize,
        k_sym=not cfg.k_asym,
        k_clip_ratio=cfg.k_clip_ratio,
    )
    reconstructed = rotation_utils.QKRotationWrapper(
        _rope,
        head_dim=4,
        k_bits=restored_cfg.k_bits,
        k_groupsize=restored_cfg.k_groupsize,
        k_sym=not restored_cfg.k_asym,
        k_clip_ratio=restored_cfg.k_clip_ratio,
    )
    q = torch.tensor([[[[1.0, 2.0, -3.0, 4.0]]]])
    k = torch.tensor([[[[8.0, -1.0, 0.0, 2.0]]]])
    expected_q, expected_k = original(q, k)
    actual_q, actual_k = reconstructed(q, k)
    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
    assert restored_cfg.k_cache_quant_aware_gptq is aware


@pytest.mark.parametrize("aware", [False, True])
def test_akv_setup_roundtrip_reinstalls_k_forward_patch(tmp_path, aware):
    cfg = _runtime_cfg(
        a_bits=16,
        a_clip_ratio=1.0,
        v_bits=16,
        v_clip_ratio=1.0,
        k_cache_quant_aware_gptq=aware,
    )
    source = _AttentionAnalyzer()
    akv.setup_aware_pre_quant(source, cfg)
    akv.setup_unaware_post_quant(source, cfg)
    path = tmp_path / f"k-patch-{aware}.pt"
    checkpoint_utils.save_quantized_checkpoint(path, source.model, cfg)

    restored_cfg = _runtime_cfg(
        a_bits=16,
        a_clip_ratio=1.0,
        v_bits=16,
        v_clip_ratio=1.0,
        k_bits=16,
        k_clip_ratio=1.0,
        k_cache_quant_aware_gptq=False,
    )
    checkpoint = checkpoint_utils.load_quantized_checkpoint(path)
    checkpoint_utils.apply_runtime_manifest(restored_cfg, checkpoint)
    target = _AttentionAnalyzer()
    checkpoint_utils.load_model_state(target.model, checkpoint)
    akv.setup_aware_pre_quant(target, restored_cfg)
    akv.setup_unaware_post_quant(target, restored_cfg)

    q = torch.tensor([[[[1.0, 2.0, -3.0, 4.0]]]])
    k = torch.tensor([[[[8.0, -1.0, 0.0, 2.0]]]])
    expected = source.model.layer.self_attn(q, k)
    actual = target.model.layer.self_attn(q, k)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    wrappers = [
        module
        for module in target.model.layer.self_attn.modules()
        if isinstance(module, rotation_utils.QKRotationWrapper)
    ]
    assert len(wrappers) == 1
    assert wrappers[0].k_bits == 4
    assert wrappers[0].k_clip_ratio == 0.9
    assert restored_cfg.k_cache_quant_aware_gptq is aware


def test_loader_accepts_legacy_object_payload_under_pytorch_26(tmp_path):
    model = _RuntimeModel()
    legacy_quantizer = quant_utils.WeightQuantizer()
    legacy_quantizer.configure(bits=4, perchannel=True, sym=True)
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "w_quantizers": {"layer": legacy_quantizer},
        },
        path,
    )

    with pytest.raises(RuntimeError, match="allow_unsafe_legacy_checkpoint"):
        checkpoint_utils.load_quantized_checkpoint(path)
    loaded = checkpoint_utils.load_quantized_checkpoint(
        path, allow_unsafe_legacy=True
    )
    assert loaded["format_version"] == 0
    assert loaded["runtime_quantization"] is None
    assert not checkpoint_utils.apply_runtime_manifest(_runtime_cfg(), loaded)


def test_unsafe_legacy_opt_in_is_an_explicit_cli_flag():
    assert parse_cli([]).allow_unsafe_legacy_checkpoint is False
    assert (
        parse_cli(["--allow_unsafe_legacy_checkpoint"])
        .allow_unsafe_legacy_checkpoint
        is True
    )


def test_checkpoint_rejects_incomplete_or_invalid_runtime_manifest(tmp_path):
    model = _RuntimeModel()
    path = tmp_path / "bad.pt"
    torch.save(
        {
            "format": checkpoint_utils.CHECKPOINT_FORMAT,
            "format_version": checkpoint_utils.CHECKPOINT_VERSION,
            "model": model.state_dict(),
            "runtime_quantization": {"a_bits": 4},
        },
        path,
    )
    with pytest.raises(ValueError, match="incomplete"):
        checkpoint_utils.load_quantized_checkpoint(path)


def test_checkpoint_rejects_non_boolean_inner_fastpath_provenance(tmp_path):
    model = _RuntimeModel()
    cfg = _runtime_cfg()
    path = tmp_path / "bad-fastpath-provenance.pt"
    checkpoint_utils.save_quantized_checkpoint(path, model, cfg)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["weight_quantization"]["quantizer_inner_fastpath"] = "false"
    torch.save(payload, path)

    with pytest.raises(
        ValueError, match="'quantizer_inner_fastpath' must be bool"
    ):
        checkpoint_utils.load_quantized_checkpoint(path)

    target_cfg = _runtime_cfg(a_bits=16, a_clip_ratio=1.0)
    with pytest.raises(
        ValueError, match="'quantizer_inner_fastpath' must be bool"
    ):
        checkpoint_utils.apply_runtime_manifest(target_cfg, payload)
    assert target_cfg.a_bits == 16


def test_artifact_identity_strictly_checks_source_rotation_tokenizer_and_dtype(
    tmp_path
):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "config.json").write_text('{"model_type":"tiny"}')
    cfg = _runtime_cfg(model=str(source_dir), rotation_seed=9)
    model = _RuntimeModel()
    tokenizer = _Tokenizer({"</s>": 0, "hello": 1})
    path = tmp_path / "identity.pt"
    checkpoint_utils.save_quantized_checkpoint(path, model, cfg, tokenizer)
    loaded = checkpoint_utils.load_quantized_checkpoint(path)

    assert checkpoint_utils.validate_artifact_identity(cfg, loaded)
    assert checkpoint_utils.validate_artifact_identity(
        cfg, loaded, model=model, tokenizer=tokenizer
    )

    wrong_rotation = _runtime_cfg(model=str(source_dir), rotation_seed=10)
    with pytest.raises(ValueError, match="artifact identity"):
        checkpoint_utils.validate_artifact_identity(wrong_rotation, loaded)

    other_source = tmp_path / "other"
    other_source.mkdir()
    (other_source / "config.json").write_text('{"model_type":"tiny"}')
    wrong_source = _runtime_cfg(model=str(other_source), rotation_seed=9)
    with pytest.raises(ValueError, match="artifact identity"):
        checkpoint_utils.validate_artifact_identity(wrong_source, loaded)

    with pytest.raises(ValueError, match="artifact identity"):
        checkpoint_utils.validate_artifact_identity(
            cfg,
            loaded,
            model=model,
            tokenizer=_Tokenizer({"</s>": 0, "different": 1}),
        )
    with pytest.raises(ValueError, match="artifact identity"):
        checkpoint_utils.validate_artifact_identity(
            cfg,
            loaded,
            model=_RuntimeModel().double(),
            tokenizer=tokenizer,
        )


def test_act_quant_dynamic_buffers_never_enter_state_dict():
    quantizer = quant_utils.ActQuantizer()
    quantizer.configure(bits=4, groupsize=-1, sym=True, clip_ratio=0.9)
    quantizer.find_params(torch.tensor([[1.0, -2.0, 3.0, -4.0]]))
    assert quantizer.scale.numel() == 4
    assert quantizer.state_dict() == {}


def test_realq_load_skips_quantization_and_saves_only_after_runtime_restore(
    monkeypatch, tmp_path
):
    from realq import pipeline

    events = []
    analyzer = SimpleNamespace(model=torch.nn.Linear(2, 2), tokenizer=object())
    cfg = Config(
        model="unused",
        rotate=False,
        skip_eval=True,
        load_qmodel_path=str(tmp_path / "input.pt"),
        save_qmodel_path=str(tmp_path / "output.pt"),
    )
    checkpoint = {
        "model": analyzer.model.state_dict(),
        "runtime_quantization": checkpoint_utils.build_runtime_manifest(cfg)[
            "runtime_quantization"
        ],
        "weight_quantization": {},
    }

    monkeypatch.setattr(
        pipeline.checkpoint_utils,
        "load_quantized_checkpoint",
        lambda _path, **_kwargs: events.append("read_manifest") or checkpoint,
    )
    monkeypatch.setattr(
        pipeline.checkpoint_utils,
        "apply_runtime_manifest",
        lambda *_args: events.append("apply_manifest") or True,
    )
    monkeypatch.setattr(
        pipeline.model_utils,
        "ModelAnalyzer",
        lambda *_args, **_kwargs: events.append("load_base") or analyzer,
    )
    monkeypatch.setattr(
        pipeline,
        "_prepare_loaded_runtime_wrappers",
        lambda *_args: events.append("prepare_wrappers"),
    )
    monkeypatch.setattr(
        pipeline.checkpoint_utils,
        "load_model_state",
        lambda *_args: events.append("load_weights"),
    )
    monkeypatch.setattr(
        pipeline.akv,
        "install_actquant_wrappers",
        lambda *_args: events.append("ensure_act_wrappers"),
    )
    monkeypatch.setattr(
        pipeline.akv,
        "setup_aware_pre_quant",
        lambda *_args: events.append("restore_aware_runtime"),
    )
    monkeypatch.setattr(
        pipeline.akv,
        "setup_unaware_post_quant",
        lambda *_args: events.append("restore_unaware_runtime"),
    )

    def save_after_runtime(*_args):
        assert events[-2:] == [
            "restore_aware_runtime",
            "restore_unaware_runtime",
        ]
        events.append("save")

    monkeypatch.setattr(
        pipeline.checkpoint_utils,
        "save_quantized_checkpoint",
        save_after_runtime,
    )
    monkeypatch.setattr(
        pipeline.precompute,
        "run",
        lambda *_args: pytest.fail("load path reran static precompute"),
    )
    monkeypatch.setattr(
        pipeline.runner,
        "quantize_all_layers",
        lambda *_args: pytest.fail("load path requantized weights"),
    )
    monkeypatch.setattr(pipeline.parallel_env, "is_main", lambda: True)
    monkeypatch.setattr(
        pipeline.parallel_env,
        "is_dist_available_and_initialized",
        lambda: False,
    )

    pipeline.run(cfg)
    assert events == [
        "read_manifest",
        "apply_manifest",
        "load_base",
        "prepare_wrappers",
        "ensure_act_wrappers",
        "load_weights",
        "restore_aware_runtime",
        "restore_unaware_runtime",
        "save",
    ]
