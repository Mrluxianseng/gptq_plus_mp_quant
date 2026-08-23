from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from experiments.efficientqat_compare.akv_unaware import (
    PostRoPEKQuantWrapper,
    assert_unaware_post_quant_no_rotation,
    install_unaware_av_and_pure_post_rope_k,
    setup_unaware_post_quant_no_rotation,
)
from experiments.efficientqat_compare.materialize import (
    SOURCE_QUANTIZER,
    decode_efficientqat_weight_cpu,
    load_materialization_manifest,
    materialize_efficientqat_model,
    materialize_packed_checkpoint,
    unpack_packed_dim0,
    unpack_packed_dim1,
    validate_materialized_model,
    write_materialization_manifest,
)
from utils import quant_utils, rotation_utils


def _pack_dim0(values: torch.Tensor, bits: int) -> torch.Tensor:
    lanes = 32 // bits
    words = torch.zeros(
        (math.ceil(values.shape[0] / lanes), values.shape[1]),
        dtype=torch.int64,
    )
    for row in range(values.shape[0]):
        words[row // lanes] |= (
            values[row].to(torch.int64) << ((row % lanes) * bits)
        )
    return (words & 0xFFFFFFFF).to(torch.int32)


def _pack_dim1(values: torch.Tensor, bits: int) -> torch.Tensor:
    lanes = 32 // bits
    words = torch.zeros(
        (values.shape[0], math.ceil(values.shape[1] / lanes)),
        dtype=torch.int64,
    )
    for column in range(values.shape[1]):
        words[:, column // lanes] |= (
            values[:, column].to(torch.int64)
            << ((column % lanes) * bits)
        )
    return (words & 0xFFFFFFFF).to(torch.int32)


class _PackedLinear(nn.Module):
    def __init__(
        self,
        codes: torch.Tensor,
        zeros: torch.Tensor,
        scales: torch.Tensor,
        *,
        bits: int,
        group_size: int,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.infeatures = codes.shape[0]
        self.outfeatures = codes.shape[1]
        self.register_buffer("qweight", _pack_dim0(codes, bits))
        self.register_buffer("qzeros", _pack_dim1(zeros, bits))
        self.register_parameter(
            "scales", nn.Parameter(scales.clone(), requires_grad=False)
        )
        self.register_buffer(
            "g_idx",
            torch.arange(self.infeatures, dtype=torch.int32) // group_size,
        )
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.clone())


class _PackedModel(nn.Module):
    def __init__(self, first: nn.Module, second: nn.Module) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict({"q_proj": first}),
                nn.ModuleDict({"down_proj": second}),
            ]
        )
        # A normal linear must not be counted or changed.
        self.lm_head = nn.Linear(3, 5, bias=False)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_cpu_unpack_matches_original_uint_codes_for_signed_int32_words(bits):
    torch.manual_seed(bits)
    maxq = (1 << bits) - 1
    dim0 = torch.randint(0, maxq + 1, (23, 13), dtype=torch.int64)
    dim1 = torch.randint(0, maxq + 1, (5, 27), dtype=torch.int64)
    packed0 = _pack_dim0(dim0, bits)
    packed1 = _pack_dim1(dim1, bits)

    # Ensure at least one test covers words whose signed int32 view is
    # negative; the decoder must recover the original uint32 bit pattern.
    if bits in (2, 4, 8):
        packed0[0, 0] = torch.tensor(-1, dtype=torch.int32)
        dim0[: 32 // bits, 0] = maxq

    assert torch.equal(
        unpack_packed_dim0(
            packed0, bits=bits, logical_rows=dim0.shape[0]
        ),
        dim0,
    )
    assert torch.equal(
        unpack_packed_dim1(
            packed1, bits=bits, logical_columns=dim1.shape[1]
        ),
        dim1,
    )


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_materialize_replaces_all_packed_modules_with_bf16_linears(bits):
    torch.manual_seed(bits + 10)
    in_features, out_features, group_size = 20, 13, 5
    groups = in_features // group_size
    maxq = (1 << bits) - 1
    codes = torch.randint(
        0, maxq + 1, (in_features, out_features), dtype=torch.int64
    )
    zeros = torch.randint(
        0, maxq + 1, (groups, out_features), dtype=torch.int64
    )
    scales = torch.rand((groups, out_features), dtype=torch.float32) + 0.01
    bias = torch.randn(out_features, dtype=torch.float32)
    expected = (
        (
            codes.reshape(groups, group_size, out_features).float()
            - zeros[:, None, :].float()
        )
        * scales[:, None, :]
    ).reshape(in_features, out_features).T.contiguous().to(torch.bfloat16)

    model = _PackedModel(
        _PackedLinear(
            codes,
            zeros,
            scales,
            bits=bits,
            group_size=group_size,
            bias=bias,
        ),
        _PackedLinear(
            codes,
            zeros,
            scales,
            bits=bits,
            group_size=group_size,
        ),
    )
    original_head = model.lm_head
    names = ("layers.0.q_proj", "layers.1.down_proj")
    manifest = materialize_efficientqat_model(
        model,
        expected_count=2,
        expected_bits=bits,
        expected_group_size=group_size,
        expected_module_names=names,
        provenance={"base_model": "synthetic", "weight_bits": bits},
    )

    assert model.lm_head is original_head
    assert manifest["module_count"] == 2
    assert manifest["module_names"] == list(names)
    assert manifest["source_quantizer"] == SOURCE_QUANTIZER
    for name in names:
        module = model.get_submodule(name)
        assert type(module) is nn.Linear
        assert module.weight.device.type == "cpu"
        assert module.weight.dtype == torch.bfloat16
        assert module.weight.requires_grad is False
        assert torch.equal(module.weight, expected)

    assert model.layers[0]["q_proj"].bias is not None
    assert torch.equal(
        model.layers[0]["q_proj"].bias,
        bias.to(torch.bfloat16),
    )
    assert model.layers[1]["down_proj"].bias is None
    validate_materialized_model(model, manifest, expected_count=2)


def test_materialization_manifest_roundtrip_and_hash_gate(tmp_path):
    bits, group_size = 4, 2
    codes = torch.tensor(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        dtype=torch.int64,
    )
    zeros = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
    scales = torch.full((2, 3), 0.25)
    model = _PackedModel(
        _PackedLinear(
            codes, zeros, scales, bits=bits, group_size=group_size
        ),
        _PackedLinear(
            codes, zeros, scales, bits=bits, group_size=group_size
        ),
    )
    manifest = materialize_efficientqat_model(
        model,
        expected_count=2,
        expected_bits=bits,
        expected_group_size=group_size,
    )
    path = tmp_path / "efficientqat_manifest.json"
    write_materialization_manifest(path, manifest)
    loaded = load_materialization_manifest(path)
    assert json.loads(path.read_text()) == loaded
    validate_materialized_model(model, loaded, expected_count=2)

    bad_source = dict(loaded)
    bad_source["source_quantizer"] = "realq_w_clip"
    with pytest.raises(ValueError, match="source quantizer"):
        validate_materialized_model(model, bad_source, expected_count=2)

    with torch.no_grad():
        model.layers[0]["q_proj"].weight[0, 0] += torch.tensor(
            1, dtype=torch.bfloat16
        )
    with pytest.raises(ValueError, match="weight hash"):
        validate_materialized_model(model, loaded, expected_count=2)


def test_materialization_count_and_scale_fail_before_any_replacement():
    codes = torch.zeros((4, 3), dtype=torch.int64)
    zeros = torch.zeros((2, 3), dtype=torch.int64)
    scales = torch.ones((2, 3))
    first = _PackedLinear(
        codes, zeros, scales, bits=4, group_size=2
    )
    second = _PackedLinear(
        codes, zeros, scales, bits=4, group_size=2
    )
    model = _PackedModel(first, second)
    with pytest.raises(RuntimeError, match="count mismatch"):
        materialize_efficientqat_model(
            model,
            expected_count=3,
            expected_bits=4,
            expected_group_size=2,
        )
    assert model.layers[0]["q_proj"] is first
    assert model.layers[1]["down_proj"] is second

    with torch.no_grad():
        second.scales[0, 0] = 0
    with pytest.raises(ValueError, match="non-positive"):
        materialize_efficientqat_model(
            model,
            expected_count=2,
            expected_bits=4,
            expected_group_size=2,
        )
    # Validation/decode is completed for every module before mutation starts.
    assert model.layers[0]["q_proj"] is first
    assert model.layers[1]["down_proj"] is second


def test_materialize_saved_packed_checkpoint_end_to_end_on_cpu(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        AutoModelForCausalLM,
        LlamaConfig,
        LlamaForCausalLM,
        PreTrainedTokenizerFast,
    )

    efficientqat_root = Path(__file__).resolve().parents[2] / "EfficientQAT"
    if str(efficientqat_root) not in sys.path:
        sys.path.insert(0, str(efficientqat_root))
    from quantize.int_linear_real import QuantLinear

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=2,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config)
    tokenizer_backend = Tokenizer(
        WordLevel(
            {
                "<unk>": 0,
                "<bos>": 1,
                "<eos>": 2,
                "token": 3,
            },
            unk_token="<unk>",
        )
    )
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<eos>",
    )

    base_dir = tmp_path / "base"
    packed_dir = tmp_path / "packed"
    output_dir = tmp_path / "materialized"
    model.save_pretrained(base_dir, safe_serialization=True)
    tokenizer.save_pretrained(base_dir)

    bits, group_size = 4, 8
    torch.manual_seed(91)
    expected_weights = {}
    layer = model.model.layers[0]
    linears = [
        (name, module)
        for name, module in layer.named_modules()
        if name and type(module) is nn.Linear
    ]
    assert len(linears) == 7
    for local_name, linear in linears:
        with torch.no_grad():
            linear.weight.uniform_(-0.35, 0.35)
        groups = linear.in_features // group_size
        scales = torch.full((groups, linear.out_features), 0.1)
        zeros = torch.full((groups, linear.out_features), 7.0)
        packed = QuantLinear(
            bits,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
        )
        packed.pack(linear, scales, zeros)
        expected_weights[f"model.layers.0.{local_name}"] = (
            decode_efficientqat_weight_cpu(
                packed,
                expected_bits=bits,
                expected_group_size=group_size,
                name=local_name,
            )
        )
        parent = layer
        parts = local_name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], packed)

    model.save_pretrained(packed_dir, safe_serialization=True)
    tokenizer.save_pretrained(packed_dir)
    report = materialize_packed_checkpoint(
        packed_checkpoint=packed_dir,
        output_dir=output_dir,
        base_model=base_dir,
        wbits=bits,
        group_size=group_size,
        output_dtype="bfloat16",
    )
    assert report["source_quantizer"] == SOURCE_QUANTIZER
    assert report["module_count"] == 7
    assert output_dir.is_dir()

    reloaded = AutoModelForCausalLM.from_pretrained(
        output_dir,
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
    )
    for name, expected in expected_weights.items():
        module = reloaded.get_submodule(name)
        assert type(module) is nn.Linear
        assert torch.equal(module.weight, expected)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor):
    return q + 0.25, k - 0.5


class _FakeAttention(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)

    def forward(self, q, k):
        return apply_rotary_pos_emb(q, k)


class _FakeMLP(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(width, width, bias=False)
        self.up_proj = nn.Linear(width, width, bias=False)
        self.down_proj = nn.Linear(width, width, bias=False)


class _FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _FakeAttention()
        self.mlp = _FakeMLP()


class _FakeAKVModel(nn.Module):
    def __init__(self, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer() for _ in range(layers)])
        self.lm_head = nn.Linear(4, 4, bias=False)


class _FakeAnalyzer:
    def __init__(self, layers: int = 2) -> None:
        self.model = _FakeAKVModel(layers)
        self.head_dim = 4

    def get_layers(self):
        return self.model.layers

    def get_quantizable_modules(self, layer):
        return {
            "self_attn.q_proj": layer.self_attn.q_proj,
            "self_attn.k_proj": layer.self_attn.k_proj,
            "self_attn.v_proj": layer.self_attn.v_proj,
            "self_attn.o_proj": layer.self_attn.o_proj,
            "mlp.gate_proj": layer.mlp.gate_proj,
            "mlp.up_proj": layer.mlp.up_proj,
            "mlp.down_proj": layer.mlp.down_proj,
        }


def _w2a4kv4_cfg(**overrides):
    values = {
        "rotate": False,
        "act_quant_aware_gptq": False,
        "k_cache_quant_aware_gptq": False,
        "a_bits": 4,
        "a_groupsize": -1,
        "a_asym": False,
        "a_clip_ratio": 0.9,
        "k_bits": 4,
        "k_groupsize": -1,
        "k_asym": False,
        "k_clip_ratio": 0.9,
        "v_bits": 4,
        "v_groupsize": -1,
        "v_asym": False,
        "v_clip_ratio": 0.9,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _manual_k4(k: torch.Tensor) -> torch.Tensor:
    bsz, heads, seq_len, head_dim = k.shape
    rows = k.transpose(1, 2).reshape(-1, heads * head_dim)
    quantizer = quant_utils.ActQuantizer()
    quantizer.configure(
        bits=4, groupsize=-1, sym=True, clip_ratio=0.9
    )
    quantizer.find_params(rows)
    result = (
        quantizer(rows)
        .reshape(bsz, seq_len, heads, head_dim)
        .transpose(1, 2)
        .to(k)
    )
    quantizer.free()
    return result


def test_unaware_akv_uses_realq_av_and_pure_post_rope_k_without_hadamard():
    analyzer = _FakeAnalyzer(layers=2)
    analyzer.model.eval()
    cfg = _w2a4kv4_cfg()

    summary = setup_unaware_post_quant_no_rotation(analyzer, cfg)
    assert summary.decoder_layers == 2
    assert summary.activation_input_sites == 14
    assert summary.value_output_sites == 2
    assert summary.post_rope_k_sites == 2
    assert summary.as_dict()["local_qk_hadamard"] is False

    q = torch.tensor(
        [[[[1.0, 2.0, -3.0, 4.0], [0.5, -1.0, 2.0, -4.0]]]]
    )
    k = torch.tensor(
        [[[[8.0, -1.0, 0.0, 2.0], [1.5, -3.0, 4.0, 0.25]]]]
    )
    expected_q, post_rope_k = apply_rotary_pos_emb(q, k)
    actual_q, actual_k = analyzer.model.layers[0].self_attn(q, k)

    # Q is the exact object returned by RoPE: no clone, cast, quantizer, or
    # local Hadamard is allowed.
    assert torch.equal(actual_q, expected_q)
    assert actual_q.data_ptr() != q.data_ptr()
    assert torch.equal(actual_k, _manual_k4(post_rope_k))
    hadamard_k = (
        rotation_utils.hadamard_utils.HadamardTransform.apply(
            post_rope_k.float()
        )
        / math.sqrt(post_rope_k.shape[-1])
    ).to(post_rope_k)
    assert not torch.equal(actual_k, _manual_k4(hadamard_k))

    assert not any(
        isinstance(module, rotation_utils.QKRotationWrapper)
        for module in analyzer.model.modules()
    )
    assert sum(
        isinstance(module, PostRoPEKQuantWrapper)
        for module in analyzer.model.modules()
    ) == 2
    assert_unaware_post_quant_no_rotation(
        analyzer, cfg, summary=summary
    )

    # Installation is idempotent for the same frozen configuration.
    again = setup_unaware_post_quant_no_rotation(analyzer, cfg)
    assert again == summary
    assert sum(
        isinstance(module, PostRoPEKQuantWrapper)
        for module in analyzer.model.modules()
    ) == 2


def test_runner_facing_unaware_akv_entry_point_uses_frozen_protocol():
    analyzer = _FakeAnalyzer(layers=1)
    analyzer.model.eval()
    summary = install_unaware_av_and_pure_post_rope_k(
        analyzer,
        bits=4,
        groupsize=-1,
        symmetric=True,
        clip_ratio=0.9,
    )
    assert summary.activation_input_sites == 7
    assert summary.value_output_sites == 1
    assert summary.post_rope_k_sites == 1
    assert summary.as_dict()["local_qk_hadamard"] is False


@pytest.mark.parametrize(
    "cfg",
    [
        _w2a4kv4_cfg(rotate=True),
        _w2a4kv4_cfg(act_quant_aware_gptq=True),
        _w2a4kv4_cfg(k_cache_quant_aware_gptq=True),
        _w2a4kv4_cfg(k_clip_ratio=1.0),
        _w2a4kv4_cfg(k_asym=True),
    ],
)
def test_unaware_akv_rejects_non_protocol_configuration(cfg):
    analyzer = _FakeAnalyzer(layers=1)
    analyzer.model.eval()
    with pytest.raises(ValueError, match="requires"):
        setup_unaware_post_quant_no_rotation(analyzer, cfg)


def test_unaware_akv_rejects_training_or_existing_qk_rotation():
    cfg = _w2a4kv4_cfg()
    training = _FakeAnalyzer(layers=1)
    with pytest.raises(RuntimeError, match=r"model\.eval"):
        setup_unaware_post_quant_no_rotation(training, cfg)

    rotated = _FakeAnalyzer(layers=1)
    rotated.model.eval()
    attention = rotated.model.layers[0].self_attn
    rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
        attention,
        "apply_rotary_pos_emb",
        head_dim=4,
        k_bits=4,
        k_groupsize=-1,
        k_sym=True,
        k_clip_ratio=0.9,
    )
    with pytest.raises(RuntimeError, match="Hadamard"):
        setup_unaware_post_quant_no_rotation(rotated, cfg)
