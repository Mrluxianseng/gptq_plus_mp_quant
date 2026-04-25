import re
import os
import json
import logging
import shutil
from types import MethodType
from collections import defaultdict
from typing import List, Dict, Optional, Union, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.modules.conv import _ConvNd
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoTokenizer, \
                         PreTrainedTokenizerBase, AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from utils import dist_utils


LINEAR_LAYERS = (nn.Linear, _ConvNd)


def _prepare_config_for_untied_lm_head(model_str: str):
    config = AutoConfig.from_pretrained(model_str, trust_remote_code=True)
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False  # TODO. disable tie_word_embeddings by default
        process_word_embeddings = True
    return config, process_word_embeddings


def load_model(model_str_or_model):
    """Returns a model from a string or a model object. If a string is passed, it will be loaded from the HuggingFace"""
    if isinstance(model_str_or_model, str):
        config, process_word_embeddings = _prepare_config_for_untied_lm_head(model_str_or_model)
        model = AutoModelForCausalLM.from_pretrained(
            model_str_or_model,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='cpu',
            # attn_implementation='eager',
        )
        model.tie_word_embeddings = process_word_embeddings
        if process_word_embeddings:
            model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    else:
        assert isinstance(model_str_or_model, PreTrainedModel), "model must be a string or a PreTrainedModel"
        model = model_str_or_model

    model.eval()

    return model


def _fsdp_shard_model_for_precompute(model, analyzer, fsdp_cpu_offload: bool):
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy
    from torch.distributed.device_mesh import init_device_mesh

    world = dist_utils.get_world_size()
    if world < 1:
        raise RuntimeError("FSDP precompute requires an initialized distributed process group.")
    mesh = init_device_mesh("cuda", (world,))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    offload_policy = CPUOffloadPolicy(pin_memory=True) if fsdp_cpu_offload else None

    kwargs = {"mesh": mesh, "mp_policy": mp_policy}
    if offload_policy is not None:
        kwargs["offload_policy"] = offload_policy

    for layer in analyzer.get_layers():
        fully_shard(layer, **kwargs)
    fully_shard(model, **kwargs)
    model._gptqplus_fsdp_prepared = True
    model._gptqplus_fsdp_cpu_offload = bool(fsdp_cpu_offload)


def _save_prepared_checkpoint(analyzer, output_dir: str, args, meta: dict) -> None:
    from utils import memory_utils

    tmp_dir = f"{output_dir}.tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    analyzer.model.save_pretrained(
        tmp_dir,
        safe_serialization=True,
        max_shard_size=getattr(args, "fsdp_prepared_max_shard_size", "5GB"),
    )
    analyzer.tokenizer.save_pretrained(tmp_dir)
    with open(os.path.join(tmp_dir, "gptqplus_fsdp_meta_checkpoint.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(tmp_dir, "_SUCCESS"), "w") as f:
        f.write("ok\n")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.replace(tmp_dir, output_dir)
    memory_utils.cleanup_memory()


def _build_rotated_checkpoint_on_rank0(args, rotated_dir: str) -> None:
    """Materialize and rotate exactly one full CPU model, then save it for sharded load."""
    import transformers
    from utils import rotation_utils, memory_utils

    logging.info(
        "fsdp_meta_init rotate: rank0 building rotated checkpoint at %s. "
        "Only rank0 materializes the full CPU model for this preprocessing step.",
        rotated_dir,
    )
    analyzer = ModelAnalyzer(args.model, args.seq_len)
    model = analyzer.model

    rotation_utils.fuse_layer_norms(analyzer)
    rotation_utils.rotate_model(args, analyzer)
    memory_utils.cleanup_memory()

    meta = {
        "kind": "rotated",
        "source_model": args.model,
        "seq_len": int(args.seq_len),
        "seed": int(getattr(args, "seed", 0)),
        "rotate": True,
        "optimized_rotation_path": getattr(args, "optimized_rotation_path", None),
        "transformers_version": transformers.__version__,
    }
    _save_prepared_checkpoint(analyzer, rotated_dir, args, meta)
    del analyzer, model
    memory_utils.cleanup_memory()


def _prepared_checkpoint_ready(checkpoint_dir: str, args, *, kind: str, rotate: bool) -> bool:
    success_path = os.path.join(checkpoint_dir, "_SUCCESS")
    meta_path = os.path.join(checkpoint_dir, "gptqplus_fsdp_meta_checkpoint.json")
    config_path = os.path.join(checkpoint_dir, "config.json")
    if not (os.path.exists(success_path) and os.path.exists(meta_path) and os.path.exists(config_path)):
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        return False
    return (
        meta.get("kind") == kind
        and meta.get("source_model") == args.model
        and int(meta.get("seed", -1)) == int(getattr(args, "seed", 0))
        and bool(meta.get("rotate")) is bool(rotate)
        and meta.get("optimized_rotation_path") == getattr(args, "optimized_rotation_path", None)
    )


def _build_untied_checkpoint_on_rank0(args, untied_dir: str) -> None:
    import transformers
    from utils import memory_utils

    logging.info(
        "fsdp_meta_init: rank0 building untied checkpoint at %s because the source "
        "checkpoint ties word embeddings and may not contain lm_head.weight.",
        untied_dir,
    )
    analyzer = ModelAnalyzer(args.model, args.seq_len)
    meta = {
        "kind": "untied",
        "source_model": args.model,
        "seq_len": int(args.seq_len),
        "seed": int(getattr(args, "seed", 0)),
        "rotate": False,
        "optimized_rotation_path": getattr(args, "optimized_rotation_path", None),
        "transformers_version": transformers.__version__,
    }
    _save_prepared_checkpoint(analyzer, untied_dir, args, meta)
    del analyzer
    memory_utils.cleanup_memory()


def _get_fsdp_meta_checkpoint_path(args) -> Tuple[str, bool]:
    base_dir = getattr(args, "static_cache_path", None)
    if base_dir is None:
        base_dir = os.path.join(getattr(args, "cache_dir", "./cache"), "fsdp_meta_prepared")
    if not getattr(args, "rotate", False):
        src_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        if not getattr(src_config, "tie_word_embeddings", False):
            return args.model, False
        untied_dir = os.path.join(
            base_dir,
            "_prepared_checkpoints",
            f"{getattr(args, 'model_name', os.path.basename(args.model))}_untied",
        )
        if dist_utils.is_main() and not _prepared_checkpoint_ready(
            untied_dir, args, kind="untied", rotate=False,
        ):
            _build_untied_checkpoint_on_rank0(args, untied_dir)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return untied_dir, False
    if getattr(args, "optimized_rotation_path", None) is not None:
        opt_tag = os.path.basename(str(args.optimized_rotation_path)).replace("/", "_")
    else:
        opt_tag = "hadamard"
    seed_tag = f"seed{int(getattr(args, 'seed', 0))}"
    rotated_dir = os.path.join(
        base_dir,
        "_prepared_checkpoints",
        f"{getattr(args, 'model_name', os.path.basename(args.model))}_rot_{opt_tag}_{seed_tag}",
    )
    if dist_utils.is_main() and not _prepared_checkpoint_ready(
        rotated_dir, args, kind="rotated", rotate=True,
    ):
        _build_rotated_checkpoint_on_rank0(args, rotated_dir)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return rotated_dir, True


def load_model_fsdp_meta_for_precompute(args):
    """Initialize on meta, FSDP-shard, then load checkpoint directly into shards."""
    from accelerate import init_empty_weights
    from accelerate.utils import load_checkpoint_in_model

    checkpoint_path, checkpoint_is_rotated = _get_fsdp_meta_checkpoint_path(args)
    config, process_word_embeddings = _prepare_config_for_untied_lm_head(checkpoint_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    model.tie_word_embeddings = process_word_embeddings
    model.seqlen = args.seq_len

    analyzer = ModelAnalyzer(
        model,
        args.seq_len,
        tokenizer_source=checkpoint_path,
        skip_state_dict=True,
    )
    _fsdp_shard_model_for_precompute(
        model,
        analyzer,
        fsdp_cpu_offload=bool(getattr(args, "fsdp_cpu_offload", False)),
    )
    logging.info(
        "fsdp_meta_init: loading checkpoint %s into FSDP2 shards (rotated=%s, cpu_offload=%s)",
        checkpoint_path,
        checkpoint_is_rotated,
        bool(getattr(args, "fsdp_cpu_offload", False)),
    )
    load_checkpoint_in_model(
        model,
        checkpoint_path,
        dtype=torch.bfloat16,
        strict=False,
        full_state_dict=True,
        broadcast_from_rank0=True,
    )
    model.eval()
    model._gptqplus_fsdp_meta_init = True
    model._gptqplus_checkpoint_is_rotated = checkpoint_is_rotated
    return analyzer


def load_tokenizer(model_str_or_model_or_tokenizer):
    """Returns a tokenizer from the model string or model object or tokenizer object"""
    if isinstance(model_str_or_model_or_tokenizer, str):
        model_str = model_str_or_model_or_tokenizer
        return AutoTokenizer.from_pretrained(model_str, trust_remote_code=True)
    elif isinstance(model_str_or_model_or_tokenizer, PreTrainedModel):
        model_str = model_str_or_model_or_tokenizer.name_or_path
        return AutoTokenizer.from_pretrained(model_str, trust_remote_code=True)
    else:
        assert isinstance(model_str_or_model_or_tokenizer, PreTrainedTokenizerBase), \
            f"Unsupported type for model_str_or_model_or_tokenizer: {type(model_str_or_model_or_tokenizer)}"
        return model_str_or_model_or_tokenizer


def select_layers(
    model: nn.Module,
    layer_prefix: Optional[str] = "",
    layer_regex: str = ".*",
    layer_classes: Union[nn.Module, List[nn.Module]] = nn.Module,
) -> Dict[str, nn.Module]:
    layers = {}
    for layer_name, layer in model.named_modules():
        if (
            isinstance(layer, layer_classes)
            and re.search(layer_regex, layer_name)
            and layer_name.startswith(layer_prefix)
        ):
            layers[layer_name] = layer
    return layers


class ModelAnalyzer:
    """ModelAnalyzer is a class that provides an interface to access relevant model information for quantization.
    """

    def __init__(self, model_str_or_model, seq_len, tokenizer_source=None, skip_state_dict=False):
        self.model = load_model(model_str_or_model)
        self.tokenizer = load_tokenizer(tokenizer_source if tokenizer_source is not None else model_str_or_model)
        self.config = self.model.config

        self.state_dict = None if skip_state_dict else self.model.state_dict()
        assert len(self.config.architectures) == 1
        self.model_arch = self.config.architectures[0]

        self.model.seqlen = seq_len
        self.num_layers = len(self.get_layers())
        self.hidden_size = self.config.hidden_size
        self.intermediate_size = self.config.intermediate_size
        self.num_attention_heads = self.config.num_attention_heads
        self.num_key_value_heads = getattr(self.config, "num_key_value_heads", self.num_attention_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.head_dim = getattr(self.config, "head_dim", self.hidden_size // self.num_attention_heads)
        self.tie_word_embeddings = self.model.tie_word_embeddings

    def get_lm_head(self):
        if self.model_arch in ["Qwen3ForCausalLM", "Qwen3MoeForCausalLM", "LlamaForCausalLM"]:
            return self.get_model_attribute("lm_head", self.model)
        else:
            raise NotImplementedError

    def get_embed_layer(self):
        if self.model_arch in ["Qwen3ForCausalLM", "Qwen3MoeForCausalLM", "LlamaForCausalLM"]:
            return self.get_model_attribute("model.embed_tokens", self.model)
        else:
            raise NotImplementedError

    def get_layernorm_before_head(self):
        if self.model_arch in ["Qwen3ForCausalLM", "Qwen3MoeForCausalLM", "LlamaForCausalLM"]:
            return self.get_model_attribute("model.norm", self.model)
        else:
            raise NotImplementedError

    def get_layers(self):
        """Return the layers of the model."""
        if self.model_arch in ["Qwen3ForCausalLM", "Qwen3MoeForCausalLM", "LlamaForCausalLM"]:
            return self.get_model_attribute("model.layers", self.model)
        else:
            raise NotImplementedError

    def get_pre_block_modules(self):
        """Return pre-block modules of the model."""
        if self.model_arch in ["Qwen3ForCausalLM", "Qwen3MoeForCausalLM", "LlamaForCausalLM"]:
            return [
                self.get_model_attribute(name, self.model) for name in [
                    "model.embed_tokens",
                    "model.rotary_emb",
                ]
            ]
        else:
            raise NotImplementedError

    def get_quantizable_modules(self, layer):
        """Return the quantizable modules of the layer."""
        if self.model_arch in ["Qwen3ForCausalLM", "Qwen3MoeForCausalLM", "LlamaForCausalLM"]:
            return select_layers(layer, "", ".*((q|k|v|o|gate|up|down)_proj)", LINEAR_LAYERS)
        else:
            raise NotImplementedError

    def get_sequential_quantizable_module_names(self):
        """Return the quantizable module names of the layer in sequential order."""
        if self.model_arch in ["Qwen3ForCausalLM", "LlamaForCausalLM"]:
            return [
                [
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                ],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        elif self.model_arch in ["Qwen3MoeForCausalLM"]:
            return [
                [
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                ],
                ["self_attn.o_proj"],
                [f"mlp.experts.{i}.up_proj" for i in range(self.config.num_experts)] + \
                [f"mlp.experts.{i}.gate_proj" for i in range(self.config.num_experts)],
                [f"mlp.experts.{i}.down_proj" for i in range(self.config.num_experts)],
            ]
        else:
            raise NotImplementedError

    def get_layernorms(self, layer):
        if self.model_arch in ["Qwen3ForCausalLM", "Qwen3MoeForCausalLM", "LlamaForCausalLM"]:
            return [
                self.get_model_attribute(name, layer) for name in [
                    "input_layernorm",
                    "post_attention_layernorm",
                ]
            ]
        else:
            raise NotImplementedError

    def get_perlayer_input_modules(self, layer):
        if self.model_arch in ["Qwen3ForCausalLM", "LlamaForCausalLM"]:
            return [
                [
                    self.get_model_attribute(name, layer) for name in [
                        "self_attn.q_proj",
                        "self_attn.k_proj",
                        "self_attn.v_proj"
                    ]
                ],
                [
                    self.get_model_attribute(name, layer) for name in [
                        "mlp.up_proj",
                        "mlp.gate_proj"
                    ]
                ],
            ]
        elif self.model_arch in ["Qwen3MoeForCausalLM"]:
            return [
                [
                    self.get_model_attribute(name, layer) for name in [
                        "self_attn.q_proj",
                        "self_attn.k_proj",
                        "self_attn.v_proj"
                    ]
                ],
                [self.get_model_attribute(f"mlp.experts.{i}.up_proj", layer) for i in range(layer.mlp.num_experts)] + \
                [self.get_model_attribute(f"mlp.experts.{i}.gate_proj", layer) for i in range(layer.mlp.num_experts)] + \
                [self.get_model_attribute("mlp.gate", layer)],
            ]
        else:
            raise NotImplementedError

    def get_perlayer_output_modules(self, layer):
        if self.model_arch in ["Qwen3ForCausalLM", "LlamaForCausalLM"]:
            return [
                [self.get_model_attribute("self_attn.o_proj", layer)],
                [self.get_model_attribute("mlp.down_proj", layer)],
            ]
        elif self.model_arch in ["Qwen3MoeForCausalLM"]:
            return [
                [self.get_model_attribute("self_attn.o_proj", layer)],
                [self.get_model_attribute(f"mlp.experts.{i}.down_proj", layer) for i in range(layer.mlp.num_experts)],
            ]
        else:
            raise NotImplementedError

    def get_perlayer_down_proj(self, layer):
        if self.model_arch in ["Qwen3ForCausalLM", "LlamaForCausalLM"]:
            return [self.get_model_attribute("mlp.down_proj", layer)]
        elif self.model_arch in ["Qwen3MoeForCausalLM"]:
            return [self.get_model_attribute(f"mlp.experts.{i}.down_proj", layer) for i in range(layer.mlp.num_experts)]
        else:
            raise NotImplementedError

    def get_perlayer_o_proj(self, layer):
        if self.model_arch in ["Qwen3ForCausalLM", "Qwen3MoeForCausalLM", "LlamaForCausalLM"]:
            return self.get_model_attribute("self_attn.o_proj", layer)
        else:
            raise NotImplementedError

    def get_perlayer_v_proj(self, layer):
        if self.model_arch in ["Qwen3ForCausalLM", "Qwen3MoeForCausalLM", "LlamaForCausalLM"]:
            return self.get_model_attribute("self_attn.v_proj", layer)
        else:
            raise NotImplementedError

    def get_model_attribute(self, attr_str, module):
        for attrib_name in attr_str.split('.'):
            module = getattr(module, attrib_name)
        return module
