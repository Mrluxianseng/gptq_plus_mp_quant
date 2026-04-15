import re
import os
from types import MethodType
from collections import defaultdict
from typing import List, Dict, Optional, Union, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoTokenizer, \
                         PreTrainedTokenizerBase, AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer


LINEAR_LAYERS = (nn.Linear, _ConvNd)


def load_model(model_str_or_model):
    """Returns a model from a string or a model object. If a string is passed, it will be loaded from the HuggingFace"""
    if isinstance(model_str_or_model, str):
        config = AutoConfig.from_pretrained(model_str_or_model)
        process_word_embeddings = False
        if config.tie_word_embeddings:
            config.tie_word_embeddings = False  # TODO. disable tie_word_embeddings by default
            process_word_embeddings = True
        model = AutoModelForCausalLM.from_pretrained(
            model_str_or_model,
            config=config,
            trust_remote_code=True,
            torch_dtype='auto',
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

    def __init__(self, model_str_or_model, seq_len):
        self.model = load_model(model_str_or_model)
        self.tokenizer = load_tokenizer(model_str_or_model)
        self.config = self.model.config

        self.state_dict = self.model.state_dict()
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
        if self.model_arch in ["Qwen3ForCausalLM", "Qwen3MoeForCausalLM"]:
            return self.get_model_attribute("lm_head", self.model)
        else:
            raise NotImplementedError

    def get_embed_layer(self):
        if self.model_arch in ["Qwen3ForCausalLM", "Qwen3MoeForCausalLM"]:
            return self.get_model_attribute("model.embed_tokens", self.model)
        else:
            raise NotImplementedError

    def get_layernorm_before_head(self):
        if self.model_arch in ["Qwen3ForCausalLM", "Qwen3MoeForCausalLM"]:
            return self.get_model_attribute("model.norm", self.model)
        else:
            raise NotImplementedError

    def get_layers(self):
        """Return the layers of the model."""
        if self.model_arch in ["Qwen3ForCausalLM", "Qwen3MoeForCausalLM"]:
            return self.get_model_attribute("model.layers", self.model)
        else:
            raise NotImplementedError

    def get_pre_block_modules(self):
        """Return pre-block modules of the model."""
        if self.model_arch in ["Qwen3ForCausalLM", "Qwen3MoeForCausalLM"]:
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
        if self.model_arch in ["Qwen3ForCausalLM", "Qwen3MoeForCausalLM"]:
            return select_layers(layer, "", ".*((q|k|v|o|gate|up|down)_proj)", LINEAR_LAYERS)
        else:
            raise NotImplementedError

    def get_sequential_quantizable_module_names(self):
        """Return the quantizable module names of the layer in sequential order."""
        if self.model_arch in ["Qwen3ForCausalLM"]:
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
        if self.model_arch in ["Qwen3ForCausalLM", "Qwen3MoeForCausalLM"]:
            return [
                self.get_model_attribute(name, layer) for name in [
                    "input_layernorm",
                    "post_attention_layernorm",
                ]
            ]
        else:
            raise NotImplementedError

    def get_perlayer_input_modules(self, layer):
        if self.model_arch in ["Qwen3ForCausalLM"]:
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
        if self.model_arch in ["Qwen3ForCausalLM"]:
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
        if self.model_arch in ["Qwen3ForCausalLM"]:
            return [self.get_model_attribute("mlp.down_proj", layer)]
        elif self.model_arch in ["Qwen3MoeForCausalLM"]:
            return [self.get_model_attribute(f"mlp.experts.{i}.down_proj", layer) for i in range(layer.mlp.num_experts)]
        else:
            raise NotImplementedError

    def get_perlayer_o_proj(self, layer):
        if self.model_arch in ["Qwen3ForCausalLM", "Qwen3MoeForCausalLM"]:
            return self.get_model_attribute("self_attn.o_proj", layer)
        else:
            raise NotImplementedError

    def get_perlayer_v_proj(self, layer):
        if self.model_arch in ["Qwen3ForCausalLM", "Qwen3MoeForCausalLM"]:
            return self.get_model_attribute("self_attn.v_proj", layer)
        else:
            raise NotImplementedError

    def get_model_attribute(self, attr_str, module):
        for attrib_name in attr_str.split('.'):
            module = getattr(module, attrib_name)
        return module
