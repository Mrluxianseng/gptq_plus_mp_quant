"""Architecture adapter for the isolated REAL-Q MoE implementation.

The first supported sparse architecture is the unfused Hugging Face
``Qwen3MoeForCausalLM`` implementation shipped by Transformers 4.56.2.  Keep
all architecture-dependent paths in this module so the quantisation core does
not grow model-name conditionals.

Dense layers deliberately retain the exact module groups inherited from
``realq``.  Sparse layers expose attention through those same groups and route
the expert projections through the MoE-specific runner.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import torch.nn as nn


ATTENTION_GROUP_ORDER = ("attn_in", "attn_out")
ATTENTION_GROUP_MODULES: dict[str, tuple[str, ...]] = {
    "attn_in": (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
    ),
    "attn_out": ("self_attn.o_proj",),
}
DENSE_MLP_GROUP_ORDER = ("mlp_in", "mlp_out")
DENSE_MLP_GROUP_MODULES: dict[str, tuple[str, ...]] = {
    # Preserve the historical REAL-Q order.
    "mlp_in": ("mlp.up_proj", "mlp.gate_proj"),
    "mlp_out": ("mlp.down_proj",),
}
EXPERT_PROJECTION_ORDER = ("up_proj", "gate_proj", "down_proj")
SUPPORTED_MOE_ARCHITECTURES = frozenset({"Qwen3MoeForCausalLM"})


@dataclass(frozen=True)
class LayerPlan:
    """Static description of one decoder layer."""

    is_sparse: bool
    num_experts: int
    top_k: int
    router_path: str | None


def _unwrap_linear(module: nn.Module, *, path: str) -> nn.Linear:
    """Return the underlying Linear from a plain or ActQuantWrapper module."""

    if isinstance(module, nn.Linear):
        return module
    inner = getattr(module, "module", None)
    if isinstance(inner, nn.Linear):
        return inner
    raise RuntimeError(
        f"REAL-Q MoE expected nn.Linear or ActQuantWrapper at {path!r}, "
        f"got {type(module).__name__}."
    )


def resolve_module(root: nn.Module, path: str) -> nn.Module:
    module = root
    for component in path.split("."):
        module = getattr(module, component)
    return module


def resolve_linear(root: nn.Module, path: str) -> nn.Linear:
    return _unwrap_linear(resolve_module(root, path), path=path)


def is_sparse_moe_layer(layer: nn.Module) -> bool:
    mlp = getattr(layer, "mlp", None)
    return (
        mlp is not None
        and isinstance(getattr(mlp, "experts", None), nn.ModuleList)
        and isinstance(getattr(mlp, "gate", None), nn.Linear)
        and int(getattr(mlp, "num_experts", 0)) > 0
    )


def describe_layer(layer: nn.Module) -> LayerPlan:
    if not is_sparse_moe_layer(layer):
        return LayerPlan(
            is_sparse=False,
            num_experts=0,
            top_k=0,
            router_path=None,
        )
    mlp = layer.mlp
    num_experts = int(mlp.num_experts)
    if len(mlp.experts) != num_experts:
        raise RuntimeError(
            "Qwen3-MoE descriptor mismatch: "
            f"mlp.num_experts={num_experts}, len(mlp.experts)={len(mlp.experts)}."
        )
    top_k = int(getattr(mlp, "top_k", 0))
    if not 0 < top_k <= num_experts:
        raise RuntimeError(
            f"Qwen3-MoE invalid top_k={top_k} for num_experts={num_experts}."
        )
    return LayerPlan(
        is_sparse=True,
        num_experts=num_experts,
        top_k=top_k,
        router_path="mlp.gate",
    )


def group_order(layer: nn.Module) -> tuple[str, ...]:
    if is_sparse_moe_layer(layer):
        return ATTENTION_GROUP_ORDER
    return ATTENTION_GROUP_ORDER + DENSE_MLP_GROUP_ORDER


def group_paths(layer: nn.Module, group_name: str) -> tuple[str, ...]:
    if group_name in ATTENTION_GROUP_MODULES:
        return ATTENTION_GROUP_MODULES[group_name]
    if group_name in DENSE_MLP_GROUP_MODULES and not is_sparse_moe_layer(layer):
        return DENSE_MLP_GROUP_MODULES[group_name]
    raise KeyError(
        f"group {group_name!r} is not valid for "
        f"{'sparse' if is_sparse_moe_layer(layer) else 'dense'} layer."
    )


def get_group_linears(
    layer: nn.Module,
    group_name: str,
) -> "OrderedDict[str, nn.Linear]":
    out: "OrderedDict[str, nn.Linear]" = OrderedDict()
    for path in group_paths(layer, group_name):
        out[path] = resolve_linear(layer, path)
    return out


def expert_projection_path(expert_idx: int, projection: str) -> str:
    if projection not in EXPERT_PROJECTION_ORDER:
        raise ValueError(
            f"unknown expert projection {projection!r}; "
            f"expected one of {EXPERT_PROJECTION_ORDER}."
        )
    if expert_idx < 0:
        raise ValueError(f"expert_idx must be non-negative, got {expert_idx}.")
    return f"mlp.experts.{expert_idx}.{projection}"


def iter_expert_projection_paths(layer: nn.Module) -> Iterable[tuple[int, str, str]]:
    plan = describe_layer(layer)
    if not plan.is_sparse:
        return
    for expert_idx in range(plan.num_experts):
        for projection in EXPERT_PROJECTION_ORDER:
            yield (
                expert_idx,
                projection,
                expert_projection_path(expert_idx, projection),
            )


def quantizable_linear_paths(layer: nn.Module) -> tuple[str, ...]:
    paths = [
        path
        for group_name in ATTENTION_GROUP_ORDER
        for path in ATTENTION_GROUP_MODULES[group_name]
    ]
    plan = describe_layer(layer)
    if plan.is_sparse:
        paths.extend(path for _, _, path in iter_expert_projection_paths(layer))
    else:
        paths.extend(
            path
            for group_name in DENSE_MLP_GROUP_ORDER
            for path in DENSE_MLP_GROUP_MODULES[group_name]
        )
    return tuple(paths)


def validate_analyzer(analyzer) -> None:
    """Fail closed on an unknown sparse layout before doing expensive work."""

    layers = list(analyzer.get_layers())
    sparse_flags = [is_sparse_moe_layer(layer) for layer in layers]
    has_sparse = any(sparse_flags)
    if not has_sparse:
        return
    if analyzer.model_arch not in SUPPORTED_MOE_ARCHITECTURES:
        raise NotImplementedError(
            "REAL-Q MoE currently supports only the unfused Transformers "
            "4.56.2 Qwen3MoeForCausalLM layout; got "
            f"{analyzer.model_arch!r}."
        )
    if not all(sparse_flags):
        dense_indices = [
            idx for idx, is_sparse in enumerate(sparse_flags) if not is_sparse
        ]
        raise NotImplementedError(
            "The first REAL-Q MoE implementation is intentionally scoped to "
            "the all-sparse Qwen3-30B-A3B layout. Mixed dense/sparse "
            f"Qwen3-MoE layers are not yet supported; dense layers={dense_indices}."
        )
    for layer_idx, layer in enumerate(layers):
        plan = describe_layer(layer)
        if not plan.is_sparse:
            continue
        router = resolve_module(layer, plan.router_path)
        if not isinstance(router, nn.Linear):
            raise RuntimeError(
                f"layer {layer_idx} router must remain an unwrapped nn.Linear; "
                f"got {type(router).__name__}."
            )
        for expert_idx, projection, path in iter_expert_projection_paths(layer):
            linear = resolve_linear(layer, path)
            if projection in ("up_proj", "gate_proj"):
                expected_in = int(analyzer.config.hidden_size)
                expected_out = int(analyzer.config.moe_intermediate_size)
            else:
                expected_in = int(analyzer.config.moe_intermediate_size)
                expected_out = int(analyzer.config.hidden_size)
            if (
                linear.in_features != expected_in
                or linear.out_features != expected_out
            ):
                raise RuntimeError(
                    f"layer {layer_idx} expert {expert_idx} {projection} has "
                    f"shape ({linear.out_features}, {linear.in_features}), "
                    f"expected ({expected_out}, {expected_in})."
                )


def repair_moe_down_rotation_wrappers(analyzer) -> None:
    """Set routed down-proj Hadamard metadata from its actual input width.

    The shared dense utility uses ``config.intermediate_size`` for every
    ``down_proj`` while routed experts are defined by their actual
    ``in_features``.  Qwen3-30B-A3B happens to select the same Hadamard
    factor for widths 6144 and 768 in the current table, so this is a
    fail-closed architecture hardening change rather than an observed target
    model numerical fix or performance optimisation.
    """

    from utils import hadamard_utils

    for layer in analyzer.get_layers():
        if not is_sparse_moe_layer(layer):
            continue
        for _, projection, path in iter_expert_projection_paths(layer):
            if projection != "down_proj":
                continue
            wrapper = resolve_module(layer, path)
            inner = getattr(wrapper, "module", None)
            if not isinstance(inner, nn.Linear):
                raise RuntimeError(
                    f"expected routed {path} to be ActQuantWrapper after "
                    f"rotation, got {type(wrapper).__name__}."
                )
            had_k, k = hadamard_utils.get_hadK(inner.in_features)
            wrapper.online_full_had = True
            # ``get_hadK`` loads explicit factors on CPU.  This repair is also
            # called after the sparse model has become fully CUDA resident
            # (for example from the idempotent AKV setup immediately before
            # Stage 1), so assigning that CPU tensor verbatim would silently
            # reintroduce a host buffer.  ``fp32_had=False`` means the wrapper
            # consumes the factor in the activation/weight compute dtype.
            wrapper.had_K = (
                None
                if had_k is None
                else had_k.to(
                    device=inner.weight.device,
                    dtype=inner.weight.dtype,
                )
            )
            wrapper.K = k
            wrapper.fp32_had = False
