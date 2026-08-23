"""Module-group definitions for one transformer layer.

Four groups, ordered by data dependency. Within a group, all modules share
the same input X (so we can capture them in a single forward). Across
groups, downstream X depends on upstream quantised weights, hence the
``capture → quantise → re-capture`` loop in ``runner.layer_loop``.

    attn_in   : q_proj, k_proj, v_proj   (share input_layernorm output)
    attn_out  : o_proj                   (input is attention output)
    mlp_in    : gate_proj, up_proj       (share post_attn_layernorm output)
    mlp_out   : down_proj                (input is SiLU(gate) * up)

The order matches the dependency graph: quantising one group changes the
inputs seen by all later groups.
"""
from __future__ import annotations

from typing import OrderedDict

import torch.nn as nn

# Order matters — drives the per-layer quantisation sequence.
GROUP_ORDER = ("attn_in", "attn_out", "mlp_in", "mlp_out")
GROUP_MODULES: dict[str, tuple[str, ...]] = {
    "attn_in":  ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "attn_out": ("self_attn.o_proj",),
    # `up` BEFORE `gate` to match old GPTQ+ ``get_sequential_quantizable_module_names``.
    # Order is observable through the per-row find_params output even though
    # the two modules are mathematically independent (Quantizer.find_params
    # allocates a `best` tensor whose fill order ties to the call sequence;
    # also future passes that share state across modules in a group will be
    # order-sensitive).
    "mlp_in":   ("mlp.up_proj", "mlp.gate_proj"),
    "mlp_out":  ("mlp.down_proj",),
}


def get_group_modules(layer: nn.Module, group_name: str) -> dict[str, nn.Module]:
    """Resolve {module_name: nn.Linear} for one group of one layer.

    Returns the underlying ``nn.Linear`` even when wrapped by
    ``ActQuantWrapper`` (rotate path); the wrapper's forward is what the
    activation goes through, but for hooks-on-input we want the wrapper so
    the X reflects the activation transformation.
    """
    name_to_mod = dict(layer.named_modules())
    out: "OrderedDict[str, nn.Module]" = OrderedDict()
    for name in GROUP_MODULES[group_name]:
        mod = name_to_mod.get(name)
        if mod is None:
            raise RuntimeError(
                f"runner.module_groups: layer is missing {name!r} for group {group_name!r}."
            )
        # If wrapped by ActQuantWrapper, the inner Linear lives at `.module`.
        # We want the Linear itself so add_batch sees the actual matrix input.
        if hasattr(mod, "module") and isinstance(mod.module, nn.Linear):
            mod = mod.module
        out[name] = mod
    return out
