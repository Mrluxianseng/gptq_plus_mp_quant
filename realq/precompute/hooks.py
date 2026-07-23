"""Forward + backward hooks used during static end-to-end precompute.

Two managers, attached together once per model:

* ``SaliencyHookManager`` — per-(layer, module) tensor backward hook on each
  linear module's output. Records ``mean_g(grad²)`` over output channel
  groups; the result is the rank-local saliency tensor used to weight the
  GPTQ Hessian when quantising that module.

* ``FisherHookManager`` — per-layer tensor backward hook on the transformer
  block's output. Accumulates ``g g^T`` over (sample, token) on GPU fp32. The
  caller is responsible for the global all-reduce + token-count
  normalisation; this class only owns the per-rank running sum.

Both register a forward hook to grab the activation tensor, then attach a
``tensor.register_hook`` to capture its gradient during backward. The
forward hook is fired once per batch; the gradient hook fires once per
backward — this matches how the old `gptq_plus_utils` precompute works.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from utils.saliency_utils import clip_global_percentile_


# Loss-grad scaling: the NLL is multiplied by this scalar before backward, so
# fp16 activations don't underflow on the way down. Saliency is `grad²` and
# Fisher is `g g^T` — both pick up the scale². Saliency is only used
# relatively (X^T diag(s) X) so we leave the scaling baked in; Fisher is an
# absolute quantity used in `0.5 Δy^T F Δy`, so we divide it out at hook time.
LOSS_GRAD_SCALE = 1000.0
QUADRATIC_SCALE = LOSS_GRAD_SCALE * LOSS_GRAD_SCALE


class SaliencyHookManager:
    """Per-(layer_idx, module_name) saliency at each linear module's output.

    After all forwards/backwards, ``finalize()`` returns
    ``list[dict[name, (N_local, T, num_groups) cpu fp32 tensor]]``.
    """

    def __init__(self, num_groups: int, clip_percentile: float | None) -> None:
        self.num_groups = num_groups
        self.clip_percentile = clip_percentile
        # _data[layer_idx][module_name] = list of (B, T, num_groups) cpu chunks
        self._data: list[dict[str, list[torch.Tensor]]] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def attach(self, layer_modules: list[dict[str, nn.Module]]) -> None:
        """Register hooks on every module of every layer.

        ``layer_modules[i]`` is a dict ``{module_name: nn.Module}`` for layer
        ``i``. Module names are the caller's choice (typically the dotted
        path inside the layer, e.g. ``self_attn.q_proj``).
        """
        for layer_idx, modules in enumerate(layer_modules):
            self._data.append({name: [] for name in modules})
            for name, module in modules.items():
                handle = module.register_forward_hook(
                    self._make_forward_hook(layer_idx, name)
                )
                self._handles.append(handle)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def finalize(self) -> list[dict[str, torch.Tensor]]:
        out: list[dict[str, torch.Tensor]] = []
        for layer_chunks in self._data:
            layer_out: dict[str, torch.Tensor] = {}
            for name, chunks in layer_chunks.items():
                if not chunks:
                    raise RuntimeError(
                        f"SaliencyHookManager.finalize: no batches recorded for "
                        f"module {name}"
                    )
                combined = torch.cat(chunks, dim=0)  # (N_local, T, G)
                layer_out[name] = clip_global_percentile_(
                    combined, self.clip_percentile
                )
                chunks.clear()
            out.append(layer_out)
        return out

    def _make_forward_hook(self, layer_idx: int, module_name: str):
        def forward_hook(_module, _inputs, outputs):
            out_tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            # Defer to backward time; capture (layer_idx, name) by closure.
            out_tensor.register_hook(self._make_grad_hook(layer_idx, module_name))
        return forward_hook

    def _make_grad_hook(self, layer_idx: int, module_name: str):
        num_groups = self.num_groups
        def grad_hook(grad: torch.Tensor) -> None:
            # grad: (B, T, H_out)
            B, T, H = grad.shape
            if H % num_groups != 0:
                raise ValueError(
                    f"SaliencyHookManager: module {module_name} output dim {H} "
                    f"not divisible by num_groups {num_groups}."
                )
            group_size = H // num_groups
            grad_fp32 = grad.detach().float()
            sal = (
                grad_fp32.pow(2)
                .view(B, T, num_groups, group_size)
                .mean(dim=-1)
            )  # (B, T, G)
            self._data[layer_idx][module_name].append(sal.cpu())
        return grad_hook


class FisherHookManager:
    """Per-layer Fisher matrix at the transformer block's output.

    Each rank accumulates a running fp32 (H, H) sum on GPU. Caller does the
    cross-rank all-reduce + token-count normalisation; ``finalize()`` only
    returns the raw running sums, plus the per-rank token count.
    """

    def __init__(self) -> None:
        self._sums: list[torch.Tensor | None] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._token_count = 0

    def attach(self, layers: Iterable[nn.Module]) -> None:
        layers = list(layers)
        self._sums = [None] * len(layers)
        for layer_idx, layer in enumerate(layers):
            handle = layer.register_forward_hook(self._make_forward_hook(layer_idx))
            self._handles.append(handle)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def add_token_count(self, n_tokens: int) -> None:
        """Caller reports how many (sample × token) entries went into the
        last forward — Fisher needs the same denominator across ranks."""
        self._token_count += n_tokens

    def finalize(self) -> tuple[list[torch.Tensor | None], int]:
        return self._sums, self._token_count

    def _make_forward_hook(self, layer_idx: int):
        def forward_hook(_module, _inputs, outputs):
            out_tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            out_tensor.register_hook(self._make_grad_hook(layer_idx))
        return forward_hook

    def _make_grad_hook(self, layer_idx: int):
        def grad_hook(grad: torch.Tensor) -> None:
            grad_fp32 = grad.detach().float()
            grad_flat = grad_fp32.reshape(-1, grad_fp32.shape[-1])  # (B·T, H)
            # Divide by the loss-grad scaling so the stored value is in the
            # same units as a loss-scale=1 backward would produce.
            block = grad_flat.t() @ grad_flat
            block.div_(QUADRATIC_SCALE)
            cur = self._sums[layer_idx]
            if cur is None:
                self._sums[layer_idx] = block
            else:
                cur.add_(block)
                del block
        return grad_hook
