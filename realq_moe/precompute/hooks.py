"""Forward + backward hooks used during static end-to-end precompute.

Two managers, attached together once per model:

* ``SaliencyHookManager`` — per-(layer, module) tensor backward hook on each
  linear module's output. Records the squared gradient norm (sum of
  ``grad²``) over each output-channel group; the result is the rank-local
  saliency tensor used to weight the
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

from utils.saliency_utils import (
    clip_global_percentile_,
    grouped_gradient_norm_squared,
)


# Loss-grad scaling: the NLL is multiplied by this scalar before backward, so
# fp16 activations don't underflow on the way down. Saliency is `grad²` and
# Fisher is `g g^T` — both pick up the scale². Saliency is only used
# relatively (X^T diag(s) X) so we leave the scaling baked in; Fisher is an
# absolute quantity used in `0.5 Δy^T F Δy`, so we divide it out at hook time.
LOSS_GRAD_SCALE = 1000.0
QUADRATIC_SCALE = LOSS_GRAD_SCALE * LOSS_GRAD_SCALE


class SaliencyHookManager:
    """Per-(layer_idx, module_name) saliency at each linear module's output.

    After all forwards/backwards, ``finalize()`` returns dense saliency as
    ``(N_local,T,G)`` and routed-expert saliency as ragged ``(A_local,1,G)``
    CUDA fp32 tensors.  The MoE speed-first path never offloads these
    Stage-0 statistics to CPU.
    """

    def __init__(self, num_groups: int, clip_percentile: float | None) -> None:
        self.num_groups = num_groups
        self.clip_percentile = clip_percentile
        # _data[layer_idx][module_name] = list of CUDA saliency chunks.
        self._data: list[dict[str, list[torch.Tensor]]] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._expert_module_indices: list[dict[str, int]] = []
        self._route_manager = None

    def attach(
        self,
        layer_modules: list[dict[str, nn.Module]],
        *,
        expert_module_indices: list[dict[str, int]] | None = None,
        route_manager=None,
    ) -> None:
        """Register hooks on every module of every layer.

        ``layer_modules[i]`` is a dict ``{module_name: nn.Module}`` for layer
        ``i``. Module names are the caller's choice (typically the dotted
        path inside the layer, e.g. ``self_attn.q_proj``).
        """
        if expert_module_indices is None:
            expert_module_indices = [{} for _ in layer_modules]
        if len(expert_module_indices) != len(layer_modules):
            raise ValueError(
                "expert_module_indices must have one mapping per layer."
            )
        if any(expert_module_indices) and route_manager is None:
            raise ValueError(
                "expert saliency hooks require a RouteCaptureManager."
            )
        self._expert_module_indices = expert_module_indices
        self._route_manager = route_manager
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
        for layer_idx, layer_chunks in enumerate(self._data):
            layer_out: dict[str, torch.Tensor] = {}
            for name, chunks in layer_chunks.items():
                if not chunks:
                    if name in self._expert_module_indices[layer_idx]:
                        device = next(
                            (
                                chunk.device
                                for peer_chunks in layer_chunks.values()
                                for chunk in peer_chunks
                            ),
                            torch.device(
                                f"cuda:{torch.cuda.current_device()}"
                            ),
                        )
                        layer_out[name] = torch.empty(
                            (0, 1, self.num_groups),
                            dtype=torch.float32,
                            device=device,
                        )
                        continue
                    raise RuntimeError(
                        f"SaliencyHookManager.finalize: no batches recorded "
                        f"for dense module {name}"
                    )
                combined = torch.cat(chunks, dim=0)  # (N_local, T, G)
                if name in self._expert_module_indices[layer_idx]:
                    # Expert clipping is packed by layer/projection after all
                    # modules finalize.  Calling the dense helper here would
                    # reject a local-empty/global-nonempty expert and issue
                    # 3*E collectives per layer.
                    layer_out[name] = combined
                else:
                    layer_out[name] = clip_global_percentile_(
                        combined, self.clip_percentile
                    )
                chunks.clear()
            out.append(layer_out)
        return out

    def _make_forward_hook(self, layer_idx: int, module_name: str):
        def forward_hook(_module, _inputs, outputs):
            out_tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            assignment_count = None
            expert_idx = self._expert_module_indices[layer_idx].get(
                module_name
            )
            if expert_idx is not None:
                if out_tensor.dim() != 2:
                    raise RuntimeError(
                        f"expert module {module_name} expected a 2D output, "
                        f"got {tuple(out_tensor.shape)}."
                    )
                # The expert receives exactly the official torch.where route
                # rows.  Reading 128 GPU route counters into Python for every
                # layer/batch would serialize the Stage-0 stream; retain the
                # already-authoritative output extent and validate it against
                # the packed route once all batches are finalized.
                assignment_count = int(out_tensor.shape[0])
            # Only a Python count is retained by autograd.  RouteCaptureManager
            # can release its current route immediately after model.forward,
            # before backward retains the much larger activation graph.
            out_tensor.register_hook(
                self._make_grad_hook(
                    layer_idx,
                    module_name,
                    assignment_count=assignment_count,
                )
            )
        return forward_hook

    def _make_grad_hook(
        self,
        layer_idx: int,
        module_name: str,
        *,
        assignment_count: int | None = None,
    ):
        num_groups = self.num_groups
        def grad_hook(grad: torch.Tensor) -> None:
            if assignment_count is None:
                if grad.dim() != 3:
                    raise RuntimeError(
                        f"dense module {module_name} expected gradient "
                        f"(B,T,H), got {tuple(grad.shape)}."
                    )
            else:
                if grad.dim() != 2:
                    raise RuntimeError(
                        f"expert module {module_name} expected gradient "
                        f"(assignments,H), got {tuple(grad.shape)}."
                    )
                if grad.shape[0] != assignment_count:
                    raise RuntimeError(
                        f"expert module {module_name} backward assignment "
                        "count no longer matches its captured route."
                    )
            H = grad.shape[-1]
            if H % num_groups != 0:
                raise ValueError(
                    f"SaliencyHookManager: module {module_name} output dim {H} "
                    f"not divisible by num_groups {num_groups}."
                )
            sal = grouped_gradient_norm_squared(
                grad.detach(), num_groups
            )
            if assignment_count is not None:
                # Reuse RealQLayer's sample cursor by treating each routed
                # assignment as a one-token sample.
                sal = sal.unsqueeze(1)  # (assignments, 1, G)
            self._data[layer_idx][module_name].append(sal)
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
