"""Packed GPU-resident fixed-route replay for one joint MoE projection.

The current-layer Block-GD target uses routes captured from the same student
state immediately before expert quantization.  Re-running the native sparse
MLP for every column boundary needlessly repeats router, softmax, top-k,
one-hot and expert-hit discovery.  This module packs those already-known
assignments into an expert-major, sample-major CSR.  Tensor payloads have
exactly ``A`` rows, where ``A`` is the number of natural route assignments;
there is no ``(E,N,M,*)`` cache and no ``(E,B*M)`` padded GEMM.

A refresh selecting ``B`` samples gathers only their real assignments.  The
selected rows stay expert-major and are evaluated with
``torch._grouped_mm(A, W.transpose(1, 2), offs=cumulative_counts)``.  Repeated
offsets represent cold experts without removing them from the single
``(E,R,C)`` candidate leaf.  Route selection, offsets, and mixing remain CUDA
resident.

Projection-specific immutable values are cached once:

* ``up_proj``: the fixed ``act(gate_proj(x))`` factor;
* ``gate_proj``: the already-quantized fixed ``up_proj(x)`` factor;
* ``down_proj``: the already-quantized ``act(gate_proj(x))*up_proj(x)``
  after the down wrapper's fixed input transform.

Only the candidate projection is expressed as a single ``(E,R,C)`` autograd
leaf.  For up/gate refresh, a supported common down-wrapper input transform
(identity or the formal online full Hadamard) is applied to compact rows, then
all fixed down weights run in one second grouped GEMM.  A next-layer slide
remains outside this helper and continues to use the complete natural
next-layer forward.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn

from realq_moe.utils.module_capture import (
    capture_inner_linear_input_without_gemm,
)
from utils import hadamard_utils

if TYPE_CHECKING:
    from realq_moe.precompute.routed_stats import PackedLayerRoutes


ProjectionName = Literal["up_proj", "gate_proj", "down_proj"]
_PROJECTIONS = ("up_proj", "gate_proj", "down_proj")


@dataclass(frozen=True)
class FixedRoutePaddingStats:
    """Construction-time packing properties for the complete cache.

    The historical name is retained for diagnostics API compatibility.
    ``padded_rows`` and ``rows_per_sample_expert`` are always zero on the
    packed path.
    """

    active_assignments: int
    packed_rows: int
    padded_rows: int
    rows_per_sample_expert: int
    padding_efficiency: float


@dataclass(frozen=True)
class FixedRouteBatchStats:
    """Hot-path packed-work metrics.

    ``active_assignments`` and ``padding_efficiency`` deliberately remain CUDA
    scalar tensors.  Consumers may log them outside a timed refresh; producing
    the metrics itself performs no device synchronization.
    """

    active_assignments: torch.Tensor
    grouped_mm_rows: int
    padded_rows: int
    bmm_rows_per_expert: int
    padding_efficiency: torch.Tensor


@dataclass(frozen=True)
class FixedRouteForward:
    hidden_states: torch.Tensor
    padding: FixedRouteBatchStats


@dataclass(frozen=True)
class FixedRouteMoeCache:
    """Compact projection cache whose tensor payload is CUDA resident."""

    projection: ProjectionName
    num_samples: int
    seq_len: int
    num_experts: int
    assignments_per_sample: int
    # Assignment payload is expert-major, then sample-major, with exactly A
    # rows.  ``assignment_starts/counts`` address each (expert, sample) segment.
    token_indices: torch.Tensor
    route_weights: torch.Tensor
    assignment_counts: torch.Tensor
    assignment_starts: torch.Tensor
    target_inputs: torch.Tensor
    fixed_values: torch.Tensor
    target_biases: torch.Tensor | None
    expert_modules: tuple[nn.Module, ...]
    down_weights: torch.Tensor | None
    down_biases: torch.Tensor | None
    down_transform: Literal["identity", "online_full_had"]
    down_had_k: torch.Tensor | None
    down_had_K: int
    down_fp32_had: bool
    target_output_size: int
    output_hidden_size: int
    padding: FixedRoutePaddingStats

    @property
    def device(self) -> torch.device:
        return self.target_inputs.device

    def assert_cuda(self) -> None:
        tensors = {
            "token_indices": self.token_indices,
            "route_weights": self.route_weights,
            "assignment_counts": self.assignment_counts,
            "assignment_starts": self.assignment_starts,
            "target_inputs": self.target_inputs,
            "fixed_values": self.fixed_values,
            "target_biases": self.target_biases,
            "down_weights": self.down_weights,
            "down_biases": self.down_biases,
            "down_had_k": self.down_had_k,
        }
        expected = self.device
        if expected.type != "cuda":
            raise RuntimeError(
                f"fixed-route cache must be CUDA resident, got {expected}."
            )
        for name, tensor in tensors.items():
            if tensor is not None and tensor.device != expected:
                raise RuntimeError(
                    f"fixed-route cache {name} must be on {expected}, "
                    f"got {tensor.device}."
                )
        for expert_idx, expert in enumerate(self.expert_modules):
            _require_module_device(
                expert,
                device=expected,
                name=f"expert_modules[{expert_idx}]",
            )


def _require_module_device(
    module: nn.Module,
    *,
    device: torch.device,
    name: str,
) -> None:
    for parameter_name, parameter in module.named_parameters():
        if parameter.device != device:
            raise ValueError(
                f"{name}.{parameter_name} must remain on {device}, "
                f"got {parameter.device}."
            )
    for buffer_name, buffer in module.named_buffers():
        if buffer.device != device:
            raise ValueError(
                f"{name}.{buffer_name} must remain on {device}, "
                f"got {buffer.device}."
            )


def _tensor_output(value, *, source: str) -> torch.Tensor:
    output = value[0] if isinstance(value, (tuple, list)) else value
    if not torch.is_tensor(output):
        raise RuntimeError(
            f"{source} must return a Tensor or tensor-first tuple/list, "
            f"got {type(output).__name__}."
        )
    return output


def _projection_module(expert: nn.Module, projection: ProjectionName) -> nn.Module:
    module = getattr(expert, projection, None)
    if not isinstance(module, nn.Module):
        raise TypeError(
            f"expert.{projection} must be nn.Module, "
            f"got {type(module).__name__}."
        )
    return module


def _validate_target_module(
    module: nn.Module,
    linear: nn.Linear,
    *,
    label: str,
) -> None:
    """Ensure direct candidate bmm reproduces the module's post-linear path.

    Plain ``nn.Linear`` and the repository's ``ActQuantWrapper`` are accepted.
    The latter may transform/quantize its input (captured once below), but
    expert projection outputs are required to stay at 16 bits, so there is no
    candidate-dependent post-linear transform to reproduce.
    """

    if module is linear:
        return
    if getattr(module, "module", None) is not linear:
        raise TypeError(
            f"{label} must be the target Linear or a one-level wrapper around "
            "it."
        )
    input_quantizer = getattr(module, "quantizer", None)
    output_quantizer = getattr(module, "out_quantizer", None)
    if input_quantizer is None or output_quantizer is None:
        raise TypeError(
            f"{label} uses an unsupported target wrapper "
            f"{type(module).__name__}."
        )
    if int(getattr(input_quantizer, "bits", 16)) != 16:
        raise ValueError(
            f"{label} input quantization must be disabled for the frozen "
            "W4A16 packed grouped-mm path."
        )
    if int(getattr(output_quantizer, "bits", 16)) != 16:
        raise ValueError(
            f"{label} output quantization must be disabled for direct batched "
            "candidate replay."
        )


def _require_grouped_mm_capability(
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Fail closed before PyTorch can enter its host-offset fallback."""

    grouped_mm = getattr(torch, "_grouped_mm", None)
    if not callable(grouped_mm):
        raise RuntimeError(
            "packed fixed-route replay requires callable torch._grouped_mm."
        )
    if device.type != "cuda" or torch.version.cuda is None:
        raise RuntimeError(
            "packed fixed-route replay requires a CUDA PyTorch build."
        )
    if str(torch.__version__).split("+", 1)[0] != "2.9.1":
        raise RuntimeError(
            "packed fixed-route replay is validated only on PyTorch 2.9.1; "
            f"got {torch.__version__}."
        )
    if dtype != torch.bfloat16:
        raise ValueError(
            "packed fixed-route grouped-mm requires BF16 W4A16 operands; "
            f"got {dtype}."
        )
    capability = torch.cuda.get_device_capability(device)
    if capability != (10, 0):
        raise RuntimeError(
            "packed fixed-route grouped-mm requires the validated SM100 "
            "BF16 fast path.  Refusing PyTorch's host-offset fallback on "
            f"compute capability {capability}."
        )


def _underlying_linear(module: nn.Module, *, label: str) -> nn.Linear:
    inner = getattr(module, "module", module)
    if not isinstance(inner, nn.Linear):
        raise TypeError(
            f"{label} must be nn.Linear or a one-level wrapper around one, "
            f"got {type(module).__name__}."
        )
    return inner


def _down_replay_signature(
    module: nn.Module,
    linear: nn.Linear,
    *,
    label: str,
) -> tuple[
    Literal["identity", "online_full_had"],
    int,
    bool,
    torch.Tensor | None,
]:
    """Validate the W4A16 fixed-down subset reproduced by the packed path."""

    if module is linear:
        return ("identity", 1, False, None)
    if getattr(module, "module", None) is not linear:
        raise TypeError(
            f"{label} must be the down Linear or a one-level wrapper around it."
        )
    input_quantizer = getattr(module, "quantizer", None)
    output_quantizer = getattr(module, "out_quantizer", None)
    if input_quantizer is None or output_quantizer is None:
        raise TypeError(
            f"{label} uses unsupported wrapper {type(module).__name__}."
        )
    if (
        int(getattr(input_quantizer, "bits", 16)) != 16
        or int(getattr(output_quantizer, "bits", 16)) != 16
    ):
        raise ValueError(
            f"{label} must keep input/output activation quantization at 16 bits."
        )
    if bool(getattr(module, "online_partial_had", False)):
        raise ValueError(
            f"{label} online_partial_had is unsupported by packed replay."
        )
    if not bool(getattr(module, "online_full_had", False)):
        return ("identity", 1, False, None)
    had_K = int(getattr(module, "K", 0))
    if had_K <= 0:
        raise ValueError(f"{label}.K must be positive, got {had_K}.")
    had_k = getattr(module, "had_K", None)
    if had_k is not None:
        if not torch.is_tensor(had_k) or had_k.device != linear.weight.device:
            raise ValueError(
                f"{label}.had_K must remain on {linear.weight.device}."
            )
        if had_k.dtype != linear.weight.dtype:
            raise ValueError(
                f"{label}.had_K dtype {had_k.dtype} must match "
                f"{linear.weight.dtype}."
            )
    return (
        "online_full_had",
        had_K,
        bool(getattr(module, "fp32_had", False)),
        had_k,
    )


def _activation(expert: nn.Module, values: torch.Tensor) -> torch.Tensor:
    activation = getattr(expert, "act_fn", None)
    if activation is None:
        return torch.nn.functional.silu(values)
    if not callable(activation):
        raise TypeError(
            f"expert.act_fn must be callable, got {type(activation).__name__}."
        )
    return activation(values)


def _activation_signature(expert: nn.Module) -> tuple[object, ...]:
    activation = getattr(expert, "act_fn", None)
    if activation is None:
        return ("fallback_silu",)
    return (
        type(activation),
        getattr(activation, "__name__", None),
    )


def _compact_sample_expert_layout(
    routes: "PackedLayerRoutes",
    *,
    num_samples: int,
    seq_len: int,
    num_experts: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    FixedRoutePaddingStats,
]:
    """Stable-pack the canonical expert CSR by sample without padding."""

    assignment_count = int(routes.assignment_count)
    device = routes.device
    if len(routes) != num_experts:
        raise ValueError(
            f"student routes contain {len(routes)} experts, expected "
            f"{num_experts}."
        )
    routes.assert_resident_on(device)
    if device.type != "cuda":
        raise ValueError(
            f"student PackedLayerRoutes must remain CUDA resident, got {device}."
        )

    flat_ids = routes.flat_token_indices.to(dtype=torch.int64)
    if assignment_count:
        # One construction-time scalar read validates bounds.  There are no
        # route scalar reads or Amax reads in any refresh invocation.
        invalid = (
            (flat_ids < 0)
            | (flat_ids >= num_samples * seq_len)
        ).any()
        if bool(invalid.item()):
            raise ValueError(
                "student route contains a flat token index outside "
                f"[0, {num_samples * seq_len})."
            )

    expert_counts = (
        routes.expert_offsets[1:] - routes.expert_offsets[:-1]
    ).to(dtype=torch.int64)
    expert_ids = torch.repeat_interleave(
        torch.arange(
            num_experts,
            dtype=torch.int64,
            device=device,
        ),
        expert_counts,
        output_size=assignment_count,
    )
    sample_ids = torch.div(flat_ids, seq_len, rounding_mode="floor")
    pair_ids = expert_ids * num_samples + sample_ids
    pair_order = torch.argsort(pair_ids, stable=True)
    pair_counts = torch.bincount(
        pair_ids,
        minlength=num_samples * num_experts,
    ).reshape(num_experts, num_samples)
    if num_samples <= 0 or assignment_count % num_samples:
        raise ValueError(
            f"route assignment count {assignment_count} is not divisible by "
            f"num_samples={num_samples}."
        )
    assignments_per_sample = assignment_count // num_samples
    sample_counts = torch.bincount(sample_ids, minlength=num_samples)
    if bool((sample_counts != assignments_per_sample).any().item()):
        raise ValueError(
            "packed grouped-mm requires the same natural top-k assignment "
            "count for every calibration sample."
        )

    ordered_flat_ids = flat_ids.index_select(0, pair_order)
    token_indices = torch.remainder(
        ordered_flat_ids,
        seq_len,
    ).to(dtype=torch.int64)
    route_weights = routes.route_weights.index_select(0, pair_order)
    pair_prefix = torch.cumsum(pair_counts, dim=1) - pair_counts
    assignment_starts = (
        routes.expert_offsets[:-1, None] + pair_prefix
    ).to(dtype=torch.int64)
    stats = FixedRoutePaddingStats(
        active_assignments=assignment_count,
        packed_rows=assignment_count,
        padded_rows=0,
        rows_per_sample_expert=0,
        padding_efficiency=1.0,
    )
    return (
        pair_order,
        token_indices,
        route_weights,
        pair_counts,
        assignment_starts,
        assignments_per_sample,
        stats,
    )


@torch.no_grad()
def build_fixed_route_moe_cache(
    *,
    student_routes: "PackedLayerRoutes",
    projection: ProjectionName,
    expert_modules: Sequence[nn.Module],
    target_linears: Sequence[nn.Linear],
    mlp_inputs: torch.Tensor,
    seq_len: int,
) -> FixedRouteMoeCache:
    """Build one projection's immutable GPU cache.

    This function may perform two construction-time scalar reads (route bounds
    and ``M``).  Evaluation performs none and gathers only selected samples.
    """

    from realq_moe.precompute.routed_stats import PackedLayerRoutes

    if not isinstance(student_routes, PackedLayerRoutes):
        raise TypeError(
            "student_routes must be PackedLayerRoutes, got "
            f"{type(student_routes).__name__}."
        )
    if projection not in _PROJECTIONS:
        raise ValueError(
            f"projection must be one of {_PROJECTIONS}, got {projection!r}."
        )
    if (
        not isinstance(seq_len, int)
        or isinstance(seq_len, bool)
        or seq_len <= 0
    ):
        raise ValueError(f"seq_len must be positive, got {seq_len!r}.")
    if not torch.is_tensor(mlp_inputs) or mlp_inputs.dim() != 3:
        shape = tuple(mlp_inputs.shape) if torch.is_tensor(mlp_inputs) else None
        raise ValueError(
            f"mlp_inputs must have shape (N,T,H), got {shape}."
        )
    if mlp_inputs.device.type != "cuda":
        raise ValueError(
            f"mlp_inputs must remain CUDA resident, got {mlp_inputs.device}."
        )
    if int(mlp_inputs.shape[1]) != seq_len:
        raise ValueError(
            f"mlp_inputs sequence length {mlp_inputs.shape[1]} != {seq_len}."
        )
    experts = tuple(expert_modules)
    linears = tuple(target_linears)
    if not experts or len(experts) != len(linears):
        raise ValueError(
            "expert_modules and target_linears must be non-empty and aligned: "
            f"{len(experts)} != {len(linears)}."
        )
    if any(not isinstance(expert, nn.Module) for expert in experts):
        raise TypeError("expert_modules must contain only nn.Module objects.")
    if any(not isinstance(linear, nn.Linear) for linear in linears):
        raise TypeError("target_linears must contain underlying nn.Linear modules.")
    weight_dtypes = {linear.weight.dtype for linear in linears}
    if weight_dtypes != {mlp_inputs.dtype}:
        raise ValueError(
            "all target weights and mlp_inputs must share one dtype for the "
            f"candidate bmm; weights={weight_dtypes}, "
            f"inputs={mlp_inputs.dtype}."
        )
    _require_grouped_mm_capability(
        device=mlp_inputs.device,
        dtype=mlp_inputs.dtype,
    )
    activation_signatures = {
        _activation_signature(expert) for expert in experts
    }
    if len(activation_signatures) != 1:
        raise ValueError(
            "joint fixed-route experts must share one activation function; "
            f"got {activation_signatures}."
        )

    device = mlp_inputs.device
    target_modules = tuple(
        _projection_module(expert, projection) for expert in experts
    )
    down_modules = tuple(
        _projection_module(expert, "down_proj") for expert in experts
    )
    down_linears = tuple(
        _underlying_linear(
            module,
            label=f"expert_modules[{expert_idx}].down_proj",
        )
        for expert_idx, module in enumerate(down_modules)
    )
    for expert_idx, (expert, target_module, target_linear) in enumerate(
        zip(experts, target_modules, linears)
    ):
        _require_module_device(
            expert,
            device=device,
            name=f"expert_modules[{expert_idx}]",
        )
        _validate_target_module(
            target_module,
            target_linear,
            label=f"expert_modules[{expert_idx}].{projection}",
        )

    target_shapes = {
        (linear.out_features, linear.in_features) for linear in linears
    }
    if len(target_shapes) != 1:
        raise ValueError(
            f"joint target projection shapes differ: {sorted(target_shapes)}."
        )
    target_rows, target_columns = next(iter(target_shapes))
    if target_columns % 8:
        raise ValueError(
            "BF16 grouped-mm target input width must be 16-byte aligned; "
            f"got {target_columns}."
        )
    down_shapes = {
        (linear.out_features, linear.in_features) for linear in down_linears
    }
    if len(down_shapes) != 1:
        raise ValueError(
            f"fixed down projection shapes differ: {sorted(down_shapes)}."
        )
    down_output, down_input = next(iter(down_shapes))
    if down_input % 8:
        raise ValueError(
            "BF16 grouped-mm down input width must be 16-byte aligned; "
            f"got {down_input}."
        )
    down_dtypes = {linear.weight.dtype for linear in down_linears}
    if down_dtypes != {mlp_inputs.dtype}:
        raise ValueError(
            "fixed down weights must share the BF16 cache dtype; "
            f"got {down_dtypes}."
        )
    output_widths = {linear.out_features for linear in down_linears}
    if len(output_widths) != 1 or next(iter(output_widths)) <= 0:
        raise ValueError(
            f"down projection output widths differ or are invalid: "
            f"{sorted(output_widths)}."
        )
    output_hidden = next(iter(output_widths))

    down_signatures = tuple(
        _down_replay_signature(
            module,
            linear,
            label=f"expert_modules[{expert_idx}].down_proj",
        )
        for expert_idx, (module, linear) in enumerate(
            zip(down_modules, down_linears)
        )
    )
    down_transform, down_had_K, down_fp32_had, down_had_k = down_signatures[0]
    for expert_idx, signature in enumerate(down_signatures[1:], start=1):
        kind, had_K, fp32_had, had_k = signature
        if (kind, had_K, fp32_had) != (
            down_transform,
            down_had_K,
            down_fp32_had,
        ):
            raise ValueError(
                "all fixed down wrappers must share one transform; "
                f"expert 0={(down_transform, down_had_K, down_fp32_had)}, "
                f"expert {expert_idx}={(kind, had_K, fp32_had)}."
            )
        if (down_had_k is None) != (had_k is None):
            raise ValueError("fixed down had_K presence differs across experts.")
        if (
            down_had_k is not None
            and had_k is not None
            and not torch.equal(down_had_k, had_k)
        ):
            raise ValueError("fixed down had_K values differ across experts.")

    (
        route_order,
        token_indices,
        route_weights,
        assignment_counts,
        assignment_starts,
        assignments_per_sample,
        padding_stats,
    ) = _compact_sample_expert_layout(
        student_routes,
        num_samples=int(mlp_inputs.shape[0]),
        seq_len=seq_len,
        num_experts=len(experts),
    )
    num_samples = int(mlp_inputs.shape[0])
    target_inputs = torch.zeros(
        (student_routes.assignment_count, target_columns),
        dtype=mlp_inputs.dtype,
        device=device,
    )
    fixed_width = (
        target_rows if projection in ("up_proj", "gate_proj")
        else target_columns
    )
    fixed_values = (
        target_inputs
        if projection == "down_proj"
        else torch.zeros(
            (student_routes.assignment_count, fixed_width),
            dtype=mlp_inputs.dtype,
            device=device,
        )
    )
    mlp_flat = mlp_inputs.reshape(-1, int(mlp_inputs.shape[-1]))

    for expert_idx, (
        expert,
        target_module,
        target_linear,
    ) in enumerate(zip(experts, target_modules, linears)):
        start = student_routes._offset_values[expert_idx]
        end = student_routes._offset_values[expert_idx + 1]
        if start == end:
            continue
        source_positions = route_order[start:end]
        flat_ids = student_routes.flat_token_indices.index_select(
            0,
            source_positions,
        ).to(dtype=torch.int64)
        routed_inputs = mlp_flat.index_select(0, flat_ids)

        up_module = _projection_module(expert, "up_proj")
        gate_module = _projection_module(expert, "gate_proj")
        if projection == "up_proj":
            target_input = capture_inner_linear_input_without_gemm(
                target_module,
                target_linear,
                routed_inputs,
                label=f"expert[{expert_idx}].up_proj",
            )
            fixed = _activation(
                expert,
                _tensor_output(
                    gate_module(routed_inputs),
                    source=f"expert[{expert_idx}].gate_proj",
                ),
            )
        elif projection == "gate_proj":
            target_input = capture_inner_linear_input_without_gemm(
                target_module,
                target_linear,
                routed_inputs,
                label=f"expert[{expert_idx}].gate_proj",
            )
            fixed = _tensor_output(
                up_module(routed_inputs),
                source=f"expert[{expert_idx}].up_proj",
            )
        else:
            up = _tensor_output(
                up_module(routed_inputs),
                source=f"expert[{expert_idx}].up_proj",
            )
            gate = _tensor_output(
                gate_module(routed_inputs),
                source=f"expert[{expert_idx}].gate_proj",
            )
            product = _activation(expert, gate) * up
            target_input = capture_inner_linear_input_without_gemm(
                target_module,
                target_linear,
                product,
                label=f"expert[{expert_idx}].down_proj",
            )
            # The cached down operand is the exact input seen by the underlying
            # candidate Linear, including fixed Hadamard/input-quant transforms.
            fixed = target_input

        if tuple(target_input.shape) != (
            end - start,
            target_columns,
        ):
            raise RuntimeError(
                f"expert {expert_idx} target input shape "
                f"{tuple(target_input.shape)} != "
                f"({end - start}, {target_columns})."
            )
        if tuple(fixed.shape) != (
            end - start,
            fixed_width,
        ):
            raise RuntimeError(
                f"expert {expert_idx} fixed cache shape "
                f"{tuple(fixed.shape)} != "
                f"({end - start}, {fixed_width})."
            )
        target_inputs[start:end].copy_(target_input.detach())
        if projection != "down_proj":
            fixed_values[start:end].copy_(fixed.detach())

    biases = tuple(linear.bias for linear in linears)
    target_biases = None
    if any(bias is not None for bias in biases):
        target_biases = torch.stack(
            [
                (
                    bias.detach()
                    if bias is not None
                    else torch.zeros(
                        target_rows,
                        dtype=linears[0].weight.dtype,
                        device=device,
                    )
                )
                for bias in biases
            ],
            dim=0,
        )

    down_weights = None
    down_biases = None
    if projection != "down_proj":
        if down_input != target_rows:
            raise ValueError(
                "candidate up/gate output width must match fixed down input: "
                f"{target_rows} != {down_input}."
            )
        down_weights = torch.stack(
            [linear.weight.detach() for linear in down_linears],
            dim=0,
        )
        down_linear_biases = tuple(linear.bias for linear in down_linears)
        if any(bias is not None for bias in down_linear_biases):
            down_biases = torch.stack(
                [
                    (
                        bias.detach()
                        if bias is not None
                        else torch.zeros(
                            down_output,
                            dtype=mlp_inputs.dtype,
                            device=device,
                        )
                    )
                    for bias in down_linear_biases
                ],
                dim=0,
            )

    cache = FixedRouteMoeCache(
        projection=projection,
        num_samples=num_samples,
        seq_len=seq_len,
        num_experts=len(experts),
        assignments_per_sample=assignments_per_sample,
        token_indices=token_indices,
        route_weights=route_weights,
        assignment_counts=assignment_counts,
        assignment_starts=assignment_starts,
        target_inputs=target_inputs,
        fixed_values=fixed_values,
        target_biases=target_biases,
        expert_modules=experts,
        down_weights=down_weights,
        down_biases=down_biases,
        down_transform=down_transform,
        down_had_k=down_had_k,
        down_had_K=down_had_K,
        down_fp32_had=down_fp32_had,
        target_output_size=target_rows,
        output_hidden_size=output_hidden,
        padding=padding_stats,
    )
    cache.assert_cuda()
    logging.info(
        "[realq_moe.fixed_route] projection=%s assignments=%d "
        "packed_rows=%d padded_rows=0 efficiency=%.6f",
        projection,
        padding_stats.active_assignments,
        padding_stats.packed_rows,
        padding_stats.padding_efficiency,
    )
    return cache


def _apply_down_input_transform(
    cache: FixedRouteMoeCache,
    values: torch.Tensor,
) -> torch.Tensor:
    if cache.down_transform == "identity":
        return values
    if cache.down_transform != "online_full_had":
        raise RuntimeError(
            f"unsupported packed down transform {cache.down_transform!r}."
        )
    source_dtype = values.dtype
    transformed = values.float() if cache.down_fp32_had else values
    transformed = hadamard_utils.matmul_hadU_cuda(
        transformed,
        cache.down_had_k,
        cache.down_had_K,
    )
    return transformed.to(dtype=source_dtype)


def evaluate_fixed_route_moe(
    cache: FixedRouteMoeCache,
    candidate_weights: torch.Tensor,
    sample_indices: torch.Tensor,
) -> FixedRouteForward:
    """Replay selected samples with compact fixed routes and grouped GEMMs.

    ``candidate_weights`` must be the single ``(E,R,C)`` differentiable leaf.
    The returned hidden states contain only the sparse MLP output; callers add
    the cached post-attention residual.
    """

    if (
        not torch.is_tensor(candidate_weights)
        or candidate_weights.dim() != 3
    ):
        shape = (
            tuple(candidate_weights.shape)
            if torch.is_tensor(candidate_weights)
            else None
        )
        raise ValueError(
            f"candidate_weights must have shape (E,R,C), got {shape}."
        )
    if candidate_weights.device != cache.device:
        raise ValueError(
            f"candidate_weights must be on {cache.device}, "
            f"got {candidate_weights.device}."
        )
    if candidate_weights.dtype != torch.bfloat16:
        raise ValueError(
            "candidate_weights must be BF16 for the SM100 grouped-mm path, "
            f"got {candidate_weights.dtype}."
        )
    if not candidate_weights.is_contiguous():
        raise ValueError(
            "candidate_weights must be contiguous in natural (E,R,C) layout."
        )
    if not torch.is_tensor(sample_indices) or sample_indices.dim() != 1:
        raise ValueError("sample_indices must be a 1D tensor.")
    if sample_indices.device != cache.device:
        raise ValueError(
            f"sample_indices must be on {cache.device}, "
            f"got {sample_indices.device}."
        )
    if sample_indices.dtype != torch.int64:
        raise ValueError(
            f"sample_indices must be int64, got {sample_indices.dtype}."
        )
    if not sample_indices.is_contiguous():
        raise ValueError("sample_indices must be contiguous.")
    batch_size = int(sample_indices.numel())
    if batch_size <= 0:
        raise ValueError("sample_indices must select at least one sample.")

    selected_counts = cache.assignment_counts.index_select(
        1,
        sample_indices,
    )
    selected_starts = cache.assignment_starts.index_select(
        1,
        sample_indices,
    )
    experts = cache.num_experts
    selected_rows = batch_size * cache.assignments_per_sample
    flat_counts = selected_counts.reshape(-1)
    flat_starts = selected_starts.reshape(-1)
    segment_ids = torch.repeat_interleave(
        torch.arange(
            experts * batch_size,
            dtype=torch.int64,
            device=cache.device,
        ),
        flat_counts,
        output_size=selected_rows,
    )
    output_segment_starts = torch.cumsum(
        flat_counts,
        dim=0,
    ) - flat_counts
    source_positions = (
        flat_starts.index_select(0, segment_ids)
        + torch.arange(
            selected_rows,
            dtype=torch.int64,
            device=cache.device,
        )
        - output_segment_starts.index_select(0, segment_ids)
    )
    selected_inputs = cache.target_inputs.index_select(
        0,
        source_positions,
    )
    columns = int(selected_inputs.shape[1])
    expected_weight_shape = (
        experts,
        cache.target_output_size,
        columns,
    )
    if tuple(candidate_weights.shape) != expected_weight_shape:
        raise ValueError(
            "candidate weight/input shape mismatch: "
            f"{tuple(candidate_weights.shape)} != {expected_weight_shape}."
        )
    if selected_inputs.dtype != torch.bfloat16:
        raise ValueError(
            "cached grouped-mm inputs must be BF16, got "
            f"{selected_inputs.dtype}."
        )
    grouped_offsets = torch.cumsum(
        selected_counts.sum(dim=1),
        dim=0,
        dtype=torch.int32,
    ).contiguous()
    candidate_outputs = torch._grouped_mm(
        selected_inputs,
        candidate_weights.transpose(1, 2),
        offs=grouped_offsets,
    )
    expected_candidate_shape = (
        selected_rows,
        cache.target_output_size,
    )
    if tuple(candidate_outputs.shape) != expected_candidate_shape:
        raise RuntimeError(
            "grouped candidate output shape mismatch: "
            f"{tuple(candidate_outputs.shape)} != {expected_candidate_shape}."
        )
    assignment_experts = torch.div(
        segment_ids,
        batch_size,
        rounding_mode="floor",
    )
    if cache.target_biases is not None:
        candidate_outputs = candidate_outputs + cache.target_biases.index_select(
            0,
            assignment_experts,
        )

    selected_tokens = cache.token_indices.index_select(
        0,
        source_positions,
    )
    selected_weights = cache.route_weights.index_select(
        0,
        source_positions,
    )
    assignment_batches = torch.remainder(segment_ids, batch_size)
    destinations = selected_tokens + assignment_batches * cache.seq_len
    mixed = candidate_outputs.new_zeros(
        (batch_size * cache.seq_len, cache.output_hidden_size)
    )

    if cache.projection == "down_proj":
        expert_outputs = candidate_outputs
    else:
        fixed = cache.fixed_values.index_select(
            0,
            source_positions,
        )
        if cache.projection == "up_proj":
            down_inputs = fixed * candidate_outputs
        else:
            # Qwen3 experts share one configured activation. Applying expert
            # zero's callable to the additional leading expert dimension is
            # elementwise-equivalent and launches one kernel instead of E.
            activated = _activation(
                cache.expert_modules[0],
                candidate_outputs,
            )
            down_inputs = activated * fixed
        if cache.down_weights is None:
            raise RuntimeError(
                "packed up/gate replay is missing fixed down weights."
            )
        down_inputs = _apply_down_input_transform(cache, down_inputs)
        expert_outputs = torch._grouped_mm(
            down_inputs,
            cache.down_weights.transpose(1, 2),
            offs=grouped_offsets,
        )
        if cache.down_biases is not None:
            expert_outputs = expert_outputs + cache.down_biases.index_select(
                0,
                assignment_experts,
            )
    expected_output_shape = (
        selected_rows,
        cache.output_hidden_size,
    )
    if tuple(expert_outputs.shape) != expected_output_shape:
        raise RuntimeError(
            "packed expert output shape mismatch: "
            f"{tuple(expert_outputs.shape)} != {expected_output_shape}."
        )
    contributions = expert_outputs * selected_weights.to(
        dtype=expert_outputs.dtype
    ).unsqueeze(-1)
    # Rows are expert-major, so this is the closest one-kernel analogue of the
    # native expert-ascending loop.  The SM100 BF16 oracle test covers output,
    # all-expert gradients, cold experts, and repeatability.
    mixed.index_add_(0, destinations, contributions)

    active_assignments = selected_counts.sum()
    efficiency = torch.ones(
        (),
        dtype=torch.float32,
        device=cache.device,
    )
    return FixedRouteForward(
        hidden_states=mixed.reshape(
            batch_size,
            cache.seq_len,
            cache.output_hidden_size,
        ),
        padding=FixedRouteBatchStats(
            active_assignments=active_assignments,
            grouped_mm_rows=selected_rows,
            padded_rows=0,
            bmm_rows_per_expert=0,
            padding_efficiency=efficiency,
        ),
    )
