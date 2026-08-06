"""Natural-routing metadata and packed expert-saliency finalisation.

Qwen3-MoE invokes each expert on a ragged two-dimensional assignment tensor,
so the dense ``(batch, token, hidden)`` saliency contract is insufficient.
This module mirrors the Transformers 4.56.2 routing order exactly and stores
the assignment metadata needed to gather Stage-1 expert inputs.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from realq_moe import model_adapter
from realq_moe.parallel import env as parallel_env


@dataclass(frozen=True)
class PackedRouteBatch:
    """One sparse layer's complete route for one calibration batch.

    Assignments are grouped into a CSR-by-expert layout.  Within every expert
    the order is the exact Qwen3-MoE order for that batch: top-k slot first,
    then the model's flattened ``(sample, token)`` order.  Appending these
    packs in forward order therefore also preserves calibration-batch order.

    The tensors stay on the router's CUDA device.  Keeping one packed object
    per layer/batch avoids both the old ``3 * num_experts`` tiny objects and
    every device-to-host transfer in the Stage-0 hot path.
    """

    expert_offsets: torch.Tensor
    flat_token_indices: torch.Tensor
    topk_slots: torch.Tensor
    route_weights: torch.Tensor

    def __post_init__(self) -> None:
        _validate_packed_route_tensors(
            expert_offsets=self.expert_offsets,
            flat_token_indices=self.flat_token_indices,
            topk_slots=self.topk_slots,
            route_weights=self.route_weights,
            offset_values=None,
            label="PackedRouteBatch",
            validate_offset_contents=False,
        )

    @property
    def assignment_count(self) -> int:
        return int(self.flat_token_indices.numel())


class _ExpertRouteView(Mapping[str, torch.Tensor]):
    """Three zero-copy tensor views for one expert in a packed layer."""

    _FIELDS = (
        "flat_token_indices",
        "topk_slots",
        "route_weights",
    )

    def __init__(self, routes: "PackedLayerRoutes", expert_idx: int) -> None:
        self._routes = routes
        self._expert_idx = expert_idx

    def __len__(self) -> int:
        return len(self._FIELDS)

    def __iter__(self) -> Iterator[str]:
        return iter(self._FIELDS)

    def __getitem__(self, field_name: str) -> torch.Tensor:
        if field_name not in self._FIELDS:
            raise KeyError(field_name)
        start = self._routes._offset_values[self._expert_idx]
        end = self._routes._offset_values[self._expert_idx + 1]
        return getattr(self._routes, field_name)[start:end]


@dataclass(frozen=True)
class PackedLayerRoutes(Mapping[int, Mapping[str, torch.Tensor]]):
    """Canonical compact expert-major CSR for one sparse layer.

    The Mapping interface preserves ``routes[expert_idx][field]`` for existing
    consumers.  Those accesses return views; the object stores only four
    tensors, never 128 widened per-expert copies.
    """

    expert_offsets: torch.Tensor
    flat_token_indices: torch.Tensor
    topk_slots: torch.Tensor
    route_weights: torch.Tensor
    _offset_values: tuple[int, ...] = field(repr=False)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the full CSR, including serialized slice metadata."""

        _validate_packed_route_tensors(
            expert_offsets=self.expert_offsets,
            flat_token_indices=self.flat_token_indices,
            topk_slots=self.topk_slots,
            route_weights=self.route_weights,
            offset_values=self._offset_values,
            label="PackedLayerRoutes",
            validate_offset_contents=True,
        )

    @property
    def device(self) -> torch.device:
        return self.expert_offsets.device

    @property
    def assignment_count(self) -> int:
        return int(self.flat_token_indices.numel())

    def __len__(self) -> int:
        return int(self.expert_offsets.numel()) - 1

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self)))

    def __getitem__(self, expert_idx: int) -> Mapping[str, torch.Tensor]:
        if (
            not isinstance(expert_idx, int)
            or isinstance(expert_idx, bool)
            or expert_idx < 0
            or expert_idx >= len(self)
        ):
            raise KeyError(expert_idx)
        return _ExpertRouteView(self, expert_idx)

    def assert_resident_on(self, device: torch.device) -> None:
        expected = torch.device(device)
        if expected.type == "cuda" and expected.index is None:
            expected = torch.device(
                f"cuda:{torch.cuda.current_device()}"
            )
        actual = {
            self.expert_offsets.device,
            self.flat_token_indices.device,
            self.topk_slots.device,
            self.route_weights.device,
        }
        if actual != {expected}:
            raise RuntimeError(
                "packed route residency mismatch: "
                f"expected {expected}, got {sorted(map(str, actual))}."
            )


def _validate_packed_route_tensors(
    *,
    expert_offsets: torch.Tensor,
    flat_token_indices: torch.Tensor,
    topk_slots: torch.Tensor,
    route_weights: torch.Tensor,
    offset_values: tuple[int, ...] | None,
    label: str,
    validate_offset_contents: bool,
) -> None:
    tensors = (
        expert_offsets,
        flat_token_indices,
        topk_slots,
        route_weights,
    )
    if any(not torch.is_tensor(tensor) for tensor in tensors):
        raise TypeError(f"{label} fields must all be tensors.")
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise ValueError(
            f"{label} fields must share one device, got "
            f"{sorted(map(str, devices))}."
        )
    if expert_offsets.dtype != torch.int64 or expert_offsets.dim() != 1:
        raise ValueError(
            f"{label}.expert_offsets must be 1D int64, got "
            f"{tuple(expert_offsets.shape)} {expert_offsets.dtype}."
        )
    if int(expert_offsets.numel()) < 1:
        raise ValueError(f"{label}.expert_offsets must contain at least zero.")
    if (
        flat_token_indices.dtype != torch.int32
        or flat_token_indices.dim() != 1
    ):
        raise ValueError(
            f"{label}.flat_token_indices must be 1D int32, got "
            f"{tuple(flat_token_indices.shape)} "
            f"{flat_token_indices.dtype}."
        )
    if topk_slots.dtype != torch.uint8 or topk_slots.dim() != 1:
        raise ValueError(
            f"{label}.topk_slots must be 1D uint8, got "
            f"{tuple(topk_slots.shape)} {topk_slots.dtype}."
        )
    if route_weights.dim() != 1 or not route_weights.is_floating_point():
        raise ValueError(
            f"{label}.route_weights must be a 1D floating tensor, got "
            f"{tuple(route_weights.shape)} {route_weights.dtype}."
        )
    assignment_count = int(flat_token_indices.numel())
    if (
        int(topk_slots.numel()) != assignment_count
        or int(route_weights.numel()) != assignment_count
    ):
        raise ValueError(f"{label} assignment field lengths do not match.")
    if validate_offset_contents:
        invalid_offsets = torch.stack(
            (
                expert_offsets[0] != 0,
                expert_offsets[-1] != assignment_count,
                (expert_offsets[1:] < expert_offsets[:-1]).any(),
            )
        ).any()
        if bool(invalid_offsets.item()):
            raise ValueError(
                f"{label}.expert_offsets must be nondecreasing from zero to "
                f"{assignment_count}."
            )
    if offset_values is not None:
        if len(offset_values) != int(expert_offsets.numel()):
            raise ValueError(
                f"{label} offset metadata length {len(offset_values)} does "
                f"not match tensor length {expert_offsets.numel()}."
            )
        if not offset_values or offset_values[0] != 0:
            raise ValueError(f"{label} offsets must start at zero.")
        if offset_values[-1] != assignment_count:
            raise ValueError(
                f"{label} final offset {offset_values[-1]} does not match "
                f"assignment count {assignment_count}."
            )
        if any(
            right < left
            for left, right in zip(offset_values, offset_values[1:])
        ):
            raise ValueError(f"{label} offsets must be nondecreasing.")
        expected = torch.tensor(
            offset_values,
            dtype=torch.int64,
            device=expert_offsets.device,
        )
        if not torch.equal(expert_offsets, expected):
            raise ValueError(
                f"{label} tensor offsets differ from slice metadata."
            )


class RouteCaptureManager:
    """Capture teacher routes while the unmodified model performs Stage 0.

    ``begin_batch`` must be called immediately before each model forward.  A
    sparse-MLP pre-hook records ``(B,T)`` and the router hook reproduces the
    official full-softmax → top-k → optional renormalisation semantics, then
    stable-packs the official expert-mask traversal order. Projection saliency
    hooks retain only each expert's assignment count for backward validation.
    """

    def __init__(self, layers: list[nn.Module]) -> None:
        self._layers = layers
        self._plans = [model_adapter.describe_layer(layer) for layer in layers]
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._batch_local_start: int | None = None
        self._batch_size: int | None = None
        self._seq_len: int | None = None
        self._batch_serial = -1
        self._layer_shapes: dict[int, tuple[int, int, torch.dtype]] = {}
        self._layer_seen_serial: dict[int, int] = {}
        self._current_assignment_counts: list[dict[int, int]] = [
            {} for _ in self._plans
        ]
        self._batch_packs: list[list[PackedRouteBatch]] = []

        for layer_idx, (layer, plan) in enumerate(zip(layers, self._plans)):
            if not plan.is_sparse:
                self._batch_packs.append([])
                continue
            if plan.top_k > 256:
                raise RuntimeError(
                    f"layer {layer_idx} top_k={plan.top_k} cannot be stored "
                    "losslessly in the packed uint8 topk_slots field."
                )
            self._batch_packs.append([])
            mlp = layer.mlp
            self._handles.append(
                mlp.register_forward_pre_hook(
                    self._make_mlp_pre_hook(layer_idx)
                )
            )
            self._handles.append(
                mlp.gate.register_forward_hook(
                    self._make_router_hook(layer_idx)
                )
            )

    def begin_batch(
        self,
        *,
        local_start: int,
        batch_size: int,
        seq_len: int,
    ) -> None:
        if self._batch_local_start is not None:
            raise RuntimeError(
                "RouteCaptureManager.release_current must run after the "
                "previous model forward and before begin_batch."
            )
        if local_start < 0 or batch_size <= 0 or seq_len <= 0:
            raise ValueError(
                "invalid route batch context: "
                f"local_start={local_start}, B={batch_size}, T={seq_len}."
            )
        self._batch_local_start = int(local_start)
        self._batch_size = int(batch_size)
        self._seq_len = int(seq_len)
        self._batch_serial += 1
        self._layer_shapes.clear()

    def release_current(self) -> None:
        """Validate and release the just-completed forward's live route state.

        Expert output hooks close over plain Python assignment counts, not
        route tensors, so the current forward can be released before backward.
        """

        if self._batch_local_start is None:
            raise RuntimeError(
                "RouteCaptureManager.release_current called without an active "
                "Stage-0 batch."
            )
        if self._batch_size is None or self._seq_len is None:
            raise RuntimeError(
                "RouteCaptureManager active batch metadata is incomplete."
            )
        for layer_idx, plan in enumerate(self._plans):
            if not plan.is_sparse:
                continue
            if self._layer_seen_serial.get(layer_idx) != self._batch_serial:
                raise RuntimeError(
                    f"sparse router layer {layer_idx} did not run in Stage-0 "
                    f"batch serial {self._batch_serial}."
                )
            expected = self._batch_size * self._seq_len * plan.top_k
            if not self._batch_packs[layer_idx]:
                raise RuntimeError(
                    f"sparse router layer {layer_idx} produced no route pack."
                )
            actual = self._batch_packs[layer_idx][-1].assignment_count
            if actual != expected:
                raise RuntimeError(
                    f"sparse router layer {layer_idx} recorded {actual} "
                    f"assignments, expected B*T*top_k={expected}."
                )
            self._current_assignment_counts[layer_idx].clear()
        self._layer_shapes.clear()
        self._batch_local_start = None
        self._batch_size = None
        self._seq_len = None

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for counts in self._current_assignment_counts:
            counts.clear()
        self._layer_shapes.clear()
        self._batch_local_start = None
        self._batch_size = None
        self._seq_len = None

    def current_assignment_count(
        self,
        layer_idx: int,
        expert_idx: int,
    ) -> int:
        try:
            return self._current_assignment_counts[layer_idx][expert_idx]
        except KeyError as exc:
            raise RuntimeError(
                f"no current route for layer={layer_idx}, expert={expert_idx}; "
                "the expert projection fired before its router hook."
            ) from exc

    def finalize(
        self,
    ) -> list[Mapping[int, Mapping[str, torch.Tensor]]]:
        if self._batch_local_start is not None:
            raise RuntimeError(
                "RouteCaptureManager.finalize called before release_current."
            )
        out: list[Mapping[int, Mapping[str, torch.Tensor]]] = []
        for layer_idx, batch_packs in enumerate(self._batch_packs):
            plan = self._plans[layer_idx]
            if not plan.is_sparse:
                out.append({})
                continue
            if not batch_packs:
                raise RuntimeError(
                    f"sparse layer {layer_idx} produced no route batches."
                )
            route_device = batch_packs[0].expert_offsets.device
            route_weight_dtype = batch_packs[0].route_weights.dtype
            for batch_idx, pack in enumerate(batch_packs):
                if (
                    int(pack.expert_offsets.numel())
                    != plan.num_experts + 1
                ):
                    raise RuntimeError(
                        f"layer {layer_idx} batch {batch_idx} has "
                        f"{pack.expert_offsets.numel() - 1} experts; expected "
                        f"{plan.num_experts}."
                    )
                if pack.expert_offsets.device != route_device:
                    raise RuntimeError(
                        f"layer {layer_idx} batch {batch_idx} route device "
                        f"{pack.expert_offsets.device} differs from "
                        f"{route_device}."
                    )
                if pack.route_weights.dtype != route_weight_dtype:
                    raise RuntimeError(
                        f"layer {layer_idx} batch {batch_idx} route-weight "
                        f"dtype {pack.route_weights.dtype} differs from "
                        f"{route_weight_dtype}."
                    )
            # Batch packs are already expert-major and stable within each
            # expert.  Build each batch's final destination indices entirely on
            # the source device, then scatter-copy into one expert-major output.
            # This is a GPU k-way stable merge: no offsets or assignments are
            # staged through CPU, and no O(A log A) re-sort is required.
            counts_by_batch = torch.stack(
                [
                    pack.expert_offsets[1:] - pack.expert_offsets[:-1]
                    for pack in batch_packs
                ],
                dim=0,
            )
            total_counts = counts_by_batch.sum(dim=0)
            expert_offsets = torch.empty(
                plan.num_experts + 1,
                dtype=torch.int64,
                device=route_device,
            )
            expert_offsets[0] = 0
            torch.cumsum(total_counts, dim=0, out=expert_offsets[1:])
            assignment_count = sum(
                pack.assignment_count for pack in batch_packs
            )
            flat_token_indices = torch.empty(
                assignment_count,
                dtype=torch.int32,
                device=route_device,
            )
            topk_slots = torch.empty(
                assignment_count,
                dtype=torch.uint8,
                device=route_device,
            )
            route_weights = torch.empty(
                assignment_count,
                dtype=route_weight_dtype,
                device=route_device,
            )
            prefix_by_batch = (
                torch.cumsum(counts_by_batch, dim=0) - counts_by_batch
            )
            expert_ids = torch.arange(
                plan.num_experts,
                dtype=torch.int64,
                device=route_device,
            )
            for batch_idx, pack in enumerate(batch_packs):
                counts = counts_by_batch[batch_idx]
                assignment_experts = torch.repeat_interleave(
                    expert_ids,
                    counts,
                    output_size=pack.assignment_count,
                )
                source_positions = torch.arange(
                    pack.assignment_count,
                    dtype=torch.int64,
                    device=route_device,
                )
                within_expert = source_positions - pack.expert_offsets.index_select(
                    0, assignment_experts
                )
                destinations = (
                    expert_offsets.index_select(0, assignment_experts)
                    + prefix_by_batch[batch_idx].index_select(
                        0, assignment_experts
                    )
                    + within_expert
                )
                flat_token_indices.index_copy_(
                    0, destinations, pack.flat_token_indices
                )
                topk_slots.index_copy_(
                    0, destinations, pack.topk_slots
                )
                route_weights.index_copy_(
                    0, destinations, pack.route_weights
                )

            # The Mapping protocol needs Python slice bounds.  This is one
            # E+1 scalar metadata read per completed layer, never route tensor
            # backing storage or a runtime offload tier.
            offset_values = tuple(
                int(value) for value in expert_offsets.tolist()
            )
            layer_out = PackedLayerRoutes(
                expert_offsets=expert_offsets,
                flat_token_indices=flat_token_indices,
                topk_slots=topk_slots,
                route_weights=route_weights,
                _offset_values=offset_values,
            )
            if route_device.type == "cuda":
                layer_out.assert_resident_on(route_device)
            out.append(layer_out)
            batch_packs.clear()
        return out

    def _make_mlp_pre_hook(self, layer_idx: int):
        def hook(_module, inputs):
            if self._batch_local_start is None:
                raise RuntimeError(
                    "RouteCaptureManager.begin_batch must run before model forward."
                )
            if not inputs or not torch.is_tensor(inputs[0]):
                raise RuntimeError(
                    f"sparse MLP layer {layer_idx} did not receive a tensor input."
                )
            hidden = inputs[0]
            if hidden.dim() != 3:
                raise RuntimeError(
                    f"sparse MLP layer {layer_idx} expected (B,T,H), "
                    f"got {tuple(hidden.shape)}."
                )
            batch_size, seq_len = int(hidden.shape[0]), int(hidden.shape[1])
            if (
                batch_size != self._batch_size
                or seq_len != self._seq_len
            ):
                raise RuntimeError(
                    f"sparse MLP layer {layer_idx} shape {(batch_size, seq_len)} "
                    f"does not match active route batch "
                    f"{(self._batch_size, self._seq_len)}."
                )
            self._layer_shapes[layer_idx] = (
                batch_size,
                seq_len,
                hidden.dtype,
            )

        return hook

    def _make_router_hook(self, layer_idx: int):
        def hook(_module, inputs, outputs):
            if self._layer_seen_serial.get(layer_idx) == self._batch_serial:
                raise RuntimeError(
                    f"sparse router layer {layer_idx} ran more than once in "
                    "one Stage-0 model forward."
                )
            self._layer_seen_serial[layer_idx] = self._batch_serial
            if layer_idx not in self._layer_shapes:
                raise RuntimeError(
                    f"sparse router layer {layer_idx} ran before MLP pre-hook."
                )
            batch_size, seq_len, hidden_dtype = self._layer_shapes[layer_idx]
            router_logits = (
                outputs[0]
                if isinstance(outputs, (tuple, list))
                else outputs
            )
            if router_logits.dim() != 2:
                raise RuntimeError(
                    f"router layer {layer_idx} expected 2D logits, "
                    f"got {tuple(router_logits.shape)}."
                )
            plan = self._plans[layer_idx]
            expected_shape = (
                batch_size * seq_len,
                plan.num_experts,
            )
            if tuple(router_logits.shape) != expected_shape:
                raise RuntimeError(
                    f"router layer {layer_idx} logits "
                    f"{tuple(router_logits.shape)} != {expected_shape}."
                )

            # Match transformers.models.qwen3_moe.modeling_qwen3_moe
            # Qwen3MoeSparseMoeBlock.forward exactly.
            routing_weights = F.softmax(
                router_logits.detach(), dim=1, dtype=torch.float
            )
            routing_weights, selected_experts = torch.topk(
                routing_weights, plan.top_k, dim=-1
            )
            mlp = self._layers[layer_idx].mlp
            if bool(mlp.norm_topk_prob):
                routing_weights = routing_weights / routing_weights.sum(
                    dim=-1, keepdim=True
                )
            routing_weights = routing_weights.to(hidden_dtype)

            # The official per-expert ``torch.where(expert_mask[e])`` walks
            # the mask in slot-major, flattened-token-minor order.  Flatten
            # the same axes first, then stable-group by expert to produce one
            # CSR pack without E independent D2H copies.
            num_tokens = batch_size * seq_len
            expert_ids = selected_experts.transpose(0, 1).reshape(-1)
            weights = routing_weights.transpose(0, 1).reshape(-1)

            # ``stable=True`` is semantically important: equal expert ids
            # retain the exact slot-major/token-major upstream order.
            order = torch.argsort(expert_ids, stable=True)
            expert_ids = expert_ids[order]
            weights = weights[order]
            topk_slots = torch.div(
                order, num_tokens, rounding_mode="floor"
            )
            flat_in_batch = torch.remainder(order, num_tokens)

            counts = torch.bincount(
                expert_ids, minlength=plan.num_experts
            )
            offsets = torch.empty(
                plan.num_experts + 1,
                dtype=torch.int64,
                device=counts.device,
            )
            offsets[0] = 0
            offsets[1:] = torch.cumsum(counts, dim=0)
            flat_local = (
                int(self._batch_local_start) * seq_len + flat_in_batch
            )
            max_flat_local = (
                (int(self._batch_local_start) + batch_size) * seq_len - 1
            )
            if max_flat_local > 2**31 - 1:
                raise OverflowError(
                    "route flat_token_indices exceed int32 capacity."
                )

            # Keep the compact route pack on the router device.  Finalize
            # stable-merges these batches into one equally compact
            # PackedLayerRoutes object.
            pack = PackedRouteBatch(
                expert_offsets=offsets.detach().to(dtype=torch.int64),
                flat_token_indices=flat_local.detach().to(
                    dtype=torch.int32
                ),
                topk_slots=topk_slots.detach().to(
                    dtype=torch.uint8
                ),
                route_weights=weights.detach().to(
                    dtype=hidden_dtype
                ),
            )
            if pack.assignment_count != num_tokens * plan.top_k:
                raise RuntimeError(
                    f"layer {layer_idx} packed {pack.assignment_count} "
                    f"assignments, expected {num_tokens * plan.top_k}."
                )
            self._batch_packs[layer_idx].append(pack)
            # Expert output hooks use their output row extent directly.  Do
            # not read 128 CUDA counters back into Python here.
            self._current_assignment_counts[layer_idx].clear()

        return hook


def _collective_device(
    preferred: torch.device | None = None,
) -> torch.device:
    if preferred is not None:
        return preferred
    if (
        parallel_env.is_dist_available_and_initialized()
        and "nccl" in str(dist.get_backend()).lower()
    ):
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def packed_clip_expert_saliency_(
    tensors: list[torch.Tensor],
    percentile: float | None,
) -> torch.Tensor:
    """Clip E local expert tensors with one packed collective.

    Returns the global element count for each expert.  Local-empty experts are
    valid as long as another rank has assignments; globally empty experts are
    reported with count zero so the caller can apply the configured cold
    policy before Stage 1.
    """

    num_experts = len(tensors)
    if num_experts == 0:
        return torch.empty(0, dtype=torch.int64)
    source_device = tensors[0].device
    for expert_idx, tensor in enumerate(tensors):
        if tensor.dim() != 3:
            raise ValueError(
                f"expert {expert_idx} saliency must be (A,1,G), "
                f"got {tuple(tensor.shape)}."
            )
        if tensor.device != source_device:
            raise ValueError(
                "expert saliency tensors must share one device; "
                f"expert 0 is on {source_device}, expert {expert_idx} is on "
                f"{tensor.device}."
            )

    device = _collective_device(source_device)
    local_counts = torch.tensor(
        [tensor.numel() for tensor in tensors],
        dtype=torch.int64,
        device=device,
    )
    world = parallel_env.get_world_size()
    if world > 1:
        gathered_counts = [
            torch.empty_like(local_counts) for _ in range(world)
        ]
        dist.all_gather(gathered_counts, local_counts)
        counts_by_rank = torch.stack(gathered_counts, dim=0)
    else:
        counts_by_rank = local_counts.unsqueeze(0)
    global_counts = counts_by_rank.sum(dim=0)

    if percentile is None or not (0.0 < float(percentile) < 1.0):
        return global_counts

    local_flat = (
        torch.cat([tensor.reshape(-1) for tensor in tensors])
        if any(tensor.numel() for tensor in tensors)
        else torch.empty(
            0, dtype=torch.float32, device=device
        )
    ).to(device=device, dtype=torch.float32)
    local_total = int(local_flat.numel())
    totals = counts_by_rank.sum(dim=1)
    max_total = int(totals.max().item())
    padded = torch.zeros(max_total, dtype=torch.float32, device=device)
    if local_total:
        padded[:local_total].copy_(local_flat)
    if world > 1:
        gathered_values = [
            torch.empty_like(padded) for _ in range(world)
        ]
        dist.all_gather(gathered_values, padded)
    else:
        gathered_values = [padded]

    caps = torch.zeros(num_experts, dtype=torch.float32, device=device)
    if parallel_env.is_main():
        rank_offsets = torch.zeros_like(counts_by_rank)
        if num_experts > 1:
            rank_offsets[:, 1:] = torch.cumsum(
                counts_by_rank[:, :-1], dim=1
            )
        for expert_idx in range(num_experts):
            pieces = []
            for rank_idx in range(world):
                count = int(counts_by_rank[rank_idx, expert_idx].item())
                if count == 0:
                    continue
                start = int(rank_offsets[rank_idx, expert_idx].item())
                pieces.append(
                    gathered_values[rank_idx][start : start + count]
                )
            if pieces:
                values = torch.cat(pieces, dim=0)
                caps[expert_idx] = torch.quantile(
                    values, float(percentile)
                )
    if world > 1:
        dist.broadcast(caps, src=0)

    for expert_idx, tensor in enumerate(tensors):
        if tensor.numel():
            tensor.clamp_(max=caps[expert_idx])
    return global_counts


def global_expert_coverage(
    routes: list[Mapping[int, Mapping[str, torch.Tensor]]],
    *,
    seq_len: int,
) -> list[dict[str, torch.Tensor]]:
    """Compute globally reduced route coverage for every sparse layer.

    Calibration shards are disjoint across data-parallel ranks, so summing
    rank-local unique token/sample counts produces the corresponding global
    counts even though ``flat_token_indices`` are local to each rank's shard.
    ``affinity_mass`` is accumulated in fp64; all other fields are int64.
    """

    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}.")
    first_route_tensor = next(
        (
            payload["flat_token_indices"]
            for layer_routes in routes
            for payload in layer_routes.values()
        ),
        None,
    )
    device = _collective_device(
        first_route_tensor.device
        if first_route_tensor is not None
        else None
    )
    world = parallel_env.get_world_size()
    out: list[dict[str, torch.Tensor]] = []
    for layer_idx, layer_routes in enumerate(routes):
        num_experts = len(layer_routes)
        if world > 1:
            expert_extent = torch.tensor(
                [num_experts, -num_experts],
                dtype=torch.int64,
                device=device,
            )
            dist.all_reduce(expert_extent, op=dist.ReduceOp.MAX)
            max_experts = int(expert_extent[0].item())
            min_experts = -int(expert_extent[1].item())
            if min_experts != max_experts:
                raise RuntimeError(
                    f"layer {layer_idx} route expert extent differs across "
                    f"ranks: min={min_experts}, max={max_experts}."
                )
        if num_experts == 0:
            out.append({})
            continue
        local_counts = torch.zeros(
            (3, num_experts), dtype=torch.int64, device=device
        )
        local_affinity = torch.zeros(
            num_experts, dtype=torch.float64, device=device
        )
        local_error: str | None = None
        try:
            for expert_idx in range(num_experts):
                try:
                    payload = layer_routes[expert_idx]
                except KeyError as exc:
                    raise RuntimeError(
                        f"route payload is missing expert {expert_idx}."
                    ) from exc
                flat_tokens = payload["flat_token_indices"]
                topk_slots = payload["topk_slots"]
                route_weights = payload["route_weights"]
                if (
                    flat_tokens.dim() != 1
                    or topk_slots.dim() != 1
                    or route_weights.dim() != 1
                ):
                    raise RuntimeError(
                        f"expert {expert_idx} route fields must all be "
                        "one-dimensional."
                    )
                assignments = int(flat_tokens.numel())
                if (
                    int(topk_slots.numel()) != assignments
                    or int(route_weights.numel()) != assignments
                ):
                    raise RuntimeError(
                        f"expert {expert_idx} route field lengths do not match."
                    )
                if assignments:
                    if bool((flat_tokens < 0).any().item()):
                        raise RuntimeError(
                            f"expert {expert_idx} has a negative flat token "
                            "index."
                        )
                    if not bool(torch.isfinite(route_weights).all().item()):
                        raise RuntimeError(
                            f"expert {expert_idx} has a non-finite route "
                            "weight."
                        )
                    if bool((route_weights < 0).any().item()):
                        raise RuntimeError(
                            f"expert {expert_idx} has a negative route weight."
                        )
                    unique_tokens = int(torch.unique(flat_tokens).numel())
                    # torch.topk returns distinct expert ids for a token, so a
                    # duplicate token within one expert would prove route
                    # packing or ordering corruption.
                    if unique_tokens != assignments:
                        raise RuntimeError(
                            f"expert {expert_idx} contains "
                            f"{assignments - unique_tokens} duplicate token "
                            "assignments."
                        )
                    unique_samples = int(
                        torch.unique(
                            torch.div(
                                flat_tokens,
                                seq_len,
                                rounding_mode="floor",
                            )
                        ).numel()
                    )
                    affinity = route_weights.to(torch.float64).sum()
                else:
                    unique_tokens = 0
                    unique_samples = 0
                    affinity = torch.tensor(
                        0.0, dtype=torch.float64, device=device
                    )
                local_counts[0, expert_idx] = assignments
                local_counts[1, expert_idx] = unique_tokens
                local_counts[2, expert_idx] = unique_samples
                local_affinity[expert_idx] = affinity
        except Exception as exc:
            local_error = str(exc)

        failed = torch.tensor(
            [int(local_error is not None)],
            dtype=torch.int32,
            device=device,
        )
        if world > 1:
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if bool(failed.item()):
            detail = (
                local_error
                or "a peer rank reported an invalid route payload"
            )
            raise RuntimeError(
                f"layer {layer_idx} route coverage validation failed: {detail}"
            )

        counts = local_counts
        affinity = local_affinity
        if world > 1:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(affinity, op=dist.ReduceOp.SUM)
        out.append(
            {
                "assignment_count": counts[0],
                "unique_token_count": counts[1],
                "unique_sample_count": counts[2],
                "affinity_mass": affinity,
            }
        )
    return out


def global_assignment_counts(
    routes: list[Mapping[int, Mapping[str, torch.Tensor]]],
) -> list[torch.Tensor]:
    """Return one globally reduced assignment-count vector per sparse layer."""

    first_route_tensor = next(
        (
            payload["flat_token_indices"]
            for layer_routes in routes
            for payload in layer_routes.values()
        ),
        None,
    )
    device = _collective_device(
        first_route_tensor.device
        if first_route_tensor is not None
        else None
    )
    world = parallel_env.get_world_size()
    out: list[torch.Tensor] = []
    for layer_routes in routes:
        if not layer_routes:
            out.append(
                torch.empty(0, dtype=torch.int64, device=device)
            )
            continue
        local = torch.tensor(
            [
                layer_routes[expert_idx]["flat_token_indices"].numel()
                for expert_idx in range(len(layer_routes))
            ],
            dtype=torch.int64,
            device=device,
        )
        if world > 1:
            dist.all_reduce(local, op=dist.ReduceOp.SUM)
        out.append(local)
    return out
