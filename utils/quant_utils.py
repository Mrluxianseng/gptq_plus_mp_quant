# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import math

import torch
import torch.nn as nn
from torch._tensor import Tensor
import transformers

from utils import hadamard_utils, model_utils


def get_minq_maxq(bits, sym):
    if sym:
        maxq = torch.tensor(2 ** (bits - 1) - 1)
        minq = -maxq - 1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0

    return minq, maxq


def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero


def asym_dequant(q, scale, zero):
    return scale * (q - zero)


def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))


def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
    return q, scale


def sym_dequant(q, scale):
    return scale * q


def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))


_W_CLIP_SEARCH_IMPLEMENTATIONS = frozenset(
    {"cartesian_legacy", "symmetric_union_exact"}
)
_W_CLIP_UPDATE_IMPLEMENTATIONS = frozenset({"guarded", "where_out"})


def _symmetric_union_candidates_and_first_keys(
    xmin, xmax, *, grid, candidate_count
):
    """Return symmetric endpoint candidates and their first legacy visits.

    The legacy symmetric MSE search visits every ``(i, j)`` pair in
    lexicographic order and derives its clipping range as::

        max(abs((1 - i / grid) * xmin), (1 - j / grid) * xmax)

    A maximum is one of its two operands.  Consequently, the complete
    ``candidate_count ** 2`` range set is covered by the union of the
    ``candidate_count`` negative and positive endpoints.  ``first_keys``
    records the earliest legacy pair that produces each endpoint, per
    row/group lane.  It is later used to preserve the observable first-winner
    behavior of the legacy strict-``<`` update.

    Each scalar multiplication stays in the same order as the legacy loops.
    In particular, ``abs`` remains after multiplication so signed zero and
    floating-point rounding are not silently changed.
    """

    negative_candidates = torch.stack(
        [
            torch.abs((1 - i / grid) * xmin)
            for i in range(candidate_count)
        ]
    )
    positive_candidates = torch.stack(
        [
            (1 - j / grid) * xmax
            for j in range(candidate_count)
        ]
    )

    sentinel = candidate_count * candidate_count
    key_shape = negative_candidates.shape
    negative_keys = torch.full(
        key_shape, sentinel, dtype=torch.long, device=xmin.device
    )
    positive_keys = torch.full(
        key_shape, sentinel, dtype=torch.long, device=xmin.device
    )
    key_broadcast_shape = (candidate_count,) + (1,) * xmin.ndim
    endpoint_indices = torch.arange(
        candidate_count, dtype=torch.long, device=xmin.device
    ).reshape(key_broadcast_shape)

    # These are cheap endpoint metadata comparisons, not weight QDQ/error
    # evaluations.  Avoid an MxMxlane temporary because grouped production
    # matrices can contain many lanes.
    for j in range(candidate_count):
        eligible = positive_candidates[j].unsqueeze(0) <= negative_candidates
        unassigned = negative_keys == sentinel
        visit_keys = endpoint_indices * candidate_count + j
        negative_keys = torch.where(
            eligible & unassigned, visit_keys, negative_keys
        )
    for i in range(candidate_count):
        eligible = negative_candidates[i].unsqueeze(0) <= positive_candidates
        unassigned = positive_keys == sentinel
        visit_keys = i * candidate_count + endpoint_indices
        positive_keys = torch.where(
            eligible & unassigned, visit_keys, positive_keys
        )

    return (
        torch.cat([negative_candidates, positive_candidates], dim=0),
        torch.cat([negative_keys, positive_keys], dim=0),
    )


def _select_symmetric_union_scale(
    x,
    xmin,
    xmax,
    *,
    maxq,
    norm,
    grid,
    candidate_count,
    grouped_error_order,
):
    """Evaluate 2M symmetric ranges and select the exact legacy winner.

    ``x`` has one final reduction dimension and ``xmin``/``xmax`` contain its
    lane-wise endpoints.  Selection is lexicographic in
    ``(error, first_legacy_visit)``.  That is equivalent to the legacy
    sequential strict-``<`` scan, including duplicate/tied ranges and the
    case where every candidate error is ``inf`` or ``nan``.
    """

    candidates, first_keys = _symmetric_union_candidates_and_first_keys(
        xmin,
        xmax,
        grid=grid,
        candidate_count=candidate_count,
    )
    scales = []
    errors = []
    for candidate in candidates.unbind(0):
        scale = candidate.clamp(min=1e-5) / maxq
        q = sym_quant_dequant(x, scale.unsqueeze(-1), maxq)
        if grouped_error_order:
            # Preserve find_params_weight_groupwise's exact expression.
            err = (q - x).abs().pow(norm).sum(-1)
        else:
            # Preserve find_params's mutation/kernel order.
            q -= x
            q.abs_()
            q.pow_(norm)
            err = torch.sum(q, -1)
        scales.append(scale)
        errors.append(err)
    scales = torch.stack(scales)
    errors = torch.stack(errors)

    sentinel = candidate_count * candidate_count
    valid_candidate = first_keys < sentinel
    # Legacy starts at +inf and updates only for ``err < best``.  Therefore
    # +inf and NaN are never winners; finite overflow in every candidate keeps
    # the initial unclipped scale.
    improving_error = errors < float("inf")
    eligible_errors = torch.where(
        valid_candidate & improving_error,
        errors,
        torch.full_like(errors, float("inf")),
    )
    best_error = eligible_errors.amin(dim=0)
    tied_for_best = (
        valid_candidate
        & improving_error
        & (errors == best_error.unsqueeze(0))
    )
    tie_keys = torch.where(
        tied_for_best,
        first_keys,
        torch.full_like(first_keys, sentinel),
    )
    winner = tie_keys.argmin(dim=0)
    selected = torch.gather(scales, 0, winner.unsqueeze(0)).squeeze(0)

    initial = (
        torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5) / maxq
    )
    has_winner = best_error < float("inf")
    result = initial.clone()
    result[has_winner] = selected[has_winner]
    return result


class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, maxq):
        scale = scale.to(x.device)
        q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
        return scale * q

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: just pass the gradient through
        return grad_output, None, None


class AsymSTEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero, maxq):
        scale = scale.to(x.device)
        zero = zero.to(x.device)
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class ActQuantizer(torch.nn.Module):
    """
    A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
    for the activations.
    """

    def __init__(self) -> None:
        super(ActQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(1))
        self.register_buffer("zero", torch.zeros(1))
        # These values are derived from the primitive runtime configuration
        # and re-estimated for every input token. Persisting them makes a
        # checkpoint depend on whichever sample happened to run last while
        # still failing to restore ``bits/groupsize/sym/clip_ratio``.
        self._non_persistent_buffers_set.update({"maxq", "scale", "zero"})
        self.bits = 16
        self.groupsize = -1
        self.sym = False
        self.clip_ratio = 1.0

    def free(self) -> None:
        self.zero = None
        self.scale = None

    def forward(self, x):
        x_dtype = x.dtype
        if self.bits == 16:
            return x
        elif self.sym:
            return STEQuantize.apply(x, self.scale, self.maxq).to(x_dtype)
        return AsymSTEQuantize.apply(x, self.scale, self.zero, self.maxq).to(x_dtype)

    # Different from `forward`, this method returns quantized integers, scales (and zeros if asymmetric).
    def quantize(self, x):
        if self.sym:
            return sym_quant(x, self.scale, self.maxq)
        else:
            return asym_quant(x, self.scale, self.zero, self.maxq)

    def configure(
        self, bits: int, groupsize: int = -1, sym: bool = False, clip_ratio: float = 1.0
    ) -> None:
        if not isinstance(bits, int) or isinstance(bits, bool) or not (2 <= bits <= 16):
            raise ValueError(
                "Activation/KV bit-width must be an integer in [2, 16], where "
                f"16 disables fake quantization; got {bits!r}."
            )
        if not isinstance(groupsize, int) or isinstance(groupsize, bool):
            raise ValueError(f"groupsize must be an integer; got {groupsize!r}.")
        if groupsize != -1 and groupsize <= 0:
            raise ValueError(
                f"groupsize must be -1 (per-token) or positive; got {groupsize}."
            )
        if not math.isfinite(float(clip_ratio)) or not (0.0 < clip_ratio <= 1.0):
            raise ValueError(
                f"Activation/KV clip ratio must be finite and in (0, 1]; got {clip_ratio}."
            )
        _, self.maxq = get_minq_maxq(bits, sym)
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio

    def find_params_per_token_groupwise(self, x) -> None:
        """Find one dynamic range per token and feature group.

        All leading dimensions identify independent token rows; grouping is
        exclusively over the last (feature) dimension.  A short final group is
        valid.  Including zero in every affine range mirrors the ordinary
        per-token path and avoids a negative zero-point for all-positive groups.
        """
        if x.ndim == 0 or x.shape[-1] == 0:
            raise ValueError(
                "Activation/KV group quantization requires a non-empty feature "
                f"dimension; got shape {tuple(x.shape)}."
            )

        scale_groups = []
        zero_groups = []
        for start in range(0, x.shape[-1], self.groupsize):
            end = min(start + self.groupsize, x.shape[-1])
            group = x[..., start:end]
            rows = group.reshape(-1, end - start)
            zero_ref = torch.zeros(rows.shape[0], device=x.device, dtype=x.dtype)
            xmin = torch.minimum(rows.min(dim=1).values, zero_ref) * self.clip_ratio
            xmax = torch.maximum(rows.max(dim=1).values, zero_ref) * self.clip_ratio

            if self.sym:
                absmax = torch.maximum(torch.abs(xmin), xmax)
                all_zero = absmax == 0
                scale = absmax / self.maxq
                scale[all_zero] = 1
                zero = torch.zeros_like(scale)
            else:
                all_zero = (xmin == 0) & (xmax == 0)
                xmin = xmin.clone()
                xmax = xmax.clone()
                xmin[all_zero] = -1
                xmax[all_zero] = +1
                scale = (xmax - xmin) / self.maxq
                zero = torch.round(-xmin / scale)

            expanded_shape = (*group.shape[:-1], end - start)
            scale_groups.append(
                scale.unsqueeze(1).expand(-1, end - start).reshape(expanded_shape)
            )
            zero_groups.append(
                zero.unsqueeze(1).expand(-1, end - start).reshape(expanded_shape)
            )

        self.scale = torch.cat(scale_groups, dim=-1)
        self.zero = torch.cat(zero_groups, dim=-1)

    def find_params(self, x) -> None:
        if self.bits == 16:
            return

        dev = x.device
        self.maxq = self.maxq.to(dev)

        init_shape = x.shape

        if self.groupsize > 0:
            # group-wise per-token quantization
            self.find_params_per_token_groupwise(x)
            # utils.cleanup_memory(verbos=False)
            return

        reshaped_x = x.reshape((-1, x.shape[-1]))

        tmp = torch.zeros(reshaped_x.shape[0], device=dev)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = (xmax / self.maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
            self.scale[tmp] = 1
            self.scale = self.scale.reshape(init_shape)
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

            self.scale = (
                self.scale.unsqueeze(1)
                .repeat(1, reshaped_x.shape[-1])
                .reshape(init_shape)
            )
            self.zero = (
                self.zero.unsqueeze(1)
                .repeat(1, reshaped_x.shape[-1])
                .reshape(init_shape)
            )


class ActQuantWrapper(torch.nn.Module):
    """
    This class is a wrapper for the activation quantization.
    We extract the FP features in the forward pass and quantize the rest using
    the self.quantizer object.
    If a rotation Q is provided, the weight matrix will be rotated,
    a pre-forward hook will be registered to rotate the activation before quantization.
    """

    def __init__(self, module: torch.nn.Linear) -> None:
        super(ActQuantWrapper, self).__init__()
        # assert isinstance(module, torch.nn.Linear)
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.quantizer = ActQuantizer()
        self.out_quantizer = ActQuantizer()
        self.register_buffer("had_K", torch.tensor(0))
        self._buffers["had_K"] = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.had_dim = 0
        self.fp32_had = False

    def extra_repr(self) -> str:
        str_ = f"Input Quantizer Bits: {self.quantizer.bits}"
        if self.quantizer.bits < 16:
            str_ += (
                f" (Asymmetric Per-Token)"
                if not self.quantizer.sym
                else f" (Symmetric Per-Token)"
            )

        str_ += f"\nOutput Quantizer Bits: {self.out_quantizer.bits}"
        if self.out_quantizer.bits < 16:
            str_ += (
                f" (Asymmetric Per-Token)"
                if not self.out_quantizer.sym
                else f" (Symmetric Per-Token)"
            )

        return str_

    def forward(self, x, R1=None, R2=None, transpose=False):
        x_dtype = x.dtype

        # Rotate, if needed
        if self.online_full_had:
            if self.fp32_had:  # Full Hadamard in FP32
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K).to(
                    x_dtype
                )
            else:  # Full Hadamard in FP16
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)

        elif self.online_partial_had:
            # todo: implement this in QAttention to avoid reshaping!

            if self.fp32_had:
                x = x.float()

            init_shape = x.shape
            if self.K == 1:
                x = (
                    hadamard_utils.HadamardTransform.apply(
                        x.reshape(
                            -1, init_shape[-1] // self.had_dim, self.had_dim
                        ).transpose(1, 2)
                    )
                    / math.sqrt(init_shape[-1] // self.had_dim)
                ).transpose(1, 2)
            else:
                x = (
                    self.had_K.to(x.dtype)
                    @ x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim)
                ) / math.sqrt(init_shape[-1] // self.had_dim)

            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)

        if self.quantizer.bits < 16:  # Quantize, if needed
            self.quantizer.find_params(x)
            x = self.quantizer(x).to(x_dtype)
            self.quantizer.free()
        if R1 is not None:
            x = self.module(x, R1, R2, transpose).to(x_dtype)
        else:
            x = self.module(x).to(x_dtype)

        if self.out_quantizer.bits < 16:  # Quantize the output, if needed
            self.out_quantizer.find_params(x)
            x = self.out_quantizer(x).to(x_dtype)
            self.out_quantizer.free()

        return x


class _PrevalidatedWeightFakeQuant:
    """Opaque state for :meth:`WeightQuantizer._fake_quantize_prevalidated`.

    REAL-Q quantizes one weight column at a time.  The public
    :meth:`WeightQuantizer.fake_quantize` entry point deliberately validates
    readiness, row slicing, and natural-column coordinates on every call.
    Repeating those tensor-wide checks in the inner GPTQ loop is expensive,
    especially because turning a CUDA boolean into a Python branch
    synchronizes the stream.

    This private context moves the checks to the enclosing block boundary.
    It also snapshots every quantizer object that can affect the arithmetic.
    The hot-path method checks object identity and PyTorch Tensor version
    counters before each use, detecting ordinary buffer replacement and
    tracked in-place mutation.

    This is not a concurrency or raw-storage safety boundary.  PyTorch's
    explicitly unsafe ``.data`` API, a NumPy alias, or a custom raw-storage
    writer can mutate bytes without advancing ``_version``; callers must not
    use those escape hatches or concurrently mutate the quantizer while a
    context is live.  REAL-Q gives each quantization loop exclusive ownership.
    """

    __slots__ = (
        "owner",
        "source_scale",
        "source_scale_version",
        "source_maxq",
        "source_maxq_version",
        "prepared_scale",
        "prepared_scale_version",
        "bits",
        "weight_groupsize",
        "input_rows",
        "column_count",
        "device",
        "dtype",
        "grouped",
    )

    @staticmethod
    def _tracked_version(tensor, label):
        try:
            return tensor._version
        except RuntimeError as exc:
            raise RuntimeError(
                "The prevalidated weight fast path requires version-tracked "
                f"{label} tensors and cannot run under torch.inference_mode(); "
                "use the REAL-Q runner's torch.no_grad() path."
            ) from exc

    def __init__(
        self,
        *,
        owner,
        source_scale,
        source_maxq,
        prepared_scale,
        input_rows,
        column_count,
        device,
        dtype,
        grouped,
    ) -> None:
        self.owner = owner
        self.source_scale = source_scale
        self.source_scale_version = self._tracked_version(
            source_scale, "source scale"
        )
        self.source_maxq = source_maxq
        self.source_maxq_version = self._tracked_version(
            source_maxq, "source maxq"
        )
        self.prepared_scale = prepared_scale
        self.prepared_scale_version = self._tracked_version(
            prepared_scale, "prepared scale"
        )
        self.bits = owner.bits
        self.weight_groupsize = owner.weight_groupsize
        self.input_rows = input_rows
        self.column_count = column_count
        self.device = device
        self.dtype = dtype
        self.grouped = grouped


class WeightQuantizer(torch.nn.Module):
    """From GPTQ Repo"""

    def __init__(self, shape: int = 1) -> None:
        super(WeightQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits,
        perchannel: bool = False,
        sym: bool = True,
        mse: bool = False,
        norm: float = 2.4,
        grid: int = 50,
        maxshrink: float = 0.5,
        weight_groupsize: int = -1,
        w_clip_search_impl: str = "cartesian_legacy",
        w_clip_update_impl: str = "guarded",
    ) -> None:
        if w_clip_search_impl not in _W_CLIP_SEARCH_IMPLEMENTATIONS:
            raise ValueError(
                "w_clip_search_impl must be 'cartesian_legacy' or "
                f"'symmetric_union_exact'; got {w_clip_search_impl!r}."
            )
        if w_clip_update_impl not in _W_CLIP_UPDATE_IMPLEMENTATIONS:
            raise ValueError(
                "w_clip_update_impl must be 'guarded' or 'where_out'; "
                f"got {w_clip_update_impl!r}."
            )
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.weight_groupsize = weight_groupsize
        self.w_clip_search_impl = w_clip_search_impl
        self.w_clip_update_impl = w_clip_update_impl
        if sym:
            self.maxq = torch.tensor(2 ** (bits - 1) - 1)
        else:
            self.maxq = torch.tensor(2**bits - 1)

    def find_params_weight_groupwise(self, x) -> None:
        if x.dim() != 2:
            raise ValueError(
                "weight group quantization expects a 2-D (rows, columns) tensor; "
                f"got shape {tuple(x.shape)}."
            )
        if self.weight_groupsize <= 0:
            raise ValueError(
                "find_params_weight_groupwise requires weight_groupsize > 0; "
                f"got {self.weight_groupsize}."
            )

        rows, columns = x.shape
        use_symmetric_union = self._can_use_symmetric_union(x)
        use_where_out_update = self._can_use_where_out_clip_update(x)

        def _params_for_equal_width_groups(grouped_x):
            """Return scale/zero expanded to ``grouped_x``'s last dimension.

            ``grouped_x`` has shape ``(rows, num_groups, group_width)``.  Keeping
            the MSE search vectorised over all equally-sized groups preserves the
            old implementation's arithmetic while allowing a final short group
            to be handled separately below.
            """
            xmax = torch.amax(grouped_x, dim=-1, keepdim=True)
            xmin = torch.amin(grouped_x, dim=-1, keepdim=True)

            if self.sym:
                scale = (
                    torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
                    / self.maxq
                )
                zero = torch.zeros_like(scale)
            else:
                all_zero = (xmin == 0) & (xmax == 0)
                xmin = xmin.clone()
                xmax = xmax.clone()
                xmin[all_zero] = -1
                xmax[all_zero] = +1
                scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
                zero = torch.round(-xmin / scale)

            if self.mse and use_symmetric_union:
                scale = _select_symmetric_union_scale(
                    grouped_x,
                    xmin.squeeze(-1),
                    xmax.squeeze(-1),
                    maxq=self.maxq,
                    norm=self.norm,
                    grid=self.grid,
                    candidate_count=int(self.maxshrink * self.grid),
                    grouped_error_order=True,
                ).unsqueeze(-1)
                zero = torch.zeros_like(scale)
            elif self.mse:
                best = torch.full(
                    grouped_x.shape[:2],
                    float("inf"),
                    device=grouped_x.device,
                    dtype=grouped_x.dtype,
                )
                for i in range(int(self.maxshrink * self.grid)):
                    for j in range(int(self.maxshrink * self.grid)):
                        xmin1 = (1 - i / self.grid) * xmin
                        xmax1 = (1 - j / self.grid) * xmax

                        if self.sym:
                            scale1 = (
                                torch.maximum(torch.abs(xmin1), xmax1)
                                .clamp(min=1e-5)
                                / self.maxq
                            )
                            zero1 = torch.zeros_like(scale1)
                            q = sym_quant_dequant(grouped_x, scale1, self.maxq)
                        else:
                            scale1 = (xmax1 - xmin1).clamp(min=1e-5) / self.maxq
                            zero1 = torch.round(-xmin1 / scale1)
                            q = asym_quant_dequant(
                                grouped_x, scale1, zero1, self.maxq
                            )

                        err = (q - grouped_x).abs().pow(self.norm).sum(-1)
                        improved = err < best
                        if use_where_out_update:
                            torch.where(improved, err, best, out=best)
                            value_mask = improved.unsqueeze(-1)
                            torch.where(
                                value_mask, scale1, scale, out=scale
                            )
                            torch.where(
                                value_mask, zero1, zero, out=zero
                            )
                        elif torch.any(improved):
                            best[improved] = err[improved]
                            scale[improved] = scale1[improved]
                            zero[improved] = zero1[improved]

            group_width = grouped_x.shape[-1]
            return (
                scale.expand(-1, -1, group_width).reshape(rows, -1),
                zero.expand(-1, -1, group_width).reshape(rows, -1),
            )

        # Vectorise all complete groups together.  The legacy implementation
        # builds one quantizer for ``min(start + groupsize, columns)`` and thus
        # permits a short final group; the former reshape-based implementation
        # accidentally rejected that valid case.
        full_columns = (columns // self.weight_groupsize) * self.weight_groupsize
        scale_parts = []
        zero_parts = []
        if full_columns:
            grouped = x[:, :full_columns].reshape(
                rows, full_columns // self.weight_groupsize, self.weight_groupsize
            )
            scale, zero = _params_for_equal_width_groups(grouped)
            scale_parts.append(scale)
            zero_parts.append(zero)
        if full_columns < columns:
            tail = x[:, full_columns:].unsqueeze(1)
            scale, zero = _params_for_equal_width_groups(tail)
            scale_parts.append(scale)
            zero_parts.append(zero)

        self.scale = torch.cat(scale_parts, dim=1)
        self.zero = torch.cat(zero_parts, dim=1)

    def _can_use_symmetric_union(self, x) -> bool:
        """Whether the exact union backend supports this observer input.

        The optimized backend is intentionally conservative.  Symmetry and
        finite floating-point weights are required by its endpoint proof.
        Standard GPTQ clipping also has a positive integer grid and
        ``0 < maxshrink <= 1``; other historical/custom settings remain owned
        by the byte-preserving Cartesian implementation.
        """

        if self.w_clip_search_impl != "symmetric_union_exact":
            return False
        # RealQLayer always observes ``linear.weight.clone().float()``.  Keep
        # the optimization on that production dtype: the historical ordinary
        # row path's ``best`` buffer is float32 and has different (including
        # erroring) behavior for float64/half inputs, which an opt-in backend
        # must not accidentally "fix".
        if not self.mse or not self.sym or x.dtype != torch.float32:
            return False
        if (
            not isinstance(self.grid, int)
            or isinstance(self.grid, bool)
            or self.grid <= 0
            or not isinstance(self.maxshrink, (int, float))
            or isinstance(self.maxshrink, bool)
            or not math.isfinite(float(self.maxshrink))
            or not 0.0 < float(self.maxshrink) <= 1.0
            or int(self.maxshrink * self.grid) <= 0
        ):
            return False
        # One synchronization per observer is required so non-finite data
        # follows the exact legacy fallback instead of entering an unproven
        # domain.
        return bool(torch.isfinite(x).all())

    def _can_use_where_out_clip_update(self, x) -> bool:
        """Whether fixed-shape winner updates preserve historical behavior.

        ``out=`` operators do not support autograd and the historical
        ordinary-row observer has dtype-specific behavior outside FP32.
        RealQLayer observes cloned FP32 weights under no-grad, so optimize
        exactly that production domain and leave all other cases on the
        original guarded/indexed implementation.
        """

        return (
            self.w_clip_update_impl == "where_out"
            and x.dtype == torch.float32
            and not x.requires_grad
        )

    def find_params(self, x) -> None:
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape

        if self.weight_groupsize > 0:
            # group-wise per-token quantization
            self.find_params_weight_groupwise(x)
            # utils.cleanup_memory(verbos=False)
            return
        elif self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        use_where_out_update = self._can_use_where_out_clip_update(x)
        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            self.scale = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5) / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse and self._can_use_symmetric_union(x):
            self.scale = _select_symmetric_union_scale(
                x,
                xmin,
                xmax,
                maxq=self.maxq,
                norm=self.norm,
                grid=self.grid,
                candidate_count=int(self.maxshrink * self.grid),
                grouped_error_order=False,
            )
            self.zero = torch.zeros_like(self.scale)
        elif self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                for j in range(int(self.maxshrink * self.grid)):
                    xmin1 = (1 - i / self.grid) * xmin
                    xmax1 = (1 - j / self.grid) * xmax

                    if self.sym:
                        scale1 = torch.maximum(torch.abs(xmin1), xmax1).clamp(min=1e-5) / self.maxq
                        zero1 = torch.zeros_like(scale1)
                        q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                    else:
                        scale1 = (xmax1 - xmin1) / self.maxq
                        zero1 = torch.round(-xmin1 / scale1)
                        q = asym_quant_dequant(
                            x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq
                        )

                    q -= x
                    q.abs_()
                    q.pow_(self.norm)
                    err = torch.sum(q, 1)
                    tmp = err < best
                    if use_where_out_update:
                        torch.where(tmp, err, best, out=best)
                        torch.where(
                            tmp, scale1, self.scale, out=self.scale
                        )
                        torch.where(
                            tmp, zero1, self.zero, out=self.zero
                        )
                    elif torch.any(tmp):
                        best[tmp] = err[tmp]
                        self.scale[tmp] = scale1[tmp]
                        self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return

    # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
    def quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if self.sym:
                return STEQuantize.apply(x, self.scale, self.maxq).to(x_dtype)
            return AsymSTEQuantize.apply(x, self.scale, self.zero, self.maxq).to(
                x_dtype
            )
        return x

    # Return int value and scale in addtional to fake quantized weight
    def fake_quantize(self, x, st_idx=None, end_idx=None, col_idx=None):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            scale = self.scale.to(x.device)
            if st_idx is not None and end_idx is not None:
                scale = scale[st_idx:end_idx]
            if self.weight_groupsize > 0 and col_idx is not None:
                # Group parameters are stored expanded in natural column order
                # so full-matrix RTN remains a simple broadcast. GPTQ quantizes
                # one (possibly act-order permuted) column at a time and must
                # explicitly select that column's original group.
                col_idx = torch.as_tensor(
                    col_idx, dtype=torch.long, device=scale.device
                ).reshape(-1)
                if col_idx.numel() != x.shape[-1]:
                    raise ValueError(
                        "col_idx must provide one natural column index per input "
                        f"column; got {col_idx.numel()} indices for x.shape={tuple(x.shape)}."
                    )
                if torch.any(col_idx < 0) or torch.any(col_idx >= scale.shape[-1]):
                    raise IndexError(
                        f"col_idx is outside [0, {scale.shape[-1]}) for grouped "
                        "weight quantization."
                    )
                scale = scale.index_select(-1, col_idx)
            elif self.weight_groupsize > 0 and x.shape[-1] != scale.shape[-1]:
                raise ValueError(
                    "Grouped fake_quantize on a partial weight tensor requires "
                    "col_idx in natural (pre-act-order) column coordinates; "
                    f"got x.shape={tuple(x.shape)} and scale.shape={tuple(scale.shape)}."
                )
            q = torch.clamp(torch.round(x / scale), -(self.maxq + 1), self.maxq)
            return (scale * q).to(x_dtype), q, scale
        else:
            return None, None, None

    def _prepare_fake_quantize_inner(
        self,
        *,
        input_rows,
        column_count,
        device,
        dtype,
        st_idx=None,
        end_idx=None,
        col_idx=None,
    ):
        """Prevalidate one REAL-Q block's private column-wise fast path.

        This is intentionally narrower than :meth:`fake_quantize`: it accepts
        only the ``(rows, 1)`` symmetric weight-column layout used by
        :class:`realq.quant.realq_layer.RealQLayer`.  Public callers must keep
        using :meth:`fake_quantize`.

        Grouped parameters are stored in natural (pre-act-order) column
        coordinates.  ``col_idx`` therefore names every natural column in the
        block, in the exact order in which the inner loop will consume them.
        The complete mapping is bounds-checked once here.  Selected scales are
        transposed into a contiguous ``(columns, rows, 1)`` layout so each hot
        iteration sees the same contiguous ``(rows, 1)`` operand as the public
        scalar ``index_select`` path.
        """

        if (
            not isinstance(input_rows, int)
            or isinstance(input_rows, bool)
            or input_rows <= 0
        ):
            raise ValueError(
                f"input_rows must be a positive integer; got {input_rows!r}."
            )
        if (
            not isinstance(column_count, int)
            or isinstance(column_count, bool)
            or column_count <= 0
        ):
            raise ValueError(
                "column_count must be a positive integer; "
                f"got {column_count!r}."
            )
        if (st_idx is None) != (end_idx is None):
            raise ValueError(
                "st_idx and end_idx must either both be provided or both be None."
            )
        if not hasattr(self, "bits") or self.bits >= 16:
            raise RuntimeError(
                "The prevalidated weight fast path requires an enabled "
                "quantizer with bits < 16."
            )
        if not self.ready():
            raise RuntimeError(
                "The prevalidated weight fast path requires ready scale "
                "parameters."
            )

        device = torch.device(device)
        source_scale = self.scale
        source_maxq = self.maxq
        scale = source_scale.to(device)
        if st_idx is not None and end_idx is not None:
            scale = scale[st_idx:end_idx]
        if scale.dim() != 2 or scale.shape[0] != input_rows:
            raise ValueError(
                "The prevalidated weight fast path requires a 2-D scale with "
                "one row per input row after slicing; got "
                f"scale.shape={tuple(scale.shape)}, input_rows={input_rows}."
            )
        if source_maxq.device != device:
            raise ValueError(
                "maxq must already be on the input device, matching the public "
                "fake_quantize arithmetic; got "
                f"maxq.device={source_maxq.device}, input device={device}."
            )

        grouped = self.weight_groupsize > 0
        if grouped:
            if col_idx is None:
                raise ValueError(
                    "Grouped prevalidated fake quantization requires one "
                    "natural col_idx per inner-loop column."
                )
            raw_col_idx = torch.as_tensor(col_idx, device=scale.device).reshape(-1)
            integer_dtypes = {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }
            if raw_col_idx.dtype not in integer_dtypes:
                raise TypeError(
                    "Grouped prevalidated col_idx must contain integer natural "
                    f"column coordinates; got dtype={raw_col_idx.dtype}."
                )
            natural_columns = raw_col_idx.to(dtype=torch.long)
            if natural_columns.numel() != column_count:
                raise ValueError(
                    "col_idx must provide one natural column index per inner "
                    f"iteration; got {natural_columns.numel()} indices for "
                    f"column_count={column_count}."
                )
            if torch.any(natural_columns < 0) or torch.any(
                natural_columns >= scale.shape[-1]
            ):
                raise IndexError(
                    f"col_idx is outside [0, {scale.shape[-1]}) for grouped "
                    "weight quantization."
                )
            selected = scale.index_select(-1, natural_columns)
            prepared_scale = selected.transpose(0, 1).contiguous().unsqueeze(-1)
        else:
            # The public column path broadcasts one per-row scale over a
            # (rows, 1) input.  Refuse broader broadcasting here: it is not a
            # RealQLayer hot-loop shape and could hide a stale/wrong observer.
            if scale.shape[-1] != 1:
                raise ValueError(
                    "Per-row prevalidated fake quantization requires "
                    f"scale.shape=(rows, 1); got {tuple(scale.shape)}."
                )
            prepared_scale = scale

        return _PrevalidatedWeightFakeQuant(
            owner=self,
            source_scale=source_scale,
            source_maxq=source_maxq,
            prepared_scale=prepared_scale,
            input_rows=input_rows,
            column_count=column_count,
            device=device,
            dtype=dtype,
            grouped=grouped,
        )

    def _fake_quantize_prevalidated(self, x, prepared, column_offset):
        """Fake-quantize one weight column using a validated block context.

        Arithmetic intentionally stays byte-for-byte in the public method's
        order: divide, round, clamp with the *tensor* ``maxq`` operand,
        multiply, then cast back to the input dtype.

        The context relies on tracked Tensor mutations and exclusive ownership;
        ``.data``/raw-storage writes and concurrent quantizer mutation are
        unsupported for the same reason documented on
        :class:`_PrevalidatedWeightFakeQuant`.
        """

        if not isinstance(prepared, _PrevalidatedWeightFakeQuant):
            raise TypeError(
                "prepared must come from _prepare_fake_quantize_inner()."
            )
        if prepared.owner is not self:
            raise RuntimeError(
                "The prevalidated fake-quant context belongs to another quantizer."
            )
        if (
            self.scale is not prepared.source_scale
            or self.scale._version != prepared.source_scale_version
            or self.maxq is not prepared.source_maxq
            or self.maxq._version != prepared.source_maxq_version
            or self.bits != prepared.bits
            or self.weight_groupsize != prepared.weight_groupsize
            or prepared.prepared_scale._version
            != prepared.prepared_scale_version
        ):
            raise RuntimeError(
                "Stale prevalidated fake-quant context: scale, maxq, bits, or "
                "weight_groupsize changed after block-boundary validation."
            )
        if (
            not isinstance(column_offset, int)
            or isinstance(column_offset, bool)
            or not 0 <= column_offset < prepared.column_count
        ):
            raise IndexError(
                "column_offset is outside the prevalidated block: "
                f"{column_offset!r} not in [0, {prepared.column_count})."
            )
        if (
            x.dim() != 2
            or tuple(x.shape) != (prepared.input_rows, 1)
            or x.device != prepared.device
            or x.dtype != prepared.dtype
        ):
            raise ValueError(
                "Input no longer matches the prevalidated REAL-Q column "
                "contract; expected "
                f"shape=({prepared.input_rows}, 1), device={prepared.device}, "
                f"dtype={prepared.dtype}, got shape={tuple(x.shape)}, "
                f"device={x.device}, dtype={x.dtype}."
            )

        scale = (
            prepared.prepared_scale[column_offset]
            if prepared.grouped
            else prepared.prepared_scale
        )
        x_dtype = x.dtype
        q = torch.clamp(
            torch.round(x / scale),
            -(self.maxq + 1),
            self.maxq,
        )
        return (scale * q).to(x_dtype), q, scale

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


def add_actquant(analyzer: model_utils.ModelAnalyzer) -> None:
    """
    Replaces specific quantizable layers in the model with ActQuantWrapper 
    based on the ModelAnalyzer's selection criteria.
    """
    for layer in analyzer.get_layers():
        quant_modules = analyzer.get_quantizable_modules(layer)
        
        for name, module in quant_modules.items():
            if isinstance(module, ActQuantWrapper):
                continue

            # Navigate to the parent module for nested names (e.g., 'self_attn.q_proj')
            parts = name.split('.')
            parent_module = layer
            
            for part in parts[:-1]:
                parent_module = getattr(parent_module, part)
            
            # Replace the target module
            target_name = parts[-1]
            setattr(parent_module, target_name, ActQuantWrapper(module))


def find_qlayers(
    module,
    layers=[ActQuantWrapper],
    name: str = "",
):
    # fix for llama embedding layer
    if type(module) in [torch.nn.Embedding] and type(module) in layers:
        return {"embed_tokens": module}
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_qlayers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def disable_act_quant(module):
    bits_config = {}
    for name, m in module.named_modules():
        if isinstance(m, ActQuantWrapper):
            bits_config[name] = m.quantizer.bits
            m.quantizer.bits = 16
            if m.out_quantizer.bits != 16:
                bits_config[name+'out'] = m.out_quantizer.bits
                m.out_quantizer.bits = 16

    return bits_config


def enable_act_quant(module, bits_config):
    for name, m in module.named_modules():
        if isinstance(m, ActQuantWrapper):
            m.quantizer.bits = bits_config[name]
            if name+'out' in bits_config.keys():
                m.out_quantizer.bits = bits_config[name+'out']
