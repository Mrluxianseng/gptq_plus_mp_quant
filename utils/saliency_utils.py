"""Shared Hessian-side saliency clipping semantics."""

from __future__ import annotations

import torch
import torch.distributed as dist

from utils import dist_utils


def grouped_gradient_norm_squared(
    gradient: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """Return the paper's per-token, per-row-group squared gradient norm.

    The output-channel dimension is partitioned into equal contiguous groups.
    For each token and group, saliency is ``sum_i gradient_i**2``—the squared
    Euclidean norm in ``main.tex``—not its channel mean.  Keeping this primitive
    shared prevents the legacy and refactored collectors from drifting.
    """
    if num_groups <= 0:
        raise ValueError(f"num_groups must be positive, got {num_groups}")
    hidden_size = gradient.shape[-1]
    if hidden_size % num_groups:
        raise ValueError(
            f"gradient width {hidden_size} is not divisible by "
            f"num_groups {num_groups}"
        )
    group_size = hidden_size // num_groups
    return (
        gradient.float()
        .reshape(*gradient.shape[:-1], num_groups, group_size)
        .square()
        .sum(dim=-1)
    )


def grouped_channel_gram(
    matrix: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """Return one unnormalised channel Gram matrix per contiguous row group.

    ``matrix`` has shape ``(output_channels, rank)``.  This is the low-rank
    counterpart of :func:`grouped_gradient_norm_squared`: both use a channel
    sum, never a channel mean.
    """
    if matrix.dim() != 2:
        raise ValueError(
            f"matrix must have shape (channels, rank), got {tuple(matrix.shape)}"
        )
    if num_groups <= 0:
        raise ValueError(f"num_groups must be positive, got {num_groups}")
    channels, rank = matrix.shape
    if channels % num_groups:
        raise ValueError(
            f"matrix channels {channels} are not divisible by "
            f"num_groups {num_groups}"
        )
    grouped = matrix.float().reshape(num_groups, channels // num_groups, rank)
    return torch.bmm(grouped.transpose(1, 2), grouped)


def _linear_quantile(values: torch.Tensor, percentile: float) -> torch.Tensor:
    """``torch.quantile(..., interpolation='linear')`` without its size cap."""
    flat = values.reshape(-1)
    n = flat.numel()
    if n == 0:
        raise ValueError("cannot take a percentile of an empty tensor")
    # Some torch/CUDA versions reject >2^24 inputs in torch.quantile.
    if n <= 16 * 1024 * 1024:
        try:
            return torch.quantile(flat, percentile)
        except RuntimeError:
            pass
    rank = float(percentile) * float(n - 1)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, n - 1)
    lower = flat.kthvalue(lower_index + 1).values
    if upper_index == lower_index:
        return lower
    upper = flat.kthvalue(upper_index + 1).values
    fraction = rank - float(lower_index)
    return lower + (upper - lower) * fraction


def global_percentile(
    values: torch.Tensor,
    percentile: float,
) -> torch.Tensor:
    """Exact percentile over the concatenation of every rank's values."""
    if not (0.0 < float(percentile) < 1.0):
        raise ValueError(
            f"percentile must be in (0, 1), got {percentile}"
        )

    world = dist_utils.get_world_size()
    distributed = world > 1
    if distributed and not dist_utils.is_dist_available_and_initialized():
        raise RuntimeError(
            "world_size > 1 requires an initialized process group for global "
            "saliency percentile clipping"
        )

    backend = dist.get_backend() if distributed else None
    if backend == "nccl":
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    elif values.is_cuda:
        device = values.device
    else:
        device = torch.device("cpu")

    local = values.detach().reshape(-1).to(
        device=device, dtype=torch.float32
    )
    if not distributed:
        if local.numel() == 0:
            raise ValueError("cannot take a percentile of an empty tensor")
        cap = _linear_quantile(local, float(percentile))
    else:
        local_count = torch.tensor(
            [local.numel()], device=device, dtype=torch.int64
        )
        gathered_counts = [
            torch.empty_like(local_count) for _ in range(world)
        ]
        dist.all_gather(gathered_counts, local_count)
        counts = [int(value.item()) for value in gathered_counts]
        if sum(counts) == 0:
            raise ValueError(
                "cannot take a distributed percentile of globally empty tensors"
            )
        max_count = max(counts)
        if local.numel() < max_count:
            padded = torch.empty(
                max_count, device=device, dtype=torch.float32
            )
            padded[: local.numel()].copy_(local)
            padded[local.numel() :].zero_()
        else:
            padded = local
        gathered = [
            torch.empty(max_count, device=device, dtype=torch.float32)
            for _ in range(world)
        ]
        dist.all_gather(gathered, padded)
        cap = torch.zeros((), device=device, dtype=torch.float32)
        if dist_utils.is_main():
            global_values = torch.cat(
                [value[:count] for value, count in zip(gathered, counts)]
            )
            cap.copy_(
                _linear_quantile(global_values, float(percentile))
            )
            del global_values
        dist.broadcast(cap, src=0)
    return cap


def clip_global_percentile_(
    saliency: torch.Tensor,
    percentile: float | None,
) -> torch.Tensor:
    """Clip one module's complete calibration saliency at its global quantile.

    ``saliency`` is the rank-local ``(N_local, T, G)`` shard.  In distributed
    runs all rank shards are gathered only for this one module, rank 0 computes
    the exact quantile over the complete calibration set, and the scalar cap is
    broadcast.  The returned tensor remains rank-local and is modified in
    place.

    This intentionally happens after all precompute mini-batches have been
    concatenated.  Computing one quantile per mini-batch makes the Hessian
    depend on ``global_loss_bsz`` and does not implement the paper's stated
    "99th percentile" over calibration saliency values.
    """
    if percentile is None or not (0.0 < float(percentile) < 1.0):
        return saliency
    if saliency.numel() == 0:
        raise ValueError("cannot percentile-clip an empty saliency tensor")

    cap = global_percentile(saliency, float(percentile))
    saliency.clamp_(max=float(cap.item()))
    return saliency
