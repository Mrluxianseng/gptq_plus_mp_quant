"""Numerical acceptance probe for REAL-Q full-block sliding Block-GD.

Run the same deterministic refresh once with one rank and once with eight
ranks, then compare the saved rank-zero states::

    PYTHONPATH=. python -m torch.distributed.run --standalone --nproc_per_node=1 \
        tools/validate_realq_full_block_8gpu.py --output /tmp/realq-1gpu.pt
    PYTHONPATH=. python -m torch.distributed.run --standalone --nproc_per_node=8 \
        tools/validate_realq_full_block_8gpu.py --output /tmp/realq-8gpu.pt
    PYTHONPATH=. python tools/validate_realq_full_block_8gpu.py \
        --compare /tmp/realq-1gpu.pt /tmp/realq-8gpu.pt

The probe exercises the merged behavior directly: the current linear keeps
its quantized prefix locked, later linears in the current Transformer block
receive Adam updates, and every linear in the next sliding-window block is
updated through the same backward.  It also proves that the global-sample
data-parallel result agrees with the one-rank reference.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn

from realq.config import Config
from realq.refresh.block_gd import (
    BlockRefreshState,
    RefreshContext,
    _SharedSampleScheduler,
    make_grad_refresh_fn,
)


_TOTAL_SAMPLES = 8
_HIDDEN_SIZE = 8
_SEQUENCE_LENGTH = 3
_PERMUTATION = (2, 0, 4, 1, 7, 5, 6, 3)
_TRAILING_START = 3


class _ToyTransformerBlock(nn.Module):
    """Small differentiable block with two independently active linears."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        self.first = nn.Linear(_HIDDEN_SIZE, _HIDDEN_SIZE, bias=False)
        self.second = nn.Linear(_HIDDEN_SIZE, _HIDDEN_SIZE, bias=False)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            self.first.weight.copy_(
                torch.randn(
                    self.first.weight.shape,
                    generator=generator,
                    dtype=torch.float32,
                )
                * 0.2
            )
            self.second.weight.copy_(
                torch.randn(
                    self.second.weight.shape,
                    generator=generator,
                    dtype=torch.float32,
                )
                * 0.2
            )

    def forward(self, hidden_states: torch.Tensor, **_kwargs):
        hidden_states = torch.tanh(self.first(hidden_states))
        return (self.second(hidden_states),)


def _named_linears(
    block: _ToyTransformerBlock,
) -> list[tuple[str, nn.Module]]:
    return [("first", block.first), ("second", block.second)]


def _assert_optimized_defaults() -> None:
    cfg = Config()
    expected = {
        "quantizer_inner_fastpath": True,
        "w_clip_search_impl": "symmetric_union_exact",
        "fisher_fp32_cache": True,
        "act_order_stitch_impl": "prefix_q_trailing_w_exact",
        "w_clip_update_impl": "where_out",
        "w_group_param_layout": "compact",
        "prepared_clamp_bound_cache": True,
        "triton_column_block": True,
        "loss_slide_window": True,
        "moe_gpu_resident": True,
        "moe_joint_column_block": True,
        "moe_expert_loss_slide_window": True,
    }
    actual = {name: getattr(cfg, name) for name in expected}
    if actual != expected:
        raise AssertionError(
            f"optimized default drift: expected={expected}, actual={actual}"
        )


def _rank_consistency(tensor: torch.Tensor, name: str) -> None:
    reference = tensor.detach().clone()
    dist.broadcast(reference, src=0)
    max_abs = (tensor - reference).abs().max()
    if max_abs.item() != 0.0:
        raise AssertionError(
            f"rank divergence for {name}: max_abs={max_abs.item():.9e}"
        )


def _changed(before: torch.Tensor, after: torch.Tensor, name: str) -> None:
    delta = after - before
    if not torch.isfinite(delta).all():
        raise AssertionError(f"non-finite state delta for {name}")
    if torch.count_nonzero(delta).item() == 0:
        raise AssertionError(f"expected a non-zero update for {name}")


def _run(output: Path) -> None:
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("run mode must be launched by torch.distributed.run")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world not in (1, 8):
        raise RuntimeError(f"expected world size 1 or 8, got {world}")

    try:
        _assert_optimized_defaults()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        current = _ToyTransformerBlock(seed=101).to(device).eval()
        following = _ToyTransformerBlock(seed=211).to(device).eval()
        current.requires_grad_(False)
        following.requires_grad_(False)

        generator = torch.Generator(device="cpu").manual_seed(307)
        global_inputs = torch.randn(
            _TOTAL_SAMPLES,
            _SEQUENCE_LENGTH,
            _HIDDEN_SIZE,
            generator=generator,
            dtype=torch.float32,
        ).to(device)
        with torch.no_grad():
            global_current_targets = current(global_inputs)[0].clone()
            global_next_targets = following(global_current_targets)[0].clone()

        if _TOTAL_SAMPLES % world != 0:
            raise AssertionError("sample count must divide the world size")
        local_count = _TOTAL_SAMPLES // world
        shard = slice(rank * local_count, (rank + 1) * local_count)
        inputs = global_inputs[shard].clone()
        current_targets = global_current_targets[shard].clone()
        next_targets = global_next_targets[shard].clone()

        current_teacher_before = {
            name: module.weight.detach().clone()
            for name, module in _named_linears(current)
        }
        next_teacher_before = {
            name: module.weight.detach().clone()
            for name, module in _named_linears(following)
        }
        current_state = BlockRefreshState(current, _named_linears(current))
        next_state = BlockRefreshState(following, _named_linears(following))
        current_master = current_state.begin_quantization("first")
        future_before = current_state._states["second"].master.clone()
        next_before = {
            name: state.master.clone()
            for name, state in next_state._states.items()
        }

        layer_state = SimpleNamespace(
            inps=inputs,
            fp_inps=inputs.clone(),
            attention_mask=None,
            position_ids=None,
            position_embeddings=None,
        )
        context = RefreshContext(
            module=current.first,
            layer_lr=1.0e-2,
            grad_clip=-1.0,
            backward_bsz=1,
            scheduler=_SharedSampleScheduler(
                n_total=_TOTAL_SAMPLES,
                chunk_size=_TOTAL_SAMPLES,
                seed=0,
            ),
        )
        refresh = make_grad_refresh_fn(
            layer=current,
            module=current.first,
            module_name="first",
            block_state=current_state,
            layer_state=layer_state,
            fp_out_for_this_layer=current_targets,
            fisher=torch.eye(_HIDDEN_SIZE, device=device),
            ctx=context,
            next_layer=following,
            next_block_state=next_state,
            next_fp_out=next_targets,
            next_fisher=torch.eye(_HIDDEN_SIZE, device=device),
            slide_alpha_fn=lambda: 0.4,
        )
        stitched = current_master.clone()
        stitched.add_(0.125)
        permutation = torch.tensor(
            _PERMUTATION, dtype=torch.long, device=device
        )
        update = refresh(
            stitched,
            trailing_col_start=_TRAILING_START,
            perm=permutation,
        )

        locked = permutation[:_TRAILING_START]
        active = permutation[_TRAILING_START:]
        if torch.count_nonzero(update.index_select(1, locked)).item() != 0:
            raise AssertionError("the already-quantized natural columns moved")
        if torch.count_nonzero(update.index_select(1, active)).item() == 0:
            raise AssertionError("the active current-linear columns did not move")
        if not torch.isfinite(update).all():
            raise AssertionError("current-linear update contains non-finite values")

        current_future = current_state._states["second"].master
        _changed(future_before, current_future, "current.second")
        for name, before in next_before.items():
            _changed(before, next_state._states[name].master, f"next.{name}")
        if current_state._states["first"].step != 1:
            raise AssertionError("current.first Adam step did not advance")
        if current_state._states["second"].step != 1:
            raise AssertionError("current.second Adam step did not advance")
        if any(state.step != 1 for state in next_state._states.values()):
            raise AssertionError("a next-block Adam step did not advance")

        for name, module in _named_linears(current):
            if not torch.equal(module.weight, current_teacher_before[name]):
                raise AssertionError(f"teacher storage was mutated: current.{name}")
        for name, module in _named_linears(following):
            if not torch.equal(module.weight, next_teacher_before[name]):
                raise AssertionError(f"teacher storage was mutated: next.{name}")

        results = {
            "current_update": update.detach(),
            "current_future_master": current_future.detach(),
            "next_first_master": next_state._states["first"].master.detach(),
            "next_second_master": next_state._states["second"].master.detach(),
        }
        for name, tensor in results.items():
            _rank_consistency(tensor, name)
        dist.barrier()
        if rank == 0:
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({name: tensor.cpu() for name, tensor in results.items()}, output)
            summary = {
                "status": "passed",
                "world_size": world,
                "output": str(output),
                "current_active_update_l2": float(
                    update.index_select(1, active).norm().item()
                ),
                "current_future_delta_l2": float(
                    (current_future - future_before).norm().item()
                ),
                "next_first_delta_l2": float(
                    (
                        next_state._states["first"].master
                        - next_before["first"]
                    ).norm().item()
                ),
                "next_second_delta_l2": float(
                    (
                        next_state._states["second"].master
                        - next_before["second"]
                    ).norm().item()
                ),
            }
            print(json.dumps(summary, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _compare(reference_path: Path, distributed_path: Path) -> None:
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    distributed = torch.load(
        distributed_path, map_location="cpu", weights_only=True
    )
    if set(reference) != set(distributed):
        raise AssertionError(
            f"result keys differ: {set(reference)} != {set(distributed)}"
        )
    metrics: dict[str, dict[str, float]] = {}
    for name in sorted(reference):
        left = reference[name]
        right = distributed[name]
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise AssertionError(f"non-finite comparison tensor: {name}")
        delta = (left - right).abs()
        max_abs = float(delta.max().item())
        mean_abs = float(delta.mean().item())
        scale = max(float(left.abs().max().item()), 1.0)
        tolerance = 3.0e-5 * scale
        if max_abs > tolerance:
            raise AssertionError(
                f"1-rank/8-rank drift for {name}: max_abs={max_abs:.9e}, "
                f"tolerance={tolerance:.9e}"
            )
        metrics[name] = {
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "tolerance": tolerance,
        }
    print(
        json.dumps(
            {
                "status": "passed",
                "comparison": "1-rank-vs-8-rank",
                "metrics": metrics,
            },
            sort_keys=True,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("ONE_RANK", "EIGHT_RANK"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output is not None:
        _run(args.output)
    else:
        _compare(*args.compare)


if __name__ == "__main__":
    main()
