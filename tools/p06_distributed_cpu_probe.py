"""Distributed CPU/CUDA exactness probe for P06 act-order weight stitching.

CPU/Gloo is the default and requires CUDA to be hidden:

    CUDA_VISIBLE_DEVICES='' torchrun --standalone --nproc_per_node=2 \
        tools/p06_distributed_cpu_probe.py
    CUDA_VISIBLE_DEVICES='' torchrun --standalone --nproc_per_node=4 \
        tools/p06_distributed_cpu_probe.py

CUDA/NCCL is opt-in. The caller must set ``CUBLAS_WORKSPACE_CONFIG`` before
Python imports torch:

    CUBLAS_WORKSPACE_CONFIG=:4096:8 REALQ_P06_PROBE_DEVICE=cuda \
      torchrun --standalone --nproc_per_node=4 \
        tools/p06_distributed_cpu_probe.py

The probe exercises real Fisher-MSE backward, gradient all-reduce, and Adam
updates. Each case runs both legacy->exact and exact->legacy to detect
allocator/order-sensitive drift.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn

from realq.quant.realq_layer import RealQLayer
from realq.refresh.block_gd import (
    RefreshContext,
    _SharedSampleScheduler,
    make_grad_refresh_fn,
)
from utils.quant_utils import WeightQuantizer


LEGACY = "full_weight_legacy"
EXACT = "prefix_q_trailing_w_exact"


def _raw_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return torch.equal(
        left.detach().contiguous().view(torch.uint8),
        right.detach().contiguous().view(torch.uint8),
    )


def _sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


class _ToyLayer(nn.Module):
    def __init__(self, columns: int, rows: int) -> None:
        super().__init__()
        self.proj = nn.Linear(columns, rows, bias=False)

    def forward(self, hidden_states, **_kwargs):
        return (self.proj(hidden_states),)


@dataclass(frozen=True)
class _Case:
    rows: int
    columns: int
    blocksize: int
    num_groups: int
    weight_groupsize: int
    seed: int


@dataclass
class _Result:
    weight: torch.Tensor
    stitched_inputs: list[torch.Tensor]
    updates: list[torch.Tensor]
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor
    adam_step: int
    collective_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]]


def _ready_quantizer(
    weight: torch.Tensor,
    weight_groupsize: int,
) -> WeightQuantizer:
    quantizer = WeightQuantizer()
    quantizer.configure(
        bits=4,
        perchannel=True,
        sym=True,
        mse=False,
        weight_groupsize=weight_groupsize,
    )
    quantizer.to(weight.device)
    # Isolate refresh communication from qparam scale/zero communication.
    quantizer.find_params(weight)
    return quantizer


def _run_case(
    case: _Case,
    implementation: str,
    device: torch.device,
) -> _Result:
    rank = dist.get_rank()
    world = dist.get_world_size()
    generator = torch.Generator().manual_seed(case.seed)
    weight = torch.randn(
        case.rows, case.columns, generator=generator
    ).to(device)
    weight[:, ::2].mul_(0.25)
    weight[:, 1::3].mul_(3.0)

    full_hessians = []
    for group in range(case.num_groups):
        factor = torch.randn(
            case.columns, case.columns, generator=generator
        )
        full_hessians.append(
            factor.matmul(factor.T).add_(
                torch.eye(case.columns) * (0.5 + group)
            )
        )
    full_hessians = torch.stack(full_hessians).to(device)
    act_square = torch.rand(case.columns, generator=generator).to(device)

    samples_per_rank = 2
    global_sample_count = world * samples_per_rank
    global_inputs = torch.randn(
        global_sample_count, 3, case.columns, generator=generator
    ).to(device)
    global_targets = torch.randn(
        global_sample_count, 3, case.rows, generator=generator
    ).to(device)
    local_slice = slice(
        rank * samples_per_rank,
        (rank + 1) * samples_per_rank,
    )

    layer = _ToyLayer(case.columns, case.rows).to(device)
    layer.proj.weight.data.copy_(weight)
    realq = RealQLayer(
        linear=layer.proj,
        # RealQLayer deliberately persists saliency on CPU.
        saliency=torch.ones(1, 1, case.num_groups),
        quantizer=_ready_quantizer(weight, case.weight_groupsize),
        num_groups=case.num_groups,
        dev=device,
        group_parallel_quant="rank",
    )
    realq.H = full_hessians.index_select(
        0, realq.hessian_group_ids.to(device)
    )
    realq.act_square = act_square
    realq._finalized = True

    layer_state = SimpleNamespace(
        inps=global_inputs[local_slice].clone(),
        attention_mask=None,
        position_ids=None,
        position_embeddings=None,
    )
    context = RefreshContext(
        module=layer.proj,
        layer_lr=3e-4,
        grad_clip=1.0,
        backward_bsz=1,
        scheduler=_SharedSampleScheduler(
            global_sample_count,
            global_sample_count,
            seed=77,
        ),
    )
    real_refresh = make_grad_refresh_fn(
        layer=layer,
        module=layer.proj,
        layer_state=layer_state,
        fp_out_for_this_layer=global_targets[local_slice].clone(),
        fisher=torch.eye(case.rows, device=device),
        ctx=context,
    )
    stitched_inputs: list[torch.Tensor] = []
    updates: list[torch.Tensor] = []

    def observed_refresh(stitched, trailing_col_start, perm=None):
        stitched_inputs.append(stitched.clone())
        update = real_refresh(
            stitched,
            trailing_col_start,
            perm=perm,
        )
        updates.append(update.clone())
        return update

    collective_shapes: list[
        tuple[tuple[int, ...], tuple[int, ...]]
    ] = []
    original_all_gather = dist.all_gather_into_tensor

    def observed_all_gather(output, local, *args, **kwargs):
        collective_shapes.append((tuple(output.shape), tuple(local.shape)))
        return original_all_gather(output, local, *args, **kwargs)

    dist.all_gather_into_tensor = observed_all_gather
    try:
        realq.quantize(
            blocksize=case.blocksize,
            percdamp=0.01,
            act_order=True,
            w_clip=False,
            grad_refresh_fn=observed_refresh,
            group_parallel_quant="rank",
            act_order_stitch_impl=implementation,
        )
    finally:
        dist.all_gather_into_tensor = original_all_gather

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return _Result(
        weight=layer.proj.weight.detach().clone(),
        stitched_inputs=stitched_inputs,
        updates=updates,
        exp_avg=context.exp_avg.clone(),
        exp_avg_sq=context.exp_avg_sq.clone(),
        adam_step=context.adam_step,
        collective_shapes=collective_shapes,
    )


def _expected_collectives(
    case: _Case,
    *,
    world: int,
    implementation: str,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    local_rows = case.rows // world
    expected = []
    for i1 in range(0, case.columns, case.blocksize):
        i2 = min(i1 + case.blocksize, case.columns)
        if i2 >= case.columns:
            break
        count = i2 - i1
        expected.extend(
            [
                ((case.rows, count), (local_rows, count)),
                (
                    (case.rows, case.columns - i2),
                    (local_rows, case.columns - i2),
                ),
            ]
        )
        if implementation == LEGACY:
            expected.append(
                ((case.rows, case.columns), (local_rows, case.columns))
            )
    # Final replicated Q is required under both implementations.
    expected.append(
        ((case.rows, case.columns), (local_rows, case.columns))
    )
    return expected


def _assert_results_equal(
    reference: _Result,
    candidate: _Result,
    case: _Case,
) -> None:
    assert _raw_equal(candidate.weight, reference.weight)
    assert candidate.adam_step == reference.adam_step
    assert len(candidate.stitched_inputs) == len(reference.stitched_inputs)
    assert len(candidate.updates) == len(reference.updates)
    assert all(
        _raw_equal(left, right)
        for left, right in zip(
            candidate.stitched_inputs,
            reference.stitched_inputs,
        )
    )
    assert all(
        _raw_equal(left, right)
        for left, right in zip(candidate.updates, reference.updates)
    )
    assert _raw_equal(candidate.exp_avg, reference.exp_avg)
    assert _raw_equal(candidate.exp_avg_sq, reference.exp_avg_sq)

    expected_steps = math.ceil(case.columns / case.blocksize) - 1
    assert candidate.adam_step == expected_steps


def _assert_rejection_boundaries(
    device: torch.device,
) -> dict[str, str]:
    world = dist.get_world_size()
    rank = dist.get_rank()
    errors: dict[str, str] = {}

    uneven_rows = world * 2 + 1
    columns = 5
    uneven_weight = torch.randn(
        uneven_rows,
        columns,
        generator=torch.Generator().manual_seed(901),
    ).to(device)
    uneven_layer = nn.Linear(
        columns, uneven_rows, bias=False
    ).to(device)
    uneven_layer.weight.data.copy_(uneven_weight)
    uneven_realq = RealQLayer(
        uneven_layer,
        torch.ones(1, 1, 1),
        _ready_quantizer(uneven_weight, -1),
        1,
        device,
        group_parallel_quant="rank",
    )
    uneven_realq.H = torch.eye(
        columns, device=device
    ).unsqueeze(0)
    uneven_realq.act_square = torch.arange(
        columns, dtype=torch.float32, device=device
    )
    uneven_realq._finalized = True
    try:
        uneven_realq.quantize(
            blocksize=2,
            act_order=True,
            grad_refresh_fn=lambda *args, **kwargs: None,
            group_parallel_quant="rank",
            act_order_stitch_impl=EXACT,
        )
    except ValueError as error:
        errors["uneven_rows"] = str(error)
    else:
        raise AssertionError("uneven output rows unexpectedly accepted")

    # Three groups are irregular against both tested world sizes (2 and 4).
    irregular_rows = 12
    irregular_weight = torch.randn(
        irregular_rows,
        columns,
        generator=torch.Generator().manual_seed(902),
    ).to(device)
    irregular_layer = nn.Linear(
        columns, irregular_rows, bias=False
    ).to(device)
    irregular_layer.weight.data.copy_(irregular_weight)
    try:
        RealQLayer(
            irregular_layer,
            torch.ones(1, 1, 3),
            _ready_quantizer(irregular_weight, -1),
            3,
            device,
            group_parallel_quant="rank",
        )
    except ValueError as error:
        errors["irregular_groups"] = str(error)
    else:
        raise AssertionError(
            f"irregular world/group relation unexpectedly accepted: "
            f"world={world}, groups=3, rank={rank}"
        )

    gathered = [None] * world
    dist.all_gather_object(gathered, errors)
    assert all(item == errors for item in gathered)
    return errors


def _resolve_probe_device() -> tuple[torch.device, str, int | None]:
    requested = os.environ.get("REALQ_P06_PROBE_DEVICE", "cpu").strip().lower()
    if requested not in ("cpu", "cuda"):
        raise ValueError(
            "REALQ_P06_PROBE_DEVICE must be 'cpu' or 'cuda', got "
            f"{requested!r}."
        )

    if requested == "cpu":
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", (
            "CPU P06 probe requires CUDA_VISIBLE_DEVICES='' and never uses a GPU"
        )
        return torch.device("cpu"), "gloo", None

    workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace_config not in (":4096:8", ":16:8"):
        raise RuntimeError(
            "CUDA P06 probe requires the caller to set "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) before Python starts; "
            f"got {workspace_config!r}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "REALQ_P06_PROBE_DEVICE=cuda requested but CUDA is unavailable."
        )
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
    except (KeyError, ValueError) as error:
        raise RuntimeError(
            "CUDA P06 probe requires integer LOCAL_RANK from torchrun."
        ) from error
    if not 0 <= local_rank < torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is outside the visible CUDA device "
            f"range [0, {torch.cuda.device_count()})."
        )
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return torch.device("cuda", local_rank), "nccl", local_rank


def _run_implementation_order(
    case: _Case,
    implementations: tuple[str, str],
    device: torch.device,
) -> dict[str, _Result]:
    results = {}
    for implementation in implementations:
        results[implementation] = _run_case(
            case,
            implementation,
            device,
        )
        # Keep every rank on the same implementation boundary. This is
        # particularly important when the candidate deliberately removes one
        # NCCL collective from each refresh.
        dist.barrier()
    return results


def main() -> None:
    device, backend, local_rank = _resolve_probe_device()
    torch.set_num_threads(1)
    dist.init_process_group(backend)
    world = dist.get_world_size()
    if world not in (2, 4):
        raise ValueError(f"P06 probe requires world size 2 or 4, got {world}.")

    cases = [
        _Case(8, 10, 4, 1, -1, 8100 + world),
        _Case(8, 9, 4, 2, 4, 8200 + world),
        _Case(8, 8, 4, 4, -1, 8300 + world),
    ]
    summaries = []
    for case in cases:
        legacy_then_exact = _run_implementation_order(
            case,
            (LEGACY, EXACT),
            device,
        )
        exact_then_legacy = _run_implementation_order(
            case,
            (EXACT, LEGACY),
            device,
        )

        # Equality must hold within either execution order.
        _assert_results_equal(
            legacy_then_exact[LEGACY],
            legacy_then_exact[EXACT],
            case,
        )
        _assert_results_equal(
            exact_then_legacy[LEGACY],
            exact_then_legacy[EXACT],
            case,
        )
        # Each implementation must also be stable across the two allocator /
        # collective histories.
        for implementation in (LEGACY, EXACT):
            _assert_results_equal(
                legacy_then_exact[implementation],
                exact_then_legacy[implementation],
                case,
            )
            expected_collectives = _expected_collectives(
                case,
                world=world,
                implementation=implementation,
            )
            assert (
                legacy_then_exact[implementation].collective_shapes
                == expected_collectives
            )
            assert (
                exact_then_legacy[implementation].collective_shapes
                == expected_collectives
            )

        exact = legacy_then_exact[EXACT]
        legacy = legacy_then_exact[LEGACY]
        digest = _sha256(exact.weight)
        rank_digests = [None] * world
        dist.all_gather_object(rank_digests, digest)
        assert all(item == digest for item in rank_digests)
        summaries.append(
            {
                "rows": case.rows,
                "columns": case.columns,
                "blocksize": case.blocksize,
                "num_groups": case.num_groups,
                "weight_groupsize": case.weight_groupsize,
                "refreshes": exact.adam_step,
                "legacy_collectives": len(legacy.collective_shapes),
                "exact_collectives": len(exact.collective_shapes),
                "execution_orders": [
                    "legacy_then_exact",
                    "exact_then_legacy",
                ],
                "sha256": digest,
            }
        )

    errors = _assert_rejection_boundaries(device)
    if dist.get_rank() == 0:
        print(
            json.dumps(
                {
                    "world_size": world,
                    "backend": dist.get_backend(),
                    "probe_device": os.environ.get(
                        "REALQ_P06_PROBE_DEVICE",
                        "cpu",
                    ),
                    "local_device": str(device),
                    "local_rank": local_rank,
                    "cuda_visible_devices": os.environ.get(
                        "CUDA_VISIBLE_DEVICES"
                    ),
                    "cublas_workspace_config": os.environ.get(
                        "CUBLAS_WORKSPACE_CONFIG"
                    ),
                    "cases": summaries,
                    "rejections": errors,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
