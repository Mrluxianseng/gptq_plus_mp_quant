"""Two-rank Gloo exactness probe for P01 (never touches CUDA)."""

from __future__ import annotations

import hashlib
import json
import os

import torch
import torch.distributed as dist
import torch.nn as nn

from realq.quant.realq_layer import RealQLayer
from utils.quant_utils import WeightQuantizer


def run_case(case_index: int, case: tuple, fast: bool) -> torch.Tensor:
    groupsize, num_groups, act_order, columns, blocksize = case
    generator = torch.Generator().manual_seed(9000 + case_index)
    weight = torch.randn(4, columns, generator=generator)
    weight[:, ::2].mul_(0.125)
    weight[:, 1::3].mul_(4.0)
    full_hessians = []
    for group in range(num_groups):
        factor = torch.randn(columns, columns, generator=generator)
        hessian = factor.matmul(factor.T)
        hessian.add_(torch.eye(columns) * (0.25 + group))
        full_hessians.append(hessian)
    full_hessians = torch.stack(full_hessians)

    linear = nn.Linear(columns, 4, bias=False)
    linear.weight.data.copy_(weight)
    quantizer = WeightQuantizer()
    quantizer.configure(
        bits=4,
        perchannel=True,
        sym=True,
        mse=False,
        weight_groupsize=groupsize,
    )
    realq = RealQLayer(
        linear=linear,
        saliency=torch.ones(1, 1, num_groups),
        quantizer=quantizer,
        num_groups=num_groups,
        dev=torch.device("cpu"),
        group_parallel_quant="rank",
    )
    realq.H = full_hessians.index_select(0, realq.hessian_group_ids)
    realq.act_square = torch.rand(columns, generator=generator)
    realq._finalized = True
    realq.quantize(
        blocksize=blocksize,
        percdamp=0.01,
        act_order=act_order,
        w_clip=False,
        group_parallel_quant="rank",
        quantizer_inner_fastpath=fast,
    )
    return linear.weight.detach().clone()


def main() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    cases = [
        (-1, 1, False, 9, 4),
        (3, 1, True, 10, 4),
        (4, 2, False, 10, 4),
        (3, 2, True, 10, 4),
    ]
    hashes = []
    for index, case in enumerate(cases):
        baseline = run_case(index, case, False)
        candidate = run_case(index, case, True)
        assert torch.equal(
            baseline.contiguous().view(torch.uint8),
            candidate.contiguous().view(torch.uint8),
        )
        hashes.append(
            hashlib.sha256(candidate.contiguous().numpy().tobytes()).hexdigest()
        )
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, hashes)
    assert all(item == hashes for item in gathered)
    if dist.get_rank() == 0:
        print(json.dumps({"world_size": dist.get_world_size(), "hashes": hashes}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
