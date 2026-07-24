"""Emit deterministic CPU output hashes for P01's parent/candidate comparison."""

from __future__ import annotations

import hashlib
import inspect
import json
import os

import torch
import torch.nn as nn

from realq.quant.realq_layer import RealQLayer
from utils.quant_utils import WeightQuantizer


def run_case(case_index: int, case: tuple) -> str:
    (
        groupsize,
        num_groups,
        group_parallel_quant,
        act_order,
        columns,
        blocksize,
        w_clip,
    ) = case
    generator = torch.Generator().manual_seed(7000 + case_index)
    weight = torch.randn(4, columns, generator=generator)
    weight[:, ::2].mul_(0.125)
    weight[:, 1::3].mul_(4.0)
    hessians = []
    for group in range(num_groups):
        factor = torch.randn(columns, columns, generator=generator)
        hessian = factor.matmul(factor.T)
        hessian.add_(torch.eye(columns) * (0.25 + group))
        hessians.append(hessian)

    linear = nn.Linear(columns, 4, bias=False)
    linear.weight.data.copy_(weight)
    quantizer = WeightQuantizer()
    quantizer.configure(
        bits=4,
        perchannel=True,
        sym=True,
        mse=w_clip,
        grid=4,
        maxshrink=0.5,
        weight_groupsize=groupsize,
    )
    realq = RealQLayer(
        linear=linear,
        saliency=torch.ones(1, 1, num_groups),
        quantizer=quantizer,
        num_groups=num_groups,
        dev=torch.device("cpu"),
        group_parallel_quant=group_parallel_quant,
    )
    realq.H = torch.stack(hessians)
    realq.act_square = torch.rand(columns, generator=generator)
    realq._finalized = True
    kwargs = {
        "blocksize": blocksize,
        "percdamp": 0.01,
        "act_order": act_order,
        "w_clip": w_clip,
        "group_parallel_quant": group_parallel_quant,
    }
    if "quantizer_inner_fastpath" in inspect.signature(realq.quantize).parameters:
        kwargs["quantizer_inner_fastpath"] = False
    realq.quantize(**kwargs)
    payload = linear.weight.detach().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(1)
    cases = [
        (-1, 1, "none", False, 9, 4, False),
        (-1, 1, "rank", True, 9, 4, True),
        (-1, 2, "none", True, 7, 3, False),
        (-1, 2, "rank", False, 7, 3, True),
        (4, 1, "rank", False, 10, 4, False),
        (4, 2, "rank", False, 10, 4, True),
        (3, 1, "rank", True, 10, 4, False),
        (3, 2, "rank", True, 10, 4, True),
        (3, 2, "none", True, 10, 4, False),
    ]
    hashes = [run_case(index, case) for index, case in enumerate(cases)]
    aggregate = hashlib.sha256("".join(hashes).encode()).hexdigest()
    print(json.dumps({"case_hashes": hashes, "aggregate": aggregate}, sort_keys=True))


if __name__ == "__main__":
    main()
