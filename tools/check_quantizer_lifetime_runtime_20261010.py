"""Exercise the actual GPTQPlus/WeightQuantizer API before model-size work."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from gptq_utils import gptq_plus_utils as impl
from utils.quant_utils import WeightQuantizer
from tools.prepare_quantizer_residency_pilot_20261010 import INJECTION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    original = impl.GPTQPlus.free
    argv = list(sys.argv)
    rows = []
    try:
        for cpu in (False, True):
            root = args.output / ("candidate" if cpu else "control")
            root.mkdir()
            torch.manual_seed(44)
            layer = torch.nn.Linear(32, 64, bias=False, device="cuda")
            worker = impl.GPTQPlus(layer, torch.ones(1, 2, 4, device="cuda"),
                                   torch.zeros(4, 32), 4, 0.0, 0.0)
            worker.quantizer = WeightQuantizer()
            worker.quantizer.configure(4, perchannel=True, sym=True)
            worker.quantizer.find_params(layer.weight.detach())
            worker.layer_idx, worker.layer_name = 0, "runtime_fixture"
            impl.GPTQPlus.free = original
            sys.argv = ["runtime_fixture"] + (["--research_stage2_quantizer_metadata_cpu"] if cpu else [])
            namespace = dict(impl=impl, torch=torch, sys=sys, json=json,
                             hashlib=hashlib, root=root, os=os)
            exec(compile(INJECTION, "quantizer_lifetime_injection", "exec"), namespace)
            worker.free()
            devices = {name: str(tensor.device) for name, tensor in worker.quantizer.state_dict().items()}
            row = json.loads((root / "quantizer_metadata_lifetime_rank0.json").read_text())[0]
            if worker.H is not None or (cpu and any(value != "cpu" for value in devices.values())):
                raise RuntimeError("free or metadata transfer did not execute")
            rows.append(dict(buffers=row["buffers"], devices=devices, offloaded=row["offloaded"]))
            del worker, layer
            torch.cuda.empty_cache()
        exact = rows[0]["buffers"] == rows[1]["buffers"]
        if not exact:
            raise RuntimeError("metadata changed in actual runtime API fixture")
        result = dict(valid=True, exact_metadata=True, rows=rows,
                      scope="Small actual CUDA API preflight; no model-scale memory or speed claim")
        (args.output / "status.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
    finally:
        impl.GPTQPlus.free = original
        sys.argv = argv


if __name__ == "__main__":
    main()
