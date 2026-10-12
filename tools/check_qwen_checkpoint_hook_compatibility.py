"""CPU-only actual-weight Qwen block check before Stage-0 recomputation.

This is not a GPU memory result, a FSDP compatibility test, or a real
calibration experiment. It isolates saved-tensor recomputation/grad-hook
semantics with two retained-graph VJPs and frozen real model weights.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.realq_checkpoint_forward_prototype import wrap_block_forward, unwrap_block_forward

import torch
from torch.utils.checkpoint import checkpoint
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding


def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=64)
    args = parser.parse_args()
    if args.seq_len <= 0:
        parser.error("seq-len must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(44)
    config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    config._attn_implementation = "eager"
    layer = Qwen3DecoderLayer(config, 0).to(dtype=torch.bfloat16)
    prefix = "model.layers.0."
    loaded = {}
    for file in sorted(args.model_path.glob("*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as source:
            for key in source.keys():
                if key.startswith(prefix):
                    loaded[key[len(prefix):]] = source.get_tensor(key)
    layer.load_state_dict(loaded, strict=True)
    del loaded
    layer.eval()
    for parameter in layer.parameters():
        parameter.requires_grad_(False)
    rotary = Qwen3RotaryEmbedding(config)
    x = torch.randn(1, args.seq_len, config.hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(args.seq_len).unsqueeze(0)
    embeddings = rotary(x, position_ids)
    # A real causal shape with no dropout; input values are synthetic.
    mask = torch.full((args.seq_len,args.seq_len), float("-inf"), dtype=torch.float32).triu(1)[None,None]
    targets = [torch.randn_like(x) for _ in range(2)]
    names = ("block.output", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    reports = {}
    for mode in ("ordinary", "nonreentrant_checkpoint", "bound_forward_checkpoint"):
        counters = {name: {"forward": 0, "grad": 0, "grad_sha256": []} for name in names}
        handles = []
        def make_hook(name):
            def forward_hook(module, inputs, output):
                counters[name]["forward"] += 1
                def grad_hook(grad):
                    counters[name]["grad"] += 1
                    counters[name]["grad_sha256"].append(digest(grad))
                output.register_hook(grad_hook)
            return forward_hook
        for name in names:
            module = layer if name == "block.output" else layer.get_submodule(name)
            handles.append(module.register_forward_hook(make_hook(name)))
        value = x.detach().clone().requires_grad_(True)
        def forward(current):
            return layer(current, attention_mask=mask, position_ids=position_ids,
                         position_embeddings=embeddings, use_cache=False)
        started = time.monotonic()
        if mode == "bound_forward_checkpoint":
            wrap_block_forward(layer)
        try:
            output = checkpoint(forward, value, use_reentrant=False) if mode == "nonreentrant_checkpoint" else forward(value)
            if isinstance(output, tuple):
                output = output[0]
            gradients = []
            for probe, target in enumerate(targets):
                value.grad = None
                (output.float()*target.float()).sum().backward(retain_graph=probe == 0)
                gradients.append(digest(value.grad))
            reports[mode] = {"output_sha256": digest(output), "input_gradient_sha256_by_probe": gradients,
                             "hook_records": counters, "cpu_diagnostic_seconds": time.monotonic()-started}
        finally:
            for handle in handles:
                handle.remove()
            if mode == "bound_forward_checkpoint":
                unwrap_block_forward(layer)
    left = reports["ordinary"]
    exact = True
    for mode in ("nonreentrant_checkpoint", "bound_forward_checkpoint"):
        right = reports[mode]
        exact = exact and left["output_sha256"] == right["output_sha256"] and left["input_gradient_sha256_by_probe"] == right["input_gradient_sha256_by_probe"]
        exact = exact and all(left["hook_records"][name]["grad_sha256"] == right["hook_records"][name]["grad_sha256"]
                              and left["hook_records"][name]["grad"] == right["hook_records"][name]["grad"] == 2 for name in names)
    result = {"scope": "CPU single actual-weight block; synthetic inputs/VJPs; not Stage-0/FSDP/GPU-memory validation",
              "model": str(args.model_path), "layer": 0, "seq_len": args.seq_len,
              "torch": str(torch.__version__), "parameter_requires_grad": False,
              "exact_output_input_and_module_gradients": exact, "reports": reports}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key:value for key,value in result.items() if key != "reports"}, indent=2))
    print(json.dumps({mode: {name:(record["forward"],record["grad"]) for name,record in report["hook_records"].items()} for mode,report in reports.items()}, indent=2))
    if not exact:
        raise RuntimeError("checkpoint changed output/gradients or gradient-hook counts; do not integrate")


if __name__ == "__main__":
    main()
