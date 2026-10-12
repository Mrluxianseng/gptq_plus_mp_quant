"""CUDA actual-weight Qwen block check before Stage-0 recomputation.

This is not a FSDP or real calibration experiment. Its local block peaks
are not full-model memory claims. Use on an idle GPU after current full
campaigns finish; never run concurrently with their profiled workloads.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import sys
import gc
import os
from contextlib import nullcontext
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.realq_checkpoint_forward_prototype import wrap_block_forward, unwrap_block_forward
from utils.frozen_parameter_offload_prototype import FrozenParameterRegistry, FrozenParameterCpuCache

import torch
from torch.utils.checkpoint import checkpoint
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding


def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--attention", choices=("eager", "math-sdpa"), default="math-sdpa")
    parser.add_argument("--all-frozen-parameter-offload", action="store_true")
    args = parser.parse_args()
    if args.seq_len <= 0:
        parser.error("seq-len must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(44)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this is not the CPU preflight tool")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    config._attn_implementation = "eager" if args.attention == "eager" else "sdpa"
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
    layer = layer.cuda()
    layer.eval()
    for parameter in layer.parameters():
        parameter.requires_grad_(False)
    rotary = Qwen3RotaryEmbedding(config).cuda()
    x = torch.randn(1, args.seq_len, config.hidden_size, dtype=torch.bfloat16, device="cuda")
    position_ids = torch.arange(args.seq_len, device="cuda").unsqueeze(0)
    embeddings = rotary(x, position_ids)
    # A real causal shape with no dropout; input values are synthetic.
    mask = torch.full((args.seq_len,args.seq_len), float("-inf"), dtype=torch.float32, device="cuda").triu(1)[None,None]
    targets = [torch.randn_like(x) for _ in range(2)]
    # cuBLAS lazily creates persistent workspaces on the first GEMMs/backward.
    # Initialize them before any mode baseline so that ordinary is not charged
    # for a one-time library allocation missing from the later modes.
    warm_value = x.detach().clone().requires_grad_(True)
    warm_output = layer(warm_value, attention_mask=mask, position_ids=position_ids,
                        position_embeddings=embeddings, use_cache=False)
    if isinstance(warm_output, tuple):
        warm_output = warm_output[0]
    (warm_output.float()*targets[0].float()).sum().backward()
    del warm_output, warm_value
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    names = ("block.output", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    reports = {}
    modes = ["ordinary", "nonreentrant_checkpoint", "bound_forward_checkpoint"]
    if args.attention == "math-sdpa":
        modes.append("attention_core_checkpoint")
    modes.append("frozen_parameter_saved_offload")
    if args.all_frozen_parameter_offload:
        modes.append("all_frozen_parameter_saved_offload")
    for mode in modes:
        frozen_registry = FrozenParameterRegistry()
        frozen_cache = FrozenParameterCpuCache(4 * 2**20, registry=frozen_registry)
        if mode == "all_frozen_parameter_saved_offload":
            from tools.frozen_residency_prototype_20261010 import AllFrozenParameterRegistry, AllFrozenParameterCpuCache
            frozen_registry = AllFrozenParameterRegistry()
            frozen_cache = AllFrozenParameterCpuCache(4 * 2**20, registry=frozen_registry)
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
        if mode in ("frozen_parameter_saved_offload", "all_frozen_parameter_saved_offload"):
            handles.extend(frozen_registry.attach(layer))
        value = x.detach().clone().requires_grad_(True)
        def forward(current):
            return layer(current, attention_mask=mask, position_ids=position_ids,
                         position_embeddings=embeddings, use_cache=False)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_allocated = torch.cuda.memory_allocated()
        started = time.monotonic()
        if mode == "bound_forward_checkpoint":
            wrap_block_forward(layer)
        original_sdpa = torch.nn.functional.scaled_dot_product_attention
        sdpa_calls = [0]
        if mode == "attention_core_checkpoint":
            def checkpoint_sdpa(*values, **options):
                sdpa_calls[0] += 1
                return checkpoint(original_sdpa, *values, use_reentrant=False, **options)
            torch.nn.functional.scaled_dot_product_attention = checkpoint_sdpa
        output = None
        try:
            with frozen_cache.context() if mode in ("frozen_parameter_saved_offload", "all_frozen_parameter_saved_offload") else nullcontext():
                output = checkpoint(forward, value, use_reentrant=False) if mode == "nonreentrant_checkpoint" else forward(value)
                if isinstance(output, tuple):
                    output = output[0]
                gradients = []
                for probe, target in enumerate(targets):
                    value.grad = None
                    (output.float()*target.float()).sum().backward(retain_graph=probe == 0)
                    gradients.append(digest(value.grad))
            torch.cuda.synchronize()
            reports[mode] = {"output_sha256": digest(output), "input_gradient_sha256_by_probe": gradients,
                             "hook_records": counters, "cuda_diagnostic_seconds": time.monotonic()-started,
                             "checkpointed_sdpa_forward_calls": sdpa_calls[0],
                             "saved_offload_stats": dict(frozen_cache.stats),
                             "baseline_allocated_bytes": baseline_allocated,
                             "single_block_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                             "single_block_peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        finally:
            for handle in handles:
                handle.remove()
            if mode == "bound_forward_checkpoint":
                unwrap_block_forward(layer)
            torch.nn.functional.scaled_dot_product_attention = original_sdpa
            del value, output
    left = reports["ordinary"]
    exact = True
    for mode in modes[1:]:
        right = reports[mode]
        exact = exact and left["output_sha256"] == right["output_sha256"] and left["input_gradient_sha256_by_probe"] == right["input_gradient_sha256_by_probe"]
        exact = exact and all(left["hook_records"][name]["grad_sha256"] == right["hook_records"][name]["grad_sha256"]
                              and left["hook_records"][name]["grad"] == right["hook_records"][name]["grad"] == 2 for name in names)
    if "attention_core_checkpoint" in reports:
        exact = exact and reports["attention_core_checkpoint"]["checkpointed_sdpa_forward_calls"] > 0
    exact = exact and reports["frozen_parameter_saved_offload"]["saved_offload_stats"]["frozen_parameter_requests"] > 0
    if args.all_frozen_parameter_offload:
        exact = exact and reports["all_frozen_parameter_saved_offload"]["saved_offload_stats"]["frozen_parameter_requests"] > reports["frozen_parameter_saved_offload"]["saved_offload_stats"]["frozen_parameter_requests"]
    result = {"scope": "CUDA single actual-weight block; synthetic inputs/VJPs; not Stage-0/FSDP/full-model validation",
              "model": str(args.model_path), "layer": 0, "seq_len": args.seq_len,
              "attention": args.attention, "gpu": torch.cuda.get_device_name(),
              "torch": str(torch.__version__), "parameter_requires_grad": False,
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "all_frozen_parameter_mode_requested": args.all_frozen_parameter_offload,
              "exact_output_input_and_module_gradients": exact, "reports": reports}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key:value for key,value in result.items() if key != "reports"}, indent=2))
    print(json.dumps({mode: {name:(record["forward"],record["grad"]) for name,record in report["hook_records"].items()} for mode,report in reports.items()}, indent=2))
    if not exact:
        raise RuntimeError("checkpoint changed output/gradients or gradient-hook counts; do not integrate")


if __name__ == "__main__":
    main()
