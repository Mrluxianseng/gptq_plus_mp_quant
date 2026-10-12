"""Portable baseline-only state/held-out layer fingerprint wrapper.

The full-pipeline runner copies this entry and adds process-peak reporting.
It does not depend on historical files under outputs/ or experimental repair
monkeypatches. Diagnostic hashes are outside timed quantization.
"""
import hashlib
import json
import logging
import os
from pathlib import Path
import runpy
import sys
import time

sys.path.insert(0, str(Path.cwd()))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from gptq_utils import gptq_plus_utils as impl
from utils import eval_utils

# Experimental entry only: remove this marker before the production parser.
# Both arms use this same entry/source. No default model behavior changes.
_checkpoint_enabled = "--research_stage0_checkpoint" in sys.argv
_attention_checkpoint_enabled = "--research_stage0_attention_checkpoint" in sys.argv
_frozen_weight_offload_enabled = "--research_stage0_frozen_weight_offload" in sys.argv
_no_backward_unshard = "--research_stage0_no_backward_unshard" in sys.argv
if _checkpoint_enabled and _attention_checkpoint_enabled:
    raise ValueError("choose block or attention checkpoint, not both")
if _checkpoint_enabled:
    sys.argv.remove("--research_stage0_checkpoint")
if _attention_checkpoint_enabled:
    sys.argv.remove("--research_stage0_attention_checkpoint")
if _frozen_weight_offload_enabled:
    sys.argv.remove("--research_stage0_frozen_weight_offload")
if _no_backward_unshard:
    sys.argv.remove("--research_stage0_no_backward_unshard")
    if not _frozen_weight_offload_enabled:
        raise ValueError("backward-unshard pilot requires independent saved frozen parameter migration")
if _frozen_weight_offload_enabled and (_checkpoint_enabled or _attention_checkpoint_enabled):
    raise ValueError("frozen-weight and checkpoint candidates must first be tested independently")
_original_static_collector = impl.collect_static_end_to_end_saliency_and_fisher


def _checkpoint_static_collector(*, model, analyzer, **kwargs):
    from functools import wraps
    from torch.utils.checkpoint import checkpoint
    from utils.frozen_parameter_offload_prototype import FrozenParameterRegistry, FrozenParameterCpuCache
    from contextlib import nullcontext
    # Bind the tools-only helper into the copied entry snapshot. It is checked
    # again after collection, so changing it invalidates the worker rather than
    # escaping the production runner's utils/gptq_utils source snapshot.
    helper_path = Path.cwd() / "tools/frozen_residency_prototype_20261010.py"
    helper_sha = "06feca6cc26991f1d2e700af3e8e73e5ff1e643ab8f6da5b2ae96d32abcde287"
    if hashlib.sha256(helper_path.read_bytes()).hexdigest() != helper_sha:
        raise RuntimeError("frozen residency helper differs from entry-bound snapshot")
    from tools.frozen_residency_prototype_20261010 import AllFrozenParameterRegistry, AllFrozenParameterCpuCache, fsdp_without_backward_unshard
    layers = analyzer.get_layers()
    originals, handles = [], []
    counters = {}
    frozen_registry = AllFrozenParameterRegistry() if _no_backward_unshard else FrozenParameterRegistry()
    frozen_caches = []
    original_cpu_cache = impl.SavedTensorCpuCache
    # Same lightweight identity observers in both arms. Registry stores weak
    # storage references and versions, never a GPU tensor or parameter.
    handles.extend(frozen_registry.attach(model))
    if _frozen_weight_offload_enabled:
        if not kwargs.get("saved_tensor_offload_deduplicate") or kwargs.get("saved_tensor_offload_threshold_mb", 0) <= 0:
            raise ValueError("frozen-weight pilot requires positive saved-offload threshold and deduplication")
        cache_type = AllFrozenParameterCpuCache if _no_backward_unshard else FrozenParameterCpuCache
        class RegisteredFrozenCache(cache_type):
            def __init__(self, threshold_bytes, *, deduplicate=True):
                super().__init__(threshold_bytes, registry=frozen_registry, deduplicate=deduplicate)
                frozen_caches.append(self)
        impl.SavedTensorCpuCache = RegisteredFrozenCache
    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    sdpa_calls = [0]
    def checkpoint_sdpa(*values, **options):
        if not torch.is_grad_enabled():
            return original_sdpa(*values, **options)
        sdpa_calls[0] += 1
        return checkpoint(original_sdpa, *values, use_reentrant=False, **options)
    def capture(name):
        def forward_hook(module, inputs, output):
            counters[name]["forward"] += 1
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                def backward_hook(gradient):
                    counters[name]["grad"] += 1
                tensor.register_hook(backward_hook)
        return forward_hook
    def wrapped(original):
        @wraps(original)
        def forward(*values, **options):
            if not torch.is_grad_enabled():
                return original(*values, **options)
            return checkpoint(original, *values, use_reentrant=False, **options)
        return forward
    for index, layer in enumerate(layers):
        for module_name in ("", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            module = layer if not module_name else layer.get_submodule(module_name)
            name = f"{index}.{module_name or 'block.output'}"
            counters[name] = {"forward": 0, "grad": 0}
            handles.append(module.register_forward_hook(capture(name)))
        if _checkpoint_enabled:
            originals.append((layer, layer.forward))
            layer.forward = wrapped(layer.forward)
    if _attention_checkpoint_enabled:
        torch.nn.functional.scaled_dot_product_attention = checkpoint_sdpa
    try:
        if _no_backward_unshard and not kwargs.get("use_fsdp"):
            raise ValueError("backward-unshard pilot requires actual FSDP collection")
        with fsdp_without_backward_unshard(model) if _no_backward_unshard else nullcontext({}) as fsdp_report:
            result = _original_static_collector(model=model, analyzer=analyzer, **kwargs)
        if _attention_checkpoint_enabled and sdpa_calls[0] == 0:
            raise RuntimeError("attention checkpoint requested but no SDPA operation was checkpointed")
        if _frozen_weight_offload_enabled and sum(cache.stats["frozen_parameter_requests"] for cache in frozen_caches) == 0:
            raise RuntimeError("frozen-weight offload requested but no saved frozen parameter matched")
        return result
    finally:
        helper_changed = hashlib.sha256(helper_path.read_bytes()).hexdigest() != helper_sha
        impl.SavedTensorCpuCache = original_cpu_cache
        torch.nn.functional.scaled_dot_product_attention = original_sdpa
        for layer, original in originals:
            layer.forward = original
        for handle in handles:
            handle.remove()
        report = {"checkpoint_enabled": _checkpoint_enabled or _attention_checkpoint_enabled,
                  "checkpoint_kind": "block" if _checkpoint_enabled else "attention" if _attention_checkpoint_enabled else "off",
                  "frozen_weight_offload_enabled": _frozen_weight_offload_enabled,
                  "frozen_parameter_registration_calls": frozen_registry.registration_calls,
                  "frozen_parameter_saved_offload_stats": [dict(cache.stats) for cache in frozen_caches],
                  "backward_unshard_pilot": _no_backward_unshard,
                  "backward_unshard_report": locals().get("fsdp_report", {}),
                  "residency_helper_sha256": helper_sha,
                  "checkpointed_sdpa_calls": sdpa_calls[0], "rank": int(os.environ.get("RANK", "0")),
                  "scope": "actual static collector grad-hook counts; cache comparison still required", "modules": counters}
        destination = Path(os.environ["RESEARCH_RUN_DIR"])
        (destination / f"checkpoint_hook_audit_rank{os.environ.get('RANK', '0')}.json").write_text(json.dumps(report, indent=2))
        logging.info("STAGE0_CHECKPOINT_HOOK_AUDIT %s", json.dumps({"enabled": report["checkpoint_enabled"],
                     "kind": report["checkpoint_kind"], "checkpointed_sdpa_calls": sdpa_calls[0],
                     "frozen_weight_offload": _frozen_weight_offload_enabled,
                     "modules": len(counters), "grad_hook_calls": sum(row["grad"] for row in counters.values())}))
        if helper_changed:
            raise RuntimeError("frozen residency helper changed during collection; hooks restored")


impl.collect_static_end_to_end_saliency_and_fisher = _checkpoint_static_collector

mode = sys.argv.pop(1)
if mode != "baseline-profile":
    raise ValueError("the memory entry supports baseline-profile only")
root = Path(os.environ["RESEARCH_RUN_DIR"])
root.mkdir(parents=True, exist_ok=True)
original_fwrd = impl.gptq_fwrd


def measured_fwrd(args, analyzer, dataloader, dev):
    torch.cuda.synchronize()
    started = time.monotonic()
    result = original_fwrd(args, analyzer, dataloader, dev)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    digest = hashlib.sha256()
    count = 0
    for name, tensor in sorted(analyzer.model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        count += 1
    row = {"mode": mode, "quantization_seconds": elapsed,
           "state_sha256": digest.hexdigest(), "state_tensors": count,
           "peak_allocated": torch.cuda.max_memory_allocated(),
           "peak_reserved": torch.cuda.max_memory_reserved()}
    if int(os.environ.get("RANK", "0")) == 0:
        (root / "baseline-profile_metrics.json").write_text(json.dumps(row, indent=2))
    # Keep this hook point stable for the rank-safe full-pipeline wrapper.
    logging.info('OVERNIGHT_METRICS %s',json.dumps(row))
    return result


impl.gptq_fwrd = measured_fwrd
_original_dump_wall_summary = impl.QuantProfileRecorder.dump_wall_summary
impl.QuantProfileRecorder.dump_wall_summary = classmethod(lambda cls, top_k=60: _original_dump_wall_summary(top_k=1000))
_layer_trace_original_eval = eval_utils.kl_ppl_eval


def _layer_trace_eval(args, analyzer, *eval_args, **eval_kwargs):
    model = analyzer.model
    body = getattr(model, "model", None)
    layers = getattr(body, "layers", None) if body is not None else None
    if layers is None:
        layers = getattr(model, "layers", None)
    if layers is None:
        raise RuntimeError("cannot locate Transformer layers for held-out fingerprints")
    records = [{"digest": hashlib.sha256(), "count": 0, "shapes": []} for _ in layers]
    handles = []
    def hook(index):
        def capture(module, inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(value, torch.Tensor):
                raise TypeError("Transformer output fingerprint expects a tensor")
            value = value.detach().contiguous()
            record = records[index]
            record["digest"].update(str((tuple(value.shape), str(value.dtype))).encode())
            record["digest"].update(value.view(torch.uint8).cpu().numpy().tobytes())
            record["count"] += 1
            record["shapes"].append(list(value.shape))
        return capture
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(hook(i)))
    try:
        return _layer_trace_original_eval(args, analyzer, *eval_args, **eval_kwargs)
    finally:
        for handle in handles:
            handle.remove()
        report = [{"layer": i, "sha256": record["digest"].hexdigest(),
                   "forward_count": record["count"], "shapes": record["shapes"]}
                  for i, record in enumerate(records)]
        (root / "layer_output_fingerprints.json").write_text(json.dumps(report, indent=2))


eval_utils.kl_ppl_eval = _layer_trace_eval
sys.argv[0] = "ptq.py"
runpy.run_path('ptq.py', run_name='__main__')
