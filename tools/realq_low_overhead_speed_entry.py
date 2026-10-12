"""Speed entry without layer-output copies, NVML sampling or phase resets.

End-state SHA is calculated after timed quantization. That audit still
adds work to process wall time, which must not be called uninstrumented
end-to-end latency. Use quantization_seconds for the primary speed result.
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

mode = sys.argv.pop(1)
if mode != "baseline-profile":
    raise ValueError("the speed entry supports baseline-profile only")
destination = Path(os.environ["RESEARCH_RUN_DIR"])
rank = int(os.environ.get("RANK", "0"))
destination.mkdir(parents=True, exist_ok=True)
_checkpoint_stage0 = "--research_stage0_checkpoint" in sys.argv
if _checkpoint_stage0:
    sys.argv.remove("--research_stage0_checkpoint")
_original_static_collector = impl.collect_static_end_to_end_saliency_and_fisher


def _measured_static_collector(*, model, analyzer, **kwargs):
    from functools import wraps
    from torch.utils.checkpoint import checkpoint
    originals = []
    def wrapped(original):
        @wraps(original)
        def forward(*values, **options):
            if not torch.is_grad_enabled():
                return original(*values, **options)
            return checkpoint(original, *values, use_reentrant=False, **options)
        return forward
    if _checkpoint_stage0:
        for layer in analyzer.get_layers():
            originals.append((layer, layer.forward))
            layer.forward = wrapped(layer.forward)
    try:
        torch.cuda.synchronize()
        started = time.monotonic()
        result = _original_static_collector(model=model, analyzer=analyzer, **kwargs)
        torch.cuda.synchronize()
        record = {"rank": rank, "static_collection_seconds": time.monotonic() - started,
                  "checkpoint_enabled": _checkpoint_stage0,
                  "scope": "collector only; excludes model checkpoint preparation and external audits"}
        (destination / f"static_collection_timing_rank{rank}.json").write_text(json.dumps(record, indent=2))
        logging.info("LOW_OVERHEAD_STATIC_COLLECTION_TIMING %s", json.dumps(record))
        return result
    finally:
        for layer, original in originals:
            layer.forward = original


impl.collect_static_end_to_end_saliency_and_fisher = _measured_static_collector
original_fwrd = impl.gptq_fwrd
original_eval = eval_utils._kl_ppl_eval
quality = []


def measured_fwrd(args, analyzer, dataloader, dev):
    torch.cuda.synchronize()
    started = time.monotonic()
    result = original_fwrd(args, analyzer, dataloader, dev)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    audit_started = time.monotonic()
    digest = hashlib.sha256()
    count = 0
    for name, tensor in sorted(analyzer.model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        count += 1
    record = {"rank": rank, "quantization_seconds": elapsed,
              "state_sha256": digest.hexdigest(), "state_tensors": count,
              "post_timing_state_audit_seconds": time.monotonic()-audit_started,
              "layer_output_capture": False, "exact_phase_memory_profile": False,
              "component_event_profile": False}
    (destination / f"speed_metrics_rank{rank}.json").write_text(json.dumps(record, indent=2))
    logging.info("LOW_OVERHEAD_SPEED_METRICS %s", json.dumps(record))
    return result


def measured_eval(*args, **kwargs):
    result = original_eval(*args, **kwargs)
    ppl, kl = result
    quality.append({"kl": kl, "ppl": ppl})
    (destination / f"quality_rank{rank}.json").write_text(json.dumps(quality, indent=2))
    return result


impl.gptq_fwrd = measured_fwrd
eval_utils._kl_ppl_eval = measured_eval
sys.argv[0] = "ptq.py"
try:
    runpy.run_path("ptq.py", run_name="__main__")
except SystemExit as stopped:
    if stopped.code not in (None, 0):
        raise
if torch.distributed.is_available() and torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()
