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
