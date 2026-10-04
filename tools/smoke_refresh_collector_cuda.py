#!/usr/bin/env python3
"""Run REAL-Q's actual true-KL refresh collector on a small CUDA fixture.

Unrelated model-loading / quantization helper modules are stubbed so the
collector can be exercised without importing the entire model stack. The
collector, its functional_call path, KL loss, and gradient accumulation are
loaded from gptq_utils/gptq_plus_utils.py unchanged.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import types
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _stub_module(name: str, attrs: dict | None = None, *, package=False):
    module = types.ModuleType(name)
    if package:
        module.__path__ = []
    for key, value in (attrs or {}).items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _load_realq_collector():
    # Load the real KL primitive first, then stub only unrelated imports made
    # at module import time. None of these stubs is reached by this KL path.
    from utils.loss_utils import tokenwise_kl_from_logits  # noqa: F401

    utils_pkg = sys.modules["utils"]
    for child in ("quant_utils", "memory_utils", "model_utils", "dist_utils", "rotation_utils"):
        attrs = {"ModelAnalyzer": object} if child == "model_utils" else {}
        attrs.update({"ActQuantWrapper": type("ActQuantWrapper", (), {})} if child == "quant_utils" else {})
        mod = _stub_module(f"utils.{child}", attrs)
        setattr(utils_pkg, child, mod)
    _stub_module(
        "utils.saliency_utils",
        {
            "clip_global_percentile_": lambda *a, **k: None,
            "global_percentile": lambda *a, **k: None,
            "grouped_channel_gram": lambda *a, **k: None,
            "grouped_gradient_norm_squared": lambda *a, **k: None,
        },
    )
    _stub_module(
        "realq",
        package=True,
    )
    _stub_module(
        "realq.alignment",
        {
            "RefreshTraceWriter": object,
            "default_refresh_trace_config": lambda *a, **k: {},
            "refresh_step_from_metrics": lambda *a, **k: None,
        },
    )
    _stub_module(
        "gptq_utils.diagnostics",
        {
            "DiagnosticRegistry": object,
            "parse_diagnose_targets": lambda *a, **k: [],
        },
    )
    _stub_module(
        "gptq_utils.quant_aware_utils",
        {
            "configure_activation_quantizers_for_gptq": lambda *a, **k: None,
            "configure_k_cache_quantizers_for_gptq": lambda *a, **k: None,
            "disable_fp_path_quant": lambda *a, **k: None,
        },
    )
    return from_module()


def from_module():
    from gptq_utils.gptq_plus_utils import collect_true_weight_gradient

    return collect_true_weight_gradient


class _OneProjection(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, hidden, **_kwargs):
        return (self.proj(hidden),)


class _Analyzer:
    def __init__(self, hidden: int, vocab: int, device, dtype):
        self.norm = nn.Identity()
        self.lm_head = nn.Linear(hidden, vocab, bias=False, device=device, dtype=dtype)

    def get_layernorm_before_head(self):
        return self.norm

    def get_lm_head(self):
        return self.lm_head


def _run(collector, layer, analyzer, inps, fp_inps, refresh_mb):
    batch, seq_len, hidden = inps.shape
    result = collector(
        layer=layer,
        analyzer=analyzer,
        module_name="proj",
        full={"proj": layer.proj},
        inps=inps,
        fp_inps=fp_inps,
        attention_mask=torch.ones(1, 1, seq_len, seq_len, device=inps.device),
        position_ids=torch.zeros(1, seq_len, dtype=torch.long, device=inps.device),
        position_embeddings=(
            torch.zeros(1, seq_len, hidden, device=inps.device, dtype=inps.dtype),
            torch.zeros(1, seq_len, hidden, device=inps.device, dtype=inps.dtype),
        ),
        bsz=batch,
        kl_topk=-1,
        dev=inps.device,
        sample_indices=list(range(batch)),
        refresh_loss_type="kl",
        refresh_mb=refresh_mb,
    )
    grad_sum, count, loss_sum, _ = result
    return grad_sum / count, loss_sum / count, count


def _measure(collector, layer, analyzer, inps, fp_inps, microbatch):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    grad, loss, count = _run(collector, layer, analyzer, inps, fp_inps, microbatch)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (
        grad.detach().float().cpu(),
        float(loss),
        count,
        elapsed,
        torch.cuda.max_memory_allocated(),
        torch.cuda.max_memory_reserved(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--microbatch", type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    collector = _load_realq_collector()
    torch.manual_seed(314159)
    torch.cuda.manual_seed_all(314159)
    device, dtype = torch.device("cuda:0"), torch.bfloat16
    layer = _OneProjection(args.hidden).to(device=device, dtype=dtype)
    layer.proj.weight.data.mul_(0.15)
    analyzer = _Analyzer(args.hidden, args.vocab, device, dtype)
    analyzer.lm_head.weight.requires_grad_(False)
    inps = torch.randn(args.batch, args.seq_len, args.hidden, device=device, dtype=dtype)
    fp_inps = torch.randn_like(inps)

    full = _measure(collector, layer, analyzer, inps, fp_inps, None)
    chunked = _measure(collector, layer, analyzer, inps, fp_inps, args.microbatch)
    grad_abs = (full[0] - chunked[0]).abs()
    cos = torch.nn.functional.cosine_similarity(full[0].reshape(1, -1), chunked[0].reshape(1, -1))
    result = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dtype": str(dtype),
        "batch": args.batch,
        "seq_len": args.seq_len,
        "hidden": args.hidden,
        "vocab": args.vocab,
        "microbatch": args.microbatch,
        "implementation": "actual collect_true_weight_gradient from gptq_plus_utils.py; unrelated imports stubbed",
        "full_batch": {
            "loss": full[1], "count": full[2], "seconds": full[3],
            "peak_allocated_gib": full[4] / 1024**3,
            "peak_reserved_gib": full[5] / 1024**3,
        },
        "microbatched": {
            "loss": chunked[1], "count": chunked[2], "seconds": chunked[3],
            "peak_allocated_gib": chunked[4] / 1024**3,
            "peak_reserved_gib": chunked[5] / 1024**3,
        },
        "equivalence": {
            "loss_abs_diff": abs(full[1] - chunked[1]),
            "gradient_max_abs_diff": float(grad_abs.max()),
            "gradient_relative_l2_diff": float(grad_abs.norm() / full[0].norm().clamp_min(1e-12)),
            "gradient_cosine": float(cos),
        },
        "peak_allocated_reduction_fraction": 1.0 - chunked[4] / full[4],
    }
    result["passed"] = (
        full[2] == chunked[2] == args.batch
        and result["equivalence"]["loss_abs_diff"] < 1e-6
        and result["equivalence"]["gradient_max_abs_diff"] < 5e-6
        and result["equivalence"]["gradient_relative_l2_diff"] < 5e-3
        and result["equivalence"]["gradient_cosine"] > 0.99999
        and chunked[4] < full[4]
    )
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
