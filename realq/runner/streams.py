"""Layer-by-layer input streaming.

Mirrors ``utils.eval_utils._get_logits``'s Catcher trick: pull the first
transformer layer's input by intercepting its forward, run pre-block
modules (embed_tokens / rotary_emb) once on calibration data, save
``inps``, ``attention_mask``, ``position_ids``, ``position_embeddings``.

For sub-task 3 we only maintain ``q_inps`` (inputs reflecting all already-
quantised upstream layers). ``fp_inps`` stays a TODO until sub-task 5,
when block_gd needs it for the FP reference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from tqdm import tqdm

if TYPE_CHECKING:
    from realq.parallel.cpu_master import CpuMasterLayerManager
    from utils.model_utils import ModelAnalyzer


@dataclass
class LayerInputs:
    """Per-rank streaming state passed to ``layer_loop.quantize_one_layer``.

    ``inps`` is the q-stream: each layer sees the OUTPUT of all previously-
    quantised upstream layers. ``fp_inps`` is the FP reference stream:
    each layer sees the OUTPUT of all upstream layers when their weights
    were unchanged FP. The two diverge once quantisation begins.

    For sub-task 3 we only used ``inps``; sub-task 5 (block_gd) needs
    ``fp_inps`` because the fisher_mse loss compares the quantised layer
    output ``q_layer(fp_inps_L)`` against the FP layer output
    ``fp_layer(fp_inps_L)``.
    """
    inps: torch.Tensor                 # (N_local, T, H) on cuda
    fp_inps: torch.Tensor              # (N_local, T, H) on cuda — FP-only stream
    attention_mask: torch.Tensor | None
    position_ids: torch.Tensor | None
    position_embeddings: tuple | None


@torch.no_grad()
def capture_layer0_inputs(
    analyzer: "ModelAnalyzer",
    samples: list[torch.Tensor],
    dev: torch.device,
    layer_manager: "CpuMasterLayerManager | None" = None,
) -> LayerInputs:
    """Run the model up to (but not through) layer 0 to harvest its input.

    ``samples``: list of 1D LongTensors (length seq_len). One per local
    calibration sample. Caller is responsible for sample sharding under DP.

    ``layer_manager`` is consulted for the cpu_master path: rank>0's model
    has meta-tensor weights, so ``mod.to(dev)`` would crash; the manager
    broadcasts rank0's tensors instead. When ``layer_manager`` is None or
    ``layer_manager.enabled`` is False, behaviour is identical to plain
    ``.to(dev)`` / ``.to(cpu)``.
    """
    model = analyzer.model
    layers = analyzer.get_layers()
    pre_block = analyzer.get_pre_block_modules()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    orig_device = next(model.parameters()).device

    use_manager = layer_manager is not None and layer_manager.enabled
    if use_manager:
        layer_manager.materialize_runtime_modules(pre_block)
        layers[0] = layer_manager.materialize_layer(0)
    else:
        for mod in pre_block:
            mod.to(dev)
        layers[0] = layers[0].to(dev)

    nsamples = len(samples)
    seq_len = samples[0].numel()
    hidden = model.config.hidden_size
    dtype = next(model.parameters()).dtype
    inps = torch.zeros((nsamples, seq_len, hidden), dtype=dtype, device=dev)
    cache: dict[str, object] = {"i": 0, "attention_mask": None,
                                "position_ids": None, "position_embeddings": None}

    class Catcher(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.module = inner
            if hasattr(inner, "attention_type"):
                self.attention_type = inner.attention_type

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            cache["position_embeddings"] = kwargs.get("position_embeddings")
            raise ValueError  # unwind out of the model.forward

    layers[0] = Catcher(layers[0])
    for s in tqdm(samples, ncols=80, desc="Capture layer-0 inputs", leave=False):
        try:
            model(s.view(1, -1).to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    if use_manager:
        # Release pre_block (embed/rotary not needed in Phase E) and layer 0
        # (Phase E's quantize_one_layer will materialise it again — idempotent
        # set tracks materialised state, so a fresh broadcast happens).
        layer_manager.release_runtime_modules(pre_block, torch.device("cpu"))
        layer_manager.release_layer(0, layers[0], orig_device=torch.device("cpu"))
    else:
        layers[0] = layers[0].to(orig_device)
        for mod in pre_block:
            mod.to(orig_device)

    model.config.use_cache = use_cache
    # Move the captured inps to CPU pinned memory if it would matter; for
    # 0.6B sub-task 3 we leave them on GPU.
    return LayerInputs(
        inps=inps,
        fp_inps=inps.clone(),  # at layer 0 the q-stream and FP stream coincide
        attention_mask=cache["attention_mask"],
        position_ids=cache["position_ids"],
        position_embeddings=cache["position_embeddings"],
    )


@torch.no_grad()
def replay_layer(
    layer: nn.Module,
    state: LayerInputs,
    bsz: int,
    inps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward each calibration sample through ``layer`` with current weights;
    return the layer outputs (= input for the next layer) on the same
    device/dtype as ``inps``.

    ``inps`` defaults to ``state.inps`` (the q-stream). Pass ``state.fp_inps``
    explicitly to materialise the FP-stream output for the next layer.
    """
    if inps is None:
        inps = state.inps
    n = inps.shape[0]
    outs = torch.empty_like(inps)
    am = state.attention_mask
    pi = state.position_ids
    pe = state.position_embeddings
    for j in range(0, n, bsz):
        b = min(bsz, n - j)
        kw = {}
        if am is not None:
            kw["attention_mask"] = am.expand(b, *am.shape[1:]) if am.shape[0] != b else am
        if pi is not None:
            kw["position_ids"] = pi.expand(b, -1) if pi.shape[0] != b else pi
        if pe is not None:
            kw["position_embeddings"] = (
                pe[0].expand(b, *pe[0].shape[1:]) if pe[0].shape[0] != b else pe[0],
                pe[1].expand(b, *pe[1].shape[1:]) if pe[1].shape[0] != b else pe[1],
            )
        out = layer(inps[j : j + b], **kw)
        outs[j : j + b] = out[0] if isinstance(out, tuple) else out
    return outs
