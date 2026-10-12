"""Opt-in, single-GPU exports around one real Adam/GPTQ transition.

Only copies tensors. Loss/held-out evaluation runs in a separate process,
so diagnostic forwards cannot alter the production quantization trajectory.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import asdict
from pathlib import Path

import torch


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_copy(item) for item in value)
    return value


def tensor_hash(value):
    value = value.detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


class TransitionProbe:
    def __init__(self, cfg, analyzer, layer_idx, module_name):
        self.cfg = cfg
        self.analyzer = analyzer
        self.layer_idx = layer_idx
        self.module_name = module_name
        self.block = int(os.environ["REALQ_ADAM_PROBE_BLOCK"])
        self.directory = Path(os.environ["REALQ_ADAM_TRANSITION_DIR"])
        self.directory.mkdir(parents=True, exist_ok=False)
        self.started = False
        self.completed = False
        self.before = None

    def weight(self, W, Q, perm, column_end):
        stitched = W.clone()
        stitched[:, :column_end] = Q[:, :column_end]
        return stitched if perm is None else stitched[:, torch.argsort(perm)]

    def before_block_refresh(self, block, W, Q, perm, column_end, hinv, quantizer):
        if self.completed:
            return
        if block == self.block:
            if self.started:
                raise RuntimeError("duplicate transition target")
            self.started = True
            self.perm = cpu_copy(perm)
            self.column_end = column_end
            self.before = cpu_copy(self.weight(W, Q, perm, column_end))
            layers = self.analyzer.get_layers()
            context = {
                "config": asdict(self.cfg), "layer": self.layer_idx,
                "module": self.module_name, "block": block,
                "column_end": column_end, "perm": self.perm,
                "prefix_layers": [cpu_copy(layer.state_dict()) for layer in layers[:self.layer_idx + 1]],
                "next_layer": cpu_copy(layers[self.layer_idx + 1].state_dict()),
                "quant_state_hashes": {
                    "working_weight": tensor_hash(W), "quantized_prefix": tensor_hash(Q[:, :column_end]),
                    "hinv": tensor_hash(hinv), "perm": tensor_hash(perm) if perm is not None else None,
                    "scale": tensor_hash(quantizer.scale), "zero": tensor_hash(quantizer.zero),
                },
            }
            torch.save(context, self.directory / "context.pt")
            torch.save(self.before, self.directory / "before_update.pt")
            logging.info("[adam_transition] saved before_update: %s", self.directory)
        elif self.started and block == self.block + 1:
            self.capture_weight("after_next_quant", W, Q, perm, column_end)
            self.completed = True
            (self.directory / "COMPLETE").write_text("actual next GPTQ block and outer compensation completed\n")
            if os.environ.get("REALQ_ADAM_TRANSITION_STOP", "1") == "1":
                logging.info("[adam_transition] diagnostic endpoint reached; stopping before next refresh")
                raise SystemExit(0)

    def capture_weight(self, stage, W, Q, perm, column_end):
        weight = cpu_copy(self.weight(W, Q, perm, column_end))
        fixed = (self.perm[:self.column_end] if self.perm is not None else torch.arange(self.column_end))
        if not torch.equal(weight[:, fixed], self.before[:, fixed]):
            raise RuntimeError(f"frozen prefix changed at {stage}")
        if not torch.isfinite(weight).all():
            raise RuntimeError(f"nonfinite weight at {stage}")
        torch.save(weight, self.directory / f"{stage}.pt")
        logging.info("[adam_transition] saved %s: %s", stage, self.directory)

    def capture_refresh(self, **values):
        if not self.started or self.completed:
            raise RuntimeError("refresh export outside the target transition")
        torch.save(cpu_copy(values), self.directory / "refresh.pt")


def make_transition_probe(cfg, analyzer, layer_idx, module_name):
    if not os.environ.get("REALQ_ADAM_TRANSITION_DIR"):
        return None
    if str(layer_idx) != os.environ.get("REALQ_ADAM_PROBE_LAYER") or module_name != os.environ.get("REALQ_ADAM_PROBE_MODULE"):
        return None
    from utils import dist_utils
    if (dist_utils.get_world_size() != 1 or cfg.group_parallel_quant != "none"
            or cfg.num_groups <= 1 or cfg.full_block_refresh or cfg.rotate
            or cfg.backward_bsz != 1 or cfg.a_loss_ratio != 1.0
            or cfg.a_bits < 16 or cfg.v_bits < 16 or cfg.k_bits < 16):
        raise ValueError("transition probe requires single-GPU grouped legacy refresh, no rotation, W-only quantization, backward_bsz=1, a_loss_ratio=1")
    if layer_idx >= analyzer.num_layers - 1:
        raise ValueError("transition probe does not cover the final-layer KL refresh")
    return TransitionProbe(cfg, analyzer, layer_idx, module_name)
