"""Capture sequence-output score gradients during REAL-Q's existing Stage-0 backward.

This is a bounded diagnostic: it runs a tiny Stage-0 precompute and saves only
one full-sequence gradient vector per calibration sample for selected layers.
It does not add any forward/backward pass to the existing precompute loop.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from realq.config import Config
from realq.pipeline import run as run_pipeline
from realq.precompute import hooks as hooks_mod
from realq.precompute.hooks import LOSS_GRAD_SCALE
from utils.reproducibility import configure_reproducibility


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 3])
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    configure_reproducibility(args.seed, deterministic=True)
    torch.set_num_threads(4)

    captured: dict[int, list[torch.Tensor]] = {int(i): [] for i in args.layers}
    original_manager = hooks_mod.FisherHookManager

    class CapturingFisherHookManager(original_manager):
        def attach(self, layers):
            layers = list(layers)
            super().attach(layers)
            for layer_idx in captured:
                if layer_idx >= len(layers):
                    raise ValueError(f"requested layer {layer_idx}, model has {len(layers)} layers")

                def forward_hook(_module, _inputs, outputs, layer_idx=layer_idx):
                    tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

                    def grad_hook(grad):
                        # Match REAL-Q's unscaled output score gradient. Keep
                        # one sequence vector per sample, not tokenwise HxH.
                        values = grad.detach().float().div_(LOSS_GRAD_SCALE).cpu()
                        captured[layer_idx].extend(
                            row.contiguous() for row in values
                        )

                    tensor.register_hook(grad_hook)

                self._handles.append(layers[layer_idx].register_forward_hook(forward_hook))

    hooks_mod.FisherHookManager = CapturingFisherHookManager
    cfg = Config(
        model=args.model,
        dataset="wikitext2",
        seed=args.seed,
        nsamples=args.samples,
        seq_len=args.seq_len,
        global_loss_bsz=1,
        skip_eval=True,
        skip_kl_ppl_eval=True,
        exit_after_precompute=True,
        static_cache_path=None,
        fsdp=False,
        # The transition/HVP campaign is explicitly unrotated; keep the
        # representation identical rather than inheriting REAL-Q's default.
        rotate=False,
    )
    try:
        run_pipeline(cfg)
    finally:
        hooks_mod.FisherHookManager = original_manager

    manifest = {
        "model": args.model,
        "layers": {},
        "samples": args.samples,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "collection": "same Stage-0 sampled-label backward as REAL-Q Fisher; no added backward",
        "gradient_scale_removed": float(LOSS_GRAD_SCALE),
    }
    for layer_idx, rows in captured.items():
        if len(rows) != args.samples:
            raise RuntimeError(
                f"layer {layer_idx}: captured {len(rows)} sample gradients, expected {args.samples}"
            )
        gradients = torch.stack(rows, dim=0).float().cpu()
        target = args.out / f"layer{layer_idx}_sequence_gradients.pt"
        torch.save(gradients, target)
        manifest["layers"][str(layer_idx)] = {
            "path": str(target),
            "shape": list(gradients.shape),
            "bytes": target.stat().st_size,
            "gradient_rms": float(gradients.square().mean().sqrt()),
        }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
