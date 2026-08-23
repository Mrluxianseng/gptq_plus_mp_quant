#!/usr/bin/env python3
"""Run ``realq.ptq`` with every reversible run15 execution fast path off.

This is a diagnostic entry point, not a new production configuration.  The
ordinary CLI flags disable the public run15 switches; this module additionally
restores the pre-run14 eager Qwen3/Fisher equations, the pre-run15 separately
scaled Fast Hadamard call, and the eager activation-QDQ fallback.
"""

from __future__ import annotations

import json
import os
import sys

import torch


def _install_legacy_paths() -> None:
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError(
            "legacy all-off driver requires NVIDIA_TF32_OVERRIDE=0 before "
            "the CUDA runtime is initialized"
        )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    from utils import hadamard_utils
    from utils import triton_activation_quant
    from utils import triton_qwen3_fusions

    extension = hadamard_utils.fast_hadamard_transform
    torch_fallback = hadamard_utils._hadamard_transform_torch

    def legacy_hadamard_transform(
        value: torch.Tensor, scale: float = 1.0
    ) -> torch.Tensor:
        if extension is not None and value.is_cuda:
            output = extension.hadamard_transform(value.contiguous())
        else:
            output = torch_fallback(value)
        return output if scale == 1.0 else output * scale

    hadamard_utils._hadamard_transform = legacy_hadamard_transform
    triton_activation_quant.can_fuse = lambda *_args, **_kwargs: False
    triton_qwen3_fusions.can_fuse_qk_rmsnorm_rope = (
        lambda *_args, **_kwargs: False
    )
    triton_qwen3_fusions.can_fuse_swiglu = lambda *_args, **_kwargs: False
    triton_qwen3_fusions.install_qwen3_swiglu_fusion = (
        lambda _model: 0
    )

    from realq.refresh import block_gd
    from realq.refresh import fisher_loss

    def fisher_mse_loss_fp32(
        q_out: torch.Tensor,
        fp_out: torch.Tensor,
        fisher: torch.Tensor,
        a_loss_ratio: float = 1.0,
        a_loss_threshold: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if q_out.shape != fp_out.shape:
            raise ValueError("all-off Fisher inputs have different shapes")
        hidden = q_out.shape[-1]
        if fisher.shape != (hidden, hidden):
            raise ValueError("all-off Fisher matrix has the wrong shape")
        delta = q_out - fp_out
        if a_loss_ratio < 1.0:
            delta = fisher_loss._scale_delta_by_abs_quantile(
                delta,
                float(a_loss_ratio),
                threshold=a_loss_threshold,
            )
        delta_flat = delta.float().reshape(-1, hidden)
        fisher_fp32 = fisher.to(device=delta.device, dtype=torch.float32)
        quadratic = (delta_flat @ fisher_fp32 * delta_flat).sum(dim=-1)
        return 0.5 * quadratic.mean()

    fisher_loss.fisher_mse_loss = fisher_mse_loss_fp32
    block_gd.fisher_mse_loss = fisher_mse_loss_fp32

    print(
        "REALQ_ALLOPTS_ABLATION="
        + json.dumps(
            {
                "attention": "deterministic_sdpa",
                "hessian": "fp32",
                "fisher": "eager_fp32",
                "qwen3_qk_rmsnorm_rope": "eager",
                "qwen3_swiglu": "eager",
                "activation_qdq": "eager",
                "fast_hadamard_scale": "separate",
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    _install_legacy_paths()
    from realq.ptq import main as ptq_main

    ptq_main()


if __name__ == "__main__":
    main()
