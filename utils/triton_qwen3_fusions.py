"""Triton forward/backward fusions for dense Qwen3 refresh forwards.

The production graph uses two deliberately narrow fusions:

* Q/K head-wise RMSNorm + RoPE.  Qwen3 applies an RMSNorm over the
  ``head_dim`` immediately before RoPE.  Executing both in one kernel avoids
  materialising the FP32 normalised tensors and the ``rotate_half`` neg/cat
  temporaries.
* SwiGLU.  SiLU and the following gate/up product share one streaming pass in
  forward; backward writes the gate and up gradients in one pass.

Neither equation contains a matrix multiply, so WGMMA/TCGen05 is inapplicable.
The useful Blackwell/Hopper optimisation is to keep each activation in
registers across the complete elementwise chain and use warp reductions for
RMSNorm rather than repeatedly traversing HBM.
"""
from __future__ import annotations

import types

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by CPU-only environments.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _qk_rmsnorm_rope_forward_kernel(
        x_ptr,
        weight_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        inv_rms_ptr,
        X_STRIDE_B: tl.constexpr,
        X_STRIDE_H: tl.constexpr,
        X_STRIDE_S: tl.constexpr,
        X_STRIDE_D: tl.constexpr,
        COS_STRIDE_B: tl.constexpr,
        COS_STRIDE_S: tl.constexpr,
        COS_STRIDE_D: tl.constexpr,
        SIN_STRIDE_B: tl.constexpr,
        SIN_STRIDE_S: tl.constexpr,
        SIN_STRIDE_D: tl.constexpr,
        N_HEADS: tl.constexpr,
        SEQ_LEN: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        COS_BATCH: tl.constexpr,
        SIN_BATCH: tl.constexpr,
        EPS: tl.constexpr,
        N_ROWS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        row_mask = rows < N_ROWS
        seq = rows % SEQ_LEN
        head_batch = rows // SEQ_LEN
        head = head_batch % N_HEADS
        batch = head_batch // N_HEADS
        dims = tl.arange(0, BLOCK_D)
        mask = row_mask[:, None] & (dims[None, :] < HEAD_DIM)

        x_offsets = (
            batch[:, None] * X_STRIDE_B
            + head[:, None] * X_STRIDE_H
            + seq[:, None] * X_STRIDE_S
            + dims[None, :] * X_STRIDE_D
        )
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=1) / HEAD_DIM
        inv_rms = tl.rsqrt(variance + EPS)

        # Match Qwen3RMSNorm's observable BF16/FP16 rounding points:
        # normalise in FP32, cast to the deployed dtype, then apply the norm
        # weight in that dtype.
        deployed_dtype = x_ptr.dtype.element_ty
        normalised = (x * inv_rms[:, None]).to(deployed_dtype)
        weight = tl.load(
            weight_ptr + dims[None, :],
            mask=dims[None, :] < HEAD_DIM,
            other=0.0,
        ).to(
            deployed_dtype
        )
        affine = (normalised * weight).to(deployed_dtype)

        half = HEAD_DIM // 2
        partner_dims = tl.where(dims < half, dims + half, dims - half)
        partner_offsets = (
            batch[:, None] * X_STRIDE_B
            + head[:, None] * X_STRIDE_H
            + seq[:, None] * X_STRIDE_S
            + partner_dims[None, :] * X_STRIDE_D
        )
        partner_x = tl.load(
            x_ptr + partner_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        partner_normalised = (partner_x * inv_rms[:, None]).to(deployed_dtype)
        partner_weight = tl.load(
            weight_ptr + partner_dims[None, :],
            mask=dims[None, :] < HEAD_DIM,
            other=0.0,
        ).to(deployed_dtype)
        partner_affine = (partner_normalised * partner_weight).to(
            deployed_dtype
        )
        rotated = tl.where(dims < half, -partner_affine, partner_affine).to(
            deployed_dtype
        )

        cos_batch = batch
        sin_batch = batch
        if COS_BATCH == 1:
            cos_batch = batch * 0
        if SIN_BATCH == 1:
            sin_batch = batch * 0
        cos_offsets = (
            cos_batch[:, None] * COS_STRIDE_B
            + seq[:, None] * COS_STRIDE_S
            + dims[None, :] * COS_STRIDE_D
        )
        sin_offsets = (
            sin_batch[:, None] * SIN_STRIDE_B
            + seq[:, None] * SIN_STRIDE_S
            + dims[None, :] * SIN_STRIDE_D
        )
        cos = tl.load(cos_ptr + cos_offsets, mask=mask, other=0.0).to(
            deployed_dtype
        )
        sin = tl.load(sin_ptr + sin_offsets, mask=mask, other=0.0).to(
            deployed_dtype
        )
        direct = (affine * cos).to(deployed_dtype)
        crossed = (rotated * sin).to(deployed_dtype)
        output = (direct + crossed).to(deployed_dtype)

        tl.store(
            out_ptr + rows[:, None] * HEAD_DIM + dims[None, :],
            output,
            mask=mask,
        )
        tl.store(inv_rms_ptr + rows, inv_rms, mask=row_mask)


    @triton.jit
    def _qk_rmsnorm_rope_backward_kernel(
        grad_ptr,
        x_ptr,
        weight_ptr,
        cos_ptr,
        sin_ptr,
        inv_rms_ptr,
        grad_x_ptr,
        G_STRIDE_B: tl.constexpr,
        G_STRIDE_H: tl.constexpr,
        G_STRIDE_S: tl.constexpr,
        G_STRIDE_D: tl.constexpr,
        X_STRIDE_B: tl.constexpr,
        X_STRIDE_H: tl.constexpr,
        X_STRIDE_S: tl.constexpr,
        X_STRIDE_D: tl.constexpr,
        COS_STRIDE_B: tl.constexpr,
        COS_STRIDE_S: tl.constexpr,
        COS_STRIDE_D: tl.constexpr,
        SIN_STRIDE_B: tl.constexpr,
        SIN_STRIDE_S: tl.constexpr,
        SIN_STRIDE_D: tl.constexpr,
        N_HEADS: tl.constexpr,
        SEQ_LEN: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        COS_BATCH: tl.constexpr,
        SIN_BATCH: tl.constexpr,
        N_ROWS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        row_mask = rows < N_ROWS
        seq = rows % SEQ_LEN
        head_batch = rows // SEQ_LEN
        head = head_batch % N_HEADS
        batch = head_batch // N_HEADS
        dims = tl.arange(0, BLOCK_D)
        mask = row_mask[:, None] & (dims[None, :] < HEAD_DIM)
        half = HEAD_DIM // 2
        deployed_dtype = x_ptr.dtype.element_ty

        grad_offsets = (
            batch[:, None] * G_STRIDE_B
            + head[:, None] * G_STRIDE_H
            + seq[:, None] * G_STRIDE_S
            + dims[None, :] * G_STRIDE_D
        )
        grad = tl.load(grad_ptr + grad_offsets, mask=mask, other=0.0).to(
            deployed_dtype
        )
        cos_batch = batch
        sin_batch = batch
        if COS_BATCH == 1:
            cos_batch = batch * 0
        if SIN_BATCH == 1:
            sin_batch = batch * 0
        cos_offsets = (
            cos_batch[:, None] * COS_STRIDE_B
            + seq[:, None] * COS_STRIDE_S
            + dims[None, :] * COS_STRIDE_D
        )
        sin_offsets = (
            sin_batch[:, None] * SIN_STRIDE_B
            + seq[:, None] * SIN_STRIDE_S
            + dims[None, :] * SIN_STRIDE_D
        )
        cos = tl.load(cos_ptr + cos_offsets, mask=mask, other=0.0).to(
            deployed_dtype
        )
        direct_grad = (grad * cos).to(deployed_dtype)

        # rotate_half's transpose maps the gradient from the opposite half
        # back with the complementary sign.  Load that partner directly; no
        # cat/slice temporary is formed.
        partner_dims = tl.where(dims < half, dims + half, dims - half)
        partner_grad_offsets = (
            batch[:, None] * G_STRIDE_B
            + head[:, None] * G_STRIDE_H
            + seq[:, None] * G_STRIDE_S
            + partner_dims[None, :] * G_STRIDE_D
        )
        partner_sin_offsets = (
            sin_batch[:, None] * SIN_STRIDE_B
            + seq[:, None] * SIN_STRIDE_S
            + partner_dims[None, :] * SIN_STRIDE_D
        )
        partner_grad = tl.load(
            grad_ptr + partner_grad_offsets, mask=mask, other=0.0
        ).to(deployed_dtype)
        partner_sin = tl.load(
            sin_ptr + partner_sin_offsets, mask=mask, other=0.0
        ).to(deployed_dtype)
        rotated_grad = (partner_grad * partner_sin).to(deployed_dtype)
        rotated_grad = tl.where(
            dims < half, rotated_grad, -rotated_grad
        ).to(deployed_dtype)
        grad_affine = (direct_grad + rotated_grad).to(deployed_dtype)

        weight = tl.load(
            weight_ptr + dims[None, :],
            mask=dims[None, :] < HEAD_DIM,
            other=0.0,
        ).to(
            deployed_dtype
        )
        grad_normalised = (grad_affine * weight).to(deployed_dtype).to(
            tl.float32
        )
        x_offsets = (
            batch[:, None] * X_STRIDE_B
            + head[:, None] * X_STRIDE_H
            + seq[:, None] * X_STRIDE_S
            + dims[None, :] * X_STRIDE_D
        )
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        inv_rms = tl.load(
            inv_rms_ptr + rows, mask=row_mask, other=0.0
        ).to(tl.float32)
        dot = tl.sum(grad_normalised * x, axis=1)
        grad_x = inv_rms[:, None] * grad_normalised - (
            x
            * (inv_rms[:, None] * inv_rms[:, None] * inv_rms[:, None] / HEAD_DIM)
            * dot[:, None]
        )
        tl.store(
            grad_x_ptr + rows[:, None] * HEAD_DIM + dims[None, :],
            grad_x.to(deployed_dtype),
            mask=mask,
        )


    @triton.jit
    def _swiglu_forward_kernel(
        gate_ptr,
        up_ptr,
        out_ptr,
        N_ELEMENTS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N_ELEMENTS
        deployed_dtype = gate_ptr.dtype.element_ty
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(
            deployed_dtype
        )
        # Eager Qwen3 first stores SiLU in the deployed dtype, then performs
        # the gate/up multiplication in that dtype.
        silu = (gate * tl.sigmoid(gate)).to(deployed_dtype)
        output = (silu * up).to(deployed_dtype)
        tl.store(out_ptr + offsets, output, mask=mask)


    @triton.jit
    def _swiglu_backward_kernel(
        grad_ptr,
        gate_ptr,
        up_ptr,
        grad_gate_ptr,
        grad_up_ptr,
        N_ELEMENTS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N_ELEMENTS
        deployed_dtype = gate_ptr.dtype.element_ty
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(
            deployed_dtype
        )
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(
            deployed_dtype
        )
        sigmoid = tl.sigmoid(gate)
        silu = (gate * sigmoid).to(deployed_dtype)
        grad_up = (grad * silu).to(deployed_dtype)
        grad_silu = (grad * up).to(deployed_dtype).to(tl.float32)
        silu_derivative = sigmoid * (1.0 + gate * (1.0 - sigmoid))
        grad_gate = (grad_silu * silu_derivative).to(deployed_dtype)
        tl.store(grad_gate_ptr + offsets, grad_gate, mask=mask)
        tl.store(grad_up_ptr + offsets, grad_up, mask=mask)


def is_available() -> bool:
    return triton is not None


def _normalise_position_tensor(
    tensor: torch.Tensor, batch: int, seq_len: int, head_dim: int
) -> torch.Tensor | None:
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if (
        tensor.ndim != 3
        or tensor.shape[0] not in (1, batch)
        or tensor.shape[1] != seq_len
        or tensor.shape[2] != head_dim
        or tensor.stride(-1) != 1
    ):
        return None
    return tensor


def can_fuse_qk_rmsnorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 1,
) -> bool:
    if triton is None or unsqueeze_dim != 1:
        return False
    if q.ndim != 4 or k.ndim != 4 or q.shape[0] != k.shape[0]:
        return False
    if q.shape[2:] != k.shape[2:] or q.shape[-1] % 2 != 0:
        return False
    if q.shape[-1] <= 0 or q.shape[-1] > 1024:
        return False
    if not (q.is_cuda and k.is_cuda and q.device == k.device):
        return False
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype:
        return False
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        return False
    if (
        q_weight.shape != (q.shape[-1],)
        or k_weight.shape != (q.shape[-1],)
        or q_weight.device != q.device
        or k_weight.device != q.device
        or q_weight.dtype != q.dtype
        or k_weight.dtype != q.dtype
        or not q_weight.is_contiguous()
        or not k_weight.is_contiguous()
    ):
        return False
    cos_view = _normalise_position_tensor(
        cos, q.shape[0], q.shape[2], q.shape[3]
    )
    sin_view = _normalise_position_tensor(
        sin, q.shape[0], q.shape[2], q.shape[3]
    )
    return bool(
        cos_view is not None
        and sin_view is not None
        and cos_view.device == q.device
        and sin_view.device == q.device
        and cos_view.dtype == q.dtype
        and sin_view.dtype == q.dtype
    )


def _launch_qk_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, seq_len, head_dim = x.shape
    rows = batch * heads * seq_len
    output = torch.empty(
        x.shape, dtype=x.dtype, device=x.device, memory_format=torch.contiguous_format
    )
    inv_rms = torch.empty(rows, dtype=torch.float32, device=x.device)
    block_d = triton.next_power_of_2(head_dim)
    # A Qwen3 row has only 128 values.  Group independent rows into one CTA
    # to amortise scheduling overhead while keeping all intermediates local.
    block_rows = 16 if block_d <= 128 else 8 if block_d <= 256 else 2
    grid = (triton.cdiv(rows, block_rows),)
    _qk_rmsnorm_rope_forward_kernel[grid](
        x,
        weight,
        cos,
        sin,
        output,
        inv_rms,
        X_STRIDE_B=x.stride(0),
        X_STRIDE_H=x.stride(1),
        X_STRIDE_S=x.stride(2),
        X_STRIDE_D=x.stride(3),
        COS_STRIDE_B=cos.stride(0),
        COS_STRIDE_S=cos.stride(1),
        COS_STRIDE_D=cos.stride(2),
        SIN_STRIDE_B=sin.stride(0),
        SIN_STRIDE_S=sin.stride(1),
        SIN_STRIDE_D=sin.stride(2),
        N_HEADS=heads,
        SEQ_LEN=seq_len,
        HEAD_DIM=head_dim,
        COS_BATCH=cos.shape[0],
        SIN_BATCH=sin.shape[0],
        EPS=float(eps),
        N_ROWS=rows,
        BLOCK_ROWS=block_rows,
        BLOCK_D=block_d,
        num_warps=8,
    )
    return output, inv_rms


def _launch_qk_backward(
    grad: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    inv_rms: torch.Tensor,
) -> torch.Tensor:
    batch, heads, seq_len, head_dim = x.shape
    rows = batch * heads * seq_len
    output = torch.empty(
        x.shape, dtype=x.dtype, device=x.device, memory_format=torch.contiguous_format
    )
    block_d = triton.next_power_of_2(head_dim)
    block_rows = 16 if block_d <= 128 else 8 if block_d <= 256 else 2
    grid = (triton.cdiv(rows, block_rows),)
    _qk_rmsnorm_rope_backward_kernel[grid](
        grad,
        x,
        weight,
        cos,
        sin,
        inv_rms,
        output,
        G_STRIDE_B=grad.stride(0),
        G_STRIDE_H=grad.stride(1),
        G_STRIDE_S=grad.stride(2),
        G_STRIDE_D=grad.stride(3),
        X_STRIDE_B=x.stride(0),
        X_STRIDE_H=x.stride(1),
        X_STRIDE_S=x.stride(2),
        X_STRIDE_D=x.stride(3),
        COS_STRIDE_B=cos.stride(0),
        COS_STRIDE_S=cos.stride(1),
        COS_STRIDE_D=cos.stride(2),
        SIN_STRIDE_B=sin.stride(0),
        SIN_STRIDE_S=sin.stride(1),
        SIN_STRIDE_D=sin.stride(2),
        N_HEADS=heads,
        SEQ_LEN=seq_len,
        HEAD_DIM=head_dim,
        COS_BATCH=cos.shape[0],
        SIN_BATCH=sin.shape[0],
        N_ROWS=rows,
        BLOCK_ROWS=block_rows,
        BLOCK_D=block_d,
        num_warps=8,
    )
    return output


class _FusedQKRMSNormRoPE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        q_eps: float,
        k_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = _normalise_position_tensor(cos, q.shape[0], q.shape[2], q.shape[3])
        sin = _normalise_position_tensor(sin, q.shape[0], q.shape[2], q.shape[3])
        q_out, q_inv = _launch_qk_forward(q, q_weight, cos, sin, q_eps)
        k_out, k_inv = _launch_qk_forward(k, k_weight, cos, sin, k_eps)
        ctx.save_for_backward(
            q, k, q_weight, k_weight, cos, sin, q_inv, k_inv
        )
        return q_out, k_out

    @staticmethod
    def backward(ctx, grad_q: torch.Tensor, grad_k: torch.Tensor):
        q, k, q_weight, k_weight, cos, sin, q_inv, k_inv = ctx.saved_tensors
        grad_q_input = _launch_qk_backward(
            grad_q, q, q_weight, cos, sin, q_inv
        )
        grad_k_input = _launch_qk_backward(
            grad_k, k, k_weight, cos, sin, k_inv
        )
        # Q/K RMSNorm parameters are fixed in REAL-Q PTQ.  They are frozen
        # when this fusion is installed, so no weight-gradient reduction is
        # needed here.
        return grad_q_input, grad_k_input, None, None, None, None, None, None


def fused_qk_rmsnorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    q_eps: float,
    k_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not can_fuse_qk_rmsnorm_rope(
        q, k, q_weight, k_weight, cos, sin
    ):
        raise ValueError("Q/K RMSNorm+RoPE fusion preconditions are not met.")
    return _FusedQKRMSNormRoPE.apply(
        q, k, q_weight, k_weight, cos, sin, float(q_eps), float(k_eps)
    )


def can_fuse_swiglu(gate: torch.Tensor, up: torch.Tensor) -> bool:
    return bool(
        triton is not None
        and gate.is_cuda
        and up.is_cuda
        and gate.device == up.device
        and gate.dtype == up.dtype
        and gate.dtype in (torch.float16, torch.bfloat16)
        and gate.shape == up.shape
        and gate.is_contiguous()
        and up.is_contiguous()
    )


class _FusedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(gate)
        block = 1024
        _swiglu_forward_kernel[(triton.cdiv(gate.numel(), block),)](
            gate,
            up,
            output,
            N_ELEMENTS=gate.numel(),
            BLOCK=block,
            num_warps=8,
        )
        ctx.save_for_backward(gate, up)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gate, up = ctx.saved_tensors
        grad_gate = torch.empty_like(gate)
        grad_up = torch.empty_like(up)
        block = 1024
        _swiglu_backward_kernel[(triton.cdiv(gate.numel(), block),)](
            grad_output,
            gate,
            up,
            grad_gate,
            grad_up,
            N_ELEMENTS=gate.numel(),
            BLOCK=block,
            num_warps=8,
        )
        return grad_gate, grad_up


def fused_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Use the fused CUDA path and retain exact eager behavior elsewhere."""

    if can_fuse_swiglu(gate, up):
        return _FusedSwiGLU.apply(gate, up)
    return F.silu(gate) * up


def _qwen3_mlp_fused_forward(module, x: torch.Tensor) -> torch.Tensor:
    gate = module.gate_proj(x)
    up = module.up_proj(x)
    return module.down_proj(fused_swiglu(gate, up))


def install_qwen3_swiglu_fusion(model: torch.nn.Module) -> int:
    """Patch dense Qwen3 MLP instances without changing state_dict paths."""

    count = 0
    for module in model.modules():
        if type(module).__name__ != "Qwen3MLP":
            continue
        if bool(getattr(module, "_realq_fused_swiglu", False)):
            continue
        if not all(
            hasattr(module, name)
            for name in ("gate_proj", "up_proj", "down_proj")
        ):
            continue
        config = getattr(module, "config", None)
        if config is not None and getattr(config, "hidden_act", "silu") != "silu":
            # The fused derivative below is specifically SiLU.  Avoid changing
            # semantics for a custom Qwen3 checkpoint that selects another
            # activation through the generic Transformers config.
            continue
        module.forward = types.MethodType(_qwen3_mlp_fused_forward, module)
        module._realq_fused_swiglu = True
        count += 1
    return count
