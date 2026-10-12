"""Numerically stable shared loss primitives."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def tokenwise_kl_from_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    """Return ``KL(teacher || student)`` per token in fp32.

    Casting before log-softmax is essential for close bf16 distributions.  The
    explicit ``p * (log p - log q)`` form also avoids the opaque per-element
    cancellation behavior of ``F.kl_div`` and leaves reduction policy to the
    caller.
    """
    log_student = F.log_softmax(student_logits.float(), dim=-1)
    log_teacher = F.log_softmax(teacher_logits.float(), dim=-1)
    return (log_teacher.exp() * (log_teacher - log_student)).sum(dim=-1)


class _ChunkedTokenwiseKL(torch.autograd.Function):
    """Exact tokenwise KL with token-chunked FP32 intermediates.

    The input logits remain live for the caller's linear-layer backward, but
    log-softmax/probability temporaries are limited to ``chunk_tokens`` rows.
    Teacher logits are treated as constants, matching the refresh path.
    """

    @staticmethod
    def forward(ctx, student_logits, teacher_logits, chunk_tokens, topk):
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                "student/teacher logits must have matching shapes, got "
                f"{tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}"
            )
        if student_logits.dim() < 2:
            raise ValueError("logits must have at least token and vocabulary dimensions")

        vocab = int(student_logits.shape[-1])
        rows = int(student_logits.numel() // vocab)
        chunk_tokens = max(1, int(chunk_tokens))
        topk = int(topk)
        student_flat = student_logits.reshape(rows, vocab)
        teacher_flat = teacher_logits.reshape(rows, vocab)
        loss_sum = torch.zeros((), dtype=torch.float32, device=student_logits.device)
        for start in range(0, rows, chunk_tokens):
            end = min(start + chunk_tokens, rows)
            student_chunk = student_flat[start:end].float()
            teacher_chunk = teacher_flat[start:end].float()
            if topk > 0:
                teacher_chunk, indices = teacher_chunk.topk(
                    topk, dim=-1, sorted=False
                )
                student_chunk = student_chunk.gather(-1, indices)
            loss_sum = loss_sum + tokenwise_kl_from_logits(
                student_chunk, teacher_chunk
            ).sum()

        ctx.save_for_backward(student_logits, teacher_logits)
        ctx.chunk_tokens = chunk_tokens
        ctx.topk = topk
        ctx.rows = rows
        ctx.vocab = vocab
        return loss_sum / max(rows, 1)

    @staticmethod
    def backward(ctx, grad_output):
        student_logits, teacher_logits = ctx.saved_tensors
        student_flat = student_logits.reshape(ctx.rows, ctx.vocab)
        teacher_flat = teacher_logits.reshape(ctx.rows, ctx.vocab)
        grad_flat = torch.empty_like(student_flat)

        for start in range(0, ctx.rows, ctx.chunk_tokens):
            end = min(start + ctx.chunk_tokens, ctx.rows)
            with torch.enable_grad():
                student_leaf = student_flat[start:end].detach().requires_grad_(True)
                student_chunk = student_leaf
                teacher_chunk = teacher_flat[start:end].detach()
                if ctx.topk > 0:
                    teacher_chunk, indices = teacher_chunk.float().topk(
                        ctx.topk, dim=-1, sorted=False
                    )
                    student_chunk = student_chunk.float().gather(-1, indices)
                chunk_loss = tokenwise_kl_from_logits(
                    student_chunk, teacher_chunk
                ).sum() / max(ctx.rows, 1)
                chunk_grad = torch.autograd.grad(
                    chunk_loss, student_leaf, grad_outputs=grad_output,
                )[0]
            grad_flat[start:end].copy_(chunk_grad)

        return grad_flat.reshape_as(student_logits), None, None, None


def chunked_tokenwise_kl_from_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    chunk_tokens: int = 16,
    topk: int = -1,
) -> torch.Tensor:
    """Mean ``KL(teacher || student)`` with bounded FP32 logit temporaries.

    Chunking is over flattened token rows (all leading dimensions); the result
    and student-logit gradient match ``tokenwise_kl_from_logits(...).mean()``
    up to FP32 reduction-order differences. Teacher logits receive no gradient.
    """
    return _ChunkedTokenwiseKL.apply(
        student_logits, teacher_logits, int(chunk_tokens), int(topk)
    )


class _RecomputedLinearHeadKL(torch.autograd.Function):
    """Token-chunked KL for a shared linear vocabulary head.

    Unlike :class:`_ChunkedTokenwiseKL`, this function takes the head inputs
    and head parameters, not materialized ``[batch, sequence, vocab]`` logits.
    It computes and discards one token chunk of logits in the forward pass,
    then recomputes the same chunk in backward.  The teacher branch and head
    parameters are constants, matching REAL-Q's refresh path where only the
    current quantized block receives gradients.

    It deliberately keeps the entire vocabulary in each chunk, so it preserves
    the existing full-vocabulary KL support and reduction instead of
    approximating the distribution.  In backward it assembles one full
    ``d_logits`` workspace and performs one full-row linear backward GEMM;
    this matches the original BF16 gradient numerics.  Thus it avoids retaining
    both full student and teacher logits plus their autograd graph, but does
    not eliminate every ``O(tokens * vocab)`` object.  Chunk-local linear
    backward GEMMs use less memory but can change the update trajectory.
    """

    @staticmethod
    def forward(ctx, student_hidden, teacher_hidden, weight, bias, chunk_tokens, topk):
        if student_hidden.shape != teacher_hidden.shape:
            raise ValueError("student and teacher head inputs must have matching shapes")
        if student_hidden.shape[-1] != weight.shape[-1]:
            raise ValueError(
                f"head input width {student_hidden.shape[-1]} does not match "
                f"weight width {weight.shape[-1]}"
            )
        if weight.dim() != 2:
            raise ValueError(f"expected a 2D linear head weight, got {tuple(weight.shape)}")
        if bias is not None and bias.numel() != weight.shape[0]:
            raise ValueError("head bias width does not match vocabulary width")

        rows = int(student_hidden.numel() // student_hidden.shape[-1])
        hidden = int(student_hidden.shape[-1])
        vocab = int(weight.shape[0])
        chunk_tokens = max(1, int(chunk_tokens))
        topk = int(topk)
        student_flat = student_hidden.reshape(rows, hidden)
        teacher_flat = teacher_hidden.reshape(rows, hidden)
        loss_sum = torch.zeros((), dtype=torch.float32, device=student_hidden.device)
        frozen_weight = weight.detach()
        frozen_bias = bias.detach() if bias is not None else None
        for start in range(0, rows, chunk_tokens):
            end = min(start + chunk_tokens, rows)
            student_logits = F.linear(student_flat[start:end], frozen_weight, frozen_bias)
            teacher_logits = F.linear(teacher_flat[start:end], frozen_weight, frozen_bias)
            student_logits = student_logits.float()
            teacher_logits = teacher_logits.float()
            if topk > 0:
                teacher_logits, indices = teacher_logits.topk(
                    topk, dim=-1, sorted=False
                )
                student_logits = student_logits.gather(-1, indices)
            loss_sum = loss_sum + tokenwise_kl_from_logits(
                student_logits, teacher_logits
            ).sum()

        ctx.has_bias = bias is not None
        if bias is None:
            ctx.save_for_backward(student_hidden, teacher_hidden, weight)
        else:
            ctx.save_for_backward(student_hidden, teacher_hidden, weight, bias)
        ctx.rows = rows
        ctx.hidden = hidden
        ctx.vocab = vocab
        ctx.chunk_tokens = chunk_tokens
        ctx.topk = topk
        return loss_sum / max(rows, 1)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.has_bias:
            student_hidden, teacher_hidden, weight, bias = ctx.saved_tensors
        else:
            student_hidden, teacher_hidden, weight = ctx.saved_tensors
            bias = None
        rows, hidden = ctx.rows, ctx.hidden
        student_flat = student_hidden.reshape(rows, hidden)
        teacher_flat = teacher_hidden.reshape(rows, hidden)
        # Match nn.Linear's normal backward data path: first produce the
        # output-gradient in the logits dtype for every row, then issue one
        # full-row matrix multiply. Doing the linear backward independently
        # inside each row chunk changes GEMM tiling and can perturb REAL-Q's
        # Adam/sign-sensitive quantization trajectory.
        grad_logits_flat = torch.empty(
            (rows, ctx.vocab), dtype=weight.dtype, device=weight.device
        )
        frozen_weight = weight.detach()
        frozen_bias = bias.detach() if bias is not None else None

        for start in range(0, rows, ctx.chunk_tokens):
            end = min(start + ctx.chunk_tokens, rows)
            with torch.enable_grad():
                student_rows = student_flat[start:end]
                with torch.no_grad():
                    teacher_logits = F.linear(
                        teacher_flat[start:end], frozen_weight, frozen_bias
                    ).float()
                    student_logits = F.linear(
                        student_rows, frozen_weight, frozen_bias
                    ).detach()
                student_logits_leaf = student_logits.requires_grad_(True)
                if ctx.topk > 0:
                    teacher_logits, indices = teacher_logits.topk(
                        ctx.topk, dim=-1, sorted=False
                    )
                    student_logits_for_loss = student_logits_leaf.gather(-1, indices)
                else:
                    student_logits_for_loss = student_logits_leaf
                chunk_loss = tokenwise_kl_from_logits(
                    student_logits_for_loss.float(), teacher_logits
                ).sum() / max(rows, 1)
                chunk_grad = torch.autograd.grad(
                    chunk_loss, student_logits_leaf, grad_outputs=grad_output
                )[0]
            grad_logits_flat[start:end].copy_(chunk_grad)

        grad_flat = grad_logits_flat.matmul(frozen_weight)

        return grad_flat.reshape_as(student_hidden), None, None, None, None, None


def recomputed_linear_head_tokenwise_kl(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    chunk_tokens: int = 16,
    topk: int = -1,
) -> torch.Tensor:
    """Mean ``KL(teacher || student)`` without retaining full head logits.

    ``student_hidden`` and ``teacher_hidden`` must be the outputs of the exact
    pre-head transform (REAL-Q's final normalization).  Only the student input
    receives a gradient. The vocabulary remains complete unless ``topk > 0``.
    The backward retains one full logits-gradient workspace to preserve the
    reference single-GEMM gradient path; use a fused custom kernel to remove
    that last ``tokens * vocab`` workspace without changing the arithmetic.
    """
    return _RecomputedLinearHeadKL.apply(
        student_hidden,
        teacher_hidden,
        weight,
        bias,
        int(chunk_tokens),
        int(topk),
    )
