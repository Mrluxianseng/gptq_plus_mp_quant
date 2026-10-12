"""Research-only KL storage policy; retains the original head GEMM shapes.

The original caller still materializes logits. Only the two logits saved by
the custom KL node move to CPU. Backward restores individual token slices,
assembles the same full logits-gradient and leaves the outer linear backward
unchanged. No vocabulary truncation or new mathematical loss is introduced.
"""
import torch
from utils.loss_utils import tokenwise_kl_from_logits


class _CpuSavedLogitsKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student_logits, teacher_logits, chunk_tokens, topk):
        if student_logits.shape != teacher_logits.shape or student_logits.dim() < 2:
            raise ValueError("matching token/vocabulary logits are required")
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
                teacher_chunk, indices = teacher_chunk.topk(topk, dim=-1, sorted=False)
                student_chunk = student_chunk.gather(-1, indices)
            loss_sum = loss_sum + tokenwise_kl_from_logits(student_chunk, teacher_chunk).sum()
        ctx.shape, ctx.device, ctx.dtype = tuple(student_logits.shape), student_logits.device, student_logits.dtype
        ctx.chunk_tokens, ctx.topk, ctx.rows, ctx.vocab = chunk_tokens, topk, rows, vocab
        # copy=True also avoids CPU test inputs aliasing saved data. Blocking
        # transfers make completion explicit; async/pinned overlap is unproven.
        ctx.save_for_backward(student_logits.detach().to("cpu", copy=True),
                              teacher_logits.detach().to("cpu", copy=True))
        return loss_sum / max(rows, 1)

    @staticmethod
    def backward(ctx, grad_output):
        student_cpu, teacher_cpu = ctx.saved_tensors
        student_flat = student_cpu.reshape(ctx.rows, ctx.vocab)
        teacher_flat = teacher_cpu.reshape(ctx.rows, ctx.vocab)
        grad_flat = torch.empty((ctx.rows, ctx.vocab), dtype=ctx.dtype, device=ctx.device)
        for start in range(0, ctx.rows, ctx.chunk_tokens):
            end = min(start + ctx.chunk_tokens, ctx.rows)
            with torch.enable_grad():
                student_leaf = student_flat[start:end].to(ctx.device).detach().requires_grad_(True)
                student_chunk = student_leaf
                teacher_chunk = teacher_flat[start:end].to(ctx.device).detach()
                if ctx.topk > 0:
                    teacher_chunk, indices = teacher_chunk.float().topk(ctx.topk, dim=-1, sorted=False)
                    student_chunk = student_chunk.float().gather(-1, indices)
                chunk_loss = tokenwise_kl_from_logits(student_chunk, teacher_chunk).sum() / max(ctx.rows, 1)
                chunk_grad = torch.autograd.grad(chunk_loss, student_leaf, grad_outputs=grad_output)[0]
            grad_flat[start:end].copy_(chunk_grad)
        return grad_flat.reshape(ctx.shape), None, None, None


def cpu_saved_logits_kl(student_logits, teacher_logits, chunk_tokens=16, topk=-1):
    return _CpuSavedLogitsKL.apply(student_logits, teacher_logits, int(chunk_tokens), int(topk))
