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
