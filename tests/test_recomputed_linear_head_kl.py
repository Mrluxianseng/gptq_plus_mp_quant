import pytest
import torch
import torch.nn.functional as F

from utils.loss_utils import (
    chunked_tokenwise_kl_from_logits,
    recomputed_linear_head_tokenwise_kl,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk_tokens", [1, 3, 16])
@pytest.mark.parametrize("topk", [-1, 7])
def test_recomputed_head_kl_matches_materialized_value_and_gradient(
    dtype, chunk_tokens, topk
):
    torch.manual_seed(123)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows, hidden, vocab = 9, 13, 31
    weight = torch.randn(vocab, hidden, dtype=dtype, device=device) / hidden**0.5
    bias = torch.randn(vocab, dtype=dtype, device=device) * 0.01
    student = torch.randn(rows, hidden, dtype=dtype, device=device).requires_grad_(True)
    teacher = torch.randn(rows, hidden, dtype=dtype, device=device)

    ref_student = F.linear(student, weight, bias)
    ref_teacher = F.linear(teacher, weight, bias)
    ref_loss = chunked_tokenwise_kl_from_logits(
        ref_student, ref_teacher, chunk_tokens=chunk_tokens, topk=topk
    )
    ref_grad = torch.autograd.grad(ref_loss, student)[0]

    test_student = student.detach().clone().requires_grad_(True)
    test_loss = recomputed_linear_head_tokenwise_kl(
        test_student,
        teacher,
        weight,
        bias,
        chunk_tokens=chunk_tokens,
        topk=topk,
    )
    test_grad = torch.autograd.grad(test_loss, test_student)[0]

    assert torch.allclose(test_loss, ref_loss, rtol=2e-5, atol=2e-6)
    assert torch.allclose(test_grad, ref_grad, rtol=3e-5, atol=3e-6)


def test_recomputed_head_kl_propagates_only_student_input_gradient():
    student = torch.randn(4, 5, requires_grad=True)
    teacher = torch.randn(4, 5, requires_grad=True)
    weight = torch.randn(11, 5, requires_grad=True)
    loss = recomputed_linear_head_tokenwise_kl(
        student, teacher, weight, chunk_tokens=2
    )
    grad_student, grad_teacher, grad_weight = torch.autograd.grad(
        loss,
        (student, teacher, weight),
        allow_unused=True,
    )
    assert grad_student is not None
    assert grad_teacher is None
    assert grad_weight is None

