import pytest
import torch

from gptq_utils.triton_gptq_kernels import fused_gptq_column_


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_fused_column_loop_matches_torch_bit_exactly(noncontiguous):
    torch.manual_seed(1701)
    device = "cuda"
    rows, cols = 64, 128
    if noncontiguous:
        W = torch.randn(rows, cols + 7, device=device, dtype=torch.float32)[:, :cols]
        GH = torch.randn(rows, cols + 11, device=device, dtype=torch.float32)[:, :cols]
    else:
        W = torch.randn(rows, cols, device=device, dtype=torch.float32)
        GH = torch.randn(rows, cols, device=device, dtype=torch.float32)
    Z = torch.randn(rows, cols, device=device, dtype=torch.float32)
    A = torch.randn(cols, cols, device=device, dtype=torch.float32)
    H = A.T @ A / cols + torch.eye(cols, device=device)
    scale = torch.rand(rows, device=device, dtype=torch.float32) * 0.02 + 0.005
    maxq, qlo, second_order_scale = 7, -8, 0.75

    W_ref = torch.empty_strided(W.shape, W.stride(), device=device, dtype=W.dtype)
    GH_ref = torch.empty_strided(GH.shape, GH.stride(), device=device, dtype=GH.dtype)
    W_ref.copy_(W)
    GH_ref.copy_(GH)
    # The production block allocates these output workspaces contiguously even
    # when W/GH are views with different row strides.
    Q_ref = torch.zeros((rows, cols), device=device, dtype=torch.float32)
    W_int_ref = torch.zeros_like(Q_ref)
    Err_ref = torch.zeros_like(Q_ref)
    for col in range(cols):
        w = W_ref[:, col]
        int_weight = torch.clamp(torch.round(w / scale), qlo, maxq)
        q = (scale * int_weight).to(w.dtype)
        Q_ref[:, col] = q
        W_int_ref[:, col] = int_weight
        err = (w - q - GH_ref[:, col]) / H[col, col]
        Err_ref[:, col] = err
        update = err.unsqueeze(1).matmul(H[col, col:].unsqueeze(0))
        W_ref[:, col:] -= second_order_scale * (update + GH_ref[:, col:])
        GH_ref[:, col:].sub_(Z[:, col].unsqueeze(1).matmul(H[col, col:].unsqueeze(0)))

    W_test = torch.empty_strided(W.shape, W.stride(), device=device, dtype=W.dtype)
    GH_test = torch.empty_strided(GH.shape, GH.stride(), device=device, dtype=GH.dtype)
    W_test.copy_(W)
    GH_test.copy_(GH)
    Q_test = torch.zeros((rows, cols), device=device, dtype=torch.float32)
    W_int_test = torch.zeros_like(Q_test)
    Err_test = torch.zeros_like(Q_test)
    for col in range(cols):
        fused_gptq_column_(
            W_test,
            GH_test,
            Z,
            H,
            scale,
            Q_test,
            W_int_test,
            Err_test,
            col,
            maxq=maxq,
            qlo=qlo,
            second_order_scale=second_order_scale,
        )

    torch.cuda.synchronize()
    for actual, expected in (
        (W_test, W_ref),
        (GH_test, GH_ref),
        (Q_test, Q_ref),
        (W_int_test, W_int_ref),
        (Err_test, Err_ref),
    ):
        assert torch.equal(actual, expected)
