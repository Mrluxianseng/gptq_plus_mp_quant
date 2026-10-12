"""Correctness smoke test for a fused GPTQ inner-column update.

This is deliberately an isolated prototype. It models the existing
`block_gd` frozen-inner loop with per-row symmetric quantization and does not
alter the production quantizer path.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import statistics
import time

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gptq_utils.triton_gptq_kernels import fused_gptq_column_


@torch.no_grad()
def fused_loop(W, GH, Z, H, scale, *, maxq, qlo, second_order_scale, capture_snapshots=True):
    rows, cols = W.shape
    q = torch.empty_like(W)
    w_int = torch.empty_like(W)
    err = torch.empty_like(W)
    snapshots = []
    gh_columns = []
    gh_last_columns = []
    for col in range(cols):
        gh_columns.append(GH[:, col].clone())
        gh_last_columns.append(GH[:, -1].clone())
        fused_gptq_column_(
            W,
            GH,
            Z,
            H,
            scale,
            q,
            w_int,
            err,
            col,
            maxq=maxq,
            qlo=qlo,
            second_order_scale=second_order_scale,
        )
        if capture_snapshots:
            snapshots.append(W.clone())
    return q, w_int, err, snapshots, gh_columns, gh_last_columns


@torch.no_grad()
def reference_loop(W, GH, Z, H, scale, *, maxq, qlo, second_order_scale, capture_snapshots=True):
    rows, cols = W.shape
    q_out = torch.empty_like(W)
    w_int_out = torch.empty_like(W)
    err_out = torch.empty_like(W)
    snapshots = []
    gh_columns = []
    gh_last_columns = []
    for col in range(cols):
        gh_columns.append(GH[:, col].clone())
        gh_last_columns.append(GH[:, -1].clone())
        w = W[:, col]
        int_weight = torch.clamp(torch.round(w.unsqueeze(1) / scale), qlo, maxq)
        q = (scale * int_weight).to(w.dtype).flatten()
        err = (w - q - GH[:, col]) / H[col, col]
        update = err.unsqueeze(1).matmul(H[col, col:].unsqueeze(0))
        W[:, col:] -= second_order_scale * (update + GH[:, col:])
        GH[:, col:].sub_(Z[:, col].unsqueeze(1).matmul(H[col, col:].unsqueeze(0)))
        q_out[:, col] = q
        w_int_out[:, col] = int_weight.flatten()
        err_out[:, col] = err
        if capture_snapshots:
            snapshots.append(W.clone())
    return q_out, w_int_out, err_out, snapshots, gh_columns, gh_last_columns


def max_abs(a, b):
    return float((a.float() - b.float()).abs().max().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--cols", type=int, default=128)
    parser.add_argument(
        "--parent-cols", type=int, default=None,
        help="Backing-matrix column stride; larger than cols reproduces REAL-Q block views.",
    )
    parser.add_argument("--seed", type=int, default=173)
    parser.add_argument("--second-order-scale", type=float, default=1.0)
    parser.add_argument("--bench-iters", type=int, default=3)
    args = parser.parse_args()
    if args.rows <= 0 or args.cols <= 0 or args.cols > 128:
        parser.error("rows must be positive and cols must be in [1, 128]")
    if args.bench_iters < 0:
        parser.error("bench-iters must be non-negative")
    parent_cols = args.cols if args.parent_cols is None else args.parent_cols
    if parent_cols < args.cols:
        parser.error("parent-cols must be at least cols")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton smoke test")

    torch.manual_seed(args.seed)
    device = "cuda"
    W_storage = (torch.randn(args.rows, parent_cols, device=device, dtype=torch.float32) * 0.15).contiguous()
    GH_storage = (torch.randn_like(W_storage) * 0.002).contiguous()
    Z_storage = torch.randn_like(W_storage).contiguous()
    W0 = W_storage[:, :args.cols]
    GH0 = GH_storage[:, :args.cols]
    Z = Z_storage[:, :args.cols]
    raw = torch.randn(args.cols, args.cols, device=device, dtype=torch.float32)
    hessian = raw @ raw.T / args.cols + torch.eye(args.cols, device=device) * 0.25
    hessian_inv = torch.cholesky_inverse(torch.linalg.cholesky(hessian))
    # REAL-Q's `Hinv` is the upper Cholesky factor of the inverse Hessian,
    # rather than the raw Hessian or a dense inverse.
    H = torch.linalg.cholesky(hessian_inv, upper=True).contiguous()
    # Row-wise symmetric int4 scale; keep exactly the same per-row values for
    # both paths instead of mixing parameter-search quality into this test.
    scale = (W0.abs().amax(dim=1, keepdim=True) / 7.0).clamp_min(1e-6).contiguous()
    maxq, qlo = 7, -8
    q_probe = torch.clamp(torch.round(W0[:, :1] / scale), qlo, maxq) * scale
    err_probe = (W0[:, 0] - q_probe.flatten() - GH0[:, 0]) / H[0, 0]
    outer_matmul_probe = err_probe.unsqueeze(1).matmul(H[0, :].unsqueeze(0))
    outer_mul_probe = err_probe.unsqueeze(1) * H[0, :].unsqueeze(0)

    Wr_storage, GHr_storage = W_storage.clone(), GH_storage.clone()
    Wr, GHr = Wr_storage[:, :args.cols], GHr_storage[:, :args.cols]
    Qr, IR, Er, ref_snapshots, ref_gh_columns, ref_gh_last_columns = reference_loop(
        Wr, GHr, Z, H, scale, maxq=maxq, qlo=qlo,
        second_order_scale=args.second_order_scale,
    )
    Wt_storage, GHt_storage = W_storage.clone(), GH_storage.clone()
    Wt, GHt = Wt_storage[:, :args.cols], GHt_storage[:, :args.cols]
    Qt, IT, Et, triton_snapshots, triton_gh_columns, triton_gh_last_columns = fused_loop(
        Wt, GHt, Z, H, scale, maxq=maxq, qlo=qlo,
        second_order_scale=args.second_order_scale,
    )
    torch.cuda.synchronize()

    q_int_mismatches = int((IR != IT).sum().item())
    q_mismatch = torch.nonzero(IR != IT)
    err_delta = (Er.float() - Et.float()).abs()
    err_flat_index = int(err_delta.argmax().item())
    err_row, err_col = divmod(err_flat_index, args.cols)
    probe_col = min(39, args.cols - 1)
    probe_outer = Er[:, probe_col : probe_col + 1] @ H[probe_col, probe_col:][None, :]
    probe_mul = Er[:, probe_col : probe_col + 1] * H[probe_col, probe_col:][None, :]
    probe_delta = (probe_outer - probe_mul).abs()
    probe_flat = int(probe_delta.argmax().item())
    probe_row, probe_offset = divmod(probe_flat, probe_delta.shape[1])
    step_errors = torch.tensor(
        [max_abs(ref, tri) for ref, tri in zip(ref_snapshots, triton_snapshots)],
        dtype=torch.float64,
    )
    worst_col = int(step_errors.argmax().item())
    first_diff = None
    for step, (ref_w, tri_w) in enumerate(zip(ref_snapshots, triton_snapshots)):
        diff = (ref_w - tri_w).abs()
        if diff.max() > 0:
            flat = int(diff.argmax().item())
            row, col = divmod(flat, args.cols)
            first_diff = {
                "step": step,
                "row": row,
                "col": col,
                "ref": float(ref_w[row, col].item()),
                "triton": float(tri_w[row, col].item()),
                "max_abs": float(diff[row, col].item()),
                "step_w_before_ref": float(
                    (Wr[:, :] if step == 0 else ref_snapshots[step - 1])[row, step].item()
                ),
                "step_w_before_triton": float(
                    (Wt[:, :] if step == 0 else triton_snapshots[step - 1])[row, step].item()
                ),
                "step_q_ref": float(Qr[row, step].item()),
                "step_q_triton": float(Qt[row, step].item()),
                "step_gh_ref": float(ref_gh_columns[step][row].item()),
                "step_gh_triton": float(triton_gh_columns[step][row].item()),
                "step_err_ref": float(Er[row, step].item()),
                "step_err_triton": float(Et[row, step].item()),
                "step_hdiag": float(H[step, step].item()),
                "step_ghlast_ref": float(ref_gh_last_columns[step][row].item()),
                "step_ghlast_triton": float(triton_gh_last_columns[step][row].item()),
                "step_hlast": float(H[step, -1].item()),
                "step_expected_last_ref": float(
                    (ref_snapshots[step - 1][row, -1] if step else W0[row, -1])
                    - (Er[row, step] * H[step, -1] + ref_gh_last_columns[step][row])
                    * args.second_order_scale
                ),
                "step_expected_last_triton": float(
                    (triton_snapshots[step - 1][row, -1] if step else W0[row, -1])
                    - (Et[row, step] * H[step, -1] + triton_gh_last_columns[step][row])
                    * args.second_order_scale
                ),
            }
            break
    report = {
        "rows": args.rows,
        "cols": args.cols,
        "parent_cols": parent_cols,
        "input_stride0": W0.stride(0),
        "output_q_stride0": torch.empty_like(W0).stride(0),
        "device": torch.cuda.get_device_name(),
        "q_int_mismatches": q_int_mismatches,
        "q_mismatch_first_coords": q_mismatch[:10].tolist(),
        "q_max_abs": max_abs(Qr, Qt),
        "err_max_abs": max_abs(Er, Et),
        "err_max_abs_location": [err_row, err_col],
        "err_ref_at_max": float(Er[err_row, err_col].item()),
        "err_triton_at_max": float(Et[err_row, err_col].item()),
        "weight_max_abs": max_abs(Wr, Wt),
        "ghinv_max_abs": max_abs(GHr, GHt),
        "first_outer_matmul_vs_mul_max_abs": max_abs(outer_matmul_probe, outer_mul_probe),
        "allrows_outer_probe_col": probe_col,
        "allrows_outer_matmul_vs_mul_max_abs": float(probe_delta.max().item()),
        "allrows_outer_diff_location": [probe_row, probe_col + probe_offset],
        "allrows_outer_ref": float(probe_outer[probe_row, probe_offset].item()),
        "allrows_outer_mul": float(probe_mul[probe_row, probe_offset].item()),
        "max_weight_abs_by_column_peak": float(step_errors.max().item()),
        "worst_column_index": worst_col,
        "first_weight_divergence": first_diff,
        "first_column_over_1e-6": next(
            (i for i, value in enumerate(step_errors.tolist()) if value > 1e-6), None
        ),
        "step_errors_first_16": step_errors[:16].tolist(),
        "step_errors_near_first_drift": step_errors[36:48].tolist(),
        "weight_allclose_1e5": bool(torch.allclose(Wr, Wt, atol=1e-5, rtol=1e-5)),
        "ghinv_allclose_1e5": bool(torch.allclose(GHr, GHt, atol=1e-5, rtol=1e-5)),
    }
    for key, value in report.items():
        print(f"{key}: {value}")
    if (
        q_int_mismatches
        or report["q_max_abs"] != 0.0
        or report["err_max_abs"] != 0.0
        or report["weight_max_abs"] != 0.0
        or report["ghinv_max_abs"] != 0.0
        or report["max_weight_abs_by_column_peak"] != 0.0
    ):
        raise SystemExit("correctness smoke failed")

    if args.bench_iters:
        # Warm both paths before timing; snapshots are disabled so this measures
        # the loop, not the diagnostic copies used above for numerical auditing.
        def clone_inputs():
            return (
                W_storage.clone()[:, :args.cols],
                GH_storage.clone()[:, :args.cols],
            )

        warm_w, warm_gh = clone_inputs()
        reference_loop(
            warm_w, warm_gh, Z, H, scale, maxq=maxq, qlo=qlo,
            second_order_scale=args.second_order_scale, capture_snapshots=False,
        )
        warm_w, warm_gh = clone_inputs()
        fused_loop(
            warm_w, warm_gh, Z, H, scale, maxq=maxq, qlo=qlo,
            second_order_scale=args.second_order_scale, capture_snapshots=False,
        )

        def measure(fn):
            values = []
            for _ in range(args.bench_iters):
                W, GH = clone_inputs()
                torch.cuda.synchronize()
                start = time.perf_counter()
                fn(
                    W, GH, Z, H, scale, maxq=maxq, qlo=qlo,
                    second_order_scale=args.second_order_scale,
                    capture_snapshots=False,
                )
                torch.cuda.synchronize()
                values.append((time.perf_counter() - start) * 1000.0)
            return statistics.median(values), values

        reference_ms, reference_runs = measure(reference_loop)
        fused_ms, fused_runs = measure(fused_loop)
        print(f"reference_loop_ms_median: {reference_ms:.3f}")
        print(f"fused_loop_ms_median: {fused_ms:.3f}")
        print(f"microbenchmark_speedup: {reference_ms / fused_ms:.3f}x")
        print(f"reference_loop_ms_runs: {reference_runs}")
        print(f"fused_loop_ms_runs: {fused_runs}")


if __name__ == "__main__":
    main()

