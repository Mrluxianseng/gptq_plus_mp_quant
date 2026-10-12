import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.run_third_order_taylor_demo import (
    TinySwiGLUTail,
    exact_small_dimension_third_contraction,
    grad_at,
    hvp,
    make_kl_fn,
    rel_error,
)


def test_central_hvp_recovers_contracted_third_derivative():
    torch.manual_seed(101)
    hidden = 8
    tail = TinySwiGLUTail(hidden, 2 * hidden, 3 * hidden, 202).eval()
    y0 = torch.randn(1, hidden)
    with torch.no_grad():
        teacher = F.softmax(tail(y0), dim=-1)
    loss_fn = make_kl_fn(tail, teacher)
    delta = torch.randn_like(y0) * 0.04

    exact_t, _ = exact_small_dimension_third_contraction(loss_fn, y0, delta)
    step = 0.25
    estimate = (hvp(loss_fn, y0 + step * delta, delta) - hvp(loss_fn, y0 - step * delta, delta)) / (2 * step)

    assert rel_error(estimate, exact_t) < 1e-2


def test_cubic_correction_improves_local_kl_gradient_on_fixed_case():
    torch.manual_seed(303)
    hidden = 16
    tail = TinySwiGLUTail(hidden, 2 * hidden, 4 * hidden, 404).eval()
    y0 = torch.randn(2, hidden)
    with torch.no_grad():
        teacher = F.softmax(tail(y0), dim=-1)
    loss_fn = make_kl_fn(tail, teacher)
    delta = torch.randn_like(y0)
    delta = delta * (0.05 * y0.norm() / delta.norm())

    exact = grad_at(loss_fn, y0 + delta)
    quadratic = hvp(loss_fn, y0, delta)
    step = 0.25
    third = (hvp(loss_fn, y0 + step * delta, delta) - hvp(loss_fn, y0 - step * delta, delta)) / (2 * step)
    cubic = quadratic + 0.5 * third

    assert rel_error(cubic, exact) < rel_error(quadratic, exact)


if __name__ == "__main__":
    test_central_hvp_recovers_contracted_third_derivative()
    test_cubic_correction_improves_local_kl_gradient_on_fixed_case()
    print("2 third-order Taylor checks passed")
