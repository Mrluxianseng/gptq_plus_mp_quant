"""CPU-only checks of actual source functions, isolated from the PTQ runtime.

Extracting functions via AST avoids importing CUDA/distributed pipeline state.
This is not a substitute for integration testing.
"""
import ast
import json
from pathlib import Path
import torch
from torch.func import functional_call

torch.set_num_threads(2)
torch.manual_seed(1927)
source = Path('gptq_utils/gptq_plus_utils.py').read_text(encoding='utf-8')
tree = ast.parse(source)
names = {'compute_cross_module_compensation', '_two_sided_block_push', '_ldl_unit', '_two_sided_round'}
scope = {'torch': torch, 'functional_call': functional_call}
scope['resolve_quant_module'] = lambda full, name: (name, full[name])
scope['functional_weight_name_for_quant_module'] = lambda layer, name, resolved, module: name + '.weight'
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in names:
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual-source>', 'exec'), scope)

class LinearPair(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feeder = torch.nn.Linear(4, 4, bias=False)
        self.target = torch.nn.Linear(4, 3, bias=False)
    def forward(self, x, **kwargs):
        return self.target(self.feeder(x))

layer = LinearPair()
x = torch.randn(3, 17, 4)
fp = layer.feeder.weight.detach().clone()
cur = fp + 0.07 * torch.randn_like(fp)
with torch.no_grad():
    y0 = functional_call(layer, {'feeder.weight': fp}, (x,))
    y1 = functional_call(layer, {'feeder.weight': cur}, (x,))
    h = torch.nn.functional.linear(x, cur).reshape(-1, 4)
    d = (y1-y0).reshape(-1, 3)
    H = h.T @ h
    lam = 0.01 * H.diagonal().mean()
    exact = torch.linalg.solve(H + lam * torch.eye(4), -h.T @ d).T

results = {'cross': []}
for fscale in (None, 1.0, 0.01, 10.0):
    F = None if fscale is None else fscale * torch.eye(3)
    got = scope['compute_cross_module_compensation'](
        layer, {'feeder': layer.feeder, 'target': layer.target}, x,
        None, None, None, 'cpu', 'feeder', 'target', fp, cur,
        layer_output_fisher=F)
    # Objective with equally weighted ridge and residual: output F acts on both.
    # Scaling F cannot change its minimizer.
    row = {'fscale': fscale, 'relative_error_to_ridge': float((got-exact).norm()/exact.norm()),
           'residual_before': float(d.square().sum()),
           'residual_after': float((d+h @ got.T).square().sum()),
           'ridge_stationarity': float((got@(H+lam*torch.eye(4))+d.T@h).norm())}
    results['cross'].append(row)
    if fscale in (None, 1.0):
        torch.testing.assert_close(got, exact, rtol=2e-5, atol=2e-6)

# Independent scalar GPTQ implementation checks identity-output-metric reduction.
errors = []
for m, n in ((3, 5), (5, 3), (4, 4)):
    W = torch.randn(m, n, dtype=torch.float64)
    B = torch.randn(n, n, dtype=torch.float64)
    H = B @ B.T + torch.eye(n, dtype=torch.float64)
    U = torch.linalg.cholesky(torch.linalg.inv(H), upper=True)
    scale = torch.full_like(W, 0.3)
    got, _, ge = scope['_two_sided_block_push'](W, U, torch.eye(m, dtype=W.dtype), scale, -8, 7)
    wr = W.clone()
    qr = torch.zeros_like(W)
    er = torch.zeros_like(W)
    for j in range(n):
        for r in range(m):
            qr[r,j] = (wr[r,j]/scale[r,j]).round().clamp(-8,7)*scale[r,j]
            er[r,j] = (wr[r,j]-qr[r,j])/U[j,j]
            for k in range(j,n):
                wr[r,k] -= er[r,j]*U[j,k]
    torch.testing.assert_close(got, qr, rtol=0, atol=1e-12)
    torch.testing.assert_close(ge, er, rtol=1e-12, atol=1e-12)
    errors.append({'shape':[m,n], 'q_error':float((got-qr).abs().max())})
results['two_sided_identity_reduction'] = errors
tail_checks = []
for m, n in ((3, 5), (5, 3), (4, 4)):
    W = torch.randn(m, n, dtype=torch.float64)
    B = torch.randn(n, n, dtype=torch.float64)
    C = torch.randn(m, m, dtype=torch.float64)
    H = B@B.T + torch.eye(n, dtype=W.dtype)
    M = C@C.T + torch.eye(m, dtype=W.dtype)
    U = torch.linalg.cholesky(torch.linalg.inv(H), upper=True)
    A = torch.linalg.cholesky(torch.linalg.inv(M), upper=True)
    Q, _, E = scope['_two_sided_block_push'](W, U, A, torch.full_like(W,0.3), -8,7)
    # The true block residual must reproduce tail compensation independently
    # of the rounding strategy: W-Q = Err_effective @ U.
    effective = torch.linalg.solve_triangular(U.T, (W-Q).T, upper=False).T
    row_transport = (A/A.diagonal()[:,None]).T @ E
    torch.testing.assert_close(row_transport, effective, rtol=1e-10, atol=1e-10)
    tail_checks.append({'shape':[m,n],
        'returned_error_relative_mismatch':float((E-effective).norm()/effective.norm()),
        'row_transported_error_relative_mismatch':float((row_transport-effective).norm()/effective.norm())})
results['two_sided_tail_compensation'] = tail_checks
rounding_checks=[]
for seed in range(20):
    torch.manual_seed(seed)
    m,n=4,5
    W=torch.randn(m,n,dtype=torch.float64)
    B=torch.randn(n,n,dtype=W.dtype)
    C=torch.randn(m,m,dtype=W.dtype)
    H=B@B.T+torch.eye(n,dtype=W.dtype)
    M=C@C.T+torch.eye(m,dtype=W.dtype)
    U=torch.linalg.cholesky(torch.linalg.inv(H),upper=True)
    A=torch.linalg.cholesky(torch.linalg.inv(M),upper=True)
    scale=torch.full_like(W,0.3)
    push,_,_=scope['_two_sided_block_push'](W,U,A,scale,-8,7)
    gather,_=scope['_two_sided_round'](W.flip((0,1)),M.flip((0,1)),H.flip((0,1)),
                                      scale.flip((0,1)),-8,7,percdamp=0)
    gather=gather.flip((0,1))
    torch.testing.assert_close(push,gather,rtol=0,atol=1e-12)
    rounding_checks.append(float((push-gather).abs().max()))
results['two_sided_push_vs_reversed_ldl_gather']={'seeds':20,'max_error':max(rounding_checks)}
results['limitations'] = ['Only CPU synthetic linear tests; not full-model correctness.',
    'Identity reduction does not validate full output-metric feedback or block-to-block integration.']
out = Path('outputs/overnight_20260927_0143/math_audit.json')
out.write_text(json.dumps(results, indent=2), encoding='utf-8')
print(json.dumps(results, indent=2))
