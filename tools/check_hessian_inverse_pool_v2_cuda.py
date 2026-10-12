"""Fresh pool preflight with retry/fallback plus actual two-model CUDA fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import check_hessian_inverse_lifetime_cuda as fixture
from tools.hessian_inverse_pool_prototype import install_isolated

torch, impl = fixture.torch, fixture.impl


def helper_case(h_cpu, *, inplace=False, damping=.01, increment=.0015):
    original = impl.GPTQPlus._compute_hessian_inverse_with_fallback
    rows = []
    try:
        for enabled in (False, True):
            obj = impl.GPTQPlus.__new__(impl.GPTQPlus)
            obj.alpha, obj.num_groups = 0., 1
            obj.layer_idx, obj.layer_name = 0, "mlp.down_proj"
            obj.H = h_cpu[None].cuda().clone()
            h = h_cpu.cuda().clone()
            records = []
            install_isolated(impl, enabled=enabled, records=records)
            init, factor, damp, fallback = obj._compute_hessian_inverse_with_fallback(
                h, percdamp=damping, damp_auto_increment=increment,
                inplace_damping=inplace, need_hinv_init=False)
            rows.append(dict(inverse=init, factor=fixture.digest(factor),
                             stride=list(factor.stride()), input_after=fixture.digest(h),
                             damping=damp, fallback=fallback, records=records))
            del obj, h, init, factor
            impl.GPTQPlus._compute_hessian_inverse_with_fallback = original
        fields = ("inverse", "factor", "stride", "input_after", "damping", "fallback")
        return dict(exact=all(rows[0][k] == rows[1][k] for k in fields), arms=rows)
    finally:
        impl.GPTQPlus._compute_hessian_inverse_with_fallback = original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-directory", type=Path, required=True)
    p.add_argument("--model-paths", type=Path, nargs="+", required=True)
    args = p.parse_args()
    out = args.output_directory
    out.mkdir(exist_ok=False)
    fixture.configure_reproducibility(44, deterministic=True)
    torch.set_num_threads(1)
    from tools.hessian_inverse_lifetime_prototype import build_method
    _, source = build_method(impl)
    (out/"pool_prototype.py").write_bytes((ROOT/"tools/hessian_inverse_pool_prototype.py").read_bytes())
    result = dict(valid=False, models=[], helper_cases=[], torch=str(torch.__version__),
                  helper_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  pool_source_sha256=hashlib.sha256((out/"pool_prototype.py").read_bytes()).hexdigest(),
                  scope="CUDA components; no Adam/full-model or speed claim")
    try:
        for dtype in (torch.float32, torch.float64):
            x = torch.randn(40,32,dtype=dtype)
            for inplace in (False, True):
                row = helper_case(x.T@x, inplace=inplace)
                row.update(dtype=str(dtype), case="spd", inplace=inplace)
                result["helper_cases"].append(row)
        for kind in ("indefinite", "nonfinite"):
            h = torch.eye(32)
            h[0,0] = -2 if kind == "indefinite" else float("nan")
            row = helper_case(h, damping=.98, increment=.01)
            row["case"] = kind
            result["helper_cases"].append(row)
        if not all(x["exact"] for x in result["helper_cases"]):
            raise RuntimeError("private-pool helper or fallback mismatch")
        print("[inverse-pool] six CUDA SPD/inplace/retry/fallback cases exact", flush=True)
        fixture.install_down_proj = install_isolated
        for path in args.model_paths:
            print(f"[inverse-pool] actual Qwen fixture {path}", flush=True)
            weight, inputs, tokens = fixture.actual_first_block(path, 2, 256)
            row = dict(model=str(path), input_sha256=fixture.digest(inputs),
                       tokens_sha256=[fixture.digest(t) for t in tokens])
            row["api"] = fixture.api_pair(weight, inputs, None, out, factor_reuse=True)
            result["models"].append(row)
            if not row["api"]["exact"]:
                raise RuntimeError("actual down_proj CUDA mismatch")
            print(json.dumps(row), flush=True)
            del weight, inputs, tokens
        result["valid"] = True
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        result["finished"] = time.strftime("%F %T")
        (out/"status.json").write_text(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
