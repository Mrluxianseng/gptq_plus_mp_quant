"""Bound CUDA preflight: inverse reuse and stable final weight destination."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import check_hessian_inverse_lifetime_cuda as fixture
from tools.hessian_stable_weight_prototype import build_stable_fasterquant, install_fasterquant


def main():
    # Reuse the unchanged actual Qwen/WikiText fixture and inverse/fallback
    # cases. Both arms additionally share the same storage ownership observer.
    source = textwrap.dedent(inspect.getsource(fixture.api_pair))
    before = 'impl.GPTQPlus.fasterquant = reused if enabled and factor_reuse else original_fasterquant'
    after = '''impl.GPTQPlus.fasterquant = original_fasterquant
        write_calls = []
        install_fasterquant(impl, enabled=enabled, records=write_calls)'''
    if source.count(before) != 1:
        raise RuntimeError("fixture install boundary changed")
    source = source.replace(before, after)
    source = source.replace('calls=calls,reuse_calls=reuse_calls)',
                            'calls=calls,reuse_calls=reuse_calls,weight_write_calls=write_calls)')
    namespace = dict(vars(fixture), build_reusing_fasterquant=build_stable_fasterquant,
                     install_fasterquant=install_fasterquant)
    exec(compile(source, __file__, "exec"), namespace)
    fixture.api_pair = namespace["api_pair"]
    if "--factor-reuse" not in sys.argv:
        sys.argv.append("--factor-reuse")
    out = Path(sys.argv[sys.argv.index("--output-directory")+1])
    fixture.main()
    report = json.loads((out/"status.json").read_text())
    component_valid = report["valid"]
    report["valid"] = False
    (out/"status.json").write_text(json.dumps(report,indent=2))
    # Verify the final cast itself across destination dtypes and both layouts.
    torch = fixture.torch
    cases = []
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        for transposed in (False, True):
            src = torch.randn(128,256,device="cuda")
            if transposed:
                src = src.T
            expected = src.to(dtype).contiguous()
            destination = torch.empty(src.shape,dtype=dtype,device="cuda")
            destination.copy_(src)
            cases.append(dict(dtype=str(dtype),transposed_source=transposed,
                              exact=fixture.digest(expected)==fixture.digest(destination)))
            del src, expected, destination
    report["copy_cast_cases"] = cases
    helper = ROOT/"tools/hessian_stable_weight_prototype.py"
    (out/"stable_weight_prototype.py").write_bytes(helper.read_bytes())
    report["stable_weight_source_sha256"] = hashlib.sha256(helper.read_bytes()).hexdigest()
    report["valid"] = component_valid and all(c["exact"] for c in cases) and all(
        len(m["api"]["arms"][1]["weight_write_calls"]) == 1 and
        m["api"]["arms"][1]["weight_write_calls"][0]["pointer_preserved"]
        for m in report["models"])
    (out/"status.json").write_text(json.dumps(report,indent=2))
    print("[stable-weight] bound CUDA preflight valid="+str(report["valid"]),flush=True)
    if not report["valid"]:
        raise RuntimeError("cast/source/weight ownership preflight failed")


if __name__ == "__main__":
    main()
