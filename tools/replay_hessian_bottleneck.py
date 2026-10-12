"""Serial CUDA solver screening on captured actual REAL-Q Hessians.

Fresh allocator environment; not a full trajectory or KL/PPL quality proof.
Separate processes per method avoid solver allocations leaking between arms.
"""
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.hessian_bottleneck_capture import file_sha, tensor_identity


def run_worker(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from gptq_utils import gptq_plus_utils as impl
    from utils.reproducibility import configure_reproducibility
    configure_reproducibility(44, deterministic=True)
    torch.set_num_threads(1)
    provenance = json.loads((args.capture / "provenance.json").read_text())
    if provenance["torch"] != str(torch.__version__) or provenance["torch_cuda"] != torch.version.cuda:
        raise RuntimeError("capture/replay torch runtime differs")
    for filename, expected in {**provenance["source_sha256"], **provenance["capture_sources"]}.items():
        if file_sha(filename) != expected:
            raise RuntimeError(f"capture/replay source changed: {filename}")
    expected_file = next(x for x in json.loads((args.capture / "index.json").read_text())["records"]
                         if x["file"] == args.fixture.name)
    if file_sha(args.fixture) != expected_file["file_sha256"]:
        raise RuntimeError("fixture file identity changed")
    case = torch.load(args.fixture, map_location="cpu", weights_only=True)
    for key, identity in expected_file["tensors"].items():
        actual = None if case[key] is None else tensor_identity(case[key])
        if actual != identity:
            raise RuntimeError(f"fixture tensor mismatch: {key}")
    original = impl.GPTQPlus._compute_hessian_inverse_with_fallback
    method, source = original, inspect.getsource(original)
    if args.method != "reference":
        if args.method == "workspace":
            from tools.hessian_solver_workspace_prototype import build_method
        else:
            from tools.hessian_inverse_lifetime_prototype import build_method
        method, source = build_method(impl)
    helper = impl.GPTQPlus.__new__(impl.GPTQPlus)
    h = torch.empty_strided(case["H"].shape, case["H"].stride(),
                           device="cuda", dtype=case["H"].dtype)
    h.copy_(case["H"])
    # Warm CUDA solver libraries outside the measured window with the same H.
    with torch.no_grad():
        warm = original(helper, h.clone(), **case["options"])
        del warm
    gc.collect()
    torch.cuda.empty_cache()
    records = []
    for repeat in range(args.repeats):
        working = torch.empty_strided(h.shape, h.stride(), device=h.device, dtype=h.dtype)
        working.copy_(h)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline = dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved())
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.no_grad():
            inv, factor, damping, fallback = method(helper, working, **case["options"])
        torch.cuda.synchronize()
        # Capture peaks before hashing/host copies.
        record = dict(repeat=repeat, seconds=time.perf_counter()-started, baseline=baseline,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        identities = dict(H_after=tensor_identity(working), inverse=None if inv is None else tensor_identity(inv),
                          factor=tensor_identity(factor))
        record.update(tensors=identities, damping=damping, fallback=fallback,
                      exact=all(identities[k] == expected_file["tensors"][k] for k in identities)
                            and damping == case["damping"] and fallback == case["fallback"])
        records.append(record)
        del working, inv, factor
        if not record["exact"]:
            break
    report = dict(valid=all(x["exact"] for x in records), method=args.method,
                  fixture=str(args.fixture), fixture_sha256=expected_file["file_sha256"],
                  method_source_sha256=hashlib.sha256(source.encode()).hexdigest(), records=records,
                  torch=str(torch.__version__), device=torch.cuda.get_device_name(),
                  scope="Solver-only fresh allocation environment; diagnostic time; no full-process resource proof")
    args.output.write_text(json.dumps(report, indent=2))
    if not report["valid"]:
        raise RuntimeError("captured solver mismatch; stop before trajectory tests")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--methods", nargs="+", choices=("reference", "lifetime", "workspace"),
                   default=["reference", "lifetime", "workspace"])
    p.add_argument("--worker", action="store_true")
    p.add_argument("--fixture", type=Path)
    p.add_argument("--method", choices=("reference", "lifetime", "workspace"))
    args = p.parse_args()
    if args.repeats < 1:
        p.error("positive repetitions required")
    if args.worker:
        return run_worker(args)
    args.output.mkdir(exist_ok=False)
    reports = []
    for row in json.loads((args.capture / "index.json").read_text())["records"]:
        for method in args.methods:
            target = args.output / f"layer{row['layer']}-{method}.json"
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--capture", str(args.capture),
                       "--fixture", str(args.capture / row["file"]), "--method", method,
                       "--output", str(target), "--repeats", str(args.repeats)]
            print(f"[bottleneck-replay] layer={row['layer']} method={method}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            reports.append(json.loads(target.read_text()))
    (args.output / "status.json").write_text(json.dumps(dict(valid=bool(reports) and all(x["valid"] for x in reports),
        results=reports, scope="Component screening only; no allocator history restoration"), indent=2))


if __name__ == "__main__":
    main()
