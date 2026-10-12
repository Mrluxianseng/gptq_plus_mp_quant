"""Research-only isolated solver pool preflight after AS fragmentation failure.

All allocations remain on CUDA and count toward process/NVML memory. Actual
Qwen down_proj API fixtures are not full REAL-Q/Adam/quality evidence.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from tools import check_hessian_inverse_lifetime_cuda as fixture
from tools.hessian_inverse_lifetime_prototype import build_method

torch, impl = fixture.torch, fixture.impl


def install_isolated(implementation, *, enabled, records):
    original = implementation.GPTQPlus._compute_hessian_inverse_with_fallback
    proposed, source = build_method(implementation)

    def inverse(self, h_sub, *args, **kwargs):
        target = str(getattr(self, "layer_name", "")).endswith("down_proj")
        if enabled and target:
            if self.alpha != 0 or kwargs.get("need_hinv_init", False):
                raise RuntimeError("private-pool prototype only supports alpha0 without retained inverse")
            subgroup = getattr(self, "_isolated_hessian_next_subgroup", 0)
            if subgroup >= self.num_groups:
                raise RuntimeError("private-pool prototype requires a single fasterquant pass")
            storage = self.H[subgroup]
            if not storage.is_contiguous() or storage.shape != h_sub.shape:
                raise RuntimeError("private-pool prototype requires matching contiguous group storage")
            pool = torch.cuda.MemPool()
            with torch.cuda.use_mem_pool(pool, device=h_sub.device):
                init, factor, damp, fallback = proposed(self, h_sub, *args, **kwargs)
                if init is not None:
                    raise RuntimeError("unexpected retained inverse")
                if factor.stride() not in ((1, storage.shape[0]), (storage.shape[0], 1)):
                    raise RuntimeError("unexpected factor layout")
                destination = storage.as_strided(factor.shape, factor.stride())
                destination.copy_(factor)
                del factor
            # The solver tensors must not escape into the normal allocator.
            # Explicit synchronization is diagnostic; its cost must be measured
            # before this can be adopted in the production path.
            torch.cuda.synchronize(h_sub.device)
            snapshot = pool.snapshot(include_traces=False)
            active = sum(block["size"] for segment in snapshot
                         for block in segment["blocks"]
                         if block["state"].startswith("active"))
            if active:
                raise RuntimeError(f"solver pool still has {active} active bytes")
            del snapshot, pool
            self._isolated_hessian_next_subgroup = subgroup + 1
            result = init, destination, damp, fallback
            released = True
        else:
            result = original(self, h_sub, *args, **kwargs)
            released = False
        if target:
            init, factor, damp, fallback = result
            records.append(dict(layer=getattr(self, "layer_idx", None),
                                module=self.layer_name, enabled=bool(enabled),
                                shape=list(h_sub.shape), factor_stride=list(factor.stride()),
                                damping=damp, identity_fallback=fallback,
                                private_pool_released=released))
        return result

    implementation.GPTQPlus._compute_hessian_inverse_with_fallback = inverse
    return source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-directory", type=Path, required=True)
    p.add_argument("--model-paths", type=Path, nargs="+", required=True)
    args = p.parse_args()
    args.output_directory.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    fixture.configure_reproducibility(44, deterministic=True)
    fixture.install_down_proj = install_isolated
    result = dict(valid=False, models=[], torch=str(torch.__version__),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope="actual first-block down_proj CUDA fixture; no Adam/full-model quality proof",
                  synchronization="per subgroup diagnostic synchronization; not speed evidence")
    try:
        for path in args.model_paths:
            print(f"[inverse-pool] actual Qwen fixture {path}", flush=True)
            weight, inputs, tokens = fixture.actual_first_block(path, 2, 256)
            row = dict(model=str(path), input_sha256=fixture.digest(inputs),
                       tokens_sha256=[fixture.digest(x) for x in tokens])
            row["api"] = fixture.api_pair(weight, inputs, None, args.output_directory, factor_reuse=True)
            result["models"].append(row)
            if not row["api"]["exact"]:
                raise RuntimeError("isolated solver pool changes quantization results")
            if not all(x["private_pool_released"] for x in row["api"]["arms"][1]["calls"]):
                raise RuntimeError("not all four private pools released")
            print(json.dumps(row), flush=True)
            del weight, inputs, tokens
            gc.collect()
            torch.cuda.empty_cache()
        result["valid"] = True
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        result["finished"] = time.strftime("%F %T")
        (args.output_directory / "status.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
