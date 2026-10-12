"""CUDA API/arithmetic check before full REAL-Q export-buffer experiments.

Small real GPTQPlus API fixtures, not model-scale memory/quality evidence.
Checks both non-export and export consumers; full-model tests remain required.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from gptq_utils import gptq_plus_utils as impl
from utils.quant_utils import WeightQuantizer
from utils.reproducibility import configure_reproducibility
from tools.export_buffer_lifetime_prototype import build_method
from gptq_utils.diagnostics import DiagnosticRecorder


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=False)
    original = impl.GPTQPlus.fasterquant
    proposed, source = build_method(impl)
    (args.output_dir / "patched_fasterquant.py").write_text(source)
    rows = []
    try:
        for export, diagnostic in ((False, False), (True, False), (False, True)):
            arms = []
            for lazy in (False, True):
                configure_reproducibility(44, deterministic=True)
                layer = torch.nn.Linear(32, 64, bias=False, device="cuda")
                worker = impl.GPTQPlus(layer, torch.ones(1, 2, 4, device="cuda"),
                                      torch.zeros(64, 32), 4, 0.0, 0.0)
                worker.H.copy_(torch.eye(32, device="cuda").expand_as(worker.H))
                worker.quantizer = WeightQuantizer()
                worker.quantizer.configure(4, perchannel=True, sym=True)
                worker.layer_idx, worker.layer_name = 0, "small_api_fixture"
                impl.GPTQPlus.fasterquant = proposed if lazy else original
                recorder = (DiagnosticRecorder(str(args.output_dir/f"diagnostic-{lazy}"), 0, "fixture")
                            if diagnostic else None)
                worker.fasterquant(blocksize=16, groupsize=-1, actorder=True,
                                  enable_gradient_update=False, export_to_et=export, inner_kernel_impl="torch",
                                  diagnostic_recorder=recorder)
                tensors = {name: digest(value) for name, value in layer.state_dict().items()}
                saved_exports = {}
                if recorder is not None:
                    for name in ("W_int_final", "Scale_final"):
                        path = Path(recorder.root) / "module_level" / f"{name}.pt"
                        if not path.exists():
                            raise RuntimeError(f"diagnostic consumer did not receive {name}")
                        saved_exports[name] = digest(torch.load(path, map_location="cpu", weights_only=True))
                arms.append(dict(lazy=lazy, tensors=tensors,
                                 quantizer={name: digest(value) for name, value in worker.quantizer.state_dict().items()},
                                 saved_diagnostic_exports=saved_exports))
                del layer, worker
                torch.cuda.empty_cache()
            exact = (arms[0]["tensors"] == arms[1]["tensors"] and arms[0]["quantizer"] == arms[1]["quantizer"]
                     and arms[0]["saved_diagnostic_exports"] == arms[1]["saved_diagnostic_exports"])
            rows.append(dict(export=export, diagnostic=diagnostic, exact=exact, arms=arms))
        result = dict(valid=all(row["exact"] for row in rows), rows=rows,
                      patched_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                      scope="Tiny actual CUDA API fixtures; no model-scale memory/speed or Adam-path proof")
        (args.output_dir / "status.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        if not result["valid"]:
            raise SystemExit(1)
    finally:
        impl.GPTQPlus.fasterquant = original


if __name__ == "__main__":
    main()
