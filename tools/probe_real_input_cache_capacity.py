"""Actual Qwen input capture/clone capacity probe, not a quantization result.

Uses WikiText training rows and the real model prefix before block zero.
The selected storage policy is the production Stage-2 allocation policy.
Only this phase is executed: no Fisher collection, Adam, KL/PPL or four-rank
claim. Separate workers isolate allocator history and expected CUDA OOM.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.data_utils import get_loaders


def fingerprint_rows(values):
    digest = hashlib.sha256()
    digest.update(str((tuple(values.shape), str(values.dtype))).encode())
    for row in values:
        value = row.detach().cpu().contiguous()
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def memory():
    torch.cuda.synchronize()
    return {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved(),
            "peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved(),
            "rss": psutil.Process().memory_info().rss}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--cpu-init", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples <= 0 or args.seq_len <= 0 or args.output.exists():
        raise ValueError("positive dimensions and a new output file are required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(2)
    torch.manual_seed(44)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    rows = get_loaders("wikitext2", "train", tokenizer, args.seq_len, args.samples, seed=44)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16,
                                               device_map="cpu", attn_implementation="sdpa", local_files_only=True).eval()
    model.config.use_cache = False
    model.model.embed_tokens.to("cuda")
    model.model.rotary_emb.to("cuda")
    shape = (args.samples, args.seq_len, model.config.hidden_size)
    payload = args.samples * args.seq_len * model.config.hidden_size * 2
    if args.cpu_init and psutil.virtual_memory().available < 2 * payload + 1024 ** 3:
        raise RuntimeError("insufficient host headroom for two CPU input caches")
    record = {"scope": "real Qwen prefix input capture and reference clone only; not full REAL-Q",
              "args": vars(args) | {"output": str(args.output)}, "shape": shape, "one_cache_payload_bytes": payload,
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "phases": {"model_prefix_ready": memory()}, "completed": False}
    count = 0
    inps = None
    original = model.model.layers[0]

    class CapturedInput(Exception):
        pass

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type
        def forward(self, value, **kwargs):
            nonlocal count
            inps[count] = value
            count += 1
            raise CapturedInput()

    phase = "allocate_inputs"
    started = time.monotonic()
    try:
        inps = torch.zeros(shape, dtype=torch.bfloat16, device="cpu" if args.cpu_init else "cuda")
        model.model.layers[0] = Catcher(original)
        phase = "capture_inputs"
        for index, batch in enumerate(rows):
            try:
                model(batch[0].to("cuda"))
            except CapturedInput:
                pass
            if (index + 1) % 64 == 0:
                print(f"[input-capacity] captured {index+1}/{args.samples}", flush=True)
        if count != args.samples:
            raise RuntimeError("input capture count mismatch")
        record["phases"]["after_capture"] = memory()
        phase = "clone_reference_inputs"
        fp_inps = inps.clone()
        record["phases"]["after_clone"] = memory()
        phase = "record_numerical_fingerprints"
        record["inps_sha256"] = fingerprint_rows(inps)
        record["fp_inps_sha256"] = fingerprint_rows(fp_inps)
        if record["inps_sha256"] != record["fp_inps_sha256"]:
            raise RuntimeError("reference input clone changed values")
        record["completed"] = True
    except torch.OutOfMemoryError as exc:
        record.update(cuda_oom=True, failed_phase=phase, error=str(exc),
                      phases=record["phases"] | {"after_oom": memory()})
        if inps is not None and count == args.samples:
            record["inps_sha256"] = fingerprint_rows(inps)
    finally:
        model.model.layers[0] = original
        record.update(captured_samples=count, wall_seconds=time.monotonic()-started,
                      finished_local=time.strftime("%F %T"))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".runner.py").write_bytes(Path(__file__).read_bytes())
        args.output.write_text(json.dumps(record, indent=2))
        print(json.dumps({k: v for k, v in record.items() if k != "phases"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
