"""Small functional checks or actual trained-head CUDA fidelity preflight.

The hidden vectors are synthetic shape fixtures, not real quantization
residuals. This establishes API/arithmetic fidelity only, never full REAL-Q
quality, resource or speed. Original trained head and GEMM dimensions are
preserved identically for both policies.
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
import torch.nn.functional as F
from utils.loss_utils import chunked_tokenwise_kl_from_logits
from tools.cpu_saved_logits_kl_prototype import cpu_saved_logits_kl
from utils.reproducibility import configure_reproducibility


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--rows", type=int, default=19)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.rows <= 0 or args.output.exists():
        raise ValueError("positive rows and a fresh output file are required")
    torch.set_num_threads(2)
    configure_reproducibility(44, deterministic=True)
    if args.model_path:
        from safetensors import safe_open
        config = json.loads((args.model_path / "config.json").read_text())
        model_root = args.model_path.resolve()
        index_path = model_root / "model.safetensors.index.json"
        index_sha = None
        if index_path.exists():
            index_bytes = index_path.read_bytes()
            index_sha = hashlib.sha256(index_bytes).hexdigest()
            weight_map = json.loads(index_bytes)["weight_map"]
        else:
            weight_map = {}
            for filename in sorted(model_root.glob("*.safetensors")):
                with safe_open(filename, framework="pt", device="cpu") as storage:
                    for tensor_key in storage.keys():
                        if tensor_key in weight_map:
                            raise ValueError(f"duplicate tensor across shards: {tensor_key}")
                        weight_map[tensor_key] = filename.name
        key = "lm_head.weight" if "lm_head.weight" in weight_map else "model.embed_tokens.weight"
        if key == "model.embed_tokens.weight" and not config.get("tie_word_embeddings"):
            raise ValueError("cannot substitute embedding for an untied trained head")
        if key not in weight_map:
            raise ValueError("trained head tensor is absent from model checkpoint")
        shard = (model_root / weight_map[key]).resolve()
        if not shard.is_relative_to(model_root) or not shard.is_file():
            raise ValueError("trained head shard is outside model directory or missing")
        with safe_open(shard, framework="pt", device="cpu") as storage:
            weight_cpu = storage.get_tensor(key).to(torch.bfloat16)
        head_sha = digest(weight_cpu)
        weight = weight_cpu.to(args.device)
        del weight_cpu
    else:
        key, head_sha, shard, index_sha = "synthetic", None, None, None
        weight = torch.randn(127, 32, dtype=torch.bfloat16, device=args.device) / 32**0.5
    teacher = torch.randn(args.rows, weight.shape[1], dtype=weight.dtype, device=args.device)
    source = teacher + 0.05 * torch.randn_like(teacher)
    results = []
    for dtype in ((torch.bfloat16,) if args.model_path else (torch.float32, torch.bfloat16)):
        case_weight = weight.to(dtype)
        ref_hidden = source.to(dtype).detach().requires_grad_(True)
        cand_hidden = source.to(dtype).detach().clone().requires_grad_(True)
        c_logits = F.linear(ref_hidden, case_weight)
        n_logits = F.linear(cand_hidden, case_weight)
        teacher_logits = F.linear(teacher.to(dtype), case_weight).detach()
        ref_loss = chunked_tokenwise_kl_from_logits(c_logits, teacher_logits, chunk_tokens=16, topk=-1)
        new_loss = cpu_saved_logits_kl(n_logits, teacher_logits, chunk_tokens=16, topk=-1)
        ref_grad = torch.autograd.grad(ref_loss, ref_hidden)[0]
        new_grad = torch.autograd.grad(new_loss, cand_hidden)[0]
        results.append(dict(dtype=str(dtype), loss_exact=torch.equal(ref_loss, new_loss),
                            hidden_gradient_exact=torch.equal(ref_grad, new_grad),
                            loss_delta=float((ref_loss-new_loss).detach().abs()),
                            gradient_max_abs_delta=float((ref_grad-new_grad).abs().max()),
                            gradient_sha256_control=digest(ref_grad), gradient_sha256_candidate=digest(new_grad)))
        del c_logits, n_logits, teacher_logits, ref_loss, new_loss, ref_grad, new_grad, ref_hidden, cand_hidden
    result = dict(valid=all(row["loss_exact"] and row["hidden_gradient_exact"] for row in results),
                  rows=args.rows, vocab=int(weight.shape[0]), hidden=int(weight.shape[1]), trained_head_key=key,
                  trained_head_sha256=head_sha, results=results,
                  trained_head_shard=str(shard) if shard is not None else None,
                  model_index_sha256=index_sha,
                  deterministic=torch.are_deterministic_algorithms_enabled(),
                  cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                  prototype_sha256=hashlib.sha256((ROOT/"tools/cpu_saved_logits_kl_prototype.py").read_bytes()).hexdigest(),
                  scope="Synthetic hidden vectors with original GEMM shapes; no true quantization residual or full pipeline/memory/speed assertion")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
