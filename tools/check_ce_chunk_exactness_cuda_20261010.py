"""Idle-GPU CE fidelity check, not a quantization or speed experiment.

Separately tests per-token kernel fidelity and final reduction order. Retains
negative current-path observations instead of weakening the exactness gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from utils.eval_utils import _causal_nll_from_logits


def ordered_chunk(logits, labels, size):
    pieces = []
    for start in range(0, logits.shape[0] - 1, size):
        end = min(start + size, logits.shape[0] - 1)
        pieces.append(F.cross_entropy(logits[start:end].float()[None].permute(0, 2, 1),
                                     labels[start + 1:end + 1][None], reduction="none"))
    # Retain only O(tokens) FP32 loss values, then use the original reduction.
    return torch.cat(pieces, dim=1).float().mean(dim=1)[0]


def layout_preserving_chunk(logits, labels, size):
    """Research-only: keep Torch2.11 spatial log-softmax in its serial branch.

    The reference inner dimension is token count. For inner <=64 CUDA reduces
    vocabulary across threads; >64 accumulates it sequentially in each thread.
    No tile may cross this branch threshold relative to the reference. A short
    tail is merged; the actual tile is bounded by max(size,65)+64 rows.
    """
    count = logits.shape[0] - 1
    physical_size = max(size, 65) if count > 64 else count
    pieces = []
    start = 0
    while start < count:
        end = min(start + physical_size, count)
        if count > 64 and 0 < count - end <= 64:
            end = count
        pieces.append(F.cross_entropy(logits[start:end].float()[None].permute(0, 2, 1),
                                     labels[start + 1:end + 1][None], reduction="none"))
        start = end
    return torch.cat(pieces, dim=1).float().mean(dim=1)[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layout-preserving", action="store_true", help="Also test a research branch-preserving path, including short tails")
    p.add_argument("--require-production-exact", action="store_true", help="Fail unless the integrated production path matches every reference")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(1)
    rows = []
    # Non-power-of-two tails, two lengths and both Qwen vocab distributions.
    for length in ((33, 65, 66, 129, 258, 513, 777, 1024) if args.layout_preserving else (513, 1024)):
        for seed, scale in ((44, 1.0), (45, 4.0), (46, 8.0)):
            gen = torch.Generator(device="cuda").manual_seed(seed)
            logits = torch.randn(length, 151936, device="cuda", dtype=torch.bfloat16, generator=gen) * scale
            labels = torch.randint(151936, (length,), device="cuda", generator=gen)
            reference = _causal_nll_from_logits(logits, labels, chunk_tokens=0)
            for size in (16, 128, 256):
                current = _causal_nll_from_logits(logits, labels, chunk_tokens=size)
                ordered = ordered_chunk(logits, labels, size)
                rows.append({"length": length, "seed": seed, "scale": scale, "chunk": size,
                             "reference": reference.item(), "current": current.item(), "ordered": ordered.item(),
                             "current_exact": torch.equal(reference, current),
                             "ordered_exact": torch.equal(reference, ordered)})
                if args.layout_preserving:
                    preserved = layout_preserving_chunk(logits, labels, size)
                    rows[-1].update(layout_preserving=preserved.item(), layout_preserving_exact=torch.equal(reference, preserved))
            del logits, labels
    result = {"scope": "synthetic BF16 CUDA logits; no full-model memory/speed claim",
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "current_all_exact": all(r["current_exact"] for r in rows),
              "ordered_all_exact": all(r["ordered_exact"] for r in rows), "cases": rows}
    if args.layout_preserving:
        result["layout_preserving_all_exact"] = all(r["layout_preserving_exact"] for r in rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".runner.py").write_bytes(Path(__file__).read_bytes())
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))
    if args.layout_preserving and not result["layout_preserving_all_exact"]:
        raise RuntimeError("branch-preserving path changed CE; do not integrate")
    if args.require_production_exact and not result["current_all_exact"]:
        raise RuntimeError("production CE differs from reference; do not expand model experiments")
    if not args.layout_preserving and not result["ordered_all_exact"]:
        raise RuntimeError("ordered reduction alone does not restore kernel fidelity; inspect values")


if __name__ == "__main__":
    main()
