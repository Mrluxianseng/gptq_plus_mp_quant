"""Frozen numerical configuration; portable paths and explicit topology binding."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

from .resources import MODELS, TASKS
from .sources import DEFAULT_VARIANTS, VARIANTS
from .storage import identity, read_json


def numerical_config(model: str) -> dict:
    if model not in MODELS:
        raise ValueError("Only q06/q17 are registered")
    return deepcopy(read_json(Path(__file__).with_name("frozen_config.json"))[model])


def cache_identity(model: str, assets: dict, world: int, runtime: dict) -> dict:
    if world != 2:
        raise ValueError("This version admits exactly two ranks")
    cfg = numerical_config(model)
    # Bind ALL numerical settings conservatively. Variant is intentionally absent:
    # CPU streams/solver/lifetimes do not change the teacher calculation.
    return {"schema": "portable-v4-stage0-v1", "model": model, "world_size": world,
            "model_content": assets[model]["content_sha256"],
            "dataset_content": assets["wikitext2"]["content_sha256"],
            "numerics": cfg, "runtime": runtime,
            "teacher_source": "official-v4-full-checkpoint"}


def make_config(model: str, assets: dict, cache: Path, output: Path,
                *, qa: str, cache_hit: bool, save_checkpoint: bool = False) -> dict:
    if qa not in ("full", "none"):
        raise ValueError("Explicit QA scope required")
    cfg = numerical_config(model)
    cfg.update(model=assets[model]["path"], static_cache_path=str(cache / "static"),
        fsdp_prepared_dir=str(cache / "rotated"), cache_dir=str(cache / "reference"),
        tokens_cache_path=str(cache / "tokens"), tokens_cache_file=str(cache / "calibration.pt"),
        output_dir=str(output / "native"), exp="portable-two-gpu-" + model,
        save_qmodel_path=str(output / "quantized.pt") if save_checkpoint else None,
        require_static_cache_hit=cache_hit, require_reference_cache_hit=cache_hit,
        lm_eval=qa == "full", reasoning_data_dir="./datasets/reasoning_eval",
        reasoning_lcb_source_dir="./datasets/reasoning_eval/vendor/LiveCodeBench")
    return cfg


def jobs(models, variants) -> list[dict]:
    if not models or len(set(models)) != len(models) or not set(models) <= set(MODELS):
        raise ValueError("Unique registered models required")
    if not variants or len(set(variants)) != len(variants) or not set(variants) <= set(VARIANTS):
        raise ValueError("Unique registered variants required")
    if variants[0] != "full_v4":
        raise ValueError("Each model requires a full_v4 control FIRST")
    return [{"model": model, "variant": variant} for model in models for variant in variants]


def validate_scope(plan: dict) -> None:
    if plan["world_size"] != 2 or len(plan["devices"]) != 2 or len(set(plan["devices"])) != 2:
        raise RuntimeError("Two unique explicit device UUIDs required")
    if any(not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value) for value in plan["devices"]):
        raise RuntimeError("Full GPU UUIDs required; no ordinal guessing or MIG slices")
    if plan["jobs"] != jobs(plan["models"], plan["variants"]):
        raise RuntimeError("Serial preregistered matrix differs")
    if plan["qa"] not in ("full", "none") or plan["qa_batch_size"] != 16 or plan["qa_tasks"] != list(TASKS):
        raise RuntimeError("Canonical QA configuration required")


def semantic_digest(value, *, allow_cuda: bool = False) -> str:
    """Tensor-content identity using bounded CPU byte chunks, no CUDA transfers.

    Shared Stage0 consumers must receive CPU tensors. Never retain a second
    full tensor tree, or call numpy().tobytes() on a whole multi-GB payload.
    """
    import hashlib
    import torch
    result = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            if item.device.type != "cpu" and not allow_cuda:
                raise RuntimeError("Persisted Stage0 tensors must actually be on CPU")
            result.update(b"tensor")
            result.update(identity({"shape": list(item.shape), "dtype": str(item.dtype)}).encode())
            data = item.detach().contiguous().reshape(-1).view(torch.uint8)
            for start in range(0, data.numel(), 4 * 1024 * 1024):
                result.update(memoryview(data[start:start + 4 * 1024 * 1024].cpu().numpy()))
        elif isinstance(item, dict):
            result.update(b"dict" + identity(len(item)).encode())
            for key in sorted(item):
                result.update(identity(key).encode()); visit(item[key])
        elif isinstance(item, (tuple, list)):
            result.update(b"sequence" + identity(len(item)).encode())
            for part in item: visit(part)
        else:
            result.update(identity(item).encode())
    visit(value)
    return result.hexdigest()


def model_fingerprint(model) -> dict:
    """Complete native 535-state/28-Hadamard evidence, bounded device readback."""
    import torch
    state = model.state_dict()
    if len(state) != 535 or any(p.dtype != torch.bfloat16 for p in model.parameters()):
        raise RuntimeError("Complete 535-state BF16 quantized model required")
    states = {name: {"shape": list(t.shape), "dtype": str(t.dtype),
                     "sha256": semantic_digest(t, allow_cuda=True)} for name, t in state.items()}
    had = []
    for name, module in model.named_modules():
        if "down_proj" in name and hasattr(module, "online_full_had"):
            if not module.online_full_had or module.fp32_had:
                raise RuntimeError("Original online Hadamard behavior required")
            had.append({"name": name, "K": module.K, "fp32_had": module.fp32_had,
                        "online_partial_had": module.online_partial_had, "had_dim": module.had_dim})
    if len(had) != 28:
        raise RuntimeError("All 28 online Hadamard modules required")
    return {"state": states, "online_hadamard": had}
