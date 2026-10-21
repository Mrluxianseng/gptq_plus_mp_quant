"""One admitted torchrun worker. Frozen math, versioned I/O and observations.

Not the historical four-GPU formal auditor. CUDA acceptance requires a real
two-GPU run; CPU tests do not grant it. Nothing executes on module import.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import fields, asdict
from functools import wraps
import gc
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

from .storage import atomic_json, digest, exclusive_lock, file_manifest, identity, read_json


def execute(spec_path: Path):
    spec = read_json(spec_path)
    if spec.get("schema") != "portable-two-gpu-worker-v1":
        raise RuntimeError("Explicit worker spec required")
    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    if world != 2 or rank not in (0, 1) or os.environ["CUDA_VISIBLE_DEVICES"] != ",".join(spec["devices"]):
        raise RuntimeError("Bound two-GPU topology required")
    # The parent holds the campaign/device/cache leases through the actual wait.
    import psutil
    parent = psutil.Process(spec["parent_pid"])
    if parent.create_time() != spec["parent_create_time"] or parent.pid not in [p.pid for p in psutil.Process().parents()]:
        raise RuntimeError("Exact live controller ancestor required")
    source = Path(spec["source"])
    from .storage import verify_files
    verify_files(source / "realq", read_json(source / "manifest.json"))
    if any(name == "realq" or name.startswith("realq.") for name in sys.modules):
        raise RuntimeError("A different REAL-Q package was already imported")
    sys.path.insert(0, str(source))
    import torch
    if torch.cuda.device_count() != world:
        raise RuntimeError("Exactly two whole visible GPUs required")
    torch.cuda.set_device(rank)
    if torch.version.cuda != "12.8" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA12.8 and native BF16 support required")
    from realq import config, ptq, pipeline, precompute, runner
    from realq.parallel import env as parallel
    from realq.runner import layer_loop
    from realq.utils import data_utils, eval_utils, checkpoint_utils, dist_utils
    from realq.precompute import cache as static_cache
    from .observe import Observer
    from .protocol import semantic_digest, model_fingerprint
    from .resources import TASKS, validate_wikitext
    from .transport import HostTransport

    for name, module in list(sys.modules.items()):
        if name == "realq" or name.startswith("realq."):
            file = getattr(module, "__file__", None)
            if file is None or not Path(file).resolve().is_relative_to(source / "realq"):
                raise RuntimeError("Foreign REAL-Q module imported: " + name)

    output = Path(spec["output"])
    rank_dir = output / ("rank" + str(rank))
    rank_dir.mkdir(exist_ok=False)
    kwargs = {key: value for key, value in spec["config"].items()
              if key in {f.name for f in fields(config.Config) if f.init}}
    cfg = config.Config(**kwargs)
    # Native Config derives the display/cache name from basename(model). Hub
    # snapshots end in a commit hash; retain the registered model name instead.
    cfg.model_name = spec["config"]["model_name"]
    # No unregistered config fields or silently changed native defaults.
    effective = asdict(cfg)
    if set(effective) != set(spec["config"]) or any(effective[k] != v for k, v in spec["config"].items()):
        raise RuntimeError("Native configuration differs from the complete frozen contract")
    atomic_json(rank_dir / "configuration.json", effective)
    observer = Observer(torch, rank_dir / "allocator.jsonl", rank)
    guard = HostTransport(torch)
    metrics, stages, layers = {}, {}, []
    patches = []
    cache_root = Path(spec["cache"])
    control_cache = read_json(cache_root / "complete.json") if spec["cache_hit"] else None
    fresh_static = spec["variant"] in ("without_static_checkpoint", "baseline")
    data_files = validate_wikitext(Path(spec["assets"]["wikitext2"]["path"]))

    def patch(module, name, function):
        original = getattr(module, name)
        patches.append((module, name, original))
        setattr(module, name, function(original))

    def data_loader(original):
        @wraps(original)
        def load(path, name=None, *, split=None, **kwargs):
            if path != "./datasets/wikitext" or name != "wikitext-2-raw-v1" or split not in data_files:
                raise RuntimeError("Unregistered dataset resolution request")
            from datasets import load_dataset
            return load_dataset("parquet", data_files=data_files, split=split,
                                cache_dir=str(Path(spec["shared"]) / "hf/datasets"))
        return load

    def tokens_wrapper(original):
        @wraps(original)
        def call(dataset, split, tokenizer, seq_len, num_samples, save_path=None, seed=0):
            expected = ("wikitext2", "train", 2048, 2048, 1)
            if (dataset, split, seq_len, num_samples, seed) != expected or save_path != str(cache_root / "calibration.pt"):
                raise RuntimeError("Calibration scope differs")
            with exclusive_lock(cache_root / "tokens.lock", timeout=3600):
                value = original(dataset, split, tokenizer, seq_len, num_samples, save_path, seed)
            if len(value) != num_samples or any(t.device.type != "cpu" or t.dtype != torch.int64 or list(t.shape) != [seq_len] for t in value):
                raise RuntimeError("Actual calibration tensor shape/dtype/device differs")
            token_hash = semantic_digest(value)
            prior = stages.setdefault("calibration_sha256", token_hash)
            if prior != token_hash:
                raise RuntimeError("Calibration tokens changed inside this arm")
            return value
        return call

    def cache_load(original):
        def call(directory, key, size, index):
            if fresh_static: return None
            path = Path(static_cache.cache_path(directory, key, size, index))
            if not path.exists():
                if spec["cache_hit"]:
                    raise RuntimeError("Required shared Stage0 rank file missing")
                return None
            # Never restore serialized CUDA residency; mmap avoids a second full
            # anonymous CPU copy while reading a large immutable statistics file.
            value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if not isinstance(value, dict) or set(value) != {"saliency", "fisher"}:
                raise RuntimeError("Invalid shared Stage0 payload")
            return value
        return call

    def static_wrapper(original):
        @wraps(original)
        def call(options, analyzer):
            prior_path, prior_required = options.static_cache_path, options.require_static_cache_hit
            if fresh_static:
                options.static_cache_path, options.require_static_cache_hit = None, False
            try:
                mode = "fresh-no-duplicate-save" if fresh_static else ("cache-hit" if spec["cache_hit"] else "producer")
                with observer.scope("stage0", mode=mode):
                    result = original(options, analyzer)
            finally:
                options.static_cache_path, options.require_static_cache_hit = prior_path, prior_required
            content = semantic_digest({"saliency": result.saliency, "fisher": result.fisher})
            if control_cache and content != control_cache["rank_statistics"][str(rank)]:
                raise RuntimeError("Stage0 content differs from same-topology full_v4 control")
            stages.update(stage0_mode=mode, statistics_sha256=content)
            return result
        return call

    def quant_wrapper(original):
        @wraps(original)
        def call(options, analyzer, static, trainloader):
            with observer.scope("stage1_quantization"):
                value = original(options, analyzer, static, trainloader)
            # These teacher statistics have no reader after quantization. Release
            # them BEFORE KL/PPL/QA; do not retain model/tensor args in telemetry.
            static.saliency.clear(); static.fisher.clear()
            gc.collect()
            if rank == 0:
                state = analyzer.model.state_dict()
                if any(t.device.type != "cpu" for t in state.values()):
                    raise RuntimeError("Quantized CPU master did not return to CPU")
                states = model_fingerprint(analyzer.model)
                atomic_json(output / "state.json", states)
                metrics["state_sha256"] = identity(states)
            return value
        return call

    def layer_wrapper(original):
        @wraps(original)
        def call(*args, **kwargs):
            index = args[1] if len(args) > 1 else kwargs["layer_idx"]
            state = args[4] if len(args) > 4 else kwargs["state"]
            device = "cuda" if spec["variant"] in ("without_CPU_streams", "baseline") else "cpu"
            for key in ("inps", "fp_inps"):
                tensor = getattr(state, key)
                if tensor.device.type != device or (device == "cuda" and tensor.device.index != rank):
                    raise RuntimeError("Actual input stream device differs: " + key)
            with observer.scope("quantization_layer", layer=index, stream_device=device):
                result = original(*args, **kwargs)
            layers.append(index)
            return result
        return call

    def evaluate(original):
        @wraps(original)
        def call(*args, **kwargs):
            with observer.scope("kl_ppl"):
                ppl, kl = original(*args, **kwargs)
            if not math.isfinite(ppl) or not math.isfinite(kl) or ppl <= 0 or kl < -1e-7:
                raise RuntimeError("Invalid KL/PPL")
            metrics["ppl"], metrics["kl"] = ppl, max(0.0, kl)
            return ppl, kl
        return call

    def qa_wrapper(original):
        @wraps(original)
        def call(model, tokenizer, lm_eval_batch_size=32):
            if rank != 0 or lm_eval_batch_size != 16:
                raise RuntimeError("QA must run on rank0 with fixed batch16")
            model.eval()
            if model.config.use_cache is not True:
                raise RuntimeError("Native QA KV cache must remain enabled")
            with observer.scope("qa_state_check_before", devices=[0, 1]):
                if identity(model_fingerprint(model)) != metrics["state_sha256"]:
                    raise RuntimeError("State/Hadamard behavior changed before QA")
            with observer.scope("qa_full10", devices=[0, 1]):
                results = original(model, tokenizer, lm_eval_batch_size)
            with observer.scope("qa_state_check_after", devices=[0, 1]):
                if identity(model_fingerprint(model)) != metrics["state_sha256"]:
                    raise RuntimeError("State/Hadamard behavior changed during QA")
            atomic_json(output / "qa_state_review.json", {"before_after_exact": True,
                "state_tensors": 535, "online_hadamard_modules": 28,
                "state_sha256": metrics["state_sha256"], "KV_offloaded": False})
            if set(results) != {*TASKS, "acc_avg"} or any(not math.isfinite(v) or not 0 <= v <= 100 for v in results.values()):
                raise RuntimeError("Complete finite ten-task QA results required")
            metrics["qa"] = results
            return results
        return call

    def release_evaluation(original):
        @wraps(original)
        def call(options, analyzer, head, loaders, references):
            result = original(options, analyzer, head, loaders, references)
            # Last consumers of full reference hidden states and evaluation
            # tokens have returned. QA uses model/tokenizer, not these trees.
            # Do not mutate head: it could alias a model parameter.
            references.clear()
            loaders.clear()
            gc.collect()
            return result
        return call

    def resident_map(original):
        def call(model):
            from accelerate import dispatch_model
            actual_layers = model.model.layers
            if len(actual_layers) != 28 or torch.cuda.device_count() != 2:
                raise RuntimeError("Complete dense 28-layer/two-device QA required")
            mapping = {"": 0, "model.norm": 1, "lm_head": 1}
            mapping.update({f"model.layers.{i}": i // 14 for i in range(28)})
            dispatch_model(model, device_map=mapping, offload_buffers=False)
            if any(t.device.type != "cuda" for t in model.state_dict().values()):
                raise RuntimeError("QA weights/buffers must remain on selected GPUs")
            atomic_json(output / "qa_placement.json", {"device_map": mapping,
                "weight_offload": False, "kv_offload": False, "scope": "two-GPU whole-layer resident map"})
        return call

    def checkpoint(original):
        def call(requested, model, options, tokenizer):
            # Serialize one copy per actual state+runtime identity. Every arm
            # retains its own reference and source/config evidence, never a copy.
            state_key = identity({"state": metrics["state_sha256"], "model": spec["assets"][spec["model"]]["content_sha256"],
                                  "w_bits": 4, "a_bits": 16, "rotate": True})
            target = Path(spec["shared"]) / "checkpoints" / state_key / "quantized.pt"
            receipt = target.with_suffix(".json")
            with exclusive_lock(Path(spec["shared"]) / "locks/checkpoints.lock"):
                target.parent.mkdir(parents=True, exist_ok=True)
                if receipt.exists():
                    record = read_json(receipt)
                    if not target.exists() or digest(target) != record["file_sha256"] or record["state_sha256"] != metrics["state_sha256"]:
                        raise RuntimeError("Shared quantized checkpoint changed")
                else:
                    if target.exists():
                        raise RuntimeError("Unsealed checkpoint exists; inspect previous interrupted run")
                    original(str(target), model, options, tokenizer)
                    record = {"path": str(target), "state_sha256": metrics["state_sha256"], "file_sha256": digest(target),
                              "created_by": str(output), "source_sha256": spec["source_sha256"]}
                    atomic_json(receipt, record)
            atomic_json(output / "checkpoint.json", record)
            options.save_qmodel_path = str(target)  # native success log reports actual path
        return call

    completed = False
    try:
        patch(data_utils, "load_dataset", data_loader)
        patch(data_utils, "get_tokens", tokens_wrapper)
        patch(static_cache, "try_load", cache_load)
        patch(precompute, "run", static_wrapper)
        patch(runner, "quantize_all_layers", quant_wrapper)
        patch(layer_loop, "quantize_one_layer", layer_wrapper)
        patch(eval_utils, "_kl_ppl_eval", evaluate)
        patch(eval_utils, "kl_ppl_eval", release_evaluation)
        patch(eval_utils, "qa_eval", qa_wrapper)
        patch(dist_utils, "distribute_model", resident_map)
        patch(checkpoint_utils, "save_quantized_checkpoint", checkpoint)
        patch(ptq.cfg_mod, "parse_cli", lambda _: lambda: cfg)
        with observer.scope("core_process"):
            ptq.main()
        if layers != list(range(28)) or "statistics_sha256" not in stages:
            raise RuntimeError("Full Stage0/28-layer coverage missing")
        if rank == 0 and (not {"kl", "ppl", "state_sha256"} <= set(metrics) or (spec["qa"] == "full" and "qa" not in metrics)):
            raise RuntimeError("Required quality measurements missing")
        completed = True
    finally:
        for module, name, original in reversed(patches): setattr(module, name, original)
        try:
            transport = guard.close()
            atomic_json(rank_dir / "transport.json", transport)
        finally:
            observer.close()
            atomic_json(rank_dir / "result.json", {"execution_completed": completed, "rank": rank,
                "world_size": world, "layers": layers, "stages": stages, "metrics": metrics,
                "qa_scope": spec["qa"], "historical_four_gpu_acceptance": False})
            # Process teardown frees all weights/caches and the CUDA context.
            if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()
            gc.collect()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--spec", type=Path, required=True)
    execute(parser.parse_args().spec)
