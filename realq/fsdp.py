"""FSDP2 single-stage helper.

Wires the FSDP2 wrap → precompute → checkpoint → unwrap → CPU master flow
that lets RealQ run end-to-end in one Python process while still spreading
the precompute backward across all visible GPUs.

The user-facing knob is ``Config.fsdp = True``. ``Config.fsdp_cpu_offload``
optionally pushes the FSDP master shards to CPU during precompute (cheaper
GPU memory, slower steps). ``Config.fsdp_prepared_dir`` is the on-disk
location for the post-rotate, post-precompute checkpoint that the quant
phase reloads from; defaults under ``cfg.cache_dir / fsdp_prepared``.

Why two-stage at all (instead of in-process unwrap)? FSDP2's DTensor
parameters do not survive an ad-hoc ``model.cpu()`` cleanly — see old
``gptq_plus_utils.py`` line 5212 ("In-process FSDP unwrap has been removed
— it was too fragile in PyTorch 2.9 FSDP2"). Save-then-reload sidesteps
the lifecycle issue and keeps peak CPU memory bounded by one model copy.

cpu_master path (``Config.cpu_master = True``):
  Phase A    rank0-only rotate + disk cache (under ``_prepared_checkpoints/``).
  Phase B    every rank: ``init_empty_weights`` from config → ``fully_shard``
             on the meta model → ``load_checkpoint_in_model(broadcast_from_rank0=True)``
             populates each FSDP shard from rank-0 reads. Wrappers installed
             AFTER load (key matching needs vanilla state_dict).
  Phase C    precompute (unchanged; weights immutable).
  Phase D    drop FSDP analyzer; rank0 reloads the rotated checkpoint to CPU,
             rank>0 builds a meta skeleton. ``CpuMasterLayerManager`` then
             materialises one block at a time during quant.
  Phase F    eval rank0-only with barriers; lm_eval barriers + destroys the
             process group, rank>0 returns. (Mirrors ptq.py:55-206.)

Direct ports of helpers from ``utils/model_utils.py`` (rotate cache,
sharded broadcast load) keep the cpu_master path numerically equivalent to
the old ``--fsdp_meta_init`` two-process workflow.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from typing import TYPE_CHECKING, Tuple

import torch
import torch.distributed as dist
from transformers import AutoConfig

if TYPE_CHECKING:
    from realq.config import Config
    from utils.model_utils import ModelAnalyzer


def fsdp_prepared_dir(cfg: "Config") -> str:
    if cfg.fsdp_prepared_dir:
        return cfg.fsdp_prepared_dir
    return os.path.join(cfg.cache_dir, "fsdp_prepared", cfg.model_name)


# ---------------------------------------------------------------------------
# cpu_master Phase A helpers — direct ports of utils/model_utils.py:60-226.
# ---------------------------------------------------------------------------


def _prepared_checkpoint_base_dir(cfg: "Config") -> str:
    """Where rotated/untied checkpoints live on disk.

    Mirrors ``utils/model_utils.py:181-185``. Defaults to ``cfg.cache_dir``
    when ``cfg.fsdp_prepared_dir`` is unset.
    """
    return cfg.fsdp_prepared_dir or os.path.join(cfg.cache_dir, "fsdp_meta_prepared")


def _prepared_rotated_checkpoint_dir(cfg: "Config") -> str:
    """Path to the per-(model, rotation, seed) rotated checkpoint cache.

    Mirrors ``utils/model_utils.py:188-198``. The cache key includes the
    optimised-rotation file basename (or ``"hadamard"``) and the seed so
    different rotation paths and seeds get different caches.
    """
    if cfg.optimized_rotation_path is not None:
        opt_tag = os.path.basename(str(cfg.optimized_rotation_path)).replace("/", "_")
    else:
        opt_tag = "hadamard"
    seed_tag = f"seed{int(cfg.seed)}"
    return os.path.join(
        _prepared_checkpoint_base_dir(cfg),
        "_prepared_checkpoints",
        f"{cfg.model_name}_rot_{opt_tag}_{seed_tag}",
    )


def _prepared_untied_checkpoint_dir(cfg: "Config") -> str:
    return os.path.join(
        _prepared_checkpoint_base_dir(cfg),
        "_prepared_checkpoints",
        f"{cfg.model_name}_untied",
    )


def _save_prepared_checkpoint(
    analyzer: "ModelAnalyzer", output_dir: str, cfg: "Config", meta: dict,
) -> None:
    """Atomically save ``analyzer.model`` to ``output_dir`` (rank0 caller).

    Mirrors ``utils/model_utils.py:83-103``. Writes to ``output_dir.tmp``
    first, then ``os.replace`` swaps it into place — a half-written cache
    will not be picked up by ``_prepared_checkpoint_ready`` (no ``_SUCCESS``
    sentinel). Caller must be rank 0.
    """
    from realq.utils import memory as mem_utils

    tmp_dir = f"{output_dir}.tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    analyzer.model.save_pretrained(
        tmp_dir,
        safe_serialization=True,
        max_shard_size=cfg.fsdp_max_shard_size,
    )
    analyzer.tokenizer.save_pretrained(tmp_dir)
    with open(os.path.join(tmp_dir, "realq_fsdp_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(tmp_dir, "_SUCCESS"), "w") as f:
        f.write("ok\n")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.replace(tmp_dir, output_dir)
    mem_utils.cleanup_memory()


def _prepared_checkpoint_ready(
    checkpoint_dir: str, cfg: "Config", *, kind: str, rotate: bool,
) -> bool:
    """True iff ``checkpoint_dir`` has a complete + matching cache.

    Mirrors ``utils/model_utils.py:137-154``. The kind/source/seed/rotate/
    optimized_rotation_path tuple is compared so a re-run with different
    parameters won't accidentally reuse the wrong cache.
    """
    success_path = os.path.join(checkpoint_dir, "_SUCCESS")
    meta_path = os.path.join(checkpoint_dir, "realq_fsdp_meta.json")
    config_path = os.path.join(checkpoint_dir, "config.json")
    if not (
        os.path.exists(success_path)
        and os.path.exists(meta_path)
        and os.path.exists(config_path)
    ):
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        return False
    return (
        meta.get("kind") == kind
        and meta.get("source_model") == cfg.model
        and int(meta.get("seed", -1)) == int(cfg.seed)
        and bool(meta.get("rotate")) is bool(rotate)
        and meta.get("optimized_rotation_path") == cfg.optimized_rotation_path
    )


def _build_rotated_checkpoint_on_rank0(cfg: "Config", rotated_dir: str) -> None:
    """Materialise + rotate exactly one full CPU model on rank0, then save.

    Mirrors ``utils/model_utils.py:106-134``. Caller MUST be rank 0; other
    ranks are blocked at the surrounding barrier. Peak CPU is 1×M (one
    full model). After save, the model is freed.
    """
    import transformers
    from realq.utils import memory as mem_utils
    from utils import rotation_utils
    from utils.model_utils import ModelAnalyzer

    logging.info(
        "[realq.fsdp] cpu_master Phase A: rank0 building rotated checkpoint at %s "
        "(only rank0 materialises the full CPU model for this preprocessing).",
        rotated_dir,
    )
    analyzer = ModelAnalyzer(cfg.model, cfg.seq_len)
    rotation_utils.fuse_layer_norms(analyzer)
    rotation_utils.rotate_model(cfg, analyzer)
    mem_utils.cleanup_memory()

    meta = {
        "kind": "rotated",
        "source_model": cfg.model,
        "seq_len": int(cfg.seq_len),
        "seed": int(cfg.seed),
        "rotate": True,
        "optimized_rotation_path": cfg.optimized_rotation_path,
        "transformers_version": transformers.__version__,
    }
    _save_prepared_checkpoint(analyzer, rotated_dir, cfg, meta)
    del analyzer
    mem_utils.cleanup_memory()


def _build_untied_checkpoint_on_rank0(cfg: "Config", untied_dir: str) -> None:
    """Build a checkpoint with explicit (untied) lm_head.weight.

    Mirrors ``utils/model_utils.py:157-178``. Needed when ``cfg.rotate=False``
    AND the source config has ``tie_word_embeddings=True`` — ``load_checkpoint_in_model``
    requires an explicit ``lm_head.weight`` key.
    """
    import transformers
    from realq.utils import memory as mem_utils
    from utils.model_utils import ModelAnalyzer

    logging.info(
        "[realq.fsdp] cpu_master Phase A: rank0 building untied checkpoint at %s "
        "(source ties word embeddings; explicit lm_head needed for sharded load).",
        untied_dir,
    )
    analyzer = ModelAnalyzer(cfg.model, cfg.seq_len)
    meta = {
        "kind": "untied",
        "source_model": cfg.model,
        "seq_len": int(cfg.seq_len),
        "seed": int(cfg.seed),
        "rotate": False,
        "optimized_rotation_path": cfg.optimized_rotation_path,
        "transformers_version": transformers.__version__,
    }
    _save_prepared_checkpoint(analyzer, untied_dir, cfg, meta)
    del analyzer
    mem_utils.cleanup_memory()


def _get_prepared_checkpoint_path(cfg: "Config") -> Tuple[str, bool]:
    """Resolve which checkpoint Phase B should load.

    Mirrors ``utils/model_utils.py:201-226``. Returns ``(path, is_rotated)``.

    Decision tree:
      cfg.rotate=False + source has untied embeddings ⇒ return cfg.model directly
      cfg.rotate=False + source has tied embeddings   ⇒ untied checkpoint
      cfg.rotate=True                                  ⇒ rotated checkpoint
    """
    if not cfg.rotate:
        src_config = AutoConfig.from_pretrained(cfg.model, trust_remote_code=True)
        if not getattr(src_config, "tie_word_embeddings", False):
            return cfg.model, False
        return _prepared_untied_checkpoint_dir(cfg), False
    return _prepared_rotated_checkpoint_dir(cfg), True


def _build_empty_model_from_config(checkpoint_path: str, cfg: "Config"):
    """Construct a meta-tensor model from a saved checkpoint's config.json.

    Mirrors ``utils/model_utils.py:261-275``. The model has full structure
    but every parameter is on ``device='meta'`` — ≈0 RAM. Used by Phase B
    (FSDP-shard the meta model, then load via broadcast) and Phase D
    (rank>0 skeleton).
    """
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM
    from utils.model_utils import _prepare_config_for_untied_lm_head

    config, process_word_embeddings = _prepare_config_for_untied_lm_head(checkpoint_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    model.tie_word_embeddings = process_word_embeddings
    model.seqlen = cfg.seq_len
    model.eval()
    return model


def _assert_aware_akv_disabled(cfg: "Config") -> None:
    """Defence-in-depth guard. ``Config.__post_init__`` already catches this
    at parse time; this catches programmatic misconfigurations."""
    if not cfg.cpu_master:
        return
    if cfg.act_quant_aware_gptq or cfg.k_cache_quant_aware_gptq:
        raise RuntimeError(
            f"cpu_master + aware AKV not supported in v1 "
            f"(act_quant_aware_gptq={cfg.act_quant_aware_gptq}, "
            f"k_cache_quant_aware_gptq={cfg.k_cache_quant_aware_gptq}). "
            f"See realq/TODO_CPU_MASTER.md for the relaxation plan."
        )


def _fsdp_shard_layers(analyzer: "ModelAnalyzer", cfg: "Config") -> None:
    """FSDP2-shard each transformer block AND the outer module.

    Used by the cpu_master path on a meta-tensor model — no ``.to(dev)`` is
    required because the model has no real storage yet. Sharding the outer
    module too matches the old ``_fsdp_shard_model_for_precompute``
    (model_utils.py:60-80); under cpu_master we never call ``save_pretrained``
    on the FSDP-wrapped model so the legacy ``_end_ptr`` issue does not apply.
    """
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        MixedPrecisionPolicy,
        fully_shard,
    )
    from torch.distributed.device_mesh import init_device_mesh

    from realq.parallel import env as parallel_env

    world = parallel_env.get_world_size()
    if world <= 1:
        logging.warning("[realq.fsdp] world_size=1; FSDP wrap is a no-op.")
    mesh = init_device_mesh("cuda", (world,))
    model = analyzer.model
    layers = analyzer.get_layers()
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    offload_policy = CPUOffloadPolicy(pin_memory=True) if cfg.fsdp_cpu_offload else None
    logging.info(
        "[realq.fsdp] cpu_master Phase B: sharding %d layers + outer model "
        "across world=%d, cpu_offload=%s",
        len(layers), world, cfg.fsdp_cpu_offload,
    )
    kwargs = {"mesh": mesh, "mp_policy": mp_policy}
    if offload_policy is not None:
        kwargs["offload_policy"] = offload_policy
    for layer in layers:
        fully_shard(layer, **kwargs)
    fully_shard(model, **kwargs)


# ---------------------------------------------------------------------------
# cpu_master Phase entry points — called from realq/pipeline.py.
# ---------------------------------------------------------------------------


def prepare_rotated_checkpoint(cfg: "Config") -> Tuple[str, bool]:
    """Phase A: rank0 builds + saves rotated/untied checkpoint; rank>0 barriers.

    Returns ``(path, is_rotated)``. Cache hit ⇒ no rotate work.

    Decision tree (mirrors ``_get_fsdp_meta_checkpoint_path`` in old code):
      cfg.rotate=True                                       ⇒ rotated cache (built if missing)
      cfg.rotate=False + source has tie_word_embeddings    ⇒ untied cache (built if missing)
      cfg.rotate=False + source has untied embeddings      ⇒ cfg.model returned directly
    """
    from realq.parallel import env as parallel_env

    _assert_aware_akv_disabled(cfg)

    if cfg.rotate:
        rotated_dir = _prepared_rotated_checkpoint_dir(cfg)
        if parallel_env.is_main() and not _prepared_checkpoint_ready(
            rotated_dir, cfg, kind="rotated", rotate=True,
        ):
            _build_rotated_checkpoint_on_rank0(cfg, rotated_dir)
        if dist.is_available() and dist.is_initialized():
            parallel_env.barrier()
        return rotated_dir, True

    src_config = AutoConfig.from_pretrained(cfg.model, trust_remote_code=True)
    if not getattr(src_config, "tie_word_embeddings", False):
        # No preprocessing needed — load_checkpoint_in_model can read cfg.model directly.
        return cfg.model, False

    untied_dir = _prepared_untied_checkpoint_dir(cfg)
    if parallel_env.is_main() and not _prepared_checkpoint_ready(
        untied_dir, cfg, kind="untied", rotate=False,
    ):
        _build_untied_checkpoint_on_rank0(cfg, untied_dir)
    if dist.is_available() and dist.is_initialized():
        parallel_env.barrier()
    return untied_dir, False


def load_meta_for_precompute(
    cfg: "Config", checkpoint_path: str,
) -> "ModelAnalyzer":
    """Phase B: meta init + fully_shard + sharded broadcast load.

    Returns an analyzer whose model has FSDP-sharded params populated from
    ``checkpoint_path``. CPU peak during the load is ≤ one safetensors shard
    on rank0 (default 5GB); rank>0 stays ≈0.

    Wrappers are NOT installed here — the caller does that AFTER load,
    because ``load_checkpoint_in_model`` matches keys against the saved
    state_dict (vanilla ``xxx.weight``), and wrapping replaces the key
    layout (``xxx.module.weight``).
    """
    from accelerate.utils import load_checkpoint_in_model
    from utils.model_utils import ModelAnalyzer

    model = _build_empty_model_from_config(checkpoint_path, cfg)
    analyzer = ModelAnalyzer(
        model,
        cfg.seq_len,
        tokenizer_source=checkpoint_path,
        skip_state_dict=True,
    )
    _fsdp_shard_layers(analyzer, cfg)  # NO .to(dev) — model is meta
    logging.info(
        "[realq.fsdp] cpu_master Phase B: loading checkpoint %s into FSDP2 shards "
        "(broadcast_from_rank0=True, cpu_offload=%s)",
        checkpoint_path, cfg.fsdp_cpu_offload,
    )
    load_checkpoint_in_model(
        model,
        checkpoint_path,
        dtype=torch.bfloat16,
        strict=False,
        full_state_dict=True,
        broadcast_from_rank0=True,
    )
    model.eval()
    model._realq_fsdp_wrapped = True
    return analyzer


def rebuild_asymmetric_for_quant(
    cfg: "Config", checkpoint_path: str,
) -> "ModelAnalyzer":
    """Phase D: drop FSDP analyzer; rank0 reloads full CPU; rank>0 builds meta skeleton.

    Caller MUST have already done ``del old_analyzer; cleanup_memory()`` so
    the FSDP DTensor backrefs are released.

    Sets ``_realq_cpu_master = True`` (consumed by ``CpuMasterLayerManager``)
    AND ``_gptqplus_prepared_checkpoint_path`` (consumed by
    ``utils.eval_utils.get_ref_logits`` to compute the same ref-logits
    cache key as the legacy path — see realq/REFACTOR_NOTES.md).
    """
    from realq.parallel import env as parallel_env
    from utils.model_utils import ModelAnalyzer

    if parallel_env.is_main():
        analyzer = ModelAnalyzer(
            checkpoint_path, cfg.seq_len, tokenizer_source=checkpoint_path,
        )
    else:
        model = _build_empty_model_from_config(checkpoint_path, cfg)
        analyzer = ModelAnalyzer(
            model, cfg.seq_len,
            tokenizer_source=checkpoint_path,
            skip_state_dict=True,
        )
    analyzer.model._realq_cpu_master = True
    # Match the legacy ref_logits cache key (eval_utils.py:115 reads this).
    analyzer.model._gptqplus_prepared_checkpoint_path = checkpoint_path
    analyzer.model._gptqplus_checkpoint_is_rotated = bool(cfg.rotate)
    return analyzer


# ---------------------------------------------------------------------------
# Legacy fsdp=True path (cpu_master=False). Kept intact for Commit 1's A/B
# verification. Removed in Commit 2.
# ---------------------------------------------------------------------------


def fsdp_wrap_for_precompute(analyzer: "ModelAnalyzer", cfg: "Config") -> None:
    """FSDP2-shard each transformer block + the outer module.

    Mirrors old ``gptq_plus_utils.py`` lines 4076-4103. Precompute itself
    is unchanged: when the model is FSDP2-wrapped, parameter access during
    forward/backward implicitly all-gathers the relevant shard, then
    re-shards on exit.

    The model MUST be on the current CUDA device before this call when
    ``cfg.fsdp_cpu_offload=False`` — FSDP2 keeps the post-shard parameters
    where it found them, and a CPU-resident DTensor cannot run NCCL
    all-gather (which is what ``save_pretrained`` triggers via the
    ``_end_ptr`` data-pointer probe). cpu_offload=True moves shards to CPU
    automatically; pre-move them to GPU first anyway so the wrap itself runs
    on GPU memory.
    """
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        MixedPrecisionPolicy,
        fully_shard,
    )
    from torch.distributed.device_mesh import init_device_mesh

    from realq.parallel import env as parallel_env

    world = parallel_env.get_world_size()
    if world <= 1:
        logging.warning("[realq.fsdp] world_size=1; FSDP wrap is a no-op.")
    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    analyzer.model.to(dev)
    mesh = init_device_mesh("cuda", (world,))
    model = analyzer.model
    layers = analyzer.get_layers()
    mp_policy = MixedPrecisionPolicy(
        param_dtype=next(iter(model.parameters())).dtype,
        reduce_dtype=torch.float32,
    )
    offload_policy = CPUOffloadPolicy(pin_memory=True) if cfg.fsdp_cpu_offload else None
    logging.info(
        "[realq.fsdp] sharding %d layers across world=%d, cpu_offload=%s",
        len(layers), world, cfg.fsdp_cpu_offload,
    )
    for layer in layers:
        kwargs = {"mesh": mesh, "mp_policy": mp_policy}
        if offload_policy is not None:
            kwargs["offload_policy"] = offload_policy
        fully_shard(layer, **kwargs)
    # Intentionally DO NOT ``fully_shard(model)``: that wraps the outer
    # module's embed/norm/lm_head as DTensors too, which then breaks
    # ``save_pretrained``'s ``_end_ptr`` data-pointer probe (DTensor view
    # triggers an implicit all-gather that NCCL cannot service for CPU
    # tensors). Sharding only the transformer blocks is enough memory-wise:
    # those are the dominant params on big models and the embed/norm/head
    # already replicate on every rank without harming peak memory.
    model._realq_fsdp_wrapped = True


def save_post_precompute_checkpoint(analyzer: "ModelAnalyzer", cfg: "Config") -> str:
    """Save the (rotated, FSDP-sharded) model to disk; rank 0 writes only.

    Workaround for FSDP2 + ``save_pretrained`` interaction:
    ``transformers``'s ``_end_ptr`` calls ``tensor.view(-1)[-1].data_ptr()``
    which triggers a DTensor redistribute. The redistribute path calls
    ``funcol.all_gather_tensor`` and that fails with "No backend type
    associated with device type cpu" because by then the helper has already
    requested a CPU view. Avoid the issue entirely by manually full-tensoring
    each parameter into a CPU dict on rank 0 and ``torch.save``-ing it.

    Cache layout::

        cfg.fsdp_prepared_dir/
          state_dict.pt          # {param_name: cpu_tensor, ...}
          tokenizer/             # tokenizer.save_pretrained
          config.json            # model.config.to_json
          realq_meta.json
          _SUCCESS
    """
    from torch.distributed.tensor import DTensor

    from realq.parallel import env as parallel_env

    out_dir = fsdp_prepared_dir(cfg)
    tmp_dir = f"{out_dir}.tmp"
    if parallel_env.is_main():
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
    parallel_env.barrier()

    # Gather every parameter to a single replica and stage to CPU on rank 0.
    # The saved keys MUST match a vanilla (un-wrapped) ``from_pretrained``
    # state-dict layout — the reload path constructs a fresh
    # ``ModelAnalyzer(SOURCE_MODEL)`` that doesn't have ``ActQuantWrapper``
    # sites yet, so any ``.module.`` infix introduced by the wrapper has to
    # be stripped at save time.
    cpu_state: dict = {}
    dev = torch.device(f"cuda:{torch.cuda.current_device()}")

    def _normalise_name(n: str) -> str:
        # ``ActQuantWrapper`` lifts the wrapped Linear under a `.module`
        # attribute; saved keys would look like
        # ``model.layers.0.self_attn.q_proj.module.weight``. The reload
        # target is the un-wrapped ``model.layers.0.self_attn.q_proj.weight``.
        return n.replace(".module.weight", ".weight").replace(".module.bias", ".bias")

    for name, p in analyzer.model.named_parameters():
        full = p.data
        if isinstance(full, DTensor):
            if full.to_local().device.type != "cuda":
                full = DTensor.from_local(
                    full.to_local().to(dev),
                    full.device_mesh,
                    full.placements,
                )
            full = full.full_tensor()
        if parallel_env.is_main():
            cpu_state[_normalise_name(name)] = full.detach().to("cpu")
    for name, b in analyzer.model.named_buffers():
        full = b.data
        if isinstance(full, DTensor):
            if full.to_local().device.type != "cuda":
                full = DTensor.from_local(
                    full.to_local().to(dev),
                    full.device_mesh,
                    full.placements,
                )
            full = full.full_tensor()
        if parallel_env.is_main():
            # Skip ActQuantizer / out_quantizer scratch buffers (maxq, scale,
            # zero) — they live inside the wrapper and have no counterpart in
            # the un-wrapped target model. Leaving them in causes
            # ``load_state_dict(strict=False)`` to log dozens of "unexpected"
            # warnings; more importantly they don't restore any model weight.
            if "quantizer" in name:
                continue
            cpu_state[_normalise_name(name)] = full.detach().to("cpu")
    parallel_env.barrier()

    if parallel_env.is_main():
        torch.save(cpu_state, os.path.join(tmp_dir, "state_dict.pt"))
        # Save the model config + tokenizer so a vanilla ``from_pretrained``
        # path could pick the dir up later if needed.
        analyzer.model.config.save_pretrained(tmp_dir)
        analyzer.tokenizer.save_pretrained(tmp_dir)
        meta = {
            "kind": "post_precompute",
            "source_model": cfg.model,
            "rotate": bool(cfg.rotate),
            "seed": int(cfg.seed),
        }
        with open(os.path.join(tmp_dir, "realq_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        with open(os.path.join(tmp_dir, "_SUCCESS"), "w") as f:
            f.write("ok\n")
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.replace(tmp_dir, out_dir)
    parallel_env.barrier()
    logging.info("[realq.fsdp] saved post-precompute checkpoint to %s", out_dir)
    return out_dir


def reload_on_cpu(analyzer: "ModelAnalyzer", checkpoint_dir: str) -> "ModelAnalyzer":
    """Drop the FSDP-wrapped model and rebuild a CPU master from disk.

    The new analyzer is constructed via ``ModelAnalyzer(SOURCE_MODEL, ...)``
    and then has its weights overwritten from ``state_dict.pt`` (so we get
    the rotated, post-precompute weights instead of the original
    safetensors). Returns the new analyzer; the caller should replace its
    own reference to drop DTensor backrefs.
    """
    from utils import memory_utils
    from utils.model_utils import ModelAnalyzer

    seq_len = analyzer.seqlen if hasattr(analyzer, "seqlen") else 2048
    source_model = analyzer.model.config._name_or_path or checkpoint_dir
    del analyzer
    memory_utils.cleanup_memory()

    # Construct a fresh CPU model from the original (unrotated) safetensors,
    # then override every parameter with the saved post-precompute weights.
    # We cannot ``from_pretrained(checkpoint_dir)`` directly because we only
    # saved a flat torch state dict, not a sharded safetensors layout.
    new_analyzer = ModelAnalyzer(source_model, seq_len)
    sd = torch.load(
        os.path.join(checkpoint_dir, "state_dict.pt"),
        map_location="cpu",
        weights_only=False,
    )
    missing, unexpected = new_analyzer.model.load_state_dict(sd, strict=False)
    if missing:
        logging.warning("[realq.fsdp] reload missing keys (%d): %s", len(missing), missing[:5])
    if unexpected:
        logging.warning("[realq.fsdp] reload unexpected keys (%d): %s", len(unexpected), unexpected[:5])
    return new_analyzer
