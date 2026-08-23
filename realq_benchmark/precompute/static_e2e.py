"""Static end-to-end saliency + Fisher precompute.

Runs once per (model, dataset, seed, world_size) tuple — see ``cache.py`` for
the cache key. Each rank handles a contiguous shard of calibration samples
(``nsamples / world_size`` per rank). Per-rank shard:

    for batch in rank_local_shard:
        logits     = model(batch)
        labels     = Categorical(softmax(logits)).sample()        # deterministic per global sample id
        loss       = cross_entropy(logits, labels, reduction='sum') * LOSS_GRAD_SCALE
        loss.backward()
        # forward hooks on per-(layer, module) outputs record `sum_g(grad²)`
        # forward hooks on per-layer outputs accumulate `g g^T`

After the loop:

* Saliency stays per-rank — it's per-sample data — concat over batches gives
  ``(N_local, T, num_groups)`` per (layer, module).
* Fisher is a global statistic — all-reduce the per-rank GPU sums then
  divide by the global token count, store one ``(H, H)`` bf16 matrix per
  layer.

Both end up in the same per-rank cache file.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from realq_benchmark.parallel import env as parallel_env
from realq_benchmark.parallel import reduce as parallel_reduce
from realq_benchmark.precompute import cache as cache_mod
from realq_benchmark.precompute import hooks as hooks_mod
from realq_benchmark.precompute.labels import deterministic_categorical_labels
from realq_benchmark.utils import memory as mem_utils
from realq_benchmark.utils import nvtx
from utils import data_utils, dist_utils

if TYPE_CHECKING:
    from realq_benchmark.config import Config
    from utils.model_utils import ModelAnalyzer


# Module names inside one transformer layer that we collect saliency for.
# These are the seven linears that GPTQ later quantises. Order is irrelevant
# for correctness but kept stable so cache contents don't drift.
_QUANT_MODULE_NAMES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


@dataclass
class StaticStats:
    """Output of static precompute.

    ``saliency[layer_idx][module_name]`` — ``(N_local, T, num_groups)`` cpu fp32.
    Per-sample data, rank-local (each rank holds its own shard).

    ``fisher[layer_idx]`` — ``(H, H)`` cpu bf16. Global statistic, all-reduced
    and token-count-normalised; identical on every rank.
    """
    saliency: list[dict[str, torch.Tensor]]
    fisher: list[torch.Tensor]


def _find_layer_modules(layer: nn.Module) -> dict[str, nn.Module]:
    """Resolve the seven quantisable linears inside one transformer layer.

    Walks the layer's named submodules so we pick up the underlying
    ``nn.Linear`` even when it is wrapped by ``ActQuantWrapper`` after
    rotate. Hooking the wrapped Linear is correct — its forward output is
    what gradients flow into during backward, which is the quantity saliency
    is defined against.
    """
    name_to_mod = dict(layer.named_modules())
    out: dict[str, nn.Module] = {}
    for name in _QUANT_MODULE_NAMES:
        mod = name_to_mod.get(name)
        if mod is None:
            raise RuntimeError(
                f"static_e2e: layer is missing expected submodule {name!r}; "
                f"either the model arch is unsupported or rotate has restructured the layer."
            )
        # If wrapped (rotate path), the underlying linear lives at `.module`.
        if hasattr(mod, "module") and isinstance(mod.module, nn.Linear):
            mod = mod.module
        if not isinstance(mod, nn.Linear):
            raise RuntimeError(
                f"static_e2e: expected nn.Linear at {name!r}, got {type(mod).__name__} "
                f"(submodules: {list(dict(mod.named_modules()).keys())[:10]})."
            )
        out[name] = mod
    return out


def _freeze_model_params(model: nn.Module) -> list[tuple[nn.Parameter, bool]]:
    """Freeze every param so backward does not allocate per-param grad
    buffers — the static precompute hooks read activation gradients only,
    weight gradients are never consumed. On Llama2-70B this skips a 280 GB
    fp32 grad buffer (8 GB on 4B, ~2.4 GB on Qwen3-0.6B). Returns the prior
    requires_grad flags so the caller can restore them in a finally clause.

    Pair this with a forward pre-hook on layer[0] (see
    ``_register_kick_off_hook``) — with all params frozen the embedding
    output has ``requires_grad=False`` so autograd would not build any
    graph; the hook re-enters the graph at the first transformer layer's
    input.
    """
    saved = [(p, p.requires_grad) for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    return saved


def _restore_requires_grad(saved: list[tuple[nn.Parameter, bool]]) -> None:
    for p, flag in saved:
        p.requires_grad_(flag)


def _register_kick_off_hook(layer0: nn.Module) -> torch.utils.hooks.RemovableHandle:
    """Forward pre-hook on the first transformer layer: flip its input
    tensor to ``requires_grad=True`` so autograd builds a graph from layer 0
    onward even though all model params are frozen."""

    def _hook(_module, inputs):
        if isinstance(inputs, tuple) and len(inputs) > 0 and torch.is_tensor(inputs[0]):
            inputs[0].requires_grad_(True)

    return layer0.register_forward_pre_hook(_hook)


def _shard_local_calibration(trainloader: list, rank: int, world: int) -> list[torch.Tensor]:
    """Slice the global calibration list into the per-rank shard.

    Returns a list of 1D token tensors of length ``seq_len``. The per-rank
    sample count is ``len(trainloader) // world`` (validated by
    ``shard_slice``).
    """
    sl = dist_utils.shard_slice(len(trainloader), rank=rank, world=world)
    return [trainloader[i] for i in range(sl.start, sl.stop)]


def _all_ranks_have_cache(local_hit: bool, world: int) -> bool:
    """Return true only when every distributed rank has its local cache file.

    Static precompute contains several collectives.  It is therefore invalid
    for a rank with a local hit to return while a rank with a miss recomputes:
    the latter would deadlock at the first Fisher all-reduce.  A MIN reduction
    makes all-hit the only early-return case.  For NCCL the control tensor must
    live on the rank's current CUDA device; CPU backends use a CPU tensor.
    """
    if world <= 1:
        return local_hit
    if not parallel_env.is_dist_available_and_initialized():
        raise RuntimeError(
            "static_e2e cache consensus requires an initialized process group "
            f"when world_size={world}"
        )
    backend = str(dist.get_backend()).lower()
    if "nccl" in backend:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")
    hit = torch.tensor([int(local_hit)], dtype=torch.int32, device=device)
    dist.all_reduce(hit, op=dist.ReduceOp.MIN)
    return bool(hit.item())


def run(cfg: "Config", analyzer: "ModelAnalyzer") -> StaticStats:
    """Top-level entry. Returns the StaticStats; also writes to disk if
    ``cfg.static_cache_path`` is set."""
    rank = parallel_env.get_rank()
    world = parallel_env.get_world_size()

    # 1. Cache lookup.
    if cfg.static_cache_path:
        with nvtx.nvtx_range("precompute.cache_lookup"):
            key = cache_mod.build_cache_key(cfg, world)
            cached = cache_mod.try_load(cfg.static_cache_path, key, world, rank)
            local_hit = cached is not None
            all_hit = _all_ranks_have_cache(local_hit, world)
            if all_hit:
                logging.info(
                    "[realq.precompute] cache hit (rank %d): %s",
                    rank, cache_mod.cache_path(cfg.static_cache_path, key, world, rank),
                )
                # The distributed rotation broadcast materialises every
                # parameter on the local CUDA device.  A cache miss reaches
                # the normal teardown below, which moves the non-FSDP model
                # back to CPU before layer-streamed quantisation; an early
                # cache hit must leave the model in the same state.
                if not getattr(cfg, "fsdp", False):
                    analyzer.model.cpu()
                    mem_utils.cleanup_memory()
                return StaticStats(saliency=cached["saliency"], fisher=cached["fisher"])
            if local_hit:
                logging.warning(
                    "[realq.precompute] rank %d has a cache file but at least "
                    "one peer does not; all ranks will recompute Stage 0.",
                    rank,
                )
            elif world > 1:
                logging.info(
                    "[realq.precompute] cache miss on rank %d; all ranks will "
                    "recompute Stage 0.",
                    rank,
                )
            # A rank-local hit can be very large.  Once consensus rejects the
            # early return, release it before allocating Stage-0 activations.
            cached = None
            if getattr(cfg, "require_static_cache_hit", False):
                expected_path = cache_mod.cache_path(
                    cfg.static_cache_path,
                    key,
                    world,
                    rank,
                )
                raise RuntimeError(
                    "[realq.precompute] required static cache hit was not "
                    f"available on every rank; rank={rank}, "
                    f"expected={expected_path}. Run the dedicated "
                    "realq_static precompute producer successfully before "
                    "launching sweep consumers."
                )

    # 2. Calibration data, sharded per rank.
    with nvtx.nvtx_range("precompute.load_data"):
        tokens_save_path = cfg.tokens_cache_file
        if tokens_save_path is None and cfg.tokens_cache_path:
            # data_utils.get_tokens expects a file path; build one keyed by the
            # arguments that change tokenisation output.
            import os as _os
            tokens_save_path = _os.path.join(
                cfg.tokens_cache_path,
                f"{cfg.model_name}_{cfg.dataset}_train_n{cfg.nsamples}_sl{cfg.seq_len}_seed{cfg.seed}.pt",
            )
        trainloader = data_utils.get_tokens(
            cfg.dataset, "train", analyzer.tokenizer,
            cfg.seq_len, cfg.nsamples,
            tokens_save_path, cfg.seed,
        )
        # get_tokens returns 1D LongTensors of length seq_len.
        rank_samples = _shard_local_calibration(trainloader, rank, world)
        n_local = len(rank_samples)
        if n_local == 0:
            raise RuntimeError(
                f"[realq.precompute] rank {rank} got 0 calibration samples — bump nsamples."
            )

        if cfg.global_loss_bsz % world != 0:
            raise ValueError(
                f"global_loss_bsz ({cfg.global_loss_bsz}) must be divisible by "
                f"world_size ({world})."
            )
        local_bsz = max(1, cfg.global_loss_bsz // world)

    # 3. Move model to GPU and switch to grad-enabled mode.
    with nvtx.nvtx_range("precompute.model_setup"):
        dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        model = analyzer.model
        use_cache = model.config.use_cache
        model.config.use_cache = False
        model.to(dev)
        # model.eval(): match old gptq_plus_utils.py:4631 line-for-line. Even
        # though Qwen3 has no dropout / BatchNorm so train vs eval should be a
        # no-op, the precompute saliency on Qwen3-0.6B comes out off by ~0.3-
        # 0.5% per batch under model.train() vs model.eval() (cuBLAS picks a
        # different attention kernel under model.training=True), and that off-
        # by-0.5% saliency cascades into a ~5e-2 max-diff in the quantised
        # weight after Hessian-weighted block update. Forcing eval mode (which
        # is what old code does) restores bit-exactness with the legacy
        # reference at lr=0 on 1-GPU AND 2-GPU configurations.
        model.eval()
        prev_requires_grad = _freeze_model_params(model)

        # 4. Resolve modules per layer + attach hooks.
        layers = analyzer.get_layers()
        layer_modules = [_find_layer_modules(layer) for layer in layers]
        sal_mgr = hooks_mod.SaliencyHookManager(
            num_groups=cfg.num_groups,
            clip_percentile=cfg.saliency_clip_percentile,
        )
        fisher_mgr = hooks_mod.FisherHookManager()
        sal_mgr.attach(layer_modules)
        fisher_mgr.attach(layers)
        kick_off_handle = _register_kick_off_hook(layers[0])

    # 5. Forward + backward loop over the per-rank shard.
    iterator = range(0, n_local, local_bsz)
    show_progress = parallel_env.is_main()
    if show_progress:
        iterator = tqdm(iterator, desc="Static precompute", ncols=100)
    try:
        with nvtx.nvtx_range("precompute.fwd_bwd_loop"):
            batch_counter = 0
            for local_start in iterator:
                with nvtx.nvtx_range(f"precompute.batch_{batch_counter}"):
                    local_end = min(local_start + local_bsz, n_local)
                    batch_samples = rank_samples[local_start:local_end]
                    input_ids = torch.stack([s.view(-1) for s in batch_samples], dim=0).to(dev)
                    B, T = input_ids.shape

                    # Per-sample global ids. Sharding is contiguous, so global id is
                    # just the local id offset by the rank's shard start.
                    shard = dist_utils.shard_slice(cfg.nsamples, rank=rank, world=world)
                    global_indices = [shard.start + local_start + i for i in range(B)]

                    model.zero_grad(set_to_none=True)
                    with nvtx.nvtx_range("precompute.forward"):
                        outputs = model(input_ids=input_ids)
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

                    with nvtx.nvtx_range("precompute.loss_compute"):
                        if cfg.grad_hessian_topk > 0:
                            # Reduce the vocab axis for the Fisher backward; the topk
                            # picks per-position teacher tokens, student goes through
                            # the same indices. NOTE: must include the sampled label
                            # in the topk subset, but using teacher's top-k and then
                            # sampling from teacher inside that subset is consistent.
                            teacher_logits, indices = logits.detach().topk(
                                cfg.grad_hessian_topk, dim=-1, sorted=False
                            )
                            student_logits = logits.gather(-1, indices)
                            labels = deterministic_categorical_labels(
                                teacher_logits, global_indices, base_seed=0,
                            )
                        else:
                            student_logits = logits
                            labels = deterministic_categorical_labels(
                                logits.detach(), global_indices, base_seed=0,
                            )

                        loss = F.cross_entropy(
                            student_logits.reshape(-1, student_logits.size(-1)),
                            labels.reshape(-1),
                            reduction="sum",
                        )
                    with nvtx.nvtx_range("precompute.backward"):
                        (loss * hooks_mod.LOSS_GRAD_SCALE).backward()
                    fisher_mgr.add_token_count(B * T)
                    # Free the autograd graph + per-step grad memory before next iter.
                    model.zero_grad(set_to_none=True)
                    del outputs, logits, student_logits, labels, loss
                    batch_counter += 1
    finally:
        sal_mgr.remove()
        fisher_mgr.remove()
        kick_off_handle.remove()
        _restore_requires_grad(prev_requires_grad)

    # 6. Finalize.
    with nvtx.nvtx_range("precompute.finalize_saliency"):
        saliency = sal_mgr.finalize()
    with nvtx.nvtx_range("precompute.fisher_allreduce"):
        fisher_sums, local_tokens = fisher_mgr.finalize()
        total_tokens = float(dist_utils.allreduce_sum_scalar(local_tokens))
        fisher_out: list[torch.Tensor] = []
        for layer_idx, fisher_sum in enumerate(fisher_sums):
            if fisher_sum is None:
                raise RuntimeError(
                    f"[realq.precompute] layer {layer_idx} produced no Fisher accumulator."
                )
            with nvtx.nvtx_range(f"precompute.fisher_allreduce.layer_{layer_idx}"):
                parallel_reduce.allreduce_sum_(fisher_sum)
                fisher_out.append((fisher_sum / total_tokens).to(torch.bfloat16).cpu())
        del fisher_sums

    # 7. Restore model state, free memory.
    with nvtx.nvtx_range("precompute.teardown"):
        model.config.use_cache = use_cache
        model.eval()
        model.cpu()
        mem_utils.cleanup_memory()

    static = StaticStats(saliency=saliency, fisher=fisher_out)

    # 8. Cache write.
    if cfg.static_cache_path:
        with nvtx.nvtx_range("precompute.cache_write"):
            path = cache_mod.save(
                cfg.static_cache_path, key, world, rank,
                {"saliency": static.saliency, "fisher": static.fisher},
            )
            logging.info("[realq.precompute] wrote rank-%d cache → %s", rank, path)

    return static
