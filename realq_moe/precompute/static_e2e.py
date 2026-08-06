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

from collections.abc import Mapping
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from realq_moe import model_adapter
from realq_moe.parallel import env as parallel_env
from realq_moe.parallel import reduce as parallel_reduce
from realq_moe.precompute import cache as cache_mod
from realq_moe.precompute import hooks as hooks_mod
from realq_moe.precompute.labels import deterministic_categorical_labels
from realq_moe.precompute import routed_stats
from realq_moe.utils import nvtx
from utils import data_utils, dist_utils

if TYPE_CHECKING:
    from realq_moe.config import Config
    from utils.model_utils import ModelAnalyzer


@dataclass
class StaticStats:
    """Output of static precompute.

    ``saliency[layer_idx][module_name]`` — dense modules use
    ``(N_local,T,G)`` while routed experts use ragged ``(A_local,1,G)`` CUDA
    fp32. Per-sample/assignment data is rank-local.

    ``fisher[layer_idx]`` — ``(H, H)`` CUDA bf16. Global statistic, all-reduced
    and token-count-normalised; identical on every rank.
    """
    saliency: list[dict[str, torch.Tensor]]
    fisher: list[torch.Tensor]
    routes: list[Mapping[int, Mapping[str, torch.Tensor]]] = field(
        default_factory=list
    )
    expert_global_assignment_counts: list[torch.Tensor] = field(
        default_factory=list
    )
    expert_global_coverage: list[dict[str, torch.Tensor]] = field(
        default_factory=list
    )


def _assert_static_stats_resident(
    static: StaticStats,
    device: torch.device,
    *,
    source: str,
) -> None:
    """Fail closed if a Stage-0 payload crosses the GPU boundary."""

    expected = torch.device(device)
    layer_count = len(static.routes)
    fields = (
        ("saliency", static.saliency),
        ("fisher", static.fisher),
        (
            "expert_global_assignment_counts",
            static.expert_global_assignment_counts,
        ),
        ("expert_global_coverage", static.expert_global_coverage),
    )
    mismatched_lengths = {
        name: len(values)
        for name, values in fields
        if len(values) != layer_count
    }
    if mismatched_lengths:
        raise RuntimeError(
            f"{source} Stage-0 payload has {layer_count} route layers but "
            f"mismatched fields {mismatched_lengths}."
        )

    def require_tensor(
        tensor: torch.Tensor,
        *,
        label: str,
    ) -> None:
        if not torch.is_tensor(tensor):
            raise RuntimeError(f"{source} {label} is not a tensor.")
        if tensor.device != expected:
            raise RuntimeError(
                f"{source} {label} must remain on {expected}, got "
                f"{tensor.device}."
            )

    for layer_idx in range(layer_count):
        for name, tensor in static.saliency[layer_idx].items():
            require_tensor(
                tensor,
                label=f"saliency[{layer_idx}][{name!r}]",
            )
        require_tensor(
            static.fisher[layer_idx],
            label=f"fisher[{layer_idx}]",
        )

        layer_routes = static.routes[layer_idx]
        if layer_routes:
            if not isinstance(
                layer_routes, routed_stats.PackedLayerRoutes
            ):
                raise RuntimeError(
                    f"{source} routes[{layer_idx}] is not canonical packed "
                    "CSR."
                )
            layer_routes.validate()
            layer_routes.assert_resident_on(expected)
        elif not isinstance(layer_routes, dict):
            raise RuntimeError(
                f"{source} empty routes[{layer_idx}] must be a dense-layer "
                "sentinel dict."
            )

        require_tensor(
            static.expert_global_assignment_counts[layer_idx],
            label=f"expert_global_assignment_counts[{layer_idx}]",
        )
        coverage = static.expert_global_coverage[layer_idx]
        for name, tensor in coverage.items():
            require_tensor(
                tensor,
                label=f"expert_global_coverage[{layer_idx}][{name!r}]",
            )


def _find_layer_modules(layer: nn.Module) -> dict[str, nn.Module]:
    """Resolve every quantisable linear inside one transformer layer.

    Walks the layer's named submodules so we pick up the underlying
    ``nn.Linear`` even when it is wrapped by ``ActQuantWrapper`` after
    rotate. Hooking the wrapped Linear is correct — its forward output is
    what gradients flow into during backward, which is the quantity saliency
    is defined against.
    """
    out: dict[str, nn.Module] = {}
    for name in model_adapter.quantizable_linear_paths(layer):
        out[name] = model_adapter.resolve_linear(layer, name)
    return out


def _expert_module_indices(layer: nn.Module) -> dict[str, int]:
    if not model_adapter.is_sparse_moe_layer(layer):
        return {}
    return {
        path: expert_idx
        for expert_idx, _projection, path
        in model_adapter.iter_expert_projection_paths(layer)
    }


def _raise_if_any_rank_failed_validation(
    local_error: str | None,
    *,
    context: str,
) -> None:
    """Turn a rank-local validation failure into a synchronized failure.

    A lone rank must not raise immediately while peers enter the next packed
    saliency collective.  Every rank first participates in this one-bit
    reduction, then all ranks raise together when any local payload is bad.
    """

    world = parallel_env.get_world_size()
    if world <= 1:
        if local_error is not None:
            raise RuntimeError(f"{context}: {local_error}")
        return
    if not parallel_env.is_dist_available_and_initialized():
        raise RuntimeError(
            f"{context}: world_size={world} requires an initialized process "
            "group."
        )
    backend = str(dist.get_backend()).lower()
    device = (
        torch.device(f"cuda:{torch.cuda.current_device()}")
        if "nccl" in backend
        else torch.device("cpu")
    )
    failed = torch.tensor(
        [int(local_error is not None)],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if bool(failed.item()):
        detail = local_error or "a peer rank reported an invalid local payload"
        raise RuntimeError(f"{context}: {detail}")


def _validate_teacher_coverage(
    cfg: "Config",
    expert_global_coverage: list[dict[str, torch.Tensor]],
    *,
    source: str,
) -> None:
    """Apply the frozen teacher-route gate to fresh and cached Stage-0 data.

    Coverage thresholds do not change the numerical cache payload, so they are
    intentionally absent from the cache key.  They must therefore be evaluated
    every time a payload is consumed, including an early cache hit.  A
    ``quant_stop_layer`` profile validates only the layers it will actually
    quantize; later-layer route statistics remain in the GPU payload for
    Fisher/slide use but cannot make an earlier-layer experiment fail.
    """

    threshold_fields = (
        ("assignment_count", cfg.moe_min_expert_assignments),
        ("unique_token_count", cfg.moe_min_expert_unique_tokens),
        ("unique_sample_count", cfg.moe_min_expert_unique_samples),
    )
    for layer_idx, coverage in enumerate(expert_global_coverage):
        if (
            cfg.quant_stop_layer is not None
            and layer_idx > cfg.quant_stop_layer
        ):
            break
        if not coverage:
            continue
        observed = torch.stack(
            [coverage[field].to(dtype=torch.int64) for field, _ in threshold_fields],
            dim=1,
        )
        thresholds = torch.tensor(
            [threshold for _, threshold in threshold_fields],
            dtype=torch.int64,
            device=observed.device,
        )
        zero_routed = observed[:, 0] == 0
        inconsistent_zero = zero_routed & (observed[:, 1:] != 0).any(dim=1)
        if bool(inconsistent_zero.any().item()):
            expert_ids = (
                inconsistent_zero.nonzero(as_tuple=False).flatten().tolist()
            )
            raise RuntimeError(
                "REAL-Q MoE teacher coverage is inconsistent at "
                f"layer={layer_idx}: zero-assignment experts have nonzero "
                f"token/sample counts; experts={expert_ids}."
            )
        # The only exception to the fail-closed coverage gate is an expert
        # with exactly zero global assignments.  It has no routed Hessian and
        # is quantized by deterministic grouped RTN in Stage 1.
        failing = (
            (observed < thresholds.unsqueeze(0)).any(dim=1)
            & ~zero_routed
        )
        if bool(failing.any().item()):
            expert_ids = failing.nonzero(as_tuple=False).flatten().tolist()
            details = {
                int(expert_idx): {
                    field: int(coverage[field][expert_idx].item())
                    for field, _ in threshold_fields
                }
                for expert_idx in expert_ids
            }
            raise RuntimeError(
                "REAL-Q MoE teacher routes failed the accepted fail-closed "
                f"coverage gate from {source} at layer={layer_idx}; "
                f"experts={details}, thresholds={dict(threshold_fields)}."
            )
        if bool(zero_routed.any().item()):
            expert_ids = zero_routed.nonzero(
                as_tuple=False
            ).flatten().tolist()
            logging.warning(
                "[realq_moe.routes] layer=%d teacher source=%s "
                "zero-route experts=%s fallback=%s",
                layer_idx,
                source,
                expert_ids,
                cfg.moe_zero_route_fallback,
            )
        logging.info(
            "[realq_moe.routes] layer=%d teacher source=%s assignments "
            "min=%d max=%d; unique_samples min=%d max=%d",
            layer_idx,
            source,
            int(coverage["assignment_count"].min().item()),
            int(coverage["assignment_count"].max().item()),
            int(coverage["unique_sample_count"].min().item()),
            int(coverage["unique_sample_count"].max().item()),
        )


def _clip_and_validate_expert_saliency(
    cfg: "Config",
    saliency: list[dict[str, torch.Tensor]],
    routes: list[Mapping[int, Mapping[str, torch.Tensor]]],
    *,
    percentile: float | None,
    seq_len: int,
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Packed P99 clipping plus cross-rank cold-expert validation."""

    expert_global_coverage = routed_stats.global_expert_coverage(
        routes, seq_len=seq_len
    )
    _validate_teacher_coverage(
        cfg,
        expert_global_coverage,
        source="fresh Stage-0",
    )
    empty_count_device = next(
        (
            layer_routes.device
            for layer_routes in routes
            if isinstance(layer_routes, routed_stats.PackedLayerRoutes)
        ),
        next(
            (
                tensor.device
                for layer_saliency in saliency
                for tensor in layer_saliency.values()
            ),
            torch.device(
                f"cuda:{torch.cuda.current_device()}"
                if torch.cuda.is_available()
                else "cpu"
            ),
        ),
    )
    global_route_counts = [
        (
            coverage["assignment_count"]
            if coverage
            else torch.empty(
                0,
                dtype=torch.int64,
                device=empty_count_device,
            )
        )
        for coverage in expert_global_coverage
    ]
    for layer_idx, layer_routes in enumerate(routes):
        if not layer_routes:
            continue
        num_experts = len(layer_routes)
        route_counts = global_route_counts[layer_idx]
        for projection in model_adapter.EXPERT_PROJECTION_ORDER:
            tensors: list[torch.Tensor] = []
            local_error: str | None = None
            try:
                tensors = [
                    saliency[layer_idx][
                        model_adapter.expert_projection_path(
                            expert_idx, projection
                        )
                    ]
                    for expert_idx in range(num_experts)
                ]
                for expert_idx, tensor in enumerate(tensors):
                    local_route_count = int(
                        layer_routes[expert_idx][
                            "flat_token_indices"
                        ].numel()
                    )
                    if (
                        tensor.dim() != 3
                        or tensor.shape[1] != 1
                        or int(tensor.shape[0]) != local_route_count
                    ):
                        raise RuntimeError(
                            f"expert {expert_idx} local saliency shape "
                            "does not match its local route: "
                            f"saliency={tuple(tensor.shape)}, "
                            f"route_assignments={local_route_count}."
                        )
            except Exception as exc:
                local_error = str(exc)
            _raise_if_any_rank_failed_validation(
                local_error,
                context=(
                    f"layer {layer_idx} {projection} saliency validation"
                ),
            )
            element_counts = routed_stats.packed_clip_expert_saliency_(
                tensors, percentile
            )
            expected_elements = route_counts * tensors[0].shape[-1]
            if not torch.equal(element_counts, expected_elements):
                raise RuntimeError(
                    f"layer {layer_idx} {projection} saliency counts do not "
                    "match captured teacher-route assignments."
                )
    return global_route_counts, expert_global_coverage


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
    dev = torch.device(f"cuda:{torch.cuda.current_device()}")

    # 1. Cache lookup.
    if cfg.static_cache_path:
        with nvtx.nvtx_range("precompute.cache_lookup"):
            key = cache_mod.build_cache_key(cfg, world)
            cached = cache_mod.try_load(
                cfg.static_cache_path,
                key,
                world,
                rank,
                map_location=dev,
            )
            local_hit = cached is not None
            all_hit = _all_ranks_have_cache(local_hit, world)
            if all_hit:
                logging.info(
                    "[realq.precompute] cache hit (rank %d): %s",
                    rank, cache_mod.cache_path(cfg.static_cache_path, key, world, rank),
                )
                # The speed-first MoE runtime is GPU-resident across the
                # Stage-0/Stage-1 boundary.  A cache hit must not evict the
                # model or statistics to CPU.
                static = StaticStats(
                    saliency=cached["saliency"],
                    fisher=cached["fisher"],
                    routes=cached["routes"],
                    expert_global_assignment_counts=(
                        cached["expert_global_assignment_counts"]
                    ),
                    expert_global_coverage=(
                        cached["expert_global_coverage"]
                    ),
                )
                _assert_static_stats_resident(
                    static,
                    dev,
                    source="cached",
                )
                _validate_teacher_coverage(
                    cfg,
                    static.expert_global_coverage,
                    source="Stage-0 cache",
                )
                return static
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
        rank_sample_list = _shard_local_calibration(
            trainloader, rank, world
        )
        n_local = len(rank_sample_list)
        if n_local == 0:
            raise RuntimeError(
                f"[realq.precompute] rank {rank} got 0 calibration samples — bump nsamples."
            )
        # One initial input upload is unavoidable; repeated per-batch H2D is
        # not.  Retain the complete rank-local token shard on GPU for Stage-0.
        rank_samples = torch.stack(
            [sample.view(-1) for sample in rank_sample_list], dim=0
        ).to(dev)
        rank_sample_list = None

        if cfg.global_loss_bsz % world != 0:
            raise ValueError(
                f"global_loss_bsz ({cfg.global_loss_bsz}) must be divisible by "
                f"world_size ({world})."
            )
        local_bsz = max(1, cfg.global_loss_bsz // world)

    # 3. Move model to GPU and switch to grad-enabled mode.
    with nvtx.nvtx_range("precompute.model_setup"):
        model = analyzer.model
        layers = analyzer.get_layers()
        if (
            any(model_adapter.is_sparse_moe_layer(layer) for layer in layers)
            and bool(getattr(model, "is_gradient_checkpointing", False))
        ):
            raise RuntimeError(
                "REAL-Q MoE Stage-0 route/saliency capture does not support "
                "gradient checkpointing; disable it before precompute."
            )
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
        layer_modules = [_find_layer_modules(layer) for layer in layers]
        expert_module_indices = [
            _expert_module_indices(layer) for layer in layers
        ]
        route_mgr = routed_stats.RouteCaptureManager(layers)
        sal_mgr = hooks_mod.SaliencyHookManager(
            num_groups=cfg.num_groups,
            clip_percentile=cfg.saliency_clip_percentile,
        )
        fisher_mgr = hooks_mod.FisherHookManager()
        sal_mgr.attach(
            layer_modules,
            expert_module_indices=expert_module_indices,
            route_manager=route_mgr,
        )
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
                    input_ids = rank_samples[local_start:local_end]
                    B, T = input_ids.shape
                    route_mgr.begin_batch(
                        local_start=local_start,
                        batch_size=B,
                        seq_len=T,
                    )

                    # Per-sample global ids. Sharding is contiguous, so global id is
                    # just the local id offset by the rank's shard start.
                    shard = dist_utils.shard_slice(cfg.nsamples, rank=rank, world=world)
                    global_indices = [shard.start + local_start + i for i in range(B)]

                    model.zero_grad(set_to_none=True)
                    with nvtx.nvtx_range("precompute.forward"):
                        outputs = model(input_ids=input_ids)
                        route_mgr.release_current()
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
        route_mgr.remove()
        kick_off_handle.remove()
        _restore_requires_grad(prev_requires_grad)

    # 6. Finalize.
    with nvtx.nvtx_range("precompute.finalize_saliency"):
        saliency = sal_mgr.finalize()
        routes = route_mgr.finalize()
        (
            expert_global_assignment_counts,
            expert_global_coverage,
        ) = _clip_and_validate_expert_saliency(
            cfg,
            saliency,
            routes,
            percentile=cfg.saliency_clip_percentile,
            seq_len=cfg.seq_len,
        )
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
                fisher_out.append(
                    (fisher_sum / total_tokens).to(torch.bfloat16)
                )
        del fisher_sums

    # 7. Restore logical model state.  Keep every weight/statistic on GPU:
    # CPU offload is forbidden in the speed-first MoE execution path.
    with nvtx.nvtx_range("precompute.teardown"):
        model.config.use_cache = use_cache
        model.eval()

    static = StaticStats(
        saliency=saliency,
        fisher=fisher_out,
        routes=routes,
        expert_global_assignment_counts=expert_global_assignment_counts,
        expert_global_coverage=expert_global_coverage,
    )
    _assert_static_stats_resident(
        static,
        dev,
        source="fresh",
    )

    # 8. Cache write.
    if cfg.static_cache_path:
        with nvtx.nvtx_range("precompute.cache_write"):
            path = cache_mod.save(
                cfg.static_cache_path, key, world, rank,
                {
                    "saliency": static.saliency,
                    "fisher": static.fisher,
                    "routes": static.routes,
                    "expert_global_assignment_counts": (
                        static.expert_global_assignment_counts
                    ),
                    "expert_global_coverage": (
                        static.expert_global_coverage
                    ),
                },
            )
            logging.info("[realq.precompute] wrote rank-%d cache → %s", rank, path)

    return static
