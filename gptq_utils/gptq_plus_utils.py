import copy
import logging
import os
import sys
import math
import pprint
import functools
import random
from contextlib import contextmanager, nullcontext
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call

from utils import quant_utils, memory_utils, model_utils, dist_utils, rotation_utils
from utils.saliency_utils import (
    clip_global_percentile_,
    global_percentile,
    grouped_channel_gram,
    grouped_gradient_norm_squared,
)
from utils.loss_utils import tokenwise_kl_from_logits
from realq.alignment import (
    RefreshTraceWriter,
    default_refresh_trace_config,
    refresh_step_from_metrics,
)
from gptq_utils.diagnostics import DiagnosticRegistry, parse_diagnose_targets
from gptq_utils.quant_aware_utils import (
    configure_activation_quantizers_for_gptq,
    configure_k_cache_quantizers_for_gptq,
    disable_fp_path_quant,
)


_STATIC_SALIENCY_SCHEMA_TAG = "salsumv2"

# Single reusable no-op context. `nullcontext()` instances are stateless, so we
# avoid allocating a new one at every `with ... if profile_recorder else
# nullcontext():` site inside the fasterquant/add_batch hot loops.
_NULL_CONTEXT = nullcontext()

# Static end-to-end saliency/Fisher precompute can produce very small
# activation gradients in bf16-heavy graphs. We scale the scalar loss up during
# backward, keep the saliency cache in the resulting grad^2 scale, and undo that
# scale exactly at the consumers that need the original numerical convention.
_E2E_PRECOMPUTE_LOSS_GRAD_SCALE = 1000.0
_E2E_PRECOMPUTE_QUADRATIC_SCALE = (
    _E2E_PRECOMPUTE_LOSS_GRAD_SCALE * _E2E_PRECOMPUTE_LOSS_GRAD_SCALE
)
_TORCH_QUANTILE_SAFE_NUMEL = 16 * 1024 * 1024


def format_log_value(value, digits=6):
    if value is None:
        return "None"
    return f"{float(value):.{digits}g}"


def _quantile_large(tensor, q):
    """torch.quantile chokes on very large inputs on some torch versions
    (pre-2.0 had a 16M element cap). For block_gd diagnostics where the
    flattened update can hit 100M+ entries, fall back to `kthvalue` on sort,
    which is O(n log n) but handles any size. Assumes `tensor` is non-empty
    float (callers pass `.abs()` of a float tensor)."""
    flat = tensor.reshape(-1)
    n = flat.numel()
    if n == 0:
        return None
    k = max(1, min(n, int(n * q + 0.5)))
    try:
        return torch.quantile(flat, q).item()
    except RuntimeError:
        return flat.kthvalue(k).values.item()


def _activation_clip_threshold(tensor, q):
    """Return the old activation-loss clip threshold without host sync.

    Small tensors keep torch.quantile's interpolation semantics. Large tensors
    use the old fallback's order statistic, but select only the smaller tail;
    q is usually 0.95/0.99 so this avoids kthvalue over the full activation.
    """
    flat = tensor.reshape(-1)
    n = flat.numel()
    if n == 0:
        return None
    if n <= _TORCH_QUANTILE_SAFE_NUMEL:
        try:
            return torch.quantile(flat, q)
        except RuntimeError:
            pass
    k = max(1, min(n, int(n * q + 0.5)))
    upper_count = n - k + 1
    if k <= upper_count:
        return flat.topk(k, largest=False, sorted=False).values.max()
    return flat.topk(upper_count, largest=True, sorted=False).values.min()


def _scale_delta_by_abs_quantile(
    delta,
    ratio,
    profile_recorder=None,
    threshold=None,
):
    if ratio >= 1.0:
        return delta
    with profile_recorder.section("compute_refresh_loss.a_loss_delta_scale") if profile_recorder else _NULL_CONTEXT:
        abs_delta = delta.detach().float().abs()
        if threshold is None:
            threshold = _activation_clip_threshold(
                abs_delta, float(ratio)
            )
        else:
            threshold = threshold.to(
                device=abs_delta.device, dtype=abs_delta.dtype
            )
        if threshold is None:
            return delta
        scale = abs_delta.clamp_min_(torch.finfo(abs_delta.dtype).tiny)
        scale.reciprocal_().mul_(threshold.detach()).clamp_(max=1.0)
        scale.nan_to_num_(nan=1.0, posinf=1.0, neginf=1.0)
        return delta * scale.to(delta.dtype)


def _reduce_refresh_loss_for_aggregation(refresh_loss_type, loss_tensor, batch_size):
    """Convert a local refresh loss scalar to the sum convention used by callers."""
    return float(loss_tensor.item()) * float(batch_size)


def _refresh_act_quant_wrapper_aliases(module):
    for child in module.modules():
        if isinstance(child, quant_utils.ActQuantWrapper):
            child.weight = child.module.weight
            child.bias = child.module.bias


def _alloc_module_empty_on_device(module, device):
    module.to_empty(device=device)
    _refresh_act_quant_wrapper_aliases(module)
    return module


def _iter_module_tensors(module):
    seen = set()
    for _, param in module.named_parameters(recurse=True, remove_duplicate=True):
        if id(param) in seen:
            continue
        seen.add(id(param))
        yield param
    for _, buf in module.named_buffers(recurse=True, remove_duplicate=True):
        if buf is None or id(buf) in seen:
            continue
        seen.add(id(buf))
        yield buf


@torch.no_grad()
def _broadcast_module_from_rank0(module, dev, src_module=None):
    if dist_utils.get_world_size() <= 1:
        return module.to(dev)

    rank = dist_utils.get_rank()
    if rank == 0:
        source = src_module if src_module is not None else module
        target = source.to(dev)
    else:
        target = _alloc_module_empty_on_device(module, dev)

    for tensor in _iter_module_tensors(target):
        if tensor.device.type != "cuda":
            raise RuntimeError(
                f"stage2_cpu_master broadcast expected CUDA tensors, got {tensor.device}."
            )
        if not tensor.is_contiguous():
            tensor.data = tensor.data.contiguous()
        dist.broadcast(tensor.data, src=0)
    _refresh_act_quant_wrapper_aliases(target)
    return target


@torch.no_grad()
def _free_module_to_meta(module):
    module.to_empty(device="meta")
    _refresh_act_quant_wrapper_aliases(module)
    return module


class Stage2CpuMasterLayerManager:
    def __init__(self, analyzer, dev, layers):
        self.analyzer = analyzer
        self.model = analyzer.model
        self.dev = dev
        self.layers = layers
        self.rank = dist_utils.get_rank()
        self.enabled = bool(getattr(self.model, "_gptqplus_stage2_cpu_master", False))
        self.master_layers = layers if self.enabled and self.rank == 0 else None
        self.materialized = set()

    def _master_layer(self, idx):
        if self.master_layers is None:
            return None
        return self.master_layers[idx]

    def materialize_runtime_modules(self, modules):
        if not self.enabled:
            for module in modules:
                module.to(self.dev)
            return
        for module in modules:
            _broadcast_module_from_rank0(
                module,
                self.dev,
                src_module=module if self.rank == 0 else None,
            )

    def release_runtime_modules(self, modules, orig_device):
        if not self.enabled:
            for module in modules:
                module.to(orig_device)
            return
        for module in modules:
            if self.rank == 0:
                module.to(orig_device)
            else:
                _free_module_to_meta(module)

    def materialize_layer(self, idx):
        if not self.enabled:
            layer = self.layers[idx].to(self.dev)
            self.layers[idx] = layer
            return layer
        if idx in self.materialized:
            return self.layers[idx]
        layer = _broadcast_module_from_rank0(
            self.layers[idx],
            self.dev,
            src_module=self._master_layer(idx),
        )
        self.layers[idx] = layer
        self.materialized.add(idx)
        return layer

    def release_layer(self, idx, layer=None, *, update_master=False, orig_device=None):
        if not self.enabled:
            if layer is None:
                layer = self.layers[idx]
            self.layers[idx] = layer.to(orig_device)
            return self.layers[idx]
        if layer is None:
            layer = self.layers[idx]
        if self.rank == 0:
            self.layers[idx] = layer.to(orig_device)
            self.master_layers[idx] = self.layers[idx]
        else:
            self.layers[idx] = _free_module_to_meta(layer)
        self.materialized.discard(idx)
        return self.layers[idx]


def normalize_quant_module_name(name: str) -> str:
    return name[:-7] if name.endswith(".module") else name


def resolve_quant_module(full, module_name: str):
    """Return the actual quantized Linear module and its path in `full`.

    After activation wrappers are installed, analyzer.get_quantizable_modules()
    sees the wrapped Linear as e.g. `mlp.down_proj.module`, while the quantization
    schedule and reporting use the canonical name `mlp.down_proj`. Keeping the
    resolution in one place avoids accidentally mixing the wrapper with the
    Linear that GPTQ+ updates.
    """
    candidates = [module_name]
    if module_name.endswith(".module"):
        candidates.append(normalize_quant_module_name(module_name))
    else:
        candidates.append(module_name + ".module")

    canonical_name = normalize_quant_module_name(module_name)
    for name in full.keys():
        if normalize_quant_module_name(name) == canonical_name:
            candidates.append(name)

    seen = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        module = full.get(name)
        if module is not None:
            if isinstance(module, quant_utils.ActQuantWrapper):
                return f"{name}.module", module.module
            return name, module

    raise ValueError(
        f"Unable to find module `{module_name}` in the provided layer. "
        f"Available modules: {sorted(full.keys())}"
    )


def _get_submodule_or_none(root: nn.Module, path: str):
    if not path:
        return root
    try:
        return root.get_submodule(path)
    except AttributeError:
        cur = root
        for part in path.split("."):
            if not hasattr(cur, part):
                return None
            cur = getattr(cur, part)
        return cur
    except Exception:
        return None


def functional_weight_name_for_quant_module(
    layer: nn.Module,
    module_name: str,
    resolved_module_name: str,
    module: nn.Module,
) -> str:
    """Parameter key for functional_call that targets the updated Linear.

    With rotation enabled, FFN down_proj is an ActQuantWrapper:

        wrapper.forward: online Hadamard -> wrapper.module(...)

    The online Hadamard is part of the input transform, but the weight being
    optimized/quantized is the inner Linear's weight. Use the inner parameter
    path explicitly (`*.module.weight`) instead of relying on the wrapper-level
    `*.weight` alias.
    """
    canonical_name = normalize_quant_module_name(module_name)
    wrapper = _get_submodule_or_none(layer, canonical_name)
    if (
        isinstance(wrapper, quant_utils.ActQuantWrapper)
        and getattr(wrapper, "module", None) is module
    ):
        return f"{canonical_name}.module.weight"
    return f"{resolved_module_name}.weight"


def build_quant_subset(full, names):
    subset = {}
    for name in names:
        try:
            _, module = resolve_quant_module(full, name)
        except ValueError:
            continue
        subset[name] = module
    return subset


def canonical_refresh_loss_type(loss_type: str) -> str:
    return "hidden_mse" if loss_type == "layer_mse" else loss_type


def is_hidden_mse_loss(loss_type: str) -> bool:
    return loss_type in ("hidden_mse", "layer_mse")


def is_fisher_mse_loss(loss_type: str) -> bool:
    return loss_type in ("fisher_diag_mse", "legacy_fisher_diag_mse")


def is_fisher_backed_loss(loss_type: str) -> bool:
    return is_fisher_mse_loss(loss_type) or loss_type == "refined_mse"


def is_legacy_fisher_diag_loss(loss_type: str) -> bool:
    return canonical_refresh_loss_type(loss_type) == "legacy_fisher_diag_mse"


def is_full_fisher_backed_loss(loss_type: str) -> bool:
    loss_type = canonical_refresh_loss_type(loss_type)
    return loss_type in ("fisher_diag_mse", "refined_mse")


def select_layer_output_fisher_for_loss(
    loss_type: str,
    static_fisher_by_layer,
    static_legacy_fisher_diag_by_layer,
    layer_idx: int,
):
    loss_type = canonical_refresh_loss_type(loss_type)
    if loss_type == "legacy_fisher_diag_mse":
        if static_legacy_fisher_diag_by_layer is None:
            return None
        return static_legacy_fisher_diag_by_layer[layer_idx]
    if is_full_fisher_backed_loss(loss_type):
        if static_fisher_by_layer is None:
            return None
        return static_fisher_by_layer[layer_idx]
    return None


def slice_layer_output_fisher_for_batch(
    layer_output_fisher,
    batch_indices,
    dev,
    refresh_loss_type: str,
):
    if layer_output_fisher is None:
        return None
    refresh_loss_type = canonical_refresh_loss_type(refresh_loss_type)
    if refresh_loss_type == "legacy_fisher_diag_mse":
        if layer_output_fisher.dim() != 3:
            raise ValueError(
                "legacy_fisher_diag_mse expects per-token Fisher diagonal with "
                f"shape (N_local, T, H), got {tuple(layer_output_fisher.shape)}."
            )
        return layer_output_fisher[batch_indices].to(dev).float()
    return layer_output_fisher.to(dev).float()


def get_effective_refresh_loss_type(
    layer_idx: int,
    final_layer_idx: int,
    default_refresh_loss_type: str,
    refined_mix_split_layer=None,
) -> str:
    default_refresh_loss_type = canonical_refresh_loss_type(default_refresh_loss_type)
    if layer_idx == final_layer_idx:
        return "kl"
    if default_refresh_loss_type == "refined_mix":
        if refined_mix_split_layer is None:
            raise ValueError(
                "refined_mix requires refined_mix_split_layer to be provided."
            )
        return (
            "fisher_diag_mse" if layer_idx < refined_mix_split_layer
            else "refined_residual_kl"
        )
    return default_refresh_loss_type


def get_effective_gptq_reference_loss_type(global_loss_enabled: bool, layer_refresh_loss_type: str) -> str:
    return canonical_refresh_loss_type(layer_refresh_loss_type) if global_loss_enabled else "kl"


def compute_safe_beta_from_reference_loss(
    gradients_sub: torch.Tensor,
    hinv_init: torch.Tensor,
    reference_loss: float,
    alpha: float,
):
    ghinv_init = gradients_sub.matmul(hinv_init)
    if reference_loss <= 0:
        beta = torch.zeros(gradients_sub.shape[0], device=gradients_sub.device, dtype=gradients_sub.dtype)
        return beta, ghinv_init

    c = (gradients_sub * ghinv_init).sum(dim=1) - (
        (ghinv_init ** 2) / torch.diagonal(hinv_init).unsqueeze(0)
    ).mean(1)
    target = 2 * alpha * reference_loss
    c = torch.clamp(c, min=target)
    ratio = torch.where(c > 0, target / c, torch.zeros_like(c))
    ratio = torch.clamp(ratio, min=0.0, max=1.0)
    beta = 1 - torch.sqrt(torch.clamp(1 - ratio, min=0.0))
    beta = torch.nan_to_num(beta, nan=0.0, posinf=1.0, neginf=0.0)
    return beta, ghinv_init


def compute_quant_clip_bounds(scale, zero, maxq, sym):
    scale = scale.clamp(min=1e-8)
    maxq_value = int(maxq.item()) if isinstance(maxq, torch.Tensor) else int(maxq)
    if sym:
        minq_value = -(maxq_value + 1)
        return scale * minq_value, scale * maxq_value
    return scale * (-zero), scale * (maxq_value - zero)


def clip_tensor_to_quant_bounds(weight, clip_min, clip_max):
    clip_min = clip_min.to(weight.device, dtype=weight.dtype)
    clip_max = clip_max.to(weight.device, dtype=weight.dtype)
    return torch.minimum(torch.maximum(weight, clip_min), clip_max)


def clip_module_weight_to_quant_bounds_(module, bits, sym, mse):
    if bits >= 16 or not mse:
        return
    quantizer = quant_utils.WeightQuantizer()
    quantizer.configure(bits, perchannel=True, sym=sym, mse=mse)
    weight = module.weight.data.float()
    quantizer.find_params(weight)
    clip_min, clip_max = compute_quant_clip_bounds(
        quantizer.scale.to(weight.device),
        quantizer.zero.to(weight.device),
        quantizer.maxq,
        quantizer.sym,
    )
    module.weight.data.copy_(clip_tensor_to_quant_bounds(weight, clip_min, clip_max).to(module.weight.data.dtype))


class QuantProfileRecorder:
    _WALL_ENABLED = os.environ.get("GPTQ_PLUS_WALL_PROFILE", "0") == "1"
    _WALL_STATS: "dict[str, list[tuple]]" = {}

    def __init__(self, device, prefix=None):
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.enabled = self.device.type == "cuda" and torch.cuda.is_available()
        self.prefix = prefix

    @contextmanager
    def section(self, name):
        if not self.enabled:
            yield
            return

        range_name = f"{self.prefix}.{name}" if self.prefix else name
        torch.cuda.nvtx.range_push(range_name)
        start_evt = None
        end_evt = None
        if QuantProfileRecorder._WALL_ENABLED:
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
        try:
            yield
        finally:
            if start_evt is not None:
                end_evt.record()
                QuantProfileRecorder._WALL_STATS.setdefault(range_name, []).append(
                    (start_evt, end_evt)
                )
            torch.cuda.nvtx.range_pop()

    def summary(self):
        return {}

    @classmethod
    def dump_wall_summary(cls, top_k: int = 60):
        if not cls._WALL_ENABLED:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        rows = []
        for name, events in cls._WALL_STATS.items():
            elapsed_ms = 0.0
            for s, e in events:
                elapsed_ms += s.elapsed_time(e)
            rows.append((name, elapsed_ms, len(events)))
        rows.sort(key=lambda r: -r[1])
        total = sum(r[1] for r in rows) or 1.0
        lines = ["", "=" * 80, f"Wall-clock section summary (top {top_k} by total ms)", "=" * 80]
        header = f"{'section':<70} {'calls':>6} {'total_ms':>12} {'pct':>6} {'mean_ms':>10}"
        lines.append(header)
        lines.append("-" * len(header))
        for name, ms, n in rows[:top_k]:
            pct = 100.0 * ms / total
            mean = ms / max(n, 1)
            lines.append(f"{name[:70]:<70} {n:>6d} {ms:>12.2f} {pct:>5.1f}% {mean:>10.3f}")
        lines.append("=" * 80)
        out = "\n".join(lines)
        logging.info(out)
        print(out)


def parse_quant_profile_target_layers(spec: str, num_layers: int):
    if spec.strip().lower() == "all":
        return list(range(num_layers))

    result = []
    for item in spec.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token == "mid":
            idx = num_layers // 2
        elif token == "last":
            idx = num_layers - 1
        else:
            idx = int(token)
            if idx < 0:
                idx = num_layers + idx
        if not (0 <= idx < num_layers):
            raise ValueError(f"Quant profile target layer index out of range: {item}")
        result.append(idx)
    return sorted(set(result))


def parse_quant_profile_target_modules(spec: str):
    if spec.strip().lower() == "all":
        return []
    return [item.strip() for item in spec.split(",") if item.strip()]


def parse_quant_stop_layer(spec, num_layers: int):
    if spec is None:
        return None
    if isinstance(spec, int):
        idx = spec
    else:
        token = str(spec).strip().lower()
        if token in {"", "none", "all", "-"}:
            return None
        if token == "last":
            idx = num_layers - 1
        else:
            idx = int(token)
    if idx < 0:
        idx = num_layers + idx
    if not (0 <= idx < num_layers):
        raise ValueError(f"Quant stop layer index out of range: {spec}")
    return idx


def _dbg_warm_prior_vs_gradient(prior_slice, grad_slice, t0, beta2, path):
    """WARM_ADAM_DEBUG=1 only. Scale and shape of the prior vs the real g^2."""
    denom = 1.0 - beta2 ** max(int(t0), 0)
    if denom <= 0:
        logging.info("warm-prior-dbg[%s]: t0=0, prior is zeroed; nothing to check", path)
        return

    def _corr(a, b):
        a = a - a.mean()
        b = b - b.mean()
        return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))

    with torch.no_grad():
        p = (prior_slice.detach().float() / denom).reshape(-1)
        g2 = grad_slice.detach().float().pow(2).reshape(-1)
        mask = (p > 0) & (g2 > 0)
        n = int(mask.sum().item())
        if n < 1000:
            logging.info("warm-prior-dbg[%s]: only %d usable entries, skipping", path, n)
            return
        pm, gm = p[mask], g2[mask]
        lp, lg = torch.log(pm), torch.log(gm)
        corr = _corr(lp, lg)
        # Same values, scrambled: what a misaligned prior would score.
        ctrl = _corr(lp[torch.randperm(lp.numel(), device=lp.device)], lg)
        p_med = float(pm.median().item())
        g_med = float(gm.median().item())
        logging.info(
            "warm-prior-dbg[%s]: n=%d log-corr=%.4f shuffled=%.4f "
            "prior_med=%s g2_med=%s ratio=%.3g (1.0 = matched scale)",
            path, n, corr, ctrl,
            format_log_value(p_med), format_log_value(g_med),
            p_med / g_med if g_med > 0 else float("inf"),
        )


def _reach_probe_block_row(blk, w0, q, disp, path, scale):
    """One row of the reach table, for the columns that just froze.

    Medians rather than means throughout: both |W_fp - Q| and the optimizer
    displacement are heavy-tailed, and a handful of outlier channels would
    otherwise set the headline number on their own.
    """
    dev = (w0 - q).abs().flatten().float()
    d = disp.abs().flatten().float()
    p = path.flatten().float()
    sc = scale.abs().flatten().float()
    eps = torch.finfo(torch.float32).tiny
    med_dev = dev.median().item()
    med_d = d.median().item()
    return {
        "blk": blk,
        "med_dev": med_dev,
        "med_disp": med_d,
        "med_path": p.median().item(),
        "med_grid": sc.median().item(),
        # Ratio of medians for the headline; median of per-weight ratios
        # alongside it, because the two disagree when dev is near zero for
        # the weights that happen to land on a grid point.
        "R_ratio_of_med": med_d / (med_dev + eps),
        "R_med_of_ratio": (d / (dev + eps)).median().item(),
        "disp_over_grid": (d / (sc + eps)).median().item(),
    }


def _reach_probe_emit(layer_idx, name, lr, rows, n_groups=None):
    if not rows:
        return

    def fmt(key):
        return " ".join(
            "b%d:%s" % (r["blk"], format_log_value(r[key])) for r in rows
        )

    logging.info(
        "reach_probe layer=%s module=%s lr=%s nblk=%d ngrp=%s\n"
        "  |W_fp-Q|   %s\n"
        "  |disp|     %s\n"
        "  path       %s\n"
        "  grid       %s\n"
        "  R=disp/dev %s\n"
        "  disp/grid  %s",
        layer_idx, name, format_log_value(lr), len(rows), n_groups,
        fmt("med_dev"), fmt("med_disp"), fmt("med_path"),
        fmt("med_grid"), fmt("R_ratio_of_med"), fmt("disp_over_grid"),
    )



_HORIZON_LOGGED = []


def _horizon_step_weights(n_tail, blocksize, tail_first_block, p, device, dtype):
    """Per-column step multiplier from the column's remaining update count.

    See the module note on --horizon_p. Returns None for p=0 so the production
    path stays byte-identical rather than multiplying by a tensor of ones.

    The refresh only fires while i2 < C, so i2 is always a whole number of
    blocks and `tail_first_block` = i2 // blocksize is the block index of the
    first tail column -- which is also how many updates that column will have
    received in total, since a column in block k is updated once after each of
    blocks 0 .. k-1.
    """
    if not p:
        return None
    h = torch.arange(n_tail, device=device, dtype=torch.float32)
    h.div_(blocksize).floor_()
    k = h + float(tail_first_block)
    kmax = int(k.max().item())
    # sum_{i=1..k} i^-p for every k that occurs; kmax is the block count, so a
    # table is exact and costs nothing.
    tbl = torch.cumsum(
        torch.arange(1, kmax + 1, device=device, dtype=torch.float32) ** (-p),
        dim=0,
    )
    w = k * ((h + 1.0) ** (-p)) / tbl[k.long() - 1]
    if not _HORIZON_LOGGED:
        # Once per process, print the weights that were actually built. The
        # banner only shows shell variables; WARM_PRIOR_BATCHES=8 was printed
        # as "K=8" while the flag never reached ptq.py and the arm ran at K=1.
        _HORIZON_LOGGED.append(1)
        _buckets = [float(w[i * blocksize]) for i in range(int(h.max().item()) + 1)]
        logging.info(
            "horizon_p=%g active: tail=%d blocksize=%d first_tail_block=%d, "
            "step multiplier per remaining-update bucket (h=0 is the column's "
            "last update before it freezes): %s",
            p, n_tail, blocksize, tail_first_block,
            " ".join("h%d:%.3f" % (i, v) for i, v in enumerate(_buckets)),
        )
    return w.to(dtype)


def _align_warm_prior(grad_sq_full, row_start, row_end, perm, device):
    """Put the [out, in] prior into one subgroup's row-sliced, actorder frame.

    Mirrors exactly what fasterquant does to `W_sub`, and nothing else. In
    particular the dead-column fix (`W_sub[:, dead] = 0`) is deliberately NOT
    mirrored: that zeroing exists because a dead column carries no Hessian mass,
    whereas the prior is a variance and zeroing it would hand those columns a
    denominator of `eps`. Their measured value is the right one to keep.
    """
    if grad_sq_full is None:
        return None
    prior = grad_sq_full[row_start:row_end, :]
    if perm is not None:
        prior = prior[:, perm]
    return prior.to(device=device, dtype=torch.float32).contiguous()


def _align_warm_prior_batched(grad_sq_full, G, R, C, perm, hessian_group_ids, device):
    """Same, for the group-parallel path's (G, R, C) block layout.

    The transform order has to match `W_sub` step for step -- reshape, then the
    actorder permutation on the column axis, then the hessian-group shard -- or
    the preconditioner is silently applied to the wrong weights.
    """
    if grad_sq_full is None:
        return None
    prior = grad_sq_full.reshape(G, R, C)
    if perm is not None:
        prior = prior[:, :, perm]
    if hessian_group_ids is not None:
        prior = prior.index_select(0, hessian_group_ids)
    return prior.to(device=device, dtype=torch.float32).contiguous()


class BackwardSampleScheduler:
    def __init__(self, total_samples: int, chunk_size: int, seed: int = 42):
        if chunk_size <= 0:
            raise ValueError(f"`chunk_size` must be positive. Got {chunk_size}.")
        if total_samples % chunk_size != 0:
            raise ValueError(
                f"Total samples ({total_samples}) must be divisible by backward chunk size ({chunk_size})."
            )
        self.total_samples = total_samples
        self.chunk_size = chunk_size
        self.order = list(range(total_samples))
        self.cursor = 0
        self.rng = random.Random(seed)

    def next_indices(self):
        if self.cursor >= self.total_samples:
            self.rng.shuffle(self.order)
            self.cursor = 0
        start = self.cursor
        end = start + self.chunk_size
        indices = self.order[start:end]
        self.cursor = end
        return indices

    @contextmanager
    def frozen(self):
        """Run a block without consuming any of the sample stream.

        `next_indices()` mutates three things: the cursor, the permutation, and
        the RNG that reshuffles it. Anything that calls a refresh for
        *measurement* rather than for a training step -- the warm_adam pre-pass
        -- must not advance that stream, or every subsequent refresh in the run
        sees a different batch composition than it would have otherwise. That is
        invisible in the logs and it changes the result: at
        `backward_samples == nsamples` the wrap reshuffles on every call, so a
        single extra call leaves the sample *set* identical while permuting it,
        which was enough to move final KL by 5% (0.211 -> 0.222) once Adam sits
        in its sign-dominated regime.
        """
        saved_order = list(self.order)
        saved_cursor = self.cursor
        saved_rng = self.rng.getstate()
        try:
            yield self
        finally:
            self.order = saved_order
            self.cursor = saved_cursor
            self.rng.setstate(saved_rng)


@contextmanager
def temporary_requires_grad(modules, keep_params):
    keep_param_ids = {id(param) for param in keep_params if param is not None}
    saved_states = []
    seen_param_ids = set()
    for module in modules:
        for param in module.parameters(recurse=True):
            if id(param) in seen_param_ids:
                continue
            seen_param_ids.add(id(param))
            saved_states.append((param, param.requires_grad))
            param.requires_grad_(id(param) in keep_param_ids)
    try:
        yield
    finally:
        for param, old_state in saved_states:
            param.requires_grad_(old_state)


def get_module_grad_lr(
    module_name: str,
    base_grad_lr: float,
    proj_lr_scale: float = 0.1,
    down_proj_lr_scale: float = 0.1,
) -> float:
    if module_name.endswith("down_proj"):
        return base_grad_lr * down_proj_lr_scale
    if module_name.endswith("self_attn.o_proj") or module_name.endswith("o_proj"):
        return base_grad_lr * proj_lr_scale
    return base_grad_lr


def compute_layer_lr_scale(layer_idx: int, num_layers: int, schedule: str) -> float:
    """Layer-wise LR ramp. Returns a multiplier in [0, 1].

    x = layer_idx / max(num_layers - 1, 1) ∈ [0, 1]. Mapping:
        "none"   -> 1.0  (no ramp)
        "linear" -> x
        "cosine" -> sin(pi * x / 2)            (paper reverse-cosine ramp)
        "sqrt"   -> sqrt(x)                   (rises fast early)

    Degenerate single-layer models (num_layers <= 1) get scale = 1.0.
    """
    if schedule in (None, "", "none"):
        return 1.0
    if num_layers <= 1:
        return 1.0
    x = float(layer_idx) / float(num_layers - 1)
    x = min(max(x, 0.0), 1.0)
    if schedule == "linear":
        return x
    if schedule == "cosine":
        return math.sin(math.pi * x / 2.0)
    if schedule == "sqrt":
        return math.sqrt(x)
    raise ValueError(f"Unknown grad_lr_layer_schedule={schedule!r}")


def compute_scheduled_layer_lr(target_lr: float, layer_scale: float, base_ratio: float) -> float:
    """Interpolate from a base LR to the target LR using the layer schedule."""
    base_lr = float(target_lr) * float(base_ratio)
    return base_lr + (float(target_lr) - base_lr) * float(layer_scale)


class GPTQPlus:
    def __init__(self,
        layer,
        saliency: torch.Tensor, # shape (N_local, seq_len, G) — rank-local shard in DP
        gradient: torch.Tensor, # shape (G, in_features)
        num_groups: int,
        alpha: float,
        reference_loss: float,
        hessian_saliency_scale: float = 1.0,
        hessian_group_shard: bool = False,
    ):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]

        # Instead of passing H in, we allocate a 3D Hessian buffer
        # that will hold sub-channel Hessians along the last dim.
        self.num_groups = num_groups
        assert self.num_groups == saliency.shape[2], "Number of groups for GuidedQuant must match saliency shape!"
        self.alpha = alpha
        self.reference_loss = max(reference_loss, 0)
        self.hessian_saliency_scale = float(hessian_saliency_scale)

        self.saliencies = saliency.float()
        self.gradients = gradient.float()
        # Assert row partition is valid:
        # we do the same partition as before:
        assert self.rows % self.num_groups == 0, (
            f"Number of rows ({self.rows}) must be divisible "
            f"by num_groups ({self.num_groups})"
        )
        self.rows_per_group = self.rows // self.num_groups
        self.hessian_group_sharded = False
        self.hessian_group_ids = torch.arange(self.num_groups, dtype=torch.long)
        self.hessian_group_to_pos = torch.arange(self.num_groups, dtype=torch.long)
        self.hessian_group_owner_mask = None
        self.hessian_group_owner_slots = None
        self.hessian_group_zero = None
        if hessian_group_shard and dist_utils.get_world_size() > 1:
            world = dist_utils.get_world_size()
            rank = dist_utils.get_rank()
            if self.rows % world != 0:
                raise ValueError(
                    "hessian_group_shard requires output rows "
                    f"({self.rows}) to be divisible by world_size ({world})."
                )
            local_row_start = rank * self.rows // world
            local_row_end = (rank + 1) * self.rows // world
            local_rows = torch.arange(local_row_start, local_row_end, dtype=torch.long)
            local_group_ids = torch.unique(
                torch.div(local_rows, self.rows_per_group, rounding_mode="floor")
            )
            if local_group_ids.numel() <= 0:
                raise RuntimeError("hessian_group_shard resolved an empty local group set.")
            if int(local_group_ids.min().item()) < 0 or int(local_group_ids.max().item()) >= self.num_groups:
                raise RuntimeError(
                    "hessian_group_shard produced out-of-range group ids: "
                    f"{local_group_ids.tolist()} for rows [{local_row_start}, {local_row_end})."
                )
            group_to_pos = torch.full((self.num_groups,), -1, dtype=torch.long)
            group_to_pos[local_group_ids] = torch.arange(local_group_ids.numel(), dtype=torch.long)
            group_owner_mask = torch.zeros((self.num_groups, world), dtype=torch.bool)
            for owner_rank in range(world):
                owner_row_start = owner_rank * self.rows // world
                owner_row_end = (owner_rank + 1) * self.rows // world
                owner_rows = torch.arange(owner_row_start, owner_row_end, dtype=torch.long)
                owner_group_ids = torch.unique(
                    torch.div(owner_rows, self.rows_per_group, rounding_mode="floor")
                )
                group_owner_mask[owner_group_ids, owner_rank] = True
            self.hessian_group_sharded = True
            self.hessian_group_ids = local_group_ids
            self.hessian_group_to_pos = group_to_pos
            self.hessian_group_owner_mask = group_owner_mask
            self.hessian_group_owner_slots = [
                group_owner_mask[g].nonzero(as_tuple=False).flatten().tolist()
                for g in range(self.num_groups)
            ]
            logging.info(
                "GPTQPlus Hessian group shard enabled for %s: rank=%d/%d groups=%s "
                "H_shape=(%d,%d,%d) instead of (%d,%d,%d).",
                self.layer.__class__.__name__,
                rank,
                world,
                local_group_ids.tolist(),
                int(local_group_ids.numel()),
                self.columns,
                self.columns,
                self.num_groups,
                self.columns,
                self.columns,
            )
        # Layout: (num_local_hessian_groups, columns, columns). Storing the group
        # axis first makes per-subgroup slices contiguous. In rank-parallel
        # mode this can be a strict subset of output groups, matching the rows
        # this rank owns; non-rank paths keep the legacy full group set.
        self.H = torch.zeros(
            (int(self.hessian_group_ids.numel()), self.columns, self.columns),
            device=self.dev
        )
        if self.hessian_group_sharded:
            self.hessian_group_zero = torch.zeros((), device=self.dev, dtype=self.H.dtype).expand(
                self.columns,
                self.columns,
            )
        self.act_square = torch.zeros(
            (self.columns), device=self.dev
        )
        # Rank-local sample counter; `token_count` mirrors it in units of tokens
        # (= index * seq_len) so we can recover seq_len at finalize time even
        # if the caller never handed it to us directly.
        self.nsamples = saliency.shape[0]
        self.index = 0
        self.token_count = 0
        self._finalized = False
        self.profile_recorder = None

    def _selected_column_count(self, blocksize, max_blocks):
        if max_blocks is None:
            return self.columns
        return min(self.columns, max_blocks * blocksize)

    def _compute_hessian_inverse_with_fallback(
        self,
        H_sub,
        percdamp=0.01,
        damp_auto_increment=0.0015,
        profile_recorder=None,
        profile_section=None,
        log_context="",
    ):
        """Return H^{-1} and its upper Cholesky factor for GPTQ updates.

        If damping has to grow to >= 1, the Hessian estimate is too ill
        conditioned to be useful. Fall back to an identity inverse, which keeps
        per-column quantization running while disabling cross-column second
        order propagation for this subgroup.
        """
        damp_percent = float(percdamp)
        if damp_percent <= 0:
            raise ValueError(
                f"Quantization{log_context}: `damp_percent` must be positive. current is {damp_percent}"
            )

        diag_idx = torch.arange(H_sub.shape[0], device=H_sub.device)
        diag_mean = torch.mean(torch.diag(H_sub))
        last_error = None
        while 1 > damp_percent > 0:
            try:
                section = (
                    profile_recorder.section(profile_section)
                    if profile_recorder is not None and profile_section is not None
                    else _NULL_CONTEXT
                )
                with section:
                    # Every retry must start from the original Hessian.  Reusing
                    # the prior H_work cumulatively added both old and new
                    # damping and even changed the mean used to compute it.
                    H_work = H_sub.clone()
                    damp = damp_percent * diag_mean
                    H_work[diag_idx, diag_idx] += damp
                    chol = torch.linalg.cholesky(H_work)
                    Hinv_init = torch.cholesky_inverse(chol)
                    Hinv = torch.linalg.cholesky(Hinv_init, upper=True)
                if not torch.isfinite(Hinv_init).all() or not torch.isfinite(Hinv).all():
                    raise torch._C._LinAlgError(
                        "Hinv contains non-finite values despite successful Cholesky"
                    )
                return Hinv_init, Hinv, damp_percent, False
            except torch._C._LinAlgError as e:
                last_error = e
                logging.warning(
                    "Quantization%s: Current `damp_percent = %.5f` is too low, "
                    "auto-incrementing by `%.5f`",
                    log_context,
                    damp_percent,
                    damp_auto_increment,
                )
                damp_percent += damp_auto_increment

        if damp_percent >= 1:
            eye = torch.eye(H_sub.shape[0], device=H_sub.device, dtype=H_sub.dtype)
            logging.warning(
                "Quantization%s: `damp_percent` reached %.5f; falling back to "
                "identity Hessian inverse for this subgroup. Last Cholesky error: %s",
                log_context,
                damp_percent,
                last_error,
            )
            return eye, eye, damp_percent, True

        raise ValueError(
            f"Quantization{log_context}: `damp_percent` must between 0 and 1. current is {damp_percent}"
        )

    def _compute_gradient_terms(self, gradients_sub, Hinv_init, Hinv, enable_gradient_update):
        if enable_gradient_update and self.alpha > 0:
            if Hinv_init is None:
                raise ValueError("Hinv_init is required when GPTQ+ first-order updates are enabled.")
            alpha = self.alpha / (self.rows * self.columns)
            beta, GHinv_init = compute_safe_beta_from_reference_loss(
                gradients_sub,
                Hinv_init,
                self.reference_loss,
                alpha,
            )
            beta_view = beta.unsqueeze(1)
            Z = gradients_sub.matmul(Hinv.T) * beta_view
            GHinv = Z.matmul(Hinv)
        else:
            beta = torch.zeros([1]).to(gradients_sub)
            beta_view = beta.unsqueeze(1)
            Z = torch.zeros_like(gradients_sub)
            GHinv = torch.zeros_like(gradients_sub)
        return beta, beta_view, Z, GHinv

    def _compute_hessian_inverse_batched_with_fallback(
        self,
        H_subs,
        percdamp=0.01,
        damp_auto_increment=0.0015,
        profile_recorder=None,
        profile_section=None,
        log_context="",
        need_hinv_init=True,
    ):
        """Batched version of `_compute_hessian_inverse_with_fallback`.

        The normal path runs one Cholesky batch over all output-channel groups.
        If only some groups fail, only those groups retry or fall back to an
        identity inverse.
        """
        damp_percent_value = float(percdamp)
        if damp_percent_value <= 0:
            raise ValueError(
                f"Quantization{log_context}: `damp_percent` must be positive. current is {damp_percent_value}"
            )

        num_groups, columns, _ = H_subs.shape
        device = H_subs.device
        dtype = H_subs.dtype
        diag_idx = torch.arange(columns, device=device)
        eye = torch.eye(columns, device=device, dtype=dtype)
        Hinv_init = torch.empty_like(H_subs) if need_hinv_init else None
        Hinv = torch.empty_like(H_subs)
        damp_percent = torch.full((num_groups,), damp_percent_value, device=device, dtype=torch.float32)
        damp_used = damp_percent.clone()
        fallback = torch.zeros(num_groups, device=device, dtype=torch.bool)
        pending = torch.ones(num_groups, device=device, dtype=torch.bool)
        last_info = torch.zeros(num_groups, device=device, dtype=torch.int32)

        while bool(pending.any().item()):
            active_idx = pending.nonzero(as_tuple=False).flatten()
            # Retry from the pristine subgroup Hessian, not the already damped
            # previous attempt.
            H_work = H_subs.index_select(0, active_idx).clone()
            active_damp = damp_percent.index_select(0, active_idx).to(dtype)
            active_diag_mean = torch.diagonal(H_work, dim1=-2, dim2=-1).mean(dim=1)
            H_work[:, diag_idx, diag_idx] += (active_damp * active_diag_mean).unsqueeze(1)

            section = (
                profile_recorder.section(profile_section)
                if profile_recorder is not None and profile_section is not None
                else _NULL_CONTEXT
            )
            with section:
                chol, info = torch.linalg.cholesky_ex(H_work)

            ok = info == 0
            failed = info != 0
            if bool(ok.any().item()):
                ok_local_idx = ok.nonzero(as_tuple=False).flatten()
                ok_global_idx = active_idx.index_select(0, ok_local_idx)
                hinit = torch.cholesky_inverse(chol.index_select(0, ok_local_idx))
                hchol, hinfo = torch.linalg.cholesky_ex(hinit, upper=True)
                finite = torch.isfinite(hinit).flatten(1).all(dim=1) & torch.isfinite(hchol).flatten(1).all(dim=1)
                ok2 = (hinfo == 0) & finite
                if bool(ok2.any().item()):
                    ok2_local_idx = ok2.nonzero(as_tuple=False).flatten()
                    ok2_global_idx = ok_global_idx.index_select(0, ok2_local_idx)
                    if need_hinv_init:
                        Hinv_init.index_copy_(0, ok2_global_idx, hinit.index_select(0, ok2_local_idx))
                    Hinv.index_copy_(0, ok2_global_idx, hchol.index_select(0, ok2_local_idx))
                    damp_used.index_copy_(0, ok2_global_idx, damp_percent.index_select(0, ok2_global_idx))
                    pending.index_fill_(0, ok2_global_idx, False)
                failed_ok = ~ok2
                if bool(failed_ok.any().item()):
                    failed_ok_local_idx = ok_local_idx.index_select(0, failed_ok.nonzero(as_tuple=False).flatten())
                    failed[failed_ok_local_idx] = True
                    failed_global_idx = ok_global_idx.index_select(0, failed_ok.nonzero(as_tuple=False).flatten())
                    last_info.index_copy_(0, failed_global_idx, hinfo.index_select(0, failed_ok.nonzero(as_tuple=False).flatten()).to(torch.int32))
            failed_global_idx = active_idx.index_select(0, failed.nonzero(as_tuple=False).flatten())
            if failed_global_idx.numel() > 0:
                last_info.index_copy_(0, failed_global_idx, info.index_select(0, failed.nonzero(as_tuple=False).flatten()).to(torch.int32))
                damp_percent.index_add_(
                    0,
                    failed_global_idx,
                    torch.full((failed_global_idx.numel(),), damp_auto_increment, device=device, dtype=torch.float32),
                )
                still_retry = damp_percent.index_select(0, failed_global_idx) < 1
                retry_idx = failed_global_idx.index_select(0, still_retry.nonzero(as_tuple=False).flatten())
                giveup_idx = failed_global_idx.index_select(0, (~still_retry).nonzero(as_tuple=False).flatten())
                if retry_idx.numel() > 0:
                    for idx in retry_idx.detach().cpu().tolist():
                        logging.warning(
                            "Quantization%s subgroup=%d: Current `damp_percent = %.5f` is too low, "
                            "auto-incrementing by `%.5f`",
                            log_context,
                            idx,
                            float(damp_percent[idx].item()),
                            damp_auto_increment,
                        )
                if giveup_idx.numel() > 0:
                    if need_hinv_init:
                        Hinv_init.index_copy_(0, giveup_idx, eye.expand(giveup_idx.numel(), -1, -1))
                    Hinv.index_copy_(0, giveup_idx, eye.expand(giveup_idx.numel(), -1, -1))
                    damp_used.index_copy_(0, giveup_idx, damp_percent.index_select(0, giveup_idx))
                    fallback.index_fill_(0, giveup_idx, True)
                    pending.index_fill_(0, giveup_idx, False)
                    for idx in giveup_idx.detach().cpu().tolist():
                        logging.warning(
                            "Quantization%s subgroup=%d: `damp_percent` reached %.5f; falling back to "
                            "identity Hessian inverse for this subgroup. Last Cholesky info: %s",
                            log_context,
                            idx,
                            float(damp_percent[idx].item()),
                            int(last_info[idx].item()),
                        )

        return Hinv_init, Hinv, damp_used, fallback

    def _compute_gradient_terms_batched(self, gradients_sub, Hinv_init, Hinv, enable_gradient_update):
        if enable_gradient_update and self.alpha > 0:
            if Hinv_init is None:
                raise ValueError("Hinv_init is required when GPTQ+ first-order updates are enabled.")
            GHinv_init = torch.bmm(gradients_sub, Hinv_init)
            alpha = self.alpha / (self.rows * self.columns)
            if self.reference_loss <= 0:
                beta = torch.zeros(
                    gradients_sub.shape[:2],
                    device=gradients_sub.device,
                    dtype=gradients_sub.dtype,
                )
            else:
                diag = torch.diagonal(Hinv_init, dim1=-2, dim2=-1).unsqueeze(1)
                c = (gradients_sub * GHinv_init).sum(dim=2) - ((GHinv_init ** 2) / diag).mean(dim=2)
                target = 2 * alpha * self.reference_loss
                c = torch.clamp(c, min=target)
                ratio = torch.where(c > 0, target / c, torch.zeros_like(c))
                ratio = torch.clamp(ratio, min=0.0, max=1.0)
                beta = 1 - torch.sqrt(torch.clamp(1 - ratio, min=0.0))
                beta = torch.nan_to_num(beta, nan=0.0, posinf=1.0, neginf=0.0)
        else:
            beta = torch.zeros(
                gradients_sub.shape[:2],
                device=gradients_sub.device,
                dtype=gradients_sub.dtype,
            )
        beta_view = beta.unsqueeze(-1)
        if enable_gradient_update and self.alpha > 0:
            Z = torch.bmm(gradients_sub, Hinv.transpose(1, 2)) * beta_view
            GHinv = torch.bmm(Z, Hinv)
        else:
            Z = torch.zeros_like(gradients_sub)
            GHinv = torch.zeros_like(gradients_sub)
        return beta, beta_view, Z, GHinv

    def _reduce_scatter_hessian_group_(self, global_group_id: int, block: torch.Tensor) -> torch.Tensor:
        """SUM-reduce a single global Hessian group to the ranks that own it.

        `GROUP_PARALLEL_QUANT=rank` can map one Hessian output group to multiple
        row-owner ranks (for example 8 ranks / 4 GPTQ groups -> two ranks per
        group). `reduce_scatter` supports this by placing the same local partial
        block into every owner slot and stride-0 zeros elsewhere. The output is
        written back into `block` in-place, so the peak is one CxC block plus the
        persistent local `self.H` shard instead of a full GxCxC staging tensor.
        """
        if not self.hessian_group_sharded or dist_utils.get_world_size() <= 1:
            return block

        owners = self.hessian_group_owner_slots[int(global_group_id)]
        if not owners:
            raise RuntimeError(f"Hessian group {global_group_id} has no owner ranks.")
        if self.hessian_group_zero is None or self.hessian_group_zero.shape != block.shape:
            self.hessian_group_zero = torch.zeros(
                (),
                device=block.device,
                dtype=block.dtype,
            ).expand_as(block)
        input_list = [
            block if rank in owners else self.hessian_group_zero
            for rank in range(dist_utils.get_world_size())
        ]
        dist.reduce_scatter(block, input_list, op=dist.ReduceOp.SUM)
        return block

    @staticmethod
    def _make_grad_optimizer_state_batched(
        weight_sub, grad_optimizer, curvature_full=None,
        warm_start_steps=0, adam_beta2=0.999,
    ):
        if grad_optimizer not in {"sgd", "adam", "warm_adam"}:
            raise ValueError(
                f"Unsupported `grad_optimizer={grad_optimizer}`. "
                "Expected one of: sgd, adam, warm_adam."
            )
        state = {"type": grad_optimizer, "step": 0}
        if grad_optimizer == "adam":
            state["exp_avg"] = torch.zeros_like(weight_sub)
            state["exp_avg_sq"] = torch.zeros_like(weight_sub)
        elif grad_optimizer == "warm_adam":
            # Vanilla Adam in every respect except where exp_avg_sq starts.
            # Adam begins at zero and needs t observations before v is worth
            # anything; here the pre-pass supplies the exact value, and
            # `v_offset` records how much evidence that value carries.
            #
            # Scaling: Adam's recursion produces v_t = P(1 - beta2^t) when every
            # observation equals P, and the bias correction divides by
            # (1 - beta2^t). Seeding v_0 = P raw and correcting by (1 - beta2^1)
            # would inflate it ~1000x, so seed P(1 - beta2^t0) and let the
            # correction read it back as exactly P.
            #
            # The offset belongs to v ALONE. `exp_avg` really does start from
            # zero with no prior evidence behind it, so it needs the full,
            # unmodified bias_correction1 of a cold start. Folding t0 into the
            # shared `step` would tell m's correction it has t0 observations it
            # does not have, shrinking the first update to
            # (1 - beta1) / (1 - beta1^(t0+1)) of its intended size -- 0.53x at
            # t0=1, 0.24x at t0=4 -- recovering only after ~10 of the ~23 steps
            # a module gets. That trades v's cold start for a cold start in m.
            if curvature_full is None:
                raise ValueError(
                    "`warm_adam` requires `curvature_full` (the pre-pass prior)."
                )
            prior = curvature_full.to(device=weight_sub.device, dtype=torch.float32)
            if prior.shape != weight_sub.shape:
                raise ValueError(
                    f"warm_adam prior shape {tuple(prior.shape)} does not match "
                    f"the batched weight block {tuple(weight_sub.shape)}; the "
                    "caller must align group / row-shard / actorder layout "
                    "before this point."
                )
            t0 = max(int(warm_start_steps), 0)
            state["exp_avg"] = torch.zeros_like(weight_sub)
            state["exp_avg_sq"] = prior * (1.0 - adam_beta2 ** t0)
            state["v_offset"] = t0
        return state

    @staticmethod
    def _clear_grad_optimizer_state_batched(opt_state, col_start, col_end):
        if opt_state is None or col_end <= col_start:
            return
        if opt_state["type"] in ("adam", "warm_adam"):
            opt_state["exp_avg"][:, :, col_start:col_end].zero_()
            opt_state["exp_avg_sq"][:, :, col_start:col_end].zero_()

    @staticmethod
    def _compute_grad_optimizer_update_batched(
        opt_state,
        grad_sub,
        col_start,
        lr,
        grad_clip=1.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        step_scale=None,
    ):
        grad_slice = grad_sub[:, :, col_start:]
        if grad_slice.numel() == 0:
            return grad_slice
        if grad_clip is not None and grad_clip > 0:
            grad_slice = grad_slice.clamp(min=-grad_clip, max=grad_clip)
        if opt_state["type"] == "sgd":
            upd = lr * grad_slice
            return upd if step_scale is None else upd * step_scale

        opt_state["step"] += 1
        if (
            opt_state["type"] == "warm_adam"
            and opt_state["step"] == 1
            and os.environ.get("WARM_ADAM_DEBUG") == "1"
        ):
            _dbg_warm_prior_vs_gradient(
                opt_state["exp_avg_sq"][:, :, col_start:], grad_slice,
                opt_state.get("v_offset", 0), adam_beta2, "batched",
            )
        exp_avg = opt_state["exp_avg"][:, :, col_start:]
        exp_avg_sq = opt_state["exp_avg_sq"][:, :, col_start:]
        exp_avg.mul_(adam_beta1).add_(grad_slice, alpha=1 - adam_beta1)
        exp_avg_sq.mul_(adam_beta2).addcmul_(grad_slice, grad_slice, value=1 - adam_beta2)
        bias_correction1 = 1 - adam_beta1 ** opt_state["step"]
        # `v_offset` is warm_adam's prior evidence count and is absent (0) for
        # plain adam, so this is the ordinary correction unless a prior was
        # seeded. It must not reach bias_correction1 -- see the note in
        # _make_grad_optimizer_state.
        bias_correction2 = 1 - adam_beta2 ** (
            opt_state["step"] + opt_state.get("v_offset", 0)
        )
        denom = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
        denom.add_(adam_eps)
        step_size = lr / bias_correction1
        upd = step_size * (exp_avg / denom)
        # step_scale broadcasts over the trailing (column) axis; see
        # _horizon_step_weights. None keeps this byte-identical to production.
        return upd if step_scale is None else upd * step_scale

    @staticmethod
    def _current_ghinv(base, current_weight, ref_weight, beta_view, refresh_mode):
        if refresh_mode == "frozen":
            return base
        return base + beta_view * (current_weight - ref_weight)

    @staticmethod
    def _make_grad_optimizer_state(
        weight_sub, grad_optimizer, curvature_full=None,
        warm_start_steps=0, adam_beta2=0.999,
    ):
        if grad_optimizer not in {"sgd", "adam", "warm_adam"}:
            raise ValueError(
                f"Unsupported `grad_optimizer={grad_optimizer}`. "
                "Expected one of: sgd, adam, warm_adam."
            )
        state = {"type": grad_optimizer, "step": 0}
        if grad_optimizer == "adam":
            state["exp_avg"] = torch.zeros_like(weight_sub)
            state["exp_avg_sq"] = torch.zeros_like(weight_sub)
        elif grad_optimizer == "warm_adam":
            # Vanilla Adam in every respect except where exp_avg_sq starts.
            # Adam begins at zero and needs t observations before v is worth
            # anything; here the pre-pass supplies the exact value, and
            # `v_offset` records how much evidence that value carries.
            #
            # Scaling: Adam's recursion produces v_t = P(1 - beta2^t) when every
            # observation equals P, and the bias correction divides by
            # (1 - beta2^t). Seeding v_0 = P raw and correcting by (1 - beta2^1)
            # would inflate it ~1000x, so seed P(1 - beta2^t0) and let the
            # correction read it back as exactly P.
            #
            # The offset belongs to v ALONE. `exp_avg` really does start from
            # zero with no prior evidence behind it, so it needs the full,
            # unmodified bias_correction1 of a cold start. Folding t0 into the
            # shared `step` would tell m's correction it has t0 observations it
            # does not have, shrinking the first update to
            # (1 - beta1) / (1 - beta1^(t0+1)) of its intended size -- 0.53x at
            # t0=1, 0.24x at t0=4 -- recovering only after ~10 of the ~23 steps
            # a module gets. That trades v's cold start for a cold start in m.
            if curvature_full is None:
                raise ValueError(
                    "`warm_adam` requires `curvature_full` (the pre-pass prior)."
                )
            prior = curvature_full.to(device=weight_sub.device, dtype=torch.float32)
            if prior.shape != weight_sub.shape:
                raise ValueError(
                    f"warm_adam prior shape {tuple(prior.shape)} does not match "
                    f"the weight block {tuple(weight_sub.shape)}; the caller "
                    "must align row-slice / actorder layout before this point."
                )
            t0 = max(int(warm_start_steps), 0)
            state["exp_avg"] = torch.zeros_like(weight_sub)
            state["exp_avg_sq"] = prior * (1.0 - adam_beta2 ** t0)
            state["v_offset"] = t0
        return state

    @staticmethod
    def _clear_grad_optimizer_state(opt_state, col_start, col_end):
        if opt_state is None or col_end <= col_start:
            return
        if opt_state["type"] in ("adam", "warm_adam"):
            opt_state["exp_avg"][:, col_start:col_end].zero_()
            opt_state["exp_avg_sq"][:, col_start:col_end].zero_()

    @staticmethod
    def _compute_grad_optimizer_update(
        opt_state,
        grad_sub,
        col_start,
        lr,
        grad_clip=1.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
    ):
        grad_slice = grad_sub[:, col_start:]
        if grad_slice.numel() == 0:
            return grad_slice
        if grad_clip is not None and grad_clip > 0:
            grad_slice = grad_slice.clamp(min=-grad_clip, max=grad_clip)
        if opt_state["type"] == "sgd":
            return lr * grad_slice

        opt_state["step"] += 1
        if (
            opt_state["type"] == "warm_adam"
            and opt_state["step"] == 1
            and os.environ.get("WARM_ADAM_DEBUG") == "1"
        ):
            _dbg_warm_prior_vs_gradient(
                opt_state["exp_avg_sq"][:, col_start:], grad_slice,
                opt_state.get("v_offset", 0), adam_beta2, "plain",
            )
        exp_avg = opt_state["exp_avg"][:, col_start:]
        exp_avg_sq = opt_state["exp_avg_sq"][:, col_start:]
        exp_avg.mul_(adam_beta1).add_(grad_slice, alpha=1 - adam_beta1)
        exp_avg_sq.mul_(adam_beta2).addcmul_(grad_slice, grad_slice, value=1 - adam_beta2)
        bias_correction1 = 1 - adam_beta1 ** opt_state["step"]
        # `v_offset` is warm_adam's prior evidence count and is absent (0) for
        # plain adam, so this is the ordinary correction unless a prior was
        # seeded. It must not reach bias_correction1 -- see the note in
        # _make_grad_optimizer_state.
        bias_correction2 = 1 - adam_beta2 ** (
            opt_state["step"] + opt_state.get("v_offset", 0)
        )
        denom = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
        denom.add_(adam_eps)
        step_size = lr / bias_correction1
        return step_size * (exp_avg / denom)

    @staticmethod
    def _validate_grad_regularizer(
        grad_reg_strategy,
        grad_reg_lambda=0.0,
        grad_gate_floor=0.1,
        grad_gate_sharpness=1.0,
        grad_gate_sine_amp=0.0,
    ):
        valid = {"none", "l2", "hessian", "quant_error_gate", "quant_error_gate_optimized"}
        if grad_reg_strategy not in valid:
            raise ValueError(
                f"Unsupported `grad_reg_strategy={grad_reg_strategy}`. Expected one of: {', '.join(sorted(valid))}."
            )
        if grad_reg_lambda < 0:
            raise ValueError(f"`grad_reg_lambda` must be non-negative. Got {grad_reg_lambda}.")
        if not (0.0 <= grad_gate_floor <= 1.0):
            raise ValueError(f"`grad_gate_floor` must be in [0, 1]. Got {grad_gate_floor}.")
        if grad_gate_sharpness < 0:
            raise ValueError(f"`grad_gate_sharpness` must be non-negative. Got {grad_gate_sharpness}.")
        if grad_gate_sine_amp < 0:
            raise ValueError(f"`grad_gate_sine_amp` must be non-negative. Got {grad_gate_sine_amp}.")

    @staticmethod
    def _build_gate_quant_maps(base_quantizer, weight_sub, groupsize, row_start, row_end, groups=None, perm=None):
        gate_scale = torch.zeros_like(weight_sub)
        gate_zero = torch.zeros_like(weight_sub)
        if groupsize == -1:
            gate_scale.copy_(
                base_quantizer.scale.to(weight_sub.device)[row_start:row_end].expand_as(weight_sub)
            )
            gate_zero.copy_(
                base_quantizer.zero.to(weight_sub.device)[row_start:row_end].expand_as(weight_sub)
            )
            return gate_scale, gate_zero

        if groups is None:
            raise ValueError("`groups` must be provided when groupsize != -1.")

        device = weight_sub.device
        if perm is None:
            group_ids = torch.arange(weight_sub.shape[1], device=device) // groupsize
        else:
            group_ids = perm.to(device) // groupsize

        for group_id, gate_quantizer in enumerate(groups):
            col_mask = group_ids == group_id
            if not torch.any(col_mask):
                continue
            num_group_cols = int(col_mask.sum().item())
            scale = gate_quantizer.scale.to(device)[row_start:row_end].expand(-1, num_group_cols)
            zero = gate_quantizer.zero.to(device)[row_start:row_end].expand(-1, num_group_cols)
            gate_scale[:, col_mask] = scale
            gate_zero[:, col_mask] = zero
        return gate_scale, gate_zero

    @staticmethod
    def _nearest_quant_grid(weight_slice, scale_slice, zero_slice, maxq, sym):
        scale_slice = scale_slice.clamp(min=1e-8)
        if sym:
            q_int = torch.clamp(torch.round(weight_slice / scale_slice), -(maxq + 1), maxq)
            return scale_slice * q_int
        q_int = torch.clamp(torch.round(weight_slice / scale_slice) + zero_slice, 0, maxq)
        return scale_slice * (q_int - zero_slice)

    @staticmethod
    def _sine_quant_regularizer_update(weight_slice, scale_slice, zero_slice, sym, sine_amp):
        scale_slice = scale_slice.clamp(min=1e-8)
        if sym:
            phase = 2 * math.pi * (weight_slice / scale_slice)
        else:
            phase = 2 * math.pi * ((weight_slice / scale_slice) + zero_slice)
        return sine_amp * (2 * math.pi / scale_slice) * torch.sin(phase)

    def _apply_first_order_regularizer(
        self,
        state,
        current_effective_weight,
        optimizer_update,
        col_start,
        grad_lr,
        grad_reg_strategy,
        grad_reg_lambda,
        grad_gate_floor,
        grad_gate_sharpness,
        grad_gate_sine_amp,
    ):
        if optimizer_update.numel() == 0:
            return optimizer_update, optimizer_update, torch.zeros_like(optimizer_update)

        self._validate_grad_regularizer(
            grad_reg_strategy,
            grad_reg_lambda=grad_reg_lambda,
            grad_gate_floor=grad_gate_floor,
            grad_gate_sharpness=grad_gate_sharpness,
            grad_gate_sine_amp=grad_gate_sine_amp,
        )
        trailing_current = current_effective_weight[:, col_start:]

        if grad_reg_strategy == "none":
            zero_update = torch.zeros_like(optimizer_update)
            return optimizer_update, zero_update, zero_update

        if grad_reg_strategy == "l2":
            ref_trailing = state["full_precision_weight"][:, col_start:]
            reg_update = grad_lr * grad_reg_lambda * (trailing_current - ref_trailing)
            zero_update = torch.zeros_like(reg_update)
            return optimizer_update + reg_update, reg_update, zero_update

        if grad_reg_strategy == "hessian":
            full_diff = current_effective_weight - state["full_precision_weight"]
            reg_update = grad_lr * grad_reg_lambda * full_diff.matmul(state["hessian_reg"][:, col_start:])
            zero_update = torch.zeros_like(reg_update)
            return optimizer_update + reg_update, reg_update, zero_update

        scale_slice = state["gate_scale"][:, col_start:]
        zero_slice = state["gate_zero"][:, col_start:]
        q_nearest = self._nearest_quant_grid(
            trailing_current,
            scale_slice,
            zero_slice,
            int(self.quantizer.maxq.item()),
            self.quantizer.sym,
        )
        distance = (trailing_current - q_nearest).abs() / scale_slice.clamp(min=1e-8)
        gate = grad_gate_floor + (1.0 - grad_gate_floor) * (1.0 - torch.exp(-grad_gate_sharpness * distance))
        gated_update = optimizer_update * gate
        gate_update = gated_update - optimizer_update
        if grad_reg_strategy == "quant_error_gate":
            zero_update = torch.zeros_like(gate_update)
            return gated_update, gate_update, zero_update

        sine_regularizer_update = self._sine_quant_regularizer_update(
            trailing_current,
            scale_slice,
            zero_slice,
            self.quantizer.sym,
            grad_gate_sine_amp,
        )
        sine_update = grad_lr * sine_regularizer_update
        return gated_update + sine_update, gate_update, sine_update

    def _apply_first_order_regularizer_batched(
        self,
        state,
        current_effective_weight,
        optimizer_update,
        col_start,
        grad_lr,
        grad_reg_strategy,
        grad_reg_lambda,
        grad_gate_floor,
        grad_gate_sharpness,
        grad_gate_sine_amp,
    ):
        if optimizer_update.numel() == 0:
            return optimizer_update, optimizer_update, torch.zeros_like(optimizer_update)

        self._validate_grad_regularizer(
            grad_reg_strategy,
            grad_reg_lambda=grad_reg_lambda,
            grad_gate_floor=grad_gate_floor,
            grad_gate_sharpness=grad_gate_sharpness,
            grad_gate_sine_amp=grad_gate_sine_amp,
        )
        trailing_current = current_effective_weight[:, :, col_start:]

        if grad_reg_strategy == "none":
            zero_update = torch.zeros_like(optimizer_update)
            return optimizer_update, zero_update, zero_update

        if grad_reg_strategy == "l2":
            ref_trailing = state["full_precision_weight"][:, :, col_start:]
            reg_update = grad_lr * grad_reg_lambda * (trailing_current - ref_trailing)
            zero_update = torch.zeros_like(reg_update)
            return optimizer_update + reg_update, reg_update, zero_update

        if grad_reg_strategy == "hessian":
            full_diff = current_effective_weight - state["full_precision_weight"]
            reg_update = grad_lr * grad_reg_lambda * torch.bmm(
                full_diff,
                state["hessian_reg"][:, :, col_start:],
            )
            zero_update = torch.zeros_like(reg_update)
            return optimizer_update + reg_update, reg_update, zero_update

        scale_slice = state["gate_scale"][:, :, col_start:]
        zero_slice = state["gate_zero"][:, :, col_start:]
        q_nearest = self._nearest_quant_grid(
            trailing_current,
            scale_slice,
            zero_slice,
            int(self.quantizer.maxq.item()),
            self.quantizer.sym,
        )
        distance = (trailing_current - q_nearest).abs() / scale_slice.clamp(min=1e-8)
        gate = grad_gate_floor + (1.0 - grad_gate_floor) * (1.0 - torch.exp(-grad_gate_sharpness * distance))
        gated_update = optimizer_update * gate
        gate_update = gated_update - optimizer_update
        if grad_reg_strategy == "quant_error_gate":
            zero_update = torch.zeros_like(gate_update)
            return gated_update, gate_update, zero_update

        sine_regularizer_update = self._sine_quant_regularizer_update(
            trailing_current,
            scale_slice,
            zero_slice,
            self.quantizer.sym,
            grad_gate_sine_amp,
        )
        sine_update = grad_lr * sine_regularizer_update
        return gated_update + sine_update, gate_update, sine_update

    @staticmethod
    def _find_weight_params_row_parallel(quantizer, weight, row_start, row_end):
        local_quantizer = copy.deepcopy(quantizer)
        local_quantizer.find_params(weight[row_start:row_end])

        scale = torch.zeros(
            (weight.shape[0],) + tuple(local_quantizer.scale.shape[1:]),
            device=weight.device,
            dtype=local_quantizer.scale.dtype,
        )
        zero = torch.zeros(
            (weight.shape[0],) + tuple(local_quantizer.zero.shape[1:]),
            device=weight.device,
            dtype=local_quantizer.zero.dtype,
        )
        scale[row_start:row_end] = local_quantizer.scale
        zero[row_start:row_end] = local_quantizer.zero
        dist_utils.allreduce_sum_(scale)
        dist_utils.allreduce_sum_(zero)
        quantizer.scale = scale
        quantizer.zero = zero
        quantizer.maxq = quantizer.maxq.to(weight.device)

    def _fasterquant_group_parallel(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        enable_gradient_update=True,
        g_update_mode="frozen",
        export_to_et=False,
        profile_recorder=None,
        gradient_refresh_fn=None,
        grad_lr=1e-3,
        grad_optimizer="sgd",
        grad_reg_strategy="none",
        grad_reg_lambda=0.0,
        grad_gate_floor=0.1,
        grad_gate_sharpness=1.0,
        grad_gate_sine_amp=0.0,
        second_order_scale=1.0,
        block_atomic_quant=False,
        block_observer=None,
        grad_clip=1.0,
        horizon_p=0.0,
        slide_refresh_start=0,
        slide_refresh_block_total=None,
        refresh_full_metrics=False,
        group_parallel_mode="tensor",
        grad_sq_full=None,
        warm_start_steps=0,
    ):
        W = self.layer.weight.data.clone().float()
        block_gd_mode = g_update_mode == "block_gd"
        dynamic_groups = groupsize != -1 and not static_groups
        if dynamic_groups:
            raise ValueError("group-parallel fasterquant does not support dynamic weight groups.")
        if not self.quantizer.sym:
            raise ValueError("group-parallel fasterquant currently supports symmetric weight quantization only.")
        if grad_reg_strategy in {"quant_error_gate", "quant_error_gate_optimized"}:
            raise ValueError("group-parallel fasterquant does not support quantization-error gate regularizers yet.")
        if self.quantizer.bits >= 16:
            raise ValueError("group-parallel fasterquant requires weight quantization bits < 16.")
        if g_update_mode not in {"frozen", "surrogate_block", "surrogate_online", "block_backward", "block_gd"}:
            raise ValueError(f"Unsupported `g_update_mode={g_update_mode}`.")

        with profile_recorder.section("fasterquant_group_parallel.total") if profile_recorder else _NULL_CONTEXT:
            rows_per_sub = self.rows // self.num_groups
            dev = self.dev
            G = self.num_groups
            R = rows_per_sub
            C = self.columns
            if group_parallel_mode == "rank" and dist_utils.get_world_size() > 1:
                world = dist_utils.get_world_size()
                rank = dist_utils.get_rank()
                if self.rows % world != 0:
                    raise ValueError(
                        "group_parallel_mode='rank' requires output rows "
                        f"({self.rows}) to be divisible by world_size ({world})."
                    )
                local_row_start = rank * self.rows // world
                local_row_end = (rank + 1) * self.rows // world
            else:
                world = 1
                local_row_start = 0
                local_row_end = self.rows
            local_rows = torch.arange(local_row_start, local_row_end, device=dev)
            local_group_idx = torch.div(local_rows, R, rounding_mode="floor").to(torch.long)
            local_row_idx = torch.remainder(local_rows, R).to(torch.long)
            has_local_rows = local_rows.numel() > 0
            use_hessian_group_shard = (
                self.hessian_group_sharded
                and group_parallel_mode == "rank"
                and dist_utils.get_world_size() > 1
            )
            if use_hessian_group_shard:
                hessian_group_ids = self.hessian_group_ids.to(dev)
                hessian_group_to_pos = self.hessian_group_to_pos.to(dev)
                local_hessian_group_idx = hessian_group_to_pos[local_group_idx]
                if has_local_rows and bool((local_hessian_group_idx < 0).any().item()):
                    raise RuntimeError(
                        "Hessian group shard does not cover every local row group: "
                        f"rank={dist_utils.get_rank()} local_groups="
                        f"{torch.unique(local_group_idx).detach().cpu().tolist()} "
                        f"hessian_groups={self.hessian_group_ids.tolist()}."
                    )
            else:
                hessian_group_ids = torch.arange(G, device=dev, dtype=torch.long)
                hessian_group_to_pos = torch.arange(G, device=dev, dtype=torch.long)
                local_hessian_group_idx = local_group_idx

            if not self.quantizer.ready():
                with profile_recorder.section("fasterquant_group_parallel.quantizer_find_params_initial") if profile_recorder else _NULL_CONTEXT:
                    if group_parallel_mode == "rank" and dist_utils.get_world_size() > 1:
                        self._find_weight_params_row_parallel(
                            self.quantizer,
                            W,
                            local_row_start,
                            local_row_end,
                        )
                    else:
                        self.quantizer.find_params(W)

            shared_groups = None
            if groupsize != -1 and static_groups:
                with profile_recorder.section("fasterquant_group_parallel.quantizer_build_groups") if profile_recorder else _NULL_CONTEXT:
                    shared_groups = []
                    for col_start in range(0, self.columns, groupsize):
                        col_end = min(col_start + groupsize, self.columns)
                        quantizer = copy.deepcopy(self.quantizer)
                        if group_parallel_mode == "rank" and dist_utils.get_world_size() > 1:
                            self._find_weight_params_row_parallel(
                                quantizer,
                                W[:, col_start:col_end],
                                local_row_start,
                                local_row_end,
                            )
                        else:
                            quantizer.find_params(W[:, col_start:col_end])
                        shared_groups.append(quantizer)

            maxq = int(self.quantizer.maxq.item())
            q_lo = -(maxq + 1)

            with profile_recorder.section("fasterquant_group_parallel.prepare_state") if profile_recorder else _NULL_CONTEXT:
                W_sub = W.reshape(G, R, C).clone()
                H_sub = self.H.clone()
                gradients_sub = self.gradients.to(dev).float().reshape(G, R, C).clone()
                gradients_hessian = gradients_sub.index_select(0, hessian_group_ids).contiguous()

                diag_idx = torch.arange(C, device=dev)
                dead = torch.diagonal(H_sub, dim1=-2, dim2=-1) == 0
                H_diag = torch.diagonal(H_sub, dim1=-2, dim2=-1)
                H_diag[dead] = 1
                if use_hessian_group_shard:
                    W_sub[hessian_group_ids] = W_sub[hessian_group_ids].masked_fill(dead.unsqueeze(1), 0)
                else:
                    W_sub = W_sub.masked_fill(dead.unsqueeze(1), 0)

                perm = None
                invperm = None
                if actorder:
                    perm = torch.argsort(self.act_square, descending=True)
                    W_sub = W_sub[:, :, perm]
                    H_sub = H_sub[:, perm][:, :, perm]
                    gradients_sub = gradients_sub[:, :, perm]
                    gradients_hessian = gradients_hessian[:, :, perm]
                    invperm = torch.argsort(perm)

                hessian_reg = H_sub.clone() if grad_reg_strategy == "hessian" else None
                if grad_reg_strategy in {"l2", "hessian"}:
                    full_precision_weight = (
                        W_sub.index_select(0, hessian_group_ids).clone()
                        if use_hessian_group_shard else W_sub.clone()
                    )
                else:
                    full_precision_weight = None
                anchor_weight = W_sub.clone()
                Q = torch.zeros_like(W_sub)
                W_int_sub = torch.zeros_like(W_sub)
                Scale_sub = torch.zeros_like(W_sub)

            with profile_recorder.section("fasterquant_group_parallel.compute_hinv") if profile_recorder else _NULL_CONTEXT:
                Hinv_init, Hinv, damp_percent, hessian_identity_fallback = (
                    self._compute_hessian_inverse_batched_with_fallback(
                        H_sub,
                        percdamp=percdamp,
                        profile_recorder=profile_recorder,
                        profile_section="fasterquant_group_parallel.compute_hinv.cholesky",
                        need_hinv_init=bool(enable_gradient_update and self.alpha > 0),
                    )
                )
            with profile_recorder.section("fasterquant_group_parallel.init_ghinv") if profile_recorder else _NULL_CONTEXT:
                beta, beta_view, Z, GHinv = self._compute_gradient_terms_batched(
                    gradients_hessian,
                    Hinv_init,
                    Hinv,
                    enable_gradient_update,
                )

            state = {
                "W_sub": W_sub,
                "Q": Q,
                "W_int_sub": W_int_sub,
                "Scale_sub": Scale_sub,
                "gradients_sub": gradients_hessian,
                "Hinv_init": Hinv_init,
                "Hinv": Hinv,
                "Z": Z,
                "GHinv": GHinv,
                "beta": beta,
                "beta_view": beta_view,
                "anchor_weight": anchor_weight,
                "full_precision_weight": full_precision_weight,
                "hessian_reg": hessian_reg,
                "gate_scale": None,
                "gate_zero": None,
                "grad_optimizer_state": (
                    self._make_grad_optimizer_state_batched(
                        W_sub.index_select(0, hessian_group_ids) if use_hessian_group_shard else W_sub,
                        grad_optimizer,
                        curvature_full=_align_warm_prior_batched(
                            grad_sq_full, G, R, C, perm,
                            hessian_group_ids if use_hessian_group_shard else None,
                            W_sub.device,
                        ),
                        warm_start_steps=warm_start_steps,
                    )
                    if block_gd_mode else None
                ),
            }

            base_scale = self.quantizer.scale.to(dev).reshape(G, R, -1)
            if base_scale.shape[-1] != 1:
                base_scale = base_scale[:, :, :1]
            static_group_scales = None
            if shared_groups is not None:
                static_group_scales = torch.stack(
                    [q.scale.to(dev).reshape(G, R, -1)[:, :, :1] for q in shared_groups],
                    dim=0,
                )

            def scale_for_block(col_start, col_end):
                count = col_end - col_start
                if groupsize == -1:
                    return base_scale.expand(-1, -1, count)
                scale = torch.empty((G, R, count), device=dev, dtype=W_sub.dtype)
                cols = torch.arange(col_start, col_end, device=dev)
                original_cols = perm[cols] if perm is not None else cols
                group_ids = torch.div(original_cols, groupsize, rounding_mode="floor").to(torch.long)
                for local_col, group_id in enumerate(group_ids.detach().cpu().tolist()):
                    scale[:, :, local_col] = static_group_scales[group_id].squeeze(-1)
                return scale

            def sync_block_tensors(Q1, W_int1, Scale1, Err1):
                if group_parallel_mode != "rank" or dist_utils.get_world_size() <= 1:
                    return Q1, W_int1, Scale1, Err1
                with profile_recorder.section("fasterquant_group_parallel.block.rank_all_gather") if profile_recorder else _NULL_CONTEXT:
                    local_packed = torch.cat(
                        [
                            tensor.reshape(self.rows, count)[local_row_start:local_row_end]
                            for tensor in (Q1, W_int1, Scale1, Err1)
                        ],
                        dim=1,
                    ).contiguous()
                    gathered = [torch.empty_like(local_packed) for _ in range(world)]
                    dist.all_gather(gathered, local_packed)
                    packed = torch.cat(gathered, dim=0)
                    return [
                        piece.contiguous().view(G, R, count)
                        for piece in packed.split(count, dim=1)
                    ]

            def sync_weight_rows_from_col_(weight_groups, col_start=0):
                if group_parallel_mode != "rank" or dist_utils.get_world_size() <= 1:
                    return weight_groups
                col_start = max(0, min(int(col_start), self.columns))
                if col_start >= self.columns:
                    return weight_groups
                full_rows = weight_groups.reshape(self.rows, self.columns)
                local_rows = full_rows[local_row_start:local_row_end]
                if col_start <= 0:
                    local_full_rows = local_rows.clone().contiguous()
                    with profile_recorder.section("fasterquant_group_parallel.rank_all_gather_weight") if profile_recorder else _NULL_CONTEXT:
                        dist.all_gather_into_tensor(full_rows, local_full_rows)
                else:
                    local_tail = local_rows[:, col_start:].clone().contiguous()
                    gathered_tail = torch.empty(
                        (self.rows, self.columns - col_start),
                        device=full_rows.device,
                        dtype=full_rows.dtype,
                    )
                    with profile_recorder.section("fasterquant_group_parallel.rank_all_gather_weight_tail") if profile_recorder else _NULL_CONTEXT:
                        dist.all_gather_into_tensor(gathered_tail, local_tail)
                    full_rows[:, col_start:].copy_(gathered_tail)
                return weight_groups

            def natural_order(weight_groups):
                if actorder:
                    weight_groups = weight_groups[:, :, invperm]
                return weight_groups.reshape(self.rows, self.columns)

            n_blocks_total = (C + blocksize - 1) // blocksize
            n_refresh_total = max(n_blocks_total - 1, 0)

            # See _reach_probe_block_row. The three buffers are the size of
            # W_sub, which is why this is opt-in rather than always on.
            _rp_on = os.environ.get("REACH_PROBE") == "1" and enable_gradient_update
            if _rp_on:
                _rp_w0 = state["W_sub"].detach().clone().float()
                _rp_disp = torch.zeros_like(_rp_w0)
                _rp_path = torch.zeros_like(_rp_w0)
                _rp_rows = []

            for i1 in range(0, C, blocksize):
                with profile_recorder.section("fasterquant_group_parallel.block.total") if profile_recorder else _NULL_CONTEXT:
                    i2 = min(i1 + blocksize, C)
                    count = i2 - i1
                    is_last_block = i2 >= C
                    use_atomic_quant = block_atomic_quant and not is_last_block
                    D = torch.arange(count - 1, -1, -1, device=dev, dtype=W_sub.dtype)

                    with profile_recorder.section("fasterquant_group_parallel.block.setup") if profile_recorder else _NULL_CONTEXT:
                        W1 = state["W_sub"][:, :, i1:i2].clone()
                        W_ref1 = state["anchor_weight"][:, :, i1:i2]
                        W_block_start = W1.clone()
                        Q1 = torch.zeros_like(W1)
                        W_int1 = torch.zeros_like(W1)
                        Scale1 = torch.zeros_like(W1)
                        Err1 = torch.zeros_like(W1)
                        Hinv1 = state["Hinv"][:, i1:i2, i1:i2]
                        GHinv1 = state["GHinv"][:, :, i1:i2].clone()
                        Z1 = state["Z"][:, :, i1:i2]
                        inner_update_mode = "surrogate_online" if g_update_mode == "block_backward" else "frozen" if block_gd_mode else g_update_mode
                        is_frozen_inner = inner_update_mode == "frozen"
                        is_surrogate_online = inner_update_mode == "surrogate_online"
                        if use_hessian_group_shard:
                            W1_hessian = W1.index_select(0, hessian_group_ids)
                            W_ref1_hessian = W_ref1.index_select(0, hessian_group_ids)
                            W_block_start_hessian = W_block_start.index_select(0, hessian_group_ids)
                        else:
                            W1_hessian = W1
                            W_ref1_hessian = W_ref1
                            W_block_start_hessian = W_block_start
                        GHinv1_eff = self._current_ghinv(
                            GHinv1,
                            W1_hessian if is_surrogate_online else W_block_start_hessian,
                            W_ref1_hessian,
                            state["beta_view"],
                            inner_update_mode,
                        )
                        Scale_block = scale_for_block(i1, i2)
                        if has_local_rows:
                            Scale1[local_group_idx, local_row_idx, :] = Scale_block[
                                local_group_idx,
                                local_row_idx,
                                :,
                            ]

                    if use_atomic_quant:
                        with profile_recorder.section("fasterquant_group_parallel.block.atomic_inner") if profile_recorder else _NULL_CONTEXT:
                            if has_local_rows:
                                scale_local = Scale_block[local_group_idx, local_row_idx, :]
                                W_block_start_l = W_block_start[local_group_idx, local_row_idx, :]
                                GHinv1_eff_l = GHinv1_eff[local_hessian_group_idx, local_row_idx, :]
                                q_int = torch.clamp(
                                    torch.round(W_block_start_l / scale_local),
                                    q_lo,
                                    maxq,
                                )
                                q = (scale_local * q_int).to(W_block_start.dtype)
                                Q1[local_group_idx, local_row_idx, :] = q
                                W_int1[local_group_idx, local_row_idx, :] = q_int
                                residual_block = W_block_start_l - q - GHinv1_eff_l
                                solved = torch.linalg.solve_triangular(
                                    Hinv1[local_hessian_group_idx].transpose(1, 2),
                                    residual_block.unsqueeze(-1),
                                    upper=False,
                                ).squeeze(-1)
                                Err1[local_group_idx, local_row_idx, :] = solved
                    else:
                        with profile_recorder.section("fasterquant_group_parallel.block.inner_loop") if profile_recorder else _NULL_CONTEXT:
                            if has_local_rows:
                                W1_l = W1[local_group_idx, local_row_idx, :].clone()
                                Hinv1_l = Hinv1[local_hessian_group_idx]
                                GHinv1_l = GHinv1[local_hessian_group_idx, local_row_idx, :].clone()
                                Z1_l = Z1[local_hessian_group_idx, local_row_idx, :]
                                GHinv1_eff_l = (
                                    GHinv1_l
                                    if is_frozen_inner
                                    else GHinv1_eff[local_hessian_group_idx, local_row_idx, :].clone()
                                )
                                W_block_start_l = W_block_start[local_group_idx, local_row_idx, :]
                                W_ref1_l = W_ref1[local_group_idx, local_row_idx, :]
                                beta_view_l = state["beta_view"][local_hessian_group_idx, local_row_idx, :]
                                scale_l = Scale_block[local_group_idx, local_row_idx, :]
                                Q1_l = torch.zeros_like(W1_l)
                                W_int1_l = torch.zeros_like(W1_l)
                                Err1_l = torch.zeros_like(W1_l)
                                for i in range(count):
                                    w = W1_l[:, i]
                                    q_int = torch.clamp(
                                        torch.round(w / scale_l[:, i]),
                                        q_lo,
                                        maxq,
                                    )
                                    q = (scale_l[:, i] * q_int).to(w.dtype)
                                    Q1_l[:, i] = q
                                    W_int1_l[:, i] = q_int
                                    d = Hinv1_l[:, i, i]
                                    err1 = (w - q - GHinv1_eff_l[:, i]) / d
                                    Err1_l[:, i] = err1
                                    second_order_inner_update = (
                                        err1.unsqueeze(1) * Hinv1_l[:, i, i:]
                                    )
                                    if block_gd_mode:
                                        W1_l[:, i:] -= second_order_scale * (
                                            second_order_inner_update + GHinv1_eff_l[:, i:]
                                        )
                                    else:
                                        W1_l[:, i:] -= second_order_inner_update + GHinv1_eff_l[:, i:]
                                    GHinv1_l[:, i:].sub_(
                                        Z1_l[:, i].unsqueeze(1) * Hinv1_l[:, i, i:]
                                    )
                                    if not is_frozen_inner:
                                        GHinv1_eff_l = self._current_ghinv(
                                            GHinv1_l,
                                            W1_l if is_surrogate_online else W_block_start_l,
                                            W_ref1_l,
                                            beta_view_l,
                                            inner_update_mode,
                                        )
                                Q1[local_group_idx, local_row_idx, :] = Q1_l
                                W_int1[local_group_idx, local_row_idx, :] = W_int1_l
                                Err1[local_group_idx, local_row_idx, :] = Err1_l

                    Q1, W_int1, Scale1, Err1 = sync_block_tensors(Q1, W_int1, Scale1, Err1)
                    state["Q"][:, :, i1:i2] = Q1
                    if _rp_on:
                        # These columns are frozen from here on, so whatever the
                        # optimizer moved them by is final.
                        #
                        # Statistics cover this rank's own groups only. Under
                        # group_parallel_quant=rank each rank writes the
                        # optimizer update for its shard alone, so the other
                        # groups' entries in _rp_disp stay at zero -- with 4
                        # groups across 4 ranks that is 3/4 of the tensor, which
                        # drags every median to exactly 0 and reads as "the
                        # optimizer moved nothing". Q1 and Scale1 come back full
                        # from sync_block_tensors, which is why those rows
                        # looked healthy while |disp| did not.
                        # hessian_group_ids is arange(G) when unsharded, so this
                        # is a no-op in that case.
                        _g = hessian_group_ids
                        _rp_rows.append(_reach_probe_block_row(
                            i1 // blocksize,
                            _rp_w0[:, :, i1:i2].index_select(0, _g),
                            Q1.detach().float().index_select(0, _g),
                            _rp_disp[:, :, i1:i2].index_select(0, _g),
                            _rp_path[:, :, i1:i2].index_select(0, _g),
                            Scale1.detach().float().index_select(0, _g),
                        ))
                    state["W_int_sub"][:, :, i1:i2] = W_int1
                    state["Scale_sub"][:, :, i1:i2] = Scale1

                    if g_update_mode == "block_backward" and enable_gradient_update:
                        if gradient_refresh_fn is None:
                            raise ValueError("`gradient_refresh_fn` must be provided for g_update_mode='block_backward'.")
                        with profile_recorder.section("fasterquant_group_parallel.block.block_backward_refresh") if profile_recorder else _NULL_CONTEXT:
                            current_sub_weight = state["W_sub"].clone()
                            if i1 > 0:
                                current_sub_weight[:, :, :i1] = state["Q"][:, :, :i1]
                            current_sub_weight[:, :, i1:i2] = Q1
                            if use_hessian_group_shard:
                                sync_weight_rows_from_col_(current_sub_weight, i2)
                            refreshed_grad, refresh_meta = gradient_refresh_fn(natural_order(current_sub_weight))
                            refreshed_grad = refreshed_grad.to(dev).float().reshape(G, R, C)
                            if actorder:
                                refreshed_grad = refreshed_grad[:, :, perm]
                            refreshed_grad_hessian = (
                                refreshed_grad.index_select(0, hessian_group_ids).contiguous()
                                if use_hessian_group_shard else refreshed_grad
                            )
                            state["gradients_sub"] = refreshed_grad_hessian
                            beta, beta_view, Z, GHinv = self._compute_gradient_terms_batched(
                                refreshed_grad_hessian,
                                state["Hinv_init"],
                                state["Hinv"],
                                enable_gradient_update,
                            )
                            state["beta"] = beta
                            state["beta_view"] = beta_view
                            state["Z"] = Z
                            state["GHinv"] = GHinv
                            state["anchor_weight"] = current_sub_weight.clone()
                            Z1 = Z[:, :, i1:i2]
                            if block_observer is not None:
                                block_observer(
                                    {
                                        "block_idx": i1 // blocksize,
                                        "col_start": i1,
                                        "col_end": i2,
                                        "remaining_columns": C - i2,
                                        "remaining_grad_abs_mean": None,
                                        "remaining_grad_clipped_abs_mean": None,
                                        "remaining_grad_mean_row_l2": None,
                                        "second_order_update_abs_mean": None,
                                        "second_order_update_mean_row_l2": None,
                                        "second_order_update_abs_max": None,
                                        "second_order_update_abs_q99": None,
                                        "first_order_raw_abs_mean": None,
                                        "first_order_raw_mean_row_l2": None,
                                        "first_order_update_abs_mean": None,
                                        "first_order_update_mean_row_l2": None,
                                        "first_order_update_abs_max": None,
                                        "first_order_update_abs_q99": None,
                                        "regularizer_update_abs_mean": None,
                                        "regularizer_update_mean_row_l2": None,
                                        "sine_regularizer_update_abs_mean": None,
                                        "sine_regularizer_update_mean_row_l2": None,
                                        "mean_refresh_loss": None if refresh_meta is None else refresh_meta.get("mean_refresh_loss"),
                                        "train_mean_refresh_loss": None if refresh_meta is None else refresh_meta.get("train_mean_refresh_loss"),
                                        "val_mean_refresh_loss": None if refresh_meta is None else refresh_meta.get("val_mean_refresh_loss"),
                                        "refresh_subset_mean_refresh_loss": None if refresh_meta is None else refresh_meta.get("refresh_subset_mean_refresh_loss"),
                                    }
                                )

                    with profile_recorder.section("fasterquant_group_parallel.block.outer_update") if profile_recorder else _NULL_CONTEXT:
                        outer_mode = "frozen" if g_update_mode in {"frozen", "block_backward", "block_gd"} else "surrogate_block"
                        Hrest = state["Hinv"][:, i1:i2, i2:]
                        if use_hessian_group_shard:
                            W_sub_hessian = state["W_sub"].index_select(0, hessian_group_ids)
                            anchor_hessian = state["anchor_weight"].index_select(0, hessian_group_ids)
                        else:
                            W_sub_hessian = state["W_sub"]
                            anchor_hessian = state["anchor_weight"]
                        GHinv_rest = self._current_ghinv(
                            state["GHinv"][:, :, i2:],
                            W_sub_hessian[:, :, i2:],
                            anchor_hessian[:, :, i2:],
                            state["beta_view"],
                            outer_mode,
                        )
                        G_Update_h = count * GHinv_rest - torch.einsum("grj,j,gjk->grk", Z1, D, Hrest)
                        Err1_h = Err1.index_select(0, hessian_group_ids) if use_hessian_group_shard else Err1
                        second_order_update_h = torch.bmm(Err1_h, Hrest)
                        if use_hessian_group_shard:
                            total_outer_update_h = second_order_update_h + G_Update_h
                            if block_gd_mode:
                                applied_second_order_update = second_order_scale * second_order_update_h
                                state["W_sub"][hessian_group_ids, :, i2:] -= (
                                    second_order_scale * total_outer_update_h
                                )
                            else:
                                applied_second_order_update = second_order_update_h
                                state["W_sub"][hessian_group_ids, :, i2:] -= total_outer_update_h
                        else:
                            second_order_update = second_order_update_h
                            total_outer_update = second_order_update + G_Update_h
                            if block_gd_mode:
                                applied_second_order_update = second_order_scale * second_order_update
                                state["W_sub"][:, :, i2:] -= second_order_scale * total_outer_update
                            else:
                                applied_second_order_update = second_order_update
                                state["W_sub"][:, :, i2:] -= total_outer_update
                        state["GHinv"][:, :, i2:] -= torch.bmm(state["Z"][:, :, i1:i2], Hrest)
                        if block_gd_mode:
                            self._clear_grad_optimizer_state_batched(state["grad_optimizer_state"], i1, i2)

                    if block_gd_mode and enable_gradient_update and i2 < C:
                        if gradient_refresh_fn is None:
                            raise ValueError("`gradient_refresh_fn` must be provided for g_update_mode='block_gd'.")
                        with profile_recorder.section("fasterquant_group_parallel.block.block_gd_refresh") if profile_recorder else _NULL_CONTEXT:
                            current_sub_weight = state["W_sub"].clone()
                            current_sub_weight[:, :, :i2] = state["Q"][:, :, :i2]
                            if use_hessian_group_shard:
                                sync_weight_rows_from_col_(current_sub_weight, i2)
                            refresh_idx = i1 // blocksize
                            effective_total = (
                                slide_refresh_block_total
                                if slide_refresh_block_total is not None
                                else n_refresh_total
                            )
                            effective_idx = slide_refresh_start + refresh_idx
                            slide_alpha = (
                                1.0 - effective_idx / max(effective_total - 1, 1)
                                if effective_total > 1 else 1.0
                            )
                            refreshed_grad, refresh_meta = gradient_refresh_fn(
                                natural_order(current_sub_weight),
                                slide_alpha=slide_alpha,
                            )
                            refreshed_grad = refreshed_grad.to(dev).float().reshape(G, R, C)
                            if actorder:
                                refreshed_grad = refreshed_grad[:, :, perm]

                            trailing_grad_abs_mean = None
                            trailing_grad_mean_row_l2 = None
                            trailing_grad_clipped_abs_mean = None
                            second_order_abs_mean = None
                            second_order_mean_row_l2 = None
                            second_order_abs_max = None
                            second_order_abs_q99 = None
                            first_order_raw_abs_mean = None
                            first_order_raw_mean_row_l2 = None
                            first_order_abs_mean = None
                            first_order_mean_row_l2 = None
                            first_order_abs_max = None
                            first_order_abs_q99 = None
                            regularizer_abs_mean = None
                            regularizer_mean_row_l2 = None
                            sine_regularizer_abs_mean = None
                            sine_regularizer_mean_row_l2 = None

                            if refresh_full_metrics:
                                trailing_grad = refreshed_grad[:, :, i2:]
                                if trailing_grad.numel() > 0:
                                    trailing_grad_2d = trailing_grad.reshape(-1, trailing_grad.shape[-1]).float()
                                    trailing_grad_abs_mean = trailing_grad_2d.abs().mean().item()
                                    trailing_grad_mean_row_l2 = torch.linalg.norm(trailing_grad_2d, dim=1).mean().item()
                                    if grad_clip is not None and grad_clip > 0:
                                        trailing_grad_clipped_abs_mean = (
                                            trailing_grad_2d.clamp(min=-grad_clip, max=grad_clip).abs().mean().item()
                                        )
                                    else:
                                        trailing_grad_clipped_abs_mean = trailing_grad_abs_mean
                                if applied_second_order_update.numel() > 0:
                                    second_2d = applied_second_order_update.reshape(
                                        -1,
                                        applied_second_order_update.shape[-1],
                                    ).float()
                                    second_abs = second_2d.abs()
                                    second_order_abs_mean = second_abs.mean().item()
                                    second_order_mean_row_l2 = torch.linalg.norm(second_2d, dim=1).mean().item()
                                    second_order_abs_max = second_abs.max().item()
                                    second_order_abs_q99 = _quantile_large(second_abs, 0.99)

                            # i2 is a whole number of blocks here (the last
                            # block never refreshes), so i2 // blocksize is both
                            # the first tail column's block index and the number
                            # of updates that column receives in total.
                            _hstep = _horizon_step_weights(
                                C - i2, blocksize, i2 // blocksize, horizon_p,
                                state["W_sub"].device, state["W_sub"].dtype,
                            )
                            optimizer_update_raw = self._compute_grad_optimizer_update_batched(
                                state["grad_optimizer_state"],
                                (
                                    refreshed_grad.index_select(0, hessian_group_ids).contiguous()
                                    if use_hessian_group_shard else refreshed_grad
                                ),
                                i2,
                                grad_lr,
                                grad_clip=grad_clip,
                                step_scale=_hstep,
                            )
                            optimizer_update, gate_regularizer_update, sine_regularizer_update = self._apply_first_order_regularizer_batched(
                                state,
                                (
                                    current_sub_weight.index_select(0, hessian_group_ids)
                                    if use_hessian_group_shard else current_sub_weight
                                ),
                                optimizer_update_raw,
                                i2,
                                grad_lr,
                                grad_reg_strategy,
                                grad_reg_lambda,
                                grad_gate_floor,
                                grad_gate_sharpness,
                                grad_gate_sine_amp,
                            )
                            if optimizer_update.numel() > 0:
                                if use_hessian_group_shard:
                                    state["W_sub"][hessian_group_ids, :, i2:] -= optimizer_update
                                else:
                                    state["W_sub"][:, :, i2:] -= optimizer_update
                                if _rp_on:
                                    # W_sub -= update, so the displacement is
                                    # -update; the sign matters for the net
                                    # figure, which is what R is built from.
                                    _u = optimizer_update.detach().float()
                                    if use_hessian_group_shard:
                                        _rp_disp[hessian_group_ids, :, i2:] -= _u
                                        _rp_path[hessian_group_ids, :, i2:] += _u.abs()
                                    else:
                                        _rp_disp[:, :, i2:] -= _u
                                        _rp_path[:, :, i2:] += _u.abs()

                            if refresh_full_metrics:
                                if optimizer_update_raw.numel() > 0:
                                    raw_2d = optimizer_update_raw.reshape(-1, optimizer_update_raw.shape[-1]).float()
                                    first_order_raw_abs_mean = raw_2d.abs().mean().item()
                                    first_order_raw_mean_row_l2 = torch.linalg.norm(raw_2d, dim=1).mean().item()
                                if optimizer_update.numel() > 0:
                                    upd_2d = optimizer_update.reshape(-1, optimizer_update.shape[-1]).float()
                                    upd_abs = upd_2d.abs()
                                    first_order_abs_mean = upd_abs.mean().item()
                                    first_order_mean_row_l2 = torch.linalg.norm(upd_2d, dim=1).mean().item()
                                    first_order_abs_max = upd_abs.max().item()
                                    first_order_abs_q99 = _quantile_large(upd_abs, 0.99)
                                if gate_regularizer_update.numel() > 0:
                                    reg_2d = gate_regularizer_update.reshape(-1, gate_regularizer_update.shape[-1]).float()
                                    regularizer_abs_mean = reg_2d.abs().mean().item()
                                    regularizer_mean_row_l2 = torch.linalg.norm(reg_2d, dim=1).mean().item()
                                if sine_regularizer_update.numel() > 0:
                                    sine_2d = sine_regularizer_update.reshape(-1, sine_regularizer_update.shape[-1]).float()
                                    sine_regularizer_abs_mean = sine_2d.abs().mean().item()
                                    sine_regularizer_mean_row_l2 = torch.linalg.norm(sine_2d, dim=1).mean().item()

                            if block_observer is not None:
                                block_observer(
                                    {
                                        "block_idx": i1 // blocksize,
                                        "col_start": i1,
                                        "col_end": i2,
                                        "remaining_columns": C - i2,
                                        "remaining_grad_abs_mean": trailing_grad_abs_mean,
                                        "remaining_grad_clipped_abs_mean": trailing_grad_clipped_abs_mean,
                                        "remaining_grad_mean_row_l2": trailing_grad_mean_row_l2,
                                        "second_order_update_abs_mean": second_order_abs_mean,
                                        "second_order_update_mean_row_l2": second_order_mean_row_l2,
                                        "second_order_update_abs_max": second_order_abs_max,
                                        "second_order_update_abs_q99": second_order_abs_q99,
                                        "first_order_raw_abs_mean": first_order_raw_abs_mean,
                                        "first_order_raw_mean_row_l2": first_order_raw_mean_row_l2,
                                        "first_order_update_abs_mean": first_order_abs_mean,
                                        "first_order_update_mean_row_l2": first_order_mean_row_l2,
                                        "first_order_update_abs_max": first_order_abs_max,
                                        "first_order_update_abs_q99": first_order_abs_q99,
                                        "regularizer_update_abs_mean": regularizer_abs_mean,
                                        "regularizer_update_mean_row_l2": regularizer_mean_row_l2,
                                        "sine_regularizer_update_abs_mean": sine_regularizer_abs_mean,
                                        "sine_regularizer_update_mean_row_l2": sine_regularizer_mean_row_l2,
                                        "mean_refresh_loss": None if refresh_meta is None else refresh_meta.get("mean_refresh_loss"),
                                        "train_mean_refresh_loss": None if refresh_meta is None else refresh_meta.get("train_mean_refresh_loss"),
                                        "val_mean_refresh_loss": None if refresh_meta is None else refresh_meta.get("val_mean_refresh_loss"),
                                        "refresh_subset_mean_refresh_loss": None if refresh_meta is None else refresh_meta.get("refresh_subset_mean_refresh_loss"),
                                        "slide_alpha": None if refresh_meta is None else refresh_meta.get("slide_alpha"),
                                        "mean_refresh_loss_current": None if refresh_meta is None else refresh_meta.get("mean_refresh_loss_current"),
                                        "mean_refresh_loss_next": None if refresh_meta is None else refresh_meta.get("mean_refresh_loss_next"),
                                        "sample_indices": () if refresh_meta is None else refresh_meta.get("sample_indices", ()),
                                    }
                                )

            if _rp_on:
                _reach_probe_emit(
                    getattr(self, "layer_idx", "?"),
                    getattr(self, "layer_name", "?"),
                    grad_lr, _rp_rows, int(hessian_group_ids.numel()),
                )

            with profile_recorder.section("fasterquant_group_parallel.finalize") if profile_recorder else _NULL_CONTEXT:
                Q_final = natural_order(state["Q"])
                W_int_final = natural_order(state["W_int_sub"])
                Scale_final = natural_order(state["Scale_sub"])
                if export_to_et:
                    self.layer.register_buffer(
                        "int_weight", W_int_final.reshape(self.layer.weight.shape)
                    )
                    self.layer.register_buffer("scale", Scale_final)
                self.layer.weight.data = Q_final.reshape(self.layer.weight.shape).to(
                    self.layer.weight.data.dtype
                )
                if torch.any(torch.isnan(self.layer.weight.data)):
                    logging.warning("NaN in weights")
                    raise ValueError("NaN in weights")

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor, out):
        """
        inp: shape [batch_size, seq_len, in_features]

        Accumulates an unnormalised per-rank sum into `self.H` / `self.act_square`.
        `finalize_hessian()` later all-reduces across ranks and applies the single
        global division.

        This replaces the pre-DP running-mean update. Single-GPU no longer stays
        bit-exact with the previous baseline because FP accumulation order changes,
        but the mathematical result is the same and verification uses tolerant
        allclose anyway (DP vs serial is never bit-exact).
        """
        if self._finalized:
            raise RuntimeError("add_batch called after finalize_hessian. Re-init GPTQPlus to reuse.")
        profile_recorder = self.profile_recorder
        with profile_recorder.section("add_batch.total") if profile_recorder else _NULL_CONTEXT:
            if inp.dim() == 2:
                inp = inp.unsqueeze(0)  # => [1, seq_len, dim]
            else:
                assert inp.dim() == 3, "Input must be 2D or 3D. Got %dD." % inp.dim()

            with profile_recorder.section("add_batch.slice_saliency") if profile_recorder else _NULL_CONTEXT:
                bsz = inp.shape[0]
                sal_batch = self.saliencies[self.index: self.index + bsz].to(self.dev)
                self.index += bsz

            # When --ignore_attention_sink is on, the per-token saliency stored
            # on this GPTQPlus instance has its sink positions stripped (T_eff =
            # T - sink). The activation `inp` still carries the full T from the
            # forward (KV is computed normally over the sink). Drop the leading
            # `sink` rows from `inp` so the weighted-input mul below stays shape
            # aligned. This makes the hessian accumulation see only non-sink
            # tokens, which is the consistent companion to slicing the loss.
            if sal_batch.dim() == 3 and inp.shape[1] > sal_batch.shape[1]:
                _sink = inp.shape[1] - sal_batch.shape[1]
                inp = inp[:, _sink:]

            with profile_recorder.section("add_batch.prepare_inputs") if profile_recorder else _NULL_CONTEXT:
                if inp.dim() == 3:
                    inp = inp.reshape(-1, inp.shape[-1])
                    sal_batch = sal_batch.reshape(-1, sal_batch.shape[-1])
                inp = inp.float()
                sal_batch = sal_batch.float()
                n_tokens = inp.shape[0]

            if self.hessian_group_sharded:
                with profile_recorder.section("add_batch.hessian_reduce_scatter") if profile_recorder else _NULL_CONTEXT:
                    inp_T = inp.transpose(0, 1).contiguous()
                    for group_id in range(self.num_groups):
                        weighted = inp.mul(sal_batch[:, group_id].unsqueeze(1))
                        block = inp_T.matmul(weighted)
                        self._reduce_scatter_hessian_group_(group_id, block)
                        local_pos = int(self.hessian_group_to_pos[group_id].item())
                        if local_pos >= 0:
                            self.H[local_pos].add_(block)
            else:
                with profile_recorder.section("add_batch.weighted_input") if profile_recorder else _NULL_CONTEXT:
                    weighted = inp.unsqueeze(0).mul(sal_batch.transpose(0, 1).unsqueeze(-1))

                with profile_recorder.section("add_batch.hessian_block") if profile_recorder else _NULL_CONTEXT:
                    inp_T_batched = inp.transpose(0, 1).unsqueeze(0).expand(self.H.shape[0], -1, -1)
                    block = torch.bmm(inp_T_batched, weighted)

                with profile_recorder.section("add_batch.accumulate") if profile_recorder else _NULL_CONTEXT:
                    # Pure sum; normalisation deferred to `finalize_hessian`. Tracking
                    # `token_count` lets finalize recover seq_len = token_count / index.
                    self.H.add_(block)
            with profile_recorder.section("add_batch.accumulate_act_square") if profile_recorder else _NULL_CONTEXT:
                self.act_square.add_((inp ** 2).sum(0))
                self.token_count += n_tokens

    @torch.no_grad()
    def finalize_hessian(self):
        """All-reduce the unnormalised sums across ranks (no-op at world_size=1),
        then apply the single global division to produce the same H as the
        pre-DP running-mean formula did:

            H      = sum_tokens block / (total_samples * seq_len)
            act_sq = sum_tokens (inp**2)_n / seq_len

        Called exactly once per GPTQPlus instance, between the forward that
        drives `add_batch` and `fasterquant`.
        """
        if self._finalized:
            return
        from utils import dist_utils as _dist  # local import to avoid cycles

        _dist.allreduce_sum_(self.act_square)
        total_samples = _dist.allreduce_sum_scalar(self.index)
        total_tokens = _dist.allreduce_sum_scalar(self.token_count)
        if total_samples <= 0 or total_tokens <= 0:
            raise RuntimeError("finalize_hessian called before any add_batch ran.")
        seq_len = total_tokens / total_samples
        if self.hessian_group_sharded:
            # The sharded path already reduce-scatters each global group inside
            # add_batch(), so `self.H` contains the global unnormalised sum for
            # only this rank's owner groups. Do not rebuild a full GxCxC tensor.
            pass
        else:
            _dist.allreduce_sum_(self.H)
        self.H.div_(total_samples * seq_len)
        if self.hessian_saliency_scale != 1.0:
            self.H.div_(self.hessian_saliency_scale)
        self.act_square.div_(seq_len)
        # Symmetrise H to wipe out residual asymmetry from tensor-core bmm
        # round-off (tf32 / bf16 outputs can lose exact H[i,j] == H[j,i]).
        # Without this the per-subgroup Cholesky in fasterquant silently trips
        # on deep / ill-conditioned layers (e.g. Qwen3-4B layer 24 attention)
        # and `damp_percent` has to auto-increment. Mathematically a no-op on
        # an exactly-symmetric H.
        self.H = 0.5 * (self.H + self.H.transpose(-1, -2))
        # NaN/Inf sanitisation. Once NaN leaks into H it's terminal — no damp
        # can recover Cholesky and the retry loop runs until damp exceeds 1
        # and crashes. Replace with zeros so the `dead channel` check in
        # fasterquant handles the affected columns. Logs loudly so we can
        # spot an upstream corruption (saliency/fisher produced by an NLL
        # backward with log(0) / overflow, etc).
        if not torch.isfinite(self.H).all():
            n_nan = int(torch.isnan(self.H).sum().item())
            n_inf = int(torch.isinf(self.H).sum().item())
            logging.warning(
                "finalize_hessian: non-finite in H (nan=%d, inf=%d, total=%d, module=%s); "
                "sanitising to 0 — fasterquant's dead-channel check will take over.",
                n_nan, n_inf, self.H.numel(), self.layer.__class__.__name__,
            )
            self.H = torch.nan_to_num(self.H, nan=0.0, posinf=0.0, neginf=0.0)
        self._finalized = True

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        enable_gradient_update=True,
        g_update_mode="frozen",
        export_to_et=False,
        profile_recorder=None,
        gradient_refresh_fn=None,
        grad_lr=1e-3,
        grad_optimizer="sgd",
        grad_reg_strategy="none",
        grad_reg_lambda=0.0,
        grad_gate_floor=0.1,
        grad_gate_sharpness=1.0,
        grad_gate_sine_amp=0.0,
        second_order_scale=1.0,
        block_atomic_quant=False,
        block_observer=None,
        grad_clip=1.0,
        horizon_p=0.0,
        diagnostic_recorder=None,
        slide_refresh_start=0,
        slide_refresh_block_total=None,
        refresh_full_metrics=False,
        group_parallel_mode="none",
        grad_sq_full=None,
        warm_start_steps=0,
    ):
        profile_recorder = profile_recorder or self.profile_recorder
        # Alias the recorder so call sites can do `rec and rec.save_block(...)`.
        rec = diagnostic_recorder
        self._validate_grad_regularizer(
            grad_reg_strategy,
            grad_reg_lambda=grad_reg_lambda,
            grad_gate_floor=grad_gate_floor,
            grad_gate_sharpness=grad_gate_sharpness,
            grad_gate_sine_amp=grad_gate_sine_amp,
        )
        group_parallel_mode = (group_parallel_mode or "none").lower()
        # The probe lives in _fasterquant_group_parallel only. Running it under
        # group_parallel_mode=none would produce a clean log with no reach_probe
        # lines in it, which reads exactly like "the reach is zero" -- refuse
        # instead. This file has already produced four silent no-ops of this
        # shape (see the sweep-script overrides in the project notes).
        if horizon_p and group_parallel_mode == "none":
            raise ValueError(
                "`horizon_p` is only instrumented in the group-parallel path. "
                "Set GROUP_PARALLEL_QUANT=rank (the sweep default) or leave "
                "--horizon_p at 0."
            )
        if os.environ.get("REACH_PROBE") == "1" and group_parallel_mode == "none":
            raise ValueError(
                "REACH_PROBE=1 requires group_parallel_mode != none; the probe "
                "is only instrumented in the group-parallel path. Set "
                "GROUP_PARALLEL_QUANT=rank (the sweep default) or unset "
                "REACH_PROBE."
            )
        if group_parallel_mode not in {"none", "tensor", "rank"}:
            raise ValueError(
                f"Unsupported `group_parallel_mode={group_parallel_mode}`. "
                "Expected one of: none, tensor, rank."
            )
        if group_parallel_mode != "none":
            dynamic_groups = groupsize != -1 and not static_groups
            fallback_reason = None
            if rec is not None:
                fallback_reason = "diagnostic recorder is active"
            elif dynamic_groups:
                fallback_reason = "dynamic weight groups are not supported"
            elif not getattr(self.quantizer, "sym", True):
                fallback_reason = "asymmetric weight quantization is not supported"
            elif self.quantizer.bits >= 16:
                fallback_reason = "weight quantization is disabled"
            elif grad_reg_strategy in {"quant_error_gate", "quant_error_gate_optimized"}:
                fallback_reason = "quantization-error gate regularizers are not supported"
            if fallback_reason is None:
                return self._fasterquant_group_parallel(
                    grad_sq_full=grad_sq_full,
                    warm_start_steps=warm_start_steps,
                    blocksize=blocksize,
                    percdamp=percdamp,
                    groupsize=groupsize,
                    actorder=actorder,
                    static_groups=static_groups,
                    enable_gradient_update=enable_gradient_update,
                    g_update_mode=g_update_mode,
                    export_to_et=export_to_et,
                    profile_recorder=profile_recorder,
                    gradient_refresh_fn=gradient_refresh_fn,
                    grad_lr=grad_lr,
                    grad_optimizer=grad_optimizer,
                    grad_reg_strategy=grad_reg_strategy,
                    grad_reg_lambda=grad_reg_lambda,
                    grad_gate_floor=grad_gate_floor,
                    grad_gate_sharpness=grad_gate_sharpness,
                    grad_gate_sine_amp=grad_gate_sine_amp,
                    second_order_scale=second_order_scale,
                    block_atomic_quant=block_atomic_quant,
                    block_observer=block_observer,
                    grad_clip=grad_clip,
                    horizon_p=horizon_p,
                    slide_refresh_start=slide_refresh_start,
                    slide_refresh_block_total=slide_refresh_block_total,
                    refresh_full_metrics=refresh_full_metrics,
                    group_parallel_mode=group_parallel_mode,
                )
            if self.hessian_group_sharded:
                raise RuntimeError(
                    f"group_parallel_mode={group_parallel_mode} requested and Hessian "
                    f"is group-sharded, but group-parallel fasterquant would fall back: "
                    f"{fallback_reason}. Disable group_parallel_quant or the fallback trigger."
                )
            logging.warning(
                "group_parallel_mode=%s requested but falling back to legacy fasterquant: %s",
                group_parallel_mode,
                fallback_reason,
            )
        with profile_recorder.section("fasterquant.total") if profile_recorder else _NULL_CONTEXT:
            W = self.layer.weight.data.clone()
            W = W.float()
            block_gd_mode = g_update_mode == "block_gd"
            # Diagnostic: dump the pre-quant (rotated, if applicable) module weight,
            # plus the quantizer params, gradients and saliency that feed this
            # fasterquant invocation. Everything under module_level/ is fixed for
            # the duration of fasterquant; per-block stuff lands under subgroup_X/block_Y/.
            if rec is not None:
                rec.save_module_level("original_weight_rotated", W)
                if self.quantizer.scale is not None:
                    rec.save_module_level("quantizer_scale", self.quantizer.scale)
                if hasattr(self.quantizer, "zero") and self.quantizer.zero is not None:
                    rec.save_module_level("quantizer_zero", self.quantizer.zero)
                if isinstance(self.quantizer.maxq, torch.Tensor):
                    rec.save_module_level("quantizer_maxq", self.quantizer.maxq)
                else:
                    rec.save_module_level("quantizer_maxq", torch.tensor(float(self.quantizer.maxq)))
                rec.save_module_level("H_after_accumulate", self.H)
                rec.save_module_level("act_square", self.act_square)
                rec.save_module_level("gradients", self.gradients)
                rec.save_module_level("reference_loss", torch.tensor(self.reference_loss))
                rec.save_module_level("saliencies", self.saliencies)
                rec.save_module_level("alpha", torch.tensor(self.alpha))
            with profile_recorder.section("fasterquant.allocate_outputs") if profile_recorder else _NULL_CONTEXT:
                Q_final = torch.zeros_like(W)
                W_int_final = torch.zeros_like(W)
                Scale_final = torch.zeros_like(W)

            if not self.quantizer.ready():
                with profile_recorder.section("fasterquant.quantizer_find_params_initial") if profile_recorder else _NULL_CONTEXT:
                    self.quantizer.find_params(W)

            dynamic_groups = groupsize != -1 and not static_groups
            if dynamic_groups and groupsize != blocksize:
                raise ValueError(
                    "`groupsize` must equal `blocksize` when using dynamic groups in GPTQ+. "
                    f"Got groupsize={groupsize}, blocksize={blocksize}."
                )
            if dynamic_groups and grad_reg_strategy in {"quant_error_gate", "quant_error_gate_optimized"}:
                raise ValueError(
                    "Dynamic weight groups are not supported with quant_error_gate regularization yet. "
                    "Use static groups or disable the quantization-error gate."
                )

            shared_groups = None
            if groupsize != -1 and static_groups:
                with profile_recorder.section("fasterquant.quantizer_build_groups") if profile_recorder else _NULL_CONTEXT:
                    shared_groups = []
                    for col_start in range(0, self.columns, groupsize):
                        col_end = min(col_start + groupsize, self.columns)
                        quantizer = copy.deepcopy(self.quantizer)
                        quantizer.find_params(W[:, col_start:col_end])
                        shared_groups.append(quantizer)

            # Fast path for per-row symmetric quantization with groupsize == -1.
            # Hoists scale materialisation and `quantizer.ready()` (a GPU reduction)
            # out of the per-column inner loop. When unavailable we fall back to
            # calling `quantizer.fake_quantize(...)` per column as before.
            fast_quant_enabled = (
                groupsize == -1
                and self.quantizer.bits < 16
                and self.quantizer.ready()
            )
            if fast_quant_enabled:
                fast_quant_scale_full = self.quantizer.scale.to(self.dev)
                fast_quant_maxq = int(self.quantizer.maxq.item())
                fast_quant_lo = -(fast_quant_maxq + 1)
            else:
                fast_quant_scale_full = None
                fast_quant_maxq = None
                fast_quant_lo = None

            rows_per_sub = self.rows // self.num_groups
            subgroup_states = []
            need_hessian_reg = grad_reg_strategy == "hessian"
            need_full_precision_weight = grad_reg_strategy in {"l2", "hessian"}
            need_gate_quant_maps = grad_reg_strategy in {"quant_error_gate", "quant_error_gate_optimized"}
            for sub_idx in range(self.num_groups):
                with profile_recorder.section("fasterquant.subgroup.total") if profile_recorder else _NULL_CONTEXT:
                    row_start = sub_idx * rows_per_sub
                    row_end = (sub_idx + 1) * rows_per_sub

                    with profile_recorder.section("fasterquant.subgroup.slice_inputs") if profile_recorder else _NULL_CONTEXT:
                        W_sub = W[row_start:row_end, :].clone()
                        H_sub = self.H[sub_idx].clone()
                        gradients_sub = self.gradients[row_start: row_end, :].to(self.dev).float().clone()
                        dead = torch.diag(H_sub) == 0
                        H_sub[dead, dead] = 1
                        W_sub[:, dead] = 0

                    groups = shared_groups

                    perm = None
                    invperm = None
                    if actorder:
                        with profile_recorder.section("fasterquant.subgroup.actorder_permute") if profile_recorder else _NULL_CONTEXT:
                            perm = torch.argsort(self.act_square, descending=True)
                            W_sub = W_sub[:, perm]
                            H_sub = H_sub[perm][:, perm]
                            gradients_sub = gradients_sub[:, perm]
                            invperm = torch.argsort(perm)

                    if need_hessian_reg:
                        with profile_recorder.section("fasterquant.subgroup.H_sub_clone") if profile_recorder else _NULL_CONTEXT:
                            hessian_reg = H_sub.clone()
                    else:
                        hessian_reg = None
                    if need_gate_quant_maps:
                        with profile_recorder.section("fasterquant.subgroup.build_gate_quant_maps") if profile_recorder else _NULL_CONTEXT:
                            gate_scale, gate_zero = self._build_gate_quant_maps(
                                self.quantizer,
                                W_sub,
                                groupsize,
                                row_start=row_start,
                                row_end=row_end,
                                groups=groups,
                                perm=perm,
                            )
                    else:
                        gate_scale = None
                        gate_zero = None

                    with profile_recorder.section("fasterquant.subgroup.allocate_buffers") if profile_recorder else _NULL_CONTEXT:
                        Q = torch.zeros_like(W_sub)
                        W_int_sub = torch.zeros_like(W_sub)
                        Scale_sub = torch.zeros_like(W_sub)

                    Hinv_init, Hinv, damp_percent, hessian_identity_fallback = (
                        self._compute_hessian_inverse_with_fallback(
                            H_sub,
                            percdamp=percdamp,
                            profile_recorder=profile_recorder,
                            profile_section="fasterquant.subgroup.compute_hinv",
                            log_context=f" subgroup={sub_idx}",
                        )
                    )

                    with profile_recorder.section("fasterquant.subgroup.init_ghinv") if profile_recorder else _NULL_CONTEXT:
                        beta, beta_view, Z, GHinv = self._compute_gradient_terms(
                            gradients_sub,
                            Hinv_init,
                            Hinv,
                            enable_gradient_update,
                        )

                    # Diagnostic: per-subgroup snapshot of the Hessian inverse,
                    # the permuted weight and gradient (post-actorder), and the
                    # GPTQ+ state. Everything here is fixed for the duration of
                    # the block loop for this subgroup.
                    if rec is not None:
                        rec.save_subgroup(sub_idx, "W_sub_initial", W_sub)
                        rec.save_subgroup(sub_idx, "gradients_sub", gradients_sub)
                        rec.save_subgroup(sub_idx, "H_sub_damped_but_original", None if hessian_reg is None else hessian_reg)
                        rec.save_subgroup(sub_idx, "Hinv_init", Hinv_init)
                        rec.save_subgroup(sub_idx, "Hinv_upper_cholesky", Hinv)
                        rec.save_subgroup(sub_idx, "hessian_identity_fallback", torch.tensor(hessian_identity_fallback))
                        rec.save_subgroup(sub_idx, "beta", beta)
                        rec.save_subgroup(sub_idx, "Z_initial", Z)
                        rec.save_subgroup(sub_idx, "GHinv_initial", GHinv)
                        rec.save_subgroup(sub_idx, "row_slice", torch.tensor([row_start, row_end]))
                        if perm is not None:
                            rec.save_subgroup(sub_idx, "perm", perm)
                            rec.save_subgroup(sub_idx, "invperm", invperm)

                    subgroup_states.append(
                        {
                            "sub_idx": sub_idx,
                            "row_start": row_start,
                            "row_end": row_end,
                            "W_sub": W_sub,
                            "gradients_sub": gradients_sub,
                            "Hinv_init": Hinv_init,
                            "Hinv": Hinv,
                            "Q": Q,
                            "W_int_sub": W_int_sub,
                            "Scale_sub": Scale_sub,
                            "beta": beta,
                            "beta_view": beta_view,
                            "Z": Z,
                            "GHinv": GHinv,
                            "anchor_weight": W_sub.clone(),
                            "full_precision_weight": W_sub.clone() if need_full_precision_weight else None,
                            "hessian_reg": hessian_reg,
                            "gate_scale": gate_scale,
                            "gate_zero": gate_zero,
                            "grad_optimizer_state": self._make_grad_optimizer_state(
                                W_sub,
                                grad_optimizer,
                                curvature_full=_align_warm_prior(
                                    grad_sq_full, row_start, row_end, perm, W_sub.device,
                                ),
                                warm_start_steps=warm_start_steps,
                            ) if block_gd_mode else None,
                            "groups": groups,
                            "perm": perm,
                            "invperm": invperm,
                            # Cached per-subgroup scale slice for the fast quant
                            # path; None when the fast path is disabled.
                            "fast_quant_scale": (
                                fast_quant_scale_full[row_start:row_end]
                                if fast_quant_enabled else None
                            ),
                        }
                    )

            # Number of blocks + refreshes, for the loss-slide-window schedule.
            # Refresh fires after every block except the last, so n_refresh = n_blocks - 1.
            n_blocks_total = (self.columns + blocksize - 1) // blocksize
            n_refresh_total = max(n_blocks_total - 1, 0)

            for i1 in range(0, self.columns, blocksize):
                with profile_recorder.section("fasterquant.block.total") if profile_recorder else _NULL_CONTEXT:
                    i2 = min(i1 + blocksize, self.columns)
                    count = i2 - i1
                    is_last_block = i2 >= self.columns
                    use_atomic_quant = block_atomic_quant and not is_last_block
                    D = torch.arange(count - 1, -1, -1).to(W)
                    block_states = []

                    for state in subgroup_states:
                        with profile_recorder.section("fasterquant.block.setup") if profile_recorder else _NULL_CONTEXT:
                            W1 = state["W_sub"][:, i1:i2].clone()
                            W_ref1 = state["anchor_weight"][:, i1:i2]
                            W_block_start = W1.clone()
                            Q1 = torch.zeros_like(W1)
                            W_int1 = torch.zeros_like(W1)
                            Scale1 = torch.zeros_like(W1).to(state["Scale_sub"].dtype)
                            Err1 = torch.zeros_like(W1)
                            Hinv1 = state["Hinv"][i1:i2, i1:i2]
                            GHinv1 = state["GHinv"][:, i1:i2].clone()
                            Z1 = state["Z"][:, i1:i2]
                            inner_update_mode = "surrogate_online" if g_update_mode == "block_backward" else "frozen" if block_gd_mode else g_update_mode
                            is_frozen_inner = inner_update_mode == "frozen"
                            is_surrogate_online = inner_update_mode == "surrogate_online"
                            GHinv1_eff = self._current_ghinv(
                                GHinv1,
                                W1 if is_surrogate_online else W_block_start,
                                W_ref1,
                                state["beta_view"],
                                inner_update_mode,
                            )
                            fast_quant_scale = state["fast_quant_scale"] if fast_quant_enabled else None
                            dynamic_quantizer = None
                            if dynamic_groups:
                                # Dynamic groups estimate the row-wise scale at the
                                # last possible moment before this group/block is
                                # quantized, using the current working weight after
                                # all previous GPTQ/block-GD updates.
                                with profile_recorder.section("fasterquant.block.dynamic_group_find_params") if profile_recorder else _NULL_CONTEXT:
                                    dynamic_quantizer = copy.deepcopy(self.quantizer)
                                    dynamic_quantizer.find_params(W_block_start)
                            # Diagnostic: snapshot state at the start of this block
                            # BEFORE any inner quant updates. Captures the input to
                            # the block loop: W1_start is what the inner loop will
                            # quantize; W_trailing_start is what outer + adam will
                            # later push. These reflect any previous blocks' effects.
                            if rec is not None:
                                block_idx = i1 // blocksize
                                sub_idx = state["sub_idx"]
                                rec.save_block(sub_idx, block_idx, "W1_start", W1)
                                rec.save_block(
                                    sub_idx, block_idx, "W_trailing_start",
                                    state["W_sub"][:, i2:].clone(),
                                )
                                rec.save_block(sub_idx, block_idx, "Hinv1", Hinv1)
                                rec.save_block(sub_idx, block_idx, "GHinv1_at_block_start", GHinv1)
                                rec.save_block(sub_idx, block_idx, "Z1", Z1)
                                rec.save_block(
                                    sub_idx, block_idx, "block_column_range",
                                    torch.tensor([i1, i2]),
                                )
                            # With groupsize == -1 the per-row scale is shared by
                            # every column of the block, so we can fill Scale1
                            # once here instead of reassigning it in each column
                            # iteration. Same dtype and values as the per-column
                            # assignment, so bit-exactness is preserved.
                            if fast_quant_scale is not None:
                                Scale1.copy_(fast_quant_scale.expand_as(Scale1))

                        if use_atomic_quant:
                            if groupsize == -1:
                                # In per-row quantization each column uses the same row-wise scale,
                                # so atomic block quantization can quantize the whole block at once
                                # without changing the final quantized result.
                                with profile_recorder.section("fasterquant.block.atomic_quantize_full") if profile_recorder else _NULL_CONTEXT:
                                    q, int_weight, scale = self.quantizer.fake_quantize(
                                        W_block_start,
                                        st_idx=state["row_start"],
                                        end_idx=state["row_end"],
                                    )
                                    Q1.copy_(q)
                                    W_int1.copy_(int_weight)
                                    Scale1.copy_(scale.expand_as(W_block_start))
                            else:
                                for i in range(count):
                                    with profile_recorder.section("fasterquant.column.total") if profile_recorder else _NULL_CONTEXT:
                                        w = W_block_start[:, i]

                                        quantizer = self.quantizer
                                        quant_st_idx = state["row_start"]
                                        quant_end_idx = state["row_end"]
                                        if dynamic_quantizer is not None:
                                            quantizer = dynamic_quantizer
                                            quant_st_idx = None
                                            quant_end_idx = None
                                        elif groupsize != -1:
                                            idx = i1 + i
                                            if actorder:
                                                idx = state["perm"][idx]
                                            quantizer = state["groups"][idx // groupsize]

                                        with profile_recorder.section("fasterquant.column.quantize") if profile_recorder else _NULL_CONTEXT:
                                            q, int_weight, scale = quantizer.fake_quantize(
                                                w.unsqueeze(1),
                                                st_idx=quant_st_idx,
                                                end_idx=quant_end_idx,
                                            )
                                        Q1[:, i] = q.flatten()
                                        W_int1[:, i] = int_weight.flatten()
                                        Scale1[:, i] = scale.flatten()

                            with profile_recorder.section("fasterquant.block.atomic_err_solve") if profile_recorder else _NULL_CONTEXT:
                                residual_block = W_block_start - Q1 - GHinv1_eff
                                Err1.copy_(
                                    torch.linalg.solve_triangular(
                                        Hinv1.T,
                                        residual_block.T,
                                        upper=False,
                                    ).T
                                )
                        else:
                            # Per-column `with profile_recorder.section(...)` wrappers were
                            # removed from the inner loop — evaluating the ternary plus
                            # entering/exiting a `nullcontext` 4x per column adds a few ms
                            # per block at zero benefit when profiling is disabled (the
                            # block-level `fasterquant.block.total` range already bounds
                            # the inner loop for Nsight).
                            for i in range(count):
                                w = W1[:, i]
                                d = Hinv1[i, i]

                                w_col = w.unsqueeze(1)
                                if fast_quant_scale is not None:
                                    # Inline of WeightQuantizer.fake_quantize for the
                                    # symmetric per-row, groupsize == -1 case. Same
                                    # operations and dtypes as the original call, so
                                    # the output is bit-identical.
                                    int_weight = torch.clamp(
                                        torch.round(w_col / fast_quant_scale),
                                        fast_quant_lo,
                                        fast_quant_maxq,
                                    )
                                    q_fake = (fast_quant_scale * int_weight).to(w_col.dtype)
                                    scale = fast_quant_scale
                                else:
                                    quantizer = self.quantizer
                                    quant_st_idx = state["row_start"]
                                    quant_end_idx = state["row_end"]
                                    if dynamic_quantizer is not None:
                                        quantizer = dynamic_quantizer
                                        quant_st_idx = None
                                        quant_end_idx = None
                                    elif groupsize != -1:
                                        idx = i1 + i
                                        if actorder:
                                            idx = state["perm"][idx]
                                        quantizer = state["groups"][idx // groupsize]
                                    q_fake, int_weight, scale = quantizer.fake_quantize(
                                        w_col,
                                        st_idx=quant_st_idx,
                                        end_idx=quant_end_idx,
                                    )
                                q_flat = q_fake.flatten()
                                Q1[:, i] = q_flat
                                q = q_flat
                                W_int1[:, i] = int_weight.flatten()
                                if fast_quant_scale is None:
                                    # In the groupsize != -1 path scale varies per
                                    # column group, so we still need the per-column
                                    # write. With fast_quant_scale active the block
                                    # setup pre-filled Scale1.
                                    Scale1[:, i] = scale.flatten()

                                err1 = (w - q - GHinv1_eff[:, i]) / d
                                second_order_inner_update = err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                                if block_gd_mode:
                                    W1[:, i:] -= second_order_scale * (
                                        second_order_inner_update + GHinv1_eff[:, i:]
                                    )
                                else:
                                    W1[:, i:] -= second_order_inner_update + GHinv1_eff[:, i:]
                                Err1[:, i] = err1

                                # In-place subtract avoids one tensor allocation
                                # per column compared to `GHinv1[:, i:] = GHinv1[:, i:] - ...`.
                                GHinv1[:, i:].sub_(
                                    Z1[:, i].unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                                )
                                # For frozen inner mode `_current_ghinv` just returns
                                # `GHinv1`, so `GHinv1_eff` aliases it already and
                                # sees the in-place mutation automatically.
                                if not is_frozen_inner:
                                    GHinv1_eff = self._current_ghinv(
                                        GHinv1,
                                        W1 if is_surrogate_online else W_block_start,
                                        W_ref1,
                                        state["beta_view"],
                                        inner_update_mode,
                                    )

                        with profile_recorder.section("fasterquant.block.writeback_inner") if profile_recorder else _NULL_CONTEXT:
                            state["Q"][:, i1:i2] = Q1
                            state["W_int_sub"][:, i1:i2] = W_int1
                            state["Scale_sub"][:, i1:i2] = Scale1

                        # Diagnostic: block inner-loop outputs. W1 here has
                        # received the inner error propagation for columns
                        # [i, count), but W_sub trailing hasn't been touched yet.
                        if rec is not None:
                            block_idx = i1 // blocksize
                            sub_idx = state["sub_idx"]
                            rec.save_block(sub_idx, block_idx, "Q1", Q1)
                            rec.save_block(sub_idx, block_idx, "W_int1", W_int1)
                            rec.save_block(sub_idx, block_idx, "Scale1", Scale1)
                            rec.save_block(sub_idx, block_idx, "Err1", Err1)
                            rec.save_block(sub_idx, block_idx, "W1_after_inner", W1)
                            rec.save_block(
                                sub_idx, block_idx, "GHinv1_after_inner", GHinv1,
                            )

                        # `current_sub_weight` is only consumed by the block_backward
                        # refresh path below; skipping the clone for other modes
                        # avoids O(rows_per_sub × columns) copies per block.
                        if g_update_mode == "block_backward":
                            current_sub_weight = state["W_sub"].clone()
                            if i1 > 0:
                                current_sub_weight[:, :i1] = state["Q"][:, :i1]
                            current_sub_weight[:, i1:i2] = Q1
                        else:
                            current_sub_weight = None
                        block_states.append(
                            {
                                "state": state,
                                "current_sub_weight": current_sub_weight,
                                "Err1": Err1,
                                "Z1": Z1,
                            }
                        )

                    if g_update_mode == "block_backward" and enable_gradient_update:
                        if gradient_refresh_fn is None:
                            raise ValueError("`gradient_refresh_fn` must be provided for g_update_mode='block_backward'.")
                        with profile_recorder.section("fasterquant.block.true_gradient_refresh") if profile_recorder else _NULL_CONTEXT:
                            weight_snapshot = self.layer.weight.data.clone().float()
                            for block_state in block_states:
                                state = block_state["state"]
                                current_sub_weight = block_state["current_sub_weight"]
                                current_sub_weight_orig = (
                                    current_sub_weight[:, state["invperm"]]
                                    if actorder else current_sub_weight
                                )
                                weight_snapshot[state["row_start"]:state["row_end"], :] = current_sub_weight_orig

                            refreshed_grad, refresh_meta = gradient_refresh_fn(weight_snapshot)
                            refreshed_grad = refreshed_grad.to(self.dev).float()
                            # See the block_gd branch for the rationale behind
                            # gating the trailing_grad stats on refresh_full_metrics.
                            trailing_grad_chunks = [] if refresh_full_metrics else None
                            for block_state in block_states:
                                state = block_state["state"]
                                refreshed_grad_sub = refreshed_grad[state["row_start"]:state["row_end"], :]
                                if actorder:
                                    refreshed_grad_sub = refreshed_grad_sub[:, state["perm"]]
                                if refresh_full_metrics:
                                    trailing_grad = refreshed_grad_sub[:, i2:]
                                    if trailing_grad.numel() > 0:
                                        trailing_grad_chunks.append(trailing_grad)
                                state["gradients_sub"] = refreshed_grad_sub
                                beta, beta_view, Z, GHinv = self._compute_gradient_terms(
                                    refreshed_grad_sub,
                                    state["Hinv_init"],
                                    state["Hinv"],
                                    enable_gradient_update,
                                )
                                state["beta"] = beta
                                state["beta_view"] = beta_view
                                state["Z"] = Z
                                state["GHinv"] = GHinv
                                state["anchor_weight"] = block_state["current_sub_weight"].clone()
                                block_state["Z1"] = Z[:, i1:i2]
                            trailing_grad_abs_mean = None
                            trailing_grad_mean_row_l2 = None
                            trailing_grad_clipped_abs_mean = None
                            if refresh_full_metrics and trailing_grad_chunks:
                                trailing_grad_cat = torch.cat(trailing_grad_chunks, dim=0).float()
                                trailing_grad_abs_mean = trailing_grad_cat.abs().mean().item()
                                trailing_grad_mean_row_l2 = torch.linalg.norm(
                                    trailing_grad_cat,
                                    dim=1,
                                ).mean().item()
                                # Post-clip abs mean — for verifying `grad_clip` takes effect.
                                # Adam normalises |grad|, so a small grad_clip only shows up
                                # here, not in `first_raw_abs` / the applied step size.
                                if grad_clip is not None and grad_clip > 0:
                                    trailing_grad_clipped_abs_mean = (
                                        trailing_grad_cat.clamp(min=-grad_clip, max=grad_clip)
                                        .abs().mean().item()
                                    )
                                else:
                                    trailing_grad_clipped_abs_mean = trailing_grad_abs_mean
                            if block_observer is not None:
                                block_observer(
                                    {
                                        "block_idx": i1 // blocksize,
                                        "col_start": i1,
                                        "col_end": i2,
                                        "remaining_columns": self.columns - i2,
                                        "remaining_grad_abs_mean": trailing_grad_abs_mean,
                                        "remaining_grad_clipped_abs_mean": trailing_grad_clipped_abs_mean,
                                        "remaining_grad_mean_row_l2": trailing_grad_mean_row_l2,
                                        "second_order_update_abs_mean": None,
                                        "second_order_update_mean_row_l2": None,
                                        "second_order_update_abs_max": None,
                                        "second_order_update_abs_q99": None,
                                        "first_order_raw_abs_mean": None,
                                        "first_order_raw_mean_row_l2": None,
                                        "first_order_update_abs_mean": None,
                                        "first_order_update_mean_row_l2": None,
                                        "first_order_update_abs_max": None,
                                        "first_order_update_abs_q99": None,
                                        "regularizer_update_abs_mean": None,
                                        "regularizer_update_mean_row_l2": None,
                                        "sine_regularizer_update_abs_mean": None,
                                        "sine_regularizer_update_mean_row_l2": None,
                                        "mean_refresh_loss": None if refresh_meta is None else refresh_meta.get("mean_refresh_loss"),
                                        "train_mean_refresh_loss": None if refresh_meta is None else refresh_meta.get("train_mean_refresh_loss"),
                                        "val_mean_refresh_loss": None if refresh_meta is None else refresh_meta.get("val_mean_refresh_loss"),
                                        "refresh_subset_mean_refresh_loss": None if refresh_meta is None else refresh_meta.get("refresh_subset_mean_refresh_loss"),
                                    }
                                )

                    outer_mode = "frozen" if g_update_mode in {"frozen", "block_backward", "block_gd"} else "surrogate_block"
                    block_second_order_chunks = []
                    for block_state in block_states:
                        state = block_state["state"]
                        with profile_recorder.section("fasterquant.block.outer_update_delta_w") if profile_recorder else _NULL_CONTEXT:
                            GHinv_rest = self._current_ghinv(
                                state["GHinv"][:, i2:],
                                state["W_sub"][:, i2:],
                                state["anchor_weight"][:, i2:],
                                state["beta_view"],
                                outer_mode,
                            )
                            G_Update = count * GHinv_rest - torch.einsum(
                                "ij,j,jk->ik",
                                block_state["Z1"],
                                D,
                                state["Hinv"][i1:i2, i2:],
                            )
                            second_order_update = block_state["Err1"].matmul(state["Hinv"][i1:i2, i2:])
                            total_outer_update = second_order_update + G_Update

                            # Diagnostic: outer update components BEFORE they are
                            # subtracted from W_sub[:, i2:]. We also dump
                            # GHinv_rest and `count * GHinv_rest` separately so
                            # the cancellation between the two G_Update terms
                            # can be inspected directly.
                            if rec is not None:
                                block_idx = i1 // blocksize
                                sub_idx = state["sub_idx"]
                                einsum_term = torch.einsum(
                                    "ij,j,jk->ik",
                                    block_state["Z1"],
                                    D,
                                    state["Hinv"][i1:i2, i2:],
                                )
                                rec.save_block(sub_idx, block_idx, "GHinv_rest_at_outer", GHinv_rest)
                                rec.save_block(sub_idx, block_idx, "count_times_GHinv_rest", count * GHinv_rest)
                                rec.save_block(sub_idx, block_idx, "einsum_Z1_D_Hinv_term", einsum_term)
                                rec.save_block(sub_idx, block_idx, "G_Update", G_Update)
                                rec.save_block(sub_idx, block_idx, "second_order_update", second_order_update)
                                rec.save_block(sub_idx, block_idx, "total_outer_update", total_outer_update)
                                rec.save_block(
                                    sub_idx, block_idx, "W_trailing_before_outer_update",
                                    state["W_sub"][:, i2:].clone(),
                                )

                            if block_gd_mode:
                                applied_second_order_update = second_order_scale * second_order_update
                                state["W_sub"][:, i2:] -= second_order_scale * total_outer_update
                            else:
                                applied_second_order_update = second_order_update
                                state["W_sub"][:, i2:] -= total_outer_update
                            if refresh_full_metrics and block_gd_mode and second_order_update.numel() > 0:
                                block_second_order_chunks.append(applied_second_order_update)

                            # Diagnostic: W_sub trailing AFTER outer update.
                            if rec is not None:
                                rec.save_block(
                                    state["sub_idx"], i1 // blocksize,
                                    "W_trailing_after_outer_update",
                                    state["W_sub"][:, i2:].clone(),
                                )
                                rec.save_block(
                                    state["sub_idx"], i1 // blocksize,
                                    "applied_second_order_update", applied_second_order_update,
                                )

                        with profile_recorder.section("fasterquant.block.outer_update_ghinv") if profile_recorder else _NULL_CONTEXT:
                            state["GHinv"][:, i2:] -= state["Z"][:, i1:i2].matmul(state["Hinv"][i1:i2, i2:])
                            if block_gd_mode:
                                self._clear_grad_optimizer_state(state["grad_optimizer_state"], i1, i2)

                    if block_gd_mode and enable_gradient_update and i2 < self.columns:
                        if gradient_refresh_fn is None:
                            raise ValueError("`gradient_refresh_fn` must be provided for g_update_mode='block_gd'.")

                        with profile_recorder.section("fasterquant.block.true_gradient_refresh") if profile_recorder else _NULL_CONTEXT:
                            weight_snapshot = self.layer.weight.data.clone().float()
                            for state in subgroup_states:
                                current_sub_weight = state["W_sub"].clone()
                                current_sub_weight[:, :i2] = state["Q"][:, :i2]
                                current_sub_weight_orig = (
                                    current_sub_weight[:, state["invperm"]]
                                    if actorder else current_sub_weight
                                )
                                weight_snapshot[state["row_start"]:state["row_end"], :] = current_sub_weight_orig

                            # Loss-slide-window schedule. The schedule spans
                            # the whole transformer block when the caller
                            # passes slide_refresh_block_total (sum of refresh
                            # counts across all linear modules in the block)
                            # and slide_refresh_start (cumulative count of
                            # refreshes completed in prior modules of this
                            # block). α=1 at the very first block-level
                            # refresh, α=0 at the very last. This matches the
                            # definition of next-layer loss, which is scoped
                            # to the NEXT transformer block rather than the
                            # next linear layer. When the caller doesn't pass
                            # these (slide_refresh_block_total is None), fall
                            # back to the per-module schedule.
                            refresh_idx = i1 // blocksize
                            effective_total = (
                                slide_refresh_block_total
                                if slide_refresh_block_total is not None
                                else n_refresh_total
                            )
                            effective_idx = slide_refresh_start + refresh_idx
                            slide_alpha = (
                                1.0 - effective_idx / max(effective_total - 1, 1)
                                if effective_total > 1 else 1.0
                            )
                            refreshed_grad, refresh_meta = gradient_refresh_fn(
                                weight_snapshot,
                                slide_alpha=slide_alpha,
                            )
                            refreshed_grad = refreshed_grad.to(self.dev).float()
                            # Diagnostic stats: disabled by default because each
                            # metric costs a torch op + .item() (cudaStreamSync)
                            # and the trailing tensors shrink only linearly across
                            # blocks → early blocks pay 10-20 such syncs each.
                            # Only `mean_refresh_loss` (from refresh_meta) survives
                            # when refresh_full_metrics is off; everything else is
                            # None and the log line shows "None" for those fields.
                            trailing_grad_chunks = [] if refresh_full_metrics else None
                            if refresh_full_metrics:
                                for state in subgroup_states:
                                    refreshed_grad_sub = refreshed_grad[state["row_start"]:state["row_end"], :]
                                    if actorder:
                                        refreshed_grad_sub = refreshed_grad_sub[:, state["perm"]]
                                    trailing_grad = refreshed_grad_sub[:, i2:]
                                    if trailing_grad.numel() > 0:
                                        trailing_grad_chunks.append(trailing_grad)

                            trailing_grad_abs_mean = None
                            trailing_grad_mean_row_l2 = None
                            trailing_grad_clipped_abs_mean = None
                            second_order_abs_mean = None
                            second_order_mean_row_l2 = None
                            second_order_abs_max = None
                            second_order_abs_q99 = None
                            first_order_raw_abs_mean = None
                            first_order_raw_mean_row_l2 = None
                            first_order_abs_mean = None
                            first_order_mean_row_l2 = None
                            first_order_abs_max = None
                            first_order_abs_q99 = None
                            regularizer_abs_mean = None
                            regularizer_mean_row_l2 = None
                            sine_regularizer_abs_mean = None
                            sine_regularizer_mean_row_l2 = None
                            # These accumulators are pure stats; only populate them
                            # when refresh_full_metrics is on. state["pending_optimizer_update"]
                            # is always set below (algorithm depends on it).
                            optimizer_updates_raw = [] if refresh_full_metrics else None
                            optimizer_updates = [] if refresh_full_metrics else None
                            gate_regularizer_updates = [] if refresh_full_metrics else None
                            sine_regularizer_updates = [] if refresh_full_metrics else None
                            if refresh_full_metrics and block_second_order_chunks:
                                second_order_cat = torch.cat(block_second_order_chunks, dim=0).float()
                                _so_abs = second_order_cat.abs()
                                second_order_abs_mean = _so_abs.mean().item()
                                second_order_mean_row_l2 = torch.linalg.norm(
                                    second_order_cat,
                                    dim=1,
                                ).mean().item()
                                # Outlier stats — the max and 0.99 quantile of
                                # |second-order update|. Useful to spot whether a
                                # handful of rows/cols dominate the step; the mean
                                # alone hides heavy-tailed distributions.
                                second_order_abs_max = _so_abs.max().item()
                                second_order_abs_q99 = _quantile_large(_so_abs, 0.99)
                            if refresh_full_metrics and trailing_grad_chunks:
                                trailing_grad_cat = torch.cat(trailing_grad_chunks, dim=0).float()
                                trailing_grad_abs_mean = trailing_grad_cat.abs().mean().item()
                                trailing_grad_mean_row_l2 = torch.linalg.norm(
                                    trailing_grad_cat,
                                    dim=1,
                                ).mean().item()
                                # Post-clip abs mean — for verifying `grad_clip` takes effect.
                                # Adam normalises |grad|, so a small grad_clip only shows up
                                # here, not in `first_raw_abs` / the applied step size.
                                if grad_clip is not None and grad_clip > 0:
                                    trailing_grad_clipped_abs_mean = (
                                        trailing_grad_cat.clamp(min=-grad_clip, max=grad_clip)
                                        .abs().mean().item()
                                    )
                                else:
                                    trailing_grad_clipped_abs_mean = trailing_grad_abs_mean
                            for state in subgroup_states:
                                refreshed_grad_sub = refreshed_grad[state["row_start"]:state["row_end"], :]
                                if actorder:
                                    refreshed_grad_sub = refreshed_grad_sub[:, state["perm"]]
                                current_effective_weight = state["W_sub"].clone()
                                current_effective_weight[:, :i2] = state["Q"][:, :i2]
                                optimizer_update_raw = self._compute_grad_optimizer_update(
                                    state["grad_optimizer_state"],
                                    refreshed_grad_sub,
                                    i2,
                                    grad_lr,
                                    grad_clip=grad_clip,
                                )
                                optimizer_update, gate_regularizer_update, sine_regularizer_update = self._apply_first_order_regularizer(
                                    state,
                                    current_effective_weight,
                                    optimizer_update_raw,
                                    i2,
                                    grad_lr,
                                    grad_reg_strategy,
                                    grad_reg_lambda,
                                    grad_gate_floor,
                                    grad_gate_sharpness,
                                    grad_gate_sine_amp,
                                )
                                # Diagnostic: block_gd refresh results PER SUBGROUP.
                                # refreshed_grad_sub is the per-row gradient of the
                                # refresh loss wrt this subgroup's weight; optimizer_update
                                # is what Adam/SGD will push onto W_sub trailing AFTER
                                # this loop finishes.
                                if rec is not None:
                                    block_idx = i1 // blocksize
                                    sub_idx = state["sub_idx"]
                                    rec.save_block(
                                        sub_idx, block_idx, "refreshed_grad_subgroup", refreshed_grad_sub,
                                    )
                                    rec.save_block(
                                        sub_idx, block_idx, "optimizer_update_raw", optimizer_update_raw,
                                    )
                                    rec.save_block(
                                        sub_idx, block_idx, "optimizer_update", optimizer_update,
                                    )
                                    rec.save_block(
                                        sub_idx, block_idx, "current_effective_weight_at_refresh",
                                        current_effective_weight,
                                    )
                                    opt_state = state.get("grad_optimizer_state")
                                    if opt_state is not None and opt_state.get("type") == "adam":
                                        rec.save_block(
                                            sub_idx, block_idx, "adam_exp_avg_after_update",
                                            opt_state["exp_avg"],
                                        )
                                        rec.save_block(
                                            sub_idx, block_idx, "adam_exp_avg_sq_after_update",
                                            opt_state["exp_avg_sq"],
                                        )
                                        rec.save_block(
                                            sub_idx, block_idx, "adam_step",
                                            torch.tensor(opt_state["step"]),
                                        )
                                    # Record loss so finalize() can auto-detect spikes.
                                    if refresh_meta is not None:
                                        rec.record_loss(
                                            sub_idx, block_idx,
                                            refresh_meta.get("mean_refresh_loss"),
                                        )
                                        rec.save_block(
                                            sub_idx, block_idx, "mean_refresh_loss",
                                            torch.tensor(float(refresh_meta.get("mean_refresh_loss", 0.0))),
                                        )
                                state["pending_optimizer_update"] = optimizer_update
                                if refresh_full_metrics:
                                    if optimizer_update_raw.numel() > 0:
                                        optimizer_updates_raw.append(optimizer_update_raw)
                                    if optimizer_update.numel() > 0:
                                        optimizer_updates.append(optimizer_update)
                                        gate_regularizer_updates.append(gate_regularizer_update)
                                        sine_regularizer_updates.append(sine_regularizer_update)
                            if refresh_full_metrics and optimizer_updates_raw:
                                optimizer_update_raw_cat = torch.cat(optimizer_updates_raw, dim=0).float()
                                first_order_raw_abs_mean = optimizer_update_raw_cat.abs().mean().item()
                                first_order_raw_mean_row_l2 = torch.linalg.norm(
                                    optimizer_update_raw_cat,
                                    dim=1,
                                ).mean().item()
                            if refresh_full_metrics and optimizer_updates:
                                optimizer_update_cat = torch.cat(optimizer_updates, dim=0).float()
                                _fo_abs = optimizer_update_cat.abs()
                                first_order_abs_mean = _fo_abs.mean().item()
                                first_order_mean_row_l2 = torch.linalg.norm(
                                    optimizer_update_cat,
                                    dim=1,
                                ).mean().item()
                                # Outlier stats for the applied first-order step
                                # (post regularizer / gate). Paired with the
                                # second-order outlier stats above to see which
                                # term produces the spikiest updates.
                                first_order_abs_max = _fo_abs.max().item()
                                first_order_abs_q99 = _quantile_large(_fo_abs, 0.99)
                            if refresh_full_metrics and gate_regularizer_updates:
                                regularizer_update_cat = torch.cat(gate_regularizer_updates, dim=0).float()
                                regularizer_abs_mean = regularizer_update_cat.abs().mean().item()
                                regularizer_mean_row_l2 = torch.linalg.norm(
                                    regularizer_update_cat,
                                    dim=1,
                                ).mean().item()
                            if refresh_full_metrics and sine_regularizer_updates:
                                sine_regularizer_update_cat = torch.cat(sine_regularizer_updates, dim=0).float()
                                sine_regularizer_abs_mean = sine_regularizer_update_cat.abs().mean().item()
                                sine_regularizer_mean_row_l2 = torch.linalg.norm(
                                    sine_regularizer_update_cat,
                                    dim=1,
                                ).mean().item()
                            if block_observer is not None:
                                block_observer(
                                    {
                                        "block_idx": i1 // blocksize,
                                        "col_start": i1,
                                        "col_end": i2,
                                        "remaining_columns": self.columns - i2,
                                        "remaining_grad_abs_mean": trailing_grad_abs_mean,
                                        "remaining_grad_clipped_abs_mean": trailing_grad_clipped_abs_mean,
                                        "remaining_grad_mean_row_l2": trailing_grad_mean_row_l2,
                                        "second_order_update_abs_mean": second_order_abs_mean,
                                        "second_order_update_mean_row_l2": second_order_mean_row_l2,
                                        "second_order_update_abs_max": second_order_abs_max,
                                        "second_order_update_abs_q99": second_order_abs_q99,
                                        "first_order_raw_abs_mean": first_order_raw_abs_mean,
                                        "first_order_raw_mean_row_l2": first_order_raw_mean_row_l2,
                                        "first_order_update_abs_mean": first_order_abs_mean,
                                        "first_order_update_mean_row_l2": first_order_mean_row_l2,
                                        "first_order_update_abs_max": first_order_abs_max,
                                        "first_order_update_abs_q99": first_order_abs_q99,
                                        "regularizer_update_abs_mean": regularizer_abs_mean,
                                        "regularizer_update_mean_row_l2": regularizer_mean_row_l2,
                                        "sine_regularizer_update_abs_mean": sine_regularizer_abs_mean,
                                        "sine_regularizer_update_mean_row_l2": sine_regularizer_mean_row_l2,
                                        "mean_refresh_loss": None if refresh_meta is None else refresh_meta.get("mean_refresh_loss"),
                                        "train_mean_refresh_loss": None if refresh_meta is None else refresh_meta.get("train_mean_refresh_loss"),
                                        "val_mean_refresh_loss": None if refresh_meta is None else refresh_meta.get("val_mean_refresh_loss"),
                                        "refresh_subset_mean_refresh_loss": None if refresh_meta is None else refresh_meta.get("refresh_subset_mean_refresh_loss"),
                                        "slide_alpha": None if refresh_meta is None else refresh_meta.get("slide_alpha"),
                                        "mean_refresh_loss_current": None if refresh_meta is None else refresh_meta.get("mean_refresh_loss_current"),
                                        "mean_refresh_loss_next": None if refresh_meta is None else refresh_meta.get("mean_refresh_loss_next"),
                                        "sample_indices": () if refresh_meta is None else refresh_meta.get("sample_indices", ()),
                                    }
                                )

                        for state in subgroup_states:
                            with profile_recorder.section("fasterquant.block.outer_update_grad_descent") if profile_recorder else _NULL_CONTEXT:
                                optimizer_update = state.pop("pending_optimizer_update")
                                if optimizer_update.numel() > 0:
                                    state["W_sub"][:, i2:] -= optimizer_update
                                # Diagnostic: W_sub trailing AFTER Adam apply —
                                # i.e. the state that block i+1 will see as W1_start
                                # for the columns [i2, i2+blocksize).
                                if rec is not None:
                                    rec.save_block(
                                        state["sub_idx"], i1 // blocksize,
                                        "W_trailing_after_adam",
                                        state["W_sub"][:, i2:].clone(),
                                    )

            for state in subgroup_states:
                with profile_recorder.section("fasterquant.subgroup.total") if profile_recorder else _NULL_CONTEXT:
                    Q = state["Q"]
                    W_int_sub = state["W_int_sub"]
                    Scale_sub = state["Scale_sub"]
                    if actorder:
                        with profile_recorder.section("fasterquant.subgroup.actorder_unpermute") if profile_recorder else _NULL_CONTEXT:
                            Q = Q[:, state["invperm"]]
                            W_int_sub = W_int_sub[:, state["invperm"]]
                            Scale_sub = Scale_sub[:, state["invperm"]]

                    with profile_recorder.section("fasterquant.subgroup.writeback_outputs") if profile_recorder else _NULL_CONTEXT:
                        Q_final[state["row_start"]:state["row_end"], :] = Q
                        W_int_final[state["row_start"]:state["row_end"], :] = W_int_sub
                        Scale_final[state["row_start"]:state["row_end"], :] = Scale_sub

            if export_to_et:
                with profile_recorder.section("fasterquant.export_buffers") if profile_recorder else _NULL_CONTEXT:
                    self.layer.register_buffer(
                        "int_weight", W_int_final.reshape(self.layer.weight.shape)
                    )
                    self.layer.register_buffer("scale", Scale_final)
            with profile_recorder.section("fasterquant.write_layer_weight") if profile_recorder else _NULL_CONTEXT:
                self.layer.weight.data = Q_final.reshape(self.layer.weight.shape).to(
                    self.layer.weight.data.dtype
                )
            # Diagnostic: final quantized weight (natural order, matches what gets
            # written back to the model) and its int-code / scale equivalents.
            if rec is not None:
                rec.save_module_level("Q_final", Q_final)
                rec.save_module_level("W_int_final", W_int_final)
                rec.save_module_level("Scale_final", Scale_final)
            if torch.any(torch.isnan(self.layer.weight.data)):
                logging.warning("NaN in weights")
                logging.warning(
                    "bits=%s scale_stats=(min=%s max=%s any_nan=%s any_zero=%s) "
                    "zero_stats=(min=%s max=%s any_nan=%s)",
                    self.quantizer.bits,
                    self.quantizer.scale.float().min().item() if self.quantizer.scale is not None else None,
                    self.quantizer.scale.float().max().item() if self.quantizer.scale is not None else None,
                    torch.isnan(self.quantizer.scale).any().item() if self.quantizer.scale is not None else None,
                    (self.quantizer.scale == 0).any().item() if self.quantizer.scale is not None else None,
                    self.quantizer.zero.float().min().item() if self.quantizer.zero is not None else None,
                    self.quantizer.zero.float().max().item() if self.quantizer.zero is not None else None,
                    torch.isnan(self.quantizer.zero).any().item() if self.quantizer.zero is not None else None,
                )
                raise ValueError("NaN in weights")

    def analyze_ghinv_dynamics(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        enable_gradient_update=True,
        g_update_mode="frozen",
        max_subgroups=None,
        max_blocks=None,
        return_tensors=False,
    ):
        def summarize_tensor_rows(tensor: torch.Tensor):
            if tensor.numel() == 0:
                return None
            tensor = tensor.float()
            row_l2 = torch.linalg.norm(tensor, dim=1)
            return {
                "mean_row_l2": row_l2.mean().item(),
                "median_row_l2": row_l2.median().item(),
                "max_row_l2": row_l2.max().item(),
                "fro_norm": torch.linalg.norm(tensor).item(),
                "abs_mean": tensor.abs().mean().item(),
            }

        def summarize_vector(vector: torch.Tensor):
            vector = vector.float().flatten()
            return {
                "mean": vector.mean().item(),
                "median": vector.median().item(),
                "min": vector.min().item(),
                "max": vector.max().item(),
            }

        W = self.layer.weight.data.clone().float()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        rows_per_sub = self.rows // self.num_groups
        results = {
            "rows": self.rows,
            "columns": self.columns,
            "num_groups": self.num_groups,
            "blocksize": blocksize,
            "g_update_mode": g_update_mode,
            "enable_gradient_update": bool(enable_gradient_update),
            "subgroups": [],
        }
        tensor_cache = []

        for sub_idx in range(self.num_groups):
            if max_subgroups is not None and sub_idx >= max_subgroups:
                break
            row_start = sub_idx * rows_per_sub
            row_end = (sub_idx + 1) * rows_per_sub

            W_sub = W[row_start:row_end, :].clone()
            H_sub = self.H[sub_idx].clone()
            gradients_sub = self.gradients[row_start:row_end, :].to(self.dev).clone()

            dead = torch.diag(H_sub) == 0
            H_sub[dead, dead] = 1
            W_sub[:, dead] = 0

            if static_groups:
                groups = []
                for i in range(0, self.columns, groupsize):
                    quantizer = copy.deepcopy(self.quantizer)
                    quantizer.find_params(W[:, i : (i + groupsize)])
                    groups.append(quantizer)

            if actorder:
                perm = torch.argsort(self.act_square, descending=True)
                W_sub = W_sub[:, perm]
                H_sub = H_sub[perm][:, perm]
                gradients_sub = gradients_sub[:, perm]
            selected_cols = self._selected_column_count(blocksize, max_blocks)

            Hinv_init, Hinv, damp_percent, hessian_identity_fallback = (
                self._compute_hessian_inverse_with_fallback(
                    H_sub,
                    percdamp=percdamp,
                    log_context=f" analyze_ghinv_dynamics subgroup={sub_idx}",
                )
            )

            if enable_gradient_update and self.alpha > 0:
                alpha = self.alpha / (self.rows * self.columns)
                beta, GHinv_init = compute_safe_beta_from_reference_loss(
                    gradients_sub,
                    Hinv_init,
                    self.reference_loss,
                    alpha,
                )
            else:
                beta = torch.zeros([1]).to(gradients_sub)
                GHinv_init = gradients_sub.matmul(Hinv_init)

            Z = gradients_sub.matmul(Hinv.T) * beta.unsqueeze(1)
            GHinv = Z.matmul(Hinv)
            Z_raw = gradients_sub.matmul(Hinv.T)
            GHinv_raw = Z_raw.matmul(Hinv)
            D = torch.arange(blocksize - 1, -1, -1).to(GHinv)
            beta_view = beta.unsqueeze(1)
            W_ref_sub = W_sub.clone()

            def current_ghinv(base, current_weight, ref_weight, refresh_mode):
                if refresh_mode == "frozen":
                    return base
                return base + beta_view * (current_weight - ref_weight)

            def current_raw_ghinv(base, current_weight, ref_weight, refresh_mode):
                if refresh_mode == "frozen":
                    return base
                return base + (current_weight - ref_weight)

            subgroup_result = {
                "sub_idx": sub_idx,
                "row_start": row_start,
                "row_end": row_end,
                "hessian_identity_fallback": hessian_identity_fallback,
                "beta": summarize_vector(beta),
                "initial_raw_ghinv": summarize_tensor_rows(GHinv_raw[:, :selected_cols]),
                "initial_ghinv": summarize_tensor_rows(GHinv[:, :selected_cols]),
                "blocks": [],
            }

            for i1 in range(0, self.columns, blocksize):
                if max_blocks is not None and (i1 // blocksize) >= max_blocks:
                    break
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = W_sub[:, i1:i2].clone()
                W_ref1 = W_ref_sub[:, i1:i2]
                W_block_start = W1.clone()
                Err1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]
                GHinv1 = GHinv[:, i1:i2].clone()
                GHinv1_raw = GHinv_raw[:, i1:i2].clone()
                Z1 = Z[:, i1:i2]
                Z1_raw = Z_raw[:, i1:i2]
                GHinv1_eff = current_ghinv(
                    GHinv1,
                    W1 if g_update_mode == "surrogate_online" else W_block_start,
                    W_ref1,
                    g_update_mode,
                )
                GHinv1_eff_raw = current_raw_ghinv(
                    GHinv1_raw,
                    W1 if g_update_mode == "surrogate_online" else W_block_start,
                    W_ref1,
                    g_update_mode,
                )

                block_result = {
                    "block_idx": i1 // blocksize,
                    "col_start": i1,
                    "col_end": i2,
                    "base_raw_ghinv_before": summarize_tensor_rows(GHinv_raw[:, i1:i2]),
                    "eff_raw_ghinv_before": summarize_tensor_rows(GHinv1_eff_raw),
                    "base_ghinv_before": summarize_tensor_rows(GHinv[:, i1:i2]),
                    "eff_ghinv_before": summarize_tensor_rows(GHinv1_eff),
                    "weight_delta_before": summarize_tensor_rows(W1 - W_ref1),
                }

                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    if groupsize != -1:
                        if not static_groups:
                            if (i1 + i) % groupsize == 0:
                                self.quantizer.find_params(
                                    W[:, (i1 + i) : (i1 + i + groupsize)]
                                )
                        else:
                            idx = i1 + i
                            if actorder:
                                idx = perm[idx]
                            self.quantizer = groups[idx // groupsize]

                    q, _, _ = self.quantizer.fake_quantize(
                        w.unsqueeze(1), st_idx=row_start, end_idx=row_end
                    )
                    q = q.flatten()

                    err1 = (w - q - GHinv1_eff[:, i]) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0)) + GHinv1_eff[:, i:]
                    Err1[:, i] = err1

                    GHinv1[:, i:] = GHinv1[:, i:] - Z1[:, i].unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    GHinv1_raw[:, i:] = GHinv1_raw[:, i:] - Z1_raw[:, i].unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    GHinv1_eff = current_ghinv(
                        GHinv1,
                        W1 if g_update_mode == "surrogate_online" else W_block_start,
                        W_ref1,
                        g_update_mode,
                    )
                    GHinv1_eff_raw = current_raw_ghinv(
                        GHinv1_raw,
                        W1 if g_update_mode == "surrogate_online" else W_block_start,
                        W_ref1,
                        g_update_mode,
                    )

                block_result["eff_raw_ghinv_after_inner"] = summarize_tensor_rows(GHinv1_eff_raw)
                block_result["eff_ghinv_after_inner"] = summarize_tensor_rows(GHinv1_eff)
                block_result["err_after_inner"] = summarize_tensor_rows(Err1)
                block_result["weight_delta_after_inner"] = summarize_tensor_rows(W1 - W_ref1)

                GHinv_rest = current_ghinv(
                    GHinv[:, i2:],
                    W_sub[:, i2:],
                    W_ref_sub[:, i2:],
                    "frozen" if g_update_mode == "frozen" else "surrogate_block",
                )
                GHinv_rest_raw = current_raw_ghinv(
                    GHinv_raw[:, i2:],
                    W_sub[:, i2:],
                    W_ref_sub[:, i2:],
                    "frozen" if g_update_mode == "frozen" else "surrogate_block",
                )
                block_result["trailing_eff_raw_ghinv_before"] = summarize_tensor_rows(GHinv_rest_raw)
                block_result["trailing_eff_ghinv_before"] = summarize_tensor_rows(GHinv_rest)

                G_Update = blocksize * GHinv_rest - torch.einsum("ij,j,jk->ik", Z1, D, Hinv[i1:i2, i2:])
                W_sub[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:]) + G_Update
                GHinv[:, i2:] -= Z[:, i1:i2].matmul(Hinv[i1:i2, i2:])
                GHinv_raw[:, i2:] -= Z_raw[:, i1:i2].matmul(Hinv[i1:i2, i2:])

                GHinv_rest_after = current_ghinv(
                    GHinv[:, i2:],
                    W_sub[:, i2:],
                    W_ref_sub[:, i2:],
                    "frozen" if g_update_mode == "frozen" else "surrogate_block",
                )
                GHinv_rest_after_raw = current_raw_ghinv(
                    GHinv_raw[:, i2:],
                    W_sub[:, i2:],
                    W_ref_sub[:, i2:],
                    "frozen" if g_update_mode == "frozen" else "surrogate_block",
                )
                block_result["trailing_eff_raw_ghinv_after"] = summarize_tensor_rows(GHinv_rest_after_raw)
                block_result["trailing_eff_ghinv_after"] = summarize_tensor_rows(GHinv_rest_after)
                subgroup_result["blocks"].append(block_result)

            subgroup_result["final_weight_delta"] = summarize_tensor_rows(W_sub - W_ref_sub)
            final_raw_eff = current_raw_ghinv(
                GHinv_raw,
                W_sub,
                W_ref_sub,
                "frozen" if g_update_mode == "frozen" else "surrogate_block",
            )
            subgroup_result["final_raw_eff_ghinv"] = summarize_tensor_rows(final_raw_eff[:, :selected_cols])
            subgroup_result["final_base_ghinv"] = summarize_tensor_rows(GHinv[:, :selected_cols])
            subgroup_result["final_eff_ghinv"] = summarize_tensor_rows(
                current_ghinv(
                    GHinv,
                    W_sub,
                    W_ref_sub,
                    "frozen" if g_update_mode == "frozen" else "surrogate_block",
                )[:, :selected_cols]
            )
            results["subgroups"].append(subgroup_result)
            if return_tensors:
                tensor_cache.append(
                    {
                        "sub_idx": sub_idx,
                        "initial_raw_ghinv": GHinv_raw[:, :selected_cols].detach().cpu(),
                        "final_raw_eff_ghinv": final_raw_eff[:, :selected_cols].detach().cpu(),
                    }
                )

        if return_tensors:
            results["_tensor_cache"] = tensor_cache

        return results

    def summarize_ghinv(
        self,
        percdamp=0.01,
        blocksize=128,
        actorder=False,
        enable_gradient_update=True,
        gradient_override=None,
        reference_loss_override=None,
        max_subgroups=None,
        max_blocks=None,
        return_tensors=False,
    ):
        def summarize_tensor_rows(tensor: torch.Tensor):
            if tensor.numel() == 0:
                return None
            tensor = tensor.float()
            row_l2 = torch.linalg.norm(tensor, dim=1)
            return {
                "mean_row_l2": row_l2.mean().item(),
                "median_row_l2": row_l2.median().item(),
                "max_row_l2": row_l2.max().item(),
                "fro_norm": torch.linalg.norm(tensor).item(),
                "abs_mean": tensor.abs().mean().item(),
            }

        def summarize_vector(vector: torch.Tensor):
            vector = vector.float().flatten()
            return {
                "mean": vector.mean().item(),
                "median": vector.median().item(),
                "min": vector.min().item(),
                "max": vector.max().item(),
            }

        gradients = self.gradients if gradient_override is None else gradient_override.float()
        effective_reference_loss = (
            self.reference_loss
            if reference_loss_override is None
            else max(float(reference_loss_override), 0.0)
        )
        rows_per_sub = self.rows // self.num_groups
        results = {
            "rows": self.rows,
            "columns": self.columns,
            "num_groups": self.num_groups,
            "enable_gradient_update": bool(enable_gradient_update),
            "subgroups": [],
        }
        tensor_cache = []

        for sub_idx in range(self.num_groups):
            if max_subgroups is not None and sub_idx >= max_subgroups:
                break

            row_start = sub_idx * rows_per_sub
            row_end = (sub_idx + 1) * rows_per_sub
            H_sub = self.H[sub_idx].clone()
            gradients_sub = gradients[row_start: row_end, :].to(self.dev).clone()

            dead = torch.diag(H_sub) == 0
            H_sub[dead, dead] = 1
            gradients_sub[:, dead] = 0

            if actorder:
                perm = torch.argsort(self.act_square, descending=True)
                H_sub = H_sub[perm][:, perm]
                gradients_sub = gradients_sub[:, perm]
            selected_cols = self._selected_column_count(blocksize, max_blocks)

            Hinv_init, _, damp_percent, hessian_identity_fallback = (
                self._compute_hessian_inverse_with_fallback(
                    H_sub,
                    percdamp=percdamp,
                    log_context=f" summarize_ghinv subgroup={sub_idx}",
                )
            )

            if enable_gradient_update and self.alpha > 0:
                alpha = self.alpha / (self.rows * self.columns)
                beta, GHinv_init = compute_safe_beta_from_reference_loss(
                    gradients_sub,
                    Hinv_init,
                    effective_reference_loss,
                    alpha,
                )
            else:
                beta = torch.zeros([1]).to(gradients_sub)
                GHinv_init = gradients_sub.matmul(Hinv_init)
            GHinv_raw = gradients_sub.matmul(Hinv_init)
            GHinv = GHinv_raw * beta.unsqueeze(1) if enable_gradient_update and self.alpha > 0 else torch.zeros_like(GHinv_raw)

            results["subgroups"].append(
                {
                    "sub_idx": sub_idx,
                    "row_start": row_start,
                    "row_end": row_end,
                    "hessian_identity_fallback": hessian_identity_fallback,
                    "beta": summarize_vector(beta),
                    "raw_ghinv": summarize_tensor_rows(GHinv_raw[:, :selected_cols]),
                    "ghinv": summarize_tensor_rows(GHinv[:, :selected_cols]),
                }
            )
            if return_tensors:
                tensor_cache.append(
                    {
                        "sub_idx": sub_idx,
                        "raw_ghinv": GHinv_raw[:, :selected_cols].detach().cpu(),
                        "ghinv": GHinv[:, :selected_cols].detach().cpu(),
                    }
                )

        if return_tensors:
            results["_tensor_cache"] = tensor_cache

        return results

    def summarize_weight_delta(
        self,
        blocksize=128,
        actorder=False,
        max_subgroups=None,
        max_blocks=None,
        ref_weight_override=None,
        weight_override=None,
        return_tensors=False,
    ):
        def summarize_tensor_rows(tensor: torch.Tensor):
            if tensor.numel() == 0:
                return None
            tensor = tensor.float()
            row_l2 = torch.linalg.norm(tensor, dim=1)
            return {
                "mean_row_l2": row_l2.mean().item(),
                "median_row_l2": row_l2.median().item(),
                "max_row_l2": row_l2.max().item(),
                "fro_norm": torch.linalg.norm(tensor).item(),
                "abs_mean": tensor.abs().mean().item(),
            }

        current_weight = self.layer.weight.data.clone().float() if weight_override is None else weight_override.float()
        ref_weight = current_weight.clone() if ref_weight_override is None else ref_weight_override.float()
        rows_per_sub = self.rows // self.num_groups
        selected_cols = self._selected_column_count(blocksize, max_blocks)
        results = {
            "rows": self.rows,
            "columns": self.columns,
            "num_groups": self.num_groups,
            "subgroups": [],
        }
        tensor_cache = []

        for sub_idx in range(self.num_groups):
            if max_subgroups is not None and sub_idx >= max_subgroups:
                break

            row_start = sub_idx * rows_per_sub
            row_end = (sub_idx + 1) * rows_per_sub
            delta = current_weight[row_start:row_end, :] - ref_weight[row_start:row_end, :]
            if actorder:
                perm = torch.argsort(self.act_square, descending=True)
                delta = delta[:, perm]
            delta = delta[:, :selected_cols]
            results["subgroups"].append(
                {
                    "sub_idx": sub_idx,
                    "row_start": row_start,
                    "row_end": row_end,
                    "weight_delta": summarize_tensor_rows(delta),
                }
            )
            if return_tensors:
                tensor_cache.append({"sub_idx": sub_idx, "weight_delta": delta.detach().cpu()})

        if return_tensors:
            results["_tensor_cache"] = tensor_cache

        return results

    def free(self):
        self.H = None
        self.Losses = None
        torch.cuda.empty_cache()
        memory_utils.cleanup_memory(verbos=False)


class SaliencyCache:
    """
    class for saving the output activation gradients in each layer.
    """
    def __init__(self, names, num_groups, sink_size=0):
        self.num_groups = num_groups
        self.saliency_cache = {}
        self.names = names
        for name in self.names:
            self.saliency_cache[name] = []
        self.handles = []
        self.hooks_enabled = False
        # Drop the first `sink_size` token positions from saliency. The sink
        # tokens still flow through forward (KV) and the loss-driven backward
        # populates non-zero gradients at sink positions via attention; this
        # field tells the hook to slice them off before squaring/grouping so
        # the per-token saliency tensor matches the "ignore_attention_sink"
        # interpretation in compute_refresh_loss.
        self.sink_size = max(0, int(sink_size))

    def cache_saliency(self, module, inp, out, name):
        # We'll store gradient on 'out', so we must retain it
        out.retain_grad()

        def grad_hook(grad):
            """
            grad shape typically [bsz, seq_len, hidden_dim].
            We group the channels and take the squared Euclidean norm.
            """
            if not self.hooks_enabled:
                return
            g = grad
            if self.sink_size > 0 and g.dim() >= 2 and g.shape[1] > self.sink_size:
                g = g[:, self.sink_size:]
            bsz, seq_len, hidden_dim = g.shape
            saliency = grouped_gradient_norm_squared(
                g, self.num_groups
            )  # -> [bsz, seq_len, num_groups]

            self.saliency_cache[name].append(saliency)

        # Attach the gradient hook to 'out'
        out.register_hook(grad_hook)

    def add_hook(self, full, enable=True):
        for name in self.names:
            _, module = resolve_quant_module(full, name)
            self.handles.append(
                module.register_forward_hook(
                    functools.partial(self.cache_saliency, name=name)
                )
            )
        self.hooks_enabled = enable

    def enable_hooks(self):
        self.hooks_enabled = True

    def disable_hooks(self):
        self.hooks_enabled = False

    def clear_hook(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.hooks_enabled = False
        memory_utils.cleanup_memory()

    def clear_cache(self):
        for name in self.names:
            self.saliency_cache[name] = []
        memory_utils.cleanup_memory()


class GradientCache:
    """
    class for saving the weight gradients in each layer.

    DP note: `cache_gradient` accumulates an unnormalised rank-local SUM of
    per-batch gradients; `finalize` all-reduces the sum and the batch count
    across ranks, then produces the global mean. This matches the pre-DP
    running-mean semantics: `mean_over_all_batches(batch_grad)` — as long as
    batches across ranks have the same size, the global mean equals what the
    single-GPU running-mean would produce.
    """
    def __init__(self, names, num_groups):
        self.num_groups = num_groups
        # Public result field; populated by finalize().
        self.gradients_cache = {}
        # Internal rank-local accumulators.
        self._gradients_sum = {}
        self._count = {}
        self.names = names
        for name in self.names:
            self._gradients_sum[name] = None
            self._count[name] = 0
        self.handles = []
        self.hooks_enabled = False

    def cache_gradient(self, grad, name):
        if not self.hooks_enabled:
            return
        grad_f = grad.float()
        if self._gradients_sum[name] is None:
            self._gradients_sum[name] = torch.zeros_like(grad_f)
        self._gradients_sum[name].add_(grad_f)
        self._count[name] += 1

    def finalize(self):
        """All-reduce per-module sums across ranks and divide by the global
        batch count to produce the mean. Populates `self.gradients_cache`.
        """
        from utils import dist_utils as _dist

        for name in self.names:
            total = self._gradients_sum[name]
            if total is None:
                # Nothing accumulated on any rank: produce a zero tensor only
                # after we know the shape. Defer: raise — this should not happen
                # because we only register hooks for modules that will fire.
                raise RuntimeError(
                    f"GradientCache.finalize: no gradient accumulated for `{name}`. "
                    f"Did a module silently skip its weight.grad hook?"
                )
            _dist.allreduce_sum_(total)
            global_count = _dist.allreduce_sum_scalar(self._count[name])
            if global_count <= 0:
                raise RuntimeError(f"GradientCache.finalize: zero count for `{name}`.")
            self.gradients_cache[name] = total / global_count
        # Release the raw sums — downstream code only reads `gradients_cache`.
        self._gradients_sum.clear()
        self._count.clear()

    def add_hook(self, full, enable=True):
        for name in self.names:
            _, module = resolve_quant_module(full, name)
            self.handles.append(
                module.weight.register_hook(
                    functools.partial(self.cache_gradient, name=name)
                )
            )
        self.hooks_enabled = enable

    def enable_hooks(self):
        self.hooks_enabled = True

    def disable_hooks(self):
        self.hooks_enabled = False

    def clear_hook(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.hooks_enabled = False
        memory_utils.cleanup_memory()

    def clear_cache(self):
        for name in self.names:
            self.gradients_cache[name] = 0
            self._gradients_sum[name] = None
            self._count[name] = 0
        memory_utils.cleanup_memory()


def hidden2logits(hidden_states, analyzer: model_utils.ModelAnalyzer):
    norm = analyzer.get_layernorm_before_head()
    lm_head = analyzer.get_lm_head()

    logits = lm_head(norm(hidden_states))

    return logits


def _deterministic_categorical_labels(logits: torch.Tensor, global_sample_indices,
                                      base_seed: int) -> torch.Tensor:
    """Per-sample deterministic `Categorical(logits).sample()`.

    Each row's labels depend ONLY on that row's global sample index, not on
    the batch shape or how samples are grouped. This makes static saliency /
    per-layer grad-hessian stats reproducible across shardings (1-GPU vs
    N-GPU DP): both paths produce the same labels for the same global sample
    ids, so downstream Fisher / saliency / gradient statistics match up to
    pure FP-order drift.

    logits: (bsz, ..., V)
    global_sample_indices: sequence of length bsz, each an int global sample id
    base_seed: tag that namespaces this call site (e.g. hash of layer_idx)
    """
    bsz = logits.shape[0]
    assert len(global_sample_indices) == bsz, (
        f"expected {bsz} global indices, got {len(global_sample_indices)}"
    )
    trailing_shape = logits.shape[1:-1]
    vocab = logits.shape[-1]
    out = torch.empty(logits.shape[:-1], dtype=torch.long, device=logits.device)
    for i in range(bsz):
        # torch.Generator accepts any device torch supports; we match logits.device.
        gen = torch.Generator(device=logits.device).manual_seed(
            int(1000003 * (base_seed * 100000 + int(global_sample_indices[i])) + 17)
        )
        # multinomial takes (num_rows, vocab) and a `generator` kwarg, which
        # `Categorical.sample` does not expose portably across torch versions.
        probs = torch.softmax(logits[i].reshape(-1, vocab).float(), dim=-1)
        sample = torch.multinomial(probs, 1, generator=gen).reshape(trailing_shape)
        out[i] = sample
    return out


def _deterministic_rademacher_signs(
    token_shape,
    global_sample_indices,
    base_seed: int,
    device,
    dtype=torch.float32,
) -> torch.Tensor:
    """Generate per-(sample, token) Rademacher signs on `device`.

    The seed is derived per global sample id, so a sample receives the same
    signs regardless of DP world size or batch packing. Each row is generated
    as one device tensor; the Python loop is only over batch rows.
    """
    bsz, seq_len = token_shape
    assert len(global_sample_indices) == bsz, (
        f"expected {bsz} global indices, got {len(global_sample_indices)}"
    )
    out = torch.empty((bsz, seq_len), device=device, dtype=dtype)
    for i, global_idx in enumerate(global_sample_indices):
        gen = torch.Generator(device=device).manual_seed(
            int(1000003 * (base_seed * 100000 + int(global_idx)) + 97)
        )
        signs_i = torch.randint(
            0,
            2,
            (seq_len,),
            generator=gen,
            device=device,
            dtype=torch.int8,
        )
        out[i].copy_(signs_i.to(dtype=dtype).mul_(2).sub_(1))
    return out



def collect_static_end_to_end_saliency_and_fisher(
    *,
    model,
    analyzer,
    dataloader,
    dev,
    saliency_num_groups,
    grad_hessian_topk,
    batch_size,
    collect_fisher=True,
    collect_legacy_fisher_diag=False,
    collect_refined_rkl=False,
    refined_rkl_damp=0.01,
    refined_rkl_num_A=1,
    refined_rkl_store_dtype=torch.bfloat16,
    collect_refined_diag_rkl=False,
    refined_diag_rkl_store_dtype=torch.bfloat16,
    use_fsdp=False,
    fsdp_cpu_offload=False,
    saliency_clip_percentile=0.99,
    profile_recorder=None,
    capture_fp_final=False,
    fp_final_store_dtype=torch.bfloat16,
    fisher_layer_ids=None,
    refined_rkl_layer_ids=None,
    collect_saliency=True,
    collect_dynsal=False,
    dynsal_rank=16,
    dynsal_evd_thresh=1e-6,
    sink_size=0,
    fisher_rademacher_k=0,
    rademacher_seed=0,
    num_samples_for_grad=0,
):
    fisher_rademacher_k = int(fisher_rademacher_k)
    num_samples_for_grad = int(num_samples_for_grad)
    if fisher_rademacher_k < 0:
        raise ValueError(
            f"fisher_rademacher_k must be non-negative, got {fisher_rademacher_k}."
        )
    if num_samples_for_grad < 0:
        raise ValueError(
            f"num_samples_for_grad must be non-negative, got {num_samples_for_grad}."
        )
    if fisher_rademacher_k > 0 and collect_dynsal:
        raise ValueError(
            "fisher_rademacher_k cannot be used with collect_dynsal=True because "
            "dynamic saliency's low-rank gradient decomposition is temporarily "
            "incompatible with signed-token gradient averaging."
        )
    use_rademacher_stats = bool(
        fisher_rademacher_k > 0
        and (collect_fisher or collect_saliency or collect_dynsal)
    )
    use_grad_sample_limit = bool(
        num_samples_for_grad > 0
        and (collect_fisher or collect_saliency or collect_dynsal)
    )
    if collect_legacy_fisher_diag and fisher_rademacher_k > 0:
        raise ValueError(
            "legacy_fisher_diag_mse stores per-sample/per-token g^2 from the "
            "sum-reduced NLL backward, so it cannot be combined with "
            "fisher_rademacher_k > 0."
        )
    if collect_legacy_fisher_diag and num_samples_for_grad > 0:
        raise ValueError(
            "legacy_fisher_diag_mse requires a per-token Fisher diagonal for "
            "every rank-local calibration sample. --num_samples_for_grad would "
            "collect only a prefix and break sample alignment."
        )
    logging.info(
        "Collecting static end-to-end saliency/fisher caches from a single pre-quantization full-model backward pass. "
        "Using sampled end-to-end NLL / empirical Fisher because literal KL-to-self before quantization would be zero."
    )
    if use_rademacher_stats:
        logging.info(
            "Static Fisher/saliency uses Rademacher token signs: k=%d repeated "
            "backward passes per batch, averaging g^2 / g g^T statistics.",
            fisher_rademacher_k,
        )
    if use_grad_sample_limit:
        logging.info(
            "Static Fisher/saliency precompute will use only the first %d samples "
            "from each rank-local calibration shard (%d global samples total).",
            num_samples_for_grad // max(1, dist_utils.get_world_size()),
            num_samples_for_grad,
        )
    if collect_refined_rkl:
        logging.info(
            "Also fitting refined_residual_kl matrix A per layer via least squares "
            "on (dy, dx-dy) pairs collected during the same backward pass. "
            "num_A=%d sub-matrices per layer, store_dtype=%s.",
            refined_rkl_num_A, str(refined_rkl_store_dtype).replace("torch.", ""),
        )
    if collect_refined_diag_rkl:
        logging.info(
            "Also fitting refined_diag_residual_kl per-(token, channel) diagonal J "
            "via LS on (dy, dx-dy) pairs aggregated across samples in each sub-A. "
            "num_A=%d, store_dtype=%s. Memory per layer per sub-A: (seq_len, H) bf16 "
            "≈ H×-smaller than full refined_residual_kl per sub-A when H>>seq_len.",
            refined_rkl_num_A, str(refined_diag_rkl_store_dtype).replace("torch.", ""),
        )
    collect_saliency = bool(collect_saliency or collect_dynsal)
    layers = analyzer.get_layers()
    # Wrap model with FSDP2 so the full-model backward fits on many small GPUs.
    # Params + grads are sharded across the current default process group; each
    # layer all-gathers on forward and reshards afterwards. Since all params
    # are frozen below (requires_grad=False), no reduce_scatter happens on
    # backward — FSDP becomes a pure param all-gather mechanism.
    #
    # Important: when `use_fsdp` is on, the model ends up with DTensor-wrapped
    # parameters. In-process "unwrap" attempts have proven fragile — the
    # residual DTensor state leaks into downstream `find_params` / backward
    # gradients. The supported workflow is **two-stage**: the caller must
    # combine `--fsdp_precompute` with `--exit_after_precompute` and
    # `--static_cache_path` so precompute exits immediately after saving to
    # disk; a subsequent run without `--fsdp_precompute` reads the cache and
    # performs quantisation. The sweep script wires this up automatically.
    fsdp_already_prepared = bool(getattr(model, "_gptqplus_fsdp_prepared", False))
    if use_fsdp and not fsdp_already_prepared:
        with profile_recorder.section("pipeline.static_fisher.fsdp_wrap") if profile_recorder else _NULL_CONTEXT:
            from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy
            from torch.distributed.device_mesh import init_device_mesh
            world = dist_utils.get_world_size()
            mesh = init_device_mesh("cuda", (world,))
            mp_policy = MixedPrecisionPolicy(
                param_dtype=next(iter(model.parameters())).dtype,
                reduce_dtype=torch.float32,
            )
            offload_policy = CPUOffloadPolicy(pin_memory=True) if fsdp_cpu_offload else None
            logging.info(
                "FSDP2 precompute: sharding %d layers across world=%d, cpu_offload=%s",
                len(layers), world, fsdp_cpu_offload,
            )
            # Shard each transformer block first (inner-most), then the outer
            # module so embed/norm/lm_head also get sharded.
            for layer in layers:
                kwargs = {"mesh": mesh, "mp_policy": mp_policy}
                if offload_policy is not None:
                    kwargs["offload_policy"] = offload_policy
                fully_shard(layer, **kwargs)
            kwargs = {"mesh": mesh, "mp_policy": mp_policy}
            if offload_policy is not None:
                kwargs["offload_policy"] = offload_policy
            fully_shard(model, **kwargs)
            model._gptqplus_fsdp_prepared = True
            # FSDP-managed params live on-rank already; don't model.to(dev).
    elif use_fsdp and fsdp_already_prepared:
        logging.info("FSDP2 precompute: model was already sharded before checkpoint load; skipping FSDP wrap.")
    with profile_recorder.section("pipeline.static_fisher.build_module_dicts") if profile_recorder else _NULL_CONTEXT:
        module_dicts = []
        for layer in layers:
            raw_module_dict = analyzer.get_quantizable_modules(layer)
            normalized_module_dict = {}
            for module_name, module in raw_module_dict.items():
                canonical_name = normalize_quant_module_name(module_name)
                if canonical_name in normalized_module_dict and normalized_module_dict[canonical_name] is not module:
                    raise ValueError(
                        f"Duplicate canonical quant module name `{canonical_name}` detected while collecting static saliency/Fisher."
                    )
                normalized_module_dict[canonical_name] = module
            module_dicts.append(normalized_module_dict)
    saliency_data = (
        [
            {module_name: [] for module_name in module_dict.keys()}
            for module_dict in module_dicts
        ]
        if collect_saliency else None
    )
    fisher_data = [None for _ in layers]
    legacy_fisher_diag_data = (
        [[] for _ in layers] if collect_legacy_fisher_diag else None
    )
    # Any refined-rkl variant reuses the same num_A / samples_per_A partitioning
    # and the shared dy capture at the last transformer block's output.
    _collect_any_refined = collect_refined_rkl or collect_refined_diag_rkl
    # Refined residual_kl least-squares accumulators, one per (layer, sub-A)
    # pair. We fit A_{l,a} (H×H) such that f_l(x+Δx) ≈ f_l(x) + A_{l,a}·Δx,
    # where a is the sub-A index. When num_A>1, samples are split into num_A
    # contiguous groups of size `samples_per_A = nsamples/num_A`, each group
    # gets its own A. The LS per (layer, a) comes from the backward relation
    #   dx_i - dy = A_iᵀ · dy     (column-vec convention, A_i = Jacobian)
    # which with N = batch*seq flattened samples stacked as rows gives
    #   delta = dy · A_iᵀ  ⇒  A_i = inv(dy.t()·dy + λI) · (dy.t()·delta)
    # Let H_{l,a} = Σ dyᵀdy and C_{l,a} = Σ dyᵀ·delta (both H×H), damp λ a
    # percentage of trace(H)/H, then A_{l,a} = inv(H + damp·I) @ C.
    # Accumulate on CPU fp32 (even when store_dtype=bf16) to avoid bf16
    # catastrophic cancellation in the sum.
    #
    # Peak CPU memory is bounded by ONE sub-A's worth of accumulators across
    # all layers (L × 2 × H²) rather than L × num_A × 2 × H² — we stream the
    # solve: once a sub-A's batches have all been consumed (detected by the
    # next batch's a_idx changing), solve A_{l,a} for every layer and free
    # the corresponding C/H slots immediately.
    refined_C = (
        [[None] * refined_rkl_num_A for _ in layers]
        if collect_refined_rkl else None
    )
    refined_H = (
        [[None] * refined_rkl_num_A for _ in layers]
        if collect_refined_rkl else None
    )
    static_refined_A = (
        [[None] * refined_rkl_num_A for _ in layers]
        if collect_refined_rkl else None
    )
    # refined_diag_residual_kl variant: ignore cross-channel coupling, fit a
    # per-(token, channel) diagonal of J by LS over samples within each sub-A.
    # Per-layer per-sub-A storage is (seq_len, H) in bf16, which shrinks the
    # footprint by roughly H/seq vs the full H×H per sub-A — close to an
    # order-of-magnitude saving for 70B (H=8192, seq=2048 → 4× smaller).
    # Accumulators (seq, H) fp32 stream-freed alongside the full version.
    refined_diag_C = (
        [[None] * refined_rkl_num_A for _ in layers]
        if collect_refined_diag_rkl else None
    )
    refined_diag_H = (
        [[None] * refined_rkl_num_A for _ in layers]
        if collect_refined_diag_rkl else None
    )
    static_refined_diag_A = (
        [[None] * refined_rkl_num_A for _ in layers]
        if collect_refined_diag_rkl else None
    )
    # Counter (list so inner closure can mutate) + remembered shape for the log.
    refined_fit_counter = [0]
    refined_fit_shape = [None]
    # Transient per-backward slot holding dy (grad wrt last-block output)
    # and the active sub-A index for the current batch. The last-layer
    # grad hook fires first during backward, which fills `dy`; earlier
    # layers' grad hooks then consume it in the same backward call. The
    # driver sets `a` before each backward call.
    refined_dy_buffer = {}
    refined_current_a_idx = {"a": 0}
    # Dynamic saliency low-rank decomposition: accumulate per-(layer, module)
    # G^T G on CPU fp32 across all rank-local backward batches. After the loop
    # all-reduce across ranks and do EVD to get V, Σ². G^T G itself is freed
    # right after EVD (can be ~500 MB per up/gate on Llama-7B). U is collected
    # in a SECOND backward pass below, once V, Σ are known.
    # Dynamic saliency randomized SVD: per-(layer, module) sketch state for
    # Pass 1, all kept on GPU at trivial cost compared to the previous
    # full-G^T G accumulator (~50 GB GPU for Llama-3-8B, ≥150 GB for 13B+).
    #
    # The sketch sizes are O(d_out × K) with K = 2 × dynsal_rank: roughly
    # 1-3 MB per module on Llama-3-8B (vs ~822 MB for up/gate's full G^T G).
    #
    # Layout per (layer, module):
    #   omega:       Ω ∈ ℝ^(d_out × K), Gaussian, deterministically seeded so
    #                every DP rank generates the SAME Ω (otherwise the sketch
    #                across ranks would be incoherent and all-reduce would mix
    #                projections from different sketches).
    #   sketch:      Y_sketch ∈ ℝ^(d_out × K), accumulator for G^T (G Ω). After
    #                Pass 1 we all-reduce + QR to get an orthonormal basis Q
    #                of the same shape (d_out × K). Storage gets reused: the
    #                same tensor's contents are overwritten with Q.
    # Both lazily-allocated on the first hook call to size them by the actual
    # output dim.
    dynsal_omega_data = (
        [
            {module_name: None for module_name in module_dict.keys()}
            for module_dict in module_dicts
        ]
        if collect_dynsal else None
    )
    dynsal_sketch_data = (
        [
            {module_name: None for module_name in module_dict.keys()}
            for module_dict in module_dicts
        ]
        if collect_dynsal else None
    )
    dynsal_K = (2 * int(dynsal_rank)) if collect_dynsal else 0
    # Populated by the dynsal second-pass block inside `try:` below when
    # `collect_dynsal=True`. Remains None otherwise so downstream unpacking
    # (`..., static_dynsal = collect_static_...`) always has something to bind.
    static_dynsal = None
    # Stable per-(layer, module) seed for Ω. We need every DP rank to draw the
    # SAME Gaussian Ω, so we derive a deterministic 31-bit integer from a
    # fixed algorithm namespace plus canonical (layer_idx, module_name). It
    # intentionally does not depend on the calibration seed and stays stable
    # regardless of rank or batching.
    import zlib as _zlib
    def _omega_seed(_layer_idx, _module_name):
        h = _zlib.crc32(_module_name.encode()) ^ (_layer_idx * 0x9E3779B1)
        return ((int(getattr(profile_recorder, "_dynsal_seed_base", 0xC0FFEE)) + h) & 0x7FFFFFFF)
    handles = []
    stat_repeats = max(1, int(fisher_rademacher_k))
    current_saliency_accum = {"data": None}
    current_backward_role = {"saliency_fisher": True, "refined": True}

    def make_module_hook(layer_idx, module_name):
        def forward_hook(module, inp, out):
            out_tensor = out[0] if isinstance(out, (tuple, list)) else out

            def grad_hook(grad):
                if not current_backward_role["saliency_fisher"]:
                    return
                # When --ignore_attention_sink is on, the loss above already
                # excluded sink positions, but autograd still populates non-zero
                # gradient at sink positions via attention back-flow. Drop the
                # sink slice here so saliency / dynsal sketches consistently
                # ignore the sink (matches compute_refresh_loss's slicing).
                if sink_size > 0 and grad.dim() >= 2 and grad.shape[1] > sink_size:
                    grad = grad[:, sink_size:]
                bsz_local, seq_len_local, hidden_dim = grad.shape
                if hidden_dim % saliency_num_groups != 0:
                    raise ValueError(
                        f"Module output dim ({hidden_dim}) must be divisible by saliency num_groups ({saliency_num_groups})."
                    )
                group_size = hidden_dim // saliency_num_groups
                # Single fp32 cast shared between saliency (group ||grad||²) and
                # dynsal (G^T G). Before merging these into one hook we cast
                # grad → fp32 twice per module per backward; now once.
                grad_fp32 = grad.detach().float()
                # --- saliency branch ---
                sal_per_group = grouped_gradient_norm_squared(
                    grad_fp32,
                    saliency_num_groups,
                )
                if collect_saliency:
                    sal_cpu = sal_per_group.detach().cpu()
                    if use_rademacher_stats:
                        sal_cpu.div_(float(stat_repeats))
                        accum = current_saliency_accum["data"]
                        if accum is None:
                            raise RuntimeError(
                                "Rademacher saliency accumulator is not initialised."
                            )
                        slot = accum[layer_idx][module_name]
                        if slot is None:
                            accum[layer_idx][module_name] = sal_cpu
                        else:
                            slot.add_(sal_cpu)
                    else:
                        saliency_data[layer_idx][module_name].append(sal_cpu)
                # --- dynsal branch: streaming randomized SVD sketch ---
                # `grad` is produced by a scalar loss that is summed over output
                # samples/tokens. Rows below are middle-layer tokens; the sketch is
                # an unnormalised middle-token sum. The 1/N middle-token factors
                # are applied later in `refresh_dynamic_saliency`, not here.
                # Replace the (d_out, d_out) G^T G accumulator (which alone
                # costs ~50 GB GPU on Llama-3-8B and OOMs at ≥13B) with a
                # rank-K Y_sketch = G^T (G Ω) accumulator of shape (d_out, K).
                # K = 2 × dynsal_rank covers the standard 2× oversampling.
                # Total per-module GPU is O(d_out × K × 4B) ≈ 1-3 MB at K=32.
                if collect_dynsal:
                    grad_flat = grad_fp32.reshape(-1, hidden_dim)  # (B·T, d_out)
                    Omega = dynsal_omega_data[layer_idx][module_name]
                    sketch = dynsal_sketch_data[layer_idx][module_name]
                    if Omega is None:
                        # Lazy-init Ω with a deterministic per-(layer, module)
                        # seed so every DP rank generates the same Gaussian.
                        omega_seed = _omega_seed(layer_idx, module_name)
                        omega_gen = torch.Generator(device=grad_flat.device)
                        omega_gen.manual_seed(omega_seed)
                        K_eff = min(dynsal_K, hidden_dim)
                        Omega = torch.randn(
                            hidden_dim, K_eff,
                            generator=omega_gen,
                            dtype=torch.float32,
                            device=grad_flat.device,
                        )
                        dynsal_omega_data[layer_idx][module_name] = Omega
                        sketch = torch.zeros(
                            hidden_dim, K_eff,
                            dtype=torch.float32,
                            device=grad_flat.device,
                        )
                        dynsal_sketch_data[layer_idx][module_name] = sketch
                    # Y += G^T (G Ω). Use (B·T, K) intermediate (small).
                    g_omega = grad_flat @ Omega  # (B·T, K) fp32
                    sketch.addmm_(
                        grad_flat.t(), g_omega,
                        alpha=1.0 / _E2E_PRECOMPUTE_QUADRATIC_SCALE,
                    )
                    del grad_flat, g_omega

            out_tensor.register_hook(grad_hook)

        return forward_hook

    def make_dynsal_module_hook(*args, **kwargs):
        """Removed. Dynsal G^T G is now computed inside `make_module_hook` as a
        randomized SVD sketch; calling this directly is a hard error to catch
        any out-of-tree caller that hasn't migrated."""
        raise RuntimeError(
            "make_dynsal_module_hook is obsolete; use make_module_hook (merged "
            "saliency + dynsal sketch hook)."
        )

    def make_layer_hook(layer_idx):
        """Per-layer end-to-end Fisher stats at the transformer block output.

        `collect_fisher=True` stores the full token-averaged (H, H) empirical
        Fisher used by fisher_diag_mse/refined_mse. `collect_legacy_fisher_diag`
        stores each rank-local token's diagonal g^2 separately as
        (N_local, T_eff, H), without summing across tokens.

        Same in-GPU-accumulator strategy as dynsal: keep the running sum on
        the device, transfer to CPU exactly once during aggregation. Before
        this change the hook did `(grad.t() @ grad).cpu()` + CPU fp32 add on
        every batch — 32 sync points per batch that drained the backward
        stream hard and left the GPU oscillating at low utilization."""
        def forward_hook(module, inp, out):
            out_tensor = out[0] if isinstance(out, (tuple, list)) else out

            def grad_hook(grad):
                if not current_backward_role["saliency_fisher"]:
                    return
                if sink_size > 0 and grad.dim() >= 2 and grad.shape[1] > sink_size:
                    grad = grad[:, sink_size:]
                grad_fp32 = grad.detach().float()
                if collect_legacy_fisher_diag:
                    legacy_fisher_diag_data[layer_idx].append(
                        grad_fp32.pow(2)
                        .div(_E2E_PRECOMPUTE_QUADRATIC_SCALE)
                        .to(torch.bfloat16)
                        .cpu()
                    )
                if collect_fisher:
                    grad_flat = grad_fp32.reshape(-1, grad_fp32.shape[-1])
                    # `grad` comes from an output-token SUM loss. Accumulate the
                    # unnormalised sum over middle-layer tokens here; aggregation
                    # below divides once by the global middle-token count to store
                    # the shared E_middle[g g^T] Fisher coefficient.
                    fisher_block = grad_flat.t() @ grad_flat  # stays on GPU fp32
                    fisher_block.div_(_E2E_PRECOMPUTE_QUADRATIC_SCALE * float(stat_repeats))
                    if fisher_data[layer_idx] is None:
                        fisher_data[layer_idx] = fisher_block
                    else:
                        fisher_data[layer_idx].add_(fisher_block)
                        del fisher_block

            out_tensor.register_hook(grad_hook)

        return forward_hook

    def make_refined_rkl_last_hook():
        """Capture dy = grad wrt last-transformer-block output into shared buffer.
        Runs FIRST in the backward (as this hook is closest to the loss); all
        earlier-layer refined_rkl hooks read from this slot in the same backward."""
        def forward_hook(module, inp, out):
            out_tensor = out[0] if isinstance(out, (tuple, list)) else out

            def grad_hook(grad):
                if not current_backward_role["refined"]:
                    return
                # Keep 3D fp32 on dev. The full-RKL flat view is produced on the
                # fly by the per-layer hook (reshape is a view — no copy). Diag
                # mode also needs the 3D structure so we cache once here.
                # When ignore_attention_sink is on, drop sink rows here so the
                # downstream LS fit (refined_C / refined_diag_C) only sees
                # non-sink token positions.
                if sink_size > 0 and grad.dim() >= 2 and grad.shape[1] > sink_size:
                    grad = grad[:, sink_size:]
                refined_dy_buffer["dy_3d"] = grad.detach().float()

            out_tensor.register_hook(grad_hook)

        return forward_hook

    def make_refined_rkl_layer_hook(layer_idx):
        """For layer_idx < N-1: capture dx_i, combine with dy from buffer, accumulate
        C_{l,a} = Σ dyᵀ·(dx_i - dy)  and  H_{l,a} = Σ dyᵀ·dy (both H×H fp32),
        where a = refined_current_a_idx["a"] (sub-A bucket for the current batch).

        Also accumulates per-(token, channel) diag sums when
        `collect_refined_diag_rkl=True`:
          C_diag_{l,a}[t, i] = Σ_s dy[s,t,i] · delta[s,t,i]      (seq, H)
          H_diag_{l,a}[t, i] = Σ_s dy[s,t,i]²                    (seq, H)
        Solve at sub-A flush time → A_diag_{l,a} shape (seq, H)."""
        def forward_hook(module, inp, out):
            out_tensor = out[0] if isinstance(out, (tuple, list)) else out

            def grad_hook(grad):
                if not current_backward_role["refined"]:
                    return
                if "dy_3d" not in refined_dy_buffer:
                    # Last-layer hook hasn't fired yet — shouldn't happen because
                    # backward flows last→first, but guard anyway to fail loudly.
                    raise RuntimeError(
                        f"refined_rkl layer {layer_idx}: dy not captured before "
                        "this hook. Check hook registration order."
                    )
                dy_3d = refined_dy_buffer["dy_3d"]  # (B, seq[-sink], H) fp32 on dev
                # `dy_3d` is already sliced if sink_size>0 (see last-hook above);
                # match `dx_3d` to the same length so the LS fit accumulators
                # stay shape-aligned.
                if sink_size > 0 and grad.dim() >= 2 and grad.shape[1] > sink_size:
                    dx_3d = grad[:, sink_size:].detach().float()
                else:
                    dx_3d = grad.detach().float()
                a = refined_current_a_idx["a"]

                if collect_refined_rkl:
                    dy = dy_3d.reshape(-1, dy_3d.shape[-1])
                    dx = dx_3d.reshape(-1, dx_3d.shape[-1])
                    delta = dx - dy
                    # dy.t() @ delta → (H, H); dy.t() @ dy → (H, H). Compute on dev
                    # then transfer to CPU fp32 for accumulation (avoid bf16
                    # cancellation in the sum even when A itself is stored bf16).
                    inc_C = (dy.t() @ delta).div_(_E2E_PRECOMPUTE_QUADRATIC_SCALE).cpu()
                    inc_H = (dy.t() @ dy).div_(_E2E_PRECOMPUTE_QUADRATIC_SCALE).cpu()
                    if refined_C[layer_idx][a] is None:
                        refined_C[layer_idx][a] = inc_C
                        refined_H[layer_idx][a] = inc_H
                    else:
                        refined_C[layer_idx][a] += inc_C
                        refined_H[layer_idx][a] += inc_H

                if collect_refined_diag_rkl:
                    # Per-(token, channel) LS across samples in the batch:
                    # accumulate Σ_s dy*delta and Σ_s dy² of shape (seq, H).
                    # Aggregation across batches within a sub-A happens via the
                    # running sum below; solve happens at sub-A flush.
                    delta_3d = dx_3d - dy_3d
                    inc_Cd = (dy_3d * delta_3d).sum(dim=0).div(_E2E_PRECOMPUTE_QUADRATIC_SCALE).cpu()
                    inc_Hd = dy_3d.pow(2).sum(dim=0).div(_E2E_PRECOMPUTE_QUADRATIC_SCALE).cpu()
                    if refined_diag_C[layer_idx][a] is None:
                        refined_diag_C[layer_idx][a] = inc_Cd
                        refined_diag_H[layer_idx][a] = inc_Hd
                    else:
                        refined_diag_C[layer_idx][a] += inc_Cd
                        refined_diag_H[layer_idx][a] += inc_Hd

            out_tensor.register_hook(grad_hook)

        return forward_hook

    # Optional forward-only capture of the last transformer block's output.
    # When `capture_fp_final=True`, store each rank-local batch's (B, T, H)
    # output on CPU in bf16, to be concatenated into the rank-local
    # `fp_inps_final` buffer after the loop. This lets residual_kl /
    # refined_residual_kl / refined_mse reuse the forward pass that's
    # happening anyway here, eliminating the separate bs=1 Stage 2
    # precompute in gptq_fwrd.
    fp_final_cache = [] if capture_fp_final else None
    if capture_fp_final:
        def fp_final_forward_hook(module, inp, out):
            out_tensor = out[0] if isinstance(out, (tuple, list)) else out
            # `.detach()` — autograd still builds the graph for fisher/saliency
            # grad hooks on the same out_tensor; we only copy the forward value.
            fp_final_cache.append(
                out_tensor.detach().to(fp_final_store_dtype).cpu()
            )

    for layer_idx, (layer, module_dict) in enumerate(zip(layers, module_dicts)):
        if (
            (collect_fisher or collect_legacy_fisher_diag)
            and (fisher_layer_ids is None or layer_idx in fisher_layer_ids)
        ):
            handles.append(layer.register_forward_hook(make_layer_hook(layer_idx)))
        if collect_saliency:
            for module_name, module in module_dict.items():
                # `make_module_hook` handles both saliency and (if collect_dynsal)
                # dynsal G^T G accumulation in a single grad hook to share the fp32
                # cast and halve the number of grad-hook invocations per backward.
                handles.append(module.register_forward_hook(make_module_hook(layer_idx, module_name)))
    if _collect_any_refined:
        # Register the last-layer "dy capture" hook FIRST in the hook list so
        # it runs first in the forward pass; that way `register_hook` attaches
        # the grad-hook before any earlier-layer grad-hook fires during
        # backward. (Hook firing order in autograd is reverse of output-grad
        # computation order, which flows last→first, so all layer hooks will
        # see the last-layer dy already populated.)
        last_idx = len(layers) - 1
        handles.append(layers[last_idx].register_forward_hook(make_refined_rkl_last_hook()))
        for layer_idx in range(last_idx):
            if refined_rkl_layer_ids is not None and layer_idx not in refined_rkl_layer_ids:
                continue
            handles.append(
                layers[layer_idx].register_forward_hook(make_refined_rkl_layer_hook(layer_idx))
            )
    if capture_fp_final:
        # Register AFTER the refined-rkl/fisher hooks on layers[-1] so the
        # forward order is fisher_hook → refined_rkl_hook → fp_final_hook.
        # They're independent (different closures), order doesn't affect
        # numerics.
        handles.append(layers[-1].register_forward_hook(fp_final_forward_hook))

    with profile_recorder.section("pipeline.static_fisher.shard_setup") if profile_recorder else _NULL_CONTEXT:
        token_batches = [batch[0] for batch in dataloader]
        nsamples_total = len(token_batches)
        world = dist_utils.get_world_size()
        rank = dist_utils.get_rank()
        if nsamples_total % world != 0:
            raise ValueError(
                f"static saliency: nsamples ({nsamples_total}) must be divisible by world_size ({world})."
            )
        if batch_size % world != 0:
            raise ValueError(
                f"static saliency: global_loss_bsz ({batch_size}) must be divisible by world_size ({world})."
            )
        local_batch_size = batch_size // world
        if use_grad_sample_limit:
            if num_samples_for_grad > nsamples_total:
                raise ValueError(
                    f"num_samples_for_grad ({num_samples_for_grad}) must be <= "
                    f"nsamples ({nsamples_total})."
                )
            if num_samples_for_grad % world != 0:
                raise ValueError(
                    f"num_samples_for_grad ({num_samples_for_grad}) must be divisible "
                    f"by world_size ({world})."
                )
            local_grad_samples = num_samples_for_grad // world
            if local_grad_samples <= 0:
                raise ValueError(
                    f"num_samples_for_grad ({num_samples_for_grad}) gives zero "
                    f"rank-local samples with world_size={world}."
                )
            if local_grad_samples % local_batch_size != 0:
                raise ValueError(
                    f"num_samples_for_grad // world ({local_grad_samples}) must be "
                    f"divisible by per-rank global_loss_bsz ({local_batch_size})."
                )
        else:
            local_grad_samples = None
        # refined_rkl sub-A alignment: samples are split into `num_A` contiguous
        # groups of size `samples_per_A = nsamples/num_A`, and each global batch
        # must fit within one group so the grad hook accumulates into exactly one
        # (layer, a) bucket. Enforce divisibility here rather than silently
        # dropping/merging samples across sub-A boundaries.
        refined_rkl_samples_per_A = 0
        if _collect_any_refined:
            if nsamples_total % refined_rkl_num_A != 0:
                raise ValueError(
                    f"refined_rkl: nsamples_total ({nsamples_total}) must be divisible by "
                    f"refined_rkl_num_A ({refined_rkl_num_A})."
                )
            refined_rkl_samples_per_A = nsamples_total // refined_rkl_num_A
            if refined_rkl_samples_per_A % batch_size != 0:
                raise ValueError(
                    f"refined_rkl: samples_per_A ({refined_rkl_samples_per_A}) must be divisible by "
                    f"global_loss_bsz ({batch_size}) so each batch lands in one sub-A bucket."
                )
        # Contiguous shard of sample ids; each rank only does forward/backward on
        # its own slice and keeps the collected saliency/fisher rank-local. These
        # tensors are later consumed directly by rank-local `add_batch` calls
        # (sample index alignment is preserved because `inps` is sharded the same
        # way) — no all-gather is needed.
        shard = dist_utils.shard_slice(nsamples_total, rank, world)
        local_batches = token_batches[shard]
        grad_stat_local_limit = (
            local_grad_samples if local_grad_samples is not None else len(local_batches)
        )
        loop_local_batches = local_batches
        if (
            local_grad_samples is not None
            and not capture_fp_final
            and not _collect_any_refined
        ):
            loop_local_batches = local_batches[:grad_stat_local_limit]
    with profile_recorder.section("pipeline.static_fisher.model_to_device") if profile_recorder else _NULL_CONTEXT:
        if not use_fsdp:
            # FSDP2 already placed the shards on each rank's GPU; don't try to
            # move the whole model to one device.
            model = model.to(dev)
        model.eval()
        # The precompute only reads **activation gradients** via hooks (saliency /
        # fisher = mean of grad².pow). Weight gradients are never consumed here, so
        # freeze all parameters to skip populating `param.grad` — saves a fp32 grad
        # buffer the size of the entire model (280 GB for Llama2-70B, 8 GB for 4B).
        # Restored in the finally clause below.
        prev_requires_grad = [(p, p.requires_grad) for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad_(False)
        # With all params frozen the embedding output has requires_grad=False, so
        # autograd wouldn't build any graph. Re-enter the graph at the first
        # transformer layer's input by flipping it to requires_grad=True via a
        # forward pre-hook.
        def _kick_off_grad_hook(module, inputs):
            if isinstance(inputs, tuple) and len(inputs) > 0 and torch.is_tensor(inputs[0]):
                inputs[0].requires_grad_(True)
        _kick_off_handle = analyzer.get_layers()[0].register_forward_pre_hook(_kick_off_grad_hook)

    def _flush_refined_rkl_sub_a(a_idx):
        """Solve A_{l, a_idx} for all layers and free the (l, a) C/H slots.
        Called when a batch transition signals that the sub-A has finished
        accumulating (or at the end of the backward pass). Writes into the
        pre-allocated `static_refined_A[l][a_idx]` slot. Also solves the
        diagonal-variant slot when `collect_refined_diag_rkl=True`.

        Performance: C/H accumulators live on CPU fp32, but the Cholesky /
        element-wise solve runs on `dev` (GPU) in fp32 — CPU fp64 Cholesky
        was the bottleneck at H≥4096 (minutes per transition on 7B/70B).
        fp64-on-CPU is retained only as a fallback when fp32-GPU Cholesky
        fails numerically."""
        with profile_recorder.section("pipeline.static_fisher.flush.total") if profile_recorder else _NULL_CONTEXT:
            with torch.no_grad():
                for layer_idx in range(len(layers)):
                    if refined_C is not None:
                        slot_C = refined_C[layer_idx][a_idx]
                        if slot_C is not None:
                            H_dim = slot_C.shape[0]
                            # Ship fp32 accumulators to GPU. non_blocking has no
                            # effect on non-pinned CPU tensors but doesn't hurt.
                            with profile_recorder.section("pipeline.static_fisher.flush.full.h2d") if profile_recorder else _NULL_CONTEXT:
                                C_gpu = slot_C.to(dev, dtype=torch.float32, non_blocking=True)
                                H_gpu = refined_H[layer_idx][a_idx].to(dev, dtype=torch.float32, non_blocking=True)
                                trace = (torch.diagonal(H_gpu).sum() / H_dim).item()
                            if not (trace > 0):
                                logging.warning(
                                    "refined_rkl layer %d a=%d: trace(H)/H=%.3e <= 0, setting A to zero.",
                                    layer_idx, a_idx, trace,
                                )
                                A_store = torch.zeros(
                                    H_dim, H_dim, dtype=refined_rkl_store_dtype,
                                )
                            else:
                                damp = refined_rkl_damp * trace
                                # In-place damp on the diagonal — saves a full H×H
                                # eye allocation + add, which at H=8192 is 256 MB.
                                H_gpu.diagonal().add_(damp)
                                try:
                                    with profile_recorder.section("pipeline.static_fisher.flush.full.cholesky_fp32") if profile_recorder else _NULL_CONTEXT:
                                        L = torch.linalg.cholesky(H_gpu)
                                        A_gpu = torch.cholesky_solve(C_gpu, L)
                                        A_store = A_gpu.to(dtype=refined_rkl_store_dtype).cpu().contiguous()
                                        del L, A_gpu
                                except torch._C._LinAlgError:
                                    logging.warning(
                                        "refined_rkl layer %d a=%d: fp32 GPU Cholesky failed with "
                                        "damp=%.3e; retrying in fp64 on CPU.",
                                        layer_idx, a_idx, damp,
                                    )
                                    with profile_recorder.section("pipeline.static_fisher.flush.full.cholesky_fp64_fallback") if profile_recorder else _NULL_CONTEXT:
                                        # Rebuild H_damped from the pristine CPU fp32
                                        # accumulator (H_gpu was modified in place).
                                        H64 = refined_H[layer_idx][a_idx].to(torch.float64)
                                        C64 = slot_C.to(torch.float64)
                                        H64.diagonal().add_(damp)
                                        try:
                                            L64 = torch.linalg.cholesky(H64)
                                            A_cpu = torch.cholesky_solve(C64, L64)
                                        except torch._C._LinAlgError:
                                            logging.warning(
                                                "refined_rkl layer %d a=%d: fp64 Cholesky also failed; "
                                                "falling back to direct inverse.", layer_idx, a_idx,
                                            )
                                            A_cpu = torch.linalg.inv(H64) @ C64
                                        A_store = A_cpu.to(refined_rkl_store_dtype).contiguous()
                            static_refined_A[layer_idx][a_idx] = A_store
                            refined_fit_counter[0] += 1
                            refined_fit_shape[0] = tuple(A_store.shape)
                            del C_gpu, H_gpu
                            # Free the fp32 accumulators for this slot — this is the whole
                            # point of streaming: peak CPU RAM drops from L × num_A × 2 × H²
                            # down to L × 2 × H² + the stored (bf16) A stack.
                            refined_C[layer_idx][a_idx] = None
                            refined_H[layer_idx][a_idx] = None

                    if refined_diag_C is not None:
                        slot_Cd = refined_diag_C[layer_idx][a_idx]
                        if slot_Cd is not None:
                            with profile_recorder.section("pipeline.static_fisher.flush.diag.solve") if profile_recorder else _NULL_CONTEXT:
                                # GPU element-wise solve. Cheap compute but (seq, H)
                                # tensors are memory-bandwidth heavy on CPU (64 MB each
                                # at H=8192, seq=2048) — the PCIe copy is still faster
                                # than doing it on CPU for large H.
                                Cd = slot_Cd.to(dev, dtype=torch.float32, non_blocking=True)
                                Hd = refined_diag_H[layer_idx][a_idx].to(dev, dtype=torch.float32, non_blocking=True)
                                # Trace-normalised damp: λ = damp · mean(dy²) over all (t, i)
                                # in this sub-A. Matches the full-version's damp scale so
                                # the same `refined_rkl_damp` hyper-parameter applies.
                                mean_dy_sq = Hd.mean().item()
                                if not (mean_dy_sq > 0):
                                    logging.warning(
                                        "refined_diag_rkl layer %d a=%d: mean(dy²)=%.3e <= 0, "
                                        "setting diag-A to zero.", layer_idx, a_idx, mean_dy_sq,
                                    )
                                    Ad_store = torch.zeros(
                                        *Cd.shape, dtype=refined_diag_rkl_store_dtype,
                                    )
                                else:
                                    damp_diag = refined_rkl_damp * mean_dy_sq
                                    Ad_gpu = Cd / (Hd + damp_diag)
                                    Ad_store = Ad_gpu.to(dtype=refined_diag_rkl_store_dtype).cpu().contiguous()
                                    del Ad_gpu
                            static_refined_diag_A[layer_idx][a_idx] = Ad_store
                            del Cd, Hd
                            refined_diag_C[layer_idx][a_idx] = None
                            refined_diag_H[layer_idx][a_idx] = None

    try:
        with torch.enable_grad():
            refined_prev_a = None
            for local_start in tqdm(
                range(0, len(loop_local_batches), local_batch_size),
                ncols=120,
                desc="Static E2E Saliency/Fisher",
                position=1,
                leave=False,
            ):
                with profile_recorder.section("pipeline.static_fisher.batch.total") if profile_recorder else _NULL_CONTEXT:
                    global_start = shard.start + local_start
                    if _collect_any_refined and refined_rkl_samples_per_A > 0:
                        # Route this batch's contributions to the correct sub-A
                        # bucket. We required `samples_per_A % batch_size == 0` so
                        # the whole batch has a single a_idx.
                        cur_a = global_start // refined_rkl_samples_per_A
                        if refined_prev_a is not None and cur_a != refined_prev_a:
                            # All batches belonging to `refined_prev_a` have been
                            # consumed — solve and free before continuing so peak
                            # CPU RAM is bounded by one sub-A's accumulators.
                            _flush_refined_rkl_sub_a(refined_prev_a)
                        refined_current_a_idx["a"] = cur_a
                        refined_prev_a = cur_a
                    with profile_recorder.section("pipeline.static_fisher.batch.prepare_inputs") if profile_recorder else _NULL_CONTEXT:
                        input_ids = torch.cat(loop_local_batches[local_start:local_start + local_batch_size], dim=0).to(dev)
                    with profile_recorder.section("pipeline.static_fisher.batch.forward") if profile_recorder else _NULL_CONTEXT:
                        outputs = model(input_ids=input_ids)
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                        teacher_logits = logits.detach()
                        student_logits = logits
                    if grad_hessian_topk > 0:
                        with profile_recorder.section("pipeline.static_fisher.batch.topk_slice") if profile_recorder else _NULL_CONTEXT:
                            teacher_logits, indices = teacher_logits.topk(
                                grad_hessian_topk,
                                dim=-1,
                                sorted=False,
                            )
                            student_logits = student_logits.gather(-1, indices)
                    with profile_recorder.section("pipeline.static_fisher.batch.label_sample") if profile_recorder else _NULL_CONTEXT:
                        # Per-sample deterministic label sampling: each sample's labels
                        # only depend on its global sample id, making the draw
                        # invariant to batching (1-GPU 16-per-batch vs 2-GPU 8-per-batch
                        # both produce the same labels for the same global sample id).
                        _batch_bsz = teacher_logits.shape[0]
                        _global_indices = [global_start + _i for _i in range(_batch_bsz)]
                        collect_grad_stats_this_batch = (
                            local_start + _batch_bsz <= grad_stat_local_limit
                        )
                        if (
                            not collect_grad_stats_this_batch
                            and not capture_fp_final
                            and not _collect_any_refined
                        ):
                            raise RuntimeError(
                                "Static precompute reached a batch beyond num_samples_for_grad "
                                "without any forward-only payload to collect."
                            )
                        labels = _deterministic_categorical_labels(
                            teacher_logits,
                            _global_indices,
                            base_seed=0,  # global static saliency uses its own namespace
                        )
                    with profile_recorder.section("pipeline.static_fisher.batch.nll_build") if profile_recorder else _NULL_CONTEXT:
                        sl_for_nll = student_logits
                        labels_for_nll = labels
                        if sink_size > 0 and sl_for_nll.shape[1] > sink_size:
                            sl_for_nll = sl_for_nll[:, sink_size:]
                            labels_for_nll = labels_for_nll[:, sink_size:]
                        if use_rademacher_stats:
                            per_token_nll = F.cross_entropy(
                                sl_for_nll.reshape(-1, sl_for_nll.size(-1)),
                                labels_for_nll.reshape(-1),
                                reduction="none",
                            ).view_as(labels_for_nll)
                            loss = None
                            loss_for_backward = None
                        else:
                            per_token_nll = None
                            loss = F.cross_entropy(
                                sl_for_nll.reshape(-1, sl_for_nll.size(-1)),
                                labels_for_nll.reshape(-1),
                                reduction="sum",
                            )
                            loss_for_backward = loss * _E2E_PRECOMPUTE_LOSS_GRAD_SCALE
                    with profile_recorder.section("pipeline.static_fisher.batch.backward") if profile_recorder else _NULL_CONTEXT:
                        if not collect_grad_stats_this_batch:
                            try:
                                if _collect_any_refined:
                                    current_backward_role["saliency_fisher"] = False
                                    current_backward_role["refined"] = True
                                    refined_dy_buffer.clear()
                                    model.zero_grad()
                                    refined_loss_for_backward = (
                                        per_token_nll.sum() * _E2E_PRECOMPUTE_LOSS_GRAD_SCALE
                                        if per_token_nll is not None
                                        else loss_for_backward
                                    )
                                    refined_loss_for_backward.backward()
                            finally:
                                current_backward_role["saliency_fisher"] = True
                                current_backward_role["refined"] = True
                        elif use_rademacher_stats:
                            if collect_saliency:
                                current_saliency_accum["data"] = [
                                    {name: None for name in md.keys()}
                                    for md in module_dicts
                                ]
                            try:
                                current_backward_role["saliency_fisher"] = True
                                current_backward_role["refined"] = False
                                for rep_idx in range(stat_repeats):
                                    refined_dy_buffer.clear()
                                    model.zero_grad()
                                    signs = _deterministic_rademacher_signs(
                                        per_token_nll.shape,
                                        _global_indices,
                                        base_seed=int(rademacher_seed) + rep_idx,
                                        device=per_token_nll.device,
                                        dtype=per_token_nll.dtype,
                                    )
                                    signed_loss = (per_token_nll * signs).sum()
                                    signed_loss_for_backward = signed_loss * _E2E_PRECOMPUTE_LOSS_GRAD_SCALE
                                    signed_loss_for_backward.backward(
                                        retain_graph=(
                                            rep_idx + 1 < stat_repeats
                                            or _collect_any_refined
                                        )
                                    )
                                    del signs, signed_loss, signed_loss_for_backward
                                if collect_saliency:
                                    accum = current_saliency_accum["data"]
                                    for _layer_idx, module_dict in enumerate(module_dicts):
                                        for _module_name in module_dict.keys():
                                            sal_chunk = accum[_layer_idx][_module_name]
                                            if sal_chunk is None:
                                                raise RuntimeError(
                                                    f"Rademacher saliency was not collected for "
                                                    f"layer={_layer_idx} module={_module_name}."
                                            )
                                            saliency_data[_layer_idx][_module_name].append(sal_chunk)
                                if _collect_any_refined:
                                    current_backward_role["saliency_fisher"] = False
                                    current_backward_role["refined"] = True
                                    refined_dy_buffer.clear()
                                    model.zero_grad()
                                    loss_for_backward = per_token_nll.sum() * _E2E_PRECOMPUTE_LOSS_GRAD_SCALE
                                    loss_for_backward.backward()
                            finally:
                                current_backward_role["saliency_fisher"] = True
                                current_backward_role["refined"] = True
                                current_saliency_accum["data"] = None
                        else:
                            current_backward_role["saliency_fisher"] = True
                            current_backward_role["refined"] = True
                            model.zero_grad()
                            loss_for_backward.backward()
                    del outputs, logits, teacher_logits, student_logits, labels, loss, loss_for_backward, input_ids
                    if per_token_nll is not None:
                        del per_token_nll
            # Flush the last sub-A's accumulators. The streaming flush inside
            # the loop only fires on transitions, so the final one needs to be
            # drained explicitly.
            if _collect_any_refined and refined_prev_a is not None:
                with profile_recorder.section("pipeline.static_fisher.final_flush") if profile_recorder else _NULL_CONTEXT:
                    _flush_refined_rkl_sub_a(refined_prev_a)

            # === Dynamic saliency: randomized SVD (sketch + 2nd-pass) ===
            # Replaces the explicit (d_out, d_out) G^T G EVD which OOMs at ≥13B.
            # Pass 1 (already done above by the merged hook): accumulated
            #   Y_sketch = G^T (G Ω) ∈ ℝ^(d_out × K) per (layer, module).
            # Now:
            #   1. all-reduce Y_sketch across DP ranks (Ω is identical across
            #      ranks by deterministic seeding, so the per-rank Y's are
            #      sketches of the same column space — sum is consistent).
            #   2. QR(Y_sketch) → Q ∈ ℝ^(d_out × K); same span as the top-K
            #      right singular subspace of G with high probability.
            #   3. Pass 2 (new backward): for each batch, compute B_local =
            #      G_batch @ Q ∈ ℝ^(B·T × K) and accumulate
            #      B_cov += B_local^T B_local on GPU. Also stream B_local rows
            #      into a preallocated GPU buffer for later assembly into
            #      U_Sigma. K K matrices are tiny (≪1 MB), B_full buffers are
            #      O(N_local·T·K) bf16 — same order as the previous U_Sigma
            #      buffer, just with K = 2R columns.
            #   4. eigh(B_cov) → V_hat (K-dim), truncate to top-R.
            #      V = Q @ V_hat ∈ ℝ^(d_out × R), U_Sigma = B_full @ V_hat.
            # Must happen BEFORE the outer finally because:
            #   (a) the finally pushes the model back to CPU (non-FSDP path);
            #   (b) we need to re-register hooks while _kick_off pre-hook is
            #       still attached.
            if collect_dynsal:
                with profile_recorder.section("pipeline.static_fisher.dynsal_qr") if profile_recorder else _NULL_CONTEXT:
                    # Pass-1 finalisation: all-reduce Y_sketch, QR → Q.
                    first_layer_module = next(iter(module_dicts[0].keys()))
                    local_tokens_N = sum(
                        int(chunk.shape[0] * chunk.shape[1])
                        for chunk in saliency_data[0][first_layer_module]
                    )
                    N_global = int(dist_utils.allreduce_sum_scalar(local_tokens_N))
                    static_dynsal_by_layer = []
                    dynsal_Q_dev = [{} for _ in module_dicts]
                    dynsal_H_out = [{} for _ in module_dicts]
                    for layer_idx, module_dict in enumerate(module_dicts):
                        layer_dynsal = {}
                        for module_name in module_dict.keys():
                            sketch = dynsal_sketch_data[layer_idx][module_name]
                            if sketch is None:
                                raise ValueError(
                                    f"Failed to collect Y_sketch for dynsal: "
                                    f"layer={layer_idx} module={module_name}."
                                )
                            dist_utils.allreduce_sum_(sketch)
                            # QR — Q has same shape as sketch, orthonormal columns.
                            # `mode='reduced'` is the default; sketch is (d_out, K)
                            # with d_out >= K, so Q is (d_out, K).
                            Q, _R_qr = torch.linalg.qr(sketch, mode="reduced")
                            del _R_qr
                            H_out = Q.shape[0]
                            K_eff = Q.shape[1]
                            # Stash entry shell — V / U_Sigma / Sigma_sq filled
                            # in after Pass 2's eigh.
                            layer_dynsal[module_name] = {
                                "H_out": int(H_out),
                                # R_eff finalised after eigh + threshold below.
                            }
                            dynsal_Q_dev[layer_idx][module_name] = Q.contiguous()
                            dynsal_H_out[layer_idx][module_name] = int(H_out)
                            # Drop Ω + the original sketch buffer; we no longer
                            # need them once Q is in hand.
                            dynsal_omega_data[layer_idx][module_name] = None
                            dynsal_sketch_data[layer_idx][module_name] = None
                            del sketch
                        static_dynsal_by_layer.append(layer_dynsal)
                    dynsal_omega_data = None
                    dynsal_sketch_data = None
                    memory_utils.cleanup_memory()
                    logging.info(
                        "dynsal: QR done for %d layers × %d modules; K=%d, "
                        "(R will be picked from eigh after Pass 2). N_global=%d",
                        len(layers), len(module_dicts[0]), dynsal_K, N_global,
                    )

                with profile_recorder.section("pipeline.static_fisher.dynsal_second_pass") if profile_recorder else _NULL_CONTEXT:
                    # Pass-2 backward: replace first-pass hooks with U-collection
                    # hooks that accumulate B_cov AND stream B_local into a
                    # preallocated GPU buffer.
                    for h in handles:
                        h.remove()
                    handles.clear()

                    dynsal_b_buf_dev = {}      # (N_local·T, K) bf16 per (l, m)
                    dynsal_b_cov = {}           # (K, K) fp32 per (l, m)
                    dynsal_write_cursor = {}
                    token_total_local = sum(
                        int(chunk.shape[0] * chunk.shape[1])
                        for chunk in saliency_data[0][next(iter(module_dicts[0].keys()))]
                    )
                    for layer_idx, md in enumerate(module_dicts):
                        dynsal_b_buf_dev[layer_idx] = {}
                        dynsal_b_cov[layer_idx] = {}
                        dynsal_write_cursor[layer_idx] = {name: 0 for name in md.keys()}
                        for name in md.keys():
                            K_for_module = dynsal_Q_dev[layer_idx][name].shape[1]
                            dynsal_b_buf_dev[layer_idx][name] = torch.empty(
                                token_total_local, K_for_module,
                                dtype=torch.bfloat16, device=dev,
                            )
                            dynsal_b_cov[layer_idx][name] = torch.zeros(
                                K_for_module, K_for_module,
                                dtype=torch.float32, device=dev,
                            )

                    def make_u_hook(layer_idx, module_name):
                        """Pass-2 hook: B_local = G_batch @ Q, accumulate B_cov in
                        place, and stream B_local rows into the preallocated
                        (N_local·T, K) bf16 GPU buffer for later U_Sigma = B_full
                        @ V_hat after eigh."""
                        def forward_hook(module, inp, out):
                            out_tensor = out[0] if isinstance(out, (tuple, list)) else out

                            def grad_hook(grad):
                                # Pass-1 saliency hook already drops the sink
                                # slice when sink_size>0, so saliency_data /
                                # local_tokens_N / the (N_local·T_eff, K) buffer
                                # are sized to T_eff = T - sink. Mirror the slice
                                # here so rows in B_local match the buffer.
                                g = grad
                                if sink_size > 0 and g.dim() >= 2 and g.shape[1] > sink_size:
                                    g = g[:, sink_size:]
                                grad_flat = (
                                    (g.detach().float() / _E2E_PRECOMPUTE_LOSS_GRAD_SCALE)
                                    .reshape(-1, g.shape[-1])
                                )
                                Q_m = dynsal_Q_dev[layer_idx][module_name]  # (d_out, K) fp32
                                B_local = grad_flat @ Q_m  # (B·T, K) fp32
                                # In-place accumulate B_cov += B^T B (small KxK).
                                dynsal_b_cov[layer_idx][module_name].addmm_(
                                    B_local.t(), B_local
                                )
                                # Stream rows to the preallocated bf16 buffer.
                                buf = dynsal_b_buf_dev[layer_idx][module_name]
                                n_tokens_batch = B_local.shape[0]
                                cursor = dynsal_write_cursor[layer_idx][module_name]
                                buf[cursor:cursor + n_tokens_batch].copy_(
                                    B_local.to(torch.bfloat16), non_blocking=True
                                )
                                dynsal_write_cursor[layer_idx][module_name] = cursor + n_tokens_batch
                                del B_local, grad_flat
                            out_tensor.register_hook(grad_hook)
                        return forward_hook

                    for layer_idx, (layer, module_dict) in enumerate(zip(layers, module_dicts)):
                        for module_name, module in module_dict.items():
                            handles.append(module.register_forward_hook(make_u_hook(layer_idx, module_name)))

                    with torch.enable_grad():
                        for local_start in tqdm(
                            range(0, len(local_batches), local_batch_size),
                            ncols=120,
                            desc="Dynsal 2nd pass (U)",
                            position=1,
                            leave=False,
                        ):
                            with profile_recorder.section("pipeline.static_fisher.dynsal_batch") if profile_recorder else _NULL_CONTEXT:
                                global_start = shard.start + local_start
                                input_ids = torch.cat(local_batches[local_start:local_start + local_batch_size], dim=0).to(dev)
                                outputs = model(input_ids=input_ids)
                                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                                teacher_logits = logits.detach()
                                student_logits = logits
                                if grad_hessian_topk > 0:
                                    teacher_logits, indices = teacher_logits.topk(
                                        grad_hessian_topk, dim=-1, sorted=False,
                                    )
                                    student_logits = student_logits.gather(-1, indices)
                                _batch_bsz = teacher_logits.shape[0]
                                _global_indices = [global_start + _i for _i in range(_batch_bsz)]
                                labels = _deterministic_categorical_labels(
                                    teacher_logits, _global_indices, base_seed=0,
                                )
                                sl_for_nll = student_logits
                                labels_for_nll = labels
                                if sink_size > 0 and sl_for_nll.shape[1] > sink_size:
                                    sl_for_nll = sl_for_nll[:, sink_size:]
                                    labels_for_nll = labels_for_nll[:, sink_size:]
                                loss = F.cross_entropy(
                                    sl_for_nll.reshape(-1, sl_for_nll.size(-1)),
                                    labels_for_nll.reshape(-1), reduction="sum",
                                )
                                loss_for_backward = loss * _E2E_PRECOMPUTE_LOSS_GRAD_SCALE
                                loss_for_backward.backward()
                                del outputs, logits, teacher_logits, student_logits, labels, loss, loss_for_backward, input_ids

                    # Pass-2 finalisation: per (layer, module) all-reduce B_cov,
                    # eigh, top-R truncation, build V = Q @ V_hat, U_Sigma =
                    # B_full @ V_hat. Free Q, B_full, B_cov immediately after.
                    R_eff_total = 0
                    for layer_idx, layer_d in enumerate(static_dynsal_by_layer):
                        for name, entry in layer_d.items():
                            written = dynsal_write_cursor[layer_idx][name]
                            B_full_dev = dynsal_b_buf_dev[layer_idx][name]
                            if written != B_full_dev.shape[0]:
                                raise RuntimeError(
                                    f"dynsal 2nd pass: expected {B_full_dev.shape[0]} rows for "
                                    f"layer={layer_idx} module={name}, got {written}."
                                )
                            B_cov_m = dynsal_b_cov[layer_idx][name]
                            dist_utils.allreduce_sum_(B_cov_m)
                            B_cov_m = 0.5 * (B_cov_m + B_cov_m.t())  # symmetrise
                            eigvals, eigvecs = torch.linalg.eigh(B_cov_m)
                            K_for_module = B_cov_m.shape[0]
                            R_req = min(int(dynsal_rank), K_for_module)
                            top_eigvals = eigvals[-R_req:].flip(0).clamp_min(0.0)
                            top_eigvecs = eigvecs[:, -R_req:].flip(1)  # (K, R_req)
                            max_eig = top_eigvals[0].clamp_min(1e-30)
                            keep_mask = top_eigvals >= max_eig * dynsal_evd_thresh
                            R_eff = int(keep_mask.sum().item())
                            if R_eff < 1:
                                R_eff = 1
                            Sigma_sq = top_eigvals[:R_eff].contiguous()
                            V_hat = top_eigvecs[:, :R_eff].contiguous()  # (K, R_eff)
                            # V = Q @ V_hat ∈ (d_out, R_eff) fp32 → bf16 CPU.
                            Q_m = dynsal_Q_dev[layer_idx][name]
                            V = Q_m @ V_hat  # (d_out, R_eff) fp32 dev
                            # U_Sigma = B_full @ V_hat ∈ (N_local·T, R_eff).
                            # bf16 matmul (B_full bf16, cast V_hat to bf16).
                            V_hat_bf = V_hat.to(torch.bfloat16)
                            U_Sigma_dev = B_full_dev @ V_hat_bf  # (N_local·T, R_eff) bf16
                            entry["V"] = V.to(torch.bfloat16).cpu()
                            entry["Sigma_sq"] = Sigma_sq.cpu()
                            entry["R_eff"] = int(R_eff)
                            entry["U_Sigma"] = U_Sigma_dev.cpu()
                            R_eff_total += R_eff
                            # Free per-module GPU footprint immediately.
                            dynsal_b_buf_dev[layer_idx][name] = None
                            dynsal_b_cov[layer_idx][name] = None
                            dynsal_Q_dev[layer_idx][name] = None
                            del B_full_dev, B_cov_m, eigvals, eigvecs, top_eigvals, top_eigvecs
                            del Q_m, V, V_hat, V_hat_bf, U_Sigma_dev, Sigma_sq
                    dynsal_b_buf_dev = None
                    dynsal_b_cov = None
                    dynsal_write_cursor = None
                    dynsal_Q_dev = None
                    dynsal_H_out = None
                    memory_utils.cleanup_memory()
                    static_dynsal = {
                        "by_layer": static_dynsal_by_layer,
                        "N_global": N_global,
                        "dynsal_rank": int(dynsal_rank),
                        "dynsal_evd_thresh": float(dynsal_evd_thresh),
                        "dynsal_K": int(dynsal_K),
                    }
                    logging.info(
                        "dynsal: rsvd done for %d layers × %d modules; K=%d, "
                        "EVD threshold=%.0e, mean R_eff=%.1f, N_global=%d",
                        len(layers), len(module_dicts[0]), dynsal_K, dynsal_evd_thresh,
                        float(R_eff_total) / (len(layers) * len(module_dicts[0])),
                        N_global,
                    )

                with profile_recorder.section("pipeline.static_fisher.dynsal_pack_weights") if profile_recorder else _NULL_CONTEXT:
                    # Pack C_k and W_cross once up-front (fp32 on CPU). Each
                    # module's packs are G × R × R; trivial in size.
                    G_groups = int(saliency_num_groups)
                    for layer_d in static_dynsal_by_layer:
                        for entry in layer_d.values():
                            V_dev = entry["V"].to(dev, dtype=torch.float32)   # (H_out, R_eff)
                            Sigma_sq_dev = entry["Sigma_sq"].to(dev)          # (R_eff,) fp32
                            R_eff = entry["R_eff"]
                            # Static saliency is a group squared norm
                            # (channel sum), so its dynamic low-rank
                            # correction must use the same sum scale.
                            C_packed = grouped_channel_gram(
                                V_dev, G_groups
                            )
                            if C_packed.shape != (
                                G_groups, R_eff, R_eff
                            ):
                                raise RuntimeError(
                                    "unexpected grouped dynamic-saliency "
                                    f"Gram shape {tuple(C_packed.shape)}"
                                )
                            W_cross_packed = C_packed * Sigma_sq_dev.view(1, 1, R_eff)
                            entry["C_packed"] = C_packed.cpu()
                            entry["W_cross_packed"] = W_cross_packed.cpu()
                            del V_dev, Sigma_sq_dev, C_packed, W_cross_packed
                    memory_utils.cleanup_memory()
    finally:
        with profile_recorder.section("pipeline.static_fisher.teardown") if profile_recorder else _NULL_CONTEXT:
            for handle in handles:
                handle.remove()
            _kick_off_handle.remove()
            # Restore `requires_grad` so downstream phases (quantization, eval) see
            # the model in its original training-like state.
            for p, flag in prev_requires_grad:
                p.requires_grad_(flag)
            model.zero_grad()
            if use_fsdp:
                # In-process FSDP unwrap has been removed — it was too fragile in
                # PyTorch 2.9 FSDP2. When `use_fsdp=True`, the sweep/driver MUST
                # set `--exit_after_precompute` and `--static_cache_path`; the
                # parent stage writes the saliency/fisher cache to disk and exits,
                # then the caller re-runs without `--fsdp_precompute` to do the
                # quantisation from the cached stats. See README two-stage
                # workflow. We leave the model in its FSDP-wrapped state here
                # (don't `.cpu()` because FSDP's `_apply` override would crash).
                pass
            else:
                # Non-FSDP path: push the model back to CPU so the per-layer quant
                # phase can reload layers on demand.
                model.cpu()
            memory_utils.cleanup_memory()

    with profile_recorder.section("pipeline.static_fisher.aggregate") if profile_recorder else _NULL_CONTEXT:
        static_saliency = []
        static_fisher = []
        static_legacy_fisher_diag = []
        for layer_idx, module_dict in enumerate(module_dicts):
            layer_saliency = {}
            if collect_saliency:
                for module_name in module_dict.keys():
                    if not saliency_data[layer_idx][module_name]:
                        raise ValueError(
                            f"Failed to collect static end-to-end saliency for layer={layer_idx} module={module_name}."
                        )
                    # Rank-local shard of shape (n_local, T, G). Not gathered.
                    layer_saliency[module_name] = clip_global_percentile_(
                        torch.cat(
                            saliency_data[layer_idx][module_name], dim=0
                        ),
                        saliency_clip_percentile,
                    )
                    # Free the per-batch chunks right after concat. Otherwise both
                    # the list (~14 GB for 8B across all layers) AND the catted
                    # tensor (~14 GB) are alive simultaneously until function exit,
                    # which adds a transient 14 GB CPU peak right at the boundary
                    # where the quant phase also starts allocating inps/fp_inps.
                    saliency_data[layer_idx][module_name] = None
            static_saliency.append(layer_saliency)
            if collect_fisher:
                want_fisher_this_layer = (
                    fisher_layer_ids is None or layer_idx in fisher_layer_ids
                )
                if want_fisher_this_layer:
                    if fisher_data[layer_idx] is None:
                        raise ValueError(f"Failed to collect static end-to-end Fisher for layer={layer_idx}.")
                    # fisher_data[layer_idx] is now accumulated in GPU fp32 (see
                    # make_layer_hook). In-place all-reduce then ship to CPU bf16
                    # exactly once per layer.
                    fisher_sum = fisher_data[layer_idx]
                    if fisher_sum.device.type != "cuda":
                        # Fall back to the old CPU→GPU path if it ever holds CPU
                        # (e.g. legacy caller or serialized dict).
                        fisher_sum = fisher_sum.to(dev)
                    dist_utils.allreduce_sum_(fisher_sum)
                    # Token count: derive from the already-catted saliency tensor
                    # rather than from the per-batch chunks (which we freed above
                    # to cut a 14 GB transient CPU peak). Same value either way.
                    if collect_saliency:
                        _any_cat = next(iter(layer_saliency.values()))
                        local_tokens = int(_any_cat.shape[0] * _any_cat.shape[1])
                    else:
                        token_batches = [batch[0] for batch in dataloader]
                        eff_seq_len = max(1, token_batches[0].shape[1] - max(0, sink_size))
                        local_tokens = int(grad_stat_local_limit * eff_seq_len)
                    total_tokens = dist_utils.allreduce_sum_scalar(local_tokens)
                    static_fisher.append((fisher_sum / float(total_tokens)).to(torch.bfloat16).cpu())
                    # Release the per-layer GPU fp32 Fisher immediately. For
                    # Llama-7B: 32 × 64 MB = 2 GB ~ freed as we go.
                    fisher_data[layer_idx] = None
                    del fisher_sum
                else:
                    # Caller opted out of fisher for this layer (e.g. refined_mix
                    # back half uses refined_residual_kl, which doesn't need fisher).
                    static_fisher.append(None)
            else:
                static_fisher.append(None)
            if collect_legacy_fisher_diag:
                want_legacy_this_layer = (
                    fisher_layer_ids is None or layer_idx in fisher_layer_ids
                )
                if want_legacy_this_layer:
                    chunks = legacy_fisher_diag_data[layer_idx]
                    if not chunks:
                        raise ValueError(
                            f"Failed to collect static per-token legacy Fisher diag "
                            f"for layer={layer_idx}."
                        )
                    static_legacy_fisher_diag.append(torch.cat(chunks, dim=0).cpu())
                    legacy_fisher_diag_data[layer_idx] = None
                else:
                    static_legacy_fisher_diag.append(None)
            else:
                static_legacy_fisher_diag.append(None)

    # refined_residual_kl A matrices have already been streamed into
    # `static_refined_A[l][a]` by `_flush_refined_rkl_sub_a` as each sub-A's
    # batches finished accumulating. Here we just emit the summary log.
    if collect_refined_rkl:
        total_slots = len(layers) * refined_rkl_num_A
        logging.info(
            "refined_rkl: fit %d/%d (layer, sub-A) matrices "
            "(A shape %s, num_A=%d, store_dtype=%s, damp=%.3g)",
            refined_fit_counter[0], total_slots, refined_fit_shape[0],
            refined_rkl_num_A,
            str(refined_rkl_store_dtype).replace("torch.", ""),
            refined_rkl_damp,
        )
    if collect_refined_diag_rkl:
        # Summary shape: pick first populated slot for reporting.
        _diag_shape = None
        for l_idx in range(len(layers)):
            for a_idx in range(refined_rkl_num_A):
                slot = static_refined_diag_A[l_idx][a_idx]
                if slot is not None:
                    _diag_shape = tuple(slot.shape)
                    break
            if _diag_shape is not None:
                break
        logging.info(
            "refined_diag_rkl: fit per-(token, channel) diag J "
            "(shape=%s, num_A=%d, store_dtype=%s, damp=%.3g)",
            _diag_shape, refined_rkl_num_A,
            str(refined_diag_rkl_store_dtype).replace("torch.", ""),
            refined_rkl_damp,
        )

    # Concatenate the rank-local fp_inps_final shards if captured. Shape:
    # (n_local, T, H) in fp_final_store_dtype on CPU. Consumer is free to
    # move to GPU / cast to fp_inps dtype.
    fp_inps_final = None
    if capture_fp_final:
        if not fp_final_cache:
            raise RuntimeError(
                "capture_fp_final=True but no last-layer outputs were recorded. "
                "Make sure the forward pass ran at least once."
            )
        fp_inps_final = torch.cat(fp_final_cache, dim=0)
        fp_final_cache.clear()

    return (
        static_saliency,
        static_fisher,
        static_legacy_fisher_diag,
        static_refined_A,
        static_refined_diag_A,
        fp_inps_final,
        static_dynsal,
    )


def _pick_refined_A_for_batch(
    *,
    refined_A_list,
    samples_per_A,
    loss_type,
    batch_local_start,
    batch_size_local,
    dp_rank,
    dp_shard_size,
    dev,
):
    """Pick the refined_residual_kl A matrix(es) for a given contiguous batch.

    Returns either a 2D (H, H) tensor — when num_A==1 or all samples share one
    sub-A — or a 3D (B, H, H) stacked tensor with A[a_idx_of_sample_b] per row.
    `compute_refresh_loss` routes 3D input through bmm so batches may freely
    cross sub-A boundaries.
    """
    # Fast-exit for non-refined losses or single-A lists.
    if refined_A_list is None or len(refined_A_list) == 0:
        return None
    if loss_type != "refined_residual_kl" or samples_per_A <= 0 or len(refined_A_list) == 1:
        return refined_A_list[0]

    a_per_sample = torch.tensor(
        [(dp_rank * dp_shard_size + batch_local_start + k) // samples_per_A
         for k in range(batch_size_local)],
        dtype=torch.long,
        device=dev,
    )
    unique_a = torch.unique(a_per_sample)
    ref_slot = next((s for s in refined_A_list if s is not None), None)
    if ref_slot is None:
        return None
    if unique_a.numel() == 1:
        return refined_A_list[int(unique_a.item())]
    # Build a (B, H, H) stack; None slots get zeros. Common in practice: only
    # 2-3 distinct a_idx per batch, so the whole stack is cheap to materialise.
    return torch.stack(
        [
            refined_A_list[int(a)] if refined_A_list[int(a)] is not None else torch.zeros_like(ref_slot)
            for a in a_per_sample.tolist()
        ],
        dim=0,
    )


@torch.no_grad()
def refresh_dynamic_saliency(
    *,
    dyn_entry,
    static_saliency,
    N_global,
    P_delta,
    num_groups,
    dev,
    static_saliency_scale=1.0,
):
    """Compute S_new at a module-group boundary.

    Formulas from saliency_dynamic_update_design.md §4:
        Δs_cross[t, k] = (2/N_global)  · sum( U_Sigma ⊙ (P @ W_cross[k].T), dim=-1)
        S_new          = clamp_min(S_old_exact + Δs_cross, 0)

    where P = ΔY · V, ΔY = current_Y − fp_Y (cumulative output drift). P is
    precomputed by `_collect_module_output_projections` in STREAMING fashion
    to avoid ever materialising the full (N_local, T, H_out) ΔY tensor on CPU
    — for Llama-3-8B up/gate that tensor alone is ~120 GB.

    Args:
      dyn_entry:       per-module dict with keys V / Sigma_sq / U_Sigma /
                       C_packed / W_cross_packed / H_out / R_eff.
      static_saliency: (N_local, T, G) tensor of S_old_exact on CPU.
      N_global:        global token count (int, used in 2/N and 1/N² factors).
      P_delta:         (N_local, T, R_eff) bf16 CPU — precomputed ΔY @ V.
      num_groups:      G (saliency output-channel group count, usually 4).
      dev:             target device for the matmuls.
      static_saliency_scale:
                       Positive scalar applied to the dynamic correction before
                       adding it to static_saliency. Used when static saliency is
                       intentionally kept in a loss-scale^2 domain until Hessian
                       finalization.

    Returns:
      S_new: (N_local, T, G) fp32 CPU tensor, non-negative.
    """
    U_Sigma_bf16 = dyn_entry["U_Sigma"]      # (N_local*T, R) bf16 CPU
    W_cross_cpu = dyn_entry["W_cross_packed"]  # (G, R, R) fp32 CPU
    R_eff = int(dyn_entry["R_eff"])

    N_local, T, R_in_P = P_delta.shape
    if R_in_P != R_eff:
        raise ValueError(
            f"refresh_dynamic_saliency: P_delta R ({R_in_P}) != dyn_entry R_eff ({R_eff})."
        )

    G = int(num_groups)
    total_tokens = N_local * T
    if U_Sigma_bf16.shape[0] != total_tokens:
        raise ValueError(
            f"refresh_dynamic_saliency: U_Sigma rows ({U_Sigma_bf16.shape[0]}) != N_local*T ({total_tokens})."
        )

    # Move to dev in fp32 for the matmuls.
    U_Sigma_dev = U_Sigma_bf16.to(dev, dtype=torch.float32).reshape(N_local, T, R_eff)
    W_cross_dev = W_cross_cpu.to(dev, dtype=torch.float32)
    P_dev = P_delta.to(dev, dtype=torch.float32)           # (N_local, T, R_eff)

    delta_s_cross = torch.empty(N_local, T, G, dtype=torch.float32, device=dev)
    for k in range(G):
        tmp_c = P_dev @ W_cross_dev[k].t()
        delta_s_cross[..., k] = (U_Sigma_dev * tmp_c).sum(dim=-1)
        del tmp_c

    delta_s_cross.mul_(2.0 / float(N_global))
    if static_saliency_scale != 1.0:
        # Static saliency is intentionally kept in the loss-scale^2 domain
        # until GPTQPlus.finalize_hessian() restores the Hessian scale.
        # Keep the dynamic correction in that same domain before adding.
        delta_s_cross.mul_(float(static_saliency_scale))

    S_old = static_saliency.to(dev, dtype=torch.float32)
    S_new = (S_old + delta_s_cross).clamp_min_(0.0)
    return S_new.cpu()


@torch.no_grad()
def _collect_module_output_projections(
    *,
    layer,
    inputs,
    module_dict,
    V_by_name,
    dev,
    bsz,
    attention_mask,
    position_ids,
    position_embeddings,
    sink_size=0,
):
    """Run ONE forward through `layer` with `inputs`, hook each module's output,
    and project it to the low-rank space on the fly: P_batch = out_batch @ V.
    Writes per-batch (bsz, T_eff, R_eff) bf16 slices into a preallocated CPU
    buffer `(N_local, T_eff, R_eff)` per module; NEVER materialises the full
    (N_local, T, H_out) tensor on CPU — which for Llama-3-8B up/gate is ~120 GB
    per module.

    `V_by_name` tensors MUST be bf16 on `dev`. The hook does a bf16 @ bf16
    matmul directly against the forward activation (also bf16), avoiding an
    expensive fp32 cast of the full (bsz·T, H_out) activation — for 8B up/gate
    that cast alone is ~7 GB GPU transient PER BATCH PER MODULE.

    Args:
      layer:     current transformer block (must be on `dev` already).
      inputs:    (N_local, T, H_hidden) on CPU.
      module_dict: {module_name -> nn.Linear}; only these get hooks.
      V_by_name: {module_name -> (H_out, R_eff) **bf16** tensor on `dev`}.
      bsz:       forward micro-batch size.
      sink_size: when >0, drop leading sink token positions from the captured
                 module output before projection, while still running the layer
                 forward on the full sequence.

    Returns:
      {module_name -> (N_local, T_eff, R_eff) bf16 CPU} — the P projections,
      where T_eff = T - sink_size if T > sink_size else T.
    """
    N_local = inputs.shape[0]
    T = inputs.shape[1]
    sink_size = max(0, int(sink_size))
    drop_sink = sink_size > 0 and T > sink_size
    T_eff = T - sink_size if drop_sink else T
    per_module_P = {}
    per_module_cursor = {}

    for name, V in V_by_name.items():
        R_eff = V.shape[1]
        per_module_P[name] = torch.empty(N_local, T_eff, R_eff, dtype=torch.bfloat16, device="cpu")
        per_module_cursor[name] = 0

    handles = []

    def make_hook(name):
        V_m = V_by_name[name]  # (H_out, R_eff) bf16 dev
        R_eff_local = V_m.shape[1]
        def hook(module, inp, out):
            out_tensor = out[0] if isinstance(out, (tuple, list)) else out
            bsz_local = out_tensor.shape[0]
            if drop_sink:
                out_tensor = out_tensor[:, sink_size:]
            if out_tensor.shape[1] != T_eff:
                raise RuntimeError(
                    f"_collect_module_output_projections: expected token dim {T_eff} "
                    f"after sink slicing, got {out_tensor.shape[1]} for module {name}."
                )
            # Full-sequence output reshapes as a view; sink slicing may
            # materialise only the non-sink bf16 view. In both cases we avoid an
            # fp32 cast of the full activation. Matmul result is bf16.
            out_flat = out_tensor.detach().reshape(bsz_local * T_eff, -1)
            P_flat = out_flat @ V_m  # (bsz_local*T_eff, R_eff) bf16 dev
            P_batch = P_flat.reshape(bsz_local, T_eff, R_eff_local)
            cursor = per_module_cursor[name]
            per_module_P[name][cursor:cursor + bsz_local].copy_(
                P_batch.cpu(), non_blocking=True
            )
            per_module_cursor[name] = cursor + bsz_local
            del P_flat, P_batch
        return hook

    try:
        for name, module in module_dict.items():
            handles.append(module.register_forward_hook(make_hook(name)))
        for j in range(0, N_local, bsz):
            batch_bsz = min(bsz, N_local - j)
            _ = layer(
                inputs[j : j + batch_bsz].to(dev),
                attention_mask=attention_mask.expand(batch_bsz, -1, -1, -1),
                position_ids=position_ids.expand(batch_bsz, -1),
                position_embeddings=(
                    position_embeddings[0].expand(batch_bsz, -1, -1),
                    position_embeddings[1].expand(batch_bsz, -1, -1),
                ),
            )
    finally:
        for h in handles:
            h.remove()

    for name in module_dict.keys():
        if per_module_cursor[name] != N_local:
            raise RuntimeError(
                f"_collect_module_output_projections: module {name} wrote "
                f"{per_module_cursor[name]} / {N_local} rows."
            )
    return per_module_P


# Legacy names kept for binary-compat. New code should use
# `_collect_module_output_projections` which streams the projection and uses
# O(N_local × T × R) CPU memory instead of O(N_local × T × H_out).
def _collect_fp_module_outputs_for_dynsal(*args, **kwargs):
    raise RuntimeError(
        "_collect_fp_module_outputs_for_dynsal is removed — use "
        "_collect_module_output_projections to stream P = Y · V directly."
    )


def _collect_current_module_outputs_for_dynsal(*args, **kwargs):
    raise RuntimeError(
        "_collect_current_module_outputs_for_dynsal is removed — use "
        "_collect_module_output_projections to stream P = Y · V directly."
    )


def compute_refresh_loss(
    refresh_loss_type,
    out_hidden,
    fp_hidden,
    analyzer,
    kl_topk,
    layer_output_fisher=None,
    fp_final_hidden=None,
    refined_A=None,
    layer_output_grad_exact=None,
    layer_output_grad_mean=None,
    pool_positions=None,
    profile_recorder=None,
    sink_size=0,
    a_loss_ratio=1.0,
    a_loss_threshold=None,
):
    refresh_loss_type = canonical_refresh_loss_type(refresh_loss_type)
    # `sink_size > 0` (driven by --ignore_attention_sink) drops the first
    # `sink_size` seq positions from every loss tensor before reduction. The
    # underlying activations still flow through forward / KV unchanged; only
    # the loss values (and gradients flowing back through them) skip the sink.
    # We slice every per-token tensor on dim=1; for refined_mse we additionally
    # slice the grad pool / mean grad along the same axis so the first-order
    # term's `.mean(dim=-1)` denominator is the post-sink token count too.
    def _drop_sink(x):
        if sink_size <= 0 or x is None or x.dim() < 2 or x.shape[1] <= sink_size:
            return x
        return x[:, sink_size:]

    if refresh_loss_type == "kl":
        with profile_recorder.section("compute_refresh_loss.kl.total") if profile_recorder else _NULL_CONTEXT:
            with profile_recorder.section("compute_refresh_loss.kl.logits_quant") if profile_recorder else _NULL_CONTEXT:
                logits = hidden2logits(
                    _drop_sink(out_hidden), analyzer
                ).float()
            with profile_recorder.section("compute_refresh_loss.kl.logits_fp") if profile_recorder else _NULL_CONTEXT:
                logits_fp = hidden2logits(
                    _drop_sink(fp_hidden), analyzer
                ).float()
            if kl_topk > 0:
                with profile_recorder.section("compute_refresh_loss.kl.topk_slice") if profile_recorder else _NULL_CONTEXT:
                    logits_fp, indices = logits_fp.topk(kl_topk, dim=-1, sorted=False)
                    logits = logits.gather(-1, indices)
            with profile_recorder.section("compute_refresh_loss.kl.kl_div") if profile_recorder else _NULL_CONTEXT:
                return tokenwise_kl_from_logits(
                    logits, logits_fp
                ).mean()

    delta = _drop_sink(out_hidden - fp_hidden)
    if (
        is_fisher_backed_loss(refresh_loss_type)
        or is_hidden_mse_loss(refresh_loss_type)
    ) and a_loss_ratio < 1.0:
        delta = _scale_delta_by_abs_quantile(
            delta,
            a_loss_ratio,
            profile_recorder,
            threshold=a_loss_threshold,
        )
    if is_hidden_mse_loss(refresh_loss_type):
        with profile_recorder.section("compute_refresh_loss.hidden_mse") if profile_recorder else _NULL_CONTEXT:
            return 0.5 * delta.square().sum(dim=-1).mean()

    if refresh_loss_type == "residual_kl":
        # Residual-stream shortcut: assume (layers i+1 .. N-1) act as identity on
        # the perturbation δ = out_hidden - fp_hidden. Then the final-layer
        # hidden under the perturbed path equals fp_final_hidden + δ. We only
        # run the lm_head + final norm exactly.
        #
        # Gradient only flows through `delta` (fp_final_hidden is a detached
        # reference). Both lm_head and final norm participate in the backward,
        # but `temporary_requires_grad` in the caller freezes their params so
        # nothing outside `override_weight` accumulates grad.
        if fp_final_hidden is None:
            raise ValueError(
                "`fp_final_hidden` must be provided for refresh_loss_type='residual_kl'."
            )
        fp_final_sliced = _drop_sink(fp_final_hidden)
        with profile_recorder.section("compute_refresh_loss.residual_kl.total") if profile_recorder else _NULL_CONTEXT:
            final_with_delta = fp_final_sliced + delta
            with profile_recorder.section("compute_refresh_loss.residual_kl.logits_perturbed") if profile_recorder else _NULL_CONTEXT:
                logits_perturbed = hidden2logits(
                    final_with_delta, analyzer
                ).float()
            with profile_recorder.section("compute_refresh_loss.residual_kl.logits_fp") if profile_recorder else _NULL_CONTEXT:
                logits_fp = hidden2logits(fp_final_sliced, analyzer).float()
            if kl_topk > 0:
                with profile_recorder.section("compute_refresh_loss.residual_kl.topk_slice") if profile_recorder else _NULL_CONTEXT:
                    logits_fp, indices = logits_fp.topk(kl_topk, dim=-1, sorted=False)
                    logits_perturbed = logits_perturbed.gather(-1, indices)
            with profile_recorder.section("compute_refresh_loss.residual_kl.kl_div") if profile_recorder else _NULL_CONTEXT:
                return tokenwise_kl_from_logits(
                    logits_perturbed, logits_fp
                ).mean()

    if refresh_loss_type == "refined_residual_kl":
        # One-step Jacobi refinement of residual_kl. Instead of assuming
        # f(x+Δx) ≈ f(x), approximate f(x+Δx) ≈ f(x) + A · Δx with a shared
        # per-layer linear operator A (H×H), pre-fit via least squares on
        # (dy, dx-dy) pairs during `collect_static_end_to_end_saliency_and_fisher`.
        # The perturbed final hidden becomes fp_final + Δx + A·Δx = fp_final + (I+A)·Δx.
        # In pytorch batched layout (delta: (B, T, H)), A·Δx on column-vec convention
        # translates to `delta @ A.t()`.
        # When `refined_A` has shape (B, H, H), we're in per-sample sub-A
        # dispatch mode (num_A > 1, samples in this batch map to different
        # A matrices); use bmm. When 2D, all samples share one A → matmul.
        if fp_final_hidden is None:
            raise ValueError(
                "`fp_final_hidden` must be provided for refresh_loss_type='refined_residual_kl'."
            )
        if refined_A is None:
            raise ValueError(
                "`refined_A` must be provided for refresh_loss_type='refined_residual_kl'."
            )
        fp_final_sliced = _drop_sink(fp_final_hidden)
        with profile_recorder.section("compute_refresh_loss.refined_residual_kl.total") if profile_recorder else _NULL_CONTEXT:
            with profile_recorder.section("compute_refresh_loss.refined_residual_kl.A_cast_apply") if profile_recorder else _NULL_CONTEXT:
                _A_cast = refined_A.to(delta.dtype)
                if _A_cast.dim() == 3:
                    # per-sample: (B, H, H) · Δx = torch.bmm(delta, A.transpose(-1,-2))
                    refined_delta = torch.bmm(delta, _A_cast.transpose(-1, -2))
                else:
                    refined_delta = torch.matmul(delta, _A_cast.t())
            final_with_refined = fp_final_sliced + delta + refined_delta
            with profile_recorder.section("compute_refresh_loss.refined_residual_kl.logits_perturbed") if profile_recorder else _NULL_CONTEXT:
                logits_perturbed = hidden2logits(
                    final_with_refined, analyzer
                ).float()
            with profile_recorder.section("compute_refresh_loss.refined_residual_kl.logits_fp") if profile_recorder else _NULL_CONTEXT:
                logits_fp = hidden2logits(fp_final_sliced, analyzer).float()
            if kl_topk > 0:
                with profile_recorder.section("compute_refresh_loss.refined_residual_kl.topk_slice") if profile_recorder else _NULL_CONTEXT:
                    logits_fp, indices = logits_fp.topk(kl_topk, dim=-1, sorted=False)
                    logits_perturbed = logits_perturbed.gather(-1, indices)
            with profile_recorder.section("compute_refresh_loss.refined_residual_kl.kl_div") if profile_recorder else _NULL_CONTEXT:
                return tokenwise_kl_from_logits(
                    logits_perturbed, logits_fp
                ).mean()

    if refresh_loss_type == "refined_diag_residual_kl":
        # Diagonal variant of refined_residual_kl: treat J as diagonal per
        # (token, channel), i.e. assume cross-channel coupling is negligible.
        # `refined_A` has shape (seq, H) or (B, seq, H) — element-wise multiply
        # with delta. LS fit produces A[t, i] = Σ_s dy·δ / (Σ_s dy² + λ) per
        # (token, channel), where sums are taken over samples in the sub-A.
        if fp_final_hidden is None:
            raise ValueError(
                "`fp_final_hidden` must be provided for refresh_loss_type='refined_diag_residual_kl'."
            )
        if refined_A is None:
            raise ValueError(
                "`refined_A` must be provided for refresh_loss_type='refined_diag_residual_kl'."
            )
        fp_final_sliced = _drop_sink(fp_final_hidden)
        # `refined_A` was fit only on non-sink positions in the precompute
        # hooks (or full-T positions when sink_size=0), so its token axis
        # already matches `delta`'s sliced seq dim — no extra slice here.
        with profile_recorder.section("compute_refresh_loss.refined_diag_residual_kl.total") if profile_recorder else _NULL_CONTEXT:
            with profile_recorder.section("compute_refresh_loss.refined_diag_residual_kl.A_apply") if profile_recorder else _NULL_CONTEXT:
                _A_cast = refined_A.to(delta.dtype)
                # Broadcasts: (seq, H) → (1, seq, H) vs (B, seq, H). Both element-wise.
                refined_delta = delta * _A_cast
            final_with_refined = fp_final_sliced + delta + refined_delta
            with profile_recorder.section("compute_refresh_loss.refined_diag_residual_kl.logits_perturbed") if profile_recorder else _NULL_CONTEXT:
                logits_perturbed = hidden2logits(
                    final_with_refined, analyzer
                ).float()
            with profile_recorder.section("compute_refresh_loss.refined_diag_residual_kl.logits_fp") if profile_recorder else _NULL_CONTEXT:
                logits_fp = hidden2logits(fp_final_sliced, analyzer).float()
            if kl_topk > 0:
                with profile_recorder.section("compute_refresh_loss.refined_diag_residual_kl.topk_slice") if profile_recorder else _NULL_CONTEXT:
                    logits_fp, indices = logits_fp.topk(kl_topk, dim=-1, sorted=False)
                    logits_perturbed = logits_perturbed.gather(-1, indices)
            with profile_recorder.section("compute_refresh_loss.refined_diag_residual_kl.kl_div") if profile_recorder else _NULL_CONTEXT:
                return tokenwise_kl_from_logits(
                    logits_perturbed, logits_fp
                ).mean()

    if layer_output_fisher is None:
        raise ValueError(
            f"`layer_output_fisher` must be provided for refresh_loss_type='{refresh_loss_type}' "
            "(fisher_diag_mse / legacy_fisher_diag_mse / refined_mse)."
        )
    with profile_recorder.section("compute_refresh_loss.fisher_diag_mse.total") if profile_recorder else _NULL_CONTEXT:
        with profile_recorder.section("compute_refresh_loss.fisher_diag_mse.quadratic") if profile_recorder else _NULL_CONTEXT:
            hidden_size = delta.shape[-1]
            if refresh_loss_type == "legacy_fisher_diag_mse":
                if layer_output_fisher.dim() != 3:
                    raise ValueError(
                        "legacy_fisher_diag_mse expects per-token Fisher diagonal "
                        f"with shape (B, T, H), got {tuple(layer_output_fisher.shape)}."
                    )
                if tuple(layer_output_fisher.shape) != tuple(delta.shape):
                    raise ValueError(
                        "legacy_fisher_diag_mse Fisher diag shape must match "
                        f"delta shape after sink slicing: fisher={tuple(layer_output_fisher.shape)} "
                        f"delta={tuple(delta.shape)}."
                    )
                fisher_diag = layer_output_fisher.to(device=delta.device, dtype=torch.float32)
                fisher_loss_per_token = 0.5 * (
                    delta.float().square() * fisher_diag
                ).sum(dim=-1)
                fisher_loss = fisher_loss_per_token.mean()
            else:
                if layer_output_fisher.dim() != 2 or layer_output_fisher.shape[0] != layer_output_fisher.shape[1]:
                    raise ValueError(
                        f"Expected layer-output Fisher matrix with shape (H, H), got {tuple(layer_output_fisher.shape)}."
                    )
                if layer_output_fisher.shape[0] != hidden_size:
                    raise ValueError(
                        f"Fisher matrix hidden dim ({layer_output_fisher.shape[0]}) does not match delta hidden dim ({hidden_size})."
                    )
                fisher = layer_output_fisher.to(device=delta.device, dtype=torch.float32)
                delta_flat = delta.float().reshape(-1, hidden_size)
                fisher_loss_per_token = 0.5 * (
                    delta_flat.matmul(fisher) * delta_flat
                ).sum(dim=-1)
                fisher_loss = fisher_loss_per_token.mean()

        if refresh_loss_type != "refined_mse":
            return fisher_loss

        # refined_mse = fisher_diag_mse + first-order Taylor term  g · Δy .
        # Caller provides per-layer grad pool state:
        #   - layer_output_grad_exact: (B_pool, seq, H) exact per-token grad for
        #     the subset of samples in this batch that hit the pre-collected pool.
        #   - pool_positions: (B_pool,) LongTensor giving batch-local row ids where
        #     the exact grads apply (same ordering as layer_output_grad_exact).
        #   - layer_output_grad_mean: (H,) mean of the full pool over (N_pool, seq),
        #     used for batch rows that are NOT in the pool.
        # At layer 0 the whole KL-gradient is identically zero, so the caller just
        # passes mean=zeros + empty pool_positions → first_order collapses to 0 and
        # refined_mse degenerates to fisher_diag_mse, matching the physics.
        if layer_output_grad_mean is None:
            raise ValueError(
                "`layer_output_grad_mean` must be provided for refresh_loss_type='refined_mse'."
            )
        with profile_recorder.section("compute_refresh_loss.refined_mse.first_order") if profile_recorder else _NULL_CONTEXT:
            B = delta.shape[0]
            mean_grad_cast = layer_output_grad_mean.to(delta.dtype)
            # Split loss so we avoid materialising a (B, seq, H) broadcast of mean_grad.
            if pool_positions is not None and pool_positions.numel() > 0:
                if layer_output_grad_exact is None:
                    raise ValueError(
                        "refined_mse: `pool_positions` non-empty but `layer_output_grad_exact` is None."
                    )
                with profile_recorder.section("compute_refresh_loss.refined_mse.pool_fo") if profile_recorder else _NULL_CONTEXT:
                    grad_exact_cast = _drop_sink(layer_output_grad_exact.to(delta.dtype))
                    delta_pool = delta.index_select(0, pool_positions.to(delta.device))
                    fo_pool_sum = (grad_exact_cast * delta_pool).sum(dim=-1).mean(dim=-1).sum()
                    # Build non-pool mask on the fly; fine because B is small (≤64 typical).
                    non_pool_mask = torch.ones(B, dtype=torch.bool, device=delta.device)
                    non_pool_mask[pool_positions.to(delta.device)] = False
            else:
                fo_pool_sum = torch.zeros((), dtype=delta.dtype, device=delta.device)
                non_pool_mask = torch.ones(B, dtype=torch.bool, device=delta.device)
            if non_pool_mask.any():
                with profile_recorder.section("compute_refresh_loss.refined_mse.nonpool_fo") if profile_recorder else _NULL_CONTEXT:
                    delta_nonpool_mean_t = delta[non_pool_mask].mean(dim=1)  # (B_nonpool, H)
                    fo_nonpool_sum = (delta_nonpool_mean_t @ mean_grad_cast).sum()
            else:
                fo_nonpool_sum = torch.zeros((), dtype=delta.dtype, device=delta.device)
            first_order = (fo_pool_sum + fo_nonpool_sum) / B
        return fisher_loss + first_order


def collect_layer_output_grad_for_refined_mse(
    *,
    analyzer,
    layer,
    layer_idx,
    layers,
    inps,
    fp_inps_final,
    attention_mask,
    position_ids,
    position_embeddings,
    sample_ids_local,
    backward_bsz,
    kl_topk,
    dev,
    store_dtype=torch.bfloat16,
    layer_recorder=None,
    collect_next=False,
    sink_size=0,
):
    """Capture g = ∂(top-k KL vs teacher) / ∂(out_of_layer_idx) for a per-layer
    random sample pool. Used only by refresh_loss_type='refined_mse' to supply
    the first-order Taylor term `g · Δy` on top of fisher_diag_mse.

    Semantic state:
        - inps already reflects upstream (layers 0..layer_idx-1) weight/act
          quantization errors — same buffer the main quant loop uses.
        - layer_idx and all downstream blocks are forward'd in FP (act-quant
          wrappers disabled inside this call for the duration).
        - Backward runs to `out_of_layer_idx` only; we don't need gradients at
          the input side.

    Caller must have moved `layer` + `layers[layer_idx+1:]` +
    `analyzer.get_layernorm_before_head()` + `analyzer.get_lm_head()` to
    `dev` before invoking, and must restore original residency afterwards.

    When `collect_next=True` this additionally captures ∂KL/∂(out_of_layer_{
    idx+1}) on the same single backward pass (same samples, same forward
    graph) and returns it as `grad_pool_next` / `mean_grad_next`. Used by
    loss_slide_window + refined_mse to supply the first-order term for the
    next-layer refresh loss without paying for a second forward.
    `collect_next` requires `layer_idx + 1 < len(layers)`.

    Returns
    -------
    grad_pool : Tensor of shape (N, seq, H) dtype=store_dtype on CPU
        Per-sample per-token gradient at the current layer's output.
    mean_grad : Tensor of shape (H,) fp32 on `dev`
        Mean over (sample, seq) dims of `grad_pool`. Broadcast in refresh loss
        for batch rows that don't hit the pool.
    grad_pool_next : Tensor of shape (N, seq, H) dtype=store_dtype on CPU, or
        None when `collect_next=False`. Per-sample per-token gradient at the
        next layer's output.
    mean_grad_next : Tensor of shape (H,) fp32 on `dev`, or None when
        `collect_next=False`. Mean over (sample, seq) dims of `grad_pool_next`.
    """
    N = len(sample_ids_local)
    if N == 0:
        raise ValueError("`sample_ids_local` must be non-empty.")
    if N % backward_bsz != 0:
        raise ValueError(
            f"num_samples_for_refined_mse ({N}) must be divisible by "
            f"backward_bsz ({backward_bsz})."
        )
    if collect_next and layer_idx + 1 >= len(layers):
        raise ValueError(
            f"collect_next=True requires a downstream layer (layer_idx="
            f"{layer_idx}, len(layers)={len(layers)})."
        )
    seq_len = inps.shape[1]
    hidden_size = inps.shape[2]
    # `pin_memory=True` is what lets the per-batch D2H below actually overlap
    # with the next iteration's forward+backward on the default stream —
    # cudaMemcpyAsync requires a pinned host destination. The pool is
    # freshly allocated per layer and freed at layer end, so this is extra
    # short-lived pinned memory proportional to (n_pool * seq * H * 2B);
    # Qwen3-4B at n_pool=32 → ~320 MB per layer. Well under the cgroup
    # mlock limit on typical training hosts.
    grad_pool = torch.empty(
        (N, seq_len, hidden_size), dtype=store_dtype, device="cpu",
        pin_memory=True,
    )
    grad_pool_next = None
    if collect_next:
        grad_pool_next = torch.empty(
            (N, seq_len, hidden_size), dtype=store_dtype, device="cpu",
            pin_memory=True,
        )
    # Side stream dedicated to the grad_pool D2H copies. Keeps the default
    # stream free to run the next iter's forward+backward while the copy
    # drains in the background. A single side stream is enough for both pools
    # — they're written from the same backward, so ordering their copies on
    # one stream is fine.
    d2h_stream = torch.cuda.Stream(device=dev)

    # Drop act/K-cache quant on the current + downstream layers so the forward used
    # for the gradient measurement is pure FP. (Weights are still FP — this
    # layer hasn't been quantised yet, and downstream layers haven't either.)
    with layer_recorder.section("layer.refined_mse_grad_pool.disable_act_quant") if layer_recorder else _NULL_CONTEXT:
        restore_states = []
        downstream_layers = layers[layer_idx + 1:]
        all_layers_for_grad = [layer, *downstream_layers]
        for lay in all_layers_for_grad:
            act_bits = quant_utils.disable_act_quant(lay)
            k_state = rotation_utils.disable_k_cache_quant(lay)
            restore_states.append((lay, act_bits, k_state))

    grad_modules = all_layers_for_grad + [
        analyzer.get_layernorm_before_head(),
        analyzer.get_lm_head(),
    ]
    try:
        with temporary_requires_grad(grad_modules, []):
            with torch.enable_grad():
                for start in range(0, N, backward_bsz):
                    with layer_recorder.section("layer.refined_mse_grad_pool.batch.total") if layer_recorder else _NULL_CONTEXT:
                        with layer_recorder.section("layer.refined_mse_grad_pool.batch.prepare") if layer_recorder else _NULL_CONTEXT:
                            batch_ids = sample_ids_local[start:start + backward_bsz]
                            bsz = len(batch_ids)
                            b_attn = attention_mask.expand(bsz, -1, -1, -1)
                            b_pos_ids = position_ids.expand(bsz, -1)
                            b_pos_emb = (
                                position_embeddings[0].expand(bsz, -1, -1),
                                position_embeddings[1].expand(bsz, -1, -1),
                            )
                            h_in = (
                                inps[batch_ids]
                                .to(dev)
                                .detach()
                                .clone()
                                .requires_grad_(True)
                            )
                        with layer_recorder.section("layer.refined_mse_grad_pool.batch.forward_current") if layer_recorder else _NULL_CONTEXT:
                            out = layer(
                                h_in,
                                attention_mask=b_attn,
                                position_ids=b_pos_ids,
                                position_embeddings=b_pos_emb,
                            )
                            out_i = out[0] if isinstance(out, (tuple, list)) else out
                            h = out_i
                        out_next = None
                        with layer_recorder.section("layer.refined_mse_grad_pool.batch.forward_downstream") if layer_recorder else _NULL_CONTEXT:
                            for k in range(layer_idx + 1, len(layers)):
                                out_k = layers[k](
                                    h,
                                    attention_mask=b_attn,
                                    position_ids=b_pos_ids,
                                    position_embeddings=b_pos_emb,
                                )
                                h = out_k[0] if isinstance(out_k, (tuple, list)) else out_k
                                if collect_next and k == layer_idx + 1:
                                    out_next = h
                        with layer_recorder.section("layer.refined_mse_grad_pool.batch.logits_student") if layer_recorder else _NULL_CONTEXT:
                            h_for_logits = h if sink_size <= 0 else h[:, sink_size:]
                            logits_student = hidden2logits(h_for_logits, analyzer)
                        with layer_recorder.section("layer.refined_mse_grad_pool.batch.logits_teacher") if layer_recorder else _NULL_CONTEXT:
                            fp_final_batch = fp_inps_final[batch_ids].to(dev)
                            if sink_size > 0:
                                fp_final_batch = fp_final_batch[:, sink_size:]
                            logits_teacher = hidden2logits(
                                fp_final_batch, analyzer
                            ).detach()
                        if kl_topk > 0:
                            with layer_recorder.section("layer.refined_mse_grad_pool.batch.topk_slice") if layer_recorder else _NULL_CONTEXT:
                                logits_teacher, idx_top = logits_teacher.topk(
                                    kl_topk, dim=-1, sorted=False
                                )
                                logits_student = logits_student.gather(-1, idx_top)
                        with layer_recorder.section("layer.refined_mse_grad_pool.batch.loss_build") if layer_recorder else _NULL_CONTEXT:
                            kl_loss = tokenwise_kl_from_logits(
                                logits_student, logits_teacher
                            ).sum()
                        with layer_recorder.section("layer.refined_mse_grad_pool.batch.backward") if layer_recorder else _NULL_CONTEXT:
                            if collect_next:
                                # One backward pass, two gradient outputs — cheaper
                                # than calling `autograd.grad(..., retain_graph=True)`
                                # twice.
                                grad_out, grad_out_next = torch.autograd.grad(
                                    kl_loss, [out_i, out_next], retain_graph=False
                                )
                            else:
                                grad_out = torch.autograd.grad(
                                    kl_loss, out_i, retain_graph=False
                                )[0]
                                grad_out_next = None
                        with layer_recorder.section("layer.refined_mse_grad_pool.batch.store") if layer_recorder else _NULL_CONTEXT:
                            # Cast to store dtype on the default stream so the
                            # data is produced where backward just ran, then
                            # fork the D2H onto d2h_stream. `wait_stream`
                            # serialises the copy AFTER the cast finishes;
                            # `record_stream` keeps grad_out_bf16's GPU
                            # allocation alive until d2h_stream is done
                            # reading from it (otherwise PyTorch's caching
                            # allocator could reuse the memory for the next
                            # iter's forward and corrupt the in-flight copy).
                            grad_out_bf16 = grad_out.detach().to(store_dtype)
                            grad_out_next_bf16 = (
                                grad_out_next.detach().to(store_dtype)
                                if grad_out_next is not None else None
                            )
                            d2h_stream.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(d2h_stream):
                                grad_pool[start:start + bsz].copy_(
                                    grad_out_bf16, non_blocking=True
                                )
                                grad_out_bf16.record_stream(d2h_stream)
                                if grad_out_next_bf16 is not None:
                                    grad_pool_next[start:start + bsz].copy_(
                                        grad_out_next_bf16, non_blocking=True
                                    )
                                    grad_out_next_bf16.record_stream(d2h_stream)
                        del out, out_i, h, logits_student, logits_teacher
                        del kl_loss, grad_out, h_in, grad_out_bf16
                        if out_next is not None:
                            del out_next
                        if grad_out_next is not None:
                            del grad_out_next
                        if grad_out_next_bf16 is not None:
                            del grad_out_next_bf16
    finally:
        with layer_recorder.section("layer.refined_mse_grad_pool.restore_act_quant") if layer_recorder else _NULL_CONTEXT:
            for lay, act_bits, k_state in restore_states:
                rotation_utils.enable_k_cache_quant(lay, k_state)
                quant_utils.enable_act_quant(lay, act_bits)

    with layer_recorder.section("layer.refined_mse_grad_pool.mean_reduce") if layer_recorder else _NULL_CONTEXT:
        # Drain any in-flight async D2Hs before we read grad_pool on the CPU.
        # `.float()` / `.mean()` below are plain CPU ops that bypass any CUDA
        # stream ordering, so we need an explicit sync here.
        d2h_stream.synchronize()
        # Per-token gradients in the pool come from the output-token SUM KL
        # loss above. For non-pool refresh samples we still use a shared
        # coefficient over middle-layer tokens, so this remains an average over
        # the pool's middle-token axis only.
        # When sink_size > 0 the kl_loss above already excluded sink positions,
        # so grad_pool[:, :sink_size] is exactly zero. We still need to slice
        # them out of the divisor to avoid a (T-sink)/T bias on `mean_grad`
        # (which the non-pool first-order term in compute_refresh_loss reads).
        gp_for_mean = grad_pool if sink_size <= 0 else grad_pool[:, sink_size:]
        mean_grad = gp_for_mean.float().mean(dim=(0, 1)).to(dev)
        mean_grad_next = None
        if grad_pool_next is not None:
            gpn_for_mean = grad_pool_next if sink_size <= 0 else grad_pool_next[:, sink_size:]
            mean_grad_next = gpn_for_mean.float().mean(dim=(0, 1)).to(dev)
    return grad_pool, mean_grad, grad_pool_next, mean_grad_next


def apply_dense_optimizer_step(
    param,
    grad,
    lr,
    optimizer,
    opt_state=None,
    grad_clip=1.0,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_eps=1e-8,
):
    if grad.numel() == 0 or lr == 0:
        return torch.zeros_like(grad), opt_state

    grad_step = grad
    if grad_clip is not None and grad_clip > 0:
        grad_step = grad_step.clamp(min=-grad_clip, max=grad_clip)
    if optimizer == "sgd":
        update = lr * grad_step
        param.sub_(update.to(param.dtype))
        return update, opt_state

    if optimizer != "adam":
        raise ValueError(f"Unsupported optimizer `{optimizer}`. Expected one of: sgd, adam.")

    if opt_state is None:
        opt_state = {
            "step": 0,
            "exp_avg": torch.zeros_like(param, dtype=torch.float32),
            "exp_avg_sq": torch.zeros_like(param, dtype=torch.float32),
        }
    opt_state["step"] += 1
    exp_avg = opt_state["exp_avg"]
    exp_avg_sq = opt_state["exp_avg_sq"]
    grad_step = grad_step.float()
    exp_avg.mul_(adam_beta1).add_(grad_step, alpha=1 - adam_beta1)
    exp_avg_sq.mul_(adam_beta2).addcmul_(grad_step, grad_step, value=1 - adam_beta2)
    bias_correction1 = 1 - adam_beta1 ** opt_state["step"]
    bias_correction2 = 1 - adam_beta2 ** opt_state["step"]
    denom = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
    denom.add_(adam_eps)
    step_size = lr / bias_correction1
    update = step_size * (exp_avg / denom)
    param.sub_(update.to(param.dtype))
    return update, opt_state


def collect_layer_output_fisher_only(
    *,
    model,
    layer,
    analyzer,
    inps,
    fp_inps,
    batch_attention_mask,
    batch_position_ids,
    batch_position_embeddings,
    bsz,
    kl_topk,
    grad_hessian_topk,
    dev,
    layer_idx,
    layer_recorder=None,
    sink_size=0,
    legacy_diag=False,
):
    if legacy_diag:
        raise ValueError(
            "legacy_fisher_diag_mse per-token diagonals must come from "
            "collect_static_end_to_end_saliency_and_fisher's sum-reduced NLL "
            "backward, not the layerwise KL Fisher fallback."
        )
    fisher_sum = None
    if legacy_diag:
        fisher_chunks = []
    with torch.enable_grad():
        for j in tqdm(
            range(0, inps.shape[0], bsz),
            ncols=120,
            desc=f"Layer {layer_idx} Pre-GD Fisher",
            position=1,
            leave=False,
        ):
            with layer_recorder.section("layer.pre_quant_fisher.batch.total") if layer_recorder else _NULL_CONTEXT:
                with layer_recorder.section("layer.pre_quant_fisher.forward.layer") if layer_recorder else _NULL_CONTEXT:
                    out = layer(
                        inps[j : j + bsz].to(dev),
                        attention_mask=batch_attention_mask,
                        position_ids=batch_position_ids,
                        position_embeddings=batch_position_embeddings,
                    )
                with layer_recorder.section("layer.pre_quant_fisher.forward.hidden_extract") if layer_recorder else _NULL_CONTEXT:
                    out_hidden = out[0] if isinstance(out, (tuple, list)) else out
                with layer_recorder.section("layer.pre_quant_fisher.forward.logits_quant") if layer_recorder else _NULL_CONTEXT:
                    logits = hidden2logits(out_hidden, analyzer)
                with layer_recorder.section("layer.pre_quant_fisher.forward.logits_fp") if layer_recorder else _NULL_CONTEXT:
                    logits_fp = hidden2logits(fp_inps[j : j + bsz].to(dev), analyzer)

                grad_hessian_logits = logits
                grad_hessian_logits_fp = logits_fp
                if grad_hessian_topk > 0:
                    with layer_recorder.section("layer.pre_quant_fisher.forward.topk_slice") if layer_recorder else _NULL_CONTEXT:
                        grad_hessian_logits_fp, grad_hessian_indices = logits_fp.topk(
                            grad_hessian_topk,
                            dim=-1,
                            sorted=False,
                        )
                        grad_hessian_logits = logits.gather(-1, grad_hessian_indices)

                with layer_recorder.section("layer.pre_quant_fisher.loss_build") if layer_recorder else _NULL_CONTEXT:
                    kl_logits = grad_hessian_logits if grad_hessian_topk > 0 else logits
                    kl_logits_fp = grad_hessian_logits_fp if grad_hessian_topk > 0 else logits_fp
                    if grad_hessian_topk <= 0 and kl_topk > 0:
                        kl_logits_fp, indices = logits_fp.topk(kl_topk, dim=-1, sorted=False)
                        kl_logits = logits.gather(-1, indices)
                    if sink_size > 0:
                        # Drop sink positions from the loss; sink activations still
                        # participated in the forward (KV) so the gradient at sink
                        # positions reflects attention flow, but we exclude them
                        # from the Fisher aggregation below to be consistent.
                        kl_logits = kl_logits[:, sink_size:]
                        kl_logits_fp = kl_logits_fp[:, sink_size:]
                    kl_loss = tokenwise_kl_from_logits(
                        kl_logits, kl_logits_fp
                    )
                    # Sum over output samples/tokens to match the NLL-based
                    # saliency/Fisher convention. The Fisher accumulator below
                    # then averages only over middle-layer tokens when storing
                    # the shared E_middle[g g^T] coefficient.
                    kl_loss = kl_loss.sum(dim=-1).sum()

                with layer_recorder.section("layer.pre_quant_fisher.backward") if layer_recorder else _NULL_CONTEXT:
                    out_hidden.retain_grad()

                    def layer_output_grad_hook(grad):
                        nonlocal fisher_sum
                        g = grad if sink_size <= 0 else grad[:, sink_size:]
                        grad_fp32 = g.detach().float()
                        if legacy_diag:
                            fisher_chunks.append(
                                grad_fp32.pow(2)
                                .div(_E2E_PRECOMPUTE_QUADRATIC_SCALE)
                                .to(torch.bfloat16)
                                .cpu()
                            )
                        else:
                            grad_flat = grad_fp32.reshape(-1, grad_fp32.shape[-1])
                            fisher_block = grad_flat.t() @ grad_flat
                            if fisher_sum is None:
                                fisher_sum = fisher_block
                            else:
                                fisher_sum.add_(fisher_block)

                    out_hidden.register_hook(layer_output_grad_hook)
                    model.zero_grad()
                    kl_loss.backward()

                # Per-batch cleanup_memory() was here. Removed for the same
                # reason as above — autograd state is released automatically.

    if legacy_diag:
        if not fisher_chunks:
            return None
        return torch.cat(fisher_chunks, dim=0).cpu()
    if fisher_sum is None:
        return None
    dist_utils.allreduce_sum_(fisher_sum)
    eff_seq = max(1, inps.shape[1] - max(0, sink_size))
    total_tokens = dist_utils.allreduce_sum_scalar(inps.shape[0] * eff_seq)
    return (fisher_sum / float(total_tokens)).to(torch.bfloat16).cpu()


class ModuleGradSecondMoment:
    """Exact per-weight second moment of the refresh gradient, measured instead
    of estimated by EMA.

    For a linear module y = W x a single token contributes g_W = g_y x^T, so

        E[g_W^2][i, j] = E_t[ g_y[t, i]^2 * x[t, j]^2 ]

    which is what Adam's `exp_avg_sq` spends a whole block_gd run (~23 steps for
    a 3072-column module) failing to warm up to. `g_y` is taken from the refresh
    backward, so it carries the direction of the loss block_gd actually
    optimises, not the layer-wise MSE.

    This class only measures; `warm_adam` is the consumer that turns the
    measurement into Adam's starting v.
    """

    def __init__(self, module, sink_size=0):
        self.module = module
        self.sink_size = int(sink_size)
        weight = module.weight
        self.sum_g = torch.zeros(
            weight.shape[0], weight.shape[1],
            dtype=torch.float32, device=weight.device,
        )
        self.sum_g2 = torch.zeros_like(self.sum_g)
        self.sample_count = 0
        self.token_count = 0
        # Forward and backward fire independently: the forward hook can run
        # while the gradient hook never does (module output off the autograd
        # path, a no_grad forward, a shape guard rejecting). Without separate
        # counters an all-zero prior looks the same as a genuinely zero one.
        self.fwd_calls = 0
        self.grad_calls = 0
        self.grad_skipped = 0
        self._handles = []

    def _forward_hook(self, _module, inp, out):
        x = inp[0]
        if x.dim() != 3:
            return
        if self.sink_size > 0 and x.shape[1] > self.sink_size:
            x = x[:, self.sink_size:]
        x_f = x.float()
        sink = self.sink_size
        sum_g, sum_g2 = self.sum_g, self.sum_g2

        accum = self

        def _grad_hook(grad):
            g = grad
            if g.dim() != 3:
                accum.grad_skipped += 1
                return None
            if sink > 0 and g.shape[1] > sink:
                g = g[:, sink:]
            if g.shape[1] != x_f.shape[1] or g.shape[0] != x_f.shape[0]:
                # Misalignment would silently corrupt the prior; skip instead.
                accum.grad_skipped += 1
                return None
            accum.grad_calls += 1
            bsz = g.shape[0]
            # functional_call differentiates the batch MEAN loss, so the hook
            # sees (1/bsz) * d(sum of per-sample losses)/dy. Undo that to
            # recover each sample's own gradient.
            g_s = torch.einsum("bso,bsi->boi", g.float() * float(bsz), x_f)
            sum_g.add_(g_s.sum(dim=0).to(sum_g.device))
            sum_g2.add_(g_s.pow_(2).sum(dim=0).to(sum_g2.device))
            return None

        self.fwd_calls += 1
        if isinstance(out, torch.Tensor) and out.requires_grad:
            out.register_hook(_grad_hook)
            self.sample_count += x_f.shape[0]
            self.token_count += x_f.shape[0] * x_f.shape[1]

    def attach(self):
        if not self._handles:
            self._handles.append(self.module.register_forward_hook(self._forward_hook))
        return self

    def detach(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    # Order of the per-rank statistics row; `rank_stats` is exposed so the
    # caller can print every rank's contribution on one line.
    STAT_FIELDS = ("samples", "tokens", "fwd", "grad", "skipped", "max|sum_g|")

    def all_reduce_(self):
        """Make the accumulator describe the global calibration set.

        Under DP each rank only forwards its own shard, and with
        dp_global_shuffle a rank may legitimately see none of a given refresh
        batch. gbar and Var_s are properties of the whole set, so the sums have
        to be summed across ranks before build_prior reads them.

        Counters are gathered per rank rather than summed into one number:
        "the prior is zero" means something different when one rank measured
        nothing than when all of them did, and a single total hides which.
        Returns the global sample count.
        """
        local = [
            float(self.sample_count), float(self.token_count),
            float(self.fwd_calls), float(self.grad_calls),
            float(self.grad_skipped),
            float(self.sum_g.abs().max().item()) if self.sum_g.numel() else 0.0,
        ]
        if not (dist.is_available() and dist.is_initialized()):
            self.rank_stats = [local]
            return self.sample_count

        world = dist_utils.get_world_size()
        rank = dist_utils.get_rank()
        stats = torch.zeros(
            world, len(local), dtype=torch.float32, device=self.sum_g.device
        )
        stats[rank] = torch.tensor(local, dtype=torch.float32, device=self.sum_g.device)
        dist_utils.allreduce_sum_(stats)
        self.rank_stats = stats.tolist()

        packed = torch.cat([self.sum_g.reshape(-1), self.sum_g2.reshape(-1)])
        dist_utils.allreduce_sum_(packed)
        n = self.sum_g.numel()
        self.sum_g.copy_(packed[:n].view_as(self.sum_g))
        self.sum_g2.copy_(packed[n:2 * n].view_as(self.sum_g2))

        totals = stats.sum(dim=0)
        self.sample_count = int(totals[0].item())
        self.token_count = int(totals[1].item())
        self.fwd_calls = int(totals[2].item())
        self.grad_calls = int(totals[3].item())
        self.grad_skipped = int(totals[4].item())
        return self.sample_count

    def format_rank_stats(self):
        rows = getattr(self, "rank_stats", None)
        if not rows:
            return "n/a"
        return " | ".join(
            "r%d s=%d fwd=%d grad=%d skip=%d max|g|=%s"
            % (r, int(v[0]), int(v[2]), int(v[3]), int(v[4]), format_log_value(v[5]))
            for r, v in enumerate(rows)
        )

    def build_prior(self, batch_size):
        """Exact value of what Adam's exp_avg_sq estimates, at batch size B.

        A refresh gradient is the mean of B per-sample gradients drawn from the
        calibration set, so

            E[g_batch^2] = gbar^2 + Var_s(g_s) / B

        Both terms come from this one pass: `gbar` from the running sum, the
        per-sample variance from the sum of squares. B enters explicitly, so the
        same prior is NOT valid across configurations. Note the second term is
        information Adam structurally cannot have at B = N, where every refresh
        sees the same full calibration set and its observed g is the
        deterministic gbar.
        """
        if self.sample_count == 0:
            return None
        n = float(self.sample_count)
        gbar = self.sum_g / n
        e_g2 = self.sum_g2 / n
        gbar_sq = gbar.pow(2)
        var_s = (e_g2 - gbar_sq).clamp_min(0.0)
        return gbar_sq + var_s / max(float(batch_size), 1.0)


def collect_true_weight_gradient(
    layer,
    analyzer,
    module_name,
    full,
    inps,
    fp_inps,
    attention_mask,
    position_ids,
    position_embeddings,
    bsz,
    kl_topk,
    dev,
    weight_override=None,
    sample_indices=None,
    refresh_loss_type="kl",
    layer_output_fisher=None,
    slide_alpha=1.0,
    next_layer=None,
    fp_inps_next=None,
    next_layer_output_fisher=None,
    fp_inps_final=None,
    refined_A_list=None,
    next_refined_A_list=None,
    samples_per_A=0,
    global_shuffle=False,
    dp_rank=0,
    shard_size=None,
    refresh_mb=None,
    refined_mse_pool_ids=None,
    refined_mse_grad_pool=None,
    refined_mse_mean_grad=None,
    next_refined_mse_grad_pool=None,
    next_refined_mse_mean_grad=None,
    layer_recorder=None,
    sink_size=0,
    a_loss_ratio=1.0,
    a_loss_clip_scope="local_backward_chunk",
    grad_sq_accum=None,
):
    """Compute the refresh gradient as a per-rank partial sum + count.

    Returns
    -------
    partial_grad_sum : torch.Tensor
        Sum of per-sample gradients contributed by this rank's local slice of
        `sample_indices` (unnormalised). Zeros if this rank owns 0 samples.
    partial_count : int
        Number of samples this rank actually processed.
    loss_sum : float
        Sum of per-sample losses over the processed samples (unnormalised).
    extras : dict
        Optional diagnostics (slide-window components, etc.), all in
        "sum" form so callers can allreduce then divide.

    When `global_shuffle=False` (default / stratified): `sample_indices` is
    treated as rank-local indices into `inps`; no filtering. Each rank typically
    gets an equal-sized chunk. Equivalent to the pre-global-shuffle pipeline.

    When `global_shuffle=True`: `sample_indices` is a list of GLOBAL sample
    ids; this function filters to the rank's contiguous shard
    `[dp_rank * shard_size, (dp_rank + 1) * shard_size)` and only processes
    those. Empty-shard runs still return a valid zero tensor + count=0 so the
    caller's allreduce does the right thing.
    """
    refresh_loss_type = canonical_refresh_loss_type(refresh_loss_type)
    resolved_module_name, module = resolve_quant_module(full, module_name)
    functional_weight_name = functional_weight_name_for_quant_module(
        layer, module_name, resolved_module_name, module
    )
    if sample_indices is None:
        raw_indices = list(range(inps.shape[0]))
    else:
        raw_indices = list(sample_indices)

    if global_shuffle:
        if shard_size is None:
            shard_size = inps.shape[0]
        rank_start = dp_rank * shard_size
        rank_end = rank_start + shard_size
        selected_indices = [
            gi - rank_start for gi in raw_indices if rank_start <= gi < rank_end
        ]
    else:
        selected_indices = raw_indices

    # Sub-A dispatch for refined_residual_kl with num_A > 1: each sample in a
    # batch picks its own A based on its global sample id, regardless of where
    # the batch boundary falls. Per-batch we build either a (H, H) matrix (when
    # all samples share one a_idx — fast path) or a stacked (B, H, H) tensor
    # with one A per sample (gather + bmm — slower path). No sorting / trimming
    # required; batches can freely span sub-A boundaries.
    num_A = len(refined_A_list) if refined_A_list is not None else 1
    num_A_next = len(next_refined_A_list) if next_refined_A_list is not None else 1
    sub_a_mode = (
        refined_A_list is not None
        and num_A > 1
        and samples_per_A > 0
        and refresh_loss_type == "refined_residual_kl"
    )

    # refined_mse: build {local_idx -> pool_pos} lookup once. Samples whose
    # local id lands in the pool use their own per-token gradient; everyone
    # else falls back to the pool's (batch, seq)-mean. The pool was collected
    # by `collect_layer_output_grad_for_refined_mse` before this layer's quant
    # loop opens, and stays fixed through all refresh steps on this layer.
    if refresh_loss_type == "refined_mse":
        if refined_mse_mean_grad is None:
            raise ValueError(
                "refined_mse: refined_mse_mean_grad is required (pass mean_grad "
                "from collect_layer_output_grad_for_refined_mse, or zeros(H) at "
                "layer 0)."
            )
        if refined_mse_pool_ids is not None and refined_mse_pool_ids.numel() > 0:
            refined_mse_pool_lookup = {
                int(li): int(pp) for pp, li in enumerate(refined_mse_pool_ids.tolist())
            }
        else:
            refined_mse_pool_lookup = {}
    else:
        refined_mse_pool_lookup = None

    if weight_override is None:
        override_weight = module.weight.detach().clone()
    else:
        # Previously this did .detach().to(dev, dtype=...).clone() — two copies
        # when the cast actually ran (which is the common case: weight_snapshot
        # is fp32, module.weight is bf16). `.to(...)` already produces a fresh
        # tensor when dtype/device differ, so one clone is enough.
        src = weight_override.detach()
        target_dev = module.weight.device
        target_dtype = module.weight.data.dtype
        if src.device == target_dev and src.dtype == target_dtype:
            override_weight = src.clone()
        else:
            override_weight = src.to(target_dev, dtype=target_dtype)
    override_weight.requires_grad_(True)
    partial_grad_sum = torch.zeros_like(override_weight, dtype=torch.float32)
    # Measured second moment of this module's refresh gradient. Hooks stay
    # attached only for the batch loop below; `functional_call` swaps the weight
    # but still runs the module's own forward, so they fire normally.
    if grad_sq_accum is not None:
        grad_sq_accum.attach()
    partial_count = 0
    loss_sum = 0.0
    loss_sum_current = 0.0
    loss_sum_next = 0.0

    # Slide-window mixes in a "next-layer" version of the same loss.
    #   fisher MSE variants path: needs the next-layer fisher tensor.
    #   residual_kl          path: reuses fp_inps_final (same final-head target for
    #                              both current- and next-layer deltas).
    #   refined_residual_kl  path: reuses fp_inps_final AND needs the next-layer A
    #                              matrix (A_{i+1}) in place of A_i.
    has_next_refined_A = (
        next_refined_A_list is not None
        and any(a is not None for a in next_refined_A_list)
    )
    slide_active = (
        slide_alpha < 1.0
        and next_layer is not None
        and fp_inps_next is not None
        and (
            (is_fisher_mse_loss(refresh_loss_type) and next_layer_output_fisher is not None)
            or (refresh_loss_type == "residual_kl" and fp_inps_final is not None)
            or (
                refresh_loss_type == "refined_residual_kl"
                and fp_inps_final is not None
                and has_next_refined_A
            )
            or (
                # refined_mse: needs both the next-layer fisher (second order)
                # and the next-layer grad pool (first order). At layer 0 the
                # pool tensor is None but mean_grad is zeros(H), which is a
                # valid degenerate state (first_order collapses to 0).
                refresh_loss_type == "refined_mse"
                and next_layer_output_fisher is not None
                and next_refined_mse_mean_grad is not None
            )
        )
    )

    _lm_head_loss = refresh_loss_type in (
        "kl",
        "residual_kl",
        "refined_residual_kl",
        "refined_diag_residual_kl",
    )
    _step = bsz
    if refresh_mb is not None and refresh_mb > 0 and _lm_head_loss:
        _step = min(bsz, int(refresh_mb))

    # The paper specifies P95 clipping but not its population. The explicit
    # global mode below uses one threshold over the complete selected refresh
    # sample set; the historical/default local mode leaves both thresholds
    # None so compute_refresh_loss calculates one P95 per rank/backward chunk.
    # Every rank participates in the global prepass even when global shuffle
    # assigns it zero samples.
    if a_loss_clip_scope not in (
        "global_refresh",
        "local_backward_chunk",
    ):
        raise ValueError(
            "a_loss_clip_scope must be 'global_refresh' or "
            "'local_backward_chunk', "
            f"got {a_loss_clip_scope!r}."
        )
    a_loss_threshold = None
    next_a_loss_threshold = None
    if (
        a_loss_ratio < 1.0
        and a_loss_clip_scope == "global_refresh"
        and (
            is_fisher_backed_loss(refresh_loss_type)
            or is_hidden_mse_loss(refresh_loss_type)
        )
    ):
        local_abs_delta = []
        local_abs_next_delta = []
        with torch.no_grad():
            for start in range(0, len(selected_indices), _step):
                batch_indices = selected_indices[start : start + _step]
                batch_size = len(batch_indices)
                batch_attention_mask = attention_mask.expand(
                    batch_size, -1, -1, -1
                )
                batch_position_ids = position_ids.expand(batch_size, -1)
                batch_position_embeddings = (
                    position_embeddings[0].expand(batch_size, -1, -1),
                    position_embeddings[1].expand(batch_size, -1, -1),
                )
                out = functional_call(
                    layer,
                    {functional_weight_name: override_weight},
                    (inps[batch_indices].to(dev),),
                    {
                        "attention_mask": batch_attention_mask,
                        "position_ids": batch_position_ids,
                        "position_embeddings": batch_position_embeddings,
                    },
                    strict=False,
                )
                out_hidden = (
                    out[0] if isinstance(out, (tuple, list)) else out
                )
                fp_hidden = fp_inps[batch_indices].to(dev)
                delta_for_clip = out_hidden - fp_hidden
                if (
                    sink_size > 0
                    and delta_for_clip.shape[1] > sink_size
                ):
                    delta_for_clip = delta_for_clip[:, sink_size:]
                local_abs_delta.append(
                    delta_for_clip.float().abs().reshape(-1)
                )
                if slide_active:
                    next_out = next_layer(
                        out_hidden,
                        attention_mask=batch_attention_mask,
                        position_ids=batch_position_ids,
                        position_embeddings=batch_position_embeddings,
                    )
                    next_hidden = (
                        next_out[0]
                        if isinstance(next_out, (tuple, list))
                        else next_out
                    )
                    fp_next = fp_inps_next[batch_indices].to(dev)
                    next_delta_for_clip = next_hidden - fp_next
                    if (
                        sink_size > 0
                        and next_delta_for_clip.shape[1] > sink_size
                    ):
                        next_delta_for_clip = next_delta_for_clip[
                            :, sink_size:
                        ]
                    local_abs_next_delta.append(
                        next_delta_for_clip.float().abs().reshape(-1)
                    )
        empty_clip_values = torch.empty(
            0, dtype=torch.float32, device=override_weight.device
        )
        local_clip_values = (
            torch.cat(local_abs_delta)
            if local_abs_delta
            else empty_clip_values
        )
        a_loss_threshold = global_percentile(
            local_clip_values, float(a_loss_ratio)
        ).detach()
        if slide_active:
            local_next_clip_values = (
                torch.cat(local_abs_next_delta)
                if local_abs_next_delta
                else empty_clip_values
            )
            next_a_loss_threshold = global_percentile(
                local_next_clip_values, float(a_loss_ratio)
            ).detach()
        del local_abs_delta, local_abs_next_delta, local_clip_values

    if len(selected_indices) > 0:
        grad_modules = [layer]
        if refresh_loss_type in ("kl", "residual_kl", "refined_residual_kl"):
            # These losses also send grad through the final norm + lm_head,
            # so we must include them in `temporary_requires_grad` so their
            # param.requires_grad is forced to False for the duration of the
            # refresh (only `override_weight` should accumulate grad).
            grad_modules.extend(
                [analyzer.get_layernorm_before_head(), analyzer.get_lm_head()]
            )
        if slide_active:
            grad_modules.append(next_layer)
        with temporary_requires_grad(grad_modules, []):
            with torch.enable_grad():
                # Optional memory cap: losses that go through the LM head
                # (kl / residual_kl / refined_residual_kl / refined_diag_residual_kl)
                # materialise a full (B, T, V) logits tensor during forward.
                # When vocab is huge (Llama-3: V≈150K → ~20 GB at bsz=32, bf16)
                # we split the batch into `refresh_mb`-sized micro-batches; the
                # accumulation into `partial_grad_sum` is linear so the final
                # gradient is identical to using the full `bsz`. Losses that
                # don't hit the LM head (fisher MSE variants, hidden_mse) ignore the
                # cap — memory isn't their bottleneck.
                for start in range(0, len(selected_indices), _step):
                    with layer_recorder.section("layer.true_weight_grad.batch.total") if layer_recorder else _NULL_CONTEXT:
                        with layer_recorder.section("layer.true_weight_grad.batch.prepare") if layer_recorder else _NULL_CONTEXT:
                            batch_indices = selected_indices[start:start + _step]
                            batch_size = len(batch_indices)
                            batch_attention_mask = attention_mask.expand(batch_size, -1, -1, -1)
                            batch_position_ids = position_ids.expand(batch_size, -1)
                            batch_position_embeddings = (
                                position_embeddings[0].expand(batch_size, -1, -1),
                                position_embeddings[1].expand(batch_size, -1, -1),
                            )

                        with layer_recorder.section("layer.true_weight_grad.batch.forward") if layer_recorder else _NULL_CONTEXT:
                            out = functional_call(
                                layer,
                                {functional_weight_name: override_weight},
                                (inps[batch_indices].to(dev),),
                                {
                                    "attention_mask": batch_attention_mask,
                                    "position_ids": batch_position_ids,
                                    "position_embeddings": batch_position_embeddings,
                                },
                                strict=False,
                            )
                            out_hidden = out[0] if isinstance(out, (tuple, list)) else out
                        with layer_recorder.section("layer.true_weight_grad.batch.fp_hidden") if layer_recorder else _NULL_CONTEXT:
                            fp_hidden = fp_inps[batch_indices].to(dev)
                        with layer_recorder.section("layer.true_weight_grad.batch.fisher_slice") if layer_recorder else _NULL_CONTEXT:
                            fisher_batch = slice_layer_output_fisher_for_batch(
                                layer_output_fisher,
                                batch_indices,
                                dev,
                                refresh_loss_type,
                            )
                            fp_final_batch = (
                                None if fp_inps_final is None
                                else fp_inps_final[batch_indices].to(dev)
                            )
                        # Per-sample sub-A dispatch for refined_residual_kl. batches
                        # may freely cross sub-A boundaries; each sample picks its
                        # own A_{a_idx}. See `_pick_refined_A_for_batch` for the
                        # 2D-fast-path / 3D-bmm-fallback selection.
                        with layer_recorder.section("layer.true_weight_grad.batch.refined_A_build") if layer_recorder else _NULL_CONTEXT:
                            if sub_a_mode:
                                _shard_local = shard_size if shard_size is not None else inps.shape[0]
                                # `batch_indices` here is a list of rank-local indices
                                # into `inps`; we forward the first-sample offset and
                                # batch size to the helper, which rebuilds per-sample
                                # a_idx internally. That's fine because the helper only
                                # looks at contiguous batch_local_start+k offsets.
                                # We pre-stack implicitly by calling the helper with a
                                # proxy that wraps the non-contiguous batch_indices as
                                # an explicit list of locals.
                                _a_per_sample = torch.tensor(
                                    [(dp_rank * _shard_local + li) // samples_per_A for li in batch_indices],
                                    dtype=torch.long,
                                    device=dev,
                                )
                                _unique_a = torch.unique(_a_per_sample)
                                _ref_slot = next((s for s in refined_A_list if s is not None), None)
                                if _ref_slot is None:
                                    refined_A_batch = None
                                elif _unique_a.numel() == 1:
                                    refined_A_batch = refined_A_list[int(_unique_a.item())]
                                else:
                                    refined_A_batch = torch.stack(
                                        [
                                            refined_A_list[int(a)] if refined_A_list[int(a)] is not None else torch.zeros_like(_ref_slot)
                                            for a in _a_per_sample.tolist()
                                        ],
                                        dim=0,
                                    )
                                if next_refined_A_list is not None and any(s is not None for s in next_refined_A_list):
                                    _ref2 = next(s for s in next_refined_A_list if s is not None)
                                    if _unique_a.numel() == 1:
                                        next_refined_A_batch = next_refined_A_list[int(_unique_a.item())]
                                    else:
                                        next_refined_A_batch = torch.stack(
                                            [
                                                next_refined_A_list[int(a)] if next_refined_A_list[int(a)] is not None else torch.zeros_like(_ref2)
                                                for a in _a_per_sample.tolist()
                                            ],
                                            dim=0,
                                        )
                                else:
                                    next_refined_A_batch = None
                            else:
                                refined_A_batch = (
                                    refined_A_list[0]
                                    if refined_A_list is not None and len(refined_A_list) > 0
                                    else None
                                )
                                next_refined_A_batch = (
                                    next_refined_A_list[0]
                                    if next_refined_A_list is not None and len(next_refined_A_list) > 0
                                    else None
                                )
                        # refined_mse: pick per-batch exact-grad slice + positions.
                        # When loss_slide_window is active under refined_mse, the
                        # next-layer pool uses the SAME sample ids as the current-
                        # layer pool (they were captured in one backward pass), so
                        # `pool_positions` / `batch_row_list` are identical across
                        # current and next. We just index a second (B_pool, seq, H)
                        # slice from the next-layer pool here to avoid redoing the
                        # lookup inside the slide branch.
                        with layer_recorder.section("layer.true_weight_grad.batch.refined_mse_pool_lookup") if layer_recorder else _NULL_CONTEXT:
                            refined_mse_pool_positions = None
                            refined_mse_exact_batch = None
                            refined_mse_exact_batch_next = None
                            if refresh_loss_type == "refined_mse" and refined_mse_pool_lookup:
                                pool_pos_list = []
                                batch_row_list = []
                                for p_in_batch, li in enumerate(batch_indices):
                                    pp = refined_mse_pool_lookup.get(int(li))
                                    if pp is not None:
                                        pool_pos_list.append(pp)
                                        batch_row_list.append(p_in_batch)
                                if batch_row_list:
                                    refined_mse_exact_batch = (
                                        refined_mse_grad_pool[pool_pos_list]
                                        .to(dev, dtype=out_hidden.dtype)
                                    )
                                    refined_mse_pool_positions = torch.tensor(
                                        batch_row_list, dtype=torch.long, device=dev
                                    )
                                    if next_refined_mse_grad_pool is not None:
                                        refined_mse_exact_batch_next = (
                                            next_refined_mse_grad_pool[pool_pos_list]
                                            .to(dev, dtype=out_hidden.dtype)
                                        )
                        with layer_recorder.section("layer.true_weight_grad.batch.refresh_loss_current") if layer_recorder else _NULL_CONTEXT:
                            refresh_loss_current = compute_refresh_loss(
                                refresh_loss_type,
                                out_hidden,
                                fp_hidden,
                                analyzer,
                                kl_topk,
                                layer_output_fisher=fisher_batch,
                                fp_final_hidden=fp_final_batch,
                                refined_A=refined_A_batch,
                                layer_output_grad_exact=refined_mse_exact_batch,
                                layer_output_grad_mean=(
                                    refined_mse_mean_grad
                                    if refresh_loss_type == "refined_mse" else None
                                ),
                                pool_positions=refined_mse_pool_positions,
                                profile_recorder=layer_recorder,
                                sink_size=sink_size,
                                a_loss_ratio=a_loss_ratio,
                                a_loss_threshold=a_loss_threshold,
                            )
                        if slide_active:
                            with layer_recorder.section("layer.true_weight_grad.batch.slide_next_forward") if layer_recorder else _NULL_CONTEXT:
                                next_out = next_layer(
                                    out_hidden,
                                    attention_mask=batch_attention_mask,
                                    position_ids=batch_position_ids,
                                    position_embeddings=batch_position_embeddings,
                                )
                                next_out_hidden = next_out[0] if isinstance(next_out, (tuple, list)) else next_out
                                fp_hidden_next = fp_inps_next[batch_indices].to(dev)
                                # fisher MSE variants / refined_mse: next-layer
                                # loss needs the next-layer fisher tensor.
                                # residual_kl: doesn't — reuses fp_inps_final.
                                fisher_batch_next = slice_layer_output_fisher_for_batch(
                                    next_layer_output_fisher,
                                    batch_indices,
                                    dev,
                                    refresh_loss_type,
                                )
                            with layer_recorder.section("layer.true_weight_grad.batch.refresh_loss_next") if layer_recorder else _NULL_CONTEXT:
                                refresh_loss_next = compute_refresh_loss(
                                    refresh_loss_type,
                                    next_out_hidden,
                                    fp_hidden_next,
                                    analyzer,
                                    kl_topk,
                                    layer_output_fisher=fisher_batch_next,
                                    fp_final_hidden=fp_final_batch,
                                    refined_A=next_refined_A_batch,
                                    layer_output_grad_exact=(
                                        refined_mse_exact_batch_next
                                        if refresh_loss_type == "refined_mse" else None
                                    ),
                                    layer_output_grad_mean=(
                                        next_refined_mse_mean_grad
                                        if refresh_loss_type == "refined_mse" else None
                                    ),
                                    pool_positions=(
                                        refined_mse_pool_positions
                                        if refresh_loss_type == "refined_mse" else None
                                    ),
                                    profile_recorder=layer_recorder,
                                    sink_size=sink_size,
                                    a_loss_ratio=a_loss_ratio,
                                    a_loss_threshold=next_a_loss_threshold,
                                )
                            with layer_recorder.section("layer.true_weight_grad.batch.blend") if layer_recorder else _NULL_CONTEXT:
                                refresh_loss = (
                                    slide_alpha * refresh_loss_current
                                    + (1.0 - slide_alpha) * refresh_loss_next
                                )
                                loss_sum_current += _reduce_refresh_loss_for_aggregation(
                                    refresh_loss_type, refresh_loss_current, batch_size
                                )
                                loss_sum_next += _reduce_refresh_loss_for_aggregation(
                                    refresh_loss_type, refresh_loss_next, batch_size
                                )
                        else:
                            refresh_loss = refresh_loss_current

                        with layer_recorder.section("layer.true_weight_grad.batch.backward") if layer_recorder else _NULL_CONTEXT:
                            batch_grad = torch.autograd.grad(
                                refresh_loss, override_weight, retain_graph=False
                            )[0].float()
                        with layer_recorder.section("layer.true_weight_grad.batch.accumulate") if layer_recorder else _NULL_CONTEXT:
                            # `autograd.grad` returns ∂(mean_loss)/∂W. Multiply by batch
                            # size to recover a sum-over-samples gradient so per-rank
                            # partials aggregate with a plain allreduce_sum. All refresh
                            # losses, including legacy_fisher_diag_mse, use batch/token
                            # mean convention here.
                            partial_grad_sum.add_(batch_grad, alpha=float(batch_size))
                            loss_sum += _reduce_refresh_loss_for_aggregation(
                                refresh_loss_type, refresh_loss, batch_size
                            )
                            partial_count += batch_size
                        # Per-batch `cleanup_memory()` used to run here; removing
                        # it trades a small increase in peak transient memory for
                        # dropping ~5-20 ms/call of gc.collect + synchronize +
                        # empty_cache on thousands of refreshes. PyTorch reuses
                        # cached blocks, so this is safe as long as no outer
                        # autograd graph leaks across the loop.

    if grad_sq_accum is not None:
        grad_sq_accum.detach()

    extras = {"loss_sum": loss_sum}
    if slide_active:
        extras["loss_sum_current"] = loss_sum_current
        extras["loss_sum_next"] = loss_sum_next
        extras["slide_alpha"] = slide_alpha
    return partial_grad_sum, partial_count, loss_sum, extras


def collect_layer_grad_hessian_stats(
    *,
    model,
    layer,
    analyzer,
    full,
    names,
    inps,
    fp_inps,
    attention_mask,
    position_ids,
    position_embeddings,
    bsz,
    num_groups,
    kl_topk,
    grad_hessian_topk,
    dev,
    layer_idx,
    layer_refresh_loss_type,
    gptq_reference_loss_type,
    precomputed_saliency_dict=None,
    precomputed_layer_output_fisher=None,
    fp_inps_final=None,
    refined_A_list=None,
    samples_per_A=0,
    dp_rank=0,
    dp_shard_size=None,
    layer_recorder=None,
    skip_gradient_backward=False,
    sink_size=0,
    a_loss_ratio=1.0,
):
    layer_refresh_loss_type = canonical_refresh_loss_type(layer_refresh_loss_type)
    gptq_reference_loss_type = canonical_refresh_loss_type(gptq_reference_loss_type)
    need_saliency_collection = precomputed_saliency_dict is None
    # ``skip_gradient_backward`` disables only GPTQ+'s analytical first-order
    # reference term (alpha=0). Fisher-backed Block-GD remains independently
    # active and still needs its layer-output Fisher on the non-global path.
    need_layer_output_fisher_collection = (
        is_fisher_backed_loss(layer_refresh_loss_type)
        and precomputed_layer_output_fisher is None
    )
    need_gradient_backward = not skip_gradient_backward

    # Fast path: when every per-batch quantity is already in hand (saliency
    # precomputed by collect_static_end_to_end_saliency_and_fisher, fisher
    # either precomputed or unused, and the GPTQ+ reference gradient disabled
    # via --enable_gptq_plus 0), the entire forward-loop in this function
    # produces nothing the quantiser will read — every batch would just run a
    # wasteful FP forward, load fp_hidden from CPU that nobody consumes, then
    # discard. Skip the loop and return stubs directly. This removes the
    # `grad_hessian.batch.*` NVTX ranges entirely, along with the allreduce of
    # `mean_reference_loss` (which would otherwise fire with count=0 on every
    # rank and still incur one NCCL sync). Matches the pre-fast-path results
    # exactly: gradients_dict is zeros (GPTQ+ first-order is off, so they get
    # zeroed by the slow path too), mean_reference_loss=0 (reference_losses
    # was empty), saliency/fisher are pass-throughs.
    if not (need_saliency_collection or need_layer_output_fisher_collection or need_gradient_backward):
        with layer_recorder.section("layer.grad_hessian.fast_path") if layer_recorder else _NULL_CONTEXT:
            gradients_dict = {}
            for name in names:
                _, module = resolve_quant_module(full, name)
                gradients_dict[name] = torch.zeros_like(module.weight.data, dtype=torch.float32)
            return (
                precomputed_saliency_dict,
                gradients_dict,
                0.0,
                precomputed_layer_output_fisher,
            )

    if need_layer_output_fisher_collection:
        precomputed_layer_output_fisher = collect_layer_output_fisher_only(
            model=model,
            layer=layer,
            analyzer=analyzer,
            inps=inps,
            fp_inps=fp_inps,
            batch_attention_mask=attention_mask.expand(bsz, -1, -1, -1),
            batch_position_ids=position_ids.expand(bsz, -1),
            batch_position_embeddings=(
                position_embeddings[0].expand(bsz, -1, -1),
                position_embeddings[1].expand(bsz, -1, -1),
            ),
            bsz=bsz,
            kl_topk=kl_topk,
            grad_hessian_topk=grad_hessian_topk,
            dev=dev,
            layer_idx=layer_idx,
            layer_recorder=layer_recorder,
            sink_size=sink_size,
            legacy_diag=(layer_refresh_loss_type == "legacy_fisher_diag_mse"),
        )
        need_layer_output_fisher_collection = False

    need_output_head = (
        need_saliency_collection
        or (need_gradient_backward and (
            layer_refresh_loss_type == "kl"
            or gptq_reference_loss_type == "kl"
        ))
    )
    with torch.enable_grad():
        saliency_cache = None
        if need_saliency_collection:
            saliency_cache = SaliencyCache(names, num_groups, sink_size=sink_size)
            saliency_cache.add_hook(full, enable=False)
        gradients_cache = GradientCache(names, num_groups)
        gradients_cache.add_hook(full, enable=False)
        reference_losses = []

        for j in tqdm(
            range(0, inps.shape[0], bsz),
            ncols=120,
            desc=f"Layer {layer_idx} Computing Gradients and Hessians",
            position=1,
            leave=False,
        ):
            batch_size = min(bsz, inps.shape[0] - j)
            batch_attention_mask = attention_mask.expand(batch_size, -1, -1, -1)
            batch_position_ids = position_ids.expand(batch_size, -1)
            batch_position_embeddings = (
                position_embeddings[0].expand(batch_size, -1, -1),
                position_embeddings[1].expand(batch_size, -1, -1),
            )
            with layer_recorder.section("layer.grad_hessian.batch.total") if layer_recorder else _NULL_CONTEXT:
                with layer_recorder.section("layer.grad_hessian.forward") if layer_recorder else _NULL_CONTEXT:
                    with layer_recorder.section("layer.grad_hessian.forward.layer") if layer_recorder else _NULL_CONTEXT:
                        out = layer(
                            inps[j : j + bsz].to(dev),
                            attention_mask=batch_attention_mask,
                            position_ids=batch_position_ids,
                            position_embeddings=batch_position_embeddings,
                        )
                    with layer_recorder.section("layer.grad_hessian.forward.hidden_extract") if layer_recorder else _NULL_CONTEXT:
                        out_hidden = out[0] if isinstance(out, (tuple, list)) else out
                    with layer_recorder.section("layer.grad_hessian.forward.fp_hidden") if layer_recorder else _NULL_CONTEXT:
                        fp_hidden = fp_inps[j : j + bsz].to(dev)
                    logits = None
                    logits_fp = None
                    grad_hessian_logits = None
                    grad_hessian_logits_fp = None
                    if need_output_head:
                        with layer_recorder.section("layer.grad_hessian.forward.logits_quant") if layer_recorder else _NULL_CONTEXT:
                            logits = hidden2logits(out_hidden, analyzer)
                        with layer_recorder.section("layer.grad_hessian.forward.logits_fp") if layer_recorder else _NULL_CONTEXT:
                            logits_fp = hidden2logits(fp_hidden, analyzer)
                        grad_hessian_logits = logits
                        grad_hessian_logits_fp = logits_fp
                        if grad_hessian_topk > 0:
                            with layer_recorder.section("layer.grad_hessian.forward.topk_slice") if layer_recorder else _NULL_CONTEXT:
                                grad_hessian_logits_fp, grad_hessian_indices = logits_fp.topk(
                                    grad_hessian_topk,
                                    dim=-1,
                                    sorted=False,
                                )
                                grad_hessian_logits = logits.gather(-1, grad_hessian_indices)
                    if need_saliency_collection:
                        with layer_recorder.section("layer.grad_hessian.forward.label_sample") if layer_recorder else _NULL_CONTEXT:
                            # Per-sample deterministic labels: each sample's
                            # label draw only depends on its global sample id,
                            # so 1-GPU and N-GPU DP produce identical labels for
                            # the same sample regardless of how samples are
                            # grouped into batches across ranks.
                            _dp_rank = dist_utils.get_rank()
                            _n_local = inps.shape[0]
                            _batch_bsz = grad_hessian_logits_fp.shape[0]
                            _global_indices = [
                                _dp_rank * _n_local + j + _i
                                for _i in range(_batch_bsz)
                            ]
                            labels = _deterministic_categorical_labels(
                                grad_hessian_logits_fp,
                                _global_indices,
                                base_seed=layer_idx * 131 + 7,
                            )
                        with layer_recorder.section("layer.grad_hessian.forward.nll_build") if layer_recorder else _NULL_CONTEXT:
                            ghl_for_nll = grad_hessian_logits
                            labels_for_nll = labels
                            if sink_size > 0 and ghl_for_nll.shape[1] > sink_size:
                                ghl_for_nll = ghl_for_nll[:, sink_size:]
                                labels_for_nll = labels_for_nll[:, sink_size:]
                            nll_loss = F.cross_entropy(
                                ghl_for_nll.reshape(-1, ghl_for_nll.size(-1)),
                                labels_for_nll.reshape(-1),
                                reduction="sum",
                            )

                if need_saliency_collection:
                    with layer_recorder.section("layer.grad_hessian.saliency_backward.total") if layer_recorder else _NULL_CONTEXT:
                        saliency_cache.enable_hooks()
                        with layer_recorder.section("layer.grad_hessian.saliency_backward.zero_grad") if layer_recorder else _NULL_CONTEXT:
                            model.zero_grad()
                        with layer_recorder.section("layer.grad_hessian.saliency_backward.backward") if layer_recorder else _NULL_CONTEXT:
                            nll_loss.backward(retain_graph=True)
                        saliency_cache.disable_hooks()

                batch_layer_output_fisher = None
                if is_fisher_backed_loss(layer_refresh_loss_type) and precomputed_layer_output_fisher is not None:
                    with layer_recorder.section("layer.grad_hessian.fisher_slice") if layer_recorder else _NULL_CONTEXT:
                        batch_layer_output_fisher = slice_layer_output_fisher_for_batch(
                            precomputed_layer_output_fisher,
                            list(range(j, j + batch_size)),
                            dev,
                            layer_refresh_loss_type,
                        )
                if need_gradient_backward:
                    with layer_recorder.section("layer.grad_hessian.gradient_loss_build") if layer_recorder else _NULL_CONTEXT:
                        if gptq_reference_loss_type == "kl":
                            kl_logits = grad_hessian_logits if grad_hessian_topk > 0 else logits
                            kl_logits_fp = grad_hessian_logits_fp if grad_hessian_topk > 0 else logits_fp
                            if grad_hessian_topk <= 0 and kl_topk > 0:
                                with layer_recorder.section("layer.grad_hessian.gradient_loss_build.topk") if layer_recorder else _NULL_CONTEXT:
                                    kl_logits_fp, indices = logits_fp.topk(kl_topk, dim=-1, sorted=False)
                                    kl_logits = logits.gather(-1, indices)
                            if sink_size > 0 and kl_logits.shape[1] > sink_size:
                                kl_logits = kl_logits[:, sink_size:]
                                kl_logits_fp = kl_logits_fp[:, sink_size:]
                            gradient_loss = tokenwise_kl_from_logits(
                                kl_logits, kl_logits_fp
                            ).mean()
                        else:
                            fp_final_batch = (
                                None if fp_inps_final is None
                                else fp_inps_final[j : j + bsz].to(dev)
                            )
                            # Per-sample sub-A dispatch for refined_residual_kl.
                            # Each sample picks its own A by global sample id;
                            # batches may freely cross sub-A boundaries.
                            refined_A_batch = _pick_refined_A_for_batch(
                                refined_A_list=refined_A_list,
                                samples_per_A=samples_per_A,
                                loss_type=gptq_reference_loss_type,
                                batch_local_start=j,
                                batch_size_local=min(bsz, inps.shape[0] - j),
                                dp_rank=dp_rank,
                                dp_shard_size=dp_shard_size if dp_shard_size is not None else inps.shape[0],
                                dev=dev,
                            )
                            gradient_loss = compute_refresh_loss(
                                gptq_reference_loss_type,
                                out_hidden,
                                fp_hidden,
                                analyzer,
                                kl_topk,
                                layer_output_fisher=batch_layer_output_fisher,
                                fp_final_hidden=fp_final_batch,
                                refined_A=refined_A_batch,
                                profile_recorder=layer_recorder,
                                sink_size=sink_size,
                                a_loss_ratio=a_loss_ratio,
                            )

                    with layer_recorder.section("layer.grad_hessian.gradient_backward.total") if layer_recorder else _NULL_CONTEXT:
                        gradients_cache.enable_hooks()
                        with layer_recorder.section("layer.grad_hessian.gradient_backward.zero_grad") if layer_recorder else _NULL_CONTEXT:
                            model.zero_grad()
                        with layer_recorder.section("layer.grad_hessian.gradient_backward.backward") if layer_recorder else _NULL_CONTEXT:
                            gradient_loss.backward()
                        gradients_cache.disable_hooks()

                    with layer_recorder.section("layer.grad_hessian.metrics_record") if layer_recorder else _NULL_CONTEXT:
                        reference_losses.append(gradient_loss.item())
                # Per-batch cleanup_memory() used to run here. Removed: PyTorch
                # frees autograd state automatically after gradient_loss.backward()
                # and the heavy gc+synchronize+empty_cache was a significant
                # fraction of the loop time.

        # DP: rank-local running mean already covers per-sample contributions.
        # We combine across ranks into the global mean via (sum, count)
        # allreduce of the reference_losses vector below.
        local_sum_ref_loss = sum(reference_losses)
        local_count_ref_loss = len(reference_losses)
        mean_reference_loss = dist_utils.allreduce_mean_scalar(
            local_sum_ref_loss / max(local_count_ref_loss, 1),
            count=local_count_ref_loss,
        )

    if saliency_cache is not None:
        saliency_cache.clear_hook()
    gradients_cache.clear_hook()
    # Finalise the per-module gradient sums: all-reduce across ranks and divide
    # by the global batch count to realise the same mean as the pre-DP
    # running-mean hook. Skipped when the GPTQ+ one-order term is bypassed —
    # downstream consumers get zero-shaped gradients instead.
    if need_gradient_backward:
        gradients_cache.finalize()
    else:
        for name in gradients_cache.names:
            _, module = resolve_quant_module(full, name)
            gradients_cache.gradients_cache[name] = torch.zeros_like(
                module.weight.data, dtype=torch.float32
            )
    layer_output_fisher = precomputed_layer_output_fisher
    with layer_recorder.section("layer.cache_finalize") if layer_recorder else _NULL_CONTEXT:
        if saliency_cache is not None:
            for name in saliency_cache.names:
                # Saliency stays sharded (rank-local); GPTQPlus only needs its
                # own rank's slice for add_batch, which also runs on the rank's
                # inps shard.
                saliency_cache.saliency_cache[name] = torch.cat(saliency_cache.saliency_cache[name], dim=0)
        saliency_dict = precomputed_saliency_dict if precomputed_saliency_dict is not None else saliency_cache.saliency_cache
        gradients_dict = gradients_cache.gradients_cache

    return saliency_dict, gradients_dict, mean_reference_loss, layer_output_fisher


def run_pre_quant_gd(
    *,
    layer_idx,
    layer,
    analyzer,
    full,
    inps,
    fp_inps,
    attention_mask,
    position_ids,
    position_embeddings,
    backward_bsz,
    kl_topk,
    dev,
    module_names,
    scheduler,
    num_steps,
    grad_lr,
    grad_optimizer,
    grad_clip,
    refresh_loss_type,
    layer_output_fisher_by_module,
    fp_inps_final=None,
    refined_A_list=None,
    samples_per_A=0,
    global_shuffle=False,
    dp_rank=0,
    shard_size=None,
    refresh_mb=None,
    refined_mse_pool_ids=None,
    refined_mse_grad_pool=None,
    refined_mse_mean_grad=None,
    layer_recorder=None,
    sink_size=0,
    a_loss_ratio=1.0,
    a_loss_clip_scope="local_backward_chunk",
):
    if num_steps <= 0 or not module_names:
        return

    with layer_recorder.section("layer.pre_quant_gd.setup") if layer_recorder else _NULL_CONTEXT:
        modules = []
        for module_name in module_names:
            _, module = resolve_quant_module(full, module_name)
            modules.append((module_name, module))

        optimizer_states = {}
        if grad_optimizer == "adam":
            for module_name, module in modules:
                optimizer_states[module_name] = {
                    "step": 0,
                    "exp_avg": torch.zeros_like(module.weight.data, dtype=torch.float32),
                    "exp_avg_sq": torch.zeros_like(module.weight.data, dtype=torch.float32),
                }

    world = dist_utils.get_world_size()
    for step_idx in range(num_steps):
        with layer_recorder.section("layer.pre_quant_gd.step.total") if layer_recorder else _NULL_CONTEXT:
            with layer_recorder.section("layer.pre_quant_gd.step.sample_pick") if layer_recorder else _NULL_CONTEXT:
                sample_indices = scheduler.next_indices()
            step_losses = []
            step_update_abs = []
            for module_name, module in modules:
                with layer_recorder.section("layer.pre_quant_gd.module.total") if layer_recorder else _NULL_CONTEXT:
                    fisher_tensor = layer_output_fisher_by_module.get(module_name)
                    with layer_recorder.section("layer.pre_quant_gd.module.collect_grad") if layer_recorder else _NULL_CONTEXT:
                        partial_grad_sum, partial_count, loss_sum, _extras = (
                            collect_true_weight_gradient(
                                layer=layer,
                                analyzer=analyzer,
                                module_name=module_name,
                                full=full,
                                inps=inps,
                                fp_inps=fp_inps,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                position_embeddings=position_embeddings,
                                bsz=backward_bsz,
                                kl_topk=kl_topk,
                                dev=dev,
                                weight_override=module.weight.data.float(),
                                sample_indices=sample_indices,
                                refresh_loss_type=refresh_loss_type,
                                layer_output_fisher=fisher_tensor,
                                fp_inps_final=fp_inps_final,
                                refined_A_list=refined_A_list,
                                samples_per_A=samples_per_A,
                                global_shuffle=global_shuffle,
                                dp_rank=dp_rank,
                                shard_size=shard_size if shard_size is not None else inps.shape[0],
                                refresh_mb=refresh_mb,
                                refined_mse_pool_ids=refined_mse_pool_ids,
                                refined_mse_grad_pool=refined_mse_grad_pool,
                                refined_mse_mean_grad=refined_mse_mean_grad,
                                layer_recorder=layer_recorder,
                                sink_size=sink_size,
                                a_loss_ratio=a_loss_ratio,
                                a_loss_clip_scope=a_loss_clip_scope,
                            )
                        )
                    # DP aggregation — packed into one allreduce (grad tensor +
                    # scalars) to avoid 3 back-to-back NCCL collectives per
                    # module per pre-GD step.
                    with layer_recorder.section("layer.pre_quant_gd.module.allreduce") if layer_recorder else _NULL_CONTEXT:
                        if world > 1:
                            grad_flat = partial_grad_sum.reshape(-1)
                            scalar_vec = torch.tensor(
                                [float(partial_count), float(loss_sum)],
                                dtype=grad_flat.dtype, device=grad_flat.device,
                            )
                            packed = torch.cat([grad_flat, scalar_vec])
                            dist_utils.allreduce_sum_(packed)
                            partial_grad_sum.copy_(packed[:grad_flat.numel()].view_as(partial_grad_sum))
                            global_count = int(packed[grad_flat.numel()].item())
                            global_loss_sum = packed[grad_flat.numel() + 1].item()
                        else:
                            global_count = partial_count
                            global_loss_sum = loss_sum
                    if global_count <= 0:
                        raise RuntimeError(
                            f"pre-gd refresh produced zero samples across all ranks "
                            f"(layer={layer_idx}, module={module_name})."
                        )
                    grad = partial_grad_sum / float(global_count)
                    mean_loss = global_loss_sum / float(global_count)
                    with layer_recorder.section("layer.pre_quant_gd.module.optimizer_step") if layer_recorder else _NULL_CONTEXT:
                        update, next_state = apply_dense_optimizer_step(
                            module.weight.data,
                            grad,
                            lr=grad_lr,
                            optimizer=grad_optimizer,
                            opt_state=optimizer_states.get(module_name),
                            grad_clip=grad_clip,
                        )
                    if next_state is not None:
                        optimizer_states[module_name] = next_state
                    step_losses.append(mean_loss)
                    if update.numel() > 0:
                        step_update_abs.append(update.float().abs().mean())
            mean_step_loss = sum(step_losses) / len(step_losses) if step_losses else None
            mean_step_update = (
                torch.stack(step_update_abs).mean().item()
                if step_update_abs else None
            )
        logging.info(
            "pre-gd layer=%d step=%d/%d samples=%s loss=%s update_abs=%s optimizer=%s lr=%s",
            layer_idx,
            step_idx + 1,
            num_steps,
            sample_indices,
            format_log_value(mean_step_loss, digits=6),
            format_log_value(mean_step_update, digits=6),
            grad_optimizer,
            format_log_value(grad_lr, digits=6),
        )


@torch.no_grad()
def gptq_fwrd(args, analyzer: model_utils.ModelAnalyzer, dataloader, dev):
    """
    From GPTQ repo
    """
    logging.info("-----GPTQPlus Quantization-----")
    args.grad_refresh_loss = canonical_refresh_loss_type(args.grad_refresh_loss)

    # Guard: `--fsdp_precompute` leaves the model with DTensor-wrapped params
    # after precompute (in-process unwrap was too fragile). The supported
    # workflow is two-stage. Fail fast if the caller forgot to set
    # `--exit_after_precompute` (the sweep script auto-wires both).
    if bool(getattr(args, "fsdp_precompute", False)) and not bool(getattr(args, "exit_after_precompute", False)):
        raise RuntimeError(
            "--fsdp_precompute requires --exit_after_precompute. Two-stage workflow: "
            "(1) run precompute under FSDP with --fsdp_precompute --exit_after_precompute "
            "--static_cache_path <DIR> (writes cache, exits); "
            "(2) rerun without --fsdp_precompute but with the same --static_cache_path "
            "(reads cache, quantises). The sweep script automates this when "
            "FSDP_PRECOMPUTE=1."
        )

    model = analyzer.model
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = analyzer.get_layers()
    if bool(getattr(model, "_gptqplus_stage2_cpu_master", False)) and not dist_utils.is_main():
        orig_device = torch.device("meta")
    else:
        orig_device = next(model.parameters()).device
    stage2_cpu_master = bool(getattr(model, "_gptqplus_stage2_cpu_master", False))
    global_loss_enabled = bool(getattr(args, "global_loss", False))
    if stage2_cpu_master:
        if args.grad_refresh_loss == "refined_mse":
            raise RuntimeError(
                "stage2_cpu_master does not support refined_mse because it materializes "
                "all downstream layers for a per-layer backward."
            )
        if args.load_qmodel_path:
            raise RuntimeError("stage2_cpu_master does not support load_qmodel_path yet.")
        if not global_loss_enabled:
            raise RuntimeError(
                "stage2_cpu_master requires --global_loss with an existing static cache. "
                "The non-global path would collect per-layer end-to-end stats from the "
                "partially materialized model."
            )
        logging.info(
            "Stage 2 CPU-master quantization enabled: rank0 owns the CPU model; "
            "layers are broadcast to rank-local GPUs on demand."
        )
    layer_manager = Stage2CpuMasterLayerManager(analyzer, dev, layers)
    analysis_hook = getattr(args, "_analysis_hook", None)
    analysis_need_fisher = bool(getattr(args, "_analysis_collect_fisher", False))
    analysis_need_legacy_fisher_diag = bool(
        getattr(args, "_analysis_collect_legacy_fisher_diag", False)
    )
    analysis_collect_refined_rkl = bool(
        getattr(args, "_analysis_collect_refined_rkl", False)
    )
    analysis_collect_refined_diag_rkl = bool(
        getattr(args, "_analysis_collect_refined_diag_rkl", False)
    )
    analysis_need_fp_inps_final = bool(
        getattr(args, "_analysis_need_fp_inps_final", False)
    )

    quant_profile_enabled = getattr(args, "enable_quant_profile", False)
    run_recorder = QuantProfileRecorder(dev) if quant_profile_enabled else None
    pipeline_recorder = QuantProfileRecorder(dev) if quant_profile_enabled else None
    target_layers = (
        set(parse_quant_profile_target_layers(getattr(args, "quant_profile_target_layers", "all"), len(layers)))
        if quant_profile_enabled else set()
    )
    parsed_target_modules = (
        parse_quant_profile_target_modules(getattr(args, "quant_profile_target_modules", "all"))
        if quant_profile_enabled else []
    )
    target_modules = set(parsed_target_modules)
    quant_stop_layer = parse_quant_stop_layer(getattr(args, "quant_stop_layer", None), len(layers))
    if quant_stop_layer is not None:
        logging.info("Quantization will stop after transformer layer %d.", quant_stop_layer)
    preclip_enabled = bool(args.w_clip and getattr(args, "pre_clip", True))
    enable_gptq_plus = bool(getattr(args, "enable_gptq_plus", 1))
    # Resolve once and thread through every loss/grad-collecting call site.
    # When --ignore_attention_sink is off this is 0 and every site short-
    # circuits the slicing branches (no numeric change vs the legacy code).
    sink_size = (
        int(getattr(args, "attention_sink_size", 256))
        if bool(getattr(args, "ignore_attention_sink", False))
        else 0
    )
    if sink_size > 0:
        logging.info(
            "ignore_attention_sink: dropping the first %d tokens of every loss "
            "(NLL/KL/MSE) at static-precompute and per-layer stats; sink tokens "
            "still flow through forward/KV.", sink_size,
        )
    if not enable_gptq_plus:
        # Semantic of enable_gptq_plus=0: fully equivalent to setting alpha=0.
        # Only the GPTQ+ first-order term (GHinv / Z / beta inside fasterquant's
        # inner + outer updates) is switched off — everything else (block_gd
        # refresh, loss_slide_window, pre_gd_steps, fisher precompute,
        # fp_inps_final) keeps running as the user configured.
        if args.alpha != 0:
            logging.info(
                "enable_gptq_plus=0 → overriding alpha %s -> 0 (GPTQ+ first-order disabled)",
                args.alpha,
            )
            args.alpha = 0
    # When alpha=0 (set either directly or via enable_gptq_plus=0) the reference
    # loss gradient collected in stats never survives `beta=0` in fasterquant,
    # so we can skip that second backward entirely. Pure optimisation — no
    # numerical change.
    skip_ref_backward = args.alpha == 0
    effective_pre_gd_steps = args.pre_gd_steps if preclip_enabled else 0
    if (
        (int(getattr(args, "fisher_rademacher_k", 0)) > 0
         or int(getattr(args, "num_samples_for_grad", 0)) > 0)
        and not global_loss_enabled
    ):
        raise ValueError(
            "--fisher_rademacher_k / --num_samples_for_grad only apply to the "
            "static end-to-end precompute path, so they require --global_loss."
        )
    if (
        args.grad_refresh_loss == "legacy_fisher_diag_mse"
        or analysis_need_legacy_fisher_diag
    ):
        if not global_loss_enabled:
            raise ValueError(
                "legacy_fisher_diag_mse requires --global_loss because its "
                "per-token Fisher diagonal is collected during static "
                "end-to-end NLL precompute."
            )
        if int(getattr(args, "fisher_rademacher_k", 0)) > 0:
            raise ValueError(
                "legacy_fisher_diag_mse stores per-token g^2 from a "
                "sum-reduced NLL backward, so it cannot be used with "
                "--fisher_rademacher_k > 0."
            )
        if int(getattr(args, "num_samples_for_grad", 0)) > 0:
            raise ValueError(
                "legacy_fisher_diag_mse needs per-token Fisher diagonals for "
                "all calibration samples, so it cannot be used with "
                "--num_samples_for_grad > 0."
            )
    act_quant_aware_gptq = bool(getattr(args, "act_quant_aware_gptq", False))
    k_cache_quant_aware_gptq = bool(getattr(args, "k_cache_quant_aware_gptq", False))
    if act_quant_aware_gptq and args.grad_refresh_loss == "refined_mse":
        raise ValueError(
            "--act_quant_aware_gptq does not support refined_mse yet; "
            "refined_mse's grad-pool path still assumes an FP student path."
        )
    if k_cache_quant_aware_gptq:
        if args.k_bits >= 16:
            raise ValueError("--k_cache_quant_aware_gptq requires --k_bits < 16.")
        if args.grad_refresh_loss == "refined_mse":
            raise ValueError(
                "--k_cache_quant_aware_gptq does not support refined_mse yet; "
                "refined_mse's grad-pool path still assumes an FP student path."
            )
    if act_quant_aware_gptq:
        logging.info(
            "act_quant_aware_gptq enabled: GPTQ+ student paths will use A/V fake "
            "quantization after FP teacher caches are captured (a_bits=%d, v_bits=%d).",
            args.a_bits,
            args.v_bits,
        )
    if k_cache_quant_aware_gptq:
        logging.info(
            "k_cache_quant_aware_gptq enabled: GPTQ+ student paths will use online "
            "K fake quantization after RoPE/QK rotation (k_bits=%d, k_groupsize=%d). "
            "FP teacher paths keep K quantization disabled.",
            args.k_bits,
            args.k_groupsize,
        )
    # refined_mix mode: layers [0, split) use refined_mse, [split, N-1) use
    # refined_residual_kl, final layer always kl. `split` is resolved here
    # once so the per-layer helper calls and the precompute layer-id sets
    # see a single consistent value.
    mix_mode = args.grad_refresh_loss == "refined_mix"
    if mix_mode:
        split_override = getattr(args, "refined_mix_split_layer", None)
        refined_mix_split_layer = (
            int(split_override) if split_override is not None else len(layers) // 2
        )
        if not (1 <= refined_mix_split_layer <= len(layers) - 1):
            raise ValueError(
                f"refined_mix_split_layer ({refined_mix_split_layer}) must be in "
                f"[1, {len(layers) - 1}] so each half has at least one non-final layer."
            )
        logging.info(
            "refined_mix enabled: layers [0, %d) use fisher_diag_mse, [%d, %d) use "
            "refined_residual_kl, layer %d uses kl (final). rkl_lr_ratio=%.4f.",
            refined_mix_split_layer, refined_mix_split_layer, len(layers) - 1,
            len(layers) - 1,
            float(getattr(args, "refined_mix_rkl_lr_ratio", 1.0)),
        )
    else:
        refined_mix_split_layer = None
    # Per-sub-A sample window for refined_residual_kl (num_A > 1). Computed
    # once here so the per-layer callers (run_pre_quant_gd,
    # collect_layer_grad_hessian_stats, refresh_fn) can route samples to the
    # right sub-A. Zero means single-A (no per-sample routing). Also activates
    # under refined_mix, because the back half uses refined_residual_kl.
    refined_rkl_num_A = int(getattr(args, "refined_rkl_num_A", 1))
    needs_refined_rkl_partition = (
        args.grad_refresh_loss in ("refined_residual_kl", "refined_mix")
        or analysis_collect_refined_rkl
        or analysis_collect_refined_diag_rkl
    )
    if refined_rkl_num_A > 1 and needs_refined_rkl_partition:
        if args.nsamples % refined_rkl_num_A != 0:
            raise ValueError(
                f"refined_rkl: nsamples ({args.nsamples}) must be divisible by "
                f"refined_rkl_num_A ({refined_rkl_num_A})."
            )
        refined_rkl_samples_per_A = args.nsamples // refined_rkl_num_A
    else:
        refined_rkl_samples_per_A = 0
    if args.pre_gd_steps > 0 and not preclip_enabled:
        logging.info(
            "Pre-quantization GD is disabled because preclip is off (w_clip=%s, pre_clip=%s).",
            args.w_clip,
            getattr(args, "pre_clip", True),
        )

    def should_profile_module(layer_idx, module_name):
        if not quant_profile_enabled:
            return False
        if layer_idx not in target_layers:
            return False
        return not target_modules or module_name in target_modules

    # Diagnostic dumper: activates for (layer_idx, module_substring) pairs listed
    # in `args.diagnose_targets` (e.g. "2:mlp.down_proj,1:mlp.down_proj"). No-op
    # when the flag isn't set. Writes raw matrices under `args.diagnose_dir`.
    _diag_targets = parse_diagnose_targets(getattr(args, "diagnose_targets", None))
    _diag_root = getattr(args, "diagnose_dir", None)
    if _diag_targets and not _diag_root:
        _diag_root = os.path.join(args.output_dir, "diagnostics")
    diagnostic_registry = DiagnosticRegistry(
        root_dir=_diag_root if _diag_targets else None,
        targets=_diag_targets,
        spike_ratio=float(getattr(args, "diagnose_spike_ratio", 3.0)),
    )
    if _diag_targets:
        logging.info(
            "Diagnostic dump enabled for targets=%s under %s",
            _diag_targets,
            _diag_root,
        )

    alignment_trace_path = getattr(args, "alignment_trace_path", None)
    alignment_trace_config = {}
    if alignment_trace_path is not None:
        alignment_trace_config = default_refresh_trace_config(args)
        alignment_trace_config.update(
            {
                "global_loss": bool(getattr(args, "global_loss", False)),
                "grad_refresh_loss": args.grad_refresh_loss,
                "g_update_mode": args.g_update_mode,
                "grad_optimizer": args.grad_optimizer,
                "final_layer_grad_optimizer": (
                    args.final_layer_grad_optimizer or args.grad_optimizer
                ),
                "analytical_first_order_enabled": bool(args.alpha != 0),
                "second_order_scale": float(args.second_order_scale),
                "block_atomic_quant": bool(args.block_atomic_quant),
                # REAL-Q has no separate destructive pre-clamp stage.  Its
                # w_clip flag controls only quantizer parameter search, so an
                # alignment run must explicitly use legacy --no_pre_clip.
                "pre_clip": bool(getattr(args, "pre_clip", True)),
                "dp_global_shuffle": bool(
                    getattr(args, "dp_global_shuffle", False)
                ),
            }
        )
    with (
        run_recorder.section("run.total") if run_recorder else _NULL_CONTEXT,
        RefreshTraceWriter(
            alignment_trace_path,
            implementation="legacy",
            run_id=getattr(args, "alignment_run_id", "default"),
            config=alignment_trace_config,
        ) as trace_writer,
    ):
        # `fp_inps_final` is the rank-local output of the LAST transformer
        # block, needed as the teacher hidden for residual_kl /
        # refined_residual_kl / refined_mse refresh losses. We capture it as
        # a by-product of the static saliency/fisher forward pass to avoid a
        # separate bs=1 Stage 2 precompute later. Populated on CPU in bf16;
        # moved to match fp_inps's device/dtype once `inps` is captured below.
        need_fp_inps_final = analysis_need_fp_inps_final or args.grad_refresh_loss in (
            "residual_kl", "refined_residual_kl", "refined_mse", "refined_mix",
        )
        fp_inps_final_cpu = None
        if global_loss_enabled:
            # Optional disk cache. The precompute result depends only on:
            #   model / dataset / nsamples / seq_len / concrete rotation /
            #   num_groups / grad_hessian_topk /
            #   global_loss_bsz / calibration seed. Optional Rademacher
            #   statistics additionally depend on refresh_seed.
            # Use `--static_cache_path DIR` to persist. Each rank writes/reads
            # its own shard file since saliency/fisher are rank-local.
            static_cache_dir = getattr(args, "static_cache_path", None)
            static_cache_key = None
            if static_cache_dir is not None:
                dataset_id = getattr(args, "dataset", "unknown")
                model_identity_tag = model_utils.source_model_cache_identity(args)[:12]
                rotate_flag = int(bool(getattr(args, "rotate", False)))
                rotation_identity_tag = model_utils.rotation_cache_tag(args)
                sal_clip_pct = getattr(args, "saliency_clip_percentile", 0.99)
                sal_clip_tag = f"{sal_clip_pct:.4f}".rstrip("0").rstrip(".")
                rkl_na = int(getattr(args, "refined_rkl_num_A", 1))
                # `fpfinal` flag distinguishes caches built WITH the by-product
                # fp_inps_final capture from older ones. Otherwise an old cache
                # would load successfully but miss the fp_inps_final tensor,
                # forcing the slow Stage 2 bs=1 precompute anyway.
                fpfinal_tag = int(need_fp_inps_final)
                # refined_mix: the precompute only collects fisher for the front
                # half and A for the back half, so the resulting cache is NOT
                # interchangeable with a non-mix run. Tag the key with the split
                # layer to keep them segregated. Non-mix runs keep the key
                # byte-identical to previous versions (empty tag).
                mix_tag = f"_mixsplit{refined_mix_split_layer}" if mix_mode else ""
                # dynsal tag: when --enable_dynamic_saliency=1, key the cache by
                # R and EVD threshold so sweeps over R don't collide. Disabled
                # (=0, default) path emits an empty tag and stays byte-identical
                # to pre-dynsal caches.
                dynsal_enabled = bool(int(getattr(args, "enable_dynamic_saliency", 0)))
                if dynsal_enabled:
                    dynsal_R = int(getattr(args, "dyn_sal_rank", 16))
                    dynsal_evd_tag = f"{float(getattr(args, 'dyn_sal_evd_thresh', 1e-6)):.0e}"
                    dynsal_tag = f"_dynsalR{dynsal_R}_evd{dynsal_evd_tag}"
                else:
                    dynsal_tag = ""
                fisher_rademacher_k = int(getattr(args, "fisher_rademacher_k", 0))
                num_samples_for_grad = int(getattr(args, "num_samples_for_grad", 0))
                grad_stat_tag = (
                    f"_radk{fisher_rademacher_k}_ngrad{num_samples_for_grad}"
                    + (
                        f"_rseed{int(getattr(args, 'refresh_seed', 0))}"
                        if fisher_rademacher_k > 0
                        else ""
                    )
                    if fisher_rademacher_k > 0 or num_samples_for_grad > 0
                    else ""
                )
                analysis_tag = ""
                if analysis_need_fisher and not is_full_fisher_backed_loss(args.grad_refresh_loss):
                    analysis_tag += "_anaFisher"
                if (
                    analysis_need_legacy_fisher_diag
                    and args.grad_refresh_loss != "legacy_fisher_diag_mse"
                ):
                    analysis_tag += "_anaLegacyFisherDiag"
                if analysis_collect_refined_rkl and args.grad_refresh_loss not in (
                    "refined_residual_kl",
                ):
                    analysis_tag += "_anaRKL"
                if analysis_collect_refined_diag_rkl:
                    analysis_tag += "_anaDiagRKL"
                if need_fp_inps_final and args.grad_refresh_loss not in (
                    "residual_kl", "refined_residual_kl", "refined_mse", "refined_mix",
                ):
                    analysis_tag += "_anaFPFinal"
                if is_full_fisher_backed_loss(args.grad_refresh_loss):
                    analysis_tag += "_mainFisher"
                if args.grad_refresh_loss == "legacy_fisher_diag_mse":
                    analysis_tag += "_mainLegacyFisherDiag"
                static_cache_key = (
                    f"{args.model_name}_mid{model_identity_tag}_{dataset_id}_s{args.nsamples}_"
                    f"blk{args.seq_len}_rot{rotate_flag}_rotid{rotation_identity_tag}_"
                    f"g{args.num_groups}_"
                    f"fisherfull_ghtk{args.grad_hessian_topk}_"
                    f"glbsz{args.global_loss_bsz}_cseed{args.seed}_"
                    f"salclip{sal_clip_tag}_salglobalv1_"
                    f"{_STATIC_SALIENCY_SCHEMA_TAG}_"
                    f"rklNA{rkl_na}_fpfinal{fpfinal_tag}"
                    f"_e2els{int(_E2E_PRECOMPUTE_LOSS_GRAD_SCALE)}"
                    f"{grad_stat_tag}{mix_tag}{dynsal_tag}{analysis_tag}"
                )
                # Bind `_sink{N}` to the cache key only when the option is on,
                # so legacy caches keep their byte-identical key (no migration).
                if sink_size > 0:
                    static_cache_key += f"_sink{sink_size}"
                static_cache_key += f"_world{dist_utils.get_world_size()}_rank{dist_utils.get_rank()}"
                os.makedirs(static_cache_dir, exist_ok=True)
            static_cache_file = (
                os.path.join(static_cache_dir, f"{static_cache_key}.pt")
                if static_cache_key is not None else None
            )
            if (
                static_cache_file is not None
                and not os.path.exists(static_cache_file)
                and is_full_fisher_backed_loss(args.grad_refresh_loss)
                and not analysis_need_legacy_fisher_diag
                and "_mainFisher" in static_cache_key
            ):
                legacy_cache_key = static_cache_key.replace("_mainFisher", "")
                legacy_cache_file = os.path.join(static_cache_dir, f"{legacy_cache_key}.pt")
                if os.path.exists(legacy_cache_file):
                    logging.info(
                        "Static cache %s is missing; falling back to legacy cache key %s. "
                        "The loaded payload will still be checked for Fisher tensors.",
                        static_cache_file,
                        legacy_cache_file,
                    )
                    static_cache_file = legacy_cache_file
            cache_available = (
                static_cache_file is not None
                and os.path.exists(static_cache_file)
            )
            preloaded_static_cache = None
            if cache_available:
                try:
                    preloaded_static_cache = torch.load(
                        static_cache_file,
                        map_location="cpu",
                        weights_only=True,
                    )
                    if not isinstance(preloaded_static_cache, dict):
                        raise TypeError(
                            "static cache payload is not a dictionary"
                        )
                except Exception as exc:
                    logging.warning(
                        "Ignoring unreadable static cache %s: %s",
                        static_cache_file,
                        exc,
                    )
                    preloaded_static_cache = None
                    cache_available = False
            # Static precompute contains collectives. A partial per-rank hit
            # must never let one rank return to Stage 2 while another enters
            # the precompute collectives (deadlock). Reuse only when ALL ranks
            # have their shard; otherwise every rank recomputes.
            if dist_utils.get_world_size() > 1:
                cache_hit_flag = torch.tensor(
                    int(cache_available), device=dev, dtype=torch.int32
                )
                dist.all_reduce(cache_hit_flag, op=dist.ReduceOp.MIN)
                cache_available = bool(cache_hit_flag.item())
            if stage2_cpu_master and not cache_available:
                raise RuntimeError(
                    "stage2_cpu_master requires an existing static precompute cache "
                    "for global_loss Stage 2. Run Stage 1 first with matching "
                    "--static_cache_path / cache-key parameters. Missing cache: "
                    f"{static_cache_file}"
                )

            if cache_available:
                with pipeline_recorder.section("pipeline.static_cache.load") if pipeline_recorder else _NULL_CONTEXT:
                    logging.info("Loading static saliency/fisher cache from %s", static_cache_file)
                    _loaded = preloaded_static_cache
                    static_saliency_by_layer = _loaded["saliency"]
                    static_fisher_by_layer = _loaded.get("fisher", None)
                    if static_fisher_by_layer is None:
                        static_fisher_by_layer = [None] * len(layers)
                    static_legacy_fisher_diag_by_layer = _loaded.get("legacy_fisher_diag", None)
                    if static_legacy_fisher_diag_by_layer is None:
                        static_legacy_fisher_diag_by_layer = [None] * len(layers)
                    if (
                        (analysis_need_fisher or is_full_fisher_backed_loss(args.grad_refresh_loss))
                        and not any(f is not None for f in static_fisher_by_layer)
                    ):
                        raise RuntimeError(
                            "Cached static saliency/fisher at %s does not contain "
                            "layer-output Fisher needed by %s. Delete the cache and rerun "
                            "to regenerate." % (static_cache_file, args.grad_refresh_loss)
                        )
                    if (
                        (analysis_need_legacy_fisher_diag or args.grad_refresh_loss == "legacy_fisher_diag_mse")
                        and not any(f is not None for f in static_legacy_fisher_diag_by_layer)
                    ):
                        raise RuntimeError(
                            "Cached static saliency/fisher at %s does not contain "
                            "per-token legacy Fisher diagonals needed by %s. Delete "
                            "the cache and rerun to regenerate."
                            % (static_cache_file, args.grad_refresh_loss)
                        )
                    # `refined_A` was added later; tolerate older caches that don't
                    # have it. If refined_residual_kl is requested but the cache is
                    # stale, the reloaded list will be empty/missing and we'd fail
                    # below — force the user to rebuild the cache.
                    static_refined_A_by_layer = _loaded.get("refined_A", None)
                    if args.grad_refresh_loss in ("refined_residual_kl", "refined_mix") and not static_refined_A_by_layer:
                        raise RuntimeError(
                            "Cached static saliency/fisher at %s was built before refined_A "
                            "was added. Delete the cache and rerun to regenerate." % static_cache_file
                        )
                    if analysis_collect_refined_rkl and not static_refined_A_by_layer:
                        raise RuntimeError(
                            "Cached static saliency/fisher at %s does not contain refined_A "
                            "needed by analyze_grad_cosine. Delete the cache and rerun to regenerate."
                            % static_cache_file
                        )
                    static_refined_diag_A_by_layer = _loaded.get("refined_diag_A", None)
                    if analysis_collect_refined_diag_rkl and not static_refined_diag_A_by_layer:
                        raise RuntimeError(
                            "Cached static saliency/fisher at %s does not contain refined_diag_A "
                            "needed by analyze_grad_cosine. Delete the cache and rerun to regenerate."
                            % static_cache_file
                        )
                    # fp_inps_final is only present when the cache was built
                    # with capture_fp_final=True (cache_key has fpfinal1). The
                    # fpfinal tag in the cache key guarantees we only load
                    # caches that match the current need_fp_inps_final flag.
                    fp_inps_final_cpu = _loaded.get("fp_inps_final", None)
                    # Dynamic saliency low-rank cache. Present only when the run
                    # that wrote the cache had --enable_dynamic_saliency=1. Cache
                    # key carries `_dynsalR{R}` so a loaded cache here is already
                    # guaranteed to match the current R, but we still assert the
                    # presence of the dynsal payload when the current run wants it.
                    static_dynsal = _loaded.get("dynsal", None)
                    if dynsal_enabled and static_dynsal is None:
                        raise RuntimeError(
                            f"Cached static saliency/fisher at {static_cache_file} was built "
                            f"without --enable_dynamic_saliency. Delete the cache (or use a "
                            f"different --static_cache_path) and rerun precompute."
                        )
                    del _loaded
            else:
                want_refined = analysis_collect_refined_rkl or args.grad_refresh_loss in (
                    "refined_residual_kl", "refined_mix",
                )
                # fisher is consumed by fisher MSE variants (twofold: the refresh
                # loss itself, and the slide-window blend reads the next
                # layer's fisher) AND by refined_mse (as its second-order term
                # — the first-order g·Δy is stacked on top). Everything else
                # (kl / hidden_mse / residual_kl / refined_residual_kl) does
                # not touch the static fisher cache, so skipping the collect
                # halves CPU RAM for those configurations. For refined_mix we
                # still want fisher — but only for the front-half layers; see
                # `fisher_layer_ids` below.
                want_fisher = (
                    analysis_need_fisher
                    or is_full_fisher_backed_loss(args.grad_refresh_loss)
                    or args.grad_refresh_loss == "refined_mix"
                )
                want_legacy_fisher_diag = (
                    analysis_need_legacy_fisher_diag
                    or args.grad_refresh_loss == "legacy_fisher_diag_mse"
                )
                # refined_mix layer-id filters: front half uses refined_mse →
                # needs fisher; back half (except final) uses refined_residual_kl
                # → needs A. Outside mix mode, pass None = collect on every
                # layer for the corresponding stat type.
                if mix_mode:
                    fisher_layer_ids = set(range(refined_mix_split_layer))
                    refined_rkl_layer_ids = set(
                        range(refined_mix_split_layer, len(layers) - 1)
                    )
                else:
                    fisher_layer_ids = None
                    refined_rkl_layer_ids = None
                if analysis_need_fisher or analysis_need_legacy_fisher_diag:
                    fisher_layer_ids = None
                if analysis_collect_refined_rkl or analysis_collect_refined_diag_rkl:
                    refined_rkl_layer_ids = None
                with pipeline_recorder.section("pipeline.static_end_to_end_saliency_fisher") if pipeline_recorder else _NULL_CONTEXT:
                    # 4th return (`refined_diag_A`) is only used by
                    # analyze_grad_cosine today; main quant pipeline ignores it.
                    # 5th return is the rank-local fp_inps_final on CPU in bf16
                    # (None unless `capture_fp_final=True`); we skip the
                    # separate Stage 2 bs=1 precompute when this is populated.
                    # 6th return is the dynamic-saliency low-rank decomposition
                    # wrapper {"by_layer": [...], "N_global": int}, None when
                    # `--enable_dynamic_saliency=0` (default).
                    want_dynsal = bool(int(getattr(args, "enable_dynamic_saliency", 0)))
                    static_saliency_by_layer, static_fisher_by_layer, static_legacy_fisher_diag_by_layer, static_refined_A_by_layer, static_refined_diag_A_by_layer, fp_inps_final_cpu, static_dynsal = \
                        collect_static_end_to_end_saliency_and_fisher(
                            model=model,
                            analyzer=analyzer,
                            dataloader=dataloader,
                            dev=dev,
                            saliency_num_groups=args.num_groups,
                            grad_hessian_topk=args.grad_hessian_topk,
                            batch_size=args.global_loss_bsz,
                            use_fsdp=bool(getattr(args, "fsdp_precompute", False)),
                            fsdp_cpu_offload=bool(getattr(args, "fsdp_cpu_offload", False)),
                            saliency_clip_percentile=getattr(args, "saliency_clip_percentile", 0.99),
                            collect_fisher=want_fisher,
                            collect_legacy_fisher_diag=want_legacy_fisher_diag,
                            collect_refined_rkl=want_refined,
                            refined_rkl_damp=getattr(args, "refined_rkl_damp", 0.01),
                            refined_rkl_num_A=int(getattr(args, "refined_rkl_num_A", 1)),
                            collect_refined_diag_rkl=analysis_collect_refined_diag_rkl,
                            profile_recorder=pipeline_recorder,
                            capture_fp_final=need_fp_inps_final,
                            fisher_layer_ids=fisher_layer_ids,
                            refined_rkl_layer_ids=refined_rkl_layer_ids,
                            collect_dynsal=want_dynsal,
                            dynsal_rank=int(getattr(args, "dyn_sal_rank", 16)),
                            dynsal_evd_thresh=float(getattr(args, "dyn_sal_evd_thresh", 1e-6)),
                            sink_size=sink_size,
                            fisher_rademacher_k=int(getattr(args, "fisher_rademacher_k", 0)),
                            # Optional estimator randomness is an
                            # optimization-time artifact. It must remain fixed
                            # when only the calibration-sampling seed changes.
                            rademacher_seed=(
                                int(getattr(args, "refresh_seed", 0)) + 1701
                            ),
                            num_samples_for_grad=int(getattr(args, "num_samples_for_grad", 0)),
                        )
                if static_cache_file is not None:
                    with pipeline_recorder.section("pipeline.static_cache.save") if pipeline_recorder else _NULL_CONTEXT:
                        logging.info("Saving static saliency/fisher cache to %s", static_cache_file)
                        _to_save = {
                            "saliency": static_saliency_by_layer,
                            "fisher": static_fisher_by_layer,
                            "legacy_fisher_diag": static_legacy_fisher_diag_by_layer,
                            "refined_A": static_refined_A_by_layer,
                        }
                        if static_refined_diag_A_by_layer is not None:
                            _to_save["refined_diag_A"] = static_refined_diag_A_by_layer
                        if fp_inps_final_cpu is not None:
                            _to_save["fp_inps_final"] = fp_inps_final_cpu
                        if static_dynsal is not None:
                            _to_save["dynsal"] = static_dynsal
                        tmp_cache_file = (
                            f"{static_cache_file}.tmp.{os.getpid()}"
                        )
                        try:
                            torch.save(_to_save, tmp_cache_file)
                            os.replace(tmp_cache_file, static_cache_file)
                        finally:
                            if os.path.exists(tmp_cache_file):
                                os.remove(tmp_cache_file)
                        del _to_save
            logging.info(
                "Collected frozen end-to-end saliency/Fisher caches before quantization with global_loss_bsz=%d. "
                "These cached coefficients will be reused for Hessian estimation and Fisher-backed MSE losses throughout quantization.",
                args.global_loss_bsz,
            )
            # Exit right after persisting the cache — the FSDP-wrapped model
            # can't gracefully drop into the per-layer quant phase, so the
            # intended workflow is: (1) torchrun precompute with FSDP, (2)
            # separately run the quant pass without FSDP which reads the cache.
            if bool(getattr(args, "exit_after_precompute", False)):
                logging.info(
                    "exit_after_precompute=1 → finished static saliency/fisher stage, exiting. "
                    "Rerun without --exit_after_precompute to perform quantization."
                )
                if dist.is_initialized():
                    dist.barrier()
                    dist.destroy_process_group()
                sys.exit(0)
        else:
            static_saliency_by_layer = [None] * len(layers)
            static_fisher_by_layer = [None] * len(layers)
            static_legacy_fisher_diag_by_layer = [None] * len(layers)
            static_refined_A_by_layer = None
            static_refined_diag_A_by_layer = None
            static_dynsal = None
            logging.info(
                "Global loss mode is disabled. Saliency/Fisher caches will be collected layerwise with the output head, "
                "and GPTQ+ second-order terms will use layerwise KL."
            )

        per_layer_runtime_modules = list(analyzer.get_pre_block_modules())
        per_layer_runtime_modules.extend(
            [
                analyzer.get_layernorm_before_head(),
                analyzer.get_lm_head(),
            ]
        )
        with pipeline_recorder.section("pipeline.move_to_device") if pipeline_recorder else _NULL_CONTEXT:
            layer_manager.materialize_runtime_modules(per_layer_runtime_modules)
            layers[0] = layer_manager.materialize_layer(0)

        dtype = next(iter(model.parameters())).dtype
        # DP: shard calibration samples contiguously by rank. Each rank only
        # allocates / captures its own slice of inps; subsequent per-layer loops
        # iterate over `inps.shape[0] == n_local`.
        dp_world = dist_utils.get_world_size()
        dp_rank = dist_utils.get_rank()
        if args.nsamples % dp_world != 0:
            raise ValueError(
                f"nsamples ({args.nsamples}) must be divisible by world_size ({dp_world}) for DP."
            )
        n_local = args.nsamples // dp_world
        dp_shard = slice(dp_rank * n_local, (dp_rank + 1) * n_local)
        inps = torch.zeros(
            (n_local, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
        )
        cache = {"global_i": 0, "attention_mask": None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                if hasattr(module, "attention_type"):
                    self.attention_type = module.attention_type

            def forward(self, inp, **kwargs):
                # Only materialise this rank's shard; other ranks drop the sample.
                global_i = cache["global_i"]
                if dp_shard.start <= global_i < dp_shard.stop:
                    inps[global_i - dp_shard.start] = inp
                cache["global_i"] += 1
                cache["attention_mask"] = kwargs["attention_mask"]
                cache["position_ids"] = kwargs["position_ids"]
                cache["position_embeddings"] = kwargs["position_embeddings"]
                raise ValueError

        layers[0] = Catcher(layers[0])
        with pipeline_recorder.section("pipeline.capture_inputs") if pipeline_recorder else _NULL_CONTEXT:
            for batch in dataloader:
                try:
                    model(batch[0].to(dev))
                except ValueError:
                    pass
        layers[0] = layers[0].module

        layer_manager.release_layer(0, layers[0], update_master=False, orig_device=orig_device)
        memory_utils.cleanup_memory(False)

        attention_mask = cache["attention_mask"]
        position_ids = cache["position_ids"]
        position_embeddings = cache["position_embeddings"]

        sequential = analyzer.get_sequential_quantizable_module_names()
        names = [n for ns in sequential for n in ns]
        fp_inps = inps.clone()

        if act_quant_aware_gptq:
            if stage2_cpu_master:
                raise RuntimeError(
                    "stage2_cpu_master does not support --act_quant_aware_gptq yet; "
                    "configuring all meta-resident layers would materialize or mutate "
                    "non-broadcast layers."
                )
            with pipeline_recorder.section("pipeline.configure_act_quant_for_gptq") if pipeline_recorder else _NULL_CONTEXT:
                input_q_count, v_q_count = configure_activation_quantizers_for_gptq(args, model)
                logging.info(
                    "Configured activation quantization for GPTQ+ student paths: "
                    "input_wrappers=%d v_out_wrappers=%d.",
                    input_q_count,
                    v_q_count,
                )

        if k_cache_quant_aware_gptq:
            if stage2_cpu_master:
                raise RuntimeError(
                    "stage2_cpu_master does not support --k_cache_quant_aware_gptq yet; "
                    "QK wrappers are installed across all layers before the streaming loop."
                )
            with pipeline_recorder.section("pipeline.configure_k_cache_quant_for_gptq") if pipeline_recorder else _NULL_CONTEXT:
                k_q_count = configure_k_cache_quantizers_for_gptq(args, analyzer)
                logging.info(
                    "Configured K-cache quantization for GPTQ+ student paths: "
                    "qk_wrappers=%d.",
                    k_q_count,
                )

        if args.offload_inps:
            with pipeline_recorder.section("pipeline.offload_inputs") if pipeline_recorder else _NULL_CONTEXT:
                inps = inps.cpu()
                fp_inps = fp_inps.cpu()

        quantizers = {}
        gradient_refresh_scheduler = None
        dp_global_shuffle = bool(getattr(args, "dp_global_shuffle", False))
        if args.g_update_mode in {"block_backward", "block_gd"} or effective_pre_gd_steps > 0:
            if dp_global_shuffle:
                # Single globally-shared shuffle. Every rank constructs the
                # scheduler with the same refresh_seed + same total_samples, so
                # `next_indices()` returns the identical global id list on
                # every rank. Each rank then filters to its own shard inside
                # `collect_true_weight_gradient`.
                gradient_refresh_scheduler = BackwardSampleScheduler(
                    args.nsamples,
                    args.backward_samples,
                    seed=int(getattr(args, "refresh_seed", 0)),
                )
            else:
                # Stratified: each rank owns a per-rank scheduler over its
                # own shard. `backward_samples` must divide `dp_world` so the
                # per-rank chunk is integral.
                if args.backward_samples % dp_world != 0:
                    raise ValueError(
                        f"backward_samples ({args.backward_samples}) must be divisible by world_size ({dp_world})."
                    )
                backward_samples_local = args.backward_samples // dp_world
                gradient_refresh_scheduler = BackwardSampleScheduler(
                    n_local,
                    backward_samples_local,
                    seed=int(getattr(args, "refresh_seed", 0)) + dp_rank,
                )
        final_layer_idx = len(layers) - 1

        # residual_kl / refined_residual_kl / refined_mse need the FP output
        # of the last transformer block per sample (teacher hidden at the
        # input of final norm + lm_head). When `global_loss=1` we capture it
        # as a by-product of the static saliency/fisher forward pass (stored
        # in `fp_inps_final_cpu`, bf16 CPU) and just move it to match
        # fp_inps's device/dtype here — no second forward pass. Fallback to
        # the bs=1 per-layer precompute when global_loss is off or when the
        # capture didn't run (e.g. loaded an older cache).
        fp_inps_final = None
        if need_fp_inps_final:
            if fp_inps_final_cpu is not None:
                with pipeline_recorder.section("pipeline.fp_final_from_static") if pipeline_recorder else _NULL_CONTEXT:
                    logging.info(
                        "Reusing fp_inps_final captured during static saliency/fisher "
                        "(shape=%s, %s → %s/%s).",
                        tuple(fp_inps_final_cpu.shape),
                        fp_inps_final_cpu.dtype,
                        fp_inps.device, fp_inps.dtype,
                    )
                    fp_inps_final = fp_inps_final_cpu.to(
                        device=fp_inps.device, dtype=fp_inps.dtype,
                    )
                    fp_inps_final_cpu = None
            else:
                with pipeline_recorder.section("pipeline.fp_final_precompute") if pipeline_recorder else _NULL_CONTEXT:
                    logging.info(
                        "Precomputing FP final-layer hidden states for %s "
                        "(nsamples_local=%d, layers=%d).",
                        args.grad_refresh_loss, inps.shape[0], len(layers),
                    )
                    # Work on a scratch copy so we don't disturb the main `fp_inps`
                    # buffer (which is still at the layer-0 input stage).
                    scratch = inps.detach().clone().to(dev)
                    for idx in range(len(layers)):
                        lay = layer_manager.materialize_layer(idx)
                        with disable_fp_path_quant(lay):
                            # Per-sample forward matches the per-layer fp_reference_forward
                            # pattern; batching here would change numerics (cuBLAS kernel
                            # selection) and has been shown to cause drift.
                            for j in range(scratch.shape[0]):
                                scratch[j] = lay(
                                    scratch[j].unsqueeze(0),
                                    attention_mask=attention_mask,
                                    position_ids=position_ids,
                                    position_embeddings=position_embeddings,
                                )[0].squeeze(0)
                        # Return each layer to its original (CPU) residency so the
                        # main quant loop's `layers[i].to(dev)` starts from the same
                        # state as if this precompute never happened.
                        layer_manager.release_layer(idx, lay, update_master=False, orig_device=orig_device)
                    # Match the storage convention of `fp_inps` so downstream index
                    # expressions behave identically (`fp_inps_final[batch]` / CPU-
                    # to-GPU handoff inside collect_true_weight_gradient).
                    if fp_inps.device != scratch.device:
                        fp_inps_final = scratch.to(fp_inps.device)
                    else:
                        fp_inps_final = scratch
                    del scratch

        if dp_global_shuffle:
            # Every rank sees all global sample ids; `collect_true_weight_gradient`
            # filters to the rank's shard.
            full_refresh_sample_indices = (
                list(range(args.nsamples)) if args.final_layer_full_backward else None
            )
        else:
            # In the stratified path, each rank's full-backward equals using
            # all of its local samples; together they cover all `args.nsamples`.
            full_refresh_sample_indices = (
                list(range(n_local)) if args.final_layer_full_backward else None
            )
        layer_indices = range(quant_stop_layer + 1) if quant_stop_layer is not None else range(len(layers))
        # Side stream + event used to move this layer's downstream blocks back
        # to CPU asynchronously after `collect_layer_output_grad_for_refined_mse`.
        # The sync is deferred to the start of the NEXT layer's refined_mse
        # block (when we need those CPU tensors again). Lazy-initialised so
        # non-refined_mse runs pay nothing.
        refined_mse_d2h_stream = None
        refined_mse_d2h_event = None
        pbar = tqdm(layer_indices, ncols=120, desc="Quantizing Layers", position=0)
        for i in pbar:
            layer = layer_manager.materialize_layer(i)
            full = analyzer.get_quantizable_modules(layer)
            layer_recorder = QuantProfileRecorder(dev, prefix=f"layers.{i}") if quant_profile_enabled else None
            analysis_is_target = False
            analysis_fp_weights = None
            if analysis_hook is not None:
                analysis_is_target = bool(analysis_hook("before_layer", {
                    "args": args,
                    "analyzer": analyzer,
                    "layer": layer,
                    "layer_idx": i,
                    "layers": layers,
                    "full": full,
                    "inps": inps,
                    "fp_inps": fp_inps,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "position_embeddings": position_embeddings,
                    "dev": dev,
                    "orig_device": orig_device,
                }))
            # Activation-aware paper runs disable the reverse-cosine ramp at
            # the first-layer/base LR. Only a *real* low-bit aware path counts:
            # passing an aware flag alongside A16/V16/K16 is a numerical no-op.
            activation_aware_enabled = (
                (
                    bool(getattr(args, "act_quant_aware_gptq", False))
                    and (args.a_bits < 16 or args.v_bits < 16)
                )
                or (
                    bool(getattr(args, "k_cache_quant_aware_gptq", False))
                    and args.k_bits < 16
                )
            )
            activation_aware_schedule_disabled = (
                activation_aware_enabled and i != final_layer_idx
            )
            grad_lr_layer_scale = (
                1.0
                if activation_aware_schedule_disabled
                else compute_layer_lr_scale(
                    layer_idx=i,
                    num_layers=len(layers),
                    schedule=getattr(args, "grad_lr_layer_schedule", "none"),
                )
            )
            grad_lr_layer_base_ratio = float(getattr(args, "grad_lr_layer_base_ratio", 0.01))
            if activation_aware_schedule_disabled:
                logging.info(
                    "Layer %d activation-aware LR schedule disabled: using "
                    "configured LR as a constant (scale=1; base_ratio %.4f ignored)",
                    i, grad_lr_layer_base_ratio,
                )
            elif getattr(args, "grad_lr_layer_schedule", "none") != "none":
                logging.info(
                    "Layer %d grad_lr schedule scale=%.4f base_ratio=%.4f (schedule=%s)",
                    i, grad_lr_layer_scale, grad_lr_layer_base_ratio, args.grad_lr_layer_schedule,
                )
            layer_refresh_loss_type = get_effective_refresh_loss_type(
                i,
                final_layer_idx,
                args.grad_refresh_loss,
                refined_mix_split_layer=refined_mix_split_layer,
            )
            # Slide-window blends layer i's loss with layer i+1's loss. For
            # refined_mix, i = split-1 sits right at the loss-type boundary
            # (refined_mse → refined_residual_kl), and the two losses rely on
            # incompatible per-layer state (refined_mse needs fisher + g pool;
            # refined_residual_kl needs A + fp_inps_final). Blending them would
            # require a cross-type plumbing we explicitly don't do in v1.
            # Detect this once here and suppress slide_window both when
            # collecting the next-layer grad pool and when mixing refresh
            # losses. For non-mix runs this is always True (next layer shares
            # the same loss type), so behaviour is unchanged.
            if i + 1 <= final_layer_idx:
                next_layer_refresh_loss_type = get_effective_refresh_loss_type(
                    i + 1,
                    final_layer_idx,
                    args.grad_refresh_loss,
                    refined_mix_split_layer=refined_mix_split_layer,
                )
            else:
                next_layer_refresh_loss_type = None
            next_layer_same_type = (
                next_layer_refresh_loss_type == layer_refresh_loss_type
            )
            gptq_reference_loss_type = get_effective_gptq_reference_loss_type(
                global_loss_enabled,
                layer_refresh_loss_type,
            )
            if i == final_layer_idx and layer_refresh_loss_type != args.grad_refresh_loss:
                logging.info(
                    "Overriding refresh loss for final layer %d: %s -> %s",
                    i,
                    args.grad_refresh_loss,
                    layer_refresh_loss_type,
                )
            # Pull layer-i's refined_residual_kl A matrices onto dev for the
            # duration of this layer's quantization. `static_refined_A_by_layer[i]`
            # is a list of length `refined_rkl_num_A`; each slot is either an A
            # tensor (bf16 CPU) or None (e.g. the last layer's slots). We move
            # the whole list to dev here so per-batch `refined_A_list[a_idx]`
            # dispatches don't incur a H2D copy per batch.
            layer_refined_A_list = None
            if (
                layer_refresh_loss_type == "refined_residual_kl"
                and static_refined_A_by_layer is not None
                and static_refined_A_by_layer[i] is not None
            ):
                layer_refined_A_list = [
                    (slot.to(dev) if slot is not None else None)
                    for slot in static_refined_A_by_layer[i]
                ]
            # Interpret all *_bsz knobs as GLOBAL batch sizes and split per
            # rank. With N=1 these reduce to their original values, so N=1
            # behavior w.r.t. bsz is unchanged.
            layer_backward_bsz_global = args.final_layer_backward_bsz if i == final_layer_idx else args.backward_bsz
            layer_stats_bsz_global = args.final_layer_stats_bsz if i == final_layer_idx else args.bsz
            if layer_backward_bsz_global % dp_world != 0:
                raise ValueError(
                    f"layer_backward_bsz ({layer_backward_bsz_global}) must be divisible by world_size ({dp_world})."
                )
            if layer_stats_bsz_global % dp_world != 0:
                raise ValueError(
                    f"layer_stats_bsz ({layer_stats_bsz_global}) must be divisible by world_size ({dp_world})."
                )
            layer_backward_bsz = layer_backward_bsz_global // dp_world
            layer_stats_bsz = layer_stats_bsz_global // dp_world

            # refined_mse: before this layer's quant/refresh loop opens, capture
            # ∂KL/∂(layer_i output) on a fresh per-layer random pool. The pool
            # grad is used inside `compute_refresh_loss(refined_mse, ...)` to
            # add a first-order term `g · Δy` on top of fisher_diag_mse. At
            # layer 0 the model output equals the teacher, so g ≡ 0; we skip
            # the backward and plant zeros.
            # With loss_slide_window on, we additionally capture ∂KL/∂(layer_{
            # i+1} output) on the SAME single backward pass (same sample pool)
            # so the next-layer refresh loss can add its own first-order term.
            # Gated to `i <= final_layer_idx - 2` (same as slide_window itself).
            layer_refined_mse_pool_ids = None
            layer_refined_mse_grad_pool = None
            layer_refined_mse_mean_grad = None
            layer_refined_mse_grad_pool_next = None
            layer_refined_mse_mean_grad_next = None
            slide_window_enabled = (
                getattr(args, "loss_slide_window", False)
                and args.g_update_mode == "block_gd"
                and i <= final_layer_idx - 2
                and next_layer_same_type
            )
            collect_next_refined_mse = (
                layer_refresh_loss_type == "refined_mse"
                and slide_window_enabled
                and global_loss_enabled
                and static_fisher_by_layer[i + 1] is not None
            )
            if layer_refresh_loss_type == "refined_mse":
                with layer_recorder.section("layer.refined_mse_grad_pool.total") if layer_recorder else _NULL_CONTEXT:
                    hidden_size = model.config.hidden_size
                    if i == 0:
                        layer_refined_mse_pool_ids = torch.tensor([], dtype=torch.long)
                        layer_refined_mse_mean_grad = torch.zeros(
                            hidden_size, dtype=torch.float32, device=dev
                        )
                        # Layer-0 slide-window next-layer pool is also zero
                        # (student == teacher at layer 0 → KL = 0 → g_{i+1}
                        # ≡ 0 too). Plant zeros so the non-pool broadcast in
                        # compute_refresh_loss still works.
                        if collect_next_refined_mse:
                            layer_refined_mse_mean_grad_next = torch.zeros(
                                hidden_size, dtype=torch.float32, device=dev
                            )
                    else:
                        n_pool = int(args.num_samples_for_refined_mse)
                        if n_pool > n_local:
                            raise ValueError(
                                f"num_samples_for_refined_mse ({n_pool}) exceeds rank-local "
                                f"calibration size ({n_local} = nsamples // world)."
                            )
                        rm_bwd_bsz = args.global_loss_bsz // dp_world
                        if rm_bwd_bsz <= 0 or n_pool % rm_bwd_bsz != 0:
                            raise ValueError(
                                f"refined_mse: num_samples_for_refined_mse ({n_pool}) "
                                f"must be divisible by per-rank backward bsz "
                                f"({rm_bwd_bsz} = global_loss_bsz // world)."
                            )
                        rng = random.Random(
                            int(getattr(args, "refresh_seed", 0)) + i
                        )
                        sample_ids_local = sorted(rng.sample(range(n_local), n_pool))
                        # Before reusing any downstream CPU tensor, make sure the
                        # previous layer's async D2H has drained — otherwise the
                        # H2D we're about to do might read stale / half-written
                        # memory. `.synchronize()` blocks the current stream on
                        # the event; does nothing if the event already fired.
                        if refined_mse_d2h_event is not None:
                            with layer_recorder.section("layer.refined_mse_grad_pool.prev_d2h_sync") if layer_recorder else _NULL_CONTEXT:
                                refined_mse_d2h_event.synchronize()
                                refined_mse_d2h_event = None
                        # Downstream transformer blocks live on CPU during the main
                        # quant loop; move them to dev for the end-to-end backward,
                        # restore after. pre_block / norm / lm_head are already on
                        # dev (see `per_layer_runtime_modules` setup at fn entry).
                        with layer_recorder.section("layer.refined_mse_grad_pool.downstream_to_dev") if layer_recorder else _NULL_CONTEXT:
                            for k in range(i + 1, len(layers)):
                                layer_manager.materialize_layer(k)
                        try:
                            with layer_recorder.section("layer.refined_mse_grad_pool.collect") if layer_recorder else _NULL_CONTEXT:
                                (
                                    layer_refined_mse_grad_pool,
                                    layer_refined_mse_mean_grad,
                                    layer_refined_mse_grad_pool_next,
                                    layer_refined_mse_mean_grad_next,
                                ) = collect_layer_output_grad_for_refined_mse(
                                    analyzer=analyzer,
                                    layer=layer,
                                    layer_idx=i,
                                    layers=layers,
                                    inps=inps,
                                    fp_inps_final=fp_inps_final,
                                    attention_mask=attention_mask,
                                    position_ids=position_ids,
                                    position_embeddings=position_embeddings,
                                    sample_ids_local=sample_ids_local,
                                    backward_bsz=rm_bwd_bsz,
                                    kl_topk=args.kl_topk,
                                    dev=dev,
                                    collect_next=collect_next_refined_mse,
                                    layer_recorder=layer_recorder,
                                    sink_size=sink_size,
                                )
                        finally:
                            # Launch the downstream D2H on a side stream and
                            # defer the sync to the top of the NEXT layer's
                            # refined_mse block (see prev_d2h_sync above).
                            # This lets the current layer's subsequent work
                            # (fp_reference_forward → stats → Hessian accum →
                            # fasterquant) run in parallel with the D2H on
                            # the default stream. The overlap is real only
                            # once the offloaded layer params are backed by
                            # pinned CPU memory (cudaMemcpyAsync requires
                            # pinned dest); without pinning it's a no-op but
                            # costs nothing. We also drop the prior
                            # `cleanup_memory()` call, which did a global
                            # `torch.cuda.synchronize()` + `empty_cache()`
                            # and alone took 100-500 ms per layer.
                            with layer_recorder.section("layer.refined_mse_grad_pool.downstream_to_cpu") if layer_recorder else _NULL_CONTEXT:
                                if refined_mse_d2h_stream is None:
                                    refined_mse_d2h_stream = torch.cuda.Stream(device=dev)
                                with torch.cuda.stream(refined_mse_d2h_stream):
                                    for k in range(i + 1, len(layers)):
                                        layer_manager.release_layer(k, layers[k], update_master=False, orig_device=orig_device)
                                refined_mse_d2h_event = torch.cuda.Event()
                                refined_mse_d2h_event.record(refined_mse_d2h_stream)
                        layer_refined_mse_pool_ids = torch.tensor(
                            sample_ids_local, dtype=torch.long
                        )
                        if collect_next_refined_mse:
                            logging.info(
                                "refined_mse: collected per-layer grad pool layer=%d N=%d "
                                "bwd_bsz=%d |mean_grad|=%.3e |mean_grad_next|=%.3e",
                                i, n_pool, rm_bwd_bsz,
                                layer_refined_mse_mean_grad.norm(p=2).item(),
                                layer_refined_mse_mean_grad_next.norm(p=2).item(),
                            )
                        else:
                            logging.info(
                                "refined_mse: collected per-layer grad pool layer=%d N=%d "
                                "bwd_bsz=%d |mean_grad|=%.3e",
                                i, n_pool, rm_bwd_bsz,
                                layer_refined_mse_mean_grad.norm(p=2).item(),
                            )

            if analysis_hook is not None:
                analysis_hook("before_fp_reference", {
                    "args": args,
                    "analyzer": analyzer,
                    "layer": layer,
                    "layer_idx": i,
                    "layers": layers,
                    "full": full,
                    "inps": inps,
                    "fp_inps": fp_inps,
                    "fp_inps_final": fp_inps_final,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "position_embeddings": position_embeddings,
                    "dev": dev,
                    "orig_device": orig_device,
                    "is_target": analysis_is_target,
                })

            with layer_recorder.section("layer.fp_reference_forward") if layer_recorder else _NULL_CONTEXT:
                with disable_fp_path_quant(layer):
                    # inps/fp_inps are rank-local shards of length n_local.
                    for j in range(inps.shape[0]):
                        fp_inps[j] = layer(
                            fp_inps[j].unsqueeze(0).to(dev),
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            position_embeddings=position_embeddings,
                        )[0].to(fp_inps.device)

            # --- Loss-slide-window setup: precompute reference output of the
            # next FP transformer block, so the per-block refresh can blend the
            # current-layer loss with the next-layer loss. Supported under:
            #   * fisher MSE variants — need `static_fisher_by_layer[i+1]`
            #     (so global_loss must be on).
            #   * residual_kl    — reuses the already-precomputed fp_inps_final
            #     (no next-layer fisher needed).
            #   * refined_residual_kl — reuses fp_inps_final and additionally
            #     needs the next-layer A matrix from static_refined_A_by_layer.
            #   * refined_mse    — needs `static_fisher_by_layer[i+1]` (the
            #     second-order term) AND the next-layer grad pool collected
            #     above (first-order term).
            # Skipped at and past the second-to-last layer per the spec.
            slide_active_layer = False
            if (
                getattr(args, "loss_slide_window", False)
                and args.g_update_mode == "block_gd"
                and i <= final_layer_idx - 2
                and next_layer_same_type
            ):
                if (
                    is_fisher_mse_loss(layer_refresh_loss_type)
                    and global_loss_enabled
                    and select_layer_output_fisher_for_loss(
                        layer_refresh_loss_type,
                        static_fisher_by_layer,
                        static_legacy_fisher_diag_by_layer,
                        i + 1,
                    ) is not None
                ):
                    slide_active_layer = True
                elif (
                    layer_refresh_loss_type == "residual_kl"
                    and fp_inps_final is not None
                ):
                    slide_active_layer = True
                elif (
                    layer_refresh_loss_type == "refined_residual_kl"
                    and fp_inps_final is not None
                    and static_refined_A_by_layer is not None
                    and static_refined_A_by_layer[i + 1] is not None
                    and any(s is not None for s in static_refined_A_by_layer[i + 1])
                ):
                    slide_active_layer = True
                elif (
                    layer_refresh_loss_type == "refined_mse"
                    and global_loss_enabled
                    and static_fisher_by_layer[i + 1] is not None
                    and layer_refined_mse_mean_grad_next is not None
                ):
                    # layer 0: grad_pool_next is None but mean_grad_next is
                    # zeros(H) — that's the correct degenerate state (first-
                    # order term collapses to 0, loss reduces to next-layer
                    # fisher_diag_mse).
                    slide_active_layer = True
            slide_next_layer = None
            slide_fp_inps_next = None
            slide_next_layer_output_fisher = None
            slide_next_refined_A_list = None
            if slide_active_layer:
                with layer_recorder.section("layer.slide_window.next_fp_reference") if layer_recorder else _NULL_CONTEXT:
                    slide_next_layer = layer_manager.materialize_layer(i + 1)
                    # fisher MSE variants and refined_mse both need the
                    # next-layer fisher; legacy_fisher_diag_mse differs only in
                    # how compute_refresh_loss consumes that matrix.
                    # refined_residual_kl needs the next-layer A. residual_kl
                    # needs neither — loss routes through fp_inps_final only.
                    if is_fisher_backed_loss(layer_refresh_loss_type):
                        slide_next_layer_output_fisher = select_layer_output_fisher_for_loss(
                            layer_refresh_loss_type,
                            static_fisher_by_layer,
                            static_legacy_fisher_diag_by_layer,
                            i + 1,
                        )
                    elif layer_refresh_loss_type == "refined_residual_kl":
                        # Load the next layer's full A list onto dev for
                        # per-batch sub-A dispatch in the slide branch.
                        _next_list = static_refined_A_by_layer[i + 1]
                        if _next_list is not None and len(_next_list) > 0:
                            slide_next_refined_A_list = [
                                (s.to(dev) if s is not None else None)
                                for s in _next_list
                            ]
                    slide_fp_inps_next = torch.empty_like(fp_inps)
                    with disable_fp_path_quant(slide_next_layer):
                        for j in range(fp_inps.shape[0]):
                            slide_fp_inps_next[j] = slide_next_layer(
                                fp_inps[j].unsqueeze(0).to(dev),
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                position_embeddings=position_embeddings,
                            )[0].to(slide_fp_inps_next.device)
                    logging.info(
                        "Loss-slide-window active for layer %d (next=%d, mode=%s).",
                        i,
                        i + 1,
                        layer_refresh_loss_type,
                    )

            if preclip_enabled:
                with layer_recorder.section("layer.weight_preclip") if layer_recorder else _NULL_CONTEXT:
                    for name, module in full.items():
                        if module is None or "lm_head" in name:
                            continue
                        clip_module_weight_to_quant_bounds_(
                            module,
                            bits=args.w_bits,
                            sym=not args.w_asym,
                            mse=args.w_clip,
                        )

            if analysis_hook is not None:
                if analysis_is_target:
                    analysis_fp_weights = {
                        normalize_quant_module_name(raw_name): mod.weight.detach().clone().cpu()
                        for raw_name, mod in full.items()
                        if mod is not None and hasattr(mod, "weight")
                    }
                analysis_hook("after_preclip", {
                    "args": args,
                    "analyzer": analyzer,
                    "layer": layer,
                    "layer_idx": i,
                    "layers": layers,
                    "full": full,
                    "inps": inps,
                    "fp_inps": fp_inps,
                    "fp_inps_final": fp_inps_final,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "position_embeddings": position_embeddings,
                    "dev": dev,
                    "orig_device": orig_device,
                    "is_target": analysis_is_target,
                })

            batch_attention_mask = attention_mask.expand(args.bsz, -1, -1, -1)
            batch_position_ids = position_ids.expand(args.bsz, -1)
            batch_position_embeddings = (
                position_embeddings[0].expand(args.bsz, -1, -1),
                position_embeddings[1].expand(args.bsz, -1, -1),
            )

            layer_output_fisher = None
            subset = build_quant_subset(full, names)
            layer_output_fisher_by_module = {}
            pre_gd_refresh_loss_type = layer_refresh_loss_type
            if effective_pre_gd_steps > 0 and is_fisher_backed_loss(pre_gd_refresh_loss_type):
                layer_output_fisher = select_layer_output_fisher_for_loss(
                    pre_gd_refresh_loss_type,
                    static_fisher_by_layer,
                    static_legacy_fisher_diag_by_layer,
                    i,
                )
                if layer_output_fisher is None:
                    if pre_gd_refresh_loss_type == "legacy_fisher_diag_mse":
                        raise RuntimeError(
                            "legacy_fisher_diag_mse pre-GD requires the static "
                            "per-token Fisher diagonal cache. Delete stale caches "
                            "and rerun with --global_loss."
                        )
                    with layer_recorder.section("layer.pre_quant_fisher_collect") if layer_recorder else _NULL_CONTEXT:
                        layer_output_fisher = collect_layer_output_fisher_only(
                            model=model,
                            layer=layer,
                            analyzer=analyzer,
                            inps=inps,
                            fp_inps=fp_inps,
                            batch_attention_mask=batch_attention_mask,
                            batch_position_ids=batch_position_ids,
                            batch_position_embeddings=batch_position_embeddings,
                            bsz=args.bsz,
                            kl_topk=args.kl_topk,
                            grad_hessian_topk=args.grad_hessian_topk,
                            dev=dev,
                            layer_idx=i,
                            layer_recorder=layer_recorder,
                            sink_size=sink_size,
                            legacy_diag=False,
                        )
                for name in subset:
                    if subset[name] is not None:
                        layer_output_fisher_by_module[name] = layer_output_fisher

            if effective_pre_gd_steps > 0:
                pre_grad_optimizer = (
                    args.pre_final_layer_grad_optimizer
                    if i == final_layer_idx and args.pre_final_layer_grad_optimizer is not None
                    else args.pre_grad_optimizer
                )
                pre_grad_lr = (
                    args.pre_final_layer_grad_lr
                    if i == final_layer_idx and args.pre_final_layer_grad_lr is not None
                    else compute_scheduled_layer_lr(
                        args.pre_grad_lr,
                        grad_lr_layer_scale,
                        grad_lr_layer_base_ratio,
                    )
                )
                # refined_mix: back-half layers use refined_residual_kl and get
                # their own LR relative to the front half. Multiply by the
                # configured ratio once here so both pre_gd and block_gd see
                # the same LR cadence. Final layer keeps its dedicated override.
                if (
                    mix_mode
                    and i != final_layer_idx
                    and layer_refresh_loss_type == "refined_residual_kl"
                ):
                    pre_grad_lr *= float(getattr(args, "refined_mix_rkl_lr_ratio", 1.0))
                # Final-layer gradients come through lm_head + final norm and are
                # often orders of magnitude larger; allow an independent clip.
                effective_grad_clip = (
                    args.final_layer_grad_clip
                    if i == final_layer_idx and args.final_layer_grad_clip is not None
                    else args.grad_clip
                )
                if pre_grad_lr > 0:
                    with layer_recorder.section("layer.pre_quant_gd") if layer_recorder else _NULL_CONTEXT:
                        logging.info(
                            "Running pre-quantization GD for layer=%d steps=%d lr=%s optimizer=%s refresh_loss=%s",
                            i,
                            effective_pre_gd_steps,
                            format_log_value(pre_grad_lr, digits=6),
                            pre_grad_optimizer,
                            layer_refresh_loss_type,
                        )
                        run_pre_quant_gd(
                            layer_idx=i,
                            layer=layer,
                            analyzer=analyzer,
                            full=full,
                            inps=inps,
                            fp_inps=fp_inps,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            position_embeddings=position_embeddings,
                            backward_bsz=layer_backward_bsz,
                            kl_topk=args.kl_topk,
                            dev=dev,
                            module_names=list(subset.keys()),
                            scheduler=gradient_refresh_scheduler,
                            num_steps=effective_pre_gd_steps,
                            grad_lr=pre_grad_lr,
                            grad_optimizer=pre_grad_optimizer,
                            grad_clip=effective_grad_clip,
                            refresh_loss_type=layer_refresh_loss_type,
                            layer_output_fisher_by_module=layer_output_fisher_by_module,
                            fp_inps_final=fp_inps_final,
                            refined_A_list=layer_refined_A_list,
                            samples_per_A=refined_rkl_samples_per_A,
                            global_shuffle=dp_global_shuffle,
                            dp_rank=dp_rank,
                            shard_size=n_local,
                            refresh_mb=getattr(args, "refresh_mb", None),
                            refined_mse_pool_ids=layer_refined_mse_pool_ids,
                            refined_mse_grad_pool=layer_refined_mse_grad_pool,
                            refined_mse_mean_grad=layer_refined_mse_mean_grad,
                            layer_recorder=layer_recorder,
                            sink_size=sink_size,
                            a_loss_ratio=args.a_loss_ratio,
                            a_loss_clip_scope=args.a_loss_clip_scope,
                        )

            # Compute slide-window refresh span over the whole transformer block,
            # then quantize modules group-by-group so later groups collect Hessian
            # inputs after earlier groups have already been quantized.
            slide_refreshes_per_module = {}
            for _name, _module in subset.items():
                if _module is None:
                    slide_refreshes_per_module[_name] = 0
                    continue
                _cols = _module.weight.shape[1]
                _n_blocks = (_cols + args.blocksize - 1) // args.blocksize
                slide_refreshes_per_module[_name] = max(_n_blocks - 1, 0)
            slide_refresh_block_total = sum(slide_refreshes_per_module.values())
            slide_refresh_cursor = 0
            gptq = {}
            saliency_dict = None
            gradients_dict = None

            # === Dynamic saliency: one FP forward per layer, stream P = Y · V ===
            # Instead of caching all 7 modules' (N_local, T, H_out) outputs to
            # CPU (which is ~400 GB for Llama-3-8B — up/gate alone is 2 × 112 GB
            # and OOM-kills the host), compute P_fp = FP_Y · V in the forward
            # hook on the fly, writing (bsz, T, R_eff) bf16 slices into a
            # preallocated per-module (N_local, T, R_eff) CPU buffer. For R=16
            # that's 128 MB per module × 7 = ~900 MB for the whole layer —
            # ~450× smaller than caching FP_Y directly.
            #
            # `dyn_sal_refresh_mode`:
            #   per_boundary (default): refresh S and accumulate Hessian before
            #     each of the qkv / o / up+gate / down boundaries. Captures both
            #     upstream drift AND the drift from already-quantized modules in
            #     this layer.
            #   per_layer: refresh S and accumulate Hessian ONCE per layer, at
            #     layer entry with all weights still FP. Captures upstream drift
            #     only, then reuses the resulting per-module H through the four
            #     module-group quantization boundaries.
            dynsal_enabled = (
                bool(int(getattr(args, "enable_dynamic_saliency", 0)))
                and static_dynsal is not None
            )
            dynsal_refresh_mode = str(getattr(args, "dyn_sal_refresh_mode", "per_boundary"))
            P_fp_by_canonical = None
            dynsal_V_dev_for_layer = None
            full_canonical = None
            precomputed_S_new_by_canonical = None
            if dynsal_enabled:
                with layer_recorder.section("layer.dynsal.fp_forward") if layer_recorder else _NULL_CONTEXT:
                    _fp_fwd_bsz = args.hessian_accum_bsz if args.hessian_accum_bsz is not None else args.bsz
                    _fp_fwd_bsz = max(1, min(_fp_fwd_bsz, fp_inps.shape[0]))
                    # Key everything by CANONICAL module name (the
                    # `.module`-stripped form that `module_dicts` used during
                    # precompute and that `group_names` / `subset` use later).
                    # `full.keys()` may carry the `.module` suffix on
                    # QuantWrapped layers; we normalise once and pass the
                    # canonical-keyed dict through the rest of the dynsal code.
                    layer_dyn = static_dynsal["by_layer"][i]
                    full_canonical = {
                        normalize_quant_module_name(n): m for n, m in full.items()
                    }
                    dynsal_V_dev_for_layer = {}
                    for _canonical, _module in full_canonical.items():
                        _entry = layer_dyn.get(_canonical)
                        if _entry is None:
                            raise KeyError(
                                f"Missing dynsal V for layer={i} module (canonical)={_canonical}. "
                                f"Available keys: {sorted(layer_dyn.keys())}"
                            )
                        # bf16 V: avoids fp32-casting the full (bsz·T, H_out)
                        # forward activation inside the projection hook.
                        dynsal_V_dev_for_layer[_canonical] = _entry["V"].to(dev, dtype=torch.bfloat16)
                    with disable_fp_path_quant(layer):
                        P_fp_by_canonical = _collect_module_output_projections(
                            layer=layer,
                            inputs=fp_inps,
                            module_dict=full_canonical,
                            V_by_name=dynsal_V_dev_for_layer,
                            dev=dev,
                            bsz=_fp_fwd_bsz,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            position_embeddings=position_embeddings,
                            sink_size=sink_size,
                        )

                # per_layer mode: compute S_new for every module once at layer
                # entry using `layer(inps)` with all weights still FP. Cache the
                # per-module S_new and let each boundary just pull from it.
                if dynsal_refresh_mode == "per_layer":
                    with layer_recorder.section("layer.dynsal.per_layer_refresh") if layer_recorder else _NULL_CONTEXT:
                        _cur_fwd_bsz = args.hessian_accum_bsz if args.hessian_accum_bsz is not None else args.bsz
                        _cur_fwd_bsz = max(1, min(_cur_fwd_bsz, inps.shape[0]))
                        P_cur_by_canonical = _collect_module_output_projections(
                            layer=layer,
                            inputs=inps,
                            module_dict=full_canonical,
                            V_by_name=dynsal_V_dev_for_layer,
                            dev=dev,
                            bsz=_cur_fwd_bsz,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            position_embeddings=position_embeddings,
                            sink_size=sink_size,
                        )
                        N_global = int(static_dynsal["N_global"])
                        precomputed_S_new_by_canonical = {}
                        for _canonical in full_canonical.keys():
                            dyn_entry = layer_dyn[_canonical]  # already verified above
                            static_sal = static_saliency_by_layer[i].get(_canonical)
                            if static_sal is None:
                                raise KeyError(
                                    f"Missing static saliency for layer={i} module (canonical)={_canonical}."
                                )
                            P_delta = (
                                P_cur_by_canonical[_canonical].float()
                                - P_fp_by_canonical[_canonical].float()
                            ).to(torch.bfloat16)
                            precomputed_S_new_by_canonical[_canonical] = refresh_dynamic_saliency(
                                dyn_entry=dyn_entry,
                                static_saliency=static_sal,
                                N_global=N_global,
                                P_delta=P_delta,
                                num_groups=args.num_groups,
                                dev=dev,
                                static_saliency_scale=_E2E_PRECOMPUTE_QUADRATIC_SCALE,
                            )
                            del P_delta
                        # P_fp / P_cur no longer needed — per-layer S_new is
                        # cached. Drop them to bound CPU peak.
                        del P_cur_by_canonical
                        P_fp_by_canonical = None
                        memory_utils.cleanup_memory()

            layer_hessian_once = dynsal_refresh_mode == "per_layer"
            analysis_hessians = {} if analysis_is_target else None

            def build_gptq_for_subset(subset_for_setup, saliency_for_setup, gradients_for_setup, reference_loss_for_setup):
                gptq_local = {}
                with layer_recorder.section("layer.gptq_setup") if layer_recorder else _NULL_CONTEXT:
                    for module_name, module in subset_for_setup.items():
                        layer_weight_bits = args.w_bits
                        layer_weight_sym = not args.w_asym
                        if module is None or "lm_head" in module_name:
                            continue
                        saliency = saliency_for_setup.get(module_name, saliency_for_setup.get(module_name + ".module", None))
                        if saliency is None:
                            raise KeyError(
                                f"Missing saliency cache for layer={i} module={module_name}. "
                                f"Available keys: {sorted(saliency_for_setup.keys())}"
                            )

                        gptq_local[module_name] = GPTQPlus(
                            module,
                            saliency=saliency,
                            gradient=gradients_for_setup[module_name],
                            num_groups=args.num_groups,
                            alpha=args.alpha,
                            reference_loss=reference_loss_for_setup,
                            hessian_saliency_scale=(
                                _E2E_PRECOMPUTE_QUADRATIC_SCALE
                                if global_loss_enabled else 1.0
                            ),
                            hessian_group_shard=(
                                getattr(args, "group_parallel_quant", "none") == "rank"
                                and dist_utils.get_world_size() > 1
                            ),
                        )
                        gptq_local[module_name].quantizer = quant_utils.WeightQuantizer()
                        gptq_local[module_name].quantizer.configure(
                            layer_weight_bits,
                            perchannel=True,
                            sym=layer_weight_sym,
                            mse=args.w_clip,
                        )
                return gptq_local

            def accumulate_hessian_for_gptq(gptq_for_accum, subset_for_accum):
                hessian_sample_count = int(getattr(args, "num_samples_for_grad", 0) or 0)
                if global_loss_enabled and hessian_sample_count > 0:
                    if hessian_sample_count % dp_world != 0:
                        raise ValueError(
                            f"num_samples_for_grad ({hessian_sample_count}) must be "
                            f"divisible by world_size ({dp_world})."
                        )
                    hessian_local_samples = hessian_sample_count // dp_world
                    if hessian_local_samples <= 0 or hessian_local_samples > inps.shape[0]:
                        raise ValueError(
                            f"num_samples_for_grad // world ({hessian_local_samples}) "
                            f"must be in [1, {inps.shape[0]}]."
                        )
                else:
                    hessian_local_samples = inps.shape[0]

                def add_batch(name):
                    def tmp(_, inp, out):
                        gptq_for_accum[name].add_batch(inp[0].data, out.data)

                    return tmp

                handles = []
                add_batch_recorders = {}
                for module_name in gptq_for_accum:
                    if should_profile_module(i, module_name):
                        add_batch_recorders[module_name] = QuantProfileRecorder(dev, prefix=f"layers.{i}.{module_name}")
                        gptq_for_accum[module_name].profile_recorder = add_batch_recorders[module_name]
                    handles.append(subset_for_accum[module_name].register_forward_hook(add_batch(module_name)))
                with layer_recorder.section("layer.hessian_accumulation_forward") if layer_recorder else _NULL_CONTEXT:
                    # Batch the accumulation forward so we don't pay a per-sample
                    # kernel-launch tax. `add_batch` already handles arbitrary
                    # batch sizes (it reshapes to [bsz*seq, dim] internally), so
                    # the math is bit-exact regardless of bsz.
                    hessian_accum_bsz = args.hessian_accum_bsz if args.hessian_accum_bsz is not None else args.bsz
                    hessian_accum_bsz = max(1, min(hessian_accum_bsz, hessian_local_samples))
                    for j in tqdm(
                        range(0, hessian_local_samples, hessian_accum_bsz),
                        ncols=120,
                        desc=f"Layer {i} Hessian accumulation",
                        position=1,
                        leave=False,
                    ):
                        batch_bsz = min(hessian_accum_bsz, hessian_local_samples - j)
                        _ = layer(
                            inps[j : j + batch_bsz].to(dev),
                            attention_mask=attention_mask.expand(batch_bsz, -1, -1, -1),
                            position_ids=position_ids.expand(batch_bsz, -1),
                            position_embeddings=(
                                position_embeddings[0].expand(batch_bsz, -1, -1),
                                position_embeddings[1].expand(batch_bsz, -1, -1),
                            ),
                        )[0]
                for h in handles:
                    h.remove()
                for module_name in add_batch_recorders:
                    gptq_for_accum[module_name].profile_recorder = None

                # Close out the Hessian accumulation: all-reduce the per-rank sums
                # and apply the global normalisation exactly once. After this call
                # fasterquant sees a globally-averaged H that is bit-identical on
                # every rank (NCCL all_reduce is deterministic for a given op+shape).
                with layer_recorder.section("layer.hessian_finalize") if layer_recorder else _NULL_CONTEXT:
                    for module_name in gptq_for_accum:
                        gptq_for_accum[module_name].finalize_hessian()
                if analysis_hessians is not None:
                    for module_name, gptq_obj in gptq_for_accum.items():
                        analysis_hessians[module_name] = (
                            gptq_obj.H.detach().float().clone().cpu()
                        )

            if layer_hessian_once:
                with layer_recorder.section("layer.per_layer_stats_and_hessian") if layer_recorder else _NULL_CONTEXT:
                    layerwide_subset = build_quant_subset(full, names)
                    precomputed_saliency_for_layer = static_saliency_by_layer[i]
                    if dynsal_enabled:
                        precomputed_saliency_for_layer = {
                            **static_saliency_by_layer[i],
                            **precomputed_S_new_by_canonical,
                        }
                    saliency_dict, gradients_dict, mean_reference_loss, layer_output_fisher = collect_layer_grad_hessian_stats(
                        model=model,
                        layer=layer,
                        analyzer=analyzer,
                        full=full,
                        names=list(layerwide_subset.keys()),
                        inps=inps,
                        fp_inps=fp_inps,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                        bsz=layer_stats_bsz,
                        num_groups=args.num_groups,
                        kl_topk=args.kl_topk,
                        grad_hessian_topk=args.grad_hessian_topk,
                        dev=dev,
                        layer_idx=i,
                        layer_refresh_loss_type=layer_refresh_loss_type,
                        gptq_reference_loss_type=gptq_reference_loss_type,
                        precomputed_saliency_dict=precomputed_saliency_for_layer,
                        precomputed_layer_output_fisher=select_layer_output_fisher_for_loss(
                            layer_refresh_loss_type,
                            static_fisher_by_layer,
                            static_legacy_fisher_diag_by_layer,
                            i,
                        ),
                        fp_inps_final=fp_inps_final,
                        refined_A_list=layer_refined_A_list,
                        samples_per_A=refined_rkl_samples_per_A,
                        dp_rank=dp_rank,
                        dp_shard_size=n_local,
                        layer_recorder=layer_recorder,
                        skip_gradient_backward=skip_ref_backward,
                        sink_size=sink_size,
                        a_loss_ratio=args.a_loss_ratio,
                    )
                    gptq = build_gptq_for_subset(
                        layerwide_subset,
                        saliency_dict,
                        gradients_dict,
                        mean_reference_loss,
                    )
                    accumulate_hessian_for_gptq(gptq, layerwide_subset)

            for group_names in sequential:
                subset = build_quant_subset(full, group_names)
                if not subset:
                    continue

                # === Dynamic saliency boundary refresh ===
                # per_boundary: run a lightweight current-state forward covering
                #   this group only, compute S_new per module on the fly.
                # per_layer: S_new was already computed at layer entry; just
                #   look up the cached value by canonical name.
                dynsal_refresh_overrides = None
                if dynsal_enabled and dynsal_refresh_mode == "per_boundary":
                    with layer_recorder.section("layer.dynsal.boundary_refresh") if layer_recorder else _NULL_CONTEXT:
                        _cur_fwd_bsz = args.hessian_accum_bsz if args.hessian_accum_bsz is not None else args.bsz
                        _cur_fwd_bsz = max(1, min(_cur_fwd_bsz, inps.shape[0]))
                        V_for_group = {
                            name: dynsal_V_dev_for_layer[name] for name in subset.keys()
                        }
                        P_cur_for_group = _collect_module_output_projections(
                            layer=layer,
                            inputs=inps,
                            module_dict=subset,
                            V_by_name=V_for_group,
                            dev=dev,
                            bsz=_cur_fwd_bsz,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            position_embeddings=position_embeddings,
                            sink_size=sink_size,
                        )
                        dynsal_by_layer = static_dynsal["by_layer"]
                        N_global = int(static_dynsal["N_global"])
                        dynsal_refresh_overrides = {}
                        for _canonical in subset.keys():
                            dyn_entry = dynsal_by_layer[i].get(_canonical)
                            if dyn_entry is None:
                                raise KeyError(
                                    f"Missing dynsal cache for layer={i} module (canonical)={_canonical}. "
                                    f"Available keys: {sorted(dynsal_by_layer[i].keys())}"
                                )
                            static_sal = static_saliency_by_layer[i].get(_canonical)
                            if static_sal is None:
                                raise KeyError(
                                    f"Missing static saliency for layer={i} module (canonical)={_canonical}."
                                )
                            if _canonical not in P_fp_by_canonical:
                                raise KeyError(
                                    f"P_fp entry for layer={i} module (canonical)={_canonical} was "
                                    f"already popped — boundary refresh ordering is broken."
                                )
                            P_delta = (
                                P_cur_for_group[_canonical].float()
                                - P_fp_by_canonical[_canonical].float()
                            ).to(torch.bfloat16)
                            S_new = refresh_dynamic_saliency(
                                dyn_entry=dyn_entry,
                                static_saliency=static_sal,
                                N_global=N_global,
                                P_delta=P_delta,
                                num_groups=args.num_groups,
                                dev=dev,
                                static_saliency_scale=_E2E_PRECOMPUTE_QUADRATIC_SCALE,
                            )
                            dynsal_refresh_overrides[_canonical] = S_new
                            # Release this module's P_fp / P_delta now — they
                            # won't be read again for this boundary or any future
                            # boundary in the same layer.
                            del P_fp_by_canonical[_canonical]
                            del P_delta
                        del P_cur_for_group
                        memory_utils.cleanup_memory()
                elif dynsal_enabled and dynsal_refresh_mode == "per_layer":
                    # Layer-level refresh: look up pre-computed S_new for each
                    # module in this group (canonical key).
                    dynsal_refresh_overrides = {
                        _canonical: precomputed_S_new_by_canonical[_canonical]
                        for _canonical in subset.keys()
                    }

                # Merge any dynsal-refreshed saliency with the static shards. The
                # downstream `collect_layer_grad_hessian_stats` only uses the dict
                # to bypass its own saliency-collection pass; replacing specific
                # module entries with S_new is sufficient.
                precomputed_saliency_for_group = static_saliency_by_layer[i]
                if dynsal_refresh_overrides:
                    precomputed_saliency_for_group = {
                        **static_saliency_by_layer[i],
                        **dynsal_refresh_overrides,
                    }

                if not layer_hessian_once:
                    saliency_dict, gradients_dict, mean_reference_loss, layer_output_fisher = collect_layer_grad_hessian_stats(
                        model=model,
                        layer=layer,
                        analyzer=analyzer,
                        full=full,
                        names=group_names,
                        inps=inps,
                        fp_inps=fp_inps,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                        bsz=layer_stats_bsz,
                        num_groups=args.num_groups,
                        kl_topk=args.kl_topk,
                        grad_hessian_topk=args.grad_hessian_topk,
                        dev=dev,
                        layer_idx=i,
                        layer_refresh_loss_type=layer_refresh_loss_type,
                        gptq_reference_loss_type=gptq_reference_loss_type,
                        precomputed_saliency_dict=precomputed_saliency_for_group,
                        precomputed_layer_output_fisher=select_layer_output_fisher_for_loss(
                            layer_refresh_loss_type,
                            static_fisher_by_layer,
                            static_legacy_fisher_diag_by_layer,
                            i,
                        ),
                        fp_inps_final=fp_inps_final,
                        refined_A_list=layer_refined_A_list,
                        samples_per_A=refined_rkl_samples_per_A,
                        dp_rank=dp_rank,
                        dp_shard_size=n_local,
                        layer_recorder=layer_recorder,
                        skip_gradient_backward=skip_ref_backward,
                        sink_size=sink_size,
                        a_loss_ratio=args.a_loss_ratio,
                    )
                    gptq = build_gptq_for_subset(
                        subset,
                        saliency_dict,
                        gradients_dict,
                        mean_reference_loss,
                    )
                    accumulate_hessian_for_gptq(gptq, subset)

                # module_name -> ModuleGradSecondMoment. One per module, kept across
                # every refresh of that module so the pre-pass sees the whole
                # calibration pass before `build_prior` reads it.
                grad_sq_accums = {}
                collect_grad_sq = "warm_adam" in {
                    getattr(args, "grad_optimizer", None),
                    getattr(args, "final_layer_grad_optimizer", None),
                }

                def make_gradient_refresh_fn(
                    module_name,
                    slide_next_layer=None,
                    slide_fp_inps_next=None,
                    slide_next_layer_output_fisher=None,
                    slide_next_refined_A_list=None,
                    slide_next_refined_mse_grad_pool=None,
                    slide_next_refined_mse_mean_grad=None,
                ):
                    def refresh_fn(weight_snapshot, slide_alpha=1.0):
                        grad_sq_accum = None
                        if collect_grad_sq:
                            grad_sq_accum = grad_sq_accums.get(module_name)
                            if grad_sq_accum is None:
                                _mod = full.get(
                                    module_name, full.get(module_name + ".module", None)
                                )
                                if _mod is not None:
                                    grad_sq_accum = ModuleGradSecondMoment(
                                        _mod, sink_size=sink_size
                                    )
                                    grad_sq_accums[module_name] = grad_sq_accum
                        if args.final_layer_full_backward and i == final_layer_idx:
                            sample_indices = full_refresh_sample_indices
                        else:
                            sample_indices = gradient_refresh_scheduler.next_indices()
                        partial_grad_sum, partial_count, loss_sum, grad_extras = (
                            collect_true_weight_gradient(
                                layer=layer,
                                analyzer=analyzer,
                                module_name=module_name,
                                full=full,
                                inps=inps,
                                fp_inps=fp_inps,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                position_embeddings=position_embeddings,
                                bsz=layer_backward_bsz,
                                kl_topk=args.kl_topk,
                                dev=dev,
                                weight_override=weight_snapshot,
                                sample_indices=sample_indices,
                                refresh_loss_type=layer_refresh_loss_type,
                                layer_output_fisher=layer_output_fisher,
                                slide_alpha=slide_alpha,
                                next_layer=slide_next_layer,
                                fp_inps_next=slide_fp_inps_next,
                                next_layer_output_fisher=slide_next_layer_output_fisher,
                                fp_inps_final=fp_inps_final,
                                refined_A_list=layer_refined_A_list,
                                next_refined_A_list=slide_next_refined_A_list,
                                samples_per_A=refined_rkl_samples_per_A,
                                global_shuffle=dp_global_shuffle,
                                dp_rank=dp_rank,
                                shard_size=n_local,
                                refresh_mb=getattr(args, "refresh_mb", None),
                                refined_mse_pool_ids=layer_refined_mse_pool_ids,
                                refined_mse_grad_pool=layer_refined_mse_grad_pool,
                                refined_mse_mean_grad=layer_refined_mse_mean_grad,
                                next_refined_mse_grad_pool=slide_next_refined_mse_grad_pool,
                                next_refined_mse_mean_grad=slide_next_refined_mse_mean_grad,
                                layer_recorder=layer_recorder,
                                sink_size=sink_size,
                                a_loss_ratio=args.a_loss_ratio,
                                a_loss_clip_scope=args.a_loss_clip_scope,
                                grad_sq_accum=grad_sq_accum,
                            )
                        )
                        # DP aggregation. When world_size > 1 we pack the grad sum
                        # and all scalar aggregates into a single contiguous tensor
                        # and do one all_reduce instead of 3-5 back-to-back NCCL
                        # collectives. Each collective costs one cudaStreamSynchronize
                        # on every rank; packing cuts 20-40% off true_gradient_refresh
                        # wall time in multi-rank runs.
                        if dp_world > 1:
                            extra_scalar_keys = [
                                k for k in ("loss_sum_current", "loss_sum_next")
                                if k in grad_extras
                            ]
                            # Layout: [grad_flat | count | loss_sum | extra_scalar_keys...]
                            # fp32 is enough: partial_count fits exactly (< 2^24), and
                            # a single-collective sum matches the precision of the
                            # original per-tensor allreduces (NCCL uses fp32 paths too).
                            grad_flat = partial_grad_sum.reshape(-1)
                            scalar_vec = torch.tensor(
                                [float(partial_count), float(loss_sum)]
                                + [float(grad_extras[k]) for k in extra_scalar_keys],
                                dtype=grad_flat.dtype, device=grad_flat.device,
                            )
                            packed = torch.cat([grad_flat, scalar_vec])
                            dist_utils.allreduce_sum_(packed)
                            partial_grad_sum.copy_(packed[:grad_flat.numel()].view_as(partial_grad_sum))
                            scalar_out = packed[grad_flat.numel():]
                            global_count = int(scalar_out[0].item())
                            global_loss_sum = scalar_out[1].item()
                            for idx, k in enumerate(extra_scalar_keys, start=2):
                                grad_extras[k] = scalar_out[idx].item()
                        else:
                            global_count = partial_count
                            global_loss_sum = loss_sum
                        if global_count <= 0:
                            raise RuntimeError(
                                f"refresh produced zero samples across all ranks; "
                                f"sample_indices={sample_indices[:8]}..."
                            )
                        grad = partial_grad_sum / float(global_count)
                        mean_refresh_loss = global_loss_sum / float(global_count)
                        trace_sample_indices = sample_indices
                        if (
                            trace_writer.enabled
                            and dp_world > 1
                            and not dp_global_shuffle
                        ):
                            # Stratified schedulers expose rank-local indices.
                            # The trace schema always stores global calibration
                            # ids, in deterministic rank-major order.
                            local_global_indices = torch.tensor(
                                [
                                    dp_rank * n_local + int(index)
                                    for index in sample_indices
                                ],
                                dtype=torch.long,
                                device=partial_grad_sum.device,
                            )
                            gathered_indices = [
                                torch.empty_like(local_global_indices)
                                for _ in range(dp_world)
                            ]
                            dist.all_gather(
                                gathered_indices,
                                local_global_indices,
                            )
                            trace_sample_indices = torch.cat(
                                gathered_indices,
                            ).detach().cpu().tolist()
                        meta = {
                            "mean_refresh_loss": mean_refresh_loss,
                            "sample_indices": trace_sample_indices,
                        }
                        if slide_next_layer is not None:
                            # Record α=1 explicitly at the first slide step.
                            # collect_true_weight_gradient deliberately skips
                            # evaluating the zero-weight next arm there, so its
                            # diagnostics historically omitted this value.
                            meta["slide_alpha"] = float(slide_alpha)
                        if "loss_sum_current" in grad_extras:
                            meta["mean_refresh_loss_current"] = (
                                grad_extras["loss_sum_current"] / float(global_count)
                            )
                        else:
                            meta["mean_refresh_loss_current"] = mean_refresh_loss
                        if "loss_sum_next" in grad_extras:
                            meta["mean_refresh_loss_next"] = (
                                grad_extras["loss_sum_next"] / float(global_count)
                            )
                        return grad, meta

                    return refresh_fn

                def make_block_observer(module_name, effective_grad_optimizer):
                    def observer(payload):
                        logging.info(
                            "block-metrics layer=%d module=%s mode=%s grad_opt=%s block=%d cols=[%d,%d) remain=%d grad_abs_mean=%s grad_clipped_abs_mean=%s grad_row_l2=%s loss=%s refresh_loss=%s train_loss=%s val_loss=%s second_abs=%s second_abs_max=%s second_abs_q99=%s first_raw_abs=%s first_abs=%s first_abs_max=%s first_abs_q99=%s reg_abs=%s sine_abs=%s slide_alpha=%s loss_cur=%s loss_next=%s",
                            i,
                            module_name,
                            args.g_update_mode,
                            effective_grad_optimizer,
                            payload["block_idx"],
                            payload["col_start"],
                            payload["col_end"],
                            payload["remaining_columns"],
                            format_log_value(payload["remaining_grad_abs_mean"], digits=4),
                            format_log_value(payload.get("remaining_grad_clipped_abs_mean"), digits=4),
                            format_log_value(payload["remaining_grad_mean_row_l2"], digits=4),
                            format_log_value(payload["mean_refresh_loss"], digits=6),
                            format_log_value(payload["refresh_subset_mean_refresh_loss"], digits=6),
                            format_log_value(payload["train_mean_refresh_loss"], digits=6),
                            format_log_value(payload["val_mean_refresh_loss"], digits=6),
                            format_log_value(payload["second_order_update_abs_mean"], digits=4),
                            format_log_value(payload.get("second_order_update_abs_max"), digits=4),
                            format_log_value(payload.get("second_order_update_abs_q99"), digits=4),
                            format_log_value(payload["first_order_raw_abs_mean"], digits=4),
                            format_log_value(payload["first_order_update_abs_mean"], digits=4),
                            format_log_value(payload.get("first_order_update_abs_max"), digits=4),
                            format_log_value(payload.get("first_order_update_abs_q99"), digits=4),
                            format_log_value(payload["regularizer_update_abs_mean"], digits=4),
                            format_log_value(payload["sine_regularizer_update_abs_mean"], digits=4),
                            format_log_value(payload.get("slide_alpha"), digits=3),
                            format_log_value(payload.get("mean_refresh_loss_current"), digits=6),
                            format_log_value(payload.get("mean_refresh_loss_next"), digits=6),
                        )
                        if trace_writer.enabled and args.g_update_mode == "block_gd":
                            trace_writer.record(
                                refresh_step_from_metrics(
                                    layer=i,
                                    module=module_name,
                                    metrics=payload,
                                )
                            )

                    return observer

                with layer_recorder.section("layer.module_quantization") if layer_recorder else _NULL_CONTEXT:
                    for name in subset:
                        if name not in gptq:
                            continue
                        pbar.set_postfix(module=f"layers.{i}." + name, loss=f"{mean_reference_loss:.2e}")
                        layer_w_groupsize = args.w_groupsize
                        effective_grad_optimizer = (
                            args.final_layer_grad_optimizer
                            if i == final_layer_idx and args.final_layer_grad_optimizer is not None
                            else args.grad_optimizer
                        )
                        base_grad_lr = (
                            args.final_layer_grad_lr
                            if i == final_layer_idx and args.final_layer_grad_lr is not None
                            else compute_scheduled_layer_lr(
                                args.grad_lr,
                                grad_lr_layer_scale,
                                grad_lr_layer_base_ratio,
                            )
                        )
                        # refined_mix: back-half layers (refined_residual_kl) use
                        # `grad_lr * refined_mix_rkl_lr_ratio`. Final-layer override
                        # wins (final layer is forced to kl and typically has its
                        # own tuned lr), so the ratio only applies when we're NOT
                        # on the final layer and the effective loss is refined_rkl.
                        if (
                            mix_mode
                            and i != final_layer_idx
                            and layer_refresh_loss_type == "refined_residual_kl"
                        ):
                            base_grad_lr *= float(getattr(args, "refined_mix_rkl_lr_ratio", 1.0))
                        effective_grad_reg_strategy = "none" if i == final_layer_idx else args.grad_reg_strategy
                        # Final-layer gradients come through lm_head + final norm and
                        # are often orders of magnitude larger; allow an independent
                        # clip. Falls back to `--grad_clip` for non-final layers or
                        # when the override is not set.
                        effective_main_grad_clip = (
                            args.final_layer_grad_clip
                            if i == final_layer_idx and args.final_layer_grad_clip is not None
                            else args.grad_clip
                        )
                        effective_grad_lr = get_module_grad_lr(
                            name,
                            base_grad_lr,
                            proj_lr_scale=args.proj_lr_scale,
                            down_proj_lr_scale=args.down_proj_lr_scale,
                        )
                        if (
                            effective_grad_lr != base_grad_lr
                            or base_grad_lr != args.grad_lr
                            or effective_grad_optimizer != args.grad_optimizer
                        ):
                            logging.info(
                                "Applying block_gd config for layer=%d module=%s: global_base_lr=%.4e layer_base_lr=%.4e effective_lr=%.4e global_opt=%s effective_opt=%s refresh_loss=%s",
                                i,
                                name,
                                args.grad_lr,
                                base_grad_lr,
                                effective_grad_lr,
                                args.grad_optimizer,
                                effective_grad_optimizer,
                                layer_refresh_loss_type,
                            )
                        if i == final_layer_idx and args.grad_reg_strategy != "none":
                            logging.info(
                                "Disabling first-order regularization for final layer=%d module=%s: %s -> none",
                                i,
                                name,
                                args.grad_reg_strategy,
                            )
                        module_recorder = (
                            QuantProfileRecorder(dev, prefix=f"layers.{i}.{name}")
                            if should_profile_module(i, name) else None
                        )
                        if dist_utils.is_main() and getattr(args, "enable_debug", False):
                            _H = gptq[name].H
                            _grad = gptq[name].gradients
                            _sal = gptq[name].saliencies
                            _act = gptq[name].act_square
                            logging.info(
                                "dp-probe layer=%d module=%s H_mean=%.6e H_absmax=%.6e grad_mean=%.6e grad_absmax=%.6e "
                                "act_mean=%.6e sal_mean=%.6e sal_shape=%s refloss=%.6e tokens=%d idx=%d",
                                i, name,
                                _H.float().mean().item(), _H.float().abs().max().item(),
                                _grad.float().mean().item(), _grad.float().abs().max().item(),
                                _act.float().mean().item(),
                                _sal.float().mean().item(), tuple(_sal.shape),
                                gptq[name].reference_loss,
                                gptq[name].token_count, gptq[name].index,
                            )
                        # warm_adam needs a prior before the first column block
                        # closes, so spend one extra refresh here at the still
                        # unquantised weight. At backward_samples == nsamples a
                        # single refresh already covers every calibration sample;
                        # below that the prior is built from that many samples and
                        # is correspondingly noisier.
                        #
                        # The refresh function has to be hoisted out of the call
                        # below so the pre-pass and the quant loop share one
                        # instance (and therefore one accumulator).
                        _refresh_fn_for_module = (
                            make_gradient_refresh_fn(
                                name,
                                slide_next_layer=slide_next_layer,
                                slide_fp_inps_next=slide_fp_inps_next,
                                slide_next_layer_output_fisher=slide_next_layer_output_fisher,
                                slide_next_refined_A_list=slide_next_refined_A_list,
                                slide_next_refined_mse_grad_pool=layer_refined_mse_grad_pool_next,
                                slide_next_refined_mse_mean_grad=layer_refined_mse_mean_grad_next,
                            )
                            if args.g_update_mode in {"block_backward", "block_gd"}
                            else None
                        )
                        grad_sq_full = None
                        warm_start_steps = 0
                        # May be downgraded to adam below when the pre-pass finds
                        # no gradient signal to build a prior from.
                        _module_grad_optimizer = effective_grad_optimizer
                        if effective_grad_optimizer == "warm_adam":
                            if _refresh_fn_for_module is None:
                                raise ValueError(
                                    "`grad_optimizer=warm_adam` needs a refresh "
                                    "function; use --g_update_mode block_gd."
                                )
                            _B = max(
                                int(getattr(args, "backward_samples", 0) or inps.shape[0]), 1
                            )
                            grad_sq_accums.pop(name, None)
                            # The pre-pass is a measurement, not a step: freeze the
                            # sample scheduler so the refreshes that follow see
                            # exactly the batches they would have seen without it.
                            # Without this, `--warm_start_steps 0` -- which is
                            # mathematically identical to adam -- does not
                            # reproduce adam.
                            _freeze = (
                                gradient_refresh_scheduler.frozen()
                                if gradient_refresh_scheduler is not None
                                else _NULL_CONTEXT
                            )
                            # W_fp is where block 0 starts, so it is the
                            # right point to predict block 0's gradient from --
                            # measured, the prior matches the first observed g^2
                            # to within 0.65-3.1x here, while an RTN-quantised
                            # measurement point overshot by 5.6-268x and lost
                            # most of the per-coordinate structure as well.
                            _measure_at = "W_fp"
                            # K refresh batches, so the prior rests on K*B
                            # samples and t0 (derived below from the samples
                            # actually measured) comes out as K. One frozen()
                            # block wraps the whole loop: the cursor advances
                            # inside it, giving K distinct batches, and is
                            # restored on exit so the training stream is
                            # untouched. Wrapping each call separately would
                            # hand back the same batch K times -- same cost, no
                            # new evidence, and a t0 of K that is not earned.
                            _K = max(int(getattr(args, "warm_prior_batches", 1) or 1), 1)
                            _seen_batches = []
                            with _freeze:
                                for _k in range(_K):
                                    _, _pre_meta = _refresh_fn_for_module(
                                        subset[name].weight.data.float()
                                    )
                                    _seen_batches.append(
                                        tuple(_pre_meta.get("sample_indices", ()))
                                    )
                            # t0 is only honest if the batches really differ.
                            if _K > 1 and len(set(_seen_batches)) != _K:
                                raise RuntimeError(
                                    f"warm_adam pre-pass for {name} drew {_K} batches "
                                    f"but only {len(set(_seen_batches))} distinct ones; "
                                    "t0 would claim evidence the prior does not have."
                                )
                            _pre = grad_sq_accums.get(name)
                            if _pre is None:
                                raise RuntimeError(
                                    f"warm_adam pre-pass never built an accumulator "
                                    f"for {name}."
                                )
                            # A rank seeing no tokens is normal under DP: with
                            # dp_global_shuffle the refresh batch may fall
                            # entirely inside another rank's shard. Only global
                            # emptiness is an error, and the prior has to be the
                            # global quantity anyway.
                            _n_global = _pre.all_reduce_()
                            if _n_global == 0:
                                raise RuntimeError(
                                    f"warm_adam pre-pass produced no tokens for {name} "
                                    "on any rank; the module hooks never fired."
                                )
                            grad_sq_full = _pre.build_prior(_B)
                            # Control: collapse the prior to one scalar per
                            # tensor, destroying per-coordinate shape while
                            # keeping a magnitude. Which magnitude matters: P is
                            # heavy-tailed, so its arithmetic mean sits far above
                            # the typical coordinate and would suppress most of
                            # them while relaxing the few large ones -- a change
                            # of scale on top of the change of shape. The
                            # geometric mean tracks the typical coordinate
                            # instead, so running both separates the two.
                            _scalar_mode = str(
                                getattr(args, "warm_prior_scalar", "none") or "none"
                            )
                            if _scalar_mode != "none":
                                if _scalar_mode == "mean":
                                    _c = float(grad_sq_full.mean().item())
                                else:
                                    _pos = grad_sq_full[grad_sq_full > 0]
                                    _c = (
                                        float(torch.exp(torch.log(_pos).mean()).item())
                                        if _pos.numel() > 0 else 0.0
                                    )
                                logging.info(
                                    "warm_adam prior collapsed to a %s scalar: %s "
                                    "(was median=%s max=%s)",
                                    _scalar_mode, format_log_value(_c),
                                    format_log_value(grad_sq_full.median().item()),
                                    format_log_value(grad_sq_full.max().item()),
                                )
                                grad_sq_full = torch.full_like(grad_sq_full, _c)
                            # t0 is what the prior is worth in refresh-steps:
                            # samples measured / samples per refresh. Deriving it
                            # from nsamples instead would assume the pre-pass
                            # covered the whole calibration set, which is only
                            # true when backward_samples == nsamples.
                            _t0_override = int(getattr(args, "warm_start_steps", -1))
                            warm_start_steps = (
                                _t0_override if _t0_override >= 0
                                else max(int(round(_n_global / max(_B, 1))), 1)
                            )
                            logging.info(
                                "warm_adam pre-pass layer=%d module=%s at=%s K=%d "
                                "samples=%d tokens=%d fwd=%d grad=%d skipped=%d "
                                "B=%d t0=%d prior_median=%s prior_max=%s "
                                "|| per-rank: %s",
                                i, name, _measure_at, _K,
                                _pre.sample_count, _pre.token_count,
                                _pre.fwd_calls, _pre.grad_calls, _pre.grad_skipped,
                                _B, warm_start_steps,
                                format_log_value(grad_sq_full.median().item()),
                                format_log_value(grad_sq_full.max().item()),
                                _pre.format_rank_stats(),
                            )
                            # A zero prior is the correct measurement before
                            # anything has been quantised: the refresh loss is
                            # anchored at the FP output, so its gradient there is
                            # zero. It means there is no signal to warm-start
                            # from, not that the pre-pass failed -- so this
                            # module runs plain Adam, which is what warm_adam
                            # with no evidence reduces to anyway.
                            if float(grad_sq_full.abs().max().item()) == 0.0:
                                logging.info(
                                    "warm_adam layer=%d module=%s: prior is "
                                    "identically zero at %s (no refresh gradient "
                                    "before any quantisation); falling back to "
                                    "plain adam for this module.",
                                    i, name, _measure_at,
                                )
                                _module_grad_optimizer = "adam"
                                grad_sq_full = None
                                warm_start_steps = 0

                        # Identity for logging only; nothing reads it unless
                        # a probe is enabled (see _reach_probe_emit).
                        gptq[name].layer_idx = i
                        gptq[name].layer_name = name
                        gptq[name].fasterquant(
                            grad_sq_full=grad_sq_full,
                            warm_start_steps=warm_start_steps,
                            blocksize=args.blocksize,
                            percdamp=args.percdamp,
                            groupsize=layer_w_groupsize,
                            actorder=args.act_order,
                            static_groups=args.act_order,
                            enable_gradient_update=True,
                            g_update_mode=args.g_update_mode,
                            export_to_et=args.export_to_et,
                            profile_recorder=module_recorder,
                            gradient_refresh_fn=_refresh_fn_for_module,
                            grad_lr=effective_grad_lr,
                            grad_optimizer=_module_grad_optimizer,
                            grad_reg_strategy=effective_grad_reg_strategy,
                            grad_reg_lambda=args.grad_reg_lambda,
                            grad_gate_floor=args.grad_gate_floor,
                            grad_gate_sharpness=args.grad_gate_sharpness,
                            grad_gate_sine_amp=args.grad_gate_sine_amp,
                            second_order_scale=args.second_order_scale,
                            block_atomic_quant=args.block_atomic_quant,
                            block_observer=make_block_observer(name, _module_grad_optimizer) if args.g_update_mode in {"block_backward", "block_gd"} else None,
                            grad_clip=effective_main_grad_clip,
                            horizon_p=float(getattr(args, "horizon_p", 0.0) or 0.0),
                            diagnostic_recorder=diagnostic_registry.get_or_create(i, name),
                            slide_refresh_start=slide_refresh_cursor,
                            slide_refresh_block_total=slide_refresh_block_total,
                            refresh_full_metrics=bool(getattr(args, "refresh_full_metrics", False)),
                            group_parallel_mode=getattr(args, "group_parallel_quant", "none"),
                        )
                        slide_refresh_cursor += slide_refreshes_per_module[name]
                        # DP correctness check (debug only): fasterquant is meant
                        # to be deterministic given identical inputs, and since H /
                        # gradients / act_square are bit-identical across ranks after
                        # all-reduce, the output weight must be bit-identical too.
                        # Any drift here compounds across layers, so we fail fast.
                        if getattr(args, "enable_debug", False):
                            dist_utils.assert_bit_exact(
                                subset[name].weight.data,
                                tag=f"layer{i}.{name}.weight_after_fasterquant",
                            )
                        quantizers["model.layers.%d.%s" % (i, name)] = gptq[name].quantizer
                        gptq[name].free()

            if analysis_hook is not None:
                analysis_hook("after_layer_quantized", {
                    "args": args,
                    "analyzer": analyzer,
                    "layer": layer,
                    "layer_idx": i,
                    "layers": layers,
                    "full": full,
                    "inps": inps,
                    "fp_inps": fp_inps,
                    "fp_inps_final": fp_inps_final,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "position_embeddings": position_embeddings,
                    "dev": dev,
                    "orig_device": orig_device,
                    "static_fisher_by_layer": static_fisher_by_layer,
                    "static_legacy_fisher_diag_by_layer": static_legacy_fisher_diag_by_layer,
                    "static_refined_A_by_layer": static_refined_A_by_layer,
                    "static_refined_diag_A_by_layer": static_refined_diag_A_by_layer,
                    "samples_per_A": refined_rkl_samples_per_A,
                    "fp_weights": analysis_fp_weights,
                    "hessians": analysis_hessians,
                    "is_target": analysis_is_target,
                })

            with layer_recorder.section("layer.quantized_replay_forward") if layer_recorder else _NULL_CONTEXT:
                for j in range(inps.shape[0]):
                    inps[j] = layer(
                        inps[j].unsqueeze(0).to(dev),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )[0].squeeze(0).to(inps.device)

            with layer_recorder.section("layer.cleanup") if layer_recorder else _NULL_CONTEXT:
                if slide_active_layer and slide_next_layer is not None:
                    del slide_fp_inps_next
                    layer_manager.release_layer(i + 1, slide_next_layer, update_master=False, orig_device=orig_device)
                    slide_next_layer = None
                    slide_fp_inps_next = None
                    slide_next_layer_output_fisher = None
                    slide_next_refined_A_list = None
                layer_manager.release_layer(i, layer, update_master=True, orig_device=orig_device)
                del layer
                del gptq
                del saliency_dict, gradients_dict
                layer_output_fisher = None
                layer_output_fisher_by_module = None
                layer_refined_A_list = None
                precomputed_saliency_for_layer = None
                precomputed_saliency_for_group = None
                dynsal_refresh_overrides = None
                if layer_refined_mse_grad_pool is not None:
                    del layer_refined_mse_grad_pool
                if layer_refined_mse_mean_grad is not None:
                    del layer_refined_mse_mean_grad
                if layer_refined_mse_pool_ids is not None:
                    del layer_refined_mse_pool_ids
                if layer_refined_mse_grad_pool_next is not None:
                    del layer_refined_mse_grad_pool_next
                if layer_refined_mse_mean_grad_next is not None:
                    del layer_refined_mse_mean_grad_next
                if P_fp_by_canonical is not None:
                    # By this point the 4 boundaries should have popped every
                    # module from P_fp_by_canonical (per_boundary mode) or we
                    # explicitly set it to None (per_layer mode). Explicit del
                    # bounds the lifetime for good measure and releases V_dev
                    # too.
                    del P_fp_by_canonical
                if dynsal_V_dev_for_layer is not None:
                    del dynsal_V_dev_for_layer
                if full_canonical is not None:
                    del full_canonical
                if precomputed_S_new_by_canonical is not None:
                    del precomputed_S_new_by_canonical
                if static_saliency_by_layer is not None and i < len(static_saliency_by_layer):
                    static_saliency_by_layer[i] = None
                if static_fisher_by_layer is not None and i < len(static_fisher_by_layer):
                    static_fisher_by_layer[i] = None
                if (
                    static_legacy_fisher_diag_by_layer is not None
                    and i < len(static_legacy_fisher_diag_by_layer)
                ):
                    static_legacy_fisher_diag_by_layer[i] = None
                if static_refined_A_by_layer is not None and i < len(static_refined_A_by_layer):
                    static_refined_A_by_layer[i] = None
                if static_refined_diag_A_by_layer is not None and i < len(static_refined_diag_A_by_layer):
                    static_refined_diag_A_by_layer[i] = None
                if static_dynsal is not None and isinstance(static_dynsal, dict):
                    _dyn_layers = static_dynsal.get("by_layer")
                    if _dyn_layers is not None and i < len(_dyn_layers):
                        _dyn_layers[i] = None
                memory_utils.cleanup_memory(trim_cpu=True)
            if analysis_hook is not None:
                dist_utils.barrier()

            if quant_stop_layer is not None and i >= quant_stop_layer:
                logging.info("Stopping quantization after transformer layer %d due to --quant_stop_layer.", i)
                break

        # Drain any pending async D2H from the final layer before
        # restore_modules starts moving tensors around. `cleanup_memory` below
        # would catch this via `torch.cuda.synchronize()`, but explicit sync
        # keeps the NVTX trace readable.
        if refined_mse_d2h_event is not None:
            refined_mse_d2h_event.synchronize()
            refined_mse_d2h_event = None

        with pipeline_recorder.section("pipeline.restore_modules") if pipeline_recorder else _NULL_CONTEXT:
            layer_manager.release_runtime_modules(per_layer_runtime_modules, orig_device)
            model.config.use_cache = use_cache
        memory_utils.cleanup_memory(verbos=True)

    # Flush per-recorder meta.json files (loss trajectory + auto-detected spikes).
    diagnostic_registry.finalize_all()

    if quant_profile_enabled:
        logging.info("Quant profile NVTX ranges emitted. Inspect them with Nsight Systems/Compute.")
        QuantProfileRecorder.dump_wall_summary()
    logging.info("-----GPTQPlus Quantization Done-----\n")
    return quantizers
