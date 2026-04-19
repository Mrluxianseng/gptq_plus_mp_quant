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

from utils import quant_utils, memory_utils, model_utils, dist_utils
from gptq_utils.diagnostics import DiagnosticRegistry, parse_diagnose_targets


# Single reusable no-op context. `nullcontext()` instances are stateless, so we
# avoid allocating a new one at every `with ... if profile_recorder else
# nullcontext():` site inside the fasterquant/add_batch hot loops.
_NULL_CONTEXT = nullcontext()


def format_log_value(value, digits=6):
    if value is None:
        return "None"
    return f"{float(value):.{digits}g}"


def normalize_quant_module_name(name: str) -> str:
    return name[:-7] if name.endswith(".module") else name


def get_effective_refresh_loss_type(layer_idx: int, final_layer_idx: int, default_refresh_loss_type: str) -> str:
    if layer_idx == final_layer_idx:
        return "kl"
    return default_refresh_loss_type


def get_effective_gptq_reference_loss_type(global_loss_enabled: bool, layer_refresh_loss_type: str) -> str:
    return layer_refresh_loss_type if global_loss_enabled else "kl"


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
        "cosine" -> 0.5 * (1 - cos(pi * x))   (smooth S-curve from 0 to 1)
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
        return 0.5 * (1.0 - math.cos(math.pi * x))
    if schedule == "sqrt":
        return math.sqrt(x)
    raise ValueError(f"Unknown grad_lr_layer_schedule={schedule!r}")


class GPTQPlus:
    def __init__(self,
        layer,
        saliency: torch.Tensor, # shape (N_local, seq_len, G) — rank-local shard in DP
        gradient: torch.Tensor, # shape (G, in_features)
        num_groups: int,
        alpha: float,
        reference_loss: float,
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

        self.saliencies = saliency.float()
        self.gradients = gradient.float()
        # Layout: (num_groups, columns, columns). Storing the group axis first
        # makes per-subgroup slices `self.H[g]` contiguous, so the bmm-based
        # accumulation in `add_batch` and the per-subgroup clone in
        # `fasterquant` get a fast contiguous read/write path.
        #
        # DP note: `H` / `act_square` hold an unnormalised SUM. `finalize_hessian`
        # all-reduces across ranks and applies the single global division. This
        # gives mathematically the same result as the old running-mean update
        # but is invariant to sample-order / sample-shard, which is required
        # once multiple ranks each see only part of `nsamples`.
        self.H = torch.zeros(
            (self.num_groups, self.columns, self.columns),
            device=self.dev
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

        # Assert row partition is valid:
        # we do the same partition as before:
        assert self.rows % self.num_groups == 0, (
            f"Number of rows ({self.rows}) must be divisible "
            f"by num_groups ({self.num_groups})"
        )

    def _selected_column_count(self, blocksize, max_blocks):
        if max_blocks is None:
            return self.columns
        return min(self.columns, max_blocks * blocksize)

    def _compute_gradient_terms(self, gradients_sub, Hinv_init, Hinv, enable_gradient_update):
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
        beta_view = beta.unsqueeze(1)
        Z = gradients_sub.matmul(Hinv.T) * beta_view
        GHinv = Z.matmul(Hinv)
        return beta, beta_view, Z, GHinv

    @staticmethod
    def _current_ghinv(base, current_weight, ref_weight, beta_view, refresh_mode):
        if refresh_mode == "frozen":
            return base
        return base + beta_view * (current_weight - ref_weight)

    @staticmethod
    def _make_grad_optimizer_state(weight_sub, grad_optimizer):
        if grad_optimizer not in {"sgd", "adam"}:
            raise ValueError(f"Unsupported `grad_optimizer={grad_optimizer}`. Expected one of: sgd, adam.")
        state = {"type": grad_optimizer, "step": 0}
        if grad_optimizer == "adam":
            state["exp_avg"] = torch.zeros_like(weight_sub)
            state["exp_avg_sq"] = torch.zeros_like(weight_sub)
        return state

    @staticmethod
    def _clear_grad_optimizer_state(opt_state, col_start, col_end):
        if opt_state is None or col_end <= col_start:
            return
        if opt_state["type"] == "adam":
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
        exp_avg = opt_state["exp_avg"][:, col_start:]
        exp_avg_sq = opt_state["exp_avg_sq"][:, col_start:]
        exp_avg.mul_(adam_beta1).add_(grad_slice, alpha=1 - adam_beta1)
        exp_avg_sq.mul_(adam_beta2).addcmul_(grad_slice, grad_slice, value=1 - adam_beta2)
        bias_correction1 = 1 - adam_beta1 ** opt_state["step"]
        bias_correction2 = 1 - adam_beta2 ** opt_state["step"]
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

            with profile_recorder.section("add_batch.prepare_inputs") if profile_recorder else _NULL_CONTEXT:
                if inp.dim() == 3:
                    inp = inp.reshape(-1, inp.shape[-1])
                    sal_batch = sal_batch.reshape(-1, sal_batch.shape[-1])
                inp = inp.float()
                sal_batch = sal_batch.float()
                n_tokens = inp.shape[0]

            with profile_recorder.section("add_batch.weighted_input") if profile_recorder else _NULL_CONTEXT:
                weighted = inp.unsqueeze(0).mul(sal_batch.transpose(0, 1).unsqueeze(-1))

            with profile_recorder.section("add_batch.hessian_block") if profile_recorder else _NULL_CONTEXT:
                inp_T_batched = inp.transpose(0, 1).unsqueeze(0).expand(self.num_groups, -1, -1)
                block = torch.bmm(inp_T_batched, weighted)

            with profile_recorder.section("add_batch.accumulate") if profile_recorder else _NULL_CONTEXT:
                # Pure sum; normalisation deferred to `finalize_hessian`. Tracking
                # `token_count` lets finalize recover seq_len = token_count / index.
                self.H.add_(block)
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

        _dist.allreduce_sum_(self.H)
        _dist.allreduce_sum_(self.act_square)
        total_samples = _dist.allreduce_sum_scalar(self.index)
        total_tokens = _dist.allreduce_sum_scalar(self.token_count)
        if total_samples <= 0 or total_tokens <= 0:
            raise RuntimeError("finalize_hessian called before any add_batch ran.")
        seq_len = total_tokens / total_samples
        self.H.div_(total_samples * seq_len)
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
        diagnostic_recorder=None,
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

            shared_groups = None
            if groupsize != -1:
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

                    damp_percent = percdamp
                    damp_auto_increment = 0.0015
                    while 1 > damp_percent > 0:
                        try:
                            with profile_recorder.section("fasterquant.subgroup.compute_hinv") if profile_recorder else _NULL_CONTEXT:
                                damp = damp_percent * torch.mean(torch.diag(H_sub))
                                diag = torch.arange(self.columns, device=self.dev)
                                H_sub[diag, diag] += damp
                                H_sub = torch.linalg.cholesky(H_sub)
                                H_sub = torch.cholesky_inverse(H_sub)
                                Hinv_init = H_sub
                                H_sub = torch.linalg.cholesky(H_sub, upper=True)
                                Hinv = H_sub
                            if not torch.isfinite(Hinv).all():
                                # Cholesky succeeded but Hinv has NaN/Inf —
                                # H was close enough to singular that the
                                # inverse overflowed. Route back into the
                                # damp-autoincrement path.
                                raise torch._C._LinAlgError(
                                    "Hinv contains non-finite values despite successful Cholesky"
                                )
                            break
                        except torch._C._LinAlgError as e:
                            logging.warning(f"Quantization: Current `damp_percent = {damp_percent:.5f}` is too low, auto-incrementing by `{damp_auto_increment:.5f}`")
                            damp_percent += damp_auto_increment

                    if not (0 < damp_percent < 1):
                        raise ValueError(f"Quantization: `damp_percent` must between 0 and 1. current is {damp_percent}")

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
                            "grad_optimizer_state": self._make_grad_optimizer_state(W_sub, grad_optimizer) if block_gd_mode else None,
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
                                        if groupsize != -1:
                                            idx = i1 + i
                                            if actorder:
                                                idx = state["perm"][idx]
                                            quantizer = state["groups"][idx // groupsize]

                                        with profile_recorder.section("fasterquant.column.quantize") if profile_recorder else _NULL_CONTEXT:
                                            q, int_weight, scale = quantizer.fake_quantize(
                                                w.unsqueeze(1),
                                                st_idx=state["row_start"],
                                                end_idx=state["row_end"],
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
                                    if groupsize != -1:
                                        idx = i1 + i
                                        if actorder:
                                            idx = state["perm"][idx]
                                        quantizer = state["groups"][idx // groupsize]
                                    q_fake, int_weight, scale = quantizer.fake_quantize(
                                        w_col,
                                        st_idx=state["row_start"],
                                        end_idx=state["row_end"],
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
                            trailing_grad_chunks = []
                            for block_state in block_states:
                                state = block_state["state"]
                                refreshed_grad_sub = refreshed_grad[state["row_start"]:state["row_end"], :]
                                if actorder:
                                    refreshed_grad_sub = refreshed_grad_sub[:, state["perm"]]
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
                            if trailing_grad_chunks:
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
                                        "first_order_raw_abs_mean": None,
                                        "first_order_raw_mean_row_l2": None,
                                        "first_order_update_abs_mean": None,
                                        "first_order_update_mean_row_l2": None,
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
                            if block_gd_mode and second_order_update.numel() > 0:
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

                            # Loss-slide-window schedule: α=1 at the first
                            # refresh, α=0 at the last. refresh_idx = i1 /
                            # blocksize, and n_refresh_total refreshes fire
                            # per module. When only one refresh fires we keep
                            # α=1 (pure current-layer loss).
                            refresh_idx = i1 // blocksize
                            slide_alpha = (
                                1.0 - refresh_idx / max(n_refresh_total - 1, 1)
                                if n_refresh_total > 1 else 1.0
                            )
                            refreshed_grad, refresh_meta = gradient_refresh_fn(
                                weight_snapshot,
                                slide_alpha=slide_alpha,
                            )
                            refreshed_grad = refreshed_grad.to(self.dev).float()
                            trailing_grad_chunks = []
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
                            first_order_raw_abs_mean = None
                            first_order_raw_mean_row_l2 = None
                            first_order_abs_mean = None
                            first_order_mean_row_l2 = None
                            regularizer_abs_mean = None
                            regularizer_mean_row_l2 = None
                            sine_regularizer_abs_mean = None
                            sine_regularizer_mean_row_l2 = None
                            optimizer_updates_raw = []
                            optimizer_updates = []
                            gate_regularizer_updates = []
                            sine_regularizer_updates = []
                            if block_second_order_chunks:
                                second_order_cat = torch.cat(block_second_order_chunks, dim=0).float()
                                second_order_abs_mean = second_order_cat.abs().mean().item()
                                second_order_mean_row_l2 = torch.linalg.norm(
                                    second_order_cat,
                                    dim=1,
                                ).mean().item()
                            if trailing_grad_chunks:
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
                                if optimizer_update_raw.numel() > 0:
                                    optimizer_updates_raw.append(optimizer_update_raw)
                                if optimizer_update.numel() > 0:
                                    optimizer_updates.append(optimizer_update)
                                    gate_regularizer_updates.append(gate_regularizer_update)
                                    sine_regularizer_updates.append(sine_regularizer_update)
                            if optimizer_updates_raw:
                                optimizer_update_raw_cat = torch.cat(optimizer_updates_raw, dim=0).float()
                                first_order_raw_abs_mean = optimizer_update_raw_cat.abs().mean().item()
                                first_order_raw_mean_row_l2 = torch.linalg.norm(
                                    optimizer_update_raw_cat,
                                    dim=1,
                                ).mean().item()
                            if optimizer_updates:
                                optimizer_update_cat = torch.cat(optimizer_updates, dim=0).float()
                                first_order_abs_mean = optimizer_update_cat.abs().mean().item()
                                first_order_mean_row_l2 = torch.linalg.norm(
                                    optimizer_update_cat,
                                    dim=1,
                                ).mean().item()
                            if gate_regularizer_updates:
                                regularizer_update_cat = torch.cat(gate_regularizer_updates, dim=0).float()
                                regularizer_abs_mean = regularizer_update_cat.abs().mean().item()
                                regularizer_mean_row_l2 = torch.linalg.norm(
                                    regularizer_update_cat,
                                    dim=1,
                                ).mean().item()
                            if sine_regularizer_updates:
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
                                        "first_order_raw_abs_mean": first_order_raw_abs_mean,
                                        "first_order_raw_mean_row_l2": first_order_raw_mean_row_l2,
                                        "first_order_update_abs_mean": first_order_abs_mean,
                                        "first_order_update_mean_row_l2": first_order_mean_row_l2,
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

            damp_percent = percdamp
            damp_auto_increment = 0.0015
            while 1 > damp_percent > 0:
                try:
                    damp = damp_percent * torch.mean(torch.diag(H_sub))
                    diag = torch.arange(self.columns, device=self.dev)
                    H_sub[diag, diag] += damp
                    H_sub = torch.linalg.cholesky(H_sub)
                    H_sub = torch.cholesky_inverse(H_sub)
                    Hinv_init = H_sub
                    H_sub = torch.linalg.cholesky(H_sub, upper=True)
                    Hinv = H_sub
                    break
                except torch._C._LinAlgError:
                    damp_percent += damp_auto_increment

            if not (0 < damp_percent < 1):
                raise ValueError(f"Quantization: `damp_percent` must between 0 and 1. current is {damp_percent}")

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

            damp_percent = percdamp
            damp_auto_increment = 0.0015
            while 1 > damp_percent > 0:
                try:
                    damp = damp_percent * torch.mean(torch.diag(H_sub))
                    diag = torch.arange(self.columns, device=self.dev)
                    H_sub[diag, diag] += damp
                    H_sub = torch.linalg.cholesky(H_sub)
                    H_sub = torch.cholesky_inverse(H_sub)
                    Hinv_init = H_sub
                    break
                except torch._C._LinAlgError:
                    damp_percent += damp_auto_increment

            if not (0 < damp_percent < 1):
                raise ValueError(f"Quantization: `damp_percent` must between 0 and 1. current is {damp_percent}")

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
    def __init__(self, names, num_groups):
        self.num_groups = num_groups
        self.saliency_cache = {}
        self.names = names
        for name in self.names:
            self.saliency_cache[name] = []
        self.handles = []
        self.hooks_enabled = False

    def cache_saliency(self, module, inp, out, name):
        # We'll store gradient on 'out', so we must retain it
        out.retain_grad()

        def grad_hook(grad):
            """
            grad shape typically [bsz, seq_len, hidden_dim].
            We group the channels, take abs, then average.
            """
            if not self.hooks_enabled:
                return
            bsz, seq_len, hidden_dim = grad.shape
            group_size = hidden_dim // self.num_groups

            grad_squared = grad.float().pow(2).view(bsz, seq_len, self.num_groups, group_size)
            mean_squared_grad = grad_squared.mean(dim=-1)  # -> [bsz, seq_len, num_groups]

            self.saliency_cache[name].append(mean_squared_grad)

        # Attach the gradient hook to 'out'
        out.register_hook(grad_hook)

    def add_hook(self, full, enable=True):
        for name in self.names:
            self.handles.append(
                full.get(name, full.get(name + ".module", None)).register_forward_hook(
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
            self.handles.append(
                full.get(name, full.get(name + ".module", None)).weight.register_hook(
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



def collect_static_end_to_end_saliency_and_fisher(
    *,
    model,
    analyzer,
    dataloader,
    dev,
    saliency_num_groups,
    fisher_num_groups,
    grad_hessian_topk,
    batch_size,
    collect_fisher=True,
    use_fsdp=False,
    fsdp_cpu_offload=False,
    saliency_clip_percentile=0.99,
):
    logging.info(
        "Collecting static end-to-end saliency/fisher caches from a single pre-quantization full-model backward pass. "
        "Using sampled end-to-end NLL / empirical Fisher because literal KL-to-self before quantization would be zero."
    )
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
    if use_fsdp:
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
        # FSDP-managed params live on-rank already; don't model.to(dev).
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
    saliency_data = [
        {module_name: [] for module_name in module_dict.keys()}
        for module_dict in module_dicts
    ]
    fisher_data = [[] for _ in layers]
    handles = []

    def make_module_hook(layer_idx, module_name):
        def forward_hook(module, inp, out):
            out_tensor = out[0] if isinstance(out, (tuple, list)) else out
            out_tensor.retain_grad()

            def grad_hook(grad):
                bsz_local, seq_len_local, hidden_dim = grad.shape
                if hidden_dim % saliency_num_groups != 0:
                    raise ValueError(
                        f"Module output dim ({hidden_dim}) must be divisible by saliency num_groups ({saliency_num_groups})."
                    )
                group_size = hidden_dim // saliency_num_groups
                grad_squared = grad.float().pow(2).view(
                    bsz_local,
                    seq_len_local,
                    saliency_num_groups,
                    group_size,
                )
                sal_per_group = grad_squared.mean(dim=-1)
                # Clip saliency outliers at a configurable percentile. For
                # deep layers the NLL backward amplifies a handful of tokens
                # by 10-12 orders of magnitude, which makes the downstream
                # weighted Hessian (`inp.T @ diag(s) @ inp`) effectively
                # rank-1 and Cholesky fails (even with big damp). Capping
                # the top fraction keeps the "which tokens matter" ordering
                # while bounding dynamic range; default 0.99 keeps 99% of
                # tokens' saliency untouched.
                if saliency_clip_percentile is not None and 0 < saliency_clip_percentile < 1:
                    flat = sal_per_group.detach().flatten()
                    # torch.quantile is O(n log n) but this tensor is small
                    # (batch * seq * NG elements per hook call), so negligible.
                    cap = torch.quantile(flat, saliency_clip_percentile)
                    sal_per_group = torch.clamp(sal_per_group, max=cap)
                saliency_data[layer_idx][module_name].append(
                    sal_per_group.detach().cpu()
                )

            out_tensor.register_hook(grad_hook)

        return forward_hook

    def make_layer_hook(layer_idx):
        def forward_hook(module, inp, out):
            out_tensor = out[0] if isinstance(out, (tuple, list)) else out
            out_tensor.retain_grad()

            def grad_hook(grad):
                bsz_local, seq_len_local, hidden_dim = grad.shape
                if hidden_dim % fisher_num_groups != 0:
                    raise ValueError(
                        f"Layer output dim ({hidden_dim}) must be divisible by fisher_num_groups ({fisher_num_groups})."
                    )
                group_size = hidden_dim // fisher_num_groups
                grad_squared = grad.float().pow(2).view(
                    bsz_local,
                    seq_len_local,
                    fisher_num_groups,
                    group_size,
                )
                # Cache as bf16 on CPU to halve RAM footprint. Consumers .float()
                # on `.to(dev)` so compute stays fp32 and numerics are unchanged.
                fisher_data[layer_idx].append(grad_squared.mean(dim=-1).detach().to(torch.bfloat16).cpu())

            out_tensor.register_hook(grad_hook)

        return forward_hook

    for layer_idx, (layer, module_dict) in enumerate(zip(layers, module_dicts)):
        if collect_fisher:
            handles.append(layer.register_forward_hook(make_layer_hook(layer_idx)))
        for module_name, module in module_dict.items():
            handles.append(module.register_forward_hook(make_module_hook(layer_idx, module_name)))

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
    # Contiguous shard of sample ids; each rank only does forward/backward on
    # its own slice and keeps the collected saliency/fisher rank-local. These
    # tensors are later consumed directly by rank-local `add_batch` calls
    # (sample index alignment is preserved because `inps` is sharded the same
    # way) — no all-gather is needed.
    shard = dist_utils.shard_slice(nsamples_total, rank, world)
    local_batches = token_batches[shard]
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
    try:
        with torch.enable_grad():
            for local_start in tqdm(
                range(0, len(local_batches), local_batch_size),
                ncols=120,
                desc="Static E2E Saliency/Fisher",
                position=1,
                leave=False,
            ):
                global_start = shard.start + local_start
                input_ids = torch.cat(local_batches[local_start:local_start + local_batch_size], dim=0).to(dev)
                outputs = model(input_ids=input_ids)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                teacher_logits = logits.detach()
                student_logits = logits
                if grad_hessian_topk > 0:
                    teacher_logits, indices = teacher_logits.topk(
                        grad_hessian_topk,
                        dim=-1,
                        sorted=False,
                    )
                    student_logits = student_logits.gather(-1, indices)
                # Per-sample deterministic label sampling: each sample's labels
                # only depend on its global sample id, making the draw
                # invariant to batching (1-GPU 16-per-batch vs 2-GPU 8-per-batch
                # both produce the same labels for the same global sample id).
                _batch_bsz = teacher_logits.shape[0]
                _global_indices = [global_start + _i for _i in range(_batch_bsz)]
                labels = _deterministic_categorical_labels(
                    teacher_logits,
                    _global_indices,
                    base_seed=0,  # global static saliency uses its own namespace
                )
                loss = F.cross_entropy(
                    student_logits.view(-1, student_logits.size(-1)),
                    labels.view(-1),
                    reduction="sum",
                )
                loss.backward()
                del outputs, logits, teacher_logits, student_logits, labels, loss, input_ids
    finally:
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

    static_saliency = []
    static_fisher = []
    for layer_idx, module_dict in enumerate(module_dicts):
        layer_saliency = {}
        for module_name in module_dict.keys():
            if not saliency_data[layer_idx][module_name]:
                raise ValueError(
                    f"Failed to collect static end-to-end saliency for layer={layer_idx} module={module_name}."
                )
            # Rank-local shard of shape (n_local, T, G). Not gathered.
            layer_saliency[module_name] = torch.cat(saliency_data[layer_idx][module_name], dim=0)
        static_saliency.append(layer_saliency)
        if collect_fisher:
            if not fisher_data[layer_idx]:
                raise ValueError(f"Failed to collect static end-to-end Fisher for layer={layer_idx}.")
            static_fisher.append(torch.cat(fisher_data[layer_idx], dim=0))
        else:
            static_fisher.append(None)

    return static_saliency, static_fisher


def compute_refresh_loss(
    refresh_loss_type,
    out_hidden,
    fp_hidden,
    analyzer,
    kl_topk,
    layer_output_fisher=None,
    fp_final_hidden=None,
):
    if refresh_loss_type == "kl":
        logits = hidden2logits(out_hidden, analyzer)
        logits_fp = hidden2logits(fp_hidden, analyzer)
        if kl_topk > 0:
            logits_fp, indices = logits_fp.topk(kl_topk, dim=-1, sorted=False)
            logits = logits.gather(-1, indices)
        kl_loss = F.kl_div(
            F.log_softmax(logits, dim=-1),
            F.softmax(logits_fp, dim=-1),
            reduction="none",
        )
        return kl_loss.sum(dim=-1).mean()

    delta = out_hidden - fp_hidden
    if refresh_loss_type == "hidden_mse":
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
        final_with_delta = fp_final_hidden + delta
        logits_perturbed = hidden2logits(final_with_delta, analyzer)
        logits_fp = hidden2logits(fp_final_hidden, analyzer)
        if kl_topk > 0:
            logits_fp, indices = logits_fp.topk(kl_topk, dim=-1, sorted=False)
            logits_perturbed = logits_perturbed.gather(-1, indices)
        kl_loss = F.kl_div(
            F.log_softmax(logits_perturbed, dim=-1),
            F.softmax(logits_fp, dim=-1),
            reduction="none",
        )
        return kl_loss.sum(dim=-1).mean()

    if layer_output_fisher is None:
        raise ValueError("`layer_output_fisher` must be provided for refresh_loss_type='fisher_diag_mse'.")
    num_groups = layer_output_fisher.shape[-1]
    if delta.shape[-1] % num_groups != 0:
        raise ValueError(
            f"Output hidden dim ({delta.shape[-1]}) must be divisible by layer-output Fisher groups ({num_groups})."
        )
    hidden_size = delta.shape[-1]
    group_size = hidden_size // num_groups
    delta_grouped = delta.view(delta.shape[0], delta.shape[1], num_groups, group_size)
    # Normalize Fisher weights independently for each (batch, token) across the
    # group axis so the weighting reflects only relative saliency structure for
    # that token, not the absolute Fisher magnitude of the current batch slice.
    fisher_l2 = torch.linalg.vector_norm(layer_output_fisher, ord=2, dim=-1, keepdim=True)
    fisher_norm = layer_output_fisher / (fisher_l2 + 1e-12)
    weighted_sq = fisher_norm.unsqueeze(-1) * delta_grouped.square()
    return 0.5 * weighted_sq.sum(dim=(-1, -2)).mean() / hidden_size


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
    fisher_num_groups,
    kl_topk,
    grad_hessian_topk,
    dev,
    layer_idx,
    layer_recorder=None,
):
    layer_output_fisher_cache = []
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
                    logits = hidden2logits(out, analyzer)
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
                    kl_loss = F.kl_div(
                        F.log_softmax(kl_logits, dim=-1),
                        F.softmax(kl_logits_fp, dim=-1),
                        reduction="none",
                    )
                    kl_loss = kl_loss.sum(dim=-1).mean()

                with layer_recorder.section("layer.pre_quant_fisher.backward") if layer_recorder else _NULL_CONTEXT:
                    out_hidden.retain_grad()

                    def layer_output_grad_hook(grad):
                        bsz_local, seq_len_local, hidden_dim = grad.shape
                        if hidden_dim % fisher_num_groups != 0:
                            raise ValueError(
                                f"Output hidden dim ({hidden_dim}) must be divisible by fisher_num_groups ({fisher_num_groups})."
                            )
                        group_size = hidden_dim // fisher_num_groups
                        token_count = bsz_local * seq_len_local
                        grad_unmean = grad.float() * token_count
                        grad_squared = grad_unmean.pow(2).view(
                            bsz_local, seq_len_local, fisher_num_groups, group_size
                        )
                        layer_output_fisher_cache.append(grad_squared.mean(dim=-1).detach())

                    out_hidden.register_hook(layer_output_grad_hook)
                    model.zero_grad()
                    kl_loss.backward()

                # Per-batch cleanup_memory() was here. Removed for the same
                # reason as above — autograd state is released automatically.

    return torch.cat(layer_output_fisher_cache, dim=0) if layer_output_fisher_cache else None


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
    global_shuffle=False,
    dp_rank=0,
    shard_size=None,
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
    module = full.get(module_name, full.get(module_name + ".module", None))
    if module is None:
        raise ValueError(f"Unable to find module `{module_name}` in the provided layer.")
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
    partial_count = 0
    loss_sum = 0.0
    loss_sum_current = 0.0
    loss_sum_next = 0.0

    # Slide-window mixes in a "next-layer" version of the same loss.
    #   fisher_diag_mse path: needs the next-layer fisher tensor.
    #   residual_kl    path: reuses fp_inps_final (same final-head target for
    #                         both current- and next-layer deltas).
    slide_active = (
        slide_alpha < 1.0
        and next_layer is not None
        and fp_inps_next is not None
        and (
            (refresh_loss_type == "fisher_diag_mse" and next_layer_output_fisher is not None)
            or (refresh_loss_type == "residual_kl" and fp_inps_final is not None)
        )
    )

    if len(selected_indices) > 0:
        grad_modules = [layer]
        if refresh_loss_type in ("kl", "residual_kl"):
            # `residual_kl` also sends grad through the final norm + lm_head,
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
                for start in range(0, len(selected_indices), bsz):
                    batch_indices = selected_indices[start:start + bsz]
                    batch_size = len(batch_indices)
                    batch_attention_mask = attention_mask.expand(batch_size, -1, -1, -1)
                    batch_position_ids = position_ids.expand(batch_size, -1)
                    batch_position_embeddings = (
                        position_embeddings[0].expand(batch_size, -1, -1),
                        position_embeddings[1].expand(batch_size, -1, -1),
                    )

                    out = functional_call(
                        layer,
                        {f"{module_name}.weight": override_weight},
                        (inps[batch_indices].to(dev),),
                        {
                            "attention_mask": batch_attention_mask,
                            "position_ids": batch_position_ids,
                            "position_embeddings": batch_position_embeddings,
                        },
                        strict=False,
                    )
                    out_hidden = out[0] if isinstance(out, (tuple, list)) else out
                    fp_hidden = fp_inps[batch_indices].to(dev)
                    fisher_batch = (
                        None if layer_output_fisher is None
                        else layer_output_fisher[batch_indices].to(dev).float()
                    )
                    fp_final_batch = (
                        None if fp_inps_final is None
                        else fp_inps_final[batch_indices].to(dev)
                    )
                    refresh_loss_current = compute_refresh_loss(
                        refresh_loss_type,
                        out_hidden,
                        fp_hidden,
                        analyzer,
                        kl_topk,
                        layer_output_fisher=fisher_batch,
                        fp_final_hidden=fp_final_batch,
                    )
                    if slide_active:
                        next_out = next_layer(
                            out_hidden,
                            attention_mask=batch_attention_mask,
                            position_ids=batch_position_ids,
                            position_embeddings=batch_position_embeddings,
                        )
                        next_out_hidden = next_out[0] if isinstance(next_out, (tuple, list)) else next_out
                        fp_hidden_next = fp_inps_next[batch_indices].to(dev)
                        # For fisher_diag_mse the next-layer loss needs the
                        # next-layer fisher diagonal; for residual_kl it reuses
                        # the same fp_inps_final as the current-layer loss.
                        fisher_batch_next = (
                            None if next_layer_output_fisher is None
                            else next_layer_output_fisher[batch_indices].to(dev).float()
                        )
                        refresh_loss_next = compute_refresh_loss(
                            refresh_loss_type,
                            next_out_hidden,
                            fp_hidden_next,
                            analyzer,
                            kl_topk,
                            layer_output_fisher=fisher_batch_next,
                            fp_final_hidden=fp_final_batch,
                        )
                        refresh_loss = (
                            slide_alpha * refresh_loss_current
                            + (1.0 - slide_alpha) * refresh_loss_next
                        )
                        loss_sum_current += refresh_loss_current.item() * batch_size
                        loss_sum_next += refresh_loss_next.item() * batch_size
                    else:
                        refresh_loss = refresh_loss_current

                    batch_grad_mean = torch.autograd.grad(
                        refresh_loss, override_weight, retain_graph=False
                    )[0].float()
                    # `autograd.grad` returns ∂(mean_loss)/∂W. Multiply by batch
                    # size to recover a sum-over-samples gradient so per-rank
                    # partials aggregate with a plain allreduce_sum.
                    partial_grad_sum.add_(batch_grad_mean, alpha=float(batch_size))
                    loss_sum += refresh_loss.item() * batch_size
                    partial_count += batch_size
                    # Per-batch `cleanup_memory()` used to run here; removing
                    # it trades a small increase in peak transient memory for
                    # dropping ~5-20 ms/call of gc.collect + synchronize +
                    # empty_cache on thousands of refreshes. PyTorch reuses
                    # cached blocks, so this is safe as long as no outer
                    # autograd graph leaks across the loop.

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
    fisher_num_groups,
    kl_topk,
    grad_hessian_topk,
    dev,
    layer_idx,
    layer_refresh_loss_type,
    gptq_reference_loss_type,
    precomputed_saliency_dict=None,
    precomputed_layer_output_fisher=None,
    fp_inps_final=None,
    layer_recorder=None,
    skip_gradient_backward=False,
):
    need_saliency_collection = precomputed_saliency_dict is None
    # When the caller sets skip_gradient_backward, we're running pure GPTQ with
    # enable_gptq_plus=0: no gradient reference loss, no fisher collection on
    # this path (fisher only feeds fisher_diag_mse refresh, which is also off).
    need_layer_output_fisher_collection = (
        not skip_gradient_backward
        and layer_refresh_loss_type == "fisher_diag_mse"
        and precomputed_layer_output_fisher is None
    )
    need_gradient_backward = not skip_gradient_backward
    need_output_head = (
        need_saliency_collection
        or (need_gradient_backward and (
            layer_refresh_loss_type == "kl"
            or gptq_reference_loss_type == "kl"
        ))
        or need_layer_output_fisher_collection
    )
    with torch.enable_grad():
        saliency_cache = None
        if need_saliency_collection:
            saliency_cache = SaliencyCache(names, num_groups)
            saliency_cache.add_hook(full, enable=False)
        gradients_cache = GradientCache(names, num_groups)
        gradients_cache.add_hook(full, enable=False)
        layer_output_fisher_cache = []
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
                            logits = hidden2logits(out, analyzer)
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
                            nll_loss = F.cross_entropy(
                                grad_hessian_logits.view(-1, grad_hessian_logits.size(-1)),
                                labels.view(-1),
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
                if layer_refresh_loss_type == "fisher_diag_mse" and precomputed_layer_output_fisher is not None:
                    with layer_recorder.section("layer.grad_hessian.fisher_slice") if layer_recorder else _NULL_CONTEXT:
                        batch_layer_output_fisher = precomputed_layer_output_fisher[j : j + bsz].to(dev).float()
                elif need_layer_output_fisher_collection:
                    with layer_recorder.section("layer.grad_hessian.fisher_collect") if layer_recorder else _NULL_CONTEXT:
                        kl_logits = grad_hessian_logits if grad_hessian_topk > 0 else logits
                        kl_logits_fp = grad_hessian_logits_fp if grad_hessian_topk > 0 else logits_fp
                        if grad_hessian_topk <= 0 and kl_topk > 0:
                            with layer_recorder.section("layer.grad_hessian.fisher_collect.topk") if layer_recorder else _NULL_CONTEXT:
                                kl_logits_fp, indices = logits_fp.topk(kl_topk, dim=-1, sorted=False)
                                kl_logits = logits.gather(-1, indices)
                        with layer_recorder.section("layer.grad_hessian.fisher_collect.loss_build") if layer_recorder else _NULL_CONTEXT:
                            fisher_kl_loss = F.kl_div(
                                F.log_softmax(kl_logits, dim=-1),
                                F.softmax(kl_logits_fp, dim=-1),
                                reduction="none",
                            )
                            fisher_kl_loss = fisher_kl_loss.sum(dim=-1).mean()
                        with layer_recorder.section("layer.grad_hessian.fisher_collect.hook_register") if layer_recorder else _NULL_CONTEXT:
                            out_hidden.retain_grad()

                            def layer_output_grad_hook(grad):
                                bsz_local, seq_len_local, hidden_dim = grad.shape
                                if hidden_dim % fisher_num_groups != 0:
                                    raise ValueError(
                                        f"Output hidden dim ({hidden_dim}) must be divisible by fisher_num_groups ({fisher_num_groups})."
                                    )
                                group_size = hidden_dim // fisher_num_groups
                                token_count = bsz_local * seq_len_local
                                grad_unmean = grad.float() * token_count
                                grad_squared = grad_unmean.pow(2).view(
                                    bsz_local, seq_len_local, fisher_num_groups, group_size
                                )
                                layer_output_fisher_cache.append(grad_squared.mean(dim=-1).detach())

                            out_hidden.register_hook(layer_output_grad_hook)
                        with layer_recorder.section("layer.grad_hessian.fisher_collect.backward") if layer_recorder else _NULL_CONTEXT:
                            model.zero_grad()
                            fisher_kl_loss.backward(retain_graph=True)
                        batch_layer_output_fisher = layer_output_fisher_cache[-1]

                if need_gradient_backward:
                    with layer_recorder.section("layer.grad_hessian.gradient_loss_build") if layer_recorder else _NULL_CONTEXT:
                        if gptq_reference_loss_type == "kl":
                            kl_logits = grad_hessian_logits if grad_hessian_topk > 0 else logits
                            kl_logits_fp = grad_hessian_logits_fp if grad_hessian_topk > 0 else logits_fp
                            if grad_hessian_topk <= 0 and kl_topk > 0:
                                with layer_recorder.section("layer.grad_hessian.gradient_loss_build.topk") if layer_recorder else _NULL_CONTEXT:
                                    kl_logits_fp, indices = logits_fp.topk(kl_topk, dim=-1, sorted=False)
                                    kl_logits = logits.gather(-1, indices)
                            gradient_loss = F.kl_div(
                                F.log_softmax(kl_logits, dim=-1),
                                F.softmax(kl_logits_fp, dim=-1),
                                reduction="none",
                            )
                            gradient_loss = gradient_loss.sum(dim=-1).mean()
                        else:
                            fp_final_batch = (
                                None if fp_inps_final is None
                                else fp_inps_final[j : j + bsz].to(dev)
                            )
                            gradient_loss = compute_refresh_loss(
                                gptq_reference_loss_type,
                                out_hidden,
                                fp_hidden,
                                analyzer,
                                kl_topk,
                                layer_output_fisher=batch_layer_output_fisher,
                                fp_final_hidden=fp_final_batch,
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
            module = full.get(name, full.get(name + ".module", None))
            gradients_cache.gradients_cache[name] = torch.zeros_like(
                module.weight.data, dtype=torch.float32
            )
    layer_output_fisher = precomputed_layer_output_fisher
    if need_layer_output_fisher_collection:
        # Keep the layer-output Fisher sharded across ranks — every consumer
        # indexes it by rank-local sample ids (fp_inps local shard).
        layer_output_fisher = torch.cat(layer_output_fisher_cache, dim=0)

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
    global_shuffle=False,
    dp_rank=0,
    shard_size=None,
):
    if num_steps <= 0 or not module_names:
        return

    modules = []
    for module_name in module_names:
        module = full.get(module_name, full.get(module_name + ".module", None))
        if module is None:
            raise ValueError(f"Unable to find module `{module_name}` in the provided layer.")
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
        sample_indices = scheduler.next_indices()
        step_losses = []
        step_update_abs = []
        for module_name, module in modules:
            fisher_tensor = layer_output_fisher_by_module.get(module_name)
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
                    global_shuffle=global_shuffle,
                    dp_rank=dp_rank,
                    shard_size=shard_size if shard_size is not None else inps.shape[0],
                )
            )
            # DP aggregation via sum + count. The same code path handles both
            # stratified (equal counts per rank) and global-shuffle (possibly
            # uneven counts, even zero on some ranks).
            if world > 1:
                dist_utils.allreduce_sum_(partial_grad_sum)
                global_count = dist_utils.allreduce_sum_scalar(partial_count)
                global_loss_sum = dist_utils.allreduce_sum_scalar(loss_sum)
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
    orig_device = next(model.parameters()).device

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
    global_loss_enabled = bool(getattr(args, "global_loss", False))
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

    with run_recorder.section("run.total") if run_recorder else _NULL_CONTEXT:
        if global_loss_enabled:
            # Optional disk cache. The precompute result depends only on:
            #   model / dataset / nsamples / seq_len / rotate setting /
            #   num_groups / fisher_num_groups / grad_hessian_topk /
            #   global_loss_bsz / seed.
            # Use `--static_cache_path DIR` to persist. Each rank writes/reads
            # its own shard file since saliency/fisher are rank-local.
            static_cache_dir = getattr(args, "static_cache_path", None)
            static_cache_key = None
            if static_cache_dir is not None:
                dataset_id = getattr(args, "dataset", "unknown")
                rotate_flag = int(bool(getattr(args, "rotate", False)))
                sal_clip_pct = getattr(args, "saliency_clip_percentile", 0.99)
                sal_clip_tag = f"{sal_clip_pct:.4f}".rstrip("0").rstrip(".")
                static_cache_key = (
                    f"{args.model_name}_{dataset_id}_s{args.nsamples}_"
                    f"blk{args.seq_len}_rot{rotate_flag}_g{args.num_groups}_"
                    f"fng{args.fisher_num_groups}_ghtk{args.grad_hessian_topk}_"
                    f"glbsz{args.global_loss_bsz}_seed{args.seed}_"
                    f"salclip{sal_clip_tag}"
                )
                static_cache_key += f"_world{dist_utils.get_world_size()}_rank{dist_utils.get_rank()}"
                os.makedirs(static_cache_dir, exist_ok=True)
            static_cache_file = (
                os.path.join(static_cache_dir, f"{static_cache_key}.pt")
                if static_cache_key is not None else None
            )

            if static_cache_file is not None and os.path.exists(static_cache_file):
                logging.info("Loading static saliency/fisher cache from %s", static_cache_file)
                _loaded = torch.load(static_cache_file, map_location="cpu", weights_only=True)
                static_saliency_by_layer = _loaded["saliency"]
                static_fisher_by_layer = _loaded["fisher"]
            else:
                with pipeline_recorder.section("pipeline.static_end_to_end_saliency_fisher") if pipeline_recorder else _NULL_CONTEXT:
                    static_saliency_by_layer, static_fisher_by_layer = collect_static_end_to_end_saliency_and_fisher(
                        model=model,
                        analyzer=analyzer,
                        dataloader=dataloader,
                        dev=dev,
                        saliency_num_groups=args.num_groups,
                        fisher_num_groups=args.fisher_num_groups,
                        grad_hessian_topk=args.grad_hessian_topk,
                        batch_size=args.global_loss_bsz,
                        use_fsdp=bool(getattr(args, "fsdp_precompute", False)),
                        fsdp_cpu_offload=bool(getattr(args, "fsdp_cpu_offload", False)),
                        saliency_clip_percentile=getattr(args, "saliency_clip_percentile", 0.99),
                    )
                if static_cache_file is not None:
                    logging.info("Saving static saliency/fisher cache to %s", static_cache_file)
                    torch.save(
                        {"saliency": static_saliency_by_layer, "fisher": static_fisher_by_layer},
                        static_cache_file,
                    )
            logging.info(
                "Collected frozen end-to-end saliency/Fisher caches before quantization with global_loss_bsz=%d. "
                "These cached coefficients will be reused for Hessian estimation and fisher_diag_mse throughout quantization.",
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
            for module in per_layer_runtime_modules:
                module.to(dev)
            layers[0] = layers[0].to(dev)

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

        layers[0] = layers[0].to(orig_device)
        memory_utils.cleanup_memory(False)

        attention_mask = cache["attention_mask"]
        position_ids = cache["position_ids"]
        position_embeddings = cache["position_embeddings"]

        sequential = analyzer.get_sequential_quantizable_module_names()
        names = [n for ns in sequential for n in ns]
        fp_inps = inps.clone()

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
                # scheduler with the same seed + same total_samples, so
                # `next_indices()` returns the identical global id list on
                # every rank. Each rank then filters to its own shard inside
                # `collect_true_weight_gradient`.
                gradient_refresh_scheduler = BackwardSampleScheduler(
                    args.nsamples,
                    args.backward_samples,
                    seed=args.seed,
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
                    seed=args.seed + dp_rank,
                )
        final_layer_idx = len(layers) - 1

        # residual_kl refresh loss needs the FP output of the last transformer
        # block per sample (input to final norm + lm_head). We compute it once
        # here with a full FP forward over all layers and store a buffer
        # `fp_inps_final` with the same (n_local, seq, hidden) shape as fp_inps.
        # Any other refresh loss keeps `fp_inps_final = None` and pays nothing.
        fp_inps_final = None
        if args.grad_refresh_loss == "residual_kl":
            with pipeline_recorder.section("pipeline.fp_final_precompute") if pipeline_recorder else _NULL_CONTEXT:
                logging.info(
                    "Precomputing FP final-layer hidden states for residual_kl "
                    "(nsamples_local=%d, layers=%d).",
                    inps.shape[0], len(layers),
                )
                # Work on a scratch copy so we don't disturb the main `fp_inps`
                # buffer (which is still at the layer-0 input stage).
                scratch = inps.detach().clone().to(dev)
                for idx in range(len(layers)):
                    lay = layers[idx].to(dev)
                    bits_cfg = quant_utils.disable_act_quant(lay)
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
                    quant_utils.enable_act_quant(lay, bits_cfg)
                    # Return each layer to its original (CPU) residency so the
                    # main quant loop's `layers[i].to(dev)` starts from the same
                    # state as if this precompute never happened.
                    layers[idx] = lay.to(orig_device)
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
        pbar = tqdm(layer_indices, ncols=120, desc="Quantizing Layers", position=0)
        for i in pbar:
            layer = layers[i].to(dev)
            full = analyzer.get_quantizable_modules(layer)
            layer_recorder = QuantProfileRecorder(dev, prefix=f"layers.{i}") if quant_profile_enabled else None
            # Per-layer LR ramp. scale==1.0 when schedule="none", so this is
            # a no-op for the default config.
            grad_lr_layer_scale = compute_layer_lr_scale(
                layer_idx=i,
                num_layers=len(layers),
                schedule=getattr(args, "grad_lr_layer_schedule", "none"),
            )
            if getattr(args, "grad_lr_layer_schedule", "none") != "none":
                logging.info(
                    "Layer %d grad_lr schedule scale=%.4f (schedule=%s)",
                    i, grad_lr_layer_scale, args.grad_lr_layer_schedule,
                )
            layer_refresh_loss_type = get_effective_refresh_loss_type(
                i,
                final_layer_idx,
                args.grad_refresh_loss,
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

            with layer_recorder.section("layer.fp_reference_forward") if layer_recorder else _NULL_CONTEXT:
                bits_config = quant_utils.disable_act_quant(layer)
                # inps/fp_inps are rank-local shards of length n_local.
                for j in range(inps.shape[0]):
                    fp_inps[j] = layer(
                        fp_inps[j].unsqueeze(0).to(dev),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )[0].to(fp_inps.device)
                quant_utils.enable_act_quant(layer, bits_config)

            # --- Loss-slide-window setup: precompute reference output of the
            # next FP transformer block, so the per-block refresh can blend the
            # current-layer loss with the next-layer loss. Supported under two
            # refresh types:
            #   * fisher_diag_mse — needs `static_fisher_by_layer[i+1]` (so
            #     global_loss must be on).
            #   * residual_kl    — reuses the already-precomputed fp_inps_final
            #     (no next-layer fisher needed).
            # Skipped at and past the second-to-last layer per the spec.
            slide_active_layer = False
            if (
                getattr(args, "loss_slide_window", False)
                and args.g_update_mode == "block_gd"
                and i <= final_layer_idx - 2
            ):
                if (
                    layer_refresh_loss_type == "fisher_diag_mse"
                    and global_loss_enabled
                    and static_fisher_by_layer[i + 1] is not None
                ):
                    slide_active_layer = True
                elif (
                    layer_refresh_loss_type == "residual_kl"
                    and fp_inps_final is not None
                ):
                    slide_active_layer = True
            slide_next_layer = None
            slide_fp_inps_next = None
            slide_next_layer_output_fisher = None
            slide_next_bits_config = None
            if slide_active_layer:
                with layer_recorder.section("layer.slide_window.next_fp_reference") if layer_recorder else _NULL_CONTEXT:
                    slide_next_layer = layers[i + 1].to(dev)
                    slide_next_bits_config = quant_utils.disable_act_quant(slide_next_layer)
                    # Only fisher_diag_mse needs the next-layer fisher. For
                    # residual_kl we leave this None and the loss computation
                    # in collect_true_weight_gradient routes through
                    # fp_inps_final instead.
                    if layer_refresh_loss_type == "fisher_diag_mse":
                        slide_next_layer_output_fisher = static_fisher_by_layer[i + 1]
                    slide_fp_inps_next = torch.empty_like(fp_inps)
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

            batch_attention_mask = attention_mask.expand(args.bsz, -1, -1, -1)
            batch_position_ids = position_ids.expand(args.bsz, -1)
            batch_position_embeddings = (
                position_embeddings[0].expand(args.bsz, -1, -1),
                position_embeddings[1].expand(args.bsz, -1, -1),
            )

            layer_output_fisher = None
            subset = {n: full.get(n, full.get(n + ".module", None)) for n in names}
            layer_output_fisher_by_module = {}
            pre_gd_refresh_loss_type = layer_refresh_loss_type
            if effective_pre_gd_steps > 0 and pre_gd_refresh_loss_type == "fisher_diag_mse":
                layer_output_fisher = static_fisher_by_layer[i]
                if layer_output_fisher is None:
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
                            fisher_num_groups=args.fisher_num_groups,
                            kl_topk=args.kl_topk,
                            grad_hessian_topk=args.grad_hessian_topk,
                            dev=dev,
                            layer_idx=i,
                            layer_recorder=layer_recorder,
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
                    else args.pre_grad_lr * grad_lr_layer_scale
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
                            grad_clip=args.grad_clip,
                            refresh_loss_type=layer_refresh_loss_type,
                            layer_output_fisher_by_module=layer_output_fisher_by_module,
                            fp_inps_final=fp_inps_final,
                            global_shuffle=dp_global_shuffle,
                            dp_rank=dp_rank,
                            shard_size=n_local,
                        )

            saliency_dict, gradients_dict, mean_reference_loss, layer_output_fisher = collect_layer_grad_hessian_stats(
                model=model,
                layer=layer,
                analyzer=analyzer,
                full=full,
                names=names,
                inps=inps,
                fp_inps=fp_inps,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                bsz=layer_stats_bsz,
                num_groups=args.num_groups,
                fisher_num_groups=args.fisher_num_groups,
                kl_topk=args.kl_topk,
                grad_hessian_topk=args.grad_hessian_topk,
                dev=dev,
                layer_idx=i,
                layer_refresh_loss_type=layer_refresh_loss_type,
                gptq_reference_loss_type=gptq_reference_loss_type,
                precomputed_saliency_dict=static_saliency_by_layer[i],
                precomputed_layer_output_fisher=(
                    static_fisher_by_layer[i]
                    if layer_refresh_loss_type == "fisher_diag_mse"
                    else None
                ),
                fp_inps_final=fp_inps_final,
                layer_recorder=layer_recorder,
                skip_gradient_backward=skip_ref_backward,
            )

            gptq = {}
            with layer_recorder.section("layer.gptq_setup") if layer_recorder else _NULL_CONTEXT:
                for name in subset:
                    layer_weight_bits = args.w_bits
                    layer_weight_sym = not args.w_asym
                    if "lm_head" in name:
                        continue
                    saliency = saliency_dict.get(name, saliency_dict.get(name + ".module", None))
                    if saliency is None:
                        raise KeyError(
                            f"Missing saliency cache for layer={i} module={name}. "
                            f"Available keys: {sorted(saliency_dict.keys())}"
                        )

                    gptq[name] = GPTQPlus(
                        subset[name],
                        saliency=saliency,
                        gradient=gradients_dict[name],
                        num_groups=args.num_groups,
                        alpha=args.alpha,
                        reference_loss=mean_reference_loss,
                    )
                    gptq[name].quantizer = quant_utils.WeightQuantizer()
                    gptq[name].quantizer.configure(
                        layer_weight_bits,
                        perchannel=True,
                        sym=layer_weight_sym,
                        mse=args.w_clip,
                    )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            add_batch_recorders = {}
            for name in gptq:
                if should_profile_module(i, name):
                    add_batch_recorders[name] = QuantProfileRecorder(dev, prefix=f"layers.{i}.{name}")
                    gptq[name].profile_recorder = add_batch_recorders[name]
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            with layer_recorder.section("layer.hessian_accumulation_forward") if layer_recorder else _NULL_CONTEXT:
                # Batch the accumulation forward so we don't pay a per-sample
                # kernel-launch tax. `add_batch` already handles arbitrary
                # batch sizes (it reshapes to [bsz*seq, dim] internally), so
                # the math is bit-exact regardless of bsz.
                hessian_accum_bsz = args.hessian_accum_bsz if args.hessian_accum_bsz is not None else args.bsz
                hessian_accum_bsz = max(1, min(hessian_accum_bsz, inps.shape[0]))
                for j in tqdm(
                    range(0, inps.shape[0], hessian_accum_bsz),
                    ncols=120,
                    desc=f"Layer {i} Hessian accumulation",
                    position=1,
                    leave=False,
                ):
                    batch_bsz = min(hessian_accum_bsz, inps.shape[0] - j)
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
            for name in add_batch_recorders:
                gptq[name].profile_recorder = None

            # Close out the Hessian accumulation: all-reduce the per-rank sums
            # and apply the global normalisation exactly once. After this call
            # fasterquant sees a globally-averaged H that is bit-identical on
            # every rank (NCCL all_reduce is deterministic for a given op+shape).
            with layer_recorder.section("layer.hessian_finalize") if layer_recorder else _NULL_CONTEXT:
                for name in gptq:
                    gptq[name].finalize_hessian()

            def make_gradient_refresh_fn(
                module_name,
                slide_next_layer=None,
                slide_fp_inps_next=None,
                slide_next_layer_output_fisher=None,
            ):
                def refresh_fn(weight_snapshot, slide_alpha=1.0):
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
                            global_shuffle=dp_global_shuffle,
                            dp_rank=dp_rank,
                            shard_size=n_local,
                        )
                    )
                    # DP aggregation via sum + count. Works uniformly whether
                    # each rank got exactly `backward_samples_local` (stratified)
                    # or a binomial-distributed subset (global shuffle, and
                    # occasionally zero).
                    if dp_world > 1:
                        dist_utils.allreduce_sum_(partial_grad_sum)
                        global_count = dist_utils.allreduce_sum_scalar(partial_count)
                        global_loss_sum = dist_utils.allreduce_sum_scalar(loss_sum)
                        for k in ("loss_sum_current", "loss_sum_next"):
                            if k in grad_extras:
                                grad_extras[k] = dist_utils.allreduce_sum_scalar(
                                    grad_extras[k]
                                )
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
                    meta = {
                        "mean_refresh_loss": mean_refresh_loss,
                        "sample_indices": sample_indices,
                    }
                    if "slide_alpha" in grad_extras:
                        meta["slide_alpha"] = grad_extras["slide_alpha"]
                    if "loss_sum_current" in grad_extras:
                        meta["mean_refresh_loss_current"] = (
                            grad_extras["loss_sum_current"] / float(global_count)
                        )
                    if "loss_sum_next" in grad_extras:
                        meta["mean_refresh_loss_next"] = (
                            grad_extras["loss_sum_next"] / float(global_count)
                        )
                    return grad, meta

                return refresh_fn

            def make_block_observer(module_name, effective_grad_optimizer):
                def observer(payload):
                    logging.info(
                        "block-metrics layer=%d module=%s mode=%s grad_opt=%s block=%d cols=[%d,%d) remain=%d grad_abs_mean=%s grad_clipped_abs_mean=%s grad_row_l2=%s loss=%s refresh_loss=%s train_loss=%s val_loss=%s second_abs=%s first_raw_abs=%s first_abs=%s reg_abs=%s sine_abs=%s slide_alpha=%s loss_cur=%s loss_next=%s",
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
                        format_log_value(payload["first_order_raw_abs_mean"], digits=4),
                        format_log_value(payload["first_order_update_abs_mean"], digits=4),
                        format_log_value(payload["regularizer_update_abs_mean"], digits=4),
                        format_log_value(payload["sine_regularizer_update_abs_mean"], digits=4),
                        format_log_value(payload.get("slide_alpha"), digits=3),
                        format_log_value(payload.get("mean_refresh_loss_current"), digits=6),
                        format_log_value(payload.get("mean_refresh_loss_next"), digits=6),
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
                        else args.grad_lr * grad_lr_layer_scale
                    )
                    effective_grad_reg_strategy = "none" if i == final_layer_idx else args.grad_reg_strategy
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
                    gptq[name].fasterquant(
                        blocksize=args.blocksize,
                        percdamp=args.percdamp,
                        groupsize=layer_w_groupsize,
                        actorder=args.act_order,
                        static_groups=args.act_order,
                        enable_gradient_update=True,
                        g_update_mode=args.g_update_mode,
                        export_to_et=args.export_to_et,
                        profile_recorder=module_recorder,
                        gradient_refresh_fn=make_gradient_refresh_fn(
                            name,
                            slide_next_layer=slide_next_layer,
                            slide_fp_inps_next=slide_fp_inps_next,
                            slide_next_layer_output_fisher=slide_next_layer_output_fisher,
                        ) if args.g_update_mode in {"block_backward", "block_gd"} else None,
                        grad_lr=effective_grad_lr,
                        grad_optimizer=effective_grad_optimizer,
                        grad_reg_strategy=effective_grad_reg_strategy,
                        grad_reg_lambda=args.grad_reg_lambda,
                        grad_gate_floor=args.grad_gate_floor,
                        grad_gate_sharpness=args.grad_gate_sharpness,
                        grad_gate_sine_amp=args.grad_gate_sine_amp,
                        second_order_scale=args.second_order_scale,
                        block_atomic_quant=args.block_atomic_quant,
                        block_observer=make_block_observer(name, effective_grad_optimizer) if args.g_update_mode in {"block_backward", "block_gd"} else None,
                        grad_clip=args.grad_clip,
                        diagnostic_recorder=diagnostic_registry.get_or_create(i, name),
                    )
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
                    # Restore next-layer bit widths so iter i+1 sees a clean
                    # act-quant state when it calls disable_act_quant again.
                    quant_utils.enable_act_quant(slide_next_layer, slide_next_bits_config)
                    del slide_fp_inps_next
                    slide_next_layer = None
                    slide_fp_inps_next = None
                    slide_next_layer_output_fisher = None
                    slide_next_bits_config = None
                layers[i] = layer.to(orig_device)
                del layer
                del gptq
                del saliency_dict, gradients_dict
                memory_utils.cleanup_memory()

            if quant_stop_layer is not None and i >= quant_stop_layer:
                logging.info("Stopping quantization after transformer layer %d due to --quant_stop_layer.", i)
                break

        with pipeline_recorder.section("pipeline.restore_modules") if pipeline_recorder else _NULL_CONTEXT:
            for module in per_layer_runtime_modules:
                module.to(orig_device)
            model.config.use_cache = use_cache
        memory_utils.cleanup_memory(verbos=True)

    # Flush per-recorder meta.json files (loss trajectory + auto-detected spikes).
    diagnostic_registry.finalize_all()

    if quant_profile_enabled:
        logging.info("Quant profile NVTX ranges emitted. Inspect them with Nsight Systems/Compute.")
        QuantProfileRecorder.dump_wall_summary()
    logging.info("-----GPTQPlus Quantization Done-----\n")
    return quantizers
