import copy
import logging
import os
import math
import pprint
import functools
import random
from contextlib import contextmanager, nullcontext
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call

from utils import quant_utils, memory_utils, model_utils


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
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

    def summary(self):
        return {}


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


class GPTQPlus:
    def __init__(self, 
        layer, 
        saliency: torch.Tensor, # shape (N, seq_len, G)
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
        self.H = torch.zeros(
            (self.columns, self.columns, self.num_groups),
            device=self.dev
        )
        self.act_square = torch.zeros(
            (self.columns), device=self.dev
        )
        self.nsamples = saliency.shape[0]
        self.index = 0
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
        if opt_state["type"] == "sgd":
            if grad_clip is not None and grad_clip > 0:
                grad_slice = grad_slice.clamp(min=-grad_clip, max=grad_clip)
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

        We'll slice self.saliencies[index: index + batch_size]
        do the einsum => accumulate into self.H
        then index += batch_size.
        """
        profile_recorder = self.profile_recorder
        with profile_recorder.section("add_batch.total") if profile_recorder else nullcontext():
            # If input is 2D or 1D, reshape to [batch, seq_len, dim] for consistency
            if inp.dim() == 2:
                inp = inp.unsqueeze(0)  # => [1, seq_len, dim]
            else:
                assert inp.dim() == 3, "Input must be 2D or 3D. Got %dD." % inp.dim()

            with profile_recorder.section("add_batch.slice_saliency") if profile_recorder else nullcontext():
                bsz = inp.shape[0]
                # slice out shape => (bsz, seq_len, G)
                sal_batch = self.saliencies[self.index: self.index + bsz].to(self.dev)
                self.H *= self.index / (self.index + bsz)
                self.index += bsz

            with profile_recorder.section("add_batch.prepare_inputs") if profile_recorder else nullcontext():
                if inp.dim() == 3:
                    inp = inp.reshape(-1, inp.shape[-1])
                    sal_batch = sal_batch.reshape(-1, sal_batch.shape[-1])
                inp = inp.float()
                sal_batch = sal_batch.float()
                n_tokens = inp.shape[0]

            with profile_recorder.section("add_batch.weighted_input") if profile_recorder else nullcontext():
                sal_weighted_inp = torch.einsum("nj,ng->njg", inp, sal_batch)

            with profile_recorder.section("add_batch.hessian_block") if profile_recorder else nullcontext():
                block = torch.einsum("ni,njg->ijg", inp, sal_weighted_inp)

            with profile_recorder.section("add_batch.accumulate") if profile_recorder else nullcontext():
                self.H.add_(block, alpha=1 / (n_tokens * self.index))
                self.act_square.add_((inp ** 2).sum(0), alpha=1 / n_tokens)

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
    ):
        profile_recorder = profile_recorder or self.profile_recorder
        self._validate_grad_regularizer(
            grad_reg_strategy,
            grad_reg_lambda=grad_reg_lambda,
            grad_gate_floor=grad_gate_floor,
            grad_gate_sharpness=grad_gate_sharpness,
            grad_gate_sine_amp=grad_gate_sine_amp,
        )
        with profile_recorder.section("fasterquant.total") if profile_recorder else nullcontext():
            W = self.layer.weight.data.clone()
            W = W.float()
            block_gd_mode = g_update_mode == "block_gd"
            with profile_recorder.section("fasterquant.allocate_outputs") if profile_recorder else nullcontext():
                Q_final = torch.zeros_like(W)
                W_int_final = torch.zeros_like(W)
                Scale_final = torch.zeros_like(W)

            if not self.quantizer.ready():
                with profile_recorder.section("fasterquant.quantizer_find_params_initial") if profile_recorder else nullcontext():
                    self.quantizer.find_params(W)

            shared_groups = None
            if groupsize != -1:
                with profile_recorder.section("fasterquant.quantizer_build_groups") if profile_recorder else nullcontext():
                    shared_groups = []
                    for col_start in range(0, self.columns, groupsize):
                        col_end = min(col_start + groupsize, self.columns)
                        quantizer = copy.deepcopy(self.quantizer)
                        quantizer.find_params(W[:, col_start:col_end])
                        shared_groups.append(quantizer)

            rows_per_sub = self.rows // self.num_groups
            subgroup_states = []
            for sub_idx in range(self.num_groups):
                with profile_recorder.section("fasterquant.subgroup.total") if profile_recorder else nullcontext():
                    row_start = sub_idx * rows_per_sub
                    row_end = (sub_idx + 1) * rows_per_sub

                    with profile_recorder.section("fasterquant.subgroup.slice_inputs") if profile_recorder else nullcontext():
                        W_sub = W[row_start:row_end, :].clone()
                        H_sub = self.H[:, :, sub_idx].clone()
                        gradients_sub = self.gradients[row_start: row_end, :].to(self.dev).float().clone()
                        dead = torch.diag(H_sub) == 0
                        H_sub[dead, dead] = 1
                        W_sub[:, dead] = 0

                    groups = shared_groups

                    perm = None
                    invperm = None
                    if actorder:
                        with profile_recorder.section("fasterquant.subgroup.actorder_permute") if profile_recorder else nullcontext():
                            perm = torch.argsort(self.act_square, descending=True)
                            W_sub = W_sub[:, perm]
                            H_sub = H_sub[perm][:, perm]
                            gradients_sub = gradients_sub[:, perm]
                            invperm = torch.argsort(perm)
                    
                    with profile_recorder.section("fasterquant.subgroup.H_sub_clone") if profile_recorder else nullcontext():
                        hessian_reg = H_sub.clone()
                    with profile_recorder.section("fasterquant.subgroup.build_gate_quant_maps") if profile_recorder else nullcontext():
                        gate_scale, gate_zero = self._build_gate_quant_maps(
                            self.quantizer,
                            W_sub,
                            groupsize,
                            row_start=row_start,
                            row_end=row_end,
                            groups=groups,
                            perm=perm,
                        )

                    with profile_recorder.section("fasterquant.subgroup.allocate_buffers") if profile_recorder else nullcontext():
                        Losses = torch.zeros_like(W_sub)
                        Q = torch.zeros_like(W_sub)
                        W_int_sub = torch.zeros_like(W_sub)
                        Scale_sub = torch.zeros_like(W_sub)

                    damp_percent = percdamp
                    damp_auto_increment = 0.0015
                    while 1 > damp_percent > 0:
                        try:
                            with profile_recorder.section("fasterquant.subgroup.compute_hinv") if profile_recorder else nullcontext():
                                damp = damp_percent * torch.mean(torch.diag(H_sub))
                                diag = torch.arange(self.columns, device=self.dev)
                                H_sub[diag, diag] += damp
                                H_sub = torch.linalg.cholesky(H_sub)
                                H_sub = torch.cholesky_inverse(H_sub)
                                Hinv_init = H_sub
                                H_sub = torch.linalg.cholesky(H_sub, upper=True)
                                Hinv = H_sub
                            break
                        except torch._C._LinAlgError as e:
                            logging.warning(f"Quantization: Current `damp_percent = {damp_percent:.5f}` is too low, auto-incrementing by `{damp_auto_increment:.5f}`")
                            damp_percent += damp_auto_increment

                    if not (0 < damp_percent < 1):
                        raise ValueError(f"Quantization: `damp_percent` must between 0 and 1. current is {damp_percent}")

                    with profile_recorder.section("fasterquant.subgroup.init_ghinv") if profile_recorder else nullcontext():
                        beta, beta_view, Z, GHinv = self._compute_gradient_terms(
                            gradients_sub,
                            Hinv_init,
                            Hinv,
                            enable_gradient_update,
                        )

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
                            "Losses": Losses,
                            "W_int_sub": W_int_sub,
                            "Scale_sub": Scale_sub,
                            "beta": beta,
                            "beta_view": beta_view,
                            "Z": Z,
                            "GHinv": GHinv,
                            "anchor_weight": W_sub.clone(),
                            "full_precision_weight": W_sub.clone(),
                            "hessian_reg": hessian_reg,
                            "gate_scale": gate_scale,
                            "gate_zero": gate_zero,
                            "grad_optimizer_state": self._make_grad_optimizer_state(W_sub, grad_optimizer) if block_gd_mode else None,
                            "groups": groups,
                            "perm": perm,
                            "invperm": invperm,
                        }
                    )

            for i1 in range(0, self.columns, blocksize):
                with profile_recorder.section("fasterquant.block.total") if profile_recorder else nullcontext():
                    i2 = min(i1 + blocksize, self.columns)
                    count = i2 - i1
                    is_last_block = i2 >= self.columns
                    use_atomic_quant = block_atomic_quant and not is_last_block
                    D = torch.arange(count - 1, -1, -1).to(W)
                    block_states = []

                    for state in subgroup_states:
                        with profile_recorder.section("fasterquant.block.setup") if profile_recorder else nullcontext():
                            W1 = state["W_sub"][:, i1:i2].clone()
                            W_ref1 = state["anchor_weight"][:, i1:i2]
                            W_block_start = W1.clone()
                            Q1 = torch.zeros_like(W1)
                            W_int1 = torch.zeros_like(W1)
                            Scale1 = torch.zeros_like(W1).to(state["Scale_sub"].dtype)
                            Err1 = torch.zeros_like(W1)
                            Losses1 = torch.zeros_like(W1)
                            Hinv1 = state["Hinv"][i1:i2, i1:i2]
                            GHinv1 = state["GHinv"][:, i1:i2].clone()
                            Z1 = state["Z"][:, i1:i2]
                            inner_update_mode = "surrogate_online" if g_update_mode == "block_backward" else "frozen" if block_gd_mode else g_update_mode
                            GHinv1_eff = self._current_ghinv(
                                GHinv1,
                                W1 if inner_update_mode == "surrogate_online" else W_block_start,
                                W_ref1,
                                state["beta_view"],
                                inner_update_mode,
                            )

                        if use_atomic_quant:
                            if groupsize == -1:
                                # In per-row quantization each column uses the same row-wise scale,
                                # so atomic block quantization can quantize the whole block at once
                                # without changing the final quantized result.
                                with profile_recorder.section("fasterquant.block.atomic_quantize_full") if profile_recorder else nullcontext():
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
                                    with profile_recorder.section("fasterquant.column.total") if profile_recorder else nullcontext():
                                        w = W_block_start[:, i]

                                        quantizer = self.quantizer
                                        if groupsize != -1:
                                            idx = i1 + i
                                            if actorder:
                                                idx = state["perm"][idx]
                                            quantizer = state["groups"][idx // groupsize]

                                        with profile_recorder.section("fasterquant.column.quantize") if profile_recorder else nullcontext():
                                            q, int_weight, scale = quantizer.fake_quantize(
                                                w.unsqueeze(1),
                                                st_idx=state["row_start"],
                                                end_idx=state["row_end"],
                                            )
                                        Q1[:, i] = q.flatten()
                                        W_int1[:, i] = int_weight.flatten()
                                        Scale1[:, i] = scale.flatten()

                            with profile_recorder.section("fasterquant.block.atomic_err_solve") if profile_recorder else nullcontext():
                                residual_block = W_block_start - Q1 - GHinv1_eff
                                diag_view = torch.diagonal(Hinv1).unsqueeze(0)
                                Losses1.copy_(residual_block.square() / diag_view.square())
                                Err1.copy_(
                                    torch.linalg.solve_triangular(
                                        Hinv1.T,
                                        residual_block.T,
                                        upper=False,
                                    ).T
                                )
                        else:
                            for i in range(count):
                                with profile_recorder.section("fasterquant.column.total") if profile_recorder else nullcontext():
                                    w = W1[:, i]
                                    d = Hinv1[i, i]

                                    quantizer = self.quantizer
                                    if groupsize != -1:
                                        idx = i1 + i
                                        if actorder:
                                            idx = state["perm"][idx]
                                        quantizer = state["groups"][idx // groupsize]

                                    with profile_recorder.section("fasterquant.column.quantize") if profile_recorder else nullcontext():
                                        q, int_weight, scale = quantizer.fake_quantize(
                                            w.unsqueeze(1),
                                            st_idx=state["row_start"],
                                            end_idx=state["row_end"],
                                        )
                                    Q1[:, i] = q.flatten()
                                    q = q.flatten()
                                    W_int1[:, i] = int_weight.flatten()
                                    Scale1[:, i] = scale.flatten()

                                    Losses1[:, i] = (w - q - GHinv1_eff[:, i]) ** 2 / d**2

                                    with profile_recorder.section("fasterquant.column.inner_update_delta_w") if profile_recorder else nullcontext():
                                        err1 = (w - q - GHinv1_eff[:, i]) / d
                                        second_order_inner_update = err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                                        if block_gd_mode:
                                            W1[:, i:] -= second_order_scale * (
                                                second_order_inner_update + GHinv1_eff[:, i:]
                                            )
                                        else:
                                            W1[:, i:] -= second_order_inner_update + GHinv1_eff[:, i:]
                                        Err1[:, i] = err1

                                    with profile_recorder.section("fasterquant.column.inner_update_ghinv") if profile_recorder else nullcontext():
                                        GHinv1[:, i:] = GHinv1[:, i:] - Z1[:, i].unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                                        GHinv1_eff = self._current_ghinv(
                                            GHinv1,
                                            W1 if inner_update_mode == "surrogate_online" else W_block_start,
                                            W_ref1,
                                            state["beta_view"],
                                            inner_update_mode,
                                        )

                        with profile_recorder.section("fasterquant.block.writeback_inner") if profile_recorder else nullcontext():
                            state["Q"][:, i1:i2] = Q1
                            state["W_int_sub"][:, i1:i2] = W_int1
                            state["Scale_sub"][:, i1:i2] = Scale1
                            state["Losses"][:, i1:i2] = Losses1 / 2

                        current_sub_weight = state["W_sub"].clone()
                        if i1 > 0:
                            current_sub_weight[:, :i1] = state["Q"][:, :i1]
                        current_sub_weight[:, i1:i2] = Q1
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
                        with profile_recorder.section("fasterquant.block.true_gradient_refresh") if profile_recorder else nullcontext():
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
                            if trailing_grad_chunks:
                                trailing_grad_cat = torch.cat(trailing_grad_chunks, dim=0).float()
                                trailing_grad_abs_mean = trailing_grad_cat.abs().mean().item()
                                trailing_grad_mean_row_l2 = torch.linalg.norm(
                                    trailing_grad_cat,
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
                        with profile_recorder.section("fasterquant.block.outer_update_delta_w") if profile_recorder else nullcontext():
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
                            if block_gd_mode:
                                applied_second_order_update = second_order_scale * second_order_update
                                state["W_sub"][:, i2:] -= second_order_scale * total_outer_update
                            else:
                                applied_second_order_update = second_order_update
                                state["W_sub"][:, i2:] -= total_outer_update
                            if block_gd_mode and second_order_update.numel() > 0:
                                block_second_order_chunks.append(applied_second_order_update)

                        with profile_recorder.section("fasterquant.block.outer_update_ghinv") if profile_recorder else nullcontext():
                            state["GHinv"][:, i2:] -= state["Z"][:, i1:i2].matmul(state["Hinv"][i1:i2, i2:])
                            if block_gd_mode:
                                self._clear_grad_optimizer_state(state["grad_optimizer_state"], i1, i2)

                    if block_gd_mode and enable_gradient_update and i2 < self.columns:
                        if gradient_refresh_fn is None:
                            raise ValueError("`gradient_refresh_fn` must be provided for g_update_mode='block_gd'.")

                        with profile_recorder.section("fasterquant.block.true_gradient_refresh") if profile_recorder else nullcontext():
                            weight_snapshot = self.layer.weight.data.clone().float()
                            for state in subgroup_states:
                                current_sub_weight = state["W_sub"].clone()
                                current_sub_weight[:, :i2] = state["Q"][:, :i2]
                                current_sub_weight_orig = (
                                    current_sub_weight[:, state["invperm"]]
                                    if actorder else current_sub_weight
                                )
                                weight_snapshot[state["row_start"]:state["row_end"], :] = current_sub_weight_orig

                            refreshed_grad, refresh_meta = gradient_refresh_fn(weight_snapshot)
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
                                    }
                                )

                        for state in subgroup_states:
                            with profile_recorder.section("fasterquant.block.outer_update_grad_descent") if profile_recorder else nullcontext():
                                optimizer_update = state.pop("pending_optimizer_update")
                                if optimizer_update.numel() > 0:
                                    state["W_sub"][:, i2:] -= optimizer_update

            for state in subgroup_states:
                with profile_recorder.section("fasterquant.subgroup.total") if profile_recorder else nullcontext():
                    Q = state["Q"]
                    W_int_sub = state["W_int_sub"]
                    Scale_sub = state["Scale_sub"]
                    if actorder:
                        with profile_recorder.section("fasterquant.subgroup.actorder_unpermute") if profile_recorder else nullcontext():
                            Q = Q[:, state["invperm"]]
                            W_int_sub = W_int_sub[:, state["invperm"]]
                            Scale_sub = Scale_sub[:, state["invperm"]]

                    with profile_recorder.section("fasterquant.subgroup.writeback_outputs") if profile_recorder else nullcontext():
                        Q_final[state["row_start"]:state["row_end"], :] = Q
                        W_int_final[state["row_start"]:state["row_end"], :] = W_int_sub
                        Scale_final[state["row_start"]:state["row_end"], :] = Scale_sub

            if export_to_et:
                with profile_recorder.section("fasterquant.export_buffers") if profile_recorder else nullcontext():
                    self.layer.register_buffer(
                        "int_weight", W_int_final.reshape(self.layer.weight.shape)
                    )
                    self.layer.register_buffer("scale", Scale_final)
            with profile_recorder.section("fasterquant.write_layer_weight") if profile_recorder else nullcontext():
                self.layer.weight.data = Q_final.reshape(self.layer.weight.shape).to(
                    self.layer.weight.data.dtype
                )
            if torch.any(torch.isnan(self.layer.weight.data)):
                logging.warning("NaN in weights")

                pprint.pprint(
                    self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point
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
            H_sub = self.H[:, :, sub_idx].clone()
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
            H_sub = self.H[:, :, sub_idx].clone()
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
    """
    def __init__(self, names, num_groups):
        self.num_groups = num_groups
        self.gradients_cache = {}
        self.index = {}
        self.names = names
        for name in self.names:
            self.gradients_cache[name] = 0
            self.index[name] = 0
        self.handles = []
        self.hooks_enabled = False

    def cache_gradient(self, grad, name):
        if not self.hooks_enabled:
            return
        self.gradients_cache[name] *= self.index[name] / (self.index[name] + 1)
        self.index[name] += 1
        self.gradients_cache[name] += grad.float() / self.index[name]

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
            self.index[name] = 0
        memory_utils.cleanup_memory()


def hidden2logits(hidden_states, analyzer: model_utils.ModelAnalyzer):
    norm = analyzer.get_layernorm_before_head()
    lm_head = analyzer.get_lm_head()

    logits = lm_head(norm(hidden_states))

    return logits


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
):
    logging.info(
        "Collecting static end-to-end saliency/fisher caches from a single pre-quantization full-model backward pass. "
        "Using sampled end-to-end NLL / empirical Fisher because literal KL-to-self before quantization would be zero."
    )
    layers = analyzer.get_layers()
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
                saliency_data[layer_idx][module_name].append(
                    grad_squared.mean(dim=-1).detach().cpu()
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
                fisher_data[layer_idx].append(grad_squared.mean(dim=-1).detach().cpu())

            out_tensor.register_hook(grad_hook)

        return forward_hook

    for layer_idx, (layer, module_dict) in enumerate(zip(layers, module_dicts)):
        handles.append(layer.register_forward_hook(make_layer_hook(layer_idx)))
        for module_name, module in module_dict.items():
            handles.append(module.register_forward_hook(make_module_hook(layer_idx, module_name)))

    token_batches = [batch[0] for batch in dataloader]
    model = model.to(dev)
    model.eval()
    try:
        with torch.enable_grad():
            for start in tqdm(
                range(0, len(token_batches), batch_size),
                ncols=120,
                desc="Static E2E Saliency/Fisher",
                position=1,
                leave=False,
            ):
                input_ids = torch.cat(token_batches[start:start + batch_size], dim=0).to(dev)
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
                labels = torch.distributions.Categorical(logits=teacher_logits).sample()
                loss = F.cross_entropy(
                    student_logits.view(-1, student_logits.size(-1)),
                    labels.view(-1),
                    reduction="sum",
                )
                model.zero_grad()
                loss.backward()
                del outputs, logits, teacher_logits, student_logits, labels, loss, input_ids
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad()
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
            layer_saliency[module_name] = torch.cat(saliency_data[layer_idx][module_name], dim=0)
        if not fisher_data[layer_idx]:
            raise ValueError(f"Failed to collect static end-to-end Fisher for layer={layer_idx}.")
        static_saliency.append(layer_saliency)
        static_fisher.append(torch.cat(fisher_data[layer_idx], dim=0))

    return static_saliency, static_fisher


def compute_refresh_loss(
    refresh_loss_type,
    out_hidden,
    fp_hidden,
    analyzer,
    kl_topk,
    layer_output_fisher=None,
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

    if layer_output_fisher is None:
        raise ValueError("`layer_output_fisher` must be provided for refresh_loss_type='fisher_diag_mse'.")
    num_groups = layer_output_fisher.shape[-1]
    if delta.shape[-1] % num_groups != 0:
        raise ValueError(
            f"Output hidden dim ({delta.shape[-1]}) must be divisible by layer-output Fisher groups ({num_groups})."
        )
    group_size = delta.shape[-1] // num_groups
    delta_grouped = delta.view(delta.shape[0], delta.shape[1], num_groups, group_size)
    fisher_norm = layer_output_fisher / (layer_output_fisher.mean() + 1e-12)
    weighted_sq = fisher_norm.unsqueeze(-1) * delta_grouped.square()
    return 0.5 * weighted_sq.sum(dim=(-1, -2)).mean()


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
    if optimizer == "sgd":
        if grad_clip is not None and grad_clip > 0:
            grad_step = grad_step.clamp(min=-grad_clip, max=grad_clip)
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
            with layer_recorder.section("layer.pre_quant_fisher.batch.total") if layer_recorder else nullcontext():
                with layer_recorder.section("layer.pre_quant_fisher.forward.layer") if layer_recorder else nullcontext():
                    out = layer(
                        inps[j : j + bsz].to(dev),
                        attention_mask=batch_attention_mask,
                        position_ids=batch_position_ids,
                        position_embeddings=batch_position_embeddings,
                    )
                with layer_recorder.section("layer.pre_quant_fisher.forward.hidden_extract") if layer_recorder else nullcontext():
                    out_hidden = out[0] if isinstance(out, (tuple, list)) else out
                with layer_recorder.section("layer.pre_quant_fisher.forward.logits_quant") if layer_recorder else nullcontext():
                    logits = hidden2logits(out, analyzer)
                with layer_recorder.section("layer.pre_quant_fisher.forward.logits_fp") if layer_recorder else nullcontext():
                    logits_fp = hidden2logits(fp_inps[j : j + bsz].to(dev), analyzer)

                grad_hessian_logits = logits
                grad_hessian_logits_fp = logits_fp
                if grad_hessian_topk > 0:
                    with layer_recorder.section("layer.pre_quant_fisher.forward.topk_slice") if layer_recorder else nullcontext():
                        grad_hessian_logits_fp, grad_hessian_indices = logits_fp.topk(
                            grad_hessian_topk,
                            dim=-1,
                            sorted=False,
                        )
                        grad_hessian_logits = logits.gather(-1, grad_hessian_indices)

                with layer_recorder.section("layer.pre_quant_fisher.loss_build") if layer_recorder else nullcontext():
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

                with layer_recorder.section("layer.pre_quant_fisher.backward") if layer_recorder else nullcontext():
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

                with layer_recorder.section("layer.pre_quant_fisher.cleanup") if layer_recorder else nullcontext():
                    memory_utils.cleanup_memory()

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
):
    module = full.get(module_name, full.get(module_name + ".module", None))
    if module is None:
        raise ValueError(f"Unable to find module `{module_name}` in the provided layer.")
    losses = []
    selected_indices = list(range(inps.shape[0])) if sample_indices is None else list(sample_indices)
    if len(selected_indices) == 0:
        raise ValueError("`sample_indices` must contain at least one sample.")
    override_weight = (
        module.weight.detach().clone()
        if weight_override is None
        else weight_override.detach().to(module.weight.device, dtype=module.weight.data.dtype).clone()
    )
    override_weight.requires_grad_(True)
    grad = torch.zeros_like(override_weight, dtype=torch.float32)
    grad_count = 0
    grad_modules = [layer]
    if refresh_loss_type == "kl":
        grad_modules.extend(
            [
                analyzer.get_layernorm_before_head(),
                analyzer.get_lm_head(),
            ]
        )
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
                fisher_batch = None if layer_output_fisher is None else layer_output_fisher[batch_indices].to(dev)
                refresh_loss = compute_refresh_loss(
                    refresh_loss_type,
                    out_hidden,
                    fp_hidden,
                    analyzer,
                    kl_topk,
                    layer_output_fisher=fisher_batch,
                )
                batch_grad = torch.autograd.grad(refresh_loss, override_weight, retain_graph=False)[0]
                grad.mul_(grad_count / (grad_count + 1))
                grad.add_(batch_grad.float(), alpha=1.0 / (grad_count + 1))
                grad_count += 1
                losses.append(refresh_loss.item())
                memory_utils.cleanup_memory()

    mean_refresh_loss = sum(losses) / len(losses)
    return grad, mean_refresh_loss


def collect_layer_grad_hessian_stats(
    *,
    model,
    layer,
    analyzer,
    full,
    names,
    inps,
    fp_inps,
    batch_attention_mask,
    batch_position_ids,
    batch_position_embeddings,
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
    layer_recorder=None,
):
    need_saliency_collection = precomputed_saliency_dict is None
    need_layer_output_fisher_collection = (
        layer_refresh_loss_type == "fisher_diag_mse" and precomputed_layer_output_fisher is None
    )
    need_output_head = (
        need_saliency_collection
        or layer_refresh_loss_type == "kl"
        or gptq_reference_loss_type == "kl"
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
            with layer_recorder.section("layer.grad_hessian.batch.total") if layer_recorder else nullcontext():
                with layer_recorder.section("layer.grad_hessian.forward") if layer_recorder else nullcontext():
                    with layer_recorder.section("layer.grad_hessian.forward.layer") if layer_recorder else nullcontext():
                        out = layer(
                            inps[j : j + bsz].to(dev),
                            attention_mask=batch_attention_mask,
                            position_ids=batch_position_ids,
                            position_embeddings=batch_position_embeddings,
                        )
                    with layer_recorder.section("layer.grad_hessian.forward.hidden_extract") if layer_recorder else nullcontext():
                        out_hidden = out[0] if isinstance(out, (tuple, list)) else out
                    with layer_recorder.section("layer.grad_hessian.forward.fp_hidden") if layer_recorder else nullcontext():
                        fp_hidden = fp_inps[j : j + bsz].to(dev)
                    logits = None
                    logits_fp = None
                    grad_hessian_logits = None
                    grad_hessian_logits_fp = None
                    if need_output_head:
                        with layer_recorder.section("layer.grad_hessian.forward.logits_quant") if layer_recorder else nullcontext():
                            logits = hidden2logits(out, analyzer)
                        with layer_recorder.section("layer.grad_hessian.forward.logits_fp") if layer_recorder else nullcontext():
                            logits_fp = hidden2logits(fp_hidden, analyzer)
                        grad_hessian_logits = logits
                        grad_hessian_logits_fp = logits_fp
                        if grad_hessian_topk > 0:
                            with layer_recorder.section("layer.grad_hessian.forward.topk_slice") if layer_recorder else nullcontext():
                                grad_hessian_logits_fp, grad_hessian_indices = logits_fp.topk(
                                    grad_hessian_topk,
                                    dim=-1,
                                    sorted=False,
                                )
                                grad_hessian_logits = logits.gather(-1, grad_hessian_indices)
                    if need_saliency_collection:
                        with layer_recorder.section("layer.grad_hessian.forward.label_sample") if layer_recorder else nullcontext():
                            labels = torch.distributions.Categorical(logits=grad_hessian_logits_fp).sample()
                        with layer_recorder.section("layer.grad_hessian.forward.nll_build") if layer_recorder else nullcontext():
                            nll_loss = F.cross_entropy(
                                grad_hessian_logits.view(-1, grad_hessian_logits.size(-1)),
                                labels.view(-1),
                                reduction="sum",
                            )

                if need_saliency_collection:
                    with layer_recorder.section("layer.grad_hessian.saliency_backward.total") if layer_recorder else nullcontext():
                        saliency_cache.enable_hooks()
                        with layer_recorder.section("layer.grad_hessian.saliency_backward.zero_grad") if layer_recorder else nullcontext():
                            model.zero_grad()
                        with layer_recorder.section("layer.grad_hessian.saliency_backward.backward") if layer_recorder else nullcontext():
                            nll_loss.backward(retain_graph=True)
                        saliency_cache.disable_hooks()

                batch_layer_output_fisher = None
                if layer_refresh_loss_type == "fisher_diag_mse" and precomputed_layer_output_fisher is not None:
                    with layer_recorder.section("layer.grad_hessian.fisher_slice") if layer_recorder else nullcontext():
                        batch_layer_output_fisher = precomputed_layer_output_fisher[j : j + bsz].to(dev)
                elif need_layer_output_fisher_collection:
                    with layer_recorder.section("layer.grad_hessian.fisher_collect") if layer_recorder else nullcontext():
                        kl_logits = grad_hessian_logits if grad_hessian_topk > 0 else logits
                        kl_logits_fp = grad_hessian_logits_fp if grad_hessian_topk > 0 else logits_fp
                        if grad_hessian_topk <= 0 and kl_topk > 0:
                            with layer_recorder.section("layer.grad_hessian.fisher_collect.topk") if layer_recorder else nullcontext():
                                kl_logits_fp, indices = logits_fp.topk(kl_topk, dim=-1, sorted=False)
                                kl_logits = logits.gather(-1, indices)
                        with layer_recorder.section("layer.grad_hessian.fisher_collect.loss_build") if layer_recorder else nullcontext():
                            fisher_kl_loss = F.kl_div(
                                F.log_softmax(kl_logits, dim=-1),
                                F.softmax(kl_logits_fp, dim=-1),
                                reduction="none",
                            )
                            fisher_kl_loss = fisher_kl_loss.sum(dim=-1).mean()
                        with layer_recorder.section("layer.grad_hessian.fisher_collect.hook_register") if layer_recorder else nullcontext():
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
                        with layer_recorder.section("layer.grad_hessian.fisher_collect.backward") if layer_recorder else nullcontext():
                            model.zero_grad()
                            fisher_kl_loss.backward(retain_graph=True)
                        batch_layer_output_fisher = layer_output_fisher_cache[-1]

                with layer_recorder.section("layer.grad_hessian.gradient_loss_build") if layer_recorder else nullcontext():
                    if gptq_reference_loss_type == "kl":
                        kl_logits = grad_hessian_logits if grad_hessian_topk > 0 else logits
                        kl_logits_fp = grad_hessian_logits_fp if grad_hessian_topk > 0 else logits_fp
                        if grad_hessian_topk <= 0 and kl_topk > 0:
                            with layer_recorder.section("layer.grad_hessian.gradient_loss_build.topk") if layer_recorder else nullcontext():
                                kl_logits_fp, indices = logits_fp.topk(kl_topk, dim=-1, sorted=False)
                                kl_logits = logits.gather(-1, indices)
                        gradient_loss = F.kl_div(
                            F.log_softmax(kl_logits, dim=-1),
                            F.softmax(kl_logits_fp, dim=-1),
                            reduction="none",
                        )
                        gradient_loss = gradient_loss.sum(dim=-1).mean()
                    else:
                        gradient_loss = compute_refresh_loss(
                            gptq_reference_loss_type,
                            out_hidden,
                            fp_hidden,
                            analyzer,
                            kl_topk,
                            layer_output_fisher=batch_layer_output_fisher,
                        )

                with layer_recorder.section("layer.grad_hessian.gradient_backward.total") if layer_recorder else nullcontext():
                    gradients_cache.enable_hooks()
                    with layer_recorder.section("layer.grad_hessian.gradient_backward.zero_grad") if layer_recorder else nullcontext():
                        model.zero_grad()
                    with layer_recorder.section("layer.grad_hessian.gradient_backward.backward") if layer_recorder else nullcontext():
                        gradient_loss.backward()
                    gradients_cache.disable_hooks()

                with layer_recorder.section("layer.grad_hessian.metrics_record") if layer_recorder else nullcontext():
                    reference_losses.append(gradient_loss.item())
                with layer_recorder.section("layer.grad_hessian.cleanup") if layer_recorder else nullcontext():
                    memory_utils.cleanup_memory()

        mean_reference_loss = sum(reference_losses) / len(reference_losses)

    if saliency_cache is not None:
        saliency_cache.clear_hook()
    gradients_cache.clear_hook()
    layer_output_fisher = precomputed_layer_output_fisher
    if need_layer_output_fisher_collection:
        layer_output_fisher = torch.cat(layer_output_fisher_cache, dim=0)

    with layer_recorder.section("layer.cache_finalize") if layer_recorder else nullcontext():
        if saliency_cache is not None:
            for name in saliency_cache.names:
                saliency_cache.saliency_cache[name] = torch.cat(saliency_cache.saliency_cache[name], dim=0)
        for name in gradients_cache.names:
            gradients_cache.gradients_cache[name] = gradients_cache.gradients_cache[name]
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

    for step_idx in range(num_steps):
        sample_indices = scheduler.next_indices()
        step_losses = []
        step_update_abs = []
        for module_name, module in modules:
            fisher_tensor = layer_output_fisher_by_module.get(module_name)
            grad, mean_loss = collect_true_weight_gradient(
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
            )
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

    with run_recorder.section("run.total") if run_recorder else nullcontext():
        if global_loss_enabled:
            with pipeline_recorder.section("pipeline.static_end_to_end_saliency_fisher") if pipeline_recorder else nullcontext():
                static_saliency_by_layer, static_fisher_by_layer = collect_static_end_to_end_saliency_and_fisher(
                    model=model,
                    analyzer=analyzer,
                    dataloader=dataloader,
                    dev=dev,
                    saliency_num_groups=args.num_groups,
                    fisher_num_groups=args.fisher_num_groups,
                    grad_hessian_topk=args.grad_hessian_topk,
                    batch_size=args.global_loss_bsz,
                )
            logging.info(
                "Collected frozen end-to-end saliency/Fisher caches before quantization with global_loss_bsz=%d. "
                "These cached coefficients will be reused for Hessian estimation and fisher_diag_mse throughout quantization.",
                args.global_loss_bsz,
            )
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
        with pipeline_recorder.section("pipeline.move_to_device") if pipeline_recorder else nullcontext():
            for module in per_layer_runtime_modules:
                module.to(dev)
            layers[0] = layers[0].to(dev)

        dtype = next(iter(model.parameters())).dtype
        inps = torch.zeros(
            (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
        )
        cache = {"i": 0, "attention_mask": None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                if hasattr(module, "attention_type"):
                    self.attention_type = module.attention_type

            def forward(self, inp, **kwargs):
                inps[cache["i"]] = inp
                cache["i"] += 1
                cache["attention_mask"] = kwargs["attention_mask"]
                cache["position_ids"] = kwargs["position_ids"]
                cache["position_embeddings"] = kwargs["position_embeddings"]
                raise ValueError

        layers[0] = Catcher(layers[0])
        with pipeline_recorder.section("pipeline.capture_inputs") if pipeline_recorder else nullcontext():
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
            with pipeline_recorder.section("pipeline.offload_inputs") if pipeline_recorder else nullcontext():
                inps = inps.cpu()
                fp_inps = fp_inps.cpu()

        quantizers = {}
        gradient_refresh_scheduler = None
        if args.g_update_mode in {"block_backward", "block_gd"} or effective_pre_gd_steps > 0:
            gradient_refresh_scheduler = BackwardSampleScheduler(
                args.nsamples,
                args.backward_samples,
                seed=args.seed,
            )
        final_layer_idx = len(layers) - 1
        full_refresh_sample_indices = list(range(args.nsamples)) if args.final_layer_full_backward else None
        layer_indices = range(quant_stop_layer + 1) if quant_stop_layer is not None else range(len(layers))
        pbar = tqdm(layer_indices, ncols=120, desc="Quantizing Layers", position=0)
        for i in pbar:
            layer = layers[i].to(dev)
            full = analyzer.get_quantizable_modules(layer)
            layer_recorder = QuantProfileRecorder(dev, prefix=f"layers.{i}") if quant_profile_enabled else None
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
            layer_backward_bsz = args.final_layer_backward_bsz if i == final_layer_idx else args.backward_bsz

            with layer_recorder.section("layer.fp_reference_forward") if layer_recorder else nullcontext():
                bits_config = quant_utils.disable_act_quant(layer)
                for j in range(args.nsamples):
                    fp_inps[j] = layer(
                        fp_inps[j].unsqueeze(0).to(dev),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )[0].to(fp_inps.device)
                quant_utils.enable_act_quant(layer, bits_config)

            if preclip_enabled:
                with layer_recorder.section("layer.weight_preclip") if layer_recorder else nullcontext():
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
                    with layer_recorder.section("layer.pre_quant_fisher_collect") if layer_recorder else nullcontext():
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
                    else args.pre_grad_lr
                )
                if pre_grad_lr > 0:
                    with layer_recorder.section("layer.pre_quant_gd") if layer_recorder else nullcontext():
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
                        )

            saliency_dict, gradients_dict, mean_reference_loss, layer_output_fisher = collect_layer_grad_hessian_stats(
                model=model,
                layer=layer,
                analyzer=analyzer,
                full=full,
                names=names,
                inps=inps,
                fp_inps=fp_inps,
                batch_attention_mask=batch_attention_mask,
                batch_position_ids=batch_position_ids,
                batch_position_embeddings=batch_position_embeddings,
                bsz=args.bsz,
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
                layer_recorder=layer_recorder,
            )

            gptq = {}
            with layer_recorder.section("layer.gptq_setup") if layer_recorder else nullcontext():
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
            with layer_recorder.section("layer.hessian_accumulation_forward") if layer_recorder else nullcontext():
                for j in range(args.nsamples):
                    _ = layer(
                        inps[j].unsqueeze(0).to(dev),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )[0]
            for h in handles:
                h.remove()
            for name in add_batch_recorders:
                gptq[name].profile_recorder = None

            def make_gradient_refresh_fn(module_name):
                def refresh_fn(weight_snapshot):
                    if args.final_layer_full_backward and i == final_layer_idx:
                        sample_indices = full_refresh_sample_indices
                    else:
                        sample_indices = gradient_refresh_scheduler.next_indices()
                    grad, mean_refresh_loss = collect_true_weight_gradient(
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
                    )
                    return grad, {"mean_refresh_loss": mean_refresh_loss, "sample_indices": sample_indices}

                return refresh_fn

            def make_block_observer(module_name, effective_grad_optimizer):
                def observer(payload):
                    logging.info(
                        "block-metrics layer=%d module=%s mode=%s grad_opt=%s block=%d cols=[%d,%d) remain=%d grad_abs_mean=%s grad_row_l2=%s loss=%s refresh_loss=%s train_loss=%s val_loss=%s second_abs=%s first_raw_abs=%s first_abs=%s reg_abs=%s sine_abs=%s",
                        i,
                        module_name,
                        args.g_update_mode,
                        effective_grad_optimizer,
                        payload["block_idx"],
                        payload["col_start"],
                        payload["col_end"],
                        payload["remaining_columns"],
                        format_log_value(payload["remaining_grad_abs_mean"], digits=4),
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
                    )

                return observer

            with layer_recorder.section("layer.module_quantization") if layer_recorder else nullcontext():
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
                        else args.grad_lr
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
                    module_recorder = (
                        QuantProfileRecorder(dev, prefix=f"layers.{i}.{name}")
                        if should_profile_module(i, name) else None
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
                        gradient_refresh_fn=make_gradient_refresh_fn(name) if args.g_update_mode in {"block_backward", "block_gd"} else None,
                        grad_lr=effective_grad_lr,
                        grad_optimizer=effective_grad_optimizer,
                        grad_reg_strategy=args.grad_reg_strategy,
                        grad_reg_lambda=args.grad_reg_lambda,
                        grad_gate_floor=args.grad_gate_floor,
                        grad_gate_sharpness=args.grad_gate_sharpness,
                        grad_gate_sine_amp=args.grad_gate_sine_amp,
                        second_order_scale=args.second_order_scale,
                        block_atomic_quant=args.block_atomic_quant,
                        block_observer=make_block_observer(name, effective_grad_optimizer) if args.g_update_mode in {"block_backward", "block_gd"} else None,
                        grad_clip=args.grad_clip,
                    )
                    quantizers["model.layers.%d.%s" % (i, name)] = gptq[name].quantizer
                    gptq[name].free()

            with layer_recorder.section("layer.quantized_replay_forward") if layer_recorder else nullcontext():
                for j in range(args.nsamples):
                    inps[j] = layer(
                        inps[j].unsqueeze(0).to(dev),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )[0].squeeze(0).to(inps.device)

            with layer_recorder.section("layer.cleanup") if layer_recorder else nullcontext():
                layers[i] = layer.to(orig_device)
                del layer
                del gptq
                del saliency_dict, gradients_dict
                memory_utils.cleanup_memory()

            if quant_stop_layer is not None and i >= quant_stop_layer:
                logging.info("Stopping quantization after transformer layer %d due to --quant_stop_layer.", i)
                break

        with pipeline_recorder.section("pipeline.restore_modules") if pipeline_recorder else nullcontext():
            for module in per_layer_runtime_modules:
                module.to(orig_device)
            model.config.use_cache = use_cache
        memory_utils.cleanup_memory(verbos=True)

    if quant_profile_enabled:
        logging.info("Quant profile NVTX ranges emitted. Inspect them with Nsight Systems/Compute.")
    logging.info("-----GPTQPlus Quantization Done-----\n")
    return quantizers
