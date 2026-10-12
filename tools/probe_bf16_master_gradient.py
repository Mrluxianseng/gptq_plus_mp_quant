"""Compare BF16 gradient storage with FP32 gradient accumulation under the
exact same BF16 Linear forward at saved REAL-Q refresh states.

The candidate is diagnostic only: the custom Linear has a BF16 forward and
returns an FP32 weight gradient accumulated from BF16 inputs/grad-output.
No quantizer or production optimizer path is modified.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import functional_call

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from realq.attention import configure_attention_backend
from realq.refresh.fisher_loss import fisher_mse_loss
from utils import model_utils, quant_utils
from utils.reproducibility import configure_reproducibility


class _BF16ForwardFP32WeightGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight_fp32, bias):
        if x.dtype != torch.bfloat16 or weight_fp32.dtype != torch.float32:
            raise TypeError(f"expected bf16 input/fp32 weight, got {x.dtype}/{weight_fp32.dtype}")
        weight_bf16 = weight_fp32.to(torch.bfloat16)
        bias_bf16 = None if bias is None else bias.to(torch.bfloat16)
        ctx.save_for_backward(x, weight_bf16)
        return F.linear(x, weight_bf16, bias_bf16)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight_bf16 = ctx.saved_tensors
        x32 = x.reshape(-1, x.shape[-1]).float()
        grad32 = grad_output.reshape(-1, grad_output.shape[-1]).float()
        grad_weight = grad32.transpose(0, 1) @ x32
        grad_input = (grad32 @ weight_bf16.float()).reshape_as(x).to(x.dtype)
        # Bias is frozen in this probe. Its gradient is deliberately omitted.
        return grad_input, grad_weight, None


def select_sample(value, index: int, batch_size: int):
    if isinstance(value, torch.Tensor):
        return value[index:index + 1] if value.ndim and value.shape[0] == batch_size else value
    if isinstance(value, tuple):
        return tuple(select_sample(x, index, batch_size) for x in value)
    if isinstance(value, list):
        return [select_sample(x, index, batch_size) for x in value]
    if isinstance(value, dict):
        return {k: select_sample(v, index, batch_size) for k, v in value.items()}
    return value


def move_to_cuda(value):
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, tuple):
        return tuple(move_to_cuda(x) for x in value)
    if isinstance(value, list):
        return [move_to_cuda(x) for x in value]
    if isinstance(value, dict):
        return {k: move_to_cuda(v) for k, v in value.items()}
    return value


def loss_for_output(q_out, i, kw, refresh, next_layer):
    fp = refresh["fp_target"][i:i + 1].cuda()
    loss = fisher_mse_loss(
        q_out, fp, refresh["fisher"].cuda(),
        a_loss_ratio=float(refresh["a_loss_ratio"]),
        a_loss_threshold=refresh["a_loss_threshold"],
    )
    alpha = refresh["slide_alpha"]
    if alpha is not None and alpha < 1.0:
        next_out = next_layer(q_out, **kw)
        next_out = next_out[0] if isinstance(next_out, tuple) else next_out
        fp_next = refresh["next_fp_target"][i:i + 1].cuda()
        loss_next = fisher_mse_loss(
            next_out, fp_next, refresh["next_fisher"].cuda(),
            a_loss_ratio=float(refresh["a_loss_ratio"]),
            a_loss_threshold=refresh["next_a_loss_threshold"],
        )
        loss = float(alpha) * loss + (1.0 - float(alpha)) * loss_next
    return loss


def run_one(model_path: str, campaign: Path, key: str):
    summary = json.loads((campaign / "summary.json").read_text())
    arm = summary["results"][key]["adam"]
    directory = Path(arm["directory"]) / "transition"
    context = torch.load(directory / "context.pt", map_location="cpu", weights_only=False)
    refresh = torch.load(directory / "refresh.pt", map_location="cpu", weights_only=False)
    before = torch.load(directory / "before_update.pt", map_location="cpu", weights_only=True)

    analyzer = model_utils.ModelAnalyzer(model_path, 64, skip_state_dict=True)
    configure_attention_backend(analyzer.model, context["config"]["attention_backend"])
    quant_utils.add_actquant(analyzer)
    analyzer.model.eval().requires_grad_(False).cuda()
    for i, state in enumerate(context["prefix_layers"]):
        analyzer.get_layers()[i].load_state_dict(state, strict=True)
    analyzer.get_layers()[context["layer"] + 1].load_state_dict(context["next_layer"], strict=True)
    layer = analyzer.get_layers()[context["layer"]]
    target = layer.get_submodule(context["module"])
    next_layer = analyzer.get_layers()[context["layer"] + 1]
    weight_name = refresh["weight_name"]
    if target.weight.dtype != torch.bfloat16:
        raise RuntimeError(f"expected native BF16 target weight, got {target.weight.dtype}")

    original_forward = target.module.forward
    def fp32_grad_bf16_forward(module, x, *args, **kwargs):
        if args or kwargs:
            raise RuntimeError("unexpected extra arguments to target Linear")
        return _BF16ForwardFP32WeightGrad.apply(x, module.weight, module.bias)

    sums = {"bf16": torch.zeros_like(before, dtype=torch.float64),
            "fp32_master": torch.zeros_like(before, dtype=torch.float64)}
    losses = {"bf16": 0.0, "fp32_master": 0.0}
    exact_forward = True
    batch_size = refresh["inps"].shape[0]

    for mode in ("bf16", "fp32_master"):
        target.module.forward = types.MethodType(
            fp32_grad_bf16_forward, target.module
        ) if mode == "fp32_master" else original_forward
        leaf = (before.to(device="cuda", dtype=torch.bfloat16).detach().requires_grad_(True)
                if mode == "bf16" else before.to(device="cuda", dtype=torch.float32).detach().requires_grad_(True))
        for i in refresh["selected"]:
            x = refresh["inps"][i:i + 1].cuda()
            kw = move_to_cuda(select_sample(refresh["kw"], i, batch_size))
            mapping = {weight_name: leaf}
            with torch.enable_grad():
                output = functional_call(layer, mapping, (x,), kw, strict=False)
                q_out = output[0] if isinstance(output, tuple) else output
                loss = loss_for_output(q_out, i, kw, refresh, next_layer)
                (grad,) = torch.autograd.grad(loss, leaf)
            if mode != "bf16":
                q_custom = q_out.detach()
                with torch.no_grad():
                    target.module.forward = original_forward
                    ref = functional_call(layer, {weight_name: leaf.detach().to(torch.bfloat16)}, (x,), kw, strict=False)
                    ref = ref[0] if isinstance(ref, tuple) else ref
                    exact_forward = exact_forward and torch.equal(q_custom, ref)
                    target.module.forward = types.MethodType(fp32_grad_bf16_forward, target.module)
            sums[mode].add_(grad.detach().cpu().double())
            losses[mode] += float(loss.detach().item())
            del output, q_out, loss, grad
        del leaf

    target.module.forward = original_forward
    n = len(refresh["selected"])
    grad_bf16_full = (sums["bf16"] / n).float()
    grad_fp32_full = (sums["fp32_master"] / n).float()
    perm = context["perm"]
    column_end = int(context["column_end"])
    if perm is not None:
        perm = perm.to(dtype=torch.long)
        grad_bf16_full = grad_bf16_full[:, perm]
        grad_fp32_full = grad_fp32_full[:, perm]
    grad_bf16 = grad_bf16_full[:, column_end:]
    grad_fp32 = grad_fp32_full[:, column_end:]
    stored = refresh["raw_gradient"].float()
    flat_a, flat_b = grad_bf16.flatten(), grad_fp32.flatten()
    result = {
        "target": key,
        "weight_name": weight_name,
        "selected_samples": refresh["selected"],
        "loss_mean_bf16": losses["bf16"] / n,
        "loss_mean_fp32_master": losses["fp32_master"] / n,
        "stored_online_loss": refresh["actual_loss"],
        "bf16_forward_exactly_matches_custom_forward": exact_forward,
        "stored_gradient_replay_relative_l2": float((grad_bf16 - stored).norm() / stored.norm().clamp_min(1e-30)),
        "bf16_vs_fp32_master_gradient_cosine": float(F.cosine_similarity(flat_a, flat_b, dim=0).clamp(-1, 1)),
        "bf16_vs_fp32_master_gradient_relative_l2": float((grad_bf16 - grad_fp32).norm() / grad_bf16.norm().clamp_min(1e-30)),
        "bf16_gradient_norm": float(grad_bf16.norm()),
        "fp32_master_gradient_norm": float(grad_fp32.norm()),
        "bf16_zero_fraction": float((grad_bf16 == 0).float().mean()),
        "fp32_zero_fraction": float((grad_fp32 == 0).float().mean()),
        "scope": "same BF16 forward; only target Linear weight-gradient accumulation dtype differs; no Adam state/update modified",
    }
    if not exact_forward:
        raise RuntimeError(f"custom BF16 forward is not bit-exact for {key}")
    if not math.isclose(result["loss_mean_bf16"], refresh["actual_loss"], rel_tol=1e-5, abs_tol=1e-8):
        raise RuntimeError(f"BF16 loss replay mismatch: {result['loss_mean_bf16']} vs {refresh['actual_loss']}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target", action="append", required=True)
    args = parser.parse_args()
    configure_reproducibility(0, deterministic=True)
    torch.set_num_threads(4)
    out_path = args.campaign / "bf16_master_gradient_probe.json"
    all_results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for key in args.target:
        print(f"[grad-precision] START {key}", flush=True)
        all_results[key] = run_one(args.model, args.campaign, key)
        out_path.write_text(json.dumps(all_results, indent=2))
        print(f"[grad-precision] DONE {key}: " + json.dumps(all_results[key]), flush=True)
