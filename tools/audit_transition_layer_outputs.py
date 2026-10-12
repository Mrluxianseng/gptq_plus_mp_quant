"""Offline layerwise audit of saved REAL-Q Adam transitions.

Compares the actual BF16-forward path with a smooth FP32 replay at the same
saved transition, recording where the update response first differs.
This is diagnostic only; it never mutates the quantization path.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from realq.attention import configure_attention_backend
from utils import model_utils, quant_utils
from utils.loss_utils import tokenwise_kl_from_logits
from utils.reproducibility import configure_reproducibility


def hidden(value):
    return value[0] if isinstance(value, tuple) else value


def select_sample(value, index: int, batch_size: int):
    if isinstance(value, torch.Tensor):
        if value.ndim and value.shape[0] == batch_size:
            return value[index:index + 1]
        return value
    if isinstance(value, tuple):
        return tuple(select_sample(x, index, batch_size) for x in value)
    if isinstance(value, list):
        return [select_sample(x, index, batch_size) for x in value]
    if isinstance(value, dict):
        return {k: select_sample(v, index, batch_size) for k, v in value.items()}
    return value


def to_device(value, device, float_tensors=False):
    if isinstance(value, torch.Tensor):
        value = value.to(device)
        return value.float() if float_tensors and value.is_floating_point() else value
    if isinstance(value, tuple):
        return tuple(to_device(x, device, float_tensors) for x in value)
    if isinstance(value, list):
        return [to_device(x, device, float_tensors) for x in value]
    if isinstance(value, dict):
        return {k: to_device(v, device, float_tensors) for k, v in value.items()}
    return value


def delta_stats(before: torch.Tensor, after: torch.Tensor):
    a, b = before.float(), after.float()
    delta = b - a
    return {
        "before_rms": a.square().mean().sqrt().item(),
        "after_rms": b.square().mean().sqrt().item(),
        "delta_rms": delta.square().mean().sqrt().item(),
        "delta_l2": delta.norm().item(),
        "delta_linf": delta.abs().max().item(),
        "changed_fraction": (a != b).float().mean().item(),
    }


@torch.no_grad()
def replay_path(analyzer, context, refresh, target, before, after, fp32: bool):
    start_layer = context["layer"]
    layers = analyzer.get_layers()
    if fp32:
        for block in layers[start_layer:]:
            block.float()
        analyzer.get_layernorm_before_head().float()
        analyzer.get_lm_head().float()
        p = torch.nn.Parameter(before.to(device="cuda", dtype=torch.float32), requires_grad=False)
        target.weight = p
        if hasattr(target, "module") and hasattr(target.module, "weight"):
            target.module.weight = p

    module_outputs = {}
    def save_module_output(_module, _inputs, output):
        module_outputs["value"] = hidden(output).detach().float().cpu()

    hook = target.register_forward_hook(save_module_output)
    block_outputs = {}
    kl_by_state = {}
    batch_size = refresh["inps"].shape[0]
    selected = refresh["selected"]

    # Teacher logits use the saved FP target-layer outputs and the same suffix
    # as the transition objective, matching the existing finite-difference run.
    teacher_logits = {}
    for i in selected:
        kw = to_device(select_sample(refresh["kw"], i, batch_size), "cuda", fp32)
        ref = refresh["fp_target"][i:i + 1].cuda()
        if fp32:
            ref = ref.float()
        for block in layers[start_layer + 1:]:
            ref = hidden(block(ref, **kw))
        normed = analyzer.get_layernorm_before_head()(ref)
        teacher_logits[i] = analyzer.get_lm_head()(normed).float().detach()

    for state_name, weight in (("before", before), ("after", after)):
        target.weight.copy_(weight.to(device="cuda", dtype=torch.float32 if fp32 else target.weight.dtype))
        if hasattr(target, "module") and hasattr(target.module, "weight"):
            target.module.weight = target.weight
        rows = []
        losses = []
        for i in selected:
            x = refresh["inps"][i:i + 1].cuda()
            if fp32:
                x = x.float()
            kw = to_device(select_sample(refresh["kw"], i, batch_size), "cuda", fp32)
            module_outputs.clear()
            y = hidden(layers[start_layer](x, **kw))
            module_out = module_outputs.get("value")
            if module_out is None:
                raise RuntimeError("target module forward hook did not fire")
            out_by_layer = {"target_module": module_out}
            out_by_layer[f"layer_{start_layer}"] = y.detach().float().cpu()
            for index in range(start_layer + 1, len(layers)):
                y = hidden(layers[index](y, **kw))
                out_by_layer[f"layer_{index}"] = y.detach().float().cpu()
            logits = analyzer.get_lm_head()(analyzer.get_layernorm_before_head()(y)).float()
            losses.append(tokenwise_kl_from_logits(logits, teacher_logits[i]).double().mean().item())
            rows.append(out_by_layer)
        keys = rows[0].keys()
        block_outputs[state_name] = {
            name: torch.cat([row[name] for row in rows], dim=0) for name in keys
        }
        kl_by_state[state_name] = sum(losses) / len(losses)
    hook.remove()
    return block_outputs, kl_by_state


def run(model_path: str, campaign: Path, target_keys: list[str]):
    summary = json.loads((campaign / "summary.json").read_text())
    if summary.get("failure") or not summary.get("sources_unchanged"):
        raise RuntimeError("campaign is invalid or source hashes changed")
    fd_path = campaign / "finite_difference_kl.json"
    fd = json.loads(fd_path.read_text())
    result_path = campaign / "layer_output_audit.json"
    results = json.loads(result_path.read_text()) if result_path.exists() else {}

    for key in target_keys:
        print(f"[layer-audit] START {key}", flush=True)
        arm = summary["results"][key]["adam"]
        directory = Path(arm["directory"]) / "transition"
        context = torch.load(directory / "context.pt", map_location="cpu", weights_only=False)
        refresh = torch.load(directory / "refresh.pt", map_location="cpu", weights_only=False)
        before = torch.load(directory / "before_update.pt", map_location="cpu", weights_only=True)
        after = torch.load(directory / "after_update.pt", map_location="cpu", weights_only=True)

        analyzer = model_utils.ModelAnalyzer(model_path, 128, skip_state_dict=True)
        configure_attention_backend(analyzer.model, context["config"]["attention_backend"])
        quant_utils.add_actquant(analyzer)
        analyzer.model.eval().requires_grad_(False).cuda()
        for i, state in enumerate(context["prefix_layers"]):
            analyzer.get_layers()[i].load_state_dict(state, strict=True)
        analyzer.get_layers()[context["layer"] + 1].load_state_dict(context["next_layer"], strict=True)
        target = analyzer.get_layers()[context["layer"]].get_submodule(context["module"])
        native_dtype = str(target.weight.dtype)

        mode_results = {}
        mode_outputs = {}
        for mode, fp32 in (("bf16", False), ("fp32", True)):
            outputs, kl = replay_path(analyzer, context, refresh, target, before, after, fp32)
            mode_outputs[mode] = outputs
            deltas = {}
            for name in outputs["before"]:
                deltas[name] = delta_stats(outputs["before"][name], outputs["after"][name])
            mode_results[mode] = {"kl_before": kl["before"], "kl_after": kl["after"],
                                  "kl_delta": kl["after"] - kl["before"], "output_deltas": deltas}
        expected = fd[key]
        for mode, fd_key in (("bf16", "candidate_actual_step_kl_change_bf16path"),
                             ("fp32", "candidate_actual_step_kl_change_fp32path")):
            observed = mode_results[mode]["kl_delta"]
            if not math.isclose(observed, expected[fd_key], rel_tol=3e-3, abs_tol=3e-7):
                raise RuntimeError(f"{key} {mode} KL replay mismatch: {observed} vs {expected[fd_key]}")

        common = set(mode_results["bf16"]["output_deltas"]) & set(mode_results["fp32"]["output_deltas"])
        for name in common:
            # Compare the update-response vectors, not the absolute BF16 and
            # FP32 activations, whose base values differ by precision.
            bf16_delta = (mode_outputs["bf16"]["after"][name] - mode_outputs["bf16"]["before"][name]).flatten()
            fp32_delta = (mode_outputs["fp32"]["after"][name] - mode_outputs["fp32"]["before"][name]).flatten()
            cosine = torch.nn.functional.cosine_similarity(bf16_delta, fp32_delta, dim=0).item()
            mode_results.setdefault("response_comparison", {})[name] = {
                "bf16_vs_fp32_delta_rms_ratio": mode_results["bf16"]["output_deltas"][name]["delta_rms"] /
                    max(mode_results["fp32"]["output_deltas"][name]["delta_rms"], 1e-30),
                "bf16_vs_fp32_delta_l2_cosine": cosine,
            }

        # The per-path statistics above are enough to locate the first large
        # response. Save a concise ordering to make inspection easy.
        results[key] = {
            "precision": mode_results,
            "native_model_parameter_dtype": native_dtype,
            "snapshot_shapes": {"before": list(before.shape), "after": list(after.shape)},
            "selected_samples": list(refresh["selected"]),
            "scope": "saved refresh inputs; target block and all downstream blocks; no parameter update or production-path change",
        }
        result_path.write_text(json.dumps(results, indent=2))
        print(f"[layer-audit] DONE {key}: BF16 ΔKL={mode_results['bf16']['kl_delta']:.8g}; "
              f"FP32 ΔKL={mode_results['fp32']['kl_delta']:.8g}", flush=True)
        del analyzer, target
        torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target", action="append", required=True)
    args = parser.parse_args()
    configure_reproducibility(0, deterministic=True)
    torch.set_num_threads(4)
    run(args.model, args.campaign, args.target)
