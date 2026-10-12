"""Summarize matched-state Cautious transition probes; no GPU work."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def analyze(campaign: Path):
    summary = json.loads((campaign / "summary.json").read_text())
    if summary["failure"] is not None or not summary["sources_unchanged"]:
        raise RuntimeError("Cannot analyze an incomplete or source-mutated campaign")
    rows, mechanisms, factors = [], {}, []
    for target, arms in summary["results"].items():
        base = arms["adam"]["evaluation"]
        directory = Path(arms["adam"]["directory"]) / "transition"
        refresh = torch.load(directory / "refresh.pt", map_location="cpu", weights_only=False)
        d, g = refresh["adam_update"].double(), refresh["clipped_gradient"].double()
        product = d*g
        conflict = product < 0
        mask = product > 0
        mechanisms[target] = {
            "adam_step": refresh["adam_step"],
            "conflict_coordinate_fraction": conflict.double().mean().item(),
            "conflict_abs_first_order_fraction": (product[conflict].abs().sum()/product.abs().sum().clamp_min(1e-30)).item(),
            "conflict_update_l1_fraction": (d[conflict].abs().sum()/d.abs().sum().clamp_min(1e-30)).item(),
            "conflict_update_squared_l2_fraction": (d[conflict].square().sum()/d.square().sum().clamp_min(1e-30)).item(),
            "retained_coordinate_fraction": mask.double().mean().item(),
            "alpha": base["update"]["alpha"],
            "gradient_clipping_relative_l2": base["update"]["raw_vs_clipped_gradient_relative_l2"],
        }
        for policy, arm in arms.items():
            evaluation = arm["evaluation"]
            for stage in ("after_update", "after_next_quant"):
                current, reference = evaluation["stages"][stage], base["stages"][stage]
                differences = [a-b for a,b in zip(current["per_window_kl"], reference["per_window_kl"])]
                rows.append({
                    "target": target, "policy": policy, "stage": stage,
                    "kl": current["kl"], "kl_delta_vs_adam": current["kl"]-reference["kl"],
                    "kl_percent_vs_adam": 100*(current["kl"]/reference["kl"]-1),
                    "kl_delta_vs_before": current["kl"]-base["stages"]["before_update"]["kl"],
                    "ppl": current["ppl"], "ppl_delta_vs_adam": current["ppl"]-reference["ppl"],
                    "surrogate": current["surrogate"]["mean"],
                    "surrogate_percent_vs_adam": 100*(current["surrogate"]["mean"]/reference["surrogate"]["mean"]-1),
                    "surrogate_delta_vs_before": current["surrogate"]["mean"]-base["stages"]["before_update"]["surrogate"]["mean"],
                    "windows_better_vs_adam": sum(x<0 for x in differences),
                    "windows_worse_vs_adam": sum(x>0 for x in differences),
                    "next_quantized_block_changed_fraction": arm.get("audit", {}).get("next_quantized_block_changed_fraction", 0),
                    "first_order_prediction": evaluation["update"]["predicted_first_order_loss_change"],
                    "bf16_weight_changed_fraction": evaluation["update"]["bf16_weight_changed_fraction"],
                })
        if all(p in arms for p in ("cautious_mask", "cautious_scale", "cautious")):
            for stage in ("after_update", "after_next_quant"):
                k = {p:a["evaluation"]["stages"][stage]["kl"] for p,a in arms.items()}
                factors.append({"target":target,"stage":stage,
                    "mask_effect_without_scale":k["cautious_mask"]-k["adam"],
                    "mask_effect_with_scale":k["cautious"]-k["cautious_scale"],
                    "scale_effect_without_mask":k["cautious_scale"]-k["adam"],
                    "scale_effect_with_mask":k["cautious"]-k["cautious_mask"],
                    "interaction":k["cautious"]-k["cautious_mask"]-k["cautious_scale"]+k["adam"]})
    with (campaign / "transition_comparison.csv").open("w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    output={"mechanisms":mechanisms,"comparisons":rows,"factor_effects":factors}
    (campaign / "analysis.json").write_text(json.dumps(output,indent=2))
    lines=["# Cautious transition diagnostic", "",
           "Negative deltas favor the candidate. These are local checkpoints evaluated with the full downstream model, not final fully quantized model results.", "",
           "| Target | Policy | KL change after update (%) | KL change after next quant (%) | Next-block changed (%) |",
           "|---|---|---:|---:|---:|"]
    for target,arms in summary["results"].items():
        for policy in arms:
            if policy=="adam": continue
            pair=[r for r in rows if r["target"]==target and r["policy"]==policy]
            a,b=pair
            lines.append(f"| {target} | {policy} | {a['kl_percent_vs_adam']:+.4f} | {b['kl_percent_vs_adam']:+.4f} | {100*b['next_quantized_block_changed_fraction']:.4f} |")
    lines += ["", "## Mechanism diagnostics", "", "```json",json.dumps(mechanisms,indent=2),"```", "",
              "The conflict first-order fraction weights coordinates by |gradient × update|; it is not the fraction of update norm. Separate L1 and squared-L2 fractions are reported above."]
    (campaign / "analysis.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines[:len(rows)//2+5]))
    print(f"Artifacts: {campaign / 'analysis.json'}")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign",type=Path)
    args=parser.parse_args()
    torch.set_num_threads(4)
    analyze(args.campaign)
