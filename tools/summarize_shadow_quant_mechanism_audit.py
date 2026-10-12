"""Summarize paired shadow/actual next-block state and gradient records."""
import json
import argparse
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _cosine(a, b):
    a, b = a.reshape(-1).double(), b.reshape(-1).double()
    den = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if den == 0:
        return None
    return float(torch.dot(a, b) / den)


def _compare(a, b):
    a, b = a.float(), b.float()
    diff = a - b
    denom = torch.linalg.vector_norm(b)
    return {
        "shape_equal": list(a.shape) == list(b.shape),
        "mse": float(diff.square().mean()),
        "relative_l2": float(torch.linalg.vector_norm(diff) / denom.clamp_min(1e-30)),
        "cosine": _cosine(a, b),
        "max_abs": float(diff.abs().max()),
    }


def _sign_disagreement(a, b):
    a, b = a.float(), b.float()
    active = (a != 0) | (b != 0)
    if not bool(active.any()):
        return 0.0
    return float(((a[active] * b[active]) < 0).float().mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", default="outputs/shadow_quant_mechanism_audit_qproj_20260929")
    args = parser.parse_args()
    run = Path(args.run_dir)
    if not run.is_absolute():
        run = ROOT / run
    events = run / "events"
    records = []
    for path in sorted(events.glob("event_*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=False)
        meta = data["metadata"]
        if meta["sample_indices"] != meta["actual_sample_indices"]:
            raise RuntimeError(f"Sample mismatch in {path.name}")
        c0, c1 = meta["predicted_cols"]
        a0, a1 = meta["actual_cols"]
        if [c0, c1] != [a0, a1]:
            raise RuntimeError(f"Column mismatch in {path.name}")
        pred_w, actual_w = data["predicted_weight"], data["actual_weight"]
        pred_g, actual_g = data["predicted_gradient"], data["actual_gradient"]
        pred_y, actual_y = data["predicted_module_output"], data["actual_module_output"]
        if any(t is None for t in (pred_y, actual_y)):
            raise RuntimeError(f"Missing module output capture in {path.name}")
        record = {
            "file": path.name,
            "metadata": meta,
            "predicted_next_block_weight": _compare(pred_w[:, c0:c1], actual_w[:, c0:c1]),
            "remaining_tail_weight": _compare(pred_w[:, c1:], actual_w[:, c1:]),
            "predicted_next_block_gradient": _compare(pred_g[:, c0:c1], actual_g[:, c0:c1]),
            "remaining_tail_gradient": _compare(pred_g[:, c1:], actual_g[:, c1:]),
            "all_gradient": _compare(pred_g, actual_g),
            "module_output": _compare(pred_y, actual_y),
        }
        adam_aware_weight = data.get("adam_aware_next_block_weight")
        if adam_aware_weight is not None:
            record["adam_aware_preview_next_block_weight"] = _compare(
                adam_aware_weight, actual_w[:, c0:c1]
            )
        pred_adam = data.get("predicted_adam_update_same_state")
        actual_adam = data.get("actual_gradient_adam_update_same_state")
        if pred_adam is not None and actual_adam is not None:
            record["predicted_next_block_adam_update_same_state"] = _compare(
                pred_adam, actual_adam
            )
            record["adam_update_sign_disagreement_fraction"] = _sign_disagreement(
                pred_adam, actual_adam
            )
            record["adam_update_norms"] = {
                "shadow": float(torch.linalg.vector_norm(pred_adam)),
                "actual_gradient": float(torch.linalg.vector_norm(actual_adam)),
            }
        records.append(record)
    summary = {
        "run": str(run),
        "event_count": len(records),
        "records": records,
    }
    output = run / "mechanism_summary.json"
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
