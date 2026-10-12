"""Compare same-Stage-0 sequence score sketches with exact KL-HVP snapshots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def cosine(a, b):
    return float(F.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0).clamp(-1, 1))


def metrics(pred, truth):
    return {
        "cosine": cosine(pred.float(), truth.float()),
        "rms_ratio": float(pred.square().mean().sqrt() / truth.square().mean().sqrt().clamp_min(1e-30)),
        "per_sequence_cosine": [cosine(a, b) for a, b in zip(pred, truth)],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sketch-dir", type=Path, required=True)
    p.add_argument("--hvp-dir", type=Path, required=True)
    p.add_argument("--target", required=True, help="HVP transition key")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    g = torch.load(a.sketch_dir / f"layer{a.layer}_sequence_gradients.pt", map_location="cpu", weights_only=True).double()
    item = torch.load(a.hvp_dir / f"{a.target}_true_kl_vectors.pt", map_location="cpu", weights_only=True)
    delta = item["delta"].double()
    true = item["true_kl_gradient"].double()
    hvp = item["exact_kl_hvp"].double()
    if tuple(g.shape[1:]) != tuple(delta.shape[1:]):
        raise ValueError(f"shape mismatch: sketch {tuple(g.shape)}, transition {tuple(delta.shape)}")
    n, t, h = g.shape
    flat_g = g.reshape(n, -1)
    flat_delta = delta.reshape(delta.shape[0], -1)
    # E[g_seq g_seq^T] Delta / T. This is precisely the sequence cross-token
    # covariance that is absent from REAL-Q's shared token-row HxH Fisher.
    pred = ((flat_delta @ flat_g.T) @ flat_g) / float(n * t)
    pred = pred.reshape_as(delta)

    # Existing shared Fisher in the transition snapshot, reconstructed from
    # per-position score vectors to compare like-for-like on the same delta.
    row_g = g.reshape(-1, h)
    shared_f = row_g.T @ row_g / float(n * t)
    shared = torch.einsum("hk,ntk->nth", shared_f, delta) / float(t)

    result = {
        "target": a.target,
        "static_layer": a.layer,
        "calibration_gradient_shape": list(g.shape),
        "calibration_samples": n,
        "sequence_length": t,
        "stored_gradient_bytes": int((a.sketch_dir / f"layer{a.layer}_sequence_gradients.pt").stat().st_size),
        "operator": "sample sequence-score covariance (rank <= calibration_samples), applied as C @ delta / T",
        "sequence_covariance_vs_true_kl_gradient": metrics(pred, true),
        "sequence_covariance_vs_exact_kl_hvp": metrics(pred, hvp),
        "shared_token_fisher_same_capture_vs_true_kl_gradient": metrics(shared, true),
        "shared_token_fisher_same_capture_vs_exact_kl_hvp": metrics(shared, hvp),
        "exact_hvp_vs_true_kl_gradient": metrics(hvp, true),
        "scope": "screening diagnostic only; capture reused REAL-Q Stage-0 backward, but only 8 sampled calibration sequences and one transition seed",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
