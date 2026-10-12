"""Read-only probe for same-role Adam second-moment transferability.

The probe records rowwise summaries from the ordinary refresh gradients and
Adam ``v`` state.  It adds no backward pass and never changes optimizer state.
It is deliberately restricted to the single-linear REAL-Q refresh path.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch


_DONOR: dict[tuple[str, int, int], dict[str, object]] = {}


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float | None:
    a = a.double().flatten()
    b = b.double().flatten()
    if a.numel() < 2 or a.numel() != b.numel():
        return None
    a = a - a.mean()
    b = b - b.mean()
    den = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if not torch.isfinite(den) or den.item() == 0:
        return None
    return float((torch.dot(a, b) / den).item())


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float | None:
    if a.numel() < 2 or a.numel() != b.numel():
        return None
    # Average ranks for ties, implemented with stable argsort and CPU tensors.
    def ranks(x: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(x, stable=True)
        sorted_x = x[order]
        result = torch.empty_like(x, dtype=torch.float64)
        start = 0
        while start < x.numel():
            stop = start + 1
            while stop < x.numel() and sorted_x[stop] == sorted_x[start]:
                stop += 1
            result[order[start:stop]] = 0.5 * (start + stop - 1)
            start = stop
        return result
    return _pearson(ranks(a), ranks(b))


def _column_tile_log_profile(value: torch.Tensor, bins: int = 32) -> torch.Tensor:
    """Compact row-by-column sketch for a coordinate-structure screen."""

    matrix = value.detach().float().cpu()
    if matrix.ndim != 2:
        raise ValueError(f"expected a matrix for tile profile, got {matrix.shape}")
    pooled = torch.nn.functional.adaptive_avg_pool1d(
        matrix.unsqueeze(1), output_size=min(int(bins), int(matrix.shape[1]))
    ).squeeze(1).double()
    tiny = torch.finfo(torch.float64).tiny
    logged = pooled.clamp_min(tiny).log()
    return logged - logged.mean()


def _compact_device_tile_log_profile(
    value: torch.Tensor, bins: int = 32
) -> torch.Tensor:
    """Pool a matrix to a small tile sketch before copying it to the CPU."""

    if value.ndim != 2:
        raise ValueError(f"expected a matrix for tile profile, got {value.shape}")
    pooled = torch.nn.functional.adaptive_avg_pool1d(
        value.detach().float().unsqueeze(1),
        output_size=min(int(bins), int(value.shape[1])),
    ).squeeze(1)
    tiny = torch.finfo(pooled.dtype).tiny
    logged = pooled.clamp_min(tiny).log()
    logged = logged - logged.mean()
    return logged.cpu().double()


def record_moment_drift(
    *,
    layer: int,
    module: str,
    refresh_index: int,
    adam_step: int,
    grad: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    beta2: float,
) -> dict[str, object] | None:
    """Record current-gradient vs bias-corrected v profiles without mutation.

    This observer runs only when ``REALQ_ADAM_MOMENT_DRIFT_DIAG_DIR`` is set.
    It computes compact row/tile summaries after the normal Adam EMA update;
    it does not retain full gradient or moment tensors and does not alter them.
    """

    directory = os.environ.get("REALQ_ADAM_MOMENT_DRIFT_DIAG_DIR")
    if not directory or grad.ndim != 2 or exp_avg_sq.shape != grad.shape:
        return None
    allowed = {
        item.strip()
        for item in os.environ.get(
            "REALQ_ADAM_MOMENT_DRIFT_DIAG_MODULES",
            "self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj",
        ).split(",")
        if item.strip()
    }
    max_layer = int(os.environ.get("REALQ_ADAM_MOMENT_DRIFT_DIAG_MAX_LAYER", "8"))
    if module not in allowed or layer > max_layer or adam_step < 1:
        return None

    with torch.no_grad():
        grad2 = grad.detach().float().square()
        bc2 = 1.0 - float(beta2) ** int(adam_step)
        vhat = exp_avg_sq.detach().float() / max(bc2, 1e-30)
        tiny = torch.finfo(torch.float32).tiny

        grad_row = grad2.mean(dim=1).clamp_min(tiny)
        vhat_row = vhat.mean(dim=1).clamp_min(tiny)
        grad_row_log = grad_row.log()
        vhat_row_log = vhat_row.log()
        grad_row_profile = grad_row_log - grad_row_log.mean()
        vhat_row_profile = vhat_row_log - vhat_row_log.mean()
        grad_tiles = _compact_device_tile_log_profile(grad2)
        vhat_tiles = _compact_device_tile_log_profile(vhat)

        mean_g2 = float(grad2.mean().item())
        mean_vhat = float(vhat.mean().item())
        row_log_error = vhat_row_profile - grad_row_profile
        tile_log_error = vhat_tiles - grad_tiles
        record: dict[str, object] = {
            "layer": int(layer),
            "module": str(module),
            "refresh_index": int(refresh_index),
            "adam_step": int(adam_step),
            "active_rows": int(grad.shape[0]),
            "active_columns": int(grad.shape[1]),
            "mean_vhat_to_g2_ratio": mean_vhat / max(mean_g2, 1e-300),
            "log10_mean_vhat_to_g2_ratio": float(
                torch.log10(torch.tensor(max(mean_vhat, 1e-300) / max(mean_g2, 1e-300))).item()
            ),
            "row_log_profile_pearson": _pearson(grad_row_profile, vhat_row_profile),
            "row_log_profile_rmse": float(torch.sqrt(row_log_error.square().mean()).item()),
            "tile_log_profile_pearson": _pearson(grad_tiles, vhat_tiles),
            "tile_log_profile_rmse": float(torch.sqrt(tile_log_error.square().mean()).item()),
            "g2_coefficient_of_variation": float(
                grad2.std(unbiased=False).div(grad2.mean().clamp_min(tiny)).item()
            ),
            "vhat_coefficient_of_variation": float(
                vhat.std(unbiased=False).div(vhat.mean().clamp_min(tiny)).item()
            ),
        }
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "moment_drift.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def record_refresh(
    *,
    layer: int,
    module: str,
    refresh_index: int,
    adam_step: int,
    grad: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    beta2: float,
) -> None:
    """Record whether previous-layer same-role ``v`` predicts current ``g²``.

    We compare normalized per-output-row profiles, not raw parameter
    coordinates: same-name matrices across layers need not share a meaningful
    elementwise basis. The candidate premise is therefore tested at the
    coarsest axis-compatible structure first.
    """
    directory = os.environ.get("REALQ_ADAM_TRANSFER_DIAG_DIR")
    if not directory:
        return
    allowed = {
        item.strip()
        for item in os.environ.get(
            "REALQ_ADAM_TRANSFER_DIAG_MODULES",
            "self_attn.o_proj,mlp.down_proj",
        ).split(",")
        if item.strip()
    }
    max_layer = int(os.environ.get("REALQ_ADAM_TRANSFER_DIAG_MAX_LAYER", "4"))
    if module not in allowed or layer > max_layer:
        return
    if grad.ndim != 2 or exp_avg_sq.ndim != 2 or grad.shape != exp_avg_sq.shape:
        return

    grad2_row = grad.detach().float().square().mean(dim=1).cpu().double()
    bc2 = 1.0 - float(beta2) ** max(int(adam_step), 1)
    vhat = exp_avg_sq.detach().float() / bc2
    vhat_row = vhat.mean(dim=1).cpu().double()
    tiny = torch.finfo(torch.float64).tiny
    grad_profile = torch.log(grad2_row.clamp_min(tiny) / grad2_row.mean().clamp_min(tiny))
    v_profile = torch.log(vhat_row.clamp_min(tiny) / vhat_row.mean().clamp_min(tiny))

    key = (module, int(layer), int(refresh_index))
    donor_key = (module, int(layer) - 1, int(refresh_index))
    if refresh_index < 3:
        donor = _DONOR.get(donor_key)
        if layer > 0 and donor is not None:
            donor_profile = donor["profile"]
            record = {
                "layer": int(layer),
                "module": module,
                "refresh_index": int(refresh_index),
                "adam_step": int(adam_step),
                "donor_layer": int(layer) - 1,
                "donor_refresh_index": int(donor["refresh_index"]),
                "donor_adam_step": int(donor["adam_step"]),
                "shared_output_rows": min(int(grad_profile.numel()), int(torch.as_tensor(donor_profile).numel())),
                "donor_active_input_columns": int(donor["active_input_columns"]),
                "target_active_input_columns": int(grad.shape[1]),
                "normalized_row_profile_pearson": None,
                "normalized_row_profile_spearman": None,
                "donor_vhat_to_donor_g2_scale": donor["vhat_mean"] / max(donor["grad2_mean"], 1e-300),
                "donor_vhat_to_target_grad2_scale": donor["vhat_mean"] / max(float(grad2_row.mean().item()), 1e-300),
                "donor_vhat_mean": donor["vhat_mean"],
                "target_grad2_mean": float(grad2_row.mean().item()),
            }
            donor_tile = donor.get("tile_profile")
            target_tile = _column_tile_log_profile(
                grad.detach().float().square()
            )
            if donor_tile is not None and donor_tile.shape == target_tile.shape:
                record["coordinate_tile_bins"] = int(target_tile.shape[1])
                record["coordinate_tile_log_pearson"] = _pearson(
                    torch.as_tensor(donor_tile, dtype=torch.float64), target_tile
                )
            else:
                record["coordinate_tile_bins"] = None
                record["coordinate_tile_log_pearson"] = None
            # The compared profiles must summarize the same number of active
            # input columns. Otherwise a change in suffix width is confounded
            # with any apparent cross-layer profile similarity.
            if (
                grad_profile.numel() == torch.as_tensor(donor_profile).numel()
                and int(donor["active_input_columns"]) == int(grad.shape[1])
            ):
                donor_profile = torch.as_tensor(donor_profile, dtype=torch.float64)
                record["normalized_row_profile_pearson"] = _pearson(donor_profile, grad_profile)
                record["normalized_row_profile_spearman"] = _spearman(donor_profile, grad_profile)
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            with (path / "row_profile_correlations.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")

    # Match corresponding early refreshes across layers. Caching only the
    # latest donor refresh would compare a narrow trailing suffix to the
    # target's broad first-refresh suffix, confounding the diagnostic.
    if refresh_index < 3:
        _DONOR[key] = {
            "profile": v_profile,
            "tile_profile": _column_tile_log_profile(vhat),
            "vhat_mean": float(vhat_row.mean().item()),
            "grad2_mean": float(grad2_row.mean().item()),
            "adam_step": int(adam_step),
            "refresh_index": int(refresh_index),
            "active_input_columns": int(grad.shape[1]),
        }


def initialize_v_from_previous_row_profile(
    *,
    layer: int,
    module: str,
    refresh_index: int,
    grad: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    beta2: float,
) -> dict[str, float | int] | None:
    """Optionally seed one target refresh's v from a matched donor profile.

    This experimental path is disabled by default. It is limited to the
    target layer/module specified in the environment, the first matched
    refresh, and an online row-profile correlation threshold. The donor's
    normalized row profile is scaled by the target gradient's already
    available mean squared magnitude; no extra backward pass is performed.
    """
    mode = os.environ.get("REALQ_ADAM_V_TRANSFER_MODE")
    if mode != "normalized_row_prior":
        return None
    try:
        target_layer = int(os.environ["REALQ_ADAM_V_TRANSFER_TARGET_LAYER"])
        target_module = os.environ["REALQ_ADAM_V_TRANSFER_TARGET_MODULE"]
        min_corr = float(os.environ.get("REALQ_ADAM_V_TRANSFER_MIN_CORR", "0.4"))
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "normalized_row_prior requires an integer target layer, target "
            "module, and valid min correlation"
        ) from exc
    is_target = (
        layer == target_layer and module == target_module and refresh_index == 0
    )
    if is_target:
        logging.info(
            "[adam_v_transfer_decision] layer=%d module=%s refresh=%d "
            "entry=1 mode=%s grad_shape=%s v_shape=%s",
            layer,
            module,
            refresh_index,
            mode,
            tuple(grad.shape),
            tuple(exp_avg_sq.shape),
        )
    if not is_target:
        return None
    if grad.ndim != 2 or exp_avg_sq.ndim != 2 or grad.shape != exp_avg_sq.shape:
        logging.info(
            "[adam_v_transfer_decision] layer=%d module=%s refresh=%d "
            "result=skip reason=shape_mismatch grad_shape=%s v_shape=%s",
            layer,
            module,
            refresh_index,
            tuple(grad.shape),
            tuple(exp_avg_sq.shape),
        )
        return None

    donor = _DONOR.get((module, int(layer) - 1, int(refresh_index)))
    if donor is None:
        logging.info(
            "[adam_v_transfer_decision] layer=%d module=%s refresh=%d "
            "result=skip reason=donor_missing donor_layer=%d",
            layer,
            module,
            refresh_index,
            layer - 1,
        )
        return None
    if int(donor["active_input_columns"]) != int(grad.shape[1]):
        logging.info(
            "[adam_v_transfer_decision] layer=%d module=%s refresh=%d "
            "result=skip reason=active_suffix_mismatch donor_columns=%d "
            "target_columns=%d",
            layer,
            module,
            refresh_index,
            int(donor["active_input_columns"]),
            int(grad.shape[1]),
        )
        return None
    donor_log_profile = torch.as_tensor(donor["profile"], dtype=torch.float64)
    grad2_row = grad.detach().float().square().mean(dim=1).cpu().double()
    if donor_log_profile.numel() != grad2_row.numel():
        logging.info(
            "[adam_v_transfer_decision] layer=%d module=%s refresh=%d "
            "result=skip reason=row_count_mismatch donor_rows=%d target_rows=%d",
            layer,
            module,
            refresh_index,
            donor_log_profile.numel(),
            grad2_row.numel(),
        )
        return None
    target_log_profile = torch.log(
        grad2_row.clamp_min(torch.finfo(torch.float64).tiny)
        / grad2_row.mean().clamp_min(torch.finfo(torch.float64).tiny)
    )
    correlation = _pearson(donor_log_profile, target_log_profile)
    if correlation is None:
        logging.info(
            "[adam_v_transfer_decision] layer=%d module=%s refresh=%d "
            "result=skip reason=invalid_correlation threshold=%.6g",
            layer,
            module,
            refresh_index,
            min_corr,
        )
        return None
    if correlation < min_corr:
        logging.info(
            "[adam_v_transfer_decision] layer=%d module=%s refresh=%d "
            "result=skip reason=correlation_below_gate corr=%.8g threshold=%.8g",
            layer,
            module,
            refresh_index,
            correlation,
            min_corr,
        )
        return None

    shrinkage = float(os.environ.get("REALQ_ADAM_V_TRANSFER_SHRINKAGE", "1.0"))
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError(
            "REALQ_ADAM_V_TRANSFER_SHRINKAGE must be in [0, 1], "
            f"got {shrinkage!r}"
        )
    row_prior = donor_log_profile.exp()
    row_prior = row_prior / row_prior.mean().clamp_min(torch.finfo(torch.float64).tiny)
    row_prior = row_prior.to(device=grad.device)
    target_scale = grad2_row.mean().to(device=grad.device)
    donor_vhat_prior = row_prior[:, None] * target_scale
    target_g2 = grad.detach().double().square()
    tiny = torch.finfo(torch.float64).tiny
    # Interpolate in log space between a cross-layer row prior and the
    # target's already-observed coordinatewise g^2. At shrinkage=0 the
    # virtual prior equals this refresh's g^2, so after the ordinary EMA
    # update its bias-corrected vhat is exactly the baseline first-step g^2.
    # This makes the knob a true bounded perturbation of the Adam preconditioner.
    log_prior = (
        shrinkage * donor_vhat_prior.clamp_min(tiny).log()
        + (1.0 - shrinkage) * target_g2.clamp_min(tiny).log()
    )
    vhat_prior = log_prior.exp().to(
        device=exp_avg_sq.device, dtype=exp_avg_sq.dtype
    )
    donor_age = max(int(donor["adam_step"]), 1)
    # Store the bias-uncorrected EMA corresponding to the donor's local age.
    # Only v borrows this virtual age; the target's m remains at its true local
    # age because no donor first moment is transferred.
    exp_avg_sq.copy_(
        (1.0 - float(beta2) ** donor_age) * vhat_prior
    )
    donor_to_target_scale = float(donor["vhat_mean"]) / max(
        float(target_scale.item()), 1e-300
    )
    return {
        "correlation": float(correlation),
        "donor_age": donor_age,
        "target_v_step": donor_age + 1,
        "donor_to_target_scale": donor_to_target_scale,
        "shrinkage": shrinkage,
    }
