"""Versioned, reproducible checkpoints for fake-quantized REAL-Q models.

The quantized weights live in the ordinary model ``state_dict``.  Activation
and KV-cache fake quantization, however, is runtime behavior: its bit-widths
and clipping settings are plain Python attributes and the K-cache path is
installed by a forward monkeypatch.  A state_dict alone therefore cannot
reconstruct the model that was evaluated.

This module keeps the tensor payload safe for ``torch.load(weights_only=True)``
and stores the runtime behavior in a primitive-only, versioned manifest.
"""
from __future__ import annotations

import logging
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from typing import Any

import torch


CHECKPOINT_FORMAT = "realq.quantized_model"
CHECKPOINT_VERSION = 1

_RUNTIME_FIELDS = (
    "rotate",
    "a_bits",
    "a_groupsize",
    "a_asym",
    "a_clip_ratio",
    "v_bits",
    "v_groupsize",
    "v_asym",
    "v_clip_ratio",
    "k_bits",
    "k_groupsize",
    "k_asym",
    "k_clip_ratio",
    "act_quant_aware_gptq",
    "k_cache_quant_aware_gptq",
)
_WEIGHT_PROVENANCE_FIELDS = (
    "w_bits",
    "w_groupsize",
    "w_asym",
    "w_clip",
    "w_method",
    "quantizer_inner_fastpath",
)
_BOOL_FIELDS = {
    "rotate",
    "a_asym",
    "v_asym",
    "k_asym",
    "act_quant_aware_gptq",
    "k_cache_quant_aware_gptq",
}
_DYNAMIC_QUANT_BUFFER_SUFFIXES = (
    ".quantizer.maxq",
    ".quantizer.scale",
    ".quantizer.zero",
    ".out_quantizer.maxq",
    ".out_quantizer.scale",
    ".out_quantizer.zero",
    ".k_quantizer.maxq",
    ".k_quantizer.scale",
    ".k_quantizer.zero",
)


def _model_config_identity(model: torch.nn.Module) -> str:
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "to_dict"):
        return "none"
    payload = dict(config.to_dict())
    # These describe the loader/location rather than the architecture.
    for key in ("_name_or_path", "name_or_path", "transformers_version"):
        payload.pop(key, None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tokenizer_identity(tokenizer: Any, fallback_source: str) -> str:
    if tokenizer is None:
        from utils.cache_identity import artifact_identity

        return f"source-artifact:{artifact_identity(fallback_source)}"
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    special = getattr(tokenizer, "special_tokens_map", {})
    payload = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "vocab": sorted((str(token), int(index)) for token, index in vocab.items()),
        "special_tokens_map": special,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _artifact_manifest(
    config: Any,
    model: torch.nn.Module,
    tokenizer: Any,
) -> dict[str, Any]:
    from utils.cache_identity import artifact_identity
    from utils.model_utils import rotation_cache_identity

    source = str(getattr(config, "model", ""))
    return {
        "source_model": artifact_identity(source),
        "tokenizer": _tokenizer_identity(tokenizer, source),
        "rotation": rotation_cache_identity(config),
        "rotation_seed": int(getattr(config, "rotation_seed", 0)),
        "model_config": _model_config_identity(model),
        "parameter_dtypes": sorted(
            {str(parameter.dtype) for parameter in model.parameters()}
        ),
    }


def build_runtime_manifest(
    config: Any,
    model: torch.nn.Module | None = None,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """Return the primitive-only runtime/provenance manifest."""
    runtime = {name: getattr(config, name) for name in _RUNTIME_FIELDS}
    _validate_runtime_manifest(runtime)
    provenance = {
        name: getattr(config, name)
        for name in _WEIGHT_PROVENANCE_FIELDS
        if hasattr(config, name)
    }
    manifest = {
        "runtime_quantization": runtime,
        "weight_quantization": provenance,
        "base_model": str(getattr(config, "model", "")),
    }
    if model is not None:
        manifest["artifact_identity"] = _artifact_manifest(
            config, model, tokenizer
        )
    return manifest


def _validate_runtime_manifest(runtime: Mapping[str, Any]) -> None:
    missing = [name for name in _RUNTIME_FIELDS if name not in runtime]
    if missing:
        raise ValueError(
            "Quantized checkpoint runtime manifest is incomplete; missing "
            f"{missing}."
        )
    for name in _BOOL_FIELDS:
        if type(runtime[name]) is not bool:
            raise ValueError(f"Checkpoint field {name!r} must be bool.")
    for prefix in ("a", "v", "k"):
        bits = runtime[f"{prefix}_bits"]
        groupsize = runtime[f"{prefix}_groupsize"]
        ratio = runtime[f"{prefix}_clip_ratio"]
        if type(bits) is not int or not 2 <= bits <= 16:
            raise ValueError(
                f"Checkpoint field {prefix}_bits must be an integer in [2, 16]."
            )
        if type(groupsize) is not int or (groupsize != -1 and groupsize <= 0):
            raise ValueError(
                f"Checkpoint field {prefix}_groupsize must be -1 or positive."
            )
        if not isinstance(ratio, (int, float)) or isinstance(ratio, bool):
            raise ValueError(
                f"Checkpoint field {prefix}_clip_ratio must be numeric."
            )
        if not 0.0 < float(ratio) <= 1.0:
            raise ValueError(
                f"Checkpoint field {prefix}_clip_ratio must be in (0, 1]."
            )


def save_quantized_checkpoint(
    path: str,
    model: torch.nn.Module,
    config: Any,
    tokenizer: Any = None,
) -> None:
    """Atomically save weights plus the runtime fake-quantization manifest."""
    destination = os.path.abspath(os.fspath(path))
    parent = os.path.dirname(destination) or "."
    os.makedirs(parent, exist_ok=True)
    manifest = build_runtime_manifest(config, model, tokenizer)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        **manifest,
    }

    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.",
        suffix=".tmp",
        dir=parent,
    )
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_quantized_checkpoint(
    path: str,
    *,
    map_location: str | torch.device = "cpu",
    allow_unsafe_legacy: bool = False,
) -> dict[str, Any]:
    """Load a new safe checkpoint or a trusted legacy checkpoint.

    PyTorch 2.6 defaults ``torch.load`` to ``weights_only=True``.  New
    checkpoints are deliberately compatible with that mode.  Historical
    GPTQ+ payloads embedded ``WeightQuantizer`` module objects, so a safe load
    cannot unpickle them.  They are accepted only when the caller explicitly
    opts into unsafe pickle loading; the object-valued quantizer mapping is
    then immediately discarded because it is not needed to execute the
    already fake-quantized weights.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Quantized checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as safe_error:
        if not allow_unsafe_legacy:
            raise RuntimeError(
                f"Checkpoint {path} could not be loaded safely with "
                "weights_only=True. If and only if this is a trusted legacy "
                "GPTQ+ artifact, retry with --allow_unsafe_legacy_checkpoint. "
                f"Original error: {safe_error}"
            ) from safe_error
        logging.warning(
            "Explicitly authorized unsafe pickle load for trusted legacy "
            "checkpoint %s (weights_only failure: %s).",
            path,
            safe_error,
        )
        payload = torch.load(path, map_location=map_location, weights_only=False)

    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Quantized checkpoint must contain a mapping, got {type(payload)!r}."
        )

    # Also accept a raw state_dict for the earliest user-produced artifacts.
    if "model" not in payload and payload and all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in payload.items()
    ):
        payload = {"model": payload}

    state = payload.get("model")
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ValueError("Quantized checkpoint does not contain a valid model state_dict.")

    checkpoint_format = payload.get("format")
    if checkpoint_format is None:
        logging.warning(
            "Loading legacy quantized checkpoint without a runtime manifest; "
            "A/K/V behavior will use the current command-line configuration."
        )
        return {
            "format": "legacy.gptq_plus",
            "format_version": 0,
            "model": state,
            "runtime_quantization": None,
            "weight_quantization": {},
            "base_model": "",
            "artifact_identity": None,
        }

    if checkpoint_format != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported quantized checkpoint format {checkpoint_format!r}.")
    version = payload.get("format_version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported {CHECKPOINT_FORMAT} version {version!r}; "
            f"expected {CHECKPOINT_VERSION}."
        )
    runtime = payload.get("runtime_quantization")
    if not isinstance(runtime, Mapping):
        raise ValueError("Quantized checkpoint runtime manifest must be a mapping.")
    _validate_runtime_manifest(runtime)
    weight_quantization = payload.get("weight_quantization", {})
    if not isinstance(weight_quantization, Mapping):
        raise ValueError("Checkpoint weight-quantization provenance must be a mapping.")
    artifact_manifest = payload.get("artifact_identity")
    if not isinstance(artifact_manifest, Mapping):
        raise ValueError("Checkpoint artifact-identity manifest must be a mapping.")
    required_identities = {
        "source_model",
        "tokenizer",
        "rotation",
        "rotation_seed",
        "model_config",
        "parameter_dtypes",
    }
    missing_identities = sorted(required_identities - set(artifact_manifest))
    if missing_identities:
        raise ValueError(
            "Checkpoint artifact-identity manifest is incomplete; missing "
            f"{missing_identities}."
        )
    return {
        "format": checkpoint_format,
        "format_version": version,
        "model": state,
        "runtime_quantization": dict(runtime),
        "weight_quantization": dict(weight_quantization),
        "base_model": str(payload.get("base_model", "")),
        "artifact_identity": dict(artifact_manifest),
    }


def apply_runtime_manifest(config: Any, checkpoint: Mapping[str, Any]) -> bool:
    """Restore A/V/K/rotation behavior onto an argparse namespace or Config.

    Returns ``False`` for a legacy checkpoint that had no manifest.
    """
    runtime = checkpoint.get("runtime_quantization")
    if runtime is None:
        return False
    _validate_runtime_manifest(runtime)
    for name in _RUNTIME_FIELDS:
        previous = getattr(config, name, None)
        restored = runtime[name]
        if previous != restored:
            logging.info(
                "Restoring checkpoint runtime setting %s=%r (command line had %r).",
                name,
                restored,
                previous,
            )
        setattr(config, name, restored)
    # These fields no longer execute quantization when loading, but restoring
    # them keeps evaluation labels/config dumps truthful and, importantly,
    # prevents the legacy entry point from treating a W4 artifact as W16.
    provenance = checkpoint.get("weight_quantization", {})
    for name in _WEIGHT_PROVENANCE_FIELDS:
        if name in provenance and hasattr(config, name):
            setattr(config, name, provenance[name])
    identities = checkpoint.get("artifact_identity") or {}
    if "rotation_seed" in identities and hasattr(config, "rotation_seed"):
        setattr(config, "rotation_seed", int(identities["rotation_seed"]))
    return True


def validate_artifact_identity(
    config: Any,
    checkpoint: Mapping[str, Any],
    *,
    model: torch.nn.Module | None = None,
    tokenizer: Any = None,
) -> bool:
    """Strictly reject a checkpoint prepared for another model/runtime.

    This check intentionally happens both before model construction
    (source+rotation) and after it (architecture, dtype, tokenizer).  A legacy
    checkpoint has no identities and returns ``False`` for compatibility.
    """
    identities = checkpoint.get("artifact_identity")
    if identities is None:
        return False
    from utils.cache_identity import artifact_identity
    from utils.model_utils import rotation_cache_identity

    actual = {
        "source_model": artifact_identity(getattr(config, "model", "")),
        "rotation": rotation_cache_identity(config),
        "rotation_seed": int(getattr(config, "rotation_seed", 0)),
    }
    if model is not None:
        actual["model_config"] = _model_config_identity(model)
        actual["parameter_dtypes"] = sorted(
            {str(parameter.dtype) for parameter in model.parameters()}
        )
    if tokenizer is not None:
        actual["tokenizer"] = _tokenizer_identity(
            tokenizer, str(getattr(config, "model", ""))
        )
    mismatches = {
        key: {"checkpoint": identities.get(key), "current": value}
        for key, value in actual.items()
        if identities.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Quantized checkpoint artifact identity does not match the current "
            f"model/runtime: {mismatches}."
        )
    return True


def load_model_state(
    model: torch.nn.Module,
    checkpoint: Mapping[str, Any],
) -> None:
    """Load weights strictly except for obsolete dynamic quantizer buffers."""
    incompat = model.load_state_dict(checkpoint["model"], strict=False)
    unexpected = [
        key
        for key in incompat.unexpected_keys
        if not key.endswith(_DYNAMIC_QUANT_BUFFER_SUFFIXES)
    ]
    missing = [
        key
        for key in incompat.missing_keys
        if not key.endswith(_DYNAMIC_QUANT_BUFFER_SUFFIXES)
    ]
    if missing or unexpected:
        raise RuntimeError(
            "Quantized checkpoint does not match the prepared model: "
            f"missing={missing}, unexpected={unexpected}."
        )
