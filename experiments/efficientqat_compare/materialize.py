"""CPU-only materialisation of EfficientQAT packed linear layers.

EfficientQAT's E2E-QP checkpoint stores each transformer projection as an
``int_linear_real.QuantLinear``: integer weights/zero-points are packed into
signed ``int32`` tensors while the learned scales remain floating point.  The
controlled comparison evaluates an ordinary Hugging Face model, so packed
modules must be decoded and replaced by standard BF16 ``nn.Linear`` modules.

The bit decoder deliberately does not call EfficientQAT's Triton kernels.
The checkpoint-level entry point uses the upstream loader only to construct
the packed module graph; decoding itself is ordinary CPU PyTorch bitwise
arithmetic and is therefore usable with no visible CUDA device.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


MATERIALIZATION_FORMAT = "efficientqat.materialized_hf"
MATERIALIZATION_VERSION = 1
TARGET_DTYPE = torch.bfloat16
SOURCE_QUANTIZER = "efficientqat_unsigned_asymmetric_minmax"
SYMMETRIC_SOURCE_QUANTIZER = (
    "efficientqat_symmetric_signed_grid_fixed_zero"
)
SUPPORTED_SOURCE_QUANTIZERS = frozenset(
    {SOURCE_QUANTIZER, SYMMETRIC_SOURCE_QUANTIZER}
)
_SUPPORTED_BITS = frozenset({2, 3, 4, 8})
_PACKED_FIELDS = (
    "qweight",
    "qzeros",
    "scales",
    "bits",
    "group_size",
    "infeatures",
    "outfeatures",
)


def _positive_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return value


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    # NumPy cannot represent bfloat16 directly.  Viewing the contiguous tensor
    # as bytes hashes its exact serialized payload for every PyTorch dtype.
    digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _module_digest(records: list[dict[str, Any]]) -> str:
    payload = [
        {
            "name": record["name"],
            "weight_sha256": record["weight_sha256"],
            "bias_sha256": record["bias_sha256"],
        }
        for record in records
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def is_efficientqat_packed_linear(module: nn.Module) -> bool:
    """Return whether ``module`` exposes the complete EfficientQAT payload."""

    return all(hasattr(module, field) for field in _PACKED_FIELDS)


def _unsigned_words(packed: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(packed, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if packed.dtype != torch.int32:
        raise TypeError(
            f"{name} must use torch.int32 packed words; got {packed.dtype}."
        )
    if packed.device.type != "cpu":
        raise ValueError(
            f"{name} is on {packed.device}; CPU materialisation requires a "
            "CPU-resident packed checkpoint."
        )
    # Conversion sign-extends negative int32 values.  Masking restores the
    # original uint32 bit pattern used by EfficientQAT's NumPy packer.
    return packed.detach().contiguous().to(torch.int64) & 0xFFFFFFFF


def unpack_packed_dim0(
    packed: torch.Tensor,
    *,
    bits: int,
    logical_rows: int,
) -> torch.Tensor:
    """Unpack values packed along dimension 0.

    This is the CPU equivalent of EfficientQAT's ``dequant_dim0`` primitive.
    The returned tensor is integer-valued ``int64`` with shape
    ``[logical_rows, packed.shape[1]]``.
    """

    if bits not in _SUPPORTED_BITS:
        raise ValueError(
            f"EfficientQAT packed bits must be one of "
            f"{sorted(_SUPPORTED_BITS)}; got {bits!r}."
        )
    logical_rows = _positive_int(logical_rows, name="logical_rows")
    if packed.ndim != 2:
        raise ValueError(
            f"dim0 packed tensor must be rank 2; got shape {tuple(packed.shape)}."
        )
    lanes = 32 // bits
    expected_words = math.ceil(logical_rows / lanes)
    if packed.shape[0] != expected_words:
        raise ValueError(
            "dim0 packed row count is invalid: "
            f"{packed.shape[0]} != ceil({logical_rows}/{lanes})={expected_words}."
        )

    words = _unsigned_words(packed, name="qweight")
    mask = (1 << bits) - 1
    decoded = torch.stack(
        [
            (words >> (lane * bits)) & mask
            for lane in range(lanes)
        ],
        dim=1,
    )
    return decoded.reshape(-1, packed.shape[1])[:logical_rows].contiguous()


def unpack_packed_dim1(
    packed: torch.Tensor,
    *,
    bits: int,
    logical_columns: int,
) -> torch.Tensor:
    """Unpack values packed along dimension 1.

    This is the CPU equivalent of EfficientQAT's ``dequant_dim1`` primitive.
    The returned tensor is integer-valued ``int64`` with shape
    ``[packed.shape[0], logical_columns]``.
    """

    if bits not in _SUPPORTED_BITS:
        raise ValueError(
            f"EfficientQAT packed bits must be one of "
            f"{sorted(_SUPPORTED_BITS)}; got {bits!r}."
        )
    logical_columns = _positive_int(
        logical_columns, name="logical_columns"
    )
    if packed.ndim != 2:
        raise ValueError(
            f"dim1 packed tensor must be rank 2; got shape {tuple(packed.shape)}."
        )
    lanes = 32 // bits
    expected_words = math.ceil(logical_columns / lanes)
    if packed.shape[1] != expected_words:
        raise ValueError(
            "dim1 packed column count is invalid: "
            f"{packed.shape[1]} != "
            f"ceil({logical_columns}/{lanes})={expected_words}."
        )

    words = _unsigned_words(packed, name="qzeros")
    mask = (1 << bits) - 1
    decoded = torch.stack(
        [
            (words >> (lane * bits)) & mask
            for lane in range(lanes)
        ],
        dim=2,
    )
    return decoded.reshape(packed.shape[0], -1)[
        :, :logical_columns
    ].contiguous()


def _validate_packed_module(
    module: nn.Module,
    *,
    name: str,
    expected_bits: int,
    expected_group_size: int,
) -> tuple[int, int, int, int]:
    if not is_efficientqat_packed_linear(module):
        raise TypeError(f"{name!r} is not an EfficientQAT packed QuantLinear.")

    bits = int(module.bits)
    group_size = int(module.group_size)
    in_features = int(module.infeatures)
    out_features = int(module.outfeatures)
    if bits != expected_bits:
        raise ValueError(
            f"{name}: packed bit-width {bits} != expected {expected_bits}."
        )
    if group_size != expected_group_size:
        raise ValueError(
            f"{name}: packed group size {group_size} != "
            f"expected {expected_group_size}."
        )
    if in_features <= 0 or out_features <= 0:
        raise ValueError(
            f"{name}: invalid feature shape ({in_features}, {out_features})."
        )
    if in_features % group_size != 0:
        raise ValueError(
            f"{name}: in_features={in_features} is not divisible by "
            f"group_size={group_size}; upstream E2E-QP cannot represent a "
            "partial final group."
        )

    groups = in_features // group_size
    scales = module.scales
    if not isinstance(scales, torch.Tensor):
        raise TypeError(f"{name}.scales must be a tensor.")
    if scales.device.type != "cpu":
        raise ValueError(
            f"{name}.scales is on {scales.device}; CPU materialisation "
            "requires a CPU-resident checkpoint."
        )
    if tuple(scales.shape) != (groups, out_features):
        raise ValueError(
            f"{name}.scales has shape {tuple(scales.shape)}; "
            f"expected {(groups, out_features)}."
        )
    if not torch.isfinite(scales).all():
        raise ValueError(f"{name}.scales contains non-finite values.")
    if not torch.all(scales > 0):
        raise ValueError(
            f"{name}.scales contains a non-positive learned quantization step."
        )

    lanes = 32 // bits
    expected_qweight = (math.ceil(in_features / lanes), out_features)
    expected_qzeros = (groups, math.ceil(out_features / lanes))
    if tuple(module.qweight.shape) != expected_qweight:
        raise ValueError(
            f"{name}.qweight has shape {tuple(module.qweight.shape)}; "
            f"expected {expected_qweight}."
        )
    if tuple(module.qzeros.shape) != expected_qzeros:
        raise ValueError(
            f"{name}.qzeros has shape {tuple(module.qzeros.shape)}; "
            f"expected {expected_qzeros}."
        )
    _unsigned_words(module.qweight, name=f"{name}.qweight")
    _unsigned_words(module.qzeros, name=f"{name}.qzeros")

    g_idx = getattr(module, "g_idx", None)
    if g_idx is not None:
        if not isinstance(g_idx, torch.Tensor) or g_idx.device.type != "cpu":
            raise ValueError(f"{name}.g_idx must be a CPU tensor.")
        expected_g_idx = torch.arange(
            in_features, dtype=torch.int32
        ) // group_size
        if g_idx.dtype != torch.int32 or not torch.equal(
            g_idx.detach().contiguous(), expected_g_idx
        ):
            raise ValueError(
                f"{name}.g_idx does not describe contiguous natural groups."
            )

    bias = getattr(module, "bias", None)
    if bias is not None:
        if not isinstance(bias, torch.Tensor):
            raise TypeError(f"{name}.bias must be a tensor or None.")
        if bias.device.type != "cpu":
            raise ValueError(
                f"{name}.bias is on {bias.device}; CPU materialisation "
                "requires a CPU-resident checkpoint."
            )
        if tuple(bias.shape) != (out_features,):
            raise ValueError(
                f"{name}.bias has shape {tuple(bias.shape)}; "
                f"expected {(out_features,)}."
            )
        if not torch.isfinite(bias).all():
            raise ValueError(f"{name}.bias contains non-finite values.")

    return bits, group_size, in_features, out_features


@torch.no_grad()
def decode_efficientqat_weight_cpu(
    module: nn.Module,
    *,
    expected_bits: int,
    expected_group_size: int,
    name: str = "<packed-linear>",
    source_quantizer: str = SOURCE_QUANTIZER,
) -> torch.Tensor:
    """Decode one packed EfficientQAT weight into ``[out, in]`` BF16."""

    if source_quantizer not in SUPPORTED_SOURCE_QUANTIZERS:
        raise ValueError(
            "unsupported EfficientQAT source quantizer: "
            f"{source_quantizer!r}"
        )

    bits, group_size, in_features, out_features = _validate_packed_module(
        module,
        name=name,
        expected_bits=expected_bits,
        expected_group_size=expected_group_size,
    )
    groups = in_features // group_size
    codes = unpack_packed_dim0(
        module.qweight,
        bits=bits,
        logical_rows=in_features,
    ).to(torch.float32)
    zeros = unpack_packed_dim1(
        module.qzeros,
        bits=bits,
        logical_columns=out_features,
    ).to(torch.float32)
    if source_quantizer == SYMMETRIC_SOURCE_QUANTIZER:
        expected_zero = 2 ** (bits - 1)
        if not torch.all(zeros == expected_zero):
            unique = torch.unique(zeros).tolist()
            raise ValueError(
                f"{name}: symmetric EfficientQAT requires fixed packed "
                f"zero-point {expected_zero}; found {unique[:16]!r}."
            )
    scales = module.scales.detach().to(device="cpu", dtype=torch.float32)

    weight_in_out = (
        (
            codes.reshape(groups, group_size, out_features)
            - zeros.unsqueeze(1)
        )
        * scales.unsqueeze(1)
    ).reshape(in_features, out_features)
    weight = weight_in_out.transpose(0, 1).contiguous().to(TARGET_DTYPE)
    if not torch.isfinite(weight).all():
        raise ValueError(f"{name}: decoded BF16 weight contains non-finite values.")
    return weight


def _resolve_parent(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    if not name:
        raise ValueError("Cannot replace the root module itself.")
    parts = name.split(".")
    parent: nn.Module = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
        if not isinstance(parent, nn.Module):
            raise TypeError(
                f"Path component {part!r} in {name!r} is not an nn.Module."
            )
    return parent, parts[-1]


def _set_module(root: nn.Module, name: str, replacement: nn.Module) -> None:
    parent, leaf = _resolve_parent(root, name)
    if leaf.isdigit():
        parent[int(leaf)] = replacement
    else:
        setattr(parent, leaf, replacement)


def _get_module(root: nn.Module, name: str) -> nn.Module:
    module: nn.Module = root
    for part in name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    if not isinstance(module, nn.Module):
        raise TypeError(f"Manifest path {name!r} did not resolve to an nn.Module.")
    return module


def _jsonable_provenance(value: Mapping[str, Any] | None) -> dict[str, Any]:
    result = {} if value is None else dict(value)
    try:
        json.dumps(result, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Materialization provenance must contain only finite "
            "JSON-serializable primitive values."
        ) from exc
    return result


def _config_identity(config: Any) -> dict[str, Any]:
    """Return the architecture fields that must survive checkpoint conversion."""

    fields = (
        "model_type",
        "architectures",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
        "tie_word_embeddings",
    )
    return {
        name: getattr(config, name, None)
        for name in fields
    }


def _expected_llama_projection_names(model: nn.Module) -> tuple[str, ...]:
    """Resolve the exact seven-projection Llama topology, failing closed."""

    body = getattr(model, "model", None)
    layers = getattr(body, "layers", None)
    if not isinstance(layers, (nn.ModuleList, nn.Sequential)) or not layers:
        raise TypeError(
            "Controlled materialisation requires model.model.layers to be a "
            "non-empty ModuleList/Sequential."
        )
    suffixes = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    names: list[str] = []
    for index, layer in enumerate(layers):
        for suffix in suffixes:
            module: nn.Module = layer
            for part in suffix.split("."):
                module = getattr(module, part)
            if not is_efficientqat_packed_linear(module):
                raise TypeError(
                    f"model.layers.{index}.{suffix} is not an EfficientQAT "
                    "packed QuantLinear."
                )
            names.append(f"model.layers.{index}.{suffix}")

    actual = [
        name
        for name, module in model.named_modules()
        if name and is_efficientqat_packed_linear(module)
    ]
    if actual != names:
        raise RuntimeError(
            "Packed-module topology is not exactly the seven Llama "
            f"projections per layer: actual={actual!r}, expected={names!r}."
        )
    return tuple(names)


def _load_upstream_packed_checkpoint(
    packed_checkpoint: Path,
    *,
    wbits: int,
    group_size: int,
):
    """Construct EfficientQAT's packed module graph on CPU and load its state."""

    efficientqat_root = Path(__file__).resolve().parents[2] / "EfficientQAT"
    if not efficientqat_root.is_dir():
        raise FileNotFoundError(
            f"EfficientQAT source checkout is missing: {efficientqat_root}"
        )
    root_string = str(efficientqat_root)
    if root_string not in sys.path:
        # Preserve the caller's workspace ``utils`` package; the upstream
        # checkout is needed only for its otherwise-unique ``quantize`` tree.
        sys.path.append(root_string)

    try:
        from quantize.int_linear_real import (  # type: ignore[import-not-found]
            load_quantized_model,
        )
    except Exception as exc:
        raise RuntimeError(
            "Unable to import EfficientQAT's packed-checkpoint loader."
        ) from exc

    model, tokenizer = load_quantized_model(
        str(packed_checkpoint),
        wbits,
        group_size,
        device_map={"": "cpu"},
    )
    if not isinstance(model, nn.Module):
        raise TypeError("EfficientQAT loader did not return an nn.Module.")
    model.cpu()
    return model, tokenizer


@torch.no_grad()
def materialize_efficientqat_model(
    model: nn.Module,
    *,
    expected_count: int,
    expected_bits: int,
    expected_group_size: int = 128,
    expected_module_names: tuple[str, ...] | list[str] | None = None,
    provenance: Mapping[str, Any] | None = None,
    source_quantizer: str = SOURCE_QUANTIZER,
) -> dict[str, Any]:
    """Replace every packed projection with a standard CPU BF16 ``nn.Linear``.

    All modules are validated and decoded before the first mutation, so a
    malformed layer or count mismatch cannot leave a partially converted
    model.  The returned deterministic manifest is intended to be saved next
    to the ordinary Hugging Face checkpoint and validated again after a fresh
    reload.
    """

    expected_count = _positive_int(expected_count, name="expected_count")
    expected_group_size = _positive_int(
        expected_group_size, name="expected_group_size"
    )
    if expected_bits not in _SUPPORTED_BITS:
        raise ValueError(
            f"expected_bits must be one of {sorted(_SUPPORTED_BITS)}; "
            f"got {expected_bits!r}."
        )
    if source_quantizer not in SUPPORTED_SOURCE_QUANTIZERS:
        raise ValueError(
            "unsupported EfficientQAT source quantizer: "
            f"{source_quantizer!r}"
        )

    packed = [
        (name, module)
        for name, module in model.named_modules()
        if name and is_efficientqat_packed_linear(module)
    ]
    names = [name for name, _ in packed]
    if len(packed) != expected_count:
        raise RuntimeError(
            "EfficientQAT packed-linear count mismatch: "
            f"{len(packed)} != {expected_count}; names={names!r}."
        )
    if len(set(names)) != len(names):
        raise RuntimeError("EfficientQAT packed-linear paths are not unique.")
    if expected_module_names is not None:
        expected_names = list(expected_module_names)
        if names != expected_names:
            raise RuntimeError(
                "EfficientQAT packed-linear module names differ from the "
                f"frozen topology: actual={names!r}, expected={expected_names!r}."
            )

    decoded: list[
        tuple[str, nn.Module, torch.Tensor, torch.Tensor | None]
    ] = []
    for name, module in packed:
        weight = decode_efficientqat_weight_cpu(
            module,
            expected_bits=expected_bits,
            expected_group_size=expected_group_size,
            name=name,
            source_quantizer=source_quantizer,
        )
        bias = getattr(module, "bias", None)
        decoded.append(
            (
                name,
                module,
                weight,
                None
                if bias is None
                else bias.detach().cpu().to(TARGET_DTYPE).contiguous(),
            )
        )

    records: list[dict[str, Any]] = []
    replacements: list[tuple[str, nn.Linear]] = []
    for name, source, weight, bias in decoded:
        replacement = nn.Linear(
            int(source.infeatures),
            int(source.outfeatures),
            bias=bias is not None,
            device="cpu",
            dtype=TARGET_DTYPE,
        )
        replacement.requires_grad_(False)
        replacement.weight.copy_(weight)
        if bias is not None:
            assert replacement.bias is not None
            replacement.bias.copy_(bias)
        replacements.append((name, replacement))
        records.append(
            {
                "name": name,
                "bits": int(source.bits),
                "group_size": int(source.group_size),
                "in_features": int(source.infeatures),
                "out_features": int(source.outfeatures),
                "has_bias": bias is not None,
                "weight_shape": list(replacement.weight.shape),
                "weight_dtype": str(replacement.weight.dtype),
                "weight_sha256": _tensor_sha256(replacement.weight),
                "bias_sha256": (
                    None
                    if replacement.bias is None
                    else _tensor_sha256(replacement.bias)
                ),
            }
        )

    for name, replacement in replacements:
        _set_module(model, name, replacement)

    remaining = [
        name
        for name, module in model.named_modules()
        if name and is_efficientqat_packed_linear(module)
    ]
    if remaining:
        raise RuntimeError(
            "Packed EfficientQAT modules remain after materialisation: "
            f"{remaining!r}."
        )

    manifest = {
        "format": MATERIALIZATION_FORMAT,
        "format_version": MATERIALIZATION_VERSION,
        "source_quantizer": source_quantizer,
        "target_module": "torch.nn.Linear",
        "target_dtype": str(TARGET_DTYPE),
        "module_count": len(records),
        "module_names": names,
        "modules_sha256": _module_digest(records),
        "modules": records,
        "provenance": _jsonable_provenance(provenance),
    }
    validate_materialized_model(
        model,
        manifest,
        expected_count=expected_count,
    )
    return manifest


def _validate_manifest_shape(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("format") != MATERIALIZATION_FORMAT:
        raise ValueError(
            f"Unexpected materialization format {manifest.get('format')!r}."
        )
    if manifest.get("format_version") != MATERIALIZATION_VERSION:
        raise ValueError(
            "Unsupported materialization manifest version "
            f"{manifest.get('format_version')!r}."
        )
    if manifest.get("target_module") != "torch.nn.Linear":
        raise ValueError("Materialization manifest target module is invalid.")
    if manifest.get("target_dtype") != str(TARGET_DTYPE):
        raise ValueError("Materialization manifest target dtype is not BF16.")
    if manifest.get("source_quantizer") not in SUPPORTED_SOURCE_QUANTIZERS:
        raise ValueError(
            "Materialization manifest source quantizer is unsupported."
        )
    records = manifest.get("modules")
    names = manifest.get("module_names")
    if not isinstance(records, list) or not isinstance(names, list):
        raise ValueError(
            "Materialization manifest modules/module_names must be lists."
        )
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("Materialization module records must be mappings.")
    if names != [record.get("name") for record in records]:
        raise ValueError(
            "Materialization manifest module_names disagree with module records."
        )
    if len(names) != len(set(names)):
        raise ValueError("Materialization manifest module names are not unique.")
    if manifest.get("module_count") != len(records):
        raise ValueError("Materialization manifest module_count is inconsistent.")
    if manifest.get("modules_sha256") != _module_digest(records):
        raise ValueError("Materialization manifest module digest is invalid.")
    _jsonable_provenance(manifest.get("provenance", {}))
    return records


@torch.no_grad()
def validate_materialized_model(
    model: nn.Module,
    manifest: Mapping[str, Any],
    *,
    expected_count: int | None = None,
) -> None:
    """Fail closed unless a live/freshly-reloaded model matches its manifest."""

    if not isinstance(manifest, Mapping):
        raise TypeError("Materialization manifest must be a mapping.")
    records = _validate_manifest_shape(manifest)
    if expected_count is not None:
        expected_count = _positive_int(
            expected_count, name="expected_count"
        )
        if len(records) != expected_count:
            raise ValueError(
                f"Materialized module count {len(records)} != "
                f"expected {expected_count}."
            )

    remaining = [
        name
        for name, module in model.named_modules()
        if name and is_efficientqat_packed_linear(module)
    ]
    if remaining:
        raise RuntimeError(
            f"Model still contains packed EfficientQAT modules: {remaining!r}."
        )

    actual_records: list[dict[str, Any]] = []
    for record in records:
        name = record["name"]
        module = _get_module(model, name)
        if not isinstance(module, nn.Linear):
            raise TypeError(
                f"Materialized path {name!r} is "
                f"{type(module).__qualname__}, not nn.Linear."
            )
        if module.weight.device.type != "cpu":
            raise ValueError(
                f"Materialized path {name!r} is on {module.weight.device}; "
                "the save/reload gate must run on CPU."
            )
        if module.weight.dtype != TARGET_DTYPE:
            raise TypeError(
                f"Materialized path {name!r} has dtype {module.weight.dtype}; "
                f"expected {TARGET_DTYPE}."
            )
        if list(module.weight.shape) != record.get("weight_shape"):
            raise ValueError(
                f"Materialized path {name!r} weight shape changed."
            )
        if _tensor_sha256(module.weight) != record.get("weight_sha256"):
            raise ValueError(
                f"Materialized path {name!r} weight hash changed."
            )
        has_bias = module.bias is not None
        if has_bias is not bool(record.get("has_bias")):
            raise ValueError(
                f"Materialized path {name!r} bias topology changed."
            )
        bias_sha = None if module.bias is None else _tensor_sha256(module.bias)
        if bias_sha != record.get("bias_sha256"):
            raise ValueError(
                f"Materialized path {name!r} bias hash changed."
            )
        actual_records.append(
            {
                "name": name,
                "weight_sha256": _tensor_sha256(module.weight),
                "bias_sha256": bias_sha,
            }
        )

    if _module_digest(actual_records) != manifest.get("modules_sha256"):
        raise ValueError("Live materialized-model digest differs from manifest.")


def write_materialization_manifest(
    path: str | Path,
    manifest: Mapping[str, Any],
) -> None:
    """Atomically persist a validated primitive-only manifest."""

    _validate_manifest_shape(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                dict(manifest),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def load_materialization_manifest(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a materialization manifest."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Materialization manifest not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("Materialization manifest root must be an object.")
    _validate_manifest_shape(manifest)
    return manifest


@torch.no_grad()
def materialize_packed_checkpoint(
    *,
    packed_checkpoint: str | Path,
    output_dir: str | Path,
    base_model: str | Path,
    wbits: int,
    group_size: int,
    output_dtype: str = "bfloat16",
    source_quantizer: str = SOURCE_QUANTIZER,
) -> dict[str, Any]:
    """Convert a saved EfficientQAT checkpoint into an ordinary BF16 HF model.

    The source graph is constructed with EfficientQAT's own checkpoint loader,
    but all unpacking is performed by this module on CPU.  The destination is
    built in a temporary sibling directory, freshly reloaded through
    ``AutoModelForCausalLM``, checked against the materialisation manifest, and
    only then atomically renamed into place.
    """

    if output_dtype != "bfloat16":
        raise ValueError(
            "The controlled evaluator requires output_dtype='bfloat16'; "
            f"got {output_dtype!r}."
        )
    if wbits not in _SUPPORTED_BITS:
        raise ValueError(
            f"wbits must be one of {sorted(_SUPPORTED_BITS)}; got {wbits!r}."
        )
    group_size = _positive_int(group_size, name="group_size")

    source = Path(packed_checkpoint).resolve()
    base = Path(base_model).resolve()
    destination = Path(output_dir).resolve()
    if not source.is_dir() or not (source / "config.json").is_file():
        raise FileNotFoundError(
            f"Packed EfficientQAT checkpoint is incomplete: {source}"
        )
    if not base.is_dir() or not (base / "config.json").is_file():
        raise FileNotFoundError(f"Base model checkpoint is incomplete: {base}")
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite materialized checkpoint: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoConfig, AutoModelForCausalLM

    base_config = AutoConfig.from_pretrained(base, trust_remote_code=True)
    packed_config = AutoConfig.from_pretrained(source, trust_remote_code=True)
    base_identity = _config_identity(base_config)
    packed_identity = _config_identity(packed_config)
    if packed_identity != base_identity:
        raise ValueError(
            "Packed EfficientQAT/base-model architecture identity differs: "
            f"packed={packed_identity!r}, base={base_identity!r}."
        )

    model, tokenizer = _load_upstream_packed_checkpoint(
        source,
        wbits=wbits,
        group_size=group_size,
    )
    if _config_identity(model.config) != base_identity:
        raise ValueError(
            "Loaded packed model architecture differs from the frozen base model."
        )
    expected_names = _expected_llama_projection_names(model)
    manifest = materialize_efficientqat_model(
        model,
        expected_count=len(expected_names),
        expected_bits=wbits,
        expected_group_size=group_size,
        expected_module_names=expected_names,
        provenance={
            "packed_checkpoint": str(source),
            "base_model": str(base),
            "architecture": base_identity,
            "weight_bits": wbits,
            "weight_group_size": group_size,
            "output_dtype": output_dtype,
        },
        source_quantizer=source_quantizer,
    )
    model.eval()
    model.config.torch_dtype = TARGET_DTYPE

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
    )
    manifest_name = "efficientqat_materialization.json"
    try:
        model.save_pretrained(
            temporary,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        tokenizer.save_pretrained(temporary)
        write_materialization_manifest(temporary / manifest_name, manifest)

        reloaded = AutoModelForCausalLM.from_pretrained(
            temporary,
            dtype=TARGET_DTYPE,
            device_map={"": "cpu"},
            trust_remote_code=True,
        )
        reloaded.eval()
        if _config_identity(reloaded.config) != base_identity:
            raise ValueError(
                "Freshly reloaded materialized model changed architecture identity."
            )
        validate_materialized_model(
            reloaded,
            manifest,
            expected_count=len(expected_names),
        )
        del reloaded
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    manifest_path = destination / manifest_name
    return {
        "format": MATERIALIZATION_FORMAT,
        "format_version": MATERIALIZATION_VERSION,
        "source_quantizer": source_quantizer,
        "packed_checkpoint": str(source),
        "base_model": str(base),
        "output_dir": str(destination),
        "output_dtype": output_dtype,
        "weight_bits": wbits,
        "weight_group_size": group_size,
        "module_count": len(expected_names),
        "modules_sha256": manifest["modules_sha256"],
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
    }


__all__ = [
    "MATERIALIZATION_FORMAT",
    "MATERIALIZATION_VERSION",
    "SOURCE_QUANTIZER",
    "SUPPORTED_SOURCE_QUANTIZERS",
    "SYMMETRIC_SOURCE_QUANTIZER",
    "TARGET_DTYPE",
    "decode_efficientqat_weight_cpu",
    "is_efficientqat_packed_linear",
    "load_materialization_manifest",
    "materialize_efficientqat_model",
    "materialize_packed_checkpoint",
    "unpack_packed_dim0",
    "unpack_packed_dim1",
    "validate_materialized_model",
    "write_materialization_manifest",
]
