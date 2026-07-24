#!/usr/bin/env python3
"""Canonical raw-byte manifests and comparisons for REAL-Q checkpoints.

The checkpoint archive produced by ``torch.save`` is not itself a stable
correctness oracle: archive metadata can differ even when every model tensor is
identical.  This tool canonicalizes the harmless ``ActQuantWrapper.module``
alias, then hashes every tensor's canonical key, dtype, shape, and raw bytes.

It intentionally never initializes CUDA and always loads checkpoints onto CPU.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import struct
from typing import Any

import torch


_DIGEST_ALGORITHM = "REALQ_CANONICAL_STATE_V1"
_PATH_CONFIG_FIELDS = {
    "cache_dir",
    "exp",
    "output_dir",
    "save_qmodel_path",
    "static_cache_path",
    "tokens_cache_path",
}
_CHECKPOINT_METADATA_FIELDS = (
    "format",
    "format_version",
    "base_model",
    "runtime_quantization",
    "weight_quantization",
    "artifact_identity",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_field(digest: Any, raw: bytes | memoryview) -> None:
    digest.update(struct.pack("<Q", len(raw)))
    digest.update(raw)


def _checkpoint_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint {path} is not a mapping")
    state = payload.get("model")
    if not isinstance(state, dict):
        raise RuntimeError(f"checkpoint {path} has no mapping-valued 'model'")
    return payload


def _canonical_state_dict(
    state: dict[str, Any],
) -> dict[str, tuple[str, torch.Tensor]]:
    """Fold a wrapper alias only when its plain peer exists and matches.

    ``ActQuantWrapper`` serializes both ``x.weight`` and
    ``x.module.weight``.  A module can also legitimately have ``module`` in
    its name without being such an alias, so a unique ``.module.`` key must
    retain its original identity.
    """

    def byte_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        left_bytes = (
            left.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        )
        right_bytes = (
            right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        )
        return torch.equal(left_bytes, right_bytes)

    validated: dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        if not isinstance(raw_key, str):
            raise RuntimeError(f"non-string state_dict key: {raw_key!r}")
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(
                f"state_dict value for {raw_key!r} is not a tensor"
            )
        validated[raw_key] = value

    aliases_by_plain: dict[str, list[str]] = {}
    for wrapped_key, wrapped_value in validated.items():
        if ".module." not in wrapped_key:
            continue
        plain_key = wrapped_key.replace(".module.", ".")
        if plain_key not in validated:
            continue
        if not byte_equal(validated[plain_key], wrapped_value):
            raise RuntimeError(
                "state_dict canonicalization collision with unequal values: "
                f"{plain_key!r} and {wrapped_key!r} both map to "
                f"{plain_key!r}"
            )
        aliases_by_plain.setdefault(plain_key, []).append(wrapped_key)

    wrapped_aliases = {
        wrapped
        for wrapped_keys in aliases_by_plain.values()
        for wrapped in wrapped_keys
    }
    canonical: dict[str, tuple[str, torch.Tensor]] = {}
    for raw_key, value in validated.items():
        if raw_key in wrapped_aliases:
            continue
        wrapped_keys = aliases_by_plain.get(raw_key)
        serialized_key = (
            sorted(wrapped_keys)[0] if wrapped_keys else raw_key
        )
        canonical[raw_key] = (serialized_key, value)
    return canonical


def build_state_manifest(path: str | Path) -> dict[str, Any]:
    """Build a complete canonical per-tensor raw-byte manifest."""

    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    payload = _checkpoint_payload(checkpoint)
    state = _canonical_state_dict(payload["model"])
    state_digest = hashlib.sha256(
        (_DIGEST_ALGORITHM + "\0").encode("ascii")
    )
    tensors: dict[str, dict[str, Any]] = {}
    for canonical_key in sorted(state):
        raw_key, tensor = state[canonical_key]
        cpu = tensor.detach().cpu().contiguous()
        byte_array = cpu.reshape(-1).view(torch.uint8).numpy()
        raw = memoryview(byte_array)
        dtype = str(cpu.dtype)
        shape = list(cpu.shape)
        tensor_digest = hashlib.sha256(raw).hexdigest()
        _update_field(state_digest, canonical_key.encode("utf-8"))
        _update_field(state_digest, dtype.encode("ascii"))
        _update_field(
            state_digest,
            json.dumps(shape, separators=(",", ":")).encode("ascii"),
        )
        _update_field(state_digest, raw)
        tensors[canonical_key] = {
            "serialized_key": raw_key,
            "dtype": dtype,
            "shape": shape,
            "numel": cpu.numel(),
            "nbytes": raw.nbytes,
            "sha256": tensor_digest,
        }

    metadata = {
        key: payload.get(key)
        for key in _CHECKPOINT_METADATA_FIELDS
        if key in payload
    }
    manifest = {
        "schema_version": 1,
        "digest_algorithm": _DIGEST_ALGORITHM,
        "checkpoint_path": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_archive_sha256": _file_sha256(checkpoint),
        "canonical_state_sha256": state_digest.hexdigest(),
        "canonical_tensor_count": len(tensors),
        "checkpoint_metadata": metadata,
        "tensors": tensors,
    }
    del state
    del payload
    gc.collect()
    return manifest


def _resolve_checkpoint(raw: str | Path) -> tuple[Path, Path | None]:
    path = Path(raw).expanduser().resolve()
    if path.is_file():
        return path, None
    if not path.is_dir():
        raise FileNotFoundError(f"path does not exist: {path}")
    candidates = (
        path / "checkpoint" / "model.pt",
        path / "model.pt",
    )
    found = [candidate for candidate in candidates if candidate.is_file()]
    if len(found) != 1:
        rendered = ", ".join(str(item) for item in candidates)
        raise RuntimeError(
            f"expected exactly one checkpoint below {path}; checked {rendered}"
        )
    return found[0].resolve(), path


def _load_run_context(root: Path | None) -> dict[str, Any] | None:
    if root is None:
        return None
    manifest_path = root / "manifest.json"
    config_path = root / "resolved_config.json"
    if not manifest_path.is_file() or not config_path.is_file():
        return None
    return {
        "root": str(root),
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        "config": json.loads(config_path.read_text(encoding="utf-8")),
    }


def _candidate_names(manifest: dict[str, Any]) -> set[str]:
    names = set()
    for spec in manifest.get("candidate_arg_specs", []):
        if not isinstance(spec, str) or "=" not in spec:
            raise RuntimeError(f"invalid candidate arg in manifest: {spec!r}")
        names.add(spec.split("=", 1)[0])
    return names


def _run_compatibility(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> dict[str, Any]:
    if left is None or right is None:
        return {
            "available": False,
            "passed": None,
            "reason": (
                "pass two correctness run roots, rather than bare checkpoint "
                "files, to enforce A/B provenance compatibility"
            ),
        }
    left_manifest = left["manifest"]
    right_manifest = right["manifest"]
    left_config = left["config"]
    right_config = right["config"]

    def equal_present(field: str) -> bool:
        missing = object()
        left_value = left_manifest.get(field, missing)
        right_value = right_manifest.get(field, missing)
        return (
            left_value is not missing
            and right_value is not missing
            and left_value is not None
            and right_value is not None
            and left_value == right_value
        )

    declared = _candidate_names(left_manifest) | _candidate_names(right_manifest)
    actual = {
        key: {"left": left_config.get(key), "right": right_config.get(key)}
        for key in sorted(set(left_config) | set(right_config))
        if key not in _PATH_CONFIG_FIELDS
        and left_config.get(key) != right_config.get(key)
    }
    identity_checks = {
        "baseline_id_equal": equal_present("baseline_id"),
        "case_equal": equal_present("case"),
        "git_commit_equal": equal_present("git_commit"),
        "physical_gpu_ids_equal": equal_present("physical_gpu_ids"),
        "physical_gpu_uuids_equal": equal_present(
            "physical_gpu_uuids_in_rank_order"
        ),
        "world_size_equal": equal_present("world_size"),
        "model_artifact_identity_equal": equal_present(
            "model_artifact_identity"
        ),
        "harness_sha256_equal": equal_present("harness_sha256"),
        "checkpoint_compare_tool_sha256_equal": equal_present(
            "checkpoint_compare_tool_sha256"
        ),
        "source_cache_identity_equal": equal_present(
            "source_cache_identity"
        ),
        "both_runs_passed": (
            left_manifest.get("status") == "passed"
            and right_manifest.get("status") == "passed"
        ),
        "resolved_config_diff_is_exactly_declared_candidates": (
            set(actual) == declared
        ),
    }
    return {
        "available": True,
        "passed": all(identity_checks.values()),
        "ignored_artifact_path_fields": sorted(_PATH_CONFIG_FIELDS),
        "declared_candidate_fields": sorted(declared),
        "actual_resolved_config_differences": actual,
        "checks": identity_checks,
        "left_run_root": left["root"],
        "right_run_root": right["root"],
    }


def compare_checkpoints(
    left_raw: str | Path,
    right_raw: str | Path,
) -> dict[str, Any]:
    """Strictly compare canonical key/dtype/shape/raw bytes per tensor."""

    left_path, left_root = _resolve_checkpoint(left_raw)
    right_path, right_root = _resolve_checkpoint(right_raw)
    left = build_state_manifest(left_path)
    right = build_state_manifest(right_path)
    left_tensors = left["tensors"]
    right_tensors = right["tensors"]
    common = sorted(set(left_tensors) & set(right_tensors))
    missing = sorted(set(left_tensors) - set(right_tensors))
    extra = sorted(set(right_tensors) - set(left_tensors))
    mismatched: list[dict[str, Any]] = []
    for key in common:
        a = left_tensors[key]
        b = right_tensors[key]
        differences = {
            field: {"left": a.get(field), "right": b.get(field)}
            for field in ("dtype", "shape", "nbytes", "sha256")
            if a.get(field) != b.get(field)
        }
        if differences:
            mismatched.append({"key": key, "differences": differences})

    tensor_bytes_equal = (
        not missing
        and not extra
        and not mismatched
        and left["canonical_state_sha256"]
        == right["canonical_state_sha256"]
    )
    compatibility = _run_compatibility(
        _load_run_context(left_root),
        _load_run_context(right_root),
    )
    metadata_differences = {
        key: {
            "left": left["checkpoint_metadata"].get(key),
            "right": right["checkpoint_metadata"].get(key),
        }
        for key in sorted(
            set(left["checkpoint_metadata"])
            | set(right["checkpoint_metadata"])
        )
        if left["checkpoint_metadata"].get(key)
        != right["checkpoint_metadata"].get(key)
    }
    passed = tensor_bytes_equal and (
        compatibility["passed"]
        if compatibility["available"]
        else True
    )
    return {
        "schema_version": 1,
        "comparison": "canonical_per_tensor_raw_bytes",
        "archive_sha_equal_not_required": True,
        "left_checkpoint": str(left_path),
        "right_checkpoint": str(right_path),
        "left_archive_sha256": left["checkpoint_archive_sha256"],
        "right_archive_sha256": right["checkpoint_archive_sha256"],
        "left_canonical_state_sha256": left["canonical_state_sha256"],
        "right_canonical_state_sha256": right["canonical_state_sha256"],
        "canonical_state_sha256_equal": (
            left["canonical_state_sha256"]
            == right["canonical_state_sha256"]
        ),
        "compared_tensor_count": len(common),
        "missing_tensor_keys": missing,
        "extra_tensor_keys": extra,
        "mismatched_tensors": mismatched,
        "all_tensor_bytes_equal": tensor_bytes_equal,
        "checkpoint_metadata_equal": not metadata_differences,
        "checkpoint_metadata_differences_audit_only": metadata_differences,
        "run_compatibility": compatibility,
        "passed": passed,
    }


def _write_json(path: str | None, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path is None or path == "-":
        print(rendered, end="")
        return
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    state = subparsers.add_parser(
        "state", help="write one checkpoint's canonical tensor manifest"
    )
    state.add_argument("checkpoint")
    state.add_argument("-o", "--output")
    compare = subparsers.add_parser(
        "compare",
        help=(
            "strictly compare two checkpoints or two correctness run roots"
        ),
    )
    compare.add_argument("left")
    compare.add_argument("right")
    compare.add_argument("-o", "--output")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "state":
        payload = build_state_manifest(args.checkpoint)
    else:
        payload = compare_checkpoints(args.left, args.right)
    _write_json(args.output, payload)
    return 0 if payload.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
