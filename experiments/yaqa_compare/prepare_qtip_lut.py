#!/usr/bin/env python3
"""Generate once, freeze, and restore YAQA's shared QTIP HYB LUT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import scipy
import torch


TMP_LUT = Path("/tmp/kmeans_9_2.pt")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_bytes_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _load_and_validate(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or tuple(value.shape) != (512, 2)
        or not torch.isfinite(value).all()
    ):
        raise RuntimeError(
            f"{path}: expected finite FP32[512,2], got "
            f"{type(value).__qualname__}, {getattr(value, 'dtype', None)}, "
            f"{getattr(value, 'shape', None)}"
        )
    std = float(value.std(unbiased=False))
    if not 0.96 < std < 0.98:
        raise RuntimeError(f"{path}: unexpected LUT std {std}")
    return value


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        os.chmod(temporary, 0o644)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    generated = False
    if not output.exists():
        # bitshift_codebook owns the first-party generation recipe and writes
        # its result to the fixed /tmp path used by every later quantization.
        TMP_LUT.unlink(missing_ok=True)
        torch.manual_seed(args.seed)
        from lib.codebook.bitshift import bitshift_codebook

        codebook = bitshift_codebook(
            L=16,
            K=2,
            V=2,
            tlut_bits=9,
            decode_mode="quantlut_sym",
        )
        del codebook
        _load_and_validate(TMP_LUT)
        _copy_atomic(TMP_LUT, output)
        generated = True

    tensor = _load_and_validate(output)
    output_sha256 = _sha256_file(output)
    tensor_sha256 = _tensor_bytes_sha256(tensor)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": 1,
            "path": str(output),
            "file_sha256": output_sha256,
            "tensor_sha256": tensor_sha256,
            "shape": [512, 2],
            "dtype": "torch.float32",
            "seed": args.seed,
        }
        mismatch = {
            key: (manifest.get(key), value)
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatch:
            raise RuntimeError(
                f"frozen QTIP LUT manifest mismatch: {mismatch!r}"
            )
    else:
        manifest = {
            "schema_version": 1,
            "path": str(output),
            "file_sha256": output_sha256,
            "tensor_sha256": tensor_sha256,
            "shape": [512, 2],
            "dtype": "torch.float32",
            "std_unbiased_false": float(tensor.std(unbiased=False)),
            "seed": args.seed,
            "torch_version": torch.__version__,
            "scipy_version": scipy.__version__,
            "recipe": {
                "L": 16,
                "V": 2,
                "tlut_bits": 9,
                "decode_mode": "quantlut_sym",
                "source": "YAQA/lib/codebook/bitshift.py",
            },
        }
        _write_json_atomic(manifest_path, manifest)

    if (
        not TMP_LUT.exists()
        or _sha256_file(TMP_LUT) != output_sha256
    ):
        _copy_atomic(output, TMP_LUT)
    # Job containers and the host access this shared workspace under different
    # identities. Keep immutable campaign metadata readable from both sides.
    os.chmod(output, 0o644)
    os.chmod(manifest_path, 0o644)
    _load_and_validate(TMP_LUT)
    if _sha256_file(TMP_LUT) != output_sha256:
        raise RuntimeError("/tmp QTIP LUT does not match frozen artifact")

    print(
        json.dumps(
            {
                "status": "ready",
                "generated": generated,
                "path": str(output),
                "manifest": str(manifest_path),
                "file_sha256": output_sha256,
                "tensor_sha256": tensor_sha256,
                "tmp_path": str(TMP_LUT),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
