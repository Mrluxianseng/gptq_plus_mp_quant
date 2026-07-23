"""Cheap, mutation-sensitive identities for model and rotation artifacts."""

from __future__ import annotations

import hashlib
import os


_CONTENT_HASH_LIMIT_BYTES = 64 * 1024 * 1024
_WEIGHT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".pt",
    ".pth",
    ".safetensors",
}


def _update_file_identity(hasher, path: str, logical_name: str) -> None:
    """Add path/stat identity and, for small metadata, exact contents."""
    resolved = os.path.realpath(path)
    stat = os.stat(resolved)
    hasher.update(
        (
            f"file={logical_name}|real={resolved}|size={stat.st_size}|"
            f"mtime_ns={stat.st_mtime_ns}|"
        ).encode()
    )
    suffix = os.path.splitext(resolved)[1].lower()
    if suffix not in _WEIGHT_SUFFIXES and stat.st_size <= _CONTENT_HASH_LIMIT_BYTES:
        with open(resolved, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)


def artifact_identity(value: str | os.PathLike | None) -> str:
    """Return a stable digest for a local artifact or exact external id.

    Reading every byte of a multi-billion-parameter checkpoint merely to look
    up a cache would be prohibitive. Weight-like files therefore contribute
    resolved path, size and nanosecond mtime; ordinary metadata/config/tokenizer
    files up to 64 MiB additionally contribute exact contents.
    """
    if value is None:
        return "none"
    raw = os.fspath(value)
    if not os.path.exists(raw):
        return hashlib.sha256(f"external:{raw}".encode()).hexdigest()

    resolved = os.path.realpath(raw)
    hasher = hashlib.sha256()
    hasher.update(f"root={resolved}|".encode())
    if os.path.isfile(resolved):
        _update_file_identity(hasher, resolved, os.path.basename(resolved))
    elif os.path.isdir(resolved):
        for root, dirnames, filenames in os.walk(resolved):
            dirnames.sort()
            filenames.sort()
            for filename in filenames:
                path = os.path.join(root, filename)
                if os.path.isfile(path):
                    _update_file_identity(
                        hasher, path, os.path.relpath(path, resolved)
                    )
    else:
        stat = os.stat(resolved)
        hasher.update(
            f"special={resolved}|mode={stat.st_mode}|mtime_ns={stat.st_mtime_ns}".encode()
        )
    return hasher.hexdigest()


def artifact_cache_tag(
    value: str | os.PathLike | None, *, length: int = 12
) -> str:
    """Filename-safe prefix of :func:`artifact_identity`."""
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}.")
    return artifact_identity(value)[:length]
