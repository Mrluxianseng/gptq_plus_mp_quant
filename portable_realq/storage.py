"""Atomic manifests, streaming identities and persistent interprocess locks."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterator


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def identity(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: Path, value, *, replace: bool = False) -> None:
    """Publish complete JSON. Immutable evidence uses link(), never replace()."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # fails if evidence already exists
    finally:
        Path(temporary).unlink(missing_ok=True)


def file_manifest(root: Path, *, exclude: tuple[str, ...] = ()) -> dict:
    root = Path(root).resolve(strict=True)
    rows = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if any(relative == part or relative.startswith(part + "/") for part in exclude):
            continue
        if path.is_file():
            # Hub snapshot symlinks are fine, but identities always hash actual bytes.
            rows[relative] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    return rows


def verify_files(root: Path, rows: dict) -> None:
    root = Path(root).resolve(strict=True)
    if not rows:
        raise RuntimeError("An empty artifact is not complete")
    for relative, expected in rows.items():
        part = Path(relative)
        if part.is_absolute() or ".." in part.parts or "\\" in relative:
            raise RuntimeError("Unsafe artifact member: " + relative)
        path = root / part
        if not path.is_file() or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise RuntimeError("Artifact missing or changed: " + str(path))


@contextmanager
def exclusive_lock(path: Path, *, timeout: float = 0) -> Iterator[None]:
    """Kernel-backed lock releases on crash; do not delete the lock file."""
    from filelock import FileLock
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path), timeout=timeout):
        yield
