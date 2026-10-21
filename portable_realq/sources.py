"""Verify frozen v4 bytes and materialize registered ablation source overlays."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from .storage import atomic_json, digest, exclusive_lock, identity, read_json, verify_files

VENDOR = Path(__file__).resolve().parents[1] / "vendor/official_v4"
VARIANTS = ("full_v4", "without_static_checkpoint", "without_CPU_streams",
            "without_solver_reuse_bundle", "without_single_buffer_symmetry",
            "without_early_H_lifetimes", "baseline")
DEFAULT_VARIANTS = VARIANTS[:4]


def verify_vendor() -> dict:
    manifest = read_json(VENDOR / "manifest.json")
    if manifest["official_commit"] != "dbf35ae27b5a2a35d6823cf234b7959fc747c02d" or len(manifest["sources"]) != 239:
        raise RuntimeError("Full pinned 239-file v4 source pair required")
    for side in ("baseline", "storage"):
        rows = {key.split("/", 1)[1]: value for key, value in manifest["sources"].items()
                if key.startswith(side + "/")}
        verify_files(VENDOR / side / "realq", rows)
    return manifest


def materialize(root: Path, variant: str) -> tuple[Path, str]:
    if variant not in VARIANTS:
        raise ValueError("Unregistered ablation")
    manifest = verify_vendor()
    patches = read_json(VENDOR / "ablations.json")
    side = "baseline" if variant == "baseline" else "storage"
    rows = {key.split("/", 1)[1]: row for key, row in manifest["sources"].items() if key.startswith(side + "/")}
    edits = patches.get(variant, {})
    for name, edit in edits.items():
        if rows[name]["sha256"] != edit["parent_sha256"] or hashlib.sha256(edit["content"].encode()).hexdigest() != edit["sha256"]:
            raise RuntimeError("Registered ablation source differs")
        ast.parse(edit["content"])
        rows[name] = {"bytes": len(edit["content"].encode()), "sha256": edit["sha256"]}
    key = identity(rows)
    target = Path(root) / "sources" / key
    with exclusive_lock(Path(root) / "locks/sources.lock"):
        if (target / "manifest.json").exists():
            if read_json(target / "manifest.json") != rows:
                raise RuntimeError("Materialized source identity differs")
            verify_files(target / "realq", rows)
            return target, key
        # Sources are tiny compared to weights. Exclusive publication, resumable
        # preparation after a crash; an existing differing member is never replaced.
        for name, row in rows.items():
            path = target / "realq" / name
            data = edits[name]["content"].encode() if name in edits else (VENDOR / side / "realq" / name).read_bytes()
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if digest(path) != row["sha256"]:
                    raise RuntimeError("Incomplete source has unexpected bytes")
            else:
                with path.open("xb") as stream:
                    stream.write(data)
        atomic_json(target / "manifest.json", rows)
    return target, key
