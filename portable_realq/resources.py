"""Prepare once, persist resolved paths, and resume HF downloads on the next call.

No model weights are loaded into RAM here. The Hub cache is the ONLY downloaded
model copy; experiment directories contain identities and paths, never copies.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

from .storage import atomic_json, digest, exclusive_lock, file_manifest, identity, read_json, verify_files

MODELS = {
    "q06": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", 1024),
    "q17": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", 2048),
}
TASKS = ("piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande",
         "lambada_openai", "ceval-valid", "boolq", "openbookqa", "social_iqa")


def environment(root: Path) -> dict[str, str]:
    root = Path(root).resolve()
    return {"HF_HOME": str(root / "hf"), "HF_HUB_CACHE": str(root / "hf/hub"),
            "HF_DATASETS_CACHE": str(root / "hf/datasets"),
            "HF_DATASETS_TRUST_REMOTE_CODE": "1"}


def validate_model(path: Path, model: str) -> None:
    cfg = read_json(path / "config.json")
    if cfg.get("model_type") != "qwen3" or cfg.get("num_hidden_layers") != 28 or cfg.get("hidden_size") != MODELS[model][2]:
        raise RuntimeError("Unexpected model architecture: " + model)
    if not (path / "tokenizer.json").is_file() or not (path / "tokenizer_config.json").is_file():
        raise RuntimeError("Tokenizer files missing")
    read_json(path / "tokenizer.json")
    read_json(path / "tokenizer_config.json")
    index = path / "model.safetensors.index.json"
    shards = set(read_json(index)["weight_map"].values()) if index.is_file() else {"model.safetensors"}
    for name in shards:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not (path / relative).is_file() or (path / relative).stat().st_size == 0:
            raise RuntimeError("Missing/invalid model weight shard: " + name)
        from safetensors import safe_open
        with safe_open(str(path / relative), framework="pt", device="cpu") as handle:
            if not list(handle.keys()):
                raise RuntimeError("Empty safetensors model shard")


def validate_wikitext(path: Path) -> dict[str, list[str]]:
    rows = {}
    for split in ("train", "validation", "test"):
        files = sorted((path / "wikitext-2-raw-v1").glob(split + "-*.parquet"))
        if not files or any(p.stat().st_size == 0 for p in files):
            raise RuntimeError("WikiText-2 split missing: " + split)
        import pyarrow.parquet as parquet
        for file in files:
            metadata = parquet.ParquetFile(file)
            if metadata.metadata.num_rows <= 0 or "text" not in metadata.schema.names:
                raise RuntimeError("Invalid WikiText parquet schema: " + str(file))
        rows[split] = [str(p.resolve()) for p in files]
    return rows


def resolve(root: Path, name: str, *, local: Path | None = None, hub=None) -> dict:
    """A failed call leaves the registry AND Hub .incomplete chunks intact.

    Hub snapshot_download resumes incomplete files itself. Never use local_dir
    (which can introduce another payload copy), force_download, or shutil.copy.
    A supplied incomplete local path falls back to the registered Hub source.
    """
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    registry = root / "resources.json"
    with exclusive_lock(root / "locks/download.lock"):
        records = read_json(registry) if registry.exists() else {}
        old = records.get(name, {})
        same_request = local is None or str(local.resolve()) in (old.get("path"), old.get("requested_local_path"))
        if old.get("state") == "ready" and same_request:
            try:
                path = Path(old["path"])
                verify_files(path, old["files"])
                validate_model(path, name) if name in MODELS else validate_wikitext(path)
                return old
            except (OSError, ValueError, RuntimeError, KeyError):
                # Record a failed validation; do not silently use broken local data.
                pass
        if name in MODELS:
            repo_id, revision, _ = MODELS[name]
            kind = "model"
            patterns = ["*.json", "*.safetensors", "*.txt", "*.model", "LICENSE"]
        elif name == "wikitext2":
            repo_id, kind = "Salesforce/wikitext", "dataset"
            revision = old.get("revision")
            patterns = ["wikitext-2-raw-v1/*.parquet", "README.md"]
        else:
            raise ValueError("Unregistered resource: " + name)
        record = {"state": "resolving", "repo_id": repo_id, "repo_type": kind,
                  "revision": revision, "cache_dir": str(root / "hf/hub"),
                  "requested_local_path": str(local.resolve()) if local else None,
                  "updated_utc": datetime.now(timezone.utc).isoformat()}
        records[name] = record
        atomic_json(registry, records, replace=True)  # persisted BEFORE network I/O
        try:
            path = None
            if local is not None:
                try:
                    validate_model(local, name) if name in MODELS else validate_wikitext(local)
                    path = local.resolve(strict=True)
                    record["origin"] = "explicit-local"
                except (OSError, ValueError, RuntimeError, KeyError):
                    record["local_validation_failed"] = True
            if path is None:
                if hub is None:
                    import huggingface_hub as hub
                if revision is None:
                    # Pin a dataset commit before downloading, and retain it on failure.
                    info = hub.HfApi().dataset_info(repo_id)
                    revision = info.sha
                if not re.fullmatch(r"[0-9a-f]{40}", revision):
                    raise RuntimeError("An immutable Hub commit is required")
                record.update(state="downloading", revision=revision)
                atomic_json(registry, records, replace=True)
                path = Path(hub.snapshot_download(repo_id=repo_id, repo_type=kind,
                    revision=revision, cache_dir=str(root / "hf/hub"),
                    allow_patterns=patterns, max_workers=2))
                record["origin"] = "hub-cache"
                info = hub.HfApi().repo_info(repo_id=repo_id, repo_type=kind,
                                             revision=revision, files_metadata=True)
                if info.sha != revision:
                    raise RuntimeError("Downloaded revision differs from the pinned Hub commit")
                expected = {row.rfilename: row for row in info.siblings}
                for file in path.rglob("*"):
                    if not file.is_file(): continue
                    relative = file.relative_to(path).as_posix()
                    if relative.startswith(".cache/"): continue
                    row = expected.get(relative)
                    if row is None or file.stat().st_size != row.size:
                        raise RuntimeError("Hub resource file/length differs: " + relative)
                    if row.lfs:
                        sha256 = row.lfs["sha256"] if isinstance(row.lfs, dict) else row.lfs.sha256
                        if digest(file) != sha256:
                            raise RuntimeError("Hub LFS SHA256 differs: " + relative)
                    else:
                        import hashlib
                        git = hashlib.sha1(f"blob {file.stat().st_size}\0".encode())
                        with file.open("rb") as stream:
                            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                                git.update(block)
                        if git.hexdigest() != row.blob_id:
                            raise RuntimeError("Hub Git blob identity differs: " + relative)
            validate_model(path, name) if name in MODELS else validate_wikitext(path)
            files = file_manifest(path, exclude=(".cache",))
            record.update(state="ready", path=str(path.resolve()), files=files,
                          content_sha256=identity(files))
            atomic_json(registry, records, replace=True)
            return record
        except BaseException as exc:
            # Do not persist signed download URLs, credentials, or bearer tokens.
            record.update(state="incomplete", error_type=type(exc).__name__)
            atomic_json(registry, records, replace=True)
            raise


def prepare_qa(root: Path) -> dict:
    """Resolve the complete ten tasks before leasing GPU time; shared HF cache."""
    os.environ.update(environment(root))
    import lm_eval.tasks
    manager = lm_eval.tasks.TaskManager(include_defaults=True)
    tasks = lm_eval.tasks.get_task_dict(list(TASKS), task_manager=manager)
    if set(tasks) != set(TASKS):
        raise RuntimeError("All ten canonical QA tasks must be resolved")
    configs = {}
    for name, task in tasks.items():
        # Record the actual resolved task config, including dataset paths/splits.
        configs[name] = task.config.to_dict()
    return {"tasks": list(TASKS), "configuration": json.loads(json.dumps(configs, default=str)),
            "scope": "native official qa_eval; not historical four-GPU mapped observer acceptance"}
