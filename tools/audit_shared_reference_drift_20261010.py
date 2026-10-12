"""Read-only cross-campaign provenance check; not an A/B acceptance audit."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def options(command):
    result = {}
    for index, token in enumerate(command):
        if token.startswith("--"):
            if "=" in token:
                name, value = token.split("=", 1)
                result[name] = value
                continue
            result[token] = command[index+1] if index+1 < len(command) and not command[index+1].startswith("--") else True
    return result

def main():
    older = ROOT / "outputs/full_pipeline_profile_qwen06b-full-offload-dedup-2x1024-20261010-r1-control"
    newer = ROOT / "outputs/full_pipeline_profile_qwen06b-checkpoint-ce-full-2x1024-threepairs-20261010-restart1408-r1-control"
    manifests = [json.loads((directory / "manifest.json").read_text()) for directory in (older, newer)]
    result = {"scope": "cross-configuration provenance only; neither an exactness nor memory acceptance test"}
    for stage in ("stage0", "stage2"):
        left, right = (options(manifest[f"{stage}_command"]) for manifest in manifests)
        result[f"{stage}_option_differences"] = {key: [left.get(key), right.get(key)] for key in sorted(left.keys() | right.keys())
                                                  if left.get(key) != right.get(key)}
    stats = []
    for directory in (older, newer):
        files = list((directory / "static_cache").glob("*.pt"))
        if len(files) != 1:
            raise RuntimeError("expected one statistics file")
        stats.append({"name": files[0].name, "bytes": files[0].stat().st_size,
                      "sha256": hashlib.sha256(files[0].read_bytes()).hexdigest()})
    result["statistics_files"] = stats
    result["statistics_file_bytes_exact"] = stats[0] == stats[1]
    path = ROOT / "outputs/reference-drift-provenance-20261010-1423.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
