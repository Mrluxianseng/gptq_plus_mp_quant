"""Read-only environment provenance without initializing PyTorch/CUDA."""
import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fields = ["name", "uuid", "driver_version", "memory.total", "compute_cap"]
    text = subprocess.check_output(["nvidia-smi", "--query-gpu="+",".join(fields),
                                    "--format=csv,noheader,nounits"], text=True)
    gpus = [dict(zip(fields, [value.strip() for value in row])) for row in csv.reader(text.splitlines())]
    cpu = Path("/proc/cpuinfo").read_text()
    names = sorted(set(line.split(":",1)[1].strip() for line in cpu.splitlines() if line.startswith("model name")))
    memory = {line.split(":",1)[0]: line.split(":",1)[1].strip() for line in Path("/proc/meminfo").read_text().splitlines()
              if line.startswith(("MemTotal:", "SwapTotal:"))}
    tracked_environment = ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG",
                           "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "NCCL_SHM_DISABLE")
    data = {"scope": "read-only host snapshot; per-worker runtime package versions remain in the experiment",
            "os": platform.platform(), "python": platform.python_version(),
            "cpu_models": names, "logical_cpu_count": os.cpu_count(), "wsl_memory_limits": memory,
            "gpus": gpus, "packages": {name: importlib.metadata.version(name) for name in
                  ("torch", "transformers", "accelerate", "safetensors")},
            "selected_parent_environment": {key: os.environ.get(key) for key in tracked_environment}}
    with args.output.open("x") as stream:
        json.dump(data, stream, indent=2)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
