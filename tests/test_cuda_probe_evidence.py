from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


# Import by absolute path because this repository does not package ``tools``.
REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPO_ROOT / "tools/run_cuda_probe_evidence.py"
EVIDENCE_SPEC = importlib.util.spec_from_file_location(
    "run_cuda_probe_evidence", EVIDENCE_PATH
)
if EVIDENCE_SPEC is None or EVIDENCE_SPEC.loader is None:
    raise RuntimeError(f"cannot import evidence runner from {EVIDENCE_PATH}")
evidence = importlib.util.module_from_spec(EVIDENCE_SPEC)
EVIDENCE_SPEC.loader.exec_module(evidence)
PROBES = [
    REPO_ROOT / "tools/p03_p04_cuda_probe.py",
    REPO_ROOT / "tools/p06_distributed_cpu_probe.py",
]


@pytest.mark.parametrize("path", PROBES)
def test_correctness_probes_do_not_use_removable_assert(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))


@pytest.mark.parametrize("path", PROBES)
def test_correctness_probes_reject_optimized_python_before_torch_import(path):
    completed = subprocess.run(
        [sys.executable, "-O", str(path)],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert completed.returncode != 0
    assert "refuses optimized Python at startup" in completed.stdout
    # The module-level guard must run before importing optional/heavy runtime
    # dependencies, otherwise this gate could false-pass on a host without
    # the CUDA environment.
    assert "No module named 'torch'" not in completed.stdout


def test_gpu_snapshot_preserves_global_id_uuid_mapping():
    parsed = evidence._parse_gpu_snapshot(
        "\n".join(
            [
                "4, GPU-aaaa, NVIDIA L20C, 183359, 1, 0, 570.00",
                "5, GPU-bbbb, NVIDIA L20C, 183359, 0, 0, 570.00",
            ]
        )
    )
    selected = evidence._validate_selected_idle(parsed, [], [4, 5])
    assert [item["physical_gpu_id"] for item in selected] == [4, 5]
    assert [item["uuid"] for item in selected] == [
        "GPU-aaaa",
        "GPU-bbbb",
    ]


def test_gpu_idle_gate_rejects_selected_process_and_busy_memory():
    parsed = evidence._parse_gpu_snapshot(
        "4, GPU-aaaa, NVIDIA L20C, 183359, 32, 0, 570.00"
    )
    with pytest.raises(evidence.EvidenceFailure, match="not idle"):
        evidence._validate_selected_idle(parsed, [], [4])

    parsed[0]["memory_used_mib"] = 0
    with pytest.raises(evidence.EvidenceFailure, match="compute processes"):
        evidence._validate_selected_idle(
            parsed,
            [{"uuid": "GPU-aaaa", "pid": 7, "used_memory_mib": 1}],
            [4],
        )


def test_uuid_normalization_matches_torch_and_nvidia_forms():
    assert evidence._normalize_uuid(
        "GPU-a2934718-b34d-d8c9-a88a-c342cddcef21"
    ) == evidence._normalize_uuid(
        "a2934718-b34d-d8c9-a88a-c342cddcef21"
    )
