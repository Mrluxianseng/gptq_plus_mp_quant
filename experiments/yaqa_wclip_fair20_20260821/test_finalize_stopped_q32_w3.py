from __future__ import annotations

from pathlib import Path

import pytest

from experiments.yaqa_wclip_fair20_20260821 import (
    finalize_stopped_q32_w3 as finalizer,
)
from experiments.yaqa_wclip_fair20_20260821 import relocate_q32_w2 as relocation


def _process(*, state, cmdline, start_ticks, ppid):
    return {
        "pid": 1,
        "ppid": ppid,
        "state": state,
        "cmdline": cmdline,
        "start_ticks": start_ticks,
        "fd_targets": [],
    }


def _handoff(parent_pid=101, quantizer_pid=202):
    return {
        "source": {
            "queue_parent": _process(
                state="T (stopped)",
                cmdline="formal worker",
                start_ticks=111,
                ppid=10,
            )
            | {"pid": parent_pid},
            "active_quantizer": _process(
                state="S (sleeping)",
                cmdline="W3 quantizer",
                start_ticks=222,
                ppid=parent_pid,
            )
            | {"pid": quantizer_pid},
        }
    }


def test_frozen_contract_selects_exact_q32_w3_predecessor():
    plan, _source, _w2, w3, _preflight = relocation._load_contract()
    assert w3["stage_id"] == "YQ-Q32-W3"
    assert w3["model"] == "qwen3-32b"
    assert w3["setting"] == "W3A16KV16"
    assert w3["w_bits"] == 3
    assert w3["hessian_id"] == "YH-Q32-A16"
    assert plan["stages"].index(w3) < plan["stages"].index(
        relocation._stage(plan, relocation.STAGE_ID)
    )


def test_process_identity_allows_zombie_cmdline_to_be_empty(monkeypatch):
    parent_pid, quantizer_pid = 101, 202
    handoff = _handoff(parent_pid, quantizer_pid)
    parent = handoff["source"]["queue_parent"]
    zombie = _process(
        state="Z (zombie)", cmdline="", start_ticks=222, ppid=parent_pid
    ) | {"pid": quantizer_pid}

    def lookup(pid):
        return parent if pid == parent_pid else zombie

    monkeypatch.setattr(relocation, "_process", lookup)
    observed_parent, observed_quantizer = finalizer._validate_processes(
        handoff, parent_pid=parent_pid, quantizer_pid=quantizer_pid
    )
    assert observed_parent == parent
    assert observed_quantizer == zombie


def test_process_identity_still_rejects_running_cmdline_change(monkeypatch):
    parent_pid, quantizer_pid = 101, 202
    handoff = _handoff(parent_pid, quantizer_pid)
    parent = handoff["source"]["queue_parent"]
    changed = _process(
        state="R (running)",
        cmdline="different process",
        start_ticks=222,
        ppid=parent_pid,
    ) | {"pid": quantizer_pid}
    monkeypatch.setattr(
        relocation,
        "_process",
        lambda pid: parent if pid == parent_pid else changed,
    )
    with pytest.raises(finalizer.FinalizationError, match="identity changed"):
        finalizer._validate_processes(
            handoff, parent_pid=parent_pid, quantizer_pid=quantizer_pid
        )


def test_wait_parent_retired_accepts_an_inert_zombie(monkeypatch):
    class FakeStat:
        def read_text(self, *, encoding):
            del encoding
            return "123 (worker) Z " + "0 " * 49

    monkeypatch.setattr(finalizer, "Path", lambda value: FakeStat())
    assert finalizer._wait_parent_retired(123) is False


def test_runtime_environment_restores_original_source_gpu(monkeypatch):
    monkeypatch.setattr(
        relocation,
        "_runtime_environment",
        lambda plan: {"CUDA_VISIBLE_DEVICES": str(relocation.TARGET_GPU)},
    )
    assert finalizer._runtime_environment({})["CUDA_VISIBLE_DEVICES"] == str(
        relocation.SOURCE_GPU
    )


def test_retry_launch_artifacts_are_append_only():
    root = Path("/tmp/finalizer-test")
    initial = finalizer._launch_artifacts(root, "initial")
    retry = finalizer._launch_artifacts(root, "retry2")
    assert initial == (
        root / f"{relocation.SOURCE_HOST}_w3_finalizer.log",
        root / "w3_finalizer_launch.json",
    )
    assert retry == (
        root / f"{relocation.SOURCE_HOST}_w3_finalizer_retry2.log",
        root / "w3_finalizer_launch_retry2.json",
    )
    with pytest.raises(finalizer.FinalizationError, match="invalid launch attempt"):
        finalizer._launch_artifacts(root, "../overwrite")
