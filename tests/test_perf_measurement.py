from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from realq.config import Config, parse_cli
from realq.runner import layer_loop


def test_perf_measure_layer_default_is_strict_runtime_noop(monkeypatch, tmp_path):
    """The default loop must retain the direct, uninstrumented layer call."""
    cfg = Config(
        model="fake",
        output_dir=str(tmp_path),
        exp="default_off",
        nsamples=1,
        grad_lr=0.0,
        final_layer_grad_lr=0.0,
        loss_slide_window=False,
    )
    assert cfg.perf_measure_layer is None

    layer = object()
    state = SimpleNamespace(fp_inps=object())
    analyzer = SimpleNamespace(get_layers=lambda: [layer])
    manager = object()
    direct_calls = []

    monkeypatch.setattr(layer_loop.parallel_env, "get_rank", lambda: 0)
    monkeypatch.setattr(layer_loop.parallel_env, "get_world_size", lambda: 1)
    monkeypatch.setattr(layer_loop.parallel_env, "is_main", lambda: True)
    monkeypatch.setattr(layer_loop.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        layer_loop,
        "CpuMasterLayerManager",
        lambda *_args, **_kwargs: manager,
    )
    monkeypatch.setattr(
        layer_loop.streams,
        "capture_layer0_inputs",
        lambda *_args, **_kwargs: state,
    )
    monkeypatch.setattr(
        layer_loop,
        "RefreshTraceWriter",
        lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None),
    )

    def direct_quantize(*args, **kwargs):
        direct_calls.append((args, kwargs))
        return state

    def forbidden(*_args, **_kwargs):
        raise AssertionError("default-off path entered performance instrumentation")

    monkeypatch.setattr(layer_loop, "quantize_one_layer", direct_quantize)
    monkeypatch.setattr(layer_loop, "_measure_quantize_one_layer", forbidden)
    monkeypatch.setattr(layer_loop.parallel_env, "barrier", forbidden)
    monkeypatch.setattr(layer_loop.torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(
        layer_loop.torch.cuda, "reset_peak_memory_stats", forbidden
    )
    monkeypatch.setattr(layer_loop.torch.cuda, "memory_allocated", forbidden)
    monkeypatch.setattr(layer_loop.torch.cuda, "memory_reserved", forbidden)
    monkeypatch.setattr(layer_loop.torch.cuda, "max_memory_allocated", forbidden)
    monkeypatch.setattr(layer_loop.torch.cuda, "max_memory_reserved", forbidden)
    monkeypatch.setattr(layer_loop.torch.cuda, "get_device_properties", forbidden)
    monkeypatch.setattr(layer_loop.time, "perf_counter_ns", forbidden)
    monkeypatch.setattr(layer_loop, "_utc_now_iso", forbidden)
    monkeypatch.setattr(layer_loop, "_atomic_json_dump", forbidden)

    layer_loop.quantize_all_layers(cfg, analyzer, object(), [object()])

    assert len(direct_calls) == 1
    args, kwargs = direct_calls[0]
    assert args[:3] == (cfg, 0, layer)
    assert kwargs["num_layers"] == 1
    assert kwargs["layer_manager"] is manager


def test_enabled_perf_measurement_call_order_and_json_schema(
    monkeypatch, tmp_path
):
    cfg = Config(
        model="fake",
        output_dir=str(tmp_path),
        exp="measured",
        perf_measure_layer=2,
    )
    dev = torch.device("cuda:3")
    events: list[str] = []
    allocated = iter((100, 130))
    reserved = iter((200, 240))
    perf_ticks = iter((1_000, 1_275))
    utc_ticks = iter(
        ("2026-07-24T12:00:00.000000Z", "2026-07-24T12:00:00.000001Z")
    )

    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(layer_loop.parallel_env, "get_rank", lambda: 5)
    monkeypatch.setattr(layer_loop.parallel_env, "get_world_size", lambda: 8)
    monkeypatch.setattr(layer_loop.socket, "gethostname", lambda: "test-node")
    monkeypatch.setattr(layer_loop.os, "getpid", lambda: 4242)

    def properties(device):
        events.append(f"properties:{device}")
        return SimpleNamespace(name="Test GPU", uuid="GPU-test-uuid")

    def barrier():
        events.append("barrier")

    def synchronize(device):
        events.append(f"synchronize:{device}")

    def reset(device):
        events.append(f"reset:{device}")

    def memory_allocated(device):
        events.append(f"allocated:{device}")
        return next(allocated)

    def memory_reserved(device):
        events.append(f"reserved:{device}")
        return next(reserved)

    def max_allocated(device):
        events.append(f"max_allocated:{device}")
        return 175

    def max_reserved(device):
        events.append(f"max_reserved:{device}")
        return 290

    def utc_now():
        events.append("utc")
        return next(utc_ticks)

    def perf_counter_ns():
        events.append("perf")
        return next(perf_ticks)

    original_atomic_dump = layer_loop._atomic_json_dump

    def tracked_atomic_dump(path, payload):
        events.append("write")
        original_atomic_dump(path, payload)

    marker = object()

    def quantize_call():
        events.append("quantize")
        return marker

    monkeypatch.setattr(layer_loop.torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(layer_loop.parallel_env, "barrier", barrier)
    monkeypatch.setattr(layer_loop.torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(layer_loop.torch.cuda, "reset_peak_memory_stats", reset)
    monkeypatch.setattr(layer_loop.torch.cuda, "memory_allocated", memory_allocated)
    monkeypatch.setattr(layer_loop.torch.cuda, "memory_reserved", memory_reserved)
    monkeypatch.setattr(
        layer_loop.torch.cuda, "max_memory_allocated", max_allocated
    )
    monkeypatch.setattr(layer_loop.torch.cuda, "max_memory_reserved", max_reserved)
    monkeypatch.setattr(layer_loop, "_utc_now_iso", utc_now)
    monkeypatch.setattr(layer_loop.time, "perf_counter_ns", perf_counter_ns)
    monkeypatch.setattr(layer_loop, "_atomic_json_dump", tracked_atomic_dump)

    result = layer_loop._measure_quantize_one_layer(
        cfg, 2, dev, quantize_call
    )

    assert result is marker
    assert events == [
        "properties:3",
        "synchronize:cuda:3",
        "barrier",
        "synchronize:cuda:3",
        "reset:cuda:3",
        "allocated:cuda:3",
        "reserved:cuda:3",
        "utc",
        "perf",
        "quantize",
        "synchronize:cuda:3",
        "perf",
        "utc",
        "allocated:cuda:3",
        "reserved:cuda:3",
        "max_allocated:cuda:3",
        "max_reserved:cuda:3",
        "barrier",
        "write",
    ]

    output_path = (
        tmp_path
        / "measured"
        / "perf_measure_layer_2_rank5.json"
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "schema_version",
        "metric_name",
        "output_dir",
        "exp",
        "global_rank",
        "local_rank",
        "world_size",
        "hostname",
        "pid",
        "layer_idx",
        "cuda_device_index",
        "cuda_device_name",
        "cuda_device_uuid",
        "start_utc",
        "end_utc",
        "start_perf_counter_ns",
        "end_perf_counter_ns",
        "elapsed_ns",
        "cuda_start_allocated_bytes",
        "cuda_end_allocated_bytes",
        "cuda_peak_allocated_bytes",
        "cuda_peak_allocated_delta_bytes",
        "cuda_start_reserved_bytes",
        "cuda_end_reserved_bytes",
        "cuda_peak_reserved_bytes",
        "cuda_peak_reserved_delta_bytes",
    }
    assert payload == {
        "schema_version": 1,
        "metric_name": "quant_layer_critical_wall",
        "output_dir": str(tmp_path.resolve()),
        "exp": "measured",
        "global_rank": 5,
        "local_rank": 3,
        "world_size": 8,
        "hostname": "test-node",
        "pid": 4242,
        "layer_idx": 2,
        "cuda_device_index": 3,
        "cuda_device_name": "Test GPU",
        "cuda_device_uuid": "GPU-test-uuid",
        "start_utc": "2026-07-24T12:00:00.000000Z",
        "end_utc": "2026-07-24T12:00:00.000001Z",
        "start_perf_counter_ns": 1_000,
        "end_perf_counter_ns": 1_275,
        "elapsed_ns": 275,
        "cuda_start_allocated_bytes": 100,
        "cuda_end_allocated_bytes": 130,
        "cuda_peak_allocated_bytes": 175,
        "cuda_peak_allocated_delta_bytes": 75,
        "cuda_start_reserved_bytes": 200,
        "cuda_end_reserved_bytes": 240,
        "cuda_peak_reserved_bytes": 290,
        "cuda_peak_reserved_delta_bytes": 90,
    }


def test_perf_measure_layer_cli_and_validation():
    assert parse_cli(["--perf_measure_layer", "7"]).perf_measure_layer == 7
    with pytest.raises(ValueError, match="non-negative integer"):
        Config(perf_measure_layer=-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        Config(perf_measure_layer=True)
