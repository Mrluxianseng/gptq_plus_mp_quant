"""CPU-only fault and contract tests for the NEW portable execution layer."""
from __future__ import annotations

from dataclasses import asdict, fields
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from portable_realq import campaign, protocol, resources, sources
from portable_realq.storage import atomic_json, digest, exclusive_lock, file_manifest, identity, read_json, verify_files


def tiny_model(path, model="q06"):
    from safetensors.torch import save_file
    path.mkdir(parents=True, exist_ok=True)
    atomic_json(path / "config.json", {"model_type": "qwen3", "num_hidden_layers": 28,
                                      "hidden_size": resources.MODELS[model][2]})
    atomic_json(path / "tokenizer.json", {})
    atomic_json(path / "tokenizer_config.json", {})
    save_file({"weight": torch.arange(4)}, str(path / "model.safetensors"))
    return path


class Hub:
    def __init__(self, path, *, fail_once=False, corrupt_hash=False):
        self.path, self.fail_once, self.calls = path, fail_once, []
        self.corrupt_hash = corrupt_hash

    def snapshot_download(self, **kwargs):
        self.calls.append(kwargs)
        marker = self.path.parent / "download.incomplete"
        if self.fail_once:
            marker.write_bytes(b"first network chunk")
            self.fail_once = False
            raise ConnectionError("interrupted")
        if len(self.calls) > 1: assert marker.read_bytes() == b"first network chunk"
        return str(self.path)

    def HfApi(self):
        return self

    def repo_info(self, **kwargs):
        rows = []
        for file in self.path.iterdir():
            data = file.read_bytes()
            rows.append(SimpleNamespace(rfilename=file.name, size=len(data),
                lfs={"sha256": "0" * 64 if self.corrupt_hash else hashlib.sha256(data).hexdigest()}, blob_id=None))
        return SimpleNamespace(sha=kwargs["revision"], siblings=rows)


def test_download_resume_registry_and_reuse_without_network(tmp_path):
    model = tiny_model(tmp_path / "snapshot")
    hub = Hub(model, fail_once=True)
    root = tmp_path / "shared"
    with pytest.raises(ConnectionError): resources.resolve(root, "q06", hub=hub)
    record = read_json(root / "resources.json")["q06"]
    assert record["state"] == "incomplete" and record["revision"] == resources.MODELS["q06"][1]
    ready = resources.resolve(root, "q06", hub=hub)
    assert ready["state"] == "ready" and ready["path"] == str(model.resolve())
    assert len(hub.calls) == 2
    assert "local_dir" not in hub.calls[0] and not hub.calls[0].get("force_download", False)
    assert resources.resolve(root, "q06", hub=object()) == ready
    assert not any((root / "runs").glob("**/*.safetensors"))


def test_incomplete_local_directory_falls_back_to_pinned_model(tmp_path):
    partial = tmp_path / "broken-local"
    partial.mkdir()
    hub = Hub(tiny_model(tmp_path / "snapshot"))
    row = resources.resolve(tmp_path / "shared", "q06", local=partial, hub=hub)
    assert row["local_validation_failed"] and row["origin"] == "hub-cache"
    assert hub.calls[0]["revision"] == resources.MODELS["q06"][1]


def test_explicit_new_local_path_is_not_silently_ignored(tmp_path):
    first, second = tiny_model(tmp_path / "one"), tiny_model(tmp_path / "two")
    root = tmp_path / "shared"
    resources.resolve(root, "q06", local=first, hub=object())
    assert resources.resolve(root, "q06", local=second, hub=object())["path"] == str(second.resolve())


def test_wrong_official_download_hash_is_not_ready(tmp_path):
    hub = Hub(tiny_model(tmp_path / "snapshot"), corrupt_hash=True)
    with pytest.raises(RuntimeError, match="SHA256"):
        resources.resolve(tmp_path / "shared", "q06", hub=hub)
    assert read_json(tmp_path / "shared/resources.json")["q06"]["state"] == "incomplete"


def test_model_shard_header_and_architecture_are_validated(tmp_path):
    path = tiny_model(tmp_path / "model")
    resources.validate_model(path, "q06")
    (path / "model.safetensors").write_bytes(b"truncated")
    with pytest.raises(Exception): resources.validate_model(path, "q06")


def test_wikitext_all_splits_and_schema(tmp_path):
    import pyarrow as arrow
    import pyarrow.parquet as parquet
    root = tmp_path / "wikitext-2-raw-v1"
    root.mkdir()
    for split in ("train", "validation", "test"):
        parquet.write_table(arrow.table({"text": ["first", "second"]}), root / (split + "-00000-of-00001.parquet"))
    assert set(resources.validate_wikitext(tmp_path)) == {"train", "validation", "test"}
    (root / "test-00000-of-00001.parquet").write_bytes(b"invalid")
    with pytest.raises(Exception): resources.validate_wikitext(tmp_path)


def test_atomic_evidence_cannot_overwrite_and_unsafe_manifest_rejected(tmp_path):
    path = tmp_path / "receipt.json"
    atomic_json(path, {"first": 1})
    with pytest.raises(FileExistsError): atomic_json(path, {"first": 2})
    assert read_json(path) == {"first": 1}
    assert not list(tmp_path.glob(".publish-*"))
    with pytest.raises(RuntimeError, match="Unsafe"):
        verify_files(tmp_path, {"../foreign": {"bytes": 1, "sha256": "x"}})


def test_lock_excludes_another_owner_and_releases_after_exception(tmp_path):
    from filelock import Timeout
    path = tmp_path / "resource.lock"
    with pytest.raises(ValueError):
        with exclusive_lock(path):
            with pytest.raises(Timeout):
                with exclusive_lock(path): pass
            raise ValueError("producer failed")
    with exclusive_lock(path): pass


@pytest.mark.parametrize("variant", sources.VARIANTS)
def test_registered_sources_are_exact_and_idempotent(tmp_path, variant):
    before = file_manifest(sources.VENDOR)
    path, sha = sources.materialize(tmp_path, variant)
    assert sources.materialize(tmp_path, variant) == (path, sha)
    rows = read_json(path / "manifest.json")
    assert len(rows) == (119 if variant == "baseline" else 120)
    assert sha == identity(rows)
    assert file_manifest(sources.VENDOR) == before


def test_materialized_source_tampering_is_rejected(tmp_path):
    root, _ = sources.materialize(tmp_path, "full_v4")
    (root / "realq/config.py").write_text("tampered\n")
    with pytest.raises(RuntimeError, match="changed"):
        sources.materialize(tmp_path, "full_v4")


@pytest.mark.parametrize("model", ["q06", "q17"])
@pytest.mark.parametrize("qa", ["none", "full"])
def test_native_configuration_complete_and_global_batches_unchanged(tmp_path, model, qa):
    path = sources.VENDOR / "storage/realq/config.py"
    module_spec = importlib.util.spec_from_file_location("portable_native_config_fixture", path)
    module = importlib.util.module_from_spec(module_spec)
    import sys
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    assets = {model: {"path": str(tmp_path / model)}}
    cfg = protocol.make_config(model, assets, tmp_path / "shared", tmp_path / "run", qa=qa, cache_hit=True, save_checkpoint=True)
    native = module.Config(**{k: v for k, v in cfg.items() if k in {f.name for f in fields(module.Config) if f.init}})
    native.model_name = cfg["model_name"]
    assert asdict(native) == cfg
    assert (cfg["global_loss_bsz"], cfg["hessian_accum_bsz"], cfg["nsamples"], cfg["seq_len"]) == (4, 8, 2048, 2048)
    assert cfg["backward_bsz"] == cfg["backward_samples"] == 32
    assert cfg["lm_eval_batch_size"] == 16 and cfg["lm_eval"] == (qa == "full")
    assert cfg["saliency_storage_dtype"] == "fp32" and cfg["attention_backend"] == "sdpa"


def test_shared_identity_binds_content_runtime_and_topology():
    assets = {"q06": {"content_sha256": "model"}, "wikitext2": {"content_sha256": "data"}}
    original = protocol.cache_identity("q06", assets, 2, {"driver": "a"})
    assert identity(original) != identity(protocol.cache_identity("q06", assets, 2, {"driver": "b"}))
    assets["wikitext2"]["content_sha256"] = "different-data"
    assert identity(original) != identity(protocol.cache_identity("q06", assets, 2, {"driver": "a"}))
    with pytest.raises(ValueError): protocol.cache_identity("q06", assets, 4, {})


def test_default_eight_arms_and_control_required():
    matrix = protocol.jobs(["q06", "q17"], list(sources.DEFAULT_VARIANTS))
    assert len(matrix) == 8
    assert [j["variant"] for j in matrix[::4]] == ["full_v4", "full_v4"]
    with pytest.raises(ValueError): protocol.jobs(["q06"], ["without_CPU_streams"])


def test_device_selection_rejects_ordinals_slices_and_unsafe_paths():
    plan = {"world_size": 2, "devices": ["GPU-11111111-1111-1111-1111-111111111111", "GPU-22222222-2222-2222-2222-222222222222"],
        "models": ["q06"], "variants": ["full_v4"], "jobs": protocol.jobs(["q06"], ["full_v4"]),
        "qa": "none", "qa_batch_size": 16, "qa_tasks": list(resources.TASKS)}
    protocol.validate_scope(plan)
    for invalid in ("0", "MIG-123", "GPU-short", "GPU-../../foreign"):
        with pytest.raises(RuntimeError):
            protocol.validate_scope({**plan, "devices": [plan["devices"][0], invalid]})


def test_semantic_tensor_identity_binds_shape_dtype_values_and_handles_bf16():
    value = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    assert protocol.semantic_digest(value) == protocol.semantic_digest(value.clone())
    assert protocol.semantic_digest(value) != protocol.semantic_digest(value.reshape(4, 3))
    assert protocol.semantic_digest(value) != protocol.semantic_digest(value.float())
    other = value.clone(); other[0, 0] = 1
    assert protocol.semantic_digest(value) != protocol.semantic_digest(other)


def test_quality_fail_stop_conditions():
    value = {"state_sha256": "same", "kl": 0.1, "ppl": 9.0, "qa": {"piqa": 50}}
    assert campaign.quality(value, dict(value))["valid"]
    for key, different in (("state_sha256", "different"), ("kl", 0.2), ("ppl", float("nan")), ("qa", {})):
        assert not campaign.quality(value, {**value, key: different})["valid"]


def test_streaming_observer_preserves_outer_peak_and_failure_endpoint(tmp_path):
    from portable_realq.observe import Observer
    class CUDA:
        def __init__(self): self.peak = 0
        def synchronize(self, device): pass
        def memory_allocated(self, device): return 1
        def memory_reserved(self, device): return 2
        def max_memory_allocated(self, device): return self.peak
        def max_memory_reserved(self, device): return self.peak + 1
        def reset_peak_memory_stats(self, device): self.peak = 0
    cuda = CUDA()
    observer = Observer(SimpleNamespace(cuda=cuda), tmp_path / "allocator.jsonl", 0)
    with pytest.raises(ValueError):
        with observer.scope("outer"):
            cuda.peak = 100
            with observer.scope("inner"): cuda.peak = 10
            raise ValueError("failure")
    observer.close()
    rows = [json.loads(line) for line in (tmp_path / "allocator.jsonl").read_text().splitlines()]
    assert rows[-1]["event"] == "end" and not rows[-1]["completed"]
    assert rows[-1]["peaks"]["0"]["allocated"] == 100


def test_failed_cuda_measurement_keeps_endpoint_and_original_error(tmp_path):
    from portable_realq.observe import Observer
    class CUDA:
        failed = False
        def synchronize(self, device):
            if self.failed: raise RuntimeError("CUDA failure")
        def memory_allocated(self, device): return 1
        memory_reserved = memory_allocated
        max_memory_allocated = memory_allocated
        max_memory_reserved = memory_allocated
        def reset_peak_memory_stats(self, device): pass
    cuda = CUDA()
    observer = Observer(SimpleNamespace(cuda=cuda), tmp_path / "failure.jsonl", 0)
    with pytest.raises(ValueError, match="original"):
        with observer.scope("stage"):
            cuda.failed = True
            raise ValueError("original")
    observer.close()
    rows = [json.loads(line) for line in (tmp_path / "failure.jsonl").read_text().splitlines()]
    assert rows[-1]["measurement_completed"] is False
    assert rows[-1]["completed"] is False


def test_import_and_help_never_load_torch_or_download():
    import subprocess, sys
    result = subprocess.run([sys.executable, "-c", "import sys; import portable_realq.campaign; assert 'torch' not in sys.modules"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_host_transport_preserves_actual_cpu_autograd_and_restores_methods():
    from portable_realq.transport import HostTransport
    original = torch.Tensor.to
    guard = HostTransport(torch)
    try:
        value = torch.arange(5.0, requires_grad=True)
        output = value.to(dtype=torch.float64).square().sum()
        output.backward()
        assert torch.equal(value.grad, 2 * value.detach())
        destination = torch.zeros(5)
        destination.copy_(value.detach())
        assert torch.equal(destination, value.detach())
    finally:
        assert guard.close()["records"] == []
    assert torch.Tensor.to is original


def fixture_plan(tmp_path, monkeypatch):
    source_rows = {}
    for variant in sources.DEFAULT_VARIANTS:
        path, sha = sources.materialize(tmp_path / "shared", variant)
        source_rows[variant] = {"path": str(path), "sha256": sha}
    output = tmp_path / "runs/one"
    output.mkdir(parents=True)
    plan = {"schema": "portable-two-gpu-plan-v1", "world_size": 2,
        "devices": ["GPU-11111111-1111-1111-1111-111111111111", "GPU-22222222-2222-2222-2222-222222222222"], "models": ["q06"], "variants": list(sources.DEFAULT_VARIANTS),
        "jobs": protocol.jobs(["q06"], list(sources.DEFAULT_VARIANTS)), "qa": "none", "qa_batch_size": 16,
        "qa_tasks": list(resources.TASKS), "runtime": {"test": True}, "assets": {},
        "output": str(output), "shared": str(tmp_path / "shared"), "sources": source_rows,
        "code_bindings": campaign.code_bindings(), "timeout_seconds": 5,
        "caches": {"q06": {"key": "test-cache", "definition": {"world": 2}}}}
    model = tiny_model(tmp_path / "model")
    plan["assets"] = {"q06": {"path": str(model), "content_sha256": "model", "files": file_manifest(model)}}
    atomic_json(output / "plan.json", plan)
    monkeypatch.setattr(campaign, "runtime_info", lambda _: plan["runtime"])
    monkeypatch.setattr(campaign, "require_idle", lambda _: None)
    monkeypatch.setattr(campaign, "runtime_budget", lambda *args: {})
    monkeypatch.setattr(campaign.tempfile, "gettempdir", lambda: str(tmp_path))
    return plan, output


def fake_arm(command, output, env, timeout, devices):
    spec = read_json(output / "spec.json")
    cache = Path(spec["cache"])
    (cache / "static").mkdir(exist_ok=True)
    (cache / "static/world2_rank0.pt").write_bytes(b"rank0 payload")
    (cache / "static/world2_rank1.pt").write_bytes(b"rank1 payload")
    (cache / "calibration.pt").write_bytes(b"tokens")
    state = {"state": {str(i): {"test_fixture_only": True} for i in range(535)},
             "online_hadamard": [{"test_fixture_only": i} for i in range(28)]}
    atomic_json(output / "state.json", state)
    for rank in range(2):
        atomic_json(output / f"rank{rank}/result.json", {"execution_completed": True, "world_size": 2,
            "rank": rank, "layers": list(range(28)),
            "stages": {"calibration_sha256": "tokens", "statistics_sha256": f"rank{rank}"},
            "metrics": {"kl": 0.1, "ppl": 8.0, "state_sha256": identity(state)}})
    return {"exit_code": 0, "timed_out": False, "wall_seconds": 2, "sampler_stopped": True, "telemetry_available": True}


def test_controller_serial_reuse_and_immutable_attempt(tmp_path, monkeypatch):
    plan, output = fixture_plan(tmp_path, monkeypatch)
    calls = []
    def execute(*args):
        calls.append(read_json(args[1] / "spec.json"))
        return fake_arm(*args)
    monkeypatch.setattr(campaign, "wait_owned", execute)
    campaign.run(output / "plan.json")
    assert len(calls) == 4
    assert [s["cache_hit"] for s in calls] == [False, True, True, True]
    assert len({s["cache"] for s in calls}) == 1
    assert read_json(output / "status.json")["execution_completed"]
    with pytest.raises(RuntimeError, match="already attempted"):
        campaign.run(output / "plan.json")
    assert len(calls) == 4


@pytest.mark.parametrize("failure", ["exit", "quality", "rank", "timeout", "telemetry"])
def test_controller_stops_on_first_failure_and_keeps_evidence(tmp_path, monkeypatch, failure):
    plan, output = fixture_plan(tmp_path, monkeypatch)
    calls = []
    def execute(*args):
        calls.append(args[1])
        row = fake_arm(*args)
        if failure == "exit": row["exit_code"] = 1
        if failure == "timeout": row["timed_out"] = True
        if failure == "telemetry": row["telemetry_available"] = False
        if failure in ("quality", "rank"):
            target = args[1] / "rank0/result.json"
            value = read_json(target)
            if failure == "rank": value["layers"] = [0]
            else: value["metrics"]["ppl"] = float("inf")
            # JSON evidence intentionally rejects nonfinite values; use a very
            # large finite metric and compare against the control in arm two.
            if failure == "quality":
                if len(calls) == 1: value["metrics"]["ppl"] = 8.0
                else: value["metrics"]["ppl"] = 80.0
            atomic_json(target, value, replace=True)
        return row
    monkeypatch.setattr(campaign, "wait_owned", execute)
    with pytest.raises(RuntimeError): campaign.run(output / "plan.json")
    assert len(calls) == (2 if failure == "quality" else 1)
    assert (calls[0] / "spec.json").exists()
    assert read_json(output / "status.json")["execution_completed"] is False


def test_unsealed_cache_does_not_get_overwritten(tmp_path, monkeypatch):
    plan, output = fixture_plan(tmp_path, monkeypatch)
    cache = Path(plan["shared"]) / "stage0/test-cache"
    cache.mkdir(parents=True)
    (cache / "failed-producer.bin").write_bytes(b"retain")
    monkeypatch.setattr(campaign, "wait_owned", lambda *a: pytest.fail("must not launch"))
    with pytest.raises(RuntimeError, match="Unsealed"):
        campaign.run(output / "plan.json")
    assert (cache / "failed-producer.bin").read_bytes() == b"retain"


def test_full_model_fingerprint_rejects_missing_state_and_changed_hadamard():
    model = torch.nn.Module()
    for index in range(507):
        model.register_parameter("p" + str(index), torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16)))
    for index in range(28):
        module = torch.nn.Module()
        module.register_buffer("had_K", torch.ones(1))
        module.online_full_had, module.fp32_had = True, False
        module.online_partial_had, module.K, module.had_dim = False, 1, 0
        model.add_module("down_proj" + str(index), module)
    reference = protocol.model_fingerprint(model)
    assert len(reference["state"]) == 535 and len(reference["online_hadamard"]) == 28
    model.down_proj0.had_K[0] = 2
    assert identity(protocol.model_fingerprint(model)) != identity(reference)
    model.down_proj0.fp32_had = True
    with pytest.raises(RuntimeError, match="Hadamard"): protocol.model_fingerprint(model)


def test_cpu_import_preflight_rejects_incompatible_runtime(monkeypatch):
    monkeypatch.setattr(campaign.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="incompatible dependency"))
    with pytest.raises(RuntimeError, match="CPU import preflight failed"):
        campaign.cpu_import_preflight(Path("example"))
