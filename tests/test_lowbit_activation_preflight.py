import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lowbit_activation_preflight",
    ROOT / "tools" / "lowbit_activation_preflight.py",
)
assert SPEC is not None and SPEC.loader is not None
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def _plan(*, tuning_world_size=1, final_world_size=4, **extra):
    plan = {
        "tuning": {"world_size": tuning_world_size},
        "final": {"world_size": final_world_size},
        "python_environment": {
            "venv": "/test/experiment/.venv",
            "activation_required": True,
        },
        "runtime_versions": {
            "torch": "test",
            "torch_runtime": "2.test",
            "transformers": "test",
            "lm-eval": preflight.EXPECTED_LM_EVAL_VERSION,
        },
    }
    plan.update(extra)
    return plan


def test_task_manager_omits_include_path_when_custom_tasks_are_absent(
    tmp_path,
    monkeypatch,
):
    calls = []

    class TaskManager:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.all_tasks = ["piqa"]

    fake_tasks = SimpleNamespace(TaskManager=TaskManager)
    fake_lm_eval = SimpleNamespace(tasks=fake_tasks)
    fake_utils = SimpleNamespace(
        pattern_match=lambda patterns, all_tasks: [
            task for task in all_tasks if task in patterns
        ]
    )

    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    monkeypatch.setattr(
        preflight.importlib,
        "import_module",
        lambda name: {
            "lm_eval": fake_lm_eval,
            "lm_eval.utils": fake_utils,
        }[name],
    )

    result, errors = preflight._lm_eval_tasks(["piqa"], load_datasets=False)

    assert errors == []
    assert result["resolved"] == {"piqa": ["piqa"]}
    assert calls == [{"include_defaults": True}]


def test_task_manager_passes_existing_custom_task_directory(
    tmp_path,
    monkeypatch,
):
    custom_tasks = tmp_path / "datasets" / "lm_eval_configs" / "tasks"
    custom_tasks.mkdir(parents=True)
    calls = []

    class TaskManager:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.all_tasks = ["piqa"]

    fake_tasks = SimpleNamespace(TaskManager=TaskManager)
    fake_lm_eval = SimpleNamespace(tasks=fake_tasks)
    fake_utils = SimpleNamespace(pattern_match=lambda _patterns, _tasks: ["piqa"])

    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    monkeypatch.setattr(
        preflight.importlib,
        "import_module",
        lambda name: {
            "lm_eval": fake_lm_eval,
            "lm_eval.utils": fake_utils,
        }[name],
    )

    _, errors = preflight._lm_eval_tasks(["piqa"], load_datasets=False)

    assert errors == []
    assert calls == [
        {
            "include_defaults": True,
            "include_path": str(custom_tasks),
        }
    ]


def test_path_probe_uses_nearest_existing_parent(tmp_path, monkeypatch):
    existing_output_parent = tmp_path / "output"
    existing_output_parent.mkdir()
    existing_cache_parent = tmp_path / "cache"
    existing_cache_parent.mkdir()
    monkeypatch.setattr(preflight, "ROOT", tmp_path)

    checks, errors = preflight._path_probe(
        {
            "output_root": "output/not-created/yet/results",
            "static_cache_root": "cache/not-created/yet/static",
        }
    )

    assert errors == []
    assert checks["output_root"]["probe_path"] == str(existing_output_parent)
    assert checks["output_root"]["parent_exists"] is True
    assert checks["output_root"]["parent_writable"] is True
    assert checks["cache_root"]["probe_path"] == str(existing_cache_parent)
    assert checks["cache_root"]["parent_exists"] is True
    assert checks["cache_root"]["parent_writable"] is True


def test_model_check_rejects_a_wrong_reviewed_signature(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "hidden_size": 2048,
                "num_hidden_layers": 16,
                "architectures": ["LlamaForCausalLM"],
            }
        ),
        encoding="utf-8",
    )
    (model / "model.safetensors").write_bytes(b"weights")

    result = preflight._check_model(
        model,
        expected_signature={
            "model_type": "llama",
            "hidden_size": 3072,
            "num_hidden_layers": 28,
            "architectures": ["LlamaForCausalLM"],
        },
    )

    assert any("signature mismatch" in error for error in result["errors"])


def test_runtime_rejects_wrong_pinned_distribution_version(monkeypatch):
    fake_torch = SimpleNamespace(
        __version__="2.9.1+cu128",
        version=SimpleNamespace(cuda="12.8"),
        cuda=_FakeCuda([183] * 8),
    )
    versions = {
        "torch": "2.9.1+cu128",
        "transformers": "wrong",
        "accelerate": "test",
        "datasets": "test",
        "lm-eval": preflight.EXPECTED_LM_EVAL_VERSION,
    }
    monkeypatch.setattr(preflight, "_version", versions.get)
    monkeypatch.setattr(
        preflight.importlib,
        "import_module",
        lambda name: fake_torch if name == "torch" else None,
    )
    monkeypatch.setattr(
        preflight,
        "_python_environment",
        lambda _plan_value: (
            {"expected_venv": "/test/experiment/.venv", "valid": True},
            [],
        ),
    )

    _, errors = preflight._runtime(
        _plan(
            runtime_versions={
                "torch": "2.9.1+cu128",
                "torch_runtime": "2.9.1+cu128",
                "transformers": "4.56.2",
                "lm-eval": preflight.EXPECTED_LM_EVAL_VERSION,
            },
            runtime_requirements={
                "minimum_gpu_count": 8,
                "minimum_gpu_memory_gib": 120,
            },
        ),
        min_gpu_memory_gib=None,
    )

    assert any(
        "transformers version must be 4.56.2, got wrong" in error
        for error in errors
    )


def test_runtime_requirements_enforce_eight_gpus_and_planned_world_size():
    requirements, errors = preflight._runtime_requirements(
        _plan(),
        min_gpu_memory_gib=None,
    )
    assert errors == []
    assert requirements["minimum_gpu_count"] == 8
    assert requirements["minimum_gpu_memory_bytes"] is None

    requirements, errors = preflight._runtime_requirements(
        _plan(final_world_size=16),
        min_gpu_memory_gib=None,
    )
    assert errors == []
    assert requirements["minimum_gpu_count"] == 16


def test_runtime_requirements_accept_optional_memory_from_plan_or_cli():
    requirements, errors = preflight._runtime_requirements(
        _plan(
            runtime_requirements={
                "minimum_gpu_memory_gib": 80,
            }
        ),
        min_gpu_memory_gib=None,
    )
    assert errors == []
    assert requirements["minimum_gpu_memory_bytes"] == 80 * preflight.GIB
    assert requirements["memory_requirement_source"] == "plan"

    requirements, errors = preflight._runtime_requirements(
        _plan(
            runtime_requirements={
                "minimum_gpu_memory_gib": 80,
            }
        ),
        min_gpu_memory_gib=160,
    )
    assert errors == []
    assert requirements["minimum_gpu_memory_bytes"] == 160 * preflight.GIB
    assert requirements["memory_requirement_source"] == "plan+cli"

    requirements, errors = preflight._runtime_requirements(
        _plan(
            runtime_requirements={
                "minimum_gpu_memory_gib": 80,
            }
        ),
        min_gpu_memory_gib=40,
    )
    assert errors == []
    assert requirements["minimum_gpu_memory_bytes"] == 80 * preflight.GIB
    assert requirements["memory_requirement_source"] == "plan+cli"


def test_runtime_requirements_fail_closed_on_invalid_plan_values():
    _, errors = preflight._runtime_requirements(
        _plan(
            tuning_world_size=True,
            runtime_requirements={
                "minimum_gpu_memory_gib": -1,
            },
        ),
        min_gpu_memory_gib=None,
    )

    assert any(
        "tuning.world_size must be a positive integer" in error
        for error in errors
    )
    assert any("positive finite GiB value" in error for error in errors)


class _FakeUuid:
    """Mimic torch 2.9's non-JSON-native private UUID object."""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


class _FakeCuda:
    def __init__(self, memory_gib):
        self._memory_bytes = [value * preflight.GIB for value in memory_gib]

    def is_available(self):
        return True

    def device_count(self):
        return len(self._memory_bytes)

    def get_device_properties(self, index):
        return SimpleNamespace(
            name=f"GPU-{index}",
            uuid=_FakeUuid(f"uuid-{index}"),
            major=9,
            minor=0,
        )

    def mem_get_info(self, index):
        total = self._memory_bytes[index]
        return total, total


def _runtime_with_fake_cuda(monkeypatch, memory_gib, *, minimum_memory=None):
    fake_torch = SimpleNamespace(
        __version__="2.test",
        version=SimpleNamespace(cuda="12.test"),
        cuda=_FakeCuda(memory_gib),
    )
    monkeypatch.setattr(
        preflight,
        "_version",
        lambda distribution: (
            preflight.EXPECTED_LM_EVAL_VERSION
            if distribution == "lm-eval"
            else "test"
        ),
    )
    monkeypatch.setattr(
        preflight.importlib,
        "import_module",
        lambda name: fake_torch if name == "torch" else None,
    )
    monkeypatch.setattr(
        preflight,
        "_python_environment",
        lambda _plan_value: (
            {"expected_venv": "/test/experiment/.venv", "valid": True},
            [],
        ),
    )
    return preflight._runtime(
        _plan(),
        min_gpu_memory_gib=minimum_memory,
    )


def test_python_environment_requires_exact_activated_venv(
    tmp_path,
    monkeypatch,
):
    venv = tmp_path / ".venv"
    (venv / "bin").mkdir(parents=True)
    plan_value = {
        "python_environment": {
            "venv": str(venv),
            "activation_required": True,
        }
    }
    monkeypatch.setenv("VIRTUAL_ENV", str(venv))
    monkeypatch.setattr(preflight.sys, "prefix", str(venv))
    monkeypatch.setattr(
        preflight.sys,
        "executable",
        str(venv / "bin" / "python"),
    )
    monkeypatch.setattr(
        preflight.shutil,
        "which",
        lambda name: str(venv / "bin" / name),
    )

    runtime, errors = preflight._python_environment(plan_value)

    assert errors == []
    assert runtime["valid"] is True
    assert runtime["expected_venv"] == str(venv)


def test_python_environment_rejects_system_python(tmp_path, monkeypatch):
    venv = tmp_path / ".venv"
    venv.mkdir()
    plan_value = {
        "python_environment": {
            "venv": str(venv),
            "activation_required": True,
        }
    }
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(preflight.sys, "prefix", "/usr")
    monkeypatch.setattr(preflight.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(
        preflight.shutil,
        "which",
        lambda name: f"/usr/bin/{name}",
    )

    runtime, errors = preflight._python_environment(plan_value)

    assert runtime["valid"] is False
    assert any("not activated" in error for error in errors)
    assert any("current Python" in error for error in errors)
    assert any("outside" in error for error in errors)


def test_canoe_context_requires_planned_job_and_matching_hostname(monkeypatch):
    plan_value = {"resolutions": {"canoe_job_id": "j-planned"}}
    monkeypatch.setattr(
        preflight.socket,
        "gethostname",
        lambda: "j-planned-master-0",
    )

    context, errors = preflight._canoe_context(
        plan_value,
        canoe_job_id="j-planned",
    )

    assert errors == []
    assert context["valid"] is True

    context, errors = preflight._canoe_context(
        plan_value,
        canoe_job_id="j-other",
    )

    assert context["valid"] is False
    assert any("does not match plan" in error for error in errors)


def test_canoe_context_rejects_other_job_hostname(monkeypatch):
    monkeypatch.setattr(
        preflight.socket,
        "gethostname",
        lambda: "j-forbidden-master-0",
    )

    context, errors = preflight._canoe_context(
        {"resolutions": {"canoe_job_id": "j-planned"}},
        canoe_job_id="j-planned",
    )

    assert context["valid"] is False
    assert any("does not belong" in error for error in errors)


def test_runtime_fails_closed_with_fewer_than_eight_visible_gpus(monkeypatch):
    runtime, errors = _runtime_with_fake_cuda(monkeypatch, [183] * 7)

    assert runtime["requirements"]["minimum_gpu_count"] == 8
    assert runtime["cuda"]["requirement_check"]["passed"] is False
    assert any(
        "at least 8 visible CUDA GPUs are required, detected 7" in error
        for error in errors
    )


def test_runtime_fails_closed_when_too_few_gpus_meet_memory_floor(monkeypatch):
    runtime, errors = _runtime_with_fake_cuda(
        monkeypatch,
        [183] * 7 + [79],
        minimum_memory=80,
    )

    check = runtime["cuda"]["requirement_check"]
    assert check["detected_gpu_count"] == 8
    assert check["eligible_gpu_count"] == 7
    assert check["passed"] is False
    assert any(
        "at least 8 visible CUDA GPUs with >=80.0 GiB" in error
        for error in errors
    )


def test_runtime_accepts_eight_gpus_meeting_optional_memory_floor(monkeypatch):
    runtime, errors = _runtime_with_fake_cuda(
        monkeypatch,
        [183] * 8,
        minimum_memory=80,
    )

    assert errors == []
    assert runtime["cuda"]["requirement_check"]["passed"] is True
    assert runtime["cuda"]["devices"][0]["uuid"] == "uuid-0"
    json.dumps(runtime)
