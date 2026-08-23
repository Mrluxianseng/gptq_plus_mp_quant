import builtins
import importlib.util
import json
import socket
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lowbit_activation_formal_eval_audit",
    ROOT / "tools" / "lowbit_activation_formal_eval_audit.py",
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _write(path: Path, content: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def _yaml_scalar(value):
    if value is None:
        return "null"
    return json.dumps(value)


def _leaf_yaml(name, protocol):
    split_key = (
        "test_split" if protocol["split"] == "test" else "validation_split"
    )
    return "\n".join(
        (
            f"task: {name}",
            f"dataset_path: {_yaml_scalar(protocol['dataset_path'])}",
            f"dataset_name: {_yaml_scalar(protocol['task_dataset_name'])}",
            f"{split_key}: {_yaml_scalar(protocol['split'])}",
            "metric_list:",
            f"  - metric: {protocol['metric']}",
            "metadata:",
            f"  version: {_yaml_scalar(protocol['task_version'])}",
            "",
        )
    )


def _dataset_cache(
    hf_home: Path,
    *,
    dataset_name: str,
    config_name: str,
    version: str,
    fingerprint: str,
    split: str,
    sample_count: int,
) -> Path:
    directory = (
        hf_home
        / "datasets"
        / dataset_name
        / config_name
        / version
        / fingerprint
    )
    info = {
        "dataset_name": dataset_name,
        "config_name": config_name,
        "version": {"version_str": version},
        "splits": {
            split: {
                "name": split,
                "num_examples": sample_count,
                "dataset_name": dataset_name,
            }
        },
    }
    info_path = _write(
        directory / "dataset_info.json",
        json.dumps(info, sort_keys=True),
    )
    _write(
        directory / f"{dataset_name}-{split}.arrow",
        f"{dataset_name}/{config_name}/{split}/{sample_count}".encode(),
    )
    return info_path


def _task_closure_hash(task_root: Path):
    index, errors = audit._index_task_tree(task_root)
    assert errors == []
    paths = set()
    for name in audit.EXPECTED_TASKS:
        kind = "group" if name == "ceval-valid" else "task"
        definition = audit._single_definition(index, name, kind)
        _, closure = audit._effective_yaml(
            task_root / definition["path"],
            task_root,
        )
        paths.update(closure)
    for name in audit.CEVAL_LEAVES:
        definition = audit._single_definition(index, name, "task")
        _, closure = audit._effective_yaml(
            task_root / definition["path"],
            task_root,
        )
        paths.update(closure)
    return audit._hash_file_set(task_root, paths)["combined_sha256"]


@pytest.fixture
def formal_fixture(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    venv = repo / ".venv"
    lm_eval_root = (
        venv / "lib" / "python3.12" / "site-packages" / "lm_eval"
    )
    task_root = lm_eval_root / "tasks"
    hf_home = repo / "hf-cache"

    evaluator_path = _write(
        lm_eval_root / "evaluator.py",
        """
def simple_evaluate(
    model,
    tasks=None,
    num_fewshot=None,
    limit=None,
    system_instruction=None,
    apply_chat_template=False,
    fewshot_as_multiturn=False,
    task_manager=None,
):
    return {}
""".lstrip(),
    )
    for relative in audit.REQUIRED_LM_EVAL_RUNTIME_FILES:
        if relative == "evaluator.py":
            continue
        _write(
            lm_eval_root / relative,
            f"# synthetic reviewed runtime source: {relative}\n",
        )

    task_paths = {
        "piqa": "piqa/piqa.yaml",
        "hellaswag": "hellaswag/hellaswag.yaml",
        "arc_easy": "arc/arc_easy.yaml",
        "arc_challenge": "arc/arc_challenge.yaml",
        "winogrande": "winogrande/default.yaml",
        "lambada_openai": "lambada/lambada_openai.yaml",
        "boolq": "super_glue/boolq/default.yaml",
        "openbookqa": "openbookqa/openbookqa.yaml",
        "social_iqa": "siqa/siqa.yaml",
    }
    for name, protocol in audit.TASK_PROTOCOLS.items():
        if name == "arc_challenge":
            continue
        content = _leaf_yaml(name, protocol)
        if name == "hellaswag":
            content += "process_docs: !function utils.process_docs\n"
        _write(task_root / task_paths[name], content)
    _write(
        task_root / "hellaswag" / "utils.py",
        "def process_docs(dataset):\n    return dataset\n",
    )
    _write(
        task_root / task_paths["arc_challenge"],
        "\n".join(
            (
                "include: arc_easy.yaml",
                "task: arc_challenge",
                'dataset_name: "ARC-Challenge"',
                "",
            )
        ),
    )

    _write(
        task_root / "ceval" / "_default_ceval_yaml",
        """
dataset_path: "ceval/ceval-exam"
validation_split: val
metric_list:
  - metric: acc
  - metric: acc_norm
metadata:
  version: "2.0"
""".lstrip(),
    )
    group_lines = [
        "group: ceval-valid",
        "task:",
        *(f"  - {leaf}" for leaf in audit.CEVAL_LEAVES),
        "aggregate_metric_list:",
        "  - metric: acc",
        "    aggregation: mean",
        "    weight_by_size: true",
        "  - metric: acc_norm",
        "    aggregation: mean",
        "    weight_by_size: true",
        "metadata:",
        '  version: "2.0"',
        "",
    ]
    ceval_group_path = _write(
        task_root / "ceval" / "_ceval-valid.yaml",
        "\n".join(group_lines),
    )
    for leaf in audit.CEVAL_LEAVES:
        suffix = leaf.removeprefix("ceval-valid_")
        _write(
            task_root / "ceval" / f"{leaf}.yaml",
            "\n".join(
                (
                    "include: _default_ceval_yaml",
                    f"task: {leaf}",
                    f'dataset_name: "{suffix}"',
                    "",
                )
            ),
        )

    eval_utils_path = _write(
        repo / "utils" / "eval_utils.py",
        f"""
PAPER_QA_TASKS = {audit.EXPECTED_TASKS!r}


def _resolve_paper_qa_tasks(pattern_match, all_tasks):
    return list(PAPER_QA_TASKS)


def _task_accuracy(task_name, result):
    if "acc_norm,none" in result:
        return result["acc_norm,none"]
    if "acc,none" in result:
        return result["acc,none"]
    raise RuntimeError(task_name)


def qa_eval(model, tokenizer, lm_eval_batch_size=32):
    custom_task_dir = "./datasets/lm_eval_configs/tasks"
    if os.path.isdir(custom_task_dir):
        task_manager = lm_eval.tasks.TaskManager(
            include_path=custom_task_dir,
            include_defaults=True,
        )
    else:
        task_manager = lm_eval.tasks.TaskManager(include_defaults=True)
    task_names = _resolve_paper_qa_tasks(
        lm_eval_utils.pattern_match,
        task_manager.all_tasks,
    )
    for task_name in task_names:
        result = lm_eval.simple_evaluate(
            hflm,
            tasks=[task_name],
            task_manager=task_manager,
        )
    return result
""".lstrip(),
    )
    _write(
        repo / "utils" / "data_utils.py",
        """
def _get_wikitext2(split):
    return load_dataset(
        "./datasets/wikitext",
        "wikitext-2-raw-v1",
        split=split,
        trust_remote_code=True,
    )
""".lstrip(),
    )
    _write(
        repo / "realq" / "pipeline.py",
        """
def _setup_eval(cfg, analyzer):
    return data_utils.get_loaders(
        ds,
        split="test",
        tokenizer=analyzer.tokenizer,
        seq_len=cfg.eval_seq_len,
        num_samples=cfg.nsamples,
    )
""".lstrip(),
    )
    _write(
        repo / "ptq.py",
        """
def main():
    return data_utils.get_loaders(
        eval_dataset,
        split="test",
        tokenizer=tokenizer,
        seq_len=args.eval_seq_len,
        num_samples=args.nsamples,
    )
""".lstrip(),
    )

    task_info_paths = {}
    for name, protocol in audit.TASK_PROTOCOLS.items():
        task_info_paths[name] = _dataset_cache(
            hf_home,
            dataset_name=protocol["cache_dataset_name"],
            config_name=protocol["cache_config_name"],
            version=protocol["dataset_version"],
            fingerprint=protocol["builder_fingerprint"],
            split=protocol["split"],
            sample_count=protocol["sample_count"],
        )
    ceval_info_paths = {}
    for leaf in audit.CEVAL_LEAVES:
        suffix = leaf.removeprefix("ceval-valid_")
        ceval_info_paths[leaf] = _dataset_cache(
            hf_home,
            dataset_name="ceval-exam",
            config_name=suffix,
            version="0.0.0",
            fingerprint=audit._COMMON_DATASET_FINGERPRINTS["ceval"],
            split="val",
            sample_count=audit.CEVAL_SAMPLE_COUNTS[suffix],
        )
    wikitext_info_path = _dataset_cache(
        hf_home,
        dataset_name=audit.WIKITEXT_PROTOCOL["cache_dataset_name"],
        config_name=audit.WIKITEXT_PROTOCOL["cache_config_name"],
        version=audit.WIKITEXT_PROTOCOL["dataset_version"],
        fingerprint=audit.WIKITEXT_PROTOCOL["builder_fingerprint"],
        split=audit.WIKITEXT_PROTOCOL["split"],
        sample_count=audit.WIKITEXT_PROTOCOL["sample_count"],
    )

    wikitext_root = (
        repo / "datasets" / "wikitext" / "wikitext-2-raw-v1"
    )
    for index, name in enumerate(sorted(audit.EXPECTED_WIKITEXT_FILES), 1):
        _write(wikitext_root / name, f"fake-parquet-{index}".encode())
    wikitext_hash = audit._hash_file_set(
        wikitext_root,
        wikitext_root.glob("*.parquet"),
    )
    expected_wikitext_files = {
        item["relative_path"]: {
            "sha256": item["sha256"],
            "size_bytes": item["size_bytes"],
        }
        for item in wikitext_hash["files"]
    }

    plan = {
        "paper_zero_shot_tasks": list(audit.EXPECTED_TASKS),
        "fixed_numerics": {
            "dataset": "wikitext2",
            "eval_datasets": ["wikitext2"],
            "eval_seq_len": 2048,
        },
        "final": {
            "lm_eval": True,
            "lm_eval_batch_size": 8,
        },
        "runtime_environment": {
            "HF_HOME": str(hf_home),
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "runtime_versions": dict(audit.EXPECTED_RUNTIME_VERSIONS),
        "python_environment": {
            "venv": str(venv),
            "activation_required": True,
        },
    }
    plan_path = _write(
        repo / "experiments" / "lowbit_activation" / "plan.json",
        json.dumps(plan, sort_keys=True),
    )

    monkeypatch.setattr(
        audit,
        "EXPECTED_EVALUATOR_SHA256",
        audit._sha256_file(evaluator_path),
    )
    monkeypatch.setattr(
        audit,
        "EXPECTED_TASK_CLOSURE_SHA256",
        _task_closure_hash(task_root),
    )
    monkeypatch.setattr(
        audit,
        "EXPECTED_LM_EVAL_PYTHON_CLOSURE_SHA256",
        audit._hash_file_set(
            lm_eval_root,
            lm_eval_root.rglob("*.py"),
        )["combined_sha256"],
    )
    expected_dataset_artifacts = {}
    for name, info_path in task_info_paths.items():
        split = audit.TASK_PROTOCOLS[name]["split"]
        expected_dataset_artifacts[name] = audit._hash_file_set(
            hf_home,
            [info_path, *info_path.parent.glob(f"*-{split}.arrow")],
        )["combined_sha256"]
    for leaf, info_path in ceval_info_paths.items():
        expected_dataset_artifacts[leaf] = audit._hash_file_set(
            hf_home,
            [info_path, *info_path.parent.glob("*-val.arrow")],
        )["combined_sha256"]
    expected_dataset_artifacts["wikitext2"] = audit._hash_file_set(
        hf_home,
        [
            wikitext_info_path,
            *wikitext_info_path.parent.glob("*-test.arrow"),
        ],
    )["combined_sha256"]
    monkeypatch.setattr(
        audit,
        "EXPECTED_DATASET_ARTIFACT_SHA256",
        expected_dataset_artifacts,
    )
    project_paths = {
        "data_utils": repo / "utils" / "data_utils.py",
        "eval_utils": eval_utils_path,
        "realq_pipeline": repo / "realq" / "pipeline.py",
        "legacy_ptq": repo / "ptq.py",
    }
    monkeypatch.setattr(
        audit,
        "EXPECTED_PROJECT_SOURCE_SHA256",
        {
            name: audit._sha256_file(path)
            for name, path in project_paths.items()
        },
    )
    monkeypatch.setattr(
        audit,
        "EXPECTED_WIKITEXT_FILES",
        expected_wikitext_files,
    )
    monkeypatch.setattr(
        audit,
        "EXPECTED_WIKITEXT_COMBINED_SHA256",
        wikitext_hash["combined_sha256"],
    )

    return {
        "repo": repo,
        "venv": venv,
        "lm_eval_root": lm_eval_root,
        "task_root": task_root,
        "hf_home": hf_home,
        "wikitext_root": wikitext_root,
        "plan_path": plan_path,
        "eval_utils_path": eval_utils_path,
        "ceval_group_path": ceval_group_path,
        "ceval_info_paths": ceval_info_paths,
        "wikitext_info_path": wikitext_info_path,
        "task_info_paths": task_info_paths,
        "runtime_environment": plan["runtime_environment"],
        "project_paths": project_paths,
    }


def _run(
    fixture,
    *,
    versions=None,
    process_environment=None,
    process_cwd=None,
):
    return audit.run_audit(
        fixture["plan_path"],
        repo_root=fixture["repo"],
        lm_eval_root=fixture["lm_eval_root"],
        installed_versions=(
            dict(audit.EXPECTED_RUNTIME_VERSIONS)
            if versions is None
            else versions
        ),
        python_prefix=fixture["venv"],
        hf_cache_root=fixture["hf_home"],
        wikitext_root=fixture["wikitext_root"],
        process_cwd=(
            fixture["repo"] if process_cwd is None else process_cwd
        ),
        process_environment=(
            fixture["runtime_environment"]
            if process_environment is None
            else process_environment
        ),
    )


def test_valid_fixture_is_complete_and_manifest_is_atomic(
    formal_fixture,
    tmp_path,
):
    report = _run(formal_fixture)

    assert report["valid"] is True
    assert report["errors"] == []
    assert report["network_access"] is False
    assert report["dataset_loading"] is False
    assert list(report["lm_eval"]["resolved"]) == list(audit.EXPECTED_TASKS)
    assert report["lm_eval"]["ceval"]["leaf_count"] == 52
    assert report["lm_eval"]["ceval"]["sample_count"] == 1346
    assert report["lm_eval"]["python_execution_closure"]["matches_expected"]
    assert (
        report["lm_eval"]["function_execution_closure"]["reference_count"]
        == 1
    )
    assert all(
        item["dataset"]["artifacts"]["matches_expected"]
        for item in report["lm_eval"]["tasks"].values()
    )
    assert all(
        item["artifacts"]["matches_expected"]
        for item in report["lm_eval"]["ceval"]["datasets"].values()
    )
    assert report["wikitext2"]["local_cache"]["artifacts"][
        "matches_expected"
    ]
    assert all(
        item["matches_expected"]
        for item in report["project_evaluation_sources"].values()
    )
    assert report["evaluation_semantics"]["effective_zero_shot"] is True
    assert report["evaluation_semantics"]["apply_chat_template"] is False
    assert report["evaluation_semantics"]["limit"] is None
    assert report["evaluation_semantics"]["system_instruction"] is None
    assert report["evaluation_semantics"]["project_semantics"][
        "paper_qa_tasks"
    ] == list(audit.EXPECTED_TASKS)

    manifest = tmp_path / "formal-eval-audit.json"
    manifest.write_text("stale", encoding="utf-8")
    audit._atomic_write(manifest, report)

    assert manifest.read_bytes() == audit._canonical_bytes(report)
    assert json.loads(manifest.read_text(encoding="utf-8")) == report
    assert list(tmp_path.glob(".formal-eval-audit.json.*.tmp")) == []


def test_hash_file_set_is_invariant_to_root_and_input_order(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_paths = [
        _write(first / "nested" / "b.yaml", b"beta"),
        _write(first / "a.yaml", b"alpha"),
    ]
    second_paths = [
        _write(second / "a.yaml", b"alpha"),
        _write(second / "nested" / "b.yaml", b"beta"),
    ]

    first_hash = audit._hash_file_set(first, reversed(first_paths))
    second_hash = audit._hash_file_set(second, second_paths)

    assert first_hash == second_hash
    assert first_hash["combined_sha256"] == audit._stable_digest(
        first_hash["files"]
    )


def test_rejects_nonzero_task_fewshot(formal_fixture):
    piqa = formal_fixture["task_root"] / "piqa" / "piqa.yaml"
    piqa.write_text(
        piqa.read_text(encoding="utf-8") + "num_fewshot: 5\n",
        encoding="utf-8",
    )

    report = _run(formal_fixture)

    assert report["valid"] is False
    assert any(
        "piqa: num_fewshot must be None/0" in error
        for error in report["errors"]
    )


def test_rejects_chat_template_override_in_actual_qa_caller(formal_fixture):
    source = formal_fixture["eval_utils_path"].read_text(encoding="utf-8")
    formal_fixture["eval_utils_path"].write_text(
        source.replace(
            "task_manager=task_manager,",
            "task_manager=task_manager,\n            apply_chat_template=True,",
        ),
        encoding="utf-8",
    )

    report = _run(formal_fixture)

    assert report["valid"] is False
    assert any(
        "qa_eval effective apply_chat_template=True" in error
        for error in report["errors"]
    )


def test_rejects_incomplete_ceval_group_and_custom_override(formal_fixture):
    group = formal_fixture["ceval_group_path"]
    group.write_text(
        group.read_text(encoding="utf-8").replace(
            f"  - {audit.CEVAL_LEAVES[-1]}\n",
            "",
        ),
        encoding="utf-8",
    )
    _write(
        formal_fixture["repo"]
        / "datasets"
        / "lm_eval_configs"
        / "tasks"
        / "piqa.yaml",
        "task: piqa\n",
    )

    report = _run(formal_fixture)

    assert report["valid"] is False
    assert any(
        "exact reviewed 52-leaf list" in error for error in report["errors"]
    )
    assert any(
        "custom task definitions override formal protocol names" in error
        for error in report["errors"]
    )


def test_rejects_wrong_sample_count_and_unpinned_version(formal_fixture):
    info_path = next(
        path
        for path in formal_fixture["hf_home"].rglob("dataset_info.json")
        if json.loads(path.read_text(encoding="utf-8")).get("dataset_name")
        == "piqa"
    )
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["splits"]["validation"]["num_examples"] = 1
    info_path.write_text(json.dumps(info), encoding="utf-8")
    versions = dict(audit.EXPECTED_RUNTIME_VERSIONS)
    versions["lm-eval"] = "9.9.9"

    report = _run(formal_fixture, versions=versions)

    assert report["valid"] is False
    assert any(
        "piqa: sample_count=1, expected 1838" in error
        for error in report["errors"]
    )
    assert any(
        "installed runtime version lm-eval='9.9.9'" in error
        for error in report["errors"]
    )


def test_rejects_tampered_dataset_split_artifact(formal_fixture):
    info_path = formal_fixture["task_info_paths"]["piqa"]
    arrow_path = next(info_path.parent.glob("*-validation.arrow"))
    arrow_path.write_bytes(arrow_path.read_bytes() + b"tampered")

    report = _run(formal_fixture)

    assert report["valid"] is False
    assert any(
        "piqa: artifact SHA256=" in error for error in report["errors"]
    )
    assert not report["lm_eval"]["tasks"]["piqa"]["dataset"]["artifacts"][
        "matches_expected"
    ]


def test_rejects_tampered_lm_eval_function_python(formal_fixture):
    helper = formal_fixture["task_root"] / "hellaswag" / "utils.py"
    helper.write_text(
        helper.read_text(encoding="utf-8").replace(
            "return dataset",
            "return list(dataset)",
        ),
        encoding="utf-8",
    )

    report = _run(formal_fixture)

    assert report["valid"] is False
    assert any(
        "lm_eval Python closure SHA256 is" in error
        for error in report["errors"]
    )


@pytest.mark.parametrize(
    "relative",
    (
        "tasks/__init__.py",
        "utils.py",
        "api/task.py",
        "api/metrics.py",
        "api/filter.py",
        "filters/selection.py",
    ),
)
def test_rejects_tampered_lm_eval_runtime_execution_source(
    formal_fixture,
    relative,
):
    source = formal_fixture["lm_eval_root"] / relative
    source.write_text(
        source.read_text(encoding="utf-8") + "# tampered\n",
        encoding="utf-8",
    )

    report = _run(formal_fixture)

    assert report["valid"] is False
    assert any(
        "lm_eval Python closure SHA256 is" in error
        for error in report["errors"]
    )


def test_ast_rejects_task_protocol_even_if_source_hash_is_retrusted(
    formal_fixture,
    monkeypatch,
):
    source_path = formal_fixture["eval_utils_path"]
    source_path.write_text(
        source_path.read_text(encoding="utf-8").replace(
            "'social_iqa'",
            "'not_the_paper_task'",
        ),
        encoding="utf-8",
    )
    expected = dict(audit.EXPECTED_PROJECT_SOURCE_SHA256)
    expected["eval_utils"] = audit._sha256_file(source_path)
    monkeypatch.setattr(audit, "EXPECTED_PROJECT_SOURCE_SHA256", expected)

    report = _run(formal_fixture)

    assert report["valid"] is False
    assert any(
        "utils.eval_utils.PAPER_QA_TASKS=" in error
        for error in report["errors"]
    )


def test_ast_rejects_metric_extractor_even_if_source_hash_is_retrusted(
    formal_fixture,
    monkeypatch,
):
    source_path = formal_fixture["eval_utils_path"]
    source_path.write_text(
        source_path.read_text(encoding="utf-8").replace(
            '"acc_norm,none"',
            '"wrong_metric,none"',
        ),
        encoding="utf-8",
    )
    expected = dict(audit.EXPECTED_PROJECT_SOURCE_SHA256)
    expected["eval_utils"] = audit._sha256_file(source_path)
    monkeypatch.setattr(audit, "EXPECTED_PROJECT_SOURCE_SHA256", expected)

    report = _run(formal_fixture)

    assert report["valid"] is False
    assert any(
        "_task_accuracy must recognize both" in error
        for error in report["errors"]
    )


def test_rejects_noncanonical_cwd_and_active_online_environment(
    formal_fixture,
    tmp_path,
):
    environment = dict(formal_fixture["runtime_environment"])
    environment["HF_HUB_OFFLINE"] = "0"

    report = _run(
        formal_fixture,
        process_environment=environment,
        process_cwd=tmp_path,
    )

    assert report["valid"] is False
    assert any("active cwd" in error for error in report["errors"])
    assert any(
        "active environment HF_HUB_OFFLINE='0'" in error
        for error in report["errors"]
    )


def test_audit_does_not_import_dataset_stacks_or_open_network(
    formal_fixture,
    monkeypatch,
):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"datasets", "lm_eval"}:
            raise AssertionError(f"forbidden import: {name}")
        return original_import(name, *args, **kwargs)

    def reject_network(*_args, **_kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(socket, "create_connection", reject_network)

    report = _run(formal_fixture)

    assert report["valid"] is True


def test_cli_success_writes_the_exact_stdout_manifest(
    formal_fixture,
    tmp_path,
    monkeypatch,
    capsys,
):
    valid = _run(formal_fixture)
    manifest = tmp_path / "formal-eval-audit.json"
    monkeypatch.setattr(audit, "run_audit", lambda _path: valid)

    return_code = audit.main(
        [
            "--plan",
            str(formal_fixture["plan_path"]),
            "--write-manifest",
            str(manifest),
        ]
    )
    output = capsys.readouterr().out

    assert return_code == 0
    assert output == audit.canonical_json(valid)
    assert manifest.read_text(encoding="utf-8") == output


def test_cli_returns_nonzero_and_emits_canonical_json(
    formal_fixture,
    monkeypatch,
    capsys,
):
    invalid = {
        "schema_version": audit.SCHEMA_VERSION,
        "valid": False,
        "errors": ["intentional"],
        "network_access": False,
        "dataset_loading": False,
    }
    monkeypatch.setattr(audit, "run_audit", lambda _path: invalid)

    return_code = audit.main(["--plan", str(formal_fixture["plan_path"])])
    output = capsys.readouterr().out

    assert return_code == 1
    assert output == audit.canonical_json(invalid)
