from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "tools" / "lowbit_activation_runner.py"
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "lowbit_activation_runner",
    RUNNER_PATH,
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = runner
RUNNER_SPEC.loader.exec_module(runner)

EXECUTOR_PATH = ROOT / "tools" / "lowbit_activation_execute.py"
EXECUTOR_SPEC = importlib.util.spec_from_file_location(
    "lowbit_activation_execute",
    EXECUTOR_PATH,
)
assert EXECUTOR_SPEC is not None and EXECUTOR_SPEC.loader is not None
executor = importlib.util.module_from_spec(EXECUTOR_SPEC)
sys.modules[EXECUTOR_SPEC.name] = executor
EXECUTOR_SPEC.loader.exec_module(executor)


class LowbitActivationExecuteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.cwd = Path(self.temporary_directory.name)
        self.output_root = self.cwd / "outputs"
        self.output_dir = self.output_root / "fake-run"
        self.model_dir = self.cwd / "model"
        self.model_dir.mkdir()
        (self.model_dir / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["FakeForCausalLM"],
                    "hidden_size": 8,
                    "model_type": "fake",
                }
            ),
            encoding="utf-8",
        )
        (self.model_dir / "model-00001-of-00001.safetensors").write_bytes(
            b"not-a-real-safetensors-file"
        )
        (self.model_dir / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 27},
                    "weight_map": {
                        "model.embed_tokens.weight": (
                            "model-00001-of-00001.safetensors"
                        )
                    },
                }
            ),
            encoding="utf-8",
        )
        self.plan = {
            "output_root": str(self.output_root),
            "models": {"fake": str(self.model_dir)},
        }
        self.plan_path = self.cwd / "plan.json"
        self.plan_path.write_text(
            json.dumps(self.plan, indent=2),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def rendered(
        self,
        code: str,
        *,
        env: dict[str, str] | None = None,
    ):
        return runner.RenderedCommand(
            env=env or {"EXECUTOR_TEST": "present"},
            argv=[
                sys.executable,
                "-c",
                code,
                "--model",
                str(self.model_dir),
            ],
            run_id="fake-run",
            output_dir=self.output_dir,
        )

    def metadata_mocks(self):
        return mock.patch.multiple(
            executor,
            collect_gpu_snapshot=mock.DEFAULT,
            collect_git_snapshot=mock.DEFAULT,
            collect_package_versions=mock.DEFAULT,
            _require_runtime_preflight=mock.DEFAULT,
            _require_current_runtime=mock.DEFAULT,
            _require_rendered_executable=mock.DEFAULT,
        )

    def install_metadata_mock_values(self, patched):
        patched["collect_gpu_snapshot"].return_value = {
            "query": {"available": False, "error": "test"},
            "list": {"available": False, "error": "test"},
        }
        patched["collect_git_snapshot"].return_value = {
            "head": {"available": False, "error": "test"},
            "status": {"available": False, "error": "test"},
        }
        patched["collect_package_versions"].return_value = {
            "torch": {"distribution": "torch", "version": None}
        }
        preflight_identity = {
            "configured": True,
            "path": str(self.plan_path),
            "sha256": executor._sha256_file(self.plan_path),
        }
        patched["_require_runtime_preflight"].return_value = preflight_identity
        patched["_require_current_runtime"].return_value = {
            "python": sys.executable,
            "distributions": {},
            "model": {},
            "gpu": {},
        }
        patched["_require_rendered_executable"].return_value = Path(
            sys.executable
        )

    def test_default_preview_is_read_only_and_complete(self):
        rendered = self.rendered("print('must not launch')")
        with self.metadata_mocks() as patched:
            self.install_metadata_mock_values(patched)
            result = executor.run_rendered(
                rendered,
                plan=self.plan,
                plan_path=self.plan_path,
                cwd=self.cwd,
            )

        self.assertEqual(result.manifest["status"], "dry_run")
        self.assertFalse(result.manifest["execute_requested"])
        self.assertIsNone(result.exit_code)
        self.assertFalse(self.output_root.exists())
        self.assertEqual(result.manifest["command"]["argv"], rendered.argv)
        self.assertEqual(
            result.manifest["command"]["env"],
            {"EXECUTOR_TEST": "present"},
        )
        self.assertEqual(result.manifest["paths"]["cwd"], str(self.cwd))
        self.assertEqual(result.manifest["plan"]["content"], self.plan)
        self.assertEqual(
            result.manifest["plan"]["sha256"],
            executor._sha256_file(self.plan_path),
        )
        self.assertTrue(result.manifest["model"]["complete"])
        self.assertIn(
            "combined_identity_sha256",
            result.manifest["model"],
        )
        self.assertIn("config", result.manifest["model"])
        self.assertEqual(len(result.manifest["model"]["shards"]), 1)

    def test_execute_merges_streams_and_atomically_finishes_manifest(self):
        code = (
            "import os,sys;"
            "print('stdout=' + os.environ['EXECUTOR_TEST']);"
            "print('stderr-line', file=sys.stderr)"
        )
        with self.metadata_mocks() as patched:
            self.install_metadata_mock_values(patched)
            result = executor.run_rendered(
                self.rendered(code),
                plan=self.plan,
                plan_path=self.plan_path,
                cwd=self.cwd,
                execute=True,
            )

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.manifest["status"], "succeeded")
        self.assertEqual(result.manifest["exit_code"], 0)
        self.assertTrue(result.manifest_path.is_file())
        self.assertTrue(result.log_path.is_file())
        log = result.log_path.read_text(encoding="utf-8")
        self.assertIn("stdout=present", log)
        self.assertIn("stderr-line", log)
        on_disk = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["status"], "succeeded")
        self.assertEqual(
            on_disk["command"]["argv"],
            result.manifest["command"]["argv"],
        )
        self.assertIsInstance(on_disk["process"]["pid"], int)
        self.assertEqual(on_disk["process"]["returncode"], 0)
        self.assertTrue(on_disk["safety"]["lock_acquired"])
        self.assertFalse(
            list(self.output_dir.glob(".execution_manifest.json.*.tmp"))
        )

    def test_execute_rejects_wrong_required_venv_before_writing(self):
        self.plan["python_environment"] = {
            "venv": str(self.cwd / "required-venv"),
            "activation_required": True,
        }
        self.plan_path.write_text(
            json.dumps(self.plan, indent=2),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            executor.ExecutionError,
            "required experiment venv is not active",
        ):
            executor.run_rendered(
                self.rendered("print('must not launch')"),
                plan=self.plan,
                plan_path=self.plan_path,
                cwd=self.cwd,
                execute=True,
            )

        self.assertFalse(self.output_root.exists())

    def test_execute_requires_configured_runtime_preflight_before_writing(self):
        self.plan["resolutions"] = {
            "runtime_preflight_manifest": "preflight/missing.json",
        }
        self.plan_path.write_text(
            json.dumps(self.plan, indent=2),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            executor.ExecutionError,
            "runtime preflight manifest is missing",
        ):
            executor.run_rendered(
                self.rendered("print('must not launch')"),
                plan=self.plan,
                plan_path=self.plan_path,
                cwd=self.cwd,
                execute=True,
            )

        self.assertFalse(self.output_root.exists())

    def test_execute_rejects_missing_runtime_preflight_configuration(self):
        with self.assertRaisesRegex(
            executor.ExecutionError,
            "not configured",
        ):
            executor.run_rendered(
                self.rendered("print('must not launch')"),
                plan=self.plan,
                plan_path=self.plan_path,
                cwd=self.cwd,
                execute=True,
            )

        self.assertFalse(self.output_root.exists())

    def test_rendered_executable_must_be_exact_venv_python(self):
        plan = {
            "python_environment": {
                "venv": str(self.cwd / ".venv"),
                "activation_required": True,
            }
        }
        rendered = self.rendered("print('unused')")
        with self.assertRaisesRegex(
            executor.ExecutionError,
            "outside the exact experiment venv",
        ):
            executor._require_rendered_executable(rendered, plan)

    def test_current_distribution_gate_rejects_post_preflight_upgrade(self):
        plan = {
            "runtime_versions": {
                "torch": "2.9.1",
                "torch_runtime": "2.9.1+cu128",
                "transformers": "4.56.2",
                "lm-eval": "0.4.4",
            }
        }
        actual = {
            "torch": "2.9.1",
            "transformers": "4.57.0",
            "lm-eval": "0.4.4",
        }
        with mock.patch.object(
            executor.importlib.metadata,
            "version",
            side_effect=lambda name: actual[name],
        ):
            with self.assertRaisesRegex(
                executor.ExecutionError,
                "transformers version changed after preflight",
            ):
                executor._current_distribution_versions(plan)

    def test_current_gpu_gate_rejects_out_of_range_requested_index(self):
        rendered = self.rendered("print('unused')")
        rendered = runner.RenderedCommand(
            env={"CUDA_VISIBLE_DEVICES": "8"},
            argv=rendered.argv,
            run_id=rendered.run_id,
            output_dir=rendered.output_dir,
        )
        plan = {
            "runtime_requirements": {
                "minimum_gpu_count": 8,
                "minimum_gpu_memory_gib": 120,
            },
            "runtime_versions": {"torch_runtime": "2.9.1+cu128"},
        }
        rows = "\n".join(
            f"{index}, 143000, 142000" for index in range(8)
        )
        completed = mock.Mock(returncode=0, stdout=rows, stderr="")
        with mock.patch.object(
            executor.subprocess,
            "run",
            return_value=completed,
        ):
            with self.assertRaisesRegex(
                executor.ExecutionError,
                "not currently eligible",
            ):
                executor._current_gpu_gate(
                    rendered,
                    plan=plan,
                    cwd=self.cwd,
                )

    def test_preflight_identity_change_before_launch_is_rejected(self):
        rendered = self.rendered("print('must not launch')")
        with self.metadata_mocks() as patched:
            self.install_metadata_mock_values(patched)
            first = patched["_require_runtime_preflight"].return_value
            patched["_require_runtime_preflight"].side_effect = [
                first,
                {**first, "sha256": "f" * 64},
            ]
            with self.assertRaisesRegex(
                executor.ExecutionError,
                "changed after the initial execution gate",
            ):
                executor.run_rendered(
                    rendered,
                    plan=self.plan,
                    plan_path=self.plan_path,
                    cwd=self.cwd,
                    execute=True,
                )

        self.assertFalse(self.output_root.exists())

    def test_runtime_preflight_gate_accepts_sorted_json_key_order(self):
        tasks = ("task_b", "task_a")
        models = ("model_b", "model_a")
        preflight_path = self.cwd / "preflight.json"
        plan = {
            "resolutions": {
                "runtime_preflight_manifest": str(preflight_path),
                "canoe_job_id": "j-test",
            },
            "python_environment": {"venv": "/required/.venv"},
            "runtime_versions": {
                "torch": "2.9.1",
                "torch_runtime": "2.9.1+cu128",
                "transformers": "4.56.2",
                "lm-eval": "0.4.4",
            },
            "paper_zero_shot_tasks": list(tasks),
            "models": {name: str(self.cwd / name) for name in models},
            "model_signatures": {
                name: {
                    "model_type": "fake",
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "architectures": ["FakeForCausalLM"],
                }
                for name in models
            },
        }
        plan_path = self.cwd / "preflight-plan.json"
        plan_path.write_text(
            json.dumps(plan, indent=2),
            encoding="utf-8",
        )
        payload = {
            "schema_version": 1,
            "valid": True,
            "load_task_datasets": True,
            "plan": {"sha256": executor._sha256_file(plan_path)},
            "canoe": {
                "valid": True,
                "expected_job_id": "j-test",
                "supplied_job_id": "j-test",
                "hostname": executor.socket.gethostname(),
            },
            "runtime": {
                "python_environment": {
                    "valid": True,
                    "expected_venv": "/required/.venv",
                },
                "distributions": {
                    "torch": "2.9.1",
                    "transformers": "4.56.2",
                    "lm-eval": "0.4.4",
                },
                "cuda": {
                    "torch_version": "2.9.1+cu128",
                    "requirement_check": {"passed": True},
                },
            },
            "lm_eval": {
                "resolved": {task: [task] for task in reversed(tasks)},
                "datasets_loaded": {
                    task: True for task in reversed(tasks)
                },
            },
            "models": {
                model: {
                    "exists": True,
                    "errors": [],
                    "path": str((self.cwd / model).resolve()),
                    "actual_signature": {
                        "model_type": "fake",
                        "hidden_size": 8,
                        "num_hidden_layers": 1,
                        "architectures": ["FakeForCausalLM"],
                    },
                }
                for model in reversed(models)
            },
        }
        preflight_path.write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )

        identity = executor._require_runtime_preflight(
            plan,
            plan_path=plan_path,
            cwd=self.cwd,
        )

        self.assertTrue(identity["configured"])
        self.assertEqual(identity["sha256"], executor._sha256_file(preflight_path))

    def test_nonzero_child_exit_is_recorded_and_returned(self):
        with self.metadata_mocks() as patched:
            self.install_metadata_mock_values(patched)
            result = executor.run_rendered(
                self.rendered("raise SystemExit(7)"),
                plan=self.plan,
                plan_path=self.plan_path,
                cwd=self.cwd,
                execute=True,
            )

        self.assertEqual(result.exit_code, 7)
        self.assertEqual(result.manifest["status"], "failed")
        self.assertEqual(result.manifest["exit_code"], 7)
        self.assertEqual(
            json.loads(result.manifest_path.read_text(encoding="utf-8"))[
                "exit_code"
            ],
            7,
        )

    def test_launch_failure_is_atomically_recorded(self):
        rendered = runner.RenderedCommand(
            env={},
            argv=[
                str(self.cwd / "definitely-missing-executable"),
                "--model",
                str(self.model_dir),
            ],
            run_id="fake-run",
            output_dir=self.output_dir,
        )
        with self.metadata_mocks() as patched:
            self.install_metadata_mock_values(patched)
            with self.assertRaises(executor.ExecutionError):
                executor.run_rendered(
                    rendered,
                    plan=self.plan,
                    plan_path=self.plan_path,
                    cwd=self.cwd,
                    execute=True,
                )

        manifest_path = self.output_dir / executor.MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "launch_failed")
        self.assertIsNone(manifest["exit_code"])
        self.assertIn("FileNotFoundError", manifest["process"]["launch_error"])
        self.assertTrue((self.output_dir / executor.LOG_FILENAME).is_file())

    def test_existing_artifacts_refuse_launch_without_overwrite(self):
        self.output_dir.mkdir(parents=True)
        sentinel = self.output_dir / "existing-result.json"
        sentinel.write_text("keep-me", encoding="utf-8")
        marker = self.cwd / "child-launched"
        code = (
            "from pathlib import Path;"
            f"Path({str(marker)!r}).write_text('bad')"
        )
        with self.metadata_mocks() as patched:
            self.install_metadata_mock_values(patched)
            with self.assertRaises(executor.ExistingRunError):
                executor.run_rendered(
                    self.rendered(code),
                    plan=self.plan,
                    plan_path=self.plan_path,
                    cwd=self.cwd,
                    execute=True,
                )

        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep-me")
        self.assertFalse(marker.exists())
        self.assertFalse((self.output_dir / executor.MANIFEST_FILENAME).exists())

    def test_concurrent_lock_refuses_second_launcher(self):
        with self.metadata_mocks() as patched:
            self.install_metadata_mock_values(patched)
            with executor.execution_lock(self.output_dir):
                with self.assertRaises(executor.ConcurrentRunError):
                    executor.run_rendered(
                        self.rendered("print('must not launch')"),
                        plan=self.plan,
                        plan_path=self.plan_path,
                        cwd=self.cwd,
                        execute=True,
                    )

    def test_plan_change_after_render_is_rejected(self):
        rendered = self.rendered("print('must not launch')")
        original_prepare = executor.prepare_manifest

        def prepare_then_change(*args, **kwargs):
            manifest, output_dir = original_prepare(*args, **kwargs)
            self.plan_path.write_text('{"changed": true}', encoding="utf-8")
            return manifest, output_dir

        with self.metadata_mocks() as patched:
            self.install_metadata_mock_values(patched)
            with mock.patch.object(
                executor,
                "prepare_manifest",
                side_effect=prepare_then_change,
            ):
                with self.assertRaises(executor.ExecutionError):
                    executor.run_rendered(
                        rendered,
                        plan=self.plan,
                        plan_path=self.plan_path,
                        cwd=self.cwd,
                        execute=True,
                    )

        self.assertFalse((self.output_dir / executor.MANIFEST_FILENAME).exists())
        self.assertFalse((self.output_dir / executor.LOG_FILENAME).exists())

    def test_sensitive_inherited_environment_is_fingerprinted_not_disclosed(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": "do-not-record-me"}):
            with self.metadata_mocks() as patched:
                self.install_metadata_mock_values(patched)
                result = executor.run_rendered(
                    self.rendered("print('unused')"),
                    plan=self.plan,
                    plan_path=self.plan_path,
                    cwd=self.cwd,
                )
        token_record = result.manifest["effective_environment"]["HF_TOKEN"]
        self.assertTrue(token_record["redacted"])
        self.assertNotIn("do-not-record-me", json.dumps(result.manifest))
        self.assertIn(
            "HF_TOKEN",
            result.manifest["effective_environment_redacted_keys"],
        )

    def test_sensitive_explicit_environment_is_not_leaked_in_shell_preview(self):
        with self.metadata_mocks() as patched:
            self.install_metadata_mock_values(patched)
            result = executor.run_rendered(
                self.rendered(
                    "print('unused')",
                    env={"HF_TOKEN": "explicit-secret"},
                ),
                plan=self.plan,
                plan_path=self.plan_path,
                cwd=self.cwd,
            )
        command = result.manifest["command"]
        self.assertTrue(command["env"]["HF_TOKEN"]["redacted"])
        self.assertEqual(command["env_redacted_keys"], ["HF_TOKEN"])
        self.assertIsNone(command["shell_preview"])
        self.assertTrue(command["shell_preview_omitted_for_sensitive_env"])
        self.assertNotIn("explicit-secret", json.dumps(result.manifest))

    def test_output_escape_is_rejected_even_in_preview(self):
        escaped = runner.RenderedCommand(
            env={},
            argv=[
                sys.executable,
                "-c",
                "print('must not launch')",
                "--model",
                str(self.model_dir),
            ],
            run_id="escape",
            output_dir=self.cwd / "outside",
        )
        with self.assertRaises(executor.ExecutionError):
            executor.run_rendered(
                escaped,
                plan=self.plan,
                plan_path=self.plan_path,
                cwd=self.cwd,
            )

    def test_model_identity_changes_when_shard_content_changes(self):
        rendered = self.rendered("print('unused')")
        first = executor.collect_model_identity(rendered, cwd=self.cwd)
        shard = self.model_dir / "model-00001-of-00001.safetensors"
        shard.write_bytes(b"different-content")
        stat = shard.stat()
        os.utime(
            shard,
            ns=(stat.st_atime_ns, stat.st_mtime_ns + 1),
        )
        second = executor.collect_model_identity(rendered, cwd=self.cwd)
        self.assertNotEqual(
            first["combined_identity_sha256"],
            second["combined_identity_sha256"],
        )

    def test_cli_dispatches_realq_static_for_exact_target_phase_cache(self):
        plan = runner.load_plan(
            ROOT / "experiments" / "lowbit_activation" / "plan.json"
        )
        cases = (
            ("tune", "0", "-1", "256"),
            ("final", "0,1,2,3", "128", "128"),
        )
        for target_phase, cuda_devices, expected_group, expected_block in cases:
            with self.subTest(target_phase=target_phase):
                if target_phase == "final":
                    plan["selected_grad_lr_by_model_setting"]["qwen3-4b"][
                        "3W16A"
                    ] = 5e-5
                args = executor._build_cli().parse_args(
                    [
                        "--phase",
                        "precompute",
                        "--method",
                        "realq_static",
                        "--target-phase",
                        target_phase,
                        "--model",
                        "qwen3-4b",
                        "--cuda-devices",
                        cuda_devices,
                    ]
                )
                rendered = executor._render_from_args(args, plan)
                command = rendered.shell()
                self.assertEqual(
                    rendered.run_id,
                    f"precompute_realq_static_{target_phase}_qwen3-4b",
                )
                self.assertIn(f"--w_groupsize {expected_group}", command)
                self.assertIn(f"--blocksize {expected_block}", command)
                self.assertIn("--exit_after_precompute true", command)
                # Static precompute also materializes the exact WikiText-2
                # reference-logit cache used by the KL/PPL pass.
                self.assertIn("--skip_eval false", command)
                self.assertIn("--lm_eval false", command)

    def test_cli_allows_static_global_loss_oom_override(self):
        plan = runner.load_plan(
            ROOT / "experiments" / "lowbit_activation" / "plan.json"
        )
        args = executor._build_cli().parse_args(
            [
                "--phase",
                "precompute",
                "--method",
                "realq_static",
                "--target-phase",
                "tune",
                "--model",
                "qwen3-32b",
                "--cuda-devices",
                "0",
                "--override",
                "global_loss_bsz=4",
            ]
        )
        rendered = executor._render_from_args(args, plan)
        self.assertIn("--global_loss_bsz 4", rendered.shell())
        self.assertTrue(rendered.run_id.endswith("_global_loss_bsz4"))

    def test_attempt_identity_allows_immutable_retry_without_overwrite(self):
        rendered = runner.RenderedCommand(
            env={},
            argv=[
                sys.executable,
                "-c",
                "print('retry')",
                "--exp",
                "fake-run",
                "--model",
                str(self.model_dir),
            ],
            run_id="fake-run",
            output_dir=self.output_dir,
        )
        retry = executor._with_attempt_identity(rendered, 2)
        self.assertEqual(retry.run_id, "fake-run_attempt2")
        self.assertEqual(retry.output_dir, self.output_root / "fake-run_attempt2")
        self.assertEqual(
            retry.argv[retry.argv.index("--exp") + 1],
            "fake-run_attempt2",
        )
        self.assertEqual(
            retry.env["LOWBIT_ACTIVATION_ATTEMPT_INDEX"],
            "2",
        )
        first = executor._with_attempt_identity(rendered, 1)
        self.assertEqual(
            first.env["LOWBIT_ACTIVATION_ATTEMPT_INDEX"],
            "1",
        )

        with self.assertRaisesRegex(
            runner.PlanError,
            "positive integer",
        ):
            executor._with_attempt_identity(rendered, 0)
        with self.assertRaisesRegex(
            runner.PlanError,
            "cannot exceed 20",
        ):
            executor._with_attempt_identity(rendered, 21)

    def test_cli_requires_target_phase_for_realq_static(self):
        plan = runner.load_plan(
            ROOT / "experiments" / "lowbit_activation" / "plan.json"
        )
        args = executor._build_cli().parse_args(
            [
                "--phase",
                "precompute",
                "--method",
                "realq_static",
                "--model",
                "qwen3-4b",
                "--cuda-devices",
                "0",
            ]
        )
        with self.assertRaisesRegex(
            runner.PlanError,
            "requires --target-phase",
        ):
            executor._render_from_args(args, plan)

    def test_cli_rejects_target_phase_for_every_other_method(self):
        plan = runner.load_plan(
            ROOT / "experiments" / "lowbit_activation" / "plan.json"
        )
        cases = {
            "realq": [
                "--phase",
                "final",
                "--setting",
                "3W16A",
                "--grad-lr",
                "1e-5",
            ],
            "gptaq": ["--phase", "final", "--setting", "3W16A"],
            "guided_gptq": ["--phase", "final", "--setting", "3W16A"],
            "guided_saliency": ["--phase", "precompute"],
            "bf16": ["--phase", "final"],
        }
        for method, method_args in cases.items():
            with self.subTest(method=method):
                args = executor._build_cli().parse_args(
                    [
                        *method_args,
                        "--method",
                        method,
                        "--target-phase",
                        "tune",
                        "--model",
                        "qwen3-4b",
                        "--cuda-devices",
                        "0",
                    ]
                )
                with self.assertRaisesRegex(
                    runner.PlanError,
                    "only for realq_static",
                ):
                    executor._render_from_args(args, plan)


if __name__ == "__main__":
    unittest.main()
