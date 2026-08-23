from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
EXPERIMENT = ROOT / "experiments" / "lowbit_activation"
for directory in (str(TOOLS), str(EXPERIMENT)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

import formal_timed_campaign as timed_campaign
import formal_timing_adapter as timing
import lowbit_activation_campaign as base_campaign
import lowbit_activation_execute as base_executor
import lowbit_activation_gpu_hours as gpu_hours
import lowbit_activation_runner as runner


EXPECTED_NUMERICAL_SHA256 = (
    "53c916a9ababb36c4e9fc16a78d04a51d360ebe46794495f3b18c992249cb64a"
)


def _plan() -> dict:
    value = json.loads(
        (EXPERIMENT / "plan.json").read_text(encoding="utf-8")
    )
    for model in runner.MODEL_ORDER:
        for setting in runner.SETTING_ORDER:
            value["selected_grad_lr_by_model_setting"][model][setting] = 1e-5
    return value


def _args(
    *,
    method: str,
    phase: str,
    model: str = "llama3.2-3b",
    setting: str | None = "3W16A",
    target_phase: str | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        method=method,
        phase=phase,
        model=model,
        setting=setting,
        target_phase=target_phase,
    )


def _formalized(
    args: argparse.Namespace,
    rendered: runner.RenderedCommand,
) -> runner.RenderedCommand:
    argv = list(rendered.argv)
    synthetic_defaults = (
        ("--model", "modelzoo/Llama/Llama-3.2-3B"),
        ("--dataset", "wikitext2"),
        ("--nsamples", "256"),
        ("--seq_len", "2048"),
        ("--seed", "1"),
    )
    for flag, value in synthetic_defaults:
        if timing._option(argv, flag) is None:
            argv.extend([flag, value])
    rendered = runner.RenderedCommand(
        env=rendered.env,
        argv=argv,
        run_id=rendered.run_id,
        output_dir=rendered.output_dir,
    )
    return timing.prepare_formal_rendered(args, rendered)


def _manifest(
    *,
    run_id: str,
    argv: list[str],
    env: dict[str, str],
    effective_environment: dict[str, str],
    gpu_indices: list[int],
    status: str = "succeeded",
    exit_code: int = 0,
) -> dict:
    sha = "a" * 64
    return {
        "schema_version": 1,
        "execution_id": "execution-1",
        "run_id": run_id,
        "status": status,
        "exit_code": exit_code,
        "plan": {
            "sha256": sha,
            "sha256_at_end": sha,
            "changed_during_execution": False,
        },
        "model": {
            "complete": True,
            "combined_identity_sha256": "b" * 64,
        },
        "numerical_source_tree": {
            "combined_sha256": EXPECTED_NUMERICAL_SHA256,
            "combined_sha256_at_end": EXPECTED_NUMERICAL_SHA256,
            "changed_during_execution": False,
        },
        "command": {"argv": argv, "env": env},
        "effective_environment": effective_environment,
        "launch_gate": {
            "gpu": {"requested_gpu_indices": gpu_indices}
        },
        "process": {
            "pid": 123,
            "returncode": exit_code,
            "termination_signal": None,
            "launch_error": None,
            "wrapper_error": None,
        },
    }


def _rank_row(
    spec: timing.TimingSpec,
    provenance: timing.TimingProvenance,
    *,
    rank: int,
    elapsed: float,
) -> dict:
    start = 1_000_000_000
    end = start + round(elapsed * 1_000_000_000)
    site_sha = next(
        item["sha256"]
        for item in provenance.sources
        if item["name"] == "formal_timing_sitecustomize"
    )
    return {
        "schema_version": 1,
        "scope_id": timing.SCOPE_ID,
        "source_set_sha256": provenance.source_set_sha256,
        "sitecustomize_sha256": site_sha,
        "spec_sha256": spec.sha256,
        "run_id": spec.run_id,
        "mode": spec.mode,
        "method": spec.method,
        "stage": spec.stage,
        "model": spec.model,
        "setting": spec.setting,
        "phase": spec.phase,
        "target_phase": spec.target_phase,
        "rank": rank,
        "local_rank": rank,
        "world_size": len(spec.gpu_indices),
        "physical_gpu_index": spec.gpu_indices[rank],
        "clock": "time.perf_counter_ns",
        "started_monotonic_ns": start,
        "rotation_ended_monotonic_ns": start + 1,
        "ended_monotonic_ns": end,
        "elapsed_seconds": (end - start) / 1_000_000_000,
        "synchronized_start": True,
        "synchronized_end": True,
        "timing_status": "complete",
        "complete": True,
        "error": None,
        "cache_lookups": [],
        "cache_writes": [],
        "artifacts": [],
    }


def _write_fake_torch(root: Path) -> None:
    package = root / "torch"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "class _Cuda:\n"
        "    @staticmethod\n"
        "    def is_available(): return False\n"
        "    @staticmethod\n"
        "    def synchronize(): pass\n"
        "cuda = _Cuda()\n",
        encoding="utf-8",
    )
    (package / "distributed.py").write_text(
        "def is_available(): return True\n"
        "def is_initialized(): return False\n"
        "def barrier(): raise AssertionError('unexpected barrier')\n",
        encoding="utf-8",
    )


def _site_environment(
    *,
    fake_root: Path,
    output: Path,
    mode: str,
    method: str,
    stage: str,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(
                [str(timing.SITE_DIR), str(fake_root)]
            ),
            "CUDA_VISIBLE_DEVICES": "7",
            timing.ENV_KEYS["dir"]: str(output),
            timing.ENV_KEYS["expected_world"]: "1",
            timing.ENV_KEYS["source_set_sha"]: "c" * 64,
            timing.ENV_KEYS["site_sha"]: hashlib.sha256(
                timing.SITE_PATH.read_bytes()
            ).hexdigest(),
            timing.ENV_KEYS["spec_sha"]: "d" * 64,
            timing.ENV_KEYS["mode"]: mode,
            timing.ENV_KEYS["method"]: method,
            timing.ENV_KEYS["stage"]: stage,
            timing.ENV_KEYS["model"]: "model",
            timing.ENV_KEYS["setting"]: (
                "" if "precompute" in mode else "3W16A"
            ),
            timing.ENV_KEYS["phase"]: (
                "" if "precompute" in mode else "final"
            ),
            timing.ENV_KEYS["target_phase"]: (
                "final" if "precompute" in mode else ""
            ),
            timing.ENV_KEYS["run_id"]: f"run_{mode}",
        }
    )
    if mode != "guided_precompute":
        env.update(
            {
                "RANK": "0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "1",
            }
        )
    return env


class FormalTimingAdapterTest(unittest.TestCase):
    def _finalized_gptaq(
        self,
        output: Path,
        *,
        status: str = "succeeded",
        timing_status: str = "complete",
        elapsed: float = 2.5,
    ) -> tuple[
        base_executor.ExecutionResult,
        Path,
        timing.TimingSpec,
        timing.TimingProvenance,
    ]:
        run_id = "final_gptaq_llama3.2-3b_3w16a"
        argv = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc-per-node=1",
            "ptq.py",
            "--exp",
            run_id,
            "--w_method",
            "gptaq",
            "--rotate",
            "--lm_eval",
        ]
        rendered = runner.RenderedCommand(
            env={"CUDA_VISIBLE_DEVICES": "7"},
            argv=argv,
            run_id=run_id,
            output_dir=output,
        )
        rendered = _formalized(
            _args(method="gptaq", phase="final"), rendered
        )
        argv = list(rendered.argv)
        spec = timing.TimingSpec(
            mode="legacy_quantization",
            kind="quantization",
            method="gptaq",
            stage="gptaq_quantization",
            model="llama3.2-3b",
            setting="3W16A",
            phase="final",
            target_phase=None,
            run_id=run_id,
            gpu_indices=(7,),
        )
        _, provenance = timing.instrument_rendered(
            rendered, spec, repo_root=ROOT
        )
        row = _rank_row(
            spec, provenance, rank=0, elapsed=elapsed
        )
        if timing_status != "complete":
            row.update(
                {
                    "timing_status": timing_status,
                    "complete": False,
                    "synchronized_start": (
                        timing_status != "not_started"
                    ),
                    "synchronized_end": False,
                    "error": f"{timing_status} test failure",
                }
            )
            if timing_status == "not_started":
                row.update(
                    {
                        "started_monotonic_ns": None,
                        "rotation_ended_monotonic_ns": None,
                        "ended_monotonic_ns": None,
                        "elapsed_seconds": None,
                    }
                )
        timing._atomic_write_json(
            output / "phase_timing_rank0.json", row
        )
        exit_code = 0 if status == "succeeded" else 1
        manifest = _manifest(
            run_id=run_id,
            argv=argv,
            env=rendered.env,
            effective_environment=dict(provenance.environment),
            gpu_indices=[7],
            status=status,
            exit_code=exit_code,
        )
        manifest_path = output / base_executor.MANIFEST_FILENAME
        timing._atomic_write_json(manifest_path, manifest)
        if status == "succeeded":
            checkpoint = output / "checkpoint" / "model.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"formal-test-checkpoint")
        result = base_executor.ExecutionResult(
            manifest=manifest,
            exit_code=exit_code,
            manifest_path=manifest_path,
            log_path=output / base_executor.LOG_FILENAME,
        )
        finalized = timing.finalize_execution(
            result,
            rendered=rendered,
            spec=spec,
            provenance=provenance,
            plan={"output_root": str(output)},
            repo_root=ROOT,
        )
        return finalized, manifest_path, spec, provenance

    def test_numerical_tree_is_unchanged_by_external_adapter(self):
        tree = base_executor.collect_numerical_source_tree(cwd=ROOT)
        self.assertEqual(
            tree["combined_sha256"], EXPECTED_NUMERICAL_SHA256
        )
        self.assertNotIn(
            "experiments/lowbit_activation/formal_timing_adapter.py",
            {item["relative_path"] for item in tree["files"]},
        )

    def test_specs_cover_formal_methods_and_reject_tune(self):
        plan = _plan()
        cases = [
            (
                _args(method="realq", phase="final"),
                runner.render_realq(
                    plan,
                    phase="final",
                    model="llama3.2-3b",
                    setting_name="3W16A",
                    grad_lr=1e-5,
                    cuda_devices="0,1,2,3",
                ),
                ("quantization", "realq", "realq_quantization"),
            ),
            (
                _args(method="gptaq", phase="final"),
                runner.render_baseline(
                    plan,
                    method="gptaq",
                    model="llama3.2-3b",
                    setting_name="3W16A",
                    cuda_devices="4",
                ),
                ("quantization", "gptaq", "legacy_quantization"),
            ),
            (
                _args(method="guided_gptq", phase="final"),
                runner.render_baseline(
                    plan,
                    method="guided_gptq",
                    model="llama3.2-3b",
                    setting_name="3W16A",
                    cuda_devices="5",
                ),
                (
                    "quantization",
                    "guided_gptq",
                    "legacy_quantization",
                ),
            ),
            (
                _args(
                    method="realq_static",
                    phase="precompute",
                    setting=None,
                    target_phase="final",
                ),
                runner.render_realq_static_precompute(
                    plan,
                    target_phase="final",
                    model="llama3.2-3b",
                    cuda_devices="0,1,2,3",
                ),
                (
                    "shared_precompute",
                    "realq",
                    "realq_shared_precompute",
                ),
            ),
            (
                _args(
                    method="guided_saliency",
                    phase="precompute",
                    setting=None,
                ),
                runner.render_guided_saliency(
                    plan,
                    model="llama3.2-3b",
                    cuda_devices="6",
                ),
                (
                    "shared_precompute",
                    "guided_gptq",
                    "guided_precompute",
                ),
            ),
        ]
        for args, rendered, expected in cases:
            with self.subTest(method=args.method):
                rendered = _formalized(args, rendered)
                spec = timing.spec_from_executor_args(args, rendered)
                self.assertIsNotNone(spec)
                self.assertEqual(
                    (spec.kind, spec.method, spec.mode), expected
                )

        bf16 = runner.render_baseline(
            plan,
            method="bf16",
            model="llama3.2-3b",
            setting_name=None,
            cuda_devices="7",
        )
        self.assertIsNone(
            timing.spec_from_executor_args(
                _args(
                    method="bf16",
                    phase="final",
                    setting=None,
                ),
                bf16,
            )
        )

        tune = runner.render_realq(
            plan,
            phase="tune",
            model="llama3.2-3b",
            setting_name="3W16A",
            grad_lr=1e-5,
            cuda_devices="0",
        )
        with self.assertRaisesRegex(
            timing.TimingAdapterError, "refuses REAL-Q tune"
        ):
            timing.spec_from_executor_args(
                _args(method="realq", phase="tune"), tune
            )

    def test_child_overlay_preserves_reviewed_command_env(self):
        plan = _plan()
        rendered = runner.render_baseline(
            plan,
            method="gptaq",
            model="llama3.2-3b",
            setting_name="3W16A",
            cuda_devices="3",
        )
        rendered = _formalized(
            _args(method="gptaq", phase="final"), rendered
        )
        spec = timing.spec_from_executor_args(
            _args(method="gptaq", phase="final"), rendered
        )
        original_env = dict(rendered.env)
        timed, provenance = timing.instrument_rendered(
            rendered, spec, repo_root=ROOT
        )
        self.assertIs(timed, rendered)
        self.assertEqual(timed.env, original_env)
        key = timing.ENV_KEYS["mode"]
        previous = os.environ.get(key)
        with timing.child_environment(provenance):
            self.assertEqual(
                os.environ[key], "legacy_quantization"
            )
            self.assertEqual(
                os.environ[timing.ENV_KEYS["spec_sha"]],
                spec.sha256,
            )
        self.assertEqual(os.environ.get(key), previous)

    def test_formal_campaign_substitutes_only_applicable_executor(self):
        base_command = [
            sys.executable,
            str(ROOT / "tools" / "lowbit_activation_execute.py"),
            "--execute",
        ]
        controller = timed_campaign.FormalTimedCampaign.__new__(
            timed_campaign.FormalTimedCampaign
        )
        with mock.patch.object(
            base_campaign.Campaign,
            "_executor_cli",
            return_value=base_command,
        ), mock.patch.object(
            controller,
            "_assert_timing_gate",
            return_value=timed_campaign._timing_gate_now(),
        ):
            formal = controller._executor_cli(
                {
                    "spec": {
                        "kind": "baseline",
                        "method": "gptaq",
                        "phase": "final",
                    }
                },
                [0],
            )
            bf16 = controller._executor_cli(
                {
                    "spec": {
                        "kind": "baseline",
                        "method": "bf16",
                        "phase": "final",
                    }
                },
                [1],
            )
            tune = controller._executor_cli(
                {
                    "spec": {
                        "kind": "realq",
                        "method": "realq",
                        "phase": "tune",
                    }
                },
                [2],
            )
        self.assertEqual(
            Path(formal[1]), timing.TIMED_EXECUTOR_PATH
        )
        self.assertEqual(bf16, base_command)
        self.assertEqual(tune, base_command)

    def test_formal_success_waits_for_outer_exit_and_revalidates_tamper(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            finalized, manifest_path, _, _ = self._finalized_gptaq(
                output
            )
            self.assertEqual(finalized.exit_code, 0)
            gate = timed_campaign._timing_gate_now()
            task = {
                "task_id": "formal-gptaq",
                "status": "RUNNING",
                "spec": {
                    "kind": "baseline",
                    "method": "gptaq",
                    "phase": "final",
                    "model": "llama3.2-3b",
                    "setting": "3W16A",
                },
                "gpu_count": 1,
                "gpus": [7],
                "manifest": str(manifest_path),
            }
            controller = timed_campaign.FormalTimedCampaign.__new__(
                timed_campaign.FormalTimedCampaign
            )
            controller.state = {
                "campaign_id": "test",
                "formal": {"timing_gate": gate, "models": {}},
                "tasks": {task["task_id"]: task},
                "static_caches": {},
                "event_seq": 0,
                "anomaly_keys": [],
            }
            controller.processes = {}
            controller.execute = False
            controller.preview_events = []

            with mock.patch.object(
                controller,
                "_task_process_alive",
                side_effect=[True, False],
            ), mock.patch.object(
                controller, "_validate_successful_task"
            ) as validate_result:
                controller._reconcile_one_task(task)
                self.assertEqual(task["status"], "RUNNING")
                validate_result.assert_not_called()
                controller._reconcile_one_task(task)
                self.assertEqual(task["status"], "SUCCEEDED")
                validate_result.assert_called_once()
            self.assertEqual(
                task["formal_timing"]["timing_status"], "complete"
            )

            rank_path = output / "phase_timing_rank0.json"
            rank_path.write_bytes(rank_path.read_bytes() + b"tamper\n")
            with mock.patch.object(
                controller,
                "_task_process_alive",
                return_value=False,
            ):
                controller._reconcile_one_task(task)
            self.assertEqual(task["status"], "INVALID_RESULT")
            self.assertEqual(
                task["failure_class"], "formal_timing_invalid"
            )

    def test_producer_dependency_is_not_released_in_finalize_race(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / base_executor.MANIFEST_FILENAME
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "succeeded",
                        "exit_code": 0,
                        "timestamps": {"ended_utc": "now"},
                    }
                ),
                encoding="utf-8",
            )
            task = {
                "task_id": "producer",
                "status": "RUNNING",
                "spec": {
                    "kind": "realq_static",
                    "method": "realq_static",
                    "target_phase": "final",
                    "model": "llama3.2-3b",
                },
                "gpu_count": 4,
                "gpus": [0, 1, 2, 3],
                "manifest": str(manifest_path),
            }
            entry = {
                "status": "PENDING",
                "producer_task": "producer",
                "validation": None,
            }
            controller = timed_campaign.FormalTimedCampaign.__new__(
                timed_campaign.FormalTimedCampaign
            )
            controller.state = {
                "campaign_id": "test",
                "formal": {"timing_gate": {}, "models": {}},
                "tasks": {"producer": task},
                "static_caches": {
                    "final/llama3.2-3b": entry
                },
                "event_seq": 0,
                "anomaly_keys": [],
            }
            controller.processes = {}
            controller.execute = False
            controller.preview_events = []
            component = {
                "component_id": "a" * 64,
                "timing_status": "complete",
                "wall_seconds": 1.0,
                "gpu_hour_status": "exact",
                "allocated_gpu_hours": 4 / 3600,
                "gpu_hours_lower_bound": 4 / 3600,
                "missing_ranks": [],
                "sidecar": "phase_timing.json",
                "sidecar_sha256": "b" * 64,
            }

            def mark_ready(_task, _path):
                entry["status"] = "READY"

            with mock.patch.object(
                controller,
                "_task_process_alive",
                side_effect=[True, False],
            ), mock.patch.object(
                controller,
                "_validate_task_timing",
                return_value=component,
            ), mock.patch.object(
                controller,
                "_validate_successful_task",
                side_effect=mark_ready,
            ) as validate_result:
                controller._reconcile_one_task(task)
                self.assertEqual(entry["status"], "PENDING")
                validate_result.assert_not_called()
                controller._reconcile_one_task(task)
                self.assertEqual(task["status"], "SUCCEEDED")
                self.assertEqual(entry["status"], "READY")

    def test_frozen_timed_state_blocks_base_controller_bypass(self):
        gate = timed_campaign._timing_gate_now()
        base_identity = {
            "numerical_source_sha256": "1" * 64,
            "runner_sha256": "2" * 64,
            "executor_sha256": "3" * 64,
            "campaign_sha256": gate["base_campaign_sha256"],
            "results_sha256": "5" * 64,
        }
        frozen = timed_campaign._formal_source_identity(
            base_identity, gate
        )
        self.assertEqual(
            frozen["campaign_sha256"], gate["base_campaign_sha256"]
        )
        self.assertEqual(
            frozen["formal_timed_campaign_sha256"],
            gate["controller_sha256"],
        )
        controller = base_campaign.Campaign.__new__(
            base_campaign.Campaign
        )
        controller.state = {
            "campaign_id": "test",
            "source_identity": frozen,
            "status": "FORMAL",
            "blockers": [],
            "event_seq": 0,
            "anomaly_keys": [],
        }
        controller.execute = False
        controller.preview_events = []
        with mock.patch.object(
            base_campaign.Campaign,
            "_current_source_identity",
            return_value=base_identity,
        ):
            self.assertFalse(controller._check_source_identity())
        self.assertEqual(controller.state["status"], "NEEDS_USER_ACTION")
        self.assertIn(
            "runner/executor/numerical source changed",
            controller.state["blockers"][0],
        )
        drifted_base = dict(base_identity)
        drifted_base["campaign_sha256"] = "f" * 64
        with self.assertRaisesRegex(
            base_campaign.CampaignBlocked,
            "base campaign source",
        ):
            timed_campaign._formal_source_identity(
                drifted_base,
                gate,
            )

    def test_world4_wall_time_uses_max_rank_not_sum(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_id = "final_gptaq_llama3.2-3b_3w16a"
            rendered = runner.RenderedCommand(
                env={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
                argv=[
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--nproc-per-node=4",
                    "ptq.py",
                    "--exp",
                    run_id,
                    "--w_method",
                    "gptaq",
                    "--rotate",
                    "--lm_eval",
                ],
                run_id=run_id,
                output_dir=output,
            )
            rendered = _formalized(
                _args(method="gptaq", phase="final"), rendered
            )
            spec = timing.TimingSpec(
                mode="legacy_quantization",
                kind="quantization",
                method="gptaq",
                stage="gptaq_quantization",
                model="llama3.2-3b",
                setting="3W16A",
                phase="final",
                target_phase=None,
                run_id=run_id,
                gpu_indices=(0, 1, 2, 3),
            )
            _, provenance = timing.instrument_rendered(
                rendered, spec, repo_root=ROOT
            )
            for rank, elapsed in enumerate((1.0, 1.2, 1.1, 1.3)):
                timing._atomic_write_json(
                    output
                    / timing.RANK_EVIDENCE_TEMPLATE.format(rank=rank),
                    _rank_row(
                        spec,
                        provenance,
                        rank=rank,
                        elapsed=elapsed,
                    ),
                )
            rows, _ = timing._rank_evidence(
                output_dir=output,
                spec=spec,
                provenance=provenance,
                require_complete=True,
            )
            manifest = _manifest(
                run_id=run_id,
                argv=rendered.argv,
                env=rendered.env,
                effective_environment=dict(provenance.environment),
                gpu_indices=[0, 1, 2, 3],
            )
            component = timing._build_component(
                manifest=manifest,
                rendered=rendered,
                spec=spec,
                plan={"output_root": str(output)},
                rows=rows,
                repo_root=ROOT,
            )
            self.assertAlmostEqual(component["wall_seconds"], 1.3)
            self.assertAlmostEqual(
                component["allocated_gpu_hours"], 1.3 * 4 / 3600
            )

    def test_failed_world4_missing_rank_is_only_a_lower_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_id = "final_gptaq_llama3.2-3b_3w16a"
            rendered = runner.RenderedCommand(
                env={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
                argv=[
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--nproc-per-node=4",
                    "ptq.py",
                    "--exp",
                    run_id,
                    "--w_method",
                    "gptaq",
                    "--rotate",
                    "--lm_eval",
                ],
                run_id=run_id,
                output_dir=output,
            )
            rendered = _formalized(
                _args(method="gptaq", phase="final"), rendered
            )
            spec = timing.TimingSpec(
                mode="legacy_quantization",
                kind="quantization",
                method="gptaq",
                stage="gptaq_quantization",
                model="llama3.2-3b",
                setting="3W16A",
                phase="final",
                target_phase=None,
                run_id=run_id,
                gpu_indices=(0, 1, 2, 3),
            )
            _, provenance = timing.instrument_rendered(
                rendered, spec, repo_root=ROOT
            )
            for rank, elapsed in enumerate((1.0, 1.2, 1.1)):
                row = _rank_row(
                    spec,
                    provenance,
                    rank=rank,
                    elapsed=elapsed,
                )
                row.update(
                    {
                        "timing_status": "partial",
                        "complete": False,
                        "synchronized_end": False,
                        "error": "SIGTERM",
                    }
                )
                timing._atomic_write_json(
                    output
                    / timing.RANK_EVIDENCE_TEMPLATE.format(rank=rank),
                    row,
                )
            rows, _ = timing._rank_evidence(
                output_dir=output,
                spec=spec,
                provenance=provenance,
                require_complete=False,
            )
            manifest = _manifest(
                run_id=run_id,
                argv=rendered.argv,
                env=rendered.env,
                effective_environment=dict(provenance.environment),
                gpu_indices=[0, 1, 2, 3],
                status="failed",
                exit_code=1,
            )
            component = timing._build_partial_component(
                manifest=manifest,
                spec=spec,
                rows=rows,
            )
            self.assertEqual(component["gpu_hour_status"], "lower_bound")
            self.assertIsNone(component["allocated_gpu_hours"])
            self.assertEqual(component["missing_ranks"], [3])
            self.assertAlmostEqual(
                component["gpu_hours_lower_bound"],
                (1.0 + 1.2 + 1.1) / 3600,
            )
            self.assertNotAlmostEqual(
                component["gpu_hours_lower_bound"],
                1.2 * 4 / 3600,
            )

    def test_successful_child_without_rank_evidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_id = "final_gptaq_llama3.2-3b_3w16a"
            rendered = runner.RenderedCommand(
                env={"CUDA_VISIBLE_DEVICES": "0"},
                argv=[
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--nproc-per-node=1",
                    "ptq.py",
                    "--exp",
                    run_id,
                    "--w_method",
                    "gptaq",
                    "--rotate",
                    "--lm_eval",
                ],
                run_id=run_id,
                output_dir=output,
            )
            rendered = _formalized(
                _args(method="gptaq", phase="final"), rendered
            )
            spec = timing.TimingSpec(
                mode="legacy_quantization",
                kind="quantization",
                method="gptaq",
                stage="gptaq_quantization",
                model="llama3.2-3b",
                setting="3W16A",
                phase="final",
                target_phase=None,
                run_id=run_id,
                gpu_indices=(0,),
            )
            _, provenance = timing.instrument_rendered(
                rendered, spec, repo_root=ROOT
            )
            manifest = _manifest(
                run_id=run_id,
                argv=rendered.argv,
                env=rendered.env,
                effective_environment=dict(provenance.environment),
                gpu_indices=[0],
            )
            path = output / base_executor.MANIFEST_FILENAME
            timing._atomic_write_json(path, manifest)
            result = base_executor.ExecutionResult(
                manifest=manifest,
                exit_code=0,
                manifest_path=path,
                log_path=output / base_executor.LOG_FILENAME,
            )
            finalized = timing.finalize_execution(
                result,
                rendered=rendered,
                spec=spec,
                provenance=provenance,
                plan={"output_root": str(output)},
                repo_root=ROOT,
            )
            self.assertEqual(finalized.exit_code, 2)
            self.assertEqual(
                finalized.manifest["status"], "wrapper_failed"
            )
            self.assertIn(
                "rank timing evidence",
                finalized.manifest["phase_timing_error"],
            )

    def test_failed_attempt_pins_strict_partial_gpu_hour_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            finalized, manifest_path, _, _ = self._finalized_gptaq(
                output,
                status="failed",
                timing_status="partial",
                elapsed=12.0,
            )
            self.assertEqual(finalized.exit_code, 1)
            self.assertEqual(finalized.manifest["status"], "failed")
            component = gpu_hours._parse_component(
                finalized.manifest,
                manifest_path,
                primary=False,
            )
            self.assertEqual(component["timing_status"], "partial")
            self.assertEqual(component["wall_seconds"], 12.0)
            self.assertAlmostEqual(
                component["allocated_gpu_hours"], 12 / 3600
            )
            self.assertTrue(component["timing_provenance_strict"])
            (output / base_executor.LOG_FILENAME).write_text(
                "torch.cuda.OutOfMemoryError: CUDA out of memory\n",
                encoding="utf-8",
            )
            task = {
                "task_id": "oom-formal",
                "status": "RUNNING",
                "spec": {
                    "kind": "baseline",
                    "method": "gptaq",
                    "phase": "final",
                    "model": "llama3.2-3b",
                    "setting": "3W16A",
                },
                "gpu_count": 1,
                "gpus": [7],
                "manifest": str(manifest_path),
            }
            controller = timed_campaign.FormalTimedCampaign.__new__(
                timed_campaign.FormalTimedCampaign
            )
            controller.state = {
                "campaign_id": "test",
                "formal": {
                    "timing_gate": timed_campaign._timing_gate_now(),
                    "models": {},
                },
                "tasks": {"oom-formal": task},
                "static_caches": {},
                "event_seq": 0,
                "anomaly_keys": [],
            }
            controller.processes = {}
            controller.execute = False
            controller.preview_events = []
            with mock.patch.object(
                controller,
                "_task_process_alive",
                return_value=False,
            ):
                controller._reconcile_one_task(task)
            self.assertEqual(task["status"], "OOM")
            self.assertEqual(
                task["formal_timing"]["timing_status"], "partial"
            )

    def test_gptaq_sidecar_round_trips_strict_gpu_hour_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_id = "final_gptaq_llama3.2-3b_3w16a"
            argv = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nproc-per-node=1",
                "ptq.py",
                "--exp",
                run_id,
                "--w_method",
                "gptaq",
                "--rotate",
                "--lm_eval",
            ]
            rendered = runner.RenderedCommand(
                env={"CUDA_VISIBLE_DEVICES": "7"},
                argv=argv,
                run_id=run_id,
                output_dir=output,
            )
            rendered = _formalized(
                _args(method="gptaq", phase="final"), rendered
            )
            spec = timing.TimingSpec(
                mode="legacy_quantization",
                kind="quantization",
                method="gptaq",
                stage="gptaq_quantization",
                model="llama3.2-3b",
                setting="3W16A",
                phase="final",
                target_phase=None,
                run_id=run_id,
                gpu_indices=(7,),
            )
            _, provenance = timing.instrument_rendered(
                rendered, spec, repo_root=ROOT
            )
            timing._atomic_write_json(
                output / "phase_timing_rank0.json",
                _rank_row(
                    spec, provenance, rank=0, elapsed=2.5
                ),
            )
            manifest = _manifest(
                run_id=run_id,
                argv=rendered.argv,
                env=rendered.env,
                effective_environment=dict(provenance.environment),
                gpu_indices=[7],
            )
            manifest_path = output / base_executor.MANIFEST_FILENAME
            timing._atomic_write_json(manifest_path, manifest)
            checkpoint = output / "checkpoint" / "model.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"formal-test-checkpoint")
            result = base_executor.ExecutionResult(
                manifest=manifest,
                exit_code=0,
                manifest_path=manifest_path,
                log_path=output / base_executor.LOG_FILENAME,
            )
            finalized = timing.finalize_execution(
                result,
                rendered=rendered,
                spec=spec,
                provenance=provenance,
                plan={"output_root": str(output)},
                repo_root=ROOT,
            )
            self.assertEqual(finalized.exit_code, 0)
            component = gpu_hours._parse_component(
                finalized.manifest, manifest_path
            )
            self.assertEqual(component["method"], "gptaq")
            self.assertEqual(component["wall_seconds"], 2.5)
            sidecar = timing._load_json_no_duplicates(
                output / timing.SIDECAR_FILENAME
            )
            sources = sidecar["timing_provenance"]["sources"]
            self.assertEqual(len(sources), 7)
            self.assertIn(
                "lowbit_activation_campaign",
                {source["name"] for source in sources},
            )
            for source in sources:
                snapshot = output / source["snapshot_path"]
                self.assertEqual(
                    hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                    source["sha256"],
                )

    def test_sitecustomize_legacy_hooks_exclude_eval_on_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "fake"
            (fake / "utils").mkdir(parents=True)
            _write_fake_torch(fake)
            (fake / "utils" / "__init__.py").write_text(
                "",
                encoding="utf-8",
            )
            (fake / "utils" / "rotation_utils.py").write_text(
                "calls=[]\n"
                "def fuse_layer_norms(*args, **kwargs):\n"
                "    calls.append('fuse')\n"
                "    return 'rotated'\n",
                encoding="utf-8",
            )
            (fake / "utils" / "eval_utils.py").write_text(
                "calls=[]\n"
                "def kl_ppl_eval(*args, **kwargs):\n"
                "    calls.append('eval')\n"
                "    return 'evaluated'\n",
                encoding="utf-8",
            )
            (fake / "utils" / "checkpoint_utils.py").write_text(
                "calls=[]\n"
                "def save_quantized_checkpoint(*args, **kwargs):\n"
                "    calls.append('save')\n"
                "    return 'saved'\n",
                encoding="utf-8",
            )
            output = root / "timing"
            env = _site_environment(
                fake_root=fake,
                output=output,
                mode="legacy_quantization",
                method="gptaq",
                stage="gptaq_quantization",
            )
            evidence_path = output / "phase_timing_rank0.json"
            code = (
                "from pathlib import Path;"
                "from utils import checkpoint_utils, eval_utils, rotation_utils;"
                f"evidence=Path({str(evidence_path)!r});"
                "assert rotation_utils.fuse_layer_norms() == 'rotated';"
                "assert not evidence.exists();"
                "assert checkpoint_utils.save_quantized_checkpoint() == "
                "'saved';"
                "assert evidence.exists();"
                "assert eval_utils.kl_ppl_eval() == 'evaluated';"
                "assert rotation_utils.calls == ['fuse'];"
                "assert checkpoint_utils.calls == ['save'];"
                "assert eval_utils.calls == ['eval']"
            )
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            evidence = json.loads(
                (output / "phase_timing_rank0.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(evidence["complete"])
            self.assertTrue(evidence["synchronized_end"])
            self.assertIsNone(evidence["error"])
            self.assertEqual(evidence["physical_gpu_index"], 7)

    def test_multirank_sigterm_persists_partial_rank_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "fake"
            (fake / "utils").mkdir(parents=True)
            _write_fake_torch(fake)
            (fake / "utils" / "__init__.py").write_text(
                "", encoding="utf-8"
            )
            (fake / "utils" / "rotation_utils.py").write_text(
                "def fuse_layer_norms(*args, **kwargs): return 'rotated'\n",
                encoding="utf-8",
            )
            (fake / "utils" / "eval_utils.py").write_text(
                "def kl_ppl_eval(*args, **kwargs): return 'eval'\n",
                encoding="utf-8",
            )
            (fake / "utils" / "checkpoint_utils.py").write_text(
                "def save_quantized_checkpoint(*args, **kwargs):\n"
                "    return 'saved'\n",
                encoding="utf-8",
            )
            output = root / "timing"
            for rank in (0, 1):
                env = _site_environment(
                    fake_root=fake,
                    output=output,
                    mode="legacy_quantization",
                    method="gptaq",
                    stage="gptaq_quantization",
                )
                env.update(
                    {
                        "CUDA_VISIBLE_DEVICES": "2,3",
                        timing.ENV_KEYS["expected_world"]: "2",
                        "RANK": str(rank),
                        "LOCAL_RANK": str(rank),
                        "WORLD_SIZE": "2",
                    }
                )
                code = (
                    "import os, signal;"
                    "from utils import rotation_utils;"
                    "assert rotation_utils.fuse_layer_norms() == 'rotated';"
                    "os.kill(os.getpid(), signal.SIGTERM)"
                )
                completed = subprocess.run(
                    [sys.executable, "-c", code],
                    cwd=root,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                self.assertEqual(
                    completed.returncode, -15, completed.stdout
                )
            for rank, physical in ((0, 2), (1, 3)):
                evidence = json.loads(
                    (
                        output / f"phase_timing_rank{rank}.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(evidence["timing_status"], "partial")
                self.assertFalse(evidence["complete"])
                self.assertFalse(evidence["synchronized_end"])
                self.assertIn("SIGTERM", evidence["error"])
                self.assertEqual(
                    evidence["physical_gpu_index"], physical
                )

    def test_sitecustomize_realq_shared_precompute_hooks_on_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "fake"
            _write_fake_torch(fake)
            package = fake / "realq_benchmark"
            (package / "precompute").mkdir(parents=True)
            (package / "__init__.py").write_text(
                "",
                encoding="utf-8",
            )
            (package / "akv.py").write_text(
                "def setup_unaware_post_quant(*args, **kwargs):\n"
                "    return 'post'\n",
                encoding="utf-8",
            )
            (package / "precompute" / "cache.py").write_text(
                "def cache_path(cache_dir, key, world_size, rank):\n"
                "    return f'{cache_dir}/{key}_world{world_size}_rank{rank}.pt'\n"
                "def try_load(cache_dir, key, world_size, rank):\n"
                "    return None\n"
                "def save(cache_dir, key, world_size, rank, payload):\n"
                "    return cache_path(cache_dir, key, world_size, rank)\n",
                encoding="utf-8",
            )
            (package / "precompute" / "__init__.py").write_text(
                "from . import cache\n"
                "def run(*args, **kwargs):\n"
                "    value = cache.try_load('cache', 'key', 1, 0)\n"
                "    assert value is None\n"
                "    cache.save('cache', 'key', 1, 0, {'value': 1})\n"
                "    return 'static'\n",
                encoding="utf-8",
            )
            (package / "pipeline.py").write_text(
                "from . import akv, precompute\n"
                "def _maybe_rotate(*args, **kwargs):\n"
                "    return 'rotated'\n",
                encoding="utf-8",
            )
            output = root / "timing"
            env = _site_environment(
                fake_root=fake,
                output=output,
                mode="realq_shared_precompute",
                method="realq",
                stage="realq_static",
            )
            code = (
                "from realq_benchmark import pipeline;"
                "assert pipeline._maybe_rotate() == 'rotated';"
                "assert pipeline.precompute.run() == 'static'"
            )
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            evidence = json.loads(
                (output / "phase_timing_rank0.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(evidence["complete"])
            self.assertEqual(
                evidence["cache_lookups"],
                [
                    {
                        "hit": False,
                        "key": "key",
                        "path": "cache/key_world1_rank0.pt",
                    }
                ],
            )
            self.assertEqual(
                evidence["cache_writes"],
                [
                    {
                        "key": "key",
                        "path": "cache/key_world1_rank0.pt",
                    }
                ],
            )

    def test_sitecustomize_realq_quantization_ends_at_post_quant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "fake"
            _write_fake_torch(fake)
            package = fake / "realq_benchmark"
            (package / "precompute").mkdir(parents=True)
            (package / "__init__.py").write_text(
                "", encoding="utf-8"
            )
            (package / "akv.py").write_text(
                "def setup_unaware_post_quant(*args, **kwargs):\n"
                "    return 'post-quant'\n",
                encoding="utf-8",
            )
            (package / "precompute" / "cache.py").write_text(
                "def cache_path(cache_dir, key, world_size, rank):\n"
                "    return f'{cache_dir}/{key}.pt'\n"
                "def try_load(*args, **kwargs): return object()\n"
                "def save(*args, **kwargs): return 'cache.pt'\n",
                encoding="utf-8",
            )
            (package / "precompute" / "__init__.py").write_text(
                "from . import cache\n"
                "def run(*args, **kwargs): return 'precompute'\n",
                encoding="utf-8",
            )
            (package / "pipeline.py").write_text(
                "from . import akv, precompute\n"
                "def _maybe_rotate(*args, **kwargs): return 'rotated'\n",
                encoding="utf-8",
            )
            output = root / "timing"
            env = _site_environment(
                fake_root=fake,
                output=output,
                mode="realq_quantization",
                method="realq",
                stage="realq_quantization",
            )
            evidence_path = output / "phase_timing_rank0.json"
            code = (
                "from pathlib import Path;"
                "from realq_benchmark import pipeline;"
                f"evidence=Path({str(evidence_path)!r});"
                "assert pipeline._maybe_rotate() == 'rotated';"
                "assert not evidence.exists();"
                "assert pipeline.precompute.run() == 'precompute';"
                "assert not evidence.exists();"
                "assert pipeline.akv.setup_unaware_post_quant() == "
                "'post-quant';"
                "assert evidence.exists()"
            )
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            evidence = json.loads(
                evidence_path.read_text(encoding="utf-8")
            )
            self.assertEqual(evidence["timing_status"], "complete")
            self.assertTrue(evidence["complete"])

    def test_sitecustomize_guided_precompute_records_artifacts_on_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "fake"
            _write_fake_torch(fake)
            (fake / "utils").mkdir(parents=True)
            (fake / "utils" / "__init__.py").write_text(
                "",
                encoding="utf-8",
            )
            (fake / "utils" / "data_utils.py").write_text(
                "calls=[]\n"
                "def get_tokens(*args, **kwargs):\n"
                "    calls.append('tokens')\n"
                "    return [1]\n",
                encoding="utf-8",
            )
            (fake / "utils" / "gradients.py").write_text(
                "calls=[]\n"
                "def get_gradients(*args, **kwargs):\n"
                "    calls.append('gradients')\n"
                "    return ['gradient']\n",
                encoding="utf-8",
            )
            output = root / "timing"
            env = _site_environment(
                fake_root=fake,
                output=output,
                mode="guided_precompute",
                method="guided_gptq",
                stage="guided_saliency",
            )
            code = (
                "from utils import data_utils, gradients;"
                "assert data_utils.get_tokens() == [1];"
                "assert gradients.get_gradients("
                "None, [1], 'grad.pt', 'saliency', 4"
                ") == ['gradient'];"
                "assert data_utils.calls == ['tokens'];"
                "assert gradients.calls == ['gradients']"
            )
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            evidence = json.loads(
                (output / "phase_timing_rank0.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(evidence["complete"])
            self.assertEqual(
                evidence["artifacts"],
                [
                    {
                        "gradients_path": "grad.pt",
                        "num_groups": 4,
                        "saliency_path": "saliency",
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
