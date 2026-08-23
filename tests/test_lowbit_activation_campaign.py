from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lowbit_activation_campaign",
    ROOT / "tools" / "lowbit_activation_campaign.py",
)
assert SPEC is not None and SPEC.loader is not None
campaign = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = campaign
SPEC.loader.exec_module(campaign)


def _policy() -> dict:
    return {
        "strategy": "adaptive_bracket_then_canonical_refine",
        "canonical_mantissas": [1, 2, 3, 5, 7],
        "boundary_expansion_batch_size": 2,
        "reserved_refinement_attempts": 3,
        "zero_boundary_probe_ratios": [0.1, 0.3, 0.7],
        "fixed_upper_lr_ceiling": None,
        "require_complete_coarse_round": True,
        "require_local_refinement": True,
    }


COARSE = [0.0, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]


def _metrics(best_lr: float) -> dict[float, dict[str, float]]:
    return {
        lr: {
            "kl": 0.01 + abs(math.log10(lr) - math.log10(best_lr))
            if lr > 0
            else 10.0,
            "ppl": 10.0
            + (
                abs(math.log10(lr) - math.log10(best_lr))
                if lr > 0
                else 10.0
            ),
        }
        for lr in COARSE
    }


class SearchPolicyTests(unittest.TestCase):
    def test_canonical_upper_expansion_has_no_fixed_small_ceiling(self):
        self.assertEqual(
            campaign.canonical_sequence_values(
                start_exclusive=5e-3,
                direction="up",
                count=4,
                mantissas=[1, 2, 3, 5, 7],
            ),
            [7e-3, 1e-2, 2e-2, 3e-2],
        )
        much_higher = campaign.canonical_sequence_values(
            start_exclusive=7e3,
            direction="up",
            count=2,
            mantissas=[1, 2, 3, 5, 7],
        )
        self.assertEqual(much_higher, [1e4, 2e4])

    def test_canonical_local_refinement_matches_reviewed_example(self):
        values = campaign.local_refinement_candidates(
            [1e-5, 5e-5, 1e-4],
            5e-5,
            mantissas=[1, 2, 3, 5, 7],
        )
        self.assertEqual(values, [2e-5, 3e-5, 7e-5])

    def test_interior_coarse_minimum_requires_local_refinement(self):
        decision = campaign.search_decision(
            metrics=_metrics(5e-5),
            coarse_lrs=COARSE,
            attempted_count=9,
            pending_count=0,
            refinement_done=False,
            zero_probes_done=[],
            policy=_policy(),
            max_attempts=20,
        )
        self.assertEqual(decision["action"], "local_refine")
        self.assertEqual(decision["candidates"], [2e-5, 3e-5, 7e-5])

    def test_first_positive_winner_has_bounded_zero_edge_refinement(self):
        decision = campaign.search_decision(
            metrics=_metrics(1e-6),
            coarse_lrs=COARSE,
            attempted_count=9,
            pending_count=0,
            refinement_done=False,
            zero_probes_done=[],
            policy=_policy(),
            max_attempts=20,
        )
        self.assertEqual(decision["action"], "local_refine")
        self.assertEqual(
            decision["candidates"],
            [5e-7, 1.5e-6, 3.5e-6],
        )

    def test_upper_boundary_expands_two_and_reserves_three(self):
        metrics = {
            lr: {"kl": float(len(COARSE) - index), "ppl": 20.0}
            for index, lr in enumerate(COARSE)
        }
        decision = campaign.search_decision(
            metrics=metrics,
            coarse_lrs=COARSE,
            attempted_count=9,
            pending_count=0,
            refinement_done=False,
            zero_probes_done=[],
            policy=_policy(),
            max_attempts=20,
        )
        self.assertEqual(decision["action"], "expand_upper_boundary")
        self.assertEqual(decision["candidates"], [7e-3, 1e-2])

        exhausted = campaign.search_decision(
            metrics=metrics,
            coarse_lrs=COARSE,
            attempted_count=17,
            pending_count=0,
            refinement_done=False,
            zero_probes_done=[],
            policy=_policy(),
            max_attempts=20,
        )
        self.assertEqual(exhausted["action"], "needs_user_action")
        self.assertIn("reserve", exhausted["reason"])

    def test_attempt_twenty_cannot_skip_required_refinement(self):
        decision = campaign.search_decision(
            metrics=_metrics(5e-5),
            coarse_lrs=COARSE,
            attempted_count=20,
            pending_count=0,
            refinement_done=False,
            zero_probes_done=[],
            policy=_policy(),
            max_attempts=20,
        )
        self.assertEqual(decision["action"], "needs_user_action")
        self.assertNotIn("selected_lr", decision)

    def test_completed_refinement_can_select_at_attempt_twenty(self):
        metrics = _metrics(5e-5)
        metrics.update(
            {
                2e-5: {"kl": 0.3, "ppl": 11.0},
                3e-5: {"kl": 0.2, "ppl": 10.5},
                7e-5: {"kl": 0.4, "ppl": 11.5},
            }
        )
        metrics[5e-5] = {"kl": 0.1, "ppl": 10.0}
        decision = campaign.search_decision(
            metrics=metrics,
            coarse_lrs=COARSE,
            attempted_count=20,
            pending_count=0,
            refinement_done=True,
            zero_probes_done=[],
            policy=_policy(),
            max_attempts=20,
        )
        self.assertEqual(decision["action"], "select")
        self.assertEqual(decision["selected_lr"], 5e-5)

    def test_zero_boundary_uses_plan_ratios_then_refuses_fake_best(self):
        metrics = _metrics(1e-6)
        metrics[0.0] = {"kl": 0.0, "ppl": 1.0}
        first = campaign.search_decision(
            metrics=metrics,
            coarse_lrs=COARSE,
            attempted_count=9,
            pending_count=0,
            refinement_done=False,
            zero_probes_done=[],
            policy=_policy(),
            max_attempts=20,
        )
        self.assertEqual(first["action"], "expand_zero_boundary")
        self.assertEqual(first["candidates"], [1e-7, 3e-7])

        for lr in (1e-7, 3e-7, 7e-7):
            metrics[lr] = {"kl": 1.0, "ppl": 2.0}
        stopped = campaign.search_decision(
            metrics=metrics,
            coarse_lrs=COARSE,
            attempted_count=12,
            pending_count=0,
            refinement_done=False,
            zero_probes_done=[1e-7, 3e-7, 7e-7],
            policy=_policy(),
            max_attempts=20,
        )
        self.assertEqual(stopped["action"], "needs_user_action")
        self.assertNotIn("selected_lr", stopped)


class PureSafetyTests(unittest.TestCase):
    def test_agent_ready_requires_literal_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "AGENT.md"
            path.write_text("venv 还没安装好\n", encoding="utf-8")
            ready, _ = campaign.agent_ready(path, "依赖 READY")
            self.assertFalse(ready)
            path.write_text("依赖 READY\n", encoding="utf-8")
            ready, _ = campaign.agent_ready(path, "依赖 READY")
            self.assertTrue(ready)

    def test_venv_symlink_entries_are_checked_lexically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv = root / ".venv"
            bin_dir = venv / "bin"
            bin_dir.mkdir(parents=True)
            target = Path(sys.executable).resolve()
            for name in ("python", "python3", "torchrun"):
                (bin_dir / name).symlink_to(target)
                self.assertFalse(
                    str((bin_dir / name).resolve()).startswith(str(venv))
                )
            plan = {
                "python_environment": {
                    "venv": str(venv),
                    "activation_required": True,
                }
            }
            with (
                mock.patch.dict(
                    campaign.os.environ,
                    {"VIRTUAL_ENV": str(venv)},
                    clear=False,
                ),
                mock.patch.object(campaign.sys, "prefix", str(venv)),
                mock.patch.object(
                    campaign.shutil,
                    "which",
                    side_effect=lambda name: str(bin_dir / name),
                ),
            ):
                valid, details = campaign.venv_ready(plan)
            self.assertTrue(valid, details)
            self.assertTrue(all(details["tools_inside_venv"].values()))

    def test_gpu_assignment_preserves_contiguous_four_gpu_groups(self):
        self.assertEqual(
            campaign.assign_gpus(
                [0, 1, 2, 3, 5, 6, 7],
                count=4,
                contiguous=True,
            ),
            (0, 1, 2, 3),
        )
        self.assertIsNone(
            campaign.assign_gpus(
                [0, 1, 3, 4, 6, 7],
                count=4,
                contiguous=True,
            )
        )

    def test_oom_and_cache_miss_classification(self):
        self.assertEqual(
            campaign.classify_failure(
                "torch.cuda.OutOfMemoryError: CUDA out of memory"
            ),
            "oom",
        )
        self.assertEqual(
            campaign.classify_failure("required static cache hit: cache miss"),
            "cache_miss",
        )
        self.assertEqual(campaign.classify_failure("ValueError"), "other")

    def test_static_cache_validation_requires_every_rank_same_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for rank in range(4):
                path = root / f"samekey_world4_rank{rank}.pt"
                path.write_bytes(b"nonempty")
                paths.append(path)
            valid, payload = campaign.validate_static_cache_paths(
                paths, expected_world_size=4
            )
            self.assertTrue(valid)
            self.assertEqual(payload["cache_key"], "samekey")
            paths.pop()
            valid, payload = campaign.validate_static_cache_paths(
                paths, expected_world_size=4
            )
            self.assertFalse(valid)
            self.assertTrue(payload["errors"])

    def test_curve_anomalies_are_machine_readable(self):
        anomalies = campaign.metric_anomalies(
            [
                (1e-6, 0.1, 10.0),
                (2e-6, 10.0, 9.0),
                (3e-6, -0.1, 11.0),
            ]
        )
        codes = {item["code"] for item in anomalies}
        self.assertIn("negative_exact_kl", codes)
        self.assertIn("curve_spike", codes)
        self.assertIn("ppl_kl_divergence", codes)


class ResumeAndDryRunTests(unittest.TestCase):
    def _plan(self, root: Path) -> Path:
        plan = json.loads(
            (ROOT / "experiments" / "lowbit_activation" / "plan.json").read_text(
                encoding="utf-8"
            )
        )
        plan["output_root"] = "out"
        plan["static_cache_root"] = "cache/static"
        plan["legacy_cache_root"] = "cache/legacy"
        path = root / "plan.json"
        path.write_text(
            json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path

    def _fake_tune_manifest(
        self,
        root: Path,
        plan: dict,
        *,
        leaf: str,
        attempt_index: int,
    ) -> None:
        output = root / "out" / leaf
        output.mkdir(parents=True)
        setting = plan["settings"]["3W16A"]
        argv = [
            "python3",
            "-m",
            "realq.ptq",
            "--model",
            plan["models"]["llama3.2-3b"],
            "--w_bits",
            str(setting["w_bits"]),
            "--a_bits",
            str(setting["a_bits"]),
            "--k_bits",
            str(setting["k_bits"]),
            "--v_bits",
            str(setting["v_bits"]),
            "--grad_lr",
            "1e-5",
            "--global_loss_bsz",
            "8",
            "--hessian_accum_bsz",
            "32",
            "--backward_bsz",
            "16",
            "--final_layer_backward_bsz",
            "16",
        ]
        payload = {
            "run_id": (
                "tune_realq_llama3.2-3b_3w16a_lr0p00001"
                + (f"_attempt{attempt_index}" if attempt_index > 1 else "")
            ),
            "status": "failed",
            "exit_code": 1,
            "command": {
                "argv": argv,
                "env": {
                    "LOWBIT_ACTIVATION_ATTEMPT_INDEX": str(attempt_index)
                },
            },
            "process": {"pid": None},
            "timestamps": {
                "prepared_utc": "2026-01-01T00:00:00+00:00",
                "started_utc": "2026-01-01T00:00:01+00:00",
                "ended_utc": "2026-01-01T00:00:02+00:00",
            },
        }
        (output / "execution_manifest.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        (output / "execution.log").write_text(
            "CUDA out of memory\n", encoding="utf-8"
        )

    def test_manifest_scan_rebuilds_attempt_count_after_state_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = self._plan(root)
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self._fake_tune_manifest(
                root, plan, leaf="attempt-one", attempt_index=1
            )
            self._fake_tune_manifest(
                root, plan, leaf="attempt-two", attempt_index=2
            )
            agent = root / "AGENT.md"
            agent.write_text("READY\n", encoding="utf-8")

            first = campaign.Campaign(
                plan_path=plan_path,
                state_path=root / "state-a.json",
                events_path=root / "events-a.jsonl",
                agent_path=agent,
                ready_marker="READY",
                execute=False,
                poll_seconds=1,
                repo_root=root,
            )
            second = campaign.Campaign(
                plan_path=plan_path,
                state_path=root / "state-b.json",
                events_path=root / "events-b.jsonl",
                agent_path=agent,
                ready_marker="READY",
                execute=False,
                poll_seconds=1,
                repo_root=root,
            )
            try:
                key = "llama3.2-3b/3W16A"
                self.assertEqual(first.state["tuning"][key]["attempt_count"], 2)
                self.assertEqual(second.state["tuning"][key]["attempt_count"], 2)
                imported = [
                    task
                    for task in second.state["tasks"].values()
                    if task.get("imported")
                ]
                self.assertEqual(len(imported), 2)
            finally:
                first.close()
                second.close()

    def test_event_ledger_counts_launch_without_manifest_after_state_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = self._plan(root)
            plan_sha = campaign._sha256_file(plan_path)
            events = root / "events.jsonl"
            output_dir = root / "out" / "launch-failed-before-manifest"
            event = {
                "schema_version": 1,
                "seq": 1,
                "timestamp_utc": "2026-01-01T00:00:00+00:00",
                "campaign_id": "old-campaign",
                "category": "task",
                "code": "task_launching",
                "details": {
                    "task_id": "realq:old",
                    "plan_sha256": plan_sha,
                    "kind": "realq",
                    "method": "realq",
                    "phase": "tune",
                    "model": "llama3.2-3b",
                    "setting": "3W16A",
                    "grad_lr": 1e-5,
                    "profile_level": 0,
                    "generation": 0,
                    "attempt_index": 1,
                    "output_dir": str(output_dir),
                },
            }
            events.write_text(json.dumps(event) + "\n", encoding="utf-8")
            agent = root / "AGENT.md"
            agent.write_text("READY\n", encoding="utf-8")
            controller = campaign.Campaign(
                plan_path=plan_path,
                state_path=root / "fresh-state.json",
                events_path=events,
                agent_path=agent,
                ready_marker="READY",
                execute=False,
                poll_seconds=1,
                repo_root=root,
            )
            try:
                self.assertEqual(
                    controller.state["tuning"]["llama3.2-3b/3W16A"][
                        "attempt_count"
                    ],
                    1,
                )
            finally:
                controller.close()

    def test_repeated_launch_events_same_output_each_count_as_an_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = self._plan(root)
            plan_sha = campaign._sha256_file(plan_path)
            events = root / "events.jsonl"
            output_dir = root / "out" / "same-output"
            records = []
            for seq in (1, 2):
                records.append(
                    {
                        "schema_version": 1,
                        "seq": seq,
                        "timestamp_utc": "2026-01-01T00:00:00+00:00",
                        "campaign_id": f"campaign-{seq}",
                        "category": "task",
                        "code": "task_launching",
                        "details": {
                            "task_id": "realq:same",
                            "plan_sha256": plan_sha,
                            "kind": "realq",
                            "method": "realq",
                            "phase": "tune",
                            "model": "llama3.2-3b",
                            "setting": "3W16A",
                            "grad_lr": 1e-5,
                            "output_dir": str(output_dir),
                        },
                    }
                )
            events.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            agent = root / "AGENT.md"
            agent.write_text("READY\n", encoding="utf-8")
            controller = campaign.Campaign(
                plan_path=plan_path,
                state_path=root / "fresh-state.json",
                events_path=events,
                agent_path=agent,
                ready_marker="READY",
                execute=False,
                poll_seconds=1,
                repo_root=root,
            )
            try:
                self.assertEqual(
                    controller.state["tuning"]["llama3.2-3b/3W16A"][
                        "attempt_count"
                    ],
                    2,
                )
            finally:
                controller.close()

    def test_default_run_is_read_only_and_waits_for_agent_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = self._plan(root)
            state_path = root / "state.json"
            agent = root / "AGENT.md"
            agent.write_text("still installing\n", encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                returncode = campaign.main(
                    [
                        "--plan",
                        str(plan_path),
                        "--state",
                        str(state_path),
                        "--agent",
                        str(agent),
                        "--agent-ready-marker",
                        "READY",
                        "run",
                        "--once",
                    ]
                )
            self.assertEqual(returncode, 0)
            self.assertFalse(state_path.exists())
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["status"], "WAIT_ENV_READY")

    def test_selected_lr_patch_requires_all_fifteen(self):
        state = {
            "tuning": {
                f"{model}/{setting}": {
                    "status": "SELECTED",
                    "selected_lr": 1e-5,
                }
                for model in campaign.runner.MODEL_ORDER
                for setting in campaign.runner.SETTING_ORDER
            }
        }
        patch = campaign.selected_lr_patch(state)
        self.assertIsNotNone(patch)
        state["tuning"]["llama3.2-3b/3W16A"]["status"] = "COARSE"
        self.assertIsNone(campaign.selected_lr_patch(state))

    def test_commit_dry_run_never_writes_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = self._plan(root)
            before = plan_path.read_bytes()
            agent = root / "AGENT.md"
            agent.write_text("READY\n", encoding="utf-8")
            controller = campaign.Campaign(
                plan_path=plan_path,
                state_path=root / "state.json",
                events_path=root / "events.jsonl",
                agent_path=agent,
                ready_marker="READY",
                execute=False,
                poll_seconds=1,
                repo_root=root,
            )
            try:
                for group in controller.state["tuning"].values():
                    group["status"] = "SELECTED"
                    group["selected_lr"] = 1e-5
                result = controller.commit_selected_lrs()
            finally:
                controller.close()
            self.assertEqual(result["status"], "DRY_RUN")
            self.assertEqual(plan_path.read_bytes(), before)

    def test_live_plan_drift_is_detected_before_next_tick(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = self._plan(root)
            agent = root / "AGENT.md"
            agent.write_text("READY\n", encoding="utf-8")
            controller = campaign.Campaign(
                plan_path=plan_path,
                state_path=root / "state.json",
                events_path=root / "events.jsonl",
                agent_path=agent,
                ready_marker="READY",
                execute=False,
                poll_seconds=1,
                repo_root=root,
            )
            try:
                changed = json.loads(plan_path.read_text(encoding="utf-8"))
                changed["notes"] = "external mutation"
                plan_path.write_text(
                    json.dumps(changed, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                self.assertTrue(controller._refresh_plan_snapshot())
                self.assertFalse(controller._check_plan_transition())
                self.assertEqual(
                    controller.state["status"], "NEEDS_USER_ACTION"
                )
                self.assertIn(
                    "outside an approved LR commit",
                    controller.state["blockers"][0],
                )
            finally:
                controller.close()

    def test_success_manifest_from_other_plan_is_never_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = self._plan(root)
            agent = root / "AGENT.md"
            agent.write_text("READY\n", encoding="utf-8")
            controller = campaign.Campaign(
                plan_path=plan_path,
                state_path=root / "state.json",
                events_path=root / "events.jsonl",
                agent_path=agent,
                ready_marker="READY",
                execute=False,
                poll_seconds=1,
                repo_root=root,
            )
            output = root / "stale"
            output.mkdir()
            manifest_path = output / campaign.executor.MANIFEST_FILENAME
            old_plan = json.loads(plan_path.read_text(encoding="utf-8"))
            old_plan["notes"] = "different frozen protocol"
            old_sha = "a" * 64
            manifest_path.write_text(
                json.dumps(
                    {
                        "plan": {
                            "sha256": old_sha,
                            "sha256_at_end": old_sha,
                            "changed_during_execution": False,
                            "content": old_plan,
                        }
                    }
                ),
                encoding="utf-8",
            )
            task = {
                "task_id": "stale-formal",
                "spec": {
                    "kind": "baseline",
                    "method": "bf16",
                    "phase": "final",
                    "target_phase": None,
                    "model": "llama3.2-3b",
                    "setting": None,
                },
                "status": "SUCCEEDED",
            }
            try:
                with mock.patch.object(
                    campaign.results, "parse_result"
                ) as parse_result:
                    controller._validate_successful_task(
                        task, manifest_path
                    )
                self.assertEqual(task["status"], "INVALID_RESULT")
                self.assertEqual(
                    task["failure_class"], "plan_identity_mismatch"
                )
                parse_result.assert_not_called()
            finally:
                controller.close()

    def test_canary_is_not_launched_when_full_search_cannot_fit_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = self._plan(root)
            agent = root / "AGENT.md"
            agent.write_text("READY\n", encoding="utf-8")
            controller = campaign.Campaign(
                plan_path=plan_path,
                state_path=root / "state.json",
                events_path=root / "events.jsonl",
                agent_path=agent,
                ready_marker="READY",
                execute=False,
                poll_seconds=1,
                repo_root=root,
            )
            try:
                target_key = "llama3.2-3b/3W16A"
                target = controller.state["tuning"][target_key]
                target["attempt_count"] = 9
                for key, group in controller.state["tuning"].items():
                    if key != target_key:
                        group["status"] = "SELECTED"
                        group["selected_lr"] = 1e-5
                with (
                    mock.patch.object(
                        controller, "_static_ready", return_value=True
                    ),
                    mock.patch.object(
                        controller, "_add_tune_candidates"
                    ) as add_candidates,
                ):
                    controller._advance_tuning_groups()
                self.assertEqual(target["status"], "NEEDS_USER_ACTION")
                self.assertIn("memory canary", target["needs_user_action_reason"])
                add_candidates.assert_not_called()
            finally:
                controller.close()

    def test_foreign_gpu_process_blocks_all_new_launches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = self._plan(root)
            agent = root / "AGENT.md"
            agent.write_text("READY\n", encoding="utf-8")
            controller = campaign.Campaign(
                plan_path=plan_path,
                state_path=root / "state.json",
                events_path=root / "events.jsonl",
                agent_path=agent,
                ready_marker="READY",
                execute=False,
                poll_seconds=1,
                repo_root=root,
            )
            snapshot = campaign.GPUSnapshot(
                devices=tuple(
                    {
                        "index": index,
                        "uuid": f"GPU-{index}",
                        "memory_total_mib": 128 * 1024,
                    }
                    for index in range(8)
                ),
                compute_processes=(
                    {
                        "gpu_uuid": "GPU-3",
                        "pid": 12345,
                        "process_name": "python",
                        "used_memory_mib": 1000,
                    },
                ),
            )
            try:
                with (
                    mock.patch.object(
                        controller, "_ready_tasks", return_value=[{"fake": True}]
                    ),
                    mock.patch.object(controller, "_launch_task") as launch,
                ):
                    launched = controller._schedule(snapshot)
                self.assertEqual(launched, 0)
                self.assertEqual(controller.state["status"], "WAIT_GPUS")
                self.assertTrue(controller.state["blockers"])
                launch.assert_not_called()
            finally:
                controller.close()

    def test_tune_oom_stops_pending_siblings_before_profile_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = self._plan(root)
            agent = root / "AGENT.md"
            agent.write_text("READY\n", encoding="utf-8")
            controller = campaign.Campaign(
                plan_path=plan_path,
                state_path=root / "state.json",
                events_path=root / "events.jsonl",
                agent_path=agent,
                ready_marker="READY",
                execute=False,
                poll_seconds=1,
                repo_root=root,
            )
            target = controller.state["tuning"]["llama3.2-3b/3W16A"]
            common = {
                "kind": "realq",
                "phase": "tune",
                "model": "llama3.2-3b",
                "setting": "3W16A",
                "generation": 0,
                "profile_level": 0,
            }
            controller.state["tasks"] = {
                "oom": {
                    "task_id": "oom",
                    "spec": {**common, "grad_lr": 1e-5},
                    "status": "OOM",
                },
                "pending": {
                    "task_id": "pending",
                    "spec": {**common, "grad_lr": 5e-5},
                    "status": "PENDING",
                },
                "running": {
                    "task_id": "running",
                    "spec": {**common, "grad_lr": 1e-4},
                    "status": "RUNNING",
                },
            }
            try:
                controller._handle_group_oom(target)
                self.assertEqual(
                    controller.state["tasks"]["pending"]["status"],
                    "SUPERSEDED",
                )
                self.assertEqual(
                    controller.state["tasks"]["running"]["status"], "RUNNING"
                )
                self.assertEqual(target["status"], "OOM_DRAINING")
            finally:
                controller.close()

    def test_formal_oom_stops_pending_settings_before_profile_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = self._plan(root)
            agent = root / "AGENT.md"
            agent.write_text("READY\n", encoding="utf-8")
            controller = campaign.Campaign(
                plan_path=plan_path,
                state_path=root / "state.json",
                events_path=root / "events.jsonl",
                agent_path=agent,
                ready_marker="READY",
                execute=False,
                poll_seconds=1,
                repo_root=root,
            )
            model_state = {
                "realq_profile_level": 0,
                "realq_generation": 0,
                "status": "PENDING",
            }
            common = {
                "kind": "realq",
                "phase": "final",
                "model": "qwen3-32b",
                "generation": 0,
                "profile_level": 0,
            }
            controller.state["tasks"] = {
                "oom": {
                    "task_id": "oom",
                    "spec": {**common, "setting": "2W4A"},
                    "status": "OOM",
                },
                "pending": {
                    "task_id": "pending",
                    "spec": {**common, "setting": "3W16A"},
                    "status": "PENDING",
                },
                "running": {
                    "task_id": "running",
                    "spec": {**common, "setting": "3W4A"},
                    "status": "RUNNING",
                },
            }
            try:
                allowed = controller._advance_formal_realq_profile(
                    "qwen3-32b",
                    model_state,
                    controller._memory_profiles(
                        phase="final", static_only=False
                    ),
                )
                self.assertFalse(allowed)
                self.assertEqual(
                    controller.state["tasks"]["pending"]["status"],
                    "SUPERSEDED",
                )
                self.assertEqual(model_state["status"], "OOM_DRAINING")
            finally:
                controller.close()

    def test_bf16_result_accepts_intentional_none_kl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = self._plan(root)
            agent = root / "AGENT.md"
            agent.write_text("READY\n", encoding="utf-8")
            controller = campaign.Campaign(
                plan_path=plan_path,
                state_path=root / "state.json",
                events_path=root / "events.jsonl",
                agent_path=agent,
                ready_marker="READY",
                execute=False,
                poll_seconds=1,
                repo_root=root,
            )
            output = root / "bf16"
            output.mkdir()
            manifest = output / campaign.executor.MANIFEST_FILENAME
            plan_content = json.loads(plan_path.read_text(encoding="utf-8"))
            plan_sha = campaign._sha256_file(plan_path)
            manifest.write_text(
                json.dumps(
                    {
                        "plan": {
                            "sha256": plan_sha,
                            "sha256_at_end": plan_sha,
                            "changed_during_execution": False,
                            "content": plan_content,
                        }
                    }
                ),
                encoding="utf-8",
            )
            parsed = types.SimpleNamespace(
                phase="final",
                method="bf16",
                model="llama3.2-3b",
                setting=None,
                attempt_index=1,
                kl_wikitext2=None,
                ppl_wikitext2=8.0,
                acc_avg=0.5,
                tasks={"boolq": {"acc,none": 0.5}},
                grad_lr=None,
                final_layer_grad_lr=None,
                params={"lm_eval_batch_size": 16},
                identities={"plan_sha256": plan_sha},
                plan_content=plan_content,
            )
            task = {
                "task_id": "bf16",
                "spec": {
                    "kind": "baseline",
                    "method": "bf16",
                    "model": "llama3.2-3b",
                    "phase": "final",
                },
                "status": "SUCCEEDED",
            }
            try:
                with (
                    mock.patch.object(
                        campaign.results, "parse_result", return_value=parsed
                    ),
                    mock.patch.object(
                        controller,
                        "_manifest_task_command_identity",
                        return_value=(True, {"gpu_ids": [0]}),
                    ),
                ):
                    controller._validate_successful_task(task, manifest)
                self.assertEqual(task["status"], "SUCCEEDED")
                self.assertIsNone(task["metric"]["kl"])
                self.assertEqual(task["metric"]["ppl"], 8.0)
            finally:
                controller.close()

    def test_normal_tick_publishes_eta_in_state_status_and_heartbeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = self._plan(root)
            agent = root / "AGENT.md"
            agent.write_text("READY\n", encoding="utf-8")
            controller = campaign.Campaign(
                plan_path=plan_path,
                state_path=root / "state.json",
                events_path=root / "events.jsonl",
                agent_path=agent,
                ready_marker="READY",
                execute=False,
                poll_seconds=1,
                repo_root=root,
            )
            snapshot = campaign.GPUSnapshot((), ())
            try:
                with (
                    mock.patch.object(
                        controller, "_reconcile_tasks"
                    ),
                    mock.patch.object(
                        controller, "_update_static_entries_from_tasks"
                    ),
                    mock.patch.object(
                        controller, "_gates", return_value=(True, snapshot)
                    ),
                    mock.patch.object(
                        controller, "_campaign_status", return_value="TUNING"
                    ),
                    mock.patch.object(
                        controller, "_ensure_tune_static_tasks"
                    ),
                    mock.patch.object(
                        controller, "_advance_tuning_groups"
                    ),
                ):
                    payload = controller.tick()
                self.assertIs(payload["eta"], controller.state["eta"])
                heartbeat = next(
                    event
                    for event in reversed(controller.preview_events)
                    if event["code"] == "scheduler_tick"
                )
                self.assertEqual(
                    heartbeat["details"]["eta"]["schema_version"], 1
                )
                self.assertIn("optimistic", payload["eta"]["whole"]["eta"])
            finally:
                controller.close()

    def test_blocked_active_tick_keeps_eta_and_streams_heartbeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = self._plan(root)
            agent = root / "AGENT.md"
            agent.write_text("READY\n", encoding="utf-8")
            controller = campaign.Campaign(
                plan_path=plan_path,
                state_path=root / "state.json",
                events_path=root / "events.jsonl",
                agent_path=agent,
                ready_marker="READY",
                execute=False,
                poll_seconds=1,
                repo_root=root,
            )
            controller.state["tasks"]["active"] = {
                "task_id": "active",
                "spec": {
                    "kind": "realq",
                    "method": "realq",
                    "phase": "tune",
                    "target_phase": None,
                    "model": "qwen3-4b",
                    "setting": "3W16A",
                    "profile_level": 0,
                    "generation": 0,
                },
                "status": "RUNNING",
                "gpu_count": 1,
                "priority": 1,
                "gpus": [0],
                "executor_pid": 123,
                "manifest": None,
                "launched_utc": campaign._utc_now(),
            }

            def blocked_gates():
                controller.state["status"] = "WAIT_GPUS"
                controller.state["blockers"] = ["probe unavailable"]
                return False, None

            try:
                with (
                    mock.patch.object(controller, "_reconcile_tasks"),
                    mock.patch.object(
                        controller, "_update_static_entries_from_tasks"
                    ),
                    mock.patch.object(
                        controller, "_gates", side_effect=blocked_gates
                    ),
                ):
                    payload = controller.tick()
                self.assertEqual(payload["eta"]["counts"]["running"], 1)
                self.assertTrue(payload["eta"]["blocked"])
                heartbeat = next(
                    event
                    for event in reversed(controller.preview_events)
                    if event["code"] == "scheduler_blocked_tick"
                )
                self.assertEqual(
                    heartbeat["details"]["eta"]["counts"]["running"], 1
                )
                output = io.StringIO()
                with (
                    mock.patch.object(
                        controller, "tick", return_value=payload
                    ),
                    contextlib.redirect_stdout(output),
                ):
                    controller.run(once=True, stream_heartbeats=True)
                streamed = json.loads(output.getvalue())
                self.assertEqual(
                    streamed["record_type"], "campaign_heartbeat"
                )
                self.assertEqual(streamed["eta"]["counts"]["running"], 1)
            finally:
                controller.close()

    def test_master_ports_are_stable_and_unique_for_eight_concurrent_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = self._plan(root)
            agent = root / "AGENT.md"
            agent.write_text("READY\n", encoding="utf-8")
            controller = campaign.Campaign(
                plan_path=plan_path,
                state_path=root / "state.json",
                events_path=root / "events.jsonl",
                agent_path=agent,
                ready_marker="READY",
                execute=False,
                poll_seconds=1,
                repo_root=root,
            )
            try:
                ports = []
                with mock.patch.object(
                    controller, "_master_port_available", return_value=True
                ):
                    for index in range(8):
                        task = {
                            "task_id": f"task-{index}",
                            "status": "RUNNING",
                            "attempt_index": 1,
                        }
                        controller.state["tasks"][task["task_id"]] = task
                        ports.append(controller._assign_master_port(task))
                    self.assertEqual(len(set(ports)), 8)
                    self.assertEqual(
                        controller._assign_master_port(
                            controller.state["tasks"]["task-0"]
                        ),
                        ports[0],
                    )
            finally:
                controller.close()

    def test_static_port_collision_retries_at_distinct_attempt_two_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = self._plan(root)
            agent = root / "AGENT.md"
            agent.write_text("READY\n", encoding="utf-8")
            controller = campaign.Campaign(
                plan_path=plan_path,
                state_path=root / "state.json",
                events_path=root / "events.jsonl",
                agent_path=agent,
                ready_marker="READY",
                execute=False,
                poll_seconds=1,
                repo_root=root,
            )
            try:
                controller._ensure_tune_static_tasks()
                entry = controller.state["static_caches"][
                    "tune/llama3.2-3b"
                ]
                failed = controller.state["tasks"][entry["producer_task"]]
                failed["status"] = "FAILED"
                failed["failure_class"] = "master_port_collision"
                controller._update_static_entries_from_tasks()
                retry = controller.state["tasks"][entry["producer_task"]]
                self.assertIsNot(retry, failed)
                self.assertEqual(retry["attempt_index"], 2)
                self.assertEqual(retry["spec"]["attempt_index"], 2)
                rendered = controller._render_task(retry, [0])
                self.assertTrue(rendered.run_id.endswith("_attempt2"))
                self.assertTrue(
                    str(rendered.output_dir).endswith("_attempt2")
                )
            finally:
                controller.close()

    def test_dead_active_manifest_is_orphaned_after_bounded_grace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = self._plan(root)
            agent = root / "AGENT.md"
            agent.write_text("READY\n", encoding="utf-8")
            controller = campaign.Campaign(
                plan_path=plan_path,
                state_path=root / "state.json",
                events_path=root / "events.jsonl",
                agent_path=agent,
                ready_marker="READY",
                execute=False,
                poll_seconds=1,
                repo_root=root,
            )
            manifest = root / "running" / campaign.executor.MANIFEST_FILENAME
            manifest.parent.mkdir()
            manifest.write_text(
                json.dumps({"status": "running"}), encoding="utf-8"
            )
            task = {
                "task_id": "dead",
                "spec": {},
                "status": "RUNNING",
                "gpus": [0],
                "executor_pid": 999999,
                "manifest": str(manifest),
            }
            try:
                with mock.patch.object(
                    controller, "_task_process_alive", return_value=False
                ):
                    for _ in range(campaign.DEAD_PROCESS_GRACE_POLLS):
                        controller._reconcile_one_task(task)
                self.assertEqual(task["status"], "ORPHANED")
                self.assertEqual(task["failure_class"], "executor_disappeared")
                self.assertEqual(task["gpus"], [])
            finally:
                controller.close()

    def test_three_adopted_running_manifests_reserve_physical_gpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = self._plan(root)
            agent = root / "AGENT.md"
            agent.write_text("READY\n", encoding="utf-8")
            controller = campaign.Campaign(
                plan_path=plan_path,
                state_path=root / "state.json",
                events_path=root / "events.jsonl",
                agent_path=agent,
                ready_marker="READY",
                execute=False,
                poll_seconds=1,
                repo_root=root,
            )
            commands = {}
            adopted = []
            for gpu, model in enumerate(campaign.runner.MODEL_ORDER):
                spec = {
                    "kind": "realq_static",
                    "method": "realq_static",
                    "phase": "precompute",
                    "target_phase": "tune",
                    "model": model,
                    "setting": None,
                    "grad_lr": None,
                    "overrides": {},
                    "profile_level": 0,
                    "generation": 0,
                    "purpose": "tune_static_single_producer",
                }
                task = {
                    "task_id": f"adopt-{gpu}",
                    "spec": spec,
                    "status": "RUNNING",
                    "gpu_count": 1,
                    "priority": 100,
                    "gpus": [],
                    "attempt_index": 1,
                    "executor_pid": None,
                }
                rendered = controller._render_task(task, [gpu])
                pid = 5000 + gpu
                commands[pid] = rendered.argv
                manifest = {
                    "run_id": rendered.run_id,
                    "command": {
                        "argv": rendered.argv,
                        "env": rendered.env,
                    },
                    "process": {"pid": pid},
                }
                adopted.append((task, manifest, pid))
            identities = {
                pid: {
                    "pid": pid,
                    "ppid": 1,
                    "session_id": pid,
                    "start_ticks": 100 + pid,
                }
                for _, _, pid in adopted
            }
            try:
                with (
                    mock.patch.object(
                        campaign,
                        "_proc_cmdline",
                        side_effect=lambda pid: commands.get(pid),
                    ),
                    mock.patch.object(
                        campaign,
                        "_proc_identity",
                        side_effect=lambda pid: identities.get(pid),
                    ),
                    mock.patch.object(
                        campaign,
                        "_pid_is_descendant",
                        side_effect=lambda pid, root_pid: pid == root_pid,
                    ),
                ):
                    for task, manifest, _ in adopted:
                        self.assertTrue(
                            controller._adopt_manifest_process_identity(
                                task, manifest
                            )
                        )
                        controller.state["tasks"][task["task_id"]] = task
                    self.assertEqual(controller._reserved_gpus(), {0, 1, 2})
                    pending = {"gpu_count": 1}
                    devices = tuple(
                        {
                            "index": index,
                            "uuid": f"GPU-{index}",
                            "memory_total_mib": 128 * 1024,
                        }
                        for index in range(8)
                    )
                    processes = tuple(
                        {
                            "pid": pid,
                            "gpu_uuid": f"GPU-{gpu}",
                            "used_memory_mib": 1,
                        }
                        for gpu, (_, _, pid) in enumerate(adopted)
                    )
                    with (
                        mock.patch.object(
                            controller, "_ready_tasks", return_value=[pending]
                        ),
                        mock.patch.object(
                            controller, "_launch_task"
                        ) as launch,
                    ):
                        self.assertEqual(
                            controller._schedule(
                                campaign.GPUSnapshot(devices, processes)
                            ),
                            1,
                        )
                    launch.assert_called_once_with(pending, (3,))
            finally:
                controller.close()

    def test_gpu_owner_accepts_verified_descendant_with_new_session_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = self._plan(root)
            agent = root / "AGENT.md"
            agent.write_text("READY\n", encoding="utf-8")
            controller = campaign.Campaign(
                plan_path=plan_path,
                state_path=root / "state.json",
                events_path=root / "events.jsonl",
                agent_path=agent,
                ready_marker="READY",
                execute=False,
                poll_seconds=1,
                repo_root=root,
            )
            task = {
                "task_id": "owned",
                "status": "RUNNING",
                "executor_pid": 100,
                "executor_start_ticks": 10,
                "executor_session_id": 100,
                "executor_process_group_id": 100,
                "process_identity_origin": "controller_session",
                "gpus": [0],
            }
            controller.state["tasks"] = {"owned": task}
            identities = {
                100: {
                    "pid": 100,
                    "ppid": 1,
                    "process_group_id": 100,
                    "session_id": 100,
                    "start_ticks": 10,
                },
                200: {
                    "pid": 200,
                    "ppid": 100,
                    "process_group_id": 200,
                    "session_id": 200,
                    "start_ticks": 20,
                },
                300: {
                    "pid": 300,
                    "ppid": 1,
                    "process_group_id": 300,
                    "session_id": 300,
                    "start_ticks": 30,
                },
            }
            try:
                with (
                    mock.patch.object(
                        campaign,
                        "_proc_identity",
                        side_effect=lambda pid: identities.get(pid),
                    ),
                    mock.patch.object(
                        campaign,
                        "_pid_is_descendant",
                        side_effect=lambda pid, root_pid: (
                            pid == 200 and root_pid == 100
                        ),
                    ),
                ):
                    self.assertIs(
                        controller._gpu_process_owner(200), task
                    )
                    self.assertIsNone(controller._gpu_process_owner(300))
            finally:
                controller.close()

    def test_nested_descendant_on_wrong_gpu_is_an_ownership_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = self._plan(root)
            agent = root / "AGENT.md"
            agent.write_text("READY\n", encoding="utf-8")
            controller = campaign.Campaign(
                plan_path=plan_path,
                state_path=root / "state.json",
                events_path=root / "events.jsonl",
                agent_path=agent,
                ready_marker="READY",
                execute=False,
                poll_seconds=1,
                repo_root=root,
            )
            task = {
                "task_id": "owned",
                "status": "RUNNING",
                "executor_pid": 100,
                "executor_start_ticks": 10,
                "executor_session_id": 100,
                "executor_process_group_id": 100,
                "process_identity_origin": "controller_session",
                "gpus": [0],
            }
            controller.state["tasks"] = {"owned": task}
            identities = {
                100: {
                    "pid": 100,
                    "ppid": 1,
                    "process_group_id": 100,
                    "session_id": 100,
                    "start_ticks": 10,
                },
                200: {
                    "pid": 200,
                    "ppid": 100,
                    "process_group_id": 200,
                    "session_id": 200,
                    "start_ticks": 20,
                },
            }
            snapshot = campaign.GPUSnapshot(
                (
                    {
                        "index": 0,
                        "uuid": "GPU-0",
                        "memory_total_mib": 128 * 1024,
                    },
                    {
                        "index": 1,
                        "uuid": "GPU-1",
                        "memory_total_mib": 128 * 1024,
                    },
                ),
                (
                    {
                        "pid": 200,
                        "gpu_uuid": "GPU-1",
                        "used_memory_mib": 1,
                    },
                ),
            )
            try:
                with (
                    mock.patch.object(
                        campaign,
                        "_proc_identity",
                        side_effect=lambda pid: identities.get(pid),
                    ),
                    mock.patch.object(
                        campaign,
                        "_pid_is_descendant",
                        side_effect=lambda pid, root_pid: (
                            pid == 200 and root_pid == 100
                        ),
                    ),
                    mock.patch.object(
                        campaign,
                        "_resolve_gpu_process_pids",
                        return_value=({200: 200}, {}),
                    ),
                    mock.patch.object(
                        controller, "_launch_task"
                    ) as launch,
                ):
                    self.assertEqual(controller._schedule(snapshot), 0)
                launch.assert_not_called()
                self.assertEqual(controller.state["status"], "WAIT_GPUS")
                event = controller.preview_events[-1]
                self.assertEqual(event["code"], "gpu_foreign_process")
                conflicts = event["details"]["ownership_conflicts"]
                self.assertEqual(len(conflicts), 1)
                self.assertEqual(conflicts[0]["owner_task_id"], "owned")
                self.assertEqual(conflicts[0]["observed_gpu"], 1)
                self.assertEqual(conflicts[0]["reserved_gpus"], [0])
            finally:
                controller.close()

    def test_gpu_pid_namespace_mapping_is_unique_or_fails_closed(self):
        entries = [
            Path("/proc/12"),
            Path("/proc/13"),
            Path("/proc/not-a-pid"),
        ]
        namespace_ids = {
            12: (4988, 12),
            13: (4990, 13),
        }
        with (
            mock.patch.object(
                campaign.Path, "iterdir", return_value=entries
            ),
            mock.patch.object(
                campaign,
                "_proc_namespace_ids",
                side_effect=lambda pid: namespace_ids.get(pid),
            ),
        ):
            resolved, ambiguous = campaign._resolve_gpu_process_pids(
                [4988, 4990]
            )
        self.assertEqual(resolved, {4988: 12, 4990: 13})
        self.assertEqual(ambiguous, {})

        namespace_ids[13] = (4988, 13)
        with (
            mock.patch.object(
                campaign.Path, "iterdir", return_value=entries
            ),
            mock.patch.object(
                campaign,
                "_proc_namespace_ids",
                side_effect=lambda pid: namespace_ids.get(pid),
            ),
        ):
            resolved, ambiguous = campaign._resolve_gpu_process_pids([4988])
        self.assertNotIn(4988, resolved)
        self.assertEqual(ambiguous, {4988: [12, 13]})

    def test_controller_only_source_migration_is_audited_and_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = self._plan(root)
            agent = root / "AGENT.md"
            agent.write_text("READY\n", encoding="utf-8")
            controller = campaign.Campaign(
                plan_path=plan_path,
                state_path=root / "state.json",
                events_path=root / "events.jsonl",
                agent_path=agent,
                ready_marker="READY",
                execute=True,
                poll_seconds=1,
                repo_root=root,
            )
            previous = {
                "numerical_source_sha256": "1" * 64,
                "runner_sha256": "2" * 64,
                "executor_sha256": "3" * 64,
                "campaign_sha256": "4" * 64,
                "results_sha256": "5" * 64,
            }
            current = {**previous, "campaign_sha256": "6" * 64}
            controller.state["source_identity"] = previous
            try:
                with (
                    mock.patch.object(
                        controller,
                        "_current_source_identity",
                        return_value=current,
                    ),
                    mock.patch.object(
                        controller, "_refresh_plan_snapshot", return_value=True
                    ),
                    mock.patch.object(
                        controller, "_check_plan_transition", return_value=True
                    ),
                    mock.patch.object(
                        campaign, "agent_ready", return_value=(True, "ready")
                    ),
                    mock.patch.object(
                        campaign,
                        "venv_ready",
                        return_value=(True, {"valid": True}),
                    ),
                    mock.patch.object(controller, "_reconcile_tasks"),
                    mock.patch.object(
                        controller, "_update_static_entries_from_tasks"
                    ),
                    mock.patch.object(
                        campaign,
                        "query_gpus",
                        return_value=campaign.GPUSnapshot((), ()),
                    ),
                    mock.patch.object(
                        controller,
                        "_active_ownership_evidence",
                        return_value={
                            "active_tasks": [],
                            "gpu_processes": [],
                            "resolved_gpu_pids": {},
                            "ambiguous_gpu_pids": {},
                        },
                    ),
                ):
                    payload = controller.migrate_controller_source(
                        expected_old_campaign_sha256="4" * 64,
                        accept_campaign_sha256="6" * 64,
                    )
                self.assertEqual(payload["status"], "MIGRATED")
                self.assertEqual(controller.state["source_identity"], current)
                self.assertIsNone(
                    controller.state[
                        "pending_controller_source_transition"
                    ]
                )
                persisted = json.loads(
                    (root / "state.json").read_text(encoding="utf-8")
                )
                self.assertEqual(persisted["source_identity"], current)
                events = [
                    json.loads(line)
                    for line in (root / "events.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                self.assertEqual(
                    events[-1]["code"],
                    "controller_source_hotfix_accepted",
                )
            finally:
                controller.close()

    def test_controller_source_migration_requires_exact_hash_intent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = self._plan(root)
            agent = root / "AGENT.md"
            agent.write_text("READY\n", encoding="utf-8")
            controller = campaign.Campaign(
                plan_path=plan_path,
                state_path=root / "state.json",
                events_path=root / "events.jsonl",
                agent_path=agent,
                ready_marker="READY",
                execute=True,
                poll_seconds=1,
                repo_root=root,
            )
            previous = {
                "numerical_source_sha256": "1" * 64,
                "runner_sha256": "2" * 64,
                "executor_sha256": "3" * 64,
                "campaign_sha256": "4" * 64,
                "results_sha256": "5" * 64,
            }
            current = {**previous, "campaign_sha256": "6" * 64}
            controller.state["source_identity"] = previous
            try:
                with (
                    mock.patch.object(
                        controller,
                        "_current_source_identity",
                        return_value=current,
                    ),
                    mock.patch.object(controller, "persist") as persist,
                ):
                    with self.assertRaisesRegex(
                        campaign.CampaignBlocked,
                        "pinned campaign SHA256",
                    ):
                        controller.migrate_controller_source(
                            expected_old_campaign_sha256="7" * 64,
                            accept_campaign_sha256="6" * 64,
                        )
                    with self.assertRaisesRegex(
                        campaign.CampaignBlocked,
                        "current campaign SHA256",
                    ):
                        controller.migrate_controller_source(
                            expected_old_campaign_sha256="4" * 64,
                            accept_campaign_sha256="7" * 64,
                        )
                persist.assert_not_called()
            finally:
                controller.close()

    def test_controller_source_migration_fails_closed_on_pending_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = self._plan(root)
            agent = root / "AGENT.md"
            agent.write_text("READY\n", encoding="utf-8")
            controller = campaign.Campaign(
                plan_path=plan_path,
                state_path=root / "state.json",
                events_path=root / "events.jsonl",
                agent_path=agent,
                ready_marker="READY",
                execute=True,
                poll_seconds=1,
                repo_root=root,
            )
            controller.state["pending_controller_source_transition"] = {
                "transition_id": "a" * 64,
                "status": "PENDING_EVENT",
            }
            try:
                with (
                    mock.patch.object(
                        controller, "_current_source_identity"
                    ) as source_identity,
                    mock.patch.object(controller, "persist") as persist,
                ):
                    with self.assertRaisesRegex(
                        campaign.CampaignBlocked,
                        "incomplete controller-source migration",
                    ):
                        controller.migrate_controller_source(
                            expected_old_campaign_sha256="4" * 64,
                            accept_campaign_sha256="6" * 64,
                        )
                source_identity.assert_not_called()
                persist.assert_not_called()
            finally:
                controller.close()

    def test_controller_source_migration_rehashes_after_live_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = self._plan(root)
            agent = root / "AGENT.md"
            agent.write_text("READY\n", encoding="utf-8")
            controller = campaign.Campaign(
                plan_path=plan_path,
                state_path=root / "state.json",
                events_path=root / "events.jsonl",
                agent_path=agent,
                ready_marker="READY",
                execute=True,
                poll_seconds=1,
                repo_root=root,
            )
            previous = {
                "numerical_source_sha256": "1" * 64,
                "runner_sha256": "2" * 64,
                "executor_sha256": "3" * 64,
                "campaign_sha256": "4" * 64,
                "results_sha256": "5" * 64,
            }
            current = {**previous, "campaign_sha256": "6" * 64}
            changed = {**current, "campaign_sha256": "7" * 64}
            controller.state["source_identity"] = previous
            try:
                with (
                    mock.patch.object(
                        controller,
                        "_current_source_identity",
                        side_effect=[current, changed],
                    ),
                    mock.patch.object(
                        controller, "_refresh_plan_snapshot", return_value=True
                    ),
                    mock.patch.object(
                        controller, "_check_plan_transition", return_value=True
                    ),
                    mock.patch.object(
                        campaign, "agent_ready", return_value=(True, "ready")
                    ),
                    mock.patch.object(
                        campaign,
                        "venv_ready",
                        return_value=(True, {"valid": True}),
                    ),
                    mock.patch.object(
                        campaign,
                        "query_gpus",
                        return_value=campaign.GPUSnapshot((), ()),
                    ),
                    mock.patch.object(
                        controller,
                        "_active_ownership_evidence",
                        return_value={
                            "active_tasks": [],
                            "gpu_processes": [],
                            "resolved_gpu_pids": {},
                            "ambiguous_gpu_pids": {},
                        },
                    ),
                    mock.patch.object(controller, "persist") as persist,
                    mock.patch.object(controller, "emit") as emit,
                ):
                    with self.assertRaisesRegex(
                        campaign.CampaignBlocked,
                        "source identity changed",
                    ):
                        controller.migrate_controller_source(
                            expected_old_campaign_sha256="4" * 64,
                            accept_campaign_sha256="6" * 64,
                        )
                persist.assert_not_called()
                emit.assert_not_called()
            finally:
                controller.close()

    def test_migration_rejects_unrelated_manifest_target_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = self._plan(root)
            agent = root / "AGENT.md"
            agent.write_text("READY\n", encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "process": {"pid": 200},
                        "command": {"argv": ["python", "run.py"]},
                    }
                ),
                encoding="utf-8",
            )
            controller = campaign.Campaign(
                plan_path=plan_path,
                state_path=root / "state.json",
                events_path=root / "events.jsonl",
                agent_path=agent,
                ready_marker="READY",
                execute=False,
                poll_seconds=1,
                repo_root=root,
            )
            controller.state["tasks"] = {
                "active": {
                    "task_id": "active",
                    "status": "RUNNING",
                    "executor_pid": 100,
                    "manifest": str(manifest_path),
                }
            }
            try:
                with (
                    mock.patch.object(
                        controller, "_task_process_alive", return_value=True
                    ),
                    mock.patch.object(
                        campaign,
                        "_proc_identity",
                        return_value={
                            "pid": 200,
                            "ppid": 1,
                            "process_group_id": 200,
                            "session_id": 200,
                            "start_ticks": 20,
                        },
                    ),
                    mock.patch.object(
                        campaign,
                        "_proc_cmdline",
                        return_value=["python", "run.py"],
                    ),
                    mock.patch.object(
                        campaign, "_pid_is_descendant", return_value=False
                    ),
                ):
                    with self.assertRaisesRegex(
                        campaign.CampaignBlocked,
                        "not a live descendant",
                    ):
                        controller._active_ownership_evidence(
                            campaign.GPUSnapshot((), ())
                        )
            finally:
                controller.close()


if __name__ == "__main__":
    unittest.main()
