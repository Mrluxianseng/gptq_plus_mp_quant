import datetime as dt
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lowbit_activation_eta",
    ROOT / "tools" / "lowbit_activation_eta.py",
)
eta = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = eta
SPEC.loader.exec_module(eta)


NOW = dt.datetime(2026, 7, 25, 0, 0, tzinfo=dt.timezone.utc)


def iso(seconds):
    return (NOW + dt.timedelta(seconds=seconds)).isoformat()


def spec(
    *,
    kind="realq",
    phase="tune",
    model="qwen3-4b",
    setting="3W16A",
    profile=0,
    generation=0,
    method=None,
):
    return {
        "kind": kind,
        "method": method or kind,
        "phase": phase,
        "target_phase": None,
        "model": model,
        "setting": setting,
        "profile_level": profile,
        "generation": generation,
    }


def task(
    task_id,
    *,
    status,
    task_spec=None,
    gpu_count=1,
    launched=None,
    finished=None,
    gpus=None,
):
    return {
        "task_id": task_id,
        "status": status,
        "spec": task_spec or spec(),
        "gpu_count": gpu_count,
        "priority": 80,
        "gpus": list(gpus or []),
        "launched_utc": launched,
        "finished_utc": finished,
    }


def minimal_plan(*, models=None, settings=None, cap=20):
    return {
        "models": models or {},
        "settings": settings or {},
        "tuning": {
            "max_attempts_per_model_setting": cap,
            "lr_candidates": [0, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
            "search_policy": {"reserved_refinement_attempts": 3},
        },
    }


def minimal_state(tasks=None, *, tuning=None, status="TUNING"):
    return {
        "created_utc": iso(-1000),
        "status": status,
        "blockers": [],
        "tasks": {item["task_id"]: item for item in (tasks or [])},
        "tuning": tuning or {},
        "formal": {"initialized": False, "models": {}},
        "static_caches": {},
    }


class LowbitActivationEtaTest(unittest.TestCase):
    def test_json_serializable_cold_start_has_explicit_low_confidence(self):
        plan = minimal_plan(
            models={"qwen3-32b": "unused"},
            settings={"3W16A": {}},
        )
        payload = eta.estimate_campaign_eta(
            minimal_state(),
            plan,
            now_utc=NOW,
        )
        json.dumps(payload)
        self.assertEqual(payload["elapsed_seconds"], 1000)
        self.assertEqual(payload["whole"]["confidence"], "low")
        self.assertTrue(
            any(
                source.startswith("fallback:")
                for source in payload["whole"]["duration_sources"]
            )
        )
        self.assertGreater(
            payload["whole"]["eta"]["median"]["remaining_seconds"],
            0,
        )
        self.assertEqual(payload["capacity"]["one_gpu_parallelism"], 8)
        self.assertEqual(payload["capacity"]["four_gpu_parallelism"], 2)
        group = payload["tuning_candidate_forecast"]["groups"][
            "qwen3-32b/3W16A"
        ]
        self.assertEqual(group["future_candidates"]["optimistic"], 12)
        self.assertEqual(group["future_candidates"]["median"], 12)

    def test_eight_one_gpu_jobs_run_in_one_wave(self):
        history = task(
            "history",
            status="SUCCEEDED",
            launched=iso(-200),
            finished=iso(-100),
        )
        pending = [
            task(f"p{index}", status="PENDING") for index in range(8)
        ]
        payload = eta.estimate_campaign_eta(
            minimal_state([history, *pending]),
            minimal_plan(),
            now_utc=NOW,
        )
        self.assertEqual(
            payload["stages"]["tuning"]["eta"]["median"][
                "remaining_seconds"
            ],
            100,
        )

    def test_two_four_gpu_jobs_run_in_one_wave_and_three_need_two(self):
        four_spec = spec(kind="realq", phase="final")
        history = task(
            "history",
            status="SUCCEEDED",
            task_spec=four_spec,
            gpu_count=4,
            launched=iso(-200),
            finished=iso(-100),
        )
        two = [
            task(
                f"p{index}",
                status="PENDING",
                task_spec=four_spec,
                gpu_count=4,
            )
            for index in range(2)
        ]
        first = eta.estimate_campaign_eta(
            minimal_state([history, *two], status="FORMAL"),
            minimal_plan(),
            now_utc=NOW,
        )
        self.assertEqual(
            first["stages"]["formal"]["eta"]["median"]["remaining_seconds"],
            100,
        )
        third = task(
            "p2",
            status="PENDING",
            task_spec=four_spec,
            gpu_count=4,
        )
        second = eta.estimate_campaign_eta(
            minimal_state([history, *two, third], status="FORMAL"),
            minimal_plan(),
            now_utc=NOW,
        )
        self.assertEqual(
            second["stages"]["formal"]["eta"]["median"]["remaining_seconds"],
            200,
        )

    def test_running_residual_subtracts_elapsed_from_observed_total(self):
        history = task(
            "history",
            status="SUCCEEDED",
            launched=iso(-200),
            finished=iso(-100),
        )
        running = task(
            "running",
            status="RUNNING",
            launched=iso(-30),
            gpus=[0],
        )
        payload = eta.estimate_campaign_eta(
            minimal_state([history, running]),
            minimal_plan(),
            now_utc=NOW,
        )
        self.assertEqual(payload["counts"]["running"], 1)
        self.assertEqual(
            payload["stages"]["tuning"]["eta"]["median"][
                "remaining_seconds"
            ],
            70,
        )

    def test_exact_profile_generation_history_precedes_broad_history(self):
        exact = task(
            "exact",
            status="SUCCEEDED",
            task_spec=spec(profile=2, generation=3),
            launched=iso(-200),
            finished=iso(-100),
        )
        broad = task(
            "broad",
            status="SUCCEEDED",
            task_spec=spec(profile=1, generation=0),
            launched=iso(-1200),
            finished=iso(-200),
        )
        pending = task(
            "pending",
            status="PENDING",
            task_spec=spec(profile=2, generation=3),
        )
        payload = eta.estimate_campaign_eta(
            minimal_state([exact, broad, pending]),
            minimal_plan(),
            now_utc=NOW,
        )
        self.assertEqual(
            payload["stages"]["tuning"]["eta"]["median"][
                "remaining_seconds"
            ],
            100,
        )
        sources = payload["stages"]["tuning"]["duration_sources"]
        self.assertIn(
            "observed:model_phase_kind_profile_generation:successful",
            sources,
        )

    def test_failed_and_oom_durations_never_train_success_eta(self):
        for terminal_status in ("FAILED", "OOM"):
            with self.subTest(status=terminal_status):
                short_failure = task(
                    f"short-{terminal_status.lower()}",
                    status=terminal_status,
                    launched=iso(-2),
                    finished=iso(-1),
                )
                pending = task("pending", status="PENDING")
                payload = eta.estimate_campaign_eta(
                    minimal_state([short_failure, pending]),
                    minimal_plan(),
                    now_utc=NOW,
                )
                self.assertEqual(
                    payload["stages"]["tuning"]["eta"]["median"][
                        "remaining_seconds"
                    ],
                    50 * 60,
                )
                sources = payload["stages"]["tuning"][
                    "duration_sources"
                ]
                self.assertTrue(
                    all(":terminal" not in source for source in sources)
                )
                self.assertIn(
                    "fallback:tuning:realq:model_size_scaled",
                    sources,
                )

    def test_static_success_cannot_shorten_candidate_fallback(self):
        static_spec = spec(
            kind="realq_static",
            phase="precompute",
            setting=None,
        )
        static_spec["target_phase"] = "tune"
        short_static = task(
            "static",
            status="SUCCEEDED",
            task_spec=static_spec,
            launched=iso(-2),
            finished=iso(-1),
        )
        pending = task("candidate", status="PENDING")
        payload = eta.estimate_campaign_eta(
            minimal_state([short_static, pending]),
            minimal_plan(),
            now_utc=NOW,
        )
        self.assertEqual(
            payload["stages"]["tuning"]["eta"]["median"][
                "remaining_seconds"
            ],
            50 * 60,
        )
        self.assertIn(
            "fallback:tuning:realq:model_size_scaled",
            payload["stages"]["tuning"]["duration_sources"],
        )

    def test_tuning_success_cannot_shorten_formal_fallback(self):
        short_tune = task(
            "tune-success",
            status="SUCCEEDED",
            launched=iso(-2),
            finished=iso(-1),
        )
        formal_spec = spec(kind="realq", phase="final", setting="3W16A")
        pending_formal = task(
            "formal-pending",
            status="PENDING",
            task_spec=formal_spec,
            gpu_count=4,
        )
        payload = eta.estimate_campaign_eta(
            minimal_state([short_tune, pending_formal], status="FORMAL"),
            minimal_plan(),
            now_utc=NOW,
        )
        self.assertEqual(
            payload["stages"]["formal"]["eta"]["median"][
                "remaining_seconds"
            ],
            3 * 60 * 60,
        )
        self.assertIn(
            "fallback:formal:realq:model_size_scaled",
            payload["stages"]["formal"]["duration_sources"],
        )

    def test_small_model_success_cannot_shorten_32b_fallback(self):
        small_success = task(
            "small-success",
            status="SUCCEEDED",
            task_spec=spec(model="llama3.2-3b"),
            launched=iso(-2),
            finished=iso(-1),
        )
        large_pending = task(
            "large-pending",
            status="PENDING",
            task_spec=spec(model="qwen3-32b"),
        )
        payload = eta.estimate_campaign_eta(
            minimal_state([small_success, large_pending]),
            minimal_plan(),
            now_utc=NOW,
        )
        self.assertGreater(
            payload["stages"]["tuning"]["eta"]["median"][
                "remaining_seconds"
            ],
            8000,
        )
        self.assertIn(
            "fallback:tuning:realq:model_size_scaled",
            payload["stages"]["tuning"]["duration_sources"],
        )

    def test_baseline_methods_do_not_share_duration_history(self):
        bf16_spec = spec(
            kind="baseline",
            phase="final",
            setting=None,
            method="bf16",
        )
        gptaq_spec = spec(
            kind="baseline",
            phase="final",
            setting="3W16A",
            method="gptaq",
        )
        short_bf16 = task(
            "bf16",
            status="SUCCEEDED",
            task_spec=bf16_spec,
            launched=iso(-2),
            finished=iso(-1),
        )
        pending_gptaq = task(
            "gptaq",
            status="PENDING",
            task_spec=gptaq_spec,
        )
        payload = eta.estimate_campaign_eta(
            minimal_state([short_bf16, pending_gptaq], status="FORMAL"),
            minimal_plan(),
            now_utc=NOW,
        )
        self.assertEqual(
            payload["stages"]["formal"]["eta"]["median"][
                "remaining_seconds"
            ],
            75 * 60,
        )
        self.assertIn(
            "fallback:formal:baseline:model_size_scaled",
            payload["stages"]["formal"]["duration_sources"],
        )

    def test_candidate_forecast_never_exceeds_twenty_attempt_cap(self):
        plan = minimal_plan(
            models={"qwen3-4b": "unused"},
            settings={"3W16A": {}},
            cap=20,
        )
        group = {
            "qwen3-4b/3W16A": {
                "status": "COARSE",
                "profile_level": 0,
                "generation": 0,
                "attempt_count": 19,
            }
        }
        pending = task("pending", status="PENDING")
        payload = eta.estimate_campaign_eta(
            minimal_state([pending], tuning=group),
            plan,
            now_utc=NOW,
        )
        forecast = payload["tuning_candidate_forecast"]["groups"][
            "qwen3-4b/3W16A"
        ]
        self.assertEqual(forecast["remaining_cap_slots"], 0)
        self.assertEqual(
            forecast["future_candidates"],
            {"optimistic": 0, "median": 0, "conservative": 0},
        )
        self.assertLessEqual(
            forecast["attempted"]
            + forecast["known_pending"]
            + forecast["future_candidates"]["conservative"],
            20,
        )

    def test_oom_restart_likely_and_conservative_stay_under_cap(self):
        plan = minimal_plan(
            models={"qwen3-4b": "unused"},
            settings={"3W16A": {}},
            cap=20,
        )
        group = {
            "qwen3-4b/3W16A": {
                "status": "OOM_DRAINING",
                "profile_level": 1,
                "generation": 1,
                "attempt_count": 8,
            }
        }
        payload = eta.estimate_campaign_eta(
            minimal_state(tuning=group),
            plan,
            now_utc=NOW,
        )
        forecast = payload["tuning_candidate_forecast"]["groups"][
            "qwen3-4b/3W16A"
        ]
        self.assertEqual(
            forecast["future_candidates"]["conservative"],
            12,
        )
        self.assertEqual(forecast["future_candidates"]["median"], 12)
        self.assertEqual(
            forecast["attempted"]
            + forecast["future_candidates"]["conservative"],
            20,
        )

    def test_wait_state_with_active_child_still_has_live_eta(self):
        history = task(
            "history",
            status="SUCCEEDED",
            launched=iso(-200),
            finished=iso(-100),
        )
        running = task(
            "running",
            status="RUNNING",
            launched=iso(-25),
            gpus=[3],
        )
        state = minimal_state(
            [history, running],
            status="WAIT_GPUS",
        )
        state["blockers"] = ["foreign process"]
        payload = eta.estimate_campaign_eta(
            state,
            minimal_plan(),
            now_utc=NOW,
        )
        self.assertTrue(payload["blocked"])
        self.assertEqual(payload["counts"]["running"], 1)
        self.assertEqual(
            payload["stages"]["tuning"]["eta"]["median"][
                "remaining_seconds"
            ],
            75,
        )

    def test_stalled_and_validating_are_active_not_pending(self):
        stalled = task(
            "stalled",
            status="STALLED",
            launched=iso(-10),
            gpus=[0],
        )
        validating = task(
            "validating",
            status="VALIDATING",
            launched=iso(-20),
            gpus=[1],
        )
        payload = eta.estimate_campaign_eta(
            minimal_state([stalled, validating]),
            minimal_plan(),
            now_utc=NOW,
        )
        self.assertEqual(payload["counts"]["running"], 2)
        self.assertEqual(payload["counts"]["pending"], 0)

    def test_static_precompute_and_tuning_arm_are_dependency_waves(self):
        static_spec = spec(
            kind="realq_static",
            phase="precompute",
            setting=None,
        )
        static_spec["target_phase"] = "tune"
        static_history = task(
            "static-history",
            status="SUCCEEDED",
            task_spec=static_spec,
            launched=iso(-300),
            finished=iso(-200),
        )
        arm_history = task(
            "arm-history",
            status="SUCCEEDED",
            launched=iso(-200),
            finished=iso(-100),
        )
        static_pending = task(
            "realq_static:pending",
            status="PENDING",
            task_spec=static_spec,
        )
        arm_pending = task("realq:pending", status="PENDING")
        payload = eta.estimate_campaign_eta(
            minimal_state(
                [static_history, arm_history, static_pending, arm_pending]
            ),
            minimal_plan(),
            now_utc=NOW,
        )
        self.assertEqual(
            payload["stages"]["tuning"]["eta"]["median"][
                "remaining_seconds"
            ],
            200,
        )

    def test_known_four_gpu_work_gets_conservative_oom_replacement(self):
        formal_spec = spec(kind="realq", phase="final", setting="2W4A")
        pending = task(
            "realq:known",
            status="PENDING",
            task_spec=formal_spec,
            gpu_count=4,
        )
        payload = eta.estimate_campaign_eta(
            minimal_state([pending], status="FORMAL"),
            minimal_plan(),
            now_utc=NOW,
        )
        self.assertEqual(
            payload["formal_forecast"][
                "conservative_one_wave_oom_replacements"
            ],
            1,
        )
        self.assertGreater(
            payload["stages"]["formal"]["eta"]["conservative"][
                "work_items"
            ],
            payload["stages"]["formal"]["eta"]["median"]["work_items"],
        )

    def test_elapsed_uses_earliest_task_timestamp_after_state_rebuild(self):
        old = task(
            "old",
            status="RUNNING",
            launched=iso(-1000),
            gpus=[0],
        )
        state = minimal_state([old])
        state["created_utc"] = iso(-100)
        payload = eta.estimate_campaign_eta(
            state,
            minimal_plan(),
            now_utc=NOW,
        )
        self.assertEqual(payload["elapsed_seconds"], 1000)
        self.assertIn("earliest_task", payload["elapsed_source"])

    def test_overdue_running_task_keeps_nonzero_uncertainty(self):
        history = task(
            "history",
            status="SUCCEEDED",
            launched=iso(-300),
            finished=iso(-200),
        )
        running = task(
            "running",
            status="RUNNING",
            launched=iso(-200),
            gpus=[0],
        )
        payload = eta.estimate_campaign_eta(
            minimal_state([history, running]),
            minimal_plan(),
            now_utc=NOW,
        )
        self.assertEqual(payload["overdue_running_tasks"], ["running"])
        for bound in eta.BOUNDS:
            self.assertGreater(
                payload["stages"]["tuning"]["eta"][bound][
                    "remaining_seconds"
                ],
                0,
            )

    def test_same_state_and_timestamp_is_restart_stable(self):
        running = task(
            "running",
            status="RUNNING",
            launched=iso(-25),
            gpus=[0],
        )
        state = minimal_state([running])
        plan = minimal_plan()
        first = eta.estimate_campaign_eta(state, plan, now_utc=NOW)
        second = eta.estimate_campaign_eta(state, plan, now_utc=NOW)
        self.assertEqual(first, second)

    def test_future_or_invalid_timestamps_never_create_negative_time(self):
        running = task(
            "running",
            status="RUNNING",
            launched=iso(100),
            gpus=[0],
        )
        state = minimal_state([running])
        state["created_utc"] = "not-a-time"
        payload = eta.estimate_campaign_eta(
            state,
            minimal_plan(),
            now_utc=NOW,
        )
        self.assertGreaterEqual(payload["elapsed_seconds"], 0)
        for bound in eta.BOUNDS:
            self.assertGreaterEqual(
                payload["whole"]["eta"][bound]["remaining_seconds"],
                0,
            )

    def test_formal_forecast_includes_expected_tasks_and_one_oom_wave(self):
        plan = minimal_plan(
            models={"qwen3-4b": "unused"},
            settings={"3W16A": {}, "2W4A": {}},
        )
        payload = eta.estimate_campaign_eta(
            minimal_state(),
            plan,
            now_utc=NOW,
        )
        # static + saliency + bf16 + 2 * (GPTAQ + GuidedGPTQ + REAL-Q)
        self.assertEqual(payload["formal_forecast"]["base_missing_tasks"], 9)
        # static plus two REAL-Q settings are four-GPU replacement candidates.
        self.assertEqual(
            payload["formal_forecast"][
                "conservative_one_wave_oom_replacements"
            ],
            3,
        )
        self.assertEqual(
            payload["stages"]["formal"]["forecast_future_tasks"],
            {"optimistic": 9, "median": 9, "conservative": 12},
        )
        self.assertEqual(
            payload["tuning_static_forecast"]["base_missing_tasks"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
