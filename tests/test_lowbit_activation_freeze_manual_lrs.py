from __future__ import annotations

import copy
import contextlib
import decimal
import fcntl
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
EXPERIMENT = ROOT / "experiments" / "lowbit_activation"
for directory in (str(TOOLS), str(EXPERIMENT)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

SPEC = importlib.util.spec_from_file_location(
    "lowbit_activation_freeze_manual_lrs",
    TOOLS / "lowbit_activation_freeze_manual_lrs.py",
)
assert SPEC is not None and SPEC.loader is not None
freeze = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = freeze
SPEC.loader.exec_module(freeze)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_pretty(path: Path, value: object, *, sort_keys: bool = True) -> bytes:
    raw = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=sort_keys,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _write_canonical(path: Path, value: object) -> bytes:
    raw = freeze._canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


class Fixture:
    def __init__(self, root: Path) -> None:
        self.patch_stack = contextlib.ExitStack()
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.output_root = root / "output"
        self.plan_path = root / "plan.json"
        self.state_path = self.output_root / "_campaign" / "state.json"
        self.selection_path = root / "selection.json"
        self.agent_path = root / "AGENT.md"
        self.agent_path.write_text(
            f"{freeze.DEFAULT_READY_MARKER}\n", encoding="utf-8"
        )
        self.source = {
            "numerical_source_sha256": (
                freeze.QWEN32_MANUAL_INVALID_SOURCE_START_SHA256
            ),
            "runner_sha256": _digest("runner"),
            "executor_sha256": _digest("executor"),
            "campaign_sha256": _digest("campaign"),
            "results_sha256": _digest("results"),
        }
        self.gate = self._gate()
        self.report_attempt_overrides: dict[tuple[str, str], int] = {}
        self.report_rejected: list[dict] = []
        self.report_informational: list[dict] = []
        self.report_retired: list[dict] = []
        self.report_excluded_records: set[Path] = set()

        self.plan = json.loads(freeze.DEFAULT_PLAN.read_text(encoding="utf-8"))
        self.plan["output_root"] = str(self.output_root)
        self.plan_raw = _write_pretty(
            self.plan_path, self.plan, sort_keys=False
        )
        self.plan_sha = freeze._sha256_bytes(self.plan_raw)
        self.parsed: dict[Path, SimpleNamespace] = {}
        self.selections, tasks = self._evidence_and_selections()
        self.selection_raw = _write_canonical(
            self.selection_path,
            {
                "schema_version": 1,
                "selections": self.selections,
            },
        )
        self.state = self._state(tasks)
        self.state_raw = _write_pretty(self.state_path, self.state)
        self.state_sha = freeze._sha256_bytes(self.state_raw)
        self.campaign_id = self.state["campaign_id"]
        self.lock_path = self.state_path.with_suffix(
            self.state_path.suffix + ".lock"
        )
        self.lock_path.touch()

        self.patch(
            freeze.base_campaign.Campaign,
            "_current_source_identity",
            lambda _controller: dict(self.source),
        )
        self.patch(
            freeze.timed_campaign,
            "_timing_gate_now",
            lambda: copy.deepcopy(self.gate),
        )
        self.patch(
            freeze,
            "agent_ready",
            lambda path, marker: (
                path == self.agent_path
                and marker == freeze.DEFAULT_READY_MARKER,
                "ready",
            ),
        )
        self.patch(
            freeze, "venv_ready", lambda _plan: (True, {"valid": True})
        )
        self.patch(
            freeze,
            "query_gpus",
            lambda: freeze.base_campaign.GPUSnapshot((), ()),
        )
        self.patch(freeze, "proc_identity", lambda _pid: None)
        self.patch(freeze, "proc_cmdline", lambda _pid: None)
        self.patch(
            freeze.results,
            "parse_result",
            lambda path: copy.deepcopy(self.parsed[path.resolve()]),
        )
        self.patch(
            freeze.results,
            "build_report",
            lambda _root: self._report(),
        )

    def patch(self, target, name: str, value) -> None:
        self.patch_stack.enter_context(mock.patch.object(target, name, value))

    def close(self) -> None:
        self.patch_stack.close()

    def _report(self) -> dict:
        record_values = [
            parsed
            for path, parsed in self.parsed.items()
            if path not in self.report_excluded_records
        ]
        all_values = list(self.parsed.values())
        return {
            "retirement_ledger": {"valid": True, "errors": []},
            "rejected": copy.deepcopy(self.report_rejected),
            "informational": copy.deepcopy(self.report_informational),
            "retired_attempts": copy.deepcopy(self.report_retired),
            "records": [
                parsed.public_dict()
                if hasattr(parsed, "public_dict")
                else {
                    "manifest": str(parsed.manifest_path),
                    "model": parsed.model,
                    "setting": parsed.setting,
                    "method": parsed.method,
                    "phase": parsed.phase,
                    "grad_lr": parsed.grad_lr,
                    "kl_wikitext2": parsed.kl_wikitext2,
                }
                for parsed in record_values
            ],
            "realq_tune_selection": [
                {
                    "model": model,
                    "setting": setting,
                    "attempt_count": self.report_attempt_overrides.get(
                        (model, setting),
                        sum(
                            1
                            for parsed in all_values
                            if parsed.model == model
                            and parsed.setting == setting
                            and parsed.method == "realq"
                            and parsed.phase == "tune"
                        ),
                    ),
                }
                for model, setting in freeze._matrix_keys()
            ],
        }

    def _gate(self) -> dict:
        source_names = (
            "formal_timed_campaign",
            "formal_timed_execute",
            "formal_timing_adapter",
            "formal_timing_sitecustomize",
            "lowbit_activation_campaign",
            "lowbit_activation_gpu_hours",
            "validate_guided_saliency",
        )
        sources = [
            {
                "name": name,
                "original_path": str(self.root / f"{name}.py"),
                "sha256": (
                    self.source["campaign_sha256"]
                    if name == "lowbit_activation_campaign"
                    else _digest(name)
                ),
            }
            for name in source_names
        ]
        gate = {
            "schema_version": 1,
            "controller_path": sources[0]["original_path"],
            "controller_sha256": sources[0]["sha256"],
            "base_campaign_path": next(
                item["original_path"]
                for item in sources
                if item["name"] == "lowbit_activation_campaign"
            ),
            "base_campaign_sha256": self.source["campaign_sha256"],
            "timed_executor_path": next(
                item["original_path"]
                for item in sources
                if item["name"] == "formal_timed_execute"
            ),
            "timed_executor_sha256": next(
                item["sha256"]
                for item in sources
                if item["name"] == "formal_timed_execute"
            ),
            "gpu_hour_aggregator_sha256": next(
                item["sha256"]
                for item in sources
                if item["name"] == "lowbit_activation_gpu_hours"
            ),
            "guided_validator_sha256": next(
                item["sha256"]
                for item in sources
                if item["name"] == "validate_guided_saliency"
            ),
            "source_set_sha256": _digest("timing-source-set"),
            "sources": sources,
        }
        gate["gate_sha256"] = freeze._sha256_bytes(
            freeze._canonical_json_bytes(gate)
        )
        return gate

    def _add_evidence(
        self,
        *,
        model: str,
        setting: str,
        lr: float,
        kl: float,
        ordinal: int,
    ) -> tuple[dict, dict]:
        run_dir = (
            self.output_root
            / model
            / setting
            / "realq"
            / "tune"
            / f"run-{ordinal}"
        )
        manifest_path = run_dir / freeze.results.MANIFEST_FILENAME
        log_path = run_dir / freeze.results.LOG_FILENAME
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        execution_id = f"execution-{ordinal}"
        run_id = f"tune_realq_{model}_{setting.lower()}_{ordinal}"
        _write_pretty(
            manifest_path,
            {
                "schema_version": 1,
                "execution_id": execution_id,
                "run_id": run_id,
                "status": "succeeded",
                "exit_code": 0,
                "plan": {
                    "sha256": self.plan_sha,
                    "content": copy.deepcopy(self.plan),
                },
                "model": {
                    "combined_identity_sha256": _digest(model),
                },
                "source_files": {
                    "runner": {"sha256": self.source["runner_sha256"]},
                    "executor": {
                        "sha256": self.source["executor_sha256"]
                    },
                },
                "numerical_source_tree": {
                    "combined_sha256": self.source[
                        "numerical_source_sha256"
                    ],
                    "combined_sha256_at_end": self.source[
                        "numerical_source_sha256"
                    ],
                    "changed_during_execution": False,
                },
                "command": {
                    "argv": [
                        "python",
                        "realq.ptq",
                        "--grad_lr",
                        repr(lr),
                    ],
                    "env": {
                        "LOWBIT_ACTIVATION_ATTEMPT_INDEX": "1",
                    },
                },
                "process": {"pid": None},
            },
        )
        log_path.write_text(
            f"Exact KL&PPL on wikitext2: {kl}, 10.0\n",
            encoding="utf-8",
        )
        parsed = SimpleNamespace(
            manifest_path=manifest_path.resolve(),
            log_path=log_path.resolve(),
            execution_id=execution_id,
            run_id=run_id,
            model=model,
            setting=setting,
            method="realq",
            phase="tune",
            world_size=1,
            attempt_index=1,
            grad_lr=lr,
            final_layer_grad_lr=1e-5,
            kl_wikitext2=kl,
            ppl_wikitext2=10.0,
            tasks={},
            acc_avg=None,
            lm_eval_version="0.4.4",
            params={},
            identities={
                "plan_sha256": self.plan_sha,
                "model_sha256": _digest(model),
                "runner_sha256": self.source["runner_sha256"],
                "executor_sha256": self.source["executor_sha256"],
                "numerical_source_sha256": self.source[
                    "numerical_source_sha256"
                ],
            },
            expected_lr_candidates=tuple(
                self.plan["tuning"]["lr_candidates"]
            ),
            max_attempts=20,
            search_policy=copy.deepcopy(
                self.plan["tuning"]["search_policy"]
            ),
            plan_content=copy.deepcopy(self.plan),
        )
        self.parsed[manifest_path.resolve()] = parsed
        task = {
            "task_id": f"task-{ordinal}",
            "spec": {
                "kind": "realq",
                "method": "realq",
                "phase": "tune",
                "target_phase": None,
                "model": model,
                "setting": setting,
                "grad_lr": lr,
                "overrides": {},
                "profile_level": 0,
                "generation": 0,
                "purpose": "fixture",
            },
            "status": "SUCCEEDED",
            "gpu_count": 1,
            "priority": 1,
            "gpus": [],
            "attempt_index": 1,
            "executor_pid": None,
            "manifest": str(manifest_path.resolve()),
            "log": str(log_path.resolve()),
            "exit_code": 0,
            "plan_compatible": True,
            "metric": {
                "kl": kl,
                "ppl": 10.0,
                "grad_lr": lr,
                "identities": copy.deepcopy(parsed.identities),
            },
        }
        evidence = {
            "manifest": str(manifest_path.resolve()),
            "grad_lr": lr,
            "kl_wikitext2": kl,
        }
        return evidence, task

    def add_manual_invalid_result(
        self,
        *,
        model: str | None = None,
        setting: str | None = None,
        lr: float | None = None,
        ordinal: int = 9001,
        source_end: str | None = None,
    ) -> Path:
        model = model or freeze.MANUAL_INVALID_KEY[0]
        setting = setting or freeze.MANUAL_INVALID_KEY[1]
        lr = freeze.MANUAL_INVALID_LR if lr is None else lr
        _, task = self._add_evidence(
            model=model,
            setting=setting,
            lr=lr,
            kl=99.0,
            ordinal=ordinal,
        )
        manifest_path = Path(task["manifest"]).resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["numerical_source_tree"][
            "combined_sha256_at_end"
        ] = source_end or freeze.QWEN4_MANUAL_INVALID_SOURCE_END_SHA256
        payload["numerical_source_tree"][
            "changed_during_execution"
        ] = True
        _write_pretty(manifest_path, payload)
        task["status"] = "INVALID_RESULT"
        self.state["tasks"][task["task_id"]] = task
        self.report_excluded_records.add(manifest_path)
        self.report_rejected.append(
            {
                "manifest": str(manifest_path),
                "errors": list(freeze.MANUAL_INVALID_REJECTION_ERRORS),
            }
        )
        self.state_raw = _write_pretty(self.state_path, self.state)
        self.state_sha = freeze._sha256_bytes(self.state_raw)
        return manifest_path

    def _evidence_and_selections(self) -> tuple[list[dict], dict[str, dict]]:
        selections: list[dict] = []
        tasks: dict[str, dict] = {}
        ordinal = 0
        for model, setting in freeze._matrix_keys():
            if (model, setting) == freeze.QWEN_OVERRIDE_KEY:
                evidence: list[dict] = []
                for lr, kl in ((1.18e-4, 0.21), (1.25e-4, 0.22)):
                    ordinal += 1
                    item, task = self._add_evidence(
                        model=model,
                        setting=setting,
                        lr=lr,
                        kl=kl,
                        ordinal=ordinal,
                    )
                    evidence.append(item)
                    tasks[task["task_id"]] = task
                selections.append(
                    {
                        "model": model,
                        "setting": setting,
                        "selected_lr": 1.2e-4,
                        "decision_kind": "user_override_interpolated",
                        "evidence": evidence,
                        "rationale": "User-reviewed interpolation.",
                    }
                )
            else:
                ordinal += 1
                lr = float(ordinal) * 1e-6
                item, task = self._add_evidence(
                    model=model,
                    setting=setting,
                    lr=lr,
                    kl=0.1 + ordinal / 1000,
                    ordinal=ordinal,
                )
                tasks[task["task_id"]] = task
                selected_kl = float(item["kl_wikitext2"])
                selections.append(
                    {
                        "model": model,
                        "setting": setting,
                        "selected_lr": lr,
                        "decision_kind": "exact",
                        "evidence": [item],
                        "rationale": "Exact minimum in accepted evidence.",
                    }
                )
                if model == "qwen3-32b":
                    for neighbor_lr in (lr * 0.8, lr * 1.2):
                        ordinal += 1
                        _, neighbor_task = self._add_evidence(
                            model=model,
                            setting=setting,
                            lr=neighbor_lr,
                            kl=selected_kl * 1.01,
                            ordinal=ordinal,
                        )
                        tasks[neighbor_task["task_id"]] = neighbor_task
        return selections, tasks

    def _state(self, tasks: dict[str, dict]) -> dict:
        tuning: dict[str, dict] = {}
        for model, setting in freeze._matrix_keys():
            evidence_count = sum(
                1
                for task in tasks.values()
                if task["spec"]["model"] == model
                and task["spec"]["setting"] == setting
            )
            tuning[f"{model}/{setting}"] = {
                "model": model,
                "setting": setting,
                "status": "NEEDS_USER_ACTION",
                "profile_level": 0,
                "generation": 0,
                "canary_lr": 1e-5,
                "canary_succeeded": True,
                "attempt_count": evidence_count,
                "refinement_done": True,
                "zero_probes_done": [],
                "selected_lr": None,
                "selected_manifest": None,
                "needs_user_action_reason": "manual review",
            }
        return {
            "schema_version": 1,
            "campaign_id": "lowbit-test-campaign",
            "created_utc": "2026-07-25T00:00:00+00:00",
            "updated_utc": "2026-07-25T00:00:00+00:00",
            "status": "TUNING",
            "execute_requested": True,
            "plan": {
                "path": str(self.plan_path.resolve()),
                "tune_sha256": self.plan_sha,
                "tune_protocol_sha256": freeze.base_campaign._protocol_sha256(
                    self.plan
                ),
                "active_sha256": self.plan_sha,
                "formal_sha256": None,
                "formal_protocol_sha256": None,
                "pending_transition_sha256": None,
            },
            "agent": {
                "path": str(self.agent_path.resolve()),
                "ready_marker": freeze.DEFAULT_READY_MARKER,
            },
            "preflight": {"valid": True},
            "source_identity": copy.deepcopy(self.source),
            "heartbeat": None,
            "eta": None,
            "eta_event_tracker": None,
            "tasks": tasks,
            "static_caches": {
                f"tune/{model}": {
                    "phase": "tune",
                    "model": model,
                    "status": "READY",
                    "profile_level": 0,
                    "producer_task": None,
                    "validation": {"valid": True},
                }
                for model in freeze.runner.MODEL_ORDER
            },
            "tuning": tuning,
            "formal": {"initialized": False, "models": {}},
            "selected_lr_patch": None,
            "blockers": [],
            "anomaly_keys": [],
            "event_seq": 0,
            "last_event": None,
        }

    def call(self, *, execute: bool = False, **overrides):
        kwargs = {
            "selection_path": self.selection_path,
            "expected_state_sha256": self.state_sha,
            "expected_plan_sha256": self.plan_sha,
            "expected_campaign_id": self.campaign_id,
            "plan_path": self.plan_path,
            "state_path": self.state_path,
            "agent_path": self.agent_path,
            "ready_marker": freeze.DEFAULT_READY_MARKER,
            "execute": execute,
        }
        kwargs.update(overrides)
        return freeze.freeze_manual_lrs(**kwargs)


class FreezeManualLRTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = Fixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.fixture.close()
        self.temporary.cleanup()

    def test_default_dry_run_is_byte_for_byte_read_only(self):
        fixture = self.fixture
        before_plan = fixture.plan_path.read_bytes()
        before_state = fixture.state_path.read_bytes()
        before_selection = fixture.selection_path.read_bytes()
        before_files = sorted(
            str(path.relative_to(fixture.root))
            for path in fixture.root.rglob("*")
            if path.is_file()
        )

        result = fixture.call()

        self.assertEqual(result["status"], "DRY_RUN")
        self.assertIs(result["preflight_valid"], False)
        self.assertIs(
            result["downstream_required_gate"]["required"], True
        )
        self.assertEqual(fixture.plan_path.read_bytes(), before_plan)
        self.assertEqual(fixture.state_path.read_bytes(), before_state)
        self.assertEqual(
            fixture.selection_path.read_bytes(), before_selection
        )
        self.assertEqual(
            sorted(
                str(path.relative_to(fixture.root))
                for path in fixture.root.rglob("*")
                if path.is_file()
            ),
            before_files,
        )
        self.assertFalse(freeze._journal_path(fixture.state_path).exists())
        self.assertFalse(
            (
                fixture.output_root
                / "_campaign"
                / freeze.ARTIFACT_FILENAME
            ).exists()
        )

    def test_execute_commits_exact_timed_handoff_and_is_idempotent(self):
        fixture = self.fixture
        result = fixture.call(execute=True)
        self.assertEqual(result["status"], "WAIT_PREFLIGHT")

        formal_plan_raw = fixture.plan_path.read_bytes()
        formal_plan = json.loads(formal_plan_raw)
        state = json.loads(fixture.state_path.read_text(encoding="utf-8"))
        wrapper = formal_plan["manual_lr_selection_freeze"]
        ledger = wrapper["ledger"]
        artifact_path = (
            fixture.output_root
            / "_campaign"
            / freeze.ARTIFACT_FILENAME
        )
        artifact_raw = artifact_path.read_bytes()
        artifact = json.loads(artifact_raw)

        freeze.runner.validate_structure(formal_plan)
        self.assertEqual(
            artifact_raw, freeze._canonical_json_bytes(artifact) + b"\n"
        )
        self.assertEqual(
            hashlib.sha256(artifact_raw).hexdigest(),
            wrapper["artifact"]["sha256"],
        )
        self.assertEqual(artifact["ledger"], ledger)
        self.assertEqual(
            artifact["pre_freeze_state"]["sha256"], fixture.state_sha
        )
        self.assertEqual(state["manual_lr_selection_freeze"], wrapper)
        self.assertEqual(
            state["manual_lr_handoff"]["phase"], "committed"
        )
        self.assertEqual(state["plan"]["tune_sha256"], fixture.plan_sha)
        self.assertEqual(
            state["plan"]["active_sha256"],
            hashlib.sha256(formal_plan_raw).hexdigest(),
        )
        self.assertEqual(
            state["plan"]["formal_sha256"],
            state["plan"]["active_sha256"],
        )
        self.assertEqual(
            state["plan"]["tune_protocol_sha256"],
            freeze.base_campaign._protocol_sha256(fixture.plan),
        )
        self.assertEqual(
            state["plan"]["formal_protocol_sha256"],
            freeze.base_campaign._protocol_sha256(formal_plan),
        )
        self.assertIsNone(state["plan"]["pending_transition_sha256"])
        self.assertIs(state["preflight"]["valid"], False)
        self.assertEqual(state["status"], "WAIT_PREFLIGHT")
        self.assertEqual(
            set(state["source_identity"]),
            {
                *freeze.SOURCE_IDENTITY_KEYS,
                "formal_timed_campaign_sha256",
                "formal_timing_gate_sha256",
                "formal_timing_source_set_sha256",
                "formal_timed_executor_sha256",
            },
        )
        override = next(
            item
            for item in ledger["selections"]
            if (item["model"], item["setting"])
            == freeze.QWEN_OVERRIDE_KEY
        )
        self.assertEqual(
            override["decision_kind"], "user_override_interpolated"
        )
        self.assertIsNone(override["selected_manifest"])
        self.assertEqual(override["selected_lr"], 1.2e-4)
        self.assertEqual(
            [item["grad_lr"] for item in override["evidence"]],
            [1.18e-4, 1.25e-4],
        )
        exact = next(
            item
            for item in ledger["selections"]
            if item["decision_kind"] == "exact"
        )
        self.assertEqual(
            exact["selected_manifest"], exact["evidence"][0]["manifest"]
        )

        again = fixture.call(execute=True)
        self.assertEqual(again["status"], "IDEMPOTENT")
        self.assertEqual(fixture.plan_path.read_bytes(), formal_plan_raw)

        changed = copy.deepcopy(
            json.loads(fixture.selection_path.read_text(encoding="utf-8"))
        )
        changed["selections"][0][
            "rationale"
        ] = "Different reviewed decision."
        _write_canonical(fixture.selection_path, changed)
        with self.assertRaisesRegex(
            freeze.FreezeError, "selection|journal"
        ):
            fixture.call(execute=True)

    def test_every_durable_fault_recovers_to_one_committed_state(self):
        points = (
            "after_journal_prepared",
            "after_artifact_written",
            "after_journal_artifact_written",
            "after_state_frozen",
            "after_journal_state_frozen",
            "after_plan_replaced",
            "after_journal_plan_replaced",
            "after_state_committed",
            "after_journal_committed",
        )
        for fault_point in points:
            with self.subTest(fault_point=fault_point):
                case_root = Path(self.temporary.name) / fault_point
                case = Fixture(case_root)
                fired = False

                def inject(point: str):
                    nonlocal fired
                    if not fired and point == fault_point:
                        fired = True
                        raise freeze.InjectedFault(point)

                try:
                    with mock.patch.object(freeze, "_fault", inject):
                        with self.assertRaisesRegex(
                            freeze.InjectedFault, fault_point
                        ):
                            case.call(execute=True)
                    with mock.patch.object(
                        freeze, "_fault", lambda _point: None
                    ):
                        recovered = case.call(execute=True)
                    self.assertIn(
                        recovered["status"],
                        {"WAIT_PREFLIGHT", "IDEMPOTENT"},
                    )
                    plan_sha = hashlib.sha256(
                        case.plan_path.read_bytes()
                    ).hexdigest()
                    state_sha = hashlib.sha256(
                        case.state_path.read_bytes()
                    ).hexdigest()
                    self.assertEqual(
                        plan_sha, recovered["plan"]["formal_sha256"]
                    )
                    self.assertEqual(
                        state_sha, recovered["state"]["committed_sha256"]
                    )
                    journal = json.loads(
                        freeze._journal_path(case.state_path).read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertEqual(journal["phase"], "committed")
                finally:
                    case.close()

    def test_recovery_rebuilds_full_authority_and_rejects_toctou(self):
        fixture = self.fixture
        selected_manifests = {
            item["evidence"][0]["manifest"]
            for item in fixture.selections
            if item["decision_kind"] == "exact"
        }
        nonwinner = next(
            parsed
            for path, parsed in fixture.parsed.items()
            if str(path) not in selected_manifests
            and parsed.model == "qwen3-32b"
        )

        def mutate_nonwinner(point: str):
            if point == "after_journal_prepared":
                nonwinner.log_path.write_bytes(
                    nonwinner.log_path.read_bytes() + b"tamper\n"
                )
                raise freeze.InjectedFault(point)

        with mock.patch.object(freeze, "_fault", mutate_nonwinner):
            with self.assertRaises(freeze.InjectedFault):
                fixture.call(execute=True)
        with self.assertRaisesRegex(
            freeze.FreezeError, "authoritative tune snapshot|differs"
        ):
            fixture.call(execute=True)

    def test_recovery_rejects_raw_prefix_change_before_commit(self):
        fixture = self.fixture
        retirement_path = (
            fixture.output_root
            / str(
                getattr(
                    freeze.results,
                    "RETIREMENT_LEDGER_RELATIVE_PATH",
                    "_campaign/retirements.jsonl",
                )
            )
        )

        def add_prefix(point: str):
            if point == "after_journal_prepared":
                retirement_path.parent.mkdir(parents=True, exist_ok=True)
                retirement_path.write_text("{}\n", encoding="utf-8")
                raise freeze.InjectedFault(point)

        with mock.patch.object(freeze, "_fault", add_prefix):
            with self.assertRaises(freeze.InjectedFault):
                fixture.call(execute=True)
        with self.assertRaisesRegex(
            freeze.FreezeError, "authoritative tune snapshot|raw"
        ):
            fixture.call(execute=True)

    def test_complete_launch_projection_rejects_same_count_substitution(self):
        fixture = self.fixture
        model, setting = freeze._matrix_keys()[0]
        _, task = fixture._add_evidence(
            model=model,
            setting=setting,
            lr=7.77e-4,
            kl=88.0,
            ordinal=9400,
        )
        manifest_path = Path(task["manifest"]).resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["status"] = "failed"
        payload["exit_code"] = 1
        _write_pretty(manifest_path, payload)
        fixture.report_excluded_records.add(manifest_path)
        fixture.report_informational.append(
            {
                "manifest": str(manifest_path),
                "kind": "execution_attempt",
                "status": "failed",
            }
        )

        def substitute_identity(point: str):
            if point == "after_journal_prepared":
                changed = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                changed["execution_id"] = "same-count-substitute"
                _write_pretty(manifest_path, changed)
                raise freeze.InjectedFault(point)

        with mock.patch.object(freeze, "_fault", substitute_identity):
            with self.assertRaises(freeze.InjectedFault):
                fixture.call(execute=True)
        with self.assertRaisesRegex(
            freeze.FreezeError, "authoritative tune snapshot|differs"
        ):
            fixture.call(execute=True)

    def test_recovery_rejects_a_third_plan_hash(self):
        fixture = self.fixture

        def inject(point: str):
            if point == "after_state_frozen":
                raise freeze.InjectedFault(point)

        with mock.patch.object(freeze, "_fault", inject):
            with self.assertRaises(freeze.InjectedFault):
                fixture.call(execute=True)
        fixture.plan_path.write_bytes(fixture.plan_path.read_bytes() + b" ")
        with self.assertRaisesRegex(
            freeze.FreezeError, "third, unauthorized"
        ):
            fixture.call(execute=True)

    def test_selection_is_strict_and_override_cannot_masquerade_exact(self):
        fixture = self.fixture
        fixture.selection_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "selections": fixture.selections,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(freeze.FreezeError, "canonical JSON"):
            fixture.call()

        _write_canonical(
            fixture.selection_path,
            {"schema_version": 1, "selections": fixture.selections},
        )
        root = json.loads(
            fixture.selection_path.read_text(encoding="utf-8")
        )
        override = next(
            item
            for item in root["selections"]
            if (item["model"], item["setting"])
            == freeze.QWEN_OVERRIDE_KEY
        )
        override["decision_kind"] = "exact"
        override["selected_lr"] = 1.18e-4
        override["evidence"] = [override["evidence"][0]]
        _write_canonical(fixture.selection_path, root)
        with self.assertRaisesRegex(
            freeze.FreezeError, "only reviewed override"
        ):
            fixture.call()

    def test_manifest_identity_attempt_and_gpu_gates_fail_closed(self):
        fixture = self.fixture
        first_path = Path(
            fixture.selections[0]["evidence"][0]["manifest"]
        )
        fixture.parsed[first_path].attempt_index = 21
        with self.assertRaisesRegex(freeze.FreezeError, "<=20"):
            fixture.call()

        fixture.parsed[first_path].attempt_index = 1
        fixture.parsed[first_path].identities[
            "runner_sha256"
        ] = _digest("drift")
        with self.assertRaisesRegex(freeze.FreezeError, "runner_sha256"):
            fixture.call()

        fixture.parsed[first_path].identities[
            "runner_sha256"
        ] = fixture.source["runner_sha256"]
        with mock.patch.object(
            freeze,
            "query_gpus",
            lambda: freeze.base_campaign.GPUSnapshot(
                (),
                ({"pid": 123, "gpu_uuid": "GPU-0"},),
            ),
        ):
            with self.assertRaisesRegex(freeze.FreezeError, "GPU compute"):
                fixture.call()

    def test_manifest_only_live_process_and_execution_lock_block(self):
        fixture = self.fixture
        task_id, task = next(iter(fixture.state["tasks"].items()))
        manifest_path = Path(task["manifest"])
        fixture.state["tasks"].pop(task_id)
        fixture.state_raw = _write_pretty(
            fixture.state_path, fixture.state
        )
        fixture.state_sha = freeze._sha256_bytes(fixture.state_raw)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["process"]["pid"] = 7777
        argv = payload["command"]["argv"]
        _write_pretty(manifest_path, payload)
        with (
            mock.patch.object(
                freeze,
                "proc_identity",
                lambda pid: {"start_ticks": 1} if pid == 7777 else None,
            ),
            mock.patch.object(
                freeze,
                "proc_cmdline",
                lambda pid: list(argv) if pid == 7777 else None,
            ),
            mock.patch.object(
                freeze, "cmdline_matches_argv", lambda actual, expected: True
            ),
        ):
            with self.assertRaisesRegex(
                freeze.FreezeError, "output manifest still owns|still alive"
            ):
                fixture.call()

        payload["process"]["pid"] = None
        _write_pretty(manifest_path, payload)
        lock_path = manifest_path.parent / ".execution.lock"
        lock_path.touch()
        descriptor = lock_path.open("r+")
        try:
            fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(
                freeze.FreezeError, "still owns execution lock"
            ):
                fixture.call()
        finally:
            fcntl.flock(descriptor.fileno(), fcntl.LOCK_UN)
            descriptor.close()

    def test_exact_selection_must_be_unique_minimum_over_all_active_points(
        self,
    ):
        fixture = self.fixture
        selected = fixture.selections[0]
        model = selected["model"]
        setting = selected["setting"]
        extra_evidence, extra_task = fixture._add_evidence(
            model=model,
            setting=setting,
            lr=9.9e-5,
            kl=0.00001,
            ordinal=999,
        )
        del extra_evidence
        fixture.state["tasks"][extra_task["task_id"]] = extra_task
        fixture.state["tuning"][f"{model}/{setting}"]["attempt_count"] = 2
        fixture.state_raw = _write_pretty(
            fixture.state_path, fixture.state
        )
        fixture.state_sha = freeze._sha256_bytes(fixture.state_raw)

        with self.assertRaisesRegex(
            freeze.FreezeError, "not the unique accepted active"
        ):
            fixture.call()

        # An exact KL tie is also ambiguous and must not be resolved by order.
        selected_path = Path(selected["evidence"][0]["manifest"])
        fixture.parsed[
            Path(extra_task["manifest"]).resolve()
        ].kl_wikitext2 = (
            fixture.parsed[selected_path].kl_wikitext2
        )
        extra_task["metric"]["kl"] = fixture.parsed[
            selected_path
        ].kl_wikitext2
        fixture.state_raw = _write_pretty(
            fixture.state_path, fixture.state
        )
        fixture.state_sha = freeze._sha256_bytes(fixture.state_raw)
        with self.assertRaisesRegex(
            freeze.FreezeError, "no unique Exact KL minimum"
        ):
            fixture.call()

    def test_qwen32_exact_minimum_requires_strict_two_sided_bracket(self):
        fixture = self.fixture
        selection = next(
            item
            for item in fixture.selections
            if item["model"] == "qwen3-32b"
        )
        key = f"{selection['model']}/{selection['setting']}"
        selected_lr = float(selection["selected_lr"])
        higher_ids = [
            task_id
            for task_id, task in fixture.state["tasks"].items()
            if task["spec"]["model"] == selection["model"]
            and task["spec"]["setting"] == selection["setting"]
            and float(task["spec"]["grad_lr"]) > selected_lr
        ]
        self.assertTrue(higher_ids)
        for task_id in higher_ids:
            manifest = Path(fixture.state["tasks"][task_id]["manifest"])
            fixture.state["tasks"].pop(task_id)
            fixture.parsed.pop(manifest.resolve())
            manifest.unlink()
            (manifest.parent / freeze.results.LOG_FILENAME).unlink()
        fixture.state["tuning"][key]["attempt_count"] -= len(higher_ids)
        fixture.state_raw = _write_pretty(
            fixture.state_path, fixture.state
        )
        fixture.state_sha = freeze._sha256_bytes(fixture.state_raw)

        with self.assertRaisesRegex(
            freeze.FreezeError, "both LR sides"
        ):
            fixture.call()

    def test_qwen32_soft_eight_launch_and_exact_two_percent_rule(self):
        fixture = self.fixture
        selection = next(
            item
            for item in fixture.selections
            if item["model"] == "qwen3-32b"
        )
        selected_path = Path(selection["evidence"][0]["manifest"]).resolve()
        selected = fixture.parsed[selected_path]
        group_points = sorted(
            [
                parsed
                for parsed in fixture.parsed.values()
                if parsed.model == selected.model
                and parsed.setting == selected.setting
            ],
            key=lambda parsed: parsed.grad_lr,
        )
        selected_index = group_points.index(selected)
        neighbors = (
            group_points[selected_index - 1],
            group_points[selected_index + 1],
        )
        exact_boundary = float(
            decimal.Decimal(str(selected.kl_wikitext2))
            * decimal.Decimal("1.02")
        )
        for neighbor in neighbors:
            neighbor.kl_wikitext2 = exact_boundary
        self.assertEqual(fixture.call()["status"], "DRY_RUN")

        neighbors[1].kl_wikitext2 = float(
            decimal.Decimal(str(selected.kl_wikitext2))
            * decimal.Decimal("1.0200001")
        )
        with self.assertRaisesRegex(freeze.FreezeError, "above 2%"):
            fixture.call()

        # Eight is a soft target: once eight immutable launches exist, a
        # unique two-sided minimum is accepted even when the nearest gap is
        # larger than 2%.
        for offset, multiplier in enumerate((0.1, 0.2, 2.0, 3.0, 4.0), 1):
            fixture._add_evidence(
                model=selected.model,
                setting=selected.setting,
                lr=float(selected.grad_lr) * multiplier,
                kl=float(selected.kl_wikitext2) + 1.0 + offset,
                ordinal=9200 + offset,
            )
        self.assertEqual(fixture.call()["status"], "DRY_RUN")

    def test_qwen32_zero_best_requires_eight_launches(self):
        fixture = self.fixture
        selection = next(
            item
            for item in fixture.selections
            if item["model"] == "qwen3-32b"
        )
        selected_path = Path(selection["evidence"][0]["manifest"]).resolve()
        fixture.parsed[selected_path].kl_wikitext2 = 0.0
        selection_root = json.loads(
            fixture.selection_path.read_text(encoding="utf-8")
        )
        selected_input = next(
            item
            for item in selection_root["selections"]
            if item["model"] == selection["model"]
            and item["setting"] == selection["setting"]
        )
        selected_input["evidence"][0]["kl_wikitext2"] = 0.0
        _write_canonical(fixture.selection_path, selection_root)
        with self.assertRaisesRegex(freeze.FreezeError, "above 2%"):
            fixture.call()

        for offset, multiplier in enumerate((0.1, 0.2, 2.0, 3.0, 4.0), 1):
            fixture._add_evidence(
                model=selection["model"],
                setting=selection["setting"],
                lr=float(selection["selected_lr"]) * multiplier,
                kl=1.0 + offset,
                ordinal=9300 + offset,
            )
        self.assertEqual(fixture.call()["status"], "DRY_RUN")

    def test_retirement_ledger_and_report_active_set_must_match_state(self):
        fixture = self.fixture
        invalid = fixture._report()
        invalid["retirement_ledger"]["valid"] = False
        with mock.patch.object(
            freeze.results, "build_report", lambda _root: invalid
        ):
            with self.assertRaisesRegex(
                freeze.FreezeError, "retirement ledger"
            ):
                fixture.call()

        mismatch = fixture._report()
        mismatch["records"] = mismatch["records"][1:]
        with mock.patch.object(
            freeze.results, "build_report", lambda _root: mismatch
        ):
            with self.assertRaisesRegex(
                freeze.FreezeError, "lacks accepted|disagree|absent"
            ):
                fixture.call()

        rejected = fixture._report()
        rejected["rejected"] = [
            {"manifest": "/bad/execution_manifest.json", "errors": ["bad"]}
        ]
        with mock.patch.object(
            freeze.results, "build_report", lambda _root: rejected
        ):
            with self.assertRaisesRegex(
                freeze.FreezeError, "rejected manifests|unresolved"
            ):
                fixture.call()

    def test_results_source_adoption_is_explicit_narrow_and_audited(self):
        fixture = self.fixture
        old = _digest("old-results")
        fixture.state["source_identity"]["results_sha256"] = old
        fixture.state_raw = _write_pretty(
            fixture.state_path, fixture.state
        )
        fixture.state_sha = freeze._sha256_bytes(fixture.state_raw)

        with self.assertRaisesRegex(
            freeze.FreezeError, "expected-old-results-sha256"
        ):
            fixture.call()
        with self.assertRaisesRegex(freeze.FreezeError, "does not match"):
            fixture.call(expected_old_results_sha256=_digest("wrong"))

        result = fixture.call(
            expected_old_results_sha256=old,
            execute=True,
        )
        self.assertEqual(result["status"], "WAIT_PREFLIGHT")
        ledger = json.loads(
            fixture.plan_path.read_text(encoding="utf-8")
        )["manual_lr_selection_freeze"]["ledger"]
        self.assertEqual(
            ledger["results_source_adoption"],
            {
                "kind": freeze.SOURCE_ADOPTION_KIND,
                "expected_old_results_sha256": old,
                "accepted_current_results_sha256": fixture.source[
                    "results_sha256"
                ],
            },
        )
        self.assertEqual(
            ledger["source_identity_before"]["results_sha256"],
            fixture.source["results_sha256"],
        )

    def test_manual_invalid_prepare_is_safe_narrow_and_counted(self):
        fixture = self.fixture
        manifest_path = fixture.add_manual_invalid_result()
        plan_before = fixture.plan_path.read_bytes()
        state_before = fixture.state_path.read_bytes()
        ledger_path = (
            fixture.output_root
            / freeze.MANUAL_INVALID_LEDGER_RELATIVE_PATH
        )
        common = {
            "manifest_path": manifest_path,
            "authorized_by": "experiment-owner",
            "authorization_rationale": (
                "Reviewed the sole transient source change; preserve launch "
                "count but exclude it from numerical evidence."
            ),
            "expected_state_sha256": fixture.state_sha,
            "expected_plan_sha256": fixture.plan_sha,
            "expected_campaign_id": fixture.campaign_id,
            "plan_path": fixture.plan_path,
            "state_path": fixture.state_path,
            "agent_path": fixture.agent_path,
        }

        preview = freeze.prepare_manual_invalid_result(**common)
        self.assertEqual(preview["status"], "PREVIEW")
        self.assertFalse(ledger_path.exists())
        self.assertEqual(fixture.plan_path.read_bytes(), plan_before)
        self.assertEqual(fixture.state_path.read_bytes(), state_before)
        self.assertIs(
            preview["entry"]["launch_event_present"], False
        )
        self.assertIsNone(preview["entry"]["launch_event"])
        self.assertEqual(
            preview["entry"]["numerical_source_sha256_at_start"],
            fixture.source["numerical_source_sha256"],
        )
        self.assertNotEqual(
            preview["entry"]["numerical_source_sha256_at_start"],
            preview["entry"]["numerical_source_sha256_at_end"],
        )

        prepared = freeze.prepare_manual_invalid_result(
            **common, execute=True
        )
        self.assertEqual(prepared["status"], "PREPARED")
        self.assertTrue(ledger_path.is_file())
        self.assertEqual(fixture.plan_path.read_bytes(), plan_before)
        self.assertEqual(fixture.state_path.read_bytes(), state_before)
        self.assertEqual(
            freeze.prepare_manual_invalid_result(
                **common, execute=True
            )["status"],
            "IDEMPOTENT",
        )

        result = fixture.call(execute=True)
        self.assertEqual(result["status"], "WAIT_PREFLIGHT")
        artifact = json.loads(
            (
                fixture.output_root
                / "_campaign"
                / freeze.ARTIFACT_FILENAME
            ).read_text(encoding="utf-8")
        )
        reconciliation = artifact["state_reconciliation"]
        self.assertEqual(len(reconciliation["invalid_result_resolutions"]), 1)
        invalid_launch = [
            item
            for item in reconciliation["tune_launches"]
            if item["classification"] == "manual_invalid_result"
        ]
        self.assertEqual(len(invalid_launch), 1)
        self.assertEqual(
            invalid_launch[0]["manifest"]["absolute_path"],
            str(manifest_path),
        )
        qwen_count = next(
            item["attempt_count"]
            for item in reconciliation["group_attempt_counts"]
            if (item["model"], item["setting"])
            == freeze.MANUAL_INVALID_KEY
        )
        self.assertEqual(qwen_count, 3)
        accepted_paths = {
            item["manifest"]
            for item in reconciliation["accepted_tune_results"]
        }
        self.assertNotIn(str(manifest_path), accepted_paths)

    def test_manual_invalid_ledger_appends_exact_second_reviewed_launch(self):
        fixture = self.fixture
        first_manifest = fixture.add_manual_invalid_result()
        common = {
            "authorized_by": "experiment-owner",
            "authorization_rationale": "Reviewed immutable source drift.",
            "expected_plan_sha256": fixture.plan_sha,
            "expected_campaign_id": fixture.campaign_id,
            "plan_path": fixture.plan_path,
            "state_path": fixture.state_path,
            "agent_path": fixture.agent_path,
        }
        freeze.prepare_manual_invalid_result(
            manifest_path=first_manifest,
            expected_state_sha256=fixture.state_sha,
            execute=True,
            **common,
        )
        ledger_path = (
            fixture.output_root
            / freeze.MANUAL_INVALID_LEDGER_RELATIVE_PATH
        )
        first_prefix = ledger_path.read_bytes()
        launches_before_second = {
            str(path.resolve())
            for path in fixture.output_root.rglob(
                freeze.results.MANIFEST_FILENAME
            )
            if freeze.results._tune_attempt_context(
                json.loads(path.read_text(encoding="utf-8")),
                path.resolve(),
            )
            is not None
        }

        second_manifest = fixture.add_manual_invalid_result(
            model=freeze.QWEN32_MANUAL_INVALID_KEY[0],
            setting=freeze.QWEN32_MANUAL_INVALID_KEY[1],
            lr=freeze.QWEN32_MANUAL_INVALID_LR,
            ordinal=9002,
            source_end=freeze.QWEN32_MANUAL_INVALID_SOURCE_END_SHA256,
        )
        plan_before = fixture.plan_path.read_bytes()
        state_before = fixture.state_path.read_bytes()
        preview = freeze.prepare_manual_invalid_result(
            manifest_path=second_manifest,
            expected_state_sha256=fixture.state_sha,
            **common,
        )
        self.assertEqual(preview["status"], "PREVIEW")
        self.assertEqual(ledger_path.read_bytes(), first_prefix)
        self.assertEqual(preview["entry"]["seq"], 2)
        self.assertEqual(
            preview["entry"]["prev_record_sha256"],
            json.loads(first_prefix)["record_sha256"],
        )
        self.assertEqual(
            preview["entry"]["numerical_source_sha256_at_start"],
            freeze.QWEN32_MANUAL_INVALID_SOURCE_START_SHA256,
        )
        self.assertEqual(
            preview["entry"]["numerical_source_sha256_at_end"],
            freeze.QWEN32_MANUAL_INVALID_SOURCE_END_SHA256,
        )

        prepared = freeze.prepare_manual_invalid_result(
            manifest_path=second_manifest,
            expected_state_sha256=fixture.state_sha,
            execute=True,
            **common,
        )
        self.assertEqual(prepared["status"], "PREPARED")
        ledger_raw = ledger_path.read_bytes()
        self.assertTrue(ledger_raw.startswith(first_prefix))
        entries = [
            json.loads(line) for line in ledger_raw.splitlines()
        ]
        self.assertEqual([entry["seq"] for entry in entries], [1, 2])
        self.assertEqual(
            entries[1]["prev_record_sha256"],
            entries[0]["record_sha256"],
        )
        second_parsed = fixture.parsed[second_manifest]
        self.assertEqual(
            entries[1]["manifest"]["absolute_path"],
            str(second_manifest),
        )
        self.assertEqual(
            entries[1]["manifest"]["sha256"],
            freeze._sha256_bytes(second_manifest.read_bytes()),
        )
        self.assertEqual(
            entries[1]["log"]["sha256"],
            freeze._sha256_bytes(second_parsed.log_path.read_bytes()),
        )
        for identity_field in (
            "plan_sha256",
            "model_sha256",
            "runner_sha256",
            "executor_sha256",
        ):
            self.assertEqual(
                entries[1][identity_field],
                second_parsed.identities[identity_field],
            )
        self.assertEqual(fixture.plan_path.read_bytes(), plan_before)
        self.assertEqual(fixture.state_path.read_bytes(), state_before)
        self.assertEqual(
            freeze.prepare_manual_invalid_result(
                manifest_path=second_manifest,
                expected_state_sha256=fixture.state_sha,
                execute=True,
                **common,
            )["status"],
            "IDEMPOTENT",
        )

        result = fixture.call(execute=True)
        self.assertEqual(result["status"], "WAIT_PREFLIGHT")
        artifact = json.loads(
            (
                fixture.output_root
                / "_campaign"
                / freeze.ARTIFACT_FILENAME
            ).read_text(encoding="utf-8")
        )
        reconciliation = artifact["state_reconciliation"]
        resolutions = reconciliation["invalid_result_resolutions"]
        self.assertEqual(len(resolutions), 2)
        self.assertEqual(
            {
                entry["manifest"]["absolute_path"] for entry in resolutions
            },
            {str(first_manifest), str(second_manifest)},
        )
        projected_launches = reconciliation["tune_launches"]
        self.assertEqual(
            {
                launch["manifest"]["absolute_path"]
                for launch in projected_launches
            },
            launches_before_second | {str(second_manifest)},
        )
        self.assertEqual(
            {
                launch["manifest"]["absolute_path"]
                for launch in projected_launches
                if launch["classification"] == "manual_invalid_result"
            },
            {str(first_manifest), str(second_manifest)},
        )
        qwen32_count = next(
            item["attempt_count"]
            for item in reconciliation["group_attempt_counts"]
            if (item["model"], item["setting"])
            == freeze.QWEN32_MANUAL_INVALID_KEY
        )
        self.assertEqual(
            qwen32_count,
            sum(
                1
                for path_text in launches_before_second
                if (
                    fixture.parsed[Path(path_text)].model,
                    fixture.parsed[Path(path_text)].setting,
                )
                == freeze.QWEN32_MANUAL_INVALID_KEY
            )
            + 1,
        )
        accepted_paths = {
            item["manifest"]
            for item in reconciliation["accepted_tune_results"]
        }
        self.assertNotIn(str(first_manifest), accepted_paths)
        self.assertNotIn(str(second_manifest), accepted_paths)

    def test_manual_invalid_ledger_accepts_both_reviewed_launches_already_present(
        self,
    ):
        fixture = self.fixture
        first_manifest = fixture.add_manual_invalid_result()
        second_manifest = fixture.add_manual_invalid_result(
            model=freeze.QWEN32_MANUAL_INVALID_KEY[0],
            setting=freeze.QWEN32_MANUAL_INVALID_KEY[1],
            lr=freeze.QWEN32_MANUAL_INVALID_LR,
            ordinal=9002,
            source_end=freeze.QWEN32_MANUAL_INVALID_SOURCE_END_SHA256,
        )
        common = {
            "authorized_by": "experiment-owner",
            "authorization_rationale": "Reviewed immutable source drift.",
            "expected_state_sha256": fixture.state_sha,
            "expected_plan_sha256": fixture.plan_sha,
            "expected_campaign_id": fixture.campaign_id,
            "plan_path": fixture.plan_path,
            "state_path": fixture.state_path,
            "agent_path": fixture.agent_path,
        }

        first_preview = freeze.prepare_manual_invalid_result(
            manifest_path=first_manifest,
            **common,
        )
        self.assertEqual(first_preview["status"], "PREVIEW")
        freeze.prepare_manual_invalid_result(
            manifest_path=first_manifest,
            execute=True,
            **common,
        )
        second_preview = freeze.prepare_manual_invalid_result(
            manifest_path=second_manifest,
            **common,
        )
        self.assertEqual(second_preview["status"], "PREVIEW")
        freeze.prepare_manual_invalid_result(
            manifest_path=second_manifest,
            execute=True,
            **common,
        )

        ledger_path = (
            fixture.output_root
            / freeze.MANUAL_INVALID_LEDGER_RELATIVE_PATH
        )
        entries = [
            json.loads(line) for line in ledger_path.read_bytes().splitlines()
        ]
        self.assertEqual(
            {
                entry["manifest"]["absolute_path"] for entry in entries
            },
            {str(first_manifest), str(second_manifest)},
        )
        self.assertEqual([entry["seq"] for entry in entries], [1, 2])

    def test_manual_invalid_ledger_rejects_any_third_target(self):
        fixture = self.fixture
        manifest_path = fixture.add_manual_invalid_result(
            model="qwen3-32b",
            setting="2W4A",
            lr=7e-6,
            ordinal=9003,
            source_end=freeze.QWEN32_MANUAL_INVALID_SOURCE_END_SHA256,
        )
        with self.assertRaisesRegex(
            freeze.FreezeError, "reviewed|authorized|source-drift"
        ):
            freeze.prepare_manual_invalid_result(
                manifest_path=manifest_path,
                authorized_by="experiment-owner",
                authorization_rationale="Must reject an unreviewed target.",
                expected_state_sha256=fixture.state_sha,
                expected_plan_sha256=fixture.plan_sha,
                expected_campaign_id=fixture.campaign_id,
                plan_path=fixture.plan_path,
                state_path=fixture.state_path,
                agent_path=fixture.agent_path,
            )

    def test_uncovered_or_arbitrary_rejected_manifest_still_blocks(self):
        fixture = self.fixture
        manifest_path = fixture.add_manual_invalid_result()
        with self.assertRaisesRegex(
            freeze.FreezeError, "unresolved|rejected"
        ):
            fixture.call()

        fixture.report_rejected[0]["errors"] = ["arbitrary parse failure"]
        with self.assertRaisesRegex(
            freeze.FreezeError, "sole rejected|source-drift"
        ):
            freeze.prepare_manual_invalid_result(
                manifest_path=manifest_path,
                authorized_by="experiment-owner",
                authorization_rationale="Must not authorize arbitrary errors.",
                expected_state_sha256=fixture.state_sha,
                expected_plan_sha256=fixture.plan_sha,
                expected_campaign_id=fixture.campaign_id,
                plan_path=fixture.plan_path,
                state_path=fixture.state_path,
                agent_path=fixture.agent_path,
            )

    def test_manual_invalid_ledger_tamper_blocks_freeze(self):
        fixture = self.fixture
        manifest_path = fixture.add_manual_invalid_result()
        freeze.prepare_manual_invalid_result(
            manifest_path=manifest_path,
            authorized_by="experiment-owner",
            authorization_rationale="Exact reviewed source-drift resolution.",
            expected_state_sha256=fixture.state_sha,
            expected_plan_sha256=fixture.plan_sha,
            expected_campaign_id=fixture.campaign_id,
            plan_path=fixture.plan_path,
            state_path=fixture.state_path,
            agent_path=fixture.agent_path,
            execute=True,
        )
        ledger_path = (
            fixture.output_root
            / freeze.MANUAL_INVALID_LEDGER_RELATIVE_PATH
        )
        entry = json.loads(ledger_path.read_text(encoding="utf-8"))
        entry["authorization"]["rationale"] = "tampered"
        ledger_path.chmod(0o644)
        _write_canonical(ledger_path, entry)
        with self.assertRaisesRegex(
            freeze.FreezeError, "record_sha256|invalid"
        ):
            fixture.call()

    def test_stale_state_is_reconciled_into_terminal_evidence_tasks(self):
        fixture = self.fixture
        task_ids = sorted(fixture.state["tasks"])
        missing_ids = set(task_ids[::3])
        for task_id in missing_ids:
            fixture.state["tasks"].pop(task_id)
        retained = next(iter(fixture.state["tasks"].values()))
        retained["status"] = "RUNNING"
        retained["metric"] = {
            "kl": 999.0,
            "ppl": 999.0,
            "grad_lr": 999.0,
            "identities": {},
        }
        fixture.state["tasks"]["stale-pending-cache"] = {
            "task_id": "stale-pending-cache",
            "spec": {
                "kind": "realq_static",
                "method": "realq_static",
                "phase": "precompute",
                "target_phase": "tune",
                "model": "qwen3-4b",
                "setting": None,
            },
            "status": "PENDING",
            "gpus": [],
            "executor_pid": None,
        }
        for group in fixture.state["tuning"].values():
            group["attempt_count"] = 0
        fixture.state_raw = _write_pretty(
            fixture.state_path, fixture.state
        )
        fixture.state_sha = freeze._sha256_bytes(fixture.state_raw)

        result = fixture.call(execute=True)
        self.assertEqual(result["status"], "WAIT_PREFLIGHT")
        state = json.loads(
            fixture.state_path.read_text(encoding="utf-8")
        )
        artifact = json.loads(
            (
                fixture.output_root
                / "_campaign"
                / freeze.ARTIFACT_FILENAME
            ).read_text(encoding="utf-8")
        )
        reconciliation = artifact["state_reconciliation"]
        imported = reconciliation["imported_tasks"]
        accepted = reconciliation["accepted_tune_results"]
        self.assertEqual(len(imported), len(accepted))
        self.assertTrue(
            all(
                task["status"] == "SUCCEEDED"
                and task["spec"]["kind"]
                == "reconciled_realq_tune_result"
                and task["imported"] is False
                and task["imported_by_manual_lr_freeze"] is True
                for task in imported
            )
        )
        for task in imported:
            self.assertEqual(state["tasks"][task["task_id"]], task)
        self.assertEqual(
            state["tasks"]["stale-pending-cache"]["status"],
            "SUPERSEDED",
        )
        self.assertEqual(retained["status"], "RUNNING")
        retained_after = state["tasks"][retained["task_id"]]
        self.assertEqual(retained_after["status"], "SUPERSEDED")
        self.assertEqual(retained_after["metric"]["kl"], 999.0)
        for count in reconciliation["group_attempt_counts"]:
            key = f"{count['model']}/{count['setting']}"
            self.assertEqual(
                state["tuning"][key]["attempt_count"],
                count["attempt_count"],
            )
        successful_by_manifest: dict[str, list[str]] = {}
        for task_id, task in state["tasks"].items():
            if task.get("status") == "SUCCEEDED" and task.get("manifest"):
                successful_by_manifest.setdefault(
                    str(Path(task["manifest"]).resolve()), []
                ).append(task_id)
        for point in accepted:
            self.assertEqual(
                successful_by_manifest[point["manifest"]],
                ["reconciled_tune:" + point["manifest_sha256"]],
            )

        # A real timed-controller reload and preflight gate must tolerate the
        # custom terminal evidence kind without entering the legacy imported
        # runnable-task profile path.
        controller = freeze.timed_campaign.FormalTimedCampaign(
            plan_path=fixture.plan_path,
            state_path=fixture.state_path,
            events_path=fixture.state_path.with_name("events.jsonl"),
            agent_path=fixture.agent_path,
            ready_marker=freeze.DEFAULT_READY_MARKER,
            execute=False,
            poll_seconds=1.0,
            repo_root=ROOT,
        )
        try:
            gates_ok, _ = controller._gates()
            self.assertIs(gates_ok, False)
        finally:
            controller.close()

    def test_stale_running_is_reconciled_but_live_or_formal_is_rejected(self):
        fixture = self.fixture
        task = next(iter(fixture.state["tasks"].values()))
        task["status"] = "RUNNING"
        fixture.state_raw = _write_pretty(
            fixture.state_path, fixture.state
        )
        fixture.state_sha = freeze._sha256_bytes(fixture.state_raw)
        self.assertEqual(fixture.call()["status"], "DRY_RUN")

        task["executor_pid"] = 4242
        task["executor_start_ticks"] = 101
        task["executor_session_id"] = 202
        task["executor_process_group_id"] = 303
        fixture.state_raw = _write_pretty(
            fixture.state_path, fixture.state
        )
        fixture.state_sha = freeze._sha256_bytes(fixture.state_raw)
        with mock.patch.object(
            freeze,
            "proc_identity",
            lambda pid: (
                {
                    "start_ticks": 101,
                    "session_id": 202,
                    "process_group_id": 303,
                }
                if pid == 4242
                else None
            ),
        ):
            with self.assertRaisesRegex(freeze.FreezeError, "still alive"):
                fixture.call()

        task["status"] = "SUCCEEDED"
        task["executor_pid"] = None
        for field in (
            "executor_start_ticks",
            "executor_session_id",
            "executor_process_group_id",
        ):
            task.pop(field, None)
        task["spec"]["phase"] = "final"
        fixture.state_raw = _write_pretty(
            fixture.state_path, fixture.state
        )
        fixture.state_sha = freeze._sha256_bytes(fixture.state_raw)
        with self.assertRaisesRegex(freeze.FreezeError, "formal tasks"):
            fixture.call()

        with self.assertRaisesRegex(freeze.FreezeError, "CAS failed"):
            fixture.call(expected_state_sha256=_digest("not-state"))


if __name__ == "__main__":
    unittest.main()
