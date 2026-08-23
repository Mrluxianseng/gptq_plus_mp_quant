from __future__ import annotations

import csv
import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "lowbit_activation_gpu_hours.py"
SPEC = importlib.util.spec_from_file_location(
    "lowbit_activation_gpu_hours", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
gpu_hours = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gpu_hours
SPEC.loader.exec_module(gpu_hours)


PLAN_SHA = "1" * 64
MODEL_SHA = "2" * 64
SOURCE_SHA = "3" * 64
CACHE_SHA = "4" * 64
ARTIFACT_SHA = "5" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GPUHourTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _base_manifest(
        self,
        *,
        run_id: str,
        method: str,
        gpu_indices: list[int],
        status: str = "succeeded",
        producer: bool = False,
        target_phase: str = "final",
    ) -> dict:
        world_size = len(gpu_indices)
        if producer and method == "guided_gptq":
            argv = [
                "python",
                "save_grads.py",
                "--exp",
                run_id,
            ]
        elif method == "realq":
            argv = [
                "python",
                "-m",
                "realq.ptq",
                "--exp",
                run_id,
            ]
            if producer:
                argv.extend(["--exit_after_precompute", "true"])
        else:
            argv = [
                "python",
                "ptq.py",
                "--w_method",
                "gptaq" if method == "gptaq" else "gptq_guided",
                "--exp",
                run_id,
            ]
        if world_size != 1:
            argv.extend([f"--nproc-per-node={world_size}"])
        return {
            "schema_version": 1,
            "execution_id": f"exec-{run_id}",
            "run_id": run_id,
            "status": status,
            "exit_code": 0 if status == "succeeded" else 1,
            "command": {
                "argv": argv,
                "env": {
                    "CUDA_VISIBLE_DEVICES": ",".join(
                        str(value) for value in gpu_indices
                    )
                },
            },
            "launch_gate": {
                "gpu": {"requested_gpu_indices": gpu_indices}
            },
            "plan": {
                "sha256": PLAN_SHA,
                "sha256_at_end": PLAN_SHA,
                "changed_during_execution": False,
            },
            "model": {
                "complete": True,
                "combined_identity_sha256": MODEL_SHA,
            },
            "numerical_source_tree": {
                "combined_sha256": SOURCE_SHA,
                "combined_sha256_at_end": SOURCE_SHA,
                "changed_during_execution": False,
            },
        }

    def _write_execution(
        self,
        name: str,
        *,
        model: str,
        setting: str | None,
        method: str,
        seconds: float,
        gpu_indices: list[int],
        phase: str | None = "final",
        target_phase: str | None = None,
        status: str = "succeeded",
        producer_ref: dict | None = None,
        write_sidecar: bool = True,
        timing_status: str = "complete",
        strict_provenance: bool = True,
        attempt_index: int = 1,
        observed_ranks: list[int] | None = None,
    ) -> tuple[Path, dict | None]:
        run_dir = self.root / name
        run_dir.mkdir()
        producer = target_phase is not None
        if producer:
            run_id = (
                f"precompute_realq_static_{target_phase}_{model}"
                if method == "realq"
                else f"precompute_guided_saliency_{model}"
            )
        else:
            run_id = f"{phase}_{method}_{model}_{str(setting).lower()}"
        if attempt_index > 1:
            run_id = f"{run_id}_attempt{attempt_index}"
        manifest = self._base_manifest(
            run_id=run_id,
            method=method,
            gpu_indices=gpu_indices,
            status=status,
            producer=producer,
            target_phase=target_phase or "final",
        )
        manifest_path = run_dir / gpu_hours.MANIFEST_FILENAME
        parsed_component = None
        if write_sidecar:
            identity = {
                "execution_id": manifest["execution_id"],
                "run_id": manifest["run_id"],
                "plan_sha256": PLAN_SHA,
                "model_sha256": MODEL_SHA,
                "numerical_source_sha256": SOURCE_SHA,
            }
            kind = "shared_precompute" if producer else "quantization"
            if producer:
                artifacts = [
                    {
                        "path": f"/cache/{method}/{model}/artifact.pt",
                        "sha256": ARTIFACT_SHA,
                    }
                ]
                artifact_set_sha = gpu_hours._canonical_sha256(artifacts)
                stage = gpu_hours._PRECOMPUTE_STAGES[method]
                refs: list[dict] = []
                cache_outcome = "computed" if status == "succeeded" else "partial"
                cache_identity = CACHE_SHA
            else:
                artifacts = []
                artifact_set_sha = None
                stage = gpu_hours._QUANT_STAGES[method]
                refs = [producer_ref] if producer_ref else []
                cache_outcome = None
                cache_identity = None
            rank_timings = []
            elapsed_ns = int(seconds * 1_000_000_000)
            ranks = (
                list(range(len(gpu_indices)))
                if observed_ranks is None
                else list(observed_ranks)
            )
            for rank in ranks:
                started = None if timing_status == "not_started" else 10
                ended = (
                    None
                    if timing_status == "not_started"
                    else 10 + elapsed_ns
                )
                rank_timings.append(
                    {
                        "rank": rank,
                        "timing_status": timing_status,
                        "started_monotonic_ns": started,
                        "ended_monotonic_ns": ended,
                        "elapsed_seconds": (
                            0.0
                            if timing_status == "not_started"
                            else seconds
                        ),
                    }
                )
            coverage_complete = set(ranks) == set(
                range(len(gpu_indices))
            )
            exact_gpu_hours = coverage_complete and (
                timing_status in {"complete", "partial", "not_started"}
            )
            wall_seconds = (
                0.0 if timing_status == "not_started" else seconds
            )
            exact_hours = (
                wall_seconds * len(gpu_indices) / 3600.0
            )
            lower_bound = (
                exact_hours
                if exact_gpu_hours
                else (
                    sum(
                        float(row["elapsed_seconds"])
                        for row in rank_timings
                    )
                    / 3600.0
                )
            )
            component = {
                "component_id": "",
                "kind": kind,
                "stage": stage,
                "method": method,
                "model": model,
                "setting": setting,
                "phase": None if producer else phase,
                "target_phase": target_phase if producer else None,
                "included_segments": gpu_hours._INCLUDED_SEGMENTS[
                    (kind, method)
                ],
                "excluded_segments": gpu_hours._EXCLUDED_SEGMENTS,
                "clock": "time.perf_counter_ns",
                "timing_status": timing_status,
                "complete": timing_status == "complete",
                "synchronized_start": (
                    timing_status != "not_started"
                ),
                "synchronized_end": timing_status == "complete",
                "gpu_indices": gpu_indices,
                "gpu_count": len(gpu_indices),
                "rank_timings": rank_timings,
                "observed_rank_count": len(rank_timings),
                "rank_coverage_complete": coverage_complete,
                "missing_ranks": sorted(
                    set(range(len(gpu_indices))) - set(ranks)
                ),
                "wall_seconds": wall_seconds,
                "gpu_hour_status": (
                    "exact" if exact_gpu_hours else "lower_bound"
                ),
                "allocated_gpu_hours": (
                    exact_hours if exact_gpu_hours else None
                ),
                "gpu_hours_lower_bound": lower_bound,
                "shared_precompute_refs": refs,
                "cache_outcome": cache_outcome,
                "cache_identity_sha256": cache_identity,
                "artifact_set_sha256": artifact_set_sha,
                "artifacts": artifacts,
            }
            component["component_id"] = (
                gpu_hours.component_identity_sha256(
                    scope_id=gpu_hours.SCOPE_ID,
                    identity=identity,
                    component=component,
                )
            )
            sidecar = {
                "schema_version": gpu_hours.SIDECAR_SCHEMA_VERSION,
                "scope_id": gpu_hours.SCOPE_ID,
                "identity": identity,
                "manifest_status": status,
                "component": component,
            }
            source_set_sha = None
            if strict_provenance:
                snapshot_dir = run_dir / "phase_timing_sources"
                snapshot_dir.mkdir()
                sources = []
                source_sha_by_name = {}
                original_by_name = {}
                for source_name in gpu_hours._FORMAL_TIMING_SOURCE_NAMES:
                    content = f"source:{source_name}\n".encode()
                    snapshot = snapshot_dir / f"{source_name}.py"
                    snapshot.write_bytes(content)
                    digest = hashlib.sha256(content).hexdigest()
                    original = f"/strict/{source_name}.py"
                    source_sha_by_name[source_name] = digest
                    original_by_name[source_name] = original
                    sources.append(
                        {
                            "name": source_name,
                            "original_path": original,
                            "snapshot_path": (
                                f"phase_timing_sources/{source_name}.py"
                            ),
                            "sha256": digest,
                        }
                    )
                source_set_sha = gpu_hours._canonical_sha256(
                    [
                        {
                            "name": item["name"],
                            "sha256": item["sha256"],
                        }
                        for item in sources
                    ]
                )
                mode = {
                    ("shared_precompute", "realq"): (
                        "realq_shared_precompute"
                    ),
                    ("shared_precompute", "guided_gptq"): (
                        "guided_precompute"
                    ),
                    ("quantization", "realq"): "realq_quantization",
                    ("quantization", "gptaq"): "legacy_quantization",
                    ("quantization", "guided_gptq"): (
                        "legacy_quantization"
                    ),
                }[(kind, method)]
                spec_sha = gpu_hours._canonical_sha256(
                    {
                        "schema_version": 1,
                        "scope_id": gpu_hours.SCOPE_ID,
                        "mode": mode,
                        "kind": kind,
                        "method": method,
                        "stage": stage,
                        "model": model,
                        "setting": setting,
                        "phase": None if producer else phase,
                        "target_phase": (
                            target_phase if producer else None
                        ),
                        "run_id": run_id,
                        "gpu_indices": gpu_indices,
                    }
                )
                rank_evidence = []
                for row in rank_timings:
                    rank = row["rank"]
                    row_status = row["timing_status"]
                    payload = {
                        "schema_version": 1,
                        "scope_id": gpu_hours.SCOPE_ID,
                        "source_set_sha256": source_set_sha,
                        "sitecustomize_sha256": source_sha_by_name[
                            "formal_timing_sitecustomize"
                        ],
                        "spec_sha256": spec_sha,
                        "run_id": run_id,
                        "mode": mode,
                        "method": method,
                        "stage": stage,
                        "model": model,
                        "setting": setting,
                        "phase": None if producer else phase,
                        "target_phase": (
                            target_phase if producer else None
                        ),
                        "rank": rank,
                        "local_rank": rank,
                        "world_size": len(gpu_indices),
                        "physical_gpu_index": gpu_indices[rank],
                        "clock": "time.perf_counter_ns",
                        "started_monotonic_ns": row[
                            "started_monotonic_ns"
                        ],
                        "rotation_ended_monotonic_ns": (
                            None
                            if row_status == "not_started"
                            else 11
                        ),
                        "ended_monotonic_ns": row[
                            "ended_monotonic_ns"
                        ],
                        "elapsed_seconds": (
                            None
                            if row_status == "not_started"
                            else row["elapsed_seconds"]
                        ),
                        "synchronized_start": (
                            row_status != "not_started"
                        ),
                        "synchronized_end": row_status == "complete",
                        "timing_status": row_status,
                        "complete": row_status == "complete",
                        "error": (
                            None
                            if row_status == "complete"
                            else f"{row_status} failure"
                        ),
                        "cache_lookups": [],
                        "cache_writes": [],
                        "artifacts": [],
                    }
                    evidence_path = (
                        run_dir / f"phase_timing_rank{rank}.json"
                    )
                    evidence_path.write_text(
                        json.dumps(payload, sort_keys=True),
                        encoding="utf-8",
                    )
                    rank_evidence.append(
                        {
                            "path": evidence_path.name,
                            "sha256": _sha256(evidence_path),
                        }
                    )
                sidecar["timing_provenance"] = {
                    "schema_version": 1,
                    "source_set_sha256": source_set_sha,
                    "sources": sources,
                    "rank_evidence": rank_evidence,
                }
                sidecar["instrumentation"] = {
                    "path": original_by_name[
                        "formal_timing_sitecustomize"
                    ],
                    "sha256": source_sha_by_name[
                        "formal_timing_sitecustomize"
                    ],
                }
            sidecar_path = run_dir / gpu_hours.SIDECAR_FILENAME
            sidecar_path.write_text(
                json.dumps(sidecar, sort_keys=True),
                encoding="utf-8",
            )
            manifest["phase_timing"] = {
                "schema_version": gpu_hours.SIDECAR_SCHEMA_VERSION,
                "scope_id": gpu_hours.SCOPE_ID,
                "path": gpu_hours.SIDECAR_FILENAME,
                "sha256": _sha256(sidecar_path),
                "stable_during_hash": True,
            }
            if source_set_sha is not None:
                manifest["phase_timing"][
                    "source_set_sha256"
                ] = source_set_sha
            parsed_component = component
        manifest["timestamps"] = {"duration_seconds": 999999}
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True),
            encoding="utf-8",
        )
        return manifest_path, parsed_component

    def _record(
        self,
        manifest_path: Path,
        *,
        model: str,
        setting: str,
        method: str,
        world_size: int,
    ) -> dict:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "manifest": str(manifest_path),
            "execution_id": manifest["execution_id"],
            "run_id": manifest["run_id"],
            "model": model,
            "setting": setting,
            "method": method,
            "phase": "final",
            "world_size": world_size,
            "identities": {
                "plan_sha256": PLAN_SHA,
                "model_sha256": MODEL_SHA,
                "numerical_source_sha256": SOURCE_SHA,
            },
        }

    def _report(
        self,
        *,
        records: list[dict],
        informational: list[dict],
        retired_attempts: list[dict] | None = None,
        rejected: list[dict] | None = None,
    ) -> dict:
        return {
            "schema_version": 1,
            "ok": True,
            "records": records,
            "informational": informational,
            "retired_attempts": retired_attempts or [],
            "rejected": rejected or [],
            "matrix_completeness": {"complete": True},
        }

    @staticmethod
    def _producer_ref(component: dict, manifest_path: Path) -> dict:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "component_id": component["component_id"],
            "producer_execution_id": manifest["execution_id"],
            "cache_identity_sha256": component[
                "cache_identity_sha256"
            ],
            "artifact_set_sha256": component[
                "artifact_set_sha256"
            ],
        }

    def test_shared_precompute_is_unique_bf16_is_zero_and_attempt_is_separate(
        self,
    ) -> None:
        realq_pre_path, realq_pre = self._write_execution(
            "realq-pre",
            model="m",
            setting=None,
            method="realq",
            seconds=1800,
            gpu_indices=[0, 1, 2, 3],
            phase=None,
            target_phase="final",
        )
        guided_pre_path, guided_pre = self._write_execution(
            "guided-pre",
            model="m",
            setting=None,
            method="guided_gptq",
            seconds=1800,
            gpu_indices=[4],
            phase=None,
            target_phase="final",
        )
        assert realq_pre is not None and guided_pre is not None
        rq_ref = self._producer_ref(realq_pre, realq_pre_path)
        guided_ref = self._producer_ref(guided_pre, guided_pre_path)

        records = []
        for setting, seconds in (("3W16A", 3600), ("2W16A", 7200)):
            path, _ = self._write_execution(
                f"realq-{setting}",
                model="m",
                setting=setting,
                method="realq",
                seconds=seconds,
                gpu_indices=[0, 1, 2, 3],
                producer_ref=rq_ref,
            )
            records.append(
                self._record(
                    path,
                    model="m",
                    setting=setting,
                    method="realq",
                    world_size=4,
                )
            )
        gptaq_path, _ = self._write_execution(
            "gptaq",
            model="m",
            setting="3W16A",
            method="gptaq",
            seconds=3600,
            gpu_indices=[4],
        )
        records.append(
            self._record(
                gptaq_path,
                model="m",
                setting="3W16A",
                method="gptaq",
                world_size=1,
            )
        )
        guided_path, _ = self._write_execution(
            "guided",
            model="m",
            setting="3W16A",
            method="guided_gptq",
            seconds=3600,
            gpu_indices=[5],
            producer_ref=guided_ref,
        )
        records.append(
            self._record(
                guided_path,
                model="m",
                setting="3W16A",
                method="guided_gptq",
                world_size=1,
            )
        )
        records.append(
            {
                "manifest": "/not/read/for/bf16",
                "execution_id": "bf16",
                "run_id": "final_bf16_m_bf16",
                "model": "m",
                "setting": "BF16",
                "method": "bf16",
                "phase": "final",
                "world_size": 1,
            }
        )
        failed_path, _ = self._write_execution(
            "failed",
            model="m",
            setting="4W4A",
            method="gptaq",
            seconds=1800,
            gpu_indices=[6],
            status="failed",
        )
        report = self._report(
            records=records,
            informational=[
                {
                    "manifest": str(realq_pre_path),
                    "kind": "precompute",
                    "status": "succeeded",
                    "run_id": "precompute_realq_static_final_m",
                },
                {
                    "manifest": str(guided_pre_path),
                    "kind": "precompute",
                    "status": "succeeded",
                    "run_id": "precompute_guided_saliency_m",
                },
                {
                    "manifest": str(failed_path),
                    "kind": "execution_attempt",
                    "status": "failed",
                },
            ],
        )

        result = gpu_hours.build_gpu_hour_report(report)

        self.assertTrue(result["ok"])
        self.assertEqual(len(result["shared_precompute_rows"]), 2)
        self.assertAlmostEqual(
            result["totals"]["exclusive_quantization_gpu_hours"],
            14.0,
        )
        self.assertAlmostEqual(
            result["totals"]["shared_precompute_gpu_hours"],
            2.5,
        )
        self.assertAlmostEqual(
            result["totals"]["unique_successful_gpu_hours"],
            16.5,
        )
        self.assertAlmostEqual(
            result["totals"]["failed_attempt_spend_gpu_hours"],
            0.5,
        )
        self.assertAlmostEqual(
            result["totals"]["unique_total_gpu_hours"],
            17.0,
        )
        self.assertAlmostEqual(
            result["totals"]["unique_total_gpu_hours"],
            result["totals"]["unique_successful_gpu_hours"]
            + result["totals"]["failed_attempt_spend_gpu_hours"],
        )
        bf16 = next(
            row
            for row in result["quantization_rows"]
            if row["method"] == "bf16"
        )
        self.assertEqual(bf16["status"], "not_applicable")
        self.assertIsNone(bf16["exclusive_gpu_hours"])
        self.assertIsNone(bf16["standalone_inclusive_gpu_hours"])
        self.assertIsNone(bf16["primary_gpu_hours"])
        self.assertIsNone(bf16["shared_precompute_gpu_hours"])
        self.assertEqual(
            result["totals"]["bf16_contribution_gpu_hours"], 0.0
        )
        self.assertEqual(
            result["totals"]["unique_total_gpu_hours"], 17.0
        )
        rq_rows = [
            row
            for row in result["quantization_rows"]
            if row["method"] == "realq"
        ]
        self.assertEqual(
            sorted(row["primary_gpu_hours"] for row in rq_rows),
            [6.0, 10.0],
        )
        self.assertEqual(
            [
                row["standalone_inclusive_gpu_hours"]
                for row in rq_rows
            ],
            [row["primary_gpu_hours"] for row in rq_rows],
        )
        self.assertEqual(
            {row["shared_precompute_gpu_hours"] for row in rq_rows},
            {2.0},
        )
        attempt_rows = list(
            csv.DictReader(io.StringIO(gpu_hours.attempts_csv(result)))
        )
        self.assertEqual(len(attempt_rows), 1)
        self.assertEqual(attempt_rows[0]["classification"], "failed")

        results_path = self.root / "results.json"
        json_out = self.root / "gpu-hours.json"
        groups_out = self.root / "groups.csv"
        precompute_out = self.root / "precompute.csv"
        attempts_out = self.root / "attempts.csv"
        results_path.write_text(json.dumps(report), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            return_code = gpu_hours.main(
                [
                    "--results-json",
                    str(results_path),
                    "--json-out",
                    str(json_out),
                    "--groups-csv-out",
                    str(groups_out),
                    "--precompute-csv-out",
                    str(precompute_out),
                    "--attempts-csv-out",
                    str(attempts_out),
                ]
            )
        self.assertEqual(return_code, 0)
        self.assertTrue(json_out.is_file())
        group_rows = list(
            csv.DictReader(io.StringIO(groups_out.read_text(encoding="utf-8")))
        )
        precompute_rows = list(
            csv.DictReader(
                io.StringIO(precompute_out.read_text(encoding="utf-8"))
            )
        )
        cli_attempt_rows = list(
            csv.DictReader(
                io.StringIO(attempts_out.read_text(encoding="utf-8"))
            )
        )
        self.assertEqual(
            len(group_rows),
            5,
        )
        bf16_csv = next(
            row for row in group_rows if row["method"] == "bf16"
        )
        self.assertEqual(bf16_csv["primary_gpu_hours"], "N/A")
        self.assertEqual(
            bf16_csv["standalone_inclusive_gpu_hours"], "N/A"
        )
        self.assertEqual(
            bf16_csv["shared_precompute_gpu_hours"], "N/A"
        )
        self.assertEqual(len(precompute_rows), 2)
        self.assertEqual(len(cli_attempt_rows), 1)

    def test_missing_sidecar_rejects_manifest_duration_fallback(self) -> None:
        path, _ = self._write_execution(
            "missing",
            model="m",
            setting="3W16A",
            method="gptaq",
            seconds=3600,
            gpu_indices=[0],
            write_sidecar=False,
        )
        report = self._report(
            records=[
                self._record(
                    path,
                    model="m",
                    setting="3W16A",
                    method="gptaq",
                    world_size=1,
                )
            ],
            informational=[],
        )

        result = gpu_hours.build_gpu_hour_report(report)

        self.assertFalse(result["ok"])
        self.assertEqual(result["primary_status"], "timing_incomplete")
        self.assertIsNone(result["totals"]["unique_successful_gpu_hours"])
        self.assertIn(
            "phase_timing",
            result["quantization_rows"][0]["issues"][0],
        )

    def test_missing_failed_attempt_is_incomplete_but_primary_stays_complete(
        self,
    ) -> None:
        primary_path, _ = self._write_execution(
            "primary",
            model="m",
            setting="3W16A",
            method="gptaq",
            seconds=3600,
            gpu_indices=[0],
        )
        failed_path, _ = self._write_execution(
            "failed-no-timing",
            model="m",
            setting="4W4A",
            method="gptaq",
            seconds=1,
            gpu_indices=[1],
            status="failed",
            write_sidecar=False,
        )
        report = self._report(
            records=[
                self._record(
                    primary_path,
                    model="m",
                    setting="3W16A",
                    method="gptaq",
                    world_size=1,
                )
            ],
            informational=[
                {
                    "manifest": str(failed_path),
                    "status": "failed",
                    "kind": "execution_attempt",
                }
            ],
        )

        result = gpu_hours.build_gpu_hour_report(report)

        self.assertFalse(result["ok"])
        self.assertEqual(result["primary_status"], "complete")
        self.assertEqual(
            result["attempt_spend_status"], "timing_incomplete"
        )
        self.assertEqual(result["totals"]["unique_successful_gpu_hours"], 1.0)
        self.assertIsNone(
            result["totals"]["failed_attempt_spend_gpu_hours"]
        )
        self.assertIsNone(result["totals"]["unique_total_gpu_hours"])
        self.assertEqual(
            result["totals"][
                "failed_attempt_spend_gpu_hours_lower_bound"
            ],
            0.0,
        )

    def test_retired_oom_is_deduplicated_from_informational(self) -> None:
        primary_path, _ = self._write_execution(
            "primary",
            model="m",
            setting="3W16A",
            method="gptaq",
            seconds=3600,
            gpu_indices=[0],
        )
        retired_path, _ = self._write_execution(
            "retired",
            model="m",
            setting="4W4A",
            method="realq",
            seconds=900,
            gpu_indices=[1],
            phase="final",
            status="failed",
        )
        report = self._report(
            records=[
                self._record(
                    primary_path,
                    model="m",
                    setting="3W16A",
                    method="gptaq",
                    world_size=1,
                )
            ],
            informational=[
                {
                    "manifest": str(retired_path),
                    "status": "failed",
                    "kind": "execution_attempt",
                }
            ],
            retired_attempts=[
                {
                    "manifest": str(retired_path),
                    "status": "oom",
                    "execution_id": json.loads(
                        retired_path.read_text(encoding="utf-8")
                    )["execution_id"],
                    "model": "m",
                    "setting": "4W4A",
                    "phase": "final",
                }
            ],
        )

        result = gpu_hours.build_gpu_hour_report(report)

        self.assertTrue(result["ok"])
        self.assertEqual(len(result["attempt_spend_rows"]), 1)
        self.assertEqual(
            result["attempt_spend_rows"][0]["classification"],
            "retired_oom",
        )
        self.assertAlmostEqual(
            result["totals"]["failed_attempt_spend_gpu_hours"],
            0.25,
        )

    def test_tuning_attempt_spend_is_outside_formal_total(self) -> None:
        primary_path, _ = self._write_execution(
            "primary-with-old-tune-attempt",
            model="m",
            setting="3W16A",
            method="gptaq",
            seconds=3600,
            gpu_indices=[0],
        )
        tune_path, _ = self._write_execution(
            "old-tune-failure",
            model="m",
            setting="4W4A",
            method="realq",
            seconds=900,
            gpu_indices=[1],
            phase="tune",
            status="failed",
            write_sidecar=False,
        )
        result = gpu_hours.build_gpu_hour_report(
            self._report(
                records=[
                    self._record(
                        primary_path,
                        model="m",
                        setting="3W16A",
                        method="gptaq",
                        world_size=1,
                    )
                ],
                informational=[
                    {
                        "manifest": str(tune_path),
                        "kind": "execution_attempt",
                        "status": "failed",
                        "run_id": "tune_realq_m_4w4a",
                        "counts_toward_tune_budget": True,
                    }
                ],
            )
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["attempt_spend_rows"], [])
        self.assertEqual(
            result["totals"]["unique_successful_gpu_hours"], 1.0
        )
        self.assertEqual(
            result["totals"]["unique_total_gpu_hours"], 1.0
        )

    def test_sparse_tuning_attempt_is_excluded_after_sidecar_parse(
        self,
    ) -> None:
        primary_path, _ = self._write_execution(
            "primary-with-sparse-tune-attempt",
            model="m",
            setting="3W16A",
            method="gptaq",
            seconds=3600,
            gpu_indices=[0],
        )
        tune_path, _ = self._write_execution(
            "strict-sparse-tune-failure",
            model="m",
            setting="4W4A",
            method="realq",
            seconds=900,
            gpu_indices=[1],
            phase="tune",
            status="failed",
        )

        result = gpu_hours.build_gpu_hour_report(
            self._report(
                records=[
                    self._record(
                        primary_path,
                        model="m",
                        setting="3W16A",
                        method="gptaq",
                        world_size=1,
                    )
                ],
                informational=[
                    {
                        "manifest": str(tune_path),
                        "kind": "execution_attempt",
                        "status": "failed",
                    }
                ],
            )
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["attempt_spend_rows"], [])
        self.assertEqual(
            result["totals"]["unique_successful_gpu_hours"], 1.0
        )
        self.assertEqual(
            result["totals"]["failed_attempt_spend_gpu_hours"], 0.0
        )
        self.assertEqual(
            result["totals"]["unique_total_gpu_hours"], 1.0
        )

    def test_copied_failed_attempt_evidence_is_counted_once(self) -> None:
        primary_path, _ = self._write_execution(
            "primary-with-copied-attempt",
            model="m",
            setting="3W16A",
            method="gptaq",
            seconds=3600,
            gpu_indices=[0],
        )
        failed_path, _ = self._write_execution(
            "failed-final-attempt",
            model="m",
            setting="4W4A",
            method="realq",
            seconds=900,
            gpu_indices=[1],
            status="failed",
        )
        copied_dir = self.root / "failed-final-attempt-copy"
        shutil.copytree(failed_path.parent, copied_dir)
        copied_path = copied_dir / failed_path.name

        result = gpu_hours.build_gpu_hour_report(
            self._report(
                records=[
                    self._record(
                        primary_path,
                        model="m",
                        setting="3W16A",
                        method="gptaq",
                        world_size=1,
                    )
                ],
                informational=[
                    {
                        "manifest": str(failed_path),
                        "kind": "execution_attempt",
                        "status": "failed",
                    },
                    {
                        "manifest": str(copied_path),
                        "kind": "execution_attempt",
                        "status": "failed",
                    },
                ],
            )
        )

        self.assertTrue(result["ok"])
        self.assertEqual(len(result["attempt_spend_rows"]), 1)
        self.assertAlmostEqual(
            result["totals"]["unique_successful_gpu_hours"], 1.0
        )
        self.assertAlmostEqual(
            result["totals"]["failed_attempt_spend_gpu_hours"], 0.25
        )
        self.assertAlmostEqual(
            result["totals"]["unique_total_gpu_hours"], 1.25
        )

    def test_identity_hash_mismatch_is_timing_incomplete(self) -> None:
        path, _ = self._write_execution(
            "bad-identity",
            model="m",
            setting="3W16A",
            method="gptaq",
            seconds=3600,
            gpu_indices=[0],
        )
        record = self._record(
            path,
            model="m",
            setting="3W16A",
            method="gptaq",
            world_size=1,
        )
        record["identities"]["model_sha256"] = "f" * 64

        result = gpu_hours.build_gpu_hour_report(
            self._report(records=[record], informational=[])
        )

        self.assertFalse(result["ok"])
        self.assertIn(
            "model_sha256",
            result["quantization_rows"][0]["issues"][0],
        )

    def test_partial_failed_attempt_uses_only_observed_algorithm_time(
        self,
    ) -> None:
        primary_path, _ = self._write_execution(
            "primary-for-partial",
            model="m",
            setting="3W16A",
            method="gptaq",
            seconds=3600,
            gpu_indices=[0],
        )
        failed_path, _ = self._write_execution(
            "partial-failed",
            model="m",
            setting="4W4A",
            method="gptaq",
            seconds=12,
            gpu_indices=[1, 2, 3, 4],
            status="failed",
            timing_status="partial",
            strict_provenance=True,
        )
        result = gpu_hours.build_gpu_hour_report(
            self._report(
                records=[
                    self._record(
                        primary_path,
                        model="m",
                        setting="3W16A",
                        method="gptaq",
                        world_size=1,
                    )
                ],
                informational=[
                    {
                        "manifest": str(failed_path),
                        "status": "failed",
                        "kind": "execution_attempt",
                    }
                ],
            )
        )

        self.assertTrue(result["ok"], result["attempt_spend_issues"])
        attempt = result["attempt_spend_rows"][0]
        self.assertEqual(attempt["status"], "recorded")
        self.assertEqual(attempt["wall_seconds"], 12.0)
        self.assertAlmostEqual(attempt["gpu_hours"], 12 * 4 / 3600)
        self.assertNotEqual(attempt["wall_seconds"], 999999)

    def test_missing_allocated_rank_is_lower_bound_not_exact_total(
        self,
    ) -> None:
        primary_path, _ = self._write_execution(
            "primary-for-missing-rank",
            model="m",
            setting="3W16A",
            method="gptaq",
            seconds=3600,
            gpu_indices=[0],
        )
        failed_path, _ = self._write_execution(
            "failed-missing-rank",
            model="m",
            setting="4W4A",
            method="gptaq",
            seconds=12,
            gpu_indices=[1, 2, 3, 4],
            status="failed",
            timing_status="partial",
            observed_ranks=[0, 1, 2],
        )
        result = gpu_hours.build_gpu_hour_report(
            self._report(
                records=[
                    self._record(
                        primary_path,
                        model="m",
                        setting="3W16A",
                        method="gptaq",
                        world_size=1,
                    )
                ],
                informational=[
                    {
                        "manifest": str(failed_path),
                        "status": "failed",
                        "kind": "execution_attempt",
                    }
                ],
            )
        )

        self.assertFalse(result["ok"])
        attempt = result["attempt_spend_rows"][0]
        self.assertEqual(attempt["status"], "timing_incomplete")
        self.assertEqual(attempt["gpu_hour_status"], "lower_bound")
        self.assertIsNone(attempt["gpu_hours"])
        self.assertEqual(attempt["missing_ranks"], [3])
        self.assertAlmostEqual(
            attempt["gpu_hours_lower_bound"], 12 * 3 / 3600
        )
        self.assertIsNone(
            result["totals"]["failed_attempt_spend_gpu_hours"]
        )
        self.assertIsNone(
            result["totals"]["unique_total_gpu_hours"]
        )
        self.assertAlmostEqual(
            result["totals"][
                "failed_attempt_spend_gpu_hours_lower_bound"
            ],
            12 * 3 / 3600,
        )
        self.assertAlmostEqual(
            result["totals"]["unique_total_gpu_hours_lower_bound"],
            1.0 + 12 * 3 / 3600,
        )
        sidecar_path = failed_path.parent / gpu_hours.SIDECAR_FILENAME
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar["component"]["gpu_hour_status"] = "exact"
        sidecar["component"]["allocated_gpu_hours"] = 12 * 4 / 3600
        sidecar_path.write_text(
            json.dumps(sidecar, sort_keys=True),
            encoding="utf-8",
        )
        manifest = json.loads(failed_path.read_text(encoding="utf-8"))
        manifest["phase_timing"]["sha256"] = _sha256(sidecar_path)
        failed_path.write_text(
            json.dumps(manifest, sort_keys=True),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            gpu_hours.GPUHourError,
            "gpu_hour_status",
        ):
            gpu_hours._parse_component(
                manifest,
                failed_path,
                primary=False,
            )

    def test_legacy_sidecar_without_strict_provenance_is_rejected(
        self,
    ) -> None:
        path, _ = self._write_execution(
            "legacy-sidecar",
            model="m",
            setting="3W16A",
            method="gptaq",
            seconds=3,
            gpu_indices=[0],
            strict_provenance=False,
        )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(
            gpu_hours.GPUHourError,
            "source_set_sha256",
        ):
            gpu_hours._parse_component(manifest, path)

    def test_strict_provenance_rejects_snapshot_and_rank_tampering(
        self,
    ) -> None:
        for tamper_kind in ("source", "rank"):
            with self.subTest(tamper=tamper_kind):
                path, _ = self._write_execution(
                    f"strict-{tamper_kind}",
                    model="m",
                    setting="3W16A",
                    method="gptaq",
                    seconds=3,
                    gpu_indices=[0],
                    strict_provenance=True,
                )
                manifest = json.loads(path.read_text(encoding="utf-8"))
                parsed = gpu_hours._parse_component(manifest, path)
                self.assertTrue(parsed["timing_provenance_strict"])
                if tamper_kind == "source":
                    target = (
                        path.parent
                        / "phase_timing_sources"
                        / "formal_timing_adapter.py"
                    )
                else:
                    target = path.parent / "phase_timing_rank0.json"
                target.write_bytes(target.read_bytes() + b"tamper\n")
                with self.assertRaisesRegex(
                    gpu_hours.GPUHourError, "SHA256"
                ):
                    gpu_hours._parse_component(manifest, path)

    def test_guided_attempt2_producer_links_to_consumer(self) -> None:
        producer_path, producer = self._write_execution(
            "guided-attempt2-producer",
            model="m",
            setting=None,
            method="guided_gptq",
            seconds=30,
            gpu_indices=[0],
            phase=None,
            target_phase="final",
            attempt_index=2,
        )
        assert producer is not None
        reference = self._producer_ref(producer, producer_path)
        consumer_path, _ = self._write_execution(
            "guided-attempt2-consumer",
            model="m",
            setting="3W16A",
            method="guided_gptq",
            seconds=60,
            gpu_indices=[1],
            producer_ref=reference,
        )
        report = self._report(
            records=[
                self._record(
                    consumer_path,
                    model="m",
                    setting="3W16A",
                    method="guided_gptq",
                    world_size=1,
                )
            ],
            informational=[
                {
                    "manifest": str(producer_path),
                    "kind": "precompute",
                    "status": "succeeded",
                    "run_id": "precompute_guided_saliency_m_attempt2",
                }
            ],
        )

        result = gpu_hours.build_gpu_hour_report(report)

        self.assertTrue(result["ok"], result["issues"])
        row = result["quantization_rows"][0]
        self.assertEqual(len(row["shared_precompute_ids"]), 1)
        self.assertAlmostEqual(
            row["shared_precompute_gpu_hours"], 30 / 3600
        )


if __name__ == "__main__":
    unittest.main()
