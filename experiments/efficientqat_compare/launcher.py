#!/usr/bin/env python3
"""Auditable, fail-closed launcher for the EfficientQAT comparison.

The default action is ``plan``.  ``launch`` is also a dry run unless the
caller supplies ``--execute`` and every safety gate in plan.json is satisfied.
There is deliberately no stop/delete/kill command in this launcher.
"""

from __future__ import print_function

import argparse
import errno
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time


COMMANDS = ("plan", "launch", "status")
DEFAULT_PLAN = Path(__file__).with_name("plan.json")
EXPECTED_TASKS = (
    "piqa",
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "lambada_openai",
    "ceval-valid",
    "boolq",
    "openbookqa",
    "social_iqa",
)
EXPECTED_PAIRS = {
    ("llama3.2-1b", "W4A16KV16"),
    ("llama3.2-1b", "W3A16KV16"),
    ("llama3.2-1b", "W2A4KV4"),
    ("llama3.2-3b", "W4A16KV16"),
    ("llama3.2-3b", "W3A16KV16"),
    ("llama3.2-3b", "W2A4KV4"),
}


class PlanError(RuntimeError):
    """Raised when a launch-plan invariant is not satisfied."""


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan_sha256(path):
    return _sha256_file(path)


def unresolved_confirmations(plan):
    pending = []
    for key, value in sorted(plan.get("confirmations", {}).items()):
        if value.get("required", False) and value.get("status") != "resolved":
            pending.append(key)
    return pending


def validate_plan(plan):
    errors = []
    execution = plan.get("execution_control", {})
    runtime = plan.get("runtime", {})
    runs = plan.get("runs", [])

    if execution.get("default_command") != "plan":
        errors.append("execution_control.default_command must be plan")
    if execution.get("dry_run_by_default") is not True:
        errors.append("execution_control.dry_run_by_default must be true")
    if tuple(execution.get("allowed_commands", [])) != COMMANDS:
        errors.append("allowed_commands must be exactly plan/launch/status")
    if len(runs) != 6:
        errors.append("plan must contain exactly six runs")

    run_ids = [run.get("run_id") for run in runs]
    if len(set(run_ids)) != len(run_ids):
        errors.append("run_id values must be unique")
    pairs = {(run.get("model"), run.get("setting")) for run in runs}
    if pairs != EXPECTED_PAIRS:
        errors.append("run model/setting matrix is not the required 2x3 matrix")

    train_gpus = [run.get("training_gpu") for run in runs]
    if sorted(train_gpus) != [0, 1, 2, 3, 4, 5]:
        errors.append("training GPUs must be a one-to-one mapping onto 0..5")
    eval_gpus = {run.get("evaluation_gpu") for run in runs}
    if eval_gpus != {6, 7}:
        errors.append("evaluation GPUs must use the dedicated pool 6 and 7")

    if runtime.get("canoe_job_id") != "j-7x9o0je4pk":
        errors.append("runtime.canoe_job_id is not the user-selected job")
    if runtime.get("canoe_pod") != "j-7x9o0je4pk-master-0":
        errors.append("runtime.canoe_pod is not the selected master pod")
    for key in ("workspace", "venv", "hf_home", "output_root"):
        value = runtime.get(key)
        if not value or not os.path.isabs(value):
            errors.append("runtime.%s must be an absolute path" % key)

    calibration = plan.get("calibration_contract", {})
    if (
        calibration.get("num_samples") != 2048
        or calibration.get("seq_len") != 2048
        or calibration.get("seed") != 1
    ):
        errors.append("calibration must be the fixed 2048x2048 seed-1 tensor")
    artifact = calibration.get("token_artifact", {})
    if artifact.get("length") != 2048 or artifact.get("item_shape") != [2048]:
        errors.append("token artifact object structure is not [2048, 2048]")
    expected_digest = artifact.get("sha256", "")
    if len(expected_digest) != 64:
        errors.append("token artifact SHA256 is missing or malformed")

    method = plan.get("method_contract", {})
    quantizer = method.get("weight_quantizer", {})
    if quantizer.get("family") != "EfficientQAT official unsigned asymmetric min-max":
        errors.append("weight quantizer must remain EfficientQAT native min-max")
    if method.get("rotation", {}).get("enabled") is not False:
        errors.append("EfficientQAT rotation must remain disabled")
    if method.get("block_ap", {}).get("precision") != "fp16_amp":
        errors.append("Block-AP precision must be the user-confirmed fp16_amp")
    e2e_qp = method.get("e2e_qp", {})
    if e2e_qp.get("precision") != "bf16":
        errors.append("E2E-QP precision must be bf16")
    if e2e_qp.get("optimizer") != "torch.optim.AdamW":
        errors.append("E2E-QP must use the confirmed torch.optim.AdamW substitution")
    if (
        calibration.get("validation_policy")
        != "disabled_validation_size_0_all_2048_sequences_train_only"
    ):
        errors.append("validation must be disabled with all 2048 sequences for training")
    w2 = method.get("activation_kv_quantization", {}).get("W2A4KV4", {})
    if (
        w2.get("aware") is not False
        or w2.get("qk_hadamard") is not False
        or w2.get("a_clip_ratio") != 0.9
        or w2.get("k_clip_ratio") != 0.9
        or w2.get("v_clip_ratio") != 0.9
    ):
        errors.append("W2A4KV4 must be pure unaware A/K/V QDQ with clip 0.9")

    evaluation = plan.get("unified_eval_contract", {})
    if evaluation.get("implementation_module") != "utils.eval_utils":
        errors.append("evaluation must import the shared RealQ eval module")
    if tuple(evaluation.get("qa_tasks", [])) != EXPECTED_TASKS:
        errors.append("evaluation task list/order differs from RealQ")
    if evaluation.get("lm_eval_version") != "0.4.4":
        errors.append("lm-eval version must be 0.4.4")
    if evaluation.get("kl_topk") != -1:
        errors.append("KL must use the full vocabulary (kl_topk=-1)")
    if evaluation.get("qa_raw_scores_must_be_persisted") is not False:
        errors.append(
            "QA must preserve the current RealQ two-decimal evaluator protocol"
        )
    if "current shared RealQ qa_eval" not in evaluation.get(
        "qa_score_protocol", ""
    ):
        errors.append("QA score protocol must identify the shared RealQ return")

    timing = plan.get("timing_contract", {})
    included_timing = timing.get("quantization_gpu_hours_include", [])
    if "weight materialization" in included_timing:
        errors.append("CPU-only materialization cannot count as GPU-hours")

    code_manifest = plan.get("code_identity", {}).get(
        "controlled_code_manifest", {}
    )
    if not os.path.isabs(code_manifest.get("path", "")):
        errors.append("controlled code manifest path must be absolute")
    if len(code_manifest.get("sha256", "")) != 64:
        errors.append("controlled code manifest SHA256 is missing or malformed")

    expected_resolutions = {
        "block_ap_precision": "fp16_amp",
        "e2e_optimizer": "torch.optim.AdamW compatibility substitution",
        "validation_policy": (
            "disabled; validation_size=0; all 2048 saved sequences are training data"
        ),
    }
    confirmations = plan.get("confirmations", {})
    for key, resolution in expected_resolutions.items():
        confirmation = confirmations.get(key, {})
        if confirmation.get("status") != "resolved":
            errors.append("required confirmation is not resolved: %s" % key)
        elif confirmation.get("resolution") != resolution:
            errors.append("confirmation resolution changed unexpectedly: %s" % key)

    return errors


def load_plan(path):
    plan = _read_json(path)
    errors = validate_plan(plan)
    if errors:
        raise PlanError("Invalid plan:\n- " + "\n- ".join(errors))
    return plan


def verify_artifacts(plan):
    failures = []
    artifact = plan["calibration_contract"]["token_artifact"]
    artifact_path = Path(artifact["path"])
    if not artifact_path.is_file():
        failures.append("missing token artifact: %s" % artifact_path)
    else:
        actual_size = artifact_path.stat().st_size
        if actual_size != artifact["size_bytes"]:
            failures.append(
                "token artifact size mismatch: %s != %s"
                % (actual_size, artifact["size_bytes"])
            )
        actual_digest = _sha256_file(artifact_path)
        if actual_digest != artifact["sha256"]:
            failures.append(
                "token artifact SHA256 mismatch: %s != %s"
                % (actual_digest, artifact["sha256"])
            )

    for model_key, model in sorted(plan["models"].items()):
        model_path = Path(model["path"])
        if not model_path.is_dir():
            failures.append("missing model directory for %s: %s" % (model_key, model_path))
            continue
        config_path = model_path / "config.json"
        if not config_path.is_file():
            failures.append("missing config.json for %s" % model_key)
        elif _sha256_file(config_path) != model["config_sha256"]:
            failures.append("config.json SHA256 mismatch for %s" % model_key)
        for filename in model["checkpoint_files"]:
            if not (model_path / filename).is_file():
                failures.append("missing checkpoint file for %s: %s" % (model_key, filename))

    if failures:
        raise PlanError("Artifact verification failed:\n- " + "\n- ".join(failures))

    expected_lm_eval = plan["unified_eval_contract"]["lm_eval_version"]
    try:
        actual_lm_eval = importlib.metadata.version("lm_eval")
    except importlib.metadata.PackageNotFoundError:
        actual_lm_eval = None
    if actual_lm_eval != expected_lm_eval:
        raise PlanError(
            "lm-eval runtime mismatch: %r != %r"
            % (actual_lm_eval, expected_lm_eval)
        )

    _load_verified_reference_preflight(plan)
    _verify_controlled_code(plan)


def _verify_controlled_code(plan):
    contract = plan["code_identity"]["controlled_code_manifest"]
    manifest_path = Path(contract["path"])
    if not manifest_path.is_file():
        raise PlanError("controlled code manifest is missing: %s" % manifest_path)
    actual_manifest_sha = _sha256_file(manifest_path)
    if actual_manifest_sha != contract["sha256"]:
        raise PlanError(
            "controlled code manifest SHA mismatch: %s != %s"
            % (actual_manifest_sha, contract["sha256"])
        )
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise PlanError("unsupported controlled code manifest schema")
    workspace = Path(plan["runtime"]["workspace"]).resolve()
    failures = []
    for relative_path, expected_sha in sorted(
        manifest.get("files_sha256", {}).items()
    ):
        source_path = (workspace / relative_path).resolve()
        try:
            source_path.relative_to(workspace)
        except ValueError:
            failures.append("source escapes workspace: %s" % relative_path)
            continue
        if not source_path.is_file():
            failures.append("missing controlled source: %s" % relative_path)
        elif _sha256_file(source_path) != expected_sha:
            failures.append("controlled source SHA changed: %s" % relative_path)
    if failures:
        raise PlanError(
            "Controlled source verification failed:\n- "
            + "\n- ".join(failures)
        )
    return manifest


def _load_verified_reference_preflight(plan):
    contract = plan["unified_eval_contract"]["reference_cache_preflight"]
    report_path = Path(contract["path"])
    if not report_path.is_file():
        raise PlanError(
            "reference-cache preflight report is missing: %s" % report_path
        )
    actual_sha = _sha256_file(report_path)
    if actual_sha != contract["sha256"]:
        raise PlanError(
            "reference-cache preflight report SHA mismatch: %s != %s"
            % (actual_sha, contract["sha256"])
        )
    report = _read_json(report_path)
    if report.get("status") != "succeeded":
        raise PlanError("reference-cache preflight did not succeed")
    if report.get("cuda_available") is not False:
        raise PlanError("reference-cache preflight was not CPU-only")
    if report.get("evaluator_sha256") != contract["evaluator_sha256"]:
        raise PlanError("reference-cache preflight evaluator SHA changed")
    if set(report.get("models", {})) != set(contract["models"]):
        raise PlanError("reference-cache preflight model set changed")
    return report


def verify_reference_cache_payloads(plan):
    """Hash both large cache archives once, immediately before launching."""

    report = _load_verified_reference_preflight(plan)
    failures = []
    for model_key, record in sorted(report["models"].items()):
        cache_path = Path(record["cache_path"])
        if not cache_path.is_file():
            failures.append("missing %s cache: %s" % (model_key, cache_path))
            continue
        actual_size = cache_path.stat().st_size
        if actual_size != record["cache_size_bytes"]:
            failures.append(
                "%s cache size mismatch: %s != %s"
                % (model_key, actual_size, record["cache_size_bytes"])
            )
            continue
        actual_sha = _sha256_file(cache_path)
        if actual_sha != record["cache_sha256"]:
            failures.append(
                "%s cache SHA mismatch: %s != %s"
                % (model_key, actual_sha, record["cache_sha256"])
            )
    if failures:
        raise PlanError(
            "Reference-cache payload verification failed:\n- "
            + "\n- ".join(failures)
        )


def _runner_argv(plan_path, plan, run):
    python_path = str(Path(plan["runtime"]["venv"]) / "bin" / "python")
    return [
        python_path,
        "-m",
        plan["execution_control"]["runner_module"],
        "--plan-file",
        str(Path(plan_path).resolve()),
        "--expected-plan-sha256",
        plan_sha256(plan_path),
        "--run-id",
        run["run_id"],
        "--train-device",
        "cuda:0",
        "--eval-device",
        "cuda:1",
        "--physical-train-gpu",
        str(run["training_gpu"]),
        "--physical-eval-gpu",
        str(run["evaluation_gpu"]),
    ]


def _render_command(plan_path, plan, run):
    visible = "%s,%s" % (run["training_gpu"], run["evaluation_gpu"])
    argv = _runner_argv(plan_path, plan, run)
    return "CUDA_VISIBLE_DEVICES=%s %s" % (
        visible,
        " ".join(shlex.quote(value) for value in argv),
    )


def _summary(plan_path, plan):
    return {
        "plan_id": plan["plan_id"],
        "plan_file": str(Path(plan_path).resolve()),
        "plan_sha256": plan_sha256(plan_path),
        "job_id": plan["runtime"]["canoe_job_id"],
        "pod": plan["runtime"]["canoe_pod"],
        "output_root": plan["runtime"]["output_root"],
        "launch_enabled": plan["execution_control"]["launch_enabled"],
        "pending_confirmations": unresolved_confirmations(plan),
        "runs": [
            {
                "run_id": run["run_id"],
                "model": run["model"],
                "setting": run["setting"],
                "training_gpu": run["training_gpu"],
                "evaluation_gpu": run["evaluation_gpu"],
                "status": run["status"],
                "command": _render_command(plan_path, plan, run),
            }
            for run in plan["runs"]
        ],
    }


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


def collect_status(plan):
    output_root = Path(plan["runtime"]["output_root"])
    statuses = []
    for run in plan["runs"]:
        run_dir = output_root / run["output_subdir"]
        launcher_manifest = run_dir / "launcher.json"
        result_manifest = run_dir / "result.json"
        failure_manifest = run_dir / "failure.json"
        state = "planned"
        detail = None
        pid = None

        if result_manifest.is_file():
            state = "completed"
            detail = str(result_manifest)
        elif failure_manifest.is_file():
            state = "failed"
            detail = str(failure_manifest)
        elif launcher_manifest.is_file():
            try:
                launch = _read_json(launcher_manifest)
                pid = launch.get("pid")
                state = "running" if pid and _pid_alive(pid) else "launched_not_running"
                detail = str(launcher_manifest)
            except (OSError, ValueError, TypeError):
                state = "invalid_launcher_manifest"
                detail = str(launcher_manifest)
        elif run_dir.exists():
            state = "output_directory_only"
            detail = str(run_dir)

        statuses.append(
            {
                "run_id": run["run_id"],
                "state": state,
                "pid": pid,
                "detail": detail,
            }
        )
    return statuses


def _write_json_atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp.%s" % os.getpid())
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def execute_launch(plan_path, plan, confirmed_sha):
    execution = plan["execution_control"]
    if execution.get("launch_enabled") is not True:
        raise PlanError("launch_enabled is false; resolve the plan before execution")
    if execution.get("runner_implementation_status") != "ready":
        raise PlanError("runner implementation is not marked ready")
    pending = unresolved_confirmations(plan)
    if pending:
        raise PlanError("required confirmations are unresolved: %s" % ", ".join(pending))

    actual_sha = plan_sha256(plan_path)
    if confirmed_sha != actual_sha:
        raise PlanError(
            "--confirm-plan-sha256 does not match: expected %s" % actual_sha
        )

    expected_pod = plan["runtime"]["canoe_pod"]
    actual_host = socket.gethostname()
    if actual_host != expected_pod:
        raise PlanError(
            "launch must run inside %s; current hostname is %s"
            % (expected_pod, actual_host)
        )

    verify_artifacts(plan)
    verify_reference_cache_payloads(plan)
    runner_module = execution["runner_module"]
    runner_path = (
        Path(plan["runtime"]["workspace"])
        / Path(*runner_module.split("."))
    ).with_suffix(".py")
    if not runner_path.is_file():
        raise PlanError(
            "runner implementation is unavailable: %s (%s)"
            % (runner_module, runner_path)
        )

    output_root = Path(plan["runtime"]["output_root"])
    for run in plan["runs"]:
        run_dir = output_root / run["output_subdir"]
        if (
            (run_dir / "launcher.json").exists()
            or (run_dir / "result.json").exists()
            or (run_dir / "failure.json").exists()
        ):
            raise PlanError(
                "refusing a non-fresh launch; immutable run state exists: %s"
                % run_dir
            )

    launched = []
    for run in plan["runs"]:
        run_dir = output_root / run["output_subdir"]
        run_dir.mkdir(parents=True, exist_ok=False)
        log_path = run_dir / "execution.log"
        argv = _runner_argv(plan_path, plan, run)
        env = os.environ.copy()
        env.update(plan["runtime"]["offline_environment"])
        env["CUDA_VISIBLE_DEVICES"] = "%s,%s" % (
            run["training_gpu"],
            run["evaluation_gpu"],
        )
        env["EFFICIENTQAT_PHYSICAL_TRAIN_GPU"] = str(run["training_gpu"])
        env["EFFICIENTQAT_PHYSICAL_EVAL_GPU"] = str(run["evaluation_gpu"])
        with log_path.open("ab") as log_handle:
            process = subprocess.Popen(
                argv,
                cwd=plan["runtime"]["workspace"],
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        manifest = {
            "schema_version": 1,
            "plan_id": plan["plan_id"],
            "plan_sha256": plan_sha256(plan_path),
            "run_id": run["run_id"],
            "pid": process.pid,
            "argv": argv,
            "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
            "hostname": actual_host,
            "launched_unix_time": time.time(),
            "log_path": str(log_path),
        }
        _write_json_atomic(run_dir / "launcher.json", manifest)
        launched.append(manifest)
    return launched


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    plan_parser = subparsers.add_parser("plan", help="show and validate the plan")
    plan_parser.add_argument("--plan-file", default=str(DEFAULT_PLAN))
    plan_parser.add_argument("--json", action="store_true")
    plan_parser.add_argument("--verify-artifacts", action="store_true")

    launch_parser = subparsers.add_parser(
        "launch", help="render commands; start only with all explicit gates"
    )
    launch_parser.add_argument("--plan-file", default=str(DEFAULT_PLAN))
    launch_parser.add_argument(
        "--execute",
        action="store_true",
        help="request execution; omitted means dry-run",
    )
    launch_parser.add_argument("--confirm-plan-sha256")
    launch_parser.add_argument("--json", action="store_true")

    status_parser = subparsers.add_parser("status", help="read existing run state")
    status_parser.add_argument("--plan-file", default=str(DEFAULT_PLAN))
    status_parser.add_argument("--json", action="store_true")
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        argv = ["plan"]
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        plan_path = Path(args.plan_file)
        plan = load_plan(plan_path)

        if args.command == "plan":
            if args.verify_artifacts:
                verify_artifacts(plan)
            value = _summary(plan_path, plan)
            if args.json:
                print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                print("plan_id: %s" % value["plan_id"])
                print("plan_sha256: %s" % value["plan_sha256"])
                print("job/pod: %s / %s" % (value["job_id"], value["pod"]))
                print("launch_enabled: %s" % value["launch_enabled"])
                print(
                    "pending_confirmations: %s"
                    % (
                        ", ".join(value["pending_confirmations"])
                        if value["pending_confirmations"]
                        else "none"
                    )
                )
                for run in value["runs"]:
                    print(
                        "%s train_gpu=%s eval_gpu=%s %s"
                        % (
                            run["run_id"],
                            run["training_gpu"],
                            run["evaluation_gpu"],
                            run["status"],
                        )
                    )
            return 0

        if args.command == "launch":
            summary = _summary(plan_path, plan)
            if not args.execute:
                value = {
                    "mode": "dry-run",
                    "plan_sha256": summary["plan_sha256"],
                    "launch_enabled": summary["launch_enabled"],
                    "pending_confirmations": summary["pending_confirmations"],
                    "commands": [
                        {"run_id": run["run_id"], "command": run["command"]}
                        for run in summary["runs"]
                    ],
                }
                if args.json:
                    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
                else:
                    print("DRY RUN: no process was started")
                    print("plan_sha256: %s" % value["plan_sha256"])
                    for item in value["commands"]:
                        print("%s: %s" % (item["run_id"], item["command"]))
                return 0
            if not args.confirm_plan_sha256:
                raise PlanError("--execute requires --confirm-plan-sha256")
            launched = execute_launch(plan_path, plan, args.confirm_plan_sha256)
            if args.json:
                print(json.dumps(launched, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                for item in launched:
                    print("launched %s pid=%s" % (item["run_id"], item["pid"]))
            return 0

        if args.command == "status":
            value = collect_status(plan)
            if args.json:
                print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                for item in value:
                    suffix = " (%s)" % item["detail"] if item["detail"] else ""
                    print("%s: %s%s" % (item["run_id"], item["state"], suffix))
            return 0

        raise PlanError("unsupported command: %s" % args.command)
    except PlanError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
