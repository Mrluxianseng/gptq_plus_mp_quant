from __future__ import annotations

import base64
import copy
import csv
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "lowbit_activation_results.py"
SPEC = importlib.util.spec_from_file_location(
    "lowbit_activation_results", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
results = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = results
SPEC.loader.exec_module(results)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
TRUSTED_SOURCE_IDENTITY = (
    results._current_trusted_base_source_identity()
)
COARSE_LRS = (1e-5, 5e-5, 1e-4)
REFINEMENT_LRS = (2e-5, 3e-5, 7e-5)


def experiment_plan(lrs=COARSE_LRS, *, setting="3W16A"):
    w_bits, activation_bits = {
        "2W4A": (2, 4),
        "2W16A": (2, 16),
        "3W16A": (3, 16),
    }[setting]
    return {
        "schema_version": 1,
        "models": {"qwen3-4b": "modelzoo/Qwen3/Qwen3-4B"},
        "settings": {
            setting: {
                "w_bits": w_bits,
                "a_bits": activation_bits,
                "k_bits": activation_bits,
                "v_bits": activation_bits,
            }
        },
        "methods": ["realq", "gptaq", "guided_gptq", "bf16"],
        "fixed_numerics": {
            "dataset": "wikitext2",
            "eval_datasets": ["wikitext2"],
            "eval_seq_len": 2048,
            "num_groups": 4,
            "w_clip": True,
            "act_order": True,
            "percdamp": 0.01,
            "rotate": True,
            "seed": 1,
            "rotation_seed": 0,
            "refresh_seed": 0,
            "grad_clip": 1.0,
            "grad_hessian_topk": -1,
            "kl_topk": -1,
            "saliency_clip_percentile": 0.99,
            "a_loss_clip_scope": "local_backward_chunk",
            "loss_slide_window": True,
            "group_parallel_quant": "rank",
            "fsdp": False,
            "cpu_master": False,
            "quantizer_inner_fastpath": False,
            "w_clip_search_impl": "cartesian_legacy",
            "fisher_fp32_cache": False,
            "act_order_stitch_impl": "full_weight_legacy",
            "w_clip_update_impl": "guarded",
            "w_group_param_layout": "expanded",
            "activation_clip_ratio": 0.9,
            "qwen3_4b_a_loss_ratio": 0.95,
            "other_model_a_loss_ratio": 1.0,
        },
        "tuning": {
            "world_size": 1,
            "nsamples": 256,
            "seq_len": 2048,
            "bsz": 32,
            "global_loss_bsz": 8,
            "hessian_accum_bsz": 32,
            "backward_samples": 16,
            "backward_bsz": 16,
            "final_layer_backward_bsz": 16,
            "blocksize": 256,
            "lm_eval": False,
            "max_attempts_per_model_setting": 20,
            "search_policy": copy.deepcopy(results.EXPECTED_SEARCH_POLICY),
            "lr_candidates": list(lrs),
        },
        "final": {
            "world_size": 4,
            "nsamples": 256,
            "seq_len": 2048,
            "bsz": 128,
            "global_loss_bsz": 32,
            "hessian_accum_bsz": 128,
            "backward_samples": 32,
            "backward_bsz": 32,
            "final_layer_backward_bsz": 32,
            "blocksize": 128,
            "lm_eval": True,
            "lm_eval_batch_size": 32,
        },
        "final_layer_grad_lr_by_model": {"qwen3-4b": 1e-5},
        "selected_grad_lr_by_model_setting": {
            "qwen3-4b": {setting: None}
        },
        "baseline_numerics": {
            "offload_inps": False,
            "gptaq_alpha": 0.25,
        },
        "qwen3_32b_memory_policy": {
            "adjust_only_batch_knobs_on_oom": True,
            "tuning_ladders": {
                "global_loss_bsz": [8, 4, 2, 1],
                "hessian_accum_bsz": [32, 16, 8, 4, 2, 1],
                "backward_bsz": [16, 8, 4, 2, 1],
                "final_layer_backward_bsz": [16, 8, 4, 2, 1],
            },
            "final_ladders": {
                "global_loss_bsz": [32, 16, 8, 4],
                "hessian_accum_bsz": [128, 64, 32, 16, 8, 4, 2, 1],
                "backward_bsz": [32, 16, 8, 4],
                "final_layer_backward_bsz": [32, 16, 8, 4],
                "lm_eval_batch_size": [32, 16, 8, 4, 2, 1],
            },
        },
        "paper_zero_shot_tasks": list(results.EXPECTED_TASKS),
        "output_root": "output/lowbit_activation",
        "static_cache_root": "cache/lowbit_activation/static",
        "legacy_cache_root": "cache/lowbit_activation/legacy",
    }


def _append_values(argv, values):
    for name, value in values.items():
        argv.extend([f"--{name}", str(value)])


def _append_bools(argv, values):
    for name, value in values.items():
        argv.extend([f"--{name}", str(value).lower()])


def _replace_option(argv, name, value):
    index = argv.index(f"--{name}")
    argv[index + 1] = str(value)


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def realq_argv(
    *, phase="tune", lr=5e-5, attempt_index=1, setting="3W16A"
):
    w_bits, activation_bits = {
        "2W4A": (2, 4),
        "2W16A": (2, 16),
        "3W16A": (3, 16),
    }[setting]
    base_exp = (
        f"{phase}_realq_qwen3-4b_{setting.lower()}_lr{lr:.12g}"
    )
    exp = (
        base_exp
        if attempt_index == 1
        else f"{base_exp}_attempt{attempt_index}"
    )
    if phase == "tune":
        argv = ["python3", "-m", "realq.ptq"]
        world_size = 1
    else:
        argv = [
            "/minimax-avatar-new/zhangqian/realq/gptq_plus/.venv/bin/python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=4",
            "-m",
            "realq.ptq",
        ]
        world_size = 4
    phase_values = {
        "bsz": 32 if phase == "tune" else 128,
        "global_loss_bsz": 8 if phase == "tune" else 32,
        "hessian_accum_bsz": 32 if phase == "tune" else 128,
        "backward_samples": 16 if phase == "tune" else 32,
        "backward_bsz": 16 if phase == "tune" else 32,
        "final_layer_backward_bsz": 16 if phase == "tune" else 32,
    }
    argv.extend(["--model", "modelzoo/Qwen3/Qwen3-4B", "--exp", exp])
    values = {
        "dataset": "wikitext2",
        "eval_datasets": "wikitext2",
        "eval_seq_len": 2048,
        "w_bits": w_bits,
        "w_groupsize": -1 if phase == "tune" else 128,
        "a_bits": activation_bits,
        "k_bits": activation_bits,
        "v_bits": activation_bits,
        "a_groupsize": -1,
        "k_groupsize": -1,
        "v_groupsize": -1,
        "a_clip_ratio": 1.0,
        "k_clip_ratio": 1.0,
        "v_clip_ratio": 1.0,
        "num_groups": 4,
        "percdamp": 0.01,
        "blocksize": 256 if phase == "tune" else 128,
        "kl_topk": -1,
        "nsamples": 256,
        "seq_len": 2048,
        "seed": 1,
        "rotation_seed": 0,
        "refresh_seed": 0,
        **phase_values,
        "grad_lr": lr,
        "final_layer_grad_lr": 1e-5,
        "grad_clip": 1.0,
        "grad_lr_layer_schedule": "cosine",
        "grad_lr_layer_base_ratio": 0.01,
        "a_loss_ratio": 0.95,
        "a_loss_clip_scope": "local_backward_chunk",
        "saliency_clip_percentile": 0.99,
        "grad_hessian_topk": -1,
        "group_parallel_quant": "rank",
        "static_cache_path": (
            "cache/lowbit_activation/static/qwen3-4b/"
            f"{phase}_world{world_size}_glbsz{phase_values['global_loss_bsz']}"
        ),
        "w_clip_search_impl": "cartesian_legacy",
        "act_order_stitch_impl": "full_weight_legacy",
        "w_clip_update_impl": "guarded",
        "w_group_param_layout": "expanded",
    }
    if phase == "final":
        values["lm_eval_batch_size"] = 32
    _append_values(argv, values)
    _append_bools(
        argv,
        {
            "w_asym": False,
            "a_asym": False,
            "k_asym": False,
            "v_asym": False,
            "w_clip": True,
            "act_order": True,
            "rotate": True,
            "act_quant_aware_gptq": False,
            "k_cache_quant_aware_gptq": False,
            "loss_slide_window": True,
            "fsdp": False,
            "cpu_master": False,
            "require_static_cache_hit": True,
            "require_reference_cache_hit": True,
            "skip_eval": False,
            "lm_eval": phase == "final",
            "log_column_block_loss": True,
            "quantizer_inner_fastpath": False,
            "fisher_fp32_cache": False,
        },
    )
    return argv, exp


def legacy_argv(method):
    setting = "bf16" if method == "bf16" else "3w16a"
    exp = f"final_{method}_qwen3-4b_{setting}"
    argv = [
        "/minimax-avatar-new/zhangqian/realq/gptq_plus/.venv/bin/python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=1",
        "ptq.py",
        "--model",
        "modelzoo/Qwen3/Qwen3-4B",
        "--exp",
        exp,
    ]
    is_bf16 = method == "bf16"
    values = {
        "dataset": "wikitext2",
        "eval_datasets": "wikitext2",
        "eval_seq_len": 2048,
        "w_bits": 16 if is_bf16 else 3,
        "w_groupsize": -1 if is_bf16 else 128,
        "a_bits": 16,
        "k_bits": 16,
        "v_bits": 16,
        "a_groupsize": -1,
        "k_groupsize": -1,
        "v_groupsize": -1,
        "a_clip_ratio": 1.0,
        "k_clip_ratio": 1.0,
        "v_clip_ratio": 1.0,
        "num_groups": 4,
        "percdamp": 0.01,
        "blocksize": 128,
        "kl_topk": -1,
        "nsamples": 256,
        "seq_len": 2048,
        "seed": 1,
        "rotation_seed": 0,
        "refresh_seed": 0,
        "lm_eval_batch_size": 32,
    }
    if method == "gptaq":
        values["w_method"] = "gptaq"
        values["alpha"] = 0.25
    elif method == "guided_gptq":
        values["w_method"] = "gptq_guided"
    _append_values(argv, values)
    argv.append("--lm_eval")
    if not is_bf16:
        argv.extend(["--w_clip", "--act_order", "--rotate"])
    return argv, exp


def metric_log(
    kl,
    ppl,
    *,
    qa=False,
    ppl_only=False,
    duplicate_metric=False,
    exact_kl_text=None,
    exact_ppl_text=None,
):
    exact_kl_text = (
        format(kl, ".17g") if exact_kl_text is None else exact_kl_text
    )
    exact_ppl_text = (
        format(ppl, ".17g") if exact_ppl_text is None else exact_ppl_text
    )
    if ppl_only:
        metric_table = (
            "| PPL-wikitext2 |\n"
            "| --- |\n"
            f"| {ppl:.2f} |\n"
        )
    else:
        metric_table = (
            "| KL-wikitext2 | PPL-wikitext2 |\n"
            "| --- | --- |\n"
            f"| {kl:.2e} | {ppl:.2f} |\n"
        )
    text = (
        f"INFO Exact KL&PPL on wikitext2: "
        f"{exact_kl_text}, {exact_ppl_text}\n"
        f"INFO KL&PPL on wikitext2: {kl:.2e}, {ppl:.2f}\n"
        f"{metric_table}"
    )
    if duplicate_metric:
        text += metric_table
    if qa:
        values = [float(index) for index in range(51, 61)]
        average = sum(values) / len(values)
        text += (
            "| "
            + " | ".join((*results.EXPECTED_TASKS, "acc_avg"))
            + " |\n| "
            + " | ".join("---" for _ in range(11))
            + " |\n| "
            + " | ".join(
                [*(f"{value:.2f}" for value in values), f"{average:.2f}"]
            )
            + " |\n"
        )
    return text


def result_manifest(
    argv,
    exp,
    *,
    plan_value=None,
    plan_sha=SHA_A,
    status="succeeded",
    attempt_index=1,
):
    plan_value = copy.deepcopy(plan_value or experiment_plan())
    return {
        "schema_version": 1,
        "execution_id": f"execution-{exp}",
        "run_id": exp,
        "status": status,
        "exit_code": 0 if status == "succeeded" else 1,
        "command": {
            "argv": argv,
            "cwd": "/repo",
            "env": {
                "LOWBIT_ACTIVATION_ATTEMPT_INDEX": str(attempt_index)
            },
        },
        "plan": {
            "path": "/repo/experiments/lowbit_activation/plan.json",
            "sha256": plan_sha,
            "sha256_at_end": plan_sha,
            "changed_during_execution": False,
            "size_bytes": 100,
            "content": plan_value,
        },
        "model": {
            "complete": True,
            "argument": "modelzoo/Qwen3/Qwen3-4B",
            "resolved_path": "/repo/modelzoo/Qwen3/Qwen3-4B",
            "combined_identity_sha256": SHA_B,
            "config": {
                "sha256": SHA_C,
                "stable_during_hash": True,
            },
            "shards": [
                {
                    "relative_path": "model.safetensors",
                    "sampled_sha256": SHA_D,
                    "stable_during_hash": True,
                }
            ],
            "missing_shards": [],
        },
        "python_packages": {
            "lm_eval": {
                "distribution": "lm-eval",
                "version": results.EXPECTED_LM_EVAL_VERSION,
            }
        },
        "source_files": {
            "runner": {
                "exists": True,
                "path": "/repo/tools/lowbit_activation_runner.py",
                "sha256": TRUSTED_SOURCE_IDENTITY["runner_sha256"],
                "stable_during_hash": True,
            },
            "executor": {
                "exists": True,
                "path": "/repo/tools/lowbit_activation_execute.py",
                "sha256": TRUSTED_SOURCE_IDENTITY["executor_sha256"],
                "stable_during_hash": True,
            },
        },
        "numerical_source_tree": {
            "file_count": 42,
            "combined_sha256": TRUSTED_SOURCE_IDENTITY[
                "numerical_source_sha256"
            ],
            "combined_sha256_at_end": TRUSTED_SOURCE_IDENTITY[
                "numerical_source_sha256"
            ],
            "changed_during_execution": False,
        },
    }


class LowBitActivationResultsTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_world_size_accepts_exact_venv_and_legacy_launchers(self):
        common = [
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=4",
            "-m",
            "realq.ptq",
        ]
        self.assertEqual(
            results._command_world_size(
                [
                    (
                        "/minimax-avatar-new/zhangqian/realq/gptq_plus/"
                        ".venv/bin/python"
                    ),
                    "-m",
                    "torch.distributed.run",
                    *common,
                ]
            ),
            4,
        )
        self.assertEqual(
            results._command_world_size(["torchrun", *common]),
            4,
        )
        with self.assertRaisesRegex(
            results.ResultError,
            "at most one",
        ):
            results._command_world_size(
                [
                    "torchrun",
                    (
                        "/minimax-avatar-new/zhangqian/realq/gptq_plus/"
                        ".venv/bin/python"
                    ),
                    "-m",
                    "torch.distributed.run",
                    *common,
                ]
            )

    def add_realq(
        self,
        name,
        *,
        phase="tune",
        lr=5e-5,
        kl=0.1,
        ppl=10.0,
        qa=False,
        status="succeeded",
        attempt_index=1,
        profile_level=0,
        plan_value=None,
        plan_sha=SHA_A,
        failure_log=None,
        setting="3W16A",
        **metric_options,
    ):
        run_dir = self.root / name
        run_dir.mkdir(parents=True)
        argv, exp = realq_argv(
            phase=phase,
            lr=lr,
            attempt_index=attempt_index,
            setting=setting,
        )
        if profile_level:
            policy = (plan_value or experiment_plan())[
                "qwen3_32b_memory_policy"
            ]
            ladder_name = (
                "tuning_ladders" if phase == "tune" else "final_ladders"
            )
            for option, values in policy[ladder_name].items():
                if option == "global_loss_bsz":
                    continue
                _replace_option(
                    argv,
                    option,
                    values[min(profile_level, len(values) - 1)],
                )
        if plan_value is None:
            plan_value = experiment_plan(setting=setting)
            if phase == "final":
                plan_value["selected_grad_lr_by_model_setting"][
                    "qwen3-4b"
                ][setting] = lr
        payload = result_manifest(
            argv,
            exp,
            plan_value=plan_value,
            plan_sha=plan_sha,
            status=status,
            attempt_index=attempt_index,
        )
        (run_dir / results.MANIFEST_FILENAME).write_text(
            json.dumps(payload), encoding="utf-8"
        )
        if status == "succeeded":
            (run_dir / results.LOG_FILENAME).write_text(
                metric_log(
                    kl,
                    ppl,
                    qa=qa,
                    **metric_options,
                ),
                encoding="utf-8",
            )
        elif failure_log is not None:
            (run_dir / results.LOG_FILENAME).write_text(
                failure_log, encoding="utf-8"
            )
        return run_dir

    def add_legacy(
        self,
        name,
        *,
        method,
        kl=0.0,
        ppl=9.0,
        plan_value=None,
        plan_sha=SHA_E,
    ):
        run_dir = self.root / name
        run_dir.mkdir(parents=True)
        argv, exp = legacy_argv(method)
        payload = result_manifest(
            argv,
            exp,
            plan_value=plan_value or experiment_plan(),
            plan_sha=plan_sha,
        )
        (run_dir / results.MANIFEST_FILENAME).write_text(
            json.dumps(payload), encoding="utf-8"
        )
        (run_dir / results.LOG_FILENAME).write_text(
            metric_log(
                kl,
                ppl,
                qa=True,
                ppl_only=method == "bf16",
                exact_kl_text="N/A" if method == "bf16" else None,
            ),
            encoding="utf-8",
        )
        return run_dir

    def add_refined_sweep(
        self,
        *,
        exact_close=False,
        plan_value=None,
        setting="3W16A",
    ):
        if exact_close:
            values = {
                1e-5: 0.1009,
                5e-5: 0.10044,
                1e-4: 0.1008,
                2e-5: 0.10043,
                3e-5: 0.10042,
                7e-5: 0.10041,
            }
        else:
            values = {
                1e-5: 0.3,
                5e-5: 0.1,
                1e-4: 0.2,
                2e-5: 0.14,
                3e-5: 0.12,
                7e-5: 0.13,
            }
        for index, lr in enumerate((*COARSE_LRS, *REFINEMENT_LRS)):
            self.add_realq(
                f"tune-{index}",
                lr=lr,
                kl=values[lr],
                ppl=10.0 + values[lr],
                plan_value=plan_value,
                setting=setting,
            )

    def _add_reviewed_manual_invalid_launch(
        self,
        name,
        *,
        model,
        setting,
        lr,
        source_end,
    ):
        plan = experiment_plan(setting=setting)
        plan["output_root"] = str(self.root.resolve())
        if model == "qwen3-32b":
            plan["models"] = {
                "qwen3-32b": "modelzoo/Qwen3/Qwen3-32B"
            }
        run = self.add_realq(
            name,
            lr=lr,
            plan_value=plan,
            setting=setting,
        )
        manifest_path = run / results.MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if model == "qwen3-32b":
            argv = manifest["command"]["argv"]
            argv[argv.index("--model") + 1] = "modelzoo/Qwen3/Qwen3-32B"
            manifest["model"]["argument"] = "modelzoo/Qwen3/Qwen3-32B"
            manifest["model"]["resolved_path"] = (
                "/repo/modelzoo/Qwen3/Qwen3-32B"
            )
            manifest["model"]["combined_identity_sha256"] = SHA_E
            old_run_id = manifest["run_id"]
            new_run_id = old_run_id.replace(
                "qwen3-4b", "qwen3-32b", 1
            )
            manifest["run_id"] = new_run_id
            manifest["execution_id"] = f"execution-{new_run_id}"
            exp_index = argv.index("--exp") + 1
            argv[exp_index] = new_run_id
            for option, value in (
                ("--a_bits", "4"),
                ("--k_bits", "4"),
                ("--v_bits", "4"),
                ("--a_loss_ratio", "1.0"),
            ):
                argv[argv.index(option) + 1] = value
        manifest["numerical_source_tree"][
            "combined_sha256_at_end"
        ] = source_end
        manifest["numerical_source_tree"][
            "changed_during_execution"
        ] = True
        manifest_path.write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return manifest_path.resolve()

    def _manual_invalid_entry(
        self,
        manifest_path,
        *,
        campaign_id,
        seq,
        previous_hash,
    ):
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        identity = results._manual_lr_manifest_launch_identity(
            manifest, manifest_path=manifest_path
        )
        entry = {
            "schema_version": 1,
            "seq": seq,
            "timestamp_utc": f"2026-07-25T0{seq}:00:00+00:00",
            "campaign_id": campaign_id,
            "reason_code": "numerical_source_changed_during_execution",
            "authorization": {
                "kind": "user_authorized_manual_invalid_result_v1",
                "authorized_by": "experiment-owner",
                "rationale": "Reviewed immutable source drift.",
            },
            "launch_event_present": False,
            "launch_event": None,
            "manifest": results._manual_lr_file_ref(
                manifest_path,
                root=self.root.resolve(),
                name="test invalid manifest",
            ),
            "log": results._manual_lr_file_ref(
                manifest_path.parent / results.LOG_FILENAME,
                root=self.root.resolve(),
                name="test invalid log",
            ),
            "execution_id": identity["execution_id"],
            "run_id": identity["run_id"],
            "phase": "tune",
            "method": "realq",
            "model": identity["model"],
            "setting": identity["setting"],
            "grad_lr": identity["grad_lr"],
            "attempt_index": identity["attempt_index"],
            "manifest_status": identity["status"],
            "exit_code": identity["exit_code"],
            "plan_sha256": identity["plan_sha256"],
            "model_sha256": identity["model_sha256"],
            "runner_sha256": identity["runner_sha256"],
            "executor_sha256": identity["executor_sha256"],
            "numerical_source_sha256_at_start": identity[
                "numerical_source_sha256_at_start"
            ],
            "numerical_source_sha256_at_end": identity[
                "numerical_source_sha256_at_end"
            ],
            "numerical_source_changed_during_execution": identity[
                "numerical_source_changed_during_execution"
            ],
            "expected_rejection_errors": [
                (
                    "manifest.numerical_source_tree."
                    "changed_during_execution must be false"
                )
            ],
            "prev_record_sha256": previous_hash,
            "record_sha256": "",
        }
        unsigned = dict(entry)
        unsigned.pop("record_sha256")
        entry["record_sha256"] = results._strict_canonical_sha256(
            unsigned, "test manual invalid entry"
        )
        return entry

    def test_two_reviewed_manual_invalid_results_preserve_launch_projection(self):
        campaign_id = "campaign-two-reviewed-invalid-results"
        first_manifest = self._add_reviewed_manual_invalid_launch(
            "invalid-qwen4",
            model="qwen3-4b",
            setting="2W16A",
            lr=1.5e-4,
            source_end=(
                "817c218d7d969a754323d5d77dc1cb6f40d833b6be2ba818831223ad67d5a63e"
            ),
        )
        second_manifest = self._add_reviewed_manual_invalid_launch(
            "invalid-qwen32",
            model="qwen3-32b",
            setting="2W4A",
            lr=5e-6,
            source_end=(
                "2c4f87b78eb862526cd00b534cbac65df89f116ed730b635485998be064018a2"
            ),
        )
        first = self._manual_invalid_entry(
            first_manifest,
            campaign_id=campaign_id,
            seq=1,
            previous_hash=None,
        )
        second = self._manual_invalid_entry(
            second_manifest,
            campaign_id=campaign_id,
            seq=2,
            previous_hash=first["record_sha256"],
        )
        ledger_path = self.root / results.MANUAL_INVALID_LEDGER_RELATIVE_PATH
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_bytes(
            results._strict_canonical_json_bytes(
                first, "test first manual invalid entry"
            )
            + b"\n"
            + results._strict_canonical_json_bytes(
                second, "test second manual invalid entry"
            )
            + b"\n"
        )
        source_identity = {
            **TRUSTED_SOURCE_IDENTITY,
            "numerical_source_sha256": (
                "53c916a9ababb36c4e9fc16a78d04a51d360ebe46794495f3b18c992249cb64a"
            ),
        }
        resolutions = results._load_manual_invalid_result_resolutions(
            self.root.resolve(),
            campaign_id=campaign_id,
            plan_sha256=SHA_A,
            source_identity=source_identity,
        )
        self.assertEqual(resolutions, [first, second])
        rejected = [
            {
                "manifest": str(path),
                "errors": list(
                    results.REVIEWED_MANUAL_INVALID_RESULT[
                        "rejection_errors"
                    ]
                ),
            }
            for path in (first_manifest, second_manifest)
        ]
        launches = results._manual_lr_tune_launch_projection(
            self.root.resolve(),
            matrix_order=(
                ("qwen3-4b", "2W16A"),
                ("qwen3-32b", "2W4A"),
            ),
            records=[],
            rejected=rejected,
            informational=[],
            retired_attempts=[],
            invalid_result_resolutions=resolutions,
        )
        self.assertEqual(len(launches), 2)
        self.assertEqual(
            {
                item["manifest"]["absolute_path"] for item in launches
            },
            {str(first_manifest), str(second_manifest)},
        )
        self.assertEqual(
            {
                item["classification"] for item in launches
            },
            {"manual_invalid_result"},
        )
        self.assertEqual(
            [
                (
                    entry["model"],
                    entry["setting"],
                    entry["grad_lr"],
                    entry["attempt_index"],
                )
                for entry in resolutions
            ],
            list(results.REVIEWED_MANUAL_INVALID_RESULTS),
        )

        third = copy.deepcopy(second)
        third["seq"] = 3
        third["grad_lr"] = 7e-6
        third["prev_record_sha256"] = second["record_sha256"]
        unsigned = dict(third)
        unsigned.pop("record_sha256")
        third["record_sha256"] = results._strict_canonical_sha256(
            unsigned, "test unreviewed manual invalid entry"
        )
        ledger_path.write_bytes(
            ledger_path.read_bytes()
            + results._strict_canonical_json_bytes(
                third, "test unreviewed manual invalid entry"
            )
            + b"\n"
        )
        with self.assertRaisesRegex(
            results.ResultError, "reviewed target set"
        ):
            results._load_manual_invalid_result_resolutions(
                self.root.resolve(),
                campaign_id=campaign_id,
                plan_sha256=SHA_A,
                source_identity=source_identity,
            )

    def _freeze_evidence(self, run):
        manifest_path = (run / results.MANIFEST_FILENAME).resolve()
        record = results.parse_result(manifest_path)
        return {
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "log": str(record.log_path),
            "log_sha256": _sha256_file(record.log_path),
            "execution_id": record.execution_id,
            "run_id": record.run_id,
            "attempt_index": record.attempt_index,
            "grad_lr": record.grad_lr,
            "kl_wikitext2": record.kl_wikitext2,
            **record.identities,
        }

    def _rehash_manual_lr_freeze(self, plan):
        wrapper = plan["manual_lr_selection_freeze"]
        ledger = wrapper["ledger"]
        projection = []
        for selection in ledger["selections"]:
            selection["evidence_set_sha256"] = (
                results._strict_canonical_sha256(
                    selection["evidence"], "test evidence"
                )
            )
            projection.append(
                {
                    "model": selection["model"],
                    "setting": selection["setting"],
                    "selected_lr": selection["selected_lr"],
                    "decision_kind": selection["decision_kind"],
                    "evidence": [
                        {
                            "manifest": item["manifest"],
                            "grad_lr": item["grad_lr"],
                            "kl_wikitext2": item["kl_wikitext2"],
                        }
                        for item in selection["evidence"]
                    ],
                    "rationale": selection["rationale"],
                }
            )
        ledger["selection_sha256"] = results._strict_canonical_sha256(
            {"schema_version": 1, "selections": projection},
            "test selection",
        )
        wrapper["ledger_sha256"] = results._strict_canonical_sha256(
            ledger, "test ledger"
        )

    def _write_manual_lr_attestation(
        self,
        plan,
        *,
        timing_gate=None,
        old_results_sha=None,
        pre_tasks=None,
    ):
        wrapper = plan["manual_lr_selection_freeze"]
        ledger = wrapper["ledger"]
        timing_gate = (
            results._current_formal_timing_gate()
            if timing_gate is None
            else copy.deepcopy(timing_gate)
        )
        ledger["timing_gate_sha256"] = timing_gate["gate_sha256"]
        ledger["timing_source_set_sha256"] = timing_gate[
            "source_set_sha256"
        ]
        tune_manifest_path = Path(
            ledger["selections"][0]["evidence"][0]["manifest"]
        )
        tune_record = results.parse_result(tune_manifest_path)
        tune_protocol_sha = results._canonical_sha256(
            tune_record.plan_content
        )
        pre_source_identity = copy.deepcopy(
            ledger["source_identity_before"]
        )
        if old_results_sha is not None:
            pre_source_identity["results_sha256"] = old_results_sha
            ledger["results_source_adoption"] = {
                "kind": results.MANUAL_LR_RESULTS_ADOPTION_KIND,
                "expected_old_results_sha256": old_results_sha,
                "accepted_current_results_sha256": (
                    ledger["source_identity_before"]["results_sha256"]
                ),
            }
        else:
            ledger["results_source_adoption"] = None
        pre_tasks = copy.deepcopy(pre_tasks or {})
        pre_state = {
            "schema_version": 1,
            "campaign_id": ledger["campaign_id"],
            "plan": {
                "tune_sha256": ledger["expected_plan_sha256"],
                "tune_protocol_sha256": tune_protocol_sha,
                "active_sha256": ledger["expected_plan_sha256"],
                "formal_sha256": None,
                "formal_protocol_sha256": None,
                "pending_transition_sha256": None,
            },
            "source_identity": pre_source_identity,
            "formal": {"initialized": False, "models": {}},
            "tasks": pre_tasks,
        }
        pre_state_raw = (
            json.dumps(pre_state, indent=2, sort_keys=True) + "\n"
        ).encode()
        ledger["expected_state_sha256"] = hashlib.sha256(
            pre_state_raw
        ).hexdigest()
        self._rehash_manual_lr_freeze(plan)

        pre_report = results.build_report(self.root)
        accepted_records = [
            results.parse_result(Path(item["manifest"]))
            for item in pre_report["records"]
        ]
        reconciliation_snapshot = (
            results._manual_lr_reconciliation_snapshot(
                root=self.root.resolve(),
                plan=plan,
                records=accepted_records,
                tune_selections=pre_report["realq_tune_selection"],
                retirement_ledger=pre_report["retirement_ledger"],
                rejected=pre_report["rejected"],
                informational=pre_report["informational"],
                retired_attempts=pre_report["retired_attempts"],
                campaign_id=ledger["campaign_id"],
                expected_plan_sha256=ledger["expected_plan_sha256"],
                source_identity=ledger["source_identity_before"],
            )
        )
        raw_prefixes = results._manual_lr_raw_input_prefixes(
            self.root.resolve()
        )
        terminalized_tasks = []
        terminal_states = {
            "SUCCEEDED",
            "FAILED",
            "OOM",
            "INVALID_RESULT",
            "ORPHANED",
            "SUPERSEDED",
            "CANCELLED",
            "LAUNCH_FAILED",
        }
        for task_id, task in sorted(
            pre_tasks.items(), key=lambda item: str(item[0])
        ):
            previous = task["status"]
            spec = task.get("spec")
            tune_realq = (
                isinstance(spec, dict)
                and spec.get("kind") == "realq"
                and spec.get("phase") == "tune"
            )
            if not (
                (tune_realq and previous != "SUPERSEDED")
                or previous not in terminal_states
            ):
                continue
            terminalized_tasks.append(
                {
                    "task_id": str(task_id),
                    "previous_status": previous,
                    "reconciled_status": "SUPERSEDED",
                    "reason": (
                        "old_tune_task_superseded_by_authoritative_import"
                        if tune_realq
                        else (
                            "stale_runnable_task_closed_before_formal_handoff"
                        )
                    ),
                }
            )
        created_utc = "2026-07-25T06:00:00+00:00"
        imported_tasks = []
        for point in reconciliation_snapshot["accepted_tune_results"]:
            imported_tasks.append(
                {
                    "task_id": (
                        "reconciled_tune:" + point["manifest_sha256"]
                    ),
                    "spec": {
                        "kind": "reconciled_realq_tune_result",
                        "method": "realq",
                        "phase": "tune",
                        "target_phase": None,
                        "model": point["model"],
                        "setting": point["setting"],
                        "grad_lr": point["grad_lr"],
                        "overrides": {},
                        "profile_level": None,
                        "static_profile_level": None,
                        "generation": None,
                        "purpose": (
                            "manual_lr_freeze_authoritative_import"
                        ),
                    },
                    "status": "SUCCEEDED",
                    "gpu_count": point["world_size"],
                    "priority": 0,
                    "gpus": [],
                    "attempt_index": point["attempt_index"],
                    "executor_pid": None,
                    "created_utc": created_utc,
                    "launched_utc": None,
                    "finished_utc": created_utc,
                    "output_dir": str(Path(point["manifest"]).parent),
                    "manifest": point["manifest"],
                    "log": point["log"],
                    "exit_code": 0,
                    "failure_class": None,
                    "cache_validation": None,
                    "imported": False,
                    "plan_compatible": True,
                    "metric": {
                        "kl": point["kl_wikitext2"],
                        "ppl": point["ppl_wikitext2"],
                        "grad_lr": point["grad_lr"],
                        "identities": {
                            key: point[key]
                            for key in (
                                "plan_sha256",
                                "model_sha256",
                                "runner_sha256",
                                "executor_sha256",
                                "numerical_source_sha256",
                            )
                        },
                    },
                    "imported_by_manual_lr_freeze": True,
                    "reconciliation_reason": (
                        "retirement_filtered_results_authoritative_import"
                    ),
                }
            )
        report_snapshot = {
            "accepted_active_set_sha256": reconciliation_snapshot[
                "accepted_active_set_sha256"
            ],
            "group_attempt_counts_sha256": reconciliation_snapshot[
                "group_attempt_counts_sha256"
            ],
            "tune_launches_sha256": reconciliation_snapshot[
                "tune_launches_sha256"
            ],
            "invalid_result_resolutions_sha256": reconciliation_snapshot[
                "invalid_result_resolutions_sha256"
            ],
            "raw_input_prefixes_sha256": (
                results._strict_canonical_sha256(
                    raw_prefixes, "test raw input prefixes"
                )
            ),
        }
        state_reconciliation = {
            "schema_version": 1,
            "policy_id": results.MANUAL_LR_RECONCILIATION_POLICY_ID,
            **copy.deepcopy(reconciliation_snapshot),
            "raw_input_prefixes": raw_prefixes,
            "raw_input_prefixes_sha256": report_snapshot[
                "raw_input_prefixes_sha256"
            ],
            "report_snapshot_sha256": (
                results._strict_canonical_sha256(
                    report_snapshot, "test report snapshot"
                )
            ),
            "terminalized_tasks": terminalized_tasks,
            "terminalized_tasks_sha256": (
                results._strict_canonical_sha256(
                    terminalized_tasks, "test terminalized tasks"
                )
            ),
            "imported_tasks": imported_tasks,
            "imported_tasks_sha256": (
                results._strict_canonical_sha256(
                    imported_tasks, "test imported tasks"
                )
            ),
        }
        artifact = {
            "schema_version": 1,
            "scope_id": results.MANUAL_LR_FREEZE_SCOPE_ID,
            "artifact_kind": results.MANUAL_LR_FREEZE_ARTIFACT_KIND,
            "transaction_id": "manual-lr-freeze-transaction-test",
            "created_utc": created_utc,
            "ledger_sha256": wrapper["ledger_sha256"],
            "ledger": copy.deepcopy(ledger),
            "pre_freeze_state": {
                "sha256": ledger["expected_state_sha256"],
                "size_bytes": len(pre_state_raw),
                "content_base64": base64.b64encode(
                    pre_state_raw
                ).decode(),
            },
            "timing_gate": timing_gate,
            "state_reconciliation": state_reconciliation,
        }
        artifact_raw = (
            results._strict_canonical_json_bytes(
                artifact, "test artifact"
            )
            + b"\n"
        )
        campaign_dir = self.root / "_campaign"
        campaign_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = (
            campaign_dir
            / results.MANUAL_LR_FREEZE_ARTIFACT_FILENAME
        )
        artifact_path.write_bytes(artifact_raw)
        wrapper["artifact"] = {
            "filename": results.MANUAL_LR_FREEZE_ARTIFACT_FILENAME,
            "sha256": hashlib.sha256(artifact_raw).hexdigest(),
            "size_bytes": len(artifact_raw),
        }
        formal_plan_path = campaign_dir / "formal_plan.json"
        formal_plan_raw = (
            json.dumps(
                plan,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        formal_plan_path.write_bytes(formal_plan_raw)
        formal_plan_sha = hashlib.sha256(formal_plan_raw).hexdigest()
        self._formal_plan_sha = formal_plan_sha
        state = {
            "schema_version": 1,
            "campaign_id": ledger["campaign_id"],
            "plan": {
                "path": str(formal_plan_path.resolve()),
                "tune_sha256": ledger["expected_plan_sha256"],
                "tune_protocol_sha256": tune_protocol_sha,
                "active_sha256": formal_plan_sha,
                "formal_sha256": formal_plan_sha,
                "formal_protocol_sha256": results._canonical_sha256(plan),
                "pending_transition_sha256": None,
            },
            "manual_lr_selection_freeze": copy.deepcopy(wrapper),
            "manual_lr_handoff": {
                "schema_version": 1,
                "scope_id": results.MANUAL_LR_FREEZE_SCOPE_ID,
                "transaction_id": artifact["transaction_id"],
                "phase": "committed",
                "artifact_sha256": wrapper["artifact"]["sha256"],
                "ledger_sha256": wrapper["ledger_sha256"],
                "selection_sha256": ledger["selection_sha256"],
                "expected_state_sha256": ledger[
                    "expected_state_sha256"
                ],
                "expected_plan_sha256": ledger[
                    "expected_plan_sha256"
                ],
                "formal_plan_sha256": formal_plan_sha,
            },
            "manual_lr_state_reconciliation": {
                "schema_version": 1,
                "policy_id": results.MANUAL_LR_RECONCILIATION_POLICY_ID,
                "phase": "committed",
                "artifact_sha256": wrapper["artifact"]["sha256"],
                "accepted_active_set_sha256": state_reconciliation[
                    "accepted_active_set_sha256"
                ],
                "group_attempt_counts_sha256": state_reconciliation[
                    "group_attempt_counts_sha256"
                ],
                "tune_launches_sha256": state_reconciliation[
                    "tune_launches_sha256"
                ],
                "invalid_result_resolutions_sha256": state_reconciliation[
                    "invalid_result_resolutions_sha256"
                ],
                "raw_input_prefixes_sha256": state_reconciliation[
                    "raw_input_prefixes_sha256"
                ],
                "report_snapshot_sha256": state_reconciliation[
                    "report_snapshot_sha256"
                ],
                "terminalized_tasks_sha256": state_reconciliation[
                    "terminalized_tasks_sha256"
                ],
                "imported_tasks_sha256": state_reconciliation[
                    "imported_tasks_sha256"
                ],
            },
            "downstream_required_gate": copy.deepcopy(
                ledger["downstream_required_gate"]
            ),
            "source_identity": {
                **copy.deepcopy(ledger["source_identity_before"]),
                "formal_timed_campaign_sha256": timing_gate[
                    "controller_sha256"
                ],
                "formal_timing_gate_sha256": timing_gate["gate_sha256"],
                "formal_timing_source_set_sha256": timing_gate[
                    "source_set_sha256"
                ],
                "formal_timed_executor_sha256": timing_gate[
                    "timed_executor_sha256"
                ],
            },
            "formal": {
                "initialized": True,
                "timing_gate": timing_gate,
            },
            "tasks": copy.deepcopy(pre_tasks),
        }
        counts_by_pair = {
            (item["model"], item["setting"]): item["attempt_count"]
            for item in state_reconciliation["group_attempt_counts"]
        }
        selected_map = {
            str(model): {
                str(setting): next(
                    selection["selected_lr"]
                    for selection in ledger["selections"]
                    if selection["model"] == str(model)
                    and selection["setting"] == str(setting)
                )
                for setting in plan["settings"]
            }
            for model in plan["models"]
        }
        state["selected_lr_patch"] = {
            "selected_grad_lr_by_model_setting": selected_map
        }
        state["tuning"] = {}
        for selection in ledger["selections"]:
            pair = (selection["model"], selection["setting"])
            group_key = f"{pair[0]}/{pair[1]}"
            state["tuning"][group_key] = {
                "model": pair[0],
                "setting": pair[1],
                "status": "SELECTED",
                "selected_lr": selection["selected_lr"],
                "selected_manifest": selection["selected_manifest"],
                "attempt_count": counts_by_pair[pair],
                "needs_user_action_reason": None,
                "manual_selection_decision_kind": selection[
                    "decision_kind"
                ],
                "manual_selection_evidence_set_sha256": selection[
                    "evidence_set_sha256"
                ],
                "selected_parameters": {
                    "grad_lr": selection["selected_lr"],
                    "decision_kind": selection["decision_kind"],
                    "rationale": selection["rationale"],
                    "selected_manifest": selection["selected_manifest"],
                    "evidence_set_sha256": selection[
                        "evidence_set_sha256"
                    ],
                    "evidence": copy.deepcopy(selection["evidence"]),
                    "manual_lr_selection_scope_id": (
                        results.MANUAL_LR_FREEZE_SCOPE_ID
                    ),
                    "selection_sha256": ledger["selection_sha256"],
                    "ledger_sha256": wrapper["ledger_sha256"],
                },
            }
        for transition in terminalized_tasks:
            task = state["tasks"][transition["task_id"]]
            task["reconciled_previous_status"] = transition[
                "previous_status"
            ]
            task["status"] = transition["reconciled_status"]
            task["reconciliation_reason"] = transition["reason"]
            task["gpus"] = []
            task["executor_pid"] = None
            for field in (
                "executor_start_ticks",
                "executor_session_id",
                "executor_process_group_id",
                "executor_cli",
                "rendezvous",
            ):
                task.pop(field, None)
            task["finished_utc"] = (
                task.get("finished_utc") or created_utc
            )
        state["tasks"].update(
            {
                task["task_id"]: copy.deepcopy(task)
                for task in imported_tasks
            }
        )
        (campaign_dir / "state.json").write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n"
        )

    def _manual_override_plan(
        self,
        base_plan,
        evidence_runs,
        *,
        selected_lr=1.2e-4,
        decision_kind="user_override_interpolated",
    ):
        plan = copy.deepcopy(base_plan)
        setting = next(iter(plan["settings"]))
        plan["selected_grad_lr_by_model_setting"]["qwen3-4b"][
            setting
        ] = selected_lr
        evidence = [
            self._freeze_evidence(run) for run in evidence_runs
        ]
        selection = {
            "model": "qwen3-4b",
            "setting": setting,
            "selected_lr": selected_lr,
            "decision_kind": decision_kind,
            "rationale": (
                "reviewed interpolation between two adjacent arms"
                if decision_kind == "user_override_interpolated"
                else "exact minimum selected by reviewed tuning protocol"
            ),
            "selected_manifest": (
                None
                if decision_kind == "user_override_interpolated"
                else evidence[0]["manifest"]
            ),
            "evidence_set_sha256": SHA_A,
            "evidence": evidence,
        }
        ledger = {
            "schema_version": 1,
            "scope_id": results.MANUAL_LR_FREEZE_SCOPE_ID,
            "campaign_id": "campaign-manual-lr-freeze-test",
            "selection_sha256": SHA_A,
            "expected_state_sha256": SHA_B,
            "expected_plan_sha256": SHA_A,
            "source_identity_before": {
                **copy.deepcopy(TRUSTED_SOURCE_IDENTITY),
            },
            "timing_gate_sha256": SHA_C,
            "timing_source_set_sha256": SHA_D,
            "results_source_adoption": None,
            "downstream_required_gate": {
                "gate_id": results.MANUAL_LR_FREEZE_GATE_ID,
                "required": decision_kind == "user_override_interpolated",
                "status": (
                    "required_before_formal_results_acceptance"
                    if decision_kind == "user_override_interpolated"
                    else "not_required"
                ),
            },
            "selections": [selection],
        }
        plan["manual_lr_selection_freeze"] = {
            "schema_version": 1,
            "scope_id": results.MANUAL_LR_FREEZE_SCOPE_ID,
            "ledger_sha256": SHA_A,
            "artifact": {
                "filename": results.MANUAL_LR_FREEZE_ARTIFACT_FILENAME,
                "sha256": SHA_A,
                "size_bytes": 1,
            },
            "ledger": ledger,
        }
        self._write_manual_lr_attestation(plan)
        return plan

    def _manual_override_fixture(self, *, complete_grid=True):
        base_plan = experiment_plan(setting="2W16A")
        base_plan["output_root"] = str(self.root.resolve())
        values = {
            1e-5: 0.3,
            5e-5: 0.1,
            1e-4: 0.2,
            2e-5: 0.14,
            3e-5: 0.12,
            7e-5: 0.13,
        }
        if complete_grid:
            for index, lr in enumerate((*COARSE_LRS, *REFINEMENT_LRS)):
                self.add_realq(
                    f"override-tune-{index}",
                    lr=lr,
                    kl=values[lr],
                    ppl=10.0 + values[lr],
                    plan_value=base_plan,
                    setting="2W16A",
                )
        lower = self.add_realq(
            "override-evidence-lower",
            lr=1.18e-4,
            kl=0.21,
            ppl=10.21,
            plan_value=base_plan,
            setting="2W16A",
        )
        upper = self.add_realq(
            "override-evidence-upper",
            lr=1.25e-4,
            kl=0.22,
            ppl=10.22,
            plan_value=base_plan,
            setting="2W16A",
        )
        formal_plan = self._manual_override_plan(
            base_plan, [lower, upper]
        )
        return base_plan, lower, upper, formal_plan

    def _manual_exact_incomplete_fixture(
        self,
        *,
        selected_lr=3e-5,
    ):
        base_plan = experiment_plan(
            lrs=(1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4, 2e-4)
        )
        base_plan["output_root"] = str(self.root.resolve())
        values = {
            1e-5: 0.2,
            2e-5: 0.12,
            3e-5: 0.1,
            5e-5: 0.15,
        }
        runs = {}
        for index, (lr, kl) in enumerate(values.items()):
            runs[lr] = self.add_realq(
                f"manual-exact-tune-{index}",
                lr=lr,
                kl=kl,
                ppl=100.0 if lr == 3e-5 else 1.0 + index,
                plan_value=base_plan,
            )
        formal_plan = self._manual_override_plan(
            base_plan,
            [runs[selected_lr]],
            selected_lr=selected_lr,
            decision_kind="exact",
        )
        return base_plan, runs, formal_plan

    def _artifact_reference(self, path):
        path = path.resolve()
        return {
            "absolute_path": str(path),
            "relative_path": path.relative_to(self.root.resolve()).as_posix(),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    def _write_tune_retirement(self, retired_runs, trigger_run):
        campaign_id = "campaign-retirement-test"
        campaign_dir = self.root / "_campaign"
        campaign_dir.mkdir(parents=True, exist_ok=True)
        events = []
        items = []
        trigger_path = (
            trigger_run / results.MANIFEST_FILENAME
        ).resolve()
        for sequence, run in enumerate(retired_runs, 1):
            manifest_path = (run / results.MANIFEST_FILENAME).resolve()
            log_path = (run / results.LOG_FILENAME).resolve()
            context = results._retirement_manifest_context(
                manifest_path, log_path
            )
            status = "oom" if manifest_path == trigger_path else "succeeded"
            task_id = f"task:{run.name}"
            item = {
                "task_id": task_id,
                "launch_event_seq": sequence,
                "phase": context["phase"],
                "model": context["model"],
                "setting": context["setting"],
                "grad_lr": context["grad_lr"],
                "profile_level": 0,
                "generation": 0,
                "effective_overrides": {},
                "status": status,
                "manifest_status": context["manifest_status"],
                "manifest": self._artifact_reference(manifest_path),
                "log": self._artifact_reference(log_path),
                "execution_id": context["execution_id"],
                "run_id": context["run_id"],
                "plan_sha256": context["plan_sha256"],
                "model_sha256": context["model_sha256"],
            }
            items.append(item)
            events.append(
                {
                    "schema_version": 1,
                    "seq": sequence,
                    "timestamp_utc": "2026-07-25T00:00:00+00:00",
                    "campaign_id": campaign_id,
                    "category": "task",
                    "code": "task_launching",
                    "details": {
                        "task_id": task_id,
                        "plan_sha256": context["plan_sha256"],
                        "phase": context["phase"],
                        "model": context["model"],
                        "setting": context["setting"],
                        "grad_lr": context["grad_lr"],
                        "profile_level": 0,
                        "generation": 0,
                        "effective_overrides": {},
                        "output_dir": str(manifest_path.parent),
                    },
                }
            )
        items.sort(key=lambda item: item["manifest"]["absolute_path"])
        trigger = next(
            item
            for item in items
            if item["manifest"]["absolute_path"] == str(trigger_path)
        )
        plan_sha = trigger["plan_sha256"]
        base = {
            "schema_version": 1,
            "seq": 1,
            "timestamp_utc": "2026-07-25T00:01:00+00:00",
            "campaign_id": campaign_id,
            "prev_record_sha256": None,
            "plan_sha256": plan_sha,
            "phase": "tune",
            "model": "qwen3-4b",
            "setting": "3W16A",
            "reason": "oom_profile_transition",
            "trigger": trigger,
            "old_profile": {
                "level": 0,
                "generation": 0,
                "effective_overrides": {},
            },
            "replacement_profile": {
                "level": 1,
                "generation": 1,
                "effective_overrides": {
                    "hessian_accum_bsz": 16,
                    "backward_bsz": 8,
                    "final_layer_backward_bsz": 8,
                },
            },
            "retired_manifests": items,
        }
        entry = {
            **base,
            "record_sha256": results._canonical_sha256(base),
        }
        events_path = campaign_dir / "events.jsonl"
        events_path.write_text(
            "".join(
                json.dumps(
                    event,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
                for event in events
            ),
            encoding="utf-8",
        )
        ledger_path = campaign_dir / "retirements.jsonl"
        ledger_path.write_text(
            json.dumps(
                entry,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        return ledger_path

    def _retirement_fixture(self):
        self.add_refined_sweep()
        retired_runs = [self.root / f"tune-{index}" for index in range(6)]
        trigger = self.add_realq(
            "old-oom",
            lr=9e-4,
            status="failed",
            failure_log="torch.cuda.OutOfMemoryError: CUDA out of memory\n",
        )
        retired_runs.append(trigger)
        values = {
            1e-5: 0.3,
            5e-5: 0.1,
            1e-4: 0.2,
            2e-5: 0.14,
            3e-5: 0.12,
            7e-5: 0.13,
        }
        for index, lr in enumerate((*COARSE_LRS, *REFINEMENT_LRS)):
            self.add_realq(
                f"active-{index}",
                lr=lr,
                kl=values[lr],
                ppl=10.0 + values[lr],
                attempt_index=2,
                profile_level=1,
            )
        return self._write_tune_retirement(retired_runs, trigger)

    def _rewrite_retirement(self, ledger_path, mutate, *, rehash=True):
        entry = json.loads(ledger_path.read_text(encoding="utf-8"))
        mutate(entry)
        if rehash:
            unsigned = dict(entry)
            unsigned.pop("record_sha256", None)
            entry["record_sha256"] = results._canonical_sha256(unsigned)
        ledger_path.write_text(
            json.dumps(
                entry,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )

    def test_valid_retirement_excludes_old_success_but_counts_attempts(self):
        self._retirement_fixture()

        report = results.build_report(self.root)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["retirement_ledger"]["valid"])
        self.assertEqual(report["accepted_count"], 6)
        self.assertEqual(report["retired_attempt_count"], 7)
        self.assertEqual(
            {item["status"] for item in report["retired_attempts"]},
            {"succeeded", "oom"},
        )
        selection = report["realq_tune_selection"][0]
        self.assertEqual(selection["status"], "selected")
        self.assertEqual(selection["attempt_count"], 13)
        self.assertEqual(selection["candidate_count"], 6)
        self.assertTrue(
            all(
                record["params"]["hessian_accum_bsz"] == 16
                for record in report["records"]
            )
        )

    def test_retirement_artifact_tamper_fails_closed_without_hiding_results(self):
        self._retirement_fixture()
        trigger_log = self.root / "old-oom" / results.LOG_FILENAME
        trigger_log.write_text(
            "ordinary non-OOM failure after ledger creation\n",
            encoding="utf-8",
        )

        report = results.build_report(self.root)

        self.assertFalse(report["ok"])
        self.assertFalse(report["retirement_ledger"]["valid"])
        self.assertEqual(report["retired_attempts"], [])
        self.assertIn(
            "does not match the current file",
            report["retirement_ledger"]["errors"][0],
        )

    def test_retirement_semantic_conflicts_fail_closed(self):
        cases = {
            "field-mismatch": (
                lambda entry: entry["retired_manifests"][0].__setitem__(
                    "model_sha256", SHA_A
                ),
                "model_sha256",
            ),
            "duplicate": (
                lambda entry: entry["retired_manifests"].append(
                    copy.deepcopy(entry["retired_manifests"][-1])
                ),
                "duplicate",
            ),
            "non-oom-trigger": (
                lambda entry: entry.__setitem__(
                    "trigger",
                    next(
                        item
                        for item in entry["retired_manifests"]
                        if item["status"] == "succeeded"
                    ),
                ),
                "trigger",
            ),
            "non-next-ladder": (
                lambda entry: entry["replacement_profile"].update(
                    {
                        "level": 2,
                        "effective_overrides": {
                            "hessian_accum_bsz": 8,
                            "backward_bsz": 4,
                            "final_layer_backward_bsz": 4,
                        },
                    }
                ),
                "strict next",
            ),
        }
        original_root = self.root
        try:
            for name, (mutate, expected) in cases.items():
                with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                    self.root = Path(directory)
                    ledger_path = self._retirement_fixture()
                    self._rewrite_retirement(ledger_path, mutate)

                    report = results.build_report(self.root)

                    self.assertFalse(report["ok"])
                    self.assertFalse(report["retirement_ledger"]["valid"])
                    self.assertEqual(report["retired_attempts"], [])
                    self.assertIn(
                        expected,
                        report["retirement_ledger"]["errors"][0],
                    )
        finally:
            self.root = original_root

    def test_noninitial_profile_requires_retirement_ledger(self):
        self.add_realq(
            "profile-one",
            lr=5e-5,
            profile_level=1,
        )

        report = results.build_report(self.root)

        self.assertFalse(report["ok"])
        self.assertFalse(report["retirement_ledger"]["valid"])
        self.assertIn(
            "missing retirement ledger",
            report["retirement_ledger"]["errors"][0],
        )

    def test_coarse_grid_cannot_select_before_local_refinement(self):
        for lr, kl in zip(COARSE_LRS, (0.3, 0.1, 0.2)):
            self.add_realq(f"coarse-{lr}", lr=lr, kl=kl)

        report = results.build_report(self.root)
        selection = report["realq_tune_selection"][0]
        self.assertFalse(report["ok"])
        self.assertEqual(selection["status"], "needs_action")
        self.assertIsNone(selection["selected"])
        self.assertEqual(
            selection["recommended_lrs"], list(REFINEMENT_LRS)
        )
        self.assertEqual(selection["attempt_count"], 3)
        self.assertEqual(selection["budget_remaining"], 17)
        self.assertIn(
            "missing_local_refinement",
            [issue["code"] for issue in selection["issues"]],
        )

    def test_exact_metrics_choose_between_identical_rounded_kl_values(self):
        self.add_refined_sweep(exact_close=True)

        report = results.build_report(self.root)
        self.assertTrue(report["ok"])
        selection = report["realq_tune_selection"][0]
        self.assertEqual(selection["status"], "selected")
        self.assertEqual(selection["selected"]["grad_lr"], 7e-5)
        self.assertEqual(
            selection["selected"]["kl_wikitext2"], 0.10041
        )
        rounded = {
            f"{record['kl_wikitext2']:.2e}"
            for record in report["records"]
            if record["grad_lr"] in (5e-5, 2e-5, 3e-5, 7e-5)
        }
        self.assertEqual(rounded, {"1.00e-01"})

        rows = list(csv.DictReader(io.StringIO(results.report_csv(report))))
        selected = [row for row in rows if row["tune_selected"] == "True"]
        self.assertEqual(len(selected), 1)
        self.assertEqual(float(selected[0]["grad_lr"]), 7e-5)
        self.assertEqual(
            selected[0]["numerical_source_sha256"],
            TRUSTED_SOURCE_IDENTITY["numerical_source_sha256"],
        )

    def test_boundary_expansion_still_requires_local_refinement(self):
        for index, (lr, kl) in enumerate(
            (
                (1e-5, 0.4),
                (5e-5, 0.3),
                (1e-4, 0.1),
                (2e-4, 0.2),
                (3e-4, 0.25),
            )
        ):
            self.add_realq(f"arm-{index}", lr=lr, kl=kl)

        report = results.build_report(self.root)
        selection = report["realq_tune_selection"][0]
        self.assertEqual(selection["status"], "needs_action")
        self.assertNotIn(
            "boundary_minimum",
            [issue["code"] for issue in selection["issues"]],
        )
        self.assertIn(
            "missing_local_refinement",
            [issue["code"] for issue in selection["issues"]],
        )
        self.assertEqual(selection["recommended_lrs"], [7e-5])

        self.add_realq("local-refinement", lr=7e-5, kl=0.15)
        completed = results.build_report(self.root)
        self.assertTrue(completed["ok"])
        self.assertEqual(
            completed["realq_tune_selection"][0]["selected"]["grad_lr"],
            1e-4,
        )

    def test_formal_realq_matches_embedded_ledger_and_tune_winner(self):
        self.add_refined_sweep()
        formal_plan = experiment_plan()
        formal_plan["selected_grad_lr_by_model_setting"]["qwen3-4b"][
            "3W16A"
        ] = 5e-5
        self.add_realq(
            "formal",
            phase="final",
            lr=5e-5,
            kl=0.11,
            ppl=10.1,
            qa=True,
            plan_value=formal_plan,
            plan_sha=SHA_E,
        )

        report = results.build_report(self.root)
        self.assertTrue(report["ok"])
        validation = report["realq_final_lr_validation"][0]
        self.assertEqual(validation["status"], "valid")
        self.assertEqual(validation["embedded_selected_grad_lr"], 5e-5)
        self.assertEqual(validation["selected_tune_grad_lr"], 5e-5)

    def test_exact_freeze_keeps_the_original_tune_winner_gate(self):
        base_plan = experiment_plan()
        base_plan["output_root"] = str(self.root.resolve())
        self.add_refined_sweep(plan_value=base_plan)
        formal_plan = self._manual_override_plan(
            base_plan,
            [self.root / "tune-1"],
            selected_lr=5e-5,
            decision_kind="exact",
        )
        self.add_realq(
            "formal",
            phase="final",
            lr=5e-5,
            qa=True,
            plan_value=formal_plan,
            plan_sha=self._formal_plan_sha,
        )

        report = results.build_report(self.root)
        self.assertTrue(report["ok"], report)
        validation = report["realq_final_lr_validation"][0]
        self.assertEqual(validation["status"], "valid")
        self.assertEqual(validation["decision_kind"], "exact")
        self.assertEqual(validation["freeze_selected_grad_lr"], 5e-5)
        self.assertEqual(validation["selected_tune_grad_lr"], 5e-5)
        rows = list(
            csv.DictReader(io.StringIO(results.report_csv(report)))
        )
        formal_row = next(row for row in rows if row["phase"] == "final")
        self.assertEqual(formal_row["final_lr_decision_kind"], "exact")
        self.assertEqual(
            float(formal_row["freeze_selected_grad_lr"]), 5e-5
        )
        self.assertEqual(
            formal_row["freeze_ledger_sha256"],
            validation["freeze_ledger_sha256"],
        )
        self.assertEqual(
            formal_row["freeze_artifact_sha256"],
            validation["freeze_artifact_sha256"],
        )

    def test_exact_freeze_cannot_override_the_parsed_tune_winner(self):
        base_plan = experiment_plan()
        base_plan["output_root"] = str(self.root.resolve())
        self.add_refined_sweep(plan_value=base_plan)
        formal_plan = self._manual_override_plan(
            base_plan,
            [self.root / "tune-2"],
            selected_lr=1e-4,
            decision_kind="exact",
        )
        self.add_realq(
            "formal",
            phase="final",
            lr=1e-4,
            qa=True,
            plan_value=formal_plan,
            plan_sha=self._formal_plan_sha,
        )

        report = results.build_report(self.root)
        self.assertFalse(report["ok"])
        validation = report["realq_final_lr_validation"][0]
        self.assertIn("final_lr_mismatch", validation["issues"])
        self.assertIn(
            "manual_lr_selection_freeze_exact_mismatch",
            validation["issues"],
        )

    def test_exact_freeze_resolves_an_incomplete_manual_grid(self):
        _, _, formal_plan = self._manual_exact_incomplete_fixture()
        self.add_realq(
            "manual-exact-formal",
            phase="final",
            lr=3e-5,
            qa=True,
            plan_value=formal_plan,
            plan_sha=self._formal_plan_sha,
        )

        report = results.build_report(self.root)
        self.assertTrue(report["ok"], report)
        selection = report["realq_tune_selection"][0]
        self.assertNotEqual(selection["status"], "selected")
        self.assertEqual(selection["effective_status"], "selected")
        self.assertTrue(selection["resolved_by_manual_freeze"])
        validation = report["realq_final_lr_validation"][0]
        self.assertTrue(validation["freeze_resolves_tune_selection"])
        self.assertEqual(validation["selected_tune_grad_lr"], 3e-5)

    def test_exact_freeze_rejects_a_nonminimum_manual_choice(self):
        _, _, formal_plan = self._manual_exact_incomplete_fixture(
            selected_lr=2e-5
        )
        self.add_realq(
            "manual-exact-formal",
            phase="final",
            lr=2e-5,
            qa=True,
            plan_value=formal_plan,
            plan_sha=self._formal_plan_sha,
        )

        report = results.build_report(self.root)
        self.assertFalse(report["ok"])
        self.assertIn(
            "manual_lr_selection_freeze_invalid",
            report["realq_final_lr_validation"][0]["issues"],
        )

    def test_exact_freeze_scans_every_later_accepted_active_point(self):
        original_root = self.root
        for case in ("lower_kl", "tied_kl", "source", "protocol"):
            with self.subTest(case=case):
                self.root = original_root / case
                base_plan, _, formal_plan = (
                    self._manual_exact_incomplete_fixture()
                )
                new_kl = 0.1 if case == "tied_kl" else (
                    0.05 if case == "lower_kl" else 0.3
                )
                extra = self.add_realq(
                    "manual-exact-extra",
                    lr=4e-5,
                    kl=new_kl,
                    ppl=0.5,
                    plan_value=base_plan,
                )
                if case in {"source", "protocol"}:
                    manifest_path = extra / results.MANIFEST_FILENAME
                    payload = json.loads(manifest_path.read_text())
                    if case == "source":
                        payload["numerical_source_tree"][
                            "combined_sha256"
                        ] = SHA_D
                        payload["numerical_source_tree"][
                            "combined_sha256_at_end"
                        ] = SHA_D
                    else:
                        payload["plan"]["content"]["legacy_cache_root"] = (
                            "cache/lowbit_activation/protocol_tamper"
                        )
                    manifest_path.write_text(json.dumps(payload))
                self.add_realq(
                    "manual-exact-formal",
                    phase="final",
                    lr=3e-5,
                    qa=True,
                    plan_value=formal_plan,
                    plan_sha=self._formal_plan_sha,
                )
                report = results.build_report(self.root)
                self.assertFalse(report["ok"])
                self.assertIn(
                    "manual_lr_selection_freeze_invalid",
                    report["realq_final_lr_validation"][0]["issues"],
                )
        self.root = original_root

    def test_qwen32_exact_freeze_requires_strict_two_sided_bracketing(self):
        runs = [
            self.add_realq(
                f"qwen32-bracket-{index}",
                lr=lr,
                kl=kl,
                ppl=10.0 + index,
            )
            for index, (lr, kl) in enumerate(
                ((1e-5, 0.2), (2e-5, 0.1), (3e-5, 0.3))
            )
        ]
        lower, winner, upper = [
            replace(
                results.parse_result(run / results.MANIFEST_FILENAME),
                model="qwen3-32b",
            )
            for run in runs
        ]
        evidence = self._freeze_evidence(runs[1])
        evidence.update(
            {
                "execution_id": winner.execution_id,
                "run_id": winner.run_id,
                "attempt_index": winner.attempt_index,
                "grad_lr": winner.grad_lr,
                "kl_wikitext2": winner.kl_wikitext2,
                **winner.identities,
            }
        )
        freeze = {
            "ledger": {
                "expected_plan_sha256": SHA_A,
            },
            "selection": {
                "selected_lr": winner.grad_lr,
                "decision_kind": "exact",
                "selected_manifest": str(winner.manifest_path),
                "evidence": [evidence],
            },
            "source_identity_before": {
                "runner_sha256": TRUSTED_SOURCE_IDENTITY["runner_sha256"],
                "executor_sha256": TRUSTED_SOURCE_IDENTITY[
                    "executor_sha256"
                ],
                "numerical_source_sha256": TRUSTED_SOURCE_IDENTITY[
                    "numerical_source_sha256"
                ],
            },
        }
        formal = replace(winner, phase="final")
        protocol_sha = results._canonical_sha256(winner.plan_content)

        results._validate_manual_lr_freeze_evidence(
            formal,
            freeze,
            {
                str(item.manifest_path): item
                for item in (lower, winner, upper)
            },
            expected_tune_protocol_sha256=protocol_sha,
            reconciled_attempt_count=8,
        )
        cases = {
            "low_missing": (winner, upper),
            "high_missing": (lower, winner),
            "side_kl_not_worse": (
                replace(lower, kl_wikitext2=winner.kl_wikitext2),
                winner,
                upper,
            ),
        }
        for case, points in cases.items():
            with self.subTest(case=case), self.assertRaises(
                results.ResultError
            ):
                results._validate_manual_lr_freeze_evidence(
                    formal,
                    freeze,
                    {
                        str(item.manifest_path): item
                        for item in points
                    },
                    expected_tune_protocol_sha256=protocol_sha,
                    reconciled_attempt_count=8,
                )

    def test_qwen32_exact_freeze_uses_reconciled_count_and_nearest_gap(
        self,
    ):
        runs = [
            self.add_realq(
                f"qwen32-gap-{index}",
                lr=lr,
                kl=0.1 + index,
                ppl=10.0 + index,
            )
            for index, lr in enumerate(
                (1e-5, 2e-5, 3e-5, 4e-5, 5e-5)
            )
        ]
        parsed = [
            replace(
                results.parse_result(run / results.MANIFEST_FILENAME),
                model="qwen3-32b",
            )
            for run in runs
        ]

        def validate(kls, *, count=7, duplicate_lr=False):
            points = [
                replace(item, kl_wikitext2=kl)
                for item, kl in zip(parsed, kls)
            ]
            if duplicate_lr:
                points[0] = replace(
                    points[0], grad_lr=points[1].grad_lr
                )
            winner = points[2]
            evidence = self._freeze_evidence(runs[2])
            evidence.update(
                {
                    "execution_id": winner.execution_id,
                    "run_id": winner.run_id,
                    "attempt_index": winner.attempt_index,
                    "grad_lr": winner.grad_lr,
                    "kl_wikitext2": winner.kl_wikitext2,
                    **winner.identities,
                }
            )
            freeze = {
                "ledger": {"expected_plan_sha256": SHA_A},
                "selection": {
                    "selected_lr": winner.grad_lr,
                    "decision_kind": "exact",
                    "selected_manifest": str(winner.manifest_path),
                    "evidence": [evidence],
                },
                "source_identity_before": {
                    key: TRUSTED_SOURCE_IDENTITY[key]
                    for key in (
                        "runner_sha256",
                        "executor_sha256",
                        "numerical_source_sha256",
                    )
                },
            }
            results._validate_manual_lr_freeze_evidence(
                replace(winner, phase="final"),
                freeze,
                {
                    str(item.manifest_path): item for item in points
                },
                expected_tune_protocol_sha256=(
                    results._canonical_sha256(winner.plan_content)
                ),
                reconciled_attempt_count=count,
            )

        # Decimal-exact 2% on both nearest sides permits an early 7-launch
        # freeze even though the farther points are much worse.
        validate((0.5, 0.102, 0.1, 0.102, 0.6), count=7)
        # At eight launches, strict two-sided closure remains sufficient.
        validate((0.5, 0.4, 0.1, 0.3, 0.6), count=8)

        rejected = {
            "over_2_percent": (
                (0.101, 0.1020000001, 0.1, 0.102, 0.101),
                {},
            ),
            "nearest_not_far": (
                (0.101, 0.2, 0.1, 0.102, 0.101),
                {},
            ),
            "zero_best": (
                (0.1, 0.001, 0.0, 0.001, 0.1),
                {},
            ),
            "duplicate_lr": (
                (0.5, 0.102, 0.1, 0.102, 0.6),
                {"duplicate_lr": True},
            ),
        }
        for case, (kls, options) in rejected.items():
            with self.subTest(case=case), self.assertRaises(
                results.ResultError
            ):
                validate(kls, count=7, **options)

    def test_formal_realq_rejects_embedded_ledger_mismatch(self):
        self.add_refined_sweep()
        formal_plan = experiment_plan()
        formal_plan["selected_grad_lr_by_model_setting"]["qwen3-4b"][
            "3W16A"
        ] = 1e-4
        self.add_realq(
            "formal",
            phase="final",
            lr=5e-5,
            qa=True,
            plan_value=formal_plan,
            plan_sha=SHA_E,
        )

        report = results.build_report(self.root)
        self.assertFalse(report["ok"])
        self.assertEqual(report["rejected_count"], 1)
        self.assertIn("--grad_lr", report["rejected"][0]["errors"][0])

    def test_formal_ledger_value_must_also_equal_parsed_tune_winner(self):
        self.add_refined_sweep()
        formal_plan = experiment_plan()
        formal_plan["selected_grad_lr_by_model_setting"]["qwen3-4b"][
            "3W16A"
        ] = 1e-4
        self.add_realq(
            "formal",
            phase="final",
            lr=1e-4,
            qa=True,
            plan_value=formal_plan,
            plan_sha=SHA_E,
        )

        report = results.build_report(self.root)
        self.assertFalse(report["ok"])
        validation = report["realq_final_lr_validation"][0]
        self.assertNotIn("final_lr_ledger_mismatch", validation["issues"])
        self.assertIn("final_lr_mismatch", validation["issues"])
        self.assertIn(
            "final_lr_drift",
            [item["code"] for item in report["anomalies"]],
        )

    def test_interpolated_freeze_accepts_two_sided_2w16a_evidence(self):
        _, lower, upper, formal_plan = self._manual_override_fixture()
        self.add_realq(
            "override-formal",
            phase="final",
            lr=1.2e-4,
            qa=True,
            plan_value=formal_plan,
            plan_sha=self._formal_plan_sha,
            setting="2W16A",
        )

        report = results.build_report(self.root)
        self.assertTrue(report["ok"], report)
        validation = report["realq_final_lr_validation"][0]
        self.assertEqual(validation["status"], "valid")
        self.assertEqual(
            validation["decision_kind"], "user_override_interpolated"
        )
        self.assertEqual(validation["freeze_selected_grad_lr"], 1.2e-4)
        self.assertEqual(validation["selected_tune_grad_lr"], 5e-5)
        self.assertEqual(
            validation["freeze_evidence_manifests"],
            [
                str((lower / results.MANIFEST_FILENAME).resolve()),
                str((upper / results.MANIFEST_FILENAME).resolve()),
            ],
        )
        self.assertEqual(validation["freeze_errors"], [])
        rows = list(
            csv.DictReader(io.StringIO(results.report_csv(report)))
        )
        formal_row = next(row for row in rows if row["phase"] == "final")
        self.assertEqual(
            formal_row["final_lr_decision_kind"],
            "user_override_interpolated",
        )
        self.assertEqual(
            float(formal_row["freeze_selected_grad_lr"]), 1.2e-4
        )
        self.assertEqual(
            json.loads(formal_row["freeze_evidence_manifests_json"]),
            validation["freeze_evidence_manifests"],
        )

    def test_interpolated_freeze_accepts_explicit_results_source_adoption(self):
        _, _, _, formal_plan = self._manual_override_fixture()
        self._write_manual_lr_attestation(
            formal_plan, old_results_sha=SHA_A
        )
        self.add_realq(
            "override-formal",
            phase="final",
            lr=1.2e-4,
            qa=True,
            plan_value=formal_plan,
            plan_sha=self._formal_plan_sha,
            setting="2W16A",
        )

        report = results.build_report(self.root)
        self.assertTrue(report["ok"], report)

    def test_interpolated_freeze_resolves_an_incomplete_manual_grid(self):
        _, _, _, formal_plan = self._manual_override_fixture(
            complete_grid=False
        )
        self.add_realq(
            "override-formal",
            phase="final",
            lr=1.2e-4,
            qa=True,
            plan_value=formal_plan,
            plan_sha=self._formal_plan_sha,
            setting="2W16A",
        )

        report = results.build_report(self.root)
        self.assertTrue(report["ok"], report)
        selection = report["realq_tune_selection"][0]
        self.assertNotEqual(selection["status"], "selected")
        self.assertEqual(selection["effective_status"], "selected")
        self.assertTrue(selection["resolved_by_manual_freeze"])
        validation = report["realq_final_lr_validation"][0]
        self.assertTrue(validation["freeze_resolves_tune_selection"])
        rows = list(
            csv.DictReader(io.StringIO(results.report_csv(report)))
        )
        formal_row = next(row for row in rows if row["phase"] == "final")
        self.assertEqual(
            formal_row["final_lr_decision_kind"],
            "user_override_interpolated",
        )
        closeout = results.build_report(
            self.root, require_complete_matrix=True
        )
        self.assertEqual(
            closeout["matrix_completeness"]["missing_tune_selections"],
            [],
        )

    def test_recomputed_source_and_timing_attestations_still_fail_closed(self):
        original_root = self.root
        for case in ("campaign_source", "results_source", "timing_source"):
            with self.subTest(case=case):
                self.root = original_root / case
                _, _, _, formal_plan = self._manual_override_fixture()
                ledger = formal_plan["manual_lr_selection_freeze"][
                    "ledger"
                ]
                if case == "campaign_source":
                    ledger["source_identity_before"][
                        "campaign_sha256"
                    ] = SHA_A
                    self._write_manual_lr_attestation(formal_plan)
                elif case == "results_source":
                    ledger["source_identity_before"][
                        "results_sha256"
                    ] = SHA_A
                    self._write_manual_lr_attestation(formal_plan)
                else:
                    timing_gate = results._current_formal_timing_gate()
                    timing_gate["source_set_sha256"] = SHA_A
                    unsigned_gate = dict(timing_gate)
                    unsigned_gate.pop("gate_sha256")
                    timing_gate["gate_sha256"] = (
                        results._strict_canonical_sha256(
                            unsigned_gate, "tampered timing gate"
                        )
                    )
                    self._write_manual_lr_attestation(
                        formal_plan, timing_gate=timing_gate
                    )
                self.add_realq(
                    "override-formal",
                    phase="final",
                    lr=1.2e-4,
                    qa=True,
                    plan_value=formal_plan,
                    plan_sha=self._formal_plan_sha,
                    setting="2W16A",
                )
                report = results.build_report(self.root)
                self.assertFalse(report["ok"])
                self.assertIn(
                    "manual_lr_selection_freeze_invalid",
                    report["realq_final_lr_validation"][0]["issues"],
                )
        self.root = original_root

    def test_interpolated_freeze_sha_tampering_fails_closed(self):
        original_root = self.root
        for case in (
            "ledger_sha",
            "selection_sha",
            "expected_plan_sha",
            "expected_state_sha",
            "timing_gate_sha",
            "timing_source_set_sha",
            "evidence_manifest_sha",
            "campaign_source_sha",
            "results_source_sha",
        ):
            with self.subTest(case=case):
                self.root = original_root / case
                _, _, _, formal_plan = self._manual_override_fixture()
                wrapper = formal_plan["manual_lr_selection_freeze"]
                ledger = wrapper["ledger"]
                selection = ledger["selections"][0]
                if case == "ledger_sha":
                    wrapper["ledger_sha256"] = SHA_A
                elif case == "selection_sha":
                    ledger["selection_sha256"] = SHA_E
                    wrapper["ledger_sha256"] = (
                        results._strict_canonical_sha256(
                            ledger, "test ledger"
                        )
                    )
                elif case == "expected_plan_sha":
                    ledger["expected_plan_sha256"] = SHA_B
                    self._rehash_manual_lr_freeze(formal_plan)
                elif case == "expected_state_sha":
                    ledger["expected_state_sha256"] = SHA_C
                elif case == "timing_gate_sha":
                    ledger["timing_gate_sha256"] = SHA_A
                    self._rehash_manual_lr_freeze(formal_plan)
                elif case == "timing_source_set_sha":
                    ledger["timing_source_set_sha256"] = SHA_A
                    self._rehash_manual_lr_freeze(formal_plan)
                elif case == "evidence_manifest_sha":
                    selection["evidence"][0][
                        "manifest_sha256"
                    ] = SHA_B
                    self._rehash_manual_lr_freeze(formal_plan)
                elif case == "campaign_source_sha":
                    ledger["source_identity_before"][
                        "campaign_sha256"
                    ] = SHA_A
                    self._rehash_manual_lr_freeze(formal_plan)
                else:
                    ledger["source_identity_before"][
                        "results_sha256"
                    ] = SHA_A
                    self._rehash_manual_lr_freeze(formal_plan)
                self.add_realq(
                    "override-formal",
                    phase="final",
                    lr=1.2e-4,
                    qa=True,
                    plan_value=formal_plan,
                    plan_sha=self._formal_plan_sha,
                    setting="2W16A",
                )
                report = results.build_report(self.root)
                self.assertFalse(report["ok"])
                validation = report["realq_final_lr_validation"][0]
                self.assertIn(
                    "manual_lr_selection_freeze_invalid",
                    validation["issues"],
                )
        self.root = original_root

    def test_interpolated_freeze_evidence_must_be_accepted_and_provenant(self):
        original_root = self.root
        for case in ("failed", "exit_nonzero", "source", "protocol"):
            with self.subTest(case=case):
                self.root = original_root / case
                _, lower, _, formal_plan = self._manual_override_fixture()
                manifest_path = lower / results.MANIFEST_FILENAME
                payload = json.loads(manifest_path.read_text())
                if case == "failed":
                    payload["status"] = "failed"
                    payload["exit_code"] = 1
                elif case == "exit_nonzero":
                    payload["exit_code"] = 7
                elif case == "source":
                    payload["numerical_source_tree"][
                        "combined_sha256"
                    ] = SHA_D
                    payload["numerical_source_tree"][
                        "combined_sha256_at_end"
                    ] = SHA_D
                else:
                    payload["plan"]["content"]["output_root"] = (
                        "output/lowbit_activation_protocol_tamper"
                    )
                manifest_path.write_text(json.dumps(payload))
                evidence = formal_plan["manual_lr_selection_freeze"][
                    "ledger"
                ]["selections"][0]["evidence"][0]
                evidence["manifest_sha256"] = _sha256_file(manifest_path)
                if case == "source":
                    evidence["numerical_source_sha256"] = SHA_D
                self._rehash_manual_lr_freeze(formal_plan)
                self.add_realq(
                    "override-formal",
                    phase="final",
                    lr=1.2e-4,
                    qa=True,
                    plan_value=formal_plan,
                    plan_sha=self._formal_plan_sha,
                    setting="2W16A",
                )
                report = results.build_report(self.root)
                self.assertFalse(report["ok"])
                validation = report["realq_final_lr_validation"][0]
                self.assertIn(
                    "manual_lr_selection_freeze_invalid",
                    validation["issues"],
                )
        self.root = original_root

    def test_interpolated_freeze_requires_matching_row_and_strict_bracket(self):
        original_root = self.root
        for case in ("model_setting", "upper_side", "reversed_evidence"):
            with self.subTest(case=case):
                self.root = original_root / case
                _, _, _, formal_plan = self._manual_override_fixture()
                selection = formal_plan["manual_lr_selection_freeze"][
                    "ledger"
                ]["selections"][0]
                formal_lr = 1.2e-4
                if case == "model_setting":
                    selection["setting"] = "3W16A"
                elif case == "upper_side":
                    formal_lr = 1.3e-4
                    selection["selected_lr"] = formal_lr
                    formal_plan["selected_grad_lr_by_model_setting"][
                        "qwen3-4b"
                    ]["2W16A"] = formal_lr
                else:
                    selection["evidence"].reverse()
                self._rehash_manual_lr_freeze(formal_plan)
                self.add_realq(
                    "override-formal",
                    phase="final",
                    lr=formal_lr,
                    qa=True,
                    plan_value=formal_plan,
                    plan_sha=self._formal_plan_sha,
                    setting="2W16A",
                )
                report = results.build_report(self.root)
                self.assertFalse(report["ok"])
                validation = report["realq_final_lr_validation"][0]
                self.assertIn(
                    "manual_lr_selection_freeze_invalid",
                    validation["issues"],
                )
        self.root = original_root

    def test_interpolated_freeze_rejects_unreviewed_two_sided_substitutes(self):
        base_plan, _, _, _ = self._manual_override_fixture()
        lower = self.add_realq(
            "override-unreviewed-lower",
            lr=1.1e-4,
            kl=0.205,
            ppl=10.205,
            plan_value=base_plan,
            setting="2W16A",
        )
        upper = self.add_realq(
            "override-unreviewed-upper",
            lr=1.3e-4,
            kl=0.225,
            ppl=10.225,
            plan_value=base_plan,
            setting="2W16A",
        )
        formal_plan = self._manual_override_plan(
            base_plan, [lower, upper]
        )
        self.add_realq(
            "override-formal",
            phase="final",
            lr=1.2e-4,
            qa=True,
            plan_value=formal_plan,
            plan_sha=self._formal_plan_sha,
            setting="2W16A",
        )

        report = results.build_report(self.root)
        self.assertFalse(report["ok"])
        validation = report["realq_final_lr_validation"][0]
        self.assertIn(
            "manual_lr_selection_freeze_invalid",
            validation["issues"],
        )

    def test_freeze_rejects_current_state_protocol_hash_tampering(self):
        original_root = self.root
        for field in (
            "tune_protocol_sha256",
            "formal_protocol_sha256",
        ):
            with self.subTest(field=field):
                self.root = original_root / field
                _, _, _, formal_plan = self._manual_override_fixture()
                state_path = self.root / "_campaign" / "state.json"
                state = json.loads(state_path.read_text())
                state["plan"][field] = SHA_A
                state_path.write_text(
                    json.dumps(state, indent=2, sort_keys=True) + "\n"
                )
                self.add_realq(
                    "override-formal",
                    phase="final",
                    lr=1.2e-4,
                    qa=True,
                    plan_value=formal_plan,
                    plan_sha=self._formal_plan_sha,
                    setting="2W16A",
                )
                report = results.build_report(self.root)
                self.assertFalse(report["ok"])
                self.assertIn(
                    "manual_lr_selection_freeze_invalid",
                    report["realq_final_lr_validation"][0]["issues"],
                )
        self.root = original_root

    def test_freeze_rejects_current_state_source_identity_tampering(self):
        _, _, _, formal_plan = self._manual_override_fixture()
        state_path = self.root / "_campaign" / "state.json"
        state = json.loads(state_path.read_text())
        state["source_identity"]["formal_timing_gate_sha256"] = SHA_A
        state_path.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n"
        )
        self.add_realq(
            "override-formal",
            phase="final",
            lr=1.2e-4,
            qa=True,
            plan_value=formal_plan,
            plan_sha=self._formal_plan_sha,
            setting="2W16A",
        )

        report = results.build_report(self.root)
        self.assertFalse(report["ok"])
        self.assertIn(
            "manual_lr_selection_freeze_invalid",
            report["realq_final_lr_validation"][0]["issues"],
        )

    def test_bf16_exports_ppl_with_na_kl_and_metric_provenance(self):
        self.add_legacy("bf16", method="bf16", kl=0.0, ppl=8.75)

        report = results.build_report(self.root)
        self.assertTrue(report["ok"])
        record = report["records"][0]
        self.assertEqual(record["method"], "bf16")
        self.assertIsNone(record["kl_wikitext2"])
        self.assertEqual(record["ppl_wikitext2"], 8.75)
        self.assertEqual(tuple(record["tasks"]), results.EXPECTED_TASKS)
        self.assertEqual(
            record["task_metric_keys"], results.EXPECTED_TASK_METRICS
        )
        self.assertEqual(
            record["lm_eval_version"], results.EXPECTED_LM_EVAL_VERSION
        )

    def test_informational_failures_and_precomputes_are_not_rejected(self):
        self.add_refined_sweep()
        failed = self.add_realq(
            "failed-oom",
            lr=9e-4,
            status="failed",
        )
        failed_manifest = failed / results.MANIFEST_FILENAME
        failed_payload = json.loads(failed_manifest.read_text())
        failed_payload["command"]["env"][
            "LOWBIT_ACTIVATION_ATTEMPT_INDEX"
        ] = "broken"
        failed_manifest.write_text(json.dumps(failed_payload))
        precompute = self.root / "precompute"
        precompute.mkdir()
        (precompute / results.MANIFEST_FILENAME).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "execution_id": "precompute",
                    "run_id": "precompute_realq_static_qwen3-4b_tune",
                    "status": "succeeded",
                    "command": {
                        "argv": [
                            "python3",
                            "-m",
                            "realq.ptq",
                            "--exit_after_precompute",
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )

        report = results.build_report(self.root)
        self.assertTrue(report["ok"])
        self.assertEqual(report["rejected_count"], 0)
        self.assertEqual(report["informational_count"], 2)
        selection = report["realq_tune_selection"][0]
        self.assertEqual(selection["attempt_count"], 7)
        self.assertEqual(selection["budget_remaining"], 13)
        self.assertIn(
            "attempt_metadata_drift",
            [item["code"] for item in report["anomalies"]],
        )

    def test_attempt_budget_exhaustion_requires_user_action(self):
        self.add_realq("only-success", lr=5e-5, kl=0.1)
        for index in range(19):
            self.add_realq(
                f"failed-{index}",
                lr=1e-3 + index * 1e-5,
                status="failed",
            )

        report = results.build_report(self.root)
        selection = report["realq_tune_selection"][0]
        self.assertEqual(selection["attempt_count"], 20)
        self.assertEqual(selection["budget_remaining"], 0)
        self.assertEqual(selection["status"], "needs_user_action")
        self.assertEqual(selection["recommended_lrs"], [])
        self.assertIn(
            "attempt_budget_exhausted",
            [issue["code"] for issue in selection["issues"]],
        )

        self.add_realq("failed-over-cap", lr=2e-3, status="failed")
        exceeded = results.build_report(self.root)["realq_tune_selection"][0]
        self.assertEqual(exceeded["attempt_count"], 21)
        self.assertIn(
            "attempt_budget_exceeded",
            [issue["code"] for issue in exceeded["issues"]],
        )

    def test_invalid_exact_metric_is_rejected_and_reported_as_anomaly(self):
        self.add_realq(
            "bad-exact",
            lr=5e-5,
            kl=0.1,
            exact_kl_text="nan",
        )

        report = results.build_report(self.root)
        self.assertEqual(report["accepted_count"], 0)
        self.assertEqual(report["rejected_count"], 1)
        self.assertIn(
            "invalid_exact_kl",
            [item["code"] for item in report["anomalies"]],
        )

    def test_curve_and_ppl_rank_anomalies_are_reported(self):
        kls = {
            1e-5: 0.3,
            2e-5: 0.1,
            3e-5: 0.2,
            5e-5: 0.05,
            7e-5: 0.15,
            1e-4: 0.25,
        }
        for index, lr in enumerate((*COARSE_LRS, *REFINEMENT_LRS)):
            self.add_realq(
                f"anomaly-{index}",
                lr=lr,
                kl=kls[lr],
                ppl=30.0 - kls[lr],
            )

        report = results.build_report(self.root)
        self.assertTrue(report["ok"])
        codes = {item["code"] for item in report["anomalies"]}
        self.assertIn("non_unimodal_kl_curve", codes)
        self.assertIn("kl_curve_spike", codes)
        self.assertIn("ppl_kl_rank_inversion", codes)

    def test_protocol_package_and_source_tree_drift_fail_closed(self):
        cases = (
            (
                "world-size",
                lambda payload: payload["command"]["argv"].__setitem__(
                    payload["command"]["argv"].index("--nnodes=1"),
                    "--nnodes=2",
                ),
                "nnodes=1",
            ),
            (
                "cache-hit",
                lambda payload: payload["command"]["argv"].__setitem__(
                    payload["command"]["argv"].index(
                        "--require_static_cache_hit"
                    )
                    + 1,
                    "false",
                ),
                "require_static_cache_hit",
            ),
            (
                "lm-version",
                lambda payload: payload["python_packages"]["lm_eval"].update(
                    {"version": "0.4.3"}
                ),
                "lm-eval version",
            ),
            (
                "source-tree",
                lambda payload: payload["numerical_source_tree"].update(
                    {"changed_during_execution": True}
                ),
                "changed_during_execution",
            ),
        )
        for name, mutate, expected_error in cases:
            with self.subTest(name=name):
                run = self.add_realq(
                    name,
                    phase="final" if name == "world-size" else "tune",
                    lr=5e-5,
                    qa=name == "world-size",
                )
                manifest_path = run / results.MANIFEST_FILENAME
                payload = json.loads(manifest_path.read_text())
                mutate(payload)
                manifest_path.write_text(json.dumps(payload))
                with self.assertRaisesRegex(
                    results.ResultError, expected_error
                ):
                    results.parse_result(manifest_path)

    def test_duplicate_success_and_non_lr_config_drift_fail_closed(self):
        self.add_refined_sweep()
        self.add_realq(
            "duplicate",
            lr=5e-5,
            kl=0.09,
            attempt_index=2,
        )
        drift_run = self.root / "tune-0" / results.MANIFEST_FILENAME
        payload = json.loads(drift_run.read_text())
        payload["numerical_source_tree"]["combined_sha256"] = SHA_D
        payload["numerical_source_tree"]["combined_sha256_at_end"] = SHA_D
        drift_run.write_text(json.dumps(payload))

        report = results.build_report(self.root)
        selection = report["realq_tune_selection"][0]
        codes = [issue["code"] for issue in selection["issues"]]
        self.assertFalse(report["ok"])
        self.assertIn("duplicate_candidate", codes)
        self.assertIn("config_inconsistency", codes)
        self.assertIn(
            "config_drift",
            [item["code"] for item in report["anomalies"]],
        )

    def test_complete_matrix_gate_is_separate_from_incremental_summary(self):
        self.add_refined_sweep()

        incremental = results.build_report(self.root)
        closeout = results.build_report(
            self.root, require_complete_matrix=True
        )
        self.assertTrue(incremental["ok"])
        self.assertFalse(closeout["ok"])
        self.assertFalse(closeout["matrix_completeness"]["complete"])
        self.assertEqual(incremental["scope"], "incremental")
        self.assertEqual(closeout["scope"], "complete_matrix")

    def test_complete_matrix_accepts_mutated_selected_lr_ledger_only(self):
        self.add_refined_sweep()
        formal_plan = experiment_plan()
        formal_plan["selected_grad_lr_by_model_setting"]["qwen3-4b"][
            "3W16A"
        ] = 5e-5
        self.add_realq(
            "formal-realq",
            phase="final",
            lr=5e-5,
            qa=True,
            plan_value=formal_plan,
            plan_sha=SHA_E,
        )
        self.add_legacy(
            "formal-gptaq",
            method="gptaq",
            plan_value=formal_plan,
        )
        self.add_legacy(
            "formal-guided",
            method="guided_gptq",
            plan_value=formal_plan,
        )
        self.add_legacy(
            "formal-bf16",
            method="bf16",
            plan_value=formal_plan,
        )

        report = results.build_report(
            self.root, require_complete_matrix=True
        )
        self.assertTrue(report["ok"], report)
        self.assertTrue(report["matrix_completeness"]["complete"])

    def test_cli_writes_outputs_and_returns_nonzero_for_unresolved_search(self):
        for lr, kl in zip(COARSE_LRS, (0.3, 0.1, 0.2)):
            self.add_realq(f"coarse-{lr}", lr=lr, kl=kl)
        json_path = self.root / "summary.json"
        csv_path = self.root / "summary.csv"

        exit_code = results.main(
            [
                str(self.root),
                "--json-out",
                str(json_path),
                "--csv-out",
                str(csv_path),
            ]
        )
        self.assertEqual(exit_code, 2)
        self.assertFalse(json.loads(json_path.read_text())["ok"])
        with csv_path.open(encoding="utf-8", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 3)


if __name__ == "__main__":
    unittest.main()
