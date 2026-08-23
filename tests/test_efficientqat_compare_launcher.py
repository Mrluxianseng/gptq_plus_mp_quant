from __future__ import print_function

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = REPO_ROOT / "experiments" / "efficientqat_compare"
PLAN_PATH = CAMPAIGN_DIR / "plan.json"
LAUNCHER_PATH = CAMPAIGN_DIR / "launcher.py"


def load_launcher():
    spec = importlib.util.spec_from_file_location(
        "efficientqat_compare_launcher", str(LAUNCHER_PATH)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EfficientQATComparePlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launcher = load_launcher()
        cls.plan = cls.launcher.load_plan(PLAN_PATH)

    def test_exact_six_run_matrix_and_gpu_mapping(self):
        runs = self.plan["runs"]
        self.assertEqual(len(runs), 6)
        self.assertEqual(
            {(run["model"], run["setting"]) for run in runs},
            self.launcher.EXPECTED_PAIRS,
        )
        self.assertEqual(
            sorted(run["training_gpu"] for run in runs),
            [0, 1, 2, 3, 4, 5],
        )
        self.assertEqual({run["evaluation_gpu"] for run in runs}, {6, 7})
        self.assertEqual(len({run["run_id"] for run in runs}), 6)

    def test_fixed_job_data_models_versions_and_eval_contract(self):
        runtime = self.plan["runtime"]
        self.assertEqual(runtime["canoe_job_id"], "j-7x9o0je4pk")
        self.assertEqual(runtime["canoe_pod"], "j-7x9o0je4pk-master-0")
        self.assertTrue(Path(runtime["output_root"]).is_absolute())

        artifact = self.plan["calibration_contract"]["token_artifact"]
        self.assertEqual(artifact["length"], 2048)
        self.assertEqual(artifact["item_shape"], [2048])
        self.assertEqual(artifact["item_dtype"], "torch.int64")
        self.assertEqual(
            artifact["sha256"],
            "681fc508df99ab1e0bbe985d7cc918ebaa848b042615b353d6b53cc9f0f96efb",
        )
        self.assertTrue(Path(artifact["path"]).is_absolute())

        tokenizer_hashes = {
            value["tokenizer_json_sha256"]
            for value in self.plan["models"].values()
        }
        self.assertEqual(len(tokenizer_hashes), 1)
        self.assertEqual(self.plan["runtime_versions"]["lm_eval"], "0.4.4")
        evaluation = self.plan["unified_eval_contract"]
        self.assertEqual(evaluation["implementation_module"], "utils.eval_utils")
        self.assertEqual(
            tuple(evaluation["qa_tasks"]),
            self.launcher.EXPECTED_TASKS,
        )
        self.assertEqual(evaluation["kl_topk"], -1)
        self.assertFalse(evaluation["qa_raw_scores_must_be_persisted"])
        self.assertIn(
            "current shared RealQ qa_eval",
            evaluation["qa_score_protocol"],
        )
        self.assertNotIn(
            "weight materialization",
            self.plan["timing_contract"]["quantization_gpu_hours_include"],
        )
        cache_preflight = evaluation["reference_cache_preflight"]
        self.assertEqual(cache_preflight["status"], "succeeded")
        self.assertFalse(cache_preflight["cuda_available"])
        self.assertEqual(
            set(cache_preflight["models"]),
            {"llama3.2-1b", "llama3.2-3b"},
        )

    def test_method_contract_is_native_minmax_no_rotation_and_unaware_w2(self):
        method = self.plan["method_contract"]
        self.assertEqual(
            method["weight_quantizer"]["family"],
            "EfficientQAT official unsigned asymmetric min-max",
        )
        self.assertFalse(method["rotation"]["enabled"])
        w2 = method["activation_kv_quantization"]["W2A4KV4"]
        self.assertTrue(w2["enabled"])
        self.assertFalse(w2["aware"])
        self.assertFalse(w2["qk_hadamard"])
        self.assertFalse(w2["independent_q_quantization"])
        self.assertEqual(
            (w2["a_clip_ratio"], w2["k_clip_ratio"], w2["v_clip_ratio"]),
            (0.9, 0.9, 0.9),
        )

    def test_confirmations_are_resolved_and_gated_runner_is_ready(self):
        self.assertTrue(self.plan["execution_control"]["launch_enabled"])
        self.assertEqual(
            self.plan["execution_control"]["runner_implementation_status"],
            "ready",
        )
        self.assertEqual(self.launcher.unresolved_confirmations(self.plan), [])
        confirmations = self.plan["confirmations"]
        self.assertEqual(confirmations["block_ap_precision"]["resolution"], "fp16_amp")
        self.assertEqual(
            confirmations["e2e_optimizer"]["resolution"],
            "torch.optim.AdamW compatibility substitution",
        )
        self.assertIn(
            "validation_size=0",
            confirmations["validation_policy"]["resolution"],
        )
        self.assertEqual(
            self.plan["method_contract"]["e2e_qp"]["optimizer"],
            "torch.optim.AdamW",
        )
        self.assertEqual(
            self.plan["method_contract"]["block_ap"]["precision"],
            "fp16_amp",
        )
        self.assertEqual(self.launcher.COMMANDS, ("plan", "launch", "status"))

    def test_default_plan_and_dry_launch_never_create_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            plan = json.loads(json.dumps(self.plan))
            output_root = temporary_path / "must_not_exist"
            plan["runtime"]["output_root"] = str(output_root)
            temp_plan = temporary_path / "plan.json"
            temp_plan.write_text(
                json.dumps(plan, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = self.launcher.main(
                    ["launch", "--plan-file", str(temp_plan), "--json"]
                )
            self.assertEqual(rc, 0)
            rendered = json.loads(stdout.getvalue())
            self.assertEqual(rendered["mode"], "dry-run")
            self.assertEqual(len(rendered["commands"]), 6)
            self.assertTrue(
                all(
                    "--expected-plan-sha256" in item["command"]
                    for item in rendered["commands"]
                )
            )
            self.assertFalse(output_root.exists())

            # Exercise the execute refusal without ever entering the process
            # creation path, even when this test itself runs on the selected
            # Canoe pod after the real plan has been enabled.
            plan["execution_control"]["launch_enabled"] = False
            temp_plan.write_text(
                json.dumps(plan, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            digest = hashlib.sha256(temp_plan.read_bytes()).hexdigest()
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = self.launcher.main(
                    [
                        "launch",
                        "--plan-file",
                        str(temp_plan),
                        "--execute",
                        "--confirm-plan-sha256",
                        digest,
                    ]
                )
            self.assertEqual(rc, 2)
            self.assertIn("launch_enabled is false", stderr.getvalue())
            self.assertFalse(output_root.exists())

    def test_status_is_read_only_for_unstarted_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            plan = json.loads(json.dumps(self.plan))
            output_root = temporary_path / "status_must_not_create"
            plan["runtime"]["output_root"] = str(output_root)
            temp_plan = temporary_path / "plan.json"
            temp_plan.write_text(json.dumps(plan), encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = self.launcher.main(
                    ["status", "--plan-file", str(temp_plan), "--json"]
                )
            self.assertEqual(rc, 0)
            statuses = json.loads(stdout.getvalue())
            self.assertEqual(len(statuses), 6)
            self.assertEqual({item["state"] for item in statuses}, {"planned"})
            self.assertFalse(output_root.exists())


if __name__ == "__main__":
    unittest.main()
