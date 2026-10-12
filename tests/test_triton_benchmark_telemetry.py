import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.run_triton_paper_benchmark import (
    GpuMonitor,
    summarize,
    summarize_gpu_telemetry,
)
from tools.run_triton_gptq_reproducibility import aggregate_profile_sections
from gptq_utils.gptq_plus_utils import QuantProfileRecorder


class BenchmarkTelemetryTests(unittest.TestCase):
    def test_monitor_refuses_to_truncate_an_existing_campaign_file(self):
        with tempfile.TemporaryDirectory() as temp:
            out_dir = Path(temp)
            (out_dir / "gpu_telemetry.csv").write_text("owned by another run\n")
            monitor = GpuMonitor(out_dir, "0", 0.1)

            with self.assertRaises(FileExistsError):
                monitor.start()

            self.assertEqual(
                (out_dir / "gpu_telemetry.csv").read_text(),
                "owned by another run\n",
            )

    def test_campaign_summary_requires_layer_fingerprints(self):
        def make_side(tag):
            return {
                "tag": tag,
                "quantization_seconds": 10.0,
                "wall_seconds": 12.0,
                "state_sha256": "same",
                "state_tensors": 507,
                "kl": 0.1,
                "ppl": 10.0,
                "peak_allocated_bytes": 100,
                "peak_reserved_bytes": 120,
                "profile_section_totals_ms": {},
            }

        with tempfile.TemporaryDirectory() as temp:
            out_dir = Path(temp)
            pair = {"control": make_side("control"), "candidate": make_side("candidate")}
            (out_dir / "paired_repetitions.json").write_text(
                __import__("json").dumps([pair])
            )
            environment = {
                "host": "test",
                "nvidia_smi_gpu_query": "test GPU",
                "git_commit": "test",
            }
            config = {"pairs": 1, "model_path": "test-model"}

            missing = summarize(out_dir, "missing-traces", [pair], environment, config)
            self.assertFalse(missing["valid"])

            pair["layer_output_fingerprints"] = {
                "control": [{"layer": 0, "sha256": "layer0"}],
                "candidate": [{"layer": 0, "sha256": "layer0"}],
            }
            (out_dir / "paired_repetitions.json").write_text(
                __import__("json").dumps([pair])
            )
            complete = summarize(out_dir, "with-traces", [pair], environment, config)

        self.assertTrue(complete["valid"])
        self.assertTrue(complete["pair_results"][0]["layer_output_fingerprints_exact"])

    def test_gpu_telemetry_skips_repeated_headers_and_keeps_per_gpu_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "gpu_telemetry.csv"
            fields = [
                "timestamp", "gpu_index", "util_gpu_pct", "util_mem_pct",
                "memory_used_mib", "memory_total_mib", "graphics_clock_mhz",
                "power_w", "temp_c",
            ]
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(fields)
                writer.writerow(["2026-10-04T10:00:00+00:00", 0, 80, 45, 12000, 32607, 2400, 300, 70])
                writer.writerow(fields)  # stale/concurrent header is not a sample
                writer.writerow(["2026-10-04T10:00:05+00:00", 1, 60, 35, 9000, 32607, 2200, 250, 65])

            result = summarize_gpu_telemetry(path)

        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["rows_seen"], 3)
        self.assertEqual(result["invalid_rows"], 1)
        self.assertEqual(result["gpu_indices"], ["0", "1"])
        self.assertEqual(result["util_gpu_pct"]["mean"], 70)
        self.assertEqual(result["by_gpu"]["0"]["util_gpu_pct"]["max"], 80)
        self.assertEqual(result["by_gpu"]["1"]["memory_used_mib"]["mean"], 9000)


    def test_component_profile_parser_uses_untruncated_names_and_sums_layers(self):
        sections = [
            {"name": "layers.0.self_attn.q_proj.fasterquant.block.inner_column_loop", "total_ms": 12.5},
            {"name": "layers.1.mlp.down_proj.fasterquant.block.inner_column_loop", "total_ms": 7.25},
            {"name": "layers.1.mlp.down_proj.fasterquant.block.true_gradient_refresh", "total_ms": 2.0},
            {"name": "layers.1.mlp.down_proj.fasterquant.block.inner_column", "total_ms": 100.0},
        ]

        result = aggregate_profile_sections(sections)

        self.assertEqual(result["fasterquant.block.inner_column_loop"], 19.75)
        self.assertEqual(result["fasterquant.block.true_gradient_refresh"], 2.0)
        self.assertNotIn("fasterquant.block.outer_update_delta_w", result)

    def test_wall_profile_writes_full_names_to_json(self):
        class FakeEvent:
            def elapsed_time(self, _other):
                return 12.5

        old_enabled = QuantProfileRecorder._WALL_ENABLED
        old_stats = QuantProfileRecorder._WALL_STATS
        with tempfile.TemporaryDirectory() as temp:
            profile_path = Path(temp) / "wall_profile_sections.json"
            QuantProfileRecorder._WALL_ENABLED = True
            QuantProfileRecorder._WALL_STATS = {
                "layers.0.mlp.down_proj.fasterquant.block.inner_column_loop": [
                    (FakeEvent(), FakeEvent()),
                ]
            }
            try:
                with patch.dict("os.environ", {"GPTQ_PLUS_WALL_PROFILE_JSON": str(profile_path)}):
                    with patch("torch.cuda.is_available", return_value=False):
                        QuantProfileRecorder.dump_wall_summary(top_k=1)
                payload = __import__("json").loads(profile_path.read_text())
            finally:
                QuantProfileRecorder._WALL_ENABLED = old_enabled
                QuantProfileRecorder._WALL_STATS = old_stats

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(
            payload["sections"][0]["name"],
            "layers.0.mlp.down_proj.fasterquant.block.inner_column_loop",
        )
