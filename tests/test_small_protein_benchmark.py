"""Tests for the fast small-protein benchmark harness."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmarks.small_protein import SmallProteinBenchmarkRunner, SmallProteinBenchmarkSpec
from config import ProteinEntityGroup


class SmallProteinBenchmarkTests(unittest.TestCase):
    """Keep the single-chain benchmark path runnable and honest."""

    def test_runner_produces_report_for_single_chain_barnase(self) -> None:
        pdb_path = Path("benchmarks/reference_cases/data/1BRS.pdb").resolve()
        spec = SmallProteinBenchmarkSpec(
            name="test_small_barnase_benchmark",
            pdb_path=str(pdb_path),
            entity_group=ProteinEntityGroup(entity_id="barnase", chain_ids=("A",)),
            residues_per_bead=10,
            repeats=1,
            rollout_steps=1,
            warmup_training_passes=2,
            manual_region_size=4,
        )

        report = SmallProteinBenchmarkRunner(spec=spec).run()

        self.assertEqual(report.entity_id, "barnase")
        self.assertGreater(report.residue_count, 0)
        self.assertGreater(report.bead_count, 0)
        self.assertTrue(report.parity_report.all_passed())
        self.assertEqual(report.execution_plan.selection.selected_backend, "reference_cpu_backend")
        self.assertIn("production_hybrid_engine", report.benchmark_report.case_names())
        self.assertIn("classical_only", report.benchmark_report.case_names())
        self.assertIn("hybrid_production", report.to_dict()["engine_modes"])
        production_case = report.benchmark_report.case_for("production_hybrid_engine")
        self.assertIn("preliminary_action", production_case.metadata)
        self.assertIn("final_action", production_case.metadata)
        self.assertIn("qcloud_requested", production_case.metadata)
        self.assertIn("qcloud_applied", production_case.metadata)
        self.assertIn("selected_region_count", production_case.metadata)
        self.assertIn("replay_buffer_size", production_case.metadata)
        self.assertIn("trace_record_count", production_case.metadata)
        self.assertIn("open_episode_count", production_case.metadata)

    def test_script_writes_json_report(self) -> None:
        from scripts.run_small_protein_benchmark import main

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "small_protein_report.json"
            exit_code = main(
                [
                    "--pdb",
                    str(Path("benchmarks/reference_cases/data/1BRS.pdb").resolve()),
                    "--chain",
                    "A",
                    "--repeats",
                    "1",
                    "--warmup-training-passes",
                    "1",
                    "--manual-region-size",
                    "4",
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())
            payload = output_path.read_text(encoding="utf-8")
            self.assertIn("barnase_small_protein_benchmark", payload)
            self.assertIn("production_hybrid_engine", payload)
            self.assertIn("classical_only", payload)

    def test_plot_script_writes_html_report(self) -> None:
        from scripts.plot_small_protein_benchmark import main

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "benchmark_plot"
            exit_code = main(
                [
                    "--pdb",
                    str(Path("benchmarks/reference_cases/data/1BRS.pdb").resolve()),
                    "--chain",
                    "A",
                    "--repeats",
                    "1",
                    "--warmup-training-passes",
                    "1",
                    "--manual-region-size",
                    "4",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            html_path = output_dir / "index.html"
            json_path = output_dir / "benchmark.json"
            self.assertTrue(html_path.exists())
            self.assertTrue(json_path.exists())
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("classical_only", html)
            self.assertIn("hybrid_production", html)
            self.assertIn("production_hybrid_engine", html)


if __name__ == "__main__":
    unittest.main()
