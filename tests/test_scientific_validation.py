"""Tests for repeated scientific-validation reporting and plotting."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.live_dashboard import build_live_dashboard_context, collect_dashboard_step_data
from scripts.plot_scientific_validation import main as scientific_validation_main
from validation import ScientificValidationSample


class ScientificValidationTests(unittest.TestCase):
    """Verify scientific-validation sampling and export behavior."""

    def test_sample_can_be_built_from_live_spike_ace2_reports(self) -> None:
        context = build_live_dashboard_context("spike_ace2")
        step_data = collect_dashboard_step_data(context, benchmark_repeats=1)
        structure_report = context.scenario.build_structure_report(
            step_data.state,
            progress=step_data.progress,
        )
        fidelity_report = context.scenario.build_fidelity_report(
            step_data.state,
            baseline_evaluation=step_data.base_evaluation,
            corrected_evaluation=step_data.qcloud_result.force_evaluation,
            progress=step_data.progress,
        )

        sample = ScientificValidationSample.from_reports(
            replicate_index=0,
            sample_step=0,
            state_step=step_data.state.step,
            time=step_data.state.time,
            progress=step_data.progress,
            structure_report=structure_report,
            fidelity_report=fidelity_report,
            benchmark_report=step_data.benchmark_report,
            metadata={"scenario": context.scenario.name},
        )

        self.assertEqual(sample.replicate_index, 0)
        self.assertGreaterEqual(sample.assembly_score, 0.0)
        self.assertLessEqual(sample.contact_recovery_fraction, 1.0)
        self.assertIn("qcloud_coupling", sample.benchmark_case_seconds.to_dict())
        self.assertGreaterEqual(sample.atomistic_centroid_rmsd, 0.0)

    def test_script_writes_plotted_validation_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            exit_code = scientific_validation_main(
                [
                    "--output-dir",
                    temp_dir,
                    "--replicates",
                    "1",
                    "--steps",
                    "4",
                    "--sample-interval",
                    "2",
                    "--benchmark-repeats",
                    "1",
                    "--deterministic",
                ]
            )

            payload = json.loads((Path(temp_dir) / "validation.json").read_text(encoding="utf-8"))
            html = (Path(temp_dir) / "index.html").read_text(encoding="utf-8")

            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["scenario_name"], "spike_ace2")
            self.assertEqual(payload["replicates"], 1)
            self.assertEqual(len(payload["samples"]), 3)
            self.assertIn("Shadow Energy Error", html)
            self.assertIn("Architecture Timing", html)

    def test_script_supports_barnase_barstar_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            exit_code = scientific_validation_main(
                [
                    "--scenario",
                    "barnase_barstar",
                    "--output-dir",
                    temp_dir,
                    "--replicates",
                    "1",
                    "--steps",
                    "2",
                    "--sample-interval",
                    "1",
                    "--benchmark-repeats",
                    "1",
                    "--deterministic",
                ]
            )

            payload = json.loads((Path(temp_dir) / "validation.json").read_text(encoding="utf-8"))

            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["scenario_name"], "barnase_barstar")
            self.assertEqual(payload["replicates"], 1)


if __name__ == "__main__":
    unittest.main()
