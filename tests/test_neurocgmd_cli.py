"""End-to-end tests for the manifest-driven NeuroCGMD CLI."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class NeuroCGMDCLITests(unittest.TestCase):
    """Verify prepare/run/analyze stay on the shared engine path."""

    def _write_manifest(self, directory: Path) -> Path:
        output_dir = directory / "outputs"
        manifest_path = directory / "run.toml"
        manifest_path.write_text(
            "\n".join(
                [
                    "[system]",
                    'name = "barnase_barstar_cli_run"',
                    'structure = "benchmarks/reference_cases/data/1BRS.pdb"',
                    "",
                    "[[system.entity_groups]]",
                    'entity_id = "barnase"',
                    'chain_ids = ["A"]',
                    "",
                    "[[system.entity_groups]]",
                    'entity_id = "barstar"',
                    'chain_ids = ["D"]',
                    "",
                    "[solvent]",
                    'solvent_mode = "explicit"',
                    'water_model = "tip3p"',
                    'box_type = "dodecahedron"',
                    "padding_nm = 1.0",
                    "",
                    "[ions]",
                    "neutralize = true",
                    'salt = "NaCl"',
                    "ionic_strength_molar = 0.15",
                    "",
                    "[outputs]",
                    f'output_dir = "{output_dir}"',
                    'prepared_bundle = "prepared_bundle.json"',
                    'trajectory = "trajectory.jsonl"',
                    'energy = "energies.csv"',
                    'checkpoint = "checkpoint.json"',
                    'run_summary = "run_summary.json"',
                    'analysis_json = "analysis.json"',
                    'analysis_html = "analysis.html"',
                    'log = "run.log"',
                    "",
                    "[stages.em]",
                    "enabled = true",
                    "max_steps = 2",
                    "tolerance = 1000000.0",
                    "step_size = 0.0005",
                    "",
                    "[stages.nvt]",
                    "enabled = true",
                    'ensemble = "NVT"',
                    "dt = 0.002",
                    "nsteps = 2",
                    "temperature = 300.0",
                    "friction_coefficient = 0.7",
                    "trajectory_stride = 1",
                    "energy_stride = 1",
                    "checkpoint_stride = 1",
                    "",
                    "[stages.npt]",
                    "enabled = false",
                    "",
                    "[stages.production]",
                    "enabled = true",
                    'ensemble = "NVT"',
                    "dt = 0.002",
                    "nsteps = 2",
                    "temperature = 300.0",
                    "friction_coefficient = 0.7",
                    "trajectory_stride = 1",
                    "energy_stride = 1",
                    "checkpoint_stride = 1",
                    "",
                    "[hybrid.qcloud]",
                    "enabled = true",
                    "max_regions = 2",
                    "trigger_threshold = 0.45",
                    "",
                    "[hybrid.ml]",
                    "enabled = true",
                    'model = "scalable_residual"',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return manifest_path

    def test_prepare_run_and_analyze_commands_emit_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = self._write_manifest(temp_path)

            prepare_result = subprocess.run(
                [sys.executable, "scripts/neurocgmd.py", "prepare", "--config", str(manifest_path)],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(prepare_result.returncode, 0, msg=prepare_result.stderr or prepare_result.stdout)

            run_result = subprocess.run(
                [
                    sys.executable,
                    "scripts/neurocgmd.py",
                    "run",
                    "--config",
                    str(manifest_path),
                    "--benchmark-repeats",
                    "0",
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(run_result.returncode, 0, msg=run_result.stderr or run_result.stdout)

            analyze_result = subprocess.run(
                [sys.executable, "scripts/neurocgmd.py", "analyze", "--config", str(manifest_path)],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(analyze_result.returncode, 0, msg=analyze_result.stderr or analyze_result.stdout)

            output_dir = temp_path / "outputs"
            run_summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
            analysis = json.loads((output_dir / "analysis.json").read_text(encoding="utf-8"))

            self.assertTrue((output_dir / "prepared_bundle.json").exists())
            self.assertTrue((output_dir / "trajectory.jsonl").exists())
            self.assertTrue((output_dir / "energies.csv").exists())
            self.assertTrue((output_dir / "checkpoint.json").exists())
            self.assertTrue((output_dir / "analysis.html").exists())
            self.assertGreaterEqual(len(run_summary["stage_records"]), 2)
            self.assertIn("headline_metrics", analysis)


if __name__ == "__main__":
    unittest.main()
