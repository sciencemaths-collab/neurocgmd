"""Tests for structural-reference ingestion and proxy comparison metrics."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from benchmarks import barnase_barstar_structure_target, spike_ace2_structure_target
from validation import LandmarkObservation, best_fit_rmsd, compare_landmark_observations


class StructureMetricsTests(unittest.TestCase):
    """Verify the first honest structural-comparison layer."""

    def test_best_fit_rmsd_is_zero_for_identical_landmarks(self) -> None:
        points = ((0.0, 0.0, 0.0), (1.0, 0.5, 0.0), (0.3, 1.0, 0.2))

        self.assertAlmostEqual(best_fit_rmsd(points, points), 0.0, places=6)

    def test_spike_structure_target_comparison_recovers_all_landmarks(self) -> None:
        target = spike_ace2_structure_target()
        observations = tuple(
            LandmarkObservation(label=landmark.label, position=landmark.target_position)
            for landmark in target.landmarks
        )

        report = compare_landmark_observations(
            state_id="state-structure",
            target=target,
            observations=observations,
        )

        self.assertEqual(report.title, "Atomistic Alignment")
        metric_map = {metric.label: metric for metric in report.metrics}
        self.assertEqual(metric_map["Matched Residue Groups"].value, "12/12")
        self.assertEqual(metric_map["Contact Recovery"].value, "5/5")
        self.assertTrue(metric_map["Atomistic Centroid RMSD"].value.startswith("0.000"))

    def test_barnase_structure_target_comparison_recovers_all_landmarks(self) -> None:
        target = barnase_barstar_structure_target()
        observations = tuple(
            LandmarkObservation(label=landmark.label, position=landmark.target_position)
            for landmark in target.landmarks
        )

        report = compare_landmark_observations(
            state_id="state-barnase-structure",
            target=target,
            observations=observations,
        )

        metric_map = {metric.label: metric for metric in report.metrics}
        self.assertEqual(metric_map["Matched Residue Groups"].value, "6/6")
        self.assertEqual(metric_map["Contact Recovery"].value, "4/4")
        self.assertTrue(metric_map["Atomistic Centroid RMSD"].value.startswith("0.000"))

    def test_import_safe_pdb_loader_parses_local_file(self) -> None:
        module_path = Path(__file__).resolve().parents[1] / "io" / "pdb_loader.py"
        spec = importlib.util.spec_from_file_location("test_pdb_loader_module", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules.setdefault("test_pdb_loader_module", module)
        spec.loader.exec_module(module)

        pdb_text = (
            "ATOM      1  CA  ALA A   1      11.104  13.207   9.550  1.00 21.46           C\n"
            "ATOM      2  CB  ALA A   1      12.560  13.540   9.220  1.00 19.20           C\n"
            "ATOM      3  CA  GLY B   2       8.100   4.210   2.330  1.00 18.10           C\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            pdb_path = Path(temp_dir) / "sample.pdb"
            pdb_path.write_text(pdb_text, encoding="utf-8")
            structure = module.load_pdb_file(pdb_path)

        self.assertEqual(structure.structure_id, "sample")
        self.assertEqual(len(structure.atoms), 3)
        self.assertEqual(structure.atoms_for_chain("A")[0].atom_name, "CA")
        self.assertEqual(structure.first_atom(chain_id="B", residue_sequence=2, atom_name="CA").chain_id, "B")


if __name__ == "__main__":
    unittest.main()
