"""Tests for multi-protein spatial transfer tuning."""

from __future__ import annotations

import unittest

from qcloud import ProteinShadowTuningPreset
from validation.protein_transfer_tuning import (
    ProteinTransferTuningRunner,
    SpatialTransferCandidate,
    build_spatial_transfer_candidate_grid,
)


class ProteinTransferTuningTests(unittest.TestCase):
    """Verify shared spatial-prior tuning across multiple protein benchmarks."""

    def test_candidate_grid_builds_named_candidates(self) -> None:
        candidates = build_spatial_transfer_candidate_grid(
            ProteinShadowTuningPreset(),
            include_baseline=True,
            preferred_distance_scales=(1.0,),
            distance_tolerance_scales=(1.0,),
            attraction_scales=(1.0,),
            repulsion_scales=(1.0,),
            directional_scales=(1.0,),
            chemistry_scales=(1.0, 1.1),
            local_field_energy_scales=(1.0,),
            local_field_repulsion_scales=(1.0,),
            alignment_floors=(0.28,),
        )

        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(isinstance(candidate, SpatialTransferCandidate) for candidate in candidates))
        self.assertEqual(candidates[0].name, "baseline")
        self.assertTrue(candidates[1].name.startswith("protein_transfer_"))

    def test_runner_scores_single_candidate_across_spike_and_barnase(self) -> None:
        runner = ProteinTransferTuningRunner(
            replicates=1,
            steps_per_replicate=0,
            sample_interval=1,
            benchmark_repeats=1,
        )
        candidate = SpatialTransferCandidate(
            name="baseline",
            preset=ProteinShadowTuningPreset(),
        )

        report = runner.tune((candidate,))

        self.assertEqual(report.scenario_names, ("spike_ace2", "barnase_barstar"))
        self.assertEqual(report.best_candidate().candidate.name, "baseline")
        self.assertEqual(len(report.best_candidate().scenario_scores), 2)
        self.assertGreaterEqual(report.best_candidate().mean_score, 0.0)

    def test_report_exposes_baseline_and_top_candidates(self) -> None:
        runner = ProteinTransferTuningRunner(
            replicates=1,
            steps_per_replicate=0,
            sample_interval=1,
            benchmark_repeats=1,
        )
        candidates = (
            SpatialTransferCandidate(name="baseline", preset=ProteinShadowTuningPreset(), metadata={"is_baseline": True}),
            SpatialTransferCandidate(
                name="variant",
                preset=ProteinShadowTuningPreset(spatial_profile_chemistry_scale=1.05),
            ),
        )

        report = runner.tune(candidates)

        self.assertIsNotNone(report.baseline_candidate())
        self.assertEqual(report.top_candidates(limit=1)[0].candidate.name, report.best_candidate().candidate.name)
        self.assertIn("best_margin_over_baseline", report.to_dict())


if __name__ == "__main__":
    unittest.main()
