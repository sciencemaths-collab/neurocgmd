"""Tests for experimentally grounded reference benchmark cases."""

from __future__ import annotations

import unittest

from benchmarks import (
    barnase_barstar_reference_case,
    barnase_barstar_structure_target,
    build_barnase_barstar_proxy_report,
    build_spike_ace2_proxy_report,
    spike_ace2_reference_case,
    spike_ace2_structure_target,
)


class ReferenceCaseTests(unittest.TestCase):
    """Verify the real-world benchmark target layer."""

    def test_barnase_barstar_reference_case_exposes_known_answers(self) -> None:
        case = barnase_barstar_reference_case()

        self.assertEqual(case.name, "barnase_barstar")
        self.assertEqual(case.structural_reference.bound_complex_pdb_id, "1BRS")
        self.assertEqual(case.structural_reference.unbound_partner_pdb_ids, ("1A2P", "1A19"))
        self.assertAlmostEqual(case.observable_for("association_rate_constant").expected_value, 6.0e8)
        self.assertAlmostEqual(case.observable_for("dissociation_constant").expected_value, 1.3e-14)
        self.assertIn("Estimate k_on, k_off, and K_d", case.recommended_comparisons[2])

    def test_reference_case_serializes_stable_metadata(self) -> None:
        payload = barnase_barstar_reference_case().to_dict()

        self.assertEqual(payload["title"], "Barnase-Barstar Association Benchmark")
        self.assertEqual(payload["structural_reference"]["bound_complex_pdb_id"], "1BRS")
        self.assertEqual(len(payload["sources"]), 6)
        self.assertGreaterEqual(len(payload["observables"]), 5)

    def test_proxy_comparison_report_exposes_known_targets(self) -> None:
        report = build_barnase_barstar_proxy_report(
            reference_case=barnase_barstar_reference_case(),
            current_stage="Electrostatic Steering",
            interface_distance=1.8,
            bound_distance=1.25,
            cross_contact_count=2,
            target_contact_count=5,
            bound=False,
        )

        self.assertEqual(report.title, "Experimental Reference")
        self.assertEqual(report.metrics[0].target_value, "1BRS")
        self.assertIn("1.30e-14", report.metrics[2].target_value)
        self.assertEqual(report.metrics[-1].current_value, "1.80 / 2")

    def test_spike_ace2_reference_case_exposes_known_answers(self) -> None:
        case = spike_ace2_reference_case()

        self.assertEqual(case.name, "spike_ace2")
        self.assertEqual(case.structural_reference.bound_complex_pdb_id, "6M0J")
        self.assertEqual(case.structural_reference.unbound_partner_pdb_ids, ("1R42", "6VYB"))
        self.assertAlmostEqual(case.observable_for("apparent_dissociation_constant").expected_value, 1.47e-8)
        self.assertIn("distributed hotspot network", case.recommended_comparisons[3])

    def test_spike_proxy_comparison_report_exposes_known_targets(self) -> None:
        report = build_spike_ace2_proxy_report(
            reference_case=spike_ace2_reference_case(),
            current_stage="Long-Range Steering",
            interface_distance=2.3,
            bound_distance=1.2,
            cross_contact_count=3,
            target_contact_count=8,
            bound=False,
        )

        self.assertEqual(report.title, "Experimental Reference")
        self.assertEqual(report.metrics[0].target_value, "6M0J")
        self.assertIn("1.47e-08", report.metrics[2].target_value)
        self.assertEqual(report.metrics[-1].current_value, "2.30 / 3")

    def test_spike_structure_target_exposes_landmarks(self) -> None:
        target = spike_ace2_structure_target()

        self.assertEqual(target.source_pdb_id, "6M0J")
        self.assertEqual(len(target.landmarks), 12)
        self.assertEqual(len(target.interface_contacts), 5)
        self.assertEqual(target.landmark_for("ACE2_alpha1_hotspot").residue_ids, (31, 35))

    def test_barnase_structure_target_exposes_landmarks(self) -> None:
        target = barnase_barstar_structure_target()

        self.assertEqual(target.source_pdb_id, "1BRS")
        self.assertEqual(len(target.landmarks), 6)
        self.assertEqual(len(target.interface_contacts), 4)
        self.assertEqual(target.landmark_for("Barnase_basic_patch").residue_ids, (27, 59, 83, 87))


if __name__ == "__main__":
    unittest.main()
