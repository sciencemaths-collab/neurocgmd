"""Tests for the protein-general shadow tuning layer."""

from __future__ import annotations

import unittest

from qcloud import ProteinShadowTuner, ProteinShadowTuningPreset
from sampling.scenarios import BarnaseBarstarScenario, SpikeAce2Scenario


class ProteinShadowTuningTests(unittest.TestCase):
    """Verify the generalized protein shadow bundle across benchmark proteins."""

    def test_runtime_bundle_builds_for_spike_topology(self) -> None:
        scenario = SpikeAce2Scenario()
        topology = scenario.build_setup().topology

        bundle = ProteinShadowTuner().build_runtime_bundle(
            topology=topology,
            scenario_label=scenario.name,
            base_time_step=scenario.recommended_time_step,
            base_friction=scenario.recommended_friction,
            reference_label=scenario.reference_case().name,
        )

        self.assertEqual(
            set(bundle.mapping_library.registered_bead_types()),
            set(topology.bead_type_names),
        )
        self.assertGreater(bundle.dynamics_recommendation.time_step, scenario.recommended_time_step)
        self.assertGreater(bundle.dynamics_recommendation.friction_coefficient, scenario.recommended_friction)
        spatial_profile = bundle.spatial_parameter_set.profile_for_bead_types("hotspot", "helix")
        self.assertGreater(spatial_profile.attraction_strength, 0.0)
        self.assertTrue(bundle.metadata["spatial_forcefield_enabled"])
        self.assertEqual(bundle.metadata["spatial_profile_attraction_scale"], 1.0)

    def test_runtime_bundle_builds_for_barnase_topology(self) -> None:
        scenario = BarnaseBarstarScenario()
        topology = scenario.build_setup().topology

        bundle = ProteinShadowTuner().build_runtime_bundle(
            topology=topology,
            scenario_label=scenario.name,
            base_time_step=scenario.recommended_time_step,
            base_friction=scenario.recommended_friction,
            reference_label=scenario.reference_case().name,
        )

        self.assertEqual(
            set(bundle.mapping_library.registered_bead_types()),
            set(topology.bead_type_names),
        )
        self.assertIn("basic", topology.bead_type_names)
        self.assertIn("acidic", topology.bead_type_names)
        profile = bundle.parameter_set.nonbonded_profile_for_bead_types("basic", "acidic")
        self.assertLess(profile.electrostatic_strength, 0.0)
        self.assertEqual(profile.fidelity_label, "protein_general_shadow")
        spatial_profile = bundle.spatial_parameter_set.profile_for_bead_types("basic", "acidic")
        self.assertGreater(spatial_profile.chemistry_strength, 0.5)
        self.assertGreater(bundle.spatial_field_policy.energy_scale, 0.0)

    def test_runtime_bundle_applies_spatial_profile_scaling_from_preset(self) -> None:
        scenario = BarnaseBarstarScenario()
        topology = scenario.build_setup().topology
        tuner = ProteinShadowTuner(
            preset=ProteinShadowTuningPreset(
                spatial_profile_attraction_scale=1.10,
                spatial_profile_repulsion_scale=0.95,
                spatial_profile_chemistry_scale=1.20,
            )
        )

        bundle = tuner.build_runtime_bundle(
            topology=topology,
            scenario_label=scenario.name,
            base_time_step=scenario.recommended_time_step,
            base_friction=scenario.recommended_friction,
            reference_label=scenario.reference_case().name,
        )

        spatial_profile = bundle.spatial_parameter_set.profile_for_bead_types("basic", "acidic")
        self.assertAlmostEqual(bundle.metadata["spatial_profile_attraction_scale"], 1.10)
        self.assertAlmostEqual(bundle.metadata["spatial_profile_repulsion_scale"], 0.95)
        self.assertAlmostEqual(bundle.metadata["spatial_profile_chemistry_scale"], 1.20)
        self.assertGreater(spatial_profile.attraction_strength, 1.35)


if __name__ == "__main__":
    unittest.main()
