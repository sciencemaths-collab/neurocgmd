"""Tests for the spatially aware intelligent local force layer."""

from __future__ import annotations

import unittest

from core.state import ParticleState, SimulationState, StateProvenance, ThermodynamicState, UnitSystem
from core.types import BeadId, SimulationId, StateId
from forcefields import ProteinSpatialProfileFactory
from qcloud import RefinementRegion, RegionTriggerKind, SpatialSemanticFieldModel
from topology import Bead, BeadRole, BeadType, Bond, SystemTopology


class SpatialSemanticForcefieldTests(unittest.TestCase):
    """Verify the new spatial-semantic interaction layer stays explicit and usable."""

    def _build_state(self) -> SimulationState:
        return SimulationState(
            units=UnitSystem.md_nano(),
            particles=ParticleState(
                positions=((0.0, 0.0, 0.0), (0.85, 0.0, 0.0), (1.80, 0.0, 0.0), (2.65, 0.0, 0.0)),
                masses=(1.0, 1.0, 1.0, 1.0),
            ),
            thermodynamics=ThermodynamicState(),
            provenance=StateProvenance(
                simulation_id=SimulationId("sim-spatial"),
                state_id=StateId("state-spatial"),
                parent_state_id=None,
                created_by="unit-test",
                stage="checkpoint",
            ),
            time=0.0,
            step=0,
            potential_energy=0.0,
        )

    def _build_topology(self) -> SystemTopology:
        return SystemTopology(
            system_id="spatial-system",
            bead_types=(
                BeadType(name="acidic", role=BeadRole.FUNCTIONAL),
                BeadType(name="basic", role=BeadRole.FUNCTIONAL),
            ),
            beads=(
                Bead(BeadId("a0"), 0, "acidic", "acidic_support"),
                Bead(BeadId("a1"), 1, "acidic", "acidic_hotspot"),
                Bead(BeadId("b0"), 2, "basic", "basic_hotspot"),
                Bead(BeadId("b1"), 3, "basic", "basic_support"),
            ),
            bonds=(Bond(0, 1), Bond(2, 3)),
        )

    def test_spatial_profile_factory_builds_distance_aware_profiles(self) -> None:
        parameter_set = ProteinSpatialProfileFactory().build_parameter_set(
            self._build_topology(),
            scenario_label="unit_test",
            reference_label="synthetic_reference",
        )

        profile = parameter_set.profile_for_bead_types("acidic", "basic")

        self.assertLess(profile.preferred_distance, profile.cutoff)
        self.assertGreater(profile.attraction_strength, 0.5)
        self.assertGreater(profile.chemistry_strength, 0.5)

    def test_spatial_field_model_produces_intelligent_local_correction(self) -> None:
        state = self._build_state()
        topology = self._build_topology()
        parameter_set = ProteinSpatialProfileFactory().build_parameter_set(
            topology,
            scenario_label="unit_test",
        )
        region = RefinementRegion(
            region_id="region-spatial",
            state_id=state.provenance.state_id,
            particle_indices=(0, 1, 2, 3),
            seed_pairs=((1, 2),),
            trigger_kinds=(RegionTriggerKind.ADAPTIVE_EDGE,),
            score=1.0,
        )

        evaluation = SpatialSemanticFieldModel(parameter_set=parameter_set).evaluate(state, topology, region)

        self.assertNotEqual(evaluation.energy_delta, 0.0)
        self.assertTrue(evaluation.force_deltas)
        self.assertGreater(len(evaluation.pair_contributions), 0)
        self.assertGreater(evaluation.quality_score, 0.2)
        self.assertIn("protein_spatial_semantic_priors", evaluation.metadata["source_labels"])


if __name__ == "__main__":
    unittest.main()
