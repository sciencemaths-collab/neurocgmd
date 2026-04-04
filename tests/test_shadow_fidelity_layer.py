"""Tests for the post-roadmap shadow coarse-grained fidelity layer."""

from __future__ import annotations

import unittest

from core.state import ParticleState, SimulationState, StateProvenance, ThermodynamicState, UnitSystem
from core.types import BeadId, SimulationId, StateId
from forcefields import TrustedNonbondedProfile, TrustedParameterSet, TrustedScienceSource
from physics.forces.composite import ForceEvaluation
from qcloud import (
    QCloudForceCoupler,
    RefinementRegion,
    RegionTriggerKind,
    ShadowCloudBuilder,
    ShadowCorrectionPolicy,
    ShadowDrivenCorrectionModel,
    ShadowMappingLibrary,
    ShadowMappingRule,
    ShadowSiteTemplate,
)
from topology import Bead, BeadRole, BeadType, SystemTopology
from validation import ReferenceForceTarget, ShadowFidelityAssessor


class ShadowFidelityLayerTests(unittest.TestCase):
    """Verify the new trusted-source and shadow-fidelity substrate."""

    def _build_state(self) -> SimulationState:
        return SimulationState(
            units=UnitSystem.md_nano(),
            particles=ParticleState(
                positions=((0.0, 0.0, 0.0), (1.2, 0.0, 0.0)),
                masses=(1.0, 1.0),
            ),
            thermodynamics=ThermodynamicState(),
            provenance=StateProvenance(
                simulation_id=SimulationId("sim-shadow"),
                state_id=StateId("state-shadow"),
                parent_state_id=None,
                created_by="unit-test",
                stage="checkpoint",
            ),
            time=0.0,
            step=1,
            potential_energy=0.0,
        )

    def _build_topology(self) -> SystemTopology:
        return SystemTopology(
            system_id="shadow-system",
            bead_types=(
                BeadType(name="receptor", role=BeadRole.FUNCTIONAL),
                BeadType(name="ligand", role=BeadRole.FUNCTIONAL),
            ),
            beads=(
                Bead(bead_id=BeadId("b0"), particle_index=0, bead_type="receptor", label="R0"),
                Bead(bead_id=BeadId("b1"), particle_index=1, bead_type="ligand", label="L0"),
            ),
        )

    def _build_trusted_parameter_set(self) -> TrustedParameterSet:
        return TrustedParameterSet(
            name="trusted_proxy_parameters",
            summary="Literature-inspired trusted pair interaction used by the shadow layer.",
            sources=(
                TrustedScienceSource(
                    label="literature_pair_profile",
                    source_type="literature_parameterization",
                    citation="Example trusted source for unit testing.",
                    adaptation_notes="Reimplemented in repo-native shadow correction form.",
                ),
            ),
            nonbonded_profiles=(
                TrustedNonbondedProfile(
                    bead_type_a="ligand",
                    bead_type_b="receptor",
                    sigma=1.0,
                    epsilon=0.6,
                    cutoff=3.0,
                    electrostatic_strength=-0.25,
                    source_label="literature_pair_profile",
                ),
            ),
        )

    def _build_shadow_mapping_library(self) -> ShadowMappingLibrary:
        return ShadowMappingLibrary(
            rules=(
                ShadowMappingRule(
                    bead_type="receptor",
                    source_label="literature_pair_profile",
                    site_templates=(
                        ShadowSiteTemplate(
                            site_name="core",
                            relative_offset=(0.0, 0.0, 0.0),
                            sigma_scale=1.0,
                            epsilon_scale=1.0,
                            charge_scale=1.0,
                        ),
                        ShadowSiteTemplate(
                            site_name="halo",
                            relative_offset=(0.0, 0.15, 0.0),
                            sigma_scale=0.9,
                            epsilon_scale=0.8,
                            charge_scale=0.4,
                            occupancy=0.5,
                        ),
                    ),
                ),
                ShadowMappingRule(
                    bead_type="ligand",
                    source_label="literature_pair_profile",
                    site_templates=(
                        ShadowSiteTemplate(
                            site_name="core",
                            relative_offset=(0.0, 0.0, 0.0),
                            sigma_scale=1.0,
                            epsilon_scale=1.0,
                            charge_scale=-1.0,
                        ),
                    ),
                ),
            )
        )

    def test_trusted_parameter_set_resolves_profile_by_bead_types(self) -> None:
        parameter_set = self._build_trusted_parameter_set()

        profile = parameter_set.nonbonded_profile_for_bead_types("receptor", "ligand")

        self.assertEqual(profile.source_label, "literature_pair_profile")
        self.assertEqual(parameter_set.source_for("literature_pair_profile").source_type, "literature_parameterization")

    def test_shadow_cloud_builder_builds_mirrored_sites_for_selected_particles(self) -> None:
        builder = ShadowCloudBuilder(mapping_library=self._build_shadow_mapping_library())

        snapshot = builder.build(
            self._build_state(),
            self._build_topology(),
            particle_indices=(0,),
        )

        self.assertEqual(snapshot.particle_indices, (0,))
        self.assertEqual(snapshot.site_count(), 2)
        self.assertEqual(snapshot.sites[0].site_name, "core")
        self.assertEqual(snapshot.sites[1].position, (0.0, 0.15, 0.0))
        self.assertEqual(snapshot.source_labels, ("literature_pair_profile",))

    def test_shadow_driven_correction_model_generates_local_qcloud_correction(self) -> None:
        model = ShadowDrivenCorrectionModel(
            trusted_parameter_set=self._build_trusted_parameter_set(),
            shadow_builder=ShadowCloudBuilder(mapping_library=self._build_shadow_mapping_library()),
            policy=ShadowCorrectionPolicy(minimum_site_distance=0.05, max_interaction_distance=2.5),
        )
        state = self._build_state()
        region = RefinementRegion(
            region_id="region-shadow-001",
            state_id=state.provenance.state_id,
            particle_indices=(0, 1),
            seed_pairs=((0, 1),),
            trigger_kinds=(RegionTriggerKind.ADAPTIVE_EDGE,),
            score=1.0,
        )

        correction = model.evaluate(
            state,
            self._build_topology(),
            region,
        )

        self.assertEqual(correction.region_id, "region-shadow-001")
        self.assertEqual(correction.method_label, "shadow_driven_correction_model")
        self.assertNotEqual(correction.energy_delta, 0.0)
        self.assertEqual(correction.affected_particles(), (0, 1))
        self.assertIn("literature_pair_profile", correction.metadata["source_labels"])

    def test_shadow_correction_couples_into_force_evaluation(self) -> None:
        model = ShadowDrivenCorrectionModel(
            trusted_parameter_set=self._build_trusted_parameter_set(),
            shadow_builder=ShadowCloudBuilder(mapping_library=self._build_shadow_mapping_library()),
        )
        state = self._build_state()
        topology = self._build_topology()
        region = RefinementRegion(
            region_id="region-shadow-002",
            state_id=state.provenance.state_id,
            particle_indices=(0, 1),
            seed_pairs=((0, 1),),
            trigger_kinds=(RegionTriggerKind.ADAPTIVE_EDGE,),
            score=1.0,
        )
        coupler = QCloudForceCoupler(max_energy_delta_magnitude=10.0, max_force_delta_magnitude=10.0)

        result = coupler.couple(
            ForceEvaluation(
                forces=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                potential_energy=0.0,
            ),
            state,
            topology,
            (region,),
            model,
        )

        self.assertNotEqual(result.force_evaluation.potential_energy, 0.0)
        self.assertEqual(len(result.applied_corrections), 1)
        self.assertEqual(result.metadata["applied_region_count"], 1)

    def test_shadow_fidelity_assessor_reports_improvement(self) -> None:
        assessor = ShadowFidelityAssessor()
        target = ReferenceForceTarget(
            label="trusted_unit_target",
            potential_energy=-0.4,
            forces=((0.5, 0.0, 0.0), (-0.5, 0.0, 0.0)),
        )
        baseline = ForceEvaluation(
            forces=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            potential_energy=0.0,
        )
        corrected = ForceEvaluation(
            forces=((0.4, 0.0, 0.0), (-0.4, 0.0, 0.0)),
            potential_energy=-0.35,
            metadata={"shadow_corrected": True},
        )

        report = assessor.assess(
            target=target,
            baseline=baseline,
            corrected=corrected,
        )

        self.assertTrue(report.passed())
        self.assertTrue(all(metric.improved for metric in report.metrics))
        self.assertLess(
            report.metric_for("force_rms_error").corrected_error,
            report.metric_for("force_rms_error").baseline_error,
        )


if __name__ == "__main__":
    unittest.main()
