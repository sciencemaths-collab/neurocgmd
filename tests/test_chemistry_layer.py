"""Tests for chemistry semantics, interface analysis, and live feature encoding."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from chemistry import ChemistryInterfaceAnalyzer, ChargeClass, PolarityClass, ProteinChemistryModel
from compartments import CompartmentDomain, CompartmentRegistry
from core.state import ParticleState, SimulationState, StateProvenance, ThermodynamicState, UnitSystem
from core.types import BeadId, SimulationId, StateId
from graph import ConnectivityGraph, DynamicEdgeKind, DynamicEdgeState
from ml.live_features import LiveFeatureEncoder
from topology import Bead, BeadRole, BeadType, Bond, SystemTopology
from validation.fidelity_checks import FidelityComparisonReport, FidelityMetric
from validation.structure_metrics import StructureComparisonReport, StructureMetric


class ChemistryLayerTests(unittest.TestCase):
    """Verify chemistry semantics stay explicit and feed observer-side features."""

    def _build_state(self) -> SimulationState:
        return SimulationState(
            units=UnitSystem.md_nano(),
            particles=ParticleState(
                positions=((0.0, 0.0, 0.0), (0.6, 0.8, 0.0), (1.1, 0.0, 0.0), (1.7, 0.8, 0.0)),
                masses=(1.0, 1.0, 1.0, 1.0),
            ),
            thermodynamics=ThermodynamicState(),
            provenance=StateProvenance(
                simulation_id=SimulationId("sim-chem"),
                state_id=StateId("state-chem"),
                parent_state_id=None,
                created_by="unit-test",
                stage="checkpoint",
            ),
            time=0.0,
            step=0,
            potential_energy=-1.0,
        )

    def _build_topology(self) -> SystemTopology:
        return SystemTopology(
            system_id="chem-system",
            bead_types=(
                BeadType(name="hotspot", role=BeadRole.ANCHOR),
                BeadType(name="support", role=BeadRole.STRUCTURAL),
            ),
            beads=(
                Bead(BeadId("a0"), 0, "hotspot", "acidic_hotspot", residue_name="GLU", compartment_hint="A"),
                Bead(BeadId("a1"), 1, "support", "aromatic_support", residue_name="TYR", compartment_hint="A"),
                Bead(BeadId("b0"), 2, "hotspot", "basic_hotspot", residue_name="LYS", compartment_hint="B"),
                Bead(BeadId("b1"), 3, "support", "ridge_anchor", residue_name="TRP", compartment_hint="B"),
            ),
            bonds=(Bond(0, 1), Bond(2, 3)),
        )

    def _build_compartments(self) -> CompartmentRegistry:
        return CompartmentRegistry(
            particle_count=4,
            domains=(
                CompartmentDomain.from_members("A", "A", (0, 1)),
                CompartmentDomain.from_members("B", "B", (2, 3)),
            ),
        )

    def _build_graph(self) -> ConnectivityGraph:
        return ConnectivityGraph(
            particle_count=4,
            step=0,
            edges=(
                DynamicEdgeState(0, 1, DynamicEdgeKind.STRUCTURAL_LOCAL, 1.0, 1.0, 0, 0),
                DynamicEdgeState(2, 3, DynamicEdgeKind.STRUCTURAL_LOCAL, 1.0, 1.0, 0, 0),
                DynamicEdgeState(0, 2, DynamicEdgeKind.ADAPTIVE_LONG_RANGE, 0.9, 1.1, 0, 0),
            ),
        )

    def test_protein_chemistry_model_prefers_residue_library_when_available(self) -> None:
        descriptor = ProteinChemistryModel().descriptor_for_bead(self._build_topology(), 0)

        self.assertEqual(descriptor.charge_class, ChargeClass.NEGATIVE)
        self.assertEqual(descriptor.polarity_class, PolarityClass.POLAR)
        self.assertAlmostEqual(descriptor.formal_charge, -1.0)
        self.assertIn("residue:GLU", descriptor.descriptor_source)

    def test_interface_analyzer_reports_favorable_cross_interface_pairs(self) -> None:
        report = ChemistryInterfaceAnalyzer().assess(
            self._build_state(),
            self._build_topology(),
            self._build_compartments(),
            compartment_ids=("A", "B"),
            distance_cutoff=1.9,
        )

        self.assertGreater(report.favorable_pair_fraction, 0.45)
        self.assertGreater(report.charge_complementarity, 0.55)
        self.assertGreater(report.mean_pair_score, 0.45)
        self.assertEqual(report.compartment_ids, ("A", "B"))
        self.assertTrue(report.dominant_pairs)

    def test_live_feature_encoder_captures_chemistry_structure_and_fidelity(self) -> None:
        chemistry_report = ChemistryInterfaceAnalyzer().assess(
            self._build_state(),
            self._build_topology(),
            self._build_compartments(),
            compartment_ids=("A", "B"),
            distance_cutoff=1.9,
        )
        structure_report = StructureComparisonReport(
            title="Atomistic Alignment",
            summary="test",
            metrics=(
                StructureMetric("Atomistic Centroid RMSD", "4.000"),
                StructureMetric("Contact Recovery", "3/4"),
                StructureMetric("Dominant Pair Error", "0.900"),
            ),
        )
        fidelity_report = FidelityComparisonReport(
            title="Shadow Fidelity",
            target_label="test-target",
            metrics=(
                FidelityMetric("energy_absolute_error", 1.0, 0.8, True),
                FidelityMetric("force_rms_error", 1.2, 1.3, False),
                FidelityMetric("max_force_component_error", 1.6, 1.1, True),
            ),
        )
        progress = SimpleNamespace(
            assembly_score=0.55,
            interface_distance=1.4,
            cross_contact_count=3,
            target_contact_count=4,
            graph_bridge_count=1,
            target_graph_bridge_count=2,
        )

        features = LiveFeatureEncoder().encode(
            self._build_state(),
            self._build_graph(),
            progress=progress,
            chemistry_report=chemistry_report,
            structure_report=structure_report,
            fidelity_report=fidelity_report,
        )

        self.assertAlmostEqual(features.value("structure_contact_recovery"), 0.75)
        self.assertAlmostEqual(features.value("shadow_force_regression"), 1.0)
        self.assertGreater(features.value("chemistry_mean_pair_score"), 0.4)


if __name__ == "__main__":
    unittest.main()
