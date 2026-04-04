"""Tests for Section 13 validation and benchmarking helpers."""

from __future__ import annotations

import unittest

from ai_control import ExecutiveController
from benchmarks import BaselineBenchmarkSuite
from compartments import CompartmentDomain, CompartmentRegistry
from core.state import ParticleState, SimulationState, StateProvenance, ThermodynamicState, UnitSystem
from core.state_registry import SimulationStateRegistry
from core.types import BeadId, SimulationId, StateId
from forcefields.base_forcefield import BaseForceField, BondParameter, NonbondedParameter
from graph import ConnectivityGraphManager
from integrators.langevin import LangevinIntegrator
from memory import ReplayBuffer, TraceRecord
from ml import ResidualMemoryModel, UncertaintyEstimate
from physics.forces.composite import BaselineForceEvaluator, ForceEvaluation
from qcloud import (
    LocalRegionSelector,
    ParticleForceDelta,
    QCloudCorrection,
    QCloudForceCoupler,
    RegionSelectionPolicy,
    RefinementRegion,
)
from sampling.simulation_loop import SimulationLoop
from topology import Bead, BeadRole, BeadType, Bond, SystemTopology
from validation import DriftThresholds, FoundationSanityChecker, TrajectoryDriftChecker


class _ZeroForceEvaluator:
    name = "zero_force_evaluator"
    classification = "[test]"

    def evaluate(self, state: SimulationState, topology: SystemTopology, forcefield: BaseForceField) -> ForceEvaluation:
        del topology, forcefield
        return ForceEvaluation(
            forces=tuple((0.0, 0.0, 0.0) for _ in range(state.particle_count)),
            potential_energy=0.0,
        )


class _SimpleCorrectionModel:
    name = "simple_correction_model"
    classification = "[test]"

    def evaluate(
        self,
        state: SimulationState,
        topology: SystemTopology,
        region: RefinementRegion,
    ) -> QCloudCorrection:
        del state, topology
        anchor = region.particle_indices[0]
        return QCloudCorrection(
            region_id=region.region_id,
            method_label=self.name,
            energy_delta=0.1,
            force_deltas=(ParticleForceDelta(anchor, (0.2, 0.0, 0.0)),),
            confidence=0.8,
        )


class ValidationAndBenchmarkingTests(unittest.TestCase):
    """Verify Section 13 observer-side validation and benchmark helpers."""

    def _build_state_topology_forcefield(self) -> tuple[SimulationState, SystemTopology, BaseForceField]:
        state = SimulationState(
            units=UnitSystem.md_nano(),
            particles=ParticleState(
                positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
                masses=(1.0, 1.0, 1.0, 1.0),
                velocities=((0.0, 0.0, 0.0),) * 4,
            ),
            thermodynamics=ThermodynamicState(),
            provenance=StateProvenance(
                simulation_id=SimulationId("sim-validation"),
                state_id=StateId("state-validation"),
                parent_state_id=None,
                created_by="unit-test",
                stage="checkpoint",
            ),
            time=0.1,
            step=1,
            potential_energy=-0.2,
        )
        topology = SystemTopology(
            system_id="validation-system",
            bead_types=(
                BeadType(name="bb", role=BeadRole.STRUCTURAL),
                BeadType(name="site", role=BeadRole.FUNCTIONAL),
            ),
            beads=(
                Bead(bead_id=BeadId("b0"), particle_index=0, bead_type="bb", label="B0"),
                Bead(bead_id=BeadId("b1"), particle_index=1, bead_type="bb", label="B1"),
                Bead(bead_id=BeadId("b2"), particle_index=2, bead_type="site", label="S0"),
                Bead(bead_id=BeadId("b3"), particle_index=3, bead_type="site", label="S1"),
            ),
            bonds=(Bond(0, 1), Bond(2, 3)),
        )
        forcefield = BaseForceField(
            name="validation-forcefield",
            bond_parameters=(
                BondParameter("bb", "bb", equilibrium_distance=1.0, stiffness=50.0),
                BondParameter("site", "site", equilibrium_distance=1.0, stiffness=40.0),
            ),
            nonbonded_parameters=(
                NonbondedParameter("bb", "bb", sigma=1.0, epsilon=0.2, cutoff=3.1),
                NonbondedParameter("bb", "site", sigma=1.0, epsilon=0.3, cutoff=3.1),
                NonbondedParameter("site", "site", sigma=1.1, epsilon=0.4, cutoff=3.1),
            ),
        )
        return state, topology, forcefield

    def _build_trace_record(self, state: SimulationState) -> TraceRecord:
        return TraceRecord(
            record_id="trace-validation",
            simulation_id=state.provenance.simulation_id,
            state_id=state.provenance.state_id,
            parent_state_id=None,
            stage=state.provenance.stage,
            step=state.step,
            time=state.time,
            particle_count=state.particle_count,
            kinetic_energy=state.kinetic_energy(),
            potential_energy=state.potential_energy,
            active_edge_count=5,
            structural_edge_count=2,
            adaptive_edge_count=3,
            plasticity_trace_count=1,
            compartment_ids=("A", "B"),
            tags=("priority", "instability"),
        )

    def _build_replay_buffer(self, trace_record: TraceRecord) -> ReplayBuffer:
        replay_buffer = ReplayBuffer(capacity=4)
        replay_buffer.add_from_record(trace_record, score=2.0, tags=("priority",))
        return replay_buffer

    def _build_uncertainty(self, state: SimulationState) -> UncertaintyEstimate:
        return UncertaintyEstimate(
            state_id=state.provenance.state_id,
            energy_uncertainty=0.7,
            force_uncertainty=0.8,
            total_uncertainty=0.8,
            trigger_qcloud=True,
            metadata={"prediction_confidence": 0.1},
        )

    def _build_compartments(self) -> CompartmentRegistry:
        return CompartmentRegistry(
            particle_count=4,
            domains=(
                CompartmentDomain.from_members("A", "domain-a", (0, 1)),
                CompartmentDomain.from_members("B", "domain-b", (2, 3)),
            ),
        )

    def test_foundation_sanity_checker_accepts_aligned_outputs(self) -> None:
        state, topology, forcefield = self._build_state_topology_forcefield()
        force_evaluation = BaselineForceEvaluator().evaluate(state, topology, forcefield)
        graph = ConnectivityGraphManager().initialize(state, topology)
        trace_record = self._build_trace_record(state)
        replay_buffer = self._build_replay_buffer(trace_record)
        qcloud_result = QCloudForceCoupler().evaluate_with_selector(
            state=state,
            topology=topology,
            forcefield=forcefield,
            base_force_evaluator=BaselineForceEvaluator(),
            correction_model=_SimpleCorrectionModel(),
            region_selector=LocalRegionSelector(
                policy=RegionSelectionPolicy(max_regions=1, max_region_size=4, min_region_score=0.1)
            ),
            graph=graph,
            compartments=self._build_compartments(),
            trace_record=trace_record,
            focus_compartments=("A",),
        )
        residual_prediction = ResidualMemoryModel().predict(state, force_evaluation)
        controller_decision = ExecutiveController().decide(
            state,
            graph,
            trace_record=trace_record,
            uncertainty_estimate=self._build_uncertainty(state),
            replay_buffer=replay_buffer,
            qcloud_result=qcloud_result,
        )

        report = FoundationSanityChecker().run(
            state,
            topology=topology,
            force_evaluation=qcloud_result.force_evaluation,
            graph=graph,
            qcloud_result=qcloud_result,
            residual_prediction=residual_prediction,
            controller_decision=controller_decision,
        )

        self.assertTrue(report.passed())
        self.assertEqual(report.failed_checks(), ())
        self.assertIn("controller_alignment", {check.name for check in report.checks})
        self.assertIn("qcloud_alignment", {check.name for check in report.checks})

    def test_foundation_sanity_checker_flags_force_energy_mismatch(self) -> None:
        state, _, _ = self._build_state_topology_forcefield()
        report = FoundationSanityChecker().run(
            state,
            force_evaluation=ForceEvaluation(
                forces=((0.0, 0.0, 0.0),) * state.particle_count,
                potential_energy=0.0,
                component_energies={"bonded": 1.0},
            ),
        )

        self.assertFalse(report.passed())
        self.assertIn("force_energy_accounting", {check.name for check in report.failed_checks()})

    def test_trajectory_drift_checker_accepts_monotonic_low_drift_sequence(self) -> None:
        registry = SimulationStateRegistry(created_by="unit-test")
        initial_state = registry.create_initial_state(
            particles=ParticleState(
                positions=((0.0, 0.0, 0.0),),
                masses=(1.0,),
                velocities=((1.0, 0.0, 0.0),),
            ),
            thermodynamics=ThermodynamicState(),
        )
        topology = SystemTopology(
            system_id="drift-system",
            bead_types=(BeadType(name="bb"),),
            beads=(Bead(bead_id=BeadId("b0"), particle_index=0, bead_type="bb", label="B0"),),
        )
        forcefield = BaseForceField(
            name="drift-forcefield",
            nonbonded_parameters=(NonbondedParameter("bb", "bb", sigma=1.0, epsilon=0.0, cutoff=1.0),),
        )
        SimulationLoop(
            topology=topology,
            forcefield=forcefield,
            integrator=LangevinIntegrator(time_step=0.1),
            force_evaluator=_ZeroForceEvaluator(),
            registry=registry,
        ).run(3)

        states = tuple(registry.get_state(state_id) for state_id in registry.state_ids())
        report = TrajectoryDriftChecker(
            thresholds=DriftThresholds(max_energy_drift=0.1, max_position_displacement=0.31)
        ).assess(states)

        self.assertTrue(report.passed())
        self.assertTrue(report.time_monotonic)
        self.assertTrue(report.step_sequence_ok)
        self.assertAlmostEqual(report.max_energy_drift, 0.0)
        self.assertEqual(initial_state.provenance.simulation_id, report.simulation_id)

    def test_trajectory_drift_checker_flags_energy_and_step_drift(self) -> None:
        state = SimulationState(
            units=UnitSystem.md_nano(),
            particles=ParticleState(
                positions=((0.0, 0.0, 0.0),),
                masses=(1.0,),
            ),
            thermodynamics=ThermodynamicState(),
            provenance=StateProvenance(
                simulation_id=SimulationId("sim-drift"),
                state_id=StateId("state-drift-0"),
                parent_state_id=None,
                created_by="unit-test",
                stage="initialization",
            ),
            time=0.0,
            step=0,
            potential_energy=0.0,
        )
        drifted_state = SimulationState(
            units=state.units,
            particles=state.particles.with_positions(((5.0, 0.0, 0.0),)),
            thermodynamics=state.thermodynamics,
            provenance=StateProvenance(
                simulation_id=state.provenance.simulation_id,
                state_id=StateId("state-drift-2"),
                parent_state_id=state.provenance.state_id,
                created_by="unit-test",
                stage="integration",
            ),
            time=0.05,
            step=2,
            potential_energy=4.0,
        )

        report = TrajectoryDriftChecker(
            thresholds=DriftThresholds(max_energy_drift=0.5, max_position_displacement=1.0)
        ).assess((state, drifted_state))

        self.assertFalse(report.passed())
        self.assertFalse(report.step_sequence_ok)
        self.assertIn("energy drift exceeded the configured threshold.", report.violations)
        self.assertIn("particle displacement exceeded the configured threshold.", report.violations)

    def test_benchmark_suite_reports_foundation_cases(self) -> None:
        state, topology, forcefield = self._build_state_topology_forcefield()
        graph_manager = ConnectivityGraphManager()
        graph = graph_manager.initialize(state, topology)
        trace_record = self._build_trace_record(state)
        report = BaselineBenchmarkSuite(default_repeats=2).run_foundation_suite(
            state=state,
            topology=topology,
            forcefield=forcefield,
            force_evaluator=BaselineForceEvaluator(),
            graph_manager=graph_manager,
            controller=ExecutiveController(),
            previous_graph=graph,
            qcloud_coupler=QCloudForceCoupler(),
            qcloud_correction_model=_SimpleCorrectionModel(),
            qcloud_region_selector=LocalRegionSelector(
                policy=RegionSelectionPolicy(max_regions=1, max_region_size=4, min_region_score=0.1)
            ),
            compartments=self._build_compartments(),
            trace_record=trace_record,
            focus_compartments=("A",),
            uncertainty_estimate=self._build_uncertainty(state),
            replay_buffer=self._build_replay_buffer(trace_record),
            residual_model=ResidualMemoryModel(),
            repeats=2,
        )

        self.assertEqual(
            report.case_names(),
            (
                "force_evaluation",
                "graph_update",
                "qcloud_coupling",
                "residual_prediction",
                "controller_decision",
            ),
        )
        self.assertEqual(report.metadata["case_count"], 5)
        self.assertGreaterEqual(report.case_for("controller_decision").elapsed_seconds, 0.0)
        self.assertGreaterEqual(report.total_elapsed_seconds(), 0.0)


if __name__ == "__main__":
    unittest.main()
