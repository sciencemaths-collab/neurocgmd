"""Tests for Section 12 executive control monitoring, allocation, and decisions."""

from __future__ import annotations

import unittest

from ai_control import (
    ChemistryControlGuidance,
    ControllerActionKind,
    DeterministicExecutivePolicy,
    ExecutiveController,
    ExecutionBudget,
    ResourceAllocator,
    StabilityLevel,
    StabilityMonitor,
)
from chemistry.interface_logic import ChemistryInterfaceReport
from core.state import ParticleState, SimulationState, StateProvenance, ThermodynamicState, UnitSystem
from core.types import SimulationId, StateId
from graph import ConnectivityGraph, DynamicEdgeKind, DynamicEdgeState
from memory import EpisodeKind, EpisodeRegistry, ReplayBuffer, TraceRecord
from ml import LiveFeatureVector, UncertaintyEstimate
from physics.forces.composite import ForceEvaluation
from qcloud import ParticleForceDelta, QCloudCorrection, QCloudCouplingResult, RefinementRegion, RegionTriggerKind


class AIControlTests(unittest.TestCase):
    """Verify Section 12 control logic remains explicit and well-bounded."""

    def _build_state(self) -> SimulationState:
        return SimulationState(
            units=UnitSystem.md_nano(),
            particles=ParticleState(
                positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
                masses=(1.0, 1.0, 1.0, 1.0),
            ),
            thermodynamics=ThermodynamicState(),
            provenance=StateProvenance(
                simulation_id=SimulationId("sim-control"),
                state_id=StateId("state-control"),
                parent_state_id=None,
                created_by="unit-test",
                stage="checkpoint",
            ),
            time=0.2,
            step=2,
            potential_energy=-0.4,
        )

    def _build_graph(self) -> ConnectivityGraph:
        return ConnectivityGraph(
            particle_count=4,
            step=2,
            edges=(
                DynamicEdgeState(0, 1, DynamicEdgeKind.STRUCTURAL_LOCAL, 1.0, 1.0, 0, 2),
                DynamicEdgeState(1, 2, DynamicEdgeKind.ADAPTIVE_LONG_RANGE, 0.8, 1.4, 1, 2),
                DynamicEdgeState(2, 3, DynamicEdgeKind.ADAPTIVE_LOCAL, 0.7, 0.9, 1, 2),
            ),
        )

    def _build_trace_record(self, state: SimulationState) -> TraceRecord:
        return TraceRecord(
            record_id="trace-control",
            simulation_id=state.provenance.simulation_id,
            state_id=state.provenance.state_id,
            parent_state_id=None,
            stage=state.provenance.stage,
            step=state.step,
            time=state.time,
            particle_count=state.particle_count,
            kinetic_energy=state.kinetic_energy(),
            potential_energy=state.potential_energy,
            active_edge_count=3,
            structural_edge_count=1,
            adaptive_edge_count=2,
            plasticity_trace_count=1,
            compartment_ids=("A", "B"),
            tags=("priority", "instability"),
        )

    def _build_uncertainty(self, state: SimulationState) -> UncertaintyEstimate:
        return UncertaintyEstimate(
            state_id=state.provenance.state_id,
            energy_uncertainty=0.8,
            force_uncertainty=0.85,
            total_uncertainty=0.85,
            trigger_qcloud=True,
            metadata={"prediction_confidence": 0.1},
        )

    def _build_replay_buffer(self, record: TraceRecord) -> ReplayBuffer:
        replay_buffer = ReplayBuffer(capacity=4)
        replay_buffer.add_from_record(record, score=2.0, tags=("priority",))
        replay_buffer.add_from_record(
            TraceRecord(
                record_id="trace-control-2",
                simulation_id=record.simulation_id,
                state_id=StateId("state-control-2"),
                parent_state_id=record.state_id,
                stage="integration",
                step=3,
                time=0.3,
                particle_count=record.particle_count,
                kinetic_energy=record.kinetic_energy,
                potential_energy=record.potential_energy,
                tags=("train",),
            ),
            score=1.0,
            tags=("train",),
        )
        return replay_buffer

    def _build_qcloud_result(self, state: SimulationState) -> QCloudCouplingResult:
        region = RefinementRegion(
            region_id="region-control",
            state_id=state.provenance.state_id,
            particle_indices=(1, 2),
            seed_pairs=((1, 2),),
            compartment_ids=("A",),
            trigger_kinds=(RegionTriggerKind.ADAPTIVE_EDGE,),
            score=1.0,
        )
        correction = QCloudCorrection(
            region_id=region.region_id,
            method_label="test-qcloud",
            energy_delta=0.2,
            force_deltas=(ParticleForceDelta(1, (0.5, 0.0, 0.0)),),
            confidence=0.9,
        )
        return QCloudCouplingResult(
            force_evaluation=ForceEvaluation(
                forces=((0.0, 0.0, 0.0),) * state.particle_count,
                potential_energy=0.2,
                component_energies={"qcloud": 0.2},
            ),
            selected_regions=(region,),
            applied_corrections=(correction,),
            metadata={"applied_region_count": 1},
        )

    def test_stability_monitor_escalates_uncertain_priority_state(self) -> None:
        state = self._build_state()
        graph = self._build_graph()
        trace_record = self._build_trace_record(state)
        uncertainty = self._build_uncertainty(state)
        episode_registry = EpisodeRegistry()
        episode_registry.open_episode(state, kind=EpisodeKind.INSTABILITY, tags=("instability",))

        assessment = StabilityMonitor().assess(
            state,
            graph,
            trace_record=trace_record,
            uncertainty_estimate=uncertainty,
            episode_registry=episode_registry,
        )

        self.assertEqual(assessment.level, StabilityLevel.CRITICAL)
        self.assertTrue(assessment.trigger_qcloud)
        self.assertEqual(assessment.recommended_focus_compartments, ("A", "B"))
        self.assertGreaterEqual(assessment.normalized_risk, 0.8)
        self.assertEqual(
            {signal.source for signal in assessment.signals},
            {"ml_uncertainty", "graph_adaptivity", "memory_priority", "episode_registry"},
        )

    def test_resource_allocator_accounts_for_existing_qcloud_work(self) -> None:
        state = self._build_state()
        allocation = ResourceAllocator(
            budget=ExecutionBudget(max_qcloud_regions=2, max_ml_examples=4, max_focus_compartments=2)
        ).allocate(
            assessment=StabilityMonitor().assess(
                state,
                self._build_graph(),
                trace_record=self._build_trace_record(state),
                uncertainty_estimate=self._build_uncertainty(state),
            ),
            replay_buffer=self._build_replay_buffer(self._build_trace_record(state)),
            qcloud_result=self._build_qcloud_result(state),
        )

        self.assertEqual(allocation.level, StabilityLevel.CRITICAL)
        self.assertEqual(allocation.qcloud_region_budget, 1)
        self.assertEqual(allocation.ml_example_budget, 2)
        self.assertEqual(allocation.monitoring_intensity.value, "critical")
        self.assertEqual(allocation.focus_compartments, ("A", "B"))

    def test_resource_allocator_merges_chemistry_guidance(self) -> None:
        assessment = StabilityMonitor().assess(self._build_state(), self._build_graph())
        guidance = ChemistryControlGuidance(
            state_id=assessment.state_id,
            chemistry_risk=0.7,
            recommend_qcloud_boost=True,
            recommend_ml_boost=True,
            review_required=True,
            focus_compartments=("B", "chem"),
            summary="chemistry mismatch",
        )
        replay_buffer = self._build_replay_buffer(self._build_trace_record(self._build_state()))

        allocation = ResourceAllocator(
            budget=ExecutionBudget(max_qcloud_regions=2, max_ml_examples=4, max_focus_compartments=3)
        ).allocate(
            assessment=assessment,
            replay_buffer=replay_buffer,
            chemistry_guidance=guidance,
        )

        self.assertEqual(allocation.qcloud_region_budget, 1)
        self.assertEqual(allocation.ml_example_budget, 2)
        self.assertEqual(allocation.focus_compartments, ("B", "chem"))

    def test_policy_holds_steady_when_no_escalation_or_budget_is_needed(self) -> None:
        state = self._build_state()
        assessment = StabilityMonitor().assess(state, self._build_graph())
        allocation = ResourceAllocator().allocate(assessment)

        actions = DeterministicExecutivePolicy().build_actions(
            state.provenance.state_id,
            assessment,
            allocation,
        )

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].kind, ControllerActionKind.HOLD_STEADY)

    def test_controller_skips_duplicate_instability_episode_action(self) -> None:
        state = self._build_state()
        trace_record = self._build_trace_record(state)
        episode_registry = EpisodeRegistry()
        episode_registry.open_episode(state, kind=EpisodeKind.INSTABILITY, tags=("instability",))

        decision = ExecutiveController().decide(
            state,
            self._build_graph(),
            trace_record=trace_record,
            uncertainty_estimate=self._build_uncertainty(state),
            episode_registry=episode_registry,
            replay_buffer=self._build_replay_buffer(trace_record),
        )

        self.assertNotIn(ControllerActionKind.OPEN_INSTABILITY_EPISODE, {action.kind for action in decision.actions})

    def test_executive_controller_prioritizes_qcloud_then_monitoring_then_ml(self) -> None:
        state = self._build_state()
        trace_record = self._build_trace_record(state)
        decision = ExecutiveController().decide(
            state,
            self._build_graph(),
            trace_record=trace_record,
            uncertainty_estimate=self._build_uncertainty(state),
            replay_buffer=self._build_replay_buffer(trace_record),
        )

        self.assertEqual(decision.highest_priority_action().kind, ControllerActionKind.REQUEST_QCLOUD_REFINEMENT)
        self.assertEqual(
            tuple(action.kind for action in decision.actions[:4]),
            (
                ControllerActionKind.REQUEST_QCLOUD_REFINEMENT,
                ControllerActionKind.ESCALATE_MONITORING,
                ControllerActionKind.OPEN_INSTABILITY_EPISODE,
                ControllerActionKind.REQUEST_ML_UPDATE,
            ),
        )
        self.assertIn(ControllerActionKind.REVIEW_COMPARTMENT_FOCUS, {action.kind for action in decision.actions})
        self.assertEqual(decision.metadata["highest_priority_action"], "request_qcloud_refinement")

    def test_controller_emits_chemistry_review_when_interface_is_implausible(self) -> None:
        state = self._build_state()
        trace_record = self._build_trace_record(state)
        chemistry_report = ChemistryInterfaceReport(
            title="Interface Chemistry",
            compartment_ids=("A", "B"),
            evaluated_pair_count=4,
            favorable_pair_fraction=0.10,
            mean_pair_score=0.20,
            charge_complementarity=0.25,
            hydropathy_alignment=0.40,
            flexibility_pressure=0.80,
            hotspot_pair_fraction=0.15,
        )
        live_features = LiveFeatureVector(
            state_id=state.provenance.state_id,
            values={
                "structure_rmsd_normalized": 0.75,
                "shadow_force_regression": 1.0,
            },
        )

        decision = ExecutiveController().decide(
            state,
            self._build_graph(),
            trace_record=trace_record,
            replay_buffer=self._build_replay_buffer(trace_record),
            chemistry_report=chemistry_report,
            live_features=live_features,
        )

        self.assertIn(ControllerActionKind.REVIEW_CHEMISTRY_ALIGNMENT, {action.kind for action in decision.actions})
        self.assertEqual(decision.allocation.focus_compartments, ("A", "B"))
        self.assertGreater(decision.metadata["chemistry_risk"], 0.5)


if __name__ == "__main__":
    unittest.main()
